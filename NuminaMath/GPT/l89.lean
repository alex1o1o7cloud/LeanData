import Mathlib

namespace cost_of_first_15_kgs_l89_89795

def cost_33_kg := 333
def cost_36_kg := 366
def kilo_33 := 33
def kilo_36 := 36
def first_limit := 30
def extra_3kg := 3  -- 33 - 30
def extra_6kg := 6  -- 36 - 30

theorem cost_of_first_15_kgs (l q : ℕ) 
  (h1 : first_limit * l + extra_3kg * q = cost_33_kg)
  (h2 : first_limit * l + extra_6kg * q = cost_36_kg) :
  15 * l = 150 :=
by
  sorry

end cost_of_first_15_kgs_l89_89795


namespace eval_sqrt4_8_pow12_l89_89298

-- Define the fourth root of 8
def fourthRootOfEight : ℝ := 8 ^ (1 / 4)

-- Define the original expression
def expr := (fourthRootOfEight) ^ 12

-- The theorem to prove
theorem eval_sqrt4_8_pow12: expr = 512 := by
  sorry

end eval_sqrt4_8_pow12_l89_89298


namespace magnitude_of_z_8_l89_89655

def z : Complex := 2 + 3 * Complex.I

theorem magnitude_of_z_8 : Complex.abs (z ^ 8) = 28561 := by
  sorry

end magnitude_of_z_8_l89_89655


namespace number_of_people_tasting_apple_pies_l89_89507

/-- Sedrach's apple pie problem -/
def apple_pies : ℕ := 13
def halves_per_apple_pie : ℕ := 2
def bite_size_samples_per_half : ℕ := 5

theorem number_of_people_tasting_apple_pies :
    (apple_pies * halves_per_apple_pie * bite_size_samples_per_half) = 130 :=
by
  sorry

end number_of_people_tasting_apple_pies_l89_89507


namespace tangent_circles_locus_l89_89060

noncomputable def locus_condition (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

theorem tangent_circles_locus (a b : ℝ) (r : ℝ) (h : locus_condition a b r) :
  a^2 + 7 * b^2 - 34 * a - 57 = 0 :=
sorry

end tangent_circles_locus_l89_89060


namespace total_students_l89_89563

theorem total_students (rank_right rank_left : ℕ) (h_right : rank_right = 18) (h_left : rank_left = 12) : rank_right + rank_left - 1 = 29 := 
by
  sorry

end total_students_l89_89563


namespace nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l89_89788

theorem nat_no_solution_x3_plus_5y_eq_y3_plus_5x (x y : ℕ) (h₁ : x ≠ y) : 
  x^3 + 5 * y ≠ y^3 + 5 * x :=
sorry

theorem positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + 5 * y = y^3 + 5 * x :=
sorry

end nat_no_solution_x3_plus_5y_eq_y3_plus_5x_positive_real_solution_exists_x3_plus_5y_eq_y3_plus_5x_l89_89788


namespace geom_seq_expression_l89_89638

theorem geom_seq_expression (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 + a 3 = 10) (h2 : a 2 + a 4 = 5) :
  ∀ n, a n = 2 ^ (4 - n) :=
by
  -- sorry is used to skip the proof
  sorry

end geom_seq_expression_l89_89638


namespace walkway_area_correct_l89_89521

/-- Define the dimensions of a single flower bed. --/
def flower_bed_length : ℝ := 8
def flower_bed_width : ℝ := 3

/-- Define the number of flower beds in rows and columns. --/
def rows : ℕ := 4
def cols : ℕ := 3

/-- Define the width of the walkways surrounding the flower beds. --/
def walkway_width : ℝ := 2

/-- Calculate the total dimensions of the garden including walkways. --/
def total_garden_width : ℝ := (cols * flower_bed_length) + ((cols + 1) * walkway_width)
def total_garden_height : ℝ := (rows * flower_bed_width) + ((rows + 1) * walkway_width)

/-- Calculate the total area of the garden including walkways. --/
def total_garden_area : ℝ := total_garden_width * total_garden_height

/-- Calculate the total area of the flower beds. --/
def flower_bed_area : ℝ := flower_bed_length * flower_bed_width
def total_flower_beds_area : ℝ := rows * cols * flower_bed_area

/-- Calculate the total area of the walkways. --/
def walkway_area := total_garden_area - total_flower_beds_area

theorem walkway_area_correct : walkway_area = 416 := 
by
  -- Proof omitted
  sorry

end walkway_area_correct_l89_89521


namespace max_sin_A_plus_sin_C_l89_89228

variables {a b c S : ℝ}
variables {A B C : ℝ}

-- Assume the sides of the triangle
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- Assume the angles of the triangle
variables (hA : A > 0) (hB : B > (Real.pi / 2)) (hC : C > 0)
variables (hSumAngles : A + B + C = Real.pi)

-- Assume the relationship between the area and the sides
variables (hArea : S = (1/2) * a * c * Real.sin B)

-- Assume the given equation holds
variables (hEquation : 4 * b * S = a * (b^2 + c^2 - a^2))

-- The statement to prove
theorem max_sin_A_plus_sin_C : (Real.sin A + Real.sin C) ≤ 9 / 8 :=
sorry

end max_sin_A_plus_sin_C_l89_89228


namespace value_of_r_squared_plus_s_squared_l89_89144

theorem value_of_r_squared_plus_s_squared (r s : ℝ) (h1 : r * s = 24) (h2 : r + s = 10) :
  r^2 + s^2 = 52 :=
sorry

end value_of_r_squared_plus_s_squared_l89_89144


namespace deadlift_weight_loss_is_200_l89_89852

def initial_squat : ℕ := 700
def initial_bench : ℕ := 400
def initial_deadlift : ℕ := 800
def lost_squat_percent : ℕ := 30
def new_total : ℕ := 1490

theorem deadlift_weight_loss_is_200 : initial_deadlift - (new_total - ((initial_squat * (100 - lost_squat_percent)) / 100 + initial_bench)) = 200 :=
by
  sorry

end deadlift_weight_loss_is_200_l89_89852


namespace prime_power_seven_l89_89482

theorem prime_power_seven (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (eqn : p + 25 = q^7) : p = 103 := by
  sorry

end prime_power_seven_l89_89482


namespace incorrect_statement_C_l89_89479

/-- 
  Prove that the function y = -1/2 * x + 3 does not intersect the y-axis at (6,0).
-/
theorem incorrect_statement_C 
: ∀ (x y : ℝ), y = -1/2 * x + 3 → (x, y) ≠ (6, 0) :=
by
  intros x y h
  sorry

end incorrect_statement_C_l89_89479


namespace quiz_probability_l89_89398

theorem quiz_probability :
  let probMCQ := 1/3
  let probTF1 := 1/2
  let probTF2 := 1/2
  probMCQ * probTF1 * probTF2 = 1/12 := by
  sorry

end quiz_probability_l89_89398


namespace total_surface_area_correct_l89_89650

noncomputable def total_surface_area_of_cylinder (radius height : ℝ) : ℝ :=
  let lateral_surface_area := 2 * Real.pi * radius * height
  let top_and_bottom_area := 2 * Real.pi * radius^2
  lateral_surface_area + top_and_bottom_area

theorem total_surface_area_correct : total_surface_area_of_cylinder 3 10 = 78 * Real.pi :=
by
  sorry

end total_surface_area_correct_l89_89650


namespace sugar_in_lollipop_l89_89541

-- Definitions based on problem conditions
def chocolate_bars := 14
def sugar_per_bar := 10
def total_sugar := 177

-- The theorem to prove
theorem sugar_in_lollipop : total_sugar - (chocolate_bars * sugar_per_bar) = 37 :=
by
  -- we are not providing the proof, hence using sorry
  sorry

end sugar_in_lollipop_l89_89541


namespace find_abc_values_l89_89313

-- Define the problem conditions as lean definitions
def represents_circle (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x - b * y + c = 0

def circle_center_and_radius_condition (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 3^2

-- Lean 4 statement for the proof problem
theorem find_abc_values (a b c : ℝ) :
  (∀ x y : ℝ, represents_circle a b c x y ↔ circle_center_and_radius_condition x y) →
  a = -2 ∧ b = 6 ∧ c = 4 :=
by
  intro h
  sorry

end find_abc_values_l89_89313


namespace intersection_complement_M_and_N_l89_89802
open Set

def U := @univ ℝ
def M := {x : ℝ | x^2 + 2*x - 8 ≤ 0}
def N := {x : ℝ | -1 < x ∧ x < 3}
def complement_M := {x : ℝ | ¬ (x ∈ M)}

theorem intersection_complement_M_and_N :
  (complement_M ∩ N) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_complement_M_and_N_l89_89802


namespace brian_traveled_correct_distance_l89_89269

def miles_per_gallon : Nat := 20
def gallons_used : Nat := 3
def expected_miles : Nat := 60

theorem brian_traveled_correct_distance : (miles_per_gallon * gallons_used) = expected_miles := by
  sorry

end brian_traveled_correct_distance_l89_89269


namespace shortest_distance_between_circles_is_zero_l89_89889

open Real

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop :=
  x^2 - 12 * x + y^2 - 8 * y - 12 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop :=
  x^2 + 10 * x + y^2 - 10 * y + 34 = 0

-- Statement of the proof problem: 
-- Prove the shortest distance between the two circles defined by circle1 and circle2 is 0.
theorem shortest_distance_between_circles_is_zero :
    ∀ (x1 y1 x2 y2 : ℝ),
      circle1 x1 y1 →
      circle2 x2 y2 →
      0 = 0 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end shortest_distance_between_circles_is_zero_l89_89889


namespace parabola_and_line_solutions_l89_89590

-- Definition of the parabola with its focus
def parabola_with_focus (p : ℝ) : Prop :=
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (x : ℝ), x = 1 / 2)

-- Definitions of conditions for intersection and orthogonal vectors
def line_intersecting_parabola (slope t : ℝ) (p : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), 
  (y1 = 2 * x1 + t) ∧ (y2 = 2 * x2 + t) ∧
  (y1^2 = 2 * x1) ∧ (y2^2 = 2 * x2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧
  (x1 * x2 = (t^2) / 4) ∧ (x1 * x2 + y1 * y2 = 0)

-- Lean statement for the proof problem
theorem parabola_and_line_solutions :
  ∀ p t : ℝ, 
  parabola_with_focus p → 
  (line_intersecting_parabola 2 t p → t = -4)
  → p = 1 :=
by
  intros p t h_parabola h_line
  sorry

end parabola_and_line_solutions_l89_89590


namespace minimum_value_of_x_plus_y_l89_89778

noncomputable def minValueSatisfies (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y + x * y = 2 → x + y ≥ 2 * Real.sqrt 3 - 2

theorem minimum_value_of_x_plus_y (x y : ℝ) : minValueSatisfies x y :=
by sorry

end minimum_value_of_x_plus_y_l89_89778


namespace profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l89_89856

theorem profitability_when_x_gt_94 (A : ℕ) (x : ℕ) (hx : x > 94) : 
  1/3 * x * A - (2/3 * x * (A / 2)) = 0 := 
sorry

theorem daily_profit_when_x_le_94 (A : ℕ) (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 94) : 
  ∃ T : ℕ, T = (x - 3 * x / (2 * (96 - x))) * A := 
sorry

theorem max_profit_occurs_at_84 (A : ℕ) : 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 94 ∧ 
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 94 → 
    (y - 3 * y / (2 * (96 - y))) * A ≤ (84 - 3 * 84 / (2 * (96 - 84))) * A) := 
sorry

end profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l89_89856


namespace number_of_integers_satisfying_inequalities_l89_89716

theorem number_of_integers_satisfying_inequalities :
  ∃ (count : ℕ), count = 3 ∧
    (∀ x : ℤ, -4 * x ≥ x + 10 → -3 * x ≤ 15 → -5 * x ≥ 3 * x + 24 → 2 * x ≤ 18 →
      x = -5 ∨ x = -4 ∨ x = -3) :=
sorry

end number_of_integers_satisfying_inequalities_l89_89716


namespace expected_number_of_different_faces_l89_89633

theorem expected_number_of_different_faces :
  let p := (6 : ℕ) ^ 6
  let q := (5 : ℕ) ^ 6
  6 * (1 - (5 / 6)^6) = (p - q) / (6 ^ 5) :=
by
  sorry

end expected_number_of_different_faces_l89_89633


namespace cuboid_layers_l89_89078

theorem cuboid_layers (V : ℕ) (n_blocks : ℕ) (volume_per_block : ℕ) (blocks_per_layer : ℕ)
  (hV : V = 252) (hvol : volume_per_block = 1) (hblocks : n_blocks = V / volume_per_block) (hlayer : blocks_per_layer = 36) :
  (n_blocks / blocks_per_layer) = 7 :=
by
  sorry

end cuboid_layers_l89_89078


namespace xy_eq_zero_l89_89260

theorem xy_eq_zero (x y : ℝ) (h1 : x - y = 3) (h2 : x^3 - y^3 = 27) : x * y = 0 := by
  sorry

end xy_eq_zero_l89_89260


namespace factor_polynomial_l89_89602

theorem factor_polynomial :
  (x^2 + 5 * x + 4) * (x^2 + 11 * x + 30) + (x^2 + 8 * x - 10) =
  (x^2 + 8 * x + 7) * (x^2 + 8 * x + 19) := by
  sorry

end factor_polynomial_l89_89602


namespace num_mittens_per_box_eq_six_l89_89715

theorem num_mittens_per_box_eq_six 
    (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ)
    (h1 : num_boxes = 4) (h2 : scarves_per_box = 2) (h3 : total_clothing = 32) :
    (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 :=
by
  sorry

end num_mittens_per_box_eq_six_l89_89715


namespace rhombus_height_l89_89194

theorem rhombus_height (a d1 d2 : ℝ) (h : ℝ)
  (h_a_positive : 0 < a)
  (h_d1_positive : 0 < d1)
  (h_d2_positive : 0 < d2)
  (h_side_geometric_mean : a^2 = d1 * d2) :
  h = a / 2 :=
sorry

end rhombus_height_l89_89194


namespace max_cos_a_l89_89764

theorem max_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = Real.cos b) 
  (h2 : Real.sin b = Real.cos c) 
  (h3 : Real.sin c = Real.cos a) : 
  Real.cos a = Real.sqrt 2 / 2 := by
sorry

end max_cos_a_l89_89764


namespace no_integer_solutions_l89_89074

theorem no_integer_solutions (x y : ℤ) : ¬ (3 * x^2 + 2 = y^2) :=
sorry

end no_integer_solutions_l89_89074


namespace find_missing_number_l89_89285

theorem find_missing_number (x y : ℝ) 
  (h1 : (x + 50 + 78 + 104 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 76.4) : 
  y = 28 :=
by
  sorry

end find_missing_number_l89_89285


namespace compute_expression_l89_89540

-- The definition and conditions
def is_nonreal_root_of_unity (ω : ℂ) : Prop := ω ^ 3 = 1 ∧ ω ≠ 1

-- The statement
theorem compute_expression (ω : ℂ) (hω : is_nonreal_root_of_unity ω) : 
  (1 - 2 * ω + 2 * ω ^ 2) ^ 6 + (1 + 2 * ω - 2 * ω ^ 2) ^ 6 = 0 :=
sorry

end compute_expression_l89_89540


namespace tom_paid_correct_amount_l89_89902

def quantity_of_apples : ℕ := 8
def rate_per_kg_apples : ℕ := 70
def quantity_of_mangoes : ℕ := 9
def rate_per_kg_mangoes : ℕ := 45

def cost_of_apples : ℕ := quantity_of_apples * rate_per_kg_apples
def cost_of_mangoes : ℕ := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid : ℕ := cost_of_apples + cost_of_mangoes

theorem tom_paid_correct_amount :
  total_amount_paid = 965 :=
sorry

end tom_paid_correct_amount_l89_89902


namespace arithmetic_geometric_sequences_l89_89519

theorem arithmetic_geometric_sequences :
  ∀ (a₁ a₂ b₁ b₂ b₃ : ℤ), 
  (a₂ = a₁ + (a₁ - (-1))) ∧ 
  (-4 = -1 + 3 * (a₂ - a₁)) ∧ 
  (-4 = -1 * (b₃/b₁)^4) ∧ 
  (b₂ = b₁ * (b₂/b₁)^2) →
  (a₂ - a₁) / b₂ = 1 / 2 := 
by
  intros a₁ a₂ b₁ b₂ b₃ h
  sorry

end arithmetic_geometric_sequences_l89_89519


namespace ages_of_Xs_sons_l89_89858

def ages_problem (x y : ℕ) : Prop :=
x ≠ y ∧ x ≤ 10 ∧ y ≤ 10 ∧
∀ u v : ℕ, u * v = x * y → u ≤ 10 ∧ v ≤ 10 → (u, v) = (x, y) ∨ (u, v) = (y, x) ∨
(∀ z w : ℕ, z / w = x / y → z = x ∧ w = y ∨ z = y ∧ w = x → u ≠ z ∧ v ≠ w) →
(∀ a b : ℕ, a - b = (x - y) ∨ b - a = (y - x) → (x, y) = (a, b) ∨ (x, y) = (b, a))

theorem ages_of_Xs_sons : ages_problem 8 2 := 
by {
  sorry
}


end ages_of_Xs_sons_l89_89858


namespace find_t_l89_89262

variables (s t : ℚ)

theorem find_t (h1 : 12 * s + 7 * t = 154) (h2 : s = 2 * t - 3) : t = 190 / 31 :=
by
  sorry

end find_t_l89_89262


namespace problem_f8_minus_f4_l89_89559

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 2

theorem problem_f8_minus_f4 : f 8 - f 4 = -1 :=
by sorry

end problem_f8_minus_f4_l89_89559


namespace fraction_identity_l89_89569

theorem fraction_identity (a b : ℝ) (h : a / b = 5 / 2) : (a + 2 * b) / (a - b) = 3 :=
by sorry

end fraction_identity_l89_89569


namespace box_surface_area_l89_89909

theorem box_surface_area (w l s tab : ℕ):
  w = 40 → l = 60 → s = 8 → tab = 2 →
  (40 * 60 - 4 * 8 * 8 + 2 * (2 * (60 - 2 * 8) + 2 * (40 - 2 * 8))) = 2416 :=
by
  intros _ _ _ _
  sorry

end box_surface_area_l89_89909


namespace answer_l89_89589

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, Real.exp x > 1

theorem answer (hp : p) (hq : ¬ q) : p ∧ ¬ q :=
  by
    exact ⟨hp, hq⟩

end answer_l89_89589


namespace inverse_B2_l89_89461

def matrix_B_inv : Matrix (Fin 2) (Fin 2) ℝ := !![3, 7; -2, -4]

def matrix_B2_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-5, -7; 2, 2]

theorem inverse_B2 (B : Matrix (Fin 2) (Fin 2) ℝ) (hB_inv : B⁻¹ = matrix_B_inv) :
  (B^2)⁻¹ = matrix_B2_inv :=
sorry

end inverse_B2_l89_89461


namespace replace_with_30_digit_nat_number_l89_89800

noncomputable def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000

theorem replace_with_30_digit_nat_number (a : Fin 10 → ℕ) (h : ∀ i, is_three_digit (a i)) :
  ∃ b : ℕ, (b < 10^30 ∧ ∃ x : ℤ, (a 9) * x^9 + (a 8) * x^8 + (a 7) * x^7 + (a 6) * x^6 + (a 5) * x^5 + 
           (a 4) * x^4 + (a 3) * x^3 + (a 2) * x^2 + (a 1) * x + (a 0) = b) :=
by
  sorry

end replace_with_30_digit_nat_number_l89_89800


namespace students_no_A_l89_89943

theorem students_no_A
  (total_students : ℕ)
  (A_in_English : ℕ)
  (A_in_math : ℕ)
  (A_in_both : ℕ)
  (total_students_eq : total_students = 40)
  (A_in_English_eq : A_in_English = 10)
  (A_in_math_eq : A_in_math = 18)
  (A_in_both_eq : A_in_both = 6) :
  total_students - ((A_in_English + A_in_math) - A_in_both) = 18 :=
by
  sorry

end students_no_A_l89_89943


namespace count_right_triangles_l89_89476

theorem count_right_triangles: 
  ∃ n : ℕ, n = 9 ∧ ∃ (a b : ℕ), a^2 + b^2 = (b+2)^2 ∧ b < 100 ∧ a > 0 ∧ b > 0 := by
  sorry

end count_right_triangles_l89_89476


namespace quadratic_inequality_solution_l89_89341

theorem quadratic_inequality_solution (a : ℝ) (h1 : ∀ x : ℝ, ax^2 + (a + 1) * x + 1 ≥ 0) : a = 1 := by
  sorry

end quadratic_inequality_solution_l89_89341


namespace evaluate_expression_l89_89010

-- Define x as given in the condition
def x : ℤ := 5

-- State the theorem we need to prove
theorem evaluate_expression : x^3 - 3 * x = 110 :=
by
  -- Proof will be provided here
  sorry

end evaluate_expression_l89_89010


namespace color_of_last_bead_l89_89556

-- Define the sequence and length of repeated pattern
def bead_pattern : List String := ["red", "red", "orange", "yellow", "yellow", "yellow", "green", "green", "blue"]
def pattern_length : Nat := bead_pattern.length

-- Define the total number of beads in the bracelet
def total_beads : Nat := 85

-- State the theorem to prove the color of the last bead
theorem color_of_last_bead : bead_pattern.get? ((total_beads - 1) % pattern_length) = some "yellow" :=
by
  sorry

end color_of_last_bead_l89_89556


namespace quadratic_complete_square_l89_89694

theorem quadratic_complete_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) : b + c = -106 :=
by
  sorry

end quadratic_complete_square_l89_89694


namespace ferris_wheel_time_l89_89501

theorem ferris_wheel_time (R T : ℝ) (t : ℝ) (h : ℝ → ℝ) :
  R = 30 → T = 90 → (∀ t, h t = R * Real.cos ((2 * Real.pi / T) * t) + R) → h t = 45 → t = 15 :=
by
  intros hR hT hFunc hHt
  sorry

end ferris_wheel_time_l89_89501


namespace perimeter_of_excircle_opposite_leg_l89_89284

noncomputable def perimeter_of_right_triangle (a varrho_a : ℝ) : ℝ :=
  2 * varrho_a * a / (2 * varrho_a - a)

theorem perimeter_of_excircle_opposite_leg
  (a varrho_a : ℝ) (h_a_pos : 0 < a) (h_varrho_a_pos : 0 < varrho_a) :
  (perimeter_of_right_triangle a varrho_a = 2 * varrho_a * a / (2 * varrho_a - a)) :=
by
  sorry

end perimeter_of_excircle_opposite_leg_l89_89284


namespace maximum_value_of_n_l89_89363

noncomputable def max_n (a b c : ℝ) (n : ℕ) :=
  a > b ∧ b > c ∧ (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2

theorem maximum_value_of_n (a b c : ℝ) (n : ℕ) : 
  a > b → b > c → (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2 :=
  by sorry

end maximum_value_of_n_l89_89363


namespace remainder_of_x_plus_2_power_2008_l89_89996

-- Given: x^3 ≡ 1 (mod x^2 + x + 1)
def given_condition : Prop := ∀ x : ℤ, (x^3 - 1) % (x^2 + x + 1) = 0

-- To prove: The remainder when (x + 2)^2008 is divided by x^2 + x + 1 is 1
theorem remainder_of_x_plus_2_power_2008 (x : ℤ) (h : given_condition) :
  ((x + 2) ^ 2008) % (x^2 + x + 1) = 1 := by
  sorry

end remainder_of_x_plus_2_power_2008_l89_89996


namespace circle_radius_of_equal_area_l89_89090

theorem circle_radius_of_equal_area (A B C D : Type) (r : ℝ) (π : ℝ) 
  (h_rect_area : 8 * 9 = 72)
  (h_circle_area : π * r ^ 2 = 36) :
  r = 6 / Real.sqrt π :=
by
  sorry

end circle_radius_of_equal_area_l89_89090


namespace roots_poly_eq_l89_89818

theorem roots_poly_eq (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : d = 0) (root1_eq : 64 * a + 16 * b + 4 * c = 0) (root2_eq : -27 * a + 9 * b - 3 * c = 0) :
  (b + c) / a = -13 :=
by {
  sorry
}

end roots_poly_eq_l89_89818


namespace stickers_on_first_page_l89_89294

theorem stickers_on_first_page :
  ∀ (a b c d e : ℕ), 
    (b = 16) →
    (c = 24) →
    (d = 32) →
    (e = 40) →
    (b - a = 8) →
    (c - b = 8) →
    (d - c = 8) →
    (e - d = 8) →
    a = 8 :=
by
  intros a b c d e hb hc hd he h1 h2 h3 h4
  -- Proof would go here
  sorry

end stickers_on_first_page_l89_89294


namespace marie_saves_money_in_17_days_l89_89318

noncomputable def number_of_days_needed (cash_register_cost revenue tax_rate costs : ℝ) : ℕ := 
  let net_revenue := revenue / (1 + tax_rate) 
  let daily_profit := net_revenue - costs
  Nat.ceil (cash_register_cost / daily_profit)

def marie_problem_conditions : Prop := 
  let bread_daily_revenue := 40 * 2
  let bagels_daily_revenue := 20 * 1.5
  let cakes_daily_revenue := 6 * 12
  let muffins_daily_revenue := 10 * 3
  let daily_revenue := bread_daily_revenue + bagels_daily_revenue + cakes_daily_revenue + muffins_daily_revenue
  let fixed_daily_costs := 20 + 2 + 80 + 30
  fixed_daily_costs = 132 ∧ daily_revenue = 212 ∧ 8 / 100 = 0.08

theorem marie_saves_money_in_17_days : marie_problem_conditions → number_of_days_needed 1040 212 0.08 132 = 17 := 
by 
  intro h
  -- Proof goes here.
  sorry

end marie_saves_money_in_17_days_l89_89318


namespace speed_in_still_water_l89_89250

variable (upstream downstream : ℝ)

-- Conditions
def upstream_speed : Prop := upstream = 26
def downstream_speed : Prop := downstream = 40

-- Question and correct answer
theorem speed_in_still_water (h1 : upstream_speed upstream) (h2 : downstream_speed downstream) :
  (upstream + downstream) / 2 = 33 := by
  sorry

end speed_in_still_water_l89_89250


namespace min_product_of_three_numbers_l89_89709

def SetOfNumbers : Set ℤ := {-9, -5, -1, 1, 3, 5, 8}

theorem min_product_of_three_numbers : 
  ∃ (a b c : ℤ), a ∈ SetOfNumbers ∧ b ∈ SetOfNumbers ∧ c ∈ SetOfNumbers ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -360 :=
by {
  sorry
}

end min_product_of_three_numbers_l89_89709


namespace treasure_chest_coins_l89_89288

theorem treasure_chest_coins (hours : ℕ) (coins_per_hour : ℕ) (total_coins : ℕ) :
  hours = 8 → coins_per_hour = 25 → total_coins = hours * coins_per_hour → total_coins = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end treasure_chest_coins_l89_89288


namespace find_k_l89_89355

theorem find_k (k : ℝ) :
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, 7)
  ((a.1 - c.1) * b.2 - (a.2 - c.2) * b.1 = 0) → k = 5 := 
by
  sorry

end find_k_l89_89355


namespace ConeCannotHaveSquarePlanView_l89_89673

def PlanViewIsSquare (solid : Type) : Prop :=
  -- Placeholder to denote the property that the plan view of a solid is a square
  sorry

def IsCone (solid : Type) : Prop :=
  -- Placeholder to denote the property that the solid is a cone
  sorry

theorem ConeCannotHaveSquarePlanView (solid : Type) :
  (PlanViewIsSquare solid) → ¬ (IsCone solid) :=
sorry

end ConeCannotHaveSquarePlanView_l89_89673


namespace alcohol_to_water_ratio_l89_89373

theorem alcohol_to_water_ratio (p q r : ℝ) :
  let alcohol := (p / (p + 1) + q / (q + 1) + r / (r + 1))
  let water := (1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1))
  (alcohol / water) = (p * q * r + p * q + p * r + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 1) :=
sorry

end alcohol_to_water_ratio_l89_89373


namespace sin_double_angle_given_sum_identity_l89_89376

theorem sin_double_angle_given_sum_identity {α : ℝ} 
  (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 5 / 5) : 
  Real.sin (2 * α) = -3 / 5 := 
by 
  sorry

end sin_double_angle_given_sum_identity_l89_89376


namespace vertical_asymptote_l89_89970

noncomputable def y (x : ℝ) : ℝ := (3 * x + 1) / (7 * x - 10)

theorem vertical_asymptote (x : ℝ) : (7 * x - 10 = 0) → (x = 10 / 7) :=
by
  intro h
  linarith [h]

#check vertical_asymptote

end vertical_asymptote_l89_89970


namespace vampire_conversion_l89_89829

theorem vampire_conversion (x : ℕ) 
  (h_population : village_population = 300)
  (h_initial_vampires : initial_vampires = 2)
  (h_two_nights_vampires : 2 + 2 * x + x * (2 + 2 * x) = 72) :
  x = 5 :=
by
  -- Proof will be added here
  sorry

end vampire_conversion_l89_89829


namespace calculate_a_plus_b_l89_89513

theorem calculate_a_plus_b (a b : ℝ) (h1 : 3 = a + b / 2) (h2 : 2 = a + b / 4) : a + b = 5 :=
by
  sorry

end calculate_a_plus_b_l89_89513


namespace inequality_solution_l89_89603

theorem inequality_solution (x : ℝ) :
  (x - 2 > 1) ∧ (-2 * x ≤ 4) ↔ (x > 3) :=
by
  sorry

end inequality_solution_l89_89603


namespace triangle_AX_length_l89_89765

noncomputable def length_AX (AB AC BC : ℝ) (h1 : AB = 60) (h2 : AC = 34) (h3 : BC = 52) : ℝ :=
  1020 / 43

theorem triangle_AX_length 
  (AB AC BC AX : ℝ)
  (h1 : AB = 60)
  (h2 : AC = 34)
  (h3 : BC = 52)
  (h4 : AX + (AB - AX) = AB)
  (h5 : AX / (AB - AX) = AC / BC) :
  AX = 1020 / 43 := 
sorry

end triangle_AX_length_l89_89765


namespace least_integer_x_l89_89797

theorem least_integer_x (x : ℤ) : (2 * |x| + 7 < 17) → x = -4 := by
  sorry

end least_integer_x_l89_89797


namespace coefficient_a9_of_polynomial_l89_89616

theorem coefficient_a9_of_polynomial (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, x^3 + x^10 = a_0 + 
    a_1 * (x + 1) + 
    a_2 * (x + 1)^2 + 
    a_3 * (x + 1)^3 + 
    a_4 * (x + 1)^4 + 
    a_5 * (x + 1)^5 + 
    a_6 * (x + 1)^6 + 
    a_7 * (x + 1)^7 + 
    a_8 * (x + 1)^8 + 
    a_9 * (x + 1)^9 + 
    a_10 * (x + 1)^10) 
  → a_9 = -10 :=
by
  intro h
  sorry

end coefficient_a9_of_polynomial_l89_89616


namespace least_subtraction_divisible_l89_89106

theorem least_subtraction_divisible (n : ℕ) (h : n = 3830) (lcm_val : ℕ) (hlcm : lcm_val = Nat.lcm (Nat.lcm 3 7) 11) 
(largest_multiple : ℕ) (h_largest : largest_multiple = (n / lcm_val) * lcm_val) :
  ∃ x : ℕ, x = n - largest_multiple ∧ x = 134 := 
by
  sorry

end least_subtraction_divisible_l89_89106


namespace prime_condition_l89_89746

theorem prime_condition (p : ℕ) (h_prime: Nat.Prime p) :
  (∃ m n : ℤ, p = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) ↔ p = 2 ∨ p = 5 :=
by
  sorry

end prime_condition_l89_89746


namespace initial_range_without_telescope_l89_89962

variable (V : ℝ)

def telescope_increases_range (V : ℝ) : Prop :=
  V + 0.875 * V = 150

theorem initial_range_without_telescope (V : ℝ) (h : telescope_increases_range V) : V = 80 :=
by
  sorry

end initial_range_without_telescope_l89_89962


namespace problem_statement_l89_89724

-- Definitions based on the conditions
def P : Prop := ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (x / (x - 1) < 0)
def Q : Prop := ∀ (A B : ℝ), (A > B) → (A > 90 ∨ B < 90)

-- The proof problem statement
theorem problem_statement : P ∧ ¬Q := 
by
  sorry

end problem_statement_l89_89724


namespace tim_total_spent_l89_89181

-- Define the given conditions
def lunch_cost : ℝ := 50.20
def tip_percentage : ℝ := 0.20

-- Define the total amount spent
def total_amount_spent : ℝ := 60.24

-- Prove the total amount spent given the conditions
theorem tim_total_spent : lunch_cost + (tip_percentage * lunch_cost) = total_amount_spent := by
  -- This is the proof statement corresponding to the problem; the proof itself is not required for this task
  sorry

end tim_total_spent_l89_89181


namespace find_f_neg1_l89_89906

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_neg1 :
  (∀ x, f (-x) = -f x) →
  (∀ x, (0 < x) → f x = 2 * x * (x + 1)) →
  f (-1) = -4 := by
  intros h1 h2
  sorry

end find_f_neg1_l89_89906


namespace rainy_days_l89_89546

theorem rainy_days
  (rain_on_first_day : ℕ) (rain_on_second_day : ℕ) (rain_on_third_day : ℕ) (sum_of_first_two_days : ℕ)
  (h1 : rain_on_first_day = 4)
  (h2 : rain_on_second_day = 5 * rain_on_first_day)
  (h3 : sum_of_first_two_days = rain_on_first_day + rain_on_second_day)
  (h4 : rain_on_third_day = sum_of_first_two_days - 6) :
  rain_on_third_day = 18 :=
by
  sorry

end rainy_days_l89_89546


namespace new_room_area_l89_89646

def holden_master_bedroom : Nat := 309
def holden_master_bathroom : Nat := 150

theorem new_room_area : 
  (holden_master_bedroom + holden_master_bathroom) * 2 = 918 := 
by
  -- This is where the proof would go
  sorry

end new_room_area_l89_89646


namespace french_fries_cost_is_correct_l89_89492

def burger_cost : ℝ := 5
def soft_drink_cost : ℝ := 3
def special_burger_meal_cost : ℝ := 9.5

def french_fries_cost : ℝ :=
  special_burger_meal_cost - (burger_cost + soft_drink_cost)

theorem french_fries_cost_is_correct :
  french_fries_cost = 1.5 :=
by
  unfold french_fries_cost
  unfold special_burger_meal_cost
  unfold burger_cost
  unfold soft_drink_cost
  sorry

end french_fries_cost_is_correct_l89_89492


namespace unit_digit_3_pow_2023_l89_89127

def unit_digit_pattern (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0

theorem unit_digit_3_pow_2023 : unit_digit_pattern 2023 = 7 :=
by sorry

end unit_digit_3_pow_2023_l89_89127


namespace father_l89_89656

variable {son_age : ℕ} -- Son's present age
variable {father_age : ℕ} -- Father's present age

-- Conditions
def father_is_four_times_son (son_age father_age : ℕ) : Prop := father_age = 4 * son_age
def sum_of_ages_ten_years_ago (son_age father_age : ℕ) : Prop := (son_age - 10) + (father_age - 10) = 60

-- Theorem statement
theorem father's_present_age 
  (son_age father_age : ℕ)
  (h1 : father_is_four_times_son son_age father_age) 
  (h2 : sum_of_ages_ten_years_ago son_age father_age) : 
  father_age = 64 :=
sorry

end father_l89_89656


namespace problem1_problem2_l89_89328

theorem problem1 (α : ℝ) (h : Real.tan α = -2) : 
  (Real.sin α + 5 * Real.cos α) / (-2 * Real.cos α + Real.sin α) = -3 / 4 :=
sorry

theorem problem2 (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (α - 5 * Real.pi) * Real.sin (3 * Real.pi / 2 - α) = -2 / 5 :=
sorry

end problem1_problem2_l89_89328


namespace solve_quadratic_equation_l89_89988

theorem solve_quadratic_equation (x : ℝ) :
  (6 * x^2 - 3 * x - 1 = 2 * x - 2) ↔ (x = 1 / 3 ∨ x = 1 / 2) :=
by sorry

end solve_quadratic_equation_l89_89988


namespace trekking_adults_l89_89952

theorem trekking_adults
  (A : ℕ)
  (C : ℕ)
  (meal_for_adults : ℕ)
  (meal_for_children : ℕ)
  (remaining_food_children : ℕ) :
  C = 70 →
  meal_for_adults = 70 →
  meal_for_children = 90 →
  remaining_food_children = 72 →
  A - 14 = (meal_for_adults - 14) →
  A = 56 :=
sorry

end trekking_adults_l89_89952


namespace coat_lifetime_15_l89_89112

noncomputable def coat_lifetime : ℕ :=
  let cost_coat_expensive := 300
  let cost_coat_cheap := 120
  let years_cheap := 5
  let year_saving := 120
  let duration_comparison := 30
  let yearly_cost_cheaper := cost_coat_cheap / years_cheap
  let yearly_savings := year_saving / duration_comparison
  let cost_savings := yearly_cost_cheaper * duration_comparison - cost_coat_expensive * duration_comparison / (yearly_savings + (cost_coat_expensive / cost_coat_cheap))
  cost_savings

theorem coat_lifetime_15 : coat_lifetime = 15 := by
  sorry

end coat_lifetime_15_l89_89112


namespace number_of_correct_conclusions_l89_89621

noncomputable def A (x : ℝ) : ℝ := 2 * x^2
noncomputable def B (x : ℝ) : ℝ := x + 1
noncomputable def C (x : ℝ) : ℝ := -2 * x
noncomputable def D (y : ℝ) : ℝ := y^2
noncomputable def E (x y : ℝ) : ℝ := 2 * x - y

def conclusion1 (y : ℤ) : Prop := 
  0 < ((B (0 : ℝ)) * (C (0 : ℝ)) + A (0 : ℝ) + D y + E (0) (y : ℝ))

def conclusion2 : Prop := 
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

def M (A B C : ℝ → ℝ) (x m : ℝ) : ℝ :=
  3 * (A x - B x) + m * B x * C x

def linear_term_exists (m : ℝ) : Prop :=
  (0 : ℝ) ≠ -3 - 2 * m

def conclusion3 : Prop := 
 ∀ m : ℝ, (¬ linear_term_exists m ∧ M A B C (0 : ℝ) m > -3) 

def p (x y : ℝ) := 
  2 * (x + 1) ^ 2 + (y - 1) ^ 2 = 1

theorem number_of_correct_conclusions : Prop := 
  (¬ conclusion1 1) ∧ (conclusion2) ∧ (¬ conclusion3)

end number_of_correct_conclusions_l89_89621


namespace unique_int_pair_exists_l89_89907

theorem unique_int_pair_exists (a b : ℤ) : 
  ∃! (x y : ℤ), (x + 2 * y - a)^2 + (2 * x - y - b)^2 ≤ 1 :=
by
  sorry

end unique_int_pair_exists_l89_89907


namespace new_year_season_markup_l89_89409

variable {C : ℝ} (hC : 0 < C)

theorem new_year_season_markup (h1 : ∀ C, C > 0 → ∃ P1, P1 = 1.20 * C)
                              (h2 : ∀ (P1 M : ℝ), M >= 0 → ∃ P2, P2 = P1 * (1 + M / 100))
                              (h3 : ∀ P2, ∃ P3, P3 = P2 * 0.91)
                              (h4 : ∃ P3, P3 = 1.365 * C) :
  ∃ M, M = 25 := 
by 
  sorry

end new_year_season_markup_l89_89409


namespace range_of_m_l89_89851

noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, -m * x ^ 2 + 2 * x - m > 0
noncomputable def q (m : ℝ) : Prop := ∀ x > 0, (4 / x + x - m + 1) > 2

theorem range_of_m : 
  (∃ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m)) → (∃ (m : ℝ), -1 ≤ m ∧ m < 3) :=
by
  intros h
  sorry

end range_of_m_l89_89851


namespace min_buses_needed_l89_89101

-- Given definitions from conditions
def students_per_bus : ℕ := 45
def total_students : ℕ := 495

-- The proposition to prove
theorem min_buses_needed : ∃ n : ℕ, 45 * n ≥ 495 ∧ (∀ m : ℕ, 45 * m ≥ 495 → n ≤ m) :=
by
  -- Preliminary calculations that lead to the solution
  let n := total_students / students_per_bus
  have h : total_students % students_per_bus = 0 := by sorry
  
  -- Conclude that the minimum n so that 45 * n ≥ 495 is indeed 11
  exact ⟨n, by sorry, by sorry⟩

end min_buses_needed_l89_89101


namespace find_m_l89_89613

-- Definitions for vectors and dot products
structure Vector :=
  (i : ℝ)
  (j : ℝ)

def dot_product (a b : Vector) : ℝ :=
  a.i * b.i + a.j * b.j

-- Given conditions
def i : Vector := ⟨1, 0⟩
def j : Vector := ⟨0, 1⟩

def a : Vector := ⟨2, 3⟩
def b (m : ℝ) : Vector := ⟨1, -m⟩

-- The main goal
theorem find_m (m : ℝ) (h: dot_product a (b m) = 1) : m = 1 / 3 :=
by {
  -- Calculation reaches the same \(m = 1/3\)
  sorry
}

end find_m_l89_89613


namespace cats_and_dogs_biscuits_l89_89326

theorem cats_and_dogs_biscuits 
  (d c : ℕ) 
  (h1 : d + c = 10) 
  (h2 : 6 * d + 5 * c = 56) 
  : d = 6 ∧ c = 4 := 
by 
  sorry

end cats_and_dogs_biscuits_l89_89326


namespace cos_alpha_value_l89_89845

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : Real.cos α = 1 / 5 :=
sorry

end cos_alpha_value_l89_89845


namespace cats_in_shelter_l89_89412

-- Define the initial conditions
def initial_cats := 20
def monday_addition := 2
def tuesday_addition := 1
def wednesday_subtraction := 3 * 2

-- Problem statement: Prove that the total number of cats after all events is 17
theorem cats_in_shelter : initial_cats + monday_addition + tuesday_addition - wednesday_subtraction = 17 :=
by
  sorry

end cats_in_shelter_l89_89412


namespace intersection_of_A_and_B_l89_89413

-- Definitions of sets A and B
def A : Set ℝ := { x | -2 < x ∧ x < 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

-- Definition of the expected intersection of A and B
def expected_intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The main theorem stating the proof problem
theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ (A ∩ B) ↔ x ∈ expected_intersection :=
by
  intro x
  sorry

end intersection_of_A_and_B_l89_89413


namespace lee_charge_per_action_figure_l89_89743

def cost_of_sneakers : ℕ := 90
def amount_saved : ℕ := 15
def action_figures_sold : ℕ := 10
def amount_left_after_purchase : ℕ := 25
def amount_charged_per_action_figure : ℕ := 10

theorem lee_charge_per_action_figure :
  (cost_of_sneakers - amount_saved + amount_left_after_purchase = 
  action_figures_sold * amount_charged_per_action_figure) :=
by
  -- The proof steps will go here, but they are not required in the statement.
  sorry

end lee_charge_per_action_figure_l89_89743


namespace adult_meals_sold_l89_89615

theorem adult_meals_sold (k a : ℕ) (h1 : 10 * a = 7 * k) (h2 : k = 70) : a = 49 :=
by
  sorry

end adult_meals_sold_l89_89615


namespace units_digit_sum_2_pow_a_5_pow_b_l89_89263

theorem units_digit_sum_2_pow_a_5_pow_b (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 100)
  (h2 : 1 ≤ b ∧ b ≤ 100) :
  (2 ^ a + 5 ^ b) % 10 ≠ 8 :=
sorry

end units_digit_sum_2_pow_a_5_pow_b_l89_89263


namespace problem_I_problem_II_l89_89312

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem problem_I (x : ℝ) : (f x > 7 - x) ↔ (x < -6 ∨ x > 2) := 
by 
  sorry

theorem problem_II (m : ℝ) : (∃ x : ℝ, f x ≤ abs (3 * m - 2)) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
by 
  sorry

end problem_I_problem_II_l89_89312


namespace crayons_left_l89_89835

theorem crayons_left (initial_crayons erasers_left more_crayons_than_erasers : ℕ)
    (H1 : initial_crayons = 531)
    (H2 : erasers_left = 38)
    (H3 : more_crayons_than_erasers = 353) :
    (initial_crayons - (initial_crayons - (erasers_left + more_crayons_than_erasers)) = 391) :=
by 
  sorry

end crayons_left_l89_89835


namespace johns_pool_depth_l89_89183

theorem johns_pool_depth : 
  ∀ (j s : ℕ), (j = 2 * s + 5) → (s = 5) → (j = 15) := 
by 
  intros j s h1 h2
  rw [h2] at h1
  exact h1

end johns_pool_depth_l89_89183


namespace relationship_between_f_l89_89032

-- Given definitions
def quadratic_parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def axis_of_symmetry (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y ↔ x + y = 2 * α

-- The problem statement to prove in Lean 4
theorem relationship_between_f (a b c x : ℝ) (hpos : x > 0) (apos : a > 0) :
  axis_of_symmetry (quadratic_parabola a b c) 1 →
  quadratic_parabola a b c (3^x) > quadratic_parabola a b c (2^x) :=
by
  sorry

end relationship_between_f_l89_89032


namespace people_and_cars_equation_l89_89174

theorem people_and_cars_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end people_and_cars_equation_l89_89174


namespace farm_problem_l89_89916

theorem farm_problem
    (initial_cows : ℕ := 12)
    (initial_pigs : ℕ := 34)
    (remaining_animals : ℕ := 30)
    (C : ℕ)
    (P : ℕ)
    (h1 : P = 3 * C)
    (h2 : initial_cows - C + (initial_pigs - P) = remaining_animals) :
    C = 4 :=
by
  sorry

end farm_problem_l89_89916


namespace friends_share_difference_l89_89701

-- Define the initial conditions
def gift_cost : ℕ := 120
def initial_friends : ℕ := 10
def remaining_friends : ℕ := 6

-- Define the initial and new shares
def initial_share : ℕ := gift_cost / initial_friends
def new_share : ℕ := gift_cost / remaining_friends

-- Define the difference between the new share and the initial share
def share_difference : ℕ := new_share - initial_share

-- The theorem to be proved
theorem friends_share_difference : share_difference = 8 :=
by
  sorry

end friends_share_difference_l89_89701


namespace total_workers_l89_89604

theorem total_workers (h_beavers : ℕ := 318) (h_spiders : ℕ := 544) :
  h_beavers + h_spiders = 862 :=
by
  sorry

end total_workers_l89_89604


namespace dot_product_eq_neg29_l89_89606

def v := (3, -2)
def w := (-5, 7)

theorem dot_product_eq_neg29 : (v.1 * w.1 + v.2 * w.2) = -29 := 
by 
  -- this is where the detailed proof will occur
  sorry

end dot_product_eq_neg29_l89_89606


namespace forty_percent_jacqueline_candy_l89_89182

def fred_candy : ℕ := 12
def uncle_bob_candy : ℕ := fred_candy + 6
def total_fred_uncle_bob_candy : ℕ := fred_candy + uncle_bob_candy
def jacqueline_candy : ℕ := 10 * total_fred_uncle_bob_candy

theorem forty_percent_jacqueline_candy : (40 * jacqueline_candy) / 100 = 120 := by
  sorry

end forty_percent_jacqueline_candy_l89_89182


namespace modulus_of_z_eq_sqrt2_l89_89451

noncomputable def complex_z : ℂ := (1 + 3 * Complex.I) / (2 - Complex.I)

theorem modulus_of_z_eq_sqrt2 : Complex.abs complex_z = Real.sqrt 2 := by
  sorry

end modulus_of_z_eq_sqrt2_l89_89451


namespace rise_in_water_level_l89_89362

-- Define the conditions related to the cube and the vessel
def edge_length := 15 -- in cm
def base_length := 20 -- in cm
def base_width := 15 -- in cm

-- Calculate volumes and areas
def V_cube := edge_length ^ 3
def A_base := base_length * base_width

-- Declare the mathematical proof problem statement
theorem rise_in_water_level : 
  (V_cube / A_base : ℝ) = 11.25 :=
by
  -- edge_length, V_cube, A_base are all already defined
  -- This particularly proves (15^3) / (20 * 15) = 11.25
  sorry

end rise_in_water_level_l89_89362


namespace find_a_l89_89148

open Set

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := {2, 4, a^3 - 2 * a^2 - a + 7}
def B (a : ℝ) : Set ℝ := {-4, a + 3, a^2 - 2 * a + 2, a^3 + a^2 + 3 * a + 7}

theorem find_a (h : (A a ∩ B a) = {2, 5}) : a = 2 :=
sorry

end find_a_l89_89148


namespace not_valid_mapping_circle_triangle_l89_89361

inductive Point
| mk : ℝ → ℝ → Point

inductive Circle
| mk : ℝ → ℝ → ℝ → Circle

inductive Triangle
| mk : Point → Point → Point → Triangle

open Point (mk)
open Circle (mk)
open Triangle (mk)

def valid_mapping (A B : Type) (f : A → B) := ∀ a₁ a₂ : A, f a₁ = f a₂ → a₁ = a₂

def inscribed_triangle_mapping (c : Circle) : Triangle := sorry -- map a circle to one of its inscribed triangles

theorem not_valid_mapping_circle_triangle :
  ¬ valid_mapping Circle Triangle inscribed_triangle_mapping :=
sorry

end not_valid_mapping_circle_triangle_l89_89361


namespace brick_width_l89_89072

theorem brick_width (length_courtyard : ℕ) (width_courtyard : ℕ) (num_bricks : ℕ) (brick_length : ℕ) (total_area : ℕ) (brick_area : ℕ) (w : ℕ)
  (h1 : length_courtyard = 1800)
  (h2 : width_courtyard = 1200)
  (h3 : num_bricks = 30000)
  (h4 : brick_length = 12)
  (h5 : total_area = length_courtyard * width_courtyard)
  (h6 : total_area = num_bricks * brick_area)
  (h7 : brick_area = brick_length * w) :
  w = 6 :=
by
  sorry

end brick_width_l89_89072


namespace arccos_cos_eight_l89_89791

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi :=
by sorry

end arccos_cos_eight_l89_89791


namespace license_plate_combinations_l89_89974

def consonants_count := 21
def vowels_count := 5
def digits_count := 10

theorem license_plate_combinations : 
  consonants_count * vowels_count * consonants_count * digits_count * vowels_count = 110250 :=
by
  sorry

end license_plate_combinations_l89_89974


namespace simplifiedtown_path_difference_l89_89523

/-- In Simplifiedtown, all streets are 30 feet wide. Each enclosed block forms a square with 
each side measuring 400 feet. Sarah runs exactly next to the block on a path that is 400 feet 
from the block's inner edge while Maude runs on the outer edge of the street opposite to 
Sarah. Prove that Maude runs 120 feet more than Sarah for each lap around the block. -/
theorem simplifiedtown_path_difference :
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  maude_lap - sarah_lap = 120 :=
by
  let street_width := 30
  let block_side := 400
  let sarah_path := block_side
  let maude_path := block_side + street_width
  let sarah_lap := 4 * sarah_path
  let maude_lap := 4 * maude_path
  show maude_lap - sarah_lap = 120
  sorry

end simplifiedtown_path_difference_l89_89523


namespace eugene_boxes_needed_l89_89721

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end eugene_boxes_needed_l89_89721


namespace sin_double_angle_l89_89888

theorem sin_double_angle (x : ℝ) (h : Real.cos (π / 4 - x) = -3 / 5) : Real.sin (2 * x) = -7 / 25 :=
by
  sorry

end sin_double_angle_l89_89888


namespace bus_stop_time_l89_89524

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℕ) 
(distance : ℕ) (time_without_stoppages time_with_stoppages : ℝ) :
  speed_without_stoppages = 80 ∧ speed_with_stoppages = 40 ∧ distance = 80 ∧
  time_without_stoppages = distance / speed_without_stoppages ∧
  time_with_stoppages = distance / speed_with_stoppages →
  (time_with_stoppages - time_without_stoppages) * 60 = 30 :=
by
  sorry

end bus_stop_time_l89_89524


namespace pine_tree_next_one_in_between_l89_89806

theorem pine_tree_next_one_in_between (n : ℕ) (p s : ℕ) (trees : n = 2019) (pines : p = 1009) (spruces : s = 1010)
    (equal_intervals : true) : 
    ∃ (i : ℕ), (i < n) ∧ ((i + 1) % n ∈ {j | j < p}) ∧ ((i + 3) % n ∈ {j | j < p}) :=
  sorry

end pine_tree_next_one_in_between_l89_89806


namespace nickel_ate_3_chocolates_l89_89924

theorem nickel_ate_3_chocolates (R N : ℕ) (h1 : R = 7) (h2 : R = N + 4) : N = 3 := by
  sorry

end nickel_ate_3_chocolates_l89_89924


namespace symmetric_point_coordinates_l89_89517

theorem symmetric_point_coordinates (a b : ℝ) (hp : (3, 4) = (a + 3, b + 4)) :
  (a, b) = (5, 2) :=
  sorry

end symmetric_point_coordinates_l89_89517


namespace two_pow_58_plus_one_factored_l89_89734

theorem two_pow_58_plus_one_factored :
  ∃ (a b c : ℕ), 2 < a ∧ 2 < b ∧ 2 < c ∧ 2 ^ 58 + 1 = a * b * c :=
sorry

end two_pow_58_plus_one_factored_l89_89734


namespace car_speed_l89_89030

theorem car_speed {vp vc : ℚ} (h1 : vp = 7 / 2) (h2 : vc = 6 * vp) : 
  vc = 21 := 
by 
  sorry

end car_speed_l89_89030


namespace correct_equation_l89_89531

theorem correct_equation (x : ℝ) (hx : x > 80) : 
  353 / (x - 80) - 353 / x = 5 / 3 :=
sorry

end correct_equation_l89_89531


namespace remainder_when_divided_by_7_l89_89351

theorem remainder_when_divided_by_7 (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ p = 5) : n % 7 = 5 :=
by
  sorry

end remainder_when_divided_by_7_l89_89351


namespace cash_price_eq_8000_l89_89742

noncomputable def cash_price (d m s : ℕ) : ℕ :=
  d + 30 * m - s

theorem cash_price_eq_8000 :
  cash_price 3000 300 4000 = 8000 :=
by
  -- Proof omitted.
  sorry

end cash_price_eq_8000_l89_89742


namespace third_cyclist_speed_l89_89710

theorem third_cyclist_speed (s1 s3 : ℝ) :
  (∃ s1 s3 : ℝ,
    (∀ t : ℝ, t > 0 → (s1 > s3) ∧ (20 = abs (10 * t - s1 * t)) ∧ (5 = abs (s1 * t - s3 * t)) ∧ (s1 ≥ 10))) →
  (s3 = 25 ∨ s3 = 5) :=
by sorry

end third_cyclist_speed_l89_89710


namespace value_of_expression_l89_89229

theorem value_of_expression (p q r s : ℝ) (h : -27 * p + 9 * q - 3 * r + s = -7) : 
  4 * p - 2 * q + r - s = 7 :=
by
  sorry

end value_of_expression_l89_89229


namespace bee_total_correct_l89_89435

def initial_bees : Nat := 16
def incoming_bees : Nat := 10
def total_bees : Nat := initial_bees + incoming_bees

theorem bee_total_correct : total_bees = 26 := by
  sorry

end bee_total_correct_l89_89435


namespace bicycle_cost_calculation_l89_89579

theorem bicycle_cost_calculation 
  (CP_A CP_B CP_C : ℝ)
  (h1 : CP_B = 1.20 * CP_A)
  (h2 : CP_C = 1.25 * CP_B)
  (h3 : CP_C = 225) :
  CP_A = 150 :=
by
  sorry

end bicycle_cost_calculation_l89_89579


namespace Mark_less_than_Craig_l89_89577

-- Definitions for the conditions
def Dave_weight : ℕ := 175
def Dave_bench_press : ℕ := Dave_weight * 3
def Craig_bench_press : ℕ := (20 * Dave_bench_press) / 100
def Mark_bench_press : ℕ := 55

-- The theorem to be proven
theorem Mark_less_than_Craig : Craig_bench_press - Mark_bench_press = 50 :=
by
  sorry

end Mark_less_than_Craig_l89_89577


namespace addends_are_negative_l89_89939

theorem addends_are_negative (a b : ℤ) (h1 : a + b < a) (h2 : a + b < b) : a < 0 ∧ b < 0 := 
sorry

end addends_are_negative_l89_89939


namespace households_both_brands_l89_89367

theorem households_both_brands
  (T : ℕ) (N : ℕ) (A : ℕ) (B : ℕ)
  (hT : T = 300) (hN : N = 80) (hA : A = 60) (hB : ∃ X : ℕ, B = 3 * X ∧ T = N + A + B + X) :
  ∃ X : ℕ, X = 40 :=
by
  -- Upon extracting values from conditions, solving for both brand users X = 40
  sorry

end households_both_brands_l89_89367


namespace total_amount_l89_89908

noncomputable def initial_amounts (a j t : ℕ) := (t = 24)
noncomputable def redistribution_amounts (a j t a' j' t' : ℕ) :=
  a' = 3 * (2 * (a - 2 * j - 24)) ∧
  j' = 3 * (3 * j - (a - 2 * j - 24 + 48)) ∧
  t' = 144 - (6 * (a - 2 * j - 24) + 9 * j - 3 * (a - 2 * j - 24 + 48))

theorem total_amount (a j t a' j' t' : ℕ) (h1 : t = 24)
  (h2 : redistribution_amounts a j t a' j' t')
  (h3 : t' = 24) : 
  a + j + t = 72 :=
sorry

end total_amount_l89_89908


namespace expected_profit_correct_l89_89046

-- Define the conditions
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit calculation
def expected_profit : ℝ := (winning_probability * prize) - ticket_cost

-- The theorem we want to prove
theorem expected_profit_correct : expected_profit = -1.5 := by
  sorry

end expected_profit_correct_l89_89046


namespace find_min_sum_of_squares_l89_89605

open Real

theorem find_min_sum_of_squares
  (x1 x2 x3 : ℝ)
  (h1 : 0 < x1)
  (h2 : 0 < x2)
  (h3 : 0 < x3)
  (h4 : 2 * x1 + 4 * x2 + 6 * x3 = 120) :
  x1^2 + x2^2 + x3^2 >= 350 :=
sorry

end find_min_sum_of_squares_l89_89605


namespace real_condition_complex_condition_pure_imaginary_condition_l89_89799

-- Definitions for our conditions
def is_real (z : ℂ) : Prop := z.im = 0
def is_complex (z : ℂ) : Prop := z.im ≠ 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- The given complex number definition
def z (m : ℝ) : ℂ := { re := m^2 + m, im := m^2 - 1 }

-- Prove that for z to be a real number, m must be ±1
theorem real_condition (m : ℝ) : is_real (z m) ↔ m = 1 ∨ m = -1 := 
sorry

-- Prove that for z to be a complex number, m must not be ±1 
theorem complex_condition (m : ℝ) : is_complex (z m) ↔ m ≠ 1 ∧ m ≠ -1 := 
sorry 

-- Prove that for z to be a pure imaginary number, m must be 0
theorem pure_imaginary_condition (m : ℝ) : is_pure_imaginary (z m) ↔ m = 0 := 
sorry 

end real_condition_complex_condition_pure_imaginary_condition_l89_89799


namespace trajectory_equation_l89_89860

-- Definitions and conditions
noncomputable def tangent_to_x_axis (M : ℝ × ℝ) := M.snd = 0
noncomputable def internally_tangent (M : ℝ × ℝ) := ∃ (r : ℝ), 0 < r ∧ M.1^2 + (M.2 - r)^2 = 4

-- The theorem stating the proof problem
theorem trajectory_equation (M : ℝ × ℝ) (h_tangent : tangent_to_x_axis M) (h_internal_tangent : internally_tangent M) :
  (∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧ M.fst^2 = 4 * (y - 1)) :=
sorry

end trajectory_equation_l89_89860


namespace temperature_decrease_l89_89370

theorem temperature_decrease (T : ℝ) 
    (h1 : T * (3 / 4) = T - 21)
    (h2 : T > 0) : 
    T = 84 := 
  sorry

end temperature_decrease_l89_89370


namespace quadruple_dimensions_increase_volume_l89_89685

theorem quadruple_dimensions_increase_volume 
  (V_original : ℝ) (quad_factor : ℝ)
  (initial_volume : V_original = 5)
  (quad_factor_val : quad_factor = 4) :
  V_original * (quad_factor ^ 3) = 320 := 
by 
  -- Introduce necessary variables and conditions
  let V_modified := V_original * (quad_factor ^ 3)
  
  -- Assert the calculations based on the given conditions
  have initial : V_original = 5 := initial_volume
  have quad : quad_factor = 4 := quad_factor_val
  
  -- Skip the detailed proof with sorry
  sorry


end quadruple_dimensions_increase_volume_l89_89685


namespace base7_sum_correct_l89_89960

theorem base7_sum_correct : 
  ∃ (A B C : ℕ), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (A = 2 ∨ A = 3 ∨ A = 5) ∧
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧
  A + B + C = 16 :=
by
  sorry

end base7_sum_correct_l89_89960


namespace find_integer_l89_89719

theorem find_integer (x : ℕ) (h1 : (4 * x)^2 + 2 * x = 3528) : x = 14 := by
  sorry

end find_integer_l89_89719


namespace largest_reciprocal_l89_89058

-- Definitions of the given numbers
def num1 := 1 / 6
def num2 := 2 / 7
def num3 := (2 : ℝ)
def num4 := (8 : ℝ)
def num5 := (1000 : ℝ)

-- The main problem: prove that the reciprocal of 1/6 is the largest
theorem largest_reciprocal :
  (1 / num1 > 1 / num2) ∧ (1 / num1 > 1 / num3) ∧ (1 / num1 > 1 / num4) ∧ (1 / num1 > 1 / num5) :=
by
  sorry

end largest_reciprocal_l89_89058


namespace park_area_l89_89991

-- Define the width (w) and length (l) of the park
def width : Float := 11.25
def length : Float := 33.75

-- Define the perimeter and area functions
def perimeter (w l : Float) : Float := 2 * (w + l)
def area (w l : Float) : Float := w * l

-- Provide the conditions
axiom width_is_one_third_length : width = length / 3
axiom perimeter_is_90 : perimeter width length = 90

-- Theorem to prove the area given the conditions
theorem park_area : area width length = 379.6875 := by
  sorry

end park_area_l89_89991


namespace a3_eq_5_l89_89868

-- Define the geometric sequence and its properties
variables {a : ℕ → ℝ} {q : ℝ}

-- Assumptions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a 1 * (q ^ n)
axiom a1_pos : a 1 > 0
axiom a2a4_eq_25 : a 2 * a 4 = 25
axiom geom : geom_seq a q

-- Statement to prove
theorem a3_eq_5 : a 3 = 5 :=
by sorry

end a3_eq_5_l89_89868


namespace toy_playing_dogs_ratio_l89_89360

theorem toy_playing_dogs_ratio
  (d_t : ℕ) (d_r : ℕ) (d_n : ℕ) (d_b : ℕ) (d_p : ℕ)
  (h1 : d_t = 88)
  (h2 : d_r = 12)
  (h3 : d_n = 10)
  (h4 : d_b = d_t / 4)
  (h5 : d_p = d_t - d_r - d_b - d_n) :
  d_p / d_t = 1 / 2 :=
by sorry

end toy_playing_dogs_ratio_l89_89360


namespace car_journey_delay_l89_89687

theorem car_journey_delay (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time1 : ℝ) (time2 : ℝ) (delay : ℝ) :
  distance = 225 ∧ speed1 = 60 ∧ speed2 = 50 ∧ time1 = distance / speed1 ∧ time2 = distance / speed2 ∧ 
  delay = (time2 - time1) * 60 → delay = 45 :=
by
  sorry

end car_journey_delay_l89_89687


namespace sample_size_calculation_l89_89332

theorem sample_size_calculation : 
  ∀ (high_school_students junior_high_school_students sampled_high_school_students n : ℕ), 
  high_school_students = 3500 →
  junior_high_school_students = 1500 →
  sampled_high_school_students = 70 →
  n = (3500 + 1500) * 70 / 3500 →
  n = 100 :=
by
  intros high_school_students junior_high_school_students sampled_high_school_students n
  intros h1 h2 h3 h4
  sorry

end sample_size_calculation_l89_89332


namespace eval_expression_l89_89042

theorem eval_expression : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end eval_expression_l89_89042


namespace angle_A_measure_l89_89805

theorem angle_A_measure 
  (B : ℝ) 
  (angle_in_smaller_triangle : ℝ) 
  (sum_of_triangle_angles_eq_180 : ∀ (x y z : ℝ), x + y + z = 180)
  (C : ℝ) 
  (angle_pair_linear : ∀ (x y : ℝ), x + y = 180) 
  (A : ℝ) 
  (C_eq_180_minus_B : C = 180 - B) 
  (A_eq_180_minus_angle_in_smaller_triangle_minus_C : 
    A = 180 - angle_in_smaller_triangle - C) :
  A = 70 :=
by
  sorry

end angle_A_measure_l89_89805


namespace sufficient_not_necessary_l89_89466

theorem sufficient_not_necessary (x : ℝ) : (x > 3 → x > 1) ∧ ¬ (x > 1 → x > 3) :=
by 
  sorry

end sufficient_not_necessary_l89_89466


namespace cleaned_area_correct_l89_89423

def lizzie_cleaned : ℚ := 3534 + 2/3
def hilltown_team_cleaned : ℚ := 4675 + 5/8
def green_valley_cleaned : ℚ := 2847 + 7/9
def riverbank_cleaned : ℚ := 6301 + 1/3
def meadowlane_cleaned : ℚ := 3467 + 4/5

def total_cleaned : ℚ := lizzie_cleaned + hilltown_team_cleaned + green_valley_cleaned + riverbank_cleaned + meadowlane_cleaned
def total_farmland : ℚ := 28500

def remaining_area_to_clean : ℚ := total_farmland - total_cleaned

theorem cleaned_area_correct : remaining_area_to_clean = 7672.7964 :=
by
  sorry

end cleaned_area_correct_l89_89423


namespace unique_spicy_pair_l89_89426

def is_spicy (n : ℕ) : Prop :=
  let A := (n / 100) % 10
  let B := (n / 10) % 10
  let C := n % 10
  n = A^3 + B^3 + C^3

theorem unique_spicy_pair : ∃! n : ℕ, is_spicy n ∧ is_spicy (n + 1) ∧ 100 ≤ n ∧ n < 1000 ∧ n = 370 := 
sorry

end unique_spicy_pair_l89_89426


namespace equilateral_triangle_lines_l89_89844

-- Define the properties of an equilateral triangle
structure EquilateralTriangle :=
(sides_length : ℝ) -- All sides are of equal length
(angle : ℝ := 60)  -- All internal angles are 60 degrees

-- Define the concept that altitudes, medians, and angle bisectors coincide
structure CoincidingLines (T : EquilateralTriangle) :=
(altitude : T.angle = 60)
(median : T.angle = 60)
(angle_bisector : T.angle = 60)

-- Define a statement that proves the number of distinct lines in the equilateral triangle
theorem equilateral_triangle_lines (T : EquilateralTriangle) (L : CoincidingLines T) :  
  -- The total number of distinct lines consisting of altitudes, medians, and angle bisectors
  (3 = 3) :=
by
  sorry

end equilateral_triangle_lines_l89_89844


namespace g_five_eq_248_l89_89623

-- We define g, and assume it meets the conditions described.
variable (g : ℤ → ℤ)

-- Condition 1: g(1) > 1
axiom g_one_gt_one : g 1 > 1

-- Condition 2: Functional equation for g
axiom g_funct_eq (x y : ℤ) : g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y

-- Condition 3: Recursive relationship for g
axiom g_recur_eq (x : ℤ) : 3 * g x = g (x + 1) + 2 * x - 1

-- Theorem we want to prove
theorem g_five_eq_248 : g 5 = 248 := by
  sorry

end g_five_eq_248_l89_89623


namespace total_surface_area_with_holes_l89_89257

def cube_edge_length : ℝ := 5
def hole_side_length : ℝ := 2

/-- Calculate the total surface area of a modified cube with given edge length and holes -/
theorem total_surface_area_with_holes 
  (l : ℝ) (h : ℝ)
  (hl_pos : l > 0) (hh_pos : h > 0) (hh_lt_hl : h < l) : 
  (6 * l^2 - 6 * h^2 + 6 * 4 * h^2) = 222 :=
by sorry

end total_surface_area_with_holes_l89_89257


namespace terminal_angle_quadrant_l89_89932

theorem terminal_angle_quadrant : 
  let angle := -558
  let reduced_angle := angle % 360
  90 < reduced_angle ∧ reduced_angle < 180 →
  SecondQuadrant := 
by 
  intro angle reduced_angle h 
  sorry

end terminal_angle_quadrant_l89_89932


namespace project_inflation_cost_increase_l89_89885

theorem project_inflation_cost_increase :
  let original_lumber_cost := 450
  let original_nails_cost := 30
  let original_fabric_cost := 80
  let lumber_inflation := 0.2
  let nails_inflation := 0.1
  let fabric_inflation := 0.05
  
  let new_lumber_cost := original_lumber_cost * (1 + lumber_inflation)
  let new_nails_cost := original_nails_cost * (1 + nails_inflation)
  let new_fabric_cost := original_fabric_cost * (1 + fabric_inflation)
  
  let total_increased_cost := (new_lumber_cost - original_lumber_cost) 
                            + (new_nails_cost - original_nails_cost) 
                            + (new_fabric_cost - original_fabric_cost)
  total_increased_cost = 97 := sorry

end project_inflation_cost_increase_l89_89885


namespace quadratic_inequality_solution_set_l89_89808

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) : 
  ∀ x : ℝ, (x^2 - (m + 1/m) * x + 1 < 0) ↔ m < x ∧ x < 1/m :=
by
  sorry

end quadratic_inequality_solution_set_l89_89808


namespace sin_add_cos_l89_89043

theorem sin_add_cos (s72 c18 c72 s18 : ℝ) (h1 : s72 = Real.sin (72 * Real.pi / 180)) (h2 : c18 = Real.cos (18 * Real.pi / 180)) (h3 : c72 = Real.cos (72 * Real.pi / 180)) (h4 : s18 = Real.sin (18 * Real.pi / 180)) :
  s72 * c18 + c72 * s18 = 1 :=
by 
  sorry

end sin_add_cos_l89_89043


namespace mortgage_payoff_months_l89_89408

-- Declare the initial payment (P), the common ratio (r), and the total amount (S)
def initial_payment : ℕ := 100
def common_ratio : ℕ := 3
def total_amount : ℕ := 12100

-- Define a function that calculates the sum of a geometric series
noncomputable def geom_series_sum (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * (1 - r ^ n) / (1 - r)

-- The statement we need to prove
theorem mortgage_payoff_months : ∃ n : ℕ, geom_series_sum initial_payment common_ratio n = total_amount :=
by
  sorry -- Proof to be provided

end mortgage_payoff_months_l89_89408


namespace geometric_arithmetic_sequence_relation_l89_89195

theorem geometric_arithmetic_sequence_relation 
    (a : ℕ → ℝ) (b : ℕ → ℝ) (q d a1 : ℝ)
    (h1 : a 1 = a1) (h2 : b 1 = a1) (h3 : a 3 = a1 * q^2)
    (h4 : b 3 = a1 + 2 * d) (h5 : a 3 = b 3) (h6 : a1 > 0) (h7 : q^2 ≠ 1) :
    a 5 > b 5 :=
by
  -- Proof goes here
  sorry

end geometric_arithmetic_sequence_relation_l89_89195


namespace total_area_painted_correct_l89_89662

-- Defining the properties of the shed
def shed_w := 12  -- width in yards
def shed_l := 15  -- length in yards
def shed_h := 7   -- height in yards

-- Calculating area to be painted
def wall_area_1 := 2 * (shed_w * shed_h)
def wall_area_2 := 2 * (shed_l * shed_h)
def floor_ceiling_area := 2 * (shed_w * shed_l)
def total_painted_area := wall_area_1 + wall_area_2 + floor_ceiling_area

-- The theorem to be proved
theorem total_area_painted_correct :
  total_painted_area = 738 := by
  sorry

end total_area_painted_correct_l89_89662


namespace average_speed_problem_l89_89740

noncomputable def average_speed (d₁ d₂ d₃ d₄ t₁ t₂ t₃ t₄ : ℝ) : ℝ :=
  (d₁ + d₂ + d₃ + d₄) / (t₁ + t₂ + t₃ + t₄)

theorem average_speed_problem :
  average_speed 30 40 37.5 7 (30 / 35) (40 / 55) 0.5 (10 / 60) = 51 :=
by
  -- skip the proof
  sorry

end average_speed_problem_l89_89740


namespace quadratic_coefficients_l89_89641

theorem quadratic_coefficients :
  ∃ a b c : ℤ, a = 4 ∧ b = 0 ∧ c = -3 ∧ 4 * x^2 = 3 := sorry

end quadratic_coefficients_l89_89641


namespace smallest_integer_cube_ends_in_528_l89_89109

theorem smallest_integer_cube_ends_in_528 :
  ∃ (n : ℕ), (n^3 % 1000 = 528 ∧ ∀ m : ℕ, (m^3 % 1000 = 528) → m ≥ n) ∧ n = 428 :=
by
  sorry

end smallest_integer_cube_ends_in_528_l89_89109


namespace total_amount_paid_l89_89488

theorem total_amount_paid (cost_of_manicure : ℝ) (tip_percentage : ℝ) (total : ℝ) 
  (h1 : cost_of_manicure = 30) (h2 : tip_percentage = 0.3) (h3 : total = cost_of_manicure + cost_of_manicure * tip_percentage) : 
  total = 39 :=
by
  sorry

end total_amount_paid_l89_89488


namespace ab_value_l89_89137

theorem ab_value 
  (a b : ℝ) 
  (hx : 2 = b + 1) 
  (hy : a = -3) : 
  a * b = -3 :=
by
  sorry

end ab_value_l89_89137


namespace option_b_represents_factoring_l89_89166

theorem option_b_represents_factoring (x y : ℤ) :
  x^2 - 2*x*y = x * (x - 2*y) :=
sorry

end option_b_represents_factoring_l89_89166


namespace pool_capacity_l89_89209

theorem pool_capacity (C : ℝ) (h1 : 300 = 0.30 * C) : C = 1000 :=
by
  sorry

end pool_capacity_l89_89209


namespace Jeongyeon_record_is_1_44_m_l89_89534

def Eunseol_record_in_cm : ℕ := 100 + 35
def Jeongyeon_record_in_cm : ℕ := Eunseol_record_in_cm + 9
def Jeongyeon_record_in_m : ℚ := Jeongyeon_record_in_cm / 100

theorem Jeongyeon_record_is_1_44_m : Jeongyeon_record_in_m = 1.44 := by
  sorry

end Jeongyeon_record_is_1_44_m_l89_89534


namespace problem_1_problem_2_l89_89200

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 1|

theorem problem_1 : {x : ℝ | f x < 4} = {x : ℝ | -4 / 3 < x ∧ x < 4 / 3} :=
by 
  sorry

theorem problem_2 (x₀ : ℝ) (h : ∀ t : ℝ, f x₀ < |m + t| + |t - m|) : 
  {m : ℝ | ∃ x t, f x < |m + t| + |t - m|} = {m : ℝ | m < -3 / 4 ∨ m > 3 / 4} :=
by 
  sorry

end problem_1_problem_2_l89_89200


namespace shortest_minor_arc_line_equation_l89_89822

noncomputable def pointM : (ℝ × ℝ) := (1, -2)
noncomputable def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

theorem shortest_minor_arc_line_equation :
  (∀ x y : ℝ, (x + 2 * y + 3 = 0) ↔ 
  ((x = 1 ∧ y = -2) ∨ ∃ (k_l : ℝ), (k_l * (2) = -1) ∧ (y + 2 = -k_l * (x - 1)))) :=
sorry

end shortest_minor_arc_line_equation_l89_89822


namespace people_per_column_in_second_scenario_l89_89769

def total_people (num_people_per_column_1 : ℕ) (num_columns_1 : ℕ) : ℕ :=
  num_people_per_column_1 * num_columns_1

def people_per_column_second_scenario (P: ℕ) (num_columns_2 : ℕ) : ℕ :=
  P / num_columns_2

theorem people_per_column_in_second_scenario
  (num_people_per_column_1 : ℕ)
  (num_columns_1 : ℕ)
  (num_columns_2 : ℕ)
  (P : ℕ)
  (h1 : total_people num_people_per_column_1 num_columns_1 = P) :
  people_per_column_second_scenario P num_columns_2 = 48 :=
by
  -- the proof would go here
  sorry

end people_per_column_in_second_scenario_l89_89769


namespace chair_cost_l89_89567

theorem chair_cost :
  (∃ (C : ℝ), 3 * C + 50 + 40 = 130 - 4) → 
  (∃ (C : ℝ), C = 12) :=
by
  sorry

end chair_cost_l89_89567


namespace ring_matching_possible_iff_odd_l89_89929

theorem ring_matching_possible_iff_odd (n : ℕ) (hn : n ≥ 3) :
  (∃ f : ℕ → ℕ, (∀ k : ℕ, k < n → ∃ j : ℕ, j < n ∧ f (j + k) % n = k % n) ↔ Odd n) :=
sorry

end ring_matching_possible_iff_odd_l89_89929


namespace total_GDP_l89_89514

noncomputable def GDP_first_quarter : ℝ := 232
noncomputable def GDP_fourth_quarter : ℝ := 241

theorem total_GDP (x y : ℝ) (h1 : GDP_first_quarter < x)
                  (h2 : x < y) (h3 : y < GDP_fourth_quarter)
                  (h4 : (x + y) / 2 = (GDP_first_quarter + x + y + GDP_fourth_quarter) / 4) :
  GDP_first_quarter + x + y + GDP_fourth_quarter = 946 :=
by
  sorry

end total_GDP_l89_89514


namespace initial_apples_9_l89_89303

def initial_apple_count (picked : ℕ) (remaining : ℕ) : ℕ :=
  picked + remaining

theorem initial_apples_9 (picked : ℕ) (remaining : ℕ) :
  picked = 2 → remaining = 7 → initial_apple_count picked remaining = 9 := by
sorry

end initial_apples_9_l89_89303


namespace average_temperature_second_to_fifth_days_l89_89516

variable (T1 T2 T3 T4 T5 : ℝ)

theorem average_temperature_second_to_fifth_days 
  (h1 : (T1 + T2 + T3 + T4) / 4 = 58)
  (h2 : T1 / T5 = 7 / 8)
  (h3 : T5 = 32) :
  (T2 + T3 + T4 + T5) / 4 = 59 :=
by
  sorry

end average_temperature_second_to_fifth_days_l89_89516


namespace divisible_values_l89_89175

def is_digit (n : ℕ) : Prop :=
  n >= 0 ∧ n <= 9

def N (x y : ℕ) : ℕ :=
  30 * 10^7 + x * 10^6 + 7 * 10^4 + y * 10^3 + 3

def is_divisible_by_37 (n : ℕ) : Prop :=
  n % 37 = 0

theorem divisible_values :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ is_divisible_by_37 (N x y) ∧ ((x, y) = (8, 1) ∨ (x, y) = (4, 4) ∨ (x, y) = (0, 7)) :=
by {
  sorry
}

end divisible_values_l89_89175


namespace largest_sum_distinct_factors_l89_89014

theorem largest_sum_distinct_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
  (h4 : A * B * C = 2023) : A + B + C = 297 :=
sorry

end largest_sum_distinct_factors_l89_89014


namespace find_x_l89_89450

theorem find_x (x : ℝ) (h : 3 * x = (20 - x) + 20) : x = 10 :=
sorry

end find_x_l89_89450


namespace gas_volume_at_25_degrees_l89_89038

theorem gas_volume_at_25_degrees :
  (∀ (T V : ℕ), (T = 40 → V = 30) →
  (∀ (k : ℕ), T = 40 - 5 * k → V = 30 - 6 * k) → 
  (25 = 40 - 5 * 3) → 
  (V = 30 - 6 * 3) → 
  V = 12) := 
by
  sorry

end gas_volume_at_25_degrees_l89_89038


namespace greatest_two_digit_with_product_12_l89_89637

theorem greatest_two_digit_with_product_12 : ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧ (∃ (a b : ℕ), n = 10 * a + b ∧ a * b = 12) ∧ 
  ∀ (m : ℕ), 10 ≤ m ∧ m < 100 ∧ (∃ (c d : ℕ), m = 10 * c + d ∧ c * d = 12) → m ≤ 62 :=
sorry

end greatest_two_digit_with_product_12_l89_89637


namespace actual_cost_of_article_l89_89467

theorem actual_cost_of_article (x : ℝ) (h : 0.80 * x = 620) : x = 775 :=
sorry

end actual_cost_of_article_l89_89467


namespace find_remainder_l89_89804

theorem find_remainder : 
    ∃ (d q r : ℕ), 472 = d * q + r ∧ 427 = d * (q - 5) + r ∧ r = 4 :=
by
  sorry

end find_remainder_l89_89804


namespace perimeter_square_C_l89_89533

theorem perimeter_square_C (pA pB pC : ℕ) (hA : pA = 16) (hB : pB = 32) (hC : pC = (pA + pB) / 2) : pC = 24 := by
  sorry

end perimeter_square_C_l89_89533


namespace part_1_part_2_equality_case_l89_89893

variables {m n : ℝ}

-- Definition of positive real numbers and given condition m > n and n > 1
def conditions_1 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m > n ∧ n > 1

-- Prove that given conditions, m^2 + n > mn + m
theorem part_1 (m n : ℝ) (h : conditions_1 m n) : m^2 + n > m * n + m :=
  by sorry

-- Definition of the condition m + 2n = 1
def conditions_2 (m n : ℝ) : Prop := m > 0 ∧ n > 0 ∧ m + 2 * n = 1

-- Prove that given conditions, (2/m) + (1/n) ≥ 8
theorem part_2 (m n : ℝ) (h : conditions_2 m n) : (2 / m) + (1 / n) ≥ 8 :=
  by sorry

-- Prove that the minimum value is obtained when m = 2n = 1/2
theorem equality_case (m n : ℝ) (h : conditions_2 m n) : 
  (2 / m) + (1 / n) = 8 ↔ m = 1/2 ∧ n = 1/4 :=
  by sorry

end part_1_part_2_equality_case_l89_89893


namespace sixth_term_sequence_l89_89691

theorem sixth_term_sequence (a : ℕ → ℕ) (h₁ : a 0 = 3) (h₂ : ∀ n, a (n + 1) = (a n)^2) : 
  a 5 = 1853020188851841 := 
by {
  sorry
}

end sixth_term_sequence_l89_89691


namespace kho_kho_only_l89_89095

theorem kho_kho_only (K H B total : ℕ) (h1 : K + B = 10) (h2 : B = 5) (h3 : K + H + B = 25) : H = 15 :=
by {
  sorry
}

end kho_kho_only_l89_89095


namespace find_number_l89_89135

theorem find_number (n : ℕ) (h1 : n % 20 = 1) (h2 : n / 20 = 9) : n = 181 := 
by {
  -- proof not required
  sorry
}

end find_number_l89_89135


namespace min_expression_value_l89_89141

theorem min_expression_value (x y z : ℝ) (xyz_eq : x * y * z = 1) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n : ℝ, (∀ x y z : ℝ, x * y * z = 1 → 0 < x → 0 < y → 0 < z → 2 * x^2 + 8 * x * y + 32 * y^2 + 16 * y * z + 8 * z^2 ≥ n)
    ∧ n = 72 :=
sorry

end min_expression_value_l89_89141


namespace cubic_eq_one_complex_solution_l89_89359

theorem cubic_eq_one_complex_solution (k : ℂ) :
  (∃ (x : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0) ∧
  (∀ (x y z : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0 → 8 * y^3 + 12 * y^2 + k * y + 1 = 0
    → 8 * z^3 + 12 * z^2 + k * z + 1 = 0 → x = y ∧ y = z) →
  k = 6 :=
sorry

end cubic_eq_one_complex_solution_l89_89359


namespace original_radius_eq_n_div_3_l89_89255

theorem original_radius_eq_n_div_3 (r n : ℝ) (h : (r + n)^2 = 4 * r^2) : r = n / 3 :=
by
  sorry

end original_radius_eq_n_div_3_l89_89255


namespace find_f_three_l89_89334

variable {α : Type*} [LinearOrderedField α]

def f (a b c x : α) := a * x^5 - b * x^3 + c * x - 3

theorem find_f_three (a b c : α) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by sorry

end find_f_three_l89_89334


namespace geometric_sequence_sum_l89_89268

theorem geometric_sequence_sum 
  (a r : ℝ) 
  (h1 : a + a * r = 8)
  (h2 : a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 = 120) :
  a * (1 + r + r^2 + r^3) = 30 := 
by
  sorry

end geometric_sequence_sum_l89_89268


namespace total_number_of_students_l89_89382

theorem total_number_of_students (girls boys : ℕ) 
  (h_ratio : 8 * girls = 5 * boys) 
  (h_girls : girls = 160) : 
  girls + boys = 416 := 
sorry

end total_number_of_students_l89_89382


namespace find_decreased_value_l89_89722

theorem find_decreased_value (x v : ℝ) (hx : x = 7)
  (h : x - v = 21 * (1 / x)) : v = 4 :=
by
  sorry

end find_decreased_value_l89_89722


namespace calculate_expression_l89_89768

theorem calculate_expression : 
  (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := 
by sorry

end calculate_expression_l89_89768


namespace remainders_equal_l89_89707

theorem remainders_equal (P P' D R k s s' : ℕ) (h1 : P > P') 
  (h2 : P % D = 2 * R) (h3 : P' % D = R) (h4 : R < D) :
  (k * (P + P')) % D = s → (k * (2 * R + R)) % D = s' → s = s' :=
by
  sorry

end remainders_equal_l89_89707


namespace school_club_profit_l89_89890

theorem school_club_profit :
  let pencils := 1200
  let buy_rate := 4 / 3 -- pencils per dollar
  let sell_rate := 5 / 4 -- pencils per dollar
  let cost_per_pencil := 3 / 4 -- dollars per pencil
  let sell_per_pencil := 4 / 5 -- dollars per pencil
  let cost := pencils * cost_per_pencil
  let revenue := pencils * sell_per_pencil
  let profit := revenue - cost
  profit = 60 := 
by
  sorry

end school_club_profit_l89_89890


namespace younger_brother_age_l89_89242

variable (x y : ℕ)

-- Conditions
axiom sum_of_ages : x + y = 46
axiom younger_is_third_plus_ten : y = x / 3 + 10

theorem younger_brother_age : y = 19 := 
by
  sorry

end younger_brother_age_l89_89242


namespace chameleons_cannot_all_turn_to_single_color_l89_89875

theorem chameleons_cannot_all_turn_to_single_color
  (W : ℕ) (B : ℕ)
  (hW : W = 20)
  (hB : B = 25)
  (h_interaction: ∀ t : ℕ, ∃ W' B' : ℕ,
    W' + B' = W + B ∧
    (W - B) % 3 = (W' - B') % 3) :
  ∀ t : ℕ, (W - B) % 3 ≠ 0 :=
by
  sorry

end chameleons_cannot_all_turn_to_single_color_l89_89875


namespace integer_divisibility_l89_89151

open Nat

theorem integer_divisibility {a b : ℕ} :
  (2 * b^2 + 1) ∣ (a^3 + 1) ↔ a = 2 * b^2 + 1 := sorry

end integer_divisibility_l89_89151


namespace rex_has_399_cards_left_l89_89424

def Nicole_cards := 700

def Cindy_cards := 3 * Nicole_cards + (40 / 100) * (3 * Nicole_cards)
def Tim_cards := (4 / 5) * Cindy_cards
def combined_total := Nicole_cards + Cindy_cards + Tim_cards
def Rex_and_Joe_cards := (60 / 100) * combined_total

def cards_per_person := Nat.floor (Rex_and_Joe_cards / 9)

theorem rex_has_399_cards_left : cards_per_person = 399 := by
  sorry

end rex_has_399_cards_left_l89_89424


namespace bus_driver_total_compensation_l89_89756

theorem bus_driver_total_compensation :
  let regular_rate := 16
  let regular_hours := 40
  let overtime_hours := 60 - regular_hours
  let overtime_rate := regular_rate + 0.75 * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1200 := by
  sorry

end bus_driver_total_compensation_l89_89756


namespace mike_peaches_eq_120_l89_89225

def original_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def total_peaches (orig : ℝ) (picked : ℝ) : ℝ := orig + picked

theorem mike_peaches_eq_120 : total_peaches original_peaches picked_peaches = 120.0 := 
by
  sorry

end mike_peaches_eq_120_l89_89225


namespace nailcutter_sound_count_l89_89675

-- Definitions based on conditions
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3
def sound_per_nail : ℕ := 1

-- The statement to prove 
theorem nailcutter_sound_count :
  (nails_per_person * number_of_customers * sound_per_nail) = 60 := by
  sorry

end nailcutter_sound_count_l89_89675


namespace compute_paths_in_grid_l89_89752

def grid : List (List Char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'C', 'O', 'C', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'C', 'O', 'M', 'O', 'C', ' ', ' ', ' '],
  [' ', ' ', ' ', 'C', 'O', 'M', 'P', 'M', 'O', 'C', ' ', ' '],
  [' ', ' ', 'C', 'O', 'M', 'P', 'U', 'P', 'M', 'O', 'C', ' '],
  [' ', 'C', 'O', 'M', 'P', 'U', 'T', 'U', 'P', 'M', 'O', 'C'],
  ['C', 'O', 'M', 'P', 'U', 'T', 'E', 'T', 'U', 'P', 'M', 'O', 'C']
]

def is_valid_path (path : List (Nat × Nat)) : Bool :=
  -- This function checks if a given path is valid according to the problem's grid and rules.
  sorry

def count_paths_from_C_to_E (grid: List (List Char)) : Nat :=
  -- This function would count the number of valid paths from a 'C' in the leftmost column to an 'E' in the rightmost column.
  sorry

theorem compute_paths_in_grid : count_paths_from_C_to_E grid = 64 :=
by
  sorry

end compute_paths_in_grid_l89_89752


namespace average_of_11_numbers_l89_89097

theorem average_of_11_numbers (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ)
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = 58)
  (h2 : (a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 6 = 65)
  (h3 : a₆ = 78) : 
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 11 = 60 := 
by 
  sorry 

end average_of_11_numbers_l89_89097


namespace fifteenth_battery_replacement_month_l89_89308

theorem fifteenth_battery_replacement_month :
  (98 % 12) + 1 = 4 :=
by
  sorry

end fifteenth_battery_replacement_month_l89_89308


namespace can_cut_into_equal_parts_l89_89551

-- We assume the existence of a shape S and some grid G along with a function cut
-- that cuts the shape S along grid G lines and returns two parts.
noncomputable def Shape := Type
noncomputable def Grid := Type
noncomputable def cut (S : Shape) (G : Grid) : Shape × Shape := sorry

-- We assume a function superimpose that checks whether two shapes can be superimposed
noncomputable def superimpose (S1 S2 : Shape) : Prop := sorry

-- Assume the given shape S and grid G
variable (S : Shape) (G : Grid)

-- The question rewritten as a Lean statement
theorem can_cut_into_equal_parts : ∃ (S₁ S₂ : Shape), cut S G = (S₁, S₂) ∧ superimpose S₁ S₂ := sorry

end can_cut_into_equal_parts_l89_89551


namespace triangle_ratio_l89_89446

theorem triangle_ratio (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let p := 12;
  let q := 8;
  let segment_length := L / p;
  let segment_width := W / q;
  let area_X := (segment_length * segment_width) / 2;
  let area_rectangle := L * W;
  (area_X / area_rectangle) = (1 / 192) :=
by 
  sorry

end triangle_ratio_l89_89446


namespace next_month_has_5_Wednesdays_l89_89215

-- The current month characteristics
def current_month_has_5_Saturdays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_5_Sundays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 5
def current_month_has_4_Mondays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def current_month_has_4_Fridays : Prop := ∃ month : ℕ, month = 30 ∧ ∃ day : ℕ, day = 4
def month_ends_on_Sunday : Prop := ∃ day : ℕ, day = 30 ∧ day % 7 = 0

-- Prove next month has 5 Wednesdays
theorem next_month_has_5_Wednesdays 
  (h1 : current_month_has_5_Saturdays) 
  (h2 : current_month_has_5_Sundays)
  (h3 : current_month_has_4_Mondays)
  (h4 : current_month_has_4_Fridays)
  (h5 : month_ends_on_Sunday) :
  ∃ month : ℕ, month = 31 ∧ ∃ day : ℕ, day = 5 := 
sorry

end next_month_has_5_Wednesdays_l89_89215


namespace line_intersects_y_axis_at_point_l89_89168

theorem line_intersects_y_axis_at_point :
  let x1 := 3
  let y1 := 20
  let x2 := -7
  let y2 := 2

  -- line equation from 2 points: y - y1 = m * (x - x1)
  -- slope m = (y2 - y1) / (x2 - x1)
  -- y-intercept when x = 0:
  
  (0, 14.6) ∈ { p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ 
    m = (y2 - y1) / (x2 - x1) ∧ 
    b = y1 - m * x1 }
  :=
  sorry

end line_intersects_y_axis_at_point_l89_89168


namespace net_income_difference_l89_89981

theorem net_income_difference
    (terry_daily_income : ℝ := 24) (terry_daily_hours : ℝ := 6) (terry_days : ℕ := 7)
    (jordan_daily_income : ℝ := 30) (jordan_daily_hours : ℝ := 8) (jordan_days : ℕ := 6)
    (standard_week_hours : ℝ := 40) (overtime_rate_multiplier : ℝ := 1.5)
    (terry_tax_rate : ℝ := 0.12) (jordan_tax_rate : ℝ := 0.15) :
    jordan_daily_income * jordan_days - jordan_daily_income * jordan_days * jordan_tax_rate 
      + jordan_daily_income * jordan_days * jordan_daily_hours * (overtime_rate_multiplier - 1) * jordan_tax_rate
    - (terry_daily_income * terry_days - terry_daily_income * terry_days * terry_tax_rate 
      + terry_daily_income * terry_days * terry_daily_hours * (overtime_rate_multiplier - 1) * terry_tax_rate) 
      = 32.85 := 
sorry

end net_income_difference_l89_89981


namespace custom_op_diff_l89_89356

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_diff : custom_op 8 5 - custom_op 5 8 = -12 :=
by
  sorry

end custom_op_diff_l89_89356


namespace comic_book_arrangement_l89_89452

theorem comic_book_arrangement :
  let spiderman_books := 7
  let archie_books := 6
  let garfield_books := 5
  let groups := 3
  Nat.factorial spiderman_books * Nat.factorial archie_books * Nat.factorial garfield_books * Nat.factorial groups = 248005440 :=
by
  sorry

end comic_book_arrangement_l89_89452


namespace ordering_abc_l89_89677

noncomputable def a : ℝ := Real.sqrt 1.01
noncomputable def b : ℝ := Real.exp 0.01 / 1.01
noncomputable def c : ℝ := Real.log (1.01 * Real.exp 1)

theorem ordering_abc : b < a ∧ a < c := by
  -- Proof of the theorem goes here
  sorry

end ordering_abc_l89_89677


namespace Xiaoxi_has_largest_final_answer_l89_89917

def Laura_final : ℕ := 8 - 2 * 3 + 3
def Navin_final : ℕ := (8 * 3) - 2 + 3
def Xiaoxi_final : ℕ := (8 - 2 + 3) * 3

theorem Xiaoxi_has_largest_final_answer : 
  Xiaoxi_final > Laura_final ∧ Xiaoxi_final > Navin_final :=
by
  unfold Laura_final Navin_final Xiaoxi_final
  -- Proof steps would go here, but we skip them as per instructions
  sorry

end Xiaoxi_has_largest_final_answer_l89_89917


namespace triangle_construction_conditions_l89_89459

open Classical

noncomputable def construct_triangle (m_a m_b s_c : ℝ) : Prop :=
  m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c

theorem triangle_construction_conditions (m_a m_b s_c : ℝ) :
  construct_triangle m_a m_b s_c ↔ (m_a ≤ 2 * s_c ∧ m_b ≤ 2 * s_c) :=
by
  sorry

end triangle_construction_conditions_l89_89459


namespace parabola_tangent_to_hyperbola_l89_89954

theorem parabola_tangent_to_hyperbola (m : ℝ) :
  (∀ x y : ℝ, y = x^2 + 4 → y^2 - m * x^2 = 4) ↔ m = 8 := 
sorry

end parabola_tangent_to_hyperbola_l89_89954


namespace find_initial_population_l89_89187

-- Define the initial population, conditions and the final population
variable (P : ℕ)

noncomputable def initial_population (P : ℕ) :=
  (0.85 * (0.92 * P) : ℝ) = 3553

theorem find_initial_population (P : ℕ) (h : initial_population P) : P = 4546 := sorry

end find_initial_population_l89_89187


namespace ellipse_slope_ratio_l89_89512

theorem ellipse_slope_ratio (a b : ℝ) (k1 k2 : ℝ) 
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a > 2)
  (h4 : k2 = k1 * (a^2 + 5) / (a^2 - 1)) : 
  1 < (k2 / k1) ∧ (k2 / k1) < 3 :=
by
  sorry

end ellipse_slope_ratio_l89_89512


namespace parabola_opening_downwards_l89_89612

theorem parabola_opening_downwards (a : ℝ) :
  (∀ x, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) → -1 < a ∧ a < 0 :=
by 
  intro h
  sorry

end parabola_opening_downwards_l89_89612


namespace expression_evaluates_to_47_l89_89329

theorem expression_evaluates_to_47 : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by 
  sorry

end expression_evaluates_to_47_l89_89329


namespace point_outside_circle_l89_89950

theorem point_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) : a^2 + b^2 > 1 :=
sorry

end point_outside_circle_l89_89950


namespace find_other_number_l89_89766

theorem find_other_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 14) (lcm_ab : Nat.lcm a b = 396) (h : a = 36) : b = 154 :=
by
  sorry

end find_other_number_l89_89766


namespace base_number_is_five_l89_89102

variable (a x y : Real)

theorem base_number_is_five (h1 : xy = 1) (h2 : (a ^ (x + y) ^ 2) / (a ^ (x - y) ^ 2) = 625) : a = 5 := 
sorry

end base_number_is_five_l89_89102


namespace isosceles_right_triangle_area_l89_89741

/--
Given an isosceles right triangle with a hypotenuse of 6√2 units, prove that the area
of this triangle is 18 square units.
-/
theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (hyp : h = 6 * Real.sqrt 2) 
  (isosceles : h = l * Real.sqrt 2) : 
  (1/2) * l^2 = 18 :=
by
  sorry

end isosceles_right_triangle_area_l89_89741


namespace new_avg_weight_l89_89304

theorem new_avg_weight (A B C D E : ℝ) (h1 : (A + B + C) / 3 = 84) (h2 : A = 78) 
(h3 : (B + C + D + E) / 4 = 79) (h4 : E = D + 6) : 
(A + B + C + D) / 4 = 80 :=
by
  sorry

end new_avg_weight_l89_89304


namespace divisibility_of_product_l89_89118

theorem divisibility_of_product (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a ∣ b^3) (h2 : b ∣ c^3) (h3 : c ∣ a^3) : abc ∣ (a + b + c) ^ 13 := by
  sorry

end divisibility_of_product_l89_89118


namespace smaller_successive_number_l89_89853

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 9506) : n = 97 :=
sorry

end smaller_successive_number_l89_89853


namespace arithmetic_progression_complete_iff_divides_l89_89601

-- Definitions from the conditions
def complete_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, s n ≠ 0) ∧ (∀ m : ℤ, m ≠ 0 → ∃ n : ℕ, s n = m)

-- Arithmetic progression definition
def arithmetic_progression (a r : ℤ) (n : ℕ) : ℤ :=
  a + n * r

-- Lean theorem statement
theorem arithmetic_progression_complete_iff_divides (a r : ℤ) :
  (complete_sequence (arithmetic_progression a r)) ↔ (r ∣ a) := by
  sorry

end arithmetic_progression_complete_iff_divides_l89_89601


namespace average_percentage_of_kernels_popped_l89_89433

theorem average_percentage_of_kernels_popped :
  let bag1_popped := 60
  let bag1_total := 75
  let bag2_popped := 42
  let bag2_total := 50
  let bag3_popped := 82
  let bag3_total := 100
  let percentage (popped total : ℕ) := (popped : ℚ) / total * 100
  let p1 := percentage bag1_popped bag1_total
  let p2 := percentage bag2_popped bag2_total
  let p3 := percentage bag3_popped bag3_total
  let avg := (p1 + p2 + p3) / 3
  avg = 82 :=
by
  sorry

end average_percentage_of_kernels_popped_l89_89433


namespace chimney_bricks_l89_89940

theorem chimney_bricks (x : ℕ) 
  (h1 : Brenda_rate = x / 8) 
  (h2 : Brandon_rate = x / 12) 
  (h3 : Brian_rate = x / 16) 
  (h4 : effective_combined_rate = (Brenda_rate + Brandon_rate + Brian_rate) - 15) 
  (h5 : total_time = 4) :
  (4 * effective_combined_rate) = x := 
  sorry

end chimney_bricks_l89_89940


namespace find_x_l89_89486

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x)
    (h3 : 1 / a + 1 / b = 1) : x = 6 :=
sorry

end find_x_l89_89486


namespace correct_operation_l89_89232

theorem correct_operation (a b : ℝ) :
  (3 * a^2 - a^2 ≠ 3) ∧
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((-3 * a * b^2)^2 ≠ -6 * a^2 * b^4) →
  a^3 / a^2 = a :=
by
sorry

end correct_operation_l89_89232


namespace no_solution_m1_no_solution_m2_solution_m3_l89_89018

-- Problem 1: No positive integer solutions for m = 1
theorem no_solution_m1 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ x * y * z := sorry

-- Problem 2: No positive integer solutions for m = 2
theorem no_solution_m2 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ 2 * x * y * z := sorry

-- Problem 3: Only solutions for m = 3 are x = y = z = k for some k
theorem solution_m3 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z := sorry

end no_solution_m1_no_solution_m2_solution_m3_l89_89018


namespace a5_value_S8_value_l89_89336

-- Definitions based on the conditions
def seq (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else 2 * seq (n - 1)

noncomputable def S (n : ℕ) : ℕ :=
(1 - 2^n) / (1 - 2)

-- Proof statements
theorem a5_value : seq 5 = 16 := sorry

theorem S8_value : S 8 = 255 := sorry

end a5_value_S8_value_l89_89336


namespace smallest_value_3a_2_l89_89252

theorem smallest_value_3a_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 2 = - (5 / 2) := sorry

end smallest_value_3a_2_l89_89252


namespace collectively_behind_l89_89213

noncomputable def sleep_hours_behind (weeknights weekend nights_ideal: ℕ) : ℕ :=
  let total_sleep := (weeknights * 5) + (weekend * 2)
  let ideal_sleep := nights_ideal * 7
  ideal_sleep - total_sleep

def tom_weeknight := 5
def tom_weekend := 6

def jane_weeknight := 7
def jane_weekend := 9

def mark_weeknight := 6
def mark_weekend := 7

def ideal_night := 8

theorem collectively_behind :
  sleep_hours_behind tom_weeknight tom_weekend ideal_night +
  sleep_hours_behind jane_weeknight jane_weekend ideal_night +
  sleep_hours_behind mark_weeknight mark_weekend ideal_night = 34 :=
by
  sorry

end collectively_behind_l89_89213


namespace cheryl_bill_cost_correct_l89_89760

def cheryl_electricity_bill_cost : Prop :=
  ∃ (E : ℝ), 
    (E + 400) + 0.20 * (E + 400) = 1440 ∧ 
    E = 800

theorem cheryl_bill_cost_correct : cheryl_electricity_bill_cost :=
by
  sorry

end cheryl_bill_cost_correct_l89_89760


namespace depth_of_channel_l89_89379

theorem depth_of_channel (a b A : ℝ) (h : ℝ) (h_area : A = (1 / 2) * (a + b) * h)
  (ha : a = 12) (hb : b = 6) (hA : A = 630) : h = 70 :=
by
  sorry

end depth_of_channel_l89_89379


namespace sufficient_food_supply_l89_89193

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l89_89193


namespace ratio_when_volume_maximized_l89_89931

-- Definitions based on conditions
def cylinder_perimeter := 24

-- Definition of properties derived from maximizing the volume
def max_volume_height := 4

def max_volume_circumference := 12 - max_volume_height

-- The ratio of the circumference of the cylinder's base to its height when the volume is maximized
def max_volume_ratio := max_volume_circumference / max_volume_height

-- The theorem to be proved
theorem ratio_when_volume_maximized :
  max_volume_ratio = 2 :=
by sorry

end ratio_when_volume_maximized_l89_89931


namespace simplify_expression_l89_89987

theorem simplify_expression (s : ℤ) : 120 * s - 32 * s = 88 * s := by
  sorry

end simplify_expression_l89_89987


namespace triangle_area_45_45_90_l89_89664

/--
A right triangle has one angle of 45 degrees, and its hypotenuse measures 10√2 inches.
Prove that the area of the triangle is 50 square inches.
-/
theorem triangle_area_45_45_90 {x : ℝ} (h1 : 0 < x) (h2 : x * Real.sqrt 2 = 10 * Real.sqrt 2) : 
  (1 / 2) * x * x = 50 :=
sorry

end triangle_area_45_45_90_l89_89664


namespace max_a_plus_ab_plus_abc_l89_89011

noncomputable def f (a b c: ℝ) := a + a * b + a * b * c

theorem max_a_plus_ab_plus_abc (a b c : ℝ) (h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h2 : a + b + c = 1) :
  ∃ x, (f a b c ≤ x) ∧ (∀ y, f a b c ≤ y → y = 1) :=
sorry

end max_a_plus_ab_plus_abc_l89_89011


namespace sum_of_products_two_at_a_time_l89_89812

theorem sum_of_products_two_at_a_time (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a + b + c = 23) : 
  ab + bc + ac = 131 :=
by
  sorry

end sum_of_products_two_at_a_time_l89_89812


namespace speed_conversion_l89_89189

def speed_mps : ℝ := 10.0008
def conversion_factor : ℝ := 3.6

theorem speed_conversion : speed_mps * conversion_factor = 36.003 :=
by
  sorry

end speed_conversion_l89_89189


namespace initial_percentage_increase_l89_89598

theorem initial_percentage_increase 
  (W R : ℝ) 
  (P : ℝ)
  (h1 : R = W * (1 + P/100)) 
  (h2 : R * 0.70 = W * 1.18999999999999993) :
  P = 70 :=
by sorry

end initial_percentage_increase_l89_89598


namespace minimum_excellence_percentage_l89_89891

theorem minimum_excellence_percentage (n : ℕ) (h : n = 100)
    (m c b : ℕ) 
    (h_math : m = 70)
    (h_chinese : c = 75) 
    (h_min_both : b = c - (n - m))
    (h_percent : b = 45) :
    b = 45 :=
    sorry

end minimum_excellence_percentage_l89_89891


namespace sufficient_but_not_necessary_condition_l89_89883

-- Define a sequence of positive terms
def is_positive_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∀ i, 0 < seq i

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∃ q > 0, q ≠ 1 ∧ ∀ i j, i < j → seq j = (q ^ (j - i : ℤ)) * seq i

-- State the theorem
theorem sufficient_but_not_necessary_condition (seq : Fin 8 → ℝ) (h_pos : is_positive_sequence seq) :
  ¬is_geometric_sequence seq → seq 0 + seq 7 < seq 3 + seq 4 ∧ 
  (seq 0 + seq 7 < seq 3 + seq 4 → ¬is_geometric_sequence seq) ∧
  (¬is_geometric_sequence seq → ¬(seq 0 + seq 7 < seq 3 + seq 4) -> ¬ is_geometric_sequence seq) :=
sorry

end sufficient_but_not_necessary_condition_l89_89883


namespace price_increase_percentage_l89_89751

theorem price_increase_percentage (c : ℝ) (r : ℝ) (p : ℝ) 
  (h1 : r = 1.4 * c) 
  (h2 : p = 1.15 * r) : 
  (p - c) / c * 100 = 61 := 
sorry

end price_increase_percentage_l89_89751


namespace shots_and_hits_l89_89566

theorem shots_and_hits (n k : ℕ) (h₀ : 10 < n) (h₁ : n < 20) (h₂ : 5 * k = 3 * (n - k)) : (n = 16) ∧ (k = 6) :=
by {
  -- We state the result that we wish to prove
  sorry
}

end shots_and_hits_l89_89566


namespace calculate_wholesale_price_l89_89475

noncomputable def retail_price : ℝ := 108

noncomputable def selling_price (retail_price : ℝ) : ℝ := retail_price * 0.90

noncomputable def selling_price_alt (wholesale_price : ℝ) : ℝ := wholesale_price * 1.20

theorem calculate_wholesale_price (W : ℝ) (R : ℝ) (SP : ℝ)
  (hR : R = 108)
  (hSP1 : SP = selling_price R)
  (hSP2 : SP = selling_price_alt W) : W = 81 :=
by
  -- Proof omitted
  sorry

end calculate_wholesale_price_l89_89475


namespace janet_time_to_home_l89_89553

-- Janet's initial and final positions
def initial_position : ℕ × ℕ := (0, 0) -- (x, y)
def north_blocks : ℕ := 3
def west_multiplier : ℕ := 7
def south_blocks : ℕ := 8
def east_multiplier : ℕ := 2
def speed_blocks_per_minute : ℕ := 2

def west_blocks : ℕ := west_multiplier * north_blocks
def east_blocks : ℕ := east_multiplier * south_blocks

-- Net movement calculations
def net_south_blocks : ℕ := south_blocks - north_blocks
def net_west_blocks : ℕ := west_blocks - east_blocks

-- Time calculation
def total_blocks_to_home : ℕ := net_south_blocks + net_west_blocks
def time_to_home : ℕ := total_blocks_to_home / speed_blocks_per_minute

theorem janet_time_to_home : time_to_home = 5 := by
  -- Proof goes here
  sorry

end janet_time_to_home_l89_89553


namespace find_dinner_bill_l89_89574

noncomputable def total_dinner_bill (B : ℝ) (silas_share : ℝ) (remaining_friends_pay : ℝ) (each_friend_pays : ℝ) :=
  silas_share = (1/2) * B ∧
  remaining_friends_pay = (1/2) * B + 0.10 * B ∧
  each_friend_pays = remaining_friends_pay / 5 ∧
  each_friend_pays = 18

theorem find_dinner_bill : ∃ B : ℝ, total_dinner_bill B ((1/2) * B) ((1/2) * B + 0.10 * B) (18) → B = 150 :=
by
  sorry

end find_dinner_bill_l89_89574


namespace no_valid_arrangement_l89_89578

open Nat

theorem no_valid_arrangement :
  ¬ ∃ (f : Fin 30 → ℕ), 
    (∀ (i : Fin 30), 1 ≤ f i ∧ f i ≤ 30) ∧ 
    (∀ (i : Fin 30), ∃ n : ℕ, (f i + f (i + 1) % 30) = n^2) ∧ 
    (∀ i1 i2, i1 ≠ i2 → f i1 ≠ f i2) :=
  sorry

end no_valid_arrangement_l89_89578


namespace cary_needs_6_weekends_l89_89504

variable (shoe_cost : ℕ)
variable (current_savings : ℕ)
variable (earn_per_lawn : ℕ)
variable (lawns_per_weekend : ℕ)
variable (w : ℕ)

theorem cary_needs_6_weekends
    (h1 : shoe_cost = 120)
    (h2 : current_savings = 30)
    (h3 : earn_per_lawn = 5)
    (h4 : lawns_per_weekend = 3)
    (h5 : w * (earn_per_lawn * lawns_per_weekend) = shoe_cost - current_savings) :
    w = 6 :=
by sorry

end cary_needs_6_weekends_l89_89504


namespace triangle_is_right_triangle_l89_89034

theorem triangle_is_right_triangle (A B C : ℝ) (hC_eq_A_plus_B : C = A + B) (h_angle_sum : A + B + C = 180) : C = 90 :=
by
  sorry

end triangle_is_right_triangle_l89_89034


namespace peaches_picked_l89_89006

variable (o t : ℕ)
variable (p : ℕ)

theorem peaches_picked : (o = 34) → (t = 86) → (t = o + p) → p = 52 :=
by
  intros ho ht htot
  rw [ho, ht] at htot
  sorry

end peaches_picked_l89_89006


namespace max_g_of_15_l89_89314

noncomputable def g (x : ℝ) : ℝ := x^3  -- Assume the polynomial g(x) = x^3 based on the maximum value found.

theorem max_g_of_15 (g : ℝ → ℝ) (h_coeff : ∀ x, 0 ≤ g x)
  (h3 : g 3 = 3) (h27 : g 27 = 1701) : g 15 = 3375 :=
by
  -- According to the problem's constraint and identified solution,
  -- here is the statement asserting that the maximum value of g(15) is 3375
  sorry

end max_g_of_15_l89_89314


namespace calculate_L_l89_89156

theorem calculate_L (T H K : ℝ) (hT : T = 2 * Real.sqrt 5) (hH : H = 10) (hK : K = 2) :
  L = 100 :=
by
  let L := 50 * T^4 / (H^2 * K)
  have : T = 2 * Real.sqrt 5 := hT
  have : H = 10 := hH
  have : K = 2 := hK
  sorry

end calculate_L_l89_89156


namespace initial_quantity_of_milk_in_A_l89_89129

theorem initial_quantity_of_milk_in_A (A : ℝ) 
  (h1: ∃ C B: ℝ, B = 0.375 * A ∧ C = 0.625 * A) 
  (h2: ∃ M: ℝ, M = 0.375 * A + 154 ∧ M = 0.625 * A - 154) 
  : A = 1232 :=
by
  -- you can use sorry to skip the proof
  sorry

end initial_quantity_of_milk_in_A_l89_89129


namespace slope_of_line_l89_89352

theorem slope_of_line 
  (p l t : ℝ) (p_pos : p > 0)
  (h_parabola : (2:ℝ)*p = 4) -- Since the parabola passes through M(l,2)
  (h_incircle_center : ∃ (k m : ℝ), (k + 1 = 0) ∧ (k^2 - k - 2 = 0)) :
  ∃ (k : ℝ), k = -1 :=
by {
  sorry
}

end slope_of_line_l89_89352


namespace percent_problem_l89_89876

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_problem_l89_89876


namespace fraction_to_decimal_l89_89626

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end fraction_to_decimal_l89_89626


namespace factor_sum_l89_89343

theorem factor_sum :
  ∃ d e f : ℤ, (∀ x : ℤ, x^2 + 11 * x + 24 = (x + d) * (x + e)) ∧
              (∀ x : ℤ, x^2 + 9 * x - 36 = (x + e) * (x - f)) ∧
              d + e + f = 14 := by
  sorry

end factor_sum_l89_89343


namespace find_m_n_l89_89214

theorem find_m_n (m n x1 x2 : ℕ) (hm : 0 < m) (hn : 0 < n) (hx1 : 0 < x1) (hx2 : 0 < x2) 
  (h_eq : x1 * x2 = m + n) (h_sum : x1 + x2 = m * n) :
  (m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 2) ∨ (m = 1 ∧ n = 5) ∨ (m = 5 ∧ n = 1) := 
sorry

end find_m_n_l89_89214


namespace computation_l89_89869

theorem computation :
  ( ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * 
    ( (7^3 - 1) / (7^3 + 1) ) * ( (8^3 - 1) / (8^3 + 1) ) 
  ) = (73 / 312) :=
by
  sorry

end computation_l89_89869


namespace triangle_area_l89_89792

def point := ℝ × ℝ

def A : point := (2, -3)
def B : point := (8, 1)
def C : point := (2, 3)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area : area_triangle A B C = 18 :=
  sorry

end triangle_area_l89_89792


namespace max_value_of_z_l89_89871

theorem max_value_of_z : ∀ x : ℝ, (x^2 - 14 * x + 10 ≤ 0 - 39) :=
by
  sorry

end max_value_of_z_l89_89871


namespace min_value_of_squares_l89_89821

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end min_value_of_squares_l89_89821


namespace blue_shoes_in_warehouse_l89_89672

theorem blue_shoes_in_warehouse (total blue purple green : ℕ) (h1 : total = 1250) (h2 : green = purple) (h3 : purple = 355) :
    blue = total - (green + purple) := by
  sorry

end blue_shoes_in_warehouse_l89_89672


namespace actual_distance_in_km_l89_89554

-- Given conditions
def scale_factor : ℕ := 200000
def map_distance_cm : ℚ := 3.5

-- Proof goal: the actual distance in kilometers
theorem actual_distance_in_km : (map_distance_cm * scale_factor) / 100000 = 7 := 
by
  sorry

end actual_distance_in_km_l89_89554


namespace Pythagorean_triple_example_1_Pythagorean_triple_example_2_l89_89385

theorem Pythagorean_triple_example_1 : 3^2 + 4^2 = 5^2 := by
  sorry

theorem Pythagorean_triple_example_2 : 5^2 + 12^2 = 13^2 := by
  sorry

end Pythagorean_triple_example_1_Pythagorean_triple_example_2_l89_89385


namespace coefficient_of_x4_l89_89772

theorem coefficient_of_x4 (a : ℝ) (h : 15 * a^4 = 240) : a = 2 ∨ a = -2 := 
sorry

end coefficient_of_x4_l89_89772


namespace marathon_distance_l89_89184

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (extra_yards_per_marathon : ℕ) (yards_per_mile : ℕ) (total_miles_run : ℕ) (total_yards_run : ℕ) (remaining_yards : ℕ) :
  marathons = 15 →
  miles_per_marathon = 26 →
  extra_yards_per_marathon = 385 →
  yards_per_mile = 1760 →
  total_miles_run = (marathons * miles_per_marathon + extra_yards_per_marathon * marathons / yards_per_mile) →
  total_yards_run = (marathons * (miles_per_marathon * yards_per_mile + extra_yards_per_marathon)) →
  remaining_yards = total_yards_run - (total_miles_run * yards_per_mile) →
  0 ≤ remaining_yards ∧ remaining_yards < yards_per_mile →
  remaining_yards = 1500 :=
by
  intros
  sorry

end marathon_distance_l89_89184


namespace equiv_or_neg_equiv_l89_89877

theorem equiv_or_neg_equiv (x y : ℤ) (h : (x^2) % 239 = (y^2) % 239) :
  (x % 239 = y % 239) ∨ (x % 239 = (-y) % 239) :=
by
  sorry

end equiv_or_neg_equiv_l89_89877


namespace point_inside_circle_l89_89545

theorem point_inside_circle (a : ℝ) :
  ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end point_inside_circle_l89_89545


namespace fabric_woven_in_30_days_l89_89676

theorem fabric_woven_in_30_days :
  let a1 := 5
  let d := 16 / 29
  (30 * a1 + (30 * (30 - 1) / 2) * d) = 390 :=
by
  let a1 := 5
  let d := 16 / 29
  sorry

end fabric_woven_in_30_days_l89_89676


namespace solve_equation_l89_89658

theorem solve_equation :
  ∀ x : ℝ, (1 / 7 + 7 / x = 15 / x + 1 / 15) → x = 105 :=
by
  intros x h
  sorry

end solve_equation_l89_89658


namespace Monica_class_ratio_l89_89197

theorem Monica_class_ratio : 
  (20 + 25 + 25 + x + 28 + 28 = 136) → 
  (x = 10) → 
  (x / 20 = 1 / 2) :=
by 
  intros h h_x
  sorry

end Monica_class_ratio_l89_89197


namespace scientific_notation_26_billion_l89_89279

theorem scientific_notation_26_billion :
  ∃ (a : ℝ) (n : ℤ), (26 * 10^8 : ℝ) = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 2.6 ∧ n = 9 :=
sorry

end scientific_notation_26_billion_l89_89279


namespace pentagonal_faces_count_l89_89340

theorem pentagonal_faces_count (x y : ℕ) (h : (5 * x + 6 * y) % 6 = 0) (h1 : ∃ v e f, v - e + f = 2 ∧ f = x + y ∧ e = (5 * x + 6 * y) / 2 ∧ v = (5 * x + 6 * y) / 3 ∧ (5 * x + 6 * y) / 3 * 3 = 5 * x + 6 * y) : 
  x = 12 :=
sorry

end pentagonal_faces_count_l89_89340


namespace sum_of_first_19_terms_l89_89458

noncomputable def a_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a1 + a_n a1 d n)

theorem sum_of_first_19_terms (a1 d : ℝ) (h : a1 + 9 * d = 1) : S_n a1 d 19 = 19 := by
  sorry

end sum_of_first_19_terms_l89_89458


namespace roots_of_f_l89_89400

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (a * x)

theorem roots_of_f (a : ℝ) :
  (a < 0 → ¬∃ x : ℝ, f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a = Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :=
sorry

end roots_of_f_l89_89400


namespace sin_315_eq_neg_sqrt2_over_2_l89_89776

theorem sin_315_eq_neg_sqrt2_over_2 :
  Real.sin (315 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_315_eq_neg_sqrt2_over_2_l89_89776


namespace remainder_of_72nd_integers_div_by_8_is_5_l89_89419

theorem remainder_of_72nd_integers_div_by_8_is_5 (s : Set ℤ) (h₁ : ∀ x ∈ s, ∃ k : ℤ, x = 8 * k + r) 
  (h₂ : 573 ∈ (s : Set ℤ)) : 
  ∃ (r : ℤ), r = 5 :=
by
  sorry

end remainder_of_72nd_integers_div_by_8_is_5_l89_89419


namespace compute_combination_product_l89_89573

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem compute_combination_product :
  combination 10 3 * combination 8 3 = 6720 :=
by
  sorry

end compute_combination_product_l89_89573


namespace opposite_reciprocal_abs_value_l89_89346

theorem opposite_reciprocal_abs_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : abs m = 3) : 
  (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 := by 
  sorry

end opposite_reciprocal_abs_value_l89_89346


namespace total_sticks_needed_l89_89747

theorem total_sticks_needed (simon_sticks gerry_sticks micky_sticks darryl_sticks : ℕ):
  simon_sticks = 36 →
  gerry_sticks = (2 * simon_sticks) / 3 →
  micky_sticks = simon_sticks + gerry_sticks + 9 →
  darryl_sticks = simon_sticks + gerry_sticks + micky_sticks + 1 →
  simon_sticks + gerry_sticks + micky_sticks + darryl_sticks = 259 :=
by
  intros h_simon h_gerry h_micky h_darryl
  rw [h_simon, h_gerry, h_micky, h_darryl]
  norm_num
  sorry

end total_sticks_needed_l89_89747


namespace compute_fraction_l89_89143

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem compute_fraction : (f (g (f 1))) / (g (f (g 1))) = 6801 / 281 := 
by 
  sorry

end compute_fraction_l89_89143


namespace difference_between_number_and_its_3_5_l89_89070

theorem difference_between_number_and_its_3_5 (x : ℕ) (h : x = 155) :
  x - (3 / 5 : ℚ) * x = 62 := by
  sorry

end difference_between_number_and_its_3_5_l89_89070


namespace hyperbola_equation_l89_89411

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
                           (h3 : b = 2 * a) (h4 : ((4 : ℝ), 1) ∈ {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1}) :
    {p : ℝ × ℝ | (p.1)^2 / 12 - (p.2)^2 / 3 = 1} = {p : ℝ × ℝ | (p.1)^2 / (a^2) - (p.2)^2 / (b^2) = 1} :=
by
  sorry

end hyperbola_equation_l89_89411


namespace like_terms_calc_l89_89031

theorem like_terms_calc {m n : ℕ} (h1 : m + 2 = 6) (h2 : n + 1 = 3) : (- (m : ℤ))^3 + (n : ℤ)^2 = -60 :=
  sorry

end like_terms_calc_l89_89031


namespace Alec_goal_ratio_l89_89934

theorem Alec_goal_ratio (total_students half_votes thinking_votes more_needed fifth_votes : ℕ)
  (h_class : total_students = 60)
  (h_half : half_votes = total_students / 2)
  (remaining_students : ℕ := total_students - half_votes)
  (h_thinking : thinking_votes = 5)
  (h_fifth : fifth_votes = (remaining_students - thinking_votes) / 5)
  (h_current_votes : half_votes + thinking_votes + fifth_votes = 40)
  (h_needed : more_needed = 5)
  :
  (half_votes + thinking_votes + fifth_votes + more_needed) / total_students = 3 / 4 :=
by sorry

end Alec_goal_ratio_l89_89934


namespace remainder_mod_29_l89_89251

-- Definitions of the given conditions
def N (k : ℕ) := 899 * k + 63

-- The proof statement to be proved
theorem remainder_mod_29 (k : ℕ) : (N k) % 29 = 5 := 
by {
  sorry
}

end remainder_mod_29_l89_89251


namespace S_n_expression_l89_89287

/-- 
  Given a sequence of positive terms {a_n} with sum of the first n terms represented as S_n,
  and given a_1 = 2, and given the relationship 
  S_{n+1}(S_{n+1} - 3^n) = S_n(S_n + 3^n), prove that S_{2023} = (3^2023 + 1) / 2.
-/
theorem S_n_expression
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hr : ∀ n, S (n + 1) * (S (n + 1) - 3^n) = S n * (S n + 3^n)) :
  S 2023 = (3^2023 + 1) / 2 :=
sorry

end S_n_expression_l89_89287


namespace correct_answer_is_ln_abs_l89_89781

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, (0 < x ∧ x < y) → f x ≤ f y

theorem correct_answer_is_ln_abs :
  is_even_function (fun x => Real.log (abs x)) ∧ is_monotonically_increasing_on_pos (fun x => Real.log (abs x)) ∧
  ¬ is_even_function (fun x => x^3) ∧
  ¬ is_monotonically_increasing_on_pos (fun x => Real.cos x) :=
by
  sorry

end correct_answer_is_ln_abs_l89_89781


namespace probability_at_least_one_passes_l89_89500

theorem probability_at_least_one_passes (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 :=
by sorry

end probability_at_least_one_passes_l89_89500


namespace resultant_after_trebled_l89_89477

variable (x : ℕ)

theorem resultant_after_trebled (h : x = 7) : 3 * (2 * x + 9) = 69 := by
  sorry

end resultant_after_trebled_l89_89477


namespace students_in_class_l89_89634

theorem students_in_class (b n : ℕ) :
  6 * (b + 1) = n ∧ 9 * (b - 1) = n → n = 36 :=
by
  sorry

end students_in_class_l89_89634


namespace probability_of_Xiaojia_selection_l89_89190

theorem probability_of_Xiaojia_selection : 
  let students := 2500
  let teachers := 350
  let support_staff := 150
  let total_individuals := students + teachers + support_staff
  let sampled_individuals := 300
  let student_sample := (students : ℝ)/total_individuals * sampled_individuals
  (student_sample / students) = (1 / 10) := 
by
  sorry

end probability_of_Xiaojia_selection_l89_89190


namespace remainder_of_expression_mod7_l89_89595

theorem remainder_of_expression_mod7 :
  (7^6 + 8^7 + 9^8) % 7 = 5 :=
by
  sorry

end remainder_of_expression_mod7_l89_89595


namespace smallest_x_for_1980_power4_l89_89726

theorem smallest_x_for_1980_power4 (M : ℤ) (x : ℕ) (hx : x > 0) :
  (1980 * (x : ℤ)) = M^4 → x = 6006250 :=
by
  -- The proof goes here
  sorry

end smallest_x_for_1980_power4_l89_89726


namespace general_formula_of_geometric_seq_term_in_arithmetic_seq_l89_89221

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Condition: Geometric sequence {a_n} with a_1 = 2 and a_4 = 16
def geometric_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n, a (n + 1) = a n * q

-- General formula for the sequence {a_n}
theorem general_formula_of_geometric_seq 
  (ha : geometric_seq a) (h1 : a 1 = 2) (h4 : a 4 = 16) :
  ∀ n, a n = 2^n :=
sorry

-- Condition: Arithmetic sequence {b_n} with b_3 = a_3 and b_5 = a_5
def arithmetic_seq (b : ℕ → ℝ) := ∃ d : ℝ, ∀ n, b (n + 1) = b n + d

-- Check if a_9 is a term in the sequence {b_n} and find its term number
theorem term_in_arithmetic_seq 
  (ha : geometric_seq a) (hb : arithmetic_seq b)
  (h1 : a 1 = 2) (h4 : a 4 = 16)
  (hb3 : b 3 = a 3) (hb5 : b 5 = a 5) :
  ∃ n, b n = a 9 ∧ n = 45 :=
sorry

end general_formula_of_geometric_seq_term_in_arithmetic_seq_l89_89221


namespace minimum_omega_l89_89966

noncomputable section

-- Define the function f and its properties
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.cos (ω * x + φ)

-- Assumptions based on the given conditions
variables {ω φ : ℝ}
variables (T : ℝ) (hω_pos : 0 < ω) (hφ_range : 0 < φ ∧ φ < π)
variables (hT : f T ω φ = Real.sqrt 3 / 2)
variables (hx_zero : f (π / 9) ω φ = 0)

-- Prove the minimum value of ω is 3
theorem minimum_omega : ω = 3 := sorry

end minimum_omega_l89_89966


namespace repeating_six_as_fraction_l89_89041

theorem repeating_six_as_fraction : (∑' n : ℕ, 6 / (10 * (10 : ℝ)^n)) = (2 / 3) :=
by
  sorry

end repeating_six_as_fraction_l89_89041


namespace total_columns_l89_89226

variables (N L : ℕ)

theorem total_columns (h1 : L > 1500) (h2 : L = 30 * (N - 70)) : N = 180 :=
by
  sorry

end total_columns_l89_89226


namespace sum_of_squares_of_b_l89_89964

-- Define the constants
def b1 := 35 / 64
def b2 := 0
def b3 := 21 / 64
def b4 := 0
def b5 := 7 / 64
def b6 := 0
def b7 := 1 / 64

-- The goal is to prove the sum of squares of these constants
theorem sum_of_squares_of_b : 
  (b1 ^ 2 + b2 ^ 2 + b3 ^ 2 + b4 ^ 2 + b5 ^ 2 + b6 ^ 2 + b7 ^ 2) = 429 / 1024 :=
  by
    -- defer the proof
    sorry

end sum_of_squares_of_b_l89_89964


namespace right_handed_total_l89_89092

theorem right_handed_total
  (total_players : ℕ)
  (throwers : ℕ)
  (left_handed_non_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (non_throwers : ℕ)
  (right_handed_non_throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  non_throwers = total_players - throwers →
  left_handed_non_throwers = non_throwers / 3 →
  right_handed_non_throwers = non_throwers - left_handed_non_throwers →
  right_handed_throwers = throwers →
  right_handed_throwers + right_handed_non_throwers = 56 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end right_handed_total_l89_89092


namespace no_conclusions_deducible_l89_89454

open Set

variable {U : Type}  -- Universe of discourse

-- Conditions
variables (Bars Fins Grips : Set U)

def some_bars_are_not_fins := ∃ x, x ∈ Bars ∧ x ∉ Fins
def no_fins_are_grips := ∀ x, x ∈ Fins → x ∉ Grips

-- Lean statement
theorem no_conclusions_deducible 
  (h1 : some_bars_are_not_fins Bars Fins)
  (h2 : no_fins_are_grips Fins Grips) :
  ¬((∃ x, x ∈ Bars ∧ x ∉ Grips) ∨
    (∃ x, x ∈ Grips ∧ x ∉ Bars) ∨
    (∀ x, x ∈ Bars → x ∉ Grips) ∨
    (∃ x, x ∈ Bars ∧ x ∈ Grips)) :=
sorry

end no_conclusions_deducible_l89_89454


namespace cards_from_around_country_l89_89474

-- Define the total number of cards and the number from home
def total_cards : ℝ := 403.0
def home_cards : ℝ := 287.0

-- Define the expected number of cards from around the country
def expected_country_cards : ℝ := 116.0

-- Theorem statement
theorem cards_from_around_country :
  total_cards - home_cards = expected_country_cards :=
by
  -- Since this only requires the statement, the proof is omitted
  sorry

end cards_from_around_country_l89_89474


namespace smallest_n_perfect_square_and_cube_l89_89441

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l89_89441


namespace minimum_box_value_l89_89957

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end minimum_box_value_l89_89957


namespace perfect_square_trinomial_m_l89_89881

theorem perfect_square_trinomial_m (m : ℤ) :
  ∀ y : ℤ, ∃ a : ℤ, (y^2 - m * y + 1 = (y + a) ^ 2) ∨ (y^2 - m * y + 1 = (y - a) ^ 2) → (m = 2 ∨ m = -2) :=
by 
  sorry

end perfect_square_trinomial_m_l89_89881


namespace bouncy_balls_per_package_l89_89679

variable (x : ℝ)

def maggie_bought_packs : ℝ := 8.0 * x
def maggie_gave_away_packs : ℝ := 4.0 * x
def maggie_bought_again_packs : ℝ := 4.0 * x
def total_kept_bouncy_balls : ℝ := 80

theorem bouncy_balls_per_package :
  (maggie_bought_packs x = total_kept_bouncy_balls) → 
  x = 10 :=
by
  intro h
  sorry

end bouncy_balls_per_package_l89_89679


namespace cat_toy_cost_correct_l89_89005

-- Define the initial amount of money Jessica had.
def initial_amount : ℝ := 11.73

-- Define the amount left after spending.
def amount_left : ℝ := 1.51

-- Define the cost of the cat toy.
def toy_cost : ℝ := initial_amount - amount_left

-- Theorem and statement to prove the cost of the cat toy.
theorem cat_toy_cost_correct : toy_cost = 10.22 := sorry

end cat_toy_cost_correct_l89_89005


namespace volume_to_surface_area_ratio_l89_89925

-- Define the structure of the object consisting of unit cubes
structure CubicObject where
  volume : ℕ
  surface_area : ℕ

-- Define a specific cubic object based on given conditions
def specialCubicObject : CubicObject := {
  volume := 8,
  surface_area := 29
}

-- Statement to prove the ratio of the volume to the surface area
theorem volume_to_surface_area_ratio :
  (specialCubicObject.volume : ℚ) / (specialCubicObject.surface_area : ℚ) = 8 / 29 := by
  sorry

end volume_to_surface_area_ratio_l89_89925


namespace cylinder_surface_area_correct_l89_89700

noncomputable def cylinder_surface_area :=
  let r := 8   -- radius in cm
  let h := 10  -- height in cm
  let arc_angle := 90 -- degrees
  let x := 40
  let y := -40
  let z := 2
  x + y + z

theorem cylinder_surface_area_correct : cylinder_surface_area = 2 := by
  sorry

end cylinder_surface_area_correct_l89_89700


namespace range_of_f_l89_89438

noncomputable def f (x : ℝ) := Real.log (2 - x^2) / Real.log (1 / 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 0 := by
  sorry

end range_of_f_l89_89438


namespace second_month_interest_l89_89848

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end second_month_interest_l89_89848


namespace min_value_frac_f1_f_l89_89928

theorem min_value_frac_f1_f'0 (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_discriminant : b^2 ≤ 4 * a * c) :
  (a + b + c) / b ≥ 2 := 
by
  -- Here goes the proof
  sorry

end min_value_frac_f1_f_l89_89928


namespace ticket_price_l89_89052

theorem ticket_price (P : ℝ) (h_capacity : 50 * P - 24 * P = 208) :
  P = 8 :=
sorry

end ticket_price_l89_89052


namespace not_eq_positive_integers_l89_89076

theorem not_eq_positive_integers (a b : ℤ) (ha : a > 0) (hb : b > 0) : 
  a^3 + (a + b)^2 + b ≠ b^3 + a + 2 :=
by {
  sorry
}

end not_eq_positive_integers_l89_89076


namespace hyperbola_slopes_l89_89205

variables {x1 y1 x2 y2 x y k1 k2 : ℝ}

theorem hyperbola_slopes (h1 : y1^2 - (x1^2 / 2) = 1)
  (h2 : y2^2 - (x2^2 / 2) = 1)
  (hx : x1 + x2 = 2 * x)
  (hy : y1 + y2 = 2 * y)
  (hk1 : k1 = (y2 - y1) / (x2 - x1))
  (hk2 : k2 = y / x) :
  k1 * k2 = 1 / 2 :=
sorry

end hyperbola_slopes_l89_89205


namespace range_of_a_for_monotonic_function_l89_89235

theorem range_of_a_for_monotonic_function (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 0 ≤ (1 / x) + a) → a ≥ -1 / 2 := 
by
  sorry

end range_of_a_for_monotonic_function_l89_89235


namespace range_m_l89_89503

open Real

theorem range_m (m : ℝ)
  (hP : ¬ (∃ x : ℝ, m * x^2 + 1 ≤ 0))
  (hQ : ¬ (∃ x : ℝ, x^2 + m * x + 1 < 0)) :
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_m_l89_89503


namespace necessary_not_sufficient_condition_l89_89442

noncomputable def S (a₁ q : ℝ) : ℝ := a₁ / (1 - q)

theorem necessary_not_sufficient_condition (a₁ q : ℝ) (h₁ : |q| < 1) :
  (a₁ + q = 1) → (S a₁ q = 1) ∧ ¬((S a₁ q = 1) → (a₁ + q = 1)) :=
by
  sorry

end necessary_not_sufficient_condition_l89_89442


namespace exists_five_integers_l89_89666

theorem exists_five_integers :
  ∃ (a b c d e : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
    c ≠ d ∧ c ≠ e ∧ 
    d ≠ e ∧
    ∃ (k1 k2 k3 k4 k5 : ℕ), 
      k1^2 = (a + b + c + d) ∧ 
      k2^2 = (a + b + c + e) ∧ 
      k3^2 = (a + b + d + e) ∧ 
      k4^2 = (a + c + d + e) ∧ 
      k5^2 = (b + c + d + e) := 
sorry

end exists_five_integers_l89_89666


namespace largest_divisor_of_product_l89_89416

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Definition of P, the product of the visible numbers when an 8-sided die is rolled
def P (excluded: ℕ) : ℕ :=
  factorial 8 / excluded

-- The main theorem to prove
theorem largest_divisor_of_product (excluded: ℕ) (h₁: 1 ≤ excluded) (h₂: excluded ≤ 8): 
  ∃ n, n = 192 ∧ ∀ k, k > 192 → ¬k ∣ P excluded :=
sorry

end largest_divisor_of_product_l89_89416


namespace find_xy_l89_89469

theorem find_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 :=
sorry

end find_xy_l89_89469


namespace Jane_remaining_time_l89_89542

noncomputable def JaneRate : ℚ := 1 / 4
noncomputable def RoyRate : ℚ := 1 / 5
noncomputable def workingTime : ℚ := 2
noncomputable def cakeFractionCompletedTogether : ℚ := (JaneRate + RoyRate) * workingTime
noncomputable def remainingCakeFraction : ℚ := 1 - cakeFractionCompletedTogether
noncomputable def timeForJaneToCompleteRemainingCake : ℚ := remainingCakeFraction / JaneRate

theorem Jane_remaining_time :
  timeForJaneToCompleteRemainingCake = 2 / 5 :=
by
  sorry

end Jane_remaining_time_l89_89542


namespace max_value_of_cubes_l89_89227

theorem max_value_of_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 + ab + ac + ad + bc + bd + cd = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 :=
sorry

end max_value_of_cubes_l89_89227


namespace tomato_price_l89_89487

theorem tomato_price (P : ℝ) (W : ℝ) :
  (0.9956 * 0.9 * W = P * W + 0.12 * (P * W)) → P = 0.8 :=
by
  intro h
  sorry

end tomato_price_l89_89487


namespace math_problem_l89_89283

theorem math_problem (x y : ℝ) (h : (x + 2 * y) ^ 3 + x ^ 3 + 2 * x + 2 * y = 0) : x + y - 1 = -1 := 
sorry

end math_problem_l89_89283


namespace angle_sum_straight_line_l89_89755

theorem angle_sum_straight_line (x : ℝ) (h : 4 * x + x = 180) : x = 36 :=
sorry

end angle_sum_straight_line_l89_89755


namespace cricket_matches_total_l89_89608

theorem cricket_matches_total 
  (N : ℕ)
  (avg_total : ℕ → ℕ)
  (avg_first_8 : ℕ)
  (avg_last_4 : ℕ) 
  (h1 : avg_total N = 48)
  (h2 : avg_first_8 = 40)
  (h3 : avg_last_4 = 64) 
  (h_sum : (avg_first_8 * 8 + avg_last_4 * 4 = avg_total N * N)) :
  N = 12 := 
  sorry

end cricket_matches_total_l89_89608


namespace find_sum_of_roots_l89_89961

open Real

theorem find_sum_of_roots (p q r s : ℝ): 
  r + s = 12 * p →
  r * s = 13 * q →
  p + q = 12 * r →
  p * q = 13 * s →
  p ≠ r →
  p + q + r + s = 2028 := by
  intros
  sorry

end find_sum_of_roots_l89_89961


namespace inequality_neg_3_l89_89048

theorem inequality_neg_3 (a b : ℝ) : a < b → -3 * a > -3 * b :=
by
  sorry

end inequality_neg_3_l89_89048


namespace gain_percentage_is_30_l89_89145

-- Definitions based on the conditions
def selling_price : ℕ := 195
def gain : ℕ := 45
def cost_price : ℕ := selling_price - gain
def gain_percentage : ℕ := (gain * 100) / cost_price

-- The statement to prove the gain percentage
theorem gain_percentage_is_30 : gain_percentage = 30 := 
by 
  -- Allow usage of fictive sorry for incomplete proof
  sorry

end gain_percentage_is_30_l89_89145


namespace robot_cost_max_units_A_l89_89349

noncomputable def cost_price_A (x : ℕ) := 1600
noncomputable def cost_price_B (x : ℕ) := 2800

theorem robot_cost (x : ℕ) (y : ℕ) (a : ℕ) (b : ℕ) :
  y = 2 * x - 400 →
  a = 96000 →
  b = 168000 →
  a / x = 6000 →
  b / y = 6000 →
  (x = 1600 ∧ y = 2800) :=
by sorry

theorem max_units_A (m n total_units : ℕ) : 
  total_units = 100 →
  m + n = 100 →
  m ≤ 2 * n →
  m ≤ 66 :=
by sorry

end robot_cost_max_units_A_l89_89349


namespace heat_of_reaction_correct_l89_89782

def delta_H_f_NH4Cl : ℝ := -314.43  -- Enthalpy of formation of NH4Cl in kJ/mol
def delta_H_f_H2O : ℝ := -285.83    -- Enthalpy of formation of H2O in kJ/mol
def delta_H_f_HCl : ℝ := -92.31     -- Enthalpy of formation of HCl in kJ/mol
def delta_H_f_NH4OH : ℝ := -80.29   -- Enthalpy of formation of NH4OH in kJ/mol

def delta_H_rxn : ℝ :=
  ((2 * delta_H_f_NH4OH) + (2 * delta_H_f_HCl)) -
  ((2 * delta_H_f_NH4Cl) + (2 * delta_H_f_H2O))

theorem heat_of_reaction_correct :
  delta_H_rxn = 855.32 :=
  by
    -- Calculation and proof steps go here
    sorry

end heat_of_reaction_correct_l89_89782


namespace solve_equation_l89_89839

theorem solve_equation (x : ℝ) : (x + 3) * (x - 1) = 12 ↔ (x = -5 ∨ x = 3) := sorry

end solve_equation_l89_89839


namespace samantha_tenth_finger_l89_89131

def g (x : ℕ) : ℕ :=
  match x with
  | 2 => 2
  | _ => 0  -- Assume a simple piecewise definition for the sake of the example.

theorem samantha_tenth_finger : g (2) = 2 :=
by  sorry

end samantha_tenth_finger_l89_89131


namespace sum_distances_eq_6sqrt2_l89_89306

-- Define the curves C1 and C2 in Cartesian coordinates
def curve_C1 := { p : ℝ × ℝ | p.1 + p.2 = 3 }
def curve_C2 := { p : ℝ × ℝ | p.2^2 = 2 * p.1 }

-- Defining the point P in ℝ²
def point_P : ℝ × ℝ := (1, 2)

-- Find the sum of distances |PA| + |PB|
theorem sum_distances_eq_6sqrt2 : 
  ∃ A B : ℝ × ℝ, A ∈ curve_C1 ∧ A ∈ curve_C2 ∧ 
                B ∈ curve_C1 ∧ B ∈ curve_C2 ∧ 
                (dist point_P A) + (dist point_P B) = 6 * Real.sqrt 2 := 
sorry

end sum_distances_eq_6sqrt2_l89_89306


namespace inversely_proportional_x_y_l89_89039

theorem inversely_proportional_x_y {x y k : ℝ}
    (h_inv_proportional : x * y = k)
    (h_k : k = 75)
    (h_y : y = 45) :
    x = 5 / 3 :=
by
  sorry

end inversely_proportional_x_y_l89_89039


namespace weighted_mean_is_correct_l89_89393

-- Define the given values
def dollar_from_aunt : ℝ := 9
def euros_from_uncle : ℝ := 9
def dollar_from_sister : ℝ := 7
def dollar_from_friends_1 : ℝ := 22
def dollar_from_friends_2 : ℝ := 23
def euros_from_friends_3 : ℝ := 18
def pounds_from_friends_4 : ℝ := 15
def dollar_from_friends_5 : ℝ := 22

-- Define the exchange rates
def exchange_rate_euro_to_usd : ℝ := 1.20
def exchange_rate_pound_to_usd : ℝ := 1.38

-- Calculate the amounts in USD
def dollar_from_uncle : ℝ := euros_from_uncle * exchange_rate_euro_to_usd
def dollar_from_friends_3_converted : ℝ := euros_from_friends_3 * exchange_rate_euro_to_usd
def dollar_from_friends_4_converted : ℝ := pounds_from_friends_4 * exchange_rate_pound_to_usd

-- Define total amounts from family and friends in USD
def family_total : ℝ := dollar_from_aunt + dollar_from_uncle + dollar_from_sister
def friends_total : ℝ := dollar_from_friends_1 + dollar_from_friends_2 + dollar_from_friends_3_converted + dollar_from_friends_4_converted + dollar_from_friends_5

-- Define weights
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Calculate the weighted mean
def weighted_mean : ℝ := (family_total * family_weight) + (friends_total * friends_weight)

theorem weighted_mean_is_correct : weighted_mean = 76.30 := by
  sorry

end weighted_mean_is_correct_l89_89393


namespace factorize_expression_l89_89394

theorem factorize_expression (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by sorry

end factorize_expression_l89_89394


namespace find_divisor_l89_89344

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_dividend : dividend = 190) (h_quotient : quotient = 9) (h_remainder : remainder = 1) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 21 := 
by
  sorry

end find_divisor_l89_89344


namespace new_perimeter_is_20_l89_89660

/-
Ten 1x1 square tiles are arranged to form a figure whose outside edges form a polygon with a perimeter of 16 units.
Four additional tiles of the same size are added to the figure so that each new tile shares at least one side with 
one of the squares in the original figure. Prove that the new perimeter of the figure could be 20 units.
-/

theorem new_perimeter_is_20 (initial_perimeter : ℕ) (num_initial_tiles : ℕ) 
                            (num_new_tiles : ℕ) (shared_sides : ℕ) 
                            (total_tiles : ℕ) : 
  initial_perimeter = 16 → num_initial_tiles = 10 → num_new_tiles = 4 → 
  shared_sides ≤ 8 → total_tiles = 14 → (initial_perimeter + 2 * (num_new_tiles - shared_sides)) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end new_perimeter_is_20_l89_89660


namespace find_quadruples_l89_89496

def quadrupleSolution (a b c d : ℝ): Prop :=
  (a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b))

def isSolution (a b c d : ℝ): Prop :=
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
  (a = 1 ∧ b = -1 + Real.sqrt 2 ∧ c = -1 ∧ d = 1 - Real.sqrt 2) ∨
  (a = 1 ∧ b = -1 - Real.sqrt 2 ∧ c = -1 ∧ d = 1 + Real.sqrt 2)

theorem find_quadruples (a b c d : ℝ) :
  quadrupleSolution a b c d ↔ isSolution a b c d :=
sorry

end find_quadruples_l89_89496


namespace perpendicular_condition_l89_89201

theorem perpendicular_condition (a : ℝ) :
  let l1 (x y : ℝ) := x + a * y - 2
  let l2 (x y : ℝ) := x - a * y - 1
  (∀ x y, (l1 x y = 0 ↔ l2 x y ≠ 0) ↔ 1 - a * a = 0) →
  (a = -1) ∨ (a = 1) :=
by
  intro
  sorry

end perpendicular_condition_l89_89201


namespace notebook_area_l89_89689

variable (w h : ℝ)

def width_to_height_ratio (w h : ℝ) : Prop := w / h = 7 / 5
def perimeter (w h : ℝ) : Prop := 2 * w + 2 * h = 48
def area (w h : ℝ) : ℝ := w * h

theorem notebook_area (w h : ℝ) (ratio : width_to_height_ratio w h) (peri : perimeter w h) :
  area w h = 140 :=
by
  sorry

end notebook_area_l89_89689


namespace sector_area_l89_89023

theorem sector_area (r : ℝ) (h1 : r = 2) (h2 : 2 * r + r * ((2 * π * r - 2) / r) = 4 * π) :
  (1 / 2) * r^2 * ((4 * π - 2) / r) = 4 * π - 2 :=
by
  sorry

end sector_area_l89_89023


namespace bank_check_problem_l89_89055

theorem bank_check_problem :
  ∃ (x y : ℕ), (0 ≤ y ∧ y ≤ 99) ∧ (y + (x : ℚ) / 100 - 0.05 = 2 * (x + (y : ℚ) / 100)) ∧ x = 31 ∧ y = 63 :=
by
  -- Definitions and Conditions
  sorry

end bank_check_problem_l89_89055


namespace max_f_value_l89_89107

noncomputable def S_n (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def f (n : ℕ) : ℝ := (S_n n : ℝ) / ((n + 32) * S_n (n + 1))

theorem max_f_value : ∃ n : ℕ, f n = 1 / 50 := by
  sorry

end max_f_value_l89_89107


namespace sequence_divisibility_count_l89_89505

theorem sequence_divisibility_count :
  ∀ (f : ℕ → ℕ), (∀ n, n ≥ 2 → f n = 10^n - 1) → 
  (∃ count, count = 504 ∧ ∀ i, 2 ≤ i ∧ i ≤ 2023 → (101 ∣ f i ↔ i % 4 = 0)) :=
by { sorry }

end sequence_divisibility_count_l89_89505


namespace rectangle_length_fraction_of_circle_radius_l89_89942

noncomputable def square_side (area : ℕ) : ℕ :=
  Nat.sqrt area

noncomputable def rectangle_length (breadth area : ℕ) : ℕ :=
  area / breadth

theorem rectangle_length_fraction_of_circle_radius
  (square_area : ℕ)
  (rectangle_breadth : ℕ)
  (rectangle_area : ℕ)
  (side := square_side square_area)
  (radius := side)
  (length := rectangle_length rectangle_breadth rectangle_area) :
  square_area = 4761 →
  rectangle_breadth = 13 →
  rectangle_area = 598 →
  length / radius = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end rectangle_length_fraction_of_circle_radius_l89_89942


namespace fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l89_89645

theorem fraction_inequalities (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1 / 2 ≤ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ∧ (a ^ 3 + b ^ 3) / (a ^ 2 + b ^ 2) ≤ 1 :=
sorry

theorem fraction_inequality_equality_right (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1 - a) * (1 - b) = 0 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0) :=
sorry

theorem fraction_inequality_equality_left (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  a = b ↔ a = 1 / 2 ∧ b = 1 / 2 :=
sorry

end fraction_inequalities_fraction_inequality_equality_right_fraction_inequality_equality_left_l89_89645


namespace aarti_completes_work_multiple_l89_89796

-- Define the condition that Aarti can complete one piece of work in 9 days.
def aarti_work_rate (work_size : ℕ) : ℕ := 9

-- Define the task to find how many times she will complete the work in 27 days.
def aarti_work_multiple (total_days : ℕ) (work_size: ℕ) : ℕ :=
  total_days / (aarti_work_rate work_size)

-- The theorem to prove the number of times Aarti will complete the work.
theorem aarti_completes_work_multiple : aarti_work_multiple 27 1 = 3 := by
  sorry

end aarti_completes_work_multiple_l89_89796


namespace Papi_Calot_has_to_buy_141_plants_l89_89051

noncomputable def calc_number_of_plants : Nat :=
  let initial_plants := 7 * 18
  let additional_plants := 15
  initial_plants + additional_plants

theorem Papi_Calot_has_to_buy_141_plants :
  calc_number_of_plants = 141 :=
by
  sorry

end Papi_Calot_has_to_buy_141_plants_l89_89051


namespace not_basic_logic_structure_l89_89447

def SequenceStructure : Prop := true
def ConditionStructure : Prop := true
def LoopStructure : Prop := true
def DecisionStructure : Prop := true

theorem not_basic_logic_structure : ¬ (SequenceStructure ∨ ConditionStructure ∨ LoopStructure) -> DecisionStructure := by
  sorry

end not_basic_logic_structure_l89_89447


namespace initially_planned_days_l89_89674

theorem initially_planned_days (D : ℕ) (h1 : 6 * 3 + 10 * 3 = 6 * D) : D = 8 := by
  sorry

end initially_planned_days_l89_89674


namespace people_in_the_theater_l89_89896

theorem people_in_the_theater : ∃ P : ℕ, P = 100 ∧ 
  P = 19 + (1/2 : ℚ) * P + (1/4 : ℚ) * P + 6 := by
  sorry

end people_in_the_theater_l89_89896


namespace solve_eq_l89_89317

noncomputable def fx (x : ℝ) : ℝ :=
  ((x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 3) * (x - 2) * (x - 1) * (x - 5)) /
  ((x - 2) * (x - 4) * (x - 2) * (x - 5))

theorem solve_eq (x : ℝ) (h : x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5) :
  fx x = 1 ↔ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
by
  sorry

end solve_eq_l89_89317


namespace problem1_problem2_problem3_l89_89690

-- Problem 1
theorem problem1 :
  1 - 1^2022 + ((-1/2)^2) * (-2)^3 * (-2)^2 - |Real.pi - 3.14|^0 = -10 :=
by sorry

-- Problem 2
variables (a b : ℝ)

theorem problem2 :
  a^3 * (-b^3)^2 + (-2 * a * b)^3 = a^3 * b^6 - 8 * a^3 * b^3 :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) :
  (2 * a^3 * b^2 - 3 * a^2 * b - 4 * a) * 2 * b = 4 * a^3 * b^3 - 6 * a^2 * b^2 - 8 * a * b :=
by sorry

end problem1_problem2_problem3_l89_89690


namespace ben_fraction_of_taxes_l89_89815

theorem ben_fraction_of_taxes 
  (gross_income : ℝ) (car_payment : ℝ) (fraction_spend_on_car : ℝ) (after_tax_income_fraction : ℝ) 
  (h1 : gross_income = 3000) (h2 : car_payment = 400) (h3 : fraction_spend_on_car = 0.2) :
  after_tax_income_fraction = (1 / 3) :=
by
  sorry

end ben_fraction_of_taxes_l89_89815


namespace baking_trays_used_l89_89850

-- Let T be the number of baking trays Anna used.
variable (T : ℕ)

-- Condition: Each tray has 20 cupcakes.
def cupcakes_per_tray : ℕ := 20

-- Condition: Each cupcake was sold for $2.
def cupcake_price : ℕ := 2

-- Condition: Only 3/5 of the cupcakes were sold.
def fraction_sold : ℚ := 3 / 5

-- Condition: Anna earned $96 from sold cupcakes.
def earnings : ℕ := 96

-- Derived expressions:
def total_cupcakes (T : ℕ) : ℕ := cupcakes_per_tray * T

def sold_cupcakes (T : ℕ) : ℚ := fraction_sold * total_cupcakes T

def total_earnings (T : ℕ) : ℚ := cupcake_price * sold_cupcakes T

-- The statement to be proved: Given the conditions, the number of trays T must be 4.
theorem baking_trays_used (h : total_earnings T = earnings) : T = 4 := by
  sorry

end baking_trays_used_l89_89850


namespace solve_equation_l89_89570

theorem solve_equation :
  ∃ x : ℝ, (3 * x^2 / (x - 2)) - (4 * x + 11) / 5 + (7 - 9 * x) / (x - 2) + 2 = 0 :=
sorry

end solve_equation_l89_89570


namespace hiker_distance_l89_89572

variable (s t d : ℝ)
variable (h₁ : (s + 1) * (2 / 3 * t) = d)
variable (h₂ : (s - 1) * (t + 3) = d)

theorem hiker_distance  : d = 6 :=
by
  sorry

end hiker_distance_l89_89572


namespace solution_set_of_inequality_l89_89956

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 ≥ 0 } = { x : ℝ | x ≤ -2 ∨ 1 ≤ x } :=
by
  sorry

end solution_set_of_inequality_l89_89956


namespace sum_of_roots_l89_89180

theorem sum_of_roots (α β : ℝ) (h1 : α^2 - 4 * α + 3 = 0) (h2 : β^2 - 4 * β + 3 = 0) (h3 : α ≠ β) :
  α + β = 4 :=
sorry

end sum_of_roots_l89_89180


namespace xy_sufficient_not_necessary_l89_89489

theorem xy_sufficient_not_necessary (x y : ℝ) :
  (xy ≠ 6) → (x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) := by
  sorry

end xy_sufficient_not_necessary_l89_89489


namespace missing_digit_in_138_x_6_divisible_by_9_l89_89639

theorem missing_digit_in_138_x_6_divisible_by_9 :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ (1 + 3 + 8 + x + 6) % 9 = 0 ∧ x = 0 :=
by
  sorry

end missing_digit_in_138_x_6_divisible_by_9_l89_89639


namespace ratio_of_ages_in_two_years_l89_89254

theorem ratio_of_ages_in_two_years
    (S : ℕ) (M : ℕ) 
    (h1 : M = S + 32)
    (h2 : S = 30) : 
    (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l89_89254


namespace exists_coprime_less_than_100_l89_89774

theorem exists_coprime_less_than_100 (a b c : ℕ) (ha : a < 1000000) (hb : b < 1000000) (hc : c < 1000000) :
  ∃ d, d < 100 ∧ gcd d a = 1 ∧ gcd d b = 1 ∧ gcd d c = 1 :=
by sorry

end exists_coprime_less_than_100_l89_89774


namespace inequality_proof_equality_conditions_l89_89838

theorem inequality_proof
  (x y : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) ≥
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

theorem equality_conditions
  (x y : ℝ) :
  (x = y ∨ x = 1 ∨ y = 1) ↔
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) =
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

end inequality_proof_equality_conditions_l89_89838


namespace smallest_value_l89_89832

theorem smallest_value : 54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end smallest_value_l89_89832


namespace negation_of_universal_statement_l89_89272

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_statement_l89_89272


namespace prove_expression_value_l89_89846

theorem prove_expression_value (m n : ℝ) (h : m^2 + 3 * n - 1 = 2) : 2 * m^2 + 6 * n + 1 = 7 := by
  sorry

end prove_expression_value_l89_89846


namespace grape_juice_percentage_l89_89935

theorem grape_juice_percentage
  (initial_volume : ℝ) (initial_percentage : ℝ) (added_juice : ℝ)
  (h_initial_volume : initial_volume = 50)
  (h_initial_percentage : initial_percentage = 0.10)
  (h_added_juice : added_juice = 10) :
  ((initial_percentage * initial_volume + added_juice) / (initial_volume + added_juice) * 100) = 25 := 
by
  sorry

end grape_juice_percentage_l89_89935


namespace nth_position_equation_l89_89686

theorem nth_position_equation (n : ℕ) (h : n > 0) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end nth_position_equation_l89_89686


namespace find_center_of_circle_l89_89162

theorem find_center_of_circle :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  ( ∀ (x y : ℝ), ( (x = 1 ∧ y = 2) ∨ (x = 1 ∧ y = 1) ∨ (∃ t : ℝ, y = 2 * t + 3) ) → 
  (x - a)^2 + (y - b)^2 = (1 - a)^2 + (1 - b)^2 ) :=
sorry

end find_center_of_circle_l89_89162


namespace sequences_with_both_properties_are_constant_l89_89204

-- Definitions according to the problem's conditions
def arithmetic_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)

def geometric_sequence (seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) / seq n = seq (n + 2) / seq (n + 1)

-- Definition of the sequence properties combined
def arithmetic_and_geometric_sequence (seq : ℕ → ℝ) : Prop :=
  arithmetic_sequence seq ∧ geometric_sequence seq

-- Problem to prove
theorem sequences_with_both_properties_are_constant (seq : ℕ → ℝ) :
  arithmetic_and_geometric_sequence seq → ∀ n m : ℕ, seq n = seq m :=
sorry

end sequences_with_both_properties_are_constant_l89_89204


namespace correct_formula_l89_89959

def table : List (ℕ × ℕ) :=
    [(1, 3), (2, 8), (3, 15), (4, 24), (5, 35)]

theorem correct_formula : ∀ x y, (x, y) ∈ table → y = x^2 + 4 * x + 3 :=
by
  intros x y H
  sorry

end correct_formula_l89_89959


namespace painted_cells_l89_89775

theorem painted_cells (k l : ℕ) (h : k * l = 74) :
    (2 * k + 1) * (2 * l + 1) - k * l = 301 ∨ 
    (2 * k + 1) * (2 * l + 1) - k * l = 373 :=
sorry

end painted_cells_l89_89775


namespace golf_tees_per_member_l89_89391

theorem golf_tees_per_member (T : ℕ) : 
  (∃ (t : ℕ), 
     t = 4 * T ∧ 
     (∀ (g : ℕ), g ≤ 2 → g * 12 + 28 * 2 = t)
  ) → T = 20 :=
by
  intros h
  -- problem statement is enough for this example
  sorry

end golf_tees_per_member_l89_89391


namespace elizabeth_needs_to_borrow_more_money_l89_89780

-- Define the costs of the items
def pencil_cost : ℝ := 6.00 
def notebook_cost : ℝ := 3.50 
def pen_cost : ℝ := 2.25 

-- Define the amount of money Elizabeth initially has and what she borrowed
def elizabeth_money : ℝ := 5.00 
def borrowed_money : ℝ := 0.53 

-- Define the total cost of the items
def total_cost : ℝ := pencil_cost + notebook_cost + pen_cost

-- Define the total amount of money Elizabeth has
def total_money : ℝ := elizabeth_money + borrowed_money

-- Define the additional amount Elizabeth needs to borrow
def amount_needed_to_borrow : ℝ := total_cost - total_money

-- The theorem to prove that Elizabeth needs to borrow an additional $6.22
theorem elizabeth_needs_to_borrow_more_money : 
  amount_needed_to_borrow = 6.22 := by 
    -- Proof goes here
    sorry

end elizabeth_needs_to_borrow_more_money_l89_89780


namespace sculpture_height_correct_l89_89081

/-- Define the conditions --/
def base_height_in_inches : ℝ := 4
def total_height_in_feet : ℝ := 3.1666666666666665
def inches_per_foot : ℝ := 12

/-- Define the conversion from feet to inches for the total height --/
def total_height_in_inches : ℝ := total_height_in_feet * inches_per_foot

/-- Define the height of the sculpture in inches --/
def sculpture_height_in_inches : ℝ := total_height_in_inches - base_height_in_inches

/-- The proof problem in Lean 4 statement --/
theorem sculpture_height_correct :
  sculpture_height_in_inches = 34 := by
  sorry

end sculpture_height_correct_l89_89081


namespace initial_volume_kola_solution_l89_89325

-- Initial composition of the kola solution
def initial_composition_sugar (V : ℝ) : ℝ := 0.20 * V

-- Final volume after additions
def final_volume (V : ℝ) : ℝ := V + 3.2 + 12 + 6.8

-- Final amount of sugar after additions
def final_amount_sugar (V : ℝ) : ℝ := initial_composition_sugar V + 3.2

-- Final percentage of sugar in the solution
def final_percentage_sugar (total_sol : ℝ) : ℝ := 0.1966850828729282 * total_sol

theorem initial_volume_kola_solution : 
  ∃ V : ℝ, final_amount_sugar V = final_percentage_sugar (final_volume V) :=
sorry

end initial_volume_kola_solution_l89_89325


namespace solve_inequality_for_a_l89_89381

theorem solve_inequality_for_a (a : ℝ) :
  (∀ x : ℝ, abs (x^2 + 3 * a * x + 4 * a) ≤ 3 → x = -3 * a / 2)
  ↔ (a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13) :=
by 
  sorry

end solve_inequality_for_a_l89_89381


namespace length_of_base_l89_89431

-- Define the conditions of the problem
def base_of_triangle (b : ℕ) : Prop :=
  ∃ c : ℕ, b + 3 + c = 12 ∧ 9 + b*b = c*c

-- Statement to prove
theorem length_of_base : base_of_triangle 4 :=
  sorry

end length_of_base_l89_89431


namespace find_p5_l89_89264

-- Definitions based on conditions from the problem
def p (x : ℝ) : ℝ :=
  x^4 - 10 * x^3 + 35 * x^2 - 50 * x + 18  -- this construction ensures it's a quartic monic polynomial satisfying provided conditions

-- The main theorem we want to prove
theorem find_p5 :
  p 1 = 3 ∧ p 2 = 7 ∧ p 3 = 13 ∧ p 4 = 21 → p 5 = 51 :=
by
  -- The proof will be inserted here later
  sorry

end find_p5_l89_89264


namespace atomic_weight_of_calcium_l89_89315

theorem atomic_weight_of_calcium (Ca I : ℝ) (h1 : 294 = Ca + 2 * I) (h2 : I = 126.9) : Ca = 40.2 :=
by
  sorry

end atomic_weight_of_calcium_l89_89315


namespace find_a12_l89_89783

variable (a : ℕ → ℝ) (q : ℝ)
variable (h1 : ∀ n, a (n + 1) = a n * q)
variable (h2 : abs q > 1)
variable (h3 : a 1 + a 6 = 2)
variable (h4 : a 3 * a 4 = -15)

theorem find_a12 : a 11 = -25 / 3 :=
by sorry

end find_a12_l89_89783


namespace caffeine_over_goal_l89_89455

theorem caffeine_over_goal (cups_per_day : ℕ) (mg_per_cup : ℕ) (caffeine_goal : ℕ) (total_cups : ℕ) :
  total_cups = 3 ->
  cups_per_day = 3 ->
  mg_per_cup = 80 ->
  caffeine_goal = 200 ->
  (cups_per_day * mg_per_cup) - caffeine_goal = 40 := by
  sorry

end caffeine_over_goal_l89_89455


namespace undefined_hydrogen_production_l89_89305

-- Define the chemical species involved as follows:
structure ChemQty where
  Ethane : ℕ
  Oxygen : ℕ
  CarbonDioxide : ℕ
  Water : ℕ

-- Balanced reaction equation
def balanced_reaction : ChemQty :=
  { Ethane := 2, Oxygen := 7, CarbonDioxide := 4, Water := 6 }

-- Given conditions as per problem scenario
def initial_state : ChemQty :=
  { Ethane := 1, Oxygen := 2, CarbonDioxide := 0, Water := 0 }

-- The statement reflecting the unclear result of the reaction under the given conditions.
theorem undefined_hydrogen_production :
  initial_state.Oxygen < balanced_reaction.Oxygen / balanced_reaction.Ethane * initial_state.Ethane →
  ∃ water_products : ℕ, water_products ≤ 6 * initial_state.Ethane / 2 := 
by
  -- Due to incomplete reaction
  sorry

end undefined_hydrogen_production_l89_89305


namespace parabola_focus_l89_89564

theorem parabola_focus (a : ℝ) (h : a ≠ 0) (h_directrix : ∀ x y : ℝ, y^2 = a * x → x = -1) : 
    ∃ x y : ℝ, (y = 0 ∧ x = 1 ∧ y^2 = a * x) :=
sorry

end parabola_focus_l89_89564


namespace find_x_l89_89520

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h : a = (Real.cos (3 * x / 2), Real.sin (3 * x / 2)) ∧ b = (Real.cos (x / 2), -Real.sin (x / 2)) ∧ (a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2 = 1 ∧ 0 ≤ x ∧ x ≤ Real.pi)  :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_l89_89520


namespace shaded_region_area_l89_89017

-- Definitions based on given conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def squares_in_large_square : ℕ := 16

-- The area of the entire shaded region
def area_of_shaded_region : ℝ := 78.125

-- Theorem to prove 
theorem shaded_region_area 
  (num_squares : ℕ) 
  (diagonal_length : ℝ) 
  (squares_in_large_square : ℕ) : 
  (num_squares = 25) → 
  (diagonal_length = 10) → 
  (squares_in_large_square = 16) → 
  area_of_shaded_region = 78.125 := 
by {
  sorry -- proof to be filled
}

end shaded_region_area_l89_89017


namespace tickets_per_ride_factor_l89_89464

theorem tickets_per_ride_factor (initial_tickets spent_tickets remaining_tickets : ℕ) 
  (h1 : initial_tickets = 40) 
  (h2 : spent_tickets = 28) 
  (h3 : remaining_tickets = initial_tickets - spent_tickets) : 
  ∃ k : ℕ, remaining_tickets = 12 ∧ (∀ m : ℕ, m ∣ remaining_tickets → m = k) → (k ∣ 12) :=
by
  sorry

end tickets_per_ride_factor_l89_89464


namespace arithmetic_geometric_sequences_l89_89083

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end arithmetic_geometric_sequences_l89_89083


namespace range_of_a_l89_89161

theorem range_of_a (a : ℝ) : (2 * a - 8) / 3 < 0 → a < 4 :=
by sorry

end range_of_a_l89_89161


namespace even_perfect_square_factors_l89_89712

theorem even_perfect_square_factors : 
  (∃ count : ℕ, count = 3 * 2 * 3 ∧ 
    (∀ (a b c : ℕ), 
      (1 ≤ a ∧ a ≤ 6 ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ b ≤ 3 ∧ c % 2 = 0 ∧ c ≤ 4) → 
      (2^a * 7^b * 3^c ∣ 2^6 * 7^3 * 3^4))) :=
sorry

end even_perfect_square_factors_l89_89712


namespace interior_edges_sum_l89_89007

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l89_89007


namespace lisa_flight_time_l89_89555

theorem lisa_flight_time
  (distance : ℕ) (speed : ℕ) (time : ℕ)
  (h_distance : distance = 256)
  (h_speed : speed = 32)
  (h_time : time = distance / speed) :
  time = 8 :=
by sorry

end lisa_flight_time_l89_89555


namespace no_m_for_P_eq_S_m_le_3_for_P_implies_S_l89_89125

namespace ProofProblem

def P (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def S (m x : ℝ) : Prop := |x - 1| ≤ m

theorem no_m_for_P_eq_S : ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S m x := sorry

theorem m_le_3_for_P_implies_S : ∀ (m : ℝ), (m ≤ 3) → (∀ x, S m x → P x) := sorry

end ProofProblem

end no_m_for_P_eq_S_m_le_3_for_P_implies_S_l89_89125


namespace roots_form_parallelogram_l89_89794

theorem roots_form_parallelogram :
  let polynomial := fun (z : ℂ) (a : ℝ) =>
    z^4 - 8*z^3 + 13*a*z^2 - 2*(3*a^2 + 2*a - 4)*z - 2
  let a1 := 7.791
  let a2 := -8.457
  ∀ z1 z2 z3 z4 : ℂ,
    ( (polynomial z1 a1 = 0) ∧ (polynomial z2 a1 = 0) ∧ (polynomial z3 a1 = 0) ∧ (polynomial z4 a1 = 0)
    ∨ (polynomial z1 a2 = 0) ∧ (polynomial z2 a2 = 0) ∧ (polynomial z3 a2 = 0) ∧ (polynomial z4 a2 = 0) )
    → ( (z1 + z2 + z3 + z4) / 4 = 2 )
    → ( Complex.abs (z1 - z2) = Complex.abs (z3 - z4) 
      ∧ Complex.abs (z1 - z3) = Complex.abs (z2 - z4) ) := sorry

end roots_form_parallelogram_l89_89794


namespace first_company_managers_percentage_l89_89622

-- Definitions from the conditions
variable (F M : ℝ) -- total workforce of first company and merged company
variable (x : ℝ) -- percentage of managers in the first company
variable (cond1 : 0.25 * M = F) -- 25% of merged company's workforce originated from the first company
variable (cond2 : 0.25 * M / M = 0.25) -- resulting merged company's workforce consists of 25% managers

-- The statement to prove
theorem first_company_managers_percentage : x = 25 :=
by
  sorry

end first_company_managers_percentage_l89_89622


namespace total_number_of_birds_l89_89428

theorem total_number_of_birds (B C G S W : ℕ) (h1 : C = 2 * B) (h2 : G = 4 * B)
  (h3 : S = (C + G) / 2) (h4 : W = 8) (h5 : B = 2 * W) :
  C + G + S + W + B = 168 :=
  by
  sorry

end total_number_of_birds_l89_89428


namespace area_triangle_QCA_l89_89280

/--
  Given:
  - θ (θ is acute) is the angle at Q between QA and QC
  - Q is at the coordinates (0, 12)
  - A is at the coordinates (3, 12)
  - C is at the coordinates (0, p)

  Prove that the area of triangle QCA is (3/2) * (12 - p) * sin(θ).
-/
theorem area_triangle_QCA (p θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) :
  let Q := (0, 12)
  let A := (3, 12)
  let C := (0, p)
  let base := 3
  let height := (12 - p) * Real.sin θ
  let area := (1 / 2) * base * height
  area = (3 / 2) * (12 - p) * Real.sin θ := by
  sorry

end area_triangle_QCA_l89_89280


namespace correct_location_l89_89699

variable (A B C D : Prop)

axiom student_A_statement : ¬ A ∧ B
axiom student_B_statement : ¬ B ∧ C
axiom student_C_statement : ¬ B ∧ ¬ D
axiom ms_Hu_response : 
  ( (¬ A ∧ B = true) ∨ (¬ B ∧ C = true) ∨ (¬ B ∧ ¬ D = true) ) ∧ 
  ( (¬ A ∧ B = false) ∨ (¬ B ∧ C = false) ∨ (¬ B ∧ ¬ D = false) = false ) ∧ 
  ( (¬ A ∧ B ∨ ¬ B ∧ C ∨ ¬ B ∧ ¬ D) -> false )

theorem correct_location : B ∨ A := 
sorry

end correct_location_l89_89699


namespace shoes_multiple_l89_89364

-- Define the number of shoes each has
variables (J E B : ℕ)

-- Conditions
axiom h1 : B = 22
axiom h2 : J = E / 2
axiom h3 : J + E + B = 121

-- Prove the multiple of E to B is 3
theorem shoes_multiple : E / B = 3 :=
by
  -- Inject the provisional proof
  sorry

end shoes_multiple_l89_89364


namespace probability_not_snow_l89_89697

theorem probability_not_snow (P_snow : ℚ) (h : P_snow = 2 / 5) : (1 - P_snow = 3 / 5) :=
by 
  rw [h]
  norm_num

end probability_not_snow_l89_89697


namespace david_started_with_15_samsung_phones_l89_89093

-- Definitions
def SamsungPhonesAtEnd : ℕ := 10 -- S_e
def IPhonesAtEnd : ℕ := 5 -- I_e
def SamsungPhonesThrownOut : ℕ := 2 -- S_d
def IPhonesThrownOut : ℕ := 1 -- I_d
def TotalPhonesSold : ℕ := 4 -- C

-- Number of iPhones sold
def IPhonesSold : ℕ := IPhonesThrownOut

-- Assume: The remaining phones sold are Samsung phones
def SamsungPhonesSold : ℕ := TotalPhonesSold - IPhonesSold

-- Calculate the number of Samsung phones David started the day with
def SamsungPhonesAtStart : ℕ := SamsungPhonesAtEnd + SamsungPhonesThrownOut + SamsungPhonesSold

-- Statement
theorem david_started_with_15_samsung_phones : SamsungPhonesAtStart = 15 := by
  sorry

end david_started_with_15_samsung_phones_l89_89093


namespace perimeter_of_triangle_l89_89866

def point (x y : ℝ) := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter_triangle (a b c : ℝ × ℝ) : ℝ :=
  distance a b + distance b c + distance c a

theorem perimeter_of_triangle :
  let A := point 1 2
  let B := point 6 8
  let C := point 1 5
  perimeter_triangle A B C = Real.sqrt 61 + Real.sqrt 34 + 3 :=
by
  -- proof steps can be provided here
  sorry

end perimeter_of_triangle_l89_89866


namespace point_in_second_quadrant_l89_89591

/-- Define the quadrants in the Cartesian coordinate system -/
def quadrant (x y : ℤ) : String :=
  if x > 0 ∧ y > 0 then "First quadrant"
  else if x < 0 ∧ y > 0 then "Second quadrant"
  else if x < 0 ∧ y < 0 then "Third quadrant"
  else if x > 0 ∧ y < 0 then "Fourth quadrant"
  else "On the axis"

theorem point_in_second_quadrant :
  quadrant (-3) 2005 = "Second quadrant" :=
by
  sorry

end point_in_second_quadrant_l89_89591


namespace bridge_length_l89_89050

/-- The length of the bridge that a train 110 meters long and traveling at 45 km/hr can cross in 30 seconds is 265 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (cross_time_sec : ℝ) (bridge_length : ℝ) :
  train_length = 110 ∧ train_speed_kmh = 45 ∧ cross_time_sec = 30 ∧ bridge_length = 265 → 
  (train_speed_kmh * (1000 / 3600) * cross_time_sec - train_length = bridge_length) :=
by
  sorry

end bridge_length_l89_89050


namespace snooker_tournament_total_cost_l89_89392

def VIP_cost : ℝ := 45
def GA_cost : ℝ := 20
def total_tickets_sold : ℝ := 320
def vip_and_general_admission_relationship := 276

def total_cost_of_tickets : ℝ := 6950

theorem snooker_tournament_total_cost 
  (V G : ℝ)
  (h1 : VIP_cost * V + GA_cost * G = total_cost_of_tickets)
  (h2 : V + G = total_tickets_sold)
  (h3 : V = G - vip_and_general_admission_relationship) : 
  VIP_cost * V + GA_cost * G = total_cost_of_tickets := 
by {
  sorry
}

end snooker_tournament_total_cost_l89_89392


namespace smaller_number_is_25_l89_89900

theorem smaller_number_is_25 (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end smaller_number_is_25_l89_89900


namespace range_of_m_l89_89024

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_m (m : ℝ) 
  (h_odd : ∀ x, f (-x) = -f x) 
  (h_incr : ∀ x y, x < y → f x < f y) : 
  f (2 * m - 1) + f (3 - m) > 0 ↔ m > -2 := 
by 
  sorry

end range_of_m_l89_89024


namespace no_solution_in_nat_for_xx_plus_2yy_eq_zz_l89_89301

theorem no_solution_in_nat_for_xx_plus_2yy_eq_zz :
  ¬∃ (x y z : ℕ), x^x + 2 * y^y = z^z := by
  sorry

end no_solution_in_nat_for_xx_plus_2yy_eq_zz_l89_89301


namespace birgit_hiking_time_l89_89062

def hiking_conditions
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ) : Prop :=
  ∃ (average_speed_time : ℝ) (birgit_speed_time : ℝ) (total_minutes_hiked : ℝ),
    total_minutes_hiked = hours_hiked * 60 ∧
    average_speed_time = total_minutes_hiked / distance_km ∧
    birgit_speed_time = average_speed_time - time_faster ∧
    (birgit_speed_time * distance_target_km = 48)

theorem birgit_hiking_time
  (hours_hiked : ℝ)
  (distance_km : ℝ)
  (time_faster : ℝ)
  (distance_target_km : ℝ)
  : hiking_conditions hours_hiked distance_km time_faster distance_target_km :=
by
  use 10, 6, 210
  sorry

end birgit_hiking_time_l89_89062


namespace quadratic_vertex_property_l89_89389

variable {a b c x0 y0 m n : ℝ}

-- Condition 1: (x0, y0) is a fixed point on the graph of the quadratic function y = ax^2 + bx + c
axiom fixed_point_on_graph : y0 = a * x0^2 + b * x0 + c

-- Condition 2: (m, n) is a moving point on the graph of the quadratic function
axiom moving_point_on_graph : n = a * m^2 + b * m + c

-- Condition 3: For any real number m, a(y0 - n) ≤ 0
axiom inequality_condition : ∀ m : ℝ, a * (y0 - (a * m^2 + b * m + c)) ≤ 0

-- Statement to prove
theorem quadratic_vertex_property : 2 * a * x0 + b = 0 := 
sorry

end quadratic_vertex_property_l89_89389


namespace ordered_pair_a_c_l89_89002

theorem ordered_pair_a_c (a c : ℝ) (h_quad: ∀ x : ℝ, a * x^2 + 16 * x + c = 0)
    (h_sum: a + c = 25) (h_ineq: a < c) : (a = 3 ∧ c = 22) :=
by
  -- The proof is omitted
  sorry

end ordered_pair_a_c_l89_89002


namespace rectangle_dimensions_l89_89535

theorem rectangle_dimensions (w l : ℕ) (h : l = w + 5) (hp : 2 * l + 2 * w = 34) : w = 6 ∧ l = 11 := 
by 
  sorry

end rectangle_dimensions_l89_89535


namespace triangle_area_half_l89_89160

theorem triangle_area_half (AB AC BC : ℝ) (h₁ : AB = 8) (h₂ : AC = BC) (h₃ : AC * AC = AB * AB / 2) (h₄ : AC = BC) : 
  (1 / 2) * (1 / 2 * AB * AB) = 16 :=
  by
  sorry

end triangle_area_half_l89_89160


namespace twenty_first_term_is_4641_l89_89819

def nthGroupStart (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

def sumGroup (start n : ℕ) : ℕ :=
  (n * (start + (start + n - 1))) / 2

theorem twenty_first_term_is_4641 : sumGroup (nthGroupStart 21) 21 = 4641 := by
  sorry

end twenty_first_term_is_4641_l89_89819


namespace angle_MON_l89_89071

theorem angle_MON (O M N : ℝ × ℝ) (D : ℝ) :
  (O = (0, 0)) →
  (M = (-2, 2)) →
  (N = (2, 2)) →
  (x^2 + y^2 + D * x - 4 * y = 0) →
  (D = 0) →
  ∃ θ : ℝ, θ = 90 :=
by
  sorry

end angle_MON_l89_89071


namespace exponent_sum_equality_l89_89999

theorem exponent_sum_equality {a : ℕ} (h1 : 2^12 + 1 = 17 * a) (h2: a = 2^8 + 2^7 + 2^6 + 2^5 + 2^0) : 
  ∃ a1 a2 a3 a4 a5 : ℕ, 
    a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ 
    2^a1 + 2^a2 + 2^a3 + 2^a4 + 2^a5 = a ∧ 
    a1 = 0 ∧ a2 = 5 ∧ a3 = 6 ∧ a4 = 7 ∧ a5 = 8 ∧ 
    5 = 5 :=
by {
  sorry
}

end exponent_sum_equality_l89_89999


namespace music_tool_cost_l89_89073

noncomputable def flute_cost : ℝ := 142.46
noncomputable def song_book_cost : ℝ := 7
noncomputable def total_spent : ℝ := 158.35

theorem music_tool_cost :
    total_spent - (flute_cost + song_book_cost) = 8.89 :=
by
  sorry

end music_tool_cost_l89_89073


namespace percentage_passed_l89_89749

-- Definitions corresponding to the conditions
def F_H : ℝ := 25
def F_E : ℝ := 35
def F_B : ℝ := 40

-- Main theorem stating the question's proof.
theorem percentage_passed :
  (100 - (F_H + F_E - F_B)) = 80 :=
by
  -- we can transcribe the remaining process here if needed.
  sorry

end percentage_passed_l89_89749


namespace sum_of_two_coprimes_l89_89009

theorem sum_of_two_coprimes (n : ℤ) (h : n ≥ 7) : 
  ∃ a b : ℤ, a + b = n ∧ Int.gcd a b = 1 ∧ a > 1 ∧ b > 1 :=
by
  sorry

end sum_of_two_coprimes_l89_89009


namespace pandemic_cut_percentage_l89_89968

-- Define the conditions
def initial_planned_production : ℕ := 200
def decrease_due_to_metal_shortage : ℕ := 50
def doors_per_car : ℕ := 5
def total_doors_produced : ℕ := 375

-- Define the quantities after metal shortage and before the pandemic
def production_after_metal_shortage : ℕ := initial_planned_production - decrease_due_to_metal_shortage
def doors_after_metal_shortage : ℕ := production_after_metal_shortage * doors_per_car
def cars_after_pandemic : ℕ := total_doors_produced / doors_per_car
def reduction_in_production : ℕ := production_after_metal_shortage - cars_after_pandemic

-- Define the expected percentage cut
def expected_percentage_cut : ℕ := 50

-- Prove that the percentage of production cut due to the pandemic is as required
theorem pandemic_cut_percentage : (reduction_in_production * 100 / production_after_metal_shortage) = expected_percentage_cut := by
  sorry

end pandemic_cut_percentage_l89_89968


namespace find_y_l89_89339

-- Conditions as definitions in Lean 4
def angle_AXB : ℝ := 180
def angle_AX : ℝ := 70
def angle_BX : ℝ := 40
def angle_CY : ℝ := 130

-- The Lean statement for the proof problem
theorem find_y (angle_AXB_eq : angle_AXB = 180)
               (angle_AX_eq : angle_AX = 70)
               (angle_BX_eq : angle_BX = 40)
               (angle_CY_eq : angle_CY = 130) : 
               ∃ y : ℝ, y = 60 :=
by
  sorry -- The actual proof goes here.

end find_y_l89_89339


namespace boys_count_l89_89440

variable (B G : ℕ)

theorem boys_count (h1 : B + G = 466) (h2 : G = B + 212) : B = 127 := by
  sorry

end boys_count_l89_89440


namespace polynomial_solution_l89_89434

noncomputable def p (x : ℝ) := 2 * Real.sqrt 3 * x^4 - 6

theorem polynomial_solution (x : ℝ) : 
  (p (x^4) - p (x^4 - 3) = (p x)^3 - 18) :=
by
  sorry

end polynomial_solution_l89_89434


namespace time_to_complete_job_l89_89223

-- Define the conditions
variables {A B : ℕ} -- Efficiencies of A and B

-- Assume B's efficiency is 100 units, and A is 130 units.
def efficiency_A : ℕ := 130
def efficiency_B : ℕ := 100

-- Given: A can complete the job in 23 days
def days_A : ℕ := 23

-- Compute total work W. Since A can complete the job in 23 days and its efficiency is 130 units/day:
def total_work : ℕ := efficiency_A * days_A

-- Combined efficiency of A and B
def combined_efficiency : ℕ := efficiency_A + efficiency_B

-- Determine the time taken by A and B working together
def time_A_B_together : ℕ := total_work / combined_efficiency

-- Prove that the time A and B working together is 13 days
theorem time_to_complete_job : time_A_B_together = 13 :=
by
  sorry -- Proof is omitted as per instructions

end time_to_complete_job_l89_89223


namespace farm_horses_cows_difference_l89_89958

-- Definitions based on provided conditions
def initial_ratio_horses_to_cows (horses cows : ℕ) : Prop := 5 * cows = horses
def transaction (horses cows sold bought : ℕ) : Prop :=
  horses - sold = 5 * cows - 15 ∧ cows + bought = cows + 15

-- Definitions to represent the ratios
def pre_transaction_ratio (horses cows : ℕ) : Prop := initial_ratio_horses_to_cows horses cows
def post_transaction_ratio (horses cows : ℕ) (sold bought : ℕ) : Prop :=
  transaction horses cows sold bought ∧ 7 * (horses - sold) = 17 * (cows + bought)

-- Statement of the theorem
theorem farm_horses_cows_difference :
  ∀ (horses cows : ℕ), 
    pre_transaction_ratio horses cows → 
    post_transaction_ratio horses cows 15 15 →
    (horses - 15) - (cows + 15) = 50 :=
by
  intros horses cows pre_ratio post_ratio
  sorry

end farm_horses_cows_difference_l89_89958


namespace min_students_l89_89456

theorem min_students (S a b c : ℕ) (h1 : 3 * a > S) (h2 : 10 * b > 3 * S) (h3 : 11 * c > 4 * S) (h4 : S = a + b + c) : S ≥ 173 :=
by
  sorry

end min_students_l89_89456


namespace prob_green_is_correct_l89_89713

-- Define the probability of picking any container
def prob_pick_container : ℚ := 1 / 4

-- Define the probability of drawing a green ball from each container
def prob_green_A : ℚ := 6 / 10
def prob_green_B : ℚ := 3 / 10
def prob_green_C : ℚ := 3 / 10
def prob_green_D : ℚ := 5 / 10

-- Define the individual probabilities for a green ball, accounting for container selection
def prob_green_given_A : ℚ := prob_pick_container * prob_green_A
def prob_green_given_B : ℚ := prob_pick_container * prob_green_B
def prob_green_given_C : ℚ := prob_pick_container * prob_green_C
def prob_green_given_D : ℚ := prob_pick_container * prob_green_D

-- Calculate the total probability of selecting a green ball
def prob_green_total : ℚ := prob_green_given_A + prob_green_given_B + prob_green_given_C + prob_green_given_D

-- Theorem statement: The probability of selecting a green ball is 17/40
theorem prob_green_is_correct : prob_green_total = 17 / 40 :=
by
  -- Proof will be provided here.
  sorry

end prob_green_is_correct_l89_89713


namespace product_even_if_sum_odd_l89_89321

theorem product_even_if_sum_odd (a b : ℤ) (h : (a + b) % 2 = 1) : (a * b) % 2 = 0 :=
sorry

end product_even_if_sum_odd_l89_89321


namespace simplify_expression_l89_89259

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 - 2) / x) * ((y^2 - 2) / y) - ((x^2 + 2) / y) * ((y^2 + 2) / x) = -4 * (x / y + y / x) :=
by
  sorry

end simplify_expression_l89_89259


namespace artist_paints_37_sq_meters_l89_89123

-- Define the structure of the sculpture
def top_layer : ℕ := 1
def middle_layer : ℕ := 5
def bottom_layer : ℕ := 11
def edge_length : ℕ := 1

-- Define the exposed surface areas
def exposed_surface_top_layer := 5 * top_layer
def exposed_surface_middle_layer := 1 * 5 + 4 * 4
def exposed_surface_bottom_layer := bottom_layer

-- Calculate the total exposed surface area
def total_exposed_surface_area := exposed_surface_top_layer + exposed_surface_middle_layer + exposed_surface_bottom_layer

-- The final theorem statement
theorem artist_paints_37_sq_meters (hyp1 : top_layer = 1)
  (hyp2 : middle_layer = 5)
  (hyp3 : bottom_layer = 11)
  (hyp4 : edge_length = 1)
  : total_exposed_surface_area = 37 := 
by
  sorry

end artist_paints_37_sq_meters_l89_89123


namespace sum_of_odd_integers_l89_89873

theorem sum_of_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : (n * n = 4900) :=
by sorry

end sum_of_odd_integers_l89_89873


namespace acme_cheaper_min_shirts_l89_89265

theorem acme_cheaper_min_shirts :
  ∃ x : ℕ, 60 + 11 * x < 10 + 16 * x ∧ x = 11 :=
by {
  sorry
}

end acme_cheaper_min_shirts_l89_89265


namespace total_keys_needed_l89_89642

-- Definitions based on given conditions
def num_complexes : ℕ := 2
def num_apartments_per_complex : ℕ := 12
def keys_per_lock : ℕ := 3
def num_locks_per_apartment : ℕ := 1

-- Theorem stating the required number of keys
theorem total_keys_needed : 
  (num_complexes * num_apartments_per_complex * keys_per_lock = 72) :=
by
  sorry

end total_keys_needed_l89_89642


namespace unique_solution_xp_eq_1_l89_89404

theorem unique_solution_xp_eq_1 (x p q : ℕ) (h1 : x ≥ 2) (h2 : p ≥ 2) (h3 : q ≥ 2):
  ((x + 1)^p - x^q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by 
  sorry

end unique_solution_xp_eq_1_l89_89404


namespace number_of_football_players_l89_89737

theorem number_of_football_players
  (cricket_players : ℕ)
  (hockey_players : ℕ)
  (softball_players : ℕ)
  (total_players : ℕ) :
  cricket_players = 22 →
  hockey_players = 15 →
  softball_players = 19 →
  total_players = 77 →
  total_players - (cricket_players + hockey_players + softball_players) = 21 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_football_players_l89_89737


namespace find_larger_number_l89_89684

theorem find_larger_number (a b : ℤ) (h1 : a + b = 27) (h2 : a - b = 5) : a = 16 := by
  sorry

end find_larger_number_l89_89684


namespace isabella_hair_length_l89_89075

theorem isabella_hair_length (original : ℝ) (increase_percent : ℝ) (new_length : ℝ) 
    (h1 : original = 18) (h2 : increase_percent = 0.75) 
    (h3 : new_length = original + increase_percent * original) : 
    new_length = 31.5 := by sorry

end isabella_hair_length_l89_89075


namespace contradiction_proof_l89_89975

theorem contradiction_proof (a b c : ℝ) (h : (a⁻¹ * b⁻¹ * c⁻¹) > 0) : (a ≤ 1) ∧ (b ≤ 1) ∧ (c ≤ 1) → False :=
sorry

end contradiction_proof_l89_89975


namespace min_expression_value_l89_89695

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  a^2 + 4 * a * b + 8 * b^2 + 10 * b * c + 3 * c^2

theorem min_expression_value (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 3) :
  minimum_value a b c ≥ 27 :=
sorry

end min_expression_value_l89_89695


namespace initial_cakes_count_l89_89941

theorem initial_cakes_count (f : ℕ) (a b : ℕ) 
  (condition1 : f = 5)
  (condition2 : ∀ i, i ∈ Finset.range f → a = 4)
  (condition3 : ∀ i, i ∈ Finset.range f → b = 20 / 2)
  (condition4 : f * a = 2 * b) : 
  b = 40 := 
by
  sorry

end initial_cakes_count_l89_89941


namespace find_k_values_l89_89969

theorem find_k_values (a b k : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b % a = 0) 
  (h₄ : ∀ (m : ℤ), (a : ℤ) = k * (a : ℤ) + m ∧ (8 * (b : ℤ)) = k * (b : ℤ) + m) :
  k = 9 ∨ k = 15 :=
by
  { sorry }

end find_k_values_l89_89969


namespace H_iterated_l89_89840

variable (H : ℝ → ℝ)

-- Conditions as hypotheses
axiom H_2 : H 2 = -4
axiom H_neg4 : H (-4) = 6
axiom H_6 : H 6 = 6

-- The theorem we want to prove
theorem H_iterated (H : ℝ → ℝ) (h1 : H 2 = -4) (h2 : H (-4) = 6) (h3 : H 6 = 6) : 
  H (H (H (H (H 2)))) = 6 := by
  sorry

end H_iterated_l89_89840


namespace find_x_l89_89704

theorem find_x (x : ℤ) (h1 : 5 < x) (h2 : x < 21) (h3 : 7 < x) (h4 : x < 18) (h5 : 2 < x) (h6 : x < 13) (h7 : 9 < x) (h8 : x < 12) (h9 : x < 12) :
  x = 10 :=
sorry

end find_x_l89_89704


namespace johns_money_left_l89_89053

def dog_walking_days_in_april := 26
def earnings_per_day := 10
def money_spent_on_books := 50
def money_given_to_sister := 50

theorem johns_money_left : (dog_walking_days_in_april * earnings_per_day) - (money_spent_on_books + money_given_to_sister) = 160 := 
by
  sorry

end johns_money_left_l89_89053


namespace square_area_l89_89786

theorem square_area {d : ℝ} (h : d = 12 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 144 ∧ ( ∃ s : ℝ, s = d / Real.sqrt 2 ∧ A = s^2 ) :=
by
  sorry

end square_area_l89_89786


namespace valid_elixir_combinations_l89_89296

theorem valid_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let incompatible_herbs := 3
  let incompatible_crystals := 2
  let total_combinations := herbs * crystals
  let incompatible_combinations := incompatible_herbs * incompatible_crystals
  total_combinations - incompatible_combinations = 18 :=
by
  sorry

end valid_elixir_combinations_l89_89296


namespace optimal_response_l89_89823

theorem optimal_response (n : ℕ) (m : ℕ) (s : ℕ) (a_1 : ℕ) (a_2 : ℕ -> ℕ) (a_opt : ℕ):
  n = 100 → 
  m = 107 →
  (∀ i, i ≥ 1 ∧ i ≤ 99 → a_2 i = a_opt) →
  a_1 = 7 :=
by
  sorry

end optimal_response_l89_89823


namespace comparison_abc_l89_89913

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem comparison_abc : a > c ∧ c > b :=
by
  sorry

end comparison_abc_l89_89913


namespace real_part_of_complex_pow_l89_89207

open Complex

theorem real_part_of_complex_pow (a b : ℝ) : a = 1 → b = -2 → (realPart ((a : ℂ) + (b : ℂ) * Complex.I)^5) = 41 :=
by
  sorry

end real_part_of_complex_pow_l89_89207


namespace solve_equation_l89_89405

def equation_params (a x : ℝ) : Prop :=
  a * (1 / (Real.cos x) - Real.tan x) = 1

def valid_solutions (a x : ℝ) (k : ℤ) : Prop :=
  (a ≠ 0) ∧ (Real.cos x ≠ 0) ∧ (
    (|a| ≥ 1 ∧ x = Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k) ∨
    ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) ∧ x = - Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k)
  )

theorem solve_equation (a x : ℝ) (k : ℤ) :
  equation_params a x → valid_solutions a x k := by
  sorry

end solve_equation_l89_89405


namespace seventh_root_binomial_expansion_l89_89402

theorem seventh_root_binomial_expansion : 
  (∃ (n : ℕ), n = 137858491849 ∧ (∃ (k : ℕ), n = (10 + 1) ^ k)) →
  (∃ a, a = 11 ∧ 11 ^ 7 = 137858491849) := 
by {
  sorry 
}

end seventh_root_binomial_expansion_l89_89402


namespace probability_of_all_heads_or_tails_l89_89560

def num_favorable_outcomes : ℕ := 2

def total_outcomes : ℕ := 2 ^ 5

def probability_all_heads_or_tails : ℚ := num_favorable_outcomes / total_outcomes

theorem probability_of_all_heads_or_tails :
  probability_all_heads_or_tails = 1 / 16 := by
  -- Proof goes here
  sorry

end probability_of_all_heads_or_tails_l89_89560


namespace trucks_needed_for_coal_transport_l89_89544

def number_of_trucks (total_coal : ℕ) (capacity_per_truck : ℕ) (x : ℕ) : Prop :=
  capacity_per_truck * x = total_coal

theorem trucks_needed_for_coal_transport :
  number_of_trucks 47500 2500 19 :=
by
  sorry

end trucks_needed_for_coal_transport_l89_89544


namespace norris_money_left_l89_89302

-- Defining the conditions
def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def dec_savings : ℕ := 35
def jan_savings : ℕ := 40

def initial_savings : ℕ := sept_savings + oct_savings + nov_savings + dec_savings + jan_savings
def interest_rate : ℝ := 0.02

def total_interest : ℝ :=
  sept_savings * interest_rate + 
  (sept_savings + oct_savings) * interest_rate + 
  (sept_savings + oct_savings + nov_savings) * interest_rate +
  (sept_savings + oct_savings + nov_savings + dec_savings) * interest_rate

def total_savings_with_interest : ℝ := initial_savings + total_interest
def hugo_owes_norris : ℕ := 20 - 10

-- The final statement to prove Norris' total amount of money
theorem norris_money_left : total_savings_with_interest + hugo_owes_norris = 175.76 := by
  sorry

end norris_money_left_l89_89302


namespace hyperbola_foci_coordinates_l89_89725

theorem hyperbola_foci_coordinates :
  ∀ (x y : ℝ), x^2 - (y^2 / 3) = 1 → (∃ c : ℝ, c = 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_coordinates_l89_89725


namespace bill_soaking_time_l89_89668

theorem bill_soaking_time 
  (G M : ℕ) 
  (h₁ : M = G + 7) 
  (h₂ : 3 * G + M = 19) : 
  G = 3 := 
by {
  sorry
}

end bill_soaking_time_l89_89668


namespace butterflies_left_l89_89171

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l89_89171


namespace minimum_value_l89_89463

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum : a + b + c = 3)

theorem minimum_value : 
  (1 / (3 * a + 5 * b)) + (1 / (3 * b + 5 * c)) + (1 / (3 * c + 5 * a)) ≥ 9 / 8 :=
by
  sorry

end minimum_value_l89_89463


namespace find_a_l89_89410

def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

theorem find_a (a : ℝ) (h : F a 3 4 = F a 2 5) : a = 1 / 2 :=
by
  sorry

end find_a_l89_89410


namespace range_of_absolute_difference_l89_89045

theorem range_of_absolute_difference : (∃ x : ℝ, y = |x + 4| - |x - 5|) → y ∈ [-9, 9] :=
sorry

end range_of_absolute_difference_l89_89045


namespace find_xy_l89_89126

theorem find_xy (x y : ℤ) 
  (h1 : (2 + 11 + 6 + x) / 4 = (14 + 9 + y) / 3) : 
  x = -35 ∧ y = -35 :=
by 
  sorry

end find_xy_l89_89126


namespace find_cd_l89_89224

noncomputable def period := (3 / 4) * Real.pi
noncomputable def x_value := (1 / 8) * Real.pi
noncomputable def y_value := 3
noncomputable def tangent_value := Real.tan (Real.pi / 6) -- which is 1 / sqrt(3)
noncomputable def c_value := 3 * Real.sqrt 3

theorem find_cd (c d : ℝ) 
  (h_period : d = 4 / 3) 
  (h_point : y_value = c * Real.tan (d * x_value)) :
  c * d = 4 * Real.sqrt 3 := 
sorry

end find_cd_l89_89224


namespace calculate_expression_l89_89115

theorem calculate_expression : 
  |(-3)| - 2 * Real.tan (Real.pi / 4) + (-1:ℤ)^(2023) - (Real.sqrt 3 - Real.pi)^(0:ℤ) = -1 :=
  by
  sorry

end calculate_expression_l89_89115


namespace smaller_rectangle_ratio_l89_89406

theorem smaller_rectangle_ratio
  (length_large : ℝ) (width_large : ℝ) (area_small : ℝ)
  (h_length : length_large = 40)
  (h_width : width_large = 20)
  (h_area : area_small = 200) : 
  ∃ r : ℝ, (length_large * r) * (width_large * r) = area_small ∧ r = 0.5 :=
by
  sorry

end smaller_rectangle_ratio_l89_89406


namespace system1_solution_system2_solution_l89_89239

-- System (1)
theorem system1_solution (x y : ℝ) (h1 : x + y = 1) (h2 : 3 * x + y = 5) : x = 2 ∧ y = -1 := sorry

-- System (2)
theorem system2_solution (x y : ℝ) (h1 : 3 * (x - 1) + 4 * y = 1) (h2 : 2 * x + 3 * (y + 1) = 2) : x = 16 ∧ y = -11 := sorry

end system1_solution_system2_solution_l89_89239


namespace min_dot_product_l89_89307

noncomputable def vec_a (m : ℝ) : ℝ × ℝ := (1 + 2^m, 1 - 2^m)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (4^m - 3, 4^m + 5)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem min_dot_product : ∃ m : ℝ, dot_product (vec_a m) (vec_b m) = -6 := by
  sorry

end min_dot_product_l89_89307


namespace square_of_cube_of_smallest_prime_l89_89299

def smallest_prime : Nat := 2

theorem square_of_cube_of_smallest_prime :
  ((smallest_prime ^ 3) ^ 2) = 64 := by
  sorry

end square_of_cube_of_smallest_prime_l89_89299


namespace first_player_guaranteed_win_l89_89054

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 ^ k

theorem first_player_guaranteed_win (n : ℕ) (h : n > 1) : 
  ¬ is_power_of_two n ↔ ∃ m : ℕ, 1 ≤ m ∧ m < n ∧ (∀ k : ℕ, m ≤ k + 1 → ∀ t, t ≤ m → ∃ r, r = k + 1 ∧ r <= m) → 
                                (∃ l : ℕ, (l = 1) → true) :=
sorry

end first_player_guaranteed_win_l89_89054


namespace product_of_last_two_digits_of_divisible_by_6_l89_89583

-- Definitions
def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def sum_of_last_two_digits (n : ℤ) (a b : ℤ) : Prop := (n % 100) = 10 * a + b

-- Theorem statement
theorem product_of_last_two_digits_of_divisible_by_6 (x a b : ℤ)
  (h1 : is_divisible_by_6 x)
  (h2 : sum_of_last_two_digits x a b)
  (h3 : a + b = 15) :
  (a * b = 54 ∨ a * b = 56) := 
sorry

end product_of_last_two_digits_of_divisible_by_6_l89_89583


namespace g_value_range_l89_89628

noncomputable def g (x y z : ℝ) : ℝ :=
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_value_range (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  (3/2 : ℝ) ≤ g x y z ∧ g x y z ≤ (3 : ℝ) / 2 := 
sorry

end g_value_range_l89_89628


namespace ajay_total_gain_l89_89124

noncomputable def ajay_gain : ℝ :=
  let cost1 := 15 * 14.50
  let cost2 := 10 * 13
  let total_cost := cost1 + cost2
  let total_weight := 15 + 10
  let selling_price := total_weight * 15
  selling_price - total_cost

theorem ajay_total_gain :
  ajay_gain = 27.50 := by
  sorry

end ajay_total_gain_l89_89124


namespace find_a_l89_89323

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := 
by 
  sorry

end find_a_l89_89323


namespace milo_running_distance_l89_89951

theorem milo_running_distance
  (run_speed skateboard_speed cory_speed : ℕ)
  (h1 : skateboard_speed = 2 * run_speed)
  (h2 : cory_speed = 2 * skateboard_speed)
  (h3 : cory_speed = 12) :
  run_speed * 2 = 6 :=
by
  sorry

end milo_running_distance_l89_89951


namespace difference_abs_eq_200_l89_89085

theorem difference_abs_eq_200 (x y : ℤ) (h1 : x + y = 250) (h2 : y = 225) : |x - y| = 200 := sorry

end difference_abs_eq_200_l89_89085


namespace zinc_to_copper_ratio_l89_89854

theorem zinc_to_copper_ratio (total_weight zinc_weight copper_weight : ℝ) 
  (h1 : total_weight = 64) 
  (h2 : zinc_weight = 28.8) 
  (h3 : copper_weight = total_weight - zinc_weight) : 
  (zinc_weight / 0.4) / (copper_weight / 0.4) = 9 / 11 :=
by
  sorry

end zinc_to_copper_ratio_l89_89854


namespace soap_box_missing_dimension_l89_89773

theorem soap_box_missing_dimension
  (x : ℕ) -- The missing dimension of the soap box
  (Volume_carton : ℕ := 25 * 48 * 60)
  (Volume_soap_box : ℕ := 8 * x * 5)
  (Max_soap_boxes : ℕ := 300)
  (condition : Max_soap_boxes * Volume_soap_box ≤ Volume_carton) :
  x ≤ 6 := by
sorry

end soap_box_missing_dimension_l89_89773


namespace solve_equation_l89_89021

theorem solve_equation : ∀ x : ℝ, (3 * (x - 2) + 1 = x - (2 * x - 1)) → x = 3 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l89_89021


namespace find_original_denominator_l89_89588

variable (d : ℕ)

theorem find_original_denominator
  (h1 : ∀ n : ℕ, n = 3)
  (h2 : 3 + 7 = 10)
  (h3 : (10 : ℕ) = 1 * (d + 7) / 3) :
  d = 23 := by
  sorry

end find_original_denominator_l89_89588


namespace triangle_inequality_l89_89138

theorem triangle_inequality (S : Finset (ℕ × ℕ)) (m n : ℕ) (hS : S.card = m)
  (h_ab : ∀ (a b : ℕ), (a, b) ∈ S → (1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n ∧ a ≠ b)) :
  ∃ (t : Finset (ℕ × ℕ × ℕ)),
    (t.card ≥ (4 * m / (3 * n)) * (m - (n^2) / 4)) ∧
    ∀ (a b c : ℕ), (a, b, c) ∈ t → (a, b) ∈ S ∧ (b, c) ∈ S ∧ (c, a) ∈ S := by
  sorry

end triangle_inequality_l89_89138


namespace Adam_final_amount_l89_89718

def initial_amount : ℝ := 5.25
def spent_on_game : ℝ := 2.30
def spent_on_snacks : ℝ := 1.75
def found_dollar : ℝ := 1.00
def allowance : ℝ := 5.50

theorem Adam_final_amount :
  (initial_amount - spent_on_game - spent_on_snacks + found_dollar + allowance) = 7.70 :=
by
  sorry

end Adam_final_amount_l89_89718


namespace parabola_directrix_l89_89955

theorem parabola_directrix (y x : ℝ) (h : y = x^2) : 4 * y + 1 = 0 :=
sorry

end parabola_directrix_l89_89955


namespace geom_seq_sum_first_eight_l89_89196

def geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

def sum_geom_seq (a₀ r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then a₀ * n else a₀ * (1 - r^n) / (1 - r)

theorem geom_seq_sum_first_eight :
  let a₀ := 1 / 3
  let r := 1 / 3
  let n := 8
  sum_geom_seq a₀ r n = 3280 / 6561 :=
by
  sorry

end geom_seq_sum_first_eight_l89_89196


namespace work_completion_time_l89_89004

theorem work_completion_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : B + C = 1 / 15)
  (h3 : C + A = 1 / 20) :
  1 / (A + B + C) = 10 :=
by
  sorry

end work_completion_time_l89_89004


namespace parabola_equation_origin_l89_89652

theorem parabola_equation_origin (x0 : ℝ) :
  ∃ (p : ℝ), (p > 0) ∧ (x0^2 = 2 * p * 2) ∧ (p = 2) ∧ (x0^2 = 4 * 2) := 
by 
  sorry

end parabola_equation_origin_l89_89652


namespace rectangle_y_coordinate_l89_89493

theorem rectangle_y_coordinate (x1 x2 y1 A : ℝ) (h1 : x1 = -8) (h2 : x2 = 1) (h3 : y1 = 1) (h4 : A = 72)
    (hL : x2 - x1 = 9) (hA : A = 9 * (y - y1)) :
    (y = 9) :=
by
  sorry

end rectangle_y_coordinate_l89_89493


namespace solve_inequality_l89_89831

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def given_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x >= 0 → f x = x^3 - 8

theorem solve_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_given : given_function f) :
  {x | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  sorry

end solve_inequality_l89_89831


namespace find_a_l89_89098

noncomputable def f (a x : ℝ) : ℝ := a^x - 4 * a + 3

theorem find_a (H : ∃ (a : ℝ), ∃ (x y : ℝ), f a x = y ∧ f y x = a ∧ x = 2 ∧ y = -1): ∃ a : ℝ, a = 2 :=
by
  obtain ⟨a, x, y, hx, hy, hx2, hy1⟩ := H
  --skipped proof
  sorry

end find_a_l89_89098


namespace complex_fraction_simplification_l89_89117

theorem complex_fraction_simplification (i : ℂ) (hi : i^2 = -1) : 
  ((2 - i) / (1 + 4 * i)) = (-2 / 17 - (9 / 17) * i) :=
  sorry

end complex_fraction_simplification_l89_89117


namespace calc_pow_product_l89_89084

theorem calc_pow_product : (0.25 ^ 2023) * (4 ^ 2023) = 1 := 
  by 
  sorry

end calc_pow_product_l89_89084


namespace determine_range_of_a_l89_89111

noncomputable def f (x a : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

noncomputable def g (x a : ℝ) : ℝ := f x a - 2*x

theorem determine_range_of_a (a : ℝ) :
  (∀ x, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) →
  (-1 ≤ a ∧ a < 2) :=
by
  intro h
  sorry

end determine_range_of_a_l89_89111


namespace find_f6_l89_89208

noncomputable def f : ℝ → ℝ :=
sorry

theorem find_f6 (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
                (h2 : f 5 = 6) :
  f 6 = 36 / 5 :=
sorry

end find_f6_l89_89208


namespace total_settings_weight_l89_89067

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l89_89067


namespace not_possible_127_points_l89_89387

theorem not_possible_127_points (n_correct n_unanswered n_incorrect : ℕ) :
  n_correct + n_unanswered + n_incorrect = 25 →
  127 ≠ 5 * n_correct + 2 * n_unanswered - n_incorrect :=
by
  intro h_total
  sorry

end not_possible_127_points_l89_89387


namespace gcd_of_6Tn2_and_nplus1_eq_2_l89_89849

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end gcd_of_6Tn2_and_nplus1_eq_2_l89_89849


namespace problem_statement_l89_89926

open Set

noncomputable def U := ℝ

def A : Set ℝ := { x | 0 < 2 * x + 4 ∧ 2 * x + 4 < 10 }
def B : Set ℝ := { x | x < -4 ∨ x > 2 }
def C (a : ℝ) (h : a < 0) : Set ℝ := { x | x^2 - 4 * a * x + 3 * a^2 < 0 }

theorem problem_statement (a : ℝ) (ha : a < 0) :
    A ∪ B = { x | x < -4 ∨ x > -2 } ∧
    compl (A ∪ B) ⊆ C a ha → -2 < a ∧ a < -4 / 3 :=
sorry

end problem_statement_l89_89926


namespace unique_solution_mnk_l89_89113

theorem unique_solution_mnk :
  ∀ (m n k : ℕ), 3^n + 4^m = 5^k → (m, n, k) = (0, 1, 1) :=
by
  intros m n k h
  sorry

end unique_solution_mnk_l89_89113


namespace verify_toothpick_count_l89_89735

def toothpick_problem : Prop :=
  let L := 45
  let W := 25
  let Mv := 8
  let Mh := 5
  -- Calculate the total number of vertical toothpicks
  let verticalToothpicks := (L + 1 - Mv) * W
  -- Calculate the total number of horizontal toothpicks
  let horizontalToothpicks := (W + 1 - Mh) * L
  -- Calculate the total number of toothpicks
  let totalToothpicks := verticalToothpicks + horizontalToothpicks
  -- Ensure the total matches the expected result
  totalToothpicks = 1895

theorem verify_toothpick_count : toothpick_problem :=
by
  sorry

end verify_toothpick_count_l89_89735


namespace solution_set_quadratic_inequality_l89_89295

theorem solution_set_quadratic_inequality (a b : ℝ) (h1 : a < 0)
    (h2 : ∀ x, ax^2 - bx - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
    ∀ x, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 := 
by
  sorry

end solution_set_quadratic_inequality_l89_89295


namespace intersection_M_N_l89_89468

def M := {x : ℝ | x < 1}

def N := {y : ℝ | ∃ x : ℝ, y = Real.exp x}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
  sorry

end intersection_M_N_l89_89468


namespace find_a_l89_89222

variable {x y a : ℤ}

theorem find_a (h1 : 3 * x + y = 1 + 3 * a) (h2 : x + 3 * y = 1 - a) (h3 : x + y = 0) : a = -1 := 
sorry

end find_a_l89_89222


namespace amount_left_after_expenses_l89_89320

namespace GirlScouts

def totalEarnings : ℝ := 30
def poolEntryCosts : ℝ :=
  5 * 3.5 + 3 * 2.0 + 2 * 1.0
def transportationCosts : ℝ :=
  6 * 1.5 + 4 * 0.75
def snackCosts : ℝ :=
  3 * 3.0 + 4 * 2.5 + 3 * 2.0
def totalExpenses : ℝ :=
  poolEntryCosts + transportationCosts + snackCosts
def amountLeft : ℝ :=
  totalEarnings - totalExpenses

theorem amount_left_after_expenses :
  amountLeft = -32.5 :=
by
  sorry

end GirlScouts

end amount_left_after_expenses_l89_89320


namespace problem_l89_89347

variable (f g h : ℕ → ℕ)

-- Define the conditions as hypotheses
axiom h1 : ∀ (n m : ℕ), n ≠ m → h n ≠ h m
axiom h2 : ∀ y, ∃ x, g x = y
axiom h3 : ∀ n, f n = g n - h n + 1

theorem problem : ∀ n, f n = 1 := 
by 
  sorry

end problem_l89_89347


namespace product_of_b_l89_89291

noncomputable def b_product : ℤ :=
  let y1 := 3
  let y2 := 8
  let x1 := 2
  let l := y2 - y1 -- Side length of the square
  let b₁ := x1 - l -- One possible value of b
  let b₂ := x1 + l -- Another possible value of b
  b₁ * b₂ -- Product of possible values of b

theorem product_of_b :
  b_product = -21 := by
  sorry

end product_of_b_l89_89291


namespace delivery_payment_l89_89439

-- Define the problem conditions and the expected outcome
theorem delivery_payment 
    (deliveries_Oula : ℕ) 
    (deliveries_Tona : ℕ) 
    (difference_in_pay : ℝ) 
    (P : ℝ) 
    (H1 : deliveries_Oula = 96) 
    (H2 : deliveries_Tona = 72) 
    (H3 : difference_in_pay = 2400) :
    96 * P - 72 * P = 2400 → P = 100 :=
by
  intro h1
  sorry

end delivery_payment_l89_89439


namespace L_shape_area_and_perimeter_l89_89525

def rectangle1_length := 0.5
def rectangle1_width := 0.3
def rectangle2_length := 0.2
def rectangle2_width := 0.5

def area_rectangle1 := rectangle1_length * rectangle1_width
def area_rectangle2 := rectangle2_length * rectangle2_width
def total_area := area_rectangle1 + area_rectangle2

def perimeter_L_shape := rectangle1_length + rectangle1_width + rectangle1_width + rectangle2_length + rectangle2_length + rectangle2_width

theorem L_shape_area_and_perimeter :
  total_area = 0.25 ∧ perimeter_L_shape = 2.0 :=
by
  sorry

end L_shape_area_and_perimeter_l89_89525


namespace problems_per_hour_l89_89529

theorem problems_per_hour :
  ∀ (mathProblems spellingProblems totalHours problemsPerHour : ℕ), 
    mathProblems = 36 →
    spellingProblems = 28 →
    totalHours = 8 →
    (mathProblems + spellingProblems) / totalHours = problemsPerHour →
    problemsPerHour = 8 :=
by
  intros
  subst_vars
  sorry

end problems_per_hour_l89_89529


namespace geometric_sequence_sum_l89_89248

/-- Let {a_n} be a geometric sequence with positive common ratio, a_1 = 2, and a_3 = a_2 + 4.
    Prove the general formula for a_n is 2^n, and the sum of the first n terms, S_n, of the sequence { (2n+1)a_n }
    is (2n-1) * 2^(n+1) + 2. -/
theorem geometric_sequence_sum
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h3 : a 3 = a 2 + 4) :
  (∀ n, a n = 2^n) ∧
  (∀ S : ℕ → ℕ, ∀ n, S n = (2 * n - 1) * 2 ^ (n + 1) + 2) :=
by sorry

end geometric_sequence_sum_l89_89248


namespace necessary_not_sufficient_condition_l89_89203

theorem necessary_not_sufficient_condition (x : ℝ) : 
  x^2 - 2 * x - 3 < 0 → -2 < x ∧ x < 3 :=
by  
  sorry

end necessary_not_sufficient_condition_l89_89203


namespace class_tree_total_l89_89753

theorem class_tree_total
  (trees_A : ℕ)
  (trees_B : ℕ)
  (hA : trees_A = 8)
  (hB : trees_B = 7)
  : trees_A + trees_B = 15 := 
by
  sorry

end class_tree_total_l89_89753


namespace frank_money_l89_89219

-- Define the initial amount, expenses, and incomes as per the conditions
def initialAmount : ℕ := 11
def spentOnGame : ℕ := 3
def spentOnKeychain : ℕ := 2
def receivedFromAlice : ℕ := 4
def allowance : ℕ := 14
def spentOnBusTicket : ℕ := 5

-- Define the total money left for Frank
def finalAmount (initial : ℕ) (game : ℕ) (keychain : ℕ) (gift : ℕ) (allowance : ℕ) (bus : ℕ) : ℕ :=
  initial - game - keychain + gift + allowance - bus

-- Define the theorem stating that the final amount is 19
theorem frank_money : finalAmount initialAmount spentOnGame spentOnKeychain receivedFromAlice allowance spentOnBusTicket = 19 :=
by
  sorry

end frank_money_l89_89219


namespace martha_total_clothes_l89_89462

def jackets_purchased : ℕ := 4
def tshirts_purchased : ℕ := 9
def jackets_free : ℕ := jackets_purchased / 2
def tshirts_free : ℕ := tshirts_purchased / 3
def total_jackets : ℕ := jackets_purchased + jackets_free
def total_tshirts : ℕ := tshirts_purchased + tshirts_free

theorem martha_total_clothes : total_jackets + total_tshirts = 18 := by
  sorry

end martha_total_clothes_l89_89462


namespace number_of_trees_l89_89192

-- Define the yard length and the distance between consecutive trees
def yard_length : ℕ := 300
def distance_between_trees : ℕ := 12

-- Prove that the number of trees planted in the garden is 26
theorem number_of_trees (yard_length distance_between_trees : ℕ) 
  (h1 : yard_length = 300) (h2 : distance_between_trees = 12) : 
  ∃ n : ℕ, n = 26 :=
by
  sorry

end number_of_trees_l89_89192


namespace cistern_filling_time_l89_89948

/-
Given the following conditions:
- Pipe A fills the cistern in 10 hours.
- Pipe B fills the cistern in 12 hours.
- Exhaust pipe C drains the cistern in 15 hours.
- Exhaust pipe D drains the cistern in 20 hours.

Prove that if all four pipes are opened simultaneously, the cistern will be filled in 15 hours.
-/

theorem cistern_filling_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 12
  let rate_C := -(1 / 15)
  let rate_D := -(1 / 20)
  let combined_rate := rate_A + rate_B + rate_C + rate_D
  let time_to_fill := 1 / combined_rate
  time_to_fill = 15 :=
by 
  sorry

end cistern_filling_time_l89_89948


namespace mr_willson_friday_work_time_l89_89473

theorem mr_willson_friday_work_time :
  let monday := 3 / 4
  let tuesday := 1 / 2
  let wednesday := 2 / 3
  let thursday := 5 / 6
  let total_work := 4
  let time_monday_to_thursday := monday + tuesday + wednesday + thursday
  let time_friday := total_work - time_monday_to_thursday
  time_friday * 60 = 75 :=
by
  sorry

end mr_willson_friday_work_time_l89_89473


namespace yellow_paint_percentage_l89_89610

theorem yellow_paint_percentage 
  (total_gallons_mixture : ℝ)
  (light_green_paint_gallons : ℝ)
  (dark_green_paint_gallons : ℝ)
  (dark_green_paint_percentage : ℝ)
  (mixture_percentage : ℝ)
  (X : ℝ) 
  (h_total_gallons : total_gallons_mixture = light_green_paint_gallons + dark_green_paint_gallons)
  (h_dark_green_paint_yellow_amount : dark_green_paint_gallons * dark_green_paint_percentage = 1.66666666667 * 0.4)
  (h_mixture_yellow_amount : total_gallons_mixture * mixture_percentage = 5 * X + 1.66666666667 * 0.4) :
  X = 0.2 :=
by
  sorry

end yellow_paint_percentage_l89_89610


namespace votes_cast_l89_89777

theorem votes_cast (candidate_percentage : ℝ) (vote_difference : ℝ) (total_votes : ℝ) 
  (h1 : candidate_percentage = 0.30) 
  (h2 : vote_difference = 1760) 
  (h3 : total_votes = vote_difference / (1 - 2 * candidate_percentage)) 
  : total_votes = 4400 := by
  sorry

end votes_cast_l89_89777


namespace parallelogram_perimeter_eq_60_l89_89390

-- Given conditions from the problem
variables (P Q R M N O : Type*)
variables (PQ PR QR PM MN NO PO : ℝ)
variables {PQ_eq_PR : PQ = PR}
variables {PQ_val : PQ = 30}
variables {PR_val : PR = 30}
variables {QR_val : QR = 28}
variables {MN_parallel_PR : true}  -- Parallel condition we can treat as true for simplification
variables {NO_parallel_PQ : true}  -- Another parallel condition treated as true

-- Statement of the problem to be proved
theorem parallelogram_perimeter_eq_60 :
  PM + MN + NO + PO = 60 :=
sorry

end parallelogram_perimeter_eq_60_l89_89390


namespace combined_weight_of_contents_l89_89121

theorem combined_weight_of_contents
    (weight_pencil : ℝ := 28.3)
    (weight_eraser : ℝ := 15.7)
    (weight_paperclip : ℝ := 3.5)
    (weight_stapler : ℝ := 42.2)
    (num_pencils : ℕ := 5)
    (num_erasers : ℕ := 3)
    (num_paperclips : ℕ := 4)
    (num_staplers : ℕ := 2) :
    num_pencils * weight_pencil +
    num_erasers * weight_eraser +
    num_paperclips * weight_paperclip +
    num_staplers * weight_stapler = 287 := 
sorry

end combined_weight_of_contents_l89_89121


namespace measurable_masses_l89_89104

theorem measurable_masses (k : ℤ) (h : -121 ≤ k ∧ k ≤ 121) : 
  ∃ (a b c d e : ℤ), k = a * 1 + b * 3 + c * 9 + d * 27 + e * 81 ∧ 
  (a = -1 ∨ a = 0 ∨ a = 1) ∧
  (b = -1 ∨ b = 0 ∨ b = 1) ∧
  (c = -1 ∨ c = 0 ∨ c = 1) ∧
  (d = -1 ∨ d = 0 ∨ d = 1) ∧
  (e = -1 ∨ e = 0 ∨ e = 1) :=
sorry

end measurable_masses_l89_89104


namespace soccer_camp_ratio_l89_89663

theorem soccer_camp_ratio :
  let total_kids := 2000
  let half_total := total_kids / 2
  let afternoon_camp := 750
  let morning_camp := half_total - afternoon_camp
  half_total ≠ 0 → 
  (morning_camp / half_total) = 1 / 4 := by
  sorry

end soccer_camp_ratio_l89_89663


namespace four_digit_numbers_sum_even_l89_89427

theorem four_digit_numbers_sum_even : 
  ∃ N : ℕ, 
    (∀ (digits : Finset ℕ) (thousands hundreds tens units : ℕ), 
      digits = {1, 2, 3, 4, 5, 6} ∧ 
      ∀ n ∈ digits, (0 < n ∧ n < 10) ∧ 
      (thousands ∈ digits ∧ hundreds ∈ digits ∧ tens ∈ digits ∧ units ∈ digits) ∧ 
      (thousands ≠ hundreds ∧ thousands ≠ tens ∧ thousands ≠ units ∧ 
       hundreds ≠ tens ∧ hundreds ≠ units ∧ tens ≠ units) ∧ 
      (tens + units) % 2 = 0 → N = 324) :=
sorry

end four_digit_numbers_sum_even_l89_89427


namespace count_jianzhan_count_gift_boxes_l89_89937

-- Definitions based on given conditions
def firewood_red_clay : Int := 90
def firewood_white_clay : Int := 60
def electric_red_clay : Int := 75
def electric_white_clay : Int := 75
def total_red_clay : Int := 1530
def total_white_clay : Int := 1170

-- Proof problem 1: Number of "firewood firing" and "electric firing" Jianzhan produced
theorem count_jianzhan (x y : Int) (hx : firewood_red_clay * x + electric_red_clay * y = total_red_clay)
  (hy : firewood_white_clay * x + electric_white_clay * y = total_white_clay) : 
  x = 12 ∧ y = 6 :=
sorry

-- Definitions based on given conditions for Part 2
def total_jianzhan : Int := 18
def box_a_capacity : Int := 2
def box_b_capacity : Int := 6

-- Proof problem 2: Number of purchasing plans for gift boxes
theorem count_gift_boxes (m n : Int) (h : box_a_capacity * m + box_b_capacity * n = total_jianzhan) : 
  ∃ s : Finset (Int × Int), s.card = 4 ∧ ∀ (p : Int × Int), p ∈ s ↔ (p = (9, 0) ∨ p = (6, 1) ∨ p = (3, 2) ∨ p = (0, 3)) :=
sorry

end count_jianzhan_count_gift_boxes_l89_89937


namespace students_prefer_windows_l89_89133

theorem students_prefer_windows (total_students students_prefer_mac equally_prefer_both no_preference : ℕ) 
  (h₁ : total_students = 210)
  (h₂ : students_prefer_mac = 60)
  (h₃ : equally_prefer_both = 20)
  (h₄ : no_preference = 90) :
  total_students - students_prefer_mac - equally_prefer_both - no_preference = 40 := 
  by
    -- Proof goes here
    sorry

end students_prefer_windows_l89_89133


namespace inequality_solution_l89_89167

theorem inequality_solution :
  ∀ x : ℝ, (x - 3) / (x^2 + 4 * x + 10) ≥ 0 ↔ x ≥ 3 :=
by
  sorry

end inequality_solution_l89_89167


namespace express_set_M_l89_89625

def is_divisor (a b : ℤ) : Prop := ∃ k : ℤ, a = b * k

def M : Set ℤ := {m | is_divisor 10 (m + 1)}

theorem express_set_M :
  M = {-11, -6, -3, -2, 0, 1, 4, 9} :=
by
  sorry

end express_set_M_l89_89625


namespace problem_3034_1002_20_04_div_sub_l89_89015

theorem problem_3034_1002_20_04_div_sub:
  3034 - (1002 / 20.04) = 2984 :=
by
  sorry

end problem_3034_1002_20_04_div_sub_l89_89015


namespace price_of_second_oil_l89_89552

open Real

-- Define conditions
def litres_of_first_oil : ℝ := 10
def price_per_litre_first_oil : ℝ := 50
def litres_of_second_oil : ℝ := 5
def total_volume_of_mixture : ℝ := 15
def rate_of_mixture : ℝ := 55.67
def total_cost_of_mixture : ℝ := total_volume_of_mixture * rate_of_mixture

-- Define total cost of the first oil
def total_cost_first_oil : ℝ := litres_of_first_oil * price_per_litre_first_oil

-- Define total cost of the second oil in terms of unknown price P
def total_cost_second_oil (P : ℝ) : ℝ := litres_of_second_oil * P

-- Theorem to prove price per litre of the second oil
theorem price_of_second_oil : ∃ P : ℝ, total_cost_first_oil + (total_cost_second_oil P) = total_cost_of_mixture ∧ P = 67.01 :=
by
  sorry

end price_of_second_oil_l89_89552


namespace men_left_hostel_l89_89407

variable (x : ℕ)
variable (h1 : 250 * 36 = (250 - x) * 45)

theorem men_left_hostel : x = 50 :=
by
  sorry

end men_left_hostel_l89_89407


namespace flagpole_height_l89_89377

/-
A flagpole is of certain height. It breaks, folding over in half, such that what was the tip of the flagpole is now dangling two feet above the ground. 
The flagpole broke 7 feet from the base. Prove that the height of the flagpole is 16 feet.
-/

theorem flagpole_height (H : ℝ) (h1 : H > 0) (h2 : H - 7 > 0) (h3 : H - 9 = 7) : H = 16 :=
by
  /- the proof is omitted -/
  sorry

end flagpole_height_l89_89377


namespace fraction_of_men_married_is_two_thirds_l89_89842

-- Define the total number of faculty members
def total_faculty_members : ℕ := 100

-- Define the number of women as 70% of the faculty members
def women : ℕ := (70 * total_faculty_members) / 100

-- Define the number of men as 30% of the faculty members
def men : ℕ := (30 * total_faculty_members) / 100

-- Define the number of married faculty members as 40% of the faculty members
def married_faculty : ℕ := (40 * total_faculty_members) / 100

-- Define the number of single men as 1/3 of the men
def single_men : ℕ := men / 3

-- Define the number of married men as 2/3 of the men
def married_men : ℕ := (2 * men) / 3

-- Define the fraction of men who are married
def fraction_married_men : ℚ := married_men / men

-- The proof statement
theorem fraction_of_men_married_is_two_thirds : fraction_married_men = 2 / 3 := 
by sorry

end fraction_of_men_married_is_two_thirds_l89_89842


namespace irene_total_income_l89_89731

noncomputable def irene_income (weekly_hours : ℕ) (base_pay : ℕ) (overtime_pay : ℕ) (hours_worked : ℕ) : ℕ :=
  base_pay + (if hours_worked > weekly_hours then (hours_worked - weekly_hours) * overtime_pay else 0)

theorem irene_total_income :
  irene_income 40 500 20 50 = 700 :=
by
  sorry

end irene_total_income_l89_89731


namespace intersection_A_B_complement_l89_89841

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ 1}
def B_complement : Set ℝ := U \ B

theorem intersection_A_B_complement : A ∩ B_complement = {x | x > 1} := 
by 
  sorry

end intersection_A_B_complement_l89_89841


namespace combined_molecular_weight_l89_89632

-- Define atomic masses of elements
def atomic_mass_Ca : Float := 40.08
def atomic_mass_Br : Float := 79.904
def atomic_mass_Sr : Float := 87.62
def atomic_mass_Cl : Float := 35.453

-- Define number of moles for each compound
def moles_CaBr2 : Float := 4
def moles_SrCl2 : Float := 3

-- Define molar masses of compounds
def molar_mass_CaBr2 : Float := atomic_mass_Ca + 2 * atomic_mass_Br
def molar_mass_SrCl2 : Float := atomic_mass_Sr + 2 * atomic_mass_Cl

-- Define total mass calculation for each compound
def total_mass_CaBr2 : Float := moles_CaBr2 * molar_mass_CaBr2
def total_mass_SrCl2 : Float := moles_SrCl2 * molar_mass_SrCl2

-- Prove the combined molecular weight
theorem combined_molecular_weight :
  total_mass_CaBr2 + total_mass_SrCl2 = 1275.13 :=
  by
    -- The proof will be here
    sorry

end combined_molecular_weight_l89_89632


namespace intersecting_circles_l89_89443

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l89_89443


namespace right_triangle_side_length_l89_89270

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 12) 
  (h_right : a^2 + b^2 = c^2) : 
  b = Real.sqrt 119 :=
by
  sorry

end right_triangle_side_length_l89_89270


namespace average_marks_l89_89380

theorem average_marks (P C M : ℝ) (h1 : P = 95) (h2 : (P + M) / 2 = 90) (h3 : (P + C) / 2 = 70) :
  (P + C + M) / 3 = 75 := 
by
  sorry

end average_marks_l89_89380


namespace average_score_girls_proof_l89_89895

noncomputable def average_score_girls_all_schools (A a B b C c : ℕ)
  (adams_boys : ℕ) (adams_girls : ℕ) (adams_comb : ℕ)
  (baker_boys : ℕ) (baker_girls : ℕ) (baker_comb : ℕ)
  (carter_boys : ℕ) (carter_girls : ℕ) (carter_comb : ℕ)
  (all_boys_comb : ℕ) : ℕ :=
  -- Assume number of boys and girls per school A, B, C (boys) and a, b, c (girls)
  if (adams_boys * A + adams_girls * a) / (A + a) = adams_comb ∧
     (baker_boys * B + baker_girls * b) / (B + b) = baker_comb ∧
     (carter_boys * C + carter_girls * c) / (C + c) = carter_comb ∧
     (adams_boys * A + baker_boys * B + carter_boys * C) / (A + B + C) = all_boys_comb
  then (85 * a + 92 * b + 80 * c) / (a + b + c) else 0

theorem average_score_girls_proof (A a B b C c : ℕ)
  (adams_boys : ℕ := 82) (adams_girls : ℕ := 85) (adams_comb : ℕ := 83)
  (baker_boys : ℕ := 87) (baker_girls : ℕ := 92) (baker_comb : ℕ := 91)
  (carter_boys : ℕ := 78) (carter_girls : ℕ := 80) (carter_comb : ℕ := 80)
  (all_boys_comb : ℕ := 84) :
  average_score_girls_all_schools A a B b C c adams_boys adams_girls adams_comb baker_boys baker_girls baker_comb carter_boys carter_girls carter_comb all_boys_comb = 85 :=
by
  sorry

end average_score_girls_proof_l89_89895


namespace f_periodic_analytic_expression_f_distinct_real_roots_l89_89859

noncomputable def f (x : ℝ) (k : ℤ) : ℝ := (x - 2 * k)^2

def I_k (k : ℤ) : Set ℝ := { x | 2 * k - 1 < x ∧ x ≤ 2 * k + 1 }

def M_k (k : ℕ) : Set ℝ := { a | 0 < a ∧ a ≤ 1 / (2 * ↑k + 1) }

theorem f_periodic (x : ℝ) (k : ℤ) : f x k = f (x - 2 * k) 0 := by
  sorry

theorem analytic_expression_f (x : ℝ) (k : ℤ) (hx : x ∈ I_k k) : f x k = (x - 2 * k)^2 := by
  sorry

theorem distinct_real_roots (k : ℕ) (a : ℝ) (h : a ∈ M_k k) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ I_k k ∧ x2 ∈ I_k k ∧ f x1 k = a * x1 ∧ f x2 k = a * x2 := by
  sorry

end f_periodic_analytic_expression_f_distinct_real_roots_l89_89859


namespace obtuse_triangle_sum_range_l89_89418

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end obtuse_triangle_sum_range_l89_89418


namespace width_of_box_l89_89886

theorem width_of_box (w : ℝ) (h1 : w > 0) 
    (length : ℝ) (h2 : length = 60) 
    (area_lawn : ℝ) (h3 : area_lawn = 2109) 
    (width_road : ℝ) (h4 : width_road = 3) 
    (crossroads : ℝ) (h5 : crossroads = 2 * (60 / 3 * 3)) :
    60 * w - 120 = 2109 → w = 37.15 := 
by 
  intro h6
  sorry

end width_of_box_l89_89886


namespace smallest_possible_n_l89_89483

theorem smallest_possible_n (n : ℕ) (h : ∃ k : ℕ, 15 * n - 2 = 11 * k) : n % 11 = 6 :=
by
  sorry

end smallest_possible_n_l89_89483


namespace prime_roots_eq_l89_89693

theorem prime_roots_eq (n : ℕ) (hn : 0 < n) :
  (∃ (x1 x2 : ℕ), Prime x1 ∧ Prime x2 ∧ 2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧ 
                    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 ∧ x1 ≠ x2 ∧ x1 < x2) →
  n = 3 ∧ ∃ x1 x2 : ℕ, x1 = 2 ∧ x2 = 5 ∧ Prime x1 ∧ Prime x2 ∧
    2*x1^2 - 8*n*x1 + 10*x1 - n^2 + 35*n - 76 = 0 ∧
    2*x2^2 - 8*n*x2 + 10*x2 - n^2 + 35*n - 76 = 0 := 
by
  sorry

end prime_roots_eq_l89_89693


namespace percent_of_div_l89_89397

theorem percent_of_div (P: ℝ) (Q: ℝ) (R: ℝ) : ( ( P / 100 ) * Q ) / R = 354.2 :=
by
  -- Given P = 168, Q = 1265, R = 6
  let P := 168
  let Q := 1265
  let R := 6
  -- sorry to skip the actual proof.
  sorry

end percent_of_div_l89_89397


namespace mrs_martin_pays_l89_89460

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def mr_martin_scoops : ℕ := 1
def mrs_martin_scoops : ℕ := 1
def children_scoops : ℕ := 2
def teenage_children_scoops : ℕ := 3

def total_cost : ℕ :=
  (mr_martin_scoops + mrs_martin_scoops) * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost

theorem mrs_martin_pays : total_cost = 32 :=
  by sorry

end mrs_martin_pays_l89_89460


namespace simplify_fraction_l89_89820

theorem simplify_fraction :
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
  sorry

end simplify_fraction_l89_89820


namespace tan_beta_minus_pi_over_4_l89_89810

theorem tan_beta_minus_pi_over_4 (α β : ℝ) 
  (h1 : Real.tan (α + β) = 1/2) 
  (h2 : Real.tan (α + π/4) = -1/3) : 
  Real.tan (β - π/4) = 1 := 
sorry

end tan_beta_minus_pi_over_4_l89_89810


namespace ellipse_area_constant_l89_89338

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def passes_through (x_a y_a x_b y_b : ℝ) (p : ℝ × ℝ) : Prop := 
  p.1 = x_a ∧ p.2 = y_a ∨ p.1 = x_b ∧ p.2 = y_b

def area_ABNM_constant (x y : ℝ) : Prop :=
  let x_0 := x;
  let y_0 := y;
  let y_M := -2 * y_0 / (x_0 - 2);
  let BM := 1 + 2 * y_0 / (x_0 - 2);
  let x_N := - x_0 / (y_0 - 1);
  let AN := 2 + x_0 / (y_0 - 1);
  (1 / 2) * AN * BM = 2

theorem ellipse_area_constant :
  ∀ (a b : ℝ), (a = 2 ∧ b = 1) → 
  (∀ (x y : ℝ), 
    ellipse_equation a b x y → 
    passes_through 2 0 0 1 (x, y) → 
    (x < 0 ∧ y < 0) →
    area_ABNM_constant x y) :=
by
  intros
  sorry

end ellipse_area_constant_l89_89338


namespace compare_magnitude_l89_89670

theorem compare_magnitude (a b : ℝ) (h : a ≠ 1) : a^2 + b^2 > 2 * (a - b - 1) :=
by
  sorry

end compare_magnitude_l89_89670


namespace count_integer_values_l89_89337

theorem count_integer_values (x : ℕ) (h : 3 < Real.sqrt x ∧ Real.sqrt x < 5) : 
  ∃! n, (n = 15) ∧ ∀ k, (3 < Real.sqrt k ∧ Real.sqrt k < 5) → (k ≥ 10 ∧ k ≤ 24) :=
by
  sorry

end count_integer_values_l89_89337


namespace allocation_schemes_l89_89499

theorem allocation_schemes :
  ∃ (n : ℕ), n = 240 ∧ (∀ (volunteers : Fin 5 → Fin 4), 
    (∃ (assign : Fin 5 → Fin 4), 
      (∀ (i : Fin 4), ∃ (j : Fin 5), assign j = i)
      ∧ (∀ (k : Fin 5), assign k ≠ assign k)) 
    → true) := sorry

end allocation_schemes_l89_89499


namespace city_partition_exists_l89_89234

-- Define a market and street as given
structure City where
  markets : Type
  street : markets → markets → Prop
  leaves_exactly_two : ∀ (m : markets), ∃ (m1 m2 : markets), street m m1 ∧ street m m2

-- Our formal proof statement
theorem city_partition_exists (C : City) : 
  ∃ (partition : C.markets → Fin 1014), 
    (∀ (m1 m2 : C.markets), C.street m1 m2 → partition m1 ≠ partition m2) ∧
    (∀ (d1 d2 : Fin 1014) (m1 m2 : C.markets), (partition m1 = d1) ∧ (partition m2 = d2) → 
     (C.street m1 m2 ∨ C.street m2 m1) →  (∀ (k l : Fin 1014), (k = d1) → (l = d2) → (∀ (a b : C.markets), (partition a = k) → (partition b = l) → (C.street a b ∨ C.street b a)))) :=
sorry

end city_partition_exists_l89_89234


namespace add_numerator_denominator_add_numerator_denominator_gt_one_l89_89210

variable {a b n : ℕ}

/-- Adding the same natural number to both the numerator and the denominator of a fraction 
    increases the fraction if it is less than one, and decreases the fraction if it is greater than one. -/
theorem add_numerator_denominator (h1: a < b) : (a + n) / (b + n) > a / b := sorry

theorem add_numerator_denominator_gt_one (h2: a > b) : (a + n) / (b + n) < a / b := sorry

end add_numerator_denominator_add_numerator_denominator_gt_one_l89_89210


namespace polygonal_chain_segments_l89_89938

theorem polygonal_chain_segments (n : ℕ) :
  (∃ (S : Type) (chain : S → Prop), (∃ (closed_non_self_intersecting : S → Prop), 
  (∀ s : S, chain s → closed_non_self_intersecting s) ∧
  ∀ line_segment : S, chain line_segment → 
  (∃ other_segment : S, chain other_segment ∧ line_segment ≠ other_segment))) ↔ 
  (∃ k : ℕ, (n = 2 * k ∧ 5 ≤ k) ∨ (n = 2 * k + 1 ∧ 7 ≤ k)) :=
by sorry

end polygonal_chain_segments_l89_89938


namespace fraction_value_l89_89502

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 3) = 9 / 8 := 
by sorry

end fraction_value_l89_89502


namespace largest_of_four_integers_l89_89644

theorem largest_of_four_integers (n : ℤ) (h1 : n % 2 = 0) (h2 : (n+2) % 2 = 0) (h3 : (n+4) % 2 = 0) (h4 : (n+6) % 2 = 0) (h : n * (n+2) * (n+4) * (n+6) = 6720) : max (max (max n (n+2)) (n+4)) (n+6) = 14 := 
sorry

end largest_of_four_integers_l89_89644


namespace probability_heads_equals_7_over_11_l89_89922

theorem probability_heads_equals_7_over_11 (p : ℝ) (q : ℝ)
  (h1 : q = 1 - p)
  (h2 : 120 * p^7 * q^3 = 210 * p^6 * q^4) :
  p = 7 / 11 :=
by {
  sorry
}

end probability_heads_equals_7_over_11_l89_89922


namespace period_of_function_is_2pi_over_3_l89_89333

noncomputable def period_of_f (x : ℝ) : ℝ :=
  4 * (Real.sin x)^3 - Real.sin x + 2 * (Real.sin (x / 2) - Real.cos (x / 2))^2

theorem period_of_function_is_2pi_over_3 : ∀ x, period_of_f (x + (2 * Real.pi) / 3) = period_of_f x :=
by sorry

end period_of_function_is_2pi_over_3_l89_89333


namespace incorrect_statement_is_B_l89_89681

-- Define the conditions
def genotype_AaBb_meiosis_results (sperm_genotypes : List String) : Prop :=
  sperm_genotypes = ["AB", "Ab", "aB", "ab"]

def spermatogonial_cell_AaXbY (malformed_sperm_genotype : String) (other_sperm_genotypes : List String) : Prop :=
  malformed_sperm_genotype = "AAaY" ∧ other_sperm_genotypes = ["aY", "X^b", "X^b"]

def spermatogonial_secondary_spermatocyte_Y_chromosomes (contains_two_Y : Bool) : Prop :=
  ¬ contains_two_Y

def female_animal_meiosis (primary_oocyte_alleles : Nat) (max_oocyte_b_alleles : Nat) : Prop :=
  primary_oocyte_alleles = 10 ∧ max_oocyte_b_alleles ≤ 5

-- The main statement that needs to be proved
theorem incorrect_statement_is_B :
  ∃ (sperm_genotypes : List String) 
    (malformed_sperm_genotype : String) 
    (other_sperm_genotypes : List String) 
    (contains_two_Y : Bool) 
    (primary_oocyte_alleles max_oocyte_b_alleles : Nat),
    genotype_AaBb_meiosis_results sperm_genotypes ∧ 
    spermatogonial_cell_AaXbY malformed_sperm_genotype other_sperm_genotypes ∧ 
    spermatogonial_secondary_spermatocyte_Y_chromosomes contains_two_Y ∧ 
    female_animal_meiosis primary_oocyte_alleles max_oocyte_b_alleles 
    ∧ (malformed_sperm_genotype = "AAaY" → false) := 
sorry

end incorrect_statement_is_B_l89_89681


namespace simplified_expression_evaluation_l89_89316

def expression (x y : ℝ) : ℝ :=
  3 * (x^2 - 2 * x^2 * y) - 3 * x^2 + 2 * y - 2 * (x^2 * y + y)

def x := 1/2
def y := -3

theorem simplified_expression_evaluation : expression x y = 6 :=
  sorry

end simplified_expression_evaluation_l89_89316


namespace no_integer_m_l89_89997

theorem no_integer_m (n r m : ℕ) (hn : 1 ≤ n) (hr : 2 ≤ r) : 
  ¬ (∃ m : ℕ, n * (n + 1) * (n + 2) = m ^ r) :=
sorry

end no_integer_m_l89_89997


namespace ways_to_write_2020_as_sum_of_twos_and_threes_l89_89949

def write_as_sum_of_twos_and_threes (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n / 2 + 1) else 0

theorem ways_to_write_2020_as_sum_of_twos_and_threes :
  write_as_sum_of_twos_and_threes 2020 = 337 :=
sorry

end ways_to_write_2020_as_sum_of_twos_and_threes_l89_89949


namespace danielle_travel_time_is_30_l89_89218

noncomputable def chase_speed : ℝ := sorry
noncomputable def chase_time : ℝ := 180 -- in minutes
noncomputable def cameron_speed : ℝ := 2 * chase_speed
noncomputable def danielle_speed : ℝ := 3 * cameron_speed
noncomputable def distance : ℝ := chase_speed * chase_time
noncomputable def danielle_time : ℝ := distance / danielle_speed

theorem danielle_travel_time_is_30 :
  danielle_time = 30 :=
sorry

end danielle_travel_time_is_30_l89_89218


namespace f_a_plus_b_eq_neg_one_l89_89335

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if x ≥ 0 then x * (x - b) else a * x * (x + 2)

theorem f_a_plus_b_eq_neg_one (a b : ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) a b = -f x a b) 
  (ha : a = -1) 
  (hb : b = 2) : 
  f (a + b) a b = -1 :=
by
  sorry

end f_a_plus_b_eq_neg_one_l89_89335


namespace factorization_correct_l89_89936

noncomputable def factorize_diff_of_squares (a b : ℝ) : ℝ :=
  36 * a * a - 4 * b * b

theorem factorization_correct (a b : ℝ) : factorize_diff_of_squares a b = 4 * (3 * a + b) * (3 * a - b) :=
by
  sorry

end factorization_correct_l89_89936


namespace bridge_length_is_correct_l89_89933

-- Train length in meters
def train_length : ℕ := 130

-- Train speed in km/hr
def train_speed_kmh : ℕ := 45

-- Time to cross bridge in seconds
def time_to_cross_bridge : ℕ := 30

-- Conversion factor from km/hr to m/s
def kmh_to_mps (kmh : ℕ) : ℚ := (kmh * 1000) / 3600

-- Train speed in m/s
def train_speed_mps := kmh_to_mps train_speed_kmh

-- Total distance covered by the train in 30 seconds
def total_distance := train_speed_mps * time_to_cross_bridge

-- Length of the bridge
def bridge_length := total_distance - train_length

theorem bridge_length_is_correct : bridge_length = 245 := by
  sorry

end bridge_length_is_correct_l89_89933


namespace problem_statement_l89_89998

-- Define a : ℝ such that (a + 1/a)^3 = 7
variables (a : ℝ) (h : (a + 1/a)^3 = 7)

-- Goal: Prove that a^4 + 1/a^4 = 1519/81
theorem problem_statement (a : ℝ) (h : (a + 1/a)^3 = 7) : a^4 + 1/a^4 = 1519 / 81 := 
sorry

end problem_statement_l89_89998


namespace symmetric_circle_equation_l89_89354

theorem symmetric_circle_equation (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 5 → (x + 1)^2 + (y - 2)^2 = 5 :=
by
  sorry

end symmetric_circle_equation_l89_89354


namespace greatest_integer_lesser_200_gcd_45_eq_9_l89_89720

theorem greatest_integer_lesser_200_gcd_45_eq_9 :
  ∃ n : ℕ, n < 200 ∧ Int.gcd n 45 = 9 ∧ ∀ m : ℕ, (m < 200 ∧ Int.gcd m 45 = 9) → m ≤ n :=
by
  sorry

end greatest_integer_lesser_200_gcd_45_eq_9_l89_89720


namespace B_work_rate_l89_89918

theorem B_work_rate :
  let A := (1 : ℝ) / 8
  let C := (1 : ℝ) / 4.8
  (A + B + C = 1 / 2) → (B = 1 / 6) :=
by
  intro h
  let A : ℝ := 1 / 8
  let C : ℝ := 1 / 4.8
  let B : ℝ := 1 / 6
  sorry

end B_work_rate_l89_89918


namespace radius_of_ball_is_13_l89_89682

-- Define the conditions
def hole_radius : ℝ := 12
def hole_depth : ℝ := 8

-- The statement to prove
theorem radius_of_ball_is_13 : (∃ x : ℝ, x^2 + hole_radius^2 = (x + hole_depth)^2) → x + hole_depth = 13 :=
by
  sorry

end radius_of_ball_is_13_l89_89682


namespace parallel_lines_intersection_value_of_c_l89_89619

theorem parallel_lines_intersection_value_of_c
  (a b c : ℝ) (h_parallel : a = -4 * b)
  (h1 : a * 2 - 2 * (-4) = c) (h2 : 2 * 2 + b * (-4) = c) :
  c = 0 :=
by 
  sorry

end parallel_lines_intersection_value_of_c_l89_89619


namespace triangle_side_a_l89_89178

theorem triangle_side_a {a b c : ℝ} (A : ℝ) (hA : A = (2 * Real.pi / 3)) (hb : b = Real.sqrt 2) 
(h_area : 1 / 2 * b * c * Real.sin A = Real.sqrt 3) :
  a = Real.sqrt 14 :=
by 
  sorry

end triangle_side_a_l89_89178


namespace textbook_weight_difference_l89_89538

theorem textbook_weight_difference :
  let chem_weight := 7.125
  let geom_weight := 0.625
  chem_weight - geom_weight = 6.5 :=
by
  sorry

end textbook_weight_difference_l89_89538


namespace walk_fraction_correct_l89_89830

def bus_fraction := 1/3
def automobile_fraction := 1/5
def bicycle_fraction := 1/8
def metro_fraction := 1/15

def total_transport_fraction := bus_fraction + automobile_fraction + bicycle_fraction + metro_fraction

def walk_fraction := 1 - total_transport_fraction

theorem walk_fraction_correct : walk_fraction = 11/40 := by
  sorry

end walk_fraction_correct_l89_89830


namespace proposition_d_correct_l89_89278

theorem proposition_d_correct (a b c : ℝ) (h : a > b) : a - c > b - c := 
by
  sorry

end proposition_d_correct_l89_89278


namespace sum_of_first_100_terms_l89_89617

theorem sum_of_first_100_terms (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 1) 
  (h3 : ∀ n, a (n+2) = a n + 1) : 
  (Finset.sum (Finset.range 100) a) = 2550 :=
sorry

end sum_of_first_100_terms_l89_89617


namespace part1_part2_l89_89330

-- Define m as a positive integer greater than or equal to 2
def m (k : ℕ) := k ≥ 2

-- Part 1: Existential statement for x_i's
theorem part1 (m : ℕ) (h : m ≥ 2) :
  ∃ (x : ℕ → ℤ),
    ∀ i, 1 ≤ i ∧ i ≤ m →
    x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1 := by
  sorry

-- Part 2: Infinite sequence y_k
theorem part2 (x : ℕ → ℤ) (m : ℕ) (h : m ≥ 2) :
  (∀ i, 1 ≤ i ∧ i ≤ m → x i * x (m + i) = x (i + 1) * x (m + i - 1) + 1) →
  ∃ (y : ℤ → ℤ),
    (∀ k : ℤ, y k * y (m + k) = y (k + 1) * y (m + k - 1) + 1) ∧
    (∀ i, 1 ≤ i ∧ i ≤ 2 * m → y i = x i) := by
  sorry

end part1_part2_l89_89330


namespace x4_plus_y4_l89_89149
noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

theorem x4_plus_y4 :
  (x^2 + (1 / x^2) = 7) →
  (x * y = 1) →
  (x^4 + y^4 = 47) :=
by
  intros h1 h2
  -- The proof will go here.
  sorry

end x4_plus_y4_l89_89149


namespace brendan_taxes_l89_89717

def total_hours (num_8hr_shifts : ℕ) (num_12hr_shifts : ℕ) : ℕ :=
  (num_8hr_shifts * 8) + (num_12hr_shifts * 12)

def total_wage (hourly_wage : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_wage * hours_worked

def total_tips (hourly_tips : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_tips * hours_worked

def reported_tips (total_tips : ℕ) (report_fraction : ℕ) : ℕ :=
  total_tips / report_fraction

def reported_income (wage : ℕ) (tips : ℕ) : ℕ :=
  wage + tips

def taxes (income : ℕ) (tax_rate : ℚ) : ℚ :=
  income * tax_rate

theorem brendan_taxes (num_8hr_shifts num_12hr_shifts : ℕ)
    (hourly_wage hourly_tips report_fraction : ℕ) (tax_rate : ℚ) :
    (hourly_wage = 6) →
    (hourly_tips = 12) →
    (report_fraction = 3) →
    (tax_rate = 0.2) →
    (num_8hr_shifts = 2) →
    (num_12hr_shifts = 1) →
    taxes (reported_income (total_wage hourly_wage (total_hours num_8hr_shifts num_12hr_shifts))
            (reported_tips (total_tips hourly_tips (total_hours num_8hr_shifts num_12hr_shifts))
            report_fraction))
          tax_rate = 56 :=
by
  intros
  sorry

end brendan_taxes_l89_89717


namespace count_valid_arrangements_l89_89977

-- Definitions based on conditions
def total_chairs : Nat := 48

def valid_factor_pairs (n : Nat) : List (Nat × Nat) :=
  [ (2, 24), (3, 16), (4, 12), (6, 8), (8, 6), (12, 4), (16, 3), (24, 2) ]

def count_valid_arrays : Nat := valid_factor_pairs total_chairs |>.length

-- The theorem we want to prove
theorem count_valid_arrangements : count_valid_arrays = 8 := 
  by
    -- proof should be provided here
    sorry

end count_valid_arrangements_l89_89977


namespace accounting_majors_count_l89_89923

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end accounting_majors_count_l89_89923


namespace ordering_of_powers_l89_89176

theorem ordering_of_powers : (3 ^ 17) < (8 ^ 9) ∧ (8 ^ 9) < (4 ^ 15) := 
by 
  -- We proved (3 ^ 17) < (8 ^ 9)
  have h1 : (3 ^ 17) < (8 ^ 9) := sorry
  
  -- We proved (8 ^ 9) < (4 ^ 15)
  have h2 : (8 ^ 9) < (4 ^ 15) := sorry

  -- Therefore, combining both
  exact ⟨h1, h2⟩

end ordering_of_powers_l89_89176


namespace fraction_addition_l89_89790

theorem fraction_addition :
  (3 / 4) / (5 / 8) + (1 / 2) = 17 / 10 :=
by
  sorry

end fraction_addition_l89_89790


namespace max_value_of_sample_l89_89146

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end max_value_of_sample_l89_89146


namespace remainders_inequalities_l89_89080

theorem remainders_inequalities
  (X Y M A B s t u : ℕ)
  (h1 : X > Y)
  (h2 : X = Y + 8)
  (h3 : X % M = A)
  (h4 : Y % M = B)
  (h5 : s = (X^2) % M)
  (h6 : t = (Y^2) % M)
  (h7 : u = (A * B)^2 % M) :
  s ≠ t ∧ t ≠ u ∧ s ≠ u :=
sorry

end remainders_inequalities_l89_89080


namespace smallest_digit_is_one_l89_89654

-- Given a 4-digit integer x.
def four_digit_integer (x : ℕ) : Prop :=
  1000 ≤ x ∧ x < 10000

-- Define function for the product of digits of x.
def product_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 * d2 * d3 * d4

-- Define function for the sum of digits of x.
def sum_of_digits (x : ℕ) : ℕ :=
  let d1 := x % 10
  let d2 := (x / 10) % 10
  let d3 := (x / 100) % 10
  let d4 := (x / 1000) % 10
  d1 + d2 + d3 + d4

-- Assume p is a prime number.
def is_prime (p : ℕ) : Prop :=
  ¬ ∃ d, d ∣ p ∧ d ≠ 1 ∧ d ≠ p

-- Proof problem: Given conditions for T(x) and S(x),
-- prove that the smallest digit in x is 1.
theorem smallest_digit_is_one (x p k : ℕ) (h1 : four_digit_integer x)
  (h2 : is_prime p) (h3 : product_of_digits x = p^k)
  (h4 : sum_of_digits x = p^p - 5) : 
  ∃ d1 d2 d3 d4, d1 <= d2 ∧ d1 <= d3 ∧ d1 <= d4 ∧ d1 = 1 
  ∧ (d1 + d2 + d3 + d4 = p^p - 5) 
  ∧ (d1 * d2 * d3 * d4 = p^k) := 
sorry

end smallest_digit_is_one_l89_89654


namespace find_b_value_l89_89026

theorem find_b_value 
  (point1 : ℝ × ℝ) (point2 : ℝ × ℝ) (b : ℝ) 
  (h1 : point1 = (0, -2))
  (h2 : point2 = (1, 0))
  (h3 : (∃ m c, ∀ x y, y = m * x + c ↔ (x, y) = point1 ∨ (x, y) = point2))
  (h4 : ∀ x y, y = 2 * x - 2 → (x, y) = (7, b)) :
  b = 12 :=
sorry

end find_b_value_l89_89026


namespace proposition_relation_l89_89657

theorem proposition_relation :
  (∀ (x : ℝ), x < 3 → x < 5) ↔ (∀ (x : ℝ), x ≥ 5 → x ≥ 3) :=
by
  sorry

end proposition_relation_l89_89657


namespace final_score_is_89_l89_89395

def final_score (s_e s_l s_b : ℝ) (p_e p_l p_b : ℝ) : ℝ :=
  s_e * p_e + s_l * p_l + s_b * p_b

theorem final_score_is_89 :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := 
by
  sorry

end final_score_is_89_l89_89395


namespace region_diff_correct_l89_89230

noncomputable def hexagon_area : ℝ := (3 * Real.sqrt 3) / 2
noncomputable def one_triangle_area : ℝ := (Real.sqrt 3) / 4
noncomputable def triangles_area : ℝ := 18 * one_triangle_area
noncomputable def R_area : ℝ := hexagon_area + triangles_area
noncomputable def S_area : ℝ := 4 * (1 + Real.sqrt 2)
noncomputable def region_diff : ℝ := S_area - R_area

theorem region_diff_correct :
  region_diff = 4 + 4 * Real.sqrt 2 - 6 * Real.sqrt 3 :=
by
  sorry

end region_diff_correct_l89_89230


namespace base_sum_correct_l89_89879

theorem base_sum_correct :
  let C := 12
  let a := 3 * 9^2 + 5 * 9^1 + 7 * 9^0
  let b := 4 * 13^2 + C * 13^1 + 2 * 13^0
  a + b = 1129 :=
by
  sorry

end base_sum_correct_l89_89879


namespace second_quarter_profit_l89_89636

theorem second_quarter_profit (q1 q3 q4 annual : ℕ) (h1 : q1 = 1500) (h2 : q3 = 3000) (h3 : q4 = 2000) (h4 : annual = 8000) :
  annual - (q1 + q3 + q4) = 1500 :=
by
  sorry

end second_quarter_profit_l89_89636


namespace base9_perfect_square_l89_89202

theorem base9_perfect_square (a b d : ℕ) (h1 : a ≠ 0) (h2 : a < 9) (h3 : b < 9) (h4 : d < 9) (h5 : ∃ n : ℕ, (729 * a + 81 * b + 36 + d) = n * n) : d = 0 ∨ d = 1 ∨ d = 4 :=
by sorry

end base9_perfect_square_l89_89202


namespace arithmetic_expression_l89_89159

theorem arithmetic_expression : 4 * 6 * 8 + 18 / 3 - 2 ^ 3 = 190 :=
by
  -- Proof goes here
  sorry

end arithmetic_expression_l89_89159


namespace cos_pi_over_8_cos_5pi_over_8_l89_89993

theorem cos_pi_over_8_cos_5pi_over_8 :
  (Real.cos (Real.pi / 8)) * (Real.cos (5 * Real.pi / 8)) = - (Real.sqrt 2 / 4) :=
by
  sorry

end cos_pi_over_8_cos_5pi_over_8_l89_89993


namespace distance_between_A_B_is_16_l89_89079

-- The given conditions are translated as definitions
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- The theorem stating the proof problem
theorem distance_between_A_B_is_16 :
  let A : ℝ × ℝ := (4, 8)
  let B : ℝ × ℝ := (4, -8)
  let d : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  d = 16 :=
by
  sorry

end distance_between_A_B_is_16_l89_89079


namespace equality_of_a_b_c_l89_89130

theorem equality_of_a_b_c
  (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (eqn : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c :=
by
  sorry

end equality_of_a_b_c_l89_89130


namespace find_y_when_x_is_8_l89_89692

theorem find_y_when_x_is_8 (x y : ℕ) (k : ℕ) (h1 : x + y = 36) (h2 : x - y = 12) (h3 : x * y = k) (h4 : k = 288) : y = 36 :=
by
  -- Given the conditions
  sorry

end find_y_when_x_is_8_l89_89692


namespace correct_exponential_rule_l89_89212

theorem correct_exponential_rule (a : ℝ) : (a^3)^2 = a^6 :=
by sorry

end correct_exponential_rule_l89_89212


namespace landscaping_charges_l89_89140

theorem landscaping_charges
    (x : ℕ)
    (h : 63 * x + 9 * 11 + 10 * 9 = 567) :
  x = 6 :=
by
  sorry

end landscaping_charges_l89_89140


namespace base8_subtraction_l89_89177

def subtract_base_8 (a b : Nat) : Nat :=
  sorry  -- This is a placeholder for the actual implementation.

theorem base8_subtraction :
  subtract_base_8 0o5374 0o2645 = 0o1527 :=
by
  sorry

end base8_subtraction_l89_89177


namespace set_complement_union_l89_89543

-- Definitions of the sets
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

-- The statement to prove
theorem set_complement_union : (U \ A) ∪ (U \ B) = {1, 4, 5} :=
by sorry

end set_complement_union_l89_89543


namespace number_of_hexagonal_faces_geq_2_l89_89748

noncomputable def polyhedron_condition (P H : ℕ) : Prop :=
  ∃ V E : ℕ, 
    V - E + (P + H) = 2 ∧ 
    3 * V = 2 * E ∧ 
    E = (5 * P + 6 * H) / 2 ∧
    P > 0 ∧ H > 0

theorem number_of_hexagonal_faces_geq_2 (P H : ℕ) (h : polyhedron_condition P H) : H ≥ 2 :=
sorry

end number_of_hexagonal_faces_geq_2_l89_89748


namespace sum_of_remainders_l89_89744

theorem sum_of_remainders
  (a b c : ℕ)
  (h₁ : a % 36 = 15)
  (h₂ : b % 36 = 22)
  (h₃ : c % 36 = 9) :
  (a + b + c) % 36 = 10 :=
by
  sorry

end sum_of_remainders_l89_89744


namespace simplify_division_l89_89087

theorem simplify_division :
  (2 * 10^12) / (4 * 10^5 - 1 * 10^4) = 5.1282 * 10^6 :=
by
  -- problem statement
  sorry

end simplify_division_l89_89087


namespace inequality_solution_set_l89_89258

theorem inequality_solution_set 
  (c : ℝ) (a : ℝ) (b : ℝ) (h : c > 0) (hb : b = (5 / 2) * c) (ha : a = - (3 / 2) * c) :
  ∀ x : ℝ, (a * x^2 + b * x + c ≥ 0) ↔ (- (1 / 3) ≤ x ∧ x ≤ 2) :=
sorry

end inequality_solution_set_l89_89258


namespace fraction_given_to_classmates_l89_89403

theorem fraction_given_to_classmates
  (total_boxes : ℕ) (pens_per_box : ℕ)
  (percentage_to_friends : ℝ) (pens_left_after_classmates : ℕ) :
  total_boxes = 20 →
  pens_per_box = 5 →
  percentage_to_friends = 0.40 →
  pens_left_after_classmates = 45 →
  (15 / (total_boxes * pens_per_box - percentage_to_friends * total_boxes * pens_per_box)) = 1 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_given_to_classmates_l89_89403


namespace haley_initial_cupcakes_l89_89449

-- Define the conditions
def todd_eats : ℕ := 11
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 3

-- Initial cupcakes calculation
def initial_cupcakes := packages * cupcakes_per_package + todd_eats

-- The theorem to prove
theorem haley_initial_cupcakes : initial_cupcakes = 20 :=
by
  -- Mathematical proof would go here,
  -- but we leave it as sorry for now.
  sorry

end haley_initial_cupcakes_l89_89449


namespace alpha_is_30_or_60_l89_89599

theorem alpha_is_30_or_60
  (α : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2) -- α is acute angle
  (a : ℝ × ℝ := (3 / 4, Real.sin α))
  (b : ℝ × ℝ := (Real.cos α, 1 / Real.sqrt 3))
  (h2 : a.1 * b.2 = a.2 * b.1)  -- a ∥ b
  : α = Real.pi / 6 ∨ α = Real.pi / 3 := 
sorry

end alpha_is_30_or_60_l89_89599


namespace interval_intersection_l89_89680

theorem interval_intersection :
  {x : ℝ | 1 < 3 * x ∧ 3 * x < 2 ∧ 1 < 5 * x ∧ 5 * x < 2} =
  {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (2 / 5 : ℝ)} :=
by
  -- Need a proof here
  sorry

end interval_intersection_l89_89680


namespace find_b_l89_89983

noncomputable def h (x : ℝ) : ℝ := x^2 + 9
noncomputable def j (x : ℝ) : ℝ := x^2 + 1

theorem find_b (b : ℝ) (hjb : h (j b) = 15) (b_pos : b > 0) : b = Real.sqrt (Real.sqrt 6 - 1) := by
  sorry

end find_b_l89_89983


namespace joe_eggs_club_house_l89_89082

theorem joe_eggs_club_house (C : ℕ) (h : C + 5 + 3 = 20) : C = 12 :=
by 
  sorry

end joe_eggs_club_house_l89_89082


namespace root_interval_range_l89_89562

theorem root_interval_range (m : ℝ) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ x^3 - 3*x + m = 0) → (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end root_interval_range_l89_89562


namespace most_consistent_player_l89_89596

section ConsistentPerformance

variables (σA σB σC σD : ℝ)
variables (σA_eq : σA = 0.023)
variables (σB_eq : σB = 0.018)
variables (σC_eq : σC = 0.020)
variables (σD_eq : σD = 0.021)

theorem most_consistent_player : σB < σC ∧ σB < σD ∧ σB < σA :=
by 
  rw [σA_eq, σB_eq, σC_eq, σD_eq]
  sorry

end ConsistentPerformance

end most_consistent_player_l89_89596


namespace range_of_m_for_distinct_real_roots_l89_89894

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2 * x₁ + m = 0 ∧ x₂^2 + 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end range_of_m_for_distinct_real_roots_l89_89894


namespace range_of_k_l89_89386

theorem range_of_k (k : ℝ) :
  (∃ (x : ℝ), 2 < x ∧ x < 3 ∧ x^2 + (1 - k) * x - 2 * (k + 1) = 0) →
  1 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l89_89386


namespace find_number_l89_89233

theorem find_number (x : ℤ) (h : (x + 305) / 16 = 31) : x = 191 :=
sorry

end find_number_l89_89233


namespace decreasing_function_range_l89_89547

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else -a * x

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1 / 8 ≤ a ∧ a < 1 / 3) :=
by
  sorry

end decreasing_function_range_l89_89547


namespace half_day_division_l89_89077

theorem half_day_division : 
  ∃ (n m : ℕ), n * m = 43200 ∧ (∃! (k : ℕ), k = 60) := sorry

end half_day_division_l89_89077


namespace price_of_cookie_cookie_price_verification_l89_89348

theorem price_of_cookie 
  (total_spent : ℝ) 
  (cost_per_cupcake : ℝ)
  (num_cupcakes : ℕ)
  (cost_per_doughnut : ℝ)
  (num_doughnuts : ℕ)
  (cost_per_pie_slice : ℝ)
  (num_pie_slices : ℕ)
  (num_cookies : ℕ)
  (total_cookies_cost : ℝ)
  (total_cost : ℝ) :
  (num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  + num_cookies * total_cookies_cost = total_spent) → 
  total_cookies_cost = 0.60 :=
by
  sorry

noncomputable def sophie_cookies_price : ℝ := 
  let total_cost := 33
  let num_cupcakes := 5
  let cost_per_cupcake := 2
  let num_doughnuts := 6
  let cost_per_doughnut := 1
  let num_pie_slices := 4
  let cost_per_pie_slice := 2
  let num_cookies := 15
  let total_spent_on_other_items := 
    num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  let remaining_cost := total_cost - total_spent_on_other_items 
  remaining_cost / num_cookies

theorem cookie_price_verification :
  sophie_cookies_price = 0.60 :=
by
  sorry

end price_of_cookie_cookie_price_verification_l89_89348


namespace simplify_expression_l89_89557

theorem simplify_expression (x y z : ℝ) (h1 : x ≠ 2) (h2 : y ≠ 3) (h3 : z ≠ 4) :
  (x - 2) / (4 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 :=
by 
sorry

end simplify_expression_l89_89557


namespace range_of_a_l89_89905

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x - a * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≥ 2 * a - (1 / 2) * a^2) ↔ 0 ≤ a :=
by
  sorry

end range_of_a_l89_89905


namespace total_collected_funds_l89_89770

theorem total_collected_funds (A B T : ℕ) (hA : A = 5) (hB : B = 3 * A + 3) (h_quotient : B / 3 = 6) (hT : T = B * (B / 3) + A) : 
  T = 113 := 
by 
  sorry

end total_collected_funds_l89_89770


namespace find_solutions_l89_89558

theorem find_solutions (x y : ℕ) : 33 ^ x + 31 = 2 ^ y → (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := 
by
  sorry

end find_solutions_l89_89558


namespace sin_cos_fraction_l89_89188

theorem sin_cos_fraction (α : ℝ) (h1 : Real.sin α - Real.cos α = 1 / 5) (h2 : α ∈ Set.Ioo (-Real.pi / 2) (Real.pi / 2)) :
    Real.sin α * Real.cos α / (Real.sin α + Real.cos α) = 12 / 35 :=
by
  sorry

end sin_cos_fraction_l89_89188


namespace probability_yellow_face_l89_89110

-- Define the total number of faces and the number of yellow faces on the die
def total_faces := 12
def yellow_faces := 4

-- Define the probability calculation
def probability_of_yellow := yellow_faces / total_faces

-- State the theorem
theorem probability_yellow_face : probability_of_yellow = 1 / 3 := by
  sorry

end probability_yellow_face_l89_89110


namespace sum_of_roots_eq_six_l89_89027

variable (a b : ℝ)

theorem sum_of_roots_eq_six (h1 : a * (a - 6) = 7) (h2 : b * (b - 6) = 7) (h3 : a ≠ b) : a + b = 6 :=
sorry

end sum_of_roots_eq_six_l89_89027


namespace max_S_n_l89_89244

theorem max_S_n (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a (n + 1) = a n + d) (h2 : d < 0) (h3 : S 6 = 5 * (a 1) + 10 * d) :
  ∃ n, (n = 5 ∨ n = 6) ∧ (∀ m, S m ≤ S n) :=
by
  sorry

end max_S_n_l89_89244


namespace number_of_5_letter_words_with_at_least_one_vowel_l89_89671

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  let vowels := ['A', 'E']
  ∃ n : ℕ, n = 7^5 - 5^5 ∧ n = 13682 :=
by
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l89_89671


namespace matrix_norm_min_l89_89414

-- Definition of the matrix
def matrix_mul (a b c d : ℤ) : Option (ℤ × ℤ × ℤ × ℤ) :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 then
    some (a^2 + b * c, a * b + b * d, a * c + c * d, b * c + d^2)
  else
    none

-- Main theorem statement
theorem matrix_norm_min (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hc : c ≠ 0) (hd : d ≠ 0) :
  matrix_mul a b c d = some (8, 0, 0, 5) → 
  |a| + |b| + |c| + |d| = 9 :=
by
  sorry

end matrix_norm_min_l89_89414


namespace rational_solution_l89_89548

theorem rational_solution (a b c : ℚ) 
  (h : (3 * a - 2 * b + c - 4)^2 + (a + 2 * b - 3 * c + 6)^2 + (2 * a - b + 2 * c - 2)^2 ≤ 0) : 
  2 * a + b - 4 * c = -4 := 
by
  sorry

end rational_solution_l89_89548


namespace tangency_of_parabolas_l89_89437

theorem tangency_of_parabolas :
  ∃ x y : ℝ, y = x^2 + 12*x + 40
  ∧ x = y^2 + 44*y + 400
  ∧ x = -11 / 2
  ∧ y = -43 / 2 := by
sorry

end tangency_of_parabolas_l89_89437


namespace fraction_zero_implies_value_l89_89319

theorem fraction_zero_implies_value (x : ℝ) (h : (|x| - 2) / (x + 2) = 0) (h_non_zero : x + 2 ≠ 0) : x = 2 :=
sorry

end fraction_zero_implies_value_l89_89319


namespace largest_of_given_numbers_l89_89057

theorem largest_of_given_numbers :
  (0.99 > 0.9099) ∧
  (0.99 > 0.9) ∧
  (0.99 > 0.909) ∧
  (0.99 > 0.9009) →
  ∀ (x : ℝ), (x = 0.99 ∨ x = 0.9099 ∨ x = 0.9 ∨ x = 0.909 ∨ x = 0.9009) → 
  x ≤ 0.99 :=
by
  sorry

end largest_of_given_numbers_l89_89057


namespace simplify_120_div_180_l89_89122

theorem simplify_120_div_180 : (120 : ℚ) / 180 = 2 / 3 :=
by sorry

end simplify_120_div_180_l89_89122


namespace amount_spent_on_belt_correct_l89_89216

variable (budget shirt pants coat socks shoes remaining : ℕ)

-- Given conditions
def initial_budget : ℕ := 200
def spent_shirt : ℕ := 30
def spent_pants : ℕ := 46
def spent_coat : ℕ := 38
def spent_socks : ℕ := 11
def spent_shoes : ℕ := 41
def remaining_amount : ℕ := 16

-- The amount spent on the belt
def amount_spent_on_belt : ℕ :=
  budget - remaining - (shirt + pants + coat + socks + shoes)

-- The theorem statement we need to prove
theorem amount_spent_on_belt_correct :
  initial_budget = budget →
  spent_shirt = shirt →
  spent_pants = pants →
  spent_coat = coat →
  spent_socks = socks →
  spent_shoes = shoes →
  remaining_amount = remaining →
  amount_spent_on_belt budget shirt pants coat socks shoes remaining = 18 := by
    simp [initial_budget, spent_shirt, spent_pants, spent_coat, spent_socks, spent_shoes, remaining_amount, amount_spent_on_belt]
    sorry

end amount_spent_on_belt_correct_l89_89216


namespace fixed_point_parabola_l89_89910

theorem fixed_point_parabola (t : ℝ) : 4 * 3^2 + t * 3 - t^2 - 3 * t = 36 := by
  sorry

end fixed_point_parabola_l89_89910


namespace numbers_product_l89_89485

theorem numbers_product (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) : x * y = 128 := by
  sorry

end numbers_product_l89_89485


namespace members_play_both_eq_21_l89_89511

-- Given definitions
def TotalMembers := 80
def MembersPlayBadminton := 48
def MembersPlayTennis := 46
def MembersPlayNeither := 7

-- Inclusion-Exclusion Principle application to solve the problem
def MembersPlayBoth : ℕ := MembersPlayBadminton + MembersPlayTennis - (TotalMembers - MembersPlayNeither)

-- The theorem we want to prove
theorem members_play_both_eq_21 : MembersPlayBoth = 21 :=
by
  -- skipping the proof
  sorry

end members_play_both_eq_21_l89_89511


namespace length_of_pond_l89_89399

-- Define the problem conditions
variables (W L S : ℝ)
variables (h1 : L = 2 * W) (h2 : L = 24) 
variables (A_field A_pond : ℝ)
variables (h3 : A_pond = 1 / 8 * A_field)

-- State the theorem
theorem length_of_pond :
  A_field = L * W ∧ A_pond = S^2 ∧ A_pond = 1 / 8 * A_field ∧ L = 24 ∧ L = 2 * W → 
  S = 6 :=
by
  sorry

end length_of_pond_l89_89399


namespace mark_paid_more_than_anne_by_three_dollars_l89_89142

theorem mark_paid_more_than_anne_by_three_dollars :
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  mark_total - anne_total = 3 :=
by
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  sorry

end mark_paid_more_than_anne_by_three_dollars_l89_89142


namespace min_value_y_of_parabola_l89_89582

theorem min_value_y_of_parabola :
  ∃ y : ℝ, ∃ x : ℝ, (∀ y' x', (y' + x') = (y' - x')^2 + 3 * (y' - x') + 3 → y' ≥ y) ∧
            y = -1/2 :=
by
  sorry

end min_value_y_of_parabola_l89_89582


namespace complex_fraction_identity_l89_89016

theorem complex_fraction_identity
  (a b : ℂ) (ζ : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : ζ ^ 3 = 1) (h4 : ζ ≠ 1) 
  (h5 : a ^ 2 + a * b + b ^ 2 = 0) :
  (a ^ 9 + b ^ 9) / ((a - b) ^ 9) = (2 : ℂ) / (81 * (ζ - 1)) :=
sorry

end complex_fraction_identity_l89_89016


namespace inequality_sum_l89_89989

variable {a b c d : ℝ}

theorem inequality_sum (h1 : a > b) (h2 : c > d) : a + c > b + d := 
by {
  sorry
}

end inequality_sum_l89_89989


namespace num_valid_seating_arrangements_l89_89172

-- Define the dimensions of the examination room
def rows : Nat := 5
def columns : Nat := 6
def total_seats : Nat := rows * columns

-- Define the condition for students not sitting next to each other
def valid_seating_arrangements (rows columns : Nat) : Nat := sorry

-- The theorem to prove the number of seating arrangements
theorem num_valid_seating_arrangements : valid_seating_arrangements rows columns = 772 := 
by 
  sorry

end num_valid_seating_arrangements_l89_89172


namespace solve_x_1_solve_x_2_solve_x_3_l89_89425

-- Proof 1: Given 356 * x = 2492, prove that x = 7
theorem solve_x_1 (x : ℕ) (h : 356 * x = 2492) : x = 7 :=
sorry

-- Proof 2: Given x / 39 = 235, prove that x = 9165
theorem solve_x_2 (x : ℕ) (h : x / 39 = 235) : x = 9165 :=
sorry

-- Proof 3: Given 1908 - x = 529, prove that x = 1379
theorem solve_x_3 (x : ℕ) (h : 1908 - x = 529) : x = 1379 :=
sorry

end solve_x_1_solve_x_2_solve_x_3_l89_89425


namespace combined_balance_l89_89157

theorem combined_balance (b : ℤ) (g1 g2 : ℤ) (h1 : b = 3456) (h2 : g1 = b / 4) (h3 : g2 = b / 4) : g1 + g2 = 1728 :=
by {
  sorry
}

end combined_balance_l89_89157


namespace r_exceeds_s_by_two_l89_89611

theorem r_exceeds_s_by_two (x y r s : ℝ) (h1 : 3 * x + 2 * y = 16) (h2 : 5 * x + 3 * y = 26)
  (hr : r = x) (hs : s = y) : r - s = 2 :=
by
  sorry

end r_exceeds_s_by_two_l89_89611


namespace sunzi_wood_problem_l89_89022

theorem sunzi_wood_problem (x : ℝ) :
  (∃ (length_of_rope : ℝ), length_of_rope = x + 4.5 ∧
    ∃ (half_length_of_rope : ℝ), half_length_of_rope = length_of_rope / 2 ∧ 
      (half_length_of_rope + 1 = x)) ↔ 
  (1 / 2 * (x + 4.5) = x - 1) :=
by
  sorry

end sunzi_wood_problem_l89_89022


namespace same_cost_duration_l89_89912

-- Define the cost function for Plan A
def cost_plan_a (x : ℕ) : ℚ :=
 if x ≤ 8 then 0.60 else 0.60 + 0.06 * (x - 8)

-- Define the cost function for Plan B
def cost_plan_b (x : ℕ) : ℚ :=
 0.08 * x

-- The duration of a call for which the company charges the same under Plan A and Plan B is 14 minutes
theorem same_cost_duration (x : ℕ) : cost_plan_a x = cost_plan_b x ↔ x = 14 :=
by
  -- The proof is not required, using sorry to skip the proof steps
  sorry

end same_cost_duration_l89_89912


namespace midpoint_between_points_l89_89972

theorem midpoint_between_points : 
  let (x1, y1, z1) := (2, -3, 5)
  let (x2, y2, z2) := (8, 1, 3)
  (1 / 2 * (x1 + x2), 1 / 2 * (y1 + y2), 1 / 2 * (z1 + z2)) = (5, -1, 4) :=
by
  sorry

end midpoint_between_points_l89_89972


namespace hulk_first_jump_more_than_500_l89_89576

def hulk_jumping_threshold : Prop :=
  ∃ n : ℕ, (3^n > 500) ∧ (∀ m < n, 3^m ≤ 500)

theorem hulk_first_jump_more_than_500 : ∃ n : ℕ, n = 6 ∧ hulk_jumping_threshold :=
  sorry

end hulk_first_jump_more_than_500_l89_89576


namespace inequality_problem_l89_89585

theorem inequality_problem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by
  sorry

end inequality_problem_l89_89585


namespace common_divisors_9240_13860_l89_89984

def num_divisors (n : ℕ) : ℕ :=
  -- function to calculate the number of divisors (implementation is not provided here)
  sorry

theorem common_divisors_9240_13860 :
  let d := Nat.gcd 9240 13860
  d = 924 → num_divisors d = 24 := by
  intros d gcd_eq
  rw [gcd_eq]
  sorry

end common_divisors_9240_13860_l89_89984


namespace sum_series_eq_four_ninths_l89_89324

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l89_89324


namespace good_carrots_l89_89659

theorem good_carrots (Faye_picked : ℕ) (Mom_picked : ℕ) (bad_carrots : ℕ)
    (total_carrots : Faye_picked + Mom_picked = 28)
    (bad_carrots_count : bad_carrots = 16) : 
    28 - bad_carrots = 12 := by
  -- Proof goes here
  sorry

end good_carrots_l89_89659


namespace parabola_directrix_l89_89522

theorem parabola_directrix (a : ℝ) : 
  (∃ y, (y ^ 2 = 4 * a * (-2))) → a = 2 :=
by
  sorry

end parabola_directrix_l89_89522


namespace sqrt_15_estimate_l89_89510

theorem sqrt_15_estimate : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 :=
by
  sorry

end sqrt_15_estimate_l89_89510


namespace find_q_r_s_l89_89729

noncomputable def is_valid_geometry 
  (AD : ℝ) (AL : ℝ) (AM : ℝ) (AN : ℝ) (q : ℕ) (r : ℕ) (s : ℕ) : Prop :=
  AD = 10 ∧ AL = 3 ∧ AM = 3 ∧ AN = 3 ∧ ¬(∃ p : ℕ, p^2 ∣ s)

theorem find_q_r_s : ∃ (q r s : ℕ), is_valid_geometry 10 3 3 3 q r s ∧ q + r + s = 711 :=
by
  sorry

end find_q_r_s_l89_89729


namespace set_representation_l89_89271

def A (x : ℝ) := -3 < x ∧ x < 1
def B (x : ℝ) := x ≤ -1
def C (x : ℝ) := -2 < x ∧ x ≤ 2

theorem set_representation :
  (∀ x, A x ↔ (A x ∧ (B x ∨ C x))) ∧
  (∀ x, A x ↔ (A x ∨ (B x ∧ C x))) ∧
  (∀ x, A x ↔ ((A x ∧ B x) ∨ (A x ∧ C x))) :=
by
  sorry

end set_representation_l89_89271


namespace max_height_l89_89903

-- Given definitions
def height_eq (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 10

def max_height_problem : Prop :=
  ∃ t : ℝ, height_eq t = 74 ∧ ∀ t' : ℝ, height_eq t' ≤ height_eq t

-- Statement of the proof
theorem max_height : max_height_problem := sorry

end max_height_l89_89903


namespace fractions_sum_to_one_l89_89787

theorem fractions_sum_to_one :
  ∃ (a b c : ℕ), (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 1) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ ((a, b, c) = (2, 3, 6) ∨ (a, b, c) = (2, 6, 3) ∨ (a, b, c) = (3, 2, 6) ∨ (a, b, c) = (3, 6, 2) ∨ (a, b, c) = (6, 2, 3) ∨ (a, b, c) = (6, 3, 2)) :=
by
  sorry

end fractions_sum_to_one_l89_89787


namespace find_n_l89_89994

noncomputable def first_term_1 : ℝ := 12
noncomputable def second_term_1 : ℝ := 4
noncomputable def sum_first_series : ℝ := 18

noncomputable def first_term_2 : ℝ := 12
noncomputable def second_term_2 (n : ℝ) : ℝ := 4 + 2 * n
noncomputable def sum_second_series : ℝ := 90

theorem find_n (n : ℝ) : 
  (first_term_1 = 12) → 
  (second_term_1 = 4) → 
  (sum_first_series = 18) →
  (first_term_2 = 12) →
  (second_term_2 n = 4 + 2 * n) →
  (sum_second_series = 90) →
  (sum_second_series = 5 * sum_first_series) →
  n = 6 :=
by
  intros _ _ _ _ _ _ _
  sorry

end find_n_l89_89994


namespace inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l89_89478

theorem inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * a * c + 4 * b * c ∧
  (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * a * c + 4 * b * c → a = 0 ∧ b = 0 ∧ c = 0) := sorry

end inequality_5a2_5b2_5c2_ge_4ab_4ac_4bc_l89_89478


namespace union_of_A_and_B_l89_89158

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := 
by 
  sorry

end union_of_A_and_B_l89_89158


namespace parabola_standard_equation_l89_89904

variable {a : ℝ} (h : a < 0)

theorem parabola_standard_equation (h : a < 0) :
  ∃ (p : ℝ), p = -2 * a ∧ (∀ (x y : ℝ), y^2 = -2 * p * x ↔ y^2 = 4 * a * x) :=
sorry

end parabola_standard_equation_l89_89904


namespace magnitude_z1_condition_z2_range_condition_l89_89261

-- Define and set up the conditions and problem statements
open Complex

def complex_number_condition (z₁ : ℂ) (m : ℝ) : Prop :=
  z₁ = 1 + m * I ∧ ((z₁ * (1 - I)).re = 0)

def z₂_condition (z₂ z₁ : ℂ) (n : ℝ) : Prop :=
  z₂ = z₁ * (n - I) ∧ z₂.re < 0 ∧ z₂.im < 0

-- Prove that if z₁ = 1 + m * I and z₁ * (1 - I) is pure imaginary, then |z₁| = sqrt 2
theorem magnitude_z1_condition (m : ℝ) (z₁ : ℂ) 
  (h₁ : complex_number_condition z₁ m) : abs z₁ = Real.sqrt 2 :=
by sorry

-- Prove that if z₂ = z₁ * (n + i^3) is in the third quadrant, then n is in the range (-1, 1)
theorem z2_range_condition (n : ℝ) (m : ℝ) (z₁ z₂ : ℂ)
  (h₁ : complex_number_condition z₁ m)
  (h₂ : z₂_condition z₂ z₁ n) : -1 < n ∧ n < 1 :=
by sorry

end magnitude_z1_condition_z2_range_condition_l89_89261


namespace calculate_difference_l89_89920

theorem calculate_difference :
  let a := 3.56
  let b := 2.1
  let c := 1.5
  a - (b * c) = 0.41 :=
by
  let a := 3.56
  let b := 2.1
  let c := 1.5
  show a - (b * c) = 0.41
  sorry

end calculate_difference_l89_89920


namespace greatest_number_of_consecutive_integers_sum_to_91_l89_89874

theorem greatest_number_of_consecutive_integers_sum_to_91 :
  ∃ N, (∀ (a : ℤ), (N : ℕ) > 0 → (N * (2 * a + N - 1) = 182)) ∧ (N = 182) :=
by {
  sorry
}

end greatest_number_of_consecutive_integers_sum_to_91_l89_89874


namespace circle_radius_3_l89_89238

theorem circle_radius_3 :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 2 * y - 7 = 0) → (∃ r : ℝ, r = 3) :=
by
  sorry

end circle_radius_3_l89_89238


namespace no_always_1x3_rectangle_l89_89565

/-- From a sheet of graph paper measuring 8 x 8 cells, 12 rectangles of size 1 x 2 were cut out along the grid lines. 
Prove that it is not necessarily possible to always find a 1 x 3 checkered rectangle in the remaining part. -/
theorem no_always_1x3_rectangle (grid_size : ℕ) (rectangles_removed : ℕ) (rect_size : ℕ) :
  grid_size = 64 → rectangles_removed * rect_size = 24 → ¬ (∀ remaining_cells, remaining_cells ≥ 0 → remaining_cells ≤ 64 → ∃ (x y : ℕ), remaining_cells = x * y ∧ x = 1 ∧ y = 3) :=
  by
  intro h1 h2 h3
  /- Exact proof omitted for brevity -/
  sorry

end no_always_1x3_rectangle_l89_89565


namespace curve_is_line_l89_89982

noncomputable def curve_representation (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1) * (-1) = 0

theorem curve_is_line (x y : ℝ) (h : curve_representation x y) : 2 * x + 3 * y - 1 = 0 :=
by
  sorry

end curve_is_line_l89_89982


namespace opposite_number_l89_89152

theorem opposite_number (x : ℤ) (h : -x = -2) : x = 2 :=
sorry

end opposite_number_l89_89152


namespace men_per_table_l89_89571

theorem men_per_table (total_tables : ℕ) (women_per_table : ℕ) (total_customers : ℕ) (total_women : ℕ)
    (h1 : total_tables = 9)
    (h2 : women_per_table = 7)
    (h3 : total_customers = 90)
    (h4 : total_women = women_per_table * total_tables)
    (h5 : total_women + total_men = total_customers) :
  total_men / total_tables = 3 :=
by
  have total_women := 7 * 9
  have total_men := 90 - total_women
  exact sorry

end men_per_table_l89_89571


namespace vinnie_makes_more_l89_89310

-- Define the conditions
def paul_tips : ℕ := 14
def vinnie_tips : ℕ := 30

-- Define the theorem to prove
theorem vinnie_makes_more :
  vinnie_tips - paul_tips = 16 := by
  sorry

end vinnie_makes_more_l89_89310


namespace students_not_in_any_activity_l89_89758

def total_students : ℕ := 1500
def students_chorus : ℕ := 420
def students_band : ℕ := 780
def students_chorus_and_band : ℕ := 150
def students_drama : ℕ := 300
def students_drama_and_other : ℕ := 50

theorem students_not_in_any_activity :
  total_students - ((students_chorus + students_band - students_chorus_and_band) + (students_drama - students_drama_and_other)) = 200 :=
by
  sorry

end students_not_in_any_activity_l89_89758


namespace three_pow_255_mod_7_l89_89761

theorem three_pow_255_mod_7 : 3^255 % 7 = 6 :=
by 
  have h1 : 3^1 % 7 = 3 := by norm_num
  have h2 : 3^2 % 7 = 2 := by norm_num
  have h3 : 3^3 % 7 = 6 := by norm_num
  have h4 : 3^4 % 7 = 4 := by norm_num
  have h5 : 3^5 % 7 = 5 := by norm_num
  have h6 : 3^6 % 7 = 1 := by norm_num
  sorry

end three_pow_255_mod_7_l89_89761


namespace union_of_sets_l89_89614

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2, 6}) (hB : B = {2, 3, 6}) :
  A ∪ B = {1, 2, 3, 6} :=
by
  rw [hA, hB]
  ext x
  simp [Set.union]
  sorry

end union_of_sets_l89_89614


namespace triangle_inequality_l89_89882

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  (a / Real.sqrt (2*b^2 + 2*c^2 - a^2)) + (b / Real.sqrt (2*c^2 + 2*a^2 - b^2)) + 
  (c / Real.sqrt (2*a^2 + 2*b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end triangle_inequality_l89_89882


namespace johny_travelled_South_distance_l89_89618

theorem johny_travelled_South_distance :
  ∃ S : ℝ, S + (S + 20) + 2 * (S + 20) = 220 ∧ S = 40 :=
by
  sorry

end johny_travelled_South_distance_l89_89618


namespace kennedy_is_larger_l89_89309

-- Definitions based on given problem conditions
def KennedyHouse : ℕ := 10000
def BenedictHouse : ℕ := 2350
def FourTimesBenedictHouse : ℕ := 4 * BenedictHouse

-- Goal defined as a theorem to be proved
theorem kennedy_is_larger : KennedyHouse - FourTimesBenedictHouse = 600 :=
by 
  -- these are the conditions translated into Lean format
  let K := KennedyHouse
  let B := BenedictHouse
  let FourB := 4 * B
  let Goal := K - FourB
  -- prove the goal
  sorry

end kennedy_is_larger_l89_89309


namespace maximize_profit_l89_89899

variable {k : ℝ} (hk : k > 0)
variable {x : ℝ} (hx : 0 < x ∧ x < 0.06)

def deposit_volume (x : ℝ) : ℝ := k * x
def interest_paid (x : ℝ) : ℝ := k * x ^ 2
def profit (x : ℝ) : ℝ := (0.06 * k^2 * x) - (k * x^2)

theorem maximize_profit : 0.03 = x :=
by
  sorry

end maximize_profit_l89_89899


namespace reflected_rectangle_has_no_point_neg_3_4_l89_89384

structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq, Repr

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def is_not_vertex (pts: List Point) (p: Point) : Prop :=
  ¬ (p ∈ pts)

theorem reflected_rectangle_has_no_point_neg_3_4 :
  let initial_pts := [ Point.mk 1 3, Point.mk 1 1, Point.mk 4 1, Point.mk 4 3 ]
  let reflected_pts := initial_pts.map reflect_y
  is_not_vertex reflected_pts (Point.mk (-3) 4) :=
by
  sorry

end reflected_rectangle_has_no_point_neg_3_4_l89_89384


namespace k_value_l89_89930

theorem k_value (k m : ℤ) (h : (m - 8) ∣ (m^2 - k * m - 24)) : k = 5 := by
  have : (m - 8) ∣ (m^2 - 8 * m - 24) := sorry
  sorry

end k_value_l89_89930


namespace sum_pattern_l89_89245

theorem sum_pattern (a b : ℕ) : (6 + 7 = 13) ∧ (8 + 9 = 17) ∧ (5 + 6 = 11) ∧ (7 + 8 = 15) ∧ (3 + 3 = 6) → (6 + 7 = 12) :=
by
  sorry

end sum_pattern_l89_89245


namespace set_intersection_complement_equiv_l89_89600

open Set

variable {α : Type*}
variable {x : α}

def U : Set ℝ := univ
def M : Set ℝ := {x | 0 ≤ x}
def N : Set ℝ := {x | x^2 < 1}

theorem set_intersection_complement_equiv :
  M ∩ (U \ N) = {x | 1 ≤ x} :=
by
  sorry

end set_intersection_complement_equiv_l89_89600


namespace bases_to_make_equality_l89_89528

theorem bases_to_make_equality (a b : ℕ) (h : 3 * a^2 + 4 * a + 2 = 9 * b + 7) : 
  (3 * a^2 + 4 * a + 2 = 342) ∧ (9 * b + 7 = 97) :=
by
  sorry

end bases_to_make_equality_l89_89528


namespace not_a_cube_l89_89061

theorem not_a_cube (a b : ℤ) : ¬ ∃ c : ℤ, a^3 + b^3 + 4 = c^3 := 
sorry

end not_a_cube_l89_89061


namespace stratified_sampling_elderly_count_l89_89688

-- Definitions of conditions
def elderly := 30
def middleAged := 90
def young := 60
def totalPeople := elderly + middleAged + young
def sampleSize := 36
def samplingFraction := sampleSize / totalPeople
def expectedElderlySample := elderly * samplingFraction

-- The theorem we want to prove
theorem stratified_sampling_elderly_count : expectedElderlySample = 6 := 
by 
  -- Proof is omitted
  sorry

end stratified_sampling_elderly_count_l89_89688


namespace total_balls_l89_89762

theorem total_balls (jungkook_balls : ℕ) (yoongi_balls : ℕ) (h1 : jungkook_balls = 3) (h2 : yoongi_balls = 4) : 
  jungkook_balls + yoongi_balls = 7 :=
by
  -- This is a placeholder for the proof
  sorry

end total_balls_l89_89762


namespace find_f_of_3pi_by_4_l89_89001

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 2)

theorem find_f_of_3pi_by_4 : f (3 * Real.pi / 4) = -Real.sqrt 2 / 2 := by
  sorry

end find_f_of_3pi_by_4_l89_89001


namespace arithmetic_sequence_terms_l89_89365

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 3 + a 4 = 10) 
  (h2 : a (n - 3) + a (n - 2) = 30) 
  (h3 : (n * (a 1 + a n)) / 2 = 100) : 
  n = 10 :=
sorry

end arithmetic_sequence_terms_l89_89365


namespace supplementary_angle_ratio_l89_89779

theorem supplementary_angle_ratio (x : ℝ) (hx : 4 * x + x = 180) : x = 36 :=
by sorry

end supplementary_angle_ratio_l89_89779


namespace bertha_no_daughters_count_l89_89518

open Nat

-- Definitions for the conditions
def daughters : ℕ := 8
def total_women : ℕ := 42
def granddaughters : ℕ := total_women - daughters
def daughters_who_have_daughters := granddaughters / 6
def daughters_without_daughters := daughters - daughters_who_have_daughters
def total_without_daughters := granddaughters + daughters_without_daughters

-- The theorem to prove
theorem bertha_no_daughters_count : total_without_daughters = 37 := by
  sorry

end bertha_no_daughters_count_l89_89518


namespace average_speed_first_girl_l89_89037

theorem average_speed_first_girl (v : ℝ) 
  (start_same_point : True)
  (opp_directions : True)
  (avg_speed_second_girl : ℝ := 3)
  (distance_after_12_hours : (v + avg_speed_second_girl) * 12 = 120) :
  v = 7 :=
by
  sorry

end average_speed_first_girl_l89_89037


namespace limit_perimeters_eq_l89_89154

universe u

noncomputable def limit_perimeters (s : ℝ) : ℝ :=
  let a := 4 * s
  let r := 1 / 2
  a / (1 - r)

theorem limit_perimeters_eq (s : ℝ) : limit_perimeters s = 8 * s := by
  sorry

end limit_perimeters_eq_l89_89154


namespace ratio_of_democrats_l89_89648

theorem ratio_of_democrats (F M : ℕ) (h1 : F + M = 750) (h2 : (1/2 : ℚ) * F = 125) (h3 : (1/4 : ℚ) * M = 125) :
  (125 + 125 : ℚ) / 750 = 1 / 3 := by
  sorry

end ratio_of_democrats_l89_89648


namespace p_q_r_inequality_l89_89358

theorem p_q_r_inequality (p q r : ℝ) (h₁ : ∀ x, (x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) ↔ (x - p) * (x - q) ≤ 0) (h₂ : p < q) : p + 2 * q + 3 * r = 1 :=
by
  sorry

end p_q_r_inequality_l89_89358


namespace central_cell_value_l89_89003

theorem central_cell_value (a1 a2 a3 a4 a5 a6 a7 a8 C : ℕ) 
  (h1 : a1 + a3 + C = 13) (h2 : a2 + a4 + C = 13)
  (h3 : a5 + a7 + C = 13) (h4 : a6 + a8 + C = 13)
  (h5 : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 40) : 
  C = 3 := 
sorry

end central_cell_value_l89_89003


namespace solution_of_inequality_l89_89759

theorem solution_of_inequality (x : ℝ) : 3 * x > 2 * x + 4 ↔ x > 4 := 
sorry

end solution_of_inequality_l89_89759


namespace find_b_value_l89_89028

theorem find_b_value (x : ℝ) (h_neg : x < 0) (h_eq : 1 / (x + 1 / (x + 2)) = 2) : 
  x + 7 / 2 = 2 :=
sorry

end find_b_value_l89_89028


namespace monotonic_function_range_maximum_value_condition_function_conditions_l89_89094

-- Part (1): Monotonicity condition
theorem monotonic_function_range (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0) ↔ (m ≥ 3) := sorry

-- Part (2): Maximum value condition
theorem maximum_value_condition (m : ℝ) :
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4) ↔ (m = -2) := sorry

-- Combined statement (optional if you want to show entire problem in one go)
theorem function_conditions (m : ℝ) :
  (∀ x : ℝ, deriv (fun x => (m - 3) * x^3 + 9 * x) x ≥ 0 ∧ 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (m - 3) * 8 + 18 = 4)) ↔ (m = -2 ∨ m ≥ 3) := sorry

end monotonic_function_range_maximum_value_condition_function_conditions_l89_89094


namespace apples_total_l89_89847

theorem apples_total (lexie_apples : ℕ) (tom_apples : ℕ) (h1 : lexie_apples = 12) (h2 : tom_apples = 2 * lexie_apples) : lexie_apples + tom_apples = 36 :=
by
  sorry

end apples_total_l89_89847


namespace simplify_expression_l89_89220

theorem simplify_expression (y : ℝ) : 3 * y + 5 * y + 6 * y + 10 = 14 * y + 10 :=
by
  sorry

end simplify_expression_l89_89220


namespace salary_increase_l89_89436

theorem salary_increase (original_salary reduced_salary : ℝ) (hx : reduced_salary = original_salary * 0.5) : 
  (reduced_salary + reduced_salary * 1) = original_salary :=
by
  -- Prove the required increase percent to return to original salary
  sorry

end salary_increase_l89_89436


namespace quadratic_inequality_solution_set_l89_89536

theorem quadratic_inequality_solution_set
  (a b : ℝ)
  (h1 : 2 + 3 = -a)
  (h2 : 2 * 3 = b) :
  ∀ x : ℝ, 6 * x^2 - 5 * x + 1 > 0 ↔ x < (1 / 3) ∨ x > (1 / 2) := by
  sorry

end quadratic_inequality_solution_set_l89_89536


namespace Brittany_older_by_3_years_l89_89640

-- Define the necessary parameters as assumptions
variable (Rebecca_age : ℕ) (Brittany_return_age : ℕ) (vacation_years : ℕ)

-- Initial conditions
axiom h1 : Rebecca_age = 25
axiom h2 : Brittany_return_age = 32
axiom h3 : vacation_years = 4

-- Definition to capture Brittany's age before vacation
def Brittany_age_before_vacation (return_age vacation_period : ℕ) : ℕ := return_age - vacation_period

-- Theorem stating that Brittany is 3 years older than Rebecca
theorem Brittany_older_by_3_years :
  Brittany_age_before_vacation Brittany_return_age vacation_years - Rebecca_age = 3 :=
by
  sorry

end Brittany_older_by_3_years_l89_89640


namespace water_consumption_and_bill_34_7_l89_89246

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end water_consumption_and_bill_34_7_l89_89246


namespace calculate_ratio_l89_89019

theorem calculate_ratio (l m n : ℝ) :
  let D := (l + 1, 1, 1)
  let E := (1, m + 1, 1)
  let F := (1, 1, n + 1)
  let AB_sq := 4 * ((n - m) ^ 2)
  let AC_sq := 4 * ((l - n) ^ 2)
  let BC_sq := 4 * ((m - l) ^ 2)
  (AB_sq + AC_sq + BC_sq + 3) / (l^2 + m^2 + n^2 + 3) = 8 := by
  sorry

end calculate_ratio_l89_89019


namespace no_two_primes_sum_to_10003_l89_89457

-- Define what it means for a number to be a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the specific numbers involved
def even_prime : ℕ := 2
def target_number : ℕ := 10003
def candidate : ℕ := target_number - even_prime

-- State the main proposition in question
theorem no_two_primes_sum_to_10003 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = target_number :=
sorry

end no_two_primes_sum_to_10003_l89_89457


namespace inequality_proof_l89_89241

variable (a b c d : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d)
variable (h_sum : a + b + c + d = 1)

theorem inequality_proof :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ (1 / 27) + (176 / 27) * a * b * c * d :=
by
  sorry

end inequality_proof_l89_89241


namespace remainder_of_1999_pow_11_mod_8_l89_89206

theorem remainder_of_1999_pow_11_mod_8 :
  (1999 ^ 11) % 8 = 7 :=
  sorry

end remainder_of_1999_pow_11_mod_8_l89_89206


namespace store_sells_2_kg_per_week_l89_89494

def packets_per_week := 20
def grams_per_packet := 100
def grams_per_kg := 1000
def kg_per_week (p : Nat) (gr_per_pkt : Nat) (gr_per_kg : Nat) : Nat :=
  (p * gr_per_pkt) / gr_per_kg

theorem store_sells_2_kg_per_week :
  kg_per_week packets_per_week grams_per_packet grams_per_kg = 2 :=
  sorry

end store_sells_2_kg_per_week_l89_89494


namespace cannot_finish_third_l89_89290

variable (P Q R S T U : ℕ)
variable (beats : ℕ → ℕ → Prop)
variable (finishes_after : ℕ → ℕ → Prop)
variable (finishes_before : ℕ → ℕ → Prop)

noncomputable def race_conditions (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) : Prop :=
  beats P Q ∧
  beats P R ∧
  beats Q S ∧
  finishes_after T P ∧
  finishes_before T Q ∧
  finishes_after U R ∧
  beats U T

theorem cannot_finish_third (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) :
  race_conditions P Q R S T U beats finishes_after finishes_before →
  ¬ (finishes_before P T ∧ finishes_before T S ∧ finishes_after P R ∧ finishes_after P S) ∧ ¬ (finishes_before S T ∧ finishes_before T P) :=
sorry

end cannot_finish_third_l89_89290


namespace jose_work_time_l89_89114

-- Define the variables for days taken by Jose and Raju
variables (J R T : ℕ)

-- State the conditions:
-- 1. Raju completes work in 40 days
-- 2. Together, Jose and Raju complete work in 8 days
axiom ra_work : R = 40
axiom together_work : T = 8

-- State the theorem that needs to be proven:
theorem jose_work_time (J R T : ℕ) (h1 : R = 40) (h2 : T = 8) : J = 10 :=
sorry

end jose_work_time_l89_89114


namespace min_value_x_y_l89_89757

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 / (x + 1) + 1 / (y + 1) = 1) :
  x + y ≥ 14 :=
sorry

end min_value_x_y_l89_89757


namespace china_junior_1990_problem_l89_89020

theorem china_junior_1990_problem 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1)
  (h1 : a * x = y * z / (y + z))
  (h2 : b * y = x * z / (x + z))
  (h3 : c * z = x * y / (x + y)) :
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1) :=
sorry

end china_junior_1990_problem_l89_89020


namespace final_price_of_coat_after_discounts_l89_89872

def original_price : ℝ := 120
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.20

theorem final_price_of_coat_after_discounts : 
    (1 - second_discount) * (1 - first_discount) * original_price = 72 := 
by
    sorry

end final_price_of_coat_after_discounts_l89_89872


namespace derivative_at_1_derivative_at_neg_2_derivative_at_x0_l89_89375

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_at_1 : (deriv f 1) = -1 :=
sorry

theorem derivative_at_neg_2 : (deriv f (-2)) = 1 / 2 :=
sorry

theorem derivative_at_x0 (x0 : ℝ) : (deriv f x0) = -2 / (x0^2) + 1 :=
sorry

end derivative_at_1_derivative_at_neg_2_derivative_at_x0_l89_89375


namespace min_shift_symmetric_y_axis_l89_89013

theorem min_shift_symmetric_y_axis :
  ∃ (m : ℝ), m = 7 * Real.pi / 6 ∧ 
             (∀ x : ℝ, 2 * Real.cos (x + Real.pi / 3) = 2 * Real.cos (x + Real.pi / 3 + m)) ∧ 
             m > 0 :=
by
  sorry

end min_shift_symmetric_y_axis_l89_89013


namespace find_second_month_sale_l89_89592

/-- Given sales for specific months and required sales goal -/
def sales_1 := 4000
def sales_3 := 5689
def sales_4 := 7230
def sales_5 := 6000
def sales_6 := 12557
def avg_goal := 7000
def months := 6

theorem find_second_month_sale (x2 : ℕ) :
  (sales_1 + x2 + sales_3 + sales_4 + sales_5 + sales_6) / months = avg_goal →
  x2 = 6524 :=
by
  sorry

end find_second_month_sale_l89_89592


namespace max_min_values_l89_89153

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem max_min_values :
  let max_val := 2
  let min_val := -25
  ∃ x_max x_min, 
    0 ≤ x_max ∧ x_max ≤ 4 ∧ f x_max = max_val ∧ 
    0 ≤ x_min ∧ x_min ≤ 4 ∧ f x_min = min_val :=
sorry

end max_min_values_l89_89153


namespace neither_jia_nor_yi_has_winning_strategy_l89_89185

/-- 
  There are 99 points, each marked with a number from 1 to 99, placed 
  on 99 equally spaced points on a circle. Jia and Yi take turns 
  placing one piece at a time, with Jia going first. The player who 
  first makes the numbers on three consecutive points form an 
  arithmetic sequence wins. Prove that neither Jia nor Yi has a 
  guaranteed winning strategy, and both possess strategies to avoid 
  losing.
-/
theorem neither_jia_nor_yi_has_winning_strategy :
  ∀ (points : Fin 99 → ℕ), -- 99 points on the circle
  (∀ i, 1 ≤ points i ∧ points i ≤ 99) → -- Each point is numbered between 1 and 99
  ¬(∃ (player : Fin 99 → ℕ) (h : ∀ (i : Fin 99), player i ≠ 0 ∧ (player i = 1 ∨ player i = 2)),
    ∃ i : Fin 99, (points i + points (i + 1) + points (i + 2)) / 3 = points i)
:=
by
  sorry

end neither_jia_nor_yi_has_winning_strategy_l89_89185


namespace equation_result_l89_89813

theorem equation_result : 
  ∀ (n : ℝ), n = 5.0 → (4 * n + 7 * n) = 55.0 :=
by
  intro n h
  rw [h]
  norm_num

end equation_result_l89_89813


namespace Alex_age_l89_89491

theorem Alex_age : ∃ (x : ℕ), (∃ (y : ℕ), x - 2 = y^2) ∧ (∃ (z : ℕ), x + 2 = z^3) ∧ x = 6 := by
  sorry

end Alex_age_l89_89491


namespace brick_height_calculation_l89_89817

theorem brick_height_calculation :
  ∀ (num_bricks : ℕ) (brick_length brick_width brick_height : ℝ)
    (wall_length wall_height wall_width : ℝ),
    num_bricks = 1600 →
    brick_length = 100 →
    brick_width = 11.25 →
    wall_length = 800 →
    wall_height = 600 →
    wall_width = 22.5 →
    wall_length * wall_height * wall_width = 
    num_bricks * brick_length * brick_width * brick_height →
    brick_height = 60 :=
by
  sorry

end brick_height_calculation_l89_89817


namespace volume_of_prism_l89_89100

noncomputable def volume_of_triangular_prism
  (area_lateral_face : ℝ)
  (distance_cc1_to_lateral_face : ℝ) : ℝ :=
  area_lateral_face * distance_cc1_to_lateral_face

theorem volume_of_prism (area_lateral_face : ℝ) 
    (distance_cc1_to_lateral_face : ℝ)
    (h_area : area_lateral_face = 4)
    (h_distance : distance_cc1_to_lateral_face = 2):
  volume_of_triangular_prism area_lateral_face distance_cc1_to_lateral_face = 4 := by
  sorry

end volume_of_prism_l89_89100


namespace area_of_second_side_l89_89186

theorem area_of_second_side 
  (L W H : ℝ) 
  (h1 : L * H = 120) 
  (h2 : L * W = 60) 
  (h3 : L * W * H = 720) : 
  W * H = 72 :=
sorry

end area_of_second_side_l89_89186


namespace largest_difference_l89_89350

noncomputable def A : ℕ := 3 * 2010 ^ 2011
noncomputable def B : ℕ := 2010 ^ 2011
noncomputable def C : ℕ := 2009 * 2010 ^ 2010
noncomputable def D : ℕ := 3 * 2010 ^ 2010
noncomputable def E : ℕ := 2010 ^ 2010
noncomputable def F : ℕ := 2010 ^ 2009

theorem largest_difference :
  (A - B = 2 * 2010 ^ 2011) ∧ 
  (B - C = 2010 ^ 2010) ∧ 
  (C - D = 2006 * 2010 ^ 2010) ∧ 
  (D - E = 2 * 2010 ^ 2010) ∧ 
  (E - F = 2009 * 2010 ^ 2009) ∧ 
  (2 * 2010 ^ 2011 > 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2006 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2009 * 2010 ^ 2009) :=
by
  sorry

end largest_difference_l89_89350


namespace counterexample_exists_l89_89327

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := ¬ is_prime n ∧ n > 1

theorem counterexample_exists :
  ∃ n, is_composite n ∧ is_composite (n - 3) ∧ n = 18 := by
  sorry

end counterexample_exists_l89_89327


namespace wrapping_paper_needs_l89_89584

theorem wrapping_paper_needs :
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  first_present + second_present + third_present = 7 := by
  let first_present := 2
  let second_present := (3 / 4) * first_present
  let third_present := first_present + second_present
  sorry

end wrapping_paper_needs_l89_89584


namespace problem_proof_l89_89132

theorem problem_proof :
  (3 ∣ 18) ∧
  (17 ∣ 187 ∧ ¬ (17 ∣ 52)) ∧
  ¬ ((24 ∣ 72) ∧ (24 ∣ 67)) ∧
  ¬ (13 ∣ 26 ∧ ¬ (13 ∣ 52)) ∧
  (8 ∣ 160) :=
by 
  sorry

end problem_proof_l89_89132


namespace abigail_lost_money_l89_89472

-- Conditions
def initial_money := 11
def money_spent := 2
def money_left := 3

-- Statement of the problem as a Lean theorem
theorem abigail_lost_money : initial_money - money_spent - money_left = 6 := by
  sorry

end abigail_lost_money_l89_89472


namespace octahedron_vertices_sum_l89_89963

noncomputable def octahedron_faces_sum (a b c d e f : ℕ) : ℕ :=
  a + b + c + d + e + f

theorem octahedron_vertices_sum (a b c d e f : ℕ) 
  (h : 8 * (octahedron_faces_sum a b c d e f) = 440) : 
  octahedron_faces_sum a b c d e f = 147 :=
by
  sorry

end octahedron_vertices_sum_l89_89963


namespace pet_store_cages_l89_89448

theorem pet_store_cages 
  (snakes parrots rabbits snake_cage_capacity parrot_cage_capacity rabbit_cage_capacity : ℕ)
  (h_snakes : snakes = 4) 
  (h_parrots : parrots = 6) 
  (h_rabbits : rabbits = 8) 
  (h_snake_cage_capacity : snake_cage_capacity = 2) 
  (h_parrot_cage_capacity : parrot_cage_capacity = 3) 
  (h_rabbit_cage_capacity : rabbit_cage_capacity = 4) 
  : (snakes / snake_cage_capacity) + (parrots / parrot_cage_capacity) + (rabbits / rabbit_cage_capacity) = 6 := 
by 
  sorry

end pet_store_cages_l89_89448


namespace line_always_passes_through_fixed_point_l89_89267

theorem line_always_passes_through_fixed_point (k : ℝ) : 
  ∀ x y, y + 2 = k * (x + 1) → (x = -1 ∧ y = -2) :=
by
  sorry

end line_always_passes_through_fixed_point_l89_89267


namespace lava_lamp_probability_l89_89119

/-- Ryan has 4 red lava lamps and 2 blue lava lamps; 
    he arranges them in a row on a shelf randomly, and then randomly turns 3 of them on. 
    Prove that the probability that the leftmost lamp is blue and off, 
    and the rightmost lamp is red and on is 2/25. -/
theorem lava_lamp_probability : 
  let total_arrangements := (Nat.choose 6 2) 
  let total_on := (Nat.choose 6 3)
  let favorable_arrangements := (Nat.choose 4 1)
  let favorable_on := (Nat.choose 4 2)
  let favorable_outcomes := 4 * 6
  let probability := (favorable_outcomes : ℚ) / (total_arrangements * total_on : ℚ)
  probability = 2 / 25 := 
by
  sorry

end lava_lamp_probability_l89_89119


namespace steps_in_staircase_l89_89594

theorem steps_in_staircase :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n = 19 :=
by
  sorry

end steps_in_staircase_l89_89594


namespace max_value_of_expression_l89_89816

theorem max_value_of_expression (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  8 * a + 3 * b + 5 * c ≤ 7 * Real.sqrt 2 :=
sorry

end max_value_of_expression_l89_89816


namespace joseph_cards_l89_89593

theorem joseph_cards (cards_per_student : ℕ) (students : ℕ) (cards_left : ℕ) 
    (H1 : cards_per_student = 23)
    (H2 : students = 15)
    (H3 : cards_left = 12) 
    : (cards_per_student * students + cards_left = 357) := 
  by
  sorry

end joseph_cards_l89_89593


namespace jessica_monthly_car_insurance_payment_l89_89914

theorem jessica_monthly_car_insurance_payment
  (rent_last_year : ℤ := 1000)
  (food_last_year : ℤ := 200)
  (car_insurance_last_year : ℤ)
  (rent_increase_rate : ℕ := 3 / 10)
  (food_increase_rate : ℕ := 1 / 2)
  (car_insurance_increase_rate : ℕ := 3)
  (additional_expenses_this_year : ℤ := 7200) :
  car_insurance_last_year = 300 :=
by
  sorry

end jessica_monthly_car_insurance_payment_l89_89914


namespace other_position_in_arithmetic_progression_l89_89396

theorem other_position_in_arithmetic_progression 
  (a d : ℝ) (x : ℕ)
  (h1 : a + (4 - 1) * d + a + (x - 1) * d = 20)
  (h2 : 5 * (2 * a + 9 * d) = 100) :
  x = 7 := by
  sorry

end other_position_in_arithmetic_progression_l89_89396


namespace kira_travel_time_l89_89597

theorem kira_travel_time :
  let time_between_stations := 2 * 60 -- converting hours to minutes
  let break_time := 30 -- in minutes
  let total_time := 2 * time_between_stations + break_time
  total_time = 270 :=
by
  let time_between_stations := 2 * 60
  let break_time := 30
  let total_time := 2 * time_between_stations + break_time
  exact rfl

end kira_travel_time_l89_89597


namespace train_length_is_250_l89_89430

-- Define the length of the train
def train_length (L : ℝ) (V : ℝ) :=
  -- Condition 1
  (V = L / 10) → 
  -- Condition 2
  (V = (L + 1250) / 60) → 
  -- Question
  L = 250

-- Here's the statement that we expect to prove
theorem train_length_is_250 (L V : ℝ) : train_length L V :=
by {
  -- sorry is a placeholder to indicate the theorem proof is omitted
  sorry
}

end train_length_is_250_l89_89430


namespace find_number_l89_89696

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 129) : x = 19 :=
by
  sorry

end find_number_l89_89696


namespace smallest_pieces_left_l89_89754

theorem smallest_pieces_left (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) : 
    ∃ k, (k = 2 ∧ (m * n) % 3 = 0) ∨ (k = 1 ∧ (m * n) % 3 ≠ 0) :=
by
    sorry

end smallest_pieces_left_l89_89754


namespace trapezoid_area_l89_89049

/-- Given that the area of the outer square is 36 square units and the area of the inner square is 
4 square units, the area of one of the four congruent trapezoids formed between the squares is 8 
square units. -/
theorem trapezoid_area (outer_square_area inner_square_area : ℕ) 
  (h_outer : outer_square_area = 36)
  (h_inner : inner_square_area = 4) : 
  (outer_square_area - inner_square_area) / 4 = 8 :=
by sorry

end trapezoid_area_l89_89049


namespace total_distance_of_trip_l89_89953

theorem total_distance_of_trip (x : ℚ)
  (highway : x / 4 ≤ x)
  (city : 30 ≤ x)
  (country : x / 6 ≤ x)
  (middle_part_fraction : 1 - 1 / 4 - 1 / 6 = 7 / 12) :
  (7 / 12) * x = 30 → x = 360 / 7 :=
by
  sorry

end total_distance_of_trip_l89_89953


namespace linear_function_value_l89_89120

theorem linear_function_value
  (a b c : ℝ)
  (h1 : 3 * a + b = 8)
  (h2 : -2 * a + b = 3)
  (h3 : -3 * a + b = c) :
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 13 :=
by
  sorry

end linear_function_value_l89_89120


namespace tangent_line_equation_l89_89978

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x1 : ℝ := 1
  let y1 : ℝ := f 1
  ∀ x y : ℝ, 
    (y - y1 = (1 / (x1 + 1)) * (x - x1)) ↔ 
    (x - 2 * y + 2 * Real.log 2 - 1 = 0) :=
by
  sorry

end tangent_line_equation_l89_89978


namespace total_bouquets_sold_l89_89733

-- defining the sale conditions
def monday_bouquets := 12
def tuesday_bouquets := 3 * monday_bouquets
def wednesday_bouquets := tuesday_bouquets / 3

-- defining the total sale
def total_bouquets := monday_bouquets + tuesday_bouquets + wednesday_bouquets

-- stating the theorem
theorem total_bouquets_sold : total_bouquets = 60 := by
  -- the proof would go here
  sorry

end total_bouquets_sold_l89_89733


namespace geometric_sequence_ratio_l89_89698

theorem geometric_sequence_ratio
  (a₁ : ℝ) (q : ℝ) (hq : q ≠ 1)
  (S : ℕ → ℝ)
  (hS₃ : S 3 = a₁ * (1 - q^3) / (1 - q))
  (hS₆ : S 6 = a₁ * (1 - q^6) / (1 - q))
  (hS₃_val : S 3 = 2)
  (hS₆_val : S 6 = 18) :
  S 10 / S 5 = 1 + 2^(1/3) + 2^(2/3) :=
sorry

end geometric_sequence_ratio_l89_89698


namespace uranus_appears_7_minutes_after_6AM_l89_89870

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end uranus_appears_7_minutes_after_6AM_l89_89870


namespace Q_div_P_l89_89240

theorem Q_div_P (P Q : ℚ) (h : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
  P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x * (x + 3) * (x - 5))) :
  Q / P = 1 / 3 :=
by
  sorry

end Q_div_P_l89_89240


namespace find_a_l89_89705

theorem find_a (a : ℝ) :
  let A := {5}
  let B := { x : ℝ | a * x - 1 = 0 }
  A ∩ B = B ↔ (a = 0 ∨ a = 1 / 5) :=
by
  sorry

end find_a_l89_89705


namespace complement_A_in_U_l89_89703

noncomputable def U : Set ℕ := {0, 1, 2}
noncomputable def A : Set ℕ := {x | x^2 - x = 0}
noncomputable def complement_U (A : Set ℕ) : Set ℕ := U \ A

theorem complement_A_in_U : 
  complement_U {x | x^2 - x = 0} = {2} := 
sorry

end complement_A_in_U_l89_89703


namespace largest_number_of_square_plots_l89_89366

/-- A rectangular field measures 30 meters by 60 meters with 2268 meters of internal fencing to partition into congruent, square plots. The entire field must be partitioned with sides of squares parallel to the edges. Prove the largest number of square plots is 722. -/
theorem largest_number_of_square_plots (s n : ℕ) (h_length : 60 = n * s) (h_width : 30 = s * 2 * n) (h_fence : 120 * n - 90 ≤ 2268) :
(s * 2 * n) = 722 :=
sorry

end largest_number_of_square_plots_l89_89366


namespace sin_half_angle_identity_l89_89897

theorem sin_half_angle_identity (theta : ℝ) (h : Real.sin (Real.pi / 2 + theta) = - 1 / 2) :
  2 * Real.sin (theta / 2) ^ 2 - 1 = 1 / 2 := 
by
  sorry

end sin_half_angle_identity_l89_89897


namespace find_pairs_of_numbers_l89_89432

theorem find_pairs_of_numbers (a b : ℝ) :
  (a^2 + b^2 = 15 * (a + b)) ∧ (a^2 - b^2 = 3 * (a - b) ∨ a^2 - b^2 = -3 * (a - b))
  ↔ (a = 6 ∧ b = -3) ∨ (a = -3 ∧ b = 6) ∨ (a = 0 ∧ b = 0) ∨ (a = 15 ∧ b = 15) :=
sorry

end find_pairs_of_numbers_l89_89432


namespace freds_sister_borrowed_3_dimes_l89_89033

-- Define the conditions
def original_dimes := 7
def remaining_dimes := 4

-- Define the question and answer
def borrowed_dimes := original_dimes - remaining_dimes

-- Statement to prove
theorem freds_sister_borrowed_3_dimes : borrowed_dimes = 3 := by
  sorry

end freds_sister_borrowed_3_dimes_l89_89033


namespace elderly_people_pears_l89_89979

theorem elderly_people_pears (x y : ℕ) :
  (y = x + 1) ∧ (2 * x = y + 2) ↔
  (x = y - 1) ∧ (2 * x = y + 2) := by
  sorry

end elderly_people_pears_l89_89979


namespace weather_forecast_probability_l89_89620

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem weather_forecast_probability :
  binomial_probability 3 2 0.8 = 0.384 :=
by
  sorry

end weather_forecast_probability_l89_89620


namespace initial_speed_l89_89801

variable (D T : ℝ) -- Total distance D and total time T
variable (S : ℝ)   -- Initial speed S

theorem initial_speed :
  (2 * D / 3) = (S * T / 3) →
  (35 = (D / (2 * T))) →
  S = 70 :=
by
  intro h1 h2
  -- Skipping the proof with 'sorry'
  sorry

end initial_speed_l89_89801


namespace decimal_sum_difference_l89_89506

theorem decimal_sum_difference :
  (0.5 - 0.03 + 0.007 + 0.0008 = 0.4778) :=
by
  sorry

end decimal_sum_difference_l89_89506


namespace tank_filling_time_l89_89992

-- Define the rates at which pipes fill or drain the tank
def capacity : ℕ := 1200
def rate_A : ℕ := 50
def rate_B : ℕ := 35
def rate_C : ℕ := 20
def rate_D : ℕ := 40

-- Define the times each pipe is open
def time_A : ℕ := 2
def time_B : ℕ := 4
def time_C : ℕ := 3
def time_D : ℕ := 5

-- Calculate the total time for one cycle
def cycle_time : ℕ := time_A + time_B + time_C + time_D

-- Calculate the net amount of water added in one cycle
def net_amount_per_cycle : ℕ := (rate_A * time_A) + (rate_B * time_B) + (rate_C * time_C) - (rate_D * time_D)

-- Calculate the number of cycles needed to fill the tank
def num_cycles : ℕ := capacity / net_amount_per_cycle

-- Calculate the total time to fill the tank
def total_time : ℕ := num_cycles * cycle_time

-- Prove that the total time to fill the tank is 168 minutes
theorem tank_filling_time : total_time = 168 := by
  sorry

end tank_filling_time_l89_89992


namespace least_possible_value_of_z_minus_w_l89_89480

variable (x y z w k m : Int)
variable (h1 : Even x)
variable (h2 : Odd y)
variable (h3 : Odd z)
variable (h4 : ∃ n : Int, w = - (2 * n + 1) / 3)
variable (h5 : w < x)
variable (h6 : x < y)
variable (h7 : y < z)
variable (h8 : 0 < k)
variable (h9 : (y - x) > k)
variable (h10 : 0 < m)
variable (h11 : (z - w) > m)
variable (h12 : k > m)

theorem least_possible_value_of_z_minus_w
  : z - w = 6 := sorry

end least_possible_value_of_z_minus_w_l89_89480


namespace cost_price_of_table_l89_89276

theorem cost_price_of_table (CP : ℝ) (SP : ℝ) (h1 : SP = CP * 1.10) (h2 : SP = 8800) : CP = 8000 :=
by
  sorry

end cost_price_of_table_l89_89276


namespace find_c_l89_89539

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem find_c (h1 : x = 2.5 * y) (h2 : 2 * y = (c / 100) * x) : c = 80 :=
sorry

end find_c_l89_89539


namespace jose_cupcakes_l89_89293

theorem jose_cupcakes (lemons_needed : ℕ) (tablespoons_per_lemon : ℕ) (tablespoons_per_dozen : ℕ) (target_lemons : ℕ) : 
  (lemons_needed = 12) → 
  (tablespoons_per_lemon = 4) → 
  (target_lemons = 9) → 
  ((target_lemons * tablespoons_per_lemon / lemons_needed) = 3) :=
by
  intros h1 h2 h3
  sorry

end jose_cupcakes_l89_89293


namespace adjustments_to_equal_boys_and_girls_l89_89509

theorem adjustments_to_equal_boys_and_girls (n : ℕ) :
  let initial_boys := 40
  let initial_girls := 0
  let boys_after_n := initial_boys - 3 * n
  let girls_after_n := initial_girls + 2 * n
  boys_after_n = girls_after_n → n = 8 :=
by
  sorry

end adjustments_to_equal_boys_and_girls_l89_89509


namespace N_def_M_intersection_CU_N_def_M_union_N_def_l89_89803

section Sets

variable {α : Type}

-- Declarations of conditions
def U := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def M := {x : ℝ | -1 < x ∧ x < 1}
def CU (N : Set ℝ) := {x : ℝ | 0 < x ∧ x < 2}

-- Problem statements
theorem N_def (N : Set ℝ) : N = {x : ℝ | (-3 ≤ x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x ≤ 3)} ↔ CU N = {x : ℝ | 0 < x ∧ x < 2} :=
by sorry

theorem M_intersection_CU_N_def (N : Set ℝ) : (M ∩ CU N) = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

theorem M_union_N_def (N : Set ℝ) : (M ∪ N) = {x : ℝ | (-3 ≤ x ∧ x < 1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
by sorry

end Sets

end N_def_M_intersection_CU_N_def_M_union_N_def_l89_89803


namespace water_fee_part1_water_fee_part2_water_fee_usage_l89_89155

theorem water_fee_part1 (x : ℕ) (h : 0 < x ∧ x ≤ 6) : y = 2 * x :=
sorry

theorem water_fee_part2 (x : ℕ) (h : x > 6) : y = 3 * x - 6 :=
sorry

theorem water_fee_usage (y : ℕ) (h : y = 27) : x = 11 :=
sorry

end water_fee_part1_water_fee_part2_water_fee_usage_l89_89155


namespace find_number_l89_89785

variable (x : ℝ)

theorem find_number (hx : 5100 - (102 / x) = 5095) : x = 20.4 := 
by
  sorry

end find_number_l89_89785


namespace polynomial_divisibility_l89_89537

theorem polynomial_divisibility (n : ℕ) (h : 0 < n) : 
  ∃ g : Polynomial ℚ, 
    (Polynomial.X + 1)^(2*n + 1) + Polynomial.X^(n + 2) = g * (Polynomial.X^2 + Polynomial.X + 1) := 
by
  sorry

end polynomial_divisibility_l89_89537


namespace find_percentage_l89_89044

variable (P : ℝ)
variable (num : ℝ := 70)
variable (result : ℝ := 25)

theorem find_percentage (h : ((P / 100) * num) - 10 = result) : P = 50 := by
  sorry

end find_percentage_l89_89044


namespace find_valid_pairs_l89_89089

theorem find_valid_pairs (x y : ℤ) : 
  (x^3 + y) % (x^2 + y^2) = 0 ∧ 
  (x + y^3) % (x^2 + y^2) = 0 ↔ 
  (x, y) = (1, 1) ∨ (x, y) = (1, 0) ∨ (x, y) = (1, -1) ∨ 
  (x, y) = (0, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (-1, 1) ∨ 
  (x, y) = (-1, 0) ∨ (x, y) = (-1, -1) :=
sorry

end find_valid_pairs_l89_89089


namespace hyperbola_distance_to_left_focus_l89_89357

theorem hyperbola_distance_to_left_focus (P : ℝ × ℝ)
  (h1 : (P.1^2) / 9 - (P.2^2) / 16 = 1)
  (dPF2 : dist P (4, 0) = 4) : dist P (-4, 0) = 10 := 
sorry

end hyperbola_distance_to_left_focus_l89_89357


namespace f_2012_l89_89169

noncomputable def f : ℝ → ℝ := sorry -- provided as a 'sorry' to be determined

axiom odd_function (hf : ℝ → ℝ) : ∀ x : ℝ, hf (-x) = -hf (x)

axiom f_shift : ∀ x : ℝ, f (x + 3) = -f (x)
axiom f_one : f 1 = 2

theorem f_2012 : f 2012 = 2 :=
by
  -- proofs would go here, but 'sorry' is enough to define the theorem statement
  sorry

end f_2012_l89_89169


namespace isosceles_triangle_largest_angle_l89_89661

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_isosceles : A = B) (h_angles : A = 60 ∧ B = 60) :
  max A (max B C) = 60 :=
by
  sorry

end isosceles_triangle_largest_angle_l89_89661


namespace original_polygon_sides_l89_89647

noncomputable def sum_of_interior_angles (n : ℕ) : ℝ :=
(n - 2) * 180

theorem original_polygon_sides (x : ℕ) (h1 : sum_of_interior_angles (2 * x) = 2160) : x = 7 :=
by
  sorry

end original_polygon_sides_l89_89647


namespace race_course_length_l89_89069

variable (v d : ℝ)

theorem race_course_length (h1 : 4 * v > 0) (h2 : ∀ t : ℝ, t > 0 → (d / (4 * v)) = ((d - 72) / v)) : d = 96 := by
  sorry

end race_course_length_l89_89069


namespace tan_double_angle_l89_89862

theorem tan_double_angle (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : Real.tan (2 * α) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l89_89862


namespace earnings_total_l89_89828

-- Define the earnings for each day based on given conditions
def Monday_earnings : ℝ := 0.20 * 10 * 3
def Tuesday_earnings : ℝ := 0.25 * 12 * 4
def Wednesday_earnings : ℝ := 0.10 * 15 * 5
def Thursday_earnings : ℝ := 0.15 * 8 * 6
def Friday_earnings : ℝ := 0.30 * 20 * 2

-- Compute total earnings over the five days
def total_earnings : ℝ :=
  Monday_earnings + Tuesday_earnings + Wednesday_earnings + Thursday_earnings + Friday_earnings

-- Lean statement to prove the total earnings
theorem earnings_total :
  total_earnings = 44.70 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end earnings_total_l89_89828


namespace xy_fraction_equivalence_l89_89211

theorem xy_fraction_equivalence
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (x^2 + 4 * x * y) / (y^2 - 4 * x * y) = 3) :
  (x^2 - 4 * x * y) / (y^2 + 4 * x * y) = -1 :=
sorry

end xy_fraction_equivalence_l89_89211


namespace garden_area_l89_89635

theorem garden_area (length perimeter : ℝ) (length_50 : 50 * length = 1500) (perimeter_20 : 20 * perimeter = 1500) (rectangular : perimeter = 2 * length + 2 * (perimeter / 2 - length)) :
  length * (perimeter / 2 - length) = 225 := 
by
  sorry

end garden_area_l89_89635


namespace arctan_sum_pi_over_four_l89_89128

theorem arctan_sum_pi_over_four (a b c : ℝ) (C : ℝ) (h : Real.sin C = c / (a + b + c)) :
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 :=
sorry

end arctan_sum_pi_over_four_l89_89128


namespace work_ratio_l89_89444

theorem work_ratio (r : ℕ) (w : ℕ) (m₁ m₂ d₁ d₂ : ℕ)
  (h₁ : m₁ = 5) 
  (h₂ : d₁ = 15) 
  (h₃ : m₂ = 3) 
  (h₄ : d₂ = 25)
  (h₅ : w = (m₁ * r * d₁) + (m₂ * r * d₂)) :
  ((m₁ * r * d₁):ℚ) / (w:ℚ) = 1 / 2 := by
  sorry

end work_ratio_l89_89444


namespace total_pennies_l89_89826

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l89_89826


namespace range_of_a_l89_89631

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, (2 * (x : ℝ) - 7 < 0) ∧ ((x : ℝ) - a > 0) ↔ (x = 3)) →
  (2 ≤ a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l89_89631


namespace derivative_not_in_second_quadrant_l89_89550

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (b c x : ℝ) : ℝ := x^2 + b * x + c
noncomputable def f_derivative (x : ℝ) : ℝ := 2 * x - 4

-- Given condition: Axis of symmetry is x = 2
def axis_of_symmetry (b : ℝ) : Prop := b = -4

-- Additional condition: behavior of the derivative and quadrant check
def not_in_second_quadrant (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f' x < 0

-- The main theorem to be proved
theorem derivative_not_in_second_quadrant (b c : ℝ) (h : axis_of_symmetry b) :
  not_in_second_quadrant f_derivative :=
by {
  sorry
}

end derivative_not_in_second_quadrant_l89_89550


namespace men_build_walls_l89_89745

-- Define the variables
variables (a b d y : ℕ)

-- Define the work rate based on given conditions
def rate := d / (a * b)

-- Theorem to prove that y equals (a * a) / d given the conditions
theorem men_build_walls (h : a * b * y = a * a * d / a) : 
  y = a * a / d :=
by sorry

end men_build_walls_l89_89745


namespace range_of_f_l89_89134

noncomputable def f (t : ℝ) : ℝ := (t^2 + (1/2)*t) / (t^2 + 1)

theorem range_of_f : Set.Icc (-1/4 : ℝ) (1/4) = Set.range f :=
by
  sorry

end range_of_f_l89_89134


namespace number_of_children_l89_89973

theorem number_of_children (crayons_per_child total_crayons : ℕ) (h1 : crayons_per_child = 12) (h2 : total_crayons = 216) : total_crayons / crayons_per_child = 18 :=
by
  have h3 : total_crayons / crayons_per_child = 216 / 12 := by rw [h1, h2]
  norm_num at h3
  exact h3

end number_of_children_l89_89973


namespace largest_possible_s_l89_89807

theorem largest_possible_s 
  (r s : ℕ) 
  (hr : r ≥ s) 
  (hs : s ≥ 3) 
  (hangles : (r - 2) * 60 * s = (s - 2) * 61 * r) : 
  s = 121 := 
sorry

end largest_possible_s_l89_89807


namespace tiffany_daily_miles_l89_89750

-- Definitions for running schedule
def billy_sunday_miles := 1
def billy_monday_miles := 1
def billy_tuesday_miles := 1
def billy_wednesday_miles := 1
def billy_thursday_miles := 1
def billy_friday_miles := 1
def billy_saturday_miles := 1

def tiffany_wednesday_miles := 1 / 3
def tiffany_thursday_miles := 1 / 3
def tiffany_friday_miles := 1 / 3

-- Total miles is the sum of miles for the week
def billy_total_miles := billy_sunday_miles + billy_monday_miles + billy_tuesday_miles +
                         billy_wednesday_miles + billy_thursday_miles + billy_friday_miles +
                         billy_saturday_miles

def tiffany_total_miles (T : ℝ) := T * 3 + 
                                   tiffany_wednesday_miles + tiffany_thursday_miles + tiffany_friday_miles

-- Proof problem: show that Tiffany runs 2 miles each day on Sunday, Monday, and Tuesday
theorem tiffany_daily_miles : ∃ T : ℝ, (tiffany_total_miles T = billy_total_miles) ∧ T = 2 :=
by
  sorry

end tiffany_daily_miles_l89_89750


namespace average_weight_is_15_l89_89289

-- Define the ages of the 10 children
def ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

-- Define the regression function
def weight (age : ℕ) : ℕ := 2 * age + 7

-- Function to calculate average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Define the weights of the children based on the regression function
def weights : List ℕ := ages.map weight

-- State the theorem to find the average weight of the children
theorem average_weight_is_15 : average weights = 15 := by
  sorry

end average_weight_is_15_l89_89289


namespace natasha_average_speed_climbing_l89_89401

theorem natasha_average_speed_climbing :
  ∀ D : ℝ,
    (total_time = 3 + 2) →
    (total_distance = 2 * D) →
    (average_speed = total_distance / total_time) →
    (average_speed = 3) →
    (D = 7.5) →
    (climb_speed = D / 3) →
    (climb_speed = 2.5) :=
by
  intros D total_time_eq total_distance_eq average_speed_eq average_speed_is_3 D_is_7_5 climb_speed_eq
  sorry

end natasha_average_speed_climbing_l89_89401


namespace rectangle_base_length_l89_89237

theorem rectangle_base_length
  (h : ℝ) (b : ℝ)
  (common_height_nonzero : h ≠ 0)
  (triangle_base : ℝ := 24)
  (same_area : (1/2) * triangle_base * h = b * h) :
  b = 12 :=
by
  sorry

end rectangle_base_length_l89_89237


namespace find_ordered_pair_l89_89683

variables {A B Q : Type} -- Points A, B, Q
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q]
variables {a b q : A} -- Vectors at points A, B, Q
variables (r : ℝ) -- Ratio constant

-- Define the conditions from the original problem
def ratio_aq_qb (A B Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q] (a b q : A) (r : ℝ) :=
  r = 7 / 2

-- Define the goal theorem using the conditions above
theorem find_ordered_pair (h : ratio_aq_qb A B Q a b q r) : 
  q = (7 / 9) • a + (2 / 9) • b :=
sorry

end find_ordered_pair_l89_89683


namespace min_value_x_add_2y_l89_89029

theorem min_value_x_add_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end min_value_x_add_2y_l89_89029


namespace major_premise_wrong_l89_89490

-- Definitions of the given conditions in Lean
def is_parallel_to_plane (line : Type) (plane : Type) : Prop := sorry -- Provide an appropriate definition
def contains_line (plane : Type) (line : Type) : Prop := sorry -- Provide an appropriate definition
def is_parallel_to_line (line1 : Type) (line2 : Type) : Prop := sorry -- Provide an appropriate definition

-- Given conditions
variables (b α a : Type)
variable (H1 : ¬ contains_line α b)  -- Line b is not contained in plane α
variable (H2 : contains_line α a)    -- Line a is contained in plane α
variable (H3 : is_parallel_to_plane b α) -- Line b is parallel to plane α

-- Proposition to prove: The major premise is wrong
theorem major_premise_wrong : ¬(∀ (a b : Type), is_parallel_to_plane b α → contains_line α a → is_parallel_to_line b a) :=
by
  sorry

end major_premise_wrong_l89_89490


namespace minimum_positive_temperatures_announced_l89_89706

theorem minimum_positive_temperatures_announced (x y : ℕ) :
  x * (x - 1) = 110 →
  y * (y - 1) + (x - y) * (x - y - 1) = 54 →
  (∀ z : ℕ, z * (z - 1) + (x - z) * (x - z - 1) = 54 → y ≤ z) →
  y = 4 :=
by
  sorry

end minimum_positive_temperatures_announced_l89_89706


namespace speeds_of_cars_l89_89417

theorem speeds_of_cars (d_A d_B : ℝ) (v_A v_B : ℝ) (h1 : d_A = 300) (h2 : d_B = 250) (h3 : v_A = v_B + 5) (h4 : d_A / v_A = d_B / v_B) :
  v_B = 25 ∧ v_A = 30 :=
by
  sorry

end speeds_of_cars_l89_89417


namespace remainder_equality_l89_89738

theorem remainder_equality
  (P P' K D R R' r r' : ℕ)
  (h1 : P > P')
  (h2 : P % K = 0)
  (h3 : P' % K = 0)
  (h4 : P % D = R)
  (h5 : P' % D = R')
  (h6 : (P * K - P') % D = r)
  (h7 : (R * K - R') % D = r') :
  r = r' :=
sorry

end remainder_equality_l89_89738


namespace jacks_speed_is_7_l89_89495

-- Define the constants and speeds as given in conditions
def initial_distance : ℝ := 150
def christina_speed : ℝ := 8
def lindy_speed : ℝ := 10
def lindy_total_distance : ℝ := 100

-- Hypothesis stating when the three meet
theorem jacks_speed_is_7 :
  ∃ (jack_speed : ℝ), (∃ (time : ℝ), 
    time = lindy_total_distance / lindy_speed
    ∧ christina_speed * time + jack_speed * time = initial_distance) 
  → jack_speed = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end jacks_speed_is_7_l89_89495


namespace quadratic_complete_square_l89_89927

theorem quadratic_complete_square : ∀ x : ℝ, (x^2 - 8*x - 1) = (x - 4)^2 - 17 :=
by sorry

end quadratic_complete_square_l89_89927


namespace calculate_expression_l89_89587

theorem calculate_expression : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 :=
sorry

end calculate_expression_l89_89587


namespace petals_vs_wings_and_unvisited_leaves_l89_89063

def flowers_petals_leaves := 5
def petals_per_flower := 2
def bees_wings := 3
def wings_per_bee := 4
def leaves_per_flower := 3
def visits_per_bee := 2
def total_flowers := flowers_petals_leaves
def total_bees := bees_wings

def total_petals : ℕ := total_flowers * petals_per_flower
def total_wings : ℕ := total_bees * wings_per_bee
def more_wings_than_petals := total_wings - total_petals

def total_leaves : ℕ := total_flowers * leaves_per_flower
def total_visits : ℕ := total_bees * visits_per_bee
def leaves_per_visit := leaves_per_flower
def visited_leaves : ℕ := min total_leaves (total_visits * leaves_per_visit)
def unvisited_leaves : ℕ := total_leaves - visited_leaves

theorem petals_vs_wings_and_unvisited_leaves :
  more_wings_than_petals = 2 ∧ unvisited_leaves = 0 :=
by
  sorry

end petals_vs_wings_and_unvisited_leaves_l89_89063


namespace quadratic_factorization_l89_89708

theorem quadratic_factorization (a b : ℕ) (h1 : x^2 - 18 * x + 72 = (x - a) * (x - b))
  (h2 : a > b) : 2 * b - a = 0 :=
sorry

end quadratic_factorization_l89_89708


namespace value_of_a_plus_b_minus_c_l89_89919

def a : ℤ := 1 -- smallest positive integer
def b : ℤ := 0 -- number with the smallest absolute value
def c : ℤ := -1 -- largest negative integer

theorem value_of_a_plus_b_minus_c : a + b - c = 2 := by
  -- skipping the proof
  sorry

end value_of_a_plus_b_minus_c_l89_89919


namespace factor_expression_l89_89372

theorem factor_expression (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := 
by
  sorry

end factor_expression_l89_89372


namespace area_of_field_l89_89946

-- Define the variables and conditions
variables {L W : ℝ}

-- Given conditions
def length_side (L : ℝ) : Prop := L = 30
def fencing_equation (L W : ℝ) : Prop := L + 2 * W = 70

-- Prove the area of the field is 600 square feet
theorem area_of_field : length_side L → fencing_equation L W → (L * W = 600) :=
by
  intros hL hF
  rw [length_side, fencing_equation] at *
  sorry

end area_of_field_l89_89946


namespace part_a_l89_89921

-- Power tower with 100 twos
def power_tower_100_t2 : ℕ := sorry

theorem part_a : power_tower_100_t2 > 3 := sorry

end part_a_l89_89921


namespace bus_driver_regular_rate_l89_89086

theorem bus_driver_regular_rate (R : ℝ) (h1 : 976 = (40 * R) + (14.32 * (1.75 * R))) : 
  R = 15 := 
by
  sorry

end bus_driver_regular_rate_l89_89086


namespace driver_net_pay_rate_l89_89727

theorem driver_net_pay_rate
    (hours : ℕ) (distance_per_hour : ℕ) (distance_per_gallon : ℕ) 
    (pay_per_mile : ℝ) (gas_cost_per_gallon : ℝ) :
    hours = 3 →
    distance_per_hour = 50 →
    distance_per_gallon = 25 →
    pay_per_mile = 0.75 →
    gas_cost_per_gallon = 2.50 →
    (pay_per_mile * (distance_per_hour * hours) - gas_cost_per_gallon * ((distance_per_hour * hours) / distance_per_gallon)) / hours = 32.5 :=
by
  intros h_hours h_dph h_dpg h_ppm h_gcpg
  sorry

end driver_net_pay_rate_l89_89727


namespace cubic_inequality_l89_89667

theorem cubic_inequality (a b : ℝ) : a > b → a^3 > b^3 :=
sorry

end cubic_inequality_l89_89667


namespace sum_of_fractions_as_decimal_l89_89471

theorem sum_of_fractions_as_decimal : (3 / 8 : ℝ) + (5 / 32) = 0.53125 := by
  sorry

end sum_of_fractions_as_decimal_l89_89471


namespace comparison_theorem_l89_89798

open Real

noncomputable def comparison (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : Prop :=
  let a := log (sin x)
  let b := sin x
  let c := exp (sin x)
  a < b ∧ b < c

theorem comparison_theorem (x : ℝ) (h : 0 < x ∧ x < π / 2) : comparison x h.1 h.2 :=
by { sorry }

end comparison_theorem_l89_89798


namespace watermelons_remaining_l89_89116

theorem watermelons_remaining :
  let initial_watermelons := 10 * 12
  let yesterdays_sale := 0.40 * initial_watermelons
  let remaining_after_yesterday := initial_watermelons - yesterdays_sale
  let todays_sale := (1 / 4) * remaining_after_yesterday
  let remaining_after_today := remaining_after_yesterday - todays_sale
  let tomorrows_sales := 1.5 * todays_sale
  let remaining_after_tomorrow := remaining_after_today - tomorrows_sales
  remaining_after_tomorrow = 27 :=
by
  sorry

end watermelons_remaining_l89_89116


namespace smaller_denom_is_five_l89_89702

-- Define the conditions
def num_smaller_bills : ℕ := 4
def num_ten_dollar_bills : ℕ := 8
def total_bills : ℕ := num_smaller_bills + num_ten_dollar_bills
def ten_dollar_bill_value : ℕ := 10
def total_value : ℕ := 100

-- Define the smaller denomination value
def value_smaller_denom (x : ℕ) : Prop :=
  num_smaller_bills * x + num_ten_dollar_bills * ten_dollar_bill_value = total_value

-- Prove that the value of the smaller denomination bill is 5
theorem smaller_denom_is_five : value_smaller_denom 5 :=
by
  sorry

end smaller_denom_is_five_l89_89702


namespace no_valid_arithmetic_operation_l89_89843

-- Definition for arithmetic operations
inductive Operation
| div : Operation
| mul : Operation
| add : Operation
| sub : Operation

open Operation

-- Given conditions
def equation (op : Operation) : Prop :=
  match op with
  | div => (8 / 2) + 5 - (3 - 2) = 12
  | mul => (8 * 2) + 5 - (3 - 2) = 12
  | add => (8 + 2) + 5 - (3 - 2) = 12
  | sub => (8 - 2) + 5 - (3 - 2) = 12

-- Statement to prove
theorem no_valid_arithmetic_operation : ∀ op : Operation, ¬ equation op := by
  sorry

end no_valid_arithmetic_operation_l89_89843


namespace xy_square_sum_l89_89068

theorem xy_square_sum (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 132) : x^2 + y^2 = 1336 :=
by
  sorry

end xy_square_sum_l89_89068


namespace velocity_at_3_seconds_l89_89103

variable (t : ℝ)
variable (s : ℝ)

def motion_eq (t : ℝ) : ℝ := 1 + t + t^2

theorem velocity_at_3_seconds : 
  (deriv motion_eq 3) = 7 :=
by
  sorry

end velocity_at_3_seconds_l89_89103


namespace arithmetic_sequence_fifth_term_l89_89498

noncomputable def fifth_term_of_arithmetic_sequence (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : ℝ :=
(2 * x / y) - 2 * y

theorem arithmetic_sequence_fifth_term (x y : ℝ)
  (h1 : 2 * x + y = 2 * x + y)
  (h2 : 2 * x - y = 2 * x + y - 2 * y)
  (h3 : 2 * x * y = 2 * x - 2 * y - 2 * y)
  (h4 : 2 * x / y = 2 * x * y - 5 * y^2 - 2 * y)
  : fifth_term_of_arithmetic_sequence x y h1 h2 h3 h4 = -77 / 10 :=
sorry

end arithmetic_sequence_fifth_term_l89_89498


namespace exponent_problem_l89_89273

theorem exponent_problem (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  intro h
  sorry

end exponent_problem_l89_89273


namespace slope_of_line_through_PQ_is_4_l89_89586

theorem slope_of_line_through_PQ_is_4
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_a4 : a 4 = 15)
  (h_a9 : a 9 = 55) :
  let a3 := a 3
  let a8 := a 8
  (a 9 - a 4) / (9 - 4) = 8 → (a 8 - a 3) / (13 - 3) = 4 := by
  sorry

end slope_of_line_through_PQ_is_4_l89_89586


namespace probability_is_one_over_145_l89_89630

-- Define the domain and properties
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even (n : ℕ) : Prop :=
  n % 2 = 0

-- Total number of ways to pick 2 distinct numbers from 1 to 30
def total_ways_to_pick_two_distinct : ℕ :=
  (30 * 29) / 2

-- Calculate prime numbers between 1 and 30
def primes_from_1_to_30 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Filter valid pairs where both numbers are prime and at least one of them is 2
def valid_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  [(2, 3), (2, 5), (2, 7), (2, 11), (2, 13), (2, 17), (2, 19), (2, 23), (2, 29)]

def count_valid_pairs (l : List (ℕ × ℕ)) : ℕ :=
  l.length

-- Probability calculation
def probability_prime_and_even : ℚ :=
  count_valid_pairs (valid_pairs primes_from_1_to_30) / total_ways_to_pick_two_distinct

-- Prove that the probability is 1/145
theorem probability_is_one_over_145 : probability_prime_and_even = 1 / 145 :=
by
  sorry

end probability_is_one_over_145_l89_89630


namespace find_n_l89_89422

theorem find_n (a : ℝ) (x : ℝ) (y : ℝ) (h1 : 0 < a) (h2 : a * x + 0.6 * a * y = 5 / 10)
(h3 : 1.6 * a * x + 1.2 * a * y = 1 - 1 / 10) : 
∃ n : ℕ, n = 10 :=
by
  sorry

end find_n_l89_89422


namespace decreasing_function_l89_89834

-- Define the functions
noncomputable def fA (x : ℝ) : ℝ := 3^x
noncomputable def fB (x : ℝ) : ℝ := Real.logb 0.5 x
noncomputable def fC (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fD (x : ℝ) : ℝ := 1/x

-- Define the domains
def domainA : Set ℝ := Set.univ
def domainB : Set ℝ := {x | x > 0}
def domainC : Set ℝ := {x | x ≥ 0}
def domainD : Set ℝ := {x | x < 0} ∪ {x | x > 0}

-- Prove that fB is the only decreasing function in its domain
theorem decreasing_function:
  (∀ x y, x ∈ domainA → y ∈ domainA → x < y → fA x > fA y) = false ∧
  (∀ x y, x ∈ domainB → y ∈ domainB → x < y → fB x > fB y) ∧
  (∀ x y, x ∈ domainC → y ∈ domainC → x < y → fC x > fC y) = false ∧
  (∀ x y, x ∈ domainD → y ∈ domainD → x < y → fD x > fD y) = false :=
  sorry

end decreasing_function_l89_89834


namespace algebraic_expression_value_l89_89368

variable (x y : ℝ)

def condition1 : Prop := y - x = -1
def condition2 : Prop := x * y = 2

def expression : ℝ := -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3

theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) : expression x y = -4 := 
by
  sorry

end algebraic_expression_value_l89_89368


namespace base6_addition_correct_l89_89191

theorem base6_addition_correct (S H E : ℕ) (h1 : S < 6) (h2 : H < 6) (h3 : E < 6) 
  (distinct : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (h4: S + H * 6 + E * 6^2 +  H * 6 = H + E * 6 + H * 6^2 + E * 6^1) :
  S + H + E = 12 :=
by sorry

end base6_addition_correct_l89_89191


namespace simplify_expression_l89_89198

theorem simplify_expression : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end simplify_expression_l89_89198


namespace number_of_positive_integer_pairs_l89_89056

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l89_89056


namespace expression_increase_l89_89811

variable {x y : ℝ}

theorem expression_increase (hx : x > 0) (hy : y > 0) :
  let original_expr := 3 * x^2 * y
  let new_x := 1.2 * x
  let new_y := 2.4 * y
  let new_expr := 3 * new_x ^ 2 * new_y
  (new_expr / original_expr) = 3.456 :=
by
-- original_expr is 3 * x^2 * y
-- new_x = 1.2 * x
-- new_y = 2.4 * y
-- new_expr = 3 * (1.2 * x)^2 * (2.4 * y)
-- (new_expr / original_expr) = (10.368 * x^2 * y) / (3 * x^2 * y)
-- (new_expr / original_expr) = 10.368 / 3
-- (new_expr / original_expr) = 3.456
sorry

end expression_increase_l89_89811


namespace range_x_satisfies_inequality_l89_89771

theorem range_x_satisfies_inequality (x : ℝ) : (x^2 < |x|) ↔ (-1 < x ∧ x < 1 ∧ x ≠ 0) :=
sorry

end range_x_satisfies_inequality_l89_89771


namespace outer_boundary_diameter_l89_89767

theorem outer_boundary_diameter (d_pond : ℝ) (w_picnic : ℝ) (w_track : ℝ)
  (h_pond_diam : d_pond = 16) (h_picnic_width : w_picnic = 10) (h_track_width : w_track = 4) :
  2 * (d_pond / 2 + w_picnic + w_track) = 44 :=
by
  -- We avoid the entire proof, we only assert the statement in Lean
  sorry

end outer_boundary_diameter_l89_89767


namespace sufficient_but_not_necessary_condition_for_parallelism_l89_89898

-- Define the two lines
def line1 (x y : ℝ) (m : ℝ) : Prop := 2 * x - m * y = 1
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 1) * x - y = 1

-- Define the parallel condition for the two lines
def parallel (m : ℝ) : Prop :=
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 m ∧ line2 x2 y2 m ∧ (2 * m + 1 = 0 ∧ m^2 - m - 2 = 0)) ∨ 
  (∃ x1 y1 x2 y2 : ℝ, line1 x1 y1 2 ∧ line2 x2 y2 2)

theorem sufficient_but_not_necessary_condition_for_parallelism :
  ∀ m, (parallel m) ↔ (m = 2) :=
by sorry

end sufficient_but_not_necessary_condition_for_parallelism_l89_89898


namespace positive_integer_sum_l89_89976

theorem positive_integer_sum (n : ℤ) :
  (n > 0) ∧
  (∀ stamps_cannot_form : ℤ, (∀ a b c : ℤ, 7 * a + n * b + (n + 2) * c ≠ 120) ↔
  ¬ (0 ≤ 7*a ∧ 7*a ≤ 120 ∧ 0 ≤ n*b ∧ n*b ≤ 120 ∧ 0 ≤ (n + 2)*c ∧ (n + 2)*c ≤ 120)) ∧
  (∀ postage_formed : ℤ, (120 < postage_formed ∧ postage_formed ≤ 125 →
  ∃ a b c : ℤ, 7 * a + n * b + (n + 2) * c = postage_formed)) →
  n = 21 :=
by {
  -- proof omittted 
  sorry
}

end positive_integer_sum_l89_89976


namespace magnitude_of_z_l89_89527

theorem magnitude_of_z (z : ℂ) (h : z * (1 + 2 * Complex.I) + Complex.I = 0) : 
  Complex.abs z = Real.sqrt (5) / 5 := 
sorry

end magnitude_of_z_l89_89527


namespace ella_emma_hotdogs_l89_89728

-- Definitions based on the problem conditions
def hotdogs_each_sister_wants (E : ℕ) :=
  let luke := 2 * E
  let hunter := 3 * E
  E + E + luke + hunter = 14

-- Statement we need to prove
theorem ella_emma_hotdogs (E : ℕ) (h : hotdogs_each_sister_wants E) : E = 2 :=
by
  sorry

end ella_emma_hotdogs_l89_89728


namespace ian_lottery_win_l89_89064

theorem ian_lottery_win 
  (amount_paid_to_colin : ℕ)
  (amount_left : ℕ)
  (amount_paid_to_helen : ℕ := 2 * amount_paid_to_colin)
  (amount_paid_to_benedict : ℕ := amount_paid_to_helen / 2)
  (total_debts_paid : ℕ := amount_paid_to_colin + amount_paid_to_helen + amount_paid_to_benedict)
  (total_money_won : ℕ := total_debts_paid + amount_left)
  (h1 : amount_paid_to_colin = 20)
  (h2 : amount_left = 20) :
  total_money_won = 100 := 
sorry

end ian_lottery_win_l89_89064


namespace find_f_neg2_l89_89863

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ a b : ℝ, f (a + b) = f a * f b
axiom cond2 : ∀ x : ℝ, f x > 0
axiom cond3 : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 := sorry

end find_f_neg2_l89_89863


namespace incenter_coordinates_l89_89040

theorem incenter_coordinates (p q r : ℝ) (h₁ : p = 8) (h₂ : q = 6) (h₃ : r = 10) :
  ∃ x y z : ℝ, x + y + z = 1 ∧ x = p / (p + q + r) ∧ y = q / (p + q + r) ∧ z = r / (p + q + r) ∧
  x = 1 / 3 ∧ y = 1 / 4 ∧ z = 5 / 12 :=
by
  sorry

end incenter_coordinates_l89_89040


namespace solve_otimes_eq_l89_89179

def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

theorem solve_otimes_eq : ∃ x : ℝ, otimes (-4) (x + 3) = 6 ↔ x = -5 :=
by
  use -5
  simp [otimes]
  sorry

end solve_otimes_eq_l89_89179


namespace nth_equation_l89_89345

open Nat

theorem nth_equation (n : ℕ) (hn : 0 < n) :
  (n + 1)/((n + 1) * (n + 1) - 1) - (1/(n * (n + 1) * (n + 2))) = 1/(n + 1) := 
by
  sorry

end nth_equation_l89_89345


namespace unique_valid_quintuple_l89_89231

theorem unique_valid_quintuple :
  ∃! (a b c d e : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧
    a^2 + b^2 + c^3 + d^3 + e^3 = 5 ∧
    (a + b + c + d + e) * (a^3 + b^3 + c^2 + d^2 + e^2) = 25 :=
sorry

end unique_valid_quintuple_l89_89231


namespace average_remaining_two_l89_89915

theorem average_remaining_two (a b c d e : ℝ) 
  (h1 : (a + b + c + d + e) / 5 = 12) 
  (h2 : (a + b + c) / 3 = 4) : 
  (d + e) / 2 = 24 :=
by 
  sorry

end average_remaining_two_l89_89915


namespace find_monic_polynomial_l89_89730

-- Define the original polynomial
def polynomial_1 (x : ℝ) := x^3 - 4 * x^2 + 9

-- Define the monic polynomial we are seeking
def polynomial_2 (x : ℝ) := x^3 - 12 * x^2 + 243

theorem find_monic_polynomial :
  ∀ (r1 r2 r3 : ℝ), 
    polynomial_1 r1 = 0 → 
    polynomial_1 r2 = 0 → 
    polynomial_1 r3 = 0 → 
    polynomial_2 (3 * r1) = 0 ∧ polynomial_2 (3 * r2) = 0 ∧ polynomial_2 (3 * r3) = 0 :=
by
  intros r1 r2 r3 h1 h2 h3
  sorry

end find_monic_polynomial_l89_89730


namespace Amith_current_age_l89_89036

variable (A D : ℕ)

theorem Amith_current_age
  (h1 : A - 5 = 3 * (D - 5))
  (h2 : A + 10 = 2 * (D + 10)) :
  A = 50 := by
  sorry

end Amith_current_age_l89_89036


namespace line_equation_passes_through_and_has_normal_l89_89065

theorem line_equation_passes_through_and_has_normal (x y : ℝ) 
    (H1 : ∃ l : ℝ → ℝ, l 3 = 4)
    (H2 : ∃ n : ℝ × ℝ, n = (1, 2)) : 
    x + 2 * y - 11 = 0 :=
sorry

end line_equation_passes_through_and_has_normal_l89_89065


namespace completing_the_square_correct_l89_89629

theorem completing_the_square_correct :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro x h
  sorry

end completing_the_square_correct_l89_89629


namespace population_after_10_years_l89_89736

def initial_population : ℕ := 100000
def birth_increase_percent : ℝ := 0.6
def emigration_per_year : ℕ := 2000
def immigration_per_year : ℕ := 2500
def years : ℕ := 10

theorem population_after_10_years :
  let birth_increase := initial_population * birth_increase_percent
  let total_emigration := emigration_per_year * years
  let total_immigration := immigration_per_year * years
  let net_movement := total_immigration - total_emigration
  let final_population := initial_population + birth_increase + net_movement
  final_population = 165000 :=
by
  sorry

end population_after_10_years_l89_89736


namespace problem_distribution_count_l89_89066

theorem problem_distribution_count : 12^6 = 2985984 := 
by
  sorry

end problem_distribution_count_l89_89066


namespace f_inequality_l89_89561

variables {n1 n2 d : ℕ} (f : ℕ → ℕ → ℕ)

theorem f_inequality (hn1 : n1 > 0) (hn2 : n2 > 0) (hd : d > 0) :
  f (n1 * n2) d ≤ f n1 d + n1 * (f n2 d - 1) :=
sorry

end f_inequality_l89_89561


namespace ming_wins_inequality_l89_89861

variables (x : ℕ)

def remaining_distance (x : ℕ) : ℕ := 10000 - 200 * x
def ming_remaining_distance (x : ℕ) : ℕ := remaining_distance x - 200

-- Ensure that Xiao Ming's winning inequality holds:
theorem ming_wins_inequality (h1 : 0 < x) :
  (ming_remaining_distance x) / 250 > (remaining_distance x) / 300 :=
sorry

end ming_wins_inequality_l89_89861


namespace cone_slant_height_l89_89833

noncomputable def slant_height (r : ℝ) (CSA : ℝ) : ℝ := CSA / (Real.pi * r)

theorem cone_slant_height : slant_height 10 628.3185307179587 = 20 :=
by
  sorry

end cone_slant_height_l89_89833


namespace satisfies_equation_l89_89995

noncomputable def y (x : ℝ) : ℝ := -Real.sqrt (x^4 - x^2)
noncomputable def dy (x : ℝ) : ℝ := x * (1 - 2 * x^2) / Real.sqrt (x^4 - x^2)

theorem satisfies_equation (x : ℝ) (hx : x ≠ 0) : x * y x * dy x - (y x)^2 = x^4 := 
sorry

end satisfies_equation_l89_89995


namespace not_constant_expression_l89_89763

noncomputable def is_centroid (A B C G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

noncomputable def squared_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem not_constant_expression (A B C P G : ℝ × ℝ)
  (hG : is_centroid A B C G)
  (hP_on_AB : ∃ x, P = (x, A.2) ∧ A.2 = B.2) :
  ∃ dPA dPB dPC dPG : ℝ,
    dPA = squared_distance P A ∧
    dPB = squared_distance P B ∧
    dPC = squared_distance P C ∧
    dPG = squared_distance P G ∧
    (dPA + dPB + dPC - dPG) ≠ dPA + dPB + dPC - dPG := by
  sorry

end not_constant_expression_l89_89763


namespace math_problem_l89_89445

theorem math_problem
  (x : ℕ) (y : ℕ)
  (h1 : x = (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id)
  (h2 : y = ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card)
  (h3 : x + y = 611) :
  (Finset.range (60 + 1 + 1) \ Finset.range 50).sum id = 605 ∧
  ((Finset.range (60 + 1) \ Finset.range 50).filter (λ n => n % 2 = 0)).card = 6 := 
by
  sorry

end math_problem_l89_89445


namespace park_shape_l89_89378

def cost_of_fencing (side_count : ℕ) (side_cost : ℕ) := side_count * side_cost

theorem park_shape (total_cost : ℕ) (side_cost : ℕ) (h_total : total_cost = 224) (h_side : side_cost = 56) : 
  (∃ sides : ℕ, sides = total_cost / side_cost ∧ sides = 4) ∧ (∀ (sides : ℕ),  cost_of_fencing sides side_cost = total_cost → sides = 4 → sides = 4 ∧ (∀ (x y z w : ℕ), x = y → y = z → z = w → w = x)) :=
by
  sorry

end park_shape_l89_89378


namespace geometric_progression_terms_l89_89311

theorem geometric_progression_terms 
  (q b4 S_n : ℚ) 
  (hq : q = 1/3) 
  (hb4 : b4 = 1/54) 
  (hS : S_n = 121/162) 
  (b1 : ℚ) 
  (hb1 : b1 = b4 * q^3)
  (Sn : ℚ) 
  (hSn : Sn = b1 * (1 - q^5) / (1 - q)) : 
  ∀ (n : ℕ), S_n = Sn → n = 5 :=
by
  intro n hn
  sorry

end geometric_progression_terms_l89_89311


namespace problem_l89_89526

open Set

theorem problem (M : Set ℤ) (N : Set ℤ) (hM : M = {1, 2, 3, 4}) (hN : N = {-2, 2}) : 
  M ∩ N = {2} :=
by
  sorry

end problem_l89_89526


namespace solve_system_l89_89173

theorem solve_system (x y : ℝ) (h1 : 5 * x + y = 19) (h2 : x + 3 * y = 1) : 3 * x + 2 * y = 10 :=
by
  sorry

end solve_system_l89_89173


namespace trains_total_distance_l89_89243

theorem trains_total_distance (speedA_kmph speedB_kmph time_min : ℕ)
                             (hA : speedA_kmph = 70)
                             (hB : speedB_kmph = 90)
                             (hT : time_min = 15) :
    let speedA_kmpm := (speedA_kmph : ℝ) / 60
    let speedB_kmpm := (speedB_kmph : ℝ) / 60
    let distanceA := speedA_kmpm * (time_min : ℝ)
    let distanceB := speedB_kmpm * (time_min : ℝ)
    distanceA + distanceB = 40 := 
by 
  sorry

end trains_total_distance_l89_89243


namespace geometric_sequence_fourth_term_l89_89047

theorem geometric_sequence_fourth_term (x : ℝ) (h1 : (2 * x + 2) ^ 2 = x * (3 * x + 3))
  (h2 : x ≠ -1) : (3*x + 3) * (3/2) = -27/2 :=
by
  sorry

end geometric_sequence_fourth_term_l89_89047


namespace proof_inequalities_l89_89515

variable {R : Type} [LinearOrder R] [Ring R]

def odd_function (f : R → R) : Prop :=
∀ x : R, f (-x) = -f x

def decreasing_function (f : R → R) : Prop :=
∀ x y : R, x ≤ y → f y ≤ f x

theorem proof_inequalities (f : R → R) (a b : R) 
  (h_odd : odd_function f)
  (h_decr : decreasing_function f)
  (h : a + b ≤ 0) :
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by
  sorry

end proof_inequalities_l89_89515


namespace car_speed_l89_89855

theorem car_speed (distance time : ℝ) (h₁ : distance = 50) (h₂ : time = 5) : (distance / time) = 10 :=
by
  rw [h₁, h₂]
  norm_num

end car_speed_l89_89855


namespace reciprocal_neg_six_l89_89836

-- Define the concept of reciprocal
def reciprocal (a : ℤ) (h : a ≠ 0) : ℚ := 1 / a

theorem reciprocal_neg_six : reciprocal (-6) (by norm_num) = -1 / 6 := 
by 
  sorry

end reciprocal_neg_six_l89_89836


namespace boxes_sold_l89_89497

theorem boxes_sold (start_boxes sold_boxes left_boxes : ℕ) (h1 : start_boxes = 10) (h2 : left_boxes = 5) (h3 : start_boxes - sold_boxes = left_boxes) : sold_boxes = 5 :=
by
  sorry

end boxes_sold_l89_89497


namespace DE_eq_DF_l89_89739

variable {Point : Type}
variable {E A B C D F : Point}
variable (square : Π (A B C D : Point), Prop ) 
variable (is_parallel : Π (A B : Point), Prop) 
variable (E_outside_square : Prop)
variable (BE_eq_BD : Prop)
variable (BE_intersects_AD_at_F : Prop)

theorem DE_eq_DF
  (H1 : square A B C D)
  (H2 : is_parallel AE BD)
  (H3 : BE_eq_BD)
  (H4 : BE_intersects_AD_at_F) :
  DE = DF := 
sorry

end DE_eq_DF_l89_89739


namespace degrees_to_radians_conversion_l89_89669

theorem degrees_to_radians_conversion : (-300 : ℝ) * (Real.pi / 180) = - (5 / 3) * Real.pi :=
by
  sorry

end degrees_to_radians_conversion_l89_89669


namespace grandmother_current_age_l89_89665

theorem grandmother_current_age (yoojung_age_current yoojung_age_future grandmother_age_future : ℕ)
    (h1 : yoojung_age_current = 5)
    (h2 : yoojung_age_future = 10)
    (h3 : grandmother_age_future = 60) :
    grandmother_age_future - (yoojung_age_future - yoojung_age_current) = 55 :=
by 
  sorry

end grandmother_current_age_l89_89665


namespace sarah_boxes_l89_89945

theorem sarah_boxes (b : ℕ) 
  (h1 : ∀ x : ℕ, x = 7) 
  (h2 : 49 = 7 * b) :
  b = 7 :=
sorry

end sarah_boxes_l89_89945


namespace part1_part2_l89_89965

-- Define the conditions
def cost_price := 30
def initial_selling_price := 40
def initial_sales_volume := 600
def sales_decrease_per_yuan := 10

-- Define the profit calculation function
def profit (selling_price : ℕ) : ℕ :=
  let profit_per_unit := selling_price - cost_price
  let new_sales_volume := initial_sales_volume - sales_decrease_per_yuan * (selling_price - initial_selling_price)
  profit_per_unit * new_sales_volume

-- Statements to prove
theorem part1 :
  profit 50 = 10000 :=
by
  sorry

theorem part2 :
  let max_profit_price := 60
  let max_profit := 12000
  max_profit = (fun price => max (profit price) 0) 60 :=
by
  sorry

end part1_part2_l89_89965


namespace maria_ann_age_problem_l89_89300

theorem maria_ann_age_problem
  (M A : ℕ)
  (h1 : M = 7)
  (h2 : M = A - 3) :
  ∃ Y : ℕ, 7 - Y = 1 / 2 * (10 - Y) := by
  sorry

end maria_ann_age_problem_l89_89300


namespace g_six_l89_89649

theorem g_six (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x * g y) (H2 : g 2 = 4) : g 6 = 64 :=
by
  sorry

end g_six_l89_89649


namespace shipping_cost_correct_l89_89253

-- Definitions of given conditions
def total_weight_of_fish : ℕ := 540
def weight_of_each_crate : ℕ := 30
def total_shipping_cost : ℚ := 27

-- Calculating the number of crates
def number_of_crates : ℕ := total_weight_of_fish / weight_of_each_crate

-- Definition of the target shipping cost per crate
def shipping_cost_per_crate : ℚ := total_shipping_cost / number_of_crates

-- Lean statement to prove the given problem
theorem shipping_cost_correct :
  shipping_cost_per_crate = 1.50 := by
  sorry

end shipping_cost_correct_l89_89253


namespace part_a_part_b_l89_89947

theorem part_a (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ∃ (a b c d : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) := sorry

theorem part_b (a b c d e : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) : 
  ∃ (a b c d e : ℕ), 1 / (a : ℝ) + 1 / (b : ℝ) = 1 / (c : ℝ) + 1 / (d : ℝ) + 1 / (e : ℝ) := sorry

end part_a_part_b_l89_89947


namespace problem1_problem3_l89_89281

-- Define the function f(x)
def f (x : ℚ) : ℚ := (1 - x) / (1 + x)

-- Problem 1: Prove f(1/x) = -f(x), given x ≠ -1, x ≠ 0
theorem problem1 (x : ℚ) (hx1 : x ≠ -1) (hx2 : x ≠ 0) : f (1 / x) = -f x :=
by sorry

-- Problem 2: Comment on graph transformations for f(x)
-- This is a conceptual question about graph translation and is not directly translatable to a Lean theorem.

-- Problem 3: Find the minimum value of M - m such that m ≤ f(x) ≤ M for x ∈ ℤ
theorem problem3 : ∃ (M m : ℤ), (∀ x : ℤ, m ≤ f x ∧ f x ≤ M) ∧ (M - m = 4) :=
by sorry

end problem1_problem3_l89_89281


namespace simplify_expression_l89_89420

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : (2 * x ^ 3) ^ 3 = 8 * x ^ 9 := by
  sorry

end simplify_expression_l89_89420


namespace agent_007_encryption_l89_89837

theorem agent_007_encryption : ∃ (m n : ℕ), (0.07 : ℝ) = (1 / m : ℝ) + (1 / n : ℝ) := 
sorry

end agent_007_encryption_l89_89837


namespace equilateral_triangle_side_length_l89_89789

theorem equilateral_triangle_side_length (perimeter : ℕ) (h_perimeter : perimeter = 69) : 
  ∃ (side_length : ℕ), side_length = perimeter / 3 := 
by
  sorry

end equilateral_triangle_side_length_l89_89789


namespace cone_lateral_area_l89_89857

theorem cone_lateral_area (C l r A : ℝ) (hC : C = 4 * Real.pi) (hl : l = 3) 
  (hr : 2 * Real.pi * r = 4 * Real.pi) (hA : A = Real.pi * r * l) : A = 6 * Real.pi :=
by
  sorry

end cone_lateral_area_l89_89857


namespace roots_of_equation_l89_89165

theorem roots_of_equation (x : ℝ) : 3 * x * (x - 1) = 2 * (x - 1) → (x = 1 ∨ x = 2 / 3) :=
by 
  intros h
  sorry

end roots_of_equation_l89_89165


namespace count_odd_numbers_distinct_digits_l89_89297

theorem count_odd_numbers_distinct_digits : 
  ∃ n : ℕ, (∀ x : ℕ, 200 ≤ x ∧ x ≤ 999 ∧ x % 2 = 1 ∧ (∀ d ∈ [digit1, digit2, digit3], d ≤ 7) ∧ (digit1 ≠ digit2 ∧ digit2 ≠ digit3 ∧ digit1 ≠ digit3) → True) ∧
  n = 120 :=
sorry

end count_odd_numbers_distinct_digits_l89_89297


namespace watched_videos_correct_l89_89653

-- Conditions
def num_suggestions_per_time : ℕ := 15
def times : ℕ := 5
def chosen_position : ℕ := 5

-- Question
def total_videos_watched : ℕ := num_suggestions_per_time * times - (num_suggestions_per_time - chosen_position)

-- Proof
theorem watched_videos_correct : total_videos_watched = 65 := by
  sorry

end watched_videos_correct_l89_89653


namespace ann_susan_age_sum_l89_89880

theorem ann_susan_age_sum (ann_age : ℕ) (susan_age : ℕ) (h1 : ann_age = 16) (h2 : ann_age = susan_age + 5) : ann_age + susan_age = 27 :=
by
  sorry

end ann_susan_age_sum_l89_89880


namespace path_count_correct_l89_89105

-- Define the graph-like structure for the octagonal lattice with directional constraints
structure OctagonalLattice :=
  (vertices : Type)
  (edges : vertices → vertices → Prop) -- Directed edges

-- Define a path from A to B respecting the constraints
def path_num_lattice (L : OctagonalLattice) (A B : L.vertices) : ℕ :=
  sorry -- We assume a function counting valid paths exists here

-- Assert the specific conditions for the bug's movement
axiom LatticeStructure : OctagonalLattice
axiom vertex_A : LatticeStructure.vertices
axiom vertex_B : LatticeStructure.vertices

-- Example specific path counting for the problem's lattice
noncomputable def paths_from_A_to_B : ℕ :=
  path_num_lattice LatticeStructure vertex_A vertex_B

theorem path_count_correct : paths_from_A_to_B = 2618 :=
  sorry -- This is where the proof would go

end path_count_correct_l89_89105


namespace opposite_neg_three_over_two_l89_89099

-- Define the concept of the opposite number
def opposite (x : ℚ) : ℚ := -x

-- State the problem: The opposite number of -3/2 is 3/2
theorem opposite_neg_three_over_two :
  opposite (- (3 / 2 : ℚ)) = (3 / 2 : ℚ) := 
  sorry

end opposite_neg_three_over_two_l89_89099


namespace gcd_a_b_is_one_l89_89286

-- Definitions
def a : ℤ := 100^2 + 221^2 + 320^2
def b : ℤ := 101^2 + 220^2 + 321^2

-- Theorem statement
theorem gcd_a_b_is_one : Int.gcd a b = 1 := by
  sorry

end gcd_a_b_is_one_l89_89286


namespace distinct_values_count_l89_89867

noncomputable def f : ℕ → ℤ := sorry -- The actual function definition is not required

theorem distinct_values_count :
  ∃! n, n = 3 ∧ 
  (∀ x : ℕ, 
    (f x = f (x - 1) + f (x + 1) ∧ 
     (x = 1 → f x = 2009) ∧ 
     (x = 3 → f x = 0))) := 
sorry

end distinct_values_count_l89_89867


namespace seashells_calculation_l89_89059

theorem seashells_calculation :
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  leigh_seashells = 16 :=
by
  let mimi_seashells := 24
  let kyle_seashells := 2 * mimi_seashells
  let leigh_seashells := kyle_seashells / 3
  show leigh_seashells = 16
  sorry

end seashells_calculation_l89_89059


namespace points_on_line_l89_89147

-- Define the points involved
def point1 : ℝ × ℝ := (4, 10)
def point2 : ℝ × ℝ := (-2, -8)
def candidate_points : List (ℝ × ℝ) := [(1, 1), (0, -1), (2, 3), (-1, -5), (3, 7)]
def correct_points : List (ℝ × ℝ) := [(1, 1), (-1, -5), (3, 7)]

-- Define a function to check if a point lies on the line defined by point1 and point2
def lies_on_line (p : ℝ × ℝ) : Prop :=
  let m := (10 - (-8)) / (4 - (-2))
  let b := 10 - m * 4
  p.2 = m * p.1 + b

-- Main theorem statement
theorem points_on_line :
  ∀ p ∈ candidate_points, p ∈ correct_points ↔ lies_on_line p :=
sorry

end points_on_line_l89_89147


namespace option_c_incorrect_l89_89732

theorem option_c_incorrect (a : ℝ) : a + a^2 ≠ a^3 :=
sorry

end option_c_incorrect_l89_89732


namespace initial_group_size_l89_89421

theorem initial_group_size
  (n : ℕ) (W : ℝ)
  (h_avg_increase : ∀ W n, ((W + 12) / n) = (W / n + 3))
  (h_new_person_weight : 82 = 70 + 12) : n = 4 :=
by
  sorry

end initial_group_size_l89_89421


namespace g_of_neg_5_is_4_l89_89470

def f (x : ℝ) : ℝ := 3 * x - 8
def g (y : ℝ) : ℝ := 2 * y^2 + 5 * y - 3

theorem g_of_neg_5_is_4 : g (-5) = 4 :=
by
  sorry

end g_of_neg_5_is_4_l89_89470


namespace garden_length_l89_89170

open Nat

def perimeter : ℕ → ℕ → ℕ := λ l w => 2 * (l + w)

theorem garden_length (width : ℕ) (perimeter_val : ℕ) (length : ℕ) 
  (h1 : width = 15) 
  (h2 : perimeter_val = 80) 
  (h3 : perimeter length width = perimeter_val) :
  length = 25 := by
  sorry

end garden_length_l89_89170


namespace dogs_in_pet_shop_l89_89887

variable (D C B : ℕ) (x : ℕ)

theorem dogs_in_pet_shop
  (h1 : D = 3 * x)
  (h2 : C = 7 * x)
  (h3 : B = 12 * x)
  (h4 : D + B = 375) :
  D = 75 :=
by
  sorry

end dogs_in_pet_shop_l89_89887


namespace solve_system_of_inequalities_l89_89322

theorem solve_system_of_inequalities {x : ℝ} :
  (x + 3 ≥ 2) ∧ (2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by
  sorry

end solve_system_of_inequalities_l89_89322


namespace value_of_expression_l89_89484

theorem value_of_expression {a b c : ℝ} (h_eqn : a + b + c = 15)
  (h_ab_bc_ca : ab + bc + ca = 13) (h_abc : abc = 8)
  (h_roots : Polynomial.roots (Polynomial.X^3 - 15 * Polynomial.X^2 + 13 * Polynomial.X - 8) = {a, b, c}) :
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 199/9 :=
by sorry

end value_of_expression_l89_89484


namespace min_value_of_exp_l89_89643

noncomputable def minimum_value_of_expression (a b : ℝ) : ℝ :=
  (1 - a)^2 + (1 - 2 * b)^2 + (a - 2 * b)^2

theorem min_value_of_exp (a b : ℝ) (h : a^2 ≥ 8 * b) : minimum_value_of_expression a b = 9 / 8 :=
by
  sorry

end min_value_of_exp_l89_89643


namespace triangle_obtuse_l89_89508

theorem triangle_obtuse (a b c : ℝ) (A B C : ℝ) 
  (hBpos : 0 < B) 
  (hBpi : B < Real.pi) 
  (sin_C_lt_cos_A_sin_B : Real.sin C / Real.sin B < Real.cos A) 
  (hC_eq : C = A + B) 
  (ha2 : A + B + C = Real.pi) :
  B > Real.pi / 2 := 
sorry

end triangle_obtuse_l89_89508


namespace residue_625_mod_17_l89_89627

theorem residue_625_mod_17 : 625 % 17 = 13 :=
by
  sorry

end residue_625_mod_17_l89_89627


namespace intersection_necessary_but_not_sufficient_l89_89164

variables {M N P : Set α}

theorem intersection_necessary_but_not_sufficient : 
  (M ∩ P = N ∩ P) → (M ≠ N) :=
sorry

end intersection_necessary_but_not_sufficient_l89_89164


namespace value_of_expression_l89_89607

theorem value_of_expression :
  let x := 1
  let y := -1
  let z := 0
  2 * x + 3 * y + 4 * z = -1 :=
by
  sorry

end value_of_expression_l89_89607


namespace john_spent_half_on_fruits_and_vegetables_l89_89827

theorem john_spent_half_on_fruits_and_vegetables (M : ℝ) (F : ℝ) 
  (spent_on_meat : ℝ) (spent_on_bakery : ℝ) (spent_on_candy : ℝ) :
  (M = 120) → 
  (spent_on_meat = (1 / 3) * M) → 
  (spent_on_bakery = (1 / 10) * M) → 
  (spent_on_candy = 8) → 
  (F * M + spent_on_meat + spent_on_bakery + spent_on_candy = M) → 
  (F = 1 / 2) := 
  by 
    sorry

end john_spent_half_on_fruits_and_vegetables_l89_89827


namespace point_not_on_graph_l89_89575

theorem point_not_on_graph : ∀ (x y : ℝ), (x, y) = (-1, 1) → ¬ (∃ z : ℝ, z ≠ -1 ∧ y = z / (z + 1)) :=
by {
  sorry
}

end point_not_on_graph_l89_89575


namespace christine_speed_l89_89096

def distance : ℕ := 20
def time : ℕ := 5

theorem christine_speed :
  (distance / time) = 4 := 
sorry

end christine_speed_l89_89096


namespace find_unknown_number_l89_89371

theorem find_unknown_number (x : ℝ) (h : (28 + 48 / x) * x = 1980) : x = 69 :=
sorry

end find_unknown_number_l89_89371


namespace distinct_real_roots_l89_89568

theorem distinct_real_roots :
  ∀ x : ℝ, (x^3 - 3*x^2 + x - 2) * (x^3 - x^2 - 4*x + 7) + 6*x^2 - 15*x + 18 = 0 ↔
  x = 1 ∨ x = -2 ∨ x = 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 1 + Real.sqrt 2 :=
by sorry

end distinct_real_roots_l89_89568


namespace series_sum_l89_89353

theorem series_sum :
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  -- We define the sequence in list form for clarity
  (s.sum = 29) :=
by
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  sorry

end series_sum_l89_89353


namespace trig_identity_evaluation_l89_89990

theorem trig_identity_evaluation :
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end trig_identity_evaluation_l89_89990


namespace solution_proof_l89_89971

noncomputable def problem_statement : Prop :=
  let a : ℝ := 0.10
  let b : ℝ := 0.50
  let c : ℝ := 500
  a * (b * c) = 25

theorem solution_proof : problem_statement := by
  sorry

end solution_proof_l89_89971


namespace compare_real_numbers_l89_89580

theorem compare_real_numbers (a b : ℝ) : (a > b) ∨ (a = b) ∨ (a < b) :=
sorry

end compare_real_numbers_l89_89580


namespace num_terminating_decimals_l89_89000

-- Define the problem conditions and statement
def is_terminating_decimal (n : ℕ) : Prop :=
  n % 3 = 0

theorem num_terminating_decimals : 
  ∃ (k : ℕ), k = 220 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 660 → is_terminating_decimal n ↔ n % 3 = 0) := 
by
  sorry

end num_terminating_decimals_l89_89000


namespace find_m_range_l89_89256

noncomputable def quadratic_inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem find_m_range :
  { m : ℝ | quadratic_inequality_condition m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
sorry

end find_m_range_l89_89256


namespace rectangle_coloring_problem_l89_89784

theorem rectangle_coloring_problem :
  let n := 3
  let m := 4
  ∃ n, ∃ m, n = 3 ∧ m = 4 := sorry

end rectangle_coloring_problem_l89_89784


namespace minimal_abs_diff_l89_89723

theorem minimal_abs_diff (a b : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b - 8 * a + 7 * b = 569) : abs (a - b) = 23 :=
sorry

end minimal_abs_diff_l89_89723


namespace inverse_g_neg1_l89_89864

noncomputable def g (c d x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_neg1 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, g c d y = -1 ∧ y = (-1 - d) / c := 
by
  unfold g
  sorry

end inverse_g_neg1_l89_89864


namespace correct_option_is_B_l89_89369

-- Define the total number of balls
def total_black_balls : ℕ := 3
def total_red_balls : ℕ := 7
def total_balls : ℕ := total_black_balls + total_red_balls

-- Define the event of drawing balls
def drawing_balls (n : ℕ) : Prop := n = 3

-- Define what a random variable is within this context
def is_random_variable (n : ℕ) : Prop :=
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3

-- The main statement to prove
theorem correct_option_is_B (n : ℕ) :
  drawing_balls n → is_random_variable n :=
by
  intro h
  sorry

end correct_option_is_B_l89_89369


namespace part_a_part_b_l89_89012

theorem part_a (A B : ℕ) (hA : 1 ≤ A) (hB : 1 ≤ B) : 
  (A + B = 70) → 
  (A * (4 : ℚ) / 35 + B * (4 : ℚ) / 35 = 8) :=
  by
    sorry

theorem part_b (C D : ℕ) (r : ℚ) (hC : C > 1) (hD : D > 1) (hr : r > 1) :
  (C + D = 8 / r) → 
  (C * r + D * r = 8) → 
  (∃ ki : ℕ, (C + D = (70 : ℕ) / ki ∧ 1 < ki ∧ ki ∣ 70)) :=
  by
    sorry

end part_a_part_b_l89_89012


namespace Vihaan_more_nephews_than_Alden_l89_89136

theorem Vihaan_more_nephews_than_Alden :
  ∃ (a v : ℕ), (a = 100) ∧ (a + v = 260) ∧ (v - a = 60) := by
  sorry

end Vihaan_more_nephews_than_Alden_l89_89136


namespace bernold_wins_game_l89_89282

/-- A game is played on a 2007 x 2007 grid. Arnold's move consists of taking a 2 x 2 square,
 and Bernold's move consists of taking a 1 x 1 square. They alternate turns with Arnold starting.
  When Arnold can no longer move, Bernold takes all remaining squares. The goal is to prove that 
  Bernold can always win the game by ensuring that Arnold cannot make enough moves to win. --/
theorem bernold_wins_game (N : ℕ) (hN : N = 2007) :
  let admissible_points := (N - 1) * (N - 1)
  let arnold_moves_needed := (N / 2) * (N / 2 + 1) / 2 + 1
  admissible_points < arnold_moves_needed :=
by
  let admissible_points := 2006 * 2006
  let arnold_moves_needed := 1003 * 1004 / 2 + 1
  exact sorry

end bernold_wins_game_l89_89282


namespace maximum_distance_l89_89532

-- Defining the conditions
def highway_mileage : ℝ := 12.2
def city_mileage : ℝ := 7.6
def gasoline_amount : ℝ := 22

-- Mathematical equivalent proof statement
theorem maximum_distance (h_mileage : ℝ) (g_amount : ℝ) : h_mileage = 12.2 ∧ g_amount = 22 → g_amount * h_mileage = 268.4 :=
by
  intro h
  sorry

end maximum_distance_l89_89532


namespace find_x_value_l89_89901

-- Definitions based on the conditions
def varies_inversely_as_square (k : ℝ) (x y : ℝ) : Prop := x = k / y^2

def given_condition (k : ℝ) : Prop := 1 = k / 3^2

-- The main proof problem to solve
theorem find_x_value (k : ℝ) (y : ℝ) (h1 : varies_inversely_as_square k 1 3) (h2 : y = 9) : 
  varies_inversely_as_square k (1/9) y :=
sorry

end find_x_value_l89_89901


namespace find_number_l89_89944

theorem find_number (n : ℕ) (h : n + 19 = 47) : n = 28 :=
by {
    sorry
}

end find_number_l89_89944


namespace interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l89_89088

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem interval_of_increase (a : ℝ) : 
  (∀ x : ℝ, 0 < a → (Real.exp x - a ≥ 0 ↔ x ≥ Real.log a)) ∧ 
  (∀ x : ℝ, a ≤ 0 → (Real.exp x - a ≥ 0)) :=
by sorry

theorem monotone_increasing (a : ℝ) (h : ∀ x : ℝ, Real.exp x - a ≥ 0) : 
  a ≤ 0 :=
by sorry

theorem monotonically_increasing_decreasing : 
  ∃ a : ℝ, (∀ x ≤ 0, Real.exp x - a ≤ 0) ∧ 
           (∀ x ≥ 0, Real.exp x - a ≥ 0) ↔ a = 1 :=
by sorry

end interval_of_increase_monotone_increasing_monotonically_increasing_decreasing_l89_89088


namespace initial_ratio_l89_89980

variables {p q : ℝ}

theorem initial_ratio (h₁ : p + q = 20) (h₂ : p / (q + 1) = 4 / 3) : p / q = 3 / 2 :=
sorry

end initial_ratio_l89_89980


namespace show_revenue_and_vacancies_l89_89247

theorem show_revenue_and_vacancies:
  let total_seats := 600
  let vip_seats := 50
  let general_seats := 400
  let balcony_seats := 150
  let vip_price := 40
  let general_price := 25
  let balcony_price := 15
  let vip_filled_rate := 0.80
  let general_filled_rate := 0.70
  let balcony_filled_rate := 0.50
  let vip_filled := vip_filled_rate * vip_seats
  let general_filled := general_filled_rate * general_seats
  let balcony_filled := balcony_filled_rate * balcony_seats
  let vip_revenue := vip_filled * vip_price
  let general_revenue := general_filled * general_price
  let balcony_revenue := balcony_filled * balcony_price
  let overall_revenue := vip_revenue + general_revenue + balcony_revenue
  let vip_vacant := vip_seats - vip_filled
  let general_vacant := general_seats - general_filled
  let balcony_vacant := balcony_seats - balcony_filled
  vip_revenue = 1600 ∧
  general_revenue = 7000 ∧
  balcony_revenue = 1125 ∧
  overall_revenue = 9725 ∧
  vip_vacant = 10 ∧
  general_vacant = 120 ∧
  balcony_vacant = 75 :=
by
  sorry

end show_revenue_and_vacancies_l89_89247


namespace sum_binomials_l89_89091

-- Defining binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem sum_binomials : binom 12 4 + binom 10 3 = 615 :=
by
  -- Here we state the problem, and the proof will be left as 'sorry'.
  sorry

end sum_binomials_l89_89091


namespace sequence_a8_l89_89825

theorem sequence_a8 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a n + a (n + 1)) 
  (h2 : a 7 = 120) : 
  a 8 = 194 :=
sorry

end sequence_a8_l89_89825


namespace percentage_of_additional_money_is_10_l89_89150

-- Define the conditions
def months := 11
def payment_per_month := 15
def total_borrowed := 150

-- Define the function to calculate the total amount paid
def total_paid (months payment_per_month : ℕ) : ℕ :=
  months * payment_per_month

-- Define the function to calculate the additional amount paid
def additional_paid (total_paid total_borrowed : ℕ) : ℕ :=
  total_paid - total_borrowed

-- Define the function to calculate the percentage of the additional amount
def percentage_additional (additional_paid total_borrowed : ℕ) : ℕ :=
  (additional_paid * 100) / total_borrowed

-- State the theorem to prove the percentage of the additional money is 10%
theorem percentage_of_additional_money_is_10 :
  percentage_additional (additional_paid (total_paid months payment_per_month) total_borrowed) total_borrowed = 10 :=
by
  sorry

end percentage_of_additional_money_is_10_l89_89150


namespace reciprocal_sqrt5_minus_2_l89_89581

theorem reciprocal_sqrt5_minus_2 : 1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := 
by
  sorry

end reciprocal_sqrt5_minus_2_l89_89581


namespace man_older_than_son_by_46_l89_89985

-- Given conditions about the ages
def sonAge : ℕ := 44

def manAge_in_two_years (M : ℕ) : Prop := M + 2 = 2 * (sonAge + 2)

-- The problem to verify
theorem man_older_than_son_by_46 (M : ℕ) (h : manAge_in_two_years M) : M - sonAge = 46 :=
by
  sorry

end man_older_than_son_by_46_l89_89985


namespace corresponding_side_of_larger_triangle_l89_89986

-- Conditions
variables (A1 A2 : ℕ) (s1 s2 : ℕ)
-- A1 is the area of the larger triangle
-- A2 is the area of the smaller triangle
-- s1 is a side of the smaller triangle = 4 feet
-- s2 is the corresponding side of the larger triangle

-- Given conditions as hypotheses
axiom diff_in_areas : A1 - A2 = 32
axiom ratio_of_areas : A1 = 9 * A2
axiom side_of_smaller_triangle : s1 = 4

-- Theorem to prove the corresponding side of the larger triangle
theorem corresponding_side_of_larger_triangle 
  (h1 : A1 - A2 = 32)
  (h2 : A1 = 9 * A2)
  (h3 : s1 = 4) : 
  s2 = 12 :=
sorry

end corresponding_side_of_larger_triangle_l89_89986


namespace cricket_innings_l89_89824

theorem cricket_innings (n : ℕ) 
  (avg_run_inn : n * 36 = n * 36)  -- average runs is 36 (initially true for any n)
  (increase_avg_by_4 : (36 * n + 120) / (n + 1) = 40) : 
  n = 20 := 
sorry

end cricket_innings_l89_89824


namespace part_a_part_b_l89_89651

-- Part (a): Prove that \( 2^n - 1 \) is divisible by 7 if and only if \( 3 \mid n \).
theorem part_a (n : ℕ) : 7 ∣ (2^n - 1) ↔ 3 ∣ n := sorry

-- Part (b): Prove that \( 2^n + 1 \) is not divisible by 7 for all natural numbers \( n \).
theorem part_b (n : ℕ) : ¬ (7 ∣ (2^n + 1)) := sorry

end part_a_part_b_l89_89651


namespace hyperbola_asymptote_eccentricity_l89_89714

-- Problem statement: We need to prove that the eccentricity of hyperbola 
-- given the specific asymptote is sqrt(5).

noncomputable def calc_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptote_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_asymptote : b = 2 * a) :
  calc_eccentricity a b = Real.sqrt 5 := 
by
  -- Insert the proof step here
  sorry

end hyperbola_asymptote_eccentricity_l89_89714


namespace one_third_of_1206_is_201_percent_of_200_l89_89035

theorem one_third_of_1206_is_201_percent_of_200 : 
  (1 / 3) * 1206 = 402 ∧ 402 / 200 = 201 / 100 :=
by
  sorry

end one_third_of_1206_is_201_percent_of_200_l89_89035


namespace greatest_drop_june_increase_april_l89_89217

-- January price change
def jan : ℝ := -1.00

-- February price change
def feb : ℝ := 3.50

-- March price change
def mar : ℝ := -3.00

-- April price change
def apr : ℝ := 4.50

-- May price change
def may : ℝ := -1.50

-- June price change
def jun : ℝ := -3.50

def greatest_drop : List (ℝ × String) := [(jan, "January"), (mar, "March"), (may, "May"), (jun, "June")]

def greatest_increase : List (ℝ × String) := [(feb, "February"), (apr, "April")]

theorem greatest_drop_june_increase_april :
  (∀ d ∈ greatest_drop, d.1 ≤ jun) ∧ (∀ i ∈ greatest_increase, i.1 ≤ apr) :=
by
  sorry

end greatest_drop_june_increase_april_l89_89217


namespace sum_of_coefficients_equals_28_l89_89274

def P (x : ℝ) : ℝ :=
  2 * (4 * x^8 - 5 * x^5 + 9 * x^3 - 6) + 8 * (x^6 - 4 * x^3 + 6)

theorem sum_of_coefficients_equals_28 : P 1 = 28 := by
  sorry

end sum_of_coefficients_equals_28_l89_89274


namespace number_of_true_propositions_l89_89342

-- Definitions based on conditions
def prop1 (x : ℝ) : Prop := x^2 - x + 1 > 0
def prop2 (x : ℝ) : Prop := x^2 + x - 6 < 0 → x ≤ 2
def prop3 (x : ℝ) : Prop := (x^2 - 5*x + 6 = 0) → x = 2

-- Main theorem
theorem number_of_true_propositions : 
  (∀ x : ℝ, prop1 x) ∧ (∀ x : ℝ, prop2 x) ∧ (∃ x : ℝ, ¬ prop3 x) → 
  2 = 2 :=
by sorry

end number_of_true_propositions_l89_89342


namespace arithmetic_sequence_21st_term_l89_89429

theorem arithmetic_sequence_21st_term (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 13) (h3 : a 3 = 23) :
  a 21 = 203 :=
by
  sorry

end arithmetic_sequence_21st_term_l89_89429


namespace brick_width_is_correct_l89_89275

-- Defining conditions
def wall_length : ℝ := 200 -- wall length in cm
def wall_width : ℝ := 300 -- wall width in cm
def wall_height : ℝ := 2   -- wall height in cm
def brick_length : ℝ := 25 -- brick length in cm
def brick_height : ℝ := 6  -- brick height in cm
def num_bricks : ℝ := 72.72727272727273

-- Total volume of wall
def vol_wall : ℝ := wall_length * wall_width * wall_height

-- Volume of one brick
def vol_brick (width : ℝ) : ℝ := brick_length * width * brick_height

-- Proof statement
theorem brick_width_is_correct : ∃ width : ℝ, vol_wall = vol_brick width * num_bricks ∧ width = 11 :=
by
  sorry

end brick_width_is_correct_l89_89275


namespace exist_m_eq_l89_89609

theorem exist_m_eq (n b : ℕ) (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_zero : n ≠ 0) (hb_zero : b ≠ 0)
  (h_div : p ∣ (b^(2^n) + 1)) :
  ∃ m : ℕ, p = 2^(n+1) * m + 1 :=
by
  sorry

end exist_m_eq_l89_89609


namespace tens_digit_19_2021_l89_89249

theorem tens_digit_19_2021 : (19^2021 % 100) / 10 % 10 = 1 :=
by sorry

end tens_digit_19_2021_l89_89249


namespace num_ordered_pairs_l89_89266

theorem num_ordered_pairs : ∃ (n : ℕ), n = 24 ∧ ∀ (a b : ℂ), a^4 * b^6 = 1 ∧ a^8 * b^3 = 1 → n = 24 :=
by
  sorry

end num_ordered_pairs_l89_89266


namespace magic_square_sum_l89_89793

theorem magic_square_sum (a b c d e f S : ℕ) 
  (h1 : 30 + b + 22 = S) 
  (h2 : 19 + c + d = S) 
  (h3 : a + 28 + f = S)
  (h4 : 30 + 19 + a = S)
  (h5 : b + c + 28 = S)
  (h6 : 22 + d + f = S)
  (h7 : 30 + c + f = S)
  (h8 : 22 + c + a = S)
  (h9 : e = b) :
  d + e = 54 := 
by 
  sorry

end magic_square_sum_l89_89793


namespace larry_channels_l89_89878

-- Initial conditions
def init_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def channels_reduce_request : ℕ := 10
def sports_package : ℕ := 8
def supreme_sports_package : ℕ := 7

-- Calculation representing the overall change step-by-step
theorem larry_channels : 
  init_channels - channels_taken_away + channels_replaced - channels_reduce_request + sports_package + supreme_sports_package = 147 :=
by sorry

end larry_channels_l89_89878


namespace fractional_equation_positive_root_l89_89465

theorem fractional_equation_positive_root (a : ℝ) (ha : ∃ x : ℝ, x > 0 ∧ (6 / (x - 2) - 1 = a * x / (2 - x))) : a = -3 :=
by
  sorry

end fractional_equation_positive_root_l89_89465


namespace car_gas_cost_l89_89892

def car_mpg_city : ℝ := 30
def car_mpg_highway : ℝ := 40
def city_distance_one_way : ℝ := 60
def highway_distance_one_way : ℝ := 200
def gas_cost_per_gallon : ℝ := 3
def total_gas_cost : ℝ := 42

theorem car_gas_cost :
  (city_distance_one_way / car_mpg_city * 2 + highway_distance_one_way / car_mpg_highway * 2) * gas_cost_per_gallon = total_gas_cost := 
  sorry

end car_gas_cost_l89_89892


namespace arithmetic_sequence_c_d_sum_l89_89025

theorem arithmetic_sequence_c_d_sum :
  let c := 19 + (11 - 3)
  let d := c + (11 - 3)
  c + d = 62 :=
by
  sorry

end arithmetic_sequence_c_d_sum_l89_89025


namespace geometric_sequence_third_term_l89_89108

theorem geometric_sequence_third_term :
  ∃ (a : ℕ) (r : ℝ), a = 5 ∧ a * r^3 = 500 ∧ a * r^2 = 5 * 100^(2/3) :=
by
  sorry

end geometric_sequence_third_term_l89_89108


namespace problem_statement_l89_89453

namespace CoinFlipping

/-- 
Define the probability that Alice and Bob both get the same number of heads
when flipping three coins where two are fair and one is biased with a probability
of 3/5 for heads. We aim to calculate p + q where p/q is this probability and 
output the final result - p + q should equal 263.
-/
def same_heads_probability_sum : ℕ :=
  let p := 63
  let q := 200
  p + q

theorem problem_statement : same_heads_probability_sum = 263 :=
  by
  -- proof to be filled in
  sorry

end CoinFlipping

end problem_statement_l89_89453


namespace correct_comparison_l89_89415

theorem correct_comparison :
  ( 
    (-1 > -0.1) = false ∧ 
    (-4 / 3 < -5 / 4) = true ∧ 
    (-1 / 2 > -(-1 / 3)) = false ∧ 
    (Real.pi = 3.14) = false 
  ) :=
by
  sorry

end correct_comparison_l89_89415


namespace compare_rental_fees_l89_89884

namespace HanfuRental

def store_A_rent_price : ℝ := 120
def store_B_rent_price : ℝ := 160
def store_A_discount : ℝ := 0.20
def store_B_discount_limit : ℕ := 6
def store_B_excess_rate : ℝ := 0.50
def x : ℕ := 40 -- number of Hanfu costumes

def y₁ (x : ℕ) : ℝ := (store_A_rent_price * (1 - store_A_discount)) * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ store_B_discount_limit then store_B_rent_price * x
  else store_B_rent_price * store_B_discount_limit + store_B_excess_rate * store_B_rent_price * (x - store_B_discount_limit)

theorem compare_rental_fees (x : ℕ) (hx : x = 40) :
  y₂ x ≤ y₁ x :=
sorry

end HanfuRental

end compare_rental_fees_l89_89884


namespace product_modulo_6_l89_89292

theorem product_modulo_6 :
  (2017 * 2018 * 2019 * 2020) % 6 = 0 :=
by
  -- Conditions provided:
  have h1 : 2017 ≡ 5 [MOD 6] := by sorry
  have h2 : 2018 ≡ 0 [MOD 6] := by sorry
  have h3 : 2019 ≡ 1 [MOD 6] := by sorry
  have h4 : 2020 ≡ 2 [MOD 6] := by sorry
  -- Proof of the theorem:
  sorry

end product_modulo_6_l89_89292


namespace fourth_polygon_is_square_l89_89163

theorem fourth_polygon_is_square
  (angle_triangle angle_square angle_hexagon : ℕ)
  (h_triangle : angle_triangle = 60)
  (h_square : angle_square = 90)
  (h_hexagon : angle_hexagon = 120)
  (h_total : angle_triangle + angle_square + angle_hexagon = 270) :
  ∃ angle_fourth : ℕ, angle_fourth = 90 ∧ (angle_fourth + angle_triangle + angle_square + angle_hexagon = 360) :=
sorry

end fourth_polygon_is_square_l89_89163


namespace solve_fraction_eq_l89_89277

theorem solve_fraction_eq : 
  ∀ x : ℝ, (x - 3) ≠ 0 → (x + 6) / (x - 3) = 4 → x = 6 := by
  intros x h_ne_zero h_eq
  sorry

end solve_fraction_eq_l89_89277


namespace exists_a_b_divisible_l89_89809

theorem exists_a_b_divisible (n : ℕ) (hn : 0 < n) : 
  ∃ a b : ℤ, (4 * a^2 + 9 * b^2 - 1) % n = 0 := 
sorry

end exists_a_b_divisible_l89_89809


namespace Cheryl_total_distance_l89_89388

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end Cheryl_total_distance_l89_89388


namespace paint_fence_together_time_l89_89383

-- Define the times taken by Jamshid and Taimour
def Taimour_time := 18 -- Taimour takes 18 hours to paint the fence
def Jamshid_time := Taimour_time / 2 -- Jamshid takes half the time Taimour takes

-- Define the work rates
def Taimour_rate := 1 / Taimour_time
def Jamshid_rate := 1 / Jamshid_time

-- Define the combined work rate
def combined_rate := Taimour_rate + Jamshid_rate

-- Define the total time taken when working together
def together_time := 1 / combined_rate

-- State the main theorem
theorem paint_fence_together_time : together_time = 6 := 
sorry

end paint_fence_together_time_l89_89383


namespace time_left_after_council_room_is_zero_l89_89481

-- Define the conditions
def totalTimeAllowed : ℕ := 30
def travelToSchoolTime : ℕ := 25
def walkToLibraryTime : ℕ := 3
def returnBooksTime : ℕ := 4
def walkToCouncilRoomTime : ℕ := 5
def submitProjectTime : ℕ := 3

-- Calculate time spent up to the student council room
def timeSpentUpToCouncilRoom : ℕ :=
  travelToSchoolTime + walkToLibraryTime + returnBooksTime + walkToCouncilRoomTime + submitProjectTime

-- Question: How much time is left after leaving the student council room to reach the classroom without being late?
theorem time_left_after_council_room_is_zero (totalTimeAllowed travelToSchoolTime walkToLibraryTime returnBooksTime walkToCouncilRoomTime submitProjectTime : ℕ):
  totalTimeAllowed - timeSpentUpToCouncilRoom = 0 := by
  sorry

end time_left_after_council_room_is_zero_l89_89481


namespace Robert_GRE_exam_l89_89236

/-- Robert started preparation for GRE entrance examination in the month of January and prepared for 5 months. Prove that he could write the examination any date after the end of May.-/
theorem Robert_GRE_exam (start_month : ℕ) (prep_duration : ℕ) : 
  start_month = 1 → prep_duration = 5 → ∃ exam_date, exam_date > 5 :=
by
  sorry

end Robert_GRE_exam_l89_89236


namespace height_difference_petronas_empire_state_l89_89530

theorem height_difference_petronas_empire_state :
  let esb_height := 443
  let pt_height := 452
  pt_height - esb_height = 9 := by
  sorry

end height_difference_petronas_empire_state_l89_89530


namespace both_firms_participate_number_of_firms_participate_social_optimality_l89_89711

-- Definitions for general conditions
variable (α V IC : ℝ)
variable (hα : 0 < α ∧ α < 1)

-- Condition for both firms to participate
def condition_to_participate (V : ℝ) (α : ℝ) (IC : ℝ) : Prop :=
  V * α * (1 - 0.5 * α) ≥ IC

-- Part (a): Under what conditions will both firms participate?
theorem both_firms_participate (α V IC : ℝ) (hα : 0 < α ∧ α < 1) :
  condition_to_participate V α IC → (V * α * (1 - 0.5 * α) ≥ IC) :=
by sorry

-- Part (b): Given V=16, α=0.5, and IC=5, determine the number of firms participating
theorem number_of_firms_participate :
  (condition_to_participate 16 0.5 5) :=
by sorry

-- Part (c): To determine if the number of participating firms is socially optimal
def total_profit (α V IC : ℝ) (both : Bool) :=
  if both then 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
  else α * V - IC

theorem social_optimality :
   (total_profit 0.5 16 5 true ≠ max (total_profit 0.5 16 5 true) (total_profit 0.5 16 5 false)) :=
by sorry

end both_firms_participate_number_of_firms_participate_social_optimality_l89_89711


namespace find_b_for_perpendicular_lines_l89_89865

theorem find_b_for_perpendicular_lines:
  (∃ b : ℝ, ∀ (x y : ℝ), (3 * x + y - 5 = 0) ∧ (b * x + y + 2 = 0) → b = -1/3) :=
by
  sorry

end find_b_for_perpendicular_lines_l89_89865


namespace molecular_weight_BaCl2_l89_89814

def molecular_weight_one_mole (w_four_moles : ℕ) (n : ℕ) : ℕ := 
    w_four_moles / n

theorem molecular_weight_BaCl2 
    (w_four_moles : ℕ)
    (H : w_four_moles = 828) :
  molecular_weight_one_mole w_four_moles 4 = 207 :=
by
  -- sorry to skip the proof
  sorry

end molecular_weight_BaCl2_l89_89814


namespace min_value_frac_sum_l89_89624

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (3 * z / (x + 2 * y) + 5 * x / (2 * y + 3 * z) + 2 * y / (3 * x + z)) ≥ 3 / 4 :=
by
  sorry

end min_value_frac_sum_l89_89624


namespace find_a_b_l89_89199

theorem find_a_b :
  ∃ (a b : ℚ), 
    (∀ x : ℚ, x = 2 → (a * x^3 - 6 * x^2 + b * x - 5 - 3 = 0)) ∧
    (∀ x : ℚ, x = -1 → (a * x^3 - 6 * x^2 + b * x - 5 - 7 = 0)) ∧
    (a = -2/3 ∧ b = -52/3) :=
by {
  sorry
}

end find_a_b_l89_89199


namespace function_satisfies_conditions_l89_89549

-- Define the conditions
def f (n : ℕ) : ℕ := n + 1

-- Prove that the function f satisfies the given conditions
theorem function_satisfies_conditions : 
  (f 0 = 1) ∧ (f 2012 = 2013) :=
by
  sorry

end function_satisfies_conditions_l89_89549


namespace count_unique_lists_of_five_l89_89911

theorem count_unique_lists_of_five :
  (∃ (f : ℕ → ℕ), ∀ (i j : ℕ), i < j → f (i + 1) - f i = 3 ∧ j = 5 → f 5 % f 1 = 0) →
  (∃ (n : ℕ), n = 6) :=
by
  sorry

end count_unique_lists_of_five_l89_89911


namespace complements_intersection_l89_89678

open Set

noncomputable def U : Set ℕ := { x | x ≤ 5 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complements_intersection :
  (U \ A) ∩ (U \ B) = {0, 5} :=
by
  sorry

end complements_intersection_l89_89678


namespace points_lie_on_line_l89_89008

noncomputable def x (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 + 2 * t + 2) / t
noncomputable def y (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 - 2 * t + 2) / t

theorem points_lie_on_line : ∀ (t : ℝ) (ht : t ≠ 0), y t ht = x t ht - 4 :=
by 
  intros t ht
  simp [x, y]
  sorry

end points_lie_on_line_l89_89008


namespace triangle_obtuse_l89_89139

def is_obtuse_triangle (A B C : ℝ) : Prop := A > 90 ∨ B > 90 ∨ C > 90

theorem triangle_obtuse (A B C : ℝ) (h1 : A > 3 * B) (h2 : C < 2 * B) (h3 : A + B + C = 180) : is_obtuse_triangle A B C :=
by sorry

end triangle_obtuse_l89_89139


namespace cylinder_not_occupied_volume_l89_89374

theorem cylinder_not_occupied_volume :
  let r := 10
  let h_cylinder := 30
  let h_full_cone := 10
  let volume_cylinder := π * r^2 * h_cylinder
  let volume_full_cone := (1 / 3) * π * r^2 * h_full_cone
  let volume_half_cone := (1 / 2) * volume_full_cone
  let volume_unoccupied := volume_cylinder - (volume_full_cone + volume_half_cone)
  volume_unoccupied = 2500 * π := 
by
  sorry

end cylinder_not_occupied_volume_l89_89374


namespace dance_relationship_l89_89331

theorem dance_relationship (b g : ℕ) 
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ b → i = 1 → ∃ m, m = 7)
  (h2 : b = g - 6) 
  : 7 + (b - 1) = g := 
by
  sorry

end dance_relationship_l89_89331


namespace movie_theater_ticket_cost_l89_89967

theorem movie_theater_ticket_cost
  (adult_ticket_cost : ℝ)
  (child_ticket_cost : ℝ)
  (total_moviegoers : ℝ)
  (total_amount_paid : ℝ)
  (number_of_adults : ℝ)
  (H_child_ticket_cost : child_ticket_cost = 6.50)
  (H_total_moviegoers : total_moviegoers = 7)
  (H_total_amount_paid : total_amount_paid = 54.50)
  (H_number_of_adults : number_of_adults = 3)
  (H_number_of_children : total_moviegoers - number_of_adults = 4) :
  adult_ticket_cost = 9.50 :=
sorry

end movie_theater_ticket_cost_l89_89967
