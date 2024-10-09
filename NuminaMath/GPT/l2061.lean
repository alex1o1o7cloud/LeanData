import Mathlib

namespace class_student_numbers_l2061_206113

theorem class_student_numbers (a b c d : ℕ) 
    (h_avg : (a + b + c + d) / 4 = 46)
    (h_diff_ab : a - b = 4)
    (h_diff_bc : b - c = 3)
    (h_diff_cd : c - d = 2)
    (h_max_a : a > b ∧ a > c ∧ a > d) : 
    a = 51 ∧ b = 47 ∧ c = 44 ∧ d = 42 := 
by 
  sorry

end class_student_numbers_l2061_206113


namespace intersection_A_B_l2061_206115

noncomputable def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
noncomputable def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l2061_206115


namespace find_g_of_5_l2061_206121

theorem find_g_of_5 (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x * y) = g x * g y) 
  (h2 : g 1 = 2) : 
  g 5 = 32 := 
by 
  sorry

end find_g_of_5_l2061_206121


namespace ratio_of_projection_l2061_206103

theorem ratio_of_projection (x y : ℝ)
  (h : ∀ (x y : ℝ), (∃ x y : ℝ, 
  (3/25 * x + 4/25 * y = x) ∧ (4/25 * x + 12/25 * y = y))) : x / y = 2 / 11 :=
sorry

end ratio_of_projection_l2061_206103


namespace tablet_screen_area_difference_l2061_206165

theorem tablet_screen_area_difference (d1 d2 : ℝ) (A1 A2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 7) :
  A1 - A2 = 7.5 :=
by
  -- Note: The proof is omitted as the prompt requires only the statement.
  sorry

end tablet_screen_area_difference_l2061_206165


namespace value_of_x_l2061_206110

theorem value_of_x (x : ℝ) : (12 - x)^3 = x^3 → x = 12 :=
by
  sorry

end value_of_x_l2061_206110


namespace k_value_tangent_l2061_206152

-- Defining the equations
def line (k : ℝ) (x y : ℝ) : Prop := 3 * x + 5 * y + k = 0
def parabola (x y : ℝ) : Prop := y^2 = 24 * x

-- The main theorem stating that k must be 50 for the line to be tangent to the parabola
theorem k_value_tangent (k : ℝ) : (∀ x y : ℝ, line k x y → parabola x y → True) → k = 50 :=
by 
  -- The proof can be constructed based on the discriminant condition provided in the problem
  sorry

end k_value_tangent_l2061_206152


namespace train_speed_ratio_l2061_206173

theorem train_speed_ratio 
  (v_A v_B : ℝ)
  (h1 : v_A = 2 * v_B)
  (h2 : 27 = L_A / v_A)
  (h3 : 17 = L_B / v_B)
  (h4 : 22 = (L_A + L_B) / (v_A + v_B))
  (h5 : v_A + v_B ≤ 60) :
  v_A / v_B = 2 := by
  sorry

-- Conditions given must be defined properly
variables (L_A L_B : ℝ)

end train_speed_ratio_l2061_206173


namespace triangle_perimeter_l2061_206151

theorem triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) 
  (h1 : area = 150)
  (h2 : leg1 = 30)
  (h3 : 0 < leg2)
  (h4 : hypotenuse = (leg1^2 + leg2^2).sqrt)
  (hArea : area = 0.5 * leg1 * leg2)
  : hypotenuse = 10 * Real.sqrt 10 ∧ leg2 = 10 ∧ (leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10) := 
by
  sorry

end triangle_perimeter_l2061_206151


namespace fruit_box_assignment_proof_l2061_206157

-- Definitions of the boxes with different fruits
inductive Fruit | Apple | Pear | Orange | Banana
open Fruit

-- Define a function representing the placement of fruits in the boxes
def box_assignment := ℕ → Fruit

-- Conditions based on the problem statement
def conditions (assign : box_assignment) : Prop :=
  assign 1 ≠ Orange ∧
  assign 2 ≠ Pear ∧
  (assign 1 = Banana → assign 3 ≠ Apple ∧ assign 3 ≠ Pear) ∧
  assign 4 ≠ Apple

-- The correct assignment of fruits to boxes
def correct_assignment (assign : box_assignment) : Prop :=
  assign 1 = Banana ∧
  assign 2 = Apple ∧
  assign 3 = Orange ∧
  assign 4 = Pear

-- Theorem statement
theorem fruit_box_assignment_proof : ∃ assign : box_assignment, conditions assign ∧ correct_assignment assign :=
sorry

end fruit_box_assignment_proof_l2061_206157


namespace planes_are_perpendicular_l2061_206137

-- Define the normal vectors
def N1 : List ℝ := [2, 3, -4]
def N2 : List ℝ := [5, -2, 1]

-- Define the dot product function
def dotProduct (v1 v2 : List ℝ) : ℝ :=
  List.zipWith (fun a b => a * b) v1 v2 |>.sum

-- State the theorem
theorem planes_are_perpendicular :
  dotProduct N1 N2 = 0 :=
by
  sorry

end planes_are_perpendicular_l2061_206137


namespace max_frac_a_c_squared_l2061_206192

theorem max_frac_a_c_squared 
  (a b c : ℝ) (y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order: a ≥ b ∧ b ≥ c)
  (h_system: a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2)
  (h_bounds: 0 ≤ y ∧ y < a ∧ 0 ≤ z ∧ z < c) :
  (a/c)^2 ≤ 4/3 :=
sorry

end max_frac_a_c_squared_l2061_206192


namespace sum_of_numbers_l2061_206189

theorem sum_of_numbers : 145 + 33 + 29 + 13 = 220 :=
by
  sorry

end sum_of_numbers_l2061_206189


namespace larger_of_two_numbers_l2061_206143

noncomputable def larger_number (HCF LCM A B : ℕ) : ℕ :=
  if HCF = 23 ∧ LCM = 23 * 9 * 10 ∧ A * B = HCF * LCM ∧ (A = 10 ∧ B = 23 * 9 ∨ B = 10 ∧ A = 23 * 9)
  then max A B
  else 0

theorem larger_of_two_numbers : larger_number (23) (23 * 9 * 10) 230 207 = 230 := by
  sorry

end larger_of_two_numbers_l2061_206143


namespace average_sales_l2061_206162

theorem average_sales
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 90)
  (h2 : a2 = 50)
  (h3 : a3 = 70)
  (h4 : a4 = 110)
  (h5 : a5 = 80) :
  (a1 + a2 + a3 + a4 + a5) / 5 = 80 :=
by
  sorry

end average_sales_l2061_206162


namespace sequence_terms_proof_l2061_206169

theorem sequence_terms_proof (P Q R T U V W : ℤ) (S : ℤ) 
  (h1 : S = 10) 
  (h2 : P + Q + R + S = 40) 
  (h3 : Q + R + S + T = 40) 
  (h4 : R + S + T + U = 40) 
  (h5 : S + T + U + V = 40) 
  (h6 : T + U + V + W = 40) : 
  P + W = 40 := 
by 
  have h7 : P + Q + R + 10 = 40 := by rwa [h1] at h2
  have h8 : Q + R + 10 + T = 40 := by rwa [h1] at h3
  have h9 : R + 10 + T + U = 40 := by rwa [h1] at h4
  have h10 : 10 + T + U + V = 40 := by rwa [h1] at h5
  have h11 : T + U + V + W = 40 := h6
  sorry

end sequence_terms_proof_l2061_206169


namespace no_integer_solutions_l2061_206139

theorem no_integer_solutions (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hq : Nat.Prime (2*p + 1)) :
  ∀ (x y z : ℤ), x^p + 2 * y^p + 5 * z^p = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end no_integer_solutions_l2061_206139


namespace T_perimeter_l2061_206177

theorem T_perimeter (l w : ℝ) (h1 : l = 4) (h2 : w = 2) :
  let rect_perimeter := 2 * l + 2 * w
  let overlap := 2 * w
  2 * rect_perimeter - overlap = 20 :=
by
  -- Proof will be added here
  sorry

end T_perimeter_l2061_206177


namespace analytical_expression_maximum_value_l2061_206167

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1

theorem analytical_expression (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, abs (x - (x + (Real.pi / (2 * ω)))) = Real.pi / 2) : 
  f x 2 = 2 * Real.sin (2 * x - Real.pi / 6) + 1 :=
sorry

theorem maximum_value (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  2 * Real.sin (2 * x - Real.pi / 6) + 1 ≤ 3 :=
sorry

end analytical_expression_maximum_value_l2061_206167


namespace area_of_connected_colored_paper_l2061_206141

noncomputable def side_length : ℕ := 30
noncomputable def overlap : ℕ := 7
noncomputable def sheets : ℕ := 6
noncomputable def total_length : ℕ := side_length + (sheets - 1) * (side_length - overlap)
noncomputable def width : ℕ := side_length

theorem area_of_connected_colored_paper : total_length * width = 4350 := by
  sorry

end area_of_connected_colored_paper_l2061_206141


namespace cube_roof_ratio_proof_l2061_206156

noncomputable def cube_roof_edge_ratio : Prop :=
  ∃ (a b : ℝ), (∃ isosceles_triangles symmetrical_trapezoids : ℝ, isosceles_triangles = 2 ∧ symmetrical_trapezoids = 2)
  ∧ (∀ edge : ℝ, edge = a)
  ∧ (∀ face1 face2 : ℝ, face1 = face2)
  ∧ b = (Real.sqrt 5 - 1) / 2 * a

theorem cube_roof_ratio_proof : cube_roof_edge_ratio :=
sorry

end cube_roof_ratio_proof_l2061_206156


namespace carpet_dimensions_l2061_206191

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end carpet_dimensions_l2061_206191


namespace percentage_calculation_l2061_206160

theorem percentage_calculation (Part Whole : ℕ) (h1 : Part = 90) (h2 : Whole = 270) : 
  ((Part : ℝ) / (Whole : ℝ) * 100) = 33.33 :=
by
  sorry

end percentage_calculation_l2061_206160


namespace initial_weight_of_load_l2061_206142

variable (W : ℝ)
variable (h : 0.8 * 0.9 * W = 36000)

theorem initial_weight_of_load :
  W = 50000 :=
by
  sorry

end initial_weight_of_load_l2061_206142


namespace interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l2061_206148

section
open Real

noncomputable def interval1 (x : ℝ) : Real := log (1 - x ^ 2)
noncomputable def interval2 (x : ℝ) : Real := x * (1 + 2 * sqrt x)
noncomputable def interval3 (x : ℝ) : Real := log (abs x)

-- Function 1: p = ln(1 - x^2)
theorem interval1_increase_decrease :
  (∀ x : ℝ, -1 < x ∧ x < 0 → deriv interval1 x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → deriv interval1 x < 0) := by
  sorry

-- Function 2: z = x(1 + 2√x)
theorem interval2_increasing :
  ∀ x : ℝ, x ≥ 0 → deriv interval2 x > 0 := by
  sorry

-- Function 3: y = ln|x|
theorem interval3_increase_decrease :
  (∀ x : ℝ, x < 0 → deriv interval3 x < 0) ∧
  (∀ x : ℝ, x > 0 → deriv interval3 x > 0) := by
  sorry

end

end interval1_increase_decrease_interval2_increasing_interval3_increase_decrease_l2061_206148


namespace hundredth_number_is_100_l2061_206161

/-- Define the sequence of numbers said by Jo, Blair, and Parker following the conditions described. --/
def next_number (turn : ℕ) : ℕ :=
  -- Each turn increments by one number starting from 1
  turn

-- Prove that the 100th number in the sequence is 100
theorem hundredth_number_is_100 :
  next_number 100 = 100 := 
by sorry

end hundredth_number_is_100_l2061_206161


namespace subset_exists_l2061_206186

theorem subset_exists (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ) (hA : A.card = p - 1) 
  (hA_div : ∀ a ∈ A, ¬ p ∣ a) :
  ∀ n ∈ Finset.range p, ∃ B ⊆ A, (B.sum id) % p = n :=
by
  -- Proof goes here
  sorry

end subset_exists_l2061_206186


namespace sam_has_8_marbles_l2061_206188

theorem sam_has_8_marbles :
  ∀ (steve sam sally : ℕ),
  sam = 2 * steve →
  sally = sam - 5 →
  steve + 3 = 10 →
  sam - 6 = 8 :=
by
  intros steve sam sally
  intros h1 h2 h3
  sorry

end sam_has_8_marbles_l2061_206188


namespace smallest_positive_multiple_l2061_206174

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end smallest_positive_multiple_l2061_206174


namespace polynomials_equality_l2061_206179

open Polynomial

variable {F : Type*} [Field F]

theorem polynomials_equality (P Q : Polynomial F) (h : ∀ x, P.eval (P.eval (P.eval x)) = Q.eval (Q.eval (Q.eval x)) ∧ P.eval (P.eval (P.eval x)) = Q.eval (P.eval (P.eval x))) : 
  P = Q := 
sorry

end polynomials_equality_l2061_206179


namespace sphere_surface_area_quadruple_l2061_206116

theorem sphere_surface_area_quadruple (r : ℝ) :
  (4 * π * (2 * r)^2) = 4 * (4 * π * r^2) :=
by
  sorry

end sphere_surface_area_quadruple_l2061_206116


namespace pow_ge_double_plus_one_l2061_206155

theorem pow_ge_double_plus_one (n : ℕ) (h : n ≥ 3) : 2^n ≥ 2 * (n + 1) :=
sorry

end pow_ge_double_plus_one_l2061_206155


namespace area_of_rectangle_ABCD_l2061_206195

-- Definitions for the conditions
def small_square_area := 4
def total_small_squares := 2
def large_square_area := (2 * (2 : ℝ)) * (2 * (2 : ℝ))
def total_squares_area := total_small_squares * small_square_area + large_square_area

-- The main proof statement
theorem area_of_rectangle_ABCD : total_squares_area = 24 := 
by
  sorry

end area_of_rectangle_ABCD_l2061_206195


namespace sine_central_angle_of_circular_sector_eq_4_5_l2061_206101

theorem sine_central_angle_of_circular_sector_eq_4_5
  (R : Real)
  (α : Real)
  (h : π * R ^ 2 * Real.sin α = 2 * π * R ^ 2 * (1 - Real.cos α)) :
  Real.sin α = 4 / 5 := by
  sorry

end sine_central_angle_of_circular_sector_eq_4_5_l2061_206101


namespace trapezoid_larger_base_length_l2061_206197

theorem trapezoid_larger_base_length
  (x : ℝ)
  (h_ratio : 3 = 3 * 1)
  (h_midline : (x + 3 * x) / 2 = 24) :
  3 * x = 36 :=
by
  sorry

end trapezoid_larger_base_length_l2061_206197


namespace range_of_a_l2061_206185

noncomputable def f : ℝ → ℝ → ℝ
| a, x =>
  if x ≥ -1 then a * x ^ 2 + 2 * x 
  else (1 - 3 * a) * x - 3 / 2

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → 0 < a ∧ a ≤ 1/4 :=
sorry

end range_of_a_l2061_206185


namespace rhombus_area_l2061_206135

def d1 : ℝ := 10
def d2 : ℝ := 30

theorem rhombus_area (d1 d2 : ℝ) : (d1 * d2) / 2 = 150 := by
  sorry

end rhombus_area_l2061_206135


namespace sufficient_but_not_necessary_condition_l2061_206106

theorem sufficient_but_not_necessary_condition
  (p q r : Prop)
  (h_p_sufficient_q : p → q)
  (h_r_necessary_q : q → r)
  (h_p_not_necessary_q : ¬ (q → p))
  (h_r_not_sufficient_q : ¬ (r → q)) :
  (p → r) ∧ ¬ (r → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2061_206106


namespace min_value_fraction_l2061_206114

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  ∃ c, (c = 9) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ x + y = 1) → (1/x + 4/y ≥ c)) :=
by
  sorry

end min_value_fraction_l2061_206114


namespace scientific_notation_example_l2061_206130

theorem scientific_notation_example : 10500 = 1.05 * 10^4 :=
by
  sorry

end scientific_notation_example_l2061_206130


namespace angle_triple_complement_l2061_206198

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end angle_triple_complement_l2061_206198


namespace product_of_five_consecutive_numbers_not_square_l2061_206193

theorem product_of_five_consecutive_numbers_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) :=
by
  sorry

end product_of_five_consecutive_numbers_not_square_l2061_206193


namespace compare_trig_values_l2061_206138

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 7)
noncomputable def b : ℝ := Real.tan (5 * Real.pi / 7)
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 7)

theorem compare_trig_values :
  (0 < 2 * Real.pi / 7 ∧ 2 * Real.pi / 7 < Real.pi / 2) →
  (Real.pi / 2 < 5 * Real.pi / 7 ∧ 5 * Real.pi / 7 < 3 * Real.pi / 4) →
  b < c ∧ c < a :=
by
  intro h1 h2
  sorry

end compare_trig_values_l2061_206138


namespace grant_earnings_proof_l2061_206168

noncomputable def total_earnings (X Y Z W : ℕ): ℕ :=
  let first_month := X
  let second_month := 3 * X + Y
  let third_month := 2 * second_month - Z
  let average := (first_month + second_month + third_month) / 3
  let fourth_month := average + W
  first_month + second_month + third_month + fourth_month

theorem grant_earnings_proof : total_earnings 350 30 20 50 = 5810 := by
  sorry

end grant_earnings_proof_l2061_206168


namespace initial_volume_of_solution_l2061_206150

variable (V : ℝ)
variables (h1 : 0.10 * V = 0.08 * (V + 16))
variables (V_correct : V = 64)

theorem initial_volume_of_solution : V = 64 := by
  sorry

end initial_volume_of_solution_l2061_206150


namespace sum_of_eight_numbers_l2061_206129

theorem sum_of_eight_numbers (avg : ℝ) (S : ℝ) (h1 : avg = 5.3) (h2 : avg = S / 8) : S = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l2061_206129


namespace A_receives_more_than_B_l2061_206153

variable (A B C : ℝ)

axiom h₁ : A = 1/3 * (B + C)
axiom h₂ : B = 2/7 * (A + C)
axiom h₃ : A + B + C = 720

theorem A_receives_more_than_B : A - B = 20 :=
by
  sorry

end A_receives_more_than_B_l2061_206153


namespace permutation_average_sum_l2061_206105

theorem permutation_average_sum :
  let p := 286
  let q := 11
  p + q = 297 :=
by
  sorry

end permutation_average_sum_l2061_206105


namespace mystical_words_count_l2061_206144

-- We define a function to count words given the conditions
def count_possible_words : ℕ := 
  let total_words : ℕ := (20^1 - 19^1) + (20^2 - 19^2) + (20^3 - 19^3) + (20^4 - 19^4) + (20^5 - 19^5)
  total_words

theorem mystical_words_count : count_possible_words = 755761 :=
by 
  unfold count_possible_words
  sorry

end mystical_words_count_l2061_206144


namespace range_of_m_l2061_206172

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0
def q (m : ℝ) : Prop := ∃ y : ℝ, ∀ x : ℝ, (x^2)/(m-1) + y^2 = 1
def not_p (m : ℝ) : Prop := ¬ (p m)
def p_and_q (m : ℝ) : Prop := (p m) ∧ (q m)

theorem range_of_m (m : ℝ) : (¬ (not_p m) ∧ ¬ (p_and_q m)) → 1 < m ∧ m ≤ 2 :=
sorry

end range_of_m_l2061_206172


namespace sequence_a4_l2061_206158

theorem sequence_a4 (S : ℕ → ℚ) (a : ℕ → ℚ) 
  (hS : ∀ n, S n = (n + 1) / (n + 2))
  (hS0 : S 0 = a 0)
  (hSn : ∀ n, S (n + 1) = S n + a (n + 1)) :
  a 4 = 1 / 30 := 
sorry

end sequence_a4_l2061_206158


namespace min_questions_to_determine_product_50_numbers_l2061_206133

/-- Prove that to uniquely determine the product of 50 numbers each either +1 or -1 
arranged on the circumference of a circle by asking for the product of three 
consecutive numbers, one must ask a minimum of 50 questions. -/
theorem min_questions_to_determine_product_50_numbers : 
  ∀ (a : ℕ → ℤ), (∀ i, a i = 1 ∨ a i = -1) → 
  (∀ i, ∃ b : ℤ, b = a i * a (i+1) * a (i+2)) → 
  ∃ n, n = 50 :=
by
  sorry

end min_questions_to_determine_product_50_numbers_l2061_206133


namespace evaluate_expression_l2061_206120

theorem evaluate_expression : 
  (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = 1372 * 10^1003 := 
by sorry

end evaluate_expression_l2061_206120


namespace books_on_shelf_l2061_206181

theorem books_on_shelf (original_books : ℕ) (books_added : ℕ) (total_books : ℕ) (h1 : original_books = 38) 
(h2 : books_added = 10) : total_books = 48 :=
by 
  sorry

end books_on_shelf_l2061_206181


namespace checkerboard_7_strips_l2061_206178

theorem checkerboard_7_strips (n : ℤ) :
  (n % 7 = 3) →
  ∃ m : ℤ, n^2 = 9 + 7 * m :=
by
  intro h
  sorry

end checkerboard_7_strips_l2061_206178


namespace shekar_marks_math_l2061_206126

theorem shekar_marks_math (M : ℕ) (science : ℕ) (social_studies : ℕ) (english : ℕ) 
(biology : ℕ) (average : ℕ) (num_subjects : ℕ) 
(h_science : science = 65)
(h_social : social_studies = 82)
(h_english : english = 67)
(h_biology : biology = 55)
(h_average : average = 69)
(h_num_subjects : num_subjects = 5) :
M + science + social_studies + english + biology = average * num_subjects →
M = 76 :=
by
  sorry

end shekar_marks_math_l2061_206126


namespace ratio_of_sum_l2061_206134

theorem ratio_of_sum (a b c : ℚ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := 
by
  sorry

end ratio_of_sum_l2061_206134


namespace bob_correct_answers_l2061_206128

-- Define the variables, c for correct answers, w for incorrect answers, total problems 15, score 54
variables (c w : ℕ)

-- Define the conditions
axiom total_problems : c + w = 15
axiom total_score : 6 * c - 3 * w = 54

-- Prove that the number of correct answers is 11
theorem bob_correct_answers : c = 11 :=
by
  -- Here, you would provide the proof, but for the sake of the statement, we'll use sorry.
  sorry

end bob_correct_answers_l2061_206128


namespace imaginary_unit_calculation_l2061_206164

theorem imaginary_unit_calculation (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := 
by
  sorry

end imaginary_unit_calculation_l2061_206164


namespace min_sum_is_11_over_28_l2061_206183

-- Definition of the problem
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the minimum sum problem
def min_sum (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits →
  ((A : ℚ) / B + (C : ℚ) / D) = (11 : ℚ) / 28

-- The theorem statement
theorem min_sum_is_11_over_28 :
  ∃ A B C D : ℕ, min_sum A B C D :=
sorry

end min_sum_is_11_over_28_l2061_206183


namespace rabbit_catch_up_time_l2061_206127

theorem rabbit_catch_up_time :
  let rabbit_speed := 25 -- miles per hour
  let cat_speed := 20 -- miles per hour
  let head_start := 15 / 60 -- hours, which is 0.25 hours
  let initial_distance := cat_speed * head_start
  let relative_speed := rabbit_speed - cat_speed
  initial_distance / relative_speed = 1 := by
  sorry

end rabbit_catch_up_time_l2061_206127


namespace quadratic_root_condition_l2061_206100

theorem quadratic_root_condition (b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + b*x1 + c = 0) ∧ (x2^2 + b*x2 + c = 0)) ↔ (b^2 - 4*c ≥ 0) :=
by
  sorry

end quadratic_root_condition_l2061_206100


namespace percent_students_own_cats_l2061_206154

theorem percent_students_own_cats 
  (total_students : ℕ) (cat_owners : ℕ) (h1 : total_students = 300) (h2 : cat_owners = 45) :
  (cat_owners : ℚ) / total_students * 100 = 15 := 
by
  sorry

end percent_students_own_cats_l2061_206154


namespace ticket_price_reduction_l2061_206107

theorem ticket_price_reduction
    (original_price : ℝ := 50)
    (increase_in_tickets : ℝ := 1 / 3)
    (increase_in_revenue : ℝ := 1 / 4)
    (x : ℝ)
    (reduced_price : ℝ)
    (new_tickets : ℝ := x * (1 + increase_in_tickets))
    (original_revenue : ℝ := x * original_price)
    (new_revenue : ℝ := new_tickets * reduced_price) :
    new_revenue = (1 + increase_in_revenue) * original_revenue →
    reduced_price = original_price - (original_price / 2) :=
    sorry

end ticket_price_reduction_l2061_206107


namespace no_unboxed_products_l2061_206146

-- Definitions based on the conditions
def big_box_capacity : ℕ := 50
def small_box_capacity : ℕ := 40
def total_products : ℕ := 212

-- Theorem statement proving the least number of unboxed products
theorem no_unboxed_products (big_box_capacity small_box_capacity total_products : ℕ) : 
  (total_products - (total_products / big_box_capacity) * big_box_capacity) % small_box_capacity = 0 :=
by
  sorry

end no_unboxed_products_l2061_206146


namespace gain_percentage_of_watch_l2061_206136

theorem gain_percentage_of_watch :
  let CP := 1076.923076923077
  let S1 := CP * 0.90
  let S2 := S1 + 140
  let gain_percentage := ((S2 - CP) / CP) * 100
  gain_percentage = 3 := by
  sorry

end gain_percentage_of_watch_l2061_206136


namespace product_sqrt_50_l2061_206145

theorem product_sqrt_50 (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : a^2 < 50) (h₄ : 50 < b^2) : a * b = 56 := by
  sorry

end product_sqrt_50_l2061_206145


namespace repunit_polynomial_characterization_l2061_206112

noncomputable def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def polynomial_condition (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_repunit n → is_repunit (f n)

theorem repunit_polynomial_characterization :
  ∀ (f : ℕ → ℕ), polynomial_condition f ↔
  ∃ m r : ℕ, m ≥ 0 ∧ r ≥ 1 - m ∧ ∀ n : ℕ, f n = (10^r * (9 * n + 1)^m - 1) / 9 :=
by
  sorry

end repunit_polynomial_characterization_l2061_206112


namespace sphere_radius_eq_three_l2061_206171

theorem sphere_radius_eq_three (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 := 
sorry

end sphere_radius_eq_three_l2061_206171


namespace two_students_exist_l2061_206149

theorem two_students_exist (scores : Fin 49 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (i j : Fin 49), i ≠ j ∧ (scores i).1 ≥ (scores j).1 ∧ (scores i).2.1 ≥ (scores j).2.1 ∧ (scores i).2.2 ≥ (scores j).2.2 := 
by
  sorry

end two_students_exist_l2061_206149


namespace range_of_a_l2061_206159

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 + a ≤ 0
def q (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a

-- The theorem statement: if p is false and q is true, then 1 < a < 2
theorem range_of_a (a : ℝ) (h1 : ¬ p a) (h2 : q a) : 1 < a ∧ a < 2 :=
sorry

end range_of_a_l2061_206159


namespace determine_function_l2061_206104

theorem determine_function (f : ℕ → ℕ) :
  (∀ a b c d : ℕ, 2 * a * b = c^2 + d^2 → f (a + b) = f a + f b + f c + f d) →
  ∃ k : ℕ, ∀ n : ℕ, f n = k * n^2 :=
by
  sorry

end determine_function_l2061_206104


namespace no_real_roots_quadratic_eq_l2061_206108

theorem no_real_roots_quadratic_eq :
  ¬ ∃ x : ℝ, 7 * x^2 - 4 * x + 6 = 0 :=
by sorry

end no_real_roots_quadratic_eq_l2061_206108


namespace correct_equation_l2061_206102

theorem correct_equation (a b : ℝ) : 
  (a + b)^2 = a^2 + 2 * a * b + b^2 := by
  sorry

end correct_equation_l2061_206102


namespace least_common_denominator_l2061_206176

-- We first need to define the function to compute the LCM of a list of natural numbers.
def lcm_list (ns : List ℕ) : ℕ :=
ns.foldr Nat.lcm 1

theorem least_common_denominator : 
  lcm_list [3, 4, 5, 8, 9, 11] = 3960 := 
by
  -- Here's where the proof would go
  sorry

end least_common_denominator_l2061_206176


namespace ball_placement_count_l2061_206124

-- Definitions for the balls and their numbering
inductive Ball
| b1
| b2
| b3
| b4

-- Definitions for the boxes and their numbering
inductive Box
| box1
| box2
| box3

-- Function that checks if an assignment is valid given the conditions.
def isValidAssignment (assignment : Ball → Box) : Prop :=
  assignment Ball.b1 ≠ Box.box1 ∧ assignment Ball.b3 ≠ Box.box3

-- Main statement to prove
theorem ball_placement_count : 
  ∃ (assignments : Finset (Ball → Box)), 
    (∀ f ∈ assignments, isValidAssignment f) ∧ assignments.card = 14 := 
sorry

end ball_placement_count_l2061_206124


namespace union_of_M_and_N_l2061_206190

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_M_and_N : M ∪ N = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_M_and_N_l2061_206190


namespace b_bounded_l2061_206182

open Real

-- Define sequences of real numbers
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define initial conditions and properties
axiom a0_gt_half : a 0 > 1/2
axiom a_non_decreasing : ∀ n : ℕ, a (n + 1) ≥ a n
axiom b_recursive : ∀ n : ℕ, b (n + 1) = a n * (b n + b (n + 2))

-- Prove the sequence (b_n) is bounded
theorem b_bounded : ∃ M : ℝ, ∀ n : ℕ, b n ≤ M :=
by
  sorry

end b_bounded_l2061_206182


namespace gotham_street_termite_ridden_not_collapsing_l2061_206131

def fraction_termite_ridden := 1 / 3
def fraction_collapsing_given_termite_ridden := 4 / 7
def fraction_not_collapsing := 3 / 21

theorem gotham_street_termite_ridden_not_collapsing
  (h1: fraction_termite_ridden = 1 / 3)
  (h2: fraction_collapsing_given_termite_ridden = 4 / 7) :
  fraction_termite_ridden * (1 - fraction_collapsing_given_termite_ridden) = fraction_not_collapsing :=
sorry

end gotham_street_termite_ridden_not_collapsing_l2061_206131


namespace committee_count_8_choose_4_l2061_206184

theorem committee_count_8_choose_4 : (Nat.choose 8 4) = 70 :=
  by
  -- proof skipped
  sorry

end committee_count_8_choose_4_l2061_206184


namespace product_of_millions_l2061_206132

-- Define the conditions
def a := 5 * (10 : ℝ) ^ 6
def b := 8 * (10 : ℝ) ^ 6

-- State the proof problem
theorem product_of_millions : (a * b) = 40 * (10 : ℝ) ^ 12 := 
by
  sorry

end product_of_millions_l2061_206132


namespace primes_x_y_eq_l2061_206123

theorem primes_x_y_eq 
  {p q x y : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q)
  (hx : 0 < x) (hy : 0 < y)
  (hp_lt_x : x < p) (hq_lt_y : y < q)
  (h : (p : ℚ) / x + (q : ℚ) / y = (p * y + q * x) / (x * y)) :
  x = y :=
sorry

end primes_x_y_eq_l2061_206123


namespace original_pencils_count_l2061_206196

theorem original_pencils_count (total_pencils : ℕ) (added_pencils : ℕ) (original_pencils : ℕ) : total_pencils = original_pencils + added_pencils → original_pencils = 2 :=
by
  sorry

end original_pencils_count_l2061_206196


namespace janina_must_sell_21_pancakes_l2061_206163

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end janina_must_sell_21_pancakes_l2061_206163


namespace nanometers_to_scientific_notation_l2061_206147

   theorem nanometers_to_scientific_notation :
     (0.000000001 : Float) = 1 * 10 ^ (-9) :=
   by
     sorry
   
end nanometers_to_scientific_notation_l2061_206147


namespace trigonometric_inequality_l2061_206118

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  0 < (1 / (Real.sin x)^2) - (1 / x^2) ∧ (1 / (Real.sin x)^2) - (1 / x^2) < 1 := 
sorry

end trigonometric_inequality_l2061_206118


namespace puppy_cost_l2061_206199

variable (P : ℕ)  -- Cost of one puppy

theorem puppy_cost (P : ℕ) (kittens : ℕ) (cost_kitten : ℕ) (total_value : ℕ) :
  kittens = 4 → cost_kitten = 15 → total_value = 100 → 
  2 * P + kittens * cost_kitten = total_value → P = 20 :=
by sorry

end puppy_cost_l2061_206199


namespace union_of_sets_l2061_206166

noncomputable def A : Set ℕ := {1, 2, 4}
noncomputable def B : Set ℕ := {2, 4, 6}

theorem union_of_sets : A ∪ B = {1, 2, 4, 6} := 
by 
sorry

end union_of_sets_l2061_206166


namespace find_rs_l2061_206125

noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry
def cond1 := r > 0 ∧ s > 0
def cond2 := r^2 + s^2 = 1
def cond3 := r^4 + s^4 = (3 : ℝ) / 4

theorem find_rs (h1 : cond1) (h2 : cond2) (h3 : cond3) : r * s = Real.sqrt 2 / 4 :=
by sorry

end find_rs_l2061_206125


namespace petes_average_speed_l2061_206140

-- Definitions of the conditions
def map_distance : ℝ := 5 -- in inches
def driving_time : ℝ := 6.5 -- in hours
def map_scale : ℝ := 0.01282051282051282 -- in inches per mile

-- Theorem statement: If the conditions are given, then the average speed is 60 miles per hour
theorem petes_average_speed :
  (map_distance / map_scale) / driving_time = 60 :=
by
  -- The proof will go here
  sorry

end petes_average_speed_l2061_206140


namespace susan_arrives_before_sam_by_14_minutes_l2061_206187

theorem susan_arrives_before_sam_by_14_minutes (d : ℝ) (susan_speed sam_speed : ℝ) (h1 : d = 2) (h2 : susan_speed = 12) (h3 : sam_speed = 5) : 
  let susan_time := d / susan_speed
  let sam_time := d / sam_speed
  let susan_minutes := susan_time * 60
  let sam_minutes := sam_time * 60
  sam_minutes - susan_minutes = 14 := 
by
  sorry

end susan_arrives_before_sam_by_14_minutes_l2061_206187


namespace arithmetic_geometric_sequence_S6_l2061_206194

noncomputable def S_6 (a : Nat) (q : Nat) : Nat :=
  (q ^ 6 - 1) / (q - 1)

theorem arithmetic_geometric_sequence_S6 (a : Nat) (q : Nat) (h1 : a * q ^ 1 = 2) (h2 : a * q ^ 3 = 8) (hq : q > 0) : S_6 a q = 63 :=
by
  sorry

end arithmetic_geometric_sequence_S6_l2061_206194


namespace expression_value_l2061_206109

theorem expression_value (a b m n : ℚ) 
  (ha : a = -7/4) 
  (hb : b = -2/3) 
  (hmn : m + n = 0) : 
  4 * a / b + 3 * (m + n) = 21 / 2 :=
by 
  sorry

end expression_value_l2061_206109


namespace reduced_price_l2061_206119

theorem reduced_price (
  P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 9 = 900 / R - 900 / P)
  (h3 : P = 42.8571) :
  R = 30 :=
by {
  sorry
}

end reduced_price_l2061_206119


namespace Sherry_catches_train_within_5_minutes_l2061_206122

-- Defining the probabilities given in the conditions
def P_A : ℝ := 0.75  -- Probability of train arriving
def P_N : ℝ := 0.75  -- Probability of Sherry not noticing the train

-- Event that no train arrives combined with event that train arrives but not noticed
def P_not_catch_in_a_minute : ℝ := 1 - P_A + P_A * P_N

-- Generalizing to 5 minutes
def P_not_catch_in_5_minutes : ℝ := P_not_catch_in_a_minute ^ 5

-- Probability Sherry catches the train within 5 minutes
def P_C : ℝ := 1 - P_not_catch_in_5_minutes

theorem Sherry_catches_train_within_5_minutes : P_C = 1 - (13 / 16) ^ 5 := by
  sorry

end Sherry_catches_train_within_5_minutes_l2061_206122


namespace haley_extra_tickets_l2061_206117

theorem haley_extra_tickets (cost_per_ticket : ℤ) (tickets_bought_for_self_and_friends : ℤ) (total_spent : ℤ) 
    (h1 : cost_per_ticket = 4) (h2 : tickets_bought_for_self_and_friends = 3) (h3 : total_spent = 32) : 
    (total_spent / cost_per_ticket) - tickets_bought_for_self_and_friends = 5 :=
by
  sorry

end haley_extra_tickets_l2061_206117


namespace tangent_line_eq_l2061_206180

noncomputable def f (x : ℝ) : ℝ := x + Real.log x

theorem tangent_line_eq :
  ∃ (m b : ℝ), (m = (deriv f 1)) ∧ (b = (f 1 - m * 1)) ∧
   (∀ (x y : ℝ), y = m * (x - 1) + b ↔ y = 2 * x - 1) :=
by sorry

end tangent_line_eq_l2061_206180


namespace xiaodong_election_l2061_206175

theorem xiaodong_election (V : ℕ) (h1 : 0 < V) :
  let total_needed := (3 : ℚ) / 4 * V
  let votes_obtained := (5 : ℚ) / 6 * (2 : ℚ) / 3 * V
  let remaining_votes := V - (2 : ℚ) / 3 * V
  total_needed - votes_obtained = (7 : ℚ) / 12 * remaining_votes :=
by 
  sorry

end xiaodong_election_l2061_206175


namespace continuity_sum_l2061_206111

noncomputable def piecewise_function (x : ℝ) (a b c : ℝ) : ℝ :=
if h : x > 1 then a * (2 * x + 1) + 2
else if h' : -1 <= x && x <= 1 then b * x + 3
else 3 * x - c

theorem continuity_sum (a b c : ℝ) (h_cont1 : 3 * a = b + 1) (h_cont2 : c = 3 * a + 1) :
  a + c = 4 * a + 1 :=
by
  sorry

end continuity_sum_l2061_206111


namespace simplify_fraction_l2061_206170

theorem simplify_fraction (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := sorry

end simplify_fraction_l2061_206170
