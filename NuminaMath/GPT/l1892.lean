import Mathlib

namespace percentage_increased_is_correct_l1892_189265

-- Define the initial and final numbers
def initial_number : Nat := 150
def final_number : Nat := 210

-- Define the function to compute the percentage increase
def percentage_increase (initial final : Nat) : Float :=
  ((final - initial).toFloat / initial.toFloat) * 100.0

-- The theorem we need to prove
theorem percentage_increased_is_correct :
  percentage_increase initial_number final_number = 40 := 
by
  simp [percentage_increase, initial_number, final_number]
  sorry

end percentage_increased_is_correct_l1892_189265


namespace min_a_b_sum_l1892_189209

theorem min_a_b_sum (a b : ℕ) (x : ℕ → ℕ)
  (h0 : x 1 = a)
  (h1 : x 2 = b)
  (h2 : ∀ n, x (n+2) = x n + x (n+1))
  (h3 : ∃ n, x n = 1000) : a + b = 10 :=
sorry

end min_a_b_sum_l1892_189209


namespace solve_x_for_equation_l1892_189264

theorem solve_x_for_equation :
  ∃ (x : ℚ), 3 * x - 5 = abs (-20 + 6) ∧ x = 19 / 3 :=
by
  sorry

end solve_x_for_equation_l1892_189264


namespace f_value_at_5pi_over_6_l1892_189218

noncomputable def f (x ω : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 3))

theorem f_value_at_5pi_over_6
  (ω : ℝ) (ω_pos : ω > 0)
  (α β : ℝ)
  (h1 : f α ω = 2)
  (h2 : f β ω = 0)
  (h3 : Real.sqrt ((α - β)^2 + 4) = Real.sqrt (4 + (Real.pi^2 / 4))) :
  f (5 * Real.pi / 6) ω = -1 := 
sorry

end f_value_at_5pi_over_6_l1892_189218


namespace distinct_positive_integer_roots_l1892_189259

theorem distinct_positive_integer_roots (m a b : ℤ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = -m) (h5 : a * b = -m + 1) : m = -5 := 
by
  sorry

end distinct_positive_integer_roots_l1892_189259


namespace focus_of_parabola_l1892_189295

theorem focus_of_parabola (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) ∈ {p : ℝ × ℝ | ∃ x y, y = 4 * x^2 ∧ p = (0, 1 / (4 * (1 / y)))} :=
by
  sorry

end focus_of_parabola_l1892_189295


namespace average_marks_l1892_189280

theorem average_marks :
  let a1 := 76
  let a2 := 65
  let a3 := 82
  let a4 := 67
  let a5 := 75
  let n := 5
  let total_marks := a1 + a2 + a3 + a4 + a5
  let avg_marks := total_marks / n
  avg_marks = 73 :=
by
  sorry

end average_marks_l1892_189280


namespace study_time_l1892_189243

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time_l1892_189243


namespace number_of_true_propositions_l1892_189238

theorem number_of_true_propositions : 
  let original_p := ∀ (a : ℝ), a > -1 → a > -2
  let converse_p := ∀ (a : ℝ), a > -2 → a > -1
  let inverse_p := ∀ (a : ℝ), a ≤ -1 → a ≤ -2
  let contrapositive_p := ∀ (a : ℝ), a ≤ -2 → a ≤ -1
  (original_p ∧ contrapositive_p ∧ ¬converse_p ∧ ¬inverse_p) → (2 = 2) :=
by
  intros
  sorry

end number_of_true_propositions_l1892_189238


namespace find_height_of_larger_cuboid_l1892_189298

-- Define the larger cuboid dimensions
def Length_large : ℝ := 18
def Width_large : ℝ := 15
def Volume_large (Height_large : ℝ) : ℝ := Length_large * Width_large * Height_large

-- Define the smaller cuboid dimensions
def Length_small : ℝ := 5
def Width_small : ℝ := 6
def Height_small : ℝ := 3
def Volume_small : ℝ := Length_small * Width_small * Height_small

-- Define the total volume of 6 smaller cuboids
def Total_volume_small : ℝ := 6 * Volume_small

-- State the problem and the proof goal
theorem find_height_of_larger_cuboid : 
  ∃ H : ℝ, Volume_large H = Total_volume_small :=
by
  use 2
  sorry

end find_height_of_larger_cuboid_l1892_189298


namespace part1_part2_l1892_189263

open Real

def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

theorem part1 (m : ℝ) : (∃ x : ℝ, f x ≤ m) ↔ m ≥ -3 := 
sorry

theorem part2 : {x : ℝ | x^2 - 8 * x + 15 + f x ≤ 0} = {x : ℝ | 5 - sqrt 3 ≤ x ∧ x ≤ 6} :=
sorry

end part1_part2_l1892_189263


namespace find_a8_l1892_189299

variable {a : ℕ → ℝ} -- Assuming the sequence is real-valued for generality

-- Defining the necessary properties and conditions of the arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a 0 + n * (a 1 - a 0)

-- Given conditions as hypothesis
variable (h_seq : arithmetic_sequence a) 
variable (h_sum : a 3 + a 6 + a 10 + a 13 = 32)

-- The proof statement
theorem find_a8 : a 8 = 8 :=
by
  sorry -- The proof itself

end find_a8_l1892_189299


namespace zhang_san_not_losing_probability_l1892_189203

theorem zhang_san_not_losing_probability (p_win p_draw : ℚ) (h_win : p_win = 1 / 3) (h_draw : p_draw = 1 / 4) : 
  p_win + p_draw = 7 / 12 := by
  sorry

end zhang_san_not_losing_probability_l1892_189203


namespace tan_identity_l1892_189277

theorem tan_identity (A B : ℝ) (hA : A = 30) (hB : B = 30) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = (4 + 2 * Real.sqrt 3)/3 := by
  sorry

end tan_identity_l1892_189277


namespace haley_tv_total_hours_l1892_189227

theorem haley_tv_total_hours (h_sat : Nat) (h_sun : Nat) (H_sat : h_sat = 6) (H_sun : h_sun = 3) :
  h_sat + h_sun = 9 := by
  sorry

end haley_tv_total_hours_l1892_189227


namespace pens_multiple_91_l1892_189291

theorem pens_multiple_91 (S : ℕ) (P : ℕ) (total_pencils : ℕ) 
  (h1 : S = 91) (h2 : total_pencils = 910) (h3 : total_pencils % S = 0) :
  ∃ (x : ℕ), P = S * x :=
by 
  sorry

end pens_multiple_91_l1892_189291


namespace profit_percent_l1892_189269

-- Definitions for the given conditions
variables (P C : ℝ)
-- Condition given: selling at (2/3) of P results in a loss of 5%, i.e., (2/3) * P = 0.95 * C
def condition : Prop := (2 / 3) * P = 0.95 * C

-- Theorem statement: Given the condition, the profit percent when selling at price P is 42.5%
theorem profit_percent (h : condition P C) : ((P - C) / C) * 100 = 42.5 :=
sorry

end profit_percent_l1892_189269


namespace volume_ratio_of_spheres_l1892_189273

theorem volume_ratio_of_spheres
  (r1 r2 r3 : ℝ)
  (A1 A2 A3 : ℝ)
  (V1 V2 V3 : ℝ)
  (hA : A1 / A2 = 1 / 4 ∧ A2 / A3 = 4 / 9)
  (hSurfaceArea : A1 = 4 * π * r1^2 ∧ A2 = 4 * π * r2^2 ∧ A3 = 4 * π * r3^2)
  (hVolume : V1 = (4 / 3) * π * r1^3 ∧ V2 = (4 / 3) * π * r2^3 ∧ V3 = (4 / 3) * π * r3^3) :
  V1 / V2 = 1 / 8 ∧ V2 / V3 = 8 / 27 := by
  sorry

end volume_ratio_of_spheres_l1892_189273


namespace equal_if_fraction_is_positive_integer_l1892_189282

theorem equal_if_fraction_is_positive_integer
  (a b : ℕ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (K : ℝ := Real.sqrt ((a^2 + b^2:ℕ)/2))
  (A : ℝ := (a + b:ℕ)/2)
  (h_int_pos : ∃ (n : ℕ), n > 0 ∧ K / A = n) :
  a = b := sorry

end equal_if_fraction_is_positive_integer_l1892_189282


namespace total_length_of_river_is_80_l1892_189249

-- Definitions based on problem conditions
def straight_part_length := 20
def crooked_part_length := 3 * straight_part_length
def total_length_of_river := straight_part_length + crooked_part_length

-- Theorem stating that the total length of the river is 80 miles
theorem total_length_of_river_is_80 :
  total_length_of_river = 80 := by
    -- The proof is omitted
    sorry

end total_length_of_river_is_80_l1892_189249


namespace rectangle_not_equal_118_l1892_189204

theorem rectangle_not_equal_118 
  (a b : ℕ) (h₀ : a > 0) (h₁ : b > 0) (A : ℕ) (P : ℕ)
  (h₂ : A = a * b) (h₃ : P = 2 * (a + b)) :
  (a + 2) * (b + 2) - 2 ≠ 118 :=
sorry

end rectangle_not_equal_118_l1892_189204


namespace gambler_largest_amount_received_l1892_189297

def largest_amount_received_back (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : ℕ :=
  3000 - (30 * a + 100 * b)

theorem gambler_largest_amount_received (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : 
    largest_amount_received_back x y a b h1 h2 h3 = 2030 :=
by sorry

end gambler_largest_amount_received_l1892_189297


namespace determine_a_l1892_189261

theorem determine_a (a p q : ℚ) (h1 : p^2 = a) (h2 : 2 * p * q = 28) (h3 : q^2 = 9) : a = 196 / 9 :=
by
  sorry

end determine_a_l1892_189261


namespace factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l1892_189206

-- Problem 1
theorem factorize_2x2_minus_4x (x : ℝ) : 
  2 * x^2 - 4 * x = 2 * x * (x - 2) := 
by 
  sorry

-- Problem 2
theorem factorize_xy2_minus_2xy_plus_x (x y : ℝ) :
  x * y^2 - 2 * x * y + x = x * (y - 1)^2 :=
by 
  sorry

end factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l1892_189206


namespace trapezoid_area_l1892_189223

theorem trapezoid_area (A_outer A_inner : ℝ) (n : ℕ)
  (h_outer : A_outer = 36)
  (h_inner : A_inner = 4)
  (h_n : n = 4) :
  (A_outer - A_inner) / n = 8 := by
  sorry

end trapezoid_area_l1892_189223


namespace sum_of_minimums_is_zero_l1892_189279

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

-- Conditions: P(Q(x)) has zeros at -5, -3, -1, 1
lemma zeroes_PQ : 
  P.eval (Q.eval (-5)) = 0 ∧ 
  P.eval (Q.eval (-3)) = 0 ∧ 
  P.eval (Q.eval (-1)) = 0 ∧ 
  P.eval (Q.eval (1)) = 0 := 
  sorry

-- Conditions: Q(P(x)) has zeros at -7, -5, -1, 3
lemma zeroes_QP : 
  Q.eval (P.eval (-7)) = 0 ∧ 
  Q.eval (P.eval (-5)) = 0 ∧ 
  Q.eval (P.eval (-1)) = 0 ∧ 
  Q.eval (P.eval (3)) = 0 := 
  sorry

-- Definition to find the minimum value of a polynomial
noncomputable def min_value (P : Polynomial ℝ) : ℝ := sorry

-- Main theorem
theorem sum_of_minimums_is_zero :
  min_value P + min_value Q = 0 := 
  sorry

end sum_of_minimums_is_zero_l1892_189279


namespace correct_answer_l1892_189287

def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3*x - 2

theorem correct_answer : f (g 3) = 79 := by
  sorry

end correct_answer_l1892_189287


namespace find_xyz_l1892_189248

-- Let a, b, c, x, y, z be nonzero complex numbers
variables (a b c x y z : ℂ)
-- Given conditions
variables (h1 : a = (b + c) / (x - 2))
variables (h2 : b = (a + c) / (y - 2))
variables (h3 : c = (a + b) / (z - 2))
variables (h4 : x * y + x * z + y * z = 5)
variables (h5 : x + y + z = 3)

-- Prove that xyz = 5
theorem find_xyz : x * y * z = 5 :=
by
  sorry

end find_xyz_l1892_189248


namespace negation_proof_l1892_189240

-- Definitions based on conditions
def atMostTwoSolutions (solutions : ℕ) : Prop := solutions ≤ 2
def atLeastThreeSolutions (solutions : ℕ) : Prop := solutions ≥ 3

-- Statement of the theorem
theorem negation_proof (solutions : ℕ) : atMostTwoSolutions solutions ↔ ¬ atLeastThreeSolutions solutions :=
by
  sorry

end negation_proof_l1892_189240


namespace triangle_inscribed_in_semicircle_l1892_189216

variables {R : ℝ} (P Q R' : ℝ) (PR QR : ℝ)
variables (hR : 0 < R) (h_pq_diameter: P = -R ∧ Q = R)
variables (h_pr_square_qr_square : PR^2 + QR^2 = 4 * R^2)
variables (t := PR + QR)

theorem triangle_inscribed_in_semicircle (h_pos_pr : 0 < PR) (h_pos_qr : 0 < QR) : 
  t^2 ≤ 8 * R^2 :=
sorry

end triangle_inscribed_in_semicircle_l1892_189216


namespace curve_transformation_l1892_189202

-- Define the scaling transformation
def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (5 * x, 3 * y)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop :=
  2 * x' ^ 2 + 8 * y' ^ 2 = 1

-- Define the curve C's equation after scaling
def curve_C (x y : ℝ) : Prop :=
  50 * x ^ 2 + 72 * y ^ 2 = 1

-- Statement of the proof problem
theorem curve_transformation (x y : ℝ) (h : transformed_curve (5 * x) (3 * y)) : curve_C x y :=
by {
  -- The actual proof would be filled in here
  sorry
}

end curve_transformation_l1892_189202


namespace inscribed_circle_radius_l1892_189234

noncomputable def side1 := 13
noncomputable def side2 := 13
noncomputable def side3 := 10
noncomputable def s := (side1 + side2 + side3) / 2
noncomputable def area := Real.sqrt (s * (s - side1) * (s - side2) * (s - side3))
noncomputable def inradius := area / s

theorem inscribed_circle_radius :
  inradius = 10 / 3 :=
by
  sorry

end inscribed_circle_radius_l1892_189234


namespace angle_B_pi_div_3_triangle_perimeter_l1892_189285

-- Problem 1: Prove that B = π / 3 given the condition.
theorem angle_B_pi_div_3 (A B C : ℝ) (hTriangle : A + B + C = Real.pi) 
  (hCos : Real.cos B = Real.cos ((A + C) / 2)) : 
  B = Real.pi / 3 :=
sorry

-- Problem 2: Prove the perimeter given the conditions.
theorem triangle_perimeter (a b c : ℝ) (m : ℝ) 
  (altitude : ℝ) 
  (hSides : 8 * a = 3 * c) 
  (hAltitude : altitude = 12 * Real.sqrt 3 / 7) 
  (hAngleB : ∃ B, B = Real.pi / 3) :
  a + b + c = 18 := 
sorry

end angle_B_pi_div_3_triangle_perimeter_l1892_189285


namespace Alyssa_number_of_quarters_l1892_189278

def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25
def num_pennies : ℕ := 7
def total_money : ℝ := 3.07

def num_quarters (q : ℕ) : Prop :=
  total_money - (num_pennies * value_penny) = q * value_quarter

theorem Alyssa_number_of_quarters : ∃ q : ℕ, num_quarters q ∧ q = 12 :=
by
  sorry

end Alyssa_number_of_quarters_l1892_189278


namespace boxes_contain_same_number_of_apples_l1892_189232

theorem boxes_contain_same_number_of_apples (total_apples boxes : ℕ) (h1 : total_apples = 49) (h2 : boxes = 7) : 
  total_apples / boxes = 7 :=
by
  sorry

end boxes_contain_same_number_of_apples_l1892_189232


namespace find_integer_pairs_l1892_189276

-- Define the plane and lines properties
def horizontal_lines (h : ℕ) : Prop := h > 0
def non_horizontal_lines (s : ℕ) : Prop := s > 0
def non_parallel (s : ℕ) : Prop := s > 0
def no_three_intersect (total_lines : ℕ) : Prop := total_lines > 0

-- Function to calculate regions from the given formula
def calculate_regions (h s : ℕ) : ℕ :=
  h * (s + 1) + 1 + (s * (s + 1)) / 2

-- Prove that the given (h, s) pairs divide the plane into 1992 regions
theorem find_integer_pairs :
  (horizontal_lines 995 ∧ non_horizontal_lines 1 ∧ non_parallel 1 ∧ no_three_intersect (995 + 1) ∧ calculate_regions 995 1 = 1992)
  ∨ (horizontal_lines 176 ∧ non_horizontal_lines 10 ∧ non_parallel 10 ∧ no_three_intersect (176 + 10) ∧ calculate_regions 176 10 = 1992)
  ∨ (horizontal_lines 80 ∧ non_horizontal_lines 21 ∧ non_parallel 21 ∧ no_three_intersect (80 + 21) ∧ calculate_regions 80 21 = 1992) :=
by
  -- Include individual cases to verify correctness of regions calculation
  sorry

end find_integer_pairs_l1892_189276


namespace problem1_problem2_l1892_189221

-- Define the quadratic equation and condition for real roots
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Problem 1
theorem problem1 (m : ℝ) : ((m - 2) * (m - 2) * (m - 2) + 2 * 2 * (2 - m) * 2 * (-1) ≥ 0) → (m ≤ 3 ∧ m ≠ 2) := sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (∀ x, (x = 1 ∨ x = 2) → (m - 2) * x^2 + 2 * x + 1 = 0) → (-1 ≤ m ∧ m < (3 / 4)) := 
sorry

end problem1_problem2_l1892_189221


namespace greatest_value_x_l1892_189257

theorem greatest_value_x (x : ℕ) (h : lcm (lcm x 12) 18 = 108) : x ≤ 108 := sorry

end greatest_value_x_l1892_189257


namespace sequence_term_2023_l1892_189228

theorem sequence_term_2023 (a : ℕ → ℚ) (h₁ : a 1 = 2) 
  (h₂ : ∀ n, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1 / 2 := 
sorry

end sequence_term_2023_l1892_189228


namespace fraction_of_menu_vegan_soy_free_l1892_189235

def num_vegan_dishes : Nat := 6
def fraction_menu_vegan : ℚ := 1 / 4
def num_vegan_dishes_with_soy : Nat := 4

def num_vegan_soy_free_dishes : Nat := num_vegan_dishes - num_vegan_dishes_with_soy
def fraction_vegan_soy_free : ℚ := num_vegan_soy_free_dishes / num_vegan_dishes
def fraction_menu_vegan_soy_free : ℚ := fraction_vegan_soy_free * fraction_menu_vegan

theorem fraction_of_menu_vegan_soy_free :
  fraction_menu_vegan_soy_free = 1 / 12 := by
  sorry

end fraction_of_menu_vegan_soy_free_l1892_189235


namespace farmer_plow_l1892_189224

theorem farmer_plow (P : ℕ) (M : ℕ) (H1 : M = 12) (H2 : 8 * P + M * (8 - (55 / P)) = 30) (H3 : 55 % P = 0) : P = 10 :=
by
  sorry

end farmer_plow_l1892_189224


namespace gold_coins_l1892_189283

theorem gold_coins (c n : ℕ) 
  (h₁ : n = 8 * (c - 1))
  (h₂ : n = 5 * c + 4) :
  n = 24 :=
by
  sorry

end gold_coins_l1892_189283


namespace krakozyabr_count_l1892_189255

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l1892_189255


namespace percentage_increase_l1892_189296

theorem percentage_increase (original_price new_price : ℝ) (h₀ : original_price = 300) (h₁ : new_price = 420) :
  ((new_price - original_price) / original_price) * 100 = 40 :=
by
  -- Insert the proof here
  sorry

end percentage_increase_l1892_189296


namespace cauchy_schwarz_equivalent_iag_l1892_189220

theorem cauchy_schwarz_equivalent_iag (a b c d : ℝ) :
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (Real.sqrt x * Real.sqrt y) ≤ (x + y) / 2) ↔
  ((a * c + b * d) ^ 2 ≤ (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2)) := by
  sorry

end cauchy_schwarz_equivalent_iag_l1892_189220


namespace sum_of_perpendiculars_l1892_189250

-- define the points on the rectangle
variables {A B C D P S R Q F : Type}

-- define rectangle ABCD and points P, S, R, Q, F
def is_rectangle (A B C D : Type) : Prop := sorry -- conditions for ABCD to be a rectangle
def point_on_segment (P A B: Type) : Prop := sorry -- P is a point on segment AB
def perpendicular (X Y Z : Type) : Prop := sorry -- definition for perpendicular between two segments
def length (X Y : Type) : ℝ := sorry -- definition for the length of a segment

-- Given conditions
axiom rect : is_rectangle A B C D
axiom p_on_ab : point_on_segment P A B
axiom ps_perp_bd : perpendicular P S D
axiom pr_perp_ac : perpendicular P R C
axiom af_perp_bd : perpendicular A F D
axiom pq_perp_af : perpendicular P Q F

-- Prove that PR + PS = AF
theorem sum_of_perpendiculars :
  length P R + length P S = length A F :=
sorry

end sum_of_perpendiculars_l1892_189250


namespace smallest_possible_fourth_number_l1892_189268

theorem smallest_possible_fourth_number 
  (a b : ℕ) 
  (h1 : 21 + 34 + 65 = 120)
  (h2 : 1 * (21 + 34 + 65 + 10 * a + b) = 4 * (2 + 1 + 3 + 4 + 6 + 5 + a + b)) :
  10 * a + b = 12 := 
sorry

end smallest_possible_fourth_number_l1892_189268


namespace angle_in_first_quadrant_l1892_189211

-- Define the condition and equivalence proof problem in Lean 4
theorem angle_in_first_quadrant (deg : ℤ) (h1 : deg = 721) : (deg % 360) > 0 := 
by 
  have : deg % 360 = 1 := sorry
  exact sorry

end angle_in_first_quadrant_l1892_189211


namespace red_beads_cost_l1892_189207

theorem red_beads_cost (R : ℝ) (H : 4 * R + 4 * 2 = 10 * 1.72) : R = 2.30 :=
by
  sorry

end red_beads_cost_l1892_189207


namespace probability_both_counterfeit_given_one_counterfeit_l1892_189275

-- Conditions
def total_bills := 20
def counterfeit_bills := 5
def selected_bills := 2
def at_least_one_counterfeit := true

-- Definition of events
def eventA := "both selected bills are counterfeit"
def eventB := "at least one of the selected bills is counterfeit"

-- The theorem to prove
theorem probability_both_counterfeit_given_one_counterfeit : 
  at_least_one_counterfeit →
  ( (counterfeit_bills * (counterfeit_bills - 1)) / (total_bills * (total_bills - 1)) ) / 
    ( (counterfeit_bills * (counterfeit_bills - 1) + counterfeit_bills * (total_bills - counterfeit_bills)) / (total_bills * (total_bills - 1)) ) = 2/17 :=
by
  sorry

end probability_both_counterfeit_given_one_counterfeit_l1892_189275


namespace first_number_in_proportion_l1892_189253

variable (x y : ℝ)

theorem first_number_in_proportion
  (h1 : x = 0.9)
  (h2 : y / x = 5 / 6) : 
  y = 0.75 := 
  by 
    sorry

end first_number_in_proportion_l1892_189253


namespace mean_of_xyz_l1892_189251

theorem mean_of_xyz (mean7 : ℕ) (mean10 : ℕ) (x y z : ℕ) (h1 : mean7 = 40) (h2 : mean10 = 50) : (x + y + z) / 3 = 220 / 3 :=
by
  have sum7 := 7 * mean7
  have sum10 := 10 * mean10
  have sum_xyz := sum10 - sum7
  have mean_xyz := sum_xyz / 3
  sorry

end mean_of_xyz_l1892_189251


namespace total_charge_rush_hour_trip_l1892_189266

def initial_fee : ℝ := 2.35
def non_rush_hour_cost_per_two_fifths_mile : ℝ := 0.35
def rush_hour_cost_increase_percentage : ℝ := 0.20
def traffic_delay_cost_per_mile : ℝ := 1.50
def distance_travelled : ℝ := 3.6

theorem total_charge_rush_hour_trip (initial_fee : ℝ) 
  (non_rush_hour_cost_per_two_fifths_mile : ℝ) 
  (rush_hour_cost_increase_percentage : ℝ)
  (traffic_delay_cost_per_mile : ℝ)
  (distance_travelled : ℝ) : 
  initial_fee = 2.35 → 
  non_rush_hour_cost_per_two_fifths_mile = 0.35 →
  rush_hour_cost_increase_percentage = 0.20 →
  traffic_delay_cost_per_mile = 1.50 →
  distance_travelled = 3.6 →
  (initial_fee + ((5/2) * (non_rush_hour_cost_per_two_fifths_mile * (1 + rush_hour_cost_increase_percentage))) * distance_travelled + (traffic_delay_cost_per_mile * distance_travelled)) = 11.53 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_charge_rush_hour_trip_l1892_189266


namespace factor_polynomial_l1892_189292

-- Define the necessary polynomials
def p (x : ℝ) : ℝ := x^2 + 4*x + 3
def q (x : ℝ) : ℝ := x^2 + 8*x + 15
def r (x : ℝ) : ℝ := x^2 + 6*x - 8

-- State the main theorem
theorem factor_polynomial (x : ℝ) : 
  (p x * q x) + r x = (x^2 + 7*x + 1) * (x^2 + 4*x + 7) :=
by
  sorry

end factor_polynomial_l1892_189292


namespace find_x_l1892_189254

open Real

theorem find_x 
  (x y : ℝ) 
  (hx_pos : 0 < x)
  (hy_pos : 0 < y) 
  (h_eq : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) 
  : x = 7 := 
sorry

end find_x_l1892_189254


namespace kyle_and_miles_total_marble_count_l1892_189226

noncomputable def kyle_marble_count (F : ℕ) (K : ℕ) : Prop :=
  F = 4 * K

noncomputable def miles_marble_count (F : ℕ) (M : ℕ) : Prop :=
  F = 9 * M

theorem kyle_and_miles_total_marble_count :
  ∀ (F K M : ℕ), F = 36 → kyle_marble_count F K → miles_marble_count F M → K + M = 13 :=
by
  intros F K M hF hK hM
  sorry

end kyle_and_miles_total_marble_count_l1892_189226


namespace find_b_when_a_is_negative12_l1892_189231

theorem find_b_when_a_is_negative12 (a b : ℝ) (h1 : a + b = 60) (h2 : a = 3 * b) (h3 : ∃ k, a * b = k) : b = -56.25 :=
sorry

end find_b_when_a_is_negative12_l1892_189231


namespace problem_1_problem_2_l1892_189284

variable (a : ℝ) (x : ℝ)

theorem problem_1 (h : a ≠ 1) : (a^2 / (a - 1)) - (a / (a - 1)) = a := 
sorry

theorem problem_2 (h : x ≠ -1) : (x^2 / (x + 1)) - x + 1 = 1 / (x + 1) := 
sorry

end problem_1_problem_2_l1892_189284


namespace largest_possible_b_l1892_189271

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c ≤ b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l1892_189271


namespace sum_of_a_b_l1892_189290

theorem sum_of_a_b (a b : ℝ) (h1 : ∀ x : ℝ, (a * (b * x + a) + b = x))
  (h2 : ∀ y : ℝ, (b * (a * y + b) + a = y)) : a + b = -2 := 
sorry

end sum_of_a_b_l1892_189290


namespace two_digit_even_multiple_of_7_l1892_189236

def all_digits_product_square (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  (d1 * d2) > 0 ∧ ∃ k, d1 * d2 = k * k

theorem two_digit_even_multiple_of_7 (n : ℕ) :
  10 ≤ n ∧ n < 100 ∧ n % 2 = 0 ∧ n % 7 = 0 ∧ all_digits_product_square n ↔ n = 14 ∨ n = 28 ∨ n = 70 :=
by sorry

end two_digit_even_multiple_of_7_l1892_189236


namespace intersection_eq_l1892_189289

def setA : Set ℝ := { x | abs (x - 3) < 2 }
def setB : Set ℝ := { x | (x - 4) / x ≥ 0 }

theorem intersection_eq : setA ∩ setB = { x | 4 ≤ x ∧ x < 5 } :=
by 
  sorry

end intersection_eq_l1892_189289


namespace seahawks_touchdowns_l1892_189272

theorem seahawks_touchdowns (total_points : ℕ) (points_per_touchdown : ℕ) (points_per_field_goal : ℕ) (field_goals : ℕ) (touchdowns : ℕ) :
  total_points = 37 →
  points_per_touchdown = 7 →
  points_per_field_goal = 3 →
  field_goals = 3 →
  total_points = (touchdowns * points_per_touchdown) + (field_goals * points_per_field_goal) →
  touchdowns = 4 :=
by
  intros h_total_points h_points_per_touchdown h_points_per_field_goal h_field_goals h_equation
  sorry

end seahawks_touchdowns_l1892_189272


namespace brick_height_calc_l1892_189258

theorem brick_height_calc 
  (length_wall : ℝ) (height_wall : ℝ) (width_wall : ℝ) 
  (num_bricks : ℕ) 
  (length_brick : ℝ) (width_brick : ℝ) 
  (H : ℝ) 
  (volume_wall : ℝ) 
  (volume_brick : ℝ)
  (condition1 : length_wall = 800) 
  (condition2 : height_wall = 600) 
  (condition3 : width_wall = 22.5)
  (condition4 : num_bricks = 3200) 
  (condition5 : length_brick = 50) 
  (condition6 : width_brick = 11.25) 
  (condition7 : volume_wall = length_wall * height_wall * width_wall) 
  (condition8 : volume_brick = length_brick * width_brick * H) 
  (condition9 : num_bricks * volume_brick = volume_wall) 
  : H = 6 := 
by
  sorry

end brick_height_calc_l1892_189258


namespace find_g_l1892_189214

-- Given conditions
def line_equation (x y : ℝ) : Prop := y = 2 * x - 10
def parameterization (g : ℝ → ℝ) (t : ℝ) : Prop := 20 * t - 8 = 2 * g t - 10

-- Statement to prove
theorem find_g (g : ℝ → ℝ) (t : ℝ) :
  (∀ x y, line_equation x y → parameterization g t) →
  g t = 10 * t + 1 :=
sorry

end find_g_l1892_189214


namespace expression_bounds_l1892_189230

noncomputable def expression (p q r s : ℝ) : ℝ :=
  Real.sqrt (p^2 + (2 - q)^2) + Real.sqrt (q^2 + (2 - r)^2) +
  Real.sqrt (r^2 + (2 - s)^2) + Real.sqrt (s^2 + (2 - p)^2)

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2)
  (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) : 
  4 * Real.sqrt 2 ≤ expression p q r s ∧ expression p q r s ≤ 8 :=
by
  sorry

end expression_bounds_l1892_189230


namespace profit_percentage_is_correct_l1892_189213

noncomputable def cost_price (SP : ℝ) : ℝ := 0.81 * SP

noncomputable def profit (SP CP : ℝ) : ℝ := SP - CP

noncomputable def profit_percentage (profit CP : ℝ) : ℝ := (profit / CP) * 100

theorem profit_percentage_is_correct (SP : ℝ) (h : SP = 100) :
  profit_percentage (profit SP (cost_price SP)) (cost_price SP) = 23.46 :=
by
  sorry

end profit_percentage_is_correct_l1892_189213


namespace number_of_birds_l1892_189260

-- Conditions
def geese : ℕ := 58
def ducks : ℕ := 37

-- Proof problem statement
theorem number_of_birds : geese + ducks = 95 :=
by
  -- The actual proof is to be provided
  sorry

end number_of_birds_l1892_189260


namespace min_trials_correct_l1892_189210

noncomputable def minimum_trials (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) : ℕ :=
  Nat.floor ((Real.log (1 - α)) / (Real.log (1 - p))) + 1

-- The theorem to prove the correctness of minimum_trials
theorem min_trials_correct (α p : ℝ) (hα : 0 < α ∧ α < 1) (hp : 0 < p ∧ p < 1) :
  ∃ n : ℕ, minimum_trials α p hα hp = n ∧ (1 - (1 - p)^n ≥ α) :=
by
  sorry

end min_trials_correct_l1892_189210


namespace paint_coverage_l1892_189270

-- Define the conditions
def cost_per_gallon : ℝ := 45
def total_area : ℝ := 1600
def number_of_coats : ℝ := 2
def total_contribution : ℝ := 180 + 180

-- Define the target statement to prove
theorem paint_coverage (H : total_contribution = 360) : 
  let cost_per_gallon := 45 
  let number_of_gallons := total_contribution / cost_per_gallon
  let total_coverage := total_area * number_of_coats
  let coverage_per_gallon := total_coverage / number_of_gallons
  coverage_per_gallon = 400 :=
by
  sorry

end paint_coverage_l1892_189270


namespace prove_a_range_l1892_189252

noncomputable def f (x : ℝ) : ℝ := 1 / (2 ^ x + 2)

theorem prove_a_range (a : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x + f (a - 2 * x) ≤ 1 / 2) → 5 ≤ a :=
by
  sorry

end prove_a_range_l1892_189252


namespace regular_polygon_sides_l1892_189233

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l1892_189233


namespace price_of_other_stock_l1892_189239

theorem price_of_other_stock (total_shares : ℕ) (total_spent : ℝ) (share_1_quantity : ℕ) (share_1_price : ℝ) :
  total_shares = 450 ∧ total_spent = 1950 ∧ share_1_quantity = 400 ∧ share_1_price = 3 →
  (750 / 50 = 15) :=
by sorry

end price_of_other_stock_l1892_189239


namespace hyperbola_range_k_l1892_189247

theorem hyperbola_range_k (k : ℝ) : 
  (1 < k ∧ k < 3) ↔ (∃ x y : ℝ, (3 - k > 0) ∧ (k - 1 > 0) ∧ (x * x) / (3 - k) - (y * y) / (k - 1) = 1) :=
by {
  sorry
}

end hyperbola_range_k_l1892_189247


namespace relationship_of_y_l1892_189225

theorem relationship_of_y (k y1 y2 y3 : ℝ)
  (hk : k < 0)
  (hy1 : y1 = k / -2)
  (hy2 : y2 = k / 1)
  (hy3 : y3 = k / 2) :
  y2 < y3 ∧ y3 < y1 := by
  -- Proof omitted
  sorry

end relationship_of_y_l1892_189225


namespace max_distance_with_optimal_swapping_l1892_189288

-- Define the conditions
def front_tire_lifetime : ℕ := 24000
def rear_tire_lifetime : ℕ := 36000

-- Prove that the maximum distance the car can travel given optimal tire swapping is 48,000 km
theorem max_distance_with_optimal_swapping : 
    ∃ x : ℕ, x < 24000 ∧ x < 36000 ∧ (x + min (24000 - x) (36000 - x) = 48000) :=
by {
  sorry
}

end max_distance_with_optimal_swapping_l1892_189288


namespace arithmetic_sequence_sum_l1892_189293

theorem arithmetic_sequence_sum :
  ∃ (a l d n : ℕ), a = 71 ∧ l = 109 ∧ d = 2 ∧ n = ((l - a) / d) + 1 ∧ 
    (3 * (n * (a + l) / 2) = 5400) := sorry

end arithmetic_sequence_sum_l1892_189293


namespace largest_sum_is_three_fourths_l1892_189208

-- Definitions of sums
def sum1 := (1 / 4) + (1 / 2)
def sum2 := (1 / 4) + (1 / 9)
def sum3 := (1 / 4) + (1 / 3)
def sum4 := (1 / 4) + (1 / 10)
def sum5 := (1 / 4) + (1 / 6)

-- The theorem stating that sum1 is the maximum of the sums
theorem largest_sum_is_three_fourths : max (max (max (max sum1 sum2) sum3) sum4) sum5 = 3 / 4 := 
sorry

end largest_sum_is_three_fourths_l1892_189208


namespace can_place_circles_l1892_189237

theorem can_place_circles (r: ℝ) (h: r = 2008) :
  ∃ (n: ℕ), (n > 4016) ∧ ((n: ℝ) / 2 > r) :=
by 
  sorry

end can_place_circles_l1892_189237


namespace quadratic_has_minimum_l1892_189200

theorem quadratic_has_minimum (a b : ℝ) (h : a > b^2) :
  ∃ (c : ℝ), c = (4 * b^2 / a) - 3 ∧ (∃ x : ℝ, a * x ^ 2 + 2 * b * x + c < 0) :=
by sorry

end quadratic_has_minimum_l1892_189200


namespace sum_of_consecutive_negative_integers_with_product_3080_l1892_189229

theorem sum_of_consecutive_negative_integers_with_product_3080 :
  ∃ (n : ℤ), n < 0 ∧ (n * (n + 1) = 3080) ∧ (n + (n + 1) = -111) :=
sorry

end sum_of_consecutive_negative_integers_with_product_3080_l1892_189229


namespace initial_number_correct_l1892_189256

def initial_number_problem : Prop :=
  ∃ (x : ℝ), x + 3889 - 47.80600000000004 = 3854.002 ∧
            x = 12.808000000000158

theorem initial_number_correct : initial_number_problem :=
by
  -- proof goes here
  sorry

end initial_number_correct_l1892_189256


namespace solve_for_x_l1892_189244

theorem solve_for_x (x : ℝ) (y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3/8 :=
by
  -- The proof will go here
  sorry

end solve_for_x_l1892_189244


namespace abc_positive_l1892_189241

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  -- Proof goes here
  sorry

end abc_positive_l1892_189241


namespace number_of_toys_bought_l1892_189294

def toy_cost (T : ℕ) : ℕ := 10 * T
def card_cost : ℕ := 2 * 5
def shirt_cost : ℕ := 5 * 6
def total_cost (T : ℕ) : ℕ := toy_cost T + card_cost + shirt_cost

theorem number_of_toys_bought (T : ℕ) : total_cost T = 70 → T = 3 :=
by
  intro h
  sorry

end number_of_toys_bought_l1892_189294


namespace sum_of_a_equals_five_l1892_189215

theorem sum_of_a_equals_five
  (f : ℕ → ℕ → ℕ)  -- Represents the function f defined by Table 1
  (a : ℕ → ℕ)  -- Represents the occurrences a₀, a₁, ..., a₄
  (h1 : a 0 + a 1 + a 2 + a 3 + a 4 = 5)  -- Condition 1
  (h2 : 0 * a 0 + 1 * a 1 + 2 * a 2 + 3 * a 3 + 4 * a 4 = 5)  -- Condition 2
  : a 0 + a 1 + a 2 + a 3 = 5 :=
sorry

end sum_of_a_equals_five_l1892_189215


namespace arccos_cos_11_eq_l1892_189217

theorem arccos_cos_11_eq: Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end arccos_cos_11_eq_l1892_189217


namespace algae_difference_l1892_189222

-- Define the original number of algae plants.
def original_algae := 809

-- Define the current number of algae plants.
def current_algae := 3263

-- Statement to prove: The difference between the current number of algae plants and the original number of algae plants is 2454.
theorem algae_difference : current_algae - original_algae = 2454 := by
  sorry

end algae_difference_l1892_189222


namespace area_of_ABC_l1892_189245

def point : Type := ℝ × ℝ

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_ABC : area_of_triangle (0, 0) (1, 0) (0, 1) = 0.5 :=
by
  sorry

end area_of_ABC_l1892_189245


namespace students_just_passed_l1892_189286

theorem students_just_passed (total_students : ℕ) (first_division : ℕ) (second_division : ℕ) (just_passed : ℕ)
  (h1 : total_students = 300)
  (h2 : first_division = 26 * total_students / 100)
  (h3 : second_division = 54 * total_students / 100)
  (h4 : just_passed = total_students - (first_division + second_division)) :
  just_passed = 60 :=
sorry

end students_just_passed_l1892_189286


namespace ratio_proof_l1892_189281

noncomputable def total_capacity : ℝ := 10 -- million gallons
noncomputable def amount_end_month : ℝ := 6 -- million gallons
noncomputable def normal_level : ℝ := total_capacity - 5 -- million gallons

theorem ratio_proof (h1 : amount_end_month = 0.6 * total_capacity)
                    (h2 : normal_level = total_capacity - 5) :
  (amount_end_month / normal_level) = 1.2 :=
by sorry

end ratio_proof_l1892_189281


namespace jan_uses_24_gallons_for_plates_and_clothes_l1892_189246

theorem jan_uses_24_gallons_for_plates_and_clothes :
  (65 - (2 * 7 + (2 * 7 - 11))) / 2 = 24 :=
by sorry

end jan_uses_24_gallons_for_plates_and_clothes_l1892_189246


namespace ratio_37m48s_2h13m15s_l1892_189219

-- Define the total seconds for 37 minutes and 48 seconds
def t1 := 37 * 60 + 48

-- Define the total seconds for 2 hours, 13 minutes, and 15 seconds
def t2 := 2 * 3600 + 13 * 60 + 15

-- Prove the ratio t1 / t2 = 2268 / 7995
theorem ratio_37m48s_2h13m15s : t1 / t2 = 2268 / 7995 := 
by sorry

end ratio_37m48s_2h13m15s_l1892_189219


namespace selection_options_l1892_189262

theorem selection_options (group1 : Fin 5) (group2 : Fin 4) : (group1.1 + group2.1 + 1 = 9) :=
sorry

end selection_options_l1892_189262


namespace solve_for_a_l1892_189212

theorem solve_for_a (a : ℝ) (h : 2 * a + (1 - 4 * a) = 0) : a = 1 / 2 :=
sorry

end solve_for_a_l1892_189212


namespace find_b_l1892_189267

noncomputable def a (c : ℚ) : ℚ := 10 * c - 10
noncomputable def b (c : ℚ) : ℚ := 10 * c + 10
noncomputable def c_val := (200 : ℚ) / 21

theorem find_b : 
  let a := a c_val
  let b := b c_val
  let c := c_val
  a + b + c = 200 ∧ 
  a + 10 = b - 10 ∧ 
  a + 10 = 10 * c → 
  b = 2210 / 21 :=
by
  intros
  sorry

end find_b_l1892_189267


namespace arithmetic_sequence_5th_term_l1892_189274

theorem arithmetic_sequence_5th_term :
  let a1 := 3
  let d := 4
  a1 + 4 * (5 - 1) = 19 :=
by
  sorry

end arithmetic_sequence_5th_term_l1892_189274


namespace find_larger_number_l1892_189205

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 := 
sorry

end find_larger_number_l1892_189205


namespace chemistry_marks_l1892_189201

-- Definitions based on given conditions
def total_marks (P C M : ℕ) : Prop := P + C + M = 210
def avg_physics_math (P M : ℕ) : Prop := (P + M) / 2 = 90
def physics_marks (P : ℕ) : Prop := P = 110
def avg_physics_other_subject (P C : ℕ) : Prop := (P + C) / 2 = 70

-- The proof problem statement
theorem chemistry_marks {P C M : ℕ} (h1 : total_marks P C M) (h2 : avg_physics_math P M) (h3 : physics_marks P) : C = 30 ∧ avg_physics_other_subject P C :=
by 
  -- Proof goes here
  sorry

end chemistry_marks_l1892_189201


namespace total_number_of_squares_l1892_189242

variable (x y : ℕ) -- Variables for the number of 10 cm and 20 cm squares

theorem total_number_of_squares
  (h1 : 100 * x + 400 * y = 2500) -- Condition for area
  (h2 : 40 * x + 80 * y = 280)    -- Condition for cutting length
  : (x + y = 16) :=
sorry

end total_number_of_squares_l1892_189242
