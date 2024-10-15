import Mathlib

namespace NUMINAMATH_GPT_clothes_washer_final_price_l676_67693

theorem clothes_washer_final_price
  (P : ℝ) (d1 d2 d3 : ℝ)
  (hP : P = 500)
  (hd1 : d1 = 0.10)
  (hd2 : d2 = 0.20)
  (hd3 : d3 = 0.05) :
  (P * (1 - d1) * (1 - d2) * (1 - d3)) / P = 0.684 :=
by
  sorry

end NUMINAMATH_GPT_clothes_washer_final_price_l676_67693


namespace NUMINAMATH_GPT_even_k_l676_67671

theorem even_k :
  ∀ (a b n k : ℕ),
  1 ≤ a → 1 ≤ b → 0 < n →
  2^n - 1 = a * b →
  (a * b + a - b - 1) % 2^k = 0 →
  (a * b + a - b - 1) % 2^(k+1) ≠ 0 →
  Even k :=
by
  intros a b n k ha hb hn h1 h2 h3
  sorry

end NUMINAMATH_GPT_even_k_l676_67671


namespace NUMINAMATH_GPT_find_root_and_m_l676_67681

theorem find_root_and_m (x₁ m : ℝ) (h₁ : -2 * x₁ = 2) (h₂ : x^2 + m * x + 2 = 0) : x₁ = -1 ∧ m = 3 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_root_and_m_l676_67681


namespace NUMINAMATH_GPT_hose_Z_fill_time_l676_67680

theorem hose_Z_fill_time (P X Y Z : ℝ) (h1 : X + Y = P / 3) (h2 : Y = P / 9) (h3 : X + Z = P / 4) (h4 : X + Y + Z = P / 2.5) : Z = P / 15 :=
sorry

end NUMINAMATH_GPT_hose_Z_fill_time_l676_67680


namespace NUMINAMATH_GPT_factor_polynomial_l676_67630

-- Statement of the proof problem
theorem factor_polynomial (x y z : ℝ) :
    x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 =
    (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l676_67630


namespace NUMINAMATH_GPT_transform_eq_l676_67619

theorem transform_eq (m n x y : ℕ) (h1 : m + x = n + y) (h2 : x = y) : m = n :=
sorry

end NUMINAMATH_GPT_transform_eq_l676_67619


namespace NUMINAMATH_GPT_min_value_y_l676_67604

theorem min_value_y : ∃ x : ℝ, (y = 2 * x^2 + 8 * x + 18) ∧ (∀ x : ℝ, y ≥ 10) :=
by
  sorry

end NUMINAMATH_GPT_min_value_y_l676_67604


namespace NUMINAMATH_GPT_triangle_construction_possible_l676_67622

-- Define the entities involved
variables {α β : ℝ} {a c : ℝ}

-- State the theorem
theorem triangle_construction_possible (a c : ℝ) (h : α = 2 * β) : a > (2 / 3) * c :=
sorry

end NUMINAMATH_GPT_triangle_construction_possible_l676_67622


namespace NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l676_67645

-- Define the equation of the curve
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Prove that when k=2, the curve is a circle
theorem option_A (x y : ℝ) : curve 2 x y ↔ x^2 + y^2 = 3 :=
by
  sorry

-- Prove the necessary and sufficient condition for the curve to be an ellipse
theorem option_B (k : ℝ) : (-1 < k ∧ k < 5) ↔ ∃ x y, curve k x y ∧ (k ≠ 2) :=
by
  sorry

-- Prove the condition for the curve to be a hyperbola with foci on the y-axis
theorem option_C (k : ℝ) : k < -1 ↔ ∃ x y, curve k x y ∧ (k < -1 ∧ k < 5) :=
by
  sorry

-- Prove that there does not exist a real number k such that the curve is a parabola
theorem option_D : ¬ (∃ k x y, curve k x y ∧ ∃ a b, x = a ∧ y = b) :=
by
  sorry

end NUMINAMATH_GPT_option_A_option_B_option_C_option_D_l676_67645


namespace NUMINAMATH_GPT_surface_area_rectangular_solid_l676_67672

def length := 5
def width := 4
def depth := 1

def surface_area (l w d : ℕ) := 2 * (l * w) + 2 * (l * d) + 2 * (w * d)

theorem surface_area_rectangular_solid : surface_area length width depth = 58 := 
by 
sorry

end NUMINAMATH_GPT_surface_area_rectangular_solid_l676_67672


namespace NUMINAMATH_GPT_shark_fin_falcata_area_is_correct_l676_67607

noncomputable def radius_large : ℝ := 3
noncomputable def center_large : ℝ × ℝ := (0, 0)

noncomputable def radius_small : ℝ := 3 / 2
noncomputable def center_small : ℝ × ℝ := (0, 3 / 2)

noncomputable def area_large_quarter_circle : ℝ := (1 / 4) * Real.pi * (radius_large ^ 2)
noncomputable def area_small_semicircle : ℝ := (1 / 2) * Real.pi * (radius_small ^ 2)

noncomputable def shark_fin_falcata_area (area_large_quarter_circle area_small_semicircle : ℝ) : ℝ := 
  area_large_quarter_circle - area_small_semicircle

theorem shark_fin_falcata_area_is_correct : 
  shark_fin_falcata_area area_large_quarter_circle area_small_semicircle = (9 * Real.pi) / 8 := 
by
  sorry

end NUMINAMATH_GPT_shark_fin_falcata_area_is_correct_l676_67607


namespace NUMINAMATH_GPT_gcd_polynomial_l676_67659

theorem gcd_polynomial (b : ℤ) (h : 1729 ∣ b) : Int.gcd (b^2 + 11*b + 28) (b + 5) = 2 := 
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l676_67659


namespace NUMINAMATH_GPT_combined_profit_percentage_correct_l676_67634

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end NUMINAMATH_GPT_combined_profit_percentage_correct_l676_67634


namespace NUMINAMATH_GPT_profit_percent_eq_20_l676_67688

-- Define cost price 'C' and original selling price 'S'
variable (C S : ℝ)

-- Hypothesis: selling at 2/3 of the original price results in a 20% loss 
def condition (C S : ℝ) : Prop :=
  (2 / 3) * S = 0.8 * C

-- Main theorem: profit percent when selling at the original price is 20%
theorem profit_percent_eq_20 (C S : ℝ) (h : condition C S) : (S - C) / C * 100 = 20 :=
by
  -- Proof steps would go here but we use sorry to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_profit_percent_eq_20_l676_67688


namespace NUMINAMATH_GPT_true_or_false_is_true_l676_67618

theorem true_or_false_is_true (p q : Prop) (hp : p = true) (hq : q = false) : p ∨ q = true :=
by
  sorry

end NUMINAMATH_GPT_true_or_false_is_true_l676_67618


namespace NUMINAMATH_GPT_polynomial_factorization_l676_67643

theorem polynomial_factorization : (∀ x : ℤ, x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1)) := by
  intro x
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l676_67643


namespace NUMINAMATH_GPT_number_4_div_p_equals_l676_67638

-- Assume the necessary conditions
variables (p q : ℝ)
variables (h1 : 4 / q = 18) (h2 : p - q = 0.2777777777777778)

-- Define the proof problem
theorem number_4_div_p_equals (N : ℝ) (hN : 4 / p = N) : N = 8 :=
by 
  sorry

end NUMINAMATH_GPT_number_4_div_p_equals_l676_67638


namespace NUMINAMATH_GPT_right_angled_triangle_setB_l676_67685

def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a * a + b * b = c * c

theorem right_angled_triangle_setB :
  isRightAngledTriangle 1 1 (Real.sqrt 2) ∧
  ¬isRightAngledTriangle 1 2 3 ∧
  ¬isRightAngledTriangle 6 8 11 ∧
  ¬isRightAngledTriangle 2 3 4 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_setB_l676_67685


namespace NUMINAMATH_GPT_inscribed_circle_radius_inequality_l676_67624

open Real

variables (ABC ABD BDC : Type) -- Representing the triangles

noncomputable def r (ABC : Type) : ℝ := sorry -- radius of the inscribed circle in ABC
noncomputable def r1 (ABD : Type) : ℝ := sorry -- radius of the inscribed circle in ABD
noncomputable def r2 (BDC : Type) : ℝ := sorry -- radius of the inscribed circle in BDC

noncomputable def p (ABC : Type) : ℝ := sorry -- semiperimeter of ABC
noncomputable def p1 (ABD : Type) : ℝ := sorry -- semiperimeter of ABD
noncomputable def p2 (BDC : Type) : ℝ := sorry -- semiperimeter of BDC

noncomputable def S (ABC : Type) : ℝ := sorry -- area of ABC
noncomputable def S1 (ABD : Type) : ℝ := sorry -- area of ABD
noncomputable def S2 (BDC : Type) : ℝ := sorry -- area of BDC

lemma triangle_area_sum (ABC ABD BDC : Type) :
  S ABC = S1 ABD + S2 BDC := sorry

lemma semiperimeter_area_relation (ABC ABD BDC : Type) :
  S ABC = p ABC * r ABC ∧
  S1 ABD = p1 ABD * r1 ABD ∧
  S2 BDC = p2 BDC * r2 BDC := sorry

theorem inscribed_circle_radius_inequality (ABC ABD BDC : Type) :
  r1 ABD + r2 BDC > r ABC := sorry

end NUMINAMATH_GPT_inscribed_circle_radius_inequality_l676_67624


namespace NUMINAMATH_GPT_additional_rows_added_l676_67661

theorem additional_rows_added
  (initial_tiles : ℕ) (initial_rows : ℕ) (initial_columns : ℕ) (new_columns : ℕ) (new_rows : ℕ)
  (h1 : initial_tiles = 48)
  (h2 : initial_rows = 6)
  (h3 : initial_columns = initial_tiles / initial_rows)
  (h4 : new_columns = initial_columns - 2)
  (h5 : new_rows = initial_tiles / new_columns) :
  new_rows - initial_rows = 2 := by sorry

end NUMINAMATH_GPT_additional_rows_added_l676_67661


namespace NUMINAMATH_GPT_sq_sum_ge_one_third_l676_67684

theorem sq_sum_ge_one_third (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 := 
sorry

end NUMINAMATH_GPT_sq_sum_ge_one_third_l676_67684


namespace NUMINAMATH_GPT_interest_rate_proof_l676_67695

noncomputable def remaining_interest_rate (total_investment yearly_interest part_investment interest_rate_part amount_remaining_interest : ℝ) : Prop :=
  (part_investment * interest_rate_part) + amount_remaining_interest = yearly_interest ∧
  (total_investment - part_investment) * (amount_remaining_interest / (total_investment - part_investment)) = amount_remaining_interest

theorem interest_rate_proof :
  remaining_interest_rate 3000 256 800 0.1 176 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_proof_l676_67695


namespace NUMINAMATH_GPT_factorization_of_a_square_minus_one_l676_67635

theorem factorization_of_a_square_minus_one (a : ℤ) : a^2 - 1 = (a + 1) * (a - 1) := 
  by sorry

end NUMINAMATH_GPT_factorization_of_a_square_minus_one_l676_67635


namespace NUMINAMATH_GPT_sum_of_distinct_product_GH_l676_67650

def divisible_by_45 (n : ℕ) : Prop :=
  45 ∣ n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_single_digit (d : ℕ) : Prop :=
  d < 10

theorem sum_of_distinct_product_GH : 
  ∀ (G H : ℕ), 
    is_single_digit G ∧ is_single_digit H ∧ 
    divisible_by_45 (8620000307 + 10000000 * G + H) → 
    (if H = 5 then GH = 6 else if H = 0 then GH = 0 else GH = 0) := 
  sorry

-- Note: This is a simplified representation; tailored more complex conditions and steps may be encapsulated in separate definitions and theorems as needed.

end NUMINAMATH_GPT_sum_of_distinct_product_GH_l676_67650


namespace NUMINAMATH_GPT_fifth_term_arithmetic_sequence_is_19_l676_67647

def arithmetic_sequence_nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem fifth_term_arithmetic_sequence_is_19 :
  arithmetic_sequence_nth_term 3 4 5 = 19 := 
  by
  sorry

end NUMINAMATH_GPT_fifth_term_arithmetic_sequence_is_19_l676_67647


namespace NUMINAMATH_GPT_decorations_count_l676_67692

-- Define the conditions as Lean definitions
def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def large_cauldron := 1
def budget_more_decorations := 20
def left_to_put_up := 10

-- Define the total decorations
def decorations_already_up := plastic_skulls + broomsticks + spiderwebs + pumpkins + large_cauldron
def additional_decorations := budget_more_decorations + left_to_put_up
def total_decorations := decorations_already_up + additional_decorations

-- Prove the total number of decorations will be 83
theorem decorations_count : total_decorations = 83 := by 
  sorry

end NUMINAMATH_GPT_decorations_count_l676_67692


namespace NUMINAMATH_GPT_shelves_needed_is_five_l676_67682

-- Definitions for the conditions
def initial_bears : Nat := 15
def additional_bears : Nat := 45
def bears_per_shelf : Nat := 12

-- Adding the number of bears received to the initial stock
def total_bears : Nat := initial_bears + additional_bears

-- Calculating the number of shelves used
def shelves_used : Nat := total_bears / bears_per_shelf

-- Statement to prove
theorem shelves_needed_is_five : shelves_used = 5 :=
by
  -- Insert specific step only if necessary, otherwise use sorry
  sorry

end NUMINAMATH_GPT_shelves_needed_is_five_l676_67682


namespace NUMINAMATH_GPT_problems_per_page_l676_67673

theorem problems_per_page (total_problems : ℕ) (percent_solved : ℝ) (pages_left : ℕ)
  (h_total : total_problems = 550)
  (h_percent : percent_solved = 0.65)
  (h_pages : pages_left = 3) :
  (total_problems - Nat.ceil (percent_solved * total_problems)) / pages_left = 64 := by
  sorry

end NUMINAMATH_GPT_problems_per_page_l676_67673


namespace NUMINAMATH_GPT_num_persons_initially_l676_67611

theorem num_persons_initially (N : ℕ) (avg_weight : ℝ) 
  (h_increase_avg : avg_weight + 5 = avg_weight + 40 / N) :
  N = 8 := by
    sorry

end NUMINAMATH_GPT_num_persons_initially_l676_67611


namespace NUMINAMATH_GPT_arithmetic_sequence_closed_form_l676_67636

noncomputable def B_n (n : ℕ) : ℝ :=
  2 * (1 - (-2)^n) / 3

theorem arithmetic_sequence_closed_form (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h1 : a_n 1 = 1) (h2 : S_n 3 = 0) :
  B_n n = 2 * (1 - (-2)^n) / 3 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_closed_form_l676_67636


namespace NUMINAMATH_GPT_eval_f_a_plus_1_l676_67662

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the condition
axiom a : ℝ

-- State the theorem to be proven
theorem eval_f_a_plus_1 : f (a + 1) = a^2 + 2*a + 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_f_a_plus_1_l676_67662


namespace NUMINAMATH_GPT_determinant_trig_matrix_eq_one_l676_67675

theorem determinant_trig_matrix_eq_one (α θ : ℝ) :
  Matrix.det ![
  ![Real.cos α * Real.cos θ, Real.cos α * Real.sin θ, Real.sin α],
  ![Real.sin θ, -Real.cos θ, 0],
  ![Real.sin α * Real.cos θ, Real.sin α * Real.sin θ, -Real.cos α]
  ] = 1 :=
by
  sorry

end NUMINAMATH_GPT_determinant_trig_matrix_eq_one_l676_67675


namespace NUMINAMATH_GPT_sum_of_squares_ge_two_ab_l676_67613

theorem sum_of_squares_ge_two_ab (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b := 
  sorry

end NUMINAMATH_GPT_sum_of_squares_ge_two_ab_l676_67613


namespace NUMINAMATH_GPT_no_solutions_l676_67679

/-- Prove that there are no pairs of positive integers (x, y) such that x² + y² + x = 2x³. -/
theorem no_solutions : ∀ x y : ℕ, 0 < x → 0 < y → (x^2 + y^2 + x = 2 * x^3) → false :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_l676_67679


namespace NUMINAMATH_GPT_centipede_and_earthworm_meeting_time_l676_67654

noncomputable def speed_centipede : ℚ := 5 / 3
noncomputable def speed_earthworm : ℚ := 5 / 2
noncomputable def initial_gap : ℚ := 20

theorem centipede_and_earthworm_meeting_time : 
  ∃ t : ℚ, (5 / 2) * t = initial_gap + (5 / 3) * t ∧ t = 24 := 
by
  sorry

end NUMINAMATH_GPT_centipede_and_earthworm_meeting_time_l676_67654


namespace NUMINAMATH_GPT_employed_females_percentage_l676_67621

theorem employed_females_percentage (E M : ℝ) (hE : E = 60) (hM : M = 42) : ((E - M) / E) * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_employed_females_percentage_l676_67621


namespace NUMINAMATH_GPT_real_solutions_count_l676_67625

theorem real_solutions_count : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), (2 : ℝ) ^ (3 * x ^ 2 - 8 * x + 4) = 1 → x = 2 ∨ x = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_count_l676_67625


namespace NUMINAMATH_GPT_sum_of_natural_numbers_eq_4005_l676_67606

theorem sum_of_natural_numbers_eq_4005 :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 4005 ∧ n = 89 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_natural_numbers_eq_4005_l676_67606


namespace NUMINAMATH_GPT_equal_lengths_imply_equal_segments_l676_67608

theorem equal_lengths_imply_equal_segments 
  (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h₁ : a₁ = a₂) 
  (h₂ : b₁ = b₂) : 
  x = y := 
sorry

end NUMINAMATH_GPT_equal_lengths_imply_equal_segments_l676_67608


namespace NUMINAMATH_GPT_rational_solutions_product_l676_67603

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem rational_solutions_product :
  ∀ c : ℕ, (c > 0) → (is_perfect_square (49 - 12 * c)) → (∃ a b : ℕ, a = 4 ∧ b = 2 ∧ a * b = 8) :=
by sorry

end NUMINAMATH_GPT_rational_solutions_product_l676_67603


namespace NUMINAMATH_GPT_time_between_four_and_five_straight_line_l676_67665

theorem time_between_four_and_five_straight_line :
  ∃ t : ℚ, t = 21 + 9/11 ∨ t = 54 + 6/11 :=
by
  sorry

end NUMINAMATH_GPT_time_between_four_and_five_straight_line_l676_67665


namespace NUMINAMATH_GPT_min_expression_l676_67689

theorem min_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1/a + 1/b = 1) : 
  (∃ x : ℝ, x = min ((1 / (a - 1)) + (4 / (b - 1))) 4) :=
sorry

end NUMINAMATH_GPT_min_expression_l676_67689


namespace NUMINAMATH_GPT_find_nm_l676_67623

theorem find_nm :
  ∃ n m : Int, (-120 : Int) ≤ n ∧ n ≤ 120 ∧ (-120 : Int) ≤ m ∧ m ≤ 120 ∧ 
  (Real.sin (n * Real.pi / 180) = Real.sin (580 * Real.pi / 180)) ∧ 
  (Real.cos (m * Real.pi / 180) = Real.cos (300 * Real.pi / 180)) ∧ 
  n = -40 ∧ m = -60 := by
  sorry

end NUMINAMATH_GPT_find_nm_l676_67623


namespace NUMINAMATH_GPT_number_of_circumcenter_quadrilaterals_l676_67617

-- Definitions for each type of quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

def is_square (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_kite (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def has_circumcenter (q : Quadrilateral) : Prop := sorry

-- List of quadrilaterals
def square : Quadrilateral := sorry
def rectangle : Quadrilateral := sorry
def rhombus : Quadrilateral := sorry
def kite : Quadrilateral := sorry
def trapezoid : Quadrilateral := sorry

-- Proof that the number of quadrilaterals with a point equidistant from all vertices is 2
theorem number_of_circumcenter_quadrilaterals :
  (has_circumcenter square) ∧
  (has_circumcenter rectangle) ∧
  ¬ (has_circumcenter rhombus) ∧
  ¬ (has_circumcenter kite) ∧
  ¬ (has_circumcenter trapezoid) →
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_circumcenter_quadrilaterals_l676_67617


namespace NUMINAMATH_GPT_math_problem_l676_67656

noncomputable def problem_statement (f : ℚ → ℝ) : Prop :=
  (∀ r s : ℚ, ∃ n : ℤ, f (r + s) = f r + f s + n) →
  ∃ (q : ℕ) (p : ℤ), abs (f (1 / q) - p) ≤ 1 / 2012

-- To state this problem as a theorem in Lean 4
theorem math_problem (f : ℚ → ℝ) :
  problem_statement f :=
sorry

end NUMINAMATH_GPT_math_problem_l676_67656


namespace NUMINAMATH_GPT_bus_total_distance_l676_67610

theorem bus_total_distance
  (distance40 : ℝ)
  (distance60 : ℝ)
  (speed40 : ℝ)
  (speed60 : ℝ)
  (total_time : ℝ)
  (distance40_eq : distance40 = 100)
  (speed40_eq : speed40 = 40)
  (speed60_eq : speed60 = 60)
  (total_time_eq : total_time = 5)
  (time40 : ℝ)
  (time40_eq : time40 = distance40 / speed40)
  (time_equation : time40 + distance60 / speed60 = total_time) :
  distance40 + distance60 = 250 := sorry

end NUMINAMATH_GPT_bus_total_distance_l676_67610


namespace NUMINAMATH_GPT_quadratic_expression_value_l676_67629

theorem quadratic_expression_value (a : ℝ)
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 + 2 * (a - 1) * x₁ + a^2 - 7 * a - 4 = 0 ∧ x₂^2 + 2 * (a - 1) * x₂ + a^2 - 7 * a - 4 = 0)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ * x₂ - 3 * x₁ - 3 * x₂ - 2 = 0) :
  (1 + 4 / (a^2 - 4)) * (a + 2) / a = 2 := 
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l676_67629


namespace NUMINAMATH_GPT_min_photos_for_condition_l676_67651

noncomputable def minimum_photos (girls boys : ℕ) : ℕ :=
  if (girls = 4 ∧ boys = 8) 
  then 33
  else 0

theorem min_photos_for_condition (girls boys : ℕ) (photos : ℕ) :
  girls = 4 → boys = 8 → photos = minimum_photos girls boys
  → ∃ (pa : ℕ), pa >= 33 → pa = photos :=
by
  intros h1 h2 h3
  use minimum_photos girls boys
  rw [h3]
  sorry

end NUMINAMATH_GPT_min_photos_for_condition_l676_67651


namespace NUMINAMATH_GPT_expression_for_C_value_of_C_l676_67697

variables (x y : ℝ)

-- Definitions based on the given conditions
def A := x^2 - 2 * x * y + y^2
def B := x^2 + 2 * x * y + y^2

-- The algebraic expression for C
def C := - x^2 + 10 * x * y - y^2

-- Prove that the expression for C is correct
theorem expression_for_C (h : 3 * A x y - 2 * B x y + C x y = 0) : 
  C x y = - x^2 + 10 * x * y - y^2 := 
by {
  sorry
}

-- Prove the value of C when x = 1/2 and y = -2
theorem value_of_C : C (1/2) (-2) = -57/4 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_for_C_value_of_C_l676_67697


namespace NUMINAMATH_GPT_necklace_sum_l676_67615

theorem necklace_sum (H J x S : ℕ) (hH : H = 25) (h1 : H = J + 5) (h2 : x = J / 2) (h3 : S = 2 * H) : H + J + x + S = 105 :=
by 
  sorry

end NUMINAMATH_GPT_necklace_sum_l676_67615


namespace NUMINAMATH_GPT_intersection_A_B_eq_l676_67698

def A : Set ℝ := { x | (x / (x - 1)) ≥ 0 }

def B : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B_eq :
  (A ∩ B) = { y : ℝ | 1 < y } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_eq_l676_67698


namespace NUMINAMATH_GPT_circle_radius_l676_67632

theorem circle_radius (A B C O : Type) (AB AC : ℝ) (OA : ℝ) (r : ℝ) 
  (h1 : AB * AC = 60)
  (h2 : OA = 8) 
  (h3 : (8 + r) * (8 - r) = 60) : r = 2 :=
sorry

end NUMINAMATH_GPT_circle_radius_l676_67632


namespace NUMINAMATH_GPT_intersection_A_B_l676_67691

def A : Set ℝ := { x | Real.sqrt x ≤ 3 }
def B : Set ℝ := { x | x^2 ≤ 9 }

theorem intersection_A_B : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l676_67691


namespace NUMINAMATH_GPT_third_car_year_l676_67626

theorem third_car_year (y1 y2 y3 : ℕ) (h1 : y1 = 1970) (h2 : y2 = y1 + 10) (h3 : y3 = y2 + 20) : y3 = 2000 :=
by
  sorry

end NUMINAMATH_GPT_third_car_year_l676_67626


namespace NUMINAMATH_GPT_total_deposit_amount_l676_67616

def markDeposit : ℕ := 88
def bryanDeposit (markAmount : ℕ) : ℕ := 5 * markAmount - 40
def totalDeposit (markAmount bryanAmount : ℕ) : ℕ := markAmount + bryanAmount

theorem total_deposit_amount : totalDeposit markDeposit (bryanDeposit markDeposit) = 488 := 
by sorry

end NUMINAMATH_GPT_total_deposit_amount_l676_67616


namespace NUMINAMATH_GPT_function_intersects_all_lines_l676_67660

theorem function_intersects_all_lines :
  (∃ f : ℝ → ℝ, (∀ a : ℝ, ∃ y : ℝ, y = f a) ∧ (∀ k b : ℝ, ∃ x : ℝ, f x = k * x + b)) :=
sorry

end NUMINAMATH_GPT_function_intersects_all_lines_l676_67660


namespace NUMINAMATH_GPT_total_weight_mason_hotdogs_l676_67669

-- Definitions from conditions
def weight_hotdog := 2
def weight_burger := 5
def weight_pie := 10
def noah_burgers := 8
def jacob_pies := noah_burgers - 3
def mason_hotdogs := 3 * jacob_pies

-- Statement to prove
theorem total_weight_mason_hotdogs : mason_hotdogs * weight_hotdog = 30 := 
by 
  sorry

end NUMINAMATH_GPT_total_weight_mason_hotdogs_l676_67669


namespace NUMINAMATH_GPT_intersection_points_circle_l676_67664

-- Defining the two lines based on the parameter u
def line1 (u : ℝ) (x y : ℝ) : Prop := 2 * u * x - 3 * y - 2 * u = 0
def line2 (u : ℝ) (x y : ℝ) : Prop := x - 3 * u * y + 2 = 0

-- Proof statement that shows the intersection points lie on a circle
theorem intersection_points_circle (u x y : ℝ) :
  line1 u x y → line2 u x y → (x - 1)^2 + y^2 = 1 :=
by {
  -- This completes the proof statement, but leaves implementation as exercise
  sorry
}

end NUMINAMATH_GPT_intersection_points_circle_l676_67664


namespace NUMINAMATH_GPT_quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l676_67696

theorem quadratic_has_two_real_roots_for_any_m (m : ℝ) : 
  ∃ (α β : ℝ), (α^2 - 3*α + 2 - m^2 - m = 0) ∧ (β^2 - 3*β + 2 - m^2 - m = 0) :=
sorry

theorem find_m_given_roots_conditions (α β : ℝ) (m : ℝ) 
  (h1 : α^2 - 3*α + 2 - m^2 - m = 0) 
  (h2 : β^2 - 3*β + 2 - m^2 - m = 0) 
  (h3 : α^2 + β^2 = 9) : 
  m = -2 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_real_roots_for_any_m_find_m_given_roots_conditions_l676_67696


namespace NUMINAMATH_GPT_percent_difference_calculation_l676_67631

theorem percent_difference_calculation :
  (0.80 * 45) - ((4 / 5) * 25) = 16 :=
by sorry

end NUMINAMATH_GPT_percent_difference_calculation_l676_67631


namespace NUMINAMATH_GPT_distinct_roots_quadratic_l676_67605

theorem distinct_roots_quadratic (a x₁ x₂ : ℝ) (h₁ : x^2 + a*x + 8 = 0) 
  (h₂ : x₁ ≠ x₂) (h₃ : x₁ - 64 / (17 * x₂^3) = x₂ - 64 / (17 * x₁^3)) : 
  a = 12 ∨ a = -12 := 
sorry

end NUMINAMATH_GPT_distinct_roots_quadratic_l676_67605


namespace NUMINAMATH_GPT_bond_selling_price_l676_67627

def bond_face_value : ℝ := 5000
def bond_interest_rate : ℝ := 0.06
def interest_approx : ℝ := bond_face_value * bond_interest_rate
def selling_price_interest_rate : ℝ := 0.065
def approximate_selling_price : ℝ := 4615.38

theorem bond_selling_price :
  interest_approx = selling_price_interest_rate * approximate_selling_price :=
sorry

end NUMINAMATH_GPT_bond_selling_price_l676_67627


namespace NUMINAMATH_GPT_min_time_to_same_side_l676_67687

def side_length : ℕ := 50
def speed_A : ℕ := 5
def speed_B : ℕ := 3

def time_to_same_side (side_length speed_A speed_B : ℕ) : ℕ :=
  30

theorem min_time_to_same_side :
  time_to_same_side side_length speed_A speed_B = 30 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_min_time_to_same_side_l676_67687


namespace NUMINAMATH_GPT_find_a3_l676_67614

-- Given conditions
def sequence_sum (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n^2 + n

-- Define the sequence term calculation from the sum function.
def seq_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem find_a3 (S : ℕ → ℕ) (h : sequence_sum S) :
  seq_term S 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l676_67614


namespace NUMINAMATH_GPT_inequality_always_holds_l676_67633

theorem inequality_always_holds (a b : ℝ) (h : a * b > 0) : (b / a + a / b) ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_always_holds_l676_67633


namespace NUMINAMATH_GPT_range_m_l676_67677

def A (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_m (m : ℝ) :
  (∀ x, B m x → A x) ↔ -3 ≤ m ∧ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_m_l676_67677


namespace NUMINAMATH_GPT_number_of_questions_in_test_l676_67653

theorem number_of_questions_in_test (x : ℕ) (sections questions_correct : ℕ)
  (h_sections : sections = 5)
  (h_questions_correct : questions_correct = 32)
  (h_percentage : 0.70 < (questions_correct : ℚ) / x ∧ (questions_correct : ℚ) / x < 0.77) 
  (h_multiple_of_sections : x % sections = 0) : 
  x = 45 :=
sorry

end NUMINAMATH_GPT_number_of_questions_in_test_l676_67653


namespace NUMINAMATH_GPT_visitors_saturday_l676_67639

def friday_visitors : ℕ := 3575
def saturday_visitors : ℕ := 5 * friday_visitors

theorem visitors_saturday : saturday_visitors = 17875 := by
  -- proof details would go here
  sorry

end NUMINAMATH_GPT_visitors_saturday_l676_67639


namespace NUMINAMATH_GPT_sum_of_integers_c_with_four_solutions_l676_67601

noncomputable def g (x : ℝ) : ℝ :=
  ((x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 120) - 2

theorem sum_of_integers_c_with_four_solutions :
  (∃ (c : ℤ), ∀ x : ℝ, -4.5 ≤ x ∧ x ≤ 4.5 → g x = c ↔ c = -2) → c = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_c_with_four_solutions_l676_67601


namespace NUMINAMATH_GPT_total_amount_l676_67641

def shares (a b c : ℕ) : Prop :=
  b = 1800 ∧ 2 * b = 3 * a ∧ 3 * c = 4 * b

theorem total_amount (a b c : ℕ) (h : shares a b c) : a + b + c = 5400 :=
by
  have h₁ : 2 * b = 3 * a := h.2.1
  have h₂ : 3 * c = 4 * b := h.2.2
  have hb : b = 1800 := h.1
  sorry

end NUMINAMATH_GPT_total_amount_l676_67641


namespace NUMINAMATH_GPT_find_n_in_sequence_l676_67666

theorem find_n_in_sequence (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ n, a (n+1) = 2 * a n) 
    (h3 : S n = 126) 
    (h4 : S n = 2^(n+1) - 2) : 
  n = 6 :=
sorry

end NUMINAMATH_GPT_find_n_in_sequence_l676_67666


namespace NUMINAMATH_GPT_lcm_is_2310_l676_67628

def a : ℕ := 210
def b : ℕ := 605
def hcf : ℕ := 55

theorem lcm_is_2310 (lcm : ℕ) : Nat.lcm a b = 2310 :=
by 
  have h : a * b = lcm * hcf := by sorry
  sorry

end NUMINAMATH_GPT_lcm_is_2310_l676_67628


namespace NUMINAMATH_GPT_square_units_digit_eq_9_l676_67620

/-- The square of which whole number has a units digit of 9? -/
theorem square_units_digit_eq_9 (n : ℕ) (h : ∃ m : ℕ, n = m^2 ∧ m % 10 = 9) : n = 3 ∨ n = 7 := by
  sorry

end NUMINAMATH_GPT_square_units_digit_eq_9_l676_67620


namespace NUMINAMATH_GPT_velocity_at_t1_l676_67640

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2 * t

-- Define the velocity function as the derivative of s
def velocity (t : ℝ) : ℝ := -2 * t + 2

-- Prove that the velocity at t = 1 is 0
theorem velocity_at_t1 : velocity 1 = 0 :=
by
  -- Apply the definition of velocity
    sorry

end NUMINAMATH_GPT_velocity_at_t1_l676_67640


namespace NUMINAMATH_GPT_inverse_of_f_inverse_of_f_inv_l676_67674

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + 1

noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.log x / Real.log 3

theorem inverse_of_f (x : ℝ) (hx : x > 1) : f_inv (f x) = x :=
by
  sorry

theorem inverse_of_f_inv (x : ℝ) (hx : x > 1) : f (f_inv x) = x :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_f_inverse_of_f_inv_l676_67674


namespace NUMINAMATH_GPT_problem_proof_l676_67655

-- Define the mixed numbers and their conversions to improper fractions
def mixed_number_1 := 84 * 19 + 4  -- 1600
def mixed_number_2 := 105 * 19 + 5 -- 2000 

-- Define the improper fractions
def improper_fraction_1 := mixed_number_1 / 19
def improper_fraction_2 := mixed_number_2 / 19

-- Define the decimals and their conversions to fractions
def decimal_1 := 11 / 8  -- 1.375
def decimal_2 := 9 / 10  -- 0.9

-- Perform the multiplications
def multiplication_1 := (improper_fraction_1 * decimal_1 : ℚ)
def multiplication_2 := (improper_fraction_2 * decimal_2 : ℚ)

-- Perform the addition
def addition_result := multiplication_1 + multiplication_2

-- The final result is converted to a fraction for comparison
def final_result := 4000 / 19

-- Define and state the theorem
theorem problem_proof : addition_result = final_result := by
  sorry

end NUMINAMATH_GPT_problem_proof_l676_67655


namespace NUMINAMATH_GPT_find_constants_l676_67657

variable (x : ℝ)

def A := 3
def B := -3
def C := 11

theorem find_constants (h₁ : x ≠ 2) (h₂ : x ≠ 4) :
  (5 * x + 2) / ((x - 2) * (x - 4)^2) = A / (x - 2) + B / (x - 4) + C / (x - 4)^2 :=
by
  unfold A B C
  sorry

end NUMINAMATH_GPT_find_constants_l676_67657


namespace NUMINAMATH_GPT_beads_needed_for_jewelry_l676_67600

/-
  We define the parameters based on the problem statement.
-/

def green_beads : ℕ := 3
def purple_beads : ℕ := 5
def red_beads : ℕ := 2 * green_beads
def total_beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

def repeats_per_bracelet : ℕ := 3
def repeats_per_necklace : ℕ := 5

/-
  We calculate the total number of beads for 1 bracelet and 10 necklaces.
-/

def beads_per_bracelet : ℕ := total_beads_per_pattern * repeats_per_bracelet
def beads_per_necklace : ℕ := total_beads_per_pattern * repeats_per_necklace
def total_beads_needed : ℕ := beads_per_bracelet + beads_per_necklace * 10

theorem beads_needed_for_jewelry:
  total_beads_needed = 742 :=
by 
  sorry

end NUMINAMATH_GPT_beads_needed_for_jewelry_l676_67600


namespace NUMINAMATH_GPT_complex_real_part_of_product_l676_67663

theorem complex_real_part_of_product (z1 z2 : ℂ) (i : ℂ) 
  (hz1 : z1 = 4 + 29 * Complex.I)
  (hz2 : z2 = 6 + 9 * Complex.I)
  (hi : i = Complex.I) : 
  ((z1 - z2) * i).re = 20 := 
by
  sorry

end NUMINAMATH_GPT_complex_real_part_of_product_l676_67663


namespace NUMINAMATH_GPT_find_number_exceeds_sixteen_percent_l676_67676

theorem find_number_exceeds_sixteen_percent (x : ℝ) (h : x - 0.16 * x = 63) : x = 75 :=
sorry

end NUMINAMATH_GPT_find_number_exceeds_sixteen_percent_l676_67676


namespace NUMINAMATH_GPT_even_sum_count_l676_67652

theorem even_sum_count (x y : ℕ) 
  (hx : x = (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)) 
  (hy : y = ((60 - 40) / 2 + 1)) : 
  x + y = 561 := 
by 
  sorry

end NUMINAMATH_GPT_even_sum_count_l676_67652


namespace NUMINAMATH_GPT_sin_cos_identity_l676_67683

theorem sin_cos_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) 
  - Real.cos (200 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 := 
by
  -- This would be where the proof goes
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l676_67683


namespace NUMINAMATH_GPT_stans_average_speed_l676_67644

/-- Given that Stan drove 420 miles in 6 hours, 480 miles in 7 hours, and 300 miles in 5 hours,
prove that his average speed for the entire trip is 1200/18 miles per hour. -/
theorem stans_average_speed :
  let total_distance := 420 + 480 + 300
  let total_time := 6 + 7 + 5
  total_distance / total_time = 1200 / 18 :=
by
  sorry

end NUMINAMATH_GPT_stans_average_speed_l676_67644


namespace NUMINAMATH_GPT_parabola_point_b_l676_67612

variable {a b : ℝ}

theorem parabola_point_b (h1 : 6 = 2^2 + 2*a + b) (h2 : -14 = (-2)^2 - 2*a + b) : b = -8 :=
by
  -- sorry as a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_parabola_point_b_l676_67612


namespace NUMINAMATH_GPT_solve_x4_minus_inv_x4_l676_67649

-- Given condition
def condition (x : ℝ) : Prop := x - (1 / x) = 5

-- Theorem statement ensuring the problem is mathematically equivalent
theorem solve_x4_minus_inv_x4 (x : ℝ) (hx : condition x) : x^4 - (1 / x^4) = 723 :=
by
  sorry

end NUMINAMATH_GPT_solve_x4_minus_inv_x4_l676_67649


namespace NUMINAMATH_GPT_letters_with_dot_not_line_l676_67694

-- Definitions from conditions
def D_inter_S : ℕ := 23
def S : ℕ := 42
def Total_letters : ℕ := 70

-- Problem statement
theorem letters_with_dot_not_line : (Total_letters - S - D_inter_S) = 5 :=
by sorry

end NUMINAMATH_GPT_letters_with_dot_not_line_l676_67694


namespace NUMINAMATH_GPT_number_of_boys_in_second_class_l676_67690

def boys_in_first_class : ℕ := 28
def portion_of_second_class (b2 : ℕ) : ℚ := 7 / 8 * b2

theorem number_of_boys_in_second_class (b2 : ℕ) (h : portion_of_second_class b2 = boys_in_first_class) : b2 = 32 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_boys_in_second_class_l676_67690


namespace NUMINAMATH_GPT_number_of_math_fun_books_l676_67609

def intelligence_challenge_cost := 18
def math_fun_cost := 8
def total_spent := 92

theorem number_of_math_fun_books (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : intelligence_challenge_cost * x + math_fun_cost * y = total_spent) : y = 7 := 
by
  sorry

end NUMINAMATH_GPT_number_of_math_fun_books_l676_67609


namespace NUMINAMATH_GPT_cube_vertices_faces_edges_l676_67642

theorem cube_vertices_faces_edges (V F E : ℕ) (hv : V = 8) (hf : F = 6) (euler : V - E + F = 2) : E = 12 :=
by
  sorry

end NUMINAMATH_GPT_cube_vertices_faces_edges_l676_67642


namespace NUMINAMATH_GPT_side_of_square_is_25_l676_67637

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end NUMINAMATH_GPT_side_of_square_is_25_l676_67637


namespace NUMINAMATH_GPT_remainder_numGreenRedModal_l676_67646

def numGreenMarbles := 7
def numRedMarbles (n : ℕ) := 7 + n
def validArrangement (g r : ℕ) := (g + r = numGreenMarbles + numRedMarbles r) ∧ 
  (g = r)

theorem remainder_numGreenRedModal (N' : ℕ) :
  N' % 1000 = 432 :=
sorry

end NUMINAMATH_GPT_remainder_numGreenRedModal_l676_67646


namespace NUMINAMATH_GPT_total_amount_l676_67668

noncomputable def A : ℝ := 396.00000000000006
noncomputable def B : ℝ := A * (3 / 2)
noncomputable def C : ℝ := B * 4

theorem total_amount (A_eq : A = 396.00000000000006) (A_B_relation : A = (2 / 3) * B) (B_C_relation : B = (1 / 4) * C) :
  396.00000000000006 + B + C = 3366.000000000001 := by
  sorry

end NUMINAMATH_GPT_total_amount_l676_67668


namespace NUMINAMATH_GPT_minimum_value_expression_l676_67670

open Real

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2)) / (a * b * c) ≥ 216 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l676_67670


namespace NUMINAMATH_GPT_sum_bn_2999_l676_67648

def b_n (n : ℕ) : ℕ :=
  if n % 17 = 0 ∧ n % 19 = 0 then 15
  else if n % 19 = 0 ∧ n % 13 = 0 then 18
  else if n % 13 = 0 ∧ n % 17 = 0 then 17
  else 0

theorem sum_bn_2999 : (Finset.range 3000).sum b_n = 572 := by
  sorry

end NUMINAMATH_GPT_sum_bn_2999_l676_67648


namespace NUMINAMATH_GPT_probability_of_different_cousins_name_l676_67602

theorem probability_of_different_cousins_name :
  let total_letters := 19
  let amelia_letters := 6
  let bethany_letters := 7
  let claire_letters := 6
  let probability := 
    2 * ((amelia_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)) +
         (amelia_letters / (total_letters : ℚ)) * (claire_letters / (total_letters - 1 : ℚ)) +
         (claire_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)))
  probability = 40 / 57 := sorry

end NUMINAMATH_GPT_probability_of_different_cousins_name_l676_67602


namespace NUMINAMATH_GPT_sum_of_first_and_third_is_68_l676_67686

theorem sum_of_first_and_third_is_68
  (A B C : ℕ)
  (h1 : A + B + C = 98)
  (h2 : A * 3 = B * 2)  -- implying A / B = 2 / 3
  (h3 : B * 8 = C * 5)  -- implying B / C = 5 / 8
  (h4 : B = 30) :
  A + C = 68 :=
sorry

end NUMINAMATH_GPT_sum_of_first_and_third_is_68_l676_67686


namespace NUMINAMATH_GPT_weight_of_each_piece_l676_67658

theorem weight_of_each_piece 
  (x : ℝ)
  (h : 2 * x + 0.08 = 0.75) : 
  x = 0.335 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_each_piece_l676_67658


namespace NUMINAMATH_GPT_selena_left_with_l676_67667

/-- Selena got a tip of $99 and spent money on various foods whose individual costs are provided. 
Prove that she will be left with $38. -/
theorem selena_left_with : 
  let tip := 99
  let steak_cost := 24
  let num_steaks := 2
  let burger_cost := 3.5
  let num_burgers := 2
  let ice_cream_cost := 2
  let num_ice_cream := 3
  let total_spent := (steak_cost * num_steaks) + (burger_cost * num_burgers) + (ice_cream_cost * num_ice_cream)
  tip - total_spent = 38 := 
by 
  sorry

end NUMINAMATH_GPT_selena_left_with_l676_67667


namespace NUMINAMATH_GPT_base_number_of_equation_l676_67699

theorem base_number_of_equation (n : ℕ) (h_n: n = 17)
  (h_eq: 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^18) : some_number = 2 := by
  sorry

end NUMINAMATH_GPT_base_number_of_equation_l676_67699


namespace NUMINAMATH_GPT_star_inequalities_not_all_true_simultaneously_l676_67678

theorem star_inequalities_not_all_true_simultaneously
  (AB BC CD DE EF FG GH HK KL LA : ℝ)
  (h1 : BC > AB)
  (h2 : DE > CD)
  (h3 : FG > EF)
  (h4 : HK > GH)
  (h5 : LA > KL) :
  False :=
  sorry

end NUMINAMATH_GPT_star_inequalities_not_all_true_simultaneously_l676_67678
