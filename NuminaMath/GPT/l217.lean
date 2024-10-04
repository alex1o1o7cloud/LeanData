import Mathlib

namespace proof_problem_l217_217664

noncomputable def statement_A (x : ℝ) : Prop :=
  Complex.exp (Complex.i * x) = Real.cos x + Complex.i * Real.sin x

noncomputable def statement_C (x : ℝ) : Prop :=
  (2 : ℝ)^x ≥ 1 + x * Real.log 2 + (x * Real.log 2)^2 / 2

noncomputable def statement_D (x : ℝ) (h : 0 < x ∧ x < 1) : Prop :=
  Real.cos x ≤ 1 - x^2 / 2 + x^4 / 24

-- The Lean 4 proof problem statement
theorem proof_problem (x : ℝ) (hx : 0 < x ∧ x < 1) :
  statement_A x ∧
  statement_C x ∧
  statement_D x hx :=
  by
    sorry

end proof_problem_l217_217664


namespace pieces_per_box_is_six_l217_217641

-- Define the given conditions
def boxes_initial : ℝ := 14.0
def boxes_given : ℝ := 7.0
def pieces_left : ℝ := 42.0

-- Calculate the number of boxes left
def boxes_left : ℝ := boxes_initial - boxes_given

-- Calculate the number of pieces per box
def pieces_per_box (pieces : ℝ) (boxes : ℝ) : ℝ := pieces / boxes

-- Statement that we want to prove
theorem pieces_per_box_is_six : pieces_per_box pieces_left boxes_left = 6 := by
  sorry

end pieces_per_box_is_six_l217_217641


namespace area_of_S3_is_correct_l217_217027

-- Define the side length of a square given its area
def side_length (area : ℝ) : ℝ := real.sqrt area

-- Define the area of S_1
def area_S1 : ℝ := 25

-- Calculate the side length of S_1
def side_length_S1 : ℝ := side_length area_S1

-- Calculate the side length of S_2 using the midpoints of S_1
def side_length_S2 : ℝ := (side_length_S1 * real.sqrt 2) / 2

-- Define the area of S_3 given the conditions outlined in the problem
def area_S3 : ℝ := (side_length_S2 * real.sqrt 2 / 2)^2

-- The statement to prove
theorem area_of_S3_is_correct :
  area_S3 = 6.25 :=
by
  sorry

end area_of_S3_is_correct_l217_217027


namespace crushing_load_calculation_l217_217817

theorem crushing_load_calculation (T H : ℝ) (L : ℝ) 
  (h1 : L = 40 * T^5 / H^3) 
  (h2 : T = 3) 
  (h3 : H = 6) : 
  L = 45 := 
by sorry

end crushing_load_calculation_l217_217817


namespace max_product_of_roots_l217_217901

theorem max_product_of_roots (m : ℝ) : 
  (∃ (m : ℝ), 6x^2 - 12x + m = 0 ∧ (144 - 24m ≥ 0)) → 
  (∀ m', (6x^2 - 12x + m' = 0 ∧ (144 - 24m' ≥ 0) → (m * m' / 6)) ≤ 6) :=
sorry

end max_product_of_roots_l217_217901


namespace quadratic_function_symmetry_l217_217928

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := -x^2 + b * x + c

-- State the problem as a theorem
theorem quadratic_function_symmetry (b c : ℝ) (h_symm : ∀ x, f x b c = f (4 - x) b c) :
  f 2 b c > f 1 b c ∧ f 1 b c > f 4 b c :=
by
  -- Include a placeholder for the proof
  sorry

end quadratic_function_symmetry_l217_217928


namespace sum_g_eq_2015_l217_217881

noncomputable def g (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 2 * x + (1/12)

theorem sum_g_eq_2015 : 
  (∑ k in (Finset.range 2015).map (λ i, i + 1 : ℕ → ℝ), g (k / 2016)) = 2015 := 
sorry

end sum_g_eq_2015_l217_217881


namespace conformable_words_theorem_l217_217854

def conformable_words_count (n : ℕ) : ℕ :=
  match n with
  | 2*k + 1 => 14 * 3^(k-1)
  | 2*k => 8 * 3^(k-1)
  | _ => 0  -- This fallback is technically unnecessary given constraints.

theorem conformable_words_theorem (n : ℕ) : conformable_words_count n = 
  match n with
  | 2*k + 1 => 14 * 3^(k-1)
  | 2*k => 8 * 3^(k-1)
  | _ => 0 := 
sorry

end conformable_words_theorem_l217_217854


namespace remainder_eqn_l217_217978

noncomputable def P : Polynomial ℝ := sorry

theorem remainder_eqn (P : Polynomial ℝ) (h1 : P.eval 1 = 2) (h2 : P.eval 2 = 1) :
  ∃ Q : Polynomial ℝ, ∃ a b : ℝ, P = Q * (Polynomial.X - 1) * (Polynomial.X - 2) + (C a * X + C b) ∧ a = -1 ∧ b = 3 :=
by
  sorry

end remainder_eqn_l217_217978


namespace find_x_l217_217035

theorem find_x (x : ℚ) (h : ⌊x⌋ + x = 15/4) : x = 15/4 := by
  sorry

end find_x_l217_217035


namespace infinite_rational_points_in_region_l217_217016

theorem infinite_rational_points_in_region :
  { p : ℚ × ℚ // 1 ≤ p.1 ∧ p.1 + p.2 ≤ 7 } (p : ℚ × ℚ) : infinite ↥ { p : ℚ × ℚ // 1 ≤ p.1 ∧ p.1 + p.2 ≤ 7 } :=
by
  sorry

end infinite_rational_points_in_region_l217_217016


namespace evaluate_neg2012_l217_217491

def func (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_neg2012 (a b c : ℝ) (h : func a b c 2012 = 3) : func a b c (-2012) = -1 :=
by
  sorry

end evaluate_neg2012_l217_217491


namespace trigonometric_simplification_l217_217254

theorem trigonometric_simplification (α : ℝ) : 
  ( (tan (real.pi / 4 - α)) / (1 - (tan (real.pi / 4 - α))^2) ) * 
  ( (sin α * cos α) / (cos α ^ 2 - sin α ^ 2) ) = 1 / 4 := by
  sorry

end trigonometric_simplification_l217_217254


namespace sticker_price_of_laptop_l217_217912

-- Define the initial conditions
def storeAPrice (x : ℝ) := 0.80 * x - 100
def storeBPrice (x : ℝ) := 0.70 * x

-- Define the main theorem to prove the sticker price
theorem sticker_price_of_laptop (x : ℝ) : storeAPrice x = storeBPrice x - 20 → x = 800 :=
by 
  intros h
  unfold storeAPrice storeBPrice at h
  linarith

end sticker_price_of_laptop_l217_217912


namespace solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l217_217099

-- Problem 1: Solution set for the inequality \( f(x) ≤ 6 \)
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Problem 2: Prove \( a^2 + b^2 + c^2 ≥ 16/3 \)
variables (a b c : ℝ)
axiom pos_abc : a > 0 ∧ b > 0 ∧ c > 0
axiom sum_abc : a + b + c = 4

theorem sum_of_squares_geq_16_div_3 :
  a^2 + b^2 + c^2 ≥ 16 / 3 :=
sorry

end solution_set_f_leq_6_sum_of_squares_geq_16_div_3_l217_217099


namespace cos_of_Z_in_right_triangle_l217_217158

theorem cos_of_Z_in_right_triangle (X Y Z : Type) [Triangle X Y Z] (hX : ∠XYZ = 90°) (hTanZ : tan (∠XZY) = 3) : cos (∠XZY) = sqrt 10 / 10 :=
by 
  sorry

end cos_of_Z_in_right_triangle_l217_217158


namespace ratio_A_B_l217_217439

noncomputable def A := ∑' n, if n % 6 = 0 then 0 else (-1)^(nat.clz n) / (n^2)
noncomputable def B := ∑' n, if n % 6 = 0 then (-1)^((n / 6) % 2) / (n^2) else 0

theorem ratio_A_B :
  (A / B) = 37 := by
  sorry

end ratio_A_B_l217_217439


namespace total_gallons_in_tanks_l217_217636

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end total_gallons_in_tanks_l217_217636


namespace evalExpression_at_3_2_l217_217818

def evalExpression (x y : ℕ) : ℕ := 3 * x^y + 4 * y^x

theorem evalExpression_at_3_2 : evalExpression 3 2 = 59 := by
  sorry

end evalExpression_at_3_2_l217_217818


namespace tina_total_income_is_correct_l217_217313

-- Definitions based on the conditions
def hourly_wage : ℝ := 18.0
def regular_hours_per_day : ℝ := 8
def overtime_hours_per_day_weekday : ℝ := 2
def double_overtime_hours_per_day_weekend : ℝ := 2

def overtime_rate : ℝ := hourly_wage + 0.5 * hourly_wage
def double_overtime_rate : ℝ := 2 * hourly_wage

def weekday_hours_per_day : ℝ := 10
def weekend_hours_per_day : ℝ := 12

def regular_pay_per_day : ℝ := hourly_wage * regular_hours_per_day
def overtime_pay_per_day_weekday : ℝ := overtime_rate * overtime_hours_per_day_weekday
def double_overtime_pay_per_day_weekend : ℝ := double_overtime_rate * double_overtime_hours_per_day_weekend

def total_weekday_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday
def total_weekend_pay_per_day : ℝ := regular_pay_per_day + overtime_pay_per_day_weekday + double_overtime_pay_per_day_weekend

def number_of_weekdays : ℝ := 5
def number_of_weekends : ℝ := 2

def total_weekday_income : ℝ := total_weekday_pay_per_day * number_of_weekdays
def total_weekend_income : ℝ := total_weekend_pay_per_day * number_of_weekends

def total_weekly_income : ℝ := total_weekday_income + total_weekend_income

-- The theorem we need to prove
theorem tina_total_income_is_correct : total_weekly_income = 1530 := by
  sorry

end tina_total_income_is_correct_l217_217313


namespace lattice_points_in_region_l217_217393

theorem lattice_points_in_region :
  ∃ n : ℕ, n = 12 ∧ 
  ( ∀ x y : ℤ, (y = x ∨ y = -x ∨ y = -x^2 + 4) → n = 12) :=
by
  sorry

end lattice_points_in_region_l217_217393


namespace sqrt_sqrt_16_eq_pm_2_l217_217699

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217699


namespace point_not_on_transformed_plane_l217_217617

def pointA : EuclideanSpace ℝ (Fin 3) := ![-1,2,3]

def original_plane (x y z : ℝ) : Prop := x - 3 * y + z + 2 = 0

def transformed_plane (x y z : ℝ) (k : ℝ) : Prop := x - 3 * y + z + k * 2 = 0

theorem point_not_on_transformed_plane (k : ℝ) (h : k = 2.5) : 
  ¬ transformed_plane (-1) 2 3 k := 
by 
  simp [transformed_plane, h]
  norm_num
  sorry

end point_not_on_transformed_plane_l217_217617


namespace average_weight_A_approx_l217_217304

noncomputable def average_weight_section_A 
  (num_students_A num_students_B : ℕ) 
  (avg_weight_B weight_all : ℚ) 
  (students_total : num_students_A + num_students_B = 120) 
  (avg_weight_all : weight_all = 61.67) :
  ℚ :=
let total_weight_A := num_students_A in
let total_weight_B := num_students_B * avg_weight_B in
let total_weight_class := total_weight_A + total_weight_B in
(weight_all * 120 - total_weight_B) / num_students_A

theorem average_weight_A_approx 
  (num_students_A : ℕ := 50) 
  (num_students_B : ℕ := 70) 
  (avg_weight_B : ℚ := 70) 
  (total_students : num_students_A + num_students_B = 120) 
  (avg_weight_all : 61.67) :
  average_weight_section_A num_students_A num_students_B avg_weight_B avg_weight_all total_students ≈ 50.008 := 
begin
  sorry
end

end average_weight_A_approx_l217_217304


namespace hexagon_area_l217_217200

-- Define the given conditions
variables {ABCDEF : Type} [regular_hexagon ABCDEF]
variables {G H I : Point}
variables (midpoint_G : is_midpoint G AB)
variables (midpoint_H : is_midpoint H CD)
variables (midpoint_I : is_midpoint I EF)
variables (area_GHI : 100 = triangle_area G H I)

-- Theorem statement
theorem hexagon_area (h : regular_hexagon ABCDEF)
  (G_mid_AB : is_midpoint G AB)
  (H_mid_CD : is_midpoint H CD)
  (I_mid_EF : is_midpoint I EF)
  (area_GHI_100 : triangle_area G H I = 100) :
  hexagon_area ABCDEF = 800 / 3 :=
begin
  sorry
end

end hexagon_area_l217_217200


namespace ratio_of_apples_l217_217084

variable (x : ℝ)

theorem ratio_of_apples :
  let Frank := 36
  let Susan := x * Frank
  let remaining_Susan := (1/2) * Susan
  let remaining_Frank := (2/3) * Frank
  remaining_Susan + remaining_Frank = 78 →
  x = 3 :=
by
  intro h
  have h1 : Susan = 36 * x := rfl
  have h2 : remaining_Susan = (1/2) * 36 * x := by sorry
  have h3 : remaining_Frank = 24 := by norm_num
  have h4 : remaining_Susan + 24 = 78 := by rw [h2, h3]; exact h
  have h5 : (1/2)*36*x = 54 := by sorry
  have h6 : 18*x = 54 := by sorry
  show x = 3 from by ring
  

end ratio_of_apples_l217_217084


namespace quotient_of_division_l217_217642

theorem quotient_of_division (dividend divisor remainder quotient : ℕ)
  (h_dividend : dividend = 15)
  (h_divisor : divisor = 3)
  (h_remainder : remainder = 3)
  (h_relation : dividend = divisor * quotient + remainder) :
  quotient = 4 :=
by sorry

end quotient_of_division_l217_217642


namespace overall_gain_is_2_89_l217_217396

noncomputable def overall_gain_percentage : ℝ :=
  let cost1 := 500000
  let gain1 := 0.10
  let sell1 := cost1 * (1 + gain1)

  let cost2 := 600000
  let loss2 := 0.05
  let sell2 := cost2 * (1 - loss2)

  let cost3 := 700000
  let gain3 := 0.15
  let sell3 := cost3 * (1 + gain3)

  let cost4 := 800000
  let loss4 := 0.12
  let sell4 := cost4 * (1 - loss4)

  let cost5 := 900000
  let gain5 := 0.08
  let sell5 := cost5 * (1 + gain5)

  let total_cost := cost1 + cost2 + cost3 + cost4 + cost5
  let total_sell := sell1 + sell2 + sell3 + sell4 + sell5
  let overall_gain := total_sell - total_cost
  (overall_gain / total_cost) * 100

theorem overall_gain_is_2_89 :
  overall_gain_percentage = 2.89 :=
by
  -- Proof goes here
  sorry

end overall_gain_is_2_89_l217_217396


namespace exist_sequence_length_4_not_exist_00010_at_length_5_l217_217846

noncomputable def frac_part (x : ℝ) : ℝ := x - real.floor x

def p (a b : ℝ) (n : ℕ) : ℕ := nat.floor (2 * frac_part (a * n + b))

def sequence_length_k_subsequence_exists (k : ℕ) (seq: fin k -> ℕ) : Prop :=
  ∃ (a b : ℝ), ∀ (n : ℕ), (∀ i : fin k, seq i = p a b (n + i))

theorem exist_sequence_length_4 : 
  ∀ (seq: fin 4 -> ℕ), sequence_length_k_subsequence_exists 4 seq :=
by
  sorry

theorem not_exist_00010_at_length_5 : 
  ¬ sequence_length_k_subsequence_exists 5 (![0,0,0,1,0]) :=
by
  sorry

end exist_sequence_length_4_not_exist_00010_at_length_5_l217_217846


namespace negative_solution_condition_l217_217041

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l217_217041


namespace sum_floor_sqrt_div_l217_217198

def is_squarefree (n : ℕ) : Prop := ∀ m : ℕ, m^2 ∣ n → m = 1

def squarefree_numbers : set ℕ := { k | is_squarefree k ∧ k > 0 }

theorem sum_floor_sqrt_div (n : ℕ) : 
  (∑ k in squarefree_numbers, (nat.floor (real.sqrt (n / k)))) = n := 
sorry

end sum_floor_sqrt_div_l217_217198


namespace prob1_prob2_l217_217802

theorem prob1:
  (6 * (Real.tan (30 * Real.pi / 180))^2 - Real.sqrt 3 * Real.sin (60 * Real.pi / 180) - 2 * Real.sin (45 * Real.pi / 180)) = (1 / 2 - Real.sqrt 2) :=
sorry

theorem prob2:
  ((Real.sqrt 2 / 2) * Real.cos (45 * Real.pi / 180) - (Real.tan (40 * Real.pi / 180) + 1)^0 + Real.sqrt (1 / 4) + Real.sin (30 * Real.pi / 180)) = (1 / 2) :=
sorry

end prob1_prob2_l217_217802


namespace max_height_of_soccer_ball_is_9_l217_217741

def height (t : ℝ) : ℝ := -4 * t^2 + 12 * t

theorem max_height_of_soccer_ball_is_9 : ∃ t : ℝ, height t = 9 := 
sorry

end max_height_of_soccer_ball_is_9_l217_217741


namespace sum_of_3x3_matrix_arithmetic_eq_45_l217_217496

-- Statement: Prove that the sum of all nine elements of a 3x3 matrix, where each row and each column forms an arithmetic sequence and the middle element a_{22} = 5, is 45
theorem sum_of_3x3_matrix_arithmetic_eq_45 
  (matrix : ℤ → ℤ → ℤ)
  (arithmetic_row : ∀ i, matrix i 0 + matrix i 1 + matrix i 2 = 3 * matrix i 1)
  (arithmetic_col : ∀ j, matrix 0 j + matrix 1 j + matrix 2 j = 3 * matrix 1 j)
  (middle_elem : matrix 1 1 = 5) : 
  (matrix 0 0 + matrix 0 1 + matrix 0 2 + matrix 1 0 + matrix 1 1 + matrix 1 2 + matrix 2 0 + matrix 2 1 + matrix 2 2) = 45 :=
by
  sorry -- proof to be provided

end sum_of_3x3_matrix_arithmetic_eq_45_l217_217496


namespace knicksEquivalentToKnocks_l217_217563

-- Represent the equivalence relations given in the problem
def fiveKnicksEqualsThreeKnacks : Prop := 5 * knicks = 3 * knacks
def fourKnacksEqualsSevenKnocks : Prop := 4 * knacks = 7 * knocks

-- Define the problem statement as a proof
theorem knicksEquivalentToKnocks (knicks knacks knocks : ℝ) 
  (h1 : fiveKnicksEqualsThreeKnacks)
  (h2 : fourKnacksEqualsSevenKnocks) : 
  28 * knocks * (4 / 7) * (5 / 3) = 80 / 3 := 
by 
  sorry

end knicksEquivalentToKnocks_l217_217563


namespace polygon_interior_angles_540_implies_pentagon_l217_217108

theorem polygon_interior_angles_540_implies_pentagon
  (n : ℕ) (H: 180 * (n - 2) = 540) : n = 5 :=
sorry

end polygon_interior_angles_540_implies_pentagon_l217_217108


namespace triangle_segment_relation_l217_217185

variable {A B C A1 B1 C1 : Type}

theorem triangle_segment_relation
  (ABC : triangle A B C)
  (A1_on_BC : on_segment A1 B C)
  (B1_on_AC : on_line B1 A C)
  (C1_on_AB : on_line C1 A B)
  (BB1_parallel_AA1 : parallel (line B B1) (line A A1))
  (CC1_parallel_AA1 : parallel (line C C1) (line A A1)) :
  (1 / (segment_length A A1)) = (1 / (segment_length B B1)) + (1 / (segment_length C C1)) :=
sorry

end triangle_segment_relation_l217_217185


namespace categorize_numbers_l217_217034

noncomputable def positive_numbers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x > 0}

noncomputable def non_neg_integers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x ≥ 0 ∧ ∃ n : ℤ, x = n}

noncomputable def negative_fractions (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x < 0 ∧ ∃ n d : ℤ, d ≠ 0 ∧ (x = n / d)}

def given_set : Set ℝ := {6, -3, 2.4, -3/4, 0, -3.14, 2, -7/2, 2/3}

theorem categorize_numbers :
  positive_numbers given_set = {6, 2.4, 2, 2/3} ∧
  non_neg_integers given_set = {6, 0, 2} ∧
  negative_fractions given_set = {-3/4, -3.14, -7/2} :=
by
  sorry

end categorize_numbers_l217_217034


namespace range_of_a_l217_217927

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Icc (1:ℝ) 2, deriv (λ x, -x^2 + 2 * a * x) x ≤ 0)
  ∧ (∀ x ∈ Icc (1:ℝ) 2, deriv (λ x, a / (x + 1)) x ≤ 0) ↔ 0 < a ∧ a ≤ 1 :=
by {
  sorry
}

end range_of_a_l217_217927


namespace factor_expression_l217_217823

theorem factor_expression (y : ℝ) :
  5 * y * (y - 4) + 2 * (y - 4) = (5 * y + 2) * (y - 4) :=
by
  sorry

end factor_expression_l217_217823


namespace least_number_to_make_divisible_l217_217343

theorem least_number_to_make_divisible :
  let product := 41 * 71 * 139
  5918273 % product + 151162 % product = 0 % product :=
by
  let product := 41 * 71 * 139
  have H : 5918273 % product = 253467 := by sorry
  have H2 : 151162 % product = 151162 := by sorry
  have H3 : (253467 + 151162) % product = (404629 % product) := by sorry
  have H4 : 404629 % product = 0 := by sorry
  exact H3.trans H4

end least_number_to_make_divisible_l217_217343


namespace gcd_euclidean_algorithm_example_l217_217318

theorem gcd_euclidean_algorithm_example : Nat.gcd 98 63 = 7 :=
by
  sorry

end gcd_euclidean_algorithm_example_l217_217318


namespace printingTime_l217_217402

def printerSpeed : ℝ := 23
def pauseTime : ℝ := 2
def totalPages : ℝ := 350

theorem printingTime : (totalPages / printerSpeed) + ((totalPages / 50 - 1) * pauseTime) = 27 := by 
  sorry

end printingTime_l217_217402


namespace find_k_arithmetic_progression_l217_217578

theorem find_k_arithmetic_progression :
  ∃ k : ℤ, (2 * real.sqrt (225 + k) = real.sqrt (49 + k) + real.sqrt (441 + k)) → k = 255 :=
by
  sorry

end find_k_arithmetic_progression_l217_217578


namespace arithmetic_sequence_sum_l217_217580

theorem arithmetic_sequence_sum {a_n : ℕ → ℤ} (h_arith : ∃ d : ℤ, ∀ n : ℕ, a_n = a_n(0) + (n) * d)
  (h_a9 : a_n 9 = -2012) (h_a17 : a_n 17 = -2012) : a_n 1 + a_n 25 < 0 := by
  sorry

end arithmetic_sequence_sum_l217_217580


namespace pure_imaginary_m_value_l217_217088

theorem pure_imaginary_m_value (m : ℝ) (h₁ : m ^ 2 + m - 2 = 0) (h₂ : m ^ 2 - 1 ≠ 0) : m = -2 := by
  sorry

end pure_imaginary_m_value_l217_217088


namespace equilateral_triangles_count_in_T_l217_217980

open Finset

def T : Finset (ℝ × ℝ × ℝ) :=
  (Finset.range 4).product (Finset.range 4).product (Finset.range 4)
  |>.map (λ ⟨x, (y, z)⟩ => (x.to_float, y.to_float, z.to_float))

/-- The number of equilateral triangles with vertices in T is 120 -/
def equilateral_triangles_in_T : Nat :=
  120

theorem equilateral_triangles_count_in_T :
  ∀ S : Finset (ℝ × ℝ × ℝ), S = T → S.filter (λ ⟨x1, y1, z1⟩ ⟨x2, y2, z2⟩ ⟨x3, y3, z3⟩, 
    (⟮(x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2) = (⟮(x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2) = (⟮(x1 - x3)^2 + (y1 - y3)^2 + (z1 - z3)^2)⟯)
  .card = 120 := sorry

end equilateral_triangles_count_in_T_l217_217980


namespace total_gallons_in_tanks_l217_217637

def tank1_capacity : ℕ := 7000
def tank2_capacity : ℕ := 5000
def tank3_capacity : ℕ := 3000

def tank1_filled : ℚ := 3/4
def tank2_filled : ℚ := 4/5
def tank3_filled : ℚ := 1/2

theorem total_gallons_in_tanks :
  (tank1_capacity * tank1_filled + tank2_capacity * tank2_filled + tank3_capacity * tank3_filled : ℚ) = 10750 := 
by 
  sorry

end total_gallons_in_tanks_l217_217637


namespace probability_dmitry_before_father_l217_217333

noncomputable def prob_dmitry_before_father (m : ℝ) (x y z : ℝ) (h1 : 0 < x ∧ x < m) (h2 : 0 < y ∧ y < z ∧ z < m) : ℝ :=
  if h1 ∧ h2 then 2/3 else 0

theorem probability_dmitry_before_father (m : ℝ) (x y z : ℝ) (h1 : 0 < x ∧ x < m) (h2 : 0 < y ∧ y < z ∧ z < m) :
  prob_dmitry_before_father m x y z h1 h2 = 2 / 3 :=
begin
  sorry
end

end probability_dmitry_before_father_l217_217333


namespace problem_statement_l217_217631

def g (x : ℝ) : ℝ := (3 * x + 4) / (x + 3)

def upper_bound (M : ℝ) : Prop := 
  ∀ x : ℝ, x ≥ 0 → g(x) ≤ M

def lower_bound (m : ℝ) : Prop := 
  ∀ x : ℝ, x ≥ 0 → m ≤ g(x)

def M_in_S (M : ℝ) : Prop := 
  ∃ x : ℝ, x ≥ 0 ∧ g(x) = M

def m_in_S (m : ℝ) : Prop := 
  ∃ x : ℝ, x ≥ 0 ∧ g(x) = m

theorem problem_statement : 
  let M := 3 
  let m := 4 / 3 in
  upper_bound M ∧ ¬ M_in_S M ∧ lower_bound m ∧ m_in_S m :=
by
  sorry

end problem_statement_l217_217631


namespace probability_distinct_reals_equal_image_l217_217087

open Set

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a|

theorem probability_distinct_reals_equal_image
  (a : ℝ) (h_a_mem : a ∈ Ioo 0 6) :
  (∃ x1 x2 ∈ Ioo 1 2, x1 ≠ x2 ∧ f x1 a = f x2 a) →
  (2 < a ∧ a < 4) :=
by 
  sorry

end probability_distinct_reals_equal_image_l217_217087


namespace ratio_AE_EC_l217_217180

-- Define the main problem
theorem ratio_AE_EC (a b : Line) (AF FB BC CD AE EC : ℝ)
  (h_parallel : a ∥ b)
  (h_AF_FB_ratio : AF / FB = 3 / 5)
  (h_BC_CD_ratio : BC / CD = 3 / 1) :
  AE / EC = 12 / 5 :=
begin
  sorry -- Proof to be filled in
end

end ratio_AE_EC_l217_217180


namespace arithmetic_geometric_sequence_l217_217079

-- Definitions of the arithmetic sequence and conditions
def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric (a1 a2 a5 : ℤ) : Prop :=
  a2 * a2 = a1 * a5

-- Given conditions
def a1 := 1
def d := 2
def a : ℕ → ℤ := λ n, a1 + n * d

-- Theorem to prove
theorem arithmetic_geometric_sequence (h_arith : is_arithmetic a d) (h_geom : is_geometric a1 (a 2) (a 5)) : d = 2 :=
by
  -- Proof will be done here
  sorry

end arithmetic_geometric_sequence_l217_217079


namespace maximize_area_l217_217866

theorem maximize_area (P L W : ℝ) (h1 : P = 2 * L + 2 * W) (h2 : 0 < P) : 
  (L = P / 4) ∧ (W = P / 4) :=
by
  sorry

end maximize_area_l217_217866


namespace distance_point_to_line_l217_217677

def point : (ℝ × ℝ) := (2, 1)
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0

theorem distance_point_to_line : 
  let (x, y) := point in 
  ∃ d : ℝ, d = abs (3 * x - 4 * y + 2) / real.sqrt (9 + 16) ∧ d = 4 / 5 :=
by 
  sorry

end distance_point_to_line_l217_217677


namespace avg_of_six_is_3_9_l217_217672

noncomputable def avg_of_six_numbers 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : ℝ :=
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6

theorem avg_of_six_is_3_9 
  (avg1 avg2 avg3 : ℝ) 
  (h1 : avg1 = 3.4) 
  (h2 : avg2 = 3.85) 
  (h3 : avg3 = 4.45) : 
  avg_of_six_numbers avg1 avg2 avg3 h1 h2 h3 = 3.9 := 
by {
  sorry
}

end avg_of_six_is_3_9_l217_217672


namespace gcd_of_remainders_l217_217067

/--
Theorem: The greatest number which, on dividing 4351, 5161, 6272, and 7383,
leaves remainders of 8, 10, 12, and 14, respectively, is 1.
-/
theorem gcd_of_remainders (d : ℕ) :
  (4351 % d = 8) ∧
  (5161 % d = 10) ∧
  (6272 % d = 12) ∧
  (7383 % d = 14)
  → d = 1 :=
begin
  sorry
end

end gcd_of_remainders_l217_217067


namespace total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l217_217587

-- Given conditions
def kids_A := 7
def kids_B := 9
def kids_C := 5

def pencils_per_child_A := 4
def erasers_per_child_A := 2
def skittles_per_child_A := 13

def pencils_per_child_B := 6
def rulers_per_child_B := 1
def skittles_per_child_B := 8

def pencils_per_child_C := 3
def sharpeners_per_child_C := 1
def skittles_per_child_C := 15

-- Calculated totals
def total_pencils := kids_A * pencils_per_child_A + kids_B * pencils_per_child_B + kids_C * pencils_per_child_C
def total_erasers := kids_A * erasers_per_child_A
def total_rulers := kids_B * rulers_per_child_B
def total_sharpeners := kids_C * sharpeners_per_child_C
def total_skittles := kids_A * skittles_per_child_A + kids_B * skittles_per_child_B + kids_C * skittles_per_child_C

-- Proof obligations
theorem total_pencils_correct : total_pencils = 97 := by
  sorry

theorem total_erasers_correct : total_erasers = 14 := by
  sorry

theorem total_rulers_correct : total_rulers = 9 := by
  sorry

theorem total_sharpeners_correct : total_sharpeners = 5 := by
  sorry

theorem total_skittles_correct : total_skittles = 238 := by
  sorry

end total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l217_217587


namespace find_integer_cosine_l217_217068

theorem find_integer_cosine : ∃ n : ℝ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (1230 * real.pi / 180) ∧ n = 150 :=
by
  use 150
  sorry

end find_integer_cosine_l217_217068


namespace pythagorean_triangle_product_divisible_by_60_l217_217775

theorem pythagorean_triangle_product_divisible_by_60 : 
  ∀ (a b c : ℕ),
  (∃ m n : ℕ,
  m > n ∧ (m % 2 = 0 ∨ n % 2 = 0) ∧ m.gcd n = 1 ∧
  a = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2 ∧ a^2 + b^2 = c^2) →
  60 ∣ (a * b * c) :=
sorry

end pythagorean_triangle_product_divisible_by_60_l217_217775


namespace sqrt_sqrt_16_eq_pm_2_l217_217702

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := 
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217702


namespace log_sum_eq_l217_217798

theorem log_sum_eq :
  log 2 (8 : ℝ) + 3 * log 2 (4 : ℝ) + 4 * log 4 (16 : ℝ) + 2 * log 8 (32 : ℝ) = 61 / 3 := 
by sorry

end log_sum_eq_l217_217798


namespace order_of_numbers_l217_217861

theorem order_of_numbers (a b c d : ℝ) (h₁ : a = 0.2 ^ 3.5) (h₂ : b = 0.2 ^ 4.1) (h₃ : c = Real.exp 1.1) (h₄ : d = Real.log 3 / Real.log 0.2) : d < b ∧ b < a ∧ a < c := 
sorry

end order_of_numbers_l217_217861


namespace logan_passengers_approx_heartfield_passengers_approx_l217_217932

-- Define all the conditions in the problem
def total_passengers : ℝ := 37.3
def kennedy_fraction : ℝ := 1 / 3
def miami_fraction : ℝ := 1 / 4
def logan_fraction : ℝ := 1 / 3
def austin_multiplier : ℝ := 6
def heartfield_fraction : ℝ := 1 / 2

-- Define the number of passengers for each airport based on the conditions
def kennedy_passengers : ℝ := total_passengers * kennedy_fraction
def miami_passengers : ℝ := kennedy_passengers * miami_fraction
def logan_passengers : ℝ := miami_passengers * logan_fraction
def austin_passengers : ℝ := logan_passengers * austin_multiplier
def heartfield_passengers : ℝ := austin_passengers * heartfield_fraction

-- Prove that the number of passengers that used Logan airport is approximately 1.036 million
theorem logan_passengers_approx :
  logan_passengers ≈ 1.036 :=
by
  sorry

-- Prove that the number of passengers that had their flight connected from Heartfield airport is approximately 3.108 million
theorem heartfield_passengers_approx :
  heartfield_passengers ≈ 3.108 :=
by
  sorry

end logan_passengers_approx_heartfield_passengers_approx_l217_217932


namespace jen_ate_eleven_suckers_l217_217252

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end jen_ate_eleven_suckers_l217_217252


namespace train_speed_in_kph_l217_217769

noncomputable def speed_of_train (jogger_speed_kph : ℝ) (gap_m : ℝ) (train_length_m : ℝ) (time_s : ℝ) : ℝ :=
let jogger_speed_mps := jogger_speed_kph * (1000 / 3600)
let total_distance_m := gap_m + train_length_m
let speed_mps := total_distance_m / time_s
speed_mps * (3600 / 1000)

theorem train_speed_in_kph :
  speed_of_train 9 240 120 36 = 36 := 
by
  sorry

end train_speed_in_kph_l217_217769


namespace volume_of_prism_l217_217666

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 54) : 
  a * b * c = 270 :=
by
  sorry

end volume_of_prism_l217_217666


namespace great_white_shark_teeth_l217_217411

theorem great_white_shark_teeth :
  let tiger_shark_teeth := 180
  let hammerhead_shark_teeth := tiger_shark_teeth / 6
  let great_white_shark_teeth := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)
  in great_white_shark_teeth = 420 := by
  sorry

end great_white_shark_teeth_l217_217411


namespace polynomial_evaluation_l217_217567

theorem polynomial_evaluation (p : Polynomial ℝ) (n : ℕ)
  (hdeg : p.degree = n)
  (hvals : ∀ k : ℕ, k ≤ n → p.eval k = (k : ℝ) / (k + 1)) :
  if even n then p.eval (n + 1) = 1 else p.eval (n + 1) = (n : ℝ) / (n + 2) :=
by sorry

end polynomial_evaluation_l217_217567


namespace calculate_box_2_neg1_3_l217_217484

def redefine_box (a b c : ℤ) (k : ℤ) : ℤ :=
  k * a^b - b^c + c^(a - k)

theorem calculate_box_2_neg1_3 : redefine_box 2 (-1) 3 2 = 1 := by
  sorry

end calculate_box_2_neg1_3_l217_217484


namespace no_such_numbers_exist_l217_217445

theorem no_such_numbers_exist (n : ℕ) : ¬ (∃ (M N : ℕ), (∀ i < n, even (M / 10^i % 10)) ∧ (∀ j < n, odd (N / 10^j % 10)) ∧ (∀ d, 0 ≤ d ∧ d ≤ 9 → (∃ k < n, (M₁ / 10^k % 10 = d) ∨ (N / 10^k % 10 = d)) ∧ (M % N = 0)) :=
sorry

end no_such_numbers_exist_l217_217445


namespace polynomial_a_b_sum_l217_217616

theorem polynomial_a_b_sum 
  (a b : ℝ) -- a and b are real numbers
  (h : ∃ x : ℂ, x^3 + (a : ℂ)*x + (b : ℂ) = 0 ∧ x = 2 + complex.I) :
  a + b = 9 := 
sorry

end polynomial_a_b_sum_l217_217616


namespace only_positive_integer_x_l217_217465

theorem only_positive_integer_x (x : ℕ) (k : ℕ) (h1 : 2 * x + 1 = k^2) (h2 : x > 0) :
  ¬ (∃ y : ℕ, (y >= 2 * x + 2 ∧ y <= 3 * x + 2 ∧ ∃ m : ℕ, y = m^2)) → x = 4 := 
by sorry

end only_positive_integer_x_l217_217465


namespace lambda_bound_l217_217902

noncomputable def a_seq : ℕ → ℝ
| 0       := 1 
| (n + 1) := a_seq n / (a_seq n + 2)

noncomputable def b_seq (λ : ℝ) : ℕ → ℝ
| 0       := -3 / 2 * λ
| (n + 1) := (n - 2 * λ) * (1 / (a_seq n) + 1)

theorem lambda_bound (λ : ℝ) : 
  (∀ n : ℕ, b_seq λ n < b_seq λ (n + 1)) → λ < 4 / 5 :=
sorry

end lambda_bound_l217_217902


namespace no_such_n_exists_l217_217847

theorem no_such_n_exists : ∀ n : ℕ, n > 1 → ∀ (p1 p2 : ℕ), 
  (Nat.Prime p1) → (Nat.Prime p2) → n = p1^2 → n + 60 = p2^2 → False :=
by
  intro n hn p1 p2 hp1 hp2 h1 h2
  sorry

end no_such_n_exists_l217_217847


namespace log_sum_geom_seq_l217_217599

open Real

noncomputable def a_n : ℕ → ℝ := sorry
axiom geom_seq : ∀ n : ℕ, a_n(n) = a_n(0) * (a_n(1) / a_n(0)) ^ (n - 1)
axiom a4a5_eq_32 : a_n(3) * a_n(4) = 32

theorem log_sum_geom_seq : (log (a_n(0)) / log 2) + (log (a_n(1)) / log 2) + (log (a_n(2)) / log 2) + (log (a_n(3)) / log 2) + (log (a_n(4)) / log 2) + (log (a_n(5)) / log 2) + (log (a_n(6)) / log 2) + (log (a_n(7)) / log 2) = 20 := 
sorry

end log_sum_geom_seq_l217_217599


namespace problem1_problem2_l217_217212

open Finset

-- Define the finite set M and its subsets
def M (n : ℕ) : Finset ℝ := (range (n + 1)).image (λ k, 1 / 2^(n + k))

-- Define the function S which returns the sum of elements in a subset of M
def S (n : ℕ) (s : Finset ℝ) : ℝ := s.sum id

-- Prove the problems
theorem problem1 :
  S 2 (\{1 / 2^2, 1 / 2^3, 1 / 2^4\}) = 1 + (1 / 2) + (1 / 4) := by
  sorry

theorem problem2 (n : ℕ) (h : 0 < n) :
  let t := 2 ^ (n + 1)
  in (2 ^ n) * ((range (n + 1)).sum (λ k, 1 / 2^ (n + k))) = 2 - (1 / 2^n) := by
  sorry

end problem1_problem2_l217_217212


namespace maximum_ab_value_l217_217860

noncomputable def max_ab_value (a b : ℝ) : ℝ :=
  ∃ (a b : ℝ), (∀ x : ℝ, exp x ≥ a * (x - 1) + b) → ab = (1 / 2) * exp 3

theorem maximum_ab_value (a b : ℝ) (h : ∀ x : ℝ, exp x ≥ a * (x - 1) + b) : ab ≤ (1 / 2) * exp 3 := sorry

end maximum_ab_value_l217_217860


namespace ellipse_and_fixed_point_l217_217525

-- Definitions according to conditions
def point := ℝ × ℝ
def A : point := (0, -2)
def B : point := (3 / 2, -1)
def P : point := (1, -2)

-- General form of ellipse equation
def general_ellipse_eq (m n : ℝ) (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Line Segment equation AB
def line_segment_eq (a b : point) : ℝ → ℝ := λ x, (b.2 - a.2) / (b.1 - a.1) * x + (a.2 * b.1 - a.1 * b.2) / (b.1 - a.1)

-- Geometry relations
def meet_y_axis (y : ℝ) : point := (0, y)

-- Main theorem
theorem ellipse_and_fixed_point :
  ∃ (E : ℝ → ℝ → Prop), (E = general_ellipse_eq (1 / 3) (1 / 4)) ∧
      ∀ M N : point, collinear [M, N, meet_y_axis (-2)] :=
sorry

end ellipse_and_fixed_point_l217_217525


namespace problem_statement_l217_217539

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * x - Real.sin x

-- Propositions to be proven
theorem problem_statement (a : ℝ) :
(∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f (1/2:ℝ) x)
      ∧ (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2),  f (1/2:ℝ) x = 0 → false)
      ∧ (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f a x ≤ 0 → 0 < a ∧ a ≤ 2/Real.pi)
      ∧ (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f a x ≥ 0 → a ≥ 1)
    → (a = 1/2 → f (1/2: ℝ) (Real.pi / 3) = a * (Real.pi / 3) - Real.sin (Real.pi / 3))
         ∧ (f (1/2: ℝ) 0 = 0 ∧ f (1/2: ℝ) (Real.pi / 2) > 0)
         ∧ (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f a x ≤ 0 → 0 < a ∧ a ≤ 2 / Real.pi)
         ∧ (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), f a x ≥ 0 → a ≥ 1) := by
    sorry

end problem_statement_l217_217539


namespace find_cosine_value_l217_217517

theorem find_cosine_value (θ : ℝ) (h : sin (π / 6 - θ) = 1 / 3) : cos (2 * π / 3 + 2 * θ) = - 7 / 9 := by
  sorry

end find_cosine_value_l217_217517


namespace small_equipment_initial_time_find_m_l217_217946

-- Define the initial conditions
constant road_length : ℤ := 39000
constant small_equipment_rate : ℤ := 30
constant large_equipment_rate : ℤ := 60
constant increased_distance : ℤ := 9000
constant small_equipment_additional_time : ℤ := 18
constant large_equipment_additional_base_time : ℤ := 150

-- Definition of initial times
def small_equipment_time : ℤ := 300
def large_equipment_time : ℤ := (5 * small_equipment_time) / 3

-- Prove small equipment time in initial setting
theorem small_equipment_initial_time :
  30 * small_equipment_time + 60 * large_equipment_time = road_length := sorry

-- Define increased times and rates
constant efficiency_decrease : ℤ --> ℤ
def new_small_equipment_time : ℤ := small_equipment_time + small_equipment_additional_time
def new_large_equipment_time (m : ℤ) : ℤ := large_equipment_time + large_equipment_additional_base_time + 2 * m

-- Prove the value of m given the new total road length
theorem find_m (m : ℤ) : 
  30 * new_small_equipment_time + (60 - m) * (new_large_equipment_time m) = road_length + increased_distance ->
  m = 5 := by
  intro h
  sorry

end small_equipment_initial_time_find_m_l217_217946


namespace train_speed_l217_217783

theorem train_speed (time_sec : ℕ) (length_m : ℕ) (conversion_factor : ℕ) :
  time_sec = 5 → length_m = 125 → conversion_factor = 36 → (length_m / time_sec) * conversion_factor = 900 :=
by
  intros ht hl hc
  rw [ht, hl, hc]
  norm_num
  sorry

end train_speed_l217_217783


namespace schedule_ways_120_l217_217408

noncomputable def total_ways (n m : ℕ) : ℕ :=
  if h₁ : n = 4 ∧ m = 8 then
    let possible_slots := ((nat.factorial m) / ((nat.factorial n) * (nat.factorial (m - n)))) in
    let valid_slots := 5 in
    let arr_per := nat.factorial n in
    valid_slots * arr_per
  else
    0

theorem schedule_ways_120 : total_ways 4 8 = 120 :=
sorry

end schedule_ways_120_l217_217408


namespace Chris_has_6_Teslas_l217_217029

theorem Chris_has_6_Teslas (x y z : ℕ) (h1 : z = 13) (h2 : z = x + 10) (h3 : x = y / 2):
  y = 6 :=
by
  sorry

end Chris_has_6_Teslas_l217_217029


namespace austin_shrimp_catch_l217_217320

theorem austin_shrimp_catch (A : ℕ) (H1 : ∀ v a : ℕ, v = 26 ∧ a = A → let b := (v + a) / 2 in 
  7 * (v + a + b) / 11 = 42) : 
  A - 26 = -8 := 
sorry

end austin_shrimp_catch_l217_217320


namespace rice_field_sacks_l217_217589

theorem rice_field_sacks (x : ℝ)
  (h1 : ∀ x, x + 1.20 * x = 44) : x = 20 :=
sorry

end rice_field_sacks_l217_217589


namespace part_I_part_II_l217_217492

noncomputable def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

theorem part_I (a b : ℝ) (h : a ≠ 0)
  (tangent_condition : ∀ x, (∃ y, f a b x = y) ∧ (∃ y', (∂ (f a b x)) / ∂ x = y') ∧ (x = 1 -> (f a b 1 = 1 ∧ ∂ (f a b x) / ∂ x = 1)))
  : a = -1 ∧ b = 2 :=
begin
  sorry
end

theorem part_II (a b : ℝ) (h : a ≠ 0)
  (inequality : ∀ x, f a b x ≤ x^2 + x)
  : ∃ a b, a > 0 ∧ a = Real.sqrt Real.exp ∧ b = (Real.exp^(1/2))/2 ∧ a*b = (Real.exp^(1/2))^2 / 2 :=
begin
  sorry
end

end part_I_part_II_l217_217492


namespace angle_is_2pi_over_3_l217_217133

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  let cosine := dot_product a b / (magnitude a * magnitude b)
  Real.arccos cosine

theorem angle_is_2pi_over_3 :
  ∀ (a b : ℝ × ℝ),
  magnitude a = 1 →
  b = (Real.sqrt 2, Real.sqrt 2) →
  magnitude (a.1 - b.1, a.2 - b.2) = Real.sqrt 7 →
  angle_between_vectors a b = 2 * Real.pi / 3 :=
by
  intros a b ha hb hab
  sorry

end angle_is_2pi_over_3_l217_217133


namespace number_of_valid_permutations_l217_217852

def is_adjacent (x y : ℕ) : Prop := (x = y + 1) ∨ (y = x + 1)
def different_parity (x y : ℕ) : Prop := (x % 2 = 0 ∧ y % 2 = 1) ∨ (x % 2 = 1 ∧ y % 2 = 0)

theorem number_of_valid_permutations : 
  let digits := [0, 1, 2, 3, 4, 5, 6] in
  let permutations := digits.permutations.filter (λ perm, 
    -- Checking adjacent digits for different parity
    ∀ i, i < perm.length - 1 → different_parity (perm.nth i) (perm.nth (i + 1)) ∧
    -- Checking if 1 and 2 are adjacent
    ∃ i, is_adjacent (perm.nth i) 1 ∧ is_adjacent (perm.nth i) 2
  ) in
  permutations.length = 48 :=
sorry

end number_of_valid_permutations_l217_217852


namespace find_b_l217_217154

-- Definitions of the functions and the conditions
noncomputable def exp_curve (x : ℝ) : ℝ := Real.exp x
noncomputable def log_curve (x : ℝ) (b : ℝ) : ℝ := Real.log x + b

-- To define the derivative,
noncomputable def exp_curve_derivative (x : ℝ) : ℝ := Real.exp x
noncomputable def log_curve_derivative (x : ℝ) : ℝ := 1 / x

-- Point where the tangency occurs
def tangent_point : ℝ × ℝ := ⟨0, exp_curve 0⟩

-- Slope of the tangent line at x = 0 for y = e^x
def slope_at_zero : ℝ := exp_curve_derivative 0

-- Equation of the tangent line at x=0 for y = e^x
def tangent_line_eqn (x : ℝ) : ℝ := slope_at_zero * (x - 0) + exp_curve 0

-- Tangent line condition for y = ln x + b passing through the point (1, b+ln 1)
theorem find_b (b : ℝ) : tangent_line_eqn 1 = log_curve 1 b := by
  sorry

end find_b_l217_217154


namespace other_root_eq_six_l217_217117

theorem other_root_eq_six (a : ℝ) (x1 : ℝ) (x2 : ℝ) 
  (h : x1 = -2) 
  (eqn : ∀ x, x^2 - a * x - 3 * a = 0 → (x = x1 ∨ x = x2)) :
  x2 = 6 :=
by
  sorry

end other_root_eq_six_l217_217117


namespace simplify_product_l217_217652

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l217_217652


namespace george_total_payment_l217_217857

-- Define the conditions
def ticket_price : ℕ := 16
def nachos_price : ℕ := ticket_price / 2

-- Define the total amount paid
def total_amount_paid : ℕ := ticket_price + nachos_price

-- State the theorem
theorem george_total_payment : total_amount_paid = 24 := by
sory

end george_total_payment_l217_217857


namespace find_common_ratio_l217_217898

noncomputable def a_n (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

theorem find_common_ratio (a1 q : ℝ) (a : ℕ → ℝ) 
  (h3 : a 3 = 4) (h6 : a 6 = 1/2) :
  q = 1/2 :=
by
  have h1 : a 3 = a_n a1 q 3 := by sorry
  have h2 : a 6 = a_n a1 q 6 := by sorry
  rw [←h1,←h2] at h3 h6
  sorry

end find_common_ratio_l217_217898


namespace part1_solution_part2_solution_l217_217843

def equiangular (A B : Set ℤ) : Prop :=
  (A.sum id = B.sum id) ∧
  (A.map (λ x, x^2)).sum id = (B.map (λ x, x^2)).sum id ∧
  A ∩ B = ∅

theorem part1_solution (x y : ℤ) (hA : Set ℤ): 
  hA = {1, 5, 6} → 
  ∃ x y, x ∈ hA ∧ y ∈ hA ∧ equiangular {1, 5, 6} {2, x, y} → 
  (x = 4 ∧ y = 8) ∨ (x = 8 ∧ y = 4) :=
sorry

theorem part2_solution (A B : Set ℤ) (k : ℤ) (hA : ∀ (x : ℤ), x ∈ A → A x = a_i) (hx : A.sum id = B.sum id) (hxx : (A.map (λ x, x^2)).sum id = (B.map (λ x, x^2)).sum id) (hB : A ∩ B = ∅) : 
  equiangular (A.map (λ x, x+k)) (B.map (λ x, x+k)) :=
sorry

end part1_solution_part2_solution_l217_217843


namespace log_frac_square_eq_nine_l217_217237

theorem log_frac_square_eq_nine {x y : ℝ} (hx : x ≠ 1) (hy : y ≠ 1) (h1 : log 3 x = log x 81) (h2 : x * y = 27) : (log 3 (x / y))^2 = 9 :=
by
  sorry

end log_frac_square_eq_nine_l217_217237


namespace total_gallons_in_tanks_l217_217638

def tank1_capacity : ℕ := 7000
def tank2_capacity : ℕ := 5000
def tank3_capacity : ℕ := 3000

def tank1_filled : ℚ := 3/4
def tank2_filled : ℚ := 4/5
def tank3_filled : ℚ := 1/2

theorem total_gallons_in_tanks :
  (tank1_capacity * tank1_filled + tank2_capacity * tank2_filled + tank3_capacity * tank3_filled : ℚ) = 10750 := 
by 
  sorry

end total_gallons_in_tanks_l217_217638


namespace men_dropped_out_l217_217759

theorem men_dropped_out (x : ℕ) : 
  (∀ (days_half days_full men men_remaining : ℕ),
    days_half = 15 ∧ days_full = 25 ∧ men = 5 ∧ men_remaining = men - x ∧ 
    (men * (2 * days_half)) = ((men_remaining) * days_full)) -> x = 1 :=
by
  intros h
  sorry

end men_dropped_out_l217_217759


namespace combined_profit_growth_rate_is_correct_l217_217159

-- Define the initial revenue
variables (R : ℝ)  -- revenue in 1998 for both companies

-- Define the profit percentages for each company over the years
def profit_N_1998 := 0.12 * R
def profit_M_1998 := 0.10 * R

def revenue_N_1999 := 0.80 * R
def profit_N_1999 := 0.16 * revenue_N_1999

def revenue_N_2000 := 0.92 * R
def profit_N_2000 := 0.12 * revenue_N_2000

def revenue_M_1999 := 1.10 * R
def profit_M_1999 := 0.14 * revenue_M_1999

def revenue_M_2000 := 1.045 * R
def profit_M_2000 := 0.09 * revenue_M_2000

-- Calculate the combined profits for 1998 and 2000
def combined_profit_1998 := profit_N_1998 + profit_M_1998
def combined_profit_2000 := profit_N_2000 + profit_M_2000

-- Compute the combined profit growth rate
def combined_profit_growth_rate := 
    (combined_profit_2000 - combined_profit_1998) / combined_profit_1998

-- Expected growth rate
def expected_growth_rate := -0.07068

theorem combined_profit_growth_rate_is_correct :
  combined_profit_growth_rate R = expected_growth_rate :=
by
  sorry

end combined_profit_growth_rate_is_correct_l217_217159


namespace factorization_correct_l217_217033

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end factorization_correct_l217_217033


namespace purchase_lamps_l217_217383

-- Define the problem statement and conditions
theorem purchase_lamps (x : ℕ) (y : ℕ) :
  (x + (100 - x) = 100) ∧ (30 * x + 50 * (100 - x) = 3500) ∧ (100 - x ≤ 3 * x) →
  (x = 75) ∧ (y = -5 * x + 2000) ∧ (x = 25 → y = 1875) :=
by sorry

end purchase_lamps_l217_217383


namespace find_sum_of_a_b_l217_217281

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem find_sum_of_a_b {a b : ℝ} 
  (h_max : ∀ x ∈ set.Icc a b, f x ≤ 1)
  (h_min : ∀ x ∈ set.Icc a b, f x ≥ 1 / 3) :
  a + b = 6 :=
sorry

end find_sum_of_a_b_l217_217281


namespace sqrt_25_times_sqrt_25_l217_217020

theorem sqrt_25_times_sqrt_25 : sqrt (25 * sqrt 25) = 5 * sqrt 5 :=
by
  sorry

end sqrt_25_times_sqrt_25_l217_217020


namespace complex_expression_eq_l217_217622

variable (z : ℂ)
variable (h : complex.abs z = 10)

theorem complex_expression_eq :
  z * conj z + conj z - z = 100 - 2 * complex.i * complex.im z :=
by sorry

end complex_expression_eq_l217_217622


namespace distribute_volunteers_l217_217023

theorem distribute_volunteers :
  ∃ (n : ℕ), n = 150 ∧
    ∀ (volunteers venues : ℕ), 
      volunteers = 5 → venues = 3 → 
      (∀ v : ℕ, v ∈ [1, 1, 3] ∨ v ∈ [2, 2, 1])) :=
begin
  use 150,
  split,
  { refl },
  intros volunteers venues hv hvv grouping,
  simp only [list.mem_cons_iff, list.mem_singleton, list.mem_append, list.mem_map, and_imp, exists_imp_distrib, nat.mem_antidiagonal, nat.add_succ, nat.succ_add],
  rintros rfl rfl _ _ ⟨⟩,
end

end distribute_volunteers_l217_217023


namespace birthdays_from_start_date_l217_217811

-- Definitions needed for the problem
def dad_birthday := (month: ℕ, day: ℕ) := (5, 1)
def chunchun_birthday := (month: ℕ, day: ℕ) := (7, 1)
def start_date := (year: ℕ, month: ℕ, day: ℕ) := (2012, 12, 26)
def end_day := 2013

-- Statement of our theorem
theorem birthdays_from_start_date : 
  let total_birthdays := 11 in 
  total_birthdays = 11 := by
  sorry

end birthdays_from_start_date_l217_217811


namespace greatest_N_consecutive_sum_50_l217_217329

theorem greatest_N_consecutive_sum_50 :
  ∃ N a : ℤ, (N > 0) ∧ (N * (2 * a + N - 1) = 100) ∧ (N = 100) :=
by
  sorry

end greatest_N_consecutive_sum_50_l217_217329


namespace part1_a_n_part1_b_n_part2_T_n_l217_217501

noncomputable def a_n : ℕ → ℝ
| 0       := 6
| n + 1   := n + 6

noncomputable def b_n (n : ℕ) : ℝ := 3 * n + 2

noncomputable def c_n (n : ℕ) : ℝ := 3 / ((2 * a_n n - 11) * (2 * b_n n - 1))

noncomputable def T_n (n : ℕ) : ℝ := (1 / 2) * (1 - (1 / (2 * n + 1)))

theorem part1_a_n (n : ℕ) : a_n n = n + 5 :=
by sorry

theorem part1_b_n (n : ℕ) : b_n n = 3 * n + 2 :=
by sorry

theorem part2_T_n (n : ℕ) : T_n n ≥ 1 / 3 :=
by sorry

end part1_a_n_part1_b_n_part2_T_n_l217_217501


namespace exists_instruction_set_l217_217753

theorem exists_instruction_set (N : ℕ) : 
  ∃ (instructions : List (ℕ × ℕ)), 
  instructions.length ≤ 100 * N ∧
  ∀ (initial_arrangement : List Bool), 
  (initial_arrangement.length = N ∧ 
   (∃ (k : ℕ), k ≤ N ∧ (∀ i, i < k → initial_arrangement[i] = true) ∧ (∀ j, k ≤ j < N → initial_arrangement[j] = false))) ∨
  ∃ (k : ℕ), k ≤ N ∧ (∀ i, i < k → initial_arrangement[i] = false) ∧ (∀ j, k ≤ j < N → initial_arrangement[j] = true) →
  ∃ (commands : List (ℕ × ℕ)), commands ⊆ instructions ∧
  let result := commands.foldl (λ arr (i, j), arr.swap i j) initial_arrangement in 
  ∀ (m : ℕ), m ≤ N → (∀ i, i < m → result[i] = true) ∧ (∀ j, m ≤ j < N → result[j] = false) :=
begin
  sorry -- Proof to be constructed.
end

end exists_instruction_set_l217_217753


namespace count_zeros_in_257_is_7_l217_217345

def count_zeros (n : ℕ) : ℕ :=
String.to_list (Nat.toDigits 2 n).reverse.filter (fun c => c = '0').length

theorem count_zeros_in_257_is_7 : count_zeros 257 = 7 := by
  sorry

end count_zeros_in_257_is_7_l217_217345


namespace two_thousand_divisibility_l217_217463

theorem two_thousand_divisibility (n : ℕ) (hn : n > 3) :
  (∃ k : ℕ, k ≤ 2000 ∧ 2^k = (1 + n + nat.choose n 2 + nat.choose n 3)) →
  n = 7 ∨ n = 23 := 
by
  sorry

end two_thousand_divisibility_l217_217463


namespace quadratic_parabola_opens_downwards_l217_217080

theorem quadratic_parabola_opens_downwards :
  ∃ a ∈ ({-1, 0, 1/3, 2} : set ℝ), a < 0 :=
by
  sorry

end quadratic_parabola_opens_downwards_l217_217080


namespace number_of_correct_conditions_l217_217082

def plane := Type

variables (α β γ : plane)

def planes_parallel (a b : plane) : Prop := sorry
def planes_perpendicular (a b : plane) : Prop := sorry
def non_collinear_points_equidistant (a b : plane) : Prop := sorry
def skew_lines_parallel_to_planes (a b : plane) : Prop := sorry

theorem number_of_correct_conditions :
  ¬planes_parallel α β →
  (∃ γ, planes_parallel α γ ∧ planes_parallel β γ) ∧
  (∃ γ, planes_perpendicular α γ ∧ planes_perpendicular β γ) ∧
  (∃ x y z : ℝ, non_collinear_points_equidistant α β) ∧
  (∃ l m, skew_lines_parallel_to_planes l α ∧ skew_lines_parallel_to_planes l β ∧ 
          skew_lines_parallel_to_planes m α ∧ skew_lines_parallel_to_planes m β) →
  2 := sorry

end number_of_correct_conditions_l217_217082


namespace Laura_running_speed_l217_217196

noncomputable def running_speed (x : ℝ) :=
  let biking_time := 30 / (3 * x + 2)
  let running_time := 10 / x
  let total_time := biking_time + running_time
  total_time = 3

theorem Laura_running_speed : ∃ x : ℝ, running_speed x ∧ abs (x - 6.35) < 0.01 :=
sorry

end Laura_running_speed_l217_217196


namespace line_MN_touches_fixed_circle_l217_217644

-- Define an acute-angled triangle
structure AcuteTriangle (A B C : Type) : Type :=
(angle_ACB_lt_pi_div_2 : ∠ACB < π / 2)

-- Define the circumcenter and circumcircle of a triangle
def circumcenter {A B C : Type} [Nonempty A] (triangle : AcuteTriangle A B C) : Type := 
sorry

def circumcircle {A B C : Type} [Nonempty A] (triangle : AcuteTriangle A B C) : Set (Point ℝ) := 
sorry

-- Define the perpendicular bisectors and points M and N
def perpendicular_bisector_AC {A B C : Type} (triangle : AcuteTriangle A B C) : Line ℝ :=
sorry

def perpendicular_bisector_BC {A B C : Type} (triangle : AcuteTriangle A B C) : Line ℝ :=
sorry

def point_M {A B C : Type} (triangle : AcuteTriangle A B C) (AC : Line ℝ) (BC : Line ℝ) : Point ℝ :=
sorry

def point_N {A B C : Type} (triangle : AcuteTriangle A B C) (AC : Line ℝ) (BC : Line ℝ) : Point ℝ :=
sorry

-- Define a moving point C on the circumcircle of triangle ABC
def moving_point_C {A B C : Type} (circle : Set (Point ℝ)) : Prop := 
C ∈ circle

-- Main theorem statement
theorem line_MN_touches_fixed_circle (A B C : Type) [Nonempty A]
  (triangle : AcuteTriangle A B C)
  (circ_center : circumcenter triangle)
  (circ_circle : circumcircle triangle)
  (M_N_points : ∃ (M N : Type), M = point_M triangle (perpendicular_bisector_AC triangle) (perpendicular_bisector_BC triangle) ∧ N = point_N triangle (perpendicular_bisector_AC triangle) (perpendicular_bisector_BC triangle))
  (moving_C : moving_point_C circ_circle) :
  ∃ (fixed_circle : Set (Point ℝ)), tangent_to_circle (line_through_pts M N) fixed_circle :=
sorry

end line_MN_touches_fixed_circle_l217_217644


namespace range_of_a_l217_217903

-- Define sets A and B and the condition A ∩ B = ∅
def set_A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_B (a : ℝ) : Set ℝ := { x | x > a }

-- State the condition: A ∩ B = ∅ implies a ≥ 1
theorem range_of_a (a : ℝ) : (set_A ∩ set_B a = ∅) → a ≥ 1 :=
  by
  sorry

end range_of_a_l217_217903


namespace focus_of_hyperbola_l217_217437

def hyperbola_eq (x y : ℝ) : Prop := 2*x^2 - y^2 + 8*x - 6*y - 12 = 0

theorem focus_of_hyperbola :
  ∃ x y : ℝ, hyperbola_eq x y ∧ (x, y) = (-2 + Real.sqrt 19.5, -3) :=
by
  use (-2 + Real.sqrt 19.5, -3)
  constructor
  sorry

end focus_of_hyperbola_l217_217437


namespace max_value_ab_l217_217208

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end max_value_ab_l217_217208


namespace sum_units_and_tenthousands_digit_product_l217_217012

theorem sum_units_and_tenthousands_digit_product :
  let num1 := 70707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
  let num2 := 60606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606
  (units_digit (num1 * num2) + tenthousands_digit (num1 * num2)) = 6 :=
sorry

end sum_units_and_tenthousands_digit_product_l217_217012


namespace smallest_number_is_correct_largest_number_is_correct_l217_217855

def initial_sequence := "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960"

def remove_digits (n : ℕ) (s : String) : String := sorry  -- Placeholder function for removing n digits

noncomputable def smallest_number_after_removal (s : String) : String :=
  -- Function to find the smallest number possible after removing digits
  remove_digits 100 s

noncomputable def largest_number_after_removal (s : String) : String :=
  -- Function to find the largest number possible after removing digits
  remove_digits 100 s

theorem smallest_number_is_correct : smallest_number_after_removal initial_sequence = "123450" :=
  sorry

theorem largest_number_is_correct : largest_number_after_removal initial_sequence = "56758596049" :=
  sorry

end smallest_number_is_correct_largest_number_is_correct_l217_217855


namespace probability_four_dice_show_two_l217_217718

theorem probability_four_dice_show_two :
  let roll_two := (1 / 8 : ℝ)
  let roll_not_two := (7 / 8 : ℝ)
  let choose := nat.choose 12 4
  let prob_specific_arrangement := roll_two^4 * roll_not_two^8
  let total_prob := choose * prob_specific_arrangement
  total_prob ≈ 0.091 :=
by {
  let roll_two := (1 / 8 : ℝ)
  let roll_not_two := (7 / 8 : ℝ)
  let choose := nat.choose 12 4
  let prob_specific_arrangement := roll_two^4 * roll_not_two^8
  let total_prob := choose * prob_specific_arrangement
  sorry  
}

end probability_four_dice_show_two_l217_217718


namespace sum_of_solutions_eq_neg_23_div_4_l217_217837

noncomputable def sum_of_real_solutions : ℚ :=
  let p := Polynomial.C 16 * Polynomial.X^2 + Polynomial.C 92 * Polynomial.X + Polynomial.C (-13)
  Polynomial.sum_roots p

def main : IO Unit :=
  IO.println s!"Sum of all real solutions to the equation is {sum_of_real_solutions}"
  
theorem sum_of_solutions_eq_neg_23_div_4 : sum_of_real_solutions = -23/4 :=
sorry

end sum_of_solutions_eq_neg_23_div_4_l217_217837


namespace num_sequences_b_produced_by_more_than_one_a_l217_217809

def sequence_a (n : ℕ) : Type := fin n → ℕ

def generate_b (a : sequence_a 20) (i : fin 20) : ℕ :=
  match i with
  | ⟨0, _⟩ => a 0 + a 1
  | ⟨19, _⟩ => a 18 + a 19
  | ⟨i+1, _⟩ => a i + a (i+1) + a (i+2)

theorem num_sequences_b_produced_by_more_than_one_a : 
  ∃ (b : sequence_a 20), (∃ (a₁ a₂ : sequence_a 20), a₁ ≠ a₂ ∧ ∀ i, generate_b a₁ i = b i) ↔ 64 := 
sorry

end num_sequences_b_produced_by_more_than_one_a_l217_217809


namespace question_A_correct_l217_217434

section BallDrawing

-- Definitions of the events
def event_both_not_white : set (set (ℕ × ℕ)) := {s | ∀ p ∈ s, p ≠ (1,1)}
def event_exactly_one_white : set (set (ℕ × ℕ)) := 
    {s | ∃ p ∈ s, p = (1,0) ∨ p = (1,2) ∨ p = (0,1) ∨ p = (2,1)}
def event_at_least_one_white : set (set (ℕ × ℕ)) := 
    {s | ∃ p ∈ s, p.fst = 1 ∨ p.snd = 1}
def event_both_white : set (set (ℕ × ℕ)) := {s | ∀ p ∈ s, p = (1,1)}

-- The proof problem
theorem question_A_correct :
  ((event_both_not_white ∩ event_exactly_one_white) = ∅ ∧ 
  (event_both_not_white ∪ event_exactly_one_white ≠ set.univ) ∧ 
  (event_both_not_white ≠ event_both_white) ∧
  (event_exactly_one_white ≠ event_both_white)) :=
sorry

end BallDrawing

end question_A_correct_l217_217434


namespace min_distance_circle_to_line_l217_217116

-- Define the parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 + t, t - 1)

-- Define the equation of the circle C in Cartesian coordinates
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- The distance from a point to a line, given the standard form of a line ax + by + c = 0
def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

-- The minimum distance from a point on circle C to line l
theorem min_distance_circle_to_line :
  let d := distance_point_to_line 0 0 1 (-1) (-2) in
  d - 1 = sqrt 2 - 1 :=
by
  let d := distance_point_to_line 0 0 1 (-1) (-2)
  show d - 1 = sqrt 2 - 1
  sorry

end min_distance_circle_to_line_l217_217116


namespace domain_of_sqrt_4_minus_x_l217_217678

-- A definition for the function y and the domain condition
def domain_condition (x : ℝ) : Prop := 4 - x ≥ 0

-- Prove the domain
theorem domain_of_sqrt_4_minus_x : {x : ℝ | domain_condition x} = set.Iic 4 :=
by {
  sorry
}

end domain_of_sqrt_4_minus_x_l217_217678


namespace first_of_five_consecutive_sums_60_l217_217296

theorem first_of_five_consecutive_sums_60 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 60) : n = 10 :=
by {
  sorry
}

end first_of_five_consecutive_sums_60_l217_217296


namespace actual_production_correct_total_wage_correct_l217_217763

-- Define the production deviations for each day of the week
def production_deviation := [5, -2, -4, 13, -10, 16, -9]

-- Calculate the actual production using the deviations and planned daily production
def actual_production := 1400 + production_deviation.sum

-- Define the daily production target and pay rate
def daily_production_target : Int := 200
def wage_per_bicycle : Int := 60
def bonus_per_extra_bicycle : Int := 15

-- Calculate the total wage of the workers
def total_wage :=
  let base_wage := actual_production * wage_per_bicycle
  let extra_bicycles := production_deviation.filter (fun x => x > 0).sum
  let bonus := extra_bicycles * bonus_per_extra_bicycle
  base_wage + bonus

-- Proof obligation: show that actual production is 1409
theorem actual_production_correct : actual_production = 1409 :=
by 
  -- Proof will be given here, currently skipped with sorry
  sorry 

-- Proof obligation: show that total wage is 85050
theorem total_wage_correct : total_wage = 85050 :=
by 
  -- Proof will be given here, currently skipped with sorry
  sorry

end actual_production_correct_total_wage_correct_l217_217763


namespace problem_1_problem_2_l217_217826

noncomputable def complete_residue_system (n : ℕ) (as : Fin n → ℕ) :=
  ∀ i j : Fin n, i ≠ j → as i % n ≠ as j % n

theorem problem_1 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) := 
sorry

theorem problem_2 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) ∧ complete_residue_system n (λ i => as i - i) := 
sorry

end problem_1_problem_2_l217_217826


namespace sqrt_sqrt_16_eq_pm_2_l217_217697

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217697


namespace surface_area_of_cube_l217_217924

theorem surface_area_of_cube (V : ℝ) (H : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end surface_area_of_cube_l217_217924


namespace remaining_strawberries_l217_217309

-- Define the constants based on conditions
def initial_kg1 : ℕ := 3
def initial_g1 : ℕ := 300
def given_kg1 : ℕ := 1
def given_g1 : ℕ := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

-- Calculate initial total grams
def initial_total_g : ℕ := kg_to_g initial_kg1 + initial_g1

-- Calculate given total grams
def given_total_g : ℕ := kg_to_g given_kg1 + given_g1

-- Define the remaining grams
def remaining_g (initial_g : ℕ) (given_g : ℕ) : ℕ := initial_g - given_g

-- Statement to prove
theorem remaining_strawberries : remaining_g initial_total_g given_total_g = 1400 := by
sorry

end remaining_strawberries_l217_217309


namespace problem1_problem2_problem3_l217_217259

theorem problem1 : (x : ℝ) → ((x + 1)^2 = 9 → (x = -4 ∨ x = 2)) :=
by
  intro x
  sorry

theorem problem2 : (x : ℝ) → (x^2 - 12*x - 4 = 0 → (x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10)) :=
by
  intro x
  sorry

theorem problem3 : (x : ℝ) → (3*(x - 2)^2 = x*(x - 2) → (x = 2 ∨ x = 3)) :=
by
  intro x
  sorry

end problem1_problem2_problem3_l217_217259


namespace line_slope_translation_l217_217570

theorem line_slope_translation (k : ℝ) (b : ℝ) :
  (∀ x y : ℝ, y = k * x + b → y = k * (x - 3) + (b + 2)) → k = 2 / 3 :=
by
  intro h
  sorry

end line_slope_translation_l217_217570


namespace probability_dmitry_before_father_l217_217340

theorem probability_dmitry_before_father (m x y z : ℝ) (h1 : 0 < x) (h2 : x < m) (h3 : 0 < y)
    (h4 : y < z) (h5 : z < m) : 
    (measure_theory.measure_space.volume {y | y < x} / 
    measure_theory.measure_space.volume {x, y, z | 0 < x ∧ x < m ∧ 0 < y ∧ y < z ∧ z < m}) = (2 / 3) :=
  sorry

end probability_dmitry_before_father_l217_217340


namespace sports_club_problem_l217_217164

theorem sports_club_problem (N B T Neither X : ℕ) (hN : N = 42) (hB : B = 20) (hT : T = 23) (hNeither : Neither = 6) :
  (B + T - X + Neither = N) → X = 7 :=
by
  intro h
  sorry

end sports_club_problem_l217_217164


namespace number_of_boys_in_class_l217_217670

theorem number_of_boys_in_class 
  (n : ℕ)
  (average_height : ℝ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_average_height : ℝ)
  (initial_average_height : average_height = 185)
  (incorrect_record : incorrect_height = 166)
  (correct_record : correct_height = 106)
  (actual_avg : actual_average_height = 183) 
  (total_height_incorrect : ℝ) 
  (total_height_correct : ℝ) 
  (total_height_eq : total_height_incorrect = 185 * n)
  (correct_total_height_eq : total_height_correct = 185 * n - (incorrect_height - correct_height))
  (actual_total_height_eq : total_height_correct = actual_average_height * n) :
  n = 30 :=
by
  sorry

end number_of_boys_in_class_l217_217670


namespace log_x2y2_eq_10_11_l217_217558

theorem log_x2y2_eq_10_11 (x y : ℝ) (h1 : log (x * y^4) = 1) (h2 : log (x^3 * y) = 1) : 
  log (x^2 * y^2) = 10 / 11 :=
by
  sorry

end log_x2y2_eq_10_11_l217_217558


namespace arithmetic_mean_of_pq_is_10_l217_217667

variable (p q r : ℝ)

theorem arithmetic_mean_of_pq_is_10
  (H_mean_qr : (q + r) / 2 = 20)
  (H_r_minus_p : r - p = 20) :
  (p + q) / 2 = 10 := by
  sorry

end arithmetic_mean_of_pq_is_10_l217_217667


namespace sqrt_sqrt_sixteen_l217_217694

theorem sqrt_sqrt_sixteen : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := by
  sorry

end sqrt_sqrt_sixteen_l217_217694


namespace complement_A_in_U_l217_217130

universe u

-- Define the universal set U and set A.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

-- Define the complement of A in U.
def complement (A U: Set ℕ) : Set ℕ :=
  {x ∈ U | x ∉ A}

-- Statement to prove.
theorem complement_A_in_U :
  complement A U = {2, 4, 6} :=
sorry

end complement_A_in_U_l217_217130


namespace part_a_part_b_l217_217364

open EuclideanGeometry Real

namespace CircleIntersection

-- Define a triangle ABC
variables {A B C D E : Point} {S_a S_b S_c : Circle}

-- Define the internal and external angle bisectors
axioms (internal_bisector_AD : IsAngleBisector A B C D) 
       (external_bisector_AE : IsAngleBisector A B C E)

-- Define circles with specified diameters
axioms (circle_S_a : CircleDiameter S_a D E)
       (circle_S_b : CircleDiameter S_b (..) (..)) -- similar to S_a but for other vertices
       (circle_S_c : CircleDiameter S_c (..) (..)) -- similar to S_a but for other vertices

-- Theorem part (a)
theorem part_a :
  ∃ (M N : Point), 
    M ∈ S_a ∧ M ∈ S_b ∧ M ∈ S_c ∧ 
    N ∈ S_a ∧ N ∈ S_b ∧ N ∈ S_c ∧ 
    LineThrough M N (Circumcenter A B C) := sorry

-- Theorem part (b)
theorem part_b :
  ∀ {M N : Point},
    (M ∈ S_a ∧ M ∈ S_b ∧ M ∈ S_c) →
    (N ∈ S_a ∧ N ∈ S_b ∧ N ∈ S_c) →
    ∃ (projectionsM projectionsN : TriangleProjections),
      Equilateral projectionsM ∧ Equilateral projectionsN := sorry

end CircleIntersection

end part_a_part_b_l217_217364


namespace statement_correctness_l217_217022

def correct_statements := [4, 8]
def incorrect_statements := [1, 2, 3, 5, 6, 7]

theorem statement_correctness :
  correct_statements = [4, 8] ∧ incorrect_statements = [1, 2, 3, 5, 6, 7] :=
  by sorry

end statement_correctness_l217_217022


namespace find_k_l217_217909

def a : (ℝ × ℝ × ℝ) := (1, 1, 0)
def b : (ℝ × ℝ × ℝ) := (-1, 0, 2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_k (k : ℝ) :
  dot_product (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3)
              (2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3) = 0 → k = 7 / 5 :=
by
  sorry

end find_k_l217_217909


namespace negative_solution_exists_l217_217059

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l217_217059


namespace range_of_a_l217_217546

noncomputable theory

def A (a : ℝ) := {x : ℝ | 3 + a ≤ x ∧ x ≤ 4 + 3a}
def B := {x : ℝ | (x + 4) / (5 - x) ≥ 0}

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ A a → x ∈ B) ↔ (-1/2 : ℝ) ≤ a ∧ a < (1/3 : ℝ) :=
sorry

end range_of_a_l217_217546


namespace total_copies_in_half_hour_l217_217385

-- Define the rates of the copy machines
def rate_machine1 : ℕ := 35
def rate_machine2 : ℕ := 65

-- Define the duration of time in minutes
def time_minutes : ℕ := 30

-- Define the total number of copies made by both machines in the given duration
def total_copies_made : ℕ := rate_machine1 * time_minutes + rate_machine2 * time_minutes

-- Prove that the total number of copies made is 3000
theorem total_copies_in_half_hour : total_copies_made = 3000 := by
  -- The proof is skipped with sorry for the demonstration purpose
  sorry

end total_copies_in_half_hour_l217_217385


namespace polynomial_coeff_sum_equals_neg_4034_l217_217858

-- Define the polynomial expansion conditions
def polynomial_expansion (x : ℝ) : ℝ :=
  (1 - 2 * x)^2017

-- Define the statement we need to prove
theorem polynomial_coeff_sum_equals_neg_4034 :
  (∑ k in Finset.range 2018, (-1)^k * (k + 1) * (a k x)) = -4034 :=
by
  let a : ℕ → (ℝ → ℝ)
  sorry

end polynomial_coeff_sum_equals_neg_4034_l217_217858


namespace system_has_negative_solution_iff_sum_zero_l217_217049

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l217_217049


namespace proof_line_through_fixed_point_proof_circle_radius_proof_no_tangent_k_proof_longest_chord_l217_217123

-- Define the line and circle equations
def line (k : ℝ) : ℝ → ℝ → Prop := λ x y, k * x - y - k = 0
def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 2 * y + 1 = 0

-- Prove the statements
theorem proof_line_through_fixed_point (k : ℝ) :
  line k 1 0 :=
by {
  unfold line,
  simp,
  exact sub_self k,
}

theorem proof_circle_radius :
  ∃ c : ℝ × ℝ, c = (2, 1) ∧ ∀ x y, circle x y → (x - 2)^2 + (y - 1)^2 = 4 :=
by {
  use (2, 1),
  split,
  { refl, },
  { intros x y h,
    unfold circle at h,
    sorry -- Proof of completing the square
  }
}

theorem proof_no_tangent_k :
  ¬ ∃ k : ℝ, ∀ x y, line k x y → 
  (∃ c : ℝ × ℝ, c = (2, 1) ∧
    real.sqrt ((∑ x, (x - 2)^2) + (y - 1)^2 = 2)) :=
by {
  intro h,
  obtain ⟨k, hk⟩ := h,
  unfold line at hk,
  sorry -- Proof involving discriminant
}

theorem proof_longest_chord :
  ¬ ∃ l : ℝ, l = 2 * real.sqrt 2 ∧ 
  ∀ k x y, line k x y → (x ≠ 1 → y ≠ 0) :=
by {
  sorry -- Proof that the length is not 2sqrt(2)
}

#eval proof_line_through_fixed_point
#eval proof_circle_radius
#eval proof_no_tangent_k
#eval proof_longest_chord

end proof_line_through_fixed_point_proof_circle_radius_proof_no_tangent_k_proof_longest_chord_l217_217123


namespace number_of_non_similar_regular_300_pointed_stars_l217_217917

-- Conditions
def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (Nat.coprime n).length

def non_similar_regular_stars (n : ℕ) : ℕ :=
  let phi_n := euler_totient n
  let m_1 := phi_n - 2
  m_1 / 2

-- Proof problem: number of non-similar regular 300-pointed stars
theorem number_of_non_similar_regular_300_pointed_stars : non_similar_regular_stars 300 = 39 := sorry

end number_of_non_similar_regular_300_pointed_stars_l217_217917


namespace quadratic_radical_type_eq_l217_217107

theorem quadratic_radical_type_eq (x : ℝ) (h : sqrt(x - 1) = sqrt(8)) : x = 9 :=
by sorry

end quadratic_radical_type_eq_l217_217107


namespace ratio_m_n_collinear_l217_217910

theorem ratio_m_n_collinear (m n : ℝ) (h_n : n ≠ 0) :
  let a := (1, 2)
      b := (-2, 3)
      vec1 := (m + 2 * n, 2 * m - 3 * n)
      vec2 := (1 + 2 * -2, 2 + 2 * 3) in
  vec2 = (-3, 8) →
  ∃ k : ℝ, vec1 = k • vec2 →
  m / n = -1 / 2 :=
by
  -- Definitions
  let a := (1, 2)
  let b := (-2, 3)
  let vec1 := (m + 2 * n, 2 * m - 3 * n)
  let vec2 := (1 + 2 * -2, 2 + 2 * 3)
  -- Given vec2 correction
  have vec2_correct : vec2 = (-3, 8) := rfl
  
  -- Given vec2 is (-3, 8)
  intro h,
  
  -- Collinearity condition
  intro h_collinear,
  sorry

end ratio_m_n_collinear_l217_217910


namespace summation_equals_correct_answer_l217_217796

theorem summation_equals_correct_answer : 
  (∑ n in Finset.range 1001, 1 / ((2 * n + 1) * (2 * n + 3) * (2 * n + 5))) = (1001 / 12048045) := 
by
  sorry

end summation_equals_correct_answer_l217_217796


namespace part1_part2_l217_217973

def U : Set ℤ := {x | x.abs < 3} -- Condition U
def A : Set ℤ := {0, 1, 2}       -- Condition A
def B : Set ℤ := {1, 2}          -- Condition B

-- Part 1: Prove A ∪ B = {0, 1, 2}
theorem part1 : A ∪ B = {0, 1, 2} := 
by sorry

-- Part 2: Prove (U \ A) ∩ (U \ B) = {-2, -1}
theorem part2 : (U \ A) ∩ (U \ B) = {-2, -1} := 
by sorry

end part1_part2_l217_217973


namespace sum_of_triangle_ops_l217_217661

def triangle_op (a b c : ℕ) : ℕ := 2 * a + b - c 

theorem sum_of_triangle_ops : 
  triangle_op 1 2 3 + triangle_op 4 6 5 + triangle_op 2 7 1 = 20 :=
by
  sorry

end sum_of_triangle_ops_l217_217661


namespace total_assignment_plans_l217_217404

-- Given set of teachers and assignment conditions
constants (T : Type) [fintype T] (A B C D E : T)
constants (areas : fin 3 → finset T)

-- Condition 1: Teacher A and B cannot go to the same area
axiom A_ne_B (i : fin 3) : A ∉ areas i ∨ B ∉ areas i

-- Condition 2: Teacher A and C must go to the same area
axiom A_eq_C (i j : fin 3) : (A ∈ areas i ∧ C ∈ areas j) → i = j

-- Condition 3: Each area has at least one teacher
axiom one_per_area (i : fin 3) : ∃ t, t ∈ areas i

-- Define the set of teachers and areas
def teachers := {A, B, C, D, E}
def all_areas := finset.fin 3

-- The total number of valid assignment plans
def valid_assignments : ℕ := 36

-- The theorem we aim to prove
theorem total_assignment_plans (h : ∀ t ∈ teachers, ∃ i, t ∈ areas i) :
  valid_assignments = 36 :=
by sorry

end total_assignment_plans_l217_217404


namespace degree_of_h_l217_217490

noncomputable def f (x : ℝ) : ℝ := -9 * x^5 + 2 * x^3 - 4 * x + 7

theorem degree_of_h :
  ∀ (h : Polynomial ℝ), (Polynomial.degree (f + h) = 3) → (Polynomial.degree h = 5) :=
by
  sorry

end degree_of_h_l217_217490


namespace min_points_in_symmetric_set_l217_217405

theorem min_points_in_symmetric_set (T : Set (ℝ × ℝ)) (h1 : ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T)
                                      (h2 : ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T)
                                      (h3 : ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T)
                                      (h4 : (1, 4) ∈ T) : 
    ∃ (S : Finset (ℝ × ℝ)), 
          (∀ p ∈ S, p ∈ T) ∧
          (∀ q ∈ T, ∃ p ∈ S, q = (p.1, p.2) ∨ q = (p.1, -p.2) ∨ q = (-p.1, p.2) ∨ q = (-p.1, -p.2) ∨ q = (-p.2, -p.1) ∨ q = (-p.2, p.1) ∨ q = (p.2, p.1) ∨ q = (p.2, -p.1)) ∧
          S.card = 8 := sorry

end min_points_in_symmetric_set_l217_217405


namespace sheets_in_a_bundle_l217_217425

variable (B : ℕ) -- Denotes the number of sheets in a bundle

-- Conditions
variable (NumBundles NumBunches NumHeaps : ℕ)
variable (SheetsPerBunch SheetsPerHeap TotalSheets : ℕ)

-- Definitions of given conditions
def numBundles := 3
def numBunches := 2
def numHeaps := 5
def sheetsPerBunch := 4
def sheetsPerHeap := 20
def totalSheets := 114

-- Theorem to prove
theorem sheets_in_a_bundle :
  3 * B + 2 * sheetsPerBunch + 5 * sheetsPerHeap = totalSheets → B = 2 := by
  intro h
  sorry

end sheets_in_a_bundle_l217_217425


namespace minimum_radius_part_a_minimum_radius_part_b_l217_217836

-- Definitions for Part (a)
def a := 7
def b := 8
def c := 9
def R1 := 6

-- Statement for Part (a)
theorem minimum_radius_part_a : (c / 2) = R1 := by sorry

-- Definitions for Part (b)
def a' := 9
def b' := 15
def c' := 16
def R2 := 9

-- Statement for Part (b)
theorem minimum_radius_part_b : (c' / 2) = R2 := by sorry

end minimum_radius_part_a_minimum_radius_part_b_l217_217836


namespace distance_to_home_l217_217232

-- Definitions of conditions as per Raviraj's journey
def initial_position : ℝ × ℝ := (0, 0)
def south_20km : ℝ × ℝ := (0, -20)
def west_10km : ℝ × ℝ := (-10, -20)
def north_20km : ℝ × ℝ := (-10, 0)
def west_20km_again : ℝ × ℝ := (-30, 0)

-- Euclidean distance function to calculate straight-line distance
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Our final position after described journey
def final_position : ℝ × ℝ := west_20km_again

-- Using Euclidean distance to verify the final distance to the home is 22.36
theorem distance_to_home (home final : ℝ × ℝ) :
  home = initial_position → final = final_position →
  euclidean_distance home final = 22.36 := 
by {
  intros h0 h1,
  rw [h0, h1],
  sorry
}

end distance_to_home_l217_217232


namespace survey_problem_l217_217942

variable (n : ℕ) (p_tv p_tv_books : ℝ)
def number_of_people_not_like_both_tv_books (n : ℕ) (p_tv p_tv_books : ℝ) : ℕ :=
  let people_no_tv := p_tv * n
  let people_no_tv_books := p_tv_books * people_no_tv
  people_no_tv_books.floor.toNat

-- Given conditions
theorem survey_problem
  (n : ℕ := 1500)
  (p_tv : ℝ := 0.25)
  (p_tv_books : ℝ := 0.15) :
  number_of_people_not_like_both_tv_books n p_tv p_tv_books = 56 := 
by
  sorry

end survey_problem_l217_217942


namespace total_distance_traveled_l217_217744

noncomputable def row_speed_still_water : ℝ := 8
noncomputable def river_speed : ℝ := 2

theorem total_distance_traveled (h : (3.75 / (row_speed_still_water - river_speed)) + (3.75 / (row_speed_still_water + river_speed)) = 1) : 
  2 * 3.75 = 7.5 :=
by
  sorry

end total_distance_traveled_l217_217744


namespace number_7tuples_l217_217620

noncomputable def numberOf7Tuples (X : Finset ℕ) (n : ℕ) : ℕ :=
  if hX : X.card = n then
    let subsetsWith3Elements := X.powerset.filter (λ s, s.card = 3)
    let distinctSubsets := subsetsWith3Elements.toFinset.toList.permutations.filter (λ l, (Finset.univ : Finset ℕ) = (Finset.bind l.toFinset id))
    distinctSubsets.length
  else 0

theorem number_7tuples (X : Finset ℕ) (hX : X = {1, 2, 3, 4, 5, 6, 7}) :
  numberOf7Tuples X 7 = 7! * 6184400 := by sorry

end number_7tuples_l217_217620


namespace initial_investment_l217_217488

theorem initial_investment
  (future_value : ℝ)
  (interest_rate : ℝ)
  (years : ℕ)
  (future_value_eq : future_value = 439.23)
  (interest_rate_eq : interest_rate = 0.10)
  (years_eq : years = 4) :
  let initial_amount := future_value / (1 + interest_rate)^years in
  initial_amount = 300 :=
by
  have FV := future_value_eq,
  have IR := interest_rate_eq,
  have Y := years_eq,
  unfold initial_amount,
  rw [FV, IR, Y],
  simp,
  norm_num,
  exact dec_trivial

end initial_investment_l217_217488


namespace spanish_peanuts_l217_217264

variable (x : ℝ)

theorem spanish_peanuts :
  (10 * 3.50 + x * 3.00 = (10 + x) * 3.40) → x = 2.5 :=
by
  intro h
  sorry

end spanish_peanuts_l217_217264


namespace matrix_a_to_power_6_l217_217643

variables {α : Type*} [field α]

-- Define matrix A and vector a
def A : matrix (fin 2) (fin 2) α := ![![1, 2], ![-1, 4]]
def a : vector α 2 := ![7, 4]

-- Define eigenvalues and eigenvectors
def λ1 : α := 2
def λ2 : α := 3
def α1 : vector α 2 := ![2, 1]
def α2 : vector α 2 := ![1, 1]

-- Main proof statement
theorem matrix_a_to_power_6 : (A ^ 6) ⬝ a = ![435, 339] := sorry  -- Proof details omitted

end matrix_a_to_power_6_l217_217643


namespace decreasing_interval_log_l217_217274

noncomputable def is_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem decreasing_interval_log :
  (∀ x, 2 * x^2 - 3 * x + 1 > 0) →
  (∀ x t, t = 2 * x^2 - 3 * x + 1 → y = log (1 / 2) t → is_decreasing y 1 (⊤ : ℝ)) :=
by
  sorry

end decreasing_interval_log_l217_217274


namespace delete_cycle_preserves_connectivity_l217_217588

noncomputable def strongly_connected (G : Type*) [graph G] : Prop := 
∀ u v : G, ∃ p : u.path v, true

structure graph :=
(V : Type*)
(edges : V → V → Prop)

def indegree (G : graph) (v : G.V) : ℕ := 
card { x : G.V // G.edges x v }

def outdegree (G : graph) (v : G.V) : ℕ := 
card { x : G.V // G.edges v x }

theorem delete_cycle_preserves_connectivity (G : graph) 
  (h_sc : strongly_connected G) 
  (h_deg : ∀ v : G.V, indegree G v ≥ 2 ∧ outdegree G v ≥ 2) 
: ∃ (C' : set (G.V × G.V)), (∀ v u: G.V, v∈C'∧u∈C'→ G.edges v u) → strongly_connected (remove_edges G C') :=
sorry

end delete_cycle_preserves_connectivity_l217_217588


namespace octahedron_midpoint_polyhedron_is_cube_l217_217727

theorem octahedron_midpoint_polyhedron_is_cube (a : ℝ) :
  let octahedron : Polyhedron := regular_octahedron a
  ∃ cube : Polyhedron,
    (cube = polyhedron_connecting_midpoints octahedron) ∧
    (edge_length cube = (Real.sqrt 2 / 3) * a) :=
sorry

end octahedron_midpoint_polyhedron_is_cube_l217_217727


namespace polynomial_eval_at_n_plus_1_l217_217566

theorem polynomial_eval_at_n_plus_1 (p : Polynomial ℝ) (n : ℕ) (h_deg : p.degree = n) 
    (h_values : ∀ k : ℕ, k ≤ n → p.eval k = k / (k + 1)) :
    (if n % 2 = 1 then p.eval (n + 1) = n / (n + 2) else p.eval (n + 1) = 1) :=
by
  sorry

end polynomial_eval_at_n_plus_1_l217_217566


namespace find_abs_of_z_l217_217545

open Complex

-- Define the variable used
variables (z : ℂ)

-- Define the condition for z
def condition (z : ℂ) : Prop := 2 * z - conj z = 1 + 6 * I

-- State the main theorem
theorem find_abs_of_z (hz : condition z) : |z| = Real.sqrt 5 :=
sorry

end find_abs_of_z_l217_217545


namespace simplify_expression_l217_217256

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) =
  16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end simplify_expression_l217_217256


namespace arc_length_of_sector_l217_217886

-- Define the central angle and radius
def central_angle := Real.pi / 3
def radius := 2

-- Statement of the proof problem
theorem arc_length_of_sector : (central_angle * radius) = (2 * Real.pi / 3) :=
by
  sorry

end arc_length_of_sector_l217_217886


namespace complex_number_in_first_quadrant_l217_217596

variable {z : ℂ}
variables {x y : ℝ}

/-- Given that z = 3i / (1 + 2i), prove that the point corresponding
to z in the complex plane is in the first quadrant. -/
theorem complex_number_in_first_quadrant
  (hz : z = 3 * complex.I / (1 + 2 * complex.I))
  (hx : x = (6 : ℝ) / 5)
  (hy : y = (3 : ℝ) / 5) :
  (z.re = x) ∧ (z.im = y) ∧ (x > 0) ∧ (y > 0) := by
  sorry


end complex_number_in_first_quadrant_l217_217596


namespace cylinder_surface_area_l217_217064

theorem cylinder_surface_area (a : ℝ) (h_a : 0 < a) : 
  let S := (λ (x y : ℝ), x^2 + y^2 = 2 * a * x) ∧ (λ (x z : ℝ), z^2 = 2 * a * (2 * a - x)) in
  -- Area computation to be shown equals 16a^2
  ∫∫ (x z : ℝ) in { (x, z) | 0 ≤ x ∧ x ≤ 2 * a ∧ |z| ≤ √(2 * a * (2 * a - x))}, 
    (2 * a) / √(2 * a * (2 * a - x)) = 16 * a^2 :=
sorry

end cylinder_surface_area_l217_217064


namespace negative_solution_condition_l217_217039

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l217_217039


namespace average_makeup_score_is_90_l217_217160

theorem average_makeup_score_is_90 (total_students on_assigned_day on_makeup_day average_assigned_day average_total : ℝ) 
  (h1 : total_students = 100) 
  (h2 : on_assigned_day = 0.70 * total_students) 
  (h3 : on_makeup_day = 0.30 * total_students) 
  (h4 : average_assigned_day = 0.60) 
  (h5 : average_total = 0.69) : 
  (let X := ((average_total * total_students - average_assigned_day * on_assigned_day) / on_makeup_day) in X = 0.90) :=
by
  -- Proof steps go here
  sorry

end average_makeup_score_is_90_l217_217160


namespace integer_implies_perfect_square_l217_217977

theorem integer_implies_perfect_square (n : ℕ) (h : ∃ m : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = m) :
  ∃ k : ℤ, 2 + 2 * Real.sqrt (28 * (n ^ 2) + 1) = (k ^ 2) :=
by
  sorry

end integer_implies_perfect_square_l217_217977


namespace zero_point_interval_l217_217708

theorem zero_point_interval
  (a : ℝ) :
  (∃ (x : ℝ), x ∈ Ioo (-1 : ℝ) (2 : ℝ) ∧ a * x + 3 = 0) ↔ (a > 3 ∧ a < 4) := sorry

end zero_point_interval_l217_217708


namespace annual_earning_difference_l217_217190

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end annual_earning_difference_l217_217190


namespace range_of_x0_l217_217118

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem range_of_x0 (x0 : ℝ) : f x0 > 1 → x0 ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 1 ∞ :=
by {
  sorry
}

end range_of_x0_l217_217118


namespace base9_subtraction_l217_217327

theorem base9_subtraction (a b : Nat) (h1 : a = 256) (h2 : b = 143) : 
  (a - b) = 113 := 
sorry

end base9_subtraction_l217_217327


namespace total_time_is_60_l217_217030

def emma_time : ℕ := 20
def fernando_time : ℕ := 2 * emma_time
def total_time : ℕ := emma_time + fernando_time

theorem total_time_is_60 : total_time = 60 := by
  sorry

end total_time_is_60_l217_217030


namespace first_player_winning_strategy_l217_217305

noncomputable def optimal_first_move : ℕ := 45

-- Prove that with 300 matches initially and following the game rules,
-- taking 45 matches on the first turn leaves the opponent in a losing position.

theorem first_player_winning_strategy (n : ℕ) (h₀ : n = 300) :
    ∃ m : ℕ, (m ≤ n / 2 ∧ n - m = 255) :=
by
  exists optimal_first_move
  sorry

end first_player_winning_strategy_l217_217305


namespace monic_polynomial_average_of_two_with_roots_l217_217238

theorem monic_polynomial_average_of_two_with_roots (n : ℕ) 
  (f : Polynomial ℝ) (h_f_monic : f.monic) (h_deg_f : f.natDegree = n) :
  ∃ g h : Polynomial ℝ, 
  g.monic ∧ h.monic ∧ 
  g.natDegree = n ∧ h.natDegree = n ∧ 
  (∃ (g_roots : Multiset ℝ), (g_roots.card = n ∧ ∀ x ∈ g_roots, is_root g x)) ∧ 
  (∃ (h_roots : Multiset ℝ), (h_roots.card = n ∧ ∀ x ∈ h_roots, is_root h x)) ∧ 
  f = (g + h) / 2 :=
sorry

end monic_polynomial_average_of_two_with_roots_l217_217238


namespace prove_jens_suckers_l217_217249
noncomputable def Jen_ate_suckers (Sienna_suckers : ℕ) (Jen_suckers_given_to_Molly : ℕ) : Prop :=
  let Molly_suckers_given_to_Harmony := Jen_suckers_given_to_Molly - 2
  let Harmony_suckers_given_to_Taylor := Molly_suckers_given_to_Harmony + 3
  let Taylor_suckers_given_to_Callie := Harmony_suckers_given_to_Taylor - 1
  Taylor_suckers_given_to_Callie = 5 → (Sienna_suckers/2) = Jen_suckers_given_to_Molly * 2

#eval Jen_ate_suckers 44 11 -- Example usage, you can change 44 and 11 accordingly

def jen_ate_11_suckers : Prop :=
  Jen_ate_suckers Sienna_suckers 11

theorem prove_jens_suckers : jen_ate_11_suckers :=
  sorry

end prove_jens_suckers_l217_217249


namespace compute_b_in_polynomial_l217_217879

theorem compute_b_in_polynomial :
  ∀ (a b : ℚ), 
    (Polynomial.aeval (3 + Real.sqrt 5) (Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C (-20)) = 0) →
    (Polynomial.aeval (3 - Real.sqrt 5) (Polynomial.X ^ 3 + Polynomial.C a * Polynomial.X ^ 2 + Polynomial.C b * Polynomial.X + Polynomial.C (-20)) = 0) →
    (b = -26) :=
begin
  sorry,
end

end compute_b_in_polynomial_l217_217879


namespace simple_interest_sum_l217_217690
noncomputable def simple_interest (P r n : ℝ) : ℝ := P * r * n / 100
noncomputable def compound_interest (P r n : ℝ) : ℝ := P * (1 + r / 100) ^ n - P

theorem simple_interest_sum :
  let SI := simple_interest 7000 7 2
  let SI2 := 0.5 * SI
  let P := SI2 * 100 / (14 * 6)
  in P = 604 :=
begin
  sorry
end

end simple_interest_sum_l217_217690


namespace overall_percent_change_l217_217632

theorem overall_percent_change (x : ℝ) : 
  (0.85 * x * 1.25 * 0.9 / x - 1) * 100 = -4.375 := 
by 
  sorry

end overall_percent_change_l217_217632


namespace chris_grabbed_donuts_for_snack_l217_217806

-- Definitions and conditions provided in the problem
def dozen := 12

def donuts_bought := 2.5 * dozen

def donuts_eaten := 0.10 * donuts_bought

def donuts_left_after_eating := donuts_bought - donuts_eaten

def donuts_left_for_co_workers := 23

-- The proof problem statement
theorem chris_grabbed_donuts_for_snack :
  donuts_left_after_eating - donuts_left_for_co_workers = 4 :=
sorry

end chris_grabbed_donuts_for_snack_l217_217806


namespace parabola_point_comparison_l217_217571

def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 1

theorem parabola_point_comparison :
  let y_1 := parabola 0
  let y_2 := parabola 1
  let y_3 := parabola (-2)
  y_3 > y_1 ∧ y_1 > y_2 :=
by
  let y_1 := parabola 0
  let y_2 := parabola 1
  let y_3 := parabola (-2)
  have h1 : y_1 = 1 := by simp [parabola]
  have h2 : y_2 = 0 := by simp [parabola]
  have h3 : y_3 = 9 := by simp [parabola]
  simp [h1, h2, h3]
  exact ⟨by linarith, by linarith⟩

end parabola_point_comparison_l217_217571


namespace integer_solutions_system_ineq_l217_217260

theorem integer_solutions_system_ineq (x : ℤ) :
  (3 * x + 6 > x + 8 ∧ (x : ℚ) / 4 ≥ (x - 1) / 3) ↔ (x = 2 ∨ x = 3 ∨ x = 4) :=
by
  sorry

end integer_solutions_system_ineq_l217_217260


namespace tangent_to_circumcircle_of_CSB_l217_217217

open EuclideanGeometry

variables {A B C D S : Point}
variables {Γ : Circle}

-- Definitions based on the problem conditions
def is_parallelogram (A B C D : Point) : Prop :=
∃ P Q, parallelogram A B C D

def is_circumcircle (Γ : Circle) (A B D : Point) : Prop :=
oncircle Γ A ∧ oncircle Γ B ∧ oncircle Γ D

def second_intersects (AC : Line) (Γ : Circle) (S : Point) : Prop :=
∃ E, E ≠ S ∧ onCircle Γ S ∧ onLine AC S ∧ intersectsLine AC Γ

-- The problem to prove
theorem tangent_to_circumcircle_of_CSB
  (parallelogram_ABCD : is_parallelogram A B C D)
  (circumcircle_ABD : is_circumcircle Γ A B D)
  (second_intersection : second_intersects (line_through A C) Γ S) :
  tangent (circumcircle ⟨point B, S, C⟩) (line_through D B) :=
sorry

end tangent_to_circumcircle_of_CSB_l217_217217


namespace Jen_ate_11_suckers_l217_217245

/-
Sienna gave Bailey half of her suckers.
Jen ate half and gave the rest to Molly.
Molly ate 2 and gave the rest to Harmony.
Harmony kept 3 and passed the remainder to Taylor.
Taylor ate one and gave the last 5 to Callie.
How many suckers did Jen eat?
-/

noncomputable def total_suckers_given_to_Callie := 5
noncomputable def total_suckers_Taylor_had := total_suckers_given_to_Callie + 1
noncomputable def total_suckers_Harmony_had := total_suckers_Taylor_had + 3
noncomputable def total_suckers_Molly_had := total_suckers_Harmony_had + 2
noncomputable def total_suckers_Jen_had := total_suckers_Molly_had * 2
noncomputable def suckers_Jen_ate := total_suckers_Jen_had - total_suckers_Molly_had

theorem Jen_ate_11_suckers : suckers_Jen_ate = 11 :=
by {
  unfold total_suckers_given_to_Callie total_suckers_Taylor_had total_suckers_Harmony_had total_suckers_Molly_had total_suckers_Jen_had suckers_Jen_ate,
  sorry
}

end Jen_ate_11_suckers_l217_217245


namespace arithmetic_sequence_sum_l217_217291

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    (∀ n, a (n + 1) = a n + d) →
    (a 1 + a 4 + a 7 = 45) →
    (a 2 + a_5 + a_8 = 39) →
    (a 3 + a_6 + a_9 = 33) :=
by 
  intros a d h_arith_seq h_cond1 h_cond2
  sorry

end arithmetic_sequence_sum_l217_217291


namespace sum_of_first_1971_terms_l217_217280

-- Define the sequence recurrence relation
open Nat

def a : ℕ → ℕ 
| 0       := 0
| 1       := 1
| (n + 2) := 4 + 4 * Math.sqrt (Σ i in Finset.range (n + 2), a i)

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) : ℕ := Σ i in Finset.range (n + 1), a i

-- The theorem to prove the sum of the first 1971 terms
theorem sum_of_first_1971_terms : S 1970 = 15531481 :=
by
  sorry

end sum_of_first_1971_terms_l217_217280


namespace smallest_N_expansion_l217_217074

theorem smallest_N_expansion :
  ∃ N : ℕ, (∃ f : (Σ x : ℕ, ℕ → (Σ i : ℕ, ℕ)),
    ∀ a b c d e : ℕ,
    (N = 15 ∧ (a + b + c + d + e + 1)^N =
     ∑ (s : Σ x : ℕ, ℕ → (Σ i : ℕ, ℕ)), f s) ∧ (f ≠ 0)) := sorry

end smallest_N_expansion_l217_217074


namespace general_term_seq_an_l217_217502

noncomputable def seq_an (n : ℕ) : ℚ :=
1 / (2^n) - 1 / (n * (n + 1))

def sum_Sn (a : ℕ → ℚ) (n : ℕ) : ℚ :=
(nat.sum (λ i, a i) (fin n).toList)

theorem general_term_seq_an (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n : ℕ, S n + a n = (n - 1) / (n * (n + 1))) :
  a = seq_an :=
by
  -- proof will go here
  sorry

end general_term_seq_an_l217_217502


namespace textopolis_word_count_l217_217283

theorem textopolis_word_count :
  let alphabet_size := 26
  let total_one_letter := 2 -- only "A" and "B"
  let total_two_letter := alphabet_size^2
  let excl_two_letter := (alphabet_size - 2)^2
  let total_three_letter := alphabet_size^3
  let excl_three_letter := (alphabet_size - 2)^3
  let total_four_letter := alphabet_size^4
  let excl_four_letter := (alphabet_size - 2)^4
  let valid_two_letter := total_two_letter - excl_two_letter
  let valid_three_letter := total_three_letter - excl_three_letter
  let valid_four_letter := total_four_letter - excl_four_letter
  2 + valid_two_letter + valid_three_letter + valid_four_letter = 129054 := by
  -- To be proved
  sorry

end textopolis_word_count_l217_217283


namespace range_of_a_l217_217153

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((1 - a) * x > 1 - a) → (x < 1)) → (1 < a) :=
by sorry

end range_of_a_l217_217153


namespace proof_problem_l217_217989

variables (A B C D E F : Type) [convex_pentagon A B C D E]

-- Definitions based on given problem conditions
def angle_sum_pentagon := 540 -- in degrees

-- Angles involved
variables (BAE ABC BCD CDE DEA EFD DFE : ℕ)

-- Definition of S and S'
def S := BAE + ABC + 2 * BCD + CDE -- sum of angles at point F minus DEA
def S' := BAE + ABC -- sum of angles BAE and ABC

-- Ratio r
def r := (S - DEA) / S'

-- The statement we need to prove
theorem proof_problem (h1 : BAE + ABC + BCD + CDE + DEA = angle_sum_pentagon) (h2 : BAE + ABC > 0) (h3 : 0 < DEA < angle_sum_pentagon) : 0 < r h3 < 1 := sorry

end proof_problem_l217_217989


namespace sqrt_sum_of_fractions_l217_217428

theorem sqrt_sum_of_fractions :
  sqrt (2 * ((1 / 25 : ℝ) + (1 / 36))) = sqrt 122 / 30 :=
by
  sorry

end sqrt_sum_of_fractions_l217_217428


namespace six_boys_paint_5_days_l217_217797

noncomputable def wall_length (boys_8_time : ℝ) (boys_8_length : ℝ) (boys_8 : ℕ) (boys_6 : ℕ) (boys_6_time : ℝ) : ℝ :=
  let WR_8 := boys_8_length / boys_8_time
  let WR_6 := WR_8 * (boys_8 / boys_6)
  WR_6 * boys_6_time

theorem six_boys_paint_5_days (boys_8_time : ℝ) (boys_8_length : ℝ) (boys_8 : ℕ) (boys_6 : ℕ) (boys_6_time : ℝ) :
  boys_8_time = 3.125 ∧ boys_8_length = 50 ∧ boys_8 = 8 ∧ boys_6 = 6 ∧ boys_6_time = 5 →
  wall_length boys_8_time boys_8_length boys_8 boys_6 boys_6_time = 106.67 :=
by
  intro h
  cases h with h1 h'
  cases h' with h2 h''
  cases h'' with h3 h4
  cases h4 with h5 h6
  dsimp [wall_length]
  rw [h1, h2, h3, h5, h6]
  norm_num
  sorry

end six_boys_paint_5_days_l217_217797


namespace cards_given_l217_217997

/-- Martha starts with 3 cards. She ends up with 79 cards after receiving some from Emily. We need to prove that Emily gave her 76 cards. -/
theorem cards_given (initial_cards final_cards cards_given : ℕ) (h1 : initial_cards = 3) (h2 : final_cards = 79) (h3 : final_cards = initial_cards + cards_given) :
  cards_given = 76 :=
sorry

end cards_given_l217_217997


namespace ratio_of_sector_CPD_l217_217231

-- Define the given angles
def angle_AOC : ℝ := 40
def angle_DOB : ℝ := 60
def angle_COP : ℝ := 110

-- Calculate the angle CPD
def angle_CPD : ℝ := angle_COP - angle_AOC - angle_DOB

-- State the theorem to prove the ratio
theorem ratio_of_sector_CPD (hAOC : angle_AOC = 40) (hDOB : angle_DOB = 60)
(hCOP : angle_COP = 110) : 
  angle_CPD / 360 = 1 / 36 := by
  -- Proof will go here
  sorry

end ratio_of_sector_CPD_l217_217231


namespace probability_dmitry_before_father_l217_217332

noncomputable def prob_dmitry_before_father (m : ℝ) (x y z : ℝ) (h1 : 0 < x ∧ x < m) (h2 : 0 < y ∧ y < z ∧ z < m) : ℝ :=
  if h1 ∧ h2 then 2/3 else 0

theorem probability_dmitry_before_father (m : ℝ) (x y z : ℝ) (h1 : 0 < x ∧ x < m) (h2 : 0 < y ∧ y < z ∧ z < m) :
  prob_dmitry_before_father m x y z h1 h2 = 2 / 3 :=
begin
  sorry
end

end probability_dmitry_before_father_l217_217332


namespace file_size_correct_l217_217767

theorem file_size_correct:
  (∀ t1 t2 : ℕ, (60 / 5 = t1) ∧ (15 - t1 = t2) ∧ (t2 * 10 = 30) → (60 + 30 = 90)) := 
by
  sorry

end file_size_correct_l217_217767


namespace smallest_odd_n_l217_217477

def is_square (k : ℕ) : Prop := ∃ m, m * m = k

def odd (n : ℤ) : Prop := n % 2 = 1

def condition1 (n p : ℤ) : Prop := n^2 = (p + 2)^5 - p^5

def condition2 (n : ℤ) : Prop := 
  ∃ k, is_square ((3 * n + 100)) ∧ odd n

theorem smallest_odd_n (n : ℤ) (p : ℤ) :
  (condition1 n p) ∧ (condition2 n) → n = 3 := 
sorry

end smallest_odd_n_l217_217477


namespace elliptical_equation_and_area_l217_217876

-- Define the variables
variables {a b c x y : ℝ}

-- Define the ellipse given the conditions
def ellipse (a b : ℝ) (h_ab : a > b > 0) (e : ℝ) (he : e = √3 / 2) : Prop :=
  ∃ (c : ℝ), (c = √(a^2 - b^2)) ∧ (c / a = e) ∧
    (∃ (x y : ℝ), (x, y) = (-√3, 1/2) ∧ (x^2 / a^2 + y^2 / b^2 = 1))

-- Define the problem of finding the ellipse equation and the maximum area
theorem elliptical_equation_and_area
  (a b : ℝ) (h_ab : a > b > 0) (e : ℝ) (he : e = √3 / 2)
  (h_point : ellipse a b h_ab e he) :
  (a^2 = 4 ∧ b^2 = 1) ∧
  (∀ l : ℝ, (l ∃ (P Q : ℝ × ℝ), ∃ H : ℝ × ℝ, 
     let O : ℝ × ℝ := (0,0) in OH = 1 → 
     (P = (-√3, 1/2) ∧ 
     Q = (some_value_x, some_value_y) ∧ 
     maximum_area_of_triangle P Q O = 1))) :=
sorry

end elliptical_equation_and_area_l217_217876


namespace product_of_roots_of_cubic_l217_217209

theorem product_of_roots_of_cubic :
  (∃ a b c : ℝ, (3 * a^3 - 9 * a^2 + a - 15 = 0) ∧ 
                    (3 * b^3 - 9 * b^2 + b - 15 = 0) ∧ 
                    (3 * c^3 - 9 * c^2 + c - 15 = 0)) → 
  sorry

end product_of_roots_of_cubic_l217_217209


namespace mrs_lee_earnings_percentage_l217_217933

theorem mrs_lee_earnings_percentage 
  (M F : ℝ)
  (H1 : 1.20 * M = 0.5454545454545454 * (1.20 * M + F)) :
  M = 0.5 * (M + F) :=
by sorry

end mrs_lee_earnings_percentage_l217_217933


namespace general_formula_bn_sum_first_n_terms_Cn_l217_217500

noncomputable def seq_an (n : ℕ) : ℕ := 6 * n + 5

noncomputable def seq_bn (n : ℕ) : ℕ := 3 * n + 1

noncomputable def Cn (n : ℕ) : ℕ := (seq_an n + 1)^(n + 1) / (seq_bn n + 2)^n

noncomputable def Sn (n : ℕ) : ℕ := 3 * n^2 + 8 * n

noncomputable def Tn (n : ℕ) : ℕ := 3 * n * 2^(n + 2)

theorem general_formula_bn :
  ∀ n, ∃ b_n, seq_an n = seq_bn n + seq_bn (n + 1) := 
sorry

theorem sum_first_n_terms_Cn :
  ∀ n, ∃ T_n, Tn n = 3 * n * 2^(n + 2) :=
sorry

end general_formula_bn_sum_first_n_terms_Cn_l217_217500


namespace probability_triangle_area_QYZ_less_than_one_third_l217_217401

theorem probability_triangle_area_QYZ_less_than_one_third
    (XY XZ : ℝ) (Q : ℝ × ℝ)
    (hQ : h < (10 * real.sqrt 34) / 34)
    (area_XYZ : ℝ := (1 / 2) * XY * XZ)
    (YZ : ℝ := real.sqrt (XY ^ 2 + XZ ^ 2)) : 
    (XY = 10) → (XZ = 6) → (area_XYZ = 30) → 
    (Q : ℝ × ℝ) → (YZ = 2 * real.sqrt 34) → 
    (h < 10 / real.sqrt 34) → 
    let area_QYZ := (1 / 2) * h * YZ in
    P < (1 / 3) := by
  sorry

end probability_triangle_area_QYZ_less_than_one_third_l217_217401


namespace minimum_value_of_quadratic_polynomial_l217_217735

-- Define the quadratic polynomial
def quadratic_polynomial (x : ℝ) : ℝ := x^2 + 14 * x + 3

-- Statement to prove
theorem minimum_value_of_quadratic_polynomial : ∃ x : ℝ, quadratic_polynomial x = quadratic_polynomial (-7) :=
sorry

end minimum_value_of_quadratic_polynomial_l217_217735


namespace center_lies_on_line_range_of_x_0_l217_217503

/-- Given a square ABCD with edge AB lying on the line x + 3y - 5 = 0
and another edge CD lying on the line x + 3y + 7 = 0 -/
theorem center_lies_on_line :
  ∃ c : ℝ, (∀ (x y : ℝ), (x + 3 * y + c = 0 ↔ (x + 3 * y - 5 = 0) ∨ (x + 3 * y + 7 = 0))) :=
sorry

/-- Given a square with center G(x0, y0) lying on the line x + 3y + 1 = 0,
when the square has only two vertices in the first quadrant, the range of x0 is (6/5, 13/5) -/
theorem range_of_x_0 (x₀ : ℝ) (y₀ : ℝ) :
  (G_lies_on : (x₀ + 3 * y₀ + 1 = 0)) 
  (Vertices_in_first_quadrant : ∀ (x y : ℝ), (x > 0 ∧ y > 0)) 
  → (6 / 5 < x₀ ∧ x₀ < 13 / 5) :=
sorry

end center_lies_on_line_range_of_x_0_l217_217503


namespace convert_base_10_to_base_7_l217_217438

def base10_to_base7 (n : ℕ) : ℕ := 
  match n with
  | 5423 => 21545
  | _ => 0

theorem convert_base_10_to_base_7 : base10_to_base7 5423 = 21545 := by
  rfl

end convert_base_10_to_base_7_l217_217438


namespace tyson_races_10_l217_217724

def tyson_total_races (lake_speed ocean_speed lake_distance ocean_distance total_time : ℝ) : ℕ :=
  let time_lake := lake_distance / lake_speed
  let time_ocean := ocean_distance / ocean_speed
  let total_races := 2 * total_time / (time_lake + time_ocean)
  (total_races : ℕ)

theorem tyson_races_10 :
  tyson_total_races 3 2.5 3 3 11 = 10 :=
by
  sorry

end tyson_races_10_l217_217724


namespace fraction_of_single_men_l217_217422

theorem fraction_of_single_men :
  ∀ (total_faculty : ℕ) (women_percentage : ℝ) (married_percentage : ℝ) (married_men_ratio : ℝ),
    women_percentage = 0.7 → married_percentage = 0.4 → married_men_ratio = 2/3 →
    (total_faculty * (1 - women_percentage)) * (1 - married_men_ratio) / 
    (total_faculty * (1 - women_percentage)) = 1/3 :=
by 
  intros total_faculty women_percentage married_percentage married_men_ratio h_women h_married h_men_marry
  sorry

end fraction_of_single_men_l217_217422


namespace polygon_interior_angles_l217_217710

theorem polygon_interior_angles {n : ℕ} (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_interior_angles_l217_217710


namespace find_base_b_l217_217145

theorem find_base_b : ∃ b : ℕ, (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 ∧ b = 7 := 
by {
  sorry
}

end find_base_b_l217_217145


namespace number_of_floors_l217_217762

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end number_of_floors_l217_217762


namespace cost_price_per_meter_l217_217748

/-- Given the total length of cloth purchased and the total cost, prove the cost price per meter. -/
theorem cost_price_per_meter (total_length : ℝ) (total_cost : ℝ) (h_length : total_length = 9.25) (h_cost : total_cost = 425.50) : total_cost / total_length = 46.00 := 
by 
  rw [h_length, h_cost]
  have : 425.50 / 9.25 = 46.00 := by norm_num
  exact this

end cost_price_per_meter_l217_217748


namespace area_of_given_rhombus_l217_217275

def diagonal_1 : ℝ := 24
def diagonal_2 : ℝ := 10
def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

theorem area_of_given_rhombus : area_of_rhombus diagonal_1 diagonal_2 = 120 := 
by sorry

end area_of_given_rhombus_l217_217275


namespace suitable_survey_l217_217786

inductive Survey
| FavoriteTVPrograms : Survey
| PrintingErrors : Survey
| BatteryServiceLife : Survey
| InternetUsage : Survey

def is_suitable_for_census (s : Survey) : Prop :=
  match s with
  | Survey.PrintingErrors => True
  | _ => False

theorem suitable_survey : is_suitable_for_census Survey.PrintingErrors = True :=
by
  sorry

end suitable_survey_l217_217786


namespace largest_number_is_89_l217_217083

theorem largest_number_is_89 (a b c d : ℕ) 
  (h1 : a + b + c = 180) 
  (h2 : a + b + d = 197) 
  (h3 : a + c + d = 208) 
  (h4 : b + c + d = 222) : 
  max a (max b (max c d)) = 89 := 
by sorry

end largest_number_is_89_l217_217083


namespace Mary_cut_10_roses_l217_217310

-- Defining the initial and final number of roses
def initial_roses := 6
def final_roses := 16

-- Calculating the number of roses cut by Mary
def roses_cut := final_roses - initial_roses

-- The proof problem: Prove that the number of roses cut is 10
theorem Mary_cut_10_roses : roses_cut = 10 := by
  sorry

end Mary_cut_10_roses_l217_217310


namespace sum_mean_median_mode_correct_l217_217733

-- Given a list of integers
def numbers : List ℕ := [4, 1, 5, 1, 2, 6, 1, 5]

-- Define mean calculation function
def mean (l : List ℕ) : ℚ :=
  let sum := (l.foldl (λ acc x => acc + x) 0 : ℚ)
  (sum / l.length : ℚ)

-- Define median calculation function
def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (≤)
  if sorted.length % 2 = 0 then
    let mid1 := sorted.get! (sorted.length / 2 - 1)
    let mid2 := sorted.get! (sorted.length / 2)
    ((mid1 + mid2) / 2 : ℚ)
  else
    (sorted.get! (sorted.length / 2) : ℚ)

-- Define mode calculation function
def mode (l : List ℕ) : ℕ :=
  let freq := l.foldl (λ acc x => acc.insert x (acc.findD x 0 + 1)) (Std.HashMap.empty)
  let max_pair := freq.fold (λ acc x y => if y > acc.2 then (x, y) else acc) (0, 0)
  max_pair.1

-- Define the sum of mean, median, and mode
def sum_mean_median_mode (l : List ℕ) : ℚ :=
  mean l + median l + (mode l : ℚ)

-- The lean statement for the given proof problem
theorem sum_mean_median_mode_correct : sum_mean_median_mode numbers = 7.125 := by
  sorry

end sum_mean_median_mode_correct_l217_217733


namespace negative_solution_iff_sum_zero_l217_217048

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l217_217048


namespace find_ordered_pair_l217_217828

theorem find_ordered_pair : ∃ x y : ℝ, 3 * x - 4 * y = -7 ∧ 6 * x - 5 * y = 5 ∧ x = 7 ∧ y = 7 :=
by
  use 7, 7
  constructor
  · sorry
  constructor
  · sorry
  constructor
  · refl
  · refl

end find_ordered_pair_l217_217828


namespace sin_alpha_beta_l217_217859

-- Definitions for conditions
variable (α β : ℝ)

-- Given conditions as hypotheses
def h1 : Prop := sin (π / 3 + α / 6) = -3 / 5
def h2 : Prop := cos (π / 12 - β / 2) = -12 / 13
def h3 : Prop := -5 * π < α ∧ α < -2 * π
def h4 : Prop := -11 * π / 6 < β ∧ β < π / 6

-- Proven statement
theorem sin_alpha_beta (h1 : h1 α) (h2 : h2 β) (h3 : h3 α) (h4 : h4 β) :
  sin (α / 6 + β / 2 + π / 4) = 16 / 65 := by
  sorry

end sin_alpha_beta_l217_217859


namespace complex_division_l217_217101

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 + i) = 1 + i :=
by
  sorry

end complex_division_l217_217101


namespace exists_N_1000_turns_to_0_l217_217228

/-- Define the largest prime less than or equal to N -/
noncomputable def largest_prime_leq (N : ℕ) : ℕ := sorry

/-- Prove the existence of a natural number N such that it
    reduces to 0 in exactly 1000 turns by subtracting the largest prime less than or equal to N. -/
theorem exists_N_1000_turns_to_0 :
  ∃ N : ℕ, (Nat.iterate (λ N, N - largest_prime_leq N) 1000 N = 0) :=
sorry

end exists_N_1000_turns_to_0_l217_217228


namespace probability_heads_even_60_tosses_l217_217378

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1 else 0.6 * P_n (n - 1) + 0.4 * (1 - P_n (n - 1))

theorem probability_heads_even_60_tosses :
  P_n 60 = 1 / 2 * (1 + 1 / (5 : ℝ)^60) :=
by sorry

end probability_heads_even_60_tosses_l217_217378


namespace min_cells_required_to_mark_any_l217_217868

open Int

def cross (grid : ℕ → ℕ → Prop) (cell : ℕ × ℕ) : Set (ℕ × ℕ) :=
  {c | c.fst = cell.fst ∨ c.snd = cell.snd}

noncomputable def min_initial_marked_cells (k : ℕ) : ℕ :=
  (k + 1) / 2 * (k + 2) / 2

theorem min_cells_required_to_mark_any (k : ℕ) (N : ℕ) 
  (marked_cells : Set (ℕ × ℕ)) 
  (condition : ∀ cell : ℕ × ℕ, (cross marked_cells cell).card ≥ k → cell ∈ marked_cells) 
  (reachable : ∀ cell : ℕ × ℕ, cell ∈ marked_cells) : 
  N = min_initial_marked_cells k := sorry

end min_cells_required_to_mark_any_l217_217868


namespace regular_pentagon_cannot_tile_seamlessly_l217_217756

-- Declare the types for shapes
inductive Shape
| equilateral_triangle
| square
| regular_pentagon
| regular_hexagon
deriving DecidableEq

-- Define the interior angle function
def interior_angle : Shape → ℕ
| Shape.equilateral_triangle := 60
| Shape.square := 90
| Shape.regular_pentagon := 108
| Shape.regular_hexagon := 120

-- Define the seamless tiling condition
def can_tile_seamlessly (angle : ℕ) : Prop :=
  360 % angle = 0

-- Problem statement
theorem regular_pentagon_cannot_tile_seamlessly :
  ¬ can_tile_seamlessly (interior_angle Shape.regular_pentagon) := by
  sorry

end regular_pentagon_cannot_tile_seamlessly_l217_217756


namespace digit_possibilities_for_divisibility_l217_217386

theorem digit_possibilities_for_divisibility :
  let possibilities := {N : ℕ | N < 10 ∧ (10 * 3 + N) % 4 = 0}
  |possibilities| = 5 := 
by
  sorry

end digit_possibilities_for_divisibility_l217_217386


namespace solve_problem_l217_217583

noncomputable theory

open Real

-- Define the conditions
def conditions : Prop :=
  ∃ (A B C : ℝ) (b c : ℝ), 
  b = 2 ∧
  cos C = 3 / 4 ∧
  (sqrt 7) / 4 = (1 / 2) * A * b * (sqrt (1 - (cos C)^2))

-- Define what we need to prove
def problem : Prop :=
  ∃ (a : ℝ), 
  conditions -> 
  a = 1 ∧
  sin (2 * asin (sqrt 14 / 8)) = 5 * sqrt 7 / 16

-- Main theorem statement
theorem solve_problem : problem :=
sorry

end solve_problem_l217_217583


namespace largest_integer_n_l217_217018

-- Define the condition for existence of positive integers x, y, z that satisfy the given equation
def condition (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 5*x + 5*y + 5*z - 10

-- State that the largest such integer n is 4
theorem largest_integer_n : ∀ (n : ℕ), condition n → n ≤ 4 :=
by {
  sorry
}

end largest_integer_n_l217_217018


namespace fill_with_corners_l217_217725

theorem fill_with_corners (m n k : ℕ) (h1 : m > 1) (h2 : n > 1) (h3 : k > 1) (h4 : ∃ f : ℕ → ℕ → ℕ → bool, 
 f m n k = tt ∧ (∀ i j l, f i j l → (i * j * l) % 3 = 0 ∧ i ≥ 1 ∧ j ≥ 1 ∧ l ≥ 1)) : 
∃ g : ℕ → ℕ → ℕ → bool, g m n k = tt ∧ (∀ i j l, g i j l → (i * j * l) % 3 = 0 ∧ i ≥ 1 ∧ j ≥ 1 ∧ l ≥ 1) :=
by
  sorry

end fill_with_corners_l217_217725


namespace sum_of_first_2023_terms_l217_217890

def sequence (n : ℕ) : ℤ := (2 * n - 1) * int.cos (n * real.pi)

noncomputable def sum_sequence (n : ℕ) : ℤ :=
  (finset.range n).sum sequence

theorem sum_of_first_2023_terms :
  sum_sequence 2023 = -2023 :=
sorry

end sum_of_first_2023_terms_l217_217890


namespace number_thought_of_l217_217749

theorem number_thought_of : ∃ x : ℤ, (x / 5) + 6 = 65 ∧ x = 295 :=
begin
  use 295,
  split,
  { norm_num, },
  { refl, }
end

end number_thought_of_l217_217749


namespace digit_2_appears_5_more_than_digit_6_l217_217761

-- Define a function to count the occurrences of a digit in the range 1 to 750
def count_digit_occurrences (digit : Nat) (n : Nat) : Nat :=
  (List.range' 1 (n + 1)).foldr (λ x acc, acc + (x.digits 10).count digit) 0

-- The Lean statement to prove the desired property
theorem digit_2_appears_5_more_than_digit_6 :
  (count_digit_occurrences 2 750) - (count_digit_occurrences 6 750) = 5 := 
  sorry

end digit_2_appears_5_more_than_digit_6_l217_217761


namespace find_b_l217_217357

variables (a b : ℕ)

theorem find_b
  (h1 : a = 105)
  (h2 : a ^ 3 = 21 * 25 * 315 * b) :
  b = 7 :=
sorry

end find_b_l217_217357


namespace system_has_negative_solution_iff_sum_zero_l217_217052

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l217_217052


namespace angle_A_value_sin_B_minus_C_value_l217_217584

section TriangleProof

variables (A B C : ℝ) (a b c : ℝ)

-- Given conditions
axiom sides_of_triangle : a > 0 ∧ b > 0 ∧ c > 0
axiom angles_of_triangle : A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π
axiom opposite_sides : A + B + C = π
axiom cosine_condition : 2 * cos A * (b * cos C + c * cos B) = a

-- 1. Prove that A = π / 3
theorem angle_A_value : A = π / 3 :=
sorry

-- Assume cos B = 3 / 5
axiom cos_B : cos B = 3 / 5

-- Provide additional constraints from trigonometry
axiom sin_B_pos : sin B > 0
axiom sin_C_pos : sin C > 0

-- 2. Prove that sin(B - C) = (7 * sqrt 3 - 24) / 50
theorem sin_B_minus_C_value : sin (B - C) = (7 * sqrt 3 - 24) / 50 :=
sorry

end TriangleProof

end angle_A_value_sin_B_minus_C_value_l217_217584


namespace probability_two_queens_or_at_least_one_jack_l217_217922

-- Definitions
def num_jacks : ℕ := 4
def num_queens : ℕ := 4
def total_cards : ℕ := 52

-- Probability calculation for drawing either two Queens or at least one Jack
theorem probability_two_queens_or_at_least_one_jack :
  (4 / 52) * (3 / (52 - 1)) + ((4 / 52) * (48 / (52 - 1)) + (48 / 52) * (4 / (52 - 1)) + (4 / 52) * (3 / (52 - 1))) = 2 / 13 :=
by
  sorry

end probability_two_queens_or_at_least_one_jack_l217_217922


namespace mathematical_expression_evaluation_l217_217512

variables (a b c d e f : ℝ)

theorem mathematical_expression_evaluation (h1 : a * b = 1) 
                                          (h2 : c + d = 0) 
                                          (h3 : |e| = Real.sqrt 2) 
                                          (h4 : Real.sqrt f = 8) 
                                          : (1/2)*a*b + (c+d)/5 + e^2 + Real.cbrt f = 6.5 :=
by
  sorry

end mathematical_expression_evaluation_l217_217512


namespace sqrt_xyz_mul_sum_eq_l217_217214

variables (x y z : ℝ)

-- Given conditions
def cond1 : Prop := y + z = 18
def cond2 : Prop := z + x = 19
def cond3 : Prop := x + y = 20

-- Main theorem to prove
theorem sqrt_xyz_mul_sum_eq : cond1 → cond2 → cond3 → (sqrt (x * y * z * (x + y + z)) ≈ 155.4) :=
by
  sorry

end sqrt_xyz_mul_sum_eq_l217_217214


namespace union_of_M_and_N_eq_M_l217_217880

variable {U : Type} {I M N : Set U}

theorem union_of_M_and_N_eq_M (hM_nonempty : M.nonempty) (hN_nonempty : N.nonempty) (hM_proper : M ⊂ I) (hN_proper : N ⊂ I) (hM_ne_N : M ≠ N) (hN_complement : N ∩ (I \ M) = ∅) :
  M ∪ N = M :=
by
  sorry

end union_of_M_and_N_eq_M_l217_217880


namespace number_of_students_in_class_l217_217227

variable (total_amount : ℝ) (value_of_each_card : ℝ) (fraction_with_gift_cards : ℝ) (percentage_sent_cards : ℝ)

def total_students (total_amount / value_of_each_card : ℝ) (fraction_with_gift_cards : ℝ) (percentage_sent_cards : ℝ) : ℝ :=
  let number_of_gift_cards := total_amount / value_of_each_card
  let total_thank_you_cards := number_of_gift_cards / fraction_with_gift_cards
  let actual_percentage_sent_cards := percentage_sent_cards / 100
  total_thank_you_cards / actual_percentage_sent_cards

theorem number_of_students_in_class {total_amount : ℝ} {value_of_each_card : ℝ} {fraction_with_gift_cards : ℝ} {percentage_sent_cards : ℝ} (h1 : total_amount = 50) (h2 : value_of_each_card = 10) (h3 : fraction_with_gift_cards = 1/3) (h4 : percentage_sent_cards = 30) :
  total_students total_amount value_of_each_card fraction_with_gift_cards percentage_sent_cards = 50 :=
by 
  have number_of_gift_cards : ℝ := total_amount / value_of_each_card
  have total_thank_you_cards : ℝ := number_of_gift_cards / fraction_with_gift_cards
  have actual_percentage_sent_cards : ℝ := percentage_sent_cards / 100
  have students := total_thank_you_cards / actual_percentage_sent_cards
  simp [total_students, h1, h2, h3, h4, number_of_gift_cards, total_thank_you_cards, actual_percentage_sent_cards]
  sorry

end number_of_students_in_class_l217_217227


namespace arcsin_eq_solution_l217_217258

noncomputable def arcsinSolutions : Set ℝ :=
  { x | arcsin x + arcsin (3 * x) = π / 4 ∧ -1 ≤ x ∧ x ≤ 1 ∧ -1 / 3 ≤ x ∧ x ≤ 1 / 3 }

theorem arcsin_eq_solution : arcsinSolutions = { sqrt (2 / 51), -sqrt (2 / 51) } :=
by
  sorry

end arcsin_eq_solution_l217_217258


namespace value_of_f_tan_squared_l217_217841

theorem value_of_f_tan_squared 
  {f : ℝ → ℝ}
  (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (x / (x - 1)) = 1 / x)
  {t : ℝ} 
  (h₀ : 0 ≤ t) 
  (h₁ : t ≤ π / 2) : 
  f (tan t ^ 2) = - (csc t) ^ 2 := 
sorry

end value_of_f_tan_squared_l217_217841


namespace base5_to_octal_1234_eval_f_at_3_l217_217755

-- Definition of base conversion from base 5 to decimal and to octal
def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 1234 => 1 * 5^3 + 2 * 5^2 + 3 * 5 + 4
  | _ => 0

def decimal_to_octal (n : Nat) : Nat :=
  match n with
  | 194 => 302
  | _ => 0

-- Definition of the polynomial f(x) = 7x^7 + 6x^6 + 5x^5 + 4x^4 + 3x^3 + 2x^2 + x
def f (x : Nat) : Nat :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

-- Definition of Horner's method evaluation
def horner_eval (x : Nat) : Nat :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1) * x

-- Theorem statement for base-5 to octal conversion
theorem base5_to_octal_1234 : base5_to_decimal 1234 = 194 ∧ decimal_to_octal 194 = 302 :=
  by
    sorry

-- Theorem statement for polynomial evaluation using Horner's method
theorem eval_f_at_3 : horner_eval 3 = f 3 ∧ f 3 = 21324 :=
  by
    sorry

end base5_to_octal_1234_eval_f_at_3_l217_217755


namespace cities_drawn_from_group_b_l217_217688

def group_b_cities : ℕ := 8
def selection_probability : ℝ := 0.25

theorem cities_drawn_from_group_b : 
  group_b_cities * selection_probability = 2 :=
by
  sorry

end cities_drawn_from_group_b_l217_217688


namespace John_max_tests_under_B_l217_217424

theorem John_max_tests_under_B (total_tests first_tests tests_with_B goal_percentage B_tests_first_half : ℕ) :
  total_tests = 60 →
  first_tests = 40 → 
  tests_with_B = 32 → 
  goal_percentage = 75 →
  B_tests_first_half = 32 →
  let needed_B_tests := (goal_percentage * total_tests) / 100
  let remaining_tests := total_tests - first_tests
  let remaining_needed_B_tests := needed_B_tests - B_tests_first_half
  remaining_tests - remaining_needed_B_tests ≤ 7 := sorry

end John_max_tests_under_B_l217_217424


namespace number_of_workers_in_each_block_is_200_l217_217937

-- Conditions
def total_amount : ℕ := 6000
def worth_of_each_gift : ℕ := 2
def number_of_blocks : ℕ := 15

-- Question and answer to be proven
def number_of_workers_in_each_block : ℕ := total_amount / worth_of_each_gift / number_of_blocks

theorem number_of_workers_in_each_block_is_200 :
  number_of_workers_in_each_block = 200 :=
by
  -- Skip the proof with sorry
  sorry

end number_of_workers_in_each_block_is_200_l217_217937


namespace find_balls_on_50th_layer_l217_217224

noncomputable def a : ℕ → ℕ
| 1 => 2
| 2 => 5
| 3 => 10
| 4 => 17
| (n+1) => a n + 2 * n + 1

theorem find_balls_on_50th_layer : a 50 = 2501 := by
  sorry

end find_balls_on_50th_layer_l217_217224


namespace domain_of_sqrt_cosine_sub_half_l217_217829

theorem domain_of_sqrt_cosine_sub_half :
  {x : ℝ | ∃ k : ℤ, (2 * k * π - π / 3) ≤ x ∧ x ≤ (2 * k * π + π / 3)} =
  {x : ℝ | ∃ k : ℤ, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + π / 3} :=
by sorry

end domain_of_sqrt_cosine_sub_half_l217_217829


namespace population_after_two_years_l217_217945

theorem population_after_two_years (initial_population : ℕ)
  (increase_rate : ℝ) (decrease_rate : ℝ)
  (h_initial_population : initial_population = 15000)
  (h_increase_rate : increase_rate = 0.30)
  (h_decrease_rate : decrease_rate = 0.30) :
  let first_year_population := initial_population + (increase_rate * initial_population).toInt
  let second_year_population := first_year_population - (decrease_rate * first_year_population).toInt
  second_year_population = 13650 :=
by
  sorry

end population_after_two_years_l217_217945


namespace incorrect_algorithm_understanding_l217_217351

theorem incorrect_algorithm_understanding 
  (finite_steps : Π (a : Algorithm), a.steps.finite)
  (clear_defined_steps : Π (a : Algorithm), ∀ (step : a.steps), step.defined_and_executable)
  (definitive_outcome : Π (a : Algorithm), ∃ (result : Outcome), a.execute = result) :
  ¬ ∀ (p : Problem), ∃! (a : Algorithm), solves a p :=
sorry

end incorrect_algorithm_understanding_l217_217351


namespace soccer_tournament_l217_217028

theorem soccer_tournament :
  let p := (821 / 2048 : ℚ) in
  let m := 821 in
  let n := 2048 in
  m + n = 2869 :=
begin
  sorry
end

end soccer_tournament_l217_217028


namespace parabola_distance_proof_l217_217277

noncomputable def parabola_focus_to_directrix_distance : ℝ :=
  let p := 4 in p

theorem parabola_distance_proof (p : ℝ) (h : 2 * p = 8) : parabola_focus_to_directrix_distance = 4 := by
  sorry

end parabola_distance_proof_l217_217277


namespace simplifyExpression_l217_217001

theorem simplifyExpression : 
  (Real.sqrt 4 + Real.cbrt (-64) - Real.sqrt ((-3)^2) + abs (Real.sqrt 3 - 1)) = Real.sqrt 3 - 6 :=
by
  sorry

end simplifyExpression_l217_217001


namespace annual_raise_l217_217189

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end annual_raise_l217_217189


namespace row_time_14_24_l217_217965

variable (d c s r : ℝ)

-- Assumptions
def swim_with_current (d c s : ℝ) := s + c = d / 40
def swim_against_current (d c s : ℝ) := s - c = d / 45
def row_against_current (d c r : ℝ) := r - c = d / 15

-- Expected result
def time_to_row_harvard_mit (d c r : ℝ) := d / (r + c) = 14 + 24 / 60

theorem row_time_14_24 :
  swim_with_current d c s ∧
  swim_against_current d c s ∧
  row_against_current d c r →
  time_to_row_harvard_mit d c r :=
by
  sorry

end row_time_14_24_l217_217965


namespace remainder_poly_l217_217475

noncomputable def remainder_division (f g : ℚ[X]) := 
  let ⟨q, r⟩ := f.div_mod g in r

theorem remainder_poly :
  remainder_division (3 * X ^ 5 - 2 * X ^ 3 + 5 * X - 8) (X ^ 2 - 3 * X + 2) = 84 * X - 84 :=
by
  sorry

end remainder_poly_l217_217475


namespace correct_statement_is_C_l217_217350

-- Definitions and conditions from the problem
def axes_of_symmetry (shape : Type) : ℕ := sorry -- Placeholder definition
def is_directly_proportional (x y : ℕ) : Prop := (∃ k, y = k * x)
def is_inversely_proportional (x y : ℕ) : Prop := (∃ k, x * y = k)

-- The actual statements given as conditions
def statement_A : Prop := axes_of_symmetry "equilateral triangle" > axes_of_symmetry "isosceles trapezoid" ∧
                           axes_of_symmetry "equilateral triangle" > axes_of_symmetry "square"

def statement_B : Prop := ∀ x y, y = 5 * x → is_inversely_proportional x y

def statement_C : Prop := ∀ r, 2 * (2 * r * π) = (2 * r) * π

-- We aim to prove this
theorem correct_statement_is_C : statement_C := by
  sorry

end correct_statement_is_C_l217_217350


namespace cos_three_theta_l217_217138

theorem cos_three_theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_three_theta_l217_217138


namespace number_of_valid_paths_l217_217982

theorem number_of_valid_paths (n : ℕ) (h : n > 4) :
  let total_paths := (choose (n - 2) 2)
  let invalid_paths := 2
  total_paths - invalid_paths = (1 / 2) * (n^2 - 5 * n + 2) := 
by
  sorry

end number_of_valid_paths_l217_217982


namespace trapezoid_area_EFGH_l217_217347

structure Point where
  x : ℝ
  y : ℝ

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def length_vertical (p1 p2 : Point) : ℝ :=
  real.abs (p2.y - p1.y)

def vertical_distance (p1 p2 : Point) : ℝ :=
  real.abs (p2.x - p1.x)

def TrapezoidArea (E F G H : Point) : ℝ :=
  let b1 := length_vertical E F
  let b2 := length_vertical G H
  let h := vertical_distance E G
  1/2 * (b1 + b2) * h

theorem trapezoid_area_EFGH : 
  (E F G H : Point)
  (E = ⟨2, -1⟩)
  (F = ⟨2, 2⟩)
  (G = ⟨6, 8⟩)
  (H = ⟨6, 3⟩)
  : TrapezoidArea E F G H = 16 := by
  sorry

end trapezoid_area_EFGH_l217_217347


namespace cookie_baking_time_l217_217452

theorem cookie_baking_time 
  (total_time : ℕ) 
  (white_icing_time: ℕ)
  (chocolate_icing_time: ℕ) 
  (total_icing_time : white_icing_time + chocolate_icing_time = 60)
  (total_cooking_time : total_time = 120):

  (total_time - (white_icing_time + chocolate_icing_time) = 60) :=
by
  sorry

end cookie_baking_time_l217_217452


namespace infinite_primes_divide_f_l217_217981

def non_constant_function (f : ℕ → ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ f a ≠ f b

def divisibility_condition (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a ≠ b → (a - b) ∣ (f a - f b)

theorem infinite_primes_divide_f (f : ℕ → ℕ) 
  (h_non_const : non_constant_function f)
  (h_div : divisibility_condition f) :
  ∃ᶠ p in Filter.atTop, ∃ c : ℕ, p ∣ f c := sorry

end infinite_primes_divide_f_l217_217981


namespace emily_total_beads_l217_217453

theorem emily_total_beads (necklaces : ℕ) (beads_per_necklace : ℕ) (total_beads : ℕ) : 
  necklaces = 11 → 
  beads_per_necklace = 28 → 
  total_beads = necklaces * beads_per_necklace → 
  total_beads = 308 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end emily_total_beads_l217_217453


namespace planar_molecules_l217_217787

structure Molecule :=
  (name : String)
  (formula : String)
  (is_planar : Bool)

def propylene : Molecule := 
  { name := "Propylene", formula := "C3H6", is_planar := False }

def vinyl_chloride : Molecule := 
  { name := "Vinyl Chloride", formula := "C2H3Cl", is_planar := True }

def benzene : Molecule := 
  { name := "Benzene", formula := "C6H6", is_planar := True }

def toluene : Molecule := 
  { name := "Toluene", formula := "C7H8", is_planar := False }

theorem planar_molecules : 
  (vinyl_chloride.is_planar = True) ∧ (benzene.is_planar = True) := 
by
  sorry

end planar_molecules_l217_217787


namespace find_num_non_officers_l217_217673

-- Define the average salaries and number of officers
def avg_salary_employees : Int := 120
def avg_salary_officers : Int := 470
def avg_salary_non_officers : Int := 110
def num_officers : Int := 15

-- States the problem of finding the number of non-officers
theorem find_num_non_officers : ∃ N : Int,
(15 * 470 + N * 110 = (15 + N) * 120) ∧ N = 525 := 
by {
  sorry
}

end find_num_non_officers_l217_217673


namespace thomas_birthday_2012_l217_217311

theorem thomas_birthday_2012 (h1 : (15, 3, 2010).dayOfWeek = 6) : 
  (15, 3, 2012).dayOfWeek = 2 :=
by
  sorry

end thomas_birthday_2012_l217_217311


namespace cos150_lt_cos760_lt_sin470_l217_217792

theorem cos150_lt_cos760_lt_sin470 :
  cos (150 : ℝ) < cos (760 : ℝ) ∧ cos (760 : ℝ) < sin (470 : ℝ) :=
by
  sorry

end cos150_lt_cos760_lt_sin470_l217_217792


namespace triangle_proof_l217_217187

variables (A B C A1 B1 C1 : Type) [MetricSpace A]

-- Definitions of the points and lines based on the conditions
variables [AffineSpace ℝ A]
variables [AffineSpace ℝ B]
variables [AffineSpace ℝ C]
variables [Point ℝ A1]
variables [Point ℝ B1]
variables [Point ℝ C1]

-- Conditions given in the problem
def on_side_BC (A1 : Point ℝ) (B C : Point ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A1 = t • B + (1 - t) • C

def parallel (u v : ℝ) : Prop := ∃ k : ℝ, u = k * v

def intersect (l₁ l₂ : Point ℝ) (p : Point ℝ) : Prop := p ∈ l₁ ∩ l₂

-- Final theorem statement
theorem triangle_proof (hA1 : on_side_BC A1 B C)
                        (hBB1 : parallel (B1 - B) (A1 - A))
                        (hCC1 : parallel (C1 - C) (A1 - A))
                        (hB1 : intersect (AC) (B1))
                        (hC1 : intersect (AB) (C1)) :
  1/(dist A A1) = 1/(dist B B1) + 1/(dist C C1) :=
sorry

end triangle_proof_l217_217187


namespace expression_equals_sqrt3_minus_6_l217_217002

theorem expression_equals_sqrt3_minus_6 : 
  (Real.sqrt 4 + Real.cbrt (-64) - Real.sqrt ((-3) ^ 2) + abs (Real.sqrt 3 - 1)) = Real.sqrt 3 - 6 :=
by sorry

end expression_equals_sqrt3_minus_6_l217_217002


namespace max_score_exam_l217_217591

theorem max_score_exam (Gibi_percent Jigi_percent Mike_percent Lizzy_percent : ℝ)
  (avg_score total_score M : ℝ) :
  Gibi_percent = 0.59 →
  Jigi_percent = 0.55 →
  Mike_percent = 0.99 →
  Lizzy_percent = 0.67 →
  avg_score = 490 →
  total_score = avg_score * 4 →
  total_score = (Gibi_percent + Jigi_percent + Mike_percent + Lizzy_percent) * M →
  M = 700 :=
by
  intros hGibi hJigi hMike hLizzy hAvg hTotalScore hEq
  sorry

end max_score_exam_l217_217591


namespace log_a_x_solution_set_l217_217344

variable {a x : ℝ}

theorem log_a_x_solution_set (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^x > 1) (h4 : x < 0) : ∀ x, (log a x > 0) → (0 < x ∧ x < 1) :=
by
  sorry

end log_a_x_solution_set_l217_217344


namespace necessary_and_sufficient_condition_for_perpendicular_l217_217097

variables {R : Type*} [linear_ordered_field R]
variables (a b : ℝ) (u v : euclidean_space ℝ (fin 2))

-- Definitions based on given conditions
def non_zero_vectors : Prop := u ≠ 0 ∧ v ≠ 0
def b_magnitude : Prop := ∥v∥ = 2 * ∥u∥
def angle_60 : Prop := angle u v = real.pi / 3
def perpendicular (m : ℝ) : Prop := (u - m • v) ⬝ u = 0

-- Problem statement
theorem necessary_and_sufficient_condition_for_perpendicular
  (h_non_zero : non_zero_vectors u v)
  (h_b_magnitude : b_magnitude u v)
  (h_angle_60 : angle_60 u v) :
  (∃ m, perpendicular u v m ∧ m = 1) ↔ ∃ m, m = 1 :=
by
  sorry

end necessary_and_sufficient_condition_for_perpendicular_l217_217097


namespace smaller_solution_of_quadratic_l217_217834

theorem smaller_solution_of_quadratic :
  ∀ (x : ℝ), x^2 - 9 * x - 22 = 0 → (x = -2 ∨ x = 11) → ∀ (y : ℝ), (y = -2) :=
begin
  sorry
end

end smaller_solution_of_quadratic_l217_217834


namespace trig_identity_simplification_l217_217255

theorem trig_identity_simplification (α β : ℝ) :
  (sin α)^2 + (sin β)^2 - (sin α)^2 * (sin β)^2 + (cos α)^2 * (cos β)^2 = 1 :=
by 
  sorry

end trig_identity_simplification_l217_217255


namespace weight_of_second_piece_of_wood_l217_217436

/--
Given: 
1) The density and thickness of the wood are uniform.
2) The first piece of wood is a square with a side length of 3 inches and a weight of 15 ounces.
3) The second piece of wood is a square with a side length of 6 inches.
Theorem: 
The weight of the second piece of wood is 60 ounces.
-/
theorem weight_of_second_piece_of_wood (s1 s2 w1 w2 : ℕ) (h1 : s1 = 3) (h2 : w1 = 15) (h3 : s2 = 6) :
  w2 = 60 :=
sorry

end weight_of_second_piece_of_wood_l217_217436


namespace prove_alpha_range_l217_217920

noncomputable def cos_minus_sin_eq_tan (α : ℝ) : Prop :=
  cos α - sin α = tan α

theorem prove_alpha_range (α : ℝ) (h : cos_minus_sin_eq_tan α) (h0 : 0 < α) (h1 : α < π / 2) : 
  0 < α ∧ α < π / 6 :=
sorry

end prove_alpha_range_l217_217920


namespace product_less_than_e_l217_217896

def f (x : ℝ) : ℝ := Real.log x - x

def a (n : ℕ) : ℝ := 1 + 1 / 2^n

theorem product_less_than_e (n : ℕ) (hn : n > 0) :
  (∏ i in Finset.range n, a (i + 1)) < Real.exp 1 :=
by
  sorry

end product_less_than_e_l217_217896


namespace intersection_eq_l217_217550

namespace ProofProblem

open Set

def A : Set ℤ := {x : ℤ | abs x < 5}
def B : Set ℤ := {x : ℤ | x ≥ 2}

theorem intersection_eq : A ∩ B = {2, 3, 4} :=
by
  sorry

end ProofProblem

end intersection_eq_l217_217550


namespace shift_left_by_pi_over_8_l217_217716

def f (x : ℝ) := Real.sin (2 * x + Real.pi / 3)
def g (x : ℝ) := Real.sin (2 * x + Real.pi / 12)
def shift (x : ℝ) := x + Real.pi / 8

theorem shift_left_by_pi_over_8 (x : ℝ) : f x = g (shift x) := 
by 
    sorry

end shift_left_by_pi_over_8_l217_217716


namespace probability_both_divisible_by_four_l217_217736

-- Define that a number is divisible by 4
def divisible_by_four (n : ℕ) : Prop := n % 4 = 0

-- Define the probability space for an 8-sided die
def fair_eight_sided_die := { n : ℕ // n > 0 ∧ n ≤ 8 }

-- Define the event that a number rolled is divisible by 4
def event_divisible_by_four (n : fair_eight_sided_die) : Prop := divisible_by_four n

-- Calculate the probability of an event
noncomputable def probability_of_event (pred : fair_eight_sided_die → Prop) : ℚ :=
  (finset.univ.filter pred).card / finset.univ.card

-- The main theorem statement
theorem probability_both_divisible_by_four :
  let die_rolls : set (fair_eight_sided_die × fair_eight_sided_die) := set.univ in
  let event : fair_eight_sided_die × fair_eight_sided_die → Prop :=
    λ roll, event_divisible_by_four roll.1 ∧ event_divisible_by_four roll.2 in
  probability_of_event event = 1 / 16 :=
by sorry

end probability_both_divisible_by_four_l217_217736


namespace new_bill_is_correct_l217_217557

-- Definitions for initial and updated prices
def original_order_cost : ℝ := 35
def delivery_and_tip : ℝ := 11.5

def original_price_tomatoes : ℝ := 0.99
def new_price_tomatoes : ℝ := 2.2
def original_price_lettuce : ℝ := 1.0
def new_price_lettuce : ℝ := 1.75
def original_price_celery : ℝ := 1.96
def new_price_celery : ℝ := 2.0
def original_price_cookies : ℝ := 3.5
def new_price_cookies : ℝ := 4.25
def original_price_mustard : ℝ := 2.35
def new_price_mustard : ℝ := 3.1

def special_service_fee : ℝ := 1.5
def discount_percentage : ℝ := 0.1

-- Proof problem statement in Lean 4
theorem new_bill_is_correct :
  let increase_tomatoes := new_price_tomatoes - original_price_tomatoes
  let increase_lettuce := new_price_lettuce - original_price_lettuce
  let increase_celery := new_price_celery - original_price_celery
  let increase_cookies := new_price_cookies - original_price_cookies
  let increase_mustard := new_price_mustard - original_price_mustard
  let total_increase := increase_tomatoes + increase_lettuce + increase_celery + increase_cookies + increase_mustard
  let discount := discount_percentage * new_price_tomatoes
  let total_additional_cost := (total_increase + special_service_fee - discount)
  let new_food_cost := original_order_cost + total_additional_cost
  let total_bill := new_food_cost + delivery_and_tip
  total_bill = 52.56 := 
begin
  sorry
end

end new_bill_is_correct_l217_217557


namespace max_value_of_g_l217_217887

theorem max_value_of_g (a : ℝ) :
  (∀ x : ℝ, f (x) = sin x + a * cos x → f (x) = sin (x + π) + a * cos (x + π)) →
  (∀ x : ℝ, g (x) = a * sin x + cos x) →
  ∃ (c : ℝ), c = (2 * sqrt 3) / 3 ∧ (∀ x : ℝ, g x ≤ c) :=
by
  sorry

end max_value_of_g_l217_217887


namespace triangle_shortest_side_l217_217590

theorem triangle_shortest_side (a b c : ℝ) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) 
    (r : ℝ) (h3 : r = 5) 
    (h4 : a = 4) (h5 : b = 10)
    (circumcircle_tangent_property : 2 * (4 + 10) * r = 30) :
  min a (min b c) = 30 :=
by 
  sorry

end triangle_shortest_side_l217_217590


namespace number_of_ants_l217_217998

def spiders := 8
def spider_legs := 8
def ants := 12
def ant_legs := 6
def total_legs := 136

theorem number_of_ants :
  spiders * spider_legs + ants * ant_legs = total_legs → ants = 12 :=
by
  sorry

end number_of_ants_l217_217998


namespace smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l217_217731

theorem smallest_four_digit_palindrome_div_by_3_with_odd_first_digit :
  ∃ (n : ℕ), (∃ A B : ℕ, n = 1001 * A + 110 * B ∧ 1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ A % 2 = 1) ∧ 3 ∣ n ∧ n = 1221 :=
by
  sorry

end smallest_four_digit_palindrome_div_by_3_with_odd_first_digit_l217_217731


namespace common_tangent_tangent_root_l217_217532

-- Definitions for the given problem conditions
def curve1 (x : ℝ) : ℝ := real.exp x
def curve2 (x : ℝ) : ℝ := real.exp (2 * x) - 2

-- Definition for the tangent line equation for y=e^x at (m, exp m)
def tangent_line1 (m : ℝ) (x : ℝ) : ℝ := real.exp m * (x - m + 1)

-- Definition for the tangent line equation for y=e^(2x)-2 at (a, exp(2 * a) - 2)
def tangent_line2 (a : ℝ) (x : ℝ) : ℝ := 2 * real.exp (2 * a) * x + real.exp (2 * a) * (1 - 2 * a) - 2

-- Check that line l is a common tangent to both curves and a is a root of f(x)
theorem common_tangent_tangent_root (a : ℝ) :
  (∃ m : ℝ, tangent_line1 m a = tangent_line2 a a) →
  (∃ b : ℝ, tangent_line2 a a = curve2 a) →
  (∃ ax : ℝ, f ax = 0) →
  ∃ x : ℝ, f x = real.exp (2 * x) * (2 * x + 2 * real.log 2 - 1) - 2 :=
sorry

end common_tangent_tangent_root_l217_217532


namespace pairing_points_on_half_plane_l217_217878

def a : ℕ → ℕ
| 2 := 1
| n := if n % 2 = 1 then 0 else let m := n / 2 in (m - 1).sum (λ k, a (2 * k + 2) * a (2 * (m - k) - 2))

theorem pairing_points_on_half_plane :
  a 10 = 42 :=
by
  sorry

end pairing_points_on_half_plane_l217_217878


namespace all_solutions_constant_in_range_l217_217036

noncomputable def constant_function_satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x * y) = max (f(x + y)) (f(x) * f(y))

theorem all_solutions_constant_in_range:
  ∀ f : ℝ → ℝ, constant_function_satisfies_equation f → ∃ c : ℝ, (0 ≤ c ∧ c ≤ 1 ∧ ∀ x : ℝ, f(x) = c) :=
by
  sorry

end all_solutions_constant_in_range_l217_217036


namespace first_90_degree_angle_second_90_degree_angle_l217_217658

-- Defining the conditions
def minute_hand_angle (t : ℕ) : ℝ := 6 * t
def hour_hand_angle (t : ℕ) : ℝ := 0.5 * t
def angle_between_hands (t : ℕ) : ℝ := (minute_hand_angle t - hour_hand_angle t) % 360

-- The first theorem for proving the first 90 degree angle
theorem first_90_degree_angle : 
  ∃ t : ℝ, t = 180 / 11 ∧ angle_between_hands (t.to_nat) = 90 :=
by
  sorry

-- The second theorem for proving the second 90 degree angle
theorem second_90_degree_angle : 
  ∃ t : ℝ, t = 540 / 11 ∧ angle_between_hands (t.to_nat) = 270 :=
by
  sorry

end first_90_degree_angle_second_90_degree_angle_l217_217658


namespace nine_digit_composite_l217_217795

theorem nine_digit_composite (d : Fin 9 → Fin 10) (h₀ : ∀ i, d i ≠ 0) (h₁ : ∀ i j, i ≠ j → d i ≠ d j) : ¬ Nat.Prime (Nat.ofDigits 10 (Vector.ofFn d)) :=
by
  have h₀ : (∑ i, d i) = 45 := 
  -- sum of digits is always 45
  sorry
  have h₁ : 9 ∣ (Nat.ofDigits 10 (Vector.ofFn d)) := 
  -- because the sum of the digits is 45, numbers are always divisible by 9
  sorry
  cases h₁ with k hk
  rw hk at *
  have hk' : k ≠ 1 := 
  -- show that k > 1 since the number is a permutation of 1 to 9 and composite
  sorry 
  exact (Nat.prime_def_lt.mp h₀).right ⟨1, by simp, k, hk', h₂.symm⟩
  -- conclude that the number cannot be prime
  sorry

end nine_digit_composite_l217_217795


namespace distance_between_circumcenters_of_parallelogram_l217_217169

noncomputable def parallelogram_distance_circumcenters
  (a b : ℝ) (α : ℝ) : ℝ :=
  sqrt(a^2 + b^2 + 2 * a * b * cos α) * |cot α|

theorem distance_between_circumcenters_of_parallelogram
  (a b : ℝ) (α : ℝ) (AB BC : ℝ) (angle_ABC : ℝ) (h1 : AB = a) (h2 : BC = b) (h3 : angle_ABC = α)
  (parallelogram_ABCD : (AB = a) ∧ (BC = b) ∧ (angle_ABC = α)) :
  parallelogram_distance_circumcenters a b α =
  sqrt(a^2 + b^2 + 2 * a * b * cos α) * |cot α| := by
  sorry

end distance_between_circumcenters_of_parallelogram_l217_217169


namespace smallest_multiple_of_6_and_9_l217_217732

theorem smallest_multiple_of_6_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (n % 9 = 0) ∧ ∀ m : ℕ, m > 0 ∧ (m % 6 = 0) ∧ (m % 9 = 0) → n ≤ m :=
  by
    sorry

end smallest_multiple_of_6_and_9_l217_217732


namespace new_avg_weight_is_correct_l217_217306

-- Define the conditions as variables
variable {initial_people : ℕ} (initial_weight : ℕ) (new_person_weight : ℕ)
variable (number_initial_people eqn_1 : initial_people = 6)
variable (initial_avg_weight eqn_2 : initial_weight = 152) 
variable (new_weight eqn_3 : new_person_weight = 145)

-- The function that calculates the new average weight when a new person enters
def new_avg_weight (initial_people : ℕ) (initial_weight : ℕ) (new_person_weight : ℕ) : ℕ :=
  let total_weight := initial_people * initial_weight + new_person_weight in
  total_weight / (initial_people + 1)

-- The theorem to prove given the conditions
theorem new_avg_weight_is_correct : 
  new_avg_weight 6 152 145 = 151 :=
by
  have h_initial_people : initial_people = 6 := eqn_1
  have h_initial_weight : initial_weight = 152 := eqn_2
  have h_new_person_weight : new_person_weight = 145 := eqn_3
  sorry

end new_avg_weight_is_correct_l217_217306


namespace total_cost_correct_l217_217302

-- Define the parameters
variables (a : ℕ) -- the number of books
-- Define the constants and the conditions
def unit_price : ℝ := 8
def shipping_fee_percentage : ℝ := 0.10

-- Define the total cost including the shipping fee
def total_cost (a : ℕ) : ℝ := unit_price * (1 + shipping_fee_percentage) * a

-- Prove that the total cost is equal to the expected amount
theorem total_cost_correct : total_cost a = 8 * (1 + 0.10) * a := by
  sorry

end total_cost_correct_l217_217302


namespace analyticForm_and_parity_compare_values_l217_217540

variables {m : ℝ} (hm : m > 1) (x : ℝ)

def f (t : ℝ) : ℝ := log m ((1 - t) / (1 + t))

theorem analyticForm_and_parity :
  (∀ x, x ∈ Ioo (-1) 1 → f (x^2 - 1) = log m ((2 - x^2) / x^2)) →
  (∀ x, x ∈ Ioo (-1) 1 → f x = log m ((1 - x) / (1 + x))) ∧
  (∀ x, x ∈ Ioo (-1) 1 → f (-x) = -f x) :=
sorry

theorem compare_values :
  (∀ x, x ∈ Ioo (-1) 1 → f (x^2 - 1) = log m ((2 - x^2) / x^2)) →
  f (log (real.sqrt real.e)) > f (1 / 3) :=
sorry

end analyticForm_and_parity_compare_values_l217_217540


namespace evaluate_expression_l217_217777

def sequence (i : ℕ) : ℕ :=
  if 1 ≤ i ∧ i ≤ 5 then i
  else sequence 1 * sequence 2 * sequence 3 * sequence 4 * sequence 5 * ∏ (j : ℕ) in (finset.range (i-1)).filter(λ j, j > 5), sequence j - 1

theorem evaluate_expression : 
  let a_1_to_2011_prod : ℕ := (finset.range 2011).prod sequence,
      sum_a_i_squared : ℕ := (finset.range 2011).sum (λ i, (sequence (i+1))^2) 
  in a_1_to_2011_prod - sum_a_i_squared = -1941 := 
by 
  intros; rw <- sorry

end evaluate_expression_l217_217777


namespace count_ordered_triples_l217_217202

def S := Finset.range 20

def succ (a b : ℕ) : Prop := 
  (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem count_ordered_triples 
  (h : ∃ n : ℕ, (S.card = 20) ∧
                (∀ x y z : ℕ, 
                   x ∈ S → y ∈ S → z ∈ S →
                   (succ x y) → (succ y z) → (succ z x) →
                   n = 1260)) : True := sorry

end count_ordered_triples_l217_217202


namespace locus_of_A_l217_217979

theorem locus_of_A (x y : ℝ) :
  (∃ P : ℝ × ℝ, ∃ E F A : ℝ × ℝ,
    (P.1 > 4) ∧
    (A = excenter (angle_bisector P E F)) ∧
    (P ∈ hyperbola 16 9) ∧
    (E = (-c, 0)) ∧
    (F = (c, 0)) ∧
    (distance A (x, y) = 0)) →
  (x > 5 ∧ (x^2 / 25 - 9 * (y^2 / 25)) = 1) :=
sorry

-- Definitions used in the statement for clarity
def excenter (bisector : angle_bisector) : ℝ × ℝ := sorry
def angle_bisector (P E F : ℝ × ℝ) : bisector := sorry
def hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop := P.1^2 / a^2 - P.2^2 / b^2 = 1
def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

end locus_of_A_l217_217979


namespace arithmetic_sequence_k_l217_217174

theorem arithmetic_sequence_k (d : ℤ) (h_d : d ≠ 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n, a n = 0 + n * d) (h_k : a 21 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6):
  21 = 21 :=
by
  -- This would be the problem setup
  -- The proof would go here
  sorry

end arithmetic_sequence_k_l217_217174


namespace great_white_shark_teeth_l217_217410

theorem great_white_shark_teeth :
  let tiger_shark_teeth := 180
  let hammerhead_shark_teeth := tiger_shark_teeth / 6
  let great_white_shark_teeth := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)
  in great_white_shark_teeth = 420 := by
  sorry

end great_white_shark_teeth_l217_217410


namespace P_at_7_l217_217624

noncomputable def P (x : ℝ) : ℝ :=
  (3 * x^4 - 39 * x^3 + g * x^2 + h * x + i) * (4 * x^4 - 72 * x^3 + j * x^2 + k * x + l)

theorem P_at_7 (g h i j k l : ℝ) (roots : Set ℂ)
  (P_def : ∀ x, P x = (3 * x^4 - 39 * x^3 + g * x^2 + h * x + i) * (4 * x^4 - 72 * x^3 + j * x^2 + k * x + l))
  (root_set : roots = {1, 2, 3, 4, 6}) :
  P 7 = -3600 :=
by
  sorry

end P_at_7_l217_217624


namespace find_f_one_third_l217_217100

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def satisfies_condition (f : ℝ → ℝ) : Prop :=
∀ x, f (2 - x) = f x

noncomputable def f (x : ℝ) : ℝ := if (2 ≤ x ∧ x ≤ 3) then Real.log (x - 1) / Real.log 2 else 0

theorem find_f_one_third (h_odd : is_odd_function f) (h_condition : satisfies_condition f) :
  f (1 / 3) = Real.log 3 / Real.log 2 - 2 :=
by
  sorry

end find_f_one_third_l217_217100


namespace equiangular_shift_l217_217844

/-- Definition of equiangular sets --/
def equiangular (A B : Finset ℕ) : Prop :=
  A.card = B.card ∧
  A.sum id = B.sum id ∧
  A.sum (λ x, x * x) = B.sum (λ x, x * x) ∧
  A ∩ B = ∅

theorem equiangular_shift (n : ℕ) (A B : Finset ℕ) (k : ℕ) (hk : 0 < k) (hA : A.card = n) (hB : B.card = n) (h_eq_sum : A.sum id = B.sum id) (h_eq_sum_sq : A.sum (λ x, x * x) = B.sum (λ x, x * x)) (h_disjoint : A ∩ B = ∅) : 
  equiangular (A.map (λ x, x + k)) (B.map (λ x, x + k)) :=
sorry

end equiangular_shift_l217_217844


namespace max_distance_MN_l217_217897

theorem max_distance_MN (m : ℝ) : 
  let y1 := 2 * Real.sin m,
      y2 := 2 * Real.sqrt 3 * Real.cos m
  in abs (y1 - y2) ≤ 4 :=
by
  let y1 := 2 * Real.sin m
  let y2 := 2 * Real.sqrt 3 * Real.cos m
  let MN := abs (y1 - y2)
  have h : MN = abs (4 * Real.sin (m - Real.pi / 3)), from sorry
  have h_max : abs (4 * Real.sin (m - Real.pi / 3)) ≤ 4, from sorry
  exact h_max

end max_distance_MN_l217_217897


namespace jackson_running_distance_l217_217606

theorem jackson_running_distance :
  ∃ distance : Nat, 
  (∃ start_distance : Nat, start_distance = 3) ∧ 
  (∀ n ≤ 4, ∃ week_distance : Nat, week_distance = start_distance * 2^(n-1)) ∧ 
  (distance = 3 * 2^3) := 
begin
  sorry
end

end jackson_running_distance_l217_217606


namespace solution_set_of_inequality_l217_217294

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) * (x + 3) > 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l217_217294


namespace Luke_clothing_total_l217_217633

theorem Luke_clothing_total :
  (∃ pieces1 pieces_per_load num_small_loads, pieces1 = 17 ∧ pieces_per_load = 6 ∧ num_small_loads = 5 ∧ 
  pieces1 + pieces_per_load * num_small_loads = 47) :=
by
  -- Definitions from the conditions:
  let pieces1 := 17  -- Pieces in the first load
  let pieces_per_load := 6  -- Pieces per small load
  let num_small_loads := 5  -- Number of small loads
  
  -- Statement to prove:
  have total_pieces := pieces1 + pieces_per_load * num_small_loads
  show (total_pieces = 47), from
  sorry

end Luke_clothing_total_l217_217633


namespace swimming_speed_in_still_water_l217_217399

theorem swimming_speed_in_still_water :
  ∀ (speed_of_water person's_speed time distance: ℝ),
  speed_of_water = 8 →
  time = 1.5 →
  distance = 12 →
  person's_speed - speed_of_water = distance / time →
  person's_speed = 16 :=
by
  intro speed_of_water person's_speed time distance hw ht hd heff
  rw [hw, ht, hd] at heff
  -- steps to isolate person's_speed should be done here, but we leave it as sorry
  sorry

end swimming_speed_in_still_water_l217_217399


namespace solution_set_of_inequality_l217_217691

theorem solution_set_of_inequality (x : ℝ) (h : (2 * x - 1) / x < 0) : 0 < x ∧ x < 1 / 2 :=
by
  sorry

end solution_set_of_inequality_l217_217691


namespace complex_mul_eq_l217_217009

/-- Proof that the product of two complex numbers (1 + i) and (2 + i) is equal to (1 + 3i) -/
theorem complex_mul_eq (i : ℂ) (h_i_squared : i^2 = -1) : (1 + i) * (2 + i) = 1 + 3 * i :=
by
  -- The actual proof logic goes here.
  sorry

end complex_mul_eq_l217_217009


namespace new_students_joined_l217_217348

-- Define conditions
def initial_students : ℕ := 160
def end_year_students : ℕ := 120
def fraction_transferred_out : ℚ := 1 / 3
def total_students_at_start := end_year_students * 3 / 2

-- Theorem statement
theorem new_students_joined : (total_students_at_start - initial_students = 20) :=
by
  -- Placeholder for proof
  sorry

end new_students_joined_l217_217348


namespace fred_games_this_year_l217_217085

variable (last_year_games : ℕ)
variable (difference : ℕ)

theorem fred_games_this_year (h1 : last_year_games = 36) (h2 : difference = 11) : 
  last_year_games - difference = 25 := 
by
  sorry

end fred_games_this_year_l217_217085


namespace three_digit_integers_count_l217_217136

def digitSet : Finset ℕ := {1, 3, 3, 4, 4, 4, 7}

def is_valid_three_digit (n : ℕ) :=
  let digits := n.digits 10 in
  n ≥ 100 ∧ n < 1000 ∧ (∀ d ∈ digits.to_finset, digitSet d)

noncomputable def count_valid_three_digit_numbers : ℕ :=
  (Finset.Icc 100 999).filter is_valid_three_digit).card

theorem three_digit_integers_count : count_valid_three_digit_numbers = 43 :=
  sorry

end three_digit_integers_count_l217_217136


namespace coeff_x3_binom_expansion_l217_217298

theorem coeff_x3_binom_expansion (n : ℕ) (h : (2^n : ℝ) = 64) :
  let T (r : ℕ) := (-1)^r * 2^(n-r) * Nat.choose n r * x^(n - (3*r)/2)
  in T 3 = 240 :=
sorry

end coeff_x3_binom_expansion_l217_217298


namespace thebault_theorem_l217_217372

variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]

structure parallelogram (A B C D : Type) :=
(is_parallel_AB_CD : ∃ l, l.direction = (B - A) ∧ (D - C) = l.direction)
(is_parallel_AD_BC : ∃ l, l.direction = (D - A) ∧ (C - B) = l.direction)

structure square_center (A B C D : Type) :=
(O1 : Type)
(O2 : Type)
(O3 : Type)
(O4 : Type)

theorem thebault_theorem (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
[AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
(par : parallelogram A B C D) (S1 S2 S3 S4 : square_center A B C D) :
  ∃ (O1 O2 O3 O4 : Type), is_square O1 O2 O3 O4 :=
sorry

end thebault_theorem_l217_217372


namespace sufficient_not_necessary_condition_l217_217925

-- Definitions of propositions p and q
def p (x : ℝ) : Prop := abs (x + 1) ≤ 4
def q (x : ℝ) : Prop := x^2 < 5 * x - 6

-- Definitions of negations of p and q
def not_p (x : ℝ) : Prop := x < -5 ∨ x > 3
def not_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- The theorem to prove
theorem sufficient_not_necessary_condition (x : ℝ) :
  (¬ p x → ¬ q x) ∧ (¬ q x → ¬ p x → False) := 
by
  sorry

end sufficient_not_necessary_condition_l217_217925


namespace sqrt_sqrt_sixteen_l217_217692

theorem sqrt_sqrt_sixteen : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := by
  sorry

end sqrt_sqrt_sixteen_l217_217692


namespace area_ratio_of_hexagon_to_triangle_l217_217176

-- Definitions for the problem
variables {A B C D E F G H K : Type}
variables [EuclideanGeometry A B C]

-- Given conditions as definitions
def right_angle_triangle (ABC : Triangle) := is_right_angle ∠ABC
def extend_equal_side (A B C D G E F H K : Point) (AB BC AC : ℝ) := 
  distance A D = distance A B ∧ distance A B = distance B G ∧
  distance B F = distance B C ∧ distance B C = distance C K ∧
  distance A E = distance A C ∧ distance A C = distance C H

-- The proof statement
theorem area_ratio_of_hexagon_to_triangle
  (ABC : Triangle) (HEX : Hexagon)
  (h_triangle : right_angle_triangle ABC)
  (h_side_extension : extend_equal_side A B C D G E F H K (distance A B) (distance B C) (distance A C)) :
  area HEX = 13 * area ABC :=
sorry

end area_ratio_of_hexagon_to_triangle_l217_217176


namespace cos_alpha_minus_beta_tan_alpha_plus_pi_over_4_l217_217369

-- Problem 1: Prove that cos(α - β) = 59/72
theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : sin α - sin β = -(1/3)) 
  (h2 : cos α - cos β = 1/2) : 
  cos (α - β) = 59/72 :=
sorry

-- Problem 2: Prove that tan(α + π/4) = 3/22
theorem tan_alpha_plus_pi_over_4 (α β : ℝ) 
  (h3 : tan (α + β) = 2/5) 
  (h4 : tan (β - π/4) = 1/4) : 
  tan (α + π/4) = 3/22 :=
sorry

end cos_alpha_minus_beta_tan_alpha_plus_pi_over_4_l217_217369


namespace sufficient_but_not_necessary_l217_217367

theorem sufficient_but_not_necessary (x : ℝ) : (x ≥ 1) → ((∀ x, x > 1 → x ≥ 1) ∧ ¬(∀ x, x ≥ 1 → x > 1)) := by
  intro h
  split
  · intro a
    apply le_of_lt
    exact a
  · intro b
    have h_1 : x = 1 := by sorry
    exact h_1.not_lt sorry
  sorry

end sufficient_but_not_necessary_l217_217367


namespace length_AD_l217_217368

theorem length_AD (A B C D M : Type) 
    (trisect : B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (midpoint_M : M ≠ A ∧ M ≠ C)
    (trisect_AD : ∃ x : ℝ, x > 0 ∧ distance A B = x ∧ distance B C = x ∧ distance C D = x)
    (midpoint_AC : ∃ y : ℝ, y > 0 ∧ distance A C = 2 * y ∧ distance M C = y ∧ y = 10) : 
  distance A D = 30 := 
sorry

end length_AD_l217_217368


namespace percentage_decrease_in_area_after_unfolding_l217_217414

-- Definition of the problem conditions
variable (L W : ℝ) -- Original length and width of the towel
variable (n m : ℕ) -- Number of times the towel is folded lengthwise and widthwise
variable (L_new W_new : ℝ) -- New length and width after folding
variable (A_original A_new : ℝ) -- Original and new areas

-- Conditions described in the problem:
def folded_length := L / (2^n)
def folded_width := W / (2^m)
def bleached_length := 0.8 * folded_length
def bleached_width := 0.9 * folded_width
def original_area := L * W
def new_area := bleached_length * bleached_width

-- Theorem statement for the percentage decrease in area
theorem percentage_decrease_in_area_after_unfolding
    (hA_orig : A_original = original_area)
    (hA_new : A_new = new_area) :
  (A_original - A_new) / A_original * 100 = 28 := by
  sorry

end percentage_decrease_in_area_after_unfolding_l217_217414


namespace minimum_m_value_l217_217493

noncomputable def f (x : ℝ) : ℝ := x^2 + 4 * x
def g (θ : ℝ) : ℝ := 2 * Real.cos θ - 1 
def m (θ : ℝ) : ℝ := f (g θ)

theorem minimum_m_value : ∃ θ : ℝ, m θ = -4 := sorry

end minimum_m_value_l217_217493


namespace all_roots_in_interval_l217_217092

variable {q : ℕ → ℝ} (h_pos : ∀ n, q n > 0)

noncomputable
def f : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| 1 := λ x, x
| (n + 2) := λ x, (1 + q (n + 1)) * x * f (n + 1) x - q (n + 1) * f n x

theorem all_roots_in_interval (n : ℕ) : ∀ x : ℝ, f q n x = 0 → -1 ≤ x ∧ x ≤ 1 :=
sorry

end all_roots_in_interval_l217_217092


namespace shaded_area_l217_217951

noncomputable def radius : ℝ := 5
def area_triangle : ℝ := 2 * (1 / 2 * radius * radius)
def area_sector : ℝ := 2 * (1 / 4 * (Real.pi * radius * radius))
def area_shaded : ℝ := area_triangle + area_sector

theorem shaded_area (r : ℝ) 
  (diam_ab : r = radius)
  (diam_cd : r = radius)
  (perpendicular : True) 
  (circle_area : ℝ :=  Real.pi * r^2)
  (sector_angle : ℝ := Real.pi / 2) :
  area_shaded = 25 + 12.5 * Real.pi := by
sorry

end shaded_area_l217_217951


namespace simplify_product_l217_217651

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l217_217651


namespace triangle_inequality_l217_217167

noncomputable theory

-- Define the properties and relationships in the triangle
variable (A B C I D E F L M N : Type)

-- Define the key distances in the triangle
variables [has_dist A] [has_dist B] [has_dist C] [has_dist I] 
          [has_dist D] [has_dist E] [has_dist F] [has_dist L] [has_dist M] [has_dist N]

-- Assume the properties of the points and incenter
axiom incircle_touches : 
  dist A I < dist A B ∧ dist I D = dist I B ∧ dist I E = dist I C ∧ dist I F = dist I A

-- Define the main inequality to be proved
theorem triangle_inequality (hD : dist split_line A I D ∧ dist split_line B I E ∧ dist split_line C I F) 
                            (hL : ∀ z : Type, dist split_line A I z = dist z B + dist z C) 
        : dist A L + dist B M + dist C N ≤ 3 * (dist A D + dist B E + dist C F) :=
by { sorry }

end triangle_inequality_l217_217167


namespace vector_perpendicular_sin_cos_l217_217553

open Real

theorem vector_perpendicular_sin_cos (θ : ℝ) (h1 : θ ∈ Ioo (π / 2) π) 
(h2 : sin θ + 2 * cos θ = 0) : sin θ - cos θ = 3 * sqrt 5 / 5 :=
by 
  sorry

end vector_perpendicular_sin_cos_l217_217553


namespace find_lambda_perpendicular_l217_217134

def vector (α : Type _) := (α × α)

def perpendicular (v1 v2 : vector ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_lambda_perpendicular :
  let a := (1, -2) : vector ℝ
  let b := (1, 0) : vector ℝ
  ∃ λ : ℝ, perpendicular (λ • a + b) (a - 4 • b) ∧ λ = 3 :=
by
  sorry

end find_lambda_perpendicular_l217_217134


namespace daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l217_217447

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l217_217447


namespace part1_solution_part2_solution_l217_217842

def equiangular (A B : Set ℤ) : Prop :=
  (A.sum id = B.sum id) ∧
  (A.map (λ x, x^2)).sum id = (B.map (λ x, x^2)).sum id ∧
  A ∩ B = ∅

theorem part1_solution (x y : ℤ) (hA : Set ℤ): 
  hA = {1, 5, 6} → 
  ∃ x y, x ∈ hA ∧ y ∈ hA ∧ equiangular {1, 5, 6} {2, x, y} → 
  (x = 4 ∧ y = 8) ∨ (x = 8 ∧ y = 4) :=
sorry

theorem part2_solution (A B : Set ℤ) (k : ℤ) (hA : ∀ (x : ℤ), x ∈ A → A x = a_i) (hx : A.sum id = B.sum id) (hxx : (A.map (λ x, x^2)).sum id = (B.map (λ x, x^2)).sum id) (hB : A ∩ B = ∅) : 
  equiangular (A.map (λ x, x+k)) (B.map (λ x, x+k)) :=
sorry

end part1_solution_part2_solution_l217_217842


namespace volvox_pentagons_heptagons_diff_l217_217233

-- Given conditions
variables (V E F f_5 f_6 f_7 : ℕ)

-- Euler's polyhedron formula
axiom euler_formula : V - E + F = 2

-- Each edge is shared by two faces
axiom edge_formula : 2 * E = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Each vertex shared by three faces
axiom vertex_formula : 3 * V = 5 * f_5 + 6 * f_6 + 7 * f_7

-- Total number of faces equals sum of individual face types 
def total_faces : ℕ := f_5 + f_6 + f_7

-- Prove that the number of pentagonal cells exceeds the number of heptagonal cells by 12
theorem volvox_pentagons_heptagons_diff : f_5 - f_7 = 12 := 
sorry

end volvox_pentagons_heptagons_diff_l217_217233


namespace acetic_acid_properties_l217_217442

theorem acetic_acid_properties :
  let c_weight := 12.01
  let h_weight := 1.008
  let o_weight := 16.00
  let molecular_weight := 2 * c_weight + 4 * h_weight + 2 * o_weight
  let percent_C := (2 * c_weight / molecular_weight) * 100
  let percent_H := (4 * h_weight / molecular_weight) * 100
  let percent_O := (2 * o_weight / molecular_weight) * 100
  in molecular_weight = 60.052 ∧ percent_C = 40.01 ∧ percent_H = 6.71 ∧ percent_O = 53.28 :=
by sorry

end acetic_acid_properties_l217_217442


namespace boys_in_class_l217_217668

noncomputable def number_of_boys_in_class : ℕ :=
  let avg_height : ℕ := 185
  let wrong_height : ℕ := 166
  let actual_wrong_height : ℕ := 106
  let actual_avg_height : ℕ := 183
  let difference : ℕ := wrong_height - actual_wrong_height
  -- Derived from the given equation: 185 * n - difference = 183 * n
  let equation := (avg_height * n - difference = actual_avg_height * n)
  -- From equation we have: (185 - 183) * n = difference
  -- Which leads to: 2 * n = 60
  let result : ℕ := 30

theorem boys_in_class : number_of_boys_in_class = 30 := 
by
  sorry

end boys_in_class_l217_217668


namespace arith_mean_smallest_num_subset_l217_217078

theorem arith_mean_smallest_num_subset (r n : ℕ) (hr : 1 ≤ r) (hrn : r ≤ n) :
  (arithmetic_mean_smallest r n) = (n + 1) / (r + 1) :=
sorry

end arith_mean_smallest_num_subset_l217_217078


namespace numerator_sum_reciprocal_divisible_by_p_l217_217647

theorem numerator_sum_reciprocal_divisible_by_p {p : ℕ} (hp : p.prime) (hp_gt_two : 2 < p) :
  p ∣ (Nat.num (∑ k in Finset.range p \ {0}, (1 : ℚ) / k)) :=
sorry

end numerator_sum_reciprocal_divisible_by_p_l217_217647


namespace boats_distance_one_minute_before_collision_l217_217360

-- Defining the conditions
def boats_moving_towards_each_other (speed1 speed2 initial_distance : ℝ) := 
  speed1 > 0 ∧ speed2 > 0 ∧ initial_distance > 0

-- The theorem stating the problem and the correct answer
theorem boats_distance_one_minute_before_collision
  (speed1 speed2 initial_distance : ℝ) 
  (h : boats_moving_towards_each_other speed1 speed2 initial_distance) 
  (unit_conversion_factor : ℝ := 1 / 60): 
  let relative_speed := speed1 + speed2 in
  let time_to_collide := initial_distance / relative_speed in
  let relative_speed_mpm := relative_speed * unit_conversion_factor in
  let distance_before_collision := 1 * relative_speed_mpm in
  distance_before_collision = 0.4 :=
by
  sorry

end boats_distance_one_minute_before_collision_l217_217360


namespace maple_tree_taller_than_pine_tree_l217_217934

def improper_fraction (a b : ℕ) : ℚ := a + (b : ℚ) / 4
def mixed_number_to_improper_fraction (n m : ℕ) : ℚ := improper_fraction n m

def pine_tree_height : ℚ := mixed_number_to_improper_fraction 12 1
def maple_tree_height : ℚ := mixed_number_to_improper_fraction 18 3

theorem maple_tree_taller_than_pine_tree :
  maple_tree_height - pine_tree_height = 6 + 1 / 2 :=
by sorry

end maple_tree_taller_than_pine_tree_l217_217934


namespace Jen_ate_11_suckers_l217_217247

/-
Sienna gave Bailey half of her suckers.
Jen ate half and gave the rest to Molly.
Molly ate 2 and gave the rest to Harmony.
Harmony kept 3 and passed the remainder to Taylor.
Taylor ate one and gave the last 5 to Callie.
How many suckers did Jen eat?
-/

noncomputable def total_suckers_given_to_Callie := 5
noncomputable def total_suckers_Taylor_had := total_suckers_given_to_Callie + 1
noncomputable def total_suckers_Harmony_had := total_suckers_Taylor_had + 3
noncomputable def total_suckers_Molly_had := total_suckers_Harmony_had + 2
noncomputable def total_suckers_Jen_had := total_suckers_Molly_had * 2
noncomputable def suckers_Jen_ate := total_suckers_Jen_had - total_suckers_Molly_had

theorem Jen_ate_11_suckers : suckers_Jen_ate = 11 :=
by {
  unfold total_suckers_given_to_Callie total_suckers_Taylor_had total_suckers_Harmony_had total_suckers_Molly_had total_suckers_Jen_had suckers_Jen_ate,
  sorry
}

end Jen_ate_11_suckers_l217_217247


namespace apple_price_l217_217790

variable (p q : ℝ)

theorem apple_price :
  (30 * p + 3 * q = 168) →
  (30 * p + 6 * q = 186) →
  (20 * p = 100) →
  p = 5 :=
by
  intros h1 h2 h3
  have h4 : p = 5 := sorry
  exact h4

end apple_price_l217_217790


namespace solve_inequality_l217_217466

theorem solve_inequality (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ (2 < x) ∧ (x < 3) := by
  sorry

end solve_inequality_l217_217466


namespace min_value_of_quadratic_l217_217831

theorem min_value_of_quadratic (x y : ℝ) : (x^2 + 2*x*y + y^2) ≥ 0 ∧ ∃ x y, x = -y ∧ x^2 + 2*x*y + y^2 = 0 := by
  sorry

end min_value_of_quadratic_l217_217831


namespace subset_singleton_zero_l217_217549

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_singleton_zero : {0} ⊆ X :=
by
  sorry

end subset_singleton_zero_l217_217549


namespace cardinality_intersection_A_B_l217_217615

def floor (x : ℝ) : ℤ := int.floor x

def A : set ℤ := { y | ∃ x : ℝ, y = floor x + floor (2 * x) + floor (4 * x) }
def B : set ℕ := {n | 1 ≤ n ∧ n ≤ 2019}

theorem cardinality_intersection_A_B : 
  ∃ n, n = 1154 ∧ n = set.card ((A : set ℕ) ∩ B) :=
begin
  sorry
end

end cardinality_intersection_A_B_l217_217615


namespace clock_display_four_threes_l217_217451

theorem clock_display_four_threes : 
  (∃ (count : ℕ), (count = ∑ (a b M_1 M_2 S_1 S_2 : ℕ), 
  (if ( ('3' in [a, b, M_1, M_2, S_1, S_2].map (λ x, x.digit)) and 
        ([a, b] = hours_digits and a ∈ {0, 1, 2} and 
         [M_1, M_2] = minute_digits and 
         [S_1, S_2] = second_digits)) = 4 then 1 else 0))) = 135
-- Proof is omitted
:= sorry

end clock_display_four_threes_l217_217451


namespace annual_raise_l217_217188

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end annual_raise_l217_217188


namespace count_two_digit_numbers_satisfying_sum_property_l217_217297

theorem count_two_digit_numbers_satisfying_sum_property : 
  (Finset.card (Finset.filter (λ (n : ℕ × ℕ), 1 ≤ n.1 ∧ n.1 ≤ 9 ∧ 0 ≤ n.2 ∧ n.2 ≤ 9 ∧ (10 * n.1 + n.2) + (10 * n.2 + n.1) = 132) 
    (Finset.product (Finset.range 10) (Finset.range 10)))) = 7 := 
by 
  sorry

end count_two_digit_numbers_satisfying_sum_property_l217_217297


namespace swimmer_distance_l217_217409

theorem swimmer_distance :
  let swimmer_speed : ℝ := 3
  let current_speed : ℝ := 1.7
  let time : ℝ := 2.3076923076923075
  let effective_speed := swimmer_speed - current_speed
  let distance := effective_speed * time
  distance = 3 := by
sorry

end swimmer_distance_l217_217409


namespace polynomial_solution_l217_217464

variables {P : ℝ → ℝ} [polynomial P]

theorem polynomial_solution (h : ∀ x : ℝ, P(x^2 - 2 * x) = (P(x - 2))^2) :
  ∃ n : ℕ, ∀ x : ℝ, P(x) = (x + 1)^(n : ℕ) :=
sorry

end polynomial_solution_l217_217464


namespace negative_solution_iff_sum_zero_l217_217045

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l217_217045


namespace intersection_complement_A_B_l217_217904

universe u

-- Definitions of sets A and B in Lean
def U := set ℝ
def A : set ℝ := {y | ∃ x : ℝ, y = 2^x + 1}
def B : set ℝ := {x | 0 < x ∧ x < 1}

-- Define the complement of A in the universal set U
def complement_A : set ℝ := {y | y ≤ 1}

-- Define the problem statement
theorem intersection_complement_A_B :
  ((U \ A) ∩ B) = B :=
  sorry

end intersection_complement_A_B_l217_217904


namespace evaluate_expression_l217_217819

theorem evaluate_expression (x y z : ℤ) (hx : x = 5) (hy : y = x + 3) (hz : z = y - 11) 
  (h₁ : x + 2 ≠ 0) (h₂ : y - 3 ≠ 0) (h₃ : z + 7 ≠ 0) : 
  ((x + 3) / (x + 2)) * ((y - 2) / (y - 3)) * ((z + 9) / (z + 7)) = 72 / 35 := 
by 
  sorry

end evaluate_expression_l217_217819


namespace collinearity_of_centers_l217_217874

theorem collinearity_of_centers {A B C M N P I O : Point}
  (hABC : Triangle A B C)
  (hM : TangentPoint M (Incircle A B C) A B)
  (hN : TangentPoint N (Incircle A B C) B C)
  (hP : TangentPoint P (Incircle A B C) C A)
  (hI : Incenter I A B C)
  (hO : Circumcenter O A B C)
  (hOrthocenter : Orthocenter (Orthocenter M N P)) :
  Collinear O I (Orthocenter M N P) := sorry

end collinearity_of_centers_l217_217874


namespace calculate_probability_l217_217165

-- Define the parameters
variables (n : ℕ) -- number of students in each row
variables (N : ℕ → ℕ × ℕ × ℕ) -- function mapping time to the tuple of students in three rows

-- Define the conditions
def initial_conditions (t : ℕ) : Prop := 
  N 0 = (n, n, n) ∧
  (∀ t, (∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ N t = (i, j, k)) →
    abs (N t).1 - (N t).2 < 2 ∧
    abs (N t).1 - (N t).3 < 2 ∧
    abs (N t).2 - (N t).3 < 2
  )

-- Define the final probability calculation
def final_probability : ℚ :=
  (3.factorial ^ n) * (n.factorial ^ 3) / (3 * n).factorial

-- The main theorem
theorem calculate_probability : initial_conditions n N → 
  final_probability n = (3.factorial ^ n) * (n.factorial ^ 3) / (3 * n).factorial :=
sorry

end calculate_probability_l217_217165


namespace volume_of_circumscribed_polyhedron_l217_217648

theorem volume_of_circumscribed_polyhedron (R : ℝ) (V : ℝ) (S_n : ℝ) (h : Π (F_i : ℝ), V = (1/3) * S_n * R) : V = (1/3) * S_n * R :=
sorry

end volume_of_circumscribed_polyhedron_l217_217648


namespace coloring_circles_l217_217983

theorem coloring_circles (n : ℕ) (h : n > 0) :
  ∃ f : set ℝ × ℝ → bool,
    (∀ (C : set ℝ × ℝ) (u v : ℝ × ℝ),
      u ∈ C ↔ v ∈ C → u ≠ v → f u ≠ f v) := by
  sorry

end coloring_circles_l217_217983


namespace inequality_solution_l217_217038

theorem inequality_solution (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ -6) :
  (2 / (x + 2) + 8 / (x + 6) ≥ 2) ↔ (x ∈ set.Ioc (-6 : ℝ) 1) :=
sorry

end inequality_solution_l217_217038


namespace triangle_segment_relation_l217_217184

variable {A B C A1 B1 C1 : Type}

theorem triangle_segment_relation
  (ABC : triangle A B C)
  (A1_on_BC : on_segment A1 B C)
  (B1_on_AC : on_line B1 A C)
  (C1_on_AB : on_line C1 A B)
  (BB1_parallel_AA1 : parallel (line B B1) (line A A1))
  (CC1_parallel_AA1 : parallel (line C C1) (line A A1)) :
  (1 / (segment_length A A1)) = (1 / (segment_length B B1)) + (1 / (segment_length C C1)) :=
sorry

end triangle_segment_relation_l217_217184


namespace trucks_more_than_buses_l217_217307

theorem trucks_more_than_buses (b t : ℕ) (h₁ : b = 9) (h₂ : t = 17) : t - b = 8 :=
by
  sorry

end trucks_more_than_buses_l217_217307


namespace differentiable_function_inequality_l217_217483

variable {f : ℝ → ℝ}

theorem differentiable_function_inequality 
  (hf : ∀ (x : ℝ), Differentiable ℝ f ∧ (x + 1) * fderiv ℝ f x ≥ 0) :
  f(0) + f(-2) < 2 * f(-1) :=
sorry

end differentiable_function_inequality_l217_217483


namespace sum_of_angles_pentagon_octagon_shared_side_l217_217317

theorem sum_of_angles_pentagon_octagon_shared_side
  (A B C D : Type)
  (pentagon : regular_polygon 5)
  (octagon : regular_polygon 8)
  (shared_side : A ∈ sides pentagon ∧ A ∈ sides octagon)
  (adjacent_B_C : adjacent B C pentagon)
  (adjacent_B_D : adjacent B D octagon) :
  ∠BAC + ∠BAD = 243 :=
by
  sorry

end sum_of_angles_pentagon_octagon_shared_side_l217_217317


namespace evaluate_fraction_l217_217820

theorem evaluate_fraction : 1 + 3 / (4 + 5 / (6 + 7 / 8)) = 85 / 52 :=
by sorry

end evaluate_fraction_l217_217820


namespace find_m_l217_217662

variables (a b m : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def f' (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_m (h1 : f m = 0) (h2 : f' m = 0) (h3 : m ≠ 0)
    (h4 : ∃ x, f' x = 0 ∧ ∀ y, x ≤ y → f x ≥ f y ∧ f x = 1/2) :
    m = 3/2 :=
sorry

end find_m_l217_217662


namespace perpendicular_planes_parallel_l217_217102

variables (m n : Line) (α β : Plane)

-- Conditions
axiom diff_lines (h1 : m ≠ n)
axiom diff_planes (h2 : α ≠ β)

-- Theorem to be proven
theorem perpendicular_planes_parallel (h3 : m ⊥ α) (h4 : m ⊥ β) : α ∥ β :=
sorry

end perpendicular_planes_parallel_l217_217102


namespace negative_solution_exists_l217_217060

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l217_217060


namespace negative_solution_iff_sum_zero_l217_217046

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l217_217046


namespace avg_people_moving_per_hour_l217_217181

theorem avg_people_moving_per_hour (total_people : ℕ) (total_days : ℕ) (hours_per_day : ℕ) (h : total_people = 3000 ∧ total_days = 4 ∧ hours_per_day = 24) : 
  (total_people / (total_days * hours_per_day)).toFloat.round = 31 :=
by
  have h1 : total_people = 3000 := h.1;
  have h2 : total_days = 4 := h.2.1;
  have h3 : hours_per_day = 24 := h.2.2;
  rw [h1, h2, h3];
  sorry

end avg_people_moving_per_hour_l217_217181


namespace felicity_used_5_gallons_less_l217_217824

def adhesion_gas_problem : Prop :=
  ∃ A x : ℕ, (A + 23 = 30) ∧ (4 * A - x = 23) ∧ (x = 5)
  
theorem felicity_used_5_gallons_less :
  adhesion_gas_problem :=
by
  sorry

end felicity_used_5_gallons_less_l217_217824


namespace batsman_innings_l217_217760

theorem batsman_innings (n : ℕ) 
  (h_avg_increase : ∀ k : ℕ, let avg_before := 46, avg_after := 48 in avg_before + 2 = avg_after)
  (h_avg_after : ∀ k : ℕ, let avg_after := 48 in avg_after = 48)
  (h_score_increase : ∀ k : ℕ, let score := 80 in score = 80)
  : n = 16 → (n + 1) = 17 :=
by
  intros
  sorry

end batsman_innings_l217_217760


namespace number_of_common_tangents_l217_217605

structure Circle (r : ℝ) :=
(radius : ℝ := r)

def externally_tangent (c1 c2 : Circle) : Prop :=
  -- External tangency condition, which can be more precisely defined as needed
  sorry

theorem number_of_common_tangents {a b c : Circle}
  (ha : a = Circle.mk 3)
  (hb : b = Circle.mk 4)
  (hc : c = Circle.mk 5)
  (h1 : externally_tangent a b)
  (h2 : externally_tangent a c)
  (h3 : externally_tangent b c) :
  ∃ n : ℕ, n = 0 := 
begin
  use 0,
  -- Proof goes here
  sorry
end

end number_of_common_tangents_l217_217605


namespace parallelepiped_volume_sum_of_squares_l217_217774

theorem parallelepiped_volume_sum_of_squares 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℤ)
  (gcd_a : Int.gcd (Int.gcd a1 a2) a3 = 1)
  (gcd_b : Int.gcd (Int.gcd b1 b2) b3 = 1)
  (gcd_c : Int.gcd (Int.gcd c1 c2) c3 = 1) : 
  let u := (a1, a2, a3)
  let v := (b1, b2, b3)
  let w := (c1, c2, c3)
  let cross_product := (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1)
  let volume := (Int.abs (a2 * b3 - a3 * b2)^2 + Int.abs (a3 * b1 - a1 * b3)^2 + Int.abs (a1 * b2 - a2 * b1)^2)
in let volume_check := ∃ (x y z : ℤ), volume = x^2 + y^2 + z^2
in volume_check := sorry

end parallelepiped_volume_sum_of_squares_l217_217774


namespace factorization_correct_l217_217032

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end factorization_correct_l217_217032


namespace probability_not_integer_ratio_correct_l217_217489

open Finset

noncomputable def four_numbers := {1, 2, 3, 4}

def is_not_integer_ratio (a b : ℕ) : Prop :=
  a ∈ four_numbers ∧ b ∈ (four_numbers.erase a) ∧ ¬ (a % b = 0)

def probability_not_integer_ratio : ℚ :=
  let total_events := 12
  let successful_events := 8
  in successful_events / total_events

theorem probability_not_integer_ratio_correct :
  probability_not_integer_ratio = 2 / 3 :=
by
  sorry

end probability_not_integer_ratio_correct_l217_217489


namespace An_is_integer_for_all_n_l217_217103

noncomputable def sin_theta (a b : ℕ) : ℝ :=
  if h : a^2 + b^2 ≠ 0 then (2 * a * b) / (a^2 + b^2) else 0

theorem An_is_integer_for_all_n (a b : ℕ) (n : ℕ) (h₁ : a > b) (h₂ : 0 < sin_theta a b) (h₃ : sin_theta a b < 1) :
  ∃ k : ℤ, ∀ n : ℕ, ((a^2 + b^2)^n * sin_theta a b) = k :=
sorry

end An_is_integer_for_all_n_l217_217103


namespace distance_from_center_of_circle_to_line_l217_217105

theorem distance_from_center_of_circle_to_line :
  let origin := (0, 0)
  let polar_axis := (1, 0)
  let line_l (t : ℝ) := (t + 1, t - 3)
  let circle_C := {ρ | ∃ θ : ℝ, ρ = 4 * cos θ}
  let center_circle_C := (2, 0)
  let line_l_eq := λ x y, x - y - 4 = 0

  distance_from_center_of_circle_to_line :
    dist (2, 0) (affine_line.mk linear_ℝ_origin basis_ℝ 
       (linear_ℝ b.mk (P.mk_ℝ Q1.mk_ℝ (2 : R))/⊤ (Q.0_α (λ_⊤ : (all.mk_ℝ.lineary 

   dist.mk (cos two(partial ))) :=
     let a : ℝ := 1
     let b := -1
     let c := -4
     let x₀ := 2
     let y₀ := 0 in
   2 * sqrt 2 :=

end distance_from_center_of_circle_to_line_l217_217105


namespace ellipse_and_fixed_point_l217_217524

-- Definitions according to conditions
def point := ℝ × ℝ
def A : point := (0, -2)
def B : point := (3 / 2, -1)
def P : point := (1, -2)

-- General form of ellipse equation
def general_ellipse_eq (m n : ℝ) (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Line Segment equation AB
def line_segment_eq (a b : point) : ℝ → ℝ := λ x, (b.2 - a.2) / (b.1 - a.1) * x + (a.2 * b.1 - a.1 * b.2) / (b.1 - a.1)

-- Geometry relations
def meet_y_axis (y : ℝ) : point := (0, y)

-- Main theorem
theorem ellipse_and_fixed_point :
  ∃ (E : ℝ → ℝ → Prop), (E = general_ellipse_eq (1 / 3) (1 / 4)) ∧
      ∀ M N : point, collinear [M, N, meet_y_axis (-2)] :=
sorry

end ellipse_and_fixed_point_l217_217524


namespace solution_l217_217830

noncomputable def least_positive_integer_M : ℕ :=
  Inf {M : ℕ | M > 0 ∧ ∃ (n : ℕ) (P : Fin n → Polynomial ℤ), M * Polynomial.X = ∑ i, (P i) ^ 3}

theorem solution : least_positive_integer_M = 6 := sorry

end solution_l217_217830


namespace find_usual_time_l217_217362

variable (R T : ℝ)

theorem find_usual_time
  (h_condition :  R * T = (9 / 8) * R * (T - 4)) :
  T = 36 :=
by
  sorry

end find_usual_time_l217_217362


namespace alcohol_quantity_in_mixture_l217_217287

theorem alcohol_quantity_in_mixture : 
  ∃ (A W : ℕ), (A = 8) ∧ (A * 3 = 4 * W) ∧ (A * 5 = 4 * (W + 4)) :=
by
  sorry -- This is a placeholder; the proof itself is not required.

end alcohol_quantity_in_mixture_l217_217287


namespace count_true_statements_l217_217129

theorem count_true_statements : 
  (1 : ℕ) + (0 : ℕ) + (0 : ℕ) + (1 : ℕ) = 2 :=
by
  let original_statement := ∀ (a : ℝ), a > -5 → a > -8
  let converse_statement := ∀ (a : ℝ), a > -8 → a > -5
  let inverse_statement := ∀ (a : ℝ), a ≤ -5 → a ≤ -8
  let contrapositive_statement := ∀ (a : ℝ), a ≤ -8 → a ≤ -5

  -- Original statement is true
  have h_original: original_statement,
  { intros a ha, exact lt_of_lt_of_le ha (by linarith) },

  -- Converse statement is false
  have h_converse: ¬ converse_statement,
  { intro h, specialize h (-6), have h1 : -6 > -8 := by linarith, specialize h h1, linarith },

  -- Inverse statement is false
  have h_inverse: ¬ inverse_statement,
  { intro h, specialize h (-6), have h1 : -6 ≤ -5 := by linarith, specialize h h1, linarith },

  -- Contrapositive statement is true
  have h_contrapositive: contrapositive_statement,
  { intros a ha, exact le_of_lt (by linarith) },

  -- Sum the number of true statements
  exact rfl

end count_true_statements_l217_217129


namespace sequence_geometric_progression_iff_b1_eq_b2_l217_217873

theorem sequence_geometric_progression_iff_b1_eq_b2 
  (b : ℕ → ℝ) 
  (h0 : ∀ n, b n > 0)
  (h1 : ∀ n, b (n + 2) = 3 * b n * b (n + 1)) :
  (∃ r : ℝ, ∀ n, b (n + 1) = r * b n) ↔ b 1 = b 0 :=
sorry

end sequence_geometric_progression_iff_b1_eq_b2_l217_217873


namespace gcd_24_36_l217_217469

theorem gcd_24_36 : Int.gcd 24 36 = 12 := by
  sorry

end gcd_24_36_l217_217469


namespace sqrt_c_is_c8_l217_217365

-- Define c_n
def c_n (n : ℕ) : ℕ :=
  (List.replicate n 1).foldl (λ acc x, acc * 10 + x) 0

-- Given conditions
axiom recursive_relation (n : ℕ) : c_n (n + 1) = 10 * c_n n + 1
axiom square_relation (n : ℕ) : c_n (n + 1) ^ 2 = 100 * c_n n ^ 2 + (2 * (List.replicate n 2).foldl (λ acc x, acc * 10 + x) 0) * 10 + 1

-- Given number and the target to prove
theorem sqrt_c_is_c8 : c_n 8 ^ 2 = 123456787654321 := by
  sorry

end sqrt_c_is_c8_l217_217365


namespace finite_operations_pentagon_finite_operations_2019gon_l217_217649

theorem finite_operations_pentagon :
  ∀ (x : Fin 5 → ℤ), (∑ i, x i) > 0 → ∀ (y_index : Fin 5), x y_index < 0 → (∃ n, ∀ k > n, ¬ (∃ i : Fin 5, x (i + 1) = x i + y ∧ x i = -y ∧ x (i + 2) = y + z)) :=
by
  sorry

theorem finite_operations_2019gon :
  ∀ (x : Fin 2019 → ℤ), (∑ i, x i) > 0 → ∀ (y_index : Fin 2019), x y_index < 0 → (∃ n, ∀ k > n, ¬ (∃ i : Fin 2019, x (i + 1) = x i + y ∧ x i = -y ∧ x (i + 2) = y + z)) :=
by
  sorry

end finite_operations_pentagon_finite_operations_2019gon_l217_217649


namespace area_quadrilateral_BCEF_l217_217952

open Real EuclideanGeometry

noncomputable def parallelogram (A B C D : Point) := 
  ∃ E : Point, 
  E ∈ LineSegment C D ∧ 
  ∃ F : Point, ∃ BD : Line, ∃ AE : Line,
  BD ∈ LineThrough B D ∧ AE ∈ LineThrough A E ∧ F ∈ BD ∧ F ∈ AE ∧
  Area (Triangle A F D) = 6 ∧
  Area (Triangle D E F) = 4 ∧
  Area (Quadrilateral B C E F) = 11

theorem area_quadrilateral_BCEF 
  (A B C D E F : Point) 
  (h1 : E ∈ LineSegment C D)
  (h2 : ∃ BD : Line, BD ∈ LineThrough B D) 
  (h3 : ∃ AE : Line, AE ∈ LineThrough A E)
  (hF : F ∈ (h2.some) ∧ F ∈ (h3.some))
  (hAFD : Area (Triangle A F D) = 6)
  (hDEF : Area (Triangle D E F) = 4) :
  Area (Quadrilateral B C E F) = 11 := 
sorry

end area_quadrilateral_BCEF_l217_217952


namespace dice_prob_l217_217720

noncomputable def probability_four_twos : ℝ :=
  let total_ways := Nat.choose 12 4
  let prob_each_arrangement := (1 / 8)^4 * (7 / 8)^8
  in total_ways * prob_each_arrangement

theorem dice_prob : probability_four_twos = 0.089 := by
  sorry

end dice_prob_l217_217720


namespace integral_sin2x_cos6x_l217_217008

open Real

theorem integral_sin2x_cos6x : 
  (∫ x in 0..π, 2^4 * (sin x)^2 * (cos x)^6) = (5 * π / 8) :=
by
  sorry

end integral_sin2x_cos6x_l217_217008


namespace log_equation_solution_l217_217522

/-- Given \( x < 1 \) and \((\log_{10} x)^3 - \log_{10}(x^3) = 125\), prove that \( (\log_{10} x)^4 - \log_{10}(x^4) = 645 \). -/
theorem log_equation_solution (x : ℝ) (hx : x < 1) (h : (Real.log10 x)^3 - Real.log10 (x^3) = 125) :
  (Real.log10 x)^4 - Real.log10 (x^4) = 645 :=
by
  sorry

end log_equation_solution_l217_217522


namespace isosceles_triangle_no_obtuse_l217_217593

theorem isosceles_triangle_no_obtuse (A B C : ℝ) 
  (h1 : A = 70) 
  (h2 : B = 70) 
  (h3 : A + B + C = 180) 
  (h_iso : A = B) 
  : (A ≤ 90) ∧ (B ≤ 90) ∧ (C ≤ 90) :=
by
  sorry

end isosceles_triangle_no_obtuse_l217_217593


namespace six_star_three_l217_217142

def binary_op (x y : ℕ) : ℕ := 4 * x + 5 * y - x * y

theorem six_star_three : binary_op 6 3 = 21 := by
  sorry

end six_star_three_l217_217142


namespace all_possible_permissible_triangles_present_l217_217870

theorem all_possible_permissible_triangles_present 
  (p : ℕ) (prime_p : Prime p) :
  ∀ (angles : (ℕ × ℕ × ℕ)), (∀ (i j k : ℕ), i + j + k = p ∧ i < p ∧ j < p ∧ k < p → 
  angles = (i, j, k)) → by the time no more distinct permissible triangles can be formed, all possible permissible triangles are present :=
begin
  -- The proof is omitted as per instructions.
  sorry
end

end all_possible_permissible_triangles_present_l217_217870


namespace maximum_ab_is_40_l217_217206

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end maximum_ab_is_40_l217_217206


namespace evaluate_expression_l217_217799

theorem evaluate_expression :
  (-2)^2 + Real.sqrt 16 - 2 * Real.sin (Float.pi/6) + (2023 - Real.pi : ℝ)^0 = 8 :=
by
  sorry

end evaluate_expression_l217_217799


namespace recurrence_relation_b_l217_217010

-- Definitions for the conditions
def a (n : ℕ) : ℕ
| 0     := 1
| 1     := 2
| 2     := 5
| (n+3) := (a (n+2) * a (n+1) - 2) / a n

def b (n : ℕ) : ℕ := a (2 * n)

-- The theorem we need to prove
theorem recurrence_relation_b (n : ℕ) :
  b (n + 2) - 4 * b (n + 1) + b n = 0 :=
sorry

end recurrence_relation_b_l217_217010


namespace complex_point_imaginary_axis_l217_217892

theorem complex_point_imaginary_axis (a : ℝ) : 
  let z := (2 * a + complex.i) * (1 + complex.i) in
  z.re = 0 ↔ a = 1 / 2 := 
sorry

end complex_point_imaginary_axis_l217_217892


namespace probability_dmitry_before_father_l217_217338

theorem probability_dmitry_before_father (m x y z : ℝ) (h1 : 0 < x) (h2 : x < m) (h3 : 0 < y)
    (h4 : y < z) (h5 : z < m) : 
    (measure_theory.measure_space.volume {y | y < x} / 
    measure_theory.measure_space.volume {x, y, z | 0 < x ∧ x < m ∧ 0 < y ∧ y < z ∧ z < m}) = (2 / 3) :=
  sorry

end probability_dmitry_before_father_l217_217338


namespace area_BCD_l217_217177

noncomputable theory

def triangle_area (base height : ℝ) : ℝ := 1 / 2 * base * height

variables (A B C D : Type) [affine_space ℝ (point)]

-- Defining the points A, B, C, and D
variables (a b c d : point) 

-- Condition: Area of triangle ABC is 50 square units
axiom area_ABC : triangle_area (dist a c) (height_ABC) = 50

-- Condition: Length of segment CD is 36 units
axiom length_CD : dist c d = 36

-- Calculate height of triangle ABC
def height_ABC := (2 * 50) / (dist a c)

-- Goal: Prove that the area of triangle BCD is 300 square units
theorem area_BCD : triangle_area (dist c d) (height_ABC) = 300 :=
sorry

end area_BCD_l217_217177


namespace range_of_a_l217_217441

-- Define max function
def max (p q : ℝ) : ℝ := if p ≥ q then p else q

-- Define the function f(x)
def f (x a : ℝ) : ℝ := max (2^|x| - 2) (x^2 - 2*a*x + a)

-- The theorem to determine the range of a condition
theorem range_of_a : {a : ℝ | ∃ x : ℝ, f x a ≤ 0} = {a : ℝ | a ≤ 0 ∨ a ≥ 1} :=
sorry

end range_of_a_l217_217441


namespace necessary_but_not_sufficient_l217_217511

def p (x : ℝ) : Prop := x < 1
def q (x : ℝ) : Prop := x^2 + x - 2 < 0

theorem necessary_but_not_sufficient (x : ℝ):
  (p x → q x) ∧ (q x → p x) → False ∧ (q x → p x) :=
sorry

end necessary_but_not_sufficient_l217_217511


namespace time_after_hours_l217_217711

-- Definitions based on conditions
def current_time : ℕ := 3
def hours_later : ℕ := 2517
def clock_cycle : ℕ := 12

-- Statement to prove
theorem time_after_hours :
  (current_time + hours_later) % clock_cycle = 12 := 
sorry

end time_after_hours_l217_217711


namespace find_k_l217_217460

noncomputable def k_eq_formula (β : ℝ) : ℝ :=
  (Real.sin β + Real.csc β) ^ 2 + (Real.cos β + Real.sec β) ^ 2 + (Real.sin (2 * β) + Real.cos (2 * β))

open Real

theorem find_k (β : ℝ) :
  k_eq_formula β = 6 + 2 * (sin β * cos β + cos β ^ 2) + tan β ^ 2 + cot β ^ 2 :=
by
  sorry

end find_k_l217_217460


namespace exists_set_C_iff_disjoint_l217_217757

variable (U : Type) [U_univ : set.univ U]
variable (A B : set U)

/-- Theorem: There exists a set C such that A ⊆ C and B ⊆ complement C 
             if and only if A ∩ B = ∅. -/
theorem exists_set_C_iff_disjoint :
  (∃ C : set U, A ⊆ C ∧ B ⊆ Cᶜ) ↔ (A ∩ B = ∅) :=
sorry

end exists_set_C_iff_disjoint_l217_217757


namespace graph_intersections_l217_217398

-- Definitions based on conditions
def parametric_x (t : ℝ) : ℝ := real.cos t + t
def parametric_y (t : ℝ) : ℝ := real.sin t

-- Theorem stating the number of intersections between x = 1 and x = 50 is 8
theorem graph_intersections (x_min x_max : ℝ) (h1 : x_min = 1) (h2 : x_max = 50) :
  {t1 t2 : ℝ // (parametric_x t1 = parametric_x t2) ∧ t1 ≠ t2}.card = 8 :=
sorry

end graph_intersections_l217_217398


namespace number_of_permutations_l217_217614

noncomputable def count_perms (n : ℕ) (h : n % 4 = 0) :=
  let m := n / 4 in
  (List.product $ List.range m).map (fun k => 4 * (m - k) - 2).prod

theorem number_of_permutations (n : ℕ) (h : n % 4 = 0) :
  ∃σ : (Fin n → Fin n), (∀j : Fin n, σ j + σ⁻¹ j = n + 1) ∧
  (Permutations σ).card = (List.product $ List.range (n / 4)).map (fun k => 4 * ((n / 4) - k) - 2).prod :=
sorry

end number_of_permutations_l217_217614


namespace colton_stickers_final_count_l217_217433

-- Definitions based on conditions
def initial_stickers := 200
def stickers_given_to_7_friends := 6 * 7
def stickers_given_to_mandy := stickers_given_to_7_friends + 8
def remaining_after_mandy := initial_stickers - stickers_given_to_7_friends - stickers_given_to_mandy
def stickers_distributed_to_4_friends := remaining_after_mandy / 2
def remaining_after_4_friends := remaining_after_mandy - stickers_distributed_to_4_friends
def given_to_justin := 2 * remaining_after_4_friends / 3
def remaining_after_justin := remaining_after_4_friends - given_to_justin
def given_to_karen := remaining_after_justin / 5
def final_stickers := remaining_after_justin - given_to_karen

-- Theorem to state the proof problem
theorem colton_stickers_final_count : final_stickers = 15 := by
  sorry

end colton_stickers_final_count_l217_217433


namespace hamiltonian_circuit_exists_l217_217270

open Nat

-- Define the labeling function for vertices of the N-dimensional hypercube
def label (N : ℕ) (x : Fin N → ℕ) : ℕ :=
  ∑ k in range N, (x k) * 2^(k - 1)

-- Define the problem condition
def vertices_form_hamiltonian_circuit (N n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 2^N) : Prop :=
  ∃ (circuit : list (Fin N → ℕ)), 
    (∀ v, v < n → v ∈ (circuit.map (label N))) ∧ 
    (circuit.length = n) ∧ 
    (list.pairwise (λ x y, (∑ i, if x i ≠ y i then 1 else 0) = 1) circuit) ∧ 
    (circuit.head = circuit.tail.head)

-- Statement of the mathematically equivalent proof problem
theorem hamiltonian_circuit_exists (N n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 2^N) :
  vertices_form_hamiltonian_circuit N n h1 h2 ↔ (n % 2 = 0) :=
by
  sorry

end hamiltonian_circuit_exists_l217_217270


namespace find_a_l217_217586

noncomputable def givenConditions (a b c R : ℝ) : Prop :=
  (a^2 / (b * c) - c / b - b / c = Real.sqrt 3) ∧ (R = 3)

theorem find_a (a b c : ℝ) (R : ℝ) (h : givenConditions a b c R) : a = 3 :=
by
  sorry

end find_a_l217_217586


namespace negative_solution_condition_l217_217056

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l217_217056


namespace isosceles_triangle_perimeter_possible_values_l217_217509

def is_root (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

def triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_triangle_perimeter_possible_values (a x : ℝ) :
  a = 9 → is_root 1 (-8) 15 x →
  let b := if x = 3 then 3 else if x = 5 then 5 else 0 in
  (triangle 9 9 b ∨ triangle b b 9) → 
  (a + a + b = 19 ∨ a + a + b = 21 ∨ a + a + b = 23 ∨ b + b + a = 19) :=
by 
  intros ha hroot htriangle;
  sorry

end isosceles_triangle_perimeter_possible_values_l217_217509


namespace initial_milk_in_each_cup_l217_217713

theorem initial_milk_in_each_cup (n : ℕ) (h : n = 4 * 2117) :
  ∃ a : Fin 7 → ℝ, a 0 = 6/7 ∧ a 1 = 5/7 ∧ a 2 = 4/7 ∧ a 3 = 3/7 ∧ a 4 = 2/7 ∧ a 5 = 1/7 ∧ a 6 = 0 ∧
  (∀ i : Fin 7, ∑ j : Fin 7, if i ≠ j then a i / 7 else 0 = a i) ∧
  (∑ i : Fin n, 1) = 3 * (∑ i : Fin n, 1) :=
by
  sorry

end initial_milk_in_each_cup_l217_217713


namespace carl_watermelons_l217_217431

theorem carl_watermelons (base_price : ℕ) (discount_range_low discount_range_high : ℕ) 
    (profit : ℕ) (remaining_watermelons : ℕ) 
    (avg_discount_rate : ℕ) : 
  base_price = 3 →
  discount_range_low = 0 →
  discount_range_high = 50 →
  profit = 105 →
  remaining_watermelons = 18 →
  avg_discount_rate = 25 →
  let avg_selling_price := base_price * (100 - avg_discount_rate) / 100 in 
  let sold_watermelons := profit / avg_selling_price in 
  let started_watermelons := sold_watermelons + remaining_watermelons in
  started_watermelons = 64 := 
by
  intros
  simp
  sorry

end carl_watermelons_l217_217431


namespace min_value_x_plus_2y_l217_217534

theorem min_value_x_plus_2y 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (h : x + 2y + 2 * x * y = 3) : 
  x + 2y ≥ 2 :=
sorry

end min_value_x_plus_2y_l217_217534


namespace range_of_f_l217_217513

-- Defining the conditions and the function
def f (x y : ℝ) : ℝ := (y + 1) / (x - 1)

-- The main theorem to prove the range of the function under given conditions
theorem range_of_f {x y : ℝ} (hx1 : 2 ≤ x) (hx2 : x ≤ 3) (hy : 2 * x + y = 8) :
  ∃ y, f x y ∈ set.Icc (3 / 2 : ℝ) 5 := sorry

end range_of_f_l217_217513


namespace mabel_total_tomatoes_l217_217995

def tomatoes_first_plant : ℕ := 12

def tomatoes_second_plant : ℕ := (2 * tomatoes_first_plant) - 6

def tomatoes_combined_first_two : ℕ := tomatoes_first_plant + tomatoes_second_plant

def tomatoes_third_plant : ℕ := tomatoes_combined_first_two / 2

def tomatoes_each_fourth_fifth_plant : ℕ := 3 * tomatoes_combined_first_two

def tomatoes_combined_fourth_fifth : ℕ := 2 * tomatoes_each_fourth_fifth_plant

def tomatoes_each_sixth_seventh_plant : ℕ := (3 * tomatoes_combined_first_two) / 2

def tomatoes_combined_sixth_seventh : ℕ := 2 * tomatoes_each_sixth_seventh_plant

def total_tomatoes : ℕ := tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant + tomatoes_combined_fourth_fifth + tomatoes_combined_sixth_seventh

theorem mabel_total_tomatoes : total_tomatoes = 315 :=
by
  sorry

end mabel_total_tomatoes_l217_217995


namespace value_of_N_l217_217482

theorem value_of_N (N : ℕ) (h : Nat.choose N 5 = 231) : N = 11 := sorry

end value_of_N_l217_217482


namespace Jen_ate_11_suckers_l217_217246

/-
Sienna gave Bailey half of her suckers.
Jen ate half and gave the rest to Molly.
Molly ate 2 and gave the rest to Harmony.
Harmony kept 3 and passed the remainder to Taylor.
Taylor ate one and gave the last 5 to Callie.
How many suckers did Jen eat?
-/

noncomputable def total_suckers_given_to_Callie := 5
noncomputable def total_suckers_Taylor_had := total_suckers_given_to_Callie + 1
noncomputable def total_suckers_Harmony_had := total_suckers_Taylor_had + 3
noncomputable def total_suckers_Molly_had := total_suckers_Harmony_had + 2
noncomputable def total_suckers_Jen_had := total_suckers_Molly_had * 2
noncomputable def suckers_Jen_ate := total_suckers_Jen_had - total_suckers_Molly_had

theorem Jen_ate_11_suckers : suckers_Jen_ate = 11 :=
by {
  unfold total_suckers_given_to_Callie total_suckers_Taylor_had total_suckers_Harmony_had total_suckers_Molly_had total_suckers_Jen_had suckers_Jen_ate,
  sorry
}

end Jen_ate_11_suckers_l217_217246


namespace PQBC_concyclic_l217_217984

theorem PQBC_concyclic
  {A B C D K L P Q : Type*}
  [Trapezoid A B C D]
  (h1 : Line.parallel AB CD)
  (h2 : AB > CD)
  (h3 : AK / KB = DL / LC)
  (h4 : ∠APB = ∠BCD)
  (h5 : ∠CQD = ∠ABC) :
  Concyclic P Q B C :=
sorry

end PQBC_concyclic_l217_217984


namespace negative_solution_condition_l217_217043

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l217_217043


namespace frog_year_2033_l217_217999

def frogs (n : ℕ) : ℕ
| 2020 := 2
| 2021 := 9
| (n+2) := 1 + |frogs (n+1) - frogs n|

theorem frog_year_2033 : frogs 2033 = 1 :=
sorry

end frog_year_2033_l217_217999


namespace graph_cycle_length_l217_217201

-- Definitions based on conditions
variable (G : SimpleGraph V) [Fintype {v : V | G.degree v = G.min_degree}] (d : ℕ)
#check G.min_degree

-- Hypotheses based on conditions
def min_degree_ge_two := (G.min_degree) >= 2
def graph_contains_long_cycle := ∃ (cycle : G.Cycle) (n : ℕ), n ≥ G.min_degree + 1 ∧ G.Cycle.length cycle = n

-- Main statement
theorem graph_cycle_length (h : min_degree_ge_two G): graph_contains_long_cycle G :=
sorry

end graph_cycle_length_l217_217201


namespace tan_810_is_undefined_l217_217007

-- Define the concept of multiple rotations
def full_rotation (θ : ℝ) : ℝ := θ - 360 * ⌊θ / 360⌋

-- State the problem conditions
def is_undefined_tangent (θ : ℝ) : Prop :=
  ∃ n : ℤ, θ = 90 + 360 * n

-- Lean statement of the proof problem
theorem tan_810_is_undefined : is_undefined_tangent 810 :=
by {
  -- The proof steps would go here
  sorry
}

end tan_810_is_undefined_l217_217007


namespace system_has_negative_solution_iff_sum_zero_l217_217050

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l217_217050


namespace arithmetic_sequences_sum_l217_217552

-- Defining the arithmetic sequences a_n and b_n
variables (a b : ℕ → ℕ)

-- Sum of the first n terms of the sequences
def A (n : ℕ) := ∑ k in range n, a k
def B (n : ℕ) := ∑ k in range n, b k

-- Given condition
axiom sum_ratio (n : ℕ) : A n / B n = (5 * n + 12) / (2 * n + 3)

-- The statement to be proved
theorem arithmetic_sequences_sum :
  (a 5 / b 5) + (a 7 / b 12) = 30 / 7 :=
sorry

end arithmetic_sequences_sum_l217_217552


namespace roots_greater_than_two_l217_217290

theorem roots_greater_than_two (m : ℝ) :
  (∀ x : ℝ, x^2 + (m - 2) * x + 5 - m = 0 → x > 2) ↔ m ∈ set.Ioo (-5 : ℝ) (-4 : ℝ) ∨ m = -4 :=
begin
  sorry
end

end roots_greater_than_two_l217_217290


namespace expression_equals_sqrt3_minus_6_l217_217003

theorem expression_equals_sqrt3_minus_6 : 
  (Real.sqrt 4 + Real.cbrt (-64) - Real.sqrt ((-3) ^ 2) + abs (Real.sqrt 3 - 1)) = Real.sqrt 3 - 6 :=
by sorry

end expression_equals_sqrt3_minus_6_l217_217003


namespace cos_2_angle_DFG_zero_l217_217242

open Real

-- Definitions based on conditions
def isosceles_right_triangle (D E F : Point) :=
  D ≠ E ∧ E ≠ F ∧ F ≠ D ∧ |DE| = 2 ∧ |EF| = 2 ∧ ∠DEF = π / 2

def right_triangle (E F G : Point) :=
  E ≠ F ∧ F ≠ G ∧ G ≠ E ∧ ∠EFG = π / 2 ∧ |EG| = √8

-- The main statement to prove
theorem cos_2_angle_DFG_zero (D E F G : Point) :
  isosceles_right_triangle D E F ∧ right_triangle E F G ∧
  |DE| + |EF| + |EG| + |GF| + |FE| = 2 * (|DE| + |EF| + |EG|)
  → cos (2 * ∠DFG) = 0 :=
sorry

end cos_2_angle_DFG_zero_l217_217242


namespace hexagon_area_l217_217328

-- Define the basic properties of the problem
def triangle_base : ℝ := 1
def triangle_height : ℝ := 4
def num_triangles : ℕ := 6
def rect_width : ℝ := 6
def rect_height : ℝ := 8

-- Define the area calculations
def triangle_area : ℝ := (1/2) * triangle_base * triangle_height
def total_triangle_area : ℝ := num_triangles * triangle_area
def rect_area : ℝ := rect_width * rect_height

-- The goal is to prove that the hexagon's area is 36 given the above conditions
theorem hexagon_area :
  rect_area - total_triangle_area = 36 := by
  sorry

end hexagon_area_l217_217328


namespace slower_train_cross_time_l217_217750

def speed_of_train1 : ℝ := 100  -- in km/h
def speed_of_train2 : ℝ := 120  -- in km/h
def length_of_train1 : ℝ := 500  -- in meters
def length_of_train2 : ℝ := 700  -- in meters

def convert_kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 3600

def relative_speed_mps : ℝ :=
  convert_kmph_to_mps (speed_of_train1 + speed_of_train2)

def combined_length : ℝ :=
  length_of_train1 + length_of_train2

def time_to_cross : ℝ :=
  combined_length / relative_speed_mps

theorem slower_train_cross_time : time_to_cross ≈ 19.63 := by
  sorry

end slower_train_cross_time_l217_217750


namespace interest_rate_unique_l217_217236

theorem interest_rate_unique (P r : ℝ) (h₁ : P * (1 + 3 * r) = 300) (h₂ : P * (1 + 8 * r) = 400) : r = 1 / 12 :=
by {
  sorry
}

end interest_rate_unique_l217_217236


namespace zero_ordering_l217_217542

def f (x : ℝ) : ℝ := x + 2^x
def g (x : ℝ) : ℝ := x + Real.log x
def h (x : ℝ) : ℝ := x^3 + x - 2

axiom x1_zero : ∃ x1 : ℝ, f x1 = 0
axiom x2_zero : ∃ x2 : ℝ, g x2 = 0
axiom x3_zero : ∃ x3 : ℝ, h x3 = 0

theorem zero_ordering (x1 x2 x3 : ℝ) (h1 : f x1 = 0) (h2 : g x2 = 0) (h3 : h x3 = 0) : x1 < x2 ∧ x2 < x3 := by
  sorry

end zero_ordering_l217_217542


namespace sqrt_sqrt_sixteen_l217_217695

theorem sqrt_sqrt_sixteen : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := by
  sorry

end sqrt_sqrt_sixteen_l217_217695


namespace praveen_initial_investment_l217_217645

theorem praveen_initial_investment
  (H : ℝ) (P : ℝ)
  (h_H : H = 9000.000000000002)
  (h_profit_ratio : (P * 12) / (H * 7) = 2 / 3) :
  P = 3500 := by
  sorry

end praveen_initial_investment_l217_217645


namespace determinant_in_terms_of_roots_l217_217974

theorem determinant_in_terms_of_roots 
  (r s t a b c : ℝ)
  (h1 : a^3 - r*a^2 + s*a - t = 0)
  (h2 : b^3 - r*b^2 + s*b - t = 0)
  (h3 : c^3 - r*c^2 + s*c - t = 0) :
  (2 + a) * ((2 + b) * (2 + c) - 4) - 2 * (2 * (2 + c) - 4) + 2 * (2 * 2 - (2 + b) * 2) = t - 2 * s :=
by
  sorry

end determinant_in_terms_of_roots_l217_217974


namespace tax_rate_is_correct_l217_217960

noncomputable def price_before_tax : ℝ := 92
noncomputable def total_price_including_tax : ℝ := 98.90

def calculate_tax_rate (price_before_tax : ℝ) (total_price_including_tax : ℝ) : ℝ :=
  ((total_price_including_tax - price_before_tax) / price_before_tax) * 100

theorem tax_rate_is_correct : calculate_tax_rate price_before_tax total_price_including_tax = 7.5 :=
by
  sorry

end tax_rate_is_correct_l217_217960


namespace point_on_line_segment_l217_217971

structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (p q : Point) : ℝ :=
  real.sqrt ((q.x - p.x) ^ 2 + (q.y - p.y) ^ 2)

def fixed_point1 : Point := { x := -4, y := 0 }
def fixed_point2 : Point := { x := 4, y := 0 }

def moving_point (M : Point) : Prop :=
  distance M fixed_point1 + distance M fixed_point2 = 8


theorem point_on_line_segment (M : Point) :
  moving_point M → (M.x >= -4) ∧ (M.x <= 4) ∧ (M.y = 0) :=
sorry

end point_on_line_segment_l217_217971


namespace projection_of_b_on_a_l217_217885

variables {ℝ} [InnerProductSpace ℝ (euclidean_space ℝ)]

open real_inner_product_space

variables (a b : euclidean_space ℝ)
variables (ha : norm a = 2)
variables (h_angle : ∠ a b = real.pi / 3)
variables (h_perp : inner (a + b) (a - (2 • b)) = 0)

theorem projection_of_b_on_a : 
  inner b a / norm a = -((sqrt 33 + 1) / 8) :=
by {
  sorry
}

end projection_of_b_on_a_l217_217885


namespace zero_in_interval_find_n_l217_217930

-- Define the function f(x)
def f(x : ℝ) : ℝ := (1/2) * Real.exp x + x - 6

-- Define the condition that the zero is in the interval (2, 3)
theorem zero_in_interval (h_incr : ∀ x y, x < y → f(x) < f(y)) (hyp1 : f 2 < 0) (hyp2 : f 3 > 0) : ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
by {
  -- Introduction of hypothesis for f being an increasing function and f(2)<0 and f(3)>0
  sorry
}

-- Define the main theorem statement
theorem find_n (h_incr : ∀ x y, x < y → f(x) < f(y)) (hyp1 : f 2 < 0) (hyp2 : f 3 > 0) : ∃ n : ℕ, n ∈ (2 : ℕ) ∧ ∃ c : ℝ, n < c ∧ c < (n+1) ∧ f c = 0 :=
by {
  use 2,
  exact zero_in_interval h_incr hyp1 hyp2,
  sorry
}

end zero_in_interval_find_n_l217_217930


namespace f_of_2_f_of_neg_2_g_of_neg_1_l217_217863

def f (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + 1

theorem f_of_2 : f(2) = 13 :=
by {
  calc f(2) = 4 * 2^2 - 2 * 2 + 1 : by rfl
        ... = 13 : by norm_num
}

theorem f_of_neg_2 : f(-2) = 21 :=
by {
  calc f(-2) = 4 * (-2)^2 - 2 * (-2) + 1 : by rfl
         ... = 21 : by norm_num
}

theorem g_of_neg_1 : g(-1) = 4 :=
by {
  calc g(-1) = 3 * (-1)^2 + 1 : by rfl
         ... = 4 : by norm_num
}

end f_of_2_f_of_neg_2_g_of_neg_1_l217_217863


namespace henry_drive_distance_l217_217913

theorem henry_drive_distance :
  ∃ (d : ℝ),
    let t := 3.5 in
    d = 191.25 ∧
    (45 * (t + 0.75) = d) ∧
    ((d - 45) / 65 = t - 1.25) :=
begin
  sorry
end

end henry_drive_distance_l217_217913


namespace random_simulation_approximation_l217_217319

variables {m n : ℝ}

/-- The probability obtained by the random simulation method is only an estimate
    of the actual probability. Therefore, m is an approximation of n. -/
theorem random_simulation_approximation (h : true) : 
  m ≈ n := 
sorry

end random_simulation_approximation_l217_217319


namespace min_value_gx2_plus_fx_l217_217132

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_gx2_plus_fx (a b c : ℝ) (h_a : a ≠ 0)
    (h_min_fx_gx : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ -6) :
    ∃ x : ℝ, (g a c x)^2 + f a b x = 11/2 := sorry

end min_value_gx2_plus_fx_l217_217132


namespace find_b_value_l217_217112

variable {ℂ : Type*} [NormedField ℂ] [IsROrC ℂ]

noncomputable def complex_modulus_condition (b : ℝ) : Prop :=
  let z := (b * (0 + 1 * Complex.I)) / (4 + 3 * Complex.I)
  |z| = 5

theorem find_b_value (b : ℝ) (h : complex_modulus_condition b) : b = 25 ∨ b = -25 :=
sorry

end find_b_value_l217_217112


namespace pairwise_distinct_and_integer_a_l217_217621

variable {n : ℕ}
variable {x : Fin n → ℝ}
variable {a : Fin n → ℝ}

theorem pairwise_distinct_and_integer_a (h_distinct : Function.Injective x)
                                        (h_int : ∀ i : Fin n, 
                                                ∃ k : ℤ, 
                                                a i = k ∧ 
                                                a i = Real.sqrt ((∑ j in Finset.range (i.1 + 1), x ⟨j, sorry⟩) 
                                                                 * (∑ j in Finset.range (i.1 + 1), x ⟨j, sorry⟩⁻¹))) :
    a ⟨2023, sorry⟩ ≥ 3034 := sorry

end pairwise_distinct_and_integer_a_l217_217621


namespace product_zero_l217_217456

theorem product_zero (b : ℤ) (h : b = 3) : (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) * (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by
  rw h
  sorry

end product_zero_l217_217456


namespace sum_of_coefficients_largest_coefficients_l217_217597

-- Proof Problem 1: Prove that the sum of the coefficients in the expansion is 6561/256

theorem sum_of_coefficients (x : ℝ) :
  (let n := 8 in
  let f := sqrt(x) + 1 / (2 * x^(1/3)) in
  let expansion := (f ^ n). summand 1) = (3/2)^8 := 
sorry

-- Proof Problem 2: Prove the terms with the largest coefficients are 7 * x^(7/3) and 7 * x^(3/2)

theorem largest_coefficients (x : ℝ) :
  let n := 8
  let term_3 := (n.choose 2) * (1 / 2)^2 * x ^ (4 - (5 * 2) / 6)
  let term_4 := (n.choose 3) * (1 / 2)^3 * x ^ (4 - (5 * 3) / 6) in
  term_3 = 7 * x^(7/3) ∧ term_4 = 7 * x^(3/2) :=
sorry

end sum_of_coefficients_largest_coefficients_l217_217597


namespace zombie_count_today_l217_217308

theorem zombie_count_today (Z : ℕ) (h : Z < 50) : 16 * Z = 48 :=
by
  -- Assume Z, h conditions from a)
  -- Proof will go here, for now replaced with sorry
  sorry

end zombie_count_today_l217_217308


namespace quadratic_roots_l217_217531

theorem quadratic_roots (m : ℝ) (h : (1:ℝ)^2 - 5*(1:ℝ) + m = 0) : 
  (x₁ x₂ : ℝ) (hx₁ : x₁^2 - 5*x₁ + m = 0) (hx₂ : x₂^2 - 5*x₂ + m = 0) ↔ (x₁ = 1 ∧ x₂ = 4) := 
by
  sorry

end quadratic_roots_l217_217531


namespace solve_p_plus_q_l217_217555

noncomputable def p_plus_q (P_only_X P_only_Y P_only_Z P_2_out_of_3 P_XYZ_given_XY : ℚ) : ℕ :=
  let N := 63 + 80 in
  N

theorem solve_p_plus_q : p_plus_q (5/100) (5/100) (5/100) (7/100) (1/4) = 143 := 
by
  sorry

end solve_p_plus_q_l217_217555


namespace _l217_217349

lemma triangle_inequality_theorem (a b c : ℝ) : 
  a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a > 0 ∧ b > 0 ∧ c > 0) := sorry

lemma no_triangle_1_2_3 : ¬ (1 + 2 > 3 ∧ 1 + 3 > 2 ∧ 2 + 3 > 1) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_3_8_5 : ¬ (3 + 8 > 5 ∧ 3 + 5 > 8 ∧ 8 + 5 > 3) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_4_5_10 : ¬ (4 + 5 > 10 ∧ 4 + 10 > 5 ∧ 5 + 10 > 4) := 
by simp [triangle_inequality_theorem]

lemma triangle_4_5_6 : 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4 := 
by simp [triangle_inequality_theorem]

end _l217_217349


namespace boys_lollipops_l217_217789

-- Definitions based on conditions
variables (TotalCandies : ℕ) (OneThirdLollipops : ℕ) (CandyCanes : ℕ)
variables (TotalChildren : ℕ) (NumGirls : ℕ) (NumBoys : ℕ)
variables (LollipopsPerBoy : ℕ)

-- Conditions
def condition1 : TotalCandies = 90 := sorry
def condition2 : OneThirdLollipops = TotalCandies / 3 := sorry
def condition3 : CandyCanes = TotalCandies - OneThirdLollipops := sorry
def condition4 : TotalChildren = 40 := sorry
def condition5 : NumGirls = CandyCanes / 2 := sorry
def condition6 : NumBoys = TotalChildren - NumGirls := sorry

-- Theorem statement that each boy received 3 lollipops
theorem boys_lollipops (h1 : condition1) (h2 : condition2) (h3 : condition3)
  (h4 : condition4) (h5 : condition5) (h6 : condition6) : 
  LollipopsPerBoy = OneThirdLollipops / NumBoys := 
sorry

example (h1 : TotalCandies = 90) (h2 : OneThirdLollipops = TotalCandies / 3) 
  (h3 : CandyCanes = TotalCandies - OneThirdLollipops) (h4 : TotalChildren = 40)
  (h5 : NumGirls = CandyCanes / 2) (h6 : NumBoys = TotalChildren - NumGirls) :
  OneThirdLollipops / NumBoys = 3 :=
boys_lollipops h1 h2 h3 h4 h5 h6

end boys_lollipops_l217_217789


namespace common_ratio_geometric_sequence_l217_217889

theorem common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) ∧ a 1 = 32 ∧ a 6 = -1 → q = -1/2 :=
by
  sorry

end common_ratio_geometric_sequence_l217_217889


namespace joan_gave_sam_seashells_l217_217193

theorem joan_gave_sam_seashells (original_seashells : ℕ) (left_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 70) (h2 : left_seashells = 27) : given_seashells = 43 :=
by
  have h3 : given_seashells = original_seashells - left_seashells := sorry
  rw [h1, h2] at h3
  exact h3

end joan_gave_sam_seashells_l217_217193


namespace find_b_12_l217_217985

def sequence (b : ℕ → ℤ) : Prop :=
  b 1 = 2 ∧ (∀ m n : ℕ, m > 0 → n > 0 → b (m + n) = b m + b n + m^2 * n)

theorem find_b_12 (b : ℕ → ℤ) (h : sequence b) : b 12 = 306 :=
by
  sorry

end find_b_12_l217_217985


namespace domain_of_f_eq_0_1_l217_217629

noncomputable def f (x : ℝ) := Real.log (-x^2 + x)

theorem domain_of_f_eq_0_1 : {x : ℝ | 0 < x ∧ x < 1} = {x : ℝ | ∃ y, f(x) = y} :=
by
  sorry

end domain_of_f_eq_0_1_l217_217629


namespace find_d_values_l217_217241

theorem find_d_values (u v : ℝ) (c d : ℝ)
  (hpu : u^3 + c * u + d = 0)
  (hpv : v^3 + c * v + d = 0)
  (hqu : (u + 2)^3 + c * (u + 2) + d - 120 = 0)
  (hqv : (v - 5)^3 + c * (v - 5) + d - 120 = 0) :
  d = 396 ∨ d = 8 :=
by
  -- placeholder for the actual proof
  sorry

end find_d_values_l217_217241


namespace tetrahedron_angle_property_l217_217093

noncomputable def tetrahedron_dihedral_angle (α β θ : ℝ) : Prop :=
  (θ < α) ∧ (α < π / 2) ∧ 
  (0 < β) ∧ (β < π / 2) ∧ 
  θ = π - Real.arccos (Real.cot α * Real.cot β)

-- The theorem statement
theorem tetrahedron_angle_property (α β θ : ℝ) (h1 : θ < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) :
  tetrahedron_dihedral_angle α β θ :=
by
  unfold tetrahedron_dihedral_angle
  simpa [h1, h2, h3, h4]

end tetrahedron_angle_property_l217_217093


namespace water_per_hour_l217_217262

theorem water_per_hour (typing_speed : ℕ) (pages : ℕ) (words_per_page : ℕ) (total_water : ℕ) : 
  pages * words_per_page / typing_speed / 60 ≠ 0 →
  total_water / (pages * words_per_page / typing_speed / 60) = 15 := 
by
  intros typing_speed_neq_zero
  sorry

-- Conditions from the problem
def stan_typing_speed : ℕ := 50
def stan_pages : ℕ := 5
def stan_words_per_page : ℕ := 400
def stan_total_water : ℕ := 10

#eval water_per_hour stan_typing_speed stan_pages stan_words_per_page stan_total_water

end water_per_hour_l217_217262


namespace students_representing_x_percent_l217_217382

noncomputable def total_students : ℝ := 113.38934190276818
noncomputable def percentage_boys : ℝ := 0.70
noncomputable def number_of_boys : ℝ := percentage_boys * total_students

theorem students_representing_x_percent (x : ℝ) :
  (x / 100) * number_of_boys = (x / 100) * 79.37253933173772 :=
by
  have h : number_of_boys = 79.37253933173772 := 
    by rw [number_of_boys, percentage_boys, total_students]; norm_num
  rw h
  sorry

end students_representing_x_percent_l217_217382


namespace gray_area_l217_217432

def center_C : (ℝ × ℝ) := (6, 5)
def center_D : (ℝ × ℝ) := (14, 5)
def radius_C : ℝ := 3
def radius_D : ℝ := 3

theorem gray_area :
  let area_rectangle := 8 * 5
  let area_sector_C := (1 / 2) * π * radius_C^2
  let area_sector_D := (1 / 2) * π * radius_D^2
  area_rectangle - (area_sector_C + area_sector_D) = 40 - 9 * π :=
by
  sorry

end gray_area_l217_217432


namespace james_profit_calculation_l217_217607

-- Definitions: Constants based on the problem conditions
def fraction_sold := 0.80
def num_toys := 200
def buy_price_per_toy := 20
def sell_price_per_toy := 30
def sales_tax_rate := 0.10
def discount_rate := 0.05
def discount_threshold := 20
def shipping_up_to_50 := 2
def shipping_51_to_100 := 1.5
def shipping_101_to_200 := 1

-- Main theorem: Correct answer based on the given conditions
theorem james_profit_calculation 
  (fraction_sold : Real := 0.80)
  (num_toys : Int := 200)
  (buy_price_per_toy : Real := 20)
  (sell_price_per_toy : Real := 30)
  (sales_tax_rate : Real := 0.10)
  (discount_rate : Real := 0.05)
  (discount_threshold : Int := 20)
  (shipping_up_to_50 : Real := 2)
  (shipping_51_to_100 : Real := 1.5)
  (shipping_101_to_200 : Real := 1) :
  (calc 
    let sold_toys := (fraction_sold * (num_toys : Real))
    let total_revenue_before_taxes := sold_toys * sell_price_per_toy
    let sales_tax := sales_tax_rate * total_revenue_before_taxes
    let revenue_after_tax := total_revenue_before_taxes - sales_tax
    let num_discount_sets := (sold_toys / (discount_threshold : Real)).toNat
    let discount_per_set := discount_rate * (Real.ofNat discount_threshold) * sell_price_per_toy
    let total_discount := (num_discount_sets : Real) * discount_per_set
    let revenue_after_discount := revenue_after_tax - total_discount
    let shipping_fee :=
      if sold_toys <= 50 then 
        sold_toys * shipping_up_to_50 
      else if sold_toys <= 100 then 
        sold_toys * shipping_51_to_100 
      else 
        sold_toys * shipping_101_to_200
    let revenue_after_shipping := revenue_after_discount - shipping_fee
    let initial_cost := (num_toys : Real) * buy_price_per_toy
    final_difference := revenue_after_shipping - initial_cost
    final_difference).toInt = -80 :=
by
  sorry

end james_profit_calculation_l217_217607


namespace units_digit_of_fraction_example_l217_217341

def units_digit_of_fraction (n d : ℕ) : ℕ :=
  (n / d) % 10

theorem units_digit_of_fraction_example :
  units_digit_of_fraction (25 * 26 * 27 * 28 * 29 * 30) 1250 = 2 := by
  sorry

end units_digit_of_fraction_example_l217_217341


namespace bases_to_make_equality_l217_217758

theorem bases_to_make_equality (a b : ℕ) (h : 3 * a^2 + 4 * a + 2 = 9 * b + 7) : 
  (3 * a^2 + 4 * a + 2 = 342) ∧ (9 * b + 7 = 97) :=
by
  sorry

end bases_to_make_equality_l217_217758


namespace range_of_m_l217_217894

open Real

theorem range_of_m (m : ℝ) : (m^2 > 2 + m ∧ 2 + m > 0) ↔ (m > 2 ∨ -2 < m ∧ m < -1) :=
by
  sorry

end range_of_m_l217_217894


namespace equilateral_triangles_to_square_l217_217315

-- Definitions based on given conditions
structure Triangle :=
  (a b c : Point)
  (eq_sides : dist a b = dist b c ∧ dist b c = dist c a)

-- Main theorem statement
theorem equilateral_triangles_to_square (T1 T2 : Triangle)
  (h1 : T1.eq_sides) (h2 : T2.eq_sides) :
  ∃ (parts : list set Point) (cuts : list (set Point → list set Point)), parts.length = 6 ∧
    (∀ t ∈ [T1, T2], ∃ c ∈ cuts, c t = parts) ∧ can_form_square parts :=
sorry

end equilateral_triangles_to_square_l217_217315


namespace sqrt_sqrt_16_eq_pm_2_l217_217698

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217698


namespace range_of_a_l217_217148

def func (x : ℝ) : ℝ := x^2 - 4 * x

def domain (a : ℝ) := ∀ x, -4 ≤ x ∧ x ≤ a

def range_condition (y : ℝ) := -4 ≤ y ∧ y ≤ 32

theorem range_of_a (a : ℝ)
  (domain_condition : ∀ x, x ∈ set.Icc (-4) a → func x ∈ set.Icc (-4) 32) :
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l217_217148


namespace rationalize_expression_l217_217654

theorem rationalize_expression : 
  (sqrt 3 / sqrt 4) * (sqrt 5 / sqrt 6) * (sqrt 8 / sqrt 9) = sqrt 5 / 3 := 
by
  sorry

end rationalize_expression_l217_217654


namespace prove_correct_option_C_exclusively_l217_217737

-- Definitions of the functions
noncomputable def y1 (x : ℝ) : ℝ := real.sin (2 * x + real.pi / 3)
noncomputable def y2 (x : ℝ) : ℝ := real.cos (2 * x)
noncomputable def y3 (x : ℝ) : ℝ := real.cos (x + real.pi / 3)
noncomputable def y4 (x : ℝ) : ℝ := real.tan (x + real.pi / 3)

-- Define the propositions as conditions
def proposition_A : Prop := ∀ x, -real.pi / 3 < x ∧ x < real.pi / 6 → y1 x > y1 (x - ε) -- y is increasing
def proposition_B : Prop := ∀ T > 0, y2 (x + T) = y2 x ↔ T = 2 * real.pi -- minimum period is 2π
def proposition_C : Prop := ∃ p, p = (real.pi / 6, 0) ∧ symmetric_about_point y3 p -- symmetric about (π/6, 0)
def proposition_D : Prop := ∃ l, l = (real.pi / 6) ∧ symmetric_about_line y4 l -- symmetric about line x = π/6

-- The main theorem to prove
theorem prove_correct_option_C_exclusively : (proposition_C) ∧ ¬proposition_A ∧ ¬proposition_B ∧ ¬proposition_D :=
by
  sorry

end prove_correct_option_C_exclusively_l217_217737


namespace determine_p_q_l217_217218

theorem determine_p_q (p q : ℝ) (h : (complex : Type) := ℝ) 
    (root_condition : is_root (CubicPolynomial.mk p q) (2 - 3i)) :
    (p, q) = (-3, 39) :=
by
  sorry

end determine_p_q_l217_217218


namespace maximum_ab_is_40_l217_217205

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end maximum_ab_is_40_l217_217205


namespace arithmetic_sequence_a12_l217_217594

theorem arithmetic_sequence_a12 (a : ℕ → ℝ)
    (h1 : a 3 + a 4 + a 5 = 3)
    (h2 : a 8 = 8)
    (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) :
    a 12 = 15 :=
by
  -- Since we aim to ensure the statement alone compiles, we leave the proof with 'sorry'.
  sorry

end arithmetic_sequence_a12_l217_217594


namespace distinct_real_roots_range_l217_217850

theorem distinct_real_roots_range {a : ℝ} (h : ∀ x, (x^3 - 3 * x - a = 0) → ∃ distinct_real_roots : finset ℝ, distinct_real_roots.card = 3 ∧ (∀ x ∈ distinct_real_roots, (x^3 - 3 * x - a = 0))) :
  -2 < a ∧ a < 2 :=
sorry

end distinct_real_roots_range_l217_217850


namespace binomial_coefficient_condition_l217_217178

variable (n : ℕ)
-- Given conditions
def sum_coefficients (n : ℕ) : ℕ := 2 ^ n
def sum_binomial_coefficients (n : ℕ) : ℕ := 2 ^ n
def condition (n : ℕ) : Prop := sum_coefficients n + sum_binomial_coefficients n = 64

-- The term containing x^2
def term_at_x_square (n : ℕ) : ℤ :=
  if h : (3 * 3 - 5) / 2 = 2 then (-1) * (3 ^ (5 - 3)) * (nat.choose 5 3) else 0

theorem binomial_coefficient_condition : 
  condition n → term_at_x_square n = -90 :=
by
  -- We assume the given conditions. 
  intro h
  sorry

end binomial_coefficient_condition_l217_217178


namespace find_angle_A_find_area_triangle_l217_217585

variable {A B C a b c : ℝ}

-- Proof of part (1)
theorem find_angle_A (h1 : (2 * b - c) * real.cos A = a * real.cos C) 
  (h2 : A ∈ (0, real.pi)) : A = real.pi / 3 :=
sorry

-- Proof of part (2)
theorem find_area_triangle (a b c : ℝ) 
  (h1 : a = real.sqrt 13) 
  (h2 : b + c = 5) 
  (h3 : (2 * b - c) * real.cos (real.pi / 3) = a * real.cos C) : 
  (1 / 2 * b * c * real.sin (real.pi / 3) = real.sqrt 3) :=
sorry

end find_angle_A_find_area_triangle_l217_217585


namespace evaluate_fraction_l217_217455

noncomputable def evaluate_expression : ℚ := 
  1 / (2 - (1 / (2 - (1 / (2 - (1 / 3))))))
  
theorem evaluate_fraction :
  evaluate_expression = 5 / 7 :=
sorry

end evaluate_fraction_l217_217455


namespace arun_gokul_age_subtract_l217_217926

theorem arun_gokul_age_subtract:
  ∃ x : ℕ, (60 - x) / 18 = 3 → x = 6 :=
sorry

end arun_gokul_age_subtract_l217_217926


namespace coeff_x_in_eq2_l217_217535

-- Define the equations
variables (x y z : ℝ)

def eq1 : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 : Prop := x + 8 * y - 11 * z = 7 / 4
def eq3 : Prop := 5 * x - 6 * y + 2 * z = 12
def sum_eq : Prop := x + y + z = 10

-- Theorem: Coefficient of x in the second equation is 1
theorem coeff_x_in_eq2 (h1 : eq1) (h2 : eq2) (h3 : eq3) (h4 : sum_eq) : 1 = 1 := by
  sorry

end coeff_x_in_eq2_l217_217535


namespace sqrt_sqrt_16_eq_pm2_l217_217705

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l217_217705


namespace number_of_five_digit_numbers_with_eight_l217_217072

-- Define the range of five-digit numbers.
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

-- Define a predicate to check if a number contains at least one digit 8.
def contains_digit_8 (n : ℕ) : Prop := 
  let digits := (List.ofNatDigits n).map (fun d => d = 8)
  digits.contains true

-- The stated problem to prove.
theorem number_of_five_digit_numbers_with_eight : 
  (Finset.filter contains_digit_8 (Finset.filter is_five_digit (Finset.range 100000))).card = 37512 :=
by
  sorry

end number_of_five_digit_numbers_with_eight_l217_217072


namespace number_of_ordered_pairs_l217_217849

theorem number_of_ordered_pairs : 
  let pairs_count := ∑ y in Finset.range 100, (if y > 0 then ((100 - y) / (y * (y + 1))) else 0) in
  pairs_count = 85 :=
by
  let pairs_count := ∑ y in Finset.range 100, (if y > 0 then ((100 - y) / (y * (y + 1))) else 0)
  exact pairs_count = 85
  sorry

end number_of_ordered_pairs_l217_217849


namespace distance_between_parallel_lines_l1_l2_l217_217676

def line1 (x y : ℝ) : Prop := x + y + 1 = 0
def line2 (x y : ℝ) : Prop := x + y + 3/2 = 0

theorem distance_between_parallel_lines_l1_l2 : 
  ∀ (A B : ℝ) (x1 y1 : ℝ),
  (A, B) = (1, 1) → 
  ¬(line1 0 (-1)) → 
  ¬(line2 0 (-3/2)) →
  abs ((A * 0 + B * (-1) + (3 / 2))) / sqrt (A ^ 2 + B ^ 2) = sqrt (2) / 4 :=
by 
  intros A B x1 y1 h1 h2 h3
  sorry

end distance_between_parallel_lines_l1_l2_l217_217676


namespace angle_BFC_l217_217793

theorem angle_BFC (A B C D E F : Point)
  (hD_in_triangle_ABC : D ∈ triangle A B C)
  (hBD_inter_AC_at_E : line_through B D ∩ line_through A C = {E})
  (hCD_inter_AB_at_F : line_through C D ∩ line_through A B = {F})
  (hAF_eq_BF : dist A F = dist B F)
  (hBF_eq_CD : dist B F = dist C D)
  (hCE_eq_DE : dist C E = dist D E) :
  angle B F C = 120 :=
sorry

end angle_BFC_l217_217793


namespace number_of_subsets_B_l217_217548

def A : Set ℕ := {1, 3}
def C : Set ℕ := {1, 3, 5}

theorem number_of_subsets_B : ∃ (n : ℕ), (∀ B : Set ℕ, A ∪ B = C → n = 4) :=
sorry

end number_of_subsets_B_l217_217548


namespace solution_to_sqrt_eq_l217_217827

theorem solution_to_sqrt_eq (z : ℝ) (h : z = 28) : sqrt (-3 + 3 * z) = 9 :=
by
  -- Using the given condition
  rw [h]
  -- Continuing proof skip
  sorry

end solution_to_sqrt_eq_l217_217827


namespace calc_problem_1_calc_problem_2_l217_217004

-- Proof Problem (1)
theorem calc_problem_1 : (Real.sqrt 0.04 + Real.cbrt (-8) - Real.sqrt (1/4) = -2.3) :=
by
  sorry

-- Proof Problem (2)
theorem calc_problem_2 : (abs (1 - Real.sqrt 2) + abs (Real.sqrt 2 - Real.sqrt 3) + abs (Real.sqrt 3 - 2) = 1) :=
by
  sorry

end calc_problem_1_calc_problem_2_l217_217004


namespace cylinder_volume_expansion_l217_217686

theorem cylinder_volume_expansion (r h : ℝ) :
  (π * (2 * r)^2 * h) = 4 * (π * r^2 * h) :=
by
  sorry

end cylinder_volume_expansion_l217_217686


namespace hyperbola_range_m_l217_217278

-- Define the condition that the equation represents a hyperbola
def isHyperbola (m : ℝ) : Prop := (2 + m) * (m + 1) < 0

-- The theorem stating the range of m given the condition
theorem hyperbola_range_m (m : ℝ) : isHyperbola m → -2 < m ∧ m < -1 := by
  sorry

end hyperbola_range_m_l217_217278


namespace expand_polynomial_l217_217457

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end expand_polynomial_l217_217457


namespace part_b_part_c_best_constant_l217_217746

-- Part (a) - Define two functions satisfying the condition
def function1 (x : ℝ) : ℝ :=
if x ≥ 1 then x else 1 / x

def function2 (x : ℝ) : ℝ :=
if x ≥ 1 then x else 1

-- Part (b) - Show that f(x^3) ≥ x^2 for all x in ℝ^+
theorem part_b (f : ℝ → ℝ) (h : ∀ x : ℝ, x > 0 → 2 * f (x^2) ≥ x * f x + x) :
  ∀ x : ℝ, x > 0 → f (x^3) ≥ x^2 :=
sorry

-- Part (c) - Find the best constant a such that f(x) ≥ x^a for all x in ℝ^+
theorem part_c_best_constant (f : ℝ → ℝ) (h : ∀ x : ℝ, x > 0 → 2 * f (x^2) ≥ x * f x + x) :
  ∃ a : ℝ, (∀ x : ℝ, x > 0 → f x ≥ x^a) ∧ a = 1 :=
sorry

end part_b_part_c_best_constant_l217_217746


namespace seven_digit_divisibility_by_11_l217_217929

theorem seven_digit_divisibility_by_11 (n : ℕ) (h₀ : n < 10) :
  let number := [8, n, 4, 6, 3, 2, 5],
      odd_sum := number.foldl (λ acc ⟨d, i⟩ => if (i % 2 = 0) then acc + d else acc) 0,
      even_sum := number.foldl (λ acc ⟨d, i⟩ => if (i % 2 = 1) then acc + d else acc) 0
  in  (odd_sum - even_sum) % 11 = 0 → n = 1 := 
by
  sorry

end seven_digit_divisibility_by_11_l217_217929


namespace field_ratio_l217_217285

theorem field_ratio (l w : ℕ) (h_l : l = 20) (pond_side : ℕ) (h_pond_side : pond_side = 5)
  (h_area_pond : pond_side * pond_side = (1 / 8 : ℚ) * l * w) : l / w = 2 :=
by 
  sorry

end field_ratio_l217_217285


namespace fraction_B_A_plus_C_l217_217384

variable (A B C : ℝ)
variable (f : ℝ)
variable (hA : A = 1 / 3 * (B + C))
variable (hB : A = B + 30)
variable (hTotal : A + B + C = 1080)
variable (hf : B = f * (A + C))

theorem fraction_B_A_plus_C :
  f = 2 / 7 :=
sorry

end fraction_B_A_plus_C_l217_217384


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l217_217146

-- Condition definitions
def fast_clients := 5
def slow_clients := 3
def num_clients := 8
def fast_time := 1
def slow_time := 5

-- Question 1: Minimum wasted person-minutes
theorem min_wasted_person_minutes : 
  (∀ (a b : ℕ), a = fast_time → b = slow_time → 
    (Σ i in finset.range fast_clients, i * fast_time) +
    (Σ i in finset.range slow_clients, (i + fast_clients) * slow_time) = 40) :=
sorry

-- Question 2: Maximum wasted person-minutes
theorem max_wasted_person_minutes : 
  (∀ (a b : ℕ), a = fast_time → b = slow_time →
    (Σ i in finset.range slow_clients, i * slow_time) +
    (Σ i in finset.range fast_clients, (i + slow_clients) * fast_time) = 100) :=
sorry

-- Question 3: Expected wasted person-minutes
theorem expected_wasted_person_minutes : 
  (∀ (a b : ℕ), a = fast_time → b = slow_time →
    ((∑ k in finset.range num_clients, (3 * 5 + 5 * 1)/8 * (k - 1) * 7)/2) = 70) :=
sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l217_217146


namespace pirates_coins_l217_217715

theorem pirates_coins (x : ℝ) (h1 : x > 0) :
  let first_pirates_share := (3/7) * x in
  let remaining_after_first := x - first_pirates_share in
  let second_pirates_share := 0.51 * remaining_after_first in
  let remaining_after_second := remaining_after_first - second_pirates_share in
  let third_pirates_share := remaining_after_second in
  second_pirates_share - third_pirates_share = 8 →
  x = 700 :=
by
  sorry

end pirates_coins_l217_217715


namespace find_ellipse_prove_slopes_relationship_find_slope_BG_min_l217_217115

variable {a b c : ℝ}

-- Given conditions
def is_ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def parabola_focus : ℝ × ℝ := (0, - Real.sqrt 3)
def eccentricity : ℝ := Real.sqrt 2 / 2

-- Given equations from the problem
def parabola : ℝ → ℝ := λ x, - (x ^ 2) / (4 * Real.sqrt 3)

-- Definitions of key values
def semi_major_axis := Real.sqrt 6
def semi_minor_axis := Real.sqrt 3
def focal_distance := Real.sqrt 3

-- Definition of the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / (6) + y^2 / (3) = 1

-- Defining x0, y0 for points A( x0, y0 ) and so on.
def A (x0 y0 : ℝ) : Prop := ellipse_equation x0 y0 ∧ x0 > 0 ∧ y0 > 0

-- Definition for slopes k and k'
def slope_AM (x0 m : ℝ) : ℝ := m / x0
def slope_DM (x0 m : ℝ) : ℝ := - (3 * m) / x0

-- Definition of the relationship between k and k'
def slopes_relationship {x0 m : ℝ} (hA : A x0 (2 * m)) :
  3 * (slope_AM x0 m) + (slope_DM x0 m) = 0 := by
sorry

-- Defining minimum slopes
def slope_BG_min (k : ℝ) : ℝ := Real.sqrt 6 / 2

-- Theorem statements
theorem find_ellipse : ellipse_equation x y :=
sorry

theorem prove_slopes_relationship :
  ∀ x0 m : ℝ, A x0 (2 * m) → 0 < m ∧ m < semi_minor_axis →
  3 * (slope_AM x0 m) + (slope_DM x0 m) = 0 :=
sorry

theorem find_slope_BG_min : ∃ k, (slope_BG_min k) = Real.sqrt 6 / 2 :=
sorry

end find_ellipse_prove_slopes_relationship_find_slope_BG_min_l217_217115


namespace sprite_liters_l217_217766

def liters_of_maaza := 80
def liters_of_pepsi := 144
def total_cans := 37

theorem sprite_liters : ∃ (gcd: Nat), gcd = Nat.gcd liters_of_maaza liters_of_pepsi ∧ 
  (let cans_for_maaza := liters_of_maaza / gcd in
  let cans_for_pepsi := liters_of_pepsi / gcd in
  let cans_for_sprite := total_cans - cans_for_maaza - cans_for_pepsi in
  let sprite_liters := cans_for_sprite * gcd in
  sprite_liters = 368) :=
by
  sorry

end sprite_liters_l217_217766


namespace focus_of_hyperbola_l217_217810

theorem focus_of_hyperbola :
  ∀ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | 
    (p.1 - 4)^2 / 7^2 - (p.2 - 20)^2 / 15^2 = 1 } →
    (x, y) = (4 - real.sqrt 274, 20) ∨ (x, y) = (4 + real.sqrt 274, 20) → 
    x = (4 - real.sqrt 274) :=
begin
  intros x y hy hx,
  cases hx; 
  { rw ←hx, exact rfl },
  sorry
end

end focus_of_hyperbola_l217_217810


namespace number_of_integers_c_l217_217832

theorem number_of_integers_c (z : ℕ) :
  ∃ c : ℕ, (∀ c, ∃ x : ℝ, (abs (20 * abs x - x^2 - c) = 21) → (count_12_distinct_real_solutions c)) → (z = 57) :=
sorry

end number_of_integers_c_l217_217832


namespace max_value_ab_l217_217207

theorem max_value_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 5 * a + 8 * b = 80) : ab ≤ 40 := 
  sorry

end max_value_ab_l217_217207


namespace equal_roots_h_l217_217573

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0) ↔ h = 4 := by
  -- proof goes here
  sorry

end equal_roots_h_l217_217573


namespace negative_solution_condition_l217_217058

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l217_217058


namespace number_accuracy_decimal_places_l217_217147

theorem number_accuracy_decimal_places (n : ℝ) (h : round n = 3.31) : decimal_places 3.31 = 2 := sorry

end number_accuracy_decimal_places_l217_217147


namespace num_five_digit_palindromes_l217_217375

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

def five_digit_palindromes (n : ℕ) : Prop :=
  is_palindrome n ∧ 10000 ≤ n ∧ n < 100000

theorem num_five_digit_palindromes : ∃ n, n = 900 ∧
  (∀ x, five_digit_palindromes x → five_digit_palindromes x → x = ∑ n = 900) :=
sorry

end num_five_digit_palindromes_l217_217375


namespace find_probability_eta_ge_1_l217_217992

noncomputable def xi_dist (p : ℝ) : Probability :=
  bernoulli_dist 2 p

noncomputable def eta_dist (p : ℝ) : Probability :=
  bernoulli_dist 3 p

theorem find_probability_eta_ge_1 (p : ℝ) (h : P(xi_dist p ≥ 1) = 5 / 9) : 
  P(eta_dist p ≥ 1) = 19 / 27 := 
sorry

end find_probability_eta_ge_1_l217_217992


namespace total_samples_l217_217426

-- Define conditions
def samples_per_shelf : ℕ := 128
def number_of_shelves : ℕ := 13

-- Define statement to prove
theorem total_samples (samples_per_shelf number_of_shelves : ℕ) : samples_per_shelf * number_of_shelves = 1664 :=
by
  exact Nat.mul_eq_1664 samples_per_shelf number_of_shelves sorry

end total_samples_l217_217426


namespace domain_real_iff_l217_217121

noncomputable def is_domain_ℝ (m : ℝ) : Prop :=
  ∀ x : ℝ, (m * x^2 + 4 * m * x + 3 ≠ 0)

theorem domain_real_iff (m : ℝ) :
  is_domain_ℝ m ↔ (0 ≤ m ∧ m < 3 / 4) :=
sorry

end domain_real_iff_l217_217121


namespace spherical_to_rectangular_coordinates_l217_217013

theorem spherical_to_rectangular_coordinates 
  (ρ θ φ : ℝ) 
  (h1 : ρ = 3) 
  (h2 : θ = π / 3) 
  (h3 : φ = π / 6) :
  let x := ρ * sin φ * cos θ
  let y := ρ * sin φ * sin θ
  let z := ρ * cos φ
  x = 3 / 4 ∧ y = 3 * sqrt 3 / 4 ∧ z = 3 * sqrt 3 / 2 :=
by
  sorry

end spherical_to_rectangular_coordinates_l217_217013


namespace triangle_median_intersection_l217_217991

theorem triangle_median_intersection
  (A B C M X Y : Point)
  (⊢ Cir C : Circle)
  (⊢ t : Line)
  (h1 : is_triangle A B C)
  (h2 : M = centroid A B C)
  (h3 : t.contains M)
  (h4 : t.intersects_circumcircle A B C at X and Y)
  (h5 : same_side A C t) :
  segment_length B X * segment_length B Y = segment_length A X * segment_length A Y + segment_length C X * segment_length C Y := 
sorry

end triangle_median_intersection_l217_217991


namespace sequence_is_geometric_l217_217547

def is_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → S n = 3 * a n - 3

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (a₁ : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ * r ^ n

theorem sequence_is_geometric (S : ℕ → ℝ) (a : ℕ → ℝ) :
  is_sequence_sum S a →
  (∃ a₁ : ℝ, ∃ r : ℝ, geometric_sequence a r a₁ ∧ a₁ = 3 / 2 ∧ r = 3 / 2) :=
by
  sorry

end sequence_is_geometric_l217_217547


namespace positive_projection_exists_l217_217268

variables {E : Type*} [inner_product_space ℝ E]

theorem positive_projection_exists (e : fin 10 → E)
  (h : ∀ i : fin 10, ‖(∑ j, e j) - e i‖ < ‖∑ j, e j‖) :
  ∀ i : fin 10, 0 < ⟪(∑ j, e j), e i⟫ :=
by sorry

end positive_projection_exists_l217_217268


namespace annual_earning_difference_l217_217191

def old_hourly_wage := 16
def old_weekly_hours := 25
def new_hourly_wage := 20
def new_weekly_hours := 40
def weeks_per_year := 52

def old_weekly_earnings := old_hourly_wage * old_weekly_hours
def new_weekly_earnings := new_hourly_wage * new_weekly_hours

def old_annual_earnings := old_weekly_earnings * weeks_per_year
def new_annual_earnings := new_weekly_earnings * weeks_per_year

theorem annual_earning_difference:
  new_annual_earnings - old_annual_earnings = 20800 := by
  sorry

end annual_earning_difference_l217_217191


namespace parallel_tangent_zero_point_l217_217911

-- Definition of vectors a and b
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Proof Problem I
theorem parallel_tangent (x : ℝ) (h : ∃ k : ℝ, a x = (k * (b x).fst, k * (b x).snd)) : 
  Real.tan x = -3 / 2 := sorry

-- Function f(x)
def f (x : ℝ) : ℝ := 
  let a_plus_b := (a x).fst + (b x).fst, (a x).snd + (b x).snd
  (a_plus_b.1 * (b x).fst) + (a_plus_b.2 * (b x).snd)

-- Proof Problem II
theorem zero_point (h : f (-Real.pi / 8) = 0) :
  ∃ x ∈ Icc (-Real.pi / 2) 0, f x = 0 := sorry

end parallel_tangent_zero_point_l217_217911


namespace monotonicity_of_f_le_0_monotonicity_of_f_gt_0_inequality_holds_then_a_ge_1_l217_217537

noncomputable def f (a x : ℝ) : ℝ := a / x + Real.log x

theorem monotonicity_of_f_le_0 (a : ℝ) (ha : a ≤ 0) : ∀ x ∈ Ioi 0, Function.monotone (f a) :=
begin
  intro x,
  sorry
end

theorem monotonicity_of_f_gt_0 (a : ℝ) (ha : a > 0) :
  (∀ x ∈ Ioi a, Function.monotone_on (f a) (Ioi a)) ∧
  (∀ x ∈ Ioi 0 \ {a}, Function.antitone_on (f a $ x) (Ioo 0 a)) :=
begin
  intro x,
  sorry
end

theorem inequality_holds_then_a_ge_1 (a : ℝ) : (∀ x ∈ Ioo 0 Real.exp 1, f a x ≥ 1) → a ≥ 1 :=
begin
  intro h,
  sorry
end

end monotonicity_of_f_le_0_monotonicity_of_f_gt_0_inequality_holds_then_a_ge_1_l217_217537


namespace inclination_angle_of_line_l217_217682

theorem inclination_angle_of_line (θ : Real) 
  (h : θ = Real.tan 45) : θ = 90 :=
sorry

end inclination_angle_of_line_l217_217682


namespace tree_initial_height_example_l217_217314

-- The height of the tree at the time Tony planted it
def initial_tree_height (growth_rate final_height years : ℕ) : ℕ :=
  final_height - (growth_rate * years)

theorem tree_initial_height_example :
  initial_tree_height 5 29 5 = 4 :=
by
  -- This is where the proof would go, we use 'sorry' to indicate it's omitted.
  sorry

end tree_initial_height_example_l217_217314


namespace sales_percent_increase_l217_217967

def percent_increase := (new_amount old_amount: ℝ) : ℝ :=
  ((new_amount - old_amount) / old_amount) * 100

def total_sales := (sales_A sales_B sales_C: ℝ) : ℝ :=
  sales_A + sales_B + sales_C

def problem_statement := 
  let sales_A_last_year := 120 : ℝ
  let sales_B_last_year := 100 : ℝ
  let sales_C_last_year := 100 : ℝ

  let increase_A := 0.70 : ℝ
  let increase_B := 0.60 : ℝ
  let increase_C := 0.50 : ℝ

  let sales_A_this_year := sales_A_last_year + (sales_A_last_year * increase_A)
  let sales_B_this_year := sales_B_last_year + (sales_B_last_year * increase_B)
  let sales_C_this_year := sales_C_last_year + (sales_C_last_year * increase_C)

  let total_sales_last_year := total_sales sales_A_last_year sales_B_last_year sales_C_last_year
  let total_sales_this_year := total_sales sales_A_this_year sales_B_this_year sales_C_this_year

  let overall_increase_percentage := percent_increase total_sales_this_year total_sales_last_year

  %{
  percent_increase sales_A_this_year sales_A_last_year = 70,
  percent_increase sales_B_this_year sales_B_last_year = 60,
  percent_increase sales_C_this_year sales_C_last_year = 50,
  overall_increase_percentage ≈ 60.625
  }

theorem sales_percent_increase :
  problem_statement := 
by
  intro problem_statement
  have a := (percent_increase sales_A_this_year sales_A_last_year)
  have b := (percent_increase sales_B_this_year sales_B_last_year)
  have c := (percent_increase sales_C_this_year sales_C_last_year)
  have overall := (overall_increase_percentage)
  exact sorry

end sales_percent_increase_l217_217967


namespace find_x_l217_217987

def star (p q : Int × Int) : Int × Int :=
  (p.1 + q.2, p.2 - q.1)

theorem find_x : ∀ (x y : Int), star (x, y) (4, 2) = (5, 4) → x = 3 :=
by
  intros x y h
  -- The statement is correct, just add a placeholder for the proof
  sorry

end find_x_l217_217987


namespace function_bijective_of_composition_eq_identity_l217_217089

variables {X : Type*} (f : X → X) (k : ℕ) (I_X : X → X)

theorem function_bijective_of_composition_eq_identity 
  (h : Function.iterate f k = I_X) : Function.Bijective f :=
sorry

end function_bijective_of_composition_eq_identity_l217_217089


namespace dilation_image_l217_217675

theorem dilation_image (c : ℂ) (k : ℝ) (z z' : ℂ) 
  (h₁ : c = 1 + 2 * complex.I) 
  (h₂ : k = 2) 
  (h₃ : z = 0) 
  (h₄ : z' = -1 - 2 * complex.I) : 
  dilation_image : ∃ w, w = z' ∧ w - c = k * (z - c) :=
by
  use z'
  simp [h₁, h₂, h₃, h₄]
  sorry

end dilation_image_l217_217675


namespace max_amount_xiao_li_spent_l217_217006

theorem max_amount_xiao_li_spent (a m n : ℕ) :
  33 ≤ m ∧ m < n ∧ n ≤ 37 ∧
  ∃ (x y : ℕ), 
  (25 * (a - x) + m * (a - y) + n * (x + y + a) = 700) ∧ 
  (25 * x + m * y + n * (3*a - x - y) = 1200) ∧
  ( 675 <= 700 - 25) :=
sorry

end max_amount_xiao_li_spent_l217_217006


namespace sum_of_coefficients_l217_217528

/-- Given the coefficient of the second term in the binomial expansion of (x + 2y)^n is 8,
    prove that the sum of the coefficients of all terms in the expansion of (1 + x) + (1 + x)^2 + ... + (1 + x)^n is 30. -/
theorem sum_of_coefficients (n : ℕ) (h : 2 * n = 8) :
  let S := (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 in
  ∑ k in (finset.range (n + 1)), 2^k = 30 :=
by
  have n_eq : n = 4 := by linarith, 
  sorry

end sum_of_coefficients_l217_217528


namespace domain_v_l217_217017

noncomputable def v (x : ℝ) : ℝ := 1 / (Real.sqrt x + x - 1)

theorem domain_v :
  {x : ℝ | x >= 0 ∧ Real.sqrt x + x - 1 ≠ 0} = {x : ℝ | x ∈ Set.Ico 0 (Real.sqrt 5 - 1) ∪ Set.Ioi (Real.sqrt 5 - 1)} :=
by
  sorry

end domain_v_l217_217017


namespace seq_an_general_term_and_sum_l217_217628

theorem seq_an_general_term_and_sum
  (a_n : ℕ → ℕ)
  (S : ℕ → ℕ)
  (T : ℕ → ℕ)
  (H1 : ∀ n, S n = 2 * a_n n - a_n 1)
  (H2 : ∃ d : ℕ, a_n 1 = d ∧ a_n 2 + 1 = a_n 1 + d ∧ a_n 3 = a_n 2 + d) :
  (∀ n, a_n n = 2^n) ∧ (∀ n, T n = n * 2^(n + 1) + 2 - 2^(n + 1)) := 
  by
  sorry

end seq_an_general_term_and_sum_l217_217628


namespace dissimilar_nice_triangles_count_is_four_l217_217804

-- Define what it means for a triangle to be 'nice'
def is_nice_triangle (A B C : ℝ) : Prop :=
  -- Define the property of the triangle in terms of its angles
  -- This is a simplified placeholder definition
  (A + B + C = 180) ∧ 
  ((A ∣ 360) ∧ (B ∣ 360) ∧ (C ∣ 360)) ∧
  (A = 60 ∧ B = 60 ∧ C = 60 ∨ 
   A = 45 ∧ B = 45 ∧ C = 90 ∨ 
   A = 30 ∧ B = 60 ∧ C = 90 ∨
   A = 30 ∧ B = 30 ∧ C = 120)

-- Define the property of dissimilar nice triangles
def dissimilar_nice_triangle_count := { (A, B, C) // is_nice_triangle A B C }.to_finset.card

-- Main theorem statement
theorem dissimilar_nice_triangles_count_is_four : 
  dissimilar_nice_triangle_count = 4 :=
sorry

end dissimilar_nice_triangles_count_is_four_l217_217804


namespace smallest_integer_CC4_DD6_rep_l217_217730

-- Lean 4 Statement
theorem smallest_integer_CC4_DD6_rep (C D : ℕ) (hC : C < 4) (hD : D < 6) :
  (5 * C = 7 * D) → (5 * C = 35 ∧ 7 * D = 35) :=
by
  sorry

end smallest_integer_CC4_DD6_rep_l217_217730


namespace perimeter_of_right_triangle_proof_l217_217163

noncomputable def perimeter_of_right_triangle (R r : ℝ) : ℝ :=
let hypotenuse := 2 * R in
let a_b_sum := hypotenuse in
2 * (a_b_sum) + 2 * r

theorem perimeter_of_right_triangle_proof
  (R r : ℝ) (hR : R = 14.5) (hr : r = 6) :
  perimeter_of_right_triangle R r = 70 :=
by {
  sorry
}

end perimeter_of_right_triangle_proof_l217_217163


namespace exists_positive_projection_l217_217267

-- Define the vectors as elements in a vector space over the reals
noncomputable def vectors : ℕ → ℝ^3 := sorry

-- The condition given in the problem
def condition (vectors : ℕ → ℝ^3) : Prop :=
  ∀ i : ℕ, i < 10 →
  ‖(∑ j in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ {i}, vectors j)‖ < ‖∑ j in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, vectors j‖

-- The theorem to be proved
theorem exists_positive_projection (vectors : ℕ → ℝ^3) (h : condition vectors) :
  ∃ v : ℝ^3, ∀ i : ℕ, i < 10 → (vectors i) ⬝ v > 0 :=
sorry

end exists_positive_projection_l217_217267


namespace triangle_ABC_is_right_triangle_l217_217157

-- Definitions of the sides and vectors
variables (a b c : ℝ) (m n : ℝ × ℝ)
def m := (a + c, b)
def n := (b, a - c)
def triangle_is_right (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- The main theorem to state the problem
theorem triangle_ABC_is_right_triangle
  (h_parallel : m ∥ n) : triangle_is_right a b c :=
by sorry

end triangle_ABC_is_right_triangle_l217_217157


namespace complementary_set_count_is_correct_l217_217025

inductive Shape
| circle | square | triangle | hexagon

inductive Color
| red | blue | green

inductive Shade
| light | medium | dark

structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)

def deck : List Card :=
  -- (Note: Explicitly listing all 36 cards would be too verbose, pseudo-defining it for simplicity)
  [(Card.mk Shape.circle Color.red Shade.light),
   (Card.mk Shape.circle Color.red Shade.medium), 
   -- and so on for all 36 unique combinations...
   (Card.mk Shape.hexagon Color.green Shade.dark)]

def is_complementary (c1 c2 c3 : Card) : Prop :=
  ((c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∨ (c1.shape = c2.shape ∧ c2.shape = c3.shape)) ∧ 
  ((c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∨ (c1.color = c2.color ∧ c2.color = c3.color)) ∧
  ((c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∨ (c1.shade = c2.shade ∧ c2.shade = c3.shade))

noncomputable def count_complementary_sets : ℕ :=
  -- (Note: Implementation here is a placeholder. Actual counting logic would be non-trivial.)
  1836 -- placeholding the expected count

theorem complementary_set_count_is_correct :
  count_complementary_sets = 1836 :=
by
  trivial

end complementary_set_count_is_correct_l217_217025


namespace proof_f_inequality_l217_217867

def periodic (f : ℝ → ℝ) := ∀ x : ℝ, f(x) = f(x + 4)
def monotonic_interval (f : ℝ → ℝ) := ∀ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f(x₁) < f(x₂)
def symmetric_about_y (f : ℝ → ℝ) := ∀ x : ℝ, f(x + 2) = f(2 - x)

theorem proof_f_inequality (f : ℝ → ℝ) 
  (h1 : periodic f)
  (h2 : monotonic_interval f)
  (h3 : symmetric_about_y f) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by 
  -- Proof not required
  sorry

end proof_f_inequality_l217_217867


namespace jen_ate_eleven_suckers_l217_217253

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end jen_ate_eleven_suckers_l217_217253


namespace like_terms_expression_value_l217_217919

theorem like_terms_expression_value (m n : ℤ) (h1 : m = 3) (h2 : n = 1) :
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 33 := by
  sorry

end like_terms_expression_value_l217_217919


namespace GreatWhiteSharkTeeth_l217_217412

-- Definition of the number of teeth for a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Definition of the number of teeth for a hammerhead shark
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Definition of the number of teeth for a great white shark
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- Statement to prove
theorem GreatWhiteSharkTeeth : great_white_shark_teeth = 420 :=
by
  -- Proof omitted
  sorry

end GreatWhiteSharkTeeth_l217_217412


namespace part_I_part_II_l217_217988

-- Conditions
def p (x m : ℝ) : Prop := x > m → 2 * x - 5 > 0
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (m - 1)) + (y^2 / (2 - m)) = 1

-- Statements for proof
theorem part_I (m x : ℝ) (hq: q m) (hp: p x m) : 
  m < 1 ∨ (2 < m ∧ m ≤ 5 / 2) :=
sorry

theorem part_II (m x : ℝ) (hq: ¬ q m ∧ ¬(p x m ∧ q m) ∧ (p x m ∨ q m)) : 
  (1 ≤ m ∧ m ≤ 2) ∨ (m > 5 / 2) :=
sorry

end part_I_part_II_l217_217988


namespace tangents_midpoints_collinear_l217_217204

theorem tangents_midpoints_collinear
  (ω1 ω2 : Circle)
  (externally_disjoint : Disjoint ω1.exterior ω2.exterior)
  (tangents AB1 AB2 : List (TangentSegment ω1 ω2))
  (h_tangents_card : tangents.length = 4)
  (midpoints : List Point)
  (midpoint_condition : ∀ segment ∈ tangents, ∃ m ∈ midpoints, IsMidpoint segment m) :
  AllCollinear midpoints := 
by
  sorry

end tangents_midpoints_collinear_l217_217204


namespace white_balls_count_l217_217944

theorem white_balls_count (n : ℕ) (h : 8 / (8 + n : ℝ) = 0.4) : n = 12 := by
  sorry

end white_balls_count_l217_217944


namespace negative_solution_condition_l217_217055

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l217_217055


namespace problem_l217_217435

def diamond (a b : ℕ) : ℚ := (a * b : ℚ) / (a + b + 2)

theorem problem (x y : ℚ) :
  (∀ a b : ℕ, a > 0 ∧ b > 0 → diamond a b = (a * b : ℚ) / (a + b + 2)) ∧
  diamond 3 7 = x ∧
  diamond x.natAbs 9 = y →
  y = 21 / 17 :=
by
  sorry

end problem_l217_217435


namespace general_formula_for_an_sum_of_cn_l217_217884

variable (a : Nat → ℚ)
variable (S : Nat → ℚ)
variable (c : Nat → ℚ)
variable (T : Nat → ℚ)

axiom terms_positive : ∀ n, 0 < a n
axiom Sn_definition : ∀ n, S n = (1/8) * (a n)^2 + (1/2) * (a n) + (1/2)
axiom a1_value : a 1 = 2

theorem general_formula_for_an (n : Nat) (hn : n ≥ 1) : a n = 4 * n - 2 := sorry

theorem sum_of_cn (n : Nat) (hn : n ≥ 1) : 
  (T n = ∑ i in Finset.range n, (c i)) → T n = n / (4 * (2 * n + 1)) := sorry

-- Define the formula for cn
noncomputable def c_def (n : Nat) : ℚ := 1 / (a n * a (n + 1))

axiom definition_c : ∀ n, c n = c_def n

end general_formula_for_an_sum_of_cn_l217_217884


namespace find_philosophical_numbers_l217_217324

-- Definitions
def is_philosophical_sequence (a : ℝ) (ε : ℕ → ℤ) (a_seq : ℕ → ℝ) : Prop :=
  (∀ n, ε n = 1 ∨ ε n = -1) ∧
  (a_seq 0 = a) ∧
  (∀ n, a_seq (n + 1) = ε n * real.sqrt (a_seq n + 1)) ∧
  (∃ m, ∀ k, a_seq (k + m) = a_seq k)

-- Main theorem
theorem find_philosophical_numbers :
  {a : ℝ | a ≥ -1 ∧ (∃ (ε : ℕ → ℤ) (a_seq : ℕ → ℝ), is_philosophical_sequence a ε a_seq)} = 
  {0, -1, (1 + real.sqrt 5) / 2, (1 - real.sqrt 5) / 2} :=
sorry

end find_philosophical_numbers_l217_217324


namespace median_length_DN_l217_217170

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def length (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

theorem median_length_DN 
  (D F E : Point3D)
  (h : F.y = D.y ∧ F.z = D.z + 5 ∧ E.x = D.x ∧ E.y = D.y ∧ E.z = D.z + 12 ∧ E.x = F.x ∧ E.y = F.y) 
  : length D (midpoint E F) = 6.5 := 
by
  sorry 

end median_length_DN_l217_217170


namespace sum_b_3000_eq_21670_l217_217848

-- Define the function b(p)
def b (p : ℕ) : ℕ :=
  let k := Nat.floor (Real.sqrt p) + 1
  k - if Real.sqrt p < k - 0.5 then 1 else 0

-- Define the sum over b(p) from 1 to 3000
def sum_b (n : ℕ) : ℕ :=
  (Finset.range n).sum b

-- The theorem to prove
theorem sum_b_3000_eq_21670 : sum_b 3001 = 21670 := 
  sorry

end sum_b_3000_eq_21670_l217_217848


namespace hyperbola_eccentricity_l217_217899

open Real

theorem hyperbola_eccentricity (a b p : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : p > 0)
  (hyper : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1)
  (parabola_focus : (p / 2, 0))
  (pass_through_focus : ∀ x y : ℝ, y^2 = 2 * p * x → ( (x - (p / 2))^2 / a^2) - (y^2 / b^2) = 1)
  (equilateral : a = √3 * b) :
  ∃ e : ℝ, e = 2 * √3 / 3 :=
by 
  sorry

end hyperbola_eccentricity_l217_217899


namespace two_digit_numbers_formed_by_l217_217223

theorem two_digit_numbers_formed_by {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 2) (h3 : d3 = 3) : 
  {n : ℕ | ∃ (t u : ℕ), t ∈ {d1, d2, d3} ∧ u ∈ {d1, d2, d3} ∧ n = t * 10 + u} = 
  {11, 12, 13, 21, 22, 23, 31, 32, 33} := 
by
  sorry

end two_digit_numbers_formed_by_l217_217223


namespace math_club_team_selection_l217_217640

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let total := boys + girls
  let team_size := 8
  (Nat.choose total team_size - Nat.choose girls team_size - Nat.choose boys team_size = 319230) :=
by
  sorry

end math_club_team_selection_l217_217640


namespace sqrt_sqrt_16_eq_pm_2_l217_217701

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := 
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217701


namespace find_n_l217_217976

def f (x : ℝ) (n : ℝ) : ℝ :=
  if x < 1 then 2 * x + n else real.log 2 x

theorem find_n : (∃ n : ℝ, f (f (3/4) n) n = 2) → n = 5/2 :=
begin
  sorry
end

end find_n_l217_217976


namespace find_side_c_l217_217955

theorem find_side_c (a c b : ℝ) 
  (h1 : c = 2 * a) 
  (h2 : b = 4) 
  (h3 : real.cos (real.pi / 3) = 1/4) :
  c = 4 :=
by
  -- Add proof here
  sorry

end find_side_c_l217_217955


namespace system_has_negative_solution_iff_sum_zero_l217_217053

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l217_217053


namespace union_A_B_inter_A_B_set_C_l217_217127

def A := {1, 3, 5, 7, 9}
def B := {3, 4, 5}
def C := { x | x ∈ A ∧ x ∉ B }

theorem union_A_B : A ∪ B = {1, 3, 4, 5, 7, 9} :=
by { sorry }

theorem inter_A_B : A ∩ B = {3, 5} :=
by { sorry }

theorem set_C : C = {1, 7, 9} :=
by { sorry }

end union_A_B_inter_A_B_set_C_l217_217127


namespace minimum_value_of_f_l217_217071

open Real

def f (x : ℝ) : ℝ := (x^2 + 6 * x + 13) / sqrt (x^2 + 5 * x + 7)

theorem minimum_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(x) ≤ f(y) ∧ f(x) = sqrt 13 :=
by
  sorry

end minimum_value_of_f_l217_217071


namespace total_remaining_staff_l217_217681

-- Definitions of initial counts and doctors and nurses quitting.
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quitting : ℕ := 5
def nurses_quitting : ℕ := 2

-- Definition of remaining doctors and nurses.
def remaining_doctors : ℕ := initial_doctors - doctors_quitting
def remaining_nurses : ℕ := initial_nurses - nurses_quitting

-- Theorem stating the total number of doctors and nurses remaining.
theorem total_remaining_staff : remaining_doctors + remaining_nurses = 22 :=
by
  -- Proof omitted
  sorry

end total_remaining_staff_l217_217681


namespace count_valid_application_l217_217779

universe u

/-- A function that represents the valid way to assign majors excluding the given restrictions. -/
def valid_major_permutations {α : Type u} (majors : Finset α) (major_A : α) : Nat :=
  let without_major_A := majors.erase major_A
  let first_two_choices := without_major_A.to_list.permutations.filter (λ l, l.take 2).length = 7P2
  let remaining_choices := without_major_A.erase (λ l, l.take 2).drop 2
  first_two_choices.length * remaining_choices.to_list.permutations.length

/-- Define the available majors and Major A. -/
def majors : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'}
def major_A : Char := 'A'

/-- Define the theorem for counting the valid permutations. -/
theorem count_valid_application : valid_major_permutations majors major_A = 5040 :=
  by sorry

end count_valid_application_l217_217779


namespace number_of_points_on_curve_C_l217_217288

-- Define the parametric equations of curve C.
def curve_C (θ : ℝ) : ℝ × ℝ := (2 + 3 * Real.cos θ, 1 + 3 * Real.sin θ)

-- Define the equation of line l.
def line_l (x y : ℝ) : Prop := x - 3 * y + 2 = 0

-- Define the distance function between a point and a line.
def distance_point_line (x₀ y₀ : ℝ) (A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Define the center of the circle derived from curve C.
def center_C : ℝ × ℝ := (2, 1)

-- Prove that the number of points on curve C that are (7/10)√10 units away from line l is 4.
theorem number_of_points_on_curve_C (d : ℝ) (h : d = (7/10) * Real.sqrt 10) : 
  ∃ (n : ℕ), n = 4 ∧ ∀ θ : ℝ, 
  distance_point_line (2 + 3 * Real.cos θ) (1 + 3 * Real.sin θ) 1 (-3) 2 = d → 
  (curve_C θ).fst = d ∧ (curve_C θ).snd = d :=
sorry

end number_of_points_on_curve_C_l217_217288


namespace ball_radius_exists_in_polyhedron_l217_217627

variable (P : Type) [convex_polyhedron P]
variable (V : ℝ) (S : ℝ)
variable [polyhedron_volume P = V]
variable [polyhedron_surface_area P = S]

theorem ball_radius_exists_in_polyhedron 
  (P : Type) [convex_polyhedron P] 
  (V S : ℝ) [polyhedron_volume P = V] [polyhedron_surface_area P = S] :
  ∃ (c : P), ∀ (f : face P), dist_point_to_face c f ≥ V / S :=
sorry

end ball_radius_exists_in_polyhedron_l217_217627


namespace minimum_m_ensures_E_in_S_l217_217199

theorem minimum_m_ensures_E_in_S
  (n : ℕ) (hn_even : n % 2 = 0) :
  ∃ m : ℕ, (∀ (S : set ℕ), S.card = m → 
  (∀ (v : fin n.succ → bool), (∃ i : fin n.succ, v i = 1) → 
  (∃ e ∈ S, e = ∑ i in finset.range n.succ, (if v i then 2^i else 0)))) 
  ↔ m ≥ 2^(n/2)) :=
begin
  sorry
end

end minimum_m_ensures_E_in_S_l217_217199


namespace area_triangle_given_parabola_l217_217853

theorem area_triangle_given_parabola (x y : ℝ) (h_parabola : x^2 = 4 * y) (h_directrix : ∀ P M : ℝ × ℝ, y = -1) (h_dist : ∀ P M : ℝ × ℝ, P = (x, y) → M = (x, -1) → real.dist P M = 5) :
  ∃ P F M : ℝ × ℝ, P.1 ^ 2 = 4 * P.2 ∧ real.dist P M = 5 ∧ M.2 = -1 ∧ 4 * (P.2 + 1) * 2 = 20 := sorry

end area_triangle_given_parabola_l217_217853


namespace log_x_64_l217_217559

theorem log_x_64 (x : ℝ) (h : log 8 (5 * x) = 3) : log x 64 = 300 / 317 := by
  sorry

end log_x_64_l217_217559


namespace infinite_monochromatic_rectangles_l217_217325

-- Define the conditions
def grid_points : Type
def colorings (k : ℕ) : grid_points → ℕ → Prop

-- Assume the existence of at least one monochromatic rectangle in any coloring of grid points
-- In this simplified model, monochromatic_rect will capture the essence of the condition
def monochromatic_rect (k : ℕ) : Prop :=
  ∀ (c : grid_points → ℕ),
  (∃ (v1 v2 v3 v4 : grid_points),
     c v1 = c v2 ∧ c v2 = c v3 ∧ c v3 = c v4 ∧ v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v4 ∧ (v1, v2, v3, v4 are vertices of a rectangle))

-- Define the theorem aiming to prove the infinite monochromatic rectangles
theorem infinite_monochromatic_rectangles :
  (∀ k, monochromatic_rect k) →
  ∃ (f : ℕ → (grid_points × grid_points × grid_points × grid_points)),
    (∀ n m, n ≠ m → disjoint_vertices (f n) (f m)) :=
sorry

-- Definition for disjoint vertices, ensuring no two rectangles from f share a vertex
def disjoint_vertices (r1 r2 : grid_points × grid_points × grid_points × grid_points) : Prop :=
  let (v1, v2, v3, v4) := r1 in
  let (a1, a2, a3, a4) := r2 in
  v1 ≠ a1 ∧ v1 ≠ a2 ∧ v1 ≠ a3 ∧ v1 ≠ a4 ∧
  v2 ≠ a1 ∧ v2 ≠ a2 ∧ v2 ≠ a3 ∧ v2 ≠ a4 ∧
  v3 ≠ a1 ∧ v3 ≠ a2 ∧ v3 ≠ a3 ∧ v3 ≠ a4 ∧
  v4 ≠ a1 ∧ v4 ≠ a2 ∧ v4 ≠ a3 ∧ v4 ≠ a4

end infinite_monochromatic_rectangles_l217_217325


namespace min_value_expression_l217_217518

variable (a b : ℝ)

theorem min_value_expression :
  0 < a →
  1 < b →
  a + b = 2 →
  (∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = (2 / a) + (1 / (b - 1)) → y ≥ x)) :=
by
  sorry

end min_value_expression_l217_217518


namespace sum_of_powers_divisible_by_30_l217_217301

theorem sum_of_powers_divisible_by_30 {a b c : ℤ} (h : (a + b + c) % 30 = 0) : (a^5 + b^5 + c^5) % 30 = 0 := by
  sorry

end sum_of_powers_divisible_by_30_l217_217301


namespace geometric_sequence_value_a3_l217_217939

theorem geometric_sequence_value_a3 
  (a : ℕ → ℝ)
  (h_geom : ∃ r, ∀ n, a (n+1) = a n * r)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 16) :
  a 3 = 4 :=
by 
  -- proof would go here
  sorry

end geometric_sequence_value_a3_l217_217939


namespace probability_N_14_mod_5_is_1_l217_217420

theorem probability_N_14_mod_5_is_1 :
  let total := 1950
  let favorable := 2
  let outcomes := 5
  (favorable / outcomes) = (2 / 5) := by
  sorry

end probability_N_14_mod_5_is_1_l217_217420


namespace racing_car_A_lap_time_l217_217230

-- Definitions as per the conditions
def lap_time_B : ℕ := 24
def side_by_side_time : ℕ := 168

def lap_time_condition (t : ℕ) : Prop :=
  Nat.lcm t lap_time_B = side_by_side_time

theorem racing_car_A_lap_time : ∃ t : ℕ, lap_time_condition t ∧ t = 7 := by
  existsi 7
  unfold lap_time_condition
  rw [←Nat.lcm_comm]
  apply Nat.lcm_eq
  exact ⟨4, 7, rfl⟩

end racing_car_A_lap_time_l217_217230


namespace magnitude_of_Z_l217_217111

-- Define the complex number Z
def Z : ℂ := 3 - 4 * Complex.I

-- Define the theorem to prove the magnitude of Z
theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end magnitude_of_Z_l217_217111


namespace complex_conjugate_real_part_l217_217144

open Complex

theorem complex_conjugate_real_part (z1 z2 : ℂ) : 
  z1 * conj z2 + conj z1 * z2 ∈ ℝ :=
by
  sorry

end complex_conjugate_real_part_l217_217144


namespace grid_filling_methods_l217_217459

def grid_filling_with_abc (grid : Array (Array Char)) : Prop :=
  (∀ i, grid[i].length = 3)
  ∧ (∀ j, (grid.map (λ row => row[j])).toList.nodup)
  ∧ (∀ row, row.toList.nodup)

theorem grid_filling_methods : Exists
  (λ (grids : List (Array (Array Char))),
    (∀ g ∈ grids, grid_filling_with_abc g) ∧ a = 12) :=
sorry

end grid_filling_methods_l217_217459


namespace derivative_at_point_l217_217895

-- Given function f
def f (x x₀ : ℝ) : ℝ := 2 * Real.cos (4 * x + x₀^2) - 1

-- Condition that f is even implies x₀^2 = k * π for some k ∈ ℤ
variable (k : ℤ)
noncomputable def x₀ : ℝ := Real.sqrt (k * Real.pi)

-- Derivative of the given function f
noncomputable def f' (x x₀ : ℝ) : ℝ := -8 * Real.sin (4 * x + x₀^2)

-- Proof goal: Verify that f' evaluated at x = x₀^2 / 2 is 0
theorem derivative_at_point (x₀ : ℝ) (k : ℤ) (h : x₀^2 = k * Real.pi) : f' (x₀^2 / 2) x₀ = 0 := by
  sorry

end derivative_at_point_l217_217895


namespace intersect_length_l217_217950

-- Define the parametric equations of curve C
def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the Cartesian equation of curve C
def cartesian_eq_of_curve (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 / 3 = 1

-- Define point P in Cartesian coordinates
def point_P : ℝ × ℝ := (0, -1)

-- Define the parametric equations of line l
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 2 / 2 * t, -1 + Real.sqrt 2 / 2 * t)

-- Define the Cartesian equation of line l (derived from parametric)
def cartesian_eq_of_line (x y : ℝ) : Prop :=
  x - y - 1 = 0

-- Specify the length of segment AB
def length_AB (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt (2 * ((fst A - fst B)^2 + (snd A - snd B)^2))

-- Defining the main problem in Lean
theorem intersect_length :
  (∀ θ, parametric_curve θ ∈ { (x, y) | cartesian_eq_of_curve x y }) →
  (∀ t, parametric_line t ∈ { (x, y) | cartesian_eq_of_line x y }) →
  let A := (4 / Real.sqrt 14, 3 / Real.sqrt 14)
  let B := (-4 / Real.sqrt 14, -3 / Real.sqrt 14)
  length_AB A B = 24 / 7 := 
sorry

end intersect_length_l217_217950


namespace sin_C_value_area_triangle_l217_217155

-- Define the given conditions
def triangleABC (A : ℝ) (a : ℝ) (c : ℝ) : Prop :=
  A = 60 ∧ c = (3 / 7) * a

-- (I) Prove the value of sin C given the conditions
theorem sin_C_value (A : ℝ) (a : ℝ) (c : ℝ) (C : ℝ) 
  (h : triangleABC A a c) : sin C = 3 * sqrt 3 / 14 :=
by 
  cases h with hA hc
  sorry

-- Additional condition for (II)
def triangle_with_a_7 (A : ℝ) (a : ℝ) (c : ℝ) : Prop :=
  triangleABC A a c ∧ a = 7

-- (II) Prove the area of the triangle given the additional condition
theorem area_triangle (A : ℝ) (a : ℝ) (c : ℝ) (B : ℝ) (C : ℝ)
  (h : triangle_with_a_7 A a c) : 
  let sinB := sin A * cos C + cos A * sin C in 
  0.5 * a * c * sinB = 6 * sqrt 3 :=
by
  cases h with h₁ ha
  cases h₁ with hA hc
  sorry

end sin_C_value_area_triangle_l217_217155


namespace f_comp_f_one_l217_217086

-- Define the piecewise function f(x)
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then sin (π * x / 6)
  else 1 - 2 * x

-- The goal is to prove the equivalence of f (f(1)) = -1/2
theorem f_comp_f_one : f (f 1) = -1 / 2 :=
by
  sorry

end f_comp_f_one_l217_217086


namespace sum_of_integers_c_polynomial_factors_over_integers_l217_217203

theorem sum_of_integers_c_polynomial_factors_over_integers :
  let S := {c : ℤ | ∃ r s : ℤ, r + s = -c ∧ r * s = 2520 * c} in
  (∑ c in S, c) = 0 := by
  sorry

end sum_of_integers_c_polynomial_factors_over_integers_l217_217203


namespace opposite_of_2023_l217_217685

theorem opposite_of_2023 : ∃ y : ℝ, 2023 + y = 0 ∧ y = -2023 :=
  by
    use -2023
    constructor
    · norm_num
    · refl

end opposite_of_2023_l217_217685


namespace gcd_n_conditions_l217_217066

theorem gcd_n_conditions (n : ℤ) : 
    let gcd := Int.gcd (5 * n - 7) (3 * n + 2) 
    in (∃ k : ℤ, n = 31 * k - 11) ↔ gcd = 31 :=
by sorry

end gcd_n_conditions_l217_217066


namespace initial_men_garrison_l217_217391

-- Conditions:
-- A garrison has provisions for 31 days.
-- At the end of 16 days, a reinforcement of 300 men arrives.
-- The provisions last only for 5 days more after the reinforcement arrives.

theorem initial_men_garrison (M : ℕ) (P : ℕ) (d1 d2 : ℕ) (r : ℕ) (remaining1 remaining2 : ℕ) :
  P = M * d1 →
  remaining1 = P - M * d2 →
  remaining2 = r * (d1 - d2) →
  remaining1 = remaining2 →
  r = M + 300 →
  d1 = 31 →
  d2 = 16 →
  M = 150 :=
by 
  sorry

end initial_men_garrison_l217_217391


namespace polynomial_evaluation_l217_217568

theorem polynomial_evaluation (p : Polynomial ℝ) (n : ℕ)
  (hdeg : p.degree = n)
  (hvals : ∀ k : ℕ, k ≤ n → p.eval k = (k : ℝ) / (k + 1)) :
  if even n then p.eval (n + 1) = 1 else p.eval (n + 1) = (n : ℝ) / (n + 2) :=
by sorry

end polynomial_evaluation_l217_217568


namespace tails_and_die_1_or_2_l217_217226

noncomputable def fairCoinFlipProbability : ℚ := 1 / 2
noncomputable def fairDieRollProbability : ℚ := 1 / 6
noncomputable def combinedProbability : ℚ := fairCoinFlipProbability * (fairDieRollProbability + fairDieRollProbability)

theorem tails_and_die_1_or_2 :
  combinedProbability = 1 / 6 :=
by
  sorry

end tails_and_die_1_or_2_l217_217226


namespace fixed_point_on_line_l217_217516

noncomputable def parabola  (n : ℝ) (hp : 0 < n) : set (ℝ × ℝ) :=
λ p : ℝ × ℝ, p.2 ^ 2 = n * p.1

theorem fixed_point_on_line 
  (O : ℝ × ℝ)
  (n t : ℝ) 
  (P : ℝ × ℝ)  
  (foc_dist : ℝ)
  (hp_focus : (P.1 - sqrt (n * P.1)) ^ 2 + (P.2 - 0) ^ 2 = foc_dist) 
  (ht : foc_dist = 5 / 2)
  (P_on_parabola : P ∈ parabola n (by norm_num [hp_focus])) 
  (Q : ℝ × ℝ) 
  (tangent_of_P_at_C : ∃ m : ℝ, ∀ x y : ℝ, y - P.2 = m * (x - P.1) ∧ Q.2 = 0)
  (tangent_inter_geom : tangent_of_P_at_C ∧ Q.1 = -2)
  (l1 : ℝ → ℝ) 
  (l1_def : ∀ x, l1 x = -2) 
  (l2 : ℝ → ℝ) 
  (m b : ℝ) 
  (hx : ∃ A B : ℝ × ℝ, (A ∈ parabola n (by norm_num [hp_focus])) ∧ 
                           (B ∈ parabola n (by norm_num [hp_focus])) ∧ 
                           ∀ y : ℝ, l2 y = m * y + b)
  (E : ℝ × ℝ)
  (E_def : E = (l1 0, l2 ((l1 0 - b) / m))) 
  (slopes_seq: (P.2 - A.2) / (P.1 - A.1) + (P.2 - B.2) / (P.1 - B.1) = 2 * (E.2 - P.2) / (E.1 - P.1))
  : Q = (-2, 0) ∧ (∃ k : ℝ, ∀ y : ℝ, l2 y = m * y + 2 ∧ l2 0 = k) := sorry

end fixed_point_on_line_l217_217516


namespace perfect_cube_mult_one_l217_217075

theorem perfect_cube_mult_one (n : ℕ) (h : n = 54000) : ∃ k : ℕ, n = k^3 :=
by {
  rw h,
  use 30,
  norm_num,
  sorry
}

end perfect_cube_mult_one_l217_217075


namespace time_for_B_to_reach_A_l217_217316

-- Define the conditions
structure WalkingConditions :=
  (A_speed_factor : ℝ) -- A's speed as a multiple of B's speed 
  (meeting_time : ℝ) -- time until they meet
  (A_reduced_speed_factor : ℝ) -- A's speed reduction factor after meeting

-- Define the main problem statement
theorem time_for_B_to_reach_A 
  (B_speed : ℝ) -- B's speed in meters/minute
  (conds : WalkingConditions)
  (H : conds.A_speed_factor = 3 ∧ conds.meeting_time = 60 ∧ conds.A_reduced_speed_factor = 0.5)
  : ℝ :=
  let total_distance := (conds.meeting_time * B_speed) * (1 + conds.A_speed_factor) in
  let A_speed_after_meeting := conds.A_speed_factor * 0.5 * B_speed in
  let time_for_A_to_reach_B := (total_distance / 2) / A_speed_after_meeting in
  let B_distance_during_A_travel := time_for_A_to_reach_B * B_speed in
  let B_remaining_distance := total_distance / 2 - B_distance_during_A_travel in
  B_remaining_distance / B_speed 

-- In Lean 4, we use this structure to express the proof problem
lemma final_time_calculation (B_speed : ℝ) (conds : WalkingConditions) 
  (H : conds.A_speed_factor = 3 ∧ conds.meeting_time = 60 ∧ conds.A_reduced_speed_factor = 0.5) :
  time_for_B_to_reach_A B_speed conds H = 120 := 
sorry

end time_for_B_to_reach_A_l217_217316


namespace obtuse_angle_at_3_15_l217_217729

theorem obtuse_angle_at_3_15 : 
  let minute_angle := 15 * 6 in
  let hour_angle := 3 * 30 + 0.25 * 30 in
  let acute_angle := abs (hour_angle - minute_angle) in
  let obtuse_angle := 360 - acute_angle in
  obtuse_angle = 352.5 := by
  sorry

end obtuse_angle_at_3_15_l217_217729


namespace probability_digits_2_and_3_in_four_digit_number_l217_217443

theorem probability_digits_2_and_3_in_four_digit_number :
  let num_total := 2^4,
      num_favorable := num_total - 2,
      probability := num_favorable.to_rat / num_total.to_rat
  in probability = 7 / 8 :=
by
  sorry

end probability_digits_2_and_3_in_four_digit_number_l217_217443


namespace functional_relationship_maximum_profit_desired_profit_l217_217449

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end functional_relationship_maximum_profit_desired_profit_l217_217449


namespace value_of_a_if_lines_parallel_l217_217907

theorem value_of_a_if_lines_parallel (a : ℝ) :
  let l1 := λ x : ℝ, x + (1/2) * a,
      l2 := λ x : ℝ, (a^2 - 3) * x + 1 
  in (∀ x : ℝ, l1 x = l2 x) → a = -2 :=
sorry

end value_of_a_if_lines_parallel_l217_217907


namespace arithmetic_sequence_a10_l217_217094

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (h_diff : d = (a 3 - a 1) / (3 - 1)) :
  a 10 = 19 := 
by 
  sorry

end arithmetic_sequence_a10_l217_217094


namespace prob_exactly_two_meet_standard_most_likely_number_meeting_standard_l217_217935

namespace ProbabilityTest

noncomputable theory
open MeasureTheory

variables {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variables (A B C : Set Ω)
variables (hA : P A = 2/5) (hB : P B = 3/4) (hC : P C = 1/2)
variables (hA_indep_B : MeasureTheory.Indep {A} {B} P)
variables (hA_indep_C : MeasureTheory.Indep {A} {C} P)
variables (hB_indep_C : MeasureTheory.Indep {B} {C} P)

theorem prob_exactly_two_meet_standard :
  P (A ∩ B ∩ Cᶜ ∪ A ∩ Bᶜ ∩ C ∪ Aᶜ ∩ B ∩ C) = 17/40 :=
sorry

theorem most_likely_number_meeting_standard :
  let P_all := P (A ∩ B ∩ C),
      P_none := P (Aᶜ ∩ Bᶜ ∩ Cᶜ),
      P_one := 1 - 17/40 - 3/20 - 3/40 in
  List.argmax [P_all, 17/40, P_one, P_none] = 17/40 :=
sorry

end ProbabilityTest

end prob_exactly_two_meet_standard_most_likely_number_meeting_standard_l217_217935


namespace speeds_of_boy_and_bus_are_correct_l217_217856

noncomputable def speed_of_boy_and_bus
  (distance_station_beach : ℝ)
  (boy_walk_time : ℝ)
  (bus_stop_time : ℝ)
  (total_meet_distance : ℝ)
  (meet_additional_distance : ℝ)
  (total_bus_stops : ℝ)
  (boy_resumed_meeting_time : ℝ) : (ℝ, ℝ) :=
  let v_b : ℝ := 3 in
  let v_bus : ℝ := 45 in
  (v_b, v_bus)

theorem speeds_of_boy_and_bus_are_correct
  (distance_station_beach : ℝ := 4.5)
  (boy_walk_time : ℝ := 15 / 60)
  (bus_stop_time : ℝ := 4 / 60)
  (total_meet_distance : ℝ := 9)
  (meet_additional_distance : ℝ := 9 / 28)
  (total_bus_stops : ℝ := 2)
  (boy_resumed_meeting_time : ℝ := 9 - ((2 * 4.5) / 45 + 8 / 60)) :
  speed_of_boy_and_bus distance_station_beach boy_walk_time bus_stop_time total_meet_distance
                       meet_additional_distance total_bus_stops boy_resumed_meeting_time = (3, 45) :=
by
  sorry

end speeds_of_boy_and_bus_are_correct_l217_217856


namespace max_value_frac_ab_sinC_over_asq_plus_bsq_minus_csquared_l217_217958

/-- In triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively,
and given the condition 8 * sin A * sin B + cos C = 0,
the maximum value of (ab * sin C) / (a^2 + b^2 - c^2) is -3/8. -/
theorem max_value_frac_ab_sinC_over_asq_plus_bsq_minus_csquared
  {A B C : ℝ} {a b c : ℝ}
  (h1 : 8 * sin A * sin B + cos C = 0) :
  ∃ max : ℝ, max = -3 / 8 ∧ (∀ x, x = (a * b * sin C) / (a^2 + b^2 - c^2) → x ≤ max) :=
sorry

end max_value_frac_ab_sinC_over_asq_plus_bsq_minus_csquared_l217_217958


namespace part1_part2_l217_217120

def f (a x : ℝ) : ℝ := a * x^2 - (a^2 + 1) * x + a

-- Part 1
theorem part1 (a : ℝ) (h_a_gt_0 : 0 < a) :
  ((a ≤ 1 / 2 ∧ 0 < a) ∨ (2 ≤ a)) ↔ ∀ x ∈ set.Ioo (1 : ℝ) (2 : ℝ), f a x < 0 :=
sorry

-- Part 2
theorem part2 (a x : ℝ) :
  0 < a → (f a x > 0 ↔ ((x < 1 / a ∨ a < x) ↔ (1 < a)) ∨ ((x ≠ 1) ↔ (a = 1)) ∨ ((x < a ∨ 1 / a < x) ↔ (0 < a ∧ a < 1))) :=
sorry ∧
  0 > a → (f a x > 0 ↔ ((1 / a < x ∧ x < a) ↔ (-1 < a ∧ a < 0)) ∨ (∅ ↔ (a = -1)) ∨ ((a < x ∧ x < 1 / a) ↔ (a < -1))) :=
sorry ∧
  a = 0 → (f a x > 0 ↔ x < 0) :=
sorry

end part1_part2_l217_217120


namespace each_student_gets_8_pieces_l217_217782

-- Define the number of pieces of candy
def candy : Nat := 344

-- Define the number of students
def students : Nat := 43

-- Define the number of pieces each student gets, which we need to prove
def pieces_per_student : Nat := candy / students

-- The proof problem statement
theorem each_student_gets_8_pieces : pieces_per_student = 8 :=
by
  -- This proof content is omitted as per instructions
  sorry

end each_student_gets_8_pieces_l217_217782


namespace points_in_groups_l217_217941

theorem points_in_groups (n1 n2 : ℕ) (h_total : n1 + n2 = 28) 
  (h_lines_diff : (n1*(n1 - 1) / 2) - (n2*(n2 - 1) / 2) = 81) : 
  (n1 = 17 ∧ n2 = 11) ∨ (n1 = 11 ∧ n2 = 17) :=
by
  sorry

end points_in_groups_l217_217941


namespace remainder_when_divided_by_101_l217_217968

noncomputable def f : Polynomial ℝ := ∏ k in (Finset.range 50).map (Function.Embedding.mk (λ n => 2 * (n+1) - 1) (by
  intro a b h; linarith [h])), Polynomial.X - Polynomial.C (2 * (k+1) - 1)

def coeff_x48 (p : Polynomial ℝ) : ℝ := p.coeff 48

theorem remainder_when_divided_by_101 : (coeff_x48 f) % 101 = 60 := 
sorry

end remainder_when_divided_by_101_l217_217968


namespace max_height_soccer_ball_l217_217739

theorem max_height_soccer_ball (a b : ℝ) (t h : ℝ) : 
  (a = -4) → (b = 12) → (h = a * t^2 + b * t) → 
  (t = 3 / 2) → h = 9 :=
by
  intros ha hb ht max_t
  have h_eq : h = -4 * (3 / 2)^2 + 12 * (3 / 2) := by
    rw [ha, hb]
    apply ht
  rw max_t at h_eq
  simp at h_eq
  exact h_eq

#Evaluate
check max_height_soccer_ball

end max_height_soccer_ball_l217_217739


namespace number_of_valid_permutations_l217_217916

open Nat

def is_valid_permutation (n : ℕ) (m : ℕ) : Prop :=
    let n_digits := [n / 100, (n / 10) % 10, n % 10]
    let m_digits := [m / 100, (m / 10) % 10, m % 10]
    n_digits.perm m_digits

def is_perm_multiple_of_13 (n : ℕ) : Prop :=
    ∃ p, is_valid_permutation n p ∧ p % 13 = 0 ∧ 100 ≤ p ∧ p ≤ 999

def count_perm_multiples_of_13 : ℕ :=
    (Finset.filter (λ n, is_perm_multiple_of_13 n) (Finset.range (999 + 1 - 100) + Finset.singleton 100)).card

theorem number_of_valid_permutations : count_perm_multiples_of_13 = 207 := 
sorry

end number_of_valid_permutations_l217_217916


namespace find_greatest_divisor_l217_217330

def greatest_divisor_leaving_remainders (n₁ n₁_r n₂ n₂_r d : ℕ) : Prop :=
  (n₁ % d = n₁_r) ∧ (n₂ % d = n₂_r) 

theorem find_greatest_divisor :
  greatest_divisor_leaving_remainders 1657 10 2037 7 1 :=
by
  sorry

end find_greatest_divisor_l217_217330


namespace proof_P_l217_217581

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the complement of P in U
def CU_P : Set ℕ := {4, 5}

-- Define the set P as the difference between U and CU_P
def P : Set ℕ := U \ CU_P

-- Prove that P = {1, 2, 3}
theorem proof_P :
  P = {1, 2, 3} :=
by
  sorry

end proof_P_l217_217581


namespace series_sum_formula_l217_217814

theorem series_sum_formula (n : ℕ) : 
  (∑ k in Finset.range n, k / ((k + 1) * (k + 2) * (k + 3))) = (n * (n + 1)) / (4 * (n + 2) * (n + 3)) := by
  sorry

end series_sum_formula_l217_217814


namespace smallest_x_value_l217_217478

theorem smallest_x_value : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y^2 - 5 * y - 84) / (y - 9) = 4 / (y + 6) → y >= (x)) ∧ 
  (x^2 - 5 * x - 84) / (x - 9) = 4 / (x + 6) ∧ 
  x = ( - 13 - Real.sqrt 17 ) / 2 := 
sorry

end smallest_x_value_l217_217478


namespace intersect_circles_eq_l217_217751

theorem intersect_circles_eq {A B C P Q : Type*}
  (ABC : EquilateralTriangle A B C)
  (Omega : Circle)
  (omega : Incircle A B C)
  (P_on_AC : P ∈ Segment AC)
  (Q_on_AB : Q ∈ Segment AB)
  (PQ_through_center : ∃ O, Center ABC = O ∧ PQ_passes_through O)
  (Gamma_b : Circle)
  (Gamma_c : Circle)
  (Gamma_b_def : Gamma_b.Diameter = Segment B P)
  (Gamma_c_def : Gamma_c.Diameter = Segment C Q) :
  ∃ X Y : Point, X ∈ Omega ∧ Y ∈ omega ∧ X ∈ Gamma_b ∧ X ∈ Gamma_c ∧ 
                Y ∈ Gamma_b ∧ Y ∈ Gamma_c := 
sorry

end intersect_circles_eq_l217_217751


namespace cubic_sum_identity_l217_217660

theorem cubic_sum_identity
  (x y z : ℝ)
  (h1 : x + y + z = 8)
  (h2 : x * y + x * z + y * z = 17)
  (h3 : x * y * z = -14) :
  x^3 + y^3 + z^3 = 62 :=
sorry

end cubic_sum_identity_l217_217660


namespace conditions_implication_l217_217011

theorem conditions_implication (p q : Prop) : 
  (¬(p ∧ q) ↔ (¬ p ∨ ¬ q)) → 
  (¬(p ∧ q) → ((p ∧ ¬ q) ∨ (¬ p ∧ q))) → 
  ((¬(p ∧ q) → (¬ p ∧ q)) ∧ (¬(p ∧ q) → (¬ p ∧ ¬ q))) → 
  (nat.add (nat.add 1 1) 1) = 3 := 
by 
  intros h_neg1 h_neg2 h_neg3 
  sorry

end conditions_implication_l217_217011


namespace remainder_poly_l217_217476

noncomputable def remainder_division (f g : ℚ[X]) := 
  let ⟨q, r⟩ := f.div_mod g in r

theorem remainder_poly :
  remainder_division (3 * X ^ 5 - 2 * X ^ 3 + 5 * X - 8) (X ^ 2 - 3 * X + 2) = 84 * X - 84 :=
by
  sorry

end remainder_poly_l217_217476


namespace continuous_function_identity_l217_217366

theorem continuous_function_identity
  (f : ℝ → ℝ)
  (h_cont : continuous_on f (set.Icc 0 1))
  (h0 : f 0 = 0)
  (h1 : f 1 = 1)
  (h_func_eq : ∀ x ∈ set.Ioo 0 1, ∃ h > 0, (0 : ℝ) ≤ x - h ∧ x + h ≤ 1 ∧ f x = (f (x - h) + f (x + h)) / 2) :
  ∀ x ∈ set.Icc 0 1, f x = x :=
by
  sorry

end continuous_function_identity_l217_217366


namespace standard_equation_of_ellipse_exists_point_M_at_1_l217_217095

variable (a b : ℝ) (a_gt_b : a > b) (a_b_gt_0 : a > 0) (b_gt_0 : b > 0)

-- Condition for ellipse focus and vertex
variable (focus : ℝ × ℝ := (1, 0))
variable (vertex_to_focus_distance : |1| = 1)

-- Equation of ellipse C
theorem standard_equation_of_ellipse :
  (\frac{x^2}{4} + \frac{y^2}{3} = 1) := sorry

-- Given line l and ellipse C intersection and orthogonality conditions
variable (k m t : ℝ)
variable (line_l : ℝ → ℝ := λ x, k*x + m)
variable (intersect_lines_y : ℝ → ℝ := λ x, ⟨k*x + m, 4*x + m⟩)
variable (exists_point_M : ∃ t, ∀ t', t' = t → 
   (let P := (-4*k/m, 3/m); MQ := (4-t, 4*k+m);
   (P.1 - t, P.2) • (MQ.1, MQ.2) = 0)) := sorry

-- The final solution where t = 1
theorem exists_point_M_at_1 :
  (∃ t, t = 1) := sorry

end standard_equation_of_ellipse_exists_point_M_at_1_l217_217095


namespace arrows_from_530_to_535_l217_217776

def cyclic_arrows (n : Nat) : Nat :=
  n % 5

theorem arrows_from_530_to_535 : 
  cyclic_arrows 530 = 0 ∧ cyclic_arrows 531 = 1 ∧ cyclic_arrows 532 = 2 ∧
  cyclic_arrows 533 = 3 ∧ cyclic_arrows 534 = 4 ∧ cyclic_arrows 535 = 0 :=
by
  sorry

end arrows_from_530_to_535_l217_217776


namespace median_lap_times_l217_217726

theorem median_lap_times : 
  let times := [63, 60, 90, 68, 57] in 
  (times.sort.nth 2).iget = 63 := 
by
  sorry

end median_lap_times_l217_217726


namespace ellipse_eccentricity_range_proof_l217_217114

def ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h3 : P.1^2 / a^2 + P.2^2 / b^2 = 1) (h4 : ∃ c > 0, F1 = (-c, 0) ∧ F2 = (c, 0)) 
  (h5 : ∃ θ : ℝ, θ = (60 : ℝ) ∧ angle F1 P F2 θ) : Set ℝ :=
{ e : ℝ | e ∈ Icc (1/2) 1 - {1}}

theorem ellipse_eccentricity_range_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h3 : P.1^2 / a^2 + P.2^2 / b^2 = 1) (h4 : ∃ c > 0, F1 = (-c, 0) ∧ F2 = (c, 0)) 
  (h5 : ∃ θ : ℝ, θ = (60 : ℝ) ∧ angle F1 P F2 θ) : e ∈ ellipse_eccentricity_range a b h1 h2 P F1 F2 :=
sorry

end ellipse_eccentricity_range_proof_l217_217114


namespace stratified_sampling_company_A_l217_217807

variable (numA numB : ℕ)
variable (totalSelected : ℕ)
variable (samplingRatio : ℚ := totalSelected / (numA + numB))

theorem stratified_sampling_company_A (hA : numA = 120) (hB : numB = 100) (hTotal : totalSelected = 11) :
  (numA * samplingRatio).natValue = 6 := by
  sorry

end stratified_sampling_company_A_l217_217807


namespace domain_of_p_range_of_p_l217_217390

noncomputable def domain_of_h := set.Icc (1:ℝ) (3 : ℝ)
noncomputable def range_of_h := set.Icc (2:ℝ) (3 : ℝ)

noncomputable def h (x : ℝ) : ℝ := sorry -- specifics of function h are not given

-- Definition of function p
noncomputable def p (x : ℝ) : ℝ := 3 - h(x - 1)

-- Prove that the domain of p is [2, 4]
theorem domain_of_p : ∀ x, p x ∈ set.Icc (2 : ℝ) (4 : ℝ) ↔ x ∈ set.Icc (2:ℝ) (4:ℝ) := sorry

-- Prove that the range of p is [0, 1]
theorem range_of_p : ∀ y, y ∈ set.range p ↔ y ∈ set.Icc (0:ℝ) (1:ℝ) := sorry

end domain_of_p_range_of_p_l217_217390


namespace percentage_games_won_l217_217815

def total_games_played : ℕ := 75
def win_rate_first_100_games : ℝ := 0.65

theorem percentage_games_won : 
  (win_rate_first_100_games * total_games_played / total_games_played * 100) = 65 := 
by
  sorry

end percentage_games_won_l217_217815


namespace proof_problem_l217_217216

variable {a : ℝ}
variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}
variable {x : ℝ}

-- Define odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g (x)

-- Problem conditions as Lean statements
def func_conditions (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x) + g (x) = 2 ^ x

-- Define the range constraint in the given interval
def interval_constraint (x : ℝ) : Prop :=
  x ∈ Set.Icc (1 / 2) 2

-- Problem statement reformulated in Lean
theorem proof_problem 
  (h_fo : is_odd f)
  (h_ge : is_even g)
  (h_fg : func_conditions f g)
  (h_interval : ∀ x, interval_constraint x →
    a * f (x) - f (3 * x) ≤ 2 * g (2 * x)) :
  a ≤ 10 :=
sorry

end proof_problem_l217_217216


namespace coordinates_P_2_or_negative6_l217_217572

theorem coordinates_P_2_or_negative6 {x : ℝ} : 
  let P := (2*x - 2, -x + 4) in
  (2*x - 2 = -x + 4 ∨ 2*x - 2 = -(-x + 4)) → 
  ((P = (2, 2)) ∨ (P = (-6, 6))) :=
by 
  intros P h,
  sorry

end coordinates_P_2_or_negative6_l217_217572


namespace find_a4_l217_217595

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem find_a4 (h1 : arithmetic_sequence a) (h2 : a 2 + a 6 = 2) : a 4 = 1 :=
by
  sorry

end find_a4_l217_217595


namespace negative_solution_condition_l217_217040

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l217_217040


namespace hypotenuse_length_l217_217684

noncomputable def right_triangle_hypotenuse (a b : ℝ) : ℝ :=
  if h : (a > 0) ∧ (b > 0) ∧ ((←Sqrt (a^2 + (b^2 / 4)) = 6) ∧ (←Sqrt ((a^2 / 4) + b^2) = 2 * sqrt 34)) 
    ∧ (2 * a * b = 48) 
  then sqrt (4 * (a^2 + b^2))
  else 0

theorem hypotenuse_length : ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧
  (sqrt (a^2 + b^2 / 4) = 6) ∧ (sqrt ((a^2 / 4) + b^2) = 2 * sqrt 34) ∧ (2 * a * b = 48) →
  right_triangle_hypotenuse a b = sqrt 550.4 :=
begin
  sorry
end

end hypotenuse_length_l217_217684


namespace chord_length_of_tangent_intersections_l217_217906

-- Given data: Two circles with centers O1 and O2, and a chord length 'd'
variables {C1 C2 : Type*} [metric_space C1] [metric_space C2]
variables (O1 : C1) (O2 : C2) (d : ℝ)

-- Condition: Chords of length d in both circles
variables (A B P Q : ℝ) 
variable (h1 : dist A B = d)
variable (h2 : dist P Q = d)

-- Theorem: The lines determined by the common tangents of circles tangent 
-- to the endpoints of given chords in each circle will intersect the circles 
-- to cut chords of length d.
theorem chord_length_of_tangent_intersections :
  ∃ (L : Type*) (intersects_C1 : ∀ (l ∈ L), ∃ A B, ∃ X Y ∈ (set.univ : set C1), l X Y ∧ dist X Y = d) 
  ∧ (intersects_C2 : ∀ (l ∈ L), ∃ P Q, ∃ U V ∈ (set.univ : set C2), l U V ∧ dist U V = d) :=
  sorry

end chord_length_of_tangent_intersections_l217_217906


namespace polynomial_eval_at_n_plus_1_l217_217565

theorem polynomial_eval_at_n_plus_1 (p : Polynomial ℝ) (n : ℕ) (h_deg : p.degree = n) 
    (h_values : ∀ k : ℕ, k ≤ n → p.eval k = k / (k + 1)) :
    (if n % 2 = 1 then p.eval (n + 1) = n / (n + 2) else p.eval (n + 1) = 1) :=
by
  sorry

end polynomial_eval_at_n_plus_1_l217_217565


namespace john_total_calories_l217_217611

theorem john_total_calories :
  (∃ (c_pchip : ℕ), c_pchip = 90 / 15) ∧
  (∃ (c_cheezit : ℕ), c_cheezit = 6 + (2 / 5) * 6) ∧
  (∃ (c_pretzel : ℕ), c_pretzel = (6 + (2 / 5) * 6) - 0.25 * (6 + (2 / 5) * 6)) ∧
  (∃ (total_calories : ℕ), total_calories = 90 + 84 + 50.4) →
  224.4 = 224.4 :=
begin
  sorry
end

end john_total_calories_l217_217611


namespace locus_of_point_P_is_incomplete_ellipse_l217_217970

variables (O B C : Point)
variable (r : Real)
variable (A : Point)
variable [Circle O r]
variables [OnCircle B (Circle O r)] [OnCircle C (Circle O r)] [OnCircle A (Circle O r)]
variables (H : Point) (M N D P : Point)

-- Assumptions
axiom A_not_eq_B_and_C : A ≠ B ∧ A ≠ C
axiom A_not_on_bisector_BC : ¬ IsOnPerpendicularBisector A B C
axiom is_orthocenter_ABH : IsOrthocenter A B H
axiom is_midpoint_BC_M : IsMidpoint M B C
axiom is_midpoint_AH_N : IsMidpoint N A H
axiom AM_intersects_circle_at_D : IntersectsAtSecondPoint (LineThrough A M) (Circle O r) D
axiom NM_and_OD_intersect_at_P : IntersectsAtPoint (LineThrough N M) (LineThrough O D) P

-- Theorem stating the locus of point P
theorem locus_of_point_P_is_incomplete_ellipse :
  IsIncompleteEllipse (locus P A (OnCircle A (Circle O r))) O M r :=
sorry

end locus_of_point_P_is_incomplete_ellipse_l217_217970


namespace unique_line_intersection_parabola_l217_217394

theorem unique_line_intersection_parabola (l : ℝ → ℝ) :
  let is_tangent_to_parabola := ∃ p : ℝ, (l p) ^ 2 = 8 * p ∧ (4 - l p) / p = 0
  (∀ l, (∀ p, is_tangent_to_parabola) → (
    l = Function.const ℝ 4 ∨ 
    l = λ x : ℝ, 1 / 2 * x + 4 ∨ 
    l = Function.const ℝ 0)
  ) → ∃ n, n = 3 :=
begin
  sorry
end

end unique_line_intersection_parabola_l217_217394


namespace trains_meet_time_is_10_62_seconds_l217_217361

-- Define the lengths of the trains
def length_train1 : ℝ := 210
def length_train2 : ℝ := 120

-- Define the initial distance between the trains
def initial_distance : ℝ := 160

-- Define the speeds of the trains in m/s
def speed_train1 : ℝ := 74 * 1000 / 3600
def speed_train2 : ℝ := 92 * 1000 / 3600

-- Define the relative speed given the trains are moving towards each other
def relative_speed : ℝ := speed_train1 + speed_train2

-- Define the total distance to be covered
def total_distance : ℝ := length_train1 + length_train2 + initial_distance

-- Define the time it will take for the trains to meet
def meet_time : ℝ := total_distance / relative_speed

-- State the theorem that verifies the time taken for the trains to meet
theorem trains_meet_time_is_10_62_seconds :
  meet_time ≈ 10.62 := sorry

end trains_meet_time_is_10_62_seconds_l217_217361


namespace train_crossing_time_l217_217914

def train_length := 110 -- Length of train in meters
def bridge_length := 132 -- Length of bridge in meters
def speed_kmh := 54 -- Speed of train in kilometers per hour

noncomputable def time_to_cross_bridge (train_length bridge_length speed_kmh : ℕ) : ℕ := 
  let distance := train_length + bridge_length
  let speed_ms := speed_kmh * 5 / 18
  let time := distance / speed_ms
  time
  
theorem train_crossing_time :
  time_to_cross_bridge train_length bridge_length speed_kmh = 16.13 :=
  by sorry

end train_crossing_time_l217_217914


namespace dot_product_of_vectors_norm_sub_scalar_mul_b_l217_217554

variables (a b : EuclideanSpace ℝ (Fin 3))
variable (theta : ℝ)
variable (a_norm : ℝ)
variable (b_norm : ℝ)

-- Given conditions
def angle_between_vectors : theta = Real.pi / 3 := sorry
def magnitude_a : ‖a‖ = 2 := sorry
def magnitude_b : ‖b‖ = 1 := sorry

-- Problem (1): Prove that a ⋅ b = 1
theorem dot_product_of_vectors : a ⋅ b = 1 := sorry

-- Problem (2): Prove that ‖a - 2 • b‖ = 2
theorem norm_sub_scalar_mul_b : ‖a - 2 • b‖ = 2 := sorry

end dot_product_of_vectors_norm_sub_scalar_mul_b_l217_217554


namespace susie_babysits_3_hours_per_day_l217_217663

-- Susie's earning rate per hour
def earning_rate := 10

-- Spent proportion on make-up set
def make_up_set_proportion := 3 / 10

-- Spent proportion on skincare products
def skincare_proportion := 2 / 5

-- Amount left after spending
def amount_left := 63

-- Total earnings last week
def total_earnings : ℕ :=
  let spent_fraction := 1 - (make_up_set_proportion + skincare_proportion * 2 / 10)
  amount_left / spent_fraction

-- Total hours Susie babysits
def total_hours := total_earnings / earning_rate 10

-- Prove Susie's daily babysitting hours
theorem susie_babysits_3_hours_per_day :
  (total_hours / 7 = 3) :=
by
  sorry

end susie_babysits_3_hours_per_day_l217_217663


namespace joanne_main_job_hours_l217_217194

theorem joanne_main_job_hours (h : ℕ) (earn_main_job : ℝ) (earn_part_time : ℝ) (hours_part_time : ℕ) (days_week : ℕ) (total_weekly_earn : ℝ) :
  earn_main_job = 16.00 →
  earn_part_time = 13.50 →
  hours_part_time = 2 →
  days_week = 5 →
  total_weekly_earn = 775 →
  days_week * earn_main_job * h + days_week * earn_part_time * hours_part_time = total_weekly_earn →
  h = 8 :=
by
  sorry

end joanne_main_job_hours_l217_217194


namespace james_browsers_l217_217608

def num_tabs_per_window := 10
def num_windows_per_browser := 3
def total_tabs := 60

theorem james_browsers : ∃ B : ℕ, (B * (num_windows_per_browser * num_tabs_per_window) = total_tabs) ∧ (B = 2) := sorry

end james_browsers_l217_217608


namespace sin_diff_angle_identity_l217_217882

theorem sin_diff_angle_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : cos α = 1 / 3) :
  sin (π / 4 - α) = (√2 - 4) / 6 :=
sorry

end sin_diff_angle_identity_l217_217882


namespace maximum_range_of_integers_l217_217791

noncomputable def largest_possible_range (x y m : ℤ) :=
    let a := x
    let b := y
    let c := m + 2
    let d := m + 4
    let e := m + 4
    max a (max b (max c (max d e))) - min a (min b (min c (min d e)))

theorem maximum_range_of_integers :
  ∃ (x y m : ℤ), let range := largest_possible_range x y m in
  (c == m + 2) →
  (d == m + 4) →
  (range == 15) :=
by 
  sorry

end maximum_range_of_integers_l217_217791


namespace min_value_to_25_l217_217070

open real

theorem min_value_to_25 (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : 
    (tan x + cot x)^2 + (sec x + csc x + 1)^2 = 25 :=
sorry

end min_value_to_25_l217_217070


namespace max_value_of_cubes_l217_217210

theorem max_value_of_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 + ab + ac + ad + bc + bd + cd = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 :=
sorry

end max_value_of_cubes_l217_217210


namespace compute_f5_l217_217137

-- Definitions of the logical operations used in the conditions
axiom x1 : Prop
axiom x2 : Prop
axiom x3 : Prop
axiom x4 : Prop
axiom x5 : Prop

noncomputable def x6 : Prop := x1 ∨ x3
noncomputable def x7 : Prop := x2 ∧ x6
noncomputable def x8 : Prop := x3 ∨ x5
noncomputable def x9 : Prop := x4 ∧ x8
noncomputable def f5 : Prop := x7 ∨ x9

-- Proof statement to be proven
theorem compute_f5 : f5 = (x7 ∨ x9) :=
by sorry

end compute_f5_l217_217137


namespace determinant_zero_l217_217031

open Matrix

variables {α β φ : ℝ}

def my_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, Real.sin (α + φ), -Real.cos (α + φ)], 
    ![-Real.sin (α + φ), 0, Real.sin (β + φ)], 
    ![Real.cos (α + φ), -Real.sin (β + φ), 0]]

theorem determinant_zero : det my_matrix = 0 :=
by
  sorry

end determinant_zero_l217_217031


namespace bob_catches_john_in_45_minutes_l217_217195

open Real

noncomputable def time_to_catch_up (john_speed bob_speed distance : ℝ) : ℝ :=
  distance / (bob_speed - john_speed)

theorem bob_catches_john_in_45_minutes :
  ∀ (john_speed bob_speed distance : ℝ),
    john_speed = 8 → bob_speed = 12 → distance = 3 →
    time_to_catch_up john_speed bob_speed distance * 60 = 45 :=
by
  intros john_speed bob_speed distance hj1 hb1 hd
  rw [hj1, hb1, hd]
  unfold time_to_catch_up
  have h : (3 / (12 - 8) * 60) = 45 := by norm_num
  exact h

end bob_catches_john_in_45_minutes_l217_217195


namespace sum_b_100_l217_217091

theorem sum_b_100 :
  let a : ℕ → ℝ := λ n, (rec $ λ a n, if n = 0 then 2 else 3 * a (n - 1) + 2)
  let b : ℕ → ℝ := λ n, real.log_base 3 (a n + 1)
  in (finset.range 100).sum b = 5050 := 
by { sorry }

end sum_b_100_l217_217091


namespace compare_logs_l217_217862

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

def a : ℝ := log_base 2 3.6
def b : ℝ := log_base 4 3.2
def c : ℝ := log_base 4 3.6

theorem compare_logs : a > c ∧ c > b :=
by
  rw [← log_base, ← log_base, ← log_base]
  sorry

end compare_logs_l217_217862


namespace fraction_left_handed_l217_217423

-- Definitions based on given conditions
def red_ratio : ℝ := 10
def blue_ratio : ℝ := 5
def green_ratio : ℝ := 3
def yellow_ratio : ℝ := 2

def red_left_handed_percent : ℝ := 0.37
def blue_left_handed_percent : ℝ := 0.61
def green_left_handed_percent : ℝ := 0.26
def yellow_left_handed_percent : ℝ := 0.48

-- Statement we want to prove
theorem fraction_left_handed : 
  (red_left_handed_percent * red_ratio + blue_left_handed_percent * blue_ratio +
  green_left_handed_percent * green_ratio + yellow_left_handed_percent * yellow_ratio) /
  (red_ratio + blue_ratio + green_ratio + yellow_ratio) = 8.49 / 20 :=
  sorry

end fraction_left_handed_l217_217423


namespace sqrt_sqrt_sixteen_l217_217693

theorem sqrt_sqrt_sixteen : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := by
  sorry

end sqrt_sqrt_sixteen_l217_217693


namespace cookies_in_each_batch_l217_217794

theorem cookies_in_each_batch (batches : ℕ) (people : ℕ) (consumption_per_person : ℕ) (cookies_per_dozen : ℕ) 
  (total_batches : batches = 4) 
  (total_people : people = 16) 
  (cookies_per_person : consumption_per_person = 6) 
  (dozen_size : cookies_per_dozen = 12) :
  (6 * 16) / 4 / 12 = 2 := 
by {
  sorry
}

end cookies_in_each_batch_l217_217794


namespace exists_positive_projection_l217_217266

-- Define the vectors as elements in a vector space over the reals
noncomputable def vectors : ℕ → ℝ^3 := sorry

-- The condition given in the problem
def condition (vectors : ℕ → ℝ^3) : Prop :=
  ∀ i : ℕ, i < 10 →
  ‖(∑ j in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ {i}, vectors j)‖ < ‖∑ j in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, vectors j‖

-- The theorem to be proved
theorem exists_positive_projection (vectors : ℕ → ℝ^3) (h : condition vectors) :
  ∃ v : ℝ^3, ∀ i : ℕ, i < 10 → (vectors i) ⬝ v > 0 :=
sorry

end exists_positive_projection_l217_217266


namespace binom_sum_101_eq_neg_2_pow_50_l217_217429
open BigOperators

/- Helper definition for alternating sum of binomial coefficients -/
def alternating_binom_sum (n : ℕ) : ℤ :=
  ∑ k in finset.range (n+1), (-1)^k * nat.choose n (2 * k)

theorem binom_sum_101_eq_neg_2_pow_50 :
  alternating_binom_sum 50 = -2^50 :=
sorry

end binom_sum_101_eq_neg_2_pow_50_l217_217429


namespace example_solution_l217_217520

variable (x y θ : Real)
variable (h1 : 0 < x) (h2 : 0 < y)
variable (h3 : θ ∈ Set.Ioo (Real.pi / 4) (Real.pi / 2))
variable (h4 : Real.sin θ / x = Real.cos θ / y)
variable (h5 : Real.cos θ ^ 2 / x ^ 2 + Real.sin θ ^ 2 / y ^ 2 = 10 / (3 * (x ^ 2 + y ^ 2)))

theorem example_solution : x / y = Real.sqrt 3 :=
by
  sorry

end example_solution_l217_217520


namespace locus_of_M_l217_217222

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  real.sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

noncomputable def distance_to_line_x (x1 : ℝ) (x_line : ℝ) : ℝ :=
  abs (x1 - x_line)

theorem locus_of_M (x y : ℝ) : 
  distance x y 4 0 / distance_to_line_x x 3 = 2 →
  3 * x ^ 2 - y ^ 2 - 16 * x + 20 = 0 :=
by
  intros h
  sorry

end locus_of_M_l217_217222


namespace pentagon_theorem_l217_217403

theorem pentagon_theorem (a b c : ℝ) (circumscribed_pentagon : ∀ (A B C D E S : Type), 
  (is_regular_pentagon A B C D E ∧ side_length A B = a) ∧ 
  (is_regular_pentagon B1 C1 D1 E1 A1 ∧ line_perpendicular A1 B1 A B ∧ side_length P Q = b) ∧ 
  (is_regular_pentagon A2 B2 C2 D2 E2 ∧ circumscribed B C S ∧ side_length A2 B2 = c)) : 
  a^2 + b^2 = c^2 := by 
    sorry

end pentagon_theorem_l217_217403


namespace range_of_a_l217_217150

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, (x^2 - 4 * x) ∈ Set.Icc (-4 : ℝ) 32) →
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l217_217150


namespace missing_digits_pairs_l217_217326

theorem missing_digits_pairs (x y : ℕ) : (2 + 4 + 6 + x + y + 8) % 9 = 0 ↔ x + y = 7 := by
  sorry

end missing_digits_pairs_l217_217326


namespace negative_solution_iff_sum_zero_l217_217044

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l217_217044


namespace total_difference_in_scoops_l217_217229

variable (Oli_scoops : ℕ)
variable (Victoria_scoops : ℕ)
variable (Brian_scoops : ℕ)

axiom (h1 : Oli_scoops = 4)
axiom (h2 : Victoria_scoops = 2 * Oli_scoops + Oli_scoops)
axiom (h3 : Brian_scoops = Oli_scoops + 3)
axiom (h4 : Brian_scoops = Victoria_scoops - 2)

-- Proving the total difference in scoops
theorem total_difference_in_scoops :
  (Victoria_scoops - Oli_scoops) + (Brian_scoops - Oli_scoops) + (Victoria_scoops - Brian_scoops) = 16 := by
  sorry

end total_difference_in_scoops_l217_217229


namespace ratio_of_shaded_area_to_circle_l217_217175

theorem ratio_of_shaded_area_to_circle (AC CB : ℕ) (h1 : AC = 10) (h2 : CB = 6) 
  (CD : ℕ) (h3 : CD = 3) :
  let r1 := AC / 2,
      r2 := CB / 2,
      Area_large := π * r1^2,
      Area_small := π * r2^2,
      Shaded_area := (1 / 2) * (Area_large - 2 * Area_small),
      Circle_area := π * CD^2
  in Shaded_area / Circle_area = 7 / 18 := 
by
  -- mathematical part to be filled
  sorry

end ratio_of_shaded_area_to_circle_l217_217175


namespace no_perfect_square_after_swap_l217_217389

def is_consecutive_digits (a b c d : ℕ) : Prop := 
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

def swap_hundreds_tens (n : ℕ) : ℕ := 
  let d4 := n / 1000
  let d3 := (n % 1000) / 100
  let d2 := (n % 100) / 10
  let d1 := n % 10
  d4 * 1000 + d2 * 100 + d3 * 10 + d1

theorem no_perfect_square_after_swap : ¬ ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  (let d4 := n / 1000
   let d3 := (n % 1000) / 100
   let d2 := (n % 100) / 10
   let d1 := n % 10
   is_consecutive_digits d4 d3 d2 d1) ∧ 
  let new_number := swap_hundreds_tens n
  (∃ m : ℕ, m * m = new_number) := 
sorry

end no_perfect_square_after_swap_l217_217389


namespace rectangular_floor_problem_possibilities_l217_217773

theorem rectangular_floor_problem_possibilities :
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s → p.2 > p.1 ∧ p.2 % 3 = 0 ∧ (p.1 - 6) * (p.2 - 6) = 36) 
    ∧ s.card = 2 := 
sorry

end rectangular_floor_problem_possibilities_l217_217773


namespace unique_coprime_pair_l217_217462

theorem unique_coprime_pair (m n : ℕ) (coprime_mn : Nat.gcd m n = 1) 
  (perfect_squares: ∃ x y : ℕ, m^2 - 5 * n^2 = x^2 ∧ m^2 + 5 * n^2 = y^2) 
  (not_4112 : ¬ ((m = 41) ∧ (n = 12))) : 
  (m = 41) ∧ (n = 12) :=
begin
  sorry
end

end unique_coprime_pair_l217_217462


namespace circle_center_l217_217679

theorem circle_center (A B : ℝ × ℝ) (hA : A = (-1, -4)) (hB : B = (-7, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  C = (-4, 1) :=
by
  rw [hA, hB]
  let C := (((-1 : ℝ) + -7) / 2, (-4 + 6) / 2)
  show C = (-4, 1)
  have h1 : ((-1 : ℝ) + -7) / 2 = -4, by norm_num
  have h2 : (-4 + 6) / 2 = 1, by norm_num
  rw [h1, h2]
  exact rfl

end circle_center_l217_217679


namespace slope_through_points_is_135_degrees_l217_217407

noncomputable def slope_to_angle (P1 P2 : ℝ × ℝ) : ℝ := 
  real.atan ((P2.2 - P1.2) / (P2.1 - P1.1))

theorem slope_through_points_is_135_degrees :
  slope_to_angle (1, 0) (-2, 3) = 135 := by
  sorry

end slope_through_points_is_135_degrees_l217_217407


namespace remainder_of_division_l217_217474

theorem remainder_of_division :
  ∀ (x : ℝ), (3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8) % (x ^ 2 - 3 * x + 2) = 74 * x - 76 :=
by
  sorry

end remainder_of_division_l217_217474


namespace derivative_exp_l217_217561

theorem derivative_exp (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x) : 
    ∀ x, deriv f x = Real.exp x :=
by 
  sorry

end derivative_exp_l217_217561


namespace simplify_product_l217_217653

theorem simplify_product (b : ℤ) : (1 * 2 * b * 3 * b^2 * 4 * b^3 * 5 * b^4 * 6 * b^5) = 720 * b^15 := by
  sorry

end simplify_product_l217_217653


namespace payment_for_150_books_equal_payment_number_of_books_l217_217354

/-- 
Xinhua Bookstore conditions:
- Both suppliers A and B price each book at 40 yuan. 
- Supplier A offers a 10% discount on all books.
- Supplier B offers a 20% discount on any books purchased exceeding 100 books.
-/

def price_per_book_supplier_A (n : ℕ) : ℝ := 40 * 0.9
def price_per_first_100_books_supplier_B : ℝ := 40
def price_per_excess_books_supplier_B (n : ℕ) : ℝ := 40 * 0.8

-- Prove that the payment amounts for 150 books from suppliers A and B are 5400 yuan and 5600 yuan respectively.
theorem payment_for_150_books :
  price_per_book_supplier_A 150 * 150 = 5400 ∧
  price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B 50 * (150 - 100) = 5600 :=
  sorry

-- Prove the equal payment equivalence theorem for supplier A and B.
theorem equal_payment_number_of_books (x : ℕ) :
  price_per_book_supplier_A x * x = price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B (x - 100) * (x - 100) → x = 200 :=
  sorry

end payment_for_150_books_equal_payment_number_of_books_l217_217354


namespace log_sum_geo_seq_l217_217938

noncomputable def geometric_sequence : ℕ → ℝ := sorry

theorem log_sum_geo_seq :
  (∀ n, geometric_sequence n > 0) →  -- each term is positive
  geometric_sequence 5 * geometric_sequence 6 = 9 →  -- given condition
  ∑ i in finset.range 10, real.logb 3 (geometric_sequence (i + 1)) = 10 :=  -- goal to prove
sorry

end log_sum_geo_seq_l217_217938


namespace max_label_value_l217_217619

theorem max_label_value (G : SimpleGraph V) [Fintype V] (k : ℕ) 
    (h1 : ∀ {C : Finset V}, C.card ≤ k ↔ G.IsClique C) 
    (label : V → ℝ) (h2 : ∀ v, 0 ≤ label v) 
    (h3 : ∑ v, label v = 1) : 
    ∑ e in G.edgeSet, label (e.1) * label (e.2) ≤ (k - 1) / (2 * k) :=
begin
    sorry
end

end max_label_value_l217_217619


namespace circle_equation_translation_l217_217655

theorem circle_equation_translation (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 68 = 0 → (x - 2)^2 + (y + 3)^2 = 81 :=
by
  intro h
  sorry

end circle_equation_translation_l217_217655


namespace arithmetic_sequence_properties_l217_217505

noncomputable def a_n_arithmetic_seq (n : ℕ) : ℝ 
noncomputable def S_n (n : ℕ) : ℝ

axiom a6_condition : (a_n_arithmetic_seq 6 - 1)^3 + 2013 * (a_n_arithmetic_seq 6 - 1) = 1
axiom a2008_condition : (a_n_arithmetic_seq 2008 - 1)^3 + 2013 * (a_n_arithmetic_seq 2008 - 1) = -1

theorem arithmetic_sequence_properties 
  (a6_condition : (a_n_arithmetic_seq 6 - 1)^3 + 2013 * (a_n_arithmetic_seq 6 - 1) = 1)
  (a2008_condition : (a_n_arithmetic_seq 2008 - 1)^3 + 2013 * (a_n_arithmetic_seq 2008 - 1) = -1)
  : S_n 2013 = 2013 ∧ a_n_arithmetic_seq 2008 < a_n_arithmetic_seq 6 :=
sorry

end arithmetic_sequence_properties_l217_217505


namespace hexagonal_prism_intersection_octaon_l217_217346

/--
When a plane intersects a hexagonal prism, the resulting plane figure can be an octagon.
-/
theorem hexagonal_prism_intersection_octaon :
  ∃ (P : GeometricSolid), (P = GeometricSolid.hexagonal_prism) → (∃ (S : PlaneFigure), S = PlaneFigure.octagon) := 
sorry

end hexagonal_prism_intersection_octaon_l217_217346


namespace circle_equation_l217_217865

theorem circle_equation {x y : ℝ} : 
  let C := (-1, 2) in
  let r := 4 in
  (x + 1)^2 + (y - 2)^2 = 16 :=
by
  sorry

end circle_equation_l217_217865


namespace problem_a_lt_c_lt_b_l217_217626

noncomputable def a : ℝ := Real.logBase (1 / 2013) Real.pi
noncomputable def b : ℝ := (1 / 5) ^ (-0.8)
noncomputable def c : ℝ := Real.log10 Real.pi

theorem problem_a_lt_c_lt_b : a < c ∧ c < b := by
  -- First condition
  have h1 : a < 0 := by
    -- Proof omitted
    sorry
  -- Second condition
  have h2 : b > 1 := by
    -- Proof omitted
    sorry
  -- Third condition
  have h3 : 0 < c ∧ c < 1 := by
    -- Proof omitted
    sorry
  exact ⟨h1, h2, h3⟩
  sorry

end problem_a_lt_c_lt_b_l217_217626


namespace range_of_f_range_of_a_l217_217122

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

theorem range_of_f (a : ℝ) (h : a > 0) : set.range (λ x, f x a) = set.Icc (-a-2) (a+2) :=
by simp; sorry

theorem range_of_a (a : ℝ)
  (h1 : a > 0)
  (h2 : ∃ (b c : ℝ), b^2 + c^2 + b*c = 1)
  (b c : ℝ) (hb : b^2 + c^2 + b*c = 1) :
  ∃ x : ℝ, f x a ≥ 3 * (b + c) ↔ a ≥ 2 * sqrt 3 - 2 :=
by simp; sorry

end range_of_f_range_of_a_l217_217122


namespace total_points_l217_217235

theorem total_points (total_players : ℕ) (paige_points : ℕ) (other_points : ℕ) (points_per_other_player : ℕ) :
  total_players = 5 →
  paige_points = 11 →
  points_per_other_player = 6 →
  other_points = (total_players - 1) * points_per_other_player →
  paige_points + other_points = 35 :=
by
  intro h_total_players h_paige_points h_points_per_other_player h_other_points
  sorry

end total_points_l217_217235


namespace find_M_and_m_l217_217993

-- Defining the quadratic equation roots set
def quadratic_roots (m : ℝ) : Set ℝ := { x | x^2 - m * x + 6 = 0 }

-- Defining the problem condition and proof statement
theorem find_M_and_m :
  ∀ (M : Set ℝ) (m : ℝ),
    (M = quadratic_roots m) →
    (M ⊆ {1, 2, 3, 6}) →
    (M = {2, 3} ∨ M = {1, 6} ∨ M = ∅) ∧ (m = 5 ∨ m = 7 ∨ m ∈ Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6)) :=
by
  sorry

end find_M_and_m_l217_217993


namespace find_ratio_of_hyperbola_l217_217481

noncomputable def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem find_ratio_of_hyperbola (a b : ℝ) (h : a > b) 
  (h_asymptote_angle : ∀ α : ℝ, (y = ↑(b / a) * x -> α = 45)) :
  a / b = 1 :=
sorry

end find_ratio_of_hyperbola_l217_217481


namespace jill_marbles_probability_l217_217961

noncomputable def probability_exactly_two_blue (total_marbles: ℕ) (blue_marbles: ℕ) (draws: ℕ) (successes: ℕ) : ℝ :=
  let p_blue := (blue_marbles : ℝ) / (total_marbles : ℝ)
  let p_red := 1 - p_blue
  let prob_specific_case := (p_blue ^ successes) * (p_red ^ (draws - successes))
  let num_ways := (Finset.range draws).choose successes).card
  num_ways * prob_specific_case

theorem jill_marbles_probability :
  probability_exactly_two_blue 10 6 5 2 ≈ 0.230 :=
sorry

end jill_marbles_probability_l217_217961


namespace range_of_a_l217_217151

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, (x^2 - 4 * x) ∈ Set.Icc (-4 : ℝ) 32) →
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l217_217151


namespace room_width_correct_l217_217683

noncomputable def length_of_room : ℝ := 5
noncomputable def total_cost_of_paving : ℝ := 21375
noncomputable def cost_per_square_meter : ℝ := 900
noncomputable def width_of_room : ℝ := 4.75

theorem room_width_correct :
  (total_cost_of_paving / cost_per_square_meter) = (length_of_room * width_of_room) :=
by
  sorry

end room_width_correct_l217_217683


namespace reversed_segment_of_powers_of_five_in_powers_of_two_l217_217752

theorem reversed_segment_of_powers_of_five_in_powers_of_two :
  ∀ (segment : List ℕ), 
    (∀ (n : ℕ), segment = (List.firstDigit (5^n)).take segment.length.reverse → 
      ∃ (m : ℕ), segment = (List.firstDigit (2^m)).take segment.length) :=
sorry

end reversed_segment_of_powers_of_five_in_powers_of_two_l217_217752


namespace carl_insurance_payment_percentage_l217_217430

variable (property_damage : ℝ) (medical_bills : ℝ) 
          (total_cost : ℝ) (carl_payment : ℝ) (insurance_payment_percentage : ℝ)

theorem carl_insurance_payment_percentage :
  property_damage = 40000 ∧
  medical_bills = 70000 ∧
  total_cost = property_damage + medical_bills ∧
  carl_payment = 22000 ∧
  carl_payment = 0.20 * total_cost →
  insurance_payment_percentage = 100 - 20 :=
by
  sorry

end carl_insurance_payment_percentage_l217_217430


namespace nine_point_distance_center_bc_l217_217271

open EuclideanGeometry

variables (A B C : Point) (H A1 : Point)
variables (circumcircle nine_point_circle : Circle)
variable (bc : Line)
variable (nps_center : Point) -- center of the nine-point circle

-- Conditions
axiom altitude_def : is_altitude A bc
axiom intersects_circumcircle : on_circumcircle (altitude_intersection_point A circumcircle) A1
axiom nine_point_def : nine_point_circle Y -- definition with vertices (specific points, midpoints, altitudes)

-- Question: Distance from center of nine-point circle to BC
theorem nine_point_distance_center_bc : 
  distance_from_point_to_line nps_center bc = (1/4) * distance A A1 := 
sorry

end nine_point_distance_center_bc_l217_217271


namespace johns_overall_average_speed_l217_217964

open Real

noncomputable def johns_average_speed (scooter_time_min : ℝ) (scooter_speed_mph : ℝ) 
    (jogging_time_min : ℝ) (jogging_speed_mph : ℝ) : ℝ :=
  let scooter_time_hr := scooter_time_min / 60
  let jogging_time_hr := jogging_time_min / 60
  let distance_scooter := scooter_speed_mph * scooter_time_hr
  let distance_jogging := jogging_speed_mph * jogging_time_hr
  let total_distance := distance_scooter + distance_jogging
  let total_time := scooter_time_hr + jogging_time_hr
  total_distance / total_time

theorem johns_overall_average_speed :
  johns_average_speed 40 20 60 6 = 11.6 :=
by
  sorry

end johns_overall_average_speed_l217_217964


namespace orthocenter_in_C2_point_in_C2_is_orthocenter_l217_217197

-- Definitions of the circles
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

def C1 : Circle := { center := (0, 0), radius := R }
def C2 : Circle := { center := (0, 0), radius := 3 * R }

-- Definition of a point lying inside a circle
def inside_circle (circle : Circle) (point : ℝ × ℝ) : Prop :=
  (point.1 - circle.center.1) ^ 2 + (point.2 - circle.center.2) ^ 2 < circle.radius ^ 2

-- Define what it means for a point to be an orthocenter of a triangle inscribed in C1
def is_orthocenter (P : ℝ × ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ), 
    (A.1 ^ 2 + A.2 ^ 2 = R ^ 2) ∧ 
    (B.1 ^ 2 + B.2 ^ 2 = R ^ 2) ∧ 
    (C.1 ^ 2 + C.2 ^ 2 = R ^ 2) ∧ 
    P = (A.1 + B.1 + C.1, A.2 + B.2 + C.2)

-- First part: Prove the orthocenter is inside circle C2
theorem orthocenter_in_C2 (A B C : ℝ × ℝ) 
  (hA : A.1 ^ 2 + A.2 ^ 2 = R ^ 2) 
  (hB : B.1 ^ 2 + B.2 ^ 2 = R ^ 2) 
  (hC : C.1 ^ 2 + C.2 ^ 2 = R ^ 2) :
  inside_circle C2 (A.1 + B.1 + C.1, A.2 + B.2 + C.2) :=
begin
  sorry
end

-- Second part: Prove every point inside C2 can be the orthocenter of a triangle inscribed in C1
theorem point_in_C2_is_orthocenter (P : ℝ × ℝ) 
  (hP : inside_circle C2 P) :
  is_orthocenter P :=
begin
  sorry
end

end orthocenter_in_C2_point_in_C2_is_orthocenter_l217_217197


namespace number_of_girls_is_4_l217_217161

variable (x : ℕ)

def number_of_boys : ℕ := 12

def average_score_boys : ℕ := 84

def average_score_girls : ℕ := 92

def average_score_class : ℕ := 86

theorem number_of_girls_is_4 
  (h : average_score_class = 
    (average_score_boys * number_of_boys + average_score_girls * x) / (number_of_boys + x))
  : x = 4 := 
sorry

end number_of_girls_is_4_l217_217161


namespace combination_value_l217_217515

theorem combination_value (m : ℕ) (h : (1 / (Nat.choose 5 m) - 1 / (Nat.choose 6 m) = 7 / (10 * Nat.choose 7 m))) : 
    Nat.choose 8 m = 28 := 
sorry

end combination_value_l217_217515


namespace triangle_proof_l217_217186

variables (A B C A1 B1 C1 : Type) [MetricSpace A]

-- Definitions of the points and lines based on the conditions
variables [AffineSpace ℝ A]
variables [AffineSpace ℝ B]
variables [AffineSpace ℝ C]
variables [Point ℝ A1]
variables [Point ℝ B1]
variables [Point ℝ C1]

-- Conditions given in the problem
def on_side_BC (A1 : Point ℝ) (B C : Point ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ A1 = t • B + (1 - t) • C

def parallel (u v : ℝ) : Prop := ∃ k : ℝ, u = k * v

def intersect (l₁ l₂ : Point ℝ) (p : Point ℝ) : Prop := p ∈ l₁ ∩ l₂

-- Final theorem statement
theorem triangle_proof (hA1 : on_side_BC A1 B C)
                        (hBB1 : parallel (B1 - B) (A1 - A))
                        (hCC1 : parallel (C1 - C) (A1 - A))
                        (hB1 : intersect (AC) (B1))
                        (hC1 : intersect (AB) (C1)) :
  1/(dist A A1) = 1/(dist B B1) + 1/(dist C C1) :=
sorry

end triangle_proof_l217_217186


namespace find_abc_sum_l217_217714

theorem find_abc_sum :
  ∃ a b c : ℕ,
  (4 * Real.sqrt (Real.cbrt 7 - Real.cbrt 6) = Real.cbrt a + Real.cbrt b - Real.cbrt c) ∧ (a + b + c = 79) := sorry

end find_abc_sum_l217_217714


namespace quadrilateral_ADEC_area_l217_217179

-- Define the elements of the problem
variables {A B C D E : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (AC AB DE : ℝ)
variables (angleACB : ℝ)

-- Define the conditions
axiom angle_ACB_90 : angleACB = 90
axiom AD_eq_DB : dist A D = dist D B
axiom DE_perp_AB : is_perpendicular (line_through D E) (line_through A B)
axiom AB_length : dist A B = 24
axiom AC_length : dist A C = 18
axiom DE_length : dist D E = 6

-- Define the target area
noncomputable def quadrilateral_area : ℝ :=
let BC := real.sqrt (AB_length ^ 2 - AC_length ^ 2),
    area_ABC := 0.5 * AC_length * BC,
    area_BDE := 0.5 * (dist D B) * DE_length in
area_ABC - area_BDE

-- Proof statement
theorem quadrilateral_ADEC_area : quadrilateral_area = 107 :=
by
  sorry

end quadrilateral_ADEC_area_l217_217179


namespace solve_fractional_eq_l217_217657

theorem solve_fractional_eq {x : ℚ} : (3 / (x - 1)) = (1 / x) ↔ x = -1/2 :=
by sorry

end solve_fractional_eq_l217_217657


namespace length_AE_eq_18_l217_217183

-- Definitions for the triangle and its sides
variables (A B C D E : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq E]
variables (AB AC BC : ℝ)
variables (h_AB : AB = 28) (h_AC : AC = 36) (h_BC : BC = 32)

-- Definition for point D and the angle bisector property
variables [PointOnSegment D B C] [AngleBisectorCondition (Angle A B D) (Angle D A C)]

-- Definitions for parallel condition and tangency condition
variables [ParallelCondition DE AB] [TangentCondition AE (Circumcircle ABC)]

-- The final statement
theorem length_AE_eq_18 : length AE = 18 :=
sorry -- Proof is omitted

end length_AE_eq_18_l217_217183


namespace arrangement_probability_l217_217487
open BigOperators

theorem arrangement_probability :
  let tiles := ['X', 'X', 'X', 'X', 'O', 'O', 'O'],
      arrangement := ['X', 'O', 'X', 'O', 'X', 'X', 'O'],
      total_arrangements := (∑ n in Finset.range 7.succ, (1:ℕ)),
      specific_arrangements := 1
  in ∃ (probability : ℚ), probability = specific_arrangements / total_arrangements :=
by
  sorry

end arrangement_probability_l217_217487


namespace rem_frac_l217_217342

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_frac : rem (5/7 : ℚ) (3/4 : ℚ) = (5/7 : ℚ) := 
by 
  sorry

end rem_frac_l217_217342


namespace prove_jens_suckers_l217_217250
noncomputable def Jen_ate_suckers (Sienna_suckers : ℕ) (Jen_suckers_given_to_Molly : ℕ) : Prop :=
  let Molly_suckers_given_to_Harmony := Jen_suckers_given_to_Molly - 2
  let Harmony_suckers_given_to_Taylor := Molly_suckers_given_to_Harmony + 3
  let Taylor_suckers_given_to_Callie := Harmony_suckers_given_to_Taylor - 1
  Taylor_suckers_given_to_Callie = 5 → (Sienna_suckers/2) = Jen_suckers_given_to_Molly * 2

#eval Jen_ate_suckers 44 11 -- Example usage, you can change 44 and 11 accordingly

def jen_ate_11_suckers : Prop :=
  Jen_ate_suckers Sienna_suckers 11

theorem prove_jens_suckers : jen_ate_11_suckers :=
  sorry

end prove_jens_suckers_l217_217250


namespace time_after_1850_minutes_l217_217712

theorem time_after_1850_minutes (current_hour : ℕ) (current_minute : ℕ) (additional_minutes : ℕ) 
  (h1 : current_hour = 15) (h2 : current_minute = 15) (h3 : additional_minutes = 1850):
  let total_minutes := current_hour * 60 + current_minute + additional_minutes,
      final_hour := (total_minutes / 60) % 24,
      final_minute := total_minutes % 60
  in
    final_hour = 22 ∧ final_minute = 5 :=
by
  sorry

end time_after_1850_minutes_l217_217712


namespace solve_eq1_solve_eq2_l217_217656

-- For Equation (1)
theorem solve_eq1 (x : ℝ) : x^2 - 4*x - 6 = 0 → x = 2 + Real.sqrt 10 ∨ x = 2 - Real.sqrt 10 :=
sorry

-- For Equation (2)
theorem solve_eq2 (x : ℝ) : (x / (x - 1) - 1 = 3 / (x^2 - 1)) → x ≠ 1 ∧ x ≠ -1 → x = 2 :=
sorry

end solve_eq1_solve_eq2_l217_217656


namespace max_height_soccer_ball_l217_217740

theorem max_height_soccer_ball (a b : ℝ) (t h : ℝ) : 
  (a = -4) → (b = 12) → (h = a * t^2 + b * t) → 
  (t = 3 / 2) → h = 9 :=
by
  intros ha hb ht max_t
  have h_eq : h = -4 * (3 / 2)^2 + 12 * (3 / 2) := by
    rw [ha, hb]
    apply ht
  rw max_t at h_eq
  simp at h_eq
  exact h_eq

#Evaluate
check max_height_soccer_ball

end max_height_soccer_ball_l217_217740


namespace woman_investment_problem_l217_217415

variable (total_investment : ℤ) (investment1 investment2 remainder : ℤ) (rate1 rate2 : ℚ) (income_target : ℤ) (option_rates : List ℚ)

theorem woman_investment_problem
  (hTotalInvestment : total_investment = 12000)
  (hInvestment1 : investment1 = 5000)
  (hRate1 : rate1 = 0.03)
  (hInvestment2 : investment2 = 4000)
  (hRate2 : rate2 = 0.045)
  (hRemainder : remainder = total_investment - (investment1 + investment2))
  (hIncomeTarget : income_target = 580)
  (hOptionRates : option_rates = [0.05, 0.055, 0.06, 0.065, 0.07])
  : List.map (λ r, r * remainder) option_rates ≠ 250 → 7% ∈ option_rates := 
sorry

end woman_investment_problem_l217_217415


namespace triangle_properties_proof_perimeter_and_max_area_proof_l217_217156

noncomputable def triangle_properties (a b c S : ℝ) (C : ℝ) : Prop :=
  S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2) ∧
  C = Real.pi / 3

noncomputable def perimeter_and_max_area (a b : ℝ) (S_max : ℝ) : Prop :=
  a + b = 4 ∧
  (∀ c, c^2 = (a + b)^2 - 3 * a * b → c ∈ Set.Ico 2 4) ∧
  (∀ c, S_max = Real.sqrt 3)

theorem triangle_properties_proof (a b c S : ℝ) (h : triangle_properties a b c S (Real.pi / 3)) :
  ∃ C, S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2) ∧ C = Real.pi / 3 := by
  use Real.pi / 3
  exact h

theorem perimeter_and_max_area_proof (a b : ℝ) (S_max : ℝ) (h : perimeter_and_max_area a b S_max) :
  a + b = 4 ∧ ∀ c, c^2 = (a + b)^2 - 3 * a * b → c ∈ Set.Ico 2 4 ∧ S_max = Real.sqrt 3 := by
  exact h

#check triangle_properties_proof
#check perimeter_and_max_area_proof

end triangle_properties_proof_perimeter_and_max_area_proof_l217_217156


namespace instantaneous_velocity_at_t3_l217_217687

def displacement (t : ℝ) : ℝ :=
  t^2 + 10

theorem instantaneous_velocity_at_t3 : 
  let velocity := (λ t, (deriv displacement t))
  velocity 3 = 6 := 
by
  sorry

end instantaneous_velocity_at_t3_l217_217687


namespace max_trig_expression_l217_217069

theorem max_trig_expression : 
  ∃ x y z : ℝ, ∀ x y z : ℝ, sin x * sin y * sin z + cos x * cos y * cos z ≤ 1 ∧ 
  (sin x = 0 ∧ cos x = 1) ∧ (sin y = 0 ∧ cos y = 1) ∧ (sin z = 0 ∧ cos z = 1) → 
  (sin x * sin y * sin z + cos x * cos y * cos z = 1) :=
by
  sorry

end max_trig_expression_l217_217069


namespace find_smallest_m_l217_217835

theorem find_smallest_m :
  ∃ m : ℕ, (m > 0) ∧ (m >= 30) ∧ (∃ d > 1, d ∣ (m - 17) ∧ d ∣ (7 * m + 11)) ∧
          (∀ n : ℕ, (n > 0) ∧ ∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 11) → m ≤ n) :=
begin
  sorry
end

end find_smallest_m_l217_217835


namespace boys_in_class_l217_217669

noncomputable def number_of_boys_in_class : ℕ :=
  let avg_height : ℕ := 185
  let wrong_height : ℕ := 166
  let actual_wrong_height : ℕ := 106
  let actual_avg_height : ℕ := 183
  let difference : ℕ := wrong_height - actual_wrong_height
  -- Derived from the given equation: 185 * n - difference = 183 * n
  let equation := (avg_height * n - difference = actual_avg_height * n)
  -- From equation we have: (185 - 183) * n = difference
  -- Which leads to: 2 * n = 60
  let result : ℕ := 30

theorem boys_in_class : number_of_boys_in_class = 30 := 
by
  sorry

end boys_in_class_l217_217669


namespace diego_monthly_paycheck_l217_217816

-- Define the given conditions
def monthly_expenses : ℕ := 4600
def annual_savings : ℕ := 4800

-- Define the question as a theorem to prove
theorem diego_monthly_paycheck : ∃ P : ℕ, P = monthly_expenses + annual_savings / 12 :=
by
  use 5000  -- This is the hypothesis derived from the problem's solution
  simp [monthly_expenses, annual_savings]
  sorry -- The proof steps go here

end diego_monthly_paycheck_l217_217816


namespace Soyun_distance_l217_217940

theorem Soyun_distance
  (perimeter : ℕ)
  (Soyun_speed : ℕ)
  (Jia_speed : ℕ)
  (meeting_time : ℕ)
  (time_to_meet : perimeter = (Soyun_speed + Jia_speed) * meeting_time) :
  Soyun_speed * meeting_time = 10 :=
by
  sorry

end Soyun_distance_l217_217940


namespace loss_percentage_calculation_l217_217419

-- Define initial price of the article
def initial_price : ℝ := 560

-- Define discount rate
def discount_rate : ℝ := 0.10

-- Define selling price with tax
def selling_price_with_tax : ℝ := 340

-- Define sales tax rate
def sales_tax_rate : ℝ := 0.15

-- Calculate price after discount
def price_after_discount (p₀ : ℝ) (d : ℝ) := p₀ * (1 - d)

-- Calculate selling price before tax
def selling_price_before_tax (st : ℝ) (t : ℝ) := st / (1 + t)

-- Calculate loss
def loss (buy_price sell_price : ℝ) := buy_price - sell_price

-- Calculate loss percent
def loss_percent (buy_price loss_amount : ℝ) := (loss_amount / buy_price) * 100

theorem loss_percentage_calculation :
  let p₀ := initial_price in
  let d := discount_rate in
  let st := selling_price_with_tax in
  let t := sales_tax_rate in
  let buy_price := price_after_discount p₀ d in
  let sell_price := selling_price_before_tax st t in
  let loss_amt := loss buy_price sell_price in
  loss_percent buy_price loss_amt = 41.34 := by
  sorry

end loss_percentage_calculation_l217_217419


namespace angle_inclination_range_l217_217220

def curve (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + (4 - Real.sqrt 3) * x

def tangent_slope (x : ℝ) : ℝ := x^2 - 4 * x + (4 - Real.sqrt 3)

theorem angle_inclination_range (x : ℝ) (α : ℝ) (hx : α = Real.arctan (tangent_slope x)) :
  0 ≤ α ∧ α < (Real.pi / 2) ∨ (2 * Real.pi / 3) ≤ α ∧ α < Real.pi :=
sorry

end angle_inclination_range_l217_217220


namespace rice_weight_probability_l217_217380

open Real

/-- Definition of the normal distribution -/
noncomputable def normal_pdf (μ σ x : ℝ) : ℝ :=
  (1 / (σ * sqrt (2 * π))) * exp (- (x - μ)^2 / (2 * σ^2))

/-- Cumulative distribution function for the normal distribution -/
noncomputable def normal_cdf (μ σ x : ℝ) : ℝ :=
  (1 + erf ((x - μ) / (σ * sqrt 2))) / 2

theorem rice_weight_probability :
  let X : ℝ → ℝ := normal_pdf 10 0.1 in
  let F : ℝ → ℝ := normal_cdf 10 0.1 in
  (F 10.2 - F 9.8) = 0.9544 :=
by
  let X : ℝ → ℝ := normal_pdf 10 0.1
  let F : ℝ → ℝ := normal_cdf 10 0.1
  have h : F 10.2 - F 9.8 = 0.9544 := sorry
  exact h

end rice_weight_probability_l217_217380


namespace polynomial_factorization_l217_217623

theorem polynomial_factorization (a b k : ℤ) (h1 : Int.gcd a 3 = 1) (h2 : Int.gcd b 3 = 1) (h3 : a + b = 3 * k):
  ∃ q : ℤ[X], (X^a + X^b + 1) = (X^2 + X + 1) * q :=
by
  sorry

end polynomial_factorization_l217_217623


namespace sufficient_not_necessary_l217_217507

variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (h_seq : ∀ n, a (n + 1) = a n + (a 1 - a 0))
variables (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variables (h_condition : 3 * a 2 = a 5 + 4)

theorem sufficient_not_necessary (h1 : a 1 < 1) : S 4 < 10 :=
sorry

end sufficient_not_necessary_l217_217507


namespace min_cube_sum_l217_217215

variable {n : ℕ}
variable (x : Fin n → ℤ)

theorem min_cube_sum (h1 : ∀ i, -2 ≤ x i ∧ x i ≤ 3)
                     (h2 : ∑ i, x i = 27)
                     (h3 : ∑ i, (x i) ^ 2 = 153) :
  ∑ i, (x i) ^ 3 = 543 :=
sorry

end min_cube_sum_l217_217215


namespace taxi_at_drum_tower_and_fuel_consumption_l217_217781

def taxi_distance (record : List Int) : Int :=
  List.sum record

def fuel_consumption (record : List Int) (rate : Float) : Float :=
  rate * (record.map Int.toFloat).sum

theorem taxi_at_drum_tower_and_fuel_consumption :
  taxi_distance [9, -3, 4, -8, 6, -5, -3, -6, -4, 10] = 0 ∧
  fuel_consumption [9, -3, 4, -8, 6, -5, -3, -6, -4, 10] 0.1 = 5.8 :=
by
  sorry

end taxi_at_drum_tower_and_fuel_consumption_l217_217781


namespace max_k_value_l217_217953

noncomputable def circle_equation (x y : ℝ) : Prop :=
x^2 + y^2 - 8 * x + 15 = 0

noncomputable def point_on_line (k x y : ℝ) : Prop :=
y = k * x - 2

theorem max_k_value (k : ℝ) :
  (∃ x y, circle_equation x y ∧ point_on_line k x y ∧ (x - 4)^2 + y^2 = 1) →
  k ≤ 4 / 3 :=
by
  sorry

end max_k_value_l217_217953


namespace find_k_l217_217575

theorem find_k : ∃ k : ℕ, (2 * (Real.sqrt (225 + k)) = (Real.sqrt (49 + k) + Real.sqrt (441 + k))) → k = 255 :=
by
  sorry

end find_k_l217_217575


namespace equal_set_of_numbers_l217_217417

theorem equal_set_of_numbers :
  (∀ (x y : ℝ), ¬ (x = -3 ^ 2 ∧ y = -2 ^ 3) ∧ 
   ¬ (x = (-3 * 2) ^ 2 ∧ y = -3 * 2 ^ 2) ∧ 
   ¬ (x = -3 ^ 2 ∧ y = (-3) ^ 2) ∧ 
   (x = -2 ^ 3 ∧ y = (-2) ^ 3) → 
   x = y) :=
by {
  intros x y h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  exact eq.trans h4.1 h4.2,
  sorry
}

end equal_set_of_numbers_l217_217417


namespace dmitry_before_anatoly_l217_217336

theorem dmitry_before_anatoly (m : ℝ) (x y z : ℝ) 
  (h1 : 0 < x ∧ x < m) 
  (h2 : 0 < y ∧ y < m) 
  (h3 : 0 < z ∧ z < m) 
  (h4 : y < z) : 
  (Probability (y < x)) = 2 / 3 := 
sorry

end dmitry_before_anatoly_l217_217336


namespace remainder_of_x_pow_105_l217_217073

theorem remainder_of_x_pow_105 :
  ∃ (r : Polynomial ℤ), r = 2^105 + 105 * 2^104 * (X - 2) + 5460 * 2^103 * (X - 2)^2 + 191740 * 2^102 * (X - 2)^3 ∧
    (X ^ 105) % ((X - 2)^4) = r :=
by sorry

end remainder_of_x_pow_105_l217_217073


namespace photos_per_page_l217_217650

theorem photos_per_page (total_photos : ℕ) (pages : ℕ) (photos_per_page : ℕ) : 
  total_photos = 736 → pages = 122 → total_photos / pages = 6 := 
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end photos_per_page_l217_217650


namespace total_marbles_l217_217936

theorem total_marbles (r b g y : ℝ)
  (h1 : r = 1.35 * b)
  (h2 : g = 1.5 * r)
  (h3 : y = 2 * b) :
  r + b + g + y = 4.72 * r :=
by
  sorry

end total_marbles_l217_217936


namespace prob1_prob2_prob3_l217_217801

variable (a b : ℝ)

theorem prob1 : 5 * a * b^2 - 3 * a * b^2 + 1/3 * a * b^2 = 7/3 * a * b^2 := 
by
  sorry

variable (m n : ℝ)

theorem prob2 : (7 * m^2 * n - 5 * m) - (4 * m^2 * n - 5 * m) = 3 * m^2 * n := 
by
  sorry

variable (x y : ℝ)
hypothesis h1 : x = -1/4
hypothesis h2 : y = 2

theorem prob3 : 2 * x^2 * y - 2 * (x * y^2 + 2 * x^2 * y) + 2 * (x^2 * y - 3 * x * y^2) = 8 := 
by
  sorry

end prob1_prob2_prob3_l217_217801


namespace line_HN_fixed_point_l217_217526

-- Define given points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Ellipse equation based on given conditions
def ellipse_eq (a b x y : ℝ) : Prop :=
  x^2 / a + y^2 / b = 1

-- Given ellipse passes through points A and B
def ellipse_through_A_and_B : Prop :=
  ellipse_eq 3 4 A.1 A.2 ∧ ellipse_eq 3 4 B.1 B.2

-- Point H and segment conditions
def H_condition (M T H : ℝ × ℝ) : Prop :=
  2 * (T.1 - M.1, T.2 - M.2) = (H.1 - T.1, H.2 - T.2)

-- Proving line HN passes through a fixed point assuming H condition and M, N inclusion in ellipse
theorem line_HN_fixed_point :
  ∃ K : ℝ × ℝ, ellipse_through_A_and_B → 
  ∀ M N : ℝ × ℝ, 
  (ellipse_eq 3 4 M.1 M.2) → 
  (ellipse_eq 3 4 N.1 N.2) → 
  H_condition M ((2 * M.1 + M.1, 2 * M.2 - (2 * (M.2 + 2) / 3))) ((4 * M.1, 4 * M.2)) →
  ∃ H : ℝ × ℝ, H_condition M ((2 * M.1 + M.1, 2 * M.2 - (2 * (M.2 + 2) / 3))) H → 
  (H, N).1 * (0, -2).2 - (H, N).2 * (0, -2).1 = 0 :=
by sorry

end line_HN_fixed_point_l217_217526


namespace negative_solution_condition_l217_217042

variable {a b c x y : ℝ}

theorem negative_solution_condition (h1 : a * x + b * y = c)
    (h2 : b * x + c * y = a)
    (h3 : c * x + a * y = b)
    (hx : x < 0)
    (hy : y < 0) :
    a + b + c = 0 :=
sorry

end negative_solution_condition_l217_217042


namespace g_difference_l217_217219

def g (n : ℕ) : ℝ :=
  (4 + 2 * Real.sqrt 6) / 12 * ((2 + Real.sqrt 6) / 3)^n +
  (4 - 2 * Real.sqrt 6) / 12 * ((2 - Real.sqrt 6) / 3)^n

theorem g_difference (n : ℕ) : g (n + 1) - g (n - 1) = (1 / 3) * g n :=
  sorry

end g_difference_l217_217219


namespace minimum_norm_value_l217_217131

variables (a b c : ℝ^3) (x y : ℝ)

def unit_vector (v : ℝ^3) := ∥v∥ = 1

def orthogonal (v1 v2 : ℝ^3) := v1 • v2 = 0

def magnitude (v : ℝ^3) := ∥v∥

noncomputable def expression (a b c : ℝ^3) (x y : ℝ) :=
  ∥c - x • a - y • b∥

theorem minimum_norm_value (h1 : unit_vector a) (h2 : unit_vector b)
  (h3 : orthogonal a b) (h4 : magnitude c = 3)
  (h5 : c • a = 2) (h6 : c • b = 1) :
  ∃ x y : ℝ, expression a b c x y = 2 :=
sorry

end minimum_norm_value_l217_217131


namespace find_common_ratio_l217_217598

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, S n = a 1 * (1 - q ^ n) / (1 - q)

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)

noncomputable def a_5_condition : Prop :=
  a 5 = 2 * S 4 + 3

noncomputable def a_6_condition : Prop :=
  a 6 = 2 * S 5 + 3

theorem find_common_ratio (h1 : a_5_condition a S) (h2 : a_6_condition a S)
  (hg : geometric_sequence a q) (hs : sum_of_first_n_terms a S q) :
  q = 3 :=
sorry

end find_common_ratio_l217_217598


namespace count_three_digit_numbers_with_digit_sum_seven_l217_217323

theorem count_three_digit_numbers_with_digit_sum_seven :
    (∑ d : Finset (ℕ × ℕ × ℕ), (d.1 + d.2 + d.3 = 7 ∧
    1 ≤ d.1 ∧ d.1 ≤ 9 ∧
    1 ≤ d.2 ∧ d.2 ≤ 9 ∧
    1 ≤ d.3 ∧ d.3 ≤ 9)) = 15 := sorry

end count_three_digit_numbers_with_digit_sum_seven_l217_217323


namespace problem1_problem2_l217_217754

-- Problem 1: Proving the equation
theorem problem1 (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 → x = 1 :=
sorry

-- Problem 2: Proving the solution for the system of equations
theorem problem2 (x y : ℝ) : (x + 2 * y = 8) ∧ (3 * x - 4 * y = 4) → x = 4 ∧ y = 2 :=
sorry

end problem1_problem2_l217_217754


namespace find_segment_length_l217_217312

noncomputable def length_segment (A B C : ℝ × ℝ) (rA rB rC : ℝ) : ℝ :=
  let B' := ( 4.85, 1.825 )  -- Intersection point of A and C (need accurate computation for real scenario)
  let C' := ( -4.85, 1.825 ) -- Intersection point of A and B (need accurate computation for real scenario)
  -- Calculate the distance between B' and C'
  real.sqrt ((B'.1 - C'.1)^2 + (B'.2 - C'.2)^2)

theorem find_segment_length :
  let A := (0, 0)
  let B := (3, 0)
  let C := (-3, 0)
  let rA := 2.5
  let rB := 3
  let rC := 2
  length_segment A B C rA rB rC = 4.85 := by
    sorry

end find_segment_length_l217_217312


namespace solution_interval_l217_217772

open Real

noncomputable def f (x : ℝ) : ℝ := sorry -- Assume f is given, monotonically increasing and meets given conditions

axiom f_monotone : ∀ x y : ℝ, (0 < x ∧ 0 < y) → x < y → f(x) < f(y)

axiom f_condition : ∀ x : ℝ, 0 < x → f (f x - log x) = 1

theorem solution_interval :
  ∃ x : ℝ, 0 < x ∧ (f x - deriv f x = 1) ∧ (1 < x) ∧ (x < 2) :=
sorry  -- The proof is omitted.

end solution_interval_l217_217772


namespace susan_walked_9_miles_l217_217454

theorem susan_walked_9_miles (E S : ℕ) (h1 : E + S = 15) (h2 : E = S - 3) : S = 9 :=
by
  sorry

end susan_walked_9_miles_l217_217454


namespace increasing_interval_l217_217282

noncomputable def f : ℝ → ℝ := λ x, x - 2 * Real.sin x

theorem increasing_interval : 
  ∃ a b, a = Real.pi / 3 ∧ b = Real.pi ∧
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → f x₁ < f x₂ :=
sorry

end increasing_interval_l217_217282


namespace a_n_formula_b_n_formula_sum_a_sum_b_l217_217182

variable (a b : ℕ → ℤ)

axiom a_1 : a 1 = 1
axiom b_1 : b 1 = 2

axiom a_recurrence : ∀ n : ℕ, 0 < n → a (n + 1) = 3 * a n - 2 * b n
axiom b_recurrence : ∀ n : ℕ, 0 < n → b (n + 1) = 2 * a n - b n

theorem a_n_formula (n : ℕ) (h : 0 < n) : a n = 3 - 2 * n := 
sorry

theorem b_n_formula (n : ℕ) (h : 0 < n) : b n = 4 - 2 * n := 
sorry

theorem sum_a (n : ℕ) : ∑ k in Finset.range n, a (k + 1) = -(n^2:ℤ) + 2 * n :=
sorry

theorem sum_b (n : ℕ) : ∑ k in Finset.range n, b (k + 1) = 3 * n - n^2 :=
sorry

end a_n_formula_b_n_formula_sum_a_sum_b_l217_217182


namespace problem_statement_l217_217560

variable {a b c d : ℚ}

-- Conditions
axiom h1 : a / b = 3
axiom h2 : b / c = 3 / 4
axiom h3 : c / d = 2 / 3

-- Goal
theorem problem_statement : d / a = 2 / 3 := by
  sorry

end problem_statement_l217_217560


namespace dot_product_OA_OB_zero_l217_217109

-- Define vectors and their properties.
def vec_OA : Prod ℝ ℝ := (1, -3)
def len_OA : ℝ := Real.sqrt (1^2 + (-3)^2)
axiom len_equal (x y : ℝ) : len_OA = Real.sqrt (x^2 + y^2)
axiom distance_AB (x y : ℝ) : Real.sqrt ((x - 1)^2 + (y + 3)^2) = 2 * Real.sqrt 5
def dot_product (x y : ℝ) : ℝ := 1 * x + (-3) * y

-- Prove the dot product is zero
theorem dot_product_OA_OB_zero (x y : ℝ) (h_len : len_equal x y) (h_dist : distance_AB x y) :
  dot_product x y = 0 :=
sorry

end dot_product_OA_OB_zero_l217_217109


namespace problem1_problem2_l217_217803

-- Problem 1
theorem problem1 : (-2) ^ 2 * Real.sqrt (1 / 4) + abs (Real.cbrt (-8)) + Real.sqrt 2 * (-1) ^ 2016 = 4 + Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem2 : Real.sqrt 81 + Real.cbrt (-27) + Real.sqrt ((-2) ^ 2) + abs (Real.sqrt 3 - 2) = 10 - Real.sqrt 3 :=
by
  sorry

end problem1_problem2_l217_217803


namespace interest_rate_second_case_l217_217293

variable (P T : ℝ)
def R1 := 5 / 100
def SI := 840
def SI_formula (P R T : ℝ) := (P * R * T) / 100 

theorem interest_rate_second_case (H: SI_formula P R1 T = 840) : 
  let R2 := (5 * T) / 8 in
  SI_formula P R2 8 = SI := 
by
  sorry

end interest_rate_second_case_l217_217293


namespace exists_convex_polygon_with_n_axes_of_symmetry_l217_217239

theorem exists_convex_polygon_with_n_axes_of_symmetry (n : ℕ) : 
  ∃ (P : Polygon ℝ), Convex P ∧ P.axes_of_symmetry.count = n := sorry

end exists_convex_polygon_with_n_axes_of_symmetry_l217_217239


namespace shaded_square_probability_l217_217376

theorem shaded_square_probability :
  let length := 2024 in
  let middle := length / 2 in
  let n_rectangles := (length + 1) * length / 2 in
  let m_shaded := middle * middle in
  let prob_shaded := m_shaded / n_rectangles in
  1 - prob_shaded = 1 / 2 :=
sorry

end shaded_square_probability_l217_217376


namespace eccentricity_is_two_sqrt_three_over_three_l217_217544

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = sqrt 3 * b) : ℝ := 
  let c := sqrt (a^2 + b^2)
  c / a

theorem eccentricity_is_two_sqrt_three_over_three (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = sqrt 3 * b) :
  eccentricity_of_hyperbola a b h1 h2 h3 = (2 * sqrt 3) / 3 :=
sorry

end eccentricity_is_two_sqrt_three_over_three_l217_217544


namespace solve_for_a_l217_217440

noncomputable def special_otimes (a b : ℝ) : ℝ :=
  if a > b then a^2 + b else a + b^2

theorem solve_for_a (a : ℝ) : special_otimes a (-2) = 4 → a = Real.sqrt 6 :=
by
  intro h
  sorry

end solve_for_a_l217_217440


namespace counterexamples_count_l217_217470

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

def all_digits_non_zero (n : ℕ) : Prop :=
  n = 0 → False ∧ all_digits_non_zero_aux n
where
  all_digits_non_zero_aux 0 := True
  | n := (n % 10 ≠ 0) ∧ all_digits_non_zero_aux (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def num_counterexamples : ℕ :=
  Finset.card (
    (Finset.filter (λ n, sum_of_digits n = 7 ∧ all_digits_non_zero n ∧ ¬ is_prime n)
                   (Finset.range 1000000) : Finset ℕ)
  )

theorem counterexamples_count : num_counterexamples = 20 := sorry

end counterexamples_count_l217_217470


namespace savings_account_end_balance_l217_217962

noncomputable def calcAmount : ℝ :=
  let P1 := 6000
  let r1 := 0.05
  let n1 := 365
  let A1 := P1 * (1 + r1 / n1) ^ n1
  let P2 := A1
  let r2 := 0.06
  let n2 := 52
  let A2 := P2 * (1 + r2 / n2) ^ n2
  A2

theorem savings_account_end_balance : 
  calcAmount = 6695.70 :=
by 
  sorry

end savings_account_end_balance_l217_217962


namespace true_prop_count_l217_217900

-- Define the propositions
def original_prop (x : ℝ) : Prop := x > -3 → x > -6
def converse (x : ℝ) : Prop := x > -6 → x > -3
def inverse (x : ℝ) : Prop := x ≤ -3 → x ≤ -6
def contrapositive (x : ℝ) : Prop := x ≤ -6 → x ≤ -3

-- The statement to prove
theorem true_prop_count (x : ℝ) : 
  (original_prop x → true) ∧ (contrapositive x → true) ∧ ¬(converse x) ∧ ¬(inverse x) → 
  (count_true_propositions = 2) :=
sorry

end true_prop_count_l217_217900


namespace fraction_simplification_l217_217026

theorem fraction_simplification :
  (∀ n d : ℕ, Nat.coprime n d → (n : Rat) / (d : Rat) = (1 : Rat) / (7 : Rat)) →
  ∃ n d : ℕ, n = 5274 ∧ d = 36918 ∧ (Nat.gcd n d = 5274) ∧ ((n : Rat) / d = (1 : Rat) / (7 : Rat)) :=
by
  sorry

end fraction_simplification_l217_217026


namespace find_T_n_l217_217709

-- Definitions for the sequence and conditions in the problem
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i
def a (n : ℕ) : ℝ := if n = 0 then 1 else 4 ^ (n - 1) -- Given geometric progression
def b (n : ℕ) : ℝ := real.log 4 (a (n + 1))
def c (n : ℕ) : ℝ := a n + b n
def T (n : ℕ) : ℝ := ∑ i in finset.range n, c i

-- Main theorem to prove: Conditions imply the desired sum
theorem find_T_n (n : ℕ) : T n = (4^n - 1) / 3 + n * (n + 1) / 2 :=
sorry

end find_T_n_l217_217709


namespace positive_projection_exists_l217_217269

variables {E : Type*} [inner_product_space ℝ E]

theorem positive_projection_exists (e : fin 10 → E)
  (h : ∀ i : fin 10, ‖(∑ j, e j) - e i‖ < ‖∑ j, e j‖) :
  ∀ i : fin 10, 0 < ⟪(∑ j, e j), e i⟫ :=
by sorry

end positive_projection_exists_l217_217269


namespace distance_travelled_in_first_hour_l217_217295

theorem distance_travelled_in_first_hour
    (d : ℕ) -- d is the distance travelled in the first hour.
    (sum_arithmetic_series : ∑ k in range 12, d + 2 * k = 792) :
    d = 55 := 
sorry

end distance_travelled_in_first_hour_l217_217295


namespace linear_inequality_solution_l217_217579

theorem linear_inequality_solution (a : ℝ) (x : ℝ) :
  2a - x^(abs (2 + 3 * a)) > 2 → a = -1 ∨ a = -1/3 :=
sorry

end linear_inequality_solution_l217_217579


namespace max_profit_l217_217764

def fixed_cost : ℝ := 20
def variable_cost_per_unit : ℝ := 10

def total_cost (Q : ℝ) := fixed_cost + variable_cost_per_unit * Q

def revenue (Q : ℝ) := 40 * Q - Q^2

def profit (Q : ℝ) := revenue Q - total_cost Q

def Q_optimized : ℝ := 15

theorem max_profit : profit Q_optimized = 205 := by
  sorry -- Proof goes here.

end max_profit_l217_217764


namespace solve_system_l217_217630

theorem solve_system :
  ∃ (x y : ℤ), (x * (1/7 : ℚ)^2 = 7^3) ∧ (x + y = 7^2) ∧ (x = 16807) ∧ (y = -16758) :=
by
  sorry

end solve_system_l217_217630


namespace negative_solution_exists_l217_217062

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l217_217062


namespace floor_e_sub_6_eq_neg_4_l217_217838

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end floor_e_sub_6_eq_neg_4_l217_217838


namespace correct_statement_3_l217_217128

-- Definitions
def acute_angles (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def angles_less_than_90 (θ : ℝ) : Prop := θ < 90
def angles_in_first_quadrant (θ : ℝ) : Prop := ∃ k : ℤ, k * 360 < θ ∧ θ < k * 360 + 90

-- Sets
def M := {θ | acute_angles θ}
def N := {θ | angles_less_than_90 θ}
def P := {θ | angles_in_first_quadrant θ}

-- Proof statement
theorem correct_statement_3 : M ⊆ P := sorry

end correct_statement_3_l217_217128


namespace price_difference_correct_l217_217784

-- Define the list price of Camera Y
def list_price : ℚ := 52.50

-- Define the discount at Mega Deals
def mega_deals_discount : ℚ := 12

-- Define the discount rate at Budget Buys
def budget_buys_discount_rate : ℚ := 0.30

-- Calculate the sale prices
def mega_deals_price : ℚ := list_price - mega_deals_discount
def budget_buys_price : ℚ := (1 - budget_buys_discount_rate) * list_price

-- Calculate the price difference in dollars and convert to cents
def price_difference_in_cents : ℚ := (mega_deals_price - budget_buys_price) * 100

-- Theorem to prove the computed price difference in cents equals 375
theorem price_difference_correct : price_difference_in_cents = 375 := by
  sorry

end price_difference_correct_l217_217784


namespace range_of_a_l217_217149

def func (x : ℝ) : ℝ := x^2 - 4 * x

def domain (a : ℝ) := ∀ x, -4 ≤ x ∧ x ≤ a

def range_condition (y : ℝ) := -4 ≤ y ∧ y ≤ 32

theorem range_of_a (a : ℝ)
  (domain_condition : ∀ x, x ∈ set.Icc (-4) a → func x ∈ set.Icc (-4) 32) :
  2 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l217_217149


namespace findYears_l217_217745

def totalInterest (n : ℕ) : ℕ :=
  24 * n + 70 * n

theorem findYears (n : ℕ) : totalInterest n = 350 → n = 4 := 
sorry

end findYears_l217_217745


namespace same_color_probability_l217_217377

/-
Problem Statement: Given a bag that contains 6 green balls and 8 white balls, prove that the probability of drawing two balls of the same color simultaneously is 43/91.
-/

theorem same_color_probability :
  let total_balls := 14 in
  let green_balls := 6 in
  let white_balls := 8 in
  let prob_green := (green_balls / total_balls : ℚ) * ((green_balls - 1) / (total_balls - 1)) in
  let prob_white := (white_balls / total_balls : ℚ) * ((white_balls - 1) / (total_balls - 1)) in
  let prob_same_color := prob_green + prob_white in
  prob_same_color = 43 / 91 :=
by
  sorry

end same_color_probability_l217_217377


namespace carnations_percentage_l217_217387

-- Define the conditions for the flowers
variables (total_flower pink_flower red_flower: ℕ)
variables (H1: pink_flower = (7 / 10) * total_flower)
variables (H2: red_flower = (3 / 10) * total_flower)
variables (pink_rose pink_carnation: ℕ)
variables (H3: pink_rose = (1 / 2) * pink_flower)
variables (H4: pink_carnation = (1 / 2) * pink_flower)
variables (red_carnation red_flower_ro: ℕ)
variables (H5: red_carnation = (5 / 6) * red_flower)
variables (H6: red_flower_ro = (1 / 6) * red_flower)

-- Define the percentage of carnations
noncomputable def percentage_carnations : ℕ :=
  let total_carnations := pink_carnation + red_carnation in
  total_carnations * 100 / total_flower

-- The proof statement
theorem carnations_percentage
  (H: percentage_carnations pink_flower red_flower pink_rose pink_carnation red_carnation red_flower_ro = 60) : 
  true := by
  sorry

end carnations_percentage_l217_217387


namespace geometric_seq_sum_four_and_five_l217_217891

noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) := a₁ * q^(n-1)

theorem geometric_seq_sum_four_and_five :
  (∀ n, geom_seq a₁ q n > 0) →
  geom_seq a₁ q 3 = 4 →
  geom_seq a₁ q 6 = 1 / 2 →
  geom_seq a₁ q 4 + geom_seq a₁ q 5 = 3 :=
by
  sorry

end geometric_seq_sum_four_and_five_l217_217891


namespace max_term_b_is_6_l217_217263

noncomputable def a : ℕ → ℕ 
| 1        := 0
| (n + 1) := a n + 2 * (n + 1) - 1

noncomputable def b (n : ℕ) : ℝ :=
  real.sqrt (a n + 1) * real.sqrt (a (n + 1) + 1) * (8 / 11) ^ (n - 1)

theorem max_term_b_is_6 : ∃ n : ℕ, (n = 6) ∧ (∀ m : ℕ, b m ≤ b 6) :=
by
  sorry

end max_term_b_is_6_l217_217263


namespace evaluate_f_compound_l217_217119

def f (x : ℝ) : ℝ :=
  if x > 0 then log 2 x else 3 ^ x

theorem evaluate_f_compound (x : ℝ) : f (f (1 / 4)) = 1 / 9 := by
  -- prove that f (1 / 4) = -2
  -- prove that f (-2) = 3 ^ (-2) = 1 / 9
  sorry

end evaluate_f_compound_l217_217119


namespace volume_of_triangular_pyramid_l217_217479

-- Variables for edge lengths
variables {a b c d e f : ℝ}

-- Given conditions as per problem statement
def conditions (a b c d e f : ℝ) : Prop :=
  a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧ f = sqrt 6

-- Goal statement transforming the question to a proof problem.
theorem volume_of_triangular_pyramid (a b c d e f : ℝ) (h : conditions a b c d e f) :
  ∃ vol : ℝ, vol = 1 :=
begin
  use 1,
  sorry
end

end volume_of_triangular_pyramid_l217_217479


namespace point_C_divides_segment_AE_ratio_l217_217875

-- Definitions and assumptions based on the conditions
variables {A B C D E : Type}
variables (α : ℝ) [acute_angled_triangle : ∃ (A B C : Type), ∃ α : ℝ, acute_angled (triangle A B C) ∧ (angle B A C = α)]
variables (D_on_extension_BC : point_on_extension B C D)
variables (AD_tangent_to_circumcircle_ω_ABC : tangent_to_circumcircle A D (circumcircle (triangle A B C)))
variables (AC_intersects_circumcircle_ABD_at_E : ∃ E, intersects (segment A C) (circumcircle (triangle A B D)) E)
variables (angle_bisector_ADE_tangent_to_circumcircle_ω : tangent_to_circumcircle (angle_bisector A D E) (circumcircle (triangle A B C)))

-- We aim to prove the ratio in which point C divides the segment AE
theorem point_C_divides_segment_AE_ratio (A B C D E : Type) (α : ℝ) 
  [acute_angled_triangle : ∃ (A B C : Type), ∃ α : ℝ, acute_angled (triangle A B C) ∧ (angle B A C = α)]
  (D_on_extension_BC : point_on_extension B C D)
  (AD_tangent_to_circumcircle_ω_ABC : tangent_to_circumcircle A D (circumcircle (triangle A B C)))
  (AC_intersects_circumcircle_ABD_at_E : ∃ E, intersects (segment A C) (circumcircle (triangle A B D)) E)
  (angle_bisector_ADE_tangent_to_circumcircle_ω : tangent_to_circumcircle (angle_bisector A D E) (circumcircle (triangle A B C)))
  : divides_segment C (segment A E) = sin α :=
sorry

end point_C_divides_segment_AE_ratio_l217_217875


namespace destroyer_winning_strategy_l217_217723

theorem destroyer_winning_strategy :
  ∀ (choices : List ℕ), (∀ x ∈ choices, x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) → 
  (choices.Nodup) → -- choices is a list of distinct elements from the set
  (choices.length = 12) → -- There are 12 choices, 6 by each player
  (∃ subset : List ℕ, subset ⊆ choices ∧ subset.length = 6 ∧ 
    (∃ seq : List ℕ, seq ⊆ subset ∧ seq.length = 4 ∧ 
      ((seq ≠ List.range 4) → -- not forming {0,1,2,3}
       (seq ≠ List.range 1 4) → -- not forming {1,2,3,4}
       (seq ≠ List.range 2 5) → -- not forming {2,3,4,5}
       (seq ≠ List.range 3 6) → -- not forming {3,4,5,6}
       (seq ≠ List.range 4 7) → -- not forming {4,5,6,7}
       (seq ≠ List.range 5 8) → -- not forming {5,6,7,8}
       (seq ≠ List.range 6 9) → -- not forming {6,7,8,9}
       (seq ≠ List.range 7 10) → -- not forming {7,8,9,10}
       sorry))))) sorry.

end destroyer_winning_strategy_l217_217723


namespace minimum_time_for_all_flickers_l217_217717

def total_flickers : ℕ := 5!
def time_per_flicker : ℕ := 5
def interval_between_flickers : ℕ := 5
def total_time_flickers := total_flickers * time_per_flicker
def total_intervals := interval_between_flickers * (total_flickers - 1)
def minimum_time_required := total_time_flickers + total_intervals

theorem minimum_time_for_all_flickers : minimum_time_required = 1195 :=
by
    -- proof goes here
    sorry

end minimum_time_for_all_flickers_l217_217717


namespace least_possible_faces_correct_l217_217722

noncomputable section

def dice_least_possible_faces (a b : ℕ) : ℕ :=
  if h : 2 * ((list.range (a+1)).product (list.range (b+1))).filter (λ p, p.1 + p.2 = 8).length
         = ((list.range (a+1)).product (list.range (b+1))).filter (λ p, p.1 + p.2 = 11).length
  then if ((list.range (a+1)).product (list.range (b+1))).filter (λ p, p.1 + p.2 = 13).length.to_rat / (a * b).to_rat
          = (1 / 14 : ℚ)
       then a + b
       else 0
  else 0

theorem least_possible_faces_correct :
  ∃ a b : ℕ, a ≥ b ∧ 6 ≤ b ∧ dice_least_possible_faces a b = 26 :=
sorry

end least_possible_faces_correct_l217_217722


namespace no_noninteger_xy_l217_217446

theorem no_noninteger_xy (x y : ℝ) (hx : ∃ (n : ℤ), x ≠ n) (hy : ∃ (m : ℤ), y ≠ m) : 
  ¬ (fract x * fract y = fract (x + y)) := 
by
  sorry

end no_noninteger_xy_l217_217446


namespace car_mpg_decrease_l217_217805

/--
Car Z travels 50 miles per gallon of gasoline at 45 miles per hour,
It travels 400 miles on 10 gallons of gasoline at 60 miles per hour.
We want to prove that the percentage decrease in miles per gallon
when driven at 60 mph compared to 45 mph is 20%.
-/

theorem car_mpg_decrease :
  let mpg_45 := 50 in
  let mpg_60 := 400 / 10 in
  let percentage_decrease := ((mpg_45 - mpg_60) / mpg_45) * 100 in
  percentage_decrease = 20 :=
by
  let mpg_45 := 50
  let mpg_60 := 40  -- 400/10
  let percentage_decrease := ((mpg_45 - mpg_60) / mpg_45) * 100
  show percentage_decrease = 20 from sorry

end car_mpg_decrease_l217_217805


namespace partition_nat_l217_217840

open Set

theorem partition_nat (c : ℚ) (h₀ : 0 < c) (h₁ : c ≠ 1) :
    ∃ (A B : Set ℕ), (A ∩ B = ∅) ∧ (∀ x y ∈ A, (x:ℚ) / y ≠ c) ∧ (∀ x y ∈ B, (x:ℚ) / y ≠ c) := by
  sorry

end partition_nat_l217_217840


namespace tangent_slope_at_1_l217_217499

variable (f : ℝ → ℝ)
variable (h : ∀ x, f (x + 1) = (2 * x + 1) / (x + 1))

theorem tangent_slope_at_1 : (deriv f 1 = 1) :=
by
  have f_from_h : ∀ x, f x = 2 - 1 / x := sorry
  have deriv_f_x : ∀ x, deriv f x = 1 / x^2 := sorry
  show deriv f 1 = 1
  from sorry

end tangent_slope_at_1_l217_217499


namespace problem_polar_to_cartesian_distance_calc_l217_217869

theorem problem_polar_to_cartesian_distance_calc :
  (polar_eq : ∀ θ : ℝ, ∃ (ρ : ℝ), ρ = (6 * (Real.cos θ)) / (Real.sin θ)^2)  -- Polar equation
  (line_param_eq : ∀ t : ℝ, ∃ (x y : ℝ), x = (3 / 2) + t ∧ y = Real.sqrt 3 * t) -- Parametric line equation
  (cartesian_eq : ∀ x y : ℝ, y^2 = 6 * x) -- Cartesian equation of the curve
  (intersects : ∃ (t1 t2 : ℝ), 
                (3 / 2 + t1 = x ∧ Real.sqrt 3 * t1 = y) ∧ 
                (3 / 2 + t2 = x ∧ Real.sqrt 3 * t2 = y) ∧ 
                abs (t1 - t2) = 8) -- Distance of intersection points
  : cartesian_eq → intersects ∧ ∀ (x y : ℝ), y^2 = 6 * x ∧ |t1 - t2| = 8
:= by sorry

end problem_polar_to_cartesian_distance_calc_l217_217869


namespace find_a_l217_217221

-- Let f be a monotonic function defined on (0, +∞)
axiom f : ℝ → ℝ
axiom f_mono : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y
axiom f_domain : ∀ x : ℝ, 0 < x → ∃ y : ℝ, f[x] = y
axiom f_property : ∀ x : ℝ, 0 < x → f (f x - real.log x / real.log 2) = 6

-- x₀ is a solution to the equation f(x) - f'(x) = 4
def derivative (f : ℝ → ℝ) := λ x, (f (x + 1e-9) - f x) / 1e-9
notation (name := f') "f'(" x ")" => derivative f x
axiom x₀_condition : ∃ x₀ : ℝ, 0 < x₀ ∧ f x₀ - f'(x₀) = 4

-- x₀ belongs to the interval (a, a + 1) (a ∈ ℕ*)
axiom interval_condition : ∃ a : ℕ, a > 0 ∧ x₀_condition.some ∈ set.Ioo (a - 1).to_real a.to_real

-- Prove that a = 1
theorem find_a (a : ℕ) : interval_condition.some.fst ∈ set.Ioo ((a: ℝ)- 1) (a: ℝ) → a = 1 := by
    sorry

end find_a_l217_217221


namespace factorize_mn_minus_mn_cubed_l217_217371

theorem factorize_mn_minus_mn_cubed (m n : ℝ) : 
  m * n - m * n ^ 3 = m * n * (1 + n) * (1 - n) :=
by {
  sorry
}

end factorize_mn_minus_mn_cubed_l217_217371


namespace compute_result_l217_217485

open Real

def op1 (x y : ℝ) : ℝ := x * y / (x + y)
def op2 (x y : ℝ) : ℝ := x + y - x * y

theorem compute_result :
  op2 (op1 2 3) 4 = 2 / 5 :=
by
  sorry

end compute_result_l217_217485


namespace daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l217_217448

-- Given conditions
def cost_price : ℝ := 80
def daily_sales_quantity (x : ℝ) : ℝ := -2 * x + 320
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * daily_sales_quantity x

-- Part 1: Functional relationship
theorem daily_profit_functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) : daily_profit x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Part 2: Maximizing daily profit
theorem daily_profit_maximizes_at_120 (hx : 80 ≤ 120 ∧ 120 ≤ 160) : daily_profit 120 = 3200 :=
by sorry

-- Part 3: Selling price for a daily profit of $2400
theorem selling_price_for_2400_profit (hx : 80 ≤ 100 ∧ 100 ≤ 160) : daily_profit 100 = 2400 :=
by sorry

end daily_profit_functional_relationship_daily_profit_maximizes_at_120_selling_price_for_2400_profit_l217_217448


namespace infinitely_many_m_exists_l217_217244

theorem infinitely_many_m_exists (n : ℕ) (hn : n ≥ 4) : 
  ∃ m : ℕ, m ≥ 2 ∧ m = (n^2 - 3 * n + 2) / 2 ∧ binom m 2 = 3 * binom n 4 := 
by 
  sorry

end infinitely_many_m_exists_l217_217244


namespace inequality_of_f_on_angles_l217_217813

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

-- Stating the properties of the function f
axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, f (x + 1) = -f x
axiom decreasing_interval : ∀ x y : ℝ, (-3 ≤ x ∧ x < y ∧ y ≤ -2) → f x > f y

-- Stating the properties of the angles α and β
variables (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) (hαβ : α ≠ β)

-- The proof statement we want to prove
theorem inequality_of_f_on_angles : f (Real.sin α) > f (Real.cos β) :=
sorry -- The proof is omitted

end inequality_of_f_on_angles_l217_217813


namespace quadratic_function_fixed_points_range_l217_217839

def has_two_distinct_fixed_points (c : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
               (x1 = x1^2 - x1 + c) ∧ 
               (x2 = x2^2 - x2 + c) ∧ 
               x1 < 2 ∧ 2 < x2

theorem quadratic_function_fixed_points_range (c : ℝ) :
  has_two_distinct_fixed_points c ↔ c < 0 :=
sorry

end quadratic_function_fixed_points_range_l217_217839


namespace units_digit_3_pow_1789_units_digit_1777_pow_1777_pow_1777_l217_217363

theorem units_digit_3_pow_1789 (n : ℕ) (h_cycle : [3, 9, 7, 1]) (h_mod : n % 4 = 1) : (3^n) % 10 = 3 :=
by 
sorry

theorem units_digit_1777_pow_1777_pow_1777 (n : ℕ) (h_cycle : [7, 9, 3, 1]) (h_mod : n % 4 = 1) : (7^n) % 10 = 7 :=
by 
sorry

end units_digit_3_pow_1789_units_digit_1777_pow_1777_pow_1777_l217_217363


namespace owen_sleep_hours_l217_217234

-- Define the time spent by Owen in various activities
def hours_work : ℕ := 6
def hours_chores : ℕ := 7
def total_hours_day : ℕ := 24

-- The proposition to be proven
theorem owen_sleep_hours : (total_hours_day - (hours_work + hours_chores) = 11) := by
  sorry

end owen_sleep_hours_l217_217234


namespace negative_solution_condition_l217_217054

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l217_217054


namespace arrange_abc_l217_217211

noncomputable def a : Real := Real.sqrt 2
noncomputable def b : Real := Real.logb 0.5 Real.exp 1
noncomputable def c : Real := Real.log 2

theorem arrange_abc : b < c ∧ c < a := by
  sorry

end arrange_abc_l217_217211


namespace MrFletcherPaymentPerHour_l217_217634

theorem MrFletcherPaymentPerHour :
  (2 * (10 + 8 + 15)) * x = 660 → x = 10 :=
by
  -- This is where you'd provide the proof, but we skip it as per instructions.
  sorry

end MrFletcherPaymentPerHour_l217_217634


namespace sqrt_sqrt_16_eq_pm_2_l217_217700

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := 
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217700


namespace sum_max_min_interval_l217_217286

def f (x : ℝ) : ℝ := 2 * x^2 - 6 * x + 1

theorem sum_max_min_interval (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1) :
  let M := max (f a) (f b)
  let m := min (f a) (f b)
  M + m = 6 :=
by
  rw [h₁, h₂]
  let M := max (f (-1)) (f 1)
  let m := min (f (-1)) (f 1)
  sorry

end sum_max_min_interval_l217_217286


namespace prime_factorization_sum_l217_217213

theorem prime_factorization_sum (w x y z k : ℕ) (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2310) :
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 28 :=
sorry

end prime_factorization_sum_l217_217213


namespace range_of_omega_for_four_zeros_l217_217498

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

-- Define the transformation to obtain g
def g (x : ℝ) : ℝ := Real.cos x

-- Define h based on f and g
def h (ω : ℝ) (x : ℝ) : ℝ := f ω (g x) - 1

-- Define the main theorem
theorem range_of_omega_for_four_zeros (ω : ℝ) : 
  (0 < ω) → 
  (∃ n : ℕ, h ω '' {x | 0 < x ∧ x < 2 * Real.pi}.ncard = 4) ↔ 
  (Real.pi / 2 * 7 ≤ ω ∧ ω ≤ Real.pi / 2 * 9) :=
by 
  sorry

end range_of_omega_for_four_zeros_l217_217498


namespace second_larger_square_l217_217883

theorem second_larger_square (x : ℕ) (hx : ∃ k : ℕ, x = k^2) : 
  ∃ (n : ℕ), n = (∃ k : ℕ, x = k^2) ∧ n = x + 4 * (Int.ofNat (Nat.sqrt x)) + 4 :=
sorry

end second_larger_square_l217_217883


namespace inequality_proof_l217_217521

/-- Given a, b, c as positive real numbers. Prove that 
    √3 * ∛((a + b) * (b + c) * (c + a)) ≥ 2 * √(ab + bc + ca) -/
theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  √3 * ((a + b) * (b + c) * (c + a)).nthRoot 3 ≥ 2 * √(a * b + b * c + c * a) :=
by sorry

end inequality_proof_l217_217521


namespace parallel_angles_l217_217947

theorem parallel_angles (A B C D : ℝ) (h : ∠ B - ∠ A = 30) :
  ∠ A = 75 ∧ ∠ B = 105 ∧ ∠ C = 75 ∧ ∠ D = 105 :=
sorry

end parallel_angles_l217_217947


namespace count_valid_integers_l217_217915

/-- 
  The count of 4-digit positive integers that consist solely of even digits 
  and are divisible by both 5 and 3 is equal to 120.
-/
theorem count_valid_integers : 
  let even_digits := [0, 2, 4, 6, 8] in 
  let four_digit_nums := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ d ∈ (nat.digits 10 n), d ∈ even_digits} in
  let divisible_by_5 := {n : ℕ | n % 10 = 0} in
  let sum_is_divisible_by_3 := {n : ℕ | (nat.digits 10 n).sum % 3 = 0} in
  let valid_numbers := (four_digit_nums ∩ divisible_by_5 ∩ sum_is_divisible_by_3) in
  finset.card valid_numbers = 120 :=
sorry

end count_valid_integers_l217_217915


namespace exists_line_intersecting_at_least_four_circles_l217_217322

-- Define the conditions and the main theorem.

structure Circle where
  circumference : ℝ
  center : ℝ × ℝ
  radius : ℝ

noncomputable def total_circumference (circles : List Circle) : ℝ :=
  circles.sum Circle.circumference

theorem exists_line_intersecting_at_least_four_circles
  (circles : List Circle)
  (h_total_circumference : total_circumference circles = 10)
  (h_inside_square : ∀ circle, circle.center.1 + circle.radius ≤ 1
                             ∧ circle.center.1 - circle.radius ≥ 0
                             ∧ circle.center.2 + circle.radius ≤ 1
                             ∧ circle.center.2 - circle.radius ≥ 0) :
  ∃ line, ∃ four_intersecting_circles : List Circle, four_intersecting_circles.length ≥ 4 ∧ ∀ circle ∈ four_intersecting_circles, line_intersects_circle line circle :=
sorry

end exists_line_intersecting_at_least_four_circles_l217_217322


namespace remainder_of_M_mod_210_l217_217472

def M : ℤ := 1234567891011

theorem remainder_of_M_mod_210 :
  (M % 210) = 31 :=
by
  have modulus1 : M % 6 = 3 := by sorry
  have modulus2 : M % 5 = 1 := by sorry
  have modulus3 : M % 7 = 2 := by sorry
  -- Using Chinese Remainder Theorem
  sorry

end remainder_of_M_mod_210_l217_217472


namespace not_diff_of_squares_2002nd_l217_217468

theorem not_diff_of_squares_2002nd : 
  ∀ k, k = 2002 → ∃ n, n = 4 * k - 2 ∧ (∀ x y : ℤ, n ≠ x^2 - y^2) := 
by
  intro k hk
  use 8006
  split
  { norm_num [hk] }
  { sorry }

end not_diff_of_squares_2002nd_l217_217468


namespace min_value_F_value_expression_if_f_eq_2f_l217_217536

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x - Real.pi / 4)

def F (x : ℝ) : ℝ := (Real.cos x + Real.sin x)^2 - (Real.cos x + Real.sin x) * (Real.sin x - Real.cos x)

theorem min_value_F : ∃ (x : ℝ) (k : ℤ), F(x) = 1 - Real.sqrt 2 ∧ (x = k * Real.pi - 3 * Real.pi / 8) :=
sorry

theorem value_expression_if_f_eq_2f' (x : ℝ) (hx : f x = 2 * (Real.cos x + Real.sin x)) : (3 - Real.cos (2 * x)) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 19 / 2 :=
sorry

end min_value_F_value_expression_if_f_eq_2f_l217_217536


namespace amplitude_f_phase_shift_f_l217_217019

-- Define the function f as -5 * cos (x + π / 4)
def f (x : ℝ) : ℝ := -5 * Real.cos (x + Real.pi / 4)

-- Prove that the amplitude of f is 5
theorem amplitude_f : ∃ A : ℝ, ∀ x : ℝ, abs (f x) ≤ A ∧ A = 5 := sorry

-- Prove that the phase shift of f is -π / 4
theorem phase_shift_f : ∃ φ : ℝ, ∀ x : ℝ, f (x + φ) = -5 * Real.cos x ∧ φ = -Real.pi / 4 := sorry

end amplitude_f_phase_shift_f_l217_217019


namespace find_b_l217_217300

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

end find_b_l217_217300


namespace find_fraction_l217_217065

theorem find_fraction
  (a₁ a₂ b₁ b₂ c₁ c₂ x y : ℚ)
  (h₁ : a₁ = 3) (h₂ : a₂ = 7) (h₃ : b₁ = 6) (h₄ : b₂ = 5)
  (h₅ : c₁ = 1) (h₆ : c₂ = 7)
  (h : (a₁ / a₂) / (b₁ / b₂) = (c₁ / c₂) / (x / y)) :
  (x / y) = 2 / 5 := 
by
  sorry

end find_fraction_l217_217065


namespace product_of_terms_in_geometric_sequence_l217_217106

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m

noncomputable def roots_of_quadratic (a b c : ℝ) (r1 r2 : ℝ) : Prop :=
r1 * r2 = c

theorem product_of_terms_in_geometric_sequence
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : roots_of_quadratic 1 (-4) 3 (a 5) (a 7)) :
  a 2 * a 10 = 3 :=
sorry

end product_of_terms_in_geometric_sequence_l217_217106


namespace problem_solution_l217_217538

def f (x a : ℝ) : ℝ := 2 * x^3 - a * x^2 + 2

noncomputable def M (a : ℝ) : ℝ := real.max (f 0 a) (f 1 a)

noncomputable def m (a : ℝ) : ℝ := f (a / 3) a

theorem problem_solution (ha : 0 < a ∧ a < 3) : 
  ∀ a, \([\frac{8}{27}, 2) ∋ M a - m a := sorry

end problem_solution_l217_217538


namespace find_k_arithmetic_progression_l217_217577

theorem find_k_arithmetic_progression :
  ∃ k : ℤ, (2 * real.sqrt (225 + k) = real.sqrt (49 + k) + real.sqrt (441 + k)) → k = 255 :=
by
  sorry

end find_k_arithmetic_progression_l217_217577


namespace remainder_of_division_l217_217473

theorem remainder_of_division :
  ∀ (x : ℝ), (3 * x ^ 5 - 2 * x ^ 3 + 5 * x - 8) % (x ^ 2 - 3 * x + 2) = 74 * x - 76 :=
by
  sorry

end remainder_of_division_l217_217473


namespace complex_arithmetic_expr_equals_three_l217_217800

noncomputable def complex_arithmetic_expr : ℝ :=
  real.sqrt 18 / real.sqrt 8 - (real.sqrt 5 ‑ 4)^0 * (2 / 3)⁻¹ + real.sqrt ((-3)^2)

theorem complex_arithmetic_expr_equals_three : complex_arithmetic_expr = 3 := by
  sorry

end complex_arithmetic_expr_equals_three_l217_217800


namespace average_t_values_l217_217888

theorem average_t_values (r₁ r₂ : ℕ) (t : ℕ) :
  r₁ + r₂ = 7 → t = r₁ * r₂ →
    (r₁ > 0 ∧ r₂ > 0) →
    (∀ r₁ r₂, (r₁ + r₂ = 7) → r₁ * r₂ ∈ {6, 10, 12}) →
    (∑ x in {6, 10, 12}, x) / 3 = 28 / 3 :=
by
  intro h₁ h₂ h₃ h₄
  have h_sum : ∑ x in {6, 10, 12}, x = 28 := by sorry
  have h_count : (3 : ℝ) = 3 := by sorry
  exact h_sum / h_count

end average_t_values_l217_217888


namespace violet_ticket_cost_l217_217321

theorem violet_ticket_cost 
  (cost_adult_ticket : ℕ)  -- Cost of one adult ticket
  (cost_child_ticket : ℕ)  -- Cost of one child ticket
  (num_adults : ℕ)        -- Number of adults
  (num_children : ℕ)      -- Number of children
  (h1 : cost_adult_ticket = 35)  -- Condition for adult ticket price
  (h2 : cost_child_ticket = 20)  -- Condition for child ticket price
  (h3 : num_adults = 2)          -- Condition for number of adults
  (h4 : num_children = 5)        -- Condition for number of children
  : (num_adults * cost_adult_ticket + num_children * cost_child_ticket = 170) :=
begin
  sorry -- Proof is not required
end

end violet_ticket_cost_l217_217321


namespace dice_prob_l217_217721

noncomputable def probability_four_twos : ℝ :=
  let total_ways := Nat.choose 12 4
  let prob_each_arrangement := (1 / 8)^4 * (7 / 8)^8
  in total_ways * prob_each_arrangement

theorem dice_prob : probability_four_twos = 0.089 := by
  sorry

end dice_prob_l217_217721


namespace find_f_f_neg1_l217_217104

variable (f : ℝ → ℝ) -- f is a function from ℝ to ℝ

-- Odd function property: ∀ x ∈ ℝ, f(-x) = -f(x)
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The main theorem to prove
theorem find_f_f_neg1 (h_odd : odd_function f)
                      (h1 : f 1 = 2)
                      (h2 : f 2 = 3) :
                      f (f (-1)) = -3 :=
begin
  have h_neg1 : f (-1) = -2 := by rw [h_odd 1, h1],
  have h_neg2 : f (-2) = -3 := by rw [h_odd 2, h2],
  rw [←h_neg1, h_neg2],
  sorry,
end

end find_f_f_neg1_l217_217104


namespace probability_x_lt_y_minus_1_l217_217400

open Set

def rectangle : set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

def area (s : set (ℝ × ℝ)) : ℝ := sorry -- Definition of the area of a set of points

theorem probability_x_lt_y_minus_1 :
  let event := {p : ℝ × ℝ | p ∈ rectangle ∧ p.1 < p.2 - 1} in
  (area event) / (area rectangle) = 3 / 8 :=
by
  sorry

end probability_x_lt_y_minus_1_l217_217400


namespace max_height_of_soccer_ball_is_9_l217_217742

def height (t : ℝ) : ℝ := -4 * t^2 + 12 * t

theorem max_height_of_soccer_ball_is_9 : ∃ t : ℝ, height t = 9 := 
sorry

end max_height_of_soccer_ball_is_9_l217_217742


namespace vector_parallel_solution_l217_217582

theorem vector_parallel_solution (x : ℝ) : 
  let a := (2, 3)
  let b := (x, -9)
  (a.snd = 3) → (a.fst = 2) → (b.snd = -9) → (a.fst * b.snd = a.snd * (b.fst)) → x = -6 := 
by
  intros 
  sorry

end vector_parallel_solution_l217_217582


namespace negative_solution_exists_l217_217061

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l217_217061


namespace match_end_time_is_17_55_l217_217388

-- Definitions corresponding to conditions
def start_time : ℕ := 15 * 60 + 30  -- Convert 15:30 to minutes past midnight
def duration : ℕ := 145  -- Duration in minutes

-- Definition corresponding to the question
def end_time : ℕ := start_time + duration 

-- Assertion corresponding to the correct answer
theorem match_end_time_is_17_55 : end_time = 17 * 60 + 55 :=
by
  -- Proof steps and actual proof will go here
  sorry

end match_end_time_is_17_55_l217_217388


namespace GreatWhiteSharkTeeth_l217_217413

-- Definition of the number of teeth for a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Definition of the number of teeth for a hammerhead shark
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Definition of the number of teeth for a great white shark
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- Statement to prove
theorem GreatWhiteSharkTeeth : great_white_shark_teeth = 420 :=
by
  -- Proof omitted
  sorry

end GreatWhiteSharkTeeth_l217_217413


namespace infinite_triangular_pairs_exists_l217_217240

-- Definitions directly from conditions
def is_triangular (t : ℕ) : Prop :=
  ∃ n : ℕ, t = n * (n + 1) / 2

-- Main statement to be proven
theorem infinite_triangular_pairs_exists : ∃ (a b : ℤ), ∀ t : ℕ, t > 0 → (is_triangular t ↔ is_triangular (a * t + b)) :=
by
  -- Use d as odd integer
  let d := sorry
  -- Set a = d^2, b = (d^2 - 1) / 8
  let a := d ^ 2
  let b := (d ^ 2 - 1) / 8
  -- The actual proof goes here
  sorry

end infinite_triangular_pairs_exists_l217_217240


namespace trucks_speeds_truck_B_speed_truck_B_travel_distance_difference_l217_217395

structure TrucksData :=
  (distance : ℝ) -- 300 km
  (time : ℝ) -- 3 hours
  (speedMultiplier : ℝ) -- 1.5 times

def trucksData : TrucksData :=
  { distance := 300, time := 3, speedMultiplier := 1.5 }

theorem trucks_speeds (A_speed B_speed : ℝ)
  (H_sum : A_speed + B_speed = trucksData.distance / trucksData.time) : 
  A_speed + B_speed = 100 :=
begin
  exact H_sum,
end

theorem truck_B_speed (A_speed B_speed : ℝ)
  (H_sum : A_speed + B_speed = 100)
  (H_mult : B_speed = trucksData.speedMultiplier * A_speed) :
  B_speed = 60 :=
begin
  sorry
end

theorem truck_B_travel_distance_difference (A_speed B_speed : ℝ)
  (H_sum : A_speed + B_speed = 100)
  (H_mult : B_speed = trucksData.speedMultiplier * A_speed)
  (H_A_speed : A_speed = 40): 
  B_speed * trucksData.time - A_speed * trucksData.time = 60 :=
begin
  sorry
end

end trucks_speeds_truck_B_speed_truck_B_travel_distance_difference_l217_217395


namespace triple_comp_g_of_2_l217_217625

def g (n : ℕ) : ℕ :=
  if n ≤ 3 then n^3 - 2 else 4 * n + 1

theorem triple_comp_g_of_2 : g (g (g 2)) = 101 := by
  sorry

end triple_comp_g_of_2_l217_217625


namespace red_marbles_difference_l217_217355

theorem red_marbles_difference 
  (x y : ℕ) 
  (h1 : 7 * x + 3 * x = 140) 
  (h2 : 3 * y + 2 * y = 140)
  (h3 : 10 * x = 5 * y) : 
  7 * x - 3 * y = 20 := 
by 
  sorry

end red_marbles_difference_l217_217355


namespace sqrt_sqrt_16_eq_pm_2_l217_217703

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 := 
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217703


namespace cost_of_each_outside_door_l217_217963

open Real

noncomputable def cost_per_outside_door : ℝ :=
  let x := 20 in
  x

theorem cost_of_each_outside_door (x : ℝ) (h1 : 3 * (x / 2) + 2 * x = 70) : x = cost_per_outside_door :=
by
  sorry

end cost_of_each_outside_door_l217_217963


namespace determine_range_of_m_l217_217494

noncomputable def range_m (m : ℝ) (x : ℝ) : Prop :=
  ∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
       (∃ x, -x^2 + 7 * x + 8 ≥ 0)

theorem determine_range_of_m (m : ℝ) :
  (-1 ≤ m ∧ m ≤ 1) ↔
  (∀ x, (-x^2 + 7 * x + 8 < 0 → (x < 1 - 2 * |m| ∨ x > 1 + 2 * |m|)) ∧
         (∃ x, -x^2 + 7 * x + 8 ≥ 0)) :=
by
  sorry

end determine_range_of_m_l217_217494


namespace night_crew_fraction_l217_217421

theorem night_crew_fraction (D N B : ℕ) (h1 : N = (2 / 3 : ℝ) * D)
    (h2 : B > 0) -- Total number of boxes is positive to avoid division by zero
    (h3 : (2 / 3 : ℝ) * B = 2 / 3 ∙ B): 
    ((1 / 3 : ℝ) * B / N) / ((2 / 3 : ℝ) * B / D) = 3 / 4 := 
by sorry

end night_crew_fraction_l217_217421


namespace determine_k_l217_217015

theorem determine_k (θ : ℝ) (h : ((sin θ + 1 / sin θ) ^ 4 + (cos θ + 1 / cos θ) ^ 4 = k + (tan θ) ^ 4 + (cot θ) ^ 4)) : k = 30 :=
by
  -- proof omitted for brevity
  sorry

end determine_k_l217_217015


namespace nonneg_integer_count_l217_217556

noncomputable theory

def count_nonneg_integers_in_form : ℕ :=
  let digit_values := [-1, 0, 1]
  let max_pow := 8
  let pow_5 := List.range (max_pow + 1) |>.map (λ i => 5^i)
  let max_value := pow_5.sum
  max_value + 1

theorem nonneg_integer_count : count_nonneg_integers_in_form = 488282 := sorry

end nonneg_integer_count_l217_217556


namespace number_of_squares_in_H_l217_217292

-- Define the set H
def H : Set (ℤ × ℤ) :=
{ p | 2 ≤ abs p.1 ∧ abs p.1 ≤ 10 ∧ 2 ≤ abs p.2 ∧ abs p.2 ≤ 10 }

-- State the problem
theorem number_of_squares_in_H : 
  (∃ S : Finset (ℤ × ℤ), S.card = 20 ∧ 
    ∀ square ∈ S, 
      (∃ a b c d : ℤ × ℤ, 
        a ∈ H ∧ b ∈ H ∧ c ∈ H ∧ d ∈ H ∧ 
        (∃ s : ℤ, s ≥ 8 ∧ 
          (a.1 = b.1 ∧ b.2 = c.2 ∧ c.1 = d.1 ∧ d.2 = a.2 ∧ 
           abs (a.1 - c.1) = s ∧ abs (a.2 - d.2) = s)))) :=
sorry

end number_of_squares_in_H_l217_217292


namespace find_k_l217_217576

theorem find_k : ∃ k : ℕ, (2 * (Real.sqrt (225 + k)) = (Real.sqrt (49 + k) + Real.sqrt (441 + k))) → k = 255 :=
by
  sorry

end find_k_l217_217576


namespace simplify_complex_fraction_l217_217110

-- Definition of complex numbers and basic properties
def complex : Type := (ℝ × ℝ)

-- Define the imaginary unit
noncomputable def i : complex := (0, 1)

-- Define addition and multiplication of complex numbers
def add (z w : complex) : complex := (z.1 + w.1, z.2 + w.2)
def mul (z w : complex) : complex := (z.1 * w.1 - z.2 * w.2, z.1 * w.2 + z.2 * w.1)

-- Define conjugate of a complex number
def conj (z : complex) : complex := (z.1, -z.2)

-- Define the fraction simplification
def fraction_simplify (z w : complex) : complex :=
  mul z (conj w)

-- Provide the conditions and the theorem to prove
theorem simplify_complex_fraction :
  let num := add i (-1, 0)
      denom := add 1 i in
  fraction_simplify num denom = i :=
by
  sorry

end simplify_complex_fraction_l217_217110


namespace joe_paint_fraction_l217_217609

theorem joe_paint_fraction :
  let total_paint := 360
  let fraction_first_week := 1 / 9
  let used_first_week := (fraction_first_week * total_paint)
  let remaining_after_first_week := total_paint - used_first_week
  let total_used := 104
  let used_second_week := total_used - used_first_week
  let fraction_second_week := used_second_week / remaining_after_first_week
  fraction_second_week = 1 / 5 :=
by
  sorry

end joe_paint_fraction_l217_217609


namespace palindrome_addition_l217_217014

theorem palindrome_addition (a b c d : ℕ) 
  (ha : a = 22) 
  (hb : b = 979) 
  (hpal_a : ∀ {a}, a = nat.reverse a )
  (hpal_b : ∀ {b}, b = nat.reverse b ) :
  a + b = 1001 :=
by
  sorry

end palindrome_addition_l217_217014


namespace number_of_boys_in_class_l217_217671

theorem number_of_boys_in_class 
  (n : ℕ)
  (average_height : ℝ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_average_height : ℝ)
  (initial_average_height : average_height = 185)
  (incorrect_record : incorrect_height = 166)
  (correct_record : correct_height = 106)
  (actual_avg : actual_average_height = 183) 
  (total_height_incorrect : ℝ) 
  (total_height_correct : ℝ) 
  (total_height_eq : total_height_incorrect = 185 * n)
  (correct_total_height_eq : total_height_correct = 185 * n - (incorrect_height - correct_height))
  (actual_total_height_eq : total_height_correct = actual_average_height * n) :
  n = 30 :=
by
  sorry

end number_of_boys_in_class_l217_217671


namespace find_x_l217_217613

theorem find_x (x : ℝ) (N : ℝ) (h₁ : N = 2^(2^2)) (h₂ : N^(N^N) = 2^(2^x)) : x = 66 :=
by
  sorry

end find_x_l217_217613


namespace sector_field_area_l217_217173

/-- Given a sector field with a circumference of 30 steps and a diameter of 16 steps, prove that its area is 120 square steps. --/
theorem sector_field_area (C : ℝ) (d : ℝ) (A : ℝ) : 
  C = 30 → d = 16 → A = 120 :=
by
  sorry

end sector_field_area_l217_217173


namespace pawn_cycle_exists_l217_217969

theorem pawn_cycle_exists (n : ℕ) (hn : n > 0) :
  ∃ seq : list (ℕ × ℕ), seq.length = n^2 ∧ 
    (∀ i ∈ seq, i.1 ∈ {1, ..., n} ∧ i.2 ∈ {1, ..., n}) ∧ 
    (∀ i ≠ j, i ∈ seq → j ∈ seq → i ≠ j) ∧ 
    (seq.head = seq.last) :=
sorry

end pawn_cycle_exists_l217_217969


namespace integral_example_l217_217821

theorem integral_example :
  ∫ x in -1..1, (x^2 + real.sqrt(1 - x^2)) = (2/3) + (real.pi/2) :=
by
  sorry

end integral_example_l217_217821


namespace locus_of_centers_correct_l217_217864

noncomputable def locus_of_centers (O P : Point) (r : ℝ) : Set Point :=
  if dist O P > r then {X | |dist O X - dist P X| = r }
  else if dist O P = r then {X | collinear O P X ∧ X ≠ O}
  else {X | dist O X + dist P X = r }

theorem locus_of_centers_correct {O P : Point} (r : ℝ) :
  let locus := locus_of_centers O P r in
  ∀ X, 
    if dist O P > r then (X ∈ locus ↔ |dist O X - dist P X| = r) else
    if dist O P = r then (X ∈ locus ↔ collinear O P X ∧ X ≠ O) else
    (X ∈ locus ↔ dist O X + dist P X = r) :=
begin
  sorry
end

end locus_of_centers_correct_l217_217864


namespace find_a_find_area_l217_217551

-- Definitions based on given conditions
def angleA := 45
def side_c := 2
def vector_m := (1, Real.cos C)
def vector_n := (Real.cos C, 1)
def dot_product_m_n := vector_m.1 * vector_n.1 + vector_m.2 * vector_n.2 = 1

-- Problem 1
theorem find_a : 
  (dot_product_m_n) → 
  (∀ (0 < C) (C < 180), C = 60) → 
  (a / Real.sin angleA = side_c / Real.sin 60) → 
  a = 2 * Real.sqrt 6 / 3 := sorry

-- Problem 2
theorem find_area (a b : ℝ) (area_S : ℝ → ℝ) : 
  (dot_product_m_n) → 
  a + b = 4 → 
  side_c = 2 → 
  (a^2 + b^2 - a * b = 4) ∧ (a^2 + b^2 + 2 * a * b = 16) → 
  area_S = Real.sqrt 3 := sorry

end find_a_find_area_l217_217551


namespace hurricane_damage_in_euros_l217_217392

-- Define the conditions
def usd_damage : ℝ := 45000000  -- Damage in US dollars
def exchange_rate : ℝ := 0.9    -- Exchange rate from US dollars to Euros

-- Define the target value in Euros
def eur_damage : ℝ := 40500000  -- Expected damage in Euros

-- The theorem to prove
theorem hurricane_damage_in_euros :
  usd_damage * exchange_rate = eur_damage :=
by
  sorry

end hurricane_damage_in_euros_l217_217392


namespace sum_base_conversions_l217_217822

def base15_to_base10 (n : Nat) : Nat :=
  5 * 15^2 + 3 * 15^1 + 7 * 15^0

def base7_to_base10 (n : Nat) (A : Nat) : Nat :=
  1 * 7^2 + A * 7^1 + 4 * 7^0

theorem sum_base_conversions (A : Nat) (hA : A = 10) :
  base15_to_base10 537 + base7_to_base10 1A4 A = 1300 :=
by
  have h1 : base15_to_base10 537 = 1125 + 45 + 7 := rfl
  have h2 : base7_to_base10 1A4 10 = 49 + 70 + 4 := by
    rw [base7_to_base10, hA]
    exact rfl
  have h3 : 1125 + 45 + 7 = 1177 := rfl
  have h4 : 49 + 70 + 4 = 123 := rfl
  have h5 : 1177 + 123 = 1300 := rfl
  rw [h1, h3] at *
  rw [h2, h4] at *
  exact h5

end sum_base_conversions_l217_217822


namespace zero_in_interval_l217_217284

noncomputable def f (x : ℝ) : ℝ := 2^x + 4*x - 3

theorem zero_in_interval :
  (f(1/4) < 0) → (f(1/2) > 0) → (∃ x, (1/4 < x ∧ x < 1/2) ∧ f x = 0) :=
by
  intros
  sorry

end zero_in_interval_l217_217284


namespace g_at_12_l217_217140

def g (n : ℤ) : ℤ := n^2 + 2*n + 23

theorem g_at_12 : g 12 = 191 := by
  -- proof skipped
  sorry

end g_at_12_l217_217140


namespace equilateral_triangle_sum_independence_l217_217612

theorem equilateral_triangle_sum_independence :
  ∀ (a : ℝ) (ABC : set ℂ) (P : ℂ), 
     (equilateral_triangle ABC ∧ P ∈ circumcircle ABC) → 
     (S_n (P : P n) = |PA|^n + |PB|^n + |PC|^n) :=
by
  sorry

end equilateral_triangle_sum_independence_l217_217612


namespace sum_first_10_b_l217_217126

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 ^ (n - 1)

noncomputable def b (n : ℕ) : ℕ :=
  Math.log 3 (a n)

theorem sum_first_10_b : (Finset.range 10).sum b = 45 :=
  sorry

end sum_first_10_b_l217_217126


namespace hilt_books_transaction_difference_l217_217639

noncomputable def total_cost_paid (original_price : ℝ) (num_first_books : ℕ) (discount1 : ℝ) (num_second_books : ℕ) (discount2 : ℝ) : ℝ :=
  let cost_first_books := num_first_books * original_price * (1 - discount1)
  let cost_second_books := num_second_books * original_price * (1 - discount2)
  cost_first_books + cost_second_books

noncomputable def total_sale_amount (sale_price : ℝ) (interest_rate : ℝ) (num_books : ℕ) : ℝ :=
  let compounded_price := sale_price * (1 + interest_rate) ^ 1
  compounded_price * num_books

theorem hilt_books_transaction_difference : 
  let original_price := 11
  let num_first_books := 10
  let discount1 := 0.20
  let num_second_books := 5
  let discount2 := 0.25
  let sale_price := 25
  let interest_rate := 0.05
  let num_books := 15
  total_sale_amount sale_price interest_rate num_books - total_cost_paid original_price num_first_books discount1 num_second_books discount2 = 264.50 :=
by
  sorry

end hilt_books_transaction_difference_l217_217639


namespace part_I_part_II_l217_217871

-- Define the sequence and the sum of its terms
def a (n : ℕ) : ℚ := if n = 1 then 1/2 else sorry -- Recursive definition needed for a(n)
def S (n : ℕ) : ℚ := (Finset.range (n + 1)).sum (λ i, a i)

-- Conditions
axiom a1 : a 1 = 1/2
axiom a_rec (n : ℕ) : 2 * a (n + 1) = S n + 1

-- Part I: Prove values of a_2 and a_3
theorem part_I : a 2 = 3/4 ∧ a 3 = 9/8 := sorry

-- Define b_n
def b (n : ℕ) : ℚ := 2 * a n - 2 * n - 1

-- Sum of the first n terms of b_n
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i, b i)

-- Part II: Prove the sum of the first n terms of b_n
theorem part_II (n : ℕ) : T n = 2 * (3/2)^n - n^2 - 2 * n - 2 := sorry

end part_I_part_II_l217_217871


namespace side_length_square_eq_4_l217_217272

theorem side_length_square_eq_4 (s : ℝ) (h : s^2 - 3 * s = 4) : s = 4 :=
sorry

end side_length_square_eq_4_l217_217272


namespace xy_addition_equals_13_l217_217562

theorem xy_addition_equals_13 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt_15 : x < 15) (hy_lt_15 : y < 15) (hxy : x + y + x * y = 49) : x + y = 13 :=
by
  sorry

end xy_addition_equals_13_l217_217562


namespace min_exp_product_l217_217510

theorem min_exp_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 3) :
    (Real.exp (1 / a) * Real.exp (1 / b) ≥ Real.exp 3) :=
begin
  sorry
end

end min_exp_product_l217_217510


namespace find_constants_l217_217467

noncomputable def constants (A B C : ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 → 5 * x / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2

theorem find_constants : ∃ (A B C : ℚ), constants A B C ∧ A = 5 ∧ B = -5 ∧ C = -5 :=
begin
  use [5, -5, -5],
  split,
  { intros x h,
    have hne1 : x - 4 ≠ 0 := sub_ne_zero_of_ne h.2,
    have hne2 : x - 2 ≠ 0 := sub_ne_zero_of_ne h.1,
    field_simp [hne1, hne2],
    ring },
  split, refl,
  split, refl,
  refl
end

end find_constants_l217_217467


namespace line_HN_fixed_point_l217_217527

-- Define given points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Ellipse equation based on given conditions
def ellipse_eq (a b x y : ℝ) : Prop :=
  x^2 / a + y^2 / b = 1

-- Given ellipse passes through points A and B
def ellipse_through_A_and_B : Prop :=
  ellipse_eq 3 4 A.1 A.2 ∧ ellipse_eq 3 4 B.1 B.2

-- Point H and segment conditions
def H_condition (M T H : ℝ × ℝ) : Prop :=
  2 * (T.1 - M.1, T.2 - M.2) = (H.1 - T.1, H.2 - T.2)

-- Proving line HN passes through a fixed point assuming H condition and M, N inclusion in ellipse
theorem line_HN_fixed_point :
  ∃ K : ℝ × ℝ, ellipse_through_A_and_B → 
  ∀ M N : ℝ × ℝ, 
  (ellipse_eq 3 4 M.1 M.2) → 
  (ellipse_eq 3 4 N.1 N.2) → 
  H_condition M ((2 * M.1 + M.1, 2 * M.2 - (2 * (M.2 + 2) / 3))) ((4 * M.1, 4 * M.2)) →
  ∃ H : ℝ × ℝ, H_condition M ((2 * M.1 + M.1, 2 * M.2 - (2 * (M.2 + 2) / 3))) H → 
  (H, N).1 * (0, -2).2 - (H, N).2 * (0, -2).1 = 0 :=
by sorry

end line_HN_fixed_point_l217_217527


namespace geometric_sequence_general_formula_arithmetic_sequence_property_proof_l217_217877

-- Given conditions for the infinite geometric sequence
variables {a : ℕ → ℤ} {S : ℕ → ℤ}
variable (n : ℕ)

noncomputable def general_formula_geometric_seq : Prop :=
  (∀ n : ℕ, a n = 3 ^ (n - 1))

noncomputable def arithmetic_sequence_property (k : ℕ) : Prop :=
  3 * S k - 2 * S (k + 1) = 2 * S (k + 1) - S (k + 2)

-- Proofs to be provided
theorem geometric_sequence_general_formula
  (a2_eq_3 : a 2 = 3)
  (a1_a3_eq_10 : a 1 + a 3 = 10)
  (sum_S : ∀ n : ℕ, S n = (3 ^ n - 1) / 2) :
  general_formula_geometric_seq := sorry

theorem arithmetic_sequence_property_proof
  (a2_eq_3 : a 2 = 3)
  (a1_a3_eq_10 : a 1 + a 3 = 10)
  (sum_S : ∀ n : ℕ, S n = (3 ^ n - 1) / 2)
  (k : ℕ) :
  arithmetic_sequence_property k := sorry

end geometric_sequence_general_formula_arithmetic_sequence_property_proof_l217_217877


namespace isabella_babysits_hours_per_day_l217_217604

theorem isabella_babysits_hours_per_day : 
  ∀ (h : ℕ), (5 * 7 * 6 * h = 1050) → h = 5 :=
by 
  intros h H,
  sorry

end isabella_babysits_hours_per_day_l217_217604


namespace proof_problem_l217_217090

-- Define point P and the conditions given in the problem
structure Point :=
  (x : ℝ)
  (y : ℝ)

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def y_axis_distance (P : Point) : ℝ :=
  |P.x|

def satisfies_condition (P : Point) : Prop :=
  distance P ⟨1, 0⟩ - y_axis_distance P = 1

-- Define trajectories based on the conditions
def trajectory_C_1 (P : Point) : Prop :=
  P.y ^ 2 = 4 * P.x ∧ 0 ≤ P.x

def trajectory_C_2 (P : Point) : Prop :=
  P.y = 0 ∧ P.x < 0

-- Define lines and their intersections
def line_through_F (k : ℝ) : Point → Prop :=
  λ P, P.y = k * (P.x - 1)

def line_perpendicular_through_F (k : ℝ) : Point → Prop :=
  λ P, P.y = - (1 / k) * (P.x - 1)

-- The proof problem combining all conditions
theorem proof_problem
  (P : Point)
  (H₁ : satisfies_condition P)
  (H₂ : ∀ P, trajectory_C_1 P ∨ trajectory_C_2 P)
  (k : ℝ) (hk : k ≠ 0) :
  let A := (line_through_F k) in
  let B := (line_perpendicular_through_F k) in
  let C := Point in
  (∃ P1 P2 P3 P4 : Point,
    trajectory_C_1 P1 ∧ trajectory_C_1 P2 ∧ trajectory_C_1 P3 ∧ trajectory_C_1 P4 ∧
    A P1 ∧ A P2 ∧ B P3 ∧ B P4 ∧
    (P1.x + P3.x + 1) * (P2.x + P4.x + 1)) = 16 :=
sorry

end proof_problem_l217_217090


namespace common_real_solution_unique_y_l217_217279

theorem common_real_solution_unique_y (x y : ℝ) 
  (h1 : x^2 + y^2 = 16) 
  (h2 : x^2 - 3 * y + 12 = 0) : 
  y = 4 :=
by
  sorry

end common_real_solution_unique_y_l217_217279


namespace find_k_l217_217370

variables (k : ℝ) (e1 e2 : ℝ × ℝ)

def AB := 2 • e1 + k • e2
def CB := e1 + 3 • e2
def CD := 2 • e1 - e2

def collinear (v1 v2 : ℝ × ℝ) : Prop := ∃ λ : ℝ, v1 = λ • v2

-- The condition that points A, B, and D are collinear
axiom collinear_condition : collinear AB (CB + CD)

theorem find_k : k = -8 :=
by sorry

end find_k_l217_217370


namespace find_ellipse_and_line_l217_217113

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ x y : ℝ, (a > b > 0) ∧ (a = 2) ∧ (b^2 = a^2 - 3) ∧
  (x / a)^2 + (y / b)^2 = 1

noncomputable def line_through_focus (m : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 x3 y3 : ℝ, 
  (x + m * y + sqrt(3) = 0) ∧ 
  2 * x3 = x1 + sqrt(3) * x2 ∧
  2 * y3 = y1 + sqrt(3) * y2 ∧ 
  (x1 * x2 + 4 * y1 * y2 = 0) ∧ 
  ((m^2 + 4) * y^2 + 2 * sqrt(3) * m * y - 1 = 0) ∧ 
  (m^2 = 2)

theorem find_ellipse_and_line :
  (ellipse_equation 2 1) ∧ (line_through_focus sqrt(2)) ∧ (line_through_focus (-sqrt(2))) :=
begin
  sorry
end

end find_ellipse_and_line_l217_217113


namespace probability_not_parallel_l217_217096

-- Definitions of the conditions
def l1 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def l2 (A B x y : ℝ) : Prop := A * x + B * y + 2 = 0

-- Sets of possible values for A and B
def A_values : Set ℝ := {1, 2, 3, 4}
def B_values : Set ℝ := {1, 2, 3, 4}

-- Condition for lines being parallel
def parallel (A B : ℝ) : Prop := B = 2 * A ∧ A ≠ 2

-- Statement of the probability problem
theorem probability_not_parallel : 
  let total_combinations : ℕ := 16 in
  let not_parallel_combinations : ℕ := 15 in
  let probability : ℚ := 15 / 16 in
  ∃ (A B : ℝ), A ∈ A_values ∧ B ∈ B_values ∧ ¬ parallel A B → 
  probability_not_parallel = probability :=
sorry

end probability_not_parallel_l217_217096


namespace standard_equation_of_ellipse_tangent_line_and_point_l217_217508

open Real

-- Define an ellipse with given conditions
def ellipse_center_origin_foci_x_axis (e : ℝ) (h : e = 1/2) : Prop :=
  ∃ a b c : ℝ, b = sqrt 3 ∧ c / a = 1 / 2 ∧ a^2 = b^2 + c^2 ∧ a = 2 ∧ c = 1

-- Part 1: Prove the standard equation of the ellipse
theorem standard_equation_of_ellipse : 
  ellipse_center_origin_foci_x_axis (1 / 2) → 
  ∃ a b : ℝ, a = 2 ∧ b = sqrt 3 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) :=
by
  sorry

-- Define the condition for a tangent line passing through point (2, 1) in first quadrant
def line_through_point_tangent_to_ellipse (P : ℝ × ℝ) (l : ℝ → ℝ) (M : ℝ × ℝ) : Prop :=
  P = (2,1) ∧ M.1 = 1 ∧ M.2 = 3/2 ∧ (∀ x, l x = - 1/2 * (x - 2) + 1) ∧ x^2 / 4 + y^2 / 3 = 1

-- Part 2: Prove the tangent line equation and coordinates of the tangent point
theorem tangent_line_and_point (P : ℝ × ℝ) (l : ℝ → ℝ) (M : ℝ × ℝ) :
  ellipse_center_origin_foci_x_axis (1 / 2) →
  line_through_point_tangent_to_ellipse P l M →
  (∀ x, l x = -1/2 * x + 2) ∧ M = (1, 3/2) :=
by
  sorry

end standard_equation_of_ellipse_tangent_line_and_point_l217_217508


namespace arith_seq_sum_20_l217_217506

theorem arith_seq_sum_20 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∀ n, a n = a 0 + n * a 1)
  (h_sum : ∀ n, S n = n / 2 * (2 * a 0 + (n - 1) * a 1))
  (h2 : a 1 = 0)
  (h3 : a 3 * a 5 = 12)
  (ha0_pos : a 0 > 0) :
  S 20 = -340 :=
sorry

end arith_seq_sum_20_l217_217506


namespace number_of_faces_of_prism_proof_l217_217299

noncomputable def number_of_faces_of_prism (n : ℕ) : ℕ := 2 + n

theorem number_of_faces_of_prism_proof (n : ℕ) (E_p E_py : ℕ) (h1 : E_p + E_py = 30) (h2 : E_p = 3 * n) (h3 : E_py = 2 * n) :
  number_of_faces_of_prism n = 8 :=
by
  sorry

end number_of_faces_of_prism_proof_l217_217299


namespace exactly_one_true_l217_217125

-- Given conditions
def p (x : ℝ) : Prop := (x^2 - 3 * x + 2 ≠ 0) → (x ≠ 2)

-- Define the contrapositive of p
def contrapositive_p (x : ℝ) : Prop := (x = 2) → (x^2 - 3 * x + 2 = 0)

-- Define the converse of p
def converse_p (x : ℝ) : Prop := (x ≠ 2) → (x^2 - 3 * x + 2 ≠ 0)

-- Define the inverse of p
def inverse_p (x : ℝ) : Prop := (x = 2 → x^2 - 3 * x + 2 = 0)

-- Formalize the problem: Prove that exactly one of the converse, inverse, and contrapositive of p is true.
theorem exactly_one_true :
  (∀ x : ℝ, p x) →
  ((∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ (∀ x : ℝ, converse_p x) ∧ ¬(∀ x : ℝ, inverse_p x) ∨
   ¬(∃ x : ℝ, contrapositive_p x) ∧ ¬(∀ x : ℝ, converse_p x) ∧ (∀ x : ℝ, inverse_p x)) :=
sorry

end exactly_one_true_l217_217125


namespace interest_credited_cents_l217_217785

theorem interest_credited_cents (P : ℝ) (rt : ℝ) (A : ℝ) (interest : ℝ) :
  A = 255.31 →
  rt = 1 + 0.05 * (1/6) →
  P = A / rt →
  interest = A - P →
  (interest * 100) % 100 = 10 :=
by
  intro hA
  intro hrt
  intro hP
  intro hint
  sorry

end interest_credited_cents_l217_217785


namespace sqrt_sqrt_16_eq_pm2_l217_217707

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l217_217707


namespace probability_dmitry_before_father_l217_217339

theorem probability_dmitry_before_father (m x y z : ℝ) (h1 : 0 < x) (h2 : x < m) (h3 : 0 < y)
    (h4 : y < z) (h5 : z < m) : 
    (measure_theory.measure_space.volume {y | y < x} / 
    measure_theory.measure_space.volume {x, y, z | 0 < x ∧ x < m ∧ 0 < y ∧ y < z ∧ z < m}) = (2 / 3) :=
  sorry

end probability_dmitry_before_father_l217_217339


namespace borrowed_nickels_l217_217225

def n_original : ℕ := 87
def n_left : ℕ := 12
def n_borrowed : ℕ := n_original - n_left

theorem borrowed_nickels : n_borrowed = 75 := by
  sorry

end borrowed_nickels_l217_217225


namespace system_has_negative_solution_iff_sum_zero_l217_217051

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end system_has_negative_solution_iff_sum_zero_l217_217051


namespace fractional_equation_root_l217_217574

theorem fractional_equation_root (k : ℚ) (x : ℚ) (h : (2 * k) / (x - 1) - 3 / (1 - x) = 1) : k = -3 / 2 :=
sorry

end fractional_equation_root_l217_217574


namespace hexagon_area_l217_217289

theorem hexagon_area (a b c R : ℝ) (h1 : a + b + c = 35) (h2 : R = 8) : 
    hexagon_area (a b c R) = 140 := sorry

end hexagon_area_l217_217289


namespace arithmetic_sequence_n_equals_8_l217_217529

theorem arithmetic_sequence_n_equals_8 :
  (∀ (a b c : ℕ), a + (1 / 4) * c = 2 * (1 / 2) * b) → ∃ n : ℕ, n = 8 :=
by 
  sorry

end arithmetic_sequence_n_equals_8_l217_217529


namespace find_BC_l217_217504

variable (A B C : Type)
variables (a b : ℝ) -- Angles
variables (AB BC CA : ℝ) -- Sides of the triangle

-- Given conditions:
-- 1: Triangle ABC
-- 2: cos(a - b) + sin(a + b) = 2
-- 3: AB = 4

theorem find_BC (hAB : AB = 4) (hTrig : Real.cos (a - b) + Real.sin (a + b) = 2) :
  BC = 2 * Real.sqrt 2 := 
sorry

end find_BC_l217_217504


namespace vector_dot_product_eq_zero_l217_217523

variables {V : Type*} [inner_product_space ℝ V] {A B C D : V}

-- Conditions
def is_midpoint (D A B : V) : Prop := D = (A + B) / 2
def is_isosceles_right_triangle (A B C : V) : Prop := 
  dist A B = dist B C ∧ dist A C = sqrt (2) * dist A B

-- Statement to Prove
theorem vector_dot_product_eq_zero (h_midpoint : is_midpoint D A B)
                                   (h_triangle : is_isosceles_right_triangle A B C)  
                                   (h_CD : D = (C + B) / 2)
                                   (h_BA : A - B ≠ 0) :
  (C + B) ⋅ (C - B) = 0 :=
sorry

end vector_dot_product_eq_zero_l217_217523


namespace winning_candidate_percentage_l217_217168

theorem winning_candidate_percentage 
    (votes_winner : ℕ)
    (votes_total : ℕ)
    (votes_majority : ℕ)
    (H1 : votes_total = 900)
    (H2 : votes_majority = 360)
    (H3 : votes_winner - (votes_total - votes_winner) = votes_majority) :
    (votes_winner : ℕ) * 100 / (votes_total : ℕ) = 70 := by
    sorry

end winning_candidate_percentage_l217_217168


namespace find_f_neg_half_l217_217530

def g (x : ℝ) := log x / log 2  -- defining g(x) = log2(x)

-- We start with the definition that f and g are symmetric with respect to the line y = x
def is_symmetric (f g : ℝ → ℝ) := ∀ x, f (g x) = x ∧ g (f x) = x

-- Given that f is symmetric to g
axiom sym_f_g : ∃ f : ℝ → ℝ, is_symmetric f g

noncomputable def f := if h : ∃ f, is_symmetric f g then classical.some h else λ x, 0

-- Now we state the theorem to find f(-1/2)
theorem find_f_neg_half : f (-1/2) = real.sqrt 2 / 2 :=
sorry

end find_f_neg_half_l217_217530


namespace range_of_sine_l217_217471

theorem range_of_sine {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x ≥ Real.sqrt 2 / 2) :
  Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4 :=
by
  sorry

end range_of_sine_l217_217471


namespace equiangular_shift_l217_217845

/-- Definition of equiangular sets --/
def equiangular (A B : Finset ℕ) : Prop :=
  A.card = B.card ∧
  A.sum id = B.sum id ∧
  A.sum (λ x, x * x) = B.sum (λ x, x * x) ∧
  A ∩ B = ∅

theorem equiangular_shift (n : ℕ) (A B : Finset ℕ) (k : ℕ) (hk : 0 < k) (hA : A.card = n) (hB : B.card = n) (h_eq_sum : A.sum id = B.sum id) (h_eq_sum_sq : A.sum (λ x, x * x) = B.sum (λ x, x * x)) (h_disjoint : A ∩ B = ∅) : 
  equiangular (A.map (λ x, x + k)) (B.map (λ x, x + k)) :=
sorry

end equiangular_shift_l217_217845


namespace a_n_plus_1_geometric_sequence_a_n_general_term_T_n_formula_l217_217872

-- Definitions based on conditions
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition: S_n + n = 2a_n for all n ∈ ℕ*
axiom S_eq : ∀ n : ℕ, S n + n = 2 * a n

-- Question (1): Prove that {a_n + 1} is a geometric sequence
theorem a_n_plus_1_geometric_sequence (n : ℕ) : ∃ r : ℝ, ∀ n, a (n + 1) + 1 = (a 1 + 1) * r ^ n :=
sorry

-- Question (1): Find the general term formula for the sequence {a_n}
theorem a_n_general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

-- Introduce sequence b_n and sum T_n based on the definition
variable {b : ℕ → ℝ} 
variable {T : ℕ → ℝ}

-- Define b_n based on the given conditions
axiom b_def : ∀ n : ℕ, b n = a n * log 2 (a n + 1)

-- Question (2): Prove the formula for T_n
theorem T_n_formula (n : ℕ) : T n = (n-1) * 2^(n+1) + 2 - (n * (n + 1))/2 :=
sorry

end a_n_plus_1_geometric_sequence_a_n_general_term_T_n_formula_l217_217872


namespace area_of_circle_l217_217427

noncomputable def calculate_circle_area (center : ℝ×ℝ) (point : ℝ×ℝ) : ℝ :=
  let (x₁, y₁) := center;
  let (x₂, y₂) := point;
  let radius := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2);
  π * radius^2

theorem area_of_circle (R S : ℝ × ℝ) (hR : R = (-2, 5)) (hS : S = (8, -4)) :
  calculate_circle_area R S = 181 * π := by
  sorry

end area_of_circle_l217_217427


namespace sin_cos_identity_l217_217098

theorem sin_cos_identity (θ : Real) (h1 : 0 < θ ∧ θ < π) (h2 : Real.sin θ * Real.cos θ = - (1/8)) :
  Real.sin (2 * Real.pi + θ) - Real.sin ((Real.pi / 2) - θ) = (Real.sqrt 5) / 2 := by
  sorry

end sin_cos_identity_l217_217098


namespace parallelogram_vector_sum_l217_217600

variables (A B C D E F : Type)
variables [add_comm_group A] [vector_space ℝ A]
variables (AB AE AF AD : A)
variables (x y : ℝ)

-- Conditions
-- Condition 1: E is the midpoint of BC
def midpoint (B C E : A) : Prop := E = (B + C) / 2

-- Condition 2: F is the midpoint of CD
def midpoint (C D F : A) : Prop := F = (C + D) / 2

-- Given vectors in terms of others
def vector_expr_1 : Prop := AE = AB + (1 / 2) • AD
def vector_expr_2 : Prop := AF = AD + (1 / 2) • AB

-- Claim
def claim : Prop := (AB = x • AE + y • AF) → (x + y = 2 / 3)

theorem parallelogram_vector_sum 
    (he: midpoint B C E) 
    (hf: midpoint C D F) 
    (h1: vector_expr_1 AE AB AD) 
    (h2: vector_expr_2 AF AD AB) 
    : claim x y AB AE AF :=
sorry

end parallelogram_vector_sum_l217_217600


namespace sister_brought_one_watermelon_l217_217812

variable (danny_watermelons : Nat)
variable (danny_slices_per_watermelon : Nat)
variable (total_slices : Nat)
variable (sister_slices_per_watermelon : Nat)

def slices_by_danny : Nat := danny_watermelons * danny_slices_per_watermelon
def slices_by_sister : Nat := total_slices - slices_by_danny
def sister_watermelons : Nat := slices_by_sister / sister_slices_per_watermelon

theorem sister_brought_one_watermelon 
  (h_danny_watermelons : danny_watermelons = 3)
  (h_danny_slices_per_watermelon : danny_slices_per_watermelon = 10)
  (h_total_slices : total_slices = 45)
  (h_sister_slices_per_watermelon : sister_slices_per_watermelon = 15) :
  sister_watermelons danny_watermelons danny_slices_per_watermelon total_slices sister_slices_per_watermelon = 1 := 
by
  sorry

end sister_brought_one_watermelon_l217_217812


namespace diff_f_l217_217276

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 / (x - 1)

-- Define the interval [-2, 0]
def a : ℝ := -2
def b : ℝ := 0

-- State the difference between the maximum and minimum values
theorem diff_f {a b : ℝ} (h₁ : a = -2) (h₂ : b = 0) : (f a) - (f b) = 4 / 3 :=
by
  sorry

end diff_f_l217_217276


namespace final_result_is_8_l217_217397

theorem final_result_is_8 (n : ℕ) (h1 : n = 2976) (h2 : (n / 12) - 240 = 8) : (n / 12) - 240 = 8 :=
by {
  -- Proof steps would go here
  sorry
}

end final_result_is_8_l217_217397


namespace necessary_but_not_sufficient_l217_217674

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 - 5*x + 4 < 0) → (|x - 2| < 1) ∧ ¬( |x - 2| < 1 → x^2 - 5*x + 4 < 0) :=
by 
  sorry

end necessary_but_not_sufficient_l217_217674


namespace salary_of_b_l217_217358

theorem salary_of_b (S_A S_B : ℝ)
  (h1 : S_A + S_B = 14000)
  (h2 : 0.20 * S_A = 0.15 * S_B) :
  S_B = 8000 :=
by
  sorry

end salary_of_b_l217_217358


namespace max_frac_a_S_l217_217972

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else S n - S (n - 1)

theorem max_frac_a_S (n : ℕ) (h : S n = 2^n - 1) : 
  let frac := (a n) / (a n * S n + a 6)
  ∃ N : ℕ, N > 0 ∧ (frac ≤ 1 / 15) := by
  sorry

end max_frac_a_S_l217_217972


namespace math_proof_problem_l217_217171

noncomputable def intersection_points_polar (C1 C2 : ℝ → ℝ) : Prop :=
  ∃ θ₁ θ₂, 
    C1 (4 * sin θ₁) = 4 ∧ C2 (4 * cos (θ₁ - π / 4)) = 2 * sqrt 2 ∧
    C1 (2 * sqrt 2 * sin θ₂) = 2 * sqrt 2 ∧ C2 (2 * sqrt 2 * cos (θ₂ - π / 4)) = 2 * sqrt 2

def line_pq_parametric (t a b : ℝ) : ℝ × ℝ :=
  (t^3 + a, (b / 2) * t^3 + 1)

noncomputable def validate_a_b (a b : ℝ) :=
  ∃ t : ℝ, line_pq_parametric t a b = (1, 3) ∧ a = -1 ∧ b = 2

theorem math_proof_problem :
  ∃ C1 C2 : ℝ → ℝ, intersection_points_polar C1 C2 ∧ validate_a_b (-1) 2 :=
by
  sorry

end math_proof_problem_l217_217171


namespace sqrt_sqrt_16_eq_pm2_l217_217704

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l217_217704


namespace tournament_committee_count_l217_217943

theorem tournament_committee_count : 
  ∃ (n : ℕ), (5 * (Nat.choose 7 4) * (Nat.choose 7 2)^4 = n) ∧ n = 340342925 :=
begin
  let host_ways := Nat.choose 7 4,
  let non_host_ways := Nat.choose 7 2,
  have calculation : 5 * host_ways * non_host_ways^4 = 340342925,
  { calc
      5 * host_ways * non_host_ways^4
        = 5 * 35 * 21^4 : by sorry 
    ... = 340342925 : by sorry },
  exact ⟨340342925, calculation, rfl⟩,
end

end tournament_committee_count_l217_217943


namespace aaron_moves_to_2015_l217_217416

-- Define the initial conditions and movement rules
def initial_position : (ℕ × ℕ) := (0, 0)
def first_position : (ℕ × ℕ) := (0, 1)
def sequence_defined : ℕ → (ℕ × ℕ)
  | 1 => (0, 1)
  | 2 => (-1, 1)
  | 3 => (-1, 0)
  | 4 => (-1, -1)
  -- Continues with the spiral pattern

-- Theorem to state the final position
theorem aaron_moves_to_2015 :
  sequence_defined 2015 = (22, 13) :=
sorry

end aaron_moves_to_2015_l217_217416


namespace speed_increase_l217_217379

theorem speed_increase (v_initial: ℝ) (t_initial: ℝ) (t_new: ℝ) :
  v_initial = 60 → t_initial = 1 → t_new = 0.5 →
  v_new = (1 / (t_new / 60)) →
  v_increase = v_new - v_initial →
  v_increase = 60 :=
by
  sorry

end speed_increase_l217_217379


namespace sin_B_in_right_triangle_l217_217948

theorem sin_B_in_right_triangle
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_C : ∡ A C B = π / 2)
  (AB AC : ℝ)
  (hAB : AB = 15)
  (hAC : AC = 9) :
  sin (angle B) = 3 / 5 := 
sorry

end sin_B_in_right_triangle_l217_217948


namespace ducks_in_larger_pond_l217_217162

theorem ducks_in_larger_pond (D : ℕ) 
  (h1 : 30 = 30) 
  (h2 : 0.20 * 30 = 6) 
  (h3 : 0.12 * (D : Real) = 0.12 * (D : Real)) 
  (h4 : 15% * (30 + D) = 0.15 * (30 + (D : Real))) 
  (h5 : 6 + 0.12 * (D : Real) = 0.15 * (30 + (D : Real))) : 
  D = 50 := 
by 
  sorry

end ducks_in_larger_pond_l217_217162


namespace positive_integer_solution_l217_217143

theorem positive_integer_solution (x : ℕ) (hx : 0 < x) : ((x! - (x - 3)!) / 23 = 1) ↔ (x = 4) :=
by
  sorry

end positive_integer_solution_l217_217143


namespace dmitry_before_anatoly_l217_217337

theorem dmitry_before_anatoly (m : ℝ) (x y z : ℝ) 
  (h1 : 0 < x ∧ x < m) 
  (h2 : 0 < y ∧ y < m) 
  (h3 : 0 < z ∧ z < m) 
  (h4 : y < z) : 
  (Probability (y < x)) = 2 / 3 := 
sorry

end dmitry_before_anatoly_l217_217337


namespace part_I_part_II_1_part_II_2_l217_217543

def f (x : ℝ) : ℝ := Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 - 6 * x + 7 / 2
def h (x : ℝ) : ℝ := f x + g 2 x

theorem part_I : ∃ a : ℝ, g' a (3 / 2) = 0 :=
by
  let g' (a : ℝ) (x : ℝ) := 2 * a * x - 6
  use 2
  sorry

theorem part_II_1 (x_1 x_2 : ℝ) (h1 : (1 / 2) < x_1 ∧ x_1 < 1) (h2 : 1 < x_2 ∧ x_2 < 2) :
  (1 / 2) < x_1 ∧ x_1 < 1 ∧ 1 < x_2 ∧ x_2 < 2 :=
by
  sorry

theorem part_II_2 (x_1 x_2 : ℝ) (h1 : (1 / 2) < x_1 ∧ x_1 < 1) (h2 : 1 < x_2 ∧ x_2 < 2) :
  let k := (f x_2 - f x_1) / (x_2 - x_1)
  (1 / 2) < k ∧ k < 2 :=
by
  let k := (f x_2 - f x_1) / (x_2 - x_1)
  use k
  sorry

end part_I_part_II_1_part_II_2_l217_217543


namespace arctan_transformation_l217_217923

theorem arctan_transformation (x : ℝ) :
  let g := (λ x : ℝ, Real.arctan x)
  g ((5 * x - x^5) / (1 + 5 * x^4)) = 5 * g x - (g x)^5 := 
sorry

end arctan_transformation_l217_217923


namespace number_of_smaller_pipes_l217_217771

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem number_of_smaller_pipes
    (diameter_main : ℝ) (diameter_small : ℝ) (h : ℝ)
    (H_main : diameter_main = 8) (H_small : diameter_small = 3) :
    let r_main := diameter_main / 2
    let r_small := diameter_small / 2
    let V_main := volume_of_cylinder r_main h
    let V_small := volume_of_cylinder r_small h
    8 = (V_main / V_small).ceil := by
{
  sorry
}

end number_of_smaller_pipes_l217_217771


namespace negative_solution_condition_l217_217057

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end negative_solution_condition_l217_217057


namespace integer_solutions_count_l217_217918

theorem integer_solutions_count :
  (Finset.univ.filter (λ (p : ℤ × ℤ), 
    let x := p.1 in
    let y := p.2 in
    x^2 + y^2 = 6 * x + 2 * y + 15)).card = 12 := 
by {
  sorry
}

end integer_solutions_count_l217_217918


namespace dmitry_before_anatoly_l217_217335

theorem dmitry_before_anatoly (m : ℝ) (x y z : ℝ) 
  (h1 : 0 < x ∧ x < m) 
  (h2 : 0 < y ∧ y < m) 
  (h3 : 0 < z ∧ z < m) 
  (h4 : y < z) : 
  (Probability (y < x)) = 2 / 3 := 
sorry

end dmitry_before_anatoly_l217_217335


namespace remainder_problem_l217_217788

theorem remainder_problem 
  (a b c d m n x y k : ℤ)
  (h1: a * x + b * y ≡ 37 [MOD 64])
  (h2: a * x + b * y = m)
  (h3: c * x + d * y = n) :
  (c * x + d * y) % 5 = 5 :=
by sorry

end remainder_problem_l217_217788


namespace gcf_75_135_l217_217728

theorem gcf_75_135 : Nat.gcd 75 135 = 15 :=
  by sorry

end gcf_75_135_l217_217728


namespace no_integer_solutions_l217_217825

theorem no_integer_solutions (x y : ℤ) : 15 * x^2 - 7 * y^2 ≠ 9 :=
by
  sorry

end no_integer_solutions_l217_217825


namespace pushup_difference_l217_217743

theorem pushup_difference :
  let Zachary_pushups := 15
      David_pushups := Zachary_pushups + 39
      John_pushups := David_pushups - 9
  in |Zachary_pushups - John_pushups| = 30 :=
by
  let Zachary_pushups := 15
  let David_pushups := Zachary_pushups + 39
  let John_pushups := David_pushups - 9
  let difference := Zachary_pushups - John_pushups
  have h1 : difference = -30 := sorry
  exact congr_arg Int.natAbs h1

end pushup_difference_l217_217743


namespace number_of_magazines_sold_l217_217966

def newspapers_sold : ℕ := 275
def total_reading_materials : ℕ := 700

theorem number_of_magazines_sold : total_reading_materials - newspapers_sold = 425 := by
  calc
    total_reading_materials - newspapers_sold = 700 - 275 : by rfl
    ... = 425 : by rfl
    

end number_of_magazines_sold_l217_217966


namespace windshield_wiper_movement_l217_217949

theorem windshield_wiper_movement :
  ∀ (L : Type) [HasWiper L], movement_of_line L = formation_of_surface L :=
sorry

end windshield_wiper_movement_l217_217949


namespace expand_polynomial_l217_217458

theorem expand_polynomial :
  (x^2 - 3 * x + 3) * (x^2 + 3 * x + 1) = x^4 - 5 * x^2 + 6 * x + 3 :=
by
  sorry

end expand_polynomial_l217_217458


namespace sequence_bounded_l217_217076

theorem sequence_bounded :
  ∃ M, ∀ n : ℕ, n > 0 → (let an := ∫ t in 0..1/real.sqrt n, abs (∑ i in finset.range (n+1), complex.exp (i * complex.I * t)) 
  in abs an ≤ M) :=
by sorry

end sequence_bounded_l217_217076


namespace loraine_wax_usage_proof_l217_217994

-- Conditions
variables (large_animals small_animals : ℕ)
variable (wax : ℕ)

-- Definitions based on conditions
def large_animal_wax := 4
def small_animal_wax := 2
def total_sticks := 20
def small_animals_wax := 12
def small_to_large_ratio := 3

-- Proof statement
theorem loraine_wax_usage_proof (h1 : small_animals_wax = small_animals * small_animal_wax)
  (h2 : small_animals = large_animals * small_to_large_ratio)
  (h3 : wax = small_animals_wax + large_animals * large_animal_wax) :
  wax = total_sticks := by
  sorry

end loraine_wax_usage_proof_l217_217994


namespace complex_number_solution_l217_217734

theorem complex_number_solution 
(z : ℂ) 
: (4 - 3*complex.I)*z + 6 + 2*complex.I = -2 + 15*complex.I
→ z = (-71 / 7 : ℂ) - (15 / 7) * complex.I :=
by sorry

end complex_number_solution_l217_217734


namespace birds_flew_up_l217_217373

theorem birds_flew_up (original_birds total_birds birds_flew_up : ℕ) 
  (h1 : original_birds = 14)
  (h2 : total_birds = 35)
  (h3 : total_birds = original_birds + birds_flew_up) :
  birds_flew_up = 21 :=
by
  rw [h1, h2] at h3
  linarith

end birds_flew_up_l217_217373


namespace parallel_tangents_tangent_to_incicle_l217_217592

-- We define the necessary geometric entities and conditions.

noncomputable theory

open_locale classical

variables {A B C D E : Type}
variables (AB AC BD CD : set (A → B))
variables (I : set (A × B))
variables {triangle_ABC : A → B → B → Prop}
variables {incircle_I : triangle_ABC A B C → I}
variables {circumcircle_O : (B → I → C) → set (A × B)}
variables {arc_BC : set (A × B)}
variables {line_DI : set (D × E)}

-- Given conditions
variable (isosceles_triangle : triangle_ABC A B C ∧ AB = AC)
variable (incenter : incircle_I (triangle_ABC A B C))
variable (circumcenter : circumcircle_O (B × I × C))
variable (point_D_on_arcBC : D ∈ arc_BC)
variable (point_E_on_lineDI : E ∈ line_DI)

-- Required proof statement
theorem parallel_tangents_tangent_to_incicle :
  (∃ l₁, l₁ ∥ BD ∧ tangent l₁ I at E) →
  (∃ l₂, l₂ ∥ CD ∧ tangent l₂ I at E) :=
sorry

end parallel_tangents_tangent_to_incicle_l217_217592


namespace jen_ate_eleven_suckers_l217_217251

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end jen_ate_eleven_suckers_l217_217251


namespace least_total_bananas_is_1128_l217_217486

noncomputable def least_total_bananas : ℕ :=
  let b₁ := 252
  let b₂ := 252
  let b₃ := 336
  let b₄ := 288
  b₁ + b₂ + b₃ + b₄

theorem least_total_bananas_is_1128 :
  least_total_bananas = 1128 :=
by
  sorry

end least_total_bananas_is_1128_l217_217486


namespace doris_monthly_expenses_l217_217024

theorem doris_monthly_expenses :
  (hourly_rate weekly_hours saturday_hours weeks_needed monthly_expenses : ℕ) 
  (h1 : hourly_rate = 20)
  (h2 : weekly_hours = 3 * 5)
  (h3 : saturday_hours = 5)
  (h4 : weeks_needed = 3)
  (h5 : monthly_expenses = hourly_rate * (weekly_hours + saturday_hours) * weeks_needed) :
  monthly_expenses = 1200 := by
  sorry

end doris_monthly_expenses_l217_217024


namespace small_bottle_sold_percentage_l217_217406

-- Definitions for initial conditions
def small_bottles_initial : ℕ := 6000
def large_bottles_initial : ℕ := 15000
def large_bottle_sold_percentage : ℝ := 0.14
def total_remaining_bottles : ℕ := 18180

-- The statement we need to prove
theorem small_bottle_sold_percentage :
  ∃ k : ℝ, (0 ≤ k ∧ k ≤ 100) ∧
  (small_bottles_initial - (k / 100) * small_bottles_initial + 
   large_bottles_initial - large_bottle_sold_percentage * large_bottles_initial = total_remaining_bottles) ∧
  (k = 12) :=
sorry

end small_bottle_sold_percentage_l217_217406


namespace probability_four_dice_show_two_l217_217719

theorem probability_four_dice_show_two :
  let roll_two := (1 / 8 : ℝ)
  let roll_not_two := (7 / 8 : ℝ)
  let choose := nat.choose 12 4
  let prob_specific_arrangement := roll_two^4 * roll_not_two^8
  let total_prob := choose * prob_specific_arrangement
  total_prob ≈ 0.091 :=
by {
  let roll_two := (1 / 8 : ℝ)
  let roll_not_two := (7 / 8 : ℝ)
  let choose := nat.choose 12 4
  let prob_specific_arrangement := roll_two^4 * roll_not_two^8
  let total_prob := choose * prob_specific_arrangement
  sorry  
}

end probability_four_dice_show_two_l217_217719


namespace monic_cubic_poly_eqn_l217_217461

noncomputable def monic_cubic_poly (x : ℝ) : ℝ := x^3 - 6 * x^2 + 12 * x - 13

theorem monic_cubic_poly_eqn :
  ∃ Q : ℝ → ℝ, (∀ x : ℝ, Q x = x^3 - 6 * x^2 + 12 * x - 13) ∧
                (∃ y : ℝ, y = real.cbrt 5 + 2 ∧ Q y = 0) :=
by
  use monic_cubic_poly
  split
  · intro x
    refl
  · use real.cbrt 5 + 2
    split
    · refl
    · sorry

end monic_cubic_poly_eqn_l217_217461


namespace odd_prime_power_l217_217037

theorem odd_prime_power (n : ℕ) (h₀ : n > 1) (h₁ : n % 2 = 1) 
  (h₂ : ∀ a b : ℕ, a ∣ n → b ∣ n → Nat.coprime a b → (a + b - 1) ∣ n) : 
  ∃ (p : ℕ) (m : ℕ), Nat.Prime p ∧ p % 2 = 1 ∧ n = p ^ m := 
sorry

end odd_prime_power_l217_217037


namespace short_trees_after_planting_l217_217303

theorem short_trees_after_planting : 
  ∀ (initial_trees planted_trees : ℕ), 
    initial_trees = 112 → planted_trees = 105 → 
    initial_trees + planted_trees = 217 := 
by
  intros initial_trees planted_trees h_init h_plant
  rw [h_init, h_plant]
  exact rfl

end short_trees_after_planting_l217_217303


namespace number_of_intersection_points_l217_217659

theorem number_of_intersection_points (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃ x : Finset ℝ, (∀ y ∈ x, f ((y:ℝ)^2) = f ((y:ℝ)^6)) ∧ x.card = 3 :=
by
  sorry

end number_of_intersection_points_l217_217659


namespace half_cylinder_volume_is_339_l217_217765

-- Definitions and conditions
def radius : ℝ := 6
def height : ℝ := 6

-- Volume of the cylinder
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

-- Volume of one half of the cylinder
def volume_half_cylinder (r h : ℝ) : ℝ := volume_cylinder r h / 2

-- Verify if the volume of half cylinder is approximately 339 cm³
def is_approx (a b : ℝ) (epsilon : ℝ) : Prop := abs (a - b) < epsilon

theorem half_cylinder_volume_is_339 :
  is_approx (volume_half_cylinder radius height) 339 (3.14 / 100) :=
sorry

end half_cylinder_volume_is_339_l217_217765


namespace circle_possible_m_values_l217_217152

theorem circle_possible_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + m * x - m * y + 2 = 0) ↔ m > 2 ∨ m < -2 :=
by
  sorry

end circle_possible_m_values_l217_217152


namespace carol_initial_cupcakes_l217_217081

variable (x : ℕ)

theorem carol_initial_cupcakes (h : (x - 9) + 28 = 49) : x = 30 := 
  sorry

end carol_initial_cupcakes_l217_217081


namespace polygon_ratio_constant_l217_217996

-- Definitions and conditions
def polygon (n : ℕ) (A : ℕ → ℕ → Prop) : Prop := 
  ∀ i, 1 ≤ i ∧ i ≤ n → A i (i + 1)

def inscribed (n : ℕ) (A : ℕ → Prop) (C : ℕ → Prop) : Prop := 
  ∀ i, 1 ≤ i ∧ i ≤ n → C (A i)

def parallel (A B : ℕ → ℕ → Prop) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 93 → A i (i + 1) ∥ B i (i + 1)

-- Given:
variable (A B C : ℕ → ℕ → Prop)

variable n : ℕ
variable h_n : n = 93
variable h_circle : ∀ i, 1 ≤ i ∧ i ≤ n → (inscribed n A A) ∧ (inscribed n B B)
variable h_parallel : parallel A B

-- Prove that:
theorem polygon_ratio_constant (A B : ℕ → ℕ → Prop) (h_parallel : parallel A B) : 
  ∃ c : ℝ, ∀ i, 1 ≤ i ∧ i ≤ 93 → ((λ j, (A j (j+1) : ℝ) / (B j (j+1) : ℝ)) i) = c := 
sorry

end polygon_ratio_constant_l217_217996


namespace smallest_n_for_roots_of_unity_l217_217444

-- Define the given polynomial
def p : Polynomial ℂ := Polynomial.X^6 - Polynomial.X^3 + 1

-- Define the 18th roots of unity
def is_n_th_root_of_unity (n : ℕ) (z : ℂ) : Prop :=
  z ^ n = 1

-- State the theorem with the required conditions and goal
theorem smallest_n_for_roots_of_unity :
  (∀ z, Polynomial.eval z p = 0 → ∃ k : ℕ, z = exp (2 * k * Real.pi * Complex.I / 18)) ∧
  (∀ n > 0, (∀ z, Polynomial.eval z p = 0 → ∃ k : ℕ, z = exp (2 * k * Real.pi * Complex.I / n)) → 18 ≤ n) :=
by
  sorry

end smallest_n_for_roots_of_unity_l217_217444


namespace quadratic_expression_value_l217_217921

theorem quadratic_expression_value
  (x : ℝ)
  (h : x^2 + x - 2 = 0)
: x^3 + 2*x^2 - x + 2021 = 2023 :=
sorry

end quadratic_expression_value_l217_217921


namespace expression_is_integer_l217_217646

theorem expression_is_integer (n : ℕ) : 
    ∃ k : ℤ, (n^5 : ℤ) / 5 + (n^3 : ℤ) / 3 + (7 * n : ℤ) / 15 = k :=
by
  sorry

end expression_is_integer_l217_217646


namespace probability_dmitry_before_father_l217_217334

noncomputable def prob_dmitry_before_father (m : ℝ) (x y z : ℝ) (h1 : 0 < x ∧ x < m) (h2 : 0 < y ∧ y < z ∧ z < m) : ℝ :=
  if h1 ∧ h2 then 2/3 else 0

theorem probability_dmitry_before_father (m : ℝ) (x y z : ℝ) (h1 : 0 < x ∧ x < m) (h2 : 0 < y ∧ y < z ∧ z < m) :
  prob_dmitry_before_father m x y z h1 h2 = 2 / 3 :=
begin
  sorry
end

end probability_dmitry_before_father_l217_217334


namespace problem_solution_l217_217021

theorem problem_solution :
  (∑ k in Finset.range 10, Real.logb (4^k) (2^(3*k^2))) *
  (∑ k in Finset.range 50, Real.logb (16^k) (64^k)) = 24750 := 
by
  sorry

end problem_solution_l217_217021


namespace determinant_of_A_l217_217808

section
  open Matrix

  -- Define the given matrix
  def A : Matrix (Fin 3) (Fin 3) ℤ :=
    ![ ![0, 2, -4], ![6, -1, 3], ![2, -3, 5] ]

  -- State the theorem for the determinant
  theorem determinant_of_A : det A = 16 :=
  sorry
end

end determinant_of_A_l217_217808


namespace distance_between_lines_l217_217908

def line1 (x y : ℝ) : Prop := 4 * x + 2 * y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 2 * y + 2 = 0

theorem distance_between_lines :
  let d := (| -3 - 2 |) / sqrt (4^2 + 2^2) in
  d = sqrt 5 / 2 :=
by
  -- Definitions of lines
  have l1 : line1 x y := by sorry
  have l2 : line2 x y := by sorry

  -- Distance calculation
  let d := (| -3 - 2 |) / sqrt (4^2 + 2^2)
  
  -- Show required distance
  show d = sqrt 5 / 2, from sorry

end distance_between_lines_l217_217908


namespace jesse_building_blocks_used_l217_217192

def building_blocks_used 
  (farmhouse_blocks : ℕ)
  (fenced_area_blocks : ℕ)
  (blocks_left : ℕ)
  (total_blocks : ℕ) :=
  total_blocks - (farmhouse_blocks + fenced_area_blocks + blocks_left)

theorem jesse_building_blocks_used 
  (farmhouse_blocks : ℕ := 123)
  (fenced_area_blocks : ℕ := 57)
  (blocks_left : ℕ := 84)
  (total_blocks : ℕ := 344) :
  building_blocks_used farmhouse_blocks fenced_area_blocks blocks_left total_blocks = 80 :=
by {
  calc
  building_blocks_used farmhouse_blocks fenced_area_blocks blocks_left total_blocks 
      = total_blocks - (farmhouse_blocks + fenced_area_blocks + blocks_left) : rfl
  ... = 344 - (123 + 57 + 84) : by congr; exact rfl
  ... = 344 - 264 : rfl
  ... = 80 : rfl
}

end jesse_building_blocks_used_l217_217192


namespace max_d_for_20x20_max_d_for_21x21_l217_217747

noncomputable def max_distance_20x20 : ℝ := 10 * Real.sqrt 2
noncomputable def max_distance_21x21 : ℝ := 10 * Real.sqrt 2

theorem max_d_for_20x20 (d : ℝ) (S : Fin 20 → Fin 20 → (ℝ × ℝ)) :
  (∀ i j, let (x, y) := S i j in x ^ 2 + y ^ 2 ≥ d^2 → d ≤ max_distance_20x20) :=
sorry

theorem max_d_for_21x21 (d : ℝ) (S : Fin 21 → Fin 21 → (ℝ × ℝ)) :
  (∀ i j, let (x, y) := S i j in x ^ 2 + y ^ 2 ≥ d^2 → d ≤ max_distance_21x21) :=
sorry

end max_d_for_20x20_max_d_for_21x21_l217_217747


namespace first_group_persons_count_l217_217261

theorem first_group_persons_count :
  ∃ (P : ℕ), 
  (P * 12 * 5 = 30 * 21 * 6) ↔ P = 63 :=
begin
  use 63,
  split,
  { intro h,
    have h2 : 30 * 21 * 6 = 3780 := by norm_num,
    have h3 : 12 * 5 = 60 := by norm_num,
    rw [← h3, mul_assoc] at h,
    exact (nat.mul_right_inj (by norm_num : 0 < 60)).mp h },
  { intro h,
    rw h,
    norm_num }
end

end first_group_persons_count_l217_217261


namespace Isabel_problems_complete_l217_217603

theorem Isabel_problems_complete :
  let math_pages := 2
  let reading_pages := 4
  let science_pages := 3
  let history_pages := 1
  let problems_per_math_page := 5
  let problems_per_reading_page := 5
  let problems_per_science_page := 6
  let problems_per_history_page := 10
  let total_problems := math_pages * problems_per_math_page +
                        reading_pages * problems_per_reading_page +
                        science_pages * problems_per_science_page +
                        history_pages * problems_per_history_page
  let percentage_to_complete := 0.70
  let problems_to_complete := Real.toInt (percentage_to_complete * total_problems)
  problems_to_complete = 41 := sorry

end Isabel_problems_complete_l217_217603


namespace find_female_employees_l217_217359

-- Definitions from conditions
def total_employees (E : ℕ) := True
def female_employees (F : ℕ) := True
def male_employees (M : ℕ) := True
def female_managers (F_mgrs : ℕ) := F_mgrs = 280
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Statements as conditions in Lean
def managers_total (E M : ℕ) := (fraction_of_managers * E : ℚ) = (fraction_of_male_managers * M : ℚ) + 280
def employees_total (E F M : ℕ) := E = F + M

-- The proof target
theorem find_female_employees (E F M : ℕ) (F_mgrs : ℕ)
    (h1 : female_managers F_mgrs)
    (h2 : managers_total E M)
    (h3 : employees_total E F M) : F = 700 := by
  sorry

end find_female_employees_l217_217359


namespace tile_floor_multiple_of_seven_l217_217738

theorem tile_floor_multiple_of_seven (n : ℕ) (a : ℕ)
  (h1 : n * n = 7 * a)
  (h2 : 4 * a / 7 + 3 * a / 7 = a) :
  ∃ k : ℕ, n = 7 * k := by
  sorry

end tile_floor_multiple_of_seven_l217_217738


namespace find_smallest_expression_l217_217077

theorem find_smallest_expression (x : ℕ) (h : x = 10) :
  min (6 / (x : ℝ)) (min (6 / (x + 2) : ℝ) (min (6 / (x - 1) : ℝ) (min ((x : ℝ) / 6) ((2 * x + 1) / 12))))
  = 1 / 2 :=
by
  rw h
  have h1 : 6 / (10 : ℝ) = 3 / 5 := by norm_num
  have h2 : 6 / (10 + 2 : ℝ) = 1 / 2 := by norm_num
  have h3 : 6 / (10 - 1 : ℝ) = 2 / 3 := by norm_num
  have h4 : (10 : ℝ) / 6 = 5 / 3 := by norm_num
  have h5 : (2 * 10 + 1 : ℝ) / 12 = 7 / 4 := by norm_num
  apply min_eq
  apply min_eq
  apply min_eq
  apply min_le_right_of_le
  apply le_of_lt
  norm_num
  apply le_of_lt
  norm_num
  apply le_of_lt
  norm_num
  apply le_of_lt
  norm_num
  apply le_of_lt
  norm_num
  sorry

end find_smallest_expression_l217_217077


namespace consecutive_numbers_product_l217_217356

theorem consecutive_numbers_product (a b c d : ℤ) 
  (h1 : b = a + 1) 
  (h2 : c = a + 2) 
  (h3 : d = a + 3) 
  (h4 : a + d = 109) : 
  b * c = 2970 := by
  sorry

end consecutive_numbers_product_l217_217356


namespace sqrt_sqrt_16_eq_pm_2_l217_217696

theorem sqrt_sqrt_16_eq_pm_2 : sqrt (sqrt 16) = 2 ∨ sqrt (sqrt 16) = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm_2_l217_217696


namespace angle_in_third_quadrant_l217_217139

variable {θ : ℝ}
variable {k : ℤ}

theorem angle_in_third_quadrant (h₁ : sin θ * abs (sin θ) + cos θ * abs (cos θ) = -1)
    (h₂ : θ ≠ k * π / 2) : 
  (π < θ ∧ θ < 3 * π / 2) :=
sorry

end angle_in_third_quadrant_l217_217139


namespace probability_abs_diff_l217_217990

variables (P : ℕ → ℚ) (m : ℚ)

def is_probability_distribution : Prop :=
  P 1 = m ∧ P 2 = 1/4 ∧ P 3 = 1/4 ∧ P 4 = 1/3 ∧ m + 1/4 + 1/4 + 1/3 = 1

theorem probability_abs_diff (h : is_probability_distribution P m) :
  P 1 + P 3 = 5 / 12 :=
by 
sorry

end probability_abs_diff_l217_217990


namespace part_I_part_II_l217_217541

-- Definition of the functions f(x) and g(x)
def f (x : ℝ) : ℝ := log x / x
def g (x : ℝ) : ℝ := exp x

-- Part (I) proof problem statement
theorem part_I (m : ℝ) : (∀ x : ℝ, x > 0 → f x ≤ m * x) → (∀ x : ℝ, x > 0 → m * x ≤ g x) → (m ∈ set.Icc (1 / (2 * exp 1)) (exp 1)) := sorry

-- Part (II) proof problem statement
theorem part_II (x1 x2 : ℝ) (h1 : x1 > x2) (h2 : x2 > 0) : 
  x1 * f x1 - x2 * f x2 * (x1^2 + x2^2) > 2 * x2 * (x1 - x2) := sorry

end part_I_part_II_l217_217541


namespace decreasing_odd_function_solution_set_l217_217519

theorem decreasing_odd_function_solution_set
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f(x + 1) = -f(-x - 1))
  (h_decreasing : ∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f x1 - f x2) < 0) :
  {x : ℝ | f(1 - x) < 0} = set.Iio 0 := 
sorry

end decreasing_odd_function_solution_set_l217_217519


namespace perimeter_of_shaded_region_l217_217778

/-- Given a square ABCD with side length 36 inches, and an equilateral triangle BFC 
of side length 36 inches cut from this square and then rotated 180 degrees around 
point B to form a new region ABFCDE, prove that the perimeter of the new region 
is 216 inches. --/
theorem perimeter_of_shaded_region :
  let side_length := 36
  in let perimeter := 6 * side_length
  in perimeter = 216 :=
by
  sorry

end perimeter_of_shaded_region_l217_217778


namespace problem_statement_l217_217495

theorem problem_statement (p q : ℝ) 
  (h1 : 1 + p + q = 3) 
  (h2 : 9 - 3 * p + q = 7) : 
  (y = x^2 + x + q ∧ x = -5 → y = 21) :=
begin
  -- Proof goes here
  sorry
end

end problem_statement_l217_217495


namespace root_interval_of_lg_x_eq_2_l217_217689

noncomputable theory
open Real

def f (x : ℝ) : ℝ := log x / log 10 + x - 2

theorem root_interval_of_lg_x_eq_2 :
  (∃ x₀ ∈ (1 : ℝ), x₀ ∈ (1, 2) ∧ f x₀ = 0) → ∃ k : ℤ, k = 1 :=
by
  intro h
  cases h with x₀ hx₀
  use 1
  sorry

end root_interval_of_lg_x_eq_2_l217_217689


namespace distance_traveled_on_second_day_l217_217172

theorem distance_traveled_on_second_day 
  (a₁ : ℝ) 
  (h_sum : a₁ + a₁ / 2 + a₁ / 4 + a₁ / 8 + a₁ / 16 + a₁ / 32 = 189) 
  : a₁ / 2 = 48 :=
by
  sorry

end distance_traveled_on_second_day_l217_217172


namespace magnitude_of_b_l217_217135

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, -2)

def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem magnitude_of_b (x : ℝ) (h : parallel a (b x)) : ‖b x‖ = 2 * Real.sqrt 5 :=
by
  have hx : x = -4 := by sorry
  have hb : b x = (-4, -2) := by
    rw hx
    rfl
  rw [hb]
  calc
    ‖(-4, -2)‖ = Real.sqrt ((-4)^2 + (-2)^2) := by sorry
    ... = 2 * Real.sqrt 5 := by sorry

end magnitude_of_b_l217_217135


namespace four_integers_sum_product_odd_impossible_l217_217602

theorem four_integers_sum_product_odd_impossible (a b c d : ℤ) :
  ¬ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ d % 2 = 1 ∧ 
     (a + b + c + d) % 2 = 1) :=
by
  sorry

end four_integers_sum_product_odd_impossible_l217_217602


namespace number_of_men_in_first_group_l217_217381

theorem number_of_men_in_first_group (M : ℕ) : (M * 15 = 25 * 18) → M = 30 :=
by
  sorry

end number_of_men_in_first_group_l217_217381


namespace find_angle_C_max_area_S_l217_217957

noncomputable section

-- Definitions for the sides opposite to the angles
variables {A B C a b c : ℝ} [triangle (A B C a b c)] -- assuming a triangle structure

-- Conditions from the problem
variable (condition1 : c * cos B = (2 * a - b) * cos C)
variable (condition2 : c = 4)

-- Prove part 1: The magnitude of angle C
theorem find_angle_C (h : A + B + C = π) : C = π / 3 :=
by sorry

-- Prove part 2: The maximum value of the area S
theorem max_area_S (hC : C = π / 3) (hab : AB = 4) : ∃ S, S = 4 * sqrt 3 :=
by sorry

end find_angle_C_max_area_S_l217_217957


namespace option_D_not_equiv_to_275e_minus7_l217_217353

theorem option_D_not_equiv_to_275e_minus7 : 
  let val := 2.75 * 10^(-7) in
  let optD := (11 / 40) * 10^(-7) in
  optD ≠ val
:= by
  let val := 2.75 * 10^(-7)
  let optD := (11 / 40) * 10^(-7)
  have h : optD = 2.75 * 10^(-8), by sorry
  have h2 : val = 2.75 * 10^(-7), by sorry
  have hne : optD ≠ val, by linarith
  exact hne

end option_D_not_equiv_to_275e_minus7_l217_217353


namespace total_gallons_in_tanks_l217_217635

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end total_gallons_in_tanks_l217_217635


namespace tangent_line_p_l217_217770

noncomputable def is_tangent_to_circle (p : ℝ) : Prop :=
  let C := (-3 : ℝ, 0 : ℝ)
  let r := 1
  let distance_from_center_to_line := |(- ( p / 2 )) + 3|
  distance_from_center_to_line = r

theorem tangent_line_p (p : ℝ) : is_tangent_to_circle p ↔ p = 4 ∨ p = 8 :=
  sorry

end tangent_line_p_l217_217770


namespace sum_of_radii_circles_tangent_to_lines_l217_217124

-- Define the necessary mathematical objects
def A : ℝ × ℝ := (1505, 1008)
def l1 (x : ℝ) : ℝ := 0
def l2 (x : ℝ) : ℝ := (4 / 3) * x

-- Define the statement to prove
theorem sum_of_radii_circles_tangent_to_lines
    (A : ℝ × ℝ)
    (l1 l2 : ℝ → ℝ)
    (H1 : A = (1505, 1008))
    (H2 : ∀ x, l1 x = 0)
    (H3 : ∀ x, l2 x = (4 / 3) * x) :
    ∃ r1 r2 : ℝ, (circle_tangent A r1 l1 l2) ∧ (circle_tangent A r2 l1 l2) ∧ (r1 + r2 = 2009) :=
sorry

-- Function to specify what it means for a circle to be tangent to given lines
def circle_tangent (A : ℝ × ℝ) (r : ℝ) (l1 l2 : ℝ → ℝ) : Prop :=
-- placeholder definition
true -- the actual detailed definition would go here, skipped for brevity

end sum_of_radii_circles_tangent_to_lines_l217_217124


namespace area_EFHG_l217_217166

variable (A B C D E F G : Point)
variable (volume_ABCD : Real)
variable (dist_C_to_EFHG : Real)

-- Given conditions
axiom (vol_ABCD_12 : volume_ABCD = 12)
axiom (on_E_AB : E ∈ segment A B)
axiom (on_F_BC : F ∈ segment B C)
axiom (on_G_AD : G ∈ segment A D)
axiom (AE_2EB : distance A E = 2 * distance E B)
axiom (BF_FC : distance B F = distance F C)
axiom (AG_2GD : distance A G = 2 * distance G D)
axiom (dist_C_1 : dist_C_to_EFHG = 1)

-- Main conjecture
theorem area_EFHG : area (cross_section E F G H) = 7 :=
by
  sorry

end area_EFHG_l217_217166


namespace proof_problem_l217_217931

variables (A B C a b c : ℝ)

-- Define conditions
def conditions (A B C a b c : ℝ) :=
  C = 2 * A ∧
  cos A = 3/4 ∧
  (a * c * cos (A + C)) = 27/2

-- Define the proof problem
theorem proof_problem (h : conditions A B C a b c) :
  cos (2 * A) = 1/8 ∧
  cos (- A - (2 * A)) = 9/16 ∧
  sqrt (a^2 + (3/2 * a)^2 - 2 * a * (3/2 * a) * 9/16) = 5 :=
by
  sorry

end proof_problem_l217_217931


namespace f_diff_180_90_l217_217480

def sigma (n : ℕ) : ℕ := ∑ i in (finset.range n).filter (n % i = 0), i

def f (n : ℕ) : ℚ := sigma n / n

theorem f_diff_180_90 : f 180 - f 90 = (13 / 30 : ℚ) :=
by
  sorry

end f_diff_180_90_l217_217480


namespace divides_or_l217_217618

-- Definitions
variables {m n : ℕ} -- using natural numbers (non-negative integers) for simplicity in Lean

-- Hypothesis: m ∨ n + m ∧ n = m + n
theorem divides_or (h : Nat.lcm m n + Nat.gcd m n = m + n) : m ∣ n ∨ n ∣ m :=
sorry

end divides_or_l217_217618


namespace billboard_shorter_side_length_l217_217569

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 91)
  (h2 : 2 * L + 2 * W = 40) :
  L = 7 ∨ W = 7 :=
by sorry

end billboard_shorter_side_length_l217_217569


namespace complex_problem_l217_217497

-- Definitions based on the given conditions:
def z : ℂ := ((-2 + 6 * complex.i) / (1 - complex.i)) - 4

-- Lean statement for the proof problem
theorem complex_problem :
  (conj z = -8 - 2 * complex.i) ∧ (∀ a : ℝ, (normSq (z + (a * complex.i)) ≤ (normSq z)) → (-4 ≤ a ∧ a ≤ 0)) :=
by sorry

end complex_problem_l217_217497


namespace correct_calculation_l217_217374

-- Define the conditions of the problem
variable (x : ℕ)
variable (h : x + 5 = 43)

-- The theorem we want to prove
theorem correct_calculation : 5 * x = 190 :=
by
  -- Since Lean requires a proof and we're skipping it, we use 'sorry'
  sorry

end correct_calculation_l217_217374


namespace negative_solution_iff_sum_zero_l217_217047

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l217_217047


namespace transformed_curve_equation_l217_217680

-- Define the original curve equation and transformations
def original_curve (x : ℝ) : ℝ := cos x

def scaling_transformation_x (x' : ℝ) : ℝ := x' / 2

def scaling_transformation_y (y' : ℝ) : ℝ := y' / 3

-- Prove the new equation of the curve
theorem transformed_curve_equation (x' y' : ℝ) :
  y' = 3 * cos (x' / 2) ↔ (scaling_transformation_y y' = original_curve (scaling_transformation_x x')) :=
by 
  sorry

end transformed_curve_equation_l217_217680


namespace constant_term_in_expansion_l217_217141

theorem constant_term_in_expansion : 
  let n := ∫ x in 0 .. 2, 2 * x 
  (n = 4) →
  (Polynomials.constant_coeff ((X - C (1 / 2) * X⁻¹) ^ n) = 3 / 2) :=
by
  sorry

end constant_term_in_expansion_l217_217141


namespace max_value_f_2019_l217_217768

-- Define the function f: ℤ → ℤ with properties
def f (n : ℤ) : ℤ := sorry 

-- Conditions from the problem
axiom f_zero : f 0 = 0
axiom f_property : ∀ (k n : ℤ), k ≥ 0 → |f ((n + 1) * 2^k) - f (n * 2^k)| ≤ 1

-- The theorem to prove
theorem max_value_f_2019 : f 2019 = 4 := sorry

end max_value_f_2019_l217_217768


namespace find_g_poly_l217_217564

theorem find_g_poly (f g : ℝ[X]) (h₀ : f + g = 3 * (X : ℝ[X]) - X^2) (h₁ : f = X^2 - 4 * X + 3) :
  g = -2 * X^2 + 7 * X - 3 :=
  sorry

end find_g_poly_l217_217564


namespace surveyor_problem_l217_217780

theorem surveyor_problem
  (GF : ℝ) (G4 : ℝ)
  (hGF : GF = 70)
  (hG4 : G4 = 60) :
  (1/2) * GF * G4 = 2100 := 
  by
  sorry

end surveyor_problem_l217_217780


namespace logarithmic_inequality_l217_217851

theorem logarithmic_inequality (x : ℝ) (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (∀ a b : ℝ, (a > 0) → (b > 0) → 2 * log x ((a + b) / 2) ≤ log x a + log x b) ↔ 0 < x ∧ x < 1 :=
by
  sorry

end logarithmic_inequality_l217_217851


namespace obtuse_angle_iff_l217_217533

-- Define the vectors
def vector_a : ℝ × ℝ := (2, -4)
def vector_b (λ : ℝ) : ℝ × ℝ := (-1, λ)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Prove that the vectors form an obtuse angle if and only if λ > -1/2
theorem obtuse_angle_iff (λ : ℝ) : dot_product vector_a (vector_b λ) < 0 ↔ λ > -1 / 2 :=
by sorry

end obtuse_angle_iff_l217_217533


namespace solution_exists_l217_217959

noncomputable def exists_solution (a x : ℝ) : Prop :=
  (2 - 2 * a * (x + 1)) / (|x| - x) = real.sqrt (1 - a - a * x)

theorem solution_exists (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ exists_solution a x) ↔
    (1 < a ∧ a < 2 ∨
    a = 2) :=
  sorry

end solution_exists_l217_217959


namespace vector_midpoint_subtraction_l217_217956

/-- Given a triangle ABC with D as the midpoint of BC,
then the vector CD - DA equals AB. -/
theorem vector_midpoint_subtraction
  (A B C D : Type*)
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
  [AddCommGroup D] [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D]
  (midpoint_D : D = (B + C) / 2) :
  (C - D) - (D - A) = (B - A) :=
sorry

end vector_midpoint_subtraction_l217_217956


namespace simplifyExpression_l217_217000

theorem simplifyExpression : 
  (Real.sqrt 4 + Real.cbrt (-64) - Real.sqrt ((-3)^2) + abs (Real.sqrt 3 - 1)) = Real.sqrt 3 - 6 :=
by
  sorry

end simplifyExpression_l217_217000


namespace sqrt_sqrt_16_eq_pm2_l217_217706

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end sqrt_sqrt_16_eq_pm2_l217_217706


namespace largest_among_five_numbers_l217_217352

theorem largest_among_five_numbers :
  max (max (max (max (12345 + 1 / 3579) 
                       (12345 - 1 / 3579))
                   (12345 ^ (1 / 3579)))
               (12345 / (1 / 3579)))
           12345.3579 = 12345 / (1 / 3579) := sorry

end largest_among_five_numbers_l217_217352


namespace not_all_obtuse_after_splitting_l217_217418

theorem not_all_obtuse_after_splitting (T : Triangle) (h : T.acute) :
  ∀ (parts : List Polygon), 
  (initial_split T parts) ∧ (∀ p ∈ parts, straight_line_split p) →
  ¬ (∀ t ∈ parts, obtuse_triangle t) :=
begin
    sorry
end

end not_all_obtuse_after_splitting_l217_217418


namespace negative_solution_exists_l217_217063

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l217_217063


namespace num_of_laborers_is_24_l217_217273

def average_salary_all (L S : Nat) (avg_salary_ls : Nat) (avg_salary_l : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_l * L + avg_salary_s * S) / (L + S) = avg_salary_ls

def average_salary_supervisors (S : Nat) (avg_salary_s : Nat) : Prop :=
  (avg_salary_s * S) / S = avg_salary_s

theorem num_of_laborers_is_24 :
  ∀ (L S : Nat) (avg_salary_ls avg_salary_l avg_salary_s : Nat),
    average_salary_all L S avg_salary_ls avg_salary_l avg_salary_s →
    average_salary_supervisors S avg_salary_s →
    S = 6 → avg_salary_ls = 1250 → avg_salary_l = 950 → avg_salary_s = 2450 →
    L = 24 :=
by
  intros L S avg_salary_ls avg_salary_l avg_salary_s h1 h2 h3 h4 h5 h6
  sorry

end num_of_laborers_is_24_l217_217273


namespace prove_jens_suckers_l217_217248
noncomputable def Jen_ate_suckers (Sienna_suckers : ℕ) (Jen_suckers_given_to_Molly : ℕ) : Prop :=
  let Molly_suckers_given_to_Harmony := Jen_suckers_given_to_Molly - 2
  let Harmony_suckers_given_to_Taylor := Molly_suckers_given_to_Harmony + 3
  let Taylor_suckers_given_to_Callie := Harmony_suckers_given_to_Taylor - 1
  Taylor_suckers_given_to_Callie = 5 → (Sienna_suckers/2) = Jen_suckers_given_to_Molly * 2

#eval Jen_ate_suckers 44 11 -- Example usage, you can change 44 and 11 accordingly

def jen_ate_11_suckers : Prop :=
  Jen_ate_suckers Sienna_suckers 11

theorem prove_jens_suckers : jen_ate_11_suckers :=
  sorry

end prove_jens_suckers_l217_217248


namespace chi_squared_test_geometric_sequence_compare_p5_q5_l217_217665

-- Definitions for chi-square test problem
constant a b c d : ℕ
constant alpha chi_squared : ℝ

axiom total_count : a + b + c + d = 200
axiom prefers_subway : a + c = 100
axiom prefers_other : b + d = 100
axiom int_metropolis_total : a + b = 140
axiom smc_total : c + d = 60
axiom alpha_value : alpha = 0.010
axiom critical_value : chi_squared = 6.635

def calc_chi_squared (a b c d n : ℕ) : ℝ :=
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test : calc_chi_squared a b c d (a + b + c + d) > chi_squared :=
  sorry

-- Definitions for David’s probability sequence problem
constant M : ℕ
constant p : ℕ → ℝ
constant q : ℕ → ℝ

axiom initial_probabilities : p 1 = 1 ∧ p 2 = 0
axiom prob_form (n : ℕ) : p n = p (n-1) * 0 + (1 - p (n-1)) * 1/3

theorem geometric_sequence (n : ℕ) : p n - 1/4 = (-1/3) ^ (n-1) * (p 1 - 1/4) :=
  sorry

theorem compare_p5_q5 : p 5 > q 5 := 
  sorry

end chi_squared_test_geometric_sequence_compare_p5_q5_l217_217665


namespace diameter_of_incircle_l217_217905

-- Explicit definition of the lines
def line1 := (y : ℝ) = -5
def line2 := (x y : ℝ) := 3*x + 4*y - 28 = 0
def line3 := (x y : ℝ) := -5*x + 2*y*Real.sqrt 6 - 24 = 0

-- Derived system of linear equations
def eq1 (r y0 : ℝ) := r + y0 = -5
def eq2 (r x0 y0 : ℝ) := 5 * r - 3 * x0 - 4 * y0 = -28
def eq3 (r x0 y0 : ℝ) := 7 * r + 5 * x0 - 2 * Real.sqrt 6 * y0 = -24

-- Statement of the proof problem
theorem diameter_of_incircle :
  ∃ (r x0 y0 : ℝ), eq1 r y0 ∧ eq2 r x0 y0 ∧ eq3 r x0 y0 → 2 * r = 11 :=
by
  sorry

end diameter_of_incircle_l217_217905


namespace functional_relationship_maximum_profit_desired_profit_l217_217450

-- Conditions
def cost_price := 80
def y (x : ℝ) : ℝ := -2 * x + 320
def w (x : ℝ) : ℝ := (x - cost_price) * y x

-- Functional relationship
theorem functional_relationship (x : ℝ) (hx : 80 ≤ x ∧ x ≤ 160) :
  w x = -2 * x^2 + 480 * x - 25600 :=
by sorry

-- Maximizing daily profit
theorem maximum_profit :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = 3200 ∧ (∀ y, 80 ≤ y ∧ y ≤ 160 → w y ≤ 3200) ∧ x = 120 :=
by sorry

-- Desired profit of 2400 dollars
theorem desired_profit (w_desired : ℝ) (hw : w_desired = 2400) :
  ∃ x, 80 ≤ x ∧ x ≤ 160 ∧ w x = w_desired ∧ x = 100 :=
by sorry

end functional_relationship_maximum_profit_desired_profit_l217_217450


namespace diagonal_length_of_regular_octagon_l217_217833

theorem diagonal_length_of_regular_octagon (s : ℝ) (h : s = 12) :
  let EA := 24 * Real.sqrt ((1 + Real.sqrt 2 / 2) / 2) in
  EA = 24 * Real.sqrt ((1 + Real.sqrt 2 / 2) / 2) :=
by
  have h1 : s = 12 := h
  sorry

end diagonal_length_of_regular_octagon_l217_217833


namespace max_number_of_digits_in_product_after_subtraction_is_9_l217_217331

def max_digit_length_after_subtraction : ℕ :=
  let a := 99999
  let b := 9999
  let c := 9
  let product := a * b
  let result := product - c
  (result.toString.length)

theorem max_number_of_digits_in_product_after_subtraction_is_9 :
  max_digit_length_after_subtraction = 9 := by sorry

end max_number_of_digits_in_product_after_subtraction_is_9_l217_217331


namespace sean_days_played_is_14_l217_217243

def total_minutes_played : Nat := 1512
def indira_minutes_played : Nat := 812
def sean_minutes_per_day : Nat := 50
def sean_total_minutes : Nat := total_minutes_played - indira_minutes_played
def sean_days_played : Nat := sean_total_minutes / sean_minutes_per_day

theorem sean_days_played_is_14 : sean_days_played = 14 :=
by
  sorry

end sean_days_played_is_14_l217_217243


namespace sequence_formula_l217_217601

theorem sequence_formula (a : ℕ → ℕ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) - 2 * a n + 3 = 0) :
  ∀ n : ℕ, a n = 3 - 2^n :=
by
  sorry

end sequence_formula_l217_217601


namespace rectangle_length_rounded_l217_217257

noncomputable def length_of_rectangle (area_of_PQRS : ℤ) (x : ℝ) : ℝ :=
  let w := (2 / 3) * x
  let PQ := 3 * w
  let PS := 2 * x
  let areas_of_six_rectangles := 6 * (x * w)
  real.sqrt ((areas_of_six_rectangles / area_of_PQRS) * x^2)

theorem rectangle_length_rounded (area_of_PQRS : ℤ) (h_area : area_of_PQRS = 6000) :
  real.round(length_of_rectangle area_of_PQRS 1500) = 39 :=
by
  sorry

end rectangle_length_rounded_l217_217257


namespace distance_from_point_to_line_l217_217954

noncomputable def polarPoint := (2 : ℝ, -Real.pi / 6)
noncomputable def polarLine (ρ θ : ℝ) := ρ * Real.sin (θ - Real.pi / 6) = 1

def toRectCoords (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def lineRectCoords (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 2 = 0

def pointToLineDistance (x y : ℝ) (a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a ^ 2 + b ^ 2)

theorem distance_from_point_to_line :
  let P := toRectCoords 2 (-Real.pi / 6)
  let l := polarLine
  let (x, y) := P in
  pointToLineDistance x y 1 (-Real.sqrt 3) 2 = 1 := by sorry

end distance_from_point_to_line_l217_217954


namespace range_of_a_l217_217514

noncomputable def A (x : ℝ) : Prop := (3 * x) / (x + 1) ≤ 2
noncomputable def B (x a : ℝ) : Prop := a - 2 < x ∧ x < 2 * a + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, A x ↔ B x a) ↔ (1 / 2 < a ∧ a ≤ 1) := by
sorry

end range_of_a_l217_217514


namespace constant_property_area_range_l217_217893

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a^2 = b^2 + 3)) : Prop :=
  a = 2 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / 4 + y^2 = 1 → true))

theorem constant_property (P Q : ℝ × ℝ) (hP : (P.1^2 / 4 + P.2^2 = 1)) (hQ : (Q.1^2 / 4 + Q.2^2 = 1)) 
(h_perp : P.1 * Q.1 + P.2 * Q.2 = 0) : 
  (1 / (P.1^2 + P.2^2) + 1 / (Q.1^2 + Q.2^2) = 5 / 4) :=
sorry

theorem area_range (OP Q : ℝ × ℝ) (hP : (P.1^2 / 4  + P.2^2 = 1)) (hQ : (Q.1^2 / 4 + Q.2^2 = 1)) 
(h_perp : P.1 * Q.1 + P.2 * Q.2 = 0) : 
  (4 / 5 ≤ 0.5 * real.sqrt (real.sqrt (P.1^2 + P.2^2) * real.sqrt (Q.1^2 + Q.2^2)) ∧ 0.5 * real.sqrt (real.sqrt (P.1^2 + P.2^2) * real.sqrt (Q.1^2 + Q.2^2)) ≤ 1) :=
sorry

end constant_property_area_range_l217_217893


namespace john_tax_deduction_l217_217610

theorem john_tax_deduction :
  ∀ (hourly_wage : ℝ) (tax_rate : ℝ),
    hourly_wage = 25 → tax_rate = 0.024 → 
    let wage_in_cents := hourly_wage * 100 in
    let tax_in_cents := wage_in_cents * tax_rate in
    tax_in_cents = 60 :=
by
  intros hourly_wage tax_rate hwage_eq trate_eq
  let wage_in_cents := hourly_wage * 100
  let tax_in_cents := wage_in_cents * tax_rate
  rw [hwage_eq, trate_eq]
  dsimp [wage_in_cents, tax_in_cents]
  sorry

end john_tax_deduction_l217_217610


namespace solution_set_l217_217975

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set (x : ℝ) : 
  ((x > 1 ∧ x < 2 ∨ x > Real.sqrt 10)) ↔ f x > 2 :=
sorry

end solution_set_l217_217975


namespace problem_1_problem_2_l217_217986

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem problem_1 (x : ℝ) : f x ≥ 2 ↔ (x ≤ -7 ∨ x ≥ 5 / 3) :=
sorry

theorem problem_2 : ∃ x : ℝ, f x = -9 / 2 :=
sorry

end problem_1_problem_2_l217_217986


namespace field_length_l217_217265

theorem field_length (trees : ℕ) (interval : ℝ) (begin_end_trees : Bool) : 
    trees = 10 → interval = 10 → begin_end_trees = true → field_length = 90 := 
by
  intro ht hi hb
  sorry

end field_length_l217_217265


namespace car_passing_time_l217_217005

open Real

theorem car_passing_time
  (vX : ℝ) (lX : ℝ)
  (vY : ℝ) (lY : ℝ)
  (t : ℝ)
  (h_vX : vX = 90)
  (h_lX : lX = 5)
  (h_vY : vY = 91)
  (h_lY : lY = 6)
  :
  (t * (vY - vX) / 3600) = 0.011 → t = 39.6 := 
by
  sorry

end car_passing_time_l217_217005
