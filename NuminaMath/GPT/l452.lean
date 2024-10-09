import Mathlib

namespace apples_used_l452_45207

def initial_apples : ℕ := 43
def apples_left : ℕ := 2

theorem apples_used : initial_apples - apples_left = 41 :=
by sorry

end apples_used_l452_45207


namespace rectangle_area_l452_45263

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 250) : l * w = 2500 :=
  sorry

end rectangle_area_l452_45263


namespace projective_transformation_is_cross_ratio_preserving_l452_45292

theorem projective_transformation_is_cross_ratio_preserving (P : ℝ → ℝ) :
  (∃ a b c d : ℝ, (ad - bc ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))) ↔
  (∀ x1 x2 x3 x4 : ℝ, (x1 - x3) * (x2 - x4) / ((x1 - x4) * (x2 - x3)) =
       (P x1 - P x3) * (P x2 - P x4) / ((P x1 - P x4) * (P x2 - P x3))) :=
sorry

end projective_transformation_is_cross_ratio_preserving_l452_45292


namespace kelvin_can_win_l452_45221

-- Defining the game conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Game Strategy
def kelvin_always_wins : Prop :=
  ∀ (n : ℕ), ∀ (d : ℕ), (d ∈ (List.range 10)) → 
    ∃ (k : ℕ), k ∈ [3, 7] ∧ ¬is_perfect_square (10 * n + k)

theorem kelvin_can_win : kelvin_always_wins :=
by {
  sorry -- Proof based on strategy of adding 3 or 7 modulo 10 and modulo 100 analysis
}

end kelvin_can_win_l452_45221


namespace evaluate_f_at_5_l452_45222

def f (x : ℕ) : ℕ := x^5 + 2 * x^4 + 3 * x^3 + 4 * x^2 + 5 * x + 6

theorem evaluate_f_at_5 : f 5 = 4881 :=
by
-- proof
sorry

end evaluate_f_at_5_l452_45222


namespace determine_k_completed_square_l452_45246

theorem determine_k_completed_square (x : ℝ) :
  ∃ (a h k : ℝ), a * (x - h)^2 + k = x^2 - 7 * x ∧ k = -49/4 := sorry

end determine_k_completed_square_l452_45246


namespace evaluate_expression_l452_45281

theorem evaluate_expression :
  let a := 17
  let b := 19
  let c := 23
  let numerator1 := 136 * (1 / b - 1 / c) + 361 * (1 / c - 1 / a) + 529 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  let numerator2 := 144 * (1 / b - 1 / c) + 400 * (1 / c - 1 / a) + 576 * (1 / a - 1 / b)
  (numerator1 / denominator) * (numerator2 / denominator) = 3481 := by
  sorry

end evaluate_expression_l452_45281


namespace solve_for_D_d_Q_R_l452_45271

theorem solve_for_D_d_Q_R (D d Q R : ℕ) 
    (h1 : D = d * Q + R) 
    (h2 : d * Q = 135) 
    (h3 : R = 2 * d) : 
    D = 165 ∧ d = 15 ∧ Q = 9 ∧ R = 30 :=
by
  sorry

end solve_for_D_d_Q_R_l452_45271


namespace mass_percentage_Ba_in_BaI2_l452_45210

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 :
  (molar_mass_Ba / molar_mass_BaI2 * 100) = 35.11 :=
by
  sorry

end mass_percentage_Ba_in_BaI2_l452_45210


namespace sunil_interest_l452_45204

-- Condition definitions
def A : ℝ := 3370.80
def r : ℝ := 0.06
def n : ℕ := 1
def t : ℕ := 2

-- Derived definition for principal P
noncomputable def P : ℝ := A / (1 + r/n)^(n * t)

-- Interest I calculation
noncomputable def I : ℝ := A - P

-- Proof statement
theorem sunil_interest : I = 370.80 :=
by
  -- Insert the mathematical proof steps here.
  sorry

end sunil_interest_l452_45204


namespace money_inequality_l452_45217

-- Definitions and conditions
variables (a b : ℝ)
axiom cond1 : 6 * a + b > 78
axiom cond2 : 4 * a - b = 42

-- Theorem that encapsulates the problem and required proof
theorem money_inequality (a b : ℝ) (h1: 6 * a + b > 78) (h2: 4 * a - b = 42) : a > 12 ∧ b > 6 :=
  sorry

end money_inequality_l452_45217


namespace exists_real_solution_real_solution_specific_values_l452_45290

theorem exists_real_solution (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

theorem real_solution_specific_values  (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) : 
  ∃ x : ℝ, (a * b^x)^(x + 1) = c :=
sorry

end exists_real_solution_real_solution_specific_values_l452_45290


namespace count_with_consecutive_ones_l452_45260

noncomputable def countValidIntegers : ℕ := 512
noncomputable def invalidCount : ℕ := 89

theorem count_with_consecutive_ones :
  countValidIntegers - invalidCount = 423 :=
by
  sorry

end count_with_consecutive_ones_l452_45260


namespace overall_average_score_l452_45298

theorem overall_average_score (first_6_avg last_4_avg : ℝ) (n_first n_last n_total : ℕ) 
    (h_matches : n_first + n_last = n_total)
    (h_first_avg : first_6_avg = 41)
    (h_last_avg : last_4_avg = 35.75)
    (h_n_first : n_first = 6)
    (h_n_last : n_last = 4)
    (h_n_total : n_total = 10) :
    ((first_6_avg * n_first + last_4_avg * n_last) / n_total) = 38.9 := by
  sorry

end overall_average_score_l452_45298


namespace quadratic_csq_l452_45275

theorem quadratic_csq (x q t : ℝ) (h : 9 * x^2 - 36 * x - 81 = 0) (hq : q = -2) (ht : t = 13) :
  q + t = 11 :=
by
  sorry

end quadratic_csq_l452_45275


namespace arithmetic_sequence_ninth_term_l452_45262

theorem arithmetic_sequence_ninth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 5 * d = 11) :
  a + 8 * d = 17 := by
  sorry

end arithmetic_sequence_ninth_term_l452_45262


namespace worker_total_pay_l452_45268

def regular_rate : ℕ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def non_cellphone_surveys := total_surveys - cellphone_surveys
def higher_rate := regular_rate + (30 * regular_rate / 100)

def pay_non_cellphone_surveys := non_cellphone_surveys * regular_rate
def pay_cellphone_surveys := cellphone_surveys * higher_rate

def total_pay := pay_non_cellphone_surveys + pay_cellphone_surveys

theorem worker_total_pay : total_pay = 605 := by
  sorry

end worker_total_pay_l452_45268


namespace solve_complex_addition_l452_45241

noncomputable def complex_addition : Prop :=
  let i := Complex.I
  let z1 := 3 - 5 * i
  let z2 := -1 + 12 * i
  let result := 2 + 7 * i
  z1 + z2 = result

theorem solve_complex_addition :
  complex_addition :=
by
  sorry

end solve_complex_addition_l452_45241


namespace volume_original_cone_l452_45276

-- Given conditions
def V_cylinder : ℝ := 21
def V_truncated_cone : ℝ := 91

-- To prove: The volume of the original cone is 94.5
theorem volume_original_cone : 
    (∃ (H R h r : ℝ), (π * r^2 * h = V_cylinder) ∧ (1 / 3 * π * (R^2 + R * r + r^2) * (H - h) = V_truncated_cone)) →
    (1 / 3 * π * R^2 * H = 94.5) :=
by
  sorry

end volume_original_cone_l452_45276


namespace negation_if_proposition_l452_45239

variable (a b : Prop)

theorem negation_if_proposition (a b : Prop) : ¬ (a → b) = a ∧ ¬b := 
sorry

end negation_if_proposition_l452_45239


namespace expected_value_is_minus_one_fifth_l452_45252

-- Define the parameters given in the problem
def p_heads := 2 / 5
def p_tails := 3 / 5
def win_heads := 4
def loss_tails := -3

-- Calculate the expected value for heads and tails
def expected_heads := p_heads * win_heads
def expected_tails := p_tails * loss_tails

-- The theorem stating that the expected value is -1/5
theorem expected_value_is_minus_one_fifth :
  expected_heads + expected_tails = -1 / 5 :=
by
  -- The proof can be filled in here
  sorry

end expected_value_is_minus_one_fifth_l452_45252


namespace propositions_correct_l452_45212

def f (x : Real) (b c : Real) : Real := x * abs x + b * x + c

-- Define proposition P1: When c = 0, y = f(x) is an odd function.
def P1 (b : Real) : Prop :=
  ∀ x : Real, f x b 0 = - f (-x) b 0

-- Define proposition P2: When b = 0 and c > 0, the equation f(x) = 0 has only one real root.
def P2 (c : Real) : Prop :=
  c > 0 → ∃! x : Real, f x 0 c = 0

-- Define proposition P3: The graph of y = f(x) is symmetric about the point (0, c).
def P3 (b c : Real) : Prop :=
  ∀ x : Real, f x b c = 2 * c - f x b c

-- Define the final theorem statement
theorem propositions_correct (b c : Real) : P1 b ∧ P2 c ∧ P3 b c := sorry

end propositions_correct_l452_45212


namespace find_x_value_l452_45272

theorem find_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y^3) (h2 : x / 6 = 3 * y) : x = 18 * Real.sqrt 6 := 
by 
  sorry

end find_x_value_l452_45272


namespace PropA_impl_PropB_not_PropB_impl_PropA_l452_45284

variable {x : ℝ}

def PropA (x : ℝ) : Prop := abs (x - 1) < 5
def PropB (x : ℝ) : Prop := abs (abs x - 1) < 5

theorem PropA_impl_PropB : PropA x → PropB x :=
by sorry

theorem not_PropB_impl_PropA : ¬(PropB x → PropA x) :=
by sorry

end PropA_impl_PropB_not_PropB_impl_PropA_l452_45284


namespace rectangle_area_l452_45237

theorem rectangle_area (l : ℝ) (w : ℝ) (h_l : l = 15) (h_ratio : (2 * l + 2 * w) / w = 5) : (l * w) = 150 :=
by
  sorry

end rectangle_area_l452_45237


namespace work_problem_l452_45229

theorem work_problem (B_rate : ℝ) (C_rate : ℝ) (A_rate : ℝ) :
  (B_rate = 1/12) →
  (B_rate + C_rate = 1/3) →
  (A_rate + C_rate = 1/2) →
  (A_rate = 1/4) :=
by
  intros h1 h2 h3
  sorry

end work_problem_l452_45229


namespace weight_of_banana_l452_45234

theorem weight_of_banana (A B G : ℝ) (h1 : 3 * A = G) (h2 : 4 * B = 2 * A) (h3 : G = 576) : B = 96 :=
by
  sorry

end weight_of_banana_l452_45234


namespace find_f_three_l452_45235

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f (a + b) = f a + f b
axiom f_two : f 2 = 3

theorem find_f_three : f 3 = 9 / 2 :=
by
  sorry

end find_f_three_l452_45235


namespace practice_time_for_Friday_l452_45219

variables (M T W Th F : ℕ)

def conditions : Prop :=
  (M = 2 * T) ∧
  (T = W - 10) ∧
  (W = Th + 5) ∧
  (Th = 50) ∧
  (M + T + W + Th + F = 300)

theorem practice_time_for_Friday (h : conditions M T W Th F) : F = 60 :=
sorry

end practice_time_for_Friday_l452_45219


namespace allen_blocks_l452_45280

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end allen_blocks_l452_45280


namespace circle_area_sum_l452_45236

theorem circle_area_sum (x y z : ℕ) (A₁ A₂ A₃ total_area : ℕ) (h₁ : A₁ = 6) (h₂ : A₂ = 15) 
  (h₃ : A₃ = 83) (h₄ : total_area = 220) (hx : x = 4) (hy : y = 2) (hz : z = 2) :
  A₁ * x + A₂ * y + A₃ * z = total_area := by
  sorry

end circle_area_sum_l452_45236


namespace sum_of_money_proof_l452_45249

noncomputable def total_sum (A B C : ℝ) : ℝ := A + B + C

theorem sum_of_money_proof (A B C : ℝ) (h1 : B = 0.65 * A) (h2 : C = 0.40 * A) (h3 : C = 64) : total_sum A B C = 328 :=
by 
  sorry

end sum_of_money_proof_l452_45249


namespace solve_tetrahedron_side_length_l452_45283

noncomputable def side_length_of_circumscribing_tetrahedron (r : ℝ) (tangent_spheres : ℕ) (radius_spheres_equal : ℝ) : ℝ := 
  if h : r = 1 ∧ tangent_spheres = 4 then
    2 + 2 * Real.sqrt 6
  else
    0

theorem solve_tetrahedron_side_length :
  side_length_of_circumscribing_tetrahedron 1 4 1 = 2 + 2 * Real.sqrt 6 :=
by
  sorry

end solve_tetrahedron_side_length_l452_45283


namespace ball_highest_point_at_l452_45273

noncomputable def h (a b t : ℝ) : ℝ := a * t^2 + b * t

theorem ball_highest_point_at (a b : ℝ) :
  (h a b 3 = h a b 7) →
  t = 4.9 :=
by
  sorry

end ball_highest_point_at_l452_45273


namespace ratio_of_areas_l452_45278

-- Definitions based on the conditions given
def square_side_length : ℕ := 48
def rectangle_width : ℕ := 56
def rectangle_height : ℕ := 63

-- Areas derived from the definitions
def square_area := square_side_length * square_side_length
def rectangle_area := rectangle_width * rectangle_height

-- Lean statement to prove the ratio of areas
theorem ratio_of_areas :
  (square_area : ℚ) / rectangle_area = 2 / 3 := 
sorry

end ratio_of_areas_l452_45278


namespace roots_of_polynomial_l452_45206

theorem roots_of_polynomial (c d : ℝ) (h1 : Polynomial.eval c (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0)
    (h2 : Polynomial.eval d (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0) :
    c * d + c + d = Real.sqrt 3 := 
sorry

end roots_of_polynomial_l452_45206


namespace digit_start_l452_45299

theorem digit_start (a n p q : ℕ) (hp : a * 10^p < 2^n) (hq : 2^n < (a + 1) * 10^p)
  (hr : a * 10^q < 5^n) (hs : 5^n < (a + 1) * 10^q) :
  a = 3 :=
by
  -- The proof goes here.
  sorry

end digit_start_l452_45299


namespace distribute_balls_into_boxes_l452_45208

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l452_45208


namespace general_equation_l452_45261

theorem general_equation (n : ℤ) : 
    ∀ (a b : ℤ), 
    (a = 2 ∧ b = 6) ∨ (a = 5 ∧ b = 3) ∨ (a = 7 ∧ b = 1) ∨ (a = 10 ∧ b = -2) → 
    (a / (a - 4) + b / (b - 4) = 2) →
    (n / (n - 4) + (8 - n) / ((8 - n) - 4) = 2) :=
by
  intros a b h_cond h_eq
  sorry

end general_equation_l452_45261


namespace pairs_nat_eq_l452_45285

theorem pairs_nat_eq (n k : ℕ) :
  (n + 1) ^ k = n! + 1 ↔ (n, k) = (1, 1) ∨ (n, k) = (2, 1) ∨ (n, k) = (4, 2) :=
by
  sorry

end pairs_nat_eq_l452_45285


namespace britney_has_more_chickens_l452_45277

theorem britney_has_more_chickens :
  let susie_rhode_island_reds := 11
  let susie_golden_comets := 6
  let britney_rhode_island_reds := 2 * susie_rhode_island_reds
  let britney_golden_comets := susie_golden_comets / 2
  let susie_total := susie_rhode_island_reds + susie_golden_comets
  let britney_total := britney_rhode_island_reds + britney_golden_comets
  britney_total - susie_total = 8 := by
    sorry

end britney_has_more_chickens_l452_45277


namespace smallest_integer_to_make_multiple_of_five_l452_45288

theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k: ℕ, 0 < k ∧ (726 + k) % 5 = 0 ∧ k = 4 := 
by
  use 4
  sorry

end smallest_integer_to_make_multiple_of_five_l452_45288


namespace max_visible_unit_cubes_from_corner_l452_45211

theorem max_visible_unit_cubes_from_corner :
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  faces_visible - edges_shared + corner_cube = 331 := by
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  have result : faces_visible - edges_shared + corner_cube = 331 := by
    sorry
  exact result

end max_visible_unit_cubes_from_corner_l452_45211


namespace cargo_transport_possible_l452_45291

theorem cargo_transport_possible 
  (total_cargo_weight : ℝ) 
  (weight_limit_per_box : ℝ) 
  (number_of_trucks : ℕ) 
  (max_load_per_truck : ℝ)
  (h1 : total_cargo_weight = 13.5)
  (h2 : weight_limit_per_box = 0.35)
  (h3 : number_of_trucks = 11)
  (h4 : max_load_per_truck = 1.5) :
  ∃ (n : ℕ), n ≤ number_of_trucks ∧ (total_cargo_weight / max_load_per_truck) ≤ n :=
by
  sorry

end cargo_transport_possible_l452_45291


namespace circle_radius_l452_45257
open Real

theorem circle_radius (d : ℝ) (h_diam : d = 24) : d / 2 = 12 :=
by
  -- The proof will be here
  sorry

end circle_radius_l452_45257


namespace sum_geometric_sequence_l452_45287

variable {α : Type*} [LinearOrderedField α]

theorem sum_geometric_sequence {S : ℕ → α} {n : ℕ} (h1 : S n = 3) (h2 : S (3 * n) = 21) :
    S (2 * n) = 9 := 
sorry

end sum_geometric_sequence_l452_45287


namespace cost_of_dinner_l452_45250

theorem cost_of_dinner (x : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (total_cost : ℝ) : 
  tax_rate = 0.09 → tip_rate = 0.18 → total_cost = 36.90 → 
  1.27 * x = 36.90 → x = 29 :=
by
  intros htr htt htc heq
  rw [←heq] at htc
  sorry

end cost_of_dinner_l452_45250


namespace balls_in_boxes_l452_45243

theorem balls_in_boxes:
  ∃ (x y z : ℕ), 
  x + y + z = 320 ∧ 
  6 * x + 11 * y + 15 * z = 1001 ∧
  x > 0 ∧ y > 0 ∧ z > 0 :=
by
  sorry

end balls_in_boxes_l452_45243


namespace truck_driver_needs_more_gallons_l452_45230

-- Define the conditions
def miles_per_gallon : ℕ := 3
def total_distance : ℕ := 90
def current_gallons : ℕ := 12
def can_cover_distance : ℕ := miles_per_gallon * current_gallons
def additional_distance_needed : ℕ := total_distance - can_cover_distance

-- Define the main theorem
theorem truck_driver_needs_more_gallons :
  additional_distance_needed / miles_per_gallon = 18 :=
by
  -- Placeholder for the proof
  sorry

end truck_driver_needs_more_gallons_l452_45230


namespace sums_of_integers_have_same_remainder_l452_45297

theorem sums_of_integers_have_same_remainder (n : ℕ) (n_pos : 0 < n) : 
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ 2 * n) ∧ (1 ≤ j ∧ j ≤ 2 * n) ∧ i ≠ j ∧ ((i + i) % (2 * n) = (j + j) % (2 * n)) :=
by
  sorry

end sums_of_integers_have_same_remainder_l452_45297


namespace constant_term_expansion_l452_45225

-- Defining the binomial theorem term
noncomputable def binomial_coeff (n k : ℕ) : ℕ := 
  Nat.choose n k

-- The general term of the binomial expansion (2sqrt(x) - 1/x)^6
noncomputable def general_term (r : ℕ) (x : ℝ) : ℝ :=
  binomial_coeff 6 r * (-1)^r * (2^(6-r)) * x^((6 - 3 * r) / 2)

-- Problem statement: Show that the constant term in the expansion is 240
theorem constant_term_expansion :
  (∃ r : ℕ, (6 - 3 * r) / 2 = 0 ∧ 
            general_term r arbitrary = 240) :=
sorry

end constant_term_expansion_l452_45225


namespace measure_of_angle_x_l452_45264

-- Given conditions
def angle_ABC : ℝ := 120
def angle_BAD : ℝ := 31
def angle_BDA (x : ℝ) : Prop := x + 60 + 31 = 180 

-- Statement to prove
theorem measure_of_angle_x : 
  ∃ x : ℝ, angle_BDA x → x = 89 :=
by
  sorry

end measure_of_angle_x_l452_45264


namespace fraction_evaluation_l452_45293

theorem fraction_evaluation :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 :=
by
  sorry

end fraction_evaluation_l452_45293


namespace units_digit_2_pow_10_l452_45200

theorem units_digit_2_pow_10 : (2 ^ 10) % 10 = 4 := 
sorry

end units_digit_2_pow_10_l452_45200


namespace determine_n_l452_45227

theorem determine_n 
    (n : ℕ) (h2 : n ≥ 2) 
    (a : ℕ) (ha_div_n : a ∣ n) 
    (ha_min : ∀ d : ℕ, d ∣ n → d > 1 → d ≥ a) 
    (b : ℕ) (hb_div_n : b ∣ n)
    (h_eq : n = a^2 + b^2) : 
    n = 8 ∨ n = 20 :=
sorry

end determine_n_l452_45227


namespace sum_of_products_of_two_at_a_time_l452_45254

-- Given conditions
variables (a b c : ℝ)
axiom sum_of_squares : a^2 + b^2 + c^2 = 252
axiom sum_of_numbers : a + b + c = 22

-- The goal
theorem sum_of_products_of_two_at_a_time : a * b + b * c + c * a = 116 :=
sorry

end sum_of_products_of_two_at_a_time_l452_45254


namespace sequences_recurrence_relation_l452_45238

theorem sequences_recurrence_relation 
    (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ)
    (h1 : a 1 = 1) (h2 : b 1 = 3) (h3 : c 1 = 2)
    (ha : ∀ i : ℕ, a (i + 1) = a i + c i - b i + 2)
    (hb : ∀ i : ℕ, b (i + 1) = (3 * c i - a i + 5) / 2)
    (hc : ∀ i : ℕ, c (i + 1) = 2 * a i + 2 * b i - 3) : 
    (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2^n + 1) ∧ (∀ n, c n = 3 * 2^(n-1) - 1) := 
sorry

end sequences_recurrence_relation_l452_45238


namespace problem_a2_minus_b2_problem_a3_minus_b3_l452_45232

variable (a b : ℝ)
variable (h1 : a + b = 8)
variable (h2 : a - b = 4)

theorem problem_a2_minus_b2 :
  a^2 - b^2 = 32 := 
by
sorry

theorem problem_a3_minus_b3 :
  a^3 - b^3 = 208 := 
by
sorry

end problem_a2_minus_b2_problem_a3_minus_b3_l452_45232


namespace find_point_of_intersection_l452_45267
noncomputable def point_of_intersection_curve_line : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^y = y^x ∧ y = x ∧ x = Real.exp 1 ∧ y = Real.exp 1

theorem find_point_of_intersection : point_of_intersection_curve_line :=
sorry

end find_point_of_intersection_l452_45267


namespace ones_digit_of_8_pow_50_l452_45224

theorem ones_digit_of_8_pow_50 : (8 ^ 50) % 10 = 4 := by
  sorry

end ones_digit_of_8_pow_50_l452_45224


namespace zack_initial_marbles_l452_45218

theorem zack_initial_marbles :
  ∃ M : ℕ, (∃ k : ℕ, M = 3 * k + 5) ∧ (M - 5 - 60 = 5) ∧ M = 70 := by
sorry

end zack_initial_marbles_l452_45218


namespace problem1_correct_problem2_correct_l452_45247

noncomputable def problem1 : Real :=
  2 * Real.sqrt (2 / 3) - 3 * Real.sqrt (3 / 2) + Real.sqrt 24

theorem problem1_correct : problem1 = (7 * Real.sqrt 6) / 6 := by
  sorry

noncomputable def problem2 : Real :=
  Real.sqrt (25 / 2) + Real.sqrt 32 - Real.sqrt 18 - (Real.sqrt 2 - 1)^2

theorem problem2_correct : problem2 = (11 * Real.sqrt 2) / 2 - 3 := by
  sorry

end problem1_correct_problem2_correct_l452_45247


namespace convex_polygon_longest_sides_convex_polygon_shortest_sides_l452_45214

noncomputable def convex_polygon : Type := sorry

-- Definitions for the properties and functions used in conditions
def is_convex (P : convex_polygon) : Prop := sorry
def equal_perimeters (A B : convex_polygon) : Prop := sorry
def longest_side (P : convex_polygon) : ℝ := sorry
def shortest_side (P : convex_polygon) : ℝ := sorry

-- Problem part a
theorem convex_polygon_longest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ∃ (A B : convex_polygon), equal_perimeters A B ∧ longest_side A = longest_side B :=
sorry

-- Problem part b
theorem convex_polygon_shortest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ¬(∀ (A B : convex_polygon), equal_perimeters A B → shortest_side A = shortest_side B) :=
sorry

end convex_polygon_longest_sides_convex_polygon_shortest_sides_l452_45214


namespace compare_neg_fractions_l452_45266

theorem compare_neg_fractions :
  - (10 / 11 : ℤ) > - (11 / 12 : ℤ) :=
sorry

end compare_neg_fractions_l452_45266


namespace find_b_l452_45202

noncomputable def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_b (a b c : ℝ) (h1: (-(a / 3) = -c)) (h2 : (-(a / 3) = 1 + a + b + c)) (h3: c = 2) : b = -11 :=
by
  sorry

end find_b_l452_45202


namespace tan_80_l452_45270

theorem tan_80 (m : ℝ) (h : Real.cos (100 * Real.pi / 180) = m) :
    Real.tan (80 * Real.pi / 180) = Real.sqrt (1 - m^2) / -m :=
by
  sorry

end tan_80_l452_45270


namespace longer_diagonal_of_rhombus_l452_45223

theorem longer_diagonal_of_rhombus
  (A : ℝ) (r1 r2 : ℝ) (x : ℝ)
  (hA : A = 135)
  (h_ratio : r1 = 5) (h_ratio2 : r2 = 3)
  (h_area : (1/2) * (r1 * x) * (r2 * x) = A) :
  r1 * x = 15 :=
by
  sorry

end longer_diagonal_of_rhombus_l452_45223


namespace problem_solution_l452_45274

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
if x < 0 then 3^x else m - x^2

def p (m : ℝ) : Prop :=
∃ x, f m x = 0

def q (m : ℝ) : Prop :=
m = 1 / 9 → f m (f m (-1)) = 0

theorem problem_solution :
  ¬ (∃ m, m < 0 ∧ p m) ∧ q (1 / 9) :=
by 
  sorry

end problem_solution_l452_45274


namespace exists_positive_M_l452_45269

open Set

noncomputable def f (x : ℝ) : ℝ := sorry

theorem exists_positive_M 
  (h₁ : ∀ x ∈ Ioo (0 : ℝ) 1, f x > 0)
  (h₂ : ∀ x ∈ Ioo (0 : ℝ) 1, f (2 * x / (1 + x^2)) = 2 * f x) :
  ∃ M > 0, ∀ x ∈ Ioo (0 : ℝ) 1, f x ≤ M :=
sorry

end exists_positive_M_l452_45269


namespace investment_period_more_than_tripling_l452_45295

theorem investment_period_more_than_tripling (r : ℝ) (multiple : ℝ) (n : ℕ) 
  (h_r: r = 0.341) (h_multiple: multiple > 3) :
  (1 + r)^n ≥ multiple → n = 4 :=
by
  sorry

end investment_period_more_than_tripling_l452_45295


namespace compute_xy_l452_45201

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56 / 9 :=
by
  sorry

end compute_xy_l452_45201


namespace max_value_of_m_l452_45251

theorem max_value_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 8 > 0 → x < m) → m = -2 :=
by
  sorry

end max_value_of_m_l452_45251


namespace factorization_example_l452_45215

theorem factorization_example (x : ℝ) : (x^2 - 4 * x + 4) = (x - 2)^2 :=
by sorry

end factorization_example_l452_45215


namespace composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l452_45213

theorem composite_10201_base_gt_2 (x : ℕ) (hx : x > 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + 2*x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base (x : ℕ) (hx : x ≥ 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base_any_x (x : ℕ) (hx : x ≥ 1) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

end composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l452_45213


namespace real_solutions_of_polynomial_l452_45240

theorem real_solutions_of_polynomial (b : ℝ) :
  b < -4 → ∃! x : ℝ, x^3 - b * x^2 - 4 * b * x + b^2 - 4 = 0 :=
by
  sorry

end real_solutions_of_polynomial_l452_45240


namespace smallest_triangle_perimeter_l452_45265

theorem smallest_triangle_perimeter :
  ∃ (y : ℕ), (y % 2 = 0) ∧ (y < 17) ∧ (y > 3) ∧ (7 + 10 + y = 21) :=
by
  sorry

end smallest_triangle_perimeter_l452_45265


namespace breakfast_calories_l452_45203

theorem breakfast_calories : ∀ (planned_calories : ℕ) (B : ℕ),
  planned_calories < 1800 →
  B + 900 + 1100 = planned_calories + 600 →
  B = 400 :=
by
  intros
  sorry

end breakfast_calories_l452_45203


namespace roots_equation_l452_45296
-- We bring in the necessary Lean libraries

-- Define the conditions as Lean definitions
variable (x1 x2 : ℝ)
variable (h1 : x1^2 + x1 - 3 = 0)
variable (h2 : x2^2 + x2 - 3 = 0)

-- Lean 4 statement we need to prove
theorem roots_equation (x1 x2 : ℝ) (h1 : x1^2 + x1 - 3 = 0) (h2 : x2^2 + x2 - 3 = 0) : 
  x1^3 - 4 * x2^2 + 19 = 0 := 
sorry

end roots_equation_l452_45296


namespace susan_cars_fewer_than_carol_l452_45286

theorem susan_cars_fewer_than_carol 
  (Lindsey_cars Carol_cars Susan_cars Cathy_cars : ℕ)
  (h1 : Lindsey_cars = Cathy_cars + 4)
  (h2 : Susan_cars < Carol_cars)
  (h3 : Carol_cars = 2 * Cathy_cars)
  (h4 : Cathy_cars = 5)
  (h5 : Cathy_cars + Carol_cars + Lindsey_cars + Susan_cars = 32) :
  Carol_cars - Susan_cars = 2 :=
sorry

end susan_cars_fewer_than_carol_l452_45286


namespace product_abc_l452_45253

theorem product_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b^3 = 180) : a * b * c = 60 * c := 
sorry

end product_abc_l452_45253


namespace investment_return_formula_l452_45220

noncomputable def investment_return (x : ℕ) (x_pos : x > 0) : ℝ :=
  if x = 1 then 0.5
  else 2 ^ (x - 2)

theorem investment_return_formula (x : ℕ) (x_pos : x > 0) : investment_return x x_pos = 2 ^ (x - 2) := 
by
  sorry

end investment_return_formula_l452_45220


namespace edric_hourly_rate_l452_45242

-- Define conditions
def edric_monthly_salary : ℝ := 576
def edric_weekly_hours : ℝ := 8 * 6 -- 48 hours
def average_weeks_per_month : ℝ := 4.33
def edric_monthly_hours : ℝ := edric_weekly_hours * average_weeks_per_month -- Approx 207.84 hours

-- Define the expected result
def edric_expected_hourly_rate : ℝ := 2.77

-- Proof statement
theorem edric_hourly_rate :
  edric_monthly_salary / edric_monthly_hours = edric_expected_hourly_rate :=
by
  sorry

end edric_hourly_rate_l452_45242


namespace bridge_supports_88_ounces_l452_45216

-- Define the conditions
def weight_of_soda_per_can : ℕ := 12
def number_of_soda_cans : ℕ := 6
def weight_of_empty_can : ℕ := 2
def additional_empty_cans : ℕ := 2

-- Define the total weight the bridge must hold up
def total_weight_bridge_support : ℕ :=
  (number_of_soda_cans * weight_of_soda_per_can) + ((number_of_soda_cans + additional_empty_cans) * weight_of_empty_can)

-- Prove that the total weight is 88 ounces
theorem bridge_supports_88_ounces : total_weight_bridge_support = 88 := by
  sorry

end bridge_supports_88_ounces_l452_45216


namespace find_a_l452_45256

theorem find_a (x y z a : ℝ) (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) (h2 : a > 0) (h3 : ∀ x y z : ℝ, 2 * x^2 + 3 * y^2 + 6 * z^2 = a → (x + y + z) ≤ 1) :
  a = 1 := 
sorry

end find_a_l452_45256


namespace interest_rate_determination_l452_45258

-- Problem statement
theorem interest_rate_determination (P r : ℝ) :
  (50 = P * r * 2) ∧ (51.25 = P * ((1 + r) ^ 2 - 1)) → r = 0.05 :=
by
  intros h
  sorry

end interest_rate_determination_l452_45258


namespace find_subtracted_value_l452_45289

theorem find_subtracted_value (N : ℤ) (V : ℤ) (h1 : N = 740) (h2 : N / 4 - V = 10) : V = 175 :=
by
  sorry

end find_subtracted_value_l452_45289


namespace determine_right_triangle_l452_45248

-- Definitions based on conditions
def condition_A (A B C : ℝ) : Prop := A^2 + B^2 = C^2
def condition_B (A B C : ℝ) : Prop := A^2 - B^2 = C^2
def condition_C (A B C : ℝ) : Prop := A + B = C
def condition_D (A B C : ℝ) : Prop := A / B = 3 / 4 ∧ B / C = 4 / 5

-- Problem statement: D cannot determine that triangle ABC is a right triangle
theorem determine_right_triangle (A B C : ℝ) : ¬ condition_D A B C :=
by sorry

end determine_right_triangle_l452_45248


namespace general_term_formula_sum_formula_and_max_value_l452_45233

-- Definitions for the conditions
def tenth_term : ℕ → ℤ := λ n => 24
def twenty_fifth_term : ℕ → ℤ := λ n => -21

-- Prove the general term formula
theorem general_term_formula (a : ℕ → ℤ) (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) :
  ∀ n : ℕ, a n = -3 * n + 54 := sorry

-- Prove the sum formula and its maximum value
theorem sum_formula_and_max_value (a : ℕ → ℤ) (S : ℕ → ℤ)
  (tenth_term : a 10 = 24) (twenty_fifth_term : a 25 = -21) 
  (sum_formula : ∀ n : ℕ, S n = -3 * n^2 / 2 + 51 * n) :
  ∃ max_n : ℕ, S max_n = 578 := sorry

end general_term_formula_sum_formula_and_max_value_l452_45233


namespace part1_part2_l452_45259

def f (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x + 5)

theorem part1 : ∀ x, f x < 10 ↔ (x > -19 / 3 ∧ x ≤ -5) ∨ (-5 < x ∧ x < -1) :=
  sorry

theorem part2 (a b x : ℝ) (ha : abs a < 3) (hb : abs b < 3) :
  abs (a + b) + abs (a - b) < f x :=
  sorry

end part1_part2_l452_45259


namespace adam_and_simon_distance_l452_45245

theorem adam_and_simon_distance :
  ∀ (t : ℝ), (10 * t)^2 + (12 * t)^2 = 16900 → t = 65 / Real.sqrt 61 :=
by
  sorry

end adam_and_simon_distance_l452_45245


namespace min_value_l452_45226

theorem min_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ x, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by
  sorry

end min_value_l452_45226


namespace cube_surface_area_l452_45209

-- Define the edge length of the cube
def edge_length : ℝ := 4

-- Define the formula for the surface area of a cube
def surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Prove that given the edge length is 4 cm, the surface area is 96 cm²
theorem cube_surface_area : surface_area edge_length = 96 := by
  -- Proof goes here
  sorry

end cube_surface_area_l452_45209


namespace correct_statement_D_l452_45231

theorem correct_statement_D (h : 3.14 < Real.pi) : -3.14 > -Real.pi := by
sorry

end correct_statement_D_l452_45231


namespace simplify_expression_l452_45279

theorem simplify_expression (x y : ℝ) :  3 * x + 5 * x + 7 * x + 2 * y = 15 * x + 2 * y := 
by 
  sorry

end simplify_expression_l452_45279


namespace count_false_propositions_l452_45205

def prop (a : ℝ) := a > 1 → a > 2
def converse (a : ℝ) := a > 2 → a > 1
def inverse (a : ℝ) := a ≤ 1 → a ≤ 2
def contrapositive (a : ℝ) := a ≤ 2 → a ≤ 1

theorem count_false_propositions (a : ℝ) (h : ¬(prop a)) : 
  (¬(prop a) ∧ ¬(contrapositive a)) ∧ (converse a ∧ inverse a) ↔ 2 = 2 := 
  by
    sorry

end count_false_propositions_l452_45205


namespace tan_2theta_sin_cos_fraction_l452_45282

variable {θ : ℝ} (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1)

-- Part (I)
theorem tan_2theta (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : Real.tan (2 * θ) = 4 / 3 :=
by sorry

-- Part (II)
theorem sin_cos_fraction (h : (2 - Real.tan θ) / (1 + Real.tan θ) = 1) : 
  (Real.sin θ + Real.cos θ) / (Real.cos θ - 3 * Real.sin θ) = -3 :=
by sorry

end tan_2theta_sin_cos_fraction_l452_45282


namespace dante_eggs_l452_45255

theorem dante_eggs (E F : ℝ) (h1 : F = E / 2) (h2 : F + E = 90) : E = 60 :=
by
  sorry

end dante_eggs_l452_45255


namespace sarah_monthly_payment_l452_45244

noncomputable def monthly_payment (loan_amount : ℝ) (down_payment : ℝ) (years : ℝ) : ℝ :=
  let financed_amount := loan_amount - down_payment
  let months := years * 12
  financed_amount / months

theorem sarah_monthly_payment : monthly_payment 46000 10000 5 = 600 := by
  sorry

end sarah_monthly_payment_l452_45244


namespace problem_l452_45294

variable (p q : Prop)

theorem problem (h₁ : ¬ p) (h₂ : ¬ (p ∧ q)) : ¬ (p ∨ q) := sorry

end problem_l452_45294


namespace sum_arithmetic_sequence_l452_45228

-- Define the arithmetic sequence condition and sum of given terms
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n d : ℕ, a n = a 1 + (n - 1) * d

def given_sum_condition (a : ℕ → ℕ) : Prop :=
  a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_arithmetic_sequence (a : ℕ → ℕ) (h_arith_seq : arithmetic_sequence a) 
  (h_sum_cond : given_sum_condition a) : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry  -- Proof of the theorem

end sum_arithmetic_sequence_l452_45228
