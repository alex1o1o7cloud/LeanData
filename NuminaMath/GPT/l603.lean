import Mathlib

namespace molecular_weight_of_7_moles_of_CaO_l603_60380

/-- The molecular weight of 7 moles of calcium oxide (CaO) -/
def Ca_atomic_weight : Float := 40.08
def O_atomic_weight : Float := 16.00
def CaO_molecular_weight : Float := Ca_atomic_weight + O_atomic_weight

theorem molecular_weight_of_7_moles_of_CaO : 
    7 * CaO_molecular_weight = 392.56 := by 
sorry

end molecular_weight_of_7_moles_of_CaO_l603_60380


namespace fuel_for_empty_plane_per_mile_l603_60330

theorem fuel_for_empty_plane_per_mile :
  let F := 106000 / 400 - (35 * 3 + 70 * 2)
  F = 20 := 
by
  sorry

end fuel_for_empty_plane_per_mile_l603_60330


namespace number_of_roses_sold_l603_60372

def initial_roses : ℕ := 50
def picked_roses : ℕ := 21
def final_roses : ℕ := 56

theorem number_of_roses_sold : ∃ x : ℕ, initial_roses - x + picked_roses = final_roses ∧ x = 15 :=
by {
  sorry
}

end number_of_roses_sold_l603_60372


namespace arithmetic_sequence_sum_l603_60370

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 2 + a 12 = 32) : a 3 + a 11 = 32 :=
sorry

end arithmetic_sequence_sum_l603_60370


namespace max_value_of_expression_l603_60349

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x^2 - 2 * x * y + y^2 = 6) :
  ∃ (a b c d : ℕ), (a + b * Real.sqrt c) / d = 9 + 3 * Real.sqrt 3 ∧ a + b + c + d = 16 :=
by
  sorry

end max_value_of_expression_l603_60349


namespace max_value_of_square_diff_max_value_of_square_diff_achieved_l603_60342

theorem max_value_of_square_diff (a b : ℝ) (h : a^2 + b^2 = 4) : (a - b)^2 ≤ 8 :=
sorry

theorem max_value_of_square_diff_achieved (a b : ℝ) (h : a^2 + b^2 = 4) : ∃ a b : ℝ, (a - b)^2 = 8 :=
sorry

end max_value_of_square_diff_max_value_of_square_diff_achieved_l603_60342


namespace hexagon_transformation_l603_60307

-- Define a shape composed of 36 identical small equilateral triangles
def Shape := { s : ℕ // s = 36 }

-- Define the number of triangles needed to form a hexagon
def TrianglesNeededForHexagon : ℕ := 18

-- Proof statement: Given a shape of 36 small triangles, we need 18 more triangles to form a hexagon
theorem hexagon_transformation (shape : Shape) : TrianglesNeededForHexagon = 18 :=
by
  -- This is our formalization of the problem statement which asserts
  -- that the transformation to a hexagon needs exactly 18 additional triangles.
  sorry

end hexagon_transformation_l603_60307


namespace ellipse_min_area_contains_circles_l603_60300

-- Define the ellipse and circles
def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
def circle1 (x y : ℝ) := ((x - 2)^2 + y^2 = 4)
def circle2 (x y : ℝ) := ((x + 2)^2 + y^2 = 4)

-- Proof statement: The smallest possible area of the ellipse containing the circles
theorem ellipse_min_area_contains_circles : 
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), 
    (circle1 x y → ellipse x y) ∧ 
    (circle2 x y → ellipse x y)) ∧
  (k = 12) := 
sorry

end ellipse_min_area_contains_circles_l603_60300


namespace tuples_satisfy_equation_l603_60332

theorem tuples_satisfy_equation (a b c : ℤ) :
  (a - b)^3 * (a + b)^2 = c^2 + 2 * (a - b) + 1 ↔ (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) :=
sorry

end tuples_satisfy_equation_l603_60332


namespace chinese_pig_problem_l603_60378

variable (x : ℕ)

theorem chinese_pig_problem :
  100 * x - 90 * x = 100 :=
sorry

end chinese_pig_problem_l603_60378


namespace angle_AMC_is_70_l603_60355

theorem angle_AMC_is_70 (A B C M : Type) (angle_MBA angle_MAB angle_ACB : ℝ) (AC BC : ℝ) :
  AC = BC → 
  angle_MBA = 30 → 
  angle_MAB = 10 → 
  angle_ACB = 80 → 
  ∃ angle_AMC : ℝ, angle_AMC = 70 :=
by
  sorry

end angle_AMC_is_70_l603_60355


namespace sum_infinite_series_l603_60352

theorem sum_infinite_series : ∑' k : ℕ, (k^2 : ℝ) / (3^k) = 7 / 8 :=
sorry

end sum_infinite_series_l603_60352


namespace geraldo_drank_7_pints_l603_60391

-- Conditions
def total_gallons : ℝ := 20
def num_containers : ℕ := 80
def gallons_to_pints : ℝ := 8
def containers_drank : ℝ := 3.5

-- Problem statement
theorem geraldo_drank_7_pints :
  let total_pints : ℝ := total_gallons * gallons_to_pints
  let pints_per_container : ℝ := total_pints / num_containers
  let pints_drank : ℝ := containers_drank * pints_per_container
  pints_drank = 7 :=
by
  sorry

end geraldo_drank_7_pints_l603_60391


namespace max_gold_coins_l603_60308

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 100) : n = 94 := by
  sorry

end max_gold_coins_l603_60308


namespace ratio_sum_of_squares_l603_60318

theorem ratio_sum_of_squares (a b c : ℕ) (h : a = 6 ∧ b = 1 ∧ c = 7 ∧ 72 / 98 = (a * (b.sqrt^2)).sqrt / c) : a + b + c = 14 := by 
  sorry

end ratio_sum_of_squares_l603_60318


namespace find_k_l603_60337

theorem find_k (k : ℝ) (h : ∀ x: ℝ, (x = -2) → (1 + k / (x - 1) = 0)) : k = 3 :=
by
  sorry

end find_k_l603_60337


namespace max_difference_proof_l603_60390

-- Define the revenue function R(x)
def R (x : ℕ+) : ℝ := 3000 * (x : ℝ) - 20 * (x : ℝ) ^ 2

-- Define the cost function C(x)
def C (x : ℕ+) : ℝ := 500 * (x : ℝ) + 4000

-- Define the profit function P(x) as revenue minus cost
def P (x : ℕ+) : ℝ := R x - C x

-- Define the marginal function M
def M (f : ℕ+ → ℝ) (x : ℕ+) : ℝ := f (⟨x + 1, Nat.succ_pos x⟩) - f x

-- Define the marginal profit function MP(x)
def MP (x : ℕ+) : ℝ := M P x

-- Statement of the proof
theorem max_difference_proof : 
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → P x ≤ P x_max) → -- P achieves its maximum at some x_max within constraints
  (∃ x_max : ℕ+, ∀ x : ℕ+, x ≤ 100 → MP x ≤ MP x_max) → -- MP achieves its maximum at some x_max within constraints
  (P x_max - MP x_max = 71680) := 
sorry -- proof omitted

end max_difference_proof_l603_60390


namespace slope_of_line_l603_60360

theorem slope_of_line (x y : ℝ) (h : 3 * y = 4 * x + 9) : 4 / 3 = 4 / 3 :=
by sorry

end slope_of_line_l603_60360


namespace distinct_real_roots_iff_l603_60326

theorem distinct_real_roots_iff (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (∀ x, x^2 + 3 * x - a = 0 → (x = x₁ ∨ x = x₂))) ↔ a > - (9 : ℝ) / 4 :=
sorry

end distinct_real_roots_iff_l603_60326


namespace probability_no_shaded_square_l603_60314

theorem probability_no_shaded_square : 
  let n : ℕ := 502 * 1004
  let m : ℕ := 502^2
  let total_rectangles := 3 * n
  let rectangles_with_shaded := 3 * m
  let probability_includes_shaded := rectangles_with_shaded / total_rectangles
  1 - probability_includes_shaded = (1 : ℚ) / 2 := 
by 
  sorry

end probability_no_shaded_square_l603_60314


namespace sin_product_identity_l603_60385

theorem sin_product_identity :
  (Real.sin (12 * Real.pi / 180)) * (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * (Real.sin (72 * Real.pi / 180)) = 1 / 16 := 
by 
  sorry

end sin_product_identity_l603_60385


namespace angle_sum_eq_pi_div_2_l603_60357

open Real

theorem angle_sum_eq_pi_div_2 (θ1 θ2 : ℝ) (h1 : 0 < θ1 ∧ θ1 < π / 2) (h2 : 0 < θ2 ∧ θ2 < π / 2)
  (h : (sin θ1)^2020 / (cos θ2)^2018 + (cos θ1)^2020 / (sin θ2)^2018 = 1) :
  θ1 + θ2 = π / 2 :=
sorry

end angle_sum_eq_pi_div_2_l603_60357


namespace find_f_2_l603_60388

variable (f : ℤ → ℤ)

-- Definitions of the conditions
def is_monic_quartic (f : ℤ → ℤ) : Prop :=
  ∃ a b c d, ∀ x, f x = x^4 + a * x^3 + b * x^2 + c * x + d

variable (hf_monic : is_monic_quartic f)
variable (hf_conditions : f (-2) = -4 ∧ f 1 = -1 ∧ f 3 = -9 ∧ f (-4) = -16)

-- The main statement to prove
theorem find_f_2 : f 2 = -28 := sorry

end find_f_2_l603_60388


namespace find_a_l603_60392

theorem find_a (a : ℝ) :
  (∀ x y, x + y = a → x^2 + y^2 = 4) →
  (∀ A B : ℝ × ℝ, (A.1 + A.2 = a ∧ B.1 + B.2 = a ∧ A.1^2 + A.2^2 = 4 ∧ B.1^2 + B.2^2 = 4) →
      ‖(A.1, A.2) + (B.1, B.2)‖ = ‖(A.1, A.2) - (B.1, B.2)‖) →
  a = 2 ∨ a = -2 :=
by
  intros line_circle_intersect vector_eq_magnitude
  sorry

end find_a_l603_60392


namespace integer_pairs_prime_P_l603_60317

theorem integer_pairs_prime_P (P : ℕ) (hP_prime : Prime P) 
  (h_condition : ∃ a b : ℤ, |a + b| + (a - b)^2 = P) : 
  P = 2 ∧ ((∃ a b : ℤ, |a + b| = 2 ∧ a - b = 0) ∨ 
           (∃ a b : ℤ, |a + b| = 1 ∧ (a - b = 1 ∨ a - b = -1))) :=
by
  sorry

end integer_pairs_prime_P_l603_60317


namespace twin_primes_solution_l603_60395

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ (p = q + 2 ∨ q = p + 2)

theorem twin_primes_solution (p q : ℕ) :
  are_twin_primes p q ∧ is_prime (p^2 - p * q + q^2) ↔ (p, q) = (5, 3) ∨ (p, q) = (3, 5) := by
  sorry

end twin_primes_solution_l603_60395


namespace find_positive_number_l603_60381

theorem find_positive_number (x : ℝ) (h : x > 0) (h1 : x + 17 = 60 * (1 / x)) : x = 3 :=
sorry

end find_positive_number_l603_60381


namespace solve_for_constants_l603_60387

def f (x : ℤ) (a b c : ℤ) : ℤ :=
if x > 0 then 2 * a * x + 4
else if x = 0 then a + b
else 3 * b * x + 2 * c

theorem solve_for_constants :
  ∃ a b c : ℤ, 
    f 1 a b c = 6 ∧ 
    f 0 a b c = 7 ∧ 
    f (-1) a b c = -4 ∧ 
    a + b + c = 14 :=
by
  sorry

end solve_for_constants_l603_60387


namespace expression_divisible_by_25_l603_60328

theorem expression_divisible_by_25 (n : ℕ) : 
    (2^(n+2) * 3^n + 5 * n - 4) % 25 = 0 :=
by {
  sorry
}

end expression_divisible_by_25_l603_60328


namespace y1_greater_than_y2_l603_60375

-- Define the function and points
def parabola (x : ℝ) (m : ℝ) : ℝ := x^2 - 4 * x + m

-- Define the points A and B on the parabola
def A_y1 (m : ℝ) : ℝ := parabola 0 m
def B_y2 (m : ℝ) : ℝ := parabola 1 m

-- Theorem statement
theorem y1_greater_than_y2 (m : ℝ) : A_y1 m > B_y2 m := 
  sorry

end y1_greater_than_y2_l603_60375


namespace inequality_x_pow_n_ge_n_x_l603_60369

theorem inequality_x_pow_n_ge_n_x (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x > -1) (h3 : n > 0) : 
  (1 + x)^n ≥ n * x := by
  sorry

end inequality_x_pow_n_ge_n_x_l603_60369


namespace quadratic_form_and_sum_l603_60362

theorem quadratic_form_and_sum (x : ℝ) : 
  ∃ (a b c : ℝ), 
  (15 * x^2 + 75 * x + 375 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 298.75) := 
sorry

end quadratic_form_and_sum_l603_60362


namespace problem_statement_l603_60393

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem problem_statement : same_terminal_side (-510) 210 :=
by
  sorry

end problem_statement_l603_60393


namespace fraction_inequality_l603_60334

theorem fraction_inequality (a b m : ℝ) (h1 : b > a) (h2 : m > 0) : 
  (b / a) > ((b + m) / (a + m)) :=
sorry

end fraction_inequality_l603_60334


namespace rowing_distance_l603_60313

theorem rowing_distance :
  let row_speed := 4 -- kmph
  let river_speed := 2 -- kmph
  let total_time := 1.5 -- hours
  ∃ d, 
    let downstream_speed := row_speed + river_speed
    let upstream_speed := row_speed - river_speed
    let downstream_time := d / downstream_speed
    let upstream_time := d / upstream_speed
    downstream_time + upstream_time = total_time ∧ d = 2.25 :=
by
  sorry

end rowing_distance_l603_60313


namespace ninety_eight_squared_l603_60306

theorem ninety_eight_squared : 98^2 = 9604 :=
by 
  -- The proof steps are omitted and replaced with 'sorry'
  sorry

end ninety_eight_squared_l603_60306


namespace largest_number_of_positive_consecutive_integers_l603_60366

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end largest_number_of_positive_consecutive_integers_l603_60366


namespace find_original_number_l603_60303

theorem find_original_number (x : ℝ) (h : 1.125 * x - 0.75 * x = 30) : x = 80 :=
by
  sorry

end find_original_number_l603_60303


namespace question_proof_l603_60335

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l603_60335


namespace percentage_of_Indian_women_l603_60396

-- Definitions of conditions
def total_people := 700 + 500 + 800
def indian_men := (20 / 100) * 700
def indian_children := (10 / 100) * 800
def total_indian_people := (21 / 100) * total_people
def indian_women := total_indian_people - indian_men - indian_children

-- Statement of the theorem
theorem percentage_of_Indian_women : 
  (indian_women / 500) * 100 = 40 :=
by
  sorry

end percentage_of_Indian_women_l603_60396


namespace area_inside_arcs_outside_square_l603_60386

theorem area_inside_arcs_outside_square (r : ℝ) (θ : ℝ) (L : ℝ) (a b c d : ℝ) :
  r = 6 ∧ θ = 45 ∧ L = 12 ∧ a = 15 ∧ b = 0 ∧ c = 15 ∧ d = 144 →
  (a + b + c + d = 174) :=
by
  intros h
  sorry

end area_inside_arcs_outside_square_l603_60386


namespace regression_coeff_nonzero_l603_60320

theorem regression_coeff_nonzero (a b r : ℝ) (h : b = 0 → r = 0) : b ≠ 0 :=
sorry

end regression_coeff_nonzero_l603_60320


namespace total_amount_l603_60367

theorem total_amount (x_share : ℝ) (y_share : ℝ) (w_share : ℝ) (hx : x_share = 0.30) (hy : y_share = 0.20) (hw : w_share = 10) :
  (w_share * (1 + x_share + y_share)) = 15 := by
  sorry

end total_amount_l603_60367


namespace monotonicity_of_f_extremum_of_f_on_interval_l603_60376

noncomputable def f (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem monotonicity_of_f : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → 1 ≤ x₂ → f x₁ < f x₂ := by
  sorry

theorem extremum_of_f_on_interval : 
  f 1 = 3 / 2 ∧ f 4 = 9 / 5 := by
  sorry

end monotonicity_of_f_extremum_of_f_on_interval_l603_60376


namespace find_first_two_solutions_l603_60310

theorem find_first_two_solutions :
  ∃ (n1 n2 : ℕ), 
    (n1 ≡ 3 [MOD 7]) ∧ (n1 ≡ 4 [MOD 9]) ∧ 
    (n2 ≡ 3 [MOD 7]) ∧ (n2 ≡ 4 [MOD 9]) ∧ 
    n1 < n2 ∧ 
    n1 = 31 ∧ n2 = 94 := 
by 
  sorry

end find_first_two_solutions_l603_60310


namespace number_of_students_selected_from_school2_l603_60324

-- Definitions from conditions
def total_students : ℕ := 360
def students_school1 : ℕ := 123
def students_school2 : ℕ := 123
def students_school3 : ℕ := 114
def selected_students : ℕ := 60
def initial_selected_from_school1 : ℕ := 1 -- Student 002 is already selected

-- Proportion calculation
def remaining_selected_students : ℕ := selected_students - initial_selected_from_school1
def remaining_students : ℕ := total_students - initial_selected_from_school1

-- Placeholder for calculation used in the proof
def students_selected_from_school2 : ℕ := 20

-- The Lean proof statement
theorem number_of_students_selected_from_school2 :
  students_selected_from_school2 =
  Nat.ceil ((students_school2 * remaining_selected_students : ℚ) / remaining_students) :=
sorry

end number_of_students_selected_from_school2_l603_60324


namespace find_f7_l603_60382

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x + 7

theorem find_f7 (a b : ℝ) (h : f (-7) a b = -17) : f (7) a b = 31 := 
by
  sorry

end find_f7_l603_60382


namespace find_m_of_equation_has_positive_root_l603_60331

theorem find_m_of_equation_has_positive_root :
  (∃ x : ℝ, 0 < x ∧ (x - 1) / (x - 5) = (m * x) / (10 - 2 * x)) → m = -8 / 5 :=
by
  sorry

end find_m_of_equation_has_positive_root_l603_60331


namespace december_25_is_thursday_l603_60341

theorem december_25_is_thursday (thanksgiving : ℕ) (h : thanksgiving = 27) :
  (∀ n, n % 7 = 0 → n + thanksgiving = 25 → n / 7 = 4) :=
by
  sorry

end december_25_is_thursday_l603_60341


namespace retailer_markup_percentage_l603_60350

-- Definitions of initial conditions
def CP : ℝ := 100
def intended_profit_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25
def actual_profit_percentage : ℝ := 0.2375

-- Proving the retailer marked his goods at 65% above the cost price
theorem retailer_markup_percentage : ∃ (MP : ℝ), ((0.75 * MP - CP) / CP) * 100 = actual_profit_percentage * 100 ∧ ((MP - CP) / CP) * 100 = 65 := 
by
  -- The mathematical proof steps mean to be filled here  
  sorry

end retailer_markup_percentage_l603_60350


namespace probability_non_obtuse_l603_60397

noncomputable def P_non_obtuse_triangle : ℚ :=
  (13 / 16) ^ 4

theorem probability_non_obtuse :
  P_non_obtuse_triangle = 28561 / 65536 :=
by
  sorry

end probability_non_obtuse_l603_60397


namespace cubics_of_sum_and_product_l603_60312

theorem cubics_of_sum_and_product (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 11) : 
  x^3 + y^3 = 670 :=
by
  sorry

end cubics_of_sum_and_product_l603_60312


namespace cos_30_eq_sqrt3_div_2_l603_60383

theorem cos_30_eq_sqrt3_div_2 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l603_60383


namespace range_of_a_l603_60363

theorem range_of_a (x : ℝ) (a : ℝ) (h1 : 2 < x) (h2 : a ≤ x + 1 / (x - 2)) : a ≤ 4 := 
sorry

end range_of_a_l603_60363


namespace correct_answer_l603_60311

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem correct_answer : M ⊆ N := by
  sorry

end correct_answer_l603_60311


namespace theta_quadrant_l603_60305

theorem theta_quadrant (θ : ℝ) (h : Real.sin (2 * θ) < 0) : 
  (Real.sin θ < 0 ∧ Real.cos θ > 0) ∨ (Real.sin θ > 0 ∧ Real.cos θ < 0) :=
sorry

end theta_quadrant_l603_60305


namespace find_number_subtract_four_l603_60358

theorem find_number_subtract_four (x : ℤ) (h : 35 + 3 * x = 50) : x - 4 = 1 := by
  sorry

end find_number_subtract_four_l603_60358


namespace carlson_max_jars_l603_60336

theorem carlson_max_jars (n a : ℕ) (h1 : 13 * n - a = 8 * (n + a)) : 
  ∃ (k : ℕ), a = 5 * k ∧ n = 9 * k ∧ 13 * n = 117 * k ∧ 23 ≤ 13 * k := by {
  sorry
}

end carlson_max_jars_l603_60336


namespace car_C_has_highest_average_speed_l603_60359

-- Define the distances traveled by each car
def distance_car_A_1st_hour := 140
def distance_car_A_2nd_hour := 130
def distance_car_A_3rd_hour := 120

def distance_car_B_1st_hour := 170
def distance_car_B_2nd_hour := 90
def distance_car_B_3rd_hour := 130

def distance_car_C_1st_hour := 120
def distance_car_C_2nd_hour := 140
def distance_car_C_3rd_hour := 150

-- Define the total distance and average speed calculations
def total_distance_car_A := distance_car_A_1st_hour + distance_car_A_2nd_hour + distance_car_A_3rd_hour
def total_distance_car_B := distance_car_B_1st_hour + distance_car_B_2nd_hour + distance_car_B_3rd_hour
def total_distance_car_C := distance_car_C_1st_hour + distance_car_C_2nd_hour + distance_car_C_3rd_hour

def total_time := 3

def average_speed_car_A := total_distance_car_A / total_time
def average_speed_car_B := total_distance_car_B / total_time
def average_speed_car_C := total_distance_car_C / total_time

-- Lean proof statement
theorem car_C_has_highest_average_speed :
  average_speed_car_C > average_speed_car_A ∧ average_speed_car_C > average_speed_car_B :=
by
  sorry

end car_C_has_highest_average_speed_l603_60359


namespace tallest_building_height_l603_60316

theorem tallest_building_height :
  ∃ H : ℝ, H + (1/2) * H + (1/4) * H + (1/20) * H = 180 ∧ H = 100 := by
  sorry

end tallest_building_height_l603_60316


namespace repetend_of_five_over_eleven_l603_60315

noncomputable def repetend_of_decimal_expansion (n d : ℕ) : ℕ := sorry

theorem repetend_of_five_over_eleven : repetend_of_decimal_expansion 5 11 = 45 :=
by sorry

end repetend_of_five_over_eleven_l603_60315


namespace one_room_cheaper_by_l603_60398

-- Define the initial prices of the apartments
variables (a b : ℝ)

-- Define the increase rates and the new prices
def new_price_one_room := 1.21 * a
def new_price_two_room := 1.11 * b
def new_total_price := 1.15 * (a + b)

-- The main theorem encapsulating the problem
theorem one_room_cheaper_by : a + b ≠ 0 → 1.21 * a + 1.11 * b = 1.15 * (a + b) → b / a = 1.5 :=
by
  intro h_non_zero h_prices
  -- we assume the main theorem is true to structure the goal state
  sorry

end one_room_cheaper_by_l603_60398


namespace circle_equation_l603_60319

theorem circle_equation
  (a b r : ℝ) 
  (h1 : a^2 + b^2 = r^2) 
  (h2 : (a - 2)^2 + b^2 = r^2) 
  (h3 : b / (a - 2) = 1) : 
  (x - 1)^2 + (y + 1)^2 = 2 := 
by
  sorry

end circle_equation_l603_60319


namespace dan_money_left_l603_60325

theorem dan_money_left
  (initial_amount : ℝ := 45)
  (cost_per_candy_bar : ℝ := 4)
  (num_candy_bars : ℕ := 4)
  (price_toy_car : ℝ := 15)
  (discount_rate_toy_car : ℝ := 0.10)
  (sales_tax_rate : ℝ := 0.05) :
  initial_amount - ((num_candy_bars * cost_per_candy_bar) + ((price_toy_car - (price_toy_car * discount_rate_toy_car)) * (1 + sales_tax_rate))) = 14.02 :=
by
  sorry

end dan_money_left_l603_60325


namespace capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l603_60347

noncomputable def company_capital (n : ℕ) : ℝ :=
  if n = 0 then 1000
  else 2 * company_capital (n - 1) - 500

theorem capital_at_end_of_2014 : company_capital 4 = 8500 :=
by sorry

theorem year_capital_exceeds_32dot5_billion : ∀ n : ℕ, company_capital n > 32500 → n ≥ 7 :=
by sorry

end capital_at_end_of_2014_year_capital_exceeds_32dot5_billion_l603_60347


namespace Bennett_has_6_brothers_l603_60309

theorem Bennett_has_6_brothers (num_aaron_brothers : ℕ) (num_bennett_brothers : ℕ) 
  (h1 : num_aaron_brothers = 4) 
  (h2 : num_bennett_brothers = 2 * num_aaron_brothers - 2) : 
  num_bennett_brothers = 6 := by
  sorry

end Bennett_has_6_brothers_l603_60309


namespace missing_digit_in_decimal_representation_of_power_of_two_l603_60322

theorem missing_digit_in_decimal_representation_of_power_of_two :
  (∃ m : ℕ, m < 10 ∧
   ∀ (n : ℕ), (0 ≤ n ∧ n < 10 → n ≠ m) →
     (45 - m) % 9 = (2^29) % 9) :=
sorry

end missing_digit_in_decimal_representation_of_power_of_two_l603_60322


namespace min_value_inverse_sum_l603_60368

variable (m n : ℝ)
variable (hm : 0 < m)
variable (hn : 0 < n)
variable (b : ℝ) (hb : b = 2)
variable (hline : 3 * m + n = 1)

theorem min_value_inverse_sum : 
  (1 / m + 4 / n) = 7 + 4 * Real.sqrt 3 :=
  sorry

end min_value_inverse_sum_l603_60368


namespace abs_inequality_solution_l603_60302

theorem abs_inequality_solution (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 7 ↔ (1 ≤ x ∧ x ≤ 5) ∨ (-9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end abs_inequality_solution_l603_60302


namespace on_real_axis_in_first_quadrant_on_line_l603_60351

theorem on_real_axis (m : ℝ) : 
  (m = -3 ∨ m = 5) ↔ (m^2 - 2 * m - 15 = 0) := 
sorry

theorem in_first_quadrant (m : ℝ) : 
  (m < -3 ∨ m > 5) ↔ ((m^2 + 5 * m + 6 > 0) ∧ (m^2 - 2 * m - 15 > 0)) := 
sorry

theorem on_line (m : ℝ) : 
  (m = 1 ∨ m = -5 / 2) ↔ ((m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) + 5 = 0) := 
sorry

end on_real_axis_in_first_quadrant_on_line_l603_60351


namespace total_books_l603_60346

def shelves : ℕ := 150
def books_per_shelf : ℕ := 15

theorem total_books (shelves books_per_shelf : ℕ) : shelves * books_per_shelf = 2250 := by
  sorry

end total_books_l603_60346


namespace find_a_b_l603_60304

noncomputable def f (a b x : ℝ) := b * a^x

def passes_through (a b : ℝ) : Prop :=
  f a b 1 = 27 ∧ f a b (-1) = 3

theorem find_a_b (a b : ℝ) (h : passes_through a b) : 
  a = 3 ∧ b = 9 :=
  sorry

end find_a_b_l603_60304


namespace three_digit_powers_of_two_count_l603_60379

theorem three_digit_powers_of_two_count : 
  ∃ n_count : ℕ, (∀ n : ℕ, (100 ≤ 2^n ∧ 2^n < 1000) ↔ (n = 7 ∨ n = 8 ∨ n = 9)) ∧ n_count = 3 :=
by
  sorry

end three_digit_powers_of_two_count_l603_60379


namespace outfit_count_l603_60329

def num_shirts := 8
def num_hats := 8
def num_pants := 4

def shirt_colors := 6
def hat_colors := 6
def pants_colors := 4

def total_possible_outfits := num_shirts * num_hats * num_pants

def same_color_restricted_outfits := 4 * 8 * 7

def num_valid_outfits := total_possible_outfits - same_color_restricted_outfits

theorem outfit_count (h1 : num_shirts = 8) (h2 : num_hats = 8) (h3 : num_pants = 4)
                     (h4 : shirt_colors = 6) (h5 : hat_colors = 6) (h6 : pants_colors = 4)
                     (h7 : total_possible_outfits = 256) (h8 : same_color_restricted_outfits = 224) :
  num_valid_outfits = 32 :=
by
  sorry

end outfit_count_l603_60329


namespace negate_proposition_l603_60389

theorem negate_proposition :
  (¬ (∀ x : ℝ, x > 2 → x^2 + 2 > 6)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) :=
by sorry

end negate_proposition_l603_60389


namespace b_c_value_l603_60354

theorem b_c_value (a b c d : ℕ) 
  (h₁ : a + b = 12) 
  (h₂ : c + d = 3) 
  (h₃ : a + d = 6) : 
  b + c = 9 :=
sorry

end b_c_value_l603_60354


namespace smallest_n_inequality_l603_60399

-- Define the main statement based on the identified conditions and answer.
theorem smallest_n_inequality (x y z w : ℝ) : 
  (x^2 + y^2 + z^2 + w^2)^2 ≤ 4 * (x^4 + y^4 + z^4 + w^4) :=
sorry

end smallest_n_inequality_l603_60399


namespace sum_mod_6_l603_60374

theorem sum_mod_6 :
  (60123 + 60124 + 60125 + 60126 + 60127 + 60128 + 60129 + 60130) % 6 = 4 :=
by
  sorry

end sum_mod_6_l603_60374


namespace tomato_plants_count_l603_60301

theorem tomato_plants_count :
  ∀ (sunflowers corn tomatoes total_rows plants_per_row : ℕ),
  sunflowers = 45 →
  corn = 81 →
  plants_per_row = 9 →
  total_rows = (sunflowers / plants_per_row) + (corn / plants_per_row) →
  tomatoes = total_rows * plants_per_row →
  tomatoes = 126 :=
by
  intros sunflowers corn tomatoes total_rows plants_per_row Hs Hc Hp Ht Hm
  rw [Hs, Hc, Hp] at *
  -- Additional calculation steps could go here to prove the theorem if needed
  sorry

end tomato_plants_count_l603_60301


namespace polynomial_has_real_root_l603_60377

open Real Polynomial

variable {c d : ℝ}
variable {P : Polynomial ℝ}

theorem polynomial_has_real_root (hP1 : ∀ n : ℕ, c * |(n : ℝ)|^3 ≤ |P.eval (n : ℝ)|)
                                (hP2 : ∀ n : ℕ, |P.eval (n : ℝ)| ≤ d * |(n : ℝ)|^3)
                                (hc : 0 < c) (hd : 0 < d) : 
                                ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l603_60377


namespace emily_euros_contribution_l603_60353

-- Declare the conditions as a definition
def conditions : Prop :=
  ∃ (cost_of_pie : ℝ) (emily_usd : ℝ) (berengere_euros : ℝ) (exchange_rate : ℝ),
    cost_of_pie = 15 ∧
    emily_usd = 10 ∧
    berengere_euros = 3 ∧
    exchange_rate = 1.1

-- Define the proof problem based on the conditions and required contribution
theorem emily_euros_contribution : conditions → (∃ emily_euros_more : ℝ, emily_euros_more = 3) :=
by
  intro h
  sorry

end emily_euros_contribution_l603_60353


namespace fraction_condition_l603_60343

theorem fraction_condition (x : ℝ) (h₁ : x > 1) (h₂ : 1 / x < 1) : false :=
sorry

end fraction_condition_l603_60343


namespace sin_sum_cos_product_l603_60365

theorem sin_sum_cos_product (A B C : Real) (h : A + B + C = π) :
  Real.sin A + Real.sin B + Real.sin C = 4 * Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2) :=
by
  sorry

end sin_sum_cos_product_l603_60365


namespace triangle_proof_l603_60364

noncomputable def triangle_math_proof (A B C : ℝ) (AA1 BB1 CC1 : ℝ) : Prop :=
  AA1 = 2 * Real.sin (B + A / 2) ∧
  BB1 = 2 * Real.sin (C + B / 2) ∧
  CC1 = 2 * Real.sin (A + C / 2) ∧
  (Real.sin A + Real.sin B + Real.sin C) ≠ 0 ∧
  ∀ x, x = (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) / (Real.sin A + Real.sin B + Real.sin C) → x = 2

theorem triangle_proof (A B C AA1 BB1 CC1 : ℝ) (h : triangle_math_proof A B C AA1 BB1 CC1) :
  (AA1 * Real.cos (A / 2) + BB1 * Real.cos (B / 2) + CC1 * Real.cos (C / 2)) /
  (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end triangle_proof_l603_60364


namespace compare_neg_rational_numbers_l603_60321

theorem compare_neg_rational_numbers :
  - (3 / 2) > - (5 / 3) := 
sorry

end compare_neg_rational_numbers_l603_60321


namespace opposite_sides_line_l603_60339

theorem opposite_sides_line (a : ℝ) : (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 := by
  sorry

end opposite_sides_line_l603_60339


namespace parabola_behavior_l603_60333

-- Definitions for the conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- The proof statement
theorem parabola_behavior (a : ℝ) (x : ℝ) (ha : 0 < a) : 
  (0 < a ∧ a < 1 → parabola a x < x^2) ∧
  (a > 1 → parabola a x > x^2) ∧
  (∀ ε > 0, ∃ δ > 0, δ ≤ a → |parabola a x - 0| < ε) := 
sorry

end parabola_behavior_l603_60333


namespace andre_tuesday_ladybugs_l603_60344

theorem andre_tuesday_ladybugs (M T : ℕ) (dots_per_ladybug total_dots monday_dots tuesday_dots : ℕ)
  (h1 : M = 8)
  (h2 : dots_per_ladybug = 6)
  (h3 : total_dots = 78)
  (h4 : monday_dots = M * dots_per_ladybug)
  (h5 : tuesday_dots = total_dots - monday_dots)
  (h6 : tuesday_dots = T * dots_per_ladybug) :
  T = 5 :=
sorry

end andre_tuesday_ladybugs_l603_60344


namespace max_n_value_l603_60340

theorem max_n_value (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
by
  sorry

end max_n_value_l603_60340


namespace thin_film_radius_volume_l603_60356

theorem thin_film_radius_volume :
  ∀ (r : ℝ) (V : ℝ) (t : ℝ), 
    V = 216 → t = 0.1 → π * r^2 * t = V → r = Real.sqrt (2160 / π) :=
by
  sorry

end thin_film_radius_volume_l603_60356


namespace slope_negative_l603_60373

theorem slope_negative (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → mx1 + 5 > mx2 + 5) → m < 0 :=
by
  sorry

end slope_negative_l603_60373


namespace greatest_value_sum_eq_24_l603_60394

theorem greatest_value_sum_eq_24 {a b : ℕ} (ha : 0 < a) (hb : 1 < b) 
  (h1 : ∀ (x y : ℕ), 0 < x → 1 < y → x^y < 500 → x^y ≤ a^b) : 
  a + b = 24 := 
by
  sorry

end greatest_value_sum_eq_24_l603_60394


namespace total_animals_l603_60327

theorem total_animals (initial_elephants initial_hippos : ℕ) 
  (ratio_female_hippos : ℚ)
  (births_per_female_hippo : ℕ)
  (newborn_elephants_diff : ℕ)
  (he : initial_elephants = 20)
  (hh : initial_hippos = 35)
  (rfh : ratio_female_hippos = 5 / 7)
  (bpfh : births_per_female_hippo = 5)
  (ned : newborn_elephants_diff = 10) :
  ∃ (total_animals : ℕ), total_animals = 315 :=
by sorry

end total_animals_l603_60327


namespace unique_integer_solution_quad_eqns_l603_60384

def is_single_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem unique_integer_solution_quad_eqns : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ is_single_digit_prime a ∧ is_single_digit_prime b ∧ is_single_digit_prime c ∧ 
                     ∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ S.card = 7 :=
by
  sorry

end unique_integer_solution_quad_eqns_l603_60384


namespace transportation_degrees_l603_60338

theorem transportation_degrees
  (salaries : ℕ) (r_and_d : ℕ) (utilities : ℕ) (equipment : ℕ) (supplies : ℕ) (total_degrees : ℕ)
  (h_salaries : salaries = 60)
  (h_r_and_d : r_and_d = 9)
  (h_utilities : utilities = 5)
  (h_equipment : equipment = 4)
  (h_supplies : supplies = 2)
  (h_total_degrees : total_degrees = 360) :
  (total_degrees * (100 - (salaries + r_and_d + utilities + equipment + supplies)) / 100 = 72) :=
by {
  sorry
}

end transportation_degrees_l603_60338


namespace chef_served_173_guests_l603_60345

noncomputable def total_guests_served : ℕ :=
  let adults := 58
  let children := adults - 35
  let seniors := 2 * children
  let teenagers := seniors - 15
  let toddlers := teenagers / 2
  adults + children + seniors + teenagers + toddlers

theorem chef_served_173_guests : total_guests_served = 173 :=
  by
    -- Proof will be provided here.
    sorry

end chef_served_173_guests_l603_60345


namespace dessert_probability_l603_60348

noncomputable def P (e : Prop) : ℝ := sorry

variables (D C : Prop)

theorem dessert_probability 
  (P_D : P D = 0.6)
  (P_D_and_not_C : P (D ∧ ¬C) = 0.12) :
  P (¬ D) = 0.4 :=
by
  -- Proof is skipped using sorry, as instructed.
  sorry

end dessert_probability_l603_60348


namespace soda_price_increase_l603_60371

theorem soda_price_increase (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  -- Proof will be provided here
  sorry

end soda_price_increase_l603_60371


namespace find_prime_p_l603_60361

theorem find_prime_p
  (p : ℕ)
  (h_prime_p : Nat.Prime p)
  (h : Nat.Prime (p^3 + p^2 + 11 * p + 2)) :
  p = 3 :=
sorry

end find_prime_p_l603_60361


namespace parabola_vertex_on_x_axis_l603_60323

theorem parabola_vertex_on_x_axis (c : ℝ) : 
    (∃ h k, h = -3 ∧ k = 0 ∧ ∀ x, x^2 + 6 * x + c = x^2 + 6 * x + (c - (h^2)/4)) → c = 9 :=
by
    sorry

end parabola_vertex_on_x_axis_l603_60323
