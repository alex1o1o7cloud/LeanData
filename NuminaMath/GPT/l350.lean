import Mathlib

namespace sum_of_coefficients_l350_350120

theorem sum_of_coefficients :
  let f (x : ℝ) := (1 - 2 * x)^7
  ∃ a : ℕ → ℝ, (∀ x, f x = ∑ i in Finset.range 8, a i * x^i) → (a 0 = 1) → (∑ i in (Finset.range 8).erase 0, a i = -2) :=
begin
  intro f,
  existsi (λ n, (EuclideanDomain.coeff n (f x))),
  assume h a0_def,
  simp [f, a0_def],
  sorry
end

end sum_of_coefficients_l350_350120


namespace expression_equals_four_l350_350075

-- Definition of the problem
def expression : Real := Real.abs (-3) + Real.sqrt 3 * Real.sin (π/3) - (2⁻¹)

-- The statement of the problem
theorem expression_equals_four : expression = 4 := 
by
  sorry

end expression_equals_four_l350_350075


namespace sqrt_450_eq_15_sqrt_2_l350_350550

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350550


namespace sqrt_450_equals_15_sqrt_2_l350_350601

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350601


namespace cos_B_value_l350_350180

theorem cos_B_value (B : ℝ) (h1 : Real.tan B + Real.sec B = 3) : Real.cos B = 3 / 5 :=
sorry

end cos_B_value_l350_350180


namespace increase_fraction_l350_350107

theorem increase_fraction (A F : ℝ) 
  (h₁ : A = 83200) 
  (h₂ : A * (1 + F) ^ 2 = 105300) : 
  F = 0.125 :=
by
  sorry

end increase_fraction_l350_350107


namespace log_func_inverse_l350_350930

def f (a x : ℝ) := Real.log x / Real.log a

theorem log_func_inverse :
  ∀ (a : ℝ), a > 0 → a ≠ 1 → (f a (1 / 8) = 3) → (f a (1 / 4) = 2) := by
  intros a ha1 ha2 h_inverse
  sorry

end log_func_inverse_l350_350930


namespace effective_selling_price_correct_gain_percent_correct_l350_350805

def cycle_cost : Real := 930
def selling_price : Real := 1210
def discount_rate : Real := 8 / 100

def effective_selling_price := selling_price * (1 - discount_rate)
def gain := effective_selling_price - cycle_cost
def gain_percent := (gain / cycle_cost) * 100

theorem effective_selling_price_correct :
  effective_selling_price = 1113.2 :=
by
  sorry

theorem gain_percent_correct :
  gain_percent = 19.7 :=
by
  sorry

end effective_selling_price_correct_gain_percent_correct_l350_350805


namespace inequality_always_true_l350_350787

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ a > 1/4 :=
sorry

end inequality_always_true_l350_350787


namespace simplify_sqrt_450_l350_350636
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350636


namespace current_balance_after_deduction_l350_350093

theorem current_balance_after_deduction :
  ∀ (original_balance deduction_percent : ℕ), 
  original_balance = 100000 →
  deduction_percent = 10 →
  original_balance - (deduction_percent * original_balance / 100) = 90000 :=
by
  intros original_balance deduction_percent h1 h2
  sorry

end current_balance_after_deduction_l350_350093


namespace passage_through_midpoint_l350_350821

open Point

-- Definitions of points and conditions
variable {A B C D E X M : Point}

-- Conditions given in the problem
axiom incircle_touch (A B C Circle) : Incircle A B C Circle
axiom line_through_A (A D E Line) : LineThrough A D E Line
axiom BX_parallel_DE (BX DE : Line) : Parallel BX DE

-- We are to prove that if CX and DE intersect at M, then M is the midpoint of DE
axiom CX_intersect_DE_at_M (CX DE : Line) (M : Point) : Intersect CX DE M

-- The statement to prove
theorem passage_through_midpoint (A B C D E X M : Point) (h1 : incircle_touch A B C Circle)
  (h2 : line_through_A A D E Line) (h3 : BX_parallel_DE BX DE)
  (h4 : CX_intersect_DE_at_M CX DE M) :
  Midpoint M D E :=
by
  sorry

end passage_through_midpoint_l350_350821


namespace externally_tangent_circles_l350_350951

theorem externally_tangent_circles (a b : ℝ) 
  (h : abs ((1 : ℝ)) = 1 - abs ((1/(2 * sqrt (a^2 + b^2)))) → sqrt (a^2 + b^2) = 2) :
  let C1 := { p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 1 },
      C2 := { p : ℝ × ℝ | p.1^2 + (p.2 - b)^2 = 1 } in
  ∃ (p : ℝ × ℝ), (p ∈ C1 ∧ p ∈ C2) ∧ (∀ q, q ∈ C1 ∧ q ∈ C2 → p = q) :=
by
  intros
  sorry

end externally_tangent_circles_l350_350951


namespace centers_of_ngons_equilateral_l350_350280

noncomputable def third_root_of_unity (α : ℂ) : Prop :=
  α ≠ 1 ∧ α * α * α = 1

theorem centers_of_ngons_equilateral (A B C : ℂ) (n : ℕ) (O1 O2 O3 : ℂ)
    (hO1 : (O1 - A) / (O1 - B) = complex.exp((2 * π * I) / n))
    (hO2 : (O2 - B) / (O2 - C) = complex.exp((2 * π * I) / n))
    (hO3 : (O3 - C) / (O3 - A) = complex.exp((2 * π * I) / n)) :
    ((O1 + O2 * complex.exp((2 * π * I) / 3)) + O3 * complex.exp((4 * π * I) / 3) = 0) ↔ n = 3 :=
  sorry

end centers_of_ngons_equilateral_l350_350280


namespace sum_of_solutions_l350_350871

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | 2^(x^2 - 5*x) = 8^(x - 3)}, x) = 10 :=
sorry

end sum_of_solutions_l350_350871


namespace arithmetic_sequence_common_difference_l350_350982

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (ha2 : a 2 = 2)
  (hS7 : ∑ i in finset.range 7, a (i + 1) = 56) :
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l350_350982


namespace inverse_matrices_sum_l350_350724

theorem inverse_matrices_sum (a b c f : ℚ)
  (h : (matrix.of ![(a, 1), (c, 2)] : matrix (fin 2) (fin 2) ℚ) * (matrix.of ![(4, b), (f, 3)] : matrix (fin 2) (fin 2) ℚ) = 1) :
  a + b + c + f = -3 / 2 :=
begin
  sorry
end

end inverse_matrices_sum_l350_350724


namespace sqrt_450_eq_15_sqrt_2_l350_350565

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350565


namespace winning_candidate_percentage_l350_350789

theorem winning_candidate_percentage (votes1 votes2 votes3 : ℕ) (total_votes : ℕ) (winning_votes : ℕ) :
  votes1 = 1136 → votes2 = 7636 → votes3 = 10628 → total_votes = votes1 + votes2 + votes3 → winning_votes = votes3 →
  (winning_votes : ℝ) / total_votes * 100 ≈ 54.78 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end winning_candidate_percentage_l350_350789


namespace base2_bits_of_1A1A1_l350_350768

-- Define the base-16 number 1A1A1_16
def hex_number : ℕ := 1 * 16^4 + 10 * 16^3 + 1 * 16^2 + 10 * 16^1 + 1 * 16^0

-- Prove that the number of bits in the base-2 representation of hex_number is 17.
theorem base2_bits_of_1A1A1 :
  (nat.log2 hex_number + 1) = 17 :=
by
  sorry

end base2_bits_of_1A1A1_l350_350768


namespace second_race_distance_remaining_l350_350798

theorem second_race_distance_remaining
  (race_distance : ℕ)
  (A_finish_time : ℕ)
  (B_remaining_distance : ℕ)
  (A_start_behind : ℕ)
  (A_speed : ℝ)
  (B_speed : ℝ)
  (A_distance_second_race : ℕ)
  (B_distance_second_race : ℝ)
  (v_ratio : ℝ)
  (B_remaining_second_race : ℝ) :
  race_distance = 10000 →
  A_finish_time = 50 →
  B_remaining_distance = 500 →
  A_start_behind = 500 →
  A_speed = race_distance / A_finish_time →
  B_speed = (race_distance - B_remaining_distance) / A_finish_time →
  v_ratio = A_speed / B_speed →
  v_ratio = 20 / 19 →
  A_distance_second_race = race_distance + A_start_behind →
  B_distance_second_race = B_speed * (A_distance_second_race / A_speed) →
  B_remaining_second_race = race_distance - B_distance_second_race →
  B_remaining_second_race = 25 := 
by
  sorry

end second_race_distance_remaining_l350_350798


namespace area_is_one_third_l350_350833

noncomputable def area_enclosed_by_parabola_and_line : ℝ :=
  ∫ x in 0..1, (2 * x - 2 * x^2)

theorem area_is_one_third : area_enclosed_by_parabola_and_line = 1 / 3 :=
sorry

end area_is_one_third_l350_350833


namespace find_point_B_l350_350813

noncomputable def find_intersection_point (A B C : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) :=
  let t := (-snd A + ((plane A) - plane (0,0,0))) / (fst A + snd (snd A) + thd A)
  in (fst A + t * (fst B - fst A), snd A, thd A)

theorem find_point_B
  (A B C : ℝ × ℝ × ℝ)
  (plane : ℝ × ℝ × ℝ → ℝ) 
  (hA : A = (-2, 10, 12))
  (hC : C = (4, 4, 8))
  (h_plane : plane = λ p : ℝ × ℝ × ℝ, (p.1 + 2 * p.2 + 3 * p.3 - 20))
  :
  find_intersection_point A B C plane = (-36, 10, 12) :=
by
  sorry

end find_point_B_l350_350813


namespace area_of_quadrilateral_AOBD_l350_350828

open Point

theorem area_of_quadrilateral_AOBD (x1 x2 : ℝ) (h1 : x1² + x2² = 3 / 2) : 
  let y1 := x1^2,
      y2 := x2^2,
      C := ( (x1 + x2) / 2, (y1 + y2) / 2 ),
      k_AB := x1 + x2,
      k_CD := -1 / (x1 + x2),
      D_y := ((y1 + y2) / 2) + 1/2 in
  (0 - 0)^2 + (D_y - 0)^2 = (5 / 4)^2 ∧
  (x1 + x2)^2 = 1 ∧
  x1 + x2 = ± 1 ∧
  (1 / 2) * 2 * (5 / 4) * (Real.sin (π / 4)) = (5 * Real.sqrt 2 / 8) := 
sorry

end area_of_quadrilateral_AOBD_l350_350828


namespace min_payment_max_payment_expected_value_payment_l350_350335

-- Proof Problem 1
theorem min_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let optimized_groups := [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]] in
  (∑ g in optimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4000 := by
  sorry

-- Proof Problem 2
theorem max_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let suboptimized_groups := [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]] in
  (∑ g in suboptimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4900 := by
  sorry

-- Proof Problem 3
theorem expected_value_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let expected_savings := 100 * (∑ k in List.range 9, k * ((10 - k) * (9 - k)) / 72) in
  (5500 - expected_savings) = 4583.33 := by 
  sorry

end min_payment_max_payment_expected_value_payment_l350_350335


namespace num_terms_100_pow_10_as_sum_of_tens_l350_350211

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l350_350211


namespace simplify_sqrt_450_l350_350371

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350371


namespace ratio_of_cookies_l350_350309

theorem ratio_of_cookies (cookies_total cookies_left cookies_father : ℕ)
  (cookies_brother_more : ℕ) :
  cookies_total = 30 →
  cookies_left = 8 →
  cookies_father = 10 →
  cookies_brother_more = 2 →
  let cookies_eaten_by_family := cookies_total - cookies_left in
  let cookies_mother := (cookies_eaten_by_family - cookies_father - cookies_brother_more) / 2 in
  cookies_mother * 2 + cookies_brother_more = cookies_eaten_by_family - cookies_father →
  (cookies_mother : ℚ) / (cookies_father : ℚ) = 1 / 2 :=
begin
  intros h_total h_left h_father h_brother_more,
  let cookies_eaten_by_family := cookies_total - cookies_left,
  let cookies_mother := (cookies_eaten_by_family - cookies_father - cookies_brother_more) / 2,
  have h_family := (by linarith : cookies_mother * 2 + cookies_brother_more = cookies_eaten_by_family - cookies_father),
  exact (by norm_cast; field_simp [h_total, h_left, h_father, h_brother_more, cookies_eaten_by_family, cookies_mother,
    ne_of_gt (by linarith), ne_of_gt (by linarith)] : (cookies_mother : ℚ) / (cookies_father : ℚ) = 1 / 2)
end

end ratio_of_cookies_l350_350309


namespace tan_alpha_value_sin_cos_expression_l350_350122

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_value (α : ℝ) (h1 : Real.tan (α + Real.pi / 4) = 2) : tan_alpha α = 1 / 3 :=
by
  sorry

theorem sin_cos_expression (α : ℝ) (h2 : tan_alpha α = 1 / 3) :
  (Real.sin (2 * α) - Real.sin α ^ 2) / (1 + Real.cos (2 * α)) = 5 / 18 :=
by
  sorry

end tan_alpha_value_sin_cos_expression_l350_350122


namespace remaining_quadrilateral_perimeter_l350_350824

theorem remaining_quadrilateral_perimeter 
  (A B C D E : Type) 
  [equilateral_triangle : triangle ABC]
  (side_ABC : side_length ABC = 6)
  (isosceles_triangle : triangle DBE)
  (side_DB : side_length DB = 2)
  (side_EB : side_length EB = 2)
  (base_DE : side_length DE = 4) :
  perimeter (quadrilateral ACED) = 18 :=
sorry

end remaining_quadrilateral_perimeter_l350_350824


namespace simplify_sqrt_450_l350_350440

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350440


namespace smallest_perimeter_ABCD_l350_350994

-- Variables representing the given conditions
variable {A B C D : Type} [Point A] [Point B] [Point C] [Point D]
variable (AD : ℝ) (BC : ℝ) (area_ABC : ℝ) (area_DBC : ℝ)

-- These are the given values in the problem
axiom h1 : AD = 20
axiom h2 : BC = 13
axiom h3 : area_ABC = 338
axiom h4 : area_DBC = 212

-- The theorem stating that the smallest possible perimeter of quadrilateral ABCD is 118
theorem smallest_perimeter_ABCD : 
  ∀ (AD BC area_ABC area_DBC : ℝ), 
  AD = 20 → BC = 13 → area_ABC = 338 → area_DBC = 212 → 
  ∃ (p : ℝ), p = 118 :=
by 
    -- proof omitted
    sorry

end smallest_perimeter_ABCD_l350_350994


namespace exp_monotonic_iff_l350_350117

theorem exp_monotonic_iff (a b : ℝ) : (a > b) ↔ (Real.exp a > Real.exp b) :=
sorry

end exp_monotonic_iff_l350_350117


namespace time_after_12345_seconds_l350_350262

def initial_time := (5, 45, 0)  -- (hours, minutes, seconds)
def total_seconds := 12345
def seconds_in_minute := 60
def minutes_in_hour := 60

noncomputable def final_time : (ℕ × ℕ × ℕ) :=
  let minutes := total_seconds / seconds_in_minute
  let seconds := total_seconds % seconds_in_minute
  let hours := minutes / minutes_in_hour
  let remaining_minutes := minutes % minutes_in_hour
  let new_hours := hours + initial_time.1
  let new_minutes := remaining_minutes + initial_time.2
  
  if new_minutes >= minutes_in_hour then
    (new_hours + 1, new_minutes - minutes_in_hour, seconds + initial_time.3)  -- Adjust for excess minutes
  else
    (new_hours, new_minutes, seconds + initial_time.3)

theorem time_after_12345_seconds :
  final_time = (9, 10, 45) :=
by
  sorry

end time_after_12345_seconds_l350_350262


namespace simplify_sqrt_450_l350_350544

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350544


namespace min_squares_to_remove_for_tiling_l350_350023

def is_t_tetromino (squares : set (ℕ × ℕ)) : Prop :=
  ∃ (p : ℕ × ℕ), squares = {(p.1, p.2), (p.1 + 1, p.2), (p.1 + 2, p.2), (p.1 + 1, p.2 + 1)}

def valid_tiling (grid: set (ℕ × ℕ)) (tiles : set (set (ℕ × ℕ))) : Prop :=
  (∀ t ∈ tiles, is_t_tetromino t) ∧ 
  (⋃ t ∈ tiles, t) ⊆ grid ∧ 
  (grid \ ⋃ t ∈ tiles, t).size = 4

theorem min_squares_to_remove_for_tiling :
  ∃ (tiles : set (set (ℕ × ℕ))), valid_tiling (univ.filter (λ p : ℕ × ℕ, p.1 < 202 ∧ p.2 < 202)) tiles :=
by
  sorry

end min_squares_to_remove_for_tiling_l350_350023


namespace original_number_of_people_l350_350171

theorem original_number_of_people (x : ℕ) 
  (h1 : (x / 2) - ((x / 2) / 3) = 12) : 
  x = 36 :=
sorry

end original_number_of_people_l350_350171


namespace simplify_sqrt_450_l350_350640
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350640


namespace max_dominoes_l350_350296

theorem max_dominoes (m n : ℕ) (h : n ≥ m) :
  ∃ k, k = m * n - (m / 2 : ℕ) :=
by sorry

end max_dominoes_l350_350296


namespace simplify_sqrt_450_l350_350531

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350531


namespace simplify_sqrt_450_l350_350451

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350451


namespace sqrt_450_simplified_l350_350427

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350427


namespace integral_computation_l350_350859

noncomputable def definite_integral : ℝ := ∫ x in 0..1, (sqrt(1 - (x - 1) ^ 2) - x)

theorem integral_computation : definite_integral = (π - 2) / 4 :=
  by
  sorry

end integral_computation_l350_350859


namespace simplify_sqrt_450_l350_350505

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350505


namespace problem_1_problem_2_l350_350927

noncomputable def f (x : ℝ) := Real.sin x + (x - 1) / Real.exp x

theorem problem_1 (x : ℝ) (h₀ : x ∈ Set.Icc (-Real.pi) (Real.pi / 2)) :
  MonotoneOn f (Set.Icc (-Real.pi) (Real.pi / 2)) :=
sorry

theorem problem_2 (k : ℝ) :
  ∀ x ∈ Set.Icc (-Real.pi) 0, ((f x - Real.sin x) * Real.exp x - Real.cos x) ≤ k * Real.sin x → 
  k ∈ Set.Iic (1 + Real.pi / 2) :=
sorry

end problem_1_problem_2_l350_350927


namespace arrangement_probability_l350_350065

theorem arrangement_probability :
  let total_arrangements := 720
  let valid_arrangements := 288
  valid_arrangements / total_arrangements = 2 / 5 :=
by sorry

end arrangement_probability_l350_350065


namespace sum_ge_n_div_2n_minus_1_l350_350996

open BigOperators

theorem sum_ge_n_div_2n_minus_1 (n : ℕ) (x : Finₓ n → ℝ) (h_pos : ∀ i, 0 < x i)
  (h_sum : ∑ i, x i = 1) :
  ∑ i, x i / (2 - x i) ≥ n / (2 * n - 1) :=
by
  sorry

end sum_ge_n_div_2n_minus_1_l350_350996


namespace diagonals_in_decagon_l350_350049

theorem diagonals_in_decagon : 
  ∀ n : ℕ, n = 10 → (n * (n - 3)) / 2 = 35 := by
  intros n hn
  rw [hn]
  norm_num

end diagonals_in_decagon_l350_350049


namespace student_ratio_l350_350235

theorem student_ratio (total_students below_eight eight_years above_eight : ℕ) 
  (h1 : below_eight = total_students * 20 / 100) 
  (h2 : eight_years = 72) 
  (h3 : total_students = 150) 
  (h4 : total_students = below_eight + eight_years + above_eight) :
  (above_eight / eight_years) = 2 / 3 :=
by
  sorry

end student_ratio_l350_350235


namespace sqrt_450_eq_15_sqrt_2_l350_350568

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350568


namespace sum_of_tens_l350_350202

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l350_350202


namespace first_digit_of_number_l350_350795

theorem first_digit_of_number (n : ℕ) (last_digit : ℕ) (divisible_digits : ℕ → Prop) :
  n = 1992 →
  (∀ d1 d2 : ℕ, d1 * 10 + d2 ∈ {17, 34, 51, 68, 85, 23, 46, 69, 92} → divisible_digits (d1 * 10 + d2)) →
  last_digit = 1 →
  ∃ first_digit : ℕ, first_digit = 2 :=
by
  intros h1 h2 h3
  sorry

end first_digit_of_number_l350_350795


namespace find_vector_pointing_to_line_l350_350845

-- Definitions from conditions
def parameterized_line (t : ℝ) : ℝ × ℝ :=
  (4 * t + 2, t + 2)

def direction_vector : ℝ × ℝ :=
  (2, 1)

-- Conjecture to be proven
theorem find_vector_pointing_to_line (a b : ℝ) :
  (a, b) = (6, 3) →
  ∃ t : ℝ, parameterized_line t = (a, b) ∧ (a, b) = (2 * k, k) :=
begin
  sorry
end

end find_vector_pointing_to_line_l350_350845


namespace simplify_sqrt_450_l350_350393

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350393


namespace simplify_sqrt_450_l350_350655
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350655


namespace locus_of_midpoints_of_altitudes_l350_350807

variables {A B C H A1 B1 A2 B2 P Q : Type*}
variables [EuclideanGeometry A B C H A1 B1 A2 B2 P Q]

def is_circumcircle_of_triangle (O : Type*) (A B C : Type*) : Prop := sorry
def is_orthocenter (H : Type*) (A B C : Type*) : Prop := sorry
def foot_of_altitude (P : Type*) (X Y Z : Type*) : Prop := sorry
def is_midpoint (M : Type*) (X Y : Type*) : Prop := sorry
def moves_along_circle (P : Type*) (C : Type*) : Prop := sorry

theorem locus_of_midpoints_of_altitudes
  (acute_triangle : Type*) (circumcircle : Type*) (A B C H : Type*)
  (H_moves : moves_along_circle H C)
  (H_is_orthocenter : is_orthocenter H A B C)
  (O_is_circumcircle : is_circumcircle_of_triangle circumcircle A B C)
  (A1_is_foot : foot_of_altitude A1 A B C)
  (B1_is_foot : foot_of_altitude B1 B A C)
  (K_is_midpoint : is_midpoint K A1 B1)
  (Q_is_midpoint : is_midpoint Q C P)
  :
  locus_of_midpoints (K : Type*) (A B H K P Q) = arc_of_circle_centered_at_radius (Q) (CH / 2)
  (bounded_by (: Type*) (A2 B2)) :=
sorry

end locus_of_midpoints_of_altitudes_l350_350807


namespace simplify_sqrt_450_l350_350659

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350659


namespace sqrt_450_eq_15_sqrt_2_l350_350691

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350691


namespace simplify_sqrt_450_l350_350542

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350542


namespace div_sqrt3_mul_inv_sqrt3_eq_one_l350_350836

theorem div_sqrt3_mul_inv_sqrt3_eq_one :
  (3 / Real.sqrt 3) * (1 / Real.sqrt 3) = 1 :=
by
  sorry

end div_sqrt3_mul_inv_sqrt3_eq_one_l350_350836


namespace parabola_focus_eq_line_through_point_intersect_parabola_l350_350160

noncomputable def parabola_equation (p : ℝ) (C : set (ℝ × ℝ)) := ∃ x y, (y^2 = 2 * p * x ∧ C = (set.preimage (λ (x y : ℝ), (x, y)) (y^2 = 2 * p * x)))

theorem parabola_focus_eq (p : ℝ) (C : set (ℝ × ℝ)) (h1: p > 0) (hF: (1, 0) = (focus_of_parabola y^2 = 2 * p * x)) : y^2 = 4 * x :=
sorry

theorem line_through_point_intersect_parabola (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) (F : ℝ × ℝ)
(hline: ∃ (k : ℝ), l = λ (x : ℝ), (x, k * (x + 1)))
(hintersect: parabola_intersection l C = {A, B})
(hpoint: l (-1) = (l (-1).1, 0))
(hdistance: 2 * distance F B = distance F A):
(∃ k : ℝ, (k = 2 * sqrt 2 / 3 ∨ k = -2 * sqrt 2 / 3) ∧ ∀ x, l x = (x, k * (x + 1))) :=
sorry

end parabola_focus_eq_line_through_point_intersect_parabola_l350_350160


namespace max_n_distinct_squares_sum_3000_l350_350760

theorem max_n_distinct_squares_sum_3000 :
  ∃ (k : ℕ → ℕ),
    (∀ i j, i ≠ j → k i ≠ k j) ∧ 
    (∃ n, (∑ i in range n, (k i)^2 = 3000) → n = 19) :=
sorry

end max_n_distinct_squares_sum_3000_l350_350760


namespace product_of_two_numbers_l350_350727

theorem product_of_two_numbers (a b : ℕ) (hcf : ℕ) (lcm : ℕ) (hcf_eq : hcf = 22) (lcm_eq : lcm = 2058) (hcf_lcm_eq : a * b = hcf * lcm) : a * b = 45276 := 
by
  rw [hcf_eq, lcm_eq] at hcf_lcm_eq
  exact hcf_lcm_eq
  sorry

end product_of_two_numbers_l350_350727


namespace simplify_sqrt_450_l350_350642
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350642


namespace payment_to_B_l350_350776

theorem payment_to_B 
  (total_payment : ℕ) (work_ac_fraction : ℚ) (remaining_work_fraction : ℚ) (expected_payment : ℕ) :
  total_payment = 529 →
  work_ac_fraction = 19/23 →
  remaining_work_fraction = 4/23 →
  expected_payment = (remaining_work_fraction * total_payment).nat_abs →
  expected_payment = 92 :=
by
  intros h_total_payment h_work_ac_fraction h_remaining_work_fraction h_expected_payment
  sorry

end payment_to_B_l350_350776


namespace fencing_length_l350_350001

-- Definition of the conditions
def L : ℝ := 20
def A : ℝ := 80

-- Definition of the width of the rectangle
def W : ℝ := A / L

-- Definition of the total length of fencing required
def F : ℝ := L + 2 * W

-- Theorem: Prove that F = 28
theorem fencing_length : F = 28 := by
  -- Since we have defined everything, we can skip the proof.
  sorry

end fencing_length_l350_350001


namespace least_positive_integer_to_make_multiple_of_3_l350_350759

open Nat

theorem least_positive_integer_to_make_multiple_of_3:
  ∃ x : ℕ, 0 < x ∧ (412 + x) % 3 = 0 ∧ ∀ y : ℕ, 0 < y ∧ (412 + y) % 3 = 0 → x ≤ y :=
exists.intro 2 sorry

end least_positive_integer_to_make_multiple_of_3_l350_350759


namespace complex_division_l350_350149

variable {ℂ : Type}
variables (z1 z2 : ℂ)

theorem complex_division (h1 : abs z1 = abs (z1 - 2 * z2))
    (h2 : z1 * conj z2 = sqrt 3 * (1 - complex.i)) :
    z1 / z2 = 1 - complex.i :=
    sorry

end complex_division_l350_350149


namespace unclaimed_books_fraction_l350_350054

noncomputable def fraction_unclaimed (total_books : ℝ) : ℝ :=
  let al_share := (2 / 5) * total_books
  let bert_share := (9 / 50) * total_books
  let carl_share := (21 / 250) * total_books
  let dan_share := (189 / 2500) * total_books
  total_books - (al_share + bert_share + carl_share + dan_share)

theorem unclaimed_books_fraction (total_books : ℝ) : fraction_unclaimed total_books / total_books = 1701 / 2500 := 
begin
  sorry
end

end unclaimed_books_fraction_l350_350054


namespace problem_1_problem_2_l350_350848

section proof_problem

variables (a b c d : ℤ)
variables (op : ℤ → ℤ → ℤ)
variables (add : ℤ → ℤ → ℤ)

-- Define the given conditions
axiom op_idem : ∀ (a : ℤ), op a a = a
axiom op_zero : ∀ (a : ℤ), op a 0 = 2 * a
axiom op_add : ∀ (a b c d : ℤ), add (op a b) (op c d) = op (a + c) (b + d)

-- Define the problems to prove
theorem problem_1 : add (op 2 3) (op 0 3) = -2 := sorry
theorem problem_2 : op 1024 48 = 2000 := sorry

end proof_problem

end problem_1_problem_2_l350_350848


namespace sqrt_simplify_l350_350363

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350363


namespace simplify_sqrt_450_l350_350536

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350536


namespace simplify_sqrt_450_l350_350383

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350383


namespace problem_1_problem_2_l350_350924

noncomputable def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

theorem problem_1 (a : ℝ) (h_a1 : 0 < a) (h_a2 : a < 1) :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 a) → Deriv f a x < 0) ∧ (∀ x : ℝ, x ∈ (Set.Ioi 1) → Deriv f a x < 0) :=
by
  -- the proof
  sorry

theorem problem_2 (a b : ℝ) (h1 : 0 < a) 
  (h2 : ∀ x : ℝ, x > 0 → f a x ≥ - (1/2) * x^2 + a * x + b) :
  a * b ≤ Real.exp 1 / 2 :=
by
  -- the proof
  sorry

end problem_1_problem_2_l350_350924


namespace range_a_when_n_2_range_a_when_f_defined_l350_350886

noncomputable def f (a : ℝ) (n x : ℝ) : ℝ := log ((1 + (n-1) * x^(n-1) * a) / n)

theorem range_a_when_n_2 (a : ℝ) :
  (∃ x : ℝ, f a 2 x > log (x * 2 ^ (x - 1))) ↔ (-1 < a) :=
sorry

theorem range_a_when_f_defined (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + (n-1)^x * a > 0) ↔ (-1/2 < a) :=
sorry

end range_a_when_n_2_range_a_when_f_defined_l350_350886


namespace simplify_sqrt_450_l350_350665

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350665


namespace sqrt_450_simplified_l350_350631

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350631


namespace sqrt_450_eq_15_sqrt_2_l350_350462

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350462


namespace particle_horizontal_distance_l350_350811

theorem particle_horizontal_distance :
  let parabola := λ x : ℝ, x^2 + 3 * x - 4 in
  let y_P := 3 in
  let y_Q := -4 in
  let x_P1 := -4 in
  let x_P2 := 1 in
  let x_Q1 := 0 in
  let x_Q2 := -3 in
  (parabola x_P1 = y_P ∧ parabola x_Q2 = y_Q ∨ parabola x_P2 = y_P ∧ parabola x_Q1 = y_Q) →
  (abs (x_P1 - x_Q2) = 1 ∨ abs (x_P2 - x_Q1) = 1) :=
by
  intros parabola y_P y_Q x_P1 x_P2 x_Q1 x_Q2 h
  sorry

end particle_horizontal_distance_l350_350811


namespace sum_of_row_12_in_pascals_triangle_l350_350961

theorem sum_of_row_12_in_pascals_triangle : 
  (List.range 13).sum (λ k, Nat.choose 12 k) = 4096 :=
by
  sorry

end sum_of_row_12_in_pascals_triangle_l350_350961


namespace exists_satisfactory_coloring_l350_350839

def within_bounds (r c : ℕ) : Prop := 
  1 ≤ r ∧ r ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9

def center (r c : ℕ) : Prop := 
  r = 5 ∧ c = 5

def adjacent (r1 c1 r2 c2 : ℕ) : Prop := 
  (abs (r1 - r2) ≤ 1) ∧ (abs (c1 - c2) ≤ 1)

def visible (r c : ℕ) : Prop :=
  ∃ (m n : ℕ), within_bounds m n ∧ ¬ adj r c m n

theorem exists_satisfactory_coloring : 
  ∃ (painted_cells : Finset (ℕ × ℕ)), 
  (∀ (r c : ℕ), center r c → (r, c) ∉ painted_cells)
  ∧ (∀ (r1 c1 r2 c2 : ℕ), 
        (r1, c1) ∈ painted_cells → 
        (r2, c2) ∈ painted_cells → 
        ¬ adjacent r1 c1 r2 c2) 
  ∧ (∀ (ray : List (ℕ × ℕ)), 
        (∀ (cell : ℕ × ℕ), 
          cell ∈ ray → visible cell → (cell ∉ painted_cells))
        ∨ (∃ (cell : ℕ × ℕ), 
          cell ∈ ray ∧ cell ∈ painted_cells)) :=
by sorry

end exists_satisfactory_coloring_l350_350839


namespace lilyPadsFullCoverage_l350_350967

def lilyPadDoubling (t: ℕ) : ℕ :=
  t + 1

theorem lilyPadsFullCoverage (t: ℕ) (h: t = 47) : lilyPadDoubling t = 48 :=
by
  rw [h]
  unfold lilyPadDoubling
  rfl

end lilyPadsFullCoverage_l350_350967


namespace condition_of_inequality_l350_350123

theorem condition_of_inequality {a b c : ℝ} :
  (a > b → ac^2 > bc^2) ↔ (a > b ∧ (c = 0 ∨ c^2 > 0)) := 
begin
  sorry
end

end condition_of_inequality_l350_350123


namespace vertex_y_coord_of_h_l350_350090

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 1
def h (x : ℝ) : ℝ := f x - g x

theorem vertex_y_coord_of_h : h (-1 / 10) = 79 / 20 := by
  sorry

end vertex_y_coord_of_h_l350_350090


namespace sqrt_450_eq_15_sqrt_2_l350_350587

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350587


namespace sqrt_450_eq_15_sqrt_2_l350_350572

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350572


namespace velocity_zero_at_t_2_l350_350812

theorem velocity_zero_at_t_2 :
  (∃ t : ℝ, (t ≠ -1) ∧ (s = (1/3 : ℝ) * t^3 - (1/2 : ℝ) * t^2 - 2 * t + 1) ∧ (derivative (λ t, (1/3 : ℝ) * t^3 - (1/2 : ℝ) * t^2 - 2 * t + 1) t = 0)) :=
by
  sorry

end velocity_zero_at_t_2_l350_350812


namespace range_expression_l350_350131

theorem range_expression (a b : ℝ) (h1 : ∀ x ∈ set.Icc (-1 : ℝ) 1, 0 ≤ a * x - b ∧ a * x - b ≤ 1) :
  set.image (λ x, (3 * a + b + 1) / (a + 2 * b - 2)) set.univ ∈ set.Icc (-4/5) (2/7) :=
sorry

end range_expression_l350_350131


namespace sqrt_simplify_l350_350368

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350368


namespace sqrt_450_eq_15_sqrt_2_l350_350547

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350547


namespace digits_1498_to_1500_l350_350083

-- Definitions from conditions
def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def sequence_ends_in_3 : List ℕ := List.filter ends_in_3 (List.range 10000)

def ith_digit_in_sequence (i : ℕ) : ℕ := 
  let seq_str := sequence_ends_in_3.map (λ n => n.toString)
  let flat_seq := String.singleton ' ' ++ String.intercalate " " seq_str
  if flat_seq.length > i then flat_seq.get ⟨i, sorry⟩.toNat else 0

-- Statement to prove
theorem digits_1498_to_1500 : (ith_digit_in_sequence 1498) * 100 + 
                              (ith_digit_in_sequence 1499) * 10 + 
                              (ith_digit_in_sequence 1500) = 23 :=
by sorry

end digits_1498_to_1500_l350_350083


namespace sqrt_450_eq_15_sqrt_2_l350_350562

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350562


namespace simplify_sqrt_450_l350_350644
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350644


namespace simplify_sqrt_450_l350_350671

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350671


namespace min_value_x_plus_one_over_2x_l350_350128

theorem min_value_x_plus_one_over_2x (x : ℝ) (hx : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 := sorry

end min_value_x_plus_one_over_2x_l350_350128


namespace sqrt_450_simplified_l350_350428

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350428


namespace sqrt_450_eq_15_sqrt_2_l350_350464

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350464


namespace sqrt_450_simplified_l350_350431

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350431


namespace sqrt_450_eq_15_sqrt_2_l350_350556

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350556


namespace simplify_sqrt_450_l350_350489

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350489


namespace intersection_A_B_l350_350165

def A : Set ℝ := {1, 3, 9, 27}
def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = Real.log x / Real.log 3}
theorem intersection_A_B : A ∩ B = {1, 3} := 
by
  sorry

end intersection_A_B_l350_350165


namespace problem_m_value_l350_350731

theorem problem_m_value (m : ℝ) : 
  (3 ∈ ({1, m, m^2 - 3*m - 1} : set ℝ)) ∧ 
  (-1 ∉ ({1, m, m^2 - 3*m - 1} : set ℝ)) → 
  m = 4 :=
by sorry

end problem_m_value_l350_350731


namespace simplify_sqrt_450_l350_350373

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350373


namespace simplify_sqrt_450_l350_350494

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350494


namespace simplify_sqrt_450_l350_350660

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350660


namespace inverse_on_interval_l350_350099

def has_inverse (f : ℝ → ℝ) (I : Set ℝ) : Prop := 
  ∃ g : ℝ → ℝ, ∀ (x : ℝ), x ∈ I → (f ∘ g) x = x ∧ (g ∘ f) (g x) = g x

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x^2 - 4*x + 3) / Real.log a

theorem inverse_on_interval (a m : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : 3 < m) :
  has_inverse (λ x, f x a) (Set.Ici m) :=
sorry

end inverse_on_interval_l350_350099


namespace basic_statement_for_odd_number_l350_350316

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem basic_statement_for_odd_number (n : ℕ) (h : is_odd n) : 
  string := "INPUT \"Enter an odd number n\"; n."

end basic_statement_for_odd_number_l350_350316


namespace distance_between_points_l350_350863

def point := (ℝ × ℝ × ℝ)

def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2 + (p2.3 - p1.3) ^ 2)

theorem distance_between_points :
  dist (3, 4, 6) (7, 0, 2) = 4 * real.sqrt 3 :=
by
  sorry

end distance_between_points_l350_350863


namespace simplify_sqrt_450_l350_350648
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350648


namespace shortest_crawling_distance_l350_350960

open Real

-- Given: Regular tetrahedron ABCD with edges of length 2
-- Midpoints A1, B1, C1 of edges DA, DB, DC respectively
-- Points P1, ..., P4 from arc A1B1 divided into 5 equal parts; Q1, ..., Q4 from arc B1C1 divided into 5 equal parts

-- Prove: The shortest crawling distance from P1 to Q4 along the surface of the tetrahedron is 2 * sin (42 degrees).
theorem shortest_crawling_distance :
  ∀ {A B C D A1 B1 C1 P1 Q4 : Point}
    (H_reg_tetra : regular_tetrahedron A B C D 2)
    (H_midpoints : midpoint D A A1 ∧ midpoint D B B1 ∧ midpoint D C C1)
    (H_arcs : ∃ (arc1 arc2 : arc), arc1 = arc.ofEndpointsRadius A1 B1 1 ∧ arc2 = arc.ofEndpointsRadius B1 C1 1)
    (H_divisions : ∃ (P P_parts Q Q_parts : list Point),
      list.length P_parts = 5 ∧ list.length Q_parts = 5 ∧
      P_parts.head = P1 ∧ Q_parts.last = Q4 ∧
      by unfold_division P_parts H_arcs.arc1 ∧ unfold_division Q_parts H_arcs.arc2),
  distance P1 Q4 = 2 * sin (42 * (π / 180)) :=
sorry

end shortest_crawling_distance_l350_350960


namespace smallest_common_multiple_l350_350766

theorem smallest_common_multiple : Nat.lcm 18 35 = 630 := by
  sorry

end smallest_common_multiple_l350_350766


namespace neon_sign_blink_interval_l350_350748

theorem neon_sign_blink_interval :
  ∃ (b : ℕ), (∀ t : ℕ, t > 0 → (t % 9 = 0 ∧ t % b = 0 ↔ t % 45 = 0)) → b = 15 :=
by
  sorry

end neon_sign_blink_interval_l350_350748


namespace sqrt_450_eq_15_sqrt_2_l350_350683

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350683


namespace archie_sod_needed_l350_350063

theorem archie_sod_needed 
  (backyard_length : ℝ) (backyard_width : ℝ) (shed_length : ℝ) (shed_width : ℝ)
  (backyard_area : backyard_length = 20 ∧ backyard_width = 13)
  (shed_area : shed_length = 3 ∧ shed_width = 5)
  : backyard_length * backyard_width - shed_length * shed_width = 245 := 
by
  unfold backyard_length backyard_width shed_length shed_width
  sorry

end archie_sod_needed_l350_350063


namespace total_difference_in_cards_l350_350080

theorem total_difference_in_cards (cards_chris : ℕ) (cards_charlie : ℕ) (cards_diana : ℕ) (cards_ethan : ℕ)
  (h_chris : cards_chris = 18)
  (h_charlie : cards_charlie = 32)
  (h_diana : cards_diana = 25)
  (h_ethan : cards_ethan = 40) :
  (cards_charlie - cards_chris) + (cards_diana - cards_chris) + (cards_ethan - cards_chris) = 43 := by
  sorry

end total_difference_in_cards_l350_350080


namespace cos_B_value_l350_350179

theorem cos_B_value (B : ℝ) (h1 : Real.tan B + Real.sec B = 3) : Real.cos B = 3 / 5 :=
sorry

end cos_B_value_l350_350179


namespace complex_division_l350_350904

noncomputable def complex_value : ℂ := (2 * complex.I) / (1 + complex.I)

theorem complex_division :
  complex_value = 1 + complex.I :=
by
  sorry

end complex_division_l350_350904


namespace simplify_sqrt_450_l350_350481

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350481


namespace scoring_situations_4_students_l350_350965

noncomputable def number_of_scoring_situations (students : ℕ) (topicA_score : ℤ) (topicB_score : ℤ) : ℕ :=
  let combinations := Nat.choose 4 2
  let first_category := combinations * 2 * 2
  let second_category := 2 * combinations
  first_category + second_category

theorem scoring_situations_4_students : number_of_scoring_situations 4 100 90 = 36 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end scoring_situations_4_students_l350_350965


namespace sqrt_450_equals_15_sqrt_2_l350_350603

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350603


namespace simplify_sqrt_450_l350_350406

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350406


namespace number_of_solutions_l350_350895

noncomputable def generating_function_coefficient (p : List ℕ) (r : ℕ) : ℕ :=
    -- Function to extract the coefficient of t^r
    sorry

theorem number_of_solutions (p : List ℕ) (r : ℕ) (n : ℕ) 
    (h_p : ∀ i, i < n → 0 < p.get i) 
    (h_r : 0 < r):
    ∃ a_r : ℕ, a_r = generating_function_coefficient p r := sorry

end number_of_solutions_l350_350895


namespace find_constants_l350_350169

variables {V : Type*} [inner_product_space ℝ V]
variables (a b p : V)

theorem find_constants (h : ∥p - b∥ = 3 * ∥p - a∥) : 
  ∃ s v : ℝ, s = 9 / 8 ∧ v = -1 / 8 ∧ ∃ d : ℝ, ∥p - (s • a + v • b)∥ = d :=
sorry

end find_constants_l350_350169


namespace sqrt_450_simplified_l350_350612

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350612


namespace sqrt_simplify_l350_350352

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350352


namespace sqrt_450_equals_15_sqrt_2_l350_350607

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350607


namespace min_payment_proof_max_payment_proof_expected_payment_proof_l350_350326

noncomputable def items : List ℕ := List.range 1 11 |>.map (λ n => n * 100)

def min_amount_paid : ℕ :=
  (1000 + 900 + 700 + 600 + 400 + 300 + 100)

def max_amount_paid : ℕ :=
  (1000 + 900 + 800 + 700 + 600 + 500 + 400)

def expected_amount_paid : ℚ :=
  4583 + 33 / 100

theorem min_payment_proof :
  (∑ x in (List.range 15).filter (λ x => x % 3 ≠ 0), (items.get! x : ℕ)) = min_amount_paid := by
  sorry

theorem max_payment_proof :
  (∑ x in List.range 10, if x % 3 = 0 then 0 else (items.get! x : ℕ)) = max_amount_paid := by
  sorry

theorem expected_payment_proof :
  ∑ k in items, ((k : ℚ) * (∏ m in List.range 9, (10 - m) * (9 - m) / 72)) = expected_amount_paid := by
  sorry

end min_payment_proof_max_payment_proof_expected_payment_proof_l350_350326


namespace simplify_sqrt_450_l350_350402

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350402


namespace sum_n_satisfying_abs_diff_l350_350278

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, ...]  -- continuing list of primes
def composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, ...]  -- continuing list of composites

def abs_diff_less_than_3 (p c : ℕ) : Prop := (p - c).natAbs < 3

-- Final statement to prove
theorem sum_n_satisfying_abs_diff :
  ∑ n in [1, 4, 5, 6], n = 16 := by
  sorry

end sum_n_satisfying_abs_diff_l350_350278


namespace equal_distance_between_closest_l350_350103

-- Define the scenario of people standing on a ring road
def people_on_ring (n : ℕ) := fin n → ℕ

-- Define the inspectors and officers placement
def is_inspector (placement : people_on_ring 100) (i : fin 100) := placement i < 50
def is_officer (placement : people_on_ring 100) (i : fin 100) := placement i >= 50

-- Lean statement equivalent to the problem
theorem equal_distance_between_closest (placement : people_on_ring 100)
  (h_inspectors : ∀ x, is_inspector placement x → ∃ y, is_inspector placement y ∧ x ≠ y)
  (h_officers : ∀ x, is_officer placement x → ∃ y, is_officer placement y ∧ x ≠ y) :
  (min_dist : fin 100 → fin 100 → ℕ) (x y : fin 100) (is_min_dist : x ≠ y → is_inspector placement x → is_inspector placement y) = 
  (min_dist : fin 100 → fin 100 → ℕ) (u v : fin 100) (is_min_dist : u ≠ v → is_officer placement u → is_officer placement v) :=
by
  sorry

end equal_distance_between_closest_l350_350103


namespace find_range_of_m_l350_350127

def p (m : ℝ) : Prop :=
  ∀ x ∈ Icc 1 2, x^2 + x - m > 0

def q (m : ℝ) : Prop :=
  (m^2 - 1 > m + 2 - 1) ∧ (m^2 > m + 2)

theorem find_range_of_m (m : ℝ) (hp : p m ∨ q m) (hnpq : ¬ (p m ∧ q m)) : 
  m ∈ Set.Ici (-1) \ Set.Icc (-1) 2 ∪ Set.Ioi 2 :=
sorry

end find_range_of_m_l350_350127


namespace num_valid_five_digit_numbers_l350_350174

/-- Five-digit number formation conditions -/
def valid_five_digit_numbers : Finset (Fin 90000) :=
  {n | let n_str := n.val.toDigits 10 in 
    (n_str.length = 5) ∧ 
    (∀ i < 3, n_str[i] ∈ {0, 1, 2, 3}) ∧ 
    (∀ i ≥ 3, n_str[i] ∈ {5, 6, 7, 8}) ∧ 
    ¬(n_str[0] = 0 ∧ n_str[1] ∈ {0, 1, 2, 3}) ∧ 
    ¬(n_str[2] = 0 ∧ n_str[3] ∈ {5}) ∧
    ¬(n_str[3] = 5 ∧ n_str[4] ∈ {5, 6, 7, 8})}

/-- The number of valid five-digit numbers under given conditions is 198. -/
theorem num_valid_five_digit_numbers : ∑ n in valid_five_digit_numbers, 1 = 198 := 
by 
  sorry

end num_valid_five_digit_numbers_l350_350174


namespace extremal_values_d_l350_350890

theorem extremal_values_d (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ → Prop)
  (hC : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1)
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : ∃ (x y : ℝ), C (x, y)) :
  ∃ (max_d min_d : ℝ), max_d = 14 ∧ min_d = 10 :=
by
  -- Necessary assumptions
  have h₁ : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1 := hC
  have h₂ : A = (-1, 0) := hA
  have h₃ : B = (1, 0) := hB
  have h₄ : ∃ (x y : ℝ), C (x, y) := hP
  sorry

end extremal_values_d_l350_350890


namespace rotation_transforms_DEF_to_D_l350_350747

def point := (ℝ × ℝ)
def triangle := (point × point × point)

def D : point := (0, 0)
def E : point := (0, 10)
def F : point := (14, 0)

def D' : point := (22, 16)
def E' : point := (30, 16)
def F' : point := (22, 4)

def triangle_DEF : triangle := (D, E, F)
def triangle_D'E'F' : triangle := (D', E', F')

noncomputable def rotation_center : point := (19, -3)
noncomputable def rotation_angle : ℝ := 90

theorem rotation_transforms_DEF_to_D'E'F' :
  ∃ (u v : ℝ), (u, v) = rotation_center ∧ (90 < 180) ∧ u + v + rotation_angle = 106 := by
  use 19
  use -3
  rw [eq_comm, add_comm]
  exact And.intro rfl (And.intro (lt_of_lt_of_le (by norm_num) (le_refl _)) (by norm_num))
  sorry

end rotation_transforms_DEF_to_D_l350_350747


namespace sqrt_simplify_l350_350353

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350353


namespace total_shaded_area_is_correct_l350_350792

-- Define the dimensions of the floor
def floor_length : ℝ := 12
def floor_width : ℝ := 15

-- Define the dimensions and characteristics of the tile
def tile_side : ℝ := 2
def quarter_circle_radius : ℝ := 1

-- Total area of the floor
def total_floor_area : ℝ := floor_length * floor_width

-- Total number of tiles
def number_of_tiles : ℝ := (floor_length / tile_side) * (floor_width / tile_side)

-- Area of one tile
def area_of_one_tile : ℝ := tile_side * tile_side

-- Area of one quarter circle
def area_of_one_quarter_circle : ℝ := (Mathlib.pi * (quarter_circle_radius ^ 2)) / 4

-- Area of the white pattern in one tile (four quarter circles make one full circle)
def area_of_white_pattern : ℝ := Mathlib.pi * (quarter_circle_radius ^ 2)

-- Shaded area in one tile
def area_of_shaded_tile : ℝ := area_of_one_tile - area_of_white_pattern

-- Total shaded area in the whole floor
def total_shaded_area : ℝ := number_of_tiles * area_of_shaded_tile

-- The mathematical statement to prove: the total shaded area is 180 - 45π square feet
theorem total_shaded_area_is_correct :
  total_shaded_area = 180 - 45 * Mathlib.pi :=
  sorry

end total_shaded_area_is_correct_l350_350792


namespace ellipse_condition_necessary_not_sufficient_l350_350015

theorem ellipse_condition_necessary_not_sufficient {a b : ℝ} (h : a * b > 0):
  (∀ x y : ℝ, a * x^2 + b * y^2 = 1 → a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0) ∧ 
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) → a * b > 0) :=
sorry

end ellipse_condition_necessary_not_sufficient_l350_350015


namespace simplify_sqrt_450_l350_350510

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350510


namespace simplify_sqrt_450_l350_350635
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350635


namespace number_of_ordered_11_tuples_eq_660_l350_350867

theorem number_of_ordered_11_tuples_eq_660 :
  let exists_seq := ∃ (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} : ℤ),
                    (∀ i, 1 ≤ i ∧ i ≤ 11 →
                         (a_1 + a_2 + ... + a_{i-1} + a_{i+1} + ... + a_{11}) =
                          a_i^2)
  in count exists_seq 660 :=
sorry

end number_of_ordered_11_tuples_eq_660_l350_350867


namespace sqrt_450_eq_15_sqrt_2_l350_350560

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350560


namespace b_2018_eq_5043_l350_350137

def b (n : Nat) : Nat :=
  if n % 2 = 1 then 5 * ((n + 1) / 2) - 3 else 5 * (n / 2) - 2

theorem b_2018_eq_5043 : b 2018 = 5043 := by
  sorry

end b_2018_eq_5043_l350_350137


namespace sqrt_450_simplified_l350_350632

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350632


namespace area_of_ellipse_l350_350110

def ellipse_area (a b : ℝ) : ℝ := π * a * b

def standard_ellipse_eq (x y h w : ℝ) : Prop := 
  (x + h)^2 / 3 + (w * (y + h))^2 / 4 = 1

theorem area_of_ellipse (x y : ℝ) (h : ℝ) (w : ℝ) :
    (4 * x^2 + 16 * x + 9 * y^2 + 36 * y + 64 = 0) → h = -2 → w = 3
  → standard_ellipse_eq x y h w 
  → ellipse_area √3 (2 / √3) = 2 * π :=
by
  sorry

end area_of_ellipse_l350_350110


namespace sqrt_450_eq_15_sqrt_2_l350_350477

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350477


namespace hyperbola_standard_equation_l350_350115

theorem hyperbola_standard_equation (a b : ℝ) :
  (∃ (P Q : ℝ × ℝ), P = (-3, 2 * Real.sqrt 7) ∧ Q = (-6 * Real.sqrt 2, -7) ∧
    (∀ x y b, y^2 / b^2 - x^2 / a^2 = 1 ∧ (2 * Real.sqrt 7)^2 / b^2 - (-3)^2 / a^2 = 1
    ∧ (-7)^2 / b^2 - (-6 * Real.sqrt 2)^2 / a^2 = 1)) →
  b^2 = 25 → a^2 = 75 →
  (∀ x y, y^2 / (25:ℝ) - x^2 / (75:ℝ) = 1) :=
sorry

end hyperbola_standard_equation_l350_350115


namespace number_of_cool_polynomials_is_even_l350_350283

open Polynomial

noncomputable def is_cool (n : ℕ) (p : ℕ) (Q : Polynomial ℕ) : Prop :=
  (Q.degree ≤ 2023) ∧ (∀ i : ℕ, i ≤ Q.degree → Q.coeff i ≤ n) ∧
  p ∣ (List.prod (List.map (λ x, Q.eval x) (List.range (p - 3).succ.tail.tail)) - 1)

theorem number_of_cool_polynomials_is_even (n : ℕ) (p : ℕ) (hp : ∃ k, p = 8 * k + 5) :
  let cool_polynomials := {Q : Polynomial ℕ | is_cool n p Q} in
  Fintype.card cool_polynomials % 2 = 0 :=
by sorry

end number_of_cool_polynomials_is_even_l350_350283


namespace min_value_l350_350118

theorem min_value (x : ℝ) (h : x > 2) : ∃ y, y = 22 ∧ 
  ∀ z, (z > 2) → (y ≤ (z^2 + 8) / (Real.sqrt (z - 2))) := 
sorry

end min_value_l350_350118


namespace min_sum_distances_l350_350133

noncomputable def point_P (x y : ℝ) : Prop := y^2 = -4 * x

def distance_directrix (P : ℝ × ℝ) : ℝ := 
  abs (fst P + 1)

def distance_line (P : ℝ × ℝ) (a b c : ℝ) :
  ℝ := abs (a * (fst P) + b * (snd P) + c) / sqrt (a^2 + b^2)

theorem min_sum_distances :
  ∃ (P : ℝ × ℝ), point_P (fst P) (snd P) →
  let d1 := distance_directrix P,
  let d2 := distance_line P 1 1 (-4)
  in d1 + d2 = (5 * sqrt 2) / 2 :=
sorry

end min_sum_distances_l350_350133


namespace sin_omega_x_increasing_and_maximum_ω_l350_350923

noncomputable def function_range (ω : ℝ) : Prop :=
  (∀ x ∈ Icc (- (2/3) * real.pi / ω) (5/6 * real.pi / ω), 
    (ω * real.cos (ω * x)) > 0)
  ∧ (1/2 ≤ ω ∧ ω ≤ 3/5)

theorem sin_omega_x_increasing_and_maximum_ω (ω : ℝ) (h : ω > 0) :
  function_range ω ↔ (1/2 ≤ ω ∧ ω ≤ 3/5) := 
sorry

end sin_omega_x_increasing_and_maximum_ω_l350_350923


namespace cos_of_tan_sec_l350_350181

theorem cos_of_tan_sec {B : ℝ} (h : tan B + sec B = 3) : cos B = 3 / 5 :=
sorry

end cos_of_tan_sec_l350_350181


namespace train_average_speed_l350_350817

theorem train_average_speed :
  let start_time := 9.0 -- Start time in hours (9:00 am)
  let end_time := 13.75 -- End time in hours (1:45 pm)
  let total_distance := 348.0 -- Total distance in km
  let halt_time := 0.75 -- Halt time in hours (45 minutes)
  let scheduled_time := end_time - start_time -- Total scheduled time in hours
  let actual_travel_time := scheduled_time - halt_time -- Actual travel time in hours
  let average_speed := total_distance / actual_travel_time -- Average speed formula
  average_speed = 87.0 := sorry

end train_average_speed_l350_350817


namespace gain_percent_of_50C_eq_25S_l350_350190

variable {C S : ℝ}

theorem gain_percent_of_50C_eq_25S (h : 50 * C = 25 * S) : 
  ((S - C) / C) * 100 = 100 :=
by
  sorry

end gain_percent_of_50C_eq_25S_l350_350190


namespace sqrt_450_equals_15_sqrt_2_l350_350604

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350604


namespace part_I_part_II_part_III_l350_350126

def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def f (x : ℝ) := if x > 0 then x^2 + 2 * x else if x < 0 then -x^2 + 2 * x else 0

theorem part_I (hf : is_odd_function f) : f 0 = 0 := by
  sorry

theorem part_II (hf : is_odd_function f) :
  ∀ x, f x = if x > 0 then x^2 + 2 * x else if x < 0 then -x^2 + 2 * x else 0 := by
  sorry

theorem part_III (hf : ∀ t : ℝ, f (t + 1) + f (m - 2 * t ^ 2) < 0) : m < -3/2 := by
  sorry

end part_I_part_II_part_III_l350_350126


namespace solve_mod_equation_l350_350707

theorem solve_mod_equation :
  ∃ a m : ℕ, (∀ y : ℤ, 10 * y + 3 ≡ 7 [MOD 18] → y ≡ a [MOD m]) ∧ (m ≥ 2) ∧ (a < m) ∧ (a + m = 13) :=
by
  sorry

end solve_mod_equation_l350_350707


namespace sqrt_450_eq_15_sqrt_2_l350_350577

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350577


namespace tank_capacity_l350_350780

theorem tank_capacity (C : ℝ) (h₁ : 3/4 * C + 7 = 7/8 * C) : C = 56 :=
by
  sorry

end tank_capacity_l350_350780


namespace simplify_sqrt_450_l350_350486

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350486


namespace smallest_integer_representation_l350_350764

theorem smallest_integer_representation :
  ∃ (A B C : ℕ), 0 ≤ A ∧ A < 5 ∧ 0 ≤ B ∧ B < 7 ∧ 0 ≤ C ∧ C < 4 ∧ 6 * A = 8 * B ∧ 6 * A = 5 * C ∧ 8 * B = 5 * C ∧ (6 * A) = 24 :=
  sorry

end smallest_integer_representation_l350_350764


namespace sqrt_simplify_l350_350364

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350364


namespace quadratic_one_solution_l350_350102

theorem quadratic_one_solution (p : ℝ) : (3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0) 
  → ((-6) ^ 2 - 4 * 3 * p = 0) 
  → p = 3 :=
by
  intro h1 h2
  have h1' : 3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0 := h1
  have h2' : (-6) ^ 2 - 4 * 3 * p = 0 := h2
  sorry

end quadratic_one_solution_l350_350102


namespace ratio_S1_S2_l350_350972

-- The conditions for the triangle and the circle
variables (ABC : Type) [acute_triangle ABC] (A B C D E : ABC)
variables (S1 S2 : ℝ)

-- Given conditions: angle at A, circle with BC as diameter intersecting AB at D and AC at E
axiom angle_A_30 : angle A = 30
axiom BC_diameter : diameter (circle BC) D E
-- Areas of the triangle ADE and quadrilateral BDEC
axiom areas_division : area ADE = S1 ∧ area BDEC = S2

-- The theorem we want to prove
theorem ratio_S1_S2 (ABC : acute_triangle) (A : point ABC) (B : point ABC) (C : point ABC) 
(D : point ABC) (E : point ABC) (S1 S2 : ℝ) :
  angle A = 30 →
  diameter (circle BC) D E →
  (area ADE = S1 ∧ area BDEC = S2) →
  S1 / S2 = 3 :=
by
  sorry

end ratio_S1_S2_l350_350972


namespace simplify_sqrt_450_l350_350519

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350519


namespace simplify_sqrt_450_l350_350540

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350540


namespace min_payment_max_payment_expected_payment_l350_350321

-- Given Prices
def item_prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

-- Function to compute the actual paid amount given groups of three items
def paid_amount (groups : List (List ℕ)) : ℕ :=
  groups.foldr (λ group sum => sum + group.foldr (λ x s => s + x) 0 - group.minimum') 0

-- Optimal arrangement of items for minimal payment
def optimal_groups : List (List ℕ) :=
  [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]]

-- Suboptimal arrangement of items for maximal payment
def suboptimal_groups : List (List ℕ) :=
  [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]]

-- Expected value calculation's configuration
def num_items := 10
def num_groups := (num_items / 3).natCeil

noncomputable def expected_amount : ℕ :=
  let total_sum := item_prices.foldr (λ x s => s + x) 0
  let expected_savings := 100 * (660 / 72)
  total_sum - expected_savings

theorem min_payment : paid_amount optimal_groups = 4000 := by
  -- Proof steps and details here
  sorry

theorem max_payment : paid_amount suboptimal_groups = 4900 := by
  -- Proof steps and details here
  sorry

theorem expected_payment : expected_amount ≈ 4583 := by
  -- Proof steps and details here
  sorry

end min_payment_max_payment_expected_payment_l350_350321


namespace sqrt_450_eq_15_sqrt_2_l350_350571

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350571


namespace FC_value_l350_350882

theorem FC_value (DC CB AD FC : ℝ) (h1 : DC = 10) (h2 : CB = 9)
  (h3 : AB = (1 / 3) * AD) (h4 : ED = (3 / 4) * AD) : FC = 13.875 := by
  sorry

end FC_value_l350_350882


namespace sqrt_450_simplified_l350_350425

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350425


namespace BCD4_base10_value_l350_350091

/-- Definition of digit values in hexadecimal -/
def hex_digit_values (d : Char) : ℕ :=
  match d with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | _ => 0 -- Assuming all valid inputs are hexadecimal digits

/-- Convert a hexadecimal character list to a decimal integer -/
noncomputable def hex_to_decimal (hex : List Char) : ℕ :=
  hex.reverse.enum.map (λ ⟨i, d⟩ => (hex_digit_values d) * 16^i).sum

/-- Given condition -/
def bcd4_hex : List Char := ['B', 'C', 'D', '4']

/-- Proof statement -/
theorem BCD4_base10_value : hex_to_decimal bcd4_hex = 31444 :=
by
  -- Note: Insert the proof here
  sorry

end BCD4_base10_value_l350_350091


namespace simplify_sqrt_450_l350_350534

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350534


namespace brad_trips_to_fill_barrel_l350_350832

noncomputable def bucket_volume (r : ℝ) : ℝ := (2 / 3) * real.pi * r^3

noncomputable def barrel_volume (r : ℝ) (h : ℝ) : ℝ := real.pi * r^2 * h

theorem brad_trips_to_fill_barrel :
  ∀ (r : ℝ) (h : ℝ), r = 10 ∧ h = 15 →
  let trips := (barrel_volume r h) / (bucket_volume r)
  trips = 3 :=
by
  intros r h conditions
  simp at conditions
  obtain ⟨r_eq, h_eq⟩ := conditions
  rw [r_eq, h_eq]
  let calc_trips : ℝ := (3 / 2 * real.pi * 10^3) / (2 / 3 * real.pi * 10^3)
  have : calc_trips = 9 / 4 :=
    by norm_num
  have : 9 / 4 = 3 :=
    by norm_num
  rw [← this]
  exact sorry

end brad_trips_to_fill_barrel_l350_350832


namespace part_a_part_b_l350_350087

-- Define the required trigonometric theorems and heptagon conditions
noncomputable def proof_problem1 : Prop :=
  1 / (Real.sin (Real.pi / 7)) = 1 / (Real.sin (2 * Real.pi / 7)) + 1 / (Real.sin (3 * Real.pi / 7))

noncomputable def proof_problem2 : Prop :=
  let AG := 1 / Real.sin (2 * Real.pi / 7) in
  let AF := 1 / Real.sin (Real.pi / 7) in
  let AE := 1 / Real.sin (3 * Real.pi / 7) in
  1 / AG = 1 / AF + 1 / AE

-- Statements for proving the problems
theorem part_a : proof_problem1 := by
  sorry

theorem part_b (h: proof_problem1) : proof_problem2 := by
  sorry

end part_a_part_b_l350_350087


namespace train_speed_proofProblem_l350_350796

theorem train_speed_proofProblem (
  length_train1 : ℝ,
  speed_train2 : ℝ,
  length_train2 : ℝ,
  time_cross : ℝ,
  V1 : ℝ
) : 
  length_train1 = 280 ∧ 
  speed_train2 = 80 ∧ 
  length_train2 = 220.04 ∧ 
  time_cross = 9 ∧ 
  V1 = 120.016 
  → (V1 = 120.016) :=
by 
  intro h
  cases h with h_length_train1 h
  cases h with h_speed_train2 h
  cases h with h_length_train2 h
  cases h with h_time_cross h_V1
  rw h_V1
  exact h_V1

end train_speed_proofProblem_l350_350796


namespace min_notebooks_needed_l350_350816

variable (cost_pen cost_notebook num_pens discount_threshold : ℕ)

theorem min_notebooks_needed (x : ℕ)
    (h1 : cost_pen = 10)
    (h2 : cost_notebook = 4)
    (h3 : num_pens = 3)
    (h4 : discount_threshold = 100)
    (h5 : num_pens * cost_pen + x * cost_notebook ≥ discount_threshold) :
    x ≥ 18 := 
sorry

end min_notebooks_needed_l350_350816


namespace sqrt_450_simplified_l350_350434

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350434


namespace simplify_sqrt_450_l350_350658

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350658


namespace average_salary_for_company_l350_350778

variable (n_m : ℕ) -- number of managers
variable (n_a : ℕ) -- number of associates
variable (avg_salary_m : ℕ) -- average salary of managers
variable (avg_salary_a : ℕ) -- average salary of associates

theorem average_salary_for_company (h_n_m : n_m = 15) (h_n_a : n_a = 75) 
  (h_avg_salary_m : avg_salary_m = 90000) (h_avg_salary_a : avg_salary_a = 30000) : 
  (n_m * avg_salary_m + n_a * avg_salary_a) / (n_m + n_a) = 40000 := 
by
  sorry

end average_salary_for_company_l350_350778


namespace simplify_sqrt_450_l350_350387

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350387


namespace prob_decreasing_range_of_m_l350_350187

theorem prob_decreasing_range_of_m (m : ℝ) : 
  (∀ x y, x < y ∧ y ≤ 1 → f m x ≥ f m y) ↔ (0 ≤ m ∧ m ≤ 1/3) :=
  sorry

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x + 1

end prob_decreasing_range_of_m_l350_350187


namespace triangle_ratio_l350_350244

theorem triangle_ratio (A B C P : Type) 
  {AB AC CP x : ℝ} (right_angle_C: ∠ACB = 90) 
  (angle_BAC_lt_45: ∠BAC < 45) (AB_eq_4: AB = 4) 
  (angle_APC_eq_2_angle_ACP: ∠APC = 2 * ∠ACP)
  (CP_eq_1: CP = 1): (∃ p q r : ℕ, ∃ ratio : ℝ, 
  ratio = (3 + 2 * Real.sqrt 2) ∧ p + q + r = 7) := 
sorry

end triangle_ratio_l350_350244


namespace train_travel_time_change_l350_350050

theorem train_travel_time_change 
  (t1 t2 : ℕ) (s1 s2 d : ℕ) 
  (h1 : t1 = 4) 
  (h2 : s1 = 50) 
  (h3 : s2 = 100) 
  (h4 : d = t1 * s1) :
  t2 = d / s2 → t2 = 2 :=
by
  intros
  sorry

end train_travel_time_change_l350_350050


namespace semicircle_exists_l350_350294

theorem semicircle_exists (n : ℕ) (h : n > 0) :
  ∃ sectors : list ℕ, (∀ i, i ∈ sectors → 1 ≤ i ∧ i ≤ n) ∧ sectors.length = n :=
sorry

end semicircle_exists_l350_350294


namespace unique_function_l350_350109

theorem unique_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, m > 0 → n > 0 → (f(m) + f(n) - m * n ≠ 0) ∧ (f(m) + f(n) - m * n) ∣ (m * f(m) + n * f(n))) : 
  ∀ n : ℕ, n > 0 → f(n) = n * n :=
sorry

end unique_function_l350_350109


namespace number_of_tens_in_sum_l350_350208

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l350_350208


namespace question1_question2_l350_350140

-- Define the circle and point
def circle : Set (ℝ × ℝ) := { p | p.1 ^ 2 + p.2 ^ 2 = 4 }
def M (a : ℝ) : Point := (1, a)

-- Theorem for Question 1
theorem question1 (a : ℝ) (h_tangent : ∃! l : Line, is_tangent l M a circle) :
  a = sqrt 3 ∨ a = -sqrt 3 :=
by { sorry }

-- Theorem for Question 2
theorem question2 (a := 2) (h : a = 2) :
  (is_tangent_line (Line (0, 2)) circle ∧ is_pass_through (1, 2) (Line (0,2))) ∧
  (is_tangent_line (Line (-4/3, 10/3)) circle ∧ is_pass_through (1, 2) (Line (-4/3, 10/3))) :=
by { sorry }

-- Definitions for tangent line and passing through
def is_tangent (l : Line) (p : Point) (circle : Set (ℝ × ℝ)) : Prop := -- definition here
sorry

def is_tangent_line (l : Line) (circle : Set (ℝ × ℝ)) : Prop := -- definition here
sorry

def is_pass_through (p : Point) (l : Line) : Prop := -- definition here
sorry

end question1_question2_l350_350140


namespace find_range_of_a_l350_350746

theorem find_range_of_a (a : ℝ) (x : ℝ) (y : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) 
    (hineq : x * y ≤ a * x^2 + 2 * y^2) : 
    -1 ≤ a := sorry

end find_range_of_a_l350_350746


namespace diet_equivalence_l350_350070

variable (B E L D A : ℕ)

theorem diet_equivalence :
  (17 * B = 170 * L) →
  (100000 * A = 50 * L) →
  (10 * B = 4 * E) →
  12 * E = 600000 * A :=
sorry

end diet_equivalence_l350_350070


namespace archie_needs_sod_l350_350061

-- Define the dimensions of the backyard
def backyard_length : ℕ := 20
def backyard_width : ℕ := 13

-- Define the dimensions of the shed
def shed_length : ℕ := 3
def shed_width : ℕ := 5

-- Statement: Prove that the area of the backyard minus the area of the shed equals 245 square yards
theorem archie_needs_sod : 
  backyard_length * backyard_width - shed_length * shed_width = 245 := 
by sorry

end archie_needs_sod_l350_350061


namespace cat_did_not_eat_sixth_fish_l350_350800

def fish_cells : ℕ → ℕ
| 1 := 3
| 2 := 4
| 3 := 16
| 4 := 12
| 5 := 20
| _ := 0  -- For fish 6 and any other fish, we use 0 cells as stated "not required"

def eaten_fish_sum : ℕ := (fish_cells 1) + (fish_cells 2) + (fish_cells 3) + (fish_cells 4) + (fish_cells 5)

theorem cat_did_not_eat_sixth_fish :
  ∀ (cell_rate : ℕ), (cell_rate = 3) →
  eaten_fish_sum % 3 = 1 →
  (eaten_fish_sum + fish_cells 6) % 3 ≠ 0 :=
by
  intros cell_rate h_rate h_sum_mod
  rw [←add_assoc, add_comm 55 _, add_assoc]
  exact_mod_cast sorry

end cat_did_not_eat_sixth_fish_l350_350800


namespace factorization_l350_350861

theorem factorization (x : ℝ) : 
  x^2 * (x - 3) - 4 * (x - 3) = (x - 3) * (x + 2) * (x - 2) :=
by {
  sorry
}

end factorization_l350_350861


namespace diff_of_cubes_divisible_by_9_l350_350096

theorem diff_of_cubes_divisible_by_9 (a b : ℤ) : 9 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) := 
sorry

end diff_of_cubes_divisible_by_9_l350_350096


namespace min_payment_max_payment_expected_payment_l350_350317

-- Given Prices
def item_prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

-- Function to compute the actual paid amount given groups of three items
def paid_amount (groups : List (List ℕ)) : ℕ :=
  groups.foldr (λ group sum => sum + group.foldr (λ x s => s + x) 0 - group.minimum') 0

-- Optimal arrangement of items for minimal payment
def optimal_groups : List (List ℕ) :=
  [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]]

-- Suboptimal arrangement of items for maximal payment
def suboptimal_groups : List (List ℕ) :=
  [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]]

-- Expected value calculation's configuration
def num_items := 10
def num_groups := (num_items / 3).natCeil

noncomputable def expected_amount : ℕ :=
  let total_sum := item_prices.foldr (λ x s => s + x) 0
  let expected_savings := 100 * (660 / 72)
  total_sum - expected_savings

theorem min_payment : paid_amount optimal_groups = 4000 := by
  -- Proof steps and details here
  sorry

theorem max_payment : paid_amount suboptimal_groups = 4900 := by
  -- Proof steps and details here
  sorry

theorem expected_payment : expected_amount ≈ 4583 := by
  -- Proof steps and details here
  sorry

end min_payment_max_payment_expected_payment_l350_350317


namespace simplify_sqrt_450_l350_350637
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350637


namespace sqrt_450_simplified_l350_350614

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350614


namespace divisible_by_sum_of_digits_l350_350773

theorem divisible_by_sum_of_digits (a : ℕ) (h1 : 100 ≤ a) (h2 : a + 17 ≤ 999) :
  ∃ b ∈ finset.range 18, (a + b) % (nat.digits 10 (a + b)).sum = 0 :=
by sorry

end divisible_by_sum_of_digits_l350_350773


namespace jessica_current_age_l350_350267

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_l350_350267


namespace heartsuit_ratio_l350_350953

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by
  sorry

end heartsuit_ratio_l350_350953


namespace max_area_swept_by_arc_l350_350969

-- Given a right triangle PQR with hypotenuse of 10 units
-- and a curved segment moving from P to R along an arc always perpendicular to QR
-- Prove that the maximum area swept by the arc is (25 * π / 2) square units

theorem max_area_swept_by_arc
  (P Q R : ℝ) -- Points forming right triangle PQR
  (angle_PQR : ∠PQR = real.pi / 2) -- Angle PQR is 90 degrees
  (hypotenuse_length : dist P R = 10) -- Hypotenuse length is 10 units
  (arc_moves_perpendicular : ∀ t, arc(t) ⊥ QR) -- The arc is always perpendicular to QR
  : (∃ A : ℝ, A = (25 * real.pi / 2)) :=
sorry

end max_area_swept_by_arc_l350_350969


namespace sqrt_450_simplified_l350_350415

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350415


namespace pearls_problem_l350_350027

theorem pearls_problem :
  ∃ n : ℕ, (n % 8 = 6) ∧ (n % 7 = 5) ∧ (n = 54) ∧ (n % 9 = 0) :=
by sorry

end pearls_problem_l350_350027


namespace rate_per_sq_meter_l350_350043

-- Conditions
def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def total_cost : ℝ := 680

-- Definitions derived from conditions
def total_area := plot_length * plot_width
def inner_length := plot_length - 2 * path_width
def inner_width := plot_width - 2 * path_width
def grassy_area := inner_length * inner_width
def path_area := total_area - grassy_area

-- Theorem to prove the rate per sq. meter
theorem rate_per_sq_meter : (total_cost / path_area) = 0.8 :=
by
  have h_total_area : total_area = 7150 := by
    unfold total_area
    norm_num

  have h_inner_length : inner_length = 105 := by
    unfold inner_length
    norm_num

  have h_inner_width : inner_width = 60 := by
    unfold inner_width
    norm_num

  have h_grassy_area : grassy_area = 6300 := by
    unfold grassy_area
    rw [h_inner_length, h_inner_width]
    norm_num

  have h_path_area : path_area = 850 := by
    unfold path_area
    rw [h_total_area, h_grassy_area]
    norm_num

  rw [h_path_area]
  unfold total_cost
  norm_num
  sorry

end rate_per_sq_meter_l350_350043


namespace final_coordinates_are_correct_l350_350726

def applyTransformations(x y z : ℝ) : (ℝ × ℝ × ℝ) :=
  let p1 := (x, y, -z)
  let p2 := (-p1.1, p1.2, p1.3)
  let p3 := (p2.1, -p2.2, p2.3)
  let p4 := (p3.1, p3.2, -p3.3)
  let p5 := (p4.1, -p4.2, p4.3)
  let p6 := (p5.1, p5.2, -p5.3)
  p6

theorem final_coordinates_are_correct :
  applyTransformations 2 2 2 = (-2, 2, -2) :=
by
  sorry

end final_coordinates_are_correct_l350_350726


namespace count_special_three_digit_numbers_l350_350173

theorem count_special_three_digit_numbers : 
    let A := {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ n ≠ 5}
    let B := {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ n ≠ 5}
    let C := {2, 6}
    (finset.univ.filter (λ n, 100 ≤ n ∧ n < 1000 ∧ (n % 2 = 0) ∧ (n % 5 = 0) ∧ ¬ has_digit_5_or_0 n)).card = 128 :=
by {
  sorry
}

def has_digit_5_or_0 (n : ℕ) : bool :=
  let digits := n.digits ∖ {5, 0}
  digits.length ≠ n.digits.length

end count_special_three_digit_numbers_l350_350173


namespace heaviside_step_function_l350_350112

noncomputable def heaviside_integral (t : ℝ) : ℂ :=
  (1 / (2 * real.pi * complex.I)) * ∫ (z : ℂ) in contour, (complex.exp (-complex.I * t * z) / z)

theorem heaviside_step_function (t : ℝ) : 
  (heaviside_integral t) = 
  if t < 0 then 0 else 1 :=
sorry

end heaviside_step_function_l350_350112


namespace smallest_positive_period_l350_350197

noncomputable def tan_period (a b x : ℝ) : ℝ := 
  Real.tan ((a + b) * x / 2)

theorem smallest_positive_period 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ p > 0, ∀ x, tan_period a b (x + p) = tan_period a b x ∧ p = 2 * Real.pi :=
by
  sorry

end smallest_positive_period_l350_350197


namespace total_stamps_l350_350344

theorem total_stamps {books_of_10 books_of_15 : ℕ} (h1 : books_of_10 = 4) (h2 : books_of_15 = 6) :
  (books_of_10 * 10 + books_of_15 * 15 = 130) :=
by
  -- Definitions based on the conditions
  let total_10 := books_of_10 * 10
  let total_15 := books_of_15 * 15
  -- Summing up the total stamps
  have h_total : total_10 + total_15 = 130, from sorry
  exact h_total

end total_stamps_l350_350344


namespace simplify_sqrt_450_l350_350395

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350395


namespace angle_bisector_intersection_point_invariant_l350_350745

structure CollinearThreePoints (A B C : Type*) :=
(collinear : A -> B -> C -> Prop)

noncomputable def circle_gamma (A C P B Q : Type*) := sorry

theorem angle_bisector_intersection_point_invariant
  (A B C : Type*) [CollinearThreePoints A B C]
  (Γ : A → C → Type) (center_not_on_AC : ∀ γ, γ ≠ (A, C))
  (tangent_intersects_at_P : ∀ γ, ∃ P, (is_tangent γ A P) ∧ (is_tangent γ C P))
  (Γ_intersects_PB_at_Q : ∀ γ P B, ∃ Q, (Γ γ ∩ segment PB = Q)) :
  ∃ D, ∀ γ, angle_bisector ∠AQC ∩ line AC = D :=
sorry

end angle_bisector_intersection_point_invariant_l350_350745


namespace find_a_l350_350956

theorem find_a (a : ℝ) (h : ∀ B: ℝ × ℝ, (B = (a, 0)) → (2 - 0) * (0 - 2) = (4 - 2) * (2 - a)) : a = 4 :=
by
  sorry

end find_a_l350_350956


namespace simplify_sqrt_450_l350_350645
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350645


namespace transformation_in_region_S_l350_350814

def region_S (z : ℂ) : Prop := 
  let (x, y) := (z.re, z.im)
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

def transform (z : ℂ) : ℂ := 
  (1/2 + (1/2) * I) * z

theorem transformation_in_region_S (z : ℂ) (hz : region_S z) : region_S (transform z) :=
sorry

end transformation_in_region_S_l350_350814


namespace sum_of_tens_l350_350224

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l350_350224


namespace max_dot_product_l350_350920

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)
def A : ℝ × ℝ := (1, 3 / 2)
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2
def perpendicular (A B C : ℝ × ℝ) : Prop := (B.2 - A.2) * (C.1 - B.1) = -(B.1 - A.1) * (C.2 - B.2)

theorem max_dot_product : ∀ (P : ℝ × ℝ),
    on_ellipse P →
    ((P.1 + 1) * 0 + (P.2) * (3 / 2)) ≤ 3 * Real.sqrt 3 / 2 :=
by
  sorry

#eval max_dot_product

end max_dot_product_l350_350920


namespace calculate_scrap_cookie_radius_l350_350708

-- Declare the variables and conditions
variables (r : ℝ)

-- Large circle of cookie dough with radius 5 inches
def radius_large_dough : ℝ := 5

-- Ten cookies of radius 1 inch each
def radius_small_cookie : ℝ := 1
def number_small_cookies : ℕ := 10

-- One cookie of radius 2 inches
def radius_center_cookie : ℝ := 2

-- Calculations of areas
def area_large_dough : ℝ := π * radius_large_dough^2
def area_small_cookie : ℝ := π * radius_small_cookie^2
def area_total_small_cookies : ℝ := number_small_cookies * area_small_cookie
def area_center_cookie : ℝ := π * radius_center_cookie^2
def total_cookies_area : ℝ := area_total_small_cookies + area_center_cookie
def scrap_area : ℝ := area_large_dough - total_cookies_area

-- Let the radius of the scrap cookie be r
def area_scrap_cookie : ℝ := π * r^2

-- The equation for the area of the scrap cookie should match the leftover area
axiom scrap_cookie_radius_eq : area_scrap_cookie = scrap_area

-- The statement to prove
theorem calculate_scrap_cookie_radius : r = √11 :=
by
  sorry

end calculate_scrap_cookie_radius_l350_350708


namespace domain_of_g_l350_350758

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : {x : ℝ | x > 6^625} = {x : ℝ | ∃ y : ℝ, y = g x } := sorry

end domain_of_g_l350_350758


namespace simplify_sqrt_450_l350_350506

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350506


namespace sqrt_450_simplified_l350_350613

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350613


namespace sqrt_450_eq_15_sqrt_2_l350_350578

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350578


namespace simplify_sqrt_450_l350_350377

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350377


namespace length_of_AB_l350_350232

theorem length_of_AB (A B C D E : Type)
(ABC_right: ∠ B A C = 90)
(BC_eq_8 : dist B C = 8)
(CD_eq_DE : dist C D = dist D E)
(angle_equality : ∠ D C B = ∠ E D A)
(area_triangle_EDC : area (triangle E D C) = 50) :
dist A B = 56 :=
sorry

end length_of_AB_l350_350232


namespace min_value_expression_l350_350866

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ a b : ℝ, a = x^2 - 1 ∧ b = y^2 - 1 ∧
   (∃ ab_pos : a * b > 0, ∀ ab_sqrt : ∀ ab : ℝ, ab = (a * b) →
   1 / a + 1 / b + a / b + b / a ≥ 4) ∧
   ∀ x_eq y_eq : ∀ x' y' : ℝ, y = y' ∧ y = a + 1 ∧ b = x' + 1 ∧ a = 1 ∧ b = 1)
  ↔ (∃ f : ℝ, ∀ x y : ℝ, f = (x^2 / (y^2 - 1) + y^2 / (x^2 - 1)) ∧
  f ≥ 4) := sorry

end min_value_expression_l350_350866


namespace sqrt_450_eq_15_sqrt_2_l350_350472

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350472


namespace total_difference_in_cards_l350_350081

theorem total_difference_in_cards (cards_chris : ℕ) (cards_charlie : ℕ) (cards_diana : ℕ) (cards_ethan : ℕ)
  (h_chris : cards_chris = 18)
  (h_charlie : cards_charlie = 32)
  (h_diana : cards_diana = 25)
  (h_ethan : cards_ethan = 40) :
  (cards_charlie - cards_chris) + (cards_diana - cards_chris) + (cards_ethan - cards_chris) = 43 := by
  sorry

end total_difference_in_cards_l350_350081


namespace sqrt_450_eq_15_sqrt_2_l350_350463

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350463


namespace sqrt_450_eq_15_sqrt_2_l350_350681

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350681


namespace simplify_sqrt_450_l350_350511

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350511


namespace min_payment_max_payment_expected_payment_l350_350319

-- Given Prices
def item_prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

-- Function to compute the actual paid amount given groups of three items
def paid_amount (groups : List (List ℕ)) : ℕ :=
  groups.foldr (λ group sum => sum + group.foldr (λ x s => s + x) 0 - group.minimum') 0

-- Optimal arrangement of items for minimal payment
def optimal_groups : List (List ℕ) :=
  [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]]

-- Suboptimal arrangement of items for maximal payment
def suboptimal_groups : List (List ℕ) :=
  [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]]

-- Expected value calculation's configuration
def num_items := 10
def num_groups := (num_items / 3).natCeil

noncomputable def expected_amount : ℕ :=
  let total_sum := item_prices.foldr (λ x s => s + x) 0
  let expected_savings := 100 * (660 / 72)
  total_sum - expected_savings

theorem min_payment : paid_amount optimal_groups = 4000 := by
  -- Proof steps and details here
  sorry

theorem max_payment : paid_amount suboptimal_groups = 4900 := by
  -- Proof steps and details here
  sorry

theorem expected_payment : expected_amount ≈ 4583 := by
  -- Proof steps and details here
  sorry

end min_payment_max_payment_expected_payment_l350_350319


namespace min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350330

/- Part (a): Minimum amount the customer will pay -/
theorem min_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4000 := 
sorry

/- Part (b): Maximum amount the customer will pay -/
theorem max_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4900 := 
sorry

/- Part (c): Expected value the customer will pay -/
theorem expected_amount_to_pay :
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4583.33 := 
sorry

end min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350330


namespace simplify_sqrt_450_l350_350656

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350656


namespace sqrt_simplify_l350_350349

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350349


namespace simplify_sqrt_450_l350_350499

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350499


namespace simplify_sqrt_450_l350_350447

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350447


namespace simplify_cube_root_l350_350700

noncomputable def cube_root (x : ℝ) : ℝ := real.cbrt x

theorem simplify_cube_root : 
  let a := cube_root (5 * real.sqrt 2 + 7)
      b := cube_root (5 * real.sqrt 2 - 7)
  in a - b = 2 :=
by
  sorry

end simplify_cube_root_l350_350700


namespace log_sum_l350_350016

theorem log_sum : log 2 (1 / 4) + log 2 32 = 3 :=
by sorry

end log_sum_l350_350016


namespace distance_between_parallel_lines_l350_350915

theorem distance_between_parallel_lines 
  (x y : ℝ) 
  (m : ℝ) 
  (h_slope_parallel : (- 6) / m = (-3) / 4)
  (h_m : m ≠ 3 / 4):
  let d := abs ((-3) - (1/2)) / (real.sqrt ((3 ^ 2) + (4 ^ 2))) in
  d = 7 / 10 := by
  sorry

end distance_between_parallel_lines_l350_350915


namespace simplify_sqrt_450_l350_350446

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350446


namespace sqrt_450_equals_15_sqrt_2_l350_350600

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350600


namespace max_boxes_in_warehouse_l350_350134

def warehouse_length : ℕ := 50
def warehouse_width : ℕ := 30
def warehouse_height : ℕ := 5
def box_edge_length : ℕ := 2

theorem max_boxes_in_warehouse : (warehouse_length / box_edge_length) * (warehouse_width / box_edge_length) * (warehouse_height / box_edge_length) = 750 := 
by
  sorry

end max_boxes_in_warehouse_l350_350134


namespace problem_proof_l350_350909

/-
Definitions of the circle and the line
-/
def circle_eq (x y : ℝ) : Prop := (x - 6)^2 + (y - 5)^2 = 16

def line_eq (x y : ℝ) : Prop := x + 3 * y = 12

/-
The distance from a point (x0, y0) to the line Ax + By + C = 0
-/
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / (real.sqrt (A^2 + B^2))

/-
To prove:
1. ∀ P ∈ circle, the distance from P to the line l is less than 7
2. When the angle PAB is maximized, |PA| = 3√5
3. The equation of the line containing the common chord of the circles is 6x + y - 25 = 0
-/

theorem problem_proof :
  (∀ (x y : ℝ), circle_eq x y → distance_point_to_line x y 1 3 (-12) < 7) ∧
  (∀ (P : ℝ × ℝ), (P.1 - 6)^2 + (P.2 - 5)^2 = 16 →
    (let tangent_length := real.sqrt ((12 - 6)^2 + (0 - 5)^2 - 16) in tangent_length = 3 * real.sqrt 5)) ∧
  (∀ (BC_x BC_y : ℝ), circle_eq BC_x BC_y →
    let midpoint := ((BC_x + 0) / 2, (BC_y + 4) / 2) in
    let common_chord := λ x y, x (x - 6) + (y - 4) (y - 5) = 0 in
    common_chord x y =
    λ x y, (6 * x + y - 25)) :=
by sorry

end problem_proof_l350_350909


namespace max_jogs_possible_l350_350071

theorem max_jogs_possible :
  ∃ (x y z : ℕ), (3 * x + 4 * y + 10 * z = 100) ∧ (x + y + z ≥ 20) ∧ (x ≥ 1) ∧ (y ≥ 1) ∧ (z ≥ 1) ∧
  (∀ (x' y' z' : ℕ), (3 * x' + 4 * y' + 10 * z' = 100) ∧ (x' + y' + z' ≥ 20) ∧ (x' ≥ 1) ∧ (y' ≥ 1) ∧ (z' ≥ 1) → z' ≤ z) :=
by
  sorry

end max_jogs_possible_l350_350071


namespace f_nonnegative_when_a_ge_one_l350_350155

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp (2 * x) + (a - 2) * Real.exp x - x

noncomputable def h (a : ℝ) : ℝ := Real.log a + 1 - (1 / a)

theorem f_nonnegative_when_a_ge_one (a : ℝ) (x : ℝ) (h_a : a ≥ 1) : f a x ≥ 0 := by
  sorry  -- Placeholder for the proof.

end f_nonnegative_when_a_ge_one_l350_350155


namespace probability_divisible_by_5_l350_350857

noncomputable def count_total_numbers : ℕ := 3^4

noncomputable def divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

theorem probability_divisible_by_5 : ∀ (spins : ℕ → ℕ), 
  (∀ n, n < 4 → spins n ∈ {1, 2, 3}) →
  0 = 
  ((finset.filter (λ n, divisible_by_5 (spins 0 * 1000 + spins 1 * 100 + spins 2 * 10 + spins 3)) (finset.range count_total_numbers)).card / count_total_numbers : ℚ) :=
begin
  intros spins h,
  have no_valid_numbers : (finset.filter (λ n, divisible_by_5 (spins 0 * 1000 + spins 1 * 100 + spins 2 * 10 + spins 3)) (finset.range count_total_numbers)).card = 0,
  { 
    -- Since the spinner digits are 1, 2, or 3, it can never end in 0 or 5.
    sorry
  },
  rw no_valid_numbers,
  norm_num,
end

end probability_divisible_by_5_l350_350857


namespace simplify_sqrt_450_l350_350518

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350518


namespace remaining_employee_number_l350_350803

theorem remaining_employee_number (n : ℕ) (employees : Finset ℕ) 
  (h_size : employees.card = 4)
  (h_employee_1 : 6 ∈ employees)
  (h_employee_2 : 32 ∈ employees)
  (h_employee_3 : 45 ∈ employees)
  (h_total : Finset.range 52 = {i | i < 52}) :
  ∃ x, x ∈ employees ∧ x = 19 :=
by 
  sorry

end remaining_employee_number_l350_350803


namespace simplify_sqrt_450_l350_350401

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350401


namespace cos_half_angle_l350_350883

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < Real.pi / 2) : 
    Real.cos (α / 2) = 2 * Real.sqrt 5 / 5 := 
by 
    sorry

end cos_half_angle_l350_350883


namespace find_m_range_l350_350031

variable {α : Type*}
variable {f : α → ℝ}

def is_even (f : α → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

def is_monotonic_decreasing (f : α → ℝ) : Prop :=
∀ x1 x2 : ℝ, 0 ≤ x1 → x1 < x2 → x2 ≤ 2 → f x1 > f x2

theorem find_m_range (f_even : is_even f) (f_decreasing : is_monotonic_decreasing f):
  ∀ m : ℝ, (f (1 - m) < f m) → -2 ≤ m ∧ m < 0.5 :=
by
  intro m h
  sorry


end find_m_range_l350_350031


namespace find_B_find_A_l350_350958

variables (A B C a b c : ℝ)

-- Conditions
axiom triangle_sides : a = A ∧ b = B ∧ c = C
axiom condition1 : b * sin A = sqrt(3) * a * cos B
axiom condition2 : cos A * sin C = (sqrt(3) - 1) / 4

-- Questions to prove
theorem find_B : B = π / 3 := 
sorry

theorem find_A : A = 5 * π / 12 :=
sorry

end find_B_find_A_l350_350958


namespace find_angle_l350_350916

theorem find_angle (A : ℝ) (hA1 : 0 < A) (hA2 : A < π) 
  (hCollinear : sin A * (sin A + (√3) * cos A) - (3 / 2) = 0) : 
  A = π / 3 :=
begin
  sorry
end

end find_angle_l350_350916


namespace mutually_exclusive_events_option_C_l350_350769

-- Define the events and their possible occurrences based on the number of heads
def at_least_one_head := { (1, 2), (2, 1), (3, 0) }  -- At least one head: 1 head, 2 heads, 3 heads
def at_most_one_head := { (1, 2), (0, 3) }            -- At most one head: 1 head, 0 heads
def exactly_two_heads := { (2, 1) }                   -- Exactly two heads: 2 heads
def at_least_two_heads := { (2, 1), (3, 0) }          -- At least two heads: 2 heads, 3 heads
def exactly_one_head := { (1, 2) }                    -- Exactly one head: 1 head

-- Define mutual exclusivity
def mutually_exclusive (A B : Set (ℕ × ℕ)) : Prop := 
  (A ∩ B).empty

-- Statement of the proof problem
theorem mutually_exclusive_events_option_C :
  mutually_exclusive at_most_one_head at_least_two_heads := 
by
  -- This is the main statement we need to prove in Lean
  sorry

end mutually_exclusive_events_option_C_l350_350769


namespace inverse_function_l350_350922

-- Definitions and conditions
def f (x : ℝ) : ℝ := (1 / 2)^(x - 1)

-- Theorem statement
theorem inverse_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  f (1 - Real.log x / Real.log 2) = x := sorry

end inverse_function_l350_350922


namespace sum_of_reciprocals_inequality_l350_350297

theorem sum_of_reciprocals_inequality (m n : ℕ) (h₁ : n ≥ m) (h₂ : m ≥ 1) :
  ∑ k in Finset.range (n + 1) \ Finset.range (m), (1 / (k : ℝ) ^ 2 + 1 / (k : ℝ) ^ 3) 
  ≥ m * (∑ k in Finset.range (n + 1) \ Finset.range (m), 1 / (k : ℝ) ^ 2) ^ 2 := 
by 
  sorry

end sum_of_reciprocals_inequality_l350_350297


namespace number_of_tens_in_sum_l350_350217

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l350_350217


namespace angle_ADC_l350_350725

theorem angle_ADC (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (angle_ABC_eq_50 : ∠ B A C = 50)
  (AD_bisects_BAC : ∀ (A B C D : Type), ∠ A D C = (∠ B A D) / 2)
  (DC_bisects_BCA : ∀ (A B C D : Type), ∠ D C A = (∠ B C A) / 2):
  ∠ A D C = 115 := 
  sorry

end angle_ADC_l350_350725


namespace inequality_relationship_l350_350884

open Real

noncomputable def a := 2 ^ 0.7
noncomputable def b := (1 / 3) ^ 0.7
noncomputable def c := log_base 2 (1 / 3)

theorem inequality_relationship :
  a > b ∧ b > c := by
  sorry

end inequality_relationship_l350_350884


namespace simplify_sqrt_450_l350_350455

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350455


namespace angles_zero_and_180_l350_350010

/--
Let M₁, M₂, ..., Mₙ be arbitrary n points in the plane.
The total number of angles Mᵢ Mⱼ Mₖ (where i, j, and k take any of the values 
1, 2, 3, ..., n) formed by these points is denoted by N(n). 

Prove that:
1) n(n-1)(n-2)/3 of these angles can be equal to 0°.
2) n(n-1)(n-2)/6 of these angles can be equal to 180°.
-/
theorem angles_zero_and_180 {M : ℕ → ℝ × ℝ} {n : ℕ} (h : M 1 = (1, 0) ∧ M n = (n, 0)) : 
  (∃ k : ℕ, k = (n * (n-1) * (n-2)) / 3) ∧ (∃ m : ℕ, m = (n * (n-1) * (n-2)) / 6) :=
  sorry

end angles_zero_and_180_l350_350010


namespace sqrt_450_equals_15_sqrt_2_l350_350610

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350610


namespace min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350329

/- Part (a): Minimum amount the customer will pay -/
theorem min_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4000 := 
sorry

/- Part (b): Maximum amount the customer will pay -/
theorem max_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4900 := 
sorry

/- Part (c): Expected value the customer will pay -/
theorem expected_amount_to_pay :
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4583.33 := 
sorry

end min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350329


namespace volume_ratio_l350_350728

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_hemisphere (r : ℝ) (k : ℝ) : ℝ :=
  k * (1 / 2) * (4 / 3) * Real.pi * (3 * r)^3

theorem volume_ratio (r : ℝ) (k : ℝ) (h : k = 2 / 3) :
  (volume_sphere r / volume_hemisphere r k) = 2 / 27 :=
by
  unfold volume_sphere volume_hemisphere
  rw [h]
  have h1 : (4 / 3 : ℝ) * Real.pi * r^3 = Real.pi * (4 / 3 * r^3), by ring
  rw [h1]
  have h2 : k * (1 / 2) * (4 / 3) * Real.pi * (3 * r)^3 = Real.pi * (k * 1 / 2 * 4 / 3 * (3 * r)^3), by ring
  rw [h2, ← mul_assoc, mul_comm (3:ℝ), pow_three, mul_assoc (3^3:ℝ), nat.cast_pow, mul_comm ((nat.pow 3 3):ℝ)]
  simp only [mul_assoc, mul_comm, ← mul_div_assoc, ← div_div]
  norm_num
  sorry

end volume_ratio_l350_350728


namespace choose_5_from_11_l350_350243

theorem choose_5_from_11 : (nat.choose 11 5) = 462 := by
  sorry

end choose_5_from_11_l350_350243


namespace find_y_l350_350917

theorem find_y (x y z : ℝ) (h1 : x^2 * y = z) (h2 : x / y = 36) (h3 : sqrt(x * y) = z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  y = 1 / 14.7 := sorry

end find_y_l350_350917


namespace total_amount_is_33_l350_350032

variable (n : ℕ) (c t : ℝ)

def total_amount_paid (n : ℕ) (c t : ℝ) : ℝ :=
  let cost_before_tax := n * c
  let tax := t * cost_before_tax
  cost_before_tax + tax

theorem total_amount_is_33
  (h1 : n = 5)
  (h2 : c = 6)
  (h3 : t = 0.10) :
  total_amount_paid n c t = 33 :=
by
  rw [h1, h2, h3]
  sorry

end total_amount_is_33_l350_350032


namespace sqrt_450_eq_15_sqrt_2_l350_350558

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350558


namespace simplify_sqrt_450_l350_350490

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350490


namespace collinear_implies_m_acute_angle_implies_range_l350_350170

section

variables {m : ℝ}
variables (OA OB OC : ℝ × ℝ)
-- Given vectors
def vec_OA := (3 : ℝ, -4 : ℝ)
def vec_OB := (6 : ℝ, -3 : ℝ)
def vec_OC := (5 - m, -3 - m)

-- For proof of collinearity
def AB_proportional_AC (m : ℝ) : Prop :=
  let AB := (vec_OB.1 - vec_OA.1, vec_OB.2 - vec_OA.2) in
  let AC := (vec_OC.1 - vec_OA.1, vec_OC.2 - vec_OA.2) in
  AB.1 * AC.2 = AB.2 * AC.1

-- For proof of acute angle ∠ABC
def acute_angle_ABC (m : ℝ) : Prop :=
  let BA := (vec_OA.1 - vec_OB.1, vec_OA.2 - vec_OB.2) in
  let BC := (vec_OC.1 - vec_OB.1, vec_OC.2 - vec_OB.2) in
  BA.1 * BC.1 + BA.2 * BC.2 > 0

-- Proposition 1: Prove collinearity implies m = 1/2
theorem collinear_implies_m (h : AB_proportional_AC m) : m = 1 / 2 :=
  sorry

-- Proposition 2: Prove acute angle ∠ABC implies range of m
theorem acute_angle_implies_range (h : acute_angle_ABC m) : 
   m ∈ Set.Ioo (-3 / 4) (1 / 2) ∨ m ∈ Set.Ioi (1 / 2) :=
  sorry

end

end collinear_implies_m_acute_angle_implies_range_l350_350170


namespace f_prime_at_1_l350_350156

variable {a b : ℝ}

def f (x : ℝ) : ℝ :=
  a * x^2 + b * Real.cos x

noncomputable def f_prime (x : ℝ) : ℝ :=
  2 * a * x - b * Real.sin x

theorem f_prime_at_1 (h : f_prime (-1) = 2) : f_prime 1 = -2 :=
by
  sorry

end f_prime_at_1_l350_350156


namespace student_weight_l350_350948

-- Definitions based on conditions
variables (S R : ℝ)

-- Conditions as assertions
def condition1 : Prop := S - 5 = 2 * R
def condition2 : Prop := S + R = 104

-- The statement we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 71 :=
by
  sorry

end student_weight_l350_350948


namespace minimum_value_f_l350_350999

-- Define the sequence
def a : ℕ → ℕ
| 1     := 2
| (n+1) := a n + 2

-- Define the sum of the first n terms
def S (n : ℕ) : ℕ := Nat.rec_on n 0 (λ n S_n, S_n + a (n+1))

-- Define the function f
def f (n : ℕ) : ℝ := (S n + 60 : ℝ) / (n + 1)

-- The main theorem statement
theorem minimum_value_f : Real.toRat (Inf (Set.range (λ n : ℕ, f (n+1)))) = 29 / 2 := sorry

end minimum_value_f_l350_350999


namespace sequence_is_increasing_l350_350885

-- Define the sequence recurrence property
def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3

-- The theorem statement
theorem sequence_is_increasing (a : ℕ → ℤ) (h : sequence_condition a) : 
  ∀ n : ℕ, a n < a (n + 1) :=
by
  unfold sequence_condition at h
  intro n
  specialize h n
  sorry

end sequence_is_increasing_l350_350885


namespace simplify_sqrt_450_l350_350399

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350399


namespace sqrt_simplify_l350_350350

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350350


namespace simplify_sqrt_450_l350_350391

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350391


namespace sqrt_450_eq_15_sqrt_2_l350_350553

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350553


namespace sqrt_450_eq_15_sqrt_2_l350_350552

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350552


namespace evaluate_expression_l350_350073

theorem evaluate_expression :
  (|(-3 : ℝ)| + (real.sqrt 3 * real.sin (real.pi / 3)) - (2 : ℝ)⁻¹) = 4 :=
by
  sorry

end evaluate_expression_l350_350073


namespace inequality_holds_l350_350284

theorem inequality_holds (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 :=
by sorry

end inequality_holds_l350_350284


namespace distance_karen_covers_l350_350735

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end distance_karen_covers_l350_350735


namespace simplify_sqrt_450_l350_350527

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350527


namespace sqrt_simplify_l350_350369

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350369


namespace sqrt_450_simplified_l350_350426

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350426


namespace simplify_sqrt_450_l350_350495

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350495


namespace number_of_tens_in_sum_l350_350209

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l350_350209


namespace Johnson_Carter_Tie_August_l350_350710

structure MonthlyHomeRuns where
  March : Nat
  April : Nat
  May : Nat
  June : Nat
  July : Nat
  August : Nat
  September : Nat

def Johnson_runs : MonthlyHomeRuns := { March:= 2, April:= 11, May:= 15, June:= 9, July:= 7, August:= 9, September:= 0 }
def Carter_runs : MonthlyHomeRuns := { March:= 1, April:= 9, May:= 8, June:= 19, July:= 6, August:= 10, September:= 0 }

noncomputable def cumulative_runs (runs: MonthlyHomeRuns) (month: String) : Nat :=
  match month with
  | "March" => runs.March
  | "April" => runs.March + runs.April
  | "May" => runs.March + runs.April + runs.May
  | "June" => runs.March + runs.April + runs.May + runs.June
  | "July" => runs.March + runs.April + runs.May + runs.June + runs.July
  | "August" => runs.March + runs.April + runs.May + runs.June + runs.July + runs.August
  | _ => 0

theorem Johnson_Carter_Tie_August :
  cumulative_runs Johnson_runs "August" = cumulative_runs Carter_runs "August" := 
  by
  sorry

end Johnson_Carter_Tie_August_l350_350710


namespace tank_capacity_l350_350945

theorem tank_capacity : 
  ∀ (T : ℚ), (3 / 4) * T + 8 = (7 / 8) * T → T = 64 := by
  intros T h
  have h1 : (7 / 8) * T - (3 / 4) * T = 8 := by linarith
  have h2 : (1 / 8) * T = 8 := by linarith
  have h3 : T = 8 * 8 := by calc
    T = 8 * (8 : ℚ) : by rw [h2, rat.inv_mul_eq_iff (ne_of_eq_of_ne (by norm_num) one_ne_zero)]
  rw h3
  norm_num


end tank_capacity_l350_350945


namespace seq_formula_ineq_holds_l350_350288

-- Define the sequence as per the given condition.
def seq (n : ℕ) : ℕ := 2^n - 1

-- Define the binomial coefficient.
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the inequality.
def ineq (n : ℕ) : Prop :=
  (∑ k in Finset.range (n + 1), seq k * binom n k) < 2023

-- Prove that the sequence formula is correct.
theorem seq_formula (n : ℕ) : 
  2 * (2^n - 1 - (2^(n-1) - 1)) + 1 = 2^n - 1 :=
by sorry

-- Prove that the inequality holds for n ≤ 6.
theorem ineq_holds (n : ℕ) (h : n ≤ 6) : ineq n :=
by sorry

end seq_formula_ineq_holds_l350_350288


namespace equal_roots_quadratic_solution_l350_350952

theorem equal_roots_quadratic_solution (m : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x + m = 0 ∧ ∀ y : ℝ, (y^2 - 2 * y + m = 0 → y = x)) → m = 1 := 
begin
  sorry
end

end equal_roots_quadratic_solution_l350_350952


namespace sqrt_450_simplified_l350_350429

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350429


namespace simplify_sqrt_450_l350_350449

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350449


namespace sqrt_450_simplified_l350_350430

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350430


namespace Panthers_total_games_l350_350829

/-
Given:
1) The Panthers had won 60% of their basketball games before the district play.
2) During district play, they won four more games and lost four.
3) They finished the season having won half of their total games.
Prove that the total number of games they played in all is 48.
-/

theorem Panthers_total_games
  (y : ℕ) -- total games before district play
  (x : ℕ) -- games won before district play
  (h1 : x = 60 * y / 100) -- they won 60% of the games before district play
  (h2 : (x + 4) = 50 * (y + 8) / 100) -- they won half of the total games including district play
  : (y + 8) = 48 := -- total games they played in all
sorry

end Panthers_total_games_l350_350829


namespace sqrt_450_simplified_l350_350630

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350630


namespace simplify_sqrt_450_l350_350520

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350520


namespace sqrt_450_eq_15_sqrt_2_l350_350693

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350693


namespace simplify_sqrt_450_l350_350641
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350641


namespace number_of_sports_books_l350_350941

def total_books : ℕ := 58
def school_books : ℕ := 19
def sports_books (total_books school_books : ℕ) : ℕ := total_books - school_books

theorem number_of_sports_books : sports_books total_books school_books = 39 := by
  -- proof goes here
  sorry

end number_of_sports_books_l350_350941


namespace sqrt_450_simplified_l350_350416

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350416


namespace go_stones_problem_l350_350856

theorem go_stones_problem
  (x : ℕ) 
  (h1 : x / 7 + 40 = 555 / 5) 
  (black_stones : ℕ) 
  (h2 : black_stones = 55) :
  (x - black_stones = 442) :=
sorry

end go_stones_problem_l350_350856


namespace log_base_2x_eq_x_impl_non_square_non_cube_integer_l350_350185

theorem log_base_2x_eq_x_impl_non_square_non_cube_integer 
(h : ∃ x : ℝ, x > 0 ∧ log (2 * x) 216 = x) : 
  ∃ x : ℕ, log (2 * x) 216 = x ∧ ¬ (∃ n : ℕ, sqrt x = n) ∧ ¬ (∃ m : ℕ, cbrt x = m) := 
sorry

end log_base_2x_eq_x_impl_non_square_non_cube_integer_l350_350185


namespace sqrt_450_simplified_l350_350619

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350619


namespace simplify_sqrt_450_l350_350454

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350454


namespace sum_of_tens_l350_350203

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l350_350203


namespace A_inter_complement_B_eq_01_l350_350167

open Set

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | x ≥ 1}
def complement_B : Set ℝ := U \ B

theorem A_inter_complement_B_eq_01 : A ∩ complement_B = (Set.Ioo 0 1) := 
by 
  sorry

end A_inter_complement_B_eq_01_l350_350167


namespace imaginary_part_of_z_l350_350949

open Complex -- open complex number functions

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) * (2 - I) = 5 * I) :
  z.im = 2 :=
sorry

end imaginary_part_of_z_l350_350949


namespace num_terms_100_pow_10_as_sum_of_tens_l350_350210

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l350_350210


namespace slope_of_line_l350_350870

theorem slope_of_line (x y : ℝ) : 
  (∃ b : ℝ, 7 * x - 2 * y = 14) → 
  (∃ b : ℝ, y = (7/2) * x + b) :=
begin
  sorry
end

end slope_of_line_l350_350870


namespace set_of_a_l350_350193

noncomputable def f (a x : ℝ) := log a (x^2 + a*x + 4)

theorem set_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, ∃ y : ℝ, f a y < x) ↔ (0 < a ∧ a < 1) ∨ (a ≥ 4) :=
by
  sorry

end set_of_a_l350_350193


namespace set_complement_union_eq_l350_350303

open Set

variable (U : Set ℕ) (P : Set ℕ) (Q : Set ℕ)

theorem set_complement_union_eq :
  U = {1, 2, 3, 4, 5, 6} →
  P = {1, 3, 5} →
  Q = {1, 2, 4} →
  (U \ P) ∪ Q = {1, 2, 4, 6} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end set_complement_union_eq_l350_350303


namespace solve_problem_statement_l350_350157

noncomputable def problem_statement : Prop :=
  ∀ (a : ℝ),
  (∀ (x : ℝ), x ∈ set.Icc (-2 * Real.pi / 3) a → y = (Real.sin x)^2 + 2 * Real.cos x)
    ∧ (y ∈ set.Icc (-1/4) 2) 
    → 0 ≤ a ∧ a ≤ 2 * Real.pi / 3

theorem solve_problem_statement (a : ℝ) (h : problem_statement a) : 0 ≤ a ∧ a ≤ 2 * Real.pi / 3 := sorry

end solve_problem_statement_l350_350157


namespace shaded_area_of_floor_l350_350794

theorem shaded_area_of_floor : 
  let floor_length := 12
  let floor_width := 15
  let tile_side := 2
  let radius := 1
  let num_tiles := (floor_length / tile_side) * (floor_width / tile_side)
  let tile_area := tile_side * tile_side
  let circle_area := Real.pi * radius * radius
  let shaded_area_per_tile := tile_area - circle_area
  let total_shaded_area := num_tiles * shaded_area_per_tile
  total_shaded_area = 180 - 45 * Real.pi := 
by {
  let floor_length := 12
  let floor_width := 15
  let tile_side := 2
  let radius := 1
  let num_tiles := (floor_length / tile_side) * (floor_width / tile_side)
  let tile_area := tile_side * tile_side
  let circle_area := Real.pi * radius * radius
  let shaded_area_per_tile := tile_area - circle_area
  let total_shaded_area := num_tiles * shaded_area_per_tile
  show total_shaded_area = 180 - 45 * Real.pi
  sorry
}

end shaded_area_of_floor_l350_350794


namespace simplify_sqrt_450_l350_350663

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350663


namespace inequality_solution_l350_350775

variable (x : ℝ)

theorem inequality_solution : 
  x > 1 →
  9^(Real.log (x-1) / Real.log 2 - 1) - 8 * 5^(Real.log (x-1) / Real.log 2 - 2) > 
  9^(Real.log (x-1) / Real.log 2) - 16 * 5^(Real.log (x-1) / Real.log 2 - 1) ↔
  1 < x ∧ x < 5 := 
by
  sorry

end inequality_solution_l350_350775


namespace sin_double_alpha_trig_expression_l350_350143

theorem sin_double_alpha (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin (2 * α) = 4 * Real.sqrt 2 / 9 :=
sorry

theorem trig_expression (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin (α - 2 * π) * Real.cos (2 * π - α)) / (Real.sin (α + π / 2) ^ 2) = Real.sqrt 2 / 4 :=
sorry

end sin_double_alpha_trig_expression_l350_350143


namespace abs_diff_x_plus_1_x_minus_2_l350_350289

theorem abs_diff_x_plus_1_x_minus_2 (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : |x + 1| - |x - 2| = -3 :=
by
  sorry

end abs_diff_x_plus_1_x_minus_2_l350_350289


namespace parallel_tangent_line_exists_l350_350111

def equation_of_line_parallel_and_tangent : Prop :=
  ∃ b : ℝ, (∀ x y : ℝ, x + y + b = 0) ∧ (∀ x y : ℝ, x^2 + y^2 = 2) → (b = 2 ∨ b = -2)

-- Lean 4 statement for the theorem to be proved
theorem parallel_tangent_line_exists :
  equation_of_line_parallel_and_tangent :=
begin
  sorry
end

end parallel_tangent_line_exists_l350_350111


namespace differences_multiple_of_nine_l350_350088

theorem differences_multiple_of_nine (S : Finset ℕ) (hS : S.card = 10) (h_unique : ∀ {x y : ℕ}, x ∈ S → y ∈ S → x ≠ y → x ≠ y) : 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b) % 9 = 0 :=
by
  sorry

end differences_multiple_of_nine_l350_350088


namespace train_length_l350_350818

theorem train_length (L : ℝ) (h1 : 120 * (L / 120) = L) (h2 : 180 * (L + 600) / 180 = L + 600) : L = 1200 :=
by
  have speed_tree : L / 120 = L / 120 := by rfl
  have speed_platform : (L + 600) / 180 = (L + 600) / 180 := by rfl
  have eq_speeds : L / 120 = (L + 600) / 180 := by
    sorry
  calc
    L     = 1200  : by
      sorry

end train_length_l350_350818


namespace polynomial_root_l350_350263

theorem polynomial_root (x0 : ℝ) (z : ℝ) 
  (h1 : x0^3 - x0 - 1 = 0) 
  (h2 : z = x0^2 + 3 * x0 + 1) : 
  z^3 - 5 * z^2 - 10 * z - 11 = 0 := 
sorry

end polynomial_root_l350_350263


namespace no_closed_25_polygon_l350_350898

/--
Given a system of 25 distinct line segments all starting from a common point \( A \)
and ending on a line \( l \) that does not pass through this point \( A \),
prove that there does not exist a closed 25-sided polygon such that
each side of the polygon is equal to and parallel to one of the line segments in the system.
-/
theorem no_closed_25_polygon {l : Set (ℝ × ℝ)} {A : ℝ × ℝ}
  (h_A_not_on_l : A ∉ l) 
  (segments : Fin 25 → (ℝ × ℝ))
  (distinct_segments : Function.Injective segments)
  (line_ends : ∀ i, segments i ∈ l) :
  ¬ ∃ (polygon : Fin 25 → (ℝ × ℝ)), 
    (∀ i, polygon i = segments i ∨ polygon i = -segments i) ∧ 
    (polygon 0 + polygon 1 + polygon 2 + ... + polygon 24 = (0, 0)) :=
by
  sorry

end no_closed_25_polygon_l350_350898


namespace total_robots_correct_l350_350739

def number_of_shapes : ℕ := 3
def number_of_colors : ℕ := 4
def total_types_of_robots : ℕ := number_of_shapes * number_of_colors

theorem total_robots_correct : total_types_of_robots = 12 := by
  sorry

end total_robots_correct_l350_350739


namespace archie_sod_needed_l350_350064

theorem archie_sod_needed 
  (backyard_length : ℝ) (backyard_width : ℝ) (shed_length : ℝ) (shed_width : ℝ)
  (backyard_area : backyard_length = 20 ∧ backyard_width = 13)
  (shed_area : shed_length = 3 ∧ shed_width = 5)
  : backyard_length * backyard_width - shed_length * shed_width = 245 := 
by
  unfold backyard_length backyard_width shed_length shed_width
  sorry

end archie_sod_needed_l350_350064


namespace linear_independent_vectors_p_value_l350_350850

theorem linear_independent_vectors_p_value (p : ℝ) :
  (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ a * (2 : ℝ) + b * (5 : ℝ) = 0 ∧ a * (4 : ℝ) + b * p = 0) ↔ p = 10 :=
by
  sorry

end linear_independent_vectors_p_value_l350_350850


namespace interval_decreasing_l350_350098

noncomputable def f (x : ℝ) := x^2 - 4 * real.log (x + 1)

theorem interval_decreasing : 
  ∀ x, -1 < x ∧ x < 1 → ∃ ε > 0, ∀ δ ∈ Ioo (x - ε) (x + ε), f(δ) < f(x) :=
by
  sorry

end interval_decreasing_l350_350098


namespace geometric_sequence_arith_condition_l350_350238

-- Definitions of geometric sequence and arithmetic sequence condition
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions: \( \{a_n\} \) is a geometric sequence with \( a_2 \), \( \frac{1}{2}a_3 \), \( a_1 \) forming an arithmetic sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := ∀ n, a (n + 1) = q * a n

def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop := a 2 = (1 / 2) * a 3 + a 1

-- Final theorem to prove
theorem geometric_sequence_arith_condition (hq : q^2 - q - 1 = 0) 
  (hgeo : is_geometric_sequence a q) 
  (harith : arithmetic_sequence_condition a) : 
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_arith_condition_l350_350238


namespace circle_chords_diameter_intersection_l350_350236

theorem circle_chords_diameter_intersection
  (O : Point) (R : ℝ) (A B C D E F : Point)
  (hAB_eq_BC_eq_CD : dist A B = dist B C ∧ dist B C = dist C D)
  (h_circle : ∀ P : Point, dist O P = R)
  (h_diameter_BE : collinear B E ∧ dist B O = dist O E ∧ dist B E = 2 * R)
  (h_AD_intersection_BE_at_F : collinear A D F ∧ collinear B E F)
  (h_line_CE : collinear C E) :
  dist A B = dist A F ∧ 
  ∃ P : Point, midpoint P F D ∧ collinear C E P :=
sorry

end circle_chords_diameter_intersection_l350_350236


namespace simplify_sqrt_450_l350_350443

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350443


namespace sqrt_450_eq_15_sqrt_2_l350_350583

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350583


namespace jill_investment_value_l350_350004

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem jill_investment_value :
  compound_interest 10000 0.0396 2 2 ≈ 10815.66 :=
by
  sorry

end jill_investment_value_l350_350004


namespace simplify_expression_l350_350704

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end simplify_expression_l350_350704


namespace simplify_sqrt_450_l350_350411

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350411


namespace interval_of_monotonic_increase_l350_350301

noncomputable def f : ℝ → ℝ :=
λ x, if x < 2 then (x - 1) ^ 2 else 2 / x

theorem interval_of_monotonic_increase :
  { x : ℝ | 1 ≤ x ∧ x < 2 } = { x : ℝ | ∀ y, f y < f x → y < x ∧ y ∈ { x : ℝ | 1 ≤ x ∧ x < 2 }} :=
sorry

end interval_of_monotonic_increase_l350_350301


namespace compute_fractions_product_l350_350840

theorem compute_fractions_product :
  (2 * (2^4 - 1) / (2 * (2^4 + 1))) *
  (2 * (3^4 - 1) / (2 * (3^4 + 1))) *
  (2 * (4^4 - 1) / (2 * (4^4 + 1))) *
  (2 * (5^4 - 1) / (2 * (5^4 + 1))) *
  (2 * (6^4 - 1) / (2 * (6^4 + 1))) *
  (2 * (7^4 - 1) / (2 * (7^4 + 1)))
  = 4400 / 135 := by
sorry

end compute_fractions_product_l350_350840


namespace prove_parabola_eq_l350_350036

noncomputable def parabola_eq (p : ℝ) (hp : p > 0) : Prop :=
  ∃ (x0 : ℝ), x0 = 3 - p / 2 ∧ x0 > p / 2 ∧
  (∀ (y0 : ℝ), y0 = 2 * Real.sqrt 2 * (3 - p),
  8 * (3 - p)^2 = 2 * p * (3 - p / 2) ∧ p = 2) →
  y0 ^ 2 = 4 * x0

theorem prove_parabola_eq : parabola_eq 2 (by norm_num) :=
by
  sorry

end prove_parabola_eq_l350_350036


namespace calculate_cherry_pies_l350_350346

-- Definitions for the conditions
def total_pies : ℕ := 40
def ratio_parts_apple : ℕ := 2
def ratio_parts_blueberry : ℕ := 5
def ratio_parts_cherry : ℕ := 3
def total_ratio_parts := ratio_parts_apple + ratio_parts_blueberry + ratio_parts_cherry

-- Calculating the number of pies per part and then the number of cherry pies
def pies_per_part : ℕ := total_pies / total_ratio_parts
def cherry_pies : ℕ := ratio_parts_cherry * pies_per_part

-- Proof statement
theorem calculate_cherry_pies : cherry_pies = 12 :=
by
  -- Lean proof goes here
  sorry

end calculate_cherry_pies_l350_350346


namespace range_of_a_l350_350152

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x < 3) → (4 * a * x + 4 * (a - 3)) ≤ 0) ↔ (0 ≤ a ∧ a ≤ 3 / 4) :=
by
  sorry

end range_of_a_l350_350152


namespace geometric_sequence_root_14_l350_350983

theorem geometric_sequence_root_14
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h_roots : ∃ b4 b24 : ℝ, (3 * b4 * b4 - 2014 * b4 + 9 = 0) ∧ (3 * b24 * b24 - 2014 * b24 + 9 = 0) ∧ (∃ r : ℝ, a 4 = b4 ∧ a 24 = b24 ∧ ∀ n : ℕ, a (n + 1) = r * a n)) :
  a 14 = sqrt 3 :=
by
  sorry

end geometric_sequence_root_14_l350_350983


namespace min_payment_proof_max_payment_proof_expected_payment_proof_l350_350323

noncomputable def items : List ℕ := List.range 1 11 |>.map (λ n => n * 100)

def min_amount_paid : ℕ :=
  (1000 + 900 + 700 + 600 + 400 + 300 + 100)

def max_amount_paid : ℕ :=
  (1000 + 900 + 800 + 700 + 600 + 500 + 400)

def expected_amount_paid : ℚ :=
  4583 + 33 / 100

theorem min_payment_proof :
  (∑ x in (List.range 15).filter (λ x => x % 3 ≠ 0), (items.get! x : ℕ)) = min_amount_paid := by
  sorry

theorem max_payment_proof :
  (∑ x in List.range 10, if x % 3 = 0 then 0 else (items.get! x : ℕ)) = max_amount_paid := by
  sorry

theorem expected_payment_proof :
  ∑ k in items, ((k : ℚ) * (∏ m in List.range 9, (10 - m) * (9 - m) / 72)) = expected_amount_paid := by
  sorry

end min_payment_proof_max_payment_proof_expected_payment_proof_l350_350323


namespace height_of_Leifeng_Pagoda_l350_350709

-- Definitions from the problem
def AC : Float := 62 * Real.sqrt 2
def alpha : Float := 45 * Real.pi / 180
def beta : Float := 15 * Real.pi / 180

-- Theorem stating the height of the Leifeng Pagoda (BC)
theorem height_of_Leifeng_Pagoda : BC = 62 := by
  -- placeholder for the actual proof
  sorry

end height_of_Leifeng_Pagoda_l350_350709


namespace simplify_sqrt_450_l350_350452

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350452


namespace pants_cost_l350_350342

-- Definitions based on conditions
def num_dresses : ℕ := 5
def num_pants : ℕ := 3
def num_jackets : ℕ := 4
def cost_dress : ℕ := 20
def cost_jacket : ℕ := 30
def trans_cost : ℕ := 5
def initial_amount : ℕ := 400
def remaining_amount : ℕ := 139

-- The statement to prove
theorem pants_cost :
  let total_spent := initial_amount - remaining_amount in
  let total_dresses_cost := num_dresses * cost_dress in
  let total_jackets_cost := num_jackets * cost_jacket in
  let total_other_cost := total_dresses_cost + total_jackets_cost + trans_cost in
  let spent_on_pants := total_spent - total_other_cost in
  spent_on_pants / num_pants = 12 :=
by
  sorry

end pants_cost_l350_350342


namespace sqrt_450_eq_15_sqrt_2_l350_350476

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350476


namespace sum_of_tens_l350_350223

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l350_350223


namespace simplify_sqrt_450_l350_350652
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350652


namespace sqrt_simplify_l350_350365

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350365


namespace total_biking_distance_l350_350737

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end total_biking_distance_l350_350737


namespace total_chess_games_l350_350007

theorem total_chess_games (P : ℕ) (two_players_per_game : ℕ) 
  (players : P = 8) (games_per_player : two_players_per_game = 2) : 
  ∑ i in Finset.range (P - 1), i + 1 = 28 :=
by 
  -- sorry allows us to compile without actually proving.
  sorry


end total_chess_games_l350_350007


namespace one_is_sum_of_others_l350_350178

theorem one_is_sum_of_others {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : |a - b| ≥ c) (h2 : |b - c| ≥ a) (h3 : |c - a| ≥ b) :
    a = b + c ∨ b = a + c ∨ c = a + b :=
sorry

end one_is_sum_of_others_l350_350178


namespace simplify_sqrt_450_l350_350509

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350509


namespace arithmetic_sequence_value_of_n_l350_350729

theorem arithmetic_sequence_value_of_n :
  ∀ (a n d : ℕ), a = 1 → d = 3 → (a + (n - 1) * d = 2005) → n = 669 :=
by
  intros a n d h_a1 h_d ha_n
  sorry

end arithmetic_sequence_value_of_n_l350_350729


namespace total_votes_l350_350777

-- Conditions
variables (V : ℝ)
def candidate_votes := 0.31 * V
def rival_votes := 0.31 * V + 2451

-- Problem statement
theorem total_votes (h : candidate_votes V + rival_votes V = V) : V = 6450 :=
sorry

end total_votes_l350_350777


namespace simplify_sqrt_450_l350_350500

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350500


namespace ann_top_cost_l350_350060

noncomputable def cost_per_top (T : ℝ) := 75 = (5 * 7) + (2 * 10) + (4 * T)

theorem ann_top_cost : cost_per_top 5 :=
by {
  -- statement: prove cost per top given conditions
  sorry
}

end ann_top_cost_l350_350060


namespace problem_statement_l350_350011

noncomputable def system_has_three_solutions (a b c : ℝ) (n : ℕ) := 
 ∃ x : fin n → ℝ, ∀ i : fin n, 
  (a * x i^2 + b * x i + c = x (i + 1)) ∧ (x (n + 1) = x 1)

theorem problem_statement (n : ℕ) (hn : n ≥ 2) : system_has_three_solutions 2 0 (-1) n :=
sorry

end problem_statement_l350_350011


namespace simplify_sqrt_450_l350_350448

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350448


namespace find_exponent_l350_350932

theorem find_exponent :
  ∀ {m a : ℝ}, (∀ (x : ℝ), f x = m * x ^ a) → (f (1 / 4) = 1 / 2) → a = 1 / 2 :=
by
  intros m a h_f h_point
  -- The definitions and conditions without any proof
  sorry

end find_exponent_l350_350932


namespace sqrt_450_eq_15_sqrt_2_l350_350474

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350474


namespace sqrt_450_eq_15_sqrt_2_l350_350695

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350695


namespace simplify_sqrt_450_l350_350493

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350493


namespace sqrt_450_equals_15_sqrt_2_l350_350595

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350595


namespace failed_candidates_percentage_l350_350782

variable (total_candidates : ℕ) (girls : ℕ) (boys : ℕ) (passed_boys : ℕ) (passed_girls : ℕ)

def failed_percentage 
  (total_candidates = 2000) 
  (girls = 900) 
  (boys = total_candidates - girls)
  (passed_boys = boys * 28 / 100) 
  (passed_girls = girls * 32 / 100) 
  (failed_candidates = total_candidates - (passed_boys + passed_girls)) : ℝ :=
(failed_candidates : ℝ) / total_candidates * 100

theorem failed_candidates_percentage 
  : failed_percentage 2000 900 (2000 - 900) ((2000 - 900) * 28 / 100) (900 * 32 / 100) 1404 = 70.2 :=
by
  -- Proof would go here
  sorry

end failed_candidates_percentage_l350_350782


namespace sqrt_450_eq_15_sqrt_2_l350_350682

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350682


namespace range_of_g_l350_350275

open Real

theorem range_of_g (x : ℝ) (h₁ : x ∈ Icc (-1:ℝ) 1) : 
  let g := (arccos x)^2 - (arcsin x)^2 in 
  ∃ a b : ℝ, (g ∈ Icc a b ∧ a = -π^2 / 2 ∧ b = π^2 / 2) :=
by
  sorry

end range_of_g_l350_350275


namespace shaded_area_of_floor_l350_350793

theorem shaded_area_of_floor : 
  let floor_length := 12
  let floor_width := 15
  let tile_side := 2
  let radius := 1
  let num_tiles := (floor_length / tile_side) * (floor_width / tile_side)
  let tile_area := tile_side * tile_side
  let circle_area := Real.pi * radius * radius
  let shaded_area_per_tile := tile_area - circle_area
  let total_shaded_area := num_tiles * shaded_area_per_tile
  total_shaded_area = 180 - 45 * Real.pi := 
by {
  let floor_length := 12
  let floor_width := 15
  let tile_side := 2
  let radius := 1
  let num_tiles := (floor_length / tile_side) * (floor_width / tile_side)
  let tile_area := tile_side * tile_side
  let circle_area := Real.pi * radius * radius
  let shaded_area_per_tile := tile_area - circle_area
  let total_shaded_area := num_tiles * shaded_area_per_tile
  show total_shaded_area = 180 - 45 * Real.pi
  sorry
}

end shaded_area_of_floor_l350_350793


namespace simplify_sqrt_450_l350_350662

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350662


namespace pens_in_box_l350_350274

theorem pens_in_box (P : ℕ) (h1 : 20 * P - 8 * P - 3 * (20 * P - 8 * P) = 45) : P = 5 :=
by
  have h2 : 9 * P = 45 := by rw [mul_comm, mul_sub, sub_mul, mul_comm] at h1; exact h1
  have h3 : P = 5 := (nat.div_eq_of_eq_mul_left (by norm_num) h2.symm).symm
  exact h3

end pens_in_box_l350_350274


namespace sqrt_450_simplified_l350_350423

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350423


namespace basic_computer_price_l350_350738

theorem basic_computer_price (C P : ℝ)
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3)
  : C = 1500 :=
sorry

end basic_computer_price_l350_350738


namespace AB_dot_AF_eq_six_l350_350147

def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, Real.sqrt 3)
def F : ℝ × ℝ := (1, 0)

def vector (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem AB_dot_AF_eq_six :
  let AB := vector A B
  let AF := vector A F
  dot_product AB AF = 6 :=
by
  let AB := vector A B
  let AF := vector A F
  show dot_product AB AF = 6
  sorry

end AB_dot_AF_eq_six_l350_350147


namespace perimeter_of_quadrilateral_ABCD_l350_350250

theorem perimeter_of_quadrilateral_ABCD
  (A B C D E : Type)
  (right_triangle_AEB : ∠AEB = 90)
  (right_triangle_BCE : ∠BEC = 90)
  (right_triangle_CDE : ∠CED = 90)
  (angle_AEB_45 : ∠AEB = 45)
  (angle_BEC_45 : ∠BEC = 45)
  (angle_CED_45 : ∠CED = 45)
  (AE : ℝ)
  (AE_36 : AE = 36)
  : (36 + 36 + 36 + 72) = 180 := 
by
  sorry

end perimeter_of_quadrilateral_ABCD_l350_350250


namespace sqrt_450_eq_15_sqrt_2_l350_350551

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350551


namespace parallel_line_eq_chord_length_l350_350921

-- Define the equation of line l and the point A
def line_l (x y : ℝ) : Prop := x - y + 2 = 0
def point_A : (ℝ × ℝ) := (3, -1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 4 * y - 12 = 0

-- Prove that the equation of the line parallel to l passing through A(3, -1) is y = x - 4
theorem parallel_line_eq :
  ∀ (x y : ℝ),
  line_l x y → x = 3 → y = -1 →
  y = x - 4 :=
sorry

-- Prove the length of chord intercepted by line l on the circle is 2√2
theorem chord_length :
  ∀ (x y r : ℝ),
  radius_circle (x,y) r → radius (circle_eq x y) r → x=3 → y=-1 →  
  length_chord l r (circle_eq x y) = 2 * Real.sqrt 2 :=
sorry

end parallel_line_eq_chord_length_l350_350921


namespace simplify_sqrt_450_l350_350529

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350529


namespace sqrt_450_equals_15_sqrt_2_l350_350590

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350590


namespace horizontal_asymptote_is_2_l350_350844

-- Define the function y in terms of x
def y (x : ℝ) : ℝ := (8 * x^2 - 4) / (4 * x^2 + 8 * x + 3)

-- Define the statement to prove the horizontal asymptote b is 2 as x approaches infinity
theorem horizontal_asymptote_is_2 : ∃ b : ℝ, (∀ x : ℝ, x > 0 → (abs ((8 * x^2 - 4) / (4 * x^2 + 8 * x + 3) - b) < ε)) → b = 2 :=
sorry

end horizontal_asymptote_is_2_l350_350844


namespace sum_of_tens_l350_350227

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l350_350227


namespace relationships_among_abc_l350_350132

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry -- Derivative of f

lemma function_symmetric : ∀ x, f(x - 1) = f(2 - x) := sorry
lemma f_inequality (x : ℝ) (hx : x < 0) : f(x) + x * f'(x) < 0 := sorry

def g (x : ℝ) : ℝ := x * f(x)

noncomputable def a := (1 / 2) * f(log 2 (sqrt 2))
noncomputable def b := (log 2) * f(log 2)
noncomputable def c := 2 * f(log (1 / 2) (1 / 4))

theorem relationships_among_abc :
  a > b ∧ b > c :=
by
  sorry

end relationships_among_abc_l350_350132


namespace jeff_mean_correct_l350_350988

def jeff_scores : List ℤ := [90, 93, 85, 97, 92, 88]

def arithmetic_mean (scores : List ℤ) : ℚ :=
  let sum := scores.foldl (· + ·) 0
  let count := scores.length
  sum / count

theorem jeff_mean_correct :
  arithmetic_mean jeff_scores = 90.8333 := by
  sorry

end jeff_mean_correct_l350_350988


namespace sqrt_450_eq_15_sqrt_2_l350_350559

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350559


namespace yellow_ball_impossible_at_least_one_red_ball_probability_of_red_ball_find_x_l350_350976

-- Given conditions
def initial_white_balls := 8
def initial_red_balls := 12
def total_balls := initial_white_balls + initial_red_balls

-- Question 1(a): Drawing a yellow ball is impossible
theorem yellow_ball_impossible (total_balls : ℕ) : false :=
by 
  sorry -- The proof would go here

-- Question 1(b): Probability of drawing at least one red ball
theorem at_least_one_red_ball (total_balls : ℕ) (drawn_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) (total_balls = white_balls + red_balls) : ℝ :=
by
  have h : white_balls < drawn_balls → 1 = 1 :=
  by
    sorry -- The proof would go here
  h

-- Question 2: Probability of drawing a red ball at random
theorem probability_of_red_ball (red_balls white_balls : ℕ) : ℝ :=
  red_balls / (red_balls + white_balls)

-- Question 3: Finding x given the probability of drawing a white ball is 4/5
theorem find_x (initial_white_balls initial_red_balls : ℕ) (draw_white_prob : ℝ) : ℕ :=
by
  let x := initial_white_balls + 8 -- filter x from the probability 4/5 assumption
  sorry -- The proof would go here

end yellow_ball_impossible_at_least_one_red_ball_probability_of_red_ball_find_x_l350_350976


namespace find_purely_imaginary_z_l350_350903

theorem find_purely_imaginary_z (z : ℂ) (hz : Im z ≠ 0) (h : (z + 2) / (1 - complex.I) ∈ ℝ) : z = -2 * complex.I :=
by
  sorry

end find_purely_imaginary_z_l350_350903


namespace katie_pink_marbles_l350_350992

-- Define variables for the problem
variables (P O R : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  O = P - 9 ∧
  R = 4 * (P - 9) ∧
  P + O + R = 33

-- Desired result
def result : Prop :=
  P = 13

-- Proof statement
theorem katie_pink_marbles : conditions P O R → result P :=
by
  intros h
  sorry

end katie_pink_marbles_l350_350992


namespace simplify_sqrt_450_l350_350407

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350407


namespace find_n_l350_350116

theorem find_n (x y n : ℕ) (h1 : x = 3) (h2 : y = 27) (h3 : n^(n / (2 + x)) = y): 
  n = 15 :=
by {
  have h4 : 3^3 = 27, by norm_num,
  sorry
}

end find_n_l350_350116


namespace sqrt_450_simplified_l350_350417

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350417


namespace problem_I_problem_II_l350_350020

-- Part (I)
theorem problem_I (
  (λ : ℝ) (x y x' y' : ℝ),
  (h1 : x^2 + y^2 = 4)
  (h2 : x' = λ * x)
  (h3 : y' = 3 * y)
  (h4 : λ > 0 )
  (h5 : e = 4 / 5)  -- eccentricity of ellipse
) : λ = 5 := 
sorry

-- Part (II)
theorem problem_II (
  (ρ θ : ℝ)
  (A : ℝ × ℝ) (P : ℝ × ℝ),
  (h1 : A = (2, 0))
  (h2 : ρ = (2 + 2 * cos θ) / (sin θ)^2)
  (h3 : (x y : ℝ), P = (x, y) ∧ x ≥ -1
  )
) : dist P A = 2 * sqrt 2 := 
sorry

end problem_I_problem_II_l350_350020


namespace prime_sum_probability_correct_l350_350104

def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_sum_probability : ℚ :=
  let pairs := List.filter (λ (p : ℕ × ℕ), is_prime (p.1 + p.2)) 
                (List.bind first_twelve_primes (fun x => 
                  first_twelve_primes.map (λ y => (x, y))))
  let favorable := pairs.filter (λ p, p.1 ≠ p.2)
  favorable.length /. (first_twelve_primes.length * (first_twelve_primes.length - 1) / 2)

theorem prime_sum_probability_correct :
  prime_sum_probability = 5 / 66 :=
by
  sorry

end prime_sum_probability_correct_l350_350104


namespace sum_of_areas_of_tangent_circles_l350_350740

-- Conditions: Define the side lengths of the triangle
def side_a : ℝ := 3
def side_b : ℝ := 4
def side_c : ℝ := 5

-- Define the radii of the circles
variables (r1 r2 r3 : ℝ)

-- Constraints based on the tangency condition
axiom radii_sum_1 : r1 + r2 = side_a
axiom radii_sum_2 : r1 + r3 = side_b
axiom radii_sum_3 : r2 + r3 = side_c

-- Calculate area of each circle
def area_circle (r : ℝ) : ℝ := Real.pi * r^2

-- Problem statement
theorem sum_of_areas_of_tangent_circles :
  ∃ (r1 r2 r3 : ℝ), (r1 + r2 = side_a) ∧ (r1 + r3 = side_b) ∧ (r2 + r3 = side_c) ∧
  (area_circle r1 + area_circle r2 + area_circle r3 = 14 * Real.pi) :=
by
  sorry

end sum_of_areas_of_tangent_circles_l350_350740


namespace smallest_integer_x_l350_350765

theorem smallest_integer_x (x : ℤ) (h : ((x^2 - 4*x + 13) / (x - 5)) ∈ ℤ) : x = 6 :=
sorry

end smallest_integer_x_l350_350765


namespace direction_vector_of_line_l350_350842

theorem direction_vector_of_line 
    {a b : ℤ}
    (h1 : a > 0) 
    (h2 : Int.gcd a (Int.natAbs b) = 1) 
    (M : Matrix (Fin 2) (Fin 2) ℚ) 
    (hM : M = ![![3/5, -4/5], [-4/5, -3/5]]) 
    (v : Vector ℚ 2)
    (hv : v = ![a, b])
    (h_eq : M.mulVec v = v) :
    v = ![2, -1] :=
sorry


end direction_vector_of_line_l350_350842


namespace range_of_a_l350_350933

theorem range_of_a (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5 * x + 15 / 2 * a <= 0) -> a > 5 / 6 :=
by
  sorry

end range_of_a_l350_350933


namespace sqrt_simplify_l350_350366

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350366


namespace trim_area_dodecagon_pie_l350_350876

theorem trim_area_dodecagon_pie :
  let d := 8 -- diameter of the pie
  let r := d / 2 -- radius of the pie
  let A_circle := π * r^2 -- area of the circle
  let A_dodecagon := 3 * r^2 -- area of the dodecagon
  let A_trimmed := A_circle - A_dodecagon -- area to be trimmed
  let a := 16 -- coefficient of π in A_trimmed
  let b := 48 -- constant term in A_trimmed
  a + b = 64 := 
by 
  sorry

end trim_area_dodecagon_pie_l350_350876


namespace simplify_sqrt_450_l350_350381

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350381


namespace min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350327

/- Part (a): Minimum amount the customer will pay -/
theorem min_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4000 := 
sorry

/- Part (b): Maximum amount the customer will pay -/
theorem max_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4900 := 
sorry

/- Part (c): Expected value the customer will pay -/
theorem expected_amount_to_pay :
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4583.33 := 
sorry

end min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350327


namespace sqrt_450_simplified_l350_350626

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350626


namespace same_color_probability_sum_l350_350966

theorem same_color_probability_sum :
  let red_total := 15
  let blue_total := 15
  let total_candies := red_total + blue_total

  -- Probability computations
  let prob_terry_three_red := (red_total * (red_total - 1) * (red_total - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let prob_mary_two_remaining_red := ((red_total - 3) * (red_total - 4)) / ((total_candies - 3) * (total_candies - 4))
  let combined_red := prob_terry_three_red * prob_mary_two_remaining_red

  let prob_terry_three_blue := (blue_total * (blue_total - 1) * (blue_total - 2)) / (total_candies * (total_candies - 1) * (total_candies - 2))
  let prob_mary_two_remaining_blue := ((blue_total - 3) * (blue_total - 4)) / ((total_candies - 3) * (total_candies - 4))
  let combined_blue := prob_terry_three_blue * prob_mary_two_remaining_blue

  let total_combined_prob := combined_red + combined_blue

  let fraction := (8008 : ℚ) / 142221

in total_combined_prob = fraction → 8008 + 142221 = 150229 := 
by sorry

end same_color_probability_sum_l350_350966


namespace inscribed_triangle_area_l350_350862

theorem inscribed_triangle_area (a b : ℝ) (α : ℝ) (hab : a * b * sin α = 3 + sqrt 5):
  let S := a * b * sin α in
  (S - 2)^2 / S = 2 → 
  ∃ T : ℝ, T = sqrt 5 :=
by
  -- The proof will go here
  sorry

end inscribed_triangle_area_l350_350862


namespace sqrt_450_eq_15_sqrt_2_l350_350458

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350458


namespace jessica_current_age_l350_350269

theorem jessica_current_age : 
  ∃ J M_d M_c : ℕ, 
    J = (M_d / 2) ∧ 
    M_d = M_c - 10 ∧ 
    M_c = 70 ∧ 
    J + 10 = 40 := 
sorry

end jessica_current_age_l350_350269


namespace denominator_of_fractions_l350_350231

theorem denominator_of_fractions (y a : ℝ) (hy : y > 0) 
  (h : (2 * y) / a + (3 * y) / a = 0.5 * y) : a = 10 :=
by
  sorry

end denominator_of_fractions_l350_350231


namespace locusts_left_right_equivalence_l350_350741

/-- Given 2019 locusts on a straight line where each locust can jump over another locust to maintain the same distance, 
if it is possible to achieve a configuration where the distance between any two locusts is exactly 1 unit by only jumping to the right, 
then it is also possible to achieve this configuration by only jumping to the left. -/
theorem locusts_left_right_equivalence (locusts : Fin 2019 → ℝ) :
  (∃ f : ℕ → Fin 2019 → ℝ, ∀ n a b, ∥f n a - f n b∥ = 1 → ∥f (n+1) a - f (n+1) b∥ = 1  → 
    (∀ a b, ∥f ⟨2019, _⟩ a - f ⟨2019, _⟩ b∥ = 1)) →
  (∃ g : ℕ → Fin 2019 → ℝ, ∀ n a b, ∥g n a - g n b∥ = 1 → ∥g (n+1) a - g (n+1) b∥ = 1  → 
    (∀ a b, ∥g ⟨2019, _⟩ a - g ⟨2019, _⟩ b∥ = 1)) :=
by sorry

end locusts_left_right_equivalence_l350_350741


namespace simplify_expression_l350_350703

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end simplify_expression_l350_350703


namespace num_terms_100_pow_10_as_sum_of_tens_l350_350214

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l350_350214


namespace ellipse_properties_l350_350900

theorem ellipse_properties
  (c : ℝ)
  (a b : ℝ)
  (F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (-c, 0))
  (hF2 : F2 = (c, 0))
  (hLine : ∀ x, y = (2 * real.sqrt 3 / 3) * x)
  (hAB_dist : ∥(3:ℝVector) - (√3: ℝVector)∥ = 2 * real.sqrt 7)
  (hArea : ∃ (A B : ℝ × ℝ), |A - B| = 2 * real.sqrt 7 ∧
    (A.1 * B.2 + B.1 * F2.2 + F2.1 * F1.2 + F1.1 * A.2 - A.1 * F2.2 - F2.1 * B.2 - B.1 * F1.2 - F1.1 * A.2)
    = 4 * real.sqrt 3) :
  c = real.sqrt 3 ∧ (∀ x y, x^2 / a^2 + y^2 / b^2 = 1) :=
begin
  -- Proof would go here
  sorry
end

end ellipse_properties_l350_350900


namespace range_of_a_l350_350913

theorem range_of_a (a : ℝ) : 
  (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ f(x) > 0) ∧ (∃ x, -1 ≤ x ∧ x ≤ 1 ∧ f(x) < 0) →
  -1 < a ∧ a < (-1 / 3) :=
by 
  let f := λ x : ℝ, a * x + 2 * a + 1
  sorry

end range_of_a_l350_350913


namespace part_a_i_part_a_ii_part_a_iii_part_b_l350_350785

def tank_a_dimensions := (10 : ℝ, 8 : ℝ, 6 : ℝ)
def tank_b_dimensions := (5 : ℝ, 9 : ℝ, 8 : ℝ)
def tank_a_drain_rate := 4 : ℝ
def tank_b_fill_rate := 4 : ℝ

theorem part_a_i : 
  (∀ (t : ℝ), tank_b_fill_rate * t = 360 / 3 → t = 30) :=
sorry

theorem part_a_ii : 
  (∀ (t : ℝ), tank_b_fill_rate * t = 360 ∧ tank_a_drain_rate * t = (10 * 8 * 6 - tank_a_drain_rate * t) / (10 * 8) → t = 90 ∧ tank_a_drain_rate * t / (10 * 8) = 1.5) :=
sorry

theorem part_a_iii : 
  (∀ (d : ℝ), 80 * d + 45 * d = 480 → d = 3.84) :=
sorry

def tank_c_dimensions := (31 : ℝ, 4 : ℝ, 4 : ℝ)
def tank_d_base_side_length := 20 : ℝ
def tank_d_height := 10 : ℝ
def tank_d_fill_rate := 1 : ℝ
def tank_c_drain_rate := 2 : ℝ

theorem part_b : 
  (∀ (t : ℝ), tank_c_dimensions.1 * tank_c_dimensions.2 * tank_c_dimensions.3 - tank_c_drain_rate * (t - 2) = (1 / 3) * ((tank_d_base_side_length / tank_d_height) * d)^2 * d ∧ t = (500 / 3) → d = 5) :=
sorry

end part_a_i_part_a_ii_part_a_iii_part_b_l350_350785


namespace simplify_sqrt_450_l350_350508

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350508


namespace polygonal_pyramid_faces_l350_350039

/-- A polygonal pyramid is a three-dimensional solid. Its base is a regular polygon. Each of the vertices of the polygonal base is connected to a single point, called the apex. The sum of the number of edges and the number of vertices of a particular polygonal pyramid is 1915. This theorem states that the number of faces of this pyramid is 639. -/
theorem polygonal_pyramid_faces (n : ℕ) (hn : 2 * n + (n + 1) = 1915) : n + 1 = 639 :=
by
  sorry

end polygonal_pyramid_faces_l350_350039


namespace problem_statement_l350_350279

noncomputable def A := {a : ℤ | a ∈ {1, 2, 3}}  -- example set instance with at least 3 integers for demonstration.

axiom P : ℤ → ℤ
axiom M : ℤ
axiom m : ℤ

axiom maxA : M = 3 -- example maximum element for demonstration
axiom minA : m = 1 -- example minimum element for demonstration
axiom elements_in_A : ∀ a, a ∈ A → 1 < P(a) ∧ P(a) < 3  -- example set property for demonstration

axiom P_property1 : ∀ a, a ∈ A → (1 < P(a)) ∧ (P(a) < 3)
axiom P_property2 : ∀ a, a ∈ A ∧ a ≠ 1 ∧ a ≠ 3 → P(1) < P(a)

theorem problem_statement : 
  (∃ b c : ℤ, ∀ x ∈ A, P(x) + x^2 + b * x + c = 0) ∧ (A.card < 6) := 
by 
  sorry

end problem_statement_l350_350279


namespace sqrt_450_eq_15_sqrt_2_l350_350479

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350479


namespace sqrt_450_eq_15_sqrt_2_l350_350567

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350567


namespace trains_crossing_time_l350_350008

theorem trains_crossing_time :
  ∀ (length_of_train time1 time2 : ℝ), 
  length_of_train = 120 ∧ time1 = 5 ∧ time2 = 15 →
  time1 ≠ 0 ∧ time2 ≠ 0 →
  let speed1 := length_of_train / time1 in
  let speed2 := length_of_train / time2 in
  let relative_speed := speed1 + speed2 in
  let total_distance := 2 * length_of_train in
  let crossing_time := total_distance / relative_speed in
  crossing_time = 7.5 :=
by
  intros length_of_train time1 time2 h1 h2
  cases h1 with h_len h_times
  cases h_times with h_time1 h_time2
  have h_length := h_len
  have h_time1_eq := h_time1
  have h_time2_eq := h_time2
  have calc_speed1 : speed1 = length_of_train / time1 := rfl
  have calc_speed2 : speed2 = length_of_train / time2 := rfl
  have calc_relative_speed : relative_speed = speed1 + speed2 := rfl
  have calc_total_distance : total_distance = 2 * length_of_train := rfl
  have calc_crossing_time : crossing_time = total_distance / relative_speed := rfl
  sorry -- proof to be filled in

end trains_crossing_time_l350_350008


namespace simplify_sqrt_450_l350_350389

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350389


namespace sqrt_simplify_l350_350360

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350360


namespace sqrt_450_equals_15_sqrt_2_l350_350608

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350608


namespace num_pos_pairs_l350_350176

theorem num_pos_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 + 3 * n < 40) :
  ∃ k : ℕ, k = 45 :=
by {
  -- Additional setup and configuration if needed
  -- ...
  sorry
}

end num_pos_pairs_l350_350176


namespace simplify_sqrt_450_l350_350491

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350491


namespace even_odd_decomposition_exp_l350_350195

variable (f g : ℝ → ℝ)

-- Conditions
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x
def decomposition (f g : ℝ → ℝ) := ∀ x, f x + g x = Real.exp x

-- Main statement to prove
theorem even_odd_decomposition_exp (hf : is_even f) (hg : is_odd g) (hfg : decomposition f g) :
  f (Real.log 2) + g (Real.log (1 / 2)) = 1 / 2 := 
sorry

end even_odd_decomposition_exp_l350_350195


namespace sqrt_450_eq_15_sqrt_2_l350_350465

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350465


namespace correct_derivative_option_B_l350_350852

theorem correct_derivative_option_B (x : ℝ) : (derivative (λ x, (2 * (x ^ (3 / 2)))) x) = (3 * sqrt x) :=
by
  sorry

end correct_derivative_option_B_l350_350852


namespace value_of_a1_a2_a3_a4_a5_l350_350298

-- The hypothesis of the polynomial expansion
noncomputable def polynomial_expansion (x : ℝ) : ℝ := (2 - x) ^ 5
def a_0 := polynomial_expansion 0
def a_1 := (-31 : ℝ) - (a_0 - 1)
def a_2 := 0
def a_3 := 0
def a_4 := 0
def a_5 := 0

-- The definition of the polynomial as stated in the problem
def polynomial : ℝ → ℝ := λ x, a_0 + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5

-- Prove that the polynomial evaluated at (2 - x) ^ 5 matches the hypothesis
theorem value_of_a1_a2_a3_a4_a5 : (a_1 + a_2 + a_3 + a_4 + a_5) = -31 :=
by {
  sorry
}

end value_of_a1_a2_a3_a4_a5_l350_350298


namespace triangle_problem_l350_350959

-- Define the given conditions
def side_a : ℝ := 1
def side_b : ℝ := sqrt 3
def angle_C : ℝ := π / 6 -- 30 degrees in radians

-- Define the expected results
def expected_side_c : ℝ := 1
def expected_area_S : ℝ := sqrt 3 / 4

-- Problem statement
theorem triangle_problem :
  ∃ (c : ℝ) (S : ℝ), 
    (c = side_a ^ 2 + side_b ^ 2 - 2 * side_a * side_b * cos angle_C) ∧ 
    (S = 1 / 2 * side_a * side_b * sin angle_C) ∧ 
    (c = expected_side_c) ∧ 
    (S = expected_area_S) :=
by
  -- Issue exists with translation of the problem into a valid theorem statement.
  sorry

end triangle_problem_l350_350959


namespace areas_not_all_equal_l350_350052

-- Definitions for the points and properties in the problem
variables {ABC : Type*} [triangle ABC] (A B C M P Q : ABC)
variables {angle_A : ∠A = 90°} 
variables {PointOnBC : M ∈ Segment B C} 
variables {FootP : P = FootPerpendicular M A B}
variables {FootQ : Q = FootPerpendicular M A C}

-- Theorem statement
theorem areas_not_all_equal : 
  ¬ (area (triangle B P M) = area (triangle M Q C) ∧ area (triangle M Q C) = area (quadrilateral A Q M P)) :=
sorry

end areas_not_all_equal_l350_350052


namespace tire_mileage_l350_350025

theorem tire_mileage (total_miles_driven : ℕ) (x : ℕ) (spare_tire_miles : ℕ):
  total_miles_driven = 40000 →
  spare_tire_miles = 2 * x →
  4 * x + spare_tire_miles = total_miles_driven →
  x = 6667 := 
by
  intros h_total h_spare h_eq
  sorry

end tire_mileage_l350_350025


namespace simplify_sqrt_450_l350_350503

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350503


namespace sqrt_450_eq_15_sqrt_2_l350_350687

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350687


namespace sqrt_450_eq_15_sqrt_2_l350_350679

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350679


namespace cos_angle_POQ_l350_350130

theorem cos_angle_POQ (hxP : ℝ) (hyP : ℝ) (hxQ : ℝ) (hyQ : ℝ)
    (hP_unit_circle : hxP^2 + hyP^2 = 1)
    (hQ_unit_circle : hxQ^2 + hyQ^2 = 1)
    (hP_first_quadrant : hxP > 0 ∧ hyP > 0)
    (hQ_fourth_quadrant : hxQ > 0 ∧ hyQ < 0)
    (hyP_val : hyP = 4 / 5)
    (hxQ_val : hxQ = 5 / 13) : 
  cos (atan2 hyP hxP - atan2 hyQ hxQ) = -33 / 65 := 
by
  sorry

end cos_angle_POQ_l350_350130


namespace purely_imaginary_real_part_zero_l350_350918

theorem purely_imaginary_real_part_zero (m : ℂ) (z : ℂ) (hz : z = (m^2 - m) + m * complex.I) (hz_pure_imag : z.im ≠ 0 ∧ z.re = 0) : m = 1 :=
by
  sorry

end purely_imaginary_real_part_zero_l350_350918


namespace simplify_sqrt_450_l350_350653
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350653


namespace sqrt_450_eq_15_sqrt_2_l350_350561

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350561


namespace xiaoyu_flight_distance_l350_350742

theorem xiaoyu_flight_distance :
  ∀ (x y d : ℝ), 
    x = 20 ∧ y = 15 ∧ (x / 2 - 6)^2 + y^2 = d^2 → 
    d = 17 :=
by 
  intros x y d h
  rw [and_assoc, and_comm] at h
  cases h with left_eq right_eq
  cases left_eq with h1 h2
  cases h2 with h3 h4
  simp at *
  sorry

end xiaoyu_flight_distance_l350_350742


namespace cartesian_equation_of_polar_shortest_distance_point_l350_350981

-- Definition of the conditions:
def polar_equation := ∀ θ ∈ set.Ico 0 (2 * Real.pi), (ρ θ = 2 * Real.sin θ)
def parametric_line (t : ℝ) : ℝ × ℝ := (√3 * t + √3, -3 * t + 2)
def cartesian_line (x y : ℝ) := y = -√3 * x + 5

-- Prove the Cartesian equation of a given polar curve
theorem cartesian_equation_of_polar (θ : ℝ) : (ρ θ = 2 * Real.sin θ) ∧ θ ∈ set.Ico 0 (2 * Real.pi) →
  ∃ (x y : ℝ), (x^2 + y^2 - 2 * y = 0) ∧ 
               (PolarCoord x y = 2 * Real.sin θ) :=
sorry

-- Prove the coordinates of point D minimizing distance to a line
theorem shortest_distance_point (x y : ℝ) : 
  x^2 + (y - 1)^2 = 1 ∧ 
  ∀ (D : ℝ × ℝ), D ∈ {p : ℝ × ℝ | (PolarCoord p.1 p.2) = 2 * Real.sin θ} →
   ∃ d, 
     (forall (d1 d2: ℝ), (((d1 - D.fst)^2 + (d2 - D.snd)^2) < (d^2)) -> cartesian_line d1 d2) ∧
     (x, y) =  ⟨ √(3/2), 3/2 ⟩ :=
sorry

-- Note: polar coordinates are derived using a specific domain knowledge abstraction in Lean

end cartesian_equation_of_polar_shortest_distance_point_l350_350981


namespace simplify_sqrt_450_l350_350408

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350408


namespace scale_division_l350_350046

theorem scale_division (total_feet : ℕ) (total_inches : ℕ) (part_length : ℕ) : 
  total_feet = 10 → total_inches = 5 → part_length = 25 → 
  ((total_feet * 12 + total_inches) / part_length) = 5 := 
by 
  intro h_feet h_inches h_part_length
  have total_length := (total_feet * 12 + total_inches)
  rw [h_feet, h_inches, Nat.mul_add] at total_length
  sorry

end scale_division_l350_350046


namespace problem1_problem2_problem3_l350_350095

-- Definition of sets and functions
noncomputable def A : set (ℝ → ℝ) := 
  {f | ∃ k : ℝ, ∀ x : ℝ, 0 < x → f x < k}

noncomputable def M (n : ℕ) : set (ℝ → ℝ) := 
  {f | ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f x1 / x1^n < f x2 / x2^n}

-- Part (1)
theorem problem1 (f : ℝ → ℝ) (h : ℝ) (H : f = λ x, x^3 + h)
  (H1 : f ∈ M 1) : h ≤ 0 :=
sorry

-- Part (2)
theorem problem2 (f : ℝ → ℝ) (a b d : ℝ) (H : f ∈ M 1) 
  (Ha : 0 < a) (Hb : 0 < b) (Hab : a < b) (Hf : f a = d ∧ f b = d) :
  d < 0 ∧ f (a + b) > 2 * d :=
sorry

-- Part (3)
theorem problem3 (f : ℝ → ℝ) (m : ℝ) 
  (H : f ∈ A ∩ M 2) : m ∈ [0, +∞) :=
sorry

end problem1_problem2_problem3_l350_350095


namespace P_symm_l350_350047

def P0 (x y z : ℕ) : ℕ := 1

def P (m : ℕ) (x y z : ℕ) : ℕ :=
  Nat.recOn m 1 (λ m Pm, (x + z) * (y + z) * Pm (x) (y) (z + 1) - z * z * Pm (x) (y) (z))

theorem P_symm (m : ℕ) (x y z : ℕ) : 
  P m x y z = P m x z y ∧ P m x y z = P m y x z :=
by
  induction m with 
  | zero => 
    simp [P0]
  | succ m ih =>
    cases ih with
    | ⟨ih1, ih2⟩ => sorry

end P_symm_l350_350047


namespace train_speed_l350_350051

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) (total_distance : ℝ) 
    (speed_mps : ℝ) (speed_kmph : ℝ) 
    (h1 : train_length = 360) 
    (h2 : bridge_length = 140) 
    (h3 : time = 34.61538461538461) 
    (h4 : total_distance = train_length + bridge_length) 
    (h5 : speed_mps = total_distance / time) 
    (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 52 := 
by 
  sorry

end train_speed_l350_350051


namespace general_term_formula_sum_first_n_terms_l350_350139

-- Define the sequences and their properties
noncomputable def a (n : ℕ) : ℕ := 2 * n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in finset.range n, 1 / (S (i + 1))

-- Given condition: a_3 + S_3 = 18
axiom a3_S3_condition : a 3 + S 3 = 18

-- Prove the general term formula for the sequence {a_n}
theorem general_term_formula (n : ℕ) : a n = 2 * n :=
by sorry

-- Prove the sum of the first n terms for the sequence {1 / S_n}
theorem sum_first_n_terms (n : ℕ) : T n = (n : ℝ) / (n + 1 : ℝ) := 
by sorry

end general_term_formula_sum_first_n_terms_l350_350139


namespace total_biking_distance_l350_350736

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end total_biking_distance_l350_350736


namespace log_abs_monotonicity_l350_350194

theorem log_abs_monotonicity (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : ∀ x ∈ Ioo (1 : ℝ) 2, log a (2 - x) < log a (2 - (x+ε)) (ε > 0)) :
  ∀ x1 x2 ∈ Ioi 2, x1 < x2 → log a (x1 - 2) > log a (x2 - 2) :=
by
  sorry

end log_abs_monotonicity_l350_350194


namespace sqrt_450_simplified_l350_350414

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350414


namespace additional_money_needed_l350_350310

/-- Mrs. Smith needs to calculate the additional money required after a discount -/
theorem additional_money_needed
  (initial_amount : ℝ) (ratio_more : ℝ) (discount_rate : ℝ) (final_amount_needed : ℝ) (additional_needed : ℝ)
  (h_initial : initial_amount = 500)
  (h_ratio : ratio_more = 2/5)
  (h_discount : discount_rate = 15/100)
  (h_total_needed : final_amount_needed = initial_amount * (1 + ratio_more) * (1 - discount_rate))
  (h_additional : additional_needed = final_amount_needed - initial_amount) :
  additional_needed = 95 :=
by 
  sorry

end additional_money_needed_l350_350310


namespace sum_of_fractions_l350_350072

theorem sum_of_fractions :
  (3 / 12 : Real) + (6 / 120) + (9 / 1200) = 0.3075 :=
by
  sorry

end sum_of_fractions_l350_350072


namespace sqrt_450_eq_15_sqrt_2_l350_350461

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350461


namespace increasing_sequence_range_l350_350912

theorem increasing_sequence_range (a : ℝ) : 
  (∀ n : ℕ, a_{n+1} > a_n → f(n) : \{a_n\} ) → (3 < a) :=
by
  let f : ℝ → ℝ
      | x ≤ 1  => (2 * a - 1) * x + 4
      | x > 1  => a ^ x
  let a_n := λ n, f n
  have h_increasing : ∀ n, a_n (n + 1) > a_n n, sorry
  show 3 < a, sorry

end increasing_sequence_range_l350_350912


namespace simplify_sqrt_450_l350_350378

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350378


namespace triangle_proof_l350_350985

noncomputable def AC := sorry
noncomputable def BC := sorry
noncomputable def CD := sorry
noncomputable def CK := sorry
noncomputable def S := sorry

theorem triangle_proof
  (h1: ∠C = 90) 
  (h2: CD ⊥ AB) 
  (h3: D is_foot_of CD AB)
  (h4: P = incenter (triangle ADC))
  (h5: Q = incenter (triangle BDC))
  (h6: K = intersection (line PQ) (line CD))
  (h7: area (triangle ABC) = S) : 
  1 / CK ^ 2 - 1 / CD ^ 2 = 1 / S := sorry

end triangle_proof_l350_350985


namespace sqrt_450_eq_15_sqrt_2_l350_350555

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350555


namespace remainder_of_floor_sum_l350_350854

theorem remainder_of_floor_sum (i : ℕ) (h : i ≤ 2015) :
  (∑ k in Finset.range 2016, Int.floor ((2^k : ℚ) / 25)) % 100 = 2 := by
  sorry

end remainder_of_floor_sum_l350_350854


namespace sqrt_450_eq_15_sqrt_2_l350_350588

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350588


namespace find_simple_interest_rate_l350_350069

theorem find_simple_interest_rate (P A T SI R : ℝ)
  (hP : P = 750)
  (hA : A = 1125)
  (hT : T = 5)
  (hSI : SI = A - P)
  (hSI_def : SI = (P * R * T) / 100) : R = 10 :=
by
  -- Proof would go here
  sorry

end find_simple_interest_rate_l350_350069


namespace simplify_sqrt_450_l350_350514

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350514


namespace select_students_l350_350056

theorem select_students (males females : ℕ) (total : ℕ) (to_select : ℕ)
  (hmales : males = 5)
  (hfemales : females = 3)
  (htotal : total = males + females)
  (hto_select : to_select = 5)
  : (Nat.choose 8 5) - (Nat.choose 5 5) = 55 :=
by
  -- Introduce all relevant values and conditions
  rw [hmales, hfemales, htotal, hto_select]
  -- Simplify the calculation using Nat.choose
  have h1 : Nat.choose 8 5 = 56 := by simp
  have h2 : Nat.choose 5 5 = 1 := by simp
  -- Combine results to show the final answer is 55
  rw [h1, h2]
  exact rfl

end select_students_l350_350056


namespace x729_minus_inverse_l350_350144

theorem x729_minus_inverse (x : ℂ) (h : x - x⁻¹ = 2 * Complex.I) : x ^ 729 - x⁻¹ ^ 729 = 2 * Complex.I := 
by 
  sorry

end x729_minus_inverse_l350_350144


namespace simplify_sqrt_450_l350_350668

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350668


namespace simplify_sqrt_450_l350_350664

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350664


namespace length_QB_l350_350963

-- Lean statement for the given proof problem
theorem length_QB {D B C N Q : Type} {y : ℝ} (h_mid_arc: midpoint_arc D B C N) (h_perp: is_perpendicular N Q D B) 
  (h_chord: length_chord D B = y) (h_DQ: length D Q = y + 2) : 
  length Q B = y + 2 := 
sorry

-- Definitions used in Lean 4
-- Note: These are intended to formalize the conditions in Lean.
--       The exact formal definitions would depend on library implementation.
#check h_mid_arc
#check h_perp
#check h_chord
#check h_DQ

end length_QB_l350_350963


namespace four_digit_number_correct_l350_350030

variable (a b c d : ℕ)

theorem four_digit_number_correct :
  let four_digit_number := a * 1000 + b * 100 + c * 10 + d in
  four_digit_number = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end four_digit_number_correct_l350_350030


namespace initial_players_was_eleven_l350_350743

def initial_players (remaining_lives : ℕ) (lives_per_player : ℕ) (players_quit : ℕ) : ℕ :=
  remaining_players remaining_lives lives_per_player + players_quit

def remaining_players (remaining_lives : ℕ) (lives_per_player : ℕ) : ℕ :=
  remaining_lives / lives_per_player

theorem initial_players_was_eleven (remaining_lives : ℕ) (lives_per_player : ℕ) (players_quit : ℕ) 
  (h1 : remaining_lives = 30) (h2 : lives_per_player = 5) (h3 : players_quit = 5) :
  initial_players remaining_lives lives_per_player players_quit = 11 :=
by
  rw [h1, h2, h3]
  unfold initial_players remaining_players
  norm_num

end initial_players_was_eleven_l350_350743


namespace distance_focus_directrix_is_quarter_l350_350712

-- Define the equation of the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define what it means for distance from the focus to the directrix.
def distance_from_focus_to_directrix (p : ℝ) :=
  let standard_form_x := x^2 = (1 / 2) * (parabola x) in
  let distance := p = 1 / 4 in
  distance

-- Prove that given the parabola equation, the distance is 1/4
theorem distance_focus_directrix_is_quarter :
  distance_from_focus_to_directrix (1 / 4) :=
by
  sorry

end distance_focus_directrix_is_quarter_l350_350712


namespace sqrt_450_eq_15_sqrt_2_l350_350554

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350554


namespace glove_pair_probability_l350_350799

/-- 
A box contains 6 pairs of black gloves (i.e., 12 black gloves) and 4 pairs of beige gloves (i.e., 8 beige gloves).
We need to prove that the probability of drawing a matching pair of gloves is 47/95.
-/
theorem glove_pair_probability : 
  let total_gloves := 20
  let black_gloves := 12
  let beige_gloves := 8
  let P1_black := (black_gloves / total_gloves) * ((black_gloves - 1) / (total_gloves - 1))
  let P2_beige := (beige_gloves / total_gloves) * ((beige_gloves - 1) / (total_gloves - 1))
  let total_probability := P1_black + P2_beige
  total_probability = 47 / 95 :=
sorry

end glove_pair_probability_l350_350799


namespace jessica_current_age_l350_350266

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age_l350_350266


namespace simplify_sqrt_450_l350_350386

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350386


namespace index_cards_per_pack_l350_350077

-- Definitions of the conditions
def students_per_period := 30
def periods_per_day := 6
def index_cards_per_student := 10
def total_spent := 108
def pack_cost := 3

-- Helper Definitions
def total_students := periods_per_day * students_per_period
def total_index_cards_needed := total_students * index_cards_per_student
def packs_bought := total_spent / pack_cost

-- Theorem to prove
theorem index_cards_per_pack :
  total_index_cards_needed / packs_bought = 50 := by
  sorry

end index_cards_per_pack_l350_350077


namespace simplify_sqrt_450_l350_350541

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350541


namespace tangent_circle_l350_350786

-- Let ABCDE be a pentagon inscribed in a circle Omega.
-- Line parallel to BC intersects AB and AC at points S and T, respectively.
-- Let X be the intersection of the line BE and DS, and Y the intersection of the line CE and DT.
-- Prove that if AD is tangent to the circle (DXY), then AE is tangent to the circle (EXY).

noncomputable theory
open_locale classical

variables {Ω : Type*} [circle Ω]
variables (A B C D E S T X Y : Ω)
variables (h1 : inscribed_pentagon Ω A B C D E)
variables (h2 : parallel_line (BC) (ST))
variables (h3 : intersect (line BE) (line DS) X)
variables (h4 : intersect (line CE) (line DT) Y)
variables (h5 : tangent_line (line AD) (circle DXY))

theorem tangent_circle (h : tangent_line (line AD) (circle DXY)) : 
    tangent_line (line AE) (circle EXY) :=
sorry

end tangent_circle_l350_350786


namespace minValue_eq_2sqrt5_l350_350295

noncomputable def minValue (a b : ℝ) : ℝ :=
  a^2 + b^2 + 1/a^2 + 1/b^2 + b/a

theorem minValue_eq_2sqrt5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ a b : ℝ, ha, hb ∧ ∀ x y : ℝ, x ≠ 0 → y ≠ 0, minValue x y ≥ 2 * Real.sqrt 5 :=
sorry

end minValue_eq_2sqrt5_l350_350295


namespace sqrt_450_simplified_l350_350618

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350618


namespace quadratic_equation_with_given_means_l350_350910

theorem quadratic_equation_with_given_means (a b : ℝ) 
  (h1 : (a + b) / 2 = 6) 
  (h2 : real.sqrt (a * b) = 5) : 
  (∀ x : ℝ, x^2 - (a + b) * x + a * b = 0 ↔ x^2 - 12 * x + 25 = 0) :=
by
  sorry

end quadratic_equation_with_given_means_l350_350910


namespace simplify_sqrt_450_l350_350409

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350409


namespace sqrt_450_simplified_l350_350418

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350418


namespace cos_double_angle_BAD_l350_350261

noncomputable def triangle_ABC_isosceles_right (A B C : Type) 
  (AB BC AC : ℝ)
  (hAB : AB = 2)
  (hBC : BC = 2)
  (hAC : AC = 2 * Real.sqrt 2) : Prop :=
  AB = BC ∧ AB * AB + BC * BC = AC * AC

noncomputable def triangle_ACD_right (A C D : Type) 
  (AC CD AD : ℝ)
  (hAC : AC = 2 * Real.sqrt 2)
  (hACD : ∠ACD = 60) : Prop :=
  square_sum_sq_perimeter_CD_AD_eq_ABC 2 * (4 + 2 * Real.sqrt 2)

theorem cos_double_angle_BAD (A B C D : Type) 
  (AB BC AC CD AD : ℝ)
  (h_AB : AB = 2)
  (h_BC : BC = 2)
  (h_AC1 : AC = 2 * Real.sqrt 2)
  (h_ACD : ∠ACD = 60)
  (h_perm: 2 * (AB + BC + AC) = AC + CD + AD)
  (h_ABC_iso: triangle_ABC_isosceles_right A B C AB BC AC h_AB h_BC h_AC1)
  (h_ACD_rt: triangle_ACD_right A C D AC CD AD h_AC1 h_ACD) :
  Real.cos (2 * (∠BAD)) = -1 := 
begin
  sorry
end

end cos_double_angle_BAD_l350_350261


namespace fourth_competitor_jump_distance_l350_350723

theorem fourth_competitor_jump_distance (a b c d : ℕ) 
    (h1 : a = 22) 
    (h2 : b = a + 1)
    (h3 : c = b - 2)
    (h4 : d = c + 3):
    d = 24 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fourth_competitor_jump_distance_l350_350723


namespace num_ways_arrange_passengers_l350_350962

theorem num_ways_arrange_passengers 
  (seats : ℕ) (passengers : ℕ) (consecutive_empty : ℕ)
  (h1 : seats = 10) (h2 : passengers = 4) (h3 : consecutive_empty = 5) :
  ∃ ways, ways = 480 := by
  sorry

end num_ways_arrange_passengers_l350_350962


namespace common_chord_length_eq_2sqrt30_l350_350711

noncomputable def circle1 := { center := (5, 5), radius := real.sqrt 50 }
noncomputable def circle2 := { center := (-3, 1), radius := real.sqrt 50 }

theorem common_chord_length_eq_2sqrt30 :
  let A : ℝ × ℝ := (5, 5)
  let B : ℝ × ℝ := (-3, 1)
  let r := real.sqrt 50
  let d := ((5 * 2) + (5 * 1) - 5) / real.sqrt (2^2 + 1^2)
  d = 2 * real.sqrt 5 → 
  2 * real.sqrt (r^2 - d^2) = 2 * real.sqrt 30 := sorry

end common_chord_length_eq_2sqrt30_l350_350711


namespace line_equations_l350_350019

theorem line_equations {a : ℝ} (h1 : ∀ (x : ℝ), y = -x + 1)
  (h2 : ∀ (x y : ℝ), (√3 * x - 3 * y - 3 * √3 = 0) ↔ (x = a ∧ y = -1))
  (h3 : ∀ (x : ℝ), y = √3/3 * x - 5):
  (∀ (x y : ℝ), (x = √3/3 ∧ y = -1) → √3 * x - 3 * y - 3 * √3 = 0
  ∧ ∀ (x : ℝ), y = √3/3 * x - 5 → 3*(y + 5) = √3*x)
  → (x - 3 * y - 6 = 0)
  → (x - 3 * y - 15 = 0) := by
  sorry

end line_equations_l350_350019


namespace num_ordered_pairs_l350_350286

open Set

theorem num_ordered_pairs (A : Set ℕ) (hA : A = {1..100}) :
  let X_sets := {s ∈ powerset A | s ≠ ∅}
  let Y_sets := {t ∈ powerset A | t ≠ ∅}
  let num_pairs := (card X_sets) * (card Y_sets)
  let M := λ X, Sup X
  let m := λ Y, Inf Y
  (num_pairs - ∑ X in X_sets, ∑ Y in Y_sets, if M X > m Y then 1 else 0)
   = 2^200 - 101 * 2^100 := 
 by 
  sorry

end num_ordered_pairs_l350_350286


namespace strictly_increasing_intervals_sum_of_roots_in_interval_l350_350151

def f (x : ℝ) : ℝ := 2 * cos x^2 + 2 * sqrt 3 * sin x * cos x + 2

def g (x : ℝ) : ℝ := 3 + 2 * sin (4 * (x - π / 12) + π / 6)

theorem strictly_increasing_intervals (k : ℤ) : 
  let I1 := (k * π - π / 3)
  let I2 := (k * π + π / 6)
  (∀ x ∈ Ico I1 I2, diff f x > 0) := sorry

theorem sum_of_roots_in_interval (sum_of_roots : ℝ) :
  sum_of_roots = π / 3 :=
  let roots_in_interval := {x : ℝ | g x = 4 ∧ 0 ≤ x ∧ x ≤ π / 4} in
  sum_of_roots = ∑ x in roots_in_interval := sorry

end strictly_increasing_intervals_sum_of_roots_in_interval_l350_350151


namespace probability_problems_l350_350973

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end probability_problems_l350_350973


namespace sqrt_450_eq_15_sqrt_2_l350_350684

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350684


namespace sqrt_450_eq_15_sqrt_2_l350_350685

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350685


namespace parallel_line_equation_perpendicular_line_equation_l350_350141

def line_parallel (l1 l2 : AffinePlane.line ℝ) : Prop :=
  ∃ k : ℝ, l1.slope = l2.slope + k

def line_perpendicular (l1 l2 : AffinePlane.line ℝ) : Prop :=
  l1.slope * l2.slope = -1

def passes_through (l : AffinePlane.line ℝ) (p : AffinePlane.point ℝ) : Prop :=
  l.contains p

noncomputable def line1 : AffinePlane.line ℝ := {
  slope := 2,
  y_intercept := -1/4
}

noncomputable def point1 : AffinePlane.point ℝ := {
  x := 1,
  y := -2
}

theorem parallel_line_equation : 
  (∃ (l2 : AffinePlane.line ℝ), (line_parallel line1 l2) ∧ (passes_through l2 point1) ∧ l2.equation = λ x y, x + 2*y + 3 = 0) :=
sorry

theorem perpendicular_line_equation : 
  (∃ (l2 : AffinePlane.line ℝ), (line_perpendicular line1 l2) ∧ (passes_through l2 point1) ∧ l2.equation = λ x y, 2*x - y - 4 = 0) :=
sorry

end parallel_line_equation_perpendicular_line_equation_l350_350141


namespace simplify_sqrt_450_l350_350498

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350498


namespace spy_statement_reveals_itself_l350_350753

-- Define types of beings: Knight, Knave, and Spy
inductive Being
| knight : Being
| knave : Being
| spy : Being

-- Define a predicate for truth-telling
def tells_truth : Being → Prop
| Being.knight := true
| Being.knave := false
| Being.spy := false

-- Define the statement "I am a liar"
def statement (b : Being) : Prop :=
  b ≠ Being.knight

-- The problem condition is that the spy's statement was false
def spy_statement_is_false : Prop :=
  tells_truth Being.spy = false

-- The main problem statement: the spy will say "I am a liar" and it will be false
theorem spy_statement_reveals_itself : (statement Being.spy = true) ↔ (tells_truth Being.spy = false) :=
by
  sorry

end spy_statement_reveals_itself_l350_350753


namespace simplify_sqrt_450_l350_350400

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350400


namespace sqrt_450_eq_15_sqrt_2_l350_350467

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350467


namespace probability_investment_banking_l350_350003

noncomputable def finance_club := { members : ℕ // members = 20 }

noncomputable def interested_members := { interest : ℕ // interest = 3 * finance_club.val / 4 }

noncomputable def not_interested_members := finance_club.val - interested_members.val

noncomputable def first_not_interested_prob : ℚ := not_interested_members / finance_club.val

noncomputable def second_not_interested_prob : ℚ := (not_interested_members - 1) / (finance_club.val - 1)

noncomputable def neither_interested_prob : ℚ := first_not_interested_prob * second_not_interested_prob

noncomputable def at_least_one_interested_prob : ℚ := 1 - neither_interested_prob

theorem probability_investment_banking : 
  at_least_one_interested_prob = 18 / 19 := 
sorry

end probability_investment_banking_l350_350003


namespace second_tray_holds_l350_350314

-- The conditions and the given constants
variables (x : ℕ) (h1 : 2 * x - 20 = 500)

-- The theorem proving the number of cups the second tray holds is 240 
theorem second_tray_holds (h2 : x = 260) : x - 20 = 240 := by
  sorry

end second_tray_holds_l350_350314


namespace algebra_minimum_value_l350_350148

theorem algebra_minimum_value :
  ∀ x y : ℝ, ∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 + 6*x - 2*y + 12 ≥ m) ∧ m = 2 :=
by
  sorry

end algebra_minimum_value_l350_350148


namespace simplify_sqrt_450_l350_350372

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350372


namespace min_payment_max_payment_expected_payment_l350_350320

-- Given Prices
def item_prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

-- Function to compute the actual paid amount given groups of three items
def paid_amount (groups : List (List ℕ)) : ℕ :=
  groups.foldr (λ group sum => sum + group.foldr (λ x s => s + x) 0 - group.minimum') 0

-- Optimal arrangement of items for minimal payment
def optimal_groups : List (List ℕ) :=
  [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]]

-- Suboptimal arrangement of items for maximal payment
def suboptimal_groups : List (List ℕ) :=
  [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]]

-- Expected value calculation's configuration
def num_items := 10
def num_groups := (num_items / 3).natCeil

noncomputable def expected_amount : ℕ :=
  let total_sum := item_prices.foldr (λ x s => s + x) 0
  let expected_savings := 100 * (660 / 72)
  total_sum - expected_savings

theorem min_payment : paid_amount optimal_groups = 4000 := by
  -- Proof steps and details here
  sorry

theorem max_payment : paid_amount suboptimal_groups = 4900 := by
  -- Proof steps and details here
  sorry

theorem expected_payment : expected_amount ≈ 4583 := by
  -- Proof steps and details here
  sorry

end min_payment_max_payment_expected_payment_l350_350320


namespace number_of_subsets_of_intersection_l350_350164

open Set

variable (A : Set ℕ) (B : Set ℕ)

noncomputable def setA : Set ℕ := {0, 2, 4, 6}
noncomputable def setB : Set ℕ := {n ∈ Set.Iio 3 | n ∈ (Set.range (λ x, 2^x))}

theorem number_of_subsets_of_intersection (hA : A = setA) (hB : B = setB) : 
  (A ∩ B).toFinset.powerset.card = 4 :=
by {
  rw [hA, hB],
  show (setA ∩ setB).toFinset.powerset.card = 4,
  sorry
}

end number_of_subsets_of_intersection_l350_350164


namespace simplify_sqrt_450_l350_350638
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350638


namespace equality_of_segments_l350_350315

-- Definitions of the points and their intersections
variables {A B C D E F : Type} -- points in general space
variables (AE BE CE DE AF BF CF DF : ℝ) -- lengths are real numbers

-- Conditions:
def intersection1 (AB DC : E) := true
def intersection2 (AD BC : F) := true

-- Statement to be proved
theorem equality_of_segments (h1 : intersection1 E) (h2 : intersection2 F) :
  - ((AE * CE) / (BE * DE)) = (AF * CF) / (BF * DF) :=
sorry

end equality_of_segments_l350_350315


namespace number_called_by_2005th_student_l350_350955

theorem number_called_by_2005th_student : 
  let pattern := [1, 2, 3, 4, 3, 2]
  in pattern.get ⟨(2005 % 6) - 1, by simp; norm_num⟩ = 1 :=
by sorry

end number_called_by_2005th_student_l350_350955


namespace contestant_advancing_probability_l350_350239

noncomputable def probability_correct : ℝ := 0.8
noncomputable def probability_incorrect : ℝ := 1 - probability_correct

def sequence_pattern (q1 q2 q3 q4 : Bool) : Bool :=
  -- Pattern INCORRECT, CORRECT, CORRECT, CORRECT
  q1 == false ∧ q2 == true ∧ q3 == true ∧ q4 == true

def probability_pattern (p_corr p_incorr : ℝ) : ℝ :=
  p_incorr * p_corr * p_corr * p_corr

theorem contestant_advancing_probability :
  (probability_pattern probability_correct probability_incorrect = 0.1024) :=
by
  -- Proof required here
  sorry

end contestant_advancing_probability_l350_350239


namespace first_step_is_remove_parentheses_l350_350715

variable (x : ℝ)

def equation : Prop := 2 * x + 3 * (2 * x - 1) = 16 - (x + 1)

theorem first_step_is_remove_parentheses (x : ℝ) (eq : equation x) : 
  ∃ step : String, step = "remove the parentheses" := 
  sorry

end first_step_is_remove_parentheses_l350_350715


namespace sqrt_450_equals_15_sqrt_2_l350_350591

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350591


namespace part_one_part_two_l350_350300

variable {x : ℝ} {m : ℝ}

-- Question 1
theorem part_one (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) : -4 < m ∧ m <= 0 :=
sorry

-- Question 2
theorem part_two (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → mx^2 - mx - 1 > -m + x - 1) : m > 1 :=
sorry

end part_one_part_two_l350_350300


namespace positive_number_decreased_by_4_is_21_times_reciprocal_l350_350041

theorem positive_number_decreased_by_4_is_21_times_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x - 4 = 21 * (1 / x)) : x = 7 := 
sorry

end positive_number_decreased_by_4_is_21_times_reciprocal_l350_350041


namespace lines_same_intersections_l350_350013

theorem lines_same_intersections (n : ℕ) (lines : Fin n → Set (ℝ × ℝ))
  (h_n : n = 2000)
  (h_distinct : ∀ i j : Fin n, i ≠ j → lines i ≠ lines j) :
  ∃ (i j : Fin n), i ≠ j ∧ (intersection_count lines i = intersection_count lines j) :=
sorry

def intersection_count (lines : Fin 2000 → Set (ℝ × ℝ)) (i : Fin 2000) : ℕ :=
  (Finset.card $ Finset.filter (λ j, j ≠ i ∧ ∃ (x : (ℝ × ℝ)), x ∈ lines i ∩ lines j) Finset.univ)

end lines_same_intersections_l350_350013


namespace range_of_m_l350_350125

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then x + 3 else -x^2 + 2*x + 3

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x - Real.exp x - m ≤ 0) → m ≥ 2 :=
by
  intro h
  sorry

end range_of_m_l350_350125


namespace intervals_of_monotonicity_min_value_on_interval_l350_350788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp(-x) * (a * x^2 + a + 1)

theorem intervals_of_monotonicity (a : ℝ) :
  (a ≥ 0 → ∀ x y : ℝ, x < y → f(a, x) > f(a, y)) ∧
  (a < 0 → ∃ x1 x2 : ℝ, x1 < x2 ∧ 
    (∀ x y : ℝ, x < y → (x < x1 ∨ x2 < y) → f(a, x) < f(a, y)) ∧
    (∀ x y : ℝ, x < y → x1 < x ∧ y < x2 → f(a, x) > f(a, y))) :=
sorry

theorem min_value_on_interval (a : ℝ) (h : -1 < a ∧ a < 0) :
  ∀ x ∈ set.Icc (1 : ℝ) 2, f(a, x) ≥ f(a, 2) :=
sorry

end intervals_of_monotonicity_min_value_on_interval_l350_350788


namespace total_amount_is_33_l350_350033

variable (n : ℕ) (c t : ℝ)

def total_amount_paid (n : ℕ) (c t : ℝ) : ℝ :=
  let cost_before_tax := n * c
  let tax := t * cost_before_tax
  cost_before_tax + tax

theorem total_amount_is_33
  (h1 : n = 5)
  (h2 : c = 6)
  (h3 : t = 0.10) :
  total_amount_paid n c t = 33 :=
by
  rw [h1, h2, h3]
  sorry

end total_amount_is_33_l350_350033


namespace find_base_l350_350100

theorem find_base (b : ℕ) : 
  (251_b + 136_b = 407_b) → b = 10 :=
by
  sorry

end find_base_l350_350100


namespace simplify_sqrt_450_l350_350405

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350405


namespace find_p_and_equation_of_l_l350_350931

-- Definitions used in Lean 4 statement from conditions in problem (a)
def parabola (x y : ℝ) (p : ℝ) : Prop := y^2 = 2 * p * x
def passes_through (x y : ℝ) := (x = 4) ∧ (y = -4)
def midpoint (x1 y1 x2 y2 mx my : ℝ) := ((x1 + x2) / 2 = mx) ∧ ((y1 + y2) / 2 = my)

-- Lean 4 statement equivalent to proof problem
theorem find_p_and_equation_of_l :
  (∃ p, p > 0 ∧ ∀ (x y : ℝ), passes_through x y → parabola x y p ∧ p = 2) ∧ 
  (∀ p, p = 2 → 
    ∃ (x1 y1 x2 y2 : ℝ),
      parabola x1 y1 2 ∧ parabola x2 y2 2 ∧ midpoint x1 y1 x2 y2 2 (1 / 3) ∧
      ∃ k b : ℝ, (∀ x, (k = 6) → (b = -35/3) → l x j:= mx → (y = k * x + b) → 18 * x - 3 * y - 35 = 0)) :=
sorry

end find_p_and_equation_of_l_l350_350931


namespace problem_126_times_3_pow_6_l350_350942

theorem problem_126_times_3_pow_6 (p : ℝ) (h : 126 * 3^8 = p) : 
  126 * 3^6 = (1 / 9) * p := 
by {
  -- Placeholder for the proof
  sorry
}

end problem_126_times_3_pow_6_l350_350942


namespace simplify_sqrt_450_l350_350525

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350525


namespace solve_for_x_l350_350706

theorem solve_for_x (x : ℝ) (h : 3^(3*x + 2) = (1 / 81)) : x = -2 :=
sorry

end solve_for_x_l350_350706


namespace anne_already_made_8_drawings_l350_350825

-- Define the conditions as Lean definitions
def num_markers : ℕ := 12
def drawings_per_marker : ℚ := 3 / 2 -- Equivalent to 1.5
def remaining_drawings : ℕ := 10

-- Calculate the total number of drawings Anne can make with her markers
def total_drawings : ℚ := num_markers * drawings_per_marker

-- Calculate the already made drawings
def already_made_drawings : ℚ := total_drawings - remaining_drawings

-- The theorem to prove
theorem anne_already_made_8_drawings : already_made_drawings = 8 := 
by 
  have h1 : total_drawings = 18 := by sorry -- Calculating total drawings as 18
  have h2 : already_made_drawings = 8 := by sorry -- Calculating already made drawings as total drawings minus remaining drawings
  exact h2

end anne_already_made_8_drawings_l350_350825


namespace number_of_tens_in_sum_l350_350206

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l350_350206


namespace simplify_sqrt_450_l350_350535

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350535


namespace simplify_sqrt_450_l350_350670

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350670


namespace sqrt_450_equals_15_sqrt_2_l350_350609

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350609


namespace simplify_sqrt_450_l350_350374

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350374


namespace complex_sum_eval_l350_350106

noncomputable def complex_sum (n : ℕ) (h : n % 4 = 0) : ℂ :=
  finset.sum (finset.range n) (λ k, (2 * k + 1) * complex.I ^ k)

theorem complex_sum_eval (n : ℕ) (h : n % 4 = 0) : 
  complex_sum n h = -n - n * complex.I :=
sorry

end complex_sum_eval_l350_350106


namespace compare_abc_l350_350943

theorem compare_abc (a b c : ℤ) (h1 : a = -(1 ^ 2)) (h2 : b = (3 - real.pi) ^ 0) 
  (h3 : c = (-0.25 : ℚ) ^ 2023 * 4 ^ 2024) : b > a ∧ a > c := by
  have ha : a = -1 := by
    rw h1
    norm_num
  have hb : b = 1 := by
    rw h2
    norm_num
  have hc : c = -4 := by
    rw h3
    norm_num 
    -- use appropriate theorems to handle powers if needed
  rw [ha, hb, hc]
  exact ⟨by norm_num, by norm_num⟩

end compare_abc_l350_350943


namespace sqrt_450_simplified_l350_350627

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350627


namespace correct_statement_l350_350340

-- Definitions corresponding to conditions
def abs_value_r_indicates_strength (r : ℝ) : Bool :=
  r >= -1 ∧ r <= 1

def value_R2_indicates_strength (R2 : ℝ) : Bool :=
  R2 >= 0 ∧ R2 <= 1

-- Problem statement: Verify the correct option regarding correlation and determination coefficients
theorem correct_statement (r : ℝ) (R2 : ℝ) :
  abs_value_r_indicates_strength r →
  value_R2_indicates_strength R2 →
  B :=
by
  sorry

end correct_statement_l350_350340


namespace problem_1_problem_2_problem_3_l350_350934

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem problem_1 (a : ℝ) (h_ne_zero : a ≠ 0) (h_solutions : ∀ x : ℝ, f a x > 0 ↔ x ∈ Ioo (-1/3) (1/2)) : a = 1/5 :=
sorry

theorem problem_2 (a : ℝ) (h_range : a ∈ Set.Ico (-2) 0) (h_always_positive : ∀ x : ℝ, f a x > 0) : 
  ∀ x : ℝ, -1/2 < x ∧ x < 1 :=
sorry

theorem problem_3 (a : ℝ) (h_non_zero : a ≠ 0) (h_x_range : ∀ x ∈ Set.Icc 0 2, f a x > 0) : 
  -3/4 < a :=
sorry

end problem_1_problem_2_problem_3_l350_350934


namespace sum_of_tens_l350_350222

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l350_350222


namespace ratio_of_capitals_l350_350067

noncomputable def Ashok_loss (total_loss : ℝ) (Pyarelal_loss : ℝ) : ℝ := total_loss - Pyarelal_loss

theorem ratio_of_capitals (total_loss : ℝ) (Pyarelal_loss : ℝ) (Ashok_capital Pyarelal_capital : ℝ) 
    (h_total_loss : total_loss = 1200)
    (h_Pyarelal_loss : Pyarelal_loss = 1080)
    (h_Ashok_capital : Ashok_capital = 120)
    (h_Pyarelal_capital : Pyarelal_capital = 1080) :
    Ashok_capital / Pyarelal_capital = 1 / 9 :=
by
  sorry

end ratio_of_capitals_l350_350067


namespace sqrt_450_simplified_l350_350421

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350421


namespace total_distance_traveled_l350_350313

def initial_height : ℝ := 104
def rebounding_factor : ℝ := 0.5

theorem total_distance_traveled : sorry :=
  let first_ascent := initial_height * rebounding_factor in
  let second_descent := first_ascent in
  let second_ascent := second_descent * rebounding_factor in
  let third_descent := second_ascent in
  let total_distance := initial_height + first_ascent + second_descent + second_ascent + third_descent in
  total_distance = 260

end total_distance_traveled_l350_350313


namespace max_profit_of_investment_l350_350823

noncomputable def profit_A (x : ℝ) : ℝ := 18 - 180 / (x + 10)
noncomputable def profit_B (remaining_investment : ℝ) : ℝ := remaining_investment / 5

noncomputable def total_profit (x : ℝ) : ℝ := 
  profit_A(x) + profit_B(100 - x)

theorem max_profit_of_investment :
  ∃ x ∈ Icc (0 : ℝ) 100, total_profit x = 28 :=
by
  sorry

end max_profit_of_investment_l350_350823


namespace sqrt_450_eq_15_sqrt_2_l350_350575

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350575


namespace part1_arithmetic_sequence_part2_max_sum_terms_part3_sum_b_n_l350_350733

-- Definition of the sequence sum S_n
def S (n : ℕ) : ℤ := 33 * n - n^2

-- Prove that {a_n} is an arithmetic sequence
theorem part1_arithmetic_sequence (n : ℕ) (h: n ≥ 1) : ∃ d : ℤ, ∀ m ≥ 1, S (m) - S (m-1) = 34 - 2 * m := sorry

-- Prove the number of terms n that makes the sum S_n the largest
theorem part2_max_sum_terms : ∃ n : ℕ, (1 ≤ n ∧ n = 17) ∧ S (n) = 272 := sorry

-- Sum of |a_n|
def S' (n : ℕ) : ℤ := 
  if n ≤ 17 then 
    33 * n - n^2
  else 
    n^2 - 33 * n + 544
  
-- Prove the formula of S'_n
theorem part3_sum_b_n (n : ℕ) : S' n = if n ≤ 17 then 33 * n - n^2 else n^2 - 33 * n + 544 := sorry

end part1_arithmetic_sequence_part2_max_sum_terms_part3_sum_b_n_l350_350733


namespace speed_of_stream_l350_350946

theorem speed_of_stream
  (V S : ℝ)
  (h1 : 27 = 9 * (V - S))
  (h2 : 81 = 9 * (V + S)) :
  S = 3 :=
by
  sorry

end speed_of_stream_l350_350946


namespace sqrt_450_simplified_l350_350432

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350432


namespace root_in_interval_l350_350299

def f (x : ℝ) : ℝ := 3^x + 3*x - 8

theorem root_in_interval :
  f 1 < 0 ∧ f 1.5 > 0 ∧ f 1.25 < 0 → ∃ x ∈ (1.25, 1.5), f x = 0 :=
by 
  intros h,
  sorry

end root_in_interval_l350_350299


namespace simplify_sqrt_450_l350_350480

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350480


namespace simplify_sqrt_450_l350_350677

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350677


namespace ball_labeled_2_probability_l350_350744

noncomputable def balls_prob_event (combinations : Multiset (Multiset ℕ)) (sum_condition : ℕ) (event_combination : Multiset ℕ) : ℚ :=
  let valid_combinations := combinations.filter (λ c, c.sum = sum_condition)
  let event_count := valid_combinations.count event_combination
  (event_count : ℚ) / (valid_combinations.card : ℚ)

theorem ball_labeled_2_probability :
  let draws := ({1, 2, 3} : Multiset ℕ)
  let combinations := Multiset.replicate 3 draws.bind id
  let event_combination := {2, 2, 2}
  let result := balls_prob_event combinations 6 event_combination
  result = (1 : ℚ) / 7 := 
by sorry

end ball_labeled_2_probability_l350_350744


namespace simplify_sqrt_450_l350_350488

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350488


namespace axis_of_symmetry_l350_350914

theorem axis_of_symmetry 
  (f : ℝ → ℝ) 
  (h : ∀ x, f(x + 3) = f(-(x + 3))) : 
  ∃ a, ∀ x, f(x) = f(2*a - x) ∧ a = 3 :=
sorry

end axis_of_symmetry_l350_350914


namespace necessary_and_sufficient_example_l350_350277

noncomputable def necessary_and_sufficient_condition (a : ℕ → ℝ) := 
  (∀ k, 0 < a k ∧ a k < 1) → 
  ((∀ x : ℝ, 0 < x ∧ x < 1 → ∃ π : ℕ → ℕ, bijective π ∧ x = ∑' k, a (π k) / 2^k) ↔ (inf (set.range a) = 0 ∧ sup (set.range a) = 1))

theorem necessary_and_sufficient_example (a : ℕ → ℝ) :
  necessary_and_sufficient_condition a :=
begin
  sorry
end

end necessary_and_sufficient_example_l350_350277


namespace chocolates_bought_l350_350191

theorem chocolates_bought (cost_price selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : cost_price * 24 = selling_price)
  (h2 : gain_percent = 83.33333333333334)
  (h3 : selling_price = cost_price * 24 * (1 + gain_percent / 100)) :
  cost_price * 44 = selling_price :=
by
  sorry

end chocolates_bought_l350_350191


namespace determine_n_l350_350018

noncomputable def polynomial (n : ℕ) : ℕ → ℕ := sorry  -- Placeholder for the actual polynomial function

theorem determine_n (n : ℕ) 
  (h_deg : ∀ a, polynomial n a = 2 → (3 ∣ a) ∨ a = 0)
  (h_deg' : ∀ a, polynomial n a = 1 → (3 ∣ (a + 2)))
  (h_deg'' : ∀ a, polynomial n a = 0 → (3 ∣ (a + 1)))
  (h_val : polynomial n (3*n+1) = 730) :
  n = 4 :=
sorry

end determine_n_l350_350018


namespace simplify_sqrt_450_l350_350515

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350515


namespace solution_set_t_l350_350887

noncomputable def f (x : ℝ) : ℝ := x^4 + Real.exp (abs x)

theorem solution_set_t (t : ℝ) : 
  2 * f (Real.log t) - f (Real.log (1/t)) ≤ f 2 ↔ e^(-2) ≤ t ∧ t ≤ e^2 :=
by sorry

end solution_set_t_l350_350887


namespace larry_twelfth_finger_l350_350993

def f : ℕ → ℕ
| 5 := 4
| 4 := 3
| 3 := 6
| 6 := 5
| _ := 0  -- Assuming f is undefined for other values, set a default value

def sequence (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 := 5
  | 1 := f 5
  | 2 := f (f 5)
  | 3 := f (f (f 5))
  | _ := 0  -- This should never be hit since n % 4 produces values in {0,1,2,3}

theorem larry_twelfth_finger : sequence 11 = 4 :=
by sorry

end larry_twelfth_finger_l350_350993


namespace line_passing_quadrants_l350_350772

-- Definition of the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Prove that the line passes through Quadrants I, III, and IV
theorem line_passing_quadrants :
  ∃ x_1 > 0, line x_1 > 0 ∧      -- Quadrant I
  ∃ x_3 < 0, line x_3 < 0 ∧      -- Quadrant III
  ∃ x_4 > 0, line x_4 < 0 :=     -- Quadrant IV
sorry

end line_passing_quadrants_l350_350772


namespace length_of_XY_is_correct_l350_350865

noncomputable theory
open_locale big_operators

def length_of_XY (hypotenuse : ℝ) (angle_X : ℝ) : ℝ := hypotenuse / 2

theorem length_of_XY_is_correct :
  ∀ (XY XZ : ℝ), XZ = 6 → angle_X = 30 → XZ = 6 → XY = length_of_XY XZ angle_X → XY = 3 :=
begin
  intros XY XZ hXZ h_angle_X hXZ_again XY_def,
  rw [← XY_def, hXZ],
  exact length_of_XY _ _ = 3,
  sorry
end

end length_of_XY_is_correct_l350_350865


namespace simplify_sqrt_450_l350_350654
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350654


namespace sqrt_450_eq_15_sqrt_2_l350_350696

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350696


namespace simplify_sqrt_450_l350_350639
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350639


namespace number_of_fence_panels_is_10_l350_350029

def metal_rods_per_sheet := 10
def metal_rods_per_beam := 4
def sheets_per_panel := 3
def beams_per_panel := 2
def total_metal_rods := 380

theorem number_of_fence_panels_is_10 :
  (total_metal_rods = 380) →
  (metal_rods_per_sheet = 10) →
  (metal_rods_per_beam = 4) →
  (sheets_per_panel = 3) →
  (beams_per_panel = 2) →
  380 / (3 * 10 + 2 * 4) = 10 := 
by 
  sorry

end number_of_fence_panels_is_10_l350_350029


namespace sqrt_450_simplified_l350_350615

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350615


namespace sqrt_simplify_l350_350357

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350357


namespace problem_solution_l350_350124

theorem problem_solution
  (a b : ℝ)
  (h1 : a * b = 2)
  (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 :=
by
  sorry

end problem_solution_l350_350124


namespace simplify_sqrt_450_l350_350496

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350496


namespace quadrilateral_is_rhombus_l350_350338

-- Definition of a quadrilateral with certain properties 
variables {A B C D O : Type} [Quadrilateral A B C D] [InscribedCircleCenter A B C D O] [DiagonalsIntersectAtCenter A B C D O]

-- Statement to prove
theorem quadrilateral_is_rhombus
  (h₁ : InscribedCircleCenter A B C D O)
  (h₂ : DiagonalsIntersectAtCenter A B C D O):
  IsRhombus A B C D :=
sorry

end quadrilateral_is_rhombus_l350_350338


namespace rectangle_clipping_no_rectangle_l350_350819

theorem rectangle_clipping_no_rectangle (R : Type) [rect : R -> Prop] :
  (∀ (g : R), ¬ rect (clip_corner g) ) := 
  sorry

end rectangle_clipping_no_rectangle_l350_350819


namespace sqrt_450_eq_15_sqrt_2_l350_350690

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350690


namespace base_circumference_cone_l350_350028

theorem base_circumference_cone (r : ℝ) (h : r = 5) (θ : ℝ) (k : θ = 180) : 
  ∃ c : ℝ, c = 5 * π :=
by
  sorry

end base_circumference_cone_l350_350028


namespace probability_simplifies_to_whole_l350_350846

-- Defining sets x and y
def set_x : Set ℕ := {78, 910, 360}
def set_y : Set ℕ := {23, 45, 125}

-- Defining a function that checks if a fraction simplifies to a whole number
def simplifies_to_whole (x y : ℕ) : Prop :=
  ∃ k : ℕ, x = k * y

-- Defining the event that x / y simplifies to a whole number
def event (x y : ℕ) : Bool :=
  simplifies_to_whole x y

-- Counting the total valid combinations
def valid_combinations : ℕ :=
  (set_x.product set_y).count (λ p : ℕ × ℕ, event p.1 p.2)

-- Defining the total number of combinations
def total_combinations : ℕ :=
  set_x.size * set_y.size

-- The probability is the ratio of valid combinations to total combinations
def probability : ℚ :=
  valid_combinations / total_combinations

theorem probability_simplifies_to_whole :
  probability = 1 / 9 := 
  sorry

end probability_simplifies_to_whole_l350_350846


namespace total_ribbon_length_l350_350302

theorem total_ribbon_length (a b c d e f g h i : ℝ) 
  (H : a + b + c + d + e + f + g + h + i = 62) : 
  1.5 * (a + b + c + d + e + f + g + h + i) = 93 :=
by
  sorry

end total_ribbon_length_l350_350302


namespace perpendicular_line_x_intercept_l350_350757

theorem perpendicular_line_x_intercept (a b c x_int : ℝ) 
  (h1 : a = 4) 
  (h2 : b = 5) 
  (h3 : c = 20) 
  (h4 : x_int = 2.4) 
  (h5 : ∀ x y : ℝ, a * x + b * y = c → y = -3 → x = x_int) 
  : ∃ (x : ℝ), ∀ y : ℝ, x ≠ 0 ∧ y = 0 ∧ x = x_int :=
begin
  sorry
end

end perpendicular_line_x_intercept_l350_350757


namespace intersection_P_Q_l350_350166

def P : set ℝ := { y | ∃ x, y = (1/2)^x ∧ x ≥ 0 }
def Q : set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_P_Q : P ∩ Q = (0, 1] := by
  sorry

end intersection_P_Q_l350_350166


namespace angle_AD_BC_l350_350168

-- Defining the coordinates of points O, A, C, D
def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (0, 1, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 2 / Real.sqrt 3)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the magnitude of a vector
def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

-- Define the angle between two vectors
def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ :=
  Real.arccos (dot_product u v / (magnitude u * magnitude v))

theorem angle_AD_BC :
  angle_between_vectors 
    (D.1 - A.1, D.2 - A.2, D.3 - A.3) 
    (C.1 - O.1, C.2 - O.2, C.3 - O.3) 
  = Real.arccos (Real.sqrt (3 / 10)) := by
  sorry

end angle_AD_BC_l350_350168


namespace sqrt_450_eq_15_sqrt_2_l350_350557

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350557


namespace sqrt_450_eq_15_sqrt_2_l350_350549

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350549


namespace c_ge_a_plus_b_sin_half_C_l350_350053

-- Define a triangle with sides a, b, and c opposite to angles A, B, and C respectively, with C being the angle at vertex C
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a)
  (angles_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (angles_sum : A + B + C = π)

namespace TriangleProveInequality

open Triangle

theorem c_ge_a_plus_b_sin_half_C (t : Triangle) :
  t.c ≥ (t.a + t.b) * Real.sin (t.C / 2) := sorry

end TriangleProveInequality

end c_ge_a_plus_b_sin_half_C_l350_350053


namespace sqrt_450_simplified_l350_350628

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350628


namespace sqrt_450_simplified_l350_350422

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350422


namespace more_girls_than_boys_l350_350809

variable (total_students boys : ℕ)
variable (total_students_equals_345 : total_students = 345)
variable (boys_equals_138 : boys = 138)

theorem more_girls_than_boys : total_students - boys - boys = 69 :=
by
  have girls : ℕ := total_students - boys
  calc
    girls - boys
        = total_students - boys - boys : by rfl
    ... = 207 - 138                   : by rw [total_students_equals_345, boys_equals_138]
    ... = 69                          : by norm_num

end more_girls_than_boys_l350_350809


namespace problem_statement_l350_350162

noncomputable def polar_to_cartesian_eq (ρ θ : ℝ) : Prop :=
  let x := ρ * cos θ
  let y := ρ * sin θ
  (x - 1)^2 + (y - 2)^2 = 5

theorem problem_statement (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) :
  P = (0, 3) → 
  (∀ t, l(t) = (1/2 * t, 3 + sqrt 3 / 2 * t)) → 
  polar_to_cartesian_eq ρ θ →
  let d1 := (dist P A)
  let d2 := (dist P B)
  ∃ t1 t2 : ℝ, (l t1 = A) ∧ (l t2 = B) →
  ρ = 2 * cos θ + 4 * sin θ →
  (1 / d1 + 1 / d2 = sqrt (16 - 2 * sqrt 3) / 3) := by sorry

end problem_statement_l350_350162


namespace simplify_sqrt_450_l350_350512

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350512


namespace boys_from_other_communities_l350_350781

theorem boys_from_other_communities :
  ∀ (total_boys : ℕ) (percentage_muslims percentage_hindus percentage_sikhs : ℕ),
  total_boys = 400 →
  percentage_muslims = 44 →
  percentage_hindus = 28 →
  percentage_sikhs = 10 →
  (total_boys * (100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 72 := 
by
  intros total_boys percentage_muslims percentage_hindus percentage_sikhs h1 h2 h3 h4
  sorry

end boys_from_other_communities_l350_350781


namespace sqrt_450_simplified_l350_350435

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350435


namespace sqrt_450_eq_15_sqrt_2_l350_350470

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350470


namespace evaluate_floor_ceiling_sum_l350_350105

theorem evaluate_floor_ceiling_sum : ⌊0.99⌋ + ⌈2.99⌉ + 2 = 5 :=
by
  sorry

end evaluate_floor_ceiling_sum_l350_350105


namespace simplify_sqrt_450_l350_350528

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350528


namespace purchase_prices_profit_from_selling_number_of_basketballs_sold_l350_350026

-- Part 1: Prove the purchase prices of the basketballs
theorem purchase_prices : 
  ∃ (x y : ℕ), 20 * x + 30 * y = 3400 ∧ 30 * x + 40 * y = 4800 :=
by {
  existsi 80,
  existsi 60,
  split;
  linarith,
  sorry
}

-- Part 2: Profit from selling m Type A and n Type B basketballs
theorem profit_from_selling (m n : ℕ):
  120 * m + 90 * n = 5400 →
  (120 - 80) * m + (90 - 60) * n = 1800 :=
by {
  intros h,
  linarith,
  sorry
}

-- Part 3: Number of basketballs sold during promotion
theorem number_of_basketballs_sold:
  ∃ (a b : ℕ), 
  (120 - 80 - 10) * a + (90 * 3 - 60 * 3 - 10) * b = 600 ∧
  (a = 12 ∧ b = 3) ∨ (a = 4 ∧ b = 6) :=
by {
  existsi 12,
  existsi 3,
  split;
  linarith,
  sorry
}

end purchase_prices_profit_from_selling_number_of_basketballs_sold_l350_350026


namespace num_terms_100_pow_10_as_sum_of_tens_l350_350213

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l350_350213


namespace sqrt_450_equals_15_sqrt_2_l350_350593

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350593


namespace taxes_taken_out_l350_350989

theorem taxes_taken_out
  (gross_pay : ℕ)
  (retirement_percentage : ℝ)
  (net_pay_after_taxes : ℕ)
  (tax_amount : ℕ) :
  gross_pay = 1120 →
  retirement_percentage = 0.25 →
  net_pay_after_taxes = 740 →
  tax_amount = gross_pay - (gross_pay * retirement_percentage) - net_pay_after_taxes :=
by
  sorry

end taxes_taken_out_l350_350989


namespace range_of_f_l350_350869

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_f : set.range (λ (x : ℝ), f x) = set.Icc 2 6 :=
by
  sorry

end range_of_f_l350_350869


namespace number_of_tens_in_sum_l350_350207

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l350_350207


namespace range_of_a_l350_350929

noncomputable def f (a x : ℝ) : ℝ := Real.log x + 2 * a * (1 - x)

theorem range_of_a (a : ℝ) :
  (∀ x, x > 2 → f a x > f a 2) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici (1 / 4)) :=
by
  sorry

end range_of_a_l350_350929


namespace archie_needs_sod_l350_350062

-- Define the dimensions of the backyard
def backyard_length : ℕ := 20
def backyard_width : ℕ := 13

-- Define the dimensions of the shed
def shed_length : ℕ := 3
def shed_width : ℕ := 5

-- Statement: Prove that the area of the backyard minus the area of the shed equals 245 square yards
theorem archie_needs_sod : 
  backyard_length * backyard_width - shed_length * shed_width = 245 := 
by sorry

end archie_needs_sod_l350_350062


namespace partition_exists_iff_divisible_by_3_l350_350906

theorem partition_exists_iff_divisible_by_3 (p q : ℕ) (h_coprime : Nat.coprime p q) (h_neq : p ≠ q) :
  (∃ A B C : set ℕ, (∀ z : ℕ, (z ∈ A ∨ z ∈ B ∨ z ∈ C) ∧ (¬(z ∈ A ∧ z ∈ B) ∧ ¬(z ∈ A ∧ z ∈ C) ∧ ¬(z ∈ B ∧ z ∈ C)) ∧ (z + p ∈ A ∨ z + p ∈ B ∨ z + p ∈ C) ∧ (z + q ∈ A ∨ z + q ∈ B ∨ z + q ∈ C) ∧ ((z ∈ A → z + p ∉ A ∧ z + q ∉ A) ∧ (z ∈ B → z + p ∉ B ∧ z + q ∉ B) ∧ (z ∈ C → z + p ∉ C ∧ z + q ∉ C)))) ↔ (p + q) % 3 = 0 := by
  sorry

end partition_exists_iff_divisible_by_3_l350_350906


namespace groups_of_men_and_women_l350_350977

def problem_statement : Prop :=
  let men : Finset ℕ := {1, 2, 3, 4}
  let women : Finset ℕ := {1, 2, 3, 4, 5}
  let group_size := 3
  let total_groups := 3
  -- condition: each group must have at least one man and one woman
  ∃ (group1 group2 group3 : Finset ℕ),
    group1.card = group_size ∧ group2.card = group_size ∧ group3.card = group_size ∧
    group1 ≠ group2 ∧ group2 ≠ group3 ∧ group3 ≠ group1 ∧
    ((group1 ∩ men).nonempty ∧ (group1 ∩ women).nonempty) ∧
    ((group2 ∩ men).nonempty ∧ (group2 ∩ women).nonempty) ∧
    ((group3 ∩ men).nonempty ∧ (group3 ∩ women).nonempty) ∧
    (group1 ∪ group2 ∪ group3 = men ∪ women) ∧
    Finset.card (group1 ∪ group2 ∪ group3) = men.card + women.card

theorem groups_of_men_and_women :
  problem_statement → ∃ n : ℕ, n = 360 :=
by
  sorry

end groups_of_men_and_women_l350_350977


namespace bingo_card_possibilities_l350_350240

theorem bingo_card_possibilities :
  let numbers_set := finset.range 45 \ finset.range 34 -- numbers 35 to 44
  let column_numbers := numbers_set \ {38}
  ∃ (A B C D : ℤ), 
    35 ≤ A ∧ A ≤ 44 ∧
    35 ≤ B ∧ B ≤ 44 ∧
    35 ≤ C ∧ C ≤ 44 ∧
    35 ≤ D ∧ D ≤ 44 ∧
    A ≠ 38 ∧ B ≠ 38 ∧ C ≠ 38 ∧ D ≠ 38 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧
    B ≠ C ∧ B ≠ D ∧
    C ≠ D ∧
    9 * 8 * 7 * 6 = 3024 :=
begin
  sorry
end

end bingo_card_possibilities_l350_350240


namespace min_payment_proof_max_payment_proof_expected_payment_proof_l350_350322

noncomputable def items : List ℕ := List.range 1 11 |>.map (λ n => n * 100)

def min_amount_paid : ℕ :=
  (1000 + 900 + 700 + 600 + 400 + 300 + 100)

def max_amount_paid : ℕ :=
  (1000 + 900 + 800 + 700 + 600 + 500 + 400)

def expected_amount_paid : ℚ :=
  4583 + 33 / 100

theorem min_payment_proof :
  (∑ x in (List.range 15).filter (λ x => x % 3 ≠ 0), (items.get! x : ℕ)) = min_amount_paid := by
  sorry

theorem max_payment_proof :
  (∑ x in List.range 10, if x % 3 = 0 then 0 else (items.get! x : ℕ)) = max_amount_paid := by
  sorry

theorem expected_payment_proof :
  ∑ k in items, ((k : ℚ) * (∏ m in List.range 9, (10 - m) * (9 - m) / 72)) = expected_amount_paid := by
  sorry

end min_payment_proof_max_payment_proof_expected_payment_proof_l350_350322


namespace simplify_sqrt_450_l350_350513

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350513


namespace train_length_l350_350784

theorem train_length (L : ℕ) :
  (∀ a b : ℕ, a = 50000 / 3600 ∧ b = 36000 / 3600 → ((a - b) * 36 = 2 * L) → L = 70) :=
begin
  intros a b,
  assume hab : a = 50000 / 3600 ∧ b = 36000 / 3600,
  assume distance : (a - b) * 36 = 2 * L,
  sorry,
end

end train_length_l350_350784


namespace sqrt_450_equals_15_sqrt_2_l350_350592

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350592


namespace doctors_count_l350_350717

noncomputable def initial_number_of_doctors (D : ℕ) : Prop :=
  D = 11

theorem doctors_count :
  ∃ D : ℕ, (D - 5 + 16 = 22) ∧ D = initial_number_of_doctors D :=
by
  sorry

end doctors_count_l350_350717


namespace simplify_sqrt_450_l350_350392

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350392


namespace percentage_saved_l350_350048

theorem percentage_saved (amount_saved : ℝ) (amount_spent : ℝ) (h1 : amount_saved = 5) (h2 : amount_spent = 45) : 
  (amount_saved / (amount_spent + amount_saved)) * 100 = 10 :=
by 
  sorry

end percentage_saved_l350_350048


namespace cyclist_speed_l350_350806

theorem cyclist_speed (v : ℝ) (h1 : v > 0) :
  let t1 := 7 / v in
  let t2 := 10 / 7 in
  let total_distance := 17 in
  let total_time := t1 + t2 in
  let avg_speed := total_distance / total_time in
  avg_speed = 7.99 → v ≈ 10.01 :=
begin
  sorry
end

end cyclist_speed_l350_350806


namespace simplify_sqrt_450_l350_350524

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350524


namespace range_of_a_l350_350153

noncomputable def f (a x : ℝ) : ℝ :=
  if x > 1 then x + a / x + 1 else -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x ≤ y → f a x ≤ f a y) : -1 ≤ a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l350_350153


namespace sqrt_450_simplified_l350_350420

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350420


namespace dress_assignment_l350_350879

-- Assume we have four girls
inductive Girl
| Anya
| Valya
| Galya
| Nina
  deriving DecidableEq

open Girl

-- Assume four colors of dresses
inductive Dress
| Green
| Blue
| White
| Pink
  deriving DecidableEq

open Dress

-- Defining that each girl has a unique dress in this specific solution scenario
def costume : Girl → Dress
| Anya  => Dress.White
| Valya => Dress.Blue
| Galya => Dress.Green
| Nina  => Dress.Pink

-- The logical conditions
def conditions : Prop :=
  -- 1. The girl in the green dress is not Anya or Valya
  (costume Anya ≠ Dress.Green ∧ costume Valya ≠ Dress.Green) ∧
  -- 2. The girl in the green dress is between the girl in the blue dress and Nina
  (∃ g b n, g = Dress.Green ∧ b = Dress.Blue ∧ n = Dress.Pink ∧ 
    (costume Galya = g ∧ costume Valya = b ∧ costume Nina = n ∧
      (Valya ≠ Anya ∧ Nina ≠ Anya))) ∧
  -- 3. The girl in the white dress is between the girl in the pink dress and Valya
  (∃ w p, w = Dress.White ∧ p = Dress.Pink ∧
    (costume Anya = w ∧ costume Nina = p ∧ 
      ∀ c, c ∈ [Dress.Green, Dress.Blue, Dress.White, Dress.Pink] → p ≠ b))

-- The theorem proving the problem statement
theorem dress_assignment :
  conditions →
  (costume Anya = Dress.White ∧ costume Valya = Dress.Blue ∧ costume Galya = Dress.Green ∧ costume Nina = Dress.Pink) :=
by
  sorry

end dress_assignment_l350_350879


namespace sqrt_450_eq_15_sqrt_2_l350_350466

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350466


namespace part1_part2_part3_l350_350891

noncomputable def f (x : Real) : Real := (1 - 2^x) / (1 + 2^x)

theorem part1 : 
  is_odd f :=
sorry

theorem part2 :
  ∀ x1 x2 : Real, x1 < x2 → f x1 > f x2 :=
sorry

theorem part3 :
  ∀ t k : Real, (f (t^2 - 2 * t) < f (-2 * t^2 + k)) → k < -1 / 3 :=
sorry

end part1_part2_part3_l350_350891


namespace sum_of_tens_l350_350200

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l350_350200


namespace sqrt_simplify_l350_350356

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350356


namespace negation_of_p_range_of_m_when_p_true_range_of_m_when_p_true_and_q_false_l350_350163

variables (m : ℝ) (x₀ : ℝ)

-- Define proposition p and its negation
def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - m - 1 < 0
def neg_p : Prop := ∀ x₀ : ℝ, x₀^2 + 2 * x₀ - m - 1 ≥ 0

-- Define proposition q
def q : Prop := ∀ x ∈ Set.Icc 1 4, x + 4 / x > m

-- Theorems to be proven
theorem negation_of_p : neg_p = ¬ p := by
  sorry

theorem range_of_m_when_p_true {m : ℝ} (hp : p) : m > -2 := by
  sorry

theorem range_of_m_when_p_true_and_q_false {m : ℝ} (hp : p) (hq_false : ¬ q) :
  m ≥ 4 ∨ m ≤ -2 := by
  sorry

end negation_of_p_range_of_m_when_p_true_range_of_m_when_p_true_and_q_false_l350_350163


namespace packs_of_yellow_bouncy_balls_l350_350304

/-- Maggie bought 4 packs of red bouncy balls, some packs of yellow bouncy balls (denoted as Y), and 4 packs of green bouncy balls. -/
theorem packs_of_yellow_bouncy_balls (Y : ℕ) : 
  (4 + Y + 4) * 10 = 160 -> Y = 8 := 
by 
  sorry

end packs_of_yellow_bouncy_balls_l350_350304


namespace y_coord_of_parabola_vertex_l350_350855

noncomputable def y_coord_of_vertex (a b c : ℝ) : ℝ :=
let x_vertex := (-b) / (2 * a) in
a * x_vertex^2 + b * x_vertex + c

theorem y_coord_of_parabola_vertex :
  y_coord_of_vertex (-5) (-50) 10 = 135 := by
  sorry

end y_coord_of_parabola_vertex_l350_350855


namespace number_of_sophomores_in_sample_l350_350802

def number_of_students : ℕ := 400 + 600 + 500

def stratified_sampling (total_students : ℕ) (proportion : ℚ) (sample_size : ℕ) : ℕ :=
  (sample_size * proportion.num : ℕ) / proportion.denom

theorem number_of_sophomores_in_sample :
  let freshmen := 400
  let sophomores := 600
  let juniors := 500
  let total_students := freshmen + sophomores + juniors
  let sample_size := 100
  let sophomore_proportion := (sophomores : ℚ) / total_students
  let number_of_sophomores_in_sample := stratified_sampling total_students sophomore_proportion sample_size
  number_of_sophomores_in_sample = 40
  := by
  sorry

end number_of_sophomores_in_sample_l350_350802


namespace sqrt_450_eq_15_sqrt_2_l350_350686

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350686


namespace sum_area_triangles_lt_total_area_l350_350014

noncomputable def G : ℝ := (1 + Real.sqrt 5) / 2

noncomputable def A_k (k : ℕ+) : ℝ := sorry -- Assume we've defined A_k's expression correctly
noncomputable def S (S1 S2 : ℝ) : ℝ := 2 * S1 - S2

theorem sum_area_triangles_lt_total_area (k : ℕ+) (S1 S2 : ℝ) :
  (A_k k < S S1 S2) :=
sorry

end sum_area_triangles_lt_total_area_l350_350014


namespace bobby_candy_left_l350_350831

def initial_candy := 22
def eaten_candy1 := 9
def eaten_candy2 := 5

theorem bobby_candy_left : initial_candy - eaten_candy1 - eaten_candy2 = 8 :=
by
  sorry

end bobby_candy_left_l350_350831


namespace simplify_sqrt_450_l350_350657

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350657


namespace sqrt_450_eq_15_sqrt_2_l350_350584

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350584


namespace find_h_l350_350108

theorem find_h (h : ℝ[X]) :
  3 * (X^4) + 2 * X - 1 + h = 5 * (X^2) - 6 * X - 1 ↔ h = -3 * (X^4) + 5 * (X^2) - 8 * X :=
by
  sorry

end find_h_l350_350108


namespace probability_problems_l350_350974

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end probability_problems_l350_350974


namespace simplify_sqrt_450_l350_350482

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350482


namespace coefficient_of_x3_in_expansion_eq_20_l350_350251

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Translation of the problem
def binomialExpansionTerm (n : ℕ) (r : ℕ) (a b : ℕ) (x : ℕ) : ℕ :=
  binomial n r * a^(n-r) * b^r * x^(n - 2 * r)

-- Given conditions
def given_expansion (r : ℕ) : ℕ := binomialExpansionTerm 5 r 2 (1 / 4) x

-- Proof statement
theorem coefficient_of_x3_in_expansion_eq_20 :
  ∃ r, given_expansion r = 20 * x^3 :=
by sorry

end coefficient_of_x3_in_expansion_eq_20_l350_350251


namespace simplify_sqrt_450_l350_350522

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350522


namespace james_net_income_correct_l350_350265

def regular_price_per_hour : ℝ := 20
def discount_percent : ℝ := 0.10
def rental_hours_per_day_monday : ℝ := 8
def rental_hours_per_day_wednesday : ℝ := 8
def rental_hours_per_day_friday : ℝ := 6
def rental_hours_per_day_sunday : ℝ := 5
def sales_tax_percent : ℝ := 0.05
def car_maintenance_cost_per_week : ℝ := 35
def insurance_fee_per_day : ℝ := 15

-- Total rental hours
def total_rental_hours : ℝ :=
  rental_hours_per_day_monday + rental_hours_per_day_wednesday + rental_hours_per_day_friday + rental_hours_per_day_sunday

-- Total rental income before discount
def total_rental_income : ℝ := total_rental_hours * regular_price_per_hour

-- Discounted rental income
def discounted_rental_income : ℝ := total_rental_income * (1 - discount_percent)

-- Total income with tax
def total_income_with_tax : ℝ := discounted_rental_income * (1 + sales_tax_percent)

-- Total expenses
def total_expenses : ℝ := car_maintenance_cost_per_week + (insurance_fee_per_day * 4)

-- Net income
def net_income : ℝ := total_income_with_tax - total_expenses

theorem james_net_income_correct : net_income = 415.30 :=
  by
    -- proof omitted
    sorry

end james_net_income_correct_l350_350265


namespace race_distance_l350_350968

theorem race_distance {d a b c : ℝ} 
    (h1 : d / a = (d - 25) / b)
    (h2 : d / b = (d - 15) / c)
    (h3 : d / a = (d - 35) / c) :
  d = 75 :=
by
  sorry

end race_distance_l350_350968


namespace a_9_value_l350_350894

-- Define the sequence and its sum of the first n terms
def S (n : ℕ) : ℕ := n^2

-- Define the terms of the sequence
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- The main statement to be proved
theorem a_9_value : a 9 = 17 :=
by
  sorry

end a_9_value_l350_350894


namespace simplify_sqrt_450_l350_350533

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350533


namespace simplify_sqrt_450_l350_350484

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350484


namespace magnitude_of_b_l350_350937

open Real

-- Define the vectors and their relationship conditions
def a : ℝ × ℝ := (-4, 6)

def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Prove that in the conditions where vectors are parallel
def vectors_parallel_condition (x : ℝ) : Prop := a.1 / b x.1 = a.2 / b x.2

-- The theorem to be proved
theorem magnitude_of_b (x : ℝ) (h₁ : vectors_parallel_condition x) : 
  sqrt ((b x).1^2 + (b x).2^2) = sqrt 13 :=
by sorry

end magnitude_of_b_l350_350937


namespace simplify_sqrt_450_l350_350501

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350501


namespace vectors_orthogonal_l350_350901

variables {α : Type*} [InnerProductSpace ℝ α] (a b : α)

-- Defining the conditions
def non_zero (v : α) : Prop := v ≠ 0
def given_condition : Prop := ∥2 • a + b∥ = ∥2 • a - b∥

-- Mathematical statement to prove
theorem vectors_orthogonal (h₁ : non_zero a) (h₂ : non_zero b) (h₃ : given_condition a b) : inner a b = 0 :=
sorry

end vectors_orthogonal_l350_350901


namespace min_payment_proof_max_payment_proof_expected_payment_proof_l350_350325

noncomputable def items : List ℕ := List.range 1 11 |>.map (λ n => n * 100)

def min_amount_paid : ℕ :=
  (1000 + 900 + 700 + 600 + 400 + 300 + 100)

def max_amount_paid : ℕ :=
  (1000 + 900 + 800 + 700 + 600 + 500 + 400)

def expected_amount_paid : ℚ :=
  4583 + 33 / 100

theorem min_payment_proof :
  (∑ x in (List.range 15).filter (λ x => x % 3 ≠ 0), (items.get! x : ℕ)) = min_amount_paid := by
  sorry

theorem max_payment_proof :
  (∑ x in List.range 10, if x % 3 = 0 then 0 else (items.get! x : ℕ)) = max_amount_paid := by
  sorry

theorem expected_payment_proof :
  ∑ k in items, ((k : ℚ) * (∏ m in List.range 9, (10 - m) * (9 - m) / 72)) = expected_amount_paid := by
  sorry

end min_payment_proof_max_payment_proof_expected_payment_proof_l350_350325


namespace ellipse_equation_l350_350057

theorem ellipse_equation :
  ∃ (m n : ℝ), m > n > 0 ∧ 
    (∀ x y : ℝ, x = sqrt 3 → y = - sqrt 5 → y^2 / m + x^2 / n = 1) ∧ 
    (∀ c : ℝ, c = sqrt (25 - 9) → ∀ f1 f2 : ℝ, f1 = 4 ∧ f2 = -4 → ∀ a b x y : ℝ, 
      x * y ≠ 0 → a^2 = 25 ∧ b^2 = 9 → (y^2 / m + x^2 / n = 1 → m - n = 16 ∧ 5 / m + 3 / n = 1)) ∧ 
    m = 20 ∧ n = 4 := by
    sorry

end ellipse_equation_l350_350057


namespace crayons_left_l350_350830

theorem crayons_left :
  let initial_crayons := 5325
  let crayons_given_to_Jen := (3 * initial_crayons) / 4
  let remaining_after_Jen := initial_crayons - int.floor crayons_given_to_Jen
  let received_from_Amy := 1500
  let total_after_Amy := remaining_after_Jen + received_from_Amy
  let crayons_donated_to_school := (2 * total_after_Amy) / 3
  let final_crayons := total_after_Amy - int.floor crayons_donated_to_school
  final_crayons = 944 :=
   by {
     sorry
   }

end crayons_left_l350_350830


namespace area_difference_equal_28_5_l350_350897

noncomputable def square_side_length (d: ℝ) : ℝ := d / Real.sqrt 2
noncomputable def square_area (d: ℝ) : ℝ := (square_side_length d) ^ 2
noncomputable def circle_radius (D: ℝ) : ℝ := D / 2
noncomputable def circle_area (D: ℝ) : ℝ := Real.pi * (circle_radius D) ^ 2
noncomputable def area_difference (d D : ℝ) : ℝ := |circle_area D - square_area d|

theorem area_difference_equal_28_5 :
  ∀ (d D : ℝ), d = 10 → D = 10 → area_difference d D = 28.5 :=
by
  intros d D hd hD
  rw [hd, hD]
  -- Remaining steps involve computing the known areas and their differences
  sorry

end area_difference_equal_28_5_l350_350897


namespace probability_region_C_l350_350024

theorem probability_region_C :
  let A := 1/5
  let B := 1/3
  let C := x
  let D := C
  let E := 2 * C
  let total := 1
  1 + (1/15) = 1 /\ x = 7/60 

end probability_region_C_l350_350024


namespace average_last_three_students_is_35_l350_350790

-- Given definitions and conditions
def first_student_time : ℕ := 15
def average_all_runners_time : ℕ := 30
def number_of_runners : ℕ := 4

-- Definition for the average time of the last three students
def average_last_three_students_time : ℕ :=
  let total_time_all_students := average_all_runners_time * number_of_runners in
  let total_time_last_three_students := total_time_all_students - first_student_time in
  total_time_last_three_students / (number_of_runners - 1)

-- Proof statement
theorem average_last_three_students_is_35 :
  average_last_three_students_time = 35 :=
by
  -- The actual proof will go here
  sorry

end average_last_three_students_is_35_l350_350790


namespace can_pay_without_change_l350_350838

theorem can_pay_without_change (n : ℕ) (h : n > 7) :
  ∃ (a b : ℕ), 3 * a + 5 * b = n :=
sorry

end can_pay_without_change_l350_350838


namespace simplify_sqrt_450_l350_350675

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350675


namespace find_number_l350_350944

theorem find_number :
  let number := 10 in
  let modified_expression :=
    (9 : ℝ) - (8 : ℝ) / (7 : ℝ) * (5 : ℝ) + number in
  modified_expression = 13.285714285714286 :=
by
  sorry

end find_number_l350_350944


namespace algebraic_expression_l350_350954

-- Given conditions in the problem.
variables (x y : ℝ)

-- The statement to be proved: If 2x - 3y = 1, then 6y - 4x + 8 = 6.
theorem algebraic_expression (h : 2 * x - 3 * y = 1) : 6 * y - 4 * x + 8 = 6 :=
by 
  sorry

end algebraic_expression_l350_350954


namespace number_of_tens_in_sum_l350_350205

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l350_350205


namespace registered_voters_vote_for_candidate_A_l350_350234

theorem registered_voters_vote_for_candidate_A :
  ∀ (total_voters : ℕ) (percent_democrats percent_republicans percent_dem_vote_A percent_rep_vote_A : ℝ)
    (num_democrats num_republicans num_dem_vote_A num_rep_vote_A : ℕ),
  percent_democrats = 0.60 →
  percent_republicans = 0.40 →
  percent_dem_vote_A = 0.70 →
  percent_rep_vote_A = 0.20 →
  (num_democrats = percent_democrats * total_voters) →
  (num_republicans = percent_republicans * total_voters) →
  (num_dem_vote_A = percent_dem_vote_A * num_democrats) →
  (num_rep_vote_A = percent_rep_vote_A * num_republicans) →
  (num_dem_vote_A + num_rep_vote_A) = 0.50 * total_voters :=
by
  intros total_voters percent_democrats percent_republicans percent_dem_vote_A percent_rep_vote_A
          num_democrats num_republicans num_dem_vote_A num_rep_vote_A
          h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end registered_voters_vote_for_candidate_A_l350_350234


namespace min_payment_max_payment_expected_value_payment_l350_350336

-- Proof Problem 1
theorem min_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let optimized_groups := [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]] in
  (∑ g in optimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4000 := by
  sorry

-- Proof Problem 2
theorem max_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let suboptimized_groups := [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]] in
  (∑ g in suboptimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4900 := by
  sorry

-- Proof Problem 3
theorem expected_value_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let expected_savings := 100 * (∑ k in List.range 9, k * ((10 - k) * (9 - k)) / 72) in
  (5500 - expected_savings) = 4583.33 := by 
  sorry

end min_payment_max_payment_expected_value_payment_l350_350336


namespace cos_of_tan_sec_l350_350182

theorem cos_of_tan_sec {B : ℝ} (h : tan B + sec B = 3) : cos B = 3 / 5 :=
sorry

end cos_of_tan_sec_l350_350182


namespace lizard_eye_difference_l350_350987

def jan_eye : ℕ := 3
def jan_wrinkle : ℕ := 3 * jan_eye
def jan_spot : ℕ := 7 * jan_wrinkle

def cousin_eye : ℕ := 3
def cousin_wrinkle : ℕ := 2 * cousin_eye
def cousin_spot : ℕ := 5 * cousin_wrinkle

def total_eyes : ℕ := jan_eye + cousin_eye
def total_wrinkles : ℕ := jan_wrinkle + cousin_wrinkle
def total_spots : ℕ := jan_spot + cousin_spot
def total_spots_and_wrinkles : ℕ := total_wrinkles + total_spots

theorem lizard_eye_difference : total_spots_and_wrinkles - total_eyes = 102 := by
  sorry

end lizard_eye_difference_l350_350987


namespace sqrt_450_simplified_l350_350622

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350622


namespace sqrt_450_eq_15_sqrt_2_l350_350566

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350566


namespace pyramid_volume_l350_350341

-- Define the conditions
def isRegularHexagon (pyramidBase : Set ℝ) : Prop := sorry -- Regular hexagon definition

def isEquilateralTriangle (triangle : Set ℝ) (side : ℝ) : Prop := sorry -- Equilateral triangle definition

-- The given problem stated in Lean
theorem pyramid_volume (GHIJKL : Set ℝ) (QGHIJKL : Set ℝ) (Q G K : ℝ)
  (h₁ : isRegularHexagon GHIJKL) 
  (h₂ : isEquilateralTriangle {Q, G, K} 10) : 
  volume QGHIJKL = 562.5 := 
sorry

end pyramid_volume_l350_350341


namespace parabola_directrix_l350_350714

theorem parabola_directrix (a : ℝ) :
  (∃ y : ℝ, y = ax^2 ∧ y = -2) → a = 1/8 :=
by
  -- Solution steps are omitted.
  sorry

end parabola_directrix_l350_350714


namespace sqrt_simplify_l350_350355

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350355


namespace perpendicular_line_x_intercept_l350_350756

theorem perpendicular_line_x_intercept (a b c x_int : ℝ) 
  (h1 : a = 4) 
  (h2 : b = 5) 
  (h3 : c = 20) 
  (h4 : x_int = 2.4) 
  (h5 : ∀ x y : ℝ, a * x + b * y = c → y = -3 → x = x_int) 
  : ∃ (x : ℝ), ∀ y : ℝ, x ≠ 0 ∧ y = 0 ∧ x = x_int :=
begin
  sorry
end

end perpendicular_line_x_intercept_l350_350756


namespace number_of_4digit_integers_l350_350940

theorem number_of_4digit_integers :
  let first_two_digits := {2, 6, 9}
  let last_two_digits := {4, 6, 8}
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ 
           (∀ i ∈ {1000, 100, 10, 1}, 
            (i = 1000 ∨ i = 100 → n / i % 10 ∈ first_two_digits) ∧
            (i = 10 ∨ i = 1 → n / i % 10 ∈ last_two_digits)) ∧
           (n / 10 % 10 ≠ n % 10) ∧
           54 = {m : ℕ | (1000 ≤ m ∧ m < 10000) ∧ 
                         (∀ i ∈ {1000, 100, 10, 1}, 
                          (i = 1000 ∨ i = 100 → m / i % 10 ∈ first_two_digits) ∧
                          (i = 10 ∨ i = 1 → m / i % 10 ∈ last_two_digits)) ∧
                         (m / 10 % 10 ≠ m % 10)}.to_finset.card
:=
by
  sorry

end number_of_4digit_integers_l350_350940


namespace simplify_sqrt_450_l350_350390

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350390


namespace ursula_initial_money_l350_350751

def cost_per_hot_dog : ℝ := 1.50
def number_of_hot_dogs : ℕ := 5
def cost_per_salad : ℝ := 2.50
def number_of_salads : ℕ := 3
def change_received : ℝ := 5.00

def total_cost_of_hot_dogs : ℝ := number_of_hot_dogs * cost_per_hot_dog
def total_cost_of_salads : ℝ := number_of_salads * cost_per_salad
def total_cost : ℝ := total_cost_of_hot_dogs + total_cost_of_salads
def amount_paid : ℝ := total_cost + change_received

theorem ursula_initial_money : amount_paid = 20.00 := by
  /- Proof here, which is not required for the task -/
  sorry

end ursula_initial_money_l350_350751


namespace sqrt_expr_identity_l350_350129

-- Definitions for x and y
def x : ℝ := Real.sqrt 3 + Real.sqrt 2
def y : ℝ := Real.sqrt 3 - Real.sqrt 2

-- The statement we need to prove
theorem sqrt_expr_identity : (x^2 * y + x * y^2) = 2 * Real.sqrt 3 :=
by sorry

end sqrt_expr_identity_l350_350129


namespace sqrt_450_eq_15_sqrt_2_l350_350563

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350563


namespace simplify_sqrt_450_l350_350666

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350666


namespace min_shift_odd_func_l350_350926

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem min_shift_odd_func (hφ : ∀ x : ℝ, f (x) = -f (-x + 2 * φ + (Real.pi / 3))) (hφ_positive : φ > 0) :
  φ = Real.pi / 6 :=
sorry

end min_shift_odd_func_l350_350926


namespace parallel_lines_perpendicular_to_plane_l350_350293

-- Definitions of lines and planes
variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

-- Conditions
variables (hmn_parallel : m ∥ n)
variables (hm_perp_alpha : m ⊥ α)

-- The proof statement
theorem parallel_lines_perpendicular_to_plane (hmn_parallel : m ∥ n) (hm_perp_alpha : m ⊥ α) : n ⊥ α :=
sorry

end parallel_lines_perpendicular_to_plane_l350_350293


namespace distance_karen_covers_l350_350734

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end distance_karen_covers_l350_350734


namespace circle_distance_theorem_l350_350247

noncomputable def circle_distance_problem (m : ℝ) : Prop :=
    let A := (2 : ℝ, 2 : ℝ)
    let B := (m, 0 : ℝ)
    let distance_AB := Real.sqrt ((m - 2) ^ 2 + 4)
    (2 - 2 * Real.sqrt 3 < m ∧ m < 2) ∨ (2 < m ∧ m < 2 + 2 * Real.sqrt 3)

theorem circle_distance_theorem (m : ℝ) :
    (∃ (ℓ₁ ℓ₂ : ℝ → ℝ),
        (abs (ℓ₁ 2 - 2) / sqrt ((ℓ₁ 1)^2 + 1) = 1) ∧
        (abs (ℓ₂ m - 0) / sqrt ((ℓ₂ 1)^2 + 1) = 3) ∧
        (∀ (ℓ : ℝ → ℝ), (abs (ℓ 2 - 2) / sqrt ((ℓ 1)^2 + 1) = 1 → abs (ℓ m - 0) / sqrt ((ℓ 1)^2 + 1) = 3) →
            (ℓ = ℓ₁ ∨ ℓ = ℓ₂))) →
    circle_distance_problem m := sorry

end circle_distance_theorem_l350_350247


namespace winning_candidate_votes_l350_350068

theorem winning_candidate_votes (T W : ℕ) (d1 d2 d3 : ℕ) 
  (hT : T = 963)
  (hd1 : d1 = 53) 
  (hd2 : d2 = 79) 
  (hd3 : d3 = 105) 
  (h_sum : T = W + (W - d1) + (W - d2) + (W - d3)) :
  W = 300 := 
by
  sorry

end winning_candidate_votes_l350_350068


namespace combined_total_difference_l350_350079

theorem combined_total_difference :
  let Chris_cards := 18
  let Charlie_cards := 32
  let Diana_cards := 25
  let Ethan_cards := 40
  (Charlie_cards - Chris_cards) + (Diana_cards - Chris_cards) + (Ethan_cards - Chris_cards) = 43 :=
by
  let Chris_cards := 18
  let Charlie_cards := 32
  let Diana_cards := 25
  let Ethan_cards := 40
  have h1 : Charlie_cards - Chris_cards = 14 := by sorry
  have h2 : Diana_cards - Chris_cards = 7 := by sorry
  have h3 : Ethan_cards - Chris_cards = 22 := by sorry
  show (Charlie_cards - Chris_cards) + (Diana_cards - Chris_cards) + (Ethan_cards - Chris_cards) = 43 from
    by rw [h1, h2, h3]; exact (14 + 7 + 22).symm

end combined_total_difference_l350_350079


namespace parallelogram_diagonal_squared_sum_mnp_l350_350998

theorem parallelogram_diagonal_squared_sum_mnp 
  (ABCD : Parallelogram) (area_ABCD : area ABCD = 24) 
  (P Q R S : Point) (proj_A_on_BD : proj P A BD) (proj_C_on_BD : proj Q C BD)
  (proj_B_on_AC : proj R B AC) (proj_D_on_AC : proj S D AC)
  (PQ_len : length P Q = 10) (RS_len : length R S = 12) 
  (d : ℝ) (d_def : d = length BD) : 
  d^2 = 40 + 4 * real.sqrt 274 ∧ (let m := 40, n := 4, p := 274 in m + n + p = 318) :=
sorry

end parallelogram_diagonal_squared_sum_mnp_l350_350998


namespace sum_of_angles_l350_350260

theorem sum_of_angles (A B C D M : Point) (h1 : inside_rectangle A B C D M) (h2 : ∠ B M C + ∠ A M D = 180) :
∠ B C M + ∠ D A M = 90 :=
sorry

end sum_of_angles_l350_350260


namespace problem_l350_350252

open Real

noncomputable def intersectsAtTwoPoints (f : ℝ → ℝ) (l : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = l x₁ ∧ f x₂ = l x₂

noncomputable def circlePassesThroughThreePoints
  (circle : ℝ × ℝ → Prop) (A B O : ℝ × ℝ) : Prop :=
  circle A ∧ circle B ∧ circle O

axiom quadratic_function : ∀ x : ℝ, f x = x^2
axiom line_equation : ∀ x : ℝ, l x = x + m
axiom circle_equation : circle = λ p, p.1^2 + p.2^2 - m * p.1 - (2 + m) * p.2 = 0

theorem problem (m : ℝ) (A B : ℝ × ℝ)
  (h1 : intersectsAtTwoPoints (λ x, x^2) (λ x, x + m) m)
  (h2 : m ≠ 0) :
  m > -1/4 ∧
  (∠ O A B <= π / 2 → false → false) ∧
  circle (0,0) ∧ circle (A.1, A.2) ∧ circle (B.1, B.2) ∧
  circle (-1,1) :=
by
  sorry

end problem_l350_350252


namespace number_of_tens_in_sum_l350_350204

theorem number_of_tens_in_sum : (100^10) / 10 = 10^19 := sorry

end number_of_tens_in_sum_l350_350204


namespace lateral_surface_area_of_pyramid_l350_350135

theorem lateral_surface_area_of_pyramid {a h : ℝ} (h_base : a = 2) (h_height : h = 3) : 
  let l : ℝ := (a / 2) in
  let slant_height := Real.sqrt (l^2 + h^2) in
  let lateral_surface_area := 4 * (1 / 2) * a * slant_height in
  lateral_surface_area = 4 * Real.sqrt 10 :=
by
  sorry

end lateral_surface_area_of_pyramid_l350_350135


namespace total_interest_prove_l350_350732

variables (P R : ℝ) (SI_10 SI_total SI_5_new P_new : ℝ)
-- Given conditions
-- 1. Simple interest on a sum of money will be Rs. 700 after 10 years:
def simple_interest_10_years := (P * R * 10) / 100 = 700

-- 2. The principal is trebled after 5 years:
def treble_principal := P_new = 3 * P

-- Define interests based on given conditions
def interest_next_5_years := SI_5_new = (P_new * R * 5) / 100

def total_interest := SI_total = 700 + SI_5_new

-- Prove that the total interest at the end of the tenth year is Rs. 1750
theorem total_interest_prove :
  simple_interest_10_years P R 700 → 
  treble_principal P P_new → 
  interest_next_5_years P_new R SI_5_new → 
  total_interest 700 SI_5_new 1750 →
  SI_total = 1750 := 
sorry

end total_interest_prove_l350_350732


namespace simplify_sqrt_450_l350_350667

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350667


namespace simplify_sqrt_450_l350_350453

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350453


namespace village_assignments_l350_350022

theorem village_assignments (students villages : ℕ) (h_students : students = 5) (h_villages : villages = 3)
    (h_at_least_one : ∀ s, s ∈ (finset.range villages).powerset → s.card > 0)
    (h_exactly_one_village_A : ∃ a, a ∈ finset.range students ∧ ∀ x ∈ (finset.range students).erase a, x ≠ a) :
    ∃ n, n = 70 :=
by
  sorry  -- Proof goes here

end village_assignments_l350_350022


namespace find_subsequence_multiple_of_n_l350_350888

theorem find_subsequence_multiple_of_n (n : ℕ) (a : Fin n → ℤ) :
  ∃ (i j : Fin n), i ≤ j ∧ (∑ k in Finset.Icc i j, a k) % n = 0 := by
  sorry

end find_subsequence_multiple_of_n_l350_350888


namespace percentage_greater_l350_350957

theorem percentage_greater (x : ℝ) (h1 : x = 96) (h2 : x > 80) : ((x - 80) / 80) * 100 = 20 :=
by
  sorry

end percentage_greater_l350_350957


namespace simplify_sqrt_450_l350_350403

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350403


namespace find_midpoint_segment_l350_350936

-- Define two parallel lines and a segment on one of these lines
variables (L₁ L₂ : set Point) [parallel_lines L₁ L₂]
variables (C D : Point) (hCD : C ∈ L₁ ∧ D ∈ L₁)

-- The question is to find the midpoint of the segment [C, D]
theorem find_midpoint_segment (L₁ L₂ : set Point) [parallel_lines L₁ L₂]
  (C D : Point) (hCD : C ∈ L₁ ∧ D ∈ L₁) :
  ∃ M : Point, M ∈ line_segment C D ∧ is_midpoint M C D :=
sorry

end find_midpoint_segment_l350_350936


namespace total_stamps_l350_350343

theorem total_stamps {books_of_10 books_of_15 : ℕ} (h1 : books_of_10 = 4) (h2 : books_of_15 = 6) :
  (books_of_10 * 10 + books_of_15 * 15 = 130) :=
by
  -- Definitions based on the conditions
  let total_10 := books_of_10 * 10
  let total_15 := books_of_15 * 15
  -- Summing up the total stamps
  have h_total : total_10 + total_15 = 130, from sorry
  exact h_total

end total_stamps_l350_350343


namespace range_of_f_l350_350146

noncomputable def f : ℝ → ℝ :=
λ x, if (1 ≤ x ∧ x ≤ 2) then x + 1 else (
   if (2 < x ∧ x ≤ 4) then 2 * (x - 3) ^ 2 + 1 else 0)

theorem range_of_f : Set.range f = Set.Icc 1 3 := by
  sorry

end range_of_f_l350_350146


namespace minimum_distance_l350_350892

theorem minimum_distance
  (P Q : ℝ × ℝ)
  (hP : P.2 = Real.exp P.1)
  (hQ : Q.2 = Real.log Q.1) :
  ∀ PQ, PQ = Real.dist P Q -> PQ ≥ Real.sqrt 2 :=
sorry

end minimum_distance_l350_350892


namespace gas_volume_at_31_degrees_l350_350875

theorem gas_volume_at_31_degrees :
  (∀ T V : ℕ, (T = 45 → V = 30) ∧ (∀ k, T = 45 - 2 * k → V = 30 - 3 * k)) →
  ∃ V, (T = 31) ∧ (V = 9) :=
by
  -- The proof would go here
  sorry

end gas_volume_at_31_degrees_l350_350875


namespace sqrt_450_eq_15_sqrt_2_l350_350569

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350569


namespace simplify_sqrt_450_l350_350385

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350385


namespace min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350331

/- Part (a): Minimum amount the customer will pay -/
theorem min_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4000 := 
sorry

/- Part (b): Maximum amount the customer will pay -/
theorem max_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4900 := 
sorry

/- Part (c): Expected value the customer will pay -/
theorem expected_amount_to_pay :
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4583.33 := 
sorry

end min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350331


namespace sum_of_tens_l350_350198

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l350_350198


namespace sqrt_450_simplified_l350_350624

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350624


namespace perpendicular_line_x_intercept_l350_350755

-- Definition of the given line and its properties
def line1 : ℝ → ℝ := λ x, (-4 / 5) * x + 4

-- Definition of the line perpendicular to the given line with y-intercept -3
def line2 : ℝ → ℝ := λ x, (5 / 4) * x - 3

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, line2 x = 0) ∧ (line2 0 = -3) →
  (∃ x : ℝ, x = 12 / 5) :=
by
  sorry

end perpendicular_line_x_intercept_l350_350755


namespace simplify_sqrt_450_l350_350487

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350487


namespace simplify_sqrt_450_l350_350384

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350384


namespace fourth_competitor_jump_l350_350720

theorem fourth_competitor_jump :
  let first_jump := 22
  let second_jump := first_jump + 1
  let third_jump := second_jump - 2
  let fourth_jump := third_jump + 3
  fourth_jump = 24 := by
  sorry

end fourth_competitor_jump_l350_350720


namespace min_payment_max_payment_expected_value_payment_l350_350333

-- Proof Problem 1
theorem min_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let optimized_groups := [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]] in
  (∑ g in optimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4000 := by
  sorry

-- Proof Problem 2
theorem max_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let suboptimized_groups := [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]] in
  (∑ g in suboptimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4900 := by
  sorry

-- Proof Problem 3
theorem expected_value_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let expected_savings := 100 * (∑ k in List.range 9, k * ((10 - k) * (9 - k)) / 72) in
  (5500 - expected_savings) = 4583.33 := by 
  sorry

end min_payment_max_payment_expected_value_payment_l350_350333


namespace jars_of_pickled_mangoes_l350_350991

def total_mangoes := 54
def ratio_ripe := 1/3
def ratio_unripe := 2/3
def kept_unripe_mangoes := 16
def mangoes_per_jar := 4

theorem jars_of_pickled_mangoes : 
  (total_mangoes * ratio_unripe - kept_unripe_mangoes) / mangoes_per_jar = 5 :=
by
  sorry

end jars_of_pickled_mangoes_l350_350991


namespace sqrt_450_equals_15_sqrt_2_l350_350598

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350598


namespace sqrt_450_eq_15_sqrt_2_l350_350478

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350478


namespace remainder_div7_l350_350889

theorem remainder_div7 (x : ℕ) (h : x = 5^99 + ∑ i in finset.range 99, nat.choose 99 i * 5^(99 - 1 - i)) : x % 7 = 5 := 
sorry

end remainder_div7_l350_350889


namespace point_in_second_quadrant_l350_350017

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end point_in_second_quadrant_l350_350017


namespace hotel_flat_fee_l350_350810

theorem hotel_flat_fee (f n : ℝ) (h1 : f + n = 120) (h2 : f + 6 * n = 330) : f = 78 :=
by
  sorry

end hotel_flat_fee_l350_350810


namespace S_n_lt_2Sn_l350_350282

open Nat

def S (a : List ℕ) (n : ℕ) : ℕ :=
  List.countp (fun (x : List ℕ) => List.foldl (+) 0 (List.zipWith (*) a x) = n)
    (List.replicate (length a) (range (n + 1)))

theorem S_n_lt_2Sn
  (a : List ℕ)
  (hS : ∀ (n : ℕ), n > 0 → S a n ≠ 0) :
  ∃ N, ∀ n >= N, S a (n + 1) < 2 * S a n :=
by
  sorry

end S_n_lt_2Sn_l350_350282


namespace difference_of_squares_l350_350229

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x^2 + y^2 = 180) : |x^2 - y^2| = 108 :=
  sorry

end difference_of_squares_l350_350229


namespace sqrt_450_equals_15_sqrt_2_l350_350597

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350597


namespace simplify_sqrt_450_l350_350370

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350370


namespace sqrt_simplify_l350_350348

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350348


namespace simplify_sqrt_450_l350_350437

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350437


namespace sum_of_squares_ge_2_pow_36_l350_350276

open Nat

-- Define the conditions
def has_36_prime_divisors (n : ℕ) : Prop :=
  (n > 0) ∧ (Nat.card (Nat.finset.filter Nat.Prime (Nat.divisors n)) = 36)

def mutually_prime_count_in_interval (n : ℕ) (k : ℕ) (c : ℕ) : Prop :=
  let lower := (k - 1) * n / 5
  let upper := k * n / 5
  c = Nat.card { x ∈ {lower, lower + 1, ..., upper} | Nat.coprime x n }

def c_values_diff (c1 c2 c3 c4 c5 c_i c_j : ℕ) : Prop :=
  c_i ∈ [c1, c2, c3, c4, c5] ∧ c_j ∈ [c1, c2, c3, c4, c5] ∧ c_i ≠ c_j

-- State the theorem
theorem sum_of_squares_ge_2_pow_36 
  (n : ℕ)
  (h1 : has_36_prime_divisors n)
  (c1 c2 c3 c4 c5 : ℕ)
  (h2_k1 : mutually_prime_count_in_interval n 1 c1)
  (h2_k2 : mutually_prime_count_in_interval n 2 c2)
  (h2_k3 : mutually_prime_count_in_interval n 3 c3)
  (h2_k4 : mutually_prime_count_in_interval n 4 c4)
  (h2_k5 : mutually_prime_count_in_interval n 5 c5)
  (h3 : ∃ i j, c_values_diff c1 c2 c3 c4 c5 i j) :
  ∑ (1 ≤ i < j ≤ 5) (c_i - c_j)^2 ≥ 2^36 :=
  sorry

end sum_of_squares_ge_2_pow_36_l350_350276


namespace lift_ratio_l350_350971

theorem lift_ratio (total_weight first_lift second_lift : ℕ) (h1 : total_weight = 1500)
(h2 : first_lift = 600) (h3 : first_lift = 2 * (second_lift - 300)) : first_lift / second_lift = 1 := 
by
  sorry

end lift_ratio_l350_350971


namespace probability_multiple_4_or_15_l350_350000

open set 

-- The set of natural numbers from 1 to 30
def first30_nat : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Definition of multiples of a number within a set
def multiples_of (m : ℕ) (s : set ℕ) : set ℕ := {n ∈ s | ∃ k, n = m * k}

-- The set of multiples of 4 within the first 30 natural numbers
def multiples_of_4 : set ℕ := multiples_of 4 first30_nat

-- The set of multiples of 15 within the first 30 natural numbers
def multiples_of_15 : set ℕ := multiples_of 15 first30_nat

-- The union of multiples of 4 and 15 within the first 30 natural numbers
def multiples_of_4_or_15 : set ℕ := multiples_of_4 ∪ multiples_of_15

-- Counting function: size of a set with finite elements
def cardinality (s : set ℕ) [fintype s] : ℕ := fintype.card s

-- The probability that a number is a multiple of either 4 or 15
theorem probability_multiple_4_or_15 : 
  (cardinality multiples_of_4_or_15 : ℚ) / (cardinality first30_nat : ℚ) = 3 / 10 :=
  sorry

end probability_multiple_4_or_15_l350_350000


namespace handshakes_at_conference_l350_350021

theorem handshakes_at_conference (n : ℕ) (h : n = 10) :
  (nat.choose n 2) = 45 :=
by
  sorry

end handshakes_at_conference_l350_350021


namespace volume_calculation_correct_l350_350009

noncomputable def volume_of_solid_bounded_by_surfaces : ℝ :=
  let f (z : ℝ) : ℝ := if z ≤ 6 then π * z / 6 else 0
  ∫ z in 0..6, f z

theorem volume_calculation_correct : volume_of_solid_bounded_by_surfaces = 3 * π := by
  sorry

end volume_calculation_correct_l350_350009


namespace simplify_expression_l350_350702

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end simplify_expression_l350_350702


namespace simplify_sqrt_450_l350_350672

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350672


namespace domain_shift_l350_350192

theorem domain_shift (f : ℝ → ℝ) : (∀ x, (x + 1) ∈ Icc (-2 : ℝ) (3 : ℝ) → x ∈ Icc (-3 : ℝ) (2 : ℝ)) := 
by
  intro x h
  sorry

end domain_shift_l350_350192


namespace total_amount_paid_l350_350034

theorem total_amount_paid (num_sets : ℕ) (cost_per_set : ℕ) (tax_rate : ℝ) 
  (h1 : num_sets = 5) (h2 : cost_per_set = 6) (h3 : tax_rate = 0.1) : 
  let cost_before_tax := num_sets * cost_per_set
  let tax_amount := cost_before_tax * tax_rate
  let total_cost := cost_before_tax + tax_amount
  in total_cost = 33 :=
by
  sorry

end total_amount_paid_l350_350034


namespace simplify_sqrt_450_l350_350436

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350436


namespace power_function_has_minimum_value_at_point_l350_350196

definition power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_has_minimum_value_at_point : 
  ∃ α : ℝ, (power_function α (-2) = 4) ∧ (∀ x : ℝ, power_function α x ≥ 0) :=
by
  sorry

end power_function_has_minimum_value_at_point_l350_350196


namespace interns_survival_probability_l350_350878

-- Defining the problem
noncomputable def probability_of_survival (n : ℕ) (k : ℕ) : ℝ :=
  let num_permutations := nat.factorial n
  let favorable_permutations := finset.sum (finset.range (k+1)) 
    (λ m, nat.choose n m * nat.factorial (n - m) * nat.factorial (m - 1))
  in (favorable_permutations : ℝ) / num_permutations

-- Required proof statement:
theorem interns_survival_probability (n : ℕ) (k : ℕ) (h₁ : n = 44) (h₂ : k = 21) :
  probability_of_survival 44 21 > 0.3 :=
sorry

end interns_survival_probability_l350_350878


namespace min_lamps_l350_350815

theorem min_lamps (n p : ℕ) (h1: p > 0) (h_total_profit : 3 * (3 * p / 4 / n) + (n - 3) * (p / n + 10) - p = 100) : n = 13 :=
by
  sorry

end min_lamps_l350_350815


namespace inequality_holds_for_positive_y_l350_350040

theorem inequality_holds_for_positive_y (y : ℝ) (hy : y > 0) : y^2 ≥ 2 * y - 1 :=
by
  sorry

end inequality_holds_for_positive_y_l350_350040


namespace sqrt_simplify_l350_350361

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350361


namespace sqrt_simplify_l350_350351

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350351


namespace arithmetic_sequence_abs_sum_l350_350249

theorem arithmetic_sequence_abs_sum :
  ∀ {a : ℕ → ℤ}, 
    (a 5 = 3) →
    (a 10 = 18) →
    (∀ n m, a (n + m) = a n + a m) →
    (∑ i in Finset.range 10, |a (i + 1)|) = 78 :=
by
  intros a h₁ h₂ h₃
  sorry

end arithmetic_sequence_abs_sum_l350_350249


namespace ratio_largest_to_sum_l350_350089

theorem ratio_largest_to_sum (S : Finset ℝ) (hS : S = {1, 10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9}) :
  let largest := 10^9
  let sum_others := 1 + 10 + 10^2 + 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 in
  (largest / sum_others) = 9 :=
by
  let S := {1, 10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9}
  have hS : S = {1, 10, 10^2, 10^3, 10^4, 10^5, 10^6, 10^7, 10^8, 10^9} := rfl
  let largest := 10^9
  let sum_others := 1 + 10 + 10^2 + 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 
  have h1 : largest = 10^9 := rfl
  have h2 : sum_others = (1 + 10 + 10^2 + 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8) := rfl
  have S := sum_others
  calc (largest / S) = 9 := by sorry

end ratio_largest_to_sum_l350_350089


namespace gcd_sequence_property_l350_350730

theorem gcd_sequence_property (a : ℕ → ℕ) (m n : ℕ) (h : ∀ m n, m > n → Nat.gcd (a m) (a n) = Nat.gcd (a (m - n)) (a n)) : 
  Nat.gcd (a m) (a n) = a (Nat.gcd m n) :=
by
  sorry

end gcd_sequence_property_l350_350730


namespace remainder_of_349_divided_by_17_l350_350767

theorem remainder_of_349_divided_by_17 : 
  (349 % 17 = 9) := 
by
  sorry

end remainder_of_349_divided_by_17_l350_350767


namespace simplify_sqrt_450_l350_350388

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350388


namespace sqrt_450_eq_15_sqrt_2_l350_350468

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350468


namespace sqrt_450_simplified_l350_350419

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350419


namespace laura_change_sum_l350_350273

theorem laura_change_sum :
  ∀ (amount : ℕ), (amount < 100) → 
  ((∃ k, amount = 25*k + 5) ∧ (∃ m, amount = 10*m + 2)) →
  (amount = 30 ∨ amount = 80) →
  30 + 80 = 110 :=
by {
  -- Specify the necessary conditions and assumptions for the proof
  intros amount amount_lt_100 conditions valid_amount,
  -- Since the problem is focused on the conclusion, we postpone detailed proof (indicated by sorry)
  sorry
}

end laura_change_sum_l350_350273


namespace simplify_sqrt_450_l350_350516

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350516


namespace trajectory_of_Q_l350_350161

/-
Given points: A(0, 3), B(0, 6), C(0, -2), and D(0, 2),
point P is moving such that |PA|/|PB| = 1/2.
Let line l be the perpendicular bisector of PC, which intersects line PD at point Q.
Prove that the locus of point Q is given by the equation y^2 - x^2 / 3 = 1.
-/
theorem trajectory_of_Q :
  let A := (0, 3)
  let B := (0, 6)
  let C := (0, -2)
  let D := (0, 2)
  ∀ P : ℝ × ℝ,
    ((P.1 - A.1)^2 + (P.2 - A.2)^2).sqrt / ((P.1 - B.1)^2 + (P.2 - B.2)^2).sqrt = 1 / 2 →
    ∀ Q : ℝ × ℝ,
      -- l is the perpendicular bisector of PC
      (Q.1, (Q.2 - C.2).abs / sqrt ((Q.1 - C.1)^2 + (Q.2 - C.2)^2) = (Q.2 - P.2).abs / sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)) ∧
      -- Q is the intersection of line l with PD
      (Q.1 = (P.1 + D.1) / 2 ∧ Q.2 = (P.2 + D.2) / 2) → (Q.2 ^ 2 - Q.1 ^ 2 / 3 = 1) :=
begin
  intros A B C D P hPQ Q hQ,
  sorry -- Proof omitted
end

end trajectory_of_Q_l350_350161


namespace problem_sol_max_distance_from_circle_to_line_l350_350253

noncomputable def max_distance_circle_line : ℝ :=
  let ρ (θ : ℝ) : ℝ := 8 * Real.sin θ
  let line (θ : ℝ) : Prop := θ = Real.pi / 3
  let circle_center := (0, 4)
  let line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x
  let shortest_distance := 2  -- Already calculated in solution
  let radius := 4
  shortest_distance + radius

theorem problem_sol_max_distance_from_circle_to_line :
  max_distance_circle_line = 6 :=
by
  unfold max_distance_circle_line
  sorry

end problem_sol_max_distance_from_circle_to_line_l350_350253


namespace factorial_not_multiple_of_57_l350_350230

theorem factorial_not_multiple_of_57 (n : ℕ) (h : ¬ (57 ∣ n!)) : n < 19 := 
sorry

end factorial_not_multiple_of_57_l350_350230


namespace find_y_l350_350872

-- Define the vector projection function
def proj (w v : ℝ × ℝ) : ℝ × ℝ :=
  let dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)
  let norm_sq (a : ℝ × ℝ) : ℝ := dot a a
  let scalar := (dot v w) / (norm_sq w)
  (scalar * w.1, scalar * w.2)

theorem find_y (y : ℝ) :
  let v := (2, y)
  let w := (7, 5)
  proj w v = (-14, -10) → y = -32.4 :=
by
  intros
  let v := (2, y)
  let w := (7, 5)
  have h : proj w v = (-14, -10) := by assumption
  -- Rest of the proof skipped with sorry
  sorry

end find_y_l350_350872


namespace simplify_sqrt_450_l350_350450

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350450


namespace total_shaded_area_is_correct_l350_350791

-- Define the dimensions of the floor
def floor_length : ℝ := 12
def floor_width : ℝ := 15

-- Define the dimensions and characteristics of the tile
def tile_side : ℝ := 2
def quarter_circle_radius : ℝ := 1

-- Total area of the floor
def total_floor_area : ℝ := floor_length * floor_width

-- Total number of tiles
def number_of_tiles : ℝ := (floor_length / tile_side) * (floor_width / tile_side)

-- Area of one tile
def area_of_one_tile : ℝ := tile_side * tile_side

-- Area of one quarter circle
def area_of_one_quarter_circle : ℝ := (Mathlib.pi * (quarter_circle_radius ^ 2)) / 4

-- Area of the white pattern in one tile (four quarter circles make one full circle)
def area_of_white_pattern : ℝ := Mathlib.pi * (quarter_circle_radius ^ 2)

-- Shaded area in one tile
def area_of_shaded_tile : ℝ := area_of_one_tile - area_of_white_pattern

-- Total shaded area in the whole floor
def total_shaded_area : ℝ := number_of_tiles * area_of_shaded_tile

-- The mathematical statement to prove: the total shaded area is 180 - 45π square feet
theorem total_shaded_area_is_correct :
  total_shaded_area = 180 - 45 * Mathlib.pi :=
  sorry

end total_shaded_area_is_correct_l350_350791


namespace simplify_expression_l350_350701

theorem simplify_expression :
  sin (50 * (π / 180)) * (1 + sqrt 3 * tan (10 * (π / 180))) = (1 / 2) * cos (10 * (π / 180)) :=
by
  sorry

end simplify_expression_l350_350701


namespace solve_for_x_l350_350188

theorem solve_for_x (x : ℝ) (h : |2000 * x + 2000| = 20 * 2000) : x = 19 ∨ x = -21 := 
by
  sorry

end solve_for_x_l350_350188


namespace sqrt_simplify_l350_350354

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350354


namespace simplify_sqrt_450_l350_350375

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350375


namespace find_a_for_extreme_value_l350_350158

def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + 3 * x^2 + 3 * x + 3

def f' (a : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 + 6 * x + 3

theorem find_a_for_extreme_value :
  (∃ a : ℝ, f' a 1 = 0) → (a = -3) :=
by
  intro h
  cases h with a ha
  have : 3 * a * (1 : ℝ)^2 + 6 * (1 : ℝ) + 3 = 0 := ha
  sorry

end find_a_for_extreme_value_l350_350158


namespace quadrilateral_tangent_sphere_l350_350256

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Define points and distances
noncomputable def distance (x y : Type) [metric_space x] [metric_space y] := sorry

variable (AB CD AD BC AC : ℝ)

-- Given condition
def sides_sum_eq (AB CD AD BC : ℝ) : Prop := AB + CD = AD + BC

-- Statement of the problem
theorem quadrilateral_tangent_sphere (AB CD AD BC AC : ℝ) (h : sides_sum_eq AB CD AD BC) :
  ∃ S : Set (Set ℝ), (∀ x ∈ {A, B, C, D}, ∃ P, dist P x = S) ∧ (dist (AC) = S) :=
  sorry

end quadrilateral_tangent_sphere_l350_350256


namespace sqrt_450_simplified_l350_350433

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350433


namespace hexagon_angle_D_135_l350_350242

theorem hexagon_angle_D_135 
  (A B C D E F : ℝ)
  (h1 : A = B ∧ B = C)
  (h2 : D = E ∧ E = F)
  (h3 : A = D - 30)
  (h4 : A + B + C + D + E + F = 720) :
  D = 135 :=
by {
  sorry
}

end hexagon_angle_D_135_l350_350242


namespace simplify_sqrt_450_l350_350504

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350504


namespace simplify_sqrt_450_l350_350382

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350382


namespace johnny_marbles_l350_350990

noncomputable def johnnyMarbleSelections : ℕ :=
  nat.choose 9 3

theorem johnny_marbles : johnnyMarbleSelections = 84 := by
  -- nat.choose 9 3 is the equivalent to binomial coefficient "9 choose 3"
  -- comb(9, 3) = 9! / (3! * (9-3)!)
  -- which simplifies to (9 * 8 * 7) / (3 * 2 * 1)
  -- = 84
  sorry

end johnny_marbles_l350_350990


namespace number_of_integer_length_chords_through_point_l350_350804

theorem number_of_integer_length_chords_through_point
  (x y : ℝ) :
  let center_x := 0
      center_y := -1
      radius := 4
      p := (2 : ℝ, 2 : ℝ)
  in (x - center_x)^2 + (y - center_y)^2 = radius^2 →
       x = 2 ∧ y = 2 →
       ∃ n : ℕ, n = 9 := 
by
  sorry

end number_of_integer_length_chords_through_point_l350_350804


namespace sqrt_450_eq_15_sqrt_2_l350_350546

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350546


namespace intersection_of_sets_l350_350902

noncomputable def M : Set ℝ := { x | x - 2 > 0 }
noncomputable def N : Set ℝ := { x | log 2 (x - 1) < 1 }

theorem intersection_of_sets (x : ℝ) :
  (x ∈ M ∩ N) ↔ (2 < x ∧ x < 3) :=
by
  sorry

end intersection_of_sets_l350_350902


namespace books_returned_percentage_l350_350037

theorem books_returned_percentage :
  ∃ (L R : ℕ) (p : ℚ),
  L = 40 ∧
  R = 28 ∧
  p = (R / L) * 100 ∧
  p = 70 :=
by
  let L := 40
  let R := 28
  let p := (R / L) * 100
  have L_def : L = 40 := by rfl
  have R_def : R = 28 := by rfl
  have p_def : p = 70 := by
    calc
      p = (R / L) * 100 : by rfl
      ... = (28 / 40) * 100 : by rw [R_def, L_def]
      ... = 0.7 * 100 : by norm_num
      ... = 70 : by norm_num
  exact ⟨L, R, p, L_def, R_def, p_def, rfl⟩

end books_returned_percentage_l350_350037


namespace arithmetic_and_geometric_sequences_l350_350899

def arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d > 0, ∀ n, a_n n = 1 + (n - 1) * d

def geometric_sequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q, ∀ n, b_n n = b_n 1 * q ^ (n - 1)

def a_n := λ n, 2 * n - 1
def b_n := λ n, 3 ^ n

def c_n (n : ℕ) : ℚ :=
  1 / ( (a_n n + 3) * real.log (b_n n) / real.log 3)

def s_n (n : ℕ) : ℚ :=
  ∑ i in finset.range n, c_n i

theorem arithmetic_and_geometric_sequences :
∃ d > 0, (a_n 1 = 1) ∧ (a_n 2 = b_n 1) ∧ (a_n 5 = b_n 2) ∧ (a_n 14 = b_n 3) →
  (∀ n, a_n n = 2 * n - 1) ∧ (∀ n, b_n n = 3 ^ n) ∧ (∀ n, s_n n = n / (2 * (n + 1))) :=
by
  intro h
  use 2
  split
  -- Proofs will be written here
  sorry

end arithmetic_and_geometric_sequences_l350_350899


namespace sqrt_450_equals_15_sqrt_2_l350_350594

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350594


namespace min_winding_sum_is_24_hours_l350_350752

theorem min_winding_sum_is_24_hours (a b c d e : ℝ) :
  ∃ x : ℝ, 
    (∀ w ∈ {a, b, c, d, e}, w ≥ 0) → -- All winding amounts are non-negative.
    (∀ w₁ w₂ ∈ {a, b, c, d, e}, w₁ ≤ w₂ → (w₂ - w₁) ≤ 12) → -- Winding intervals are forward and within the 12-hour limit.
  (|a - b| + |a - c| + |a - d| + |a - e| + -- Sum of winding intervals from 'a' to others
   |b - c| + |b - d| + |b - e| + -- Sum of winding intervals from 'b' to others
   |c - d| + |c - e| + -- Sum of winding intervals from 'c' to others
   |d - e|   -- Sum of winding intervals from 'd' to 'e'
  ) = 24 :=
sorry

end min_winding_sum_is_24_hours_l350_350752


namespace length_of_bridge_l350_350783

theorem length_of_bridge (length_train : ℕ) (speed_train_kmh : ℕ) (crossing_time_sec : ℕ)
    (h_length_train : length_train = 125)
    (h_speed_train_kmh : speed_train_kmh = 45)
    (h_crossing_time_sec : crossing_time_sec = 30) : 
    ∃ (length_bridge : ℕ), length_bridge = 250 := by
  sorry

end length_of_bridge_l350_350783


namespace min_payment_max_payment_expected_value_payment_l350_350334

-- Proof Problem 1
theorem min_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let optimized_groups := [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]] in
  (∑ g in optimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4000 := by
  sorry

-- Proof Problem 2
theorem max_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let suboptimized_groups := [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]] in
  (∑ g in suboptimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4900 := by
  sorry

-- Proof Problem 3
theorem expected_value_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let expected_savings := 100 * (∑ k in List.range 9, k * ((10 - k) * (9 - k)) / 72) in
  (5500 - expected_savings) = 4583.33 := by 
  sorry

end min_payment_max_payment_expected_value_payment_l350_350334


namespace find_probability_l350_350893

noncomputable def normal_dist (mean variance : ℝ) : Type := sorry

noncomputable def P (X : ℝ → ℝ) (e : set ℝ) : ℝ := sorry

-- Conditions
variables {X : ℝ → ℝ} {a : ℝ} {σ : ℝ}
hypothesis hX : X ∼ normal_dist 3 (σ^2)
hypothesis hPa : P X {x | x > a} = 0.2

-- Goal
theorem find_probability : P X {x | x > 6 - a} = 0.8 := by
  sorry

end find_probability_l350_350893


namespace simplify_sqrt_450_l350_350442

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350442


namespace domino_tiling_l350_350337

theorem domino_tiling (a b a' b' : ℕ) : 
  ∃ (l w : ℕ), (l = max (1) (max (2 * a) (2 * a')) ∧ w = 4 * b * b') :=
begin
  sorry
end

end domino_tiling_l350_350337


namespace simplify_sqrt_450_l350_350398

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350398


namespace number_of_lines_l350_350245

-- Definitions for the points A and B and distances
def pointA : ℝ × ℝ := (1, 1)
def pointB : ℝ × ℝ := (-2, -3)

-- Predicates to represent the distance conditions
def dist_from_point (P Q : ℝ × ℝ) (d : ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = d^2

-- Statement of the theorem
theorem number_of_lines : ∃! L : set (ℝ × ℝ), 
  (∀ P ∈ L, dist_from_point P pointA 1) ∧ (∀ P ∈ L, dist_from_point P pointB 6) :=
sorry

end number_of_lines_l350_350245


namespace sqrt_450_eq_15_sqrt_2_l350_350586

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350586


namespace maximum_trips_bound_l350_350241

noncomputable def maxTrips : ℕ := 77

theorem maximum_trips_bound 
  (students : ℕ)
  (trip_days : ℕ)
  (students_per_trip : ℕ → ℕ)
  (nature_trip : ℕ → Prop)
  (museum_trip : ℕ → Prop)
  (different_num_students : ∀ i j, i ≠ j → students_per_trip i ≠ students_per_trip j)
  (no_repeat : ∀ x trip1 trip2, nature_trip trip1 → museum_trip trip2 → trip1 ≠ trip2 → x ∉ {trip1, trip2})
  (max_students : students = 2022) :
  trip_days ≤ maxTrips :=
by
  sorry

end maximum_trips_bound_l350_350241


namespace number_of_tens_in_sum_l350_350216

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l350_350216


namespace bike_price_l350_350306

theorem bike_price (x : ℝ) (h1 : 0.1 * x = 150) : x = 1500 := 
by sorry

end bike_price_l350_350306


namespace sqrt_450_eq_15_sqrt_2_l350_350581

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350581


namespace total_amount_paid_l350_350035

theorem total_amount_paid (num_sets : ℕ) (cost_per_set : ℕ) (tax_rate : ℝ) 
  (h1 : num_sets = 5) (h2 : cost_per_set = 6) (h3 : tax_rate = 0.1) : 
  let cost_before_tax := num_sets * cost_per_set
  let tax_amount := cost_before_tax * tax_rate
  let total_cost := cost_before_tax + tax_amount
  in total_cost = 33 :=
by
  sorry

end total_amount_paid_l350_350035


namespace simplify_sqrt_450_l350_350537

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350537


namespace simplify_sqrt_450_l350_350538

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350538


namespace number_of_tens_in_sum_l350_350220

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l350_350220


namespace sum_is_4_div_3_l350_350086

noncomputable def series_exponential_sum : ℝ :=
  ∑' (j : ℕ), ∑' (k : ℕ), 2^(-(4*k + 2*j + (k + j)^2))

theorem sum_is_4_div_3 : series_exponential_sum = 4 / 3 := 
  sorry

end sum_is_4_div_3_l350_350086


namespace arithmetic_progression_x_l350_350716

theorem arithmetic_progression_x (x : ℝ) :
  let a := x^2 - 3
      b := x^2 + 1
      c := 2x^2 - 1
  in (b - a = 4) → (c - b = 4) → (x = sqrt 6 ∨ x = -sqrt 6) :=
by
  intros a b c hab hbc
  sorry

end arithmetic_progression_x_l350_350716


namespace fourth_competitor_jump_distance_l350_350722

theorem fourth_competitor_jump_distance (a b c d : ℕ) 
    (h1 : a = 22) 
    (h2 : b = a + 1)
    (h3 : c = b - 2)
    (h4 : d = c + 3):
    d = 24 :=
by
  rw [h1, h2, h3, h4]
  sorry

end fourth_competitor_jump_distance_l350_350722


namespace snail_climb_days_l350_350822

def well_height : ℝ := 1.1
def climb_day : ℝ := 0.4
def slip_night : ℝ := 0.2
def net_climb_per_day : ℝ := climb_day - slip_night
def last_climb : ℝ := climb_day

theorem snail_climb_days : 
  ∀ (well_height climb_day slip_night : ℝ), 
    well_height = 1.1 → climb_day = 0.4 → slip_night = 0.2 →
    (0.7 / (climb_day - slip_night)).ceil + 1 = 4 :=
by
  intros well_height climb_day slip_night h_well_height h_climb_day h_slip_night
  sorry

end snail_climb_days_l350_350822


namespace simplify_sqrt_450_l350_350545

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350545


namespace inclination_angle_of_line_equation_l350_350718

noncomputable def inclination_angle_of_line (a b c : ℝ) (h : a + b ≠ 0) : ℝ :=
if h : b ≠ 0 then
  real.arctan (-a / b)
else if h' : a ≠ 0 then
  if a > 0 then real.pi / 2 else -real.pi / 2
else 0

theorem inclination_angle_of_line_equation :
  inclination_angle_of_line 1 1 1 1 ≠ 0 = 3 * real.pi / 4 :=
begin
  sorry
end

end inclination_angle_of_line_equation_l350_350718


namespace PQ_sum_l350_350184

theorem PQ_sum (P Q : ℕ) (h1 : 5 / 7 = P / 63) (h2 : 5 / 7 = 70 / Q) : P + Q = 143 :=
by
  sorry

end PQ_sum_l350_350184


namespace find_yxx_l350_350114

-- Conditions: Definitions of x and y parametrically
def x (t : ℝ) : ℝ := Real.exp t
def y (t : ℝ) : ℝ := Real.arcsin t

-- Function to compute first derivations of x and y with respect to t
def x'_t (t : ℝ) : ℝ := Real.exp t
def y'_t (t : ℝ) : ℝ := 1 / Real.sqrt (1 - t^2)

-- Function to compute the first derivation of y with respect to x
def y'_x (t : ℝ) : ℝ := (y'_t t) / (x'_t t)

-- Function to compute y'' in terms of t
def y''_tx (t : ℝ) : ℝ := 
  -((1 - t^2 - t) / (Real.exp(2 * t) * Real.sqrt (1 - t^2) * (1 - t^2)))

-- Final function of the second derivation y with respect to x
def y''_xx (t : ℝ) : ℝ := (y''_tx t) / (x'_t t)

-- The proof statement
theorem find_yxx'' (t : ℝ) : y''_xx t = (t^2 + t - 1) / (Real.exp (3 * t) * Real.sqrt ((1 - t^2)^3)) :=
by 
  sorry

end find_yxx_l350_350114


namespace find_expression_for_a_n_l350_350136

-- Definitions for conditions in the problem
variable (a : ℕ → ℝ) -- Sequence is of positive real numbers
variable (S : ℕ → ℝ) -- Sum of the first n terms of the sequence

-- Condition that all terms in the sequence a_n are positive and indexed by natural numbers starting from 1
axiom pos_seq : ∀ n : ℕ, 0 < a (n + 1)
-- Condition for the sum of the terms: 4S_n = a_n^2 + 2a_n for n ∈ ℕ*
axiom sum_condition : ∀ n : ℕ, 4 * S (n + 1) = (a (n + 1))^2 + 2 * a (n + 1)

-- Theorem stating that sequence a_n = 2n given the above conditions
theorem find_expression_for_a_n : ∀ n : ℕ, a (n + 1) = 2 * (n + 1) := by
  sorry

end find_expression_for_a_n_l350_350136


namespace part1_part2_l350_350154

-- Part 1: Proving the value of a given f(x) = a/x + 1 and f(-2) = 0
theorem part1 (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a / x + 1) (h2 : f (-2) = 0) : a = 2 := 
by 
-- Placeholder for the proof
sorry

-- Part 2: Proving the value of f(4) given f(x) = 6/x + 1
theorem part2 (f : ℝ → ℝ) (h1 : ∀ x, f x = 6 / x + 1) : f 4 = 5 / 2 := 
by 
-- Placeholder for the proof
sorry

end part1_part2_l350_350154


namespace simplify_sqrt_450_l350_350526

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350526


namespace simplify_sqrt_450_l350_350485

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350485


namespace sqrt_simplify_l350_350359

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350359


namespace fourth_competitor_jump_l350_350721

theorem fourth_competitor_jump :
  let first_jump := 22
  let second_jump := first_jump + 1
  let third_jump := second_jump - 2
  let fourth_jump := third_jump + 3
  fourth_jump = 24 := by
  sorry

end fourth_competitor_jump_l350_350721


namespace sum_of_roots_l350_350843

theorem sum_of_roots (P : Polynomial ℝ) (h : P = 3 * X^3 + 2 * X^2 + 5 * X - 12) : 
  (P.coeffs.sum = -2/3) := 
by
  -- sorry is used here to skip the proof
  sorry

end sum_of_roots_l350_350843


namespace inequality_solution_l350_350873

theorem inequality_solution (m : ℝ) (x : ℝ) (hm : 0 ≤ m ∧ m ≤ 1) (ineq : m * x^2 - 2 * x - m ≥ 2) : x ≤ -1 :=
sorry

end inequality_solution_l350_350873


namespace Riemann_integrable_f_l350_350347

noncomputable def f : ℝ → ℝ :=
λ x, if x = 0 then 1 else (Real.sin x) / x

theorem Riemann_integrable_f :
  (Riemann_integral (λ x, if x = 0 then 1 else (Real.sin x) / x) (Set.Ici 0)) = (real.pi / 2) ∧
  ¬(Integrable (λ x, if x = 0 then 1 else (Real.sin x) / x) (Volume.measurableSet_Ici 0)) :=
sorry

end Riemann_integrable_f_l350_350347


namespace simplify_sqrt_450_l350_350676

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350676


namespace impossible_tiling_of_chessboard_l350_350837

theorem impossible_tiling_of_chessboard : 
  ¬ ∃ (tiling : Tiling (8 * 8 - 2) (1 * 2)), 
        ∀ (i j : ℕ) (hij : i < 8 ∧ j < 8 ∧ (i, j) ≠ (1, 1) ∧ (i, j) ≠ (8, 8)), 
          tiling.covers i j :=
sorry

end impossible_tiling_of_chessboard_l350_350837


namespace simplify_sqrt_450_l350_350651
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350651


namespace mary_shirts_left_l350_350305

theorem mary_shirts_left (B_s B_g R_s R_g BR_s BR_g Y_s Y_g : ℤ)
                         (hB_s : B_s = 30) (hB_g : B_g = 3 / 5)
                         (hBR_s : BR_s = 40) (hBR_g : BR_g = 1 / 4)
                         (hR_s : R_s = 20) (hR_g : R_g = 2 / 3)
                         (hY_s : Y_s = 25) (hY_g : Y_g = 1 / 5) :
  (B_s - (B_g * B_s) + BR_s - (BR_g * BR_s) + R_s - (R_g * R_s) + Y_s - (Y_g * Y_s)) = 69 :=
by
  assume hB_s : B_s = 30,
  assume hY_s : Y_s = 25,
  assume hBR_s : BR_s = 40,
  assume hR_s : R_s = 20,
  assume hB_g : B_g = 3 / 5,
  assume hBR_g : BR_g = 1 / 4,
  assume hR_g : R_g = 2 / 3,
  assume hY_g : Y_g = 1 / 5,
  sorry

end mary_shirts_left_l350_350305


namespace sin_C_is_one_l350_350907

-- Given values and conditions
def a : ℝ := 1
def b : ℝ := Real.sqrt 3
def A : ℝ 
def B : ℝ 
def C : ℝ 

-- Conditions
axiom angle_sum : A + B + C = Real.pi
axiom angle_relation : A + C = 2 * B

-- Proving the value of sin(C)
theorem sin_C_is_one (hA : A + B + C = Real.pi) (hAC : A + C = 2 * B) : Real.sin C = 1 := by
  sorry

end sin_C_is_one_l350_350907


namespace number_of_tens_in_sum_l350_350219

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l350_350219


namespace sqrt_450_simplified_l350_350617

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350617


namespace max_k_value_l350_350719

noncomputable def maximum_k : ℝ := exp 1

theorem max_k_value :
  (∀ x : ℝ, exp x ≥ maximum_k * x) ↔ maximum_k = exp 1 :=
by
  sorry

end max_k_value_l350_350719


namespace combination_sum_formula_l350_350880

theorem combination_sum_formula
  (n k m : ℕ) (hn : 0 < n) (hk : 0 < k) (hm : 0 < m) (hkm : 1 ≤ k ∧ k < m ∧ m ≤ n) :
  (finset.range (k+1)).sum (λ i, nat.choose k i * nat.choose n (m - i)) = nat.choose (n + k) m :=
sorry

end combination_sum_formula_l350_350880


namespace simplify_sqrt_450_l350_350539

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350539


namespace sqrt_450_simplified_l350_350629

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350629


namespace password_guess_probability_l350_350045

def probability_correct_digit_within_two_attempts : Prop :=
  let total_digits := 10
  let prob_first_attempt := 1 / total_digits
  let prob_second_attempt := (9 / total_digits) * (1 / (total_digits - 1))
  (prob_first_attempt + prob_second_attempt) = 1 / 5

theorem password_guess_probability :
  probability_correct_digit_within_two_attempts :=
by
  -- proof goes here
  sorry

end password_guess_probability_l350_350045


namespace sqrt_450_eq_15_sqrt_2_l350_350698

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350698


namespace simplify_sqrt_450_l350_350438

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350438


namespace sqrt_450_eq_15_sqrt_2_l350_350564

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350564


namespace range_of_a_l350_350150

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 
  a * x + 1 - 4 * a 
else 
  x ^ 2 - 3 * a * x

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) → 
  a ∈ (Set.Ioi (2/3) ∪ Set.Iic 0) :=
sorry

end range_of_a_l350_350150


namespace sqrt_450_equals_15_sqrt_2_l350_350605

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350605


namespace simplify_sqrt_450_l350_350634
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350634


namespace largest_square_perimeter_l350_350042

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end largest_square_perimeter_l350_350042


namespace polar_symmetry_l350_350254

theorem polar_symmetry (ρ θ : ℝ) :
  symmetry_about_pole ρ θ = (ρ, θ + π) :=
by
  sorry

def symmetry_about_pole (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ, θ + π)

end polar_symmetry_l350_350254


namespace squares_nailable_l350_350311

theorem squares_nailable {k : ℕ} {n : ℕ} (h1 : n = k * k) 
  (h2 : ∀ (squares : fin k → ℕ),
    ∃ i j : fin k, i ≠ j ∧ (squares i = squares j)) 
  : ∃ color : fin k, ∀ (squares : fin k → ℕ), 
    (∀ i : fin k, squares i < n) → ∃ nails : ℕ, nails ≤ 2 * k - 2 := 
sorry

end squares_nailable_l350_350311


namespace part1_part2_part3_l350_350928
-- Importing the necessary libraries

-- Define function f
def f (x a : ℝ) : ℝ := x - log (x + a)

-- Statement for part (1): Prove the value of a
theorem part1 (a : ℝ) (h1 : f (1 - a) a = 0) (h2 : 0 < a) : a = 1 := 
sorry

-- Statement for part (2): Prove the minimum value of k
theorem part2 (k : ℝ) (h : ∀ x, 0 ≤ x → f x 1 ≤ k * x^2) : k = 0.5 :=
sorry

-- Statement for part (3): Prove the inequality
theorem part3 (n : ℕ) (h : 0 < n) : 
    (∑ i in Finset.range n, 2 / (2 * i + 1)) - log (2 * n + 1) < 2 :=
sorry

end part1_part2_part3_l350_350928


namespace battery_lasts_12_more_hours_l350_350271

-- Define initial conditions
def standby_battery_life : ℕ := 36
def active_battery_life : ℕ := 4
def total_time_on : ℕ := 12
def active_usage_time : ℕ := 90  -- in minutes

-- Conversion and calculation functions
def active_usage_hours : ℚ := active_usage_time / 60
def standby_consumption_rate : ℚ := 1 / standby_battery_life
def active_consumption_rate : ℚ := 1 / active_battery_life
def battery_used_standby : ℚ := (total_time_on - active_usage_hours) * standby_consumption_rate
def battery_used_active : ℚ := active_usage_hours * active_consumption_rate
def total_battery_used : ℚ := battery_used_standby + battery_used_active
def remaining_battery : ℚ := 1 - total_battery_used
def additional_hours_standby : ℚ := remaining_battery / standby_consumption_rate

-- Proof statement
theorem battery_lasts_12_more_hours : additional_hours_standby = 12 := by
  sorry

end battery_lasts_12_more_hours_l350_350271


namespace simplify_sqrt_450_l350_350410

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350410


namespace number_of_tens_in_sum_l350_350221

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l350_350221


namespace simplify_sqrt_450_l350_350441

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350441


namespace area_difference_correct_l350_350847

def square_diagonal := 25
def rect_diagonal := 30
def aspect_ratio_w_h := (4, 3)
def expected_difference := 119.5

theorem area_difference_correct (s d1 : ℝ) (d2 : ℝ) (ratio_w : ℝ) (ratio_h : ℝ) (expected : ℝ) :
  s = square_diagonal →
  d1 = rect_diagonal →
  ratio_w / ratio_h = 4 / 3 →
  expected = expected_difference →
  let side := s / (2).sqrt in
  let area_square := side^2 in
  let h := (d2^2 / (ratio_w^2 + ratio_h^2)).sqrt in
  let w := ratio_w / ratio_h * h in
  let area_rectangular := w * h in
  area_rectangular - area_square = expected :=
begin
  intros hs hd1 hr he,
  rw [hs, hd1, he],
  sorry
end

end area_difference_correct_l350_350847


namespace sqrt_450_simplified_l350_350625

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350625


namespace number_of_tens_in_sum_l350_350218

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l350_350218


namespace sum_m_n_is_55_l350_350281

theorem sum_m_n_is_55 (a b c : ℝ) (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1)
  (h1 : 5 / a = b + c) (h2 : 10 / b = c + a) (h3 : 13 / c = a + b) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : (a + b + c) = m / n) : m + n = 55 :=
  sorry

end sum_m_n_is_55_l350_350281


namespace ferry_distance_A_to_B_l350_350012

theorem ferry_distance_A_to_B :
  ∀ (x y : ℝ), (1 : ℝ), (40 : ℝ), (24 : ℝ),
  (∃ (z : ℝ), z = 43 / 18) →
  (∃ y, 101 * x = 126 * y ∧ y = 56) →
  (40 + 24 = 64 ∧ 40 - 24 = 16) →
  x = 192 :=
by
  intros x y one forty twentyfour
  exists sorry
  sorry    

end ferry_distance_A_to_B_l350_350012


namespace min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350328

/- Part (a): Minimum amount the customer will pay -/
theorem min_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4000 := 
sorry

/- Part (b): Maximum amount the customer will pay -/
theorem max_amount_to_pay : 
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4900 := 
sorry

/- Part (c): Expected value the customer will pay -/
theorem expected_amount_to_pay :
  (∑ i in {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000} \ 0, id i) = 4583.33 := 
sorry

end min_amount_to_pay_max_amount_to_pay_expected_amount_to_pay_l350_350328


namespace sqrt_450_eq_15_sqrt_2_l350_350694

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350694


namespace simplify_sqrt_450_l350_350445

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350445


namespace expression_equals_31_l350_350841

noncomputable def calculate_expression : ℝ :=
  (1 / Real.sqrt 0.25) + ((1 / 27) ^ (-1/3 : ℝ)) + Real.sqrt((Real.logBase 3 3) ^ 2 - Real.logBase 3 9 + 1) - Real.logBase 3 (1 / 3) + 81 ^ (0.5 * Real.logBase 3 5)

theorem expression_equals_31 : calculate_expression = 31 :=
  sorry

end expression_equals_31_l350_350841


namespace simplify_sqrt_450_l350_350376

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350376


namespace number_of_ways_to_place_digits_l350_350979

theorem number_of_ways_to_place_digits : 
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let placements := {g : Fin 4 → Fin 2 → ℕ // ∀ i, (∑ j, g i j) + 1 = (∑ j, g (i+1) j)}
  (∀ g ∈ placements, (∀ (i j : Fin 4), (g i j) ∈ digits)) →
  card placements = 64 :=
by
  sorry

end number_of_ways_to_place_digits_l350_350979


namespace part1_part2_l350_350939

-- Definition of the given vectors a and b
def vec_a := (2 : ℝ, 4 : ℝ)
def vec_b (t : ℝ) := (1 : ℝ, t)

-- Part 1: Prove t = 2 given (a + b) is parallel to (a - b)
theorem part1 (t : ℝ) :
  (vec_a.1 + vec_b t.1, vec_a.2 + vec_b t.2).1 / (vec_a.1 - vec_b t.1) = 
  (vec_a.2 + vec_b t.2) / (vec_a.2 - vec_b t.2) → t = 2 :=
sorry

-- Part 2: Prove range of m is (-10/3, 0) ∪ (0, ∞)
theorem part2 (m : ℝ) :
  (vec_b 1).2 > 20 + 6 * m → -10 / 3 < m ∧ m ≠ 0 :=
sorry

end part1_part2_l350_350939


namespace min_payment_proof_max_payment_proof_expected_payment_proof_l350_350324

noncomputable def items : List ℕ := List.range 1 11 |>.map (λ n => n * 100)

def min_amount_paid : ℕ :=
  (1000 + 900 + 700 + 600 + 400 + 300 + 100)

def max_amount_paid : ℕ :=
  (1000 + 900 + 800 + 700 + 600 + 500 + 400)

def expected_amount_paid : ℚ :=
  4583 + 33 / 100

theorem min_payment_proof :
  (∑ x in (List.range 15).filter (λ x => x % 3 ≠ 0), (items.get! x : ℕ)) = min_amount_paid := by
  sorry

theorem max_payment_proof :
  (∑ x in List.range 10, if x % 3 = 0 then 0 else (items.get! x : ℕ)) = max_amount_paid := by
  sorry

theorem expected_payment_proof :
  ∑ k in items, ((k : ℚ) * (∏ m in List.range 9, (10 - m) * (9 - m) / 72)) = expected_amount_paid := by
  sorry

end min_payment_proof_max_payment_proof_expected_payment_proof_l350_350324


namespace evaluate_expression_l350_350074

theorem evaluate_expression :
  (|(-3 : ℝ)| + (real.sqrt 3 * real.sin (real.pi / 3)) - (2 : ℝ)⁻¹) = 4 :=
by
  sorry

end evaluate_expression_l350_350074


namespace sqrt_450_eq_15_sqrt_2_l350_350475

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350475


namespace sqrt_simplify_l350_350367

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350367


namespace sqrt_450_eq_15_sqrt_2_l350_350548

theorem sqrt_450_eq_15_sqrt_2 : sqrt 450 = 15 * sqrt 2 := by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350548


namespace kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l350_350272

/-- Conditions: -/
def kitchen_clock_gain_rate : ℝ := 1.5 -- minutes per hour
def bedroom_clock_lose_rate : ℝ := 0.5 -- minutes per hour
def synchronization_time : ℝ := 0 -- time in hours when both clocks were correct

/-- Problem 1: -/
theorem kitchen_clock_correct_again :
  ∃ t : ℝ, 1.5 * t = 720 :=
by {
  sorry
}

/-- Problem 2: -/
theorem bedroom_clock_correct_again :
  ∃ t : ℝ, 0.5 * t = 720 :=
by {
  sorry
}

/-- Problem 3: -/
theorem both_clocks_same_time_again :
  ∃ t : ℝ, 2 * t = 720 :=
by {
  sorry
}

end kitchen_clock_correct_again_bedroom_clock_correct_again_both_clocks_same_time_again_l350_350272


namespace inequality_proof_l350_350877

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_geq : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
sorry

end inequality_proof_l350_350877


namespace product_of_roots_quadratic_l350_350868

theorem product_of_roots_quadratic :
  ∀ (a b c : ℚ), a ≠ 0 → 
  a = 24 → b = 60 → c = -750 → 
  (let product_of_roots := (c / a) 
  in product_of_roots = -125 / 4) :=
by
  intros a b c h a_eq b_eq c_eq
  have h_eq : a = 24 ∧ b = 60 ∧ c = -750 := ⟨a_eq, b_eq, c_eq⟩
  suffices h_root : c / a = -125 / 4 by
    exact h_root
  subst_vars
  sorry

end product_of_roots_quadratic_l350_350868


namespace integer_solutions_inequalities_l350_350175

theorem integer_solutions_inequalities :
  {y : ℤ | (2 : ℤ) * y ≤ -y + 4 ∧ (5 : ℤ) * y ≥ -10 ∧ (3 : ℤ) * y ≤ -2 * y + 20}.finite.card = 4 :=
by
  sorry

end integer_solutions_inequalities_l350_350175


namespace sqrt_simplify_l350_350362

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350362


namespace sqrt_450_eq_15_sqrt_2_l350_350699

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350699


namespace simplify_sqrt_450_l350_350523

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350523


namespace hundredth_smallest_integer_satisfying_congruences_l350_350085

theorem hundredth_smallest_integer_satisfying_congruences :
  ∃ n : ℕ, (∀ k : ℕ, (set.range(λ k : ℕ, 8 * (4 * k + 3) ≤ n ∧ n < 8 * (4 * k + 4))) →
  ∀ m : ℕ, (set.range(λ m : ℕ, 32 * (4 * m + 2) ≤ n ∧ n < 32 * (4 * m + 3))) →
  ∀ p : ℕ, (set.range(λ p : ℕ, 256 * (4 * p + 1) ≤ n ∧ n < 256 * (4 * p + 2))) →
  n = 100 → n = 6491)) :=
  sorry

end hundredth_smallest_integer_satisfying_congruences_l350_350085


namespace distinct_dice_designs_l350_350750

-- Define the problem in Lean
theorem distinct_dice_designs : 
  let faces := {1, 2, 3, 4, 5, 6},
      opposite_pairs := [(1, 2), (3, 4), (5, 6)] ∨ 
                        [(1, 2), (3, 5), (4, 6)] ∨
                        [(1, 2), (3, 6), (4, 5)] in
  ∃ (faces : Finset ℕ) 
    (opposite_pairs : List (ℕ × ℕ))
    (colors : faces → Bool),
  (∀ (x, y) ∈ opposite_pairs, x ∈ faces ∧ y ∈ faces ∧ x ≠ y) ∧
  (∀ (x, y) ∈ opposite_pairs, colors x = colors y) ∧
  (colors 1 ≠ colors 2) ∧
  1 + opposite_pairs.length * 2^3 = 48 :=
by
  sorry

end distinct_dice_designs_l350_350750


namespace hare_wolf_distance_possible_l350_350797

noncomputable def hare_speed : ℝ := 5 -- meters per second
noncomputable def wolf_speed : ℝ := 3 -- meters per second
noncomputable def track_length : ℝ := 200 -- meters
noncomputable def time_interval : ℝ := 40 -- seconds

def is_possible_distance (d : ℝ) : Prop :=
  d = 40 ∨ d = 60

theorem hare_wolf_distance_possible : ∃ d : ℝ, is_possible_distance d :=
begin
  -- The proof should show that the distance can be either 40 meters or 60 meters,
  -- given the Hare's speed, the Wolf's speed, the track length, and the time interval.
  sorry
end

end hare_wolf_distance_possible_l350_350797


namespace total_guitars_sold_l350_350233

theorem total_guitars_sold (total_revenue : ℕ) (price_electric : ℕ) (price_acoustic : ℕ)
  (num_electric_sold : ℕ) (num_acoustic_sold : ℕ) 
  (h1 : total_revenue = 3611) (h2 : price_electric = 479) 
  (h3 : price_acoustic = 339) (h4 : num_electric_sold = 4) 
  (h5 : num_acoustic_sold * price_acoustic + num_electric_sold * price_electric = total_revenue) :
  num_electric_sold + num_acoustic_sold = 9 :=
sorry

end total_guitars_sold_l350_350233


namespace sqrt_450_eq_15_sqrt_2_l350_350573

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350573


namespace simplify_sqrt_450_l350_350532

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350532


namespace tetrahedron_volume_surface_area_l350_350138

noncomputable def volume_and_surface_area (a b c : ℝ) (cond1 : sqrt (a^2 + b^2) = 5) (cond2 : sqrt (b^2 + c^2) = sqrt 34) (cond3 : sqrt (a^2 + c^2) = sqrt 41) :
    (V : ℝ) × (S : ℝ) :=
⟨(1/3) * a * b * c, 4 * Real.pi * ((sqrt (a^2 + b^2 + c^2) / 2) ^ 2)⟩

theorem tetrahedron_volume_surface_area :
    ∃ a b c : ℝ, (sqrt (a^2 + b^2) = 5) ∧ (sqrt (b^2 + c^2) = sqrt 34) ∧ (sqrt (a^2 + c^2) = sqrt 41) ∧
    (volume_and_surface_area a b c
      (by simp [Real.sqrt_sq,abs_of_nonneg]; linarith)
      (by simp [Real.sqrt_sq,abs_of_nonneg]; linarith)
      (by simp [Real.sqrt_sq,abs_of_nonneg]; linarith) =
      (20, 50 * Real.pi)) :=
by
  use 4, 3, 5
  simp [volume_and_surface_area]
  split; linarith
  split; linarith
  split; linarith
  split; linarith
  sorry

end tetrahedron_volume_surface_area_l350_350138


namespace sum_of_tens_l350_350201

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l350_350201


namespace sqrt_450_eq_15_sqrt_2_l350_350460

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350460


namespace probability_point_below_curve_in_square_l350_350248

-- Define the curve y = x^2
def curve (x : ℝ) : ℝ := x^2

-- Define the probability calculation
def probability_below_curve : ℝ := 
  (∫ x in 0..1, curve x) / (1 * 1)

-- The theorem statement
theorem probability_point_below_curve_in_square : 
  probability_below_curve = 1 / 3 :=
by
  sorry

end probability_point_below_curve_in_square_l350_350248


namespace point_in_first_quadrant_l350_350186

theorem point_in_first_quadrant (a : ℝ) (h : a < 0) : ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (-a, a^2) = (x, y) := 
by {
  let x := -a,
  let y := a^2,
  have hx : x > 0 := by exact neg_pos_of_neg h,
  have hy : y > 0 := by exact pow_pos (lt_of_le_of_ne (le_refl a) (ne_of_lt h)) 2,
  use [x, y],
  exact ⟨hx, hy, rfl⟩
}

end point_in_first_quadrant_l350_350186


namespace length_of_train_l350_350002

theorem length_of_train (speed_kmh : ℕ) (crossing_time_s : ℕ) (converted_speed : ℚ)
  (h1 : speed_kmh = 27) (h2 : crossing_time_s = 20) (h3 : converted_speed = speed_kmh * 5 / 18) :
  converted_speed * crossing_time_s = 150 :=
begin
  -- proof will go here
  sorry
end

end length_of_train_l350_350002


namespace sqrt_450_eq_15_sqrt_2_l350_350692

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350692


namespace distinct_round_table_arrangements_l350_350978

theorem distinct_round_table_arrangements (n : ℕ) (h : n = 8) : 
  (nat.factorial n) / n = nat.factorial (n - 1) := by
  sorry

end distinct_round_table_arrangements_l350_350978


namespace simplify_sqrt_450_l350_350379

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350379


namespace no_common_points_circle_ray_l350_350145

theorem no_common_points_circle_ray (a : ℝ) :
  (∀ (x y : ℝ), (x - a)^2 + y^2 ≠ 4 ∨ y ≠ sqrt(3) * x ∨ x < 0) ↔ (a < -2 ∨ a > (4/3) * sqrt 3) := 
sorry

end no_common_points_circle_ray_l350_350145


namespace sum_of_tens_l350_350199

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : (n / 10) = 10^19 := by
  sorry

end sum_of_tens_l350_350199


namespace sum_y_coordinates_l350_350082

theorem sum_y_coordinates {C : ℝ × ℝ} (hx : C = (-3, -4)) (r : ℝ) (hr : r = 7) :
  let points := [y | ∃ (y : ℝ), (0 + 3)^2 + (y + 4)^2 = r^2 ∧ y = y]
  (∃ y1 y2 : ℝ, y1 ∈ points ∧ y2 ∈ points ∧ y1 + y2 = -8) :=
by
  sorry

end sum_y_coordinates_l350_350082


namespace process_time_per_picture_l350_350055

theorem process_time_per_picture (pictures : ℕ) (total_hours : ℕ) (minutes_per_hour : ℕ) : 
  pictures = 960 ∧ total_hours = 32 ∧ minutes_per_hour = 60 → 
  (total_hours * minutes_per_hour) / pictures = 2 := 
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  simp
  sorry

end process_time_per_picture_l350_350055


namespace simplify_sqrt_450_l350_350649
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350649


namespace part1_eq_part2_if_empty_intersection_then_a_geq_3_l350_350935

open Set

variable {U : Type} {a : ℝ}

def universal_set : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B1 (a : ℝ) : Set ℝ := {x : ℝ | x > a}
def complement_B1 (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def intersection_with_complement (a : ℝ) : Set ℝ := A ∩ complement_B1 a

-- Statement for part (1)
theorem part1_eq {a : ℝ} (h : a = 2) : intersection_with_complement a = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by sorry

-- Statement for part (2)
theorem part2_if_empty_intersection_then_a_geq_3 
(h : A ∩ B1 a = ∅) : a ≥ 3 :=
by sorry

end part1_eq_part2_if_empty_intersection_then_a_geq_3_l350_350935


namespace simplify_sqrt_450_l350_350397

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350397


namespace chocolates_ratio_l350_350774

theorem chocolates_ratio (N W : ℕ) (h1 : N + W = 80) (h2 : 0.20 * N + 0.50 * W = 28) : N = W :=
by
  sorry

end chocolates_ratio_l350_350774


namespace expr1_eq_zero_expr2_eq_3sqrt2_l350_350834

-- Define the expression for the first problem
def expr1 : ℝ := real.sqrt ((-4)^2) + real.cbrt ((-4)^3) * (-1/2)^2 - real.cbrt (27)

-- Define the expression for the second problem
def expr2 : ℝ := real.sqrt 2 * (real.sqrt 2 + 2) - abs (real.sqrt 2 - 2)

-- Prove the first expression equals 0
theorem expr1_eq_zero : expr1 = 0 := 
by sorry

-- Prove the second expression equals 3 * sqrt 2
theorem expr2_eq_3sqrt2 : expr2 = 3 * real.sqrt 2 := 
by sorry

end expr1_eq_zero_expr2_eq_3sqrt2_l350_350834


namespace trapezoid_bases_length_l350_350058

def is_isosceles_trapezoid_circumscribed (diameter : ℝ) (leg_length : ℝ) 
  (base1 base2 : ℝ) : Prop :=
diameter = 15 ∧ leg_length = 17 ∧ 
(base1 = 25 ∧ base2 = 9 ∨ base1 = 9 ∧ base2 = 25)

theorem trapezoid_bases_length :
  ∃ (base1 base2 : ℝ), is_isosceles_trapezoid_circumscribed 15 17 base1 base2 :=
begin
  use [25, 9],
  unfold is_isosceles_trapezoid_circumscribed,
  sorry
end

end trapezoid_bases_length_l350_350058


namespace range_f_l350_350113

noncomputable def f (x : ℝ) : ℝ :=
  (arccot (x / 3))^2 - π * (arctan (x / 3)) + (arctan (x / 3))^2 + (π^2 / 18) * (x^2 - 3 * x + 9)

theorem range_f (x : ℝ) : 
  ∀ y, y = f x → y ∈ set.Ici 0 := 
sorry

end range_f_l350_350113


namespace max_element_ge_two_l350_350290

variable {n : ℕ}
variable {a : ℕ → ℝ}

theorem max_element_ge_two {n : ℕ} (hn : 3 < n) 
  (h1 : (∑ i in Finset.range n, a i) ≥ n) 
  (h2 : (∑ i in Finset.range n, (a i)^2) ≥ n^2) : 
  ∃ i, a i ≥ 2 := 
sorry

end max_element_ge_two_l350_350290


namespace simplify_sqrt_450_l350_350439

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350439


namespace simplify_sqrt_450_l350_350497

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350497


namespace sqrt_450_eq_15_sqrt_2_l350_350473

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350473


namespace simplify_sqrt_450_l350_350643
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350643


namespace simplify_sqrt_450_l350_350456

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350456


namespace find_length_of_side_c_l350_350259

theorem find_length_of_side_c (a b C : ℝ) (h_a : a = 3) (h_b : b = 5) (h_C : C = 120) : 
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos (C * Real.pi / 180)) 
  in c = 7 :=
by
  sorry

end find_length_of_side_c_l350_350259


namespace expression_equals_four_l350_350076

-- Definition of the problem
def expression : Real := Real.abs (-3) + Real.sqrt 3 * Real.sin (π/3) - (2⁻¹)

-- The statement of the problem
theorem expression_equals_four : expression = 4 := 
by
  sorry

end expression_equals_four_l350_350076


namespace fraction_of_august_tips_l350_350270

variable {A : ℝ} -- A denotes the average monthly tips for the other months.
variable {total_tips_6_months : ℝ} (h1 : total_tips_6_months = 6 * A)
variable {august_tips : ℝ} (h2 : august_tips = 6 * A)
variable {total_tips : ℝ} (h3 : total_tips = total_tips_6_months + august_tips)

theorem fraction_of_august_tips (h1 : total_tips_6_months = 6 * A)
                                (h2 : august_tips = 6 * A)
                                (h3 : total_tips = total_tips_6_months + august_tips) :
    (august_tips / total_tips) = 1 / 2 :=
by
    sorry

end fraction_of_august_tips_l350_350270


namespace sally_seashells_l350_350345

variable (M : ℝ)

theorem sally_seashells : 
  (1.20 * (M + M / 2) = 54) → M = 30 := 
by
  sorry

end sally_seashells_l350_350345


namespace simplify_sqrt_450_l350_350444

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350444


namespace sum_of_tens_l350_350226

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l350_350226


namespace number_of_divisors_l350_350835

theorem number_of_divisors (n : ℕ) (p : ℕ → Prop) (α : ℕ → ℕ) (k : ℕ) 
  (h_prime: ∀ i, i < k → p i → Prime i)
  (h_factorization: n = ∏ i in Finset.range k, (i ^ (α i))) :
  ∑ i in Finset.range k, (α i + 1) = (∏ i in Finset.range k, (α i + 1)) := 
by
  sorry

end number_of_divisors_l350_350835


namespace sqrt_450_eq_15_sqrt_2_l350_350697

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350697


namespace simplify_expression_l350_350705

theorem simplify_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 2) :
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 :=
by
  rw [h1, h2]
  sorry

end simplify_expression_l350_350705


namespace sqrt_450_eq_15_sqrt_2_l350_350580

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350580


namespace conjugate_of_z_l350_350950

open Complex

theorem conjugate_of_z (z : ℂ) (h : z * I = 1 + I) : conj z = 1 + I := 
by sorry

end conjugate_of_z_l350_350950


namespace g_evaluation_l350_350291

def g (x : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 5

theorem g_evaluation :
  g 11 + g (-11) = 2 :=
begin
  sorry
end

end g_evaluation_l350_350291


namespace find_complex_number_z_l350_350911

-- Given the complex number z and the equation \(\frac{z}{1+i} = i^{2015} + i^{2016}\)
-- prove that z = -2i
theorem find_complex_number_z (z : ℂ) (h : z / (1 + (1 : ℂ) * I) = I ^ 2015 + I ^ 2016) : z = -2 * I := 
by
  sorry

end find_complex_number_z_l350_350911


namespace simplify_sqrt_450_l350_350380

theorem simplify_sqrt_450 :
  let x : ℤ := 450
  let y : ℤ := 225
  let z : ℤ := 2
  let n : ℤ := 15
  (x = y * z) → (y = n^2) → (Real.sqrt x = n * Real.sqrt z) :=
by
  intros
  sorry

end simplify_sqrt_450_l350_350380


namespace scientific_notation_of_634000000_l350_350066

theorem scientific_notation_of_634000000 :
  634000000 = 6.34 * 10 ^ 8 := 
sorry

end scientific_notation_of_634000000_l350_350066


namespace simplify_sqrt_450_l350_350457

theorem simplify_sqrt_450 (h : 450 = 2 * 3^2 * 5^2) : Real.sqrt 450 = 15 * Real.sqrt 2 :=
  by
  sorry

end simplify_sqrt_450_l350_350457


namespace sqrt_450_simplified_l350_350633

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350633


namespace sqrt_450_equals_15_sqrt_2_l350_350596

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350596


namespace sqrt_450_eq_15_sqrt_2_l350_350574

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350574


namespace total_pears_sold_l350_350044

variable (morning afternoon total : ℕ)

-- Given conditions:
-- 1. The amount sold in the afternoon is twice the amount sold in the morning.
-- 2. The amount sold in the afternoon is 340 kg.
axiom afternoon_eq_twice_morning : afternoon = 2 * morning
axiom afternoon_eq_340 : afternoon = 340

-- Prove that the total amount of pears sold that day is 510 kg.
theorem total_pears_sold : total = morning + afternoon → total = 510 :=
by
  intro h
  rw [afternoon_eq_twice_morning, afternoon_eq_340] at h
  simp only [Nat.add_assoc, h]
  sorry

end total_pears_sold_l350_350044


namespace simplify_sqrt_450_l350_350413

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350413


namespace sqrt_450_equals_15_sqrt_2_l350_350599

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350599


namespace min_payment_max_payment_expected_payment_l350_350318

-- Given Prices
def item_prices : List ℕ := [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

-- Function to compute the actual paid amount given groups of three items
def paid_amount (groups : List (List ℕ)) : ℕ :=
  groups.foldr (λ group sum => sum + group.foldr (λ x s => s + x) 0 - group.minimum') 0

-- Optimal arrangement of items for minimal payment
def optimal_groups : List (List ℕ) :=
  [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]]

-- Suboptimal arrangement of items for maximal payment
def suboptimal_groups : List (List ℕ) :=
  [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]]

-- Expected value calculation's configuration
def num_items := 10
def num_groups := (num_items / 3).natCeil

noncomputable def expected_amount : ℕ :=
  let total_sum := item_prices.foldr (λ x s => s + x) 0
  let expected_savings := 100 * (660 / 72)
  total_sum - expected_savings

theorem min_payment : paid_amount optimal_groups = 4000 := by
  -- Proof steps and details here
  sorry

theorem max_payment : paid_amount suboptimal_groups = 4900 := by
  -- Proof steps and details here
  sorry

theorem expected_payment : expected_amount ≈ 4583 := by
  -- Proof steps and details here
  sorry

end min_payment_max_payment_expected_payment_l350_350318


namespace circle_equation_and_line_equation_l350_350908

theorem circle_equation_and_line_equation
  (A : ℝ × ℝ) (l1 : ℝ → ℝ → Prop) (B : ℝ × ℝ) (M N : ℝ × ℝ) :
  A = (-1, 2) →
  (l1 = λ x y, x + 2 * y + 7 = 0) →
  (∀ x y, l1 x y → ∃ r, (x + 1)^2 + (y - 2)^2 = r^2 ∧ r = 2 * Real.sqrt 5 ∧ (x + 1)^2 + (y - 2)^2 = 20) →
  B = (-4, 0) →
  |M - N| = 2 * Real.sqrt 11 →
  (∃ k : ℝ, (∀ x y, y = k * (x + 4) ∨ l = 5 * x + 12 * y + 20)) :=
by
  sorry

end circle_equation_and_line_equation_l350_350908


namespace h_2023_l350_350808

def h : ℕ → ℤ
| 1       := 2
| 2       := 3
| (n + 3) := h (n + 2) - h (n + 1) + 2 * (n + 3)

theorem h_2023 : h 2023 = 4051 :=
sorry

end h_2023_l350_350808


namespace interval_monotonic_decrease_cos_A_le_m_l350_350119

noncomputable def a (λ x : ℝ) : ℝ × ℝ := (2 * λ * sin x, sin x + cos x)
noncomputable def b (λ x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, λ * (sin x - cos x))
noncomputable def f (λ x : ℝ) : ℝ := (a λ x).fst * (b λ x).fst + (a λ x).snd * (b λ x).snd

theorem interval_monotonic_decrease (λ : ℝ) (hλ : λ = 1) (k : ℤ) :
    set.Icc (k * π + π / 3) (k * π + 5 * π / 6) ⊆ 
    { x : ℝ | decreasing_on (f λ) (set.Icc (k * π + π / 3) (k * π + 5 * π / 6)) } := sorry

theorem cos_A_le_m (A b a c m : ℝ) (h_C : cos A = (2 * b - a) / (2 * c))
  (h_fA : ∀ A, f 1 A - m > 0) :
  m ≤ -1 := sorry

end interval_monotonic_decrease_cos_A_le_m_l350_350119


namespace simplify_sqrt_450_l350_350673

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350673


namespace rhombus_inscribed_circle_radius_l350_350762

variable (d1 d2 : Real)
variable (r : Real)

def diagonals_inscribed_circle (d1 d2 : Real) : Real :=
  2 * sqrt (d1/2)^2 + (d2/2)^2

def area_rhombus (d1 d2 : Real) : Real :=
  (d1 * d2) / 2

theorem rhombus_inscribed_circle_radius (d1 d2 : Real) (h1 : d1 = 8) (h2 : d2 = 30) :
  let r := (60 / sqrt (241)) in
  area_rhombus d1 d2 = diagonals_inscribed_circle d1 d2 * r :=
by
  sorry

end rhombus_inscribed_circle_radius_l350_350762


namespace simplify_sqrt_450_l350_350412

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350412


namespace num_terms_100_pow_10_as_sum_of_tens_l350_350215

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l350_350215


namespace min_area_quadrilateral_l350_350980

section Quadrilateral

variables (ABCD : Type) (AC BD : ABCD → ABCD → Prop) (O : ABCD)
variables (area : ABCD → ℝ)
variables (A B C D : ABCD)
variables (intersects_at : (A B C D : ABCD) → Prop)
variables (triangle_area_DOC triangle_area_AOB : ℝ)
variables (area_tr_DOC : triangle_area_DOC = 4)
variables (area_tr_AOB : triangle_area_AOB = 36)

theorem min_area_quadrilateral :
  intersects_at A B C D →
  (area DOC = 4) ∧ (area AOB = 36) →
  ∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ (x + y = 40) →
  ∃ (area_ABCD : ℝ), area_ABCD = 80 :=
by 
  intros h_intersect h_areas h_x_y
  sorry

end Quadrilateral

end min_area_quadrilateral_l350_350980


namespace cyclic_ineq_l350_350874

-- Define the cyclic quadrilateral and associated lengths
variables {a b c d p q : ℝ}

-- Define the inequality to be proven
theorem cyclic_ineq (h : cyclic_quadrilateral ABCD a b c d p q) : 
  (a - c) + (b - d) ≥ 2 * |p - q| :=
sorry

end cyclic_ineq_l350_350874


namespace imaginary_unit_div_l350_350292

open Complex

theorem imaginary_unit_div (i : ℂ) (hi : i * i = -1) : (i / (1 + i) = (1 / 2) + (1 / 2) * i) :=
by
  sorry

end imaginary_unit_div_l350_350292


namespace triangle_lp_length_l350_350984

theorem triangle_lp_length
  (A B C K L P M : Point)
  (AC BC AK KC AM : ℝ)
  (h_AC : dist A C = 300)
  (h_BC : dist B C = 200)
  (h_midpoint_K : K = midpoint A C)
  (h_angle_bisector_L : angle_bisector C L A B)
  (h_intersection_P : P = line_intersection (line_through B K) (line_through C L))
  (h_AM_length : dist A M = 120)
  (h_midpoint_PM : K = midpoint P M) :
  dist L P = 480 / 7 :=
begin
  sorry
end

end triangle_lp_length_l350_350984


namespace number_of_solutions_l350_350896

noncomputable def generating_function_coefficient (p : List ℕ) (r : ℕ) : ℕ :=
    -- Function to extract the coefficient of t^r
    sorry

theorem number_of_solutions (p : List ℕ) (r : ℕ) (n : ℕ) 
    (h_p : ∀ i, i < n → 0 < p.get i) 
    (h_r : 0 < r):
    ∃ a_r : ℕ, a_r = generating_function_coefficient p r := sorry

end number_of_solutions_l350_350896


namespace sqrt_450_eq_15_sqrt_2_l350_350459

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350459


namespace height_of_parallelogram_l350_350005

theorem height_of_parallelogram (A B h : ℝ) (hA : A = 72) (hB : B = 12) (h_area : A = B * h) : h = 6 := by
  sorry

end height_of_parallelogram_l350_350005


namespace simplify_sqrt_450_l350_350502

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350502


namespace sqrt_450_eq_15_sqrt_2_l350_350689

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350689


namespace log_a2017_eq_half_l350_350255

variable {a : ℕ → ℝ}

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x^2 + 6 * x - 3

theorem log_a2017_eq_half
  (h1 : 0 < a 1)
  (h2 : 0 < a 4033)
  (h_extreme1 : f' (a 1) = 0)
  (h_extreme2 : f' (a 4033) = 0)
  (h_product : a 1 * a 4033 = 6) :
  Real.logb 6 (Real.sqrt (a 1 * a 4033)) = 1 / 2 :=
by
  sorry

end log_a2017_eq_half_l350_350255


namespace strictly_increasing_intervals_l350_350864

-- Define the function y = cos^2(x + π/2)
noncomputable def y (x : ℝ) : ℝ := (Real.cos (x + Real.pi / 2))^2

-- Define the assertion
theorem strictly_increasing_intervals (k : ℤ) : 
  StrictMonoOn y (Set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2)) :=
sorry

end strictly_increasing_intervals_l350_350864


namespace odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l350_350925

noncomputable def f (x : ℝ) (k : ℝ) := 2^x + k * 2^(-x)

-- Prove that if f(x) is an odd function, then k = -1.
theorem odd_function_k_eq_neg_one {k : ℝ} (h : ∀ x, f x k = -f (-x) k) : k = -1 :=
by sorry

-- Prove that if for all x in [0, +∞), f(x) > 2^(-x), then k > 0.
theorem f_x_greater_2_neg_x_k_gt_zero {k : ℝ} (h : ∀ x, 0 ≤ x → f x k > 2^(-x)) : k > 0 :=
by sorry

end odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l350_350925


namespace james_and_lisa_pizzas_l350_350264

theorem james_and_lisa_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) :
  slices_per_pizza = 6 →
  2 * total_slices = 3 * 8 →
  total_slices / slices_per_pizza = 2 :=
by
  intros h1 h2
  sorry

end james_and_lisa_pizzas_l350_350264


namespace incorrect_statement_is_B_l350_350771

-- Definitions based on conditions
def statement_A : Prop :=
  "The sampling survey method is suitable for surveying the interests and hobbies of 5000 students in a school."

def statement_C : Prop :=
  "China's population census uses a comprehensive survey method."

def statement_D : Prop :=
  "The sampling survey method should be used to investigate the water quality of Dongting Lake."

-- The statement that needs to be proved incorrect
def statement_B : Prop :=
  "The sampling survey method should be used to investigate the working conditions of the parents of classmates in this class."

-- Lean theorem statement that proves statement B is incorrect.
theorem incorrect_statement_is_B : ¬statement_B :=
by sorry

end incorrect_statement_is_B_l350_350771


namespace simplify_sqrt_450_l350_350674

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350674


namespace ram_journey_speed_l350_350339

theorem ram_journey_speed (v : ℝ) (t1 t2 d1 d2 total_time total_distance : ℝ) 
  (h1 : t1 = 3.2) 
  (h2 : d1 = 70 * t1) 
  (h3 : total_distance = 400) 
  (h4 : total_time = 8)
  (h5 : d2 = total_distance - d1)
  (h6 : t2 = total_time - t1)
  (h7 : v = d2 / t2) :
  v = 36.67 := 
by
  calc
    v = d2 / t2     : by rw [h7]
    ... = (total_distance - d1) / (total_time - t1) 
        : by rw [h5, h6]
    ... = (400 - 70 * 3.2) / (8 - 3.2)
        : by rw [h3, h1, h2, h4]
    ... = 36.67
        : by norm_num

end ram_journey_speed_l350_350339


namespace exists_n_balanced_subset_l350_350995

def is_n_balanced (n : ℕ) (M : Finset ℕ) : Prop :=
  let s_i := λ i, (M.powerset.filter (λ S, S.sum % n = i)).card
  ∀ i j, i ∈ Finset.range n → j ∈ Finset.range n → s_i i = s_i j

theorem exists_n_balanced_subset (n : ℕ) (hn : n % 2 = 1) : 
  ∃ M : Finset ℕ, M ≠ ∅ ∧ M ⊆ Finset.range (n + 1) ∧ is_n_balanced n M := 
sorry

end exists_n_balanced_subset_l350_350995


namespace simplify_sqrt_450_l350_350530

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350530


namespace sqrt_450_eq_15_sqrt_2_l350_350582

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350582


namespace translated_graph_pass_through_origin_l350_350246

theorem translated_graph_pass_through_origin 
    (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) 
    (passes_through_origin : 0 = Real.sin (-2 * φ + π / 3)) : 
    φ = π / 6 := 
sorry

end translated_graph_pass_through_origin_l350_350246


namespace player_beta_winning_strategy_l350_350749

-- Define the rules of the game and the winning conditions for player β
theorem player_beta_winning_strategy : 
  ∀ (sequence : list ℕ), 
  (∀ i, sequence.nth i < 10) ∧ 
  (∀ length, length > 0 → length ≤ 11 → (sequence.take length).sum % 11 ≠ 0) → 
  (∃ move, (sequence ++ [move]).sum % 11 = 0 → False) :=
sorry

end player_beta_winning_strategy_l350_350749


namespace tan_theta_half_l350_350881

open Real

theorem tan_theta_half (θ : ℝ) 
  (h0 : 0 < θ) 
  (h1 : θ < π / 2) 
  (h2 : ∃ k : ℝ, (sin (2 * θ), cos θ) = k • (cos θ, 1)) : 
  tan θ = 1 / 2 := by 
sorry

end tan_theta_half_l350_350881


namespace megan_broke_3_eggs_l350_350307

variables (total_eggs B C P : ℕ)

theorem megan_broke_3_eggs (h1 : total_eggs = 24) (h2 : C = 2 * B) (h3 : P = 24 - (B + C)) (h4 : P - C = 9) : B = 3 := by
  sorry

end megan_broke_3_eggs_l350_350307


namespace car_r_speed_l350_350006

theorem car_r_speed (v : ℝ) (h : 150 / v - 2 = 150 / (v + 10)) : v = 25 :=
sorry

end car_r_speed_l350_350006


namespace part1_part2_l350_350159

open Real

noncomputable def hyperbola (a b : ℝ) := 
  {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (B F O A P : ℝ × ℝ)
  (hB : B = (a, 0))
  (hF : F = (sqrt (a^2 + b^2), 0))
  (hO : O = (0, 0))
  (hGeomSeq : A = (a^2 / sqrt (a^2 + b^2), 0))
  (hP : ∃ (k : ℝ), P = (k, k * b / a))
  : (P - A) · O = (P - A) · F :=
sorry

theorem part2 : 
  let a := 1 
  let b := 2 
  let c := sqrt (a^2 + b^2) in 
  P = nal (D E F : ℝ × ℝ) : 
  (D E F) ∈ hyperbola a b 
  (hPerp : True) -- Placeholder for actual condition for P 
  : (| DF | / | DE |) = 3 / 2 :=
sorry

end part1_part2_l350_350159


namespace simplify_sqrt_450_l350_350646
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350646


namespace trig_identity_l350_350947

theorem trig_identity
  (α : ℝ)
  (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) :
  1 / (cos α ^ 2 + sin (2 * α)) = -2 := 
by sorry

end trig_identity_l350_350947


namespace measure_of_angle_C_length_of_side_AB_l350_350258

-- Given the conditions of triangle ABC
variables {A B C : ℝ} (a b c : ℝ)
variable (D : Point) -- Point D is the midpoint of BC
variables {AC AD AB: ℝ}
variables (cos_A cos_B cos_C : ℝ)

-- The conditions provided
axiom cos_conditions : AC = 2 ∧ AD = sqrt 7
axiom triangle_conditions : c / cos_C = (a + b) / (cos_A + cos_B)
axiom D_midpoint : D = midpoint B C
axiom angles_sum : A + B + C = π

-- Proof problem 1: Prove that angle C = π / 3
theorem measure_of_angle_C : C = π / 3 :=
by sorry

-- Proof problem 2: Prove that AB = 2 * sqrt 7
theorem length_of_side_AB (angle_C : C = π / 3) : AB = 2 * sqrt 7 :=
by sorry

end measure_of_angle_C_length_of_side_AB_l350_350258


namespace sqrt_450_eq_15_sqrt_2_l350_350471

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350471


namespace parallel_vectors_cosine_of_angle_l350_350938

section VectorProofs

variable (t : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (t, -1)
def c : ℝ × ℝ := (-3, -1)

theorem parallel_vectors (t : ℝ) : 
  (a.1 + b.1, a.2 + b.2) ∥ (2 * a.1 - c.1, 2 * a.2 - c.2) → t = 0 := by
  -- Proof goes here
  sorry

theorem cosine_of_angle (t : ℝ) : 
  (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 0) → 
  ∀ θ, cos θ = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  t = 7 → θ = Real.acos (Real.sqrt (10) / 10 ) := by
  -- Proof goes here
  sorry

end VectorProofs

end parallel_vectors_cosine_of_angle_l350_350938


namespace sqrt_450_equals_15_sqrt_2_l350_350611

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350611


namespace hexagon_area_correct_m_plus_n_l350_350285

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (b, 3)
  let F := (-3 * (3 + b) / 2, 9)  -- derived from complex numbers and angle conversion
  let hexagon_height := 12  -- height difference between the y-coordinates
  let hexagon_base := 3 * (b + 3) / 2  -- distance between parallel lines AB and DE
  36 / 2 * (b + 3) + 6 * (6 + b * Real.sqrt 3)

theorem hexagon_area_correct (b : ℝ) :
  hexagon_area b = 72 * Real.sqrt 3 :=
sorry

theorem m_plus_n : 72 + 3 = 75 := rfl

end hexagon_area_correct_m_plus_n_l350_350285


namespace sqrt_450_eq_15_sqrt_2_l350_350579

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350579


namespace yellow_ball_impossible_at_least_one_red_ball_probability_of_red_ball_find_x_l350_350975

-- Given conditions
def initial_white_balls := 8
def initial_red_balls := 12
def total_balls := initial_white_balls + initial_red_balls

-- Question 1(a): Drawing a yellow ball is impossible
theorem yellow_ball_impossible (total_balls : ℕ) : false :=
by 
  sorry -- The proof would go here

-- Question 1(b): Probability of drawing at least one red ball
theorem at_least_one_red_ball (total_balls : ℕ) (drawn_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) (total_balls = white_balls + red_balls) : ℝ :=
by
  have h : white_balls < drawn_balls → 1 = 1 :=
  by
    sorry -- The proof would go here
  h

-- Question 2: Probability of drawing a red ball at random
theorem probability_of_red_ball (red_balls white_balls : ℕ) : ℝ :=
  red_balls / (red_balls + white_balls)

-- Question 3: Finding x given the probability of drawing a white ball is 4/5
theorem find_x (initial_white_balls initial_red_balls : ℕ) (draw_white_prob : ℝ) : ℕ :=
by
  let x := initial_white_balls + 8 -- filter x from the probability 4/5 assumption
  sorry -- The proof would go here

end yellow_ball_impossible_at_least_one_red_ball_probability_of_red_ball_find_x_l350_350975


namespace magnetic_field_at_center_of_rotating_ring_l350_350312

noncomputable def magnetic_field_center (R Q ω : ℝ) : ℝ :=
  ω * Q / (8 * π * R)

theorem magnetic_field_at_center_of_rotating_ring
  (R Q ω : ℝ) : magnetic_field_center R Q ω = ω * Q / (8 * π * R) :=
by
  sorry

end magnetic_field_at_center_of_rotating_ring_l350_350312


namespace probability_solution_l350_350038

-- Define the interval and the condition for k
def interval (a b : ℝ) (k : ℝ) : Prop := a ≤ k ∧ k ≤ b

-- Define the quadratic equation
def quad_eq (k : ℝ) (x : ℝ) := (k^2 + k - 90) * x^2 + (3 * k - 8) * x + 2 = 0

-- Define the condition on the roots of the quadratic equation
def root_condition (k x1 x2 : ℝ) : Prop := x1 + x2 = (8 - 3 * k) / (k^2 + k - 90) ∧ x1 * x2 = 2 / (k^2 + k - 90) ∧ x1 ≤ 2 * x2

-- Main theorem to prove the probability
theorem probability_solution : 
  (∃ (k : ℝ), interval 12 17 k ∧ ∀ x1 x2 : ℝ, root_condition k x1 x2 → x1 ≤ 2 * x2)  ↔ 
  (2 / 3 : ℝ) :=
sorry

end probability_solution_l350_350038


namespace sqrt_450_eq_15_sqrt_2_l350_350678

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350678


namespace unique_solution_m_l350_350101

theorem unique_solution_m (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ (∀ y₁ y₂ : ℝ, 3 * y₁^2 - 6 * y₂ + m = 0 → y₁ = y₂)) → m = 3 :=
by
  sorry

end unique_solution_m_l350_350101


namespace mean_score_is_76_l350_350779

noncomputable def mean_stddev_problem := 
  ∃ (M SD : ℝ), (M - 2 * SD = 60) ∧ (M + 3 * SD = 100) ∧ (M = 76)

theorem mean_score_is_76 : mean_stddev_problem :=
sorry

end mean_score_is_76_l350_350779


namespace sqrt_450_simplified_l350_350424

theorem sqrt_450_simplified :
  ∃ (x : ℝ), (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2 → sqrt 450 = 15 * sqrt 2 :=
begin
  assume h : (450 : ℝ) = (2 : ℝ) * (3 : ℝ)^2 * (5 : ℝ)^2,
  apply exists.intro (15 * sqrt 2),
  sorry
end

end sqrt_450_simplified_l350_350424


namespace sum_of_tens_l350_350225

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l350_350225


namespace sqrt_450_eq_15_sqrt_2_l350_350680

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350680


namespace perpendicular_line_x_intercept_l350_350754

-- Definition of the given line and its properties
def line1 : ℝ → ℝ := λ x, (-4 / 5) * x + 4

-- Definition of the line perpendicular to the given line with y-intercept -3
def line2 : ℝ → ℝ := λ x, (5 / 4) * x - 3

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, line2 x = 0) ∧ (line2 0 = -3) →
  (∃ x : ℝ, x = 12 / 5) :=
by
  sorry

end perpendicular_line_x_intercept_l350_350754


namespace no_such_polynomial_exists_l350_350986

open Polynomial

noncomputable def example_polynomial : Type := {P : Polynomial ℤ // P.eval 2 = 4 ∧ P.eval (P.eval 2) = 7}

theorem no_such_polynomial_exists : ¬ ∃ P : example_polynomial, true :=
by
  sorry

end no_such_polynomial_exists_l350_350986


namespace min_payment_max_payment_expected_value_payment_l350_350332

-- Proof Problem 1
theorem min_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let optimized_groups := [[1000, 900, 800], [700, 600, 500], [400, 300, 200], [100]] in
  (∑ g in optimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4000 := by
  sorry

-- Proof Problem 2
theorem max_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let suboptimized_groups := [[1000, 900, 100], [800, 700, 200], [600, 500, 300], [400]] in
  (∑ g in suboptimized_groups, ∑ l in (g.erase (g.min' (λ x y => x ≤ y))), l) = 4900 := by
  sorry

-- Proof Problem 3
theorem expected_value_payment (prices : List ℕ) (H1 : prices = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]) :
  let expected_savings := 100 * (∑ k in List.range 9, k * ((10 - k) * (9 - k)) / 72) in
  (5500 - expected_savings) = 4583.33 := by 
  sorry

end min_payment_max_payment_expected_value_payment_l350_350332


namespace sqrt_450_eq_15_sqrt_2_l350_350570

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350570


namespace expression_positive_intervals_l350_350851

def f (x : ℝ) : ℝ := (x + 1) * (x + 3) * (x - 2)

theorem expression_positive_intervals :
  ∀ x : ℝ, 0 < f x ↔ (x ∈ set.Ioo (-3 : ℝ) (-1) ∪ set.Ioi (2 : ℝ)) :=
by
  intro x
  sorry

end expression_positive_intervals_l350_350851


namespace sqrt_simplify_l350_350358

theorem sqrt_simplify :
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  sorry

end sqrt_simplify_l350_350358


namespace simplify_sqrt_450_l350_350669

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350669


namespace simplify_sqrt_450_l350_350521

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350521


namespace simplify_sqrt_450_l350_350650
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350650


namespace amplitude_condition_l350_350092

noncomputable def func (x α β : ℝ) : ℝ := sin (x + α) + sin (x + β)

theorem amplitude_condition (α β : ℝ) (h₁ : ∀ x : ℝ, abs (func x α β) ≤ 1) :
  ∃ k : ℤ, α - β = 2 * k * π + 2 * π / 3 ∨ α - β = 2 * k * π - 2 * π / 3 :=
by
  sorry

end amplitude_condition_l350_350092


namespace sqrt_450_equals_15_sqrt_2_l350_350606

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350606


namespace remainder_of_789987_div_8_l350_350763

theorem remainder_of_789987_div_8 : (789987 % 8) = 3 := by
  sorry

end remainder_of_789987_div_8_l350_350763


namespace find_initial_investment_l350_350308

open Real

noncomputable def initial_investment (x : ℝ) (years : ℕ) (final_value : ℝ) : ℝ := 
  final_value / (3 ^ (years / (112 / x)))

theorem find_initial_investment :
  let x := 8
  let years := 28
  let final_value := 31500
  initial_investment x years final_value = 3500 := 
by 
  sorry

end find_initial_investment_l350_350308


namespace remaining_fish_l350_350189

theorem remaining_fish (initial_fish : ℝ) (moved_fish : ℝ) (remaining_fish : ℝ) : initial_fish = 212.0 → moved_fish = 68.0 → remaining_fish = 144.0 → initial_fish - moved_fish = remaining_fish := by sorry

end remaining_fish_l350_350189


namespace classroom_books_l350_350964

theorem classroom_books (students_group1 students_group2 books_per_student_group1 books_per_student_group2 books_brought books_lost : ℕ)
  (h1 : students_group1 = 20)
  (h2 : books_per_student_group1 = 15)
  (h3 : students_group2 = 25)
  (h4 : books_per_student_group2 = 10)
  (h5 : books_brought = 30)
  (h6 : books_lost = 7) :
  (students_group1 * books_per_student_group1 + students_group2 * books_per_student_group2 + books_brought - books_lost) = 573 := by
  sorry

end classroom_books_l350_350964


namespace radius_of_sphere_centered_at_O_l350_350287

theorem radius_of_sphere_centered_at_O (S A B C O N M P : Type) 
    (SA SB SC AB BC AC : ℝ) 
    (hMidpointN : N = midpoint SB)
    (hMidpointM : M = midpoint AC)
    (hMidpointP : P = midpoint MN)
    (hCondition : SA ^ 2 + SB ^ 2 + SC ^ 2 = AB ^ 2 + BC ^ 2 + AC ^ 2)
    (hSP : SP = 3 * sqrt(7))
    (hOP : OP = sqrt(21)) : 
    radius O = 2 * sqrt(21) := 
  sorry

end radius_of_sphere_centered_at_O_l350_350287


namespace jessica_current_age_l350_350268

theorem jessica_current_age : 
  ∃ J M_d M_c : ℕ, 
    J = (M_d / 2) ∧ 
    M_d = M_c - 10 ∧ 
    M_c = 70 ∧ 
    J + 10 = 40 := 
sorry

end jessica_current_age_l350_350268


namespace simplify_sqrt_450_l350_350661

theorem simplify_sqrt_450 : 
    sqrt 450 = 15 * sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350661


namespace simplify_sqrt_450_l350_350517

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350517


namespace distance_between_planes_l350_350097

theorem distance_between_planes (A B C D₁ D₂ : ℝ) (h₁ : A = 1) (h₂ : B = -4) (h₃ : C = 4) (h₄ : D₁ = 10) (h₅ : D₂ = 7) :
  let distance := (|D₂ - D₁| / Real.sqrt (A^2 + B^2 + C^2)) in
  distance = 3 / Real.sqrt 33 :=
by
  sorry

end distance_between_planes_l350_350097


namespace factorize_expression_l350_350860

noncomputable def E (a b c : ℚ) : ℚ :=
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4

noncomputable def q (a b c : ℚ) : ℚ :=
  a^2 + b^2 + c^2 + ab + bc + ca

theorem factorize_expression (a b c : ℚ) :
  E(a, b, c) = (a - b) * (b - c) * (c - a) * q(a, b, c) :=
by
  sorry

end factorize_expression_l350_350860


namespace SetD_forms_triangle_l350_350770

theorem SetD_forms_triangle :
  ∀ (a b c : ℝ), 
    (a = 1 ∧ b = 2 ∧ c = 3 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
    (a = 1 ∧ b = 1.5 ∧ c = 3 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
    (a = 3 ∧ b = 4 ∧ c = 8 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
    (a = 4 ∧ b = 5 ∧ c = 6 → (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  intro a b c
  constructor
  { rintro ⟨h₁, h₂, h₃⟩
    rw [h₁, h₂, h₃]
    simp [not_lt, le_refl] }
  constructor
  { rintro ⟨h₁, h₂, h₃⟩
    rw [h₁, h₂, h₃]
    simp [add_assoc] }
  constructor
  { rintro ⟨h₁, h₂, h₃⟩
    rw [h₁, h₂, h₃]
    simp [not_lt, add_comm] }
  { rintro ⟨h₁, h₂, h₃⟩
    rw [h₁, h₂, h₃]
    simp }

end SetD_forms_triangle_l350_350770


namespace equal_chords_divide_equally_l350_350237

theorem equal_chords_divide_equally 
  {A B C D M : ℝ} 
  (in_circle : ∃ (O : ℝ), (dist O A = dist O B) ∧ (dist O C = dist O D) ∧ (dist O M < dist O A))
  (chords_equal : dist A B = dist C D)
  (intersection_M : dist A M + dist M B = dist C M + dist M D ∧ dist A M = dist C M ∧ dist B M = dist D M) :
  dist A M = dist M B ∧ dist C M = dist M D := 
sorry

end equal_chords_divide_equally_l350_350237


namespace simplify_sqrt_450_l350_350507

theorem simplify_sqrt_450 : sqrt (450) = 15 * sqrt (2) :=
sorry

end simplify_sqrt_450_l350_350507


namespace andy_loss_more_likely_l350_350059

def probability_of_winning_all (p_win1 p_win2 p_win3 : ℝ) : ℝ :=
  p_win1 * p_win2 * p_win3

def probability_of_losing_all (p_lose1 p_lose2 p_lose3 : ℝ) : ℝ :=
  p_lose1 * p_lose2 * p_lose3

def percentage_difference (p_win1 p_win2 p_win3 : ℝ) : ℝ :=
  let p_lose1 := 1 - p_win1
  let p_lose2 := 1 - p_win2
  let p_lose3 := 1 - p_win3
  let p_win_all := probability_of_winning_all p_win1 p_win2 p_win3
  let p_lose_all := probability_of_losing_all p_lose1 p_lose2 p_lose3
  (p_lose_all - p_win_all) / p_win_all * 100

theorem andy_loss_more_likely {p_win1 p_win2 p_win3 : ℝ} (h₁ : p_win1 = 0.30) (h₂ : p_win2 = 0.50) (h₃ : p_win3 = 0.40) :
  percentage_difference p_win1 p_win2 p_win3 = 250 :=
by
  rw [h₁, h₂, h₃]
  admit

end andy_loss_more_likely_l350_350059


namespace vectors_not_coplanar_l350_350826

def vector_a : Vector ℝ 3 := ![4, 3, 1]
def vector_b : Vector ℝ 3 := ![1, -2, 1]
def vector_c : Vector ℝ 3 := ![2, 2, 2]

theorem vectors_not_coplanar : Matrix.det (λ i, fin.cases (vector_a i) (λ _, fin.cases (vector_b i) (λ _, vector_c i))) ≠ 0 :=
by
  sorry

end vectors_not_coplanar_l350_350826


namespace simplify_sqrt_450_l350_350404

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350404


namespace find_fourth_number_l350_350970

-- Define the initial conditions
variables {A B C D : ℕ}
variables (r : Ratio ℕ) (s : ℕ) -- r is for the ratio, s for the sum

-- Conditions
def ratio_condition (r : Ratio ℕ) (A B C : ℕ) : Prop :=
  r = Ratio.mk 5 3 4 ∧ r.first = A ∧ r.second = B ∧ r.third = C

def sum_condition (s : ℕ) (A B C : ℕ) : Prop :=
  A + B + C = s

def arithmetic_progression_condition (A B C D : ℕ) : Prop :=
  2 * B = A + C ∧ 2 * C = B + D

-- The Lean 4 statement
theorem find_fourth_number 
  (r : Ratio ℕ) 
  (s : ℕ)
  (A B C D : ℕ)
  (h_ratio : ratio_condition r A B C)
  (h_sum : sum_condition s A B C)
  (h_arithmetic : arithmetic_progression_condition A B C D) :
  D = 45 :=
sorry

end find_fourth_number_l350_350970


namespace simplify_sqrt_450_l350_350543

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l350_350543


namespace combined_total_difference_l350_350078

theorem combined_total_difference :
  let Chris_cards := 18
  let Charlie_cards := 32
  let Diana_cards := 25
  let Ethan_cards := 40
  (Charlie_cards - Chris_cards) + (Diana_cards - Chris_cards) + (Ethan_cards - Chris_cards) = 43 :=
by
  let Chris_cards := 18
  let Charlie_cards := 32
  let Diana_cards := 25
  let Ethan_cards := 40
  have h1 : Charlie_cards - Chris_cards = 14 := by sorry
  have h2 : Diana_cards - Chris_cards = 7 := by sorry
  have h3 : Ethan_cards - Chris_cards = 22 := by sorry
  show (Charlie_cards - Chris_cards) + (Diana_cards - Chris_cards) + (Ethan_cards - Chris_cards) = 43 from
    by rw [h1, h2, h3]; exact (14 + 7 + 22).symm

end combined_total_difference_l350_350078


namespace concurrence_of_lines_l350_350997

noncomputable def triangle_circumcircle (A B C : Point) : Circle := sorry
noncomputable def circle_internally_tangent (Γ : Circle) (P : Point) (AB AC : Line) : Circle := sorry

theorem concurrence_of_lines
  (A B C : Point)
  (Γ : Circle := triangle_circumcircle A B C)
  (A' B' C' : Point)
  (ΓA : Circle := circle_internally_tangent Γ A' (line_through A B) (line_through A C))
  (ΓB : Circle := circle_internally_tangent Γ B' (line_through B C) (line_through B A))
  (ΓC : Circle := circle_internally_tangent Γ C' (line_through C A) (line_through C B))
  : concurrent (line_through A A') (line_through B B') (line_through C C') :=
sorry

end concurrence_of_lines_l350_350997


namespace problem_solution_l350_350905

theorem problem_solution
  (h1 : 7125 / 1.25 = 5700)
  (x : ℕ)
  (a : ℕ)
  (hx : x = 3) :
  (712.5 / 12.5) ^ x = a → a = 185193 :=
by sorry

end problem_solution_l350_350905


namespace pq_over_pc_l350_350827

-- an acute triangle ABC inscribed in a circle with center O
variables {A B C O P Q : Type*}
-- acute ∆ABC is inscribed in circle △O and point P lies on extension of BC,
-- point Q is on segment BC
variables [AcuteTriangleABC] [InscribedIn A B C O] 
-- additional conditions
variables (AB_gt_AC : A B > A C)
variables (PA_tangent_to_circle_O : Tangent P A O)
variables (angle_POQ_plus_BAC_eq_90 : angle PO Q + angle BA C = 90)
variables (PA_by_PO_eq_t : ∀ t, PA / PO = t)

-- Prove PQ / PC = 1 / t^2
theorem pq_over_pc {t : ℝ} (h : PA / PO = t) : PQ / PC = 1 / t^2 :=
sorry

end pq_over_pc_l350_350827


namespace sqrt_450_eq_15_sqrt_2_l350_350589

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350589


namespace sqrt_450_eq_15_sqrt_2_l350_350688

theorem sqrt_450_eq_15_sqrt_2
  (h1 : 450 = 225 * 2)
  (h2 : real.sqrt 225 = 15) :
  real.sqrt 450 = 15 * real.sqrt 2 :=
sorry

end sqrt_450_eq_15_sqrt_2_l350_350688


namespace max_value_of_function_f_l350_350761

noncomputable def f (t : ℝ) : ℝ := (4^t - 2 * t) * t / 16^t

theorem max_value_of_function_f : ∃ t : ℝ, ∀ x : ℝ, f x ≤ f t ∧ f t = 1 / 8 := sorry

end max_value_of_function_f_l350_350761


namespace sqrt_450_eq_15_sqrt_2_l350_350576

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350576


namespace find_five_pi_over_three_l350_350094

noncomputable def f : ℝ → ℝ := sorry

theorem find_five_pi_over_three
  (f_even : ∀ x : ℝ, f x = f (-x))
  (f_periodic : ∀ x : ℝ, f x = f (x + π))
  (f_sin : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x = sin x) :
  f (5 * π / 3) = sqrt 3 / 2 :=
by sorry

end find_five_pi_over_three_l350_350094


namespace simplify_sqrt_450_l350_350647
open Real

theorem simplify_sqrt_450 :
  sqrt 450 = 15 * sqrt 2 :=
by
  have h_factored : 450 = 225 * 2 := by norm_num
  have h_sqrt_225 : sqrt 225 = 15 := by norm_num
  rw [h_factored, sqrt_mul (show 0 ≤ 225 from by norm_num) (show 0 ≤ 2 from by norm_num), h_sqrt_225]
  norm_num

end simplify_sqrt_450_l350_350647


namespace sqrt_450_equals_15_sqrt_2_l350_350602

theorem sqrt_450_equals_15_sqrt_2 :
  ∀ (n : ℕ), (n = 450) → 
  (∃ (a : ℕ), a = 225) → 
  (∃ (b : ℕ), b = 2) → 
  (∃ (c : ℕ), c = 15) → 
  (sqrt (225 * 2) = sqrt 225 * sqrt 2) → 
  (sqrt n = 15 * sqrt 2) :=
begin
  intros n hn h225 h2 h15 hsqrt,
  rw hn,
  cases h225 with a ha,
  cases h2 with b hb,
  cases h15 with c hc,
  rw [ha, hb] at hsqrt,
  rw [ha, hb, hc],
  exact hsqrt
end

end sqrt_450_equals_15_sqrt_2_l350_350602


namespace sqrt_450_simplified_l350_350623

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350623


namespace math_course_schedule_l350_350177
open Nat

theorem math_course_schedule {periods : Finset ℕ} (h : periods = {2, 3, 4, 5, 6}) :
  (∃ s : Finset ℕ, s.card = 3 ∧ (∀ (x y : ℕ), x ∈ s → y ∈ s → x ≠ y → abs (x - y) ≥ 2)) →
  6 = 6 :=
by
  intros h1
  have : periods.card = 5 := by sorry
  have valid_period : ∃! (s : Finset ℕ), s.card = 3 ∧ (∀ (x y : ℕ), x ∈ s → y ∈ s → x ≠ y → abs (x - y) ≥ 2) := 
    by sorry
  sorry

end math_course_schedule_l350_350177


namespace intersection_points_l350_350849

noncomputable def even_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem intersection_points (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono_inc : monotonically_increasing f)
  (h_sign_change : f 1 * f 2 < 0) :
  ∃! x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
sorry

end intersection_points_l350_350849


namespace bicycle_distance_l350_350172

theorem bicycle_distance (motorcycle_speed : ℕ) (half : ℝ → ℝ) (convert_minutes_to_hours : ℝ → ℝ) 
  (bicycle_speed : ℝ) (distance : ℝ) : 
  motorcycle_speed = 40 → 
  half = λ x, x / 2 → 
  convert_minutes_to_hours = λ minutes, minutes / 60 → 
  bicycle_speed = half motorcycle_speed →
  distance = bicycle_speed * convert_minutes_to_hours 30 →
  distance = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end bicycle_distance_l350_350172


namespace sqrt_450_simplified_l350_350616

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350616


namespace quadratic_root_range_l350_350228

theorem quadratic_root_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, (x₁ > 0) ∧ (x₂ < 0) ∧ (x₁^2 + 2 * (a - 1) * x₁ + 2 * a + 6 = 0) ∧ (x₂^2 + 2 * (a - 1) * x₂ + 2 * a + 6 = 0)) → a < -3 :=
by
  sorry

end quadratic_root_range_l350_350228


namespace sqrt_450_simplified_l350_350620

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350620


namespace fraction_equality_l350_350121

theorem fraction_equality (a b : ℝ) (h : (1 / a) - (1 / b) = 4) :
  (a - 2 * a * b - b) / (2 * a - 2 * b + 7 * a * b) = 6 :=
by
  sorry

end fraction_equality_l350_350121


namespace sqrt_450_simplified_l350_350621

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l350_350621


namespace find_angle_EHC_l350_350257

variable {A B C E H : Type}
variable {triangle_ABC : Prop}
variable {is_acute : ∀ (A B C), triangle_ABC → (∃ (angle_A : ℝ), angle_A < 90)}
variable {angle_bisector_AE : ∀ (A B C E), triangle_ABC → Prop}
variable {altitude_BH : ∀ (B H C), triangle_ABC → Prop}
variable {angle_AEB : ∀ (A E B), Prop}

theorem find_angle_EHC (h1 : triangle_ABC A B C)
                       (h2 : is_acute A B C h1)
                       (h3 : angle_bisector_AE A B C E h1)
                       (h4 : altitude_BH B H C h1)
                       (h5 : angle_AEB A E B ∧ angle_AEB A E B = 45) : 
                       ∃ (angle_EHC : ℝ), angle_EHC = 45 :=
sorry

end find_angle_EHC_l350_350257


namespace largest_x_satisfying_equation_l350_350853

theorem largest_x_satisfying_equation :
  (∃ x : ℝ, (| x^2 - 11 * x + 24 | + | 2 * x^2 + 6 * x - 56 | = | x^2 + 17 * x - 80 |) ∧ ∀ y : ℝ, 
    (| y^2 - 11 * y + 24 | + | 2 * y^2 + 6 * y - 56 | = | y^2 + 17 * y - 80 |) → y ≤ x) →
  x = 8 := 
sorry

end largest_x_satisfying_equation_l350_350853


namespace sqrt_450_eq_15_sqrt_2_l350_350585

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350585


namespace find_a_in_range_l350_350919

-- Definitions for the conditions
def z1 : ℂ := (1 + I)⁻¹ * (-1 + 5 * I)
def z2 (a : ℝ) : ℂ := (a - 2) - 1 * I
def z2_conjugate (a : ℝ) : ℂ := (a - 2) + 1 * I

-- The required proof problem statement
theorem find_a_in_range (a : ℝ) (h_condition : |z1 - z2_conjugate a| < |z1|) : 1 < a ∧ a < 7 :=
sorry

end find_a_in_range_l350_350919


namespace ellipse_simplification_l350_350713

theorem ellipse_simplification (x y : ℝ) :
  sqrt ((x - 2) ^ 2 + y ^ 2) + sqrt ((x + 2) ^ 2 + y ^ 2) = 10 ↔ 
  (x^2 / 25) + (y^2 / 21) = 1 := 
sorry

end ellipse_simplification_l350_350713


namespace sqrt_450_eq_15_sqrt_2_l350_350469

theorem sqrt_450_eq_15_sqrt_2 (h1 : 450 = 225 * 2) (h2 : 225 = 15 ^ 2) : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by 
  sorry

end sqrt_450_eq_15_sqrt_2_l350_350469


namespace simplify_sqrt_450_l350_350492

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350492


namespace math_proof_problem_l350_350820

theorem math_proof_problem :
  ¬ (∀ x : ℝ, 2^x > 0) ↔ ∃ x : ℝ, 2^x ≤ 0 :=
by
  sorry

end math_proof_problem_l350_350820


namespace probability_complement_B_probability_union_A_B_l350_350183

variable (Ω : Type) [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variable (A B : Set Ω)

theorem probability_complement_B
  (hB : P B = 1 / 3) : P Bᶜ = 2 / 3 :=
by
  sorry

theorem probability_union_A_B
  (hA : P A = 1 / 2) (hB : P B = 1 / 3) : P (A ∪ B) ≤ 5 / 6 :=
by
  sorry

end probability_complement_B_probability_union_A_B_l350_350183


namespace num_terms_100_pow_10_as_sum_of_tens_l350_350212

example : (10^2)^10 = 100^10 :=
by rw [pow_mul, mul_comm 2 10, pow_mul]

theorem num_terms_100_pow_10_as_sum_of_tens : (100^10) / 10 = 10^19 :=
by {
  norm_num,
  rw [←pow_add],
  exact nat.div_eq_of_lt_pow_succ ((nat.lt_base_pow_succ 10 1).left)
}

end num_terms_100_pow_10_as_sum_of_tens_l350_350212


namespace simplify_sqrt_450_l350_350483

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  have h : 450 = 2 * 3^2 * 5^2 := by norm_num
  rw [h, Real.sqrt_mul]
  rw [Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul, Real.sqrt_mul]
  rw [Real.sqrt_eq_rpow]
  repeat { rw [Real.sqrt_mul] }
  rw [Real.sqrt_rpow, Real.rpow_mul, Real.rpow_nat_cast, Real.sqrt_eq_rpow]
  rw [Real.sqrt_rpow, Real.sqrt_rpow, Real.rpow_mul, Real.sqrt_rpow, Real.sqrt_nat_cast]
  exact sorry

end simplify_sqrt_450_l350_350483


namespace valve_XY_time_correct_l350_350858

-- Given conditions
def valve_rates (x y z : ℝ) := (x + y + z = 1/2 ∧ x + z = 1/4 ∧ y + z = 1/3)
def total_fill_time (t : ℝ) (x y : ℝ) := t = 1 / (x + y)

-- The proof problem
theorem valve_XY_time_correct (x y z : ℝ) (t : ℝ) 
  (h : valve_rates x y z) : total_fill_time t x y → t = 2.4 :=
by
  -- Assume h defines the rates
  have h1 : x + y + z = 1/2 := h.1
  have h2 : x + z = 1/4 := h.2.1
  have h3 : y + z = 1/3 := h.2.2
  
  sorry

end valve_XY_time_correct_l350_350858


namespace profit_function_and_max_profit_l350_350801

noncomputable def W : ℝ → ℝ :=
λ x, if (0 < x ∧ x < 40) then -10 * x^2 + 600 * x - 1250
     else if (x ≥ 40) then -x - (10000 / x) + 8200
     else 0

theorem profit_function_and_max_profit :
    (∀ x : ℝ, (0 < x ∧ x < 40) → W x = -10 * x^2 + 600 * x - 1250) ∧
    (∀ x : ℝ, (x ≥ 40) → W x = -x - (10000 / x) + 8200) ∧
    (∀ x : ℝ, x = 100 → W x = 8000) :=
by
  sorry

end profit_function_and_max_profit_l350_350801


namespace starting_lineup_count_l350_350084

theorem starting_lineup_count 
  (total_players : ℕ)
  (star_players : ℕ)
  (required_additional_players : ℕ)
  (remaining_players : ℕ)
  (comb_value : ℕ) 
  (h1 : total_players = 15)
  (h2 : star_players = 3)
  (h3 : required_additional_players = 3)
  (h4 : remaining_players = total_players - star_players)
  (h5 : comb_value = Nat.choose remaining_players required_additional_players) :
  comb_value = 220 :=
by
  rw [total_players, star_players, required_additional_players, remaining_players] at *
  rw [h4, Nat.choose_eq_factorial_div_factorial (remaining_players - required_additional_players)]
  sorry

end starting_lineup_count_l350_350084


namespace simplify_sqrt_450_l350_350394

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350394


namespace trisector_length_is_correct_l350_350142

open Real

theorem trisector_length_is_correct {DE EF : ℝ} (hDE : DE = 5) (hEF : EF = 12) :
  let DF := sqrt (DE^2 + EF^2)
  DF = 13 ∧ (∃ FP : ℝ, FP = 10 * sqrt 3 / 3) :=
by
  have hHypotenuse : DF = sqrt (DE^2 + EF^2) := rfl
  
  /-
  The remaining proof would consist of validating that DF = 13 and 
  solving using geometric relationships similar to the given solution.
  -/
  have hDF : DF = sqrt (5^2 + 12^2) := by rw [hDE, hEF]; exact rfl

  have h13 : DF = 13 := by norm_num at hDF; exact hDF

  let FP := 10 * sqrt 3 / 3

  use FP
  exact ⟨h13, rfl⟩

end trisector_length_is_correct_l350_350142


namespace simplify_sqrt_450_l350_350396

noncomputable def sqrt_450 : ℝ := Real.sqrt 450
noncomputable def expected_value : ℝ := 15 * Real.sqrt 2

theorem simplify_sqrt_450 : sqrt_450 = expected_value := 
by sorry

end simplify_sqrt_450_l350_350396
