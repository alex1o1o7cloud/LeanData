import Mathlib

namespace find_a_b_find_k_range_l357_357797

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) : ℝ := a * x^2 + 1
def g (x : ℝ) (b : ℝ) : ℝ := x^3 + b * x
def tangent_condition (a b : ℝ) : Prop :=
  f 1 a = g 1 b ∧ (deriv (λ x, f x a) 1 = deriv (λ x, g x b) 1)
def max_value_condition (a b h : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ (x : ℝ), k ≤ x ∧ x ≤ 2 → h x ≤ 28

-- Statement for part 1
theorem find_a_b (a b : ℝ) (h : a > 0) :
  tangent_condition a b → a = 3 ∧ b = 3 :=
sorry

-- Definitions based on part 2 conditions
def h (x : ℝ) : ℝ := f x 3 + g x (-9)

-- Statement for part 2
theorem find_k_range (k : ℝ) :
  max_value_condition 3 (-9) h k → k ≤ -3 :=
sorry

end find_a_b_find_k_range_l357_357797


namespace num_subsets_containing_6_l357_357827

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l357_357827


namespace ratio_odd_even_divisors_l357_357132

-- Defining M as the given product
def M : ℕ := 26 * 26 * 45 * 252

-- Prove the ratio of the sum of the odd divisors to the sum of the even divisors is 1:16
theorem ratio_odd_even_divisors : 
  let sum_odd_divisors := ∑ d in (Nat.divisors M).filter (λ d, d % 2 = 1), d
  let sum_even_divisors := ∑ d in (Nat.divisors M).filter (λ d, d % 2 = 0), d
  sum_odd_divisors / sum_even_divisors = 1 / 16 :=
  sorry

end ratio_odd_even_divisors_l357_357132


namespace leading_coeff_P_l357_357698

def P1 : Polynomial ℝ := Polynomial.monomial 5 (-5) + Polynomial.monomial 4 5 + Polynomial.monomial 3 (-10)
def P2 : Polynomial ℝ := Polynomial.monomial 5 8 + Polynomial.C 24
def P3 : Polynomial ℝ := Polynomial.monomial 5 (-9) + Polynomial.monomial 3 (-3) + Polynomial.C (-6)

def P : Polynomial ℝ := P1 + P2 + P3

theorem leading_coeff_P : Polynomial.leadingCoeff P = -6 := 
by sorry

end leading_coeff_P_l357_357698


namespace circle_line_probability_intersect_l357_357381

theorem circle_line_probability_intersect (b : ℝ) (hb : -5 ≤ b ∧ b ≤ 5) :
  probability (∃ x y : ℝ, x^2 + y^2 = 4 ∧ y = x + b) = (2 * Real.sqrt 2) / 5 :=
sorry

end circle_line_probability_intersect_l357_357381


namespace part1_l357_357061

variable {a b : ℝ}
variable {A B C : ℝ}
variable {S : ℝ}

-- Given Conditions
def is_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (b * Real.cos C - c * Real.cos B = 2 * a) ∧ (c = a)

-- To prove
theorem part1 (h : is_triangle A B C a b a) : B = 2 * Real.pi / 3 := sorry

end part1_l357_357061


namespace fraction_squared_0_0625_implies_value_l357_357225

theorem fraction_squared_0_0625_implies_value (x : ℝ) (hx : x^2 = 0.0625) : x = 0.25 :=
sorry

end fraction_squared_0_0625_implies_value_l357_357225


namespace quadratic_real_roots_l357_357035

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ∧ (∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) → k ≥ -1 :=
by
  sorry

end quadratic_real_roots_l357_357035


namespace area_bounded_by_curves_l357_357630

noncomputable def area_of_figure_bounded_by_curves : ℝ :=
  let f_x (t : ℝ) := 6 * Real.cos t
  let f_y (t : ℝ) := 4 * Real.sin t
  let g_y : ℝ := 2 * Real.sqrt 3
  let t1 := Real.arcsin (g_y / 4)
  let t2 := π - t1
  let bounded_area := ∫ t in t2..t1, f_y t * (f_x' t) in
  let rectangle_area := g_y * (f_x t1 - f_x t2)
  bounded_area - rectangle_area

theorem area_bounded_by_curves :
  area_of_figure_bounded_by_curves = 4 * π - 12 * Real.sqrt 3 := by
  sorry

end area_bounded_by_curves_l357_357630


namespace possible_values_of_a_l357_357811

noncomputable def problem_statement (a : ℝ) : Prop :=
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 4} : set ℝ) 

theorem possible_values_of_a (a : ℝ) (h : problem_statement a) :
  a = 0 ∨ a = -1 :=
by
  sorry

end possible_values_of_a_l357_357811


namespace solve_first_system_solve_second_system_solve_third_system_l357_357964

-- First system of equations
theorem solve_first_system (x y : ℝ) 
  (h1 : 2*x + 3*y = 16)
  (h2 : x + 4*y = 13) : 
  x = 5 ∧ y = 2 := 
sorry

-- Second system of equations
theorem solve_second_system (x y : ℝ) 
  (h1 : 0.3*x - y = 1)
  (h2 : 0.2*x - 0.5*y = 19) : 
  x = 370 ∧ y = 110 := 
sorry

-- Third system of equations
theorem solve_third_system (x y : ℝ) 
  (h1 : 3 * (x - 1) = y + 5)
  (h2 : (x + 2) / 2 = ((y - 1) / 3) + 1) : 
  x = 6 ∧ y = 10 := 
sorry

end solve_first_system_solve_second_system_solve_third_system_l357_357964


namespace kaleb_spent_on_ferris_wheel_l357_357309

theorem kaleb_spent_on_ferris_wheel
  (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_cost : ℕ)
  (h1 : initial_tickets = 6) (h2 : remaining_tickets = 3) (h3 : ticket_cost = 9) :
  (initial_tickets - remaining_tickets) * ticket_cost = 27 :=
by 
  rw [h1, h2, h3]
  norm_num

end kaleb_spent_on_ferris_wheel_l357_357309


namespace probability_of_two_correct_deliveries_l357_357751

theorem probability_of_two_correct_deliveries :
  (∃ (n k : ℕ), n = 5 ∧ k = 2 ∧ (∃ (correct : Nat), correct = (Nat.choose n k) ∧ correct = 10) ∧
                  (∃ (derangements : Nat), derangements = 1) ∧ 
                  (∃ (total_permutations : Nat), total_permutations = n! ∧ total_permutations = 120) ∧
                  ((correct * derangements) / total_permutations = (1 : ℚ) / 12)) :=
by
  -- Providing the existence witnesses for the existential quantifiers
  use 5, 2,
  split,
  exact rfl, -- n = 5
  split,
  exact rfl, -- k = 2
  use 10, 
  split,
  exact Nat.choose_self.symm, -- correct = Nat.choose 5 2 i.e., correct = 10
  exact rfl, -- confirm correct, which is 10
  use 1, 
  exact rfl, -- derangements = 1
  use 120,
  split,
  exact Nat.factorial_self.symm, -- total_permutations = 5!
  exact rfl, -- confirm total_permutations, which is 120
  -- Now the main part: checking the final probability computation
  show ((10 * 1) / 120 : ℚ) = 1 / 12,
  exact (div_eq_div (mul_one 10) 120).symm, -- verify the probability
  sorry

end probability_of_two_correct_deliveries_l357_357751


namespace julia_change_l357_357458

-- Definitions based on the problem conditions
def price_of_snickers : ℝ := 1.5
def price_of_mms : ℝ := 2 * price_of_snickers
def total_cost_of_snickers (num_snickers : ℕ) : ℝ := num_snickers * price_of_snickers
def total_cost_of_mms (num_mms : ℕ) : ℝ := num_mms * price_of_mms
def total_purchase (num_snickers num_mms : ℕ) : ℝ := total_cost_of_snickers num_snickers + total_cost_of_mms num_mms
def amount_given : ℝ := 2 * 10

-- Prove the change is $8
theorem julia_change : total_purchase 2 3 = 12 ∧ (amount_given - total_purchase 2 3) = 8 :=
by
  sorry

end julia_change_l357_357458


namespace find_colored_copies_l357_357707

noncomputable def total_copies (C W : ℕ) : Prop := C + W = 400
noncomputable def copies_during_regular_hours (Cr W : ℕ) : Prop := Cr + W = 180
noncomputable def copies_after_6pm (Ca W : ℕ) : Prop := Ca = 220 - W
noncomputable def cost_equation (Cr Ca W : ℕ) : Prop := 0.10 * Cr + 0.08 * Ca + 0.05 * W = 22.50

theorem find_colored_copies
  (C W Cr Ca : ℕ)
  (h1 : total_copies C W)
  (h2 : copies_during_regular_hours Cr W)
  (h3 : copies_after_6pm Ca W)
  (h4 : cost_equation Cr Ca W) : C = 300 :=
by sorry

end find_colored_copies_l357_357707


namespace road_points_correct_l357_357633

noncomputable def road_points (d : ℝ) : set (ℝ × ℝ) :=
  { p | ((p.2 = -4*d) ∧ (p.1^2 + 16*d^2 = 25*d^2)) }

theorem road_points_correct (d : ℝ) (hd : d ≠ 0) :
  road_points d = { (3*d, -4*d), (-3*d, -4*d) } :=
begin
  sorry
end

end road_points_correct_l357_357633


namespace decreasing_function_range_l357_357403

theorem decreasing_function_range {f : ℝ → ℝ} (h_decreasing : ∀ x y : ℝ, x < y → f x > f y) :
  {x : ℝ | f (x^2 - 3 * x - 3) < f 1} = {x : ℝ | x < -1 ∨ x > 4} :=
by
  sorry

end decreasing_function_range_l357_357403


namespace series_sum_eq_quarter_l357_357696

noncomputable def sum_series : Real :=
  ∑' (n : ℕ) in Finset.range (Nat.succ n), (3^n / (1 + 3^n + 3^(n+1) + 3^(2*n + 1)))

theorem series_sum_eq_quarter : 
  sum_series = (1 / 4) :=
sorry

end series_sum_eq_quarter_l357_357696


namespace roots_polynomial_value_l357_357945

theorem roots_polynomial_value (a b c : ℝ) 
  (h1 : a + b + c = 15)
  (h2 : a * b + b * c + c * a = 25)
  (h3 : a * b * c = 12) :
  (2 + a) * (2 + b) * (2 + c) = 130 := 
by
  sorry

end roots_polynomial_value_l357_357945


namespace exists_n_multiple_of_3_pure_imaginary_l357_357724

def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ y : ℝ, z = 0 + y * (complex.I)

theorem exists_n_multiple_of_3_pure_imaginary :
  ∃ n : ℕ, 0 < n ∧ is_purely_imaginary (⟨3, 0⟩ / (⟨3/2, 0⟩ + ⟨0, real.sqrt 3/2⟩) ^ n) ↔
  ∃ k : ℕ, 0 < k ∧ 3 * k = n :=
sorry

end exists_n_multiple_of_3_pure_imaginary_l357_357724


namespace part_a_l357_357255

theorem part_a : 
  ∃ (pairs : list (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ pairs → x.1 + x.2 ∈ 
      {p ∈ primes | p ∈ {5, 7, 11, 13, 19, 23}}) ∧
    pairs.length = 6 ∧ 
    (∀ (x y : ℕ × ℕ), x ≠ y → x.1 ≠ x.2 ∧ y.1 ≠ y.2) := 
sorry

end part_a_l357_357255


namespace box_surface_area_l357_357645

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end box_surface_area_l357_357645


namespace sequence_formula_lambda_value_l357_357387

noncomputable def a_seq (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) : Prop :=
  (a 1 = 2) ∧
  (a 2 = 3) ∧
  (∀ n ≥ 2, S (n + 1) + S (n - 1) = 2 * S n + 1) ∧
  (S (n + 1) - S n = a (n + 1)) ∧
  (S n - S (n - 1) = a n)

theorem sequence_formula (a S : ℕ → ℕ) : a_seq a S n → ∀ n ≥ 1, a n = n + 1 := sorry

noncomputable def b_seq (b : ℕ → ℕ) (a : ℕ → ℕ) (λ : ℤ) : Prop :=
  ∀ n ≥ 1, b n = 4^n + (-1)^(n-1) * λ * 2^(a n)

theorem lambda_value (b : ℕ → ℕ) (a : ℕ → ℕ) (λ : ℤ) : 
  (∀ n ≥ 1, a n = n + 1) → (b_seq b a λ) → λ = -1 := sorry

end sequence_formula_lambda_value_l357_357387


namespace tangent_construction_l357_357390

theorem tangent_construction (α : ℝ) (a b r : ℝ) (hα : 0 < α ∧ α < π / 2) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_r_pos : 0 < r) :
  ∃ T : ℝ, ∃ A B : ℝ,  
    (AT / TB = a / b) ∧ 
    (OT = r) ∧
    (OT ⊥ AB) :=
sorry

end tangent_construction_l357_357390


namespace arithmetic_sequence_sum_a3_a4_a5_l357_357438

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_a3_a4_a5
  (ha : is_arithmetic_sequence a d)
  (h : a 2 + a 3 + a 4 = 12) : 
  (7 * (a 0 + a 6)) / 2 = 28 := 
sorry

end arithmetic_sequence_sum_a3_a4_a5_l357_357438


namespace sum_of_coefficients_with_y_l357_357612

theorem sum_of_coefficients_with_y :
  let expression := (5 * x + 3 * y + 2) * (2 * x + 5 * y + 7)
  let terms_with_y := [31 * y, 15 * y^2, 31 * x * y]
  sum (terms_with_y.map (λ t, coeff y t)) = 77 :=
by
  sorry

end sum_of_coefficients_with_y_l357_357612


namespace constant_term_expansion_eq_90720_l357_357223

noncomputable def constant_term_of_expansion : ℕ :=
  let coeff := binom(8, 4)
  in coeff * (3 ^ 4) * (2 ^ 4)

theorem constant_term_expansion_eq_90720 :
  constant_term_of_expansion = 90720 := by
  sorry

end constant_term_expansion_eq_90720_l357_357223


namespace red_minus_white_more_l357_357686

variable (flowers_total yellow_white red_yellow red_white : ℕ)
variable (h1 : flowers_total = 44)
variable (h2 : yellow_white = 13)
variable (h3 : red_yellow = 17)
variable (h4 : red_white = 14)

theorem red_minus_white_more : 
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end red_minus_white_more_l357_357686


namespace other_number_eq_462_l357_357986

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l357_357986


namespace x_minus_y_eq_neg_200_l357_357371

theorem x_minus_y_eq_neg_200 (x y : ℤ) (h1 : x + y = 290) (h2 : y = 245) : x - y = -200 := by
  sorry

end x_minus_y_eq_neg_200_l357_357371


namespace equilateral_triangle_side_length_l357_357957

theorem equilateral_triangle_side_length
  (P A B C : Type)
  [metric_space P]
  [metric_space A]
  [metric_space B]
  [metric_space C]
  (PA PB PC : ℝ)
  (hPA : PA = 5)
  (hPB : PB = 7)
  (hPC : PC = 8) :
  ∃ AB : ℝ, AB = sqrt 129 := by
  sorry

end equilateral_triangle_side_length_l357_357957


namespace find_principal_amount_l357_357239

noncomputable def principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (SI * 100) / (R * T)

theorem find_principal_amount :
  principal_amount 130 4.166666666666667 4 = 780 :=
by
  -- Sorry is used to denote that the proof is yet to be provided
  sorry

end find_principal_amount_l357_357239


namespace maximal_rectangle_l357_357995

theorem maximal_rectangle 
  (A B C A_1 B_1 M : Point) 
  (h_triangle : right_triangle A B C)
  (h_C : right_angle A C B)
  (h_A1 : A_1 ∈ line_segment B C)
  (h_B1 : B_1 ∈ line_segment A C)
  (h_M : M ∈ line_segment A B) :
  maximal_area A B C A_1 B_1 M ↔ M = midpoint A B :=
sorry

end maximal_rectangle_l357_357995


namespace median_on_hypotenuse_length_l357_357386

theorem median_on_hypotenuse_length
  (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) (right_triangle : (a ^ 2 + b ^ 2) = c ^ 2) :
  (1 / 2) * c = 5 :=
  sorry

end median_on_hypotenuse_length_l357_357386


namespace sum_of_roots_l357_357935

theorem sum_of_roots :
  ∀ (x1 x2 : ℝ), (x1*x2 = 2 ∧ x1 + x2 = 3 ∧ x1 ≠ x2) ↔ (x1*x2 + 3*x1*x2 = 2 * x1 * x2 * x1:     by sorry

end sum_of_roots_l357_357935


namespace compute_g_2023_l357_357804

noncomputable def g (x : ℕ) : ℕ
axiom g_function_eq (a b : ℕ) : g (a + b) = g a + g b - 3 * g (a * b)
axiom g_at_1 : g 1 = 2

theorem compute_g_2023 : g 2023 = 2 :=
sorry

end compute_g_2023_l357_357804


namespace circle_line_probability_intersect_l357_357382

theorem circle_line_probability_intersect (b : ℝ) (hb : -5 ≤ b ∧ b ≤ 5) :
  probability (∃ x y : ℝ, x^2 + y^2 = 4 ∧ y = x + b) = (2 * Real.sqrt 2) / 5 :=
sorry

end circle_line_probability_intersect_l357_357382


namespace june_ride_time_l357_357067

theorem june_ride_time (dist1 time1 dist2 time2 : ℝ) (h : dist1 = 2 ∧ time1 = 8 ∧ dist2 = 5 ∧ time2 = 20) :
  (dist2 / (dist1 / time1) = time2) := by
  -- using the defined conditions
  rcases h with ⟨h1, h2, h3, h4⟩
  rw [h1, h2, h3, h4]
  -- simplifying the expression
  sorry

end june_ride_time_l357_357067


namespace number_of_valid_starting_positions_l357_357915

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 1

-- Sequence construction
def sequence_construction (x_n : ℕ → ℝ) : Prop :=
  ∀ n, hyperbola x_n (2 * (x_n - x_n n)) ∧
       x_n (n+1) = orthogonal_projection (x_n (n+1), 2 * (x_n (n+1) - x_n n))

-- Theorem: Number of valid starting positions
theorem number_of_valid_starting_positions : 
  ∃ (P_0 : ℕ → ℝ), sequence_construction P_0 ∧ P_0 0 = P_0 1024 →
  P_0 ∈ {x : ℝ // (1 ≤ x ∧ x ≤ 2^(1024) - 2)} → 
  P_0.card = 2^(1024) - 2 :=
sorry

end number_of_valid_starting_positions_l357_357915


namespace find_d_l357_357865

theorem find_d (d : ℝ) (h1 : ∃ (x y : ℝ), y = x + d ∧ x = -y + d ∧ x = d-1 ∧ y = d) : d = 1 :=
sorry

end find_d_l357_357865


namespace sample_size_correct_l357_357651

-- Define the conditions as lean variables
def total_employees := 120
def male_employees := 90
def female_sample := 9

-- Define the proof problem statement
theorem sample_size_correct : ∃ n : ℕ, (total_employees - male_employees) / total_employees = female_sample / n ∧ n = 36 := by 
  sorry

end sample_size_correct_l357_357651


namespace sum_of_coordinates_l357_357158

-- Define the points C and D and the conditions
def point_C : ℝ × ℝ := (0, 0)

def point_D (x : ℝ) : ℝ × ℝ := (x, 5)

def slope_CD (x : ℝ) : Prop :=
  (5 - 0) / (x - 0) = 3 / 4

-- The required theorem to be proved
theorem sum_of_coordinates (D : ℝ × ℝ)
  (hD : D.snd = 5)
  (h_slope : slope_CD D.fst) :
  D.fst + D.snd = 35 / 3 :=
sorry

end sum_of_coordinates_l357_357158


namespace functional_equation_l357_357736

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
begin
  sorry
end

end functional_equation_l357_357736


namespace value_of_a_l357_357325

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l357_357325


namespace isabel_piggy_bank_balance_l357_357896

theorem isabel_piggy_bank_balance :
  let P0 := 204 : ℝ,
      P1 := P0 - 0.40 * P0,
      P2 := P1 - 0.50 * P1 - 0.20 * P1,
      P3 := P2 - 0.30 * P2,
      P4 := P3 - 0.10 * P3
  in P4 ≈ 23.13 :=
by
  let P0 : ℝ := 204
  let P1 := P0 - 0.40 * P0
  let P2 := P1 - 0.50 * P1 - 0.20 * P1
  let P3 := P2 - 0.30 * P2
  let P4 := P3 - 0.10 * P3
  have : P4 ≈ 23.13 := sorry
  exact this

end isabel_piggy_bank_balance_l357_357896


namespace sufficient_but_not_necessary_l357_357126

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1 / a^2) : a^2 > 1 / a ∧ ¬ ∀ a, a^2 > 1 / a → a > 1 / a^2 :=
by
  sorry

end sufficient_but_not_necessary_l357_357126


namespace greatest_three_digit_number_l357_357601

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end greatest_three_digit_number_l357_357601


namespace sin2_cos3_neg_l357_357579

theorem sin2_cos3_neg : (∃ θ₂ ∈ Ioo (π / 2) π, θ₂ = 2)  ∧ (∃ θ₃ ∈ Ioo (π / 2) π, θ₃ = 3) → sin 2 * cos 3 < 0 :=
by
  sorry

end sin2_cos3_neg_l357_357579


namespace tangent_line_condition_l357_357799

noncomputable def f (x m : ℝ) : ℝ := Real.exp x - m * x + 1

theorem tangent_line_condition (m : ℝ) :
  (¬ ∃ x : ℝ, deriv (λ t, Real.exp t - m * t + 1) x = -1 / Real.exp 1) ↔ m ∈ set.Iic (1 / Real.exp 1) :=
by
  sorry

end tangent_line_condition_l357_357799


namespace class_namesake_impossible_l357_357375

theorem class_namesake_impossible :
  (∃ A7A I7A R7A A7B I7B R7B : Finset String, 
    A7A.card = I7A.card ∧ I7A.card = 3 ∧
    R7A.card = 1 ∧
    A7B.card = I7B.card ∧ I7B.card = 3 ∧
    R7B.card = 1 ∧
    (A7A ∪ I7A ∪ R7A).card = 4 ∧
    (A7B ∪ I7B ∪ R7B).card = 4 ∧
    A7A ∩ A7B = ∅ ∧ I7A ∩ I7B = ∅ ∧ R7A ∩ R7B = ∅) →
  false :=
sorry

end class_namesake_impossible_l357_357375


namespace log_equation_solution_l357_357720

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

theorem log_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 
    log_base 3 (x - 1) + log_base (real.sqrt 3) (x^2 - 1) + log_base (1/3) (x - 1) = 3 ∧
    x = real.sqrt (1 + 3 * real.sqrt 3) :=
by
  sorry

end log_equation_solution_l357_357720


namespace find_n_l357_357626

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 48) (h2 : Nat.gcd n 16 = 8): n = 24 := by
  sorry

end find_n_l357_357626


namespace mike_ride_length_is_34_l357_357624

noncomputable theory

def mikeCost (m : ℕ) : ℝ := 2.50 + 0.25 * m
def annieCost : ℝ := 2.50 + 5.00 + 0.25 * 14

theorem mike_ride_length_is_34 (m : ℕ) : 
  mikeCost m = annieCost → m = 34 :=
by
  unfold mikeCost annieCost
  sorry

end mike_ride_length_is_34_l357_357624


namespace max_sum_products_l357_357206

theorem max_sum_products (x y z w : ℕ) (h : {x, y, z, w} = {2, 4, 6, 8}) :
  xy + yz + zw + wx ≤ 60 := by
  sorry

end max_sum_products_l357_357206


namespace solution_x_percentage_of_alcohol_l357_357671

variable (P : ℝ) -- percentage of alcohol by volume in solution x, in decimal form

theorem solution_x_percentage_of_alcohol :
  (0.30 : ℝ) * 200 + P * 200 = 0.20 * 400 → P = 0.10 :=
by
  intro h
  sorry

end solution_x_percentage_of_alcohol_l357_357671


namespace condylure_moves_change_color_l357_357930

theorem condylure_moves_change_color (m n : ℕ) (hm : m > 0) (hn : n > 0) :
    ∃ f : ℤ × ℤ → bool, 
    ∀ pos : ℤ × ℤ, 
    f pos ≠ f (pos.1 + m, pos.2 + n) ∧ 
    f pos ≠ f (pos.1 - m, pos.2 - n) ∧ 
    f pos ≠ f (pos.1 + n, pos.2 + m) ∧ 
    f pos ≠ f (pos.1 - n, pos.2 - m) := 
sorry

end condylure_moves_change_color_l357_357930


namespace probability_calculation_l357_357453

def prob_correct : ℝ := 0.8
def prob_incorrect : ℝ := 0.2

def probability_four_questions_before_advancing : ℝ := 0.128

theorem probability_calculation :
  (∃ scenario1 scenario2 : ℝ,
    scenario1 = prob_correct * prob_correct * prob_correct * prob_incorrect ∧
    scenario2 = prob_incorrect * prob_correct * prob_correct * prob_correct ∧
    scenario1 + scenario2 = probability_four_questions_before_advancing) :=
by {
  sorry
}

end probability_calculation_l357_357453


namespace second_hand_distance_l357_357574

theorem second_hand_distance (r : ℝ) (t : ℝ) (π : ℝ) (hand_length_6cm : r = 6) (time_15_min : t = 15) : 
  ∃ d : ℝ, d = 180 * π :=
by
  sorry

end second_hand_distance_l357_357574


namespace spell_casting_contest_orders_l357_357820

-- Definition for factorial
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem statement: number of ways to order 4 contestants is 4!
theorem spell_casting_contest_orders : factorial 4 = 24 := by
  sorry

end spell_casting_contest_orders_l357_357820


namespace min_value_of_quadratic_l357_357229

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l357_357229


namespace smallest_lcm_4_digit_integers_l357_357016

theorem smallest_lcm_4_digit_integers (k l : ℕ) (h1 : 1000 ≤ k ∧ k ≤ 9999) (h2 : 1000 ≤ l ∧ l ≤ 9999) (h3 : Nat.gcd k l = 11) : Nat.lcm k l = 92092 :=
by
  sorry

end smallest_lcm_4_digit_integers_l357_357016


namespace range_of_a_l357_357788

-- Definition of the function f satisfying the given properties
def f (x : ℝ) : ℝ
def g (x : ℝ) : ℝ := f (1 + x)

-- Given conditions
axiom f_symmetry : ∀ x : ℝ, f(x) = -f(2 - x)
axiom f_increasing_on_domain : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ ≤ x₂ → f(x₁) ≤ f(x₂)
axiom g_inequality : ∀ a : ℝ, 0 < a → 2 * g(log a / log 2) - 3 * g(1) ≤ g(log a / log 1 / 2)

-- Define the range of the real number a
def a_range := { a : ℝ | 0 < a ∧ a ≤ 2 }

-- The main theorem to prove:
theorem range_of_a : ∀ (a : ℝ), (0 < a → 2 * g(log a / log 2) - 3 * g(1) ≤ g(log a / log 1 / 2)) → (0 < a ∧ a ≤ 2) :=
begin
  sorry
end

end range_of_a_l357_357788


namespace annual_growth_rate_proof_max_communities_proof_l357_357276

-- Define the conditions
def initial_investment : ℤ := 100000000
def final_investment : ℤ := 144000000
def years : ℤ := 2
def average_cost_2022 : ℤ := 800000
def cost_increase_rate : ℝ := 0.1
def investment_growth_rate : ℝ := 0.2

-- Define the Lean 4 statements
theorem annual_growth_rate_proof :
  ∃ x : ℝ, (initial_investment : ℝ) * (1 + x)^years = (final_investment : ℝ) ∧ x = investment_growth_rate :=
by
  sorry

theorem max_communities_proof :
  ∃ y : ℕ, y = floor ((final_investment * (1 + real.of_rat investment_growth_rate) : ℝ) / (average_cost_2022 * (1 + cost_increase_rate) : ℝ)) ∧ y = 195 :=
by
  sorry

end annual_growth_rate_proof_max_communities_proof_l357_357276


namespace prob_divisors_remainder_factorization_l357_357072

theorem prob_divisors_remainder_factorization (a b : ℕ) (h1 : Nat.coprime a b) :
  let p := 12
  let n := 2007
  let m := 2000
  let a_factors := p^(n * 2) * p^n
  let b_factors := p^(m * 2) * p^m
  let prob := (b_factors + 1) * (b_factors/2 + 1) / ((a_factors + 1) * (a_factors/2 + 1))
  let form_fraction := a / b = prob
  (a + b) % 2007 = 79 :=
sorry

end prob_divisors_remainder_factorization_l357_357072


namespace map_scale_l357_357154

theorem map_scale (distance_km : ℤ) (segment_cm : ℤ) (km_to_cm : ℤ) (real_distance_cm : ℤ) (scale_factor : ℤ) :
  distance_km = 30 →
  segment_cm = 20 →
  km_to_cm = 100000 →
  real_distance_cm = distance_km * km_to_cm →
  scale_factor = (real_distance_cm / segment_cm) →
  scale_factor = 150000 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end map_scale_l357_357154


namespace ellipse_standard_equation_and_point_l357_357402
  
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2) / 25 + (y^2) / 9 = 1

def exists_dot_product_zero_point (P : ℝ × ℝ) : Prop :=
  let F1 := (-4, 0)
  let F2 := (4, 0)
  (P.1 + 4) * (P.1 - 4) + P.2 * P.2 = 0

theorem ellipse_standard_equation_and_point :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ exists_dot_product_zero_point P ∧ 
    ((P = ((5 * Real.sqrt 7) / 4, 9 / 4)) ∨ (P = (-(5 * Real.sqrt 7) / 4, 9 / 4)) ∨ 
    (P = ((5 * Real.sqrt 7) / 4, -(9 / 4))) ∨ (P = (-(5 * Real.sqrt 7) / 4, -(9 / 4)))) :=
by 
  sorry

end ellipse_standard_equation_and_point_l357_357402


namespace max_value_expression_l357_357081

noncomputable def maximum_value_expression (β α : ℂ) (θ : ℝ) : ℂ :=
  (β - α) / (1 - complex.conj α * β)

theorem max_value_expression (θ : ℝ) (β α : ℂ) (hβ : |β| = 2) (hα : α = complex.exp (complex.I * θ)) (hneq : complex.conj α * β ≠ 1) :
  |maximum_value_expression β α θ| ≤ 4 :=
sorry

end max_value_expression_l357_357081


namespace xiao_ming_returns_and_distance_is_correct_l357_357020

theorem xiao_ming_returns_and_distance_is_correct :
  ∀ (walk_distance : ℝ) (turn_angle : ℝ), 
  walk_distance = 5 ∧ turn_angle = 20 → 
  (∃ n : ℕ, (360 % turn_angle = 0) ∧ n = 360 / turn_angle ∧ walk_distance * n = 90) :=
by
  sorry

end xiao_ming_returns_and_distance_is_correct_l357_357020


namespace abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l357_357580

theorem abs_eq_ax_plus_1_one_negative_root_no_positive_roots (a : ℝ) :
  (∃ x : ℝ, |x| = a * x + 1 ∧ x < 0) ∧ (∀ x : ℝ, |x| = a * x + 1 → x ≤ 0) → a > -1 :=
by
  sorry

end abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l357_357580


namespace sum_of_two_cubes_lt_1000_l357_357001

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357001


namespace max_elements_in_A_l357_357129

noncomputable def max_elements_A (n : ℕ) (A : set (set (fin n))) : ℕ :=
if h : (∀ (a b ∈ A), a ⊆ b → a = b) then 
  nat.choose n ⌊n / 2⌋ 
else 
  0

theorem max_elements_in_A (n : ℕ) (A : set (set (fin n))) 
  (h : ∀ (a b ∈ A), a ⊆ b → a = b) : 
  ∃ k : ℕ, k = nat.choose n ⌊n / 2⌋ ∧ k = max_elements_A n A :=
begin
  use nat.choose n (n / 2),
  split,
  { sorry },
  { unfold max_elements_A,
    rw if_pos,
    { exact rfl },
    { exact h } }
end

end max_elements_in_A_l357_357129


namespace monotonically_increasing_interval_l357_357991

open Real

noncomputable def sin_interval := 
  ∀ k : ℤ, k * π + π / 3 ≤ x ∧ x ≤ k * π + 5 * π / 6

theorem monotonically_increasing_interval (k : ℤ) :
  ∃ x : ℝ, sin_interval k :=
sorry

end monotonically_increasing_interval_l357_357991


namespace rational_coordinates_of_circumcenter_l357_357551

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l357_357551


namespace num_colorings_l357_357729

theorem num_colorings (a : ℕ → ℕ) (r b y : set ℕ → Prop) :
  (∀ x, (r x → ∃ y, r y ∧ y > x ∧ even x ↔ odd y) ∧
        (b x → ∃ y, b y ∧ y > x ∧ even x ↔ odd y) ∧
        (y x → ∃ y, y y ∧ y > x ∧ even x ↔ odd y)) ∧
  (∀ S, (r S ∧ b S ∧ y S) → ∃ c, even (min c S)) →
  (a 1 = 3) ∧ (a n = 2 * a (n - 1) + 3) →
  ∀ n, a n = 3 * 2 ^ n - 3 :=
by sorry

end num_colorings_l357_357729


namespace matrices_are_equal_l357_357131

namespace MatrixProof

variables {n : ℕ} (x y : Fin n → ℝ)

def A (i j : Fin n) : ℕ :=
  if x i + y j ≥ 0 then 1 else 0

variable {B : Matrix (Fin n) (Fin n) ℕ}

-- Specification for matrix B
axiom B_spec : (∀ i, ∑ j, B i j = ∑ j, A x y i j) ∧ (∀ j, ∑ i, B i j = ∑ i, A x y i j)

theorem matrices_are_equal (B_spec : (∀ i, ∑ j, B i j = ∑ j, A x y i j) ∧ (∀ j, ∑ i, B i j = ∑ i, A x y i j)) : 
  ∀ i j, A x y i j = B i j :=
sorry

end MatrixProof

end matrices_are_equal_l357_357131


namespace original_number_is_10_l357_357857

theorem original_number_is_10 (x : ℤ) (h : 2 * x + 3 = 23) : x = 10 :=
sorry

end original_number_is_10_l357_357857


namespace active_volcanoes_count_l357_357660

theorem active_volcanoes_count (V : ℕ):
  let V' := V
  let remaining_after_first_two_months := 0.8 * V'
  let remaining_after_half_a_year := 0.6 * remaining_after_first_two_months
  let remaining_at_end_of_year := 0.5 * remaining_after_half_a_year
  remaining_at_end_of_year = 48 → V = 200 :=
by {
  assume h
  have h1 : remaining_after_first_two_months = 0.8 * V := rfl,
  have h2 : remaining_after_half_a_year = 0.48 * V := by rw [h1],
  have h3 : remaining_at_end_of_year = 0.24 * V := by rw [h2],
  rw [h3] at h,
  linarith [h],
  sorry
}

end active_volcanoes_count_l357_357660


namespace Px_is_x_plus_1_l357_357263

noncomputable def P : Polynomial ℤ := sorry

axiom posPoly (n : ℕ) (hn : n > 0) : eval n P > n

def sequence : ℕ → ℕ
| 0     := 1
| (n+1) := (eval (sequence n) P).toNat

axiom divisibility (N : ℕ) (hN : N > 0) : ∃ n : ℕ, N ∣ sequence n

theorem Px_is_x_plus_1 : P = Polynomial.C 1 * Polynomial.X + Polynomial.C 1 :=
begin
  sorry -- proof goes here
end

end Px_is_x_plus_1_l357_357263


namespace car_arrives_before_bus_l357_357647

theorem car_arrives_before_bus
  (d : ℝ) (s_bus : ℝ) (s_car : ℝ) (v : ℝ)
  (h1 : d = 240)
  (h2 : s_bus = 40)
  (h3 : s_car = v)
  : 56 < v ∧ v < 120 := 
sorry

end car_arrives_before_bus_l357_357647


namespace find_a_given_coefficient_l357_357405

theorem find_a_given_coefficient (a : ℝ) :
  (∀ x : ℝ, a ≠ 0 → x ≠ 0 → a^4 * x^4 + 4 * a^3 * x^2 * (1/x) + 6 * a^2 * (1/x)^2 * x^4 + 4 * a * (1/x)^3 * x^6 + (1/x)^4 * x^8 = (ax + 1/x)^4) → (4 * a^3 = 32) → a = 2 :=
by
  intros H1 H2
  sorry

end find_a_given_coefficient_l357_357405


namespace number_of_books_l357_357897

-- Define the conditions
def book_stack_thickness_in_inches : ℕ := 12
def pages_per_inch : ℕ := 80
def pages_per_book : ℕ := 160

-- Define the proof problem: Jack has 6 books
theorem number_of_books (book_stack_thickness_in_inches = 12) 
                        (pages_per_inch = 80)
                        (pages_per_book = 160) :
    12 * 80 / 160 = 6 := 
by
    -- The actual proof would go here
    sorry

end number_of_books_l357_357897


namespace find_m_plus_M_l357_357933

-- Given conditions
def cond1 (x y z : ℝ) := x + y + z = 4
def cond2 (x y z : ℝ) := x^2 + y^2 + z^2 = 6

-- Proof statement: The sum of the smallest and largest possible values of x is 8/3
theorem find_m_plus_M :
  ∀ (x y z : ℝ), cond1 x y z → cond2 x y z → (min (x : ℝ) (max x y) + max (x : ℝ) (min x y) = 8 / 3) :=
by
  sorry

end find_m_plus_M_l357_357933


namespace angle_of_inclination_tangent_line_at_point_l357_357743

theorem angle_of_inclination_tangent_line_at_point :
  let y := λ x : ℝ, (1/3) * x^3 - 2,
      point := (-1 : ℝ, -7/3 : ℝ) in
  let dydx := λ x : ℝ, x^2 in
  let slope := dydx point.1 in
  let theta := Real.arctan slope in
  Real.to_deg theta = 45 :=
by
  sorry

end angle_of_inclination_tangent_line_at_point_l357_357743


namespace helen_chocolate_chip_cookies_l357_357426

theorem helen_chocolate_chip_cookies :
  let cookies_yesterday := 527
  let cookies_morning := 554
  cookies_yesterday + cookies_morning = 1081 :=
by
  let cookies_yesterday := 527
  let cookies_morning := 554
  show cookies_yesterday + cookies_morning = 1081
  -- The proof is omitted according to the provided instructions 
  sorry

end helen_chocolate_chip_cookies_l357_357426


namespace projection_norm_ratio_l357_357912

variable {V : Type*} [InnerProductSpace ℝ V]

-- Definitions for the projections
def projection (u v : V) : V := (inner u v / inner v v) • v

-- Given conditions
variables (v w : V)
variables (hv : v ≠ 0) (hw : w ≠ 0)
variable (h_proj : ∥projection v w∥ / ∥v∥ = 3 / 4)

-- Statement to be proven
theorem projection_norm_ratio (p : V) (q : V)
  (hp : p = projection v w)
  (hq : q = projection p v) :
  ∥q∥ / ∥v∥ = 9 / 16 := by
  sorry

end projection_norm_ratio_l357_357912


namespace equilateral_triangles_formed_l357_357975

theorem equilateral_triangles_formed :
  ∀ k : ℤ, -8 ≤ k ∧ k ≤ 8 →
  (∃ triangles : ℕ, triangles = 426) :=
by sorry

end equilateral_triangles_formed_l357_357975


namespace percentage_of_water_in_dried_grapes_l357_357376

noncomputable def freshGrapesWaterWeight := 0.9
noncomputable def totalFreshGrapesWeight := 10
noncomputable def dryGrapesWeight := 1.25

theorem percentage_of_water_in_dried_grapes :
  (1 - totalFreshGrapesWeight * (1 - freshGrapesWaterWeight) / dryGrapesWeight) * 100 = 20 := by
  sorry

end percentage_of_water_in_dried_grapes_l357_357376


namespace number_of_roots_l357_357570

-- Define the function f(x) = 1 - x - x * ln x
def f (x : ℝ) : ℝ := 1 - x - x * Real.log x

-- The conditions of the problem:
-- 1. f'(x) is the derivative of f(x) which is less than 0 for all x in the domain (0, +∞), indicating the function is decreasing in this domain.
-- 2. f(1) = 0.
-- 3. We are looking for the number of roots in the domain (0, +∞).

theorem number_of_roots : ∀ x > 0, f(x) = 0 → x = 1 :=
  by
  sorry

end number_of_roots_l357_357570


namespace no_integer_root_l357_357160

-- Given a polynomial f with integer coefficients.
variables {R : Type*} [Ring R] [IsDomain R]
variables (f : R[X]) {a b c : ℤ} 

-- Conditions: Absolute values are 1 at three distinct integers a, b, and c.
def polynomial_condition (f : ℤ[X]) (a b c : ℤ) : Prop :=
  (|f.eval a| = 1) ∧ (|f.eval b| = 1) ∧ (|f.eval c| = 1) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c)

-- Theorem statement: Prove no integer root exists under these conditions.
theorem no_integer_root (f : ℤ[X]) (a b c x0 : ℤ) (h : polynomial_condition f a b c) :
  ¬ f.is_root x0 :=
sorry

end no_integer_root_l357_357160


namespace sum_of_roots_eq_three_l357_357937

theorem sum_of_roots_eq_three (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + 2 = 0)
  (h2 : x2^2 - 3*x2 + 2 = 0) 
  (h3 : x1 ≠ x2) : 
  x1 + x2 = 3 := 
sorry

end sum_of_roots_eq_three_l357_357937


namespace square_perimeter_l357_357966

noncomputable def side_length (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area (side_length : ℝ) : ℝ := side_length ^ 2

noncomputable def side_length_from_area (area : ℝ) : ℝ := Real.sqrt area

theorem square_perimeter
  (perimeter_P : ℝ)
  (h_perimeter_P : perimeter_P = 32)
  (h_area_ratio : ∀ (area_Q area_P : ℝ), area_Q = area_P / 3) :
  let side_length_P := side_length perimeter_P,
      area_P := area side_length_P,
      area_Q := area_P / 3,
      side_length_Q := side_length_from_area area_Q,
      perimeter_Q := 4 * side_length_Q
  in perimeter_Q = 32 * Real.sqrt 3 / 3 := sorry

end square_perimeter_l357_357966


namespace tiling_with_dominoes_l357_357599

theorem tiling_with_dominoes (m n : ℕ) (h : ∃ k : ℕ, 6 * k = m * n) :
  ∃ (T : ℕ → ℕ → bool), (∀ i j, T i j = true ∨ T i j = false) ∧
  (∃ (P : list (ℕ × ℕ)), ∀ (x : ℕ × ℕ), list.mem x P → (∃ (d : ℕ × ℕ), d = (2, 1) ∧ (x.fst + d.fst ≤ m) ∧ (x.snd + d.snd ≤ n))) :=
sorry

end tiling_with_dominoes_l357_357599


namespace box_surface_area_l357_357643

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end box_surface_area_l357_357643


namespace minimum_guests_l357_357185

theorem minimum_guests (total_food : ℝ) (max_food_per_guest : ℝ) (total_food_eq : total_food = 319) (max_food_per_guest_eq : max_food_per_guest = 2) : 
  (ceil (total_food / max_food_per_guest)) = 160 :=
by
  -- Given total food consumption is 319 pounds
  have h1 : total_food = 319, from total_food_eq,
  -- Given maximum food consumption per guest is 2 pounds
  have h2 : max_food_per_guest = 2, from max_food_per_guest_eq,
  -- Thus, the minimum number of guests is ceil(319 / 2) = 160
  sorry

end minimum_guests_l357_357185


namespace line_properties_l357_357202

theorem line_properties (x y : ℝ) : (∃ β b, β = 30 ∧ b = 2 ∧ (√3) * x - 3 * y + 6 = 0) :=
begin
  sorry,
end

end line_properties_l357_357202


namespace part1_part2_l357_357112

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357112


namespace part1_general_formula_part2_find_d_l357_357090

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357090


namespace trapezoid_area_l357_357077

/-- Define points A, B, C, D, X, Y satisfying the given conditions -/
variables {A B C D X Y : Type}
variables [has_point A] [has_point B] [has_point C] [has_point D] [has_point X] [has_point Y]

/-- Define the coordinates of points based on the given distances and conditions -/
namespace geometry_problem

/-- Define ABCD as an isosceles trapezoid with specific properties -/
structure isosceles_trapezoid (A B C D : Point) :=
(parallel_ad_bc : parallel A D B C)
(equal_ab_cd : dist A B = dist C D)
(perpendicular_ab_ad : perp A B A D)
(A := (4,0):Point)
(C := (9,0):Point)
(B := (6,4):Point)
(D := (0,-4):Point)
(X := (0,0):Point)
(Y := (6,0):Point)

/-- Proof that the area of the trapezoid is 24, given the conditions -/
theorem trapezoid_area {A B C D X Y : Point} 
(h_isosceles: isosceles_trapezoid A B C D) 
(h_angle_AXD_90 : angle A X D = 90)
(h_AX : dist A X = 4)
(h_XY : dist X Y = 2)
(h_YC : dist Y C = 3) :
area ABCD = 24 :=
sorry

end geometry_problem

end trapezoid_area_l357_357077


namespace min_chips_1x2100_strip_l357_357467

theorem min_chips_1x2100_strip : 
  ∃ (n : ℕ), 
    ((∀ i : ℕ, 1 ≤ i → i ≤ 2100 - n → abs_diff_recorded
       (i - 1).count_chips_left (i + 1).count_chips_right ≠ 0 ∧ 
       abs_diff_recorded
       (i - 1).count_chips_left (i + 1).count_chips_right ≠ (j : ℕ) ∀ j ≠ i)
    ∧ 
    ((2100 - n) ≤ (n + 1) / 2) 
    → 
    n = 1400

end min_chips_1x2100_strip_l357_357467


namespace tony_gas_expense_in_4_weeks_l357_357591

theorem tony_gas_expense_in_4_weeks :
  let miles_per_gallon := 25
  let miles_per_round_trip_per_day := 50
  let travel_days_per_week := 5
  let tank_capacity_in_gallons := 10
  let cost_per_gallon := 2
  let weeks := 4
  let total_miles_per_week := miles_per_round_trip_per_day * travel_days_per_week
  let total_miles := total_miles_per_week * weeks
  let miles_per_tank := miles_per_gallon * tank_capacity_in_gallons
  let fill_ups_needed := total_miles / miles_per_tank
  let total_gallons_needed := fill_ups_needed * tank_capacity_in_gallons
  let total_cost := total_gallons_needed * cost_per_gallon
  total_cost = 80 :=
by
  sorry

end tony_gas_expense_in_4_weeks_l357_357591


namespace part1_general_formula_part2_find_d_l357_357095

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357095


namespace ratio_eliminated_to_remaining_l357_357706

theorem ratio_eliminated_to_remaining (initial_racers : ℕ) (final_racers : ℕ)
  (eliminations_1st_segment : ℕ) (eliminations_2nd_segment : ℕ) :
  initial_racers = 100 →
  final_racers = 30 →
  eliminations_1st_segment = 10 →
  eliminations_2nd_segment = initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3 - final_racers →
  (eliminations_2nd_segment / (initial_racers - eliminations_1st_segment - (initial_racers - eliminations_1st_segment) / 3)) = 1 / 2 :=
by
  sorry

end ratio_eliminated_to_remaining_l357_357706


namespace part1_part2_l357_357111

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357111


namespace find_other_number_l357_357981

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l357_357981


namespace cos_280_eq_cos_n_l357_357744

theorem cos_280_eq_cos_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 180) : (cos (n * (Real.pi / 180))) = (cos (280 * (Real.pi / 180))) → n = 80 :=
by sorry

end cos_280_eq_cos_n_l357_357744


namespace partI_partII_partIII_l357_357175

def f (x : ℝ) (φ : ℝ) : ℝ := Math.sin (2 * x + φ)

theorem partI (hφ : -π < φ ∧ φ < 0) (h_sym : ∃ φ, f (π / 8) φ = 1 ∨ f (π / 8) φ = -1) : φ = -3 * π / 4 :=
sorry

theorem partII (φ : ℝ) (hφ : -π < φ ∧ φ < 0) (hφ_val : φ = -3 * π / 4) 
: ∀ k : ℤ, ∀ x : ℝ, k * π + π / 8 ≤ x ∧ x ≤ k * π + 5 * π / 8 → (0 : ℝ) < f x φ :=
sorry

theorem partIII (φ : ℝ) (hφ : -π < φ ∧ φ < 0) (hφ_val : φ = -3 * π / 4)
: ∀ c : ℝ, ¬ ∃ x y : ℝ, y = f x φ ∧ 5 * x - 2 * y + c = 0 :=
sorry

end partI_partII_partIII_l357_357175


namespace a_n_formula_d_value_l357_357103

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357103


namespace probability_of_at_least_5_heads_l357_357221

/--
We toss a coin four times and then we toss it as many times as there were heads in the first four tosses.
We need to determine if the probability that there will be at least 5 heads in all the tosses is \( \frac{47}{256} \).
-/
theorem probability_of_at_least_5_heads : 
  (∃ n : ℕ, ∃ m : ℕ, (n = 4 ∧ m = ∑ (i : fin n), if coin_toss i then 1 else 0 ∧ 
   (finset.filter (coin_toss) (finset.range (n + m))).card ≥ 5)) → 
  ∑ t in finset.range (n + m), P(toss = heads | toss) = 47 / 256 :=
sorry

end probability_of_at_least_5_heads_l357_357221


namespace min_value_of_quadratic_l357_357227

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l357_357227


namespace sum_of_divisors_l357_357923

def num_divisors (n : ℕ) : ℕ := (finset.range (n + 1)).filter (λ i => n % i = 0).card
def floor_sqrt (n : ℕ) : ℕ := nat.sqrt n

theorem sum_of_divisors (n : ℕ) (k : ℕ) (hk : k = floor_sqrt n) :
  (finset.range (n + 1)).sum (λ i => num_divisors i) = 
  2 * (finset.range (k + 1)).sum (λ i => n / i) - k^2 :=
by
  sorry

end sum_of_divisors_l357_357923


namespace range_of_a_l357_357440

def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * Real.exp x - a * x

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ < 0 ∧ ∀ x : ℝ, f x₀ a ≤ f x a) ↔ a ∈ Ioo (-(1 / Real.exp 1)) 0 :=
sorry

end range_of_a_l357_357440


namespace path_length_vertex_C_l357_357322

-- Define the length of the path traversed by vertex C in terms of the triangle side length and square side length
noncomputable def path_length_of_rotated_equilateral_triangle 
  (triangle_side : ℝ) (square_side : ℝ) : ℝ :=
  if triangle_side = 3 ∧ square_side = 6 then 18 * Real.pi else 0

-- State the theorem
theorem path_length_vertex_C (triangle_side square_side : ℝ) 
(ha : triangle_side = 3) (hb : square_side = 6) : 
  path_length_of_rotated_equilateral_triangle triangle_side square_side = 18 * Real.pi := 
by
  simp [path_length_of_rotated_equilateral_triangle, ha, hb]
  sorry

end path_length_vertex_C_l357_357322


namespace monotonic_on_interval_l357_357974

theorem monotonic_on_interval (k : ℝ) :
  (∀ x y : ℝ, x ≤ y → x ≤ 8 → y ≤ 8 → (4 * x ^ 2 - k * x - 8) ≤ (4 * y ^ 2 - k * y - 8)) ↔ (64 ≤ k) :=
sorry

end monotonic_on_interval_l357_357974


namespace imaginary_part_z_is_correct_l357_357859

open Complex

noncomputable def problem_conditions (z : ℂ) : Prop :=
  (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)

theorem imaginary_part_z_is_correct (z : ℂ) (hz : problem_conditions z) :
  z.im = 4 / 5 :=
sorry

end imaginary_part_z_is_correct_l357_357859


namespace sum_of_two_cubes_lt_1000_l357_357005

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357005


namespace no_solutions_for_any_z_l357_357756

theorem no_solutions_for_any_z (y : ℝ) : 
  (∀ z : ℝ, ¬ ∃ x : ℝ, x^2 + 2 * y^2 + 8 * z^2 - 2 * x * y * z - 9 = 0) ↔
  (3 / Real.sqrt 2 < Real.abs y ∧ Real.abs y ≤ 2 * Real.sqrt 2) :=
by
  sorry

end no_solutions_for_any_z_l357_357756


namespace max_value_of_function_l357_357719

noncomputable def f : ℝ → ℝ := 
  λ x, 3 * Real.sin (x + Real.pi / 9) + 5 * Real.sin (x + (4 * Real.pi / 9))

theorem max_value_of_function : ∃ x : ℝ, f x = 7 :=
by
  sorry

end max_value_of_function_l357_357719


namespace complex_sub_condition_l357_357855

noncomputable theory
open Complex

theorem complex_sub_condition (m n : ℝ) (h : m + Complex.i = (1 + 2 * Complex.i) * (n * Complex.i)) : n - m = 3 :=
sorry

end complex_sub_condition_l357_357855


namespace tiling_chessboard_impossible_l357_357501

theorem tiling_chessboard_impossible (N : ℕ) : 
  ∀ (C1 C2 : ℕ), C1 = 32 ∧ C2 = 30 → ¬(∃ M : ℕ, M = 31 ∧ possible_tiling N C1 C2 M) :=
by
  intros C1 C2 h
  sorry

def possible_tiling N C1 C2 M := false

end tiling_chessboard_impossible_l357_357501


namespace edith_books_count_l357_357347

-- Definitions for the conditions
def num_novels_first_shelf : ℕ := 67  -- 1.2 * 56 = 67.2 (round to nearest whole number)
def num_novels_second_shelf : ℕ := 56
def num_total_novels : ℕ := num_novels_first_shelf + num_novels_second_shelf
def num_writing_books : ℕ := (num_total_novels / 2).ceil  -- rounding to nearest whole number
def total_books : ℕ := num_total_novels + num_writing_books

-- The theorem to be proved
theorem edith_books_count : total_books = 185 := by
  sorry

end edith_books_count_l357_357347


namespace driving_scenario_l357_357617

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end driving_scenario_l357_357617


namespace enlarged_poster_height_l357_357952

-- Define the original and new dimensions
def original_width : ℝ := 3
def original_height : ℝ := 2
def new_width : ℝ := 15

-- Define the scaling factor
def scaling_factor := new_width / original_width

-- Define the new height according to the proportional enlargement
def new_height := original_height * scaling_factor

-- Prove that the new height is 10 inches
theorem enlarged_poster_height : new_height = 10 := by
  -- Sorry to skip the proof which will involve calculation of the scaling factor and the new height
  sorry

end enlarged_poster_height_l357_357952


namespace hyperbola_ratio_FM_to_MN_l357_357417

theorem hyperbola_ratio_FM_to_MN
    (a : ℝ)
    (h_pos : a > 0)
    (M N : ℝ × ℝ)
    (F : ℝ × ℝ)
    (P : ℝ × ℝ)
    (h_hyperbola : ∀ (x y : ℝ), (x, y) ∈ ({p : ℝ × ℝ | p.1 ^ 2 / a - p.2 ^ 2 / a = 1}))
    (h_focus : F = (Real.sqrt (2 * a), 0))
    (h_line_through_F : ∃ (m : ℝ), ∀ (x : ℝ), (x, m * (x - Real.sqrt (2 * a)) + F.2) ∈ ({p : ℝ × ℝ | p.1 ^ 2 / a - p.2 ^ 2 / a = 1}))
    (h_bisector : ∀ (Q : ℝ × ℝ), Q = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) → Q.2 = M.2 / Real.sqrt (1 + ((M.2 - N.2) / (M.1 - N.1)) ^ 2))
    (h_projection : P = (Q.1, 0))
    (h_distance_FP : ∀ (P : ℝ × ℝ), Real.dist F P = Real.sqrt (2 * a * ((M.1 - N.1) / 2)))
    (h_length_MN : Real.dist M N = Real.sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2)) :
    Real.dist F P / Real.dist M N = Real.sqrt 2 / 2 :=
by
  sorry

end hyperbola_ratio_FM_to_MN_l357_357417


namespace kw_price_approx_4266_percent_l357_357708

noncomputable def kw_price_percentage (A B C D E : ℝ) (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E) : ℝ :=
  let total_assets := A + B + C + D + E
  let price_kw := 1.5 * A
  (price_kw / total_assets) * 100

theorem kw_price_approx_4266_percent (A B C D E KW : ℝ)
  (hA : KW = 1.5 * A) (hB : KW = 2 * B) (hC : KW = 2.5 * C) (hD : KW = 2.25 * D) (hE : KW = 3 * E)
  (hB_from_A : B = 0.75 * A) (hC_from_A : C = 0.6 * A) (hD_from_A : D = 0.6667 * A) (hE_from_A : E = 0.5 * A) :
  abs ((kw_price_percentage A B C D E hA hB hC hD hE) - 42.66) < 1 :=
by sorry

end kw_price_approx_4266_percent_l357_357708


namespace geometric_sequence_collinear_vectors_l357_357782

theorem geometric_sequence_collinear_vectors (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (a2 a3 : ℝ)
  (h_a2 : a 2 = a2)
  (h_a3 : a 3 = a3)
  (h_parallel : 3 * a2 = 2 * a3) :
  (a2 + a 4) / (a3 + a 5) = 2 / 3 := 
by
  sorry

end geometric_sequence_collinear_vectors_l357_357782


namespace solve_system_of_equations_l357_357532

theorem solve_system_of_equations (x y m : ℝ) :
  ((m ≠ 1 ∧ m ≠ -2 ∧ x = 6 / (1 - m) ∧ y = (m - 4) / (1 - m)) ∨
   (m = 1 ∨ m = -2 ∧ ¬ ∃ (x y: ℝ), x + (m + 1) * y + m - 2 = 0 ∧ 2 * m * x + 4 * y + 16 = 0)) :=
begin
  sorry
end

end solve_system_of_equations_l357_357532


namespace problem_statement_l357_357018

variables (x y : ℝ)

theorem problem_statement
  (h1 : abs x = 4)
  (h2 : abs y = 2)
  (h3 : abs (x + y) = x + y) : 
  x - y = 2 ∨ x - y = 6 :=
sorry

end problem_statement_l357_357018


namespace f_2005_eq_cos_l357_357927

noncomputable def fn : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.sin x
| (n+1) := λ x, (fn n)' x

theorem f_2005_eq_cos : ∀ x : ℝ, fn 2005 x = Real.cos x :=
by intros; sorry

end f_2005_eq_cos_l357_357927


namespace find_lambda_l357_357425

namespace ParallelVectorProblem

def vector := ℤ × ℤ

def a : vector := (-2, 3)
def b : vector := (3, 1)
def c : vector := (-7, -6)

def parallel_vectors (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_lambda (λ : ℝ) (h : parallel_vectors (a.1 + (λ * b.1), a.2 + (λ * b.2)) c) : λ = 1 :=
sorry

end ParallelVectorProblem


end find_lambda_l357_357425


namespace solve_trig_equation_l357_357531

theorem solve_trig_equation (x : ℝ) (k : ℤ) 
  (h : sin x ^ 6 + cos x ^ 6 = 1 / 4) :
  x = π / 4 + k * π ∨ x = 3 * π / 4 + k * π := sorry

end solve_trig_equation_l357_357531


namespace exists_unique_valid_tuples_l357_357968

theorem exists_unique_valid_tuples :
  ∃! (S : Finset (ℕ × ℕ × ℕ × ℕ)),
  (∀ (t : ℕ × ℕ × ℕ × ℕ), t ∈ S ↔
    (t.1 ∈ Finset.range 5 ∧ t.2.1 ∈ Finset.range 5 ∧ t.2.2.1 ∈ Finset.range 5 ∧ t.2.2.2 ∈ Finset.range 5 ∧
     t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2.1 ∧ t.1 ≠ t.2.2.2 ∧
     t.2.1 ≠ t.2.2.1 ∧ t.2.1 ≠ t.2.2.2 ∧ t.2.2.1 ≠ t.2.2.2 ∧
     (if (t.1 = 1) then (t.2.1 ≠ 1 ∧ t.2.2.1 ≠ 2 ∧ t.2.2.2 ≠ 4)
     else if (t.2.1 ≠ 1) then (t.1 ≠ 1 ∧ t.2.2.1 ≠ 2 ∧ t.2.2.2 ≠ 4)
     else if (t.2.2.1 = 2) then (t.1 ≠ 1 ∧ t.2.1 ≠ 1 ∧ t.2.2.2 ≠ 4)
     else if (t.2.2.2 ≠ 4) then (t.1 ≠ 1 ∧ t.2.1 ≠ 1 ∧ t.2.2.1 ≠ 2)
     else false))) ∧ S.card = 6 := sorry

end exists_unique_valid_tuples_l357_357968


namespace set_intersection_complement_l357_357781

open Set

variable (A B U : Set ℕ)

theorem set_intersection_complement (A B : Set ℕ) (U : Set ℕ) (hU : U = {1, 2, 3, 4})
  (h1 : compl (A ∪ B) = {4}) (h2 : B = {1, 2}) :
  A ∩ compl B = {3} :=
by
  sorry

end set_intersection_complement_l357_357781


namespace hour_hand_rotations_l357_357567

theorem hour_hand_rotations (degrees_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) (rotations_per_day : ℕ) :
  degrees_per_hour = 30 →
  hours_per_day = 24 →
  rotations_per_day = (degrees_per_hour * hours_per_day) / 360 →
  days = 6 →
  rotations_per_day * days = 12 :=
by
  intros
  sorry

end hour_hand_rotations_l357_357567


namespace min_value_of_z_l357_357237

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l357_357237


namespace part1_part2_l357_357513

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -1 ∨ a = -3 := by
  sorry

theorem part2 (a : ℝ) (h : A ∪ B a = A) : a ≤ -3 := by
  sorry

end part1_part2_l357_357513


namespace real_part_of_z_l357_357075

noncomputable def complex_real_part (z : ℂ) : ℝ :=
  z.re

theorem real_part_of_z : ∀ (z : ℂ), |z| = 1 ∧ |z - 1.45| = 1.05 → complex_real_part z = 20/29 :=
by
  intros z h
  sorry

end real_part_of_z_l357_357075


namespace min_value_of_expression_l357_357017

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 3) : 
  ∃ k : ℝ, k = 4 + 2 * Real.sqrt 3 ∧ ∀ z, (z = (1 / (x - 1) + 3 / (y - 1))) → z ≥ k :=
sorry

end min_value_of_expression_l357_357017


namespace concurrency_at_B_l357_357816

-- Define the geometrical configuration
variables (O₁ O₂ A B P₁ Q₁ P₂ Q₂ M₁ M₂ C E D F: Type)
variables [plane_geom O₁ O₂ A B P₁ Q₁ P₂ Q₂ M₁ M₂ C E D F]

-- Conditions: 
-- - O₁ and O₂ are circles intersecting at A and B
-- - External tangents touch O₁ at P₁ and Q₁, O₂ at P₂ and Q₂
-- - M₁ and M₂ are midpoints of P₁Q₁ and P₂Q₂ respectively
-- - Extend AM₁ to intersect O₁ at C, AO₁ to intersect O₁ at E
-- - Extend AM₂ to intersect O₂ at D, AO₂ to intersect O₂ at F

axiom circle_intersection (hO₁ : is_circle O₁) (hO₂ : is_circle O₂) : intersects_at O₁ O₂ A B 
axiom external_tangents_touch (hT₁ : touches P₁ Q₁ O₁) (hT₂ : touches P₂ Q₂ O₂)
axiom midpoints (hM₁ : midpoint P₁ Q₁ M₁) (hM₂ : midpoint P₂ Q₂ M₂)
axiom line_intersections_circle₁ (hC : intersects_at (line_through A M₁) O₁ C) (hE : intersects_at (line_through A (center O₁)) O₁ E)
axiom line_intersections_circle₂ (hD : intersects_at (line_through A M₂) O₂ D) (hF : intersects_at (line_through A (center O₂)) O₂ F)

-- Conclusion: AB, EF, CD are concurrent
theorem concurrency_at_B : concurrent (line_through A B) (line_through E F) (line_through C D) :=
by sorry

end concurrency_at_B_l357_357816


namespace total_handshakes_l357_357308

/-
There are 4 teams of 2 women each, for a total of 8 women.
Each woman shakes hands once with each of the other players except her partner.
Prove that the total number of handshakes is 24.
-/

theorem total_handshakes : 
  ∃ (teams : Finset (Finset ℕ)) (partner : Fin 8 → Fin 8),
    (∀ w : Fin 8, w ≠ partner w ∧ partner w ≠ w) ∧
    teams.card = 4 ∧
    (∀ t ∈ teams, t.card = 2 ∧ ∀ x ∈ t, ∀ y ∈ t, x ≠ y) →
    (∑ x in Finset.univ : Finset (Fin 8), ((Finset.univ.erase x).erase (partner x)).card) / 2 = 24 :=
begin
  sorry
end

end total_handshakes_l357_357308


namespace find_angle_ACD_l357_357047

-- Define the vertices of the quadrilateral
variables {A B C D : Type*}

-- Given angles and side equality
variables (angle_DAC : ℝ) (angle_DBC : ℝ) (angle_BCD : ℝ) (eq_BC_AD : Prop)

-- The given conditions in the problem
axiom angle_DAC_is_98 : angle_DAC = 98
axiom angle_DBC_is_82 : angle_DBC = 82
axiom angle_BCD_is_70 : angle_BCD = 70
axiom BC_eq_AD : eq_BC_AD = true

-- Target angle to be proven
def angle_ACD : ℝ := 28

-- The theorem
theorem find_angle_ACD (h1 : angle_DAC = 98)
                       (h2 : angle_DBC = 82)
                       (h3 : angle_BCD = 70)
                       (h4 : eq_BC_AD) : angle_ACD = 28 := 
by
  sorry  -- Proof of the theorem

end find_angle_ACD_l357_357047


namespace maximum_value_MN_l357_357806

def f (x : ℝ) := 2 * Real.sin x
def g (x : ℝ) := 2 * Real.sqrt 3 * Real.cos x

theorem maximum_value_MN : ∀ m : ℝ, 
  let M := (m, f m)
  let N := (m, g m)
  |M.2 - N.2| ≤ 4 := 
by
  sorry

end maximum_value_MN_l357_357806


namespace other_number_of_given_conditions_l357_357984

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l357_357984


namespace part1_part2_l357_357100

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357100


namespace definite_integral_sqrtx_plus_3_l357_357348

theorem definite_integral_sqrtx_plus_3 :
  ∫ x : ℝ in 0..1, (Real.sqrt x + 3) = 11 / 3 :=
by
  sorry

end definite_integral_sqrtx_plus_3_l357_357348


namespace red_minus_white_more_l357_357687

variable (flowers_total yellow_white red_yellow red_white : ℕ)
variable (h1 : flowers_total = 44)
variable (h2 : yellow_white = 13)
variable (h3 : red_yellow = 17)
variable (h4 : red_white = 14)

theorem red_minus_white_more : 
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end red_minus_white_more_l357_357687


namespace problem1_problem2_l357_357424

variables (k x : ℝ)
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-2, 1)
def c (x : ℝ) : ℝ × ℝ := (3 - 2 * x, 2 + x)

-- First proof problem
theorem problem1 
  (h1 : a = (3, 2)) 
  (h2 : b = (-2, 1)) 
  (h3 : (k * (3, 2).1 + (-2) ≠ (3 + 2 * (-2))) ∨ (k * (3, 2).2 + (1) = (2 + 2 * (1)))) : 
  k = 6 / 5 :=
sorry

-- Second proof problem
theorem problem2 
  (h4 : a = (3, 2)) 
  (h5 : b = (-2, 1)) 
  (h6 : c x = (3 - 2 * x, 2 + x)) 
  (h7 : ∃ x, (sqrt ((3 - 2 * x)^2 + (2 + x)^2) > 0) = false) :
  angle c (x) b = π / 2 :=
sorry

end problem1_problem2_l357_357424


namespace if_received_A_then_answered_all_SA_l357_357151

-- Definitions
def receivedA (answeredAllSA : Bool) (answered90PlusMC : Bool) : Bool :=
  answeredAllSA && answered90PlusMC

-- Prove the statement
theorem if_received_A_then_answered_all_SA (answeredAllSA : Bool) (answered90PlusMC : Bool):
  receivedA answeredAllSA answered90PlusMC = true → answeredAllSA = true :=
by
  intros h1
  unfold receivedA at h1
  cases answeredAllSA
  case false => contradiction
  case true => simp
  sorry

end if_received_A_then_answered_all_SA_l357_357151


namespace vanya_four_times_faster_l357_357146

-- We let d be the total distance, and define the respective speeds
variables (d : ℝ) (v_m v_v : ℝ)

-- Conditions from the problem
-- 1. Vanya starts after Masha
axiom start_after_masha : ∀ t : ℝ, t > 0

-- 2. Vanya overtakes Masha at one-third of the distance
axiom vanya_overtakes_masha : ∀ t : ℝ, (v_v * t) = d / 3

-- 3. When Vanya reaches the school, Masha still has half of the way to go
axiom masha_halfway : ∀ t : ℝ, (v_m * t) = d / 2

-- Goal to prove
theorem vanya_four_times_faster : v_v = 4 * v_m :=
sorry

end vanya_four_times_faster_l357_357146


namespace relationship_between_a_b_c_l357_357767

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem relationship_between_a_b_c : a > b ∧ b > c :=
by
  have ha : a = 2 ^ 0.2 := rfl
  have hb : b = 0.4 ^ 0.2 := rfl
  have hc : c = 0.4 ^ 0.6 := rfl
  sorry

end relationship_between_a_b_c_l357_357767


namespace cube_cross_section_area_l357_357879

theorem cube_cross_section_area
  (A A1 B B1 C C1 D D1 E F : ℝ × ℝ × ℝ)
  (hA_A1 : dist A A1 = 1)
  (hE : E = midpoint C C1)
  (hF : F = midpoint D D1)
  (hC_C1 : dist C C1 = 1)
  (hD_D1 : dist D D1 = 1) :
  let R := sqrt (7 / 10) in
  ∃ (r : ℝ), (r = R ∧ π * r^2 = 7 * π / 10) := 
sorry

end cube_cross_section_area_l357_357879


namespace librarian_took_books_l357_357954

-- Define variables and conditions
def total_books : ℕ := 46
def books_per_shelf : ℕ := 4
def shelves_needed : ℕ := 9

-- Define the number of books Oliver has left to put away
def books_left : ℕ := shelves_needed * books_per_shelf

-- Define the number of books the librarian took
def books_taken : ℕ := total_books - books_left

-- State the theorem
theorem librarian_took_books : books_taken = 10 := by
  sorry

end librarian_took_books_l357_357954


namespace find_fraction_l357_357504

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
variable (h : a + b + c = 1)

theorem find_fraction :
  (a^3 + b^3 + c^3) / (a * b * c) = (1 + 3 * (a - b)^2) / (a * b * (1 - a - b)) :=
by
  sorry

end find_fraction_l357_357504


namespace solve_for_I_l357_357471

-- Definition of the problem's conditions
def unique_digits (a b c d e f g h i j : ℕ) : Prop :=
  list.nodup [a, b, c, d, e, f, g, h, i, j]

def valid_addition (S I X T W E L : ℕ) : Prop :=
  S = 8 ∧ I % 2 = 0 ∧ unique_digits S I X T W E L 0 1 2 3 4 5 6 7 8 9 ∧
  S * 2 + X * 2 = T * 10 + W * 10 + E + L
  
-- The main theorem
theorem solve_for_I (I : ℕ) : 
  (∀ S I X T W E L : ℕ, valid_addition S I X T W E L → I = 2) :=
sorry

end solve_for_I_l357_357471


namespace driver_A_legally_drive_in_5_hours_driver_B_rate_of_decrease_driver_B_range_of_decrease_l357_357726

section DrinkingAndDriving

variables (p1 p2 : ℝ) (t : ℕ)

/-- The blood alcohol content decreases by 30% per hour for driver A,
    show that A needs at least 5 hours to legally drive. -/
theorem driver_A_legally_drive_in_5_hours 
  (h : 1 * (1 - 0.3)^t < 0.2) : t = 5 :=
sorry

/-- If driver B is still under the influence after 5 hours, 
    his rate of decrease p2 is less than 30%. -/
theorem driver_B_rate_of_decrease
  (h1 : t = 5)
  (h2 : 1 * (1 - p2)^5 ≥ 0.2) : p2 < 0.3 :=
sorry

/-- Estimating the range of the rate of decrease for driver B who needs at least 7 hours to legally drive,
    given that after 6 hours, BAC is >= 0.2 and after 7 hours, BAC is < 0.2.
    The range of 0.21 < p2 <= 0.24 -/
theorem driver_B_range_of_decrease 
  (h1 : 1 * (1 - p2)^6 ≥ 0.2)
  (h2 : 1 * (1 - p2)^7 < 0.2) : 0.21 < p2 ∧ p2 ≤ 0.24 :=
sorry

end DrinkingAndDriving

end driver_A_legally_drive_in_5_hours_driver_B_rate_of_decrease_driver_B_range_of_decrease_l357_357726


namespace true_propositions_count_l357_357195

theorem true_propositions_count :
  let propositions := [true, false, true, true, true] in
  list.count id propositions = 4 :=
by
  sorry

end true_propositions_count_l357_357195


namespace units_digit_prime_count_l357_357658

def is_prime_units_digit (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ p % 10 = n

def digits := {1, 2, 3, 4, 5}

theorem units_digit_prime_count :
  (digits.filter is_prime_units_digit).card = 4 := by
  sorry

end units_digit_prime_count_l357_357658


namespace cube_root_sum_of_polynomial_roots_l357_357572

theorem cube_root_sum_of_polynomial_roots (r1 r2 r3 : ℝ)
  (h1 : r1^3 - 3 * r1^2 + 1 = 0)
  (h2 : r2^3 - 3 * r2^2 + 1 = 0)
  (h3 : r3^3 - 3 * r3^2 + 1 = 0)
  (sum_roots : r1 + r2 + r3 = 3) :
  (Real.cbrt (3 * r1 - 2) + Real.cbrt (3 * r2 - 2) + Real.cbrt (3 * r3 - 2) = 0) :=
by sorry

end cube_root_sum_of_polynomial_roots_l357_357572


namespace find_net_monthly_salary_l357_357258

-- Define Jill's conditions and problem
def discretionary_income (S : ℝ) := S / 5

def vacation_fund (S : ℝ) := 0.30 * discretionary_income S
def savings (S : ℝ) := 0.20 * discretionary_income S
def socializing (S : ℝ) := 0.35 * discretionary_income S
def fitness_classes (S : ℝ) := 0.05 * discretionary_income S
def gifts_and_charity (S : ℝ) := discretionary_income S - (vacation_fund S + savings S + socializing S + fitness_classes S)

axiom h_gifts_and_charity : ∀ (S : ℝ), gifts_and_charity S = 99

-- The main theorem to prove
theorem find_net_monthly_salary : ∃ (S : ℝ), 99 / 0.10 = S / 5 := by
  sorry

end find_net_monthly_salary_l357_357258


namespace no_partition_with_square_product_partition_with_square_sum_l357_357901

-- Define the set A
def A : set ℕ := {a | ∃ k, k ≤ 2014 ∧ a = 3^k}

-- Statement for part (a)
theorem no_partition_with_square_product :
  ¬ ∃ (P : finset (finset ℕ)), (∀ (S ∈ P), S.nonempty ∧ S ⊆ A) ∧ (∀ (S ∈ P), (S.prod id).sqrt ∈ ℕ) :=
sorry

-- Statement for part (b)
theorem partition_with_square_sum :
  ∃ (P : finset (finset ℕ)), (∀ (S ∈ P), S.nonempty ∧ S ⊆ A) ∧ (∀ (S ∈ P), (S.sum id).sqrt ∈ ℕ) :=
sorry

end no_partition_with_square_product_partition_with_square_sum_l357_357901


namespace part1_part2_l357_357122

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357122


namespace distribution_ways_l357_357723

theorem distribution_ways (tickets people : ℕ) (ht : tickets = 5) (hp : people = 4) :
  ∃! (n : ℕ), n = 96 ∧
  ∀ (distribution : vector (list ℕ) people),
    (∀ i, 1 ≤ list.length (distribution.nth i) ∧ list.length (distribution.nth i) ≤ 2) ∧
    (∀ i, list.length (distribution.nth i) = 2 → 
      ∃ j, j + 1 < tickets ∧ set.to_finset {j, j+1} ⊆ list.to_finset (distribution.nth i)) ∧
    (set.to_finset (distribution.join) = finset.range (succ tickets)) = n := sorry

end distribution_ways_l357_357723


namespace find_general_formula_and_d_l357_357083

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357083


namespace other_number_of_given_conditions_l357_357985

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l357_357985


namespace log_sum_l357_357356

theorem log_sum :
  log 10 50 + log 10 20 = 3 :=
by
  sorry

end log_sum_l357_357356


namespace min_value_of_f_l357_357231

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l357_357231


namespace triangle_BDG_is_isosceles_l357_357218

-- Define the conditions as given in the problem
-- Let's assume we have defined the geometric setup and segments accordingly
def squares_positioned_as_shown : Prop := sorry -- Definition of two squares positioned as per given setup
def marked_segments_equal : Prop := sorry -- Definition that the marked segments are equal

-- Triangle BDG and the task to prove it is isosceles
theorem triangle_BDG_is_isosceles
  (h1 : squares_positioned_as_shown)
  (h2 : marked_segments_equal) :
  ∃ (B D G: Point), is_triangle B D G ∧ is_isosceles_triangle B D G :=
sorry

end triangle_BDG_is_isosceles_l357_357218


namespace find_die_number_l357_357043

theorem find_die_number (sides : Finset ℕ) (h_sides : sides = {1, 2, 3, 4, 5, 6})
  (prob : ℚ) (h_prob : prob = (1/3)) : 
  ∃ n ∈ sides, (sides.filter (λ x, x > n)).card = 2 := 
by
  sorry

end find_die_number_l357_357043


namespace part1_part2_l357_357118

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357118


namespace pencil_problem_l357_357994

theorem pencil_problem
  (x y : ℕ)
  (h1 : 27 * x + 23 * y ≤ 940)
  (h2 : y - x ≤ 10)
  (h3 : x ≤ y)
  (h4 : ∀ a b : ℕ, 27 * a + 23 * b ≤ 940 → b - a ≤ 10 → a ≤ b → x + y ≥ a + b → a ≥ x) :
  x = 14 ∧ y = 24 :=
begin
  sorry
end

end pencil_problem_l357_357994


namespace machine_purchase_price_l357_357190

theorem machine_purchase_price (P : ℝ) (h : 0.80 * P = 6400) : P = 8000 :=
by
  sorry

end machine_purchase_price_l357_357190


namespace no_valid_pair_for_tangential_quadrilateral_l357_357366

theorem no_valid_pair_for_tangential_quadrilateral (a d : ℝ) (h : d > 0) :
  ¬((∃ a d, a + (a + 2 * d) = (a + d) + (a + 3 * d))) :=
by
  sorry

end no_valid_pair_for_tangential_quadrilateral_l357_357366


namespace other_number_of_given_conditions_l357_357983

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l357_357983


namespace time_to_cover_one_mile_with_semicircles_l357_357675

theorem time_to_cover_one_mile_with_semicircles 
  (one_mile : ℝ := 5280) 
  (width_in_feet : ℝ := 50) 
  (radius : ℝ := width_in_feet / 2 ) 
  (num_of_semicircles : ℕ := (one_mile / width_in_feet).ceil.to_nat) 
  (distance_per_semicircle : ℝ := radius * π) 
  (total_distance_covered : ℝ := num_of_semicircles * distance_per_semicircle) 
  (distance_in_miles : ℝ := total_distance_covered / one_mile) 
  (riding_speed_mph : ℝ := 4) 
  (time_without_breaks : ℝ := distance_in_miles / riding_speed_mph) 
  (num_of_breaks : ℕ := 1)
  (break_time_in_hours : ℝ := 5/60 * num_of_breaks)
  (total_time : ℝ := time_without_breaks + break_time_in_hours) :
  total_time = (3 * π + 2) / 24 := 
sorry

end time_to_cover_one_mile_with_semicircles_l357_357675


namespace valid_arrangements_l357_357812

def is_valid_arrangement (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  ({a, b, c, d} = {1, 2, 3, 4}) ∧ 
  (
    (a = 1 ∧ b ≠ 1 ∧ c ≠ 2 ∧ d ≠ 4) ∨
    (a ≠ 1 ∧ b = 1 ∧ c ≠ 2 ∧ d ≠ 4) ∨
    (a ≠ 1 ∧ b ≠ 1 ∧ c = 2 ∧ d ≠ 4) ∨
    (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 2 ∧ d = 4)
  )

theorem valid_arrangements :
  { (a, b, c, d) | is_valid_arrangement a b c d } = 
  { (2, 1, 4, 3), (2, 1, 3, 4), (3, 1, 4, 2), (3, 1, 2, 4), (3, 2, 1, 4), (4, 1, 3, 2) } :=
by
  sorry

end valid_arrangements_l357_357812


namespace find_w_l357_357176

noncomputable def roots_cubic_eq (x : ℝ) : ℝ := x^3 + 2 * x^2 + 5 * x - 8

def p : ℝ := sorry -- one root of x^3 + 2x^2 + 5x - 8 = 0
def q : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0
def r : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0

theorem find_w 
  (h1 : roots_cubic_eq p = 0)
  (h2 : roots_cubic_eq q = 0)
  (h3 : roots_cubic_eq r = 0)
  (h4 : p + q + r = -2): 
  ∃ w : ℝ, w = 18 := 
sorry

end find_w_l357_357176


namespace integral_solution_l357_357311

noncomputable def integral_problem : Real :=
  ∫ x in 0..3, (exp (sqrt ((3 - x) / (3 + x))) / ((3 + x) * sqrt (9 - x^2)))

theorem integral_solution : integral_problem = (exp 1 - 1) / 3 :=
by
  sorry

end integral_solution_l357_357311


namespace total_rabbits_and_chickens_l357_357209

theorem total_rabbits_and_chickens (r c : ℕ) (h₁ : r = 64) (h₂ : r = c + 17) : r + c = 111 :=
by {
  sorry
}

end total_rabbits_and_chickens_l357_357209


namespace part1_part2_l357_357097

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357097


namespace area_of_triangle_AEF_l357_357775

open Real

def regular_triangular_prism :=
  { base_edge_length : ℝ,
    lateral_edge_length : ℝ }

variables (D A B C E F : ℝ → ℝ) (prism : regular_triangular_prism)

axiom base_edge_length_is_one : prism.base_edge_length = 1
axiom lateral_edge_length_is_two : prism.lateral_edge_length = 2
axiom plane_intersects_BD_at_E : plane_passes_through A ∧ intersects BD E
axiom plane_intersects_CD_at_F : plane_passes_through A ∧ intersects CD F
axiom perimeter_minimized : minimized_perimeter A E F

theorem area_of_triangle_AEF : area_of_triangle A E F = 3 * sqrt 55 / 64 := sorry

end area_of_triangle_AEF_l357_357775


namespace tony_graduate_degree_years_l357_357590

-- Define the years spent for each degree and the total time
def D1 := 4 -- years for the first degree in science
def D2 := 4 -- years for each of the two additional degrees
def T := 14 -- total years spent in school
def G := 2 -- years spent for the graduate degree in physics

-- Theorem: Given the conditions, prove that Tony spent 2 years on his graduate degree in physics
theorem tony_graduate_degree_years : 
  D1 + 2 * D2 + G = T :=
by
  sorry

end tony_graduate_degree_years_l357_357590


namespace slope_angle_of_line_eq_x_plus_1_l357_357998

theorem slope_angle_of_line_eq_x_plus_1 : 
  (∃ θ : ℝ, 0 ≤ θ ∧ θ < 180 ∧ tan θ = 1 ∧ θ = 45) :=
sorry

end slope_angle_of_line_eq_x_plus_1_l357_357998


namespace bill_equality_minutes_l357_357595

theorem bill_equality_minutes :
  ∃ (m : ℕ), (7 + 0.25 * m = 12 + 0.20 * m) ∧ m = 100 :=
begin
  use 100,
  split,
  { norm_num, },
  { norm_num, }
end

end bill_equality_minutes_l357_357595


namespace range_of_m_l357_357805

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem range_of_m (m : ℝ) (h : ∀ x > 0, f x > m * x) : m ≤ 2 := sorry

end range_of_m_l357_357805


namespace quadratic_function_value_l357_357290

theorem quadratic_function_value (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a - b + c = 9) :
  a + 3 * b + c = 1 := 
by 
  sorry

end quadratic_function_value_l357_357290


namespace only_101_l357_357910

def bijective (f : ℤ → ℤ) : Prop :=
  function.bijective f

def is_bijective (n : ℕ) (g : (ℤ / n) → (ℤ / n)) : Prop :=
  (bijective g) ∧ 
  ∀ k : ℕ, k < 101 → bijective (λ x : (ℤ / n), g x + x * k)

theorem only_101 {n : ℕ} :
  (∃ g : (ℤ / n) → (ℤ / n), is_bijective n g) ↔ (n = 101) :=
sorry

end only_101_l357_357910


namespace polygon_sides_l357_357028

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 120) :
  ∃ (n : ℕ), n = 6 :=
begin
  use 6,
  sorry  -- Proof goes here
end

end polygon_sides_l357_357028


namespace polar_eq_C₁_area_triangle_PAB_l357_357890

-- Definitions of the parametric equations for curve C₁
def C₁_x (t : ℝ) : ℝ := t + 8 / t
def C₁_y (t : ℝ) : ℝ := t - 8 / t

-- Definition of the polar coordinate equation for curve C₂
def C₂ (θ : ℝ) : ℝ := 2 * sqrt 3 * cos θ

-- Definition of the ray l in polar coordinates
def ray_l (θ : ℝ) : Prop := θ = π / 6

-- Point P coordinates
def P : (ℝ × ℝ) := (4, 0)

-- Theorem to prove the polar coordinate equation of curve C₁
theorem polar_eq_C₁ (ρ θ : ℝ) : (∃ t : ℝ, C₁_x t = ρ * cos θ ∧ C₁_y t = ρ * sin θ) → ρ^2 * cos (2 * θ) = 32 :=
by
  sorry

-- Theorem to prove the area of triangle PAB
theorem area_triangle_PAB (ρ_A ρ_B : ℝ) (θ : ℝ) 
  (h1 : ray_l θ) 
  (h2 : ρ_A^2 * cos (2 * θ) = 32) 
  (h3 : ρ_B = C₂ θ) 
  : 1 / 2 * (ρ_A - ρ_B) * 4 * sin(π / 6) = 5 :=
by
  sorry

end polar_eq_C₁_area_triangle_PAB_l357_357890


namespace domain_of_f_l357_357559

-- Define the function
def f (x : ℝ) : ℝ := (sqrt (1 - 3 ^ x) + 1 / x ^ 2)

-- Prove the domain of the function
theorem domain_of_f :
  { x : ℝ | ∃ y, y = f x } = Iio 0 :=
begin
  sorry
end

end domain_of_f_l357_357559


namespace protractor_angle_approximation_l357_357435

noncomputable def protractor_angle (arc_length : ℝ) (radius : ℝ) : ℝ :=
  (arc_length / radius) * (180 / Real.pi)

theorem protractor_angle_approximation :
  protractor_angle 10 5 ≈ 115 :=
by {
  sorry
}

end protractor_angle_approximation_l357_357435


namespace total_goats_l357_357598

theorem total_goats (W: ℕ) (H_W: W = 180) (H_P: W + 70 = 250) : W + (W + 70) = 430 :=
by
  -- proof goes here
  sorry

end total_goats_l357_357598


namespace suff_not_necess_cond_perpendicular_l357_357392

theorem suff_not_necess_cond_perpendicular (m : ℝ) :
  (m = 1 → ∀ x y : ℝ, x - y = 0 ∧ x + y = 0) ∧
  (m ≠ 1 → ∃ (x y : ℝ), ¬ (x - y = 0 ∧ x + y = 0)) :=
sorry

end suff_not_necess_cond_perpendicular_l357_357392


namespace remainder_of_poly_l357_357609

-- Definitions based on conditions given:
def f (x : ℝ) : ℝ := x^4 - 4*x^2 + 7
def a : ℝ := 1

-- Mathematical proof problem statement:
theorem remainder_of_poly (remainder : ℝ) : 
  remainder = f a :=
by 
  have h : f 1 = 4 := by 
    calc f 1 = 1^4 - 4*1^2 + 7 : by rfl
      ... = 1 - 4 + 7 : by rfl
      ... = 4 : by rfl
  exact h

end remainder_of_poly_l357_357609


namespace find_a_l357_357328

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l357_357328


namespace hyperbola_triangle_area_l357_357808

theorem hyperbola_triangle_area {a b : ℝ} (ha : a > 0) (hb : b > 0) 
  (C : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (eccentricity : 2 * a = 2 * a)
  (perimeter : ∃ F₁ F₂ A : ℝ, F₁ + F₂ + 4 * a = 10 * a) :
  ∃ area : ℝ, area = Real.sqrt(15) * a^2 :=
by
  sorry

end hyperbola_triangle_area_l357_357808


namespace sphere_surface_area_l357_357530

theorem sphere_surface_area (R h : ℝ) (R_pos : 0 < R) (h_pos : 0 < h) :
  ∃ A : ℝ, A = 2 * Real.pi * R * h := 
sorry

end sphere_surface_area_l357_357530


namespace handshake_problem_l357_357172

theorem handshake_problem :
  ∃ (a b : ℕ), a + b = 20 ∧ (a * (a - 1) / 2) + ((20 - a) * (19 - a) / 2) = 106 ∧ a * b = 84 :=
by
  use 14, 6
  repeat
    split
  · exact rfl
  · norm_num
  · exact rfl

end handshake_problem_l357_357172


namespace box_surface_area_l357_357644

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end box_surface_area_l357_357644


namespace num_subsets_containing_6_l357_357843

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l357_357843


namespace max_value_of_f_in_interval_l357_357745

theorem max_value_of_f_in_interval :
  let f : ℝ → ℝ := λ x, Real.exp x - x
  in f 1 = (Real.exp 1) - 1 ∧ ∀ x ∈ set.Icc (-1:ℝ) 1, f x ≤ f 1 :=
by
  let f : ℝ → ℝ := λ x, Real.exp x - x
  have h_endpoint_left : f (-1) = (1 / Real.exp 1) + 1 := sorry  -- Calculated value at -1
  have h_endpoint_right : f 1 = Real.exp 1 - 1 := sorry  -- Calculated value at 1
  have h_max_value : f 1 = e - 1 := sorry  -- Verifying that f(1) equals max value e-1
  have h_decreasing : ∀ x ∈ set.Icc (-1:ℝ) 0, f x ≤ f 1 := sorry  -- Function is decreasing on [-1,0]
  have h_increasing : ∀ x ∈ set.Icc 0 1, f x ≤ f 1 := sorry  -- Function is increasing on [0,1]
  exact ⟨h_max_value, h_endpoint_left, h_endpoint_right, h_decreasing, h_increasing⟩

end max_value_of_f_in_interval_l357_357745


namespace domain_transform_l357_357271

variable (f : ℝ → ℝ)

theorem domain_transform (hf : ∀ x, -1 ≤ x ∧ x ≤ 3 → (x^2 - 1) ∈ set.Icc (-1 : ℝ) 3) :
    set.Icc (-(3 / 2) : ℝ) (1 / 2) ⊆ {x | 2 * x + 1 ∈ set.Icc (-1 : ℝ) 3} :=
sorry

end domain_transform_l357_357271


namespace main_theorem_l357_357492

noncomputable section

open Classical

variables {n : ℕ}

def A := vector (fin 2) n

def zeroSeq : A := vector.repeat 0 n

def addSeq (a b : A) : A :=
⟨(vector.zip_with (λ x y, if x = y then 0 else 1) a.to_list b.to_list), sorry⟩

def preservesHammingDistance (f : A → A) : Prop :=
∀ a b, (vector.to_list (f a)).zip (vector.to_list (f b)).countp (λ x, x.1 ≠ x.2) =
       (vector.to_list a).zip (vector.to_list b).countp (λ x, x.1 ≠ x.2)

variables (f : A → A)
variables (a b c : A)
variables (h₁ : f zeroSeq = zeroSeq)
variables (h₂ : preservesHammingDistance f)
variables (h₃ : addSeq (addSeq a b) c = zeroSeq)

theorem main_theorem : addSeq (addSeq (f a) (f b)) (f c) = zeroSeq := 
sorry

end main_theorem_l357_357492


namespace slope_product_proof_l357_357286

noncomputable def midpoint_slope_product (M P1 P2 P : Point) (k1 k2 : ℝ) : Prop :=
  let line_eq := ∀ x, P.y = k1 * (P.x + 2)
  let ellipse_eq := P1.x^2 + 2 * P1.y^2 = 4 ∧ P2.x^2 + 2 * P2.y^2 = 4 ∧ M.x = -2 ∧ M.y = 0
  let intersect := line_eq P1 ∧ line_eq P2
  let midpoint := P.x = (P1.x + P2.x) / 2 ∧ P.y = (P1.y + P2.y) / 2
  let nonzero_slope := k1 ≠ 0
  let slope_OP := k2 = P.y / P.x
  let product := k1 * k2 = - 1 / 2
  in ellipse_eq ∧ intersect ∧ midpoint ∧ nonzero_slope ∧ slope_OP → product

theorem slope_product_proof (M P1 P2 P : Point) (k1 k2 : ℝ)
  (h : midpoint_slope_product M P1 P2 P k1 k2) :
  k1 * k2 = - 1 / 2 :=
sorry

end slope_product_proof_l357_357286


namespace cost_of_downloading_360_songs_in_2005_is_144_dollars_l357_357447

theorem cost_of_downloading_360_songs_in_2005_is_144_dollars :
  (∀ (c_2004 c_2005 : ℕ), (∀ c : ℕ, c_2005 = c ∧ c_2004 = c + 32) →
  200 * c_2004 = 360 * c_2005 → 360 * c_2005 / 100 = 144) :=
  by sorry

end cost_of_downloading_360_songs_in_2005_is_144_dollars_l357_357447


namespace closest_multiple_of_12_l357_357242

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the closest multiple of 4 to 2050 (2048 and 2052)
def closest_multiple_of_4 (n m : ℕ) : ℕ :=
if n % 4 < m % 4 then n - (n % 4)
else m + (4 - (m % 4))

-- Define the conditions for being divisible by both 3 and 4
def is_multiple_of_12 (n : ℕ) : Prop := is_multiple_of n 12

-- Theorem statement
theorem closest_multiple_of_12 (n m : ℕ) (h : n = 2050) (hm : m = 2052) :
  is_multiple_of_12 m :=
sorry

end closest_multiple_of_12_l357_357242


namespace probability_same_club_l357_357295

-- Definition for the problem conditions
variable (A B : ℕ) (clubs : Finset ℕ) (n : ℕ)
variable (join_club : ℕ → Finset ℕ → Prop)
variable (prob : Finset ℕ → ℝ)

-- Given conditions
-- A school has 8 clubs
def school_clubs := 8

-- Students join clubs uniformly
axiom join_club_uniformly (x : ℕ) (clubs : Finset ℕ) : join_club x clubs ↔ (x ∈ clubs)

-- Probability of joining a specific club is 1/8
axiom prob_join_specific_club (x : ℕ) (clubs : Finset ℕ) : prob clubs = 1 / school_clubs

-- The problem to be proved
theorem probability_same_club :
  ∀ (clubs : Finset ℕ),
    clubs.card = school_clubs →
    (prob {c | join_club A {c} ∧ join_club B {c}}) = 1 / school_clubs :=
by
  -- Skipping proof, just stating the theorem
  sorry

end probability_same_club_l357_357295


namespace inconsistent_statements_l357_357069

-- Define constants
variables (K Y N B : ℕ)

-- Define conditions based on the statements
def krosh_statement : Prop := K + Y + N = 120
def yozhik_statement : Prop := N + B = 103
def nyusha_statement : Prop := K + Y + B = 152

-- The main statement we want to prove
theorem inconsistent_statements (h1 : krosh_statement K Y N) 
                               (h2 : yozhik_statement N B)
                               (h3 : nyusha_statement K Y B) : 
  False := 
by 
  -- Summing equations and simplifying
  have sum_eq : 2 * (K + Y + N + B) = 375, 
  {
    calc
      2 * (K + Y + N + B) = (K + Y + N) + (N + B) + (K + Y + B) : by linarith [h1, h2, h3]
                         ... = 120 + 103 + 152                    : by linarith [h1, h2, h3]
                         ... = 375                                : by norm_num
  },
  -- Contradiction since 375 is odd
  have h_odd : ¬(375 % 2 = 0) := by norm_num,
  have h_even : 2 * (K + Y + N + B) % 2 = 0 := by norm_num,
  exact h_odd (by norm_num : 375 % 2 = 1)

-- Proof is omitted
sorry

end inconsistent_statements_l357_357069


namespace quadrilateral_cyclic_l357_357943

noncomputable def midpoint (a b : Point) : Point := sorry
noncomputable def parallel (l1 l2 : Line) : Prop := sorry
noncomputable def cyclic_quadrilateral (a b c d : Point) : Prop := sorry

structure Triangle (A B C : Point) := 
  (AB_eq_AC : dist A B = dist A C)

variables {A B C M P X Y : Point} (T : Triangle A B C)
  (mid_M : M = midpoint B C)
  (P_condition : dist P B < dist P C)
  (parallel_condition : parallel ⟨A, P⟩ ⟨B, C⟩)
  (on_segments : B ∈ [X, P] ∧ C ∈ [Y, P])
  (angle_condition : angle P X M = angle P Y M)
 
theorem quadrilateral_cyclic : cyclic_quadrilateral A P X Y := sorry -- The desired conclusion

end quadrilateral_cyclic_l357_357943


namespace part1_part2_l357_357123

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357123


namespace nina_bracelets_sold_l357_357152

theorem nina_bracelets_sold :
  let p_n := 25
  let p_b := 15
  let p_e := 10
  let p_j := 45
  let n_n := 5
  let n_e := 20
  let n_j := 2
  let R_t := 565
  let revenue := n_n * p_n + n_e * p_e + n_j * p_j
  n_b * p_b = R_t - revenue
  ⊢ n_b = 10 := 
sorry

end nina_bracelets_sold_l357_357152


namespace smallest_root_of_quadratic_l357_357240

theorem smallest_root_of_quadratic :
  ∃ x : ℝ, (12 * x^2 - 50 * x + 48 = 0) ∧ x = 1.333 := 
sorry

end smallest_root_of_quadratic_l357_357240


namespace not_possible_coloring_possible_coloring_l357_357600

-- Problem (a): For n = 2001 and k = 4001, prove that such coloring is not possible.
theorem not_possible_coloring (n : ℕ) (k : ℕ) (h_n : n = 2001) (h_k : k = 4001) :
  ¬ ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

-- Problem (b): For n = 2^m - 1 and k = 2^(m+1) - 1, prove that such coloring is possible.
theorem possible_coloring (m : ℕ) (n k : ℕ) (h_n : n = 2^m - 1) (h_k : k = 2^(m+1) - 1) :
  ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

end not_possible_coloring_possible_coloring_l357_357600


namespace projection_ratio_l357_357914

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

noncomputable def projection (u v : ℝ) : ℝ :=
(⟪u, v⟫ / ⟪v, v⟫) • v

theorem projection_ratio 
  (v w : ℝ) (hv : v ≠ 0) (hw : w ≠ 0) 
  (h : ∥projection v w∥ / ∥v∥ = 3/4) : 
  (∥projection (projection v w) v∥ / ∥v∥ = 9/16) :=
by sorry

end projection_ratio_l357_357914


namespace f_2005_eq_cos_l357_357928

noncomputable def fn : ℕ → (ℝ → ℝ)
| 0     := λ x, Real.sin x
| (n+1) := λ x, (fn n)' x

theorem f_2005_eq_cos : ∀ x : ℝ, fn 2005 x = Real.cos x :=
by intros; sorry

end f_2005_eq_cos_l357_357928


namespace find_general_formula_and_d_l357_357084

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357084


namespace fish_minimum_catch_l357_357733

theorem fish_minimum_catch (a1 a2 a3 a4 a5 : ℕ) (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
  (h_non_increasing : a1 ≥ a2 ∧ a2 ≥ a3 ∧ a3 ≥ a4 ∧ a4 ≥ a5) : 
  a1 + a3 + a5 ≥ 50 :=
sorry

end fish_minimum_catch_l357_357733


namespace incorrect_statement_l357_357780

variable P : Prop
variable Q : Prop

-- Conditions
def P_def : P := ∅ ∈ {∅}
def Q_def : Q := ∅ ⊆ {∅}

-- Problem Statement
theorem incorrect_statement : ¬(P ∧ Q) = false :=
by
  have p_true := P_def
  have q_true := Q_def
  sorry

end incorrect_statement_l357_357780


namespace f_1_f_2023_l357_357611

noncomputable def f : ℤ → ℝ := sorry

axiom f_periodictiy (n : ℤ) (hn : n > 0) : f(n + 3) = (f(n) - 1) / (f(n) + 1)

axiom f_initial_cond1 : f 1 ≠ 0
axiom f_initial_cond2 : f 1 ≠ 1
axiom f_initial_cond3 : f 1 ≠ -1

theorem f_1_f_2023 : f 1 * f 2023 = -1 := 
by 
  -- Proof goes here
  sorry

end f_1_f_2023_l357_357611


namespace triangle_perimeter_l357_357961

/-- Let \( ABC \) be an isosceles triangle with \( AB = BC \). 
  Let \( BD \) be the median of the triangle. 
  Consider a circle with radius 4 that passes through points \( A \), \( B \), and \( D \).
  The circle intersects side \( BC \) at point \( E \) such that \( BE : BC = 7 : 8 \).
  Prove that the perimeter of triangle \( ABC \) is 20. -/
theorem triangle_perimeter 
  (A B C D E : Point)
  (h_iso : AB = BC)
  (h_median : is_median B D A C)
  (h_circle : circle_passing_through_radius B A D 4)
  (h_ratio : BE / BC = 7 / 8) :
  perimeter A B C = 20 := by
  sorry

end triangle_perimeter_l357_357961


namespace ratio_students_l357_357150

theorem ratio_students
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (h_finley : finley_students = 24)
  (h_johnson : johnson_students = 22)
  : (johnson_students : ℚ) / ((finley_students / 2 : ℕ) : ℚ) = 11 / 6 :=
by
  sorry

end ratio_students_l357_357150


namespace smallest_positive_period_f_range_of_f_zero_point_of_f_l357_357795

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 3) * cos x - sqrt 3 * cos x ^ 2 + sqrt 3 / 4

theorem smallest_positive_period_f :
  (∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ¬(∀ x : ℝ, f (x + ε) = f x))) :=
sorry

theorem range_of_f (a b : ℝ) (h : -π / 4 ≤ a ∧ b ≤ π / 4) :
  (∀ y, (∃ x, a ≤ x ∧ x ≤ b ∧ f x = y) ↔ y ∈ set.Icc (-1 : ℝ) (1 / 2 : ℝ)) :=
sorry

theorem zero_point_of_f :
  (∃ x, -π / 4 ≤ x ∧ x ≤ π / 4 ∧ f x = 0) :=
sorry

end smallest_positive_period_f_range_of_f_zero_point_of_f_l357_357795


namespace find_certain_number_l357_357019

theorem find_certain_number (n x : ℤ) (h1 : 9 - n / x = 7 + 8 / x) (h2 : x = 6) : n = 8 := by
  sorry

end find_certain_number_l357_357019


namespace perpendicular_bisector_midpoint_l357_357157

-- Define the geometric entities and their properties
structure Trapezoid (A B C D : Type) :=
(smaller_base : A = B ∨ A = D)
(perpendicular_bisector_AB : ∃ E, E bisects A B)

-- Define the geometric squares constructed outside
structure Square (A D : Type) := 
(square_outside : ∀ F G, ¬ A = F ∧ ¬ D = G)

-- The proof problem
theorem perpendicular_bisector_midpoint {A B C D E F G H : Type} 
  (t : Trapezoid A B C D)
  (s1 : Square A D)
  (s2 : Square B C)
  (midpoint_FH : ∃ M, M midpoint F H) :
  ∃ N, N bisects AB ∧ N passes_through midpoints FH := sorry

end perpendicular_bisector_midpoint_l357_357157


namespace alice_number_possibility_l357_357301

noncomputable def B := nat.prime

-- Defining the conditions from point a)
variables (A B : ℕ) (h_prime : nat.prime B) (h2 : A ≠ 1) (h3 : 100 * B + A = k ^ 2) 

-- Theorem statement that answers the question using conditions from a)
theorem alice_number_possibility
  (h_prime : nat.prime B)
  (h2 : A ≠ 1)
  (h3 : 100 * B + A = k ^ 2) : A = 24 ∨ A = 61 :=
sorry

end alice_number_possibility_l357_357301


namespace driving_scenario_l357_357618

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end driving_scenario_l357_357618


namespace sample_capacity_is_480_l357_357454

-- Problem conditions
def total_people : ℕ := 500 + 400 + 300
def selection_probability : ℝ := 0.4

-- Statement: Prove that sample capacity n equals 480
theorem sample_capacity_is_480 (n : ℕ) (h : n / total_people = selection_probability) : n = 480 := by
  sorry

end sample_capacity_is_480_l357_357454


namespace volume_of_soil_extracted_l357_357665

noncomputable def pond_base_length : ℝ := 28
noncomputable def pond_base_width : ℝ := 10
noncomputable def depth_start : ℝ := 5
noncomputable def depth_end : ℝ := 8

def linear_depth (x : ℝ) : ℝ :=
  (depth_end - depth_start) / pond_base_length * x + depth_start

def volume_integral : ℝ := ∫ x in 0..pond_base_length, pond_base_width * linear_depth x

theorem volume_of_soil_extracted :
  volume_integral = 1820 := by
  sorry

end volume_of_soil_extracted_l357_357665


namespace other_number_of_given_conditions_l357_357982

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l357_357982


namespace verify_a_l357_357331

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l357_357331


namespace song_distribution_l357_357303

-- Let us define the necessary conditions and the result as a Lean statement.

theorem song_distribution :
    ∃ (AB BC CA A B C N : Finset ℕ),
    -- Six different songs.
    (AB ∪ BC ∪ CA ∪ A ∪ B ∪ C ∪ N) = {1, 2, 3, 4, 5, 6} ∧
    -- No song is liked by all three.
    (∀ song, ¬(song ∈ AB ∩ BC ∩ CA)) ∧
    -- Each girl dislikes at least one song.
    (N ≠ ∅) ∧
    -- For each pair of girls, at least one song liked by those two but disliked by the third.
    (AB ≠ ∅ ∧ BC ≠ ∅ ∧ CA ≠ ∅) ∧
    -- The total number of ways this can be done is 735.
    True := sorry

end song_distribution_l357_357303


namespace partition_sum_of_squares_l357_357511

theorem partition_sum_of_squares (k : ℕ) (hk : k > 0) :
  ∃ (A B : Finset ℤ), (A ∪ B = (Finset.range (2*k+1)).map (Function.Embedding.subtype (λ n, 2*k^2 + k + n ∈ Finset.range (2*k + 1)))) ∧
                      (A ∩ B = ∅) ∧
                      (A.sum (λ x, x^2) = B.sum (λ x, x^2)) :=
sorry

end partition_sum_of_squares_l357_357511


namespace max_value_S_l357_357854

theorem max_value_S (a b : ℝ) (h : 3 * a^2 + 5 * |b| = 7) :
  let S := 2 * a^2 - 3 * |b|
  in S ≤ (14 / 3) :=
sorry

end max_value_S_l357_357854


namespace solve_fraction_equation_l357_357999

theorem solve_fraction_equation :
  ∀ x : ℝ, x ≠ 0 → x ≠ -5 → (1 / (3 * x) = 2 / (x + 5) ↔ x = 1) :=
by {
  intros x hx1 hx2,
  split,
  { intro h,
    have h1 : (1 : ℝ) / (3 * x) = 1 / 3 * (1 / x), from one_div_mul_one_div hx1,
    rw [h1, ← mul_eq_mul_right_iff] at h,
    obtain ⟨hemm, h2⟩ := mul_eq_zero h,
    cases hemm,
    { field_simp at hemm, 
      linarith },
    { field_simp at h2, 
      linarith } },
  { intro hx,
    rw hx,
    field_simp,
    ring_nf }
  sorry }

end solve_fraction_equation_l357_357999


namespace problem_solution_l357_357013

noncomputable def phi (d : ℕ) : ℝ := (d + Real.sqrt (d^2 + 4)) / 2

theorem problem_solution :
  let d := 2009,
      φd := phi d,
      (a, b, c) := (2009, 2009^2 + 4, 2)
  in
  φd = (a + Real.sqrt b) / c ∧ Nat.gcd a c = 1 ∧ a + b + c = 4038096 :=
by
  sorry

end problem_solution_l357_357013


namespace graph_passes_through_fixed_point_l357_357565

theorem graph_passes_through_fixed_point : ∀ (a : ℝ) (x y : ℝ), (y = 3 + log a (2 * x + 3)) → (x = -1) → (y = 3) :=
by
  intros a x y h hx
  rw [hx] at h
  simp at h
  exact h

end graph_passes_through_fixed_point_l357_357565


namespace skylar_berries_l357_357173

variables {S K : ℕ}

def stacy_berries (S : ℕ) : ℕ := 3 * S + 2

theorem skylar_berries : (∃ S K, stacy_berries S = 32 ∧ 2 * S = K) → K = 20 :=
by
  rintro ⟨S, K, h_stacy, h_steve⟩
  rw [stacy_berries, eq_comm] at h_stacy
  have h_S : S = 10 := by
    linarith
  rw h_S at h_steve
  linarith

end skylar_berries_l357_357173


namespace part1_part2_part3_l357_357801

theorem part1 (ω : ℝ) (h1 : ω > 0) :
  (∀ x ∈ Icc (-π / 4) (3 * π / 4), deriv (λ x, 4 * sin (ω * x / 2) * cos (ω * x / 2) + 1) x ≥ 0) ↔ 
  (0 < ω ∧ ω ≤ 2 / 3) := sorry

theorem part2 {a b : ℝ} (ω : ℝ) (h1 : ω < 4)
  (hg : ∀ x, g x = 2 * sin (ω * x + π * ω / 3) + 1)
  (h2 : g π / 6 = 1)
  (h3 : ∃ a b, a < b ∧ ∃ k : ℕ, k ≥ 30 ∧ ∀ x, a ≤ x ∧ x ≤ b → g x = 0) :
  b - a = 43 * π / 3 := sorry

theorem part3 (ω : ℝ) (h1 : ω < 4) (m : ℝ)
  (hg : ∀ x, g x = 2 * sin (ω * x + π * ω / 3) + 1)
  (h3 : ∀ x ∈ Icc (-π / 6) (π / 12), g^2 x - m * g x - 1 ≤ 0) :
  m ∈ Icc (8 / 3) (⊤) := sorry

end part1_part2_part3_l357_357801


namespace sum_of_two_cubes_lt_1000_l357_357011

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357011


namespace suitable_sampling_method_l357_357652

noncomputable def is_stratified_sampling_suitable (mountainous hilly flat low_lying sample_size : ℕ) (yield_dependent_on_land_type : Bool) : Bool :=
  if yield_dependent_on_land_type && mountainous + hilly + flat + low_lying > 0 then true else false

theorem suitable_sampling_method :
  is_stratified_sampling_suitable 8000 12000 24000 4000 480 true = true :=
by
  sorry

end suitable_sampling_method_l357_357652


namespace average_ratio_one_l357_357676

theorem average_ratio_one (scores : List ℝ) (h_len : scores.length = 50) :
  let A := (scores.sum / 50)
  let scores_with_averages := scores ++ [A, A]
  let A' := (scores_with_averages.sum / 52)
  A' = A :=
by
  sorry

end average_ratio_one_l357_357676


namespace inverse_sum_l357_357929

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_sum :
  (∃ x₁, g x₁ = -2 ∧ x₁ ≠ 5) ∨ (∃ x₂, g x₂ = 0 ∧ x₂ = 3) ∨ (∃ x₃, g x₃ = 4 ∧ x₃ = -1) → 
  g⁻¹ (-2) + g⁻¹ (0) + g⁻¹ (4) = 6 :=
by
  sorry

end inverse_sum_l357_357929


namespace double_sum_identity_l357_357313

theorem double_sum_identity :
  ∑ i in Finset.range 50, ∑ j in Finset.range 50, (2 * (i + 1) + 2 * (j + 1) + 3) = 262500 :=
by
  sorry

end double_sum_identity_l357_357313


namespace chris_money_l357_357318

-- Define conditions
def grandmother_gift : Nat := 25
def aunt_uncle_gift : Nat := 20
def parents_gift : Nat := 75
def total_after_birthday : Nat := 279

-- Define the proof problem to show Chris had $159 before his birthday
theorem chris_money (x : Nat) (h : x + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_birthday) :
  x = 159 :=
by
  -- Leave the proof blank
  sorry

end chris_money_l357_357318


namespace ordered_pairs_divide_square_sum_l357_357358

theorem ordered_pairs_divide_square_sum :
  { (m, n) : ℕ × ℕ | m > 0 ∧ n > 0 ∧ (mn - 1) ∣ (m^2 + n^2) } = { (1, 2), (1, 3), (2, 1), (3, 1) } := 
sorry

end ordered_pairs_divide_square_sum_l357_357358


namespace rowing_campers_l357_357965

theorem rowing_campers (total_campers campers_afternoon : ℕ) (h_total : total_campers = 62) (h_afternoon : campers_afternoon = 27) :
  total_campers - campers_afternoon = 35 :=
by {
  rw [h_total, h_afternoon],
  norm_num,
}

end rowing_campers_l357_357965


namespace max_value_of_f_l357_357368

open Real

noncomputable def f (x : ℝ) := 1 - 2*x - 3/x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ f x = 1 - 2*sqrt 6 ∧ (∀ y > 0, f y ≤ f x) :=
begin
  sorry
end

end max_value_of_f_l357_357368


namespace other_number_eq_462_l357_357989

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l357_357989


namespace triangle_shortest_side_l357_357892

theorem triangle_shortest_side (A B C F E : Type) (h1 : ∠ BAC = 90)
  (AB : ℝ) (r : ℝ) (radius : r = 5)
  (AF FB : ℝ) (segment1 : AF = 7) (segment2 : FB = 9)
  (region1 : AB = AF + FB)
  (s t : ℝ) (radius_formula : r = s / t):
  AB = 16 → 
  shortest_side ABC = 15 :=
by
  sorry

end triangle_shortest_side_l357_357892


namespace tangent_line_parallel_or_bisects_l357_357383

variables {O : Type*} [metric_space O] [normed_group O]
variables (circle : set O) (l : set O)
variables (A B T_A T_B : O)
variable (r : ℝ)

-- Given conditions
def is_circle (center : O) (radius : ℝ) (circle : set O) : Prop :=
∀ (P : O), P ∈ circle ↔ dist P center = radius

def is_tangent (P Q : O) (circle : set O) : Prop :=
∃ (X : O), X ∈ circle ∧ dist P X = dist Q X

def are_tangent_segments_equal (A B T_A T_B : O) (circle : set O) : Prop :=
is_tangent A T_A circle ∧ is_tangent B T_B circle ∧ dist A T_A = dist B T_B

-- Proof goal
theorem tangent_line_parallel_or_bisects (center : O) (radius : ℝ) (circle : set O) (l : set O)
  (h_circle : is_circle center radius circle)
  (h_tangents : are_tangent_segments_equal A B T_A T_B circle) :
  ∃ (M : O), dist A M = dist M B ∨ ∃ (k : ℝ), ∀ (T : O), T ∈ l ↔ T + k = l :=
sorry

end tangent_line_parallel_or_bisects_l357_357383


namespace range_PA_PB_l357_357394

theorem range_PA_PB (x y : ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  (x^2 / 4 + y^2 = 1) →
  (x^2 + (y - 1)^2 = 1) →
  (∃ a b : ℝ, (a - b)^2 + (a - 1)^2 = 1 → ∃ x y : ℝ, x^2 / 4 + y^2 = 1) →
  ∀ P ∈ Ellipse ∧ AB ∈ Circle, (-1 ≤ PA.dot PB) ∧ (PA.dot PB ≤ 13 / 3) :=
by
  sorry

end range_PA_PB_l357_357394


namespace log_sum_equiv_l357_357354

theorem log_sum_equiv : log 50 / log 10 + log 20 / log 10 = 3 := by
  have h1 : log 50 / log 10 + log 20 / log 10 = log 1000 / log 10 := by
    rw [←log_mul 50 20, mul_comm]
  have h2 : log 1000 / log 10 = 3 := by
    rw [log_pow 10 3, log_nat_sum_eq_one]
  rw [h1, h2]
  exact rfl

end log_sum_equiv_l357_357354


namespace radius_for_visibility_l357_357653

noncomputable def calculate_radius (r : ℝ) : Prop :=
  let side_length := 3 in
  let probability := 1/2 in
  let target_r := 6 * Real.sqrt 2 - Real.sqrt 2 in
  (prob_visible_two_sides_from_circle r side_length = probability) → r = target_r

theorem radius_for_visibility :
  calculate_radius (6 * Real.sqrt 2 - Real.sqrt 2) :=
sorry

end radius_for_visibility_l357_357653


namespace find_s_l357_357924

noncomputable theory

open polynomial

variables (s : ℝ)
variables (f g : polynomial ℝ)
variables (h1 : monic f) (h2 : monic g)
variables (h3 : ∃ c, f = (X - (s + 2)) * (X - (s + 8)) * (X - c))
variables (h4 : ∃ d, g = (X - (s + 4)) * (X - (s + 10)) * (X - d))
variables (h5 : ∀ x, f.eval x - g.eval x = s + 1)

theorem find_s : s = 111 :=
sorry

end find_s_l357_357924


namespace systematic_sampling_of_students_l357_357455

theorem systematic_sampling_of_students
  (class_size : ℕ)
  (selected_students : Finset ℕ)
  (h_class_size : class_size = 50)
  (h_selected : selected_students = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50}) :
  (sampling_method : String) 
  (h_sampling_method : sampling_method = "Systematic") :=
sorry

end systematic_sampling_of_students_l357_357455


namespace more_digits_in_base2_l357_357245

theorem more_digits_in_base2 (b: ℕ) (c: ℕ) (h₁: b = 500) (h₂: c = 2500)
: nat.log2 c - nat.log2 b = 3 := by
  have h3: nat.log2 500 = 8 := sorry
  have h4: nat.log2 2500 = 11 := sorry
  rw [h₂, h₁, h4, h3]
  norm_num
  sorry

end more_digits_in_base2_l357_357245


namespace range_of_a_for_quadratic_inequality_l357_357186

theorem range_of_a_for_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Icc (-2 : ℝ) 2 := by
sorry

end range_of_a_for_quadratic_inequality_l357_357186


namespace cos_evaluation_l357_357886

open Real

noncomputable def a (n : ℕ) : ℝ := sorry  -- since it's an arithmetic sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, a n + a k = 2 * a ((n + k) / 2)

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 6 + a 9 = 3 * a 6 ∧ a 6 = π / 4

theorem cos_evaluation :
  is_arithmetic_sequence a →
  satisfies_condition a →
  cos (a 2 + a 10 + π / 4) = - (sqrt 2 / 2) :=
by
  intros
  sorry

end cos_evaluation_l357_357886


namespace find_F_16_l357_357938

noncomputable def F : ℝ → ℝ := sorry

lemma F_condition_1 : ∀ x, (x + 4) ≠ 0 ∧ (x + 2) ≠ 0 → (F (4 * x) / F (x + 4) = 16 - (64 * x + 64) / (x^2 + 6 * x + 8)) := sorry

lemma F_condition_2 : F 8 = 33 := sorry

theorem find_F_16 : F 16 = 136 :=
by
  have h1 := F_condition_1
  have h2 := F_condition_2
  sorry

end find_F_16_l357_357938


namespace rachel_left_24_brownies_at_home_l357_357164

-- Defining the conditions
def total_brownies : ℕ := 40
def brownies_brought_to_school : ℕ := 16

-- Formulation of the theorem
theorem rachel_left_24_brownies_at_home : (total_brownies - brownies_brought_to_school = 24) :=
by
  sorry

end rachel_left_24_brownies_at_home_l357_357164


namespace father_ate_8_brownies_l357_357149

noncomputable def brownies_initial := 24
noncomputable def brownies_mooney_ate := 4
noncomputable def brownies_after_mooney := brownies_initial - brownies_mooney_ate
noncomputable def brownies_mother_made_next_day := 24
noncomputable def brownies_total_expected := brownies_after_mooney + brownies_mother_made_next_day
noncomputable def brownies_actual_on_counter := 36

theorem father_ate_8_brownies :
  brownies_total_expected - brownies_actual_on_counter = 8 :=
by
  sorry

end father_ate_8_brownies_l357_357149


namespace polygon_number_of_sides_l357_357025

-- Define the given conditions
def each_interior_angle (n : ℕ) : ℕ := 120

-- Define the property to calculate the number of sides
def num_sides (each_exterior_angle : ℕ) : ℕ := 360 / each_exterior_angle

-- Statement of the problem
theorem polygon_number_of_sides : num_sides (180 - each_interior_angle 6) = 6 :=
by
  -- Proof is omitted
  sorry

end polygon_number_of_sides_l357_357025


namespace problem_solution_l357_357702

theorem problem_solution :
  -20 + 7 * (8 - 2 / 2) = 29 :=
by 
  sorry

end problem_solution_l357_357702


namespace prove_optionC_is_suitable_l357_357615

def OptionA := "Understanding the height of students in Class 7(1)"
def OptionB := "Companies recruiting and interviewing job applicants"
def OptionC := "Investigating the impact resistance of a batch of cars"
def OptionD := "Selecting the fastest runner in our school to participate in the city-wide competition"

def is_suitable_for_sampling_survey (option : String) : Prop :=
  option = OptionC

theorem prove_optionC_is_suitable :
  is_suitable_for_sampling_survey OptionC :=
by
  sorry

end prove_optionC_is_suitable_l357_357615


namespace arrangement_count_l357_357243

theorem arrangement_count (n : ℕ) (c1_not_first c2_not_last : ℕ) (hn : n = 5) (hc1 : c1_not_first ≠ 1) (hc2 : c2_not_last ≠ n) : 
  count_arrangements n c1_not_first c2_not_last = 78 :=
sorry

end arrangement_count_l357_357243


namespace log_product_eq_one_sixth_log_y_x_l357_357717

variable (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

theorem log_product_eq_one_sixth_log_y_x :
  (Real.log x ^ 2 / Real.log (y ^ 5)) * 
  (Real.log (y ^ 3) / Real.log (x ^ 4)) *
  (Real.log (x ^ 4) / Real.log (y ^ 3)) *
  (Real.log (y ^ 5) / Real.log (x ^ 3)) *
  (Real.log (x ^ 3) / Real.log (y ^ 4)) = 
  (1 / 6) * (Real.log x / Real.log y) := 
sorry

end log_product_eq_one_sixth_log_y_x_l357_357717


namespace circumcenter_rational_l357_357544

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l357_357544


namespace expected_sufferers_l357_357156

theorem expected_sufferers 
  (fraction_condition : ℚ := 1 / 4)
  (sample_size : ℕ := 400) 
  (expected_number : ℕ := 100) : 
  fraction_condition * sample_size = expected_number := 
by 
  sorry

end expected_sufferers_l357_357156


namespace find_other_number_l357_357978

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l357_357978


namespace volume_of_inscribed_sphere_l357_357672

theorem volume_of_inscribed_sphere (a : ℝ) (h : a = 10) : 
  let r := a / 2 in
  (4 / 3) * real.pi * r^3 = (500 / 3) * real.pi :=
by 
  have hr : r = 5 := by rw [h, rfl, div_eq_mul_inv, mul_comm]; exact (div_self zero_lt_two).symm
  sorry

end volume_of_inscribed_sphere_l357_357672


namespace necessary_but_not_sufficient_l357_357637

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 1 ∨ x > 4) → (x^2 - 3 * x + 2 > 0) ∧ ¬((x^2 - 3 * x + 2 > 0) → (x < 1 ∨ x > 4)) :=
by
  sorry

end necessary_but_not_sufficient_l357_357637


namespace intersection_M_N_l357_357134

def M : Set ℝ := { x : ℝ | -3 < x ∧ x < 1 }
def N : Set ℤ := { x : ℤ | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_M_N :
  M ∩ (λ x : ℤ, x : ℝ) = { -1, 0 } :=
by
  sorry

end intersection_M_N_l357_357134


namespace find_k_l357_357030

theorem find_k (k : ℝ) :
  (∃ x : ℝ, 8 * x - k = 2 * (x + 1) ∧ 2 * (2 * x - 3) = 1 - 3 * x) → k = 4 :=
by
  sorry

end find_k_l357_357030


namespace solve_equation_l357_357171

theorem solve_equation :
  let roots := {x : ℝ | (x - 2) * (x + 1) * (x + 4) * (x + 7) = 19} in
  roots = {-5 / 2 + real.sqrt 85 / 2, -5 / 2 - real.sqrt 85 / 2, -5 / 2 + real.sqrt 5 / 2, -5 / 2 - real.sqrt 5 / 2} :=
by
  let roots := {x : ℝ | (x - 2) * (x + 1) * (x + 4) * (x + 7) = 19}
  have x12 := {-5 / 2 + real.sqrt 85 / 2, -5 / 2 - real.sqrt 85 / 2}
  have x34 := {-5 / 2 + real.sqrt 5 / 2, -5 / 2 - real.sqrt 5 / 2}
  simp [roots, x12, x34]
  sorry

end solve_equation_l357_357171


namespace positive_m_for_one_solution_l357_357373

theorem positive_m_for_one_solution :
  ∀ (m : ℝ), (∃ x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ (discriminant 9 m 36 = 0) → m = 36 :=
by
  sorry

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

end positive_m_for_one_solution_l357_357373


namespace locus_of_P_on_radical_axis_l357_357902

variable {C1 C2 : Type} [MetricSpace C1] [MetricSpace C2] -- Assuming C1 and C2 are metric spaces which model the circles

-- Definitions of the circles and their properties
variable (E F : C1) -- intersection points of C1 and C2
variable (P : C2) -- midpoint of chord AB of C2
variable (AB : C2) -- variable chord of C2 with midpoint P
variable (T : C1) -- intersection point of circle on AB as diameter with C1

-- Property that PT is tangent to C1 at T
def PT_tangent_to_C1 (O1 : C1) (r1 : ℝ) (P T : C2) : Prop :=
  is_tangent P T C1

-- The theorem stating that P lies on the line segment EF
theorem locus_of_P_on_radical_axis 
  (C1_intersects_C2_at_EF : intersects_at C1 C2 E F)
  (P_mid_AB : is_midpoint P AB)
  (circle_diam_AB_intersects_T : intersects_circle P AB T C1)
  (PT_tangent : PT_tangent_to_C1 C1 T):
  lies_on_segment P E F := sorry

end locus_of_P_on_radical_axis_l357_357902


namespace area_of_triangle_AI1I2_correct_l357_357503

noncomputable def area_of_triangle_AI1I2 (AB BC AC : ℝ) (h1 : AB = 26) (h2 : BC = 28) (h3 : AC = 30) : ℝ :=
  let alpha := 90 * (Real.pi / 180)
  let angle_AXB := (alpha / 2)
  let sin_half_A := Real.sin (angle_AXB / 2)
  have angle_I1AI2 : ℝ := angle_AXB
  let tot_area := (1 / 2) * 26 * 30 * sin_half_A
  tot_area

theorem area_of_triangle_AI1I2_correct : 
  area_of_triangle_AI1I2 26 28 30 26 28 30 = 390 * Real.sqrt 2 :=
by
  simp only [area_of_triangle_AI1I2]
  norm_num
  sorry

end area_of_triangle_AI1I2_correct_l357_357503


namespace find_number_l357_357990

variable (X : ℝ)

def mean_1 (a b c d e : ℝ) : ℝ := (a + b + c + d + e) / 5
def mean_2 (a b c d e : ℝ) : ℝ := (a + b + c + d + e) / 5

theorem find_number :
  (mean_1 28 X 70 88 104 = 67) ∧ (mean_2 50 62 97 124 X = 75.6) → X = 45 :=
by
  sorry

end find_number_l357_357990


namespace parabola_equation_l357_357560

theorem parabola_equation :
  ∃ a : ℝ, ∀ x : ℝ, ∀ y : ℝ,
  (
    (y = a * (x - 1) * (x - 4)) ∧
    (a * 1 * (-3) = 0) ∧
    (a * 16 * (-2) = 0) ∧
    ((a * (x^2 - 5 * x + 4) = 2 * x) → 
    (((9 * a^2 + 20 * a + 4 = 0) → ((a = -2 / 9 ∨ a = -2))))
  ) := sorry

end parabola_equation_l357_357560


namespace isosceles_triangle_area_l357_357878

theorem isosceles_triangle_area
  (P Q R S : Type)
  (h_isosceles : PQ = PR)
  (h_altitude : PS ⊥ QR)
  (h_bisect : QS = SR)
  (h_PQ : PQ = 13)
  (h_PR : PR = 13)
  (h_QR : QR = 10) :
  let PS := (13^2 - 5^2).sqrt in
  let area := 1/2 * QR * PS in
  area = 60 :=
by sorry

end isosceles_triangle_area_l357_357878


namespace copy_machines_total_copies_l357_357283

theorem copy_machines_total_copies (rate1 rate2 time : ℕ) (h_rate1 : rate1 = 35) (h_rate2 : rate2 = 75) (h_time : time = 30) :
  rate1 * time + rate2 * time = 3300 :=
by
  rw [h_rate1, h_rate2, h_time]
  norm_num
  sorry

end copy_machines_total_copies_l357_357283


namespace rectangle_area_l357_357667

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w^2 + (3*w)^2 = d^2) : (3 * w ^ 2 = 3 * d ^ 2 / 10) :=
by
  sorry

end rectangle_area_l357_357667


namespace ladder_length_theorem_l357_357261

noncomputable def length_of_ladder (a : ℝ) (theta : ℝ) (h : ℝ) : Prop :=
  cos theta = a / h

/-- The length of the ladder is 9.2 meters given the angle of elevation and distance from the wall. -/
theorem ladder_length_theorem : length_of_ladder 4.6 (real.pi / 3) 9.2 :=
  sorry

end ladder_length_theorem_l357_357261


namespace subsets_containing_six_l357_357838

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l357_357838


namespace max_value_ab_ac_bc_l357_357919

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end max_value_ab_ac_bc_l357_357919


namespace travel_time_reduction_l357_357977

theorem travel_time_reduction
  (original_speed : ℝ)
  (new_speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (new_time : ℝ)
  (h1 : original_speed = 80)
  (h2 : new_speed = 50)
  (h3 : time = 3)
  (h4 : distance = original_speed * time)
  (h5 : new_time = distance / new_speed) :
  new_time = 4.8 := 
sorry

end travel_time_reduction_l357_357977


namespace minimum_people_group_l357_357041

theorem minimum_people_group
    (A B C D : Set)
    (fA : |A| = 13)
    (fB : |B| = 9)
    (fC : |C| = 15)
    (fD : |D| = 6)
    (cond1 : ∀ x ∈ B, x ∈ A ∨ x ∈ C)
    (cond2 : ∀ x ∈ C, x ∈ B ∨ x ∈ D) :
    ∃ (min_people : ℕ), min_people = 22 := by
  sorry

end minimum_people_group_l357_357041


namespace min_abs_value_l357_357784

noncomputable def omega : ℂ := complex.exp (2 * complex.pi * complex.I / 3)

theorem min_abs_value (a b c : ℤ) (h : a * b * c = 60) (h_omega : omega ^ 3 = 1) (h_omega_ne : omega ≠ 1) :
  (abs (a + b * omega + c * omega^2) = complex.sqrt 3) :=
sorry

end min_abs_value_l357_357784


namespace part1_part2_l357_357110

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357110


namespace weight_of_3_moles_of_BaF2_is_correct_l357_357607

-- Definitions for the conditions
def atomic_weight_Ba : ℝ := 137.33 -- g/mol
def atomic_weight_F : ℝ := 19.00 -- g/mol

-- Definition of the molecular weight of BaF2
def molecular_weight_BaF2 : ℝ := (1 * atomic_weight_Ba) + (2 * atomic_weight_F)

-- The statement to prove
theorem weight_of_3_moles_of_BaF2_is_correct : (3 * molecular_weight_BaF2) = 525.99 :=
by
  -- Proof omitted
  sorry

end weight_of_3_moles_of_BaF2_is_correct_l357_357607


namespace R_is_product_of_linear_polynomials_l357_357904

open Polynomial

noncomputable def P (x : ℂ) : Polynomial ℂ := sorry

def Q (x y : ℂ) : Polynomial ℂ := P(x) - P(y)

def is_linear_factor (f : Polynomial ℂ) : Prop :=
  ∃ (a b : ℂ), f = a * X - b * C

def unproportional (f g : Polynomial ℂ) : Prop :=
  ¬ ∃ (c : ℂ), c ≠ 0 ∧ f = c * g

def k_linear_factors (Q : Polynomial ℂ) (k : ℕ) : Prop :=
  ∃ (factors : List (Polynomial ℂ)), (∀ f ∈ factors, is_linear_factor f) ∧
    (∀ f g ∈ factors, f ≠ g → unproportional f g) ∧
    factors.length = k

variable (k : ℕ)
variable (R : Polynomial ℂ)

-- Given conditions
axiom P_nonconstant : ¬(P = 0)
axiom Q_k_linear_factors : k_linear_factors (Q x y) k
axiom R_degree_smaller_k : R.natDegree < k

-- Prove that R(x,y) is a product of linear polynomials
theorem R_is_product_of_linear_polynomials :
  ∃ (factors : List (Polynomial ℂ)), (∀ f ∈ factors, is_linear_factor f) ∧
    (R = factors.prod) :=
sorry

end R_is_product_of_linear_polynomials_l357_357904


namespace remaining_minutes_proof_l357_357300

def total_series_minutes : ℕ := 360

def first_session_end : ℕ := 17 * 60 + 44  -- in minutes
def first_session_start : ℕ := 15 * 60 + 20  -- in minutes
def second_session_end : ℕ := 20 * 60 + 40  -- in minutes
def second_session_start : ℕ := 19 * 60 + 15  -- in minutes
def third_session_end : ℕ := 22 * 60 + 30  -- in minutes
def third_session_start : ℕ := 21 * 60 + 35  -- in minutes

def first_session_duration : ℕ := first_session_end - first_session_start
def second_session_duration : ℕ := second_session_end - second_session_start
def third_session_duration : ℕ := third_session_end - third_session_start

def total_watched : ℕ := first_session_duration + second_session_duration + third_session_duration

def remaining_time : ℕ := total_series_minutes - total_watched

theorem remaining_minutes_proof : remaining_time = 76 := 
by 
  sorry  -- Proof goes here

end remaining_minutes_proof_l357_357300


namespace max_value_ab_ac_bc_l357_357920

open Real

theorem max_value_ab_ac_bc {a b c : ℝ} (h : a + 3 * b + c = 6) : 
  ab + ac + bc ≤ 4 :=
sorry

end max_value_ab_ac_bc_l357_357920


namespace functional_equation_l357_357735

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
begin
  sorry
end

end functional_equation_l357_357735


namespace area_of_isosceles_trapezoid_l357_357305

variable (b c : ℝ)

theorem area_of_isosceles_trapezoid (ABCD : Type) [IsIsoscelesTrapezoid ABCD] 
  (B C : Point ABCD) (O : Point ABCD)
  (h_inscribed : InscribedCircle O ABCD)
  (h_OB_eq_b : distance O B = b)
  (h_OC_eq_c : distance O C = c) :
  area ABCD = 2 * b * c :=
sorry

end area_of_isosceles_trapezoid_l357_357305


namespace max_y_coord_p_l357_357307

noncomputable def maximum_y_coordinate (p : ℝ × ℝ) (a b : ℝ) : Prop :=
  let e := (p.1^2) / a^2 + (p.2^2) / b^2 = 1
  let f := (p.1 - 1)^2 + p.2^2 = 3
  let intersections : Set (ℝ × ℝ) := { q | q.2 ≥ 0 }
  let y_max := max {q.2 | q ∈ intersections}
  -- Condition: p lies on the ellipse
  e ∧ f → y_max = 3 / 4

theorem max_y_coord_p :
  maximum_y_coordinate P 2 (sqrt 3) = true :=
sorry

end max_y_coord_p_l357_357307


namespace magnitude_of_T_l357_357498

noncomputable def complex_T : ℂ := (2 + Complex.i)^20 - (2 - Complex.i)^20

theorem magnitude_of_T : Complex.abs complex_T = 19531250 := by
  sorry

end magnitude_of_T_l357_357498


namespace second_number_multiple_of_seven_l357_357976

theorem second_number_multiple_of_seven (x : ℕ) (h : gcd (gcd 105 x) 2436 = 7) : 7 ∣ x :=
sorry

end second_number_multiple_of_seven_l357_357976


namespace new_average_of_transformed_numbers_l357_357179

-- Definitions to set up the problem
def average_of_ten_numbers_eq_seven (nums : List ℝ) : Prop :=
  nums.length = 10 ∧ List.sum nums / 10 = 7

def new_average (nums : List ℝ) : ℝ :=
  List.sum (List.map (λ x => 12 * x^2) nums) / 10

-- Theorem statement proving the required result
theorem new_average_of_transformed_numbers (nums : List ℝ) (h : average_of_ten_numbers_eq_seven nums) : 
  new_average nums = 588 := 
sorry

end new_average_of_transformed_numbers_l357_357179


namespace actual_distance_traveled_l357_357860

theorem actual_distance_traveled (D T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 20 * T) : D = 20 :=
by
  sorry

end actual_distance_traveled_l357_357860


namespace part_a_part_b_part_c_l357_357632

noncomputable def Q (x : ℂ) (n : ℕ) := (x + 1)^n + x^n + 1

def P (x : ℂ) := x^2 + x + 1

theorem part_a (n : ℕ) : (P(x) ∣ Q(x, n)) ↔ ∃ k : ℕ, n = 6*k + 1 ∨ n = 6*k + 5 :=
by
  sorry

theorem part_b (n : ℕ) : (P(x)^2 ∣ Q(x, n)) ↔ ∃ k : ℕ, n = 6*k + 4 :=
by
  sorry

theorem part_c (n : ℕ) : ¬ (P(x)^3 ∣ Q(x, n)) := 
by
  sorry

end part_a_part_b_part_c_l357_357632


namespace bags_sold_on_monday_l357_357068

-- Defining the stock, sold bags on each day, and unsold percentage
def total_stock : ℕ := 600
def sold_tuesday : ℕ := 70
def sold_wednesday : ℕ := 100
def sold_thursday : ℕ := 110
def sold_friday : ℕ := 145
def unsold_percentage : ℚ := 25 / 100

-- Calculating number of unsold bags
def unsold_bags : ℕ := (unsold_percentage * total_stock).toNat

-- Total bags sold from Tuesday to Friday
def total_sold_tue_to_fri : ℕ := sold_tuesday + sold_wednesday + sold_thursday + sold_friday

-- Proving the number of bags sold on Monday
theorem bags_sold_on_monday : 
  total_stock - (unsold_bags + total_sold_tue_to_fri) = 25 := 
by 
  -- adding this so the statement builds successfully
  sorry

end bags_sold_on_monday_l357_357068


namespace circumcenter_is_rational_l357_357555

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l357_357555


namespace largest_sum_pairs_l357_357187

theorem largest_sum_pairs (a b c d : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a < b) (h₇ : b < c) (h₈ : c < d)
(h₉ : a + b = 9 ∨ a + b = 10) (h₁₀ : b + c = 9 ∨ b + c = 10)
(h₁₁ : b + d = 12) (h₁₂ : c + d = 13) :
d = 8 ∨ d = 7.5 :=
sorry

end largest_sum_pairs_l357_357187


namespace frac_abs_div_a_plus_one_l357_357853

theorem frac_abs_div_a_plus_one (a : ℝ) (h : a ≠ 0) : abs a / a + 1 = 0 ∨ abs a / a + 1 = 2 :=
by sorry

end frac_abs_div_a_plus_one_l357_357853


namespace measure_of_angle_A_l357_357483

-- Define the conditions
def angle_B : ℝ := 15
def angle_C : ℝ := 3 * angle_B
def angle_sum_in_triangle : ℝ := 180

-- Theorem statement
theorem measure_of_angle_A (h1 : angle_C = 3 * angle_B) (h2 : angle_B = 15) (h3 : ∀ A B C : ℝ, A + B + C = angle_sum_in_triangle) :
    ∃ angle_A : ℝ, angle_A = 120 :=
by
  use 120
  sorry

end measure_of_angle_A_l357_357483


namespace find_t_l357_357213

-- Define the vertices of the triangle
def A := (0 : ℝ, 10 : ℝ)
def B := (3 : ℝ, 0 : ℝ)
def C := (9 : ℝ, 0 : ℝ)

-- Defining intersection points function on lines AB and AC with a horizontal line y = t.
def intersect_AB (t : ℝ) : ℝ × ℝ := (((3 : ℝ) - (3 / 10) * t), t)
def intersect_AC (t : ℝ) : ℝ × ℝ := (((9 : ℝ) - (9 / 10) * t), t)

-- Area of triangle ATU formed by intersections at T and U.
noncomputable def area_triangle (t : ℝ) : ℝ :=
  (1 / 2) * ((6 : ℝ) - (3 / 5) * t) * ((10 : ℝ) - t)

-- Statement: Prove that if area of triangle ATU is 15, then t = 5
theorem find_t (t : ℝ) (h : area_triangle t = 15) : t = 5 :=
  sorry

end find_t_l357_357213


namespace symmetric_point_x_axis_l357_357540

def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ := (M.1, -M.2)

theorem symmetric_point_x_axis :
  ∀ (M : ℝ × ℝ), M = (3, -4) → symmetric_point M = (3, 4) :=
by
  intros M h
  rw [h]
  dsimp [symmetric_point]
  congr
  sorry

end symmetric_point_x_axis_l357_357540


namespace find_f_28_l357_357535

theorem find_f_28 (f : ℕ → ℚ) (h1 : ∀ n : ℕ, f (n + 1) = (3 * f n + n) / 3) (h2 : f 1 = 1) :
  f 28 = 127 := by
sorry

end find_f_28_l357_357535


namespace coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l357_357639

def coprime_distinct_remainders (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : Prop :=
  ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
    (∀ (i : Fin m) (j : Fin k), ∀ (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k))

def not_coprime_congruent_product (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : Prop :=
  ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
    ∃ (i : Fin m) (j : Fin k) (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a s * b t) % (m * k)

-- Example statement to assert the existence of the above properties
theorem coprime_mk_has_distinct_products 
  (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : coprime_distinct_remainders m k coprime_mk :=
sorry

theorem not_coprime_mk_has_congruent_products 
  (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : not_coprime_congruent_product m k not_coprime_mk :=
sorry

end coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l357_357639


namespace true_proposition_number_is_2_l357_357408

open Real

noncomputable def proposition1 (a b : ℝ) : Prop :=
  (a < 0 ∧ b < 0) → ¬ ((a + b) / 2 ≥ sqrt (a * b))

noncomputable def proposition2 (x : ℝ) : Prop :=
  x^2 + 1 > x

noncomputable def proposition3 {x : ℝ} (h : x ≠ 0) : Prop :=
  if x > 0 then x + (1 / x) ≥ 2 else x + (1 / x) ≤ -2

theorem true_proposition_number_is_2 :
  (∃ a b : ℝ, proposition1 a b) ∧ (∀ x : ℝ, proposition2 x) ∧ (∀ (x : ℝ) (h : x ≠ 0), proposition3 h) →
  false :=
sorry

end true_proposition_number_is_2_l357_357408


namespace mark_last_shots_l357_357682

theorem mark_last_shots (h1 : 0.60 * 15 = 9) (h2 : 0.65 * 25 = 16.25) : 
  ∀ (successful_shots_first_15 successful_shots_total: ℤ),
  successful_shots_first_15 = 9 ∧ 
  successful_shots_total = 16 → 
  successful_shots_total - successful_shots_first_15 = 7 := by
  sorry

end mark_last_shots_l357_357682


namespace initial_people_count_l357_357755

theorem initial_people_count (x : ℕ) 
  (h1 : (x + 15) % 5 = 0)
  (h2 : (x + 15) / 5 = 12) : 
  x = 45 := 
by
  sorry

end initial_people_count_l357_357755


namespace value_of_a_l357_357323

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l357_357323


namespace part1_part2_l357_357715

-- Definition of the operation '※'
def operation (a b : ℝ) : ℝ := a^2 - b^2

-- Part 1: Proving 2※(-4) = -12
theorem part1 : operation 2 (-4) = -12 := 
by
  sorry

-- Part 2: Proving the solutions to the equation (x + 5)※3 = 0 are x = -8 and x = -2
theorem part2 : (∃ x : ℝ, operation (x + 5) 3 = 0) ↔ (x = -8 ∨ x = -2) := 
by
  sorry

end part1_part2_l357_357715


namespace find_t_l357_357908

def point (t : ℝ) : ℝ × ℝ := (2 * t - 5, 3)

def point_Q (t : ℝ) : ℝ × ℝ := (1, 2 * t + 2)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.fst + Q.fst) / 2, (P.snd + Q.snd) / 2)

def distance_squared (A B : ℝ × ℝ) : ℝ :=
  (A.fst - B.fst) ^ 2 + (A.snd - B.snd) ^ 2

theorem find_t (t : ℝ) :
  let P := point t,
      Q := point_Q t,
      M := midpoint P Q in
  distance_squared P M = t ^ 2 - 3 ↔ t = 3.5 :=
by
  sorry

end find_t_l357_357908


namespace solution_l357_357778

open Nat

variable {a d : ℝ}
variable (a_seq : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def arithmetic_sequence (s : ℕ → ℝ) (a d : ℝ) : Prop :=
  ∀ n, s n = a + n * d

def sum_of_first_n_terms (S : ℕ → ℝ) (s : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a + a + (n - 1) * d) / 2

-- Additional conditions from the given problem
def conditions (S : ℕ → ℝ) : Prop :=
  S 6 < S 7 ∧ S 7 > S 8

-- The theorem statement based on the correct answer
theorem solution (h_seq : arithmetic_sequence a_seq a d) (h_sum : sum_of_first_n_terms S a_seq)
  (h_conditions : conditions S) : ∀ n, n ≥ 8 → a_seq n < 0 := 
sorry

end solution_l357_357778


namespace exists_disjoint_subsets_for_prime_products_l357_357343

theorem exists_disjoint_subsets_for_prime_products :
  ∃ (A : Fin 100 → Set ℕ), (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ S : Set ℕ, Infinite S → (∃ m : ℕ, ∃ (a : Fin 100 → ℕ),
      (∀ i, a i ∈ A i) ∧ (∀ i, ∃ p : Fin m → ℕ, (∀ k, p k ∈ S) ∧ a i = (List.prod (List.ofFn p))))) :=
sorry

end exists_disjoint_subsets_for_prime_products_l357_357343


namespace clock_minutes_to_correct_time_l357_357250

def slow_clock_time_ratio : ℚ := 14 / 15

noncomputable def slow_clock_to_correct_time (slow_clock_time : ℚ) : ℚ :=
  slow_clock_time / slow_clock_time_ratio

theorem clock_minutes_to_correct_time :
  slow_clock_to_correct_time 14 = 15 :=
by
  sorry

end clock_minutes_to_correct_time_l357_357250


namespace inner_cube_properties_l357_357655

theorem inner_cube_properties (s: ℝ) (r: ℝ) (l: ℝ) 
  (h_surface_area: 6 * s^2 = 54)
  (h_sphere_radius: r = s / 2)
  (h_inner_cube_diagonal: l * Real.sqrt 3 = 2 * r) :
  6 * l^2 = 18 ∧ l^3 = 3 * Real.sqrt 3 :=
by
  -- Given conditions
  have h_s: s = 3 := by nlinarith
  have h_r: r = 3 / 2 := by rw [h_sphere_radius, h_s]
  have h_l: l = Real.sqrt 3 := by rw [h_inner_cube_diagonal, h_r]; nlinarith
  -- Concluding the proof with the surface area and volume
  have h_surface_area_inner: 6 * l^2 = 18 := by rw [h_l]; nlinarith
  have h_volume_inner: l^3 = 3 * Real.sqrt 3 := by rw [h_l]; nlinarith
  exact ⟨h_surface_area_inner, h_volume_inner⟩

end inner_cube_properties_l357_357655


namespace Elizabeth_needs_more_cents_l357_357731

theorem Elizabeth_needs_more_cents
  (pencil_cost : ℕ) (elizabeth_money : ℕ) (borrowed_money : ℕ) :
  pencil_cost = 600 → elizabeth_money = 500 → borrowed_money = 53 →
  (pencil_cost - (elizabeth_money + borrowed_money) = 47) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Elizabeth_needs_more_cents_l357_357731


namespace tan_alpha_value_trigonometric_expression_value_l357_357791

-- Definitions for the conditions
def vertex_at_origin (α : ℝ) : Prop := α = 0
def initial_side_non_negative_half_axis (α : ℝ) : Prop := α = 0
def terminal_side_through_P (α : ℝ) (m : ℝ) : Prop := ∃ P : ℝ × ℝ, P = (-8, m)
def cos_alpha (α : ℝ) : Prop := cos α = -4/5

-- Statements to prove
theorem tan_alpha_value (α : ℝ) (m : ℝ) (h1 : vertex_at_origin α) 
  (h2 : initial_side_non_negative_half_axis α) (h3 : terminal_side_through_P α m) 
  (h4 : cos_alpha α) : tan α = 3/4 ∨ tan α = -3/4 :=
  sorry

theorem trigonometric_expression_value (α : ℝ) (m : ℝ) (h1 : vertex_at_origin α) 
  (h2 : initial_side_non_negative_half_axis α) (h3 : terminal_side_through_P α m) 
  (h4 : cos_alpha α) : 
  (2 * cos (3 * π / 2 + α) + cos (-α)) / (sin (5 * π / 2 - α) - cos (π + α)) = 5/4 ∨ 
  (2 * cos (3 * π / 2 + α) + cos (-α)) / (sin (5 * π / 2 - α) - cos (π + α)) = -1/4 :=
  sorry

end tan_alpha_value_trigonometric_expression_value_l357_357791


namespace projection_norm_ratio_l357_357911

variable {V : Type*} [InnerProductSpace ℝ V]

-- Definitions for the projections
def projection (u v : V) : V := (inner u v / inner v v) • v

-- Given conditions
variables (v w : V)
variables (hv : v ≠ 0) (hw : w ≠ 0)
variable (h_proj : ∥projection v w∥ / ∥v∥ = 3 / 4)

-- Statement to be proven
theorem projection_norm_ratio (p : V) (q : V)
  (hp : p = projection v w)
  (hq : q = projection p v) :
  ∥q∥ / ∥v∥ = 9 / 16 := by
  sorry

end projection_norm_ratio_l357_357911


namespace supplier_A_is_better_l357_357693

def delivery_days_A : List ℝ := [10, 9, 10, 10, 11, 11, 9, 11, 10, 10]
def delivery_days_B : List ℝ := [8, 10, 14, 7, 10, 11, 10, 8, 15, 12]

noncomputable def mean (lst : List ℝ) : ℝ :=
  (list.sum lst) / (list.length lst)

noncomputable def variance (lst : List ℝ) : ℝ :=
  let m := mean lst
  (list.sum (list.map (λ x, (x - m) ^ 2) lst)) / (list.length lst)

noncomputable def supplier_comparison : Prop :=
  (mean delivery_days_A < mean delivery_days_B) ∧ (variance delivery_days_A < variance delivery_days_B)

theorem supplier_A_is_better : supplier_comparison :=
sorry

end supplier_A_is_better_l357_357693


namespace shortest_distance_to_circle_correct_l357_357997

noncomputable def shortest_distance_to_circle : ℝ :=
  let line : ℝ × ℝ → Prop := λ p, p.2 = p.1 - 1 in
  let circle : ℝ × ℝ → Prop := λ p, (p.1 + 2)^2 + (p.2 - 1)^2 = 1 in
  let center := (-2, 1) in
  let radius := 1 in
  let distance_from_line := abs((1 : ℝ) * center.1 + (-1) * center.2 - 1) / real.sqrt(1^2 + (-1)^2) in
  distance_from_line - radius

theorem shortest_distance_to_circle_correct : shortest_distance_to_circle = 2 * real.sqrt 2 - 1 := by
  sorry

end shortest_distance_to_circle_correct_l357_357997


namespace inequality_S_l357_357494

def S (n m : ℕ) : ℕ := sorry

theorem inequality_S (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n :=
sorry

end inequality_S_l357_357494


namespace total_time_correct_l357_357649

variable (b n : ℕ)

def total_travel_time (b n : ℕ) : ℚ := (3*b + 4*n + 2*b) / 150

theorem total_time_correct :
  total_travel_time b n = (5 * b + 4 * n) / 150 :=
by sorry

end total_time_correct_l357_357649


namespace part1_general_formula_part2_find_d_l357_357094

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357094


namespace no_possible_assignment_l357_357064

theorem no_possible_assignment :
  ¬(∃ f : ℤ × ℤ → ℕ+, ∀ (A B C : ℤ × ℤ), collinear A B C ↔ ∃ d > 1, d ∣ f A ∧ d ∣ f B ∧ d ∣ f C) :=
sorry

end no_possible_assignment_l357_357064


namespace circle_tangent_line_l357_357054

theorem circle_tangent_line (a : ℝ) : 
  ∃ (a : ℝ), a = 2 ∨ a = -8 := 
by 
  sorry

end circle_tangent_line_l357_357054


namespace smallest_n_divisible_by_45_with_64_divisors_l357_357507

theorem smallest_n_divisible_by_45_with_64_divisors :
  ∃ n : ℕ, (n > 0) ∧ (45 ∣ n) ∧ (factors n).length = 64 ∧ (n / 45 = 3796875) :=
by sorry

end smallest_n_divisible_by_45_with_64_divisors_l357_357507


namespace marble_count_l357_357641

theorem marble_count (B R W : ℕ) (hB : B = 5) (hR : R = 9)
    (hProb : 0.75 = (R + W) / (B + R + W)) :
    B + R + W = 20 := by
  sorry

end marble_count_l357_357641


namespace vector_projection_unique_l357_357244

theorem vector_projection_unique (a : ℝ) (c d : ℝ) (h : c + 3 * d = 0) :
    ∃ p : ℝ × ℝ, (∀ a : ℝ, ∀ (v : ℝ × ℝ) (w : ℝ × ℝ), 
      v = (a, 3 * a - 2) → 
      w = (c, d) → 
      ∃ p : ℝ × ℝ, p = (3 / 5, -1 / 5)) :=
sorry

end vector_projection_unique_l357_357244


namespace count_5n_beginning_with_1_l357_357254

theorem count_5n_beginning_with_1 :
  (∀ x:Nat, (5^2018).digits = 1411 ∧ (5^2018).nth_digit 0 = 3) →
  (∃ d:Nat, d = 607) ↔ (∃ n, (1 ≤ n ∧ n ≤ 2017) ∧ (5^n).nth_digit 0 = 1) := sorry

end count_5n_beginning_with_1_l357_357254


namespace sharon_coffee_spending_l357_357170

noncomputable def coffee_cost : ℕ :=
  let light_roast_cost := (40 * 2 / 20 + if 40 * 2 % 20 = 0 then 0 else 1) * 10
  let medium_roast_cost := (40 * 1 / 25 + if 40 * 1 % 25 = 0 then 0 else 1) * 12
  let decaf_roast_cost := (40 * 1 / 30 + if 40 * 1 % 30 = 0 then 0 else 1) * 8
  in light_roast_cost + medium_roast_cost + decaf_roast_cost

theorem sharon_coffee_spending : coffee_cost = 80 :=
by
  unfold coffee_cost
  simp
  sorry

end sharon_coffee_spending_l357_357170


namespace greatest_possible_subway_takers_l357_357622

/-- In a company with 48 employees, some part-time and some full-time, exactly (1/3) of the part-time
employees and (1/4) of the full-time employees take the subway to work. Prove that the greatest
possible number of employees who take the subway to work is 15. -/
theorem greatest_possible_subway_takers
  (P F : ℕ)
  (h : P + F = 48)
  (h_subway_part : ∀ p, p = P → 0 ≤ p ∧ p ≤ 48)
  (h_subway_full : ∀ f, f = F → 0 ≤ f ∧ f ≤ 48) :
  ∃ y, y = 15 := 
sorry

end greatest_possible_subway_takers_l357_357622


namespace equilateral_triangle_BC_l357_357046

open Real

theorem equilateral_triangle_BC 
  {A B C H : Point}
  (h_triangle : equilateral_triangle A B C)
  (h_len1 : dist A B = 4) 
  (h_len2 : dist A C = 4) 
  (h_len3 : dist B C = 4) 
  (h_point : on_line A C H)
  (h_ratio : dist A H = 2 * (dist H C)) : 
  dist B C = 4 := 
sorry

end equilateral_triangle_BC_l357_357046


namespace percentage_increase_l357_357694

theorem percentage_increase (G P : ℝ) (h1 : G = 15 + (P / 100) * 15) 
                            (h2 : 15 + 2 * G = 51) : P = 20 :=
by 
  sorry

end percentage_increase_l357_357694


namespace round_trip_time_l357_357578

variable (boat_speed standing_water_speed stream_speed distance : ℕ)

theorem round_trip_time (boat_speed := 9) (stream_speed := 6) (distance := 170) : 
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed)) = 68 := by 
  sorry

end round_trip_time_l357_357578


namespace company_donations_l357_357592

theorem company_donations (x : ℕ) (hA : 60000 : ℝ) (hB : 60000 : ℝ) (hAvg : 60000 / x - 60000 / (1.2 * x) = 40) :
  x = 250 ∧ 1.2 * x = 300 :=
by
  sorry

end company_donations_l357_357592


namespace sum_of_coefficients_expansion_l357_357931

noncomputable def integral_value : ℝ := ∫ (x : ℝ) in 0..(π / 2), 4 * sin x

theorem sum_of_coefficients_expansion :
  integral_value = 4 →
  let n := integral_value in
  (∀ x : ℝ, x ≠ 0 → 
  (let expression := (x + (2 / x)) * (x - (2 / x))^n in
  expression.eval 1 = 3)) :=
by 
  sorry

end sum_of_coefficients_expansion_l357_357931


namespace a_n_formula_d_value_l357_357108

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357108


namespace find_general_formula_and_d_l357_357087

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357087


namespace factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l357_357166

-- Problem 1
theorem factorize_polynomial (x y : ℝ) : 
  x^2 - y^2 + 2*x - 2*y = (x - y)*(x + y + 2) := 
sorry

-- Problem 2
theorem triangle_equilateral (a b c : ℝ) (h : a^2 + c^2 - 2*b*(a - b + c) = 0) : 
  a = b ∧ b = c :=
sorry

-- Problem 3
theorem prove_2p_eq_m_plus_n (m n p : ℝ) (h : 1/4*(m - n)^2 = (p - n)*(m - p)) : 
  2*p = m + n :=
sorry

end factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l357_357166


namespace find_eccentricity_l357_357385

-- Define the constants and variables used.
variables (a b c : ℝ) (e : ℝ)
def ellipse := ∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1)
def parabola := ∀ x y : ℝ, (x^2 = 2 * c * y + c^2)

-- Define the given condition in terms of segment lengths in ellipse
def focal_distance : Prop := c = sqrt (a^2 - b^2)
def parabolas_focal_point : Prop := ∀ x y : ℝ, (parabola x y → ellipse x y)

-- Condition on intersection of parabola and ellipse on three points
def three_points_condition : Prop := c = 2 * b

-- The goal is to find the eccentricity of the ellipse given these conditions
theorem find_eccentricity :
  focal_distance a b c →
  parabolas_focal_point a b c →
  three_points_condition b c →
  e = 2 / ℝ.sqrt 5 := by
sorry

end find_eccentricity_l357_357385


namespace additional_men_required_l357_357692

variables (W_r : ℚ) (W : ℚ) (D : ℚ) (M : ℚ) (E : ℚ)

-- Given variables
def initial_work_rate := (2.5 : ℚ) / (50 * 100)
def remaining_work_length := (12.5 : ℚ)
def remaining_days := (200 : ℚ)
def initial_men := (50 : ℚ)
def additional_men_needed := (75 : ℚ)

-- Calculating the additional men required
theorem additional_men_required
  (calc_wr : W_r = initial_work_rate)
  (calc_wr_remain : W = remaining_work_length)
  (calc_days_remain : D = remaining_days)
  (calc_initial_men : M = initial_men)
  (calc_additional_men : M + E = (125 : ℚ)) :
  E = additional_men_needed :=
sorry

end additional_men_required_l357_357692


namespace value_of_a_minus_b_l357_357858

theorem value_of_a_minus_b (a b : ℚ) (h1 : 3015 * a + 3021 * b = 3025) (h2 : 3017 * a + 3023 * b = 3027) : 
  a - b = - (7 / 3) :=
by
  sorry

end value_of_a_minus_b_l357_357858


namespace equilateral_triangle_area_l357_357199

theorem equilateral_triangle_area (p : ℝ) (h : p > 0) : 
  let s := p in
  let area := (sqrt 3 / 4) * (s ^ 2) in
  s * 3 = 3 * p →
  area = (sqrt 3 / 4) * (p ^ 2) :=
by
  intro hs
  have hs_eq : s = p := eq_of_mul_eq_mul_left three_ne_zero hs
  rw hs_eq at area
  rw hs_eq
  exact rfl

end equilateral_triangle_area_l357_357199


namespace find_three_digit_numbers_l357_357742

-- Define the problem conditions
def valid_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ 
  let a := n / 100 in
  let b := (n % 100) / 10 in
  let c := n % 10 in
  let S := a + b + c in
  n = S + 2 * S * S

-- State the problem as a theorem to prove
theorem find_three_digit_numbers :
  ∀ n : ℕ, valid_three_digit_number n → n = 171 ∨ n = 465 ∨ n = 666 := by
  sorry

end find_three_digit_numbers_l357_357742


namespace a_n_formula_d_value_l357_357104

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357104


namespace part1_part2_l357_357096

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357096


namespace min_value_of_eccentricities_l357_357407

theorem min_value_of_eccentricities (a1 b1 a2 b2 c : ℝ) 
  (h1 : 0 < a1) (h2 : 0 < b1) (h3 : 0 < a2) (h4 : 0 < b2) 
  (h5 : 2 * c = real.sqrt ((a1 ^ 2) + (b1 ^ 2))) 
  (h6 : |abs (a1) - abs (a2)| = 0) 
  : ∃ e1 e2 : ℝ, 4 * (e1 ^ 2) + e2 ^ 2 = 9 / 2 := by
  sorry

end min_value_of_eccentricities_l357_357407


namespace concurrency_of_lines_in_parallelogram_l357_357289

theorem concurrency_of_lines_in_parallelogram
  (A B C D L K X Y B1 C1 C' : Point)
  (h1 : parallelogram A B C D)
  (h2 : excircle (triangle A B C) touches_side AB at L)
  (h3 : excircle (triangle A B C) touches_extended_side BC at K)
  (h4 : line D K ∩ line A C = X)
  (h5 : line B X ∩ median C C1 (triangle A B C) = Y)
  (B1_is_median : is_median B B1 (triangle A B C))
  (C'_is_bisector : is_bisector C C' (triangle A B C))
  (L_in_touches_AB : L ∈ touches (excircle (triangle A B C)) AB)
  (K_extends_BC : K ∈ extends (excircle (triangle A B C)) BC) :
  concurrent (line Y L) (median B B1 (triangle A B C)) (bisector C C' (triangle A B C)) :=
sorry

end concurrency_of_lines_in_parallelogram_l357_357289


namespace ac_not_perpendicular_pb_l357_357395

variable (P A B C : Type) 
variables (plane_AB : Type)
variables [IsCircle plane_AB AB] [is_perpendicular : IsPerpendicular P A plane_AB]
variables [is_different_point_C : DiffPoint C A B]

/-- AC is not perpendicular to PB given the conditions. -/
theorem ac_not_perpendicular_pb 
  (h1 : is_perpendicular P A plane_AB)
  (h2 : is_different_point_C C A B) :
  ¬ IsPerpendicular (Line AC) (Line PB) :=
sorry

end ac_not_perpendicular_pb_l357_357395


namespace johns_age_l357_357433

-- Define variables for ages of John and Matt
variables (J M : ℕ)

-- Define the conditions based on the problem statement
def condition1 : Prop := M = 4 * J - 3
def condition2 : Prop := J + M = 52

-- The goal: prove that John is 11 years old
theorem johns_age (J M : ℕ) (h1 : condition1 J M) (h2 : condition2 J M) : J = 11 := by
  -- proof will go here
  sorry

end johns_age_l357_357433


namespace rhombus_octagon_area_l357_357557

theorem rhombus_octagon_area (a b : ℝ) : 
  let T := 2 * a * b,
      r := a * b / Real.sqrt (a^2 + b^2) in
  (T - 2 * (T - a * b * ((a + b) / Real.sqrt (a^2 + b^2)))) = 
  4 * a * b * ((a + b) / Real.sqrt (a^2 + b^2) - 1) :=
  sorry

end rhombus_octagon_area_l357_357557


namespace sequence_geometric_general_formula_and_sum_l357_357776

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * (a n) + 1

theorem sequence_geometric :
  ∃ r : ℕ, ∀ n : ℕ, a n + 1 = r * 2^n := 
sorry

theorem general_formula_and_sum (n : ℕ) :
  let a_n := a n in
    a n = 2^n - 1 
    ∧ (∑ i in Finset.range (n+1), a i) = 2^(n + 1) - 2 - n := 
sorry

end sequence_geometric_general_formula_and_sum_l357_357776


namespace convert_point_to_spherical_coordinates_l357_357713

def point_rectangular := (4 * real.sqrt 3, -4, 2)

def spherical_coordinates (ρ θ φ : ℝ) := 
  ρ > 0 ∧ 
  0 ≤ θ ∧ θ < 2 * real.pi ∧ 
  0 ≤ φ ∧ φ ≤ real.pi

theorem convert_point_to_spherical_coordinates :
  ∃ (ρ θ φ : ℝ),
    spherical_coordinates ρ θ φ ∧
    (point_rectangular = 
      (ρ * real.sin φ * real.cos θ,
       ρ * real.sin φ * real.sin θ,
       ρ * real.cos φ)) ∧
    ρ = 2 * real.sqrt 17 ∧
    θ = 4 * real.pi / 3 ∧
    φ = real.arccos (1 / real.sqrt 17) :=
by sorry

end convert_point_to_spherical_coordinates_l357_357713


namespace determine_k_zero_of_ln_function_l357_357718

theorem determine_k_zero_of_ln_function :
  (∃ k : ℤ, (∀ x : ℝ, 2 ≤ x ∧ x < 3 → ln x + 2 * x - 6 = 0) ∧ 2 ≤ k ∧ k < 3) → k = 2 := by
  sorry

end determine_k_zero_of_ln_function_l357_357718


namespace sequence_poly_l357_357739

theorem sequence_poly (f : ℕ → ℕ) (h0 : f 0 = 1) (h1 : f 1 = 3) 
                      (h2 : f 2 = 7) (h3 : f 3 = 13) (h4 : f 4 = 21) :
  ∀ n, f(n) = n^2 + n + 1 :=
sorry

end sequence_poly_l357_357739


namespace complex_modulus_div_l357_357399

theorem complex_modulus_div : 
  ∀ (i : ℂ), i * i = -1 → abs (2 * i / (1 - i)) = Real.sqrt 2 :=
by
  assume (i : ℂ) (h : i * i = -1)
  sorry

end complex_modulus_div_l357_357399


namespace complex_number_quadrant_l357_357473

theorem complex_number_quadrant :
  let z := (2 - (1 * Complex.I)) / (1 + (1 * Complex.I))
  (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end complex_number_quadrant_l357_357473


namespace max_snake_length_can_turn_around_l357_357273

def Grid := Fin 3 × Fin 3

structure Snake (k : ℕ) :=
(cells : Fin k → Grid)
(distinct : ∀ i j : Fin k, i ≠ j → cells i ≠ cells j)
(adjacent : ∀ i : Fin (k - 1), (cells i.1, cells i.2) ∈ {(i, j) | (i.fst, i.snd) = (j.fst + 1, j.snd) ∨ (i.fst, i.snd) = (j.fst - 1, j.snd) ∨ (i.fst, i.snd) = (j.fst, j.snd + 1) ∨ (i.fst, i.snd) = (j.fst, j.snd - 1)})

def canTurnAround (k : ℕ) (snake : Snake k) : Prop :=
∃ seq : List Grid, seq = list.tail (list.reverse (snake.cells.to_list)) ∧ 
(∀ (i : Fin k), list.nth seq i = list.nth (snake.cells.to_list) i)

theorem max_snake_length_can_turn_around : ∃ (k : ℕ), k = 5 ∧ 
(∀ (snake : Snake k), canTurnAround k snake) ∧ 
(∀ (snake : Snake (k + 1)), ¬ canTurnAround (k + 1) snake) :=
sorry

end max_snake_length_can_turn_around_l357_357273


namespace circumcenter_rational_l357_357547

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l357_357547


namespace verify_a_l357_357330

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l357_357330


namespace sum_of_cubes_l357_357268

open Classical BigOperators

-- Define the series a, b, and c according to the conditions given
noncomputable def a (n : ℕ) (x : ℝ) : ℝ := ∑ i in (range (n / 3 + 1)).filter (λ i, i % 3 = 0), (nat.choose n i) * x ^ i
noncomputable def b (n : ℕ) (x : ℝ) : ℝ := ∑ i in (range (n / 4 + 1)).filter (λ i, i % 3 = 1), (nat.choose n i) * x ^ i
noncomputable def c (n : ℕ) (x : ℝ) : ℝ := ∑ i in (range (n / 5 + 1)).filter (λ i, i % 3 = 2), (nat.choose n i) * x ^ i

-- Define the theorem to be proven
theorem sum_of_cubes (n : ℕ) (x : ℝ) : (a n x) ^ 3 + (b n x) ^ 3 + (c n x) ^ 3 - 3 * (a n x) * (b n x) * (c n x) = (1 + x^3)^n := 
sorry

end sum_of_cubes_l357_357268


namespace inequality_for_kn_sum_l357_357850

open Real

theorem inequality_for_kn_sum (k n : ℕ) (hk : k > 0) (hn : n > 0) :
    (∑ i in range n, 1 / (kn + i)) ≥ n * (root n ((k + 1) / k) - 1) :=
by
  sorry

end inequality_for_kn_sum_l357_357850


namespace find_a_l357_357406

noncomputable def curve (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem find_a :
  ∃ a : ℝ, a = -1 ∧ ∀ x, x = (Real.pi / 2) → 
    DifferentiableAt ℝ curve x ∧ 
    DerivAt ℝ curve x x = -1 ∧
    ParallelLines x (curve x) (x - a * (curve x) + 1 = 0) :=
by
  sorry

end find_a_l357_357406


namespace fraction_of_red_marbles_after_adding_l357_357451

theorem fraction_of_red_marbles_after_adding :
  ∀ (initial_total : ℕ) (initial_blue : ℕ),
  initial_total = 50 →
  initial_blue = 30 →
  let initial_red := initial_total - initial_blue in
  let final_total := 65 in
  let added_red := final_total - initial_total in
  let final_red := initial_red + added_red in
  (initial_red + added_red) = 35 →
  final_red / final_total = 7 / 13 := 
by
  intros initial_total initial_blue h_total h_blue initial_red final_total added_red final_red h_final_red
  have h1 : initial_red = 20 := by linarith [h_total, h_blue]
  have h2 : added_red = 15 := by linarith [h_total]
  have h3 : final_red = 35 := by linarith [h1, h2]
  have h4 : final_total = 65 := rfl
  have h_fraction : final_red / final_total = 7 / 13 := by linarith [h3, h4]
  exact h_fraction

end fraction_of_red_marbles_after_adding_l357_357451


namespace daily_reading_goal_l357_357487

-- Define the problem conditions
def total_days : ℕ := 30
def goal_pages : ℕ := 600
def busy_days_13_16 : ℕ := 4
def busy_days_20_25 : ℕ := 6
def flight_day : ℕ := 1
def flight_pages : ℕ := 100

-- Define the mathematical equivalent proof problem in Lean 4
theorem daily_reading_goal :
  (total_days - busy_days_13_16 - busy_days_20_25 - flight_day) * 27 + flight_pages >= goal_pages :=
by
  sorry

end daily_reading_goal_l357_357487


namespace vertex_of_parabola_l357_357973

theorem vertex_of_parabola :
  ∃ h k, (h = 1) ∧ (k = 3) ∧
    (∀ x : ℝ, (x - h)^2 + k = (x - 1)^2 + 3) :=
begin
  use [1, 3],
  split,
  { refl },
  split,
  { refl },
  intros x,
  refl,
end

end vertex_of_parabola_l357_357973


namespace sorted_columns_preserve_row_order_l357_357668

theorem sorted_columns_preserve_row_order {α : Type*} [linear_order α] 
  (table : list (list α)) 
  (h_row_sorted : ∀ row ∈ table, list.sorted (≤) row) 
  (h_rectangular : ∃ n, ∀ row ∈ table, row.length = n) : 
  let sorted_columns_table := list.transpose (list.transpose table).map (λ col, list.sort (≤) col) 
  in ∀ row ∈ sorted_columns_table, list.sorted (≤) row :=
begin
  sorry
end

end sorted_columns_preserve_row_order_l357_357668


namespace metallic_sheet_dimension_l357_357287

theorem metallic_sheet_dimension (x : ℝ) (h₁ : ∀ (l w h : ℝ), l = x - 8 → w = 28 → h = 4 → l * w * h = 4480) : x = 48 :=
sorry

end metallic_sheet_dimension_l357_357287


namespace range_of_x_l357_357867

theorem range_of_x (x : ℝ) (h : 4 * x - 12 ≥ 0) : x ≥ 3 := 
sorry

end range_of_x_l357_357867


namespace part1_part2_l357_357121

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357121


namespace math_problem_l357_357136

variable {x y z : ℝ}

def condition1 (x : ℝ) := x = 1.2 * 40
def condition2 (x y : ℝ) := y = x - 0.35 * x
def condition3 (x y z : ℝ) := z = (x + y) / 2

theorem math_problem (x y z : ℝ) (h1 : condition1 x) (h2 : condition2 x y) (h3 : condition3 x y z) :
  z = 39.6 :=
by
  sorry

end math_problem_l357_357136


namespace kenzo_remaining_legs_l357_357880

variable (chairs : ℕ) (legsPerChair : ℕ) (tables : ℕ) (legsPerTable : ℕ) (percentDamaged : ℕ)

def total_legs_remaining (chairs legsPerChair tables legsPerTable percentDamaged : ℕ) : ℕ :=
  let totalChairLegs := chairs * legsPerChair
  let totalTableLegs := tables * legsPerTable
  let totalLegsBeforeDamage := totalChairLegs + totalTableLegs
  let damagedChairs := (percentDamaged * chairs) / 100
  let damagedLegs := damagedChairs * legsPerChair
  totalLegsBeforeDamage - damagedLegs

theorem kenzo_remaining_legs (h1 : chairs = 80) (h2 : legsPerChair = 5) (h3 : tables = 20) (h4 : legsPerTable = 3) (h5 : percentDamaged = 40) :
  total_legs_remaining chairs legsPerChair tables legsPerTable percentDamaged = 300 := by
  simp [total_legs_remaining, h1, h2, h3, h4, h5]
  sorry

end kenzo_remaining_legs_l357_357880


namespace sphere_shadow_l357_357673

/-- 
Given a sphere with radius 2 and center at (0,0,2), with a light source at (0,-2,4),
the boundary of the sphere's shadow on the xy-plane is given by the equation y = -6.
-/
theorem sphere_shadow (x : ℝ) : 
  let O := (0 : ℝ, 0 : ℝ, 2 : ℝ)
  let P := (0 : ℝ, -2 : ℝ, 4 : ℝ)
  let radius := (2 : ℝ)
  let PO := (0 : ℝ, 2 : ℝ, -2 : ℝ)
  let PX := (x : ℝ, y + 2 : ℝ, -4 : ℝ)
  (PO.1 * PX.1 + PO.2 * PX.2 + PO.3 * PX.3 = 0) → y = -6 :=
by 
  sorry

end sphere_shadow_l357_357673


namespace circumcenter_rational_l357_357541

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l357_357541


namespace perpendicular_midpoint_chords_l357_357282

/-- A convex quadrilateral ABCD is inscribed in a circle. 
    Show that the chord connecting the midpoints of the two arcs 
    \widehat{AB} and \widehat{CD} is perpendicular to the chord 
    connecting the two arc midpoints of \widehat{BC} and \widehat{DA}. -/
theorem perpendicular_midpoint_chords (A B C D : Point)
  (hABCD_circle : Circle ABCD)
  (hABCD_convex : Convex ABCD)
  (M : Point) (hM_midpoint : ArcMidpoint M A B)
  (N : Point) (hN_midpoint : ArcMidpoint N B C)
  (P : Point) (hP_midpoint : ArcMidpoint P C D)
  (Q : Point) (hQ_midpoint : ArcMidpoint Q D A)
  : Perpendicular (Chord M P) (Chord N Q) :=
sorry

end perpendicular_midpoint_chords_l357_357282


namespace complement_set_l357_357814

theorem complement_set (U : Set ℝ) (A : Set ℝ) (hU : U = Set.Univ) (hA : A = {x : ℝ | |x - 1| > 2}) :
  (U \ A) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end complement_set_l357_357814


namespace subsets_containing_six_l357_357832

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l357_357832


namespace arithmetic_seq_sum_l357_357472

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_sum : a 0 + a 1 + a 2 + a 3 = 30) : a 1 + a 2 = 15 :=
by
  sorry

end arithmetic_seq_sum_l357_357472


namespace no_solutions_system_of_inequalities_l357_357264

open Set

theorem no_solutions_system_of_inequalities :
  ∀ (x y : ℝ),
    ¬(11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧ 5 * x + y ≤ -10) :=
by
  intro x y
  rw not_and
  intro h1 h2
  let y' := -10 - 5 * x
  have h3 : y = y' := eq_of_le_of_le h2 (le_of_eq rfl)
  sorry

end no_solutions_system_of_inequalities_l357_357264


namespace sum_f_2_to_200_l357_357127

def g_k (k x : ℕ) : ℕ := 
  let d := x^2
  floor (d / (10 ^ (floor (log d / log 10) + 1 - k)))

def x_n (k : ℕ) : ℕ :=
  let a := nat.find (λ a, g_k k a - g_k k (a - 1) ≥ 2)
  a

def f (k : ℕ) : ℕ :=
  x_n k

def sum_f : ℕ := 
  (2.to 200).step(2).sum f

theorem sum_f_2_to_200 : sum_f = 2650 := 
  sorry

end sum_f_2_to_200_l357_357127


namespace part1_part2_l357_357505

noncomputable theory
open Real

def f (x w : ℝ) : ℝ := 4 * cos (w * x + π / 6) * sin (w * x) - cos (2 * w * x) + 1

def symmetry_axis (w : ℝ) : ℝ := π / 4

def period_T (w : ℝ) : ℝ := π

def increasing_interval (w : ℝ) : Set ℝ := Icc (-π/6) (π/3)

theorem part1 (w : ℝ) (h₀ : 0 < w ∧ w < 2) (h₁ : symmetry_axis w = π / 4) : 
  ∀ x, f (x + period_T w) w = f x w := sorry

theorem part2 (h₀ : f (π / 3) (3/4) < f (π / 3 + ε) (3/4)) (h₁ : f (-π / 6) (3/4) < f (-π / 6 + ε) (3/4)) : 
  0 < (3/4) ∧ (3/4) < 2 := sorry

end part1_part2_l357_357505


namespace circumscribed_quadrilateral_theorem_l357_357773

noncomputable def circumscribed_quadrilateral (a b c d : ℝ) (O A B C D : ℝ) : Prop :=
  -- The quadrilateral ABCD is circumscribed about the circle centered at O.
  -- Side lengths are a, b, c, and d.
  OA * OC + OB * OD = sqrt (a * b * c * d)

theorem circumscribed_quadrilateral_theorem (O A B C D : ℝ) (a b c d : ℝ) 
  (h : circumscribed_quadrilateral a b c d O A B C D) :
  h :=
sorry

end circumscribed_quadrilateral_theorem_l357_357773


namespace money_left_is_correct_l357_357333

-- Define initial amount of money Dan has
def initial_amount : ℕ := 3

-- Define the cost of the candy bar
def candy_cost : ℕ := 1

-- Define the money left after the purchase
def money_left : ℕ := initial_amount - candy_cost

-- The theorem stating that the money left is 2
theorem money_left_is_correct : money_left = 2 := by
  sorry

end money_left_is_correct_l357_357333


namespace cos_angle_of_ellipse_l357_357078

noncomputable def cos_angle (A B C : ℝ × ℝ) : ℝ :=
  let d_ab := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let d_ac := (A.1 - C.1)^2 + (A.2 - C.2)^2
  let d_bc := (B.1 - C.1)^2 + (B.2 - C.2)^2
  (d_ab + d_ac - d_bc) / (2 * real.sqrt d_ab * real.sqrt d_ac)

theorem cos_angle_of_ellipse :
  let O := (0, 0)
  let F1 := (-√2, 0)
  let F2 := (√2, 0)
  let P := (1, √2)
  cos_angle F1 P F2 = 1 / 3 :=
by
  sorry

end cos_angle_of_ellipse_l357_357078


namespace oleg_tulips_ways_l357_357153

theorem oleg_tulips_ways :
  (∑ k in finset.range 12, if odd k then (nat.choose 11 k) else 0) = 1024 :=
by
  sorry

end oleg_tulips_ways_l357_357153


namespace find_n_l357_357384

-- Define the first term a₁, the common ratio q, and the sum Sₙ
def a₁ : ℕ := 2
def q : ℕ := 2
def Sₙ (n : ℕ) : ℕ := 2^(n + 1) - 2

-- The sum of the first n terms is given as 126
def given_sum : ℕ := 126

-- The theorem to be proven
theorem find_n (n : ℕ) (h : Sₙ n = given_sum) : n = 6 :=
by
  sorry

end find_n_l357_357384


namespace part1_part2_l357_357114

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357114


namespace smallest_positive_integer_with_eight_distinct_positive_factors_l357_357849

open Nat

-- Defining a function that calculates the number of distinct positive factors
def numDivisors (n : Nat) : Nat := (n.factors).toFinset.card

-- Defining a predicate that identifies numbers with exactly 8 distinct positive factors
def hasEightDistinctFactors (n : Nat) : Prop :=
  numDivisors n = 8

-- Defining the smallest number with exactly 8 distinct positive factors
def smallestNumWithEightDistinctFactors : Nat :=
  24

-- The theorem stating that 24 is the smallest positive integer with exactly 8 distinct positive factors
theorem smallest_positive_integer_with_eight_distinct_positive_factors :
  smallestNumWithEightDistinctFactors = 24 ∧
  hasEightDistinctFactors 24 :=
begin
  split,
  -- Prove that 24 is the smallest number that meets the condition
  { reflexivity },
  -- Prove that 24 has exactly 8 distinct factors
  { sorry }
end

end smallest_positive_integer_with_eight_distinct_positive_factors_l357_357849


namespace sum_of_roots_eq_three_l357_357936

theorem sum_of_roots_eq_three (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + 2 = 0)
  (h2 : x2^2 - 3*x2 + 2 = 0) 
  (h3 : x1 ≠ x2) : 
  x1 + x2 = 3 := 
sorry

end sum_of_roots_eq_three_l357_357936


namespace life_journey_cd_price_l357_357012

theorem life_journey_cd_price 
(price_ADL price_WYR total_cost : ℕ) 
(h1 : price_ADL = 50) 
(h2 : price_WYR = 85) 
(h3 : total_cost = 705) :
  3 * (100 : ℕ) + 3 * price_ADL + 3 * price_WYR = total_cost :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end life_journey_cd_price_l357_357012


namespace find_f2_l357_357783

-- Definitions based on the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_g_def : ∀ x, g x = f x + 9)
variable (h_g_val : g (-2) = 3)

-- Prove the required goal
theorem find_f2 : f 2 = 6 :=
by
  sorry

end find_f2_l357_357783


namespace num_subsets_containing_6_l357_357840

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l357_357840


namespace fourth_vertex_l357_357474

-- Define the given vertices
def vertex1 := (2, 1)
def vertex2 := (4, 1)
def vertex3 := (2, 5)

-- Define what it means to be a rectangle in this context
def is_vertical_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.1 = p2.1

def is_horizontal_segment (p1 p2 : ℕ × ℕ) : Prop :=
  p1.2 = p2.2

def is_rectangle (v1 v2 v3 v4: (ℕ × ℕ)) : Prop :=
  is_vertical_segment v1 v3 ∧
  is_horizontal_segment v1 v2 ∧
  is_vertical_segment v2 v4 ∧
  is_horizontal_segment v3 v4 ∧
  is_vertical_segment v1 v4 ∧ -- additional condition to ensure opposite sides are equal
  is_horizontal_segment v2 v3

-- Prove the coordinates of the fourth vertex of the rectangle
theorem fourth_vertex (v4 : ℕ × ℕ) : 
  is_rectangle vertex1 vertex2 vertex3 v4 → v4 = (4, 5) := 
by
  intro h_rect
  sorry

end fourth_vertex_l357_357474


namespace job_completion_l357_357712

noncomputable def minimumWorkers (total_days : ℕ) (work_fraction_done : ℚ) (days_worked : ℕ) (initial_workers : ℕ) (remaining_work_fraction : ℚ) (remaining_days : ℕ) : ℕ := 
  let daily_work_per_worker := work_fraction_done / (days_worked * initial_workers)
  let required_daily_work := remaining_work_fraction / remaining_days
  (required_daily_work / daily_work_per_worker).ceil.to_nat

theorem job_completion (total_days : ℕ) (days_worked : ℕ) (initial_workers : ℕ) (work_fraction_done : ℚ)
  (remaining_work_fraction : ℚ) (remaining_days : ℕ) :
  total_days = 40 ∧ days_worked = 6 ∧ initial_workers = 10 ∧ work_fraction_done = 1 / 4 ∧ remaining_work_fraction = 3 / 4 ∧ remaining_days = 34 →
  minimumWorkers total_days work_fraction_done days_worked initial_workers remaining_work_fraction remaining_days = 6 :=
by {
  sorry
} 

end job_completion_l357_357712


namespace g_monotonically_increasing_f_monotonically_increasing_f_bounded_l357_357760

-- Definitions for the functions f and g
def f (a x : ℝ) : ℝ := exp x - (1/2) * x^2 - a * sin x - 1
def g (a x : ℝ) : ℝ := f a x + f a (-x)

-- (1) Prove that g(x) is monotonically increasing on ℝ
theorem g_monotonically_increasing (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) : 
  ∀ x y : ℝ, x ≤ y → g a x ≤ g a y := sorry

-- (2i) Prove that f(x) is monotonically increasing on ℝ
theorem f_monotonically_increasing (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) : 
  ∀ x y : ℝ, x ≤ y → f a x ≤ f a y := sorry

-- (2ii) Prove that |f(x)| ≤ M given |f'(x)| ≤ M for x in [-π/3, π/3]
theorem f_bounded (a M : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  (∀ x : ℝ, -real.pi/3 ≤ x ∧ x ≤ real.pi/3 → abs (exp x - x - a * cos x) ≤ M) → 
  (∀ x : ℝ, -real.pi/3 ≤ x ∧ x ≤ real.pi/3 → abs (f a x) ≤ M) := sorry

end g_monotonically_increasing_f_monotonically_increasing_f_bounded_l357_357760


namespace inheritance_amount_l357_357490

-- Definitions of conditions
def inheritance (y : ℝ) : Prop :=
  let federalTaxes := 0.25 * y
  let remainingAfterFederal := 0.75 * y
  let stateTaxes := 0.1125 * y
  let totalTaxes := federalTaxes + stateTaxes
  totalTaxes = 12000

-- Theorem statement
theorem inheritance_amount (y : ℝ) (h : inheritance y) : y = 33103 :=
sorry

end inheritance_amount_l357_357490


namespace no_solution_system_of_inequalities_l357_357266

theorem no_solution_system_of_inequalities :
  ¬ ∃ (x y : ℝ),
    11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧
    5 * x + y ≤ -10 :=
by
  sorry

end no_solution_system_of_inequalities_l357_357266


namespace chocolate_milk_total_l357_357316

noncomputable def milk_oz_per_glass : ℝ := 6.5
noncomputable def syrup_oz_per_glass : ℝ := 1.5
noncomputable def total_milk_oz : ℝ := 130
noncomputable def total_syrup_oz : ℝ := 60

theorem chocolate_milk_total :
  (let glasses := min (total_milk_oz / milk_oz_per_glass) (total_syrup_oz / syrup_oz_per_glass) in
  glasses * (milk_oz_per_glass + syrup_oz_per_glass) = 160) :=
by
  sorry

end chocolate_milk_total_l357_357316


namespace number_of_marks_for_passing_l357_357648

theorem number_of_marks_for_passing (T P : ℝ) 
  (h1 : 0.40 * T = P - 40) 
  (h2 : 0.60 * T = P + 20) 
  (h3 : 0.45 * T = P - 10) :
  P = 160 :=
by
  sorry

end number_of_marks_for_passing_l357_357648


namespace sum_f_1_to_2019_l357_357657

def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -(x + 2) ^ 2
  else if -1 ≤ x ∧ x < 3 then x
  else f (x - 6)

theorem sum_f_1_to_2019 : (∑ k in finset.range 2020, f k) = 4038 :=
by
  sorry

end sum_f_1_to_2019_l357_357657


namespace math_problem_l357_357583
noncomputable def sum_of_terms (a b c d : ℕ) : ℕ := a + b + c + d

theorem math_problem
  (x y : ℝ)
  (h₁ : x + y = 5)
  (h₂ : 5 * x * y = 7) :
  ∃ a b c d : ℕ, 
  x = (a + b * Real.sqrt c) / d ∧
  a = 25 ∧ b = 1 ∧ c = 485 ∧ d = 10 ∧ sum_of_terms a b c d = 521 := by
sorry

end math_problem_l357_357583


namespace series_sum_nearest_integer_l357_357139

noncomputable def series_sum : ℝ :=
  ∑ n in Finset.range 100, (1 / (n + 1) ^ (n + 1))

theorem series_sum_nearest_integer : round series_sum = 1 := 
  sorry

end series_sum_nearest_integer_l357_357139


namespace player1_max_score_l357_357594

-- Define the board state and sum conditions
def Board := Array (Array Nat) -- A 5x5 board to be filled with 1s and 0s

def player1_can_place (board : Board) : Prop :=
  ∀ (i j : Fin 5), board[i][j] = 0 → board[i][j] = 1

def player2_can_place (board : Board) : Prop :=
  ∀ (i j : Fin 5), board[i][j] = 0 → board[i][j] = 0

def board_full (board : Board) : Prop :=
  ∀ (i j : Fin 5), board[i][j] ≠ 0

def sub_square_sum (board : Board) (x y : Fin 3) : Nat :=
  (board[x*3+y*3] + board[x*3+y*1] + board[x*3+y*2] +
  board[x*1+y*3] + board[x*1+y*1] + board[x*1+y*2] +
  board[x*2+y*3] + board[x*2+y*1] + board[x*2+y*2]) 

def max_3x3_sum (board : Board) : Nat :=
  max (sub_square_sum board 0 0) 
  (max (sub_square_sum board 0 1)
  (max (sub_square_sum board 0 2)
  (max (sub_square_sum board 1 0)
  (max (sub_square_sum board 1 1)
  (max (sub_square_sum board 1 2)
  (max (sub_square_sum board 2 0)
  (max (sub_square_sum board 2 1)
  (sub_square_sum board 2 2))))))))

theorem player1_max_score : 
  ∃ (board : Board), 
  player1_can_place board ∧ 
  player2_can_place board ∧ 
  board_full board →
  max_3x3_sum board = 6 := 
sorry

end player1_max_score_l357_357594


namespace percentage_more_l357_357145

variable (J : ℝ) -- Juan's income
noncomputable def Tim_income := 0.60 * J -- T = 0.60J
noncomputable def Mart_income := 0.84 * J -- M = 0.84J

theorem percentage_more {J : ℝ} (T := Tim_income J) (M := Mart_income J) :
  ((M - T) / T) * 100 = 40 := by
  sorry

end percentage_more_l357_357145


namespace construct_pairwise_tangent_circles_l357_357585

-- Definitions of the points A, B, and C
variables (A B C : ℝ × ℝ)

-- Definitions of incenter I, and points of tangency K, L, M
def incenter (A B C : ℝ × ℝ) : ℝ × ℝ := sorry -- Placeholder for incentre calculation

def point_tangency (I A B C : ℝ × ℝ) (side : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ := sorry -- Placeholder for tangency calculation

noncomputable def K := point_tangency (incenter A B C) A B C (B, C)
noncomputable def L := point_tangency (incenter A B C) A B C (C, A)
noncomputable def M := point_tangency (incenter A B C) A B C (A, B)

-- Statement of the theorem
theorem construct_pairwise_tangent_circles :
  ∃ (c1 c2 c3 : ℝ × ℝ × ℝ), 
    c1 = (K, radius) ∧ c2 = (L, radius) ∧ c3 = (M, radius) ∧
    tangent c1 c2 ∧ tangent c2 c3 ∧ tangent c3 c1 :=
sorry

end construct_pairwise_tangent_circles_l357_357585


namespace max_marks_mike_l357_357517

theorem max_marks_mike (marks_to_pass : ℝ) (marks_scored : ℝ) (marks_short : ℝ)
  (h1 : marks_to_pass = 0.45)
  (h2 : marks_scored = 267)
  (h3 : marks_short = 43) :
  let passing_marks := marks_scored + marks_short,
      max_possible_marks := passing_marks / marks_to_pass.ceil in
  max_possible_marks = 689 :=
by
  sorry

end max_marks_mike_l357_357517


namespace problem_statement_l357_357506

variables {Point : Type} {Line Plane : Type}

-- Assume lines and planes are non-coincident
variables (m n l : Line) (α β γ : Plane)

-- Assume lines m, n, and l are distinct
axiom line_distinct : ¬(m = n ∨ n = l ∨ m = l)
-- Assume planes α, β, and γ are distinct
axiom plane_distinct : ¬(α = β ∨ β = γ ∨ α = γ)

-- Predicate for parallelism between lines and planes
---@[irreducible] def Parallel : Plane → Plane → Prop := sorry

-- Formal statement of the problem
theorem problem_statement 
  (h1 : Parallel α γ)
  (h2 : Parallel β γ) : Parallel α β := sorry

end problem_statement_l357_357506


namespace swimmer_speeds_l357_357635

variable (a s r : ℝ)
variable (x z y : ℝ)

theorem swimmer_speeds (h : s < r) (h' : r < 100 * s / (50 + s)) :
    (100 * s - 50 * r - r * s) / ((3 * s - r) * a) = x ∧ 
    (100 * s - 50 * r - r * s) / ((r - s) * a) = z := by
    sorry

end swimmer_speeds_l357_357635


namespace quadrilateral_condition_l357_357992

variables (A B C D M : Type*)

-- Conditions
variables [real_vector_space M] (a b c d m : M)
variable (MA_eq_MC : dist a m = dist c m)
variable (angle_AMB_eq : angle a m b = angle a m d + angle m c d)
variable (angle_CMD_eq : angle c m d = angle m c b + angle m a b)

-- Proof statement
theorem quadrilateral_condition 
  (MA_eq_MC : dist a m = dist c m)
  (angle_AMB_eq : angle a m b = angle a m d + angle m c d)
  (angle_CMD_eq : angle c m d = angle m c b + angle m a b) :
  (dist a b * dist c m = dist b c * dist m d) ∧
  (dist b m * dist a d = dist m a * dist c d) :=
sorry

end quadrilateral_condition_l357_357992


namespace student_passes_test_probability_l357_357877

noncomputable def probability_passes_test : ℝ :=
  let p := 0.6 in
  let n := 3 in
  let k := 2 in
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) + (Nat.choose n (k + 1)) * (p ^ (k + 1)) * ((1 - p) ^ (n - (k + 1)))

theorem student_passes_test_probability :
  probability_passes_test = 0.648 :=
by
  sorry

end student_passes_test_probability_l357_357877


namespace correct_statements_l357_357302

-- Definitions based on conditions
def synthetic_method_is_cause_and_effect (synthetic_method : Type) : Prop :=
  synthetic_method = "cause_and_effect"

def synthetic_method_is_forward_reasoning (synthetic_method : Type) : Prop :=
  synthetic_method = "forward_reasoning"

def analytical_method_is_cause_seeking (analytical_method : Type) : Prop :=
  analytical_method = "cause_seeking"

def analytical_method_is_indirect_proof (analytical_method : Type) : Prop :=
  analytical_method = "indirect_proof"

def contradiction_method_is_reverse_reasoning (contradiction_method : Type) : Prop :=
  contradiction_method = "reverse_reasoning"

-- Given definitions to prove correctness of statements ①, ②, and ③
theorem correct_statements :
  ∀ (synthetic_method analytical_method contradiction_method : Type),
    synthetic_method_is_cause_and_effect synthetic_method →
    synthetic_method_is_forward_reasoning synthetic_method →
    analytical_method_is_cause_seeking analytical_method →
    ¬analytical_method_is_indirect_proof analytical_method →
    ¬contradiction_method_is_reverse_reasoning contradiction_method →
    ({synthetic_method = "cause_and_effect", synthetic_method = "forward_reasoning", analytical_method = "cause_seeking"} = set.insert "cause_and_effect" (set.insert "forward_reasoning" (set.singleton "cause_seeking"))) := by
  sorry

end correct_statements_l357_357302


namespace angle_AO2B_l357_357539

theorem angle_AO2B (O1 O2 A B : Type) (r R : ℝ) 
  (h1: r * r = (2:ℝ) * r * r * ((- (1/2):ℝ )) + r * r + r * r)
  (h2: R = r * real.sqrt 3) : 
  ∠A O2 B = 60 := 
sorry

end angle_AO2B_l357_357539


namespace min_integer_a_l357_357864

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 1

theorem min_integer_a (a : ℤ) :
  (∃! x : ℤ, f x < g a x) = 3 ∧
  ∀ b : ℤ, ∃! x : ℤ, f x < g b x = 3 → b ≥ 3 :=
sorry

end min_integer_a_l357_357864


namespace value_of_5a_l357_357432

variable (a : ℕ)

theorem value_of_5a (h : 5 * (a - 3) = 25) : 5 * a = 40 :=
sorry

end value_of_5a_l357_357432


namespace correct_Y_when_T_reaches_5000_l357_357021

noncomputable def Y_when_T_reaches_5000 : ℕ :=
  let initial_Y := 5
  let initial_T := 0
  let step_Y := 3
  let T_threshold := 5000

  -- Helper function to compute the iteration step n when T exceeds the given threshold
  let rec find_n_Y (n : ℕ) (Y : ℕ) (T : ℕ) : ℕ :=
    if T ≥ T_threshold then Y
    else
      let new_Y := Y + step_Y
      let new_T := T + new_Y^2
      find_n_Y (n + 1) new_Y new_T

  find_n_Y 0 initial_Y initial_T

theorem correct_Y_when_T_reaches_5000 : Y_when_T_reaches_5000 = 32 :=
  by
    -- The proof steps will go here
    sorry

end correct_Y_when_T_reaches_5000_l357_357021


namespace find_area_of_triangle_l357_357514

noncomputable def area_of_triangle
  (A B C : ℝ × ℝ)
  (right_angle : ∠C = 90)
  (length_hypotenuse : dist A B = 60)
  (median_A : ∀ (p : ℝ × ℝ), line_through A p ↔ (p.2 = p.1 + 3))
  (median_B : ∀ (q : ℝ × ℝ), line_through B q ↔ (q.2 = 2 * q.1 + 4)) 
  : ℝ :=
  (1 / 2) * base * height
  where
    base := dist B C
    height := dist A C

theorem find_area_of_triangle
  (A B C : ℝ × ℝ)
  (right_angle : ∠C = 90)
  (length_hypotenuse : dist A B = 60)
  (median_A : ∀ (p : ℝ × ℝ), line_through A p ↔ (p.2 = p.1 + 3))
  (median_B : ∀ (q : ℝ × ℝ), line_through B q ↔ (q.2 = 2 * q.1 + 4))
  : area_of_triangle A B C right_angle length_hypotenuse median_A median_B = 400 := sorry

end find_area_of_triangle_l357_357514


namespace circle_line_intersection_probability_l357_357380

def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def line (x y b : ℝ) : Prop := y = x + b

def intersects (b : ℝ) : Prop := ∃ x y, circle x y ∧ line x y b

def probability_of_intersection : Real := (2 * Real.sqrt 2) / 5

theorem circle_line_intersection_probability : 
  (b : ℝ), b ∈ Set.Icc (-5 : ℝ) 5 → 
  (if intersects b then 1 else 0) = 
  probability_of_intersection := 
by sorry

end circle_line_intersection_probability_l357_357380


namespace least_perimeter_triangle_l357_357071

theorem least_perimeter_triangle
  (a b c : ℕ) 
  (h1 : ∃ M O : Type, is_median C M ∧ is_circumcenter O) 
  (h2 : circumcircle_bisects_CM) 
  (h3 : side_lengths a b c) 
  (h4 : integer_side_lengths a b c) :
  a + b + c = 24 :=
sorry

end least_perimeter_triangle_l357_357071


namespace martin_discounted_tickets_l357_357623

-- Definitions of the problem conditions
def total_tickets (F D : ℕ) := F + D = 10
def total_cost (F D : ℕ) := 2 * F + (16/10) * D = 184/10

-- Statement of the proof
theorem martin_discounted_tickets (F D : ℕ) (h1 : total_tickets F D) (h2 : total_cost F D) :
  D = 4 :=
sorry

end martin_discounted_tickets_l357_357623


namespace part1_part2_l357_357113

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357113


namespace find_k_parallel_line_l357_357721

/-- Given two points (x1, y1) and (x2, y2), calculate the slope -/
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

/-- Given the parameters. If a line through two points (5, -4) and (k, 23) is parallel to the line 6x - 2y = -8, then k = 14. -/
theorem find_k_parallel_line (k : ℝ) (p1 p2 : ℝ × ℝ) (line_eq : ℝ × ℝ) :
  let slope1 := slope 5 (-4) k 23,
      slope2 := slope 0 4 2 (-1) -- Because 6x - 2y = -8 transforms to y = 3x + 4
  in slope1 = slope2 → k = 14 :=
by
  intros slope1 slope2 H,
  sorry  -- Proof to be filled in later

end find_k_parallel_line_l357_357721


namespace hexagon_area_is_20_l357_357238

theorem hexagon_area_is_20 :
  let upper_base1 := 3
  let upper_base2 := 2
  let upper_height := 4
  let lower_base1 := 3
  let lower_base2 := 2
  let lower_height := 4
  let upper_trapezoid_area := (upper_base1 + upper_base2) * upper_height / 2
  let lower_trapezoid_area := (lower_base1 + lower_base2) * lower_height / 2
  let total_area := upper_trapezoid_area + lower_trapezoid_area
  total_area = 20 := 
by {
  sorry
}

end hexagon_area_is_20_l357_357238


namespace almost_monotonic_digits_0_to_8_l357_357314

def binom (n k : ℕ) : ℕ := Nat.choose n k

def almost_monotonic_count : ℕ :=
  let count_nondecreasing := (Finset.range 9).sum (λ n => binom (n + 8) 8)
  2 * count_nondecreasing - 9

theorem almost_monotonic_digits_0_to_8 :
  almost_monotonic_count = 97227 := by
  sorry

end almost_monotonic_digits_0_to_8_l357_357314


namespace medians_perpendicular_iff_l357_357529

-- Define the sides of the triangle
variables (a b c : ℝ)

-- Define the medians from vertices A and B
def AM : ℝ := (sqrt ((2 * b^2 + 2 * c^2 - a^2) / 4))
def BM : ℝ := (sqrt ((2 * a^2 + 2 * c^2 - b^2) / 4))

-- Define a proof problem
theorem medians_perpendicular_iff :
  ((AM a b c) * (BM a b c) = 0) ↔ (a^2 + b^2 = 5 * c^2) :=
sorry

end medians_perpendicular_iff_l357_357529


namespace zero_in_interval_l357_357207

-- Define the function f(x) = e^x - x - 2
def f (x : ℝ) : ℝ := Real.exp x - x - 2

-- Define the main theorem statement
theorem zero_in_interval :
  ∃ x ∈ Ioo 1 2, f x = 0 := by
  sorry

end zero_in_interval_l357_357207


namespace choir_average_age_l357_357178

theorem choir_average_age
  (avg_females_age : ℕ)
  (num_females : ℕ)
  (avg_males_age : ℕ)
  (num_males : ℕ)
  (females_avg_condition : avg_females_age = 28)
  (females_num_condition : num_females = 8)
  (males_avg_condition : avg_males_age = 32)
  (males_num_condition : num_males = 17) :
  ((avg_females_age * num_females + avg_males_age * num_males) / (num_females + num_males) = 768 / 25) :=
by
  sorry

end choir_average_age_l357_357178


namespace fraction_ordering_l357_357224

theorem fraction_ordering:
  (6 / 22) < (5 / 17) ∧ (5 / 17) < (8 / 24) :=
by
  sorry

end fraction_ordering_l357_357224


namespace area_of_region_is_9_div_2_l357_357939

def in_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + y + ⌊x⌋ + ⌊y⌋ ≤ 5

theorem area_of_region_is_9_div_2 :
  let R := { p : ℝ × ℝ | in_region p.1 p.2 }
  area R = 9 / 2 :=
sorry

end area_of_region_is_9_div_2_l357_357939


namespace highest_qualification_number_possible_l357_357970

theorem highest_qualification_number_possible (n : ℕ) (qualifies : ℕ → ℕ → Prop)
    (h512 : n = 512)
    (hqualifies : ∀ a b, qualifies a b ↔ (a < b ∧ b - a ≤ 2)): 
    ∃ k, k = 18 ∧ (∀ m, qualifies m k → m < k) :=
by
  sorry

end highest_qualification_number_possible_l357_357970


namespace find_k_l357_357430

theorem find_k (k : ℕ) : (1 / 2) ^ 18 * (1 / 81) ^ k = 1 / 18 ^ 18 → k = 0 := by
  intro h
  sorry

end find_k_l357_357430


namespace discriminant_is_four_l357_357576

-- Define the quadratic equation components
def quadratic_a (a : ℝ) := 1
def quadratic_b (a : ℝ) := 2 * a
def quadratic_c (a : ℝ) := a^2 - 1

-- Define the discriminant of the quadratic equation
def discriminant (a : ℝ) := quadratic_b a ^ 2 - 4 * quadratic_a a * quadratic_c a

-- Statement to prove: The discriminant is 4
theorem discriminant_is_four (a : ℝ) : discriminant a = 4 :=
by {
  sorry
}

end discriminant_is_four_l357_357576


namespace find_cosC_l357_357893

namespace Triangle

variable {A B C : Type} [linear_ordered_field A]
variables (a b c : A)

/- Given the conditions in the problem -/
axiom cond1 : 3 * a^2 + 3 * b^2 - 3 * c^2 = 2 * a * b

/- State the theorem to be proved -/
theorem find_cosC : ∃ cosC : A, (cosC = 1 / 3) :=
by
  use 1 / 3
  sorry  -- Proof goes here

end Triangle

end find_cosC_l357_357893


namespace expression_simplification_l357_357614

theorem expression_simplification (x y : ℝ) :
  20 * (x + y) - 19 * (x + y) = x + y :=
by
  sorry

end expression_simplification_l357_357614


namespace average_salary_correct_l357_357996

def A_salary := 10000
def B_salary := 5000
def C_salary := 11000
def D_salary := 7000
def E_salary := 9000

def total_salary := A_salary + B_salary + C_salary + D_salary + E_salary
def num_individuals := 5

def average_salary := total_salary / num_individuals

theorem average_salary_correct : average_salary = 8600 := by
  sorry

end average_salary_correct_l357_357996


namespace num_subsets_containing_6_l357_357842

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l357_357842


namespace f_quadruple_composition_l357_357370

def is_purely_real (z : ℂ) : Prop := z.im = 0

def f (z : ℂ) : ℂ :=
  if is_purely_real z then -z^2 else z^2

theorem f_quadruple_composition (z : ℂ) (h : z = 2 + 2 * complex.I) : f (f (f (f z))) = -16777216 :=
  by
  have h1 : f z = 8 * complex.I := by sorry
  have h2 : f (f z) = -64 := by sorry
  have h3 : f (f (f z)) = -4096 := by sorry
  have h4 : f (f (f (f z))) = -16777216 := by sorry
  exact h4

end f_quadruple_composition_l357_357370


namespace smallest_integer_in_set_l357_357874

open Real

theorem smallest_integer_in_set : 
  ∃ (n : ℤ), (λ n, let avg := (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7 
               in n + 6 > 2 * avg) n ∧ 
            ∀ k : ℤ, (λ k, let avg_k := (k + (k+1) + (k+2) + (k+3) + (k+4) + (k+5) + (k+6)) / 7 
               in k + 6 > 2 * avg_k) k → n ≤ k := 
by
  sorry

end smallest_integer_in_set_l357_357874


namespace subsets_with_six_l357_357847

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l357_357847


namespace min_operator_result_l357_357052

theorem min_operator_result : 
  min ((-3) + (-6)) (min ((-3) - (-6)) (min ((-3) * (-6)) ((-3) / (-6)))) = -9 := 
by 
  sorry

end min_operator_result_l357_357052


namespace average_price_orange_l357_357522

theorem average_price_orange :
  (15 * 5 + 25 * 5) / 20 = 10 :=
by
  -- Total cost for 5 packs of 4 oranges and 5 packs of 6 oranges
  have total_cost : 15 * 5 + 25 * 5 = 200 := by
    calc
      15 * 5 + 25 * 5 = 75 + 125 : by rw [mul_comm, mul_comm]
      ... = 200 : by norm_num
  -- Calculate average cost per orange
  calc
    (15 * 5 + 25 * 5) / 20 = total_cost / 20 : by rw total_cost
    ... = 200 / 20 : by refl
    ... = 10 : by norm_num

end average_price_orange_l357_357522


namespace other_number_eq_462_l357_357987

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l357_357987


namespace log_sum_equiv_l357_357353

theorem log_sum_equiv : log 50 / log 10 + log 20 / log 10 = 3 := by
  have h1 : log 50 / log 10 + log 20 / log 10 = log 1000 / log 10 := by
    rw [←log_mul 50 20, mul_comm]
  have h2 : log 1000 / log 10 = 3 := by
    rw [log_pow 10 3, log_nat_sum_eq_one]
  rw [h1, h2]
  exact rfl

end log_sum_equiv_l357_357353


namespace domain_lg_sqrt_l357_357337

def domain_of_function (x : ℝ) : Prop :=
  1 - x > 0 ∧ x + 2 > 0

theorem domain_lg_sqrt (x : ℝ) : 
  domain_of_function x ↔ -2 < x ∧ x < 1 :=
sorry

end domain_lg_sqrt_l357_357337


namespace overlap_32_l357_357138

section
variables (t : ℝ)
def position_A : ℝ := 120 - 50 * t
def position_B : ℝ := 220 - 50 * t
def position_N : ℝ := 30 * t - 30
def position_M : ℝ := 30 * t + 10

theorem overlap_32 :
  (∃ t : ℝ, (30 * t + 10 - (120 - 50 * t) = 32) ∨ 
            (-50 * t + 220 - (30 * t - 30) = 32)) ↔
  (t = 71 / 40 ∨ t = 109 / 40) :=
sorry
end

end overlap_32_l357_357138


namespace find_a_l357_357326

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l357_357326


namespace min_value_of_quadratic_l357_357226

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l357_357226


namespace num_subsets_containing_6_l357_357825

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l357_357825


namespace captain_age_is_25_l357_357040

noncomputable def captain_age (team_size : ℕ) (captain_age : ℕ) (keeper_age : ℕ) (average_team_age : ℕ)
  (total_team_age : ℤ) (average_remaining_age : ℕ) (total_remaining_age : ℤ) (combined_age : ℤ) : Prop :=
  captain_age = 25

theorem captain_age_is_25: 
  ∀ (C W : ℕ) (team_size : ℕ)
  (average_team_age : ℕ) (total_team_age : ℤ)
  (average_remaining_age : ℕ) (total_remaining_age : ℤ)
  (combined_age : ℤ),
  team_size = 11 →
  W = C + 3 →
  average_team_age = 22 →
  total_team_age = (11 : ℤ) * 22 →
  average_remaining_age = 21 →
  total_remaining_age = (9 : ℤ) * 21 →
  combined_age = 242 - 189 →
  combined_age = C + W →
  captain_age 11 C W 22 242 21 189 53 :=
by
  intros C W team_size average_team_age total_team_age average_remaining_age total_remaining_age combined_age 
  assume h_team_size
  assume h_keeper_age
  assume h_avg_team_age
  assume h_total_team_age
  assume h_avg_remaining_age
  assume h_total_remaining_age
  assume h_combined_age_sub
  assume h_combined_age_eq
  sorry

end captain_age_is_25_l357_357040


namespace length_of_PR_l357_357475

theorem length_of_PR (x y : ℝ) (h₁ : x^2 + y^2 = 250) : 
  ∃ PR : ℝ, PR = 10 * Real.sqrt 5 :=
by
  use Real.sqrt (2 * (x^2 + y^2))
  sorry

end length_of_PR_l357_357475


namespace smallest_domain_count_l357_357334

noncomputable def f : ℕ → ℕ
| n := if n = 15 then 22 
       else if n % 2 = 0 then n / 2 
       else 3 * n + 1

theorem smallest_domain_count : 
  {n ∈ ℕ | ∃ m, f m = n}.card = 15 :=
sorry

end smallest_domain_count_l357_357334


namespace perpendicular_AM_BC_l357_357777

open EuclideanGeometry

variables {A B C D E F G M : Point}

theorem perpendicular_AM_BC (h_triangle : triangle A B C)
  (h_bc_diameter : is_diameter_on BC (circumcircle B C)) 
  (h_semicircle_intersect : semicircle_intersects_with_diameter BC (line_through A D) (line_through A E)) 
  (h_perpendicular_DF_BC : perpendicular_from_point_to_line D BC F) 
  (h_perpendicular_EG_BC : perpendicular_from_point_to_line E BC G) 
  (h_DM_int_EF : line_through D G ∩ line_through E F = {M}) :
  perpendicular A BC :=
sorry

end perpendicular_AM_BC_l357_357777


namespace exists_line_intersecting_at_least_four_circles_l357_357894

-- Given conditions
variable (square_side_len : ℝ) (circumferences : ℝ → Prop)
-- The side length of the square is 1
def square_has_side_one (side : ℝ) : Prop := side = 1
-- There are several circles with total circumference equals 10
def total_circumference_is_ten (P : ℝ → Prop) : Prop := 
∃ (circle : ℝ → ℝ), (∀ r, P r → circle r) ∧ (P 10)

-- Main theorem statement
theorem exists_line_intersecting_at_least_four_circles (square_side_len : ℝ) (circumferences : ℝ → Prop)
  (h1 : square_has_side_one square_side_len) 
  (h2 : total_circumference_is_ten circumferences) : 
  ∃ (l : ℝ → ℝ), (∃ (circles : ℕ) (circumference : ℝ), circumferences circumference ∧ circles ≥ 4) :=
sorry

end exists_line_intersecting_at_least_four_circles_l357_357894


namespace bisecting_line_range_mn_l357_357033

theorem bisecting_line_range_mn
  (m n : ℝ)
  (H : ∀ x y : ℝ, mx + 2 * n * y - 4 = 0)
  (C : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 4 = 0 ∧ m + n = 2):
  -∞ < m * n ∧ m * n ≤ 1 :=
sorry

end bisecting_line_range_mn_l357_357033


namespace value_of_a_l357_357324

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l357_357324


namespace quadrilateral_probability_l357_357536

-- Define a list containing the lengths of the ten sticks.
def sticks : List ℕ := [1, 4, 5, 6, 8, 10, 12, 13, 14, 15]

-- Helper function to check if a set of four sticks can form a quadrilateral.
def canFormQuadrilateral (a b c d : ℕ) : Bool :=
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

-- List all combinations of four lengths that satisfy the quadrilateral condition.
def validQuadruples : List (ℕ × ℕ × ℕ × ℕ) :=
  [(4, 6, 8, 10), (4, 6, 10, 12), (4, 6, 12, 13), (4, 8, 10, 12), (4, 10, 12, 13),
   (4, 10, 12, 14), (4, 10, 13, 14), (4, 12, 13, 14), (5, 6, 8, 10), (5, 6, 10, 13),
   (5, 8, 10, 13), (5, 10, 13, 15), (6, 8, 10, 12), (6, 10, 12, 13), (6, 10, 12, 14),
   (6, 12, 13, 15), (8, 10, 12, 13), (8, 10, 12, 14), (8, 10, 13, 15), (10, 12, 13, 15), (12, 13, 14, 15)]

-- Define the main theorem that states the probability of forming a quadrilateral
-- with four randomly chosen sticks is 1/10.
theorem quadrilateral_probability :
  let chosen_sets := (sticks.combinations 4).filter (λ s, canFormQuadrilateral s[0] s[1] s[2] s[3])
  (List.length chosen_sets) / (List.length (sticks.combinations 4)) = 1 / 10 := by
    sorry

end quadrilateral_probability_l357_357536


namespace part_a_part_b_case1_part_b_case2_l357_357051

theorem part_a (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x1 / x2 + x2 / x1 = -9 / 4) : 
  p = -1 / 23 :=
sorry

theorem part_b_case1 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -3 / 8 :=
sorry

theorem part_b_case2 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -15 / 8 :=
sorry

end part_a_part_b_case1_part_b_case2_l357_357051


namespace triangles_congruence_l357_357895

theorem triangles_congruence (A_1 B_1 C_1 A_2 B_2 C_2 : ℝ)
  (angle_A1 angle_B1 angle_C1 angle_A2 angle_B2 angle_C2 : ℝ)
  (h_side1 : A_1 = A_2) 
  (h_side2 : B_1 = B_2)
  (h_angle1 : angle_A1 = angle_A2)
  (h_angle2 : angle_B1 = angle_B2)
  (h_angle3 : angle_C1 = angle_C2) : 
  ¬((A_1 = C_1) ∧ (B_1 = C_2) ∧ (angle_A1 = angle_B2) ∧ (angle_B1 = angle_A2) ∧ (angle_C1 = angle_B2) → 
     (A_1 = A_2) ∧ (B_1 = B_2) ∧ (C_1 = C_2)) :=
by {
  sorry
}

end triangles_congruence_l357_357895


namespace total_dots_not_visible_l357_357367

def total_dots_five_dice := 5 * 21
def visible_faces := [1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 4, 5, 6]
def visible_dots := visible_faces.sum
def not_visible_dots := total_dots_five_dice - visible_dots

theorem total_dots_not_visible : not_visible_dots = 49 :=
by 
  have h1 : total_dots_five_dice = 105 := rfl
  have h2 : visible_dots = 56 := rfl
  have h3 : not_visible_dots = 49 := by simp [total_dots_five_dice, visible_dots]
  exact h3 

end total_dots_not_visible_l357_357367


namespace greatest_three_digit_number_l357_357602

theorem greatest_three_digit_number 
  (n : ℕ)
  (h1 : n % 7 = 2)
  (h2 : n % 6 = 4)
  (h3 : n ≥ 100)
  (h4 : n < 1000) :
  n = 994 :=
sorry

end greatest_three_digit_number_l357_357602


namespace largest_angle_in_pentagon_l357_357881

def pentagon_angle_sum : ℝ := 540

def angle_A : ℝ := 70
def angle_B : ℝ := 90
def angle_C (x : ℝ) : ℝ := x
def angle_D (x : ℝ) : ℝ := x
def angle_E (x : ℝ) : ℝ := 3 * x - 10

theorem largest_angle_in_pentagon
  (x : ℝ)
  (h_sum : angle_A + angle_B + angle_C x + angle_D x + angle_E x = pentagon_angle_sum) :
  angle_E x = 224 :=
sorry

end largest_angle_in_pentagon_l357_357881


namespace part1_general_formula_part2_find_d_l357_357089

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357089


namespace find_x_l357_357133

variable (t b c : ℝ) 

-- Conditions
def avg1 := (t + b + c + 14 + 15) / 5 = 12
def prop := t = 2 * b

-- Target value to prove
def avg2 : ℝ := (t + b + c + 29) / 4 

theorem find_x (h_avg1 : avg1 t b c) (h_prop: prop t b) : avg2 t b c = 15 :=
by {
  sorry
}

end find_x_l357_357133


namespace dihedral_angle_planes_l357_357192

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

noncomputable def dot (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def cos_angle (v w : ℝ × ℝ × ℝ) : ℝ :=
  dot v w / (norm v * norm w)

theorem dihedral_angle_planes :
  let m := (1, 0, -1)
  let n := (0, -1, 1)
  (real.arccos (cos_angle m n) = 2 * real.pi / 3) ∨
  (real.arccos (cos_angle m n) = real.pi / 3) :=
by
  let m := (1, 0, -1)
  let n := (0, -1, 1)
  sorry

end dihedral_angle_planes_l357_357192


namespace probability_of_forming_triangle_l357_357445

def lengths : List ℕ := [2, 4, 6, 8, 12, 14, 18]

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_combinations : List (ℕ × ℕ × ℕ) :=
  ((2,4,6), (2,4,8), (2,4,12), (2,4,14), (2,4,18), (2,6,8), (2,6,12),
   (2,6,14), (2,6,18), (2,8,12), (2,8,14), (2,8,18), (2,12,14),
   (2,12,18), (2,14,18), (4,6,8), (4,6,12), (4,6,14), (4,6,18), (4,8,12),
   (4,8,14), (4,8,18), (4,12,14), (4,12,18), (4,14,18), (6,8,12),
   (6,8,14), (6,8,18), (6,12,14), (6,12,18), (6,14,18), (8,12,14),
   (8,12,18), (8,14,18), (12,14,18))

theorem probability_of_forming_triangle : ({
  comb : List ℕ × List ℕ × List ℕ
  | comb ∈ valid_combinations
  | can_form_triangle comb.fst comb.snd comb.trd
}).card.toRat / (lengths.combination 3).card.toRat = 8 / 35 :=
by {
  sorry
}

end probability_of_forming_triangle_l357_357445


namespace a_4_eq_20_l357_357810

def sequence (n : ℕ) : ℕ := n^2 + n

theorem a_4_eq_20 : sequence 4 = 20 :=
by
  sorry

end a_4_eq_20_l357_357810


namespace find_smallest_n_l357_357748

noncomputable def smallest_n (f : ℤ[x]) (n : ℕ) : ℕ :=
  if ∃ P Q : ℤ[x], (x + 1) * (x + 1)^n - 1 = (x^2 + 1) * P + 3 * Q 
  then n else smallest_n f (n + 1)

theorem find_smallest_n :
  smallest_n ((x + 1) ^ 8 - 1) 1 = 8 :=
sorry

end find_smallest_n_l357_357748


namespace triangle_centroid_GP_length_l357_357481

noncomputable def length_GP (AB AC BC : ℝ) (G P : ℝ) : ℝ :=
  (1 / 3) * ((real.sqrt (19 * 189)) / 8)

theorem triangle_centroid_GP_length 
  (AB AC BC : ℝ) (hAB hAC hBC : 0 < AB ∧ 0 < AC ∧ 0 < BC) (s : ℝ := (AB + AC + BC) / 2)
  (h_s : s = 19)
  (K : ℝ := real.sqrt (s * (s - AB) * (s - AC) * (s - BC))) 
  (h_area : K = real.sqrt (19 * 189))
  (h_altitude : (2 * K) / BC = (real.sqrt (19 * 189)) / 8)
  : GP = (real.sqrt 3591) / 24 :=
begin
  sorry
end

end triangle_centroid_GP_length_l357_357481


namespace find_a_l357_357327

def g (x : ℝ) : ℝ := 5 * x - 7

theorem find_a (a : ℝ) : g a = 0 → a = 7 / 5 := by
  intro h
  sorry

end find_a_l357_357327


namespace no_fib_in_a_l357_357907

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n + 1) + fib n

-- Define the sequence (a_k)
def a : ℕ → ℕ
| 0     := 2018
| (k+1) := a k + fib ((List.range (a k)).filter (λ n => fib n < a k)).last!

-- Prove that no term in the sequence (a_k) is a Fibonacci number
theorem no_fib_in_a : ∀ k : ℕ, ∀ m : ℕ, a k ≠ fib m :=
by
  intros k m
  sorry

end no_fib_in_a_l357_357907


namespace subsets_containing_six_l357_357829

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l357_357829


namespace rational_coordinates_of_circumcenter_l357_357550

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l357_357550


namespace functional_equation_solution_l357_357738

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l357_357738


namespace min_distinct_integers_for_ap_and_gp_l357_357606

theorem min_distinct_integers_for_ap_and_gp (n : ℕ) :
  (∀ (b q a d : ℤ), b ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 →
    (∃ (i : ℕ), i < 5 → b * (q ^ i) = a + i * d) ∧ 
    (∃ (j : ℕ), j < 5 → b * (q ^ j) ≠ a + j * d) ↔ n ≥ 6) :=
by {
  sorry
}

end min_distinct_integers_for_ap_and_gp_l357_357606


namespace range_of_a_for_monotonicity_l357_357441

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) := 
  ∀ x y ∈ I, x < y → f x ≤ f y

theorem range_of_a_for_monotonicity :
  ∀ a : ℝ, (∀ x > -1, x ∈ set.Ioi (-1) → 
  is_monotonically_increasing (λ x, (x-5) / (x-a-2)) (set.Ioi (-1))) → a ∈ set.Iic (-3) :=
begin
  sorry
end

end range_of_a_for_monotonicity_l357_357441


namespace solve_angle_EDF_l357_357891

variables {A B C D E F : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables {a b c d e f : A} -- points representing vertices and midpoints of the triangle

-- Conditions
def is_isosceles_triangle (A B C : A) : Prop :=
  dist A B = dist A C

def angle_at_A (A B C : A) : Prop :=
  ∠ A B C = 100

def is_midpoint (D E F : A) (B C A : A) : Prop :=
  dist B D = dist D C ∧ dist A E = dist E C ∧ dist A F = dist F B
  
def points_condition (C E D B F : A) : Prop :=
  dist C E = dist C D ∧ dist B F = dist B D

theorem solve_angle_EDF 
  (A B C D E F : A) 
  (h1 : is_isosceles_triangle A B C) 
  (h2 : angle_at_A A B C) 
  (h3 : is_midpoint D E F B C A) 
  (h4 : points_condition C E D B F)
  : ∠ E D F = 40 := 
sorry

end solve_angle_EDF_l357_357891


namespace money_sum_l357_357680

theorem money_sum (A B C : ℕ) (h1 : A + C = 300) (h2 : B + C = 600) (h3 : C = 200) : A + B + C = 700 :=
by
  sorry

end money_sum_l357_357680


namespace pappus_collinear_l357_357159

open Set

variables {P : Type*} [projective_geometry P]

-- Defining the points A, B, C on line l and points A1, B1, C1 on line l1
variables {A B C A1 B1 C1 : P}
variables {l l1 : Line P}

-- Conditions: Points A, B, C lie on line l and points A1, B1, C1 lie on line l1
axiom point_on_line_l : A ∈ l ∧ B ∈ l ∧ C ∈ l
axiom point_on_line_l1 : A1 ∈ l1 ∧ B1 ∈ l1 ∧ C1 ∈ l1

-- Defining the intersection points P, Q, R
def P : P := line_of_points A B1 ∩ line_of_points B A1
def Q : P := line_of_points B C1 ∩ line_of_points C B1
def R : P := line_of_points C A1 ∩ line_of_points A C1

-- Proving the intersection points are collinear
theorem pappus_collinear : collinear ({P, Q, R} : set P) :=
sorry

end pappus_collinear_l357_357159


namespace subsets_containing_six_l357_357830

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l357_357830


namespace circumcircle_area_l357_357482

theorem circumcircle_area (a b c A B C : ℝ) (h : a * Real.cos B + b * Real.cos A = 4 * Real.sin C) :
    π * (2 : ℝ) ^ 2 = 4 * π :=
by
  sorry

end circumcircle_area_l357_357482


namespace probability_hhh_before_hth_l357_357222

theorem probability_hhh_before_hth : 
  let coin_prob : ℚ := 1 / 2 in 
  let hhh_before_hth_prob : ℚ := 2 / 5 in 
  (probability_of_event_hhh_before_hth coin_prob) = hhh_before_hth_prob :=
sorry

end probability_hhh_before_hth_l357_357222


namespace find_cost_price_l357_357279

-- Defining the cost price as a variable
variable (x : ℝ)

-- The conditions based on the problem statement
def selling_price_before_discount := 1.5 * x
def discount := 20
def profit := 0.4 * x
def selling_price_after_discount := selling_price_before_discount - discount

-- The equation based on given conditions
def equation := selling_price_after_discount - x = profit

-- The theorem we want to prove
theorem find_cost_price (h : equation x) : x = 200 := 
by
  sorry

end find_cost_price_l357_357279


namespace perpendicular_sufficient_but_not_necessary_l357_357391

theorem perpendicular_sufficient_but_not_necessary (m : ℝ) :
  (m = -1 → (mx + (2m - 1)y + 1 = 0) ⊥ (3x + my + 3 = 0)) ∧ 
  ((mx + (2m - 1)y + 1 = 0) ⊥ (3x + my + 3 = 0) → m ≠ -1 → (m = 0 ∨ m = -1)) :=
sorry

end perpendicular_sufficient_but_not_necessary_l357_357391


namespace shortest_distance_l357_357284

-- Let distance formula between two points (x1, y1) and (x2, y2) be defined as follows
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the initial conditions and the final total distance
def cowboy_travel_distance : ℝ :=
  let C := (0, 0) -- initial position of the cowboy
  let stream_y := 6 -- stream is at y = 6
  let cabin := (12, -10) -- position of the cabin
  let firewood_collect := (5, 6) -- 5 miles downstream from the stream intersection
  let C_to_stream := distance 0 0 0 stream_y
  let stream_to_firewood := distance 0 stream_y 5 stream_y
  let firewood_to_cabin := distance 5 stream_y 12 (-10)
  in C_to_stream + stream_to_firewood + firewood_to_cabin

theorem shortest_distance :
  cowboy_travel_distance = 11 + sqrt 305 := by
  sorry

end shortest_distance_l357_357284


namespace projection_ratio_l357_357913

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

noncomputable def projection (u v : ℝ) : ℝ :=
(⟪u, v⟫ / ⟪v, v⟫) • v

theorem projection_ratio 
  (v w : ℝ) (hv : v ≠ 0) (hw : w ≠ 0) 
  (h : ∥projection v w∥ / ∥v∥ = 3/4) : 
  (∥projection (projection v w) v∥ / ∥v∥ = 9/16) :=
by sorry

end projection_ratio_l357_357913


namespace part_a_part_b_l357_357593

-- Conditions
variables (M N : Point) (A B C : Person)
variables (distance : ℝ) (walkSpeed bikeSpeed : ℝ)

-- Given values
def conditions := 
  (distance = 15) ∧ 
  (walkSpeed = 6) ∧ 
  (bikeSpeed = 15) ∧ 
  (A.startsWalk) ∧ 
  (B.startsBike) ∧ 
  (C.startsWalk) ∧ 
  (C.walksFromTo N M)

-- Prove part (a)
theorem part_a (h : conditions M N A B C distance walkSpeed bikeSpeed) : 
  ∃ t : ℝ, C.leaveTime = t ∧ t = 3/11 :=
sorry

-- Prove part (b)
theorem part_b (h : conditions M N A B C distance walkSpeed bikeSpeed) : 
  ∃ (x y : ℝ), A.walkDistance = x ∧ B.walkDistance = y ∧ 
  (x = 60/11) ∧ (y = 60/11) :=
sorry

end part_a_part_b_l357_357593


namespace part1_part2_l357_357117

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357117


namespace min_area_after_processing_l357_357659

-- Define the reported dimensions of the rectangular sheet
def reported_length : ℝ := 6
def reported_width : ℝ := 4

-- Define the error margin
def error_margin : ℝ := 1.0

-- Define the shrinkage factor due to the cooling process
def shrinkage_factor : ℝ := 0.9

-- Define the minimum possible length and width before shrinkage
def min_length : ℝ := reported_length - error_margin
def min_width : ℝ := reported_width - error_margin

-- Define the minimum possible length and width after shrinkage
def processed_length : ℝ := min_length * shrinkage_factor
def processed_width : ℝ := min_width * shrinkage_factor

-- Define the minimum possible area after processing
def min_processed_area : ℝ := processed_length * processed_width

-- Prove the minimum possible area of the rectangular sheet after processing
theorem min_area_after_processing : min_processed_area = 12.15 := by
  rw [reported_length, reported_width, error_margin, shrinkage_factor]
  simp [min_length, min_width, processed_length, processed_width, min_processed_area]
  norm_num
  sorry

end min_area_after_processing_l357_357659


namespace sum_in_each_row_leq_n_plus_1_div_4_l357_357130

theorem sum_in_each_row_leq_n_plus_1_div_4 (n : ℕ) (h : 0 < n)
  (a b : fin n → ℝ) (pos_a : ∀ i, 0 < a i) (pos_b : ∀ i, 0 < b i)
  (sum_a_b : ∀ i, a i + b i = 1)
  : ∃ (S : fin n → ℝ), (∀ i, S i = a i ∨ S i = b i) ∧ (∀ i, S i ≤ 1) ∧ (∑ i, S i) ≤ (↑(n) + 1) / 4 :=
begin
  sorry
end

end sum_in_each_row_leq_n_plus_1_div_4_l357_357130


namespace smallest_ducks_observed_l357_357148

variables (d c h : ℕ)

-- conditions
def ducks_in_flocks : ℕ := 13 * d
def cranes_in_flocks : ℕ := 17 * c
def herons_in_flocks : ℕ := 11 * h
def ratio_ducks_cranes_to_herons : Prop := 15 * herons_in_flocks = 11 * (ducks_in_flocks + cranes_in_flocks)
def ratio_ducks_to_cranes : Prop := 5 * c = 3 * d

theorem smallest_ducks_observed (h₁ : ratio_ducks_cranes_to_herons) (h₂ : ratio_ducks_to_cranes) : (d = 75) :=
sorry

end smallest_ducks_observed_l357_357148


namespace complement_union_result_l357_357852

open Set

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union_result :
    U = { x | x < 6 } →
    A = {1, 2, 3} → 
    B = {2, 4, 5} → 
    (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} :=
by
    intros hU hA hB
    sorry

end complement_union_result_l357_357852


namespace verify_a_l357_357329

def g (x : ℝ) : ℝ := 5 * x - 7

theorem verify_a (a : ℝ) : g a = 0 ↔ a = 7 / 5 := by
  sorry

end verify_a_l357_357329


namespace smallest_number_of_students_l357_357685

variables (n : ℕ)
def scores : ℕ → ℕ := λ i, if i < 7 then 150 else 90

theorem smallest_number_of_students :
  (∀ i < 7, scores i = 150) ∧
  (∀ i ≥ 7, scores i ≥ 90) ∧
  (∑ i in finset.range n, scores i) / n = 120 →
  n = 14 :=
by sorry

end smallest_number_of_students_l357_357685


namespace max_value_of_min_fx_gx_eq_l357_357786

noncomputable def max_min_f_g (f g : ℝ → ℝ) : ℝ :=
  let min_f_g := λ x, min (f x) (g x)
  (real.Sup (set_of (λ x, min_f_g x)))

theorem max_value_of_min_fx_gx_eq :
  (∀ x : ℝ, f x + g x = 2 * x / (x^2 + 8)) →
  max_min_f_g f g = sqrt 2 / 8 :=
by
  intros hfg
  sorry

end max_value_of_min_fx_gx_eq_l357_357786


namespace ram_leela_money_next_week_l357_357165

theorem ram_leela_money_next_week (x : ℕ)
  (initial_money : ℕ := 100)
  (total_money_after_52_weeks : ℕ := 1478)
  (sum_of_series : ℕ := 1378) :
  let n := 52
  let a1 := x
  let an := x + 51
  let S := (n / 2) * (a1 + an)
  initial_money + S = total_money_after_52_weeks → x = 1 :=
by
  sorry

end ram_leela_money_next_week_l357_357165


namespace dog_ate_cost_l357_357491

namespace CakeProblem

-- Define costs of ingredients
def flour_cost := 2.5 * 3.20
def sugar_cost := 1.5 * 2.10
def butter_cost := 0.75 * 5.50
def eggs_cost := 4 * 0.45
def baking_soda_cost := 0.60
def baking_powder_cost := 1.3
def salt_cost := 0.35
def vanilla_extract_cost := 1.75
def milk_cost := 1.25 * 1.40
def vegetable_oil_cost := 0.75 * 2.10

-- Define total cost of cake ingredients
def total_cost := flour_cost + sugar_cost + butter_cost + eggs_cost + baking_soda_cost + baking_powder_cost + salt_cost + vanilla_extract_cost + milk_cost + vegetable_oil_cost

-- Define sales tax rate and calculate sales tax
def sales_tax_rate := 0.07
def sales_tax := total_cost * sales_tax_rate

-- Define total cost including sales tax
def total_cost_with_tax := total_cost + sales_tax

-- Define number of slices and cost per slice
def slices := 12
def cost_per_slice := total_cost_with_tax / slices

-- Define the number of slices eaten by Laura's mother and calculate the remaining slices
def slices_eaten_by_mother := 4
def remaining_slices := slices - slices_eaten_by_mother

-- Calculate the cost of the slices the dog ate
def cost_of_dog_slices := cost_per_slice * remaining_slices

-- The theorem to be proved
theorem dog_ate_cost : cost_of_dog_slices = 17.44 := by
  sorry

end CakeProblem

end dog_ate_cost_l357_357491


namespace find_n_l357_357361

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.sin (n * Real.pi / 180) = Real.sin (782 * Real.pi / 180)) :
  n = 62 ∨ n = -62 := 
sorry

end find_n_l357_357361


namespace circumcenter_rational_l357_357542

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l357_357542


namespace positive_difference_between_A_and_B_l357_357711

-- Define the expressions A and B
def A := ∑ i in list.range' 1 20, (2 * i) * (2 * i + 1) + 40
def B := ∑ i in list.range' 1 19, (2 * i + 1) * (2 * i + 2)

-- The theorem stating the positive difference between A and B
theorem positive_difference_between_A_and_B : |A - B| = 1159 := by
  sorry

end positive_difference_between_A_and_B_l357_357711


namespace a_n_formula_d_value_l357_357105

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357105


namespace length_median_eq_3_l357_357815

noncomputable def A : (ℝ × ℝ × ℝ) := (3, 3, 2)
noncomputable def B : (ℝ × ℝ × ℝ) := (4, -3, 7)
noncomputable def C : (ℝ × ℝ × ℝ) := (0, 5, 1)

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

theorem length_median_eq_3 :
  distance A (midpoint B C) = 3 :=
by
  sorry

end length_median_eq_3_l357_357815


namespace percent_grape_juice_in_remaining_mixture_l357_357898

-- Define the given conditions
def container1_volume : ℝ := 40
def container1_percent_grape_juice : ℝ := 0.10

def container2_volume : ℝ := 30
def container2_percent_grape_juice : ℝ := 0.35

def additional_pure_grape_juice : ℝ := 20

def evaporation_percent : ℝ := 0.10

-- Calculate initial amount of grape juice in each container
def grape_juice_container1 : ℝ := container1_volume * container1_percent_grape_juice
def grape_juice_container2 : ℝ := container2_volume * container2_percent_grape_juice

-- Calculate total grape juice before evaporation
def total_grape_juice_before_evaporation : ℝ := grape_juice_container1 + grape_juice_container2 + additional_pure_grape_juice

-- Calculate total volume before evaporation
def total_volume_before_evaporation : ℝ := container1_volume + container2_volume + additional_pure_grape_juice

-- Calculate evaporated volume
def evaporated_volume : ℝ := total_volume_before_evaporation * evaporation_percent

-- Calculate remaining volume after evaporation
def remaining_volume : ℝ := total_volume_before_evaporation - evaporated_volume

-- Prove the percentage of grape juice in the remaining mixture
theorem percent_grape_juice_in_remaining_mixture :
  (total_grape_juice_before_evaporation / remaining_volume) * 100 ≈ 42.59 :=
by
  sorry

end percent_grape_juice_in_remaining_mixture_l357_357898


namespace nth_equation_sum_of_products_specific_sum_l357_357960

theorem nth_equation (n : ℕ) (h1 : 1 * 2 = (1 / 3) * (1 * 2 * 3 - 0 * 1 * 2))
    (h2 : 2 * 3 = (1 / 3) * (2 * 3 * 4 - 1 * 2 * 3))
    (h3 : 3 * 4 = (1 / 3) * (3 * 4 * 5 - 2 * 3 * 4)) :
  n * (n + 1) = (1 / 3) * (n * (n + 1) * (n + 2) - (n - 1) * n * (n + 1)) := 
sorry

theorem sum_of_products (n : ℕ) :
  (∑ i in finset.range n, i * (i + 1)) = (n * (n + 1) * (n + 2)) / 3 := 
sorry

theorem specific_sum :
  (∑ i in finset.range 17, (i + 1) * (i + 2) * (i + 3)) = 29070 := 
sorry

end nth_equation_sum_of_products_specific_sum_l357_357960


namespace Gunther_typing_time_l357_357819

theorem Gunther_typing_time:
  ∀ (W: ℕ) (M: ℕ), W = 25600 ∧ (M = 3 ∧ W = 160 * (x / 3)) → x = 480 :=
by 
  intros W M h1 h2
  sorry

end Gunther_typing_time_l357_357819


namespace permutation_inequality_l357_357948

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧
                   (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
                   (z = a ∨ z = b ∨ z = c ∨ z = d) ∧
                   (w = a ∨ w = b ∨ w = c ∨ w = d) ∧
                   (list.pairwise (≠) [x, y, z, w]) ∧ -- Ensure they are distinct
                   (2 * (x * z + y * w)^2 > (x^2 + y^2) * (z^2 + w^2)) :=
sorry

end permutation_inequality_l357_357948


namespace judy_caught_is_1_l357_357695

-- Defining the constants for the fish caught by each family member
constants (ben_caught billy_caught jim_caught susie_caught too_small num_filets : ℕ)
constants (judy_caught total_fish_kept : ℕ)

-- Setting the values given in the problem
def ben_caught := 4
def billy_caught := 3
def jim_caught := 2
def susie_caught := 5
def too_small := 3
def num_filets := 24

-- Calculating total fish caught
def total_caught := ben_caught + billy_caught + jim_caught + susie_caught

-- The problem conditions translated
axiom fish_filets_relation : total_fish_kept = num_filets / 2
axiom fish_kept_relation : total_fish_kept = total_caught - too_small + judy_caught

-- The proof statement
theorem judy_caught_is_1 : judy_caught = 1 := by
  sorry

end judy_caught_is_1_l357_357695


namespace log_x_y_z_l357_357428

theorem log_x_y_z (x y z : ℝ) (hx1 : log (x * y^3 * z) = 2) (hx2 : log (x^2 * y * z^2) = 3) : 
  log (x * y * z) = 8 / 5 := 
by
  sorry

end log_x_y_z_l357_357428


namespace intersection_points_sin_cos_on_interval_l357_357193

noncomputable def count_intersections (f g : ℝ → ℝ) (I : set ℝ) : ℕ :=
  finset.card (finset.filter (λ x, f x = g x) (finset.Icc(0, 3 * real.pi)))

theorem intersection_points_sin_cos_on_interval : count_intersections (λ x, real.sin (2 * x)) real.cos (set.Icc 0 (3 * real.pi)) = 7 :=
by sorry

end intersection_points_sin_cos_on_interval_l357_357193


namespace concurrency_of_cevians_l357_357499

theorem concurrency_of_cevians
  (I : Point)
  (Γ : Circle)
  (A B C D E F : Point)
  (hIcenter : I_center_of_triangle I A B C)
  (hD : D ∈ Γ ∧ perpendicular (line_through I D) (line_through B C))
  (hE : E ∈ Γ ∧ perpendicular (line_through I E) (line_through C A))
  (hF : F ∈ Γ ∧ perpendicular (line_through I F) (line_through A B))
  : concurrent (line_through A D) (line_through B E) (line_through C F) :=
sorry

end concurrency_of_cevians_l357_357499


namespace original_surface_area_l357_357022

theorem original_surface_area (R : ℝ) (h : 2 * π * R^2 = 4 * π) : 4 * π * R^2 = 8 * π :=
by
  sorry

end original_surface_area_l357_357022


namespace greatest_A_value_l357_357220

-- Conditions Definitions
def is_valid_digit_set (A B C D E F : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  {A, B, C, D, E, F} ⊆ {1, 2, 3, 4, 5, 6}

def meets_conditions (A B C D E F : ℕ) : Prop :=
  (A % 1 = 0) ∧
  ((10 * A + B) % 2 = 0) ∧
  ((100 * A + 10 * B + C) % 3 = 0) ∧
  ((1000 * A + 100 * B + 10 * C + D) % 4 = 0) ∧
  ((10000 * A + 1000 * B + 100 * C + 10 * D + E) % 5 = 0) ∧
  ((100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F) % 6 = 0)

-- Concluding the theorem
theorem greatest_A_value : ∃ (A B C D E F : ℕ), 
  is_valid_digit_set A B C D E F ∧ meets_conditions A B C D E F ∧ A = 3 :=
sorry

end greatest_A_value_l357_357220


namespace traffic_sign_painting_cost_l357_357669

noncomputable def width_in_inches := 49
noncomputable def length_in_inches := 101
noncomputable def width_in_feet := width_in_inches / 12
noncomputable def length_in_feet := length_in_inches / 12
noncomputable def cost_per_square_foot := 2
noncomputable def area_in_square_feet := width_in_feet * length_in_feet
noncomputable def total_area := 2 * area_in_square_feet
noncomputable def painting_cost := total_area * cost_per_square_foot

theorem traffic_sign_painting_cost :
  width_in_feet = 49 / 12 ∧
  length_in_feet = 101 / 12 ∧
  painting_cost = 137.40 :=
by
  -- Proof is skipped with sorry
  sorry

end traffic_sign_painting_cost_l357_357669


namespace symmetric_about_y_axis_l357_357787

theorem symmetric_about_y_axis (m n : ℝ) (A B : ℝ × ℝ)
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (n + 1, 4))
  (symmetry : A.1 = -B.1)
  : m = 2.5 ∧ n = 2 :=
by
  sorry

end symmetric_about_y_axis_l357_357787


namespace sqrt_expression_equality_l357_357339

theorem sqrt_expression_equality :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * 5^(3/4) :=
by
  sorry

end sqrt_expression_equality_l357_357339


namespace prob_two_black_balls_l357_357642

-- Definitions based on conditions
def totalBalls : ℕ := 7 + 8
def initialBlackBalls : ℕ := 8
def ballsLeftAfterOneDraw : ℕ := 14
def blackBallsLeftAfterOneDraw : ℕ := 7

-- Probability of success
def probabilityFirstBlack : ℚ := initialBlackBalls / totalBalls
def probabilitySecondBlack : ℚ := blackBallsLeftAfterOneDraw / ballsLeftAfterOneDraw

-- Combined probability for two black balls without replacement
def combinedProbability : ℚ := probabilityFirstBlack * probabilitySecondBlack

theorem prob_two_black_balls : combinedProbability = 4 / 15 :=
by
  sorry

end prob_two_black_balls_l357_357642


namespace tomatoes_eaten_l357_357821

theorem tomatoes_eaten 
  (initial_tomatoes : ℕ) 
  (final_tomatoes : ℕ) 
  (half_given : ℕ) 
  (B : ℕ) 
  (h_initial : initial_tomatoes = 127) 
  (h_final : final_tomatoes = 54) 
  (h_half : half_given = final_tomatoes * 2) 
  (h_remaining : initial_tomatoes - half_given = B)
  : B = 19 := 
by
  sorry

end tomatoes_eaten_l357_357821


namespace sin_cos_identity_l357_357762

theorem sin_cos_identity (α : ℝ) (h : cos α - sin α = 1 / 2) : sin α * cos α = 3 / 8 :=
sorry

end sin_cos_identity_l357_357762


namespace eq_length_eq_fr_l357_357270

open EuclideanGeometry

variables {ω : Circle} {A B C D E F P Q R : Point}
variables (hcyclic : CyclicQuad ABCD ω)
variables (hperp : Perpendicular AC BD)
variables (hE : ReflectPointOverLine D BA E)
variables (hF : ReflectPointOverLine D BC F)
variables (hP : IntersectionPoint BD EF P)
variables (hEPD : CircleContainingVertices ω E P D Q)
variables (hFPD : CircleContainingVertices ω F P D R)

theorem eq_length_eq_fr {A B C D E F P Q R : Point}
  (hcyclic : CyclicQuad ABCD ω)
  (hperp : Perpendicular AC BD)
  (hE : ReflectPointOverLine D BA E)
  (hF : ReflectPointOverLine D BC F)
  (hP : IntersectionPoint BD EF P)
  (hEPD : CircleContainingVertices ω E P D Q)
  (hFPD : CircleContainingVertices ω F P D R) :
  distance E Q = distance F R :=
sorry

end eq_length_eq_fr_l357_357270


namespace part1_part2_l357_357115

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357115


namespace failed_students_percentage_l357_357956

theorem failed_students_percentage :
  let total_boys := 70
  let total_girls := 130
  let examined_boys := 39 -- 55% of 70 rounded
  let examined_girls := 59 -- 45% of 130 rounded
  let passing_boys := 19 -- 48% of examined boys rounded
  let passing_girls := 22 -- 38% of examined girls rounded
  let failed_boys := examined_boys - passing_boys
  let failed_girls := examined_girls - passing_girls
  let total_failed := failed_boys + failed_girls
  let total_students := total_boys + total_girls
  let failed_percentage := (total_failed / total_students.toReal) * 100
  failed_percentage = 28.5 :=
by
  -- conditions
  let total_boys := 70
  let total_girls := 130
  let examined_boys := 39 -- 55% of 70 rounded
  let examined_girls := 59 -- 45% of 130 rounded
  let passing_boys := 19 -- 48% of examined boys rounded
  let passing_girls := 22 -- 38% of examined girls rounded
  let failed_boys := examined_boys - passing_boys
  let failed_girls := examined_girls - passing_girls
  let total_failed := failed_boys + failed_girls
  let total_students := total_boys + total_girls
  let failed_percentage := (total_failed / total_students.toReal) * 100
  sorry

end failed_students_percentage_l357_357956


namespace find_value_of_expression_l357_357851

theorem find_value_of_expression
  (x y : ℝ)
  (h1 : 3 * x + 2 * y = 11)
  (h2 : y = 1) :
  5 * x + 3 = 18 :=
begin
  sorry
end

end find_value_of_expression_l357_357851


namespace sqrt_prime_geometric_progression_impossible_l357_357705

theorem sqrt_prime_geometric_progression_impossible {p1 p2 p3 : ℕ} (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) (hneq12 : p1 ≠ p2) (hneq23 : p2 ≠ p3) (hneq31 : p3 ≠ p1) :
  ¬ ∃ (a r : ℝ) (n1 n2 n3 : ℤ), (a * r^n1 = Real.sqrt p1) ∧ (a * r^n2 = Real.sqrt p2) ∧ (a * r^n3 = Real.sqrt p3) := sorry

end sqrt_prime_geometric_progression_impossible_l357_357705


namespace least_integer_12_l357_357605

def least_integer_m (m : ℕ) : Prop :=
  ∀ (S : Finset ℕ), (∀ x ∈ S, 1 ≤ x ∧ x ≤ 2023) → (card S ≥ m → ∃ a b ∈ S, 1 < (a : ℝ) / b ∧ (a : ℝ) / b ≤ 2)

theorem least_integer_12 : least_integer_m 12 :=
begin
  sorry
end

end least_integer_12_l357_357605


namespace bary_coords_eq_areas_l357_357528

-- Define the conditions in the Lean 4 statement:
variables {A B C X : Type} -- Define the points A, B, C, and X as types
variables (S_BCX S_CAX S_ABX S_ABC : ℝ) -- Define the areas as real numbers
variables (λ1 λ2 λ3 : ℝ) -- Define the barycentric coordinates as real numbers

-- State that X lies inside the triangle ABC:
axiom inside_triangle : X → A → B → C → Prop

-- State the sums of the barycentric coordinates:
axiom bary_coord_sum : λ1 + λ2 + λ3 = 1

-- Represent coordinates with areas:
noncomputable def barycentric_coordinates :=
  (λ1, λ2, λ3) = (S_BCX / S_ABC, S_CAX / S_ABC, S_ABX / S_ABC)

-- The statement of the proof
theorem bary_coords_eq_areas (X A B C : Type) (S_BCX S_CAX S_ABX S_ABC : ℝ)
  (H : inside_triangle X A B C)
  (H1 : λ1 + λ2 + λ3 = 1)
  (λ1 λ2 λ3 : ℝ):
  (λ1, λ2, λ3) = (S_BCX / S_ABC, S_CAX / S_ABC, S_ABX / S_ABC) :=
by sorry

end bary_coords_eq_areas_l357_357528


namespace part1_general_formula_part2_find_d_l357_357091

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357091


namespace find_a_39_l357_357575

noncomputable def a : ℕ → ℕ
| 0     := 0
| 1     := 3
| (n+1) := a n + n + 2

theorem find_a_39 : a 39 = 820 :=
by
  sorry

end find_a_39_l357_357575


namespace remainder_17_pow_49_mod_5_l357_357608

theorem remainder_17_pow_49_mod_5 : (17^49) % 5 = 2 :=
by
  sorry

end remainder_17_pow_49_mod_5_l357_357608


namespace rational_coordinates_of_circumcenter_l357_357552

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l357_357552


namespace nth_inequality_l357_357409

theorem nth_inequality (n : ℕ) : 1 + ∑ i in finset.range n, (1 / (2^i - 1)) > n := 
  sorry

end nth_inequality_l357_357409


namespace min_number_of_chips_l357_357468

theorem min_number_of_chips (strip_length : ℕ) (n : ℕ) : strip_length = 2100 → 
  (∀ i, (recorded number in cell i = |(number of chips left) - (number of chips right)|) ∧ recorded numbers are different and non-zero) → 
  n ≥ 1400 :=
by 
  assume strip_length_eq existence_of_chips
  -- Apply rest of the reasoning steps
  sorry

end min_number_of_chips_l357_357468


namespace jovana_added_shells_l357_357066

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h1 : initial_amount = 5) 
  (h2 : final_amount = 17) 
  (h3 : added_amount = final_amount - initial_amount) : 
  added_amount = 12 := 
by 
  -- Since the proof is not required, we add sorry here to skip the proof.
  sorry 

end jovana_added_shells_l357_357066


namespace min_segments_red_triangle_l357_357073

noncomputable def minRedSegments (n : ℕ) : ℕ :=
  (n - 1) * (n - 2) / 2

theorem min_segments_red_triangle (n : ℕ) (h : n ≥ 4)
  {P : Finset (Fin n)} (hP : ∀ p₁ p₂ p₃ ∈ P, p₁ ≠ p₂ ∧ p₂ ≠ p₃ ∧ p₁ ≠ p₃) :
  ∃ m, m = minRedSegments n :=
by
  sorry

end min_segments_red_triangle_l357_357073


namespace total_beads_sue_necklace_l357_357684

theorem total_beads_sue_necklace (purple blue green : ℕ) (h1 : purple = 7)
  (h2 : blue = 2 * purple) (h3 : green = blue + 11) : 
  purple + blue + green = 46 := 
by 
  sorry

end total_beads_sue_necklace_l357_357684


namespace find_cost_of_ticket_l357_357661

noncomputable def cost_of_ticket (T : ℝ) : Prop :=
  let P := T - 3
  let D := T - 2
  let C := (T - 2) / 2
  let S := T + P + D + C
  S = 22

theorem find_cost_of_ticket : ∃ T : ℝ, cost_of_ticket T ∧ T = 8 :=
begin
  use 8,
  unfold cost_of_ticket,
  simp,
  norm_num,
  sorry
end

end find_cost_of_ticket_l357_357661


namespace Q_one_is_one_l357_357212

noncomputable def Q : polynomial ℚ :=
  polynomial.monic_of_degree_eq (by norm_num) x^4 - 4 * x^2 + 4

theorem Q_one_is_one : Q.eval 1 = 1 := by
  have hQ := polynomial.eval_one,
  rw [←hQ],
  -- We would fill in the steps here to conclude the proof, but for now, we use sorry.
  sorry

end Q_one_is_one_l357_357212


namespace find_angle_A_find_area_l357_357485

-- Define the geometric and trigonometric conditions of the triangle
def triangle (A B C a b c : ℝ) :=
  a = 4 * Real.sqrt 3 ∧ b + c = 8 ∧
  2 * Real.sin A * Real.cos B + Real.sin B = 2 * Real.sin C

-- Prove angle A is 60 degrees
theorem find_angle_A (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : A = Real.pi / 3 := sorry

-- Prove the area of triangle ABC is 4 * sqrt(3) / 3
theorem find_area (A B C a b c : ℝ) 
  (h : triangle A B C a b c) : 
  (1 / 2) * (a * b * Real.sin C) = (4 * Real.sqrt 3) / 3 := sorry

end find_angle_A_find_area_l357_357485


namespace sum_of_two_cubes_lt_1000_l357_357008

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357008


namespace smallest_possible_area_of_ellipse_l357_357691

theorem smallest_possible_area_of_ellipse
  (a b : ℝ)
  (h_ellipse : ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → 
    (((x - 1/2)^2 + y^2 = 1/4) ∨ ((x + 1/2)^2 + y^2 = 1/4))) :
  ∃ (k : ℝ), (a * b * π = 4 * π) :=
by
  sorry

end smallest_possible_area_of_ellipse_l357_357691


namespace relationship_between_abc_l357_357768

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem relationship_between_abc : a > b ∧ b > c :=
by
  sorry

end relationship_between_abc_l357_357768


namespace part1_part2_l357_357116

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l357_357116


namespace angle_A_pi_over_2_iff_sinC_eq_sinA_cosB_l357_357480

theorem angle_A_pi_over_2_iff_sinC_eq_sinA_cosB (A B C : ℝ) (h_ABC : A + B + C = π) :
  A = π / 2 ↔ sin C = (sin A) * (cos B) :=
by
  sorry

end angle_A_pi_over_2_iff_sinC_eq_sinA_cosB_l357_357480


namespace find_p_q_l357_357753

noncomputable def f (p q : ℝ) (x : ℝ) : ℝ :=
if x < -1 then p * x + q else 5 * x - 10

theorem find_p_q (p q : ℝ) (h : ∀ x, f p q (f p q x) = x) : p + q = 11 :=
sorry

end find_p_q_l357_357753


namespace rational_coordinates_of_circumcenter_l357_357549

open Classical

noncomputable theory

theorem rational_coordinates_of_circumcenter
  {a1 b1 a2 b2 a3 b3 : ℚ}
  (h1 : ∃ (x y : ℚ), (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
                      (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :
  ∃ (x y : ℚ),
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 := 
begin
  obtain ⟨x, y, hx⟩ := h1,
  use [x, y],
  exact hx,
end

end rational_coordinates_of_circumcenter_l357_357549


namespace bahs_equivalent_to_1500_yahs_l357_357856

-- Definitions from conditions
def bahs := ℕ
def rahs := ℕ
def yahs := ℕ

-- Conversion ratios given in conditions
def ratio_bah_rah : ℚ := 10 / 16
def ratio_rah_yah : ℚ := 9 / 15

-- Given the conditions
def condition1 (b r : ℚ) : Prop := b / r = ratio_bah_rah
def condition2 (r y : ℚ) : Prop := r / y = ratio_rah_yah

-- Goal: proving the question
theorem bahs_equivalent_to_1500_yahs (b : ℚ) (r : ℚ) (y : ℚ)
  (h1 : condition1 b r) (h2 : condition2 r y) : b * (1500 / y) = 562.5
:=
sorry

end bahs_equivalent_to_1500_yahs_l357_357856


namespace max_ab_ac_bc_l357_357922

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end max_ab_ac_bc_l357_357922


namespace find_other_number_l357_357979

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l357_357979


namespace tom_savings_end_of_month_final_balance_l357_357589

theorem tom_savings_end_of_month_final_balance :
  let initial_balance := 12.0
  let week1_spend := 4.0
  let interest_rate := 0.02
  let week1_balance := (initial_balance - week1_spend) * (1 + interest_rate)
  let week2_withdraw := 8.0
  let week2_spend := 2.0
  let week2_balance := (week1_balance + (week2_withdraw - week2_spend)) * (1 + interest_rate)
  let week3_withdraw := 8.0
  let week3_earn := 5.0
  let week3_spend := 6.5
  let week3_balance := (week2_balance + ((week3_withdraw + week3_earn) - week3_spend)) * (1 + interest_rate)
  let penultimate_withdraw := 8.0
  let penultimate_spend := 3.0
  let penultimate_balance := (week3_balance + (penultimate_withdraw - penultimate_spend)) * (1 + interest_rate)
  penultimate_balance ≈ 26.89 := 
sorry

end tom_savings_end_of_month_final_balance_l357_357589


namespace avg_remaining_two_l357_357625

variables {A B C D E : ℝ}

-- Conditions
def avg_five (A B C D E : ℝ) : Prop := (A + B + C + D + E) / 5 = 10
def avg_three (A B C : ℝ) : Prop := (A + B + C) / 3 = 4

-- Theorem to prove
theorem avg_remaining_two (A B C D E : ℝ) (h1 : avg_five A B C D E) (h2 : avg_three A B C) : ((D + E) / 2) = 19 := 
sorry

end avg_remaining_two_l357_357625


namespace inscribed_circle_radius_l357_357463

theorem inscribed_circle_radius (a b c : ℝ) (R : ℝ) (r : ℝ) :
  a = 20 → b = 20 → d = 25 → r = 6 := 
by
  -- conditions of the problem
  sorry

end inscribed_circle_radius_l357_357463


namespace condition_on_p_for_q_l357_357393

theorem condition_on_p_for_q (p q : Prop) 
  (h1 : p → q) 
  (h2 : ¬(¬p → ¬q)) : 
  ((p → q) ∧ ¬(¬p → ¬q)) → (p → q) ∧ (¬(q → p)) :=
begin
  sorry
end

end condition_on_p_for_q_l357_357393


namespace min_value_of_z_l357_357234

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l357_357234


namespace operation_identity_l357_357372

variable (x y : ℝ) (z : ℝ)

def operation (x y : ℝ) : ℝ := x * y / (x + y)

def z_value (x y : ℝ) : ℝ := x + 2 * y

theorem operation_identity (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  operation x (z_value x y) = (x^2 + 2 * x * y) / (2 * (x + y)) :=
by
  sorry

end operation_identity_l357_357372


namespace f_value_at_3_l357_357770

theorem f_value_at_3 (a b : ℝ) (h : (a * (-3)^3 - b * (-3) + 2 = -1)) : a * (3)^3 - b * 3 + 2 = 5 :=
sorry

end f_value_at_3_l357_357770


namespace find_angle_A_and_side_a_l357_357868

variables {a b c : ℝ}
variables {A B C : ℝ} -- Angles of triangle ABC

-- Conditions
def is_triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π

def parallel_vectors (m n : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ m.1 = k * n.1 ∧ m.2 = k * n.2

def given_vectors (a b c A B C : ℝ) : Prop :=
  parallel_vectors (b, c - a) (sin C + sin A, sin C - sin B)

def sides_and_area_condition (b c : ℝ) (area : ℝ) : Prop :=
  b + c = 4 ∧ area = (3 * sqrt 3) / 4

-- Lean statement for the proof problem
theorem find_angle_A_and_side_a
  (a b c A B C : ℝ)
  (h_triangle : is_triangle_ABC a b c A B C)
  (h_vectors : given_vectors a b c A B C)
  (h_conditions : sides_and_area_condition b c ((3 * sqrt 3) / 4)) :
  A = π / 3 ∧ a = sqrt 7 :=
by
  sorry

end find_angle_A_and_side_a_l357_357868


namespace number_of_unique_sums_of_two_cubes_less_than_1000_l357_357000

def is_perfect_cube_sum (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3

theorem number_of_unique_sums_of_two_cubes_less_than_1000 : 
  (Finset.filter (λ n, n < 1000 ∧ is_perfect_cube_sum n) (Finset.range 1000)).card = 47 :=
sorry

end number_of_unique_sums_of_two_cubes_less_than_1000_l357_357000


namespace intersection_count_l357_357076

noncomputable def A : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, Real.cos (2 * x))}
noncomputable def B : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, x^2 + 1)}

theorem intersection_count : (A ∩ B).toFinset.card = 1 := by
  sorry

end intersection_count_l357_357076


namespace evaluate_f_at_three_over_two_l357_357947

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -4 * x^2 + 2
  else if 0 ≤ x ∧ x < 1 then x
  else f (x - 2)

theorem evaluate_f_at_three_over_two : f (3 / 2) = 1 :=
sorry

end evaluate_f_at_three_over_two_l357_357947


namespace subsets_containing_six_l357_357834

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l357_357834


namespace cost_of_360_songs_in_2005_l357_357449

theorem cost_of_360_songs_in_2005 :
  ∀ (c : ℕ), (200 * (c + 32) = 360 * c) → 360 * c / 100 = 144 :=
by
  assume c : ℕ
  assume h : 200 * (c + 32) = 360 * c
  sorry

end cost_of_360_songs_in_2005_l357_357449


namespace range_of_a_l357_357771

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 2

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ -1 → f a x ≥ a) : -3 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l357_357771


namespace find_other_number_l357_357980

open Nat

def gcd (a b : ℕ) : ℕ := if a = 0 then b else gcd (b % a) a
noncomputable def lcm (a b : ℕ) : ℕ := a * b / gcd a b

def a : ℕ := 210
def lcm_ab : ℕ := 4620
def gcd_ab : ℕ := 21

theorem find_other_number (b : ℕ) (h_lcm : lcm a b = lcm_ab) (h_gcd : gcd a b = gcd_ab) :
  b = 462 := by
  sorry

end find_other_number_l357_357980


namespace quadratic_completing_the_square_l357_357479

theorem quadratic_completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 → (x - 2)^2 = 6 :=
by sorry

end quadratic_completing_the_square_l357_357479


namespace part_a_l357_357256

theorem part_a : 
  ∃ (pairs : list (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ pairs → x.1 + x.2 ∈ 
      {p ∈ primes | p ∈ {5, 7, 11, 13, 19, 23}}) ∧
    pairs.length = 6 ∧ 
    (∀ (x y : ℕ × ℕ), x ≠ y → x.1 ≠ x.2 ∧ y.1 ≠ y.2) := 
sorry

end part_a_l357_357256


namespace inclination_angle_y_axis_l357_357568

/-- The inclination angle of a line corresponding to the y-axis is 90°. -/
theorem inclination_angle_y_axis : 
  ∀ (x_axis y_axis : ℝ), 
  (x_axis ⟂ y_axis) → 
  inclination_angle y_axis = 90 :=
by
  sorry

end inclination_angle_y_axis_l357_357568


namespace functional_equation_solution_l357_357737

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_equation_solution_l357_357737


namespace range_of_a_for_monotonicity_l357_357803

noncomputable def f (a x : ℝ) : ℝ := Real.logBase a (x^3 - a * x)

theorem range_of_a_for_monotonicity (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x1 x2 ∈ Ioo (-1 / 2) 0, x1 < x2 → f a x1 < f a x2) ↔ a ∈ Set.Ico (3 / 4) 1 :=
sorry

end range_of_a_for_monotonicity_l357_357803


namespace keegan_essay_time_l357_357489

def total_words : ℕ := 1200
def initial_rate : ℕ := 400
def subsequent_rate : ℕ := 200
def initial_period : ℕ := 2

theorem keegan_essay_time :
  let initial_words := initial_rate * initial_period,
      remaining_words := total_words - initial_words,
      subsequent_period := remaining_words / subsequent_rate
  in initial_period + subsequent_period = 4 := by
  sorry

end keegan_essay_time_l357_357489


namespace probability_each_player_has_one_after_2022_rings_l357_357683

-- Define the players
inductive Player
| Aiden
| Bianca
| Carlos
| Diana

namespace Player
  open Player

-- Define the initial state
def initial_state : Player → ℕ
| Aiden  => 1
| Bianca => 1
| Carlos => 2
| Diana  => 1

-- Define the probability calculation function
noncomputable def probability_after_2022 : ℚ :=
  15 / 81

-- Define the theorem to be proven
theorem probability_each_player_has_one_after_2022_rings :
  (∑ p : Player, initial_state p = 5) →
  probability_after_2022 = 15 / 81 :=
sorry

end probability_each_player_has_one_after_2022_rings_l357_357683


namespace problem_statement_l357_357128

open Nat

variable (n : ℕ) (a : ℕ → ℕ) (k : ℕ)

-- Conditions: Integers 1 ≤ a_1 < a_2 < ... < a_k ≤ n
def conditions (a : ℕ → ℕ) (k n : ℕ) : Prop :=
  (∀ i , i < k → 1 ≤ a i) ∧
  (∀ i j, i < j ∧ j < k → a i < a j) ∧
  (∀ i , i < k → a i ≤ n) ∧
  (∀ i j, i < j ∧ j < k → lcm (a i) (a j) > n)

-- Goal: ∑ (i : ℕ) in finset.range k, 1 / ↑(a i) < 3 / 2
def proof_statement (a : ℕ → ℕ) (k n : ℕ) [inhabited ℚ] : Prop :=
  (∑ i in finset.range k, 1 / (a i) : ℚ) < (3 / 2)

theorem problem_statement (a : ℕ → ℕ) (k n : ℕ) :
  conditions a k n → proof_statement a k n :=
sorry


end problem_statement_l357_357128


namespace vertical_asymptote_unique_d_values_l357_357336

theorem vertical_asymptote_unique_d_values (d : ℝ) :
  (∃! x : ℝ, ∃ c : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ (x^2 - 2*x + d) = 0) ↔ (d = 0 ∨ d = -3) := 
sorry

end vertical_asymptote_unique_d_values_l357_357336


namespace circle_range_of_a_l357_357794

theorem circle_range_of_a (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0) → a < 5 := by
  sorry

end circle_range_of_a_l357_357794


namespace max_ab_ac_bc_l357_357921

theorem max_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 6) : 
    ab + ac + bc <= 8 :=
sorry

end max_ab_ac_bc_l357_357921


namespace num_subsets_containing_6_l357_357826

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l357_357826


namespace find_f1_l357_357414

def f (x m n : ℝ) := 2^(x + m) + n

theorem find_f1 (h : f (-2) m n = 2) : f 1 m n = 9 :=
sorry

end find_f1_l357_357414


namespace smallest_integer_in_set_l357_357875

theorem smallest_integer_in_set :
  ∀ (n : ℤ), (n + 6 > 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → n = -1 :=
by
  intros n h
  sorry

end smallest_integer_in_set_l357_357875


namespace find_x_y_l357_357969

theorem find_x_y (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : y ≥ x) (h4 : x + y ≤ 20) 
  (h5 : ¬(∃ s, (x * y = s) → x + y = s ∧ ∃ a b : ℕ, a * b = s ∧ a ≠ x ∧ b ≠ y))
  (h6 : ∃ s_t, (x + y = s_t) → x * y = s_t):
  x = 2 ∧ y = 11 :=
by {
  sorry
}

end find_x_y_l357_357969


namespace square_in_ellipse_area_l357_357674

theorem square_in_ellipse_area :
  (∃ a b : ℝ, (a ≠ 0 ∧ b ≠ 0) ∧ (a = b) ∧ 
    (∀ (x y : ℝ), (x, y) = (a, b) →
    x*x / 5 + y*y / 10 = 1 ) ∧
    (∀ (u v : ℝ), (u v) = (b, a) →
    u*u / 5 + v*v / 10 = 1 )) →
  (∃ A : ℝ, A = 2 * a * 2 * a ∧ A = 40 / 3) :=
by
  sorry

end square_in_ellipse_area_l357_357674


namespace circumcenter_rational_l357_357543

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l357_357543


namespace min_value_of_z_l357_357235

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l357_357235


namespace moles_of_Cu_CN_2_is_1_l357_357746

def moles_of_HCN : Nat := 2
def moles_of_CuSO4 : Nat := 1
def moles_of_Cu_CN_2_formed (hcn : Nat) (cuso4 : Nat) : Nat :=
  if hcn = 2 ∧ cuso4 = 1 then 1 else 0

theorem moles_of_Cu_CN_2_is_1 : moles_of_Cu_CN_2_formed moles_of_HCN moles_of_CuSO4 = 1 :=
by
  sorry

end moles_of_Cu_CN_2_is_1_l357_357746


namespace PQ_PQ_l357_357215

noncomputable def point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Defining points according to the problem
noncomputable def P := point a b
noncomputable def Q := point c d
noncomputable def R := point e f
noncomputable def P' := point (-b) (-a)
noncomputable def Q' := point (-d) (-c)
noncomputable def R' := point (-f) (-e)

-- Conditions for points in the second quadrant
axiom a_lt_0 : a < 0
axiom b_gt_0 : b > 0
axiom c_lt_0 : c < 0
axiom d_gt_0 : d > 0
axiom e_lt_0 : e < 0
axiom f_gt_0 : f > 0

-- Lean statement to prove that Lines PQ and P'Q' are not necessarily perpendicular

theorem PQ_PQ'_not_perpendicular :
  ¬ ((d - b) / (c - a) * ((c + d) / (b + a)) = -1) :=
by
  sorry

end PQ_PQ_l357_357215


namespace probability_two_english_teachers_l357_357191

open Finset

theorem probability_two_english_teachers 
  (english teachers : Finset ℕ) 
  (h_teacher_card : teachers.card = 9) 
  (h_english_card : english.card = 3)
  (h_subset: english ⊆ teachers) :
  (english.choose 2).card.toNat.toRat / (teachers.choose 2).card.toNat.toRat = 1 / 12 := 
  sorry

end probability_two_english_teachers_l357_357191


namespace part1_general_formula_part2_find_d_l357_357092

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357092


namespace probability_at_least_one_multiple_of_4_correct_l357_357310

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  -- Total numbers in the range
  let total_numbers := 60 in
  -- Count of multiples of 4 in the range
  let multiples_of_4 := 15 in
  -- Count of non-multiples of 4 in the range
  let non_multiples_of_4 := total_numbers - multiples_of_4 in
  -- Total combinations of two numbers with the second greater than the first
  let total_combinations := (total_numbers * (total_numbers - 1)) / 2 in
  -- Combinations of two non-multiples of 4 with the second greater than the first
  let non_multiples_combinations := (non_multiples_of_4 * (non_multiples_of_4 - 1)) / 2 in
  -- Probability that both numbers are not multiples of 4
  let probability_both_non_multiples := (non_multiples_combinations : ℚ) / total_combinations in
  -- Probability that at least one is a multiple of 4
  (1 : ℚ) - probability_both_non_multiples

theorem probability_at_least_one_multiple_of_4_correct :
  probability_at_least_one_multiple_of_4 = 26 / 59 :=
by
  sorry

end probability_at_least_one_multiple_of_4_correct_l357_357310


namespace integral_rational_function_l357_357351

theorem integral_rational_function:
  (∫, x: ℝ, (x^3 - 6 * x^2 + 13 * x - 7) / ((x + 1) * (x - 2)^3)) =
  ln (abs (x+1)) - 1 / (2 * (x-2)^2) + C :=
sorry

end integral_rational_function_l357_357351


namespace num_subsets_containing_6_l357_357841

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l357_357841


namespace a_n_formula_d_value_l357_357107

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357107


namespace eval_expression_l357_357350

theorem eval_expression : 
  2 * log 2 (real.sqrt 2) - log 10 2 - log 10 5 + 1 / (3 * (27 / 8)^2) = 4 / 9 :=
by
  sorry

end eval_expression_l357_357350


namespace BD_distance_16_l357_357155

noncomputable def distanceBD (DA AB : ℝ) (angleBDA : ℝ) : ℝ :=
  (DA^2 + AB^2 - 2 * DA * AB * Real.cos angleBDA).sqrt

theorem BD_distance_16 :
  distanceBD 10 14 (60 * Real.pi / 180) = 16 := by
  sorry

end BD_distance_16_l357_357155


namespace a_n_formula_d_value_l357_357106

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357106


namespace subsets_containing_six_l357_357835

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l357_357835


namespace andrew_eggs_l357_357306

def andrew_eggs_problem (a b : ℕ) (half_eggs_given_away : ℚ) (remaining_eggs : ℕ) : Prop :=
  a + b - (a + b) * half_eggs_given_away = remaining_eggs

theorem andrew_eggs :
  andrew_eggs_problem 8 62 (1/2 : ℚ) 35 :=
by
  sorry

end andrew_eggs_l357_357306


namespace toy_factory_difference_l357_357462

theorem toy_factory_difference : 
  let toys_per_day_A := 288 / 12
  let toys_per_day_B := 243 / 9
  (toys_per_day_B - toys_per_day_A) = 3 :=
by
  let toys_per_day_A := 288 / 12
  let toys_per_day_B := 243 / 9
  have h1 : toys_per_day_A = 24 := rfl
  have h2 : toys_per_day_B = 27 := rfl
  calc
    toys_per_day_B - toys_per_day_A = 27 - 24 : by rw [h1, h2]
                                   ... = 3      : by norm_num

end toy_factory_difference_l357_357462


namespace part1_part2_l357_357099

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357099


namespace triangle_ABC_area_is_6_l357_357214

noncomputable def point := (ℝ × ℝ)

def A : point := (2, 0)
def B : point := (6, 0)
def C : point := (6, 3)

def right_angle (a b c : point) : Prop := (a.1 = b.1 ∧ b.2 = a.2) ∨ (a.2 = b.2 ∧ b.1 = a.1)

def distance (p1 p2 : point) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def triangle_area (a b c : point) : ℝ := 0.5 * (distance a b) * (distance b c)

theorem triangle_ABC_area_is_6 :
  right_angle A B C ∧
  triangle_area A B C = 6 := by
  sorry

end triangle_ABC_area_is_6_l357_357214


namespace largest_class_students_l357_357461

theorem largest_class_students (x : ℕ) :
  let students_in_each_class := [x, x - 2, x - 4, x - 6, x - 8] in
  (students_in_each_class.sum = 100) → x = 24 :=
by
  -- Summation definition for students in each class
  let students_in_each_class := [x, x - 2, x - 4, x - 6, x - 8]
  -- Total number of students in all classes
  assume total_students : students_in_each_class.sum = 100
  -- Prove that the largest class must have 24 students
  sorry

end largest_class_students_l357_357461


namespace no_real_solutions_l357_357564

noncomputable def g (x : ℝ) : ℝ := (3 - x^4) / (2 * x^2)

theorem no_real_solutions (x : ℝ) (hx : x ≠ 0) : g(x) = g(-x) :=
by {
  unfold g,
  calc
    (3 - x^4) / (2 * x^2) = (3 - (-x)^4) / (2 * (-x)^2) : by rw [pow_succ, pow_succ, mul_neg, neg_mul_eq_neg_mul_symm, neg_neg]
}

end no_real_solutions_l357_357564


namespace problem_correct_l357_357688

open Real

-- Definitions for the propositions
def p1 := ∃ x : ℝ, (0 < x) ∧ ((1 / 2) ^ x < (1 / 3) ^ x)
def p2 := ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ (log (1 / 2) x > log (1 / 3) x)
def p3 := ∀ x : ℝ, (0 < x) → ((1 / 2) ^ x > log (1 / 2) x)
def p4 := ∀ x : ℝ, (0 < x) ∧ (x < 1 / 3) → ((1 / 2) ^ x < log (1 / 3) x)

-- Main theorem statement asserting the truth values of each proposition
theorem problem_correct :
  ¬ p1 ∧ p2 ∧ ¬ p3 ∧ p4 := by
  sorry

end problem_correct_l357_357688


namespace max_intersections_l357_357734

theorem max_intersections : 
  let points_on_X := 15
  let points_on_Y := 6
  points_on_X * points_on_Y > 0 → 
  (Finset.card (Finset.pairsOf (Finset.range points_on_X).toFinset) 
                * Finset.card (Finset.pairsOf (Finset.range points_on_Y).toFinset)) = 1575 :=
by
  let points_on_X := 15
  let points_on_Y := 6
  intro h
  sorry

end max_intersections_l357_357734


namespace find_ages_l357_357725

-- Definitions of the conditions
def cond1 (D S : ℕ) : Prop := D = 3 * S
def cond2 (D S : ℕ) : Prop := D + 5 = 2 * (S + 5)

-- Theorem statement
theorem find_ages (D S : ℕ) 
  (h1 : cond1 D S) 
  (h2 : cond2 D S) : 
  D = 15 ∧ S = 5 :=
by 
  sorry

end find_ages_l357_357725


namespace min_value_of_z_l357_357236

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l357_357236


namespace theta_value_l357_357610

noncomputable def complex_exponential_form : Prop :=
  let z := 2 + 2 * complex.I in
  let θ := real.arctan (2 / 2) in
  θ = real.pi / 4

theorem theta_value : complex_exponential_form := sorry

end theta_value_l357_357610


namespace number_of_sequences_l357_357201

theorem number_of_sequences : 
  ∃ (a : ℕ → ℤ), (length (list.range 11) = 11) ∧ 
  a 1 = 0 ∧                      
  a 11 = 4 ∧                     
  (∀ k, 1 ≤ k ∧ k < 11 → |a (k + 1) - a k| = 1) ∧
  (finset.univ.card (finset.filter (λ (s : ℕ → ℤ), (length (list.range 11) = 11) ∧ 
  s 1 = 0 ∧                      
  s 11 = 4 ∧                     
  (∀ k, 1 ≤ k ∧ k < 11 → |s (k + 1) - s k| = 1)) (finset.range 11)) = 120) :=
sorry

end number_of_sequences_l357_357201


namespace hyperbola_solution_l357_357807

def is_hyperbola (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

def is_asymptote (a b : ℝ) := b = sqrt 3 * a

def point_on_hyperbola (x y a b : ℝ) := x = sqrt 5 ∧ y = sqrt 3 ∧ is_hyperbola a b x y

noncomputable def hyperbola_equation_condition (a : ℝ) : Prop :=
∀ x y : ℝ, point_on_hyperbola x y a b → (x^2 / 4 - y^2 / 12 = 1)

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
∀ l : ℝ -> ℝ -> Prop, (∀ P Q : ℝ × ℝ, P ≠ Q ∧ l P Q → P.1 ^ 2 + P.2 ^ 2 = 0) → ∃ min_val : ℝ, min_val = 24

theorem hyperbola_solution :
(∃ a b : ℝ, a > 0 ∧ b > 0 ∧ is_asymptote a b ∧ point_on_hyperbola (sqrt 5) (sqrt 3) a b) →
(∃ a : ℝ, hyperbola_equation_condition a) ∧
(∃ a b : ℝ, minimum_value_condition a b) :=
by sorry

end hyperbola_solution_l357_357807


namespace find_x_l357_357932

theorem find_x (x : ℤ) (h_pos : x > 0) 
  (n := x^2 + 2 * x + 17) 
  (d := 2 * x + 5)
  (h_div : n = d * x + 7) : x = 2 := 
sorry

end find_x_l357_357932


namespace weight_of_replaced_person_is_correct_l357_357871

-- Define a constant representing the number of persons in the group.
def num_people : ℕ := 10
-- Define a constant representing the weight of the new person.
def new_person_weight : ℝ := 110
-- Define a constant representing the increase in average weight when the new person joins.
def avg_weight_increase : ℝ := 5
-- Define the weight of the person who was replaced.
noncomputable def replaced_person_weight : ℝ :=
  new_person_weight - num_people * avg_weight_increase

-- Prove that the weight of the replaced person is 60 kg.
theorem weight_of_replaced_person_is_correct : replaced_person_weight = 60 :=
by
  -- Skip the detailed proof steps.
  sorry

end weight_of_replaced_person_is_correct_l357_357871


namespace concurrency_of_lines_l357_357163

open EuclideanGeometry

noncomputable def quadrilateral_cyclic (A B C D : Point) (O : Circle) : Prop := 
  Circle.inscribed A B C D O

noncomputable def circumcenter (A B P : Point) : Point := 
  classical.some (exists_circumcenter A B P)

theorem concurrency_of_lines 
  (A B C D P O : Point)
  (O₁ O₂ O₃ O₄ : Point)
  (h_cyclic : quadrilateral_cyclic A B C D O)
  (h_intersect: Line.inter AC BD P)
  (h_O1 : circumcenter A B P = O₁)
  (h_O2 : circumcenter B C P = O₂)
  (h_O3 : circumcenter C D P = O₃)
  (h_O4 : circumcenter D A P = O₄)
  : concurrency (Line.segment O P) (Line.segment O₁ O₃ | O₂ O₄) :=
sorry

end concurrency_of_lines_l357_357163


namespace OH_eq_2R_l357_357502

-- Definitions of essential geometric points
variables {O H : ℝ} -- Circumcenter and orthocenter
variables {R : ℝ} -- Radius of the circumcircle
variables {A B C D E F G : Type*} -- Points in the Euclidean plane
variables (d : A → B → C → A) -- Reflection function
variables (collinear : list C → Prop) -- Collinearity

-- Conditions
def is_circumcenter (O : ℝ) (A B C : Type*) : Prop := sorry
def is_orthocenter (H : ℝ) (A B C : Type*) : Prop := sorry
def circum_radius (R : ℝ) (A B C : Type*) : Prop := sorry
def reflections (d : A → B → C → A) (A B C D E F : Type*) := 
  d(A, BC) = D ∧ d(B, CA) = E ∧ d(C, AB) = F
def are_collinear (D E F : Type*) : Prop := collinear [D, E, F]

-- Hypotheses
variables [circumcenter_O : is_circumcenter O A B C]
variables [orthocenter_H : is_orthocenter H A B C]
variables [radius_R : circum_radius R A B C]
variables [refls : reflections d A B C D E F]
variables [linearity : are_collinear D E F]

-- Proof Statement
theorem OH_eq_2R : O - H = 2 * R :=
by sorry

end OH_eq_2R_l357_357502


namespace number_condition_l357_357374

theorem number_condition (x : ℝ) (h : 45 - 3 * x^2 = 12) : x = Real.sqrt 11 ∨ x = -Real.sqrt 11 :=
sorry

end number_condition_l357_357374


namespace transformation_not_possible_l357_357628

def valid_operations (a b c d : ℕ) : Prop :=
(∀ x y : ℕ, ((x = a ∧ y = b ∧ x ≠ 9 ∧ y ≠ 9) → valid_operations (x+1) (y+1) c d) ∨ 
((x = c ∧ y = d ∧ x ≠ 0 ∧ y ≠ 0) → valid_operations a b (x-1) (y-1) ))

def M (a b c d : ℕ) : ℕ := (d + b) - (a + c)

def cannot_transform (start target : ℕ) : Prop :=
let s₀ := 1234 in
let t₀ := 2002 in
let s := M 1 2 3 4 in
let t := M 2 0 0 2 in
¬ (s₀ = start → t₀ = target → s = t)

theorem transformation_not_possible : cannot_transform 1234 2002 :=
by 
  sorry

end transformation_not_possible_l357_357628


namespace largest_number_l357_357616

theorem largest_number (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 0.9791) 
  (h₂ : x₂ = 0.97019)
  (h₃ : x₃ = 0.97909)
  (h₄ : x₄ = 0.971)
  (h₅ : x₅ = 0.97109)
  : max x₁ (max x₂ (max x₃ (max x₄ x₅))) = 0.9791 :=
  sorry

end largest_number_l357_357616


namespace find_general_formula_and_d_l357_357088

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357088


namespace cost_of_downloading_360_songs_in_2005_is_144_dollars_l357_357448

theorem cost_of_downloading_360_songs_in_2005_is_144_dollars :
  (∀ (c_2004 c_2005 : ℕ), (∀ c : ℕ, c_2005 = c ∧ c_2004 = c + 32) →
  200 * c_2004 = 360 * c_2005 → 360 * c_2005 / 100 = 144) :=
  by sorry

end cost_of_downloading_360_songs_in_2005_is_144_dollars_l357_357448


namespace jerrys_dad_reduction_l357_357488

variable (T D : ℝ)
variable (final_temp : ℝ := 59)
variable (initial_temp_is_set : 2 * T = 40)
variable (final_temp_eq : (2 * T - D) - 0.30 * (2 * T - D) + 24 = final_temp)

theorem jerrys_dad_reduction : D = 10 := by
  -- Given conditions
  have h1 : 2 * T = 40 := initial_temp_is_set
  have h2 : (2 * T - D) - 0.30 * (2 * T - D) + 24 = final_temp := final_temp_eq
  
  -- Substitute initial_temp_is_set to find T
  have T_val : T = 20 := by
    rw [mul_comm] at h1
    exact eq_of_mul_eq_mul_right (by norm_num) h1
  
  -- Substitute T = 20 into the final temperature equation to solve for D
  sorry

end jerrys_dad_reduction_l357_357488


namespace correct_option_is_D_l357_357563

def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_option_is_D :
  (∃ (a b c : ℝ), 
    quadratic_function (-2) = 6 ∧ 
    quadratic_function 0 = -4 ∧ 
    quadratic_function 1 = -6 ∧ 
    quadratic_function 3 = -4) →
  (∀ (x : ℝ), 
    quadratic_function x ≥ quadratic_function (3/2)) →
  (quadratic_function (3/2) < -6) :=
by
  intros h1 h2
  sorry

end correct_option_is_D_l357_357563


namespace investment_ratio_l357_357274

theorem investment_ratio
  (B_investment : ℝ)
  (A_multiplier : ℝ)
  (B_period : ℝ)
  (total_profit : ℝ)
  (B_profit : ℝ)
  (A_profit : ℝ)
  (profit_condition : B_profit = 7000)
  (total_profit_condition : total_profit = 49000)
  (A_period_condition : A_multiplier * B_investment * 2 * B_period) 
  (B_period_condition : B_investment * B_period)
  (ratio_condition : (A_profit / B_profit) = 6) :
  A_multiplier = 3 := 
sorry

end investment_ratio_l357_357274


namespace area_of_inscribed_square_l357_357297

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := x^2 - 6 * x + 8

-- Define the coordinates of the vertex of the parabola
def vertex := (3 : ℝ, -1 : ℝ)

-- Statement that proves the area of the inscribed square
theorem area_of_inscribed_square :
  let s : ℝ := sqrt 2 - 1 in
  let area : ℝ := (2 * s)^2 in
  area = 12 - 8 * sqrt 2 := by
sorry

end area_of_inscribed_square_l357_357297


namespace intersects_x_axis_and_minimum_distance_l357_357422

noncomputable def quadratic_intersects_x_axis_at_two_points (m : ℝ) (h : m ≠ 0) : Prop :=
  let Δ := 16 * (m - 3)^2 + 64 * m
  Δ > 0

noncomputable def minimum_distance_between_intersections (m : ℝ) (h : m ≠ 0) : Prop :=
  let distance := (4 * Real.sqrt (1 / m^2 - 2 / m + 9 / m^2))
  let is_minimum := m = 9 ∧ distance = Real.sqrt 8 / 3
  ∧ (∀ m' : ℝ, m' ≠ 0 → (4 * Real.sqrt (1 / m'^2 - 2 / m' + 9 / m'^2)) ≥ Real.sqrt 8 / 3)
  ∧ quadratic_function_vertex := ⟨-4 / 3, -32⟩
  ∧ opens_upward := 9 > 0

theorem intersects_x_axis_and_minimum_distance :
  ∀ (m : ℝ), m ≠ 0 →
  quadratic_intersects_x_axis_at_two_points m
  ∧ minimum_distance_between_intersections m :=
by
  intros
  sorry

end intersects_x_axis_and_minimum_distance_l357_357422


namespace sarees_shirts_cost_l357_357993

variable (S T : ℕ)

-- Definition of conditions
def condition1 : Prop := 2 * S + 4 * T = 2 * S + 4 * T
def condition2 : Prop := (S + 6 * T) = (2 * S + 4 * T)
def condition3 : Prop := 12 * T = 2400

-- Proof goal
theorem sarees_shirts_cost :
  condition1 S T → condition2 S T → condition3 T → 2 * S + 4 * T = 1600 := by
  sorry

end sarees_shirts_cost_l357_357993


namespace edith_books_count_l357_357346

-- Definitions for the conditions
def num_novels_first_shelf : ℕ := 67  -- 1.2 * 56 = 67.2 (round to nearest whole number)
def num_novels_second_shelf : ℕ := 56
def num_total_novels : ℕ := num_novels_first_shelf + num_novels_second_shelf
def num_writing_books : ℕ := (num_total_novels / 2).ceil  -- rounding to nearest whole number
def total_books : ℕ := num_total_novels + num_writing_books

-- The theorem to be proved
theorem edith_books_count : total_books = 185 := by
  sorry

end edith_books_count_l357_357346


namespace polygon_sides_l357_357294

theorem polygon_sides (perimeter side_length : ℕ) (h₁ : perimeter = 150) (h₂ : side_length = 15): 
  (perimeter / side_length) = 10 := 
by
  -- Here goes the proof part
  sorry

end polygon_sides_l357_357294


namespace circumcenter_rational_l357_357548

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l357_357548


namespace Gaspard_wins_l357_357758

-- Define the conditions
def plate_radius : ℝ := 0.15 -- radius in meters
def table_radius : ℝ := 1.0  -- radius in meters

-- Define areas based on the conditions
def plate_area : ℝ := π * plate_radius^2
def table_area : ℝ := π * table_radius^2

-- Define the maximum number of plates that can fit on the table
def max_plates : ℕ := ⌊table_area / plate_area⌋.to_nat

-- Define the theorem: Prove Gaspard has a winning strategy
theorem Gaspard_wins : ∀ (k : ℕ), k ≤ max_plates → Gaspard_has_winning_strategy k := by
  sorry

-- Helper definition to encapsulate Gaspard's winning strategy
def Gaspard_has_winning_strategy (k : ℕ) : Prop :=
  ∀ (positions : fin k → (ℝ × ℝ)), valid_positions positions → ∃ (position : ℝ × ℝ), valid_position (position :: positions) 

-- Define definitions to check validity of positions
def valid_position (positions : list (ℝ × ℝ)) (position : ℝ × ℝ) : Prop :=
  (position.1^2 + position.2^2 <= table_radius^2) ∧
  ∀ p ∈ positions, (position.1 - p.1)^2 + (position.2 - p.2)^2 > (2 * plate_radius)^2

def valid_positions (positions : fin k → (ℝ × ℝ)) : Prop :=
  ∀ i, valid_position (positions i :: (λ j, positions (i.succ j))) (positions i)

-- Sorry for the proof since the construction was required for the Lean statement

end Gaspard_wins_l357_357758


namespace inequality_solution_l357_357335

-- Define the function on ℝ with its derivative
variables {f : ℝ → ℝ}
variable  (f' : ℝ → ℝ)

-- Given conditions
axiom h1 : ∀ x, x * f' x + f x < -f' x
axiom h2 : f 2 = 1 / 3

-- Required theorem to prove the solution of the inequality
theorem inequality_solution (x : ℝ) : (f (exp x - 2) - (1 / (exp x - 1)) < 0) ↔ x ∈ Iio 0 ∪ Ioi (log 4) :=
sorry

end inequality_solution_l357_357335


namespace sum_series_eq_two_l357_357709

theorem sum_series_eq_two : ∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1)) = 2 :=
sorry

end sum_series_eq_two_l357_357709


namespace power_log_equality_l357_357015

theorem power_log_equality (c d : ℝ) (h1 : 30^c = 2) (h2 : 30^d = 7) : 
  18^((2 - 2*c - d) / (3 * (1 - d))) = 2 := 
  sorry

end power_log_equality_l357_357015


namespace q_value_correct_l357_357870

noncomputable def q (a_1 := 1 / (q ^ 2) : ℝ) : ℝ := (Real.sqrt 5 - 1) / 2

theorem q_value_correct (q : ℝ)
  (h1 : a_1 = 1 / (q ^ 2))
  (h2 : S_5 = S_2 + 2)
  (h_geo_seq : ∀ n, a_{n+1} = q * a_n)
  (h_all_pos : ∀ n, a_n > 0) :
  q = (Real.sqrt 5 - 1) / 2 :=
sorry

end q_value_correct_l357_357870


namespace integer_digit_strike_out_divisible_by_9_all_integers_satisfying_condition_l357_357304

theorem integer_digit_strike_out_divisible_by_9
  (N : ℕ) :
  ∃ M : ℕ, (∃ b : ℕ, N = M * 9 ∧ (∃ m, m < N ∧ m = N - b * 10^(nat.floor (real.log10 b))) ∧ M % 9 = 0) 
  ↔ (∀ d ∈ digits 10 N, d = 0 ∨ d = 9) :=
sorry

theorem all_integers_satisfying_condition :
  { N : ℕ // ∃ M : ℕ, (∃ b : ℕ, N = M * 9 ∧ (∃ m, m < N ∧ m = N - b * 10^(nat.floor (real.log10 b))) ∧ M % 9 = 0) }
  = {10125, 2025, 30375, 405, 50625, 675, 70875} :=
sorry

end integer_digit_strike_out_divisible_by_9_all_integers_satisfying_condition_l357_357304


namespace price_after_15_years_l357_357344

-- Define the initial price, the factor of price reduction, and the time period
def initial_price : ℝ := 5400
def reduction_factor : ℝ := 2 / 3
def periods : ℕ := 3 -- 15 years corresponds to 3 periods of 5 years

-- Define the final_price function with the given conditions
def final_price (p : ℝ) (f : ℝ) (n : ℕ) := p * (f ^ n)

-- State the theorem to be proved
theorem price_after_15_years : final_price initial_price reduction_factor periods = 1600 := 
by 
  -- The proof is omitted
  sorry

end price_after_15_years_l357_357344


namespace find_c_and_d_l357_357500

noncomputable def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℚ := 1 -- Identity matrix

def c := (1 : ℚ) / 36
def d := - (1 : ℚ) / 12

theorem find_c_and_d :
  (N⁻¹ = c • (N * N) + d • I) := by
  sorry

end find_c_and_d_l357_357500


namespace minimum_amount_spent_on_boxes_l357_357627

theorem minimum_amount_spent_on_boxes
  (box_length : ℕ) (box_width : ℕ) (box_height : ℕ) 
  (cost_per_box : ℝ) (total_volume_needed : ℕ) :
  box_length = 20 →
  box_width = 20 →
  box_height = 12 →
  cost_per_box = 0.50 →
  total_volume_needed = 2400000 →
  (total_volume_needed / (box_length * box_width * box_height) * cost_per_box) = 250 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end minimum_amount_spent_on_boxes_l357_357627


namespace part1_part2_l357_357203

-- Part 1: Prove the range of positive integer k such that 1 is in M
theorem part1 (k : ℤ) (h1 : (1 : ℝ) ∈ {x : ℝ | (k^2 - 2*k - 3)*x^2 - (k+1)*x - 1 < 0}) : k ∈ {1, 2, 3, 4} :=
sorry

-- Part 2: Prove the range of real number k such that M covers all real numbers
theorem part2 (k : ℝ) (hk : ∀ x : ℝ, (k^2 - 2*k - 3)*x^2 - (k+1)*x - 1 < 0) : -1 < k ∧ k < 11/5 :=
sorry

end part1_part2_l357_357203


namespace greatest_three_digit_number_condition_l357_357603

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end greatest_three_digit_number_condition_l357_357603


namespace triangle_area_correct_l357_357716

-- Define the sides of the triangle and the bisector length
def side_a : ℝ := 35
def side_b : ℝ := 14
def bisector : ℝ := 12

-- Define the expected area of the triangle
def expected_area : ℝ := 235.2

-- Prove that the area of the triangle with the given conditions is equal to the expected area
theorem triangle_area_correct :
  ∃ α : ℝ, sin α > 0 ∧ cos α > 0 ∧ 
  (147 * sin (2 * α) = expected_area) := 
sorry

end triangle_area_correct_l357_357716


namespace num_subsets_containing_6_l357_357824

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l357_357824


namespace sum_of_smallest_and_largest_three_digit_numbers_l357_357427

theorem sum_of_smallest_and_largest_three_digit_numbers : 
  ∀ (d1 d2 d3 d4 : ℕ), 
    (d1 = 0 ∨ d1 = 1 ∨ d1 = 3 ∨ d1 = 5) ∧ 
    (d2 = 0 ∨ d2 = 1 ∨ d2 = 3 ∨ d2 = 5) ∧ 
    (d3 = 0 ∨ d3 = 1 ∨ d3 = 3 ∨ d3 = 5) ∧ 
    (d4 = 0 ∨ d4 = 1 ∨ d4 = 3 ∨ d4 = 5) ∧ 
    (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ 
    (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ 
    (d3 ≠ d4) → 
    let digits := [0, 1, 3, 5] in
    let min_number := 103 in
    let max_number := 531 in
    (min_number + max_number) = 634 :=
by
  sorry -- proof omitted

end sum_of_smallest_and_largest_three_digit_numbers_l357_357427


namespace unanswered_questions_equal_nine_l357_357899

theorem unanswered_questions_equal_nine
  (x y z : ℕ)
  (h1 : 5 * x + 2 * z = 93)
  (h2 : 4 * x - y = 54)
  (h3 : x + y + z = 30) : 
  z = 9 := by
  sorry

end unanswered_questions_equal_nine_l357_357899


namespace find_X_value_l357_357863

theorem find_X_value :
  ∃ X S : ℕ, (X = 5 + 3 * (n - 1) ∧ S = n / 2 * (10 + 3 * (n - 1)) ∧ S ≥ 12000) → X = 302 :=
begin
  sorry
end

end find_X_value_l357_357863


namespace banker_l357_357537

noncomputable def banker's_gain (BD : ℝ) (Rate : ℝ) (Time : ℝ) : ℝ :=
  let FV := BD * 100 / (Rate * Time)
  let TD := FV / (1 + (Rate * Time) / 100)
  BD - TD

theorem banker's_gain_is_correct :
  banker's_gain 1978 12 6 ≈ 380.78 :=
by
  sorry

end banker_l357_357537


namespace find_x_l357_357584

-- define initial quantities of apples and oranges
def initial_apples (x : ℕ) : ℕ := 3 * x + 1
def initial_oranges (x : ℕ) : ℕ := 4 * x + 12

-- define the condition that the number of oranges is twice the number of apples
def condition (x : ℕ) : Prop := initial_oranges x = 2 * initial_apples x

-- define the final state
def final_apples : ℕ := 1
def final_oranges : ℕ := 12

-- theorem to prove that the number of times is 5
theorem find_x : ∃ x : ℕ, condition x ∧ final_apples = 1 ∧ final_oranges = 12 :=
by
  use 5
  sorry

end find_x_l357_357584


namespace seq_problem_l357_357296

-- Define the sequence a recursively
def a : ℕ → ℚ
| 1     := 3
| 2     := 5 / 11
| (n+2) := (a n * a (n+1) + 1) / (2 * a n - a (n+1))

-- Define the problem statement, proving that the sum p + q for a_50 is 168
theorem seq_problem (p q : ℕ) (h : (a 50) = (p / q) ∧ nat.gcd p q = 1) : p + q = 168 := 
sorry

end seq_problem_l357_357296


namespace income_percentage_change_is_11_11_decrease_l357_357197

-- Define the original payment per hour and original working time
def P : ℝ := sorry -- Original payment 
def T : ℝ := sorry -- Original working hours

-- Define the updated payment per hour considering a 33.33% increase
def P_new : ℝ := P * 1.3333

-- Define the updated working hours considering a 33.33% decrease
def T_new : ℝ := T * 0.6667

-- Calculate the original income
def Income_original : ℝ := P * T

-- Calculate the new income
def Income_new : ℝ := P_new * T_new

-- Define the percentage change in income
def percentage_change : ℝ := ((Income_new - Income_original) / Income_original) * 100

-- The theorem we want to prove
theorem income_percentage_change_is_11_11_decrease : percentage_change = -11.11 :=
by
  unfold percentage_change Income_new Income_original P_new T_new P T
  sorry

end income_percentage_change_is_11_11_decrease_l357_357197


namespace subsets_containing_six_l357_357833

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l357_357833


namespace arithmetic_seq_iff_arith_seq_S_div_n_l357_357400

variable {a_n : ℕ → ℝ} -- Define the sequence {a_n} indexed by natural numbers

-- Define the sum of the first n terms S(n)
noncomputable def S : ℕ → ℝ := λ n, (finset.range n).sum a_n

-- Define that {a_n} is an arithmetic sequence
def is_arithmetic_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ (A B : ℝ), ∀ n, a_n n = A * n + B

-- Define that {S(n)/n} is an arithmetic sequence
def is_arith_seq_S_div_n (S : ℕ → ℝ) : Prop :=
  ∃ (A B : ℝ), ∀ n, n ≠ 0 → S n / (n:ℝ) = A * n + B

-- The theorem to prove
theorem arithmetic_seq_iff_arith_seq_S_div_n :
  is_arithmetic_seq a_n ↔ is_arith_seq_S_div_n S :=
sorry

end arithmetic_seq_iff_arith_seq_S_div_n_l357_357400


namespace range_of_values_for_a_l357_357443

noncomputable def problem_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (|x - 2| + |x + 3| < a)

theorem range_of_values_for_a (a : ℝ) :
  problem_statement a → a ≤ 5 :=
  sorry

end range_of_values_for_a_l357_357443


namespace base_number_is_two_l357_357029

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^22)
  (h2 : n = 21) : x = 2 :=
sorry

end base_number_is_two_l357_357029


namespace ratio_of_areas_l357_357053

-- Define the circle and the geometric points
variables {O A B C D E : Type} 

-- Define the conditions
variables (hAB_diameter : is_diameter O A B)
variables (hCD_parallel : is_parallel CD AB)
variables (hAC_intersect_BD_at_E : intersects AC BD E)
variables (h_angle_AED : angle AED = α)
variables (h_angle_BOC : angle BOC = 2 * α)

-- Define the areas
noncomputable def area_CDE := area (triangle C D E)
noncomputable def area_ABE := area (triangle A B E)

-- The required theorem
theorem ratio_of_areas : (area_CDE / area_ABE) = (sin α) ^ 2 :=
sorry

end ratio_of_areas_l357_357053


namespace polynomial_evaluation_l357_357079

-- Define the context of polynomial and the given conditions
def Q : ℝ → ℝ := λ x, a * x^3 + b * x^2 + c * x + k

theorem polynomial_evaluation (a b c k : ℝ)
  (h0 : Q 0 = k)
  (h1 : Q 1 = 3 * k)
  (h_1 : Q (-1) = 4 * k) :
  Q 2 + Q (-2) = 22 * k :=
by
  sorry

end polynomial_evaluation_l357_357079


namespace log2_f_of_3_eq_16_l357_357032

theorem log2_f_of_3_eq_16 : 
  (f(x) = (finset.range 9).sum (λ k, nat.choose 8 k * x ^ k)) → 
  (log 2 (f 3) = 16) := by 
sorry

end log2_f_of_3_eq_16_l357_357032


namespace log_sum_l357_357355

theorem log_sum :
  log 10 50 + log 10 20 = 3 :=
by
  sorry

end log_sum_l357_357355


namespace min_value_expression_l357_357508

variable (p q r : ℝ)
variable (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)

theorem min_value_expression :
  (9 * r / (3 * p + 2 * q) + 9 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ≥ 2 :=
sorry

end min_value_expression_l357_357508


namespace optimal_kiosk_placement_l357_357822

open Real

noncomputable def optimal_radius (R : ℝ) : ℝ := R * sqrt 3 / 2

theorem optimal_kiosk_placement (R : ℝ) (hR : 0 < R) :
  ∃ r, r = optimal_radius R ∧ ∀ (p : ℝ × ℝ), sqrt (p.1^2 + p.2^2) <= R → 
  min (dist p (R * cos (0 : ℝ), R * sin (0 : ℝ)))
      (min (dist p (R * cos (2 * π / 3), R * sin (2 * π / 3)))
           (dist p (R * cos ((-2 * π) / 3), R * sin ((-2 * π) / 3)))) <= r :=
begin
  -- The proof steps would go here
  sorry
end

end optimal_kiosk_placement_l357_357822


namespace linda_change_l357_357137

-- Defining the conditions
def cost_per_banana : ℝ := 0.30
def number_of_bananas : ℕ := 5
def amount_paid : ℝ := 10.00

-- Proving the statement
theorem linda_change :
  amount_paid - (number_of_bananas * cost_per_banana) = 8.50 :=
by
  sorry

end linda_change_l357_357137


namespace sum_of_roots_l357_357934

theorem sum_of_roots :
  ∀ (x1 x2 : ℝ), (x1*x2 = 2 ∧ x1 + x2 = 3 ∧ x1 ≠ x2) ↔ (x1*x2 + 3*x1*x2 = 2 * x1 * x2 * x1:     by sorry

end sum_of_roots_l357_357934


namespace regular_octagon_exterior_angle_45_l357_357460

-- Define what it means to be a regular octagon
def is_regular_octagon (polygon : list (ℝ × ℝ)) : Prop :=
  -- Suppose a way to determine it's a regular octagon (omitted for simplicity)
  sorry

-- Define a function to calculate the exterior angle of a regular octagon
noncomputable def exterior_angle_regular_octagon (polygon : list (ℝ × ℝ)) [is_regular_octagon polygon] : ℝ :=
  360 / 8

-- The theorem that states the exterior angle is 45 degrees
theorem regular_octagon_exterior_angle_45 (polygon : list (ℝ × ℝ)) (h : is_regular_octagon polygon) : exterior_angle_regular_octagon polygon = 45 :=
  sorry

end regular_octagon_exterior_angle_45_l357_357460


namespace sum_of_two_cubes_lt_1000_l357_357006

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357006


namespace arithmetic_seq_sum_l357_357397

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_123 : a 0 + a 1 + a 2 = -3)
  (h_456 : a 3 + a 4 + a 5 = 6) :
  ∀ n, S n = n * (-2) + n * (n - 1) / 2 :=
by
  sorry

end arithmetic_seq_sum_l357_357397


namespace find_lengths_and_radii_l357_357180

-- Definitions for the given conditions
structure Circle (α : Type) :=
(center : α)
(radius : ℝ)

variables {α : Type} [metric_space α] [normed_group α] [normed_space ℝ α] 

-- Assuming necessary points and the circles
variables (F A B C D E P H : α)
variables (ω Ω : Circle α)
variables (l : α → α → Prop)

-- Conditions
variable h1 : externally_tangent ω Ω F
variable h2 : tangent_at ω A
variable h3 : tangent_at Ω B
variable h4 : passes_through_line l B C
variable h5 : on_circle l ω E
variable h6 : on_circle l Ω D
variable h7 : DH = HC ∧ DH = 2
variable h8 : point_between D C E
variable h9 : point_between H P F
variable h10 : BC = 60

-- Theorem: Finding the lengths and radii
theorem find_lengths_and_radii : 
  segment_length H P = 6 * sqrt 31 ∧
  ω.radius = 6 * sqrt (93 / 5) ∧ 
  Ω.radius = 8 * sqrt (155 / 3) := by
  sorry

end find_lengths_and_radii_l357_357180


namespace dividend_ratio_l357_357650

theorem dividend_ratio
  (expected_earnings_per_share : ℝ)
  (actual_earnings_per_share : ℝ)
  (dividend_per_share_increase : ℝ)
  (threshold_earnings_increase : ℝ)
  (shares_owned : ℕ)
  (h_expected_earnings : expected_earnings_per_share = 0.8)
  (h_actual_earnings : actual_earnings_per_share = 1.1)
  (h_dividend_increase : dividend_per_share_increase = 0.04)
  (h_threshold_increase : threshold_earnings_increase = 0.1)
  (h_shares_owned : shares_owned = 100)
  : (shares_owned * (expected_earnings_per_share + 
      (actual_earnings_per_share - expected_earnings_per_share) / threshold_earnings_increase * dividend_per_share_increase)) /
    (shares_owned * actual_earnings_per_share) = 46 / 55 :=
by
  sorry

end dividend_ratio_l357_357650


namespace distance_between_points_A_B_l357_357779

theorem distance_between_points_A_B :
  let A := (8, -5)
  let B := (0, 10)
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = 17 :=
by
  let A := (8, -5)
  let B := (0, 10)
  sorry

end distance_between_points_A_B_l357_357779


namespace part1_part2_l357_357120

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357120


namespace find_point_C_coordinates_l357_357785

def Point := ℕ × ℕ
def Line := ℕ → ℤ

structure Triangle :=
  (A B C : Point)
  (area: ℕ)

def onLine (p : Point) (l : Line) : Prop := l (fst p) = snd p

theorem find_point_C_coordinates
  (A := (3,2) : Point)
  (B := (-1,5) : Point)
  (C : Point)
  (line_eq : ∀ x, 3 * x - snd C + 3 = 0) --point C lies on the line 3x - y + 3 = 0
  (area_triangle_ABC : Triangle.area {A := A, B := B, C := C} = 10)
  : C = (-1, 0) ∨ C = (5/3, 8) := sorry

end find_point_C_coordinates_l357_357785


namespace oblique_projection_correct_statements_l357_357597

-- Definitions of conditions
def oblique_projection_parallel_invariant : Prop :=
  ∀ (x_parallel y_parallel : Prop), x_parallel ∧ y_parallel

def oblique_projection_length_changes : Prop :=
  ∀ (x y : ℝ), x = y / 2 ∨ x = y

def triangle_is_triangle : Prop :=
  ∀ (t : Type), t = t

def square_is_rhombus : Prop :=
  ∀ (s : Type), s = s → false

def isosceles_trapezoid_is_parallelogram : Prop :=
  ∀ (it : Type), it = it → false

def rhombus_is_rhombus : Prop :=
  ∀ (r : Type), r = r → false

-- Math proof problem
theorem oblique_projection_correct_statements :
  (triangle_is_triangle ∧ oblique_projection_parallel_invariant ∧ oblique_projection_length_changes)
  → ¬square_is_rhombus ∧ ¬isosceles_trapezoid_is_parallelogram ∧ ¬rhombus_is_rhombus :=
by 
  sorry

end oblique_projection_correct_statements_l357_357597


namespace ratio_diminished_to_total_l357_357662

-- Definitions related to the conditions
def N := 240
def P := 60
def fifth_part_increased (N : ℕ) : ℕ := (N / 5) + 6
def part_diminished (P : ℕ) : ℕ := P - 6

-- The proof problem statement
theorem ratio_diminished_to_total 
  (h1 : fifth_part_increased N = part_diminished P) : 
  (P - 6) / N = 9 / 40 :=
by sorry

end ratio_diminished_to_total_l357_357662


namespace min_value_of_f_l357_357230

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l357_357230


namespace subsets_containing_six_l357_357836

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l357_357836


namespace find_two_digit_number_l357_357749

theorem find_two_digit_number : 
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (b = 0 ∨ b = 5) ∧ (10 * a + b = 5 * (a + b)) ∧ (10 * a + b = 45) :=
by
  sorry

end find_two_digit_number_l357_357749


namespace consecutive_sums_to_300_l357_357070

def A := {x : ℕ | x % 3 ≠ 0}

def consecutive_sums_to_s (s : ℕ) (n : ℕ) : Prop :=
  ∃ (l : List ℕ), (l.length = 2 * n) ∧ (l.sum = s) ∧ (∀ x ∈ l, x ∈ A) ∧ (∀ i j, i < j → (l.nth i).get_or_else 0 < (l.nth j).get_or_else 0)

theorem consecutive_sums_to_300 (n : ℕ) : (consecutive_sums_to_s 300 n) ↔ (n = 2 ∨ n = 4) :=
by
  sorry

end consecutive_sums_to_300_l357_357070


namespace julia_change_l357_357457

-- Definitions based on the problem conditions
def price_of_snickers : ℝ := 1.5
def price_of_mms : ℝ := 2 * price_of_snickers
def total_cost_of_snickers (num_snickers : ℕ) : ℝ := num_snickers * price_of_snickers
def total_cost_of_mms (num_mms : ℕ) : ℝ := num_mms * price_of_mms
def total_purchase (num_snickers num_mms : ℕ) : ℝ := total_cost_of_snickers num_snickers + total_cost_of_mms num_mms
def amount_given : ℝ := 2 * 10

-- Prove the change is $8
theorem julia_change : total_purchase 2 3 = 12 ∧ (amount_given - total_purchase 2 3) = 8 :=
by
  sorry

end julia_change_l357_357457


namespace min_chips_1x2100_strip_l357_357466

theorem min_chips_1x2100_strip : 
  ∃ (n : ℕ), 
    ((∀ i : ℕ, 1 ≤ i → i ≤ 2100 - n → abs_diff_recorded
       (i - 1).count_chips_left (i + 1).count_chips_right ≠ 0 ∧ 
       abs_diff_recorded
       (i - 1).count_chips_left (i + 1).count_chips_right ≠ (j : ℕ) ∀ j ≠ i)
    ∧ 
    ((2100 - n) ≤ (n + 1) / 2) 
    → 
    n = 1400

end min_chips_1x2100_strip_l357_357466


namespace combined_value_correct_l357_357359

noncomputable def radius : ℝ := 13
noncomputable def pi_approx : ℝ := 3.14159

noncomputable def circumference (r : ℝ) : ℝ := 2 * pi_approx * r
noncomputable def area (r : ℝ) : ℝ := pi_approx * r^2

noncomputable def combined_value (r : ℝ) : ℝ := circumference(r) + area(r)

theorem combined_value_correct : combined_value 13 = 612.6105 :=
by
  sorry

end combined_value_correct_l357_357359


namespace quadrilateral_not_necessarily_planar_l357_357690

theorem quadrilateral_not_necessarily_planar:
  (∀ (A B C D: Type) (triangle : Set (Set Point)) (trapezoid : Set (Set Point)) (parallelogram : Set (Set Point))
   (quadrilateral : Set (Set Point)),
    (∀ (X Y Z : Point), {X, Y, Z} ∈ triangle → collinear X Y Z) ∧
    (∀ (V W X Y : Point), {V, W, X, Y} ∈ trapezoid → (parallel_lines V W X Y ∨ parallel_lines W X Y Z)) ∧ 
    (∀ (L M N P : Point), {L, M, N, P} ∈ parallelogram → parallel_lines L M N P) → 
    (∃ (Q R S T : Point), {Q, R, S, T} ∈ quadrilateral ∧ ¬coplanar Q R S T)
  ) := 
sorry

end quadrilateral_not_necessarily_planar_l357_357690


namespace sum_of_two_cubes_lt_1000_l357_357003

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357003


namespace total_revenue_l357_357369

noncomputable def student_ticket_price : ℕ := 4
noncomputable def regular_ticket_price : ℕ := 8
noncomputable def total_people : ℕ := 3210
noncomputable def regular_to_student_ratio : ℕ := 3

theorem total_revenue : 
  let S := total_people / (regular_to_student_ratio + 1),
      R := regular_to_student_ratio * S,
      student_revenue := S * student_ticket_price,
      regular_revenue := R * regular_ticket_price in
  student_revenue + regular_revenue = 22456 := 
by
  sorry

end total_revenue_l357_357369


namespace cost_of_traveling_two_roads_l357_357292

-- Definitions from conditions
def lawnLength : ℝ := 80
def lawnBreadth : ℝ := 60
def roadWidth : ℝ := 10
def costPerSqMeter : ℝ := 3

-- The problem statement to be proved
theorem cost_of_traveling_two_roads :
  let road1_area := roadWidth * lawnBreadth in
  let road2_area := roadWidth * lawnLength in
  let intersection_area := roadWidth * roadWidth in
  let total_road_area := road1_area + road2_area - intersection_area in
  let total_cost := total_road_area * costPerSqMeter in
  total_cost = 3900 := 
by 
  -- Calculation steps
  sorry

end cost_of_traveling_two_roads_l357_357292


namespace sequence_a_converges_to_0_l357_357332

noncomputable def sequence_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n ≥ 1, a (n + 1) = (3 * a (n + 1) - a n)^(1 / 3) ∧ 0 ≤ a n ∧ a n ≤ 1

theorem sequence_a_converges_to_0 (a : ℕ → ℝ) :
  sequence_a a → ∃ l : ℝ, l = 0 ∧ (tendsto (λ n, a n) at_top (𝓝 l)) :=
sorry

end sequence_a_converges_to_0_l357_357332


namespace smallest_integer_in_set_l357_357873

open Real

theorem smallest_integer_in_set : 
  ∃ (n : ℤ), (λ n, let avg := (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7 
               in n + 6 > 2 * avg) n ∧ 
            ∀ k : ℤ, (λ k, let avg_k := (k + (k+1) + (k+2) + (k+3) + (k+4) + (k+5) + (k+6)) / 7 
               in k + 6 > 2 * avg_k) k → n ≤ k := 
by
  sorry

end smallest_integer_in_set_l357_357873


namespace remaining_amount_eq_40_l357_357177

-- Definitions and conditions
def initial_amount : ℕ := 100
def food_spending : ℕ := 20
def rides_spending : ℕ := 2 * food_spending
def total_spending : ℕ := food_spending + rides_spending

-- The proposition to be proved
theorem remaining_amount_eq_40 :
  initial_amount - total_spending = 40 :=
by
  sorry

end remaining_amount_eq_40_l357_357177


namespace circumcenter_is_rational_l357_357554

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l357_357554


namespace quadratic_equal_real_roots_l357_357439

theorem quadratic_equal_real_roots :
  ∃ k : ℝ, (∀ x : ℝ, x^2 - 4 * x + k = 0) ∧ k = 4 := by
  sorry

end quadratic_equal_real_roots_l357_357439


namespace rainy_days_l357_357521

theorem rainy_days (n R NR : ℕ): (n * R + 3 * NR = 20) ∧ (3 * NR = n * R + 10) ∧ (R + NR = 7) → R = 2 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end rainy_days_l357_357521


namespace circle_area_ratio_is_correct_l357_357023

noncomputable def circle_area_ratio (R_C R_D : ℝ) : ℝ :=
  (π * R_C^2) / (π * R_D^2)

theorem circle_area_ratio_is_correct (R_C R_D : ℝ) 
  (h1 : ∀ L, L = (60/360) * (2 * π * R_C) ↔ L = (40/360) * (2 * π * R_D)) :
  circle_area_ratio R_C R_D = 4 / 9 :=
by
  -- Placeholder for proof
  sorry

end circle_area_ratio_is_correct_l357_357023


namespace not_monotonic_function_l357_357796

def f (a x : ℝ) : ℝ :=
if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

def not_monotonic_range (a : ℝ) : Prop :=
(0 < a ∧ a < 1/7) ∨ (1/3 ≤ a ∧ a < 1) ∨ (1 < a)

theorem not_monotonic_function : ∀ a : ℝ, ¬ monotone (f a) ↔ not_monotonic_range a :=
by sorry

end not_monotonic_function_l357_357796


namespace exists_root_in_interval_l357_357410

def f (x : ℝ) : ℝ := 2^x + x - 5

theorem exists_root_in_interval :
  ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f(c) = 0 :=
sorry

end exists_root_in_interval_l357_357410


namespace induced_charge_on_end_balls_l357_357272

-- Define the constants
def R : ℝ := 0.001 -- Radius in meters
def l : ℝ := 0.3 -- Segment length in meters
def E : ℝ := 100 -- Electric field intensity in V/m
def N : ℕ := 50 -- Number of balls
def k : ℝ := 9 * 10^9 -- Electrostatic constant in Nm^2/C^2

-- Define the induced charge on the end balls
def q : ℝ := (E * (N - 1) * l * R) / (2 * k)

-- The theorem to prove that the induced charge is as given
theorem induced_charge_on_end_balls : q = 8.17 * 10^(-11) := sorry

end induced_charge_on_end_balls_l357_357272


namespace find_m_value_l357_357421

theorem find_m_value :
  ∀ (m : ℝ), (m > 0) →  ((m^2 - 2 * m - 2) = 1) → m = 3 :=
begin
  intros m h_pos h_eq,
  sorry
end

end find_m_value_l357_357421


namespace parabola_equation_true_centroid_trajectory_true_minimum_distance_true_l357_357789

noncomputable def parabola_equation : Prop :=
  ∃ (focus : ℝ × ℝ) (directrix : ℝ), 
  focus = (1/2, 0) ∧ directrix = -1/2 ∧ 
  ∀ (x y : ℝ), y^2 = 2 * x

noncomputable def centroid_trajectory : Prop :=
  ∃ (focus : ℝ × ℝ) (directrix : ℝ),
  focus = (1/2, 0) ∧ directrix = -1/2 ∧
  ∀ (k : ℝ), 
  k ≠ 0 ∧
  let x1 := (k^2 + 2) / k^2 in
  let y1 := 2 / k in 
  let G := ((0 + x1) / 3, (0 + y1) / 3) in
  G.2^2 = (2/3) * G.1 - 2/9 ∨ (G = (1/3, 0))

noncomputable def minimum_distance : Prop :=
  ∃ (focus : ℝ × ℝ) (directrix : ℝ),
  focus = (1/2, 0) ∧ directrix = -1/2 ∧
  ∀ (P : ℝ × ℝ), 
  P.1^2 / 2 = P.2 ∧
  let PQ2 := (P.1 - 3)^2 + P.2^2 in
  let MN := 2 * (sqrt 2) * sqrt (1 - (2 / PQ2)) in
  PQ2 = 5 ∧ MN = 2 * sqrt 30 / 5 ∧ (P = (2, 2) ∨ P = (2, -2))

theorem parabola_equation_true : parabola_equation := sorry
theorem centroid_trajectory_true : centroid_trajectory := sorry
theorem minimum_distance_true : minimum_distance := sorry

end parabola_equation_true_centroid_trajectory_true_minimum_distance_true_l357_357789


namespace janet_owes_correctly_l357_357065

def hourly_rate_warehouse_worker := 15
def hourly_rate_manager := 20
def FICA_tax_rate := 0.10
def retirement_contribution_rate := 0.05

def hours_per_month_wkA := 20 * 6
def hours_per_month_wkB := 25 * 8
def hours_per_month_wkC := 18 * 7
def hours_per_month_wkD := 22 * 9

def hours_per_month_mg1 := 26 * 10
def hours_per_month_mg2 := 25 * 9

def wages_worker_A := hours_per_month_wkA * hourly_rate_warehouse_worker
def wages_worker_B := hours_per_month_wkB * hourly_rate_warehouse_worker
def wages_worker_C := hours_per_month_wkC * hourly_rate_warehouse_worker
def wages_worker_D := hours_per_month_wkD * hourly_rate_warehouse_worker

def wages_manager_1 := hours_per_month_mg1 * hourly_rate_manager
def wages_manager_2 := hours_per_month_mg2 * hourly_rate_manager

def total_wages_warehouse_workers := wages_worker_A + wages_worker_B + wages_worker_C + wages_worker_D
def total_wages_managers := wages_manager_1 + wages_manager_2
def total_wages := total_wages_warehouse_workers + total_wages_managers

def FICA_taxes := FICA_tax_rate * total_wages
def retirement_contributions := retirement_contribution_rate * total_wages
def total_amount_owed := total_wages + FICA_taxes + retirement_contributions

theorem janet_owes_correctly :
  total_amount_owed = 22264 := by
  sorry

end janet_owes_correctly_l357_357065


namespace intersection_A_B_l357_357135

open Set

def U := ℝ

def A : Set ℝ := { x : ℝ | x^2 - 2 * x < 0 }

def B : Set ℝ := { y : ℝ | ∃ x : ℝ, y = exp x + 1 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l357_357135


namespace convex_polyhedron_even_faces_not_necessarily_two_color_edges_l357_357345

open SimpleGraph

theorem convex_polyhedron_even_faces_not_necessarily_two_color_edges (P : Type) [Fintype P] 
  (f : P → SimpleGraph P) (h : ∀ p ∈ P, (f p).adjacencyMatrix.even) :
  ¬∀ (c : P → Fin 2), ∀ (p ∈ P), (f p).EdgeSet.card / 2 = Cardinal.mk {e ∈ (f p).EdgeSet | c e = 0} :=
by
  sorry

end convex_polyhedron_even_faces_not_necessarily_two_color_edges_l357_357345


namespace cos_3pi_plus_alpha_l357_357396

-- Conditions
def alpha_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def sin_alpha (α : ℝ) : Prop := Real.sin α = Real.sqrt 10 / 5

-- Proof statement
theorem cos_3pi_plus_alpha (α: ℝ) (h1: alpha_is_acute α) (h2: sin_alpha α) :
  Real.cos (3 * π + α) = - Real.sqrt 15 / 5 :=
sorry

end cos_3pi_plus_alpha_l357_357396


namespace circle_area_l357_357319

/-
Circle A has a diameter equal to the radius of circle B.
The area of circle A is 16π square units.
Prove the area of circle B is 64π square units.
-/

theorem circle_area (rA dA rB : ℝ) (h1 : dA = 2 * rA) (h2 : rB = dA) (h3 : π * rA ^ 2 = 16 * π) : π * rB ^ 2 = 64 * π :=
by
  sorry

end circle_area_l357_357319


namespace min_value_four_over_a_plus_nine_over_b_l357_357429

theorem min_value_four_over_a_plus_nine_over_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → (∀ x y, x > 0 → y > 0 → x + y ≥ 2 * Real.sqrt (x * y)) →
  (∃ (min_val : ℝ), min_val = (4 / a + 9 / b) ∧ min_val = 25) :=
by
  intros a b ha hb hab am_gm
  sorry

end min_value_four_over_a_plus_nine_over_b_l357_357429


namespace geometric_sequence_sum_reciprocal_less_than_two_l357_357056

-- Define the sequence {a_n}
def a : ℕ → ℕ 
| 0     := 1
| 1     := 3
| (n+2) := 3 * (a (n+1)) - 2 * (a n)

-- Prove that the sequence {a_{n+1} - a_n} is geometric and find the general formula for {a_n}
theorem geometric_sequence (n : ℕ) : a n = 2^n - 1 := sorry

-- Define the sequence {b_n} such that b_n = log_2(a_n + 1)
def b (n : ℕ) : ℕ := n

-- Define the sum S_n of the first n terms of {b_n}
def S : ℕ → ℕ 
| 0 := 0
| (n+1) := (n * (n + 1)) // 2

-- Prove that the sum of reciprocals of S_n is less than 2
theorem sum_reciprocal_less_than_two (n : ℕ) : ∑ i in range (n+1), 1 / (S i) < 2 := sorry

end geometric_sequence_sum_reciprocal_less_than_two_l357_357056


namespace tagged_fish_in_second_catch_l357_357039

noncomputable def tagged_fish_in_pond := 50
noncomputable def fish_caught_second_time := 50
noncomputable def total_fish_in_pond := 312.5

theorem tagged_fish_in_second_catch :
  ∃ (T : ℕ), T / fish_caught_second_time ≈ tagged_fish_in_pond / total_fish_in_pond ∧ T ≈ 8 :=
by
  sorry

end tagged_fish_in_second_catch_l357_357039


namespace num_subsets_containing_6_l357_357839

open Finset

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the subset containing number 6
def subsets_with_6 (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ x => 6 ∈ x)

-- Theorem: The number of subsets of {1, 2, 3, 4, 5, 6} containing the number 6 is 32
theorem num_subsets_containing_6 : (subsets_with_6 S).card = 32 := by
  sorry

end num_subsets_containing_6_l357_357839


namespace monotonicity_f_maximum_m_range_m_l357_357411

-- Part (I): Monotonicity of f(x)
theorem monotonicity_f (m : ℝ) : 
  ∀ x : ℝ, f' x > 0 ↔ (m ≥ 0 ∨ x > ln (-m)) :=
sorry

-- Part (II): Maximum value of m satisfying f(x0) = x0 * ln x0
theorem maximum_m (x0 : ℝ) (hx0 : 0 < x0) : 
  f x0 = x0 * ln x0 → m ≤ 1 - Real.exp 1 :=
sorry

-- Part (III): Range of m such that f(g(x)) < f(x) for x > 0
theorem range_m (m : ℝ) : 
  (∀ x > 0, f (g x) < f x) ↔ m ≥ -1 :=
sorry

end monotonicity_f_maximum_m_range_m_l357_357411


namespace union_of_M_N_l357_357423

open Set

variable {X : Type} [LinearOrder X]

def M (x : X) := x > (0 : X)
def N (x : X) := x^2 - (4 : X) ≥ 0

theorem union_of_M_N : {x | M x} ∪ {x | N x} = {x : X | x ≤ -2 ∨ x > 0} := 
  sorry

end union_of_M_N_l357_357423


namespace largest_polygon_area_l357_357722

def area_unit_square : ℝ := 1
def area_right_triangle : ℝ := 0.5
def area_half_unit_square : ℝ := 0.5
def area_equilateral_triangle : ℝ := Real.sqrt 3 / 4

def area_polygon_A : ℝ := 3 * area_unit_square + 3 * area_right_triangle
def area_polygon_B : ℝ := 2 * area_unit_square + 4 * area_right_triangle + area_half_unit_square
def area_polygon_C : ℝ := 4 * area_unit_square + 2 * area_equilateral_triangle
def area_polygon_D : ℝ := 5 * area_right_triangle + area_half_unit_square
def area_polygon_E : ℝ := 3 * area_equilateral_triangle + 2 * area_half_unit_square

statement : Prop :=
  (∀ {a b c d e : ℝ}, a = area_polygon_A ∧ b = area_polygon_B ∧ c = area_polygon_C ∧ d = area_polygon_D ∧ e = area_polygon_E → c > a ∧ c > b ∧ c > d ∧ c > e)

theorem largest_polygon_area : statement := 
by 
  simp [area_polygon_A, area_polygon_B, area_polygon_C, area_polygon_D, area_polygon_E, area_unit_square, area_right_triangle, area_half_unit_square, area_equilateral_triangle]
  sorry

end largest_polygon_area_l357_357722


namespace ratio_of_b_l357_357533

theorem ratio_of_b (a b k a1 a2 b1 b2 : ℝ) (h_nonzero_a2 : a2 ≠ 0) (h_nonzero_b12: b1 ≠ 0 ∧ b2 ≠ 0) :
  (a * b = k) →
  (a1 * b1 = a2 * b2) →
  (a1 / a2 = 3 / 5) →
  (b1 / b2 = 5 / 3) := 
sorry

end ratio_of_b_l357_357533


namespace subsets_containing_six_l357_357831

theorem subsets_containing_six :
  ∃ s : Finset (Fin 6), s = {1, 2, 3, 4, 5, 6} ∧ (∃ n : ℕ, n = 32 ∧ n = 2 ^ 5) := by
  sorry

end subsets_containing_six_l357_357831


namespace ant_opposite_face_probability_l357_357320

-- Definition of the dodecahedron and its properties
structure Dodecahedron :=
  (num_vertices : ℕ)
  (num_faces : ℕ)
  (vertices_per_face : ℕ)
  (faces_per_vertex : ℕ)
  (adjacency : fin num_vertices → fin 3 → fin num_vertices)

noncomputable def dodecahedron : Dodecahedron :=
  { num_vertices := 20,
    num_faces := 12,
    vertices_per_face := 5,
    faces_per_vertex := 3,
    adjacency := sorry }

-- Probability calculation of the ant's movement
def opposite_face_probability (start A B : fin 20) : ℚ :=
  if (is_opposite_face start B) then 1/3 else 0

-- Dummy function to determine if two vertices are on opposite faces
def is_opposite_face (v1 v2 : fin 20) : Prop := sorry

theorem ant_opposite_face_probability :
  ∀ (start A B : fin 20), opposite_face_probability start A B = 1/3 :=
sorry

end ant_opposite_face_probability_l357_357320


namespace number_of_real_solutions_of_equation_l357_357569

theorem number_of_real_solutions_of_equation :
  (∀ x : ℝ, ((2 : ℝ)^(4 * x + 2)) * ((4 : ℝ)^(2 * x + 8)) = ((8 : ℝ)^(3 * x + 7))) ↔ x = -3 :=
by sorry

end number_of_real_solutions_of_equation_l357_357569


namespace intercept_sum_l357_357184

theorem intercept_sum (d e f : ℝ) (h_d : d = 4) (h_e : e = (9 - Real.sqrt 33) / 6) (h_f : f = (9 + Real.sqrt 33) / 6) :
  d + e + f = 7 :=
by
  rw [h_d, h_e, h_f]
  norm_num
  rw [←add_assoc, add_sub_cancel, Real.div_eq_mul_one_div, mul_add, ←mul_div_assoc, div_eq_mul_one_div 9 6, Real.sqrt]
  sorry

end intercept_sum_l357_357184


namespace trig_eq_solutions_l357_357252

noncomputable def solve_trig_eq (x : ℝ) : Prop :=
  (∃ k : ℤ, x = (Real.pi / 4) * (2 * k + 1)) ∨
  (∃ n : ℤ, x = (Real.pi / 7) * (2 * n + 1)) ∨
  (∃ l : ℤ, x = (Real.pi / 5) * (2 * l + 1))

theorem trig_eq_solutions (x : ℝ) :
  (cos (x / 2))^2 + (cos (3 * x / 2))^2 - (sin (2 * x))^2 - (sin (4 * x))^2 = 0 ↔ solve_trig_eq x :=
by sorry

end trig_eq_solutions_l357_357252


namespace max_lim_exists_l357_357074

noncomputable def sequence (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := a ^ (sequence a n)

theorem max_lim_exists (a : ℝ) (h : 1 < a) : 
  (∃ L, ∃ (lim : ℕ → ℝ) (hlim : lim = sequence a), limit (λ n, lim n) L → 
    a = Real.exp (1 / Real.exp 1) ∧ L = Real.exp 1) := sorry

end max_lim_exists_l357_357074


namespace no_equal_num_excellent_average_l357_357055

/--
  Consider a circle of 99 students where:
  1. Each student has at least one average student among the three neighbors to their left.
  2. Each student has at least one excellent student among the five neighbors to their right.
  3. Each student has at least one good student among the four neighbors (two to the left and two to the right).
  
  Prove that it is not possible to have an equal number of excellent and average students in the circle.
-/
theorem no_equal_num_excellent_average (students : Fin 99 → Prop) : 
  (∀ i, ∃ j, students (i - j) ∧ j ∈ {1, 2, 3}) ∧ 
  (∀ i, ∃ j, students (i + j) ∧ j ∈ {1, 2, 3, 4, 5}) ∧ 
  (∀ i, ∃ j, students (i ± j) ∧ j ∈ {1, 2, -1, -2}) 
  → ¬ ∃ n, count excellent students = n ∧ count average students = n := 
begin
  sorry
end

end no_equal_num_excellent_average_l357_357055


namespace complex_fraction_real_imag_sum_l357_357418

-- Definitions based on the problem's conditions
def imaginary_unit : Type := ℂ
def a : ℝ := 3 / 2
def b : ℝ := -(1 / 2)

-- Problem statement in Lean 4
theorem complex_fraction_real_imag_sum (i : imaginary_unit) (hi : i = complex.I) :
  ∃ a b : ℝ, (2 + i) / (1 + i) = a + b * i ∧ a + b = 1 :=
by
  sorry

end complex_fraction_real_imag_sum_l357_357418


namespace find_vertex_P_l357_357060

noncomputable def point := (ℝ × ℝ × ℝ)

def midpoint (A B : point) : point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

def M := (3, 5, 0) -- Midpoint of QR
def N := (1, 3, -3) -- Midpoint of PR
def O := (5, 0, 3) -- Midpoint of PQ

def P : point := (3, 1, 0)

theorem find_vertex_P : 
  ∃ Q R : point, midpoint Q R = M ∧ midpoint P R = N ∧ midpoint P Q = O :=
sorry

end find_vertex_P_l357_357060


namespace expected_heads_40_after_conditions_l357_357140

-- Defining the problem conditions in Lean
def fairCoin : ProbabilityMassFunction Bool :=
  ProbabilityMassFunction.of_multiset [tt, ff]

-- Defining the probabilities after each coin toss
noncomputable def successive_tosses (n : ℕ) : ProbabilityMassFunction Bool :=
  ProbabilityMassFunction.bind fairCoin (fun b =>
    if b then
      if n ≤ 1 then fairCoin else successive_tosses (n - 1)
    else successive_tosses (n - 1))

-- Expected number of heads from 80 coins
noncomputable def expected_heads_after_tosses (n_coins : ℕ) (max_tosses : ℕ) : ℝ :=
  (n_coins : ℝ) * (successive_tosses max_tosses).to_dist (λ b, if b then 1 else 0)

-- The main theorem to prove: the expected number of heads after 3 tosses
theorem expected_heads_40_after_conditions : expected_heads_after_tosses 80 3 = 40 := by
  sorry

end expected_heads_40_after_conditions_l357_357140


namespace parabola_value_f_l357_357538

theorem parabola_value_f (d e f : ℝ) :
  (∀ y : ℝ, x = d * y ^ 2 + e * y + f) →
  (∀ x y : ℝ, (x + 3) = d * (y - 1) ^ 2) →
  (x = -1 ∧ y = 3) →
  y = 0 →
  f = -2.5 :=
sorry

end parabola_value_f_l357_357538


namespace train_distance_l357_357285

theorem train_distance (t : ℕ) (d : ℕ) (rate : d / t = 1 / 2) (total_time : ℕ) (h : total_time = 90) : ∃ distance : ℕ, distance = 45 := by
  sorry

end train_distance_l357_357285


namespace volleyball_team_selection_l357_357523

def isQuadruplet (x : String) : Prop :=
  x = "Bella" ∨ x = "Bria" ∨ x = "Brittany" ∨ x = "Brooke"

theorem volleyball_team_selection :
  let players := ["Bella", "Bria", "Brittany", "Brooke", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"]
  let starters := {team : Set String // team.card = 6 ∧ ∃ set4 : Finset String, set4.card = 4 ∧ ∀ x ∈ set4, isQuadruplet x}
  ∑ n in (Finset.range 5), (n ≥ 2) * choose 4 n * choose 11 (6 - n) = 2695 := by
  sorry

end volleyball_team_selection_l357_357523


namespace tidal_power_station_location_l357_357317

-- Define the conditions
def tidal_power_plants : ℕ := 9
def first_bidirectional_plant := 1980
def significant_bidirectional_plant_location : String := "Jiangxia"
def largest_bidirectional_plant : Prop := true

-- Assumptions based on conditions
axiom china_has_9_tidal_power_plants : tidal_power_plants = 9
axiom first_bidirectional_in_1980 : (first_bidirectional_plant = 1980) -> significant_bidirectional_plant_location = "Jiangxia"
axiom largest_bidirectional_in_world : largest_bidirectional_plant

-- Definition of the problem
theorem tidal_power_station_location : significant_bidirectional_plant_location = "Jiangxia" :=
by
  sorry

end tidal_power_station_location_l357_357317


namespace verify_base_case_l357_357219

theorem verify_base_case : 1 + (1 / 2) + (1 / 3) < 2 :=
sorry

end verify_base_case_l357_357219


namespace median_divides_angle_bisector_in_given_ratio_l357_357058

-- Condition definitions from step-a
variables {A B C D P : Type}
variables [has_coordinates A] [has_coordinates B] [has_coordinates C] [has_coordinates D] [has_coordinates P]
variable [geometry.triangle A B C]
variable [geometry.angle_bisector A D B C]
variables [geometry.divides_segment B D C D (2 : ℝ) (1 : ℝ)]
variable [geometry.median C E A B]

-- Goal statement from step-c
theorem median_divides_angle_bisector_in_given_ratio 
  (h1 : triangle A B C)
  (h2 : angle_bisector A D B C)
  (h3 : divides_segment BD CD (2 : ℝ) (1 : ℝ))
  (h4 : median C E A B)
  (h5 : intersection P AD CE) :
  divides_segment AP DP (3 : ℝ) (1 : ℝ) := 
sorry

end median_divides_angle_bisector_in_given_ratio_l357_357058


namespace part1_general_formula_part2_find_d_l357_357093

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l357_357093


namespace area_of_region_covered_by_congruent_squares_l357_357967
-- Import necessary libraries

-- Define the problem conditions and the theorem
namespace MathProof

theorem area_of_region_covered_by_congruent_squares :
  let AB := 12
  let area_square_1 := AB * AB
  let congruent (s1 s2 : ℝ) := s1 = s2
  let H_at_D (region1_overlap : ℝ) := region1_overlap / 4
  2 * area_square_1 - H_at_D area_square_1 = 252 :=
by
  -- Definitions
  let AB := 12
  let area_square_1 := AB * AB
  let congruent (s1 s2 : ℝ) := s1 = s2
  let H_at_D (region1_overlap : ℝ) := region1_overlap / 4
  -- Theorem statement
  have area_square_2 : ℝ := area_square_1
  have region_overlap : ℝ := H_at_D area_square_1
  have total_area : ℝ := 2 * area_square_1 - region_overlap
  show total_area = 252, from sorry

end MathProof

end area_of_region_covered_by_congruent_squares_l357_357967


namespace value_of_r_l357_357125

theorem value_of_r (a b m p : ℝ) (h1 : ∀ x, x^2 - m * x + 3 = 0 → x = a ∨ x = b)
  (h2 : a * b = 3) : 
  let r := (a + 1 / b) * (b + 1 / a) in 
  r = 16 / 3 :=
by sorry

end value_of_r_l357_357125


namespace min_value_x_plus_2y_l357_357036

variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

theorem min_value_x_plus_2y (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 := 
  sorry

end min_value_x_plus_2y_l357_357036


namespace part1_part2_l357_357946

noncomputable def f (x : Real) : Real := 2 * Real.sin (Real.pi - x) + Real.cos (-x) - Real.sin (5 * Real.pi / 2 - x) + Real.cos (Real.pi / 2 + x)

theorem part1 (α : Real) (hα : α ∈ set.Ioo (0 : Real) Real.pi) (h : f α = 2 / 3) : Real.tan α = 2 * Real.sqrt 5 / 5 ∨ Real.tan α = -(2 * Real.sqrt 5 / 5) :=
by sorry

theorem part2 (α : Real) (h : f α = 2 * Real.sin α - Real.cos α + 3 / 4) : Real.sin α * Real.cos α = 7 / 32 :=
by sorry

end part1_part2_l357_357946


namespace cyclic_quadrilateral_l357_357044

theorem cyclic_quadrilateral {A B C D E F : Point} 
  (OnSidesD : ∃ (t : ℝ), D = A + t * (B - A))
  (OnSidesE : ∃ (t : ℝ), E = A + t * (C - A))
  (IntersectF : ∃ (s t : ℝ), F = B + s * (E - B) ∧ F = C + t * (D - C))
  (given_eq : distance B C ^ 2 = distance B D * distance B A + distance C E * distance C A) :
  concyclic {A, D, F, E} :=
by
  sorry

end cyclic_quadrilateral_l357_357044


namespace evaluate_expression_c_eq_4_l357_357732

theorem evaluate_expression_c_eq_4 :
  (4^4 - 4 * (4-1)^(4-1))^(4-1) = 3241792 :=
by
  sorry

end evaluate_expression_c_eq_4_l357_357732


namespace base7_addition_l357_357299

theorem base7_addition : (26:ℕ) + (245:ℕ) = 304 :=
  sorry

end base7_addition_l357_357299


namespace find_general_formula_and_d_l357_357086

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357086


namespace perimeter_is_24_l357_357747

def dist (p1 p2 : ℝ × ℝ) := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def perimeter_triangle (a b c : ℝ × ℝ) := 
dist a b + dist b c + dist c a

def pointA := (0, 6 : ℝ)
def pointB := (8, 0 : ℝ)
def pointC := (0, 0 : ℝ)

theorem perimeter_is_24 : 
perimeter_triangle pointA pointB pointC = 24 :=
by
sorry

end perimeter_is_24_l357_357747


namespace no_real_solutions_parabolas_intersection_l357_357338

theorem no_real_solutions_parabolas_intersection :
  ¬ ∃ a b c d : ℝ, (c ≥ a ∧ 
  (b = 3 * a^2 - 6 * a + 6) ∧ (b = -2 * a^2 + a + 3) ∧
  (d = 3 * c^2 - 6 * c + 6) ∧ (d = -2 * c^2 + c + 3) ∧
  (c - a ∈ ℝ)) := sorry

end no_real_solutions_parabolas_intersection_l357_357338


namespace grace_pennies_l357_357281

theorem grace_pennies :
  (coin_value = 10) ∧ (nickel_value = 5) ∧ (num_coins = 10) ∧ (num_nickels = 10) →
  total_pennies = 150 :=
by
  intros coin_value nickel_value num_coins num_nickels h
  let coin_pennies := num_coins * coin_value
  let nickel_pennies := num_nickels * nickel_value
  let total_pennies := coin_pennies + nickel_pennies
  exact h sorry

end grace_pennies_l357_357281


namespace cos_double_angle_l357_357378

-- defining our hypothesis
def sin_eq (θ : ℝ) : Prop := sin (π - θ) = 1 / 3

-- defining our theorem to prove
theorem cos_double_angle (θ : ℝ) (h : sin_eq θ) : cos (2 * θ) = 7 / 9 := 
by {
  -- skip the proof
  sorry
}

end cos_double_angle_l357_357378


namespace number_subtract_four_l357_357246

theorem number_subtract_four (x : ℤ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end number_subtract_four_l357_357246


namespace part1_part2_l357_357101

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357101


namespace compute_product_l357_357582

variable (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop := x^3 - 3 * x * y^2 = 2010
def condition2 (x y : ℝ) : Prop := y^3 - 3 * x^2 * y = 2000

theorem compute_product (h1 : condition1 x1 y1) (h2 : condition2 x1 y1)
    (h3 : condition1 x2 y2) (h4 : condition2 x2 y2)
    (h5 : condition1 x3 y3) (h6 : condition2 x3 y3) :
    (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 100 := 
    sorry

end compute_product_l357_357582


namespace number_of_valid_permutations_l357_357905

variables {A : Finset ℕ} (f : ℕ → ℕ)

def valid_permutation (f : ℕ → ℕ) (A : Finset ℕ) : Prop :=
  (∀ x ∈ A, f x ≠ x) ∧ (∀ x ∈ A, (f^[21]) x = x)

theorem number_of_valid_permutations :
  let A := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  ∃ f : ℕ → ℕ, valid_permutation f A ∧ 
  (number_of_such_permutations f A = 172800) :=
sorry

end number_of_valid_permutations_l357_357905


namespace sum_of_two_cubes_lt_1000_l357_357007

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357007


namespace inequality_proof_l357_357942

variable {n : ℕ} (x : Fin n → ℝ)
variable (h_n : 2 ≤ n) (h_pos : ∀ i, 0 < x i) (h_sum : Finset.univ.sum x = 1)

theorem inequality_proof :
  \big(∑ i, x i / real.sqrt (1 - x i)\big) ≥ (∑ i, real.sqrt (x i)) / real.sqrt (n - 1) := 
sorry

end inequality_proof_l357_357942


namespace badminton_wins_l357_357677

theorem badminton_wins :
  ∀ (f : Fin 33 → Int), (∀ i, f i ∈ ({25, 26, ..., 57} : Set Int)) → (∀ i, (f i = 1) ∨ (f i = -1))
  → Odd (List.sum (List.ofFn (fun i => i+25) : List Int)) :=
by
  intro f hf hs
  sorry

end badminton_wins_l357_357677


namespace statement_I_statement_II_l357_357754
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

def f (x y : ℝ) : ℤ := floor x + floor y - floor (x + y)

theorem statement_I (x y : ℕ) : f x y = 0 := sorry

theorem statement_II (x y : ℝ) : f x y ≥ 0 := sorry

end statement_I_statement_II_l357_357754


namespace intersection_A_B_l357_357496

def A := {x : ℕ | -2 < x ∧ x ≤ 1}
def B := {0, 1, 2}

theorem intersection_A_B : (A ∩ B) = {0, 1} :=
by
  sorry

end intersection_A_B_l357_357496


namespace cosine_identity_l357_357813

-- Define the given conditions
def cosine_expression (A B C : ℝ) : ℝ :=
  (Real.cos A) * (Real.cos B) + (Real.cos B) * (Real.cos C) + (Real.cos C) * (Real.cos A)

theorem cosine_identity
  (A B C : ℝ)
  (h1 : A = 3 * B)
  (h2 : B = 3 * C)
  (h3 : A + B + C = Real.pi) :
  cosine_expression A B C = -1 / 4 := 
sorry

end cosine_identity_l357_357813


namespace cosine_between_vectors_l357_357360

noncomputable def A : ℝ × ℝ × ℝ := (3, 3, -1)
noncomputable def B : ℝ × ℝ × ℝ := (1, 5, -2)
noncomputable def C : ℝ × ℝ × ℝ := (4, 1, 1)

def vec_sub (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (P.1 - Q.1, P.2 - Q.2, P.3 - Q.3)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def magnitude (u : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2 + u.3^2)
def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ := dot_product u v / (magnitude u * magnitude v)

theorem cosine_between_vectors : cos_angle (vec_sub B A) (vec_sub C A) = -8 / 9 :=
by
  sorry

end cosine_between_vectors_l357_357360


namespace probability_of_multiple_of_3_or_4_l357_357196

theorem probability_of_multiple_of_3_or_4 :
  let cards := {i : ℕ | 1 ≤ i ∧ i ≤ 30}
  let multiples_of_3 := {i ∈ cards | i % 3 = 0}
  let multiples_of_4 := {i ∈ cards | i % 4 = 0}
  let multiples_of_3_or_4 := {i ∈ cards | i % 3 = 0 ∨ i % 4 = 0}
  (multiples_of_3_or_4.card : ℚ) / (cards.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_multiple_of_3_or_4_l357_357196


namespace f_2005_equals_cos_l357_357926

noncomputable def f : ℕ → (Real → Real)
| 0       => (λ x => Real.sin x)
| (n + 1) => (λ x => (f n) x.derive)

theorem f_2005_equals_cos : ∀ x, f 2005 x = Real.cos x :=
by
  sorry

end f_2005_equals_cos_l357_357926


namespace log_x_64_eq_2_l357_357434

theorem log_x_64_eq_2 (x : ℝ) (h : log 8 (5 * x) = 2) : log x 64 = 2 :=
sorry

end log_x_64_eq_2_l357_357434


namespace digit_place_value_ratio_l357_357477

theorem digit_place_value_ratio :
  let d6_value := 6 * 10
  let d1_value := 1 * 0.1
  d6_value = 100 * d1_value := by
  -- The value of digit 6 is in the tens place: 6 * 10 = 60
  have h6 : d6_value = 60 := rfl
  -- The value of digit 1 is in the tenths place: 1 * 0.1 = 0.1
  have h1 : d1_value = 0.1 := rfl
  -- The ratio of the place values:
  have ratio : d6_value / d1_value = 100 := by
    calc d6_value / d1_value
      = 60 / 0.1 : by rw [h6, h1]
      = 600       : by norm_num
      = 100       : by norm_num
  exact ratio.symm

end digit_place_value_ratio_l357_357477


namespace circumcenter_rational_l357_357546

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l357_357546


namespace restaurant_discount_l357_357042

theorem restaurant_discount :
  let coffee_price := 6
  let cheesecake_price := 10
  let discount_rate := 0.25
  let total_price := coffee_price + cheesecake_price
  let discount := discount_rate * total_price
  let final_price := total_price - discount
  final_price = 12 := by
  sorry

end restaurant_discount_l357_357042


namespace finite_perfect_squares_l357_357917

noncomputable def finite_squares (a b : ℕ) : Prop :=
  ∃ (f : Finset ℕ), ∀ n, n ∈ f ↔ 
    ∃ (x y : ℕ), a * n ^ 2 + b = x ^ 2 ∧ a * (n + 1) ^ 2 + b = y ^ 2

theorem finite_perfect_squares (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  finite_squares a b :=
sorry

end finite_perfect_squares_l357_357917


namespace only_B_can_use_bisection_l357_357750

-- Definitions for the functions
def f_A (x : ℝ) : ℝ := x^4
def f_B (x : ℝ) : ℝ := tan x + 2
def f_C (x : ℝ) : ℝ := cos x - 1
def f_D (x : ℝ) : ℝ := |2^x - 3|

-- Conditions (monotonicity and range)
def is_monotonic (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x ≤ f y

def range_ℝ (f : ℝ → ℝ) : Prop := 
  ∀ y, ∃ x, f x = y

-- Proof statement
theorem only_B_can_use_bisection :
  ¬ is_monotonic f_A ∧ (∀ x, 0 ≤ f_A x) ∧
  is_monotonic f_B ∧ range_ℝ f_B ∧
  ¬ is_monotonic f_C ∧ (∀ x, f_C x ≤ 0) ∧
  ¬ is_monotonic f_D ∧ (∀ x, 0 ≤ f_D x) →
  true := by
  sorry

end only_B_can_use_bisection_l357_357750


namespace Rogers_expense_fraction_l357_357169

variables (B m s p : ℝ)

theorem Rogers_expense_fraction (h1 : m = 0.25 * (B - s))
                              (h2 : s = 0.10 * (B - m))
                              (h3 : p = 0.10 * (m + s)) :
  m + s + p = 0.34 * B :=
by
  sorry

end Rogers_expense_fraction_l357_357169


namespace largest_constant_exists_l357_357958

-- Given conditions in Lean 4
variable (a : Fin 17 → ℝ) -- where a_i represents the array of 17 positive numbers
def condition1 := (∑ i in Finset.finRange 17, (a i) ^ 2 = 24)
def condition2 (c : ℝ) := (∑ i in Finset.finRange 17, (a i) ^ 3 + ∑ i in Finset.finRange 17, a i < c)
def triangle_inequality (i j k : Fin 17) :=
  a i + a j > a k ∧ a i + a k > a j ∧ a j + a k > a i

-- Prove that for every i, j, k in the range, the inequality holds for the specific c value
theorem largest_constant_exists :
  ∃ c, c = 10 * Real.sqrt 24 ∧ 
    condition1 a ∧ condition2 a c → 
    ∀ i j k, 1 ≤ i < j < k ≤ 17 → triangle_inequality a i j k := 
by 
  sorry

end largest_constant_exists_l357_357958


namespace red_hat_small_nose_striped_shirt_gnomes_l357_357210

variable (total_gnomes : ℕ) (red_hat_ratio : ℚ) (blue_hat_gnomes : ℕ) 
variable (big_nose_ratio : ℚ) (blue_hat_big_nose_gnomes : ℕ) (red_hat_big_nose_gnomes : ℕ)
variable (striped_shirt_red_small_nose_gnomes : ℕ)

def num_red_hat_gnomes := (red_hat_ratio * total_gnomes)
def num_blue_hat_gnomes := (total_gnomes - num_red_hat_gnomes)
def num_big_nose_gnomes := (big_nose_ratio * total_gnomes)
def num_small_nose_gnomes := (total_gnomes - num_big_nose_gnomes)
def num_red_hat_small_nose_gnomes := (num_small_nose_gnomes - 1)
def striped_gnomes := (total_gnomes / 2)

theorem red_hat_small_nose_striped_shirt_gnomes 
  (total_gnomes = 28) (red_hat_ratio = 3/4) 
  (big_nose_ratio = 1/2) (blue_hat_big_nose_gnomes = 6) 
  (blue_hat_gnomes = (total_gnomes - num_red_hat_gnomes)) 
  (num_red_hat_small_nose_gnomes = 13) 
  : striped_shirt_red_small_nose_gnomes = 6 := by
  sorry

end red_hat_small_nose_striped_shirt_gnomes_l357_357210


namespace sum_of_two_cubes_lt_1000_l357_357009

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357009


namespace length_PQ_l357_357446

theorem length_PQ (AB BC CA AH : ℝ) (P Q : ℝ) : 
  AB = 7 → BC = 8 → CA = 9 → 
  AH = 3 * Real.sqrt 5 → 
  PQ = AQ - AP → 
  AQ = 7 * (Real.sqrt 5) / 3 → 
  AP = 9 * (Real.sqrt 5) / 5 → 
  PQ = Real.sqrt 5 * 8 / 15 :=
by
  intros hAB hBC hCA hAH hPQ hAQ hAP
  sorry

end length_PQ_l357_357446


namespace smallest_a_l357_357412

noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ (1, 2) then 2 - x 
  else if ∃ (m : ℤ), x ∈ (set.Ioo (2^m : ℝ) (2^(m+1) : ℝ)) then 2^(classical.some (exists_nat_pow3_neg_two x)) + 1 - x 
  else 0 -- This is to handle other values of x arbitrarily 

lemma f_property1 (x : ℝ) (hx : 0 < x) : f (2 * x) = 2 * f x := 
sorry

lemma f_property2 (x : ℝ) (hx : x ∈ set.Ioo 1 2) : f x = 2 - x := 
sorry

theorem smallest_a :
  ∃ a : ℝ, 0 < a ∧ f(a) = f(2020) ∧ ∀ b : ℝ, 0 < b ∧ f(b) = f(2020) → a ≤ b := 
sorry

end smallest_a_l357_357412


namespace percentage_of_alcohol_in_new_mixture_l357_357640

def original_solution_volume : ℕ := 11
def added_water_volume : ℕ := 3
def alcohol_percentage_original : ℝ := 0.42

def total_volume : ℕ := original_solution_volume + added_water_volume
def amount_of_alcohol : ℝ := alcohol_percentage_original * original_solution_volume

theorem percentage_of_alcohol_in_new_mixture :
  (amount_of_alcohol / total_volume) * 100 = 33 := by
  sorry

end percentage_of_alcohol_in_new_mixture_l357_357640


namespace distinct_construction_count_up_to_rotation_l357_357656

-- Definition: types for colors and cubes
inductive Color
| white
| blue

structure UnitCube where
  color : Color

def for_constructing_cube (cubes : List UnitCube) : Prop :=
  (cubes.filter (·.color = Color.white)).length = 5 ∧ 
  (cubes.filter (·.color = Color.blue)).length = 3

def count_orbits (set_of_cubes : Set (List UnitCube)) : Int :=
  sorry -- This would be a complex function using Burnside's Lemma

theorem distinct_construction_count_up_to_rotation :
  ∃ configs : Set (List UnitCube), for_constructing_cube configs → count_orbits configs = 3 :=
by
  sorry

end distinct_construction_count_up_to_rotation_l357_357656


namespace max_subset_2013_no_diff_9_l357_357757

theorem max_subset_2013_no_diff_9 : ∀ S ⊆ (finset.range (2013 + 1)), (∀ x y ∈ S, x ≠ y → abs (x - y) ≠ 9) → S.card ≤ 1008 :=
sorry

end max_subset_2013_no_diff_9_l357_357757


namespace compute_diameter_of_garden_roller_l357_357182

noncomputable def diameter_of_garden_roller (length : ℝ) (area_per_revolution : ℝ) (pi : ℝ) :=
  let radius := (area_per_revolution / (2 * pi * length))
  2 * radius

theorem compute_diameter_of_garden_roller :
  diameter_of_garden_roller 3 (66 / 5) (22 / 7) = 1.4 := by
  sorry

end compute_diameter_of_garden_roller_l357_357182


namespace g_self_inverse_if_one_l357_357321

variables (f : ℝ → ℝ) (symm_about : ∀ x, f (f x) = x - 1)

def g (b : ℝ) (x : ℝ) : ℝ := f (x + b)

theorem g_self_inverse_if_one (b : ℝ) :
  (∀ x, g f b (g f b x) = x) ↔ b = 1 := 
by
  sorry

end g_self_inverse_if_one_l357_357321


namespace ratio_of_max_min_l357_357809

variables (a b c : EuclideanSpace ℝ (Fin 2))

def condition1 : Prop := ‖a‖ = 2 ∧ ‖b‖ = 2 ∧ (a ⬝ b) = 2
def condition2 : Prop := (c ⬝ (a + 2 • b - 2 • c)) = 2

theorem ratio_of_max_min (h1 : condition1 a b) (h2 : condition2 a b c) : ∃ M m : ℝ, (∀ x, (x = ‖a - c‖) → (x ≤ M) ∧ (x ≥ m)) ∧ (M / m = Real.sqrt 6) :=
sorry

end ratio_of_max_min_l357_357809


namespace bacteria_states_l357_357456

-- Definition of transformation rules
def transform_rr (r b : ℕ) : ℕ × ℕ := (r - 2, b + 1)
def transform_bb (r b : ℕ) : ℕ × ℕ := (r + 4, b - 2)
def transform_rb (r b : ℕ) : ℕ × ℕ := (r + 2, b - 1)

-- Theorem statement based on conditions and question
theorem bacteria_states (r b : ℕ) :
  let n := r + b in
  set_of (λ (r b : ℕ), r + b = n) = { (n, 0), (n-2, 1), (n-4, 2), ... } :=
sorry

end bacteria_states_l357_357456


namespace simplify_expression_l357_357944

variable (a b c x : ℝ)

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def p (x a b c : ℝ) : ℝ :=
  (x - a)^3/(a - b)*(a - c) + a*x +
  (x - b)^3/(b - a)*(b - c) + b*x +
  (x - c)^3/(c - a)*(c - b) + c*x

theorem simplify_expression (h : distinct a b c) :
  p x a b c = a + b + c + 3*x + 1 := by
  sorry

end simplify_expression_l357_357944


namespace polygon_number_of_sides_l357_357026

-- Define the given conditions
def each_interior_angle (n : ℕ) : ℕ := 120

-- Define the property to calculate the number of sides
def num_sides (each_exterior_angle : ℕ) : ℕ := 360 / each_exterior_angle

-- Statement of the problem
theorem polygon_number_of_sides : num_sides (180 - each_interior_angle 6) = 6 :=
by
  -- Proof is omitted
  sorry

end polygon_number_of_sides_l357_357026


namespace distinct_nonzero_digits_sum_l357_357436

theorem distinct_nonzero_digits_sum
  (x y z w : Nat)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (hxy : x ≠ y)
  (hxz : x ≠ z)
  (hxw : x ≠ w)
  (hyz : y ≠ z)
  (hyw : y ≠ w)
  (hzw : z ≠ w)
  (h1 : w + x = 10)
  (h2 : y + w = 9)
  (h3 : z + x = 9) :
  x + y + z + w = 18 :=
sorry

end distinct_nonzero_digits_sum_l357_357436


namespace initial_punch_amount_l357_357142

theorem initial_punch_amount (P : ℝ) (h1 : 16 = (P / 2 + 2 + 12)) : P = 4 :=
by
  sorry

end initial_punch_amount_l357_357142


namespace relationship_among_abc_l357_357124

noncomputable def a := Real.log 0.8 / Real.log 0.5
noncomputable def b := Real.log 0.8 / Real.log 1.1
noncomputable def c := 1.1 ^ 0.8

theorem relationship_among_abc : b < a ∧ a < c :=
by {
  sorry
}

end relationship_among_abc_l357_357124


namespace smallest_n_for_probability_l357_357167

theorem smallest_n_for_probability :
  let n := 10 in
  (∫ (x : ℝ) in 0..n, ∫ (y : ℝ) in (0..n), ∫ (z : ℝ) in (0..n),
  if abs(x - y) ≥ 1 ∧ abs(y - z) ≥ 1 ∧ abs(z - x) ≥ 1 then 1 else 0) / (n ^ 3) > 1 / 2 :=
by
  let n := 10
  have numerator := ∫ (x : ℝ) in 0..n, ∫ (y : ℝ) in (0..n), ∫ (z : ℝ) in (0..n),
    if abs (x - y) ≥ 1 ∧ abs (y - z) ≥ 1 ∧ abs (z - x) ≥ 1 then 1 else 0
  have denominator := n ^ 3
  have probability := numerator / denominator
  show probability > 1 / 2
  sorry

end smallest_n_for_probability_l357_357167


namespace total_weight_of_8_bags_total_sales_amount_of_qualified_products_l357_357573

-- Definitions
def deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def standard_weight_per_bag : ℤ := 450
def threshold : ℤ := 4
def price_per_bag : ℤ := 3

-- Part 1: Total weight of the 8 bags of laundry detergent
theorem total_weight_of_8_bags : 
  8 * standard_weight_per_bag + deviations.sum = 3598 := 
by
  sorry

-- Part 2: Total sales amount of qualified products
theorem total_sales_amount_of_qualified_products : 
  price_per_bag * (deviations.filter (fun x => abs x ≤ threshold)).length = 18 := 
by
  sorry

end total_weight_of_8_bags_total_sales_amount_of_qualified_products_l357_357573


namespace cyclists_no_point_b_l357_357216

theorem cyclists_no_point_b (v1 v2 t d : ℝ) (h1 : v1 = 35) (h2 : v2 = 25) (h3 : t = 2) (h4 : d = 30) :
  ∀ (ta tb : ℝ), ta + tb = t ∧ ta * v1 + tb * v2 < d → false :=
by
  sorry

end cyclists_no_point_b_l357_357216


namespace relationship_between_abc_l357_357769

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem relationship_between_abc : a > b ∧ b > c :=
by
  sorry

end relationship_between_abc_l357_357769


namespace main_statement_l357_357903

def Point : Type := Real × Real × Real

def OX : Point := (1, 0, 0)
def OY : Point := (0, 1, 0)
def OZ : Point := (0, 0, 1)

structure PointOnOZ (c : Real) :=
  (x : Real)
  (y : Real)
  (z : Real)
  (h : x = 0 ∧ y = 0 ∧ z = c)

structure PointOnOX :=
  (u : Real)
  (x : Real) := u
  (y : Real) := 0
  (z : Real) := 0

structure PointOnOY :=
  (v : Real)
  (x : Real) := 0
  (y : Real) := v
  (z : Real) := 0

/-- Definition of the problem-/
noncomputable
def locus_of_P (P : Point) (U : PointOnOX) (V : PointOnOY) (C : PointOnOZ c) : Prop :=
  let PU := (P.1 - U.u, P.2, P.3)
  let PV := (P.1, P.2 - V.v, P.3)
  let PC := (P.1, P.2, P.3 - c)
  (PU.1 * PV.1 + PU.2 * PV.2 + PU.3 * PV.3 = 0) ∧
  (PU.1 * PC.1 + PU.2 * PC.2 + PU.3 * PC.3 = 0) ∧
  (PV.1 * PC.1 + PV.2 * PC.2 + PV.3 * PC.3 = 0)

/-- Main statement to be proven -/
noncomputable
theorem main_statement (P : Point) (U : PointOnOX) (V : PointOnOY) (C : PointOnOZ c) :
  locus_of_P P U V C ↔ (P.1 = 0 ∧ P.2 = 0 ∧ (P.3 = 0 ∨ P.3 = 2 * c)) := 
sorry

end main_statement_l357_357903


namespace calculate_value_l357_357701

theorem calculate_value : (2200 - 2090)^2 / (144 + 25) = 64 := 
by
  sorry

end calculate_value_l357_357701


namespace Calvin_number_sum_of_digits_l357_357704

theorem Calvin_number_sum_of_digits :
  let a₀ := 1
  let a : ℕ → ℕ := λ n, if n = 0 then a₀ else 3 * a (n - 1) + 5
  let a₁₀ := a 10
  let sum_of_digits_base_9 := 3 + 4 + 4 + 4 + 4 + 2
in sum_of_digits_base_9 = 21 :=
by
  sorry

end Calvin_number_sum_of_digits_l357_357704


namespace smallest_pos_int_y_satisfies_congruence_l357_357364

theorem smallest_pos_int_y_satisfies_congruence :
  ∃ y : ℕ, (y > 0) ∧ (26 * y + 8) % 16 = 4 ∧ ∀ z : ℕ, (z > 0) ∧ (26 * z + 8) % 16 = 4 → y ≤ z :=
sorry

end smallest_pos_int_y_satisfies_congruence_l357_357364


namespace solve_for_m_l357_357764

theorem solve_for_m (a_0 a_1 a_2 a_3 a_4 a_5 m : ℝ)
  (h1 : (x : ℝ) → (x + m)^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5)
  (h2 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 32) :
  m = 2 :=
sorry

end solve_for_m_l357_357764


namespace range_of_m_l357_357792

noncomputable def circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

noncomputable def line (m x y : ℝ) : Prop :=
  m * x - y - 5 * m + 4 = 0

theorem range_of_m (m : ℝ) :
  (∃ Q : ℝ × ℝ, circle Q.1 Q.2 ∧ ∃ P : ℝ × ℝ, line m P.1 P.2 ∧ ∃ C : ℝ × ℝ, C = (-1, 0) ∧ ∃ (Q : ℝ × ℝ), (Q.1, Q.2) ∈ {p | circle p.1 p.2} ∧
  (∀ P : ℝ × ℝ, line m P.1 P.2 → ∃ θ : ℝ, θ = 30 → (distance (C, P) ≤ 4)) → 
  0 ≤ m ∧ m ≤ 12 / 5 :=
sorry

end range_of_m_l357_357792


namespace shopkeeper_gain_percentage_l357_357670

variable (c : ℝ)

-- Conditions
def marked_price (c : ℝ) : ℝ := c + 0.35 * c
def selling_price (mp : ℝ) : ℝ := mp - 0.20 * mp
def gain (sp cp : ℝ) : ℝ := sp - cp
def percentage_gain (g cp : ℝ) : ℝ := (g / cp) * 100

-- Statement to prove
theorem shopkeeper_gain_percentage :
  let c := 100 in
  let mp := marked_price c in
  let sp := selling_price mp in
  let g := gain sp c in
  percentage_gain g c = 8 := by
sorry

end shopkeeper_gain_percentage_l357_357670


namespace shortest_chord_l357_357420

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop := 2 * m * x - y - 8 * m - 3 = 0
noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 6)^2 = 25

theorem shortest_chord (m : ℝ) :
  (∃ x y, line_eq m x y ∧ circle_eq x y) →
  m = 1 / 6 :=
by sorry

end shortest_chord_l357_357420


namespace tom_total_payment_l357_357588

theorem tom_total_payment :
  let apples_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let bananas_cost := 12 * 30
  let grapes_cost := 7 * 45
  let cherries_cost := 4 * 80
  apples_cost + mangoes_cost + oranges_cost + bananas_cost + grapes_cost + cherries_cost = 2250 :=
by
  sorry

end tom_total_payment_l357_357588


namespace find_triangle_areas_l357_357181

variables (ABC : Type) [convex_quadrilateral ABC]
variables (A B C D O : ABC)
variables (area : ABC → ℝ)

-- Conditions:
-- 1. The total area of the quadrilateral ABCD is 28.
-- 2. Area(AOB) = 2 * Area(COD).
-- 3. Area(BOC) = 18 * Area(DOA).

axiom h1 : area A + area B + area C + area D = 28
axiom h2 : area A = 2 * area C
axiom h3 : area B = 18 * area D

theorem find_triangle_areas :
  area A = 4.48 ∧ 
  area B = 20.16 ∧ 
  area C = 2.24 ∧ 
  area D = 1.12 := 
by {
  have h : 4 * area C + 18 * area D + area C + area D = 28, -- From h1, h2, h3
    sorry,
  let x := 28 / 25,
  have hC : area C = 2.24, --  x * 2 = 2.24
    sorry,
  have hD : area D = 1.12, -- x 
    sorry,
  have hB : area B = 20.16, -- x * 18 = 20.16
    sorry,
  have hA : area A = 4.48, -- x * 4 = 4.48
    sorry,
  exact ⟨hA, hB, hC, hD⟩
}

end find_triangle_areas_l357_357181


namespace sales_tax_difference_l357_357697

theorem sales_tax_difference (price : ℝ) (rate1 rate2 : ℝ) : 
  rate1 = 0.085 → rate2 = 0.07 → price = 50 → 
  (price * rate1 - price * rate2) = 0.75 := 
by 
  intros h_rate1 h_rate2 h_price
  rw [h_rate1, h_rate2, h_price] 
  simp
  sorry

end sales_tax_difference_l357_357697


namespace triangle_sets_l357_357689

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def candidates : List (List ℕ) :=
  [[1, 2, 3], [2, 3, 4], [3, 4, 5], [3, 6, 9]]

theorem triangle_sets :
  let valid_sets :=
    candidates.filter (fun s => is_triangle s[0] s[1] s[2])
  valid_sets = [[2, 3, 4], [3, 4, 5]] :=
by
  sorry

end triangle_sets_l357_357689


namespace M_is_incenter_of_triangle_l357_357486

theorem M_is_incenter_of_triangle
  (A B C M O : Point)
  (h1 : M ∈ Triangle ABC)
  (h2 : ∠ BMC = 90 + 1/2 * ∠ BAC)
  (h3 : O = Circumcenter (Triangle BMC))
  (h4 : M ∈ Line AM) : 
  M = Incenter (Triangle ABC) :=
by
  sorry

end M_is_incenter_of_triangle_l357_357486


namespace max_distance_sqrt2_l357_357497

noncomputable def max_distance (z : ℂ) (hz : complex.abs z = 1) : ℝ :=
  complex.abs (((1 : ℂ) + (complex.I)) * z + 2 * complex.conj(z))

theorem max_distance_sqrt2 (z : ℂ) (hz : complex.abs z = 1) :
  (∃ w : ℂ, w = ((1 : ℂ) + (complex.I)) * z + 2 * complex.conj(z) ∧ complex.abs w = real.sqrt 2) :=
begin
  sorry
end

end max_distance_sqrt2_l357_357497


namespace time_to_pass_telegraph_post_l357_357063

noncomputable def distance := 64 : ℕ -- length of the train in meters
noncomputable def speed_kmph := 46 : ℕ -- speed of the train in kmph
noncomputable def conversion_factor := 5 / 18 : ℚ -- conversion factor from kmph to m/s
noncomputable def speed_mps := (speed_kmph : ℚ) * conversion_factor -- speed of the train in m/s

theorem time_to_pass_telegraph_post :
  (distance : ℚ) / speed_mps ≈ 5.007 :=
by
  sorry

end time_to_pass_telegraph_post_l357_357063


namespace distance_center_is_12_l357_357558

-- Define the side length of the square and the radius of the circle
def side_length_square : ℝ := 5
def radius_circle : ℝ := 1

-- The center path forms a smaller square inside the original square
-- with side length 3 units
def side_length_smaller_square : ℝ := side_length_square - 2 * radius_circle

-- The perimeter of the smaller square, which is the path length that
-- the center of the circle travels
def distance_center_travel : ℝ := 4 * side_length_smaller_square

-- Prove that the distance traveled by the center of the circle is 12 units
theorem distance_center_is_12 : distance_center_travel = 12 := by
  -- the proof is skipped
  sorry

end distance_center_is_12_l357_357558


namespace smallest_integer_in_set_l357_357876

theorem smallest_integer_in_set :
  ∀ (n : ℤ), (n + 6 > 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → n = -1 :=
by
  intros n h
  sorry

end smallest_integer_in_set_l357_357876


namespace probability_different_colors_is_one_half_l357_357581

noncomputable def probability_two_different_colors : ℚ :=
  let total_chips := 6 + 3 in
  let prob_blue_yellow := (6 / total_chips) * (3 / (total_chips - 1)) in
  let prob_yellow_blue := (3 / total_chips) * (6 / (total_chips - 1)) in
  prob_blue_yellow + prob_yellow_blue

theorem probability_different_colors_is_one_half :
  probability_two_different_colors = 1 / 2 :=
by
  sorry

end probability_different_colors_is_one_half_l357_357581


namespace arrange_composite_numbers_l357_357512

theorem arrange_composite_numbers (n : ℕ) (h : n ≥ 6) :
  ∃ (lst : list ℕ), (∀ x ∈ lst, composite x ∧ x ≤ n) ∧
    (∀ (a b : ℕ), a ∈ lst.tail → b ∈ lst → (b ≠ a) → (a, b) ∈ list.zip lst.tail lst →
    has_common_factor_gt_one a b) :=
sorry

end arrange_composite_numbers_l357_357512


namespace area_of_square_KLMN_is_25_l357_357526

-- Given a square ABCD with area 25
def ABCD_area_is_25 : Prop :=
  ∃ s : ℝ, (s * s = 25)

-- Given points K, L, M, and N forming isosceles right triangles with the sides of the square
def isosceles_right_triangles_at_vertices (A B C D K L M N : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    (a = b) ∧ (c = d) ∧
    (K - A)^2 + (B - K)^2 = (A - B)^2 ∧  -- AKB
    (L - B)^2 + (C - L)^2 = (B - C)^2 ∧  -- BLC
    (M - C)^2 + (D - M)^2 = (C - D)^2 ∧  -- CMD
    (N - D)^2 + (A - N)^2 = (D - A)^2    -- DNA

-- Given that KLMN is a square
def KLMN_is_square (K L M N : ℝ) : Prop :=
  (K - L)^2 + (L - M)^2 = (M - N)^2 + (N - K)^2

-- Proving that the area of square KLMN is 25 given the conditions
theorem area_of_square_KLMN_is_25 (A B C D K L M N : ℝ) :
  ABCD_area_is_25 → isosceles_right_triangles_at_vertices A B C D K L M N → KLMN_is_square K L M N → ∃s, s * s = 25 :=
by
  intro h1 h2 h3
  sorry

end area_of_square_KLMN_is_25_l357_357526


namespace locus_of_H_l357_357389

noncomputable theory
open_locale classical

variables {α : Type*} [linear_ordered_field α] 
variables {V : Type*} [inner_product_space α V]

-- Define the triangle vertices
variables {A B C P H : V}

-- Define the relevant planes
variables (plane_ABC : affine_subspace α V) (plane_PBC plane_PCA plane_PAB : affine_subspace α V)

-- Orthocenter condition
variable (H_is_orthocenter : H ∈ plane_ABC ∧ orthocenter H A B C)

-- Distances
variables (PH : α) (h : α) (h_A : α) (h_B : α) (h_C : α)

-- Angle conditions
variables (angle_a angle_b angle_c : α)

-- Function to get perpendicular distance
def perpendicular_distance (p : V) (plane : affine_subspace α V) : α := sorry

theorem locus_of_H (h_min_cond : PH < h_A * cos angle_a) (h_B_cond : PH < h_B * cos angle_b) (h_C_cond : PH < h_C * cos angle_c) :
  ∃ D E F : V, (H ∈ triangle DEF) :=
sorry

end locus_of_H_l357_357389


namespace present_ages_l357_357038

-- Definition of the present ages
variables (A B C D : ℕ)

-- Given Conditions
def condition1: Prop := C + 10 = 3 * (A + 10)
def condition2: Prop := A = 2 * (B - 10)
def condition3: Prop := A = B + 12
def condition4: Prop := B = D + 5
def condition5: Prop := D = C / 2

-- Goal Statement
theorem present_ages : condition1 A B C D → condition2 A B C D → condition3 A B C D → condition4 A B C D → condition5 A B C D → 
  A = 88 ∧ B = 76 ∧ C = 142 ∧ D = 71 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof will be here
  sorry

end present_ages_l357_357038


namespace no_ten_rational_numbers_l357_357342

theorem no_ten_rational_numbers (r : ℚ) :
  ¬∃ (s : Finset ℚ), s.card = 10 ∧ (∀ a b ∈ s, a ≠ b → (a * b) ∈ ℤ) ∧ (∀ a b c ∈ s, a ≠ b → b ≠ c → a ≠ c → ¬(a * b * c) ∈ ℤ) :=
by
  sorry

end no_ten_rational_numbers_l357_357342


namespace car_miles_per_tankful_on_highway_l357_357278

variables (miles_per_gallon_highway miles_per_gallon_city : ℝ)
variables (miles_per_tank_city tank_size : ℝ)
variables (miles_per_tank_highway : ℝ)

-- Conditions from the problem
hypothesis (h1: miles_per_gallon_city = 18)
hypothesis (h2: miles_per_tank_city = 336)
hypothesis (h3: miles_per_gallon_highway = miles_per_gallon_city + 6)
hypothesis (h4: tank_size = miles_per_tank_city / miles_per_gallon_city)

-- Define the target, miles per tankful on the highway
def highway_miles_per_tank : ℝ := tank_size * miles_per_gallon_highway

-- Conclusion to prove
theorem car_miles_per_tankful_on_highway : highway_miles_per_tank = 448.08 := by
  sorry

end car_miles_per_tankful_on_highway_l357_357278


namespace chime_2500_date_l357_357654
open Nat

def clock_chimes_per_half_hour : ℕ := 1
def clock_chimes_per_hour : ℕ → ℕ := λ h, h
def half_hours_per_day : ℕ := 48 -- each half hour in 24 hours
def hours_per_day : List ℕ := List.range 13 -- from 0 to 12

def chimes_before_midnight : ℕ :=
  let chimes_at_half_hours := 10 -- from 3 PM to midnight (15 half-hours)
  let chimes_at_hours := (12 - 3 + 1) * (3 + 12) / 2 -- sum from 3 to 12
  chimes_at_half_hours + chimes_at_hours + 12 -- including midnight chimes

def daily_chimes : ℕ :=
  half_hours_per_day + hours_per_day.sum

def total_chimes (days : ℕ) (chimes_before_start : ℕ) : ℕ :=
  chimes_before_start + days * daily_chimes

def date_of_2500th_chime : String :=
  let chimes_on_start_day := chimes_before_midnight
  let remaining_chimes := 2500 - chimes_on_start_day
  let full_days := remaining_chimes / daily_chimes
  if (remaining_chimes % daily_chimes = 0) then
    "April 7, 2003"
  else
    "April 8, 2003"

theorem chime_2500_date : date_of_2500th_chime = "April 8, 2003" :=
begin
  sorry
end

end chime_2500_date_l357_357654


namespace num_subsets_containing_6_l357_357828

theorem num_subsets_containing_6 : 
  (∃ (subset : set (fin 6)), 6 ∈ subset ∧ fintype.card {s : set (fin 6) | s ∈ subset}) = 32 :=
sorry

end num_subsets_containing_6_l357_357828


namespace greatest_three_digit_number_condition_l357_357604

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end greatest_three_digit_number_condition_l357_357604


namespace area_of_highest_points_l357_357666

theorem area_of_highest_points (u g : ℝ) (h_u : 0 < u) (h_g : 0 < g) : 
  ∃ k : ℝ, (∀ α: ℝ, 0 ≤ α ∧ α ≤ 2 * π →  
  let x := (u^2 / (2*g)) * Real.sin (2*α),
      y := (u^2 / (4*g)) * (1 - Real.cos (2*α)) in
  ∃ (u g : ℝ), 
  2 * π * (u^2 / (2 * g)) * (u^2 / (4 * g)) = k * (u^4 / g^2)) ∧ 
  k = π/4 :=
sorry

end area_of_highest_points_l357_357666


namespace angle_KDA_eq_BCA_or_KBA_l357_357955

variables {A B C D K : Type*}
variables [InnerProductSpace ℝ (EuclideanGeometry ℝ)]
variables (A B C D K : EuclideanGeometry ℝ)
variables (convex_quadrilateral : ConvexQuadrilateral A B C D)
variables (K_on_AC : K ∈ LineSegment A C)
variables (KD_eq_DC : (dist K D) = (dist D C))
variables (angle_BAC_eq_half_angle_KDC : ∠ B A C = ∠ K D C / 2)
variables (angle_DAC_eq_half_angle_KBC : ∠ D A C = ∠ K B C / 2)

theorem angle_KDA_eq_BCA_or_KBA :
  ∠ K D A = ∠ B C A ∨ ∠ K D A = ∠ K B A :=
sorry

end angle_KDA_eq_BCA_or_KBA_l357_357955


namespace divisible_sequence_l357_357634

theorem divisible_sequence (a : ℕ) : 
    ∃ n : ℕ, n = 2 * a - 1 ∧ 
             (∀ m : ℕ, ∃ k : ℕ, k = n ^ (nat.pow n m) + 1 ∧ a ∣ k) :=
by
    sorry

end divisible_sequence_l357_357634


namespace range_of_k_l357_357866

theorem range_of_k (k : ℝ) : (∀ x ∈ Icc (0 : ℝ) (Real.pi / 2), Real.exp x * Real.sin x ≥ k * x) →
(k ≤ 1) :=
by
  sorry

end range_of_k_l357_357866


namespace cyclic_and_ratio_l357_357493

section Geometry

variables {ABC : Type} [triangle ABC] 
variables {A B C D E F P Q M N : Point} [incircle_touches BC AC AB at D E F]
variables [point_on_arc P of (arc EF) that_not_contains D] 
variables [BP_intersects_incircle_at Q, second_intersect_point]
variables [lines_intersect_EP EQ_meet_BC M N]

theorem cyclic_and_ratio :
  (cyclic P F B M) ∧ (EM / EN = BF / BP) :=
by
  sorry

end Geometry

end cyclic_and_ratio_l357_357493


namespace part1_part2_l357_357102

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357102


namespace fraction_meaningful_l357_357031

theorem fraction_meaningful (x : ℝ) : (x - 2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end fraction_meaningful_l357_357031


namespace compound_interest_correct_l357_357519

-- Conditions
def P : ℝ := 140
def r : ℝ := 0.20
def n : ℝ := 2
def t : ℝ := 1

-- The compound interest formula
def compound_interest (P r n t : ℝ) : ℝ := P * (1 + r / n) ^ (n * t)

-- Proof statement
theorem compound_interest_correct :
  compound_interest P r n t = 169.40 :=
by
  sorry

end compound_interest_correct_l357_357519


namespace cos_fourth_minus_sin_fourth_eq_cos_double_l357_357162

-- Define the proposition to be proven
theorem cos_fourth_minus_sin_fourth_eq_cos_double (θ : ℝ) : 
  cos(θ)^4 - sin(θ)^4 = cos(2*θ) :=
sorry

end cos_fourth_minus_sin_fourth_eq_cos_double_l357_357162


namespace imaginary_part_of_i_div_1_plus_i_l357_357638

theorem imaginary_part_of_i_div_1_plus_i : 
  complex.im (complex.div complex.I (1 + complex.I)) = 1 / 2 :=
by
  sorry

end imaginary_part_of_i_div_1_plus_i_l357_357638


namespace seating_arrangements_valid_count_l357_357586

-- Define the problem with the given conditions.

variables (a1 a2 b1 b2 c1 c2 : Type)

-- Definition of the conditions for valid seating arrangements.

-- The function calculating the number of valid seating arrangements.

theorem seating_arrangements_valid_count :
  let row1 := {a1, b1, c2},
      row2 := {a2, b2, c1},
      no_siblings_next_or_front : Prop :=
        ∀ x y, x ∈ row1 → y ∈ row1 → x ≠ y → ∀ x' y', x' ∈ row2 → y' ∈ row2 → x' ≠ y' →
        (¬ (x = a1 ∧ x' = a2) ∧ ...)
  in
  (∃ row1 row2 : set Type, row1 = {a1, b1, c2} ∧ row2 = {a2, b2, c1} ∧ no_siblings_next_or_front) →
  valid_arrangements = 96 :=
sorry

end seating_arrangements_valid_count_l357_357586


namespace initial_punch_amount_l357_357144

-- Given conditions
def initial_punch : ℝ
def final_punch : ℝ := 16
def cousin_drink_half (x : ℝ) := x / 2
def mark_add (x : ℝ) := x + 4
def sally_drink (x : ℝ) := x - 2
def mark_final_addition := 12

-- Problem statement in Lean 4
theorem initial_punch_amount (initial_punch : ℝ) : 
  let after_final_addition := final_punch - mark_final_addition
  let before_sally_drink := after_final_addition + 2
  let before_second_refill := before_sally_drink - 4
  let initial_punch := cousin_drink_half (before_second_refill)
  initial_punch = 4 := 
sorry

end initial_punch_amount_l357_357144


namespace value_of_f_log_20_l357_357950

variable (f : ℝ → ℝ)
variable (h₁ : ∀ x : ℝ, f (-x) = -f x)
variable (h₂ : ∀ x : ℝ, f (x - 2) = f (x + 2))
variable (h₃ : ∀ x : ℝ, x > -1 ∧ x < 0 → f x = 2^x + 1/5)

theorem value_of_f_log_20 : f (Real.log 20 / Real.log 2) = -1 := sorry

end value_of_f_log_20_l357_357950


namespace harmonic_mean_pairs_count_l357_357566

theorem harmonic_mean_pairs_count :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x < y ∧ (2 * x * y) / (x + y) = 12^10) ∧
  (∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x < y ∧ (2 * x * y) / (x + y) = 12^10 → (x, y) ∈ {(a, b) | a < b} → 
  finite {p : ℕ × ℕ | p.1 ≠ p.2 ∧ p.1 < p.2 ∧ (2 * p.1 * p.2) / (p.1 + p.2) = 12^10}) ∧
  fintype.card {p : ℕ × ℕ | p.1 ≠ p.2 ∧ p.1 < p.2 ∧ (2 * p.1 * p.2) / (p.1 + p.2) = 12^10} = 409 := 
by
  sorry

end harmonic_mean_pairs_count_l357_357566


namespace digit_D_eq_9_l357_357188

-- Define digits and the basic operations on 2-digit numbers
def is_digit (n : ℕ) : Prop := n < 10
def tens (n : ℕ) : ℕ := n / 10
def units (n : ℕ) : ℕ := n % 10
def two_digit (a b : ℕ) : ℕ := 10 * a + b

theorem digit_D_eq_9 (A B C D : ℕ):
  is_digit A → is_digit B → is_digit C → is_digit D →
  (two_digit A B) + (two_digit C B) = two_digit D A →
  (two_digit A B) - (two_digit C B) = A →
  D = 9 :=
by sorry

end digit_D_eq_9_l357_357188


namespace number_of_monomials_in_list_l357_357194

def is_monomial (e : Expr) : Prop :=
  match e with
  | Expr.const c 0 _ => true -- constants
  | Expr.var 0 => true       -- single variable
  | Expr.mul c Expr.var 0 => true  -- constant multiplied by a variable
  | Expr.const c _ => true   -- negative constants
  | Expr.add _ _ => false    -- addition
  | Expr.div _ Expr.var _ => false -- division by variable
  | Expr.neg _ => true       -- negative numbers
  end

def count_monomials (expressions : List Expr) : Nat :=
  expressions.count is_monomial

theorem number_of_monomials_in_list 
  (expressions : List Expr) (count : Nat) :
  expressions = [Expr.const 0 0, 
                Expr.add (Expr.mul 2 (Expr.var 0)) (Expr.const (-1) 0), 
                Expr.var 0, 
                Expr.div (Expr.const 1 0) (Expr.var 0), 
                Expr.neg (Expr.const 2 3),
                Expr.div (Expr.add (Expr.var 0) (Expr.neg (Expr.var 1))) (Expr.const 1 2),
                Expr.div (Expr.mul 2 (Expr.var 0)) (Expr.const 1 5)] -> 
  count = 4 :=
by
  sorry

end number_of_monomials_in_list_l357_357194


namespace cos_double_angle_l357_357014

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 2) = 1 / 2) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l357_357014


namespace find_general_formula_and_d_l357_357082

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357082


namespace a_n_formula_d_value_l357_357109

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l357_357109


namespace inequality_hold_l357_357398

variable {a b c : ℝ}

theorem inequality_hold 
  (h₁ : a > b)
  (h₂ : c ∈ ℝ) : 
  (a/(c^2 + 1) > b/(c^2 + 1)) :=
by
  sorry

end inequality_hold_l357_357398


namespace batsman_average_17th_innings_l357_357275

theorem batsman_average_17th_innings:
  ∀ (A : ℝ), 
  (16 * A + 85 = 17 * (A + 3)) →
  (A + 3 = 37) :=
by
  intros A h
  sorry

end batsman_average_17th_innings_l357_357275


namespace initial_punch_amount_l357_357141

theorem initial_punch_amount (P : ℝ) (h1 : 16 = (P / 2 + 2 + 12)) : P = 4 :=
by
  sorry

end initial_punch_amount_l357_357141


namespace circle_radius_l357_357280

theorem circle_radius (A C : ℝ) (h1 : A / C = 25) (h2 : A = real.pi * (r * r)) (h3 : C = 2 * real.pi * r) : r = 50 := 
by
  sorry

end circle_radius_l357_357280


namespace ratio_equal_one_l357_357049

open Real EuclideanGeometry

-- Definitions for the problem in Lean
variables {A B C R E : Point}
variables {circle : Circle}
variables {theta beta : ℝ}

-- Conditions given in the problem
def conditions : Prop :=
  diameter circle A B ∧
  radius circle C R ∧
  perpendicular C R A B ∧
  (∃ E : Point, lies_on AC E ∧ lies_on BR E ∧ angle AEB = beta)

-- The theorem statement
theorem ratio_equal_one (h : conditions) : 
  area (triangle C R E) / area (triangle A B E) = 1 :=
sorry

end ratio_equal_one_l357_357049


namespace concyclic_points_l357_357495

-- Define points A, B, C, D on a circle
variables {A B C D S E F : Type}
variables [has_circle A B C D]

-- Define point S on the circle as the midpoint of the arc AB that does not contain C and D
def is_midpoint_arc (S A B : Type) (C D : Type) : Prop :=
  midpoint_arc A B S ∧ ¬contains_arc A B C D S

-- Define intersections E and F of lines SD and SC with AB
def is_intersection (SD SC AB E F : Type) : Prop :=
  intersects (line SD) (line AB) E ∧ intersects (line SC) (line AB) F

-- Con cyclicity of points C, D, E, and F
theorem concyclic_points (A B C D S E F : Type) 
  [is_midpoint_arc S A B C D] [is_intersection SD SC AB E F] : 
  is_concyclic C D E F :=
by sorry

end concyclic_points_l357_357495


namespace exists_n_h_n_right_or_decreasing_perimeter_l357_357509

def triangle (A B C : Type) : Prop :=
  -- Definition of a non-degenerate triangle
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def is_right_triangle (A B C : Type) : Prop :=
  -- Definition of a right triangle
  -- (This is a placeholder; the actual mathematical definition would depend on the coordinates or angles involved)
  sorry

def perimeter (A B C : Type) : ℝ :=
  -- Function to compute the perimeter of triangle ABC
  sorry

def h (A B C : Type) : (Type × Type × Type) :=
  -- Transformation to the orthic triangle
  sorry 

theorem exists_n_h_n_right_or_decreasing_perimeter (A B C : Type)
  (ABC_non_degenerate: triangle A B C):
  ∃ n : ℕ, let ith_triangle := (nat.iterate (λ T, h T) n (A, B, C)) in
           (is_right_triangle (ith_triangle.fst.fst) (ith_triangle.fst.snd) (ith_triangle.snd)) ∨
           (perimeter (ith_triangle.fst.fst) (ith_triangle.fst.snd) (ith_triangle.snd) < perimeter A B C) :=
begin
  sorry
end

end exists_n_h_n_right_or_decreasing_perimeter_l357_357509


namespace sum_of_two_cubes_lt_1000_l357_357002

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357002


namespace possible_values_a_l357_357861

def A : Set ℝ := {-1, 2}
def B (a : ℝ) : Set ℝ := {x | a * x^2 = 2 ∧ a ≥ 0}

def whale_swallowing (S T : Set ℝ) : Prop :=
S ⊆ T ∨ T ⊆ S

def moth_eating (S T : Set ℝ) : Prop :=
(∃ x, x ∈ S ∧ x ∈ T) ∧ ¬(S ⊆ T) ∧ ¬(T ⊆ S)

def valid_a (a : ℝ) : Prop :=
whale_swallowing A (B a) ∨ moth_eating A (B a)

theorem possible_values_a :
  {a : ℝ | valid_a a} = {0, 1/2, 2} :=
sorry

end possible_values_a_l357_357861


namespace intersection_dot_product_eq_distance_sq_l357_357470

theorem intersection_dot_product_eq_distance_sq 
  (y_A1 y_B1 y_A2 y_B2 : ℝ)
  (hy_A1_pos : 0 ≤ y_A1)
  (hy_A2_pos : 0 ≤ y_A2)
  (hy_B1_pos : 0 ≤ y_B1)
  (hy_B2_pos : 0 ≤ y_B2)
  (p : ℝ)
  (hp : p = y_A1 * y_B1 ∨ p = y_A2 * y_B2) : 
  let S := -(y_A1 * y_B2),
      T := -(y_A2 * y_B1),
      P := p in
  S * T = P^2 :=
by {
  let S := -(y_A1 * y_B2),
  let T := -(y_A2 * y_B1),
  let P := p,
  sorry
}

end intersection_dot_product_eq_distance_sq_l357_357470


namespace box_surface_area_l357_357646

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end box_surface_area_l357_357646


namespace volleyball_tournament_l357_357636

theorem volleyball_tournament (teams : Finset ℕ) (H : ∀ s : Finset ℕ, s.card = 55 → ∃ t ∈ s, (s.erase t).filter (λ x, (x, t) ∈ losses ∨ (t, x) ∈ losses).card ≤ 4) :
  ∃ t ∈ teams, (teams.erase t).filter (λ x, (x, t) ∈ losses ∨ (t, x) ∈ losses).card ≤ 4 := 
sorry

end volleyball_tournament_l357_357636


namespace ball_drawing_prob_exp_l357_357452

-- Define the variables and conditions of the problem
def numBalls := 4
def numRedBalls := 1
def numGreenBalls := 1
def numYellowBalls := 2
def probDrawFirstRed := 1 / 4
def probDrawFirstGreenThenRed := (1 / 4) * (1 / 3)
def p_xi_0 := probDrawFirstRed + probDrawFirstGreenThenRed

def exp_xi := 0 * (1 / 3) + 1 * (1 / 3) + 2 * (1 / 3)

-- The theorem statement
theorem ball_drawing_prob_exp :
  p_xi_0 = 1 / 3 ∧ exp_xi = 1 := by
  sorry

end ball_drawing_prob_exp_l357_357452


namespace find_asterisk_value_l357_357247

theorem find_asterisk_value : 
  (∃ x : ℕ, (x / 21) * (x / 189) = 1) → x = 63 :=
by
  intro h
  sorry

end find_asterisk_value_l357_357247


namespace geometric_sequence_b_value_l357_357205

theorem geometric_sequence_b_value (r b : ℝ) (h1 : 120 * r = b) (h2 : b * r = 27 / 16) (hb_pos : b > 0) : b = 15 :=
sorry

end geometric_sequence_b_value_l357_357205


namespace polynomial_no_real_roots_l357_357940

open Real

noncomputable def polynomial (c : Fin (n - 1) → ℝ) (n : ℕ) (x : ℝ) : ℝ :=
  let P := (Finset.range (n - 1)).sum (λ i, (-1) ^ (i + 1) * c ⟨i, sorry⟩ * x ^ i) 
  2 * x ^ n + P + 2

theorem polynomial_no_real_roots (n : ℕ) (c : Fin (n - 1) → ℝ)
  (h_even : Even n) (h_sum : (Finset.range (n - 1)).sum (λ i, |c ⟨i, sorry⟩ - 1|) < 1)
  : ∀ x : ℝ, polynomial c n x ≠ 0 := 
sorry

end polynomial_no_real_roots_l357_357940


namespace find_a_l357_357800

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2 : ℝ) * a * x^3 - (3 / 2 : ℝ) * x^2 + (3 / 2 : ℝ) * a^2 * x

theorem find_a (a : ℝ) (h_max : ∀ x : ℝ, f a x ≤ f a 1) : a = -2 :=
sorry

end find_a_l357_357800


namespace find_k_l357_357941

-- Definition of the vertices and conditions
variables {t k : ℝ}
def A : (ℝ × ℝ) := (0, 3)
def B : (ℝ × ℝ) := (0, k)
def C : (ℝ × ℝ) := (t, 10)
def D : (ℝ × ℝ) := (t, 0)

-- Condition that the area of the quadrilateral is 50 square units
def area_cond (height base1 base2 : ℝ) : Prop :=
  50 = (1 / 2) * height * (base1 + base2)

-- Stating the problem in Lean
theorem find_k
  (ht : t = 5)
  (hk : k > 3) 
  (t_pos : t > 0)
  (area : area_cond t (k - 3) 10) :
  k = 13 :=
  sorry

end find_k_l357_357941


namespace impossible_to_identify_all_genuine_diamonds_l357_357208

theorem impossible_to_identify_all_genuine_diamonds :
  ∀ (diamonds : Fin 100 → bool), 
  (∃ (count_genuine : Fin 100 → ℕ), (∀ i, count_genuine i = if diamonds i then 1 else 0) ∧ (∑ i, count_genuine i = 50)) →
  (∀ (expert : Fin 100 → Fin 100 → Fin 100 → (Fin 100 × Fin 100)), 
  (∀ (a b c : Fin 100), (let (x, y) := expert a b c in (x ≠ y) ∧ (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c))) →
  (¬ (∃ (determine_genuine : Fin 100 → bool), (∀ i, determine_genuine i = diamonds i)))) :=
by sorry

end impossible_to_identify_all_genuine_diamonds_l357_357208


namespace flash_catches_ace_l357_357681

theorem flash_catches_ace (v x y t : ℝ) 
  (h₀ : v > 0) 
  (h₁ : x > 1) 
  (h₂ : t = y^2 / (v * (x^2 - 1))) :
  x^2 * v * t = (x^2 * y^2) / (x^2 - 1) := 
by {
  rw [h₂],
  field_simp [ne_of_gt (sub_pos_of_lt h₁)],
  ring,
  }

end flash_catches_ace_l357_357681


namespace arithmetic_sequence_fifth_term_l357_357887

theorem arithmetic_sequence_fifth_term :
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  a5 = 19 :=
by
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  show a5 = 19
  sorry

end arithmetic_sequence_fifth_term_l357_357887


namespace sin_cos_difference_l357_357377

theorem sin_cos_difference (α : ℝ) (h1 : sin (2 * α) = 1/2) (h2 : 0 < α ∧ α < π / 2) :
  sin α - cos α = ±(sqrt 2 / 2) :=
sorry

end sin_cos_difference_l357_357377


namespace humpty_points_concyclic_l357_357664

/-- Defining the acute, non-isosceles triangle ABC and points A_1, B_1, C_1 with specified angle properties. -/
structure Triangle :=
(A B C : Point)
(acute_non_isosceles : True)  -- Just a placeholder for the condition
(A1 : Point)
(B1 : Point)
(C1 : Point)
(G : Point)
(A1_angle_condition : ∠(A1, A, B) = ∠(A1, B, C) ∧ ∠(A1, A, C) = ∠(A1, C, B))
(B1_angle_condition : ∠(B1, B, A) = ∠(B1, A, C) ∧ ∠(B1, B, C) = ∠(B1, C, A))
(C1_angle_condition : ∠(C1, C, A) = ∠(C1, A, B) ∧ ∠(C1, C, B) = ∠(C1, B, A))
(G_centroid : G = centroid A B C)

/-- Statement of the problem to prove that points A_1, B_1, C_1, G are concyclic. -/
theorem humpty_points_concyclic (T : Triangle) : 
  ∃ (circle : Circle), 
    T.A1 ∈ circle ∧ T.B1 ∈ circle ∧ T.C1 ∈ circle ∧ T.G ∈ circle :=
sorry

end humpty_points_concyclic_l357_357664


namespace part1_part2_l357_357098

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l357_357098


namespace sum_sequence_product_l357_357884

noncomputable def term_sequence (n : ℕ) : ℕ → ℝ 
| k := n / (2 * k)

noncomputable def transformed_sum (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n-1), term_sequence n (i+1) * term_sequence n (i+2)

theorem sum_sequence_product (n : ℕ) (hn : n ≥ 2) :
  transformed_sum n = n * (n - 1) / 4 :=
sorry

end sum_sequence_product_l357_357884


namespace min_value_of_f_l357_357232

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l357_357232


namespace min_colors_needed_l357_357620

theorem min_colors_needed (n : ℕ) (h : n + n.choose 2 ≥ 12) : n = 5 :=
sorry

end min_colors_needed_l357_357620


namespace min_value_of_f_l357_357233

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l357_357233


namespace volume_of_triangular_prism_l357_357703

-- Definitions
variables (A_face d : ℝ)

-- Main theorem statement
theorem volume_of_triangular_prism (h : A_face > 0) (h' : d > 0) :
  ∃ V_prism : ℝ, V_prism = (1 / 2 : ℝ) * A_face * d :=
by
  use (1 / 2 : ℝ) * A_face * d
  sorry

end volume_of_triangular_prism_l357_357703


namespace solve_for_x_l357_357963

-- Definition of the conditions under which the radicals are valid
def conditions (x : ℝ) : Prop := 
  16 + 8 * x ≥ 0 ∧ 
  5 + x ≥ 0

-- Definition of the equation to be satisfied
def equation (x : ℝ) : Prop :=
  (sqrt (9 - sqrt (16 + 8 * x)) + sqrt (5 - sqrt (5 + x)) = 3 + sqrt 5)

-- The proof problem statement
theorem solve_for_x :
  ∃ x, conditions x ∧ equation x := 
begin
  use 4,
  split,
  { -- Proof of conditions for x = 4
    split;
    norm_num },
  { -- Proof of equation for x = 4
    norm_num,
    rw [sqrt_sub, sqrt_sub],
    { simp [sqrt_sq_eq_abs, abs_of_pos]; norm_num },
    all_goals { norm_num },
    sorry, -- The rest of the proof would typically go here
  }
end

end solve_for_x_l357_357963


namespace pablo_pages_l357_357524

theorem pablo_pages (P : ℕ) (h1 : ∀ n : ℕ, n * 1 = n) 
                    (h2 : 12 * P = 1800) : P = 150 := 
by
    have h3 : 1800 / 12 = 150, from sorry,
    calc
        P = 1800 / 12 : sorry
        ... = 150     : h3

end pablo_pages_l357_357524


namespace evaluate_expression_l357_357349

-- Conditions
def sum1 : ℕ := 2 + 4 + 6 + 8
def sum2 : ℕ := 1 + 3 + 5 + 7

-- Main Statement
theorem evaluate_expression :
  (sum1 / sum2 + sum2 / sum1) = (41 / 20) :=
by
  have h1 : sum1 = 20 := by norm_num
  have h2 : sum2 = 16 := by norm_num
  sorry

end evaluate_expression_l357_357349


namespace loss_percentage_is_20_l357_357262

def cost_price : ℝ := 1500
def selling_price : ℝ := 1200
def loss_percentage (cp sp : ℝ) : ℝ := ((cp - sp) / cp) * 100

theorem loss_percentage_is_20 :
  loss_percentage cost_price selling_price = 20 := 
sorry

end loss_percentage_is_20_l357_357262


namespace initial_punch_amount_l357_357143

-- Given conditions
def initial_punch : ℝ
def final_punch : ℝ := 16
def cousin_drink_half (x : ℝ) := x / 2
def mark_add (x : ℝ) := x + 4
def sally_drink (x : ℝ) := x - 2
def mark_final_addition := 12

-- Problem statement in Lean 4
theorem initial_punch_amount (initial_punch : ℝ) : 
  let after_final_addition := final_punch - mark_final_addition
  let before_sally_drink := after_final_addition + 2
  let before_second_refill := before_sally_drink - 4
  let initial_punch := cousin_drink_half (before_second_refill)
  initial_punch = 4 := 
sorry

end initial_punch_amount_l357_357143


namespace minimum_tangent_length_l357_357024

theorem minimum_tangent_length
  (a b : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 3 = 0)
  (h_symmetry : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x + b * y + 6 = 0) :
  ∃ t : ℝ, t = 2 :=
by sorry

end minimum_tangent_length_l357_357024


namespace subsets_containing_six_l357_357837

theorem subsets_containing_six : 
  let S := {1, 2, 3, 4, 5, 6}
  in set.count (λ s, 6 ∈ s ∧ s ⊆ S) = 32 :=
sorry

end subsets_containing_six_l357_357837


namespace find_general_formula_and_d_l357_357085

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l357_357085


namespace polygon_sides_l357_357027

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 120) :
  ∃ (n : ℕ), n = 6 :=
begin
  use 6,
  sorry  -- Proof goes here
end

end polygon_sides_l357_357027


namespace find_x_l357_357444

theorem find_x (x : ℝ) (h : (2 * x) / 16 = 25) : x = 200 :=
sorry

end find_x_l357_357444


namespace max_and_min_W_l357_357772

noncomputable def W (x y z : ℝ) : ℝ := 2 * x + 6 * y + 4 * z

theorem max_and_min_W {x y z : ℝ} (h1 : x + y + z = 1) (h2 : 3 * y + z ≥ 2) (h3 : 0 ≤ x ∧ x ≤ 1) (h4 : 0 ≤ y ∧ y ≤ 2) :
  ∃ (W_max W_min : ℝ), W_max = 6 ∧ W_min = 4 :=
by
  sorry

end max_and_min_W_l357_357772


namespace long_furred_and_brown_dogs_l357_357259

-- Define the total number of dogs.
def total_dogs : ℕ := 45

-- Define the number of long-furred dogs.
def long_furred_dogs : ℕ := 26

-- Define the number of brown dogs.
def brown_dogs : ℕ := 22

-- Define the number of dogs that are neither long-furred nor brown.
def neither_long_furred_nor_brown_dogs : ℕ := 8

-- Prove that the number of dogs that are both long-furred and brown is 11.
theorem long_furred_and_brown_dogs : 
  (long_furred_dogs + brown_dogs) - (total_dogs - neither_long_furred_nor_brown_dogs) = 11 :=
by
  sorry

end long_furred_and_brown_dogs_l357_357259


namespace f_of_18_over_49_l357_357774

theorem f_of_18_over_49 
  (f : ℚ+ → ℚ+)
  (h1 : ∀ a b : ℚ+, f (a * b) = f a * f b)
  (h2 : ∀ p : ℚ+, p.prime → f p = p ^ 2) :
  f (18 / 49) = 324 / 2401 := by
  sorry

end f_of_18_over_49_l357_357774


namespace keira_guarantees_capture_l357_357510

theorem keira_guarantees_capture (k : ℕ) (n : ℕ) (h_k_pos : 0 < k) (h_n_cond : n > k / 2023) :
    k ≥ 1012 :=
sorry

end keira_guarantees_capture_l357_357510


namespace exponent_for_decimal_digits_l357_357431

theorem exponent_for_decimal_digits :
  ∃ x : ℤ, (10^4 * 3.456789)^x has 18 decimal places to the right → x = 1 :=
by sorry

end exponent_for_decimal_digits_l357_357431


namespace margin_expression_l357_357034

variable (C S M : ℝ)
variable (n : ℕ)

theorem margin_expression (h : M = (C + S) / n) : M = (2 * S) / (n + 1) :=
sorry

end margin_expression_l357_357034


namespace multiples_highest_average_l357_357341

/--
Given the sets of whole numbers representing multiples of 6, 7, 8, 9, and 10 respectively,
within the range 1 to 121, prove that the set of multiples of 10 has the highest average.
-/
theorem multiples_highest_average :
  let multiples_of_6 := { n ∈ Finset.range 122 | n % 6 = 0 }
  let multiples_of_7 := { n ∈ Finset.range 122 | n % 7 = 0 }
  let multiples_of_8 := { n ∈ Finset.range 122 | n % 8 = 0 }
  let multiples_of_9 := { n ∈ Finset.range 122 | n % 9 = 0 }
  let multiples_of_10 := { n ∈ Finset.range 122 | n % 10 = 0 }
  let avg_6 := multiples_of_6.sum / multiples_of_6.card
  let avg_7 := multiples_of_7.sum / multiples_of_7.card
  let avg_8 := multiples_of_8.sum / multiples_of_8.card
  let avg_9 := multiples_of_9.sum / multiples_of_9.card
  let avg_10 := multiples_of_10.sum / multiples_of_10.card
  avg_10 > avg_6 ∧ avg_10 > avg_7 ∧ avg_10 > avg_8 ∧ avg_10 > avg_9 := sorry

end multiples_highest_average_l357_357341


namespace polar_eq_circle_l357_357571

-- Definition of the problem condition in polar coordinates
def polar_eq (ρ : ℝ) : Prop := ρ = 1

-- Definition of the assertion we want to prove: that it represents a circle
def represents_circle (ρ : ℝ) (θ : ℝ) : Prop := (ρ = 1) → ∃ (x y : ℝ), (ρ = 1) ∧ (x^2 + y^2 = 1)

theorem polar_eq_circle : ∀ (ρ θ : ℝ), polar_eq ρ → represents_circle ρ θ :=
by
  intros ρ θ hρ hs
  sorry

end polar_eq_circle_l357_357571


namespace solve_ordered_pairs_l357_357363

theorem solve_ordered_pairs (a b : ℕ) (h : a^2 + b^2 = ab * (a + b)) : 
  (a, b) = (1, 1) ∨ (a, b) = (1, 1) :=
by 
  sorry

end solve_ordered_pairs_l357_357363


namespace math_problem_solution_l357_357710

theorem math_problem_solution : ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ x^y - 1 = y^x ∧ 2*x^y = y^x + 5 ∧ x = 2 ∧ y = 2 :=
by {
  sorry
}

end math_problem_solution_l357_357710


namespace parallelogram_area_l357_357971

theorem parallelogram_area (base height : ℕ) (h_base : base = 5) (h_height : height = 3) :
  base * height = 15 :=
by
  -- Here would be the proof, but it is omitted per instructions
  sorry

end parallelogram_area_l357_357971


namespace minimum_black_squares_on_7x8_board_l357_357730

-- Define the board and the notion of neighboring squares
structure Board (m n : ℕ) :=
(squares : Fin m → Fin n → Bool) -- Bool is used to represent black (true) or white (false)

def is_neighbor (m n : ℕ) (x1 y1 x2 y2 : Fin m) (x : Fin n) (y : Fin m) : Prop :=
(abs (x1.val - x2.val) = 1 ∧ y1 = y2)
 ∨ (abs (y1.val - y2.val) = 1 ∧ x1 = x2)

def neighboring_black_squares (m n : ℕ) (b : Board m n) : ℕ :=
Finset.card {b | ∃ x y x' y' : Fin m, is_neighbor m n x y x' y' ∧ b.squares x y = true ∧ b.squares x' y' = true}

theorem minimum_black_squares_on_7x8_board :
  ∃(b: Board 7 8), (∀ i j, Finset.subset (Finset.ofFinset_univ $ Finset.filter (λ ij, ij.isSome ∧ neighboring_black_squares b = ij.some)),
  ∑ x y, if b.squares x y then 1 else 0 = 10 := 
begin
  sorry
end

end minimum_black_squares_on_7x8_board_l357_357730


namespace partial_fraction_decomposition_l357_357352

theorem partial_fraction_decomposition (x : ℝ) :
  (5 * x - 3) / (x^2 - 5 * x - 14) = (32 / 9) / (x - 7) + (13 / 9) / (x + 2) := by
  sorry

end partial_fraction_decomposition_l357_357352


namespace sum_first_n_terms_l357_357442

noncomputable def a (n : ℕ) : ℕ := 2^n + 2 * n - 1

theorem sum_first_n_terms (n : ℕ) : 
  (∑ k in Finset.range n, a (k + 1)) = 2^(n + 1) + n^2 - 2 :=
by {
  sorry
}

end sum_first_n_terms_l357_357442


namespace tan_alpha_plus_beta_eq_two_l357_357765

theorem tan_alpha_plus_beta_eq_two (α β : ℝ)
  (h₁ : tan α = 1)
  (h₂ : 3 * sin β = sin (2 * α + β)) :
  tan (α + β) = 2 :=
by
  sorry

end tan_alpha_plus_beta_eq_two_l357_357765


namespace alpha_plus_beta_value_tan_2alpha_plus_2beta_does_not_exist_l357_357761

noncomputable def alpha : ℝ := sorry
noncomputable def beta : ℝ := sorry

noncomputable def tan_alpha : ℝ := Real.tan alpha
noncomputable def tan_beta : ℝ := Real.tan beta

axiom alpha_beta_interval : 0 < alpha ∧ alpha < π ∧ 0 < beta ∧ beta < π
axiom tan_alpha_beta_roots : tan_alpha^2 - 5 * tan_alpha + 6 = 0 ∧ tan_beta^2 - 5 * tan_beta + 6 = 0

theorem alpha_plus_beta_value : α + β = 3 * π / 4 := sorry

theorem tan_2alpha_plus_2beta_does_not_exist : ¬ ∃ x, Real.tan (2 * (α + β)) = x := sorry

end alpha_plus_beta_value_tan_2alpha_plus_2beta_does_not_exist_l357_357761


namespace correct_assignment_statement_l357_357613

theorem correct_assignment_statement (n m : ℕ) : 
  ¬ (4 = n) ∧ ¬ (n + 1 = m) ∧ ¬ (m + n = 0) :=
by
  sorry

end correct_assignment_statement_l357_357613


namespace intersection_of_sets_l357_357906

theorem intersection_of_sets :
  let A := {x : ℝ | -1 < x ∧ x < 2},
      B := {x : ℝ | 0 < x}
  in A ∩ B = {x : ℝ | 0 < x ∧ x < 2} :=
by {
  sorry
}

end intersection_of_sets_l357_357906


namespace min_value_of_quadratic_l357_357228

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l357_357228


namespace integral_of_3x_minus_2_cos_5x_l357_357629

noncomputable def indefinite_integral : Real → Real :=
  ∫ (3 * x - 2) * cos (5 * x) dx

theorem integral_of_3x_minus_2_cos_5x :
  indefinite_integral = λ x, (1/5) * (3 * x - 2) * sin (5 * x) + (3/25) * cos (5 * x) + C :=
by
  sorry

end integral_of_3x_minus_2_cos_5x_l357_357629


namespace minimum_value_ineq_l357_357918

noncomputable def problem_statement (a b c : ℝ) (h : a + b + c = 3) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) → (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2)

theorem minimum_value_ineq (a b c : ℝ) (h : a + b + c = 3) : problem_statement a b c h :=
  sorry

end minimum_value_ineq_l357_357918


namespace find_x_value_l357_357476

theorem find_x_value:
  ∃ x : ℝ, 6 * x + (7 * x + 10) + (2 * x + 10) + x = 360 ∧ x = 21.25 :=
begin
  use 21.25,
  split,
  {
    -- Proof of the equation
    linarith,
  },
  {
    -- Proof that the value of x is as given
    refl,
  }
end

end find_x_value_l357_357476


namespace gina_running_problem_l357_357759

noncomputable def gina_distance_between_speeds (d₁ d₂ : ℝ) (t₁ t₂ standing_time running_rate : ℝ) : ℝ := 
  d₂ - d₁

theorem gina_running_problem :
  let d₁ := (1 / 2 : ℝ) in -- distance when average speed was 7.5 min/km
  let d₂ := (3 : ℝ) in -- distance when average speed was 7.08333 min/km
  let standing_time := (1 / 4 : ℝ) in -- 15 seconds in minutes
  let running_rate := (7 : ℝ) in -- 7 minutes per kilometre
  let avg_speed₁ := (7.5 : ℝ) in -- 7 minutes 30 seconds in minutes
  let avg_speed₂ := (85 / 12 : ℝ) in -- 7 minutes 5 seconds in minutes
  gina_distance_between_speeds d₁ d₂ standing_time running_rate = 2.5 := 
by 
  -- Here would be the proof steps
  sorry

end gina_running_problem_l357_357759


namespace all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l357_357663

theorem all_palindromes_divisible_by_11 : 
  (∀ a b : ℕ, 1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 →
    (1001 * a + 110 * b) % 11 = 0 ) := sorry

theorem probability_palindrome_divisible_by_11 : 
  (∀ (palindromes : ℕ → Prop), 
  (∀ n, palindromes n ↔ ∃ (a b : ℕ), 
  1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 ∧ 
  n = 1001 * a + 110 * b) → 
  (∀ n, palindromes n → n % 11 = 0) →
  ∃ p : ℝ, p = 1) := sorry

end all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l357_357663


namespace subsets_with_six_l357_357846

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l357_357846


namespace integer_solutions_yk_eq_x2_plus_x_l357_357740

-- Define the problem in Lean
theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ (x y : ℤ), y^k = x^2 + x → (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
by
  sorry

end integer_solutions_yk_eq_x2_plus_x_l357_357740


namespace distance_P_to_y_axis_l357_357883

-- Definition: Given point P in Cartesian coordinates
def P : ℝ × ℝ := (-3, -4)

-- Definition: Function to calculate distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := abs p.1

-- Theorem: The distance from point P to the y-axis is 3
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 :=
by
  sorry

end distance_P_to_y_axis_l357_357883


namespace circumcenter_rational_l357_357545

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} :
  ∃ (x y : ℚ), 
    ((x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2) ∧
    ((x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2) :=
sorry

end circumcenter_rational_l357_357545


namespace smallest_positive_period_max_min_value_interval_l357_357413

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin (x + Real.pi / 3))^2 - (Real.cos x)^2 + (Real.sin x)^2

theorem smallest_positive_period : (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

theorem max_min_value_interval :
  (∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), 
    f x ≤ 3 / 2 ∧ f x ≥ 0 ∧ 
    (f (-Real.pi / 6) = 0) ∧ 
    (f (Real.pi / 6) = 3 / 2)) :=
by sorry

end smallest_positive_period_max_min_value_interval_l357_357413


namespace nat_num_sat_eq_l357_357909

theorem nat_num_sat_eq (x : ℕ) : 
  ∃ n : ℕ, n = 5 ∧ (∀ x ∈ {0, 1, 2, 3, 4}, ⌊-1.77 * x⌋ = ⌊-1.77⌋ * x) :=
by
  sorry

end nat_num_sat_eq_l357_357909


namespace sum_of_cubes_of_roots_l357_357699

noncomputable def cube_root_23 := real.cbrt 23
noncomputable def cube_root_73 := real.cbrt 73
noncomputable def cube_root_123 := real.cbrt 123

theorem sum_of_cubes_of_roots :
  let r := cube_root_23,
      s := cube_root_73,
      t := cube_root_123 in
  r^3 + s^3 + t^3 = 219.75 :=
by
  sorry

end sum_of_cubes_of_roots_l357_357699


namespace length_of_garden_l357_357621

variables (w l : ℕ)

-- Definitions based on the problem conditions
def length_twice_width := l = 2 * w
def perimeter_eq_900 := 2 * l + 2 * w = 900

-- The statement to be proved
theorem length_of_garden (h1 : length_twice_width w l) (h2 : perimeter_eq_900 w l) : l = 300 :=
sorry

end length_of_garden_l357_357621


namespace question_1_question_2_l357_357037

variable {α : Type*} [LinearOrderedField α]

theorem question_1 {a b c C B : α}
  (h1 : cos C = 3 / 4)
  (h2 : B = 2 * C) :
  b / c = 3 / 2 := sorry

theorem question_2 {a b c : α}
  (h1 : cos (c : α) = 3 / 4)
  (h2 : c = sqrt 3)
  (h3 : a * b = 2) :
  abs (a - b) = sqrt 2 := sorry

end question_1_question_2_l357_357037


namespace carmen_distance_from_start_l357_357315

noncomputable def distance_carmen (start : ℝ × ℝ) : ℝ :=
  let B := (start.1, start.2 + 3)
  let C := (B.1 + 8 * Real.cos (Real.pi / 4), B.2 + 8 * Real.sin (Real.pi / 4))
  Real.sqrt ((C.1 - start.1)^2 + (C.2 - start.2)^2)

theorem carmen_distance_from_start : 
  distance_carmen (0, 0) = Real.sqrt (73 + 24 * Real.sqrt 2) :=
by
  sorry

end carmen_distance_from_start_l357_357315


namespace border_area_correct_l357_357293

noncomputable def area_of_border (poster_height poster_width border_width : ℕ) : ℕ :=
  let framed_height := poster_height + 2 * border_width
  let framed_width := poster_width + 2 * border_width
  (framed_height * framed_width) - (poster_height * poster_width)

theorem border_area_correct :
  area_of_border 12 16 4 = 288 :=
by
  rfl

end border_area_correct_l357_357293


namespace cost_of_360_songs_in_2005_l357_357450

theorem cost_of_360_songs_in_2005 :
  ∀ (c : ℕ), (200 * (c + 32) = 360 * c) → 360 * c / 100 = 144 :=
by
  assume c : ℕ
  assume h : 200 * (c + 32) = 360 * c
  sorry

end cost_of_360_songs_in_2005_l357_357450


namespace arithmetic_progression_ratio_conditions_l357_357437

noncomputable def arithmetic_progression_ratio_sequence (a : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

theorem arithmetic_progression_ratio_conditions (a : ℕ → ℝ) (k : ℝ) :
  arithmetic_progression_ratio_sequence a k →
  k ≠ 0 ∧
  (∀ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ b ≠ 1) → arithmetic_progression_ratio_sequence (λ n, a * b^n + c) k) :=
by
  intro h
  split
  { sorry }
  { intros a b c h1
    sorry }

end arithmetic_progression_ratio_conditions_l357_357437


namespace part1_part2_l357_357119

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l357_357119


namespace area_of_path_is_675_l357_357253

def rectangular_field_length : ℝ := 75
def rectangular_field_width : ℝ := 55
def path_width : ℝ := 2.5

def area_of_path : ℝ :=
  let new_length := rectangular_field_length + 2 * path_width
  let new_width := rectangular_field_width + 2 * path_width
  let area_with_path := new_length * new_width
  let area_of_grass_field := rectangular_field_length * rectangular_field_width
  area_with_path - area_of_grass_field

theorem area_of_path_is_675 : area_of_path = 675 := by
  sorry

end area_of_path_is_675_l357_357253


namespace three_digit_divisible_by_11_l357_357959

theorem three_digit_divisible_by_11 {x y z : ℕ} 
  (h1 : 0 ≤ x ∧ x < 10) 
  (h2 : 0 ≤ y ∧ y < 10) 
  (h3 : 0 ≤ z ∧ z < 10) 
  (h4 : x + z = y) : 
  (100 * x + 10 * y + z) % 11 = 0 := 
by 
  sorry

end three_digit_divisible_by_11_l357_357959


namespace money_left_is_correct_l357_357147

-- Definitions of the given problem conditions as constants
def spent_on_clothes : ℝ := 28
def percent_spent_on_clothes : ℝ := 0.45
def percent_spent_on_books : ℝ := 0.25
def percent_spent_on_snacks : ℝ := 0.10

-- The total amount of allowance calculation
def total_allowance : ℝ := spent_on_clothes / percent_spent_on_clothes

-- The total percentage spent and percentage left calculation
def percent_spent : ℝ := percent_spent_on_clothes + percent_spent_on_books + percent_spent_on_snacks
def percent_left : ℝ := 1 - percent_spent

-- The amount of money left in the allowance
def money_left : ℝ := total_allowance * percent_left

-- The proof statement:
theorem money_left_is_correct : money_left = 12.44 := by
  -- Placeholder for proof
  sorry

end money_left_is_correct_l357_357147


namespace ratio_of_spinsters_to_cats_l357_357200

def spinsters := 22
def cats := spinsters + 55

theorem ratio_of_spinsters_to_cats : (spinsters : ℝ) / (cats : ℝ) = 2 / 7 := 
by
  sorry

end ratio_of_spinsters_to_cats_l357_357200


namespace probability_of_same_suit_l357_357211

-- Definitions for the conditions
def total_cards : ℕ := 52
def suits : ℕ := 4
def cards_per_suit : ℕ := 13
def total_draws : ℕ := 2

-- Definition of factorial for binomial coefficient calculation
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Calculation of the probability
def prob_same_suit : ℚ :=
  let ways_to_choose_2_cards_from_52 := binomial_coeff total_cards total_draws
  let ways_to_choose_2_cards_per_suit := binomial_coeff cards_per_suit total_draws
  let total_ways_to_choose_2_same_suit := suits * ways_to_choose_2_cards_per_suit
  total_ways_to_choose_2_same_suit / ways_to_choose_2_cards_from_52

theorem probability_of_same_suit :
  prob_same_suit = 4 / 17 :=
by
  sorry

end probability_of_same_suit_l357_357211


namespace tan_series_sin_identity_l357_357619

theorem tan_series_sin_identity (x : ℝ) (hx : |tan x| < 1) :
  (\frac{1 - tan x + tan x^2 - tan x^3 + tan x^4 - ...}{1 + tan x + tan x^2 + tan x^3 + tan x^4 + ...}) = 1 + sin(2 * x) →
  ∃ k : ℤ, x = k * π :=
sorry

end tan_series_sin_identity_l357_357619


namespace part_a_l357_357257

theorem part_a : 
  ∃ (pairs : list (ℕ × ℕ)), 
    (∀ (x : ℕ × ℕ), x ∈ pairs → x.1 + x.2 ∈ 
      {p ∈ primes | p ∈ {5, 7, 11, 13, 19, 23}}) ∧
    pairs.length = 6 ∧ 
    (∀ (x y : ℕ × ℕ), x ≠ y → x.1 ≠ x.2 ∧ y.1 ≠ y.2) := 
sorry

end part_a_l357_357257


namespace dave_deleted_apps_l357_357714

-- Conditions
variables 
  (initial_apps : ℕ)
  (new_apps : ℕ)
  (apps_left : ℕ)

-- The statement to prove
theorem dave_deleted_apps : 
  initial_apps = 10 → 
  new_apps = 11 →
  apps_left = 4 →
  initial_apps + new_apps - apps_left = 17 :=
by
  intros h_initial h_new h_left
  rw [h_initial, h_new, h_left]
  norm_num
  sorry

end dave_deleted_apps_l357_357714


namespace triangle_angle_sine_ratio_l357_357059

variable {A B C D : Point}
variable (angleBAC : Real)

def angleB := 45 -- 45 degrees in radians
def angleC := 60 -- 60 degrees in radians

def divides := (BCDiv: Real) → BCDiv = 1 / 3 -- D divides BC in ratio 1:2

theorem triangle_angle_sine_ratio
  (h_triangle : Triangle A B C)
  (h_angleB : ∠ B = angleB)
  (h_angleC : ∠ C = angleC)
  (h_divides : divides)
  : sin (∠ BAD) / sin (∠ CAD) = Real.sqrt 6 / 4 := by
  sorry

end triangle_angle_sine_ratio_l357_357059


namespace solve_for_x_l357_357962

theorem solve_for_x : 
  ∀ x : ℝ, 
    (x ≠ 2) ∧ (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 → 
    x = -11 / 6 :=
by
  intro x
  intro h 
  sorry

end solve_for_x_l357_357962


namespace angle_intersecting_lines_l357_357888

/-- 
Given three lines intersecting at a point forming six equal angles 
around the point, each angle equals 60 degrees.
-/
theorem angle_intersecting_lines (x : ℝ) (h : 6 * x = 360) : x = 60 := by
  sorry

end angle_intersecting_lines_l357_357888


namespace balls_to_ensure_color_l357_357869

theorem balls_to_ensure_color (total_balls : ℕ)
    (red_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (blue_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
    (H : total_balls = 100) (Hr : red_balls = 28) (Hg : green_balls = 20) (Hy : yellow_balls = 12) 
    (Hb : blue_balls = 20) (Hw : white_balls = 10) (Hb2 : black_balls = 10) :
    ∃ n : ℕ, n = 75 ∧ (∀ draws, draws > 74 → (∃ color : string, draws_by_color draws color ≥ 15)) :=
sorry

end balls_to_ensure_color_l357_357869


namespace product_of_numbers_l357_357217

theorem product_of_numbers (x y : ℝ) 
  (h₁ : x + y = 8 * (x - y)) 
  (h₂ : x * y = 40 * (x - y)) : x * y = 4032 := 
by
  sorry

end product_of_numbers_l357_357217


namespace john_learns_vowels_in_fifteen_days_l357_357752

def days_to_learn_vowels (days_per_vowel : ℕ) (num_vowels : ℕ) : ℕ :=
  days_per_vowel * num_vowels

theorem john_learns_vowels_in_fifteen_days :
  days_to_learn_vowels 3 5 = 15 :=
by
  -- Proof goes here
  sorry

end john_learns_vowels_in_fifteen_days_l357_357752


namespace trapezoid_shorter_base_l357_357679

theorem trapezoid_shorter_base (a b : ℕ) (mid_segment : ℕ) (longer_base : ℕ) 
    (h1 : mid_segment = 5) (h2 : longer_base = 105) 
    (h3 : mid_segment = (longer_base - a) / 2) : 
  a = 95 := 
by
  sorry

end trapezoid_shorter_base_l357_357679


namespace max_min_y_l357_357362

noncomputable def y (x : ℝ) : ℝ :=
  7 - 4 * (Real.sin x) * (Real.cos x) + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

theorem max_min_y :
  (∃ x : ℝ, y x = 10) ∧ (∃ x : ℝ, y x = 6) := by
  sorry

end max_min_y_l357_357362


namespace is_isosceles_right_triangle_l357_357862

theorem is_isosceles_right_triangle (a b c : ℝ) (h₁ : a = 2 * Real.sqrt 3) (h₂ : b = 2 * Real.sqrt 3) (h₃ : c = 2 * Real.sqrt 6)
  (h₄ : a^2 + b^2 = c^2) : (a = b ∧ ∀ a b c, a^2 + b^2 = c^2) := sorry

end is_isosceles_right_triangle_l357_357862


namespace sum_of_two_cubes_lt_1000_l357_357004

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357004


namespace dice_real_root_probability_l357_357587

theorem dice_real_root_probability :
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
                   (2,1), (2,2), (2,3), (2,4), (2,5), (2,6),
                   (3,1), (3,2), (3,3), (3,4), (3,5), (3,6),
                   (4,1), (4,2), (4,3), (4,4), (4,5), (4,6),
                   (5,1), (5,2), (5,3), (5,4), (5,5), (5,6),
                   (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)] in
  let favorable_outcomes := [(2,1), (3,1), (4,1), (5,1), (6,1),
                             (3,2), (4,2), (5,2), (6,2),
                             (4,3), (5,3), (6,3),
                             (4,4), (5,4), (6,4),
                             (5,5), (6,5),
                             (5,6), (6,6)] in
  (favorable_outcomes.length : ℚ) / (outcomes.length : ℚ) = 19 / 36 := 
by sorry

end dice_real_root_probability_l357_357587


namespace rectangle_angle_AMD_l357_357168

variables {A B C D M : Type} [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited M]
variables {AB AD AM MD : ℝ}

theorem rectangle_angle_AMD (h1 : AB = 8) (h2 : AD = 4) (h3 : AM = 2) 
  (h4 : ∠AMD = ∠CMD) : ∠AMD = 63 :=
begin
  sorry
end

end rectangle_angle_AMD_l357_357168


namespace not_a_polyhedron_l357_357248

def is_polyhedron (faces : ℕ) (has_curved_surface : Bool) : Bool :=
  faces ≥ 4 ∧ ¬has_curved_surface

def oblique_triangular_prism := (4, false)
def cube := (6, false)
def cylinder := (3, true)
def tetrahedron := (4, false)

theorem not_a_polyhedron :
  ¬is_polyhedron (prod.fst cylinder) (prod.snd cylinder) :=
by
  sorry

end not_a_polyhedron_l357_357248


namespace cube_faces_opposite_10_is_8_l357_357183

theorem cube_faces_opposite_10_is_8 (nums : Finset ℕ) (h_nums : nums = {6, 7, 8, 9, 10, 11})
  (sum_lateral_first : ℕ) (h_sum_lateral_first : sum_lateral_first = 36)
  (sum_lateral_second : ℕ) (h_sum_lateral_second : sum_lateral_second = 33)
  (faces_opposite_10 : ℕ) (h_faces_opposite_10 : faces_opposite_10 ∈ nums) :
  faces_opposite_10 = 8 :=
by
  sorry

end cube_faces_opposite_10_is_8_l357_357183


namespace chess_tournament_proof_l357_357464

-- Define the conditions
variables (i g n I G : ℕ)
variables (VI VG VD : ℕ)

-- Condition 1: The number of GMs is ten times the number of IMs
def condition1 : Prop := g = 10 * i
  
-- Condition 2: The sum of the points of all GMs is 4.5 times the sum of the points of all IMs
def condition2 : Prop := G = 5 * I + I / 2

-- Condition 3: The total number of players is the sum of IMs and GMs
def condition3 : Prop := n = i + g

-- Condition 4: Each player played only once against all other opponents
def condition4 : Prop := n * (n - 1) = 2 * (VI + VG + VD)

-- Condition 5: The sum of the points of all games is 5.5 times the sum of the points of all IMs
def condition5 : Prop := I + G = 11 * I / 2

-- Condition 6: Total games played
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The questions to be proven given the conditions
theorem chess_tournament_proof:
  condition1 i g →
  condition2 I G →
  condition3 i g n →
  condition4 n VI VG VD →
  condition5 I G →
  i = 1 ∧ g = 10 ∧ total_games n = 55 :=
by
  -- The proof is left as an exercise
  sorry

end chess_tournament_proof_l357_357464


namespace expected_total_rain_correct_l357_357288

-- Define the probabilities and rain amounts for one day.
def prob_sun : ℝ := 0.30
def prob_rain3 : ℝ := 0.40
def prob_rain8 : ℝ := 0.30
def rain_sun : ℝ := 0
def rain_three : ℝ := 3
def rain_eight : ℝ := 8
def days : ℕ := 7

-- Define the expected value of daily rain.
def E_daily_rain : ℝ :=
  prob_sun * rain_sun + prob_rain3 * rain_three + prob_rain8 * rain_eight

-- Define the expected total rain over seven days.
def E_total_rain : ℝ :=
  days * E_daily_rain

-- Statement of the proof problem.
theorem expected_total_rain_correct : E_total_rain = 25.2 := by
  -- Proof goes here
  sorry

end expected_total_rain_correct_l357_357288


namespace sequence_sum_abs_value_12_l357_357057

noncomputable def a_n (n a b : ℕ) : ℤ :=
2^n * a + b * n - 80

noncomputable def S_n (n a b : ℕ) : ℤ :=
∑ i in finset.range n, a_n (i + 1) a b

theorem sequence_sum_abs_value_12 
  (a b : ℕ) 
  (h1: S_n 6 a b = min (λ n, S_n n a b))
  (h2: a_n 36 a b % 7 = 0)
  (ha : a = 1)
  (hb : b = 2):
  ∑ i in finset.range 12, abs (a_n (i + 1) a b) = 8010 :=
by
  -- Proof omitted, hence sorry is used.
  sorry

end sequence_sum_abs_value_12_l357_357057


namespace train_crossing_time_l357_357678

theorem train_crossing_time
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ)
  (train_length_eq : length_train = 720)
  (bridge_length_eq : length_bridge = 320)
  (speed_eq : speed_kmh = 90) :
  (length_train + length_bridge) / (speed_kmh * (1000 / 3600)) = 41.6 := by
  sorry

end train_crossing_time_l357_357678


namespace sum_cosines_dihedral_tetrahedron_l357_357161

variables {V : Type*} [inner_product_space ℝ V]

/-- The main theorem stating the sum of the cosines of the dihedral angles of a tetrahedron is positive and does not exceed 2, 
    with equality only if the tetrahedron is equifacial -/
theorem sum_cosines_dihedral_tetrahedron (e_a e_b e_c e_d : V) 
  (ha : ∥e_a∥ = 1) (hb : ∥e_b∥ = 1) (hc : ∥e_c∥ = 1) (hd : ∥e_d∥ = 1) 
  (h_orth_1 : inner e_a e_b = cos θ) (h_orth_2 : inner e_a e_c = cos φ) (h_orth_3 : inner e_a e_d = cos ψ)
  (h_orth_4 : inner e_b e_c = cos χ) (h_orth_5 : inner e_b e_d = cos ξ) (h_orth_6 : inner e_c e_d = cos η)
  (sum_condition : ∀ a b c d ∈ {e_a, e_b, e_c, e_d}, ∥a + b + c + d∥ ≤ 2) :
  0 < (cos θ + cos φ + cos ψ + cos χ + cos ξ + cos η) ∧ 
  (cos θ + cos φ + cos ψ + cos χ + cos ξ + cos η) ≤ 2 :=
sorry

end sum_cosines_dihedral_tetrahedron_l357_357161


namespace probability_all_real_roots_l357_357291

open Real

-- Define the polynomial
def polynomial (b : ℝ) : ℝ → ℝ :=
λ x, x^4 + 2 * b * x^3 + (2 * b - 3) * x^2 + (-4 * b + 6) * x - 3

-- Probability calculation for the real roots condition
theorem probability_all_real_roots :
  let interval := Icc (-15 : ℝ) (20 : ℝ),
      prob := 1 - (sqrt 6) / 35 in
  ∀ b ∈ interval, 
  (∀ x, polynomial b x = 0 → (x ∈ (λ x, let (q_b := polynomial b) in
  (∃ y, q_b y = 0) → ((2 * b - 2)^2 - 6 ≥ 0))) ↔ Pr[b ∈ interval](you.real ℕ (prob)) :=
sorry

end probability_all_real_roots_l357_357291


namespace solution_l357_357515

noncomputable def problem : ℕ :=
  let α := real.pi / 80
  let β := real.pi / 60
  let theta := real.atan (17 / 88)
  let R := λ θ, (2 * α - θ + 2 * β - (2 * α - θ)) % (2 * real.pi)
  let is_invariant n := (n * (β - α)) % (2 * real.pi) = 0
  let m := nat.find (set_of is_invariant)
  m

theorem solution : problem = 240 := by
  sorry

end solution_l357_357515


namespace min_number_of_chips_l357_357469

theorem min_number_of_chips (strip_length : ℕ) (n : ℕ) : strip_length = 2100 → 
  (∀ i, (recorded number in cell i = |(number of chips left) - (number of chips right)|) ∧ recorded numbers are different and non-zero) → 
  n ≥ 1400 :=
by 
  assume strip_length_eq existence_of_chips
  -- Apply rest of the reasoning steps
  sorry

end min_number_of_chips_l357_357469


namespace equilateral_triangle_perimeter_l357_357562

theorem equilateral_triangle_perimeter (x : ℕ) (h : 2 * x = x + 15) : 
  3 * (2 * x) = 90 :=
by
  -- Definitions & hypothesis
  sorry

end equilateral_triangle_perimeter_l357_357562


namespace solution_l357_357534

noncomputable def problem (p q : ℕ) (x : ℝ) : Prop :=
  (sec x + tan x = 15 / 4) ∧ (csc x + cot x = p / q) ∧ gcd q p = 1

theorem solution (p q : ℕ) (x : ℝ) (h : problem p q x) : p + q = 28 :=
sorry

end solution_l357_357534


namespace triangle_PQR_is_equilateral_l357_357465

variables {A B C D O P Q R : Type}
variables [IsoscelesTrapezoid A B C D] (AB_parallel_DC : A ∥ D) (angleAOB_eq_60 : ∠ A O B = 60°)
variables (midpoint_P : P = midpoint O A) (midpoint_Q : Q = midpoint B C) (midpoint_R : R = midpoint O D)

theorem triangle_PQR_is_equilateral : EquilateralTriangle P Q R :=
sorry

end triangle_PQR_is_equilateral_l357_357465


namespace solution_to_system_of_eqns_l357_357357

theorem solution_to_system_of_eqns (x y z : ℝ) :
  (x = (2 * z ^ 2) / (1 + z ^ 2) ∧ y = (2 * x ^ 2) / (1 + x ^ 2) ∧ z = (2 * y ^ 2) / (1 + y ^ 2)) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end solution_to_system_of_eqns_l357_357357


namespace f_x_minus_1_pass_through_l357_357798

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + x

theorem f_x_minus_1_pass_through (a : ℝ) : f a (1 - 1) = 0 :=
by
  -- Proof is omitted here
  sorry

end f_x_minus_1_pass_through_l357_357798


namespace tara_dad_second_year_attendance_l357_357727

theorem tara_dad_second_year_attendance :
  let games_played_per_year := 20
  let attendance_rate := 0.90
  let first_year_games_attended := attendance_rate * games_played_per_year
  let second_year_games_difference := 4
  first_year_games_attended - second_year_games_difference = 14 :=
by
  -- We skip the proof here
  sorry

end tara_dad_second_year_attendance_l357_357727


namespace max_value_f_diff_l357_357415

open Real

noncomputable def f (A ω : ℝ) (x : ℝ) := A * sin (ω * x + π / 6) - 1

theorem max_value_f_diff {A ω : ℝ} (hA : A > 0) (hω : ω > 0)
  (h_sym : (π / 2) = π / (2 * ω))
  (h_initial : f A ω (π / 6) = 1) :
  ∀ (x1 x2 : ℝ), (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ (0 ≤ x2 ∧ x2 ≤ π / 2) →
  (f A ω x1 - f A ω x2 ≤ 3) :=
sorry

end max_value_f_diff_l357_357415


namespace positive_integer_solution_l357_357741

theorem positive_integer_solution (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (1 / (x * x : ℝ) + 1 / (y * y : ℝ) + 1 / (z * z : ℝ) + 1 / (t * t : ℝ) = 1) ↔ (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by
  sorry

end positive_integer_solution_l357_357741


namespace sum_of_arithmetic_sequence_l357_357050

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ} -- S_n is a function returning the sum of the first n terms
variable {a_4 a_8 a_6 S_11 : ℝ}

-- Condition: the sequence is arithmetic
variable (h1 : ∀ n m k : ℕ, a_n = a_m + (a_k - a_m) * (n - m)/(k - m))
-- Condition a4 + a8 = 4 
variable (h2 : a_4 + a_8 = 4)
-- Definition Sn of sum of terms in arithmetic sequence up to Sn
variable (h3 : S_11 = ∑ i in Finset.range(11), a_n i)

theorem sum_of_arithmetic_sequence : S_11 + a_n 6 = 24 :=
by
  sorry

end sum_of_arithmetic_sequence_l357_357050


namespace machine_A_time_l357_357516

def machine_A_hours (A : ℝ) (T : ℝ) : Prop :=
  ∃ (B C : ℝ),
  B = 3 ∧
  C = 6 ∧
  T = 1.3333333333333335 ∧
  (1 / A + 1 / B + 1 / C = 1 / T)

theorem machine_A_time :
  machine_A_hours 4 1.3333333333333335 :=
by
  use 3, 6
  split
  sorry

end machine_A_time_l357_357516


namespace hopps_ticket_average_l357_357520

theorem hopps_ticket_average
  (total_tickets : ℕ)
  (days_in_may : ℕ)
  (initial_days : ℕ)
  (initial_avg_tickets : ℕ)
  (required_avg_tickets : ℕ) :
  (initial_days * initial_avg_tickets) + ((days_in_may - initial_days) * required_avg_tickets) = total_tickets :=
by {
  let remaining_days := days_in_may - initial_days,
  let tickets_given := initial_days * initial_avg_tickets,
  let remaining_tickets := total_tickets - tickets_given,
  let average_needed := remaining_tickets / remaining_days,
  have : required_avg_tickets = average_needed, sorry
}

end hopps_ticket_average_l357_357520


namespace sum_of_two_cubes_lt_1000_l357_357010

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l357_357010


namespace circle_line_intersection_probability_l357_357379

def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

def line (x y b : ℝ) : Prop := y = x + b

def intersects (b : ℝ) : Prop := ∃ x y, circle x y ∧ line x y b

def probability_of_intersection : Real := (2 * Real.sqrt 2) / 5

theorem circle_line_intersection_probability : 
  (b : ℝ), b ∈ Set.Icc (-5 : ℝ) 5 → 
  (if intersects b then 1 else 0) = 
  probability_of_intersection := 
by sorry

end circle_line_intersection_probability_l357_357379


namespace probability_at_least_partly_green_no_red_l357_357953
  
theorem probability_at_least_partly_green_no_red :
  let colors := {red, green, blue, yellow, purple},
      excludeRed := colors.erase red,
      numSingleColor := excludeRed.card,
      numDoubleColors := excludeRed.subsets.card - numSingleColor + 1, -- numDoubleColors = (card choose 2) + (card choose 1)
      totalNoRed := numSingleColor + numDoubleColors,
      greenOnly := excludeRed.erase green,
      partlyGreen := greenOnly.card in
  totalNoRed = 10 → partlyGreen = 4 → (partlyGreen / totalNoRed : Real) = 2 / 5 :=
by
  intros colors excludeRed numSingleColor numDoubleColors totalNoRed greenOnly partlyGreen
  intros h_totalNoRed h_partlyGreen
  sorry

end probability_at_least_partly_green_no_red_l357_357953


namespace number_of_ways_to_assign_volunteers_l357_357728

/-- Theorem: The number of ways to assign 5 volunteers to 3 venues such that each venue has at least one volunteer is 150. -/
theorem number_of_ways_to_assign_volunteers :
  let total_ways := 3^5
  let subtract_one_empty := 3 * 2^5
  let add_back_two_empty := 3 * 1^5
  (total_ways - subtract_one_empty + add_back_two_empty) = 150 :=
by
  sorry

end number_of_ways_to_assign_volunteers_l357_357728


namespace triangle_inequality_squared_l357_357251

theorem triangle_inequality_squared {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := sorry

end triangle_inequality_squared_l357_357251


namespace not_solvable_equations_l357_357340

theorem not_solvable_equations :
  ¬(∃ x : ℝ, (x - 5) ^ 2 = -1) ∧ ¬(∃ x : ℝ, |2 * x| + 3 = 0) :=
by
  sorry

end not_solvable_equations_l357_357340


namespace find_ratio_l357_357062

section geometry_problem

variables {A B C P X Y Z : Type} 
variable [MetricSpace P]
variables [MetricSpace X] [MetricSpace Y]
variables {AB AC BC : ℝ}
variables (perpendicular_bisector_bc : P → Prop)
variables (angle_bisector_A : P → Prop)
variables (extension_AB : P → Prop)
variables (extension_AC : P → Prop)
variables xy_bc_intersection : P → P → P

-- Conditions
def condition1 : Prop := AC > AB
def condition2 (p : P) : Prop := perpendicular_bisector_bc p ∧ angle_bisector_A p
def condition3 (p : P) : Prop := (extension_AB p ∧ ⟂ p AB)
def condition4 (p : P) : Prop := (extension_AC p ∧ ⟂ p AC)
def condition5 (xy_intersect_bc : P) : Prop := xy_bc_intersection X Y = xy_intersect_bc

-- Prove that BZ / ZC = 1
theorem find_ratio : ∀ B Z C (hBZC : Z ∈ xy_bc_intersection X Y), BZ / ZC = 1 
  := sorry

end geometry_problem

end find_ratio_l357_357062


namespace parametric_line_segment_squared_sum_l357_357189

theorem parametric_line_segment_squared_sum :
  ∃ (p q r s : ℝ), 
    (q = 1) ∧ (s = -3) ∧ (p + q = -4) ∧ (r + s = 5) ∧ 
    (p^2 + q^2 + r^2 + s^2 = 99) :=
by {
  use [-5, 1, 8, -3],
  simp,
  split; ring,
}

end parametric_line_segment_squared_sum_l357_357189


namespace circumcenter_is_rational_l357_357556

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l357_357556


namespace headphones_to_case_ratio_l357_357951

theorem headphones_to_case_ratio 
    (phone_cost : ℝ) 
    (contract_cost : ℝ) 
    (case_percentage : ℝ) 
    (total_spent_first_year : ℝ) 
    (case_cost : ℝ := case_percentage * phone_cost) 
    (contract_total_cost : ℝ := contract_cost * 12) 
    (spent_without_headphones : ℝ := phone_cost + case_cost + contract_total_cost) 
    (headphones_cost : ℝ := total_spent_first_year - spent_without_headphones) 
    (ratio : ℝ := headphones_cost / case_cost) 
    (phone_cost_eq : phone_cost = 1000) 
    (contract_cost_eq : contract_cost = 200) 
    (case_percentage_eq : case_percentage = 0.20) 
    (total_spent_first_year_eq : total_spent_first_year = 3700) 
    (case_cost_eq : case_cost = 0.20 * 1000) 
    (contract_total_cost_eq : contract_total_cost = 200 * 12) 
    (spent_without_headphones_eq : spent_without_headphones = 1000 + 0.20 * 1000 + 200 * 12) 
    (headphones_cost_eq : headphones_cost = 3700 - (1000 + 0.20 * 1000 + 200 * 12))
    : ratio = 1 / 2 := 
by
  rw [phone_cost_eq, contract_cost_eq, case_percentage_eq, total_spent_first_year_eq, 
      case_cost_eq, contract_total_cost_eq, spent_without_headphones_eq, headphones_cost_eq]
  exact rfl


end headphones_to_case_ratio_l357_357951


namespace calculate_series_sum_l357_357700

-- Definition of the triangular number
def triangular_number (n : ℕ) : ℚ := n * (n + 1) / 2

-- Definition of the sum of the series involving triangular numbers
def sum_series (n : ℕ) : ℚ := 4 * (∑ i in Finset.range n, 1 / triangular_number (i + 1))

theorem calculate_series_sum : sum_series 2500 = 20000 / 2501 := 
sorry

end calculate_series_sum_l357_357700


namespace integral_equivalence_l357_357631

noncomputable def integral_function : ℝ → ℝ :=
λ x, (2 * x ^ 3 + 2 * x + 1) / ((x ^ 2 - x + 1) * (x ^ 2 + 1))

noncomputable def integral_result (C : ℝ) : ℝ → ℝ :=
λ x, (1 / 2) * Real.log (abs (x ^ 2 - x + 1)) 
        + Real.sqrt 3 * Real.arctan ((2 * x - 1) / Real.sqrt 3) 
        + (1 / 2) * Real.log (abs (x ^ 2 + 1)) 
        + C

theorem integral_equivalence (C : ℝ) :
  ∀ x, ∫ u in 0..x, integral_function u = integral_result C x :=
by
  sorry

end integral_equivalence_l357_357631


namespace polynomial_divisible_by_24_l357_357527

-- Defining the function
def f (n : ℕ) : ℕ :=
n^4 + 2*n^3 + 11*n^2 + 10*n

-- Statement of the theorem
theorem polynomial_divisible_by_24 (n : ℕ) (h : n > 0) : f n % 24 = 0 :=
sorry

end polynomial_divisible_by_24_l357_357527


namespace no_solution_system_of_inequalities_l357_357267

theorem no_solution_system_of_inequalities :
  ¬ ∃ (x y : ℝ),
    11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧
    5 * x + y ≤ -10 :=
by
  sorry

end no_solution_system_of_inequalities_l357_357267


namespace exists_k_l357_357388

-- Define the group of linear functions G and conditions
def linear_function (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define the set G and its properties
def G := {f : ℝ → ℝ // ∃ a b : ℝ, (f = linear_function a b) ∧ a ≠ 1}

axiom G_closed_composition :
  ∀ (f g : ℝ → ℝ) (hf : f ∈ G) (hg : g ∈ G), (g ∘ f) ∈ G

axiom G_closed_inverse :
  ∀ (f : ℝ → ℝ) (a b : ℝ) (hf : (f = linear_function a b) ∧ a ≠ 1),
    (λ x, (x - b) / a) ∈ G

axiom G_fixed_point :
  ∀ (f : ℝ → ℝ) (hf : f ∈ G), ∃ x_f : ℝ, f x_f = x_f

-- The statement for the proof problem
theorem exists_k :
  ∃ k : ℝ, ∀ (f : ℝ → ℝ) (hf : f ∈ G), f k = k :=
sorry

end exists_k_l357_357388


namespace incorrect_statements_identified_correctly_l357_357249

theorem incorrect_statements_identified_correctly:
  ∀ (A B C D : Point) (a b : Vector) (O : Point) 
    (non_collinear_points : ¬ collinear A B C)
    (x y z : ℝ),
  ( ( \overrightarrow{AB} + \overrightarrow{BC} + \overrightarrow{CD} + \overrightarrow{DA} = \overrightarrow{0} ∧
     ¬ (\|a\| - \|b\| = \|a + b\| ∧ collinear a b) ∧
     (collinear \overrightarrow{AB} \overrightarrow{CD} → (parallel AB CD ∨ collinear AB CD)) ∧
     ( ∃ (P : Point), \overrightarrow{OP} = x * \overrightarrow{OA} + y * \overrightarrow{OB} + z * \overrightarrow{OC} → coplanar P A B C)
   ) → 
   incorrect_statements = {B, C, D}
sorry

end incorrect_statements_identified_correctly_l357_357249


namespace part1_arithmetic_and_formulas_part2_sum_terms_part3_range_a_l357_357790

-- Part 1: Proving the formulas for aₙ and bₙ
theorem part1_arithmetic_and_formulas (n : ℕ) (n_pos : n > 0) :
  (∀ k : ℕ, k > 0 → k * b(k + 1) - (k + 1) * b(k) = k * (k + 1)) →
  b(1) = 1 →
  (∃ a_formula : ℕ → ℝ, ∃ b_formula : ℕ → ℝ, 
    (∀ n, a_formula n = 2^(n - 1)) ∧ (∀ n, b_formula n = n^2)) := 
sorry

-- Part 2: Sum of the first n terms of the sequence {cₙ}
theorem part2_sum_terms (n : ℕ) :
  (∀ k, a(k) = 2^(k - 1)) →
  (∀ k, c(k) = (-1)^(k - 1) * 4 * (k + 1) / ((3 + 2 * real.log2 (a k)) * (3 + 2 * real.log2 (a (k + 1))))) →
  ∃ T2n : ℝ, T2n = 4 * n / (12 * n + 9) :=
sorry

-- Part 3: Finding the range of real number a
theorem part3_range_a (n : ℕ) (a : ℝ) :
  (∀ k, a(k) = 2^(k - 1)) →
  (∀ k, b(k) = k^2) →
  (∀ k, S k = 2 * a(k) - 1) →
  (∀ k, d(k) = a(k) * real.sqrt (b(k))) →
  (∀ k, D k = ∑ i in finset.range k, d i) → 
  (∀ k, D k ≤ k * S k - a) →
  a ≤ 0 := 
sorry

end part1_arithmetic_and_formulas_part2_sum_terms_part3_range_a_l357_357790


namespace proof_problem_l357_357818

-- Assuming the given conditions
variables (x y : ℝ)
axiom h1 : 4^x = 6
axiom h2 : 9^y = 6

-- Statement of the proof problem
theorem proof_problem : (1/x) + (1/y) = 2 :=
by 
  sorry

end proof_problem_l357_357818


namespace Sn_3n_l357_357204

def Sn (n : ℕ) : ℕ := sorry -- Define the sum of the first n terms of an arithmetic sequence

axiom Sn_n : Sn n = 3
axiom Sn_2n : Sn (2 * n) = 10

theorem Sn_3n (n : ℕ) : Sn (3 * n) = 21 :=
by
  -- Use the properties of arithmetic sequences
  sorry

end Sn_3n_l357_357204


namespace other_number_eq_462_l357_357988

theorem other_number_eq_462 (a b : ℕ) 
  (lcm_ab : Nat.lcm a b = 4620) 
  (gcd_ab : Nat.gcd a b = 21) 
  (a_eq : a = 210) : b = 462 := 
by
  sorry

end other_number_eq_462_l357_357988


namespace prism_angle_l357_357872

/--
In a regular quadrilateral prism \( ABCDA_1B_1C_1D_1 \) with vertical edges \( AA_1 \),
\( BB_1 \), \( CC_1 \), and \( DD_1 \), consider a plane passing through the midpoints \(M\) and \(N\)
of the sides \( AD \) and \( DC \) respectively, and the vertex \( B_1 \). If the perimeter
of the cross-section is three times the diagonal of the base, then the angle between
this plane and the base plane is \( \arccos(\frac{3}{4}) \).
-/
theorem prism_angle (A B C D A1 B1 C1 D1 M N : Point) (h_prism : is_regular_quadrilateral_prism A B C D A1 B1 C1 D1)
  (h_midpoints : midpoint AD M ∧ midpoint DC N ∧ is_vertex_of_upper_base B1)
  (h_perimeter_condition : perimeter_cross_section_plane A B C D A1 B1 C1 D1 M N = 3 * diag_base_plane A B C D) :
  angle_between_planes M N B1 (base_plane A B C D) = arccos (3/4) :=
sorry

end prism_angle_l357_357872


namespace f_2005_equals_cos_l357_357925

noncomputable def f : ℕ → (Real → Real)
| 0       => (λ x => Real.sin x)
| (n + 1) => (λ x => (f n) x.derive)

theorem f_2005_equals_cos : ∀ x, f 2005 x = Real.cos x :=
by
  sorry

end f_2005_equals_cos_l357_357925


namespace zoo_lineup_l357_357518

theorem zoo_lineup (f1 f2 m1 m2 c1 c2 : Type) : 
  (∃ l : List Type, l = [f1, m1, c1, c2, m2, f2] ∨ l = [f2, m2, c1, c2, m1, f1]) ∧ 
  (∀ l : List Type, f1 = l.head ∧ f2 = l.last ∨ f2 = l.head ∧ f1 = l.last) ∧ 
  (∃ i : ℕ, c1 = l.nthLe i 0 ∧ c2 = l.nthLe (i+1) 0) → 
  24 :=
by {
  sorry 
}

end zoo_lineup_l357_357518


namespace circumcenter_is_rational_l357_357553

theorem circumcenter_is_rational (a1 a2 a3 b1 b2 b3 : ℚ)
  (h1 : (a2 - a1) ≠ 0 ∨ (b2 - b1) ≠ 0)
  (h2 : (a3 - a1) ≠ 0 ∨ (b3 - b1) ≠ 0) :
  ∃ x y : ℚ,
    ((a2 - a1) * x + (b2 - b1) * y = (a2^2 - a1^2 + b2^2 - b1^2) / 2) ∧
    ((a3 - a1) * x + (b3 - b1) * y = (a3^2 - a1^2 + b3^2 - b1^2) / 2) :=
begin
  -- proof goes here
  sorry,
end

end circumcenter_is_rational_l357_357553


namespace hyperbola_asymptotes_l357_357561

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (y^2 / 9 - x^2 / 4 = 1 →
  (y = (3 / 2) * x ∨ y = - (3 / 2) * x)) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l357_357561


namespace work_days_l357_357298

noncomputable def Wp : ℚ := 1/22
noncomputable def Wq : ℚ := Wp / 1.20

theorem work_days (p q : ℚ) 
  (h_p : p = Wp)
  (h_q : q = Wq) :
  let combined_work := p + q in
  combined_work = 48.4 / (22 * 26.4) →
  1 / combined_work = 12 :=
by
  intros combined_work hw_combined_work
  sorry

end work_days_l357_357298


namespace binom_expansion_coeff_l357_357401

theorem binom_expansion_coeff (n : ℕ) (h : (x^2 - 1/x)^n.has_term_count(6)) :
  let m := 5 in
  ∃ k : ℕ, k = binom 5 2 ∧ (x^2 - 1/x)^5.coeff_of_term_with_power(4) = 10 := by
  sorry

end binom_expansion_coeff_l357_357401


namespace min_Sn_reached_at_10_or_11_l357_357885

variable {a : ℕ → ℤ} -- Define a sequence of integers
variable {S : ℕ → ℤ} -- Define the sum sequence

theorem min_Sn_reached_at_10_or_11 (ha : ∀ n, a (n + 1) = a n + a 1)
  (h_neg : a 1 < 0) (h_sum_eq : S 9 = S 12) :
  (∀ n, S n = ∑ i in finset.range n, a (i + 1)) →
  S 10 = ∑ i in finset.range 10, a (i + 1) ∨ S 11 = ∑ i in finset.range 11, a (i + 1) :=
sorry

end min_Sn_reached_at_10_or_11_l357_357885


namespace length_greater_than_width_l357_357198

theorem length_greater_than_width
  (perimeter : ℕ)
  (P : perimeter = 150)
  (l w difference : ℕ)
  (L : l = 60)
  (W : w = 45)
  (D : difference = l - w) :
  difference = 15 :=
by
  sorry

end length_greater_than_width_l357_357198


namespace sampled_individual_l357_357596

theorem sampled_individual {population_size sample_size : ℕ} (population_size_cond : population_size = 1000)
  (sample_size_cond : sample_size = 20) (sampled_number : ℕ) (sampled_number_cond : sampled_number = 15) :
  (∃ n : ℕ, sampled_number + n * (population_size / sample_size) = 65) :=
by 
  sorry

end sampled_individual_l357_357596


namespace distance_is_sqrt_3_l357_357882

noncomputable def distance_from_vertex_P_to_base_ABC : ℝ :=
  let n : ℝ × ℝ × ℝ := (1, 1, 1)
  let CP : ℝ × ℝ × ℝ := (2, 2, -1)
  real.abs ((n.1 * CP.1 + n.2 * CP.2 + n.3 * CP.3) / real.sqrt (n.1^2 + n.2^2 + n.3^2))

theorem distance_is_sqrt_3 :
  distance_from_vertex_P_to_base_ABC = real.sqrt 3 :=
sorry

end distance_is_sqrt_3_l357_357882


namespace range_of_m_max_area_OAP_l357_357949

-- Part 1
theorem range_of_m (a m : ℝ) (h_a_pos : a > 0)
    (C₁ : ∀ x y : ℝ, y^2 = 1 - (x^2 / a^2))
    (C₂ : ∀ x y : ℝ, y^2 = 2 * (x + m))
    (common_point : ∃ x y : ℝ, C₁ x y ∧ C₂ x y ∧ y > 0) :
    -a < m ∧ m < a := 
sorry

-- Part 2
theorem max_area_OAP (a : ℝ) (h_a : 0 < a ∧ a < 1/2)
    (origin : (ℝ × ℝ := (0, 0)))
    (A : ℝ × ℝ := (-a, 0))
    (P_exists : ∃ x y : ℝ, (y > 0) ∧ (-a < x ∧ x < a) ∧ (1 - (x^2 / a^2) = (2 * (x + a))))
    (area : (∀ (P : ℝ × ℝ), 
        P_fst = x, P_snd = y, 
        S_ΔOAP = 1/2 * a * y)) :
    (∀ P : ℝ × ℝ), S_ΔOAP = a * sqrt(a - a^2) := 
sorry

end range_of_m_max_area_OAP_l357_357949


namespace real_part_zero_l357_357793

theorem real_part_zero {m : ℝ} (h₁ : m^2 - m - 2 = 0) (h₂ : m^2 - 2m - 3 ≠ 0) : m = 2 :=
sorry

end real_part_zero_l357_357793


namespace sum_of_first_n_natural_numbers_l357_357312

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end sum_of_first_n_natural_numbers_l357_357312


namespace initial_bacteria_count_l357_357972

theorem initial_bacteria_count (quad : ℕ) (num_intervals : ℕ) (final_count : ℕ) : 
  quad = 4 → 
  num_intervals = 120 / 15 → 
  final_count = (16 * quad^num_intervals) → 
  16 = (final_count / quad^num_intervals) :=
by
  intros hquad hintervals hfinal
  have quad_eq_four : ∀ (x : ℕ), x = quad → x = 4 := by 
    intro
    assumption
  have intervals_eq_eight : ∀ (x : ℕ), x = num_intervals → x = 120 / 15 := by
    intro
    assumption
  have final_count_eq : ∀ (x : ℕ), x = final_count → x = 16 * 4^8 := by
    intro
    assumption
  sorry

end initial_bacteria_count_l357_357972


namespace min_y_value_l357_357416

noncomputable def y (a x : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_y_value (a : ℝ) (h : a ≠ 0) : 
  (a ≥ 2 → ∃ x, y a x = a^2 - 2) ∧ (a < 2 → ∃ x, y a x = 2*(a-1)^2) :=
sorry

end min_y_value_l357_357416


namespace right_angle_triangle_with_two_beautiful_subtriangles_l357_357525

theorem right_angle_triangle_with_two_beautiful_subtriangles 
  (A B C M N K D E : Type) 
  [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]
  [linear_ordered_field M] [linear_ordered_field N] [linear_ordered_field K]
  [linear_ordered_field D] [linear_ordered_field E] 
  (on_side_BC : ∀ {P : Type}, P = M → P ≠ B ∧ P ≠ C)
  (on_side_CA : ∀ {P : Type}, P = N → P ≠ C ∧ P ≠ A)
  (on_side_AB : ∀ {P : Type}, P = K → P ≠ A ∧ P ≠ B)
  (beautiful_triangles : ∀ {T U: Type}, 
    (∀ {ℓ : Type}, ℓ = T → (∠BAC = ∠KMN) ∧ (∠ABC = ∠KNM)) 
    ∧ (∀ {ℓ : Type}, ℓ = U → (∠BAC = ∠KDE) ∧ (∠ABC = ∠KEN)) 
    ∧ (∃ {V : Type}, V = K)) :
  ∠ACB = 90 :=
sorry

end right_angle_triangle_with_two_beautiful_subtriangles_l357_357525


namespace trigonometric_identity_l357_357763

theorem trigonometric_identity (θ : ℝ) (h : Real.cot θ = 3) : 
    (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := 
by 
  sorry

end trigonometric_identity_l357_357763


namespace range_of_a_l357_357577

theorem range_of_a (a x : ℝ) (A : set ℝ) (h1 : 0 < a ∧ (∀ y ∈ A, log a (a - x^2 / 2) > log a (a - x))) (h2 : A ∩ set_of(λ (n : ℤ), n = 1)) : 1 < a ∧ a ∈ set.Ioi 1 :=
by
  sorry

end range_of_a_l357_357577


namespace box_length_is_24_l357_357277

theorem box_length_is_24 (L : ℕ) (h1 : ∀ s : ℕ, (L * 40 * 16 = 30 * s^3) → s ∣ 40 ∧ s ∣ 16) (h2 : ∃ s : ℕ, s ∣ 40 ∧ s ∣ 16) : L = 24 :=
by
  sorry

end box_length_is_24_l357_357277


namespace angle_AEF_l357_357484

-- Define the triangle ABC with given angle and points
variables (A B C D E F : Type) [add_group A]

-- Conditions given
axiom angle_ABC : ∠ B = 80
axiom equal_sides_1 : dist A B = dist A D
axiom equal_sides_2 : dist A D = dist D C
axiom equal_sides_3 : dist AF = dist BD
axiom equal_sides_4 : dist A B = dist A E

-- Goal to prove
theorem angle_AEF : ∠ A E F = 20 := 
sorry

end angle_AEF_l357_357484


namespace qualification_probabilities_expected_value_l357_357174

theorem qualification_probabilities :
  (∃ P₁ P₂ P₃ : ℝ,
    let P_A := 0.5 * 0.6,
    let P_B := 0.6 * 0.5,
    let P_C := 0.4 * 0.75 in
      P₁ = P_A ∧ P₂ = P_B ∧ P₃ = P_C ∧ P₁ = 0.3 ∧ P₂ = 0.3 ∧ P₃ = 0.3) :=
begin
  use [0.3, 0.3, 0.3],
  show (0.5 * 0.6 = 0.3) ∧ 
       (0.6 * 0.5 = 0.3) ∧ 
       (0.4 * 0.75 = 0.3),
  from ⟨rfl, rfl, rfl⟩,
end

theorem expected_value :
  let ξ0 := (1 - 0.3) ^ 3,
      ξ1 := 3 * 0.3 * (1 - 0.3) ^ 2,
      ξ2 := 3 * (0.3 ^ 2) * (1 - 0.3),
      ξ3 := 0.3 ^ 3 in
  (∀ ξ : ℝ, ξ = 0.9)
:=
begin
  show (1 * 0.441 + 2 * 0.189 + 3 * 0.027 = 0.9),
  from rfl,
end

end qualification_probabilities_expected_value_l357_357174


namespace sum_valid_two_digit_integers_l357_357241

theorem sum_valid_two_digit_integers :
  ∃ S : ℕ, S = 36 ∧ (∀ n, 10 ≤ n ∧ n < 100 →
    (∃ a b, n = 10 * a + b ∧ a + b ∣ n ∧ 2 * a * b ∣ n → n = 36)) :=
by
  sorry

end sum_valid_two_digit_integers_l357_357241


namespace subsets_with_six_l357_357848

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l357_357848


namespace ken_situps_l357_357900

variable (K : ℕ)

theorem ken_situps (h1 : Nathan = 2 * K)
                   (h2 : Bob = 3 * K / 2)
                   (h3 : Bob = K + 10) : 
                   K = 20 := 
by
  sorry

end ken_situps_l357_357900


namespace odd_function_proof_l357_357404

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := if x ≥ 0 then 2^x + 2 * x + b else -(2^(-x) + 2 * (-x) + b)

theorem odd_function_proof (b : ℝ) (h : b = -1) : f(-1, b) = -3 :=
by
  have h_odd : ∀ x, f(-x, b) = -f(x, b),
  from sorry,
  have h_f0 : f(0, b) = 0,
  from by
    calc
      f(0, b) = 2^0 + 2 * 0 + b := sorry
      ... = 1 + b := by norm_num
      ... = 0 := by rw [h],
  show f(-1, b) = -3,
  from sorry

end odd_function_proof_l357_357404


namespace volume_of_solid_bounded_by_surfaces_l357_357365

theorem volume_of_solid_bounded_by_surfaces :
  let f2 := λ y : ℝ, (1 / 2 - y)
  let f1 := (0 : ℝ)
  let integrand := λ (x y : ℝ), (f2 y - f1)
  let lower_x := λ y : ℝ, (2 * sqrt (2 * y))
  let upper_x := λ y : ℝ, (17 * sqrt (2 * y))
  let lower_y := (0 : ℝ)
  let upper_y := (1 / 2 : ℝ)
  let integral := ∫ y in lower_y..upper_y, ∫ x in lower_x y..upper_x y, integrand x y
  integral = 1 :=
by 
  -- We'll provide a detailed proof here.
  sorry

end volume_of_solid_bounded_by_surfaces_l357_357365


namespace subsets_with_six_l357_357845

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l357_357845


namespace intersection_product_eq_one_l357_357478

-- Define the parametric equations for Curve C1
def curveC1_param (t : ℝ) : ℝ × ℝ :=
  (1 + (sqrt 3) * t / 2, t / 2)

-- Define the polar equation for Curve C2
def curveC2_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * sin θ = 2

-- Define the Cartesian equation for Curve C1
def curveC1_cartesian (x y : ℝ) : Prop :=
  x = 1 + sqrt 3 * y

-- Define the rectangular coordinate equation for Curve C2
def curveC2_rectangular (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 6

-- Point P and intersection problem
def pointP : ℝ × ℝ := (1, 0)

-- Prove that |PA| * |PB| = 1 where A and B are intersection points
theorem intersection_product_eq_one (A B : ℝ × ℝ)
    (PA PB : ℝ) (hA : curveC1_cartesian A.1 A.2 ∧ curveC2_rectangular A.1 A.2)
    (hB : curveC1_cartesian B.1 B.2 ∧ curveC2_rectangular B.1 B.2)
    (hPA : PA = dist pointP A) (hPB : PB = dist pointP B) :
    PA * PB = 1 :=
  sorry

end intersection_product_eq_one_l357_357478


namespace subsets_with_six_l357_357844

open Finset

theorem subsets_with_six (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6}) :
  (S.filter (λ x, x = 6)).powerset.card = 32 :=
by
  rw [hS]
  have T : {1, 2, 3, 4, 5}.powerset = T.powerset := rfl
  sorry

end subsets_with_six_l357_357844


namespace digit_count_of_product_l357_357823

theorem digit_count_of_product :
  ∀ (a b c d : ℕ), a = 8 → b = 10 → c = 10 → d = 5 → 
  let product := (a * b ^ 10) * (b * b ^ d) in
  by let digits := Nat.log10 (8 * 10 ^ 16) + 1;
  exact digits = 17 :=
sorry

end digit_count_of_product_l357_357823


namespace percent_greater_than_fraction_l357_357260

theorem percent_greater_than_fraction : 
  (0.80 * 40) - (4/5) * 20 = 16 :=
by
  sorry

end percent_greater_than_fraction_l357_357260


namespace Ceva_concurrent_l357_357045

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def incircle (A B C : Triangle) : Circle := sorry
def line_through (P Q : Point) : Line := sorry
def parallel (L M : Line) : Prop := sorry
def intersection (L M : Line) : Point := sorry
def cevian (A B C : Point) (D : Point) : Line := sorry
def concurrent (L M N : Line) : Prop := sorry

theorem Ceva_concurrent (A B C D E F : Point) : Prop :=
  let I := incenter A B C
  let incircle_center := I
  let D := intersection (line_through I (parallel (line_through A B)) (incircle A B C))
  let E := intersection (line_through I (parallel (line_through B C)) (incircle A B C))
  let F := intersection (line_through I (parallel (line_through C A)) (incircle A B C))
  concurrent (cevian A B C D) (cevian B A C E) (cevian C A B F)

example : ∀ (A B C D E F : Point),
  (incenter A B C = I ∧
   incircle_center = I ∧
   D = intersection (line_through I (parallel (line_through A B)) (incircle A B C)) ∧
   E = intersection (line_through I (parallel (line_through B C)) (incircle A B C)) ∧
   F = intersection (line_through I (parallel (line_through C A)) (incircle A B C)))
  → concurrent (cevian A B C D) (cevian B A C E) (cevian C A B F) := 
by sorry

end Ceva_concurrent_l357_357045


namespace no_eventual_periodicity_mod_l357_357269

def sum_of_digits (n : ℕ) : ℕ := (n.digits 10).sum

def recursive_sequence (x : ℕ) (s : ℕ → ℕ) : ℕ → ℕ
| 0 => x
| (n + 1) => (recursive_sequence n) ^ (s n) + 1

theorem no_eventual_periodicity_mod 
  (x : ℕ)
  (x1_nat_const : x = x)
  (s : ℕ → ℕ := sum_of_digits) :
  ¬ ∃ (m : ℕ), m > 2500 ∧ ∃ (N T : ℕ), ∀ (n : ℕ), n ≥ N → m ∣ ((recursive_sequence x s n) - (recursive_sequence x s (n + T))) :=
by
  sorry

end no_eventual_periodicity_mod_l357_357269


namespace problem1_problem2_l357_357916

-- Step 1
theorem problem1 (a b c A B C : ℝ) (h1 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a^2 = b^2 + c^2 := sorry

-- Step 2
theorem problem2 (a b c : ℝ) (h_a : a = 5) (h_cosA : Real.cos A = 25 / 31) 
  (h_conditions : 2 * a^2 = b^2 + c^2 ∧ 2 * b * c = a^2 / Real.cos A) :
  a + b + c = 14 := sorry

end problem1_problem2_l357_357916


namespace angle_ADF_45_l357_357889

-- Define all conditions
variables {O A B C D E F : Type} {Circle : Type} [IsCircle Circle O]
variables (diag : Line B E) (ext_diag : Extends C diag)
variables (tangent : TangentAt CA O A) (bisector : Bisector DC ∠ACB)
variables (intersect1 : Intersects DC AE F) (intersect2 : Intersects DC AB D)

-- Define the problem statement
theorem angle_ADF_45 (h1 : IsOnLine C diag)
  (h2 : CA.TangentAt O A)
  (h3 : DC.BisectsAngle ACB)
  (h4 : DC.Intersects AE F)
  (h5 : DC.Intersects AB D) :
  ∠ADF = 45° :=
by
  sorry

end angle_ADF_45_l357_357889


namespace sin_2theta_eq_neg_12_over_13_l357_357817

variables (θ : ℝ)
def a := (Real.sin θ, 1)
def b := (-Real.sin θ, 0)
def c := (Real.cos θ, -1)

theorem sin_2theta_eq_neg_12_over_13
  (h : 2 • a θ - b θ = (3 * Real.sin θ, 2)) : Real.sin (2 * θ) = -12 / 13 := by
  sorry

end sin_2theta_eq_neg_12_over_13_l357_357817


namespace axis_of_symmetry_range_of_x_for_f_geq_one_l357_357802

open Real

noncomputable def f (x : ℝ) := cos x - sqrt 3 * sin x

theorem axis_of_symmetry : ∃ k : ℤ, ∀ x, x = k * π - π / 3 → f x = 2 * cos (x + π / 3) := 
by
  sorry

theorem range_of_x_for_f_geq_one : 
  ∃ k : ℤ, ∀ x, (x ∈ Icc (-2 * π / 3 + 2 * k * π) (2 * k * π)) ↔ (f x ≥ 1) :=
by
  sorry

end axis_of_symmetry_range_of_x_for_f_geq_one_l357_357802


namespace find_k_of_inverse_proportion_l357_357419

theorem find_k_of_inverse_proportion (k x y : ℝ) (h : y = k / x) (hx : x = 2) (hy : y = 6) : k = 12 :=
by
  sorry

end find_k_of_inverse_proportion_l357_357419


namespace total_trees_planted_l357_357459

theorem total_trees_planted :
  let trees_A := 151 / 14 + 1,
      trees_B := 210 / 18 + 1,
      trees_C := 275 / 12 + 1,
      trees_D := 345 / 20 + 1,
      trees_E := 475 / 22 + 1 in
  trees_A.floor + trees_B.floor + trees_C.floor + trees_D.floor + trees_E.floor = 86 :=
by
  let trees_A := 151 / 14 + 1,
      trees_B := 210 / 18 + 1,
      trees_C := 275 / 12 + 1,
      trees_D := 345 / 20 + 1,
      trees_E := 475 / 22 + 1
  have h1 : trees_A.floor = 11 := by sorry
  have h2 : trees_B.floor = 12 := by sorry
  have h3 : trees_C.floor = 23 := by sorry
  have h4 : trees_D.floor = 18 := by sorry
  have h5 : trees_E.floor = 22 := by sorry
  have h_total : 11 + 12 + 23 + 18 + 22 = 86 := by sorry
  exact h_total

end total_trees_planted_l357_357459


namespace cos_Y_eq_4_div_5_l357_357048

theorem cos_Y_eq_4_div_5 
  (XYZ : Type) [right_triangle XYZ]
  (X Y Z : XYZ) 
  (hX : ∠X = 90º) 
  (XY XZ : ℝ) 
  (hXY : XY = 9)
  (hXZ : XZ = 12) : 
  cos Y = 4/5 :=
sorry

end cos_Y_eq_4_div_5_l357_357048


namespace geometric_sequence_and_sum_exists_constant_l357_357080

variable {α : Type*} [LinearOrderedField α]

-- Define the geometric sequence and the sum Sn
noncomputable def geom_sequence (a q : α) (n : ℕ) : α := a * q ^ n
noncomputable def sum_geom_sequence (a q : α) (n : ℕ) : α :=
  if q = 1 then a * n else a * (q^n - 1) / (q - 1)

-- Given Conditions
variables (a q : α)
variable (h1 : geom_sequence a q 1 + geom_sequence a q 2 + geom_sequence a q 3 = 39)
variable (h2 : 2 * (geom_sequence a q 2 + 6) = geom_sequence a q 1 + geom_sequence a q 3)
variable (h3 : 1 < q)

-- Definitions consistent with conditions
def a2 : α := geom_sequence a q 1
def a3 : α := geom_sequence a q 2
def a4 : α := geom_sequence a q 3

-- Statement of the problem
theorem geometric_sequence_and_sum_exists_constant :
  sum_geom_sequence a q n = (3 ^ n - 1) / 2 ∧
  ∃ λ : α, (∀ n : ℕ, sum_geom_sequence a q n + λ = (3 ^ n) / 2) ∧
             λ = 1 / 2 :=
by
  sorry

end geometric_sequence_and_sum_exists_constant_l357_357080


namespace no_solutions_system_of_inequalities_l357_357265

open Set

theorem no_solutions_system_of_inequalities :
  ∀ (x y : ℝ),
    ¬(11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3 ∧ 5 * x + y ≤ -10) :=
by
  intro x y
  rw not_and
  intro h1 h2
  let y' := -10 - 5 * x
  have h3 : y = y' := eq_of_le_of_le h2 (le_of_eq rfl)
  sorry

end no_solutions_system_of_inequalities_l357_357265


namespace relationship_between_a_b_c_l357_357766

noncomputable def a : ℝ := 2 ^ 0.2
noncomputable def b : ℝ := 0.4 ^ 0.2
noncomputable def c : ℝ := 0.4 ^ 0.6

theorem relationship_between_a_b_c : a > b ∧ b > c :=
by
  have ha : a = 2 ^ 0.2 := rfl
  have hb : b = 0.4 ^ 0.2 := rfl
  have hc : c = 0.4 ^ 0.6 := rfl
  sorry

end relationship_between_a_b_c_l357_357766
