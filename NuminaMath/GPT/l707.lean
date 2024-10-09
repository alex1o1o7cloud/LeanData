import Mathlib

namespace irreducible_polynomial_l707_70705

open Polynomial

theorem irreducible_polynomial (n : ℕ) : Irreducible ((X^2 + X)^(2^n) + 1 : ℤ[X]) := sorry

end irreducible_polynomial_l707_70705


namespace matrix_solution_l707_70794

-- Define the 2x2 matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := 
  ![ ![30 / 7, -13 / 7], 
     ![-6 / 7, -10 / 7] ]

-- Define the vectors
def vec1 : Fin 2 → ℚ := ![2, 3]
def vec2 : Fin 2 → ℚ := ![4, -1]

-- Expected results
def result1 : Fin 2 → ℚ := ![3, -6]
def result2 : Fin 2 → ℚ := ![19, -2]

-- The proof statement
theorem matrix_solution : (N.mulVec vec1 = result1) ∧ (N.mulVec vec2 = result2) :=
  by sorry

end matrix_solution_l707_70794


namespace find_seventh_number_l707_70775

-- Let's denote the 10 numbers as A1, A2, A3, A4, A5, A6, A7, A8, A9, A10.
variables {A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ}

-- The average of all 10 numbers is 60.
def avg_10 (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10) / 10 = 60

-- The average of the first 6 numbers is 68.
def avg_first_6 (A1 A2 A3 A4 A5 A6 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6) / 6 = 68

-- The average of the last 6 numbers is 75.
def avg_last_6 (A5 A6 A7 A8 A9 A10 : ℝ) := (A5 + A6 + A7 + A8 + A9 + A10) / 6 = 75

-- Proving that the 7th number (A7) is 192.
theorem find_seventh_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) 
  (h1 : avg_10 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10) 
  (h2 : avg_first_6 A1 A2 A3 A4 A5 A6) 
  (h3 : avg_last_6 A5 A6 A7 A8 A9 A10) :
  A7 = 192 :=
by
  sorry

end find_seventh_number_l707_70775


namespace xiao_ming_math_score_l707_70741

noncomputable def math_score (C M E : ℕ) : ℕ :=
  let A := 94
  let N := 3
  let total_score := A * N
  let T_CE := (A - 1) * 2
  total_score - T_CE

theorem xiao_ming_math_score (C M E : ℕ)
    (h1 : (C + M + E) / 3 = 94)
    (h2 : (C + E) / 2 = 93) :
  math_score C M E = 96 := by
  sorry

end xiao_ming_math_score_l707_70741


namespace symmetric_graph_increasing_interval_l707_70781

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_graph_increasing_interval :
  (∀ x : ℝ, f (-x) = -f x) → -- f is odd
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 7 → f x < f y) → -- f is increasing in [3,7]
  (∀ x : ℝ, 3 ≤ x → x ≤ 7 → f x ≤ 5) → -- f has a maximum value of 5 in [3,7]
  (∀ x y : ℝ, -7 ≤ x → x < y → y ≤ -3 → f x < f y) ∧ -- f is increasing in [-7,-3]
  (∀ x : ℝ, -7 ≤ x → x ≤ -3 → f x ≥ -5) -- f has a minimum value of -5 in [-7,-3]
:= sorry

end symmetric_graph_increasing_interval_l707_70781


namespace fair_attendance_l707_70707

theorem fair_attendance :
  let this_year := 600
  let next_year := 2 * this_year
  let total_people := 2800
  let last_year := total_people - this_year - next_year
  (1200 - last_year = 200) ∧ (last_year = 1000) := by
  sorry

end fair_attendance_l707_70707


namespace sin_beta_l707_70728

variable (α β : ℝ)
variable (hα1 : 0 < α) (hα2 : α < Real.pi / 2)
variable (hβ1 : 0 < β) (hβ2: β < Real.pi / 2)
variable (h1 : Real.cos α = 5 / 13)
variable (h2 : Real.sin (α - β) = 4 / 5)

theorem sin_beta (α β : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
  (hβ1 : 0 < β) (hβ2 : β < Real.pi / 2) 
  (h1 : Real.cos α = 5 / 13) 
  (h2 : Real.sin (α - β) = 4 / 5) : 
  Real.sin β = 16 / 65 := 
by 
  sorry

end sin_beta_l707_70728


namespace solve_system_l707_70737

theorem solve_system (x y z : ℝ) :
  x^2 = y^2 + z^2 ∧
  x^2024 = y^2024 + z^2024 ∧
  x^2025 = y^2025 + z^2025 ↔
  (y = x ∧ z = 0) ∨
  (y = -x ∧ z = 0) ∨
  (y = 0 ∧ z = x) ∨
  (y = 0 ∧ z = -x) :=
by {
  sorry -- The detailed proof will be filled here.
}

end solve_system_l707_70737


namespace fraction_subtraction_l707_70724

theorem fraction_subtraction (a b : ℕ) (h₁ : a = 18) (h₂ : b = 14) :
  (↑a / ↑b - ↑b / ↑a) = (32 / 63) := by
  sorry

end fraction_subtraction_l707_70724


namespace find_a_given_even_l707_70766

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem find_a_given_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 :=
by
  unfold f
  sorry

end find_a_given_even_l707_70766


namespace one_over_x_plus_one_over_y_eq_fifteen_l707_70711

theorem one_over_x_plus_one_over_y_eq_fifteen
  (x y : ℝ)
  (h1 : xy > 0)
  (h2 : 1 / xy = 5)
  (h3 : (x + y) / 5 = 0.6) : 
  (1 / x) + (1 / y) = 15 := 
by
  sorry

end one_over_x_plus_one_over_y_eq_fifteen_l707_70711


namespace sequence_general_term_l707_70767

theorem sequence_general_term (a : ℕ → ℤ) (h₀ : a 0 = 1) (hstep : ∀ n, a (n + 1) = if a n = 1 then 0 else 1) :
  ∀ n, a n = (1 + (-1)^(n + 1)) / 2 :=
sorry

end sequence_general_term_l707_70767


namespace greatest_nat_not_sum_of_two_composites_l707_70760

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem greatest_nat_not_sum_of_two_composites :
  ¬ ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ 11 = a + b ∧
  (∀ n : ℕ, n > 11 → ¬ ∃ x y : ℕ, is_composite x ∧ is_composite y ∧ n = x + y) :=
sorry

end greatest_nat_not_sum_of_two_composites_l707_70760


namespace fraction_result_l707_70756

theorem fraction_result (x : ℚ) (h₁ : x * (3/4) = (1/6)) : (x - (1/12)) = (5/36) := 
sorry

end fraction_result_l707_70756


namespace initial_number_of_men_l707_70742

variable (M : ℕ) (A : ℕ)
variable (change_in_age: ℕ := 16)
variable (age_increment: ℕ := 2)

theorem initial_number_of_men :
  ((A + age_increment) * M = A * M + change_in_age) → M = 8 :=
by
  intros h_1
  sorry

end initial_number_of_men_l707_70742


namespace calculate_purple_pants_l707_70721

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def plaid_shirts : ℕ := 3
def non_plaid_non_purple_items : ℕ := 21

theorem calculate_purple_pants : total_pants - (non_plaid_non_purple_items - (total_shirts - plaid_shirts)) = 5 :=
by 
  sorry

end calculate_purple_pants_l707_70721


namespace vector_difference_perpendicular_l707_70773

/-- Proof that the vector difference a - b is perpendicular to b given specific vectors a and b -/
theorem vector_difference_perpendicular {a b : ℝ × ℝ} (h_a : a = (2, 0)) (h_b : b = (1, 1)) :
  (a - b) • b = 0 :=
by
  sorry

end vector_difference_perpendicular_l707_70773


namespace temperature_on_tuesday_l707_70769

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th) / 3 = 45 →
  (W + Th + F) / 3 = 50 →
  F = 53 →
  T = 38 :=
by 
  intros h1 h2 h3
  sorry

end temperature_on_tuesday_l707_70769


namespace appointment_on_tuesday_duration_l707_70730

theorem appointment_on_tuesday_duration :
  let rate := 20
  let monday_appointments := 5
  let monday_each_duration := 1.5
  let thursday_appointments := 2
  let thursday_each_duration := 2
  let saturday_duration := 6
  let weekly_earnings := 410
  let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  let tuesday_earnings := weekly_earnings - known_earnings
  (tuesday_earnings / rate = 3) :=
by
  -- let rate := 20
  -- let monday_appointments := 5
  -- let monday_each_duration := 1.5
  -- let thursday_appointments := 2
  -- let thursday_each_duration := 2
  -- let saturday_duration := 6
  -- let weekly_earnings := 410
  -- let known_earnings := (monday_appointments * monday_each_duration * rate) + (thursday_appointments * thursday_each_duration * rate) + (saturday_duration * rate)
  -- let tuesday_earnings := weekly_earnings - known_earnings
  -- exact tuesday_earnings / rate = 3
  sorry

end appointment_on_tuesday_duration_l707_70730


namespace parallel_or_identical_lines_l707_70768

theorem parallel_or_identical_lines (a b c d e f : ℝ) :
  2 * b - 3 * a = 15 → 4 * d - 6 * c = 18 → (b ≠ d → a = c) :=
by
  intros h1 h2 hneq
  sorry

end parallel_or_identical_lines_l707_70768


namespace geom_seq_val_l707_70793

noncomputable def is_geom_seq (a : ℕ → ℝ) : Prop :=
∃ q b, ∀ n, a n = b * q^n

variables (a : ℕ → ℝ)

axiom a_5_a_7 : a 5 * a 7 = 2
axiom a_2_plus_a_10 : a 2 + a 10 = 3

theorem geom_seq_val (a_geom : is_geom_seq a) :
  (a 12) / (a 4) = 2 ∨ (a 12) / (a 4) = 1 / 2 :=
sorry

end geom_seq_val_l707_70793


namespace solve_for_x_l707_70751

theorem solve_for_x : ∃ x : ℝ, 4 * x + 6 * x = 360 - 9 * (x - 4) ∧ x = 396 / 19 :=
by
  sorry

end solve_for_x_l707_70751


namespace kai_marbles_over_200_l707_70772

theorem kai_marbles_over_200 (marbles_on_day : Nat → Nat)
  (h_initial : marbles_on_day 0 = 4)
  (h_growth : ∀ n, marbles_on_day (n + 1) = 3 * marbles_on_day n) :
  ∃ k, marbles_on_day k > 200 ∧ k = 4 := by
  sorry

end kai_marbles_over_200_l707_70772


namespace necessary_but_not_sufficient_l707_70757

theorem necessary_but_not_sufficient (x : ℝ) : (x > -1) ↔ (∀ y : ℝ, (2 * y > 2) → (-1 < y)) :=
sorry

end necessary_but_not_sufficient_l707_70757


namespace running_time_square_field_l707_70712

theorem running_time_square_field
  (side : ℕ)
  (running_speed_kmh : ℕ)
  (perimeter : ℕ := 4 * side)
  (running_speed_ms : ℕ := (running_speed_kmh * 1000) / 3600)
  (time : ℕ := perimeter / running_speed_ms) 
  (h_side : side = 35)
  (h_speed : running_speed_kmh = 9) :
  time = 56 := 
by
  sorry

end running_time_square_field_l707_70712


namespace solve_for_x_l707_70752

theorem solve_for_x (x : ℕ) (h : 5 * (2 ^ x) = 320) : x = 6 :=
by
  sorry

end solve_for_x_l707_70752


namespace round_robin_tournament_l707_70782

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end round_robin_tournament_l707_70782


namespace mn_sum_l707_70723

theorem mn_sum {m n : ℤ} (h : ∀ x : ℤ, (x + 8) * (x - 1) = x^2 + m * x + n) : m + n = -1 :=
by
  sorry

end mn_sum_l707_70723


namespace coefficient_x3_y7_expansion_l707_70792

theorem coefficient_x3_y7_expansion : 
  let n := 10
  let a := (2 : ℚ) / 3
  let b := -(3 : ℚ) / 5
  let k := 3
  let binom := Nat.choose n k
  let term := binom * (a ^ k) * (b ^ (n - k))
  term = -(256 : ℚ) / 257 := 
by
  -- Proof omitted
  sorry

end coefficient_x3_y7_expansion_l707_70792


namespace trigonometric_identity_l707_70771

variable (α : Real)

theorem trigonometric_identity :
  (Real.tan (α - Real.pi / 4) = 1 / 2) →
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) :=
by
  intro h
  sorry

end trigonometric_identity_l707_70771


namespace spherical_to_rectangular_correct_l707_70729

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  spherical_to_rectangular 3 (Real.pi / 2) (Real.pi / 3) = (0, (3 * Real.sqrt 3) / 2, 3 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l707_70729


namespace tips_fraction_l707_70748

-- Define the conditions
variables (S T : ℝ) (h : T = (2 / 4) * S)

-- The statement to be proved
theorem tips_fraction : (T / (S + T)) = 1 / 3 :=
by
  sorry

end tips_fraction_l707_70748


namespace smallest_diameter_of_tablecloth_l707_70790

theorem smallest_diameter_of_tablecloth (a : ℝ) (h : a = 1) : ∃ d : ℝ, d = Real.sqrt 2 ∧ (∀ (x : ℝ), x < d → ¬(∀ (y : ℝ), (y^2 + y^2 = x^2) → y ≤ a)) :=
by 
  sorry

end smallest_diameter_of_tablecloth_l707_70790


namespace part1_part2_part3_l707_70720

-- Part 1: Simplifying the Expression
theorem part1 (a b : ℝ) : 
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 :=
by sorry

-- Part 2: Finding the Value of an Expression
theorem part2 (x y : ℝ) (h : x^2 - 2 * y = 4) : 
  3 * x^2 - 6 * y - 21 = -9 :=
by sorry

-- Part 3: Evaluating a Compound Expression
theorem part3 (a b c d : ℝ) (h1 : a - 2 * b = 6) (h2 : 2 * b - c = -8) (h3 : c - d = 9) : 
  (a - c) + (2 * b - d) - (2 * b - c) = 7 :=
by sorry

end part1_part2_part3_l707_70720


namespace needle_intersection_probability_l707_70763

noncomputable def needle_probability (a l : ℝ) (h : l < a) : ℝ :=
  (2 * l) / (a * Real.pi)

theorem needle_intersection_probability (a l : ℝ) (h : l < a) :
  needle_probability a l h = 2 * l / (a * Real.pi) :=
by
  -- This is the statement to be proved
  sorry

end needle_intersection_probability_l707_70763


namespace triangle_equilateral_l707_70764

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
a = b ∧ b = c

theorem triangle_equilateral (A B C a b c : ℝ) (hB : B = 60) (hb : b^2 = a * c) (hcos : b^2 = a^2 + c^2 - a * c):
  is_equilateral a b c :=
by
  sorry

end triangle_equilateral_l707_70764


namespace find_expression_l707_70755

theorem find_expression 
  (E a : ℤ) 
  (h1 : (E + (3 * a - 8)) / 2 = 74) 
  (h2 : a = 28) : 
  E = 72 := 
by
  sorry

end find_expression_l707_70755


namespace complex_modulus_l707_70709

open Complex

noncomputable def modulus_of_complex : ℂ :=
  (1 - 2 * Complex.I) * (1 - 2 * Complex.I) / Complex.I

theorem complex_modulus : Complex.abs modulus_of_complex = 5 :=
  sorry

end complex_modulus_l707_70709


namespace solve_equation_l707_70702

theorem solve_equation (x : ℚ) :
  (x^2 + 3 * x + 4) / (x + 5) = x + 6 ↔ x = -13 / 4 :=
by
  sorry

end solve_equation_l707_70702


namespace amount_C_l707_70798

-- Define the variables and conditions.
variables (A B C : ℝ)
axiom h1 : A = (2 / 3) * B
axiom h2 : B = (1 / 4) * C
axiom h3 : A + B + C = 544

-- State the theorem.
theorem amount_C (A B C : ℝ) (h1 : A = (2 / 3) * B) (h2 : B = (1 / 4) * C) (h3 : A + B + C = 544) : C = 384 := 
sorry

end amount_C_l707_70798


namespace find_f_log_l707_70744

def even_function (f : ℝ → ℝ) :=
  ∀ (x : ℝ), f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) :=
  ∀ (x : ℝ), f (x + p) = f x

theorem find_f_log (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 2)
  (h_condition : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f x = 3 * x + 4 / 9) :
  f (Real.log 5 / Real.log (1 / 3)) = -5 / 9 :=
by
  sorry

end find_f_log_l707_70744


namespace find_y_l707_70799

theorem find_y : 
  (6 + 10 + 14 + 22) / 4 = (15 + y) / 2 → y = 11 :=
by
  intros h
  sorry

end find_y_l707_70799


namespace helpers_cakes_l707_70765

theorem helpers_cakes (S : ℕ) (helpers large_cakes small_cakes : ℕ)
  (h1 : helpers = 10)
  (h2 : large_cakes = 2)
  (h3 : small_cakes = 700)
  (h4 : 1 * helpers * large_cakes = 20)
  (h5 : 2 * helpers * S = small_cakes) :
  S = 35 :=
by
  sorry

end helpers_cakes_l707_70765


namespace base_conversion_b_eq_3_l707_70780

theorem base_conversion_b_eq_3 (b : ℕ) (hb : b > 0) :
  (3 * 6^1 + 5 * 6^0 = 23) →
  (1 * b^2 + 3 * b + 2 = 23) →
  b = 3 :=
by {
  sorry
}

end base_conversion_b_eq_3_l707_70780


namespace ratio_of_hardback_books_is_two_to_one_l707_70787

noncomputable def ratio_of_hardback_books : ℕ :=
  let sarah_paperbacks := 6
  let sarah_hardbacks := 4
  let brother_paperbacks := sarah_paperbacks / 3
  let total_books_brother := 10
  let brother_hardbacks := total_books_brother - brother_paperbacks
  brother_hardbacks / sarah_hardbacks

theorem ratio_of_hardback_books_is_two_to_one : 
  ratio_of_hardback_books = 2 :=
by
  sorry

end ratio_of_hardback_books_is_two_to_one_l707_70787


namespace product_of_terms_in_geometric_sequence_l707_70715

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

end product_of_terms_in_geometric_sequence_l707_70715


namespace hyperbola_symmetric_asymptotes_l707_70795

noncomputable def M : ℝ := 225 / 16

theorem hyperbola_symmetric_asymptotes (M_val : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = x * (4 / 3) ∨ y = -x * (4 / 3))
  ∧ (y^2 / 25 - x^2 / M_val = 1 → y = x * (5 / Real.sqrt M_val) ∨ y = -x * (5 / Real.sqrt M_val)))
  → M_val = M := by
  sorry

end hyperbola_symmetric_asymptotes_l707_70795


namespace quadratic_root_condition_l707_70754

theorem quadratic_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 1 ∧ x2 < 1 ∧ x1^2 + 2*a*x1 + 1 = 0 ∧ x2^2 + 2*a*x2 + 1 = 0) →
  a < -1 :=
by
  sorry

end quadratic_root_condition_l707_70754


namespace theater_ticket_sales_l707_70706

theorem theater_ticket_sales (A K : ℕ) (h1 : A + K = 275) (h2 :  12 * A + 5 * K = 2150) : K = 164 := by
  sorry

end theater_ticket_sales_l707_70706


namespace number_of_sevens_l707_70779

theorem number_of_sevens (n : ℕ) : ∃ (k : ℕ), k < n ∧ ∃ (f : ℕ → ℕ), (∀ i, f i = 7) ∧ (7 * ((77 - 7) / 7) ^ 14 - 1) / (7 + (7 + 7)/7) = 7^(f k) :=
by sorry

end number_of_sevens_l707_70779


namespace car_city_mileage_l707_70735

theorem car_city_mileage (h c t : ℝ) 
  (h_eq : h * t = 462)
  (c_eq : (h - 15) * t = 336) 
  (c_def : c = h - 15) : 
  c = 40 := 
by 
  sorry

end car_city_mileage_l707_70735


namespace six_divisors_third_seven_times_second_fourth_ten_more_than_third_l707_70784

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end six_divisors_third_seven_times_second_fourth_ten_more_than_third_l707_70784


namespace candies_per_person_l707_70740

theorem candies_per_person (a b people total_candies candies_per_person : ℕ)
  (h1: a = 17)
  (h2: b = 19)
  (h3: people = 9)
  (h4: total_candies = a + b)
  (h5: candies_per_person = total_candies / people) :
  candies_per_person = 4 :=
by sorry

end candies_per_person_l707_70740


namespace last_digit_product_3_2001_7_2002_13_2003_l707_70776

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_product_3_2001_7_2002_13_2003 :
  last_digit (3^2001 * 7^2002 * 13^2003) = 9 :=
by
  sorry

end last_digit_product_3_2001_7_2002_13_2003_l707_70776


namespace constant_term_in_expansion_l707_70762

theorem constant_term_in_expansion (x : ℂ) : 
  (2 - (3 / x)) * (x ^ 2 + 2 / x) ^ 5 = 0 := 
sorry

end constant_term_in_expansion_l707_70762


namespace move_point_right_l707_70722

theorem move_point_right 
  (x y : ℤ)
  (h : (x, y) = (2, -1)) :
  (x + 3, y) = (5, -1) := 
by
  sorry

end move_point_right_l707_70722


namespace prob_contact_l707_70743

variables (p : ℝ)
def prob_no_contact : ℝ := (1 - p) ^ 40

theorem prob_contact : 1 - prob_no_contact p = 1 - (1 - p) ^ 40 := by
  sorry

end prob_contact_l707_70743


namespace pam_walked_1683_miles_l707_70725

noncomputable def pam_miles_walked 
    (pedometer_limit : ℕ)
    (initial_reading : ℕ)
    (flips : ℕ)
    (final_reading : ℕ)
    (steps_per_mile : ℕ)
    : ℕ :=
  (pedometer_limit + 1) * flips + final_reading / steps_per_mile

theorem pam_walked_1683_miles
    (pedometer_limit : ℕ := 49999)
    (initial_reading : ℕ := 0)
    (flips : ℕ := 50)
    (final_reading : ℕ := 25000)
    (steps_per_mile : ℕ := 1500) 
    : pam_miles_walked pedometer_limit initial_reading flips final_reading steps_per_mile = 1683 := 
  sorry

end pam_walked_1683_miles_l707_70725


namespace solve_eq1_solve_eq2_l707_70747

-- For Equation (1)
theorem solve_eq1 (x : ℝ) : x^2 - 4*x - 6 = 0 → x = 2 + Real.sqrt 10 ∨ x = 2 - Real.sqrt 10 :=
sorry

-- For Equation (2)
theorem solve_eq2 (x : ℝ) : (x / (x - 1) - 1 = 3 / (x^2 - 1)) → x ≠ 1 ∧ x ≠ -1 → x = 2 :=
sorry

end solve_eq1_solve_eq2_l707_70747


namespace find_values_of_pqr_l707_70774

def A (p : ℝ) := {x : ℝ | x^2 + p * x - 2 = 0}
def B (q r : ℝ) := {x : ℝ | x^2 + q * x + r = 0}
def A_union_B (p q r : ℝ) := A p ∪ B q r = {-2, 1, 5}
def A_intersect_B (p q r : ℝ) := A p ∩ B q r = {-2}

theorem find_values_of_pqr (p q r : ℝ) :
  A_union_B p q r → A_intersect_B p q r → p = -1 ∧ q = -3 ∧ r = -10 :=
by
  sorry

end find_values_of_pqr_l707_70774


namespace coordinates_of_A_equidistant_BC_l707_70708

theorem coordinates_of_A_equidistant_BC :
  ∃ z : ℚ, (∀ A B C : ℚ × ℚ × ℚ, A = (0, 0, z) ∧ B = (7, 0, -15) ∧ C = (2, 10, -12) →
  (dist A B = dist A C)) ↔ z = -(13/3) :=
by sorry

end coordinates_of_A_equidistant_BC_l707_70708


namespace smallest_odd_digit_n_l707_70783

theorem smallest_odd_digit_n {n : ℕ} (h : n > 1) : 
  (∀ d ∈ (Nat.digits 10 (9997 * n)), d % 2 = 1) → n = 3335 :=
sorry

end smallest_odd_digit_n_l707_70783


namespace percentage_supports_policy_l707_70739

theorem percentage_supports_policy (men women : ℕ) (men_favor women_favor : ℝ) (total_population : ℕ) (total_supporters : ℕ) (percentage_supporters : ℝ)
  (h1 : men = 200) 
  (h2 : women = 800)
  (h3 : men_favor = 0.70)
  (h4 : women_favor = 0.75)
  (h5 : total_population = men + women)
  (h6 : total_supporters = (men_favor * men) + (women_favor * women))
  (h7 : percentage_supporters = (total_supporters / total_population) * 100) :
  percentage_supporters = 74 := 
by
  sorry

end percentage_supports_policy_l707_70739


namespace diagonal_plane_angle_l707_70726

theorem diagonal_plane_angle
  (α : Real)
  (a : Real)
  (plane_square_angle_with_plane : Real)
  (diagonal_plane_angle : Real) 
  (h1 : plane_square_angle_with_plane = α) :
  diagonal_plane_angle = Real.arcsin (Real.sin α / Real.sqrt 2) :=
sorry

end diagonal_plane_angle_l707_70726


namespace train_length_eq_l707_70736

theorem train_length_eq 
  (speed_kmh : ℝ) (time_sec : ℝ) 
  (h_speed_kmh : speed_kmh = 126)
  (h_time_sec : time_sec = 6.856594329596489) : 
  ((speed_kmh * 1000 / 3600) * time_sec) = 239.9808045358781 :=
by
  -- We skip the proof with sorry, as per instructions
  sorry

end train_length_eq_l707_70736


namespace bob_corn_stalks_per_row_l707_70734

noncomputable def corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  ℕ :=
  (bushels * stalks_per_bushel) / rows

theorem bob_corn_stalks_per_row
  (rows : ℕ)
  (bushels : ℕ)
  (stalks_per_bushel : ℕ) :
  rows = 5 → bushels = 50 → stalks_per_bushel = 8 → corn_stalks_per_row rows bushels stalks_per_bushel = 80 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  unfold corn_stalks_per_row
  rfl

end bob_corn_stalks_per_row_l707_70734


namespace midpoint_trace_quarter_circle_l707_70717

theorem midpoint_trace_quarter_circle (L : ℝ) (hL : 0 < L):
  ∃ (C : ℝ) (M : ℝ × ℝ → ℝ), 
    (∀ (x y : ℝ), x^2 + y^2 = L^2 → M (x, y) = C) ∧ 
    (C = (1/2) * L) ∧ 
    (∀ (x y : ℝ), M (x, y) = (x/2)^2 + (y/2)^2) → 
    ∀ (x y : ℝ), x^2 + y^2 = L^2 → (x/2)^2 + (y/2)^2 = (1/2 * L)^2 := 
by
  sorry

end midpoint_trace_quarter_circle_l707_70717


namespace calculate_binary_expr_l707_70778

theorem calculate_binary_expr :
  let a := 0b11001010
  let b := 0b11010
  let c := 0b100
  (a * b) / c = 0b1001110100 := by
sorry

end calculate_binary_expr_l707_70778


namespace length_of_faster_train_l707_70731

theorem length_of_faster_train (speed_faster_train : ℝ) (speed_slower_train : ℝ) (elapsed_time : ℝ) (relative_speed : ℝ) (length_train : ℝ)
  (h1 : speed_faster_train = 50) 
  (h2 : speed_slower_train = 32) 
  (h3 : elapsed_time = 15) 
  (h4 : relative_speed = (speed_faster_train - speed_slower_train) * (1000 / 3600)) 
  (h5 : length_train = relative_speed * elapsed_time) :
  length_train = 75 :=
sorry

end length_of_faster_train_l707_70731


namespace original_group_men_l707_70700

-- Let's define the parameters of the problem
def original_days := 55
def absent_men := 15
def completed_days := 60

-- We need to show that the number of original men (x) is 180
theorem original_group_men (x : ℕ) (h : x * original_days = (x - absent_men) * completed_days) : x = 180 :=
by
  sorry

end original_group_men_l707_70700


namespace ratio_paislee_to_calvin_l707_70701

theorem ratio_paislee_to_calvin (calvin_points paislee_points : ℕ) (h1 : calvin_points = 500) (h2 : paislee_points = 125) : paislee_points / calvin_points = 1 / 4 := by
  sorry

end ratio_paislee_to_calvin_l707_70701


namespace total_cost_of_refueling_l707_70738

theorem total_cost_of_refueling 
  (smaller_tank_capacity : ℤ)
  (larger_tank_capacity : ℤ)
  (num_smaller_planes : ℤ)
  (num_larger_planes : ℤ)
  (fuel_cost_per_liter : ℤ)
  (service_charge_per_plane : ℤ)
  (total_cost : ℤ) :
  smaller_tank_capacity = 60 →
  larger_tank_capacity = 90 →
  num_smaller_planes = 2 →
  num_larger_planes = 2 →
  fuel_cost_per_liter = 50 →
  service_charge_per_plane = 100 →
  total_cost = (num_smaller_planes * smaller_tank_capacity + num_larger_planes * larger_tank_capacity) * (fuel_cost_per_liter / 100) + (num_smaller_planes + num_larger_planes) * service_charge_per_plane →
  total_cost = 550 :=
by
  intros
  sorry

end total_cost_of_refueling_l707_70738


namespace find_shortest_height_l707_70745

variable (T S P Q : ℝ)

theorem find_shortest_height (h1 : T = 77.75) (h2 : T = S + 9.5) (h3 : P = S + 5) (h4 : Q = P - 3) : S = 68.25 :=
  sorry

end find_shortest_height_l707_70745


namespace find_math_marks_l707_70753

theorem find_math_marks (subjects : ℕ)
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℝ)
  (math_marks : ℕ) :
  subjects = 5 →
  english_marks = 96 →
  physics_marks = 99 →
  chemistry_marks = 100 →
  biology_marks = 98 →
  average_marks = 98.2 →
  math_marks = 98 :=
by
  intros h_subjects h_english h_physics h_chemistry h_biology h_average
  sorry

end find_math_marks_l707_70753


namespace error_difference_l707_70749

noncomputable def total_income_without_error (T: ℝ) : ℝ :=
  T + 110000

noncomputable def total_income_with_error (T: ℝ) : ℝ :=
  T + 1100000

noncomputable def mean_without_error (T: ℝ) : ℝ :=
  (T + 110000) / 500

noncomputable def mean_with_error (T: ℝ) : ℝ :=
  (T + 1100000) / 500

theorem error_difference (T: ℝ) :
  mean_with_error T - mean_without_error T = 1980 :=
by
  sorry

end error_difference_l707_70749


namespace jury_concludes_you_are_not_guilty_l707_70791

def criminal_is_a_liar : Prop := sorry -- The criminal is a liar, known.
def you_are_a_liar : Prop := sorry -- You are a liar, unknown.
def you_are_not_guilty : Prop := sorry -- You are not guilty.

theorem jury_concludes_you_are_not_guilty :
  criminal_is_a_liar → you_are_a_liar → you_are_not_guilty → "I am guilty" = "You are not guilty" :=
by
  -- Proof construct omitted as per problem requirements
  sorry

end jury_concludes_you_are_not_guilty_l707_70791


namespace range_of_objective_function_l707_70786

def objective_function (x y : ℝ) : ℝ := 3 * x - y

theorem range_of_objective_function (x y : ℝ) 
  (h1 : x + 2 * y ≥ 2)
  (h2 : 2 * x + y ≤ 4)
  (h3 : 4 * x - y ≥ -1)
  : - 3 / 2 ≤ objective_function x y ∧ objective_function x y ≤ 6 := 
sorry

end range_of_objective_function_l707_70786


namespace cost_of_previous_hay_l707_70727

theorem cost_of_previous_hay
    (x : ℤ)
    (previous_hay_bales : ℤ)
    (better_quality_hay_cost : ℤ)
    (additional_amount_needed : ℤ)
    (better_quality_hay_bales : ℤ)
    (new_total_cost : ℤ) :
    previous_hay_bales = 10 ∧ 
    better_quality_hay_cost = 18 ∧ 
    additional_amount_needed = 210 ∧ 
    better_quality_hay_bales = 2 * previous_hay_bales ∧ 
    new_total_cost = better_quality_hay_bales * better_quality_hay_cost ∧ 
    new_total_cost - additional_amount_needed = 10 * x → 
    x = 15 := by
  sorry

end cost_of_previous_hay_l707_70727


namespace greatest_integer_b_l707_70719

theorem greatest_integer_b (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 7 ≠ 0) → b ≤ 5 :=
by sorry

end greatest_integer_b_l707_70719


namespace tomatoes_harvest_ratio_l707_70746

noncomputable def tomatoes_ratio (w t f : ℕ) (g r : ℕ) : ℕ × ℕ :=
  if (w = 400) ∧ ((w + t + f) = 2000) ∧ ((g = 700) ∧ (r = 700) ∧ ((g + r) = f)) ∧ (t = 200) then 
    (2, 1)
  else 
    sorry

theorem tomatoes_harvest_ratio : 
  ∀ (w t f : ℕ) (g r : ℕ), 
  (w = 400) → 
  (w + t + f = 2000) → 
  (g = 700) → 
  (r = 700) → 
  (g + r = f) → 
  (t = 200) →
  tomatoes_ratio w t f g r = (2, 1) :=
by {
  -- insert proof here
  sorry
}

end tomatoes_harvest_ratio_l707_70746


namespace selling_price_l707_70796

-- Definitions
def price_coffee_A : ℝ := 10
def price_coffee_B : ℝ := 12
def weight_coffee_A : ℝ := 240
def weight_coffee_B : ℝ := 240
def total_weight : ℝ := 480
def total_cost : ℝ := (weight_coffee_A * price_coffee_A) + (weight_coffee_B * price_coffee_B)

-- Theorem
theorem selling_price (h_total_weight : total_weight = weight_coffee_A + weight_coffee_B) :
  total_cost / total_weight = 11 :=
by
  sorry

end selling_price_l707_70796


namespace complement_of_angle_l707_70713

def complement_angle (deg : ℕ) (min : ℕ) : ℕ × ℕ :=
  if deg < 90 then 
    let total_min := (90 * 60)
    let angle_min := (deg * 60) + min
    let comp_min := total_min - angle_min
    (comp_min / 60, comp_min % 60) -- degrees and remaining minutes
  else 
    (0, 0) -- this case handles if the angle is not less than complement allowable range

-- Definitions based on the problem
def given_angle_deg : ℕ := 57
def given_angle_min : ℕ := 13

-- Complement calculation
def comp (deg : ℕ) (min : ℕ) : ℕ × ℕ := complement_angle deg min

-- Expected result of the complement
def expected_comp : ℕ × ℕ := (32, 47)

-- Theorem to prove the complement of 57°13' is 32°47'
theorem complement_of_angle : comp given_angle_deg given_angle_min = expected_comp := by
  sorry

end complement_of_angle_l707_70713


namespace geo_seq_a12_equal_96_l707_70718

def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_a12_equal_96
  (a : ℕ → ℝ) (q : ℝ)
  (h0 : 1 < q)
  (h1 : is_geometric a q)
  (h2 : a 3 * a 7 = 72)
  (h3 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end geo_seq_a12_equal_96_l707_70718


namespace correct_addition_result_l707_70785

theorem correct_addition_result (x : ℚ) (h : x - 13/5 = 9/7) : x + 13/5 = 227/35 := 
by sorry

end correct_addition_result_l707_70785


namespace quadratic_inequality_solution_l707_70750

theorem quadratic_inequality_solution (x : ℝ) : (x^2 - 4 * x - 21 < 0) ↔ (-3 < x ∧ x < 7) :=
sorry

end quadratic_inequality_solution_l707_70750


namespace integer_solutions_l707_70704

theorem integer_solutions :
  { (x, y) : ℤ × ℤ |
       y^2 + y = x^4 + x^3 + x^2 + x } =
  { (-1, -1), (-1, 0), (0, -1), (0, 0), (2, 5), (2, -6) } :=
by
  sorry

end integer_solutions_l707_70704


namespace new_cost_after_decrease_l707_70710

theorem new_cost_after_decrease (C new_C : ℝ) (hC : C = 1100) (h_decrease : new_C = 0.76 * C) : new_C = 836 :=
-- To be proved based on the given conditions
sorry

end new_cost_after_decrease_l707_70710


namespace system_solution_fraction_l707_70770

theorem system_solution_fraction (x y z : ℝ) (h1 : x + (-95/9) * y + 4 * z = 0)
  (h2 : 4 * x + (-95/9) * y - 3 * z = 0) (h3 : 3 * x + 5 * y - 4 * z = 0) (hx_ne_zero : x ≠ 0) 
  (hy_ne_zero : y ≠ 0) (hz_ne_zero : z ≠ 0) : 
  (x * z) / (y ^ 2) = 20 :=
sorry

end system_solution_fraction_l707_70770


namespace tiling_implies_divisibility_l707_70714

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end tiling_implies_divisibility_l707_70714


namespace ratio_out_of_state_to_in_state_l707_70759

/-
Given:
- total job applications Carly sent is 600
- job applications sent to companies in her state is 200

Prove:
- The ratio of job applications sent to companies in other states to the number sent to companies in her state is 2:1.
-/

def total_applications : ℕ := 600
def in_state_applications : ℕ := 200
def out_of_state_applications : ℕ := total_applications - in_state_applications

theorem ratio_out_of_state_to_in_state :
  (out_of_state_applications / in_state_applications) = 2 :=
by
  sorry

end ratio_out_of_state_to_in_state_l707_70759


namespace problem_1_problem_2_l707_70788

-- Problem 1
theorem problem_1 :
  -((1 / 2) / 3) * (3 - (-3)^2) = 1 :=
by
  sorry

-- Problem 2
theorem problem_2 {x : ℝ} (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 * x) / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) :=
by
  sorry

end problem_1_problem_2_l707_70788


namespace h_2023_eq_4052_l707_70733

theorem h_2023_eq_4052 (h : ℕ → ℕ) (h1 : h 1 = 2) (h2 : h 2 = 2) 
    (h3 : ∀ n ≥ 3, h n = h (n-1) - h (n-2) + 2 * n) : h 2023 = 4052 := 
by
  -- Use conditions as given
  sorry

end h_2023_eq_4052_l707_70733


namespace find_y_l707_70761

theorem find_y (y : ℝ) (h : (17.28 / 12) / (3.6 * y) = 2) : y = 0.2 :=
by {
  sorry
}

end find_y_l707_70761


namespace eval_expression_l707_70797

theorem eval_expression {p q r s : ℝ} 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := 
by 
  sorry

end eval_expression_l707_70797


namespace problem1_l707_70758

theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 4) (h2 : 3 ^ (m + 4 * n) = 324) : 2016 ^ n = 2016 := 
by 
  sorry

end problem1_l707_70758


namespace angle_C_is_120_l707_70777

theorem angle_C_is_120 (C L U A : ℝ)
  (H1 : C = L)
  (H2 : L = U)
  (H3 : A = L)
  (H4 : A + L = 180)
  (H5 : 6 * C = 720) : C = 120 :=
by
  sorry

end angle_C_is_120_l707_70777


namespace solve_for_x_l707_70789

theorem solve_for_x (x : ℚ) : (x = 70 / (8 - 3 / 4)) → (x = 280 / 29) :=
by
  intro h
  -- Proof to be provided here
  sorry

end solve_for_x_l707_70789


namespace min_value_expression_l707_70703

noncomputable section

variables {x y : ℝ}

theorem min_value_expression (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 1 ∧ 
    (∃ min_val : ℝ, min_val = (x^2 / (x + 2) + y^2 / (y + 1)) ∧ min_val = 1 / 4)) :=
  sorry

end min_value_expression_l707_70703


namespace area_half_l707_70716

theorem area_half (width height : ℝ) (h₁ : width = 25) (h₂ : height = 16) :
  (width * height) / 2 = 200 :=
by
  -- The formal proof is skipped here
  sorry

end area_half_l707_70716


namespace find_x_l707_70732

noncomputable def arithmetic_sequence (x : ℝ) : Prop := 
  (x + 1) - (1/3) = 4 * x - (x + 1)

theorem find_x :
  ∃ x : ℝ, arithmetic_sequence x ∧ x = 5 / 6 :=
by
  use 5 / 6
  unfold arithmetic_sequence
  sorry

end find_x_l707_70732
