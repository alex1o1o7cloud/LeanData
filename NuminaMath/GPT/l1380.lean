import Mathlib

namespace NUMINAMATH_GPT_sticks_form_equilateral_triangle_l1380_138055

theorem sticks_form_equilateral_triangle {n : ℕ} (h1 : n ≥ 5) (h2 : n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :
  (∃ a b c, a = b ∧ b = c ∧ a + b + c = (n * (n + 1)) / 2) :=
by
  sorry

end NUMINAMATH_GPT_sticks_form_equilateral_triangle_l1380_138055


namespace NUMINAMATH_GPT_cos_eq_neg_four_fifths_of_tan_l1380_138093

theorem cos_eq_neg_four_fifths_of_tan (α : ℝ) (h_tan : Real.tan α = 3 / 4) (h_interval : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.cos α = -4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_eq_neg_four_fifths_of_tan_l1380_138093


namespace NUMINAMATH_GPT_regression_equation_l1380_138081

-- Define the regression coefficient and correlation
def negatively_correlated (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100

-- The question is to prove that given x and y are negatively correlated,
-- the regression equation is \hat{y} = -2x + 100
theorem regression_equation (x y : ℝ) (h : negatively_correlated x y) :
  (∃ a, a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100) → ∃ (b : ℝ), b = -2 ∧ ∀ (x_val : ℝ), y = b * x_val + 100 :=
by
  sorry

end NUMINAMATH_GPT_regression_equation_l1380_138081


namespace NUMINAMATH_GPT_derivative_y_l1380_138001

noncomputable def u (x : ℝ) := 4 * x - 1 + Real.sqrt (16 * x ^ 2 - 8 * x + 2)
noncomputable def v (x : ℝ) := Real.sqrt (16 * x ^ 2 - 8 * x + 2) * Real.arctan (4 * x - 1)

noncomputable def y (x : ℝ) := Real.log (u x) - v x

theorem derivative_y (x : ℝ) :
  deriv y x = (4 * (1 - 4 * x)) / (Real.sqrt (16 * x ^ 2 - 8 * x + 2)) * Real.arctan (4 * x - 1) :=
by
  sorry

end NUMINAMATH_GPT_derivative_y_l1380_138001


namespace NUMINAMATH_GPT_juniors_in_club_l1380_138059

theorem juniors_in_club
  (j s x y : ℝ)
  (h1 : x = 0.4 * j)
  (h2 : y = 0.25 * s)
  (h3 : j + s = 36)
  (h4 : x = 2 * y) :
  j = 20 :=
by
  sorry

end NUMINAMATH_GPT_juniors_in_club_l1380_138059


namespace NUMINAMATH_GPT_fourth_vertex_of_square_l1380_138061

theorem fourth_vertex_of_square (A B C D : ℂ) : 
  A = (2 + 3 * I) ∧ B = (-3 + 2 * I) ∧ C = (-2 - 3 * I) →
  D = (0 - 0.5 * I) :=
sorry

end NUMINAMATH_GPT_fourth_vertex_of_square_l1380_138061


namespace NUMINAMATH_GPT_minimize_sum_of_f_seq_l1380_138021

def f (x : ℝ) : ℝ := x^2 - 8 * x + 10

def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem minimize_sum_of_f_seq
  (a : ℕ → ℝ)
  (h₀ : isArithmeticSequence a 1)
  (h₁ : a 1 = a₁)
  : f (a 1) + f (a 2) + f (a 3) = 3 * a₁^2 - 18 * a₁ + 30 →

  (∀ x, 3 * x^2 - 18 * x + 30 ≥ 3 * 3^2 - 18 * 3 + 30) →
  a₁ = 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_f_seq_l1380_138021


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l1380_138038

noncomputable def polynomial := λ x: ℝ => x^3 - 24 * x^2 + 88 * x - 75

theorem partial_fraction_decomposition
  (p q r A B C : ℝ)
  (hpq : p ≠ q)
  (hpr : p ≠ r)
  (hqr : q ≠ r)
  (hroots : polynomial p = 0 ∧ polynomial q = 0 ∧ polynomial r = 0)
  (hdecomposition: ∀ s: ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
                      1 / polynomial s = A / (s - p) + B / (s - q) + C / (s - r)) :
  (1 / A + 1 / B + 1 / C = 256) := sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l1380_138038


namespace NUMINAMATH_GPT_urn_probability_l1380_138072

theorem urn_probability :
  ∀ (urn: Finset (ℕ × ℕ)), 
    urn = {(2, 1)} →
    (∀ (n : ℕ) (urn' : Finset (ℕ × ℕ)), n ≤ 5 → urn = urn' → 
      (∃ (r b : ℕ), (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)} ∨ (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)}) → 
    ∃ (p : ℚ), p = 8 / 21)
  := by
    sorry

end NUMINAMATH_GPT_urn_probability_l1380_138072


namespace NUMINAMATH_GPT_three_y_squared_value_l1380_138009

theorem three_y_squared_value : ∃ x y : ℤ, 3 * x + y = 40 ∧ 2 * x - y = 20 ∧ 3 * y ^ 2 = 48 :=
by
  sorry

end NUMINAMATH_GPT_three_y_squared_value_l1380_138009


namespace NUMINAMATH_GPT_total_amount_of_currency_notes_l1380_138094

theorem total_amount_of_currency_notes (x y : ℕ) (h1 : x + y = 85) (h2 : 50 * y = 3500) : 100 * x + 50 * y = 5000 := by
  sorry

end NUMINAMATH_GPT_total_amount_of_currency_notes_l1380_138094


namespace NUMINAMATH_GPT_number_of_true_propositions_l1380_138054

-- Definitions based on the problem
def proposition1 (α β : ℝ) : Prop := (α + β = 180) → (α + β = 90)
def proposition2 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)
def proposition3 (α β γ δ : ℝ) : Prop := (α = β) → (γ = δ)

-- Proof problem statement
theorem number_of_true_propositions : ∃ n : ℕ, n = 2 :=
by
  let p1 := false
  let p2 := false
  let p3 := true
  existsi (if p3 then 1 else 0 + if p2 then 1 else 0 + if p1 then 1 else 0)
  simp
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l1380_138054


namespace NUMINAMATH_GPT_number_divisible_by_19_l1380_138086

theorem number_divisible_by_19 (n : ℕ) : (12000 + 3 * 10^n + 8) % 19 = 0 := 
by sorry

end NUMINAMATH_GPT_number_divisible_by_19_l1380_138086


namespace NUMINAMATH_GPT_fraction_passengers_from_asia_l1380_138078

theorem fraction_passengers_from_asia (P : ℕ)
  (hP : P = 108)
  (frac_NA : ℚ) (frac_EU : ℚ) (frac_AF : ℚ)
  (Other_continents : ℕ)
  (h_frac_NA : frac_NA = 1/12)
  (h_frac_EU : frac_EU = 1/4)
  (h_frac_AF : frac_AF = 1/9)
  (h_Other_continents : Other_continents = 42) :
  (P * (1 - (frac_NA + frac_EU + frac_AF)) - Other_continents) / P = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_passengers_from_asia_l1380_138078


namespace NUMINAMATH_GPT_number_of_bricks_in_wall_l1380_138042

noncomputable def rate_one_bricklayer (x : ℕ) : ℚ := x / 8
noncomputable def rate_other_bricklayer (x : ℕ) : ℚ := x / 12
noncomputable def combined_rate_with_efficiency (x : ℕ) : ℚ := (rate_one_bricklayer x + rate_other_bricklayer x - 15)
noncomputable def total_time (x : ℕ) : ℚ := 6 * combined_rate_with_efficiency x

theorem number_of_bricks_in_wall (x : ℕ) : total_time x = x → x = 360 :=
by sorry

end NUMINAMATH_GPT_number_of_bricks_in_wall_l1380_138042


namespace NUMINAMATH_GPT_sum_of_possible_remainders_l1380_138098

theorem sum_of_possible_remainders (n : ℕ) (h_even : ∃ k : ℕ, n = 2 * k) : 
  let m := 1000 * (2 * n + 6) + 100 * (2 * n + 4) + 10 * (2 * n + 2) + (2 * n)
  let remainder (k : ℕ) := (1112 * k + 6420) % 29
  23 + 7 + 20 = 50 :=
  by
  sorry

end NUMINAMATH_GPT_sum_of_possible_remainders_l1380_138098


namespace NUMINAMATH_GPT_inequality_solution_set_l1380_138082

theorem inequality_solution_set : 
  { x : ℝ | (1 - x) * (x + 1) ≤ 0 ∧ x ≠ -1 } = { x : ℝ | x < -1 ∨ x ≥ 1 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1380_138082


namespace NUMINAMATH_GPT_line_intersects_x_axis_at_neg3_l1380_138040

theorem line_intersects_x_axis_at_neg3 :
  ∃ (x y : ℝ), (5 * y - 7 * x = 21 ∧ y = 0) ↔ (x = -3 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_at_neg3_l1380_138040


namespace NUMINAMATH_GPT_correct_average_l1380_138052

theorem correct_average (incorrect_avg : ℝ) (num_values : ℕ) (misread_value actual_value : ℝ) 
  (h1 : incorrect_avg = 16) 
  (h2 : num_values = 10)
  (h3 : misread_value = 26)
  (h4 : actual_value = 46) : 
  (incorrect_avg * num_values + (actual_value - misread_value)) / num_values = 18 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_l1380_138052


namespace NUMINAMATH_GPT_circle_equation_k_range_l1380_138080

theorem circle_equation_k_range (k : ℝ) :
  ∀ x y: ℝ, x^2 + y^2 + 4*k*x - 2*y + 4*k^2 - k = 0 →
  k > -1 := 
sorry

end NUMINAMATH_GPT_circle_equation_k_range_l1380_138080


namespace NUMINAMATH_GPT_num_perfect_square_factors_of_180_l1380_138027

theorem num_perfect_square_factors_of_180 (n : ℕ) (h : n = 180) :
  ∃ k : ℕ, k = 4 ∧ ∀ d : ℕ, d ∣ n → ∃ a b c : ℕ, d = 2^a * 3^b * 5^c ∧ a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_num_perfect_square_factors_of_180_l1380_138027


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_bound_sqrt_40_l1380_138035

theorem sum_of_consecutive_integers_bound_sqrt_40 (a b : ℤ) (h₁ : a < Real.sqrt 40) (h₂ : Real.sqrt 40 < b) (h₃ : b = a + 1) : a + b = 13 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_bound_sqrt_40_l1380_138035


namespace NUMINAMATH_GPT_norm_2u_equals_10_l1380_138008

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_norm_2u_equals_10_l1380_138008


namespace NUMINAMATH_GPT_tan_315_eq_neg_1_l1380_138003

theorem tan_315_eq_neg_1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_315_eq_neg_1_l1380_138003


namespace NUMINAMATH_GPT_orchard_total_mass_l1380_138011

def num_gala_trees := 20
def yield_gala_tree := 120
def num_fuji_trees := 10
def yield_fuji_tree := 180
def num_redhaven_trees := 30
def yield_redhaven_tree := 55
def num_elberta_trees := 15
def yield_elberta_tree := 75

def total_mass_gala := num_gala_trees * yield_gala_tree
def total_mass_fuji := num_fuji_trees * yield_fuji_tree
def total_mass_redhaven := num_redhaven_trees * yield_redhaven_tree
def total_mass_elberta := num_elberta_trees * yield_elberta_tree

def total_mass_fruit := total_mass_gala + total_mass_fuji + total_mass_redhaven + total_mass_elberta

theorem orchard_total_mass : total_mass_fruit = 6975 := by
  sorry

end NUMINAMATH_GPT_orchard_total_mass_l1380_138011


namespace NUMINAMATH_GPT_factor_and_sum_coeffs_l1380_138070

noncomputable def sum_of_integer_coeffs_of_factorization (x y : ℤ) : ℤ :=
  let factors := ([(1 : ℤ), (-1 : ℤ), (5 : ℤ), (1 : ℤ), (6 : ℤ), (1 : ℤ), (1 : ℤ), (5 : ℤ), (-1 : ℤ), (6 : ℤ)])
  factors.sum

theorem factor_and_sum_coeffs (x y : ℤ) :
  (125 * (x^9:ℤ) - 216 * (y^9:ℤ) = (x - y) * (5 * x^2 + x * y + 6 * y^2) * (x + y) * (5 * x^2 - x * y + 6 * y^2))
  ∧ (sum_of_integer_coeffs_of_factorization x y = 24) :=
by
  sorry

end NUMINAMATH_GPT_factor_and_sum_coeffs_l1380_138070


namespace NUMINAMATH_GPT_area_parallelogram_l1380_138019

theorem area_parallelogram (AE EB : ℝ) (SAEF SCEF SAEC SBEC SABC SABCD : ℝ) (h1 : SAE = 2 * EB)
  (h2 : SCEF = 1) (h3 : SAE == 2 * SCEF / 3) (h4 : SAEC == SAE + SCEF) 
  (h5 : SBEC == 1/2 * SAEC) (h6 : SABC == SAEC + SBEC) (h7 : SABCD == 2 * SABC) :
  SABCD = 5 := sorry

end NUMINAMATH_GPT_area_parallelogram_l1380_138019


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1380_138099

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a - Real.sin x

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, f' a x > 0 → (a > 1)) ∧ (¬∀ x, f' a x ≥ 0 → (a > 1)) := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1380_138099


namespace NUMINAMATH_GPT_f_2013_value_l1380_138004

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, x ≠ 1 → f (2 * x + 1) + g (3 - x) = x
axiom h2 : ∀ x : ℝ, x ≠ 1 → f ((3 * x + 5) / (x + 1)) + 2 * g ((2 * x + 1) / (x + 1)) = x / (x + 1)

theorem f_2013_value : f 2013 = 1010 / 1007 :=
by
  sorry

end NUMINAMATH_GPT_f_2013_value_l1380_138004


namespace NUMINAMATH_GPT_A_and_B_finish_together_in_20_days_l1380_138092

noncomputable def W_B : ℝ := 1 / 30

noncomputable def W_A : ℝ := 1 / 2 * W_B

noncomputable def W_A_plus_B : ℝ := W_A + W_B

theorem A_and_B_finish_together_in_20_days :
  (1 / W_A_plus_B) = 20 :=
by
  sorry

end NUMINAMATH_GPT_A_and_B_finish_together_in_20_days_l1380_138092


namespace NUMINAMATH_GPT_train_length_proof_l1380_138041

def train_length_crosses_bridge (train_speed_kmh : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  distance - bridge_length_m

theorem train_length_proof : 
  train_length_crosses_bridge 72 150 20 = 250 :=
by
  let train_speed_kmh := 72
  let bridge_length_m := 150
  let crossing_time_s := 20
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  have h : distance = 400 := by sorry
  have h_eq : distance - bridge_length_m = 250 := by sorry
  exact h_eq

end NUMINAMATH_GPT_train_length_proof_l1380_138041


namespace NUMINAMATH_GPT_find_angle_and_area_l1380_138079

theorem find_angle_and_area (a b c : ℝ) (C : ℝ)
  (h₁: (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 2 * a * b)
  (h₂: c = 2)
  (h₃: b = 2 * Real.sqrt 2) : 
  C = Real.pi / 4 ∧ a = 2 ∧ (∃ S : ℝ, S = 1 / 2 * a * c ∧ S = 2) :=
by
  -- We assume sorry here since the focus is on setting up the problem statement correctly
  sorry

end NUMINAMATH_GPT_find_angle_and_area_l1380_138079


namespace NUMINAMATH_GPT_three_kids_savings_l1380_138031

theorem three_kids_savings :
  (200 / 100) + (100 / 20) + (330 / 10) = 40 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_three_kids_savings_l1380_138031


namespace NUMINAMATH_GPT_number_of_girls_l1380_138056

theorem number_of_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 2 * B / 3 + G = 18) : G = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1380_138056


namespace NUMINAMATH_GPT_cookies_left_l1380_138063

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end NUMINAMATH_GPT_cookies_left_l1380_138063


namespace NUMINAMATH_GPT_min_radius_for_area_l1380_138062

theorem min_radius_for_area (r : ℝ) (π : ℝ) (A : ℝ) (h1 : A = 314) (h2 : A = π * r^2) : r ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_min_radius_for_area_l1380_138062


namespace NUMINAMATH_GPT_range_of_m_l1380_138064

theorem range_of_m {m : ℝ} :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1380_138064


namespace NUMINAMATH_GPT_mary_should_drink_six_glasses_per_day_l1380_138047

def daily_water_goal : ℕ := 1500
def glass_capacity : ℕ := 250
def required_glasses (daily_goal : ℕ) (capacity : ℕ) : ℕ := daily_goal / capacity

theorem mary_should_drink_six_glasses_per_day :
  required_glasses daily_water_goal glass_capacity = 6 :=
by
  sorry

end NUMINAMATH_GPT_mary_should_drink_six_glasses_per_day_l1380_138047


namespace NUMINAMATH_GPT_find_dividend_l1380_138014

-- Conditions
def quotient : ℕ := 4
def divisor : ℕ := 4

-- Dividend computation
def dividend (q d : ℕ) : ℕ := q * d

-- Theorem to prove
theorem find_dividend : dividend quotient divisor = 16 := 
by
  -- Placeholder for the proof, not needed as per instructions
  sorry

end NUMINAMATH_GPT_find_dividend_l1380_138014


namespace NUMINAMATH_GPT_right_triangle_sides_l1380_138015

/-- Given a right triangle with area 2 * r^2 / 3 where r is the radius of a circle touching one leg,
the extension of the other leg, and the hypotenuse, the sides of the triangle are given by r, 4/3 * r, and 5/3 * r. -/
theorem right_triangle_sides (r : ℝ) (x y : ℝ)
  (h_area : (x * y) / 2 = 2 * r^2 / 3)
  (h_hypotenuse : (x^2 + y^2) = (2 * r + x - y)^2) :
  x = r ∧ y = 4 * r / 3 :=
sorry

end NUMINAMATH_GPT_right_triangle_sides_l1380_138015


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_a9_l1380_138002

variable (a : ℕ → ℝ)

-- This statement captures if given condition holds, prove a_1 + a_9 = 18.
theorem arithmetic_sequence_a1_a9 (h : a 4 + a 5 + a 6 = 27)
    (h_seq : ∀ (n : ℕ), a (n + 1) = a n + (a 2 - a 1)) :
    a 1 + a 9 = 18 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_a9_l1380_138002


namespace NUMINAMATH_GPT_sheepdog_catches_sheep_in_20_seconds_l1380_138044

noncomputable def speed_sheep : ℝ := 12 -- feet per second
noncomputable def speed_sheepdog : ℝ := 20 -- feet per second
noncomputable def initial_distance : ℝ := 160 -- feet

theorem sheepdog_catches_sheep_in_20_seconds :
  (initial_distance / (speed_sheepdog - speed_sheep)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_sheepdog_catches_sheep_in_20_seconds_l1380_138044


namespace NUMINAMATH_GPT_city_roads_different_colors_l1380_138045

-- Definitions and conditions
def Intersection (α : Type) := α × α × α

def City (α : Type) :=
  { intersections : α → Intersection α // 
    ∀ i : α, ∃ c₁ c₂ c₃ : α, intersections i = (c₁, c₂, c₃) 
    ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₃ ≠ c₁ 
  }

variables {α : Type}

-- Statement to prove that the three roads leading out of the city have different colors
theorem city_roads_different_colors (c : City α) 
  (roads_outside : α → Prop)
  (h : ∃ r₁ r₂ r₃, roads_outside r₁ ∧ roads_outside r₂ ∧ roads_outside r₃ ∧ 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₃ ≠ r₁) : 
  true := 
sorry

end NUMINAMATH_GPT_city_roads_different_colors_l1380_138045


namespace NUMINAMATH_GPT_hexagon_area_l1380_138084

theorem hexagon_area (A C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hC : C = (2 * Real.sqrt 3, 2)) : 
  6 * Real.sqrt 3 = 6 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_hexagon_area_l1380_138084


namespace NUMINAMATH_GPT_bank_balance_after_five_years_l1380_138065

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem bank_balance_after_five_years :
  let P0 := 5600
  let r1 := 0.03
  let r2 := 0.035
  let r3 := 0.04
  let r4 := 0.045
  let r5 := 0.05
  let D := 2000
  let A1 := compoundInterest P0 r1 1 1
  let A2 := compoundInterest A1 r2 1 1
  let A3 := compoundInterest (A2 + D) r3 1 1
  let A4 := compoundInterest A3 r4 1 1
  let A5 := compoundInterest A4 r5 1 1
  A5 = 9094.2 := by
  sorry

end NUMINAMATH_GPT_bank_balance_after_five_years_l1380_138065


namespace NUMINAMATH_GPT_algebra_expression_evaluation_l1380_138060

theorem algebra_expression_evaluation (a b c d e : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : e < 0) 
  (h4 : abs e = 1) : 
  (-a * b) ^ 2009 - (c + d) ^ 2010 - e ^ 2011 = 0 := by 
  sorry

end NUMINAMATH_GPT_algebra_expression_evaluation_l1380_138060


namespace NUMINAMATH_GPT_fractional_equation_solution_l1380_138036

noncomputable def problem_statement (x : ℝ) : Prop :=
  (1 / (x - 1) + 1 = 2 / (x^2 - 1))

theorem fractional_equation_solution :
  ∀ x : ℝ, problem_statement x → x = -2 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_fractional_equation_solution_l1380_138036


namespace NUMINAMATH_GPT_Tony_age_at_end_of_period_l1380_138077

-- Definitions based on the conditions in a):
def hours_per_day := 2
def days_worked := 60
def total_earnings := 1140
def earnings_per_hour (age : ℕ) := age

-- The main property we need to prove: Tony's age at the end of the period is 12 years old
theorem Tony_age_at_end_of_period : ∃ age : ℕ, (2 * age * days_worked = total_earnings) ∧ age = 12 :=
by
  sorry

end NUMINAMATH_GPT_Tony_age_at_end_of_period_l1380_138077


namespace NUMINAMATH_GPT_line_within_plane_l1380_138085

variable (a : Set Point) (α : Set Point)

theorem line_within_plane : a ⊆ α :=
by
  sorry

end NUMINAMATH_GPT_line_within_plane_l1380_138085


namespace NUMINAMATH_GPT_no_solution_for_inequalities_l1380_138090

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_inequalities_l1380_138090


namespace NUMINAMATH_GPT_solve_for_x_l1380_138091

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - y = 7)
  (h2 : x + 3 * y = 7) :
  x = 2.8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1380_138091


namespace NUMINAMATH_GPT_tom_pays_l1380_138013

-- Definitions based on the conditions
def number_of_lessons : Nat := 10
def cost_per_lesson : Nat := 10
def free_lessons : Nat := 2

-- Desired proof statement
theorem tom_pays {number_of_lessons cost_per_lesson free_lessons : Nat} :
  (number_of_lessons - free_lessons) * cost_per_lesson = 80 :=
by
  sorry

end NUMINAMATH_GPT_tom_pays_l1380_138013


namespace NUMINAMATH_GPT_base_of_isosceles_triangle_l1380_138076

theorem base_of_isosceles_triangle (b : ℝ) (h1 : 7 + 7 + b = 22) : b = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_of_isosceles_triangle_l1380_138076


namespace NUMINAMATH_GPT_length_breadth_difference_l1380_138000

theorem length_breadth_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 288) : L - W = 12 :=
by
  sorry

end NUMINAMATH_GPT_length_breadth_difference_l1380_138000


namespace NUMINAMATH_GPT_gun_can_hit_l1380_138075

-- Define the constants
variables (v g : ℝ)

-- Define the coordinates in the first quadrant
variables (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0)

-- Prove the condition for a point (x, y) to be in the region that can be hit by the gun
theorem gun_can_hit (hv : v > 0) (hg : g > 0) :
  y ≤ (v^2 / (2 * g)) - (g * x^2 / (2 * v^2)) :=
sorry

end NUMINAMATH_GPT_gun_can_hit_l1380_138075


namespace NUMINAMATH_GPT_order_of_f_values_l1380_138039

noncomputable def f (x : ℝ) : ℝ := if x >= 1 then 3^x - 1 else 0 -- define f such that it handles the missing part

theorem order_of_f_values :
  (∀ x: ℝ, f (2 - x) = f (1 + x)) ∧ (∀ x: ℝ, x >= 1 → f x = 3^x - 1) →
  f 0 < f 3 ∧ f 3 < f (-2) :=
by
  sorry

end NUMINAMATH_GPT_order_of_f_values_l1380_138039


namespace NUMINAMATH_GPT_sqrt_41_40_39_38_plus_1_l1380_138068

theorem sqrt_41_40_39_38_plus_1 : Real.sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := by
  sorry

end NUMINAMATH_GPT_sqrt_41_40_39_38_plus_1_l1380_138068


namespace NUMINAMATH_GPT_problem1_l1380_138025

variable {a b : ℝ}

theorem problem1 (ha : a > 0) (hb : b > 0) : 
  (1 / (a + b) ≤ 1 / 4 * (1 / a + 1 / b)) :=
sorry

end NUMINAMATH_GPT_problem1_l1380_138025


namespace NUMINAMATH_GPT_sequence_v_20_l1380_138030

noncomputable def sequence_v : ℕ → ℝ → ℝ
| 0, b => b
| (n + 1), b => - (2 / (sequence_v n b + 2))

theorem sequence_v_20 (b : ℝ) (hb : 0 < b) : sequence_v 20 b = -(2 / (b + 2)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_v_20_l1380_138030


namespace NUMINAMATH_GPT_tan_to_sin_cos_l1380_138033

theorem tan_to_sin_cos (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := 
sorry

end NUMINAMATH_GPT_tan_to_sin_cos_l1380_138033


namespace NUMINAMATH_GPT_solution_set_f1_geq_4_min_value_pq_l1380_138037

-- Define the function f(x) for the first question
def f1 (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for part (I)
theorem solution_set_f1_geq_4 (x : ℝ) : f1 x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 4 :=
by
  sorry

-- Define the function f(x) for the second question
def f2 (m x : ℝ) : ℝ := |x - m| + |x - 3|

-- Theorem for part (II)
theorem min_value_pq (p q m : ℝ) (h_pos_p : p > 0) (h_pos_q : q > 0)
    (h_eq : 1 / p + 1 / (2 * q) = m)
    (h_min_f : ∀ x : ℝ, f2 m x ≥ 3) :
    pq = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f1_geq_4_min_value_pq_l1380_138037


namespace NUMINAMATH_GPT_find_alpha_beta_l1380_138067

-- Define the conditions of the problem
variables (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π < β ∧ β < 2 * π)
variable (h_eq : ∀ x : ℝ, cos (x + α) + sin (x + β) + sqrt 2 * cos x = 0)

-- State the required proof as a theorem
theorem find_alpha_beta : α = 3 * π / 4 ∧ β = 7 * π / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_beta_l1380_138067


namespace NUMINAMATH_GPT_joe_total_toy_cars_l1380_138012

def initial_toy_cars : ℕ := 50
def uncle_additional_factor : ℝ := 1.5

theorem joe_total_toy_cars :
  (initial_toy_cars : ℝ) + uncle_additional_factor * initial_toy_cars = 125 := 
by
  sorry

end NUMINAMATH_GPT_joe_total_toy_cars_l1380_138012


namespace NUMINAMATH_GPT_geometric_sequence_properties_l1380_138018

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_properties :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a q →
    a 2 = 6 →
    a 5 - 2 * a 4 - a 3 + 12 = 0 →
    ∀ n, a n = 6 ∨ a n = 6 * (-1)^(n-2) ∨ a n = 6 * 2^(n-2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l1380_138018


namespace NUMINAMATH_GPT_bags_weight_after_removal_l1380_138023

theorem bags_weight_after_removal (sugar_weight salt_weight weight_removed : ℕ) (h1 : sugar_weight = 16) (h2 : salt_weight = 30) (h3 : weight_removed = 4) :
  sugar_weight + salt_weight - weight_removed = 42 := by
  sorry

end NUMINAMATH_GPT_bags_weight_after_removal_l1380_138023


namespace NUMINAMATH_GPT_balloons_lost_is_correct_l1380_138051

def original_balloons : ℕ := 8
def current_balloons : ℕ := 6
def lost_balloons : ℕ := original_balloons - current_balloons

theorem balloons_lost_is_correct : lost_balloons = 2 := by
  sorry

end NUMINAMATH_GPT_balloons_lost_is_correct_l1380_138051


namespace NUMINAMATH_GPT_alice_bob_numbers_sum_l1380_138024

-- Fifty slips of paper numbered 1 to 50 are placed in a hat.
-- Alice and Bob each draw one number from the hat without replacement, keeping their numbers hidden from each other.
-- Alice cannot tell who has the larger number.
-- Bob knows who has the larger number.
-- Bob's number is composite.
-- If Bob's number is multiplied by 50 and Alice's number is added, the result is a perfect square.
-- Prove that the sum of Alice's and Bob's numbers is 29.

theorem alice_bob_numbers_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 50) (hB : 1 ≤ B ∧ B ≤ 50) 
  (hAB_distinct : A ≠ B) (hA_unknown : ¬(A = 1 ∨ A = 50))
  (hB_composite : ∃ d > 1, d < B ∧ B % d = 0) (h_perfect_square : ∃ k, 50 * B + A = k ^ 2) :
  A + B = 29 := by
  sorry

end NUMINAMATH_GPT_alice_bob_numbers_sum_l1380_138024


namespace NUMINAMATH_GPT_probability_of_valid_number_l1380_138095

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≠ j → (n % (10^i) / 10^(i-1)) ≠ (n % (10^j) / 10^(j-1))

def digits_in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_number (n : ℕ) : Prop :=
  is_even n ∧ has_distinct_digits n ∧ digits_in_range n

noncomputable def count_valid_numbers : ℕ :=
  2296

noncomputable def total_numbers : ℕ :=
  9000

theorem probability_of_valid_number :
  (count_valid_numbers : ℚ) / total_numbers = 574 / 2250 :=
by sorry

end NUMINAMATH_GPT_probability_of_valid_number_l1380_138095


namespace NUMINAMATH_GPT_total_weight_fruits_in_good_condition_l1380_138073

theorem total_weight_fruits_in_good_condition :
  let oranges_initial := 600
  let bananas_initial := 400
  let apples_initial := 300
  let avocados_initial := 200
  let grapes_initial := 100
  let pineapples_initial := 50

  let oranges_rotten := 0.15 * oranges_initial
  let bananas_rotten := 0.05 * bananas_initial
  let apples_rotten := 0.08 * apples_initial
  let avocados_rotten := 0.10 * avocados_initial
  let grapes_rotten := 0.03 * grapes_initial
  let pineapples_rotten := 0.20 * pineapples_initial

  let oranges_good := oranges_initial - oranges_rotten
  let bananas_good := bananas_initial - bananas_rotten
  let apples_good := apples_initial - apples_rotten
  let avocados_good := avocados_initial - avocados_rotten
  let grapes_good := grapes_initial - grapes_rotten
  let pineapples_good := pineapples_initial - pineapples_rotten

  let weight_per_orange := 150 / 1000 -- kg
  let weight_per_banana := 120 / 1000 -- kg
  let weight_per_apple := 100 / 1000 -- kg
  let weight_per_avocado := 80 / 1000 -- kg
  let weight_per_grape := 5 / 1000 -- kg
  let weight_per_pineapple := 1 -- kg

  oranges_good * weight_per_orange +
  bananas_good * weight_per_banana +
  apples_good * weight_per_apple +
  avocados_good * weight_per_avocado +
  grapes_good * weight_per_grape +
  pineapples_good * weight_per_pineapple = 204.585 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_fruits_in_good_condition_l1380_138073


namespace NUMINAMATH_GPT_least_possible_sum_l1380_138057

theorem least_possible_sum
  (a b x y z : ℕ)
  (hpos_a : 0 < a) (hpos_b : 0 < b)
  (hpos_x : 0 < x) (hpos_y : 0 < y)
  (hpos_z : 0 < z)
  (h : 3 * a = 7 * b ∧ 7 * b = 5 * x ∧ 5 * x = 4 * y ∧ 4 * y = 6 * z) :
  a + b + x + y + z = 459 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_sum_l1380_138057


namespace NUMINAMATH_GPT_f_of_integral_ratio_l1380_138005

variable {f : ℝ → ℝ} (h_cont : ∀ x > 0, continuous_at f x)
variable (h_int : ∀ a b : ℝ, a > 0 → b > 0 → ∃ g : ℝ → ℝ, (∫ x in a..b, f x) = g (b / a))

theorem f_of_integral_ratio :
  (∃ c : ℝ, ∀ x > 0, f x = c / x) :=
sorry

end NUMINAMATH_GPT_f_of_integral_ratio_l1380_138005


namespace NUMINAMATH_GPT_symmetric_point_correct_line_passes_second_quadrant_l1380_138010

theorem symmetric_point_correct (x y: ℝ) (h_line : y = x + 1) :
  (x, y) = (-1, 2) :=
sorry

theorem line_passes_second_quadrant (m x y: ℝ) (h_line: m * x + y + m - 1 = 0) :
  (x, y) = (-1, 1) :=
sorry

end NUMINAMATH_GPT_symmetric_point_correct_line_passes_second_quadrant_l1380_138010


namespace NUMINAMATH_GPT_min_value_expression_l1380_138048

open Real

/-- 
  Given that the function y = log_a(2x+3) - 4 passes through a fixed point P and the fixed point P lies on the line l: ax + by + 7 = 0,
  prove the minimum value of 1/(a+2) + 1/(4b) is 4/9, where a > 0, a ≠ 1, and b > 0.
-/
theorem min_value_expression (a b : ℝ) (h_a : 0 < a) (h_a_ne_1 : a ≠ 1) (h_b : 0 < b)
  (h_eqn : (a * -1 + b * -4 + 7 = 0) → (a + 2 + 4 * b = 9)):
  (1 / (a + 2) + 1 / (4 * b)) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1380_138048


namespace NUMINAMATH_GPT_probability_of_different_topics_l1380_138043

theorem probability_of_different_topics (n : ℕ) (m : ℕ) (prob : ℚ)
  (h1 : n = 36)
  (h2 : m = 30)
  (h3 : prob = 5/6) :
  (m : ℚ) / (n : ℚ) = prob :=
sorry

end NUMINAMATH_GPT_probability_of_different_topics_l1380_138043


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1380_138034

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a < 1) → ((a + 1) * (a - 2) < 0) ∧ ((∃ b : ℝ, (b + 1) * (b - 2) < 0 ∧ ¬(0 < b ∧ b < 1))) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1380_138034


namespace NUMINAMATH_GPT_sum_fourth_power_l1380_138050

  theorem sum_fourth_power (x y z : ℝ) 
    (h1 : x + y + z = 2) 
    (h2 : x^2 + y^2 + z^2 = 6) 
    (h3 : x^3 + y^3 + z^3 = 8) : 
    x^4 + y^4 + z^4 = 26 := 
  by 
    sorry
  
end NUMINAMATH_GPT_sum_fourth_power_l1380_138050


namespace NUMINAMATH_GPT_olivia_hourly_rate_l1380_138053

theorem olivia_hourly_rate (h_worked_monday : ℕ) (h_worked_wednesday : ℕ) (h_worked_friday : ℕ) (h_total_payment : ℕ) (h_total_hours : h_worked_monday + h_worked_wednesday + h_worked_friday = 13) (h_total_amount : h_total_payment = 117) :
  h_total_payment / (h_worked_monday + h_worked_wednesday + h_worked_friday) = 9 :=
by
  sorry

end NUMINAMATH_GPT_olivia_hourly_rate_l1380_138053


namespace NUMINAMATH_GPT_perpendicular_vectors_parallel_vectors_l1380_138028

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end NUMINAMATH_GPT_perpendicular_vectors_parallel_vectors_l1380_138028


namespace NUMINAMATH_GPT_angle_of_inclination_l1380_138074

theorem angle_of_inclination (α : ℝ) (h: 0 ≤ α ∧ α < 180) (slope_eq : Real.tan (Real.pi * α / 180) = Real.sqrt 3) :
  α = 60 :=
sorry

end NUMINAMATH_GPT_angle_of_inclination_l1380_138074


namespace NUMINAMATH_GPT_number_is_multiple_of_15_l1380_138006

theorem number_is_multiple_of_15
  (W X Y Z D : ℤ)
  (h1 : X - W = 1)
  (h2 : Y - W = 9)
  (h3 : Y - X = 8)
  (h4 : Z - W = 11)
  (h5 : Z - X = 10)
  (h6 : Z - Y = 2)
  (hD : D - X = 5) :
  15 ∣ D :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_number_is_multiple_of_15_l1380_138006


namespace NUMINAMATH_GPT_maximum_k_inequality_l1380_138066

open Real

noncomputable def inequality_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : Prop :=
  (x / sqrt (y + z)) + (y / sqrt (z + x)) + (z / sqrt (x + y)) ≥ sqrt (3 / 2) * sqrt (x + y + z)
 
-- This is the theorem statement
theorem maximum_k_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  inequality_problem x y z h1 h2 h3 :=
  sorry

end NUMINAMATH_GPT_maximum_k_inequality_l1380_138066


namespace NUMINAMATH_GPT_find_candy_bars_per_week_l1380_138029

-- Define the conditions
variables (x : ℕ)

-- Condition: Kim's dad buys Kim x candy bars each week
def candies_bought := 16 * x

-- Condition: Kim eats one candy bar every 4 weeks
def candies_eaten := 16 / 4

-- Condition: After 16 weeks, Kim has saved 28 candy bars
def saved_candies := 28

-- The theorem we want to prove
theorem find_candy_bars_per_week : (16 * x - (16 / 4) = 28) → x = 2 := by
  -- We will skip the actual proof for now.
  sorry

end NUMINAMATH_GPT_find_candy_bars_per_week_l1380_138029


namespace NUMINAMATH_GPT_sector_area_proof_l1380_138032

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_proof
  (r : ℝ) (l : ℝ) (perimeter : ℝ) (theta : ℝ) (h1 : perimeter = 2 * r + l)
  (h2 : l = r * theta) (h3 : perimeter = 16) (h4 : theta = 2) :
  sector_area r theta = 16 := by
  sorry

end NUMINAMATH_GPT_sector_area_proof_l1380_138032


namespace NUMINAMATH_GPT_f_2019_value_l1380_138022

noncomputable def f : ℕ → ℕ := sorry

theorem f_2019_value
  (h : ∀ m n : ℕ, f (m + n) ≥ f m + f (f n) - 1) :
  f 2019 = 2019 :=
sorry

end NUMINAMATH_GPT_f_2019_value_l1380_138022


namespace NUMINAMATH_GPT_simplify_expression_l1380_138017

theorem simplify_expression (a b : ℤ) : 
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1380_138017


namespace NUMINAMATH_GPT_ma_m_gt_mb_l1380_138088

theorem ma_m_gt_mb (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m * a > m * b) → m ≥ 0 := 
  sorry

end NUMINAMATH_GPT_ma_m_gt_mb_l1380_138088


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1380_138020

theorem simplify_and_evaluate 
  (a b : ℚ) (h_a : a = -1/3) (h_b : b = -3) : 
  2 * (3 * a^2 * b - a * b^2) - (a * b^2 + 6 * a^2 * b) = 9 := 
  by 
    rw [h_a, h_b]
    sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1380_138020


namespace NUMINAMATH_GPT_petya_friends_l1380_138096

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end NUMINAMATH_GPT_petya_friends_l1380_138096


namespace NUMINAMATH_GPT_exclude_domain_and_sum_l1380_138049

noncomputable def g (x : ℝ) : ℝ :=
  1 / (2 + 1 / (2 + 1 / x))

theorem exclude_domain_and_sum :
  { x : ℝ | x = 0 ∨ x = -1/2 ∨ x = -1/4 } = { x : ℝ | ¬(x ≠ 0 ∧ (2 + 1 / x ≠ 0) ∧ (2 + 1 / (2 + 1 / x) ≠ 0)) } ∧
  (0 + (-1 / 2) + (-1 / 4) = -3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_exclude_domain_and_sum_l1380_138049


namespace NUMINAMATH_GPT_num_readers_sci_fiction_l1380_138089

theorem num_readers_sci_fiction (T L B S: ℕ) (hT: T = 250) (hL: L = 88) (hB: B = 18) (hTotal: T = S + L - B) : 
  S = 180 := 
by 
  sorry

end NUMINAMATH_GPT_num_readers_sci_fiction_l1380_138089


namespace NUMINAMATH_GPT_units_digit_of_product_composites_l1380_138026

def is_composite (n : ℕ) : Prop := 
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

theorem units_digit_of_product_composites (h1 : is_composite 9) (h2 : is_composite 10) (h3 : is_composite 12) :
  (9 * 10 * 12) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_product_composites_l1380_138026


namespace NUMINAMATH_GPT_optimal_garden_dimensions_l1380_138097

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), (2 * l + 2 * w = 400 ∧
                l ≥ 100 ∧
                w ≥ 0 ∧ 
                l * w = 10000) :=
by
  sorry

end NUMINAMATH_GPT_optimal_garden_dimensions_l1380_138097


namespace NUMINAMATH_GPT_present_age_of_son_l1380_138007

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 32) (h2 : M + 2 = 2 * (S + 2)) : S = 30 :=
by
  sorry

end NUMINAMATH_GPT_present_age_of_son_l1380_138007


namespace NUMINAMATH_GPT_cost_of_soda_l1380_138058

-- Define the system of equations
theorem cost_of_soda (b s f : ℕ): 
  3 * b + s = 390 ∧ 
  2 * b + 3 * s = 440 ∧ 
  b + 2 * f = 230 ∧ 
  s + 3 * f = 270 → 
  s = 234 := 
by 
  sorry

end NUMINAMATH_GPT_cost_of_soda_l1380_138058


namespace NUMINAMATH_GPT_inequality_problem_l1380_138071

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_problem_l1380_138071


namespace NUMINAMATH_GPT_andrew_eggs_l1380_138069

def andrew_eggs_problem (a b : ℕ) (half_eggs_given_away : ℚ) (remaining_eggs : ℕ) : Prop :=
  a + b - (a + b) * half_eggs_given_away = remaining_eggs

theorem andrew_eggs :
  andrew_eggs_problem 8 62 (1/2 : ℚ) 35 :=
by
  sorry

end NUMINAMATH_GPT_andrew_eggs_l1380_138069


namespace NUMINAMATH_GPT_Trent_tears_l1380_138083

def onions_per_pot := 4
def pots_of_soup := 6
def tears_per_3_onions := 2

theorem Trent_tears:
  (onions_per_pot * pots_of_soup) / 3 * tears_per_3_onions = 16 :=
by
  sorry

end NUMINAMATH_GPT_Trent_tears_l1380_138083


namespace NUMINAMATH_GPT_maria_tom_weather_probability_l1380_138087

noncomputable def probability_exactly_two_clear_days (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * (p ^ (n - 2)) * ((1 - p) ^ 2)

theorem maria_tom_weather_probability :
  probability_exactly_two_clear_days 0.6 5 = 1080 / 3125 :=
by
  sorry

end NUMINAMATH_GPT_maria_tom_weather_probability_l1380_138087


namespace NUMINAMATH_GPT_sequence_solution_l1380_138046

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 5 ∧ a 8 = 8 ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 20) ∧
  (a 1 = 5 ∧ a 2 = 8 ∧ a 3 = 7 ∧ a 4 = 5 ∧ a 5 = 8 ∧ a 6 = 7 ∧ a 7 = 5 ∧ a 8 = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_solution_l1380_138046


namespace NUMINAMATH_GPT_min_balls_in_circle_l1380_138016

theorem min_balls_in_circle (b w n k : ℕ) 
  (h1 : b = 2 * w)
  (h2 : n = b + w) 
  (h3 : n - 2 * k = 6 * k) :
  n >= 24 :=
sorry

end NUMINAMATH_GPT_min_balls_in_circle_l1380_138016
