import Mathlib

namespace Y_pdf_from_X_pdf_l110_110138

/-- Given random variable X with PDF p(x), prove PDF of Y = X^3 -/
noncomputable def X_pdf (σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2))

noncomputable def Y_pdf (σ : ℝ) (y : ℝ) : ℝ :=
  (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2))

theorem Y_pdf_from_X_pdf (σ : ℝ) (y : ℝ) :
  ∀ x : ℝ, X_pdf σ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2)) →
  Y_pdf σ y = (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2)) :=
sorry

end Y_pdf_from_X_pdf_l110_110138


namespace discount_rate_l110_110346

variable (P P_b P_s D : ℝ)

-- Conditions
variable (h1 : P_s = 1.24 * P)
variable (h2 : P_s = 1.55 * P_b)
variable (h3 : P_b = P * (1 - D))

theorem discount_rate :
  D = 0.2 :=
by
  sorry

end discount_rate_l110_110346


namespace number_of_impossible_d_l110_110442

-- Define the problem parameters and conditions
def perimeter_diff (t s : ℕ) : ℕ := 3 * t - 4 * s
def side_diff (t s d : ℕ) : ℕ := t - s - d
def square_perimeter_positive (s : ℕ) : Prop := s > 0

-- Define the proof problem
theorem number_of_impossible_d (t s d : ℕ) (h1 : perimeter_diff t s = 1575) (h2 : side_diff t s d = 0) (h3 : square_perimeter_positive s) : 
    ∃ n, n = 525 ∧ ∀ d, d ≤ 525 → ¬ (3 * d > 1575) :=
    sorry

end number_of_impossible_d_l110_110442


namespace convex_polyhedron_theorems_l110_110462

-- Definitions for convex polyhedron and symmetric properties
structure ConvexSymmetricPolyhedron (α : Type*) :=
  (isConvex : Bool)
  (isCentrallySymmetric : Bool)
  (crossSection : α → α → α)
  (center : α)

-- Definitions for proofs required
def largest_cross_section_area
  (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ∀ (p : ℝ), P.crossSection p P.center ≤ P.crossSection P.center P.center

def largest_radius_circle (P : ConvexSymmetricPolyhedron ℝ) : Prop :=
  ¬∀ (p : ℝ), P.crossSection p P.center = P.crossSection P.center P.center

-- The theorem combining both statements
theorem convex_polyhedron_theorems
  (P : ConvexSymmetricPolyhedron ℝ) :
  P.isConvex = true ∧ 
  P.isCentrallySymmetric = true →
  (largest_cross_section_area P) ∧ (largest_radius_circle P) :=
by 
  sorry

end convex_polyhedron_theorems_l110_110462


namespace coin_flips_sequences_count_l110_110727

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110727


namespace solve_b_values_l110_110046

open Int

theorem solve_b_values :
  {b : ℤ | ∃ x1 x2 x3 : ℤ, x1^2 + b * x1 - 2 ≤ 0 ∧ x2^2 + b * x2 - 2 ≤ 0 ∧ x3^2 + b * x3 - 2 ≤ 0 ∧
  ∀ x : ℤ, x ≠ x1 ∧ x ≠ x2 ∧ x ≠ x3 → x^2 + b * x - 2 > 0} = { -4, -3 } :=
by sorry

end solve_b_values_l110_110046


namespace distinct_real_roots_iff_l110_110938

noncomputable def operation (a b : ℝ) : ℝ := a * b^2 - b 

theorem distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ operation 1 x1 = k ∧ operation 1 x2 = k) ↔ k > -1/4 :=
by
  sorry

end distinct_real_roots_iff_l110_110938


namespace solve_equation_l110_110621

theorem solve_equation {x : ℂ} : (x - 2)^4 + (x - 6)^4 = 272 →
  x = 6 ∨ x = 2 ∨ x = 4 + 2 * Complex.I ∨ x = 4 - 2 * Complex.I :=
by
  intro h
  sorry

end solve_equation_l110_110621


namespace range_function_l110_110548

open Real

noncomputable def function_to_prove (x : ℝ) (a : ℕ) : ℝ := x + 2 * a / x

theorem range_function (a : ℕ) (h1 : a^2 - a < 2) (h2 : a ≠ 0) : 
  Set.range (function_to_prove · a) = {y : ℝ | y ≤ -2 * sqrt 2} ∪ {y : ℝ | y ≥ 2 * sqrt 2} :=
by
  sorry

end range_function_l110_110548


namespace neg_proposition_P_l110_110572

theorem neg_proposition_P : 
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
by
  sorry

end neg_proposition_P_l110_110572


namespace smallest_positive_x_for_maximum_sine_sum_l110_110033

theorem smallest_positive_x_for_maximum_sine_sum :
  ∃ x : ℝ, (0 < x) ∧ (∃ k m : ℕ, x = 450 + 1800 * k ∧ x = 630 + 2520 * m ∧ x = 12690) := by
  sorry

end smallest_positive_x_for_maximum_sine_sum_l110_110033


namespace values_of_x_plus_y_l110_110958

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end values_of_x_plus_y_l110_110958


namespace x_y_sum_vals_l110_110960

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end x_y_sum_vals_l110_110960


namespace coin_flip_sequences_l110_110710

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110710


namespace evaluate_expression_l110_110527

-- Define the base and the exponents
def base : ℝ := 64
def exponent1 : ℝ := 0.125
def exponent2 : ℝ := 0.375
def combined_result : ℝ := 8

-- Statement of the problem
theorem evaluate_expression : (base^exponent1) * (base^exponent2) = combined_result := 
by 
  sorry

end evaluate_expression_l110_110527


namespace factor_expression_l110_110186

variable {a : ℝ}

theorem factor_expression :
  ((10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32)) = 4 * (3 * a^3 * (a - 12) - 16) :=
by
  sorry

end factor_expression_l110_110186


namespace gcd_of_differences_l110_110535

theorem gcd_of_differences (a b c : ℕ) (h1 : a = 794) (h2 : b = 858) (h3 : c = 1351) : 
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a) = 4 :=
by
  sorry

end gcd_of_differences_l110_110535


namespace S_30_zero_l110_110581

variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {n : ℕ} 

-- Definitions corresponding to the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a_n n = a1 + d * n

def sum_arithmetic_sequence (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d
  
-- The given conditions
axiom S_eq (S_10 S_20 : ℝ) : S 10 = S 20

-- The theorem we need to prove
theorem S_30_zero (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_arithmetic_sequence S a_n)
  (h_eq : S 10 = S 20) :
  S 30 = 0 :=
sorry

end S_30_zero_l110_110581


namespace binomial_expansion_coefficient_x_l110_110971

theorem binomial_expansion_coefficient_x :
  (∃ (c : ℕ), (x : ℝ) → (x + 1/x^(1/2))^7 = c * x + (rest)) ∧ c = 35 := by
  sorry

end binomial_expansion_coefficient_x_l110_110971


namespace math_proof_problem_l110_110047

noncomputable def problem_statement : Prop :=
  let a_bound := 14
  let b_bound := 7
  let c_bound := 14
  let num_square_divisors := (a_bound / 2 + 1) * (b_bound / 2 + 1) * (c_bound / 2 + 1)
  let num_cube_divisors := (a_bound / 3 + 1) * (b_bound / 3 + 1) * (c_bound / 3 + 1)
  let num_sixth_power_divisors := (a_bound / 6 + 1) * (b_bound / 6 + 1) * (c_bound / 6 + 1)
  
  num_square_divisors + num_cube_divisors - num_sixth_power_divisors = 313

theorem math_proof_problem : problem_statement := by sorry

end math_proof_problem_l110_110047


namespace box_area_ratio_l110_110912

theorem box_area_ratio 
  (l w h : ℝ)
  (V : l * w * h = 5184)
  (A1 : w * h = (1/2) * l * w)
  (A2 : l * h = 288):
  (l * w) / (l * h) = 3 / 2 := 
by
  sorry

end box_area_ratio_l110_110912


namespace evaluate_polynomial_at_three_l110_110146

def polynomial (x : ℕ) : ℕ :=
  x^6 + 2 * x^5 + 4 * x^3 + 5 * x^2 + 6 * x + 12

theorem evaluate_polynomial_at_three :
  polynomial 3 = 588 :=
by
  sorry

end evaluate_polynomial_at_three_l110_110146


namespace probability_at_least_two_green_l110_110593

def total_apples := 10
def red_apples := 6
def green_apples := 4
def choose_apples := 3

noncomputable def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_at_least_two_green :
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) = 40 ∧ 
  binomial total_apples choose_apples = 120 ∧
  (binomial green_apples 3 + binomial green_apples 2 * binomial red_apples 1) / binomial total_apples choose_apples = 1 / 3 := by
sorry

end probability_at_least_two_green_l110_110593


namespace find_k_l110_110976

noncomputable def f (a b c x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_k
  (a b c : ℤ)
  (k : ℤ)
  (h1 : f a b c 1 = 0)
  (h2 : 50 < f a b c 7)
  (h3 : f a b c 7 < 60)
  (h4 : 70 < f a b c 8)
  (h5 : f a b c 8 < 80)
  (h6 : 5000 * k < f a b c 100)
  (h7 : f a b c 100 < 5000 * (k + 1)) :
  k = 3 :=
sorry

end find_k_l110_110976


namespace coin_flip_sequences_l110_110744

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110744


namespace common_chord_equation_l110_110133

-- Definitions of the given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 12 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 14*y + 15 = 0

-- Definition of the common chord line
def common_chord_line (x y : ℝ) : Prop := 6*x + 8*y - 3 = 0

-- The theorem to be proved
theorem common_chord_equation :
  (∀ x y, circle1 x y → circle2 x y → common_chord_line x y) :=
by sorry

end common_chord_equation_l110_110133


namespace remainder_n_plus_2023_mod_7_l110_110668

theorem remainder_n_plus_2023_mod_7 (n : ℤ) (h : n % 7 = 2) : (n + 2023) % 7 = 2 :=
by
  sorry

end remainder_n_plus_2023_mod_7_l110_110668


namespace complete_square_h_l110_110246

theorem complete_square_h (x h : ℝ) :
  (∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) → h = -3 / 2 :=
by
  sorry

end complete_square_h_l110_110246


namespace min_links_for_weights_l110_110657

def min_links_to_break (n : ℕ) : ℕ :=
  if n = 60 then 3 else sorry

theorem min_links_for_weights (n : ℕ) (h1 : n = 60) :
  min_links_to_break n = 3 :=
by
  rw [h1]
  trivial

end min_links_for_weights_l110_110657


namespace complex_powers_i_l110_110357

theorem complex_powers_i (i : ℂ) (h : i^2 = -1) :
  (i^123 - i^321 + i^432 = -2 * i + 1) :=
by
  -- sorry to skip the proof
  sorry

end complex_powers_i_l110_110357


namespace cylinder_volume_l110_110549

theorem cylinder_volume (r h V: ℝ) (r_pos: r = 4) (lateral_area: 2 * 3.14 * r * h = 62.8) : 
    V = 125600 :=
by
  sorry

end cylinder_volume_l110_110549


namespace expected_number_of_games_is_correct_l110_110846

noncomputable def expected_number_of_games : ℚ := 
  let p_A := 2/3
  let p_B := 1/3
  let P_ξ_2 := (p_A ^ 2 + p_B ^ 2) 
  let P_ξ_4 := (4/9) * P_ξ_2 
  let P_ξ_6 := (4/9)^2
  (2 * P_ξ_2 + 4 * P_ξ_4 + 6 * P_ξ_6)

theorem expected_number_of_games_is_correct :
  expected_number_of_games = 266/81 := 
by
  sorry

end expected_number_of_games_is_correct_l110_110846


namespace kolya_mistaken_l110_110656

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l110_110656


namespace fill_time_is_13_seconds_l110_110884

-- Define the given conditions as constants
def flow_rate_in (t : ℝ) : ℝ := 24 * t -- 24 gallons/second
def leak_rate (t : ℝ) : ℝ := 4 * t -- 4 gallons/second
def basin_capacity : ℝ := 260 -- 260 gallons

-- Main theorem to be proven
theorem fill_time_is_13_seconds : 
  ∀ t : ℝ, (flow_rate_in t - leak_rate t) * (13) = basin_capacity := 
sorry

end fill_time_is_13_seconds_l110_110884


namespace radical_conjugate_sum_l110_110503

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l110_110503


namespace train_speed_l110_110172

/-- Given that a train crosses a pole in 12 seconds and the length of the train is 200 meters,
prove that the speed of the train in km/hr is 60. -/
theorem train_speed 
  (cross_time : ℝ) (train_length : ℝ)
  (H1 : cross_time = 12) (H2 : train_length = 200) : 
  let distance_km := train_length / 1000
      time_hr := cross_time / 3600
      speed := distance_km / time_hr in
  speed = 60 :=
by
  sorry

end train_speed_l110_110172


namespace num_children_l110_110625

-- Defining the conditions
def num_adults : Nat := 10
def price_adult_ticket : Nat := 8
def total_bill : Nat := 124
def price_child_ticket : Nat := 4

-- Statement to prove: Number of children
theorem num_children (num_adults : Nat) (price_adult_ticket : Nat) (total_bill : Nat) (price_child_ticket : Nat) : Nat :=
  let cost_adults := num_adults * price_adult_ticket
  let cost_child := total_bill - cost_adults
  cost_child / price_child_ticket

example : num_children 10 8 124 4 = 11 := sorry

end num_children_l110_110625


namespace no_five_distinct_natural_numbers_feasible_l110_110191

theorem no_five_distinct_natural_numbers_feasible :
  ¬ ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ d * e = a + b + c + d + e := by
  sorry

end no_five_distinct_natural_numbers_feasible_l110_110191


namespace coin_flip_sequences_l110_110695

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110695


namespace BF_bisects_angle_PBC_tan_angle_PCB_l110_110101

variable {P B C A D E F : Type*}

/-- Problem (a) -/
theorem BF_bisects_angle_PBC (h1 : ∠ PBC = 60) 
  (h2 : tangent P (circumcircle ⟨P, B, C⟩) ∩ CB = A)
  (h3 : D ∈ PA ∧ E ∈ circumcircle ⟨P, B, C⟩ ∧ ∠DBE = 90 ∧ PD = PE)
  (h4 : BE ∩ PC = F)
  (h5 : are_concurrent [AF, BP, CD]) :
  is_angle_bisector BF (∠ PBC) := sorry

/-- Problem (b) -/
theorem tan_angle_PCB (h1 : ∠ PBC = 60) 
  (h2 : tangent P (circumcircle ⟨P, B, C⟩) ∩ CB = A)
  (h3 : D ∈ PA ∧ E ∈ circumcircle ⟨P, B, C⟩ ∧ ∠DBE = 90 ∧ PD = PE)
  (h4 : BE ∩ PC = F)
  (h5 : are_concurrent [AF, BP, CD]) :
  tan (∠ PCB) = (6 + sqrt 3) / 11 := sorry

end BF_bisects_angle_PBC_tan_angle_PCB_l110_110101


namespace factorize_expression_l110_110364

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l110_110364


namespace inverse_cos_plus_one_l110_110437

noncomputable def f (x : ℝ) : ℝ := Real.cos x + 1

theorem inverse_cos_plus_one (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) :
    f (-(Real.arccos (x - 1))) = x :=
by
  sorry

end inverse_cos_plus_one_l110_110437


namespace distinct_sequences_ten_flips_l110_110713

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110713


namespace jar_filling_fraction_l110_110141

theorem jar_filling_fraction (C1 C2 C3 W : ℝ)
  (h1 : W = (1/7) * C1)
  (h2 : W = (2/9) * C2)
  (h3 : W = (3/11) * C3)
  (h4 : C3 > C1 ∧ C3 > C2) :
  (3 * W) = (9 / 11) * C3 :=
by sorry

end jar_filling_fraction_l110_110141


namespace estimate_red_balls_l110_110400

-- Define the conditions
variable (total_balls : ℕ)
variable (prob_red_ball : ℝ)
variable (frequency_red_ball : ℝ := prob_red_ball)

-- Assume total number of balls in the bag is 20
axiom total_balls_eq_20 : total_balls = 20

-- Assume the probability (or frequency) of drawing a red ball
axiom prob_red_ball_eq_0_25 : prob_red_ball = 0.25

-- The Lean statement
theorem estimate_red_balls (H1 : total_balls = 20) (H2 : prob_red_ball = 0.25) : total_balls * prob_red_ball = 5 :=
by
  rw [H1, H2]
  norm_num
  sorry

end estimate_red_balls_l110_110400


namespace total_area_of_sheet_l110_110165

theorem total_area_of_sheet (x : ℕ) (h1 : 4 * x - x = 2208) : x + 4 * x = 3680 := 
sorry

end total_area_of_sheet_l110_110165


namespace roots_problem_l110_110280

noncomputable def polynomial_roots : Prop := 
  ∀ (p q : ℝ), 
  (p + q = 6) ∧ 
  (p * q = 8) → 
  (p^3 + (p^4 * q^2) + (p^2 * q^4) + q^3 = 1352)

theorem roots_problem : polynomial_roots := 
by
  dsimp [polynomial_roots]
  intros p q h
  apply sorry

end roots_problem_l110_110280


namespace coin_flip_sequences_l110_110687

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110687


namespace find_C_l110_110456

-- Variables and conditions
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + B + C = 1000
def condition2 : Prop := A + C = 700
def condition3 : Prop := B + C = 600

-- The statement to be proved
theorem find_C (h1 : condition1 A B C) (h2 : condition2 A C) (h3 : condition3 B C) : C = 300 :=
sorry

end find_C_l110_110456


namespace rectangle_perimeter_gt_16_l110_110066

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l110_110066


namespace find_side_b_l110_110279

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end find_side_b_l110_110279


namespace solve_for_r_l110_110790

theorem solve_for_r : ∃ r : ℝ, r ≠ 4 ∧ r ≠ 5 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 10) / (r^2 - 2*r - 15) ↔ 
  r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 := 
by {
  sorry
}

end solve_for_r_l110_110790


namespace find_least_positive_integer_l110_110925

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l110_110925


namespace average_age_increase_l110_110300

variable (A : ℝ) -- Original average age of 8 men
variable (age1 age2 : ℝ) -- The ages of the two men being replaced
variable (avg_women : ℝ) -- The average age of the two women

-- Conditions as hypotheses
def conditions : Prop :=
  8 * A - age1 - age2 + avg_women * 2 = 8 * (A + 2)

-- The theorem that needs to be proved
theorem average_age_increase (h1 : age1 = 20) (h2 : age2 = 28) (h3 : avg_women = 32) (h4 : conditions A age1 age2 avg_women) : (8 * A + 16) / 8 - A = 2 :=
by
  sorry

end average_age_increase_l110_110300


namespace run_time_difference_l110_110756

variables (distance duration_injured : ℝ) (initial_speed : ℝ)

theorem run_time_difference (H1 : distance = 20) 
                            (H2 : duration_injured = 22) 
                            (H3 : initial_speed = distance * 2 / duration_injured) :
                            duration_injured - (distance / initial_speed) = 11 :=
by
  sorry

end run_time_difference_l110_110756


namespace A_inter_B_eq_A_A_union_B_l110_110981

-- Definitions for sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + 3 * a = (a + 3) * x}
def B : Set ℝ := {x | x^2 + 3 = 4 * x}

-- Proof problem for part (1)
theorem A_inter_B_eq_A (a : ℝ) : (A a ∩ B = A a) ↔ (a = 1 ∨ a = 3) :=
by
  sorry

-- Proof problem for part (2)
theorem A_union_B (a : ℝ) : A a ∪ B = if a = 1 then {1, 3} else if a = 3 then {1, 3} else {a, 1, 3} :=
by
  sorry

end A_inter_B_eq_A_A_union_B_l110_110981


namespace find_k_l110_110555

theorem find_k (k : ℝ) :
  (∃ x y : ℝ, y = x + 2 * k ∧ y = 2 * x + k + 1 ∧ x^2 + y^2 = 4) ↔
  (k = 1 ∨ k = -1/5) := 
sorry

end find_k_l110_110555


namespace fluffy_striped_or_spotted_cats_l110_110972

theorem fluffy_striped_or_spotted_cats (total_cats : ℕ) (striped_fraction : ℚ) (spotted_fraction : ℚ)
    (fluffy_striped_fraction : ℚ) (fluffy_spotted_fraction : ℚ) (striped_spotted_fraction : ℚ) :
    total_cats = 180 ∧ striped_fraction = 1/2 ∧ spotted_fraction = 1/3 ∧
    fluffy_striped_fraction = 1/8 ∧ fluffy_spotted_fraction = 3/7 →
    striped_spotted_fraction = 36 :=
by
    sorry

end fluffy_striped_or_spotted_cats_l110_110972


namespace train_speed_60_kmph_l110_110169

theorem train_speed_60_kmph (length_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_train = 200) 
  (h_time : time_to_cross = 12) : 
  let distance_km := (length_train : ℝ) / 1000
  let time_hr := (time_to_cross : ℝ) / 3600
  let speed_kmph := distance_km / time_hr
  speed_kmph = 60 := 
by 
  rw [h_length, h_time]
  simp [distance_km, time_hr, speed_kmph]
  norm_num
  sorry

end train_speed_60_kmph_l110_110169


namespace rectangle_ratio_l110_110939

theorem rectangle_ratio (s y x : ℝ) (hs : s > 0) (hy : y > 0) (hx : x > 0)
  (h1 : s + 2 * y = 3 * s)
  (h2 : x + y = 3 * s)
  (h3 : y = s)
  (h4 : x = 2 * s) :
  x / y = 2 := by
  sorry

end rectangle_ratio_l110_110939


namespace area_excluding_hole_l110_110011

open Polynomial

theorem area_excluding_hole (x : ℝ) : 
  ((x^2 + 7) * (x^2 + 5)) - ((2 * x^2 - 3) * (x^2 - 2)) = -x^4 + 19 * x^2 + 29 :=
by
  sorry

end area_excluding_hole_l110_110011


namespace milkshake_cost_is_five_l110_110181

def initial_amount : ℝ := 132
def hamburger_cost : ℝ := 4
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6
def amount_left : ℝ := 70

theorem milkshake_cost_is_five (M : ℝ) (h : initial_amount - (num_hamburgers * hamburger_cost + num_milkshakes * M) = amount_left) : 
  M = 5 :=
by
  sorry

end milkshake_cost_is_five_l110_110181


namespace maximize_triangle_area_l110_110384

theorem maximize_triangle_area (m : ℝ) (l : ∀ x y, x + y + m = 0) (C : ∀ x y, x^2 + y^2 + 4 * y = 0) :
  m = 0 ∨ m = 4 :=
sorry

end maximize_triangle_area_l110_110384


namespace stamps_per_page_l110_110253

def a : ℕ := 924
def b : ℕ := 1386
def c : ℕ := 1848

theorem stamps_per_page : gcd (gcd a b) c = 462 :=
sorry

end stamps_per_page_l110_110253


namespace triangle_side_length_l110_110267

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l110_110267


namespace trigonometric_identity_proof_l110_110810

theorem trigonometric_identity_proof (θ : ℝ) 
  (h : Real.tan (θ + Real.pi / 4) = -3) : 
  2 * Real.sin θ ^ 2 - Real.cos θ ^ 2 = 7 / 5 :=
sorry

end trigonometric_identity_proof_l110_110810


namespace calculate_surface_area_of_modified_cube_l110_110758

-- Definitions of the conditions
def edge_length_of_cube : ℕ := 5
def side_length_of_hole : ℕ := 2

-- The main theorem statement to be proven
theorem calculate_surface_area_of_modified_cube :
  let original_surface_area := 6 * (edge_length_of_cube * edge_length_of_cube)
  let area_removed_by_holes := 6 * (side_length_of_hole * side_length_of_hole)
  let area_exposed_by_holes := 6 * 6 * (side_length_of_hole * side_length_of_hole)
  original_surface_area - area_removed_by_holes + area_exposed_by_holes = 270 :=
by
  sorry

end calculate_surface_area_of_modified_cube_l110_110758


namespace find_c_l110_110130

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x, g_inv (g x c) = x) ↔ c = 3 / 2 := by
  sorry

end find_c_l110_110130


namespace total_votes_l110_110582

-- Define the conditions
variables (V : ℝ) (votes_second_candidate : ℝ) (percent_second_candidate : ℝ)
variables (h1 : votes_second_candidate = 240)
variables (h2 : percent_second_candidate = 0.30)

-- Statement: The total number of votes is 800 given the conditions.
theorem total_votes (h : percent_second_candidate * V = votes_second_candidate) : V = 800 :=
sorry

end total_votes_l110_110582


namespace brian_traveled_correct_distance_l110_110905

def miles_per_gallon : Nat := 20
def gallons_used : Nat := 3
def expected_miles : Nat := 60

theorem brian_traveled_correct_distance : (miles_per_gallon * gallons_used) = expected_miles := by
  sorry

end brian_traveled_correct_distance_l110_110905


namespace domain_of_composite_function_l110_110084

theorem domain_of_composite_function
    (f : ℝ → ℝ)
    (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → f (x + 1) ∈ (Set.Icc (-2:ℝ) (3:ℝ))):
    ∃ s : Set ℝ, s = Set.Icc 0 (5/2) ∧ (∀ x, x ∈ s ↔ f (2 * x - 1) ∈ Set.Icc (-1) 4) :=
by
  sorry

end domain_of_composite_function_l110_110084


namespace min_value_expression_l110_110948

theorem min_value_expression (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  ∃ x : ℝ, x = 1 ∧ x = (3 * a - 2 * b + c) / (b - a) := 
  sorry

end min_value_expression_l110_110948


namespace Kolya_mistake_l110_110651

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l110_110651


namespace calculate_expected_value_of_S_l110_110999

-- Define the problem context
variables (boys girls : ℕ)
variable (boy_girl_pair_at_start : Bool)

-- Define the expected value function
def expected_S (boys girls : ℕ) (boy_girl_pair_at_start : Bool) : ℕ :=
  if boy_girl_pair_at_start then 10 else sorry  -- we only consider the given scenario

-- The theorem to prove
theorem calculate_expected_value_of_S :
  expected_S 5 15 true = 10 :=
by
  -- proof needs to be filled in
  sorry

end calculate_expected_value_of_S_l110_110999


namespace evaluate_expression_l110_110041

theorem evaluate_expression (a b : ℕ) :
  a = 3 ^ 1006 →
  b = 7 ^ 1007 →
  (a + b)^2 - (a - b)^2 = 42 * 10^x :=
by
  intro h1 h2
  sorry

end evaluate_expression_l110_110041


namespace doubling_time_of_population_l110_110298

theorem doubling_time_of_population (birth_rate_per_1000 : ℝ) (death_rate_per_1000 : ℝ) 
  (no_emigration_immigration : Prop) (birth_rate_is_39_4 : birth_rate_per_1000 = 39.4)
  (death_rate_is_19_4 : death_rate_per_1000 = 19.4) : 
  ∃ (years : ℝ), years = 35 :=
by
  have net_growth_rate_per_1000 := birth_rate_per_1000 - death_rate_per_1000
  have net_growth_rate_percentage := (net_growth_rate_per_1000 / 1000) * 100
  have doubling_time := 70 / net_growth_rate_percentage
  use doubling_time
  rw [birth_rate_is_39_4, death_rate_is_19_4] at net_growth_rate_per_1000
  norm_num at net_growth_rate_per_1000
  norm_num at net_growth_rate_percentage
  norm_num at doubling_time
  trivial
  sorry

end doubling_time_of_population_l110_110298


namespace number_of_young_fish_l110_110612

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l110_110612


namespace percentage_of_men_35_l110_110250

theorem percentage_of_men_35 (M W : ℝ) (hm1 : M + W = 100) 
  (hm2 : 0.6 * M + 0.2923 * W = 40)
  (hw : W = 100 - M) : 
  M = 35 :=
by
  sorry

end percentage_of_men_35_l110_110250


namespace compare_trig_values_l110_110793

noncomputable def a : ℝ := Real.tan (-7 * Real.pi / 6)
noncomputable def b : ℝ := Real.cos (23 * Real.pi / 4)
noncomputable def c : ℝ := Real.sin (-33 * Real.pi / 4)

theorem compare_trig_values : c < a ∧ a < b := sorry

end compare_trig_values_l110_110793


namespace value_of_a_l110_110601

theorem value_of_a (a b c : ℂ) (h_real : a.im = 0)
  (h1 : a + b + c = 5) 
  (h2 : a * b + b * c + c * a = 7) 
  (h3 : a * b * c = 2) : a = 2 := by
  sorry

end value_of_a_l110_110601


namespace percentage_brand_A_l110_110149

theorem percentage_brand_A
  (A B : ℝ)
  (h1 : 0.6 * A + 0.65 * B = 0.5 * (A + B))
  : (A / (A + B)) * 100 = 60 :=
by
  sorry

end percentage_brand_A_l110_110149


namespace sum_of_number_and_radical_conjugate_l110_110483

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l110_110483


namespace max_number_of_rectangles_in_square_l110_110018

-- Definitions and conditions
def area_square (n : ℕ) : ℕ := 4 * n^2
def area_rectangle (n : ℕ) : ℕ := n + 1
def max_rectangles (n : ℕ) : ℕ := area_square n / area_rectangle n

-- Lean theorem statement for the proof problem
theorem max_number_of_rectangles_in_square (n : ℕ) (h : n ≥ 4) :
  max_rectangles n = 4 * (n - 1) :=
sorry

end max_number_of_rectangles_in_square_l110_110018


namespace alice_savings_l110_110022

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l110_110022


namespace ducks_among_non_falcons_l110_110406

-- Definitions based on conditions
def percentage_birds := 100
def percentage_ducks := 40
def percentage_cranes := 20
def percentage_falcons := 15
def percentage_pigeons := 25

-- Question converted into the statement
theorem ducks_among_non_falcons : 
  (percentage_ducks / (percentage_birds - percentage_falcons) * percentage_birds) = 47 :=
by
  sorry

end ducks_among_non_falcons_l110_110406


namespace model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l110_110854

theorem model_to_statue_ratio_inch_per_feet (statue_height_ft : ℝ) (model_height_in : ℝ) :
  statue_height_ft = 120 → model_height_in = 6 → (120 / 6 = 20)
:= by
  intros h1 h2
  sorry

theorem model_inches_for_statue_feet (model_per_inch_feet : ℝ) :
  model_per_inch_feet = 20 → (30 / 20 = 1.5)
:= by
  intros h
  sorry

end model_to_statue_ratio_inch_per_feet_model_inches_for_statue_feet_l110_110854


namespace coin_flip_sequences_l110_110684

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110684


namespace equal_division_of_balls_l110_110153

def total_balls : ℕ := 10
def num_boxes : ℕ := 5
def balls_per_box : ℕ := total_balls / num_boxes

theorem equal_division_of_balls :
  balls_per_box = 2 :=
by
  sorry

end equal_division_of_balls_l110_110153


namespace distinct_sequences_ten_flips_l110_110724

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110724


namespace ceil_floor_subtraction_l110_110915

theorem ceil_floor_subtraction :
  ⌈(7:ℝ) / 3⌉ + ⌊- (7:ℝ) / 3⌋ - 3 = -3 := 
by
  sorry   -- Placeholder for the proof

end ceil_floor_subtraction_l110_110915


namespace perimeter_of_ABCD_l110_110587

theorem perimeter_of_ABCD
  (AD BC AB CD : ℕ)
  (hAD : AD = 4)
  (hAB : AB = 5)
  (hBC : BC = 10)
  (hCD : CD = 7)
  (hAD_lt_BC : AD < BC) :
  AD + AB + BC + CD = 26 :=
by
  -- Proof will be provided here.
  sorry

end perimeter_of_ABCD_l110_110587


namespace alice_savings_l110_110020

-- Define the constants and conditions
def gadget_sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.1

-- State the theorem to be proved
theorem alice_savings : 
  let commission := gadget_sales * commission_rate in
  let total_earnings := basic_salary + commission in
  let savings := total_earnings * savings_rate in
  savings = 29 :=
by
  sorry

end alice_savings_l110_110020


namespace infinitesimal_alpha_as_t_to_zero_l110_110428

open Real

noncomputable def alpha (t : ℝ) : ℝ × ℝ :=
  (t, sin t)

theorem infinitesimal_alpha_as_t_to_zero : 
  ∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, abs t < δ → abs (alpha t).fst + abs (alpha t).snd < ε := by
  sorry

end infinitesimal_alpha_as_t_to_zero_l110_110428


namespace rectangle_perimeter_greater_than_16_l110_110068

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l110_110068


namespace range_of_m_l110_110071

-- Definitions from conditions
def p (m : ℝ) : Prop := (∃ x y : ℝ, 2 * x^2 / m + y^2 / (m - 1) = 1)
def q (m : ℝ) : Prop := ∃ x1 : ℝ, 8 * x1^2 - 8 * m * x1 + 7 * m - 6 = 0
def proposition (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬ (p m ∧ q m)

-- Proof statement
theorem range_of_m (m : ℝ) (h : proposition m) : (m ≤ 1 ∨ (3 / 2 < m ∧ m < 2)) :=
by
  sorry

end range_of_m_l110_110071


namespace sufficient_not_necessary_l110_110334

theorem sufficient_not_necessary (p q: Prop) :
  ¬ (p ∨ q) → ¬ p ∧ (¬ p → ¬(¬ p ∧ ¬ q)) := sorry

end sufficient_not_necessary_l110_110334


namespace no_bounded_sequences_at_least_one_gt_20_l110_110052

variable (x y z : ℕ → ℝ)
variable (x1 y1 z1 : ℝ)
variable (h0 : x1 > 0) (h1 : y1 > 0) (h2 : z1 > 0)
variable (h3 : ∀ n, x (n + 1) = y n + (1 / z n))
variable (h4 : ∀ n, y (n + 1) = z n + (1 / x n))
variable (h5 : ∀ n, z (n + 1) = x n + (1 / y n))

-- Part (a)
theorem no_bounded_sequences : (∀ n, x n > 0) ∧ (∀ n, y n > 0) ∧ (∀ n, z n > 0) → ¬ (∃ M, ∀ n, x n < M ∧ y n < M ∧ z n < M) :=
sorry

-- Part (b)
theorem at_least_one_gt_20 : x 1 = x1 ∧ y 1 = y1 ∧ z 1 = z1 → x 200 > 20 ∨ y 200 > 20 ∨ z 200 > 20 :=
sorry

end no_bounded_sequences_at_least_one_gt_20_l110_110052


namespace ivan_spent_fraction_l110_110113

theorem ivan_spent_fraction (f : ℝ) (h1 : 10 - 10 * f - 5 = 3) : f = 1 / 5 :=
by
  sorry

end ivan_spent_fraction_l110_110113


namespace range_of_x_l110_110942

-- Defining the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y ≤ f x

-- Given conditions in Lean
axiom f : ℝ → ℝ
axiom h_odd : odd_function f
axiom h_decreasing_pos : ∀ x y, 0 < x ∧ x < y → f y ≤ f x
axiom h_f4 : f 4 = 0

-- To prove the range of x for which f(x-3) ≤ 0
theorem range_of_x :
    {x : ℝ | f (x - 3) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x < 3} ∪ {x : ℝ | 7 ≤ x} :=
by
  sorry

end range_of_x_l110_110942


namespace who_made_mistake_l110_110646

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l110_110646


namespace sum_radical_conjugate_l110_110501

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l110_110501


namespace coin_flip_sequences_l110_110740

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110740


namespace option_D_correct_l110_110978

theorem option_D_correct (f : ℕ+ → ℕ) (h : ∀ k : ℕ+, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2) 
  (hf : f 4 ≥ 25) : ∀ k : ℕ+, k ≥ 4 → f k ≥ k^2 :=
by
  sorry

end option_D_correct_l110_110978


namespace total_surface_area_of_cubes_aligned_side_by_side_is_900_l110_110370

theorem total_surface_area_of_cubes_aligned_side_by_side_is_900 :
  let volumes := [27, 64, 125, 216, 512]
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  (surface_areas.sum = 900) :=
by
  sorry

end total_surface_area_of_cubes_aligned_side_by_side_is_900_l110_110370


namespace points_above_line_l110_110945

theorem points_above_line {t : ℝ} (hP : 1 + t - 1 > 0) (hQ : t^2 + (t - 1) - 1 > 0) : t > 1 :=
by
  sorry

end points_above_line_l110_110945


namespace clock_angle_3_45_smaller_l110_110318

noncomputable def angle_between_clock_hands (h m : ℕ) : ℝ :=
  let hour_angle := 30 * (h % 12) + 0.5 * m
  let minute_angle := 6 * m
  let angle := |hour_angle - minute_angle|
  min angle (360 - angle)

theorem clock_angle_3_45_smaller : 
  angle_between_clock_hands 3 45 = 157.5 :=
  by 
    sorry

end clock_angle_3_45_smaller_l110_110318


namespace last_digit_m_is_9_l110_110255

def x (n : ℕ) : ℕ := 2^(2^n) + 1

def m : ℕ := List.foldr Nat.lcm 1 (List.map x (List.range' 2 (1971 - 2 + 1)))

theorem last_digit_m_is_9 : m % 10 = 9 :=
  by
    sorry

end last_digit_m_is_9_l110_110255


namespace pancake_fundraiser_l110_110132

-- Define the constants and conditions
def cost_per_stack_of_pancakes : ℕ := 4
def cost_per_slice_of_bacon : ℕ := 2
def stacks_sold : ℕ := 60
def slices_sold : ℕ := 90
def total_raised : ℕ := 420

-- Define a theorem that states what we want to prove
theorem pancake_fundraiser : 
  (stacks_sold * cost_per_stack_of_pancakes + slices_sold * cost_per_slice_of_bacon) = total_raised :=
by
  sorry -- We place a sorry here to skip the proof, as instructed.

end pancake_fundraiser_l110_110132


namespace melanie_gave_mother_l110_110985

theorem melanie_gave_mother {initial_dimes dad_dimes final_dimes dimes_given : ℕ}
  (h₁ : initial_dimes = 7)
  (h₂ : dad_dimes = 8)
  (h₃ : final_dimes = 11)
  (h₄ : initial_dimes + dad_dimes - dimes_given = final_dimes) :
  dimes_given = 4 :=
by 
  sorry

end melanie_gave_mother_l110_110985


namespace right_triangle_area_l110_110636

theorem right_triangle_area (a b c: ℝ) (h1: c = 2) (h2: a + b + c = 2 + Real.sqrt 6) (h3: (a * b) / 2 = 1 / 2) :
  (1 / 2) * (a * b) = 1 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end right_triangle_area_l110_110636


namespace sequence_general_term_l110_110086

theorem sequence_general_term (a : ℕ → ℕ) 
  (h₀ : a 1 = 4) 
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n + n^2) : 
  ∀ n : ℕ, a n = 5 * 2^n - n^2 - 2*n - 3 :=
by
  sorry

end sequence_general_term_l110_110086


namespace only_nonneg_solution_l110_110530

theorem only_nonneg_solution :
  ∀ (x y : ℕ), 2^x = y^2 + y + 1 → (x, y) = (0, 0) := by
  intros x y h
  sorry

end only_nonneg_solution_l110_110530


namespace equation1_solution_equation2_solution_l110_110297

theorem equation1_solution (x : ℝ) : x^2 - 10*x + 16 = 0 ↔ x = 2 ∨ x = 8 :=
by sorry

theorem equation2_solution (x : ℝ) : 2*x*(x-1) = x-1 ↔ x = 1 ∨ x = 1/2 :=
by sorry

end equation1_solution_equation2_solution_l110_110297


namespace candy_problem_l110_110766

theorem candy_problem 
  (weightA costA : ℕ) (weightB costB : ℕ) (avgPrice per100 : ℕ)
  (hA : weightA = 300) (hCostA : costA = 5)
  (hCostB : costB = 7) (hAvgPrice : avgPrice = 150) (hPer100 : per100 = 100)
  (totalCost : ℕ) (hTotalCost : totalCost = costA + costB)
  (totalWeight : ℕ) (hTotalWeight : totalWeight = (totalCost * per100) / avgPrice) :
  (totalWeight = weightA + weightB) -> 
  weightB = 500 :=
by {
  sorry
}

end candy_problem_l110_110766


namespace range_of_k_l110_110088

variables (k : ℝ)

def vector_a (k : ℝ) : ℝ × ℝ := (-k, 4)
def vector_b (k : ℝ) : ℝ × ℝ := (k, k + 3)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem range_of_k (h : 0 < dot_product (vector_a k) (vector_b k)) : 
  -2 < k ∧ k < 0 ∨ 0 < k ∧ k < 6 :=
sorry

end range_of_k_l110_110088


namespace triangle_problem_l110_110260

noncomputable def find_b (a b c : ℝ) : Prop :=
  let B : ℝ := 60 * Real.pi / 180 -- converting 60 degrees to radians
  b = 2 * Real.sqrt 2

theorem triangle_problem
  (a b c : ℝ)
  (h_area : (1 / 2) * a * c * Real.sin (60 * Real.pi / 180) = Real.sqrt 3)
  (h_cosine : a^2 + c^2 = 3 * a * c) : find_b a b c :=
by
  -- The proof would go here, but we're skipping it as per the instructions.
  sorry

end triangle_problem_l110_110260


namespace alice_savings_l110_110021

-- Define the constants and conditions
def gadget_sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.1

-- State the theorem to be proved
theorem alice_savings : 
  let commission := gadget_sales * commission_rate in
  let total_earnings := basic_salary + commission in
  let savings := total_earnings * savings_rate in
  savings = 29 :=
by
  sorry

end alice_savings_l110_110021


namespace subset_M_P_N_l110_110288

def setM : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def setN : Set (ℝ × ℝ) := 
  {p | (Real.sqrt ((p.1 - 1 / 2) ^ 2 + (p.2 + 1 / 2) ^ 2) + Real.sqrt ((p.1 + 1 / 2) ^ 2 + (p.2 - 1 / 2) ^ 2)) < 2 * Real.sqrt 2}

def setP : Set (ℝ × ℝ) := 
  {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_M_P_N : setM ⊆ setP ∧ setP ⊆ setN := by
  sorry

end subset_M_P_N_l110_110288


namespace axis_of_symmetry_r_minus_2s_zero_l110_110190

/-- 
Prove that if y = x is an axis of symmetry for the curve 
y = (2 * p * x + q) / (r * x - 2 * s) with p, q, r, s nonzero, 
then r - 2s = 0. 
-/
theorem axis_of_symmetry_r_minus_2s_zero
  (p q r s : ℝ) (h_p : p ≠ 0) (h_q : q ≠ 0) (h_r : r ≠ 0) (h_s : s ≠ 0) 
  (h_sym : ∀ (a b : ℝ), (b = (2 * p * a + q) / (r * a - 2 * s)) ↔ (a = (2 * p * b + q) / (r * b - 2 * s))) :
  r - 2 * s = 0 :=
sorry

end axis_of_symmetry_r_minus_2s_zero_l110_110190


namespace theater_total_seats_l110_110004

theorem theater_total_seats
  (occupied_seats : ℕ) (empty_seats : ℕ) 
  (h1 : occupied_seats = 532) (h2 : empty_seats = 218) :
  occupied_seats + empty_seats = 750 := 
by
  -- This is the placeholder for the proof
  sorry

end theater_total_seats_l110_110004


namespace max_distance_P_to_D_l110_110035

open Real

theorem max_distance_P_to_D : 
  ∀ (P : ℝ × ℝ) (u v w : ℝ), 
    let A := (0, 0 : ℝ)
    let B := (2, 0 : ℝ)
    let C := (3, 1 : ℝ)
    let D := (1, 1 : ℝ)
    u = dist P A ∧ v = dist P B ∧ w = dist P C ∧
    u^2 + w^2 = 2 * v^2
    → dist P D ≤ 1 / sqrt 2 := by 
  sorry

end max_distance_P_to_D_l110_110035


namespace solve_system_of_equations_l110_110043

-- Definition of the system of equations as conditions
def eq1 (x y : ℤ) : Prop := 3 * x + y = 2
def eq2 (x y : ℤ) : Prop := 2 * x - 3 * y = 27

-- The theorem claiming the solution set is { (3, -7) }
theorem solve_system_of_equations :
  ∀ x y : ℤ, eq1 x y ∧ eq2 x y ↔ (x, y) = (3, -7) :=
by
  sorry

end solve_system_of_equations_l110_110043


namespace find_third_root_l110_110910

noncomputable def P (a b x : ℝ) : ℝ := a * x^3 + (a + 4 * b) * x^2 + (b - 5 * a) * x + (10 - a)

theorem find_third_root (a b : ℝ) (h1 : P a b (-1) = 0) (h2 : P a b 4 = 0) : 
 ∃ c : ℝ, c ≠ -1 ∧ c ≠ 4 ∧ P a b c = 0 ∧ c = 8 / 3 :=
 sorry

end find_third_root_l110_110910


namespace A_beats_B_by_14_meters_l110_110824

theorem A_beats_B_by_14_meters :
  let distance := 70
  let time_A := 20
  let time_B := 25
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let distance_B_in_A_time := speed_B * time_A
  (distance - distance_B_in_A_time) = 14 :=
by
  sorry

end A_beats_B_by_14_meters_l110_110824


namespace V_product_is_V_form_l110_110840

noncomputable def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3 * a * b * c

theorem V_product_is_V_form (a b c x y z : ℝ) :
  V a b c * V x y z = V (a * x + b * y + c * z) (b * x + c * y + a * z) (c * x + a * y + b * z) := by
  sorry

end V_product_is_V_form_l110_110840


namespace point_in_third_quadrant_l110_110815

theorem point_in_third_quadrant (m n : ℝ) (h1 : m > 0) (h2 : n > 0) : (-m < 0) ∧ (-n < 0) :=
by
  sorry

end point_in_third_quadrant_l110_110815


namespace lexie_crayons_count_l110_110417

variable (number_of_boxes : ℕ) (crayons_per_box : ℕ)

theorem lexie_crayons_count (h1: number_of_boxes = 10) (h2: crayons_per_box = 8) :
  (number_of_boxes * crayons_per_box) = 80 := by
  sorry

end lexie_crayons_count_l110_110417


namespace remainder_of_3_pow_19_mod_10_l110_110666

-- Definition of the problem and conditions
def q := 3^19

-- Statement to prove
theorem remainder_of_3_pow_19_mod_10 : q % 10 = 7 :=
by
  sorry

end remainder_of_3_pow_19_mod_10_l110_110666


namespace frustum_volume_l110_110436

theorem frustum_volume (m : ℝ) (α : ℝ) (k : ℝ) : 
  m = 3/π ∧ 
  α = 43 + 40/60 + 42.2/3600 ∧ 
  k = 1 →
  frustumVolume = 0.79 := 
sorry

end frustum_volume_l110_110436


namespace evaluation_expression_l110_110038

theorem evaluation_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 :=
by
  rw [h1, h2]
  -- Here we would perform the arithmetic steps to show the equality
  sorry

end evaluation_expression_l110_110038


namespace coin_flip_sequences_l110_110688

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110688


namespace jenny_jellybeans_original_l110_110526

theorem jenny_jellybeans_original (x : ℝ) 
  (h : 0.75^3 * x = 45) : x = 107 := 
sorry

end jenny_jellybeans_original_l110_110526


namespace find_b_l110_110269

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end find_b_l110_110269


namespace sum_of_number_and_conjugate_l110_110493

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l110_110493


namespace domain_of_sqrt_ln_eq_l110_110852

noncomputable def domain_of_function : Set ℝ :=
  {x | 2 * x + 1 >= 0 ∧ 3 - 4 * x > 0}

theorem domain_of_sqrt_ln_eq :
  domain_of_function = Set.Icc (-1 / 2) (3 / 4) \ {3 / 4} :=
by
  sorry

end domain_of_sqrt_ln_eq_l110_110852


namespace divisible_by_other_l110_110163

theorem divisible_by_other (y : ℕ) 
  (h1 : y = 20)
  (h2 : y % 4 = 0)
  (h3 : y % 8 ≠ 0) : (∃ n, n ≠ 4 ∧ y % n = 0 ∧ n = 5) :=
by 
  sorry

end divisible_by_other_l110_110163


namespace cubic_sum_l110_110998

theorem cubic_sum (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + x * z + y * z = -5) (h3 : x * y * z = -6) :
  x^3 + y^3 + z^3 = 18 :=
by
  sorry

end cubic_sum_l110_110998


namespace peyton_total_yards_l110_110990

def distance_on_Saturday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def distance_on_Sunday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def total_distance (distance_Saturday: Nat) (distance_Sunday: Nat) : Nat :=
  distance_Saturday + distance_Sunday

theorem peyton_total_yards :
  let throws_Saturday := 20
  let yards_per_throw_Saturday := 20
  let throws_Sunday := 30
  let yards_per_throw_Sunday := 40
  distance_on_Saturday throws_Saturday yards_per_throw_Saturday +
  distance_on_Sunday throws_Sunday yards_per_throw_Sunday = 1600 :=
by
  sorry

end peyton_total_yards_l110_110990


namespace peyton_manning_total_yards_l110_110993

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end peyton_manning_total_yards_l110_110993


namespace units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l110_110841

def k : ℕ := 2012 ^ 2 + 2 ^ 2012

theorem units_digit_k_cube_plus_2_to_k_plus_1_mod_10 : (k ^ 3 + 2 ^ (k + 1)) % 10 = 2 := 
by sorry

end units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l110_110841


namespace investment_amount_l110_110342

theorem investment_amount (A_investment B_investment total_profit A_share : ℝ)
  (hA_investment : A_investment = 100)
  (hB_investment_months : B_investment > 0)
  (h_total_profit : total_profit = 100)
  (h_A_share : A_share = 50)
  (h_conditions : A_share / total_profit = (A_investment * 12) / ((A_investment * 12) + (B_investment * 6))) :
  B_investment = 200 :=
by {
  sorry
}

end investment_amount_l110_110342


namespace number_of_students_joined_l110_110881

theorem number_of_students_joined
  (A : ℝ)
  (x : ℕ)
  (h1 : A = 50)
  (h2 : (100 + x) * (A - 10) = 5400) 
  (h3 : 100 * A + 400 = 5400) :
  x = 35 := 
by 
  -- all conditions in a) are used as definitions in Lean 4 statement
  sorry

end number_of_students_joined_l110_110881


namespace triangle_division_point_distances_l110_110860

theorem triangle_division_point_distances 
  {a b c : ℝ} 
  (h1 : a = 13) 
  (h2 : b = 17) 
  (h3 : c = 24)
  (h4 : ∃ p q : ℝ, p = 9 ∧ q = 11) : 
  ∃ p q : ℝ, p = 9 ∧ q = 11 :=
  sorry

end triangle_division_point_distances_l110_110860


namespace quadratic_vertex_form_l110_110244

theorem quadratic_vertex_form (a h k x: ℝ) (h_a : a = 3) (hx : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by {
  sorry
}

end quadratic_vertex_form_l110_110244


namespace Ariella_has_more_savings_l110_110761

variable (Daniella_savings: ℝ) (Ariella_future_savings: ℝ) (interest_rate: ℝ) (time_years: ℝ)
variable (initial_Ariella_savings: ℝ)

-- Conditions
axiom h1 : Daniella_savings = 400
axiom h2 : Ariella_future_savings = 720
axiom h3 : interest_rate = 0.10
axiom h4 : time_years = 2

-- Assume simple interest formula for future savings
axiom simple_interest : Ariella_future_savings = initial_Ariella_savings * (1 + interest_rate * time_years)

-- Show the difference in savings
theorem Ariella_has_more_savings : initial_Ariella_savings - Daniella_savings = 200 :=
by sorry

end Ariella_has_more_savings_l110_110761


namespace exist_midpoints_l110_110152
open Classical

noncomputable def h (a b c : ℝ) := (a + b + c) / 3

theorem exist_midpoints (a b c : ℝ) (X Y Z : ℝ) (AX BY CZ : ℝ) :
  (0 < X) ∧ (X < a) ∧
  (0 < Y) ∧ (Y < b) ∧
  (0 < Z) ∧ (Z < c) ∧
  (X + (a - X) = (h a b c)) ∧
  (Y + (b - Y) = (h a b c)) ∧
  (Z + (c - Z) = (h a b c)) ∧
  (AX * BY * CZ = (a - X) * (b - Y) * (c - Z))
  → ∃ (X Y Z : ℝ), X = (a / 2) ∧ Y = (b / 2) ∧ Z = (c / 2) :=
by
  sorry

end exist_midpoints_l110_110152


namespace compound_interest_principal_l110_110663

theorem compound_interest_principal 
    (CI : Real)
    (r : Real)
    (n : Nat)
    (t : Nat)
    (A : Real)
    (P : Real) :
  CI = 945.0000000000009 →
  r = 0.10 →
  n = 1 →
  t = 2 →
  A = P * (1 + r / n) ^ (n * t) →
  CI = A - P →
  P = 4500.0000000000045 :=
by intros
   sorry

end compound_interest_principal_l110_110663


namespace distinct_sequences_ten_flips_l110_110714

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110714


namespace Kolya_made_the_mistake_l110_110654

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l110_110654


namespace radical_conjugate_sum_l110_110504

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l110_110504


namespace B_more_cost_effective_l110_110681

variable (x y : ℝ)
variable (hx : x ≠ y)

theorem B_more_cost_effective (x y : ℝ) (hx : x ≠ y) :
  (1/2 * x + 1/2 * y) > (2 * x * y / (x + y)) :=
by
  sorry

end B_more_cost_effective_l110_110681


namespace hyperbola_eccentricity_is_5_over_3_l110_110752

noncomputable def hyperbola_asymptote_condition (a b : ℝ) : Prop :=
  a / b = 3 / 4

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_5_over_3 (a b : ℝ) (h : hyperbola_asymptote_condition a b) :
  hyperbola_eccentricity a b = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_is_5_over_3_l110_110752


namespace smallest_square_area_l110_110337

theorem smallest_square_area (a b c d : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) (h4 : d = 5) :
  ∃ s : ℕ, s * s = 64 ∧ (a + c <= s ∧ max b d <= s) ∨ (max a c <= s ∧ b + d <= s) :=
sorry

end smallest_square_area_l110_110337


namespace find_a_l110_110304

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

-- Define the derivative of f
def f_prime (x a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_a (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a = 4 :=
by
  sorry

end find_a_l110_110304


namespace sqrt_condition_l110_110965

theorem sqrt_condition (x : ℝ) : (x - 3 ≥ 0) ↔ (x = 3) :=
by sorry

end sqrt_condition_l110_110965


namespace solve_arithmetic_sequence_l110_110849

theorem solve_arithmetic_sequence (y : ℝ) (h : 0 < y) (h_arith : ∃ (d : ℝ), 4 + d = y^2 ∧ y^2 + d = 16 ∧ 16 + d = 36) :
  y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l110_110849


namespace find_y_plus_4_div_y_l110_110800

theorem find_y_plus_4_div_y (y : ℝ) (h : y^3 + 4 / y^3 = 110) : y + 4 / y = 6 := sorry

end find_y_plus_4_div_y_l110_110800


namespace max_area_triangle_m_l110_110385

theorem max_area_triangle_m (m: ℝ) :
  ∃ (M N: ℝ × ℝ), (x + y + m = 0) ∧ (x^2 + y^2 + 4*y = 0) ∧ 
    (∀ {m₁ m₂ : ℝ}, (area_triangle_CM_N m₁ ≤ area_triangle_CM_N m ∧
      area_triangle_CM_N m₂ ≤ area_triangle_CM_N m) → 
      (m = 0 ∨ m = 4)) :=
begin
  -- Insert proof here
  sorry,
end

end max_area_triangle_m_l110_110385


namespace eval_f_at_two_eval_f_at_neg_two_l110_110382

def f (x : ℝ) : ℝ := 2 * x ^ 2 + 3 * x

theorem eval_f_at_two : f 2 = 14 :=
by
  sorry

theorem eval_f_at_neg_two : f (-2) = 2 :=
by
  sorry

end eval_f_at_two_eval_f_at_neg_two_l110_110382


namespace similar_triangles_XY_length_l110_110143

-- Defining necessary variables.
variables (PQ QR YZ XY : ℝ) (area_XYZ : ℝ)

-- Given conditions to be used in the proof.
def condition1 : PQ = 8 := sorry
def condition2 : QR = 16 := sorry
def condition3 : YZ = 24 := sorry
def condition4 : area_XYZ = 144 := sorry

-- Statement of the mathematical proof problem to show XY = 12
theorem similar_triangles_XY_length :
  PQ = 8 → QR = 16 → YZ = 24 → area_XYZ = 144 → XY = 12 :=
by
  intros hPQ hQR hYZ hArea
  sorry

end similar_triangles_XY_length_l110_110143


namespace range_of_m_l110_110802

namespace TrigonometricProof

theorem range_of_m (m : ℝ) (h : m > 0) :
  (∃ x₁ x₂ ∈ set.Icc (0 : ℝ) (Real.pi / 4),
    (sin (2 * x₁) + 2 * Real.sqrt 3 * cos x₁ ^ 2 - Real.sqrt 3) =
    (m * cos (2 * x₂ - Real.pi / 6) - 2 * m + 3)) →
    m ∈ set.Icc (2 / 3) 2 :=
by
  sorry

end TrigonometricProof

end range_of_m_l110_110802


namespace cubic_inequality_l110_110454

theorem cubic_inequality (a b : ℝ) : a > b → a^3 > b^3 :=
sorry

end cubic_inequality_l110_110454


namespace prob_relations_l110_110103

-- Define the probabilities for each method
def P1 : ℚ := 1 / 3
def P2 : ℚ := 1 / 2
def P3 : ℚ := 2 / 3

-- Prove the relations between the probabilities
theorem prob_relations : (P1 < P2) ∧ (P1 < P3) ∧ (2 * P1 = P3) :=
by {
  -- Individual proofs for each part (not needed, just 'sorry' them) 
  have h1 : P1 < P2, sorry,
  have h2 : P1 < P3, sorry,
  have h3 : 2 * P1 = P3, sorry,
  exact ⟨h1, h2, h3⟩
}

end prob_relations_l110_110103


namespace triangle_is_equilateral_l110_110056

theorem triangle_is_equilateral (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + ac + bc) : a = b ∧ b = c :=
by
  sorry

end triangle_is_equilateral_l110_110056


namespace product_decrease_l110_110845

variable (a b : ℤ)

theorem product_decrease : (a - 3) * (b + 3) - a * b = 900 → a - b = 303 → a * b - (a + 3) * (b - 3) = 918 :=
by
    intros h1 h2
    sorry

end product_decrease_l110_110845


namespace correct_divisor_l110_110823

theorem correct_divisor (X : ℕ) (D : ℕ) (H1 : X = 24 * 87) (H2 : X / D = 58) : D = 36 :=
by
  sorry

end correct_divisor_l110_110823


namespace find_y_from_exponent_equation_l110_110808

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110808


namespace ratio_of_earnings_l110_110848

theorem ratio_of_earnings (K V S : ℕ) (h1 : K + 30 = V) (h2 : V = 84) (h3 : S = 216) : S / K = 4 :=
by
  -- proof goes here
  sorry

end ratio_of_earnings_l110_110848


namespace solve_equation_l110_110296

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x = -4/3) ↔ (x^2 + 2 * x + 2) / (x + 2) = x + 3 :=
by
  sorry

end solve_equation_l110_110296


namespace coin_flip_sequences_l110_110705

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110705


namespace index_card_area_l110_110899

theorem index_card_area (a b : ℕ) (new_area : ℕ) (reduce_length reduce_width : ℕ)
  (original_length : a = 3) (original_width : b = 7)
  (reduced_area_condition : a * (b - reduce_width) = new_area)
  (reduce_width_2 : reduce_width = 2) 
  (new_area_correct : new_area = 15) :
  (a - reduce_length) * b = 7 := by
  sorry

end index_card_area_l110_110899


namespace least_integer_l110_110933

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l110_110933


namespace find_n_l110_110982

-- Definitions of the problem conditions
def sum_coefficients (n : ℕ) : ℕ := 4^n
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- The main theorem to be proved
theorem find_n (n : ℕ) (P S : ℕ) (hP : P = sum_coefficients n) (hS : S = sum_binomial_coefficients n) (h : P + S = 272) : n = 4 :=
by
  sorry

end find_n_l110_110982


namespace license_plate_increase_l110_110287

theorem license_plate_increase :
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  new_plates / old_plates = (900 / 17576) * 100 :=
by
  let old_plates := 26^3 * 10^3
  let new_plates := 30^2 * 10^5
  have h : new_plates / old_plates = (900 / 17576) * 100 := sorry
  exact h

end license_plate_increase_l110_110287


namespace kids_outside_l110_110525

theorem kids_outside (s t n c : ℕ)
  (h1 : s = 644997)
  (h2 : t = 893835)
  (h3 : n = 1538832)
  (h4 : (n - s) = t) : c = 0 :=
by {
  sorry
}

end kids_outside_l110_110525


namespace calculate_total_bricks_l110_110391

-- Given definitions based on the problem.
variables (a d g h : ℕ)

-- Definitions for the questions in terms of variables.
def days_to_build_bricks (a d g : ℕ) : ℕ :=
  (a * g) / d

def total_bricks_with_additional_men (a d g h : ℕ) : ℕ :=
  a + ((d + h) * a) / 2

theorem calculate_total_bricks (a d g h : ℕ)
  (h1 : 0 < d)
  (h2 : 0 < g)
  (h3 : 0 < a) :
  days_to_build_bricks a d g = a * g / d ∧
  total_bricks_with_additional_men a d g h = (3 * a + h * a) / 2 :=
  by sorry

end calculate_total_bricks_l110_110391


namespace frank_completes_book_in_three_days_l110_110199

-- Define the total number of pages in a book
def total_pages : ℕ := 249

-- Define the number of pages Frank reads per day
def pages_per_day : ℕ := 83

-- Define the number of days Frank needs to finish a book
def days_to_finish_book (total_pages pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

-- Theorem statement to prove that Frank finishes a book in 3 days
theorem frank_completes_book_in_three_days : days_to_finish_book total_pages pages_per_day = 3 := 
by {
  -- Proof goes here
  sorry
}

end frank_completes_book_in_three_days_l110_110199


namespace milk_required_for_flour_l110_110983

theorem milk_required_for_flour (flour_ratio milk_ratio total_flour : ℕ) : 
  (milk_ratio * (total_flour / flour_ratio)) = 160 :=
by
  let milk_ratio := 40
  let flour_ratio := 200
  let total_flour := 800
  exact sorry

end milk_required_for_flour_l110_110983


namespace triangle_area_example_l110_110314

def point := (ℝ × ℝ)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_example :
  let A : point := (3, -2)
  let B : point := (12, 5)
  let C : point := (3, 8)
  triangle_area A B C = 45 :=
by
  sorry

end triangle_area_example_l110_110314


namespace max_expression_value_l110_110450

theorem max_expression_value (a b c d e f g h k : ℤ)
  (ha : (a = 1 ∨ a = -1)) (hb : (b = 1 ∨ b = -1))
  (hc : (c = 1 ∨ c = -1)) (hd : (d = 1 ∨ d = -1))
  (he : (e = 1 ∨ e = -1)) (hf : (f = 1 ∨ f = -1))
  (hg : (g = 1 ∨ g = -1)) (hh : (h = 1 ∨ h = -1))
  (hk : (k = 1 ∨ k = -1)) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 := sorry

end max_expression_value_l110_110450


namespace maximum_cookies_andy_could_have_eaten_l110_110866

theorem maximum_cookies_andy_could_have_eaten :
  ∃ x : ℤ, (x ≥ 0 ∧ 2 * x + (x - 3) + x = 30) ∧ (∀ y : ℤ, 0 ≤ y ∧ 2 * y + (y - 3) + y = 30 → y ≤ 8) :=
by {
  sorry
}

end maximum_cookies_andy_could_have_eaten_l110_110866


namespace find_larger_page_l110_110819

theorem find_larger_page {x y : ℕ} (h1 : y = x + 1) (h2 : x + y = 125) : y = 63 :=
by
  sorry

end find_larger_page_l110_110819


namespace trajectory_of_circle_center_l110_110964

theorem trajectory_of_circle_center :
  ∀ (M : ℝ × ℝ), (∃ r : ℝ, (M.1 + r = 1 ∧ M.1 - r = -1) ∧ (M.1 - 1)^2 + (M.2 - 0)^2 = r^2) → M.2^2 = 4 * M.1 :=
by
  intros M h
  sorry

end trajectory_of_circle_center_l110_110964


namespace movie_ticket_vs_popcorn_difference_l110_110468

variable (P : ℝ) -- cost of a bucket of popcorn
variable (d : ℝ) -- cost of a drink
variable (c : ℝ) -- cost of a candy
variable (t : ℝ) -- cost of a movie ticket

-- Given conditions
axiom h1 : t = 8
axiom h2 : d = P + 1
axiom h3 : c = (P + 1) / 2
axiom h4 : t + P + d + c = 22

-- Question rewritten: Prove that the difference between the normal cost of a movie ticket and the cost of a bucket of popcorn is 3.
theorem movie_ticket_vs_popcorn_difference : t - P = 3 :=
by
  sorry

end movie_ticket_vs_popcorn_difference_l110_110468


namespace who_made_a_mistake_l110_110644

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l110_110644


namespace quadratic_extreme_values_l110_110797

theorem quadratic_extreme_values (y1 y2 y3 y4 : ℝ) 
  (h1 : y2 < y3) 
  (h2 : y3 = y4) 
  (h3 : ∀ x, ∃ (a b c : ℝ), ∀ y, y = a * x * x + b * x + c) :
  (y1 < y2) ∧ (y2 < y3) :=
by
  sorry

end quadratic_extreme_values_l110_110797


namespace distinct_sequences_ten_flips_l110_110721

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110721


namespace value_at_points_zero_l110_110478

def odd_function (v : ℝ → ℝ) := ∀ x : ℝ, v (-x) = -v x

theorem value_at_points_zero (v : ℝ → ℝ)
  (hv : odd_function v) :
  v (-2.1) + v (-1.2) + v (1.2) + v (2.1) = 0 :=
by {
  sorry
}

end value_at_points_zero_l110_110478


namespace bug_total_distance_l110_110887

theorem bug_total_distance :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let final_pos := 0
  let distance1 := |pos1 - pos2|
  let distance2 := |pos2 - pos3|
  let distance3 := |pos3 - final_pos|
  let total_distance := distance1 + distance2 + distance3
  total_distance = 29 := by
    sorry

end bug_total_distance_l110_110887


namespace triangle_problem_l110_110261

noncomputable def find_b (a b c : ℝ) : Prop :=
  let B : ℝ := 60 * Real.pi / 180 -- converting 60 degrees to radians
  b = 2 * Real.sqrt 2

theorem triangle_problem
  (a b c : ℝ)
  (h_area : (1 / 2) * a * c * Real.sin (60 * Real.pi / 180) = Real.sqrt 3)
  (h_cosine : a^2 + c^2 = 3 * a * c) : find_b a b c :=
by
  -- The proof would go here, but we're skipping it as per the instructions.
  sorry

end triangle_problem_l110_110261


namespace older_sister_age_l110_110658

theorem older_sister_age (x : ℕ) (older_sister_age : ℕ) (h1 : older_sister_age = 3 * x)
  (h2 : older_sister_age + 2 = 2 * (x + 2)) : older_sister_age = 6 :=
by
  sorry

end older_sister_age_l110_110658


namespace find_number_l110_110760

theorem find_number (x : ℕ) : ((52 + x) * 3 - 60) / 8 = 15 → x = 8 :=
by
  sorry

end find_number_l110_110760


namespace combination_permutation_value_l110_110228

theorem combination_permutation_value (n : ℕ) (h : (n * (n - 1)) = 42) : (Nat.factorial n) / (Nat.factorial 3 * Nat.factorial (n - 3)) = 35 := 
by
  sorry

end combination_permutation_value_l110_110228


namespace polynomial_evaluation_l110_110559

theorem polynomial_evaluation :
  (5 * 3^3 - 3 * 3^2 + 7 * 3 - 2 = 127) :=
by
  sorry

end polynomial_evaluation_l110_110559


namespace maximum_n_l110_110602

variable (x y z : ℝ)

theorem maximum_n (h1 : x + y + z = 12) (h2 : x * y + y * z + z * x = 30) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
by
  sorry

end maximum_n_l110_110602


namespace december_sales_multiple_l110_110829

   noncomputable def find_sales_multiple (A : ℝ) (x : ℝ) :=
     x * A = 0.3888888888888889 * (11 * A + x * A)

   theorem december_sales_multiple (A : ℝ) (x : ℝ) (h : find_sales_multiple A x) : x = 7 :=
   by 
     sorry
   
end december_sales_multiple_l110_110829


namespace alicia_masks_left_l110_110439

theorem alicia_masks_left (T G L : ℕ) (hT : T = 90) (hG : G = 51) (hL : L = T - G) : L = 39 :=
by
  rw [hT, hG] at hL
  exact hL

end alicia_masks_left_l110_110439


namespace sum_of_radical_conjugates_l110_110515

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l110_110515


namespace one_and_one_third_of_what_number_is_45_l110_110425

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l110_110425


namespace find_number_l110_110014

-- Define the condition
def exceeds_by_30 (x : ℝ) : Prop :=
  x = (3/8) * x + 30

-- Prove the main statement
theorem find_number : ∃ x : ℝ, exceeds_by_30 x ∧ x = 48 := by
  sorry

end find_number_l110_110014


namespace sum_radical_conjugate_l110_110500

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l110_110500


namespace magazine_cost_l110_110302

variable (b m : ℝ)

theorem magazine_cost (h1 : 2 * b + 2 * m = 26) (h2 : b + 3 * m = 27) : m = 7 :=
by
  sorry

end magazine_cost_l110_110302


namespace variance_uniform_l110_110369

noncomputable def variance_of_uniform (α β : ℝ) (h : α < β) : ℝ :=
  let E := (α + β) / 2
  (β - α)^2 / 12

theorem variance_uniform (α β : ℝ) (h : α < β) :
  variance_of_uniform α β h = (β - α)^2 / 12 :=
by
  -- statement of proof only, actual proof here is sorry
  sorry

end variance_uniform_l110_110369


namespace factor_expression_l110_110773

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110773


namespace sum_of_number_and_its_radical_conjugate_l110_110510

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l110_110510


namespace abs_inequalities_imply_linear_relationship_l110_110980

theorem abs_inequalities_imply_linear_relationship (a b c : ℝ)
(h1 : |a - b| ≥ |c|)
(h2 : |b - c| ≥ |a|)
(h3 : |c - a| ≥ |b|) :
a = b + c ∨ b = c + a ∨ c = a + b :=
sorry

end abs_inequalities_imply_linear_relationship_l110_110980


namespace sum_of_number_and_its_radical_conjugate_l110_110511

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l110_110511


namespace algebra_expression_value_l110_110801

theorem algebra_expression_value (m : ℝ) (h : m^2 - 3 * m - 1 = 0) : 2 * m^2 - 6 * m + 5 = 7 := by
  sorry

end algebra_expression_value_l110_110801


namespace clothes_prices_l110_110329

theorem clothes_prices (total_cost : ℕ) (shirt_more : ℕ) (trousers_price : ℕ) (shirt_price : ℕ)
  (h1 : total_cost = 185)
  (h2 : shirt_more = 5)
  (h3 : shirt_price = 2 * trousers_price + shirt_more)
  (h4 : total_cost = shirt_price + trousers_price) : 
  trousers_price = 60 ∧ shirt_price = 125 :=
  by sorry

end clothes_prices_l110_110329


namespace D_96_equals_112_l110_110410

def multiplicative_decompositions (n : ℕ) : ℕ :=
  sorry -- Define how to find the number of multiplicative decompositions

theorem D_96_equals_112 : multiplicative_decompositions 96 = 112 :=
  sorry

end D_96_equals_112_l110_110410


namespace perimeter_gt_sixteen_l110_110062

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l110_110062


namespace coin_flip_sequences_l110_110707

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110707


namespace sum_of_number_and_conjugate_l110_110495

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l110_110495


namespace tank_capacity_l110_110749

theorem tank_capacity (w c : ℝ) (h1 : w / c = 1 / 6) (h2 : (w + 5) / c = 1 / 3) : c = 30 :=
by
  sorry

end tank_capacity_l110_110749


namespace brother_birth_year_1990_l110_110115

variable (current_year : ℕ) -- Assuming the current year is implicit for the problem, it should be 2010 if Karina is 40 years old.
variable (karina_birth_year : ℕ)
variable (karina_current_age : ℕ)
variable (brother_current_age : ℕ)
variable (karina_twice_of_brother : Prop)

def karinas_brother_birth_year (karina_birth_year karina_current_age brother_current_age : ℕ) : ℕ :=
  karina_birth_year + brother_current_age

theorem brother_birth_year_1990 
  (h1 : karina_birth_year = 1970) 
  (h2 : karina_current_age = 40) 
  (h3 : karina_twice_of_brother) : 
  karinas_brother_birth_year 1970 40 20 = 1990 := 
by
  sorry

end brother_birth_year_1990_l110_110115


namespace number_of_perfect_numbers_l110_110352

-- Define the concept of a perfect number
def perfect_number (a b : ℕ) : ℕ := (a + b)^2

-- Define the proposition we want to prove
theorem number_of_perfect_numbers : ∃ n : ℕ, n = 15 ∧ 
  ∀ p, ∃ a b : ℕ, p = perfect_number a b ∧ p < 200 :=
sorry

end number_of_perfect_numbers_l110_110352


namespace find_x1_l110_110079

theorem find_x1 (x1 x2 x3 x4 : ℝ) (h1 : 0 ≤ x4) (h2 : x4 ≤ x3) (h3 : x3 ≤ x2) (h4 : x2 ≤ x1) (h5 : x1 ≤ 1)
  (h6 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 5) : x1 = 4 / 5 := 
  sorry

end find_x1_l110_110079


namespace general_term_arithmetic_seq_sum_seq_b_l110_110951

variables {a_n : ℕ → ℝ} {b_n S_n : ℕ → ℝ}

-- Define the arithmetic sequence with given conditions
def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- State the problem of finding the arithmetic sequence
theorem general_term_arithmetic_seq (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : is_arithmetic_seq a) 
  (h3 : a 2 = (a 1 + a 3) / 2 ∧ a 3 = (a 2 + (a 4 + 1)) / 2) : 
  ∀ n, a n = 2 * n := 
sorry

-- Define the sequence {b_n}
def seq_b (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, b n = 2 / ((n + 3) * (a n + 2))

-- State the problem of summing the sequence {b_n}
theorem sum_seq_b (b : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, b n = 2 / ((n + 3) * (2 * n + 2)))
  (h2 : S n = Σ k in range n, b k) :
  S n = 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := 
sorry

end general_term_arithmetic_seq_sum_seq_b_l110_110951


namespace range_of_a_l110_110221

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - 3 * x + 2 = 0) → ∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) → (a = 0 ∨ a ≥ 9 / 8) :=
by
  sorry

end range_of_a_l110_110221


namespace force_game_end_no_definitive_winning_strategy_l110_110473

-- Define the conditions
variable (A_n B_n : ℕ → EuclideanSpace ℝ (Fin 2))
variable (A_1 : EuclideanSpace ℝ (Fin 2))
variable (game_length : ℝ := 1)

-- Definitions for circles and perpendicular bisectors as required in the problem
def circle (center : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p | (dist center p) = radius}

def perpendicular_bisector (p1 p2 : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {q | ∃ l, q = (p1 + p2) / 2 + l • ⟦p2 - p1⟧}

-- Assume the rules of the game as conditions in the Lean 4 environment
axiom no_coincide (n : ℕ) (h : B_n (n+1) ≠ A_n (n+1)) : B_n (n+1) ≠ B_n n
axiom no_overlap (n m : ℕ) (h1 : n ≠ m)
  (h2 : ∀ t, t ∈ Icc (0:ℝ) 1 → A_n t ≠ A_m t ∨ B_n t ≠ B_m t) : true

-- Main proof statement
theorem force_game_end : (∃ (strategy : ℕ → EuclideanSpace ℝ (Fin 2) → Set (EuclideanSpace ℝ (Fin 2))), 
  (∀ n, strategy n B_n = circle A_n game_length ∩ perpendicular_bisector (A_n n) (B_n n)) ∧ 
  (∀ n, strategy n A_n = circle B_n game_length ∩ perpendicular_bisector (B_n n) (A_n n)))
→ (∀ n, ∃ (x : EuclideanSpace ℝ (Fin 2)), x ∈ circle (A_n n) game_length ∩ circle (B_n n) game_length)
→ (∃ N, ∃ x ∈ (circle (A_1) game_length), x = A_n N ∨ x = B_n N) :=
by sorry

noncomputable def no_winning_strategy (A B : Set (EuclideanSpace ℝ (Fin 2))) :=
  ¬(∀ x ∈ A, ∃ y ∈ B, dist x y < 1) ∧ ¬(∀ y ∈ B, ∃ x ∈ A, dist y x < 1)

-- The second part stating that there is no winning strategy
theorem no_definitive_winning_strategy : no_winning_strategy (circle A_1 game_length) (circle A_1 game_length) :=
by sorry

end force_game_end_no_definitive_winning_strategy_l110_110473


namespace train_speed_l110_110171

/-- Given that a train crosses a pole in 12 seconds and the length of the train is 200 meters,
prove that the speed of the train in km/hr is 60. -/
theorem train_speed 
  (cross_time : ℝ) (train_length : ℝ)
  (H1 : cross_time = 12) (H2 : train_length = 200) : 
  let distance_km := train_length / 1000
      time_hr := cross_time / 3600
      speed := distance_km / time_hr in
  speed = 60 :=
by
  sorry

end train_speed_l110_110171


namespace projection_of_a_onto_b_l110_110202

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-2, 4)

theorem projection_of_a_onto_b :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b_squared := vector_b.1 ^ 2 + vector_b.2 ^ 2
  let scalar_projection := dot_product / magnitude_b_squared
  let proj_vector := (scalar_projection * vector_b.1, scalar_projection * vector_b.2)
  proj_vector = (-4/5, 8/5) :=
by
  sorry

end projection_of_a_onto_b_l110_110202


namespace derek_age_calculation_l110_110312

theorem derek_age_calculation 
  (bob_age : ℕ)
  (evan_age : ℕ)
  (derek_age : ℕ) 
  (h1 : bob_age = 60)
  (h2 : evan_age = (2 * bob_age) / 3)
  (h3 : derek_age = evan_age - 10) : 
  derek_age = 30 :=
by
  -- The proof is to be filled in
  sorry

end derek_age_calculation_l110_110312


namespace max_grain_mass_l110_110464

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end max_grain_mass_l110_110464


namespace harmonic_arithmetic_sequence_common_difference_l110_110344

theorem harmonic_arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) : 
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * d)) →
  (∀ n, a n = a 1 + (n - 1) * d) →
  (a 1 = 1) →
  (d ≠ 0) →
  (∃ k, ∀ n, S n / S (2 * n) = k) →
  d = 2 :=
by
  sorry

end harmonic_arithmetic_sequence_common_difference_l110_110344


namespace part_a_l110_110180

structure is_interesting_equation (P Q : Polynomial ℤ) : Prop :=
  (degree_pos : P.degree ≥ 1 ∧ Q.degree ≥ 1)
  (inf_solutions : ∃∞ n : ℕ, P.eval n = Q.eval n)

structure yields_in (P Q F G : Polynomial ℤ) : Prop :=
  (exists_polynomial : ∃ R : Polynomial ℚ, ∀ x, F.eval x = R.eval (P.eval x) ∧ G.eval x = R.eval (Q.eval x))

def infinite_subset (S : set (ℕ × ℕ)) : Prop := 
  set.infinite S

theorem part_a {S : set (ℕ × ℕ)} (hS : infinite_subset S) :
  ∃ (P₀ Q₀ : Polynomial ℤ), 
    is_interesting_equation P₀ Q₀ ∧ 
    (∀ {P Q : Polynomial ℤ}, (is_interesting_equation P Q) → (∀ (x y : ℕ × ℕ), (x, y) ∈ S → P.eval x = Q.eval y) → yields_in P₀ Q₀ P Q) := sorry

end part_a_l110_110180


namespace probability_at_least_one_l110_110142

variable (Ω : Type)
variable [ProbabilitySpace Ω]

-- Independent events representing the successful decryption by A, B, and C
variable (A B C : Event Ω)
variable (PA : ℝ) (PB : ℝ) (PC : ℝ)

-- Given conditions
axiom independent_events : IndependentEvents [A, B, C]
axiom prob_A : ℙ[A] = 1 / 2
axiom prob_B : ℙ[B] = 1 / 3
axiom prob_C : ℙ[C] = 1 / 4

-- Define the event that at least one person decrypts the code
def at_least_one_decrypts : Event Ω := A ∪ B ∪ C

-- Statement to prove
theorem probability_at_least_one : ℙ[at_least_one_decrypts Ω A B C] = 3 / 4 := by
  sorry

end probability_at_least_one_l110_110142


namespace sum_radical_conjugate_l110_110499

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l110_110499


namespace fraction_equality_l110_110792

theorem fraction_equality (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 := 
sorry

end fraction_equality_l110_110792


namespace heaviest_lightest_difference_total_excess_weight_total_selling_price_l110_110338

-- Define deviations from standard weight and their counts
def deviations : List (ℚ × ℕ) := [(-3.5, 2), (-2, 4), (-1.5, 2), (0, 1), (1, 3), (2.5, 8)]

-- Define standard weight and price per kg
def standard_weight : ℚ := 18
def price_per_kg : ℚ := 1.8

-- Prove the three statements:
theorem heaviest_lightest_difference :
  (2.5 - (-3.5)) = 6 := by
  sorry

theorem total_excess_weight :
  (2 * -3.5 + 4 * -2 + 2 * -1.5 + 1 * 0 + 3 * 1 + 8 * 2.5) = 5 := by
  sorry

theorem total_selling_price :
  (standard_weight * 20 + 5) * price_per_kg = 657 := by
  sorry

end heaviest_lightest_difference_total_excess_weight_total_selling_price_l110_110338


namespace groom_dog_time_l110_110828

theorem groom_dog_time :
  ∃ (D : ℝ), (5 * D + 3 * 0.5 = 14) ∧ (D = 2.5) :=
by
  sorry

end groom_dog_time_l110_110828


namespace movie_final_length_l110_110005

theorem movie_final_length (original_length : ℕ) (cut_length : ℕ) (final_length : ℕ) 
  (h1 : original_length = 60) (h2 : cut_length = 8) : 
  final_length = 52 :=
by
  sorry

end movie_final_length_l110_110005


namespace calculate_fraction_l110_110765

def x : ℚ := 2 / 3
def y : ℚ := 8 / 10

theorem calculate_fraction :
  (6 * x + 10 * y) / (60 * x * y) = 3 / 8 := by
  sorry

end calculate_fraction_l110_110765


namespace triangle_side_length_l110_110265

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l110_110265


namespace fraction_simplification_l110_110435

theorem fraction_simplification :
  ( (5^1004)^4 - (5^1002)^4 ) / ( (5^1003)^4 - (5^1001)^4 ) = 25 := by
  sorry

end fraction_simplification_l110_110435


namespace quadratic_roots_x_no_real_solution_y_l110_110622

theorem quadratic_roots_x (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1) := sorry

theorem no_real_solution_y (y : ℝ) : 
  ¬∃ y : ℝ, 4*y^2 - 3*y + 2 = 0 := sorry

end quadratic_roots_x_no_real_solution_y_l110_110622


namespace proof_of_diagonals_and_angles_l110_110936

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def sum_of_internal_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem proof_of_diagonals_and_angles :
  let p_diagonals := number_of_diagonals 5
  let o_diagonals := number_of_diagonals 8
  let total_diagonals := p_diagonals + o_diagonals
  let p_internal_angles := sum_of_internal_angles 5
  let o_internal_angles := sum_of_internal_angles 8
  let total_internal_angles := p_internal_angles + o_internal_angles
  total_diagonals = 25 ∧ total_internal_angles = 1620 :=
by
  sorry

end proof_of_diagonals_and_angles_l110_110936


namespace schedule_arrangements_l110_110399

-- Define the initial setup of the problem
def subjects : List String := ["Chinese", "Mathematics", "English", "Physics", "Chemistry", "Biology"]

def periods_morning : List String := ["P1", "P2", "P3", "P4"]
def periods_afternoon : List String := ["P5", "P6", "P7"]

-- Define the constraints
def are_consecutive (subj1 subj2 : String) : Bool := 
  (subj1 = "Chinese" ∧ subj2 = "Mathematics") ∨ 
  (subj1 = "Mathematics" ∧ subj2 = "Chinese")

def can_schedule_max_one_period (subject : String) : Bool :=
  subject = "English" ∨ subject = "Physics" ∨ subject = "Chemistry" ∨ subject = "Biology"

-- Define the math problem as a proof in Lean
theorem schedule_arrangements : 
  ∃ n : Nat, n = 336 :=
by
  -- The detailed proof steps would go here
  sorry

end schedule_arrangements_l110_110399


namespace episodes_first_season_l110_110458

theorem episodes_first_season :
  ∃ (E : ℕ), (100000 * E + 200000 * (3 / 2) * E + 200000 * (3 / 2)^2 * E + 200000 * (3 / 2)^3 * E + 200000 * 24 = 16800000) ∧ E = 8 := 
by {
  sorry
}

end episodes_first_season_l110_110458


namespace triangle_area_is_14_l110_110600

def vector : Type := (ℝ × ℝ)
def a : vector := (4, -1)
def b : vector := (2 * 2, 2 * 3)

noncomputable def parallelogram_area (u v : vector) : ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  abs (ux * vy - uy * vx)

noncomputable def triangle_area (u v : vector) : ℝ :=
  (parallelogram_area u v) / 2

theorem triangle_area_is_14 : triangle_area a b = 14 :=
by
  unfold a b triangle_area parallelogram_area
  sorry

end triangle_area_is_14_l110_110600


namespace distinct_ordered_pairs_count_l110_110381

theorem distinct_ordered_pairs_count :
  ∃ (n : ℕ), n = 29 ∧ (∀ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b → a + b = 30 → ∃! p : ℕ × ℕ, p = (a, b)) :=
sorry

end distinct_ordered_pairs_count_l110_110381


namespace interval_satisfaction_l110_110532

theorem interval_satisfaction (a : ℝ) :
  (4 ≤ a / (3 * a - 6)) ∧ (a / (3 * a - 6) > 12) → a < 72 / 35 := 
by
  sorry

end interval_satisfaction_l110_110532


namespace find_square_side_length_l110_110345

open Nat

def original_square_side_length (s : ℕ) : Prop :=
  let length := s + 8
  let breadth := s + 4
  (2 * (length + breadth)) = 40 → s = 4

theorem find_square_side_length (s : ℕ) : original_square_side_length s := by
  sorry

end find_square_side_length_l110_110345


namespace correct_geometry_problems_l110_110168

-- Let A_c be the number of correct algebra problems.
-- Let A_i be the number of incorrect algebra problems.
-- Let G_c be the number of correct geometry problems.
-- Let G_i be the number of incorrect geometry problems.

def algebra_correct_incorrect_ratio (A_c A_i : ℕ) : Prop :=
  A_c * 2 = A_i * 3

def geometry_correct_incorrect_ratio (G_c G_i : ℕ) : Prop :=
  G_c * 1 = G_i * 4

def total_algebra_problems (A_c A_i : ℕ) : Prop :=
  A_c + A_i = 25

def total_geometry_problems (G_c G_i : ℕ) : Prop :=
  G_c + G_i = 35

def total_problems (A_c A_i G_c G_i : ℕ) : Prop :=
  A_c + A_i + G_c + G_i = 60

theorem correct_geometry_problems (A_c A_i G_c G_i : ℕ) :
  algebra_correct_incorrect_ratio A_c A_i →
  geometry_correct_incorrect_ratio G_c G_i →
  total_algebra_problems A_c A_i →
  total_geometry_problems G_c G_i →
  total_problems A_c A_i G_c G_i →
  G_c = 28 :=
sorry

end correct_geometry_problems_l110_110168


namespace part1_part2_l110_110074

def setA (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

def setB (a x : ℝ) : Prop := 2 * a - 1 ≤ x ∧ x < a + 5

open Set

theorem part1 : 
  let a := 0
  A = {x : ℝ | -1 < x ∧ x < 6} →
  B a = {x : ℝ | -1 ≤ x ∧ x < 5} →
  {x | (setA x) ∧ (setB a x)} = {x | -1 < x ∧ x < 5} :=
by
  sorry

theorem part2 : 
  A = {x : ℝ | -1 < x ∧ x < 6} →
  (B : ℝ → Set real) →
  (∀ x, (setA x ∨ setB a x) → setA x) →
  { a : ℝ | (0 < a ∧ a ≤ 1) ∨ a ≥ 6 } :=
by
  sorry

end part1_part2_l110_110074


namespace factorize_expression_l110_110359

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l110_110359


namespace curve_three_lines_intersect_at_origin_l110_110627

theorem curve_three_lines_intersect_at_origin (a : ℝ) :
  ((∀ x y : ℝ, (x + 2 * y + a) * (x^2 - y^2) = 0 → 
    ((y = x ∨ y = -x ∨ y = - (1/2) * x - a/2) ∧ 
     (x = 0 ∧ y = 0)))) ↔ a = 0 :=
sorry

end curve_three_lines_intersect_at_origin_l110_110627


namespace maximum_value_of_expression_l110_110835

variable (x y z : ℝ)

theorem maximum_value_of_expression (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) : 
  x + y^3 + z^4 ≤ 1 :=
sorry

end maximum_value_of_expression_l110_110835


namespace polynomial_multiple_of_six_l110_110604

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end polynomial_multiple_of_six_l110_110604


namespace solve_for_a_l110_110393

theorem solve_for_a (a b c : ℤ) (h1 : a + b = c) (h2 : b + c = 6) (h3 : c = 4) : a = 2 :=
by
  sorry

end solve_for_a_l110_110393


namespace geometric_seq_common_ratio_l110_110205

theorem geometric_seq_common_ratio 
  (a : ℝ) (q : ℝ)
  (h1 : a * q^2 = 4)
  (h2 : a * q^5 = 1 / 2) : 
  q = 1 / 2 := 
by
  sorry

end geometric_seq_common_ratio_l110_110205


namespace at_least_one_gt_one_l110_110832

theorem at_least_one_gt_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l110_110832


namespace find_b_l110_110271

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end find_b_l110_110271


namespace wire_length_before_cut_l110_110019

-- Defining the conditions
def wire_cut (L S : ℕ) : Prop :=
  S = 20 ∧ S = (2 / 5 : ℚ) * L

-- The statement we need to prove
theorem wire_length_before_cut (L S : ℕ) (h : wire_cut L S) : (L + S) = 70 := 
by 
  sorry

end wire_length_before_cut_l110_110019


namespace total_cost_toys_l110_110419

variable (c_e_actionfigs : ℕ := 60) -- number of action figures for elder son
variable (cost_e_actionfig : ℕ := 5) -- cost per action figure for elder son
variable (c_y_actionfigs : ℕ := 3 * c_e_actionfigs) -- number of action figures for younger son
variable (cost_y_actionfig : ℕ := 4) -- cost per action figure for younger son
variable (c_y_cars : ℕ := 20) -- number of cars for younger son
variable (cost_car : ℕ := 3) -- cost per car
variable (c_y_animals : ℕ := 10) -- number of stuffed animals for younger son
variable (cost_animal : ℕ := 7) -- cost per stuffed animal

theorem total_cost_toys (c_e_actionfigs c_y_actionfigs c_y_cars c_y_animals : ℕ)
                         (cost_e_actionfig cost_y_actionfig cost_car cost_animal : ℕ) :
  (c_e_actionfigs * cost_e_actionfig + c_y_actionfigs * cost_y_actionfig + 
  c_y_cars * cost_car + c_y_animals * cost_animal) = 1150 := by
  sorry

end total_cost_toys_l110_110419


namespace find_f_pi_over_4_l110_110220

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem find_f_pi_over_4
  (ω φ : ℝ)
  (hω_gt_0 : ω > 0)
  (hφ_lt_pi_over_2 : |φ| < Real.pi / 2)
  (h_mono_dec : ∀ x₁ x₂, (Real.pi / 6 < x₁ ∧ x₁ < Real.pi / 3 ∧ Real.pi / 3 < x₂ ∧ x₂ < 2 * Real.pi / 3) → f x₁ ω φ > f x₂ ω φ)
  (h_values_decreasing : f (Real.pi / 6) ω φ = 1 ∧ f (2 * Real.pi / 3) ω φ = -1) : 
  f (Real.pi / 4) 2 (Real.pi / 6) = Real.sqrt 3 / 2 :=
sorry

end find_f_pi_over_4_l110_110220


namespace max_value_of_f_l110_110049

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f (Real.exp 1) = 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l110_110049


namespace coin_flip_sequences_l110_110691

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110691


namespace range_of_k_l110_110198

theorem range_of_k :
  ∀ k : ℝ, (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ (-4 < k ∧ k ≤ 0) :=
by
  sorry

end range_of_k_l110_110198


namespace cody_candy_total_l110_110907

theorem cody_candy_total
  (C_c : ℕ) (C_m : ℕ) (P_b : ℕ)
  (h1 : C_c = 7) (h2 : C_m = 3) (h3 : P_b = 8) :
  (C_c + C_m) * P_b = 80 :=
by
  sorry

end cody_candy_total_l110_110907


namespace peyton_total_yards_l110_110991

def distance_on_Saturday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def distance_on_Sunday (throws: Nat) (yards_per_throw: Nat) : Nat :=
  throws * yards_per_throw

def total_distance (distance_Saturday: Nat) (distance_Sunday: Nat) : Nat :=
  distance_Saturday + distance_Sunday

theorem peyton_total_yards :
  let throws_Saturday := 20
  let yards_per_throw_Saturday := 20
  let throws_Sunday := 30
  let yards_per_throw_Sunday := 40
  distance_on_Saturday throws_Saturday yards_per_throw_Saturday +
  distance_on_Sunday throws_Sunday yards_per_throw_Sunday = 1600 :=
by
  sorry

end peyton_total_yards_l110_110991


namespace intersection_a_zero_range_of_a_l110_110077

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l110_110077


namespace michael_pays_106_l110_110285

def num_cats : ℕ := 2
def num_dogs : ℕ := 3
def num_parrots : ℕ := 1
def num_fish : ℕ := 4

def cost_per_cat : ℕ := 13
def cost_per_dog : ℕ := 18
def cost_per_parrot : ℕ := 10
def cost_per_fish : ℕ := 4

def total_cost : ℕ :=
  (num_cats * cost_per_cat) +
  (num_dogs * cost_per_dog) +
  (num_parrots * cost_per_parrot) +
  (num_fish * cost_per_fish)

theorem michael_pays_106 : total_cost = 106 := by
  sorry

end michael_pays_106_l110_110285


namespace clock_angle_3_45_l110_110316

/-- The smaller angle between the hour hand and the minute hand of a 12-hour analog clock at 3:45 p.m. is 202.5 degrees. -/
theorem clock_angle_3_45 :
  let hour_angle := 112.5
      minute_angle := 270
      angle_diff := abs (minute_angle - hour_angle) in
  min angle_diff (360 - angle_diff) = 202.5 :=
by
  let hour_angle := 112.5
  let minute_angle := 270
  let angle_diff := abs (minute_angle - hour_angle)
  have smaller_angle := min angle_diff (360 - angle_diff)
  sorry

end clock_angle_3_45_l110_110316


namespace no_function_satisfies_condition_l110_110194

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℤ → ℤ, ∀ x y : ℤ, f (x + f y) = f x - y :=
sorry

end no_function_satisfies_condition_l110_110194


namespace kolya_is_wrong_l110_110649

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l110_110649


namespace john_initial_running_time_l110_110594

theorem john_initial_running_time (H : ℝ) (hH1 : 1.75 * H = 168 / 12)
: H = 8 :=
sorry

end john_initial_running_time_l110_110594


namespace inequality_proof_l110_110947

open Real

theorem inequality_proof
  (a b c d : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d > 0)
  (h5 : a * b + b * c + c * d + d * a = 1) :
  (a^2 / (b + c + d) + b^2 / (c + d + a) +
   c^2 / (d + a + b) + d^2 / (a + b + c) ≥ 2 / 3) :=
by
  sorry

end inequality_proof_l110_110947


namespace complete_square_rewrite_l110_110427

theorem complete_square_rewrite (j i : ℂ) :
  let c := 8
  let p := (3 * i / 8 : ℂ)
  let q := (137 / 8 : ℂ)
  (8 * j^2 + 6 * i * j + 16 = c * (j + p)^2 + q) →
  q / p = - (137 * i / 3) :=
by
  sorry

end complete_square_rewrite_l110_110427


namespace quadratic_vertex_form_l110_110243

theorem quadratic_vertex_form (a h k x: ℝ) (h_a : a = 3) (hx : 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) : 
  h = -3 / 2 :=
by {
  sorry
}

end quadratic_vertex_form_l110_110243


namespace least_positive_integer_l110_110922

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l110_110922


namespace least_positive_integer_l110_110928

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l110_110928


namespace fraction_field_planted_l110_110916

-- Define the problem conditions
structure RightTriangle (leg1 leg2 hypotenuse : ℝ) : Prop :=
  (right_angle : ∃ (A B C : ℝ), A = 5 ∧ B = 12 ∧ hypotenuse = 13 ∧ A^2 + B^2 = hypotenuse^2)

structure SquarePatch (shortest_distance : ℝ) : Prop :=
  (distance_to_hypotenuse : shortest_distance = 3)

-- Define the statement
theorem fraction_field_planted (T : RightTriangle 5 12 13) (P : SquarePatch 3) : 
  ∃ (fraction : ℚ), fraction = 7 / 10 :=
by
  sorry

end fraction_field_planted_l110_110916


namespace farmer_shipped_30_boxes_this_week_l110_110597

-- Defining the given conditions
def last_week_boxes : ℕ := 10
def last_week_pomelos : ℕ := 240
def this_week_dozen : ℕ := 60
def pomelos_per_dozen : ℕ := 12

-- Translating conditions into mathematical statements
def pomelos_per_box_last_week : ℕ := last_week_pomelos / last_week_boxes
def this_week_pomelos_total : ℕ := this_week_dozen * pomelos_per_dozen
def boxes_shipped_this_week : ℕ := this_week_pomelos_total / pomelos_per_box_last_week

-- The theorem we prove, that given the conditions, the number of boxes shipped this week is 30.
theorem farmer_shipped_30_boxes_this_week :
  boxes_shipped_this_week = 30 :=
sorry

end farmer_shipped_30_boxes_this_week_l110_110597


namespace percentage_of_music_students_l110_110641

theorem percentage_of_music_students 
  (total_students : ℕ) 
  (dance_students : ℕ) 
  (art_students : ℕ) 
  (drama_students : ℕ)
  (h_total : total_students = 2000) 
  (h_dance : dance_students = 450) 
  (h_art : art_students = 680) 
  (h_drama : drama_students = 370) 
  : (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 
:= by 
  sorry

end percentage_of_music_students_l110_110641


namespace max_value_of_expression_l110_110837

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 :=
sorry

end max_value_of_expression_l110_110837


namespace coin_flip_sequences_l110_110732

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110732


namespace marvin_birthday_next_thursday_l110_110126

-- Define the basic properties of leap years and weekday progression.
def is_leap (year : ℕ) : Prop := (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

-- Given conditions
def may27_weekday (year : ℕ) : ℕ :=
  let start_weekday : ℕ := 5 -- Friday in the year 2007
  let number_of_days := List.range (year - 2007)
  number_of_days.foldl (λ acc y =>
    if is_leap (y + 2007) then (acc + 2) % 7 else (acc + 1) % 7
  ) start_weekday

-- Theorem to be proven: The first occurrence after 2007 where May 27 is a Thursday.
theorem marvin_birthday_next_thursday : ∃ (year : ℕ), year > 2007 ∧ may27_weekday year = 4 ∧ year = 2017 := by
  have : ∀ y ∈ List.range (2017 - 2007), may27_weekday (y + 2007) ≠ 4 := by sorry
  have : may27_weekday 2017 = 4 := by sorry
  exact ⟨2017, Nat.lt_succ_self 2007, by rwa [←Nat.succ_pred_eq_of_pos, not_ne_iff, ←List.foldr_ext (· + ·)]⟩

end marvin_birthday_next_thursday_l110_110126


namespace values_of_x_plus_y_l110_110957

theorem values_of_x_plus_y (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 :=
sorry

end values_of_x_plus_y_l110_110957


namespace calculate_expression_l110_110027

theorem calculate_expression :
  12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 :=
by
  sorry

end calculate_expression_l110_110027


namespace arithmetic_sequence_nth_term_639_l110_110631

theorem arithmetic_sequence_nth_term_639 :
  ∀ (x n : ℕ) (a₁ a₂ a₃ aₙ : ℤ),
  a₁ = 3 * x - 5 →
  a₂ = 7 * x - 17 →
  a₃ = 4 * x + 3 →
  aₙ = a₁ + (n - 1) * (a₂ - a₁) →
  aₙ = 4018 →
  n = 639 :=
by
  intros x n a₁ a₂ a₃ aₙ h₁ h₂ h₃ hₙ hₙ_eq
  sorry

end arithmetic_sequence_nth_term_639_l110_110631


namespace probability_event_l110_110212

open ProbabilityTheory

def uniform_random_variable (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 1

theorem probability_event (a : ℝ) (h : uniform_random_variable a) :
  probability (event {a | 3 * a - 1 < 0}) = 1 / 3 :=
sorry

end probability_event_l110_110212


namespace binomial_15_4_l110_110482

theorem binomial_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binomial_15_4_l110_110482


namespace rate_per_kg_for_grapes_l110_110350

theorem rate_per_kg_for_grapes (G : ℝ) (h : 9 * G + 9 * 55 = 1125) : G = 70 :=
by
  -- sorry to skip the proof
  sorry

end rate_per_kg_for_grapes_l110_110350


namespace product_of_numbers_l110_110331

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x^3 + y^3 = 9450) : x * y = -585 :=
  sorry

end product_of_numbers_l110_110331


namespace distinct_sequences_ten_flips_l110_110722

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110722


namespace at_least_one_genuine_l110_110940

/-- Given 12 products, of which 10 are genuine and 2 are defective.
    If 3 products are randomly selected, then at least one of the selected products is a genuine product. -/
theorem at_least_one_genuine : 
  ∀ (products : Fin 12 → Prop), 
  (∃ n₁ n₂ : Fin 12, (n₁ ≠ n₂) ∧ 
                   (products n₁ = true) ∧ 
                   (products n₂ = true) ∧ 
                   (∃ n₁' n₂' : Fin 12, (n₁ ≠ n₁' ∧ n₂ ≠ n₂') ∧
                                         products n₁' = products n₂' = true ∧
                                         ∀ j : Fin 3, products j = true)) → 
  (∃ m : Fin 3, products m = true) :=
sorry

end at_least_one_genuine_l110_110940


namespace smaller_angle_at_3_45_l110_110319

/-- 
  Determine the smaller angle between the hour hand and the minute hand at exactly 3:45 p.m.
  on a 12-hour analog clock.
-/
theorem smaller_angle_at_3_45 :
  let hour_hand_position := 112.5,
      minute_hand_position := 270,
      angle_between_hands := abs (minute_hand_position - hour_hand_position),
      smaller_angle := if angle_between_hands <= 180 then angle_between_hands else 360 - angle_between_hands
  in smaller_angle = 157.5 :=
by
  sorry

end smaller_angle_at_3_45_l110_110319


namespace le_condition_l110_110994

-- Given positive numbers a, b, c
variables {a b c : ℝ}
-- Assume positive values for the numbers
variables (ha : a > 0) (hb : b > 0) (hc : c > 0)
-- Given condition a² + b² - ab = c²
axiom condition : a^2 + b^2 - a*b = c^2

-- We need to prove (a - c)(b - c) ≤ 0
theorem le_condition : (a - c) * (b - c) ≤ 0 :=
sorry

end le_condition_l110_110994


namespace point_B_coordinates_l110_110213

/-
Problem Statement:
Given a point A(2, 4) which is symmetric to point B with respect to the origin,
we need to prove the coordinates of point B.
-/

structure Point where
  x : ℝ
  y : ℝ

def symmetric_wrt_origin (A B : Point) : Prop :=
  B.x = -A.x ∧ B.y = -A.y

noncomputable def point_A : Point := ⟨2, 4⟩
noncomputable def point_B : Point := ⟨-2, -4⟩

theorem point_B_coordinates : symmetric_wrt_origin point_A point_B :=
  by
    -- Proof is omitted
    sorry

end point_B_coordinates_l110_110213


namespace min_small_containers_needed_l110_110013

def medium_container_capacity : ℕ := 450
def small_container_capacity : ℕ := 28

theorem min_small_containers_needed : ⌈(medium_container_capacity : ℝ) / small_container_capacity⌉ = 17 :=
by
  sorry

end min_small_containers_needed_l110_110013


namespace lesser_fraction_l110_110216

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 10 / 11) (h2 : x * y = 1 / 8) : min x y = (80 - 2 * Real.sqrt 632) / 176 := 
by sorry

end lesser_fraction_l110_110216


namespace sum_of_number_and_conjugate_l110_110496

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l110_110496


namespace sum_max_min_X_l110_110416

def I := {i : ℕ | 1 ≤ i ∧ i ≤ 2020}

def W (a b : ℕ) : ℕ := (a + b) + (a * b)
def Y (a b : ℕ) : ℕ := (a + b) * (a * b)

def X : Set ℕ := {x | ∃ a b : ℤ, W a b = x ∧ Y a b = x}

theorem sum_max_min_X : 
  (∃ x_max x_min : ℕ, x_max ∈ X ∧ x_min ∈ X ∧ 
  (∀ x : ℕ, x ∈ X → x ≤ x_max) ∧ (∀ x : ℕ, x ∈ X → x_min ≤ x) ∧ 
  (x_max + x_min = 58)) :=
sorry

end sum_max_min_X_l110_110416


namespace prove_3a_3b_3c_l110_110569

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end prove_3a_3b_3c_l110_110569


namespace biff_break_even_night_hours_l110_110903

-- Define the constants and conditions
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def lunch_cost : ℝ := 8
def dinner_cost : ℝ := 10
def accommodation_cost : ℝ := 35

def total_expenses_without_wifi : ℝ := ticket_cost + snacks_cost + headphones_cost + lunch_cost + dinner_cost + accommodation_cost

def earnings_per_hour : ℝ := 12
def wifi_cost_day : ℝ := 2
def wifi_cost_night : ℝ := 1

-- Define the total expenses with wifi cost variable
def total_expenses (D N : ℝ) : ℝ := total_expenses_without_wifi + (wifi_cost_day * D) + (wifi_cost_night * N)

-- Define the total earnings
def total_earnings (D N : ℝ) : ℝ := earnings_per_hour * (D + N)

-- Prove that the minimum number of hours Biff needs to work at night to break even is 8 hours
theorem biff_break_even_night_hours :
  ∃ N : ℕ, N = 8 ∧ total_earnings 0 N ≥ total_expenses 0 N := 
by 
  sorry

end biff_break_even_night_hours_l110_110903


namespace solve_for_x_l110_110672

theorem solve_for_x (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end solve_for_x_l110_110672


namespace total_cost_correct_l110_110592

-- Definitions for the conditions
def num_ladders_1 : ℕ := 10
def rungs_1 : ℕ := 50
def cost_per_rung_1 : ℕ := 2

def num_ladders_2 : ℕ := 20
def rungs_2 : ℕ := 60
def cost_per_rung_2 : ℕ := 3

def num_ladders_3 : ℕ := 30
def rungs_3 : ℕ := 80
def cost_per_rung_3 : ℕ := 4

-- Total cost calculation for the client
def total_cost : ℕ :=
  (num_ladders_1 * rungs_1 * cost_per_rung_1) +
  (num_ladders_2 * rungs_2 * cost_per_rung_2) +
  (num_ladders_3 * rungs_3 * cost_per_rung_3)

-- Statement to be proved
theorem total_cost_correct : total_cost = 14200 :=
by {
  sorry
}

end total_cost_correct_l110_110592


namespace range_of_x_l110_110375

theorem range_of_x (x : ℝ) (p : x^2 - 2 * x - 3 < 0) (q : 1 / (x - 2) < 0) : -1 < x ∧ x < 2 :=
by
  sorry

end range_of_x_l110_110375


namespace time_to_destination_l110_110001

theorem time_to_destination (speed_ratio : ℕ) (mr_harris_time : ℕ) 
  (distance_multiple : ℕ) (h1 : speed_ratio = 3) 
  (h2 : mr_harris_time = 3) 
  (h3 : distance_multiple = 5) : 
  (mr_harris_time / speed_ratio) * distance_multiple = 5 := by
  sorry

end time_to_destination_l110_110001


namespace radius_of_tangent_circle_l110_110339

-- Define the conditions
def is_45_45_90_triangle (A B C : ℝ × ℝ) (AB BC AC : ℝ) : Prop :=
  (AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2) ∧
  (A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2))

def is_tangent_to_axes (O : ℝ × ℝ) (r : ℝ) : Prop :=
  O = (r, r)

def is_tangent_to_hypotenuse (O : ℝ × ℝ) (r : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - O.1) = Real.sqrt 2 * r ∧ (C.2 - O.2) = Real.sqrt 2 * r

-- Main theorem
theorem radius_of_tangent_circle :
  ∃ r : ℝ, ∀ (A B C O : ℝ × ℝ),
    is_45_45_90_triangle A B C (2) (2) (2 * Real.sqrt 2) →
    is_tangent_to_axes O r →
    is_tangent_to_hypotenuse O r C →
    r = Real.sqrt 2 :=
by
  sorry

end radius_of_tangent_circle_l110_110339


namespace quarters_to_dollars_l110_110955

theorem quarters_to_dollars (total_quarters : ℕ) (quarters_per_dollar : ℕ) (h1 : total_quarters = 8) (h2 : quarters_per_dollar = 4) : total_quarters / quarters_per_dollar = 2 :=
by {
  sorry
}

end quarters_to_dollars_l110_110955


namespace students_in_lower_grades_l110_110864

noncomputable def seniors : ℕ := 300
noncomputable def percentage_cars_seniors : ℝ := 0.40
noncomputable def percentage_cars_remaining : ℝ := 0.10
noncomputable def total_percentage_cars : ℝ := 0.15

theorem students_in_lower_grades (X : ℝ) :
  (0.15 * (300 + X) = 120 + 0.10 * X) → X = 1500 :=
by
  intro h
  sorry

end students_in_lower_grades_l110_110864


namespace last_three_digits_of_7_exp_1987_l110_110353

theorem last_three_digits_of_7_exp_1987 : (7 ^ 1987) % 1000 = 543 := by
  sorry

end last_three_digits_of_7_exp_1987_l110_110353


namespace find_k_l110_110785

def polynomial (n : ℤ) : ℤ := 3 * n^6 + 26 * n^4 + 33 * n^2 + 1

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : k ≤ 100) :
  (∃ n : ℤ, polynomial n % k = 0) ↔ k ∈ {9, 21, 27, 39, 49, 57, 63, 81, 87, 91, 93} := 
  sorry

end find_k_l110_110785


namespace triangle_problem_l110_110827

/-- In triangle ABC, the sides opposite to angles A, B, C are a, b, c respectively.
Given that b = sqrt 2, c = 3, B + C = 3A, prove:
1. The length of side a equals sqrt 5.
2. sin (B + 3π/4) equals sqrt(10) / 10.
-/
theorem triangle_problem 
  (a b c A B C : ℝ)
  (hb : b = Real.sqrt 2)
  (hc : c = 3)
  (hBC : B + C = 3 * A)
  (hA : A = π / 4)
  : (a = Real.sqrt 5)
  ∧ (Real.sin (B + 3 * π / 4) = Real.sqrt 10 / 10) :=
sorry

end triangle_problem_l110_110827


namespace construct_segment_eq_abc_div_de_l110_110377

theorem construct_segment_eq_abc_div_de 
(a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
  ∃ x : ℝ, x = (a * b * c) / (d * e) :=
by sorry

end construct_segment_eq_abc_div_de_l110_110377


namespace factor_expression_l110_110774

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110774


namespace area_S_l110_110120

noncomputable def omega : ℂ := -1/2 + (1/2) * complex.I * real.sqrt 3

def S' := {z : ℂ | ∃ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧
                             0 ≤ b ∧ b ≤ 1 ∧
                             0 ≤ c ∧ c ≤ 2 ∧ 
                             z = a + b * omega + c * omega^2}

theorem area_S'_is_6sqrt3 : complex.area S' = 6 * real.sqrt 3 := 
sorry

end area_S_l110_110120


namespace fred_cantaloupes_l110_110254

def num_cantaloupes_K : ℕ := 29
def num_cantaloupes_J : ℕ := 20
def total_cantaloupes : ℕ := 65

theorem fred_cantaloupes : ∃ F : ℕ, num_cantaloupes_K + num_cantaloupes_J + F = total_cantaloupes ∧ F = 16 :=
by
  sorry

end fred_cantaloupes_l110_110254


namespace perimeter_gt_sixteen_l110_110063

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l110_110063


namespace find_b_of_perpendicular_lines_l110_110874

theorem find_b_of_perpendicular_lines (b : ℝ) (h : 4 * b - 8 = 0) : b = 2 := 
by 
  sorry

end find_b_of_perpendicular_lines_l110_110874


namespace probability_one_first_class_probability_two_first_class_probability_no_third_class_l110_110459

open Finset Nat

-- Define the conditions of the problem
def total_pens := 6
def first_class_pens := 3
def second_class_pens := 2
def third_class_pens := 1
def drawn_pens := 3

-- Define combinations
def comb (n k : ℕ) := (n.choose k).toNat

-- Calculate probabilities
def prob_of_event (f : ℝ) (s : ℝ) (t : ℝ) (total : ℝ) : ℝ := 
  (f * s * t) / total

-- Specific events' probabilities to be proven
theorem probability_one_first_class :
  prob_of_event (comb first_class_pens 1) 
                (comb (second_class_pens + third_class_pens) 2) 
                1 
                (comb total_pens drawn_pens : ℝ)
  = 9 / 20 := sorry

theorem probability_two_first_class :
  prob_of_event (comb first_class_pens 2) 
                (comb (second_class_pens + third_class_pens) 1) 
                1 
                (comb total_pens drawn_pens : ℝ)
  = 9 / 20 := sorry

theorem probability_no_third_class :
  prob_of_event (comb (first_class_pens + second_class_pens) 
                       drawn_pens) 
                1 
                1 
                (comb total_pens drawn_pens : ℝ)
  = 1 / 2 := sorry

end probability_one_first_class_probability_two_first_class_probability_no_third_class_l110_110459


namespace smallest_M_conditions_l110_110367

theorem smallest_M_conditions :
  ∃ M : ℕ, M > 0 ∧
  ((∃ k₁, M = 8 * k₁) ∨ (∃ k₂, M + 2 = 8 * k₂) ∨ (∃ k₃, M + 4 = 8 * k₃)) ∧
  ((∃ k₄, M = 9 * k₄) ∨ (∃ k₅, M + 2 = 9 * k₅) ∨ (∃ k₆, M + 4 = 9 * k₆)) ∧
  ((∃ k₇, M = 25 * k₇) ∨ (∃ k₈, M + 2 = 25 * k₈) ∨ (∃ k₉, M + 4 = 25 * k₉)) ∧
  M = 100 :=
sorry

end smallest_M_conditions_l110_110367


namespace ratio_cost_to_marked_price_l110_110463

theorem ratio_cost_to_marked_price (x : ℝ) 
  (h_discount: ∀ y, y = marked_price → selling_price = (3/4) * y)
  (h_cost: ∀ z, z = selling_price → cost_price = (2/3) * z) :
  cost_price / marked_price = 1 / 2 :=
by
  sorry

end ratio_cost_to_marked_price_l110_110463


namespace original_cube_edge_length_l110_110748

theorem original_cube_edge_length (a : ℕ) (h1 : 6 * (a ^ 3) = 7 * (6 * (a ^ 2))) : a = 7 := 
by 
  sorry

end original_cube_edge_length_l110_110748


namespace find_length_of_other_diagonal_l110_110628

theorem find_length_of_other_diagonal
  (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h1: area = 75)
  (h2: d1 = 10) :
  d2 = 15 :=
by 
  sorry

end find_length_of_other_diagonal_l110_110628


namespace paul_score_higher_by_26_l110_110898

variable {R : Type} [LinearOrderedField R]

variables (A1 A2 A3 P1 P2 P3 : R)

-- hypotheses
variable (h1 : A1 = P1 + 10)
variable (h2 : A2 = P2 + 4)
variable (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)

-- goal
theorem paul_score_higher_by_26 : P3 - A3 = 26 := by
  sorry

end paul_score_higher_by_26_l110_110898


namespace proof_l110_110102

noncomputable def problem : Prop :=
  let a := 1
  let b := 2
  let angleC := 60 * Real.pi / 180 -- convert degrees to radians
  let cosC := Real.cos angleC
  let sinC := Real.sin angleC
  let c_squared := a^2 + b^2 - 2 * a * b * cosC
  let c := Real.sqrt c_squared
  let area := 0.5 * a * b * sinC
  c = Real.sqrt 3 ∧ area = Real.sqrt 3 / 2

theorem proof : problem :=
by
  sorry

end proof_l110_110102


namespace common_difference_is_minus_3_l110_110215

variable (a_n : ℕ → ℤ) (a1 d : ℤ)

-- Definitions expressing the conditions of the problem
def arithmetic_prog : Prop := ∀ (n : ℕ), a_n n = a1 + (n - 1) * d

def condition1 : Prop := a1 + (a1 + 6 * d) = -8

def condition2 : Prop := a1 + d = 2

-- The statement we need to prove
theorem common_difference_is_minus_3 :
  arithmetic_prog a_n a1 d ∧ condition1 a1 d ∧ condition2 a1 d → d = -3 :=
by {
  -- The proof would go here
  sorry
}

end common_difference_is_minus_3_l110_110215


namespace find_rectangle_area_l110_110309

noncomputable def rectangle_area (a b : ℕ) : ℕ :=
  a * b

theorem find_rectangle_area (a b : ℕ) :
  (5 : ℚ) / 8 = (a : ℚ) / b ∧ (a + 6) * (b + 6) - a * b = 114 ∧ a + b = 13 →
  rectangle_area a b = 40 :=
by
  sorry

end find_rectangle_area_l110_110309


namespace find_positive_a_l110_110200

noncomputable def find_a (a : ℝ) : Prop :=
  ((x - a/x)^7).coeff x^3 = 84 ∧ a > 0

theorem find_positive_a (a : ℝ) : find_a a → a = 2 :=
by
  sorry

end find_positive_a_l110_110200


namespace overall_average_marks_l110_110106

theorem overall_average_marks 
  (num_candidates : ℕ) 
  (num_passed : ℕ) 
  (avg_passed : ℕ) 
  (avg_failed : ℕ)
  (h1 : num_candidates = 120) 
  (h2 : num_passed = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (num_passed * avg_passed + (num_candidates - num_passed) * avg_failed) / num_candidates = 35 := 
by
  sorry

end overall_average_marks_l110_110106


namespace num_divisible_by_33_l110_110379

theorem num_divisible_by_33 : ∀ (x y : ℕ), 
  (0 ≤ x ∧ x ≤ 9) → (0 ≤ y ∧ y ≤ 9) →
  (19 + x + y) % 3 = 0 →
  (x - y + 1) % 11 = 0 →
  ∃! (n : ℕ), (20070002008 * 100 + x * 10 + y) = n ∧ n % 33 = 0 :=
by
  intros x y hx hy h3 h11
  sorry

end num_divisible_by_33_l110_110379


namespace cubic_expression_value_l110_110946

theorem cubic_expression_value (m : ℝ) (h : m^2 + 3 * m - 2023 = 0) :
  m^3 + 2 * m^2 - 2026 * m - 2023 = -4046 :=
by
  sorry

end cubic_expression_value_l110_110946


namespace probability_at_least_one_exceeds_one_dollar_l110_110470

noncomputable def prob_A : ℚ := 2 / 3
noncomputable def prob_B : ℚ := 1 / 2
noncomputable def prob_C : ℚ := 1 / 4

theorem probability_at_least_one_exceeds_one_dollar :
  (1 - ((1 - prob_A) * (1 - prob_B) * (1 - prob_C))) = 7 / 8 :=
by
  -- The proof can be conducted here
  sorry

end probability_at_least_one_exceeds_one_dollar_l110_110470


namespace problem_solution_l110_110961

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b = 14) 
  (h2 : b + c = 9) 
  (h3 : c + d = 3) : 
  a + d = 8 := 
by 
  sorry

end problem_solution_l110_110961


namespace largest_difference_l110_110408

noncomputable def A := 3 * 2023^2024
noncomputable def B := 2023^2024
noncomputable def C := 2022 * 2023^2023
noncomputable def D := 3 * 2023^2023
noncomputable def E := 2023^2023
noncomputable def F := 2023^2022

theorem largest_difference :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l110_110408


namespace abc_positive_l110_110387

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  -- Proof goes here
  sorry

end abc_positive_l110_110387


namespace distance_from_point_to_line_l110_110111

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def cartesian_distance_to_line (point : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  abs (point.snd - y_line)

theorem distance_from_point_to_line
  (ρ θ : ℝ)
  (h_point : ρ = 2 ∧ θ = Real.pi / 6)
  (h_line : ∀ θ, (3 : ℝ) = ρ * Real.sin θ) :
  cartesian_distance_to_line (polar_to_cartesian ρ θ) 3 = 2 :=
  sorry

end distance_from_point_to_line_l110_110111


namespace bob_tiller_swath_width_l110_110764

theorem bob_tiller_swath_width
  (plot_width plot_length : ℕ)
  (tilling_rate_seconds_per_foot : ℕ)
  (total_tilling_minutes : ℕ)
  (total_area : ℕ)
  (tilled_length : ℕ)
  (swath_width : ℕ) :
  plot_width = 110 →
  plot_length = 120 →
  tilling_rate_seconds_per_foot = 2 →
  total_tilling_minutes = 220 →
  total_area = plot_width * plot_length →
  tilled_length = (total_tilling_minutes * 60) / tilling_rate_seconds_per_foot →
  swath_width = total_area / tilled_length →
  swath_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end bob_tiller_swath_width_l110_110764


namespace abs_quadratic_inequality_solution_set_l110_110861

theorem abs_quadratic_inequality_solution_set (x : ℝ) : (|x^2 - x| < 2) ↔ (x ∈ set.Ioo (-1 : ℝ) 2) :=
by
  sorry

end abs_quadratic_inequality_solution_set_l110_110861


namespace number_of_people_in_village_l110_110825

variable (P : ℕ) -- Define the total number of people in the village

def people_not_working : ℕ := 50
def people_with_families : ℕ := 25
def people_singing_in_shower : ℕ := 75
def max_people_overlap : ℕ := 50

theorem number_of_people_in_village :
  P - people_not_working + P - people_with_families + P - people_singing_in_shower - max_people_overlap = P → 
  P = 100 :=
by
  sorry

end number_of_people_in_village_l110_110825


namespace number_of_correct_statements_l110_110057

def line : Type := sorry
def plane : Type := sorry
def parallel (x y : line) : Prop := sorry
def perpendicular (x : line) (y : plane) : Prop := sorry
def subset (x : line) (y : plane) : Prop := sorry
def skew (x y : line) : Prop := sorry

variable (m n : line) -- two different lines
variable (alpha beta : plane) -- two different planes

theorem number_of_correct_statements :
  (¬parallel m alpha ∨ subset n alpha ∧ parallel m n) ∧
  (parallel m alpha ∧ perpendicular alpha beta ∧ perpendicular m n ∧ perpendicular n beta) ∧
  (subset m alpha ∧ subset n beta ∧ perpendicular m n) ∧
  (skew m n ∧ subset m alpha ∧ subset n beta ∧ parallel m beta ∧ parallel n alpha) :=
sorry

end number_of_correct_statements_l110_110057


namespace radical_conjugate_sum_l110_110507

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l110_110507


namespace proportional_sets_l110_110348

/-- Prove that among the sets of line segments, the ones that are proportional are: -/
theorem proportional_sets : 
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  ∃ a b c d, (a, b, c, d) = C ∧ (a * d = b * c) :=
by
  let A := (3, 6, 7, 9)
  let B := (2, 5, 6, 8)
  let C := (3, 6, 9, 18)
  let D := (1, 2, 3, 4)
  sorry

end proportional_sets_l110_110348


namespace coin_flip_sequences_l110_110686

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110686


namespace line_shift_upwards_l110_110098

theorem line_shift_upwards (x y : ℝ) (h : y = -2 * x) : y + 3 = -2 * x + 3 :=
by sorry

end line_shift_upwards_l110_110098


namespace walkway_and_border_area_correct_l110_110431

-- Definitions based on the given conditions
def flower_bed_width : ℕ := 8
def flower_bed_height : ℕ := 3
def walkway_width : ℕ := 2
def border_width : ℕ := 4
def num_rows : ℕ := 4
def num_columns : ℕ := 3

-- Total width calculation
def total_width : ℕ := 
  (flower_bed_width * num_columns) + (walkway_width * (num_columns + 1)) + (border_width * 2)

-- Total height calculation
def total_height : ℕ := 
  (flower_bed_height * num_rows) + (walkway_width * (num_rows + 1)) + (border_width * 2)

-- Total area of the garden including walkways and decorative border
def total_area : ℕ := total_width * total_height

-- Total area of flower beds
def flower_bed_area : ℕ := 
  (flower_bed_width * flower_bed_height) * (num_rows * num_columns)

-- Area of the walkways and decorative border
def walkway_and_border_area : ℕ := total_area - flower_bed_area

theorem walkway_and_border_area_correct : 
  walkway_and_border_area = 912 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end walkway_and_border_area_correct_l110_110431


namespace tony_quilt_square_side_length_l110_110871

theorem tony_quilt_square_side_length (length width : ℝ) (h_length : length = 6) (h_width : width = 24) : 
  ∃ s, s * s = length * width ∧ s = 12 :=
by
  sorry

end tony_quilt_square_side_length_l110_110871


namespace lasagna_package_weight_l110_110659

theorem lasagna_package_weight 
  (beef : ℕ) 
  (noodles_needed_per_beef : ℕ) 
  (current_noodles : ℕ) 
  (packages_needed : ℕ) 
  (noodles_per_package : ℕ) 
  (H1 : beef = 10)
  (H2 : noodles_needed_per_beef = 2)
  (H3 : current_noodles = 4)
  (H4 : packages_needed = 8)
  (H5 : noodles_per_package = (2 * beef - current_noodles) / packages_needed) :
  noodles_per_package = 2 := 
by
  sorry

end lasagna_package_weight_l110_110659


namespace FlyersDistributon_l110_110763

variable (total_flyers ryan_flyers alyssa_flyers belinda_percentage : ℕ)
variable (scott_flyers : ℕ)

theorem FlyersDistributon (H : total_flyers = 200)
  (H1 : ryan_flyers = 42)
  (H2 : alyssa_flyers = 67)
  (H3 : belinda_percentage = 20)
  (H4 : scott_flyers = total_flyers - (ryan_flyers + alyssa_flyers + (belinda_percentage * total_flyers) / 100)) :
  scott_flyers = 51 :=
by
  simp [H, H1, H2, H3] at H4
  exact H4

end FlyersDistributon_l110_110763


namespace non_swimmers_play_soccer_percentage_l110_110349

theorem non_swimmers_play_soccer_percentage (N : ℕ) (hN_pos : 0 < N)
 (h1 : (0.7 * N : ℝ) = x)
 (h2 : (0.5 * N : ℝ) = y)
 (h3 : (0.6 * x : ℝ) = z)
 : (0.56 * y = 0.28 * N) := 
 sorry

end non_swimmers_play_soccer_percentage_l110_110349


namespace two_pair_probability_l110_110661

theorem two_pair_probability (total_cards : ℕ)
  (ranks : ℕ)
  (cards_per_rank : ℕ)
  (choose_five : ℕ → ℕ → ℕ := Nat.choose)
  (S : ℕ := 13 * (choose_five 4 2) * 12 * (choose_five 4 2) * 11 * (choose_five 4 1))
  (N : ℕ := choose_five 52 5) :
  total_cards = 52 → ranks = 13 → cards_per_rank = 4 → S = 247104 → N = 2598960 →
  (S / N : ℚ) = 95/999 := 
by
  intros
  sorry

end two_pair_probability_l110_110661


namespace right_triangle_integers_solutions_l110_110135

theorem right_triangle_integers_solutions :
  ∃ (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ a^2 + b^2 = c^2 ∧ (a + b + c : ℕ) = (1 / 2 * a * b : ℚ) ∧
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨ (a = 6 ∧ b = 8 ∧ c = 10)) :=
sorry

end right_triangle_integers_solutions_l110_110135


namespace max_average_speed_palindromic_journey_l110_110032

theorem max_average_speed_palindromic_journey
  (initial_odometer : ℕ)
  (final_odometer : ℕ)
  (trip_duration : ℕ)
  (max_speed : ℕ)
  (palindromic : ℕ → Prop)
  (initial_palindrome : palindromic initial_odometer)
  (final_palindrome : palindromic final_odometer)
  (max_speed_constraint : ∀ t, t ≤ trip_duration → t * max_speed ≤ final_odometer - initial_odometer)
  (trip_duration_eq : trip_duration = 5)
  (max_speed_eq : max_speed = 85)
  (initial_odometer_eq : initial_odometer = 69696)
  (final_odometer_max : final_odometer ≤ initial_odometer + max_speed * trip_duration) :
  (max_speed * (final_odometer - initial_odometer) / trip_duration : ℚ) = 82.2 :=
by sorry

end max_average_speed_palindromic_journey_l110_110032


namespace coin_flips_sequences_count_l110_110726

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110726


namespace sum_of_conjugates_l110_110488

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l110_110488


namespace train_crosses_pole_time_l110_110471

theorem train_crosses_pole_time
  (l : ℕ) (v_kmh : ℕ) (v_ms : ℚ) (t : ℕ)
  (h_l : l = 100)
  (h_v_kmh : v_kmh = 180)
  (h_v_ms_conversion : v_ms = v_kmh * 1000 / 3600)
  (h_v_ms : v_ms = 50) :
  t = l / v_ms := by
  sorry

end train_crosses_pole_time_l110_110471


namespace line_intersects_circle_l110_110637

open Real

theorem line_intersects_circle : ∃ (p : ℝ × ℝ), (p.1 + p.2 = 1) ∧ ((p.1 - 1)^2 + (p.2 - 1)^2 = 2) :=
by
  sorry

end line_intersects_circle_l110_110637


namespace triangle_side_length_l110_110264

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l110_110264


namespace domain_of_expression_l110_110786

theorem domain_of_expression : {x : ℝ | ∃ y z : ℝ, y = √(x - 3) ∧ z = √(8 - x) ∧ x - 3 ≥ 0 ∧ 8 - x > 0} = {x : ℝ | 3 ≤ x ∧ x < 8} :=
by
  sorry

end domain_of_expression_l110_110786


namespace value_of_x_plus_y_l110_110568

theorem value_of_x_plus_y (x y : ℚ) (h1 : 1 / x + 1 / y = 5) (h2 : 1 / x - 1 / y = -9) : x + y = -5 / 14 := sorry

end value_of_x_plus_y_l110_110568


namespace area_of_rhombus_l110_110081

theorem area_of_rhombus (x y : ℝ) (d1 d2 : ℝ) (hx : x^2 + y^2 = 130) (hy : d1 = 2 * x) (hz : d2 = 2 * y) (h_diff : abs (d1 - d2) = 4) : 
  4 * 0.5 * x * y = 126 :=
by
  sorry

end area_of_rhombus_l110_110081


namespace beads_taken_out_l110_110311

/--
There is 1 green bead, 2 brown beads, and 3 red beads in a container.
Tom took some beads out of the container and left 4 in.
Prove that Tom took out 2 beads.
-/
theorem beads_taken_out : 
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  initial_beads - beads_left = 2 :=
by
  let green_beads := 1
  let brown_beads := 2
  let red_beads := 3
  let initial_beads := green_beads + brown_beads + red_beads
  let beads_left := 4
  show initial_beads - beads_left = 2
  sorry

end beads_taken_out_l110_110311


namespace union_of_intervals_l110_110374

theorem union_of_intervals :
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  P ∪ Q = {x : ℝ | -2 < x ∧ x < 1} :=
by
  let P := {x : ℝ | -1 < x ∧ x < 1}
  let Q := {x : ℝ | -2 < x ∧ x < 0}
  have h : P ∪ Q = {x : ℝ | -2 < x ∧ x < 1}
  {
     sorry
  }
  exact h

end union_of_intervals_l110_110374


namespace smaller_angle_between_hands_at_3_45_l110_110320

/-
Define the initial conditions to be used in the problem.
-/
def minutes_angle (m : ℕ) : ℝ := m * 6
def hours_angle (h : ℕ) (m : ℕ) : ℝ := h * 30 + (m / 60.0) * 30

/-
State the problem as a Lean theorem statement.
-/
theorem smaller_angle_between_hands_at_3_45 : 
  let minute_hand_angle := minutes_angle 45,
      hour_hand_angle := hours_angle 3 45,
      abs_diff := abs (minute_hand_angle - hour_hand_angle)
  in min abs_diff (360 - abs_diff) = 157.5 :=
begin
  sorry -- Proof to be filled in
end

end smaller_angle_between_hands_at_3_45_l110_110320


namespace coin_flip_sequences_l110_110693

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110693


namespace number_of_solutions_l110_110538

theorem number_of_solutions (n : ℕ) : 
  ∃ solutions : ℕ, 
    (solutions = (n^2 - n + 1)) ∧ 
    (∀ x ∈ Set.Icc (1 : ℝ) n, (x^2 - ⌊x^2⌋ = frac x ^ 2) → (solutions ≠ 0)) :=
sorry

end number_of_solutions_l110_110538


namespace mul_101_101_l110_110040

theorem mul_101_101 : 101 * 101 = 10201 := 
by
  sorry

end mul_101_101_l110_110040


namespace circumference_irrational_l110_110573

theorem circumference_irrational (d : ℚ) : ¬ ∃ (r : ℚ), r = π * d :=
sorry

end circumference_irrational_l110_110573


namespace sum_of_number_and_its_radical_conjugate_l110_110512

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l110_110512


namespace value_of_b_l110_110451

theorem value_of_b :
  (∃ b : ℝ, (1 / Real.log b / Real.log 3 + 1 / Real.log b / Real.log 4 + 1 / Real.log b / Real.log 5 = 1) → b = 60) :=
by
  sorry

end value_of_b_l110_110451


namespace retail_price_l110_110670

theorem retail_price (R : ℝ) (wholesale_price : ℝ)
  (discount_rate : ℝ) (profit_rate : ℝ)
  (selling_price : ℝ) :
  wholesale_price = 81 →
  discount_rate = 0.10 →
  profit_rate = 0.20 →
  selling_price = wholesale_price * (1 + profit_rate) →
  selling_price = R * (1 - discount_rate) →
  R = 108 := 
by 
  intros h_wholesale h_discount h_profit h_selling_price h_discounted_selling_price
  sorry

end retail_price_l110_110670


namespace value_of_expression_l110_110368

theorem value_of_expression 
  (x : ℝ) 
  (h : 7 * x^2 + 6 = 5 * x + 11) 
  : (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := 
by 
  sorry

end value_of_expression_l110_110368


namespace clock_angle_at_3_45_l110_110317

/-- The degree measure of the smaller angle between the hour hand and the minute hand at 3:45 p.m. on a 12-hour analog clock is 157.5 degrees. -/
theorem clock_angle_at_3_45 : 
  ∃ θ : ℝ, θ = 157.5 ∧ 
    (∀ h m : ℝ, h = 3 + 0.75 ∧ m = 9 → 
     let hour_angle := h * 30,
         minute_angle := m * 6 in 
         let diff := abs (minute_angle - hour_angle) in
         θ = min diff (360 - diff)) :=
sorry

end clock_angle_at_3_45_l110_110317


namespace ratio_mets_to_redsox_l110_110576

theorem ratio_mets_to_redsox (Y M R : ℕ)
  (h1 : Y / M = 3 / 2)
  (h2 : M = 96)
  (h3 : Y + M + R = 360) :
  M / R = 4 / 5 :=
by sorry

end ratio_mets_to_redsox_l110_110576


namespace last_three_digits_of_8_pow_1000_l110_110210

theorem last_three_digits_of_8_pow_1000 (h : 8 ^ 125 ≡ 2 [MOD 1250]) : (8 ^ 1000) % 1000 = 256 :=
by
  sorry

end last_three_digits_of_8_pow_1000_l110_110210


namespace find_correct_fraction_l110_110104

theorem find_correct_fraction
  (mistake_frac : ℚ) (n : ℕ) (delta : ℚ)
  (correct_frac : ℚ) (number : ℕ)
  (h1 : mistake_frac = 5 / 6)
  (h2 : number = 288)
  (h3 : mistake_frac * number = correct_frac * number + delta)
  (h4 : delta = 150) :
  correct_frac = 5 / 32 :=
by
  sorry

end find_correct_fraction_l110_110104


namespace sum_of_three_numbers_l110_110453

theorem sum_of_three_numbers (a b c : ℕ)
    (h1 : a + b = 35)
    (h2 : b + c = 40)
    (h3 : c + a = 45) :
    a + b + c = 60 := 
  by sorry

end sum_of_three_numbers_l110_110453


namespace find_y_from_exponent_equation_l110_110809

theorem find_y_from_exponent_equation (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := sorry

end find_y_from_exponent_equation_l110_110809


namespace math_students_count_l110_110398

noncomputable def students_in_math (total_students history_students english_students all_three_classes two_classes : ℕ) : ℕ :=
total_students - history_students - english_students + (two_classes - all_three_classes)

theorem math_students_count :
  students_in_math 68 21 34 3 7 = 14 :=
by
  sorry

end math_students_count_l110_110398


namespace jason_fish_count_ninth_day_l110_110974

def fish_growth_day1 := 8 * 3
def fish_growth_day2 := fish_growth_day1 * 3
def fish_growth_day3 := fish_growth_day2 * 3
def fish_day4_removed := 2 / 5 * fish_growth_day3
def fish_after_day4 := fish_growth_day3 - fish_day4_removed
def fish_growth_day5 := fish_after_day4 * 3
def fish_growth_day6 := fish_growth_day5 * 3
def fish_day6_removed := 3 / 7 * fish_growth_day6
def fish_after_day6 := fish_growth_day6 - fish_day6_removed
def fish_growth_day7 := fish_after_day6 * 3
def fish_growth_day8 := fish_growth_day7 * 3
def fish_growth_day9 := fish_growth_day8 * 3
def fish_final := fish_growth_day9 + 20

theorem jason_fish_count_ninth_day : fish_final = 18083 :=
by
  -- proof steps will go here
  sorry

end jason_fish_count_ninth_day_l110_110974


namespace fg_of_2_l110_110203

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem fg_of_2 : f (g 2) = 10 :=
by
  sorry

end fg_of_2_l110_110203


namespace arithmetic_sequence_fifth_term_l110_110944

theorem arithmetic_sequence_fifth_term (a1 d : ℕ) (a_n : ℕ → ℕ) 
  (h_a1 : a1 = 2) (h_d : d = 1) (h_a_n : ∀ n : ℕ, a_n n = a1 + (n-1) * d) : 
  a_n 5 = 6 := 
    by
    -- Given the conditions, we need to prove a_n evaluated at 5 is equal to 6.
    sorry

end arithmetic_sequence_fifth_term_l110_110944


namespace Davey_Barbeck_ratio_is_1_l110_110183

-- Assume the following given conditions as definitions in Lean
variables (guitars Davey Barbeck : ℕ)

-- Condition 1: Davey has 18 guitars
def Davey_has_18 : Prop := Davey = 18

-- Condition 2: Barbeck has the same number of guitars as Davey
def Davey_eq_Barbeck : Prop := Davey = Barbeck

-- The problem statement: Prove the ratio of the number of guitars Davey has to the number of guitars Barbeck has is 1:1
theorem Davey_Barbeck_ratio_is_1 (h1 : Davey_has_18 Davey) (h2 : Davey_eq_Barbeck Davey Barbeck) :
  Davey / Barbeck = 1 :=
by
  sorry

end Davey_Barbeck_ratio_is_1_l110_110183


namespace mean_score_l110_110537

theorem mean_score (M SD : ℝ) (h1 : 58 = M - 2 * SD) (h2 : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l110_110537


namespace train_ride_length_l110_110989

noncomputable def totalMinutesUntil0900 (leaveTime : Nat) (arrivalTime : Nat) : Nat :=
  arrivalTime - leaveTime

noncomputable def walkTime : Nat := 10

noncomputable def rideTime (totalTime : Nat) (walkTime : Nat) : Nat :=
  totalTime - walkTime

theorem train_ride_length (leaveTime : Nat) (arrivalTime : Nat) :
  leaveTime = 450 → arrivalTime = 540 → rideTime (totalMinutesUntil0900 leaveTime arrivalTime) walkTime = 80 :=
by
  intros h_leaveTime h_arrivalTime
  rw [h_leaveTime, h_arrivalTime]
  unfold totalMinutesUntil0900
  unfold rideTime
  unfold walkTime
  sorry

end train_ride_length_l110_110989


namespace find_y_l110_110807

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110807


namespace last_three_digits_of_7_pow_83_l110_110196

theorem last_three_digits_of_7_pow_83 :
  (7 ^ 83) % 1000 = 886 := sorry

end last_three_digits_of_7_pow_83_l110_110196


namespace seating_arrangements_l110_110054

theorem seating_arrangements (n m k : Nat) (couples : Fin n -> Fin m -> Prop):
  let pairs : Nat := k
  let adjusted_pairs : Nat := pairs / 24
  adjusted_pairs = 5760 := by
  sorry

end seating_arrangements_l110_110054


namespace percent_of_a_is_4b_l110_110430

variables (a b : ℝ)
theorem percent_of_a_is_4b (h : a = 2 * b) : 4 * b / a = 2 :=
by 
  sorry

end percent_of_a_is_4b_l110_110430


namespace p_p_eq_twenty_l110_110121

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x ≥ 0 ∧ y < 0 then 4 * x + 2 * y
  else 3 * x + 2 * y

theorem p_p_eq_twenty : p (p 2 (-3)) (p (-3) (-4)) = 20 :=
by
  sorry

end p_p_eq_twenty_l110_110121


namespace evens_minus_odds_equal_40_l110_110673

-- Define the sum of even integers from 2 to 80
def sum_evens : ℕ := (List.range' 2 40).sum

-- Define the sum of odd integers from 1 to 79
def sum_odds : ℕ := (List.range' 1 40).sum

-- Define the main theorem to prove
theorem evens_minus_odds_equal_40 : sum_evens - sum_odds = 40 := by
  -- Proof will go here
  sorry

end evens_minus_odds_equal_40_l110_110673


namespace toy_playing_dogs_ratio_l110_110308

theorem toy_playing_dogs_ratio
  (d_t : ℕ) (d_r : ℕ) (d_n : ℕ) (d_b : ℕ) (d_p : ℕ)
  (h1 : d_t = 88)
  (h2 : d_r = 12)
  (h3 : d_n = 10)
  (h4 : d_b = d_t / 4)
  (h5 : d_p = d_t - d_r - d_b - d_n) :
  d_p / d_t = 1 / 2 :=
by sorry

end toy_playing_dogs_ratio_l110_110308


namespace february_max_diff_percentage_l110_110821

noncomputable def max_diff_percentage (D B F : ℕ) : ℚ :=
  let avg_others := (B + F) / 2
  let high_sales := max (max D B) F
  (high_sales - avg_others) / avg_others * 100

theorem february_max_diff_percentage :
  max_diff_percentage 8 5 6 = 45.45 := by
  sorry

end february_max_diff_percentage_l110_110821


namespace least_integer_l110_110931

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l110_110931


namespace part1_part2_part3_l110_110401

open Real

-- Definitions of points
structure Point :=
(x : ℝ)
(y : ℝ)

def M (m : ℝ) : Point := ⟨m - 2, 2 * m - 7⟩
def N (n : ℝ) : Point := ⟨n, 3⟩

-- Part 1
theorem part1 : 
  (M (7 / 2)).y = 0 ∧ (M (7 / 2)).x = 3 / 2 :=
by
  sorry

-- Part 2
theorem part2 (m : ℝ) : abs (m - 2) = abs (2 * m - 7) → (m = 5 ∨ m = 3) :=
by
  sorry

-- Part 3
theorem part3 (m n : ℝ) : abs ((M m).y - 3) = 2 ∧ (M m).x = n - 2 → (n = 4 ∨ n = 2) :=
by
  sorry

end part1_part2_part3_l110_110401


namespace amount_spent_on_shirt_l110_110615

-- Definitions and conditions
def total_spent_clothing : ℝ := 25.31
def spent_on_jacket : ℝ := 12.27

-- Goal: Prove the amount spent on the shirt is 13.04
theorem amount_spent_on_shirt : (total_spent_clothing - spent_on_jacket = 13.04) := by
  sorry

end amount_spent_on_shirt_l110_110615


namespace find_y_l110_110193

def vectors_orthogonal_condition (y : ℝ) : Prop :=
  (1 * -2) + (-3 * y) + (-4 * -1) = 0

theorem find_y : vectors_orthogonal_condition (2 / 3) :=
by
  sorry

end find_y_l110_110193


namespace max_mass_of_grain_l110_110466

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end max_mass_of_grain_l110_110466


namespace largest_integer_not_greater_than_expr_l110_110560

theorem largest_integer_not_greater_than_expr (x : ℝ) (hx : 20 * Real.sin x = 22 * Real.cos x) :
    ⌊(1 / (Real.sin x * Real.cos x) - 1)^7⌋ = 1 := 
sorry

end largest_integer_not_greater_than_expr_l110_110560


namespace perpendicular_lines_a_l110_110873

theorem perpendicular_lines_a (a : ℝ) :
  (∀ x y : ℝ, (a * x + (1 + a) * y = 3) ∧ ((a + 1) * x + (3 - 2 * a) * y = 2) → 
     a = -1 ∨ a = 3) :=
by
  sorry

end perpendicular_lines_a_l110_110873


namespace sequence_x_sequence_y_sequence_z_sequence_t_l110_110617

theorem sequence_x (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^2 + n = 2) else 
   if n = 2 then (n^2 + n = 6) else 
   if n = 3 then (n^2 + n = 12) else 
   if n = 4 then (n^2 + n = 20) else true) := 
by sorry

theorem sequence_y (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2 * n^2 = 2) else 
   if n = 2 then (2 * n^2 = 8) else 
   if n = 3 then (2 * n^2 = 18) else 
   if n = 4 then (2 * n^2 = 32) else true) := 
by sorry

theorem sequence_z (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (n^3 = 1) else 
   if n = 2 then (n^3 = 8) else 
   if n = 3 then (n^3 = 27) else 
   if n = 4 then (n^3 = 64) else true) := 
by sorry

theorem sequence_t (n : ℕ) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 → 
  (if n = 1 then (2^n = 2) else 
   if n = 2 then (2^n = 4) else 
   if n = 3 then (2^n = 8) else 
   if n = 4 then (2^n = 16) else true) := 
by sorry

end sequence_x_sequence_y_sequence_z_sequence_t_l110_110617


namespace shooting_challenge_sequences_l110_110577

theorem shooting_challenge_sequences : ∀ (A B C : ℕ), 
  A = 4 → B = 4 → C = 2 →
  (A + B + C = 10) →
  (Nat.factorial (A + B + C) / (Nat.factorial A * Nat.factorial B * Nat.factorial C) = 3150) :=
by
  intros A B C hA hB hC hsum
  sorry

end shooting_challenge_sequences_l110_110577


namespace vector_addition_l110_110545

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (5, -7, 8)

-- State the theorem to prove 2a + b = (7, -3, 2)
theorem vector_addition : (2 • a + b) = (7, -3, 2) := by
  sorry

end vector_addition_l110_110545


namespace find_b_l110_110256

-- Definition of the geometric problem
variables {a b c : ℝ} -- Side lengths of the triangle
variables {area : ℝ} -- Area of the triangle
variables {B : ℝ} -- Angle B in radians

-- Given conditions
def triangle_conditions : Prop :=
  area = sqrt 3 ∧
  B = π / 3 ∧
  a^2 + c^2 = 3 * a * c

-- Statement of the theorem using the given conditions to prove b = 2√2
theorem find_b (h : triangle_conditions) : b = 2 * sqrt 2 := 
  sorry

end find_b_l110_110256


namespace polynomial_difference_of_squares_l110_110291

theorem polynomial_difference_of_squares (x y : ℤ) :
  8 * x^2 + 2 * x * y - 3 * y^2 = (3 * x - y)^2 - (x + 2 * y)^2 :=
by
  sorry

end polynomial_difference_of_squares_l110_110291


namespace jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l110_110125

/-- Conditions:
1. The grandmother decorates five gingerbread cookies for every cycle.
2. Little Mary decorates three gingerbread cookies for every cycle.
3. Little John decorates two gingerbread cookies for every cycle.
4. All three together decorated five trays, with each tray holding twelve gingerbread cookies.
5. Little John also sorted the gingerbread cookies onto trays twelve at a time and carried them to the pantry.
6. The grandmother decorates one gingerbread cookie in four minutes.
-/

def decorated_cookies_per_cycle := 10
def total_trays := 5
def cookies_per_tray := 12
def total_cookies := total_trays * cookies_per_tray
def babicka_cookies_per_cycle := 5
def marenka_cookies_per_cycle := 3
def jenicek_cookies_per_cycle := 2
def babicka_time_per_cookie := 4

theorem jenicek_decorated_cookies :
  (total_cookies - (total_cookies / decorated_cookies_per_cycle * marenka_cookies_per_cycle + total_cookies / decorated_cookies_per_cycle * babicka_cookies_per_cycle)) = 4 :=
sorry

theorem total_time_for_work :
  (total_cookies / decorated_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 140 :=
sorry

theorem jenicek_decorating_time :
  (4 / jenicek_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 40 :=
sorry

end jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l110_110125


namespace remainder_when_sum_is_divided_l110_110814

theorem remainder_when_sum_is_divided (n : ℤ) : ((8 - n) + (n + 5)) % 9 = 4 := by
  sorry

end remainder_when_sum_is_divided_l110_110814


namespace Shaina_chocolate_l110_110595

-- Definitions based on the conditions
def total_chocolate : ℚ := 72 / 7
def number_of_piles : ℚ := 6
def weight_per_pile : ℚ := total_chocolate / number_of_piles
def piles_given_to_Shaina : ℚ := 2

-- Theorem stating the problem's correct answer
theorem Shaina_chocolate :
  piles_given_to_Shaina * weight_per_pile = 24 / 7 :=
by
  sorry

end Shaina_chocolate_l110_110595


namespace hyperbola_equation_l110_110918

-- Definitions based on the conditions:
def hyperbola (x y a b : ℝ) : Prop := (y^2 / a^2) - (x^2 / b^2) = 1

def point_on_hyperbola (a b : ℝ) : Prop := hyperbola 2 (-2) a b

def asymptotes (a b : ℝ) : Prop := a / b = (Real.sqrt 2) / 2

-- Prove the equation of the hyperbola
theorem hyperbola_equation :
  ∃ a b, a = Real.sqrt 2 ∧ b = 2 ∧ hyperbola y x (Real.sqrt 2) 2 :=
by
  -- Placeholder for the actual proof
  sorry

end hyperbola_equation_l110_110918


namespace find_a11_l110_110415

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom cond1 : ∀ n : ℕ, n > 0 → 4 * S n = 2 * a n - n^2 + 7 * n

-- Theorem stating the proof problem
theorem find_a11 :
  a 11 = -2 :=
sorry

end find_a11_l110_110415


namespace percentage_error_x_percentage_error_y_l110_110469

theorem percentage_error_x (x : ℝ) : 
  let correct_result := x * 10
  let erroneous_result := x / 10
  (correct_result - erroneous_result) / correct_result * 100 = 99 :=
by
  sorry

theorem percentage_error_y (y : ℝ) : 
  let correct_result := y + 15
  let erroneous_result := y - 15
  (correct_result - erroneous_result) / correct_result * 100 = (30 / (y + 15)) * 100 :=
by
  sorry

end percentage_error_x_percentage_error_y_l110_110469


namespace maximum_value_of_expression_l110_110836

variable (x y z : ℝ)

theorem maximum_value_of_expression (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) : 
  x + y^3 + z^4 ≤ 1 :=
sorry

end maximum_value_of_expression_l110_110836


namespace smallest_integer_condition_l110_110921

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l110_110921


namespace find_value_of_k_l110_110037

theorem find_value_of_k (k : ℤ) : 
  (2 + 3 * k * -1/3 = -7 * 4) → k = 30 := 
by
  sorry

end find_value_of_k_l110_110037


namespace card_average_value_l110_110895

theorem card_average_value (n : ℕ) (h : (2 * n + 1) / 3 = 2023) : n = 3034 :=
sorry

end card_average_value_l110_110895


namespace find_TU2_l110_110584

-- Define the structure of the square, distances, and points
structure square (P Q R S T U : Type) :=
(PQ : ℝ)
(PT QU QT RU TU2 : ℝ)
(h1 : PQ = 15)
(h2 : PT = 7)
(h3 : QU = 7)
(h4 : QT = 17)
(h5 : RU = 17)
(h6 : TU2 = TU^2)
(h7 : TU2 = 1073)

-- The main proof statement
theorem find_TU2 {P Q R S T U : Type} (sq : square P Q R S T U) : sq.TU2 = 1073 := by
  sorry

end find_TU2_l110_110584


namespace num_sets_M_l110_110857

theorem num_sets_M (M : Set ℕ) :
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5, 6} → ∃ n : Nat, n = 16 :=
by
  sorry

end num_sets_M_l110_110857


namespace faye_complete_bouquets_l110_110529

theorem faye_complete_bouquets :
  let roses_initial := 48
  let lilies_initial := 40
  let tulips_initial := 76
  let sunflowers_initial := 34
  let roses_wilted := 24
  let lilies_wilted := 10
  let tulips_wilted := 14
  let sunflowers_wilted := 7
  let roses_remaining := roses_initial - roses_wilted
  let lilies_remaining := lilies_initial - lilies_wilted
  let tulips_remaining := tulips_initial - tulips_wilted
  let sunflowers_remaining := sunflowers_initial - sunflowers_wilted
  let bouquets_roses := roses_remaining / 2
  let bouquets_lilies := lilies_remaining
  let bouquets_tulips := tulips_remaining / 3
  let bouquets_sunflowers := sunflowers_remaining
  let bouquets := min (min bouquets_roses bouquets_lilies) (min bouquets_tulips bouquets_sunflowers)
  bouquets = 12 :=
by
  sorry

end faye_complete_bouquets_l110_110529


namespace least_positive_integer_l110_110924

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l110_110924


namespace evaluate_expression_l110_110528

theorem evaluate_expression : ∀ (a b c d : ℤ), 
  a = 3 →
  b = a + 3 →
  c = b - 8 →
  d = a + 5 →
  (a + 2 ≠ 0) →
  (b - 4 ≠ 0) →
  (c + 5 ≠ 0) →
  (d - 3 ≠ 0) →
  ((a + 3) * (b - 2) * (c + 9) * (d + 1) = 1512 * (a + 2) * (b - 4) * (c + 5) * (d - 3)) :=
by
  intros a b c d ha hb hc hd ha2 hb4 hc5 hd3
  sorry

end evaluate_expression_l110_110528


namespace perimeter_of_figure_is_correct_l110_110630

-- Define the conditions as Lean variables and constants
def area_of_figure : ℝ := 144
def number_of_squares : ℕ := 4

-- Define the question as a theorem to be proven in Lean
theorem perimeter_of_figure_is_correct :
  let area_of_square := area_of_figure / number_of_squares
  let side_length := Real.sqrt area_of_square
  let perimeter := 9 * side_length
  perimeter = 54 :=
by
  intro area_of_square
  intro side_length
  intro perimeter
  sorry

end perimeter_of_figure_is_correct_l110_110630


namespace average_rate_of_interest_l110_110176

/-- Given:
    1. A woman has a total of $7500 invested,
    2. Part of the investment is at 5% interest,
    3. The remainder of the investment is at 7% interest,
    4. The annual returns from both investments are equal,
    Prove:
    The average rate of interest realized on her total investment is 5.8%.
-/
theorem average_rate_of_interest
  (total_investment : ℝ) (interest_5_percent : ℝ) (interest_7_percent : ℝ)
  (annual_return_equal : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent)
  (total_investment_eq : total_investment = 7500) : 
  (interest_5_percent / total_investment) = 0.058 :=
by
  -- conditions given
  have h1 : total_investment = 7500 := total_investment_eq
  have h2 : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent := annual_return_equal

  -- final step, sorry is used to skip the proof
  sorry

end average_rate_of_interest_l110_110176


namespace Kolya_made_the_mistake_l110_110653

def pencils_in_box (blue green : ℕ) : Prop :=
  (blue ≥ 4 ∨ blue < 4) ∧ (green ≥ 4 ∨ green < 4)

def boys_statements (blue green : ℕ) : Prop :=
  (Vasya : blue ≥ 4) ∧
  (Kolya : green ≥ 5) ∧
  (Petya : blue ≥ 3 ∧ green ≥ 4) ∧
  (Misha : blue ≥ 4 ∧ green ≥ 4)

theorem Kolya_made_the_mistake:
  ∀ {blue green : ℕ},
  pencils_in_box blue green →
  boys_statements blue green →
  ∃ (Vasya_truth Petya_truth Misha_truth : Prop),
  Vasya_truth ∧ Petya_truth ∧ Misha_truth ∧ ¬ Kolya_truth :=
begin
  sorry
end

end Kolya_made_the_mistake_l110_110653


namespace karina_brother_birth_year_l110_110116

theorem karina_brother_birth_year :
  ∀ (karina_birth_year karina_age_brother_ratio karina_current_age current_year brother_age birth_year: ℤ),
    karina_birth_year = 1970 →
    karina_age_brother_ratio = 2 →
    karina_current_age = 40 →
    current_year = karina_birth_year + karina_current_age →
    brother_age = karina_current_age / karina_age_brother_ratio →
    birth_year = current_year - brother_age →
    birth_year = 1990 :=
by
  intros karina_birth_year karina_age_brother_ratio karina_current_age current_year brother_age birth_year
  assume h1 : karina_birth_year = 1970
  assume h2 : karina_age_brother_ratio = 2
  assume h3 : karina_current_age = 40
  assume h4 : current_year = karina_birth_year + karina_current_age
  assume h5 : brother_age = karina_current_age / karina_age_brother_ratio
  assume h6 : birth_year = current_year - brother_age
  sorry

end karina_brother_birth_year_l110_110116


namespace factor_expression_l110_110779

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110779


namespace current_intensity_leq_3_2_l110_110890

theorem current_intensity_leq_3_2 (P : ℝ) (U : ℝ) (R : ℝ) (hP : P = 800) (hU : U = 200) (hR : R ≥ 62.5) :
  (U / R) ≤ 3.2 :=
by
  have hI : (U / R) = 200 / R := by
    rw [hU]
  have hineq : 200 / R ≤ 200 / 62.5 := by
    apply (div_le_div_left _ _ _).mpr; linarith
  rw [hI] at hineq
  norm_num at hineq
  exact hineq

end current_intensity_leq_3_2_l110_110890


namespace positive_expression_l110_110996

theorem positive_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 * (b + c) + a * (b^2 + c^2 - b * c) > 0 :=
by sorry

end positive_expression_l110_110996


namespace coin_flip_sequences_l110_110683

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110683


namespace fraction_decimal_representation_l110_110771

noncomputable def fraction_as_term_dec : ℚ := 47 / (2^3 * 5^4)

theorem fraction_decimal_representation : fraction_as_term_dec = 0.0094 :=
by
  sorry

end fraction_decimal_representation_l110_110771


namespace sum_of_conjugates_l110_110492

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l110_110492


namespace total_books_l110_110769

-- Defining the conditions
def darla_books := 6
def katie_books := darla_books / 2
def combined_books := darla_books + katie_books
def gary_books := 5 * combined_books

-- Statement to prove
theorem total_books : darla_books + katie_books + gary_books = 54 := by
  sorry

end total_books_l110_110769


namespace intersections_correct_l110_110225

-- Define the distances (in meters)
def gretzky_street_length : ℕ := 5600
def segment_a_distance : ℕ := 350
def segment_b_distance : ℕ := 400
def segment_c_distance : ℕ := 450

-- Definitions based on conditions
def segment_a_intersections : ℕ :=
  gretzky_street_length / segment_a_distance - 2 -- subtract Orr Street and Howe Street

def segment_b_intersections : ℕ :=
  gretzky_street_length / segment_b_distance

def segment_c_intersections : ℕ :=
  gretzky_street_length / segment_c_distance

-- Sum of all intersections
def total_intersections : ℕ :=
  segment_a_intersections + segment_b_intersections + segment_c_intersections

theorem intersections_correct :
  total_intersections = 40 :=
by
  sorry

end intersections_correct_l110_110225


namespace fraction_sum_l110_110351

theorem fraction_sum : (3 / 8) + (9 / 12) = 9 / 8 :=
by
  sorry

end fraction_sum_l110_110351


namespace altitudes_sum_of_triangle_formed_by_line_and_axes_l110_110767

noncomputable def sum_of_altitudes (x y : ℝ) : ℝ :=
  let intercept_x := 6
  let intercept_y := 16
  let altitude_3 := 48 / Real.sqrt (8^2 + 3^2)
  intercept_x + intercept_y + altitude_3

theorem altitudes_sum_of_triangle_formed_by_line_and_axes :
  ∀ (x y : ℝ), (8 * x + 3 * y = 48) →
  sum_of_altitudes x y = 22 + 48 / Real.sqrt 73 :=
by
  sorry

end altitudes_sum_of_triangle_formed_by_line_and_axes_l110_110767


namespace negation_exists_equation_l110_110618

theorem negation_exists_equation (P : ℝ → Prop) :
  (∃ x > 0, x^2 + 3 * x - 5 = 0) → ¬ (∃ x > 0, x^2 + 3 * x - 5 = 0) = ∀ x > 0, x^2 + 3 * x - 5 ≠ 0 :=
by sorry

end negation_exists_equation_l110_110618


namespace num_more_green_l110_110679

noncomputable def num_people : ℕ := 150
noncomputable def more_blue : ℕ := 90
noncomputable def both_green_blue : ℕ := 40
noncomputable def neither_green_blue : ℕ := 20

theorem num_more_green :
  (num_people + more_blue + both_green_blue + neither_green_blue) ≤ 150 →
  (more_blue - both_green_blue) + both_green_blue + neither_green_blue ≤ num_people →
  (num_people - 
  ((more_blue - both_green_blue) + both_green_blue + neither_green_blue)) + both_green_blue = 80 :=
by
    intros h1 h2
    sorry

end num_more_green_l110_110679


namespace loss_percentage_l110_110343

theorem loss_percentage (CP SP : ℝ) (h_CP : CP = 1300) (h_SP : SP = 1040) :
  ((CP - SP) / CP) * 100 = 20 :=
by
  sorry

end loss_percentage_l110_110343


namespace common_difference_of_arithmetic_sequence_l110_110204

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : a 5 = 10) (h3 : a 10 = -5) : d = -3 := 
by 
  sorry

end common_difference_of_arithmetic_sequence_l110_110204


namespace mairiad_distance_ratio_l110_110326

open Nat

theorem mairiad_distance_ratio :
  ∀ (x : ℕ),
  let miles_run := 40
  let miles_walked := 3 * miles_run / 5
  let total_distance := miles_run + miles_walked + x * miles_run
  total_distance = 184 →
  24 + x * 40 = 144 →
  (24 + 3 * 40) / 40 = 3.6 := 
sorry

end mairiad_distance_ratio_l110_110326


namespace complete_square_h_l110_110245

theorem complete_square_h (x h : ℝ) :
  (∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k) → h = -3 / 2 :=
by
  sorry

end complete_square_h_l110_110245


namespace next_wednesday_l110_110418
open Nat

/-- Prove that the next year after 2010 when April 16 falls on a Wednesday is 2014,
    given the conditions:
    1. 2010 is a non-leap year.
    2. The day advances by 1 day for a non-leap year and 2 days for a leap year.
    3. April 16, 2010 was a Friday. -/
theorem next_wednesday (initial_year : ℕ) (initial_day : String) (target_day : String) : 
  (initial_year = 2010) ∧
  (initial_day = "Friday") ∧ 
  (target_day = "Wednesday") →
  2014 = 2010 + 4 :=
by
  sorry

end next_wednesday_l110_110418


namespace no_solution_eqn_l110_110096

theorem no_solution_eqn (m : ℝ) : (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) ↔ m = 6 := 
by
  sorry

end no_solution_eqn_l110_110096


namespace c_values_for_one_vertical_asymptote_l110_110053

noncomputable def has_one_vertical_asymptote (c : ℝ) : Prop :=
  let numerator := (X^2 + 3 * X + C : Polynomial ℝ)
  let denominator := (X^2 - 3 * X - 10 : Polynomial ℝ)
  (numerator.eval 5 = 0 ∧ numerator.eval (-2) ≠ 0) ∨ 
  (numerator.eval (-2) = 0 ∧ numerator.eval 5 ≠ 0)

theorem c_values_for_one_vertical_asymptote :
  ∀ c : ℝ, has_one_vertical_asymptote c ↔ c = -40 ∨ c = 2 :=
by
  -- Proof goes here (sorry leaves the proof incomplete intentionally)
  sorry

end c_values_for_one_vertical_asymptote_l110_110053


namespace work_rate_b_l110_110155

theorem work_rate_b (A C B : ℝ) (hA : A = 1 / 8) (hC : C = 1 / 24) (h_combined : A + B + C = 1 / 4) : B = 1 / 12 :=
by
  -- Proof goes here
  sorry

end work_rate_b_l110_110155


namespace intersection_M_N_eq_02_l110_110561

open Set

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = 2 * x}

theorem intersection_M_N_eq_02 : M ∩ N = {0, 2} := 
by sorry

end intersection_M_N_eq_02_l110_110561


namespace inequality_solution_l110_110429

theorem inequality_solution :
  { x : ℝ | (x^3 - 4 * x) / (x^2 - 1) > 0 } = { x : ℝ | x < -2 ∨ (0 < x ∧ x < 1) ∨ 2 < x } :=
by
  sorry

end inequality_solution_l110_110429


namespace binom_difference_30_3_2_l110_110028

-- Define the binomial coefficient function.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: binom(30, 3) - binom(30, 2) = 3625
theorem binom_difference_30_3_2 : binom 30 3 - binom 30 2 = 3625 := by
  sorry

end binom_difference_30_3_2_l110_110028


namespace numerical_identity_l110_110426

theorem numerical_identity :
  1.2008 * 0.2008 * 2.4016 - 1.2008^3 - 1.2008 * 0.2008^2 = -1.2008 :=
by
  -- conditions and definitions based on a) are directly used here
  sorry -- proof is not required as per instructions

end numerical_identity_l110_110426


namespace trigonometric_inequality_l110_110880

theorem trigonometric_inequality (x : Real) (n : Int) :
  (9.286 * (Real.sin x)^3 * Real.sin ((Real.pi / 2) - 3 * x) +
   (Real.cos x)^3 * Real.cos ((Real.pi / 2) - 3 * x) > 
   3 * Real.sqrt 3 / 8) →
   (x > (Real.pi / 12) + (Real.pi * n / 2) ∧
   x < (5 * Real.pi / 12) + (Real.pi * n / 2)) :=
by
  sorry

end trigonometric_inequality_l110_110880


namespace percent_yz_of_x_l110_110234

theorem percent_yz_of_x (x y z : ℝ) 
  (h₁ : 0.6 * (x - y) = 0.3 * (x + y))
  (h₂ : 0.4 * (x + z) = 0.2 * (y + z))
  (h₃ : 0.5 * (x - z) = 0.25 * (x + y + z)) :
  y + z = 0.0 * x :=
sorry

end percent_yz_of_x_l110_110234


namespace find_maximum_value_of_f_φ_has_root_l110_110219

open Set Real

noncomputable section

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := -6 * (sin x + cos x) - 3

-- Definition of the function φ(x)
def φ (x : ℝ) : ℝ := f x + 10

-- The assumptions on the interval
def interval := Icc 0 (π / 4)

-- Statement to prove that the maximum value of f(x) is -9
theorem find_maximum_value_of_f : ∀ x ∈ interval, f x ≤ -9 ∧ ∃ x_0 ∈ interval, f x_0 = -9 := sorry

-- Statement to prove that φ(x) has a root in the interval
theorem φ_has_root : ∃ x ∈ interval, φ x = 0 := sorry

end find_maximum_value_of_f_φ_has_root_l110_110219


namespace compare_y_values_l110_110209

-- Define the quadratic function y = x^2 + 2x + c
def quadratic (x : ℝ) (c : ℝ) : ℝ := x^2 + 2 * x + c

-- Points A, B, and C on the quadratic function
variables 
  (c : ℝ) 
  (y1 y2 y3 : ℝ) 
  (hA : y1 = quadratic (-3) c) 
  (hB : y2 = quadratic (-2) c) 
  (hC : y3 = quadratic 2 c)

theorem compare_y_values :
  y3 > y1 ∧ y1 > y2 :=
by sorry

end compare_y_values_l110_110209


namespace proportion_of_line_segments_l110_110571

theorem proportion_of_line_segments (a b c d : ℕ)
  (h_proportion : a * d = b * c)
  (h_a : a = 2)
  (h_b : b = 4)
  (h_c : c = 3) :
  d = 6 :=
by
  sorry

end proportion_of_line_segments_l110_110571


namespace erased_length_l110_110620

def original_length := 100 -- in cm
def final_length := 76 -- in cm

theorem erased_length : original_length - final_length = 24 :=
by
    sorry

end erased_length_l110_110620


namespace sum_a_1_a_12_sum_b_1_b_2n_l110_110795

noncomputable def f_n (n m : ℕ) : ℚ :=
  if h : m = 0 then 1
  else (List.prod (List.map (λ k, (n - k : ℚ)) (List.range m))) / (Nat.factorial m : ℚ)

def a (m : ℕ) : ℚ := f_n 6 m

def b (n m : ℕ) : ℚ := (-1) ^ m * m * f_n n m

theorem sum_a_1_a_12 : (Finset.range 12).sum (λ m, a (m+1)) = 63 := sorry

theorem sum_b_1_b_2n (n : ℕ) : (Finset.range (2 * n)).sum (λ m, b n (m+1)) ∈ { -1, 0 } := sorry

end sum_a_1_a_12_sum_b_1_b_2n_l110_110795


namespace factorization_correct_l110_110434

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

end factorization_correct_l110_110434


namespace shortest_path_Dasha_Vasya_l110_110403

-- Definitions for the given distances
def dist_Asya_Galia : ℕ := 12
def dist_Galia_Borya : ℕ := 10
def dist_Asya_Borya : ℕ := 8
def dist_Dasha_Galia : ℕ := 15
def dist_Vasya_Galia : ℕ := 17

-- Definition for shortest distance by roads from Dasha to Vasya
def shortest_dist_Dasha_Vasya : ℕ := 18

-- Proof statement of the goal that shortest distance from Dasha to Vasya is 18 km
theorem shortest_path_Dasha_Vasya : 
  dist_Dasha_Galia + dist_Vasya_Galia - dist_Asya_Galia - dist_Galia_Borya = shortest_dist_Dasha_Vasya := by
  sorry

end shortest_path_Dasha_Vasya_l110_110403


namespace coin_flip_sequences_l110_110708

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110708


namespace find_possible_f_one_l110_110853

noncomputable def f : ℝ → ℝ := sorry

theorem find_possible_f_one (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
  f 1 = 0 ∨ (∃ c : ℝ, f 0 = 1/2 ∧ f 1 = c) :=
sorry

end find_possible_f_one_l110_110853


namespace find_missing_number_l110_110295

theorem find_missing_number (x : ℕ) : (4 + 3) + (8 - 3 - x) = 11 → x = 1 :=
by
  sorry

end find_missing_number_l110_110295


namespace angle_half_second_quadrant_l110_110566

theorem angle_half_second_quadrant (α : ℝ) (k : ℤ) :
  (π / 2 + 2 * k * π < α ∧ α < π + 2 * k * π) → 
  (∃ m : ℤ, (π / 4 + m * π < α / 2 ∧ α / 2 < π / 2 + m * π)) ∨ 
  (∃ n : ℤ, (5 * π / 4 + n * π < α / 2 ∧ α / 2 < 3 * π / 2 + n * π)) :=
by
  sorry

end angle_half_second_quadrant_l110_110566


namespace coin_flip_sequences_l110_110738

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110738


namespace call_charge_ratio_l110_110914

def elvin_jan_total_bill : ℕ := 46
def elvin_feb_total_bill : ℕ := 76
def elvin_internet_charge : ℕ := 16
def elvin_call_charge_ratio : ℕ := 2

theorem call_charge_ratio : 
  (elvin_feb_total_bill - elvin_internet_charge) / (elvin_jan_total_bill - elvin_internet_charge) = elvin_call_charge_ratio := 
by
  sorry

end call_charge_ratio_l110_110914


namespace total_young_fish_l110_110611

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l110_110611


namespace smallest_gcd_l110_110233

theorem smallest_gcd (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (H1 : Nat.gcd x y = 270) (H2 : Nat.gcd x z = 105) : Nat.gcd y z = 15 :=
sorry

end smallest_gcd_l110_110233


namespace find_side_b_l110_110277

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end find_side_b_l110_110277


namespace rectangle_perimeter_gt_16_l110_110067

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l110_110067


namespace direct_proportional_function_point_l110_110110

theorem direct_proportional_function_point 
    (h₁ : ∃ k : ℝ, ∀ x : ℝ, (2, -3).snd = k * (2, -3).fst)
    (h₂ : ∃ k : ℝ, ∀ x : ℝ, (4, -6).snd = k * (4, -6).fst)
    : (∃ k : ℝ, k = -(3 / 2)) :=
by
  sorry

end direct_proportional_function_point_l110_110110


namespace trajectory_of_point_l110_110208

/-- 
  Given points A and B on the coordinate plane, with |AB|=2, 
  and a moving point P such that the sum of the distances from P
  to points A and B is constantly 2, the trajectory of point P 
  is the line segment AB. 
-/
theorem trajectory_of_point (A B P : ℝ × ℝ) 
  (h_AB : dist A B = 2) 
  (h_sum : dist P A + dist P B = 2) :
  P ∈ segment ℝ A B :=
sorry

end trajectory_of_point_l110_110208


namespace only_C_forms_triangle_l110_110822

def triangle_sides (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem only_C_forms_triangle :
  ¬ triangle_sides 3 4 8 ∧
  ¬ triangle_sides 2 5 2 ∧
  triangle_sides 3 5 6 ∧
  ¬ triangle_sides 5 6 11 :=
by
  sorry

end only_C_forms_triangle_l110_110822


namespace number_of_n_l110_110934

theorem number_of_n (n : ℕ) (h1 : n ≤ 1000) (h2 : ∃ k : ℕ, 18 * n = k^2) : 
  ∃ K : ℕ, K = 7 :=
sorry

end number_of_n_l110_110934


namespace problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l110_110365

/-- Lean statement for the math proof problem -/

/- First problem -/
theorem problem1_equation_of_line_intersection_perpendicular :
  ∃ k, 3 * k - 2 * ( - (5 - 3 * k) / 2) - 11 = 0 :=
sorry

/- Second problem -/
theorem problem2_equation_of_line_point_equal_intercepts :
  (∃ a, (1, 2) ∈ {(x, y) | x + y = a}) ∧ a = 3
  ∨ (∃ b, (1, 2) ∈ {(x, y) | y = b * x}) ∧ b = 2 :=
sorry

end problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l110_110365


namespace orig_polygon_sides_l110_110472

theorem orig_polygon_sides (n : ℕ) (S : ℕ) :
  (n - 1 > 2) ∧ S = 1620 → (n = 10 ∨ n = 11 ∨ n = 12) :=
by
  sorry

end orig_polygon_sides_l110_110472


namespace coin_flip_sequences_l110_110700

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110700


namespace coin_flip_sequences_l110_110690

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110690


namespace symmetry_construction_complete_l110_110455

-- Conditions: The word and the chosen axis of symmetry
def word : String := "ГЕОМЕТРИя"

inductive Axis
| horizontal
| vertical

-- The main theorem which states that a symmetrical figure can be constructed for the given word and axis
theorem symmetry_construction_complete (axis : Axis) : ∃ (symmetrical : String), 
  (axis = Axis.horizontal ∨ axis = Axis.vertical) → 
   symmetrical = "яИРТЕМОЕГ" := 
by
  sorry

end symmetry_construction_complete_l110_110455


namespace largest_value_WY_cyclic_quadrilateral_l110_110830

theorem largest_value_WY_cyclic_quadrilateral :
  ∃ WZ ZX ZY YW : ℕ, 
    WZ ≠ ZX ∧ WZ ≠ ZY ∧ WZ ≠ YW ∧ ZX ≠ ZY ∧ ZX ≠ YW ∧ ZY ≠ YW ∧ 
    WZ < 20 ∧ ZX < 20 ∧ ZY < 20 ∧ YW < 20 ∧ 
    WZ * ZY = ZX * YW ∧
    (∀ WY', (∃ WY : ℕ, WY' < WY → WY <= 19 )) :=
sorry

end largest_value_WY_cyclic_quadrilateral_l110_110830


namespace period_f_l110_110952

noncomputable def f (x : ℝ) : ℝ := (Real.tan x) / (1 - (Real.tan x)^2)

theorem period_f : (∀ x : ℝ, f(x) = f(x + (π / 2))) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f(x) = f(x + T)) → T ≥ (π / 2)) := 
by 
  sorry

end period_f_l110_110952


namespace misha_second_round_score_l110_110127

def misha_score_first_round (darts : ℕ) (score_per_dart_min : ℕ) : ℕ := 
  darts * score_per_dart_min

def misha_score_second_round (score_first : ℕ) (multiplier : ℕ) : ℕ := 
  score_first * multiplier

def misha_score_third_round (score_second : ℕ) (multiplier : ℚ) : ℚ := 
  score_second * multiplier

theorem misha_second_round_score (darts : ℕ) (score_per_dart_min : ℕ) (multiplier_second : ℕ) (multiplier_third : ℚ) 
  (h_darts : darts = 8) (h_score_per_dart_min : score_per_dart_min = 3) (h_multiplier_second : multiplier_second = 2) (h_multiplier_third : multiplier_third = 1.5) :
  misha_score_second_round (misha_score_first_round darts score_per_dart_min) multiplier_second = 48 :=
by sorry

end misha_second_round_score_l110_110127


namespace remainder_when_divided_by_20_l110_110574

theorem remainder_when_divided_by_20 
  (n r : ℤ) 
  (k : ℤ)
  (h1 : n % 20 = r) 
  (h2 : 2 * n % 10 = 2)
  (h3 : 0 ≤ r ∧ r < 20)
  : r = 1 := 
sorry

end remainder_when_divided_by_20_l110_110574


namespace value_of_f_g_10_l110_110232

def g (x : ℤ) : ℤ := 4 * x + 6
def f (x : ℤ) : ℤ := 6 * x - 10

theorem value_of_f_g_10 : f (g 10) = 266 :=
by
  sorry

end value_of_f_g_10_l110_110232


namespace ones_digit_of_tripling_4567_l110_110634

theorem ones_digit_of_tripling_4567 : 
  let n := 4567 in 
  let tripled := 3 * n in
  (tripled % 10) = 1 :=
by
  sorry

end ones_digit_of_tripling_4567_l110_110634


namespace find_side_b_l110_110276

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end find_side_b_l110_110276


namespace least_positive_integer_l110_110930

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l110_110930


namespace taxi_charge_l110_110404

theorem taxi_charge :
  ∀ (initial_fee additional_charge_per_segment total_distance total_charge : ℝ),
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (5/2 * total_distance) = 0.35 :=
by
  intros initial_fee additional_charge_per_segment total_distance total_charge
  intros h_initial_fee h_total_distance h_total_charge
  -- Proof here
  sorry

end taxi_charge_l110_110404


namespace factor_expression_l110_110780

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110780


namespace value_of_expression_l110_110805

open Real

theorem value_of_expression (α : ℝ) (h : 3 * sin α + cos α = 0) :
  1 / (cos α ^ 2 + sin (2 * α)) = 10 / 3 :=
by
  sorry

end value_of_expression_l110_110805


namespace fill_pool_time_l110_110026

theorem fill_pool_time (pool_volume : ℕ := 32000) 
                       (num_hoses : ℕ := 5) 
                       (flow_rate_per_hose : ℕ := 4) 
                       (operation_minutes : ℕ := 45) 
                       (maintenance_minutes : ℕ := 15) 
                       : ℕ :=
by
  -- Calculation steps will go here in the actual proof
  sorry

example : fill_pool_time = 47 := by
  -- Proof of the theorem fill_pool_time here
  sorry

end fill_pool_time_l110_110026


namespace coefficient_of_x3_in_expansion_l110_110188

-- Definitions based on the conditions provided
def binomial_expansion_term (n r: ℕ) (a x: ℝ) : ℝ := (Nat.choose n r) * (a^(n-r)) * (x^r)

-- The main theorem to prove
theorem coefficient_of_x3_in_expansion : 
  (polynomial.coeff ((1 - X) * polynomial.expand 5 2) 3 = -40) :=
by sorry

end coefficient_of_x3_in_expansion_l110_110188


namespace marble_count_l110_110397

-- Define the variables for the number of marbles
variables (o p y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := o = 1.3 * p
def condition2 : Prop := y = 1.5 * o

-- Define the total number of marbles based on the conditions
def total_marbles : ℝ := o + p + y

-- The theorem statement that needs to be proved
theorem marble_count (h1 : condition1 o p) (h2 : condition2 o y) : total_marbles o p y = 3.269 * o :=
by sorry

end marble_count_l110_110397


namespace pounds_lost_per_month_l110_110031

variable (starting_weight : ℕ) (ending_weight : ℕ) (months_in_year : ℕ) 

theorem pounds_lost_per_month
    (h_start : starting_weight = 250)
    (h_end : ending_weight = 154)
    (h_months : months_in_year = 12) :
    (starting_weight - ending_weight) / months_in_year = 8 := 
sorry

end pounds_lost_per_month_l110_110031


namespace john_total_beats_l110_110975

noncomputable def minutes_in_hour : ℕ := 60
noncomputable def hours_per_day : ℕ := 2
noncomputable def days_played : ℕ := 3
noncomputable def beats_per_minute : ℕ := 200

theorem john_total_beats :
  (beats_per_minute * hours_per_day * minutes_in_hour * days_played) = 72000 :=
by
  -- we will implement the proof here
  sorry

end john_total_beats_l110_110975


namespace remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l110_110788

theorem remainder_8_times_10_pow_18_plus_1_pow_18_div_9 :
  (8 * 10^18 + 1^18) % 9 = 0 := 
by 
  sorry

end remainder_8_times_10_pow_18_plus_1_pow_18_div_9_l110_110788


namespace range_of_m_l110_110237

-- Define the set A and condition
def A (m : ℝ) : Set ℝ := { x : ℝ | x^2 - 2 * x + m = 0 }

-- The theorem stating the range of m
theorem range_of_m (m : ℝ) : (A m = ∅) ↔ m > 1 :=
by
  sorry

end range_of_m_l110_110237


namespace minimum_bottles_needed_l110_110012

theorem minimum_bottles_needed (medium_volume jumbo_volume : ℕ) (h_medium : medium_volume = 120) (h_jumbo : jumbo_volume = 2000) : 
  let minimum_bottles := (jumbo_volume + medium_volume - 1) / medium_volume
  minimum_bottles = 17 :=
by
  sorry

end minimum_bottles_needed_l110_110012


namespace distribution_plans_l110_110900

theorem distribution_plans (teachers schools : ℕ) (h_teachers : teachers = 3) (h_schools : schools = 6) : 
  ∃ plans : ℕ, plans = 210 :=
by
  sorry

end distribution_plans_l110_110900


namespace Diego_total_stamp_cost_l110_110335

theorem Diego_total_stamp_cost :
  let price_brazil_colombia := 0.07
  let price_peru := 0.05
  let num_brazil_50s := 6
  let num_brazil_60s := 9
  let num_peru_50s := 8
  let num_peru_60s := 5
  let num_colombia_50s := 7
  let num_colombia_60s := 6
  let total_brazil := num_brazil_50s + num_brazil_60s
  let total_peru := num_peru_50s + num_peru_60s
  let total_colombia := num_colombia_50s + num_colombia_60s
  let cost_brazil := total_brazil * price_brazil_colombia
  let cost_peru := total_peru * price_peru
  let cost_colombia := total_colombia * price_brazil_colombia
  cost_brazil + cost_peru + cost_colombia = 2.61 :=
by
  sorry

end Diego_total_stamp_cost_l110_110335


namespace triangle_internal_region_l110_110897

-- Define the three lines forming the triangle
def line1 (x y : ℝ) : Prop := x + 2 * y = 2
def line2 (x y : ℝ) : Prop := 2 * x + y = 2
def line3 (x y : ℝ) : Prop := x - y = 3

-- Define the inequalities representing the internal region of the triangle
def region (x y : ℝ) : Prop :=
  x - y < 3 ∧ x + 2 * y < 2 ∧ 2 * x + y > 2

-- State that the internal region excluding the boundary is given by the inequalities
theorem triangle_internal_region (x y : ℝ) :
  (∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 x y) → region x y :=
  sorry

end triangle_internal_region_l110_110897


namespace expected_return_correct_l110_110175

-- Define the probabilities
def p1 := 1/4
def p2 := 1/4
def p3 := 1/6
def p4 := 1/3

-- Define the payouts
def payout (n : ℕ) (previous_odd : Bool) : ℝ :=
  match n with
  | 1 => 2
  | 2 => if previous_odd then -3 else 0
  | 3 => 0
  | 4 => 5
  | _ => 0

-- Define the expected values of one throw
def E1 : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

def E2_odd : ℝ :=
  p1 * payout 1 true + p2 * payout 2 true + p3 * payout 3 true + p4 * payout 4 true

def E2_even : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

-- Define the probability of throwing an odd number first
def p_odd : ℝ := p1 + p3

-- Define the probability of not throwing an odd number first
def p_even : ℝ := 1 - p_odd

-- Define the total expected return
def total_expected_return : ℝ :=
  E1 + (p_odd * E2_odd + p_even * E2_even)


theorem expected_return_correct :
  total_expected_return = 4.18 :=
  by
    -- The proof is omitted
    sorry

end expected_return_correct_l110_110175


namespace rainy_days_l110_110675

namespace Mo

def drinks (R NR n : ℕ) :=
  -- Condition 3: Total number of days in the week equation
  R + NR = 7 ∧
  -- Condition 1-2: Total cups of drinks equation
  n * R + 3 * NR = 26 ∧
  -- Condition 4: Difference in cups of tea and hot chocolate equation
  3 * NR - n * R = 10

theorem rainy_days (R NR n : ℕ) (h: drinks R NR n) : 
  R = 1 := sorry

end Mo

end rainy_days_l110_110675


namespace sum_radical_conjugate_l110_110502

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l110_110502


namespace distribute_balls_into_boxes_l110_110089

theorem distribute_balls_into_boxes : 
  let n := 5
  let k := 4
  (n.choose (k - 1) + k - 1).choose (k - 1) = 56 :=
by
  sorry

end distribute_balls_into_boxes_l110_110089


namespace 1_part1_2_part2_l110_110539

/-
Define M and N sets
-/
def M : Set ℝ := {x | x ≥ 1 / 2}
def N : Set ℝ := {y | y ≤ 1}

/-
Theorem 1: Difference set M - N
-/
theorem part1 : (M \ N) = {x | x > 1} := by
  sorry

/-
Define A and B sets and the condition A - B = ∅
-/
def A (a : ℝ) : Set ℝ := {x | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {y | -1 / 2 < y ∧ y ≤ 2}

/-
Theorem 2: Range of values for a
-/
theorem part2 (a : ℝ) (h : A a \ B = ∅) : a ∈ Set.Iio (-12) ∪ Set.Ici 3 := by
  sorry

end 1_part1_2_part2_l110_110539


namespace find_value_of_B_l110_110542

theorem find_value_of_B (B : ℚ) (h : 4 * B + 4 = 33) : B = 29 / 4 :=
by
  sorry

end find_value_of_B_l110_110542


namespace find_angle_C_find_max_area_l110_110396

variable {A B C a b c : ℝ}

-- Given Conditions
def condition1 (c B a b C : ℝ) := c * Real.cos B + (b - 2 * a) * Real.cos C = 0
def condition2 (c : ℝ) := c = 2 * Real.sqrt 3

-- Problem (1): Prove the size of angle C
theorem find_angle_C (h : condition1 c B a b C) (h2 : condition2 c) : C = Real.pi / 3 := 
  sorry

-- Problem (2): Prove the maximum area of ΔABC
theorem find_max_area (h : condition1 c B a b C) (h2 : condition2 c) :
  ∃ (A B : ℝ), B = 2 * Real.pi / 3 - A ∧ 
    (∀ (A B : ℝ), Real.sin (2 * A - Real.pi / 6) = 1 → 
    1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 ∧ 
    a = b ∧ b = c) := 
  sorry

end find_angle_C_find_max_area_l110_110396


namespace x_y_sum_vals_l110_110959

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end x_y_sum_vals_l110_110959


namespace shorter_piece_length_l110_110179

theorem shorter_piece_length : ∃ (x : ℕ), (x + (x + 2) = 30) ∧ x = 14 :=
by {
  sorry
}

end shorter_piece_length_l110_110179


namespace triangle_problem_l110_110262

noncomputable def find_b (a b c : ℝ) : Prop :=
  let B : ℝ := 60 * Real.pi / 180 -- converting 60 degrees to radians
  b = 2 * Real.sqrt 2

theorem triangle_problem
  (a b c : ℝ)
  (h_area : (1 / 2) * a * c * Real.sin (60 * Real.pi / 180) = Real.sqrt 3)
  (h_cosine : a^2 + c^2 = 3 * a * c) : find_b a b c :=
by
  -- The proof would go here, but we're skipping it as per the instructions.
  sorry

end triangle_problem_l110_110262


namespace rhombus_area_l110_110438

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  0.5 * d1 * d2

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : area_of_rhombus d1 d2 = 6 :=
by
  sorry

end rhombus_area_l110_110438


namespace value_of_y_l110_110247

theorem value_of_y (x y : ℤ) (h1 : x + y = 270) (h2 : x - y = 200) : y = 35 :=
by
  sorry

end value_of_y_l110_110247


namespace reduced_price_per_kg_l110_110755

variables (P P' : ℝ)

-- Given conditions
def condition1 := P' = P / 2
def condition2 := 800 / P' = 800 / P + 5

-- Proof problem statement
theorem reduced_price_per_kg (P P' : ℝ) (h1 : condition1 P P') (h2 : condition2 P P') :
  P' = 80 :=
by
  sorry

end reduced_price_per_kg_l110_110755


namespace smallest_integer_condition_l110_110919

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l110_110919


namespace tiling_polygons_l110_110583

theorem tiling_polygons (n : ℕ) (h1 : 2 < n) (h2 : ∃ x : ℕ, x * (((n - 2) * 180 : ℝ) / n) = 360) :
  n = 3 ∨ n = 4 ∨ n = 6 := 
by
  sorry

end tiling_polygons_l110_110583


namespace sum_of_radical_conjugates_l110_110517

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l110_110517


namespace negation_of_at_most_three_l110_110855

theorem negation_of_at_most_three (x : ℕ) : ¬ (x ≤ 3) ↔ x > 3 :=
by sorry

end negation_of_at_most_three_l110_110855


namespace range_of_m_l110_110061

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l110_110061


namespace girls_without_notebooks_l110_110286

noncomputable def girls_in_class : Nat := 20
noncomputable def students_with_notebooks : Nat := 25
noncomputable def boys_with_notebooks : Nat := 16

theorem girls_without_notebooks : 
  (girls_in_class - (students_with_notebooks - boys_with_notebooks)) = 11 := by
  sorry

end girls_without_notebooks_l110_110286


namespace option_B_is_equal_to_a_8_l110_110178

-- Statement: (a^2)^4 equals a^8
theorem option_B_is_equal_to_a_8 (a : ℝ) : (a^2)^4 = a^8 :=
by { sorry }

end option_B_is_equal_to_a_8_l110_110178


namespace john_back_squat_increase_l110_110114

-- Definitions based on conditions
def back_squat_initial : ℝ := 200
def k : ℝ := 0.8
def j : ℝ := 0.9
def total_weight_moved : ℝ := 540

-- The variable representing the increase in back squat
variable (x : ℝ)

-- The Lean statement to prove
theorem john_back_squat_increase :
  3 * (j * k * (back_squat_initial + x)) = total_weight_moved → x = 50 := by
  sorry

end john_back_squat_increase_l110_110114


namespace perimeter_of_ABCD_l110_110588

def Point := ℝ × ℝ

def square_dist (p1 p2 : Point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt (square_dist p1 p2)

theorem perimeter_of_ABCD :
  ∀ (A B C D : Point), 
  distance (Point.mk 0 0) (Point.mk 4 0) < distance (Point.mk 0 0) (Point.mk 7 3) →
  distance D C = 5 →
  (Point.mk 0 4).2 = 4 →
  (Point.mk 7 0).1 = 7 →
  distance A B + distance B C + distance C D + distance D A = 26 := by
  intros A B C D AD_lt_BC DC_eq_5 DN_eq_4 BN_eq_7
  sorry

end perimeter_of_ABCD_l110_110588


namespace cone_slant_height_correct_l110_110747

noncomputable def cone_slant_height (r : ℝ) : ℝ := 4 * r

theorem cone_slant_height_correct (r : ℝ) (h₁ : π * r^2 + π * r * cone_slant_height r = 5 * π)
  (h₂ : 2 * π * r = (1/4) * 2 * π * cone_slant_height r) : cone_slant_height r = 4 :=
by
  sorry

end cone_slant_height_correct_l110_110747


namespace coin_flip_sequences_l110_110685

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110685


namespace simplify_trig_l110_110294

theorem simplify_trig (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) / 
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) = 
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by 
  sorry

end simplify_trig_l110_110294


namespace range_of_m_l110_110058

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end range_of_m_l110_110058


namespace star_property_l110_110558

-- Define the operation a ⋆ b = (a - b) ^ 3
def star (a b : ℝ) : ℝ := (a - b) ^ 3

-- State the theorem
theorem star_property (x y : ℝ) : star ((x - y) ^ 3) ((y - x) ^ 3) = 8 * (x - y) ^ 9 := 
by 
  sorry

end star_property_l110_110558


namespace part1_part2_l110_110073

def setA (x : ℝ) : Prop := x^2 - 5 * x - 6 < 0

def setB (a x : ℝ) : Prop := 2 * a - 1 ≤ x ∧ x < a + 5

open Set

theorem part1 : 
  let a := 0
  A = {x : ℝ | -1 < x ∧ x < 6} →
  B a = {x : ℝ | -1 ≤ x ∧ x < 5} →
  {x | (setA x) ∧ (setB a x)} = {x | -1 < x ∧ x < 5} :=
by
  sorry

theorem part2 : 
  A = {x : ℝ | -1 < x ∧ x < 6} →
  (B : ℝ → Set real) →
  (∀ x, (setA x ∨ setB a x) → setA x) →
  { a : ℝ | (0 < a ∧ a ≤ 1) ∨ a ≥ 6 } :=
by
  sorry

end part1_part2_l110_110073


namespace calculate_f_of_g_l110_110229

def g (x : ℝ) := 4 * x + 6
def f (x : ℝ) := 6 * x - 10

theorem calculate_f_of_g :
  f (g 10) = 266 := by
  sorry

end calculate_f_of_g_l110_110229


namespace g_composition_evaluation_l110_110607

def g (x : ℤ) : ℤ :=
  if x < 5 then x^3 + x^2 - 6 else 2 * x - 18

theorem g_composition_evaluation : g (g (g 16)) = 2 := by
  sorry

end g_composition_evaluation_l110_110607


namespace unique_solution_iff_a_eq_2019_l110_110818

theorem unique_solution_iff_a_eq_2019 (x a : ℝ) :
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) ↔ a = 2019 :=
by
  sorry

end unique_solution_iff_a_eq_2019_l110_110818


namespace range_of_m_l110_110059

theorem range_of_m (x y : ℝ) (m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : (1/x) + (4/y) = 1) :
  (x + y > m^2 + 8 * m) → (-9 < m ∧ m < 1) :=
by 
  sorry

end range_of_m_l110_110059


namespace find_mistake_l110_110647

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l110_110647


namespace speed_of_man_in_still_water_l110_110669

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : 5 * (v_m + v_s) = 45) (h2 : 5 * (v_m - v_s) = 25) : v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l110_110669


namespace waiter_tables_l110_110332

/-
Problem:
A waiter had 22 customers in his section.
14 of them left.
The remaining customers were seated at tables with 4 people per table.
Prove the number of tables is 2.
-/

theorem waiter_tables:
  ∃ (tables : ℤ), 
    (∀ (customers_initial customers_remaining people_per_table tables_calculated : ℤ), 
      customers_initial = 22 →
      customers_remaining = customers_initial - 14 →
      people_per_table = 4 →
      tables_calculated = customers_remaining / people_per_table →
      tables = tables_calculated) →
    tables = 2 :=
by
  sorry

end waiter_tables_l110_110332


namespace geometric_sequence_first_term_l110_110639

theorem geometric_sequence_first_term (a r : ℝ)
    (h1 : a * r^2 = 3)
    (h2 : a * r^4 = 27) :
    a = 1 / 3 := by
    sorry

end geometric_sequence_first_term_l110_110639


namespace expr_value_l110_110029

-- Define the given expression
def expr : ℕ := 11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3

-- Assert the proof goal
theorem expr_value : expr = 21 := by
  sorry

end expr_value_l110_110029


namespace minimum_value_l110_110072

variables (a b c d : ℝ)
-- Conditions
def condition1 := (b - 2 * a^2 + 3 * Real.log a)^2 = 0
def condition2 := (c - d - 3)^2 = 0

-- Theorem stating the goal
theorem minimum_value (h1 : condition1 a b) (h2 : condition2 c d) : 
  (a - c)^2 + (b - d)^2 = 8 :=
sorry

end minimum_value_l110_110072


namespace exists_n_l110_110051

def F_n (a n : ℕ) : ℕ :=
  let q := a ^ (1 / n)
  let r := a % n
  q + r

noncomputable def largest_A : ℕ :=
  53590

theorem exists_n (a : ℕ) (h : a ≤ largest_A) :
  ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    F_n (F_n (F_n (F_n (F_n (F_n a n1) n2) n3) n4) n5) n6 = 1 := 
sorry

end exists_n_l110_110051


namespace classroom_gpa_l110_110632

theorem classroom_gpa (x : ℝ) (h1 : (1 / 3) * x + (2 / 3) * 18 = 17) : x = 15 := 
by 
    sorry

end classroom_gpa_l110_110632


namespace functional_eq_implies_odd_l110_110956

variable {f : ℝ → ℝ}

def functional_eq (f : ℝ → ℝ) :=
∀ a b, f (a + b) + f (a - b) = 2 * f a * Real.cos b

theorem functional_eq_implies_odd (h : functional_eq f) (hf_non_zero : ¬∀ x, f x = 0) : 
  ∀ x, f (-x) = -f x := 
by
  sorry

end functional_eq_implies_odd_l110_110956


namespace sum_of_conjugates_l110_110490

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l110_110490


namespace total_people_in_class_l110_110003

-- Define the number of people based on their interests
def likes_both: Nat := 5
def only_baseball: Nat := 2
def only_football: Nat := 3
def likes_neither: Nat := 6

-- Define the total number of people in the class
def total_people := likes_both + only_baseball + only_football + likes_neither

-- Theorem statement
theorem total_people_in_class : total_people = 16 :=
by
  -- Proof is skipped
  sorry

end total_people_in_class_l110_110003


namespace uncool_students_in_two_classes_l110_110578

theorem uncool_students_in_two_classes
  (students_class1 : ℕ)
  (cool_dads_class1 : ℕ)
  (cool_moms_class1 : ℕ)
  (both_cool_class1 : ℕ)
  (students_class2 : ℕ)
  (cool_dads_class2 : ℕ)
  (cool_moms_class2 : ℕ)
  (both_cool_class2 : ℕ)
  (h1 : students_class1 = 45)
  (h2 : cool_dads_class1 = 22)
  (h3 : cool_moms_class1 = 25)
  (h4 : both_cool_class1 = 11)
  (h5 : students_class2 = 35)
  (h6 : cool_dads_class2 = 15)
  (h7 : cool_moms_class2 = 18)
  (h8 : both_cool_class2 = 7) :
  (students_class1 - ((cool_dads_class1 - both_cool_class1) + (cool_moms_class1 - both_cool_class1) + both_cool_class1) +
   students_class2 - ((cool_dads_class2 - both_cool_class2) + (cool_moms_class2 - both_cool_class2) + both_cool_class2)
  ) = 18 :=
sorry

end uncool_students_in_two_classes_l110_110578


namespace f_equality_2019_l110_110282

theorem f_equality_2019 (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), f (m + n) ≥ f m + f (f n) - 1) : 
  f 2019 = 2019 :=
sorry

end f_equality_2019_l110_110282


namespace exponent_form_l110_110449

theorem exponent_form (y : ℕ) (w : ℕ) (k : ℕ) : w = 3 ^ y → w % 10 = 7 → ∃ (k : ℕ), y = 4 * k + 3 :=
by
  intros h1 h2
  sorry

end exponent_form_l110_110449


namespace angle_complement_supplement_l110_110533

theorem angle_complement_supplement (x : ℝ) (h1 : 90 - x = (1 / 2) * (180 - x)) : x = 90 := by
  sorry

end angle_complement_supplement_l110_110533


namespace percent_non_condiments_l110_110166

def sandwich_weight : ℕ := 150
def condiment_weight : ℕ := 45
def non_condiment_weight (total: ℕ) (condiments: ℕ) : ℕ := total - condiments
def percentage (num denom: ℕ) : ℕ := (num * 100) / denom

theorem percent_non_condiments : 
  percentage (non_condiment_weight sandwich_weight condiment_weight) sandwich_weight = 70 :=
by
  sorry

end percent_non_condiments_l110_110166


namespace prime_count_at_least_two_l110_110937

theorem prime_count_at_least_two :
  ∃ (n1 n2 : ℕ), n1 ≥ 2 ∧ n2 ≥ 2 ∧ (n1 ≠ n2) ∧ Prime (n1^3 + n1^2 + 1) ∧ Prime (n2^3 + n2^2 + 1) := 
by
  sorry

end prime_count_at_least_two_l110_110937


namespace rectangle_perimeter_l110_110642

theorem rectangle_perimeter (a b : ℝ) (h1 : (a + 3) * (b + 3) = a * b + 48) : 
  2 * (a + 3 + b + 3) = 38 :=
by
  sorry

end rectangle_perimeter_l110_110642


namespace calculate_bubble_bath_needed_l110_110252

theorem calculate_bubble_bath_needed :
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  total_guests * bubble_bath_per_guest = 1950 := by
  let double_suites_capacity := 5 * 4
  let rooms_for_couples_capacity := 13 * 2
  let single_rooms_capacity := 14 * 1
  let family_rooms_capacity := 3 * 6
  let total_guests := double_suites_capacity + rooms_for_couples_capacity + single_rooms_capacity + family_rooms_capacity
  let bubble_bath_per_guest := 25
  sorry

end calculate_bubble_bath_needed_l110_110252


namespace average_throws_to_lasso_l110_110844

theorem average_throws_to_lasso (p : ℝ) (h₁ : 1 - (1 - p)^3 = 0.875) : (1 / p) = 2 :=
by
  sorry

end average_throws_to_lasso_l110_110844


namespace coin_flip_sequences_l110_110706

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110706


namespace minimum_sum_dimensions_l110_110164

def is_product (a b c : ℕ) (v : ℕ) : Prop :=
  a * b * c = v

def sum (a b c : ℕ) : ℕ :=
  a + b + c

theorem minimum_sum_dimensions : 
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ is_product a b c 3003 ∧ sum a b c = 45 :=
by
  sorry

end minimum_sum_dimensions_l110_110164


namespace max_total_weight_of_chocolates_l110_110128

theorem max_total_weight_of_chocolates 
  (A B C : ℕ)
  (hA : A ≤ 100)
  (hBC : B - C ≤ 100)
  (hC : C ≤ 100)
  (h_distribute : A ≤ 100 ∧ (B - C) ≤ 100)
  : (A + B = 300) :=
by 
  sorry

end max_total_weight_of_chocolates_l110_110128


namespace solution_set_l110_110862

theorem solution_set (x : ℝ) : 
  (x * (x + 2) > 0 ∧ |x| < 1) ↔ (0 < x ∧ x < 1) := 
by 
  sorry

end solution_set_l110_110862


namespace number_of_pencils_l110_110136

theorem number_of_pencils (P : ℕ) (h : ∃ (n : ℕ), n * 4 = P) : ∃ k, 4 * k = P :=
  by
  sorry

end number_of_pencils_l110_110136


namespace ratio_of_x_to_y_l110_110817

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
sorry

end ratio_of_x_to_y_l110_110817


namespace sum_of_number_and_its_radical_conjugate_l110_110508

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l110_110508


namespace coin_flips_sequences_count_l110_110729

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110729


namespace vasya_max_pencils_l110_110986

theorem vasya_max_pencils (money_for_pencils : ℕ) (rebate_20 : ℕ) (rebate_5 : ℕ) :
  money_for_pencils = 30 → rebate_20 = 25 → rebate_5 = 10 → ∃ max_pencils, max_pencils = 36 :=
by
  intros h_money h_r20 h_r5
  sorry

end vasya_max_pencils_l110_110986


namespace coin_flip_sequences_l110_110698

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110698


namespace sum_of_radical_conjugates_l110_110513

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l110_110513


namespace present_age_of_B_l110_110151

theorem present_age_of_B
  (A B : ℕ)
  (h1 : A = B + 5)
  (h2 : A + 30 = 2 * (B - 30)) :
  B = 95 :=
by { sorry }

end present_age_of_B_l110_110151


namespace gondor_laptop_earning_l110_110388

theorem gondor_laptop_earning :
  ∃ L : ℝ, (3 * 10 + 5 * 10 + 2 * L + 4 * L = 200) → L = 20 :=
by
  use 20
  sorry

end gondor_laptop_earning_l110_110388


namespace least_integer_value_satisfying_inequality_l110_110197

theorem least_integer_value_satisfying_inequality : ∃ x : ℤ, 3 * |x| + 6 < 24 ∧ (∀ y : ℤ, 3 * |y| + 6 < 24 → x ≤ y) :=
  sorry

end least_integer_value_satisfying_inequality_l110_110197


namespace min_value_x_plus_y_l110_110080

open Real

noncomputable def xy_plus_x_minus_y_minus_10_eq_zero (x y: ℝ) := x * y + x - y - 10 = 0

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : xy_plus_x_minus_y_minus_10_eq_zero x y) : 
  x + y ≥ 6 :=
by
  sorry

end min_value_x_plus_y_l110_110080


namespace red_markers_count_l110_110518

-- Define the given conditions
def blue_markers : ℕ := 1028
def total_markers : ℕ := 3343

-- Define the red_makers calculation based on the conditions
def red_markers (total_markers blue_markers : ℕ) : ℕ := total_markers - blue_markers

-- Prove that the number of red markers is 2315 given the conditions
theorem red_markers_count : red_markers total_markers blue_markers = 2315 := by
  -- We can skip the proof for this demonstration
  sorry

end red_markers_count_l110_110518


namespace sum_of_inverse_poly_roots_l110_110909

noncomputable def cubic_roots_sum_inverse (p q r : ℝ) (h1 : (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧ (0 < p ∧ p < 2) ∧ (0 < q ∧ q < 2) ∧ (0 < r ∧ r < 2)) 
(h2 : (60 * p ^ 3 - 70 * p ^ 2 + 24 * p - 2 = 0) ∧ (60 * q ^ 3 - 70 * q ^ 2 + 24 * q - 2 = 0) ∧ (60 * r ^ 3 - 70 * r ^ 2 + 24 * r - 2 = 0)) : ℝ :=
  (1 / (2 - p)) + (1 / (2 - q)) + (1 / (2 - r))

theorem sum_of_inverse_poly_roots (p q r : ℝ) (h1 : (p ≠ q ∧ q ≠ r ∧ p ≠ r) ∧ (0 < p ∧ p < 2) ∧ (0 < q ∧ q < 2) ∧ (0 < r ∧ r < 2)) 
(h2 : (60 * p ^ 3 - 70 * p ^ 2 + 24 * p - 2 = 0) ∧ (60 * q ^ 3 - 70 * q ^ 2 + 24 * q - 2 = 0) ∧ (60 * r ^ 3 - 70 * r ^ 2 + 24 * r - 2 = 0)): 
  cubic_roots_sum_inverse p q r h1 h2 = 116 / 15 := 
  sorry

end sum_of_inverse_poly_roots_l110_110909


namespace find_common_difference_find_max_sum_find_max_n_l110_110140

-- Condition for the sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement (1): Find the common difference
theorem find_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 23)
  (h2 : is_arithmetic_sequence a d)
  (h6 : a 6 > 0)
  (h7 : a 7 < 0) : d = -4 :=
sorry

-- Problem statement (2): Find the maximum value of the sum S₆
theorem find_max_sum (d : ℤ) (h : d = -4) : 6 * 23 + (6 * 5 / 2) * d = 78 :=
sorry

-- Problem statement (3): Find the maximum value of n when S_n > 0
theorem find_max_n (d : ℤ) (h : d = -4) : ∀ n : ℕ, (n > 0 ∧ (23 * n + (n * (n - 1) / 2) * d > 0)) → n ≤ 12 :=
sorry

end find_common_difference_find_max_sum_find_max_n_l110_110140


namespace largest_natural_S_n_gt_zero_l110_110283

noncomputable def S_n (n : ℕ) : ℤ :=
  let a1 := 9
  let d := -2
  n * (2 * a1 + (n - 1) * d) / 2

theorem largest_natural_S_n_gt_zero
  (a_2 : ℤ) (a_4 : ℤ)
  (h1 : a_2 = 7) (h2 : a_4 = 3) :
  ∃ n : ℕ, S_n n > 0 ∧ ∀ m : ℕ, m > n → S_n m ≤ 0 := 
sorry

end largest_natural_S_n_gt_zero_l110_110283


namespace find_2a6_minus_a4_l110_110551

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 2) = 2 * a (n + 1) - a n

theorem find_2a6_minus_a4 {a : ℕ → ℤ} 
  (h_seq : is_arithmetic_sequence a)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 6 - a 4 = 24 :=
by
  sorry

end find_2a6_minus_a4_l110_110551


namespace buses_needed_for_trip_l110_110894

theorem buses_needed_for_trip :
  ∀ (total_students students_in_vans bus_capacity : ℕ),
  total_students = 500 →
  students_in_vans = 56 →
  bus_capacity = 45 →
  ⌈(total_students - students_in_vans : ℝ) / bus_capacity⌉ = 10 :=
by
  sorry

end buses_needed_for_trip_l110_110894


namespace part1_part2_l110_110521

-- Definition of the branches of the hyperbola
def C1 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1
def C2 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1

-- Problem Part 1: Proving that P, Q, and R cannot lie on the same branch
theorem part1 (P Q R : ℝ × ℝ) (hP : C1 P) (hQ : C1 Q) (hR : C1 R) : False := by
  sorry

-- Problem Part 2: Finding the coordinates of Q and R
theorem part2 : 
  ∃ Q R : ℝ × ℝ, C1 Q ∧ C1 R ∧ 
                (Q = (2 - Real.sqrt 3, 1 / (2 - Real.sqrt 3))) ∧ 
                (R = (2 + Real.sqrt 3, 1 / (2 + Real.sqrt 3))) := 
by
  sorry

end part1_part2_l110_110521


namespace factorization_l110_110362

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := sorry

end factorization_l110_110362


namespace mark_sold_9_boxes_less_than_n_l110_110843

theorem mark_sold_9_boxes_less_than_n :
  ∀ (n M A : ℕ),
  n = 10 →
  M < n →
  A = n - 2 →
  M + A < n →
  M ≥ 1 →
  A ≥ 1 →
  M = 1 ∧ n - M = 9 :=
by
  intros n M A h_n h_M_lt_n h_A h_MA_lt_n h_M_ge_1 h_A_ge_1
  rw [h_n, h_A] at *
  sorry

end mark_sold_9_boxes_less_than_n_l110_110843


namespace coin_flips_sequences_count_l110_110728

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110728


namespace percentage_of_boys_currently_l110_110251

theorem percentage_of_boys_currently (B G : ℕ) (h1 : B + G = 50) (h2 : B + 50 = 95) : (B / 50) * 100 = 90 := by
  sorry

end percentage_of_boys_currently_l110_110251


namespace number_of_friends_l110_110608

-- Define the initial amount of money John had
def initial_money : ℝ := 20.10 

-- Define the amount spent on sweets
def sweets_cost : ℝ := 1.05 

-- Define the amount given to each friend
def money_per_friend : ℝ := 1.00 

-- Define the amount of money left after giving to friends
def final_money : ℝ := 17.05 

-- Define a theorem to find the number of friends John gave money to
theorem number_of_friends (init_money sweets_cost money_per_friend final_money : ℝ) : 
  (init_money - sweets_cost - final_money) / money_per_friend = 2 :=
by
  sorry

end number_of_friends_l110_110608


namespace expression_evaluation_l110_110878

theorem expression_evaluation : 5^3 - 3 * 5^2 + 3 * 5 - 1 = 64 :=
by
  sorry

end expression_evaluation_l110_110878


namespace solve_equation_l110_110306

theorem solve_equation (x : ℝ) (h : 2 * x + 6 = 2 + 3 * x) : x = 4 :=
by
  sorry

end solve_equation_l110_110306


namespace coin_flip_sequences_l110_110737

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110737


namespace distance_PQ_l110_110124

noncomputable theory
open set

def point (x y : ℝ) : ℝ × ℝ := (x, y)

def line1 (P : ℝ × ℝ) (θ : ℝ) : set (ℝ × ℝ) :=
  { Q | ∃ t : ℝ, Q = (P.1 + t * cos θ, P.2 + t * sin θ) }

def line2 (x y : ℝ) : set (ℝ × ℝ) :=
  { Q | Q.1 - 2 * Q.2 + 11 = 0 }

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem distance_PQ : 
  ∀ P Q : ℝ × ℝ,
  P = (3, 2) →
  ∃ θ θ' : ℝ, θ = real.arctan (3 / 4) ∧ θ' = π/2 - θ ∧
  Q ∈ line1 P θ ∧ Q ∈ line2 x y →
  distance P Q = 25 :=
begin
  intros P Q hP hQl1 hQl2,
  sorry
end

end distance_PQ_l110_110124


namespace root_quadratic_eq_l110_110092

theorem root_quadratic_eq (n m : ℝ) (h : n ≠ 0) (root_condition : n^2 + m * n + 3 * n = 0) : m + n = -3 :=
  sorry

end root_quadratic_eq_l110_110092


namespace grace_can_reach_target_sum_l110_110224

theorem grace_can_reach_target_sum :
  ∃ (half_dollars dimes pennies : ℕ),
    half_dollars ≤ 5 ∧ dimes ≤ 20 ∧ pennies ≤ 25 ∧
    (5 * 50 + 13 * 10 + 5) = 385 :=
sorry

end grace_can_reach_target_sum_l110_110224


namespace distinct_sequences_ten_flips_l110_110711

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110711


namespace speed_of_third_part_l110_110015

theorem speed_of_third_part (d : ℝ) (v : ℝ)
  (h1 : 3 * d = 3.000000000000001)
  (h2 : d / 3 + d / 4 + d / v = 47/60) :
  v = 5 := by
  sorry

end speed_of_third_part_l110_110015


namespace range_of_a_if_inequality_holds_l110_110383

noncomputable def satisfies_inequality_for_all_xy_pos (a : ℝ) :=
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9

theorem range_of_a_if_inequality_holds :
  (∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9) → (a ≥ 4) :=
by
  sorry

end range_of_a_if_inequality_holds_l110_110383


namespace mean_of_xyz_l110_110858

theorem mean_of_xyz (x y z : ℝ) (h1 : 9 * x + 3 * y - 5 * z = -4) (h2 : 5 * x + 2 * y - 2 * z = 13) : 
  (x + y + z) / 3 = 10 := 
sorry

end mean_of_xyz_l110_110858


namespace physics_marks_l110_110671

theorem physics_marks (P C M : ℕ) (h1 : P + C + M = 180) (h2 : P + M = 180) (h3 : P + C = 140) : P = 140 :=
by
  sorry

end physics_marks_l110_110671


namespace general_term_of_sequence_l110_110380

theorem general_term_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S (n + 1) = 3 * (n + 1) ^ 2 - 2 * (n + 1)) →
  a 1 = 1 →
  (∀ n, a (n + 1) = S (n + 1) - S n) →
  (∀ n, a n = 6 * n - 5) := 
by
  intros hS ha1 ha
  sorry

end general_term_of_sequence_l110_110380


namespace eval_at_2_l110_110875

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem eval_at_2 : f 2 = 62 := by
  sorry

end eval_at_2_l110_110875


namespace continuous_stripe_probability_correct_l110_110355

-- Define the conditions of the problem
def regular_tetrahedron := Type*
def stripe (T : regular_tetrahedron) := (T → T → Prop)

-- Define a tetrahedron with a stripe condition
structure tetrahedron_with_stripes (T : regular_tetrahedron) :=
  (stripes : stripe T)

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  4 / 27

-- State the main theorem
theorem continuous_stripe_probability_correct (T : regular_tetrahedron) 
  (t : tetrahedron_with_stripes T) : 
  continuous_stripe_probability = 4 / 27 :=
by
  -- proof goes here
  sorry

end continuous_stripe_probability_correct_l110_110355


namespace find_number_l110_110420

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l110_110420


namespace remainder_of_125_div_j_l110_110540

theorem remainder_of_125_div_j (j : ℕ) (h1 : j > 0) (h2 : 75 % (j^2) = 3) : 125 % j = 5 :=
sorry

end remainder_of_125_div_j_l110_110540


namespace rectangle_perimeter_greater_than_16_l110_110069

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l110_110069


namespace quadractic_integer_roots_l110_110378

theorem quadractic_integer_roots (n : ℕ) (h : n > 0) :
  (∃ x y : ℤ, x^2 - 4 * x + n = 0 ∧ y^2 - 4 * y + n = 0) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end quadractic_integer_roots_l110_110378


namespace impossible_fractions_l110_110118

theorem impossible_fractions (a b c r s t : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t)
  (h1 : a * b + 1 = r ^ 2) (h2 : a * c + 1 = s ^ 2) (h3 : b * c + 1 = t ^ 2) :
  ¬ (∃ (k1 k2 k3 : ℕ), rt / s = k1 ∧ rs / t = k2 ∧ st / r = k3) :=
by
  sorry

end impossible_fractions_l110_110118


namespace top_width_of_channel_l110_110626

theorem top_width_of_channel (b : ℝ) (A : ℝ) (h : ℝ) (w : ℝ) : 
  b = 8 ∧ A = 700 ∧ h = 70 ∧ (A = (1/2) * (w + b) * h) → w = 12 := 
by 
  intro h1
  sorry

end top_width_of_channel_l110_110626


namespace product_of_digits_l110_110570

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : 8 ∣ (10 * A + B)) : A * B = 32 :=
sorry

end product_of_digits_l110_110570


namespace time_after_4350_minutes_is_march_6_00_30_l110_110324

-- Define the start time as a date
def startDate := (2015, 3, 3, 0, 0) -- March 3, 2015 at midnight (00:00)

-- Define the total minutes to add
def totalMinutes := 4350

-- Function to convert minutes to a date and time given a start date
def addMinutes (date : (Nat × Nat × Nat × Nat × Nat)) (minutes : Nat) : (Nat × Nat × Nat × Nat × Nat) :=
  let hours := minutes / 60
  let remainMinutes := minutes % 60
  let days := hours / 24
  let remainHours := hours % 24
  let (year, month, day, hour, min) := date
  (year, month, day + days, remainHours, remainMinutes)

-- Expected result date and time
def expectedDate := (2015, 3, 6, 0, 30) -- March 6, 2015 at 00:30 AM

theorem time_after_4350_minutes_is_march_6_00_30 :
  addMinutes startDate totalMinutes = expectedDate :=
by
  sorry

end time_after_4350_minutes_is_march_6_00_30_l110_110324


namespace Rachel_made_total_amount_l110_110354

def cost_per_bar : ℝ := 3.25
def total_bars_sold : ℕ := 25 - 7
def total_amount_made : ℝ := total_bars_sold * cost_per_bar

theorem Rachel_made_total_amount :
  total_amount_made = 58.50 :=
by
  sorry

end Rachel_made_total_amount_l110_110354


namespace quilt_square_side_length_l110_110869

theorem quilt_square_side_length (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ s : ℝ, (length * width = s * s) ∧ s = 12 :=
by {
  sorry
}

end quilt_square_side_length_l110_110869


namespace part_1_part_2_l110_110055

variable {a b : ℝ}

theorem part_1 (ha : a > 0) (hb : b > 0) : a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

theorem part_2 (ha : a > 0) (hb : b > 0) : a^3 + b^3 ≥ a * b^2 + a^2 * b :=
sorry

end part_1_part_2_l110_110055


namespace find_value_of_m_l110_110236

theorem find_value_of_m (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 4*x + 2*y + m = 0 ∧ (x - 2)^2 + (y + 1)^2 = 4) →
  m = 1 :=
sorry

end find_value_of_m_l110_110236


namespace term_with_largest_binomial_coefficient_in_expansion_l110_110402

theorem term_with_largest_binomial_coefficient_in_expansion :
  ∃ k : ℕ, k = 3 ∧ ∀ n : ℕ, n ≤ 6 → nat.choose 6 n ≤ nat.choose 6 k :=
begin
  sorry
end

end term_with_largest_binomial_coefficient_in_expansion_l110_110402


namespace coin_flip_sequences_l110_110694

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110694


namespace factor_expression_l110_110772

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110772


namespace no_int_b_exists_l110_110839

theorem no_int_b_exists (k n a : ℕ) (hk3 : k ≥ 3) (hn3 : n ≥ 3) (hk_odd : k % 2 = 1) (hn_odd : n % 2 = 1)
  (ha1 : a ≥ 1) (hka : k ∣ (2^a + 1)) (hna : n ∣ (2^a - 1)) :
  ¬ ∃ b : ℕ, b ≥ 1 ∧ k ∣ (2^b - 1) ∧ n ∣ (2^b + 1) :=
sorry

end no_int_b_exists_l110_110839


namespace find_number_l110_110421

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l110_110421


namespace angle_between_hands_at_3_45_l110_110321

def anglePerHour : ℝ := 360 / 12
def minuteHandAngle at_3_45 : ℝ := 270
def hourHandAngle at_3_45 : ℝ := 3 * anglePerHour + (45 / 60) * anglePerHour
def fullAngleDiff at_3_45 : ℝ := minuteHandAngle at_3_45 - hourHandAngle at_3_45
def smallerAngle at_3_45 : ℝ := if fullAngleDiff at_3_45 > 180 
                                then 360 - fullAngleDiff at_3_45 
                                else fullAngleDiff at_3_45

theorem angle_between_hands_at_3_45 : smallerAngle at_3_45 = 202.5 := 
by 
  sorry -- proof is left as an exercise.

end angle_between_hands_at_3_45_l110_110321


namespace least_positive_integer_l110_110923

theorem least_positive_integer (d n p : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) (hp : 1 ≤ p) :
  10 ^ p * d + n = 19 * n ∧ 10 ^ p * d = 18 * n :=
∃ x : ℕ, x = 10 ^ p * d + n ∧ 
         (∀ p', p' < p → ∀ n', n' = 10 ^ p' * d ∧ n' = 18 * n' → 10 ^ p' * d + n' ≥ x)
  sorry

end least_positive_integer_l110_110923


namespace coin_flips_sequences_count_l110_110731

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110731


namespace sum_of_conjugates_l110_110491

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l110_110491


namespace fixed_point_coordinates_l110_110547

theorem fixed_point_coordinates (a b x y : ℝ) 
  (h1 : a + 2 * b = 1) 
  (h2 : (a * x + 3 * y + b) = 0) :
  x = 1 / 2 ∧ y = -1 / 6 := by
  sorry

end fixed_point_coordinates_l110_110547


namespace find_b_l110_110257

-- Definition of the geometric problem
variables {a b c : ℝ} -- Side lengths of the triangle
variables {area : ℝ} -- Area of the triangle
variables {B : ℝ} -- Angle B in radians

-- Given conditions
def triangle_conditions : Prop :=
  area = sqrt 3 ∧
  B = π / 3 ∧
  a^2 + c^2 = 3 * a * c

-- Statement of the theorem using the given conditions to prove b = 2√2
theorem find_b (h : triangle_conditions) : b = 2 * sqrt 2 := 
  sorry

end find_b_l110_110257


namespace original_two_digit_number_is_52_l110_110816

theorem original_two_digit_number_is_52 (x : ℕ) (h1 : 10 * x + 6 = x + 474) (h2 : 10 ≤ x ∧ x < 100) : x = 52 :=
sorry

end original_two_digit_number_is_52_l110_110816


namespace max_divisions_circle_and_lines_l110_110007

theorem max_divisions_circle_and_lines (n : ℕ) (h₁ : n = 5) : 
  let R_lines := n * (n + 1) / 2 + 1 -- Maximum regions formed by n lines
  let R_circle_lines := 2 * n       -- Additional regions formed by a circle intersecting n lines
  R_lines + R_circle_lines = 26 := by
  sorry

end max_divisions_circle_and_lines_l110_110007


namespace loss_per_metre_l110_110167

theorem loss_per_metre
  (total_metres : ℕ)
  (selling_price : ℕ)
  (cost_price_per_m: ℕ)
  (selling_price_total : selling_price = 18000)
  (cost_price_per_m_def : cost_price_per_m = 95)
  (total_metres_def : total_metres = 200) :
  ((cost_price_per_m * total_metres - selling_price) / total_metres) = 5 :=
by
  sorry

end loss_per_metre_l110_110167


namespace factor_expression_l110_110782

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110782


namespace saline_solution_concentration_l110_110340

theorem saline_solution_concentration
  (C : ℝ) -- concentration of the first saline solution
  (h1 : 3.6 * C + 1.4 * 9 = 5 * 3.24) : -- condition based on the total salt content
  C = 1 := 
sorry

end saline_solution_concentration_l110_110340


namespace thread_length_l110_110481

def side_length : ℕ := 13

def perimeter (s : ℕ) : ℕ := 4 * s

theorem thread_length : perimeter side_length = 52 := by
  sorry

end thread_length_l110_110481


namespace alice_savings_l110_110025

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l110_110025


namespace Diamond_evaluation_l110_110441

-- Redefine the operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^3 - b^2 + 1

-- Statement of the proof
theorem Diamond_evaluation : (Diamond 3 2) = 21 := by
  sorry

end Diamond_evaluation_l110_110441


namespace exponent_simplification_l110_110036

theorem exponent_simplification : (7^3 * (2^5)^3) / (7^2 * 2^(3*3)) = 448 := by
  sorry

end exponent_simplification_l110_110036


namespace min_value_of_function_l110_110633

noncomputable def y (θ : ℝ) : ℝ := (2 - Real.sin θ) / (1 - Real.cos θ)

theorem min_value_of_function : ∃ θ : ℝ, y θ = 3 / 4 :=
sorry

end min_value_of_function_l110_110633


namespace point_inside_circle_l110_110556

theorem point_inside_circle (a : ℝ) :
  ((1 - a) ^ 2 + (1 + a) ^ 2 < 4) → (-1 < a ∧ a < 1) :=
by
  sorry

end point_inside_circle_l110_110556


namespace polynomial_multiple_of_six_l110_110603

theorem polynomial_multiple_of_six 
  (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) :
  6 ∣ P.eval 5 :=
sorry

end polynomial_multiple_of_six_l110_110603


namespace complex_solution_l110_110941

theorem complex_solution (z : ℂ) (h : z^2 = -5 - 12 * Complex.I) :
  z = 2 - 3 * Complex.I ∨ z = -2 + 3 * Complex.I := 
sorry

end complex_solution_l110_110941


namespace improper_fraction_decomposition_l110_110341

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := 
sorry

end improper_fraction_decomposition_l110_110341


namespace car_speed_40_kmph_l110_110006

theorem car_speed_40_kmph (v : ℝ) (h : 1 / v = 1 / 48 + 15 / 3600) : v = 40 := 
sorry

end car_speed_40_kmph_l110_110006


namespace find_least_positive_integer_l110_110927

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l110_110927


namespace least_four_digit_palindrome_divisible_by_11_l110_110665

theorem least_four_digit_palindrome_divisible_by_11 : 
  ∃ (A B : ℕ), (A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ 1000 * A + 100 * B + 10 * B + A = 1111 ∧ (2 * A - 2 * B) % 11 = 0) := 
by
  sorry

end least_four_digit_palindrome_divisible_by_11_l110_110665


namespace max_grain_mass_l110_110465

def platform_length : ℝ := 10
def platform_width : ℝ := 5
def grain_density : ℝ := 1200
def angle_of_repose : ℝ := 45
def max_mass : ℝ := 175000

theorem max_grain_mass :
  let height_of_pile := platform_width / 2
  let volume_of_prism := platform_length * platform_width * height_of_pile
  let volume_of_pyramid := (1 / 3) * (platform_width * height_of_pile) * height_of_pile
  let total_volume := volume_of_prism + 2 * volume_of_pyramid
  let calculated_mass := total_volume * grain_density
  calculated_mass = max_mass :=
by {
  sorry
}

end max_grain_mass_l110_110465


namespace problem_example_l110_110879

def quadratic (eq : Expr) : Prop := ∃ a b c : ℝ, a ≠ 0 ∧ eq = (a * x^2 + b * x + c = 0)

theorem problem_example : quadratic (x^2 + 3x - 5 = 0) :=
by
  sorry

end problem_example_l110_110879


namespace D_96_l110_110409

def D : ℕ → ℕ
| 1       := 0
| n+1     := -- The full definition should be included here, along with any helper functions or constructs to properly define D(n)
sorry

theorem D_96 : D 96 = 112 :=
by
  sorry

end D_96_l110_110409


namespace sum_of_number_and_radical_conjugate_l110_110486

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l110_110486


namespace factor_expression_l110_110776

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110776


namespace coin_flip_sequences_l110_110736

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110736


namespace cubes_and_quartics_sum_l110_110094

theorem cubes_and_quartics_sum (a b : ℝ) (h1 : a + b = 2) (h2 : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 :=
by 
  sorry

end cubes_and_quartics_sum_l110_110094


namespace factory_A_higher_output_l110_110479

theorem factory_A_higher_output (a x : ℝ) (a_pos : a > 0) (x_pos : x > 0) 
  (h_eq_march : 1 + 2 * a = (1 + x) ^ 2) : 
  1 + a > 1 + x :=
by
  sorry

end factory_A_higher_output_l110_110479


namespace largest_x_is_3_l110_110523

noncomputable def largest_value_x (x : ℚ) :=
  (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x = 8 * x - 2

theorem largest_x_is_3 : ∃ x : ℚ, largest_value_x x ∧ ∀ y : ℚ, largest_value_x y → y ≤ 3 :=
begin
  use 3,
  split,
  { -- Proof part skipped
    sorry
  },
  { -- Proof part skipped
    sorry
  }
end

end largest_x_is_3_l110_110523


namespace derivative_log_base2_l110_110534

noncomputable def log_base2 (x : ℝ) := Real.log x / Real.log 2

theorem derivative_log_base2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => log_base2 x) x = 1 / (x * Real.log 2) :=
by
  sorry

end derivative_log_base2_l110_110534


namespace value_of_f_g_10_l110_110231

def g (x : ℤ) : ℤ := 4 * x + 6
def f (x : ℤ) : ℤ := 6 * x - 10

theorem value_of_f_g_10 : f (g 10) = 266 :=
by
  sorry

end value_of_f_g_10_l110_110231


namespace ratio_XZ_ZY_equals_one_l110_110850

theorem ratio_XZ_ZY_equals_one (A : ℕ) (B : ℕ) (C : ℕ) (total_area : ℕ) (area_bisected : ℕ)
  (decagon_area : total_area = 12) (halves_area : area_bisected = 6)
  (above_LZ : A + B = area_bisected) (below_LZ : C + D = area_bisected)
  (symmetry : XZ = ZY) :
  (XZ / ZY = 1) := 
by
  sorry

end ratio_XZ_ZY_equals_one_l110_110850


namespace cos_difference_identity_l110_110390

theorem cos_difference_identity (α β : ℝ) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : Real.sin β = 5 / 13) : Real.cos (α - β) = 63 / 65 := 
by 
  sorry

end cos_difference_identity_l110_110390


namespace least_integer_l110_110932

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l110_110932


namespace selection_num_ways_l110_110336

theorem selection_num_ways : 
  ∀ (volunteers : Finset ℕ) (n k l : ℕ),
  volunteers.card = 5 ∧ n = 2 ∧ k = 2 ∧ l = 3 →
  ∑ (S : Finset ℕ) in volunteers.powerset.filter (λ x, x.card = 2), 
    ∑ (T : Finset ℕ) in (volunteers \ S).powerset.filter (λ x, x.card = 2), 
    (if (S ∩ T = ∅) then 1 else 0) = 30 :=
by
  intros volunteers n k l h
  have hc1 : finset.card volunteers = 5 := h.1
  have hn2 : n = 2 := h.2.1
  have hk2 : k = 2 := h.2.2.1
  have hl3 : l = 3 := h.2.2.2
  sorry

end selection_num_ways_l110_110336


namespace find_side_b_l110_110272

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (h_area : ∃ A B C, 1/2 * a * c * sin B = sqrt 3)
  (h_B : B = π / 3) (h_eq : a ^ 2 + c ^ 2 = 3 * a * c) : b = 2 * sqrt 2 :=
by
  sorry

end find_side_b_l110_110272


namespace gcd_of_numbers_l110_110664

theorem gcd_of_numbers :
  let a := 125^2 + 235^2 + 349^2
  let b := 124^2 + 234^2 + 350^2
  gcd a b = 1 := by
  sorry

end gcd_of_numbers_l110_110664


namespace radical_conjugate_sum_l110_110505

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l110_110505


namespace distinct_sequences_ten_flips_l110_110717

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110717


namespace max_value_q_l110_110281

namespace proof

theorem max_value_q (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end proof

end max_value_q_l110_110281


namespace sum_of_conjugates_l110_110489

theorem sum_of_conjugates (x : ℝ) (h : x = 15 - real.sqrt 500) : 
  (15 - real.sqrt 500) + (15 + real.sqrt 500) = 30 :=
by {
  have h_conjugate : 15 + real.sqrt 500 = 15 + real.sqrt 500,
    from rfl,
  have h_sum := add_eq_of_eq_sub h h_conjugate,
  rw [add_sub_cancel', add_sub_cancel' _ _ 15] at h_sum,
  exact eq_add_of_sub_eq h_sum
}

end sum_of_conjugates_l110_110489


namespace line_equation_l110_110303

theorem line_equation (k : ℝ) (x1 y1 : ℝ) (P : x1 = 1 ∧ y1 = -1) (angle_slope : k = Real.tan (135 * Real.pi / 180)) : 
  ∃ (a b : ℝ), a = -1 ∧ b = -1 ∧ (y1 = k * x1 + b) ∧ (y1 = a * x1 + b) :=
by
  sorry

end line_equation_l110_110303


namespace coplanar_points_scalar_eq_l110_110598

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D O : V) (k : ℝ)

theorem coplanar_points_scalar_eq:
  (3 • (A - O) - 2 • (B - O) + 5 • (C - O) + k • (D - O) = (0 : V)) →
  k = -6 :=
by sorry

end coplanar_points_scalar_eq_l110_110598


namespace least_possible_b_l110_110107

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem least_possible_b (a b : Nat) (h1 : is_prime a) (h2 : is_prime b) (h3 : a + 2 * b = 180) (h4 : a > b) : b = 19 :=
by 
  sorry

end least_possible_b_l110_110107


namespace who_made_a_mistake_l110_110643

-- Definitions of the conditions
def at_least_four_blue_pencils (B : ℕ) : Prop := B ≥ 4
def at_least_five_green_pencils (G : ℕ) : Prop := G ≥ 5
def at_least_three_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 3 ∧ G ≥ 4
def at_least_four_blue_and_four_green_pencils (B G : ℕ) : Prop := B ≥ 4 ∧ G ≥ 4

-- The main theorem stating who made a mistake
theorem who_made_a_mistake (B G : ℕ) 
  (hv : at_least_four_blue_pencils B)
  (hk : at_least_five_green_pencils G)
  (hp : at_least_three_blue_and_four_green_pencils B G)
  (hm : at_least_four_blue_and_four_green_pencils B G) 
  (h_truth : (hv ∧ hk ∧ hp ∧ hm) ∨ (¬hv ∧ hk ∧ hp ∧ hm) ∨ (hv ∧ ¬hk ∧ hp ∧ hm) ∨ (hv ∧ hk ∧ ¬hp ∧ hm) ∨ (hv ∧ hk ∧ hp ∧ ¬hm))
  (h_truthful: ∑ b in [hv, hk, hp, hm], (if b then 1 else 0) = 3) : 
  hk = false := 
sorry

end who_made_a_mistake_l110_110643


namespace distinct_sequences_ten_flips_l110_110719

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110719


namespace matrix_power_difference_l110_110119

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 4;
     0, 1]

theorem matrix_power_difference :
  B^30 - 3 * B^29 = !![-2, 0;
                       0,  2] := 
by
  sorry

end matrix_power_difference_l110_110119


namespace tony_quilt_square_side_length_l110_110870

theorem tony_quilt_square_side_length (length width : ℝ) (h_length : length = 6) (h_width : width = 24) : 
  ∃ s, s * s = length * width ∧ s = 12 :=
by
  sorry

end tony_quilt_square_side_length_l110_110870


namespace valid_range_of_x_l110_110248

theorem valid_range_of_x (x : ℝ) : 3 * x + 5 ≥ 0 → x ≥ -5 / 3 := 
by
  sorry

end valid_range_of_x_l110_110248


namespace factor_expression_l110_110783

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110783


namespace max_u_plus_2v_l110_110979

theorem max_u_plus_2v (u v : ℝ) (h1 : 2 * u + 3 * v ≤ 10) (h2 : 4 * u + v ≤ 9) : u + 2 * v ≤ 6.1 :=
sorry

end max_u_plus_2v_l110_110979


namespace four_four_four_digits_eight_eight_eight_digits_l110_110217

theorem four_four_four_digits_eight_eight_eight_digits (n : ℕ) :
  (4 * (10 ^ (n + 1) - 1) * (10 ^ n) + 8 * (10^n - 1) + 9) = 
  (6 * 10^n + 7) * (6 * 10^n + 7) :=
sorry

end four_four_four_digits_eight_eight_eight_digits_l110_110217


namespace second_divisor_l110_110461

theorem second_divisor (N : ℤ) (k : ℤ) (D : ℤ) (m : ℤ) 
  (h1 : N = 39 * k + 20) 
  (h2 : N = D * m + 7) : 
  D = 13 := sorry

end second_divisor_l110_110461


namespace value_of_x_l110_110536

theorem value_of_x (x : ℝ) (h : 2 ≤ |x - 3| ∧ |x - 3| ≤ 6) : x ∈ Set.Icc (-3 : ℝ) 1 ∪ Set.Icc 5 9 :=
by
  sorry

end value_of_x_l110_110536


namespace max_x5_l110_110412

theorem max_x5 (x1 x2 x3 x4 x5 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) : x5 ≤ 5 :=
  sorry

end max_x5_l110_110412


namespace coin_flip_sequences_l110_110739

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110739


namespace Freddy_journey_time_l110_110192

/-- Eddy and Freddy start simultaneously from city A. Eddy travels to city B, Freddy travels to city C.
    Eddy takes 3 hours from city A to city B, which is 900 km. The distance between city A and city C is
    300 km. The ratio of average speed of Eddy to Freddy is 4:1. Prove that Freddy takes 4 hours to travel. -/
theorem Freddy_journey_time (t_E : ℕ) (d_AB : ℕ) (d_AC : ℕ) (r : ℕ) (V_E V_F t_F : ℕ)
    (h1 : t_E = 3)
    (h2 : d_AB = 900)
    (h3 : d_AC = 300)
    (h4 : r = 4)
    (h5 : V_E = d_AB / t_E)
    (h6 : V_E = r * V_F)
    (h7 : t_F = d_AC / V_F)
  : t_F = 4 := 
  sorry

end Freddy_journey_time_l110_110192


namespace order_magnitudes_ln_subtraction_l110_110218

noncomputable def ln (x : ℝ) : ℝ := Real.log x -- Assuming the natural logarithm definition for real numbers

theorem order_magnitudes_ln_subtraction :
  (ln (3/2) - (3/2)) > (ln 3 - 3) ∧ 
  (ln 3 - 3) > (ln π - π) :=
sorry

end order_magnitudes_ln_subtraction_l110_110218


namespace total_books_arithmetic_sequence_l110_110440

theorem total_books_arithmetic_sequence :
  ∃ (n : ℕ) (a₁ a₂ aₙ d S : ℤ), 
    n = 11 ∧
    a₁ = 32 ∧
    a₂ = 29 ∧
    aₙ = 2 ∧
    d = -3 ∧
    S = (n * (a₁ + aₙ)) / 2 ∧
    S = 187 :=
by sorry

end total_books_arithmetic_sequence_l110_110440


namespace train_speed_l110_110174

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l110_110174


namespace find_x_l110_110394

theorem find_x (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) (h3 : a^3 = 21 * x * 15 * b) : x = 25 :=
by
  -- This is where the proof would go
  sorry

end find_x_l110_110394


namespace calculate_down_payment_l110_110405

theorem calculate_down_payment : 
  let monthly_fee := 12
  let years := 3
  let total_paid := 482
  let num_months := years * 12
  let total_monthly_payments := num_months * monthly_fee
  let down_payment := total_paid - total_monthly_payments
  down_payment = 50 :=
by
  sorry

end calculate_down_payment_l110_110405


namespace divisor_of_1058_l110_110148

theorem divisor_of_1058 :
  ∃ (d : ℕ), (∃ (k : ℕ), 1058 = d * k) ∧ (¬ ∃ (d : ℕ), (∃ (l : ℕ), 1 < d ∧ d < 1058 ∧ 1058 = d * l)) :=
by {
  sorry
}

end divisor_of_1058_l110_110148


namespace tangent_circles_distance_l110_110803

-- Define the radii of the circles.
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 2

-- Define the condition that the circles are tangent.
def tangent (r1 r2 d : ℝ) : Prop :=
  d = r1 + r2 ∨ d = r1 - r2

-- State the theorem.
theorem tangent_circles_distance (d : ℝ) :
  tangent radius_O1 radius_O2 d → (d = 1 ∨ d = 5) :=
by
  sorry

end tangent_circles_distance_l110_110803


namespace sum_of_number_and_radical_conjugate_l110_110487

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l110_110487


namespace smallest_n_inequality_l110_110789

theorem smallest_n_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧
           (∀ m : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) → n ≤ m) ∧
           n = 4 :=
by
  let n := 4
  sorry

end smallest_n_inequality_l110_110789


namespace fraction_of_work_completed_l110_110677

-- Definitions
def work_rate_x : ℚ := 1 / 14
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 25

-- Given the combined work rate and time
def combined_work_rate : ℚ := work_rate_x + work_rate_y + work_rate_z
def time_worked : ℚ := 5

-- The fraction of work completed
def fraction_work_completed : ℚ := combined_work_rate * time_worked

-- Statement to prove
theorem fraction_of_work_completed : fraction_work_completed = 113 / 140 := by
  sorry

end fraction_of_work_completed_l110_110677


namespace one_and_one_third_of_what_number_is_45_l110_110423

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l110_110423


namespace sum_of_number_and_radical_conjugate_l110_110484

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l110_110484


namespace number_of_paths_from_C_to_D_l110_110564

-- Define the grid and positions
def C := (0,0)  -- Bottom-left corner
def D := (7,3)  -- Top-right corner
def gridWidth : ℕ := 7
def gridHeight : ℕ := 3

-- Define the binomial coefficient function
-- Note: Lean already has binomial coefficient defined in Mathlib, use Nat.choose for that

-- The statement to prove
theorem number_of_paths_from_C_to_D : Nat.choose (gridWidth + gridHeight) gridHeight = 120 :=
by
  sorry

end number_of_paths_from_C_to_D_l110_110564


namespace find_n_l110_110476

-- Define the conditions as hypothesis
variables (A B n : ℕ)

-- Hypothesis 1: This year, Ana's age is the square of Bonita's age.
-- A = B^2
#check (A = B^2) 

-- Hypothesis 2: Last year Ana was 5 times as old as Bonita.
-- A - 1 = 5 * (B - 1)
#check (A - 1 = 5 * (B - 1))

-- Hypothesis 3: Ana and Bonita were born n years apart.
-- A = B + n
#check (A = B + n)

-- Goal: The difference in their ages, n, should be 12.
theorem find_n (A B n : ℕ) (h1 : A = B^2) (h2 : A - 1 = 5 * (B - 1)) (h3 : A = B + n) : n = 12 :=
sorry

end find_n_l110_110476


namespace triangle_area_ab_l110_110962

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : ∀ (x y : ℝ), a * x + b * y = 6) (harea : (1/2) * (6 / a) * (6 / b) = 6) : 
  a * b = 3 := 
by sorry

end triangle_area_ab_l110_110962


namespace circle_radius_l110_110682

theorem circle_radius (r M N : ℝ) (hM : M = π * r^2) (hN : N = 2 * π * r) (hRatio : M / N = 20) : r = 40 := 
by
  sorry

end circle_radius_l110_110682


namespace coin_flip_sequences_l110_110702

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110702


namespace number_of_laborers_in_crew_l110_110865

theorem number_of_laborers_in_crew (present : ℕ) (percentage : ℝ) (total : ℕ) 
    (h1 : present = 70) (h2 : percentage = 44.9 / 100) (h3 : present = percentage * total) : 
    total = 156 := 
sorry

end number_of_laborers_in_crew_l110_110865


namespace geometric_sequence_sum_four_l110_110791

theorem geometric_sequence_sum_four (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h2 : q ≠ 1)
  (h3 : -3 * a 0 = -2 * a 1 - a 2)
  (h4 : a 0 = 1) : 
  S 4 = -20 :=
sorry

end geometric_sequence_sum_four_l110_110791


namespace cube_root_3375_l110_110325

theorem cube_root_3375 (c d : ℕ) (h1 : c > 0 ∧ d > 0) (h2 : c * d^3 = 3375) (h3 : ∀ k : ℕ, k > 0 → c * (d / k)^3 ≠ 3375) : 
  c + d = 16 :=
sorry

end cube_root_3375_l110_110325


namespace coeff_div_binom_eq_4_l110_110284

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def coeff_x5_expansion : ℚ :=
  binomial 8 2 * (-2) ^ 2

def binomial_coeff : ℚ :=
  binomial 8 2

theorem coeff_div_binom_eq_4 : 
  (coeff_x5_expansion / binomial_coeff) = 4 := by
  sorry

end coeff_div_binom_eq_4_l110_110284


namespace part_I_part_II_l110_110376

noncomputable def general_term (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (a 2 = 1 ∧ ∀ n, a (n + 1) - a n = d) ∧
  (d ≠ 0 ∧ (a 3)^2 = (a 2) * (a 6))

theorem part_I (a : ℕ → ℤ) (d : ℤ) : general_term a d → 
  ∀ n, a n = 2 * n - 3 := 
sorry

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : Prop :=
  (∀ n, S n = n * (a 1 + a n) / 2) ∧ 
  (general_term a d)

theorem part_II (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : sum_of_first_n_terms a d S → 
  ∃ n, n > 7 ∧ S n > 35 :=
sorry

end part_I_part_II_l110_110376


namespace efficiency_ratio_l110_110680

-- Define the work efficiencies
def EA : ℚ := 1 / 12
def EB : ℚ := 1 / 24
def EAB : ℚ := 1 / 8

-- State the theorem
theorem efficiency_ratio (EAB_eq : EAB = EA + EB) : (EA / EB) = 2 := by
  -- Insert proof here
  sorry

end efficiency_ratio_l110_110680


namespace largest_possible_e_l110_110831

noncomputable def diameter := (2 : ℝ)
noncomputable def PX := (4 / 5 : ℝ)
noncomputable def PY := (3 / 4 : ℝ)
noncomputable def e := (41 - 16 * Real.sqrt 25 : ℝ)
noncomputable def u := 41
noncomputable def v := 16
noncomputable def w := 25

theorem largest_possible_e (P Q X Y Z R S : Real) (d : diameter = 2)
  (PX_len : P - X = 4/5) (PY_len : P - Y = 3/4)
  (e_def : e = 41 - 16 * Real.sqrt 25)
  : u + v + w = 82 :=
by
  sorry

end largest_possible_e_l110_110831


namespace percentage_increase_in_savings_l110_110988

theorem percentage_increase_in_savings (I : ℝ) (hI : 0 < I) :
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  ((S_new - S) / S) * 100 = 50 :=
by
  let E := 0.75 * I
  let S := I - E
  let I_new := 1.20 * I
  let E_new := 0.825 * I
  let S_new := I_new - E_new
  sorry

end percentage_increase_in_savings_l110_110988


namespace intersection_A_B_union_A_B_range_of_a_l110_110222

open Set

-- Definitions for the given sets
def Universal : Set ℝ := univ
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 6}

-- Propositions to prove
theorem intersection_A_B : 
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7} := 
  sorry

theorem union_A_B : 
  A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := 
  sorry

theorem range_of_a (a : ℝ) : 
  (A ∪ C a = C a) → (2 ≤ a ∧ a < 3) := 
  sorry

end intersection_A_B_union_A_B_range_of_a_l110_110222


namespace temperature_at_tian_du_peak_height_of_mountain_peak_l110_110616

-- Problem 1: Temperature at the top of Tian Du Peak
theorem temperature_at_tian_du_peak
  (height : ℝ) (drop_rate : ℝ) (initial_temp : ℝ)
  (H : height = 1800) (D : drop_rate = 0.6) (I : initial_temp = 18) :
  (initial_temp - (height / 100 * drop_rate)) = 7.2 :=
by
  sorry

-- Problem 2: Height of the mountain peak
theorem height_of_mountain_peak
  (drop_rate : ℝ) (foot_temp top_temp : ℝ)
  (D : drop_rate = 0.6) (F : foot_temp = 10) (T : top_temp = -8) :
  (foot_temp - top_temp) / drop_rate * 100 = 3000 :=
by
  sorry

end temperature_at_tian_du_peak_height_of_mountain_peak_l110_110616


namespace jade_transactions_correct_l110_110676

-- Definitions for the conditions
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions * 10 / 100)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := cal_transactions + 16

-- The theorem stating what we want to prove
theorem jade_transactions_correct : jade_transactions = 82 := by
  sorry

end jade_transactions_correct_l110_110676


namespace students_in_miss_evans_class_l110_110969

theorem students_in_miss_evans_class
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (contribution_per_student : ℕ)
  (remaining_contribution : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : contribution_per_student = 4)
  (h4 : remaining_contribution = total_contribution - class_funds)
  (h5 : num_students = remaining_contribution / contribution_per_student)
  : num_students = 19 :=
sorry

end students_in_miss_evans_class_l110_110969


namespace dealer_purchase_fraction_l110_110750

theorem dealer_purchase_fraction (P C : ℝ) (h1 : ∃ S, S = 1.5 * P) (h2 : ∃ S, S = 2 * C) :
  C / P = 3 / 8 :=
by
  -- The statement of the theorem has been generated based on the problem conditions.
  sorry

end dealer_purchase_fraction_l110_110750


namespace find_d_share_l110_110896

def money_distribution (a b c d : ℕ) (x : ℕ) := 
  a = 5 * x ∧ 
  b = 2 * x ∧ 
  c = 4 * x ∧ 
  d = 3 * x ∧ 
  (c = d + 500)

theorem find_d_share (a b c d x : ℕ) (h : money_distribution a b c d x) : d = 1500 :=
by
  --proof would go here
  sorry

end find_d_share_l110_110896


namespace inequality_I_l110_110034

theorem inequality_I (a b x y : ℝ) (hx : x < a) (hy : y < b) : x * y < a * b :=
sorry

end inequality_I_l110_110034


namespace domain_of_sqrt_expr_l110_110787

theorem domain_of_sqrt_expr (x : ℝ) : x ≥ 3 ∧ x < 8 ↔ x ∈ Set.Ico 3 8 :=
by
  sorry

end domain_of_sqrt_expr_l110_110787


namespace probability_laurent_greater_chloe_l110_110185

open ProbabilityTheory

noncomputable def probability_greater (x y : ℝ) (hx : x ∈ Set.Icc 0 2017) 
  (hy : y ∈ Set.Icc 0 4034) : ℝ := 
OfReal (MeasureTheory.Measure.prod
  (MeasureTheory.Measure.restrict) x hx)
  (MeasureTheory.Measure.restrict (ofReal y hy))
  {p : ℝ × ℝ | p.2 > p.1}

theorem probability_laurent_greater_chloe :
  (probability_greater x y hx hy = (¾ : ℝ)) := 
sorry

end probability_laurent_greater_chloe_l110_110185


namespace coin_flip_sequences_l110_110697

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110697


namespace find_number_l110_110422

-- Define the conditions
variables (x : ℝ)
axiom condition : (4/3) * x = 45

-- Prove the main statement
theorem find_number : x = 135 / 4 :=
by
  sorry

end find_number_l110_110422


namespace relationship_among_g_a_0_f_b_l110_110414

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x + x^2 - 3

theorem relationship_among_g_a_0_f_b (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) : g a < 0 ∧ 0 < f b :=
by
  -- Function properties are non-trivial and are omitted.
  sorry

end relationship_among_g_a_0_f_b_l110_110414


namespace h_of_j_of_3_l110_110091

def h (x : ℝ) : ℝ := 4 * x + 3
def j (x : ℝ) : ℝ := (x + 2) ^ 2

theorem h_of_j_of_3 : h (j 3) = 103 := by
  sorry

end h_of_j_of_3_l110_110091


namespace find_n_l110_110240

theorem find_n (n : ℕ) (x y a b : ℕ) (hx : x = 1) (hy : y = 1) (ha : a = 1) (hb : b = 1)
  (h : (x + 3 * y) ^ n = (7 * a + b) ^ 10) : n = 5 :=
by
  sorry

end find_n_l110_110240


namespace part1_part2_l110_110457

theorem part1 : (π - 3)^0 + (-1)^(2023) - Real.sqrt 8 = -2 * Real.sqrt 2 := sorry

theorem part2 (x : ℝ) : (4 * x - 3 > 9) ∧ (2 + x ≥ 0) ↔ x > 3 := sorry

end part1_part2_l110_110457


namespace area_triangle_ABC_l110_110826

-- Definitions of the lengths and height
def BD : ℝ := 3
def DC : ℝ := 2 * BD
def BC : ℝ := BD + DC
def h_A_BC : ℝ := 4

-- The triangle area formula
def areaOfTriangle (base height : ℝ) : ℝ := 0.5 * base * height

-- The goal to prove that the area of triangle ABC is 18 square units
theorem area_triangle_ABC : areaOfTriangle BC h_A_BC = 18 := by
  sorry

end area_triangle_ABC_l110_110826


namespace max_b_value_l110_110798

theorem max_b_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : b ≤ 1 / 3 :=
  sorry

end max_b_value_l110_110798


namespace regression_line_l110_110953

theorem regression_line (x y : ℝ) (m : ℝ) (x1 y1 : ℝ)
  (h_slope : m = 6.5)
  (h_point : (x1, y1) = (2, 3)) :
  (y - y1) = m * (x - x1) ↔ y = 6.5 * x - 10 :=
by
  sorry

end regression_line_l110_110953


namespace factor_expression_l110_110777

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110777


namespace original_price_of_sarees_l110_110139

theorem original_price_of_sarees (P : ℝ) (h : 0.95 * 0.80 * P = 456) : P = 600 :=
by
  sorry

end original_price_of_sarees_l110_110139


namespace sin_x1_sub_x2_l110_110557

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem sin_x1_sub_x2 (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < Real.pi)
  (h₄ : f x₁ = 1 / 3) (h₅ : f x₂ = 1 / 3) : 
  Real.sin (x₁ - x₂) = - (2 * Real.sqrt 2) / 3 := 
sorry

end sin_x1_sub_x2_l110_110557


namespace model_car_cost_l110_110609

theorem model_car_cost (x : ℕ) :
  (5 * x) + (5 * 10) + (5 * 2) = 160 → x = 20 :=
by
  intro h
  sorry

end model_car_cost_l110_110609


namespace log_inequality_l110_110799

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end log_inequality_l110_110799


namespace quadratic_real_roots_l110_110950

theorem quadratic_real_roots (k : ℝ) : 
  (∀ x : ℝ, (2 * x^2 + 4 * x + k - 1 = 0) → ∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → 
  k ≤ 3 :=
by
  intro h
  have h_discriminant : 16 - 8 * k >= 0 := sorry
  linarith

end quadratic_real_roots_l110_110950


namespace min_ab_l110_110145

theorem min_ab {a b : ℝ} (h1 : (a^2) * (-b) + (a^2 + 1) = 0) : |a * b| = 2 :=
sorry

end min_ab_l110_110145


namespace one_and_one_third_of_what_number_is_45_l110_110424

theorem one_and_one_third_of_what_number_is_45 (x : ℚ) (h : (4 / 3) * x = 45) : x = 33.75 :=
by
  sorry

end one_and_one_third_of_what_number_is_45_l110_110424


namespace factorization_l110_110361

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := sorry

end factorization_l110_110361


namespace min_volume_for_cone_l110_110158

noncomputable def min_cone_volume (V1 : ℝ) : Prop :=
  ∀ V2 : ℝ, (V1 = 1) → 
    V2 ≥ (4 / 3)

-- The statement without proof
theorem min_volume_for_cone : 
  min_cone_volume 1 :=
sorry

end min_volume_for_cone_l110_110158


namespace positive_integers_sum_digits_less_than_9000_l110_110150

theorem positive_integers_sum_digits_less_than_9000 : 
  ∃ n : ℕ, n = 47 ∧ ∀ x : ℕ, (1 ≤ x ∧ x < 9000 ∧ (Nat.digits 10 x).sum = 5) → (Nat.digits 10 x).length = n :=
sorry

end positive_integers_sum_digits_less_than_9000_l110_110150


namespace variance_daily_reading_time_l110_110156

theorem variance_daily_reading_time :
  let mean10 := 2.7
  let var10 := 1
  let num10 := 800

  let mean11 := 3.1
  let var11 := 2
  let num11 := 600

  let mean12 := 3.3
  let var12 := 3
  let num12 := 600

  let num_total := num10 + num11 + num12

  let total_mean := (2.7 * 800 + 3.1 * 600 + 3.3 * 600) / 2000

  let var_total := (800 / 2000) * (1 + (2.7 - total_mean)^2) +
                   (600 / 2000) * (2 + (3.1 - total_mean)^2) +
                   (600 / 2000) * (3 + (3.3 - total_mean)^2)

  var_total = 1.966 :=
by
  sorry

end variance_daily_reading_time_l110_110156


namespace ratio_of_cats_to_dogs_sold_l110_110901

theorem ratio_of_cats_to_dogs_sold (cats dogs : ℕ) (h1 : cats = 16) (h2 : dogs = 8) :
  (cats : ℚ) / dogs = 2 / 1 :=
by
  sorry

end ratio_of_cats_to_dogs_sold_l110_110901


namespace distinct_sequences_ten_flips_l110_110718

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110718


namespace exist_x_y_l110_110407

theorem exist_x_y (a b c : ℝ) (h₁ : abs a > 2) (h₂ : a^2 + b^2 + c^2 = a * b * c + 4) :
  ∃ x y : ℝ, a = x + 1/x ∧ b = y + 1/y ∧ c = x*y + 1/(x*y) :=
sorry

end exist_x_y_l110_110407


namespace factor_expression_l110_110775

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l110_110775


namespace mary_total_nickels_l110_110614

theorem mary_total_nickels : (7 + 12 + 9 = 28) :=
by
  sorry

end mary_total_nickels_l110_110614


namespace find_angle_A_l110_110820

theorem find_angle_A (A B C a b c : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : B = (A + C) / 2)
  (h3 : 2 * b ^ 2 = 3 * a * c) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l110_110820


namespace maximum_rubles_l110_110863

-- We define the initial number of '1' and '2' cards
def num_ones : ℕ := 2013
def num_twos : ℕ := 2013
def total_digits : ℕ := num_ones + num_twos

-- Definition of the problem statement
def problem_statement : Prop :=
  ∃ (max_rubles : ℕ), 
    max_rubles = 5 ∧
    ∀ (current_k : ℕ), 
      current_k = 5 → 
      ∃ (moves : ℕ), 
        moves ≤ max_rubles ∧
        (current_k - moves * 2) % 11 = 0

-- The expected solution is proving the maximum rubles is 5
theorem maximum_rubles : problem_statement :=
by
  sorry

end maximum_rubles_l110_110863


namespace problem_solution_l110_110030

variable (x y : ℝ)

theorem problem_solution :
  (x - y + 1) * (x - y - 1) = x^2 - 2 * x * y + y^2 - 1 :=
by
  sorry

end problem_solution_l110_110030


namespace negation_of_universal_quantifier_proposition_l110_110856

variable (x : ℝ)

theorem negation_of_universal_quantifier_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1/4 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1/4 < 0) :=
sorry

end negation_of_universal_quantifier_proposition_l110_110856


namespace g_at_5_l110_110834

def g (x : ℝ) : ℝ := 3 * x ^ 4 - 22 * x ^ 3 + 47 * x ^ 2 - 44 * x + 24

theorem g_at_5 : g 5 = 104 := by
  sorry

end g_at_5_l110_110834


namespace percent_decrease_is_20_l110_110674

/-- Define the original price and sale price as constants. -/
def P_original : ℕ := 100
def P_sale : ℕ := 80

/-- Define the formula for percent decrease. -/
def percent_decrease (P_original P_sale : ℕ) : ℕ :=
  ((P_original - P_sale) * 100) / P_original

/-- Prove that the percent decrease is 20%. -/
theorem percent_decrease_is_20 : percent_decrease P_original P_sale = 20 :=
by
  sorry

end percent_decrease_is_20_l110_110674


namespace intersection_A_B_eq_complement_union_eq_subset_condition_l110_110087

open Set

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | x > 3 / 2}
noncomputable def C (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

theorem intersection_A_B_eq : A ∩ B = {x : ℝ | 3 / 2 < x ∧ x ≤ 3} :=
by sorry

theorem complement_union_eq : (univ \ B) ∪ A = {x : ℝ | x ≤ 3} :=
by sorry

theorem subset_condition (a : ℝ) : (C a ⊆ A) → (a ≤ 3) :=
by sorry

end intersection_A_B_eq_complement_union_eq_subset_condition_l110_110087


namespace clock_angle_3_45_l110_110322

theorem clock_angle_3_45 :
  let minute_angle := 45 * 6 -- in degrees
  let hour_angle := (3 * 30) + (45 * 0.5) -- in degrees
  let angle_difference := abs (hour_angle - minute_angle)
  let smaller_angle := if angle_difference <= 180 then angle_difference else 360 - angle_difference
  smaller_angle = 202.5 :=
by
  let minute_angle := 45 * 6
  let hour_angle := (3 * 30) + (45 * 0.5)
  let angle_difference := abs (hour_angle - minute_angle)
  let smaller_angle := if angle_difference <= 180 then angle_difference else 360 - angle_difference
  sorry

end clock_angle_3_45_l110_110322


namespace inequality_of_abc_l110_110292

theorem inequality_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
sorry

end inequality_of_abc_l110_110292


namespace employee_payments_l110_110446

noncomputable def amount_paid_to_Y : ℝ := 934 / 3
noncomputable def amount_paid_to_X : ℝ := 1.20 * amount_paid_to_Y
noncomputable def amount_paid_to_Z : ℝ := 0.80 * amount_paid_to_Y

theorem employee_payments :
  amount_paid_to_X + amount_paid_to_Y + amount_paid_to_Z = 934 :=
by
  sorry

end employee_payments_l110_110446


namespace triangle_perpendicular_bisector_properties_l110_110550

variables {A B C A1 A2 B1 B2 C1 C2 : Type} (triangle : triangle A B C)
  (A1_perpendicular : dropping_perpendicular_to_bisector A )
  (A2_perpendicular : dropping_perpendicular_to_bisector A )
  (B1_perpendicular : dropping_perpendicular_to_bisector B )
  (B2_perpendicular : dropping_perpendicular_to_bisector B )
  (C1_perpendicular : dropping_perpendicular_to_bisector C )
  (C2_perpendicular : dropping_perpendicular_to_bisector C )
  
-- Defining required structures
structure triangle (A B C : Type) :=
  (AB BC CA : ℝ)

structure dropping_perpendicular_to_bisector (v : Type) :=
  (perpendicular_to_bisector : ℝ)

namespace triangle_properties

theorem triangle_perpendicular_bisector_properties :
  2 * (A1_perpendicular.perpendicular_to_bisector + A2_perpendicular.perpendicular_to_bisector + 
       B1_perpendicular.perpendicular_to_bisector + B2_perpendicular.perpendicular_to_bisector + 
       C1_perpendicular.perpendicular_to_bisector + C2_perpendicular.perpendicular_to_bisector) = 
  (triangle.AB + triangle.BC + triangle.CA) :=
sorry

end triangle_properties

end triangle_perpendicular_bisector_properties_l110_110550


namespace lazy_worker_days_worked_l110_110160

theorem lazy_worker_days_worked :
  ∃ x : ℕ, 24 * x - 6 * (30 - x) = 0 ∧ x = 6 :=
by
  existsi 6
  sorry

end lazy_worker_days_worked_l110_110160


namespace solution_of_abs_square_inequality_l110_110305

def solution_set := {x : ℝ | (1 ≤ x ∧ x ≤ 3) ∨ x = -2}

theorem solution_of_abs_square_inequality (x : ℝ) :
  (abs (x^2 - 4) ≤ x + 2) ↔ (x ∈ solution_set) :=
by
  sorry

end solution_of_abs_square_inequality_l110_110305


namespace coin_flip_sequences_l110_110703

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110703


namespace train_speed_l110_110173

theorem train_speed (length : Nat) (time_sec : Nat) (length_km : length = 200)
  (time_hr : time_sec = 12) : (200 : ℝ) / (12 / 3600 : ℝ) = 60 :=
by
  -- Proof steps will go here
  sorry

end train_speed_l110_110173


namespace evaluate_g_at_3_l110_110813

def g : ℝ → ℝ := fun x => x^2 - 3 * x + 2

theorem evaluate_g_at_3 : g 3 = 2 := by
  sorry

end evaluate_g_at_3_l110_110813


namespace max_capacity_tank_l110_110746

-- Definitions of the conditions
def water_loss_1 := 32000 * 5
def water_loss_2 := 10000 * 10
def total_loss := water_loss_1 + water_loss_2
def water_added := 40000 * 3
def missing_water := 140000

-- Definition of the maximum capacity
def max_capacity := total_loss + water_added + missing_water

-- The theorem to prove
theorem max_capacity_tank : max_capacity = 520000 := by
  sorry

end max_capacity_tank_l110_110746


namespace students_take_neither_l110_110968

variable (Total Mathematic Physics Both MathPhysics ChemistryNeither Neither : ℕ)

axiom Total_students : Total = 80
axiom students_mathematics : Mathematic = 50
axiom students_physics : Physics = 40
axiom students_both : Both = 25
axiom students_chemistry_neither : ChemistryNeither = 10

theorem students_take_neither :
  Neither = Total - (Mathematic - Both + Physics - Both + Both + ChemistryNeither) :=
  by
  have Total_students := Total_students
  have students_mathematics := students_mathematics
  have students_physics := students_physics
  have students_both := students_both
  have students_chemistry_neither := students_chemistry_neither
  sorry

end students_take_neither_l110_110968


namespace stock_percentage_change_l110_110184

theorem stock_percentage_change :
  let initial_value := 100
  let value_after_first_day := initial_value * (1 - 0.25)
  let value_after_second_day := value_after_first_day * (1 + 0.35)
  let final_value := value_after_second_day * (1 - 0.15)
  let overall_percentage_change := ((final_value - initial_value) / initial_value) * 100
  overall_percentage_change = -13.9375 := 
by
  sorry

end stock_percentage_change_l110_110184


namespace total_number_of_coins_l110_110395

theorem total_number_of_coins (n : ℕ) (h : 4 * n - 4 = 240) : n^2 = 3721 :=
by
  sorry

end total_number_of_coins_l110_110395


namespace value_of_c_l110_110134

theorem value_of_c (c : ℝ) :
  (∀ x y : ℝ, (x, y) = ((2 + 8) / 2, (6 + 10) / 2) → x + y = c) → c = 13 :=
by
  -- Placeholder for proof
  sorry

end value_of_c_l110_110134


namespace value_of_x_squared_plus_y_squared_l110_110963

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x^2 = 8 * x + y) (h2 : y^2 = x + 8 * y) (h3 : x ≠ y) : 
  x^2 + y^2 = 63 := sorry

end value_of_x_squared_plus_y_squared_l110_110963


namespace smallest_a_l110_110977

-- Define the conditions and the proof goal
theorem smallest_a (a b : ℝ) (h₁ : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (15 * (x : ℝ))) (h₂ : 0 ≤ a) (h₃ : 0 ≤ b) :
  a = 15 :=
sorry

end smallest_a_l110_110977


namespace coin_flip_sequences_l110_110742

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110742


namespace fraction_videocassette_recorders_l110_110112

variable (H : ℝ) (F : ℝ)

-- Conditions
variable (cable_TV_frac : ℝ := 1 / 5)
variable (both_frac : ℝ := 1 / 20)
variable (neither_frac : ℝ := 0.75)

-- Main theorem statement
theorem fraction_videocassette_recorders (H_pos : 0 < H) 
  (cable_tv : cable_TV_frac * H > 0)
  (both : both_frac * H > 0) 
  (neither : neither_frac * H > 0) :
  F = 1 / 10 :=
by
  sorry

end fraction_videocassette_recorders_l110_110112


namespace quadratic_real_roots_k_leq_one_l110_110097

theorem quadratic_real_roots_k_leq_one (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 :=
by
  sorry

end quadratic_real_roots_k_leq_one_l110_110097


namespace problem_l110_110804

theorem problem (x : ℝ) (h : 15 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 17 := 
by sorry

end problem_l110_110804


namespace coin_flip_sequences_l110_110689

theorem coin_flip_sequences (n : ℕ) (h : n = 10) : (2 ^ n) = 1024 := by
  rw [h]
  -- 1024 is 2 ^ 10
  norm_num

end coin_flip_sequences_l110_110689


namespace productivity_increase_l110_110640

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : (7/8) * b * (1 + x / 100) = 1.05 * b)

theorem productivity_increase (x : ℝ) : x = 20 := sorry

end productivity_increase_l110_110640


namespace sum_radical_conjugate_l110_110498

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l110_110498


namespace coin_flips_sequences_count_l110_110730

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110730


namespace cherie_sparklers_count_l110_110117

-- Conditions
def koby_boxes : ℕ := 2
def koby_sparklers_per_box : ℕ := 3
def koby_whistlers_per_box : ℕ := 5
def cherie_boxes : ℕ := 1
def cherie_whistlers : ℕ := 9
def total_fireworks : ℕ := 33

-- Total number of fireworks Koby has
def koby_total_fireworks : ℕ :=
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box)

-- Total number of fireworks Cherie has
def cherie_total_fireworks : ℕ :=
  total_fireworks - koby_total_fireworks

-- Number of sparklers in Cherie's box
def cherie_sparklers : ℕ :=
  cherie_total_fireworks - cherie_whistlers

-- Proof statement
theorem cherie_sparklers_count : cherie_sparklers = 8 := by
  sorry

end cherie_sparklers_count_l110_110117


namespace xiaohong_total_score_l110_110157

theorem xiaohong_total_score :
  ∀ (midterm_score final_score : ℕ) (midterm_weight final_weight : ℝ),
    midterm_score = 80 →
    final_score = 90 →
    midterm_weight = 0.4 →
    final_weight = 0.6 →
    (midterm_score * midterm_weight + final_score * final_weight) = 86 :=
by
  intros midterm_score final_score midterm_weight final_weight
  intros h1 h2 h3 h4
  sorry

end xiaohong_total_score_l110_110157


namespace no_solution_for_floor_x_plus_x_eq_15_point_3_l110_110917

theorem no_solution_for_floor_x_plus_x_eq_15_point_3 : ¬ ∃ (x : ℝ), (⌊x⌋ : ℝ) + x = 15.3 := by
  sorry

end no_solution_for_floor_x_plus_x_eq_15_point_3_l110_110917


namespace new_average_is_minus_one_l110_110301

noncomputable def new_average_of_deducted_sequence : ℤ :=
  let n := 15
  let avg := 20
  let seq_sum := n * avg
  let x := (seq_sum - (n * (n-1) / 2)) / n
  let deductions := (n-1) * n * 3 / 2
  let new_sum := seq_sum - deductions
  new_sum / n

theorem new_average_is_minus_one : new_average_of_deducted_sequence = -1 := 
  sorry

end new_average_is_minus_one_l110_110301


namespace distinct_sequences_ten_flips_l110_110716

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110716


namespace abigail_money_left_l110_110759

def initial_amount : ℕ := 11
def spent_in_store : ℕ := 2
def amount_lost : ℕ := 6

theorem abigail_money_left :
  initial_amount - spent_in_store - amount_lost = 3 := 
by {
  sorry
}

end abigail_money_left_l110_110759


namespace range_of_m_l110_110060

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l110_110060


namespace alice_savings_l110_110023

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l110_110023


namespace area_formed_by_curve_and_line_l110_110433

noncomputable def area_under_curve (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, (f x - g x)

theorem area_formed_by_curve_and_line :
  area_under_curve (λ x, x) (λ x, x^3 - 3 * x) 0 2 * 2 = 8 := by
  sorry

end area_formed_by_curve_and_line_l110_110433


namespace find_original_price_l110_110008

theorem find_original_price (SP GP : ℝ) (h_SP : SP = 1150) (h_GP : GP = 27.77777777777778) :
  ∃ CP : ℝ, CP = 900 :=
by
  sorry

end find_original_price_l110_110008


namespace x_lt_2_necessary_not_sufficient_x_sq_lt_4_l110_110333

theorem x_lt_2_necessary_not_sufficient_x_sq_lt_4 (x : ℝ) :
  (x < 2) → (x^2 < 4) ∧ ¬((x^2 < 4) → (x < 2)) :=
by
  sorry

end x_lt_2_necessary_not_sufficient_x_sq_lt_4_l110_110333


namespace coin_flip_sequences_l110_110696

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110696


namespace sum_of_arithmetic_sequence_l110_110970

variable {a : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a_8_eq_8 : a 8 = 8

-- Proof problem statement
theorem sum_of_arithmetic_sequence :
  is_arithmetic_sequence a →
  a 8 = 8 →
  (∑ i in Finset.range 15, a (i + 1)) = 120 :=
by
  intros h_as h_a8
  sorry

end sum_of_arithmetic_sequence_l110_110970


namespace coin_flip_sequences_l110_110733

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110733


namespace coin_flip_sequences_l110_110745

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110745


namespace calculate_f_of_g_l110_110230

def g (x : ℝ) := 4 * x + 6
def f (x : ℝ) := 6 * x - 10

theorem calculate_f_of_g :
  f (g 10) = 266 := by
  sorry

end calculate_f_of_g_l110_110230


namespace fourth_person_height_l110_110307

noncomputable def height_of_fourth_person (H : ℕ) : ℕ := 
  let second_person := H + 2
  let third_person := H + 4
  let fourth_person := H + 10
  fourth_person

theorem fourth_person_height {H : ℕ} 
  (cond1 : 2 = 2)
  (cond2 : 6 = 6)
  (average_height : 76 = 76) 
  (height_sum : H + (H + 2) + (H + 4) + (H + 10) = 304) : 
  height_of_fourth_person H = 82 := sorry

end fourth_person_height_l110_110307


namespace youngest_sibling_age_l110_110444

theorem youngest_sibling_age
    (age_youngest : ℕ)
    (first_sibling : ℕ := age_youngest + 4)
    (second_sibling : ℕ := age_youngest + 5)
    (third_sibling : ℕ := age_youngest + 7)
    (average_age : ℕ := 21)
    (sum_of_ages : ℕ := 4 * average_age)
    (total_age_check : (age_youngest + first_sibling + second_sibling + third_sibling) = sum_of_ages) :
  age_youngest = 17 :=
sorry

end youngest_sibling_age_l110_110444


namespace sum_S16_over_S4_l110_110206

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a q : α) (n : ℕ) := a * q^n

def sum_of_first_n_terms (a q : α) (n : ℕ) : α :=
if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem sum_S16_over_S4
  (a q : α)
  (hq : q ≠ 1)
  (h8_over_4 : sum_of_first_n_terms a q 8 / sum_of_first_n_terms a q 4 = 3) :
  sum_of_first_n_terms a q 16 / sum_of_first_n_terms a q 4 = 15 :=
sorry

end sum_S16_over_S4_l110_110206


namespace bc_sum_condition_l110_110811

-- Define the conditions as Lean definitions
def is_positive_integer (n : ℕ) : Prop := n > 0
def not_equal_to (x y : ℕ) : Prop := x ≠ y
def less_than_or_equal_to_nine (n : ℕ) : Prop := n ≤ 9

-- Main proof statement
theorem bc_sum_condition (a b c : ℕ) (h_pos_a : is_positive_integer a) (h_pos_b : is_positive_integer b) (h_pos_c : is_positive_integer c)
  (h_a_not_1 : a ≠ 1) (h_b_not_c : b ≠ c) (h_b_le_9 : less_than_or_equal_to_nine b) (h_c_le_9 : less_than_or_equal_to_nine c)
  (h_eq : (10 * a + b) * (10 * a + c) = 100 * a * a + 110 * a + b * c) :
  b + c = 11 := by
  sorry

end bc_sum_condition_l110_110811


namespace students_count_l110_110770

theorem students_count (x y : ℕ) (h1 : 3 * x + 20 = y) (h2 : 4 * x - 25 = y) : x = 45 :=
by {
  sorry
}

end students_count_l110_110770


namespace basketball_team_heights_l110_110987

theorem basketball_team_heights :
  ∃ (second tallest third fourth shortest : ℝ),
  (tallest = 80.5 ∧
   second = tallest - 6.25 ∧
   third = second - 3.75 ∧
   fourth = third - 5.5 ∧
   shortest = fourth - 4.8 ∧
   second = 74.25 ∧
   third = 70.5 ∧
   fourth = 65 ∧
   shortest = 60.2) := sorry

end basketball_team_heights_l110_110987


namespace find_m_l110_110452

theorem find_m (n m : ℕ) (h1 : m = 13 * n + 8) (h2 : m = 15 * n) : m = 60 :=
  sorry

end find_m_l110_110452


namespace B_completion_days_l110_110888

theorem B_completion_days :
  (∃ x : ℚ, (3/14 + 1/x) + (41/(14*x)) = 1) → x = 5 := 
begin
  intro h,
  rcases h with ⟨x, hx⟩,
  -- Sorry to skip the proof steps
  sorry
end

end B_completion_days_l110_110888


namespace new_pressure_of_transferred_gas_l110_110762

theorem new_pressure_of_transferred_gas (V1 V2 : ℝ) (p1 k : ℝ) 
  (h1 : V1 = 3.5) (h2 : p1 = 8) (h3 : k = V1 * p1) (h4 : V2 = 7) :
  ∃ p2 : ℝ, p2 = 4 ∧ k = V2 * p2 :=
by
  use 4
  sorry

end new_pressure_of_transferred_gas_l110_110762


namespace train_speed_60_kmph_l110_110170

theorem train_speed_60_kmph (length_train : ℕ) (time_to_cross : ℕ) 
  (h_length : length_train = 200) 
  (h_time : time_to_cross = 12) : 
  let distance_km := (length_train : ℝ) / 1000
  let time_hr := (time_to_cross : ℝ) / 3600
  let speed_kmph := distance_km / time_hr
  speed_kmph = 60 := 
by 
  rw [h_length, h_time]
  simp [distance_km, time_hr, speed_kmph]
  norm_num
  sorry

end train_speed_60_kmph_l110_110170


namespace xy_range_l110_110100

theorem xy_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 1/x + y + 1/y = 5) :
  1/4 ≤ x * y ∧ x * y ≤ 4 :=
sorry

end xy_range_l110_110100


namespace project_completion_time_l110_110137

def process_duration (a b c d e f : Nat) : Nat :=
  let duration_c := max a b + c
  let duration_d := duration_c + d
  let duration_e := duration_c + e
  let duration_f := max duration_d duration_e + f
  duration_f

theorem project_completion_time :
  ∀ (a b c d e f : Nat), a = 2 → b = 3 → c = 2 → d = 5 → e = 4 → f = 1 →
  process_duration a b c d e f = 11 := by
  intros
  subst_vars
  sorry

end project_completion_time_l110_110137


namespace arith_to_geom_l110_110552

noncomputable def a (n : ℕ) (d : ℝ) : ℝ := 1 + (n - 1) * d

theorem arith_to_geom (m n : ℕ) (d : ℝ) 
  (h_pos : d > 0)
  (h_arith_seq : ∀ k : ℕ, a k d > 0)
  (h_geo_seq : (a 4 d + 5 / 2)^2 = (a 3 d) * (a 11 d))
  (h_mn : m - n = 8) : 
  a m d - a n d = 12 := 
sorry

end arith_to_geom_l110_110552


namespace find_b_l110_110259

-- Definition of the geometric problem
variables {a b c : ℝ} -- Side lengths of the triangle
variables {area : ℝ} -- Area of the triangle
variables {B : ℝ} -- Angle B in radians

-- Given conditions
def triangle_conditions : Prop :=
  area = sqrt 3 ∧
  B = π / 3 ∧
  a^2 + c^2 = 3 * a * c

-- Statement of the theorem using the given conditions to prove b = 2√2
theorem find_b (h : triangle_conditions) : b = 2 * sqrt 2 := 
  sorry

end find_b_l110_110259


namespace cost_of_lunch_l110_110867

-- Define the conditions: total amount and tip percentage
def total_amount : ℝ := 72.6
def tip_percentage : ℝ := 0.20

-- Define the proof problem
theorem cost_of_lunch (C : ℝ) (h : C + tip_percentage * C = total_amount) : C = 60.5 := 
sorry

end cost_of_lunch_l110_110867


namespace coin_flip_sequences_l110_110704

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110704


namespace factor_expression_l110_110778

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l110_110778


namespace smallest_integer_condition_l110_110920

theorem smallest_integer_condition (p d n x : ℕ) (h1 : 1 ≤ d ∧ d ≤ 9) 
  (h2 : x = 10^p * d + n) (h3 : x = 19 * n) : 
  x = 95 := by
  sorry

end smallest_integer_condition_l110_110920


namespace rectangle_perimeter_gt_16_l110_110065

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
  sorry

end rectangle_perimeter_gt_16_l110_110065


namespace ones_digit_of_4567_times_3_is_1_l110_110635

theorem ones_digit_of_4567_times_3_is_1 :
  let n := 4567
  let m := 3
  (n * m) % 10 = 1 :=
by
  let n := 4567
  let m := 3
  have h : (n * m) % 10 = ((4567 * 3) % 10) := by rfl -- simplifying the product
  sorry -- this is where the proof would go, if required

end ones_digit_of_4567_times_3_is_1_l110_110635


namespace coin_flip_sequences_l110_110699

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110699


namespace monomial_exponents_l110_110966

theorem monomial_exponents (m n : ℕ) 
  (h1 : m + 1 = 3)
  (h2 : n - 1 = 3) : 
  m^n = 16 := by
  sorry

end monomial_exponents_l110_110966


namespace solve_system_of_equations_l110_110623

theorem solve_system_of_equations:
  ∃ (x y z : ℝ), 
  x + y - z = 4 ∧
  x^2 + y^2 - z^2 = 12 ∧
  x^3 + y^3 - z^3 = 34 ∧
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l110_110623


namespace find_a_extreme_value_l110_110794

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem find_a_extreme_value :
  (∃ a : ℝ, ∀ x, f x a = Real.log (x + 1) - x - a * x ∧ (∃ m : ℝ, ∀ y : ℝ, f y a ≤ m)) ↔ a = -1 / 2 :=
by
  sorry

end find_a_extreme_value_l110_110794


namespace final_value_of_A_l110_110042

theorem final_value_of_A : 
  ∀ (A : Int), 
    (A = 20) → 
    (A = -A + 10) → 
    A = -10 :=
by
  intros A h1 h2
  sorry

end final_value_of_A_l110_110042


namespace sum_of_digits_5_pow_eq_2_pow_l110_110045

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_5_pow_eq_2_pow (n : ℕ) (h : sum_of_digits (5^n) = 2^n) : n = 3 :=
by
  sorry

end sum_of_digits_5_pow_eq_2_pow_l110_110045


namespace shelves_full_percentage_l110_110445

-- Define the conditions as constants
def ridges_per_record : Nat := 60
def cases : Nat := 4
def shelves_per_case : Nat := 3
def records_per_shelf : Nat := 20
def total_ridges : Nat := 8640

-- Define the total number of records
def total_records := total_ridges / ridges_per_record

-- Define the total capacity of the shelves
def total_capacity := cases * shelves_per_case * records_per_shelf

-- Define the percentage of shelves that are full
def percentage_full := (total_records * 100) / total_capacity

-- State the theorem that the percentage of the shelves that are full is 60%
theorem shelves_full_percentage : percentage_full = 60 := 
by
  sorry

end shelves_full_percentage_l110_110445


namespace cos_8_degree_l110_110389

theorem cos_8_degree (m : ℝ) (h : Real.sin (74 * Real.pi / 180) = m) :
  Real.cos (8 * Real.pi / 180) = Real.sqrt ((1 + m) / 2) :=
sorry

end cos_8_degree_l110_110389


namespace find_side_b_l110_110273

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (h_area : ∃ A B C, 1/2 * a * c * sin B = sqrt 3)
  (h_B : B = π / 3) (h_eq : a ^ 2 + c ^ 2 = 3 * a * c) : b = 2 * sqrt 2 :=
by
  sorry

end find_side_b_l110_110273


namespace probability_is_pi_over_12_l110_110016

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let radius := 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := 6 * 8
  circle_area / rectangle_area

theorem probability_is_pi_over_12 :
  probability_within_two_units_of_origin = Real.pi / 12 :=
by
  sorry

end probability_is_pi_over_12_l110_110016


namespace compare_cubics_l110_110211

variable {a b : ℝ}

theorem compare_cubics (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end compare_cubics_l110_110211


namespace find_number_l110_110154

theorem find_number (x : ℝ) (h : 5020 - (1004 / x) = 4970) : x = 20.08 := 
by
  sorry

end find_number_l110_110154


namespace triangle_side_length_l110_110266

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l110_110266


namespace largest_of_four_consecutive_integers_with_product_840_l110_110373

theorem largest_of_four_consecutive_integers_with_product_840 
  (a b c d : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c + 1 = d) (h_pos : 0 < a) (h_prod : a * b * c * d = 840) : d = 7 :=
sorry

end largest_of_four_consecutive_integers_with_product_840_l110_110373


namespace sum_of_radical_conjugates_l110_110516

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l110_110516


namespace factor_expression_l110_110781

theorem factor_expression (x : ℕ) : 75 * x + 45 = 15 * (5 * x + 3) :=
by
  sorry

end factor_expression_l110_110781


namespace ratio_x_y_z_w_l110_110520

theorem ratio_x_y_z_w (x y z w : ℝ) 
(h1 : 0.10 * x = 0.20 * y)
(h2 : 0.30 * y = 0.40 * z)
(h3 : 0.50 * z = 0.60 * w) : 
  (x / w) = 8 
  ∧ (y / w) = 4 
  ∧ (z / w) = 3
  ∧ (w / w) = 2.5 := 
sorry

end ratio_x_y_z_w_l110_110520


namespace prove_b_plus_m_equals_391_l110_110227

def matrix_A (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 3, b],
  ![0, 1, 5],
  ![0, 0, 1]
]

def matrix_power_A (m b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ := 
  (matrix_A b)^(m : ℕ)

def target_matrix : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 21, 3003],
  ![0, 1, 45],
  ![0, 0, 1]
]

theorem prove_b_plus_m_equals_391 (b m : ℕ) (h1 : matrix_power_A m b = target_matrix) : b + m = 391 := by
  sorry

end prove_b_plus_m_equals_391_l110_110227


namespace divisor_of_5025_is_5_l110_110754

/--
  Given an original number n which is 5026,
  and a resulting number after subtracting 1 from n,
  prove that the divisor of the resulting number is 5.
-/
theorem divisor_of_5025_is_5 (n : ℕ) (h₁ : n = 5026) (d : ℕ) (h₂ : (n - 1) % d = 0) : d = 5 :=
sorry

end divisor_of_5025_is_5_l110_110754


namespace distinct_sequences_ten_flips_l110_110720

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110720


namespace geometric_sequence_a3_a5_l110_110589

variable {a : ℕ → ℝ}

theorem geometric_sequence_a3_a5 (h₀ : a 1 > 0) 
                                (h₁ : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16) : 
                                a 3 + a 5 = 4 := 
sorry

end geometric_sequence_a3_a5_l110_110589


namespace find_b_l110_110270

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end find_b_l110_110270


namespace convert_base_9A3_16_to_4_l110_110522

theorem convert_base_9A3_16_to_4 :
  let h₁ := 9
  let h₂ := 10 -- A in hexadecimal
  let h₃ := 3
  let b₁ := 21 -- h₁ converted to base 4
  let b₂ := 22 -- h₂ converted to base 4
  let b₃ := 3  -- h₃ converted to base 4
  9 * 16^2 + 10 * 16^1 + 3 * 16^0 = 2 * 4^5 + 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 0 * 4^1 + 3 * 4^0 :=
by
  sorry

end convert_base_9A3_16_to_4_l110_110522


namespace logarithm_expression_evaluation_l110_110039

theorem logarithm_expression_evaluation :
  (3 / (log 3 (1000 ^ 4))) + (2 / (log 5 (1000 ^ 4))) = 1 / 12 :=
by
  sorry

end logarithm_expression_evaluation_l110_110039


namespace find_least_positive_integer_l110_110926

-- Definitions for the conditions
def leftmost_digit (x : ℕ) : ℕ :=
  x / 10^(Nat.log10 x)

def resulting_integer (x : ℕ) : ℕ :=
  x % 10^(Nat.log10 x)

def is_valid_original_integer (x : ℕ) (d n : ℕ) [Fact (d ∈ Finset.range 10 \ {0})] : Prop :=
  x = 10^Nat.log10 x * d + n ∧ n = x / 19

-- The theorem statement
theorem find_least_positive_integer :
  ∃ x : ℕ, is_valid_original_integer x (leftmost_digit x) (resulting_integer x) ∧ x = 95 :=
by {
  use 95,
  split,
  { sorry }, -- need to prove is_valid_original_integer 95 (leftmost_digit 95) (resulting_integer 95)
  { refl },
}

end find_least_positive_integer_l110_110926


namespace ratio_sea_horses_penguins_l110_110477

def sea_horses := 70
def penguins := sea_horses + 85

theorem ratio_sea_horses_penguins : (70 : ℚ) / (sea_horses + 85) = 14 / 31 :=
by
  -- Proof omitted
  sorry

end ratio_sea_horses_penguins_l110_110477


namespace sum_of_radical_conjugates_l110_110514

-- Define the numbers
def a := 15 - Real.sqrt 500
def b := 15 + Real.sqrt 500

-- Prove that the sum of a and b is 30
theorem sum_of_radical_conjugates : a + b = 30 := by
  calc
    a + b = (15 - Real.sqrt 500) + (15 + Real.sqrt 500) := by rw [a, b]
        ... = 30 := by sorry

end sum_of_radical_conjugates_l110_110514


namespace total_marbles_correct_l110_110984

-- Define the number of marbles Mary has
def MaryYellowMarbles := 9
def MaryBlueMarbles := 7
def MaryGreenMarbles := 6

-- Define the number of marbles Joan has
def JoanYellowMarbles := 3
def JoanBlueMarbles := 5
def JoanGreenMarbles := 4

-- Define the total number of marbles for Mary and Joan combined
def TotalMarbles := MaryYellowMarbles + MaryBlueMarbles + MaryGreenMarbles + JoanYellowMarbles + JoanBlueMarbles + JoanGreenMarbles

-- We want to prove that the total number of marbles is 34
theorem total_marbles_correct : TotalMarbles = 34 := by
  -- The proof is skipped with sorry
  sorry

end total_marbles_correct_l110_110984


namespace doctors_to_lawyers_ratio_l110_110579

theorem doctors_to_lawyers_ratio
  (d l : ℕ)
  (h1 : (40 * d + 55 * l) / (d + l) = 45)
  (h2 : d + l = 20) :
  d / l = 2 :=
by sorry

end doctors_to_lawyers_ratio_l110_110579


namespace two_pow_n_minus_one_div_by_seven_iff_l110_110784

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ (2^n - 1)) ↔ (∃ k : ℕ, n = 3 * k) := by
  sorry

end two_pow_n_minus_one_div_by_seven_iff_l110_110784


namespace shuttle_speed_l110_110330

theorem shuttle_speed (speed_kps : ℕ) (conversion_factor : ℕ) (speed_kph : ℕ) :
  speed_kps = 2 → conversion_factor = 3600 → speed_kph = speed_kps * conversion_factor → speed_kph = 7200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end shuttle_speed_l110_110330


namespace min_distance_l110_110371

theorem min_distance (x y z : ℝ) :
  ∃ (m : ℝ), m = (Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2)) ∧ m = Real.sqrt 6 :=
by
  sorry

end min_distance_l110_110371


namespace steps_per_flight_l110_110973

-- Define the problem conditions
def jack_flights_up := 3
def jack_flights_down := 6
def steps_height_inches := 8
def jack_height_change_feet := 24

-- Convert the height change to inches
def jack_height_change_inches := jack_height_change_feet * 12

-- Calculate the net flights down
def net_flights_down := jack_flights_down - jack_flights_up

-- Calculate total height change in inches for net flights
def total_height_change_inches := net_flights_down * jack_height_change_inches

-- Calculate the number of steps in each flight
def number_of_steps_per_flight :=
  total_height_change_inches / (steps_height_inches * net_flights_down)

theorem steps_per_flight :
  number_of_steps_per_flight = 108 :=
sorry

end steps_per_flight_l110_110973


namespace range_of_f_l110_110565

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x - 5) / (x + 3)

-- Define the domain as [1, 4)
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x < 4

theorem range_of_f : set.range (λ x, f x) (set.Ico 1 4) = set.Ico (-3/4 : ℝ) (3/7 : ℝ) := by
  sorry

end range_of_f_l110_110565


namespace kolya_mistaken_l110_110655

-- Definitions relating to the conditions
def at_least_four_blue_pencils (blue_pencils : ℕ) : Prop := blue_pencils >= 4
def at_least_five_green_pencils (green_pencils : ℕ) : Prop := green_pencils >= 5
def at_least_three_blue_pencils_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 3 ∧ green_pencils >= 4
def at_least_four_blue_and_four_green_pencils (blue_pencils green_pencils : ℕ) : Prop := blue_pencils >= 4 ∧ green_pencils >= 4

-- Speaking truth conditions
variables (blue_pencils green_pencils : ℕ)
def vasya_true : Prop := at_least_four_blue_pencils blue_pencils
def kolya_true : Prop := at_least_five_green_pencils green_pencils
def petya_true : Prop := at_least_three_blue_pencils_and_four_green_pencils blue_pencils green_pencils
def misha_true : Prop := at_least_four_blue_and_four_green_pencils blue_pencils green_pencils

-- Given known information: three are true, one is false
def known_information : Prop := (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬kolya_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ ¬misha_true blue_pencils green_pencils)
                            ∨ (vasya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬petya_true blue_pencils green_pencils)
                            ∨ (petya_true blue_pencils green_pencils ∧ kolya_true blue_pencils green_pencils ∧ misha_true blue_pencils green_pencils ∧ ¬vasya_true blue_pencils green_pencils)

-- Theorem to be proved
theorem kolya_mistaken : known_information blue_pencils green_pencils → ¬kolya_true blue_pencils green_pencils :=
sorry

end kolya_mistaken_l110_110655


namespace cab_base_price_l110_110876

theorem cab_base_price (base_price : ℝ) (total_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) 
  (H1 : total_cost = 23) 
  (H2 : cost_per_mile = 4) 
  (H3 : distance = 5) 
  (H4 : base_price = total_cost - cost_per_mile * distance) : 
  base_price = 3 :=
by 
  sorry

end cab_base_price_l110_110876


namespace population_doubling_time_l110_110299

open Real

noncomputable def net_growth_rate (birth_rate : ℝ) (death_rate : ℝ) : ℝ :=
birth_rate - death_rate

noncomputable def percentage_growth_rate (net_growth_rate : ℝ) (population_base : ℝ) : ℝ :=
(net_growth_rate / population_base) * 100

noncomputable def doubling_time (percentage_growth_rate : ℝ) : ℝ :=
70 / percentage_growth_rate

theorem population_doubling_time :
    let birth_rate := 39.4
    let death_rate := 19.4
    let population_base := 1000
    let net_growth := net_growth_rate birth_rate death_rate
    let percentage_growth := percentage_growth_rate net_growth population_base
    doubling_time percentage_growth = 35 := 
by
    sorry

end population_doubling_time_l110_110299


namespace boys_count_l110_110847

theorem boys_count (B G : ℕ) (h1 : B + G = 41) (h2 : 12 * B + 8 * G = 460) : B = 33 := 
by
  sorry

end boys_count_l110_110847


namespace find_k_value_l110_110541

theorem find_k_value (k : ℕ) :
  3 * 6 * 4 * k = Nat.factorial 8 → k = 560 :=
by
  sorry

end find_k_value_l110_110541


namespace graphs_intersect_once_l110_110050

variable {a b c d : ℝ}

theorem graphs_intersect_once 
(h1: ∃ x, (2 * a + 1 / (x - b)) = (2 * c + 1 / (x - d)) ∧ 
∃ y₁ y₂: ℝ, ∀ x, (2 * a + 1 / (x - b)) ≠ 2 * c + 1 / (x - d)) : 
∃ x, ((2 * b + 1 / (x - a)) = (2 * d + 1 / (x - c))) ∧ 
∃ y₁ y₂: ℝ, ∀ x, 2 * b + 1 / (x - a) ≠ 2 * d + 1 / (x - c) := 
sorry

end graphs_intersect_once_l110_110050


namespace range_of_m_l110_110238

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (6 - 3 * (x + 1) < x - 9) ∧ (x - m > -1) ↔ (x > 3)) → (m ≤ 4) :=
by
  sorry

end range_of_m_l110_110238


namespace powers_of_i_sum_l110_110356

theorem powers_of_i_sum :
  ∀ (i : ℂ), 
  (i^1 = i) ∧ (i^2 = -1) ∧ (i^3 = -i) ∧ (i^4 = 1) →
  i^8621 + i^8622 + i^8623 + i^8624 + i^8625 = 0 :=
by
  intros i h
  sorry

end powers_of_i_sum_l110_110356


namespace admission_fee_for_adults_l110_110432

theorem admission_fee_for_adults (C : ℝ) (N M N_c N_a : ℕ) (A : ℝ) 
  (h1 : C = 1.50) 
  (h2 : N = 2200) 
  (h3 : M = 5050) 
  (h4 : N_c = 700) 
  (h5 : N_a = 1500) :
  A = 2.67 := 
by
  sorry

end admission_fee_for_adults_l110_110432


namespace value_of_p_l110_110443

theorem value_of_p (p q : ℝ) (h1 : q = (2 / 5) * p) (h2 : p * q = 90) : p = 15 :=
by
  sorry

end value_of_p_l110_110443


namespace number_of_red_squares_in_19th_row_l110_110885

-- Define the number of squares in the n-th row
def number_of_squares (n : ℕ) : ℕ := 3 * n - 1

-- Define the number of red squares in the n-th row
def red_squares (n : ℕ) : ℕ := (number_of_squares n) / 2

-- The theorem stating the problem
theorem number_of_red_squares_in_19th_row : red_squares 19 = 28 := by
  -- Proof goes here
  sorry

end number_of_red_squares_in_19th_row_l110_110885


namespace k_value_of_polynomial_square_l110_110093

theorem k_value_of_polynomial_square (k : ℤ) :
  (∃ (f : ℤ → ℤ), ∀ x, f x = x^2 + 6 * x + k^2) → (k = 3 ∨ k = -3) :=
by
  sorry

end k_value_of_polynomial_square_l110_110093


namespace perimeter_gt_sixteen_l110_110064

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end perimeter_gt_sixteen_l110_110064


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l110_110075

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l110_110075


namespace inequality_3var_l110_110413

variable {x y z : ℝ}

-- Define conditions
def positive_real (a : ℝ) : Prop := a > 0

theorem inequality_3var (hx : positive_real x) (hy : positive_real y) (hz : positive_real z) : 
  (x * y / z + y * z / x + z * x / y) > 2 * (x^3 + y^3 + z^3)^(1 / 3) :=
by {
    sorry
}

end inequality_3var_l110_110413


namespace total_young_fish_l110_110610

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end total_young_fish_l110_110610


namespace max_volume_right_tetrahedron_l110_110207

theorem max_volume_right_tetrahedron (PA PB PC : ℝ) (h1 : 0 ≤ PA) (h2 : 0 ≤ PB) (h3 : 0 ≤ PC) (h_angle : (PA^2 + PB^2 = PC^2)) (S : ℝ) (hS : S = PA + PB + PC + real.sqrt (PA^2 + PB^2) + real.sqrt (PB^2 + PC^2) + real.sqrt (PC^2 + PA^2)) :
  ∃ (V : ℝ), V ≤ (1/162) * (5 * real.sqrt 2 - 7) * S^3 :=
begin
  sorry
end

end max_volume_right_tetrahedron_l110_110207


namespace problem1_problem2_l110_110553

-- Definitions of the sets A, B, and C based on conditions given
def setA : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def setB : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def setC (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Problem (1): Prove values of b and c
theorem problem1 (b c : ℝ) :
  (∀ x, x ∈ (setA ∩ setB) ↔ b*x^2 + 10*x + c ≥ 0) → b = -2 ∧ c = -12 := sorry

-- Universal set definition and its complement
def universalSet : Set ℝ := {x | True}
def complementA : Set ℝ := {x | (x ∉ setA)}

-- Problem (2): Range of a
theorem problem2 (a : ℝ) :
  (setC a ⊆ setB ∪ complementA) → a ∈ Set.Icc (-11/6) (9/4) := sorry

end problem1_problem2_l110_110553


namespace horner_value_at_3_l110_110313

noncomputable def horner (x : ℝ) : ℝ :=
  ((((0.5 * x + 4) * x + 0) * x - 3) * x + 1) * x - 1

theorem horner_value_at_3 : horner 3 = 5.5 :=
by
  sorry

end horner_value_at_3_l110_110313


namespace cubing_identity_l110_110567

theorem cubing_identity (x : ℂ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
  sorry

end cubing_identity_l110_110567


namespace gears_can_look_complete_l110_110872

theorem gears_can_look_complete (n : ℕ) (h1 : n = 14)
                                 (h2 : ∀ k, k = 4)
                                 (h3 : ∀ i, 0 ≤ i ∧ i < n) :
  ∃ j, 1 ≤ j ∧ j < n ∧ (∀ m1 m2, m1 ≠ m2 → ((m1 + j) % n) ≠ ((m2 + j) % n)) := 
sorry

end gears_can_look_complete_l110_110872


namespace find_result_l110_110833

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x - 3

theorem find_result : f (g 3) - g (f 3) = -6 := by
  sorry

end find_result_l110_110833


namespace proof_w3_u2_y2_l110_110123

variable (x y z w u d : ℤ)

def arithmetic_sequence := x = 1370 ∧ z = 1070 ∧ w = -180 ∧ u = -6430 ∧ (z = x + 2 * d) ∧ (y = x + d)

theorem proof_w3_u2_y2 (h : arithmetic_sequence x y z w u d) : w^3 - u^2 + y^2 = -44200100 :=
  by
    sorry

end proof_w3_u2_y2_l110_110123


namespace interval_of_increase_f_a_half_minimum_integer_value_a_l110_110085

open Real

def f (a : ℝ) (x : ℝ) := a * x^2 - log x
def g (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + x
def F (a : ℝ) (x : ℝ) := f a x - g a x

theorem interval_of_increase_f_a_half :
  {x : ℝ | 1 < x} = {x : ℝ | 1 < x ∧ ∀ y, f (1/2) y > f (1/2) x} :=
sorry

theorem minimum_integer_value_a :
  (∀ x : ℝ, F a x ≥ 1 - a * x) → a ≥ 2 :=
sorry

end interval_of_increase_f_a_half_minimum_integer_value_a_l110_110085


namespace extra_time_A_to_reach_destination_l110_110105

theorem extra_time_A_to_reach_destination (speed_ratio : ℕ -> ℕ -> Prop) (t_A t_B : ℝ)
  (h_ratio : speed_ratio 3 4)
  (time_A : t_A = 2)
  (distance_constant : ∀ a b : ℝ, a / b = (3 / 4)) :
  (t_A - t_B) * 60 = 30 :=
by
  sorry

end extra_time_A_to_reach_destination_l110_110105


namespace strength_training_sessions_l110_110596

-- Define the problem conditions
def strength_training_hours (x : ℕ) : ℝ := x * 1
def boxing_training_hours : ℝ := 4 * 1.5
def total_training_hours : ℝ := 9

-- Prove how many times a week does Kat do strength training
theorem strength_training_sessions : ∃ x : ℕ, strength_training_hours x + boxing_training_hours = total_training_hours ∧ x = 3 := 
by {
  sorry
}

end strength_training_sessions_l110_110596


namespace kolya_is_wrong_l110_110650

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end kolya_is_wrong_l110_110650


namespace angle_bisector_length_l110_110590

open Real
open Complex

-- Definitions for the problem
def side_lengths (AC BC : ℝ) : Prop :=
  AC = 6 ∧ BC = 9

def angle_C (angle : ℝ) : Prop :=
  angle = 120

-- Main statement to prove
theorem angle_bisector_length (AC BC angle x : ℝ)
  (h1 : side_lengths AC BC)
  (h2 : angle_C angle) :
  x = 18 / 5 :=
  sorry

end angle_bisector_length_l110_110590


namespace find_k_values_l110_110161

theorem find_k_values (k : ℚ) 
  (h1 : ∀ k, ∃ m, m = (3 * k + 9) / (7 - k))
  (h2 : ∀ k, m = 2 * k) : 
  (k = 9 / 2 ∨ k = 1) :=
by
  sorry

end find_k_values_l110_110161


namespace find_k_of_collinear_points_l110_110891

theorem find_k_of_collinear_points :
  ∃ k : ℚ, ∀ (x1 y1 x2 y2 x3 y3 : ℚ), (x1, y1) = (4, 10) → (x2, y2) = (-3, k) → (x3, y3) = (-8, 5) → 
  ((y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)) → k = 85 / 12 :=
by
  sorry

end find_k_of_collinear_points_l110_110891


namespace max_mass_of_grain_l110_110467

theorem max_mass_of_grain (length width : ℝ) (angle : ℝ) (density : ℝ) 
  (h_length : length = 10) (h_width : width = 5) (h_angle : angle = 45) (h_density : density = 1200) : 
  volume * density = 175000 :=
by
  let height := width / 2
  let base_area := length * width
  let prism_volume := base_area * height
  let pyramid_volume := (1 / 3) * (width / 2 * length) * height
  let total_volume := prism_volume + 2 * pyramid_volume
  let volume := total_volume
  sorry

end max_mass_of_grain_l110_110467


namespace isosceles_triangle_angle_split_l110_110519

theorem isosceles_triangle_angle_split (A B C1 C2 : ℝ)
  (h_isosceles : A = B)
  (h_greater_than_third : A > C1)
  (h_split : C1 + C2 = C) :
  C1 = C2 :=
sorry

end isosceles_triangle_angle_split_l110_110519


namespace number_of_blue_socks_l110_110310

theorem number_of_blue_socks (x : ℕ) (h : ((6 + x ^ 2 - x) / ((6 + x) * (5 + x)) = 1/5)) : x = 4 := 
sorry

end number_of_blue_socks_l110_110310


namespace cyclists_speeds_product_l110_110447

theorem cyclists_speeds_product (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h₁ : 6 / u = 6 / v + 1 / 12) 
  (h₂ : v / 3 = u / 3 + 4) : 
  u * v = 864 := 
by
  sorry

end cyclists_speeds_product_l110_110447


namespace area_triangle_ABC_l110_110586

theorem area_triangle_ABC (AB CD height : ℝ) 
  (h_parallel : AB + CD = 20)
  (h_ratio : CD = 3 * AB)
  (h_height : height = (2 * 20) / (AB + CD)) :
  (1 / 2) * AB * height = 5 := sorry

end area_triangle_ABC_l110_110586


namespace least_positive_integer_l110_110929

noncomputable def hasProperty (x n d p : ℕ) : Prop :=
  x = 10^p * d + n ∧ n = x / 19

theorem least_positive_integer : 
  ∃ (x n d p : ℕ), hasProperty x n d p ∧ x = 950 :=
by
  sorry

end least_positive_integer_l110_110929


namespace find_angle_bisector_length_l110_110591

-- Define the problem context
variable (A B C D : Type) [Triangle ABC] 
variable (AC BC : ℝ) (angle_C : ℝ) (angle_bisector_CD : ℝ)

-- Specify given conditions
axiom AC_eq_6 : AC = 6
axiom BC_eq_9 : BC = 9
axiom angle_C_eq_120 : angle_C = 120

-- The theorem to prove the problem statement
theorem find_angle_bisector_length : angle_bisector_CD = 18 / 5 :=
by
    have h1 : AC = 6 := AC_eq_6
    have h2 : BC = 9 := BC_eq_9
    have h3 : angle_C = 120 := angle_C_eq_120
    sorry

end find_angle_bisector_length_l110_110591


namespace probability_sum_18_two_12_sided_dice_l110_110751

theorem probability_sum_18_two_12_sided_dice :
  let total_outcomes := 12 * 12
  let successful_outcomes := 7
  successful_outcomes / total_outcomes = 7 / 144 := by
sorry

end probability_sum_18_two_12_sided_dice_l110_110751


namespace new_parabola_through_point_l110_110099

def original_parabola (x : ℝ) : ℝ := x ^ 2 + 2 * x - 1

theorem new_parabola_through_point : 
  (∃ b : ℝ, ∀ x : ℝ, (x ^ 2 + 2 * x - 1 + b) = (x ^ 2 + 2 * x + 3)) :=
by
  sorry

end new_parabola_through_point_l110_110099


namespace sin_cos_of_tan_is_two_l110_110546

theorem sin_cos_of_tan_is_two (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 :=
sorry

end sin_cos_of_tan_is_two_l110_110546


namespace tory_earns_more_than_bert_l110_110902

-- Define the initial prices of the toys
def initial_price_phones : ℝ := 18
def initial_price_guns : ℝ := 20

-- Define the quantities sold by Bert and Tory
def quantity_phones : ℕ := 10
def quantity_guns : ℕ := 15

-- Define the discounts
def discount_phones : ℝ := 0.15
def discounted_phones_quantity : ℕ := 3

def discount_guns : ℝ := 0.10
def discounted_guns_quantity : ℕ := 7

-- Define the tax
def tax_rate : ℝ := 0.05

noncomputable def bert_initial_earnings : ℝ := initial_price_phones * quantity_phones

noncomputable def tory_initial_earnings : ℝ := initial_price_guns * quantity_guns

noncomputable def bert_discount : ℝ := discount_phones * initial_price_phones * discounted_phones_quantity

noncomputable def tory_discount : ℝ := discount_guns * initial_price_guns * discounted_guns_quantity

noncomputable def bert_earnings_after_discount : ℝ := bert_initial_earnings - bert_discount

noncomputable def tory_earnings_after_discount : ℝ := tory_initial_earnings - tory_discount

noncomputable def bert_tax : ℝ := tax_rate * bert_earnings_after_discount

noncomputable def tory_tax : ℝ := tax_rate * tory_earnings_after_discount

noncomputable def bert_final_earnings : ℝ := bert_earnings_after_discount + bert_tax

noncomputable def tory_final_earnings : ℝ := tory_earnings_after_discount + tory_tax

noncomputable def earning_difference : ℝ := tory_final_earnings - bert_final_earnings

theorem tory_earns_more_than_bert : earning_difference = 119.805 := by
  sorry

end tory_earns_more_than_bert_l110_110902


namespace implies_neg_p_and_q_count_l110_110187

-- Definitions of the logical conditions
variables (p q : Prop)

def cond1 : Prop := p ∧ q
def cond2 : Prop := p ∧ ¬ q
def cond3 : Prop := ¬ p ∧ q
def cond4 : Prop := ¬ p ∧ ¬ q

-- Negative of the statement "p and q are both true"
def neg_p_and_q := ¬ (p ∧ q)

-- The Lean 4 statement to prove
theorem implies_neg_p_and_q_count :
  (cond2 p q → neg_p_and_q p q) ∧ 
  (cond3 p q → neg_p_and_q p q) ∧ 
  (cond4 p q → neg_p_and_q p q) ∧ 
  ¬ (cond1 p q → neg_p_and_q p q) :=
sorry

end implies_neg_p_and_q_count_l110_110187


namespace find_mistake_l110_110648

theorem find_mistake 
  (at_least_4_blue : Prop) 
  (at_least_5_green : Prop) 
  (at_least_3_blue_and_4_green : Prop) 
  (at_least_4_blue_and_4_green : Prop)
  (truths_condition : at_least_4_blue ∧ at_least_3_blue_and_4_green ∧ at_least_4_blue_and_4_green):
  ¬ at_least_5_green :=
by 
  -- sorry can be used here as proof if required
  sorry

end find_mistake_l110_110648


namespace average_molecular_weight_benzoic_acid_l110_110480

def atomic_mass_C : ℝ := (12 * 0.9893) + (13 * 0.0107)
def atomic_mass_H : ℝ := (1 * 0.99985) + (2 * 0.00015)
def atomic_mass_O : ℝ := (16 * 0.99762) + (17 * 0.00038) + (18 * 0.00200)

theorem average_molecular_weight_benzoic_acid :
  (7 * atomic_mass_C) + (6 * atomic_mass_H) + (2 * atomic_mass_O) = 123.05826 :=
by {
  sorry
}

end average_molecular_weight_benzoic_acid_l110_110480


namespace verify_salary_problem_l110_110131

def salary_problem (W : ℕ) (S_old : ℕ) (S_new : ℕ := 780) (n : ℕ := 9) : Prop :=
  (W + S_old) / n = 430 ∧ (W + S_new) / n = 420 → S_old = 870

theorem verify_salary_problem (W S_old : ℕ) (h1 : (W + S_old) / 9 = 430) (h2 : (W + 780) / 9 = 420) : S_old = 870 :=
by {
  sorry
}

end verify_salary_problem_l110_110131


namespace number_of_distinct_rationals_l110_110913

theorem number_of_distinct_rationals (L : ℕ) :
  L = 26 ↔
  (∃ (k : ℚ), |k| < 100 ∧ (∃ (x : ℤ), 7 * x^2 + k * x + 20 = 0)) :=
sorry

end number_of_distinct_rationals_l110_110913


namespace find_remaining_area_l110_110842

theorem find_remaining_area 
    (base_RST : ℕ) 
    (height_RST : ℕ) 
    (base_RSC : ℕ) 
    (height_RSC : ℕ) 
    (area_RST : ℕ := (1 / 2) * base_RST * height_RST) 
    (area_RSC : ℕ := (1 / 2) * base_RSC * height_RSC) 
    (remaining_area : ℕ := area_RST - area_RSC) 
    (h_base_RST : base_RST = 5) 
    (h_height_RST : height_RST = 4) 
    (h_base_RSC : base_RSC = 1) 
    (h_height_RSC : height_RSC = 4) : 
    remaining_area = 8 := 
by 
  sorry

end find_remaining_area_l110_110842


namespace frac_m_q_eq_one_l110_110090

theorem frac_m_q_eq_one (m n p q : ℕ) 
  (h1 : m = 40 * n)
  (h2 : p = 5 * n)
  (h3 : p = q / 8) : (m / q = 1) :=
by
  sorry

end frac_m_q_eq_one_l110_110090


namespace positive_number_square_roots_l110_110239

theorem positive_number_square_roots (m : ℝ) 
  (h : (2 * m - 1) + (2 - m) = 0) :
  (2 - m)^2 = 9 :=
by
  sorry

end positive_number_square_roots_l110_110239


namespace fraction_equality_l110_110372

theorem fraction_equality (x : ℝ) :
  (4 + 2 * x) / (7 + 3 * x) = (2 + 3 * x) / (4 + 5 * x) ↔ x = -1 ∨ x = -2 := by
  sorry

end fraction_equality_l110_110372


namespace distinct_sequences_ten_flips_l110_110723

/-- Define a CoinFlip data type representing the two possible outcomes -/
inductive CoinFlip : Type
| heads : CoinFlip
| tails : CoinFlip

/-- Define the total number of distinct sequences of coin flips -/
def countSequences (flips : ℕ) : ℕ :=
  2 ^ flips

/-- A proof statement showing that there are 1024 distinct sequences when a coin is flipped ten times -/
theorem distinct_sequences_ten_flips : countSequences 10 = 1024 :=
by
  sorry

end distinct_sequences_ten_flips_l110_110723


namespace times_faster_l110_110753

theorem times_faster (A B : ℝ) (h1 : A + B = 1 / 12) (h2 : A = 1 / 16) : 
  A / B = 3 :=
by
  sorry

end times_faster_l110_110753


namespace work_completion_days_l110_110460

theorem work_completion_days (A_days B_days : ℕ) (hA : A_days = 3) (hB : B_days = 6) : 
  (1 / ((1 / (A_days : ℚ)) + (1 / (B_days : ℚ)))) = 2 := 
by
  sorry

end work_completion_days_l110_110460


namespace find_x_solution_l110_110812

theorem find_x_solution (b x : ℝ) (hb : b > 1) (hx : x > 0) 
    (h_eq : (4 * x)^(Real.log 4 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) : 
    x = 1 / 5 :=
by
  sorry

end find_x_solution_l110_110812


namespace english_students_23_l110_110580

def survey_students_total : Nat := 35
def students_in_all_three : Nat := 2
def solely_english_three_times_than_french (x y : Nat) : Prop := y = 3 * x
def english_but_not_french_or_spanish (x y : Nat) : Prop := y + students_in_all_three = 35 ∧ y - students_in_all_three = 23

theorem english_students_23 :
  ∃ (x y : Nat), solely_english_three_times_than_french x y ∧ english_but_not_french_or_spanish x y :=
by
  sorry

end english_students_23_l110_110580


namespace sum_due_in_years_l110_110241

theorem sum_due_in_years 
  (D : ℕ)
  (S : ℕ)
  (r : ℚ)
  (H₁ : D = 168)
  (H₂ : S = 768)
  (H₃ : r = 14 / 100) :
  ∃ t : ℕ, t = 2 := 
by
  sorry

end sum_due_in_years_l110_110241


namespace calories_for_breakfast_l110_110660

theorem calories_for_breakfast :
  let cake_calories := 110
  let chips_calories := 310
  let coke_calories := 215
  let lunch_calories := 780
  let daily_limit := 2500
  let remaining_calories := 525
  let total_dinner_snacks := cake_calories + chips_calories + coke_calories
  let total_lunch_dinner := total_dinner_snacks + lunch_calories
  let total_consumed := daily_limit - remaining_calories
  total_consumed - total_lunch_dinner = 560 := by
  sorry

end calories_for_breakfast_l110_110660


namespace rectangle_perimeter_greater_than_16_l110_110070

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l110_110070


namespace coin_flip_sequences_l110_110743

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110743


namespace find_side_b_l110_110275

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (h_area : ∃ A B C, 1/2 * a * c * sin B = sqrt 3)
  (h_B : B = π / 3) (h_eq : a ^ 2 + c ^ 2 = 3 * a * c) : b = 2 * sqrt 2 :=
by
  sorry

end find_side_b_l110_110275


namespace last_third_speed_l110_110889

-- Definitions based on the conditions in the problem statement
def first_third_speed : ℝ := 80
def second_third_speed : ℝ := 30
def average_speed : ℝ := 45

-- Definition of the distance covered variable (non-zero to avoid division by zero)
variable (D : ℝ) (hD : D ≠ 0)

-- The unknown speed during the last third of the distance
noncomputable def V : ℝ := 
  D / ((D / 3 / first_third_speed) + (D / 3 / second_third_speed) + (D / 3 / average_speed))

-- The theorem to prove
theorem last_third_speed : V = 48 :=
by
  sorry

end last_third_speed_l110_110889


namespace find_b_l110_110258

-- Definition of the geometric problem
variables {a b c : ℝ} -- Side lengths of the triangle
variables {area : ℝ} -- Area of the triangle
variables {B : ℝ} -- Angle B in radians

-- Given conditions
def triangle_conditions : Prop :=
  area = sqrt 3 ∧
  B = π / 3 ∧
  a^2 + c^2 = 3 * a * c

-- Statement of the theorem using the given conditions to prove b = 2√2
theorem find_b (h : triangle_conditions) : b = 2 * sqrt 2 := 
  sorry

end find_b_l110_110258


namespace number_of_solutions_eq_one_l110_110524

theorem number_of_solutions_eq_one :
  (∃! y : ℝ, (y ≠ 0) ∧ (y ≠ 3) ∧ ((3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1)) :=
  sorry

end number_of_solutions_eq_one_l110_110524


namespace tan_bounds_l110_110293

theorem tan_bounds (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 1) :
    (2 / Real.pi) * (x / (1 - x)) ≤ Real.tan ((Real.pi * x) / 2) ∧
    Real.tan ((Real.pi * x) / 2) ≤ (Real.pi / 2) * (x / (1 - x)) :=
by
    sorry

end tan_bounds_l110_110293


namespace triangle_problem_l110_110263

noncomputable def find_b (a b c : ℝ) : Prop :=
  let B : ℝ := 60 * Real.pi / 180 -- converting 60 degrees to radians
  b = 2 * Real.sqrt 2

theorem triangle_problem
  (a b c : ℝ)
  (h_area : (1 / 2) * a * c * Real.sin (60 * Real.pi / 180) = Real.sqrt 3)
  (h_cosine : a^2 + c^2 = 3 * a * c) : find_b a b c :=
by
  -- The proof would go here, but we're skipping it as per the instructions.
  sorry

end triangle_problem_l110_110263


namespace sum_of_number_and_conjugate_l110_110494

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l110_110494


namespace initial_ratio_of_milk_to_water_l110_110967

theorem initial_ratio_of_milk_to_water (M W : ℕ) (h1 : M + 20 = 3 * W) (h2 : M + W = 40) :
  (M : ℚ) / W = 5 / 3 := by
sorry

end initial_ratio_of_milk_to_water_l110_110967


namespace problem_bound_l110_110619

theorem problem_bound (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * (x * y * z) ∧ 
  y * z + z * x + x * y - 2 * (x * y * z) ≤ 7 / 27 :=
sorry

end problem_bound_l110_110619


namespace solve_for_x_l110_110129

-- Define the variables and conditions
variable (x : ℚ)

-- Define the given condition
def condition : Prop := (x + 4)/(x - 3) = (x - 2)/(x + 2)

-- State the theorem that x = -2/11 is a solution to the condition
theorem solve_for_x (h : condition x) : x = -2 / 11 := by
  sorry

end solve_for_x_l110_110129


namespace smaller_angle_at_3_45_is_157_5_l110_110323

-- Define the conditions
def hour_hand_deg_at_3_45 : ℝ := (3 * 30) + ((45 / 60) * 30)
def minute_hand_deg_at_3_45 : ℝ := 45 * 6

-- Define the statement to prove
theorem smaller_angle_at_3_45_is_157_5 :
  abs (minute_hand_deg_at_3_45 - hour_hand_deg_at_3_45) = 157.5 :=
by
  -- Proof is skipped
  sorry

end smaller_angle_at_3_45_is_157_5_l110_110323


namespace arithmetic_seq_a7_l110_110214

theorem arithmetic_seq_a7 (a : ℕ → ℤ) (d : ℤ) (h1 : ∀ (n m : ℕ), a (n + m) = a n + m * d)
  (h2 : a 4 + a 9 = 24) (h3 : a 6 = 11) :
  a 7 = 13 :=
sorry

end arithmetic_seq_a7_l110_110214


namespace inv_mod_997_l110_110908

theorem inv_mod_997 : ∃ x : ℤ, 0 ≤ x ∧ x < 997 ∧ (10 * x) % 997 = 1 := 
sorry

end inv_mod_997_l110_110908


namespace arithmetic_mean_124_4_31_l110_110448

theorem arithmetic_mean_124_4_31 :
  let numbers := [12, 25, 39, 48]
  let total := 124
  let count := 4
  (total / count : ℝ) = 31 := by
  sorry

end arithmetic_mean_124_4_31_l110_110448


namespace distinct_sequences_ten_flips_l110_110712

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110712


namespace good_walker_catches_up_l110_110678

-- Definitions based on the conditions in the problem
def steps_good_walker := 100
def steps_bad_walker := 60
def initial_lead := 100

-- Mathematical proof problem statement
theorem good_walker_catches_up :
  ∃ x : ℕ, x = initial_lead + (steps_bad_walker * x / steps_good_walker) :=
sorry

end good_walker_catches_up_l110_110678


namespace coin_flip_sequences_l110_110701

theorem coin_flip_sequences (n : ℕ) (h1 : n = 10) : 
  2 ^ n = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110701


namespace number_multiplied_by_3_l110_110392

theorem number_multiplied_by_3 (k : ℕ) : 
  2^13 - 2^(13-2) = 3 * k → k = 2048 :=
by
  sorry

end number_multiplied_by_3_l110_110392


namespace peyton_manning_total_yards_l110_110992

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end peyton_manning_total_yards_l110_110992


namespace vertex_of_parabola_l110_110949

theorem vertex_of_parabola 
  (a b c : ℝ) 
  (h1 : a * 2^2 + b * 2 + c = 5)
  (h2 : -b / (2 * a) = 2) : 
  (2, 4 * a + 2 * b + c) = (2, 5) :=
by
  sorry

end vertex_of_parabola_l110_110949


namespace polynomial_mult_of_6_l110_110606

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end polynomial_mult_of_6_l110_110606


namespace monotonically_increasing_a_range_l110_110235

noncomputable def f (a x : ℝ) : ℝ := (a * x - 1) * Real.exp x

theorem monotonically_increasing_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f a x ≥ 0) ↔ 1 ≤ a  :=
by
  sorry

end monotonically_increasing_a_range_l110_110235


namespace bob_average_speed_l110_110904

theorem bob_average_speed
  (lap_distance : ℕ) (lap1_time lap2_time lap3_time total_laps : ℕ)
  (h_lap_distance : lap_distance = 400)
  (h_lap1_time : lap1_time = 70)
  (h_lap2_time : lap2_time = 85)
  (h_lap3_time : lap3_time = 85)
  (h_total_laps : total_laps = 3) : 
  (lap_distance * total_laps) / (lap1_time + lap2_time + lap3_time) = 5 := by
    sorry

end bob_average_speed_l110_110904


namespace sum_of_number_and_conjugate_l110_110497

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l110_110497


namespace unique_real_solution_between_consecutive_integers_l110_110883

theorem unique_real_solution_between_consecutive_integers (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k < x ∧ x < k + 1 ∧ (⌊x⌋ : ℝ) * (x^2 + 1) = x^3 := sorry

end unique_real_solution_between_consecutive_integers_l110_110883


namespace probability_valid_pairings_l110_110044

theorem probability_valid_pairings (S : Finset (Nat × Nat)) (hS : S.card = 15) :
  let m := 209
  let n := 3120
  ∃ p : ℚ, p = (m : ℚ) / n ∧ m + n = 3329 :=
by
  -- We need to prove that the probability of valid pairings is 209/3120
  -- according to the described conditions.
  sorry

end probability_valid_pairings_l110_110044


namespace fraction_to_decimal_l110_110009

theorem fraction_to_decimal : (17 : ℝ) / 50 = 0.34 := 
by 
  sorry

end fraction_to_decimal_l110_110009


namespace transform_sequence_zero_l110_110943

theorem transform_sequence_zero 
  (n : ℕ) 
  (a : ℕ → ℝ) 
  (h_nonempty : n > 0) :
  ∃ k : ℕ, k ≤ n ∧ ∀ k' ≤ k, ∃ α : ℝ, (∀ i, i < n → |a i - α| = 0) := 
sorry

end transform_sequence_zero_l110_110943


namespace negation_of_universal_quantification_l110_110328

theorem negation_of_universal_quantification (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| > 1) ↔ ∃ x ∈ S, |x| ≤ 1 :=
by
  sorry

end negation_of_universal_quantification_l110_110328


namespace usual_time_to_catch_bus_l110_110147

theorem usual_time_to_catch_bus (S T : ℝ) (h1 : S / ((5/4) * S) = (T + 5) / T) : T = 25 :=
by sorry

end usual_time_to_catch_bus_l110_110147


namespace symmetric_point_coords_l110_110585

def pointA : ℝ × ℝ := (1, 2)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

def pointB : ℝ × ℝ := translate_left pointA 2

def pointC : ℝ × ℝ := reflect_origin pointB

theorem symmetric_point_coords :
  pointC = (1, -2) :=
by
  -- Proof omitted as instructed
  sorry

end symmetric_point_coords_l110_110585


namespace polynomial_mult_of_6_l110_110605

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end polynomial_mult_of_6_l110_110605


namespace pipe_drain_rate_l110_110289

theorem pipe_drain_rate 
(T r_A r_B r_C : ℕ) 
(h₁ : T = 950) 
(h₂ : r_A = 40) 
(h₃ : r_B = 30) 
(h₄ : ∃ m : ℕ, m = 57 ∧ (T = (m / 3) * (r_A + r_B - r_C))) : 
r_C = 20 :=
sorry

end pipe_drain_rate_l110_110289


namespace factorize_expression_l110_110360

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l110_110360


namespace max_value_of_expression_l110_110838

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 :=
sorry

end max_value_of_expression_l110_110838


namespace number_of_new_students_l110_110109

theorem number_of_new_students (initial_students end_students students_left : ℕ) 
  (h_initial: initial_students = 33) 
  (h_left: students_left = 18) 
  (h_end: end_students = 29) : 
  initial_students - students_left + (end_students - (initial_students - students_left)) = 14 :=
by
  sorry

end number_of_new_students_l110_110109


namespace length_of_bridge_l110_110347

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass_bridge : ℝ) 
  (train_length_eq : train_length = 400)
  (train_speed_kmh_eq : train_speed_kmh = 60) 
  (time_to_pass_bridge_eq : time_to_pass_bridge = 72)
  : ∃ (bridge_length : ℝ), bridge_length = 800.24 := 
by
  sorry

end length_of_bridge_l110_110347


namespace coin_flip_sequences_l110_110692

theorem coin_flip_sequences : (2 ^ 10 = 1024) :=
by
  sorry

end coin_flip_sequences_l110_110692


namespace radical_conjugate_sum_l110_110506

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l110_110506


namespace tan_alpha_neg_seven_l110_110544

noncomputable def tan_alpha (α : ℝ) := Real.tan α

theorem tan_alpha_neg_seven {α : ℝ} 
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.cos α ^ 2 + Real.sin (Real.pi + 2 * α) = 3 / 10) : 
  tan_alpha α = -7 := 
sorry

end tan_alpha_neg_seven_l110_110544


namespace paperboy_delivery_sequences_l110_110892

noncomputable def D : ℕ → ℕ
| 0       => 1  -- D_0 is a dummy value to facilitate indexing
| 1       => 2
| 2       => 4
| 3       => 7
| (n + 4) => D (n + 3) + D (n + 2) + D (n + 1)

theorem paperboy_delivery_sequences : D 11 = 927 := by
  sorry

end paperboy_delivery_sequences_l110_110892


namespace Kolya_mistake_l110_110652

def boys := ["Vasya", "Kolya", "Petya", "Misha"]

constant num_blue_pencils : ℕ
constant num_green_pencils : ℕ

axiom Vasya_statement : num_blue_pencils >= 4
axiom Kolya_statement : num_green_pencils >= 5
axiom Petya_statement : num_blue_pencils >= 3 ∧ num_green_pencils >= 4
axiom Misha_statement : num_blue_pencils >= 4 ∧ num_green_pencils >= 4

axiom three_truths_one_mistake : 
  (Vasya_statement ∨ ¬Vasya_statement) ∧
  (Kolya_statement ∨ ¬Kolya_statement) ∧
  (Petya_statement ∨ ¬Petya_statement) ∧
  (Misha_statement ∨ ¬Misha_statement) ∧
  ((Vasya_statement ? true : 1) + 
   (Kolya_statement ? true : 1) + 
   (Petya_statement ? true : 1) +
   (Misha_statement ? true : 1) == 3)

theorem Kolya_mistake : ¬Kolya_statement :=
by
  sorry

end Kolya_mistake_l110_110652


namespace compute_expression_l110_110095

theorem compute_expression (x : ℝ) (h : x + 1/x = 3) : 
  (x - 3)^2 + 16 / (x - 3)^2 = 23 := 
  sorry

end compute_expression_l110_110095


namespace unique_cell_distance_50_l110_110629

noncomputable def king_dist (A B: ℤ × ℤ) : ℤ :=
  max (abs (A.1 - B.1)) (abs (A.2 - B.2))

theorem unique_cell_distance_50
  (A B C: ℤ × ℤ)
  (hAB: king_dist A B = 100)
  (hBC: king_dist B C = 100)
  (hCA: king_dist C A = 100) :
  ∃! (X: ℤ × ℤ), king_dist X A = 50 ∧ king_dist X B = 50 ∧ king_dist X C = 50 :=
sorry

end unique_cell_distance_50_l110_110629


namespace sum_of_primes_product_166_l110_110638

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m < n → m > 0 → n % m ≠ 0

theorem sum_of_primes_product_166
    (p1 p2 : ℕ)
    (prime_p1 : is_prime p1)
    (prime_p2 : is_prime p2)
    (product_condition : p1 * p2 = 166) :
    p1 + p2 = 85 :=
    sorry

end sum_of_primes_product_166_l110_110638


namespace sum_of_number_and_its_radical_conjugate_l110_110509

theorem sum_of_number_and_its_radical_conjugate : 
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by
  sorry

end sum_of_number_and_its_radical_conjugate_l110_110509


namespace can_construct_parallelogram_l110_110327

theorem can_construct_parallelogram {a b d1 d2 : ℝ} :
  (a = 3 ∧ b = 5 ∧ (a = b ∨ (‖a + b‖ ≥ ‖d1‖ ∧ ‖a + d1‖ ≥ ‖b‖ ∧ ‖b + d1‖ ≥ ‖a‖))) ∨
  (a ≠ 3 ∨ b ≠ 5 ∨ (a ≠ b ∧ (‖a + b‖ < ‖d1‖ ∨ ‖a + d1‖ < ‖b‖ ∨ ‖b + d1‖ < ‖a‖ ∨ ‖a + d1‖ < ‖d2‖ ∨ ‖b + d1‖ < ‖d2‖ ∨ ‖a + d2‖ < ‖d1‖ ∨ ‖b + d2‖ < ‖d1‖))) ↔ 
  (a = 3 ∧ b = 5 ∧ d1 = 0) :=
sorry

end can_construct_parallelogram_l110_110327


namespace rightmost_three_digits_of_3_pow_2023_l110_110662

theorem rightmost_three_digits_of_3_pow_2023 :
  (3^2023) % 1000 = 787 := 
sorry

end rightmost_three_digits_of_3_pow_2023_l110_110662


namespace coin_flip_sequences_l110_110709

theorem coin_flip_sequences (n : ℕ) : (2^10 = 1024) := 
by rfl

end coin_flip_sequences_l110_110709


namespace number_of_young_fish_l110_110613

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l110_110613


namespace restaurant_total_cost_l110_110182

def total_cost
  (adults kids : ℕ)
  (adult_meal_cost adult_drink_cost adult_dessert_cost kid_drink_cost kid_dessert_cost : ℝ) : ℝ :=
  let num_adults := adults
  let num_kids := kids
  let adult_total := num_adults * (adult_meal_cost + adult_drink_cost + adult_dessert_cost)
  let kid_total := num_kids * (kid_drink_cost + kid_dessert_cost)
  adult_total + kid_total

theorem restaurant_total_cost :
  total_cost 4 9 7 4 3 2 1.5 = 87.5 :=
by
  sorry

end restaurant_total_cost_l110_110182


namespace weight_comparison_l110_110226

theorem weight_comparison :
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  average = 45 ∧ median = 25 ∧ average - median = 20 :=
by
  let weights := [10, 20, 30, 120]
  let average := (10 + 20 + 30 + 120) / 4
  let median := (20 + 30) / 2
  have h1 : average = 45 := sorry
  have h2 : median = 25 := sorry
  have h3 : average - median = 20 := sorry
  exact ⟨h1, h2, h3⟩

end weight_comparison_l110_110226


namespace coin_flips_sequences_count_l110_110725

theorem coin_flips_sequences_count :
  (∃ (n : ℕ), n = 1024 ∧ (∀ (coin_flip : ℕ → bool), (finset.card (finset.image coin_flip (finset.range 10)) = n))) :=
by {
  sorry
}

end coin_flips_sequences_count_l110_110725


namespace distinct_sequences_ten_flips_l110_110715

-- Define the problem condition and question
def flip_count : ℕ := 10

-- Define the function to calculate the number of distinct sequences
def number_of_sequences (n : ℕ) : ℕ := 2 ^ n

-- Statement to be proven
theorem distinct_sequences_ten_flips : number_of_sequences flip_count = 1024 :=
by
  -- Proof goes here
  sorry

end distinct_sequences_ten_flips_l110_110715


namespace factorize_expression_l110_110363

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l110_110363


namespace circumcenter_on_angle_bisector_l110_110475

open EuclideanGeometry

-- Define the problem as Lean 4 statement
theorem circumcenter_on_angle_bisector 
  (A O S B C : Point) (r : ℝ)
  (h_circle_incorner : inscribed_circle O r S)
  (h_symmetry : symmetric_point O A S)
  (h_tangents : tangent_to_circle A O B ∧ tangent_to_circle A O C)
  (h_intersections : intersects_side_far A B S ∧ intersects_side_far A C S) :
  lies_on_angle_bisector (circumcenter_triangle A B C) S :=
sorry

end circumcenter_on_angle_bisector_l110_110475


namespace binary1011_eq_11_l110_110851

-- Define a function to convert a binary number represented as a list of bits to a decimal number.
def binaryToDecimal (bits : List (Fin 2)) : Nat :=
  bits.foldr (λ (bit : Fin 2) (acc : Nat) => acc * 2 + bit.val) 0

-- The binary number 1011 represented as a list of bits.
def binary1011 : List (Fin 2) := [1, 0, 1, 1]

-- The theorem stating that the decimal equivalent of binary 1011 is 11.
theorem binary1011_eq_11 : binaryToDecimal binary1011 = 11 :=
by
  sorry

end binary1011_eq_11_l110_110851


namespace A_investment_l110_110159

theorem A_investment (x : ℝ) (hx : 0 < x) :
  (∃ a b c d e : ℝ,
    a = x ∧ b = 12 ∧ c = 200 ∧ d = 6 ∧ e = 60 ∧ 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    ((a * b) / (a * b + c * d)) * 100 = e)
  → x = 150 :=
by
  sorry

end A_investment_l110_110159


namespace find_second_largest_element_l110_110162

open List

theorem find_second_largest_element 
(a1 a2 a3 a4 a5 : ℕ) 
(h_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4 ∧ 0 < a5) 
(h_sorted : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4 ∧ a4 ≤ a5) 
(h_mean : (a1 + a2 + a3 + a4 + a5) / 5 = 15) 
(h_range : a5 - a1 = 24) 
(h_mode : a2 = 10 ∧ a3 = 10) 
(h_median : a3 = 10) 
(h_three_diff : (a1 ≠ a2 ∨ a1 ≠ a3 ∨ a1 ≠ a4 ∨ a1 ≠ a5) ∧ (a4 ≠ a5)) :
a4 = 11 :=
sorry

end find_second_largest_element_l110_110162


namespace coin_flip_sequences_l110_110734

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110734


namespace somu_age_ratio_l110_110624

theorem somu_age_ratio (S F : ℕ) (h1 : S = 20) (h2 : S - 10 = (F - 10) / 5) : S / F = 1 / 3 :=
by
  sorry

end somu_age_ratio_l110_110624


namespace angle_compute_l110_110954

open Real

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b : ℝ × ℝ := (1, 2)

noncomputable def sub_vec := (b.1 - a.1, b.2 - a.2)
noncomputable def sum_vec := (a.1 + 2 * b.1, a.2 + 2 * b.2)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_between (v₁ v₂ : ℝ × ℝ) : ℝ :=
  arccos (dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂))

theorem angle_compute : angle_between sub_vec sum_vec = π / 4 :=
by {
  sorry
}

end angle_compute_l110_110954


namespace quilt_square_side_length_l110_110868

theorem quilt_square_side_length (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ s : ℝ, (length * width = s * s) ∧ s = 12 :=
by {
  sorry
}

end quilt_square_side_length_l110_110868


namespace alice_savings_l110_110024

def sales : ℝ := 2500
def basic_salary : ℝ := 240
def commission_rate : ℝ := 0.02
def savings_rate : ℝ := 0.10

theorem alice_savings :
  (basic_salary + (sales * commission_rate)) * savings_rate = 29 :=
by
  sorry

end alice_savings_l110_110024


namespace tangent_line_at_zero_l110_110195

noncomputable def curve (x : ℝ) : ℝ := Real.exp (2 * x)

theorem tangent_line_at_zero :
  ∃ m b, (∀ x, (curve x) = m * x + b) ∧
    m = 2 ∧ b = 1 :=
by 
  sorry

end tangent_line_at_zero_l110_110195


namespace balance_scale_with_blue_balls_l110_110189

variables (G Y W B : ℝ)

-- Conditions
def green_to_blue := 4 * G = 8 * B
def yellow_to_blue := 3 * Y = 8 * B
def white_to_blue := 5 * B = 3 * W

-- Proof problem statement
theorem balance_scale_with_blue_balls (h1 : green_to_blue G B) (h2 : yellow_to_blue Y B) (h3 : white_to_blue W B) : 
  3 * G + 3 * Y + 3 * W = 19 * B :=
by sorry

end balance_scale_with_blue_balls_l110_110189


namespace num_cows_correct_l110_110575

-- Definitions from the problem's conditions
def total_animals : ℕ := 500
def percentage_chickens : ℤ := 10
def remaining_animals := total_animals - (percentage_chickens * total_animals / 100)
def goats (cows: ℕ) : ℕ := 2 * cows

-- Statement to prove
theorem num_cows_correct : ∃ cows, remaining_animals = cows + goats cows ∧ 3 * cows = 450 :=
by
  sorry

end num_cows_correct_l110_110575


namespace apple_distribution_l110_110474

theorem apple_distribution : 
  (∀ (a b c d : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → (a + b + c + d = 30) → 
  ∃ k : ℕ, k = (Nat.choose 29 3) ∧ k = 3276) :=
by
  intros a b c d h_pos h_sum
  use Nat.choose 29 3
  have h_eq : Nat.choose 29 3 = 3276 := by sorry
  exact ⟨rfl, h_eq⟩

end apple_distribution_l110_110474


namespace least_value_x_y_z_l110_110002

theorem least_value_x_y_z (x y z : ℕ) (hx : x = 4 * y) (hy : y = 7 * z) (hz : 0 < z) : x - y - z = 19 :=
by
  -- placeholder for actual proof
  sorry

end least_value_x_y_z_l110_110002


namespace eleven_power_2023_mod_50_l110_110667

theorem eleven_power_2023_mod_50 :
  11^2023 % 50 = 31 :=
by
  sorry

end eleven_power_2023_mod_50_l110_110667


namespace right_triangle_perimeter_l110_110893

theorem right_triangle_perimeter (area : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) 
  (h_area : area = 120)
  (h_a : a = 24)
  (h_area_eq : area = (1/2) * a * b)
  (h_c : c^2 = a^2 + b^2) :
  a + b + c = 60 :=
by
  sorry

end right_triangle_perimeter_l110_110893


namespace coin_flip_sequences_l110_110735

theorem coin_flip_sequences : (number_of_flips : ℕ) (number_of_outcomes_per_flip : ℕ) (number_of_flips = 10) (number_of_outcomes_per_flip = 2) : 
  let total_sequences := number_of_outcomes_per_flip ^ number_of_flips 
  in total_sequences = 1024 := 
by 
  sorry

end coin_flip_sequences_l110_110735


namespace bricks_needed_per_square_meter_l110_110249

theorem bricks_needed_per_square_meter 
  (num_rooms : ℕ) (room_length room_breadth : ℕ) (total_bricks : ℕ)
  (h1 : num_rooms = 5)
  (h2 : room_length = 4)
  (h3 : room_breadth = 5)
  (h4 : total_bricks = 340) : 
  (total_bricks / (room_length * room_breadth)) = 17 := 
by
  sorry

end bricks_needed_per_square_meter_l110_110249


namespace limit_of_p_n_is_tenth_l110_110000

noncomputable def p_n (n : ℕ) : ℝ := sorry -- Definition of p_n needs precise formulation.

def tends_to_tenth_as_n_infty (p : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (p n - 1/10) < ε

theorem limit_of_p_n_is_tenth : tends_to_tenth_as_n_infty p_n := sorry

end limit_of_p_n_is_tenth_l110_110000


namespace domain_of_f_monotonicity_of_f_inequality_solution_l110_110083

open Real

noncomputable def f (x : ℝ) : ℝ := log ((1 - x) / (1 + x))

theorem domain_of_f :
  ∀ x, -1 < x ∧ x < 1 → ∃ y, y = f x :=
by
  intro x h
  use log ((1 - x) / (1 + x))
  simp [f]

theorem monotonicity_of_f :
  ∀ x y, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → x < y → f x > f y :=
sorry

theorem inequality_solution :
  ∀ x, f (2 * x - 1) < 0 ↔ (1 / 2 < x ∧ x < 1) :=
sorry

end domain_of_f_monotonicity_of_f_inequality_solution_l110_110083


namespace expand_product_l110_110358

theorem expand_product (y : ℝ) (h : y ≠ 0) : 
  (3 / 7) * ((7 / y) - 14 * y^3 + 21) = (3 / y) - 6 * y^3 + 9 := 
by 
  sorry

end expand_product_l110_110358


namespace train_length_l110_110757

theorem train_length 
  (t1 t2 : ℝ)
  (d2 : ℝ)
  (L : ℝ)
  (V : ℝ)
  (h1 : t1 = 18)
  (h2 : t2 = 27)
  (h3 : d2 = 150.00000000000006)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) :
  L = 300.0000000000001 :=
by
  sorry

end train_length_l110_110757


namespace least_faces_combined_l110_110144

theorem least_faces_combined (a b : ℕ) (h1 : a ≥ 6) (h2 : b ≥ 6)
  (h3 : (∃ k : ℕ, k * a * b = 20) → (∃ m : ℕ, 2 * m = 10 * (k + 10))) 
  (h4 : (∃ n : ℕ, n = (a * b) / 10)) (h5 : ∃ l : ℕ, l = 5) : a + b = 20 :=
by
  sorry

end least_faces_combined_l110_110144


namespace exists_unique_poly_odd_degree_l110_110995

open Polynomial

-- Statement of the theorem to be proven
theorem exists_unique_poly_odd_degree (n : ℕ) (hn : n % 2 = 1) :
  ∃! (P : Polynomial ℚ), P.degree = n ∧ ∀ x, P (x - (1 / x)) = x^n - (1 / x^n) := 
sorry

end exists_unique_poly_odd_degree_l110_110995


namespace problem_solution_l110_110411

noncomputable def problem (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α^2 + p * α - 1 = 0) ∧
  (β^2 + p * β - 1 = 0) ∧
  (γ^2 + q * γ + 1 = 0) ∧
  (δ^2 + q * δ + 1 = 0) →
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2

theorem problem_solution (p q α β γ δ : ℝ) : 
  problem p q α β γ δ := 
by sorry

end problem_solution_l110_110411


namespace find_m_l110_110223

theorem find_m (m : ℝ) : (1 : ℝ) * (-4 : ℝ) + (2 : ℝ) * m = 0 → m = 2 :=
by
  sorry

end find_m_l110_110223


namespace quadratic_positive_difference_l110_110911

theorem quadratic_positive_difference :
  ∀ x : ℝ, x^2 - 5 * x + 15 = x + 55 → x = 10 ∨ x = -4 →
  |10 - (-4)| = 14 :=
by
  intro x h1 h2
  have h3 : x = 10 ∨ x = -4 := h2
  have h4 : |10 - (-4)| = 14 := by norm_num
  exact h4

end quadratic_positive_difference_l110_110911


namespace smaller_angle_at_345_l110_110315

-- Condition definitions
def twelve_hour_analog_clock := true
def minute_hand_at_45 (h : ℕ) : ℝ := 270
def hour_hand_at_345 (h : ℕ) : ℝ := 3 * 30 + (3 / 4) * 30

-- Main theorem statement
theorem smaller_angle_at_345 (h : ℕ) (H : twelve_hour_analog_clock):
  let minute_pos := minute_hand_at_45 h,
      hour_pos := hour_hand_at_345 h,
      angle_diff := abs (minute_pos - hour_pos),
      smaller_angle := min angle_diff (360 - angle_diff)
  in smaller_angle = 157.5 :=
by sorry

end smaller_angle_at_345_l110_110315


namespace area_of_circle_with_endpoints_l110_110290

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def radius (d : ℝ) : ℝ :=
  d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circle_with_endpoints :
  area_of_circle (radius (distance (5, 9) (13, 17))) = 32 * Real.pi :=
by
  sorry

end area_of_circle_with_endpoints_l110_110290


namespace vector_parallel_solution_l110_110386

theorem vector_parallel_solution (x : ℝ) :
  let a := (1, x)
  let b := (x - 1, 2)
  (a.1 * b.2 = a.2 * b.1) → (x = 2 ∨ x = -1) :=
by
  intros a b h
  let a := (1, x)
  let b := (x - 1, 2)
  sorry

end vector_parallel_solution_l110_110386


namespace probability_of_a_l110_110554

open ProbabilityTheory

theorem probability_of_a
  (a b : Event)
  (p : Measure Event)
  (ha : 0 ≤ p a ∧ p a ≤ 1)
  (hb : p b = 2 / 5)
  (indep : indep p a b)
  (hab : p (a ∩ b) = 0.08) :
  p a = 0.4 :=
by
  sorry

end probability_of_a_l110_110554


namespace projection_of_a_onto_b_l110_110201

-- Define the vectors a and b.
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-2, 4)

-- Helper function to calculate the dot product of two vectors.
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Helper function to calculate the squared magnitude of a vector.
def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

-- Define the projection function.
def projection (u v : ℝ × ℝ) : ℝ × ℝ := 
  let coeff := (dot_product u v) / (magnitude_squared v)
  in (coeff * v.1, coeff * v.2)

-- Proposition stating the desired result.
theorem projection_of_a_onto_b :
  projection a b = (-4 / 5, 8 / 5) :=
sorry

end projection_of_a_onto_b_l110_110201


namespace intersection_a_zero_range_of_a_l110_110078

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end intersection_a_zero_range_of_a_l110_110078


namespace inequality_example_l110_110997

theorem inequality_example (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < c) (h4 : b < 0) : a + b < b + c := 
by sorry

end inequality_example_l110_110997


namespace remainder_of_poly_division_l110_110935

theorem remainder_of_poly_division :
  ∀ (x : ℝ), (x^2023 + x + 1) % (x^6 - x^4 + x^2 - 1) = x^7 + x + 1 :=
by
  sorry

end remainder_of_poly_division_l110_110935


namespace find_side_b_l110_110274

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (h_area : ∃ A B C, 1/2 * a * c * sin B = sqrt 3)
  (h_B : B = π / 3) (h_eq : a ^ 2 + c ^ 2 = 3 * a * c) : b = 2 * sqrt 2 :=
by
  sorry

end find_side_b_l110_110274


namespace equal_intercepts_lines_area_two_lines_l110_110082

-- Defining the general equation of the line l with parameter a
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = -(a + 1) * x + 2 - a

-- Problem statement for equal intercepts condition
theorem equal_intercepts_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (x = y ∨ x + y = 2*a + 2)) →
  (a = 2 ∨ a = 0) → 
  (line_eq a 1 (-3) ∨ line_eq a 1 1) :=
sorry

-- Problem statement for triangle area condition
theorem area_two_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / 2 * |x| * |y| = 2)) →
  (a = 8 ∨ a = 0) → 
  (line_eq a 1 (-9) ∨ line_eq a 1 1) :=
sorry

end equal_intercepts_lines_area_two_lines_l110_110082


namespace max_value_fraction_l110_110366

theorem max_value_fraction : ∀ x : ℝ, 
  (∃ x : ℝ, max (1 + (16 / (4 * x^2 + 8 * x + 5))) = 17) :=
by
  sorry

end max_value_fraction_l110_110366


namespace line_intersection_x_value_l110_110048

theorem line_intersection_x_value :
  let line1 (x : ℝ) := 3 * x + 14
  let line2 (x : ℝ) (y : ℝ) := 5 * x - 2 * y = 40
  ∃ x : ℝ, ∃ y : ℝ, (line1 x = y) ∧ (line2 x y) ∧ (x = -68) :=
by
  sorry

end line_intersection_x_value_l110_110048


namespace mayor_vice_mayor_happy_people_l110_110177

theorem mayor_vice_mayor_happy_people :
  (∃ (institutions_per_institution : ℕ) (num_institutions : ℕ),
    institutions_per_institution = 80 ∧
    num_institutions = 6 ∧
    num_institutions * institutions_per_institution = 480) :=
by
  sorry

end mayor_vice_mayor_happy_people_l110_110177


namespace coin_flip_sequences_l110_110741

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l110_110741


namespace delivery_driver_stops_l110_110010

theorem delivery_driver_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) (stops : ℕ) :
  total_boxes = 27 → boxes_per_stop = 9 → stops = total_boxes / boxes_per_stop → stops = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end delivery_driver_stops_l110_110010


namespace trade_and_unification_effects_l110_110768

theorem trade_and_unification_effects :
  let country_A_corn := 8
  let country_B_eggplants := 18
  let country_B_corn := 12
  let country_A_eggplants := 10
  
  -- Part (a): Absolute and comparative advantages
  (country_B_corn > country_A_corn) ∧ (country_B_eggplants > country_A_eggplants) ∧
  let opportunity_cost_A_eggplants := country_A_corn / country_A_eggplants
  let opportunity_cost_A_corn := country_A_eggplants / country_A_corn
  let opportunity_cost_B_eggplants := country_B_corn / country_B_eggplants
  let opportunity_cost_B_corn := country_B_eggplants / country_B_corn
  (opportunity_cost_B_eggplants < opportunity_cost_A_eggplants) ∧ (opportunity_cost_A_corn < opportunity_cost_B_corn) ∧

  -- Part (b): Volumes produced and consumed with trade
  let price := 1
  let income_A := country_A_corn * price
  let income_B := country_B_eggplants * price
  let consumption_A_eggplants := income_A / price / 2
  let consumption_A_corn := country_A_corn / 2
  let consumption_B_corn := income_B / price / 2
  let consumption_B_eggplants := country_B_eggplants / 2
  (consumption_A_eggplants = 4) ∧ (consumption_A_corn = 4) ∧
  (consumption_B_corn = 9) ∧ (consumption_B_eggplants = 9) ∧

  -- Part (c): Volumes after unification without trade
  let unified_eggplants := 18 - (1.5 * 4)
  let unified_corn := 8 + 4
  let total_unified_eggplants := unified_eggplants
  let total_unified_corn := unified_corn
  (total_unified_eggplants = 12) ∧ (total_unified_corn = 12) ->
  
  total_unified_eggplants = 12 ∧ total_unified_corn = 12 ∧
  (total_unified_eggplants < (consumption_A_eggplants + consumption_B_eggplants)) ∧
  (total_unified_corn < (consumption_A_corn + consumption_B_corn))
:= by
  -- Proof omitted
  sorry

end trade_and_unification_effects_l110_110768


namespace fraction_positive_implies_x_greater_than_seven_l110_110242

variable (x : ℝ)

theorem fraction_positive_implies_x_greater_than_seven (h : -6 / (7 - x) > 0) : x > 7 := by
  sorry

end fraction_positive_implies_x_greater_than_seven_l110_110242


namespace find_side_b_l110_110278

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end find_side_b_l110_110278


namespace second_machine_time_l110_110017

theorem second_machine_time (x : ℝ) : 
  (600 / 10) + (1000 / x) = 1000 / 4 ↔ 
  1 / 10 + 1 / x = 1 / 4 :=
by
  sorry

end second_machine_time_l110_110017


namespace distinct_positive_integers_mod_1998_l110_110796

theorem distinct_positive_integers_mod_1998
  (a : Fin 93 → ℕ)
  (h_distinct : Function.Injective a) :
  ∃ m n p q : Fin 93, (m ≠ n ∧ p ≠ q) ∧ (a m - a n) * (a p - a q) % 1998 = 0 :=
by
  sorry

end distinct_positive_integers_mod_1998_l110_110796


namespace gcd_39_91_l110_110877
-- Import the Mathlib library to ensure all necessary functions and theorems are available

-- Lean statement for proving the GCD of 39 and 91 is 13.
theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end gcd_39_91_l110_110877


namespace exchange_rate_l110_110886

theorem exchange_rate (a b : ℕ) (h : 5000 = 60 * a) : b = 75 * a → b = 6250 := by
  sorry

end exchange_rate_l110_110886


namespace find_y_l110_110806

theorem find_y (y : ℤ) (h : 3 ^ 6 = 27 ^ y) : y = 2 := 
by 
  sorry

end find_y_l110_110806


namespace who_made_mistake_l110_110645

-- Defining conditions for the colored pencils
def has_at_least_four_blue_pencils (b : Nat) : Prop := b >= 4
def has_at_least_five_green_pencils (g : Nat) : Prop := g >= 5
def has_at_least_three_blue_four_green_pencils (b g : Nat) : Prop := b >= 3 ∧ g >= 4
def has_at_least_four_blue_four_green_pencils (b g : Nat) : Prop := b >= 4 ∧ g >= 4

-- Statement of the problem
theorem who_made_mistake (b g : Nat) (vasya kolya petya misha : Prop) :
  has_at_least_four_blue_pencils b →
  has_at_least_five_green_pencils g →
  has_at_least_three_blue_four_green_pencils b g →
  has_at_least_four_blue_four_green_pencils b g →
  (∃ T : Set Prop, {vasya, kolya, petya, misha}.Erase T = {vasya, kolya, petya, misha} ∧
    T.Card = 3) →
  (kolya ↔ ¬ g >= 5) := 
sorry

end who_made_mistake_l110_110645


namespace probability_of_drawing_red_ball_l110_110108

def totalBalls : Nat := 3 + 5 + 2
def redBalls : Nat := 3
def probabilityOfRedBall : ℚ := redBalls / totalBalls

theorem probability_of_drawing_red_ball :
  probabilityOfRedBall = 3 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l110_110108


namespace copier_cost_l110_110882

noncomputable def total_time : ℝ := 4 + 25 / 60
noncomputable def first_quarter_hour_cost : ℝ := 6
noncomputable def hourly_cost : ℝ := 8
noncomputable def time_after_first_quarter_hour : ℝ := total_time - 0.25
noncomputable def remaining_cost : ℝ := time_after_first_quarter_hour * hourly_cost
noncomputable def total_cost : ℝ := first_quarter_hour_cost + remaining_cost

theorem copier_cost :
  total_cost = 39.33 :=
by
  -- This statement remains to be proved.
  sorry

end copier_cost_l110_110882


namespace integer_divisibility_l110_110531

open Nat

theorem integer_divisibility {a b : ℕ} :
  (2 * b^2 + 1) ∣ (a^3 + 1) ↔ a = 2 * b^2 + 1 := sorry

end integer_divisibility_l110_110531


namespace complement_union_eq_zero_or_negative_l110_110599

def U : Set ℝ := Set.univ

def P : Set ℝ := { x | x > 1 }

def Q : Set ℝ := { x | x * (x - 2) < 0 }

theorem complement_union_eq_zero_or_negative :
  (U \ (P ∪ Q)) = { x | x ≤ 0 } := by
  sorry

end complement_union_eq_zero_or_negative_l110_110599


namespace range_of_a_l110_110543

def P (a : ℝ) : Set ℝ := { x : ℝ | a - 4 < x ∧ x < a + 4 }
def Q : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 < 0 }

theorem range_of_a (a : ℝ) : (∀ x, Q x → P a x) → -1 < a ∧ a < 5 :=
by
  intro h
  sorry

end range_of_a_l110_110543


namespace part1_intersection_when_a_is_zero_part2_range_of_a_l110_110076

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x < a + 5}

-- Part (1): When a = 0, find A ∩ B
theorem part1_intersection_when_a_is_zero :
  A ∩ B 0 = {x : ℝ | -1 < x ∧ x < 5} :=
sorry

-- Part (2): If A ∪ B = A, find the range of values for a
theorem part2_range_of_a (a : ℝ) :
  (A ∪ B a = A) → (0 < a ∧ a ≤ 1) ∨ (6 ≤ a) :=
sorry

end part1_intersection_when_a_is_zero_part2_range_of_a_l110_110076


namespace find_b_l110_110268

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
variables (h_area : (1/2) * a * c * (Real.sin B) = sqrt 3)
variables (h_B : B = Real.pi / 3)
variables (h_relation : a^2 + c^2 = 3 * a * c)

-- Claim
theorem find_b :
    b = 2 * Real.sqrt 2 :=
  sorry

end find_b_l110_110268


namespace sum_of_number_and_radical_conjugate_l110_110485

theorem sum_of_number_and_radical_conjugate : 
  let a := 15 - Real.sqrt 500 in
  a + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_radical_conjugate_l110_110485


namespace complement_of_set_M_l110_110562

open Set

def universal_set : Set ℝ := univ

def set_M : Set ℝ := {x | x^2 < 2 * x}

def complement_M : Set ℝ := compl set_M

theorem complement_of_set_M :
  complement_M = {x | x ≤ 0 ∨ x ≥ 2} :=
sorry

end complement_of_set_M_l110_110562


namespace negation_p_l110_110563

theorem negation_p (p : Prop) : 
  (∃ x : ℝ, x^2 ≥ x) ↔ ¬ (∀ x : ℝ, x^2 < x) :=
by 
  -- The proof is omitted
  sorry

end negation_p_l110_110563


namespace sum_of_exponents_l110_110906

theorem sum_of_exponents : 
  (-1)^(2010) + (-1)^(2013) + 1^(2014) + (-1)^(2016) = 0 := 
by
  sorry

end sum_of_exponents_l110_110906


namespace sum_two_numbers_l110_110859

theorem sum_two_numbers (x y : ℝ) (h₁ : x * y = 16) (h₂ : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 :=
by
  -- Proof follows the steps outlined in the solution, but this is where the proof ends for now.
  sorry

end sum_two_numbers_l110_110859


namespace range_of_m_l110_110122

noncomputable def p (x : ℝ) : Prop := (x^3 - 4*x) / (2*x) ≤ 0
noncomputable def q (x m : ℝ) : Prop := (x^2 - (2*m + 1)*x + m^2 + m) ≤ 0

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, p x → q x m) ∧ ¬ (∀ x : ℝ, p x → q x m) ↔ m ∈ Set.Ico (-2 : ℝ) (-1) ∪ Set.Ioc 0 (1 : ℝ) :=
by
  sorry

end range_of_m_l110_110122
