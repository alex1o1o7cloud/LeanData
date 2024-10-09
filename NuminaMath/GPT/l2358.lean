import Mathlib

namespace find_number_l2358_235840

theorem find_number (n : ℝ) :
  (n + 2 * 1.5)^5 = (1 + 3 * 1.5)^4 → n = 0.72 :=
sorry

end find_number_l2358_235840


namespace jordan_wins_two_games_l2358_235826

theorem jordan_wins_two_games 
  (Peter_wins : ℕ) 
  (Peter_losses : ℕ)
  (Emma_wins : ℕ) 
  (Emma_losses : ℕ)
  (Jordan_losses : ℕ) 
  (hPeter : Peter_wins = 5)
  (hPeterL : Peter_losses = 4)
  (hEmma : Emma_wins = 4)
  (hEmmaL : Emma_losses = 5)
  (hJordanL : Jordan_losses = 2) : ∃ (J : ℕ), J = 2 :=
by
  -- The proof will go here
  sorry

end jordan_wins_two_games_l2358_235826


namespace distance_to_destination_l2358_235876

theorem distance_to_destination (x : ℕ) 
    (condition_1 : True)  -- Manex is a tour bus driver. Ignore in the proof.
    (condition_2 : True)  -- Ignores the fact that the return trip is using a different path.
    (condition_3 : x / 30 + (x + 10) / 30 + 2 = 6) : 
    x = 55 :=
sorry

end distance_to_destination_l2358_235876


namespace find_number_l2358_235861

theorem find_number : ∃ x : ℝ, 0 < x ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_number_l2358_235861


namespace slopes_product_l2358_235893

variables {a b c x0 y0 alpha beta : ℝ}
variables {P Q : ℝ × ℝ}
variables (M : ℝ × ℝ) (kPQ kOM : ℝ)

-- Conditions: a, b are positive real numbers
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Condition: b^2 = a c
axiom b_squared_eq_a_mul_c : b^2 = a * c

-- Condition: P and Q lie on the hyperbola
axiom P_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1
axiom Q_on_hyperbola : (Q.1^2 / a^2) - (Q.2^2 / b^2) = 1

-- Condition: M is the midpoint of P and Q
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Condition: Slopes kPQ and kOM exist
axiom kOM_def : kOM = y0 / x0
axiom kPQ_def : kPQ = beta / alpha

-- Theorem: Value of the product of the slopes
theorem slopes_product : kPQ * kOM = (1 + Real.sqrt 5) / 2 :=
sorry

end slopes_product_l2358_235893


namespace intersection_of_intervals_l2358_235892

theorem intersection_of_intervals :
  let A := {x : ℝ | x < -3}
  let B := {x : ℝ | x > -4}
  A ∩ B = {x : ℝ | -4 < x ∧ x < -3} :=
by
  sorry

end intersection_of_intervals_l2358_235892


namespace abs_abc_eq_abs_k_l2358_235881

variable {a b c k : ℝ}

noncomputable def distinct_nonzero (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem abs_abc_eq_abs_k (h_distinct : distinct_nonzero a b c)
                          (h_nonzero_k : k ≠ 0)
                          (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) :
  |a * b * c| = |k| :=
by
  sorry

end abs_abc_eq_abs_k_l2358_235881


namespace find_k_l2358_235846

/--
Given a system of linear equations:
1) x + 2 * y = -a + 1
2) x - 3 * y = 4 * a + 6
If the expression k * x - y remains unchanged regardless of the value of the constant a, 
show that k = -1.
-/
theorem find_k 
  (a x y k : ℝ) 
  (h1 : x + 2 * y = -a + 1) 
  (h2 : x - 3 * y = 4 * a + 6)
  (h3 : ∀ a₁ a₂ x₁ x₂ y₁ y₂, (x₁ + 2 * y₁ = -a₁ + 1) → (x₁ - 3 * y₁ = 4 * a₁ + 6) → 
                               (x₂ + 2 * y₂ = -a₂ + 1) → (x₂ - 3 * y₂ = 4 * a₂ + 6) → 
                               (k * x₁ - y₁ = k * x₂ - y₂)) : 
  k = -1 :=
  sorry

end find_k_l2358_235846


namespace sum_of_digits_eq_11_l2358_235808

-- Define the problem conditions
variables (p q r : ℕ)
variables (h1 : 1 ≤ p ∧ p ≤ 9)
variables (h2 : 1 ≤ q ∧ q ≤ 9)
variables (h3 : 1 ≤ r ∧ r ≤ 9)
variables (h4 : p ≠ q ∧ p ≠ r ∧ q ≠ r)
variables (h5 : (10 * p + q) * (10 * p + r) = 221)

-- Define the theorem
theorem sum_of_digits_eq_11 : p + q + r = 11 :=
by
  sorry

end sum_of_digits_eq_11_l2358_235808


namespace quadratic_real_roots_range_find_k_l2358_235862

theorem quadratic_real_roots_range (k : ℝ) (h : ∃ x1 x2 : ℝ, x^2 - 2 * (k - 1) * x + k^2 = 0):
  k ≤ 1/2 :=
  sorry

theorem find_k (k : ℝ) (x1 x2 : ℝ) (h₁ : x^2 - 2 * (k - 1) * x + k^2 = 0)
  (h₂ : x₁ * x₂ + x₁ + x₂ - 1 = 0) (h_range : k ≤ 1/2) :
    k = -3 :=
  sorry

end quadratic_real_roots_range_find_k_l2358_235862


namespace roots_value_l2358_235868

theorem roots_value (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) (h2 : Polynomial.eval n (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) : m^2 + 4 * m + n = -2 := 
sorry

end roots_value_l2358_235868


namespace A_8_coords_l2358_235852

-- Define point as a structure
structure Point where
  x : Int
  y : Int

-- Initial point A
def A : Point := {x := 3, y := 2}

-- Symmetric point about the y-axis
def sym_y (p : Point) : Point := {x := -p.x, y := p.y}

-- Symmetric point about the origin
def sym_origin (p : Point) : Point := {x := -p.x, y := -p.y}

-- Symmetric point about the x-axis
def sym_x (p : Point) : Point := {x := p.x, y := -p.y}

-- Function to get the n-th symmetric point in the sequence
def sym_point (n : Nat) : Point :=
  match n % 3 with
  | 0 => A
  | 1 => sym_y A
  | 2 => sym_origin (sym_y A)
  | _ => A  -- Fallback case (should not be reachable for n >= 0)

theorem A_8_coords : sym_point 8 = {x := 3, y := -2} := sorry

end A_8_coords_l2358_235852


namespace add_zero_eq_self_l2358_235800

theorem add_zero_eq_self (n x : ℤ) (h : n + x = n) : x = 0 := 
sorry

end add_zero_eq_self_l2358_235800


namespace julien_swims_50_meters_per_day_l2358_235836

-- Definitions based on given conditions
def distance_julien_swims_per_day : ℕ := 50
def distance_sarah_swims_per_day (J : ℕ) : ℕ := 2 * J
def distance_jamir_swims_per_day (J : ℕ) : ℕ := distance_sarah_swims_per_day J + 20
def combined_distance_per_day (J : ℕ) : ℕ := J + distance_sarah_swims_per_day J + distance_jamir_swims_per_day J
def combined_distance_per_week (J : ℕ) : ℕ := 7 * combined_distance_per_day J

-- Proof statement 
theorem julien_swims_50_meters_per_day :
  combined_distance_per_week distance_julien_swims_per_day = 1890 :=
by
  -- We are formulating the proof without solving it, to be proven formally in Lean
  sorry

end julien_swims_50_meters_per_day_l2358_235836


namespace exists_equal_mod_p_l2358_235865

theorem exists_equal_mod_p (p : ℕ) [hp_prime : Fact p.Prime] 
  (m : Fin p → ℕ) 
  (h_consecutive : ∀ i j : Fin p, (i : ℕ) < j → m i + 1 = m j) 
  (sigma : Equiv (Fin p) (Fin p)) :
  ∃ (k l : Fin p), k ≠ l ∧ (m k * m (sigma k) - m l * m (sigma l)) % p = 0 :=
by
  sorry

end exists_equal_mod_p_l2358_235865


namespace inequality_sum_l2358_235841

theorem inequality_sum
  (x y z : ℝ)
  (h : abs (x * y * z) = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 := 
sorry

end inequality_sum_l2358_235841


namespace solution_set_of_quadratic_inequality_l2358_235802

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | 2 - x - x^2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end solution_set_of_quadratic_inequality_l2358_235802


namespace coordinates_with_respect_to_origin_l2358_235863

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l2358_235863


namespace expenditure_of_negative_l2358_235815

def income := 5000
def expenditure (x : Int) : Int := -x

theorem expenditure_of_negative (x : Int) : expenditure (-x) = x :=
by
  sorry

example : expenditure (-400) = 400 :=
by 
  exact expenditure_of_negative 400

end expenditure_of_negative_l2358_235815


namespace no_solution_to_inequality_l2358_235855

theorem no_solution_to_inequality (x : ℝ) (h : x ≥ -1/4) : ¬(-1 - 1 / (3 * x + 4) < 2) :=
by sorry

end no_solution_to_inequality_l2358_235855


namespace oldest_person_Jane_babysat_age_l2358_235885

def Jane_current_age : ℕ := 32
def Jane_stop_babysitting_age : ℕ := 22 -- 32 - 10
def max_child_age_when_Jane_babysat : ℕ := Jane_stop_babysitting_age / 2  -- 22 / 2
def years_since_Jane_stopped : ℕ := Jane_current_age - Jane_stop_babysitting_age -- 32 - 22

theorem oldest_person_Jane_babysat_age :
  max_child_age_when_Jane_babysat + years_since_Jane_stopped = 21 :=
by
  sorry

end oldest_person_Jane_babysat_age_l2358_235885


namespace quadractic_b_value_l2358_235823

def quadratic_coefficients (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem quadractic_b_value :
  ∀ (a b c : ℝ), quadratic_coefficients 1 (-2) (-3) (x : ℝ) → 
  b = -2 := by
  sorry

end quadractic_b_value_l2358_235823


namespace product_of_modified_numbers_less_l2358_235850

theorem product_of_modified_numbers_less
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := 
by {
   sorry
}

end product_of_modified_numbers_less_l2358_235850


namespace weight_of_new_person_l2358_235883

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person replaces one of them. 
The weight of the replaced person is 65 kg. 
Prove that the weight of the new person is 128 kg. 
-/
theorem weight_of_new_person (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  (avg_increase = 6.3) → 
  (old_weight = 65) → 
  (new_weight = old_weight + 10 * avg_increase) → 
  new_weight = 128 := 
by
  intros
  sorry

end weight_of_new_person_l2358_235883


namespace equation_of_trisection_line_l2358_235803

/-- Let P be the point (1, 2) and let A and B be the points (2, 3) and (-3, 0), respectively. 
    One of the lines through point P and a trisection point of the line segment joining A and B has 
    the equation 3x + 7y = 17. -/
theorem equation_of_trisection_line :
  let P : ℝ × ℝ := (1, 2)
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (-3, 0)
  -- Definition of the trisection points
  let T1 : ℝ × ℝ := ((2 + (-3 - 2) / 3) / 1, (3 + (0 - 3) / 3) / 1) -- First trisection point
  let T2 : ℝ × ℝ := ((2 + 2 * (-3 - 2) / 3) / 1, (3 + 2 * (0 - 3) / 3) / 1) -- Second trisection point
  -- Equation of the line through P and T2 is 3x + 7y = 17
  3 * (P.1 + P.2) + 7 * (P.2 + T2.2) = 17 :=
sorry

end equation_of_trisection_line_l2358_235803


namespace petya_cannot_have_equal_coins_l2358_235817

def petya_initial_two_kopeck_coins : Nat := 1
def petya_initial_ten_kopeck_coins : Nat := 0
def petya_use_ten_kopeck (T G : Nat) : Nat := G - 1 + T + 5
def petya_use_two_kopeck (T G : Nat) : Nat := T - 1 + G + 5

theorem petya_cannot_have_equal_coins : ¬ (∃ n : Nat, 
  ∃ T G : Nat, 
    T = G ∧ 
    (n = petya_use_ten_kopeck T G ∨ n = petya_use_two_kopeck T G ∨ n = petya_initial_two_kopeck_coins + petya_initial_ten_kopeck_coins)) := 
by
  sorry

end petya_cannot_have_equal_coins_l2358_235817


namespace triangle_side_possible_values_l2358_235867

theorem triangle_side_possible_values (m : ℝ) (h1 : 1 < m) (h2 : m < 7) : 
  m = 5 :=
by
  sorry

end triangle_side_possible_values_l2358_235867


namespace watermelon_seeds_l2358_235806

theorem watermelon_seeds (n_slices : ℕ) (total_seeds : ℕ) (B W : ℕ) 
  (h1: n_slices = 40) 
  (h2: B = W) 
  (h3 : n_slices * B + n_slices * W = total_seeds)
  (h4 : total_seeds = 1600) : B = 20 :=
by {
  sorry
}

end watermelon_seeds_l2358_235806


namespace ratio_of_areas_l2358_235895

-- Define the conditions
variable (s : ℝ) (h_pos : s > 0)
-- The total perimeter of four small square pens is reused for one large square pen
def total_fencing_length := 16 * s
def large_square_side_length := 4 * s

-- Define the areas
def small_squares_total_area := 4 * s^2
def large_square_area := (4 * s)^2

-- The statement to prove
theorem ratio_of_areas : small_squares_total_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l2358_235895


namespace find_a_2013_l2358_235873

def sequence_a (n : ℕ) : ℤ :=
  if n = 0 then 2
  else if n = 1 then 5
  else sequence_a (n - 1) - sequence_a (n - 2)

theorem find_a_2013 :
  sequence_a 2013 = 3 :=
sorry

end find_a_2013_l2358_235873


namespace arithmetic_seq_general_formula_l2358_235854

-- Definitions based on given conditions
def f (x : ℝ) := x^2 - 2*x + 4
def a (n : ℕ) (d : ℝ) := f (d + n - 1) 

-- The general term formula for the arithmetic sequence
theorem arithmetic_seq_general_formula (d : ℝ) :
  (a 1 d = f (d - 1)) →
  (a 3 d = f (d + 1)) →
  (∀ n : ℕ, a n d = 2*n + 1) :=
by
  intros h1 h3
  sorry

end arithmetic_seq_general_formula_l2358_235854


namespace roses_in_vase_now_l2358_235819

-- Definitions of initial conditions and variables
def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def orchids_cut : ℕ := 19
def orchids_now : ℕ := 21

-- The proof problem to show that the number of roses now is still the same as initially.
theorem roses_in_vase_now : initial_roses = 12 :=
by
  -- The proof itself is left as an exercise (add proof here)
  sorry

end roses_in_vase_now_l2358_235819


namespace desired_percentage_of_alcohol_l2358_235848

def solution_x_alcohol_by_volume : ℝ := 0.10
def solution_y_alcohol_by_volume : ℝ := 0.30
def volume_solution_x : ℝ := 200
def volume_solution_y : ℝ := 600

theorem desired_percentage_of_alcohol :
  ((solution_x_alcohol_by_volume * volume_solution_x + solution_y_alcohol_by_volume * volume_solution_y) / 
  (volume_solution_x + volume_solution_y)) * 100 = 25 := 
sorry

end desired_percentage_of_alcohol_l2358_235848


namespace moles_of_MgCO3_formed_l2358_235875

theorem moles_of_MgCO3_formed 
  (moles_MgO : ℕ) (moles_CO2 : ℕ)
  (h_eq : moles_MgO = 3 ∧ moles_CO2 = 3)
  (balanced_eq : ∀ n : ℕ, n * MgO + n * CO2 = n * MgCO3) : 
  moles_MgCO3 = 3 :=
by
  sorry

end moles_of_MgCO3_formed_l2358_235875


namespace solve_equation_l2358_235814

theorem solve_equation (x : ℝ) (h : 2 * x + 6 = 2 + 3 * x) : x = 4 :=
by
  sorry

end solve_equation_l2358_235814


namespace negation_of_p_l2358_235884

def p := ∃ n : ℕ, n^2 > 2 * n - 1

theorem negation_of_p : ¬ p ↔ ∀ n : ℕ, n^2 ≤ 2 * n - 1 :=
by sorry

end negation_of_p_l2358_235884


namespace total_marks_l2358_235858

-- Variables and conditions
variables (M C P : ℕ)
variable (h1 : C = P + 20)
variable (h2 : (M + C) / 2 = 40)

-- Theorem statement
theorem total_marks (M C P : ℕ) (h1 : C = P + 20) (h2 : (M + C) / 2 = 40) : M + P = 60 :=
sorry

end total_marks_l2358_235858


namespace negation_proposition_l2358_235816

theorem negation_proposition :
  (¬(∀ x : ℝ, x^2 - x + 2 < 0) ↔ ∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
sorry

end negation_proposition_l2358_235816


namespace fluorescent_tubes_count_l2358_235822

theorem fluorescent_tubes_count 
  (x y : ℕ)
  (h1 : x + y = 13)
  (h2 : x / 3 + y / 2 = 5) : x = 9 :=
by
  sorry

end fluorescent_tubes_count_l2358_235822


namespace billboard_dimensions_l2358_235843

theorem billboard_dimensions (photo_width_cm : ℕ) (photo_length_dm : ℕ) (billboard_area_m2 : ℕ)
  (h1 : photo_width_cm = 30) (h2 : photo_length_dm = 4) (h3 : billboard_area_m2 = 48) :
  ∃ photo_length_cm : ℕ, photo_length_cm = 40 ∧
  ∃ k : ℕ, k = 20 ∧
  ∃ billboard_width_m billboard_length_m : ℕ,
    billboard_width_m = photo_width_cm * k / 100 ∧ 
    billboard_length_m = photo_length_cm * k / 100 ∧ 
    billboard_width_m = 6 ∧ 
    billboard_length_m = 8 := by
  sorry

end billboard_dimensions_l2358_235843


namespace range_of_target_function_l2358_235853

noncomputable def target_function (x : ℝ) : ℝ :=
  1 - 1 / (x^2 - 1)

theorem range_of_target_function :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ target_function x = y ↔ y ∈ (Set.Iio 1 ∪ Set.Ici 2) :=
by
  sorry

end range_of_target_function_l2358_235853


namespace train_length_is_sixteenth_mile_l2358_235810

theorem train_length_is_sixteenth_mile
  (train_speed : ℕ)
  (bridge_length : ℕ)
  (man_speed : ℕ)
  (cross_time : ℚ)
  (man_distance : ℚ)
  (length_of_train : ℚ)
  (h1 : train_speed = 80)
  (h2 : bridge_length = 1)
  (h3 : man_speed = 5)
  (h4 : cross_time = bridge_length / train_speed)
  (h5 : man_distance = man_speed * cross_time)
  (h6 : length_of_train = man_distance) :
  length_of_train = 1 / 16 :=
by sorry

end train_length_is_sixteenth_mile_l2358_235810


namespace range_for_a_l2358_235834

theorem range_for_a (f : ℝ → ℝ) (a : ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 8 = 1/4 →
  f (a+1) < f 2 →
  a < -3 ∨ a > 1 :=
by
  intros h1 h2 h3
  sorry

end range_for_a_l2358_235834


namespace solutions_of_quadratic_l2358_235812

theorem solutions_of_quadratic 
  (p q : ℚ) 
  (h₁ : 2 * p * p + 11 * p - 21 = 0) 
  (h₂ : 2 * q * q + 11 * q - 21 = 0) : 
  (p - q) * (p - q) = 289 / 4 := 
sorry

end solutions_of_quadratic_l2358_235812


namespace box_volume_l2358_235870

theorem box_volume (x : ℕ) (h_ratio : (x > 0)) (V : ℕ) (h_volume : V = 20 * x^3) : V = 160 :=
by
  sorry

end box_volume_l2358_235870


namespace relationship_among_a_ae_ea_minus_one_l2358_235859

theorem relationship_among_a_ae_ea_minus_one (a : ℝ) (h : 0 < a ∧ a < 1) :
  (Real.exp a - 1 > a ∧ a > Real.exp a - 1 ∧ a > a^(Real.exp 1)) :=
by
  sorry

end relationship_among_a_ae_ea_minus_one_l2358_235859


namespace find_sum_of_abc_l2358_235880

variable (a b c : ℝ)

-- Given conditions
axiom h1 : a^2 + a * b + b^2 = 1
axiom h2 : b^2 + b * c + c^2 = 3
axiom h3 : c^2 + c * a + a^2 = 4

-- Positivity constraints
axiom ha : a > 0
axiom hb : b > 0
axiom hc : c > 0

theorem find_sum_of_abc : a + b + c = Real.sqrt 7 := 
by
  sorry

end find_sum_of_abc_l2358_235880


namespace largest_k_for_right_triangle_l2358_235844

noncomputable def k : ℝ := (3 * Real.sqrt 2 - 4) / 2

theorem largest_k_for_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) :
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3 :=
sorry

end largest_k_for_right_triangle_l2358_235844


namespace find_S10_l2358_235829

noncomputable def S (n : ℕ) : ℤ := 2 * (-2 ^ (n - 1)) + 1

theorem find_S10 : S 10 = -1023 :=
by
  sorry

end find_S10_l2358_235829


namespace solve_for_a_l2358_235818

theorem solve_for_a (a : ℝ) (y : ℝ) (h1 : 4 * 2 + y = a) (h2 : 2 * 2 + 5 * y = 3 * a) : a = 18 :=
  sorry

end solve_for_a_l2358_235818


namespace find_a_l2358_235838

theorem find_a (a : ℤ) : 0 ≤ a ∧ a ≤ 13 ∧ (51^2015 + a) % 13 = 0 → a = 1 :=
by { sorry }

end find_a_l2358_235838


namespace quotient_real_iff_quotient_purely_imaginary_iff_l2358_235899

variables {a b c d : ℝ} -- Declare real number variables

-- Problem 1: Proving the necessary and sufficient condition for the quotient to be a real number
theorem quotient_real_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ i : ℝ, ∃ r : ℝ, a/c = r ∧ b/d = 0) ↔ (a * d - b * c = 0) := 
by sorry -- Proof to be filled in

-- Problem 2: Proving the necessary and sufficient condition for the quotient to be a purely imaginary number
theorem quotient_purely_imaginary_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ r : ℝ, ∃ i : ℝ, a/c = 0 ∧ b/d = i) ↔ (a * c + b * d = 0) := 
by sorry -- Proof to be filled in

end quotient_real_iff_quotient_purely_imaginary_iff_l2358_235899


namespace q_computation_l2358_235805

def q : ℤ → ℤ → ℤ :=
  λ x y =>
    if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
    else if x < 0 ∧ y < 0 then x - 3 * y
    else 2 * x + y

theorem q_computation : q (q 2 (-2)) (q (-4) (-1)) = 3 :=
by {
  sorry
}

end q_computation_l2358_235805


namespace toll_for_18_wheel_truck_l2358_235801

-- Define the number of wheels on the front axle and the other axles
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def total_wheels : ℕ := 18

-- Define the toll formula
def toll (x : ℕ) : ℝ := 3.50 + 0.50 * (x - 2)

-- Calculate the number of axles for the 18-wheel truck
def num_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the expected toll for the given number of axles
def expected_toll : ℝ := 5.00

-- State the theorem
theorem toll_for_18_wheel_truck : toll num_axles = expected_toll := by
    sorry

end toll_for_18_wheel_truck_l2358_235801


namespace count_not_divisible_by_2_3_5_l2358_235856

theorem count_not_divisible_by_2_3_5 : 
  let count_div_2 := (100 / 2)
  let count_div_3 := (100 / 3)
  let count_div_5 := (100 / 5)
  let count_div_6 := (100 / 6)
  let count_div_10 := (100 / 10)
  let count_div_15 := (100 / 15)
  let count_div_30 := (100 / 30)
  100 - (count_div_2 + count_div_3 + count_div_5) 
      + (count_div_6 + count_div_10 + count_div_15) 
      - count_div_30 = 26 :=
by
  let count_div_2 := 50
  let count_div_3 := 33
  let count_div_5 := 20
  let count_div_6 := 16
  let count_div_10 := 10
  let count_div_15 := 6
  let count_div_30 := 3
  sorry

end count_not_divisible_by_2_3_5_l2358_235856


namespace volume_eq_three_times_other_two_l2358_235878

-- declare the given ratio of the radii
def r1 : ℝ := 1
def r2 : ℝ := 2
def r3 : ℝ := 3

-- calculate the volumes based on the given radii
noncomputable def V (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- defining the volumes of the three spheres
noncomputable def V1 : ℝ := V r1
noncomputable def V2 : ℝ := V r2
noncomputable def V3 : ℝ := V r3

theorem volume_eq_three_times_other_two : V3 = 3 * (V1 + V2) := 
by
  sorry

end volume_eq_three_times_other_two_l2358_235878


namespace best_play_wins_probability_best_play_wins_with_certainty_l2358_235847

-- Define the conditions

variables (n : ℕ)

-- Part (a): Probability that the best play wins
theorem best_play_wins_probability (hn_pos : 0 < n) : 
  1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) = 1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) :=
  by sorry

-- Part (b): With more than two plays, the best play wins with certainty
theorem best_play_wins_with_certainty (s : ℕ) (hs : 2 < s) : 
  1 = 1 :=
  by sorry

end best_play_wins_probability_best_play_wins_with_certainty_l2358_235847


namespace range_of_m_l2358_235889

noncomputable def common_points (k : ℝ) (m : ℝ) := 
  ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)

theorem range_of_m (k : ℝ) (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)) ↔ 
  (m ∈ (Set.Ioo 1 5 ∪ Set.Ioi 5)) :=
by
  sorry

end range_of_m_l2358_235889


namespace arithmetic_sequence_sum_six_l2358_235830

open Nat

noncomputable def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  3 * (2 * a1 + 5 * d) / 3

theorem arithmetic_sequence_sum_six (a : ℕ → ℚ) (h : a 2 + a 5 = 2 / 3) : sum_first_six_terms a = 2 :=
by
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  have eq1 : a 5 = a1 + 4 * d := by sorry
  have eq2 : 3 * (2 * a1 + 5 * d) / 3 = (2 : ℚ) := by sorry
  sorry

end arithmetic_sequence_sum_six_l2358_235830


namespace distance_between_adjacent_symmetry_axes_l2358_235874

noncomputable def f (x : ℝ) : ℝ := (Real.cos (3 * x))^2 - 1/2

theorem distance_between_adjacent_symmetry_axes :
  (∃ x : ℝ, f x = f (x + π / 3)) → (∃ d : ℝ, d = π / 6) :=
by
  -- Prove the distance is π / 6 based on the properties of f(x).
  sorry

end distance_between_adjacent_symmetry_axes_l2358_235874


namespace find_other_number_l2358_235887

/-- Given HCF(A, B), LCM(A, B), and a known A, proves the value of B. -/
theorem find_other_number (A B : ℕ) 
  (hcf : Nat.gcd A B = 16) 
  (lcm : Nat.lcm A B = 396) 
  (a_val : A = 36) : B = 176 :=
by
  sorry

end find_other_number_l2358_235887


namespace sum_powers_of_i_l2358_235896

variable (n : ℕ) (i : ℂ) (h_multiple_of_6 : n % 6 = 0) (h_i : i^2 = -1)

theorem sum_powers_of_i (h_n6 : n = 6) :
    1 + 2*i + 3*i^2 + 4*i^3 + 5*i^4 + 6*i^5 + 7*i^6 = 6*i - 7 := by
  sorry

end sum_powers_of_i_l2358_235896


namespace problem1_l2358_235897

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
sorry

end problem1_l2358_235897


namespace find_original_price_l2358_235869

theorem find_original_price (SP GP : ℝ) (h_SP : SP = 1150) (h_GP : GP = 27.77777777777778) :
  ∃ CP : ℝ, CP = 900 :=
by
  sorry

end find_original_price_l2358_235869


namespace min_c_value_l2358_235833

theorem min_c_value 
  (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + 1 = b)
  (h6 : b + 1 = c)
  (h7 : c + 1 = d)
  (h8 : d + 1 = e)
  (h9 : ∃ k : ℕ, k ^ 2 = b + c + d)
  (h10 : ∃ m : ℕ, m ^ 3 = a + b + c + d + e) : 
  c = 675 := 
sorry

end min_c_value_l2358_235833


namespace water_settles_at_34_cm_l2358_235835

-- Conditions definitions
def h : ℝ := 40 -- Initial height of the liquids in cm
def ρ_w : ℝ := 1000 -- Density of water in kg/m^3
def ρ_o : ℝ := 700  -- Density of oil in kg/m^3

-- Given the conditions provided above,
-- prove that the new height level of water in the first vessel is 34 cm
theorem water_settles_at_34_cm :
  (40 / (1 + (ρ_o / ρ_w))) = 34 := 
sorry

end water_settles_at_34_cm_l2358_235835


namespace gcd_of_abcd_dcba_l2358_235828

theorem gcd_of_abcd_dcba : 
  ∀ (a : ℕ), 0 ≤ a ∧ a ≤ 3 → 
  gcd (2332 * a + 7112) (2332 * (a + 1) + 7112) = 2 ∧ 
  gcd (2332 * (a + 1) + 7112) (2332 * (a + 2) + 7112) = 2 ∧ 
  gcd (2332 * (a + 2) + 7112) (2332 * (a + 3) + 7112) = 2 := 
by 
  sorry

end gcd_of_abcd_dcba_l2358_235828


namespace evaluate_expression_l2358_235827

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem evaluate_expression : spadesuit 3 (spadesuit 6 5) = -112 := by
  sorry

end evaluate_expression_l2358_235827


namespace instantaneous_velocity_at_t2_l2358_235824

noncomputable def s (t : ℝ) : ℝ := t^3 - t^2 + 2 * t

theorem instantaneous_velocity_at_t2 : 
  deriv s 2 = 10 := 
by
  sorry

end instantaneous_velocity_at_t2_l2358_235824


namespace proportional_function_property_l2358_235825

theorem proportional_function_property :
  (∀ x, ∃ y, y = -3 * x ∧
  (x = 0 → y = 0) ∧
  (x > 0 → y < 0) ∧
  (x < 0 → y > 0) ∧
  (x = 1 → y = -3) ∧
  (∀ x, y = -3 * x → (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0))) :=
by
  sorry

end proportional_function_property_l2358_235825


namespace molecular_physics_statements_l2358_235890

theorem molecular_physics_statements :
  (¬A) ∧ B ∧ C ∧ D :=
by sorry

end molecular_physics_statements_l2358_235890


namespace compare_compound_interest_l2358_235866

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end compare_compound_interest_l2358_235866


namespace gabor_can_cross_l2358_235832

open Real

-- Definitions based on conditions
def river_width : ℝ := 100
def total_island_perimeter : ℝ := 800
def banks_parallel : Prop := true

theorem gabor_can_cross (w : ℝ) (p : ℝ) (bp : Prop) : 
  w = river_width → 
  p = total_island_perimeter → 
  bp = banks_parallel → 
  ∃ d : ℝ, d ≤ 300 := 
by
  sorry

end gabor_can_cross_l2358_235832


namespace cake_remaining_l2358_235860

theorem cake_remaining (T J: ℝ) (h1: T = 0.60) (h2: J = 0.25) :
  (1 - ((1 - T) * J + T)) = 0.30 :=
by
  sorry

end cake_remaining_l2358_235860


namespace eq_zero_l2358_235877

variable {x y z : ℤ}

theorem eq_zero (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end eq_zero_l2358_235877


namespace problem_statement_l2358_235813

variable (a b c : ℝ)

theorem problem_statement
  (h1 : a + b = 100)
  (h2 : b + c = 140) :
  c - a = 40 :=
sorry

end problem_statement_l2358_235813


namespace log_add_property_l2358_235839

theorem log_add_property (log : ℝ → ℝ) (h1 : ∀ a b : ℝ, 0 < a → 0 < b → log a + log b = log (a * b)) (h2 : log 10 = 1) :
  log 5 + log 2 = 1 :=
by
  sorry

end log_add_property_l2358_235839


namespace books_left_over_l2358_235894

theorem books_left_over 
  (n_boxes : ℕ) (books_per_box : ℕ) (books_per_new_box : ℕ)
  (total_books : ℕ) (full_boxes : ℕ) (books_left : ℕ) : 
  n_boxes = 1421 → 
  books_per_box = 27 → 
  books_per_new_box = 35 →
  total_books = n_boxes * books_per_box →
  full_boxes = total_books / books_per_new_box →
  books_left = total_books % books_per_new_box →
  books_left = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end books_left_over_l2358_235894


namespace solution_l2358_235820

open Set

theorem solution (A B : Set ℤ) :
  (∀ x, x ∈ A ∨ x ∈ B) →
  (∀ x, x ∈ A → (x - 1) ∈ B) →
  (∀ x y, x ∈ B ∧ y ∈ B → (x + y) ∈ A) →
  A = { z | ∃ n, z = 2 * n } ∧ B = { z | ∃ n, z = 2 * n + 1 } :=
by
  sorry

end solution_l2358_235820


namespace arithmetic_sequence_fifth_term_l2358_235807

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15) 
  (h2 : a + 11 * d = 21) :
  a + 4 * d = 0 :=
by
  sorry

end arithmetic_sequence_fifth_term_l2358_235807


namespace PaulineDressCost_l2358_235851

-- Lets define the variables for each dress cost
variable (P Jean Ida Patty : ℝ)

-- Condition statements
def condition1 : Prop := Patty = Ida + 10
def condition2 : Prop := Ida = Jean + 30
def condition3 : Prop := Jean = P - 10
def condition4 : Prop := P + Jean + Ida + Patty = 160

-- The proof problem statement
theorem PaulineDressCost : 
  condition1 Patty Ida →
  condition2 Ida Jean →
  condition3 Jean P →
  condition4 P Jean Ida Patty →
  P = 30 := by
  sorry

end PaulineDressCost_l2358_235851


namespace find_k_l2358_235864

-- Define the arithmetic sequence and the sum of the first n terms
def a (n : ℕ) : ℤ := 2 * n + 2
def S (n : ℕ) : ℤ := n^2 + 3 * n

-- The main assertion
theorem find_k : ∃ (k : ℕ), k > 0 ∧ (S k - a (k + 5) = 44) ∧ k = 7 :=
by
  sorry

end find_k_l2358_235864


namespace remaining_integers_count_l2358_235898

def set_of_integers_from_1_to_100 : Finset ℕ := (Finset.range 100).map ⟨Nat.succ, Nat.succ_injective⟩

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

def T : Finset ℕ := set_of_integers_from_1_to_100
def M2 : Finset ℕ := multiples_of 2 T
def M3 : Finset ℕ := multiples_of 3 T
def M5 : Finset ℕ := multiples_of 5 T

def remaining_set : Finset ℕ := T \ (M2 ∪ M3 ∪ M5)

theorem remaining_integers_count : remaining_set.card = 26 := by
  sorry

end remaining_integers_count_l2358_235898


namespace polynomial_root_condition_l2358_235821

noncomputable def polynomial_q (q x : ℝ) : ℝ :=
  x^6 + 3 * q * x^4 + 3 * x^4 + 3 * q * x^2 + x^2 + 3 * q + 1

theorem polynomial_root_condition (q : ℝ) :
  (∃ x > 0, polynomial_q q x = 0) ↔ (q ≥ 3 / 2) :=
sorry

end polynomial_root_condition_l2358_235821


namespace relationship_among_abcd_l2358_235888

theorem relationship_among_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := 
by
  sorry

end relationship_among_abcd_l2358_235888


namespace simplify_expression_l2358_235837

theorem simplify_expression (a b c d : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) :
  -5 * a + 2017 * c * d - 5 * b = 2017 :=
by
  sorry

end simplify_expression_l2358_235837


namespace find_x_plus_y_l2358_235872

theorem find_x_plus_y (x y : ℝ)
  (h1 : (x - 1)^3 + 2015 * (x - 1) = -1)
  (h2 : (y - 1)^3 + 2015 * (y - 1) = 1)
  : x + y = 2 :=
sorry

end find_x_plus_y_l2358_235872


namespace father_age_difference_l2358_235804

variables (F S X : ℕ)
variable (h1 : F = 33)
variable (h2 : F = 3 * S + X)
variable (h3 : F + 3 = 2 * (S + 3) + 10)

theorem father_age_difference : X = 3 :=
by
  sorry

end father_age_difference_l2358_235804


namespace find_b_l2358_235809

theorem find_b (a b : ℝ) (f : ℝ → ℝ) (df : ℝ → ℝ) (x₀ : ℝ)
  (h₁ : ∀ x, f x = a * x + Real.log x)
  (h₂ : ∀ x, f x = 2 * x + b)
  (h₃ : x₀ = 1)
  (h₄ : f x₀ = a) :
  b = -1 := 
by
  sorry

end find_b_l2358_235809


namespace root_interval_l2358_235857

noncomputable def f (a b x : ℝ) : ℝ := 2 * a^x - b^x

theorem root_interval (a b : ℝ) (h₀ : 0 < a) (h₁ : b ≥ 2 * a) :
  ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 0 := 
sorry

end root_interval_l2358_235857


namespace train_length_is_correct_l2358_235811

noncomputable def convert_speed (speed_kmh : ℕ) : ℝ :=
  (speed_kmh : ℝ) * 5 / 18

noncomputable def relative_speed (train_speed_kmh man's_speed_kmh : ℕ) : ℝ :=
  convert_speed train_speed_kmh + convert_speed man's_speed_kmh

noncomputable def length_of_train (train_speed_kmh man's_speed_kmh : ℕ) (time_seconds : ℝ) : ℝ := 
  relative_speed train_speed_kmh man's_speed_kmh * time_seconds

theorem train_length_is_correct :
  length_of_train 60 6 29.997600191984645 = 550 :=
by
  sorry

end train_length_is_correct_l2358_235811


namespace apple_cost_l2358_235891

theorem apple_cost (A : ℝ) (h_discount : ∃ (n : ℕ), 15 = (5 * (5: ℝ) * A + 3 * 2 + 2 * 3 - n)) : A = 1 :=
by
  sorry

end apple_cost_l2358_235891


namespace product_of_smallest_primes_l2358_235831

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end product_of_smallest_primes_l2358_235831


namespace lizette_has_813_stamps_l2358_235882

def minervas_stamps : ℕ := 688
def additional_stamps : ℕ := 125
def lizettes_stamps : ℕ := minervas_stamps + additional_stamps

theorem lizette_has_813_stamps : lizettes_stamps = 813 := by
  sorry

end lizette_has_813_stamps_l2358_235882


namespace find_roots_l2358_235842

-- Given the conditions:
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given points (x, y)
def points := [(-5, 6), (-4, 0), (-2, -6), (0, -4), (2, 6)] 

-- Prove that the roots of the quadratic equation are -4 and 1
theorem find_roots (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : quadratic_function a b c (-5) = 6)
  (h₂ : quadratic_function a b c (-4) = 0)
  (h₃ : quadratic_function a b c (-2) = -6)
  (h₄ : quadratic_function a b c (0) = -4)
  (h₅ : quadratic_function a b c (2) = 6) :
  ∃ x₁ x₂ : ℝ, quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0 ∧ x₁ = -4 ∧ x₂ = 1 := 
sorry

end find_roots_l2358_235842


namespace solution_set_l2358_235871

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem solution_set:
  ∀ x : ℝ, x > -1 ∧ x < 1/3 → f (2*x + 1) < f x := 
by
  sorry

end solution_set_l2358_235871


namespace mason_ate_15_hotdogs_l2358_235849

structure EatingContest where
  hotdogWeight : ℕ
  burgerWeight : ℕ
  pieWeight : ℕ
  noahBurgers : ℕ
  jacobPiesLess : ℕ
  masonHotdogsWeight : ℕ

theorem mason_ate_15_hotdogs (data : EatingContest)
    (h1 : data.hotdogWeight = 2)
    (h2 : data.burgerWeight = 5)
    (h3 : data.pieWeight = 10)
    (h4 : data.noahBurgers = 8)
    (h5 : data.jacobPiesLess = 3)
    (h6 : data.masonHotdogsWeight = 30) :
    (data.masonHotdogsWeight / data.hotdogWeight) = 15 :=
by
  sorry

end mason_ate_15_hotdogs_l2358_235849


namespace binary_addition_l2358_235879

def bin_to_dec1 := 511  -- 111111111_2 in decimal
def bin_to_dec2 := 127  -- 1111111_2 in decimal

theorem binary_addition : bin_to_dec1 + bin_to_dec2 = 638 := by
  sorry

end binary_addition_l2358_235879


namespace work_completion_time_l2358_235845

-- Definitions for work rates
def work_rate_B : ℚ := 1 / 7
def work_rate_A : ℚ := 1 / 10

-- Statement to prove
theorem work_completion_time (W : ℚ) : 
  (1 / work_rate_A + 1 / work_rate_B) = 70 / 17 := 
by 
  sorry

end work_completion_time_l2358_235845


namespace smallest_y_exists_l2358_235886

theorem smallest_y_exists (M : ℤ) (y : ℕ) (h : 2520 * y = M ^ 3) : y = 3675 :=
by
  have h_factorization : 2520 = 2^3 * 3^2 * 5 * 7 := sorry
  sorry

end smallest_y_exists_l2358_235886
