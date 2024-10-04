import Mathlib

namespace combination_10_3_eq_120_l90_90382

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90382


namespace least_positive_four_digit_number_congruences_l90_90915

theorem least_positive_four_digit_number_congruences : 
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
    (3 * x ≡ 12 [MOD 18]) ∧
    (5 * x + 20 ≡ 35 [MOD 15]) ∧
    (-3 * x + 2 ≡ 2 * x [MOD 34]) ∧ 
    x = 1068 :=
by
  sorry

end least_positive_four_digit_number_congruences_l90_90915


namespace train_length_l90_90836

/-- 
  Given:
  - jogger_speed is the jogger's speed in km/hr (9 km/hr)
  - train_speed is the train's speed in km/hr (45 km/hr)
  - jogger_ahead is the jogger's initial lead in meters (240 m)
  - passing_time is the time in seconds for the train to pass the jogger (36 s)
  
  Prove that the length of the train is 120 meters.
-/
theorem train_length
  (jogger_speed : ℕ) -- in km/hr
  (train_speed : ℕ) -- in km/hr
  (jogger_ahead : ℕ) -- in meters
  (passing_time : ℕ) -- in seconds
  (h_jogger_speed : jogger_speed = 9)
  (h_train_speed : train_speed = 45)
  (h_jogger_ahead : jogger_ahead = 240)
  (h_passing_time : passing_time = 36)
  : ∃ length_of_train : ℕ, length_of_train = 120 :=
by
  sorry

end train_length_l90_90836


namespace inequality_solution_l90_90933

noncomputable def solve_inequality (m : ℝ) (m_lt_neg2 : m < -2) : Set ℝ :=
  if h : m = -3 then {x | 1 < x}
  else if h' : -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
  else {x | 1 < x ∧ x < m / (m + 3)}

theorem inequality_solution (m : ℝ) (m_lt_neg2 : m < -2) :
  (solve_inequality m m_lt_neg2) = 
    if m = -3 then {x | 1 < x}
    else if -3 < m then {x | x < m / (m + 3) ∨ 1 < x}
    else {x | 1 < x ∧ x < m / (m + 3)} :=
sorry

end inequality_solution_l90_90933


namespace find_x_value_l90_90033

theorem find_x_value (x : ℝ) (h1 : x^2 + x = 6) (h2 : x^2 - 2 = 1) : x = 2 := sorry

end find_x_value_l90_90033


namespace intersecting_circles_range_l90_90977

theorem intersecting_circles_range {k : ℝ} (a b : ℝ) :
  (-36 : ℝ) ≤ k ∧ k ≤ 104 →
  (∃ (x y : ℝ), (x^2 + y^2 - 4 - 12 * x + 6 * y) = 0 ∧ (x^2 + y^2 = k + 4 * x + 12 * y)) →
  b - a = (140 : ℝ) :=
by
  intro hk hab
  sorry

end intersecting_circles_range_l90_90977


namespace inverse_function_l90_90046

noncomputable def f (x : ℝ) (h : x ≥ 0) : ℝ :=
  log (x ^ 2 + 2) / log 2

noncomputable def f_inv (y : ℝ) (h : y ≥ 1) : ℝ :=
  real.sqrt (2^y - 2)

theorem inverse_function :
  ∀ (y : ℝ) (h : y ≥ 1), 
  (∃ (x : ℝ) (hx : x ≥ 0), f x hx = y) → f_inv y h = x :=
by
  sorry

end inverse_function_l90_90046


namespace sin_difference_identity_l90_90876

theorem sin_difference_identity 
  (α β : ℝ)
  (h1 : sin α - cos β = 3 / 4)
  (h2 : cos α + sin β = -2 / 5) : 
  sin (α - β) = 511 / 800 := 
sorry

end sin_difference_identity_l90_90876


namespace polynomial_roots_coefficients_sum_nonnegative_l90_90620

theorem polynomial_roots_coefficients_sum_nonnegative
  (n : ℕ) (a : ℕ → ℝ)
  (h_n : n ≥ 2)
  (h_roots : ∃ x : ℕ → ℝ, (∀ i : ℕ, i < n → 0 ≤ x i ∧ x i ≤ 1) ∧ 
             (fun y => x.foldr (fun i acc => acc * (y - x i)) 1) = 
             (λ y, y^n + (a 1) * y^(n-1) + (a 2) * y^(n-2) + ... + (a (n-1)) * y + (a n)))
  : (a 2) + (a 3) + ... + (a n) ≥ 0 :=
sorry

end polynomial_roots_coefficients_sum_nonnegative_l90_90620


namespace binomial_10_3_l90_90505

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90505


namespace prod_add_one_ge_two_pow_n_l90_90926

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem prod_add_one_ge_two_pow_n
  (h1 : (∀ i, a i > 0))
  (h2 : (∏ i, a i = 1)) :
  (∏ i, (1 + a i)) ≥ (2 ^ n) :=
by sorry

end prod_add_one_ge_two_pow_n_l90_90926


namespace number_of_tangents_with_positive_integer_slope_l90_90862

theorem number_of_tangents_with_positive_integer_slope :
  let f := λ x : ℝ, -x^3 + 3 * x - 1
  let f' := λ x : ℝ, -3 * x^2 + 3
  ∃ k : ℕ, k > 0 ∧ ∃ x₁ x₂ : ℝ, f'(x₁) = k ∧ f'(x₂) = k → (x₁ ≠ x₂ ∨ x₁ = x₂ ∧ x₁ = 0) ∧ 
  (∀ m : ℝ, m ∉ (Set.image f' {x : ℝ | f'(x) > 0}) ∧ m ≠ 1 ∧ m ≠ 2 ∧ m ≠ 3 →
  Cardinal.mk {x : ℝ | f'(x) = m} = 0) ∧ 
  Cardinal.mk {k | k = 1 ∨ k = 2 ∨ k = 3} = 5 := sorry

end number_of_tangents_with_positive_integer_slope_l90_90862


namespace correct_calculation_l90_90229

theorem correct_calculation (m n : ℤ) :
  (m^2 * m^3 ≠ m^6) ∧
  (- (m - n) = -m + n) ∧
  (m * (m + n) ≠ m^2 + n) ∧
  ((m + n)^2 ≠ m^2 + n^2) :=
by sorry

end correct_calculation_l90_90229


namespace fraction_from_tips_l90_90910

-- Define the waiter's salary and the conditions given in the problem
variables (S : ℕ) -- S is natural assuming salary is a non-negative integer
def tips := (4/5 : ℚ) * S
def bonus := 2 * (1/10 : ℚ) * S
def total_income := S + tips S + bonus S

-- The theorem to be proven
theorem fraction_from_tips (S : ℕ) :
  (tips S / total_income S) = (2/5 : ℚ) :=
sorry

end fraction_from_tips_l90_90910


namespace angle_C_of_triangle_l90_90000

theorem angle_C_of_triangle (a b c : ℝ) (h : (a + b - c) * (a + b + c) = a * b):
  ∠ABC = 2 * Real.pi / 3 := 
sorry

end angle_C_of_triangle_l90_90000


namespace problem_statement_problem_statement_l90_90563

variable (Line : Type) (Plane : Type)

-- Definitions for perpendicular and parallel relations
variable (perp : Line → Plane → Prop) (parallel : Line → Plane → Prop) (perpL : Line → Line → Prop)

-- Define two lines l and m, and two planes α and β
variable (l m : Line) (α β : Plane)

-- Problem statement in Lean: Prove the required conditions
theorem problem_statement (h1 : perp l α) (h2 : parallel m α) : perpL l m :=
sorry

theorem problem_statement' (h1 : perp l α) (h3 : parallel l β) : perp α β :=
sorry

end problem_statement_problem_statement_l90_90563


namespace binomial_10_3_eq_120_l90_90449

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90449


namespace math_equivalent_proof_problem_l90_90667

noncomputable def A (n : ℕ) : set ℤ := 
  { z | ∃ (i : ℕ) (H : i ∈ finset.range n), z = 2 * nat.next_prime(i * 2 + 1)}

theorem math_equivalent_proof_problem (n k : ℕ) (hk : 1 ≤ k ∧ k ≤ n):
  ∃ (b : ℤ) (A : set ℤ),
    (∀ (S : finset ℤ), S ⊆ A → S.card = k - 1 → ¬ b ∣ S.prod id) ∧
    (∀ (T : finset ℤ), T ⊆ A → T.card = k → b ∣ T.prod id) ∧
    (∀ {a a' : ℤ}, a ≠ a' → a ∈ A → a' ∈ A → ¬ a ∣ a') :=
  ∃ (b : ℤ) (A : set ℤ), 
    b = 2 ^ k ∧ 
    A = {2 * p | p ∈ (finset.range n).image (λ i, nat.next_prime(i * 2 + 1))} ∧ 
    (∀ S ⊆ A, S.card = k - 1 → ¬ b ∣ S.prod id) ∧
    (∀ T ⊆ A, T.card = k → b ∣ T.prod id) ∧
    (∀ {a a' : ℤ}, a ≠ a' → a ∈ A → a' ∈ A → ¬ a ∣ a') :=
begin
  let b := 2 ^ k,
  let A := {2 * p | p ∈ (finset.range n).image (λ i, nat.next_prime(i * 2 + 1))},
  use [b, A],
  split,
  -- prove condition (i)
  sorry,
  split,
  -- prove condition (ii)
  sorry,
  -- prove condition (iii)
  sorry,
end

end math_equivalent_proof_problem_l90_90667


namespace angle_ABC_is_60_degrees_l90_90095

theorem angle_ABC_is_60_degrees
  (A B C : Type)
  [metric_space A] [metric_space B] [metric_space C]
  {a b c : ℝ}
  (h₁ : ∠C = 3 * ∠A)
  (h₂ : dist A B = 2 * dist B C) :
  ∠B = 60 :=
by
  sorry

end angle_ABC_is_60_degrees_l90_90095


namespace parsec_to_km_au_to_km_l90_90232

/-- Definitions for units and their relations. -/
def light_year_km : ℝ := 9.5 * 10^12
def parsec_in_light_years : ℝ := 3.2616
def parsec_in_au : ℝ := 206265
def au_in_parsec : ℝ := 3.2616 / 206265

/-- The theorem proving the conversion of parsec to kilometers. -/
theorem parsec_to_km : (parsec_in_light_years * light_year_km) = 3.099 * 10^13 :=
by
  sorry

/-- The theorem proving the conversion of astronomical units to kilometers. -/
theorem au_to_km : (au_in_parsec * light_year_km) = 1.502 * 10^8 :=
by
  sorry

end parsec_to_km_au_to_km_l90_90232


namespace farmer_cages_l90_90833

theorem farmer_cages (c : ℕ) (h1 : 164 + 6 = 170) (h2 : ∃ r : ℕ, c * r = 170) (h3 : ∃ r : ℕ, c * r > 164) :
  c = 10 :=
by
  sorry

end farmer_cages_l90_90833


namespace right_triangle_construction_condition_l90_90897

theorem right_triangle_construction_condition
  (b s : ℝ) 
  (h_b_pos : b > 0)
  (h_s_pos : s > 0)
  (h_perimeter : ∃ (AC BC AB : ℝ), AC = b ∧ AC + BC + AB = 2 * s ∧ (AC^2 + BC^2 = AB^2)) :
  b < s := 
sorry

end right_triangle_construction_condition_l90_90897


namespace combination_10_3_eq_120_l90_90484

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90484


namespace investment_period_l90_90526

theorem investment_period :
  ∀ (e_investment b_investment : ℝ) (e_roi_rate b_roi_rate : ℝ) (diff : ℝ),
  e_investment = 300 → b_investment = 500 →
  e_roi_rate = 0.15 → b_roi_rate = 0.10 →
  diff = 10 →
  let e_annual_roi := e_roi_rate * e_investment in
  let b_annual_roi := b_roi_rate * b_investment in
  let annual_diff := b_annual_roi - e_annual_roi in
  2 * annual_diff = diff → true :=
by
  intros e_investment b_investment e_roi_rate b_roi_rate diff he hb he_rate hb_rate hdiff e_annual_roi b_annual_roi annual_diff period_eq
  rw [he, hb, he_rate, hb_rate] at e_annual_roi b_annual_roi annual_diff
  sorry

end investment_period_l90_90526


namespace binomial_10_3_l90_90335

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90335


namespace congruent_orthocentric_triangles_l90_90635

variables {A B C H_A H_B H_C H H_1 H_2 H_3 M_1 M_2 M_3 : Type} 

-- Hypotheses
variables [triangle A B C]
variables [altitude A H_A]
variables [altitude B H_B]
variables [altitude C H_C]
variables [orthocenter A B C H]
variables [orthocenter A H_B H_C H_1]
variables [orthocenter B H_A H_C H_2]
variables [orthocenter C H_A H_B H_3]
variables [midpoint H_B H_C M_1]
variables [midpoint H_C H_A M_2]
variables [midpoint H_A H_B M_3]

theorem congruent_orthocentric_triangles :
  triangle_congruent (triangle.mk H_1 H_2 H_3) (triangle.mk H_A H_B H_C) :=
sorry

end congruent_orthocentric_triangles_l90_90635


namespace dolphins_points_l90_90990

variable (S D : ℕ)

theorem dolphins_points :
  (S + D = 36) ∧ (S = D + 12) → D = 12 :=
by
  sorry

end dolphins_points_l90_90990


namespace my_op_identity_l90_90057

def my_op (a b : ℕ) : ℕ := a + b + a * b

theorem my_op_identity (a : ℕ) : my_op (my_op a 1) 2 = 6 * a + 5 :=
by
  sorry

end my_op_identity_l90_90057


namespace preference_is_related_to_gender_expectation_of_X_correct_l90_90160

noncomputable theory

-- Given conditions
def male_students : ℕ := 100
def female_students : ℕ := 100
def group_a_total : ℕ := 96
def group_a_males : ℕ := 36
def alpha : ℝ := 0.001
def chi_square_critical : ℝ := 10.828

-- Values derived from basic arithmetic operations on given conditions
def group_a_females : ℕ := group_a_total - group_a_males
def group_b_males : ℕ := male_students - group_a_males
def group_b_females : ℕ := female_students - group_a_females
def group_b_total : ℕ := male_students + female_students - group_a_total

-- Chi-square formula components
def ad_bc : ℤ := (group_a_males * group_b_females) - (group_a_females * group_b_males)

def chi_square_value : ℝ := 
  let n : ℝ := (male_students + female_students).toReal
  let a_b : ℝ := (group_a_males + group_a_females).toReal
  let c_d : ℝ := (group_b_males + group_b_females).toReal
  let a_c : ℝ := (group_a_males + group_b_males).toReal
  let b_d : ℝ := (group_a_females + group_b_females).toReal
  (n * (ad_bc)^2) / (a_b * c_d * a_c * b_d).toReal

-- Theorem (statement, no proof)
theorem preference_is_related_to_gender : chi_square_value > chi_square_critical := sorry

-- For Part (2): Distribution table and expectation
def X_distribution : ℕ → ℝ
| 0 => (28/115)
| 1 => (54/115)
| 2 => (144/575)
| 3 => (21/575)
| _ => 0 -- Defined for completeness, though not necessary

def E_X : ℝ := 0 * (28/115) + 1 * (54/115) + 2 * (144/575) + 3 * (21/575)

-- Theorem (statement, no proof)
theorem expectation_of_X_correct : E_X = (621/575) := sorry

end preference_is_related_to_gender_expectation_of_X_correct_l90_90160


namespace product_of_x_and_y_l90_90287

theorem product_of_x_and_y (x y a b : ℝ)
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := 
  sorry

end product_of_x_and_y_l90_90287


namespace combination_10_3_eq_120_l90_90490

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90490


namespace midpoint_equidistant_from_chords_l90_90066

noncomputable def circle_center : Type := sorry
noncomputable def parallel_chords (A B C D P : Type) : Prop := sorry
noncomputable def circles_intersect (A B C D P : Type) : Prop := sorry
noncomputable def midpoint (O P : Type) : Type := sorry
noncomputable def equidistant (M AB CD : Type) : Prop := sorry

theorem midpoint_equidistant_from_chords
  (O A B C D P : Type)
  (h_center : O = circle_center)
  (h_parallel_chords : parallel_chords A B C D P)
  (h_circles_intersect : circles_intersect A B C D P) :
  equidistant (midpoint O P) AB CD := sorry

end midpoint_equidistant_from_chords_l90_90066


namespace binomial_10_3_eq_120_l90_90419

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90419


namespace max_removal_l90_90824

-- Definitions of quantities in the problem
def yellow_marbles : ℕ := 8
def red_marbles : ℕ := 7
def black_marbles : ℕ := 5
def total_marbles : ℕ := yellow_marbles + red_marbles + black_marbles

-- Definition of the condition on remaining marbles
def valid_remaining (remaining : ℕ) : Prop :=
∃ (yellow_left red_left black_left : ℕ),
  yellow_left + red_left + black_left = remaining ∧
  (yellow_left ≥ 4 ∨ red_left ≥ 4 ∨ black_left ≥ 4) ∧
  (yellow_left ≥ 3 ∨ red_left ≥ 3 ∨ black_left ≥ 3)

-- Statement of the main theorem
theorem max_removal (N : ℕ) (N ≤ 7) :
  let remaining := total_marbles - N in valid_remaining remaining := 
begin
  -- Content of the proof is omitted and replaced by sorry
  sorry
end

end max_removal_l90_90824


namespace binom_10_3_eq_120_l90_90316

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90316


namespace inequality_proof_l90_90001

theorem inequality_proof (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 :=
by
  sorry

end inequality_proof_l90_90001


namespace estimated_probability_l90_90846

-- Definition of hit and miss
def is_hit (n : ℕ) : Bool :=
  n >= 4 ∧ n <= 9

def is_miss (n : ℕ) : Bool :=
  n >= 0 ∧ n <= 3

-- A group represents 4 shots
def group_hits (g : List ℕ) : Bool :=
  g.countp is_hit >= 3

-- The given 20 groups of random numbers each representing 4 shots
def groups : List (List ℕ) :=
  [[7, 5, 2, 7], [0, 2, 9, 3], [7, 1, 4, 0], [9, 8, 5, 7],
   [0, 3, 4, 7], [4, 3, 7, 3], [8, 6, 3, 6], [6, 9, 4, 7],
   [1, 4, 1, 7], [4, 6, 9, 8], [0, 3, 7, 1], [6, 2, 3, 3],
   [2, 6, 1, 6], [8, 0, 4, 5], [6, 0, 1, 1], [3, 6, 6, 1],
   [9, 5, 9, 7], [7, 4, 2, 4], [7, 6, 1, 0], [4, 2, 8, 1]]

-- We need to prove that the probability of hitting the target at least 3 times in 4 shots is 0.4
theorem estimated_probability : 
  (groups.filter group_hits).length = 8 → 
  groups.length = 20 →
  (groups.filter group_hits).length.toFloat / groups.length.toFloat = 0.4 := 
by
  intros hits_total groups_total
  sorry

end estimated_probability_l90_90846


namespace arrangement_count_is_20_l90_90145

noncomputable def numArrangements : nat :=
  let athletes : finset (fin 5) := finset.univ
  let derangements_count : nat := derangements.fintype_card (fin 3)
  (nat.choose 5 2) * derangements_count

theorem arrangement_count_is_20 : numArrangements = 20 := by
  sorry

end arrangement_count_is_20_l90_90145


namespace magnitude_of_one_minus_i_l90_90586

-- Definitions based on conditions
def i : ℂ := complex.I
def z : ℂ := 1 - i

-- Theorem stating the desired equality
theorem magnitude_of_one_minus_i : complex.abs z = real.sqrt 2 :=
  sorry

end magnitude_of_one_minus_i_l90_90586


namespace jack_total_damage_costs_l90_90655

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end jack_total_damage_costs_l90_90655


namespace bryden_receives_amount_l90_90829

theorem bryden_receives_amount (face_value : ℝ) (percentage_offer : ℝ) (num_quarters : ℕ) :
  face_value = 0.25 ∧ percentage_offer = 2500 ∧ num_quarters = 5 →
  let multiplier := percentage_offer / 100
  let total_face_value := num_quarters * face_value
  let total_received := multiplier * total_face_value
  total_received = 31.25 :=
begin
  intros h,
  let multiplier := percentage_offer / 100,
  let total_face_value := num_quarters * face_value,
  let total_received := multiplier * total_face_value,
  cases h with h1 h2,
  cases h2 with h3 h4,
  rw [h1, h3, h4],
  simp [multiplier, total_face_value, total_received],
  exact dec_trivial,
end

end bryden_receives_amount_l90_90829


namespace combination_10_3_eq_120_l90_90478

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90478


namespace inequality_solution_set_l90_90194

theorem inequality_solution_set :
  (∀ x : ℝ, (3 * x - 2 < 2 * (x + 1) ∧ (x - 1) / 2 > 1) ↔ (3 < x ∧ x < 4)) :=
by
  sorry

end inequality_solution_set_l90_90194


namespace minimum_distance_l90_90692

theorem minimum_distance (P_on_line : ∃ P : ℝ × ℝ, P.snd = 2)
                        (Q_on_circle : ∃ Q : ℝ × ℝ, (Q.fst - 1)^2 + Q.snd^2 = 1) :
  ∃ |PQ|, |PQ| = 1 :=
by
  sorry

end minimum_distance_l90_90692


namespace should_agree_to_buy_discount_card_l90_90857

-- Define the conditions
def discount_card_cost := 100
def discount_percentage := 0.03
def cost_of_cakes := 4 * 500
def cost_of_fruits := 1600
def total_cost_without_discount_card := cost_of_cakes + cost_of_fruits
def discount_amount := total_cost_without_discount_card * discount_percentage
def cost_after_discount := total_cost_without_discount_card - discount_amount
def effective_total_cost_with_discount_card := cost_after_discount + discount_card_cost

-- Define the objective statement to prove
theorem should_agree_to_buy_discount_card : effective_total_cost_with_discount_card < total_cost_without_discount_card := by
  sorry

end should_agree_to_buy_discount_card_l90_90857


namespace find_other_tax_l90_90142

/-- Jill's expenditure breakdown and total tax conditions. -/
def JillExpenditure 
  (total : ℝ)
  (clothingPercent : ℝ)
  (foodPercent : ℝ)
  (otherPercent : ℝ)
  (clothingTaxPercent : ℝ)
  (foodTaxPercent : ℝ)
  (otherTaxPercent : ℝ)
  (totalTaxPercent : ℝ) :=
  (clothingPercent + foodPercent + otherPercent = 100) ∧
  (clothingTaxPercent = 4) ∧
  (foodTaxPercent = 0) ∧
  (totalTaxPercent = 5.2) ∧
  (total > 0)

/-- The goal is to find the tax percentage on other items which Jill paid, given the constraints. -/
theorem find_other_tax
  {total clothingAmt foodAmt otherAmt clothingTax foodTax otherTaxPercent totalTax : ℝ}
  (h_exp : JillExpenditure total 50 10 40 clothingTax foodTax otherTaxPercent totalTax) :
  otherTaxPercent = 8 :=
by
  sorry

end find_other_tax_l90_90142


namespace solve_system_l90_90155

theorem solve_system (x y : ℝ) (h1 : x^2 + y * sqrt (x * y) = 105) (h2 : y^2 + x * sqrt (x * y) = 70) :
  x = 9 ∧ y = 4 :=
by
  sorry

end solve_system_l90_90155


namespace area_of_ABCD_is_160_sqrt_6_l90_90717

noncomputable def area_of_rectangle 
  {D A F E: ℕ} 
  (DA: D = 20) 
  (FD: F = 12) 
  (AE: E = 12) 
  (ABCD_inscribed_in_semicircle: Prop) : Real :=
  DA * (8 * Real.sqrt 6)

theorem area_of_ABCD_is_160_sqrt_6 
  (D A F E: ℕ) 
  (DA: D = 20) 
  (FD: F = 12) 
  (AE: E = 12) 
  (ABCD_inscribed_in_semicircle: Prop) : 
  area_of_rectangle DA FD AE ABCD_inscribed_in_semicircle = 160 * Real.sqrt 6 := 
  sorry

end area_of_ABCD_is_160_sqrt_6_l90_90717


namespace motorists_exceeding_speed_limit_l90_90139

theorem motorists_exceeding_speed_limit (total_motorists : ℕ)
  (h1 : 0.10 * total_motorists = speeding_tickets)
  (h2 : 0.40 * exceeding_speed_limit = no_tickets)
  (h3 : speeding_tickets = 0.60 * exceeding_speed_limit)
  : (exceeding_speed_limit : ℝ) / total_motorists = 0.1667 :=
by
  sorry

end motorists_exceeding_speed_limit_l90_90139


namespace necessary_and_sufficient_condition_l90_90120

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log2 (x + Real.sqrt (x^2 + 1))

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (a + b ≥ 0) ↔ (f a + f b ≥ 0) :=
sorry

end necessary_and_sufficient_condition_l90_90120


namespace linear_function_has_inverse_graph_H_has_inverse_graph_I_has_inverse_graphs_with_inverses_l90_90614

def graph_F (x : ℝ) := x^3 / 27 + x^2 / 18 - x / 3 + 3
def graph_G (x : ℝ) := x^2 / 4 - 4
def graph_H (x : ℝ) := x * 2 / 3 + 1
def graph_I (x : ℝ) := -x / 3 + 1

theorem linear_function_has_inverse (f : ℝ → ℝ) (h : ∀ x y, f x = f y → x = y) : ∃ g, ∀ x, g (f x) = x :=
sorry

theorem graph_H_has_inverse : ∃ g, ∀ x, g (graph_H x) = x :=
linear_function_has_inverse graph_H (λ x y hxy, sorry)

theorem graph_I_has_inverse : ∃ g, ∀ x, g (graph_I x) = x :=
linear_function_has_inverse graph_I (λ x y hxy, sorry)

theorem graphs_with_inverses : ∃ gH gI, (∀ x, gH (graph_H x) = x) ∧ (∀ x, gI (graph_I x) = x) :=
⟨(λ x, (x - 1) * 3 / 2), (λ x, 3 * (1 - x)), by
  exact ⟨(λ x, by sorry), (λ x, by sorry)⟩⟩

end linear_function_has_inverse_graph_H_has_inverse_graph_I_has_inverse_graphs_with_inverses_l90_90614


namespace circle_relationship_l90_90707

noncomputable def hyperbola (a b : ℝ) : Set (ℝ × ℝ) := {p | p.1 ^ 2 / a ^ 2 - p.2 ^ 2 / b ^ 2 = 1}

def on_hyperbola (a b : ℝ) (P : ℝ × ℝ) : Prop := P ∈ hyperbola a b

def is_focus (a b : ℝ) (F : ℝ × ℝ) : Prop := F = (a * cosh 0, b * sinh 0) ∨ F = (-a * cosh 0, -b * sinh 0)

def circle_with_diameter (P F : ℝ × ℝ) : Set (ℝ × ℝ) := 
  let M := ((P.1 + F.1) / 2, (P.2 + F.2) / 2) in
  let r := dist P F / 2 in 
    {C | dist C M = r}

def circle_with_radius_a (a : ℝ) : Set (ℝ × ℝ) := 
  {C | dist C (0, 0) = a}

theorem circle_relationship (a b : ℝ) (P F : ℝ × ℝ) 
  (hP_on_hyperbola : on_hyperbola a b P)
  (hF_is_focus : is_focus a b F) :
  (∃ M, M ∈ circle_with_diameter P F ∧ distance M (0, 0) = a) → 
  true :=
sorry

end circle_relationship_l90_90707


namespace factorial_divisible_by_power_of_two_iff_l90_90724

theorem factorial_divisible_by_power_of_two_iff (n : ℕ) :
  (nat.factorial n) % (2^(n-1)) = 0 ↔ ∃ k : ℕ, n = 2^k := 
by
  sorry

end factorial_divisible_by_power_of_two_iff_l90_90724


namespace closed_polygonal_line_even_segments_l90_90909

theorem closed_polygonal_line_even_segments 
  (n : ℕ)
  (vertices : Fin n → ℤ × ℤ)
  (h_closed : (∑ i, (vertices i).1 = 0) ∧ (∑ i, (vertices i).2 = 0))
  (h_length : ∀ i, (vertices (⟨(i + 1) % n, sorry⟩ : Fin n)).fst - (vertices i).fst)^2 
                  + ((vertices (⟨(i + 1) % n, sorry⟩ : Fin n)).snd - (vertices i).snd)^2 = c^2) :
  Even n := sorry

end closed_polygonal_line_even_segments_l90_90909


namespace jack_total_cost_l90_90654

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end jack_total_cost_l90_90654


namespace product_real_imag_parts_l90_90882

theorem product_real_imag_parts :
  let complex_number : ℂ := (2 + 3 * complex.I) / (1 + complex.I)
  real_part := complex_number.re
  imag_part := complex_number.im
  real_part * imag_part = 5 / 4 :=
by
  sorry

end product_real_imag_parts_l90_90882


namespace binomial_10_3_eq_120_l90_90453

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90453


namespace travel_less_than_1500_km_l90_90991

theorem travel_less_than_1500_km (cities : Type) (roads : cities → cities → Prop)
  (length : cities → cities → ℝ)
  (h1 : ∀ a b, roads a b → length a b < 500)
  (h2 : ∀ a b, a ≠ b → ∃ p, (∀ i j ∈ p, roads i j ∧ length i j < 500) ∧ length_of_path p < 500)
  (h3 : ∀ a b, ∀ r, (roads a b → False) → ∃ p, (∀ i j ∈ p, i ≠ a ∧ j ≠ b ∧ roads i j ∧ length i j < 500) ∧ length_of_path p < 1500) :
  ∀ a b, a ≠ b → ∃ p, (∀ i j ∈ p, roads i j ∧ length i j < 500) ∧ length_of_path p < 1500 :=
sorry

end travel_less_than_1500_km_l90_90991


namespace minimum_value_of_16b_over_ac_l90_90988

noncomputable def minimum_16b_over_ac (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if (0 < B) ∧ (B < Real.pi / 2) ∧
     (Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1) ∧
     ((Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3)) then
    16 * b / (a * c)
  else 0

theorem minimum_value_of_16b_over_ac (a b c : ℝ) (A B C : ℝ)
  (h1 : 0 < B)
  (h2 : B < Real.pi / 2)
  (h3 : Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1)
  (h4 : Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3) :
  minimum_16b_over_ac a b c A B C = 16 * (2 - Real.sqrt 2) / 3 := 
sorry

end minimum_value_of_16b_over_ac_l90_90988


namespace should_agree_to_buy_discount_card_l90_90859

noncomputable def total_cost_without_discount_card (cakes_cost fruits_cost : ℕ) : ℕ :=
  cakes_cost + fruits_cost

noncomputable def total_cost_with_discount_card (cakes_cost fruits_cost discount_card_cost : ℕ) : ℕ :=
  let total_cost := cakes_cost + fruits_cost
  let discount := total_cost * 3 / 100
  (total_cost - discount) + discount_card_cost

theorem should_agree_to_buy_discount_card : 
  let cakes_cost := 4 * 500
  let fruits_cost := 1600
  let discount_card_cost := 100
  total_cost_with_discount_card cakes_cost fruits_cost discount_card_cost < total_cost_without_discount_card cakes_cost fruits_cost :=
by
  sorry

end should_agree_to_buy_discount_card_l90_90859


namespace positive_rational_representation_l90_90886

theorem positive_rational_representation (q : ℚ) (h_pos_q : 0 < q) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ q = (a^2021 + b^2023) / (c^2022 + d^2024) :=
by
  sorry

end positive_rational_representation_l90_90886


namespace arithmetic_sequence_iff_sum_condition_l90_90116

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = a (n - 1) - a (n - 2)

/-- Definition of the sum of the first n terms of a sequence -/
noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a (i + 1)

/-- Main theorem stating the equivalence of Scenario A and Scenario B -/
theorem arithmetic_sequence_iff_sum_condition (a : ℕ → ℤ) :
  (∀ n : ℕ, n ∈ ℕ ∧ n ≥ 1 → is_arithmetic_sequence a) ↔ 
  (∀ n : ℕ, n ∈ ℕ ∧ n ≥ 1 → 2 * (sum_of_first_n_terms a n) = (a 1 + a n) * n) :=
by
  sorry

end arithmetic_sequence_iff_sum_condition_l90_90116


namespace diamonds_in_F10_l90_90752

def T (n : ℕ) : ℕ := (n * (n + 1)) / 2

def F (n : ℕ) : ℕ
| 1       := 1
| (n + 2) := F (n + 1) + 4 * T (n + 2)

theorem diamonds_in_F10 : F 10 = 877 := by
  sorry

end diamonds_in_F10_l90_90752


namespace binomial_coefficient_10_3_l90_90369

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90369


namespace arithmetic_sequence_solution_l90_90997

noncomputable def arithmetic_sequence_first_term (a₁ d : ℝ) := a₁
noncomputable def arithmetic_sequence_common_difference (a₁ d : ℝ) := d
noncomputable def sum_of_first_n_terms (a₁ d n : ℕ) := (n / 2 : ℝ) * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_solution (a₁ d : ℝ) (n : ℕ) :
    (a₁ + (a₁ + 2 * d) = 8) ∧ ((a₁ + 3 * d) ^ 2 = (a₁ + d) * (a₁ + 8 * d)) →
    ((d = 0 ∧ a₁ = 4 ∧ sum_of_first_n_terms a₁ d n = 4 * n) ∨
     (d = 16 / 9 ∧ a₁ = 20 / 9 ∧ sum_of_first_n_terms a₁ d n = (8 * n ^ 2 + 12 * n) / 9)) :=
by
  intros h
  sorry

end arithmetic_sequence_solution_l90_90997


namespace mixed_number_sum_l90_90306

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end mixed_number_sum_l90_90306


namespace length_AM_l90_90641

-- Define the setup of the problem
variables (A B C M : ℝ) (AB AC BC AM BM : ℝ)

-- State the conditions
def conditions (A B C M : ℝ) (AB AC BC : ℝ) :=
  right_angle A ∧
  AB = 60 ∧
  AC = 160 ∧
  M = (B + C) / 2 ∧
  BM = BC / 2

-- State the theorem to prove
theorem length_AM (A B C M : ℝ) (AB AC BC AM BM : ℝ) (h : conditions A B C M AB AC BC) : 
  AM = 56.3 :=
  sorry

end length_AM_l90_90641


namespace exists_triangle_with_interior_bisector_exists_triangle_with_exterior_bisector_l90_90898

-- Define the necessary elements to represent the geometrical properties

noncomputable def interior_angle_bisector_condition 
  (c f g : ℝ) : Prop := 
  (g^2 - c^2) / (2 * g) < f ∧ f ≤ √(g^2 - c^2) / 2 ∧ g > c

noncomputable def exterior_angle_bisector_condition 
  (h c : ℝ) : Prop := 
  h < c

-- The main theorem statement for part (a), validating the interval conditions
theorem exists_triangle_with_interior_bisector 
  (c f g : ℝ) (h₁ : g = a + b) (h₂ : AB = c) (h₃ : CD = f)
  (h₄ : CA + CB = g) : interior_angle_bisector_condition c f g :=
sorry

-- The main theorem statement for part (b), ensuring the external angle bisector properties
theorem exists_triangle_with_exterior_bisector 
  (h c : ℝ) (h₁ : CA - CB = h) (h₂ : CD_k = f_k)
  : exterior_angle_bisector_condition h c :=
sorry

end exists_triangle_with_interior_bisector_exists_triangle_with_exterior_bisector_l90_90898


namespace binomial_10_3_l90_90326

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90326


namespace arcSin_neg_half_eq_neg_pi_over_six_l90_90310

noncomputable def arcSin_neg_half : Real :=
  Real.arcsin (-1 / 2)

theorem arcSin_neg_half_eq_neg_pi_over_six :
  arcSin_neg_half = -Real.pi / 6 :=
by
  -- Satisfy condition
  have h1 : Real.sin (-Real.pi / 6) = -1 / 2 := by
    rw [Real.sin_neg, Real.sin, Real.sin_pi_div_six]
    norm_num
  -- Check the interval condition
  have h2 : -Real.pi / 6 ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) :=
    by norm_num
  -- Conclude the proof
  sorry

end arcSin_neg_half_eq_neg_pi_over_six_l90_90310


namespace dogs_remaining_end_month_l90_90289

theorem dogs_remaining_end_month :
  let initial_dogs := 200
  let dogs_arrive_w1 := 30
  let dogs_adopt_w1 := 40
  let dogs_arrive_w2 := 40
  let dogs_adopt_w2 := 50
  let dogs_arrive_w3 := 30
  let dogs_adopt_w3 := 30
  let dogs_adopt_w4 := 70
  let dogs_return_w4 := 20
  initial_dogs + (dogs_arrive_w1 - dogs_adopt_w1) + 
  (dogs_arrive_w2 - dogs_adopt_w2) +
  (dogs_arrive_w3 - dogs_adopt_w3) + 
  (-dogs_adopt_w4 - dogs_return_w4) = 90 := by
  sorry

end dogs_remaining_end_month_l90_90289


namespace quadrilateral_is_parallelogram_l90_90648

section TriangleParallelogram

variables {A B C F S1 S2 P1 P2 : Type*}
variables [AffineSpace ℝ (fin 3)]
variables {Point : Type*} [AffineSpace.Point ℝ Point]

-- Definitions of points and properties
-- A, B, C are points forming a triangle ABC
-- F is the midpoint of AB
-- S1 is the centroid of AFC
-- S2 is the centroid of BFC
-- P1 is the intersection of AS1 with BC
-- P2 is the intersection of BS2 with AC

def is_midpoint (F : Point) (A B : Point) := 
  ∃ (F' : Point), affine_combo ℝ (list.iota 2) (list.repeat (1/2) 2) = A + B ∧ F = F'
  
def is_centroid (S : Point) (X Y Z : Point) := 
  ∃ (S' : Point), affine_combo ℝ (list.iota 3) (list.repeat (1/3) 3) = X + Y + Z ∧ S = S'

def intersects (P : Point) (L1 L2 : Line) :=
  ∃ (P' : Point), P ∈ L1 ∧ P ∈ L2 ∧ P = P'

-- Parallelogram proof statement
theorem quadrilateral_is_parallelogram
  (A B C F S1 S2 P1 P2 : Point)
  (h_midpoint : is_midpoint F A B)
  (h_centroid1 : is_centroid S1 A F C)
  (h_centroid2 : is_centroid S2 B F C)
  (h_intersect1 : intersects P1 (line_through A S1) (line_through B C))
  (h_intersect2 : intersects P2 (line_through B S2) (line_through A C)) :
  parallelogram S1 S2 P1 P2 :=
sorry

end TriangleParallelogram

end quadrilateral_is_parallelogram_l90_90648


namespace minimize_distances_l90_90651

-- Define the context: a convex quadrilateral and distances.
variables {A B C D M : Type} [MetricSpace M]

def is_convex_quadrilateral (A B C D M : M) : Prop := sorry  -- Define convex quadrilateral property

def is_diagonal_intersection (A B C D M : M) : Prop := sorry -- Define the property of M being the intersection of AC and BD

def sum_distances (point vertices : M) : ℝ := sorry  -- Define sum of distances from a point to vertices

theorem minimize_distances (A B C D M : M) 
  (h_convex : is_convex_quadrilateral A B C D M)
  (h_intersect : is_diagonal_intersection A B C D M) :
  ∀ N : M, sum_distances N {A, B, C, D} ≥ sum_distances M {A, B, C, D} :=
sorry

end minimize_distances_l90_90651


namespace combination_10_3_eq_120_l90_90381

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90381


namespace min_a_l90_90736

-- Definitions and conditions used in the problem
def eqn (a x : ℝ) : Prop :=
  2 * sin (π - (π * x^2) / 12) * cos ((π / 6) * sqrt (9 - x^2)) + 1 =
  a + 2 * sin ((π / 6) * sqrt (9 - x^2)) * cos ((π * x^2) / 12)

-- Statement to prove the minimum value of a
theorem min_a : ∃ x : ℝ, eqn 2 x := sorry

end min_a_l90_90736


namespace budget_allocations_and_percentage_changes_l90_90251

theorem budget_allocations_and_percentage_changes (X : ℝ) :
  (14 * X / 100, 24 * X / 100, 15 * X / 100, 19 * X / 100, 8 * X / 100, 20 * X / 100) = 
  (0.14 * X, 0.24 * X, 0.15 * X, 0.19 * X, 0.08 * X, 0.20 * X) ∧
  ((14 - 12) / 12 * 100 = 16.67 ∧
   (24 - 22) / 22 * 100 = 9.09 ∧
   (15 - 13) / 13 * 100 = 15.38 ∧
   (19 - 18) / 18 * 100 = 5.56 ∧
   (8 - 7) / 7 * 100 = 14.29 ∧
   ((20 - (100 - (12 + 22 + 13 + 18 + 7))) / (100 - (12 + 22 + 13 + 18 + 7)) * 100) = -28.57) := by
  sorry

end budget_allocations_and_percentage_changes_l90_90251


namespace monthly_energy_consumption_l90_90107

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end monthly_energy_consumption_l90_90107


namespace binomial_10_3_l90_90502

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90502


namespace find_f_neg1_l90_90677

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -2^(-x) + 2*x + 1

theorem find_f_neg1 : f (-1) = -3 :=
by
  -- The proof is omitted.
  sorry

end find_f_neg1_l90_90677


namespace area_of_GHCD_is_187_5_l90_90094

structure Trapezoid (Point : Type) :=
(A B C D G H : Point)
(parallel : ∀ (X Y : Point), X = A ∧ Y = B → ∃ k : ℝ, CD = k • AB)
(length_AB : ℝ)
(length_CD : ℝ)
(altitude : ℝ)
(midpoint_G : G = (A + D) / 2)
(midpoint_H : H = (B + C) / 2)

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point] (ABCD : Trapezoid Point)

def GH_length (ABCD : Trapezoid Point) : ℝ :=
(ABCD.length_AB + ABCD.length_CD) / 2

def GHCD_altitude (ABCD : Trapezoid Point) : ℝ :=
ABCD.altitude / 2

def trapezoid_area (base1 base2 altitude : ℝ) : ℝ :=
(1 / 2) * (base1 + base2) * altitude

theorem area_of_GHCD_is_187_5 :
  trapezoid_area ABCD.length_CD (GH_length ABCD) (GHCD_altitude ABCD) = 187.5 :=
by
  have hGH_length : GH_length ABCD = 20 := sorry
  have hGHCD_altitude : GHCD_altitude ABCD = 7.5 := sorry
  rw [hGH_length, hGHCD_altitude]
  norm_num

end area_of_GHCD_is_187_5_l90_90094


namespace range_of_g_l90_90515

def g (x : ℝ) : ℝ := if x ≠ -5 then 3 * (x - 4) else 0 -- function definition 

theorem range_of_g : 
  (set.range g) = (set.Iio (-27) ∪ set.Ioi (-27)) := 
by sorry

end range_of_g_l90_90515


namespace projection_is_point_X_l90_90510

variables {A B C : Type*} [AddCommGroup A] [InnerProductSpace ℝ A]

noncomputable def find_point_X (CB AC : ℝ) (h : CB^2 + AC^2 = (complex.abs (CB + AC))^2) : ℝ :=
(AC^2) / (complex.abs (CB + AC))

theorem projection_is_point_X (ABC : Type*)
  [Field ℝ] [EuclideanSpace A] [InnerProductSpace ℝ A]
  {A B C: ℝ}
  (h_right: ∡C = π/2) (h_hypotenuse: A B hypotenuse)
  (BC AC: ℝ) (X: A)
  (h_ratio: complex.abs (BC^2 / AC^2) = complex.abs (XA / XB))
  (X = projection_of C_on AB):
  (- point X = find_point_X BC AC π/2) :=
sorry

end projection_is_point_X_l90_90510


namespace infinite_rioplatense_set_l90_90841

-- Define a recursive sequence as described in the solution
def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n + 1 => 2^(sequence n) + 2

-- Define the set A as the set defined in the correct answer
def A : Set ℕ := {a | ∃ n, a = 2^(sequence n) - 2}

-- Lean 4 statement to prove the given problem
theorem infinite_rioplatense_set :
  infinite A ∧ ∀ (a b : ℕ), a ∈ A → b ∈ A → a < b → (∀ k ∈ {0, 1, 2, 3, 4}, (b + k) % (a + k) = 0) :=
by
  -- Proof will be written here
  sorry

end infinite_rioplatense_set_l90_90841


namespace probability_sum_multiple_of_3_l90_90825

-- Define the sample space
def sample_space : List (ℕ × ℕ) :=
  List.product [1, 2, 3, 4, 5] [1, 2, 3, 4, 5]

-- Define the event where the sum of the numbers is a multiple of 3
def favorable_event (x y : ℕ) : Bool :=
  (x + y) % 3 = 0

-- Define the probability calculation
def probability_event : ℚ :=
  let favorable_outcomes := (sample_space.filter (λ p, favorable_event p.1 p.2)).length
  let total_outcomes := sample_space.length
  favorable_outcomes / total_outcomes

-- The theorem statement
theorem probability_sum_multiple_of_3 :
  probability_event = 9 / 25 :=
sorry

end probability_sum_multiple_of_3_l90_90825


namespace line_perpendicular_plane_l90_90208

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions of vectors A, B, and C
variables (A B C : V) 
-- Definition of line p
variables (p : V)

-- Non-zero condition for A, B, and C
hypothesis (hA : A ≠ 0)
hypothesis (hB : B ≠ 0)
hypothesis (hC : C ≠ 0)

-- Non-collinear condition for A, B, C, meaning they form a plane
hypothesis (non_collinear : set.finite {s : set V | ∃ a b : ℝ, ∀ v ∈ s, v = a • A + b • B ∧ v = 0})

-- Linearly dependent condition for A, B, C
hypothesis (linear_dep : ∃ α β γ : ℝ, α • A + β • B + γ • C = 0 ∧ (α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0))

-- Line p forms the same angle with A, B, C
hypothesis (same_angle : ∀ (u : V), u ∈ {A, B, C} → ∃ θ : ℝ, ⟪u, p⟫ = ∥u∥ * ∥p∥ * real.cos θ)

-- Conclude that line p is perpendicular to the plane ABC
theorem line_perpendicular_plane : 
  ⟪p, A ×ₗ B⟫ = 0 :=
sorry

end line_perpendicular_plane_l90_90208


namespace combination_10_3_l90_90404

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90404


namespace combination_10_3_l90_90405

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90405


namespace combination_10_3_l90_90414

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90414


namespace vijay_work_alone_in_24_days_l90_90272

theorem vijay_work_alone_in_24_days (ajay_rate vijay_rate combined_rate : ℝ) 
  (h1 : ajay_rate = 1 / 8) 
  (h2 : combined_rate = 1 / 6) 
  (h3 : ajay_rate + vijay_rate = combined_rate) : 
  vijay_rate = 1 / 24 := 
sorry

end vijay_work_alone_in_24_days_l90_90272


namespace polar_to_rectangular_min_distance_l90_90007

theorem polar_to_rectangular (ρ θ : ℝ) (h : ρ = 2 * cos θ) :
  ∃ x y : ℝ, (x^2 + y^2 - 2*x = 0 ∧ x = ρ * cos θ ∧ y = ρ * sin θ) := sorry

theorem min_distance (t : ℝ) :
  let l := (λ t, ⟨-2/3 * t + 2, 2/3 * t - 5⟩) in
  let M := ⟨0, -3⟩ in
  let x := λ ρ θ, ρ * cos θ in
  let y := λ ρ θ, ρ * sin θ in
  let C := (λ θ, (2 * cos θ, 2 * sin θ)) in
  ∀ θ, 
    let N := C θ in
    dist M ⟨N.1, N.2⟩ ≥ sqrt 10 - 1 := sorry

end polar_to_rectangular_min_distance_l90_90007


namespace perimeter_of_triangle_XYZ_l90_90756

/-- 
  Given the inscribed circle of triangle XYZ is tangent to XY at P,
  its radius is 15, XP = 30, and PY = 36, then the perimeter of 
  triangle XYZ is 83.4.
-/
theorem perimeter_of_triangle_XYZ :
  ∀ (XYZ : Type) (P : XYZ) (radius : ℝ) (XP PY perimeter : ℝ),
    radius = 15 → 
    XP = 30 → 
    PY = 36 →
    perimeter = 83.4 :=
by 
  intros XYZ P radius XP PY perimeter h_radius h_XP h_PY
  sorry

end perimeter_of_triangle_XYZ_l90_90756


namespace number_counts_l90_90278

def is_square (n : ℕ) := ∃ m, m * m = n
def is_odd (n : ℕ) := n % 2 = 1
def is_prime (n : ℕ) := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem number_counts :
  let numbers := [1, 3, 6, 9] in
  (count is_square numbers = 2) ∧ (count is_odd numbers = 3) ∧ (count is_prime numbers = 1) :=
by {
  sorry
}

end number_counts_l90_90278


namespace collinear_circumcenter_centroid_orthocenter_l90_90511

/-!
Theorem: The circumcenter \(O\), the centroid \(G\), and the orthocenter \(H\) of any triangle \(ABC\) 
are collinear and \(OG = \frac{1}{2} GH\).
-/

variable {A B C : Type}
variable [EuclideanGeometry A]
variables {abc : A} {o g h : Point}

/-- Main theorem -/
theorem collinear_circumcenter_centroid_orthocenter 
  (circumcenter : IsCircumcenter o abc) 
  (centroid : IsCentroid g abc) 
  (orthocenter : IsOrthocenter h abc) : 
  Collinear [o, g, h] ∧ dist o g = (1 / 2) * dist g h := 
sorry

end collinear_circumcenter_centroid_orthocenter_l90_90511


namespace selected_people_take_B_l90_90793

def arithmetic_sequence (a d n : Nat) : Nat := a + (n - 1) * d

theorem selected_people_take_B (a d total sampleCount start n_upper n_lower : Nat) :
  a = 9 →
  d = 30 →
  total = 960 →
  sampleCount = 32 →
  start = 451 →
  n_upper = 25 →
  n_lower = 16 →
  (960 / 32) = d → 
  (10 = n_upper - n_lower + 1) ∧ 
  ∀ n, (n_lower ≤ n ∧ n ≤ n_upper) → (start ≤ arithmetic_sequence a d n ∧ arithmetic_sequence a d n ≤ 750) :=
by sorry

end selected_people_take_B_l90_90793


namespace altitudes_intersect_iff_l90_90149

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

noncomputable def tetrahedron (A B C D : Type) := (∀ x ∈ {A, B, C, D}, ∃ y ∈ {A, B, C, D}, x ≠ y)

theorem altitudes_intersect_iff (T : tetrahedron A B C D) : 
  (∃ P : Type, ∀ X ∈ {A, B, C, D}, P ∈ (altitude X)) ↔ 
  AB^2 + CD^2 = BC^2 + AD^2 ∧ AB^2 + CD^2 = CA^2 + BD^2 :=
sorry

end altitudes_intersect_iff_l90_90149


namespace median_homework_duration_l90_90074

theorem median_homework_duration :
  let durations := [⟨50, 14⟩, ⟨60, 11⟩, ⟨70, 10⟩, ⟨80, 15⟩]
  (∀ d ∈ durations, ∃ n : ℕ, n = d.2) →
  (10 + 17 + 13 + 11 = 51) → -- This is to specify the count of students, correcting it to 50
  let sorted_durations := durations.sort_by (λ x y, x.1 ≤ y.1)
  let cumulative_counts := sorted_durations.scanl (λ acc x, acc + x.2) 0 
  let median_indices := ⟨25, 26⟩ -- Using the tuples for indices
  let median_values := (sorted_durations.get! (median_indices.0 - 1)).1 + 
                       (sorted_durations.get! (median_indices.1 - 1)).1
  median_values / 2 = 65 :=
by
  sorry

end median_homework_duration_l90_90074


namespace solve_exponential_eq_l90_90766

theorem solve_exponential_eq (x : ℝ) : 9^x + 3^x - 2 = 0 → x = 0 :=
by
  sorry

end solve_exponential_eq_l90_90766


namespace find_b_l90_90589

theorem find_b (p q b : ℕ) (h1 : Nat.Prime p) 
               (h2 : Nat.Prime q) 
               (h3 : 18 * p + 30 * q = 186)
               (h4 : Real.log (p.to_real / (3*q + 1).to_real) / Real.log 8 >= 0) :
               b = Real.log (p.to_real / (3*q + 1).to_real) / Real.log 8 :=
by sorry

end find_b_l90_90589


namespace binomial_10_3_l90_90339

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90339


namespace radius_of_inscribed_circle_l90_90236

noncomputable def triangle_isosceles_sides : Prop :=
  ∀ (a b c : ℝ), a = 13 ∧ b = 13 ∧ c = 10 →

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c in
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def circumradius (a b c : ℝ) : ℝ := 
  (a * b * c) / (4 * heron_area a b c)

theorem radius_of_inscribed_circle :
  triangle_isosceles_sides → 
  circumradius 13 13 10 ≈ 7.04 :=
sorry

end radius_of_inscribed_circle_l90_90236


namespace binom_10_3_l90_90353

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90353


namespace distinct_triangles_in_cube_l90_90040

theorem distinct_triangles_in_cube : 
  ∃ (n : ℕ), n = 12 ∧ 
  ∀ (cube : Type) [inhabited cube], 
    (∃ (vertices : finset cube) (edges : finset (cube × cube)), 
      vertices.card = 8 ∧ edges.card = 12 ∧
      (∀ (v : cube), edges.filter (λ e, e.1 = v ∨ e.2 = v).card = 3) ∧
        n = edges.card / 2) :=
begin
  use 12,
  sorry
end

end distinct_triangles_in_cube_l90_90040


namespace part_I_part_II_l90_90174

def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else Real.log (-x) / Real.log 0.5

theorem part_I : f (f (-1/4)) = 1 := 
  sorry

theorem part_II (a : ℝ) (h : f a > f (-a)) : 
  a ∈ Set.Ioo (-1:ℝ) 0 ∪ Set.Ioi 1 :=
  sorry

end part_I_part_II_l90_90174


namespace binomial_10_3_eq_120_l90_90456

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90456


namespace concyclic_points_l90_90309

theorem concyclic_points 
  (O O1 : circle)
  (A B : point)
  (h1 : O ∩ O1 = {A, B})
  (C D: point)
  (h2 : C ∈ O1 ∧ C ∈ line(O, A))
  (h3 : D ∈ O ∧ D ∈ line(O1, A)) :
  on_circle {B, O, D, C, O1} :=
  sorry

end concyclic_points_l90_90309


namespace transformed_point_sum_l90_90954

noncomputable theory

def f : ℝ → ℝ := sorry -- defining f as a real-valued function

-- Given condition: (12, 10) is on the graph of y = f(x)
axiom point_on_graph : f 12 = 10

-- The theorem we need to prove
theorem transformed_point_sum :
  let x := 4 in
  let y := (3:ℝ) \in \{ x : ℝ | 3*(x) = (f 12) / 3 + 3 \} in
  x + y = 55/9 :=
by {
  have h1 : f 12 = 10 := point_on_graph,
  have h2 : 3 * y = (f (3 * x)) / 3 + 3,
  { subst h1, norm_num, linarith },
  sorry
}

end transformed_point_sum_l90_90954


namespace minimum_votes_for_tall_to_win_l90_90089

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l90_90089


namespace parallel_vectors_a_value_l90_90036

noncomputable section

open Real

def vector (x y z : ℝ) : (ℝ × ℝ × ℝ) := (x, y, z)

def is_parallel (u v : ℝ × ℝ × ℝ) : Prop :=
u.1 * v.2 = u.2 * v.1 ∧ u.1 * v.3 = u.3 * v.1 ∧ u.2 * v.3 = u.3 * v.2

theorem parallel_vectors_a_value :
  let m := vector 1 a 1
  let n := vector 2 (-4) 2
  is_parallel m n → a = -2 :=
by
  intros m n h
  sorry

end parallel_vectors_a_value_l90_90036


namespace problem1_problem2_l90_90604

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * a * x - 3

theorem problem1 (a : ℝ) (h : f a (a + 1) - f a a = 9) : a = 2 :=
by sorry

theorem problem2 (a : ℝ) (h : ∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) : a = 1 ∨ a = -1 :=
by sorry

end problem1_problem2_l90_90604


namespace probability_light_on_l90_90286

axiom L1 : Prop
axiom L2 : Prop
axiom L3 : Prop
axiom P : Prop → ℝ
axiom independent : ∀ (A B : Prop), Prop

-- Conditions
axiom prob_L1 : P(L1) = 0.5
axiom prob_L2 : P(L2) = 0.5
axiom prob_L3 : P(L3) = 0.5
axiom L1_ind_L2 : independent L1 L2
axiom L1_ind_L3 : independent L1 L3
axiom L2_ind_L3 : independent L2 L3

-- Required proof
theorem probability_light_on : P(L1 ∪ (L2 ∩ L3)) = 0.625 := by
  sorry

end probability_light_on_l90_90286


namespace circular_trip_exists_l90_90163

theorem circular_trip_exists (n : ℕ) (h_n : n = 2018) (h_routes : ∀ (i : ℕ), i < n → ∃ (adj : finset ℕ), adj.card ≥ 6 ∧ ∀ j ∈ adj, j < n) :
  ∃ (cycle : list ℕ), cycle.length ≥ 7 ∧ (∀ i ∈ cycle.init, i ≠ cycle.last sorry) :=
sorry

end circular_trip_exists_l90_90163


namespace smaller_solid_volume_l90_90899

-- Define the coordinates for the vertices involved
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def V := { x := 0, y := 0, z := 0 }
def P := { x := 1, y := 0, z := 2 }
def Q := { x := 2, y := 2, z := 1 }

-- The theorem stating the volume of the smaller solid
theorem smaller_solid_volume : 
  let volume : ℝ := 1 / 6 
  volume = 1 / 6 :=
by 
  sorry

end smaller_solid_volume_l90_90899


namespace minimum_route_length_l90_90166

/-- 
Given a city with the shape of a 5 × 5 square grid,
prove that the minimum length of a route that covers each street exactly once and 
returns to the starting point is 68, considering each street can be walked any number of times. 
-/
theorem minimum_route_length (n : ℕ) (h1 : n = 5) : 
  ∃ route_length : ℕ, route_length = 68 := 
sorry

end minimum_route_length_l90_90166


namespace pqrs_sum_l90_90213

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end pqrs_sum_l90_90213


namespace binomial_coefficient_10_3_l90_90370

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90370


namespace binomial_10_3_eq_120_l90_90459

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90459


namespace stream_speed_zero_l90_90798

theorem stream_speed_zero (v_c v_s : ℝ)
  (h1 : v_c - v_s - 2 = 9)
  (h2 : v_c + v_s + 1 = 12) :
  v_s = 0 := 
sorry

end stream_speed_zero_l90_90798


namespace jay_bought_3_cups_of_milk_tea_l90_90207

theorem jay_bought_3_cups_of_milk_tea
  (mitch_paid : ℕ)
  (jam_paid : ℕ)
  (jay_paid_per_cup : ℕ)
  (total_contribution : ℕ)
  (friends_contribution_each : ℕ)
  (total_contribution_sum : total_contribution = 3 * friends_contribution_each)
  (mitch_paid_21 : mitch_paid = 3 * 7)
  (jam_paid_3 : jam_paid = 2 * 1.5 )
  (total_expense_without_milk_tea : mitch_paid + jam_paid = 24) :
  (total_contribution - (mitch_paid + jam_paid)) / jay_paid_per_cup = 3 := 
by 
  sorry

end jay_bought_3_cups_of_milk_tea_l90_90207


namespace at_least_one_weight_greater_than_35_l90_90565

-- Define the properties
variables {a : ℕ → ℕ} (h_distinct : function.injective a)

-- Define the condition: for any subsets S1 and S2 with |S1| ≠ |S2|,
-- the subset with more elements has a greater sum of weights.
def condition (S1 S2 : finset ℕ) : Prop :=
  S1.card ≠ S2.card → (∀ h₁ h₂, S1.sum (λ i, a i) > S2.sum (λ i, a i))

-- Formulate the main theorem
theorem at_least_one_weight_greater_than_35 :
  (∀ S1 S2 : finset ℕ, condition a S1 S2)
  → (∃ i : ℕ, a i > 35) :=
sorry

end at_least_one_weight_greater_than_35_l90_90565


namespace partI_partII_partIII_l90_90023

-- Part Ⅰ
theorem partI (m k: ℝ) (h₁: m = 2) (h₂: k = 2) (QA QB: ℝ × ℝ) (n: ℝ)
  (h₃: QA = (6, 2 - n)) (h₄: QB = (6, 2 - n)) (dot_prod: QA.1 * QB.1 + QA.2 * QB.2 = 0) : n = 2 := 
sorry

-- Part Ⅱ
theorem partII (m: ℝ) (h₁: \( \forall k \neq 0 \))
  (dot_prod: let A := (6, 2); let B := (6, 2) in (A.1 * B.1 + A.2 * B.2 = 0)) : m = 0 ∨ m = 8 := 
sorry

-- Part Ⅲ
theorem partIII (k n: ℝ) (m: ℝ) (h₁: k = 1) (h₂: n = 0) (h₃: m < 0) : 
  let area := \( -m \times \sqrt{64 + 32m} \) in 
  area ≤ \( \frac{32\sqrt{3}}{9} \) :=
sorry

end partI_partII_partIII_l90_90023


namespace sum_of_possible_t_l90_90225

def vertices (t : ℝ) : Prop :=
  (0 ≤ t) ∧ (t ≤ 360) ∧ 
  (is_isosceles (cos ∘ degrees_to_radians 30, sin ∘ degrees_to_radians 30) 
                (cos ∘ degrees_to_radians 90, sin ∘ degrees_to_radians 90) 
                (cos ∘ degrees_to_radians t, sin ∘ degrees_to_radians t))

theorem sum_of_possible_t : 
  ∑ t in {t | vertices t}, t = 840 :=
sorry

end sum_of_possible_t_l90_90225


namespace curve_intersection_problem_l90_90607

theorem curve_intersection_problem (a b θ : ℝ) (a_pos : a > b) (b_pos : b > 0) :
  let x := a * cos θ;
  let y := b * sin θ;
  let M := (1, sqrt 3 / 2);
  let ρ : ℝ := 2 * sin θ;
  ∃ (OA OB : ℝ), (θ = π / 3) → 
  |OA| = sqrt (4 - ρ * ρ) / sqrt 2 ∧ 
  |OB| = sqrt (2 * (1 - ρ * ρ)) / sqrt 5 ∧ 
  (1 / |OA|^2 + 1 / |OB|^2 = 5 / 4) := 
by 
  admit

end curve_intersection_problem_l90_90607


namespace marge_final_plants_l90_90697

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end marge_final_plants_l90_90697


namespace sonika_initial_deposit_l90_90735

variable (P R : ℝ)

theorem sonika_initial_deposit :
  (P + (P * R * 3) / 100 = 9200) → (P + (P * (R + 2.5) * 3) / 100 = 9800) → P = 8000 :=
by
  intros h1 h2
  sorry

end sonika_initial_deposit_l90_90735


namespace larger_cookie_sugar_l90_90250

theorem larger_cookie_sugar :
  let initial_cookies := 40
  let initial_sugar_per_cookie := 1 / 8
  let total_sugar := initial_cookies * initial_sugar_per_cookie
  let larger_cookies := 25
  let sugar_per_larger_cookie := total_sugar / larger_cookies
  sugar_per_larger_cookie = 1 / 5 := by
sorry

end larger_cookie_sugar_l90_90250


namespace quadrilateral_bisect_each_other_implies_parallelogram_l90_90810

-- Definition of a quadrilateral where its diagonals bisect each other
structure Quadrilateral where
  A B C D : Type
  -- Diagonal endpoints
  AC BD : Type
  -- Midpoints of diagonals' segments
  midpoint_AC : AC
  midpoint_BD : BD
  -- The property that each diagonal midpoint coincides with the other
  bisect_each_other : midpoint_AC = midpoint_BD

-- Proposition to prove that such a quadrilateral is a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry  -- proof goes here

-- Final statement
theorem quadrilateral_bisect_each_other_implies_parallelogram (q : Quadrilateral) : is_parallelogram q := 
  sorry

end quadrilateral_bisect_each_other_implies_parallelogram_l90_90810


namespace length_of_AB_l90_90814

theorem length_of_AB :
  ∃ (a b c d e : ℝ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
  (b - a = 5) ∧ -- AB = 5
  ((c - b) = 2 * (d - c)) ∧ -- bc = 2 * cd
  (d - e) = 4 ∧ -- de = 4
  (c - a) = 11 ∧ -- ac = 11
  (e - a) = 18 := -- ae = 18
by 
  sorry

end length_of_AB_l90_90814


namespace total_musicians_is_98_l90_90777

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end total_musicians_is_98_l90_90777


namespace binom_10_3_l90_90347

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90347


namespace range_of_g_l90_90518

noncomputable def g (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_of_g :
  set.range g = {y : ℝ | y ≠ -27} :=
sorry

end range_of_g_l90_90518


namespace find_f_of_neg3_l90_90585

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then log (x + 1) / log 2 else -log (-x + 1) / log 2

theorem find_f_of_neg3 :
  (∀ x, f (-x) = -f x) →
  (∀ (x : ℝ), 0 ≤ x → f x = log (x + 1) / log 2) →
  f (-3) = -2 :=
by
  sorry

end find_f_of_neg3_l90_90585


namespace binomial_10_3_eq_120_l90_90448

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90448


namespace binom_10_3_eq_120_l90_90465

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90465


namespace problem_1_problem_2_l90_90948

variables (a b t : ℝ) (f : ℝ → ℝ) (u1 u2 u3 : ℝ)

-- Problem 1: Define the quadratic equation and the function f
def quadratic_eq (x t : ℝ) : ℝ := 4 * x^2 - 4 * t * x - 1

-- Ensure a and b are roots of the quadratic equation 
def is_root (a b t : ℝ) : Prop := quadratic_eq a t = 0 ∧ quadratic_eq b t = 0 ∧ a ≠ b

-- f is monotonically increasing on the domain [a, b]
-- Define g(t) in terms of f(a) and f(b)
noncomputable def g (t : ℝ) : ℝ := f b - f a

-- Problem 2: Given trigonometric conditions and inequality
def trigonometric_ineq (u1 u2 u3 : ℝ) : Prop := 
  u1 ∈ (0, π) ∧ u2 ∈ (0, π) ∧ u3 ∈ (0, π) ∧
  sin u1 + sin u2 + sin u3 = 1 

theorem problem_1 (h : is_root a b t) : g t = f b - f a := sorry

theorem problem_2 (h : trigonometric_ineq u1 u2 u3) : tan u1 + tan u2 + tan u3 < 72 := sorry

end problem_1_problem_2_l90_90948


namespace sum_of_digits_of_max_prime_l90_90896

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, (1 < m ∧ m < n) → n % m ≠ 0

noncomputable def subset_prod_sum (A B : set ℕ) : ℕ :=
  (A.prod id) + (B.prod id)

def set_E := {5, 6, 7, 8, 9}

theorem sum_of_digits_of_max_prime :
  ∃ N : ℕ, N ∈ {N | ∃ (A B : set ℕ), A ∪ B = set_E ∧ A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ is_prime (subset_prod_sum A B) ∧ N = subset_prod_sum A B}
  ∧ (N.to_digits.sum) = 17 :=
sorry

end sum_of_digits_of_max_prime_l90_90896


namespace binomial_10_3_eq_120_l90_90458

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90458


namespace ellipse_properties_l90_90942

theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (eccentricity_condition : c / a = (sqrt 3) / 2)
  (point_condition : ∀ x y : ℝ, ((x / a) ^ 2 + (y / b) ^ 2 = 1) → (x, y) = (1, sqrt 3 / 2))
  :
  (a = 2) ∧ (b = 1) ∧ (c = sqrt 3) ∧
  (∀ (x y : ℝ), (x/a)^2 + (y/b)^2 = 1 ↔ x^2 / 4 + y^2 = 1) ∧
  (∀ (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ),
    sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 4 →
    |(sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)) * (sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2))| ≤ 4) :=
sorry

end ellipse_properties_l90_90942


namespace monthly_energy_consumption_l90_90108

-- Defining the given conditions
def power_fan_kW : ℝ := 0.075 -- kilowatts
def hours_per_day : ℝ := 8 -- hours per day
def days_per_month : ℝ := 30 -- days per month

-- The math proof statement with conditions and the expected answer
theorem monthly_energy_consumption : (power_fan_kW * hours_per_day * days_per_month) = 18 :=
by
  -- Placeholder for proof
  sorry

end monthly_energy_consumption_l90_90108


namespace monochromatic_triangles_l90_90141

theorem monochromatic_triangles (n : ℕ) (h : n ≥ 8) 
(points : Fin n → Point)
(colors : Fin n → Color)
(no_collinear : ∀ (a b c : Fin n), a ≠ b → b ≠ c → a ≠ c → ¬collinear (points a) (points b) (points c))
(triangles_set : Set (Triangle (Fin n)))
(condition_S : ∀ (i j u v : Fin n), i ≠ j → u ≠ v → 
  count_triangles_containing i j triangles_set = count_triangles_containing u v triangles_set) : 
  ∃ t₁ t₂ ∈ triangles_set, monochromatic t₁ colors ∧ monochromatic t₂ colors :=
sorry

end monochromatic_triangles_l90_90141


namespace total_suitcases_l90_90695

/-
  Lily's family includes:
  her 6 siblings, each bringing a different number of suitcases ranging from 1 to 6,
  her parents, each bringing 3 suitcases,
  her 2 grandparents, each bringing 2 suitcases,
  and her 3 other relatives, who are bringing a total of 8 suitcases combined.
  Prove that the entire family is bringing a total of 39 suitcases on vacation.
-/

theorem total_suitcases : 
  let siblings_suitcases := 1 + 2 + 3 + 4 + 5 + 6 in
  let parents_suitcases := 3 * 2 in
  let grandparents_suitcases := 2 * 2 in
  let other_relatives_suitcases := 8 in
  siblings_suitcases + parents_suitcases + grandparents_suitcases + other_relatives_suitcases = 39 :=
by
  let siblings_suitcases := 1 + 2 + 3 + 4 + 5 + 6
  let parents_suitcases := 3 * 2
  let grandparents_suitcases := 2 * 2
  let other_relatives_suitcases := 8
  show siblings_suitcases + parents_suitcases + grandparents_suitcases + other_relatives_suitcases = 39
  sorry

end total_suitcases_l90_90695


namespace combination_10_3_l90_90411

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90411


namespace combination_10_3_l90_90412

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90412


namespace trajectory_eq_of_point_M_find_m_value_l90_90006

-- Define the given points A and B
def A : (ℝ × ℝ) := (-2, 0)
def B : (ℝ × ℝ) := (2, 0)

-- Define the condition on the slopes and derive the trajectory equation
theorem trajectory_eq_of_point_M (x y : ℝ) :
  y ≠ 0 → x ≠ 2 → x ≠ -2 →
  (y / (x + 2)) * (y / (x - 2)) = -3 / 4 →
  (x^2 / 4) + (y^2 / 3) = 1 := by
  sorry

-- Define the conditions for part II and derive the value of m
theorem find_m_value (m : ℝ) :
  m ≠ 0 →
  x = my - 2 →
  2 * sqrt(6) = (1 / 2) * (12 * m^2 / (3 * m^2 + 2)) * (4 / |m|) →
  m = sqrt(6) / 3 := by
  sorry

end trajectory_eq_of_point_M_find_m_value_l90_90006


namespace binom_10_3_eq_120_l90_90471

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90471


namespace parallel_lines_probability_triangle_inequality_probability_l90_90210

open Set

noncomputable def probability_parallel_lines : ℚ :=
  1 / 12

noncomputable def probability_triangle_inequality : ℚ :=
  5 / 12

theorem parallel_lines_probability :
  let outcomes := {(a, b) | a ∈ Finset.range 1 7 ∧ b ∈ Finset.range 1 7}
  let favorable_outcomes := { (a, b) | (a, b) ∈ outcomes ∧ (a = 1 ∧ b = 2 ∨ a = 2 ∧ b = 4 ∨ a = 3 ∧ b = 6) }
  (Finset.card favorable_outcomes) / (Finset.card outcomes) = probability_parallel_lines := 
by sorry

theorem triangle_inequality_probability :
  let outcomes := {(a, b) | a ∈ Finset.range 1 7 ∧ b ∈ Finset.range 1 7}
  let favorable_outcomes := { (a, b) | (a, b) ∈ outcomes ∧ (a + b > 2 ∧ abs (a - b) < 2) }
  (Finset.card favorable_outcomes) / (Finset.card outcomes) = probability_triangle_inequality :=
by sorry

end parallel_lines_probability_triangle_inequality_probability_l90_90210


namespace find_value_r_l90_90640

open Real

def curve1_parametric (t : ℝ) : ℝ × ℝ :=
  (sqrt 2 * cos t, sin t)

def curve2_polar (r : ℝ) (θ : ℝ) : ℝ :=
  r * (cos θ, sin θ)

def curve1_cartesian (x y : ℝ) : Prop :=
  (x ^ 2) / 2 + y ^ 2 = 1

def curve2_cartesian (x y r : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = r ^ 2

def polar_eq (ρ θ : ℝ) : ℝ :=
  ρ ^ 2 * (cos θ ^ 2 + 2 * sin θ ^ 2)

theorem find_value_r :
  ∃ r > 0, r = sqrt 6 / 3 :=
by sorry

end find_value_r_l90_90640


namespace maximize_xyz_l90_90122

theorem maximize_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x + y + z = 60) :
    (x, y, z) = (20, 40 / 3, 80 / 3) → x^3 * y^2 * z^4 ≤ 20^3 * (40 / 3)^2 * (80 / 3)^4 :=
by
  sorry

end maximize_xyz_l90_90122


namespace selling_price_l90_90263

variable (commissionRate : ℝ) (commission : ℝ)

-- Given conditions
def conditions : Prop := 
  commissionRate = 0.06 ∧ commission = 8880

-- The main theorem stating the selling price
theorem selling_price (h : conditions) : ∃ P : ℝ, commission = commissionRate * P ∧ P = 148000 :=
by 
  obtain ⟨h1, h2⟩ from h
  use 148000
  split
  · rw [h1, h2]
    norm_num
  · refl

end selling_price_l90_90263


namespace concyclic_B_M_L_N_l90_90124

variables {A B C D L K M N : Type}
variables [Geometry A] [Geometry B] [Geometry C] [Geometry D] [Geometry L] [Geometry K] [Geometry M] [Geometry N]

-- Definitions based on conditions
def point_on_minor_arc (L : Type) (C D : Type) (circumcircle : Circle) : Prop := 
  ∃ (L ∈ circumcircle.arc C D), L ≠ C ∧ L ≠ D

def intersects_line (P Q R : Type) : Prop :=
  ∃ R, Line_through P P ∩ Line_through Q Q

-- Main theorem statement
theorem concyclic_B_M_L_N 
  (square : Square A B C D) 
  (L : Type) 
  (circumcircle : Circumcircle A B C D)
  (hL : point_on_minor_arc L C D circumcircle)
  (K : Type) (hK : intersects_line A L K) 
  (M : Type) (hM : intersects_line C L M)
  (N : Type) (hN : intersects_line M K N) : 
  are_concyclic [B, M, L, N] :=
sorry

end concyclic_B_M_L_N_l90_90124


namespace sum_multiples_of_2_and_3_l90_90541

theorem sum_multiples_of_2_and_3 :
  let multiples := {n | 1 ≤ n ∧ n ≤ 100 ∧ n % 2 = 0 ∧ n % 3 = 0}.to_list
  in multiples.sum = 816 :=
by
  let multiples := {n | 1 ≤ n ∧ n ≤ 100 ∧ n % 2 = 0 ∧ n % 3 = 0}.to_list
  sorry

end sum_multiples_of_2_and_3_l90_90541


namespace min_value_ratio_l90_90689

noncomputable def min_ratio (a b c : ℝ) : ℝ :=
  a / b + b / c + c / a

theorem min_value_ratio (a b c : ℝ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h3 : ∃ r : ℝ, (a * r^3 + b * r + c = 0 ∧ b * r^3 + c * r + a = 0 ∧ c * r^3 + a * r + b = 0) 
  ∧ (¬(∃ i : ℂ, ∀ x : ℝ, polynomial.has_root (polynomial.C a * polynomial.X^3 + polynomial.C b * polynomial.X + polynomial.C c) i)
  ∧ ¬(∃ i : ℂ, ∀ x : ℝ, polynomial.has_root (polynomial.C b * polynomial.X^3 + polynomial.C c * polynomial.X + polynomial.C a) i))
  ∨ 
  (¬(∃ i : ℂ, ∀ x : ℝ, polynomial.has_root (polynomial.C b * polynomial.X^3 + polynomial.C c * polynomial.X + polynomial.C a) i)
  ∧ ¬(∃ i : ℂ, ∀ x : ℝ, polynomial.has_root (polynomial.C c * polynomial.X^3 + polynomial.C a * polynomial.X + polynomial.C b) i))
  ) 
: min_ratio a b c = 3.833 :=
by sorry

end min_value_ratio_l90_90689


namespace inequality_solution_l90_90551

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end inequality_solution_l90_90551


namespace divide_six_unit_squares_l90_90732

theorem divide_six_unit_squares (c : ℚ) :
  let total_area : ℚ := 6,
      half_area : ℚ := total_area / 2,
      line_slope : ℚ := 4 / (4 - c),
      triangle_area : ℚ := ½ * (4 - c) * 4 in
  (c = 5/2) → triangle_area = half_area :=
by
  intros,
  sorry

end divide_six_unit_squares_l90_90732


namespace data_input_rate_l90_90227

theorem data_input_rate :
  ∀ (x : ℕ), (x > 0) → (∀ (x > 0), 
  ∀ (A_input B_input : ℕ), 
    A_input = 2 * x ∧
    B_input = x ∧
    2640 / B_input - 2640 / (2 * B_input) = 2)
  → (A_input_per_minute = 22 ∧ B_input_per_minute = 11) :=
by
  sorry

end data_input_rate_l90_90227


namespace second_player_wins_optimal_play_l90_90819

structure GameGraph :=
  (vertices : set String) -- The set of vertices, e.g., {"A", "B", "C", "D", "E", "F"}
  (edges : set (String × String)) -- The set of edges connecting the vertices
  (degree : String → ℕ) -- Function mapping each vertex to its degree
  (initial_vertex : String) -- The starting vertex for the game

-- Example instantiation for the provided graph
def myGraph : GameGraph :=
  { vertices := {"A", "B", "C", "D", "E", "F"},
    edges := {("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("B", "F"), ("C", "F"), 
               ("D", "F"), ("E", "F")},
    degree := λ v, if v = "A" then 4 else 2,
    initial_vertex := "A"
  }

theorem second_player_wins_optimal_play (G : GameGraph) (hGraph : 
  ∀ v ∈ G.vertices, G.degree v % 2 = 0 ∧ (v = "A" → G.degree v = 4) ∧ (v ≠ "A" → G.degree v = 2) ) :
  "2nd player wins with optimal play" :=
sorry

end second_player_wins_optimal_play_l90_90819


namespace binomial_10_3_l90_90330

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90330


namespace Peter_buys_more_hot_dogs_than_hamburgers_l90_90921

theorem Peter_buys_more_hot_dogs_than_hamburgers :
  let chicken := 16
  let hamburgers := chicken / 2
  (exists H : Real, 16 + hamburgers + H + H / 2 = 39 ∧ (H - hamburgers = 2)) := sorry

end Peter_buys_more_hot_dogs_than_hamburgers_l90_90921


namespace tan_alpha_minus_beta_alpha_plus_beta_l90_90557

variable (α β : ℝ)

-- Conditions as hypotheses
axiom tan_alpha : Real.tan α = 2
axiom tan_beta : Real.tan β = -1 / 3
axiom alpha_range : 0 < α ∧ α < Real.pi / 2
axiom beta_range : Real.pi / 2 < β ∧ β < Real.pi

-- Proof statements
theorem tan_alpha_minus_beta : Real.tan (α - β) = 7 := by
  sorry

theorem alpha_plus_beta : α + β = 5 * Real.pi / 4 := by
  sorry

end tan_alpha_minus_beta_alpha_plus_beta_l90_90557


namespace quadratic_inequality_solution_l90_90529

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3*x + 2 < 0} = set.Ioo 1 2 :=
by
  sorry

end quadratic_inequality_solution_l90_90529


namespace cards_left_l90_90293

theorem cards_left (bask_boxes : ℕ) (bask_cards_per_box : ℕ) (base_boxes : ℕ) (base_cards_per_box : ℕ) (cards_given : ℕ) :
  bask_boxes = 4 → bask_cards_per_box = 10 → base_boxes = 5 → base_cards_per_box = 8 → cards_given = 58 →
  (bask_boxes * bask_cards_per_box + base_boxes * base_cards_per_box - cards_given) = 22 :=
begin
  sorry, -- proof is skipped as per the instructions
end

end cards_left_l90_90293


namespace find_equation_of_line_l90_90590

-- Define the given conditions
def center_of_circle : ℝ × ℝ := (0, 3)
def perpendicular_line_slope : ℝ := -1
def perpendicular_line_equation (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the proof problem
theorem find_equation_of_line (x y : ℝ) (l_passes_center : (x, y) = center_of_circle)
 (l_is_perpendicular : ∀ x y, perpendicular_line_equation x y ↔ (x-y+3=0)) : x - y + 3 = 0 :=
sorry

end find_equation_of_line_l90_90590


namespace combination_10_3_eq_120_l90_90379

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90379


namespace range_of_g_l90_90520

noncomputable def g (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_of_g :
  set.range g = {y : ℝ | y ≠ -27} :=
sorry

end range_of_g_l90_90520


namespace NH4Cl_reacts_to_NH4OH_l90_90532

namespace Chemistry

-- Define the reaction conditions as hypotheses
def NH4Cl_reacts_with_H2O (moles_NH4Cl : ℕ) (moles_H2O : ℕ) : Prop :=
  moles_NH4Cl = 1 ∧ moles_H2O = 1

def produces_HCl (moles_HCl : ℕ) : Prop :=
  moles_HCl = 1

def H2O_molar_mass (mass_H2O : ℕ) : Prop :=
  mass_H2O = 18

-- The theorem we need to prove
theorem NH4Cl_reacts_to_NH4OH :
  ∀ (moles_NH4Cl moles_H2O moles_HCl mass_H2O : ℕ),
    NH4Cl_reacts_with_H2O moles_NH4Cl moles_H2O →
    produces_HCl moles_HCl →
    H2O_molar_mass mass_H2O →
    (∃ (compound : String), compound = "NH4OH") :=
by
  assume moles_NH4Cl moles_H2O moles_HCl mass_H2O,
  assume h1 : NH4Cl_reacts_with_H2O moles_NH4Cl moles_H2O,
  assume h2 : produces_HCl moles_HCl,
  assume h3 : H2O_molar_mass mass_H2O,
  existsi "NH4OH",
  sorry

end Chemistry

end NH4Cl_reacts_to_NH4OH_l90_90532


namespace minimum_voters_for_tall_win_l90_90086

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end minimum_voters_for_tall_win_l90_90086


namespace major_axis_length_l90_90946

open Real

theorem major_axis_length (a b : ℝ) (h1 : 0 < b) (h2 : b < a)
    (h3 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → ((x = sqrt 6 ∧ y = 0) ∨
    (∃ l, ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (A.1 / a^2 + A.2 / b^2 = 1) ∧ 
    (B.1 / a^2 + B.2 / b^2 = 1) ∧ 
    (A, B, F) collinear ∧ 
    midpoint A B P ∧
    isosceles_with_base OFP OF) ∧ 
    circumcircle_area OFP = 2 * π)) :
  2 * a = 6 :=
sorry

end major_axis_length_l90_90946


namespace cube_split_with_333_l90_90542

theorem cube_split_with_333 (m : ℕ) (h1 : m > 1)
  (h2 : ∃ k : ℕ, (333 = 2 * k + 1) ∧ (333 + 2 * (k - k) + 2) * k = m^3 ) :
  m = 18 := sorry

end cube_split_with_333_l90_90542


namespace largest_root_of_polynomial_l90_90175

noncomputable def polynomial := (x : ℝ) → x^6 - 6 * x^5 + 17 * x^4 + 6 * x^3 + a * x^2 - b * x - c

theorem largest_root_of_polynomial (a b c : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    polynomial x₁ = 0 ∧ polynomial x₂ = 0 ∧ polynomial x₃ = 0) →
  ∀ (x : ℝ), polynomial x = 0 → x ≤ 2 :=
sorry

end largest_root_of_polynomial_l90_90175


namespace find_r_in_geometric_series_l90_90508

theorem find_r_in_geometric_series
  (a r : ℝ)
  (h1 : a / (1 - r) = 15)
  (h2 : a / (1 - r^2) = 6) :
  r = 2 / 3 :=
sorry

end find_r_in_geometric_series_l90_90508


namespace largest_square_plot_size_l90_90834

def field_side_length := 50
def available_fence_length := 4000

theorem largest_square_plot_size :
  ∃ (s : ℝ), (0 < s) ∧ (s ≤ field_side_length) ∧ 
  (100 * (field_side_length - s) = available_fence_length) →
  s = 10 :=
by
  sorry

end largest_square_plot_size_l90_90834


namespace cost_of_traveling_all_roads_l90_90847

noncomputable def total_cost_of_roads (length width road_width : ℝ) (cost_per_sq_m : ℝ) : ℝ :=
  let area_road_parallel_length := length * road_width
  let area_road_parallel_breadth := width * road_width
  let diagonal_length := Real.sqrt (length^2 + width^2)
  let area_road_diagonal := diagonal_length * road_width
  let total_area := area_road_parallel_length + area_road_parallel_breadth + area_road_diagonal
  total_area * cost_per_sq_m

theorem cost_of_traveling_all_roads :
  total_cost_of_roads 80 50 10 3 = 6730.2 :=
by
  sorry

end cost_of_traveling_all_roads_l90_90847


namespace probability_of_drawing_orange_marble_second_l90_90292

noncomputable def probability_second_marble_is_orange (total_A white_A black_A : ℕ) (total_B orange_B green_B blue_B : ℕ) (total_C orange_C green_C blue_C : ℕ) : ℚ := 
  let p_white := (white_A : ℚ) / total_A
  let p_black := (black_A : ℚ) / total_A
  let p_orange_B := (orange_B : ℚ) / total_B
  let p_orange_C := (orange_C : ℚ) / total_C
  (p_white * p_orange_B) + (p_black * p_orange_C)

theorem probability_of_drawing_orange_marble_second :
  probability_second_marble_is_orange 9 4 5 15 7 5 3 10 4 4 2 = 58 / 135 :=
by
  sorry

end probability_of_drawing_orange_marble_second_l90_90292


namespace lucas_quadratic_l90_90128

theorem lucas_quadratic (c : ℝ) (n : ℝ) 
  (h₁ : (x : ℝ) → x^2 + c * x + 1 / 4 = (x + n)^2 + 1 / 8) : 
  c = - (Real.sqrt 2) / 2 :=
by
  have h₂ : n^2 + 1/8 = 1/4 := by sorry,
  have h₃ : n^2 = 1/4 - 1/8 := by sorry,
  have h₄ : n = - (Real.sqrt 2) / 4 := by sorry,
  have h₅ : c = 2 * (- (Real.sqrt 2) / 4) := by sorry,
  exact h₅

end lucas_quadratic_l90_90128


namespace option_A_option_B_option_C_option_D_l90_90231

-- Definitions for required properties
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def is_collinear (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = k • a

def is_acute (a b : ℝ × ℝ) : Prop := dot_product a b > 0

-- Option A
theorem option_A : ∃ (a b c : ℝ × ℝ), dot_product a b = dot_product b c ∧ a ≠ c := sorry

-- Option B
theorem option_B (a b : ℝ × ℝ) (h₁ : a ≠ (0, 0)) (h₂ : b ≠ (0, 0)) (h₃ : |a - b| = |a| + |b|) : is_collinear a b ∧ ∃ k : ℝ, k < 0 ∧ b = k • a := sorry

-- Option C
theorem option_C (a b : ℝ × ℝ) (t : ℝ) : is_collinear (t • a + 2 • b) (2 • a + 3 • b) → t ≠ 4 / 3 := sorry

-- Option D
theorem option_D (λ : ℝ) : ¬ (∃ (a b : ℝ × ℝ) (λ ∈ ℝ), (a = (1,2)) ∧ (b = (-1,1)) ∧ (is_acute a (a + λ • b))) := sorry

end option_A_option_B_option_C_option_D_l90_90231


namespace slope_range_midpoint_coordinates_product_AM_AN_l90_90567

noncomputable def circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 4

noncomputable def line_l1 (x y k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def line_l2 (x y : ℝ) : Prop :=
  x + 2 * y + 2 = 0

noncomputable def AM (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem slope_range (k : ℝ) : 
  (∃ x y, line_l1 x y k ∧ circle x y) ↔ k > 3/4 := sorry

theorem midpoint_coordinates (k : ℝ) :
  (∃ x y, line_l1 x y k ∧ circle x y) →
  (∃ (xM yM : ℝ), xM = (k^2 + 4*k + 3) / (k^2 + 1) ∧ yM = (4*k^2 + 2*k) / (k^2 + 1)) := sorry

theorem product_AM_AN (k : ℝ) (xM yM xN yN : ℝ) : 
  line_l1 xM yM k ∧ circle xM yM ∧ 
  line_l1 xN yN k ∧ line_l2 xN yN → 
  AM 1 0 xM yM * AM 1 0 xN yN = 6 := sorry 

end slope_range_midpoint_coordinates_product_AM_AN_l90_90567


namespace total_cups_l90_90143

theorem total_cups (t1 t2 : ℕ) (h1 : t2 = 240) (h2 : t2 = t1 - 20) : t1 + t2 = 500 := by
  sorry

end total_cups_l90_90143


namespace max_real_part_value_l90_90683

noncomputable def max_real_part_zeros_poly : ℂ :=
  let z := λ j, 16 * complex.exp (complex.I * (2 * real.pi * j / 8)) in
  let w := λ j, if (real.part (z j) > 0) then z j
                else if (real.part (z j) < 0) then -z j
                else complex.I * z j in
  ∑ j in finset.range 8, w j

theorem max_real_part_value :
  (max_real_part_zeros_poly).re = 32 + 32 * real.sqrt 2 :=
sorry

end max_real_part_value_l90_90683


namespace range_of_a_l90_90539

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, (x1 ∈ set.Icc (Real.pi / 3) Real.pi ∧ x2 ∈ set.Icc (Real.pi / 3) Real.pi ∧ x1 ≠ x2 ∧ 
  Real.sin x1 = (1 - a) / 2 ∧ Real.sin x2 = (1 - a) / 2)) 
  ↔ a ∈ set.Ioc (-1) (1 - Real.sqrt 3) :=
by sorry

end range_of_a_l90_90539


namespace product_of_x_is_minus_162_l90_90675

def f (x : ℝ) : ℝ := 18 * x + 4
noncomputable def f_inv (x : ℝ) : ℝ := (x - 4) / 18

theorem product_of_x_is_minus_162 :
  (∏ x in { x : ℝ | f_inv x = f (2 * x)⁻¹}.to_finset, x) = -162 :=
  sorry

end product_of_x_is_minus_162_l90_90675


namespace no_natural_solution_l90_90148

theorem no_natural_solution :
  ¬ (∃ (x y : ℕ), 2 * x + 3 * y = 6) :=
by
sorry

end no_natural_solution_l90_90148


namespace should_agree_to_buy_discount_card_l90_90856

-- Define the conditions
def discount_card_cost := 100
def discount_percentage := 0.03
def cost_of_cakes := 4 * 500
def cost_of_fruits := 1600
def total_cost_without_discount_card := cost_of_cakes + cost_of_fruits
def discount_amount := total_cost_without_discount_card * discount_percentage
def cost_after_discount := total_cost_without_discount_card - discount_amount
def effective_total_cost_with_discount_card := cost_after_discount + discount_card_cost

-- Define the objective statement to prove
theorem should_agree_to_buy_discount_card : effective_total_cost_with_discount_card < total_cost_without_discount_card := by
  sorry

end should_agree_to_buy_discount_card_l90_90856


namespace circle_through_points_l90_90024

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x

def line_eq (x y : ℝ) : Prop := y = x - 1

def circle_eq (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circle_through_points (a b r : ℝ) :
  (parabola_eq (fst parabola_focus) (snd parabola_focus) ∧
   line_eq (fst parabola_focus) (snd parabola_focus) ∧
   ∃ x₁ y₁ x₂ y₂, parabola_eq x₁ y₁ ∧ line_eq x₁ y₁ ∧
                   parabola_eq x₂ y₂ ∧ line_eq x₂ y₂ ∧
                   circle_eq x₁ y₁ a b r ∧ circle_eq x₂ y₂ a b r ∧
                   a = 3 ∧ b = 2 ∧ r = 4) :=
by {
  sorry
}

end circle_through_points_l90_90024


namespace total_musicians_count_l90_90780

-- Define the given conditions
def orchestra_males := 11
def orchestra_females := 12
def choir_males := 12
def choir_females := 17

-- Total number of musicians in the orchestra
def orchestra_musicians := orchestra_males + orchestra_females

-- Total number of musicians in the band
def band_musicians := 2 * orchestra_musicians

-- Total number of musicians in the choir
def choir_musicians := choir_males + choir_females

-- Total number of musicians in the orchestra, band, and choir
def total_musicians := orchestra_musicians + band_musicians + choir_musicians

-- The theorem to prove
theorem total_musicians_count : total_musicians = 98 :=
by
  -- Lean proof part goes here.
  sorry

end total_musicians_count_l90_90780


namespace find_inverse_log_l90_90975

-- Given conditions:
variables (x : ℝ)

axiom log_condition : log 16 (x - 6) = 1 / 2 

-- The primary statement we aim to prove:
theorem find_inverse_log : log 10 / log 5 = 1 / log x 5 :=
by sorry

end find_inverse_log_l90_90975


namespace hit_target_at_least_twice_l90_90843

theorem hit_target_at_least_twice :
  (∃ (p : ℝ) (n : ℕ), p = 0.6 ∧ n = 3) → ∃ (probability : ℝ), probability = 0.648 :=
by
  intro h
  obtain ⟨p, n, hp, hn⟩ := h
  use (3 * p^2 * (1 - p) + p^3)
  rw [hp, hn]
  norm_num
  assumption
  sorry

end hit_target_at_least_twice_l90_90843


namespace range_of_func_l90_90905

noncomputable def func (x : ℝ) : ℝ := sin x + sin (abs x)

theorem range_of_func : set.range func = set.Icc (0 : ℝ) 2 :=
by
  sorry

end range_of_func_l90_90905


namespace factorial_power_of_two_divisibility_l90_90723

def highestPowerOfTwoDividingFactorial (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), n / (2^k)

def binaryOnesCount (n : ℕ) : ℕ :=
  n.foldl (λ acc b, acc + if b then 1 else 0) 0

theorem factorial_power_of_two_divisibility (n : ℕ) :
  (n! % 2^(n - 1) = 0) ↔ (∃ k : ℕ, n = 2^k) :=
begin
  sorry
end

end factorial_power_of_two_divisibility_l90_90723


namespace distinct_exponents_count_l90_90159

theorem distinct_exponents_count :
  let e := (3 : ℕ)^(3^(3^(3 : ℕ)))
  let e1 := (3 : ℕ)^(3^(3^3))
  let e2 := (3 : ℕ)^((3^3)^3)
  let e3 := ((3^3)^3)^3
  let e4 := (3^(3^3))^3
  let e5 := (3^3)^(3^3)
  1 + Set.card {(e2, e4, e5) | e2 ≠ e1 ∧ e4 ≠ e1 ∧ e5 ≠ e1}.to_finset.card = 4 :=
sorry

end distinct_exponents_count_l90_90159


namespace find_ellipse_eq_and_max_area_l90_90596

-- Define the ellipse with the given conditions and eccentricity
def ellipse_eq (a b : ℝ) : Prop := ∀ (x y : ℝ), 
  (x^2 / b^2) + (y^2 / a^2) = 1

def eccentricity (e a b : ℝ) : Prop := 
  e = (Math.sqrt 2) / 2 ∧ a > b ∧ b > 0

-- Define point A
def point_on_ellipse (x y a b : ℝ) : Prop := 
  x = 1 ∧ y = Math.sqrt 2 ∧ (x^2 / b^2) + (y^2 / a^2) = 1

-- Line intersection and area conditions
def line_intersect_ellipse (m : ℝ) (a b : ℝ) : Prop :=
  m = Math.sqrt 2

def max_area_triangle (a b : ℝ) : Prop :=
  ∃ A B C : ℝ, A = 1 ∧ B = Math.sqrt 2 ∧ 
  ((A^2 / b^2) + (B^2 / a^2) = 1) ∧ 
  (1/2) * (Math.sqrt 6 / 2) * (8 - Math.sqrt 2) = Math.sqrt 2

theorem find_ellipse_eq_and_max_area 
(a b : ℝ) (e : ℝ):
  ellipse_eq a b ∧ eccentricity e a b ∧ 
  point_on_ellipse 1 (Math.sqrt 2) a b ∧ 
  line_intersect_ellipse (Math.sqrt 2) a b → 
  (∃ a b : ℝ, a = 2 ∧ b = Math.sqrt 2 ∧ ∀ (x y : ℝ), 
  (x^2 / b^2) + (y^2 / a^2) = 1) ∧
  (∃ area : ℝ, area = Math.sqrt 2) :=
by
  sorry

end find_ellipse_eq_and_max_area_l90_90596


namespace range_of_g_l90_90519

noncomputable def g (x : ℝ) : ℝ := if x = -5 then 0 else 3 * (x - 4)

theorem range_of_g :
  set.range g = {y : ℝ | y ≠ -27} :=
sorry

end range_of_g_l90_90519


namespace physics_students_count_l90_90762

theorem physics_students_count 
  (total_students : ℕ)
  (r_m : ℕ) (r_p : ℕ) (r_c : ℕ)
  (H_ratio : r_m = 6 ∧ r_p = 5 ∧ r_c = 4)
  (H_total : total_students = 135) :
  let n_m := 6 * total_students / (r_m + r_p + r_c)
  let n_p := 5 * total_students / (r_m + r_p + r_c)
  let n_c := 4 * total_students / (r_m + r_p + r_c)
in n_p = 45 :=
begin
  sorry
end

end physics_students_count_l90_90762


namespace sqrt_factorial_multiplication_squared_l90_90806

theorem sqrt_factorial_multiplication_squared :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 3))^2 = 144 := 
by
  sorry

end sqrt_factorial_multiplication_squared_l90_90806


namespace find_a1_l90_90191

variable (a : ℕ → ℝ)

-- Define the condition on the sequence
def sequence_def (n : ℕ) : Prop :=
  (n ≥ 2) → (∑ i in finset.range n, a (i + 1)) = n^3 * a n

-- The proof statement
theorem find_a1 (h1 : sequence_def a) (h2 : a 50 = 1) : a 1 = 25 := 
sorry

end find_a1_l90_90191


namespace mixed_number_calculation_l90_90305

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end mixed_number_calculation_l90_90305


namespace angle_relation_l90_90568

theorem angle_relation
  {O A B C D : Type}
  [circle O]
  (radius : ℝ)
  (BC : ℝ)
  (hBC : BC = radius)
  (secant_CD : ∃ D, ∃ C, secant_through CO intersect_circle D)
  : ∠AOD = 3 * ∠ACD :=
sorry

end angle_relation_l90_90568


namespace ellipse_with_foci_condition_l90_90167

def ellipse_condition (m n : ℝ) (h : m > n > 0) (x y : ℝ) : Prop :=
  mx^2 + ny^2 = 1 → (∃ foci : ℝ × ℝ, (ellipse_with_foci_on_y_axis foci))

theorem ellipse_with_foci_condition (m n : ℝ) (h : m > n > 0) :
  (mx^2 + ny^2 = 1) ↔ (∃ foci : ℝ × ℝ, (ellipse_with_foci_on_y_axis foci)) :=
sorry

end ellipse_with_foci_condition_l90_90167


namespace binom_10_3_eq_120_l90_90390

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90390


namespace range_of_a_l90_90982

noncomputable def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a^x₁ = x₁ ∧ a^x₂ = x₂

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : has_two_distinct_real_roots a) : 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end range_of_a_l90_90982


namespace cards_left_l90_90294

theorem cards_left (bask_boxes : ℕ) (bask_cards_per_box : ℕ) (base_boxes : ℕ) (base_cards_per_box : ℕ) (cards_given : ℕ) :
  bask_boxes = 4 → bask_cards_per_box = 10 → base_boxes = 5 → base_cards_per_box = 8 → cards_given = 58 →
  (bask_boxes * bask_cards_per_box + base_boxes * base_cards_per_box - cards_given) = 22 :=
begin
  sorry, -- proof is skipped as per the instructions
end

end cards_left_l90_90294


namespace intersect_angle_bisector_perpendicular_bisector_and_altitude_at_C_l90_90097

-- Definitions
variables {A B C L K : Point}
variable {triangle_ABC : triangle ℝ A B C}
variable {AL_bisector : is_angle_bisector A L B C}
variable {altitude_BK : is_altitude B K A C}
variable {perp_bisector_AB : is_perpendicular_bisector A B}
variable {intersection_point : ∃ P : Point, P ∈ AL_bisector ∧ P ∈ perp_bisector_AB ∧ P ∈ altitude_BK}

-- Theorem to prove
theorem intersect_angle_bisector_perpendicular_bisector_and_altitude_at_C :
  ∃ Q : Point, Q ∈ angle_bisector A L C ∧ Q ∈ perpendicular_bisector C A ∧ Q ∈ altitude C H := by
  sorry

end intersect_angle_bisector_perpendicular_bisector_and_altitude_at_C_l90_90097


namespace binomial_10_3_l90_90332

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90332


namespace max_value_quadratic_l90_90077

-- Define the ranges for k ⊕ l type band regions
def in_band_region (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

-- Declare the quadratic function f, coefficients a, b, c, and value t
variables (f : ℝ → ℝ) (a b c t : ℝ)

-- Hypotheses: Points within the 0 ⊕ 4 band region
def point_conditions := 
  in_band_region (f (-2) + 2) 0 4 ∧ 
  in_band_region (f 0 + 2) 0 4 ∧ 
  in_band_region (f 2 + 2) 0 4

-- Hypothesis: Point within the -1 ⊕ 3 band region
def t_condition := in_band_region (t + 1) (-1) 3

-- Definition of the quadratic function
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_quadratic (h1 : point_conditions f)
                            (h2 : t_condition t)
                            (ha : a = (f 2 + f (-2) - 2 * f 0) / 8)
                            (hb : b = (f 2 - f (-2)) / 4)
                            (hc : c = f 0) : 
    ∀ t, -2 ≤ t → t ≤ 2 → |quadratic f a b c t| ≤ 5 / 2 := sorry

end max_value_quadratic_l90_90077


namespace hakimi_age_l90_90773

theorem hakimi_age
  (avg_age : ℕ)
  (num_friends : ℕ)
  (molly_age : ℕ)
  (age_diff : ℕ)
  (total_age := avg_age * num_friends)
  (combined_age := total_age - molly_age)
  (jared_age := age_diff)
  (hakimi_age := combined_age - jared_age)
  (avg_age = 40)
  (num_friends = 3)
  (molly_age = 30)
  (age_diff = 10)
  (combined_age = 90)
  (hakimi_age = 40) : 
  ∃ age : ℕ, age = hakimi_age :=
by
  sorry

end hakimi_age_l90_90773


namespace sale_savings_l90_90923

theorem sale_savings (price_fox : ℝ) (price_pony : ℝ) 
(discount_fox : ℝ) (discount_pony : ℝ) 
(total_discount : ℝ) (num_fox : ℕ) (num_pony : ℕ) 
(price_saved_during_sale : ℝ) :
price_fox = 15 → 
price_pony = 18 → 
num_fox = 3 → 
num_pony = 2 → 
total_discount = 22 → 
discount_pony = 15 → 
discount_fox = total_discount - discount_pony → 
price_saved_during_sale = num_fox * price_fox * (discount_fox / 100) + num_pony * price_pony * (discount_pony / 100) →
price_saved_during_sale = 8.55 := 
by sorry

end sale_savings_l90_90923


namespace positive_divisibles_under_300_l90_90041

theorem positive_divisibles_under_300 : 
  let lcm_2_4_5 := Nat.lcm (Nat.lcm 2 4) 5 in
  lcm_2_4_5 = 20 → 
  ∃ n, n = 15 ∧ ∀ m, 1 ≤ m ∧ m < 300 → m % lcm_2_4_5 = 0 → (m / lcm_2_4_5) ≤ (300 / lcm_2_4_5) :=
by
  sorry

end positive_divisibles_under_300_l90_90041


namespace parallelogram_properties_correct_l90_90753

-- Definitions based on conditions
def is_quadrilateral (P : Type) := ∀ (a b c d : P), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a
def centrally_symmetric (P : Type) := ∀ (center : P), ∃ (A : P), ∃ (B : P), center = midpoint A B
def congruent_triangles (P : Type) := ∀ (A B C D : P), triangle A B C ≃ triangle A C D
def parallel_opposite_sides (P : Type) := ∀ (A B C D : P), parallel A B ∧ parallel C D
def supplementary_angles (P : Type) := ∀ (A B C D : P), angle A B C + angle C D A = 180

-- Parallelogram properties verification
theorem parallelogram_properties_correct : 
    (∀ (P : Type), is_quadrilateral P) ∧ 
    (∀ (P : Type), centrally_symmetric P) ∧ 
    (∀ (P : Type), congruent_triangles P) ∧ 
    ¬ (∀ (P : Type), parallel_opposite_sides P ∧ supplementary_angles P) :=
by
  sorry

end parallelogram_properties_correct_l90_90753


namespace AP_bisects_CD_l90_90569

-- Conditions definitions
variables {A B C D E P : Type*}
variables [ConvexPentagon A B C D E]  -- Assume a Convex Pentagon structure to capture the geometric shape

-- Given angles
def angle_BAC_eq_angle_CAD : angle A B C = angle C A D := sorry
def angle_CAD_eq_angle_DAE : angle C A D = angle D A E := sorry
def angle_ABC_eq_angle_ACD : angle A B C = angle A C D := sorry
def angle_ACD_eq_angle_ADE : angle A C D = angle A D E := sorry

-- Given Intersection Point
def diagonals_intersect_at_P : intersect (segment B D) (segment C E) P := sorry

-- Statement to be proved
theorem AP_bisects_CD (H1 : angle_BAC_eq_angle_CAD) (H2 : angle_CAD_eq_angle_DAE)
(H3 : angle_ABC_eq_angle_ACD) (H4 : angle_ACD_eq_angle_ADE)
(H5 : diagonals_intersect_at_P) : bisects A P (segment C D) := sorry

end AP_bisects_CD_l90_90569


namespace binomial_10_3_eq_120_l90_90422

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90422


namespace binom_10_3_eq_120_l90_90467

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90467


namespace KO_eq_VI_l90_90098

theorem KO_eq_VI
  {K I A V X O : Type*}
  (KI VI : K → I)
  (X_inside : X ∈ interior_triangle K I A)
  (H1 : KI = VI)
  (H2 : ∠XKI = ∠AVI / 2)
  (H3 : ∠XIK = ∠KVA / 2)
  (H4 : O = line_intersection (line A X) KI) :
  KO = VI :=
sorry

end KO_eq_VI_l90_90098


namespace squirrel_divides_acorns_l90_90267

theorem squirrel_divides_acorns (total_acorns parts_per_month remaining_acorns month_acorns winter_months spring_acorns : ℕ)
  (h1 : total_acorns = 210)
  (h2 : parts_per_month = 3)
  (h3 : winter_months = 3)
  (h4 : remaining_acorns = 60)
  (h5 : month_acorns = total_acorns / winter_months)
  (h6 : spring_acorns = 30)
  (h7 : month_acorns - remaining_acorns = spring_acorns / parts_per_month) :
  parts_per_month = 3 :=
by
  sorry

end squirrel_divides_acorns_l90_90267


namespace sum_of_digits_next_perfect_square_222_l90_90799

-- Define the condition for the perfect square that begins with "222"
def starts_with_222 (n: ℕ) : Prop :=
  n / 10^3 = 222

-- Define the sum of the digits function
def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Statement for the Lean 4 statement: 
-- Prove that the sum of the digits of the next perfect square that starts with "222" is 18
theorem sum_of_digits_next_perfect_square_222 : sum_of_digits (492 ^ 2) = 18 :=
by
  sorry -- Proof omitted

end sum_of_digits_next_perfect_square_222_l90_90799


namespace cyclic_quad_conditions_hold_l90_90668

variables (A B C D : ℝ)

def cyclic_quadrilateral (A B C D : ℝ) : Prop :=
  A + C = 180 ∧ B + D = 180

def condition_1 (A C : ℝ) : Prop :=
  sin A = sin C

def condition_2 (A C : ℝ) : Prop :=
  sin A + sin C = 0

def condition_3 (B D : ℝ) : Prop :=
  cos B + cos D = 0

def condition_4 (B D : ℝ) : Prop :=
  cos B = cos D

theorem cyclic_quad_conditions_hold (A B C D : ℝ) (h : cyclic_quadrilateral A B C D) :
  (condition_1 A C) ∧ (condition_3 B D) ∧ ¬(condition_2 A C) ∧ ¬(condition_4 B D) := 
by
  sorry

end cyclic_quad_conditions_hold_l90_90668


namespace greatest_distance_le_two_greatest_distance_two_exists_l90_90079

noncomputable def set_C : set ℂ := {z | z^4 = 16}
noncomputable def set_D : set ℂ := {z | z^4 - 16 * z^3 + 48 * z^2 - 64 * z + 64 = 0}

theorem greatest_distance_le_two :
  ∀ (z₁ ∈ set_C) (z₂ ∈ set_D), (dist z₁ z₂) ≤ 2 :=
begin
  sorry
end

theorem greatest_distance_two_exists :
  ∃ (z₁ ∈ set_C) (z₂ ∈ set_D), (dist z₁ z₂) = 2 :=
begin
  sorry
end

end greatest_distance_le_two_greatest_distance_two_exists_l90_90079


namespace sum_of_first_100_terms_of_sequences_l90_90125

theorem sum_of_first_100_terms_of_sequences :
  (∃ (d_a d_b : ℕ → ℤ),
    (∀ (n : ℕ), a_n = 25 + (n-1) * d_a) ∧
    (∀ (n : ℕ), b_n = 75 + (n-1) * d_b) ∧
    (25 + 99*d_a) + (75 + 99*d_b) = 100) →
  ∑ n in Finset.range 100, (a_n + b_n) = 10000 :=
by sorry

end sum_of_first_100_terms_of_sequences_l90_90125


namespace AB_eq_DC_and_BC_eq_AD_l90_90644

def Tetrahedron (A B C D : Type) : Prop :=
  -- Define the Tetrahedron with vertices A, B, C, D.

variables {A B C D : Type} [Tetrahedron A B C D]

def equal_dihedral_angles (AB DC BC AD : Type) : Prop := 
  -- Define the equality of dihedral angles
  (dihedral_angle A B = dihedral_angle D C) ∧
  (dihedral_angle B C = dihedral_angle A D)

theorem AB_eq_DC_and_BC_eq_AD
  (A B C D : Type) 
  [Tetrahedron A B C D]
  [equal_dihedral_angles AB DC BC AD] :
  (length A B = length D C) ∧ (length B C = length A D) :=
begin
  sorry -- Proof goes here
end

end AB_eq_DC_and_BC_eq_AD_l90_90644


namespace factorial_divisible_by_power_of_two_iff_l90_90726

theorem factorial_divisible_by_power_of_two_iff (n : ℕ) :
  (nat.factorial n) % (2^(n-1)) = 0 ↔ ∃ k : ℕ, n = 2^k := 
by
  sorry

end factorial_divisible_by_power_of_two_iff_l90_90726


namespace case_two_thirds_possible_case_three_fourths_impossible_case_seven_tenths_impossible_l90_90235

open Real

-- Definitions for students and questions
def m : ℕ := 3 -- number of questions
def n : ℕ := 3 -- number of students

-- Definitions for fractions
def two_thirds : ℝ := 2 / 3
def three_fourths : ℝ := 3 / 4
def seven_tenths : ℝ := 7 / 10

-- Each case can be represented as a separate theorem

-- Case 1: α = 2/3
theorem case_two_thirds_possible :
  ∃ (D : ℕ) (A : ℕ), 
  (D ≥ two_thirds * m) ∧
  (∀ (d : ℕ), d ∈ D → ∃ (s : ℕ), s ∈ n ∧ s cannot answer d) ∧
  (A ≥ two_thirds * n) ∧
  (∀ (a : ℕ), a ∈ A → a answers_at_least two_thirds * m) :=
  sorry

-- Case 2: α = 3/4
theorem case_three_fourths_impossible :
  ¬(
  ∃ (D : ℕ) (A : ℕ), 
  (D ≥ three_fourths * m) ∧
  (∀ (d : ℕ), d ∈ D → ∃ (s : ℕ), s ∈ n ∧ s cannot answer d) ∧
  (A ≥ three_fourths * n) ∧
  (∀ (a : ℕ), a ∈ A → a answers_at_least three_fourths * m)) :=
  sorry

-- Case 3: α = 7/10
theorem case_seven_tenths_impossible :
  ¬(
  ∃ (D : ℕ) (A : ℕ), 
  (D ≥ seven_tenths * m) ∧
  (∀ (d : ℕ), d ∈ D → ∃ (s : ℕ), s ∈ n ∧ s cannot answer d) ∧
  (A ≥ seven_tenths * n) ∧
  (∀ (a : ℕ), a ∈ A → a answers_at_least seven_tenths * m)) :=
  sorry

end case_two_thirds_possible_case_three_fourths_impossible_case_seven_tenths_impossible_l90_90235


namespace combination_10_3_eq_120_l90_90385

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90385


namespace total_revenue_generated_l90_90261

open Nat Real

def original_price_A := 800
def discount_A := 0.10
def desired_profit_A := 0.30

def original_price_B := 1200
def discount_B := 0.15
def desired_profit_B := 0.25

def original_price_C := 1500
def discount_C := 0.20
def desired_profit_C := 0.40

def original_price_D := 1000
def discount_D := 0.05
def desired_profit_D := 0.20

def original_price_E := 1800
def discount_E := 0.30
def desired_profit_E := 0.35

def effective_cost (original : ℝ) (discount : ℝ) : ℝ := original * (1 - discount)

def selling_price (effective_cost : ℝ) (profit: ℝ) : ℝ := effective_cost * (1 + profit)

noncomputable def total_selling_price : ℝ :=
  (selling_price (effective_cost original_price_A discount_A) desired_profit_A) +
  (selling_price (effective_cost original_price_B discount_B) desired_profit_B) +
  (selling_price (effective_cost original_price_C discount_C) desired_profit_C) +
  (selling_price (effective_cost original_price_D discount_D) desired_profit_D) +
  (selling_price (effective_cost original_price_E discount_E) desired_profit_E)

theorem total_revenue_generated : total_selling_price = 6732 :=
by
  sorry

end total_revenue_generated_l90_90261


namespace tangent_line_equation_l90_90570

-- Definitions extracted from the conditions
noncomputable def f (x : ℝ) := 2 * f (2 - x) - x^2 + 8 * x - 8

-- The theorem stating the problem
theorem tangent_line_equation : 
  tangent_equation (1, f 1) f = (2 : ℝ) * x - y - 1 := 
sorry

end tangent_line_equation_l90_90570


namespace power_six_sum_l90_90619

theorem power_six_sum (x : ℝ) (h : x + 1 / x = 3) : x^6 + 1 / x^6 = 322 := 
by 
  sorry

end power_six_sum_l90_90619


namespace excircle_tangent_perpendicular_l90_90285

theorem excircle_tangent_perpendicular 
  (O1 O2 A B C E F G H P : Point) 
  (ABC : Triangle A B C)
  (h1 : Tangent_Circle O1 ABC E G)
  (h2: Tangent_Circle O2 ABC F H)
  (h3: Line_Intersection E G F H P) : Perpendicular (Line P A) (Line B C) :=
sorry

end excircle_tangent_perpendicular_l90_90285


namespace trip_count_A_to_D_is_12_l90_90301

-- Define vertices A and D
@[derive Inhabited, derive DecidableEq]
inductive Vertex : Type
| A | B | C | D | E | F | G | H

open Vertex

-- Define an edge relationship (we ignore the exact positioning for now)
def edge : Vertex → list Vertex
| A => [B, E, H]
| B => [A, C, F]
| C => [B, D, G]
| D => [C, E, H]
| E => [A, D, F]
| F => [B, E, G]
| G => [C, F, H]
| H => [A, D, G]

-- Function to check if an edge movement is valid
def valid_move (from to : Vertex) (prev : Option Vertex) : bool :=
  match prev with
  | some p => to ≠ p ∧ to ∈ edge from
  | none => to ∈ edge from

-- Function to calculate the number of valid paths recursively
def count_paths : Vertex → Vertex → ℕ → list (Vertex × Vertex) → ℕ
| start, target, 0, _ => if start = target then 1 else 0
| start, target, n, path =>
  if n = 0 then 0
  else List.foldl
    (λ acc next => if valid_move start next (path.head? :? none) then
                      acc + count_paths next target (n - 1) ((start, next) :: path)
                    else acc)
    0
    (edge start)

-- Define the number of 4 moves paths from A to D
def number_of_4_edge_trips : ℕ :=
  count_paths A D 4 []

-- Statement of the main theorem
theorem trip_count_A_to_D_is_12 : number_of_4_edge_trips = 12 :=
  sorry

end trip_count_A_to_D_is_12_l90_90301


namespace num_units_from_batch_B_l90_90854

theorem num_units_from_batch_B
  (A B C : ℝ) -- quantities of products from batches A, B, and C
  (h_arith_seq : B - A = C - B) -- batches A, B, and C form an arithmetic sequence
  (h_total : A + B + C = 240)    -- total units from three batches
  (h_sample_size : A + B + C = 60)  -- sample size drawn equals 60
  : B = 20 := 
by {
  sorry
}

end num_units_from_batch_B_l90_90854


namespace car_and_tractor_distance_l90_90169

noncomputable theory

variables {v_c v_t dist_c dist_t: ℝ}

-- Conditions
def distance_A_B : ℝ := 160
def time_first_meeting : ℝ := 4/3
def time_car_waits : ℝ := 1
def time_after_second_departure : ℝ := 1/2

-- Speeds relationship from initial conditions
def speeds_addition (v_c v_t : ℝ) : Prop := (time_first_meeting * (v_c + v_t)) = distance_A_B

-- Distances from the first meeting
def first_meeting_distances (v_c v_t : ℝ) (d_c d_t : ℝ) : Prop :=
  (d_c = v_c * time_first_meeting) ∧ (d_t = v_t * time_first_meeting) ∧ (d_c + d_t = distance_A_B)

-- Total time for tractor from start to second meeting
def total_time_tractor : ℝ := time_first_meeting + time_car_waits + time_after_second_departure

-- Distances from stat to second meeting
def second_meeting_distances (v_c v_t d_c d_t : ℝ) : Prop :=
  d_t = v_t * total_time_tractor ∧ d_c = v_c * (time_first_meeting + time_after_second_departure)

-- Combine the relationships to find the actual distances
theorem car_and_tractor_distance (v_c v_t d_c d_t : ℝ) :
  speeds_addition v_c v_t → 
  first_meeting_distances v_c v_t d_c d_t →
  second_meeting_distances v_c v_t d_c d_t →
  d_c = 165 ∧ d_t = 85 :=
sorry

end car_and_tractor_distance_l90_90169


namespace m_is_perfect_square_l90_90973

theorem m_is_perfect_square (n : ℕ) (m : ℤ) (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1) ∧ Int.sqrt (44 * n^2 + 1) * Int.sqrt (44 * n^2 + 1) = 44 * n^2 + 1) :
  ∃ k : ℕ, m = k^2 :=
by
  sorry

end m_is_perfect_square_l90_90973


namespace nonagon_non_intersecting_diagonals_l90_90302

theorem nonagon_non_intersecting_diagonals : 
  let n := 9 in 
  (n * (n - 3)) / 2 = 27 /\ 
  ∃ d : ℕ, d = 9 := sorry

end nonagon_non_intersecting_diagonals_l90_90302


namespace students_only_in_math_l90_90288

-- Define the sets and their cardinalities according to the problem conditions
def total_students : ℕ := 120
def math_students : ℕ := 85
def foreign_language_students : ℕ := 65
def sport_students : ℕ := 50
def all_three_classes : ℕ := 10

-- Define the Lean theorem to prove the number of students taking only a math class
theorem students_only_in_math (total : ℕ) (M F S : ℕ) (MFS : ℕ)
  (H_total : total = 120)
  (H_M : M = 85)
  (H_F : F = 65)
  (H_S : S = 50)
  (H_MFS : MFS = 10) :
  (M - (MFS + MFS - MFS) = 35) :=
sorry

end students_only_in_math_l90_90288


namespace cos_two_sum_l90_90935

theorem cos_two_sum {α β : ℝ} 
  (h1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h2 : 3 * (Real.sin α + Real.cos α) ^ 2 - 2 * (Real.sin β + Real.cos β) ^ 2 = 1) :
  Real.cos (2 * (α + β)) = -1 / 3 :=
sorry

end cos_two_sum_l90_90935


namespace find_other_root_and_m_l90_90004

noncomputable theory
open Polynomial

theorem find_other_root_and_m {m : ℝ} (h : (2:ℝ)^2 + 2*(2:ℝ) + 3*m - 4 = 0) :
  let f := X^2 + 2*X + (3*algebraMap ℝ (Polynomial ℝ) m - 4) in
  ∃ x₁ x₂, (x₁ = 2 ∧ (root_of f x₂) ∧ x₁ + x₂ = -(-2 : ℝ)) ∧ m = -4/3 := 
  by
    sorry

end find_other_root_and_m_l90_90004


namespace binom_10_3_l90_90341

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90341


namespace boat_upstream_distance_l90_90244

/--
Conditions:
1. The boat goes 100 km downstream in 10 hours.
2. The speed of the stream is 3 km/h.
3. The boat goes some distance upstream in 15 hours.

Goal: Prove that the boat went 60 km upstream.
-/
theorem boat_upstream_distance (V_b V_s V_downstream V_upstream : ℝ) (V_s_def : V_s = 3) 
    (V_downstream_def : V_downstream = V_b + V_s)
    (V_upstream_def : V_upstream = V_b - V_s)
    (downstream_distance : 100 = V_downstream * 10)
    (upstream_time : 15) :
    V_upstream * upstream_time = 60 := 
sorry

end boat_upstream_distance_l90_90244


namespace angle_BMC_in_isosceles_triangle_l90_90071

theorem angle_BMC_in_isosceles_triangle
    (A B C M : Type*)
    [angle_ABC : ∠(B, A, C) = 80]
    [AB_eq_BC : AB = BC]
    [M_inside_tri : Point.in_triangle M A B C]
    [angle_MAC : ∠(M, A, C) = 30]
    [angle_MCA : ∠(M, C, A) = 10] :
    ∠(B, M, C) = 70 :=
sorry

end angle_BMC_in_isosceles_triangle_l90_90071


namespace total_floor_area_covered_l90_90209

theorem total_floor_area_covered (combined_area : ℝ) (area_two_layers : ℝ) (area_three_layers : ℝ) : 
  combined_area = 200 → 
  area_two_layers = 22 → 
  area_three_layers = 19 → 
  (combined_area - (area_two_layers + 2 * area_three_layers)) = 140 := 
by
  sorry

end total_floor_area_covered_l90_90209


namespace constant_ratio_of_arithmetic_sequence_l90_90998

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ a₁ d : ℝ, ∀ n : ℕ, a n = a₁ + (n-1) * d

-- The main theorem stating the result
theorem constant_ratio_of_arithmetic_sequence 
  (a : ℕ → ℝ) (c : ℝ) (h_seq : arithmetic_sequence a)
  (h_const : ∀ n : ℕ, a n ≠ 0 ∧ a (2 * n) ≠ 0 ∧ a n / a (2 * n) = c) :
  c = 1 ∨ c = 1 / 2 :=
sorry

end constant_ratio_of_arithmetic_sequence_l90_90998


namespace tetrahedrons_volume_proportional_l90_90147

-- Define the scenario and conditions.
variable 
  (V V' : ℝ) -- Volumes of the tetrahedrons
  (a b c a' b' c' : ℝ) -- Edge lengths emanating from vertices O and O'
  (α : ℝ) -- The angle between vectors OB and OC which is assumed to be congruent

-- Theorem statement.
theorem tetrahedrons_volume_proportional
  (congruent_trihedral_angles_at_O_and_O' : α = α) -- Condition of congruent trihedral angles
  : (V' / V) = (a' * b' * c') / (a * b * c) :=
sorry

end tetrahedrons_volume_proportional_l90_90147


namespace bus_driver_hours_worked_l90_90234

-- Definitions based on the problem's conditions.
def regular_rate : ℕ := 20
def regular_hours : ℕ := 40
def overtime_rate : ℕ := regular_rate + (3 * (regular_rate / 4))  -- 75% higher
def total_compensation : ℕ := 1000

-- Theorem statement: The bus driver worked a total of 45 hours last week.
theorem bus_driver_hours_worked : 40 + ((total_compensation - (regular_rate * regular_hours)) / overtime_rate) = 45 := 
by 
  sorry

end bus_driver_hours_worked_l90_90234


namespace initial_money_l90_90109

def lunch_cost : ℕ := 5
def brother_cost : ℕ := 2
def current_money : ℕ := 15

theorem initial_money (lunch_cost brother_cost current_money : ℕ) (h1 : lunch_cost = 5) (h2 : brother_cost = 2) (h3 : current_money = 15) :
  current_money + lunch_cost + brother_cost = 22 :=
begin
  sorry
end

end initial_money_l90_90109


namespace binomial_10_3_eq_120_l90_90427

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90427


namespace ac_lt_bd_l90_90976

theorem ac_lt_bd (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) : a * c < b * d :=
by
  sorry

end ac_lt_bd_l90_90976


namespace polar_coordinate_transformation_l90_90638

theorem polar_coordinate_transformation (r θ : ℝ) (h₀ : r = -5) (h₁ : θ = π / 3) :
  ∃ r' θ', r' = 5 ∧ θ' = 4 * π / 3 ∧ 0 ≤ θ' ∧ θ' < 2 * π :=
by
  use 5, 4 * π / 3
  simp [h₀, h₁]
  split
  exact rfl
  split
  exact rfl
  split
  linarith [Real.pi_pos]
  linarith [Real.pi_pos]
  sorry

end polar_coordinate_transformation_l90_90638


namespace vasya_total_time_on_escalator_upwards_is_324_sec_l90_90615

noncomputable def vasya_speed_without_escalator (t_down t_up: ℝ) (total: ℝ) : ℝ :=
if h : 0 < t_up ∧ 0 < t_down then
  let x := 2 * t_up in
  if 3 / x = total then x else 0
else 0

noncomputable def solve_escalator_moving_down (x: ℝ) (total_down: ℝ) : ℝ :=
if h : x > 0 then
  let k := x + (x / 2) in
  if 27 * total_down = 27 * k - 216 * (x / 2) then (x + 1 / k) / 54 - x / 4 else 0
else 0

noncomputable def total_time_up_and_down_with_escalator_moving_up 
  (x y: ℝ) (escalator_up_time total_time: ℝ) : ℝ :=
if h : 0 < y ∧ x > y then
  let up_time := 1 / (x - y) in
  let down_time := 1 / (x / 2 + y) in
  if up_time + down_time = total_time then (up_time + down_time) * 60 else -1
else -1

theorem vasya_total_time_on_escalator_upwards_is_324_sec (x y t_up_t_down total_down: ℝ) :
  x = vasya_speed_without_escalator x x t_up_t_down →
  y = solve_escalator_moving_down x total_down →
  total_time_up_and_down_with_escalator_moving_up x y 5.3374 324 :=
by {sorry}

end vasya_total_time_on_escalator_upwards_is_324_sec_l90_90615


namespace f_monotonically_increasing_interval_f_max_min_values_l90_90599

noncomputable def f (x : ℝ) : ℝ := sin x ^ 2 + 2 * sin x * cos x + 3 * cos x ^ 2

-- Proving part (I)
theorem f_monotonically_increasing_interval (k : ℤ) :
  ∀ x, k * π - (3 * π / 8) ≤ x ∧ x ≤ k * π + (π / 8) → ∃ d, 0 ≤ d ∧ f (x + d) - f x ≥ 0 := sorry

-- Proving part (II)
theorem f_max_min_values :
  ∃ x_max x_min : ℝ, 
    (0 ≤ x_max ∧ x_max ≤ π / 2 ∧ f x_max = sqrt 2 + 2) ∧ 
    (0 ≤ x_min ∧ x_min ≤ π / 2 ∧ f x_min = 2 - sqrt 2) := sorry

end f_monotonically_increasing_interval_f_max_min_values_l90_90599


namespace add_candies_to_equalize_l90_90204

-- Define the initial number of candies in basket A and basket B
def candiesInA : ℕ := 8
def candiesInB : ℕ := 17

-- Problem statement: Prove that adding 9 more candies to basket A
-- makes the number of candies in basket A equal to that in basket B.
theorem add_candies_to_equalize : ∃ n : ℕ, candiesInA + n = candiesInB :=
by
  use 9  -- The value we are adding to the candies in basket A
  sorry  -- Proof goes here

end add_candies_to_equalize_l90_90204


namespace combination_10_3_eq_120_l90_90489

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90489


namespace train_should_travel_at_84_kmh_l90_90271

def required_speed_to_arrive_on_time (d : ℝ) (t : ℝ) (v1 : ℝ) (v2 : ℝ) (late_time : ℝ) (early_time : ℝ) : Prop :=
  v1 * (t + late_time) = d ∧
  v2 * (t - early_time) = d →
  d / t = 84

theorem train_should_travel_at_84_kmh (d : ℝ) (t : ℝ) :
  required_speed_to_arrive_on_time d t 80 90 (24 / 60) (32 / 60) :=
begin
  sorry
end

end train_should_travel_at_84_kmh_l90_90271


namespace inclination_angle_for_minimal_time_l90_90840

theorem inclination_angle_for_minimal_time (g x: ℝ) (hpos_g: 0 < g) (hpos_x: 0 < x) :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ α = π / 4 :=
by
  use π/4
  split
  · exact real.pi_pos.div_pos (by norm_num : 0 < 4)
  split
  · linarith
  · norm_num
  sorry

end inclination_angle_for_minimal_time_l90_90840


namespace seventh_oblong_number_l90_90864

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end seventh_oblong_number_l90_90864


namespace ab_eq_ac_l90_90645

noncomputable def triangle (α β γ : Type) := (α, β, γ)
noncomputable def area_eq {α β γ : Type} (t : triangle α β γ) := 7.8
noncomputable def extends_d_e {α β γ : Type} (t : triangle α β γ) (d e : Type) := ∃ d e, true
noncomputable def bd_eq_ce {α β γ d e p : Type} (t : triangle α β γ) (d e : Type) := true
noncomputable def angle_eq_cond {α β γ d e p k : Type} (t : triangle α β γ) (d e : Type) := ∃ k, true

theorem ab_eq_ac (α β γ d e p k : Type) (t : triangle α β γ) 
  (areaCond : area_eq t)
  (extCond : extends_d_e t d e)
  (bdceCond : bd_eq_ce t d e)
  (angleCond : angle_eq_cond t d e) : 
  true :=
sorry

end ab_eq_ac_l90_90645


namespace floor_inequality_l90_90709

theorem floor_inequality (α β : ℝ) : 
  int.floor (2 * α) + int.floor (2 * β) ≥ int.floor α + int.floor β + int.floor (α + β) := 
by
  sorry

end floor_inequality_l90_90709


namespace binomial_10_3_l90_90504

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90504


namespace kayaks_built_by_end_of_april_l90_90874

theorem kayaks_built_by_end_of_april :
  let feb_kayaks := 5 in
  let mar_kayaks := 3 * feb_kayaks in
  let apr_kayaks := 3 * mar_kayaks in
  feb_kayaks + mar_kayaks + apr_kayaks = 65 :=
by
  let feb_kayaks := 5
  let mar_kayaks := 3 * feb_kayaks
  let apr_kayaks := 3 * mar_kayaks
  have h : feb_kayaks + mar_kayaks + apr_kayaks = 5 + 3 * 5 + 3 * (3 * 5) := rfl
  have : 5 + 3 * 5 + 3 * (3 * 5) = 65 := sorry
  exact sorry

end kayaks_built_by_end_of_april_l90_90874


namespace combination_10_3_l90_90402

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90402


namespace perpendicular_passes_through_midpoint_of_PQ_l90_90070

variables 
  (A B C H A1 B1 C1 P Q: Type) 
  [Inhabited A] [Inhabited B] [Inhabited C]

-- Assume the basic conditions of the problem
variables 
  (acute_angled_triangle : ∀ (A B C : Type), Prop) 
  (altitude_intersection_H : ∀ (A A1 B B1 C C1 H : Type), Prop)
  (perpendicular_from_H_intersect_rays : ∀ (H B1 C1 A1 P Q : Type), Prop)

-- Define the main theorem
theorem perpendicular_passes_through_midpoint_of_PQ 
  (h1 : acute_angled_triangle A B C)
  (h2 : altitude_intersection_H A A1 B B1 C C1 H)
  (h3 : perpendicular_from_H_intersect_rays H B1 C1 A1 P Q) :
  ∃ M, midpoint P Q M ∧ perpendicular C A1 B1 M :=
sorry

end perpendicular_passes_through_midpoint_of_PQ_l90_90070


namespace problem_one_problem_two_l90_90628

variables {A B C a b c : ℝ}
variables (triangle : Type) [triangle ABC] (S : ℝ)

open_locale real

-- Given conditions
def given_conditions (h1 : (cos A / a + cos C / c = 1 / b))
  (h2 : b = 2) (h3 : a > c) (h4 : S = sqrt 7 / 2) : Prop :=
  h1 ∧ h2 ∧ h3 ∧ h4

-- Proving that ac = 4
theorem problem_one (h1 : cos A / a + cos C / c = 1 / b)
  (h2 : b = 2) : ac = 4 := sorry

-- Proving the values of a and c given the conditions
theorem problem_two (h1 : cos A / a + cos C / c = 1 / b)
  (h2 : b = 2) (h3 : a > c) (h4 : S = sqrt 7 / 2) :
  a = 2 * sqrt 2 ∧ c = sqrt 2 := sorry

end problem_one_problem_two_l90_90628


namespace solve_system_l90_90618

variable (x y z : ℝ)

theorem solve_system :
  (y + z = 20 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 10 - 4 * z) →
  (2 * x + 2 * y + 2 * z = 4) :=
by
  intros h1 h2 h3
  sorry

end solve_system_l90_90618


namespace parts_per_hour_l90_90770

theorem parts_per_hour (x y : ℝ) (h₁ : 90 / x = 120 / y) (h₂ : x + y = 35) : x = 15 ∧ y = 20 :=
by
  sorry

end parts_per_hour_l90_90770


namespace micheal_work_separately_40_days_l90_90699

-- Definitions based on the problem conditions
def work_complete_together (M A : ℕ) : Prop := (1/(M:ℝ) + 1/(A:ℝ) = 1/20)
def remaining_work_completed_by_adam (A : ℕ) : Prop := (1/(A:ℝ) = 1/40)

-- The theorem we want to prove
theorem micheal_work_separately_40_days (M A : ℕ) 
  (h1 : work_complete_together M A) 
  (h2 : remaining_work_completed_by_adam A) : 
  M = 40 := 
by 
  sorry  -- Placeholder for proof

end micheal_work_separately_40_days_l90_90699


namespace min_distance_l90_90028

noncomputable def minimum_distance (l : ℝ → ℝ → Prop) (M : ℝ → ℝ → Prop) : ℝ :=
let distance := λ x y a b : ℝ, abs ((a - x) + (b - y)) / sqrt 2 in
-- Center of circle C(1, -1)
let C := (1, -1) in
let center_dist := distance C.1 C.2 0 8 in
center_dist - sqrt 2

-- Problem conditions in Lean
def line_l (x y : ℝ) : Prop := x - y = 8
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- The theorem to prove
theorem min_distance:
  let l := line_l in let M := circle_M in 
    minimum_distance l M = 2 * sqrt 2 :=
by
  sorry

end min_distance_l90_90028


namespace proof_aid_l90_90820

noncomputable def problem_statement (A B C D E Ω : Type) [Circumcircle A B C D Ω] 
  (AB CD AD BC AE DE : ℝ) : Prop :=
  AB = CD ∧
  AD ≠ BC ∧
  AD > BC ∧
  (AD ∥ BC) ∧
  OnCircumcircle Ω A B C D E ∧ 
  Perpendicular E B AD ∧
  (AE + BC > DE)

theorem proof_aid (A B C D E Ω : Type) 
  (AB CD AD BC AE DE : ℝ) (h1 : AB = CD) (h2 : AD ≠ BC) (h3 : AD > BC) 
  (h4 : AD ∥ BC) 
  (h5 : OnCircumcircle Ω A B C D E) 
  (h6 : Perpendicular E B AD) :
  AE + BC > DE := by 
  sorry

end proof_aid_l90_90820


namespace price_of_soda_l90_90215

-- Definitions based on the conditions given in the problem
def initial_amount := 500
def cost_rice := 2 * 20
def cost_wheat_flour := 3 * 25
def remaining_balance := 235
def total_cost := cost_rice + cost_wheat_flour

-- Definition to be proved
theorem price_of_soda : initial_amount - total_cost - remaining_balance = 150 := by
  sorry

end price_of_soda_l90_90215


namespace jia_catches_up_after_full_laps_l90_90790

def length_of_track := 300
def speed_jia := 5
def speed_yi := 4.2

theorem jia_catches_up_after_full_laps :
  let relative_speed := speed_jia - speed_yi in
  let catch_up_time := length_of_track / relative_speed in
  let distance_jia_runs := speed_jia * catch_up_time in
  let number_of_laps := distance_jia_runs / length_of_track in
  number_of_laps = 6 :=
by
  -- Here we would provide the detailed proof steps
  sorry

end jia_catches_up_after_full_laps_l90_90790


namespace num_groups_of_consecutive_natural_numbers_l90_90032

theorem num_groups_of_consecutive_natural_numbers (n : ℕ) (h : 3 * n + 3 < 19) : n < 6 := 
  sorry

end num_groups_of_consecutive_natural_numbers_l90_90032


namespace hawkeye_fewer_mainecoons_than_gordon_l90_90658

-- Definitions based on conditions
def JamiePersians : ℕ := 4
def JamieMaineCoons : ℕ := 2
def GordonPersians : ℕ := JamiePersians / 2
def GordonMaineCoons : ℕ := JamieMaineCoons + 1
def TotalCats : ℕ := 13
def JamieTotalCats : ℕ := JamiePersians + JamieMaineCoons
def GordonTotalCats : ℕ := GordonPersians + GordonMaineCoons
def JamieAndGordonTotalCats : ℕ := JamieTotalCats + GordonTotalCats
def HawkeyeTotalCats : ℕ := TotalCats - JamieAndGordonTotalCats
def HawkeyePersians : ℕ := 0
def HawkeyeMaineCoons : ℕ := HawkeyeTotalCats - HawkeyePersians

-- Theorem statement to prove: Hawkeye owns 1 fewer Maine Coon than Gordon
theorem hawkeye_fewer_mainecoons_than_gordon : HawkeyeMaineCoons + 1 = GordonMaineCoons :=
by
  sorry

end hawkeye_fewer_mainecoons_than_gordon_l90_90658


namespace correct_factorization_l90_90808

theorem correct_factorization (a : ℝ) : 
  (a ^ 2 + 4 * a ≠ a ^ 2 * (a + 4)) ∧ 
  (a ^ 2 - 9 ≠ (a + 9) * (a - 9)) ∧ 
  (a ^ 2 + 4 * a + 2 ≠ (a + 2) ^ 2) → 
  (a ^ 2 - 2 * a + 1 = (a - 1) ^ 2) :=
by sorry

end correct_factorization_l90_90808


namespace remainder_polynomial_division_l90_90907

noncomputable def remainder_division : Polynomial ℝ := 
  (Polynomial.X ^ 4 + Polynomial.X ^ 3 - 4 * Polynomial.X + 1) % (Polynomial.X ^ 3 - 1)

theorem remainder_polynomial_division :
  remainder_division = -3 * Polynomial.X + 2 :=
by
  sorry

end remainder_polynomial_division_l90_90907


namespace binomial_coefficient_10_3_l90_90364

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90364


namespace binom_10_3_l90_90436

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90436


namespace sum_prime_factors_1260_l90_90802

theorem sum_prime_factors_1260 : 
  let factors := {2, 3, 5, 7} in 
  (Finset.min' factors (by decide) + Finset.max' factors (by decide) = 9) := by
  sorry

end sum_prime_factors_1260_l90_90802


namespace modulus_of_z_eq_l90_90981

noncomputable def z (i : ℂ) : ℂ := (1 - 2 * i^3) / (1 + i)

theorem modulus_of_z_eq (i : ℂ) (hi : i * i = -1) : 
  abs (z i) = real.sqrt 10 / 2 :=
by
  sorry

end modulus_of_z_eq_l90_90981


namespace storm_function_example_1_storm_function_example_2_storm_function_range_b_l90_90256

def storm_function (f : ℝ → ℝ) (U : set ℝ) := 
  ∀ x1 x2 ∈ U, |f x1 - f x2| < 1

open set

-- Proof for the first question
theorem storm_function_example_1 : storm_function (λ x, 2^(x - 1) + 1) (Icc (-1) 1) := 
  sorry

-- Proof for the second question
theorem storm_function_example_2 : storm_function (λ x, 1/2 * x^2 + 1) (Icc (-1) 1) := 
  sorry

-- Proof for the third question
theorem storm_function_range_b : 
  ∀ b, storm_function (λ x, 1/2 * x^2 - b * x + 1) (Icc (-1) 1) ↔ (1 - Real.sqrt 2 < b ∧ b < Real.sqrt 2 - 1) := 
  sorry

end storm_function_example_1_storm_function_example_2_storm_function_range_b_l90_90256


namespace integers_with_5_6_between_700_1500_l90_90972

theorem integers_with_5_6_between_700_1500 : 
  ∃ (n : ℕ), n = 12 ∧ (∀ (x : ℕ), x ∈ set.Icc 700 1500 → 
  x.digits.includes 5 ∧ x.digits.includes 6 → x = 12) := 
by sorry

end integers_with_5_6_between_700_1500_l90_90972


namespace trajectory_equation_l90_90198

-- Definitions and conditions
noncomputable def tangent_to_x_axis (M : ℝ × ℝ) := M.snd = 0
noncomputable def internally_tangent (M : ℝ × ℝ) := ∃ (r : ℝ), 0 < r ∧ M.1^2 + (M.2 - r)^2 = 4

-- The theorem stating the proof problem
theorem trajectory_equation (M : ℝ × ℝ) (h_tangent : tangent_to_x_axis M) (h_internal_tangent : internally_tangent M) :
  (∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧ M.fst^2 = 4 * (y - 1)) :=
sorry

end trajectory_equation_l90_90198


namespace combination_10_3_eq_120_l90_90486

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90486


namespace find_λ_value_l90_90987

noncomputable def λ_value (AB AC : ℝ) (angle_A : ℝ) (AP : ℝ → ℝ × ℝ) (BP CP : ℝ × ℝ → ℝ) : Prop :=
  AB = 1 ∧ AC = 2 ∧ angle_A = 60 ∧ 
  (∀ λ, AP λ = (1, 0) + λ * (0, 2)) ∧ 
  (∀ λ, BP (AP λ) = λ * (0, 2)) ∧
  (∀ λ, CP (AP λ) = (1, 0) + (λ - 1) * (0, 2)) ∧
  (∀ λ, BP (AP λ) ⋅ CP (AP λ) = 1) →
  (λ = -1 / 4 ∨ λ = 1)

theorem find_λ_value (AB AC : ℝ) (angle_A : ℝ) (AP : ℝ → ℝ × ℝ) (BP CP : ℝ × ℝ → ℝ) :
  λ_value AB AC angle_A AP BP CP :=
begin
  sorry
end

end find_λ_value_l90_90987


namespace inverse_of_B_cubed_l90_90947

theorem inverse_of_B_cubed (B_inv : Matrix (Fin 2) (Fin 2) ℤ) (hB_inv : B_inv = !![3, 4; -2, -2]) :
    inverse ((B_inv⁻¹) ^ 3) = !![3, 4; -6, -28] :=
by
  sorry

end inverse_of_B_cubed_l90_90947


namespace esperanzas_gross_monthly_salary_l90_90140

variables (Rent FoodExpenses MortgageBill Savings Taxes GrossSalary : ℝ)

def problem_conditions (Rent FoodExpenses MortgageBill Savings Taxes : ℝ) :=
  Rent = 600 ∧
  FoodExpenses = (3 / 5) * Rent ∧
  MortgageBill = 3 * FoodExpenses ∧
  Savings = 2000 ∧
  Taxes = (2 / 5) * Savings

theorem esperanzas_gross_monthly_salary (h : problem_conditions Rent FoodExpenses MortgageBill Savings Taxes) :
  GrossSalary = Rent + FoodExpenses + MortgageBill + Taxes + Savings → GrossSalary = 4840 :=
by
  sorry

end esperanzas_gross_monthly_salary_l90_90140


namespace proportion_terms_l90_90069

theorem proportion_terms (x v y z : ℤ) (a b c : ℤ)
  (h1 : x + v = y + z + a)
  (h2 : x^2 + v^2 = y^2 + z^2 + b)
  (h3 : x^4 + v^4 = y^4 + z^4 + c)
  (ha : a = 7) (hb : b = 21) (hc : c = 2625) :
  (x = -3 ∧ v = 8 ∧ y = -6 ∧ z = 4) :=
by
  sorry

end proportion_terms_l90_90069


namespace exists_n_for_A_of_non_perfect_square_l90_90713

theorem exists_n_for_A_of_non_perfect_square (A : ℕ) (h : ∀ k : ℕ, k^2 ≠ A) :
  ∃ n : ℕ, A = ⌊ n + Real.sqrt n + 1/2 ⌋ :=
sorry

end exists_n_for_A_of_non_perfect_square_l90_90713


namespace binom_10_3_l90_90352

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90352


namespace smallest_multiple_of_6_8_12_l90_90917

theorem smallest_multiple_of_6_8_12 : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 8 = 0 ∧ n % 12 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 6 = 0 ∧ m % 8 = 0 ∧ m % 12 = 0) → n ≤ m := 
sorry

end smallest_multiple_of_6_8_12_l90_90917


namespace P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l90_90919

-- Assume the definition of sum of digits of n and count of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum  -- Sum of digits in base 10 representation

def num_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).length  -- Number of digits in base 10 representation

def P (n : ℕ) : ℕ :=
  sum_of_digits n + num_of_digits n

-- Problem (a)
theorem P_2017 : P 2017 = 14 :=
sorry

-- Problem (b)
theorem P_eq_4 :
  {n : ℕ | P n = 4} = {3, 11, 20, 100} :=
sorry

-- Problem (c)
theorem exists_P_minus_P_succ_gt_50 : 
  ∃ n : ℕ, P n - P (n + 1) > 50 :=
sorry

end P_2017_P_eq_4_exists_P_minus_P_succ_gt_50_l90_90919


namespace three_brothers_pizza_slices_l90_90873

theorem three_brothers_pizza_slices :
  let large_pizza_slices := 14
  let small_pizza_slices := 8
  let num_brothers := 3
  let total_slices := small_pizza_slices + 2 * large_pizza_slices
  total_slices / num_brothers = 12 := by
  sorry

end three_brothers_pizza_slices_l90_90873


namespace fixed_point_of_logarithmic_function_l90_90754

theorem fixed_point_of_logarithmic_function 
    (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
    ∀ x y : ℝ, y = log a (x + 1) + 2 → (0, 2) = (0, log a 1 + 2) :=
by
  intros x y h
  rw [h, log a 1]
  exact (0, 2) 
sorry

end fixed_point_of_logarithmic_function_l90_90754


namespace cosine_of_angle_between_diagonals_is_zero_l90_90894

noncomputable theory -- Use noncomputable due to the sqrt function

-- Definitions of the vectors u and v
def u : ℝ^3 := ![3, 2, 1]
def v : ℝ^3 := ![1, 3, 2]

-- Definitions of the diagonals
def diagonal1 : ℝ^3 := u + v
def diagonal2 : ℝ^3 := v - u

-- The proof problem statement
theorem cosine_of_angle_between_diagonals_is_zero :
  let dot_product := (diagonal1 ⬝ diagonal2)
  let norm_diagonal1 := (∥diagonal1∥)
  let norm_diagonal2 := (∥diagonal2∥)
  dot_product = 0 → (norm_diagonal1 * norm_diagonal2) ≠ 0 → (dot_product / (norm_diagonal1 * norm_diagonal2)) = 0 :=
begin
  -- Definitions
  let u := ![3, 2, 1],
  let v := ![1, 3, 2],
  let diagonal1 := u + v,
  let diagonal2 := v - u,
  let dot_product := (diagonal1 ⬝ diagonal2),
  let norm_diagonal1 := (∥diagonal1∥),
  let norm_diagonal2 := (∥diagonal2∥),
  
  -- To be proved
  assume h1 : dot_product = 0,
  assume h2 : (norm_diagonal1 * norm_diagonal2) ≠ 0,
  show (dot_product / (norm_diagonal1 * norm_diagonal2)) = 0, by {
    rewrite h1,
    simp,
    exact absurd rfl h2,
  }
sorry

end cosine_of_angle_between_diagonals_is_zero_l90_90894


namespace rogue_trader_aggregate_value_l90_90850

theorem rogue_trader_aggregate_value :
  let spice := 5 * 7^3 + 2 * 7^2 + 1 * 7^1 + 3 * 7^0,
      metals := 1 * 7^3 + 6 * 7^2 + 5 * 7^1 + 3 * 7^0,
      fruits := 2 * 7^2 + 0 * 7^1 + 2 * 7^0,
      total_value := spice + metals + fruits
  in total_value = 2598 :=
by
  sorry

end rogue_trader_aggregate_value_l90_90850


namespace day_of_week_306_2003_l90_90621

-- Note: Definitions to support the conditions and the proof
def day_of_week (n : ℕ) : ℕ := n % 7

-- Theorem statement: Given conditions lead to the conclusion that the 306th day of the year 2003 falls on a Sunday
theorem day_of_week_306_2003 :
  (day_of_week (15) = 2) → (day_of_week (306) = 0) :=
by sorry

end day_of_week_306_2003_l90_90621


namespace dilation_takes_point_to_expected_result_l90_90168

-- Conditions
def center := Complex.mk 1 2  -- center at 1 + 2i
def scale_factor := 2        -- scale factor of 2
def point := Complex.mk 0 1  -- point 0 + i

-- The expected result of the dilation
def expected_result := Complex.mk (-1) 0  -- Expected result -1 + 0i

-- Proof problem: Prove the dilation takes 'point' to 'expected_result'
theorem dilation_takes_point_to_expected_result : 
  let z := scale_factor * (point - center) + center
  in z = expected_result :=
by
  -- Proof will be filled here
  sorry

end dilation_takes_point_to_expected_result_l90_90168


namespace program_output_is_1023_l90_90221

-- Definition placeholder for program output.
def program_output : ℕ := 1023

-- Theorem stating the program's output.
theorem program_output_is_1023 : program_output = 1023 := 
by 
  -- Proof details are omitted.
  sorry

end program_output_is_1023_l90_90221


namespace longest_segment_is_sum_of_others_l90_90284

-- Definitions
variables (A B C D E F P : Point)
variable (triangleABC : EquilateralTriangle A B C)
variable (altitudeAD : IsAltitude D A B C)
variable (altitudeBE : IsAltitude E A B C)
variable (altitudeCF : IsAltitude F A B C)
variable (PD : PerpendicularDistance P D)
variable (PE : PerpendicularDistance P E)
variable (PF : PerpendicularDistance P F)

-- Theorem
theorem longest_segment_is_sum_of_others (h : ∀ (P : Point),
  let PD := perpendicular_distance P D
  let PE := perpendicular_distance P E
  let PF := perpendicular_distance P F
  in max PD PE PF = PD + PE ∨ max PD PE PF = PD + PF ∨ max PD PE PF = PE + PF) :
  ∀ (P : Point), max (perpendicular_distance P D) (perpendicular_distance P E) (perpendicular_distance P F) =
                  (perpendicular_distance P D + perpendicular_distance P E) ∨
                  (perpendicular_distance P D + perpendicular_distance P F) ∨
                  (perpendicular_distance P E + perpendicular_distance P F) :=
begin
  sorry
end

end longest_segment_is_sum_of_others_l90_90284


namespace bus_final_velocity_l90_90249

noncomputable def final_velocity 
  (u1 : ℝ) (a1 : ℝ) (s1 : ℝ) 
  (a2 : ℝ) (s2 : ℝ) 
  (a3 : ℝ) (s3 : ℝ) : ℝ :=
  let v1 := (u1^2 + 2 * a1 * s1).sqrt
  let v2 := (v1^2 + 2 * a2 * s2).sqrt
  (v2^2 + 2 * a3 * s3).sqrt

theorem bus_final_velocity :
  final_velocity 0 2 50 3 80 4 70 ≈ 35.21 :=
by
  sorry

end bus_final_velocity_l90_90249


namespace binomial_10_3_l90_90331

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90331


namespace binom_10_3_l90_90437

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90437


namespace binom_10_3_eq_120_l90_90315

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90315


namespace part_a_l90_90818

theorem part_a (P : set (ℝ × ℝ)) (hP : convex P) (h_area : measure_theory.measure P = 1) :
    ∃ T : set (ℝ × ℝ), (T ⊆ P) ∧ is_triangle T ∧ measure_theory.measure T ≥ 1 / 4 :=
sorry

end part_a_l90_90818


namespace positive_difference_of_squares_and_product_l90_90196

theorem positive_difference_of_squares_and_product (x y : ℕ) 
  (h1 : x + y = 60) (h2 : x - y = 16) :
  x^2 - y^2 = 960 ∧ x * y = 836 :=
by sorry

end positive_difference_of_squares_and_product_l90_90196


namespace length_of_AB_in_45_45_90_triangle_l90_90646

theorem length_of_AB_in_45_45_90_triangle 
  (A B C : Type) 
  [Triangle A B C] 
  (h_right : is_right_angle A B C) 
  (h_45_45_90 : is_45_45_90 A B C) 
  (h_BC : BC = 6 * Real.sqrt 2) : 
  AB = 6 := 
sorry

end length_of_AB_in_45_45_90_triangle_l90_90646


namespace binom_10_3_eq_120_l90_90386

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90386


namespace initial_investment_calculation_l90_90260

theorem initial_investment_calculation
  (x : ℝ)  -- initial investment at 5% per annum
  (h₁ : x * 0.05 + 4000 * 0.08 = (x + 4000) * 0.06) :
  x = 8000 :=
by
  -- skip the proof
  sorry

end initial_investment_calculation_l90_90260


namespace S21_sum_is_4641_l90_90038

-- Define the conditions and the sum of the nth group
def first_number_in_group (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 + 1

def last_number_in_group (n : ℕ) : ℕ :=
  first_number_in_group n + (n - 1)

def sum_of_group (n : ℕ) : ℕ :=
  n * (first_number_in_group n + last_number_in_group n) / 2

-- The theorem to prove
theorem S21_sum_is_4641 : sum_of_group 21 = 4641 := by
  sorry

end S21_sum_is_4641_l90_90038


namespace binom_10_3_l90_90441

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90441


namespace tan_alpha_plus_pi_div_2_eq_one_half_l90_90958

theorem tan_alpha_plus_pi_div_2_eq_one_half (x y : ℝ) (h : x = -1 ∧ y = 2) :
  let α := real.atan2 y x in
  real.tan (α + real.pi / 2) = 1 / 2 :=
by
  obtain ⟨hx, hy⟩ := h
  have h_tan : real.tan α = y / x := by sorry
  have h_cot : real.cot α = 1 / real.tan α := by sorry
  have h_identity : real.tan (α + real.pi / 2) = -real.cot α := by sorry
  rw [h_identity, h_cot, h_tan, hx, hy]
  norm_num
  sorry

end tan_alpha_plus_pi_div_2_eq_one_half_l90_90958


namespace John_last_year_salary_l90_90660

theorem John_last_year_salary (S B : ℝ) 
  (bonus_last_year : S * B = 10000) 
  (salary_this_year : 200000) 
  (total_pay_this_year : 220000) 
  (bonus_percentage :  (200000:ℝ) * B = total_pay_this_year - salary_this_year) 
  : S = 100000 :=
by
  sorry

end John_last_year_salary_l90_90660


namespace sum_gcd_lcm_l90_90226

theorem sum_gcd_lcm (a b : ℕ) (h_a : a = 75) (h_b : b = 4500) :
  Nat.gcd a b + Nat.lcm a b = 4575 := by
  sorry

end sum_gcd_lcm_l90_90226


namespace frog_probability_on_vertical_side_l90_90835

noncomputable def P : ℕ × ℕ → ℚ
| (0, y) := 1
| (5, y) := 1
| (x, 0) := 0
| (x, 5) := 0
| (x, y) := (P (x-1, y) + P (x+1, y) + P (x, y-1) + P (x, y+1)) / 4

theorem frog_probability_on_vertical_side :
  P (2, 1) = 13 / 20 :=
sorry

end frog_probability_on_vertical_side_l90_90835


namespace projection_of_a_on_b_l90_90002

variable (a b : ℝ^3) 
variable (norm_a : ∥a∥ = 6)
variable (norm_b : ∥b∥ = 3)
variable (dot_ab : a • b = -12)

theorem projection_of_a_on_b : (a • b) / ∥b∥ = -4 := by
  sorry

end projection_of_a_on_b_l90_90002


namespace binom_10_3_l90_90351

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90351


namespace binary_polynomial_solution_l90_90530

noncomputable def homogeneous_polynomial (P : ℝ × ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (c : ℝ) (x y : ℝ), P(c * x, c * y) = c^n * P(x, y)

theorem binary_polynomial_solution (P : ℝ × ℝ → ℝ) (n : ℕ) :
  homogeneous_polynomial P n →
  (∀ a b c : ℝ, P(a+b, c) + P(b+c, a) + P(c+a, b) = 0) →
  P (1, 0) = 1 →
  ∀ x y: ℝ, P x y = (x + y)^(n - 1) * (x - 2 * y) :=
by
  intros h1 h2 h3 x y
  sorry

end binary_polynomial_solution_l90_90530


namespace find_a123_l90_90693

def sequence (a : ℕ → ℕ) := ∀ n, a (n + 3) = a (n + 2) * (a (n + 1) + 2 * a n)

constant a : ℕ → ℕ

axiom a6 : a 6 = 2288
axiom seq_recurrence : sequence a 

theorem find_a123 : a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 2 :=
sorry

end find_a123_l90_90693


namespace binomial_10_3_l90_90334

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90334


namespace angle_EHC_45deg_l90_90096

open Triangle

variables {A B C H E : Point}
variables [In(ABC : Triangle)]
variables (Altitude AH)
variables (Bisector BE)
variables (Angle BEA = 45)

theorem angle_EHC_45deg (h1 : is_altitude ABC AH)
                        (h2 : is_angle_bisector ABC BE)
                        (h3 : angle BEA = 45) :
  angle EHC = 45 :=
sorry

end angle_EHC_45deg_l90_90096


namespace james_drinks_per_day_l90_90657

-- condition: James buys 5 packs of sodas, each contains 12 sodas
def num_packs : Nat := 5
def sodas_per_pack : Nat := 12
def sodas_bought : Nat := num_packs * sodas_per_pack

-- condition: James already had 10 sodas
def sodas_already_had : Nat := 10

-- condition: James finishes all the sodas in 1 week (7 days)
def days_in_week : Nat := 7

-- total sodas
def total_sodas : Nat := sodas_bought + sodas_already_had

-- number of sodas james drinks per day
def sodas_per_day : Nat := 10

-- proof problem
theorem james_drinks_per_day : (total_sodas / days_in_week) = sodas_per_day :=
  sorry

end james_drinks_per_day_l90_90657


namespace vector_norm_subtraction_l90_90035

variables (a b : ℝ × ℝ)
variables (norm_a : ∥a∥ = 2) (norm_b : ∥b∥ = 1)
variables (orthogonal : (a + b) • a = 0)

theorem vector_norm_subtraction : ∥a - 2 • b∥ = 4 :=
by
  -- Proof steps will be filled here
  sorry

end vector_norm_subtraction_l90_90035


namespace seventh_oblong_is_56_l90_90867

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end seventh_oblong_is_56_l90_90867


namespace binom_10_3_eq_120_l90_90398

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90398


namespace find_digit_D_l90_90782

def is_digit (n : ℕ) : Prop := n < 10

theorem find_digit_D (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : B ≠ C)
  (h5 : B ≠ D) (h6 : C ≠ D) (h7 : is_digit A) (h8 : is_digit B) (h9 : is_digit C) (h10 : is_digit D) :
  (1000 * A + 100 * B + 10 * C + D) * 2 = 5472 → D = 6 := 
by
  sorry

end find_digit_D_l90_90782


namespace departure_sequences_l90_90073

-- Definitions for the conditions
def train_set : Type := {x // x ∈ {A, B, C, D, E, F, G, H}}

-- Prove the total number of different departure sequences for 8 trains is 720
theorem departure_sequences (A B C D E F G H : train_set) :
  (∀ A ∉ B) → (first_departs A) → (last_departs B) → 
  (count_departure_sequences {A, B, C, D, E, F, G, H} 4 4) = 720 :=
by
  sorry

end departure_sequences_l90_90073


namespace binomial_coefficient_10_3_l90_90362

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90362


namespace probability_of_event_l90_90049

theorem probability_of_event :
  let prob_three_kings := (4 / 52) * (3 / 51) * (2 / 50),
      prob_two_aces := ( (4 / 52) * (3 / 51) * (48 / 50) )
        + ( (4 / 52) * (48 / 51) * (3 / 50) )
        + ( (48 / 52) * (4 / 51) * (3 / 50) ),
      prob_three_aces := (4 / 52) * (3 / 51) * (2 / 50),
      prob_event := prob_three_kings + prob_two_aces + prob_three_aces
  in prob_event = (43 / 33150) :=
sorry

end probability_of_event_l90_90049


namespace hakimi_age_l90_90774

theorem hakimi_age
  (avg_age : ℕ)
  (num_friends : ℕ)
  (molly_age : ℕ)
  (age_diff : ℕ)
  (total_age := avg_age * num_friends)
  (combined_age := total_age - molly_age)
  (jared_age := age_diff)
  (hakimi_age := combined_age - jared_age)
  (avg_age = 40)
  (num_friends = 3)
  (molly_age = 30)
  (age_diff = 10)
  (combined_age = 90)
  (hakimi_age = 40) : 
  ∃ age : ℕ, age = hakimi_age :=
by
  sorry

end hakimi_age_l90_90774


namespace arithmetic_sequence_is_a_l90_90769

theorem arithmetic_sequence_is_a
  (a : ℚ) (d : ℚ)
  (h1 : 140 + d = a)
  (h2 : a + d = 45 / 28)
  (h3 : a > 0) :
  a = 3965 / 56 :=
by
  sorry

end arithmetic_sequence_is_a_l90_90769


namespace tangent_line_at_2_range_of_a_l90_90602

-- Define the function f
def f (x a : ℝ) := x * Real.log (x - 1) - a * (x - 2)

-- Part (I)
theorem tangent_line_at_2 {a : ℝ} (h : a = 2017) :
  let x2 := 2.0 in
  let f_x2 := f x2 a in
  let f_prime (x : ℝ) := Real.log (x - 1) + x / (x - 1) - a in
  let f_prime_x2 := f_prime x2 in
  f_prime_x2 = -2015 ∧ f_x2 = 0 ∧ (2015 * x + y - 4030 = 0) :=
sorry

-- Part (II)
theorem range_of_a (h : ∀ x ≥ 2, f x a ≥ 0) : a ≤ 2 :=
sorry

end tangent_line_at_2_range_of_a_l90_90602


namespace friend_wants_to_take_5_marbles_l90_90705

theorem friend_wants_to_take_5_marbles
  (total_marbles : ℝ)
  (clear_marbles : ℝ)
  (black_marbles : ℝ)
  (other_marbles : ℝ)
  (friend_marbles : ℝ)
  (h1 : clear_marbles = 0.4 * total_marbles)
  (h2 : black_marbles = 0.2 * total_marbles)
  (h3 : other_marbles = total_marbles - clear_marbles - black_marbles)
  (h4 : friend_marbles = 2)
  (friend_total_marbles : ℝ)
  (h5 : friend_marbles = 0.4 * friend_total_marbles) :
  friend_total_marbles = 5 := by
  sorry

end friend_wants_to_take_5_marbles_l90_90705


namespace binomial_10_3_l90_90333

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90333


namespace A_wins_by_75_meters_l90_90064

-- Define the conditions
def race_distance : ℝ := 500
def speed_ratio_A_B : ℝ := 3 / 4
def head_start_A : ℝ := 200

-- Define the speeds of A and B
def speed_B (x : ℝ) : ℝ := 4 * x
def speed_A (x : ℝ) : ℝ := 3 * x

-- Define the time it takes for B to complete the race
def time_B (x : ℝ) : ℝ := race_distance / (speed_B x)

-- Define the distance A covers in the time B covers the race distance
def distance_A_covers (x : ℝ) : ℝ := (speed_A x) * (time_B x)

-- Define the total distance A covers including the head start
def total_distance_A_covers (x : ℝ) : ℝ := distance_A_covers x + head_start_A

-- Lean 4 statement to prove A wins the race by 75 meters
theorem A_wins_by_75_meters (x : ℝ) : total_distance_A_covers x - race_distance = 75 :=
by
  sorry

end A_wins_by_75_meters_l90_90064


namespace Ben_Cards_Left_l90_90295

theorem Ben_Cards_Left :
  (4 * 10 + 5 * 8 - 58) = 22 :=
by
  sorry

end Ben_Cards_Left_l90_90295


namespace order_coins_with_expectation_l90_90202

-- Define the four coins
inductive Coin
| A | B | C | D

open Coin

-- Define the method to compare any two coins
def compare_coins (c1 c2 : Coin) : Prop := sorry  -- Here we define the comparison property

theorem order_coins_with_expectation :
  ∃ (method : list (Coin × Coin)), 
  ∀ (outcome : list (Coin × Coin)), 
  (expected_weighings method outcome < 4.8) :=
sorry

end order_coins_with_expectation_l90_90202


namespace binomial_10_3_l90_90499

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90499


namespace circle_intersection_range_l90_90979

theorem circle_intersection_range (a : ℝ) :
  (0 < a ∧ a < 2 * Real.sqrt 2) ∨ (-2 * Real.sqrt 2 < a ∧ a < 0) ↔
  (let C := { p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - a) ^ 2 = 4 };
   let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 4 };
   ∀ p, p ∈ C → p ∈ O) :=
sorry

end circle_intersection_range_l90_90979


namespace remaining_car_speed_l90_90822

theorem remaining_car_speed
  (n : ℕ)
  (h_n : n = 31)
  (h_speed : ∀ i : ℕ, i < n → 61 + i) :
  (∃ k : ℕ, k < n ∧ (n % 2 = 1) ∧ (61 + ((n + 1) / 2 - 1)) = 76) :=
by
  sorry

end remaining_car_speed_l90_90822


namespace graph_of_transformed_function_l90_90755

-- Define the original piecewise function f(x)
def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then
    -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then
    real.sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then
    2 * (x - 2)
  else
    0 -- Default case

-- Define the transformed function g(x) = 2f(x) - 4
def g (x : ℝ) : ℝ :=
  2 * f x - 4

-- Define a predicate indicating a graph corresponds to a mathematical function
def graph_corresponds_to (graph : string) (fn : ℝ → ℝ) : Prop :=
  graph = "A" -- This will be matched with the correct graph

-- The proposition to prove
theorem graph_of_transformed_function :
  graph_corresponds_to "A" g :=
sorry

end graph_of_transformed_function_l90_90755


namespace population_net_increase_period_l90_90996

def period_in_hours (birth_rate : ℕ) (death_rate : ℕ) (net_increase : ℕ) : ℕ :=
  let net_rate_per_second := (birth_rate / 2) - (death_rate / 2)
  let period_in_seconds := net_increase / net_rate_per_second
  period_in_seconds / 3600

theorem population_net_increase_period :
  period_in_hours 10 2 345600 = 24 :=
by
  unfold period_in_hours
  sorry

end population_net_increase_period_l90_90996


namespace sum_of_kth_and_lth_smallest_l90_90165

theorem sum_of_kth_and_lth_smallest (n k l : ℕ) (h_n_odd : n % 2 = 1) (y : ℤ) :
  let a := y - (n - 1)
  in (a + 2 * (k - 1)) + (a + 2 * (l - 1)) = 2 * (y + ↑k + ↑l - n) :=
by
  sorry

end sum_of_kth_and_lth_smallest_l90_90165


namespace primes_with_consecutives_l90_90217

-- Define what it means for a number to be prime
def is_prime (n : Nat) := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬ (n % m = 0)

-- Define the main theorem to prove
theorem primes_with_consecutives (p : Nat) : is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  sorry

end primes_with_consecutives_l90_90217


namespace arithmetic_result_l90_90222

theorem arithmetic_result :
  (3 * 13) + (3 * 14) + (3 * 17) + 11 = 143 :=
by
  sorry

end arithmetic_result_l90_90222


namespace binom_10_3_eq_120_l90_90317

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90317


namespace triangle_area_l90_90985

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let cosa := real.cos (real.acos ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)))
  0.5 * a * b * real.sin (real.acos ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)))

theorem triangle_area : 
  area_of_triangle 35 23 41 ≈ 402.65 := 
by
  sorry

end triangle_area_l90_90985


namespace binom_10_3_eq_120_l90_90318

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90318


namespace combination_10_3_eq_120_l90_90376

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90376


namespace casey_saves_by_paying_monthly_l90_90888

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_per_month := 4
  let months := 3
  let cost_monthly := monthly_rate * months
  let cost_weekly := weekly_rate * weeks_per_month * months
  let savings := cost_weekly - cost_monthly
  savings = 360 :=
by
  unfold weekly_rate monthly_rate weeks_per_month months cost_monthly cost_weekly savings
  simp
  sorry

end casey_saves_by_paying_monthly_l90_90888


namespace original_people_in_room_l90_90650

theorem original_people_in_room (x : ℕ) 
  (h1 : 3 * x / 4 - 3 * x / 20 = 16) : x = 27 :=
sorry

end original_people_in_room_l90_90650


namespace correct_propositions_count_l90_90277

def is_trapezoid_in_one_plane : Prop := ∀ vertices : Fin 4 → ℝ → ℝ, planar vertices
def three_parallel_lines_must_be_coplanar : Prop := ∀ lines : Fin 3 → (ℝ → ℝ), coplanar lines
def two_planes_with_three_common_points_coincide : Prop := ∀ planes : Fin 2 → (ℝ → ℝ), ∀ points : Fin 3 → (ℝ × ℝ), coincide planes points
def four_intersecting_lines_must_be_coplanar : Prop := ∀ lines : Fin 4 → (ℝ → ℝ), ∀ intersection_points : Fin 6 → (ℝ × ℝ), coplanar lines intersection_points

theorem correct_propositions_count :
  (is_trapezoid_in_one_plane ∧
  ¬ three_parallel_lines_must_be_coplanar ∧
  ¬ two_planes_with_three_common_points_coincide ∧
  four_intersecting_lines_must_be_coplanar) -> (2 = 2) :=
by
  sorry

end correct_propositions_count_l90_90277


namespace floor_inequality_l90_90711

theorem floor_inequality (α β : ℝ) : 
  int.floor (2 * α) + int.floor (2 * β) ≥ int.floor α + int.floor β + int.floor (α + β) := 
sorry

end floor_inequality_l90_90711


namespace binomial_10_3_eq_120_l90_90446

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90446


namespace crease_length_l90_90842

-- Define the necessary properties and conditions
noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  
def is_right_triangle (A B C : (ℝ × ℝ)) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 6^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 8^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 10^2

def is_equilateral_triangle (B C D : (ℝ × ℝ)) : Prop :=
  distance B C = 8 ∧ distance C D = 8 ∧ distance B D = 8

-- Define the midpoint of a segment
def midpoint (p1 p2 : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define folding condition
def fold_condition (A D : (ℝ × ℝ)) (M N : (ℝ × ℝ)) : Prop :=
  A = D ∧ distance M N = 2 * real.sqrt 3

-- The main theorem
theorem crease_length
  (A B C D : (ℝ × ℝ))
  (h_right : is_right_triangle A B C)
  (h_equilateral : is_equilateral_triangle B C D)
  (h_fold : fold_condition A D (midpoint A C) (midpoint B D)) :
  distance (midpoint A C) (midpoint B D) = 2 * real.sqrt 3 :=
sorry

end crease_length_l90_90842


namespace midpoint_probability_l90_90671

-- Define T as a set of 3-dimensional coordinates with the given constraints.
def T : set (ℤ × ℤ × ℤ) := 
  {p | ∃ (x y z : ℤ), p = (x, y, z) ∧ 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 0 ≤ z ∧ z ≤ 5}

-- Prove that the probability of midpoint condition holds and p + q = 63.
theorem midpoint_probability (p q : ℕ) (hpq_rel_prime : Nat.gcd p q = 1) :
  (∃ (m n : ℤ × ℤ × ℤ), m ≠ n ∧ m ∈ T ∧ n ∈ T 
    ∧ ((m.1 + n.1) / 2, (m.2 + n.2) / 2, (m.3 + n.3) / 2) ∈ T 
    ∧ p / q = 13 / 50) → p + q = 63 := 
by 
  sorry

end midpoint_probability_l90_90671


namespace num_pairs_equals_one_l90_90916

noncomputable def fractional_part (x : ℚ) : ℚ := x - x.floor

open BigOperators

theorem num_pairs_equals_one :
  ∃! (n : ℕ) (q : ℚ), 
    (0 < q ∧ q < 2000) ∧ 
    ¬ q.isInt ∧ 
    fractional_part (q^2) = fractional_part (n.choose 2000)
:= sorry

end num_pairs_equals_one_l90_90916


namespace find_g_2002_l90_90588

noncomputable def f (x : ℝ) := sorry

def g (x : ℝ) := f x + 1 - x

theorem find_g_2002 (h1 : f 1 = 1)
    (h2 : ∀ (x : ℝ), f (x + 5) ≥ f x + 5)
    (h3 : ∀ (x : ℝ), f (x + 1) ≤ f x + 1) :
    g 2002 = 1 := by
  sorry

end find_g_2002_l90_90588


namespace percentage_error_square_area_l90_90815

theorem percentage_error_square_area (S : ℝ) :
  let S' := 1.05 * S,
      A := S * S,
      A' := S' * S' in
  (A' - A) / A * 100 = 10.25 :=
by
  sorry

end percentage_error_square_area_l90_90815


namespace find_a_plus_t_l90_90583

theorem find_a_plus_t :
  (∀ n : ℕ, sqrt (n + 1 + (n + 1) / ((n + 1)^2 - 1)) = (n + 1) * sqrt ((n + 1) / ((n + 1)^2 - 1)))
  → ∃ a t : ℝ, (sqrt (6 + a / t) = 6 * sqrt (a / t)) ∧ (a + t = 41) :=
by
  intro h
  use 6, 35
  split
  -- Here we would show the equality sqrt(6 + 6 / 35) = 6 * sqrt(6 / 35)
  sorry
  -- And here we conclude a + t = 41
  sorry

end find_a_plus_t_l90_90583


namespace equation_one_solution_equation_two_solution_l90_90282

-- Proof goal for Equation (1)
theorem equation_one_solution (x : ℝ) (h : sqrt (x - 2) - 3 = 0) : x = 11 :=
by
  sorry

-- Proof goal for Equation (2)
theorem equation_two_solution (x : ℝ) (h : sqrt (4 * x^2 + 5 * x) - 2 * x = 1) : x = 1 :=
by
  sorry

end equation_one_solution_equation_two_solution_l90_90282


namespace sum_of_inradii_l90_90649

/-- Given a triangle ABC with AB = 7, AC = 9, and BC = 11, and E being the midpoint of AC. 
    Prove that the sum of the radii of the circles inscribed in triangles AEB and BEC is 3.43. -/
theorem sum_of_inradii {A B C E : Type} [EuclideanGeometry] 
  (hAB : distance A B = 7)
  (hAC : distance A C = 9)
  (hBC : distance B C = 11)
  (hE_midpoint : midpoint A C E) :
  (radius (inscribed_circle (triangle A E B)) + radius (inscribed_circle (triangle B E C))) = 3.43 := 
sorry

end sum_of_inradii_l90_90649


namespace maximum_ab_value_l90_90930

-- Define conditions
variables {a b : ℝ}

-- Statement of the theorem in Lean 4
theorem maximum_ab_value (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : sqrt 3 = sqrt (3^a * 3^b)) :
  ab ≤ 1 / 4 :=
by
  -- Proof goes here, but we add sorry to skip it
  sorry

end maximum_ab_value_l90_90930


namespace initial_amount_of_milk_l90_90133

theorem initial_amount_of_milk (M : ℝ) (h : 0 < M) (h2 : 0.10 * M = 0.05 * (M + 20)) : M = 20 := 
sorry

end initial_amount_of_milk_l90_90133


namespace domain_of_f_l90_90219

def f (x : ℝ) := 1 / (x^2 - ((x - 2) * (x + 2)))

theorem domain_of_f : ∀ x : ℝ, (x^2 - ((x - 2) * (x + 2))) ≠ 0 := by
  intro x
  have h : (x - 2) * (x + 2) = x^2 - 4 := 
    by ring
  rw [h]
  simp
  exact ne_of_gt four_pos

end domain_of_f_l90_90219


namespace binomial_10_3_eq_120_l90_90451

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90451


namespace normal_distribution_abs_prob_l90_90955

-- Define the problem
theorem normal_distribution_abs_prob :
  ∀ (X : ℝ → ℝ), (∀ x, X x ~ Normal 0 1) →
    (P (λ x, X x < -1.96) = 0.025 ) →
       (P (λ x, abs (X x) < 1.96) = 0.950) :=
by
  intro X hX hP
  sorry

end normal_distribution_abs_prob_l90_90955


namespace shaded_area_l90_90895

/-- Consider an isosceles right triangle with legs each measuring 10 units,
    partitioned into 25 smaller congruent triangles, where 15 of these smaller triangles are shaded.
    We aim to prove that the area of the shaded region is 30. -/
theorem shaded_area (leg_length : ℕ) (num_partitions : ℕ) (num_shaded : ℕ)
  (h_leg_length : leg_length = 10)
  (h_num_partitions : num_partitions = 25)
  (h_num_shaded : num_shaded = 15) : 
  let area_large := (1/2 : ℝ) * (leg_length * leg_length) in
  let area_small := area_large / num_partitions in
  let area_shaded := num_shaded * area_small in
  area_shaded = 30 := 
by
  sorry

end shaded_area_l90_90895


namespace binom_10_3_eq_120_l90_90393

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90393


namespace extremum_cond_monotonicity_of_f_product_inequality_l90_90961

noncomputable def f (a x : ℝ) : ℝ := (Math.log (1 + x^2)) + a * x

theorem extremum_cond {a : ℝ} (h : a ≤ 0) : (∃ x : ℝ, f a x = f a 0 ∧ f' a 0 = 0) ↔ a = 0 :=
sorry

theorem monotonicity_of_f {a : ℝ} (h : a ≤ 0) : 
  if a = 0 then 
    (∀x ∈ Set.Ioi 0, f' a x > 0) ∧ (∀x ∈ Set.Iio 0, f' a x < 0) 
  else if a < 0 ∧ a ≤ (-1 : ℝ) then 
    (∀x : ℝ, f' a x ≤ 0) 
  else 
    (∀x : ℝ, (f' a x > 0 → x < (1 / (3 * (x * x))))) :=
sorry

theorem product_inequality (n : ℕ) (k : ℕ) (h: k = n^2) : 
  if k > 0 then 
    let product := (List.prod (List.map (λ i, 1 + 1 / (1 + i^2)) [1..k])) 
    in product < Math.sqrt Math.e 
  else false :=
sorry

end extremum_cond_monotonicity_of_f_product_inequality_l90_90961


namespace intersection_points_l90_90534

def line_eq (t : ℝ) : ℝ × ℝ :=
  (1 - (real.sqrt 5 / 5) * t, -1 + (2 * real.sqrt 5 / 5) * t)

def curve_eq (θ : ℝ) : ℝ × ℝ :=
  (real.sin θ * real.cos θ, real.sin θ + real.cos θ)

theorem intersection_points :
  (∃ t θ : ℝ, (1 - (real.sqrt 5) / 5 * t = real.sin θ * real.cos θ) ∧
               (-1 + (2 * real.sqrt 5) / 5 * t = real.sin θ + real.cos θ)) →
  (0, 1) ∈ {p : ℝ × ℝ | ∃ t θ : ℝ, line_eq t = p ∧ curve_eq θ = p} ∧
  ((3 / 2, -2) ∈ {p : ℝ × ℝ | ∃ t θ : ℝ, line_eq t = p ∧ curve_eq θ = p}) :=
by
  sorry

end intersection_points_l90_90534


namespace tan_inverse_problem_l90_90690

theorem tan_inverse_problem (a b : ℝ) (x : ℝ) (h1 : tan x = 2 * a / (3 * b)) (h2 : tan (3 * x) = 3 * b / (2 * a + 3 * b)) (h3 : a = b) : 
  x = arctan (2 / 3) := 
begin
  sorry  -- Proof skipped as instructed
end

end tan_inverse_problem_l90_90690


namespace number_of_proper_subsets_of_M_l90_90761

-- Define the set M
def M : Set ℕ := {1, 5, 6}

-- The number of proper subsets of M.
theorem number_of_proper_subsets_of_M : 
  (Set.toFinset M).powerset.card - 1 = 7 :=
by
  sorry

end number_of_proper_subsets_of_M_l90_90761


namespace can_all_children_have_all_types_of_candies_l90_90151

theorem can_all_children_have_all_types_of_candies (n k : ℕ) (candies : Fin k → Fin n → ℕ) :
  ∃ f : Fin k → Fin k → Fin n, 
    (∀ i j, f i j ≤ candies i j) ∧
    (∀ c : Fin k, ∃ s : Set (Fin n), s = { t | ∃ i, candies c i = t } ∧ s.card = n)
:=
sorry

end can_all_children_have_all_types_of_candies_l90_90151


namespace combination_10_3_eq_120_l90_90373

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90373


namespace license_plate_count_l90_90631

theorem license_plate_count :
  let first_char_choices := 5 in
  let second_char_choices := 3 in
  let other_char_choices := 4 in
  let total_choices := first_char_choices * second_char_choices * other_char_choices * other_char_choices * other_char_choices in
  total_choices = 960 := 
by 
  let first_char_choices := 5;
  let second_char_choices := 3;
  let other_char_choices := 4;
  let total_choices := first_char_choices * second_char_choices * other_char_choices * other_char_choices * other_char_choices;
  show total_choices = 960, from sorry

end license_plate_count_l90_90631


namespace distance_from_Q_to_EH_is_correct_l90_90158

theorem distance_from_Q_to_EH_is_correct (E F G H N Q : Point)
  (side_length : ℝ) (radius1 radius2 : ℝ)
  (H_coord : H = (0, 0)) (G_coord : G = (5, 0))
  (F_coord : F = (5, 5)) (E_coord : E = (0, 5))
  (N_coord : N = (2.5, 0)) (Q_coord : Q = (2.5, 2.5))
  (circle_N : ∀ (x y : ℝ), (x - 2.5)^2 + y^2 = 6.25)
  (circle_E : ∀ (x y : ℝ), x^2 + (y - 5)^2 = 25) :
  Distance_from_Q_to_EH Q = 2.5 :=
sorry

end distance_from_Q_to_EH_is_correct_l90_90158


namespace a_8_eq_5_l90_90957

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S_eq : ∀ n m : ℕ, S n + S m = S (n + m)
axiom a1 : a 1 = 5
axiom Sn1 : ∀ n : ℕ, S (n + 1) = S n + 5

theorem a_8_eq_5 : a 8 = 5 :=
sorry

end a_8_eq_5_l90_90957


namespace sector_radius_ne_2_l90_90984

theorem sector_radius_ne_2 (R : ℝ) (l : ℝ) (h: 1/2 * l * R = 2 * R + l) : R ≠ 2 := by
  have h1 : l = 4 * R / (R - 2) := by
    sorry
  have : R > 2 := by
    apply lt_of_le_of_ne
    exact le_of_gt (calc
      (4 * R) / (R - 2) > 0 : by sorry)
    intro h2
    exact (by simp [h2]) h1
  intro h3
  linarith [h1]  -- This achieves a contradiction by substituting R = 2 in h1

end sector_radius_ne_2_l90_90984


namespace tetrahedron_volume_ratio_l90_90848

theorem tetrahedron_volume_ratio (s : ℝ) (s_pos : 0 < s) :
  let V_original := (s^3 * real.sqrt 2) / 12
  let V_small := ((s / 2)^3 * real.sqrt 2) / 12    
  let V_total_small := 4 * V_small
  V_total_small / V_original = 1 / 2 := by
  sorry

end tetrahedron_volume_ratio_l90_90848


namespace trajectory_equation_l90_90945

noncomputable def parabola_focus : (ℝ × ℝ) := (0, 1)

def parabola (x y : ℝ) : Prop := y = 1 / 4 * x^2

def midpoint (P F Q : ℝ × ℝ) : Prop :=
  Q.1 = (0 + P.1) / 2 ∧ Q.2 = (1 + P.2) / 2

def on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

theorem trajectory_equation (P Q : ℝ × ℝ) (hP_on_parabola : on_parabola P)
  (h_midpoint : midpoint P parabola_focus Q) :
  Q.1 ^ 2 = 2 * Q.2 - 1 := sorry

end trajectory_equation_l90_90945


namespace Warriors_won_33_games_l90_90186

def games_won (Hawks Falcons Warriors Knights Royals : ℕ) : Prop :=
  Hawks > Falcons ∧ -- The Hawks won more games than the Falcons.
  Warriors > Knights ∧ -- The Warriors won more games than the Knights.
  Warriors < Royals ∧ -- The Warriors won fewer games than the Royals.
  Knights > 22 ∧ -- The Knights won more than 22 games.
  ∃ x ∈ {23, 28, 33, 38, 43}, Knights = x ∧ -- The Knights' possible win numbers.
  Royals = 43 -- The Royals won the most games.

theorem Warriors_won_33_games (Hawks Falcons Warriors Knights Royals : ℕ) (h : games_won Hawks Falcons Warriors Knights Royals) :
  Warriors = 33 :=
sorry -- The implementation of the proof will go here.

end Warriors_won_33_games_l90_90186


namespace find_growth_rate_l90_90110

def avg_jan_prod := 20000
def avg_mar_prod := 24200
def growth_rate (x : ℝ) := 20000 * (1 + x) ^ 2 = 24200

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.1 :=
by
  apply Exists.intro 0.1
  split
  case left => sorry
  case right => rfl

end find_growth_rate_l90_90110


namespace casey_saves_by_paying_monthly_l90_90890

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_in_a_month := 4
  let number_of_months := 3
  let total_weeks := number_of_months * weeks_in_a_month
  let total_cost_weekly := total_weeks * weekly_rate
  let total_cost_monthly := number_of_months * monthly_rate
  let savings := total_cost_weekly - total_cost_monthly
  savings = 360 :=
by
  sorry

end casey_saves_by_paying_monthly_l90_90890


namespace fraction_female_is_correct_l90_90111

theorem fraction_female_is_correct :
  let last_year_males := 30
  let last_year_total_participants := 30 + y
  let this_year_total_participants := 1.15 * (30 + y)
  let this_year_males := 1.10 * 30
  let this_year_females := 1.25 * y
  let this_year_participants := this_year_males + this_year_females
  let fraction_females := this_year_females / this_year_participants
  in 33 + 1.25 * y = 34.5 + 1.15 * y → y = 15 → 
     this_year_females = 19 → 
     this_year_participants = 52 →
     fraction_females = 19 / 52 :=
by
  sorry

end fraction_female_is_correct_l90_90111


namespace stating_martha_painting_time_l90_90698

/-- 
  Theorem stating the time it takes for Martha to paint the kitchen is 42 hours.
-/
theorem martha_painting_time :
  let width1 := 12
  let width2 := 16
  let height := 10
  let area_pair1 := 2 * width1 * height
  let area_pair2 := 2 * width2 * height
  let total_area := area_pair1 + area_pair2
  let coats := 3
  let total_paint_area := total_area * coats
  let painting_speed := 40
  let time_required := total_paint_area / painting_speed
  time_required = 42 := by
    -- Since we are asked not to provide the proof steps, we use sorry to skip the proof.
    sorry

end stating_martha_painting_time_l90_90698


namespace seventh_oblong_number_l90_90865

/-- An oblong number is the number of dots in a rectangular grid where the number of rows is one more than the number of columns. -/
def is_oblong_number (n : ℕ) (x : ℕ) : Prop :=
  x = n * (n + 1)

/-- The 7th oblong number is 56. -/
theorem seventh_oblong_number : ∃ x, is_oblong_number 7 x ∧ x = 56 :=
by 
  use 56
  unfold is_oblong_number
  constructor
  rfl -- This confirms the computation 7 * 8 = 56
  sorry -- Wrapping up the proof, no further steps needed

end seventh_oblong_number_l90_90865


namespace total_frames_l90_90662

def frames_per_page : ℝ := 143.0

def pages : ℝ := 11.0

theorem total_frames : frames_per_page * pages = 1573.0 :=
by
  sorry

end total_frames_l90_90662


namespace eq_x_value_l90_90303

theorem eq_x_value : ∃ x : ℕ, 289 + 2 * 17 * 5 + 25 = x ∧ x = 484 := by
  use 484
  sorry

end eq_x_value_l90_90303


namespace drink_total_ounces_l90_90102

theorem drink_total_ounces (coke sprite mountainDew drPepper fanta : ℤ) (coke_ounces : ℤ) :
  coke = 4 ∧ sprite = 2 ∧ mountainDew = 5 ∧ drPepper = 3 ∧ fanta = 2 ∧ coke_ounces = 12 →
  (coke + sprite + mountainDew + drPepper + fanta) * (coke_ounces / coke) = 48 :=
by
  intros h,
  cases h with hc hs,
  cases hs with hm hf,
  cases hf with hd hp,
  cases hp with ha hb,
  sorry

end drink_total_ounces_l90_90102


namespace pets_distribution_l90_90262

theorem pets_distribution (puppies kittens hamsters : ℕ) (people : ℕ) (roles : ℕ) :
  puppies = 15 ∧ kittens = 10 ∧ hamsters = 8 ∧ people = 3 ∧ roles = 6 →
  puppies * kittens * hamsters * roles = 7200 :=
by 
  intros h,
  have h1 : puppies = 15 := h.1,
  have h2 : kittens = 10 := h.2.1,
  have h3 : hamsters = 8 := h.2.2.1,
  have h4 : people = 3 := h.2.2.2.1,
  have h5 : roles = 6 := h.2.2.2.2,
  rw [h1, h2, h3, h5],
  norm_num,
  sorry

end pets_distribution_l90_90262


namespace binomial_10_3_eq_120_l90_90447

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90447


namespace additional_people_needed_l90_90918

theorem additional_people_needed 
  (people : ℕ) (hours : ℕ) (size_factor : ℝ) (total_time : ℕ) (setup_time : ℕ)
  (h1 : people = 5) (h2 : hours = 8) (h3 : size_factor = 1.5) (h4 : total_time = 6) (h5 : setup_time = 1) 
  : (size_factor * (people * hours) / (total_time - setup_time) - people).ceil = 7 := 
by 
  sorry

end additional_people_needed_l90_90918


namespace binomial_10_3_l90_90495

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90495


namespace starting_number_divisible_by_seven_l90_90880

theorem starting_number_divisible_by_seven (x : ℕ) (h1 : x ≡ 0 [MOD 7]) (h2 : x ≤ 57) (h3 : (x + 57) / 2 = 38.5) : x = 21 :=
sorry

end starting_number_divisible_by_seven_l90_90880


namespace binomial_10_3_eq_120_l90_90430

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90430


namespace least_N_mod_1000_l90_90545

def sum_of_base_digits (n : ℕ) (b : ℕ) : ℕ :=
  nat.digits b n |>.sum

def f (n : ℕ) : ℕ := sum_of_base_digits n 5

def g (n : ℕ) : ℕ := sum_of_base_digits (f n) 7

theorem least_N_mod_1000 : 
  ∃ (N : ℕ), (g N ≥ 10) ∧ ((N % 1000) = 406) :=
sorry

end least_N_mod_1000_l90_90545


namespace Ivan_ball_ways_l90_90703

theorem Ivan_ball_ways (N : ℕ) :
  let guests := (Fin (2 * N)) in
  let hats := vector bool (2 * N) in
  let black_count := hats.toList.count true = N in
  let white_count := hats.toList.count false = N in
  let no_adj_same_color := ∀ i, hats.nth i ≠ hats.nth ((i + 1) % (2 * N)) in
  black_count → white_count → no_adj_same_color → (2 * N)! = (2 * N)! :=
by
  intros
  sorry

end Ivan_ball_ways_l90_90703


namespace p_iff_q_l90_90609

def p (x : ℝ) : Prop := -1 < log x / log (1/2) ∧ log x / log (1/2) < 0
def q (x : ℝ) : Prop := 2 ^ x > 1

theorem p_iff_q (x : ℝ) : (p x) ↔ (q x) :=
by
  sorry

end p_iff_q_l90_90609


namespace total_amount_of_money_l90_90112

-- Information and conditions
variables (T : ℝ)
variables (Ryan_fraction Leo_fraction : ℝ)
variables (Ryan_share Leo_share : ℝ)
variables (Leo_after_debt : ℝ)
variables (net_gain : ℝ)

-- Setting up the conditions
def conditions (T : ℝ) : Prop :=
  Ryan_fraction = 2 / 3 ∧
  Leo_fraction = 1 / 3 ∧
  Ryan_share = Ryan_fraction * T ∧
  Leo_share = Leo_fraction * T ∧
  Leo_after_debt = 19 ∧
  net_gain = 10 - 7 ∧
  Leo_after_debt = Leo_share + net_gain

-- The theorem we need to prove
theorem total_amount_of_money : conditions T → T = 48 :=
by
  intro h
  sorry

end total_amount_of_money_l90_90112


namespace triangle_reflection_sumsq_l90_90199

theorem triangle_reflection_sumsq {A B C A' B' C' : Type}
  (h1 : ∃ (ABC : Triangle A B C), (ABC.rightAngle = C))
  (h2 : ∃ (A' B' C' : Points), 
        A' = ImageOfInCentralSymmetryWithCenterAt(A, C) ∧ 
        B' = ImageOfInCentralSymmetryWithCenterAt(B, A) ∧ 
        C' = ImageOfInCentralSymmetryWithCenterAt(C, B)) : 
  dist A' B'^2 + dist B' C'^2 + dist C' A'^2 = 14 * dist A B^2 :=
by 
  -- proof omitted
  sorry

end triangle_reflection_sumsq_l90_90199


namespace proof_problem_l90_90885

noncomputable def problem_statement : Float := 
  ((-1 / 3)⁻¹ : Float) - (sqrt 8) - ((5 - Real.pi) ^ 0) + 4 * (Real.cos (Real.pi / 4))

theorem proof_problem : 
  problem_statement = -4 := 
by
  sorry

end proof_problem_l90_90885


namespace ratio_third_to_second_year_l90_90279

-- Define the yearly production of the apple tree
def first_year_production : Nat := 40
def second_year_production : Nat := 2 * first_year_production + 8
def total_production_three_years : Nat := 194
def third_year_production : Nat := total_production_three_years - (first_year_production + second_year_production)

-- Define the ratio calculation
def ratio (a b : Nat) : (Nat × Nat) := 
  let gcd_ab := Nat.gcd a b 
  (a / gcd_ab, b / gcd_ab)

-- Prove the ratio of the third year's production to the second year's production
theorem ratio_third_to_second_year : 
  ratio third_year_production second_year_production = (3, 4) :=
  sorry

end ratio_third_to_second_year_l90_90279


namespace inequality_solution_l90_90548

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end inequality_solution_l90_90548


namespace solution_set_of_inequality_l90_90173

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x : ℝ, f x > 1 - (deriv^[2] f x)) ∧ f 0 = 0 →
  { x : ℝ | e^x * f x > e^x - 1 } = { x : ℝ | 0 < x } :=
by
  intros h,
  sorry

end solution_set_of_inequality_l90_90173


namespace six_chests_even_distribution_eight_chests_even_distribution_l90_90813

/-
There are six chests, each containing coins.
If any two chests are opened, the coins can be evenly distributed between them.
If any 3, 4, or 5 chests are opened, the coins can be distributed such that each opened chest contains the same number of coins.
We need to determine if the coins can be evenly distributed among all six chests.
-/
theorem six_chests_even_distribution (a b c d e f : ℕ) (h1 : ∃ (k1 k2 k3 : ℕ), a + b = k1 * 2 ∧ c + d = k2 * 2 ∧ e + f = k3 * 2)
  (h2 : ∃ (k4 k5 k6 k7 : ℕ), a + b + c = k4 * 3 ∧ d + e + f = k5 * 3 ∧ a + b + c + d = k6 * 4 ∧ c + d + e + f = k7 * 4) :
  ∃ (k : ℕ), a + b + c + d + e + f = k * 6 := sorry

/-
There are eight chests, each containing coins.
Coins can be evenly distributed if any 2, 3, 4, 5, 6, or 7 chests are opened.
We need to determine if the coins can be evenly distributed among all eight chests.
-/
theorem eight_chests_even_distribution (a b c d e f g h : ℕ) 
  (h1 : ∀ (x y : ℕ), x ≠ y → ∃ (k : ℕ), (list.sum (list.filter_map (λ i, if i = x ∨ i = y then some i else none) [a,b,c,d,e,f,g,h])) = k * 2)
  (h2 : ∀ (x y z : ℕ), perm ([a,b,c,d,e,f,g,h], [x,y,z]) → ∃ (k : ℕ), (list.sum [x,y,z]) = k * 3) :
  ¬ ∃ (k : ℕ), (a + b + c + d + e + f + g + h) = k * 8 := sorry

end six_chests_even_distribution_eight_chests_even_distribution_l90_90813


namespace difference_between_digits_l90_90237

-- Define the number k as a 3-digit number.
def is_3_digit_number (k : ℕ) : Prop :=
  100 ≤ k ∧ k < 1000

-- Define the sum of the digits function.
def sum_of_digits (k : ℕ) : ℕ :=
  let a := k / 100
  let b := (k % 100) / 10
  let c := k % 10
  a + b + c

-- Define the ratio function.
def ratio (k : ℕ) : ℚ :=
  k.toRat / (sum_of_digits k).toRat

-- The mathematical statement to prove
theorem difference_between_digits (k : ℕ) (h1 : is_3_digit_number k) (h2 : ∀ n : ℕ, is_3_digit_number n → ratio k ≤ ratio n) :
  let a := k / 100
  let b := (k % 100) / 10
  abs (a - b) = 8 :=
sorry

end difference_between_digits_l90_90237


namespace min_value_of_sum_eq_l90_90559

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end min_value_of_sum_eq_l90_90559


namespace range_of_a_l90_90059

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
sorry

end range_of_a_l90_90059


namespace big_eighteen_basketball_games_count_l90_90745

def num_teams_in_division := 6
def num_teams := 18
def games_within_division := 3
def games_between_divisions := 1
def divisions := 3

theorem big_eighteen_basketball_games_count :
  (num_teams * ((num_teams_in_division - 1) * games_within_division + (num_teams - num_teams_in_division) * games_between_divisions)) / 2 = 243 :=
by
  have teams_in_other_divisions : num_teams - num_teams_in_division = 12 := rfl
  have games_per_team_within_division : (num_teams_in_division - 1) * games_within_division = 15 := rfl
  have games_per_team_between_division : 12 * games_between_divisions = 12 := rfl
  sorry

end big_eighteen_basketball_games_count_l90_90745


namespace expression_of_quadratic_function_coordinates_of_vertex_l90_90953

def quadratic_function_through_points (a b : ℝ) : Prop :=
  (0 = a * (-3)^2 + b * (-3) + 3) ∧ (-5 = a * 2^2 + b * 2 + 3)

theorem expression_of_quadratic_function :
  ∃ a b : ℝ, quadratic_function_through_points a b ∧ ∀ x : ℝ, -x^2 - 2 * x + 3 = a * x^2 + b * x + 3 :=
by
  sorry

theorem coordinates_of_vertex :
  - (1 : ℝ) * (1 : ℝ) = (-1) / (2 * (-1)) ∧ 4 = -(1 - (-1) + 3) + 4 :=
by
  sorry

end expression_of_quadratic_function_coordinates_of_vertex_l90_90953


namespace combination_10_3_eq_120_l90_90374

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90374


namespace binom_10_3_eq_120_l90_90396

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90396


namespace table_tennis_survey_analysis_l90_90744

/-- 
  The 2022 World Table Tennis Team Championships were held in Chengdu. 
  The Chinese women's team and men's team won the team championships on October 8th and October 9th, respectively. 
  In order to understand the correlation between gender and whether the audience enjoys watching table tennis matches,
  a sports channel randomly selected 200 spectators for a survey. 

  The following contingency table was obtained:
  - Male who enjoy watching: 60
  - Male who do not enjoy watching: 40
  - Female who enjoy watching: 20
  - Female who do not enjoy watching: 80
  
  Using the chi-squared formula to test for independence,
  and the reference table of critical values, our task is to determine the correct statements.
-/
theorem table_tennis_survey_analysis :
  let a := 60
  let b := 40
  let c := 20
  let d := 80
  let n := a + b + c + d
  -- Calculating chi-squared value
  let chi_squared := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))
  -- Reference critical value for α = 0.001
  let chi_squared_critical := 10.828
  -- Frequency calculations
  let freq_females_enjoying := c / (a + c)
  let freq_males_enjoying := a / (a + b)
  -- Statements
  let A := freq_females_enjoying = 1 / 4
  let B := freq_males_enjoying = 2 / 3
  let D := chi_squared > chi_squared_critical
  A ∧ D :=
by
  let a := 60
  let b := 40
  let c := 20
  let d := 80
  let n := a + b + c + d
  let chi_squared := (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))
  let chi_squared_critical := 10.828
  let freq_females_enjoying := c / (a + c)
  let freq_males_enjoying := a / (a + b)
  let A := freq_females_enjoying = 1 / 4
  let B := freq_males_enjoying = 2 / 3
  let D := chi_squared > chi_squared_critical
  -- Prove that A and D are true
  exact ⟨ sorry, sorry ⟩

end table_tennis_survey_analysis_l90_90744


namespace probability_snow_at_least_once_l90_90547

theorem probability_snow_at_least_once :
  let first_five_days_no_snow := (1 / 2 * (4 / 5) + 1 / 2 * (7 / 10)) ^ 5,
      next_five_days_no_snow := (1 / 2 * (2 / 3) + 1 / 2 * (5 / 6)) ^ 5
  in 1 - (first_five_days_no_snow * next_five_days_no_snow) = 58806 / 59049 :=
by
  sorry

end probability_snow_at_least_once_l90_90547


namespace binom_10_3_eq_120_l90_90311

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90311


namespace sum_of_first_five_terms_l90_90941

-- Define the arithmetic sequence and the sum of the first n terms
def arith_seq (a : ℕ -> ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, a i

-- Define the given conditions
variables {a : ℕ → ℝ}
hypothesis arith_seq_def : arith_seq a
hypothesis condition : a 2 + a 4 = 18

-- The statement to prove
theorem sum_of_first_five_terms : sum_arith_seq a 5 = 45 :=
  sorry

end sum_of_first_five_terms_l90_90941


namespace orthocenter_parallelogram_bisector_circumcircle_l90_90556

/-- Given H is the orthocenter of the acute triangle ABC,
G is such that the quadrilateral ABGH is a parallelogram,
and I is a point on the line GH such that AC bisects the segment HI.
If the line AC intersects the circumcircle of triangle GCI at points C and J,
then IJ = AH. -/
theorem orthocenter_parallelogram_bisector_circumcircle
  (A B C G H I J : Point)
  (h_orthocenter : is_orthocenter H (triangle.mk A B C))
  (h_parallelogram : is_parallelogram (quadrilateral.mk A B G H))
  (h_on_line : is_on_line I G H)
  (h_bisects : is_bisector (A, C) (H, I))
  (h_circumcircle : is_circumcircle (triangle.mk G C I) J (point_of) AC)
  : distance I J = distance A H := sorry

end orthocenter_parallelogram_bisector_circumcircle_l90_90556


namespace students_per_bench_l90_90184

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench_l90_90184


namespace smartphone_price_l90_90137

/-
Question: What is the sticker price of the smartphone, given the following conditions?
Conditions:
1: Store A offers a 20% discount on the sticker price, followed by a $120 rebate. Prices include an 8% sales tax applied after all discounts and fees.
2: Store B offers a 30% discount on the sticker price but adds a $50 handling fee. Prices include an 8% sales tax applied after all discounts and fees.
3: Natalie saves $27 by purchasing the smartphone at store A instead of store B.

Proof Problem:
Prove that given the above conditions, the sticker price of the smartphone is $1450.
-/

theorem smartphone_price (p : ℝ) :
  (1.08 * (0.7 * p + 50) - 1.08 * (0.8 * p - 120)) = 27 ->
  p = 1450 :=
by
  sorry

end smartphone_price_l90_90137


namespace binom_10_3_l90_90344

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90344


namespace vertices_on_sphere_surface_area_l90_90171

-- Definition of the problem with conditions
def surface_area_of_cube (a : ℝ) : ℝ := 6 * a
def edge_length_of_cube (a : ℝ) : ℝ := sqrt a
def body_diagonal_of_cube (a : ℝ) : ℝ := sqrt (3 * a)
def radius_of_sphere (a : ℝ) : ℝ := (body_diagonal_of_cube a) / 2
def surface_area_of_sphere (r : ℝ) : ℝ := 4 * π * r^2

-- Statement of the theorem
theorem vertices_on_sphere_surface_area (a : ℝ) (h : a > 0) :
  surface_area_of_cube a = 6 * a →
  surface_area_of_sphere (radius_of_sphere a) = 3 * π * a := by
  sorry

end vertices_on_sphere_surface_area_l90_90171


namespace binomial_coefficient_10_3_l90_90357

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90357


namespace simplified_expression_l90_90218

noncomputable def simplify_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : ℝ :=
  (x⁻¹ - z⁻¹)⁻¹

theorem simplified_expression (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) : 
  simplify_expression x z hx hz = x * z / (z - x) := 
by
  sorry

end simplified_expression_l90_90218


namespace part1_tangent_part2_maximum_value_part3_inequality_l90_90127

-- Part 1
theorem part1_tangent (a b : ℝ) (h1 : (a - 2 * b = 0)) (h2 : (-b = -1 / 2)) :
  a = 1 ∧ b = 1 / 2 :=
sorry

-- Part 2
theorem part2_maximum_value :
  let f : ℝ → ℝ := λ x, Real.log x - (1 / 2 * x^2)
  ∃ (x_max : ℝ), x_max ∈ Set.Icc (1 / Real.exp 1) Real.exp 1 ∧ ∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f x ≤ f x_max ∧ f x_max = -1 / 2 :=
sorry

-- Part 3
theorem part3_inequality (m : ℝ) :
  (∀ a ∈ Set.Icc 0 (3 / 2) {x : ℝ | 1 < x ∧ x ≤ Real.exp 2 → a * Real.log x - x ≥ m + x}) →
  m ≤ -Real.exp 2 :=
sorry

end part1_tangent_part2_maximum_value_part3_inequality_l90_90127


namespace ellipse_and_hyperbola_equations_l90_90943

theorem ellipse_and_hyperbola_equations (a b m n : ℝ) (a_pos : a > 0) (b_pos : b > 0) (m_pos : m > 0) (n_pos : n > 0) (a_gt_b : a > b) 
  (focus_common : ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 / m^2 - y^2 / n^2 = 1) → (x = 2 ∧ y = 0)) 
  (tangent_asymptote : ∀ x y, (x^2 / m^2 - y^2 / n^2 = 1 ∧ x > 0 ∧ y > 0) → y = n / m * x) 
  (symmetry_points : ∀ x y, (x^2 / m^2 - y^2 / n^2 = 1) → ((x - 2)^2 / a^2 + y^2 / b^2 = 1)) 
  : (a = 2 ∧ b = 1.5 ∧ m = (2/√5) ∧ n = √(16/5) ∧ (∀ x y, (x^2 / 4 + y^2 / 3 = 1 ↔ T(x,y)) ∧ (x^2 / 4 - y^2 / 16 ≤ 1 ↔ S(x,y))) :=
begin
  sorry
end

end ellipse_and_hyperbola_equations_l90_90943


namespace sin_three_pi_div_two_l90_90913

theorem sin_three_pi_div_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_div_two_l90_90913


namespace tangent_line_at_one_monotonic_increasing_l90_90962

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x^2

theorem tangent_line_at_one (a : ℝ) (x f_x : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : a = 1) 
  (hf : f = func a) (hf1 : f 1 = f_x) (h_deriv : f' = (λ x, Real.log x + 1 - 2 * x))
  (hf1_val : f 1 = -1) (hf1_slope : f' 1 = -1) :
  tangent_line_at (λ x, x * Real.log x - x^2) (1, -1) = λ x, -x := 
sorry

-- Monotonicity on [1, e] (a ≥ e)
theorem monotonic_increasing (a : ℝ) :
  (∀ x ∈ Set.Icc (1:ℝ) Real.exp, f' x ≥ 0) ↔ (a ≥ Real.exp) := 
sorry

end tangent_line_at_one_monotonic_increasing_l90_90962


namespace binomial_coefficient_10_3_l90_90365

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90365


namespace inequality_solution_l90_90549

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ (Set.Ioo 0 6) ∪ (Set.Ioi 6) :=
by
  sorry

end inequality_solution_l90_90549


namespace binom_10_3_eq_120_l90_90474

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90474


namespace floor_inequality_l90_90712

theorem floor_inequality (α β : ℝ) : 
  int.floor (2 * α) + int.floor (2 * β) ≥ int.floor α + int.floor β + int.floor (α + β) := 
sorry

end floor_inequality_l90_90712


namespace binomial_10_3_eq_120_l90_90452

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90452


namespace gas_station_total_boxes_l90_90257

theorem gas_station_total_boxes
  (chocolate_boxes : ℕ)
  (sugar_boxes : ℕ)
  (gum_boxes : ℕ)
  (licorice_boxes : ℕ)
  (sour_boxes : ℕ)
  (h_chocolate : chocolate_boxes = 3)
  (h_sugar : sugar_boxes = 5)
  (h_gum : gum_boxes = 2)
  (h_licorice : licorice_boxes = 4)
  (h_sour : sour_boxes = 7) :
  chocolate_boxes + sugar_boxes + gum_boxes + licorice_boxes + sour_boxes = 21 := by
  sorry

end gas_station_total_boxes_l90_90257


namespace subset_selection_count_l90_90939

theorem subset_selection_count {α : Type} (U : set α) (hU : U = {a, b, c, d}) :
  (∃ (A B C D : set α), A = ∅ ∧ D = U ∧ 
    (∀ X ∈ {A, B, C, D}, (X ⊆ U)) ∧
    (∀ X Y ∈ {A, B, C, D}, X ≠ Y → (X ⊆ Y ∨ Y ⊆ X)) ∧
    (finite {A,B,C,D}) ∧
    {A, B, C, D}.card = 4) →
  {select | ∃ (A B C D : set α), A = ∅ ∧ D = U ∧ 
    (∀ X ∈ {A, B, C, D}, (X ⊆ U)) ∧
    (∀ X Y ∈ {A, B, C, D}, X ≠ Y → (X ⊆ Y ∨ Y ⊆ X))}.card = 36 := 
sorry

end subset_selection_count_l90_90939


namespace binomial_10_3_eq_120_l90_90417

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90417


namespace solve_for_y_l90_90893

noncomputable def smallest_positive_angle (y : ℝ) : Prop :=
  6 * sin y * (cos y)^3 - 6 * (sin y)^3 * cos y = 1 / 2

theorem solve_for_y : ∃ y, smallest_positive_angle y ∧ y = 1 / 4 * arcsin (1 / 3) :=
by
  sorry

end solve_for_y_l90_90893


namespace right_triangle_faces_polyhedron_l90_90594

theorem right_triangle_faces_polyhedron (n : ℕ) (h : n ≥ 4) : 
  (∃ (P : Type) [Polyhedron P], (faces_count P = n) ∧ (∀ f ∈ faces P, is_right_triangle f)) ↔ (∃ k : ℕ, n = 2 * k) :=
sorry

end right_triangle_faces_polyhedron_l90_90594


namespace binom_10_3_eq_120_l90_90468

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90468


namespace max_episodes_l90_90078

theorem max_episodes (n : ℕ) (h : n = 20) :
  let max_event_per_hero := 1 + 19 + 19 in
  let max_total_events := n * max_event_per_hero in
  max_total_events = 780 :=
by
  sorry

end max_episodes_l90_90078


namespace find_height_l90_90576

variables {AA_1 BD AO A₁ C₁ : ℝ}
variables {h : ℝ}
variables {π : RealType}

-- Definitions based on given conditions
def unit_square_base (ABCD : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (x y z : ℝ × ℝ × ℝ),
    x ∈ ABCD → y ∈ ABCD → z ∈ ABCD →
    dist x y = 1 ∧ dist y z = 1 ∧ dist x z = 1 ∧ 
    (∃ a b c d, 
       a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ 
       a = x ∧ b = y ∧ c = z ∧ d = x)

def dihedral_angle_eq (A₁ BD C₁ : ℝ × ℝ × ℝ) (angle : ℝ) : Prop :=
  ∃ O : ℝ × ℝ × ℝ, O = ((1/2), (1/2), 0) ∧
  angle = (∠((A₁ - O), (C₁ - O)))

-- Lean 4 statement
theorem find_height {ABCD A₁BD C₁ : ℝ × ℝ × ℝ}
  (h_eq : h = AA_1)
  (unit_abcd : unit_square_base ABCD)
  (angle_eq : dihedral_angle_eq A₁ BD C₁ (π / 3)) :
  AA_1 = (ℝ.sqrt 6 / 2) := 
sorry

end find_height_l90_90576


namespace value_of_f_2008_l90_90936

noncomputable def f : ℕ → ℤ
| 1       := 2
| 2       := -2
| (n + 1) := f n - f (n - 1)

theorem value_of_f_2008 : f 2008 = -2 :=
sorry

end value_of_f_2008_l90_90936


namespace geom_seq_general_formula_l90_90573

-- Define the geometric sequence with conditions
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a (n + 1) = a 1 * 2 ^ n) ∧
  (0 < a 1 * uint / 1) ∧
  (a 2 + a 4 = 20) ∧
  (a 3 = 8)

-- Prove the general formula for a_n and the sum of the first n terms S_n
theorem geom_seq_general_formula (a : ℕ → ℕ) (Sn : ℕ → ℕ) 
  (h : geometric_sequence a) :
  (∀ n, a n = 2^n) ∧
  (∀ n, Sn n = 2^(n + 1) - 2) :=
sorry

end geom_seq_general_formula_l90_90573


namespace four_expressions_equal_30_l90_90233

theorem four_expressions_equal_30 :
  (6 * 6 - 6 = 30) ∧ 
  (5 * 5 + 5 = 30) ∧ 
  (33 - 3 = 30) ∧ 
  (3^3 + 3 = 30) :=
by
  split
  · show 6 * 6 - 6 = 30; sorry
  split
  · show 5 * 5 + 5 = 30; sorry
  split
  · show 33 - 3 = 30; sorry
  · show 3^3 + 3 = 30; sorry

end four_expressions_equal_30_l90_90233


namespace binom_10_3_eq_120_l90_90469

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90469


namespace Bernardo_wins_with_smallest_M_l90_90297

-- Define the operations
def Bernardo_op (n : ℕ) : ℕ := 3 * n
def Lucas_op (n : ℕ) : ℕ := n + 75

-- Define the game behavior
def game_sequence (M : ℕ) : List ℕ :=
  [M, Bernardo_op M, Lucas_op (Bernardo_op M), Bernardo_op (Lucas_op (Bernardo_op M)),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M)))),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))))]

-- Define winning condition
def Bernardo_wins (M : ℕ) : Prop :=
  let seq := game_sequence M
  seq.get! 5 < 1200 ∧ seq.get! 6 >= 1200

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- The final theorem statement
theorem Bernardo_wins_with_smallest_M :
  Bernardo_wins 9 ∧ (∀ M < 9, ¬Bernardo_wins M) ∧ sum_of_digits 9 = 9 :=
by
  sorry

end Bernardo_wins_with_smallest_M_l90_90297


namespace combination_10_3_eq_120_l90_90383

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90383


namespace cucumber_to_pencil_ratio_l90_90100

-- Define the conditions
def price_each := 20
def discount := 0.2
def num_cucumbers := 100
def total_spent := 2800

-- Calculate the discounted price for pencils
def discounted_price := price_each * (1 - discount)

-- Define the statement of the theorem
theorem cucumber_to_pencil_ratio :
  let amount_spent_on_cucumbers := num_cucumbers * price_each in
  let amount_spent_on_pencils := total_spent - amount_spent_on_cucumbers in
  let num_pencils := amount_spent_on_pencils / discounted_price in
  (num_cucumbers : ℚ) / num_pencils = 2 := 
by
  -- Contents of proof would go here
  sorry

end cucumber_to_pencil_ratio_l90_90100


namespace exactly_one_unsweepable_town_l90_90629

open Classical

variable (n : ℕ)
variable (towns : Fin n → (ℕ × ℕ))  -- (left bulldozer size, right bulldozer size)
variable (distinct_sizes : ∀ i j, i ≠ j → (towns i).fst ≠ (towns j).fst ∧ (towns i).snd ≠ (towns j).snd)

-- Define what it means for town A to sweep town B away and vice versa
def sweep_away (A B : ℕ) : Prop :=
  ∀ i j, (i < j) → (towns j).fst < (towns i).snd

theorem exactly_one_unsweepable_town (h : n ≥ 1) :
  ∃! i, ∀ j ≠ i, ¬ sweep_away j i :=
sorry

end exactly_one_unsweepable_town_l90_90629


namespace shortest_travel_time_l90_90811

open Real

-- Define the basic properties of the checkerboard
def checkerboard := (5, 6)
def bottom_left_corner := (0, 0)
def top_right_corner := (6, 5)

-- Define the speeds
def speed_on_white_or_boundary := 2
def speed_through_black := 1

-- Shortest travel time from bottom-left to top-right
theorem shortest_travel_time :
  -- Given the speeds and the checkerboard properties
  ∃ time : ℝ, time = (1 + 5 * sqrt 2) / 2 ∧ 
    -- Given the conditions as previously discussed
    shortest_path_travel_time checkerboard bottom_left_corner top_right_corner speed_on_white_or_boundary speed_through_black = time :=
begin
  sorry  -- Proof is omitted as per instructions.
end

end shortest_travel_time_l90_90811


namespace binomial_coefficient_10_3_l90_90356

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90356


namespace binom_10_3_eq_120_l90_90392

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90392


namespace Ben_Cards_Left_l90_90296

theorem Ben_Cards_Left :
  (4 * 10 + 5 * 8 - 58) = 22 :=
by
  sorry

end Ben_Cards_Left_l90_90296


namespace sector_ratio_l90_90999

theorem sector_ratio (P E F G H : Type) 
  [circle P] [diameter G H] (h1: same_side P E F G H) 
  (h2: angle_gpe: ∠GPE = 60) (h3: angle_fph: ∠FPH = 90) 
  (h4: angle_gph: ∠GPH = 180) : 
  (area_sector P E F) / (area_circle P) = 1 / 12 :=
sorry

end sector_ratio_l90_90999


namespace binom_10_3_eq_120_l90_90324

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90324


namespace minimum_k_triangle_l90_90871

theorem minimum_k_triangle (n : ℕ) :
  ∃ k, k = (n / 2).floor + 1 -> ∀ M : Finset (Fin n), (∀ x ∈ M, ∃ s : Finset (Fin n), s.card = k ∧ s ⊆ (M.erase x) ∧ ∀ y ∈ s, y ∈ M) → 
    ∃ (X Y Z : Fin n), X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ (X, Y) ∈ (relation) ∧ (X, Z) ∈ (relation) ∧ (Y, Z) ∈ (relation) :=
by
  sorry

end minimum_k_triangle_l90_90871


namespace parallelogram_count_l90_90995

theorem parallelogram_count
  (set1 : Set ℝ) (set2 : Set ℝ)
  (card_set1 : set1.card = 3)
  (card_set2 : set2.card = 5) :
  ∃ n, n = 30 ∧ (∀ l1 l2 ∈ set1, l1 ≠ l2) ∧ 
  (∀ l3 l4 ∈ set2, l3 ≠ l4) → 
  (∃ p, p.card = 30) := 
sorry

end parallelogram_count_l90_90995


namespace minimum_votes_for_tall_to_win_l90_90088

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l90_90088


namespace area_of_triangle_DEC_l90_90072

noncomputable def area_triangle_DEC : ℝ :=
  let AC := 4 in
  let AB := 6 in
  let BC := 6 in
  let AD := (2/5) * AB in
  let DE := 44 / 60 in
  let target_area := (11/90) * (1/2 * AC * (4 * Real.sqrt 2)) in
  target_area

theorem area_of_triangle_DEC : area_triangle_DEC = 44 * Real.sqrt 2 / 45 :=
by
  sorry

end area_of_triangle_DEC_l90_90072


namespace phi_eq_pi_div_two_l90_90170

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.cos (x + ϕ)

theorem phi_eq_pi_div_two (ϕ : ℝ) (h1 : 0 ≤ ϕ) (h2 : ϕ ≤ π)
  (h3 : ∀ x : ℝ, f x ϕ = -f (-x) ϕ) : ϕ = π / 2 :=
sorry

end phi_eq_pi_div_two_l90_90170


namespace find_R_l90_90686

noncomputable def Q_and_R_polynomials : Prop :=
  ∃ (Q R : Polynomial ℤ), (R.degree < 2) ∧ (∀ z : ℂ, z ^ 2023 + 1 = (z ^ 2 - z + 1) * Q.eval z + R.eval z)

theorem find_R (Q R : Polynomial ℤ) (hR_deg : R.degree < 2)
  (h_eq : ∀ z : ℂ, z ^ 2023 + 1 = (z ^ 2 - z + 1) * Q.eval z + R.eval z) :
  R = Polynomial.C 1 + Polynomial.X :=
sorry

end find_R_l90_90686


namespace combination_10_3_eq_120_l90_90476

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90476


namespace right_triangle_area_l90_90264

theorem right_triangle_area (hypotenuse : ℝ) (thirty_deg_angle : ℝ) (right_angle : ℝ) 
  (h_hyp : hypotenuse = 20) 
  (h_angle : thirty_deg_angle = real.pi / 6)
  (h_right : right_angle = real.pi / 2) :
  ∃ (area : ℝ), area = 50 * real.sqrt 3 :=
by 
  sorry

end right_triangle_area_l90_90264


namespace wire_melting_time_l90_90253

noncomputable def alpha : ℝ := 1 / 235
noncomputable def rho_0 : ℝ := 1.65e-8
noncomputable def A : ℝ := 2e-6
noncomputable def I : ℕ := 120
noncomputable def T_0 : ℝ := 313
noncomputable def c : ℝ := 385
noncomputable def rho_r : ℝ := 8960
noncomputable def T_melt : ℝ := 1358

theorem wire_melting_time (α : ℝ) (ρ₀ : ℝ) (A : ℝ) (I : ℝ) (T₀ : ℝ) (c : ℝ) (ρ_r : ℝ) (T_melt : ℝ) :
  let ΔT := T_melt - T₀
  let t : ℝ := (1 / ((ρ₀ / c) * (I / A)^2)) * ΔT / (1 + α * ΔT) in
  abs (t - 23) < 2 :=
by
  sorry

end wire_melting_time_l90_90253


namespace sum_of_naturals_between_3_and_19_halved_l90_90922

theorem sum_of_naturals_between_3_and_19_halved :
  (∑ n in Finset.filter (λ A : ℕ, 3 < 2 * A ∧ 2 * A < 19) (Finset.range 10), n) = 44 :=
by
  sorry

end sum_of_naturals_between_3_and_19_halved_l90_90922


namespace rita_months_needed_l90_90075

def total_hours_required : ℕ := 2500
def backstroke_hours : ℕ := 75
def breaststroke_hours : ℕ := 25
def butterfly_hours : ℕ := 200
def hours_per_month : ℕ := 300

def total_completed_hours : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_hours_required - total_completed_hours
def months_needed (remaining_hours hours_per_month : ℕ) : ℕ := (remaining_hours + hours_per_month - 1) / hours_per_month

theorem rita_months_needed : months_needed remaining_hours hours_per_month = 8 := by
  -- Lean 4 proof goes here
  sorry

end rita_months_needed_l90_90075


namespace train_speed_l90_90270

theorem train_speed 
  (train_length : ℕ) 
  (platform_length : ℕ) 
  (time_sec : ℚ) 
  (conversion_factor : ℚ):
  train_length = 360 → platform_length = 150 → time_sec = 40.8 → conversion_factor = 3.6 →
  ((train_length + platform_length) / time_sec * conversion_factor = 45) := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end train_speed_l90_90270


namespace increasing_interval_sin_l90_90178

open Real

theorem increasing_interval_sin (k : ℤ) :
  ∃ x, (x ∈ set.Icc (k * π + π / 3) (k * π + 5 * π / 6)) → 
       ↑(k * π + π / 3) ≤ x ∧ x ≤ ↑(k * π + 5 * π / 6) :=
sorry

end increasing_interval_sin_l90_90178


namespace combination_10_3_eq_120_l90_90482

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90482


namespace sin_4_theta_l90_90050

noncomputable def E (θ : ℂ) : ℂ := complex.exp (complex.I * θ)

theorem sin_4_theta 
  (θ : ℂ) 
  (h : E θ = ((4 : ℂ) + complex.I * real.sqrt 3) / (5 : ℂ)) : 
  real.sin (4 * θ) = (208 * real.sqrt 3) / 625 :=
sorry

end sin_4_theta_l90_90050


namespace combination_10_3_eq_120_l90_90372

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90372


namespace probability_multiple_of_3_when_die_rolled_twice_l90_90255

theorem probability_multiple_of_3_when_die_rolled_twice :
  let total_outcomes := 36
  let favorable_outcomes := 12
  (12 / 36 : ℚ) = 1 / 3 :=
by
  sorry

end probability_multiple_of_3_when_die_rolled_twice_l90_90255


namespace binom_10_3_eq_120_l90_90319

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90319


namespace product_of_two_digit_numbers_is_not_five_digits_l90_90144

theorem product_of_two_digit_numbers_is_not_five_digits :
  ∀ (a b c d : ℕ), (10 ≤ 10 * a + b) → (10 * a + b ≤ 99) → (10 ≤ 10 * c + d) → (10 * c + d ≤ 99) → 
    (10 * a + b) * (10 * c + d) < 10000 :=
by
  intros a b c d H1 H2 H3 H4
  -- proof steps would go here
  sorry

end product_of_two_digit_numbers_is_not_five_digits_l90_90144


namespace smallest_k_for_p_squared_minus_k_divisible_by_15_l90_90679

theorem smallest_k_for_p_squared_minus_k_divisible_by_15 :
  ∀ (p : ℕ), nat.prime p ∧ (2023 ≤ ∀ q, nat.prime q → nDigits q = 2023 → q ≤ p) →
  ∃ (k : ℕ), 0 < k ∧ (p^2 - k) % 15 = 0 ∧ k = 15 :=
by
  sorry

end smallest_k_for_p_squared_minus_k_divisible_by_15_l90_90679


namespace hakimi_age_is_40_l90_90772

variable (H : ℕ)
variable (Jared_age : ℕ) (Molly_age : ℕ := 30)
variable (total_age : ℕ := 120)

theorem hakimi_age_is_40 (h1 : Jared_age = H + 10) (h2 : H + Jared_age + Molly_age = total_age) : H = 40 :=
by
  sorry

end hakimi_age_is_40_l90_90772


namespace line_through_A_with_equal_intercepts_l90_90536

theorem line_through_A_with_equal_intercepts (x y : ℝ) (A : ℝ × ℝ) (hx : A = (2, 1)) :
  (∃ k : ℝ, x + y = k ∧ x + y - 3 = 0) ∨ (x - 2 * y = 0) :=
sorry

end line_through_A_with_equal_intercepts_l90_90536


namespace square_tiles_count_l90_90247

theorem square_tiles_count (p s : ℕ) (h1 : p + s = 30) (h2 : 5 * p + 4 * s = 122) : s = 28 :=
sorry

end square_tiles_count_l90_90247


namespace binom_10_3_eq_120_l90_90463

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90463


namespace x_intercepts_closest_to_2900_l90_90760

noncomputable def num_x_intercepts_in_interval 
  (f: ℝ → ℝ) 
  (a b: ℝ) 
  (ha: 0 < a) 
  (hb: a < b) 
  (h: ∀ x, a < x ∧ x < b → f x = 0 → ∃ k: ℤ, f x = sin (1/x) ∧ 1/(k * real.pi) = x) : ℤ := 
begin
  sorry -- Proof here
end

theorem x_intercepts_closest_to_2900 : 
  num_x_intercepts_in_interval (λ x, sin (1/x)) 0.0001 0.001 (by norm_num) (by norm_num) _ ≈ 2900 := 
begin
  sorry -- Proof here
end

end x_intercepts_closest_to_2900_l90_90760


namespace binomial_10_3_l90_90491

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90491


namespace binomial_10_3_eq_120_l90_90425

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90425


namespace problem_statement_l90_90964

-- Given function definition and conditions
def f (x : ℝ) (k : ℝ) : ℝ := (Real.log x - k - 1) * x

-- Statement of the problem to be proved
theorem problem_statement (x1 x2 k : ℝ) (hx1 : x1 ≠ x2) (hfx : f x1 k = f x2 k) :
  x1 * x2 < Real.exp (2 * k) := 
sorry

end problem_statement_l90_90964


namespace solve_system_of_equations_l90_90156

theorem solve_system_of_equations (x y : ℝ) (h1 : x + y = 7) (h2 : 2 * x - y = 2) :
  x = 3 ∧ y = 4 :=
by
  sorry

end solve_system_of_equations_l90_90156


namespace minimum_students_per_bench_l90_90183

theorem minimum_students_per_bench (M : ℕ) (B : ℕ) (F : ℕ) (H1 : F = 4 * M) (H2 : M = 29) (H3 : B = 29) :
  ⌈(M + F) / B⌉ = 5 :=
by
  sorry

end minimum_students_per_bench_l90_90183


namespace electric_energy_consumption_l90_90105

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end electric_energy_consumption_l90_90105


namespace monotonic_intervals_a_eq_1_tangent_lines_reciprocal_slopes_l90_90598

def f (x a : ℝ) : ℝ := log x - a * (x - 1)
def g (x : ℝ) : ℝ := exp x

-- Part (1)
theorem monotonic_intervals_a_eq_1 :
  ∀ (x : ℝ), 0 < x → (f(x, 1)' > 0 ↔ 0 < x ∧ x < 1) ∧ (f(x, 1)' < 0 ↔ x > 1) :=
sorry

-- Part (2)
theorem tangent_lines_reciprocal_slopes :
  (∀ (x1 x2 : ℝ), let k1 := f(x1, a)' in let k2 := g(x2)' in k1 * k2 = 1 
  → a = 0 ∨ (exp(1) - 1) / exp(1) < a ∧ a < (exp(2) - 1) / exp(1)) :=
sorry

end monotonic_intervals_a_eq_1_tangent_lines_reciprocal_slopes_l90_90598


namespace cos_formula_of_tan_l90_90929

theorem cos_formula_of_tan (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi) :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 := 
  sorry

end cos_formula_of_tan_l90_90929


namespace hyperbola_eccentricity_range_l90_90606

theorem hyperbola_eccentricity_range (m : ℝ) (h₀ : m > 0) (h₁ : m < 1) :
    let e := (real.sqrt 2) / (real.sqrt (1 + m)) in 1 < e ∧ e < real.sqrt 2 :=
sorry

end hyperbola_eccentricity_range_l90_90606


namespace binomial_10_3_eq_120_l90_90429

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90429


namespace perimeter_triangle_AFB_l90_90853

-- Definitions based on the conditions in the problem
def point := (ℝ × ℝ)
def A : point := (0, 2)
def B : point := (0, 0)
def C : point := (2, 0)
def D : point := (2, 2)
def B' : point := (2, 4 / 3)
def F : point := (0, 0)

-- The length function to calculate distance between two points
def length (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Perimeter calculation of triangle
def perimeter (A B C : point) : ℝ :=
  length A B + length B C + length C A

-- The Lean statement to express the problem and expected result 
theorem perimeter_triangle_AFB' : 
  perimeter A F B' = 4 + 2 * real.sqrt 13 / 3 := 
sorry

end perimeter_triangle_AFB_l90_90853


namespace work_days_l90_90812

theorem work_days (A B C : ℝ) (h₁ : A + B = 1 / 15) (h₂ : C = 1 / 7.5) : 1 / (A + B + C) = 5 :=
by
  sorry

end work_days_l90_90812


namespace hyperbola_b_value_l90_90581

theorem hyperbola_b_value {m : ℝ} (h : m > 0) : 
  (∃ a b : ℝ, (b = sqrt 3 ∧ ∀ x y : ℝ, x^2 - m * y^2 = 3 * m → (x^2 / (3 * m) - y^2 / 3 = 1) )) :=
by
  sorry

end hyperbola_b_value_l90_90581


namespace maximum_value_of_expression_l90_90121

noncomputable def maxValue (x y z : ℝ) : ℝ :=
(x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2)

theorem maximum_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  maxValue x y z ≤ 243 / 16 :=
sorry

end maximum_value_of_expression_l90_90121


namespace min_a_l90_90737

-- Definitions and conditions used in the problem
def eqn (a x : ℝ) : Prop :=
  2 * sin (π - (π * x^2) / 12) * cos ((π / 6) * sqrt (9 - x^2)) + 1 =
  a + 2 * sin ((π / 6) * sqrt (9 - x^2)) * cos ((π * x^2) / 12)

-- Statement to prove the minimum value of a
theorem min_a : ∃ x : ℝ, eqn 2 x := sorry

end min_a_l90_90737


namespace paityn_blue_hats_is_24_l90_90704

variables (Paityn_red_hats Paityn_blue_hats Zola_red_hats Zola_blue_hats total_hats : ℕ)

def paityn_red_hats := 20
def zola_red_hats := 4 * paityn_red_hats / 5
def zola_blue_hats := 2 * Paityn_blue_hats
def total_hats := (paityn_red_hats + Paityn_blue_hats + zola_red_hats + Zola_blue_hats)

theorem paityn_blue_hats_is_24 
  (Paityn_blue_hats : ℕ) 
  (h1 : paityn_red_hats = 20)
  (h2 : zola_red_hats = 4 * paityn_red_hats / 5)
  (h3 : Zola_blue_hats = 2 * Paityn_blue_hats)
  (h4 : total_hats = 108)
  (h5 : 36 + 3 * Paityn_blue_hats = total_hats):
  Paityn_blue_hats = 24 :=
by
  sorry

end paityn_blue_hats_is_24_l90_90704


namespace question1_l90_90031

def sequence1 (a : ℕ → ℕ) : Prop :=
   a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 3 * a (n - 1) + 1

noncomputable def a_n1 (n : ℕ) : ℕ := (3^n - 1) / 2

theorem question1 (a : ℕ → ℕ) (n : ℕ) : sequence1 a → a n = a_n1 n :=
by
  sorry

end question1_l90_90031


namespace binomial_10_3_eq_120_l90_90421

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90421


namespace symmetry_line_probability_l90_90634

-- Define the main properties of the square grid
def grid_points : set (ℕ × ℕ) := { (x, y) | x ∈ {1, 2, ..., 11} ∧ y ∈ {1, 2, ..., 11} }

-- Define point P
def P : (ℕ × ℕ) := (6, 6)

-- We assume Q is chosen randomly from the remaining points in the grid
def is_point_Q (Q : (ℕ × ℕ)) : Prop :=
  Q ∈ grid_points ∧ Q ≠ P

-- Define a function to check if the line PQ is a line of symmetry for the square
def is_line_of_symmetry (P Q : (ℕ × ℕ)) : Prop :=
  (Q.1 = P.1 ∨ Q.2 = P.2 ∨ (Q.1 + Q.2 = P.1 + P.2) ∨ (Q.1 - Q.2 = P.1 - P.2))

-- Define the probability computation
def probability_line_of_symmetry : ℚ := 1 / 3

-- The theorem to be proven
theorem symmetry_line_probability :
  (∃ Q ∈ grid_points, Q ≠ P ∧ is_line_of_symmetry P Q) ↔ (1 / 3) :=
sorry

end symmetry_line_probability_l90_90634


namespace tank_fraction_after_adding_water_l90_90039

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (full_capacity : ℚ) 
  (added_water : ℚ) 
  (final_fraction : ℚ) 
  (h1 : initial_fraction = 3/4) 
  (h2 : full_capacity = 56) 
  (h3 : added_water = 7) 
  (h4 : final_fraction = (initial_fraction * full_capacity + added_water) / full_capacity) : 
  final_fraction = 7 / 8 := 
by 
  sorry

end tank_fraction_after_adding_water_l90_90039


namespace fewest_seats_occupied_l90_90775

def min_seats_occupied (N : ℕ) : ℕ :=
  if h : N % 4 = 0 then (N / 2) else (N / 2) + 1

theorem fewest_seats_occupied (N : ℕ) (h : N = 150) : min_seats_occupied N = 74 := by
  sorry

end fewest_seats_occupied_l90_90775


namespace largest_prime_2023_digits_l90_90681

theorem largest_prime_2023_digits:
  ∃ k : ℕ, k = 1 ∧ ∀ p : ℕ, Prime p ∧ digit_count p = 2023 → (p^2 - k) % 15 = 0 :=
sorry

end largest_prime_2023_digits_l90_90681


namespace find_sum_ab_l90_90056

theorem find_sum_ab (a b : ℝ) 
  (h : ∀ x : ℝ, f x = (a + sin x) / (2 + cos x) + b * tan x) 
  (hf_bounds : let f_max := (a + 1) / 3, f_min := (a - 1) / 3 in (f_max + f_min = 4)) 
  : a + b = 3 :=
sorry

end find_sum_ab_l90_90056


namespace max_cos_sin_sum_le_3_l90_90538

noncomputable def max_cos_sin_sum (θ : Fin 6 → ℝ) : ℝ :=
  ∑ i, Real.cos (θ i) * Real.sin (θ ((i + 1) % 6))

theorem max_cos_sin_sum_le_3 (θ : Fin 6 → ℝ) : max_cos_sin_sum θ ≤ 3 :=
by sorry

end max_cos_sin_sum_le_3_l90_90538


namespace revenue_fell_by_percentage_l90_90748

theorem revenue_fell_by_percentage :
  let old_revenue : ℝ := 69.0
  let new_revenue : ℝ := 52.0
  let percentage_decrease : ℝ := ((old_revenue - new_revenue) / old_revenue) * 100
  abs (percentage_decrease - 24.64) < 1e-2 :=
by
  sorry

end revenue_fell_by_percentage_l90_90748


namespace expansion_coefficient_value_l90_90928

theorem expansion_coefficient_value :
  let p := (λ x : ℝ, (1 - 2 * x) ^ 2016)
  ∃ (a : ℕ → ℝ), p = (λ x, ∑ i in finset.range 2017, a i * (x - 2) ^ i)
  → (a 1 - 2 * a 2 + 3 * a 3 - 4 * a 4 + ... + 2015 * a 2015 - 2016 * a 2016) = 4032 := 
sorry

end expansion_coefficient_value_l90_90928


namespace pyramid_base_side_length_l90_90164

theorem pyramid_base_side_length
  (area_lateral_face : ℝ)
  (slant_height : ℝ)
  (side_length : ℝ)
  (h1 : area_lateral_face = 144)
  (h2 : slant_height = 24)
  (h3 : 144 = 0.5 * side_length * 24) : 
  side_length = 12 :=
by
  sorry

end pyramid_base_side_length_l90_90164


namespace sum_of_possible_x_l90_90838

theorem sum_of_possible_x (x : ℝ) :
  let list := [6, 8, 6, 8, 10, 6, x] in
  let mean := (44 + x) / 7 in
  let mode := 6 in
  (∀ y : ℝ, x = y ∧ geomean list mode mean) → (mean ≠ mode) → 
  ∃ valid_xs : list ℝ, x ∈ valid_xs ∧ sum valid_xs = 30.67 := 
by
  sorry

def geomean (list : list ℝ) (mode mean : ℝ) : Prop := 
  let sorted_list := List.sort List.lte list in
  let len := List.length list in
  let median := if x <= 6 then 6 else if 6 < x ∧ x <= 8 then x else 8 in
  [mode, median, mean] in
  ∃ r : ℝ, r ≠ 1 ∧ (mode, median, mean).Geom r


end sum_of_possible_x_l90_90838


namespace min_value_l90_90176

theorem min_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (m n : ℝ) (h3 : m > 0) (h4 : n > 0) 
(h5 : m + 4 * n = 1) : 
  1 / m + 4 / n ≥ 25 :=
by
  sorry

end min_value_l90_90176


namespace find_eccentricity_of_hyperbola_l90_90845

noncomputable def hyperbola_eccentricity {a b : ℝ} (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ) 
  (h1 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h2 : (F2.1 - P.1)^2 + (F2.2 - P.2)^2 = 4 * b^2) 
  (h3 : ∥F2∥ = a) :
  ℝ :=
  sqrt (1 + (4/3)^2)

theorem find_eccentricity_of_hyperbola 
  {a b : ℝ} (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (h2 : (F2.1 - P.1)^2 + (F2.2 - P.2)^2 = 4 * b^2)
  (h3 : ∥F2∥ = a) :
  hyperbola_eccentricity P F1 F2 h1 h2 h3 = 5 / 3 := sorry

end find_eccentricity_of_hyperbola_l90_90845


namespace find_angle_A_find_side_c_l90_90950
noncomputable theory
open_locale classical

-- Definition for part 1
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : a = c * sin B) (h2 : b = c * sin A) (h_eq : a * sin B - sqrt 3 * b * cos A = 0) : 
  A = π / 3 := 
sorry

-- Definition for part 2
theorem find_side_c (a b c A : ℝ) (hA : A = π / 3) (ha : a = sqrt 7) (hb : b = 2) : 
  c = 3 := 
sorry

end find_angle_A_find_side_c_l90_90950


namespace find_smaller_number_l90_90195

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : x = 8 :=
by
  sorry

end find_smaller_number_l90_90195


namespace simplify_expression_l90_90154

theorem simplify_expression : (90 / 150) * (35 / 21) = 1 :=
by
  -- Insert proof here 
  sorry

end simplify_expression_l90_90154


namespace overall_percentage_increase_correct_l90_90661

def initial_salary : ℕ := 60
def first_raise_salary : ℕ := 90
def second_raise_salary : ℕ := 120
def gym_deduction : ℕ := 10

def final_salary : ℕ := second_raise_salary - gym_deduction
def salary_difference : ℕ := final_salary - initial_salary
def percentage_increase : ℚ := (salary_difference : ℚ) / initial_salary * 100

theorem overall_percentage_increase_correct :
  percentage_increase = 83.33 := by
  sorry

end overall_percentage_increase_correct_l90_90661


namespace prism_properties_sum_l90_90104

/-- Prove that the sum of the number of edges, corners, and faces of a rectangular box (prism) with dimensions 2 by 3 by 4 is 26. -/
theorem prism_properties_sum :
  let edges := 12
  let corners := 8
  let faces := 6
  edges + corners + faces = 26 := 
by
  -- Provided conditions and definitions
  let edges := 12
  let corners := 8
  let faces := 6
  -- Summing up these values
  exact rfl

end prism_properties_sum_l90_90104


namespace digit_sum_m_l90_90672

-- Define the function to calculate the sum of the digits
def digit_sum (n : ℕ) : ℕ := 
  n.digits.sum

-- Define the set 
def S : Set ℕ := {n | digit_sum n = 12 ∧ n < 10^7}

-- Define the cardinality of the set S
noncomputable def m : ℕ := (Set.card S)

-- State the proof problem
theorem digit_sum_m : digit_sum m = 26 := by
  sorry

end digit_sum_m_l90_90672


namespace binomial_coefficient_10_3_l90_90358

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90358


namespace isosceles_triangle_l90_90003

theorem isosceles_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a + b = (Real.tan (C / 2)) * (a * Real.tan A + b * Real.tan B)) :
  A = B := 
sorry

end isosceles_triangle_l90_90003


namespace sqrt_two_decimal_digits_nonzero_l90_90687

theorem sqrt_two_decimal_digits_nonzero :
  ∀ n, 1000001 ≤ n ∧ n ≤ 3000000 → (decimal_of (sqrt 2) n ≠ 0) :=
sorry

end sqrt_two_decimal_digits_nonzero_l90_90687


namespace units_digit_of_specific_product_is_9_l90_90805

def odd_and_between_100_and_200 (n : ℕ) : Prop :=
  n % 2 = 1 ∧ 100 ≤ n ∧ n ≤ 200

def not_ending_in_5 (n : ℕ) : Prop :=
  n % 10 ≠ 5

def units_digit_of_product (nums : list ℕ) : ℕ :=
  (nums.foldl (λ acc x => acc * x) 1) % 10

theorem units_digit_of_specific_product_is_9 :
  units_digit_of_product 
    (list.filter (λ n => odd_and_between_100_and_200 n ∧ not_ending_in_5 n) (list.range' 101 99)) = 9 := 
by 
  sorry

end units_digit_of_specific_product_is_9_l90_90805


namespace otimes_2_1_equals_3_l90_90902

namespace MathProof

-- Define the operation
def otimes (a b : ℝ) : ℝ := a^2 - b

-- The main theorem to prove
theorem otimes_2_1_equals_3 : otimes 2 1 = 3 :=
by
  -- Proof content not needed
  sorry

end MathProof

end otimes_2_1_equals_3_l90_90902


namespace binom_10_3_l90_90439

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90439


namespace ordered_pairs_count_l90_90904

theorem ordered_pairs_count : 
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ 100 * (a + b) = a * b - 100) = 18 := sorry

end ordered_pairs_count_l90_90904


namespace second_player_wins_optimal_play_l90_90787

-- Define the board size
def board_size : ℕ := 65

-- Define the condition that no more than two checkers can be placed in any row or column
def valid_game_state (board : array (array ℕ board_size) board_size) : Prop :=
  ∀ i < board_size, ∑ j, board[i][j] ≤ 2 ∧ ∑ j, board[j][i] ≤ 2

-- Define the main theorem to be proven
theorem second_player_wins_optimal_play :
  ∀ board : array (array ℕ board_size) board_size,
    valid_game_state board →
    -- Second player's winning strategy
    strategy second_player :=
sorry

end second_player_wins_optimal_play_l90_90787


namespace quadratic_inequality_solution_l90_90579

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 + 2 * x + 1 > 0) ↔ (a > 1) :=
by
  sorry

end quadratic_inequality_solution_l90_90579


namespace eq1_eq2_l90_90574

noncomputable def circle_eq_1 : Prop :=
  ∃ (C : ℝ → ℝ → Bool), 
  (C 1 (-2) = True ∧ C (-1) 4 = True) ∧
  (∀ (r : ℝ), (C r) = (r*r + (-2)*r - 9 = 0))

noncomputable def circle_eq_2 : Prop :=
  ∃ (C : ℝ → ℝ → Bool), 
  ∃ (center : ℝ × ℝ), (center.1, center.2) = (3, 2) ∧
  (C 1 (-2) = True ∧ C (-1) 4 = True) ∧ 2 * center.1 - center.2 - 4 = 0 ∧
  (∀ (r : ℝ), (C r) = ((r - 3) * (r - 3) + (r - 2) * (r - 2) - 20 = 0))

theorem eq1 : circle_eq_1 :=
by sorry

theorem eq2 : circle_eq_2 :=
by sorry

end eq1_eq2_l90_90574


namespace binom_10_3_eq_120_l90_90395

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90395


namespace extreme_value_range_of_a_l90_90016

noncomputable def f (a x : ℝ) : ℝ := (a - log x) / x

theorem extreme_value (a : ℝ) : ∃ x_ex : ℝ, f a x_ex = -real.exp (-(a + 1)) :=
by 
  let x_ex := real.exp (a + 1)
  use x_ex
  unfold f
  sorry

theorem range_of_a (a : ℝ) : (∃ x ∈ Ioo 0 real.exp(1), f a x = -1) → (a ≤ -1 ∨ 0 ≤ a ∧ a ≤ real.exp(1)) :=
by
  unfold f
  sorry

end extreme_value_range_of_a_l90_90016


namespace problem_a_gt_c_gt_b_l90_90118

noncomputable def a := 1.7^0.2
noncomputable def b := Real.log 0.9 / Real.log 2.1
noncomputable def c := 0.8^2.1

theorem problem_a_gt_c_gt_b : a > c ∧ c > b :=
  by
  -- Definitions
  let a := 1.7^0.2
  let b := Real.log 0.9 / Real.log 2.1
  let c := 0.8^2.1
  -- Proof goes here
  sorry

end problem_a_gt_c_gt_b_l90_90118


namespace solve_puzzle_l90_90734

-- Define the digits
def is_digit (n : ℕ) : Prop := n ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define unique digits
def distinct_digits (u h a : ℕ) : Prop := (is_digit u) ∧ (is_digit h) ∧ (is_digit a) ∧ (u ≠ h) ∧ (u ≠ a) ∧ (h ≠ a)

-- Define two-digit and three-digit numbers validity
def valid_two_digit (d1 d2 : ℕ) : Prop := d1 ≠ 0
def valid_three_digit (d1 d2 d3 : ℕ) : Prop := d1 ≠ 0

-- Define the Least Common Multiple
def lcm (m n : ℕ) : ℕ := Nat.lcm m n

-- Define the statement to be proved
theorem solve_puzzle (U H A : ℕ) (HU : distinct_digits U H A) (C1 : valid_three_digit U H A) : 
  U * 100 + H * 10 + A = lcm (U * 10 + H) (U * 10 + A) (H * 10 + A) :=
sorry

end solve_puzzle_l90_90734


namespace cards_given_l90_90138

def initial_cards : ℕ := 304
def remaining_cards : ℕ := 276
def given_cards : ℕ := initial_cards - remaining_cards

theorem cards_given :
  given_cards = 28 :=
by
  unfold given_cards
  unfold initial_cards
  unfold remaining_cards
  sorry

end cards_given_l90_90138


namespace combination_10_3_l90_90403

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90403


namespace probability_product_divisible_by_3_l90_90925

open Finset

def set := {4, 5, 6, 7, 8}

def product_divisible_by_3 (a b : ℕ) : Prop :=
  (a * b) % 3 = 0

theorem probability_product_divisible_by_3 :
  (card {ab : ℕ × ℕ | ab.1 ∈ set ∧ ab.2 ∈ set ∧ ab.1 < ab.2 ∧ product_divisible_by_3 ab.1 ab.2}.to_finset /
   card {ab : ℕ × ℕ | ab.1 ∈ set ∧ ab.2 ∈ set ∧ ab.1 < ab.2}.to_finset : ℚ) = 2 / 5 :=
sorry

end probability_product_divisible_by_3_l90_90925


namespace combination_10_3_eq_120_l90_90487

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90487


namespace evaluate_expression_l90_90912

theorem evaluate_expression (a b c : ℚ) 
  (h1 : c = b - 11) 
  (h2 : b = a + 3) 
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := 
sorry

end evaluate_expression_l90_90912


namespace binom_10_3_eq_120_l90_90473

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90473


namespace jack_total_cost_l90_90653

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end jack_total_cost_l90_90653


namespace factory_sample_size_l90_90832

noncomputable def sample_size (A B C : ℕ) (sample_A : ℕ) : ℕ :=
  let total_ratio := A + B + C
  let ratio_A := A / total_ratio
  sample_A / ratio_A

theorem factory_sample_size
  (A B C : ℕ) (h_ratio : A = 2 ∧ B = 3 ∧ C = 5)
  (sample_A : ℕ) (h_sample_A : sample_A = 16) :
  sample_size A B C sample_A = 80 :=
by
  simp [h_ratio, h_sample_A, sample_size]
  sorry

end factory_sample_size_l90_90832


namespace conjugate_quadrant_third_l90_90010

theorem conjugate_quadrant_third :
  let z := Complex.mk (Real.cos (2 * Real.pi / 3)) (Real.sin (Real.pi / 3))
  let z_conj := Complex.conj z
  (z_conj.re < 0) ∧ (z_conj.im < 0) := 
by
  -- We declare the variables and skip the proof as per the instructions
  let z := Complex.mk (Real.cos (2 * Real.pi / 3)) (Real.sin (Real.pi / 3))
  let z_conj := Complex.conj z
  sorry

end conjugate_quadrant_third_l90_90010


namespace range_of_a_l90_90608

variable (a : ℝ)

theorem range_of_a
  (h : ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) :
  a < -1 ∨ a > 1 :=
by {
  sorry
}

end range_of_a_l90_90608


namespace inequality_solution_l90_90550

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end inequality_solution_l90_90550


namespace percentage_difference_10_l90_90129

def machineA_production_rate : ℝ := 3.0000000000000044
def sprockets_produced : ℝ := 330
def time_difference : ℝ := 10

-- Let's define the time it takes for Machine B to produce 330 sprockets as variable t
variable (t : ℝ)

-- Machine A takes t + 10 hours to produce 330 sprockets
def machineA_production_time := t + 10

-- Given that Machine A produces 3.0000000000000044 sprockets per hour
def machineA_sprockets := machineA_production_rate * machineA_production_time

-- Given that Machine A produces 3.0000000000000044 sprockets/hour:
def machineB_production_time := 330 / machineA_production_rate - 10
def machineB_production_rate := sprockets_produced / machineB_production_time

-- Calculate the production rate difference percentage:
def percentage_difference := 
  ((machineB_production_rate - machineA_production_rate) / machineA_production_rate) * 100


theorem percentage_difference_10 :
  percentage_difference ≈ 10 := sorry

end percentage_difference_10_l90_90129


namespace ellipse_equation_correct_midpoint_coordinates_l90_90691

-- Define the conditions
variables (a b c : ℝ) (x y : ℝ)

def ellipse_equation : Prop := (a > b > 0) ∧ (2 * c = 6) ∧ 
                              (2 * c + 2 * a = 16) ∧ 
                              (a^2 = b^2 + c^2) 

-- Problem 1: Prove the equation of ellipse C is as given
theorem ellipse_equation_correct (h : ellipse_equation a b c) :
  (frac (x ^ 2) (25 : ℝ) + frac (y ^ 2) (16 : ℝ)) = 1 := sorry

-- Define additional conditions for Problem 2
def line_equation_through (x0 : ℝ) : Prop := 
  y = (4/5) * (x - x0)

-- Problem 2: Prove the coordinates of the midpoint of the line segment
theorem midpoint_coordinates (h : ellipse_equation a b c) 
  (hline : line_equation_through 3) :
  let x1 := ((3 : ℝ) / 2),
      y1 := -((6:ℝ) / 5) 
  in (x1 = (3 / 2)) ∧ (y1 = -(6 / 5)) := sorry

end ellipse_equation_correct_midpoint_coordinates_l90_90691


namespace tournament_committees_count_l90_90092

theorem tournament_committees_count :
  ∀ (teams : Fin 5 → Fin 7 → Prop) (female_only : Fin 5) (at_least_2_females : ∀ i, 2 ≤ Finset.card {x | teams i x}) (female_only_condition : ∀ j, teams female_only j → female_only = teams female_only j),
  ∃ (count : ℕ), count = 4 * ( Nat.choose 7 3 * (Nat.choose 7 2)^3 * 1 ) ∧ count = 1,296,540 :=
by
  intros teams female_only at_least_2_females female_only_condition
  use 4 * ( Nat.choose 7 3 * (Nat.choose 7 2) ^ 3 * 1 )
  have h_formula : 4 * (35 * 21 ^ 3 * 1) = 1,296,540 := by norm_num
  exact ⟨rfl, h_formula⟩

end tournament_committees_count_l90_90092


namespace no_integer_solutions_l90_90714

theorem no_integer_solutions (a b c d : ℤ) : 
  (a^2 - b = c^2) ∧ (b^2 - a = d^2) → (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) :=
begin
  sorry
end

end no_integer_solutions_l90_90714


namespace combination_10_3_l90_90406

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90406


namespace binomial_coefficient_10_3_l90_90360

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90360


namespace complex_sum_zero_l90_90959

open Complex

noncomputable def x : ℂ := (2 * I) / (1 - I)

def C (n k : ℕ) : ℂ := Complex.ofReal (nat.choose n k)

theorem complex_sum_zero : 
  (finset.range 2016).sum (λ k, C 2016 (k+1) * x^(k+1)) = 0 := 
by
  sorry

end complex_sum_zero_l90_90959


namespace find_a_for_max_y_l90_90610

theorem find_a_for_max_y (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 2 * (x - 1)^2 - 3 ≤ 15) →
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ a ∧ 2 * (x - 1)^2 - 3 = 15) →
  a = 4 :=
by sorry

end find_a_for_max_y_l90_90610


namespace binom_10_3_eq_120_l90_90470

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90470


namespace binomial_10_3_eq_120_l90_90420

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90420


namespace perimeter_proof_l90_90789

noncomputable def perimeter_larger_triangle
  (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ)
  (h1 : area_small = 10) (h2 : area_large = 250) (h3 : hypotenuse_small = 10) :
  ℝ :=
let scale_factor := sqrt (area_large / area_small),
    leg_small := (√(10 / 0.5))/sqrt 2,
    leg_large := leg_small * scale_factor in
  2 * leg_large + hypotenuse_small * scale_factor

theorem perimeter_proof :
  perimeter_larger_triangle 10 250 10
    (by rfl) (by rfl) (by rfl) = 20 * sqrt 5 + 50 * sqrt 2 :=
sorry

end perimeter_proof_l90_90789


namespace a_can_be_any_real_l90_90974

theorem a_can_be_any_real (a b c d e : ℝ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : e ≠ 0) :
  ∃ a : ℝ, true :=
by sorry

end a_can_be_any_real_l90_90974


namespace find_x_minus_y_l90_90197

variables (x y z : ℝ)

theorem find_x_minus_y (h1 : x - (y + z) = 19) (h2 : x - y - z = 7): x - y = 13 :=
by {
  sorry
}

end find_x_minus_y_l90_90197


namespace domain_of_f_l90_90513

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := sqrt (-8*x^2 - 10*x + 12)

-- Condition that the expression under the square root must be non-negative
def condition (x : ℝ) : Prop := -8 * x ^ 2 - 10 * x + 12 ≥ 0

-- Statement that the domain of f(x) is [-2, 3/4]
theorem domain_of_f : {x : ℝ | condition x} = set.Icc (-2 : ℝ) (3 / 4 : ℝ) :=
by
  sorry

end domain_of_f_l90_90513


namespace whisker_relationship_l90_90553

theorem whisker_relationship :
  let P_whiskers := 14
  let C_whiskers := 22
  (C_whiskers - P_whiskers = 8) ∧ (C_whiskers / P_whiskers = 11 / 7) :=
by
  let P_whiskers := 14
  let C_whiskers := 22
  have h1 : C_whiskers - P_whiskers = 8 := by sorry
  have h2 : C_whiskers / P_whiskers = 11 / 7 := by sorry
  exact And.intro h1 h2

end whisker_relationship_l90_90553


namespace part1_part2_l90_90021

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem part1 (a : ℝ) : (∀ x : ℝ, f x ≥ g a x) ↔ a ≤ -2 :=
  sorry

theorem part2 (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → h a x ≤ if a ≥ 0 then 3*a + 3 else if -3 ≤ a then a + 3 else 0) :=
  sorry

end part1_part2_l90_90021


namespace compare_ratios_l90_90892

theorem compare_ratios : (sqrt 2 / 3) < (1 / 2) :=
by 
  sorry

end compare_ratios_l90_90892


namespace sin_plus_cos_value_l90_90582

theorem sin_plus_cos_value (x : ℝ) (h1 : 3 * π / 4 < x) (h2 : x < π)
  (h3 : sin x * cos x = -1 / 4) : sin x + cos x = -√2 / 2 :=
by
  sorry

end sin_plus_cos_value_l90_90582


namespace quadratic_function_example_l90_90200

theorem quadratic_function_example (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : b = 0)
  (h2 : a < 0)
  (h3 : c < 0)
  (h4 : ∀ x, f x = a * x^2 + b * x + c) :
  (∃ (g : ℝ → ℝ), g = (λ x, - x^2 - 1)) :=
  sorry

end quadratic_function_example_l90_90200


namespace mod_congruence_l90_90544

theorem mod_congruence (N : ℕ) (hN : N > 1) (h1 : 69 % N = 90 % N) (h2 : 90 % N = 125 % N) : 81 % N = 4 := 
by {
    sorry
}

end mod_congruence_l90_90544


namespace isosceles_triangle_sides_l90_90216

theorem isosceles_triangle_sides (r R : ℝ) (a b c : ℝ) (h1 : r = 3 / 2) (h2 : R = 25 / 8)
  (h3 : a = c) (h4 : 5 = a) (h5 : 6 = b) : 
  ∃ a b c, a = 5 ∧ c = 5 ∧ b = 6 := by 
  sorry

end isosceles_triangle_sides_l90_90216


namespace cartesian_of_polar_range_of_intersection_l90_90025

section
variables {t : ℝ}

-- Define the parametric equations of the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (-1 - (sqrt 3) / 2 * t, sqrt 3 + 1 / 2 * t)

-- Define the polar equation of circle C
def polar_eq_C (ρ θ : ℝ) : Prop :=
  ρ = 4 * sin (θ - π / 6)

-- Define the Cartesian coordinate equation of circle C
def cartesian_eq_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 2 * (sqrt 3) * y = 0

-- Define the intersection point condition range
def range_of_z (z : ℝ) : Prop :=
  -2 ≤ z ∧ z ≤ 2

-- Theorem stating the equivalence of polar and Cartesian equations for circle C
theorem cartesian_of_polar :
  ∃ x y : ℝ, cartesian_eq_C x y :=
sorry

-- Theorem stating the range of the intersection point of the line and the circular region
theorem range_of_intersection :
  ∃ t : ℝ, range_of_z (-t) :=
sorry

end

end cartesian_of_polar_range_of_intersection_l90_90025


namespace second_derivative_at_neg5_l90_90017

def f (x : ℝ) : ℝ := 1 / x

theorem second_derivative_at_neg5 :
  (deriv^[2] f) (-5) = -2 / 125 := by 
  sorry

end second_derivative_at_neg5_l90_90017


namespace subsequence_not_exist_l90_90093

/-- Starting Sequence -/
def initial_sequence : List ℕ := [2, 0, 1, 7]

/-- Sequence generator following the rule -/
def next_digit (a b c d : ℕ) : ℕ :=
  (a + b + c + d) % 10

/-- Verification function to check if subsequence exists at some point -/
def contains_subsequence (seq subseq : List ℕ) : Prop :=
  subseq.length ≤ seq.length ∧ ∃ i, subseq = seq.drop i.take subseq.length

/-- Prove that the sequence defined will not contain "2016" starting from the 5th digit onward -/
theorem subsequence_not_exist (h : initial_sequence = [2, 0, 1, 7]) : 
  ¬ contains_subsequence (generate_sequence initial_sequence 10000) [2, 0, 1, 6] := sorry

noncomputable def generate_sequence : List ℕ → ℕ → List ℕ
| seq, 0     := seq
| seq, (n+1) := let new_digit := 
                         next_digit seq[-4] seq[-3] seq[-2] seq[-1]
                generate_sequence (seq ++ [new_digit]) n

end subsequence_not_exist_l90_90093


namespace sin_difference_identity_l90_90877

theorem sin_difference_identity 
  (α β : ℝ)
  (h1 : sin α - cos β = 3 / 4)
  (h2 : cos α + sin β = -2 / 5) : 
  sin (α - β) = 511 / 800 := 
sorry

end sin_difference_identity_l90_90877


namespace binomial_10_3_l90_90494

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90494


namespace quadratic_even_function_range_l90_90060

theorem quadratic_even_function_range
  (a c : ℝ)
  (h1 : 1 ≤ a + c)
  (h2 : a + c ≤ 2)
  (h3 : 3 ≤ 4 * a + c)
  (h4 : 4 * a + c ≤ 4) :
  (14 / 3) ≤ 9 * a + c ∧ 9 * a + c ≤ 9 :=
begin
  sorry
end

end quadratic_even_function_range_l90_90060


namespace factorial_divisible_by_power_of_two_iff_l90_90725

theorem factorial_divisible_by_power_of_two_iff (n : ℕ) :
  (nat.factorial n) % (2^(n-1)) = 0 ↔ ∃ k : ℕ, n = 2^k := 
by
  sorry

end factorial_divisible_by_power_of_two_iff_l90_90725


namespace smallest_a_value_l90_90738

theorem smallest_a_value :
  ∃ (a : ℝ), (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
    2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) ∧
    ∀ a' : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
      2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a' + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) →
      a ≤ a'
  := sorry

end smallest_a_value_l90_90738


namespace candy_box_contains_121_l90_90827

noncomputable def num_candies (n : ℕ) : Prop :=
  n ≤ 200 ∧
  (∀ k ∈ [2, 3, 4, 6], n % k = 1) ∧
  n % 11 = 0

theorem candy_box_contains_121 :
  ∃ n : ℕ, num_candies n ∧ n = 121 :=
by
  use 121
  unfold num_candies
  split
  · exact Nat.le_refl 121
  split
  · intros k hk
    -- Check that 121 % k = 1 for k in [2, 3, 4, 6]
    fin_cases hk <;> norm_num
  · norm_num

end candy_box_contains_121_l90_90827


namespace total_musicians_is_98_l90_90778

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end total_musicians_is_98_l90_90778


namespace binom_10_3_eq_120_l90_90312

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90312


namespace conjugate_of_given_complex_number_l90_90747

noncomputable def given_complex_number := (-1 + complex.I) / (2 + complex.I)

theorem conjugate_of_given_complex_number :
  complex.conj given_complex_number = -1 - complex.I :=
by
  sorry

end conjugate_of_given_complex_number_l90_90747


namespace mixed_number_calculation_l90_90304

theorem mixed_number_calculation :
  (481 + 1/6) + (265 + 1/12) + (904 + 1/20) - (184 + 29/30) - (160 + 41/42) - (703 + 55/56) = 603 + 3/8 := 
sorry

end mixed_number_calculation_l90_90304


namespace combinatorial_identity_inequality_solution_l90_90239

-- Part 1
theorem combinatorial_identity (n : ℕ) (h1 : n = 10) :
    Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n) = 496 :=
by
  sorry

-- Part 2
theorem inequality_solution (x : ℕ) (h1 : 2 ≤ x) (h2 : x ≤ 9) :
    Nat.factorial 9 / Nat.factorial (9 - x) > 
    6 * Nat.factorial 9 / Nat.factorial (11 - x) → 
    x ∈ {2, 3, 4, 5, 6, 7} :=
by
  sorry

end combinatorial_identity_inequality_solution_l90_90239


namespace sales_percentage_l90_90162

theorem sales_percentage (pens_sales pencils_sales notebooks_sales : ℕ) 
  (h1 : pens_sales = 25)
  (h2 : pencils_sales = 20)
  (h3 : notebooks_sales = 30) :
  100 - (pens_sales + pencils_sales + notebooks_sales) = 25 :=
by
  sorry

end sales_percentage_l90_90162


namespace expected_value_of_defective_draws_l90_90205

noncomputable def X : ℕ → probability → ℕ
| n p := n * p * (1 - p) 

theorem expected_value_of_defective_draws :
  let n := 4
  let p := 1 / 4
  D(X n p) = 3 / 4 := sorry

end expected_value_of_defective_draws_l90_90205


namespace infinite_integer_solutions_iff_l90_90153

theorem infinite_integer_solutions_iff
  (a b c d : ℤ) :
  (∃ inf_int_sol : (ℤ → ℤ) → Prop, ∀ (f : (ℤ → ℤ)), inf_int_sol f) ↔ (a^2 - 4*b = c^2 - 4*d) :=
by
  sorry

end infinite_integer_solutions_iff_l90_90153


namespace units_digit_of_2_exp_2010_l90_90702

def units_digit_cycle : List ℕ := [2, 4, 8, 6]

def units_digit (n : ℕ) : ℕ :=
  units_digit_cycle[(n - 1) % 4]

theorem units_digit_of_2_exp_2010 : units_digit 2010 = 4 :=
by
  -- Proof is omitted
  sorry

end units_digit_of_2_exp_2010_l90_90702


namespace total_musicians_count_l90_90779

-- Define the given conditions
def orchestra_males := 11
def orchestra_females := 12
def choir_males := 12
def choir_females := 17

-- Total number of musicians in the orchestra
def orchestra_musicians := orchestra_males + orchestra_females

-- Total number of musicians in the band
def band_musicians := 2 * orchestra_musicians

-- Total number of musicians in the choir
def choir_musicians := choir_males + choir_females

-- Total number of musicians in the orchestra, band, and choir
def total_musicians := orchestra_musicians + band_musicians + choir_musicians

-- The theorem to prove
theorem total_musicians_count : total_musicians = 98 :=
by
  -- Lean proof part goes here.
  sorry

end total_musicians_count_l90_90779


namespace binom_10_3_eq_120_l90_90394

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90394


namespace area_of_2014th_triangle_l90_90134

/-- Given a right triangle with sides 3, 4, and 5. We iteratively draw smaller triangles by joining the midpoints of the sides of the previous triangle.
    Prove that the area of the 2014-th triangle is (1/4)^2013 * 3/2. -/
theorem area_of_2014th_triangle :
  let triangle_area (n : ℕ) : ℚ := (1 / 4 : ℚ) ^ n * (3 / 2 : ℚ)
  in triangle_area 2013 = (1 / 4 : ℚ) ^ 2013 * (3 / 2 : ℚ) :=
by
  let triangle_area (n : ℕ) : ℚ := (1 / 4 : ℚ) ^ n * (3 / 2 : ℚ)
  -- We need to prove that the area of the 2014-th triangle (which is triangle_area 2013) is:
  have h : triangle_area 2013 = (1 / 4 : ℚ) ^ 2013 * (3 / 2 : ℚ),
  from rfl,
  exact h

end area_of_2014th_triangle_l90_90134


namespace bell_rings_5_times_l90_90101

-- Define the conditions
def class_starts (n : ℕ) : Prop := n = 1
def class_ends (n : ℕ) : Prop := n = 1
def break_time (min : ℕ) : Prop := min = 15
def classes_on_monday : List String := ["Maths", "History", "Geography", "Science", "Music"]
def current_class (class : String) : Prop := class = "Geography"

-- Define the problem statement
theorem bell_rings_5_times :
  ∀ (n m : ℕ) (c : String), class_starts n → class_ends n → break_time m → current_class c → (n * 2) + (n - 1) = 5 :=
by
  sorry

end bell_rings_5_times_l90_90101


namespace time_to_clear_l90_90791

def length_train1 := 121 -- in meters
def length_train2 := 153 -- in meters
def speed_train1 := 80 * 1000 / 3600 -- converting km/h to meters/s
def speed_train2 := 65 * 1000 / 3600 -- converting km/h to meters/s

def total_distance := length_train1 + length_train2
def relative_speed := speed_train1 + speed_train2

theorem time_to_clear : 
  (total_distance / relative_speed : ℝ) = 6.80 :=
by
  sorry

end time_to_clear_l90_90791


namespace combination_10_3_l90_90407

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90407


namespace triangles_similar_l90_90123

variables (A B C G P Q E F : Type) [EuclideanGeometry A B C G P Q E F]

-- Assuming G is the centroid of ΔABC.
def is_centroid (G : A) (A B C : A) : Prop := sorry

-- A line through G parallel to BC intersects AB at P and AC at Q.
def line_parallel (G : A) (BC : A) (AB : A) (AC : A) (P Q : A) : Prop := sorry

-- BQ intersects GC at E.
def intersects_at (BQ GC E : A) : Prop := sorry

-- CP intersects GB at F.
def intersects_at' (CP GB F : A) : Prop := sorry

theorem triangles_similar 
  (h1 : is_centroid G A B C)
  (h2 : line_parallel G (B C) (A B) (A C) P Q)
  (h3 : intersects_at (B Q) (G C) E)
  (h4 : intersects_at' (C P) (G B) F) :
  similar (triangle A B C) (triangle D E F) :=
sorry

end triangles_similar_l90_90123


namespace perimeter_of_cross_section_l90_90750

variable (a : ℝ) (P : ℝ)

-- The main theorem to be proven
theorem perimeter_of_cross_section (h1 : a > 0) (h2 : P = perimeter_cross_section_through_vertex a) : 
  2 * a < P ∧ P ≤ 3 * a :=
by
  sorry

-- Definition placeholder to represent the perimeter calculation
def perimeter_cross_section_through_vertex (a : ℝ) : ℝ := 
  sorry

end perimeter_of_cross_section_l90_90750


namespace binomial_10_3_l90_90503

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90503


namespace total_amount_saved_and_discount_l90_90266

theorem total_amount_saved_and_discount (marked_price_bag sold_price_bag: ℝ) 
  (marked_price_shoes sold_price_shoes: ℝ) 
  (marked_price_jacket sold_price_jacket: ℝ) : 
  (marked_price_bag = 125) -> (sold_price_bag = 120) ->
  (marked_price_shoes = 80) -> (sold_price_shoes = 70) ->
  (marked_price_jacket = 150) -> (sold_price_jacket = 130) ->
  let total_amount_saved := (marked_price_bag - sold_price_bag) + (marked_price_shoes - sold_price_shoes) + (marked_price_jacket - sold_price_jacket)
  in total_amount_saved = 35 ∧ 
  let total_marked_price := marked_price_bag + marked_price_shoes + marked_price_jacket
      overall_discount_rate := (total_amount_saved / total_marked_price) * 100
  in overall_discount_rate ≈ 9.86 :=
by
  intros
  sorry

end total_amount_saved_and_discount_l90_90266


namespace minimum_votes_for_tall_to_win_l90_90090

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l90_90090


namespace sum_homothety_ratios_equals_one_l90_90280

-- Definitions from the conditions
structure Triangle :=
(base : ℝ)
(height : ℝ)

def is_homothetic (t1 t2 : Triangle) (r : ℝ) : Prop :=
t1.base = r * t2.base ∧ t1.height = r * t2.height

-- The mathematically equivalent proof problem
theorem sum_homothety_ratios_equals_one
  (ABC : Triangle)
  (triangles : list (Triangle × ℝ))
  (h₁ : ∀ (t : Triangle) (r : ℝ), (t, r) ∈ triangles → is_homothetic t ABC r)
  (h₂ : ∀ (r : ℝ), 0 < (list.sum (list.filter_map (λ x, if x.2 = r then some x.2 else none) triangles)) ∨ 0 < -(list.sum (list.filter_map (λ x, if x.2 = r then some x.2 else none) triangles)))
  : list.sum (list.map prod.snd triangles) = 1 :=
sorry

end sum_homothety_ratios_equals_one_l90_90280


namespace marge_final_plants_l90_90696

-- Definitions corresponding to the conditions
def seeds_planted := 23
def seeds_never_grew := 5
def plants_grew := seeds_planted - seeds_never_grew
def plants_eaten := plants_grew / 3
def uneaten_plants := plants_grew - plants_eaten
def plants_strangled := uneaten_plants / 3
def survived_plants := uneaten_plants - plants_strangled
def effective_addition := 1

-- The main statement we need to prove
theorem marge_final_plants : 
  (plants_grew - plants_eaten - plants_strangled + effective_addition) = 9 := 
by
  sorry

end marge_final_plants_l90_90696


namespace binomial_10_3_eq_120_l90_90454

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90454


namespace compound_interest_calculation_l90_90053

noncomputable def compoundInterest (P r t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simpleInterest (P r t : ℝ) : ℝ :=
  P * r * t

theorem compound_interest_calculation :
  ∃ P : ℝ, simpleInterest P 0.10 2 = 600 ∧ compoundInterest P 0.10 2 = 630 :=
by
  sorry

end compound_interest_calculation_l90_90053


namespace bob_got_15_candies_l90_90875

-- Define the problem conditions
def bob_neighbor_sam : Prop := true -- Bob is Sam's next door neighbor
def bob_accompany_sam_home : Prop := true -- Bob decided to accompany Sam home

def bob_share_chewing_gums : ℕ := 15 -- Bob's share of chewing gums
def bob_share_chocolate_bars : ℕ := 20 -- Bob's share of chocolate bars
def bob_share_candies : ℕ := 15 -- Bob's share of assorted candies

-- Define the main assertion
theorem bob_got_15_candies : bob_share_candies = 15 := 
by sorry

end bob_got_15_candies_l90_90875


namespace range_of_a_l90_90920

theorem range_of_a (a : ℝ) : (∀ x ∈ set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 4 :=
sorry

end range_of_a_l90_90920


namespace odd_function_minimum_interval_l90_90052

theorem odd_function_minimum_interval (f : ℝ → ℝ) (a b : ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_max : ∃ x ∈ set.Icc a b, ∀ y ∈ set.Icc a b, f y ≤ f x) :
  ∃ x ∈ set.Icc (-b) (-a), ∀ y ∈ set.Icc (-b) (-a), f x ≤ f y :=
sorry

end odd_function_minimum_interval_l90_90052


namespace painter_total_cost_is_119_l90_90781

def arithmetic_seq_term (a d n : ℕ) : ℕ := a + (n - 1) * d

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

def sum_costs (start diff count : ℕ) : ℕ :=
  (List.range count).map (λ i => count_digits (arithmetic_seq_term start diff (i + 1))).sum

def total_cost (side_count diff start_south start_north : ℕ) : ℕ :=
  sum_costs start_south diff side_count + sum_costs start_north diff side_count

theorem painter_total_cost_is_119 :
  total_cost 25 7 5 2 = 119 :=
by
  sorry

end painter_total_cost_is_119_l90_90781


namespace hakimi_age_is_40_l90_90771

variable (H : ℕ)
variable (Jared_age : ℕ) (Molly_age : ℕ := 30)
variable (total_age : ℕ := 120)

theorem hakimi_age_is_40 (h1 : Jared_age = H + 10) (h2 : H + Jared_age + Molly_age = total_age) : H = 40 :=
by
  sorry

end hakimi_age_is_40_l90_90771


namespace area_relationship_l90_90005

noncomputable section

variables {A B C D E F : Type}
variables {AB AC : Set}
variables (S1 S2 S3 S4 : ℝ)

-- Defining a type for points on sides AB and AC
class PointOnSide (p : Type) (side : Set) extends Set p

-- Definitions of the conditions
variables [PointOnSide D AB] [PointOnSide E AC]
variables (intersection_BE_CD : F)
variables (S_quadrilateral_EdF : ℝ) (S_triangle_BDF : ℝ) (S_triangle_BCF : ℝ) (S_triangle_CEF : ℝ)

def Areas (S1 S2 S3 S4 : ℝ) :=
  S1 = S_quadrilateral_EdF ∧
  S2 = S_triangle_BDF ∧
  S3 = S_triangle_BCF ∧
  S4 = S_triangle_CEF

theorem area_relationship {S1 S2 S3 S4 : ℝ} (h : Areas S1 S2 S3 S4) :
  S1 * S3 > S2 * S4 :=
sorry

end area_relationship_l90_90005


namespace geometric_sequence_S6_l90_90938

variable (a : ℕ → ℝ) -- represents the geometric sequence

noncomputable def S (n : ℕ) : ℝ :=
if n = 0 then 0 else ((a 0) * (1 - (a 1 / a 0) ^ n)) / (1 - a 1 / a 0)

theorem geometric_sequence_S6 (h : ∀ n, a n = (a 0) * (a 1 / a 0) ^ n) :
  S a 2 = 6 ∧ S a 4 = 18 → S a 6 = 42 := 
by 
  intros h1
  sorry

end geometric_sequence_S6_l90_90938


namespace binom_10_3_eq_120_l90_90321

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90321


namespace collinear_F_H1_H2_l90_90742

noncomputable
def are_collinear (A B C : Point) : Prop := sorry -- Define collinearity

variable {A B C D E F H1 H2 : Point}

-- Conditions of the problem
variable [cyclic_quadrilateral ABCD]
variable [intersection E (line AC) (line BD)]
variable [intersection F (line AB) (line CD)]
variable [orthocenter H1 (triangle EAD)]
variable [orthocenter H2 (triangle EBC)]

-- The collinearity of F, H1, and H2
theorem collinear_F_H1_H2 : are_collinear F H1 H2 := 
sorry

end collinear_F_H1_H2_l90_90742


namespace f_monotonicity_a_zero_f_less_than_zero_for_x_greater_than_one_l90_90603

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x * log x - (x - 1) * (a * x - a + 1)

-- Question 1: Prove the monotonicity of f when a = 0
theorem f_monotonicity_a_zero :
  ∀ x : ℝ, (0 < x ∧ x < 1 → f x 0 < f (x + 1) 0) 
    ∧ (1 < x → f (x - 1) 0 < f x 0) := sorry

-- Question 2: Find the range of values for a such that f(x) < 0 for x > 1
theorem f_less_than_zero_for_x_greater_than_one (a : ℝ) : 
  (∀ x : ℝ, 1 < x → f x a < 0) → a ≥ 1 / 2 := sorry

end f_monotonicity_a_zero_f_less_than_zero_for_x_greater_than_one_l90_90603


namespace binom_10_3_l90_90444

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90444


namespace binom_10_3_eq_120_l90_90320

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90320


namespace points_satisfying_inequality_l90_90807

theorem points_satisfying_inequality (x y : ℝ) :
  ( ( (x * y + 1) / (x + y) )^2 < 1) ↔ 
  ( (-1 < x ∧ x < 1) ∧ (y < -1 ∨ y > 1) ) ∨ 
  ( (x < -1 ∨ x > 1) ∧ (-1 < y ∧ y < 1) ) := 
sorry

end points_satisfying_inequality_l90_90807


namespace min_terminals_three_connected_windmill_l90_90830

-- Definitions based on conditions a)
def three_connected (G : Type) [graph G] := 
  -- Condition for Three-Connected Network
  ∀ (u v w : G), (u ≠ v ∧ u ≠ w ∧ v ≠ w) → 
  (∃ (x y : G), x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ x ≠ w ∧ y ≠ u ∧ y ≠ v ∧ y ≠ w ∧ x ~ y)

def windmill (G : Type) [graph G] (n : ℕ) :=
  -- Definition for n-bladed windmill
  ∃ O ∈ G, ∀ i ∈ fin n, ∃ A_i B_i ∈ G, (A_i ~ B_i) ∧ (A_i ~ O) ∧ (B_i ~ O)

-- Main Theorem statement (lean only)
theorem min_terminals_three_connected_windmill (n : ℕ) : 
  ∃ (m : ℕ), (∀ G : Type, [graph G], 
    three_connected G → 
    windmill (subgraph G) n → 
    fincard G ≥ m) ∧ 
  (∀ G : Type, [graph G],
    three_connected G →
    (¬ windmill (subgraph G) n →
    fincard G < m)) :=
by
  have f : ℕ → ℕ
  | 1 => 6
  | n => 4 * n + 1

  use f n
  sorry -- proof to be provided


end min_terminals_three_connected_windmill_l90_90830


namespace rectangle_perimeter_l90_90627

theorem rectangle_perimeter 
  (w : ℝ) (l : ℝ) (hw : w = Real.sqrt 3) (hl : l = Real.sqrt 6) : 
  2 * (w + l) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := 
by 
  sorry

end rectangle_perimeter_l90_90627


namespace train_length_l90_90861

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (speed_mps : ℝ) (distance_m : ℝ) :
  speed_kmph = 120 → time_sec = 15 → speed_mps = speed_kmph * (1000 / 3600) →
  distance_m = speed_mps * time_sec → distance_m ≈ 500 :=
by
  intro h_speed_kmph h_time_sec h_speed_mps h_distance_m
  simp [h_speed_kmph, h_time_sec, h_speed_mps, h_distance_m]
  sorry

end train_length_l90_90861


namespace remainder_sum_div_40_l90_90131

variable (k m n : ℤ)
variables (a b c : ℤ)
variable (h1 : a % 80 = 75)
variable (h2 : b % 120 = 115)
variable (h3 : c % 160 = 155)

theorem remainder_sum_div_40 : (a + b + c) % 40 = 25 :=
by
  -- Use sorry as we are not required to fill in the proof
  sorry

end remainder_sum_div_40_l90_90131


namespace quadratic_floor_eq_more_than_100_roots_l90_90869

open Int

theorem quadratic_floor_eq_more_than_100_roots (p q : ℤ) (h : p ≠ 0) :
  ∃ (S : Finset ℤ), S.card > 100 ∧ ∀ x ∈ S, ⌊(x : ℝ) ^ 2⌋ + p * x + q = 0 :=
by
  sorry

end quadratic_floor_eq_more_than_100_roots_l90_90869


namespace problem1_part1_problem1_part2_problem2_value_l90_90555

variable (a b : ℝ)

def A := 4 * a^2 * b - 3 * a * b + b^2
def B := a^2 - 3 * a^2 * b + 3 * a * b - b^2

theorem problem1_part1 : A + B = a^2 + a^2 * b :=
by
  unfold A B
  sorry

theorem problem1_part2 : 3 * A + 4 * B = 4 * a^2 + 3 * a * b - b^2 :=
by
  unfold A B
  sorry

theorem problem2_value : A - B = - 63 / 8 :=
by
  have a := 2
  have b := -1 / 4
  unfold A B
  sorry

end problem1_part1_problem1_part2_problem2_value_l90_90555


namespace length_of_train_is_correct_l90_90860

noncomputable def speed_kmh := 30 
noncomputable def time_s := 9 
noncomputable def speed_ms := (speed_kmh * 1000) / 3600 
noncomputable def length_of_train := speed_ms * time_s

theorem length_of_train_is_correct : length_of_train = 75 := 
by 
  sorry

end length_of_train_is_correct_l90_90860


namespace combination_10_3_eq_120_l90_90480

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90480


namespace even_positive_factors_l90_90978

theorem even_positive_factors (n : ℕ) (h : n = 2^4 * 3^2 * 5^2 * 7) : 
  ∃ k : ℕ, k = 72 ∧ (∀ d : ℕ, d ∣ n → even d → ∃ m : ℕ, m = k) := 
by 
  sorry

end even_positive_factors_l90_90978


namespace difference_between_shares_l90_90743

def investment_months (amount : ℕ) (months : ℕ) : ℕ :=
  amount * months

def ratio (investment_months : ℕ) (total_investment_months : ℕ) : ℚ :=
  investment_months / total_investment_months

def profit_share (ratio : ℚ) (total_profit : ℝ) : ℝ :=
  ratio * total_profit

theorem difference_between_shares :
  let suresh_investment := 18000
  let rohan_investment := 12000
  let sudhir_investment := 9000
  let suresh_months := 12
  let rohan_months := 9
  let sudhir_months := 8
  let total_profit := 3795
  let suresh_investment_months := investment_months suresh_investment suresh_months
  let rohan_investment_months := investment_months rohan_investment rohan_months
  let sudhir_investment_months := investment_months sudhir_investment sudhir_months
  let total_investment_months := suresh_investment_months + rohan_investment_months + sudhir_investment_months
  let suresh_ratio := ratio suresh_investment_months total_investment_months
  let rohan_ratio := ratio rohan_investment_months total_investment_months
  let sudhir_ratio := ratio sudhir_investment_months total_investment_months
  let rohan_share := profit_share rohan_ratio total_profit
  let sudhir_share := profit_share sudhir_ratio total_profit
  rohan_share - sudhir_share = 345 :=
by
  sorry

end difference_between_shares_l90_90743


namespace problem1_problem2_l90_90931

variables {a x y : ℝ}

theorem problem1 (h1 : a^x = 2) (h2 : a^y = 3) : a^(x + y) = 6 :=
sorry

theorem problem2 (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x - 3 * y) = 4 / 27 :=
sorry

end problem1_problem2_l90_90931


namespace combination_10_3_eq_120_l90_90479

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90479


namespace minimum_votes_for_tall_l90_90082

theorem minimum_votes_for_tall (voters : ℕ) (districts : ℕ) (precincts : ℕ) (precinct_voters : ℕ)
  (vote_majority_per_precinct : ℕ → ℕ) (precinct_majority_per_district : ℕ → ℕ) (district_majority_to_win : ℕ) :
  voters = 135 ∧ districts = 5 ∧ precincts = 9 ∧ precinct_voters = 3 ∧
  (∀ p, vote_majority_per_precinct p = 2) ∧
  (∀ d, precinct_majority_per_district d = 5) ∧
  district_majority_to_win = 3 ∧ 
  tall_won : 
  ∃ min_votes, min_votes = 30 :=
by
  sorry

end minimum_votes_for_tall_l90_90082


namespace math_proof_problem_l90_90019

-- Definitions and conditions
def f_k (a : ℝ) (k : ℤ) (x : ℝ) := a^x + k * (a^(-x))

def problem_1 (a : ℝ) :=
  a > 0 ∧ a ≠ 1 ∧ (a^(1/2) + a^(-1/2) = 3) →
  (f_k a 1 2 = 47)

def problem_2 (a : ℝ) :=
  0 < a ∧ a < 1 ∧ (∀ x : ℝ, f_k a (-1) x = -f_k a (-1) (-x)) →
  (∀ x ∈ [1, 3], ∃ (m : ℝ), m < 6/7 ∧ f_k a (-1) (m * x^2 - m * x - 1) + f_k a (-1) (m - 5) > 0)

-- Problem statement
theorem math_proof_problem
: ∀ a : ℝ, problem_1 a ∧ problem_2 a :=
by
  sorry

end math_proof_problem_l90_90019


namespace perimeter_equal_l90_90179

theorem perimeter_equal (x : ℕ) (hx : x = 4)
    (side_square : ℕ := x + 2) 
    (side_triangle : ℕ := 2 * x) 
    (perimeter_square : ℕ := 4 * side_square)
    (perimeter_triangle : ℕ := 3 * side_triangle) :
    perimeter_square = perimeter_triangle :=
by
    -- Given x = 4
    -- Calculate side lengths
    -- side_square = x + 2 = 4 + 2 = 6
    -- side_triangle = 2 * x = 2 * 4 = 8
    -- Calculate perimeters
    -- perimeter_square = 4 * side_square = 4 * 6 = 24
    -- perimeter_triangle = 3 * side_triangle = 3 * 8 = 24
    -- Therefore, perimeter_square = perimeter_triangle = 24
    sorry

end perimeter_equal_l90_90179


namespace population_increase_difference_l90_90642

noncomputable def births_per_day : ℝ := 24 / 6
noncomputable def deaths_per_day : ℝ := 24 / 16
noncomputable def net_increase_per_day : ℝ := births_per_day - deaths_per_day
noncomputable def annual_increase_regular_year : ℝ := net_increase_per_day * 365
noncomputable def annual_increase_leap_year : ℝ := net_increase_per_day * 366

theorem population_increase_difference :
  annual_increase_leap_year - annual_increase_regular_year = 2.5 :=
by {
  sorry
}

end population_increase_difference_l90_90642


namespace inequality_sum_l90_90543

noncomputable def sum (f : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, f i

theorem inequality_sum (n : ℕ) (a b : ℕ → ℝ) (h_pos : ∀ k, 1 ≤ k ∧ k ≤ n → a k > 0 ∧ b k > 0) :
  (sum (λ k, (a k * b k) / (a k + b k)) n) ≤ ((sum a n) * (sum b n)) / (sum (λ k, a k + b k) n) :=
sorry

end inequality_sum_l90_90543


namespace fence_perimeter_l90_90206

noncomputable def posts (n : ℕ) := 36
noncomputable def space_between_posts (d : ℕ) := 6
noncomputable def length_is_twice_width (l w : ℕ) := l = 2 * w

theorem fence_perimeter (n d w l perimeter : ℕ)
  (h1 : posts n = 36)
  (h2 : space_between_posts d = 6)
  (h3 : length_is_twice_width l w)
  : perimeter = 216 :=
sorry

end fence_perimeter_l90_90206


namespace unique_intersection_l90_90014

def line1 (x y : ℝ) : Prop := 3 * x - 2 * y - 9 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = -1

theorem unique_intersection : ∃! p : ℝ × ℝ, 
                             (line1 p.1 p.2) ∧ 
                             (line2 p.1 p.2) ∧ 
                             (line3 p.1) ∧ 
                             (line4 p.2) ∧ 
                             p = (3, -1) :=
by
  sorry

end unique_intersection_l90_90014


namespace ratio_product_even_odd_composite_l90_90883

theorem ratio_product_even_odd_composite :
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = (2^10) / (3^6 * 5^2 * 7) :=
by
  sorry

end ratio_product_even_odd_composite_l90_90883


namespace regression_subsidies_p_range_l90_90639

def graduates := [3, 4, 5, 6] -- Graduates in thousands
def entrepreneurs := [0.1, 0.2, 0.4, 0.5] -- Entrepreneurs in thousands

def mean (l : List ℝ) : ℝ := l.sum / l.length -- Definition for mean calculation

def linear_regression (x y : List ℝ) : ℝ × ℝ :=
  let n := x.length
  let x_mean := mean x
  let y_mean := mean y
  let numerator := (List.zip x y).sum (fun (xi, yi) => xi * yi) - n * x_mean * y_mean
  let denominator := x.sum (fun xi => xi^2) - n * x_mean^2
  let b := numerator / denominator
  let a := y_mean - b * x_mean
  (a, b)

-- Definitions of linear regression parameters based on given data
def a, b : ℝ := linear_regression graduates entrepreneurs

-- Total subsidies for University E
def subsidies_for_e(univ_e_graduates : ℝ) : ℝ :=
  let y := b * univ_e_graduates + a -- linear regression prediction
  y * 10000 -- converting to yuan (from thousands of yuan)

-- Definition for value range of p
def p_range(upper_bound : ℝ := 1.4 / 100) : Set ℝ :=
  {p : ℝ | (1/2) < p ∧ p <= 4/5 ∧ 0 <= 3 * p - 1 <= upper_bound}

-- Statement to be proved
theorem regression_subsidies_p_range :
  (linear_regression graduates entrepreneurs = (-0.33, 0.14)) ∧
  (subsidies_for_e 7 = 650 * 1000) ∧
  p_range 1.4 = {p : ℝ | (1/2) < p ∧ p <= 4/5} :=
by sorry

end regression_subsidies_p_range_l90_90639


namespace rectangular_and_general_equation_maximum_distance_l90_90027

noncomputable def rectangular_equation_C1 (ρ θ : ℝ) : Prop :=
  ρ = 4 * real.cos θ

def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (1 - (2 * real.sqrt 5) / 5 * t, 1 + (real.sqrt 5) / 5 * t)

noncomputable def parametric_curve_C2 (α : ℝ) : ℝ × ℝ :=
  (2 * real.cos α, real.sin α)

theorem rectangular_and_general_equation (ρ θ : ℝ) (t : ℝ) : 
  (∃ x y, rectangular_equation_C1 ρ θ → x^2 + y^2 - 4x = 0) ∧
  (∃ x y, parametric_line_l t = (x, y) → x + 2 * y - 3 = 0) :=
by
  sorry

theorem maximum_distance (α : ℝ) (P Q M : ℝ × ℝ) : 
  (P = (2, 2)) →
  (Q = parametric_curve_C2 α) →
  (M = ((1 + real.cos α), (1 + 1/2 * real.sin α))) →
  ∃ d, (d = sqrt(10)/5 * abs(real.sin (α + real.pi / 4))) ∧ (d ≤ sqrt(10)/5) :=
by
  sorry

end rectangular_and_general_equation_maximum_distance_l90_90027


namespace sine_of_minor_arc_PRM_l90_90067

theorem sine_of_minor_arc_PRM (r : ℝ) (PQ : ℝ) (RS : ℝ → Prop) (T : ℝ → ℝ → Prop) (H1 : r = 7) (H2 : PQ = 10) (H3 : ∀ x, RS x → T x PQ → x = 1) : 
  ∃ a b : ℕ, (sin (2 * atan (PQ / (4 * r^2 - PQ^2)^0.5))) = (a / b) ∧ Nat.gcd a b = 1 ∧ a * b = 15 := 
by
  sorry

end sine_of_minor_arc_PRM_l90_90067


namespace iterative_average_difference_l90_90509

-- Define the iterative average function
def iterative_average (seq : List ℚ) : ℚ :=
  match seq with
  | []       => 0
  | x :: xs =>
    List.foldl (fun acc x => (acc + x) / 2) x xs

-- Define the problem statement
theorem iterative_average_difference:
  let seq := [2, 4, 6, 8, 10].map (fun n => (n : ℚ)) in
  let perms := List.permutations seq in
  let averages := perms.map iterative_average in
  (List.maximum averages - List.minimum averages) = 4.25 :=
  by
  -- Placeholder for the proof
  sorry

end iterative_average_difference_l90_90509


namespace nancy_statues_max_profit_difference_l90_90136

theorem nancy_statues_max_profit_difference :
  let jade : ℕ := 1920
  let giraffe_grams : ℕ := 120
  let giraffe_price : ℕ := 150
  let elephant_grams : ℕ := 240
  let elephant_price : ℕ := 350
  let rhino_grams : ℕ := 180
  let rhino_price : ℕ := 250
  let giraffe_profit : ℕ := (jade / giraffe_grams) * giraffe_price
  let elephant_profit : ℕ := (jade / elephant_grams) * elephant_price
  let rhino_profit : ℕ := (jade / rhino_grams) * rhino_price
  in elephant_profit - giraffe_profit = 400 :=
by sorry

end nancy_statues_max_profit_difference_l90_90136


namespace quadratic_has_two_distinct_real_roots_l90_90522

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ p : ℝ, (p = x1 ∨ p = x2) → (p ^ 2 + (4 * m + 1) * p + m = 0)) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l90_90522


namespace total_toothpicks_l90_90785

theorem total_toothpicks (h v : ℕ) (height width : ℕ) (extra_h_lines extra_v_lines : ℕ) :
  height = 25 →
  width = 15 →
  extra_h_lines = (height + 1) / 5 →
  extra_v_lines = (width + 1) / 3 →
  h = (height + 1) * width + extra_h_lines * width →
  v = (width + 1) * height + extra_v_lines * height →
  h + v = 990 :=
by
  intros h_eq v_eq height_eq width_eq extra_h_lines_eq extra_v_lines_eq h_total_eq v_total_eq
  rw [height_eq, width_eq] at *
  rw [extra_h_lines_eq, extra_v_lines_eq] at *
  simp at *
  sorry

end total_toothpicks_l90_90785


namespace increasing_interval_of_f_l90_90758

-- Definitions
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Theorem statement
theorem increasing_interval_of_f : ∀ x ∈ set.Ici (-1 : ℝ), ∀ y ∈ set.Ici (-1 : ℝ), x ≤ y → f x ≤ f y :=
sorry

end increasing_interval_of_f_l90_90758


namespace unique_sums_bound_l90_90238

theorem unique_sums_bound (n : ℕ) (a : ℕ → ℕ) (k : ℕ) 
  (h1 : 1 ≤ a 1) (h2 : ∀ i j, i < j → a i < a j) 
  (h3 : ∀ i, 1 ≤ i → i ≤ k → a i ≤ n) 
  (h4 : ∀ i j, i ≤ j → j ≤ k → a i + a j ≠ a j + a (j + 1)) : k ≤ nat.sqrt (2 * n) + 1 := 
sorry

end unique_sums_bound_l90_90238


namespace integer_values_count_l90_90512

theorem integer_values_count (x : ℤ) :
  ∃ k, (∀ n : ℤ, (3 ≤ Real.sqrt (3 * n + 1) ∧ Real.sqrt (3 * n + 1) < 5) ↔ ((n = 3) ∨ (n = 4) ∨ (n = 5) ∨ (n = 6) ∨ (n = 7)) ∧ k = 5) :=
by
  sorry

end integer_values_count_l90_90512


namespace b_range_exists_solution_l90_90531

theorem b_range_exists_solution (b : ℝ) : (-15 ≤ b ∧ b ≤ 15) ↔ ∃ (a x y : ℝ), 
  (x^2 + y^2 + 2 * b * (b - x + y) = 4) ∧ 
  (y = 5 * cos (x - a) - 12 * sin (x - a)) :=
by
  sorry

end b_range_exists_solution_l90_90531


namespace sum_S60_l90_90030

def sequence (n : ℕ) : ℝ := n * (Real.cos (n * Real.pi / 3) ^ 2 - Real.sin (n * Real.pi / 3) ^ 2)

def partial_sum (n : ℕ) : ℝ := (Finset.range n).sum sequence

theorem sum_S60 : partial_sum 60 = 30 := by
sorry

end sum_S60_l90_90030


namespace probability_of_same_color_l90_90281

theorem probability_of_same_color (black_chairs brown_chairs : ℕ) (hblack : black_chairs = 15) (hbrown : brown_chairs = 18) :
  let total_chairs := black_chairs + brown_chairs in
  let same_color_prob := (black_chairs / total_chairs * (black_chairs - 1) / (total_chairs - 1)) + 
                          (brown_chairs / total_chairs * (brown_chairs - 1) / (total_chairs - 1)) in
  same_color_prob = 43 / 88 :=
by
  sorry

end probability_of_same_color_l90_90281


namespace new_pyramid_dimensions_l90_90900

noncomputable def pentagonalPyramid (a : ℝ) : ℝ :=
  let sin18 := real.sin (real.pi / 10)
  let cos18 := real.cos (real.pi / 10)
  let sin36 := real.sin (2 * real.pi / 10)
  let tan18 := real.tan (real.pi / 10)
  a / (2 * sin18 + tan18)

theorem new_pyramid_dimensions (a : ℝ) :
  ∃ x : ℝ, x = pentagonalPyramid a :=
by
  use pentagonalPyramid a
  sorry

end new_pyramid_dimensions_l90_90900


namespace smallest_sum_of_two_squares_l90_90223

theorem smallest_sum_of_two_squares :
  ∃ n : ℕ, (∀ m : ℕ, m < n → (¬ (∃ a b c d e f : ℕ, m = a^2 + b^2 ∧  m = c^2 + d^2 ∧ m = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))))) ∧
          (∃ a b c d e f : ℕ, n = a^2 + b^2 ∧  n = c^2 + d^2 ∧ n = e^2 + f^2 ∧ ((a, b) ≠ (c, d) ∧ (a, b) ≠ (e, f) ∧ (c, d) ≠ (e, f))) :=
sorry

end smallest_sum_of_two_squares_l90_90223


namespace radius_of_circle_C_l90_90076

-- Definitions for conditions
def intersects_x_axis_at (C : circle) (A B : point) : Prop :=
  -- Circle intersects x-axis at A and B
  A.y = 0 ∧ B.y = 0 ∧ (C.contains A) ∧ (C.contains B)

def is_tangent_to (C : circle) (l : line) : Prop :=
  -- Circle C is tangent to line l
  ∃ P : point, C.contains P ∧ l.contains P ∧ TangentAtPoint (circle C) (line l) P

-- Given problem statement and conditions
def problem_statement (C : circle) :=
  intersects_x_axis_at C ⟨1,0⟩ ⟨3,0⟩ ∧ is_tangent_to C (mk_line (x - y - 3)) 

-- Conjecture to prove
theorem radius_of_circle_C (C : circle) (hc : problem_statement C) :
  C.radius = sqrt(2) :=
sorry  -- proof placeholder

end radius_of_circle_C_l90_90076


namespace intersection_M_N_l90_90967

def M : Set ℝ := { y | ∃ x, y = 2^x ∧ x > 0 }
def N : Set ℝ := { y | ∃ z, y = Real.log z ∧ z ∈ M }

theorem intersection_M_N : M ∩ N = { y | y > 1 } := sorry

end intersection_M_N_l90_90967


namespace stones_required_to_pave_hall_l90_90258

theorem stones_required_to_pave_hall :
  ∀ (hall_length_m hall_breadth_m stone_length_dm stone_breadth_dm: ℕ),
  hall_length_m = 72 →
  hall_breadth_m = 30 →
  stone_length_dm = 6 →
  stone_breadth_dm = 8 →
  (hall_length_m * 10 * hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm) = 4500 := by
  intros _ _ _ _ h_length h_breadth h_slength h_sbreadth
  sorry

end stones_required_to_pave_hall_l90_90258


namespace min_n_minus_m_l90_90022

def f (x : ℝ) := Real.exp (4 * x - 1)
def g (x : ℝ) := (1 / 2) + Real.log (2 * x)

theorem min_n_minus_m (m n : ℝ) (h : f m = g n) : n - m = (1 + Real.log 2) / 4 :=
by sorry

end min_n_minus_m_l90_90022


namespace binom_10_3_l90_90431

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90431


namespace binom_10_3_eq_120_l90_90323

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90323


namespace binomial_10_3_eq_120_l90_90418

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90418


namespace ellipse_standard_equation_and_distance_l90_90595

theorem ellipse_standard_equation_and_distance :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 = 2 * b^2) ∧ (b = 1) ∧
    (∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
    (∃ (P Q : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ Q.1^2 / a^2 + Q.2^2 / b^2 = 1 ∧
                  (P.1 * Q.1 + P.2 * Q.2 = 0) ∧ 
                  (∃ (n t : ℝ), t^2 = 3 * n^2 / 2 - 1 ∧ n^2 ≥ 2 / 3 ∧ 
                  (distance (0, 0) (P, Q).line = |n| / sqrt (1 + t^2) = sqrt 6 / 3))))↦
                 ∃ (x y : ℝ), (x^2 / 2 + y^2 = 1) ∧ 
                  ∃ (P Q : ℝ × ℝ), (P.1^2 / 2 + P.2^2 = 1) ∧ (Q.1^2 / 2 + Q.2^2 = 1) ∧
                  (0 * P.1 + 0 * Q.1 + 1 * P.2 + 1 * Q.2 = 0) ↨
                 ∃ (n t : ℝ), t^2 = 3 * n^2 / 2 - 1) ∧ n^2 ≥ 2 / 3 ↔ 
                  (distance (point_line_dist P Q) = sqrt (1 + t^2))) :
  exists (d : ℝ), d = sqrt (6) /3
:= sorry


end ellipse_standard_equation_and_distance_l90_90595


namespace total_number_of_outfits_l90_90741

noncomputable def number_of_outfits (shirts pants ties jackets : ℕ) :=
  shirts * pants * ties * jackets

theorem total_number_of_outfits :
  number_of_outfits 8 5 5 3 = 600 :=
by
  sorry

end total_number_of_outfits_l90_90741


namespace sum_sin_squared_eq_10_l90_90507

theorem sum_sin_squared_eq_10 : (∑ k in Finset.range 30, Real.sin (6 * (k + 1) - 3) ^ 2) = 10 := 
by
  sorry

end sum_sin_squared_eq_10_l90_90507


namespace binom_10_3_eq_120_l90_90397

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90397


namespace binom_10_3_l90_90435

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90435


namespace binom_10_3_eq_120_l90_90389

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90389


namespace boat_length_l90_90245

theorem boat_length (breadth : ℝ) (sink : ℝ) (man_mass : ℝ) (density : ℝ) (g : ℝ) :
  breadth = 3 → sink = 0.01 → man_mass = 240 → density = 1000 → g = 9.81 → 
  ∃ L : ℝ, L = 8 :=
by
  assume h1 : breadth = 3
  assume h2 : sink = 0.01
  assume h3 : man_mass = 240
  assume h4 : density = 1000
  assume h5 : g = 9.81
  sorry

end boat_length_l90_90245


namespace club_triangle_activity_l90_90992

theorem club_triangle_activity {n : ℕ} (h1 : ∃ (people : Finset ℕ), 
  people.card = 3 * n + 1)
  (h2 : ∀ (p : ℕ) (people : Finset ℕ), p ∈ people → ∃ (tennis table_tennis chess : Finset ℕ), 
  tennis.card = n ∧ table_tennis.card = n ∧ chess.card = n ∧ ∀ (q : ℕ), 
  (q ∈ tennis ∨ q ∈ table_tennis ∨ q ∈ chess → q ≠ p ∧ q ∈ people ∧
  (q ∈ tennis → ¬(q ∈ table_tennis ∨ q ∈ chess)) ∧ 
  (q ∈ table_tennis → ¬(q ∈ tennis ∨ q ∈ chess)) ∧ 
  (q ∈ chess → ¬(q ∈ tennis ∨ q ∈ table_tennis)))) :
  ∃ (a b c : Finset ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ (x y : ℕ), x ∈ a ∧ y ∈ b ∧ x ≠ y → 
  ∃ z ∈ c, z ≠ x ∧ z ≠ y ∧ z ∈ people ∧ 
  ((activity (a, z) = "tennis" ∧ activity (b, z) = "table_tennis" ∧ activity (a, b) = "chess") ∨ 
  (activity (a, z) = "table_tennis" ∧ activity (b, z) = "chess" ∧ activity (a, b) = "tennis") ∨ 
  (activity (a, z) = "chess" ∧ activity (b, z) = "tennis" ∧ activity (a, b) = "table_tennis"))) :=
sorry

end club_triangle_activity_l90_90992


namespace part1_solution_set_a_eq_1_part2_range_of_a_l90_90525

def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|

theorem part1_solution_set_a_eq_1 :
  (λ x, f x 1 > 1) = (λ x, -2 < x ∧ x < -2 / 3) := 
sorry

theorem part2_range_of_a
  (h : ∀ x, x ∈ set.Icc 2 3 → f x a > 0) :
  -5 / 2 < a ∧ a < -2 :=
sorry

end part1_solution_set_a_eq_1_part2_range_of_a_l90_90525


namespace smallest_positive_y_l90_90521

theorem smallest_positive_y (y : ℕ) (h : 42 * y + 8 ≡ 4 [MOD 24]) : y = 2 :=
sorry

end smallest_positive_y_l90_90521


namespace minimum_voters_for_tall_win_l90_90085

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end minimum_voters_for_tall_win_l90_90085


namespace player_B_wins_l90_90706

variable {R : Type*} [Ring R]

noncomputable def polynomial_game (n : ℕ) (f : Polynomial R) : Prop :=
  (f.degree = 2 * n) ∧ (∃ (a b : R) (x y : R), f.eval x = 0 ∨ f.eval y = 0)

theorem player_B_wins (n : ℕ) (f : Polynomial ℝ)
  (h1 : n ≥ 2)
  (h2 : f.degree = 2 * n) :
  polynomial_game n f :=
by
  sorry

end player_B_wins_l90_90706


namespace monogram_count_l90_90700

theorem monogram_count (alphabet : Finset Char) (A : Char) (hA : A = 'A') (h_size : alphabet.card = 26) :
  let remaining := alphabet.erase 'A' in
  let count := (Finset.card remaining).choose 2 in
  count = 300 :=
by
  sorry

end monogram_count_l90_90700


namespace combination_10_3_eq_120_l90_90483

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90483


namespace calc_j_inverse_l90_90678

noncomputable def i : ℂ := Complex.I  -- Equivalent to i^2 = -1 definition of complex imaginary unit
noncomputable def j : ℂ := i + 1      -- Definition of j

theorem calc_j_inverse :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by 
  -- The statement here only needs to declare the equivalence, 
  -- without needing the proof
  sorry

end calc_j_inverse_l90_90678


namespace asymptotes_of_hyperbola_l90_90188

-- Define the hyperbola with parameters a and b
def hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) := 
  ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

-- Length of AB and CD given the conditions
def lengths (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) : Prop := 
  let AB := 2 * b^2 / a 
  let CD := 2 * b * c / a 
  AB = (sqrt 3 / 2) * CD

-- The theorem to prove
theorem asymptotes_of_hyperbola (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (cond : lengths a b c a_pos b_pos) : 
  (hyperbola a b a_pos b_pos → (∀ x, y = (sqrt 3) * x ∨ y = -(sqrt 3) * x)) :=
by {
  sorry
}

end asymptotes_of_hyperbola_l90_90188


namespace trajectory_and_max_area_l90_90578

open Real

theorem trajectory_and_max_area (a : ℝ) :
  let M : set (ℝ × ℝ) := { p | (p.1 - 2)^2 + p.2^2 = 1 }
  let P := { x | 2 * x - a * (x * k + a) - 3 = 0 }
  {R S : ℝ × ℝ} (hR : R ∈ M) (hS : S ∈ M) 
  (hR_t : R.1 * 2 - a * R.2 - 3 = 0) (hS_t : S.1 * 2 - a * S.2 - 3 = 0) :
  ∃ P, (∀ B C, (B ∈ M ∧ C ∈ M → (P ∈ line B C ∧ (reciprocal_sequence B P C))) ∨ P ∈ P) ∧ 
  (∀ R S, R ∈ P ∧ S ∈ P → ∃ aRS : ℝ, aRS = abs (R.2 - S.2) * sqrt ((a^2 + 3) / (a^2 + 4)^2) ≤ sqrt 3 / 4 := 
sorry

end trajectory_and_max_area_l90_90578


namespace total_sales_correct_l90_90839

-- Define the conditions
def total_tickets : ℕ := 65
def senior_ticket_price : ℕ := 10
def regular_ticket_price : ℕ := 15
def regular_tickets_sold : ℕ := 41

-- Calculate the senior citizen tickets sold
def senior_tickets_sold : ℕ := total_tickets - regular_tickets_sold

-- Calculate the revenue from senior citizen tickets
def revenue_senior : ℕ := senior_ticket_price * senior_tickets_sold

-- Calculate the revenue from regular tickets
def revenue_regular : ℕ := regular_ticket_price * regular_tickets_sold

-- Define the total sales amount
def total_sales_amount : ℕ := revenue_senior + revenue_regular

-- The statement we need to prove
theorem total_sales_correct : total_sales_amount = 855 := by
  sorry

end total_sales_correct_l90_90839


namespace num_satisfy_ineq_l90_90971

def satisfies_ineq (n : ℤ) : Prop :=
  -100 < n^3 + n^2 ∧ n^3 + n^2 < 100

theorem num_satisfy_ineq : 
  (finset.filter satisfies_ineq (finset.Icc (-5 : ℤ) 5)).card = 9 :=
by
  sorry

end num_satisfy_ineq_l90_90971


namespace compound_formed_is_barium_hydroxide_l90_90914

-- Definitions corresponding to the conditions
def barium_oxide := "BaO"
def water := "H2O"
def barium_hydroxide := "Ba(OH)2"

def moles_barium_oxide_used : ℕ := 3
def water_required_in_grams : ℕ := 54
def molar_mass_water : ℚ := 18.015

-- Lean statement to prove the compound formed
theorem compound_formed_is_barium_hydroxide
  (h1 : moles_barium_oxide_used * molar_mass_water = water_required_in_grams) :
  barium_oxide + " + " + water + " → " + barium_hydroxide :=
by
  -- Proof is omitted
  sorry

end compound_formed_is_barium_hydroxide_l90_90914


namespace tangent_to_second_circle_l90_90211

-- Define the points and circles
variable (A B C D M K : Point)
variable (circle1 circle2 : Circle)
variable [h1 : IntersectsAtTwoPoints circle1 circle2 A B]
variable [h2 : LineThrough B intersectsCircleAgainAt circle1 C]
variable [h3 : LineThrough B intersectsCircleAgainAt circle2 D]
variable [h4 : TangentAtPoint circle1 C intersectsTangentAtPoint circle2 D M]
variable (O : Point)
variable [h5 : Intersection (LineThroughPoints A M) (LineThroughPoints C D) O]
variable [h6 : LineThrough O ParallelTo (LineThroughPoints C M) intersects At K on LineThroughPoints A C]

-- Define the conclusion to prove
theorem tangent_to_second_circle :
  TangentAtPoint circle2 K B :=
sorry

end tangent_to_second_circle_l90_90211


namespace combination_10_3_l90_90401

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90401


namespace smallest_k_for_p_squared_minus_k_divisible_by_15_l90_90680

theorem smallest_k_for_p_squared_minus_k_divisible_by_15 :
  ∀ (p : ℕ), nat.prime p ∧ (2023 ≤ ∀ q, nat.prime q → nDigits q = 2023 → q ≤ p) →
  ∃ (k : ℕ), 0 < k ∧ (p^2 - k) % 15 = 0 ∧ k = 15 :=
by
  sorry

end smallest_k_for_p_squared_minus_k_divisible_by_15_l90_90680


namespace ratio_of_volumes_l90_90849

-- Define the problem conditions and question
def rightCircularCone (h r : ℝ) : Prop :=
  ∃ V₁ V₂ : ℝ,
    let V₁ := (1/3) * π * (5*r)^2 * (5*h) - (1/3) * π * (4*r)^2 * (4*h) in
    let V₂ := (1/3) * π * (4*r)^2 * (4*h) - (1/3) * π * (3*r)^2 * (3*h) in
    true

-- State the theorem, using the problem conditions to imply the conclusion
theorem ratio_of_volumes
  {h r : ℝ} (hh : 0 < h) (hr : 0 < r) : rightCircularCone h r → 
  let V₁ := (1/3) * π * (5 * r)^2 * (5 * h) - (1/3) * π * (4 * r)^2 * (4 * h) in
  let V₂ := (1/3) * π * (4 * r)^2 * (4 * h) - (1/3) * π * (3 * r)^2 * (3 * h) in
  V₂ / V₁ = 37 / 61 :=
sorry

end ratio_of_volumes_l90_90849


namespace binomial_10_3_eq_120_l90_90423

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90423


namespace binom_10_3_eq_120_l90_90475

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90475


namespace orthocenters_concyclic_and_center_l90_90115

-- Given definitions
variable {A1 A2 A3 A4 : ℂ}
variable {H1 H2 H3 H4 : ℂ}

-- Assuming that the points A1, A2, A3, A4 are vertices of a cyclic quadrilateral inscribed in a circle
def is_cyclic_quad (A1 A2 A3 A4 : ℂ) : Prop :=
  ∃ (O : ℂ) (r : ℝ), r > 0 ∧
  (A1 - O).abs = r ∧ (A2 - O).abs = r ∧ (A3 - O).abs = r ∧ (A4 - O).abs = r

-- Assuming that H1, H2, H3, H4 are orthocenters of the triangles ΔA2A3A4, ΔA3A4A1, ΔA4A1A2, ΔA1A2A3 respectively
axiom orthocenter_A2A3A4 : orthocenter A2 A3 A4 = H1
axiom orthocenter_A3A4A1 : orthocenter A3 A4 A1 = H2
axiom orthocenter_A4A1A2 : orthocenter A4 A1 A2 = H3
axiom orthocenter_A1A2A3 : orthocenter A1 A2 A3 = H4

-- Proving that H1, H2, H3, H4 lie on a common circle with center O'
theorem orthocenters_concyclic_and_center :
  is_cyclic_quad A1 A2 A3 A4 →
  H1 ≠ 0 → H2 ≠ 0 → H3 ≠ 0 → H4 ≠ 0 →
  ∃ (O' : ℂ), (H1 - O').abs = (H2 - O').abs ∧
  (H2 - O').abs = (H3 - O').abs ∧
  (H3 - O').abs = (H4 - O').abs ∧
  O' = A1 + A2 + A3 + A4 :=
by
  sorry

end orthocenters_concyclic_and_center_l90_90115


namespace average_cost_per_pencil_l90_90268

theorem average_cost_per_pencil (pencil_price : ℚ) (shipping_cost : ℚ) (discount : ℚ) (total_pencils : ℕ) :
  pencil_price = 1550 / 100 ∧
  shipping_cost = 575 / 100 ∧
  discount = 100 / 100 ∧
  total_pencils = 150 →
  (Real.toInt (Real.round ((pencil_price + shipping_cost - discount) * 100 / total_pencils))) = 14 := 
by 
  intro h
  rw [←Rat.cast_coe_nat]
  simp at h
  cases h with h1 h'
  cases h' with h2 h'
  cases h' with h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_cost_per_pencil_l90_90268


namespace sticks_per_pot_is_181_l90_90203

/-- Define the problem conditions -/
def number_of_pots : ℕ := 466
def flowers_per_pot : ℕ := 53
def total_flowers_and_sticks : ℕ := 109044

/-- Define the function to calculate the number of sticks per pot -/
def sticks_per_pot (S : ℕ) : Prop :=
  (number_of_pots * flowers_per_pot + number_of_pots * S = total_flowers_and_sticks)

/-- State the theorem -/
theorem sticks_per_pot_is_181 : sticks_per_pot 181 :=
by
  sorry

end sticks_per_pot_is_181_l90_90203


namespace minimum_votes_for_tall_l90_90084

theorem minimum_votes_for_tall (voters : ℕ) (districts : ℕ) (precincts : ℕ) (precinct_voters : ℕ)
  (vote_majority_per_precinct : ℕ → ℕ) (precinct_majority_per_district : ℕ → ℕ) (district_majority_to_win : ℕ) :
  voters = 135 ∧ districts = 5 ∧ precincts = 9 ∧ precinct_voters = 3 ∧
  (∀ p, vote_majority_per_precinct p = 2) ∧
  (∀ d, precinct_majority_per_district d = 5) ∧
  district_majority_to_win = 3 ∧ 
  tall_won : 
  ∃ min_votes, min_votes = 30 :=
by
  sorry

end minimum_votes_for_tall_l90_90084


namespace minimum_students_per_bench_l90_90182

theorem minimum_students_per_bench (M : ℕ) (B : ℕ) (F : ℕ) (H1 : F = 4 * M) (H2 : M = 29) (H3 : B = 29) :
  ⌈(M + F) / B⌉ = 5 :=
by
  sorry

end minimum_students_per_bench_l90_90182


namespace max_cells_intersected_10_radius_circle_l90_90220

noncomputable def max_cells_intersected_by_circle (radius : ℝ) (cell_size : ℝ) : ℕ :=
  if radius = 10 ∧ cell_size = 1 then 80 else 0

theorem max_cells_intersected_10_radius_circle :
  max_cells_intersected_by_circle 10 1 = 80 :=
sorry

end max_cells_intersected_10_radius_circle_l90_90220


namespace binom_10_3_l90_90346

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90346


namespace parallel_line_slope_y_intercept_l90_90797

theorem parallel_line_slope_y_intercept (x y : ℝ) (h : 3 * x - 6 * y = 12) :
  ∃ (m b : ℝ), m = 1 / 2 ∧ b = -2 := 
by { sorry }

end parallel_line_slope_y_intercept_l90_90797


namespace coefficients_sum_even_odd_split_sum_binomial_coeff_sum_l90_90554

noncomputable def problem_statement (a : ℕ → ℤ) (x : ℤ) :=
  Σ i in finset.range 8, a i * x ^ i = (1 - 2 * x) ^ 7

theorem coefficients_sum (a : ℕ → ℤ) : (Σ x in finset.range 8, a x) = 0 :=
sorry

theorem even_odd_split_sum (a : ℕ → ℤ) :
  (Σ i in finset.range 4, a (2 * i)) = 129 ∧ (Σ i in finset.range 4, a (2 * i + 1)) = -128 :=
sorry

theorem binomial_coeff_sum : (Σ i in finset.range 8, Nat.choose 7 i) = 128 :=
sorry

end coefficients_sum_even_odd_split_sum_binomial_coeff_sum_l90_90554


namespace equilateral_triangle_area_l90_90746

theorem equilateral_triangle_area (h : ℝ) (A : ℝ) (s : ℝ) 
  (h_eq : h = 3 * real.sqrt 3) 
  (area_formula : A = (real.sqrt 3 / 4) * s^2)
  (side_length_eq : h = real.sqrt 3 * (s / 2)) :
  A = 9 * real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l90_90746


namespace license_plate_combinations_l90_90291

theorem license_plate_combinations:
  let choose := Nat.choose in
  (choose 26 2) * (choose 6 2) * (choose 4 2) * (choose 24 2) * 10 * 9 * 8 = 84563400000 := by
  sorry

end license_plate_combinations_l90_90291


namespace second_year_sample_size_l90_90851

theorem second_year_sample_size (x1 x2 x3 s3 n : ℕ) (hx1 : x1 = 800) (hx2 : x2 = 1600) (hx3 : x3 = 1400) (hs3 : s3 = 70) :
  n = 80 :=
by
  have h1 : 70 / 1400 = s3 / x3 := by rw [hs3, hx3]
  have h2 : n / x2 = s3 / x3 := by sorry
  apply h2, sorry


end second_year_sample_size_l90_90851


namespace profit_percentage_with_discount_l90_90265

def CP : ℝ := 100
def profit_percentage_without_discount : ℝ := 130
def discount_percentage : ℝ := 5

theorem profit_percentage_with_discount :
    let SP := CP + CP * (profit_percentage_without_discount / 100) in
    let Discount := SP * (discount_percentage / 100) in
    let SP_discount := SP - Discount in
    let Profit_with_discount := SP_discount - CP in 
    (Profit_with_discount / CP) * 100 = 118.5 := by
    sorry

end profit_percentage_with_discount_l90_90265


namespace binomial_10_3_l90_90500

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90500


namespace shortest_side_length_rectangular_solid_geometric_progression_l90_90201

theorem shortest_side_length_rectangular_solid_geometric_progression
  (b s : ℝ)
  (h1 : (b^3 / s) = 512)
  (h2 : 2 * ((b^2 / s) + (b^2 * s) + b^2) = 384)
  : min (b / s) (min b (b * s)) = 8 := 
sorry

end shortest_side_length_rectangular_solid_geometric_progression_l90_90201


namespace det_matrix_power_l90_90617

variable {n : Type} [Fintype n] [DecidableEq n] [Field R]
variable (N : Matrix n n R)

theorem det_matrix_power (h : det N = 3) : det (N ^ 3) = 27 :=
by
  sorry  -- This is where the proof would go

end det_matrix_power_l90_90617


namespace systematic_sampling_first_group_l90_90792

theorem systematic_sampling_first_group :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 160 ∧ ∃ k, k = 16 → (let group16 := (121, 128) in 125 ∈ set.Icc group16.1 group16.2)) →
  let group1 := (1, 8) in 5 ∈ set.Icc group1.1 group1.2 :=
by
  intros n h,
  let group1 := (1, 8),
  let group16 := (121, 128),
  have h_group16_125 : 125 ∈ set.Icc group16.1 group16.2 := sorry,
  have h_group1_5 : 5 ∈ set.Icc group1.1 group1.2 := sorry,
  exact h_group1_5

end systematic_sampling_first_group_l90_90792


namespace combination_10_3_eq_120_l90_90380

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90380


namespace binom_10_3_eq_120_l90_90400

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90400


namespace combination_10_3_l90_90415

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90415


namespace range_of_g_l90_90517

def g (x : ℝ) : ℝ := if x ≠ -5 then 3 * (x - 4) else 0 -- function definition 

theorem range_of_g : 
  (set.range g) = (set.Iio (-27) ∪ set.Ioi (-27)) := 
by sorry

end range_of_g_l90_90517


namespace combination_10_3_eq_120_l90_90477

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90477


namespace total_cats_and_kittens_received_is_correct_l90_90663

-- Definitions based on conditions
def total_adult_cats := 120
def percent_female := 0.6
def percent_female_with_litters := 0.6
def average_kittens_per_litter := 5

-- The Lean statement to prove
theorem total_cats_and_kittens_received_is_correct :
  let number_of_female_cats := percent_female * total_adult_cats in
  let number_of_litters := percent_female_with_litters * number_of_female_cats in
  let total_kittens := (number_of_litters.floor) * average_kittens_per_litter in
  let total_cats_and_kittens := total_adult_cats + total_kittens in
  total_cats_and_kittens = 335 := by
  sorry

end total_cats_and_kittens_received_is_correct_l90_90663


namespace line_parallel_or_within_plane_l90_90043

noncomputable def check_line_plane_relationship (λ μ : ℝ) (AB CD CE : ℝ³) : Prop :=
  AB = λ • CD + μ • CE → ∃ k : ℝ, (k ≠ 0 ∧ AB = k • (CD + CE)) ∨ (k = 0 ∧ AB = 0)

theorem line_parallel_or_within_plane (λ μ : ℝ) (AB CD CE : ℝ³)
  (h : AB = λ • CD + μ • CE) : 
  AB ≠ 0 → (∃ k : ℝ, AB = k • (CD + CE)) ∨ (AB = 0) := 
by 
  sorry

end line_parallel_or_within_plane_l90_90043


namespace sum_prime_factors_of_1260_l90_90800

theorem sum_prime_factors_of_1260 :
  ∃ (a b : ℕ), (∀ p, p.prime → p ∣ 1260 → p = a ∨ p = b) ∧ a + b = 9 :=
by
  sorry

end sum_prime_factors_of_1260_l90_90800


namespace sin_double_angle_l90_90927

theorem sin_double_angle (α : ℝ) (h1 : sin α - cos α = sqrt 2) (h2 : 0 < α ∧ α < π) : sin (2 * α) = -1 := 
by
  sorry

end sin_double_angle_l90_90927


namespace max_m_l90_90932

-- Define the function f: ℝ → ℝ
def f (a x : ℝ) : ℝ := x^3 - (9 / 2) * x^2 + 6 * x - a

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 9 * x + 6

-- State the theorem about the maximum m
theorem max_m (a : ℝ) : ∃ m : ℝ, (∀ x : ℝ, f' a x ≥ m) ∧ (m = -3 / 4) :=
  sorry

end max_m_l90_90932


namespace binom_10_3_eq_120_l90_90461

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90461


namespace count_remainders_gte_l90_90546

def remainder (a N : ℕ) : ℕ := a % N

theorem count_remainders_gte (N : ℕ) : 
  (∀ a, a > 0 → remainder a 1000 > remainder a 1001 → N ≤ 1000000) →
  N = 499500 :=
by
  sorry

end count_remainders_gte_l90_90546


namespace blocks_to_get_home_l90_90716

-- Definitions based on conditions provided
def blocks_to_park := 4
def blocks_to_school := 7
def trips_per_day := 3
def total_daily_blocks := 66

-- The proof statement for the number of blocks Ray walks to get back home
theorem blocks_to_get_home 
  (h1: blocks_to_park = 4)
  (h2: blocks_to_school = 7)
  (h3: trips_per_day = 3)
  (h4: total_daily_blocks = 66) : 
  (total_daily_blocks / trips_per_day - (blocks_to_park + blocks_to_school) = 11) :=
by
  sorry

end blocks_to_get_home_l90_90716


namespace binom_10_3_eq_120_l90_90464

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90464


namespace combination_10_3_eq_120_l90_90378

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90378


namespace triangle_area_sqrt2_l90_90940

/-
Given:
  1. ΔABC is a triangle with an altitude CD of 1 cm splitting the triangle into two 45-45-90 triangles.
Prove:
  The area of ΔABC is √2 square centimeters.
-/

theorem triangle_area_sqrt2 
  (A B C D : Type)
  (h1 : CD = 1)
  (h2 : 45-45-90 → split_triangle ABC D) :
  area ABC = √2 :=
by sorry

end triangle_area_sqrt2_l90_90940


namespace binomial_10_3_l90_90328

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90328


namespace combination_10_3_eq_120_l90_90371

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90371


namespace market_value_of_stock_l90_90243

-- Define the conditions and the relation
theorem market_value_of_stock (face_value dividend_percentage yield_percentage : ℝ) (hface : face_value = 100) (hdiv: dividend_percentage = 0.06) (hyield: yield_percentage = 0.08) :
  let dividend_per_share := dividend_percentage * face_value in
  let market_value := (dividend_per_share / yield_percentage) * 100 in 
  market_value = 75 :=
    sorry

end market_value_of_stock_l90_90243


namespace parallel_segments_slope_l90_90172

theorem parallel_segments_slope (k : ℝ) :
  let A : ℝ × ℝ := (-6, 0),
      B : ℝ × ℝ := (0, -6),
      X : ℝ × ℝ := (0, 10),
      Y : ℝ × ℝ := (16, k) in
  (((B.2 - A.2) / (B.1 - A.1)) = ((Y.2 - X.2) / (Y.1 - X.1))) ↔ k = -6 :=
by sorry

end parallel_segments_slope_l90_90172


namespace binomial_10_3_l90_90337

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90337


namespace combination_10_3_l90_90410

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90410


namespace binom_10_3_l90_90432

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90432


namespace max_value_of_g_l90_90664

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (x * (80 - x)) + Real.sqrt (x * (10 - x))

theorem max_value_of_g :
  ∃ y_0 N, (∀ x, 0 ≤ x ∧ x ≤ 10 → g x ≤ N) ∧ g y_0 = N ∧ y_0 = 33.75 ∧ N = 22.5 := 
by
  -- Proof goes here.
  sorry

end max_value_of_g_l90_90664


namespace students_per_bench_l90_90185

-- Definitions based on conditions
def num_male_students : ℕ := 29
def num_female_students : ℕ := 4 * num_male_students
def num_benches : ℕ := 29
def total_students : ℕ := num_male_students + num_female_students

-- Theorem to prove
theorem students_per_bench : total_students / num_benches = 5 := by
  sorry

end students_per_bench_l90_90185


namespace binomial_10_3_eq_120_l90_90460

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90460


namespace prob_carol_first_six_l90_90273

noncomputable theory
open_locale big_operators

def probability_carol_first_six : ℚ :=
  5 / 36

def one_cycle_no_six : ℚ :=
  (5 / 6) ^ 3

def common_ratio : ℚ :=
  one_cycle_no_six

def geometric_series_sum (a r : ℚ) : ℚ :=
  a / (1 - r)

theorem prob_carol_first_six :
  geometric_series_sum probability_carol_first_six common_ratio = 30 / 91 :=
by
  sorry

end prob_carol_first_six_l90_90273


namespace jack_total_damage_costs_l90_90656

def cost_per_tire := 250
def number_of_tires := 3
def cost_of_window := 700

def total_cost_of_tires := cost_per_tire * number_of_tires
def total_cost_of_damages := total_cost_of_tires + cost_of_window

theorem jack_total_damage_costs : total_cost_of_damages = 1450 := 
by
  -- Using the definitions provided
  -- total_cost_of_tires = 250 * 3 = 750
  -- total_cost_of_damages = 750 + 700 = 1450
  sorry

end jack_total_damage_costs_l90_90656


namespace electric_energy_consumption_l90_90106

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end electric_energy_consumption_l90_90106


namespace find_x_squared_plus_inverse_squared_l90_90564

theorem find_x_squared_plus_inverse_squared (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + (1 / x)^2 = 7 :=
by
  sorry

end find_x_squared_plus_inverse_squared_l90_90564


namespace annika_hiking_rate_l90_90868

-- Define the conditions
def initial_hiked_distance : ℝ := 2.5
def total_time_back (T : ℝ := 45 
def total_distance_east (d2 : ℝ := 3.5

-- The theorem which we need to prove
theorem annika_hiking_rate (R : ℝ) (h : total_time_back T = 45) : 
  initial_hiked_distance + (total_distance_east d2 - initial_hiked_distance) * 2 * R + initial_hiked_distance * R = 1/R :=
sorry

end annika_hiking_rate_l90_90868


namespace distance_from_Q_to_EH_is_correct_l90_90157

theorem distance_from_Q_to_EH_is_correct (E F G H N Q : Point)
  (side_length : ℝ) (radius1 radius2 : ℝ)
  (H_coord : H = (0, 0)) (G_coord : G = (5, 0))
  (F_coord : F = (5, 5)) (E_coord : E = (0, 5))
  (N_coord : N = (2.5, 0)) (Q_coord : Q = (2.5, 2.5))
  (circle_N : ∀ (x y : ℝ), (x - 2.5)^2 + y^2 = 6.25)
  (circle_E : ∀ (x y : ℝ), x^2 + (y - 5)^2 = 25) :
  Distance_from_Q_to_EH Q = 2.5 :=
sorry

end distance_from_Q_to_EH_is_correct_l90_90157


namespace stacked_cubes_surface_area_is_945_l90_90524

def volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

def side_length (v : ℕ) : ℕ := v^(1/3)

def num_visible_faces (i : ℕ) : ℕ :=
  if i == 0 then 5 else 3 -- Bottom cube has 5 faces visible, others have 3 due to rotation

def surface_area (s : ℕ) (faces : ℕ) : ℕ :=
  faces * s^2

def total_surface_area (volumes : List ℕ) : ℕ :=
  (volumes.zipWith surface_area (volumes.enum.map (λ (i, v) => num_visible_faces i))).sum

theorem stacked_cubes_surface_area_is_945 :
  total_surface_area volumes = 945 := 
by 
  sorry

end stacked_cubes_surface_area_is_945_l90_90524


namespace binom_10_3_l90_90445

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90445


namespace diagonals_perpendicular_l90_90786

theorem diagonals_perpendicular (squares_intersect_form_octagon : Prop)
                                (diagonals_form_quadrilaterals : Prop)
                                (MN : Type) (NK : Type) (P : Type)
                                (MN_diagonal : MN)
                                (NK_diagonal : NK)
                                (MN_NK_intersect_at_P : MN_diagonal ∩ NK_diagonal = P) :
                                ∃ (MN NK : Type), ⦃MN ⦄ ⦃NK ⦄ 
                                MN_diagonal ⊥ NK_diagonal :=
sorry

end diagonals_perpendicular_l90_90786


namespace casey_saves_by_paying_monthly_l90_90887

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_per_month := 4
  let months := 3
  let cost_monthly := monthly_rate * months
  let cost_weekly := weekly_rate * weeks_per_month * months
  let savings := cost_weekly - cost_monthly
  savings = 360 :=
by
  unfold weekly_rate monthly_rate weeks_per_month months cost_monthly cost_weekly savings
  simp
  sorry

end casey_saves_by_paying_monthly_l90_90887


namespace number_of_books_in_shipment_l90_90246

theorem number_of_books_in_shipment
  (T : ℕ)                   -- The total number of books
  (displayed_ratio : ℚ)     -- Fraction of books displayed
  (remaining_books : ℕ)     -- Number of books in the storeroom
  (h1 : displayed_ratio = 0.3)
  (h2 : remaining_books = 210)
  (h3 : (1 - displayed_ratio) * T = remaining_books) :
  T = 300 := 
by
  -- Add your proof here
  sorry

end number_of_books_in_shipment_l90_90246


namespace ball_hits_ground_at_2_72_l90_90872

-- Define the initial conditions
def initial_velocity (v₀ : ℝ) := v₀ = 30
def initial_height (h₀ : ℝ) := h₀ = 200
def ball_height (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 200

-- Prove that the ball hits the ground at t = 2.72 seconds
theorem ball_hits_ground_at_2_72 (t : ℝ) (h : ℝ) 
  (v₀ : ℝ) (h₀ : ℝ) 
  (hv₀ : initial_velocity v₀) 
  (hh₀ : initial_height h₀)
  (h_eq: ball_height t = h) 
  (h₀_eq: ball_height 0 = h₀) : 
  h = 0 -> t = 2.72 :=
by
  sorry

end ball_hits_ground_at_2_72_l90_90872


namespace measure_of_angle_A_range_of_perimeter_l90_90989

-- First we define the essential conditions
variables {A B C : ℝ}
variables {a b c : ℝ}
variables {angle_A : ℝ := 60}
variables {∠A = angle_A}

-- Condition given in the problem
axiom given_condition : (2 * b - c) * real.cos A = a * real.cos C

-- Task 1: Prove the measure of angle A is 60 degrees
theorem measure_of_angle_A (h : given_condition) : A = 60 := 
by sorry

-- Additional conditions for task 2
variables (b c : ℝ)

-- Condtion a = 4
axiom side_a_four : a = 4

-- Task 2: Given \( a = 4 \), prove the range of perimeter
theorem range_of_perimeter (h : given_condition) (h_a : side_a_four) : 8 < (a + b + c) ∧ (a + b + c) ≤ 12 := 
by sorry

end measure_of_angle_A_range_of_perimeter_l90_90989


namespace binom_10_3_eq_120_l90_90387

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90387


namespace intersection_point_at_l90_90290

/-- Define the line passing through two points -/
def line_through (p1 p2 : Real × Real) (x : Real) : Real :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1)
  m * x + p1.2 - m * p1.1

/-- Define the second line as the given equation y = -x + 15 -/
def line_given (x : Real) : Real :=
  -x + 15

/-- The proof problem statement: Prove that the intersection point is (4, 11) -/
theorem intersection_point_at (p1 p2 : Real × Real) (h1 : p1 = (0, 3)) (h2 : p2 = (4, 11)) :
  ∃ x y : Real, y = line_through p1 p2 x ∧ y = line_given x ∧ x = 4 ∧ y = 11 := by
  sorry

end intersection_point_at_l90_90290


namespace distance_to_tangent_of_circle_l90_90591

theorem distance_to_tangent_of_circle (O : Type*) [metric_space O] (l : set O) : 
  (∃ (r : ℝ), ∀ (x : O), dist (metric.center O) x = 5 ∧ tan_circle l (metric.center O) x ) → 
  diameter (circle O) = 10 → 
  ∃ d : ℝ, dist (metric.center O) l = d :=
begin
    sorry
end

end distance_to_tangent_of_circle_l90_90591


namespace problem1_problem2_l90_90009

noncomputable def focal_points : Foci := ⟨⟨-2 * Real.sqrt 2, 0⟩, ⟨2 * Real.sqrt 2, 0⟩⟩
def major_axis_length : ℝ := 6

theorem problem1 (foci : Foci) (a : ℝ) :
    foci = ⟨⟨-2 * Real.sqrt 2, 0⟩, ⟨2 * Real.sqrt 2, 0⟩⟩ →
    a = 3 →
    ∃ b : ℝ, b^2 = a^2 - (2 * Real.sqrt 2)^2 ∧
    ∃ b_pos : b > 0,
    by sorry ∃ standard_eq : true, 
    Eq (elliptical_equation a b)
    (1 / 9 * x^2 + 1 * y^2 = 1) := sorry

theorem problem2 (point : Point) (slope : ℝ) :
    point = ⟨0,2⟩ →
    slope = 1 →
    (∃ intersection_points : {A B : Point // lies_on_ellipse A ∧ lies_on_ellipse B},
    line_through_points point slope (intersection_points.val.1) (intersection_points.val.2)) →
    dist_intersection_pts_eq_given (d : ℝ)
    by sorry dist = sqrt (1 + 1^2) * ((18^2 / 5^2 -4 * 27 / 10)^.5) :=
    (dist = 6 * Real.sqrt 3 / 5) := sorry

end problem1_problem2_l90_90009


namespace minimum_value_of_a_plus_2b_l90_90561

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_2b_l90_90561


namespace chord_length_is_sqrt_14_l90_90965
noncomputable def chord_length : ℝ :=
  let circle_center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 2
  let line_param : ℝ → ℝ × ℝ := λ t, (2 - 1/2 * t, -1 + 1/2 * t)
  let line_standard (x y : ℝ) : Prop := x + y = 1
  let distance_from_center_to_line : ℝ := abs(-1) / real.sqrt(2)
  let chord_length_formula (r d : ℝ) : ℝ := real.sqrt(4 * r^2 - d^2)
  chord_length_formula radius distance_from_center_to_line

theorem chord_length_is_sqrt_14 (t : ℝ) : chord_length = real.sqrt 14 := by
  sorry

end chord_length_is_sqrt_14_l90_90965


namespace monotonic_increasing_intervals_of_f_l90_90181

def f (x : ℝ) : ℝ := x / (1 - x)

theorem monotonic_increasing_intervals_of_f :
  (∀ x y : ℝ, x < y ∧ y < 1 → f x < f y) ∧ (∀ x y : ℝ, x > 1 ∧ x < y → f x < f y) := by
  sorry

end monotonic_increasing_intervals_of_f_l90_90181


namespace binom_10_3_l90_90354

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90354


namespace no_three_digit_whole_number_solves_log_eq_l90_90081

noncomputable def log_function (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem no_three_digit_whole_number_solves_log_eq :
  ¬ ∃ n : ℤ, (100 ≤ n ∧ n < 1000) ∧ log_function (3 * n) 10 + log_function (7 * n) 10 = 1 :=
by
  sorry

end no_three_digit_whole_number_solves_log_eq_l90_90081


namespace find_k_value_l90_90611

variables (k : ℝ)

theorem find_k_value (h : ∃ x : ℂ, (x^3 + 2*(k-1 : ℝ)*x^2 + 9*x + 5*(k-1 : ℝ) = 0) ∧ |x| = √5) :
    k = -1 ∨ k = 3 :=
sorry

end find_k_value_l90_90611


namespace mutually_exclusive_not_opposing_l90_90552

-- Definition of the condition: bag with balls and the possible events when drawing two balls
structure BallBag where
  redBalls : ℕ
  blackBalls : ℕ
  totalBalls : ℕ := redBalls + blackBalls

def possibleEvents (bag : BallBag) : Set (Set String) :=
  { {"Exactly one black ball"}, {"Exactly two black balls"}, {"Exactly two red balls"} }

-- Given condition of the problem
def bag : BallBag := { redBalls := 2, blackBalls := 2 }

-- The proposition to be proved
theorem mutually_exclusive_not_opposing :
  ({"Exactly one black ball"}, {"Exactly two black balls"}) ∈ possibleEvents bag ∧
  MutuallyExclusiveNotOpposing ({"Exactly one black ball"}, {"Exactly two black balls"}) := by
  sorry

end mutually_exclusive_not_opposing_l90_90552


namespace rationalize_denominator_ABC_l90_90715

theorem rationalize_denominator_ABC :
  let expr := (2 + Real.sqrt 5) / (3 - 2 * Real.sqrt 5)
  ∃ A B C : ℤ, expr = A + B * Real.sqrt C ∧ A * B * (C:ℤ) = -560 :=
by
  sorry

end rationalize_denominator_ABC_l90_90715


namespace tan_angle_A_l90_90647

-- Define the sides and the right triangle condition.
variable (A B C : Type) [inner_product_space ℝ A]
variable (AB : ℝ) [is_leg_of_right_triangle A B AB]
variable (AC : ℝ) [is_leg_of_right_triangle A C AC]
variable (BC : ℝ) [is_hypotenuse_of_right_triangle B C AB]

-- Given conditions
variables (h1 : AB = real.sqrt 17) (h2 : AC = 4) (h3 : is_right_triangle B C)

-- Define the tan function
noncomputable def tan_angle (A B C : Type) [inner_product_space ℝ A] [is_right_triangle B C] : ℝ :=
  BC / AC

-- The statement we want to prove:
theorem tan_angle_A (h1 : AB = real.sqrt 17) (h2 : AC = 4) (h3 : is_right_triangle B C) :
  tan_angle A B C = 1 / 4 := 
sorry

end tan_angle_A_l90_90647


namespace abs_difference_lt_2t_l90_90048

/-- Given conditions of absolute values with respect to t -/
theorem abs_difference_lt_2t (x y s t : ℝ) (h₁ : |x - s| < t) (h₂ : |y - s| < t) :
  |x - y| < 2 * t :=
sorry

end abs_difference_lt_2t_l90_90048


namespace binomial_10_3_l90_90493

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90493


namespace line_equation_l90_90572

-- Defining the conditions of the problem
def pointP : ℝ × ℝ := (1, 2)
def sin_alpha : ℝ := 4/5

-- Target theorem we need to prove
theorem line_equation (x y : ℝ) :
  (∃ l : ℝ, l passes through pointP ∧ sin_alpha = 4/5) →
  (y = 2 + (4/3) * (x - 1) ∨ y = 2 - (4/3) * (x - 1)) :=
sorry -- The proof is omitted

end line_equation_l90_90572


namespace local_symmetry_point_cubic_local_symmetry_point_exponential_l90_90571

-- Define the properties of the function for Problem (1)
theorem local_symmetry_point_cubic (a b c : ℝ) : ∃ x0 : ℝ, (ax^3 + bx^2 + cx - b) = (-ax^3 + bx^2 - cx - b) := sorry

-- Define the properties of the function for Problem (2)
theorem local_symmetry_point_exponential (m : ℝ) : ∃ x0 ∈ set.Icc (-1 : ℝ) 2, (4^x + 2^x + m) = -(4^x + 2^x + m) :=
  ∃ m ∈ set.Icc (-325/32) (-27/8) := sorry

end local_symmetry_point_cubic_local_symmetry_point_exponential_l90_90571


namespace birds_and_storks_on_fence_l90_90242

theorem birds_and_storks_on_fence : 
  (initial_birds initial_storks new_birds : ℕ) 
  (h1 : initial_birds = 3) (h2 : initial_storks = 2) (h3 : new_birds = 5) :
  initial_birds + initial_storks + new_birds = 10 := by {
    sorry
  }

end birds_and_storks_on_fence_l90_90242


namespace factorial_power_of_two_iff_power_of_two_l90_90728

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end factorial_power_of_two_iff_power_of_two_l90_90728


namespace common_measure_of_segments_l90_90533

theorem common_measure_of_segments (a b : ℚ) (h₁ : a = 4 / 15) (h₂ : b = 8 / 21) : 
  (∃ (c : ℚ), c = 1 / 105 ∧ ∃ (n₁ n₂ : ℕ), a = n₁ * c ∧ b = n₂ * c) := 
by {
  sorry
}

end common_measure_of_segments_l90_90533


namespace math_problem_l90_90708

theorem math_problem (x y z : ℝ) (h_cond : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz + xy + yz + zx = x + y + z + 1) :
  (1/3) * (sqrt ((1 + x^2) / (1 + x)) + sqrt ((1 + y^2) / (1 + y)) + sqrt ((1 + z^2) / (1 + z))) 
  ≤ ((x + y + z)/3)^(5/8) :=
by
  sorry

end math_problem_l90_90708


namespace jessy_total_jewelry_l90_90659

def initial_necklaces := 10
def initial_earrings := 15
def initial_bracelets := 5
def initial_rings := 8

def store_a_necklaces := 10
def store_a_earrings := 2 / 3 * initial_earrings
def store_a_bracelets := 3

def store_b_rings := 2 * initial_rings
def store_b_necklaces := 4

def mother_gift_earrings := 1 / 5 * store_a_earrings

def total_necklaces := initial_necklaces + store_a_necklaces + store_b_necklaces
def total_earrings := initial_earrings + store_a_earrings + mother_gift_earrings
def total_bracelets := initial_bracelets + store_a_bracelets
def total_rings := initial_rings + store_b_rings

def total_jewelry := total_necklaces + total_earrings + total_bracelets + total_rings

theorem jessy_total_jewelry : total_jewelry = 83 := by
  -- Values from the problem to bind to expected results
  have h1 : total_necklaces = 24 := rfl
  have h2 : total_earrings = 27 := rfl
  have h3 : total_bracelets = 8 := rfl
  have h4 : total_rings = 24 := rfl
  rw [h1, h2, h3, h4]
  have h5 : 24 + 27 + 8 + 24 = 83 := rfl
  exact h5

end jessy_total_jewelry_l90_90659


namespace minimum_voters_for_tall_win_l90_90087

-- Definitions based on the conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3
def majority_precinct_voters : ℕ := 2
def majority_precincts_per_district : ℕ := 5
def majority_districts : ℕ := 3
def tall_won : Prop := true

-- Problem statement
theorem minimum_voters_for_tall_win : 
  tall_won → (∃ n : ℕ, n = 3 * 5 * 2 ∧ n ≤ voters) :=
by
  sorry

end minimum_voters_for_tall_win_l90_90087


namespace binomial_10_3_eq_120_l90_90428

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90428


namespace binomial_10_3_eq_120_l90_90450

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90450


namespace binom_10_3_eq_120_l90_90325

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90325


namespace find_slower_train_speed_l90_90214

theorem find_slower_train_speed (l : ℝ) (vf : ℝ) (t : ℝ) (v_s : ℝ) 
  (h1 : l = 37.5)   -- Length of each train
  (h2 : vf = 46)   -- Speed of the faster train in km/hr
  (h3 : t = 27)    -- Time in seconds to pass the slower train
  (h4 : (2 * l) = ((46 - v_s) * (5 / 18) * 27))   -- Distance covered at relative speed
  : v_s = 36 := 
sorry

end find_slower_train_speed_l90_90214


namespace perfect_square_trinomial_k_l90_90058

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a : ℤ, x^2 + k*x + 25 = (x + a)^2 ∧ a^2 = 25) → (k = 10 ∨ k = -10) :=
by
  sorry

end perfect_square_trinomial_k_l90_90058


namespace restaurant_profit_l90_90823

theorem restaurant_profit (cost_bottle : ℝ) (servings_per_bottle : ℝ) (price_per_serving : ℝ) (total_revenue : ℝ) (profit : ℝ) :
  cost_bottle = 30 ∧ servings_per_bottle = 16 ∧ price_per_serving = 8 ∧ total_revenue = servings_per_bottle * price_per_serving ∧ profit = total_revenue - cost_bottle → profit = 98 :=
by 
  intros h,
  cases h with hcst hsrv,
  cases hsrv with hprc hrvl,
  cases hrvl with hrvn hprf,
  cases hprf with hr hpr,
  rw [hr, ← hpr],
  sorry

end restaurant_profit_l90_90823


namespace cubic_eq_roots_product_l90_90506

theorem cubic_eq_roots_product :
  let a := 1
  let b := -9
  let c := 27
  let d := -5
  let f := polynomial.C a * polynomial.X^3 + polynomial.C b * polynomial.X^2 + polynomial.C c * polynomial.X + polynomial.C d
  roots_of_cubic_eq_a : a ≠ 0,
  f.eval = 0 ∧
  polynomial.eval (polynomial.C a * polynomial.X^3 + polynomial.C b * polynomial.X^2 + polynomial.C c * polynomial.X + polynomial.C d) = 0 ->
  product_of_roots :=
have pqr : Nat := -d/a 
pqr = 5 :=
begin
   sorry
end

end cubic_eq_roots_product_l90_90506


namespace students_who_liked_pears_l90_90152

theorem students_who_liked_pears :
  ∀ (total_students oranges apples strawberries : ℕ),
    total_students = 450 →
    oranges = 70 →
    apples = 147 →
    strawberries = 113 →
    total_students - (oranges + apples + strawberries) = 120 :=
by
  intros total_students oranges apples strawberries h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end students_who_liked_pears_l90_90152


namespace angle_BAC_eq_2_angle_BAM_l90_90685

variable {α : Type*} [metric_space α]

-- Given: ABC is a non-degenerate triangle
variable {A B C : α}
variable (non_degenerate_triangle : ¬(A = B ∨ B = C ∨ C = A))

-- Given: M is the midpoint of the arc between B and C that does not pass through A
variable {M : α}
variable (circumcircle : (A = B) ∨ (B = C) ∨ (C = A) ∨ (∀ P : α, ∃! (Q : α), dist A Q = dist B Q ∧ dist A Q = dist C Q))
variable (arc_midpoint : circle_arc_midpoint non_degenerate_triangle circumcircle M)

-- To show: ∠BAC = 2 ∠BAM
theorem angle_BAC_eq_2_angle_BAM :
  ∠BAC = 2 * ∠BAM :=
sorry

end angle_BAC_eq_2_angle_BAM_l90_90685


namespace f_neg5_eq_sqrt2_over_2_l90_90015

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.sin (x / 4 * Real.pi) else f (x + 2)

theorem f_neg5_eq_sqrt2_over_2 : f (-5) = Real.sqrt 2 / 2 :=
by
  sorry

end f_neg5_eq_sqrt2_over_2_l90_90015


namespace trajectory_of_P_is_circle_l90_90684

noncomputable
def ellipse_trajectory (a b : ℝ) (h : a > b ∧ b > 0) (F1 F2 Q P : ℝ × ℝ) : Prop :=
  -- Define the ellipse equation
  (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) ∧ 
  -- Define the foci positions (simplifications for the sake of this task)
  (F1.1 = -c ∧ F1.2 = 0 ∧ F2.1 = c ∧ F2.2 = 0 ∧ c^2 = a^2 - b^2) ∧
  -- Define the geometric properties described in the problem
  -- (Intersecting conditions are tricky in text form; assume they are represented properly) 
  ((distance Q F1 + distance Q F2 = 2 * a) → (distance P F2 = 2 * a)) 

theorem trajectory_of_P_is_circle (a b : ℝ) (h : a > b ∧ b > 0) (F1 F2 Q P : ℝ × ℝ) :
  ellipse_trajectory a b h F1 F2 Q P →
  ∃ r : ℝ, (r = 2 * a) ∧ (distance P F2 = r) :=
by
  sorry

end trajectory_of_P_is_circle_l90_90684


namespace arithmetic_geometric_properties_l90_90584

-- Definitions based on conditions
def arithmetic_prog (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def geometric_prog (b : ℕ → ℤ) : Prop :=
  ∃ q, q > 0 ∧ ∀ n, b (n + 1) = b n * q

-- Given initial conditions
variables (a b : ℕ → ℤ)
variables (h1 : a 1 = 2)
variables (h2 : b 1 = 2)
variables (h3 : a 2 = 2 * b 1 - 1)
variables (h4 : b 3 = 2 * a 2 + 2)

-- Main theorem to prove
theorem arithmetic_geometric_properties : 
  (arithmetic_prog a) → (geometric_prog b) → 
  (∀ n, a n = n + 1 ∧ b n = 2 ^ n) ∧
  (∀ n, (finset.sum (finset.range n) (λ n, a n * b n)) = n * 2 ^ (n + 1)) := 
by 
sory

end arithmetic_geometric_properties_l90_90584


namespace more_numbers_containing_9_not_divisible_by_9_l90_90274

theorem more_numbers_containing_9_not_divisible_by_9 :
  ∀ N : ℕ, N = 10^12 →
  let A := N - 9^12 in
  let B := N / 9 in
  let C := A - B in
  (C : ℝ) / N > 1 / 2 :=
by
  assume N hN
  let A := N - 9^12
  let B := N / 9
  let C := A - B
  have : (C : ℝ) / N > 1 / 2
  -- Additional details omitted intentionally
  sorry

end more_numbers_containing_9_not_divisible_by_9_l90_90274


namespace nth_equation_l90_90701

theorem nth_equation (n : ℕ) (hn : n > 0) : 
  (∑ k in Finset.range (2*n - 1), (n + k)) = (2*n - 1)^2 :=
by
  sorry

end nth_equation_l90_90701


namespace sum_of_unique_solutions_is_70_l90_90113

def isSolution (x y : ℝ) : Prop :=
  abs (x - 5) = abs (y - 12) ∧ abs (x - 12) = 3 * abs (y - 5)

def sumOfSolutions : ℝ :=
  ∑ i in [{9, 10.5, 12}, {16, 17.5, 5}], i

theorem sum_of_unique_solutions_is_70 :
  sumOfSolutions = 70 := by
  sorry

end sum_of_unique_solutions_is_70_l90_90113


namespace coloring_plane_existence_l90_90891

def Point := ℝ × ℝ

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def colored_plane (coloring : Point → Bool) (a : ℝ) (h : a > 0) : Prop :=
  ∃ p1 p2 : Point, coloring p1 ≠ coloring p2 ∧ distance p1 p2 = a

theorem coloring_plane_existence (coloring : Point → Bool) (h_color : ∃ p1 p2 : Point, coloring p1 ≠ coloring p2) :
  ∀ a > 0, colored_plane coloring a :=
by
  intros a ha
  sorry

end coloring_plane_existence_l90_90891


namespace sum_A_otimes_B_eq_zero_l90_90126

def A : set ℤ := {-2, 1}
def B : set ℤ := {-1, 2}

def A_otimes_B : set ℤ :=
  {x | ∃ (x1 : ℤ) (x2 : ℤ), x1 ∈ A ∧ x2 ∈ B ∧ x = x1 * x2 * (x1 + x2)}

theorem sum_A_otimes_B_eq_zero : (A_otimes_B.sum = 0) :=
by
  sorry

end sum_A_otimes_B_eq_zero_l90_90126


namespace binomial_10_3_l90_90496

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90496


namespace angle_equality_l90_90870

open_locale classical
noncomputable theory

structure Triangle (α : Type) := 
(A B C : α)

variables {α : Type} [metric_space α] [normed_group α] [normed_space ℝ α] 

def is_centroid (G : α) (T : Triangle α) :=
∃ (m n p : ℝ), m + n + p = 3 ∧ m • T.A + n • T.B + p • T.C = 3 • G

def is_circumcircle (O : α) (r : ℝ) (T : Triangle α) :=
∀ P, ∥P - O∥ = r ↔ P = T.A ∨ P = T.B ∨ P = T.C

variables (A B C D G1 G2 P : α)

def is_orthogonal (θ1 θ2 : α → ℝ) := θ1 = 0 ∧ θ2 = π / 2

variables (T1 T2 : Triangle α)
variables [has_orbit_of T1 G1] [has_orbit_of T2 G2]

def geometry_conditions :=
is_centroid G1 T1 ∧
is_centroid G2 T2 ∧
is_orthogonal (∠ (G1 - A) (G2 - A)) (∠ (A - G2) (C - G2)) ∧
is_circumcircle (circumcenter (triangle A G1 C)) (circumradius (triangle A G1 C)) T1 ∧
circle_intersects_line (circumcircle (triangle A G1 C)) (line B D) P

theorem angle_equality (h : geometry_conditions A B C D G1 G2 P T1 T2) :
  angle A P D = angle C P G1 :=
sorry

end angle_equality_l90_90870


namespace line_slope_intersect_l90_90837

theorem line_slope_intersect:
  ∃ (b : ℝ), ((-8:ℝ) * 4 + b = -3) ∧ (-8 + b = 21) :=
by {
  use 29,
  split,
  { -- Check the first condition: The point (4, -3) lies on the line y = -8x + b
    calc (-8 * 4 + 29 : ℝ)
        = (-32 + 29 : ℝ) : by norm_num
        = (-3 : ℝ) : by norm_num },
  { -- Check the second condition: m + b = -8 + 29 = 21
    calc (-8 + 29 : ℝ) = 21 : by norm_num }
}

end line_slope_intersect_l90_90837


namespace tangent_line_at_one_l90_90601

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + Real.log x

theorem tangent_line_at_one (a : ℝ)
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |(f a x - f a 1) / (x - 1) - 3| < ε) :
  ∃ m b, m = 3 ∧ b = -2 ∧ (∀ x y, y = f a x → m * x = y + b) := sorry

end tangent_line_at_one_l90_90601


namespace directrix_of_parabola_l90_90535

def parabola (x : ℝ) : ℝ :=
  (2 * x^2 - 8 * x + 6) / 16

theorem directrix_of_parabola :
  ∀ x, parabola x = -(3/2) :=
sorry

end directrix_of_parabola_l90_90535


namespace inequality_nonneg_reals_equality_collinear_vectors_l90_90119

theorem inequality_nonneg_reals
  {n : ℕ} {a x : Fin n → ℝ} {r : ℝ} (h_a : ∀ i, 0 ≤ a i)
  (h_x : ∀ i, 0 < x i) (h_r : 0 < r) :
  (∑ i, (a i) ^ (r + 1) / (x i) ^ r) ≥ ((∑ i, a i) ^ (r + 1)) / ((∑ i, x i) ^ r) :=
sorry

theorem equality_collinear_vectors
  {n : ℕ} {a x : Fin n → ℝ} {r : ℝ} (h_collinear : ∃ k, ∀ i, a i = k * x i)
  (h_r : 0 < r) (h_sum_x : (∑ i, x i) ≠ 0) :
  (∑ i, (a i) ^ (r + 1) / (x i) ^ r) = ((∑ i, a i) ^ (r + 1)) / ((∑ i, x i) ^ r) :=
sorry

end inequality_nonneg_reals_equality_collinear_vectors_l90_90119


namespace sum_mean_median_set_b_l90_90966

noncomputable def set_a := {17, 27, 31, 53, 61}

noncomputable def set_b : Set ℝ := 
  { (17:ℝ)^2, 
    2 * (27:ℝ), 
    (31:ℝ) / 2, 
    Real.sqrt (53:ℝ), 
    (61:ℝ) / 5 }

noncomputable def mean (s : Set ℝ) : ℝ :=
  (s.toList.sum) / ↑(s.size)

noncomputable def median (s : Set ℝ) : ℝ :=
  let orderedList := s.toList.qsort (· < ·)
  orderedList[orderedList.length / 2]

theorem sum_mean_median_set_b :
  mean set_b + median set_b = 98.296 :=
sorry

end sum_mean_median_set_b_l90_90966


namespace right_triangle_acute_angles_l90_90577

theorem right_triangle_acute_angles (c h : ℝ) (h_hyp : h = c / 4) (h_pos : 0 < c) :
  ∃ θ1 θ2 : ℝ, θ1 = 15 ∧ θ2 = 75 ∧ 
    (∀ (a b : ℝ), a * b = c^2 / 4 ∧ a^2 + b^2 = c^2 → 
    ((θ1 = real.arcsin (a / c) ∨ θ1 = real.arccos (b / c)) ∧ (θ2 = real.arcsin (b / c) ∨ θ2 = real.arccos (a / c)))) :=
begin
  sorry
end

end right_triangle_acute_angles_l90_90577


namespace binomial_10_3_eq_120_l90_90416

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90416


namespace ellipse_equation_trajectory_equation_l90_90012

-- Conditions
variables {a b x y : ℝ} (h1 : a > b > 0) (h2 : (sqrt 2)^2 / a^2 + 1^2 / b^2 = 1)
variables (h3 : (sqrt 2) / 2 = c / a) (h4 : c = sqrt 2) [noncomputable]

-- Ellipse equation
theorem ellipse_equation : (a = 2) → (b^2 = 2) → 
  (∀ (x y : ℝ), (x^2 / 4 + y^2 / 2 = 1) ↔ (x = sqrt 2 ∧ y = 1)) :=
sorry

-- Trajectory equation
theorem trajectory_equation (M N : (ℝ × ℝ)) 
  (hx1 : fst M ^ 2 / 4 + snd M ^ 2 / 2 = 1) 
  (hx2 : fst N ^ 2 / 4 + snd N ^ 2 / 2 = 1) 
  (h_slope : (snd M / fst M) * (snd N / fst N) = -1/2) :
  (∃ P : (ℝ × ℝ), snd P = snd M + 2 * snd N ∧ 
    (fst P)^2 / 20 + (snd P)^2 / 10 = 1) :=
sorry

end ellipse_equation_trajectory_equation_l90_90012


namespace volume_of_parallelepiped_l90_90146

theorem volume_of_parallelepiped (M N A1 C B1 A C1 : Type) (a b c : ℝ)
  (midpoint_M : midpoint A A1 = M) 
  (midpoint_N : midpoint C C1 = N)
  (perpendicular_A1C_B1M : is_perpendicular A1 C B1 M)
  (perpendicular_B1M_BN : is_perpendicular B1 M B N)
  (perpendicular_BN_A1C : is_perpendicular B N A1 C)
  (length_A1C : dist A1 C = a)
  (length_B1M : dist B1 M = b)
  (length_BN : dist B N = c) : 
  volume ABCDA1B1C1D1 = (1/2) * a * b * c :=
by
  sorry

end volume_of_parallelepiped_l90_90146


namespace domain_of_f_range_of_g_l90_90018

theorem domain_of_f (x : ℝ) 
  (h1 : (2 - x) / (3 + x) ≥ 0) 
  (h2 : 3^x - 1/3 > 0) : -1 < x ∧ x ≤ 2 := 
sorry

theorem range_of_g (x : ℝ) 
  (h : -1 < x ∧ x ≤ 2) : 
  let g := 4^(x + 1/2) - 2^(x + 2) + 1
  in -1 ≤ g ∧ g ≤ 17 := 
sorry

end domain_of_f_range_of_g_l90_90018


namespace find_x_l90_90042

theorem find_x:
  (∑ n in finset.range 1987, (n + 1) * (1988 - (n + 1))) = 1987 * 994 * 663 :=
by
  sorry

end find_x_l90_90042


namespace presidency_meeting_arrangement_count_l90_90252

-- Definition of the problem conditions
def numMembers := 16
def schools := 4
def membersPerSchool := 4
def hostChoices := 4

def combinations (n k : ℕ) : ℕ := @Nat.choose n k -- use combination formula

-- Proof statement: there are 1024 ways to arrange the presidency meeting
theorem presidency_meeting_arrangement_count :
  let hostSchoolChoices := hostChoices in
  let hostSchoolRepresentatives := combinations membersPerSchool 3 in
  let otherSchoolsRepresentatives := combinations membersPerSchool 1 in
  let totalNonHostWays := otherSchoolsRepresentatives ^ 3 in
  let totalWays := hostSchoolChoices * hostSchoolRepresentatives * totalNonHostWays in
  totalWays = 1024 :=
by {
  unfold combinations,
  sorry
}

end presidency_meeting_arrangement_count_l90_90252


namespace binom_10_3_eq_120_l90_90391

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90391


namespace seventh_oblong_is_56_l90_90866

def oblong (n : ℕ) : ℕ := n * (n + 1)

theorem seventh_oblong_is_56 : oblong 7 = 56 := by
  sorry

end seventh_oblong_is_56_l90_90866


namespace binom_10_3_l90_90440

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90440


namespace factorial_power_of_two_divisibility_l90_90722

def highestPowerOfTwoDividingFactorial (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), n / (2^k)

def binaryOnesCount (n : ℕ) : ℕ :=
  n.foldl (λ acc b, acc + if b then 1 else 0) 0

theorem factorial_power_of_two_divisibility (n : ℕ) :
  (n! % 2^(n - 1) = 0) ↔ (∃ k : ℕ, n = 2^k) :=
begin
  sorry
end

end factorial_power_of_two_divisibility_l90_90722


namespace largest_code_is_25916_l90_90911

def is_power_of_five (n : ℕ) : Prop :=
  n = 5 ∨ n = 25

def is_power_of_two (n : ℕ) : Prop :=
  n = 16 ∨ n = 64

def largest_code : option ℕ :=
  let codes := [25316, 25916].filter (λ code =>
    let digits := Int.digits 10 code
    digits.nodup ∧      -- No repeated digits
    0 ∉ digits ∧        -- None of the digits are zero
    is_power_of_five (digits.take 2).as_nat ∧  -- First two digits form a power of 5
    is_power_of_two (digits.drop 3).as_nat ∧   -- Last two digits form a power of 2
    (digits.nth 2).get % 3 = 0 ∧               -- Middle digit is a multiple of 3
    digits.sum % 2 = 1                         -- Sum of the digits is odd
  ) in
  codes.maximum

theorem largest_code_is_25916 : largest_code = some 25916 :=
by
  -- Proof would go here
  sorry

end largest_code_is_25916_l90_90911


namespace correct_calculation_l90_90230

theorem correct_calculation (m n : ℤ) :
  (m^2 * m^3 ≠ m^6) ∧
  (- (m - n) = -m + n) ∧
  (m * (m + n) ≠ m^2 + n) ∧
  ((m + n)^2 ≠ m^2 + n^2) :=
by sorry

end correct_calculation_l90_90230


namespace remainder_3_45_plus_4_mod_5_l90_90796

theorem remainder_3_45_plus_4_mod_5 :
  (3 ^ 45 + 4) % 5 = 2 := 
by {
  sorry
}

end remainder_3_45_plus_4_mod_5_l90_90796


namespace hyperbola_ratio_l90_90764

theorem hyperbola_ratio :
  let vertex := (1 : ℝ, 0 : ℝ)
  let focus := (2 : ℝ, 0 : ℝ)
  let asymptote := λ x : ℝ, sqrt 3 * x
  let hyperbola := λ x y : ℝ, x^2 - y^2 / 3 = 1
  let dist_to_asymptote := λ p : ℝ × ℝ, (abs(sqrt 3 * p.1 - p.2)) / sqrt (3 + 1)
  (dist_to_asymptote vertex / dist_to_asymptote focus) = 1 / 2 :=
by
  let vertex := (1 : ℝ, 0 : ℝ)
  let focus := (2 : ℝ, 0 : ℝ)
  let asymptote := λ x : ℝ, sqrt 3 * x
  let dist_to_asymptote := λ p : ℝ × ℝ, (abs(sqrt 3 * p.1 - p.2)) / sqrt(3 + 1)
  sorry

end hyperbola_ratio_l90_90764


namespace range_of_a_l90_90580

-- Conditions for proposition p: ∀ x ∈ R, ax^2 + 2ax + 3 > 0 holds true.
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 3 > 0

-- Conditions for proposition q: ∃ x ∈ R, x^2 + 2ax + a + 2 = 0.
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

-- Prove that the range of real number 'a'
theorem range_of_a : {a : ℝ | p a ∧ q a} = set.Ico 2 3 :=
sorry

end range_of_a_l90_90580


namespace cone_volume_correct_cone_lateral_surface_area_correct_l90_90523

-- Define the radius, height, and diameter
def radius (D : ℝ) := D / 2
def height (h : ℝ) := h

-- Cone volume formula
def cone_volume (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

-- Slant height calculation using Pythagorean theorem
def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)

-- Lateral surface area of a cone
def lateral_surface_area (r l : ℝ) : ℝ := Real.pi * r * l

-- Given diameter D = 12 cm and height h = 9 cm
def cone_diameter : ℝ := 12
def cone_height : ℝ := 9

-- Radius based on given diameter
def cone_radius : ℝ := radius cone_diameter
-- Slant height based on given radius and height
def cone_slant_height : ℝ := slant_height cone_radius cone_height

-- Define the expected volume and lateral surface area
def expected_volume : ℝ := 108 * Real.pi
def expected_lateral_surface_area : ℝ := 6 * Real.pi * Real.sqrt 117

theorem cone_volume_correct : cone_volume cone_radius cone_height = expected_volume := by
  sorry

theorem cone_lateral_surface_area_correct : lateral_surface_area cone_radius cone_slant_height = expected_lateral_surface_area := by
  sorry

end cone_volume_correct_cone_lateral_surface_area_correct_l90_90523


namespace sum_prime_factors_of_1260_l90_90801

theorem sum_prime_factors_of_1260 :
  ∃ (a b : ℕ), (∀ p, p.prime → p ∣ 1260 → p = a ∨ p = b) ∧ a + b = 9 :=
by
  sorry

end sum_prime_factors_of_1260_l90_90801


namespace binom_10_3_l90_90434

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90434


namespace sticker_price_is_200_l90_90970

-- Define the conditions
variable (x : ℝ) -- The sticker price of the computer

-- Store A's price with discount and rebate
def storeA_price : ℝ := 0.85 * x - 50

-- Store B's price with discount
def storeB_price : ℝ := 0.70 * x

-- Heather saves $20 by purchasing the computer from store A instead of store B
axiom heather_saves_20 : storeB_price x - storeA_price x = 20

-- Prove that the sticker price of the computer is $200
theorem sticker_price_is_200 (hx : x = 200) : 
  (let f (x : ℝ) := 0.85 * x - 50;
       g (x : ℝ) := 0.70 * x in 
  g(x) - f(x) = 20)
  ↔ (x = 200) := sorry

end sticker_price_is_200_l90_90970


namespace binomial_10_3_l90_90340

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90340


namespace like_terms_x_plus_y_l90_90624

theorem like_terms_x_plus_y
  (x y : ℕ)
  (h1 : x = 3)
  (h2 : y = 2)
  (like_terms : 5 * (a ^ x) * (b ^ 2) = -0.2 * (a ^ 3) * (b ^ y)) :
  x + y = 5 :=
by
  rw [h1, h2]
  apply congr_arg2 Nat.add h1 h2
  sorry  -- Proof of the final equality

end like_terms_x_plus_y_l90_90624


namespace isosceles_right_triangle_area_l90_90636

-- Definitions based on conditions
def isosceles_right_triangle (A B C : Type) [triangle A B C] : Prop :=
right_angle B ∧ isosceles A B C

def altitude_to_hypotenuse (BD : Type) (length : ℝ) : Prop :=
length = 4 ∧ is_median BD

noncomputable def triangle_area {A B C : Type} [triangle A B C]
  (BD : Type) [altitude_to_hypotenuse BD 4] : ℝ :=
1/2 * (base_length A B C) * (height_length A B C)

-- Statement to prove
theorem isosceles_right_triangle_area {A B C : Type} [triangle A B C]
  (BD : Type) [isosceles_right_triangle A B C] [altitude_to_hypotenuse BD 4] :
  triangle_area BD = 16 := by
  sorry

end isosceles_right_triangle_area_l90_90636


namespace total_pastries_l90_90831

variable (P x : ℕ)

theorem total_pastries (h1 : P = 28 * (10 + x)) (h2 : P = 49 * (4 + x)) : P = 392 := 
by 
  sorry

end total_pastries_l90_90831


namespace max_sum_length_x_y_l90_90816

def length (n : ℕ) : ℕ :=
  if h : n > 1 then
    (multiset.card ∘ multiset.filter (λ a, nat.prime a) ∘ multiset.map nat.factorization) n
  else
    sorry -- Length not defined for n <= 1

theorem max_sum_length_x_y {x y : ℕ} (hx : x > 1) (hy : y > 1) (h : x + 3 * y < 960) :
  length x + length y = 15 :=
sorry

end max_sum_length_x_y_l90_90816


namespace intersection_complement_eq_l90_90613

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x : ℝ | x^2 - 2 * x ≥ 3 }
def N_complement : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_complement_eq :
  M ∩ N_complement = {1, 2} :=
by
  sorry

end intersection_complement_eq_l90_90613


namespace pythagorean_table_black_white_sum_equal_l90_90630

/-- A rectangular frame with a thickness of one cell in a Pythagorean multiplication table,
where each side of the frame consists of an odd number of cells, has the property that the
sum of the numbers in the black cells equals the sum of the numbers in the white cells. -/
theorem pythagorean_table_black_white_sum_equal
  (m n : ℕ)
  (h_m : m % 2 = 1)
  (h_n : n % 2 = 1)
  (frame : (ℕ × ℕ) → ℕ)
  (h_frame : ∀ i j, frame (i, j) = (i + 1) * (j + 1))
  (black_cells white_cells : list (ℕ × ℕ))
  (h_alternate_color : ∀ cell, cell ∈ black_cells ∨ cell ∈ white_cells)
  (h_black : ∀ cell, cell ∈ black_cells → cell.1 % 2 = cell.2 % 2)
  (h_white : ∀ cell, cell ∈ white_cells → cell.1 % 2 ≠ cell.2 % 2)
  (h_corner_black : ∀ (i j : ℕ), (i = 0 ∨ i = m - 1) ∧ (j = 0 ∨ j = n - 1) → (i, j) ∈ black_cells) :
  ∑ cell in black_cells, frame cell = ∑ cell in white_cells, frame cell :=
sorry

end pythagorean_table_black_white_sum_equal_l90_90630


namespace concentric_circle_proof_l90_90037

theorem concentric_circle_proof
  (O A P B S T : ℝ → ℝ)
  (H1 : center O [A, P, B])
  (H2 : on_circle O A)
  (H3 : on_circle O P)
  (H4 : on_larger_circle O B)
  (H5 : is_perpendicular (line A P) (line B P))
  (H6 : midpoint S A B)
  (H7 : midpoint T O P) :
  distance S T = (1/2) * distance O B := 
sorry

end concentric_circle_proof_l90_90037


namespace find_x4_y4_z4_l90_90054

theorem find_x4_y4_z4
  (x y z : ℝ)
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59 / 3 :=
by
  sorry

end find_x4_y4_z4_l90_90054


namespace sequence_extrema_l90_90562

-- Definition of the sequence a_n
def a (n : ℕ) : ℝ :=
  (n - real.sqrt 2017) / (n - real.sqrt 2016)

-- Statement to prove
theorem sequence_extrema :
  (∀ n : ℕ, 1 ≤ n → n ≤ 100 → 
    if n = 45 then 
      a n = a 45
    else if n = 44 then 
      a n = a 44
    else true) ∧
  ∃ (n_min n_max : ℕ), 1 ≤ n_min ∧ n_min ≤ 100 ∧ 1 ≤ n_max ∧ n_max ≤ 100 ∧ 
  a n_min = a 45 ∧ 
  a n_max = a 44 ∧ 
  (∀ m : ℕ, 1 ≤ m → m ≤ 100 → a n_min ≤ a m ∧ a m ≤ a n_max) :=
sorry

end sequence_extrema_l90_90562


namespace transformed_function_is_correct_l90_90720

def f (x : ℝ) : ℝ := Math.sin (2 * x - Real.pi / 3)

theorem transformed_function_is_correct :
  ∀ x : ℝ, f (x + Real.pi / 3) = Math.sin (4 * x + Real.pi / 3) :=
by
  sorry

end transformed_function_is_correct_l90_90720


namespace floor_inequality_l90_90710

theorem floor_inequality (α β : ℝ) : 
  int.floor (2 * α) + int.floor (2 * β) ≥ int.floor α + int.floor β + int.floor (α + β) := 
by
  sorry

end floor_inequality_l90_90710


namespace donna_card_shop_days_per_week_l90_90908

-- Definitions and conditions
def earnings_dog_walking_per_day := 2 * 10  -- $/day
def earnings_babysitting_per_week := 4 * 10  -- $/week
def earnings_total_per_week := 305  -- total earnings in $ over 7 days

-- Card shop specific details
def hours_per_day_card_shop := 2
def rate_per_hour_card_shop := 12.5

-- Problem: Prove the number of days Donna worked at the card shop
theorem donna_card_shop_days_per_week : 
  let earnings_dog_walking_per_week := 7 * earnings_dog_walking_per_day in
  let earnings_other_jobs := earnings_dog_walking_per_week + earnings_babysitting_per_week in
  let earnings_card_shop := earnings_total_per_week - earnings_other_jobs in
  let earnings_per_day_card_shop := hours_per_day_card_shop * rate_per_hour_card_shop in
  earnings_card_shop / earnings_per_day_card_shop = 5 :=
by
  -- Placeholder proof
  sorry

end donna_card_shop_days_per_week_l90_90908


namespace combination_10_3_l90_90413

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90413


namespace binomial_10_3_eq_120_l90_90426

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90426


namespace parallel_tangent_line_exists_l90_90276

-- We introduce the necessary constructs to define our problem
def is_tangent_to_circle (a b c : ℝ) (radius : ℝ) : Prop :=
  let d := abs c / real.sqrt (a^2 + b^2) in d = radius

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

def line_eq (x y c : ℝ) : Prop := 2*x + y + c = 0

theorem parallel_tangent_line_exists {x y c : ℝ} 
  (h_circle : circle_eq x y) 
  (h_line : line_eq x y c) :
  is_tangent_to_circle 2 1 c (real.sqrt 5) →
  c = 5 :=
by 
  sorry

end parallel_tangent_line_exists_l90_90276


namespace binom_10_3_l90_90355

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90355


namespace megatek_manufacturing_percentage_l90_90161

variables {total_degrees manufacturing_degrees : ℕ}
-- Conditions:
def is_full_circle (total_degrees : ℕ) (percentage : ℕ) : Prop := 
  percentage = 100 ∧ total_degrees = 360

def manufacturing_sector (total_degrees manufacturing_degrees : ℕ) : Prop :=
  manufacturing_degrees = 144

def manufacturing_percentage (manufacturing_degrees total_degrees : ℕ) : ℝ :=
  (manufacturing_degrees.to_nat / total_degrees.to_nat) * 100

theorem megatek_manufacturing_percentage
  (h1 : is_full_circle total_degrees 100)
  (h2 : manufacturing_sector total_degrees manufacturing_degrees) :
  manufacturing_percentage manufacturing_degrees total_degrees = 40 :=
by
  simp [is_full_circle, manufacturing_sector, manufacturing_percentage] at *
  sorry

end megatek_manufacturing_percentage_l90_90161


namespace vector_magnitude_l90_90044

theorem vector_magnitude :
  let a : ℝ × ℝ × ℝ := (1, 2, 3)
  let b : ℝ × ℝ × ℝ := (0, 1, -4)
  let v := (a.1 - 2 * b.1, a.2 - 2 * b.2, a.3 - 2 * b.3)
  |(v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)| = sqrt 122 :=
by
  sorry

end vector_magnitude_l90_90044


namespace calculate_value_l90_90884

theorem calculate_value : (2 / 3 : ℝ)^0 + Real.log 2 + Real.log 5 = 2 :=
by 
  sorry

end calculate_value_l90_90884


namespace return_trip_time_l90_90844

theorem return_trip_time 
  (d p w : ℝ) 
  (h1 : d = 90 * (p - w))
  (h2 : ∀ t, t = d / p → d / (p + w) = t - 15) : 
  d / (p + w) = 64 :=
by
  sorry

end return_trip_time_l90_90844


namespace product_of_constants_l90_90117

theorem product_of_constants :
  ∃ M₁ M₂ : ℝ, 
    (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 82) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) ∧ 
    M₁ * M₂ = -424 :=
by
  sorry

end product_of_constants_l90_90117


namespace part1_arithmetic_sequence_part2_sum_of_sequence_l90_90643

-- Define the sequence a_n
def a : ℕ → ℕ
| 0 => 5
| (n + 1) => 2 * a n + 2^(n + 1) - 1

-- Define the sequence that we will prove is arithmetic
def arith_seq (n : ℕ) : ℕ := (a n - 1) / 2^n

-- Prove the arithmetic nature of the sequence
theorem part1_arithmetic_sequence : ∀ n ≥ 1, arith_seq n - arith_seq (n - 1) = 1 := by
  sorry

-- Define the sequence b_n
def b (n : ℕ) : ℤ := (-1)^n * (Int.log2 ((a n - 1) / (n + 1)))

-- Define S_n which is the sum of the sequence b_n
def S : ℕ → ℤ
| 0 => 0
| (n + 1) => S n + b n

-- Prove the formula for S_n
theorem part2_sum_of_sequence : ∀ n : ℕ,
  S n = if even n then (n : ℤ) / 2 else (-(n : ℤ) - 1) / 2 := by
  sorry

end part1_arithmetic_sequence_part2_sum_of_sequence_l90_90643


namespace range_of_a_l90_90605

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def g (a x : ℝ) := a * Real.sqrt x
noncomputable def f' (x₀ : ℝ) := Real.exp x₀
noncomputable def g' (a t : ℝ) := a / (2 * Real.sqrt t)

theorem range_of_a (a : ℝ) (x₀ t : ℝ) (hx₀ : x₀ = 1 - t) (ht_pos : t > 0)
  (h1 : f x₀ = Real.exp x₀)
  (h2 : g a t = a * Real.sqrt t)
  (h3 : f x₀ = g' a t)
  (h4 : (Real.exp x₀ - a * Real.sqrt t) / (x₀ - t) = Real.exp x₀) :
    0 < a ∧ a ≤ Real.sqrt (2 * Real.exp 1) :=
sorry

end range_of_a_l90_90605


namespace binom_10_3_eq_120_l90_90314

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90314


namespace binomial_10_3_eq_120_l90_90457

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90457


namespace binomial_10_3_l90_90497

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90497


namespace intersection_M_N_l90_90670

def M : Set ℤ := {x : ℤ | -x^2 + 3 * x > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 < 0}

theorem intersection_M_N : M ∩ {x : ℤ | x ∈ N} = {1} := by
  sorry

end intersection_M_N_l90_90670


namespace cost_of_tax_free_items_l90_90901

theorem cost_of_tax_free_items
  (total_amount_spent : ℝ)
  (sales_tax_paid : ℝ)
  (tax_rate : ℝ)
  (total_amount_spent_eq : total_amount_spent = 40)
  (sales_tax_paid_eq : sales_tax_paid = 0.30)
  (tax_rate_eq : tax_rate = 0.06)
  : total_amount_spent - (sales_tax_paid / tax_rate) = 35 := 
by
  rw [total_amount_spent_eq, sales_tax_paid_eq, tax_rate_eq]
  norm_num
  sorry

end cost_of_tax_free_items_l90_90901


namespace binom_10_3_l90_90343

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90343


namespace binomial_coefficient_10_3_l90_90359

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90359


namespace find_blue_yarn_count_l90_90132

def scarves_per_yarn : ℕ := 3
def red_yarn_count : ℕ := 2
def yellow_yarn_count : ℕ := 4
def total_scarves : ℕ := 36

def scarves_from_red_and_yellow : ℕ :=
  red_yarn_count * scarves_per_yarn + yellow_yarn_count * scarves_per_yarn

def blue_scarves : ℕ :=
  total_scarves - scarves_from_red_and_yellow

def blue_yarn_count : ℕ :=
  blue_scarves / scarves_per_yarn

theorem find_blue_yarn_count :
  blue_yarn_count = 6 :=
by 
  sorry

end find_blue_yarn_count_l90_90132


namespace washing_machine_capacity_l90_90637

theorem washing_machine_capacity (kylie_towels daughters_towels husband_towels loads : ℕ) 
  (hk: kylie_towels = 3) 
  (hd: daughters_towels = 6) 
  (hh: husband_towels = 3) 
  (hl: loads = 3) :
  (kylie_towels + daughters_towels + husband_towels) / loads = 4 :=
by
  rw [hk, hd, hh, hl]
  norm_num

end washing_machine_capacity_l90_90637


namespace complex_quadrant_l90_90011

open Complex

theorem complex_quadrant (i_squared : Complex.i * Complex.i = -1) :
  ∃ z : Complex, z = Complex.i * (1 + Complex.i) ∧ (z.re < 0) ∧ (z.im > 0) :=
by
  -- Define the complex number z as given in the conditions
  let z := Complex.i * (1 + Complex.i)
  -- Simplify z
  have z_def : z = Complex.i - 1 := by
    calc
      z = Complex.i * (1 + Complex.i) : rfl
      ... = Complex.i * 1 + Complex.i * Complex.i : by ring
      ... = Complex.i + Complex.i * Complex.i : by rw [Complex.i_mul_left, Complex.i_mul_i, add_zero]
      ... = Complex.i - 1 : by rw [i_squared]

  -- Point corresponding to the simplified z is (-1, 1)
  existsi z
  rw [z_def]
  split
  { rfl }
  split
  { calc
      z.re = -1.re : by sorry
      thus -1 < 0 by norm_num }
  { calc
      z.im = Complex.i.im : by sorry
      z.im = 1 : thus 1 > 0 by norm_num }

end complex_quadrant_l90_90011


namespace remainder_m_squared_plus_4m_plus_6_l90_90047

theorem remainder_m_squared_plus_4m_plus_6 (m : ℤ) (k : ℤ) (hk : m = 100 * k - 2) :
  (m ^ 2 + 4 * m + 6) % 100 = 2 := 
sorry

end remainder_m_squared_plus_4m_plus_6_l90_90047


namespace binom_10_3_eq_120_l90_90472

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90472


namespace problem_l90_90968

def universal_set : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {1, 2, 3}
def N : Finset ℕ := {0, 3, 4}

def complement (U B : Finset ℕ) : Finset ℕ :=
  U.filter (λ x => ¬ B.contains x)

theorem problem : (complement universal_set M) ∩ N = {0, 4} :=
by
  -- sorry, proof to be completed
  sorry

end problem_l90_90968


namespace binom_10_3_l90_90345

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90345


namespace locus_of_points_in_triangle_l90_90666

-- Define the problem
theorem locus_of_points_in_triangle 
  (A B C M O : Point)
  (h_triangle_acute : acute_triangle A B C)
  (h_interior_M : M ∈ interior A B C)
  (h_circumcenter : is_circumcenter O A B C)
  (F G : Point)
  (h_F_perp : foot_perpendicular M B C F)
  (h_G_perp : foot_perpendicular M A C G)
  (h_equation : AB - FG = (MF * AG + MG * BF) / CM) :
  lies_on_line_segment M C O :=
sorry

end locus_of_points_in_triangle_l90_90666


namespace card_distribution_unique_ways_l90_90241

theorem card_distribution_unique_ways :
  ∃ (c : Fin 2005 → Fin 2006), (∀ i : Fin 2005, ((c i).val + 1) % (i + 1) = 0) ∧
  (Finset.univ.image (λ i, c i)).card = 13 :=
sorry

end card_distribution_unique_ways_l90_90241


namespace maximize_p_at_incenter_l90_90665

noncomputable def p (M A B C A' B' C' : Point) : ℝ :=
MA' * MB' * MC' / (MA * MB * MC)

theorem maximize_p_at_incenter (A B C : Point) (M I : Point)
  (hA' : proj A' M BC) (hB' : proj B' M CA) (hC' : proj C' M AB)
  (hI : is_incenter I A B C) :

  ∃ M, p M A B C A' B' C' ≤ p I A B C A' B' C' :=
sorry

end maximize_p_at_incenter_l90_90665


namespace combination_10_3_eq_120_l90_90485

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90485


namespace binomial_coefficient_10_3_l90_90366

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90366


namespace rachel_should_budget_940_l90_90150

-- Define the prices for Sara's shoes and dress
def sara_shoes : ℝ := 50
def sara_dress : ℝ := 200

-- Define the prices for Tina's shoes and dress
def tina_shoes : ℝ := 70
def tina_dress : ℝ := 150

-- Define the total spending for Sara and Tina, and Rachel's budget
def rachel_budget (sara_shoes sara_dress tina_shoes tina_dress : ℝ) : ℝ := 
  2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_should_budget_940 : 
  rachel_budget sara_shoes sara_dress tina_shoes tina_dress = 940 := 
by
  -- skip the proof
  sorry 

end rachel_should_budget_940_l90_90150


namespace older_brother_allowance_l90_90062

theorem older_brother_allowance 
  (sum_allowance : ℕ)
  (difference : ℕ)
  (total_sum : sum_allowance = 12000)
  (additional_amount : difference = 1000) :
  ∃ (older_brother_allowance younger_brother_allowance : ℕ), 
    older_brother_allowance = younger_brother_allowance + difference ∧
    younger_brother_allowance + older_brother_allowance = sum_allowance ∧
    older_brother_allowance = 6500 :=
by {
  sorry
}

end older_brother_allowance_l90_90062


namespace binom_10_3_eq_120_l90_90388

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90388


namespace rocco_num_piles_of_nickels_l90_90719

def num_piles_of_nickels (quarters dimes nickels pennies : ℕ) : ℕ :=
  let total_value := 0.25 * (quarters * 10) + 0.1 * (dimes * 10) + 0.05 * (nickels * 10) + 0.01 * (pennies * 10)
  nickel_pile_value := 0.05 * 10
  required_value := 21.0 - (0.25 * (quarters * 10) + 0.1 * (dimes * 10) + 0.01 * (pennies * 10))
  required_value / nickel_pile_value

theorem rocco_num_piles_of_nickels : 
  ∀ (quarters dimes nickels pennies : ℕ),
  quarters = 4 → dimes = 6 → pennies = 5 → total_value = 21 → num_piles_of_nickels quarters dimes nickels pennies = 9 := 
by
  intros quarters dimes nickels pennies hq hd hp ht
  sorry

end rocco_num_piles_of_nickels_l90_90719


namespace problem_conditions_l90_90275

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def monotonic_increasing_on_positive (f : ℝ → ℝ) := ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y

theorem problem_conditions :
  (even_function (λ x, |x|) ∧ monotonic_increasing_on_positive (λ x, |x|)) ∧
  ¬ (even_function (λ x, x ^ 3) ∧ monotonic_increasing_on_positive (λ x, x ^ 3)) ∧
  (even_function (λ x, 2 ^ |x|) ∧ monotonic_increasing_on_positive (λ x, 2 ^ |x|)) ∧
  ¬ (even_function (λ x, 1 / (x ^ 2)) ∧ monotonic_increasing_on_positive (λ x, 1 / (x ^ 2))) :=
by sorry

end problem_conditions_l90_90275


namespace find_equal_black_white_cells_l90_90065

noncomputable def count_even_black_white_squares (grid : List (List Bool)) : Nat :=
  ∑ i in [0, 1, 2, 3], ∑ j in [0, 1, 2, 3], 
    if grid[i][j] != grid[i+1][j] || grid[i][j+1] != grid[i+1][j+1] then 1 else 0 +
  ∑ i in [0, 1], ∑ j in [0, 1], 
    if grid[i][j] != grid[i+1][j] || grid[i][j+2] != grid[i+1][j+2] ||
       grid[i+2][j] != grid[i+3][j] || grid[i+2][j+2] != grid[i+3][j+2] then 1 else 0

theorem find_equal_black_white_cells (grid : List (List Bool)) (H : grid.length = 5 ∧ grid.all (λ r, r.length = 5)):
  count_even_black_white_squares grid = 16 := sorry

end find_equal_black_white_cells_l90_90065


namespace binomial_10_3_l90_90327

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90327


namespace find_K_values_l90_90956

theorem find_K_values (K N : ℕ) (h1 : (N < 50)) : 
  (∃ K, (∑ i in finset.range (K + 1), i) = N^2) ↔ (K = 1 ∨ K = 8 ∨ K = 49) :=
by
  sorry

end find_K_values_l90_90956


namespace binom_10_3_l90_90438

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90438


namespace binomial_10_3_eq_120_l90_90424

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3_eq_120 : binomial 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90424


namespace binom_10_3_eq_120_l90_90322

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90322


namespace smallest_n_geometric_sequence_l90_90674

noncomputable def a_n (n : ℕ) := 3 ^ ((3 * n - 1) / 2)

theorem smallest_n_geometric_sequence :
  ∃ n : ℕ, (odd n ∧ (3 * n + 1) / 4 > 18 ∧ ∀ m : ℕ, odd m ∧ (3 * m + 1) / 4 > 18 → n ≤ m ) :=
begin
  use 25,
  split,
  { -- Proving that 25 is odd
    sorry },
  { -- Proving the inequality for the sum
    have h1 : (3 * 25 + 1) / 4 = 18.25, by norm_num,
    linarith,
    -- Ensuring it's the smallest odd integer satisfying the inequality
    sorry
  }
end

end smallest_n_geometric_sequence_l90_90674


namespace cm_eq_cn_l90_90575

-- Definitions given in the problem
variables {A B C D M N : Type}
variables [parallelogram ABCD] 
variables [on_side M AB] (hAD_DM : AD = DM)
variables [on_side N AD] (hAB_BN : AB = BN)

-- The corresponding Lean 4 statement to prove \( CM = CN \)
theorem cm_eq_cn (h_parallelogram : parallelogram ABCD) 
  (hAD_DM : AD = DM) (hAB_BN : AB = BN) : CM = CN := 
sorry

end cm_eq_cn_l90_90575


namespace exists_n_consecutive_non_primes_l90_90730

theorem exists_n_consecutive_non_primes (n : ℕ) (h : n ≥ 1) :
  ∃ seq : Fin n → ℕ, ∀ i, ¬ Nat.Prime (seq i) ∧ seq (i + 1) = (seq i) + 1 := 
sorry

end exists_n_consecutive_non_primes_l90_90730


namespace binomial_coefficient_10_3_l90_90361

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90361


namespace max_area_triangle_PCD_l90_90013

noncomputable def PCD_triangle_max_area (a b : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  (sqrt 2 - 1) / 2 * a * b

theorem max_area_triangle_PCD (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ P : ℝ × ℝ, 
    let m := P.1
    let n := P.2
    (m > 0 ∧ n < 0 ∧ (m^2 / a^2 + n^2 / b^2 = 1)) ∧
    PCD_triangle_max_area a b h = (sqrt 2 - 1) / 2 * a * b :=
sorry

end max_area_triangle_PCD_l90_90013


namespace binom_10_3_eq_120_l90_90313

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90313


namespace triangle_area_l90_90625

theorem triangle_area
  (a b c : ℝ)
  (h₁ : a = 26)
  (h₂ : b = 22)
  (h₃ : c = 10)
  (h₄ : a + b > c)
  (h₅ : a + c > b)
  (h₆ : b + c > a) :
  abs ((sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) - 107.76) < 0.01 :=
by
sorry

end triangle_area_l90_90625


namespace combination_10_3_eq_120_l90_90375

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90375


namespace sqrt_of_number_is_387_l90_90061

theorem sqrt_of_number_is_387 (x : ℝ) (h : sqrt x = 3.87) : x = 14.9769 :=
by sorry

end sqrt_of_number_is_387_l90_90061


namespace simplest_quadratic_radicals_same_type_l90_90626

theorem simplest_quadratic_radicals_same_type (m n : ℕ)
  (h : ∀ {a : ℕ}, (a = m - 1 → a = 2) ∧ (a = 4 * n - 1 → a = 7)) :
  m + n = 5 :=
sorry

end simplest_quadratic_radicals_same_type_l90_90626


namespace cdf_from_pdf_l90_90029

noncomputable def pdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.cos x
  else 0

noncomputable def cdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
  else 1

theorem cdf_from_pdf (x : ℝ) : 
  ∀ x : ℝ, cdf x = 
    if x ≤ 0 then 0
    else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
    else 1 :=
by
  sorry

end cdf_from_pdf_l90_90029


namespace pairB_same_function_pairA_not_same_pairC_not_same_pairD_not_same_l90_90809

def fxA (x : ℝ) : ℝ := Real.sqrt (x^2)
def gxA (x : ℝ) : ℝ := Real.sqrt (x) ^ 2

def fxB (x : ℝ) : ℝ := x
def gxB (x : ℝ) : ℝ := Real.cbrt (x^3)

def fxC (x : ℝ) : ℝ := 1
def gxC (x : ℝ) : ℝ := x^0

def fxD (x : ℝ) : ℝ := x + 1
def gxD (x : ℝ) : ℝ := if x = 1 then 0 else (x^2 - 1) / (x - 1) -- Handling for domain

theorem pairB_same_function : ∀ x : ℝ, fxB x = gxB x :=
by
  sorry -- Proof to be provided

theorem pairA_not_same : ∃ x : ℝ, fxA x ≠ gxA x
:= by
  sorry -- Proof to be provided

theorem pairC_not_same : ∃ x : ℝ, fxC x ≠ gxC x
:= by
  sorry -- Proof to be provided

theorem pairD_not_same : ∃ x : ℝ, fxD x ≠ gxD x
:= by
  sorry -- Proof to be provided

end pairB_same_function_pairA_not_same_pairC_not_same_pairD_not_same_l90_90809


namespace binomial_10_3_l90_90501

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90501


namespace range_a_l90_90963

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log a x
  else abs (x + 3)

theorem range_a (a : ℝ) :
  (∃ x y, f a x = f a y ∧ x = -y ∧ x ≠ 0 ∧ y ≠ 0) ↔ (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 4) := by
  sorry

end range_a_l90_90963


namespace regular_triangular_pyramid_ratio_is_sqrt3_l90_90763

noncomputable def regular_triangular_pyramid_ratio (a : ℝ) (S
_Plane_angle : ℝ) (H_plane_angle : S_Plane_angle = 90) : ℝ :=
  let SA := a * sqrt 2 / 2 in
  let SD := SA * sqrt 2 / 2 in
  let lateral_surface_area := (3 * a) / 2 * SD in
  let base_area := (a^2 * sqrt 3) / 4 in
  lateral_surface_area / base_area

theorem regular_triangular_pyramid_ratio_is_sqrt3 {a : ℝ} (S_Plane_angle : ℝ) (H_plane_angle : S_Plane_angle = 90)
  (H_pos : 0 < a) : regular_triangular_pyramid_ratio a S_Plane_angle H_plane_angle = sqrt 3 := by
  sorry

end regular_triangular_pyramid_ratio_is_sqrt3_l90_90763


namespace binomial_10_3_l90_90498

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90498


namespace machine_value_proof_l90_90259

noncomputable def machine_value_after_five_years (initial_value : ℝ) (salvage_value : ℝ) : ℝ :=
  let value_year_1 := initial_value * (1 - 0.10) in
  let value_year_2 := value_year_1 * (1 - 0.12) in
  let value_year_3 := value_year_2 * (1 - 0.15) in
  let value_year_4 := value_year_3 * (1 - 0.13) in
  let value_year_5 := value_year_4 * (1 - 0.11) in
  max value_year_5 salvage_value

theorem machine_value_proof :
  machine_value_after_five_years 1100 400 = 573.38 :=
by
  sorry

end machine_value_proof_l90_90259


namespace geometric_segment_l90_90994

theorem geometric_segment (AB A'B' : ℝ) (P D A B P' D' A' B' : ℝ) (x y a : ℝ) :
  AB = 3 ∧ A'B' = 6 ∧ (∀ P, dist P D = x) ∧ (∀ P', dist P' D' = 2 * x) ∧ x = a → x + y = 3 * a :=
by
  sorry

end geometric_segment_l90_90994


namespace fraction_of_area_above_line_l90_90180

theorem fraction_of_area_above_line :
  let A := (3, 2)
  let B := (6, 0)
  let side_length := B.fst - A.fst
  let square_area := side_length ^ 2
  let triangle_base := B.fst - A.fst
  let triangle_height := A.snd
  let triangle_area := (1 / 2 : ℚ) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  let fraction_above_line := area_above_line / square_area
  fraction_above_line = (2 / 3 : ℚ) :=
by
  sorry

end fraction_of_area_above_line_l90_90180


namespace vector_magnitude_example_l90_90034

open Real

noncomputable def a : ℝ × ℝ := (2, 0)  -- Use specific coordinates for simplicity
noncomputable def b : ℝ × ℝ := (1, sqrt 3 - 1)  -- Some cosine constraint playing

noncomputable def c : ℝ := 2 * sqrt 3

theorem vector_magnitude_example (a b : ℝ × ℝ) (hab : angle a b = π / 3) :
    ‖a + 2 • b‖ = 2 * sqrt 3 := by
  sorry

end vector_magnitude_example_l90_90034


namespace sin_diff_identity_l90_90878

variable (α β : ℝ)

def condition1 := (Real.sin α - Real.cos β = 3 / 4)
def condition2 := (Real.cos α + Real.sin β = -2 / 5)

theorem sin_diff_identity : 
  condition1 α β → 
  condition2 α β → 
  Real.sin (α - β) = 511 / 800 :=
by
  intros h1 h2
  sorry

end sin_diff_identity_l90_90878


namespace binom_10_3_eq_120_l90_90462

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90462


namespace last_digit_322_power_111569_l90_90514

theorem last_digit_322_power_111569 : (322 ^ 111569) % 10 = 2 := 
by {
  sorry
}

end last_digit_322_power_111569_l90_90514


namespace solve_equation_1_solve_equation_2_l90_90733

theorem solve_equation_1 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 ↔ x = 4 ∨ x = -6 := sorry

theorem solve_equation_2 (x : ℝ) : (1 / 2) * (x - 1)^3 = -4 ↔ x = -1 := sorry

end solve_equation_1_solve_equation_2_l90_90733


namespace find_abc_l90_90045

-- Definitions based on given conditions
variables (a b c : ℝ)
variable (h1 : a * b = 30 * (3 ^ (1/3)))
variable (h2 : a * c = 42 * (3 ^ (1/3)))
variable (h3 : b * c = 18 * (3 ^ (1/3)))

-- Formal statement of the proof problem
theorem find_abc : a * b * c = 90 * Real.sqrt 3 :=
by
  sorry

end find_abc_l90_90045


namespace sin_diff_identity_l90_90879

variable (α β : ℝ)

def condition1 := (Real.sin α - Real.cos β = 3 / 4)
def condition2 := (Real.cos α + Real.sin β = -2 / 5)

theorem sin_diff_identity : 
  condition1 α β → 
  condition2 α β → 
  Real.sin (α - β) = 511 / 800 :=
by
  intros h1 h2
  sorry

end sin_diff_identity_l90_90879


namespace binomial_coefficient_10_3_l90_90367

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90367


namespace smallest_a_value_l90_90739

theorem smallest_a_value :
  ∃ (a : ℝ), (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
    2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) ∧
    ∀ a' : ℝ, (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 →
      2 * (Real.sin (Real.pi - (Real.pi * x^2 / 12))) * (Real.cos (Real.pi / 6 * Real.sqrt (9 - x^2))) + 1 = a' + 2 * (Real.sin (Real.pi / 6 * Real.sqrt (9 - x^2))) * (Real.cos (Real.pi * x^2 / 12))) →
      a ≤ a'
  := sorry

end smallest_a_value_l90_90739


namespace solution_set_l90_90937

def f : ℝ → ℝ := sorry  -- Assume the existence of function f

axiom f_domain (x : ℝ) : true  -- Given that f is defined for all real numbers
axiom f_condition (x1 x2 : ℝ) (h : x1 < x2) : (f(x1) - f(x2)) / (x1 - x2) > -1  -- Given condition for f
axiom f_at_one : f 1 = 1  -- Given f(1) = 1

theorem solution_set (x : ℝ) :
  (f (Real.log2 (| 3 ^ x - 1 |)) < 2 - Real.log2 (| 3 ^ x - 1 |)) ↔ x ∈ Set.Ioo 0 1 ∪ Set.Iio 0 :=
by
  sorry

end solution_set_l90_90937


namespace squares_in_large_square_l90_90593

theorem squares_in_large_square (squares : List ℝ) (h : ℝ) :
  (∀ (x : ℝ), x ∈ squares → 0 ≤ x) ∧ (∑ square in squares, square ^ 2 = 1) →
  h = Real.sqrt 2 →
  ∃ arrangement : List (ℝ × ℝ), 
    (∀ i j, i ≠ j → arrangement.nth i ≠ none → arrangement.nth j ≠ none → 
      -- Ensure no overlapping
      let (x_i, y_i) := arrangement.nth i |>.get_or_else (0, 0);
      let (x_j, y_j) := arrangement.nth j |>.get_or_else (0, 0);
      let side_i := squares.nth i |>.get_or_else 0;
      let side_j := squares.nth j |>.get_or_else 0;
      (x_i + side_i ≤ x_j ∨ x_j + side_j ≤ x_i ∨ y_i + side_i ≤ y_j ∨ y_j + side_j ≤ y_i)) ∧
    (∀ (idx : ℕ), idx < squares.length →
      let (x, y) := arrangement.nth idx |>.get_or_else (0, 0);
      -- Ensure every square is within bounds of the large square
      x + squares.nth idx |>.get_or_else 0 ≤ h ∧ y + squares.nth idx |>.get_or_else 0 ≤ h)

end squares_in_large_square_l90_90593


namespace smallest_positive_period_of_f_l90_90540

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f(x) = f(x + T) ∧ ∀ T' > 0, (∀ x, f(x) = f(x + T') → T ≤ T') :=
sorry

def f (x : ℝ) : ℝ := sin x - 4 * (sin (x / 2)) ^ 3 * cos (x / 2)

end smallest_positive_period_of_f_l90_90540


namespace sequence_ends_with_31_62_63_l90_90852

theorem sequence_ends_with_31_62_63 :
  ∀ (a b: ℕ), 
    ((a, b) = (0, 1) ∨ (a, b) = (2 * 1, 2 * 1 + 1) ∨ 
     (a, b) = (2 * (2 * 1 + 1), 2 * (2 * 1 + 1) + 1) ∨ 
     (a, b) = (2 * (2 * (2 * 1 + 1) + 1), 2 * (2 * (2 * 1 + 1) + 1) + 1) ∨ 
     (a, b) = (2 * (2 * (2 * (2 * 1 + 1) + 1) + 1), 
               2 * (2 * (2 * (2 * 1 + 1) + 1) + 1) + 1) ∨ 
     (a, b) = (2 * (2 * (2 * (2 * (2 * 1 + 1) + 1) + 1) + 1), 
               2 * (2 * (2 * (2 * (2 * 1 + 1) + 1) + 1) + 1) + 1) ∨ 
     (a, b) = (62, 63))
    → ((2 * 31 = 62) ∧ (62 + 1 = 63) ∧ (31 = 31)) :=
begin
  sorry
end

end sequence_ends_with_31_62_63_l90_90852


namespace probability_condition_l90_90248

def balls : Nat := 12

def drawing_event (draws : List Nat) (target : Nat) : Prop :=
  ∃ ball, draws.filter (λ x, x = ball).length ≥ target

def all_balls_once (draws : List Nat) : Prop :=
  ∀ ball, draws.filter (λ x, x = ball).length > 0

def simulation_result :=
  0.02236412255

theorem probability_condition (h : ∀ draws : List Nat, 
  (drawing_event draws 12 ∨ all_balls_once draws) → drawing_event draws 12) :
  Prob(drawing_event [rand_ball (n+1) | n in range 144] 12) = simulation_result := sorry

end probability_condition_l90_90248


namespace probability_at_least_two_heads_l90_90130

theorem probability_at_least_two_heads (n : ℕ) (p : ℚ) (hn : n = 4) (hp : p = 1 / 2) :
  (1 - (nat.choose 4 0 * p ^ 0 * (1 - p) ^ 4 + nat.choose 4 1 * p ^ 1 * (1 - p) ^ 3)) = 11 / 16 :=
by
  sorry

end probability_at_least_two_heads_l90_90130


namespace usual_time_is_30_minutes_l90_90795

-- Define necessary variables and conditions
def usual_speed (D S : ℝ) : ℝ := D / S
def inclined_speed (D S : ℝ) : ℝ := 1.25 * D / (3 / 4 * S)
def time_difference (D S : ℝ) : ℝ := inclined_speed D S - usual_speed D S

-- We need to prove that the usual time is 30 minutes, given the conditions
theorem usual_time_is_30_minutes (D S : ℝ) (hD : D > 0) (hS : S > 0) (h20 : time_difference D S = 20) :
  usual_speed D S = 30 := 
sorry

end usual_time_is_30_minutes_l90_90795


namespace smallest_number_l90_90224

theorem smallest_number (x : ℕ) (h1 : ∀ d ∈ (digits 10 x), d = 3 ∨ d = 4) (h2 : (digits 10 x).sum = 15) : x = 3444 := sorry

end smallest_number_l90_90224


namespace binomial_10_3_l90_90336

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90336


namespace sum_prime_factors_1260_l90_90803

theorem sum_prime_factors_1260 : 
  let factors := {2, 3, 5, 7} in 
  (Finset.min' factors (by decide) + Finset.max' factors (by decide) = 9) := by
  sorry

end sum_prime_factors_1260_l90_90803


namespace arc_length_given_curve_l90_90300

noncomputable def arc_length_polar (ρ : ℝ → ℝ) (φ₀ φ₁ : ℝ) : ℝ :=
  ∫ φ in φ₀..φ₁, sqrt ((ρ φ)^2 + (deriv ρ φ)^2)

theorem arc_length_given_curve :
  arc_length_polar (λ φ, 5 * (1 - cos φ)) (-π / 3) 0 = 20 * (1 - sqrt 3 / 2) := sorry

end arc_length_given_curve_l90_90300


namespace cannot_eliminate_var_l90_90228

variables {R : Type*} [linear_ordered_field R]

noncomputable def eq1 (x y : R) : R := 5 * x + 2 * y
noncomputable def eq2 (x y : R) : R := 2 * x - 3 * y

theorem cannot_eliminate_var
  (x y : R)
  (h1 : eq1 x y = 4)
  (h2 : eq2 x y = 10) :
  ¬ (∀ y, ∀ x, 1.5 * eq1 x y - eq2 x y = 0) :=
sorry

end cannot_eliminate_var_l90_90228


namespace tile_difference_l90_90099

theorem tile_difference (initial_red : ℕ) (initial_yellow : ℕ) (additional_yellow : ℕ) :
  initial_red = 15 → 
  initial_yellow = 9 → 
  additional_yellow = 18 → 
  (initial_yellow + additional_yellow - initial_red = 12) :=
by
  assume h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end tile_difference_l90_90099


namespace probability_playing_exactly_one_instrument_l90_90068

-- Given conditions as definitions
def total_people : ℕ := 800
def fraction_playing_at_least_one_instrument : ℚ := 1 / 5
def people_playing_at_least_one_instrument : ℕ := (fraction_playing_at_least_one_instrument * total_people).toNat
def people_playing_two_or_more_instruments : ℕ := 128
def people_playing_exactly_one_instrument : ℕ := people_playing_at_least_one_instrument - people_playing_two_or_more_instruments

-- Probability calculation as a theorem
theorem probability_playing_exactly_one_instrument :
  (people_playing_exactly_one_instrument : ℚ) / total_people = 0.04 := by
  sorry

end probability_playing_exactly_one_instrument_l90_90068


namespace negation_of_proposition_l90_90759

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by
  sorry

end negation_of_proposition_l90_90759


namespace combination_10_3_l90_90408

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90408


namespace sum_of_squares_of_coefficients_l90_90804

-- Define the polynomial function
def poly := (x^4 + 2*x^3 + 3*x^2 + 2)

-- Define the multiplier
def multiplier := 5

-- Statement to prove the sum of the squares of the coefficients
theorem sum_of_squares_of_coefficients :
  let p := multiplier * poly in
  sum (p.coeffs.map (λ c, c^2)) = 450 := 
sorry

end sum_of_squares_of_coefficients_l90_90804


namespace cos_periodic_problem_l90_90906

theorem cos_periodic_problem : cos (2017 * π / 3) = 1 / 2 :=
by
  sorry

end cos_periodic_problem_l90_90906


namespace factorial_power_of_two_divisibility_l90_90721

def highestPowerOfTwoDividingFactorial (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), n / (2^k)

def binaryOnesCount (n : ℕ) : ℕ :=
  n.foldl (λ acc b, acc + if b then 1 else 0) 0

theorem factorial_power_of_two_divisibility (n : ℕ) :
  (n! % 2^(n - 1) = 0) ↔ (∃ k : ℕ, n = 2^k) :=
begin
  sorry
end

end factorial_power_of_two_divisibility_l90_90721


namespace min_value_of_sum_eq_l90_90558

theorem min_value_of_sum_eq : ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = a * b - 1 → a + 2 * b = 5 + 2 * Real.sqrt 6 :=
by
  intros a b h
  sorry

end min_value_of_sum_eq_l90_90558


namespace exists_polynomial_h_l90_90187

noncomputable def polynomial_h_exists (f g a : Polynomial ℝ) : Prop :=
  ∀ x y, f.eval x - f.eval y = a.eval₂ x y * (g.eval x - g.eval y)

theorem exists_polynomial_h (f g a : Polynomial ℝ) 
  (h : polynomial_h_exists f g a) : 
  ∃ (H : Polynomial ℝ), ∀ x, f.eval x = H.eval (g.eval x) := 
by
  sorry

end exists_polynomial_h_l90_90187


namespace combination_10_3_eq_120_l90_90377

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90377


namespace combination_10_3_l90_90409

open Nat

-- Define the combination formula
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Prove that combination of 10 choose 3 equals 120
theorem combination_10_3 : combination 10 3 = 120 := 
by
  sorry

end combination_10_3_l90_90409


namespace sine_function_symmetry_l90_90177

theorem sine_function_symmetry : 
  ∀ x, sin (π/3 + x) = sin (π/3 + (π/6 - x + π/6)) :=
by
  sorry

end sine_function_symmetry_l90_90177


namespace binom_10_3_l90_90349

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90349


namespace fare_per_1_5_mile_l90_90980

-- Definitions and conditions
def fare_first : ℝ := 1.0
def total_fare : ℝ := 7.3
def increments_per_mile : ℝ := 5.0
def total_miles : ℝ := 3.0
def remaining_increments : ℝ := (total_miles * increments_per_mile) - 1
def remaining_fare : ℝ := total_fare - fare_first

-- Theorem to prove
theorem fare_per_1_5_mile : remaining_fare / remaining_increments = 0.45 :=
by
  sorry

end fare_per_1_5_mile_l90_90980


namespace binomial_10_3_l90_90338

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90338


namespace min_tangent_length_from_point_on_line_to_circle_l90_90951

noncomputable def minTangentLength : ℝ :=
  let line := (λ x y => x + y + 2 = 0)
  let circle := (λ x y => x^2 + y^2 = 1)
  let center := (0 : ℝ, 0 : ℝ)
  let radius := 1
  let distToLine := (λ (p : ℝ × ℝ) => |p.1 + p.2 + 2| / Real.sqrt 2)
  let minDist := distToLine center
  Real.sqrt (minDist^2 - radius^2)

theorem min_tangent_length_from_point_on_line_to_circle :
  minTangentLength = 1 :=
by
  sorry

end min_tangent_length_from_point_on_line_to_circle_l90_90951


namespace binomial_10_3_l90_90492

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l90_90492


namespace Murtha_total_pebbles_15_days_l90_90135

-- Define the sequence of pebbles collected
def pebbles : ℕ → ℕ
| 0       := 0  -- No pebbles before starting
| 1       := 1  -- First day, 1 pebble
| (n + 1) :=
  if (n % 2 = 1)  -- If n is even, it means (n+1) is odd
  then 2 * pebbles n  -- Double the previous day's collection
  else pebbles n + 1  -- Increase by 1 compared to the previous day

/-- 
Prove that the total number of pebbles Murtha collects at the end of 
the fifteenth day is 152.
-/
theorem Murtha_total_pebbles_15_days : (List.range 15).sum (λ n, pebbles (n + 1)) = 152 :=
by
  -- The proof is omitted
  sorry

end Murtha_total_pebbles_15_days_l90_90135


namespace coprime_distinct_integer_count_l90_90949

noncomputable def distinct_integers_count (p q n : ℕ) : ℕ :=
  let r := max p q in
  if n < r then (n + 1) * (n + 2) / 2 else r * (2 * n - r + 3) / 2

theorem coprime_distinct_integer_count (p q n: ℕ) (hpq: Nat.coprime p q) (hn: 0 ≤ n):
  ∃ (count: ℕ), count = distinct_integers_count p q n := sorry

end coprime_distinct_integer_count_l90_90949


namespace shoe_selection_ways_l90_90776

theorem shoe_selection_ways : 
  let pairs := 10 in
  let total_shoes := 2 * pairs in
  let selected_shoes := 4 in
  let forming_a_pair := 2 in
  let remaining_pairs := pairs - 1 in
  let ways_to_select_pair := (Nat.choose pairs 1) in
  let ways_to_select_non_pairs := (Nat.choose remaining_pairs 2) * 4 in
  ways_to_select_pair * ways_to_select_non_pairs = 1440 := 
by
  sorry

end shoe_selection_ways_l90_90776


namespace expressions_equal_constant_generalized_identity_l90_90855

noncomputable def expr1 := (Real.sin (13 * Real.pi / 180))^2 + (Real.cos (17 * Real.pi / 180))^2 - Real.sin (13 * Real.pi / 180) * Real.cos (17 * Real.pi / 180)
noncomputable def expr2 := (Real.sin (15 * Real.pi / 180))^2 + (Real.cos (15 * Real.pi / 180))^2 - Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def expr3 := (Real.sin (-18 * Real.pi / 180))^2 + (Real.cos (48 * Real.pi / 180))^2 - Real.sin (-18 * Real.pi / 180) * Real.cos (48 * Real.pi / 180)
noncomputable def expr4 := (Real.sin (-25 * Real.pi / 180))^2 + (Real.cos (55 * Real.pi / 180))^2 - Real.sin (-25 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)

theorem expressions_equal_constant :
  expr1 = 3/4 ∧ expr2 = 3/4 ∧ expr3 = 3/4 ∧ expr4 = 3/4 :=
sorry

theorem generalized_identity (α : ℝ) :
  (Real.sin α)^2 + (Real.cos (30 * Real.pi / 180 - α))^2 - Real.sin α * Real.cos (30 * Real.pi / 180 - α) = 3 / 4 :=
sorry

end expressions_equal_constant_generalized_identity_l90_90855


namespace union_complement_with_B_l90_90694

namespace SetTheory

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of A relative to U in Lean
def C_U (A U : Set ℕ) : Set ℕ := U \ A

-- Theorem statement
theorem union_complement_with_B (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) : 
  (C_U A U) ∪ B = {2, 3, 4} :=
by
  -- Proof goes here
  sorry

end SetTheory

end union_complement_with_B_l90_90694


namespace common_ratio_of_geometric_seq_l90_90765

variable {a : ℕ → ℚ} -- The sequence
variable {d : ℚ} -- Common difference

-- Assuming the arithmetic and geometric sequence properties
def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (a1 a4 a5 : ℚ) (q : ℚ) : Prop :=
  a4 = a1 * q ∧ a5 = a4 * q

theorem common_ratio_of_geometric_seq (h_arith: is_arithmetic_seq a d) (h_nonzero_d : d ≠ 0)
  (h_geometric: is_geometric_seq (a 1) (a 4) (a 5) (1 / 3)) : (a 4 / a 1) = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l90_90765


namespace sin_cos_cos_sin_unique_pair_exists_uniq_l90_90731

noncomputable def theta (x : ℝ) : ℝ := Real.sin (Real.cos x) - x

theorem sin_cos_cos_sin_unique_pair_exists_uniq (h : 0 < c ∧ c < (1/2) * Real.pi ∧ 0 < d ∧ d < (1/2) * Real.pi) :
  (∃! (c d : ℝ), Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d ∧ c < d) :=
sorry

end sin_cos_cos_sin_unique_pair_exists_uniq_l90_90731


namespace solve_log_equation_solve_inequality_l90_90821

-- Part 1
theorem solve_log_equation (x : ℝ) (h1 : x + 1 > 0) (h2 : x - 2 > 0) : 
  (log (x + 1) + log (x - 2) = log 4) → x = 3 :=
by
  sorry

-- Part 2
theorem solve_inequality (x : ℝ) : 
  (2^(1 - 2 * x) > 1 / 4) → x < 3 / 2 :=
by
  sorry

end solve_log_equation_solve_inequality_l90_90821


namespace cyclists_speed_ratio_l90_90212

theorem cyclists_speed_ratio (k r t : ℝ) (h1 : 0 < r) (h2 : 0 < t) (h3 : 0 < k) :
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ (v1 - v2 = k / r) ∧ (v1 + v2 = k / t) ∧ v1 / v2 = (t + r) / (t - r) := 
by {
  assume (c₁), (c₂), (c₃),
  sorry
}

end cyclists_speed_ratio_l90_90212


namespace slope_angle_of_line_l90_90193

theorem slope_angle_of_line (x y : ℝ) (h : (√3) * x + 3 * y - 2 = 0) : 
  ∃ (α : ℝ), 0 ≤ α ∧ α < real.pi ∧ real.tan α = -(√3 / 3) ∧ α = 150 * real.pi / 180 :=
by
  sorry

end slope_angle_of_line_l90_90193


namespace binom_10_3_l90_90433

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90433


namespace should_agree_to_buy_discount_card_l90_90858

noncomputable def total_cost_without_discount_card (cakes_cost fruits_cost : ℕ) : ℕ :=
  cakes_cost + fruits_cost

noncomputable def total_cost_with_discount_card (cakes_cost fruits_cost discount_card_cost : ℕ) : ℕ :=
  let total_cost := cakes_cost + fruits_cost
  let discount := total_cost * 3 / 100
  (total_cost - discount) + discount_card_cost

theorem should_agree_to_buy_discount_card : 
  let cakes_cost := 4 * 500
  let fruits_cost := 1600
  let discount_card_cost := 100
  total_cost_with_discount_card cakes_cost fruits_cost discount_card_cost < total_cost_without_discount_card cakes_cost fruits_cost :=
by
  sorry

end should_agree_to_buy_discount_card_l90_90858


namespace sin_triple_angle_sin_18_degrees_l90_90794

-- First statement: 
theorem sin_triple_angle (α : ℝ) : sin (3 * α) = 3 * sin α - 4 * sin α ^ 3 := 
by sorry

-- Second statement:
theorem sin_18_degrees : sin (Real.pi / 10) = (Real.sqrt 5 - 1) / 4 :=
by sorry

end sin_triple_angle_sin_18_degrees_l90_90794


namespace translation_correctness_l90_90983

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x + 5

-- Define the translated function
def translated_function (x : ℝ) : ℝ := 3 * x

-- Define the condition for passing through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- The theorem to prove the correct translation
theorem translation_correctness : passes_through_origin translated_function := by
  sorry

end translation_correctness_l90_90983


namespace binomial_coefficient_10_3_l90_90363

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90363


namespace binom_10_3_l90_90348

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90348


namespace range_of_g_l90_90516

def g (x : ℝ) : ℝ := if x ≠ -5 then 3 * (x - 4) else 0 -- function definition 

theorem range_of_g : 
  (set.range g) = (set.Iio (-27) ∪ set.Ioi (-27)) := 
by sorry

end range_of_g_l90_90516


namespace minimum_value_of_a_plus_2b_l90_90560

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_2b_l90_90560


namespace binomial_10_3_eq_120_l90_90455

open Nat

theorem binomial_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_eq_120_l90_90455


namespace sum_divisible_by_5_and_7_remainder_12_l90_90283

theorem sum_divisible_by_5_and_7_remainder_12 :
  let a := 105
  let d := 35
  let n := 2013
  let S := (n * (2 * a + (n - 1) * d)) / 2
  S % 12 = 3 :=
by
  sorry

end sum_divisible_by_5_and_7_remainder_12_l90_90283


namespace sum_of_all_two_digit_numbers_l90_90308

theorem sum_of_all_two_digit_numbers : 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  sum_tens_place + sum_ones_place = 975 :=
by 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  show sum_tens_place + sum_ones_place = 975
  sorry

end sum_of_all_two_digit_numbers_l90_90308


namespace find_percentage_decrease_l90_90190

variable (original_salary : ℝ)
variable (raise_percent : ℝ)
variable (current_salary : ℝ)

theorem find_percentage_decrease :
  original_salary = 2000 →
  raise_percent = 10 →
  current_salary = 2090 →
  ∃ d : ℝ, d = 5 :=
by
  intros h₁ h₂ h₃
  let increased_salary := original_salary * (1 + raise_percent / 100)
  have h₄: increased_salary = 2200 := by
    rw [h₁, h₂]
    norm_num
  let decrease := (increased_salary - current_salary) / increased_salary * 100
  use decrease
  rw [h₄, h₃]
  norm_num
  sorry

end find_percentage_decrease_l90_90190


namespace three_planes_divide_space_l90_90986

theorem three_planes_divide_space :
  let x := 4
  let y := 8
  y - x = 4 := 
begin
  -- definitions as provided in the problem description
  let x := 4,
  let y := 8,
  -- proof is omitted
  sorry
end

end three_planes_divide_space_l90_90986


namespace average_salary_departmental_store_l90_90254

theorem average_salary_departmental_store
 (num_managers : ℕ) (num_associates : ℕ)
 (avg_salary_manager : ℝ) (avg_salary_associate : ℝ)
 (total_salary_managers : ℝ)
 (total_managers : ℝ)
 (total_salary_associates : ℝ)
 (total_associates : ℝ)
 (total_salary_all : ℝ)
 (total_employees : ℝ)
 (avg_salary_department : ℝ) :
 num_managers = 9 →
 num_associates = 18 →
 avg_salary_manager = 1300 →
 avg_salary_associate = 12000 →
 total_salary_managers = avg_salary_manager * num_managers →
 total_salary_associates = avg_salary_associate * num_associates →
 total_salary_all = total_salary_managers + total_salary_associates →
 total_employees = num_managers + num_associates →
 avg_salary_department = total_salary_all / total_employees →
 avg_salary_department = 8433.33 := 
by
  intros h_num_managers h_num_associates h_avg_salary_manager h_avg_salary_associate
          h_total_salary_managers h_total_salary_associates h_total_salary_all
          h_total_employees h_avg_salary_department
  rw [←h_num_managers, ←h_num_associates] at h_total_employees,
  rw [←h_avg_salary_manager, ←h_avg_salary_associate,
      ←h_total_salary_managers, ←h_total_salary_associates] at h_total_salary_all,
  rw [←h_total_salary_all, ←h_total_employees] at h_avg_salary_department,
  exact h_avg_salary_department


end average_salary_departmental_store_l90_90254


namespace speed_difference_l90_90527

-- Definitions derived from the conditions
def enrique_speed : ℕ := 16
def jamal_speed : ℕ := 23

-- Theorem statement equivalent to the problem
theorem speed_difference : jamal_speed - enrique_speed = 7 := 
by
  dsimp [jamal_speed, enrique_speed]
  norm_num

-- add 'sorry' to skip the proof and make the code syntactically complete
#reduce speed_difference

end speed_difference_l90_90527


namespace binom_10_3_eq_120_l90_90466

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l90_90466


namespace empirical_formula_BaCl2_l90_90828

-- Definitions directly from the conditions
def mass_percentage_Cl : ℝ := 33.82 / 100
def mass_percentage_Ba : ℝ := 1 - mass_percentage_Cl

-- Atomic masses
def atomic_mass_Ba : ℝ := 137.33
def atomic_mass_Cl : ℝ := 35.45

-- Mass in 100 grams of compound
def mass_Ba (compound_mass: ℝ) : ℝ := mass_percentage_Ba * compound_mass
def mass_Cl (compound_mass: ℝ) : ℝ := mass_percentage_Cl * compound_mass

-- Moles calculation
def moles_Ba (compound_mass: ℝ) : ℝ := (mass_Ba compound_mass) / atomic_mass_Ba
def moles_Cl (compound_mass: ℝ) : ℝ := (mass_Cl compound_mass) / atomic_mass_Cl

-- Ratios
def ratio_Ba_to_base (compound_mass : ℝ) : ℝ := moles_Ba compound_mass / moles_Ba compound_mass
def ratio_Cl_to_base (compound_mass : ℝ) : ℝ := moles_Cl compound_mass / moles_Ba compound_mass

theorem empirical_formula_BaCl2 (compound_mass : ℝ) (h : compound_mass = 100) : ratio_Ba_to_base compound_mass = 1 ∧ ratio_Cl_to_base compound_mass = 2 :=
by
  sorry

end empirical_formula_BaCl2_l90_90828


namespace distance_between_riya_and_priya_l90_90718

theorem distance_between_riya_and_priya (speed_riya speed_priya : ℝ) (time_hours : ℝ)
  (h1 : speed_riya = 21) (h2 : speed_priya = 22) (h3 : time_hours = 1) :
  speed_riya * time_hours + speed_priya * time_hours = 43 := by
  sorry

end distance_between_riya_and_priya_l90_90718


namespace sum_third_fourth_l90_90632

noncomputable def sequence : ℕ → ℚ
| 0 => 1
| (n+1) => ((n+1) / n)^3 * sequence n

theorem sum_third_fourth :
  sequence 2 + sequence 3 = (1241 / 216 : ℚ) :=
by
  sorry

end sum_third_fourth_l90_90632


namespace factorial_power_of_two_iff_power_of_two_l90_90729

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end factorial_power_of_two_iff_power_of_two_l90_90729


namespace number_of_valid_pairs_l90_90091

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_valid_pair (a b : ℕ) : Prop :=
  (3 * a + b = 148) ∧ (a > b)

theorem number_of_valid_pairs :
  (∃! pairs, count (λ p : ℕ × ℕ, is_valid_pair p.1 p.2) pairs = 12) :=
sorry

end number_of_valid_pairs_l90_90091


namespace partition_equilateral_triangle_l90_90903

theorem partition_equilateral_triangle (x : ℝ) (h : 0 < x ∧ x < 180) :
  (∃ (triangles : List (Triangle ℝ)), (∀ (t ∈ triangles), has_angle t x) ∧ partition triangle triangles) ↔ (x ∈ Ioc 0 120) :=
sorry

end partition_equilateral_triangle_l90_90903


namespace binom_10_3_l90_90350

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90350


namespace maximum_p_value_l90_90566

noncomputable def max_p_value (a b c : ℝ) : ℝ :=
  2 / (a^2 + 1) - 2 / (b^2 + 1) + 3 / (c^2 + 1)

theorem maximum_p_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a + c = b) :
  ∃ p_max, p_max = 10 / 3 ∧ ∀ p, p = max_p_value a b c → p ≤ p_max :=
sorry

end maximum_p_value_l90_90566


namespace combination_10_3_eq_120_l90_90384

theorem combination_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end combination_10_3_eq_120_l90_90384


namespace cost_of_shoes_l90_90784

theorem cost_of_shoes (allowance : ℕ) (months_saved : ℕ) (mow_charge : ℕ) (num_lawns : ℕ) (shovel_charge : ℕ) (num_driveways : ℕ) (change_left : ℕ) :
  let total_savings := allowance * months_saved + mow_charge * num_lawns + shovel_charge * num_driveways
  in total_savings - change_left = 95 :=
by
  let total_savings := allowance * months_saved + mow_charge * num_lawns + shovel_charge * num_driveways
  show total_savings - change_left = 95, from sorry

end cost_of_shoes_l90_90784


namespace subtract_and_round_l90_90740

def round_to_nearest_hundredth (n : ℚ) : ℚ :=
  (n * 100).round / 100

theorem subtract_and_round :
  let a := 221.54321
  let b := 134.98765
  let result := a - b
  let rounded_result := round_to_nearest_hundredth result
  rounded_result = 86.56 :=
by
  let a : ℚ := 221.54321
  let b : ℚ := 134.98765
  let result := a - b
  let rounded_result := round_to_nearest_hundredth result
  show rounded_result = 86.56
  sorry

end subtract_and_round_l90_90740


namespace domain_of_function_l90_90749

-- Definitions of the conditions
def condition1 (x : ℝ) : Prop := x - 5 ≠ 0
def condition2 (x : ℝ) : Prop := x - 2 > 0

-- The theorem stating the domain of the function
theorem domain_of_function (x : ℝ) : condition1 x ∧ condition2 x ↔ 2 < x ∧ x ≠ 5 :=
by
  sorry

end domain_of_function_l90_90749


namespace exists_real_a_l90_90944

noncomputable def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem exists_real_a : ∃ a : ℝ, a = -2 ∧ A a ∩ C = ∅ ∧ ∅ ⊂ A a ∩ B := 
by {
  sorry
}

end exists_real_a_l90_90944


namespace intersection_of_curves_l90_90026

theorem intersection_of_curves 
  (t : ℝ) -- Parameter t
  (x y : ℝ) -- Rectangular coordinates
  (hC1x : x = real.sqrt t) -- Parametric equation x = √t
  (hC1y : y = real.sqrt (3 * t) / 3) -- Parametric equation y = √(3t)/3
  (hrho : x^2 + y^2 = 4) -- Polar equation ρ = 2 transformed to rectangular coordinates
  (hx_pos : 0 ≤ x) (hy_pos : 0 ≤ y) -- Non-negative conditions
  :
  x = real.sqrt 3 ∧ y = 1 :=
sorry

end intersection_of_curves_l90_90026


namespace polynomial_evaluation_l90_90528

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 3 * x - 10 = 0) (h2 : 0 < x) : 
  x^3 - 3 * x^2 - 10 * x + 5 = 5 :=
sorry

end polynomial_evaluation_l90_90528


namespace dried_grapes_water_percentage_l90_90924

theorem dried_grapes_water_percentage
  (fresh_grapes_weight : ℝ)
  (fresh_grapes_water_percentage : ℝ)
  (dried_grapes_weight : ℝ) :
  fresh_grapes_weight = 100 →
  fresh_grapes_water_percentage = 70 →
  dried_grapes_weight = 33.33333333333333 →
  (∃ P : ℝ, P = 10) :=
by
  intro h1 h2 h3
  use 10
  sorry

end dried_grapes_water_percentage_l90_90924


namespace max_PA_AB_over_PB_l90_90587

variable (P A B : Point)
variable (cone : is_right_circular_cone P)
variable (isosceles : is_isosceles_right_triangle (cross_section cone))
variable (generatrix : is_generatrix PA cone)
variable (point_base : is_point_on_base B cone)

theorem max_PA_AB_over_PB :
  ‖PA + AB‖ / ‖PB‖ ≤ sqrt (4 + 2 * sqrt 2) := by
  sorry

end max_PA_AB_over_PB_l90_90587


namespace binom_10_3_eq_120_l90_90399

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l90_90399


namespace number_of_ways_to_break_targets_l90_90633

theorem number_of_ways_to_break_targets :
  let target_seq : Multiset.Char := multiset.of_list ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z'] in
  (target_seq.powerset_len 8).card / (multiset.of_list ['X', 'X', 'X']).card! / (multiset.of_list ['Y', 'Y', 'Y']).card! / (multiset.of_list ['Z', 'Z']).card! = 560 := 
by
  let target_seq := ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Z', 'Z'];
  have h_mult_seq : multiset.of_list target_seq = multiset.of_list ['X', 'X', 'X'] + multiset.of_list ['Y', 'Y', 'Y'] + multiset.of_list ['Z', 'Z'] :=
    by rw [multiset.of_list, multiset.of_list, multiset.of_list, multiset.of_list, multiset.of_list, multiset.of_list]; refl;
  rw [h_mult_seq];
  sorry

end number_of_ways_to_break_targets_l90_90633


namespace line_parallel_plane_l90_90673

-- Given a line a and a plane alpha
variables (a : Line) (α : Plane)
-- There exists a plane beta such that a ⊆ beta and α ∥ beta
variables (β : Plane)
hypothesis (h1 : a ⊆ β)
hypothesis (h2 : α ∥ β)

-- We need to prove that a ∥ α
theorem line_parallel_plane (a : Line) (α β : Plane) (h1 : a ⊆ β) (h2 : α ∥ β) : a ∥ α :=
sorry

end line_parallel_plane_l90_90673


namespace range_of_a_for_line_in_fourth_quadrant_l90_90055

theorem range_of_a_for_line_in_fourth_quadrant (a : ℝ) : 
  ((a-2) * x + a * y + 2 * a - 3 = 0 → 
    x > 0 ∧ y < 0) ↔ 
  (a < 0 ∨ a > (3 / 2)) :=
begin
  sorry
end

end range_of_a_for_line_in_fourth_quadrant_l90_90055


namespace alpha_beta_30_l90_90008

noncomputable def a_n (n : ℕ) : ℕ := 6 * n - 3
def b_n (n : ℕ) : ℕ := 9 ^ (n - 1)

theorem alpha_beta_30 :
  (∃ (α β : ℝ), ∀ (n : ℕ), a_n n = log α (b_n n) + β) → 
  let α := 27 in
  let β := 3 in
  α + β = 30 :=
by
  intro h
  simp [←h]
  sorry

end alpha_beta_30_l90_90008


namespace tank_filled_percentage_l90_90881

def tank_volume (length width height : ℕ) : ℕ := length * width * height

def effective_urn_volume (urn_volume fraction : ℝ) : ℝ := urn_volume * fraction

def total_water_volume (urns effective_urn_volume : ℝ) : ℝ := urns * effective_urn_volume

def percentage_filled (total_water_volume tank_volume : ℝ) : ℝ := (total_water_volume / tank_volume) * 100

theorem tank_filled_percentage :
  let tank_length := 10
      tank_width := 10
      tank_height := 5
      urn_count := 703.125
      urn_volume := 0.8
      fill_fraction := 0.8
      
      V_tank := tank_volume tank_length tank_width tank_height
      V_urn_effective := effective_urn_volume urn_volume fill_fraction
      V_total_water := total_water_volume urn_count V_urn_effective
      P_filled := percentage_filled V_total_water V_tank
  in P_filled = 89.92 :=
by {
  -- all calculations are encapsulated in definitions
  sorry
}

end tank_filled_percentage_l90_90881


namespace cost_of_milk_powder_july_l90_90063

noncomputable def cost_in_july (x : ℝ) : ℝ :=
  0.4 * (0.2 * x) + 0.35 * (4 * x) + 0.25 * (1.45 * x)

theorem cost_of_milk_powder_july (h : cost_in_july x = 2.925) :
  0.2 * x ≈ 0.3174 := by
  sorry

end cost_of_milk_powder_july_l90_90063


namespace binomial_10_3_l90_90329

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l90_90329


namespace bobby_candy_total_l90_90298

theorem bobby_candy_total :
  let x := 89 in
  let y := 152 in
  x + y = 241 :=
by
  -- Skipping the proof for now
  sorry

end bobby_candy_total_l90_90298


namespace binomial_coefficient_10_3_l90_90368

theorem binomial_coefficient_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_coefficient_10_3_l90_90368


namespace part1_monotonically_decreasing_interval_part2_range_of_a_l90_90960

/-
Definition and properties for the function f(x) for part 1.
-/
def f (x : ℝ) : ℝ := 2 * x * Real.log x - x^2

/-
Part 1: Prove the function f(x) is monotonically decreasing in the interval (0, +∞) when a = 1/2.
-/
theorem part1_monotonically_decreasing_interval :
  ∀ x : ℝ, 0 < x → ∀ y : ℝ, x < y → f' x ≥ f' y :=
sorry

/-
Definition and properties for the function f(x) for part 2.
-/
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x * Real.log x - 2 * a * x^2

/-
Part 2: Prove that if f(x) satisfies the given inequality, then a ∈ [1, +∞).
-/
theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a ≤ f' x / 2 - Real.log x - 1) →
  (1 ≤ a) :=
sorry

end part1_monotonically_decreasing_interval_part2_range_of_a_l90_90960


namespace rowers_voted_l90_90189

theorem rowers_voted (c : ℕ) (v : ℕ) (r : ℕ) : (r * 3 = 36 * 5) → r = 60 := by
-- 36 coaches, each received 5 votes, and each rower voted for 3 coaches
  assume h : r * 3 = 180
  have h1 : r = 180 / 3, from Nat.eq_div_of_mul_eq_left (by norm_num) h
  show r = 60, by rw [h1]; norm_num

end rowers_voted_l90_90189


namespace problem_magnitude_l90_90622

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem problem_magnitude (m : ℝ) (h1 : is_pure_imaginary ((1 + complex.I * m) * (3 + complex.I) * complex.I))
(h2 : is_pure_imaginary complex.I * m) :
  complex.abs ((m + 3 * complex.I) / (1 - complex.I)) = 3 :=
sorry

end problem_magnitude_l90_90622


namespace axisymmetric_triangle_is_isosceles_l90_90051

-- Define a triangle and its properties
structure Triangle :=
  (a b c : ℝ) -- Triangle sides as real numbers
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

def is_axisymmetric (T : Triangle) : Prop :=
  -- Here define what it means for a triangle to be axisymmetric
  -- This is often represented as having at least two sides equal
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

def is_isosceles (T : Triangle) : Prop :=
  -- Definition of an isosceles triangle
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

-- The theorem to be proven
theorem axisymmetric_triangle_is_isosceles (T : Triangle) (h : is_axisymmetric T) : is_isosceles T :=
by {
  -- Proof would go here
  sorry
}

end axisymmetric_triangle_is_isosceles_l90_90051


namespace factorial_power_of_two_iff_power_of_two_l90_90727

-- Assuming n is a positive integer
variable {n : ℕ} (h : n > 0)

theorem factorial_power_of_two_iff_power_of_two :
  (∃ k : ℕ, n = 2^k ) ↔ ∃ m : ℕ, 2^(n-1) ∣ n! :=
by {
  sorry
}

end factorial_power_of_two_iff_power_of_two_l90_90727


namespace total_money_collected_l90_90269

theorem total_money_collected
    (n : ℕ := 25)
    (painter_charge : ℕ := 2)
    (north_start : ℤ := 5) (north_diff : ℤ := 7)
    (south_start : ℤ := 6) (south_diff : ℤ := 8) :
    let north_houses := (list.range n).map (λ i, north_start + i * north_diff),
        south_houses := (list.range n).map (λ i, south_start + i * south_diff),
        digit_cost := λ num, painter_charge * (int.to_nat (num.toString.length))
    in (north_houses.map digit_cost).sum + (south_houses.map digit_cost).sum = 234 := sorry

end total_money_collected_l90_90269


namespace average_books_per_month_l90_90826

-- Definitions based on the conditions
def books_sold_january : ℕ := 15
def books_sold_february : ℕ := 16
def books_sold_march : ℕ := 17
def total_books_sold : ℕ := books_sold_january + books_sold_february + books_sold_march
def number_of_months : ℕ := 3

-- The theorem we need to prove
theorem average_books_per_month : total_books_sold / number_of_months = 16 :=
by
  sorry

end average_books_per_month_l90_90826


namespace range_of_a_l90_90969

variable {a : ℝ}

-- Proposition p: The solution set of the inequality x^2 - (a+1)x + 1 ≤ 0 is empty
def prop_p (a : ℝ) : Prop := (a + 1) ^ 2 - 4 < 0 

-- Proposition q: The function f(x) = (a+1)^x is increasing within its domain
def prop_q (a : ℝ) : Prop := a > 0 

-- The combined conditions
def combined_conditions (a : ℝ) : Prop := (prop_p a) ∨ (prop_q a) ∧ ¬(prop_p a ∧ prop_q a)

-- The range of values for a
theorem range_of_a (h : combined_conditions a) : -3 < a ∧ a ≤ 0 ∨ a ≥ 1 :=
  sorry

end range_of_a_l90_90969


namespace quadratic_has_two_distinct_real_roots_l90_90192

theorem quadratic_has_two_distinct_real_roots :
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 :=
by
  let a := 1
  let b := 1
  let c := -1
  let discriminant := b^2 - 4 * a * c
  show discriminant > 0
  sorry

end quadratic_has_two_distinct_real_roots_l90_90192


namespace angle_AOD_is_120_deg_l90_90080

variable (OA OC OB OD : Type)
variable (A O B C D : OA)
variable (angle : OA → OC → ℝ)

/-- Given a quadrilateral with properties described below, prove the angle AOD is 120 degrees. -/
theorem angle_AOD_is_120_deg
  (h1 : ∀ v w : OA, angle v w = 90 → angle w v = 90)
  (h2 : angle O A = 90 → angle O C = 90)
  (h3 : angle O B = 90 → angle O D = 90)
  (h4 : angle O A = 3 * angle O B)
  (h5 : angle O B O D = 100)
  (h6 : angle O C O A = 100) :
  angle O A O D = 120 :=
  sorry

end angle_AOD_is_120_deg_l90_90080


namespace combination_10_3_eq_120_l90_90488

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90488


namespace combination_10_3_eq_120_l90_90481

open Nat

theorem combination_10_3_eq_120 : (10.choose 3) = 120 := 
by
  sorry

end combination_10_3_eq_120_l90_90481


namespace range_of_a_l90_90676

noncomputable def f (a x : ℝ) : ℝ :=
  log ((1 + 2^x + 4^x * a) / 3)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -3/4 :=
by
  sorry

end range_of_a_l90_90676


namespace find_room_breadth_l90_90757

noncomputable def room_breadth (b : ℕ) : Prop :=
  let length := 20 in
  let tile_side := 2 in
  let black_tile_border := 2 * tile_side in
  let inner_length := length - black_tile_border in
  let inner_breadth := b - black_tile_border in
  let inner_area := inner_length * inner_breadth in
  let blue_tiles_amount := 16 in
  let blue_tile_area := blue_tiles_amount * (tile_side * tile_side) in
  let remaining_area_ratio := 2 / 3 in
  remaining_area_ratio * inner_area = blue_tile_area

theorem find_room_breadth : ∃ b : ℕ, room_breadth b ∧ b = 10 :=
by
  use 10
  unfold room_breadth
  rw [←mul_div_assoc', div_eq_mul_inv, mul_assoc, mul_comm (2 : ℝ), inv_mul_cancel, mul_one, mul_one] 
  { -- rest of the proof would go here but skipped via sorry keyword as instructed
sorry }

end find_room_breadth_l90_90757


namespace average_of_k_from_polynomial_roots_l90_90592

theorem average_of_k_from_polynomial_roots :
  let k_values := {k | ∃ r1 r2 : ℕ, r1 * r2 = 18 ∧ r1 + r2 = k ∧ r1 ≠ r2 ∧ r1 > 0 ∧ r2 > 0} in
  (∑ k in k_values, k) / (k_values.card : ℕ) = 13 := by
  sorry

end average_of_k_from_polynomial_roots_l90_90592


namespace taxi_fare_l90_90768

-- Define the starting price
def starting_price : ℝ := 5

-- Define the price per kilometer after the initial 3 kilometers
def price_per_km : ℝ := 1.4

-- Define the total fare function for a given distance x where x > 3
def fare (x : ℝ) (h : x > 3) : ℝ :=
  starting_price + price_per_km * (x - 3)

-- Prove that the calculated fare is equivalent to 0.8 + 1.4 * x
theorem taxi_fare (x : ℝ) (h : x > 3) : fare x h = 0.8 + 1.4 * x :=
by
  unfold fare
  rw [starting_price, price_per_km]
  ring
  sorry

end taxi_fare_l90_90768


namespace problem_statement_l90_90817

noncomputable def p : ℤ := 
  (1 - 1993) * (1 - 1993^2) * (1 - 1993^3) * ... * (1 - 1993^1993) + 
  1993 * (1 - 1993^2) * (1 - 1993^3) * ... * (1 - 1993^1993) + 
  ... + 
  1993^1992 * (1 - 1993^1993) + 
  1993^1993

theorem problem_statement : 
  p = 1 :=
sorry

end problem_statement_l90_90817


namespace perfect_square_divisors_product_factorials_eq_672_l90_90616

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def product_factorials : ℕ :=
  (List.range 9).map (λ n, factorial (n + 1)).prod

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

noncomputable def count_perfect_square_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).countp (λ d, d ∣ n ∧ is_perfect_square d)

theorem perfect_square_divisors_product_factorials_eq_672 :
  count_perfect_square_divisors product_factorials = 672 := by
  sorry

end perfect_square_divisors_product_factorials_eq_672_l90_90616


namespace find_y_l90_90767

variable (x y z : ℝ)

theorem find_y
    (h₀ : x + y + z = 150)
    (h₁ : x + 10 = y - 10)
    (h₂ : y - 10 = 3 * z) :
    y = 74.29 :=
by
    sorry

end find_y_l90_90767


namespace binom_10_3_l90_90342

open Nat

theorem binom_10_3 : Nat.choose 10 3 = 120 := by
  -- The actual proof would go here, demonstrating that Nat.choose 10 3 indeed equals 120
  sorry

end binom_10_3_l90_90342


namespace binom_10_3_l90_90443

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90443


namespace m_greater_than_p_l90_90688

theorem m_greater_than_p (p m n : ℕ) (prime_p : Prime p) (pos_m : 0 < m) (pos_n : 0 < n)
    (eq : p^2 + m^2 = n^2) : m > p := 
by 
  sorry

end m_greater_than_p_l90_90688


namespace perimeter_bound_l90_90783

noncomputable def radius (D : Type) := 
  ∃ r : ℝ, r ≤ 1

noncomputable def centers (D : Type) := 
  ∃ l : ℝ, l ≥ 0

theorem perimeter_bound (D : Type) (hC : centers D) (hR : radius D) : 
  ∃ l : ℝ, ∀ discs : set D, ∃ P : ℝ, P ≤ 4 * l + 8 := 
sorry

end perimeter_bound_l90_90783


namespace fixed_point_exists_trajectory_M_trajectory_equation_l90_90597

variable (m : ℝ)
def line_l (x y : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0
def point_P (x y : ℝ) : Prop := x = -1 ∧ y = 0

theorem fixed_point_exists :
  ∃ x y : ℝ, (line_l m x y ∧ x = 1 ∧ y = -2) :=
by
  sorry

theorem trajectory_M :
  ∃ (M: ℝ × ℝ), (line_l m M.1 M.2 ∧ M = (0, -1)) :=
by
  sorry

theorem trajectory_equation (x y : ℝ) :
  ∃ (x y : ℝ), (x + 1) ^ 2  + y ^ 2 = 2 :=
by
  sorry

end fixed_point_exists_trajectory_M_trajectory_equation_l90_90597


namespace false_proposition_l90_90934

def predicate_p (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - m * x - 1 = 0

def condition_p : Prop :=
  ∀ m : ℝ, predicate_p m

def predicate_q : Prop :=
  ∃ x0 : ℕ, x0^2 - 2 * x0 - 1 ≤ 0

theorem false_proposition : ¬ (condition_p ∧ ¬ predicate_q) :=
by
  sorry

end false_proposition_l90_90934


namespace domain_of_f_solution_set_f_gt_g_l90_90020

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x) / Real.log 2
noncomputable def g (x : ℝ) : ℝ := Real.log (2*x - 2) / Real.log 2

theorem domain_of_f : {x : ℝ | x < 0 ∨ 1 < x} = {x : ℝ | x^2 - x > 0} := 
begin
  ext,
  simp only [Set.mem_set_of_eq],
  have H : x^2 - x = x * (x - 1), 
  { ring },
  rw H,
  apply lt_or_gt_of_ne (ne_of_lt (mul_pos_iff.mpr ⟨λ h, h.1, λ h, h.2⟩)),
end

theorem solution_set_f_gt_g : {x : ℝ | 2 < x} = {x : ℝ | (x < 0 ∨ 1 < x) ∧ x^2 - 3*x + 2 > 0} :=
begin
  ext,
  simp only [Set.mem_set_of_eq],
  split,
  { rintros (hx : 2 < x),
    split,
    { right,
      linarith },
    { calc
        x^2 - 3*x + 2
          = (x - 1) * (x - 2) : by ring
      ... > 0 : by { apply mul_pos; linarith } } },
  { rintros ⟨hx1, hx2⟩,
    cases hx1;
    linarith }
end

end domain_of_f_solution_set_f_gt_g_l90_90020


namespace compute_product_l90_90788

-- Define the conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x^3 - y^3 = 35)

-- Define the theorem to be proved
theorem compute_product (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := 
sorry

end compute_product_l90_90788


namespace minimum_votes_for_tall_l90_90083

theorem minimum_votes_for_tall (voters : ℕ) (districts : ℕ) (precincts : ℕ) (precinct_voters : ℕ)
  (vote_majority_per_precinct : ℕ → ℕ) (precinct_majority_per_district : ℕ → ℕ) (district_majority_to_win : ℕ) :
  voters = 135 ∧ districts = 5 ∧ precincts = 9 ∧ precinct_voters = 3 ∧
  (∀ p, vote_majority_per_precinct p = 2) ∧
  (∀ d, precinct_majority_per_district d = 5) ∧
  district_majority_to_win = 3 ∧ 
  tall_won : 
  ∃ min_votes, min_votes = 30 :=
by
  sorry

end minimum_votes_for_tall_l90_90083


namespace range_of_a_l90_90623

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo (π / 3) (π / 2), (cos (2 * x) + 2 * a * sin x + 3) ≤ (cos (2 * (x + 0.001)) + 2 * a * sin (x + 0.001) + 3))
  → a ≤ sqrt 3 :=
by
  intros h
  sorry

end range_of_a_l90_90623


namespace equation_has_at_most_one_real_root_l90_90952

def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x

theorem equation_has_at_most_one_real_root (f : ℝ → ℝ) (a : ℝ) (h : has_inverse f) :
  ∀ x1 x2 : ℝ, f x1 = a ∧ f x2 = a → x1 = x2 :=
by sorry

end equation_has_at_most_one_real_root_l90_90952


namespace mixed_number_sum_l90_90307

theorem mixed_number_sum :
  481 + 1/6  + 265 + 1/12 + 904 + 1/20 -
  (184 + 29/30) - (160 + 41/42) - (703 + 55/56) =
  603 + 3/8 :=
by
  sorry

end mixed_number_sum_l90_90307


namespace unique_two_scoop_sundaes_l90_90863

theorem unique_two_scoop_sundaes (n : ℕ) (h_n : n = 8) : 
  nat.choose n 2 = 28 :=
by
  rw h_n
  exact (nat.choose_eq_card (finset.card_eq.mpr rfl)).mpr rfl
  sorry

end unique_two_scoop_sundaes_l90_90863


namespace casey_saves_by_paying_monthly_l90_90889

theorem casey_saves_by_paying_monthly :
  let weekly_rate := 280
  let monthly_rate := 1000
  let weeks_in_a_month := 4
  let number_of_months := 3
  let total_weeks := number_of_months * weeks_in_a_month
  let total_cost_weekly := total_weeks * weekly_rate
  let total_cost_monthly := number_of_months * monthly_rate
  let savings := total_cost_weekly - total_cost_monthly
  savings = 360 :=
by
  sorry

end casey_saves_by_paying_monthly_l90_90889


namespace B_property_false_for_candidate_B_l90_90612

open Set

-- Define the set A
def A : Set ℝ := { x | x ≥ 1 }

-- Define a property for the set B having no intersection with A
def B_property (B : Set ℝ) : Prop := A ∩ B = ∅

-- Define the candidate set B
def candidate_B : Set ℝ := { x | x ≥ -1 }

-- The theorem stating that candidate_B cannot satisfy the property B_property
theorem B_property_false_for_candidate_B : ¬ B_property candidate_B :=
by sorry

end B_property_false_for_candidate_B_l90_90612


namespace average_value_of_roots_l90_90299

-- Placeholder for the existence and solutions of the quadratic equation
noncomputable def exists_quad_solutions (a b c : ℝ) : ℝ × ℝ := sorry

theorem average_value_of_roots : 
  (∃ x : ℝ, Real.sqrt (3 * x^2 + 4 * x + 1) = Real.sqrt 28) → 
  let solutions := exists_quad_solutions 3 4 (-27) in
  (1 / 2 * (solutions.1 + solutions.2) = -2 / 3) :=
by
  sorry

end average_value_of_roots_l90_90299


namespace friedas_edge_probability_l90_90993

def position := ℤ × ℤ

def grid_size := 4

def wrap_around (pos : ℤ × ℤ) : position :=
  ((pos.fst % grid_size + grid_size) % grid_size, (pos.snd % grid_size + grid_size) % grid_size)

def transitions (pos : position) : list position :=
  [wrap_around (pos.fst + 1, pos.snd),
   wrap_around (pos.fst - 1, pos.snd),
   wrap_around (pos.fst, pos.snd + 1),
   wrap_around (pos.fst, pos.snd - 1)]

def is_edge (pos : position) : Prop :=
  pos.fst = 0 ∨ pos.fst = 3 ∨ pos.snd = 0 ∨ pos.snd = 3

def initial_position : position := (2, 2)

def probability_after_three_hops : ℚ :=
  sorry -- Detailed calculation of probability

theorem friedas_edge_probability :
  probability_after_three_hops = 37 / 64 :=
  sorry

end friedas_edge_probability_l90_90993


namespace largest_prime_2023_digits_l90_90682

theorem largest_prime_2023_digits:
  ∃ k : ℕ, k = 1 ∧ ∀ p : ℕ, Prime p ∧ digit_count p = 2023 → (p^2 - k) % 15 = 0 :=
sorry

end largest_prime_2023_digits_l90_90682


namespace graph_is_regular_l90_90669

variables {V : Type*} [fintype V] (G : simple_graph V)
variables (n : ℕ) (e : ℕ)

-- Conditions
def is_simple_graph_with_conditions (G : simple_graph V) (n e : ℕ) : Prop :=
  @fintype.card V _ = n ∧
  G.edge_finset.card = e ∧
  (∑ v in (@fintype.elems V _), G.degree v * (G.degree v - 1) / 2) = 900

theorem graph_is_regular
  (h : is_simple_graph_with_conditions G 20 100) :
  G.is_regular :=
sorry

end graph_is_regular_l90_90669


namespace gradient_of_u_l90_90537

variable {α : Type*}
variables (a b : EuclideanSpace ℝ (Fin 3)) (x y z : ℝ)

def r : EuclideanSpace ℝ (Fin 3) := ![x, y, z]

noncomputable def u : ℝ :=
  ![a 0, a 1, a 2] ** ![b 0, b 1, b 2] ** r

theorem gradient_of_u : ∇ u = a × b :=
sorry

end gradient_of_u_l90_90537


namespace product_ABRML_eq_100_l90_90114

variables (A B R M L : ℝ)
variables (h1 : log 10 (A * B) + log 10 (A * M) = 2)
variables (h2 : log 10 (M * L) + log 10 (M * R) = 3)
variables (h3 : log 10 (R * A) + log 10 (R * B) = 5)

theorem product_ABRML_eq_100 : A * B * R * M * L = 100 :=
by
  sorry

end product_ABRML_eq_100_l90_90114


namespace alloy_ratio_proof_l90_90240

def ratio_lead_to_tin_in_alloy_a (x y : ℝ) (ha : 0 < x) (hb : 0 < y) : Prop :=
  let weight_tin_in_a := (y / (x + y)) * 170
  let weight_tin_in_b := (3 / 8) * 250
  let total_tin := weight_tin_in_a + weight_tin_in_b
  total_tin = 221.25

theorem alloy_ratio_proof (x y : ℝ) (ha : 0 < x) (hb : 0 < y) (hc : ratio_lead_to_tin_in_alloy_a x y ha hb) : y / x = 3 :=
by
  -- Proof is omitted
  sorry

end alloy_ratio_proof_l90_90240


namespace binom_10_3_l90_90442

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3 : combination 10 3 = 120 := by
  sorry

end binom_10_3_l90_90442


namespace faucet_drip_rate_l90_90751

variable (volume_per_drop total_volume_per_hour : ℝ)
variable (minutes_per_hour : ℝ)

theorem faucet_drip_rate (h1 : volume_per_drop = 0.05) (h2 : total_volume_per_hour = 30) (h3 : minutes_per_hour = 60) :
  let drops_per_hour := total_volume_per_hour / volume_per_drop in
  let drips_per_minute := drops_per_hour / minutes_per_hour in
  drips_per_minute = 10 :=
by
  sorry

end faucet_drip_rate_l90_90751


namespace value_of_a_plus_b_l90_90600

noncomputable def f (a b x : ℝ) := x / (a * x + b)

theorem value_of_a_plus_b (a b : ℝ) (h₁: a ≠ 0) (h₂: f a b (-4) = 4)
    (h₃: ∀ x, f a b (f a b x) = x) : a + b = 3 / 2 :=
sorry

end value_of_a_plus_b_l90_90600


namespace inequality_1_inequality_2_inequality_3_inequality_4_l90_90652

-- Definitions of distances
def d_a : ℝ := sorry
def d_b : ℝ := sorry
def d_c : ℝ := sorry
def R_a : ℝ := sorry
def R_b : ℝ := sorry
def R_c : ℝ := sorry
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

def R : ℝ := sorry -- Circumradius
def r : ℝ := sorry -- Inradius

-- Inequality 1
theorem inequality_1 : a * R_a ≥ c * d_c + b * d_b := 
  sorry

-- Inequality 2
theorem inequality_2 : d_a * R_a + d_b * R_b + d_c * R_c ≥ 2 * (d_a * d_b + d_b * d_c + d_c * d_a) :=
  sorry

-- Inequality 3
theorem inequality_3 : R_a + R_b + R_c ≥ 2 * (d_a + d_b + d_c) :=
  sorry

-- Inequality 4
theorem inequality_4 : R_a * R_b * R_c ≥ (R / (2 * r)) * (d_a + d_b) * (d_b + d_c) * (d_c + d_a) :=
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l90_90652


namespace electrician_salary_ratio_l90_90103

theorem electrician_salary_ratio
  (w : ℕ) (e p : ℕ) 
  (h1 : 2 * w + p + e = 650)
  (h2 : w = 100)
  (h3 : p = 2.5 * w) :
  e / w = 2 :=
by
  -- You should write the proof here
  sorry

end electrician_salary_ratio_l90_90103
