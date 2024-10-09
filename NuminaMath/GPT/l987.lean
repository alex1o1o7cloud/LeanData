import Mathlib

namespace remaining_length_after_cut_l987_98758

/- Definitions -/
def original_length (a b : ℕ) : ℕ := 5 * a + 4 * b
def rectangle_perimeter (a b : ℕ) : ℕ := 2 * (a + b)
def remaining_length (a b : ℕ) : ℕ := original_length a b - rectangle_perimeter a b

/- Theorem statement -/
theorem remaining_length_after_cut (a b : ℕ) : remaining_length a b = 3 * a + 2 * b := 
by 
  sorry

end remaining_length_after_cut_l987_98758


namespace cost_per_sqft_is_3_l987_98715

def deck_length : ℝ := 30
def deck_width : ℝ := 40
def extra_cost_per_sqft : ℝ := 1
def total_cost : ℝ := 4800

theorem cost_per_sqft_is_3
    (area : ℝ := deck_length * deck_width)
    (sealant_cost : ℝ := area * extra_cost_per_sqft)
    (deck_construction_cost : ℝ := total_cost - sealant_cost) :
    deck_construction_cost / area = 3 :=
by
  sorry

end cost_per_sqft_is_3_l987_98715


namespace sum_of_coordinates_of_other_endpoint_l987_98779

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (h1 : (1 + x) / 2 = 5)
  (h2 : (2 + y) / 2 = 6) :
  x + y = 19 :=
by
  sorry

end sum_of_coordinates_of_other_endpoint_l987_98779


namespace dawn_monthly_payments_l987_98794

theorem dawn_monthly_payments (annual_salary : ℕ) (saved_per_month : ℕ)
  (h₁ : annual_salary = 48000)
  (h₂ : saved_per_month = 400)
  (h₃ : ∀ (monthly_salary : ℕ), saved_per_month = (10 * monthly_salary) / 100):
  annual_salary / saved_per_month = 12 :=
by
  sorry

end dawn_monthly_payments_l987_98794


namespace sum_of_x_and_y_l987_98776

theorem sum_of_x_and_y (x y : ℝ) 
  (h₁ : |x| + x + 5 * y = 2)
  (h₂ : |y| - y + x = 7) : 
  x + y = 3 := 
sorry

end sum_of_x_and_y_l987_98776


namespace center_of_circle_polar_eq_l987_98775

theorem center_of_circle_polar_eq (ρ θ : ℝ) : 
    (∀ ρ θ, ρ = 2 * Real.cos θ ↔ (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1) → 
    ∃ x y : ℝ, x = 1 ∧ y = 0 :=
by
  sorry

end center_of_circle_polar_eq_l987_98775


namespace possible_values_of_a_l987_98721

theorem possible_values_of_a :
  ∃ (a : ℤ), (∀ (b c : ℤ), (x : ℤ) → (x - a) * (x - 8) + 4 = (x + b) * (x + c)) → (a = 6 ∨ a = 10) :=
sorry

end possible_values_of_a_l987_98721


namespace solve_D_l987_98736

-- Define the digits represented by each letter
variable (P M T D E : ℕ)

-- Each letter represents a different digit (0-9) and should be distinct
axiom distinct_digits : (P ≠ M) ∧ (P ≠ T) ∧ (P ≠ D) ∧ (P ≠ E) ∧ 
                        (M ≠ T) ∧ (M ≠ D) ∧ (M ≠ E) ∧ 
                        (T ≠ D) ∧ (T ≠ E) ∧ 
                        (D ≠ E)

-- Each letter is a digit from 0 to 9
axiom digit_range : 0 ≤ P ∧ P ≤ 9 ∧ 0 ≤ M ∧ M ≤ 9 ∧ 
                    0 ≤ T ∧ T ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 
                    0 ≤ E ∧ E ≤ 9

-- Each column sums to the digit below it, considering carry overs from right to left
axiom column1 : T + T + E = E ∨ T + T + E = 10 + E
axiom column2 : E + D + T + (if T + T + E = 10 + E then 1 else 0) = P
axiom column3 : P + M + (if E + D + T + (if T + T + E = 10 + E then 1 else 0) = 10 + P then 1 else 0) = M

-- Prove that D = 4 given the above conditions
theorem solve_D : D = 4 :=
by sorry

end solve_D_l987_98736


namespace yvette_final_bill_l987_98730

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end yvette_final_bill_l987_98730


namespace cylinder_surface_area_is_128pi_l987_98791

noncomputable def cylinder_total_surface_area (h r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_surface_area_is_128pi :
  cylinder_total_surface_area 12 4 = 128 * Real.pi :=
by
  sorry

end cylinder_surface_area_is_128pi_l987_98791


namespace smallest_number_of_eggs_l987_98743

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 100) : 102 ≤ 15 * c - 3 :=
by
  sorry

end smallest_number_of_eggs_l987_98743


namespace locus_of_centers_l987_98744

set_option pp.notation false -- To ensure nicer looking lean code.

-- Define conditions for circles C_3 and C_4
def C3 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C4 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Statement to prove the locus of centers satisfies the equation
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 1)^2) ∧ ((a - 3)^2 + b^2 = (9 - r)^2)) →
  (a^2 + 18 * b^2 - 6 * a - 440 = 0) :=
by
  sorry -- Proof not required as per the instructions

end locus_of_centers_l987_98744


namespace expression_multiple_l987_98789

theorem expression_multiple :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  (a - b) / (1/78) = 13 :=
by
  sorry

end expression_multiple_l987_98789


namespace lower_base_length_l987_98735

variable (A B C D E : Type)
variable (AD BD BE DE : ℝ)

-- Conditions of the problem
axiom hAD : AD = 12  -- upper base
axiom hBD : BD = 18  -- height
axiom hBE_DE : BE = 2 * DE  -- ratio BE = 2 * DE

-- Define the trapezoid with given lengths and conditions
def trapezoid_exists (A B C D : Type) (AD BD BE DE : ℝ) :=
  AD = 12 ∧ BD = 18 ∧ BE = 2 * DE

-- The length of BC to be proven
def BC : ℝ := 24

-- The theorem to be proven
theorem lower_base_length (h : trapezoid_exists A B C D AD BD BE DE) : BC = 2 * AD :=
by
  sorry

end lower_base_length_l987_98735


namespace fish_in_third_tank_l987_98782

-- Definitions of the conditions
def first_tank_goldfish : ℕ := 7
def first_tank_beta_fish : ℕ := 8
def first_tank_fish : ℕ := first_tank_goldfish + first_tank_beta_fish

def second_tank_fish : ℕ := 2 * first_tank_fish

def third_tank_fish : ℕ := second_tank_fish / 3

-- The statement to prove
theorem fish_in_third_tank : third_tank_fish = 10 := by
  sorry

end fish_in_third_tank_l987_98782


namespace shorter_piece_is_20_l987_98728

def shorter_piece_length (total_length : ℕ) (ratio : ℚ) (shorter_piece : ℕ) : Prop :=
    shorter_piece * 7 = 2 * (total_length - shorter_piece)

theorem shorter_piece_is_20 : ∀ (total_length : ℕ) (shorter_piece : ℕ), 
    total_length = 90 ∧
    shorter_piece_length total_length (2/7 : ℚ) shorter_piece ->
    shorter_piece = 20 :=
by
  intro total_length shorter_piece
  intro h
  have h_total_length : total_length = 90 := h.1
  have h_equation : shorter_piece_length total_length (2/7 : ℚ) shorter_piece := h.2
  sorry

end shorter_piece_is_20_l987_98728


namespace number_of_B_is_14_l987_98772

-- Define the problem conditions
variable (num_students : ℕ)
variable (num_A num_B num_C num_D : ℕ)
variable (h1 : num_A = 8 * num_B / 10)
variable (h2 : num_C = 13 * num_B / 10)
variable (h3 : num_D = 5 * num_B / 10)
variable (h4 : num_students = 50)
variable (h5 : num_A + num_B + num_C + num_D = num_students)

-- Formalize the statement to be proved
theorem number_of_B_is_14 :
  num_B = 14 := by
  sorry

end number_of_B_is_14_l987_98772


namespace emily_eggs_collected_l987_98749

theorem emily_eggs_collected :
  let number_of_baskets := 1525
  let eggs_per_basket := 37.5
  let total_eggs := number_of_baskets * eggs_per_basket
  total_eggs = 57187.5 :=
by
  sorry

end emily_eggs_collected_l987_98749


namespace intersection_of_A_and_B_l987_98756

def A : Set ℝ := { x | x ≥ 0 }
def B : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 ≤ x ∧ x < 2 } := 
by
  sorry

end intersection_of_A_and_B_l987_98756


namespace net_income_on_15th_day_l987_98793

noncomputable def net_income_15th_day : ℝ :=
  let earnings_15th_day := 3 * (3 ^ 14)
  let tax := 0.10 * earnings_15th_day
  let earnings_after_tax := earnings_15th_day - tax
  earnings_after_tax - 100

theorem net_income_on_15th_day :
  net_income_15th_day = 12913916.3 := by
  sorry

end net_income_on_15th_day_l987_98793


namespace division_of_powers_of_ten_l987_98704

theorem division_of_powers_of_ten :
  (10 ^ 0.7 * 10 ^ 0.4) / (10 ^ 0.2 * 10 ^ 0.6 * 10 ^ 0.3) = 1 := by
  sorry

end division_of_powers_of_ten_l987_98704


namespace Kevin_crates_per_week_l987_98770

theorem Kevin_crates_per_week (a b c : ℕ) (h₁ : a = 13) (h₂ : b = 20) (h₃ : c = 17) :
  a + b + c = 50 :=
by 
  sorry

end Kevin_crates_per_week_l987_98770


namespace sum_of_roots_eq_36_l987_98722

theorem sum_of_roots_eq_36 :
  (∃ x1 x2 x3 : ℝ, (11 - x1) ^ 3 + (13 - x2) ^ 3 = (24 - 2 * x3) ^ 3 ∧ 
  (11 - x2) ^ 3 + (13 - x3) ^ 3 = (24 - 2 * x1) ^ 3 ∧ 
  (11 - x3) ^ 3 + (13 - x1) ^ 3 = (24 - 2 * x2) ^ 3 ∧
  x1 + x2 + x3 = 36) :=
sorry

end sum_of_roots_eq_36_l987_98722


namespace check_point_on_curve_l987_98762

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x * y + 2 * y + 1 = 0

theorem check_point_on_curve :
  point_on_curve 0 (-1/2) :=
by
  sorry

end check_point_on_curve_l987_98762


namespace range_of_m_l987_98709

theorem range_of_m (m : ℝ) 
  (p : m < 0) 
  (q : ∀ x : ℝ, x^2 + m * x + 1 > 0) : 
  -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l987_98709


namespace perfect_power_transfer_l987_98796

-- Given Conditions
variables {x y z : ℕ}

-- Definition of what it means to be a perfect seventh power
def is_perfect_seventh_power (n : ℕ) :=
  ∃ k : ℕ, n = k^7

-- The proof problem
theorem perfect_power_transfer 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : is_perfect_seventh_power (x^3 * y^5 * z^6)) :
  is_perfect_seventh_power (x^5 * y^6 * z^3) := by
  sorry

end perfect_power_transfer_l987_98796


namespace vector_projection_condition_l987_98761

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 3 + 2 * t)
noncomputable def line_m (s : ℝ) : ℝ × ℝ := (4 + 2 * s, 5 + 3 * s)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_projection_condition 
  (t s : ℝ)
  (C : ℝ × ℝ := line_l t)
  (D : ℝ × ℝ := line_m s)
  (Q : ℝ × ℝ)
  (hQ : is_perpendicular (Q.1 - C.1, Q.2 - C.2) (2, 3))
  (v1 v2 : ℝ)
  (hv_sum : v1 + v2 = 3)
  (hv_def : ∃ k : ℝ, v1 = 3 * k ∧ v2 = -2 * k)
  : (v1, v2) = (9, -6) := 
sorry

end vector_projection_condition_l987_98761


namespace expression_value_l987_98725

theorem expression_value (a b : ℤ) (h₁ : a = -5) (h₂ : b = 3) :
  -a - b^4 + a * b = -91 := by
  sorry

end expression_value_l987_98725


namespace factor_poly_l987_98766

theorem factor_poly : 
  ∃ (a b c d e f : ℤ), a < d ∧ (a*x^2 + b*x + c)*(d*x^2 + e*x + f) = (x^2 + 6*x + 9 - 64*x^4) ∧ 
  (a = -8 ∧ b = 1 ∧ c = 3 ∧ d = 8 ∧ e = 1 ∧ f = 3) := 
sorry

end factor_poly_l987_98766


namespace vector_addition_correct_l987_98712

def vec1 : ℤ × ℤ := (5, -9)
def vec2 : ℤ × ℤ := (-8, 14)
def vec_sum (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

theorem vector_addition_correct :
  vec_sum vec1 vec2 = (-3, 5) :=
by
  -- Proof omitted
  sorry

end vector_addition_correct_l987_98712


namespace no_integer_solutions_l987_98755

theorem no_integer_solutions (n : ℕ) (h : 2 ≤ n) :
  ¬ ∃ x y z : ℤ, x^2 + y^2 = z^n :=
sorry

end no_integer_solutions_l987_98755


namespace max_distance_l987_98795

-- Definition of curve C₁ in rectangular coordinates.
def C₁_rectangular (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- Definition of curve C₂ in its general form.
def C₂_general (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Coordinates of point M, the intersection of C₂ with x-axis.
def M : ℝ × ℝ := (2, 0)

-- Condition that N is a moving point on curve C₁.
def N (x y : ℝ) : Prop := C₁_rectangular x y

-- Maximum distance |MN|.
theorem max_distance (x y : ℝ) (hN : N x y) : 
  dist (2, 0) (x, y) ≤ Real.sqrt 5 + 1 := by
  sorry

end max_distance_l987_98795


namespace minnie_takes_longer_l987_98763

def minnie_speed_flat := 25 -- kph
def minnie_speed_downhill := 35 -- kph
def minnie_speed_uphill := 10 -- kph

def penny_speed_flat := 35 -- kph
def penny_speed_downhill := 45 -- kph
def penny_speed_uphill := 15 -- kph

def distance_flat := 25 -- km
def distance_downhill := 20 -- km
def distance_uphill := 15 -- km

noncomputable def minnie_time := 
  (distance_uphill / minnie_speed_uphill) + 
  (distance_downhill / minnie_speed_downhill) + 
  (distance_flat / minnie_speed_flat) -- hours

noncomputable def penny_time := 
  (distance_uphill / penny_speed_uphill) + 
  (distance_downhill / penny_speed_downhill) + 
  (distance_flat / penny_speed_flat) -- hours

noncomputable def minnie_time_minutes := minnie_time * 60 -- minutes
noncomputable def penny_time_minutes := penny_time * 60 -- minutes

noncomputable def time_difference := minnie_time_minutes - penny_time_minutes -- minutes

theorem minnie_takes_longer : time_difference = 130 :=
  sorry

end minnie_takes_longer_l987_98763


namespace find_n_l987_98703

theorem find_n (n : ℕ) : (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n + 1) / (n + 1 : ℝ) = 2) → (n = 2) :=
by
  sorry

end find_n_l987_98703


namespace solid_brick_height_l987_98784

theorem solid_brick_height (n c base_perimeter height : ℕ) 
  (h1 : n = 42) 
  (h2 : c = 1) 
  (h3 : base_perimeter = 18)
  (h4 : n % base_area = 0)
  (h5 : 2 * (length + width) = base_perimeter)
  (h6 : base_area * height = n) : 
  height = 3 :=
by sorry

end solid_brick_height_l987_98784


namespace find_a_minus_b_l987_98786

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 3 * a * x + 4

-- Define the condition for the function being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the function f(x) with given parameters
theorem find_a_minus_b (a b : ℝ) (h_dom_range : ∀ x : ℝ, b - 3 ≤ x → x ≤ 2 * b) (h_even_f : is_even (f a)) :
  a - b = -1 :=
  sorry

end find_a_minus_b_l987_98786


namespace derivative_at_3_l987_98799

noncomputable def f (x : ℝ) := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l987_98799


namespace investment_after_8_years_l987_98741

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

theorem investment_after_8_years :
  let P := 500
  let r := 0.03
  let n := 8
  let A := compound_interest P r n
  round A = 633 :=
by
  sorry

end investment_after_8_years_l987_98741


namespace votes_cast_proof_l987_98701

variable (V : ℝ)
variable (candidate_votes : ℝ)
variable (rival_votes : ℝ)

noncomputable def total_votes_cast : Prop :=
  candidate_votes = 0.40 * V ∧ 
  rival_votes = candidate_votes + 2000 ∧ 
  rival_votes = 0.60 * V ∧ 
  V = 10000

theorem votes_cast_proof : total_votes_cast V candidate_votes rival_votes :=
by {
  sorry
  }

end votes_cast_proof_l987_98701


namespace prob_two_fours_l987_98768

-- Define the sample space for a fair die
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The probability of rolling a 4 on a fair die
def prob_rolling_four : ℚ := 1 / 6

-- Probability of two independent events both resulting in rolling a 4
def prob_both_rolling_four : ℚ := (prob_rolling_four) * (prob_rolling_four)

-- Prove that the probability of rolling two 4s in two independent die rolls is 1/36
theorem prob_two_fours : prob_both_rolling_four = 1 / 36 := by
  sorry

end prob_two_fours_l987_98768


namespace set_M_roster_method_l987_98734

open Set

theorem set_M_roster_method :
  {a : ℤ | ∃ (n : ℕ), 6 = n * (5 - a)} = {-1, 2, 3, 4} := by
  sorry

end set_M_roster_method_l987_98734


namespace common_ratio_of_geometric_sequence_l987_98729

theorem common_ratio_of_geometric_sequence 
  (a : ℝ) (log2_3 log4_3 log8_3: ℝ)
  (h1: log4_3 = log2_3 / 2)
  (h2: log8_3 = log2_3 / 3) 
  (h_geometric: ∀ i j, 
    i = a + log2_3 → 
    j = a + log4_3 →
    j / i = a + log8_3 / j / i / j
  ) :
  (a + log4_3) / (a + log2_3) = 1/3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l987_98729


namespace robins_fraction_l987_98751

theorem robins_fraction (B R J : ℕ) (h1 : R + J = B)
  (h2 : 2/3 * (R : ℚ) + 1/3 * (J : ℚ) = 7/15 * (B : ℚ)) :
  (R : ℚ) / B = 2/5 :=
by
  sorry

end robins_fraction_l987_98751


namespace five_b_value_l987_98781

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 :=
by
  sorry

end five_b_value_l987_98781


namespace complement_of_A_with_respect_to_U_l987_98787

def U : Set ℤ := {1, 2, 3, 4, 5}
def A : Set ℤ := {x | abs (x - 3) < 2}
def C_UA : Set ℤ := { x | x ∈ U ∧ x ∉ A }

theorem complement_of_A_with_respect_to_U :
  C_UA = {1, 5} :=
by
  sorry

end complement_of_A_with_respect_to_U_l987_98787


namespace area_of_small_parallelograms_l987_98777

theorem area_of_small_parallelograms (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  (1 : ℝ) / (m * n : ℝ) = 1 / (m * n) :=
by
  sorry

end area_of_small_parallelograms_l987_98777


namespace extreme_points_f_l987_98773

theorem extreme_points_f (a b : ℝ)
  (h1 : 3 * (-2)^2 + 2 * a * (-2) + b = 0)
  (h2 : 3 * 4^2 + 2 * a * 4 + b = 0) :
  a - b = 21 :=
sorry

end extreme_points_f_l987_98773


namespace mean_combined_scores_l987_98748

theorem mean_combined_scores (M A : ℝ) (m a : ℕ) 
  (hM : M = 88) 
  (hA : A = 72) 
  (hm : (m:ℝ) / (a:ℝ) = 2 / 3) :
  (88 * m + 72 * a) / (m + a) = 78 :=
by
  sorry

end mean_combined_scores_l987_98748


namespace cone_volume_l987_98742

-- Define the condition
def cylinder_volume : ℝ := 30

-- Define the statement that needs to be proven
theorem cone_volume (h_cylinder_volume : cylinder_volume = 30) : cylinder_volume / 3 = 10 := 
by 
  -- Proof omitted
  sorry

end cone_volume_l987_98742


namespace maxine_purchases_l987_98747

theorem maxine_purchases (x y z : ℕ) (h1 : x + y + z = 40) (h2 : 50 * x + 400 * y + 500 * z = 10000) : x = 40 :=
by
  sorry

end maxine_purchases_l987_98747


namespace range_of_quadratic_function_l987_98752

theorem range_of_quadratic_function : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 0 ≤ x^2 - 4 * x + 3 ∧ x^2 - 4 * x + 3 ≤ 8 :=
by
  intro x hx
  sorry

end range_of_quadratic_function_l987_98752


namespace base_8_to_base_4_l987_98767

theorem base_8_to_base_4 (n : ℕ) (h : n = 6 * 8^2 + 5 * 8^1 + 3 * 8^0) : 
  (n : ℕ) = 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 2 * 4^1 + 3 * 4^0 :=
by
  -- Conversion proof goes here
  sorry

end base_8_to_base_4_l987_98767


namespace tan_of_angle_subtraction_l987_98713

theorem tan_of_angle_subtraction (a : ℝ) (h : Real.tan (a + Real.pi / 4) = 1 / 7) : Real.tan a = -3 / 4 :=
by
  sorry

end tan_of_angle_subtraction_l987_98713


namespace profit_percentage_l987_98707

theorem profit_percentage (C S : ℝ) (hC : C = 60) (hS : S = 75) : ((S - C) / C) * 100 = 25 :=
by
  sorry

end profit_percentage_l987_98707


namespace smallest_c_for_polynomial_l987_98731

theorem smallest_c_for_polynomial :
  ∃ r1 r2 r3 : ℕ, (r1 * r2 * r3 = 2310) ∧ (r1 + r2 + r3 = 52) := sorry

end smallest_c_for_polynomial_l987_98731


namespace smallest_n_digit_sum_l987_98717

theorem smallest_n_digit_sum :
  ∃ n : ℕ, (∃ (arrangements : ℕ), arrangements > 1000000 ∧ arrangements = (1/2 * ((n + 1) * (n + 2)))) ∧ (1 + n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + n % 10 = 9) :=
sorry

end smallest_n_digit_sum_l987_98717


namespace extra_kilometers_per_hour_l987_98732

theorem extra_kilometers_per_hour (S a : ℝ) (h : a > 2) : 
  (S / (a - 2)) - (S / a) = (S / (a - 2)) - (S / a) :=
by sorry

end extra_kilometers_per_hour_l987_98732


namespace floor_equation_solution_l987_98733

theorem floor_equation_solution (a b : ℝ) :
  (∀ x y : ℝ, ⌊a * x + b * y⌋ + ⌊b * x + a * y⌋ = (a + b) * ⌊x + y⌋) → (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) := by
  sorry

end floor_equation_solution_l987_98733


namespace find_f_value_l987_98700

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 5 * Real.pi / 6)

theorem find_f_value :
  f ( -5 * Real.pi / 12 ) = Real.sqrt 3 / 2 :=
by
  sorry

end find_f_value_l987_98700


namespace molecular_weight_CuCO3_8_moles_l987_98723

-- Definitions for atomic weights
def atomic_weight_Cu : ℝ := 63.55
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Definition for the molecular formula of CuCO3
def molecular_weight_CuCO3 :=
  atomic_weight_Cu + atomic_weight_C + 3 * atomic_weight_O

-- Number of moles
def moles : ℝ := 8

-- Total weight of 8 moles of CuCO3
def total_weight := moles * molecular_weight_CuCO3

-- Proof statement
theorem molecular_weight_CuCO3_8_moles :
  total_weight = 988.48 :=
  by
  sorry

end molecular_weight_CuCO3_8_moles_l987_98723


namespace moores_law_2000_l987_98785

noncomputable def number_of_transistors (year : ℕ) : ℕ :=
  if year = 1990 then 1000000
  else 1000000 * 2 ^ ((year - 1990) / 2)

theorem moores_law_2000 :
  number_of_transistors 2000 = 32000000 :=
by
  unfold number_of_transistors
  rfl

end moores_law_2000_l987_98785


namespace correct_statement_l987_98711

-- We assume the existence of lines and planes with certain properties.
variables {Line : Type} {Plane : Type}
variables {m n : Line} {alpha beta gamma : Plane}

-- Definitions for perpendicular and parallel relations
def perpendicular (p1 p2 : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry

-- The theorem we aim to prove given the conditions
theorem correct_statement :
  line_perpendicular_to_plane m beta ∧ line_parallel_to_plane m alpha → perpendicular alpha beta :=
by sorry

end correct_statement_l987_98711


namespace train_crosses_platform_in_20s_l987_98702

noncomputable def timeToCrossPlatform (train_length : ℝ) (platform_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

theorem train_crosses_platform_in_20s :
  timeToCrossPlatform 120 213.36 60 = 20 :=
by
  sorry

end train_crosses_platform_in_20s_l987_98702


namespace road_repair_completion_time_l987_98718

theorem road_repair_completion_time (L R r : ℕ) (hL : L = 100) (hR : R = 64) (hr : r = 9) :
  (L - R) / r = 5 :=
by
  sorry

end road_repair_completion_time_l987_98718


namespace part1_part2_l987_98790

theorem part1 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ a b : students, a ≠ b ∧
  (∀ c : students, c ≠ a → d a c > d a b) ∧ 
  (∀ c : students, c ≠ b → d b c > d b a) :=
sorry

theorem part2 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ c : students, ∀ a : students, ¬ (∀ b : students, b ≠ a → d b a < d b c ∧ d a c < d a b) :=
sorry

end part1_part2_l987_98790


namespace last_score_is_71_l987_98739

theorem last_score_is_71 (scores : List ℕ) (h : scores = [71, 74, 79, 85, 88, 92]) (sum_eq: scores.sum = 489) :
  ∃ s : ℕ, s ∈ scores ∧ 
           (∃ avg : ℕ, avg = (scores.sum - s) / 5 ∧ 
           ∀ lst : List ℕ, lst = scores.erase s → (∀ n, n ∈ lst → lst.sum % (lst.length - 1) = 0)) :=
  sorry

end last_score_is_71_l987_98739


namespace even_function_analytic_expression_l987_98737

noncomputable def f (x : ℝ) : ℝ := 
if x ≥ 0 then Real.log (x^2 - 2 * x + 2) 
else Real.log (x^2 + 2 * x + 2)

theorem even_function_analytic_expression (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_nonneg : ∀ x : ℝ, 0 ≤ x → f x = Real.log (x^2 - 2 * x + 2)) :
  ∀ x : ℝ, x < 0 → f x = Real.log (x^2 + 2 * x + 2) :=
by
  sorry

end even_function_analytic_expression_l987_98737


namespace monotonic_intervals_max_min_values_l987_98745

noncomputable def f : ℝ → ℝ := λ x => (1 / 3) * x^3 + x^2 - 3 * x + 1

theorem monotonic_intervals :
  (∀ x, x < -3 → deriv f x > 0) ∧
  (∀ x, x > 1 → deriv f x > 0) ∧
  (∀ x, -3 < x ∧ x < 1 → deriv f x < 0) :=
by
  sorry

theorem max_min_values :
  f 2 = 5 / 3 ∧ f 1 = -2 / 3 :=
by
  sorry

end monotonic_intervals_max_min_values_l987_98745


namespace remaining_bread_after_three_days_l987_98706

namespace BreadProblem

def InitialBreadCount : ℕ := 200

def FirstDayConsumption (bread : ℕ) : ℕ := bread / 4
def SecondDayConsumption (remainingBreadAfterFirstDay : ℕ) : ℕ := 2 * remainingBreadAfterFirstDay / 5
def ThirdDayConsumption (remainingBreadAfterSecondDay : ℕ) : ℕ := remainingBreadAfterSecondDay / 2

theorem remaining_bread_after_three_days : 
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  breadAfterThirdDay = 45 := 
by
  let initialBread := InitialBreadCount 
  let breadAfterFirstDay := initialBread - FirstDayConsumption initialBread 
  let breadAfterSecondDay := breadAfterFirstDay - SecondDayConsumption breadAfterFirstDay 
  let breadAfterThirdDay := breadAfterSecondDay - ThirdDayConsumption breadAfterSecondDay 
  have : breadAfterThirdDay = 45 := sorry
  exact this

end BreadProblem

end remaining_bread_after_three_days_l987_98706


namespace friendly_point_pairs_l987_98738

def friendly_points (k : ℝ) (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A = (a, -1 / a) ∧ B = (-a, 1 / a) ∧
  B.2 = k * B.1 + 1 + k

theorem friendly_point_pairs : ∀ (k : ℝ), k ≥ 0 → 
  ∃ n, (n = 1 ∨ n = 2) ∧
  (∀ a : ℝ, a > 0 →
    friendly_points k a (a, -1 / a) (-a, 1 / a))
:= by
  sorry

end friendly_point_pairs_l987_98738


namespace find_y_interval_l987_98714

open Real

theorem find_y_interval {y : ℝ}
  (hy_nonzero : y ≠ 0)
  (h_denominator_nonzero : 1 + 3 * y - 4 * y^2 ≠ 0) :
  (y^2 + 9 * y - 1 = 0) →
  (∀ y, y ∈ Set.Icc (-(9 + sqrt 85)/2) (-(9 - sqrt 85)/2) \ {y | y = 0 ∨ 1 + 3 * y - 4 * y^2 = 0} ↔
  (y * (3 - 3 * y))/(1 + 3 * y - 4 * y^2) ≤ 1) :=
by
  sorry

end find_y_interval_l987_98714


namespace area_ratio_l987_98760

theorem area_ratio (A B C D E : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (AB BC AC AD AE : ℝ) (ADE_ratio : ℝ) :
  AB = 25 ∧ BC = 39 ∧ AC = 42 ∧ AD = 19 ∧ AE = 14 →
  ADE_ratio = 19 / 56 :=
by sorry

end area_ratio_l987_98760


namespace wendy_facial_products_l987_98797

def total_time (P : ℕ) : ℕ :=
  5 * (P - 1) + 30

theorem wendy_facial_products :
  (total_time 6 = 55) :=
by
  sorry

end wendy_facial_products_l987_98797


namespace compound_interest_for_2_years_l987_98774

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem compound_interest_for_2_years 
  (P : ℝ) (R : ℝ) (T : ℝ) (S : ℝ)
  (h1 : S = 600)
  (h2 : R = 5)
  (h3 : T = 2)
  (h4 : simple_interest P R T = S)
  : compound_interest P R T = 615 := 
sorry

end compound_interest_for_2_years_l987_98774


namespace suitable_sampling_method_l987_98710

-- Conditions given
def num_products : ℕ := 40
def num_top_quality : ℕ := 10
def num_second_quality : ℕ := 25
def num_defective : ℕ := 5
def draw_count : ℕ := 8

-- Possible sampling methods
inductive SamplingMethod
| DrawingLots : SamplingMethod
| RandomNumberTable : SamplingMethod
| Systematic : SamplingMethod
| Stratified : SamplingMethod

-- Problem statement (to be proved)
theorem suitable_sampling_method : 
  (num_products = 40) ∧ 
  (num_top_quality = 10) ∧ 
  (num_second_quality = 25) ∧ 
  (num_defective = 5) ∧ 
  (draw_count = 8) → 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
by sorry

end suitable_sampling_method_l987_98710


namespace solve_ordered_pair_l987_98764

theorem solve_ordered_pair : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^y + 3 = y^x ∧ 2 * x^y = y^x + 11 ∧ x = 14 ∧ y = 1 :=
by
  sorry

end solve_ordered_pair_l987_98764


namespace train2_length_is_230_l987_98769

noncomputable def train_length_proof : Prop :=
  let speed1_kmph := 120
  let speed2_kmph := 80
  let length_train1 := 270
  let time_cross := 9
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time_cross
  let length_train2 := total_distance - length_train1
  length_train2 = 230

theorem train2_length_is_230 : train_length_proof :=
  by
    sorry

end train2_length_is_230_l987_98769


namespace rectangular_prism_diagonals_l987_98798

theorem rectangular_prism_diagonals
  (num_vertices : ℕ) (num_edges : ℕ)
  (h1 : num_vertices = 12) (h2 : num_edges = 18) :
  (total_diagonals : ℕ) → total_diagonals = 20 :=
by
  sorry

end rectangular_prism_diagonals_l987_98798


namespace math_problem_l987_98759

variable (a : ℝ) (m n : ℝ)

theorem math_problem
  (h1 : a^m = 3)
  (h2 : a^n = 2) :
  a^(2*m + 3*n) = 72 := 
  sorry

end math_problem_l987_98759


namespace geometric_sum_s9_l987_98727

variable (S : ℕ → ℝ)

theorem geometric_sum_s9
  (h1 : S 3 = 7)
  (h2 : S 6 = 63) :
  S 9 = 511 :=
by
  sorry

end geometric_sum_s9_l987_98727


namespace range_of_f_l987_98754

noncomputable def f (x : ℝ) : ℝ :=
  if h : x < 1 then 3^(-x) else x^2

theorem range_of_f (x : ℝ) : (f x > 9) ↔ (x < -2 ∨ x > 3) :=
by
  sorry

end range_of_f_l987_98754


namespace seven_solutions_l987_98753

theorem seven_solutions: ∃ (pairs : List (ℕ × ℕ)), 
  (∀ (x y : ℕ), (x < y) → ((1 / (x : ℝ)) + (1 / (y : ℝ)) = 1 / 2007) ↔ (x, y) ∈ pairs) 
  ∧ pairs.length = 7 :=
sorry

end seven_solutions_l987_98753


namespace number_of_lattice_points_in_triangle_l987_98705

theorem number_of_lattice_points_in_triangle (L : ℕ) (hL : L > 1) :
  ∃ I, I = (L^2 - 1) / 2 :=
by
  sorry

end number_of_lattice_points_in_triangle_l987_98705


namespace find_point_A_coordinates_l987_98750

theorem find_point_A_coordinates (A B C : ℝ × ℝ)
  (hB : B = (1, 2)) (hC : C = (3, 4))
  (trans_left : ∃ l : ℝ, A = (B.1 + l, B.2))
  (trans_up : ∃ u : ℝ, A = (C.1, C.2 - u)) :
  A = (3, 2) := 
sorry

end find_point_A_coordinates_l987_98750


namespace probability_order_correct_l987_98788

inductive Phenomenon
| Certain
| VeryLikely
| Possible
| Impossible
| NotVeryLikely

open Phenomenon

def probability_order : Phenomenon → ℕ
| Certain       => 5
| VeryLikely    => 4
| Possible      => 3
| NotVeryLikely => 2
| Impossible    => 1

theorem probability_order_correct :
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] =
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] :=
by
  -- skips the proof
  sorry

end probability_order_correct_l987_98788


namespace correct_discount_rate_l987_98719

def purchase_price : ℝ := 200
def marked_price : ℝ := 300
def desired_profit_percentage : ℝ := 0.20

theorem correct_discount_rate :
  ∃ (x : ℝ), 300 * x = 240 ∧ x = 0.80 := 
by
  sorry

end correct_discount_rate_l987_98719


namespace log_one_eq_zero_l987_98708

theorem log_one_eq_zero : Real.log 1 = 0 := 
by
  sorry

end log_one_eq_zero_l987_98708


namespace rationalize_denominator_l987_98765

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l987_98765


namespace alexis_sew_skirt_time_l987_98716

theorem alexis_sew_skirt_time : 
  ∀ (S : ℝ), 
  (∀ (C : ℝ), C = 7) → 
  (6 * S + 4 * 7 = 40) → 
  S = 2 := 
by
  intros S _ h
  sorry

end alexis_sew_skirt_time_l987_98716


namespace max_cake_boxes_in_carton_l987_98746

-- Define the dimensions of the carton as constants
def carton_length := 25
def carton_width := 42
def carton_height := 60

-- Define the dimensions of the cake box as constants
def box_length := 8
def box_width := 7
def box_height := 5

-- Define the volume of the carton and the volume of the cake box
def volume_carton := carton_length * carton_width * carton_height
def volume_box := box_length * box_width * box_height

-- Define the theorem statement
theorem max_cake_boxes_in_carton : 
  (volume_carton / volume_box) = 225 :=
by
  -- The proof is omitted.
  sorry

end max_cake_boxes_in_carton_l987_98746


namespace range_of_a_l987_98720

def increasing {α : Type*} [Preorder α] (f : α → α) := ∀ x y, x ≤ y → f x ≤ f y

theorem range_of_a
  (f : ℝ → ℝ)
  (increasing_f : increasing f)
  (h_domain : ∀ x, 1 ≤ x ∧ x ≤ 5 → (f x = f x))
  (h_ineq : ∀ a, 1 ≤ a + 1 ∧ a + 1 ≤ 5 ∧ 1 ≤ 2 * a - 1 ∧ 2 * a - 1 ≤ 5 ∧ f (a + 1) < f (2 * a - 1)) :
  (2 : ℝ) < a ∧ a ≤ (3 : ℝ) := 
by
  sorry

end range_of_a_l987_98720


namespace correct_answers_proof_l987_98757

variable (n p q s c : ℕ)
variable (total_questions points_per_correct penalty_per_wrong total_score correct_answers : ℕ)

def num_questions := 20
def points_correct := 5
def penalty_wrong := 1
def total_points := 76

theorem correct_answers_proof :
  (total_questions * points_per_correct - (total_questions - correct_answers) * penalty_wrong) = total_points →
  correct_answers = 16 :=
by {
  sorry
}

end correct_answers_proof_l987_98757


namespace find_a_l987_98792

theorem find_a 
  (a : ℝ)
  (h : ∀ n : ℕ, (n.choose 2) * 2^(5-2) * a^2 = 80 → n = 5) :
  a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l987_98792


namespace product_of_000412_and_9243817_is_closest_to_3600_l987_98778

def product_closest_to (x y value: ℝ) : Prop := (abs (x * y - value) < min (abs (x * y - 350)) (min (abs (x * y - 370)) (min (abs (x * y - 3700)) (abs (x * y - 4000)))))

theorem product_of_000412_and_9243817_is_closest_to_3600 :
  product_closest_to 0.000412 9243817 3600 :=
by
  sorry

end product_of_000412_and_9243817_is_closest_to_3600_l987_98778


namespace john_money_left_l987_98740

variable (q : ℝ) 

def cost_soda := q
def cost_medium_pizza := 3 * q
def cost_small_pizza := 2 * q

def total_cost := 4 * cost_soda q + 2 * cost_medium_pizza q + 3 * cost_small_pizza q

theorem john_money_left (h : total_cost q = 16 * q) : 50 - total_cost q = 50 - 16 * q := by
  simp [total_cost, cost_soda, cost_medium_pizza, cost_small_pizza]
  sorry

end john_money_left_l987_98740


namespace gcd_of_a_and_b_lcm_of_a_and_b_l987_98780

def a : ℕ := 2 * 3 * 7
def b : ℕ := 2 * 3 * 3 * 5

theorem gcd_of_a_and_b : Nat.gcd a b = 6 := by
  sorry

theorem lcm_of_a_and_b : Nat.lcm a b = 630 := by
  sorry

end gcd_of_a_and_b_lcm_of_a_and_b_l987_98780


namespace which_polygon_covers_ground_l987_98771

def is_tessellatable (n : ℕ) : Prop :=
  let interior_angle := (n - 2) * 180 / n
  360 % interior_angle = 0

theorem which_polygon_covers_ground :
  is_tessellatable 6 ∧ ¬is_tessellatable 5 ∧ ¬is_tessellatable 8 ∧ ¬is_tessellatable 12 :=
by
  sorry

end which_polygon_covers_ground_l987_98771


namespace arithmetic_sequence_common_difference_l987_98726

theorem arithmetic_sequence_common_difference :
  ∃ d : ℤ, 
    (∀ n, n ≤ 6 → 23 + (n - 1) * d > 0) ∧ 
    (∀ n, n ≥ 7 → 23 + (n - 1) * d < 0) ∧
    d = -4 :=
by
  sorry

end arithmetic_sequence_common_difference_l987_98726


namespace sheila_paintings_l987_98783

theorem sheila_paintings (a b : ℕ) (h1 : a = 9) (h2 : b = 9) : a + b = 18 :=
by
  sorry

end sheila_paintings_l987_98783


namespace positive_real_x_condition_l987_98724

-- We define the conditions:
variables (x : ℝ)
#check (1 - x^4)
#check (1 + x^4)

-- The main proof statement:
theorem positive_real_x_condition (h1 : x > 0) 
    (h2 : (Real.sqrt (Real.sqrt (1 - x^4)) + Real.sqrt (Real.sqrt (1 + x^4)) = 1)) :
    (x^8 = 35 / 36) :=
sorry

end positive_real_x_condition_l987_98724
