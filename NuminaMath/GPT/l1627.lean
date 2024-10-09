import Mathlib

namespace remainder_when_divided_by_product_l1627_162733

noncomputable def Q : Polynomial ℝ := sorry

theorem remainder_when_divided_by_product (Q : Polynomial ℝ)
    (h1 : Q.eval 20 = 100)
    (h2 : Q.eval 100 = 20) :
    ∃ R : Polynomial ℝ, ∃ a b : ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 100) * R + Polynomial.C a * Polynomial.X + Polynomial.C b ∧
    a = -1 ∧ b = 120 :=
by
  sorry

end remainder_when_divided_by_product_l1627_162733


namespace sequence_n_value_l1627_162725

theorem sequence_n_value (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 3) (h3 : a n = 2008) : n = 670 :=
by
 sorry

end sequence_n_value_l1627_162725


namespace Intersect_A_B_l1627_162760

-- Defining the sets A and B according to the problem's conditions
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x ∈ Set.univ | x^2 - 5*x + 4 < 0}

-- Prove that the intersection of A and B is {2}
theorem Intersect_A_B : A ∩ B = {2} := by
  sorry

end Intersect_A_B_l1627_162760


namespace cookies_with_five_cups_of_flour_l1627_162712

-- Define the conditions
def initial_cookies : ℕ := 24
def initial_flour : ℕ := 3
def additional_flour : ℕ := 5

-- State the problem
theorem cookies_with_five_cups_of_flour :
  (initial_cookies / initial_flour) * additional_flour = 40 :=
by
  -- Placeholder for proof
  sorry

end cookies_with_five_cups_of_flour_l1627_162712


namespace hapok_max_coins_l1627_162754

/-- The maximum number of coins Hapok can guarantee himself regardless of Glazok's actions is 46 coins. -/
theorem hapok_max_coins (total_coins : ℕ) (max_handfuls : ℕ) (coins_per_handful : ℕ) :
  total_coins = 100 ∧ max_handfuls = 9 ∧ (∀ h : ℕ, h ≤ max_handfuls) ∧ coins_per_handful ≤ total_coins →
  ∃ k : ℕ, k ≤ total_coins ∧ k = 46 :=
by {
  sorry
}

end hapok_max_coins_l1627_162754


namespace largest_K_is_1_l1627_162702

noncomputable def largest_K_vip (K : ℝ) : Prop :=
  ∀ (k : ℝ) (a b c : ℝ), 
  0 ≤ k ∧ k ≤ K → 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a^2 + b^2 + c^2 + k * a * b * c = k + 3 → 
  a + b + c ≤ 3

theorem largest_K_is_1 : largest_K_vip 1 :=
sorry

end largest_K_is_1_l1627_162702


namespace largest_allowed_set_size_correct_l1627_162742

noncomputable def largest_allowed_set_size (N : ℕ) : ℕ :=
  N - Nat.floor (N / 4)

def is_allowed (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a ∣ b → b ∣ c → False)

theorem largest_allowed_set_size_correct (N : ℕ) (hN : 0 < N) : 
  ∃ S : Finset ℕ, is_allowed S ∧ S.card = largest_allowed_set_size N := sorry

end largest_allowed_set_size_correct_l1627_162742


namespace actual_price_per_gallon_l1627_162726

variable (x : ℝ)
variable (expected_price : ℝ := x) -- price per gallon that the motorist expected to pay
variable (total_cash : ℝ := 12 * x) -- total cash to buy 12 gallons at expected price
variable (actual_price : ℝ := x + 0.30) -- actual price per gallon
variable (equation : 12 * x = 10 * (x + 0.30)) -- total cash equals the cost of 10 gallons at actual price

theorem actual_price_per_gallon (x : ℝ) (h : 12 * x = 10 * (x + 0.30)) : x + 0.30 = 1.80 := 
by 
  sorry

end actual_price_per_gallon_l1627_162726


namespace june_ride_time_l1627_162771

theorem june_ride_time (d1 d2 : ℝ) (t1 : ℝ) (rate : ℝ) (t2 : ℝ) :
  d1 = 2 ∧ t1 = 6 ∧ rate = (d1 / t1) ∧ d2 = 5 ∧ t2 = d2 / rate → t2 = 15 := by
  intros h
  sorry

end june_ride_time_l1627_162771


namespace john_steps_l1627_162735

/-- John climbs up 9 flights of stairs. Each flight is 10 feet. -/
def flights := 9
def flight_height_feet := 10

/-- Conversion factor between feet and inches. -/
def feet_to_inches := 12

/-- Each step is 18 inches. -/
def step_height_inches := 18

/-- The total number of steps John climbs. -/
theorem john_steps :
  (flights * flight_height_feet * feet_to_inches) / step_height_inches = 60 :=
by
  sorry

end john_steps_l1627_162735


namespace inscribed_circle_quadrilateral_l1627_162750

theorem inscribed_circle_quadrilateral
  (AB CD BC AD AC BD E : ℝ)
  (r1 r2 r3 r4 : ℝ)
  (h1 : BC = AD)
  (h2 : AB + CD = BC + AD)
  (h3 : ∃ E, ∃ AC BD, AC * BD = E∧ AC > 0 ∧ BD > 0)
  (h_r1 : r1 > 0)
  (h_r2 : r2 > 0)
  (h_r3 : r3 > 0)
  (h_r4 : r4 > 0):
  1 / r1 + 1 / r3 = 1 / r2 + 1 / r4 := 
by
  sorry

end inscribed_circle_quadrilateral_l1627_162750


namespace jindra_initial_dice_count_l1627_162790

-- Given conditions about the dice stacking
def number_of_dice_per_layer : ℕ := 36
def layers_stacked_completely : ℕ := 6
def dice_received : ℕ := 18

-- We need to prove that the initial number of dice Jindra had is 234
theorem jindra_initial_dice_count : 
    (layers_stacked_completely * number_of_dice_per_layer + dice_received) = 234 :=
    by 
        sorry

end jindra_initial_dice_count_l1627_162790


namespace inequality_always_true_l1627_162777

theorem inequality_always_true (a : ℝ) : (∀ x : ℝ, |x - 1| - |x + 2| ≤ a) ↔ 3 ≤ a :=
by
  sorry

end inequality_always_true_l1627_162777


namespace degrees_to_radians_216_l1627_162745

theorem degrees_to_radians_216 : (216 / 180 : ℝ) * Real.pi = (6 / 5 : ℝ) * Real.pi := by
  sorry

end degrees_to_radians_216_l1627_162745


namespace repeating_decimal_sum_l1627_162776

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9

theorem repeating_decimal_sum : repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l1627_162776


namespace prime_for_all_k_l1627_162752

theorem prime_for_all_k (n : ℕ) (h_n : n ≥ 2) (h_prime : ∀ k : ℕ, k ≤ Nat.sqrt (n / 3) → Prime (k^2 + k + n)) :
  ∀ k : ℕ, k ≤ n - 2 → Prime (k^2 + k + n) :=
by
  intros
  sorry

end prime_for_all_k_l1627_162752


namespace smallest_negative_integer_solution_l1627_162704

theorem smallest_negative_integer_solution :
  ∃ x : ℤ, 45 * x + 8 ≡ 5 [ZMOD 24] ∧ x = -7 :=
sorry

end smallest_negative_integer_solution_l1627_162704


namespace find_min_length_seg_O1O2_l1627_162708

noncomputable def minimum_length_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ) (dist_YZ : ℝ) (dist_YW : ℝ)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : ℝ :=
  dist O1 O2

theorem find_min_length_seg_O1O2 
  (X Y Z W : ℝ × ℝ) 
  (dist_XY : ℝ := 1)
  (dist_YZ : ℝ := 3)
  (dist_YW : ℝ := 5)
  (O1 O2 : ℝ × ℝ) 
  (circumcenter1 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (circumcenter2 : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ × ℝ)
  (h1 : dist X Y = dist_XY) 
  (h2 : dist Y Z = dist_YZ) 
  (h3 : dist Y W = dist_YW) 
  (hO1 : O1 = circumcenter1 W X Y)
  (hO2 : O2 = circumcenter2 W Y Z)
  : minimum_length_O1O2 X Y Z W dist_XY dist_YZ dist_YW O1 O2 circumcenter1 circumcenter2 h1 h2 h3 hO1 hO2 = 2 :=
sorry

end find_min_length_seg_O1O2_l1627_162708


namespace truncated_trigonal_pyramid_circumscribed_sphere_l1627_162718

theorem truncated_trigonal_pyramid_circumscribed_sphere
  (h R_1 R_2 : ℝ)
  (O_1 T_1 O_2 T_2 : ℝ)
  (circumscribed : ∃ r : ℝ, h = 2 * r)
  (sphere_touches_lower_base : ∀ P, dist P T_1 = r)
  (sphere_touches_upper_base : ∀ Q, dist Q T_2 = r)
  (dist_O1_T1 : ℝ)
  (dist_O2_T2 : ℝ) :
  R_1 * R_2 * h^2 = (R_1^2 - dist_O1_T1^2) * (R_2^2 - dist_O2_T2^2) :=
sorry

end truncated_trigonal_pyramid_circumscribed_sphere_l1627_162718


namespace determinant_A_l1627_162753

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![
    ![  2,  4, -2],
    ![  3, -1,  5],
    ![-1,  3,  2]
  ]

theorem determinant_A : det A = -94 := by
  sorry

end determinant_A_l1627_162753


namespace min_value_f_min_achieved_l1627_162789

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 3)) + x

theorem min_value_f : ∀ x : ℝ, x > 3 → f x ≥ 5 :=
by
  intro x hx
  sorry

theorem min_achieved : f 4 = 5 :=
by
  sorry

end min_value_f_min_achieved_l1627_162789


namespace ratio_problem_l1627_162713

theorem ratio_problem (x n : ℕ) (h1 : 5 * x = n) (h2 : n = 65) : x = 13 :=
by
  sorry

end ratio_problem_l1627_162713


namespace range_of_m_min_value_of_7a_4b_l1627_162707

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + 1| - m ≥ 0) → m ≤ 2 :=
sorry

theorem min_value_of_7a_4b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_eq : 2 / (3 * a + b) + 1 / (a + 2 * b) = 2) : 7 * a + 4 * b ≥ 9 / 2 :=
sorry

end range_of_m_min_value_of_7a_4b_l1627_162707


namespace production_days_l1627_162701

theorem production_days (n : ℕ) (P : ℕ) (h1: P = n * 50) 
    (h2: (P + 110) / (n + 1) = 55) : n = 11 :=
by
  sorry

end production_days_l1627_162701


namespace find_m_l1627_162720

noncomputable def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem find_m (m : ℝ) :
  let a := (1, m)
  let b := (3, -2)
  are_parallel (vector_sum a b) b → m = -2 / 3 :=
by
  sorry

end find_m_l1627_162720


namespace length_of_24_l1627_162757

def length_of_integer (k : ℕ) : ℕ :=
  k.factors.length

theorem length_of_24 : length_of_integer 24 = 4 :=
by
  sorry

end length_of_24_l1627_162757


namespace number_of_Al_atoms_l1627_162703

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90
def number_of_Br_atoms : ℕ := 3
def molecular_weight : ℝ := 267

theorem number_of_Al_atoms (x : ℝ) : 
  molecular_weight = (atomic_weight_Al * x) + (atomic_weight_Br * number_of_Br_atoms) → 
  x = 1 :=
by
  sorry

end number_of_Al_atoms_l1627_162703


namespace simplify_x_cubed_simplify_expr_l1627_162755

theorem simplify_x_cubed (x : ℝ) : x * (x + 3) * (x + 5) = x^3 + 8 * x^2 + 15 * x := by
  sorry

theorem simplify_expr (x y : ℝ) : (5 * x + 2 * y) * (5 * x - 2 * y) - 5 * x * (5 * x - 3 * y) = -4 * y^2 + 15 * x * y := by
  sorry

end simplify_x_cubed_simplify_expr_l1627_162755


namespace sum_of_factors_eq_12_l1627_162714

-- Define the polynomial for n = 1
def poly (x : ℤ) : ℤ := x^5 + x + 1

-- Define the two factors when x = 2
def factor1 (x : ℤ) : ℤ := x^3 - x^2 + 1
def factor2 (x : ℤ) : ℤ := x^2 + x + 1

-- State the sum of the two factors at x = 2 equals 12
theorem sum_of_factors_eq_12 (x : ℤ) (h : x = 2) : factor1 x + factor2 x = 12 :=
by {
  sorry
}

end sum_of_factors_eq_12_l1627_162714


namespace no_positive_numbers_satisfy_conditions_l1627_162798

theorem no_positive_numbers_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c = ab + ac + bc) ∧ (ab + ac + bc = abc) :=
by
  sorry

end no_positive_numbers_satisfy_conditions_l1627_162798


namespace prime_sum_diff_condition_unique_l1627_162756

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def can_be_written_as_sum_of_two_primes (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime (p + q)

def can_be_written_as_difference_of_two_primes (p r : ℕ) : Prop :=
  is_prime p ∧ is_prime r ∧ is_prime (p - r)

-- Question rewritten as Lean statement
theorem prime_sum_diff_condition_unique (p q r : ℕ) :
  is_prime p →
  can_be_written_as_sum_of_two_primes (p - 2) p →
  can_be_written_as_difference_of_two_primes (p + 2) p →
  p = 5 :=
sorry

end prime_sum_diff_condition_unique_l1627_162756


namespace total_envelopes_l1627_162769

def total_stamps : ℕ := 52
def lighter_envelopes : ℕ := 6
def stamps_per_lighter_envelope : ℕ := 2
def stamps_per_heavier_envelope : ℕ := 5

theorem total_envelopes (total_stamps lighter_envelopes stamps_per_lighter_envelope stamps_per_heavier_envelope : ℕ) 
  (h : total_stamps = 52 ∧ lighter_envelopes = 6 ∧ stamps_per_lighter_envelope = 2 ∧ stamps_per_heavier_envelope = 5) : 
  lighter_envelopes + (total_stamps - (stamps_per_lighter_envelope * lighter_envelopes)) / stamps_per_heavier_envelope = 14 :=
by
  sorry

end total_envelopes_l1627_162769


namespace journey_ratio_proof_l1627_162781

def journey_ratio (x y : ℝ) : Prop :=
  (x + y = 448) ∧ (x / 21 + y / 24 = 20) → (x / y = 1)

theorem journey_ratio_proof : ∃ x y : ℝ, journey_ratio x y :=
by
  sorry

end journey_ratio_proof_l1627_162781


namespace train_length_l1627_162780

theorem train_length
  (t1 : ℕ) (t2 : ℕ)
  (d_platform : ℕ)
  (h1 : t1 = 8)
  (h2 : t2 = 20)
  (h3 : d_platform = 279)
  : ∃ (L : ℕ), (L : ℕ) = 186 :=
by
  sorry

end train_length_l1627_162780


namespace length_on_ninth_day_l1627_162706

-- Define relevant variables and conditions.
variables (a1 d : ℕ)

-- Define conditions as hypotheses.
def problem_conditions : Prop :=
  (7 * a1 + 21 * d = 28) ∧ 
  (a1 + d + a1 + 4 * d + a1 + 7 * d = 15)

theorem length_on_ninth_day (h : problem_conditions a1 d) : (a1 + 8 * d = 9) :=
  sorry

end length_on_ninth_day_l1627_162706


namespace largest_divisor_of_10000_not_dividing_9999_l1627_162775

theorem largest_divisor_of_10000_not_dividing_9999 : ∃ d, d ∣ 10000 ∧ ¬ (d ∣ 9999) ∧ ∀ y, (y ∣ 10000 ∧ ¬ (y ∣ 9999)) → y ≤ d := 
by
  sorry

end largest_divisor_of_10000_not_dividing_9999_l1627_162775


namespace number_of_cows_l1627_162795

variable (x y z : ℕ)

theorem number_of_cows (h1 : 4 * x + 2 * y + 2 * z = 24 + 2 * (x + y + z)) (h2 : z = y / 2) : x = 12 := 
sorry

end number_of_cows_l1627_162795


namespace number_of_valid_5_digit_numbers_l1627_162717

def is_multiple_of_16 (n : Nat) : Prop := 
  n % 16 = 0

theorem number_of_valid_5_digit_numbers : Nat := 
  sorry

example : number_of_valid_5_digit_numbers = 90 :=
  sorry

end number_of_valid_5_digit_numbers_l1627_162717


namespace slices_leftover_l1627_162797

def total_slices (small_pizzas large_pizzas : ℕ) : ℕ :=
  (3 * 4) + (2 * 8)

def slices_eaten_by_people (george bob susie bill fred mark : ℕ) : ℕ :=
  george + bob + susie + bill + fred + mark

theorem slices_leftover :
  total_slices 3 2 - slices_eaten_by_people 3 4 2 3 3 3 = 10 :=
by sorry

end slices_leftover_l1627_162797


namespace fraction_result_l1627_162711

theorem fraction_result (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (2 * x + 3 * y) / (x - 2 * y) = 3) : 
  (x + 2 * y) / (2 * x - y) = 11 / 17 :=
sorry

end fraction_result_l1627_162711


namespace sum_first_10_terms_l1627_162794

-- Define the conditions for the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_sequence (b c d : ℝ) : Prop :=
  2 * c = b + d

def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧
  geometric_sequence a q ∧
  arithmetic_sequence (4 * a 1) (2 * a 2) (a 3)

-- Define the sum of the first n terms of a geometric sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Prove the final result
theorem sum_first_10_terms (a : ℕ → ℝ) (q : ℝ) (h : conditions a q) :
  sum_first_n_terms a 10 = 1023 :=
sorry

end sum_first_10_terms_l1627_162794


namespace first_grade_sample_count_l1627_162761

-- Defining the total number of students and their ratio in grades 1, 2, and 3.
def total_students : ℕ := 2400
def ratio_grade1 : ℕ := 5
def ratio_grade2 : ℕ := 4
def ratio_grade3 : ℕ := 3
def total_ratio := ratio_grade1 + ratio_grade2 + ratio_grade3

-- Defining the sample size
def sample_size : ℕ := 120

-- Proving that the number of first-grade students sampled should be 50.
theorem first_grade_sample_count : 
  (sample_size * ratio_grade1) / total_ratio = 50 :=
by
  -- sorry is added here to skip the proof
  sorry

end first_grade_sample_count_l1627_162761


namespace pirates_total_coins_l1627_162758

theorem pirates_total_coins (x : ℕ) (h : (x * (x + 1)) / 2 = 5 * x) : 6 * x = 54 := by
  -- The proof will go here, but it's currently omitted with 'sorry'
  sorry

end pirates_total_coins_l1627_162758


namespace gcd_18_30_l1627_162747

theorem gcd_18_30 : Int.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l1627_162747


namespace fill_time_l1627_162764

def inflow_rate : ℕ := 24 -- gallons per second
def outflow_rate : ℕ := 4 -- gallons per second
def basin_volume : ℕ := 260 -- gallons

theorem fill_time (inflow_rate outflow_rate basin_volume : ℕ) (h₁ : inflow_rate = 24) (h₂ : outflow_rate = 4) 
  (h₃ : basin_volume = 260) : basin_volume / (inflow_rate - outflow_rate) = 13 :=
by
  sorry

end fill_time_l1627_162764


namespace line_equation_l1627_162716

theorem line_equation 
  (m b k : ℝ) 
  (h1 : ∀ k, abs ((k^2 + 4 * k + 4) - (m * k + b)) = 4)
  (h2 : m * 2 + b = 8) 
  (h3 : b ≠ 0) : 
  m = 8 ∧ b = -8 :=
by sorry

end line_equation_l1627_162716


namespace Davey_Barbeck_ratio_is_1_l1627_162773

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

end Davey_Barbeck_ratio_is_1_l1627_162773


namespace smallest_even_divisible_by_20_and_60_l1627_162723

theorem smallest_even_divisible_by_20_and_60 : ∃ x, (Even x) ∧ (x % 20 = 0) ∧ (x % 60 = 0) ∧ (∀ y, (Even y) ∧ (y % 20 = 0) ∧ (y % 60 = 0) → x ≤ y) → x = 60 :=
by
  sorry

end smallest_even_divisible_by_20_and_60_l1627_162723


namespace EmilySixthQuizScore_l1627_162774

theorem EmilySixthQuizScore (x : ℕ) : 
  let scores := [85, 92, 88, 90, 93]
  let total_scores_with_x := scores.sum + x
  let desired_average := 91
  total_scores_with_x = 6 * desired_average → x = 98 := by
  sorry

end EmilySixthQuizScore_l1627_162774


namespace actual_positions_correct_l1627_162739

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l1627_162739


namespace average_price_correct_l1627_162709

-- Define the conditions
def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1480
def books_shop2 : ℕ := 55
def price_shop2 : ℕ := 920

-- Define the total books and total price based on conditions
def total_books : ℕ := books_shop1 + books_shop2
def total_price : ℕ := price_shop1 + price_shop2

-- Define the average price based on total books and total price
def average_price : ℕ := total_price / total_books

-- Theorem stating the average price per book Sandy paid
theorem average_price_correct : average_price = 20 :=
  by
  sorry

end average_price_correct_l1627_162709


namespace fifth_graders_more_than_eighth_graders_l1627_162785

theorem fifth_graders_more_than_eighth_graders 
  (cost : ℕ) 
  (h_cost : cost > 0) 
  (h_div_234 : 234 % cost = 0) 
  (h_div_312 : 312 % cost = 0) 
  (h_40_fifth_graders : 40 > 0) : 
  (312 / cost) - (234 / cost) = 6 := 
by 
  sorry

end fifth_graders_more_than_eighth_graders_l1627_162785


namespace solve_for_question_mark_l1627_162787

theorem solve_for_question_mark :
  let question_mark := 4135 / 45
  (45 * question_mark) + (625 / 25) - (300 * 4) = 2950 + (1500 / (75 * 2)) :=
by
  let question_mark := 4135 / 45
  sorry

end solve_for_question_mark_l1627_162787


namespace total_apple_trees_l1627_162746

-- Definitions and conditions
def ava_trees : ℕ := 9
def lily_trees : ℕ := ava_trees - 3
def total_trees : ℕ := ava_trees + lily_trees

-- Statement to be proved
theorem total_apple_trees :
  total_trees = 15 := by
  sorry

end total_apple_trees_l1627_162746


namespace another_representation_l1627_162719

def positive_int_set : Set ℕ := {x | x > 0}

theorem another_representation :
  {x ∈ positive_int_set | x - 3 < 2} = {1, 2, 3, 4} :=
by
  sorry

end another_representation_l1627_162719


namespace pablo_puzzle_l1627_162749

open Nat

theorem pablo_puzzle (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
    (pieces_per_five_puzzles : ℕ) (num_five_puzzles : ℕ) (total_pieces : ℕ) 
    (num_eight_puzzles : ℕ) :

    pieces_per_hour = 100 →
    hours_per_day = 7 →
    days = 7 →
    pieces_per_five_puzzles = 500 →
    num_five_puzzles = 5 →
    num_eight_puzzles = 8 →
    total_pieces = (pieces_per_hour * hours_per_day * days) →
    num_eight_puzzles * (total_pieces - num_five_puzzles * pieces_per_five_puzzles) / num_eight_puzzles = 300 :=
by
  intros
  sorry

end pablo_puzzle_l1627_162749


namespace gcd_f100_f101_l1627_162784

def f (x : ℤ) : ℤ := x^2 - 3 * x + 2023

theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 2 :=
by
  sorry

end gcd_f100_f101_l1627_162784


namespace problem_solution_l1627_162740

theorem problem_solution (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (h : x - y = x / y) : 
  (1 / x - 1 / y = -1 / y^2) := 
by sorry

end problem_solution_l1627_162740


namespace contrapositive_equiv_l1627_162786

variable (x : Type)

theorem contrapositive_equiv (Q R : x → Prop) :
  (∀ x, Q x → R x) ↔ (∀ x, ¬ (R x) → ¬ (Q x)) :=
by
  sorry

end contrapositive_equiv_l1627_162786


namespace difference_of_numbers_l1627_162730

theorem difference_of_numbers 
  (a b : ℕ) 
  (h1 : a + b = 23976)
  (h2 : b % 8 = 0)
  (h3 : a = 7 * b / 8) : 
  b - a = 1598 :=
sorry

end difference_of_numbers_l1627_162730


namespace eval_f_at_3_l1627_162766

-- Define the polynomial function
def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x^2 + 2 * x - 1

-- State the theorem to prove f(3) = 41
theorem eval_f_at_3 : f 3 = 41 :=
by
  -- Proof would go here
  sorry

end eval_f_at_3_l1627_162766


namespace distance_AC_in_terms_of_M_l1627_162705

-- Define the given constants and the relevant equations
variables (M x : ℝ) (AB BC AC : ℝ)
axiom distance_eq_add : AB = M + BC
axiom time_AB : (M + x) / 7 = x / 5
axiom time_BC : BC = x
axiom time_S : (M + x + x) = AC

theorem distance_AC_in_terms_of_M : AC = 6 * M :=
by
  sorry

end distance_AC_in_terms_of_M_l1627_162705


namespace maria_candy_remaining_l1627_162724

theorem maria_candy_remaining :
  let c := 520.75
  let e := c / 2
  let g := 234.56
  let r := e - g
  r = 25.815 := by
  sorry

end maria_candy_remaining_l1627_162724


namespace probability_qualified_from_A_is_correct_l1627_162727

-- Given conditions:
def p_A : ℝ := 0.7
def pass_A : ℝ := 0.95

-- Define what we need to prove:
def qualified_from_A : ℝ := p_A * pass_A

-- Theorem statement
theorem probability_qualified_from_A_is_correct :
  qualified_from_A = 0.665 :=
by
  sorry

end probability_qualified_from_A_is_correct_l1627_162727


namespace frog_jump_distance_l1627_162770

variable (grasshopper_jump frog_jump mouse_jump : ℕ)
variable (H1 : grasshopper_jump = 19)
variable (H2 : grasshopper_jump = frog_jump + 4)
variable (H3 : mouse_jump = frog_jump - 44)

theorem frog_jump_distance : frog_jump = 15 := by
  sorry

end frog_jump_distance_l1627_162770


namespace quadratic_eq_with_given_roots_l1627_162799

theorem quadratic_eq_with_given_roots (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 12) :
    (a + b = 16) ∧ (a * b = 144) ∧ (∀ (x : ℝ), x^2 - (a + b) * x + (a * b) = 0 ↔ x^2 - 16 * x + 144 = 0) := by
  sorry

end quadratic_eq_with_given_roots_l1627_162799


namespace total_red_beads_l1627_162782

theorem total_red_beads (total_beads : ℕ) (pattern_length : ℕ) (green_beads : ℕ) (red_beads : ℕ) (yellow_beads : ℕ) 
                         (h_total: total_beads = 85) 
                         (h_pattern: pattern_length = green_beads + red_beads + yellow_beads) 
                         (h_cycle: green_beads = 3 ∧ red_beads = 4 ∧ yellow_beads = 1) : 
                         (red_beads * (total_beads / pattern_length)) + (min red_beads (total_beads % pattern_length)) = 42 :=
by
  sorry

end total_red_beads_l1627_162782


namespace number_of_people_after_10_years_l1627_162738

def number_of_people_after_n_years (n : ℕ) : ℕ :=
  Nat.recOn n 30 (fun k a_k => 3 * a_k - 20)

theorem number_of_people_after_10_years :
  number_of_people_after_n_years 10 = 1180990 := by
  sorry

end number_of_people_after_10_years_l1627_162738


namespace average_speed_correct_l1627_162729

def biking_time : ℕ := 30 -- in minutes
def biking_speed : ℕ := 16 -- in mph
def walking_time : ℕ := 90 -- in minutes
def walking_speed : ℕ := 4 -- in mph

theorem average_speed_correct :
  (biking_time / 60 * biking_speed + walking_time / 60 * walking_speed) / ((biking_time + walking_time) / 60) = 7 := by
  sorry

end average_speed_correct_l1627_162729


namespace domain_of_sqrt_fraction_l1627_162728

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end domain_of_sqrt_fraction_l1627_162728


namespace painting_together_time_l1627_162736

theorem painting_together_time (jamshid_time taimour_time time_together : ℝ) 
  (h1 : jamshid_time = taimour_time / 2)
  (h2 : taimour_time = 21)
  (h3 : time_together = 7) :
  (1 / taimour_time + 1 / jamshid_time) * time_together = 1 := 
sorry

end painting_together_time_l1627_162736


namespace fraction_sum_l1627_162778

variable (a b : ℝ)

theorem fraction_sum
  (hb : b + 1 ≠ 0) :
  (a / (b + 1)) + (2 * a / (b + 1)) - (3 * a / (b + 1)) = 0 :=
by sorry

end fraction_sum_l1627_162778


namespace sin_alpha_eq_sqrt5_over_3_l1627_162779

theorem sin_alpha_eq_sqrt5_over_3 {α : ℝ} (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = (Real.sqrt 5) / 3 :=
sorry

end sin_alpha_eq_sqrt5_over_3_l1627_162779


namespace quadratic_completing_square_l1627_162741

theorem quadratic_completing_square (b p : ℝ) (hb : b < 0)
  (h_quad_eq : ∀ x : ℝ, x^2 + b * x + (1 / 6) = (x + p)^2 + (1 / 18)) :
  b = - (2 / 3) :=
by
  sorry

end quadratic_completing_square_l1627_162741


namespace function_increasing_in_range_l1627_162715

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

theorem function_increasing_in_range (m : ℝ) :
  (3 / 2 ≤ m ∧ m < 3) ↔ (∀ x y : ℝ, x < y → f m x < f m y) := by
  sorry

end function_increasing_in_range_l1627_162715


namespace arithmetic_fraction_subtraction_l1627_162734

theorem arithmetic_fraction_subtraction :
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 :=
by
  sorry

end arithmetic_fraction_subtraction_l1627_162734


namespace range_of_m_inequality_system_l1627_162722

theorem range_of_m_inequality_system (m : ℝ) :
  (∀ x : ℤ, (-5 < x ∧ x ≤ m + 1) ↔ (x = -4 ∨ x = -3 ∨ x = -2)) →
  -3 ≤ m ∧ m < -2 :=
by
  sorry

end range_of_m_inequality_system_l1627_162722


namespace largest_k_consecutive_sum_l1627_162793

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum_l1627_162793


namespace silly_bills_count_l1627_162710

theorem silly_bills_count (x : ℕ) (h1 : x + 2 * (x + 11) + 3 * (x - 18) = 100) : x = 22 :=
by { sorry }

end silly_bills_count_l1627_162710


namespace find_a4_l1627_162763

variables {a : ℕ → ℝ} (q : ℝ) (h_positive : ∀ n, 0 < a n)
variables (h_seq : ∀ n, a (n+1) = q * a n)
variables (h1 : a 1 + (2/3) * a 2 = 3)
variables (h2 : (a 4)^2 = (1/9) * a 3 * a 7)

-- Proof problem statement
theorem find_a4 : a 4 = 27 :=
sorry

end find_a4_l1627_162763


namespace jane_picked_fraction_l1627_162737

-- Define the total number of tomatoes initially
def total_tomatoes : ℕ := 100

-- Define the number of tomatoes remaining at the end
def remaining_tomatoes : ℕ := 15

-- Define the number of tomatoes picked in the second week
def second_week_tomatoes : ℕ := 20

-- Define the number of tomatoes picked in the third week
def third_week_tomatoes : ℕ := 2 * second_week_tomatoes

theorem jane_picked_fraction :
  ∃ (f : ℚ), f = 1 / 4 ∧
    (f * total_tomatoes + second_week_tomatoes + third_week_tomatoes + remaining_tomatoes = total_tomatoes) :=
sorry

end jane_picked_fraction_l1627_162737


namespace linear_equation_check_l1627_162732

theorem linear_equation_check : 
  (∃ a b : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x + b = 1)) ∧ 
  ¬ (∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, a * x + b * y = 3)) ∧ 
  ¬ (∀ x : ℝ, x^2 - 2 * x = 0) ∧ 
  ¬ (∀ x : ℝ, x - 1 / x = 0) := 
sorry

end linear_equation_check_l1627_162732


namespace sqrt_factorial_product_l1627_162748

theorem sqrt_factorial_product :
  Nat.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24 := 
sorry

end sqrt_factorial_product_l1627_162748


namespace friends_meeting_time_l1627_162796

noncomputable def speed_B (t : ℕ) : ℝ := 4 + 0.75 * (t - 1)

noncomputable def distance_B (t : ℕ) : ℝ :=
  t * 4 + (0.375 * t * (t - 1))

noncomputable def distance_A (t : ℕ) : ℝ := 5 * t

theorem friends_meeting_time :
  ∃ t : ℝ, 5 * t + (t / 2) * (7.25 + 0.75 * t) = 120 ∧ t = 8 :=
by
  sorry

end friends_meeting_time_l1627_162796


namespace AMHSE_1988_l1627_162783

theorem AMHSE_1988 (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 :=
sorry

end AMHSE_1988_l1627_162783


namespace boris_possible_amount_l1627_162772

theorem boris_possible_amount (k : ℕ) : ∃ k : ℕ, 1 + 74 * k = 823 :=
by
  use 11
  sorry

end boris_possible_amount_l1627_162772


namespace calories_per_orange_is_correct_l1627_162791

noncomputable def calories_per_orange
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) : ℕ :=
by
  -- Definitions derived from conditions
  let total_pieces := oranges * pieces_per_orange
  let pieces_per_person := total_pieces / num_people
  let total_calories := calories_per_person
  have calories_per_piece := total_calories / pieces_per_person

  -- Conclusion
  have calories_per_orange := pieces_per_orange * calories_per_piece
  exact calories_per_orange

theorem calories_per_orange_is_correct
  (oranges pieces_per_orange num_people calories_per_person : ℕ)
  (h_oranges : oranges = 5)
  (h_pieces_per_orange : pieces_per_orange = 8)
  (h_num_people : num_people = 4)
  (h_calories_per_person : calories_per_person = 100) :
  calories_per_orange oranges pieces_per_orange num_people calories_per_person
    h_oranges h_pieces_per_orange h_num_people h_calories_per_person = 100 :=
by
  simp [calories_per_orange]
  sorry  -- Proof omitted

end calories_per_orange_is_correct_l1627_162791


namespace change_in_responses_max_min_diff_l1627_162765

open Classical

theorem change_in_responses_max_min_diff :
  let initial_yes := 40
  let initial_no := 40
  let initial_undecided := 20
  let end_yes := 60
  let end_no := 30
  let end_undecided := 10
  let min_change := 20
  let max_change := 80
  max_change - min_change = 60 := by
  intros; sorry

end change_in_responses_max_min_diff_l1627_162765


namespace compute_sum_bk_ck_l1627_162759

theorem compute_sum_bk_ck 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 3*x^2 - 2*x + 1 =
                (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + b3*x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -2 := 
sorry

end compute_sum_bk_ck_l1627_162759


namespace digit_divisibility_l1627_162768

theorem digit_divisibility : 
  (∃ (A : ℕ), A < 10 ∧ 
   (4573198080 + A) % 2 = 0 ∧ 
   (4573198080 + A) % 5 = 0 ∧ 
   (4573198080 + A) % 8 = 0 ∧ 
   (4573198080 + A) % 10 = 0 ∧ 
   (4573198080 + A) % 16 = 0 ∧ A = 0) := 
by { use 0; sorry }

end digit_divisibility_l1627_162768


namespace system1_solution_system2_solution_l1627_162751

-- For Question 1

theorem system1_solution (x y : ℝ) :
  (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ↔ (x = 5 ∧ y = 5) := 
sorry

-- For Question 2

theorem system2_solution (x y : ℝ) :
  (3 * (x + y) - 4 * (x - y) = 16) ∧ ((x + y)/2 + (x - y)/6 = 1) ↔ (x = 1/3 ∧ y = 7/3) := 
sorry

end system1_solution_system2_solution_l1627_162751


namespace minimal_degree_g_l1627_162700

theorem minimal_degree_g {f g h : Polynomial ℝ} 
  (h_eq : 2 * f + 5 * g = h)
  (deg_f : f.degree = 6)
  (deg_h : h.degree = 10) : 
  g.degree = 10 :=
sorry

end minimal_degree_g_l1627_162700


namespace angle_sum_around_point_l1627_162767

theorem angle_sum_around_point (x : ℝ) (h : 2 * x + 140 = 360) : x = 110 := 
  sorry

end angle_sum_around_point_l1627_162767


namespace fraction_zero_value_x_l1627_162788

theorem fraction_zero_value_x (x : ℝ) (h1 : (x - 2) / (1 - x) = 0) (h2 : 1 - x ≠ 0) : x = 2 := 
sorry

end fraction_zero_value_x_l1627_162788


namespace range_of_k_l1627_162744

-- Definitions for the conditions of p and q
def is_ellipse (k : ℝ) : Prop := (0 < k) ∧ (k < 4)
def is_hyperbola (k : ℝ) : Prop := 1 < k ∧ k < 3

-- The main proposition
theorem range_of_k (k : ℝ) : (is_ellipse k ∨ is_hyperbola k) → (1 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l1627_162744


namespace monotonic_increasing_f_l1627_162731

theorem monotonic_increasing_f (f g : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) 
  (hg : ∀ x, g (-x) = g x) (hfg : ∀ x, f x + g x = 3^x) :
  ∀ a b : ℝ, a > b → f a > f b :=
sorry

end monotonic_increasing_f_l1627_162731


namespace percentage_increase_B_over_C_l1627_162762

noncomputable def A_m : ℕ := 537600 / 12
noncomputable def C_m : ℕ := 16000
noncomputable def ratio : ℚ := 5 / 2

noncomputable def B_m (A_m : ℕ) : ℚ := (2 * A_m) / 5

theorem percentage_increase_B_over_C :
  B_m A_m = 17920 →
  C_m = 16000 →
  (B_m A_m - C_m) / C_m * 100 = 12 :=
by
  sorry

end percentage_increase_B_over_C_l1627_162762


namespace free_endpoints_eq_1001_l1627_162792

theorem free_endpoints_eq_1001 : 
  ∃ k : ℕ, 1 + 4 * k = 1001 :=
by {
  sorry
}

end free_endpoints_eq_1001_l1627_162792


namespace ratio_red_to_yellow_l1627_162743

structure MugCollection where
  total_mugs : ℕ
  red_mugs : ℕ
  blue_mugs : ℕ
  yellow_mugs : ℕ
  other_mugs : ℕ
  colors : ℕ

def HannahCollection : MugCollection :=
  { total_mugs := 40,
    red_mugs := 6,
    blue_mugs := 6 * 3,
    yellow_mugs := 12,
    other_mugs := 4,
    colors := 4 }

theorem ratio_red_to_yellow
  (hc : MugCollection)
  (h_total : hc.total_mugs = 40)
  (h_blue : hc.blue_mugs = 3 * hc.red_mugs)
  (h_yellow : hc.yellow_mugs = 12)
  (h_other : hc.other_mugs = 4)
  (h_colors : hc.colors = 4) :
  hc.red_mugs / hc.yellow_mugs = 1 / 2 := by
  sorry

end ratio_red_to_yellow_l1627_162743


namespace percentage_not_caught_l1627_162721

theorem percentage_not_caught (x : ℝ) (h1 : 22 + x = 25.88235294117647) : x = 3.88235294117647 :=
sorry

end percentage_not_caught_l1627_162721
