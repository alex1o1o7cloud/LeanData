import Mathlib

namespace NUMINAMATH_GPT_unique_passenger_counts_l1358_135815

def train_frequencies : Nat × Nat × Nat := (6, 4, 3)
def train_passengers_leaving : Nat × Nat × Nat := (200, 300, 150)
def train_passengers_taking : Nat × Nat × Nat := (320, 400, 280)
def trains_per_hour (freq : Nat) : Nat := 60 / freq

def total_passengers_leaving : Nat :=
  let t1 := (trains_per_hour 10) * 200
  let t2 := (trains_per_hour 15) * 300
  let t3 := (trains_per_hour 20) * 150
  t1 + t2 + t3

def total_passengers_taking : Nat :=
  let t1 := (trains_per_hour 10) * 320
  let t2 := (trains_per_hour 15) * 400
  let t3 := (trains_per_hour 20) * 280
  t1 + t2 + t3

theorem unique_passenger_counts :
  total_passengers_leaving = 2850 ∧ total_passengers_taking = 4360 := by
  sorry

end NUMINAMATH_GPT_unique_passenger_counts_l1358_135815


namespace NUMINAMATH_GPT_ordered_pair_proportional_l1358_135828

theorem ordered_pair_proportional (p q : ℝ) (h : (3 : ℝ) • (-4 : ℝ) = (5 : ℝ) • p ∧ (3 : ℝ) • q = (5 : ℝ) • (-4 : ℝ)) :
  (p, q) = (5 / 2, -8) :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_proportional_l1358_135828


namespace NUMINAMATH_GPT_triangle_count_lower_bound_l1358_135849

theorem triangle_count_lower_bound (n m : ℕ) (S : Finset (ℕ × ℕ))
  (hS : ∀ (a b : ℕ), (a, b) ∈ S → 1 ≤ a ∧ a < b ∧ b ≤ n) (hm : S.card = m) :
  ∃T, T ≥ 4 * m * (m - n^2 / 4) / (3 * n) := 
by 
  sorry

end NUMINAMATH_GPT_triangle_count_lower_bound_l1358_135849


namespace NUMINAMATH_GPT_chameleons_impossible_all_white_l1358_135810

/--
On Easter Island, there are initial counts of blue (12), white (25), and red (8) chameleons.
When two chameleons of different colors meet, they both change to the third color.
Prove that it is impossible for all chameleons to become white.
--/
theorem chameleons_impossible_all_white :
  let n1 := 12 -- Blue chameleons
  let n2 := 25 -- White chameleons
  let n3 := 8  -- Red chameleons
  (∀ (n1 n2 n3 : ℕ), (n1 + n2 + n3 = 45) → 
   ∀ (k : ℕ), ∃ m1 m2 m3 : ℕ, (m1 - m2) % 3 = (n1 - n2) % 3 ∧ (m1 - m3) % 3 = (n1 - n3) % 3 ∧ 
   (m2 - m3) % 3 = (n2 - n3) % 3) → False := sorry

end NUMINAMATH_GPT_chameleons_impossible_all_white_l1358_135810


namespace NUMINAMATH_GPT_double_even_l1358_135814

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Lean statement of the mathematically equivalent proof problem
theorem double_even (f : ℝ → ℝ) (h : is_even_function f) : is_even_function (f ∘ f) :=
by
  sorry

end NUMINAMATH_GPT_double_even_l1358_135814


namespace NUMINAMATH_GPT_meal_combinations_l1358_135878

def number_of_menu_items : ℕ := 15

theorem meal_combinations (different_orderings : ∀ Yann Camille : ℕ, Yann ≠ Camille → Yann ≤ number_of_menu_items ∧ Camille ≤ number_of_menu_items) : 
  (number_of_menu_items * (number_of_menu_items - 1)) = 210 :=
by sorry

end NUMINAMATH_GPT_meal_combinations_l1358_135878


namespace NUMINAMATH_GPT_ava_average_speed_l1358_135862

noncomputable def initial_odometer : ℕ := 14941
noncomputable def final_odometer : ℕ := 15051
noncomputable def elapsed_time : ℝ := 4 -- hours

theorem ava_average_speed :
  (final_odometer - initial_odometer) / elapsed_time = 27.5 :=
by
  sorry

end NUMINAMATH_GPT_ava_average_speed_l1358_135862


namespace NUMINAMATH_GPT_remaining_days_temperature_l1358_135839

theorem remaining_days_temperature (avg_temp : ℕ) (d1 d2 d3 d4 d5 : ℕ) :
  avg_temp = 60 →
  d1 = 40 →
  d2 = 40 →
  d3 = 40 →
  d4 = 80 →
  d5 = 80 →
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  total_temp - known_temp = 140 := 
by
  intros _ _ _ _ _ _
  let total_temp := avg_temp * 7
  let known_temp := d1 + d2 + d3 + d4 + d5
  sorry

end NUMINAMATH_GPT_remaining_days_temperature_l1358_135839


namespace NUMINAMATH_GPT_greatest_divisible_by_11_l1358_135806

theorem greatest_divisible_by_11 :
  ∃ (A B C : ℕ), A ≠ C ∧ A ≠ B ∧ B ≠ C ∧ 
  (∀ n, n = 10000 * A + 1000 * B + 100 * C + 10 * B + A → n = 96569) ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 11 = 0 :=
sorry

end NUMINAMATH_GPT_greatest_divisible_by_11_l1358_135806


namespace NUMINAMATH_GPT_tan_theta_correct_l1358_135876

noncomputable def cos_double_angle (θ : ℝ) : ℝ := 2 * Real.cos θ ^ 2 - 1

theorem tan_theta_correct (θ : ℝ) (hθ₁ : θ > 0) (hθ₂ : θ < Real.pi / 2) 
  (h : 15 * cos_double_angle θ - 14 * Real.cos θ + 11 = 0) : Real.tan θ = Real.sqrt 5 / 2 :=
sorry

end NUMINAMATH_GPT_tan_theta_correct_l1358_135876


namespace NUMINAMATH_GPT_Esha_behind_Anusha_l1358_135835

/-- Define conditions for the race -/

def Anusha_speed := 100
def Banu_behind_when_Anusha_finishes := 10
def Banu_run_when_Anusha_finishes := Anusha_speed - Banu_behind_when_Anusha_finishes
def Esha_behind_when_Banu_finishes := 10
def Esha_run_when_Banu_finishes := Anusha_speed - Esha_behind_when_Banu_finishes
def Banu_speed_ratio := Banu_run_when_Anusha_finishes / Anusha_speed
def Esha_speed_ratio := Esha_run_when_Banu_finishes / Anusha_speed
def Esha_to_Anusha_speed_ratio := Esha_speed_ratio * Banu_speed_ratio
def Esha_run_when_Anusha_finishes := Anusha_speed * Esha_to_Anusha_speed_ratio

/-- Prove that Esha is 19 meters behind Anusha when Anusha finishes the race -/
theorem Esha_behind_Anusha {V_A V_B V_E : ℝ} :
  (V_B / V_A = 9 / 10) →
  (V_E / V_B = 9 / 10) →
  (Esha_run_when_Anusha_finishes = Anusha_speed * (9 / 10 * 9 / 10)) →
  Anusha_speed - Esha_run_when_Anusha_finishes = 19 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_Esha_behind_Anusha_l1358_135835


namespace NUMINAMATH_GPT_find_a_even_function_l1358_135866

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f x = (x + 1) * (x + a))  
  (h2 : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_even_function_l1358_135866


namespace NUMINAMATH_GPT_cat_daytime_catches_l1358_135809

theorem cat_daytime_catches
  (D : ℕ)
  (night_catches : ℕ := 2 * D)
  (total_catches : ℕ := D + night_catches)
  (h : total_catches = 24) :
  D = 8 := by
  sorry

end NUMINAMATH_GPT_cat_daytime_catches_l1358_135809


namespace NUMINAMATH_GPT_maximum_n_l1358_135861

theorem maximum_n (n : ℕ) (G : SimpleGraph (Fin n)) :
  (∃ (A : Fin n → Set (Fin 2020)),  ∀ i j, (G.Adj i j ↔ (A i ∩ A j ≠ ∅)) →
  n ≤ 89) := sorry

end NUMINAMATH_GPT_maximum_n_l1358_135861


namespace NUMINAMATH_GPT_evan_45_l1358_135812

theorem evan_45 (k n : ℤ) (h1 : n + (k * (2 * k - 1)) = 60) : 60 - n = 45 :=
by sorry

end NUMINAMATH_GPT_evan_45_l1358_135812


namespace NUMINAMATH_GPT_fraction_is_three_eighths_l1358_135848

theorem fraction_is_three_eighths (F N : ℝ) 
  (h1 : (4 / 5) * F * N = 24) 
  (h2 : (250 / 100) * N = 199.99999999999997) : 
  F = 3 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_is_three_eighths_l1358_135848


namespace NUMINAMATH_GPT_stratified_sampling_third_grade_l1358_135872

theorem stratified_sampling_third_grade (total_students : ℕ)
  (ratio_first_second_third : ℕ × ℕ × ℕ)
  (sample_size : ℕ) (r1 r2 r3 : ℕ) (h_ratio : ratio_first_second_third = (r1, r2, r3)) :
  total_students = 3000  ∧ ratio_first_second_third = (2, 3, 1)  ∧ sample_size = 180 →
  (sample_size * r3 / (r1 + r2 + r3) = 30) :=
sorry

end NUMINAMATH_GPT_stratified_sampling_third_grade_l1358_135872


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1358_135888

-- Prove \(2x = 4\) is a "difference solution equation"
theorem problem1 (x : ℝ) : (2 * x = 4) → x = 4 - 2 :=
by
  sorry

-- Given \(4x = ab + a\) is a "difference solution equation", prove \(3(ab + a) = 16\)
theorem problem2 (x ab a : ℝ) : (4 * x = ab + a) → 3 * (ab + a) = 16 :=
by
  sorry

-- Given \(4x = mn + m\) and \(-2x = mn + n\) are both "difference solution equations", prove \(3(mn + m) - 9(mn + n)^2 = 0\)
theorem problem3 (x mn m n : ℝ) :
  (4 * x = mn + m) ∧ (-2 * x = mn + n) → 3 * (mn + m) - 9 * (mn + n)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1358_135888


namespace NUMINAMATH_GPT_cdf_of_Z_pdf_of_Z_l1358_135875

noncomputable def f1 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 2 then 0.5 else 0

noncomputable def f2 (y : ℝ) : ℝ :=
  if 0 < y ∧ y < 2 then 0.5 else 0

noncomputable def G (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1

noncomputable def g (z : ℝ) : ℝ :=
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0

theorem cdf_of_Z (z : ℝ) : G z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z^2 / 8
  else if 2 < z ∧ z ≤ 4 then 1 - (4 - z)^2 / 8
  else 1 := sorry

theorem pdf_of_Z (z : ℝ) : g z = 
  if z ≤ 0 then 0
  else if 0 < z ∧ z ≤ 2 then z / 4
  else if 2 < z ∧ z ≤ 4 then 1 - z / 4
  else 0 := sorry

end NUMINAMATH_GPT_cdf_of_Z_pdf_of_Z_l1358_135875


namespace NUMINAMATH_GPT_sum_of_cubes_decomposition_l1358_135891

theorem sum_of_cubes_decomposition :
  ∃ a b c d e : ℤ, (∀ x : ℤ, 1728 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) ∧ (a + b + c + d + e = 132) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_decomposition_l1358_135891


namespace NUMINAMATH_GPT_longest_pencil_l1358_135827

/-- Hallway dimensions and the longest pencil problem -/
theorem longest_pencil (L : ℝ) : 
    (∃ P : ℝ, P = 3 * L) :=
sorry

end NUMINAMATH_GPT_longest_pencil_l1358_135827


namespace NUMINAMATH_GPT_deepak_present_age_l1358_135802

-- We start with the conditions translated into Lean definitions.

variables (R D : ℕ)

-- Condition 1: The ratio between Rahul's and Deepak's ages is 4:3.
def age_ratio := R * 3 = D * 4

-- Condition 2: After 6 years, Rahul's age will be 38 years.
def rahul_future_age := R + 6 = 38

-- The goal is to prove that D = 24 given the above conditions.
theorem deepak_present_age 
  (h1: age_ratio R D) 
  (h2: rahul_future_age R) : D = 24 :=
sorry

end NUMINAMATH_GPT_deepak_present_age_l1358_135802


namespace NUMINAMATH_GPT_part_I_part_II_part_III_l1358_135893

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.log x

theorem part_I (a : ℝ) : (∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f a x ≥ f a 1) ↔ a ≥ -1/2 :=
by
  sorry

theorem part_II : ∀ x : ℝ, f (-Real.exp 1) x + 2 ≤ 0 :=
by
  sorry

theorem part_III : ¬ ∃ x : ℝ, |f (-Real.exp 1) x| = Real.log x / x + 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_part_III_l1358_135893


namespace NUMINAMATH_GPT_liquid_X_percent_in_mixed_solution_l1358_135836

theorem liquid_X_percent_in_mixed_solution (wP wQ : ℝ) (xP xQ : ℝ) (mP mQ : ℝ) :
  xP = 0.005 * wP →
  xQ = 0.015 * wQ →
  wP = 200 →
  wQ = 800 →
  13 / 1000 * 100 = 1.3 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_liquid_X_percent_in_mixed_solution_l1358_135836


namespace NUMINAMATH_GPT_a5_a6_less_than_a4_squared_l1358_135880

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem a5_a6_less_than_a4_squared
  (h_geo : is_geometric_sequence a q)
  (h_cond : a 5 * a 6 < (a 4) ^ 2) :
  0 < q ∧ q < 1 :=
sorry

end NUMINAMATH_GPT_a5_a6_less_than_a4_squared_l1358_135880


namespace NUMINAMATH_GPT_lisa_interest_correct_l1358_135879

noncomputable def lisa_interest : ℝ :=
  let P := 2000
  let r := 0.035
  let n := 10
  let A := P * (1 + r) ^ n
  A - P

theorem lisa_interest_correct :
  lisa_interest = 821 := by
  sorry

end NUMINAMATH_GPT_lisa_interest_correct_l1358_135879


namespace NUMINAMATH_GPT_zero_in_interval_l1358_135874

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

theorem zero_in_interval : 
    (∀ x y : ℝ, 0 < x → x < y → f x < f y) → 
    (f 1 = -2) →
    (f 2 = Real.log 2 + 5) →
    (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by 
    sorry

end NUMINAMATH_GPT_zero_in_interval_l1358_135874


namespace NUMINAMATH_GPT_exists_m_for_n_divides_2_pow_m_plus_m_l1358_135801

theorem exists_m_for_n_divides_2_pow_m_plus_m (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, 0 < m ∧ n ∣ 2^m + m :=
sorry

end NUMINAMATH_GPT_exists_m_for_n_divides_2_pow_m_plus_m_l1358_135801


namespace NUMINAMATH_GPT_total_cost_l1358_135877

def num_professionals := 2
def hours_per_professional_per_day := 6
def days_worked := 7
def hourly_rate := 15

theorem total_cost : 
  (num_professionals * hours_per_professional_per_day * days_worked * hourly_rate) = 1260 := by
  sorry

end NUMINAMATH_GPT_total_cost_l1358_135877


namespace NUMINAMATH_GPT_probability_of_selecting_particular_girl_l1358_135845

-- Define the numbers involved
def total_population : ℕ := 60
def num_girls : ℕ := 25
def num_boys : ℕ := 35
def sample_size : ℕ := 5

-- Total number of basic events
def total_combinations : ℕ := Nat.choose total_population sample_size

-- Number of basic events that include a particular girl
def girl_combinations : ℕ := Nat.choose (total_population - 1) (sample_size - 1)

-- Probability of selecting a particular girl
def probability_of_girl_selection : ℚ := girl_combinations / total_combinations

-- The theorem to be proved
theorem probability_of_selecting_particular_girl :
  probability_of_girl_selection = 1 / 12 :=
by sorry

end NUMINAMATH_GPT_probability_of_selecting_particular_girl_l1358_135845


namespace NUMINAMATH_GPT_sequence_sum_zero_l1358_135887

-- Define the sequence as a function
def seq (n : ℕ) : ℤ :=
  if (n-1) % 8 < 4
  then (n+1) / 2
  else - (n / 2)

-- Define the sum of the sequence up to a given number
def seq_sum (m : ℕ) : ℤ :=
  (Finset.range (m+1)).sum (λ n => seq n)

-- The actual problem statement
theorem sequence_sum_zero : seq_sum 2012 = 0 :=
  sorry

end NUMINAMATH_GPT_sequence_sum_zero_l1358_135887


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1358_135867

theorem quadratic_inequality_solution:
  ∀ x : ℝ, -x^2 + 3 * x - 2 ≥ 0 ↔ (1 ≤ x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1358_135867


namespace NUMINAMATH_GPT_halfway_fraction_l1358_135873

theorem halfway_fraction (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  ((a / b + c / d) / 2) = 19 / 24 :=
by
  rw [h₁, h₂, h₃, h₄]
  norm_num

end NUMINAMATH_GPT_halfway_fraction_l1358_135873


namespace NUMINAMATH_GPT_geometric_sequence_sum_inequality_l1358_135851

open Classical

variable (a_1 q : ℝ) (h1 : a_1 > 0) (h2 : q > 0) (h3 : q ≠ 1)

theorem geometric_sequence_sum_inequality :
  a_1 + a_1 * q^3 > a_1 * q + a_1 * q^2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_inequality_l1358_135851


namespace NUMINAMATH_GPT_exists_multiple_with_sum_divisible_l1358_135858

-- Define the sum of the digits
def sum_of_digits (n : ℕ) : ℕ := -- Implementation of sum_of_digits function is omitted here
sorry

-- Main theorem statement
theorem exists_multiple_with_sum_divisible (n : ℕ) (hn : n > 0) : 
  ∃ k, k % n = 0 ∧ sum_of_digits k ∣ k :=
sorry

end NUMINAMATH_GPT_exists_multiple_with_sum_divisible_l1358_135858


namespace NUMINAMATH_GPT_BKING_2023_reappears_at_20_l1358_135838

-- Defining the basic conditions of the problem
def cycle_length_BKING : ℕ := 5
def cycle_length_2023 : ℕ := 4

-- Formulating the proof problem statement
theorem BKING_2023_reappears_at_20 :
  Nat.lcm cycle_length_BKING cycle_length_2023 = 20 :=
by
  sorry

end NUMINAMATH_GPT_BKING_2023_reappears_at_20_l1358_135838


namespace NUMINAMATH_GPT_minimum_stamps_satisfying_congruences_l1358_135820

theorem minimum_stamps_satisfying_congruences (n : ℕ) :
  (n % 4 = 3) ∧ (n % 5 = 2) ∧ (n % 7 = 1) → n = 107 :=
by
  sorry

end NUMINAMATH_GPT_minimum_stamps_satisfying_congruences_l1358_135820


namespace NUMINAMATH_GPT_tetrahedron_edge_length_l1358_135807

-- Define the problem specifications
def mutuallyTangent (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop) :=
  a = b ∧ a = c ∧ a = d ∧ b = c ∧ b = d ∧ c = d

noncomputable def tetrahedronEdgeLength (r : ℝ) : ℝ :=
  2 + 2 * Real.sqrt 6

-- Proof goal: edge length of tetrahedron containing four mutually tangent balls each of radius 1
theorem tetrahedron_edge_length (r : ℝ) (a b c d : ℝ → ℝ → ℝ → Prop)
  (h1 : r = 1)
  (h2 : mutuallyTangent r a b c d)
  : tetrahedronEdgeLength r = 2 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_tetrahedron_edge_length_l1358_135807


namespace NUMINAMATH_GPT_correct_statement_dice_roll_l1358_135846

theorem correct_statement_dice_roll :
  (∃! s, s ∈ ["When flipping a coin, the head side will definitely face up.",
              "The probability of precipitation tomorrow is 80% means that 80% of the areas will have rain tomorrow.",
              "To understand the lifespan of a type of light bulb, it is appropriate to use a census method.",
              "When rolling a dice, the number will definitely not be greater than 6."] ∧
          s = "When rolling a dice, the number will definitely not be greater than 6.") :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_statement_dice_roll_l1358_135846


namespace NUMINAMATH_GPT_find_y_l1358_135823

theorem find_y
  (x y : ℝ)
  (h1 : x - y = 10)
  (h2 : x + y = 8) : y = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1358_135823


namespace NUMINAMATH_GPT_find_angle_C_l1358_135821

noncomputable def ABC_triangle (A B C a b c : ℝ) : Prop :=
b = c * Real.cos A + Real.sqrt 3 * a * Real.sin C

theorem find_angle_C (A B C a b c : ℝ) (h : ABC_triangle A B C a b c) :
  C = π / 6 :=
sorry

end NUMINAMATH_GPT_find_angle_C_l1358_135821


namespace NUMINAMATH_GPT_smallest_square_area_l1358_135860

theorem smallest_square_area :
  (∀ (x y : ℝ), (∃ (x1 x2 y1 y2 : ℝ), y1 = 3 * x1 - 4 ∧ y2 = 3 * x2 - 4 ∧ y = x^2 + 5 ∧ 
  ∀ (k : ℝ), x1 + x2 = 3 ∧ x1 * x2 = 5 - k ∧ 16 * k^2 - 332 * k + 396 = 0 ∧ 
  ((k = 1.5 ∧ 10 * (4 * k - 11) = 50) ∨ 
  (k = 16.5 ∧ 10 * (4 * k - 11) ≠ 50))) → 
  ∃ (A: Real), A = 50) :=
sorry

end NUMINAMATH_GPT_smallest_square_area_l1358_135860


namespace NUMINAMATH_GPT_least_positive_integer_l1358_135855

theorem least_positive_integer (n : ℕ) :
  (∃ n : ℕ, 25^n + 16^n ≡ 1 [MOD 121] ∧ ∀ m : ℕ, (m < n ∧ 25^m + 16^m ≡ 1 [MOD 121]) → false) ↔ n = 32 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_l1358_135855


namespace NUMINAMATH_GPT_work_completion_time_l1358_135803

/-- q can complete the work in 9 days, r can complete the work in 12 days, they work together
for 3 days, and p completes the remaining work in 10.000000000000002 days. Prove that
p alone can complete the work in approximately 24 days. -/
theorem work_completion_time (W : ℝ) (q : ℝ) (r : ℝ) (p : ℝ) :
  q = 9 → r = 12 → (p * 10.000000000000002 = (5 / 12) * W) →
  p = 24.000000000000004 :=
by 
  intros hq hr hp
  sorry

end NUMINAMATH_GPT_work_completion_time_l1358_135803


namespace NUMINAMATH_GPT_percentage_of_boys_is_60_percent_l1358_135800

-- Definition of the problem conditions
def totalPlayers := 50
def juniorGirls := 10
def half (n : ℕ) := n / 2
def girls := 2 * juniorGirls
def boys := totalPlayers - girls
def percentage_of_boys := (boys * 100) / totalPlayers

-- The theorem stating the proof problem
theorem percentage_of_boys_is_60_percent : percentage_of_boys = 60 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_percentage_of_boys_is_60_percent_l1358_135800


namespace NUMINAMATH_GPT_sqrt_mult_simplify_l1358_135898

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end NUMINAMATH_GPT_sqrt_mult_simplify_l1358_135898


namespace NUMINAMATH_GPT_isosceles_triangle_area_l1358_135816

theorem isosceles_triangle_area (PQ PR QR : ℝ) (PS : ℝ) (h1 : PQ = PR)
  (h2 : QR = 10) (h3 : PS^2 + (QR / 2)^2 = PQ^2) : 
  (1/2) * QR * PS = 60 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l1358_135816


namespace NUMINAMATH_GPT_sqrt_88200_simplified_l1358_135857

theorem sqrt_88200_simplified : Real.sqrt 88200 = 210 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_sqrt_88200_simplified_l1358_135857


namespace NUMINAMATH_GPT_greatest_x_solution_l1358_135844

theorem greatest_x_solution (x : ℝ) (h₁ : (x^2 - x - 30) / (x - 6) = 2 / (x + 4)) : x ≤ -3 :=
sorry

end NUMINAMATH_GPT_greatest_x_solution_l1358_135844


namespace NUMINAMATH_GPT_solve_chimney_bricks_l1358_135819

noncomputable def chimney_bricks (x : ℝ) : Prop :=
  let brenda_rate := x / 8
  let brandon_rate := x / 12
  let combined_rate := brenda_rate + brandon_rate - 15
  (combined_rate * 6) = x

theorem solve_chimney_bricks : ∃ (x : ℝ), chimney_bricks x ∧ x = 360 :=
by
  use 360
  unfold chimney_bricks
  sorry

end NUMINAMATH_GPT_solve_chimney_bricks_l1358_135819


namespace NUMINAMATH_GPT_minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l1358_135896

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem minimum_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

theorem decreasing_intervals_of_f : ∀ k : ℤ, ∀ x : ℝ,
  (Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ (2 * Real.pi / 3 + k * Real.pi) → ∀ y : ℝ, 
  (Real.pi / 6 + k * Real.pi) ≤ y ∧ y ≤ (2 * Real.pi / 3 + k * Real.pi) → x ≤ y → f y ≤ f x := by sorry

theorem maximum_value_of_f : ∃ k : ℤ, ∃ x : ℝ, x = (Real.pi / 6 + k * Real.pi) ∧ f x = 5 / 2 := by sorry

end NUMINAMATH_GPT_minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l1358_135896


namespace NUMINAMATH_GPT_ratio_of_areas_l1358_135826

theorem ratio_of_areas (OR : ℝ) (h : OR > 0) :
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  (area_OY / area_OR) = (1 / 9) :=
by
  -- Definitions
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1358_135826


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1358_135897

theorem solve_quadratic_eq (x : ℝ) : x^2 + 2 * x - 1 = 0 ↔ (x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1358_135897


namespace NUMINAMATH_GPT_candy_bar_sugar_calories_l1358_135841

theorem candy_bar_sugar_calories
  (candy_bars : Nat)
  (soft_drink_calories : Nat)
  (soft_drink_sugar_percentage : Float)
  (recommended_sugar_intake : Nat)
  (excess_percentage : Nat)
  (sugar_in_each_bar : Nat) :
  candy_bars = 7 ∧
  soft_drink_calories = 2500 ∧
  soft_drink_sugar_percentage = 0.05 ∧
  recommended_sugar_intake = 150 ∧
  excess_percentage = 100 →
  sugar_in_each_bar = 25 := by
  sorry

end NUMINAMATH_GPT_candy_bar_sugar_calories_l1358_135841


namespace NUMINAMATH_GPT_find_x_l1358_135883

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 119) : x = 39 :=
sorry

end NUMINAMATH_GPT_find_x_l1358_135883


namespace NUMINAMATH_GPT_div_fraction_eq_l1358_135863

theorem div_fraction_eq :
  (5 / 3) / (1 / 4) = 20 / 3 := 
by
  sorry

end NUMINAMATH_GPT_div_fraction_eq_l1358_135863


namespace NUMINAMATH_GPT_inequality_solution_set_l1358_135894

theorem inequality_solution_set :
  (∀ x : ℝ, (3 * x - 2 < 2 * (x + 1) ∧ (x - 1) / 2 > 1) ↔ (3 < x ∧ x < 4)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1358_135894


namespace NUMINAMATH_GPT_find_alpha_l1358_135840

theorem find_alpha (α : ℝ) (h0 : 0 ≤ α) (h1 : α < 360)
    (h_point : (Real.sin 215) = (Real.sin α) ∧ (Real.cos 215) = (Real.cos α)) :
    α = 235 :=
sorry

end NUMINAMATH_GPT_find_alpha_l1358_135840


namespace NUMINAMATH_GPT_marble_weight_l1358_135871

theorem marble_weight (W : ℝ) (h : 2 * W + 0.08333333333333333 = 0.75) : 
  W = 0.33333333333333335 := 
by 
  -- Skipping the proof as specified
  sorry

end NUMINAMATH_GPT_marble_weight_l1358_135871


namespace NUMINAMATH_GPT_tan_angle_addition_l1358_135868

theorem tan_angle_addition (y : ℝ) (hyp : Real.tan y = -3) : 
  Real.tan (y + Real.pi / 3) = - (5 * Real.sqrt 3 - 6) / 13 := 
by 
  sorry

end NUMINAMATH_GPT_tan_angle_addition_l1358_135868


namespace NUMINAMATH_GPT_multiples_of_three_l1358_135813

theorem multiples_of_three (a b : ℤ) (h : 9 ∣ (a^2 + a * b + b^2)) : 3 ∣ a ∧ 3 ∣ b :=
by {
  sorry
}

end NUMINAMATH_GPT_multiples_of_three_l1358_135813


namespace NUMINAMATH_GPT_compare_y_values_l1358_135811

noncomputable def parabola (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 1

theorem compare_y_values :
  ∃ y1 y2 y3, (parabola (-3) = y1) ∧ (parabola (-2) = y2) ∧ (parabola 2 = y3) ∧ (y3 < y1) ∧ (y1 < y2) :=
by
  sorry

end NUMINAMATH_GPT_compare_y_values_l1358_135811


namespace NUMINAMATH_GPT_find_m_l1358_135869

open Classical

variable {d : ℤ} (h₁ : d ≠ 0) (a : ℕ → ℤ)

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∃ a₀ : ℤ, ∀ n, a n = a₀ + n * d

theorem find_m 
  (h_seq : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_am : ∃ m, a m = 8) :
  ∃ m, m = 8 :=
sorry

end NUMINAMATH_GPT_find_m_l1358_135869


namespace NUMINAMATH_GPT_pencils_per_student_l1358_135825

theorem pencils_per_student
  (boxes : ℝ) (pencils_per_box : ℝ) (students : ℝ)
  (h1 : boxes = 4.0)
  (h2 : pencils_per_box = 648.0)
  (h3 : students = 36.0) :
  (boxes * pencils_per_box) / students = 72.0 :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_student_l1358_135825


namespace NUMINAMATH_GPT_vector_expression_identity_l1358_135808

variables (E : Type) [AddCommGroup E] [Module ℝ E]
variables (e1 e2 : E)
variables (a b : E)
variables (cond1 : a = (3 : ℝ) • e1 - (2 : ℝ) • e2) (cond2 : b = (e2 - (2 : ℝ) • e1))

theorem vector_expression_identity :
  (1 / 3 : ℝ) • a + b + a - (3 / 2 : ℝ) • b + 2 • b - a = -2 • e1 + (5 / 6 : ℝ) • e2 :=
sorry

end NUMINAMATH_GPT_vector_expression_identity_l1358_135808


namespace NUMINAMATH_GPT_ratio_x_to_y_is_12_l1358_135886

noncomputable def ratio_x_y (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ℝ := x / y

theorem ratio_x_to_y_is_12 (x y : ℝ) (h1 : y = x * (1 - 0.9166666666666666)) : ratio_x_y x y h1 = 12 :=
sorry

end NUMINAMATH_GPT_ratio_x_to_y_is_12_l1358_135886


namespace NUMINAMATH_GPT_square_difference_l1358_135889

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_square_difference_l1358_135889


namespace NUMINAMATH_GPT_minimum_pencils_l1358_135850

-- Define the given conditions
def red_pencils : ℕ := 15
def blue_pencils : ℕ := 13
def green_pencils : ℕ := 8

-- Define the requirement for pencils to ensure the conditions are met
def required_red : ℕ := 1
def required_blue : ℕ := 2
def required_green : ℕ := 3

-- The minimum number of pencils Constanza should take out
noncomputable def minimum_pencils_to_ensure : ℕ := 21 + 1

theorem minimum_pencils (red_pencils blue_pencils green_pencils : ℕ)
    (required_red required_blue required_green minimum_pencils_to_ensure : ℕ) :
    red_pencils = 15 →
    blue_pencils = 13 →
    green_pencils = 8 →
    required_red = 1 →
    required_blue = 2 →
    required_green = 3 →
    minimum_pencils_to_ensure = 22 :=
by
    intros h1 h2 h3 h4 h5 h6
    sorry

end NUMINAMATH_GPT_minimum_pencils_l1358_135850


namespace NUMINAMATH_GPT_mary_final_weight_l1358_135899

theorem mary_final_weight : 
  let initial_weight := 99
  let weight_loss1 := 12
  let weight_gain1 := 2 * weight_loss1
  let weight_loss2 := 3 * weight_loss1
  let weight_gain2 := 6
  initial_weight - weight_loss1 + weight_gain1 - weight_loss2 + weight_gain2 = 81 := by 
  sorry

end NUMINAMATH_GPT_mary_final_weight_l1358_135899


namespace NUMINAMATH_GPT_difference_of_triangular_23_and_21_l1358_135842

def triangular (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem difference_of_triangular_23_and_21 : triangular 23 - triangular 21 = 45 :=
sorry

end NUMINAMATH_GPT_difference_of_triangular_23_and_21_l1358_135842


namespace NUMINAMATH_GPT_values_of_a_and_b_l1358_135822

theorem values_of_a_and_b (a b : ℝ) 
  (hT : (2, 1) ∈ {p : ℝ × ℝ | ∃ (a : ℝ), p.1 * a + p.2 - 3 = 0})
  (hS : (2, 1) ∈ {p : ℝ × ℝ | ∃ (b : ℝ), p.1 - p.2 - b = 0}) :
  a = 1 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_values_of_a_and_b_l1358_135822


namespace NUMINAMATH_GPT_area_ratio_eq_two_l1358_135892

/-- 
  Given a unit square, let circle B be the inscribed circle and circle A be the circumscribed circle.
  Prove the ratio of the area of circle A to the area of circle B is 2.
--/
theorem area_ratio_eq_two (r_B r_A : ℝ) (hB : r_B = 1 / 2) (hA : r_A = Real.sqrt 2 / 2):
  (π * r_A ^ 2) / (π * r_B ^ 2) = 2 := by
  sorry

end NUMINAMATH_GPT_area_ratio_eq_two_l1358_135892


namespace NUMINAMATH_GPT_evens_in_triangle_l1358_135834

theorem evens_in_triangle (a : ℕ → ℕ → ℕ) (h : ∀ i j, a i.succ j = (a i (j - 1) + a i j + a i (j + 1)) % 2) :
  ∀ n ≥ 2, ∃ j, a n j % 2 = 0 :=
  sorry

end NUMINAMATH_GPT_evens_in_triangle_l1358_135834


namespace NUMINAMATH_GPT_oil_price_reduction_l1358_135882

theorem oil_price_reduction (P P_reduced : ℝ) (h1 : P_reduced = 50) (h2 : 1000 / P_reduced - 5 = 5) :
  ((P - P_reduced) / P) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_oil_price_reduction_l1358_135882


namespace NUMINAMATH_GPT_example_inequality_l1358_135890

variable (a b c : ℝ)

theorem example_inequality 
  (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_example_inequality_l1358_135890


namespace NUMINAMATH_GPT_janice_weekly_earnings_l1358_135852

-- define the conditions
def regular_days_per_week : Nat := 5
def regular_earnings_per_day : Nat := 30
def overtime_earnings_per_shift : Nat := 15
def overtime_shifts_per_week : Nat := 3

-- define the total earnings calculation
def total_earnings (regular_days : Nat) (regular_rate : Nat) (overtime_shifts : Nat) (overtime_rate : Nat) : Nat :=
  (regular_days * regular_rate) + (overtime_shifts * overtime_rate)

-- state the problem to be proved
theorem janice_weekly_earnings : total_earnings regular_days_per_week regular_earnings_per_day overtime_shifts_per_week overtime_earnings_per_shift = 195 :=
by
  sorry

end NUMINAMATH_GPT_janice_weekly_earnings_l1358_135852


namespace NUMINAMATH_GPT_range_of_f_l1358_135847

def f (x : ℤ) : ℤ := x ^ 2 - 2 * x
def domain : Set ℤ := {0, 1, 2, 3}
def expectedRange : Set ℤ := {-1, 0, 3}

theorem range_of_f : (Set.image f domain) = expectedRange :=
  sorry

end NUMINAMATH_GPT_range_of_f_l1358_135847


namespace NUMINAMATH_GPT_bike_price_l1358_135895

variable (p : ℝ)

def percent_upfront_payment : ℝ := 0.20
def upfront_payment : ℝ := 200

theorem bike_price (h : percent_upfront_payment * p = upfront_payment) : p = 1000 := by
  sorry

end NUMINAMATH_GPT_bike_price_l1358_135895


namespace NUMINAMATH_GPT_boxes_amount_l1358_135859

/-- 
  A food company has 777 kilograms of food to put into boxes. 
  If each box gets a certain amount of kilograms, they will have 388 full boxes.
  Prove that each box gets 2 kilograms of food.
-/
theorem boxes_amount (total_food : ℕ) (boxes : ℕ) (kilograms_per_box : ℕ) 
  (h_total : total_food = 777)
  (h_boxes : boxes = 388) :
  total_food / boxes = kilograms_per_box :=
by {
  -- Skipped proof
  sorry 
}

end NUMINAMATH_GPT_boxes_amount_l1358_135859


namespace NUMINAMATH_GPT_simplify_expression_l1358_135817

theorem simplify_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - a^2) / (b^2 - a^2) =
  (a^3 - 3 * a * b^2 + 2 * b^3) / (a * b * (b + a)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1358_135817


namespace NUMINAMATH_GPT_find_y_l1358_135805

variable (x y : ℤ)

-- Conditions
def cond1 : Prop := x + y = 280
def cond2 : Prop := x - y = 200

-- Proof statement
theorem find_y (h1 : cond1 x y) (h2 : cond2 x y) : y = 40 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l1358_135805


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l1358_135830

theorem perfect_square_trinomial_k (a k : ℝ) : (∃ b : ℝ, (a - b)^2 = a^2 - ka + 25) ↔ k = 10 ∨ k = -10 := 
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l1358_135830


namespace NUMINAMATH_GPT_squares_in_50th_ring_l1358_135870

-- Define the problem using the given conditions
def centered_square_3x3 : ℕ := 3 -- Represent the 3x3 centered square

-- Define the function that computes the number of unit squares in the nth ring
def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  if n = 1 then 16
  else 24 + 8 * (n - 2)

-- Define the accumulation of unit squares up to the 50th ring
def total_squares_in_50th_ring : ℕ :=
  33 + 24 * 49

theorem squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1209 :=
by
  -- Ensure that the correct value for the 50th ring can be verified
  sorry

end NUMINAMATH_GPT_squares_in_50th_ring_l1358_135870


namespace NUMINAMATH_GPT_isosceles_right_triangle_leg_length_l1358_135864

theorem isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ) 
  (h1 : m = 12) 
  (h2 : m = h / 2)
  (h3 : h = x * Real.sqrt 2) :
  x = 12 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_leg_length_l1358_135864


namespace NUMINAMATH_GPT_cos_squared_identity_l1358_135804

theorem cos_squared_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * Real.cos (π / 6 + α / 2) ^ 2 + 1 = 7 / 3 := 
by
    sorry

end NUMINAMATH_GPT_cos_squared_identity_l1358_135804


namespace NUMINAMATH_GPT_largest_satisfying_n_correct_l1358_135843
noncomputable def largest_satisfying_n : ℕ := 4

theorem largest_satisfying_n_correct :
  ∀ n x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5) 
  → n = largest_satisfying_n ∧
  ¬ (∃ x, (1 < x ∧ x < 2 ∧ 2 < x^2 ∧ x^2 < 3 ∧ 3 < x^3 ∧ x^3 < 4 ∧ 4 < x^4 ∧ x^4 < 5 ∧ 5 < x^5 ∧ x^5 < 6)) := sorry

end NUMINAMATH_GPT_largest_satisfying_n_correct_l1358_135843


namespace NUMINAMATH_GPT_pond_capacity_l1358_135837

theorem pond_capacity :
  let normal_rate := 6 -- gallons per minute
  let restriction_rate := (2/3 : ℝ) * normal_rate -- gallons per minute
  let time := 50 -- minutes
  let capacity := restriction_rate * time -- total capacity in gallons
  capacity = 200 := sorry

end NUMINAMATH_GPT_pond_capacity_l1358_135837


namespace NUMINAMATH_GPT_circle_diameter_C_l1358_135854

theorem circle_diameter_C {D C : ℝ} (hD : D = 20) (h_ratio : (π * (D/2)^2 - π * (C/2)^2) / (π * (C/2)^2) = 4) : C = 4 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_circle_diameter_C_l1358_135854


namespace NUMINAMATH_GPT_age_solution_l1358_135818

noncomputable def age_problem : Prop :=
  ∃ (A B x : ℕ),
    A = B + 5 ∧
    A + B = 13 ∧
    3 * (A + x) = 4 * (B + x) ∧
    x = 11

theorem age_solution : age_problem :=
  sorry

end NUMINAMATH_GPT_age_solution_l1358_135818


namespace NUMINAMATH_GPT_base5_representation_three_consecutive_digits_l1358_135833

theorem base5_representation_three_consecutive_digits :
  ∃ (digits : ℕ), 
    (digits = 3) ∧ 
    (∃ (a1 a2 a3 : ℕ), 
      94 = a1 * 5^2 + a2 * 5^1 + a3 * 5^0 ∧
      a1 = 3 ∧ a2 = 3 ∧ a3 = 4 ∧
      (a1 = a3 + 1) ∧ (a2 = a3 + 2)) := 
    sorry

end NUMINAMATH_GPT_base5_representation_three_consecutive_digits_l1358_135833


namespace NUMINAMATH_GPT_range_of_a_l1358_135824

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
  (∀ x, |x^3 - a * x^2| = x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) →
  a > 2 :=
by
  -- The proof is to be provided here.
  sorry

end NUMINAMATH_GPT_range_of_a_l1358_135824


namespace NUMINAMATH_GPT_triangle_side_BC_length_l1358_135865

noncomputable def triangle_side_length
  (AB : ℝ) (angle_a : ℝ) (angle_c : ℝ) : ℝ := 
  let sin_a := Real.sin angle_a
  let sin_c := Real.sin angle_c
  (AB * sin_a) / sin_c

theorem triangle_side_BC_length (AB : ℝ) (angle_a angle_c : ℝ) :
  AB = (Real.sqrt 6) / 2 →
  angle_a = (45 * Real.pi / 180) →
  angle_c = (60 * Real.pi / 180) →
  triangle_side_length AB angle_a angle_c = 1 :=
sorry

end NUMINAMATH_GPT_triangle_side_BC_length_l1358_135865


namespace NUMINAMATH_GPT_find_x_plus_y_l1358_135832

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y = 2009) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 + Real.pi / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1358_135832


namespace NUMINAMATH_GPT_hundred_div_point_two_five_eq_four_hundred_l1358_135884

theorem hundred_div_point_two_five_eq_four_hundred : 100 / 0.25 = 400 := by
  sorry

end NUMINAMATH_GPT_hundred_div_point_two_five_eq_four_hundred_l1358_135884


namespace NUMINAMATH_GPT_total_distance_correct_l1358_135829

def liters_U := 50
def liters_V := 50
def liters_W := 50
def liters_X := 50

def fuel_efficiency_U := 20 -- liters per 100 km
def fuel_efficiency_V := 25 -- liters per 100 km
def fuel_efficiency_W := 5 -- liters per 100 km
def fuel_efficiency_X := 10 -- liters per 100 km

def distance_U := (liters_U / fuel_efficiency_U) * 100 -- Distance for U in km
def distance_V := (liters_V / fuel_efficiency_V) * 100 -- Distance for V in km
def distance_W := (liters_W / fuel_efficiency_W) * 100 -- Distance for W in km
def distance_X := (liters_X / fuel_efficiency_X) * 100 -- Distance for X in km

def total_distance := distance_U + distance_V + distance_W + distance_X -- Total distance of all cars

theorem total_distance_correct :
  total_distance = 1950 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_distance_correct_l1358_135829


namespace NUMINAMATH_GPT_annie_budget_l1358_135885

theorem annie_budget :
  let budget := 120
  let hamburger_count := 8
  let milkshake_count := 6
  let hamburgerA := 4
  let milkshakeA := 5
  let hamburgerB := 3.5
  let milkshakeB := 6
  let hamburgerC := 5
  let milkshakeC := 4
  let costA := hamburgerA * hamburger_count + milkshakeA * milkshake_count
  let costB := hamburgerB * hamburger_count + milkshakeB * milkshake_count
  let costC := hamburgerC * hamburger_count + milkshakeC * milkshake_count
  let min_cost := min costA (min costB costC)
  budget - min_cost = 58 :=
by {
  sorry
}

end NUMINAMATH_GPT_annie_budget_l1358_135885


namespace NUMINAMATH_GPT_JakePresentWeight_l1358_135881

def JakeWeight (J S : ℕ) : Prop :=
  J - 33 = 2 * S ∧ J + S = 153

theorem JakePresentWeight : ∃ (J : ℕ), ∃ (S : ℕ), JakeWeight J S ∧ J = 113 := 
by
  sorry

end NUMINAMATH_GPT_JakePresentWeight_l1358_135881


namespace NUMINAMATH_GPT_cauchy_schwarz_inequality_l1358_135853

theorem cauchy_schwarz_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_cauchy_schwarz_inequality_l1358_135853


namespace NUMINAMATH_GPT_minimum_value_of_f_is_15_l1358_135856

noncomputable def f (x : ℝ) : ℝ := 9 * x + (1 / (x - 1))

theorem minimum_value_of_f_is_15 (h : ∀ x, x > 1) : ∃ x, x > 1 ∧ f x = 15 :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_f_is_15_l1358_135856


namespace NUMINAMATH_GPT_probability_C_D_l1358_135831

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end NUMINAMATH_GPT_probability_C_D_l1358_135831
