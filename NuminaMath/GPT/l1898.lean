import Mathlib

namespace sheepdog_speed_l1898_189802

theorem sheepdog_speed 
  (T : ℝ) (t : ℝ) (sheep_speed : ℝ) (initial_distance : ℝ)
  (total_distance_speed : ℝ) :
  T = 20  →
  t = 20 →
  sheep_speed = 12 →
  initial_distance = 160 →
  total_distance_speed = 20 →
  total_distance_speed * T = initial_distance + sheep_speed * t := 
by sorry

end sheepdog_speed_l1898_189802


namespace tan_square_proof_l1898_189805

theorem tan_square_proof (θ : ℝ) (h : Real.tan θ = 2) : 
  1 / (Real.sin θ ^ 2 - Real.cos θ ^ 2) = 5 / 3 := by
  sorry

end tan_square_proof_l1898_189805


namespace smallest_n_l1898_189829

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : 15 < n) : n = 52 :=
by
  sorry

end smallest_n_l1898_189829


namespace total_cookies_l1898_189867

def total_chocolate_chip_batches := 5
def cookies_per_chocolate_chip_batch := 8
def total_oatmeal_batches := 3
def cookies_per_oatmeal_batch := 7
def total_sugar_batches := 1
def cookies_per_sugar_batch := 10
def total_double_chocolate_batches := 1
def cookies_per_double_chocolate_batch := 6

theorem total_cookies : 
  (total_chocolate_chip_batches * cookies_per_chocolate_chip_batch) +
  (total_oatmeal_batches * cookies_per_oatmeal_batch) +
  (total_sugar_batches * cookies_per_sugar_batch) +
  (total_double_chocolate_batches * cookies_per_double_chocolate_batch) = 77 :=
by sorry

end total_cookies_l1898_189867


namespace range_of_b_l1898_189847

theorem range_of_b (a b : ℝ) : 
  (∀ x : ℝ, -3 < x ∧ x < 1 → (1 - a) * x^2 - 4 * x + 6 > 0) ∧
  (∀ x : ℝ, 3 * x^2 + b * x + 3 ≥ 0) →
  (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end range_of_b_l1898_189847


namespace length_of_ac_l1898_189893

theorem length_of_ac (a b c d e : ℝ) (ab bc cd de ae ac : ℝ)
  (h1 : ab = 5)
  (h2 : bc = 2 * cd)
  (h3 : de = 8)
  (h4 : ae = 22)
  (h5 : ae = ab + bc + cd + de)
  (h6 : ac = ab + bc) :
  ac = 11 := by
  sorry

end length_of_ac_l1898_189893


namespace employed_males_percentage_l1898_189882

theorem employed_males_percentage (p_employed : ℝ) (p_employed_females : ℝ) : 
  (64 / 100) * (1 - 21.875 / 100) * 100 = 49.96 :=
by
  sorry

end employed_males_percentage_l1898_189882


namespace internal_diagonal_cubes_l1898_189845

theorem internal_diagonal_cubes :
  let A := (120, 360, 400)
  let gcd_xy := gcd 120 360
  let gcd_yz := gcd 360 400
  let gcd_zx := gcd 400 120
  let gcd_xyz := gcd (gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz
  new_cubes = 720 :=
by
  -- Definitions
  let A := (120, 360, 400)
  let gcd_xy := Int.gcd 120 360
  let gcd_yz := Int.gcd 360 400
  let gcd_zx := Int.gcd 400 120
  let gcd_xyz := Int.gcd (Int.gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz

  -- Assertion
  exact Eq.refl new_cubes

end internal_diagonal_cubes_l1898_189845


namespace find_prime_triplets_l1898_189840

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_triplet (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ (p * (r + 1) = q * (r + 5))

theorem find_prime_triplets :
  { (p, q, r) | valid_triplet p q r } = {(3, 2, 7), (5, 3, 5), (7, 3, 2)} :=
by {
  sorry -- Proof is to be completed
}

end find_prime_triplets_l1898_189840


namespace find_a1_a7_l1898_189850

-- Definitions based on the problem conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def a_3_5_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 = -6

def a_2_6_condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 6 = 8

-- The theorem we need to prove
theorem find_a1_a7 (a : ℕ → ℝ) (ha : is_geometric_sequence a) (h35 : a_3_5_condition a) (h26 : a_2_6_condition a) :
  a 1 + a 7 = -9 :=
sorry

end find_a1_a7_l1898_189850


namespace jamies_shoes_cost_l1898_189830

-- Define the costs of items and the total cost.
def cost_total : ℤ := 110
def cost_coat : ℤ := 40
def cost_one_pair_jeans : ℤ := 20

-- Define the number of pairs of jeans.
def num_pairs_jeans : ℕ := 2

-- Define the cost of Jamie's shoes (to be proved).
def cost_jamies_shoes : ℤ := cost_total - (cost_coat + num_pairs_jeans * cost_one_pair_jeans)

theorem jamies_shoes_cost : cost_jamies_shoes = 30 :=
by
  -- Insert proof here
  sorry

end jamies_shoes_cost_l1898_189830


namespace row_number_sum_l1898_189868

theorem row_number_sum (n : ℕ) (h : (2 * n - 1) ^ 2 = 2015 ^ 2) : n = 1008 :=
by
  sorry

end row_number_sum_l1898_189868


namespace arithmetic_sequence_sum_l1898_189886

variable {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a n + d

-- Condition 2: Given property
def property (a : ℕ → ℝ) : Prop :=
a 7 + a 13 = 20

theorem arithmetic_sequence_sum (h_seq : is_arithmetic_sequence a) (h_prop : property a) :
  a 9 + a 10 + a 11 = 30 := 
sorry

end arithmetic_sequence_sum_l1898_189886


namespace quarters_given_by_mom_l1898_189881

theorem quarters_given_by_mom :
  let dimes := 4
  let quarters := 4
  let nickels := 7
  let value_dimes := 0.10 * dimes
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let initial_total := value_dimes + value_quarters + value_nickels
  let final_total := 3.00
  let additional_amount := final_total - initial_total
  additional_amount / 0.25 = 5 :=
by
  sorry

end quarters_given_by_mom_l1898_189881


namespace wall_width_l1898_189877

theorem wall_width
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (l_eq : l = 7 * h)
  (volume_eq : w * h * l = 6804) :
  w = 3 :=
by
  sorry

end wall_width_l1898_189877


namespace append_five_new_number_l1898_189896

theorem append_five_new_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) : 
  10 * (10 * t + u) + 5 = 100 * t + 10 * u + 5 :=
by sorry

end append_five_new_number_l1898_189896


namespace parabolic_arch_height_l1898_189856

/-- Define the properties of the parabolic arch -/
def parabolic_arch (a k x : ℝ) : ℝ := a * x^2 + k

/-- Define the conditions of the problem -/
def conditions (a k : ℝ) : Prop :=
  (parabolic_arch a k 25 = 0) ∧ (parabolic_arch a k 0 = 20)

theorem parabolic_arch_height (a k : ℝ) (condition_a_k : conditions a k) :
  parabolic_arch a k 10 = 16.8 :=
by
  unfold conditions at condition_a_k
  cases' condition_a_k with h1 h2
  sorry

end parabolic_arch_height_l1898_189856


namespace combined_selling_price_l1898_189883

theorem combined_selling_price (C_c : ℕ) (C_s : ℕ) (C_m : ℕ) (L_c L_s L_m : ℕ)
  (hc : C_c = 1600)
  (hs : C_s = 12000)
  (hm : C_m = 45000)
  (hlc : L_c = 15)
  (hls : L_s = 10)
  (hlm : L_m = 5) :
  85 * C_c / 100 + 90 * C_s / 100 + 95 * C_m / 100 = 54910 := by
  sorry

end combined_selling_price_l1898_189883


namespace harmonyNumbersWithFirstDigit2_l1898_189833

def isHarmonyNumber (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.sum = 6

def startsWithDigit (d n : ℕ) : Prop :=
  n / 1000 = d

theorem harmonyNumbersWithFirstDigit2 :
  ∃ c : ℕ, c = 15 ∧ ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → isHarmonyNumber n → startsWithDigit 2 n → ∃ m : ℕ, m < c ∧ m = n :=
sorry

end harmonyNumbersWithFirstDigit2_l1898_189833


namespace axis_of_symmetry_l1898_189878

variables (a : ℝ) (x : ℝ)

def parabola := a * (x + 1) * (x - 3)

theorem axis_of_symmetry (h : a ≠ 0) : x = 1 := 
sorry

end axis_of_symmetry_l1898_189878


namespace middle_aged_participating_l1898_189898

-- Definitions of the given conditions
def total_employees : Nat := 1200
def ratio (elderly middle_aged young : Nat) := elderly = 1 ∧ middle_aged = 5 ∧ young = 6
def selected_employees : Nat := 36

-- The stratified sampling condition implies
def stratified_sampling (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) :=
  (elderly + middle_aged + young = total) ∧
  (selected = 36)

-- The proof statement
theorem middle_aged_participating (elderly middle_aged young : Nat) (total : Nat) (selected : Nat) 
  (h_ratio : ratio elderly middle_aged young) 
  (h_total : total = total_employees)
  (h_sampled : stratified_sampling elderly middle_aged young (elderly + middle_aged + young) selected) : 
  selected * middle_aged / (elderly + middle_aged + young) = 15 := 
by sorry

end middle_aged_participating_l1898_189898


namespace vector_parallel_m_l1898_189892

theorem vector_parallel_m {m : ℝ} (h : (2:ℝ) * m - (-1 * -1) = 0) : m = 1 / 2 := 
by
  sorry

end vector_parallel_m_l1898_189892


namespace number_of_silverware_per_setting_l1898_189812

-- Conditions
def silverware_weight_per_piece := 4   -- in ounces
def plates_per_setting := 2
def plate_weight := 12  -- in ounces
def tables := 15
def settings_per_table := 8
def backup_settings := 20
def total_weight := 5040  -- in ounces

-- Let's define variables in our conditions
def settings := tables * settings_per_table + backup_settings
def plates_weight_per_setting := plates_per_setting * plate_weight
def total_silverware_weight (S : Nat) := S * silverware_weight_per_piece * settings
def total_plate_weight := plates_weight_per_setting * settings

-- Define the required proof statement
theorem number_of_silverware_per_setting : 
  ∃ S : Nat, (total_silverware_weight S + total_plate_weight = total_weight) ∧ S = 3 :=
by {
  sorry -- proof will be provided here
}

end number_of_silverware_per_setting_l1898_189812


namespace nicky_catchup_time_l1898_189824

-- Definitions related to the problem
def head_start : ℕ := 12
def speed_cristina : ℕ := 5
def speed_nicky : ℕ := 3
def time_to_catchup : ℕ := 36
def nicky_runtime_before_catchup : ℕ := head_start + time_to_catchup

-- Theorem to prove the correct runtime for Nicky before Cristina catches up
theorem nicky_catchup_time : nicky_runtime_before_catchup = 48 := by
  sorry

end nicky_catchup_time_l1898_189824


namespace proportion_red_MMs_l1898_189890

theorem proportion_red_MMs (R B : ℝ) (h1 : R + B = 1) 
  (h2 : R * (4 / 5) = B * (1 / 6)) :
  R = 5 / 29 :=
by
  sorry

end proportion_red_MMs_l1898_189890


namespace daisies_per_bouquet_l1898_189869

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end daisies_per_bouquet_l1898_189869


namespace domain_of_g_l1898_189817

def f : ℝ → ℝ := sorry  -- Placeholder for the function f

noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (2 * x + 1)

theorem domain_of_g :
  ∀ x : ℝ, g x ≠ 0 → (-1/2 < x ∧ x ≤ 3) :=
by
  intro x hx
  sorry

end domain_of_g_l1898_189817


namespace c_range_l1898_189801

open Real

theorem c_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1)
  (h2 : 1 / (a + b) + 1 / c = 1) : 1 < c ∧ c ≤ 4 / 3 := 
sorry

end c_range_l1898_189801


namespace find_a_l1898_189816

-- Define what it means for P(X = k) to be given by a particular function
def P (X : ℕ) (a : ℕ) := X / (2 * a)

-- Define the condition on the probabilities
def sum_of_probabilities_is_one (a : ℕ) :=
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) + 4 / (2 * a)) = 1

-- The theorem to prove
theorem find_a (a : ℕ) (h : sum_of_probabilities_is_one a) : a = 5 :=
by sorry

end find_a_l1898_189816


namespace cell_phone_bill_l1898_189894

-- Definitions
def base_cost : ℝ := 20
def cost_per_text : ℝ := 0.05
def cost_per_extra_minute : ℝ := 0.10
def texts_sent : ℕ := 100
def hours_talked : ℝ := 30.5
def included_hours : ℝ := 30

-- Calculate extra minutes used
def extra_minutes : ℝ := (hours_talked - included_hours) * 60

-- Total cost calculation
def total_cost : ℝ := 
  base_cost + 
  (texts_sent * cost_per_text) + 
  (extra_minutes * cost_per_extra_minute)

-- Proof problem statement
theorem cell_phone_bill : total_cost = 28 := by
  sorry

end cell_phone_bill_l1898_189894


namespace largest_difference_l1898_189836

noncomputable def A := 3 * 2023^2024
noncomputable def B := 2023^2024
noncomputable def C := 2022 * 2023^2023
noncomputable def D := 3 * 2023^2023
noncomputable def E := 2023^2023
noncomputable def F := 2023^2022

theorem largest_difference :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end largest_difference_l1898_189836


namespace arithmetic_sequence_an_12_l1898_189814

theorem arithmetic_sequence_an_12 {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 9)
  (h_a6 : a 6 = 15) :
  a 12 = 27 := 
sorry

end arithmetic_sequence_an_12_l1898_189814


namespace sum_a6_a7_a8_l1898_189832

-- Sequence definition and sum of the first n terms
def S (n : ℕ) : ℕ := n^2 + 3 * n

theorem sum_a6_a7_a8 : S 8 - S 5 = 48 :=
by
  -- Definition and proof details are skipped
  sorry

end sum_a6_a7_a8_l1898_189832


namespace prob_XYZ_wins_l1898_189871

-- Define probabilities as given in the conditions
def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_Z : ℚ := 1 / 12

-- Define the probability that one of X, Y, or Z wins, assuming events are mutually exclusive
def P_XYZ_wins : ℚ := P_X + P_Y + P_Z

theorem prob_XYZ_wins : P_XYZ_wins = 11 / 24 := by
  -- sorry is used to skip the proof
  sorry

end prob_XYZ_wins_l1898_189871


namespace problem_l1898_189891

theorem problem (w : ℝ) (h : 13 = 13 * w / (1 - w)) : w^2 = 1 / 4 :=
by
  sorry

end problem_l1898_189891


namespace leap_day_2040_is_tuesday_l1898_189806

-- Define the given condition that 29th February 2012 is Wednesday
def feb_29_2012_is_wednesday : Prop := sorry

-- Define the calculation of the day of the week for February 29, 2040
def day_of_feb_29_2040 (initial_day : Nat) : Nat := (10228 % 7 + initial_day) % 7

-- Define the proof statement
theorem leap_day_2040_is_tuesday : feb_29_2012_is_wednesday →
  (day_of_feb_29_2040 3 = 2) := -- Here, 3 represents Wednesday and 2 represents Tuesday
sorry

end leap_day_2040_is_tuesday_l1898_189806


namespace area_of_region_between_semicircles_l1898_189823

/-- Given a region between two semicircles with the same center and parallel diameters,
where the farthest distance between two points with a clear line of sight is 12 meters,
prove that the area of the region is 18π square meters. -/
theorem area_of_region_between_semicircles :
  ∃ (R r : ℝ), R > r ∧ (R - r = 6) ∧ 18 * Real.pi = (Real.pi / 2) * (R^2 - r^2) ∧ (R^2 - r^2 = 144) :=
sorry

end area_of_region_between_semicircles_l1898_189823


namespace canteen_distances_l1898_189808

theorem canteen_distances 
  (B G C : ℝ)
  (hB : B = 600)
  (hBG : G = 800)
  (hBC_eq_2x : ∃ x, C = 2 * x ∧ B = G + x + x) :
  G = 800 / 3 :=
by
  sorry

end canteen_distances_l1898_189808


namespace gcd_ab_a2b2_eq_one_or_two_l1898_189861

-- Definitions and conditions
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Problem statement
theorem gcd_ab_a2b2_eq_one_or_two (a b : ℕ) (h : coprime a b) : 
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_eq_one_or_two_l1898_189861


namespace tangent_lines_through_P_l1898_189854

noncomputable def curve_eq (x : ℝ) : ℝ := 1/3 * x^3 + 4/3

theorem tangent_lines_through_P (x y : ℝ) :
  ((4 * x - y - 4 = 0 ∨ y = x + 2) ∧ (curve_eq 2 = 4)) :=
by
  sorry

end tangent_lines_through_P_l1898_189854


namespace sally_orange_balloons_l1898_189837

def initial_orange_balloons : ℝ := 9.0
def found_orange_balloons : ℝ := 2.0

theorem sally_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := 
by
  sorry

end sally_orange_balloons_l1898_189837


namespace arithmetic_mean_six_expressions_l1898_189899

theorem arithmetic_mean_six_expressions (x : ℝ)
  (h : (x + 8 + 15 + 2 * x + 13 + 2 * x + 4 + 3 * x + 5) / 6 = 30) : x = 13.5 :=
by
  sorry

end arithmetic_mean_six_expressions_l1898_189899


namespace factorization_a_minus_b_l1898_189828

theorem factorization_a_minus_b (a b : ℤ) (h1 : 3 * b + a = -7) (h2 : a * b = -6) : a - b = 7 :=
sorry

end factorization_a_minus_b_l1898_189828


namespace group_booking_cost_correct_l1898_189853

-- Definitions based on the conditions of the problem
def weekday_rate_first_week : ℝ := 18.00
def weekend_rate_first_week : ℝ := 20.00
def weekday_rate_additional_weeks : ℝ := 11.00
def weekend_rate_additional_weeks : ℝ := 13.00
def security_deposit : ℝ := 50.00
def discount_rate : ℝ := 0.10
def group_size : ℝ := 5
def stay_duration : ℕ := 23

-- Computation of total cost
def total_cost (first_week_weekdays : ℕ) (first_week_weekends : ℕ) 
  (additional_week_weekdays : ℕ) (additional_week_weekends : ℕ) 
  (additional_days_weekdays : ℕ) : ℝ := 
  let cost_first_weekdays := first_week_weekdays * weekday_rate_first_week
  let cost_first_weekends := first_week_weekends * weekend_rate_first_week
  let cost_additional_weeks := 2 * (additional_week_weekdays * weekday_rate_additional_weeks + 
                                    additional_week_weekends * weekend_rate_additional_weeks)
  let cost_additional_days := additional_days_weekdays * weekday_rate_additional_weeks
  let total_before_deposit := cost_first_weekdays + cost_first_weekends + 
                              cost_additional_weeks + cost_additional_days
  let total_before_discount := total_before_deposit + security_deposit
  let total_discount := discount_rate * total_before_discount
  total_before_discount - total_discount

-- Proof setup
theorem group_booking_cost_correct :
  total_cost 5 2 5 2 2 = 327.60 :=
by
  -- Placeholder for the proof; steps not required for Lean statement
  sorry

end group_booking_cost_correct_l1898_189853


namespace option_c_correct_l1898_189804

theorem option_c_correct (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end option_c_correct_l1898_189804


namespace cost_price_of_cloths_l1898_189866

-- Definitions based on conditions
def SP_A := 8500 / 85
def Profit_A := 15
def CP_A := SP_A - Profit_A

def SP_B := 10200 / 120
def Profit_B := 12
def CP_B := SP_B - Profit_B

def SP_C := 4200 / 60
def Profit_C := 10
def CP_C := SP_C - Profit_C

-- Theorem to prove the cost prices
theorem cost_price_of_cloths :
    CP_A = 85 ∧
    CP_B = 73 ∧
    CP_C = 60 :=
by
    sorry

end cost_price_of_cloths_l1898_189866


namespace range_of_m_l1898_189819

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 1) + 4 / y = 1) (m : ℝ) :
  x + y / 4 > m^2 - 5 * m - 3 ↔ -1 < m ∧ m < 6 := sorry

end range_of_m_l1898_189819


namespace quadratic_roots_k_relation_l1898_189831

theorem quadratic_roots_k_relation (k a b k1 k2 : ℝ) 
    (h_eq : k * (a^2 - a) + 2 * a + 7 = 0)
    (h_eq_b : k * (b^2 - b) + 2 * b + 7 = 0)
    (h_ratio : a / b + b / a = 3)
    (h_k : k = k1 ∨ k = k2)
    (h_vieta_sum : k1 + k2 = 39)
    (h_vieta_product : k1 * k2 = 4) :
    k1 / k2 + k2 / k1 = 1513 / 4 := 
    sorry

end quadratic_roots_k_relation_l1898_189831


namespace boxes_filled_l1898_189879

theorem boxes_filled (total_toys toys_per_box : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 :=
by
  sorry

end boxes_filled_l1898_189879


namespace part1_part2_l1898_189813

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- Problem (1)
theorem part1 (a : ℝ) (h : a = 1) : 
  ∀ x : ℝ, f x a ≥ 2 ↔ x ≤ 0 ∨ x ≥ 2 := 
  sorry

-- Problem (2)
theorem part2 (a : ℝ) (h : a > 1) : 
  (∀ x : ℝ, f x a + abs (x - 1) ≥ 2) ↔ a ≥ 3 := 
  sorry

end part1_part2_l1898_189813


namespace find_x_l1898_189838

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 15) : x = 35 :=
sorry

end find_x_l1898_189838


namespace rosalina_gifts_l1898_189884

theorem rosalina_gifts (Emilio_gifts Jorge_gifts Pedro_gifts : ℕ) 
  (hEmilio : Emilio_gifts = 11) 
  (hJorge : Jorge_gifts = 6) 
  (hPedro : Pedro_gifts = 4) : 
  Emilio_gifts + Jorge_gifts + Pedro_gifts = 21 :=
by
  sorry

end rosalina_gifts_l1898_189884


namespace units_digit_quotient_4_l1898_189870

theorem units_digit_quotient_4 (n : ℕ) (h₁ : n ≥ 1) :
  (5^1994 + 6^1994) % 10 = 1 ∧ (5^1994 + 6^1994) % 7 = 5 → 
  (5^1994 + 6^1994) / 7 % 10 = 4 := 
sorry

end units_digit_quotient_4_l1898_189870


namespace total_fruit_punch_l1898_189848

/-- Conditions -/
def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def apple_juice : ℝ := cherry_punch - 1.5
def pineapple_juice : ℝ := 3
def grape_punch : ℝ := 1.5 * apple_juice

/-- Proof that total fruit punch is 35.25 liters -/
theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end total_fruit_punch_l1898_189848


namespace percentage_of_class_are_men_proof_l1898_189864

/-- Definition of the problem using the conditions provided. -/
def percentage_of_class_are_men (W M : ℝ) : Prop :=
  -- Conditions based on the problem statement
  M + W = 100 ∧
  0.10 * W + 0.85 * M = 40

/-- The proof statement we need to show: Under the given conditions, the percentage of men (M) is 40. -/
theorem percentage_of_class_are_men_proof (W M : ℝ) :
  percentage_of_class_are_men W M → M = 40 :=
by
  sorry

end percentage_of_class_are_men_proof_l1898_189864


namespace methane_required_l1898_189807

def mole_of_methane (moles_of_oxygen : ℕ) : ℕ := 
  if moles_of_oxygen = 2 then 1 else 0

theorem methane_required (moles_of_oxygen : ℕ) : 
  moles_of_oxygen = 2 → mole_of_methane moles_of_oxygen = 1 := 
by 
  intros h
  simp [mole_of_methane, h]

end methane_required_l1898_189807


namespace correct_option_for_ruler_length_l1898_189818

theorem correct_option_for_ruler_length (A B C D : String) (correct_answer : String) : 
  A = "two times as longer as" ∧ 
  B = "twice the length of" ∧ 
  C = "three times longer of" ∧ 
  D = "twice long than" ∧ 
  correct_answer = B := 
by
  sorry

end correct_option_for_ruler_length_l1898_189818


namespace even_function_a_equals_one_l1898_189859

theorem even_function_a_equals_one 
  (a : ℝ) 
  (h : ∀ x : ℝ, 2^(-x) + a * 2^x = 2^x + a * 2^(-x)) : 
  a = 1 := 
by
  sorry

end even_function_a_equals_one_l1898_189859


namespace determine_positive_integers_l1898_189834

theorem determine_positive_integers (x y z : ℕ) (h : x^2 + y^2 - 15 = 2^z) :
  (x = 0 ∧ y = 4 ∧ z = 0) ∨ (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 4 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 1) :=
sorry

end determine_positive_integers_l1898_189834


namespace transform_equation_l1898_189815

theorem transform_equation (x : ℝ) (h₁ : x ≠ 3 / 2) (h₂ : 5 - 3 * x = 1) :
  x = 4 / 3 :=
sorry

end transform_equation_l1898_189815


namespace inequality_always_true_l1898_189852

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end inequality_always_true_l1898_189852


namespace inequality_has_solutions_l1898_189843

theorem inequality_has_solutions (a : ℝ) :
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3 * a) ↔ (a < -1 ∨ 4 < a) := 
by
  sorry

end inequality_has_solutions_l1898_189843


namespace driver_average_speed_l1898_189844

theorem driver_average_speed (v t : ℝ) (h1 : ∀ d : ℝ, d = v * t → (d / (v + 10)) = (3 / 4) * t) : v = 30 := by
  sorry

end driver_average_speed_l1898_189844


namespace bottom_row_bricks_l1898_189865

theorem bottom_row_bricks (x : ℕ) 
    (h : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 200) : x = 42 :=
sorry

end bottom_row_bricks_l1898_189865


namespace value_of_a_l1898_189846

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem value_of_a (a : ℝ) (ha : a > 1) (h : f (g a) = 12) : 
  a = Real.sqrt (Real.sqrt 10 - 2) :=
by sorry

end value_of_a_l1898_189846


namespace photograph_perimeter_is_23_l1898_189827

noncomputable def photograph_perimeter (w h m : ℝ) : ℝ :=
if (w + 4) * (h + 4) = m ∧ (w + 8) * (h + 8) = m + 94 then 2 * (w + h) else 0

theorem photograph_perimeter_is_23 (w h m : ℝ) 
    (h₁ : (w + 4) * (h + 4) = m) 
    (h₂ : (w + 8) * (h + 8) = m + 94) : 
    photograph_perimeter w h m = 23 := 
by 
  sorry

end photograph_perimeter_is_23_l1898_189827


namespace additional_birds_flew_up_l1898_189855

-- Defining the conditions from the problem
def original_birds : ℕ := 179
def total_birds : ℕ := 217

-- Defining the question to be proved as a theorem
theorem additional_birds_flew_up : 
  total_birds - original_birds = 38 :=
by
  sorry

end additional_birds_flew_up_l1898_189855


namespace problem_l1898_189874

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end problem_l1898_189874


namespace greatest_number_in_consecutive_multiples_l1898_189888

theorem greatest_number_in_consecutive_multiples (s : Set ℕ) (h₁ : ∃ m : ℕ, s = {n | ∃ k < 100, n = 8 * (m + k)} ∧ m = 14) :
  (∃ n ∈ s, ∀ x ∈ s, x ≤ n) →
  ∃ n ∈ s, n = 904 :=
by
  sorry

end greatest_number_in_consecutive_multiples_l1898_189888


namespace tori_original_height_l1898_189822

-- Definitions for given conditions
def current_height : ℝ := 7.26
def height_gained : ℝ := 2.86

-- Theorem statement
theorem tori_original_height : current_height - height_gained = 4.40 :=
by sorry

end tori_original_height_l1898_189822


namespace rachel_brought_16_brownies_l1898_189851

def total_brownies : ℕ := 40
def brownies_left_at_home : ℕ := 24

def brownies_brought_to_school : ℕ :=
  total_brownies - brownies_left_at_home

theorem rachel_brought_16_brownies :
  brownies_brought_to_school = 16 :=
by
  sorry

end rachel_brought_16_brownies_l1898_189851


namespace amount_paid_l1898_189860

theorem amount_paid (lemonade_price_per_cup sandwich_price_per_item change_received : ℝ) 
    (num_lemonades num_sandwiches : ℕ)
    (h1 : lemonade_price_per_cup = 2) 
    (h2 : sandwich_price_per_item = 2.50) 
    (h3 : change_received = 11) 
    (h4 : num_lemonades = 2) 
    (h5 : num_sandwiches = 2) : 
    (lemonade_price_per_cup * num_lemonades + sandwich_price_per_item * num_sandwiches + change_received = 20) :=
by
  sorry

end amount_paid_l1898_189860


namespace inequality_holds_l1898_189841

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 :=
by
  sorry

end inequality_holds_l1898_189841


namespace zmod_field_l1898_189835

theorem zmod_field (p : ℕ) [Fact (Nat.Prime p)] : Field (ZMod p) :=
sorry

end zmod_field_l1898_189835


namespace pears_picked_l1898_189839

def Jason_pears : ℕ := 46
def Keith_pears : ℕ := 47
def Mike_pears : ℕ := 12
def total_pears : ℕ := 105

theorem pears_picked :
  Jason_pears + Keith_pears + Mike_pears = total_pears :=
by
  exact rfl

end pears_picked_l1898_189839


namespace find_apartment_number_l1898_189880

open Nat

def is_apartment_number (x a b : ℕ) : Prop :=
  x = 10 * a + b ∧ x = 17 * b

theorem find_apartment_number : ∃ x a b : ℕ, is_apartment_number x a b ∧ x = 85 :=
by
  sorry

end find_apartment_number_l1898_189880


namespace problem_proof_l1898_189858

noncomputable def problem : Prop :=
  ∀ x : ℝ, (x ≠ 2 ∧ (x-2)/(x-4) ≤ 3) ↔ (4 < x ∧ x < 5)

theorem problem_proof : problem := sorry

end problem_proof_l1898_189858


namespace number_of_students_more_than_pets_l1898_189811

theorem number_of_students_more_than_pets 
  (students_per_classroom pets_per_classroom num_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : pets_per_classroom = 3)
  (h3 : num_classrooms = 5) :
  (students_per_classroom * num_classrooms) - (pets_per_classroom * num_classrooms) = 85 := 
by
  sorry

end number_of_students_more_than_pets_l1898_189811


namespace polynomial_expansion_correct_l1898_189895

open Polynomial

noncomputable def poly1 : Polynomial ℤ := X^2 + 3 * X - 4
noncomputable def poly2 : Polynomial ℤ := 2 * X^2 - X + 5
noncomputable def expected : Polynomial ℤ := 2 * X^4 + 5 * X^3 - 6 * X^2 + 19 * X - 20

theorem polynomial_expansion_correct :
  poly1 * poly2 = expected :=
sorry

end polynomial_expansion_correct_l1898_189895


namespace find_middle_number_l1898_189887

namespace Problem

-- Define the three numbers x, y, z
variables (x y z : ℕ)

-- Given conditions from the problem
def condition1 (h1 : x + y = 18) := x + y = 18
def condition2 (h2 : x + z = 23) := x + z = 23
def condition3 (h3 : y + z = 27) := y + z = 27
def condition4 (h4 : x < y ∧ y < z) := x < y ∧ y < z

-- Statement to prove:
theorem find_middle_number (h1 : x + y = 18) (h2 : x + z = 23) (h3 : y + z = 27) (h4 : x < y ∧ y < z) : 
  y = 11 :=
by
  sorry

end Problem

end find_middle_number_l1898_189887


namespace sqrt_meaningful_range_l1898_189825

theorem sqrt_meaningful_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end sqrt_meaningful_range_l1898_189825


namespace triangle_area_is_54_l1898_189800

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l1898_189800


namespace red_socks_l1898_189849

variable {R : ℕ}

theorem red_socks (h1 : 2 * R + R + 6 * R = 90) : R = 10 := 
by
  sorry

end red_socks_l1898_189849


namespace min_value_of_n_l1898_189863

theorem min_value_of_n 
  (n k : ℕ) 
  (h1 : 8 * n = 225 * k + 3)
  (h2 : k ≡ 5 [MOD 8]) : 
  n = 141 := 
  sorry

end min_value_of_n_l1898_189863


namespace selling_price_correct_l1898_189842

variable (CostPrice GainPercent : ℝ)
variables (Profit SellingPrice : ℝ)

noncomputable def calculateProfit : ℝ := (GainPercent / 100) * CostPrice

noncomputable def calculateSellingPrice : ℝ := CostPrice + calculateProfit CostPrice GainPercent

theorem selling_price_correct 
  (h1 : CostPrice = 900) 
  (h2 : GainPercent = 30)
  : calculateSellingPrice CostPrice GainPercent = 1170 := by
  sorry

end selling_price_correct_l1898_189842


namespace binom_20_4_plus_10_l1898_189875

open Nat

noncomputable def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_20_4_plus_10 :
  binom 20 4 + 10 = 4855 := by
  sorry

end binom_20_4_plus_10_l1898_189875


namespace correct_statements_B_and_C_l1898_189876

variable {a b c : ℝ}

-- Definitions from the conditions
def conditionB (a b c : ℝ) : Prop := a > b ∧ b > 0 ∧ c < 0
def conclusionB (a b c : ℝ) : Prop := c / a^2 > c / b^2

def conditionC (a b c : ℝ) : Prop := c > a ∧ a > b ∧ b > 0
def conclusionC (a b c : ℝ) : Prop := a / (c - a) > b / (c - b)

theorem correct_statements_B_and_C (a b c : ℝ) : 
  (conditionB a b c → conclusionB a b c) ∧ 
  (conditionC a b c → conclusionC a b c) :=
by
  sorry

end correct_statements_B_and_C_l1898_189876


namespace courses_students_problem_l1898_189810

theorem courses_students_problem :
  let courses := Fin 6 -- represent 6 courses
  let students := Fin 20 -- represent 20 students
  (∀ (C C' : courses), ∀ (S : Finset students), S.card = 5 → 
    ¬ ((∀ s ∈ S, ∃ s_courses : Finset courses, C ∈ s_courses ∧ C' ∈ s_courses) ∨ 
       (∀ s ∈ S, ∃ s_courses : Finset courses, C ∉ s_courses ∧ C' ∉ s_courses))) :=
by sorry

end courses_students_problem_l1898_189810


namespace johns_overall_loss_l1898_189809

noncomputable def johns_loss_percentage : ℝ :=
  let cost_A := 1000 * 2
  let cost_B := 1500 * 3
  let cost_C := 2000 * 4
  let discount_A := 0.1
  let discount_B := 0.15
  let discount_C := 0.2
  let cost_A_after_discount := cost_A * (1 - discount_A)
  let cost_B_after_discount := cost_B * (1 - discount_B)
  let cost_C_after_discount := cost_C * (1 - discount_C)
  let total_cost_after_discount := cost_A_after_discount + cost_B_after_discount + cost_C_after_discount
  let import_tax_rate := 0.08
  let import_tax := total_cost_after_discount * import_tax_rate
  let total_cost_incl_tax := total_cost_after_discount + import_tax
  let cost_increase_rate_C := 0.04
  let new_cost_C := 2000 * (4 + 4 * cost_increase_rate_C)
  let adjusted_total_cost := cost_A_after_discount + cost_B_after_discount + new_cost_C
  let total_selling_price := (800 * 3) + (70 * 3 + 1400 * 3.5 + 900 * 5) + (130 * 2.5 + 130 * 3 + 130 * 5)
  let gain_or_loss := total_selling_price - adjusted_total_cost
  let loss_percentage := (gain_or_loss / adjusted_total_cost) * 100
  loss_percentage

theorem johns_overall_loss : abs (johns_loss_percentage + 4.09) < 0.01 := sorry

end johns_overall_loss_l1898_189809


namespace Peter_can_guarantee_victory_l1898_189885

structure Board :=
  (size : ℕ)
  (cells : Fin size × Fin size → Option Color)

inductive Player
  | Peter
  | Victor
deriving DecidableEq

inductive Color
  | Red
  | Green
  | White
deriving DecidableEq

structure Move :=
  (player : Player)
  (rectangle : Fin 2 × Fin 2)
  (position : Fin 7 × Fin 7)

def isValidMove (board : Board) (move : Move) : Prop := sorry

def applyMove (board : Board) (move : Move) : Board := sorry

def allCellsColored (board : Board) : Prop := sorry

theorem Peter_can_guarantee_victory :
  ∀ (initialBoard : Board),
    (∀ (move : Move), move.player = Player.Victor → isValidMove initialBoard move) →
    Player.Peter = Player.Peter →
    (∃ finalBoard : Board,
       allCellsColored finalBoard ∧ 
       ¬ (∃ (move : Move), move.player = Player.Victor ∧ isValidMove finalBoard move)) :=
sorry

end Peter_can_guarantee_victory_l1898_189885


namespace total_daily_salary_l1898_189803

def manager_salary : ℕ := 5
def clerk_salary : ℕ := 2
def num_managers : ℕ := 2
def num_clerks : ℕ := 3

theorem total_daily_salary : num_managers * manager_salary + num_clerks * clerk_salary = 16 := by
    sorry

end total_daily_salary_l1898_189803


namespace point_on_hyperbola_l1898_189873

theorem point_on_hyperbola : 
  (∃ x y : ℝ, (x, y) = (3, -2) ∧ y = -6 / x) :=
by
  sorry

end point_on_hyperbola_l1898_189873


namespace half_abs_sum_diff_squares_cubes_l1898_189821

theorem half_abs_sum_diff_squares_cubes (a b : ℤ) (h1 : a = 21) (h2 : b = 15) :
  (|a^2 - b^2| + |a^3 - b^3|) / 2 = 3051 := by
  sorry

end half_abs_sum_diff_squares_cubes_l1898_189821


namespace probability_each_box_2_fruits_l1898_189889

noncomputable def totalWaysToDistributePears : ℕ := (Nat.choose 8 4)
noncomputable def totalWaysToDistributeApples : ℕ := 5^6

noncomputable def case1 : ℕ := (Nat.choose 5 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2))
noncomputable def case2 : ℕ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1))
noncomputable def case3 : ℕ := (Nat.choose 5 4) * (Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1))

noncomputable def totalFavorableDistributions : ℕ := case1 + case2 + case3
noncomputable def totalPossibleDistributions : ℕ := totalWaysToDistributePears * totalWaysToDistributeApples

noncomputable def probability : ℚ := (totalFavorableDistributions : ℚ) / totalPossibleDistributions * 100

theorem probability_each_box_2_fruits :
  probability = 0.74 := 
sorry

end probability_each_box_2_fruits_l1898_189889


namespace equal_distances_l1898_189857

theorem equal_distances (c : ℝ) (distance : ℝ) :
  abs (2 - -4) = distance ∧ (abs (c - -4) = distance ∨ abs (c - 2) = distance) ↔ (c = -10 ∨ c = 8) :=
by
  sorry

end equal_distances_l1898_189857


namespace part1_part2_l1898_189820

variable (x y z : ℕ)

theorem part1 (h1 : 3 * x + 5 * y = 98) (h2 : 8 * x + 3 * y = 158) : x = 16 ∧ y = 10 :=
sorry

theorem part2 (hx : x = 16) (hy : y = 10) (hz : 16 * z + 10 * (40 - z) ≤ 550) : z ≤ 25 :=
sorry

end part1_part2_l1898_189820


namespace triangle_square_ratio_l1898_189897

theorem triangle_square_ratio (s_t s_s : ℝ) (h : 3 * s_t = 4 * s_s) : s_t / s_s = 4 / 3 := by
  sorry

end triangle_square_ratio_l1898_189897


namespace probability_no_defective_pencils_l1898_189862

theorem probability_no_defective_pencils : 
  let total_pencils := 9
  let defective_pencils := 2
  let chosen_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils chosen_pencils
  let non_defective_ways := Nat.choose non_defective_pencils chosen_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 12 := 
by
  sorry

end probability_no_defective_pencils_l1898_189862


namespace find_prime_pairs_l1898_189826

open Nat

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem as a theorem in Lean
theorem find_prime_pairs :
  ∀ (p n : ℕ), is_prime p ∧ n > 0 ∧ p^3 - 2*p^2 + p + 1 = 3^n ↔ (p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4) :=
by
  sorry

end find_prime_pairs_l1898_189826


namespace compare_abc_l1898_189872

noncomputable def a : ℝ := (1 / 4) * Real.logb 2 3
noncomputable def b : ℝ := 1 / 2
noncomputable def c : ℝ := (1 / 2) * Real.logb 5 3

theorem compare_abc : c < a ∧ a < b := sorry

end compare_abc_l1898_189872
