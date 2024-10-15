import Mathlib

namespace NUMINAMATH_GPT_angle_sum_l1002_100274

theorem angle_sum (A B C : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h_triangle : A + B + C = 180) (h_complement : 180 - C = 130) :
  A + B = 130 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_l1002_100274


namespace NUMINAMATH_GPT_employed_population_percentage_l1002_100222

variable (P : ℝ) -- Total population
variable (percentage_employed_to_population : ℝ) -- Percentage of total population employed
variable (percentage_employed_males_to_population : ℝ := 0.42) -- 42% of population are employed males
variable (percentage_employed_females_to_employed : ℝ := 0.30) -- 30% of employed people are females

theorem employed_population_percentage :
  percentage_employed_to_population = 0.60 :=
sorry

end NUMINAMATH_GPT_employed_population_percentage_l1002_100222


namespace NUMINAMATH_GPT_value_of_h_l1002_100214

theorem value_of_h (h : ℤ) : (-1)^3 + h * (-1) - 20 = 0 → h = -21 :=
by
  intro h_cond
  sorry

end NUMINAMATH_GPT_value_of_h_l1002_100214


namespace NUMINAMATH_GPT_map_length_scale_l1002_100290

theorem map_length_scale (len1 len2 : ℕ) (dist1 dist2 : ℕ) (h1 : len1 = 15) (h2 : dist1 = 90) (h3 : len2 = 20) :
  dist2 = 120 :=
by
  sorry

end NUMINAMATH_GPT_map_length_scale_l1002_100290


namespace NUMINAMATH_GPT_sequence_identity_l1002_100208

noncomputable def a_n (n : ℕ) : ℝ := n + 1
noncomputable def b_n (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := (n * (n+1)) / 2  -- Sum of first n terms of arithmetic sequence
noncomputable def T_n (n : ℕ) : ℝ := 2 * (3^n - 1) / 2  -- Sum of first n terms of geometric sequence
noncomputable def c_n (n : ℕ) : ℝ := 2 * a_n n / b_n n
noncomputable def C_n (n : ℕ) : ℝ := (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))

theorem sequence_identity :
  a_n 1 = b_n 1 ∧
  2 * a_n 2 = b_n 2 ∧
  S_n 2 + T_n 2 = 13 ∧
  2 * S_n 3 = b_n 3 →
  (∀ n : ℕ, a_n n = n + 1 ∧ b_n n = 2 * 3^(n-1)) ∧
  (∀ n : ℕ, C_n n = (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))) :=
sorry

end NUMINAMATH_GPT_sequence_identity_l1002_100208


namespace NUMINAMATH_GPT_a_congruent_b_mod_1008_l1002_100231

theorem a_congruent_b_mod_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b - b^a = 1008) : a ≡ b [MOD 1008] :=
by
  sorry

end NUMINAMATH_GPT_a_congruent_b_mod_1008_l1002_100231


namespace NUMINAMATH_GPT_pastry_problem_minimum_n_l1002_100254

theorem pastry_problem_minimum_n (fillings : Finset ℕ) (n : ℕ) : 
    fillings.card = 10 →
    (∃ pairs : Finset (ℕ × ℕ), pairs.card = 45 ∧ ∀ (p : ℕ × ℕ), p ∈ pairs → p.1 ≠ p.2 ∧ p.1 ∈ fillings ∧ p.2 ∈ fillings) →
    (∀ (remaining_pies : Finset (ℕ × ℕ)), remaining_pies.card = 45 - n → 
     ∃ f1 f2, (f1, f2) ∈ remaining_pies → (f1 ∈ fillings ∧ f2 ∈ fillings)) →
    n = 36 :=
by
  intros h_fillings h_pairs h_remaining_pies
  sorry

end NUMINAMATH_GPT_pastry_problem_minimum_n_l1002_100254


namespace NUMINAMATH_GPT_woman_waits_time_after_passing_l1002_100210

-- Definitions based only on the conditions in a)
def man_speed : ℝ := 5 -- in miles per hour
def woman_speed : ℝ := 25 -- in miles per hour
def waiting_time_man_minutes : ℝ := 20 -- in minutes

-- Equivalent proof problem statement
theorem woman_waits_time_after_passing :
  let waiting_time_man_hours := waiting_time_man_minutes / 60
  let distance_man : ℝ := man_speed * waiting_time_man_hours
  let relative_speed : ℝ := woman_speed - man_speed
  let time_woman_covers_distance_hours := distance_man / relative_speed
  let time_woman_covers_distance_minutes := time_woman_covers_distance_hours * 60
  time_woman_covers_distance_minutes = 5 :=
by
  sorry

end NUMINAMATH_GPT_woman_waits_time_after_passing_l1002_100210


namespace NUMINAMATH_GPT_boy_work_completion_days_l1002_100289

theorem boy_work_completion_days (M W B : ℚ) (D : ℚ)
  (h1 : M + W + B = 1 / 4)
  (h2 : M = 1 / 6)
  (h3 : W = 1 / 36)
  (h4 : B = 1 / D) :
  D = 18 := by
  sorry

end NUMINAMATH_GPT_boy_work_completion_days_l1002_100289


namespace NUMINAMATH_GPT_zero_is_multiple_of_all_primes_l1002_100267

theorem zero_is_multiple_of_all_primes :
  ∀ (x : ℕ), (∀ p : ℕ, Prime p → ∃ n : ℕ, x = n * p) ↔ x = 0 := by
sorry

end NUMINAMATH_GPT_zero_is_multiple_of_all_primes_l1002_100267


namespace NUMINAMATH_GPT_problem_statement_l1002_100204

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 12) : a^3 + 1/a^3 = 18 * Real.sqrt 3 :=
by
  -- We'll skip the proof as per instruction
  sorry

end NUMINAMATH_GPT_problem_statement_l1002_100204


namespace NUMINAMATH_GPT_Heather_delay_l1002_100280

noncomputable def find_start_time : ℝ :=
  let d := 15 -- Initial distance between Stacy and Heather in miles
  let H := 5 -- Heather's speed in miles/hour
  let S := H + 1 -- Stacy's speed in miles/hour
  let d_H := 5.7272727272727275 -- Distance Heather walked when they meet
  let t_H := d_H / H -- Time Heather walked till they meet in hours
  let d_S := S * t_H -- Distance Stacy walked till they meet in miles
  let total_distance := d_H + d_S -- Total distance covered when they meet in miles
  let remaining_distance := d - total_distance -- Remaining distance Stacy covers alone before Heather starts in miles
  let t_S := remaining_distance / S -- Time Stacy walked alone in hours
  let minutes := t_S * 60 -- Convert time Stacy walked alone to minutes
  minutes -- Result in minutes

theorem Heather_delay : find_start_time = 24 := by
  sorry -- Proof of the theorem

end NUMINAMATH_GPT_Heather_delay_l1002_100280


namespace NUMINAMATH_GPT_probability_of_selecting_cooking_l1002_100284

-- Define the context where Xiao Ming has to choose a course randomly
def courses := ["planting", "cooking", "pottery", "carpentry"]

-- Define a statement to prove the probability of selecting "cooking" is 1/4
theorem probability_of_selecting_cooking :
  (1 / (courses.length : ℝ)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_cooking_l1002_100284


namespace NUMINAMATH_GPT_tan_of_sine_plus_cosine_eq_neg_4_over_3_l1002_100233

variable {A : ℝ}

theorem tan_of_sine_plus_cosine_eq_neg_4_over_3 
  (h : Real.sin A + Real.cos A = -4/3) : 
  Real.tan A = -4/3 :=
sorry

end NUMINAMATH_GPT_tan_of_sine_plus_cosine_eq_neg_4_over_3_l1002_100233


namespace NUMINAMATH_GPT_mass_of_empty_glass_l1002_100282

theorem mass_of_empty_glass (mass_full : ℕ) (mass_half : ℕ) (G : ℕ) :
  mass_full = 1000 →
  mass_half = 700 →
  G = mass_full - (mass_full - mass_half) * 2 →
  G = 400 :=
by
  intros h_full h_half h_G_eq
  sorry

end NUMINAMATH_GPT_mass_of_empty_glass_l1002_100282


namespace NUMINAMATH_GPT_truck_capacity_l1002_100262

-- Definitions based on conditions
def initial_fuel : ℕ := 38
def total_money : ℕ := 350
def change : ℕ := 14
def cost_per_liter : ℕ := 3

-- Theorem statement
theorem truck_capacity :
  initial_fuel + (total_money - change) / cost_per_liter = 150 := by
  sorry

end NUMINAMATH_GPT_truck_capacity_l1002_100262


namespace NUMINAMATH_GPT_remainder_of_division_l1002_100213

def num : ℤ := 1346584
def divisor : ℤ := 137
def remainder : ℤ := 5

theorem remainder_of_division 
  (h : 0 <= divisor) (h' : divisor ≠ 0) : 
  num % divisor = remainder := 
sorry

end NUMINAMATH_GPT_remainder_of_division_l1002_100213


namespace NUMINAMATH_GPT_pencil_length_l1002_100230

theorem pencil_length
  (R P L : ℕ)
  (h1 : P = R + 3)
  (h2 : P = L - 2)
  (h3 : R + P + L = 29) :
  L = 12 :=
by
  sorry

end NUMINAMATH_GPT_pencil_length_l1002_100230


namespace NUMINAMATH_GPT_math_problem_l1002_100270

theorem math_problem
  (x : ℝ)
  (h : (1/2) * x - 300 = 350) :
  (x + 200) * 2 = 3000 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1002_100270


namespace NUMINAMATH_GPT_solution_system_inequalities_l1002_100256

theorem solution_system_inequalities (x : ℝ) : 
  (x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x) ↔ (2 < x ∧ x ≤ 4) := 
sorry

end NUMINAMATH_GPT_solution_system_inequalities_l1002_100256


namespace NUMINAMATH_GPT_total_balloons_after_destruction_l1002_100201

-- Define the initial numbers of balloons
def fredBalloons := 10.0
def samBalloons := 46.0
def destroyedBalloons := 16.0

-- Prove the total number of remaining balloons
theorem total_balloons_after_destruction : fredBalloons + samBalloons - destroyedBalloons = 40.0 :=
by
  sorry

end NUMINAMATH_GPT_total_balloons_after_destruction_l1002_100201


namespace NUMINAMATH_GPT_harriet_trip_time_to_B_l1002_100206

variables (D : ℝ) (t1 t2 : ℝ)

-- Definitions based on the given problem
def speed_to_b_town := 100
def speed_to_a_ville := 150
def total_time := 5

-- The condition for the total time for the trip
def total_trip_time_eq := t1 / speed_to_b_town + t2 / speed_to_a_ville = total_time

-- Prove that the time Harriet took to drive from A-ville to B-town is 3 hours.
theorem harriet_trip_time_to_B (h : total_trip_time_eq D D) : t1 = 3 :=
sorry

end NUMINAMATH_GPT_harriet_trip_time_to_B_l1002_100206


namespace NUMINAMATH_GPT_average_velocity_instantaneous_velocity_l1002_100255

noncomputable def s (t : ℝ) : ℝ := 8 - 3 * t^2

theorem average_velocity {Δt : ℝ} (h : Δt ≠ 0) :
  (s (1 + Δt) - s 1) / Δt = -6 - 3 * Δt :=
sorry

theorem instantaneous_velocity :
  deriv s 1 = -6 :=
sorry

end NUMINAMATH_GPT_average_velocity_instantaneous_velocity_l1002_100255


namespace NUMINAMATH_GPT_find_integers_correct_l1002_100240

noncomputable def find_integers (a b c d : ℤ) : Prop :=
  a + b + c = 6 ∧ a + b + d = 7 ∧ a + c + d = 8 ∧ b + c + d = 9

theorem find_integers_correct (a b c d : ℤ) (h : find_integers a b c d) : a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_integers_correct_l1002_100240


namespace NUMINAMATH_GPT_final_position_relative_total_fuel_needed_l1002_100219

noncomputable def navigation_records : List ℤ := [-7, 11, -6, 10, -5]

noncomputable def fuel_consumption_rate : ℝ := 0.5

theorem final_position_relative (records : List ℤ) : 
  (records.sum = 3) := by 
  sorry

theorem total_fuel_needed (records : List ℤ) (rate : ℝ) : 
  (rate * (records.map Int.natAbs).sum = 19.5) := by 
  sorry

#check final_position_relative navigation_records
#check total_fuel_needed navigation_records fuel_consumption_rate

end NUMINAMATH_GPT_final_position_relative_total_fuel_needed_l1002_100219


namespace NUMINAMATH_GPT_sum_of_consecutive_naturals_l1002_100207

theorem sum_of_consecutive_naturals (n : ℕ) : 
  ∃ S : ℕ, S = n * (n + 1) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_naturals_l1002_100207


namespace NUMINAMATH_GPT_range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l1002_100205

def quadratic_has_two_distinct_positive_roots (m : ℝ) : Prop :=
  4 * m^2 - 4 * (m + 2) > 0 ∧ -2 * m > 0 ∧ m + 2 > 0

def hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 3 < 0 ∧ 1 - 2 * m > 0

theorem range_of_m_given_q (m : ℝ) :
  hyperbola_with_foci_on_y_axis m → m < -3 :=
by
  sorry

theorem range_of_m_given_p_or_q_and_not_p_and_q (m : ℝ) :
  (quadratic_has_two_distinct_positive_roots m ∨ hyperbola_with_foci_on_y_axis m) ∧
  ¬(quadratic_has_two_distinct_positive_roots m ∧ hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l1002_100205


namespace NUMINAMATH_GPT_count_ab_bc_ca_l1002_100232

noncomputable def count_ways : ℕ :=
  (Nat.choose 9 3)

theorem count_ab_bc_ca (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) :
  (10 * a + b < 10 * b + c ∧ 10 * b + c < 10 * c + a) → count_ways = 84 :=
sorry

end NUMINAMATH_GPT_count_ab_bc_ca_l1002_100232


namespace NUMINAMATH_GPT_positive_difference_of_perimeters_l1002_100272

noncomputable def perimeter_figure1 : ℕ :=
  let outer_rectangle := 2 * (5 + 1)
  let inner_extension := 2 * (2 + 1)
  outer_rectangle + inner_extension

noncomputable def perimeter_figure2 : ℕ :=
  2 * (5 + 2)

theorem positive_difference_of_perimeters :
  (perimeter_figure1 - perimeter_figure2 = 4) :=
by
  let perimeter1 := perimeter_figure1
  let perimeter2 := perimeter_figure2
  sorry

end NUMINAMATH_GPT_positive_difference_of_perimeters_l1002_100272


namespace NUMINAMATH_GPT_inequality_l1002_100236

variable (a b m : ℝ)

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < m) (h4 : a < b) :
  a / b < (a + m) / (b + m) :=
by
  sorry

end NUMINAMATH_GPT_inequality_l1002_100236


namespace NUMINAMATH_GPT_different_tea_packets_or_miscalculation_l1002_100221

theorem different_tea_packets_or_miscalculation : 
  ∀ (n_1 n_2 : ℕ), 3 ≤ t_1 ∧ t_1 ≤ 4 ∧ 3 ≤ t_2 ∧ t_2 ≤ 4 ∧
  (74 = t_1 * x ∧ 105 = t_2 * y → x ≠ y) ∨ 
  (∃ (e_1 e_2 : ℕ), (e_1 + e_2 = 74) ∧ (e_1 + e_2 = 105) → false) :=
by
  -- Construction based on the provided mathematical problem
  sorry

end NUMINAMATH_GPT_different_tea_packets_or_miscalculation_l1002_100221


namespace NUMINAMATH_GPT_graph_of_equation_l1002_100216

theorem graph_of_equation :
  ∀ x y : ℝ, (2 * x - 3 * y) ^ 2 = 4 * x ^ 2 + 9 * y ^ 2 → (x = 0 ∨ y = 0) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_graph_of_equation_l1002_100216


namespace NUMINAMATH_GPT_inequality_solution_set_l1002_100260

theorem inequality_solution_set {a : ℝ} (x : ℝ) :
  (∀ x, (x - a) / (x^2 - 3 * x + 2) ≥ 0 ↔ (1 < x ∧ x ≤ a) ∨ (2 < x)) → (1 < a ∧ a < 2) :=
by 
  -- We would fill in the proof here. 
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1002_100260


namespace NUMINAMATH_GPT_prime_pattern_l1002_100225

theorem prime_pattern (n x : ℕ) (h1 : x = (10^n - 1) / 9) (h2 : Prime x) : Prime n :=
sorry

end NUMINAMATH_GPT_prime_pattern_l1002_100225


namespace NUMINAMATH_GPT_car_speed_in_second_hour_l1002_100288

theorem car_speed_in_second_hour (x : ℕ) : 84 = (98 + x) / 2 → x = 70 := 
sorry

end NUMINAMATH_GPT_car_speed_in_second_hour_l1002_100288


namespace NUMINAMATH_GPT_increased_percentage_l1002_100298

theorem increased_percentage (P : ℝ) (N : ℝ) (hN : N = 80) 
  (h : (N + (P / 100) * N) - (N - (25 / 100) * N) = 30) : P = 12.5 := 
by 
  sorry

end NUMINAMATH_GPT_increased_percentage_l1002_100298


namespace NUMINAMATH_GPT_HCF_48_99_l1002_100273

-- definitions and theorem stating the problem
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

theorem HCF_48_99 : HCF 48 99 = 3 :=
by
  sorry

end NUMINAMATH_GPT_HCF_48_99_l1002_100273


namespace NUMINAMATH_GPT_solveSystem_l1002_100244

variable {r p q x y z : ℝ}

theorem solveSystem :
  
  -- The given system of equations
  (x + r * y - q * z = 1) ∧
  (-r * x + y + p * z = r) ∧ 
  (q * x - p * y + z = -q) →

  -- Solution equivalence using determined
  x = (1 - r ^ 2 + p ^ 2 - q ^ 2) / (1 + r ^ 2 + p ^ 2 + q ^ 2) :=
by sorry

end NUMINAMATH_GPT_solveSystem_l1002_100244


namespace NUMINAMATH_GPT_greatest_divisor_of_consecutive_product_l1002_100203

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_consecutive_product_l1002_100203


namespace NUMINAMATH_GPT_greatest_good_t_l1002_100265

noncomputable def S (a t : ℕ) : Set ℕ := {x | ∃ n : ℕ, x = a + 1 + n ∧ n < t}

def is_good (S : Set ℕ) (k : ℕ) : Prop :=
∃ (coloring : ℕ → Fin k), ∀ (x y : ℕ), x ≠ y → x + y ∈ S → coloring x ≠ coloring y

theorem greatest_good_t {k : ℕ} (hk : k > 1) : ∃ t, ∀ a, is_good (S a t) k ∧ 
  ∀ t' > t, ¬ ∀ a, is_good (S a t') k := 
sorry

end NUMINAMATH_GPT_greatest_good_t_l1002_100265


namespace NUMINAMATH_GPT_infection_average_l1002_100253

theorem infection_average (x : ℕ) (h : 1 + x + x * (1 + x) = 196) : x = 13 :=
sorry

end NUMINAMATH_GPT_infection_average_l1002_100253


namespace NUMINAMATH_GPT_ratio_of_floors_l1002_100269

-- Define the number of floors of each building
def floors_building_A := 4
def floors_building_B := 4 + 9
def floors_building_C := 59

-- Prove the ratio of floors in Building C to Building B
theorem ratio_of_floors :
  floors_building_C / floors_building_B = 59 / 13 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_floors_l1002_100269


namespace NUMINAMATH_GPT_find_locus_of_T_l1002_100286

section Locus

variables {x y m : ℝ}
variable (M : ℝ × ℝ)

-- Condition: The equation of the ellipse
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1

-- Condition: Point P
def P := (1, 0)

-- Condition: M is any point on the ellipse, except A and B
def on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2 ∧ M ≠ (-2, 0) ∧ M ≠ (2, 0)

-- Condition: The intersection point N of line MP with the ellipse
def line_eq (m y : ℝ) := m * y + 1

-- Proposition: Locus of intersection point T of lines AM and BN
theorem find_locus_of_T 
  (hM : on_ellipse M)
  (hN : line_eq m M.2 = M.1)
  (hT : M.2 ≠ 0) :
  M.1 = 4 :=
sorry

end Locus

end NUMINAMATH_GPT_find_locus_of_T_l1002_100286


namespace NUMINAMATH_GPT_find_f_l1002_100251

theorem find_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 + x) :
  ∀ x : ℤ, f x = x^2 - x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_find_f_l1002_100251


namespace NUMINAMATH_GPT_total_cost_correct_l1002_100248

-- Condition C1: There are 13 hearts in a deck of 52 playing cards. 
def hearts_in_deck : ℕ := 13

-- Condition C2: The number of cows is twice the number of hearts.
def cows_in_Devonshire : ℕ := 2 * hearts_in_deck

-- Condition C3: Each cow is sold at $200.
def cost_per_cow : ℕ := 200

-- Question Q1: Calculate the total cost of the cows.
def total_cost_of_cows : ℕ := cows_in_Devonshire * cost_per_cow

-- Final statement we need to prove
theorem total_cost_correct : total_cost_of_cows = 5200 := by
  -- This will be proven in the proof body
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1002_100248


namespace NUMINAMATH_GPT_license_plate_count_l1002_100220

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let total_count := letters * (letters - 1) + letters
  total_count * digits = 6760 :=
by sorry

end NUMINAMATH_GPT_license_plate_count_l1002_100220


namespace NUMINAMATH_GPT_least_number_to_subtract_l1002_100211

theorem least_number_to_subtract {x : ℕ} (h : x = 13604) : 
    ∃ n : ℕ, n = 32 ∧ (13604 - n) % 87 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l1002_100211


namespace NUMINAMATH_GPT_lcm_hcf_relationship_l1002_100263

theorem lcm_hcf_relationship (a b : ℕ) (h_prod : a * b = 84942) (h_hcf : Nat.gcd a b = 33) : Nat.lcm a b = 2574 :=
by
  sorry

end NUMINAMATH_GPT_lcm_hcf_relationship_l1002_100263


namespace NUMINAMATH_GPT_problem_statement_l1002_100224

theorem problem_statement (x : ℝ) (h : 7 * x = 3) : 150 * (1 / x) = 350 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1002_100224


namespace NUMINAMATH_GPT_negation_of_existence_l1002_100235

theorem negation_of_existence (h: ¬ ∃ x : ℝ, x^2 + 1 < 0) : ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l1002_100235


namespace NUMINAMATH_GPT_problem_1_problem_2_l1002_100294

open Real

noncomputable def f (omega : ℝ) (x : ℝ) : ℝ := 
  (cos (omega * x) * cos (omega * x) + sqrt 3 * cos (omega * x) * sin (omega * x) - 1/2)

theorem problem_1 (ω : ℝ) (hω : ω > 0):
 (f ω x = sin (2 * x + π / 6)) ∧ 
 (∀ k : ℤ, ∀ x : ℝ, (-π / 3 + ↑k * π) ≤ x ∧ x ≤ (π / 6 + ↑k * π) → f ω x = sin (2 * x + π / 6)) :=
sorry

theorem problem_2 (A b S a : ℝ) (hA : A / 2 = π / 3)
  (hb : b = 1) (hS: S = sqrt 3) :
  a = sqrt 13 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1002_100294


namespace NUMINAMATH_GPT_ratio_blue_to_total_l1002_100276

theorem ratio_blue_to_total (total_marbles red_marbles green_marbles yellow_marbles blue_marbles : ℕ)
    (h_total : total_marbles = 164)
    (h_red : red_marbles = total_marbles / 4)
    (h_green : green_marbles = 27)
    (h_yellow : yellow_marbles = 14)
    (h_blue : blue_marbles = total_marbles - (red_marbles + green_marbles + yellow_marbles)) :
  blue_marbles / total_marbles = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_blue_to_total_l1002_100276


namespace NUMINAMATH_GPT_arithmetic_sequence_has_correct_number_of_terms_l1002_100278

theorem arithmetic_sequence_has_correct_number_of_terms :
  ∀ (a₁ d : ℤ) (n : ℕ), a₁ = 1 ∧ d = -2 ∧ (n : ℤ) = (a₁ + (n - 1 : ℕ) * d) → n = 46 := by
  intros a₁ d n
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_has_correct_number_of_terms_l1002_100278


namespace NUMINAMATH_GPT_angle_measure_l1002_100296

theorem angle_measure (x : ℝ) (h : 90 - x = 3 * (180 - x)) : x = 45 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1002_100296


namespace NUMINAMATH_GPT_problem_proof_l1002_100209

theorem problem_proof (x y : ℝ) (h_cond : (x + 3)^2 + |y - 2| = 0) : (x + y)^y = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1002_100209


namespace NUMINAMATH_GPT_crimson_valley_skirts_l1002_100297

theorem crimson_valley_skirts (e : ℕ) (a : ℕ) (s : ℕ) (p : ℕ) (c : ℕ) 
  (h1 : e = 120) 
  (h2 : a = 2 * e) 
  (h3 : s = 3 * a / 5) 
  (h4 : p = s / 4) 
  (h5 : c = p / 3) : 
  c = 12 := 
by 
  sorry

end NUMINAMATH_GPT_crimson_valley_skirts_l1002_100297


namespace NUMINAMATH_GPT_difference_of_x_y_l1002_100239

theorem difference_of_x_y :
  ∀ (x y : ℤ), x + y = 10 → x = 14 → x - y = 18 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_difference_of_x_y_l1002_100239


namespace NUMINAMATH_GPT_triangle_side_relation_l1002_100293

theorem triangle_side_relation
  (A B C : ℝ)
  (a b c : ℝ)
  (h : 3 * (Real.sin (A / 2)) * (Real.sin (B / 2)) * (Real.cos (C / 2)) + (Real.sin (3 * A / 2)) * (Real.sin (3 * B / 2)) * (Real.cos (3 * C / 2)) = 0)
  (law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  a^3 + b^3 = c^3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_relation_l1002_100293


namespace NUMINAMATH_GPT_smallest_multiple_of_6_and_15_l1002_100200

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end NUMINAMATH_GPT_smallest_multiple_of_6_and_15_l1002_100200


namespace NUMINAMATH_GPT_value_of_b_minus_a_l1002_100242

theorem value_of_b_minus_a (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) : b - a = 2 ∨ b - a = 4 :=
sorry

end NUMINAMATH_GPT_value_of_b_minus_a_l1002_100242


namespace NUMINAMATH_GPT_extra_food_needed_l1002_100228

theorem extra_food_needed (f1 f2 : ℝ) (h1 : f1 = 0.5) (h2 : f2 = 0.9) :
  f2 - f1 = 0.4 :=
by sorry

end NUMINAMATH_GPT_extra_food_needed_l1002_100228


namespace NUMINAMATH_GPT_maria_savings_percentage_is_33_l1002_100202

noncomputable def regular_price : ℝ := 60
noncomputable def second_pair_price : ℝ := regular_price - (0.4 * regular_price)
noncomputable def third_pair_price : ℝ := regular_price - (0.6 * regular_price)
noncomputable def total_regular_price : ℝ := 3 * regular_price
noncomputable def total_discounted_price : ℝ := regular_price + second_pair_price + third_pair_price
noncomputable def savings : ℝ := total_regular_price - total_discounted_price
noncomputable def savings_percentage : ℝ := (savings / total_regular_price) * 100

theorem maria_savings_percentage_is_33 :
  savings_percentage = 33 :=
by
  sorry

end NUMINAMATH_GPT_maria_savings_percentage_is_33_l1002_100202


namespace NUMINAMATH_GPT_sequence_properties_sum_Tn_l1002_100250

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 2^(n - 1)
noncomputable def c_n (n : ℕ) : ℤ := (2 * n - 1) / 2^(n - 1)
noncomputable def T_n (n : ℕ) : ℤ := 6 - (2 * n + 3) / 2^(n - 1)

theorem sequence_properties : (d = 2) → (S₁₀ = 100) → 
  (∀ n : ℕ, a_n n = 2 * n - 1) ∧ (∀ n : ℕ, b_n n = 2^(n - 1)) := by
  sorry

theorem sum_Tn : (d > 1) → 
  (∀ n : ℕ, T_n n = 6 - (2 * n + 3) / 2^(n - 1)) := by
  sorry

end NUMINAMATH_GPT_sequence_properties_sum_Tn_l1002_100250


namespace NUMINAMATH_GPT_ramon_3_enchiladas_4_tacos_cost_l1002_100299

theorem ramon_3_enchiladas_4_tacos_cost :
  ∃ (e t : ℝ), 2 * e + 3 * t = 2.50 ∧ 3 * e + 2 * t = 2.70 ∧ 3 * e + 4 * t = 3.54 :=
by {
  sorry
}

end NUMINAMATH_GPT_ramon_3_enchiladas_4_tacos_cost_l1002_100299


namespace NUMINAMATH_GPT_Trisha_total_distance_l1002_100285

theorem Trisha_total_distance :
  let d1 := 0.11  -- hotel to postcard shop
  let d2 := 0.11  -- postcard shop back to hotel
  let d3 := 1.52  -- hotel to T-shirt shop
  let d4 := 0.45  -- T-shirt shop to hat shop
  let d5 := 0.87  -- hat shop to purse shop
  let d6 := 2.32  -- purse shop back to hotel
  d1 + d2 + d3 + d4 + d5 + d6 = 5.38 :=
by
  sorry

end NUMINAMATH_GPT_Trisha_total_distance_l1002_100285


namespace NUMINAMATH_GPT_jamesons_sword_length_l1002_100241

theorem jamesons_sword_length (c j j' : ℕ) (hC: c = 15) 
  (hJ: j = c + 23) (hJJ: j' = j - 5) : 
  j' = 2 * c + 3 := by 
  sorry

end NUMINAMATH_GPT_jamesons_sword_length_l1002_100241


namespace NUMINAMATH_GPT_letters_identity_l1002_100246

def identity_of_letters (first second third : ℕ) : Prop :=
  (first, second, third) = (1, 0, 1)

theorem letters_identity (first second third : ℕ) :
  first + second + third = 1 →
  (first = 1 → 1 ≠ first + second) →
  (second = 0 → first + second < 2) →
  (third = 0 → first + second = 1) →
  identity_of_letters first second third :=
by sorry

end NUMINAMATH_GPT_letters_identity_l1002_100246


namespace NUMINAMATH_GPT_find_unit_prices_minimize_cost_l1002_100295

-- Definitions for the given prices and conditions
def cypress_price := 200
def pine_price := 150

def cost_eq1 (x y : ℕ) : Prop := 2 * x + 3 * y = 850
def cost_eq2 (x y : ℕ) : Prop := 3 * x + 2 * y = 900

-- Proving the unit prices of cypress and pine trees
theorem find_unit_prices (x y : ℕ) (h1 : cost_eq1 x y) (h2 : cost_eq2 x y) :
  x = cypress_price ∧ y = pine_price :=
sorry

-- Definitions for the number of trees and their costs
def total_trees := 80
def cypress_min (a : ℕ) : Prop := a ≥ 2 * (total_trees - a)
def total_cost (a : ℕ) : ℕ := 200 * a + 150 * (total_trees - a)

-- Conditions given for minimizing the cost
theorem minimize_cost (a : ℕ) (h1 : cypress_min a) : 
  a = 54 ∧ (total_trees - a) = 26 ∧ total_cost a = 14700 :=
sorry

end NUMINAMATH_GPT_find_unit_prices_minimize_cost_l1002_100295


namespace NUMINAMATH_GPT_lcm_of_numbers_l1002_100218

theorem lcm_of_numbers (a b : ℕ) (L : ℕ) 
  (h1 : a + b = 55) 
  (h2 : Nat.gcd a b = 5) 
  (h3 : (1 / (a : ℝ)) + (1 / (b : ℝ)) = 0.09166666666666666) : (Nat.lcm a b = 120) := 
sorry

end NUMINAMATH_GPT_lcm_of_numbers_l1002_100218


namespace NUMINAMATH_GPT_parabola_directrix_l1002_100277

theorem parabola_directrix (x y : ℝ) (h : x^2 = 12 * y) : y = -3 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l1002_100277


namespace NUMINAMATH_GPT_chosen_number_l1002_100275

theorem chosen_number (x : ℤ) (h : x / 12 - 240 = 8) : x = 2976 :=
by sorry

end NUMINAMATH_GPT_chosen_number_l1002_100275


namespace NUMINAMATH_GPT_total_hens_and_cows_l1002_100215

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end NUMINAMATH_GPT_total_hens_and_cows_l1002_100215


namespace NUMINAMATH_GPT_single_cow_single_bag_l1002_100217

-- Definitions given in the problem conditions
def cows : ℕ := 26
def bags : ℕ := 26
def days : ℕ := 26

-- Statement to be proved
theorem single_cow_single_bag : (1 : ℕ) = 26 := sorry

end NUMINAMATH_GPT_single_cow_single_bag_l1002_100217


namespace NUMINAMATH_GPT_A_and_C_mutually_exclusive_l1002_100271

/-- Definitions for the problem conditions. -/
def A (all_non_defective : Prop) : Prop := all_non_defective
def B (all_defective : Prop) : Prop := all_defective
def C (at_least_one_defective : Prop) : Prop := at_least_one_defective

/-- Theorem stating that A and C are mutually exclusive. -/
theorem A_and_C_mutually_exclusive (all_non_defective at_least_one_defective : Prop) :
  A all_non_defective ∧ C at_least_one_defective → false :=
  sorry

end NUMINAMATH_GPT_A_and_C_mutually_exclusive_l1002_100271


namespace NUMINAMATH_GPT_recurring_decimal_sum_is_13_over_33_l1002_100266

noncomputable def recurring_decimal_sum : ℚ :=
  let x := 1/3 -- 0.\overline{3}
  let y := 2/33 -- 0.\overline{06}
  x + y

theorem recurring_decimal_sum_is_13_over_33 : recurring_decimal_sum = 13/33 := by
  sorry

end NUMINAMATH_GPT_recurring_decimal_sum_is_13_over_33_l1002_100266


namespace NUMINAMATH_GPT_winter_sales_l1002_100229

theorem winter_sales (T : ℕ) (spring_summer_sales : ℕ) (fall_sales : ℕ) (winter_sales : ℕ) 
  (h1 : T = 20) 
  (h2 : spring_summer_sales = 12) 
  (h3 : fall_sales = 4) 
  (h4 : T = spring_summer_sales + fall_sales + winter_sales) : 
     winter_sales = 4 := 
by 
  rw [h1, h2, h3] at h4
  linarith


end NUMINAMATH_GPT_winter_sales_l1002_100229


namespace NUMINAMATH_GPT_sequence_relation_l1002_100238

theorem sequence_relation
  (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end NUMINAMATH_GPT_sequence_relation_l1002_100238


namespace NUMINAMATH_GPT_last_month_games_l1002_100237

-- Definitions and conditions
def this_month := 9
def next_month := 7
def total_games := 24

-- Question to prove
theorem last_month_games : total_games - (this_month + next_month) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_last_month_games_l1002_100237


namespace NUMINAMATH_GPT_area_of_square_with_diagonal_l1002_100252

theorem area_of_square_with_diagonal (c : ℝ) : 
  (∃ (s : ℝ), 2 * s^2 = c^4) → (∃ (A : ℝ), A = (c^4 / 2)) :=
  by
    sorry

end NUMINAMATH_GPT_area_of_square_with_diagonal_l1002_100252


namespace NUMINAMATH_GPT_proof_option_b_and_c_l1002_100261

variable (a b c : ℝ)

theorem proof_option_b_and_c (h₀ : a > b) (h₁ : b > 0) (h₂ : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1 / a > b^2 - 1 / b) :=
by
  sorry

end NUMINAMATH_GPT_proof_option_b_and_c_l1002_100261


namespace NUMINAMATH_GPT_percentage_increase_sides_l1002_100243

theorem percentage_increase_sides (P : ℝ) :
  (1 + P/100) ^ 2 = 1.3225 → P = 15 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_sides_l1002_100243


namespace NUMINAMATH_GPT_algebraic_expression_value_l1002_100279

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 11 = -5 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1002_100279


namespace NUMINAMATH_GPT_domain_of_composed_function_l1002_100212

theorem domain_of_composed_function {f : ℝ → ℝ} (h : ∀ x, -1 < x ∧ x < 1 → f x ∈ Set.Ioo (-1:ℝ) 1) :
  ∀ x, 0 < x ∧ x < 1 → f (2*x-1) ∈ Set.Ioo (-1:ℝ) 1 := by
  sorry

end NUMINAMATH_GPT_domain_of_composed_function_l1002_100212


namespace NUMINAMATH_GPT_yvette_sundae_cost_l1002_100257

noncomputable def cost_friends : ℝ := 7.50 + 10.00 + 8.50
noncomputable def final_bill : ℝ := 42.00
noncomputable def tip_percentage : ℝ := 0.20
noncomputable def tip_amount : ℝ := tip_percentage * final_bill

theorem yvette_sundae_cost : 
  final_bill - (cost_friends + tip_amount) = 7.60 := by
  sorry

end NUMINAMATH_GPT_yvette_sundae_cost_l1002_100257


namespace NUMINAMATH_GPT_magician_hat_probability_l1002_100281

def total_arrangements : ℕ := Nat.choose 6 2
def favorable_arrangements : ℕ := Nat.choose 5 1
def probability_red_chips_drawn_first : ℚ := favorable_arrangements / total_arrangements

theorem magician_hat_probability :
  probability_red_chips_drawn_first = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_magician_hat_probability_l1002_100281


namespace NUMINAMATH_GPT_Chad_saves_40_percent_of_his_earnings_l1002_100264

theorem Chad_saves_40_percent_of_his_earnings :
  let earnings_mow := 600
  let earnings_birthday := 250
  let earnings_games := 150
  let earnings_oddjobs := 150
  let amount_saved := 460
  (amount_saved / (earnings_mow + earnings_birthday + earnings_games + earnings_oddjobs) * 100) = 40 :=
by
  sorry

end NUMINAMATH_GPT_Chad_saves_40_percent_of_his_earnings_l1002_100264


namespace NUMINAMATH_GPT_children_on_bus_l1002_100234

theorem children_on_bus (initial_children additional_children total_children : ℕ) (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = initial_children + additional_children → total_children = 64 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_children_on_bus_l1002_100234


namespace NUMINAMATH_GPT_area_of_triangle_DOE_l1002_100292

-- Definitions of points D, O, and E
def D (p : ℝ) : ℝ × ℝ := (0, p)
def O : ℝ × ℝ := (0, 0)
def E : ℝ × ℝ := (15, 0)

-- Theorem statement
theorem area_of_triangle_DOE (p : ℝ) : 
  let base := 15
  let height := p
  let area := (1/2) * base * height
  area = (15 * p) / 2 :=
by sorry

end NUMINAMATH_GPT_area_of_triangle_DOE_l1002_100292


namespace NUMINAMATH_GPT_root_of_quadratic_l1002_100259

theorem root_of_quadratic :
  (∀ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 → x = 5 ∨ x = -6.5) :=
sorry

end NUMINAMATH_GPT_root_of_quadratic_l1002_100259


namespace NUMINAMATH_GPT_investment_amount_l1002_100258

noncomputable def calculate_principal (A : ℕ) (r t : ℝ) (n : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem investment_amount (A : ℕ) (r t : ℝ) (n P : ℕ) :
  A = 70000 → r = 0.08 → t = 5 → n = 12 →
  P = 46994 →
  calculate_principal A r t n = P :=
by
  intros hA hr ht hn hP
  rw [hA, hr, ht, hn, hP]
  sorry

end NUMINAMATH_GPT_investment_amount_l1002_100258


namespace NUMINAMATH_GPT_prime_p_impplies_p_eq_3_l1002_100223

theorem prime_p_impplies_p_eq_3 (p : ℕ) (hp : Prime p) (hp2 : Prime (p^2 + 2)) : p = 3 :=
sorry

end NUMINAMATH_GPT_prime_p_impplies_p_eq_3_l1002_100223


namespace NUMINAMATH_GPT_polyhedron_has_triangular_face_l1002_100283

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end NUMINAMATH_GPT_polyhedron_has_triangular_face_l1002_100283


namespace NUMINAMATH_GPT_percentage_of_truth_speakers_l1002_100247

theorem percentage_of_truth_speakers
  (L : ℝ) (hL: L = 0.2)
  (B : ℝ) (hB: B = 0.1)
  (prob_truth_or_lies : ℝ) (hProb: prob_truth_or_lies = 0.4)
  (T : ℝ)
: T = prob_truth_or_lies - L + B :=
sorry

end NUMINAMATH_GPT_percentage_of_truth_speakers_l1002_100247


namespace NUMINAMATH_GPT_job_pay_per_pound_l1002_100226

def p := 2
def M := 8 -- Monday
def T := 3 * M -- Tuesday
def W := 0 -- Wednesday
def R := 18 -- Thursday
def total_picked := M + T + W + R -- total berries picked
def money := 100 -- total money wanted

theorem job_pay_per_pound :
  total_picked = 50 → p = money / total_picked :=
by
  intro h
  rw [h]
  norm_num
  exact rfl

end NUMINAMATH_GPT_job_pay_per_pound_l1002_100226


namespace NUMINAMATH_GPT_work_completion_time_l1002_100249

theorem work_completion_time (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hAC : A + C = 1 / 2) :
  1 / (B + C) = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_work_completion_time_l1002_100249


namespace NUMINAMATH_GPT_find_A_l1002_100245

def divisible_by(a b : ℕ) := b % a = 0

def valid_digit_A (A : ℕ) : Prop := (A = 0 ∨ A = 2 ∨ A = 4 ∨ A = 6 ∨ A = 8) ∧ divisible_by A 75

theorem find_A : ∃! A : ℕ, valid_digit_A A :=
by {
  sorry
}

end NUMINAMATH_GPT_find_A_l1002_100245


namespace NUMINAMATH_GPT_haji_mother_tough_weeks_l1002_100227

/-- Let's define all the conditions: -/
def tough_week_revenue : ℕ := 800
def good_week_revenue : ℕ := 2 * tough_week_revenue
def number_of_good_weeks : ℕ := 5
def total_revenue : ℕ := 10400

/-- Let's define the proofs for intermediate steps: -/
def good_weeks_revenue : ℕ := number_of_good_weeks * good_week_revenue
def tough_weeks_revenue : ℕ := total_revenue - good_weeks_revenue
def number_of_tough_weeks : ℕ := tough_weeks_revenue / tough_week_revenue

/-- Now the theorem which states that the number of tough weeks is 3. -/
theorem haji_mother_tough_weeks : number_of_tough_weeks = 3 := by
  sorry

end NUMINAMATH_GPT_haji_mother_tough_weeks_l1002_100227


namespace NUMINAMATH_GPT_total_cost_sean_bought_l1002_100287

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_sean_bought_l1002_100287


namespace NUMINAMATH_GPT_max_distance_l1002_100291

noncomputable def starting_cost : ℝ := 10
noncomputable def additional_cost_per_km : ℝ := 1.5
noncomputable def round_up : ℝ := 1
noncomputable def total_fare : ℝ := 19

theorem max_distance (x : ℝ) : (starting_cost + additional_cost_per_km * (x - 4)) = total_fare → x = 10 :=
by sorry

end NUMINAMATH_GPT_max_distance_l1002_100291


namespace NUMINAMATH_GPT_sum_of_squares_l1002_100268

theorem sum_of_squares (x y : ℝ) (h1 : y + 6 = (x - 3)^2) (h2 : x + 6 = (y - 3)^2) (hxy : x ≠ y) : x^2 + y^2 = 43 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l1002_100268
