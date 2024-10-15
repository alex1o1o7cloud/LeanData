import Mathlib

namespace NUMINAMATH_GPT_fraction_of_work_left_l848_84826

theorem fraction_of_work_left (a_days b_days : ℕ) (together_days : ℕ) 
    (h_a : a_days = 15) (h_b : b_days = 20) (h_together : together_days = 4) : 
    (1 - together_days * ((1/a_days : ℚ) + (1/b_days))) = 8/15 := by
  sorry

end NUMINAMATH_GPT_fraction_of_work_left_l848_84826


namespace NUMINAMATH_GPT_dogs_with_no_accessories_l848_84865

theorem dogs_with_no_accessories :
  let total := 120
  let tags := 60
  let flea_collars := 50
  let harnesses := 30
  let tags_and_flea_collars := 20
  let tags_and_harnesses := 15
  let flea_collars_and_harnesses := 10
  let all_three := 5
  total - (tags + flea_collars + harnesses - tags_and_flea_collars - tags_and_harnesses - flea_collars_and_harnesses + all_three) = 25 := by
  sorry

end NUMINAMATH_GPT_dogs_with_no_accessories_l848_84865


namespace NUMINAMATH_GPT_balloon_difference_l848_84834

theorem balloon_difference (x y : ℝ) (h1 : x = 2 * y - 3) (h2 : y = x / 4 + 1) : x - y = -2.5 :=
by 
  sorry

end NUMINAMATH_GPT_balloon_difference_l848_84834


namespace NUMINAMATH_GPT_sqrt3_times_3_minus_sqrt3_bound_l848_84846

theorem sqrt3_times_3_minus_sqrt3_bound : 2 < (Real.sqrt 3) * (3 - (Real.sqrt 3)) ∧ (Real.sqrt 3) * (3 - (Real.sqrt 3)) < 3 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt3_times_3_minus_sqrt3_bound_l848_84846


namespace NUMINAMATH_GPT_tank_min_cost_l848_84814

/-- A factory plans to build an open-top rectangular tank with one fixed side length of 8m and a maximum water capacity of 72m³. The cost 
of constructing the bottom and the walls of the tank are $2a$ yuan per square meter and $a$ yuan per square meter, respectively. 
We need to prove the optimal dimensions and the minimum construction cost.
-/
theorem tank_min_cost 
  (a : ℝ)   -- cost multiplier
  (b h : ℝ) -- dimensions of the tank
  (volume_constraint : 8 * b * h = 72) : 
  (b = 3) ∧ (h = 3) ∧ (16 * a * (b + h) + 18 * a = 114 * a) :=
by
  sorry

end NUMINAMATH_GPT_tank_min_cost_l848_84814


namespace NUMINAMATH_GPT_abs_x_plus_1_plus_abs_x_minus_3_ge_a_l848_84842

theorem abs_x_plus_1_plus_abs_x_minus_3_ge_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_x_plus_1_plus_abs_x_minus_3_ge_a_l848_84842


namespace NUMINAMATH_GPT_initial_distance_l848_84855

def relative_speed (v1 v2 : ℝ) : ℝ := v1 + v2

def total_distance (rel_speed time : ℝ) : ℝ := rel_speed * time

theorem initial_distance (v1 v2 time : ℝ) : (v1 = 1.6) → (v2 = 1.9) → 
                                            (time = 100) →
                                            total_distance (relative_speed v1 v2) time = 350 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [relative_speed, total_distance]
  sorry

end NUMINAMATH_GPT_initial_distance_l848_84855


namespace NUMINAMATH_GPT_new_average_marks_l848_84894

theorem new_average_marks
  (orig_avg : ℕ) (num_papers : ℕ)
  (add_geography : ℕ) (add_history : ℕ)
  (H_orig_avg : orig_avg = 63)
  (H_num_papers : num_papers = 11)
  (H_add_geography : add_geography = 20)
  (H_add_history : add_history = 2) :
  (orig_avg * num_ppapers + add_geography + add_history) / num_papers = 65 :=
by
  -- Here would be the proof steps
  sorry

end NUMINAMATH_GPT_new_average_marks_l848_84894


namespace NUMINAMATH_GPT_find_first_number_l848_84882

variable (x y : ℕ)

theorem find_first_number (h1 : y = 11) (h2 : x + (y + 3) = 19) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_l848_84882


namespace NUMINAMATH_GPT_sum_of_digits_S_l848_84807

-- Define S as 10^2021 - 2021
def S : ℕ := 10^2021 - 2021

-- Define function to calculate sum of digits of a given number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum 

theorem sum_of_digits_S :
  sum_of_digits S = 18185 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_S_l848_84807


namespace NUMINAMATH_GPT_totalNutsInCar_l848_84830

-- Definitions based on the conditions
def busySquirrelNutsPerDay : Nat := 30
def busySquirrelDays : Nat := 35
def numberOfBusySquirrels : Nat := 2

def lazySquirrelNutsPerDay : Nat := 20
def lazySquirrelDays : Nat := 40
def numberOfLazySquirrels : Nat := 3

def sleepySquirrelNutsPerDay : Nat := 10
def sleepySquirrelDays : Nat := 45
def numberOfSleepySquirrels : Nat := 1

-- Calculate the total number of nuts stored by each type of squirrels
def totalNutsStoredByBusySquirrels : Nat := numberOfBusySquirrels * (busySquirrelNutsPerDay * busySquirrelDays)
def totalNutsStoredByLazySquirrels : Nat := numberOfLazySquirrels * (lazySquirrelNutsPerDay * lazySquirrelDays)
def totalNutsStoredBySleepySquirrel : Nat := numberOfSleepySquirrels * (sleepySquirrelNutsPerDay * sleepySquirrelDays)

-- The final theorem to prove
theorem totalNutsInCar : totalNutsStoredByBusySquirrels + totalNutsStoredByLazySquirrels + totalNutsStoredBySleepySquirrel = 4950 := by
  sorry

end NUMINAMATH_GPT_totalNutsInCar_l848_84830


namespace NUMINAMATH_GPT_sum_of_numbers_l848_84810

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : (a + b + c) / 3 = a + 20) 
  (h₂ : (a + b + c) / 3 = c - 30) 
  (h₃ : b = 10) : 
  a + b + c = 60 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l848_84810


namespace NUMINAMATH_GPT_similar_triangles_height_l848_84823

theorem similar_triangles_height (h₁ h₂ : ℝ) (a₁ a₂ : ℝ) 
  (ratio_area : a₁ / a₂ = 1 / 9) (height_small : h₁ = 4) :
  h₂ = 12 :=
sorry

end NUMINAMATH_GPT_similar_triangles_height_l848_84823


namespace NUMINAMATH_GPT_bruce_paid_amount_l848_84840

noncomputable def total_amount_paid :=
  let grapes_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let strawberries_cost := 4 * 90
  let total_cost := grapes_cost + mangoes_cost + oranges_cost + strawberries_cost
  let discount := 0.10 * total_cost
  let discounted_total := total_cost - discount
  let tax := 0.05 * discounted_total
  let final_amount := discounted_total + tax
  final_amount

theorem bruce_paid_amount :
  total_amount_paid = 1526.18 :=
by
  sorry

end NUMINAMATH_GPT_bruce_paid_amount_l848_84840


namespace NUMINAMATH_GPT_simplify_fraction_l848_84884

theorem simplify_fraction : (75 : ℚ) / (100 : ℚ) = (3 : ℚ) / (4 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l848_84884


namespace NUMINAMATH_GPT_sample_size_l848_84831

theorem sample_size (k n : ℕ) (h_ratio : 4 * k + k + 5 * k = n) 
  (h_middle_aged : 10 * (4 + 1 + 5) = n) : n = 100 := 
by
  sorry

end NUMINAMATH_GPT_sample_size_l848_84831


namespace NUMINAMATH_GPT_tank_capacity_l848_84841

noncomputable def leak_rate (C : ℝ) := C / 6
noncomputable def inlet_rate := 240
noncomputable def net_emptying_rate (C : ℝ) := C / 8

theorem tank_capacity : ∀ (C : ℝ), 
  (inlet_rate - leak_rate C = net_emptying_rate C) → 
  C = 5760 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_tank_capacity_l848_84841


namespace NUMINAMATH_GPT_increasing_interval_of_f_on_0_pi_l848_84892

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - Real.pi / 4)

theorem increasing_interval_of_f_on_0_pi {ω : ℝ} (hω : ω > 0)
  (h_symmetry : ∀ x, f ω x = g x) :
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ ∀ x1 x2, (0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ Real.pi) → f ω x1 < f ω x2} = 
  {x : ℝ | 0 ≤ x ∧ x ≤ Real.pi / 8} :=
sorry

end NUMINAMATH_GPT_increasing_interval_of_f_on_0_pi_l848_84892


namespace NUMINAMATH_GPT_tonya_hamburgers_to_beat_winner_l848_84850

-- Given conditions
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Calculate the number of hamburgers eaten last year
def hamburgers_eaten_last_year : ℕ := ounces_eaten_last_year / ounces_per_hamburger

-- Prove the number of hamburgers Tonya needs to eat to beat last year's winner
theorem tonya_hamburgers_to_beat_winner : 
  hamburgers_eaten_last_year + 1 = 22 :=
by
  -- It remains to be proven
  sorry

end NUMINAMATH_GPT_tonya_hamburgers_to_beat_winner_l848_84850


namespace NUMINAMATH_GPT_hilary_ears_per_stalk_l848_84863

-- Define the given conditions
def num_stalks : ℕ := 108
def kernels_per_ear_half1 : ℕ := 500
def kernels_per_ear_half2 : ℕ := 600
def total_kernels_to_shuck : ℕ := 237600

-- Define the number of ears of corn per stalk as the variable to prove
def ears_of_corn_per_stalk : ℕ := 4

-- The proof problem statement
theorem hilary_ears_per_stalk :
  (54 * ears_of_corn_per_stalk * kernels_per_ear_half1) + (54 * ears_of_corn_per_stalk * kernels_per_ear_half2) = total_kernels_to_shuck :=
by
  sorry

end NUMINAMATH_GPT_hilary_ears_per_stalk_l848_84863


namespace NUMINAMATH_GPT_problem_l848_84887

variable (f : ℝ → ℝ)

-- Given condition
axiom h : ∀ x : ℝ, f (1 / x) = 1 / (x + 1)

-- Prove that f(2) = 2/3
theorem problem : f 2 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_problem_l848_84887


namespace NUMINAMATH_GPT_max_min_f_l848_84837

-- Defining a and the set A
def a : ℤ := 2001

def A : Set (ℤ × ℤ) := {p | p.snd ≠ 0 ∧ p.fst < 2 * a ∧ (2 * p.snd) ∣ ((2 * a * p.fst) - (p.fst * p.fst) + (p.snd * p.snd)) ∧ ((p.snd * p.snd) - (p.fst * p.fst) + (2 * p.fst * p.snd) ≤ (2 * a * (p.snd - p.fst)))}

-- Defining the function f
def f (m n : ℤ): ℤ := (2 * a * m - m * m - m * n) / n

-- Main theorem: Proving that the maximum and minimum values of f over A are 3750 and 2 respectively
theorem max_min_f : 
  ∃ p ∈ A, f p.fst p.snd = 3750 ∧
  ∃ q ∈ A, f q.fst q.snd = 2 :=
sorry

end NUMINAMATH_GPT_max_min_f_l848_84837


namespace NUMINAMATH_GPT_find_a_l848_84808

-- Given conditions
def expand_term (a b : ℝ) (r : ℕ) : ℝ :=
  (Nat.choose 7 r) * (a ^ (7 - r)) * (b ^ r)

def coefficient_condition (a : ℝ) : Prop :=
  expand_term a 1 7 * 1 = 1

-- Main statement to prove
theorem find_a (a : ℝ) : coefficient_condition a → a = 1 / 7 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_a_l848_84808


namespace NUMINAMATH_GPT_positive_integer_solutions_l848_84874

theorem positive_integer_solutions (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) :
  1 + 2^x + 2^(2*x+1) = y^n ↔ 
  (x = 4 ∧ y = 23 ∧ n = 2) ∨ (∃ t : ℕ, 0 < t ∧ x = t ∧ y = 1 + 2^t + 2^(2*t+1) ∧ n = 1) :=
sorry

end NUMINAMATH_GPT_positive_integer_solutions_l848_84874


namespace NUMINAMATH_GPT_tangent_line_properties_l848_84824

noncomputable def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem tangent_line_properties (a b : ℝ) :
  (∀ x : ℝ, curve 0 a b = b) →
  (∀ x : ℝ, x - (curve x a b - b) + 1 = 0 → (∀ x : ℝ, 2*0 + a = 1)) →
  a + b = 2 :=
by
  intros h_curve h_tangent
  have h_b : b = 1 := by sorry
  have h_a : a = 1 := by sorry
  rw [h_a, h_b]
  norm_num

end NUMINAMATH_GPT_tangent_line_properties_l848_84824


namespace NUMINAMATH_GPT_frog_paths_l848_84868

theorem frog_paths (n : ℕ) : (∃ e_2n e_2n_minus_1 : ℕ,
  e_2n_minus_1 = 0 ∧
  e_2n = (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))) :=
by {
  sorry
}

end NUMINAMATH_GPT_frog_paths_l848_84868


namespace NUMINAMATH_GPT_eq_pow_four_l848_84891

theorem eq_pow_four (a b : ℝ) (h : a = b + 1) : a^4 = b^4 → a = 1/2 ∧ b = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_eq_pow_four_l848_84891


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l848_84869

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 15 = 90) :
  a 8 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l848_84869


namespace NUMINAMATH_GPT_variance_of_set_l848_84800

theorem variance_of_set (x : ℝ) (h : (-1 + x + 0 + 1 - 1)/5 = 0) : 
  (1/5) * ( (-1)^2 + (x)^2 + 0^2 + 1^2 + (-1)^2 ) = 0.8 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_variance_of_set_l848_84800


namespace NUMINAMATH_GPT_solution_set_of_inequality_l848_84890

open Set Real

theorem solution_set_of_inequality :
  {x : ℝ | sqrt (x + 3) > 3 - x} = {x : ℝ | 1 < x} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l848_84890


namespace NUMINAMATH_GPT_negation_of_universal_statement_l848_84805

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l848_84805


namespace NUMINAMATH_GPT_part1_part2_l848_84888

def custom_operation (a b : ℝ) : ℝ := a^2 + 2*a*b

theorem part1 : custom_operation 2 3 = 16 :=
by sorry

theorem part2 (x : ℝ) (h : custom_operation (-2) x = -2 + x) : x = 6 / 5 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l848_84888


namespace NUMINAMATH_GPT_set_complement_intersection_l848_84886

variable (U : Set ℕ) (M N : Set ℕ)

theorem set_complement_intersection
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {1, 4, 5})
  (hN : N = {2, 3}) :
  ((U \ N) ∩ M) = {1, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_set_complement_intersection_l848_84886


namespace NUMINAMATH_GPT_find_percentage_l848_84835

theorem find_percentage (p : ℝ) (h : (p / 100) * 8 = 0.06) : p = 0.75 := 
by 
  sorry

end NUMINAMATH_GPT_find_percentage_l848_84835


namespace NUMINAMATH_GPT_unique_positive_x_eq_3_l848_84821

theorem unique_positive_x_eq_3 (x : ℝ) (h_pos : 0 < x) (h_eq : x + 17 = 60 * (1 / x)) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_x_eq_3_l848_84821


namespace NUMINAMATH_GPT_ratio_of_areas_l848_84893

variables (s : ℝ)

def side_length_square := s
def longer_side_rect := 1.2 * s
def shorter_side_rect := 0.8 * s

noncomputable def area_rectangle := longer_side_rect s * shorter_side_rect s
noncomputable def area_triangle := (1 / 2) * (longer_side_rect s * shorter_side_rect s)

theorem ratio_of_areas :
  (area_triangle s) / (area_rectangle s) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l848_84893


namespace NUMINAMATH_GPT_exchange_candies_l848_84897

-- Define the problem conditions and calculate the required values
def chocolates := 7
def caramels := 9
def exchange := 5

-- Combinatorial function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem exchange_candies (h1 : chocolates = 7) (h2 : caramels = 9) (h3 : exchange = 5) :
  binomial chocolates exchange * binomial caramels exchange = 2646 := by
  sorry

end NUMINAMATH_GPT_exchange_candies_l848_84897


namespace NUMINAMATH_GPT_range_of_a_l848_84819

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

def sibling_point_pair (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = f a A.1 ∧ B.2 = f a B.1 ∧ A.1 = -B.1 ∧ A.2 = -B.2

theorem range_of_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, sibling_point_pair a A B) ↔ a > 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l848_84819


namespace NUMINAMATH_GPT_no_two_digit_prime_with_digit_sum_9_l848_84851

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end NUMINAMATH_GPT_no_two_digit_prime_with_digit_sum_9_l848_84851


namespace NUMINAMATH_GPT_complete_square_identity_l848_84817

theorem complete_square_identity (x d e : ℤ) (h : x^2 - 10 * x + 15 = 0) :
  (x + d)^2 = e → d + e = 5 :=
by
  intros hde
  sorry

end NUMINAMATH_GPT_complete_square_identity_l848_84817


namespace NUMINAMATH_GPT_abs_p_minus_1_ge_2_l848_84867

theorem abs_p_minus_1_ge_2 (p : ℝ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 1)
  (h₁ : a 1 = p)
  (h₂ : a 2 = p * (p - 1))
  (h₃ : ∀ n : ℕ, a (n + 3) = p * a (n + 2) - p * a (n + 1) + a n)
  (h₄ : ∀ n : ℕ, a n > 0)
  (h₅ : ∀ m n : ℕ, m ≥ n → a m * a n > a (m + 1) * a (n - 1)) :
  |p - 1| ≥ 2 :=
sorry

end NUMINAMATH_GPT_abs_p_minus_1_ge_2_l848_84867


namespace NUMINAMATH_GPT_arcsin_half_eq_pi_six_l848_84806

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end NUMINAMATH_GPT_arcsin_half_eq_pi_six_l848_84806


namespace NUMINAMATH_GPT_solution_set_of_inequality_l848_84843

theorem solution_set_of_inequality (x : ℝ) : (x^2 - |x| > 0) ↔ (x < -1) ∨ (x > 1) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l848_84843


namespace NUMINAMATH_GPT_average_male_students_score_l848_84848

def average_male_score (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ) : ℕ :=
  let total_sum := (male_count + female_count) * total_avg
  let female_sum := female_count * female_avg
  let male_sum := total_sum - female_sum
  male_sum / male_count

theorem average_male_students_score
  (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ)
  (h1 : total_avg = 90) (h2 : female_avg = 92) (h3 : male_count = 8) (h4 : female_count = 20) :
  average_male_score total_avg female_avg male_count female_count = 85 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_male_students_score_l848_84848


namespace NUMINAMATH_GPT_energy_of_first_particle_l848_84875

theorem energy_of_first_particle
  (E_1 E_2 E_3 : ℤ)
  (h1 : E_1^2 - E_2^2 - E_3^2 + E_1 * E_2 = 5040)
  (h2 : E_1^2 + 2 * E_2^2 + 2 * E_3^2 - 2 * E_1 * E_2 - E_1 * E_3 - E_2 * E_3 = -4968)
  (h3 : 0 < E_3)
  (h4 : E_3 ≤ E_2)
  (h5 : E_2 ≤ E_1) : E_1 = 12 :=
by sorry

end NUMINAMATH_GPT_energy_of_first_particle_l848_84875


namespace NUMINAMATH_GPT_carmina_coins_l848_84858

-- Define the conditions related to the problem
variables (n d : ℕ) -- number of nickels and dimes

theorem carmina_coins (h1 : 5 * n + 10 * d = 360) (h2 : 10 * n + 5 * d = 540) : n + d = 60 :=
sorry

end NUMINAMATH_GPT_carmina_coins_l848_84858


namespace NUMINAMATH_GPT_student_correct_answers_l848_84889

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 79) : C = 93 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l848_84889


namespace NUMINAMATH_GPT_plane_speeds_l848_84880

theorem plane_speeds (v : ℕ) 
    (h1 : ∀ (t : ℕ), t = 5 → 20 * v = 4800): 
  v = 240 ∧ 3 * v = 720 := by
  sorry

end NUMINAMATH_GPT_plane_speeds_l848_84880


namespace NUMINAMATH_GPT_four_pq_plus_four_qp_l848_84879

theorem four_pq_plus_four_qp (p q : ℝ) (h : p / q - q / p = 21 / 10) : 
  4 * p / q + 4 * q / p = 16.8 :=
sorry

end NUMINAMATH_GPT_four_pq_plus_four_qp_l848_84879


namespace NUMINAMATH_GPT_fourth_person_height_l848_84809

variables (H1 H2 H3 H4 : ℝ)

theorem fourth_person_height :
  H2 = H1 + 2 →
  H3 = H2 + 3 →
  H4 = H3 + 6 →
  H1 + H2 + H3 + H4 = 288 →
  H4 = 78.5 :=
by
  intros h2_def h3_def h4_def total_height
  -- Proof steps would follow here
  sorry

end NUMINAMATH_GPT_fourth_person_height_l848_84809


namespace NUMINAMATH_GPT_Greenwood_High_School_chemistry_students_l848_84804

theorem Greenwood_High_School_chemistry_students 
    (U : Finset ℕ) (B C P : Finset ℕ) 
    (hU_card : U.card = 20) 
    (hB_subset_U : B ⊆ U) 
    (hC_subset_U : C ⊆ U)
    (hP_subset_U : P ⊆ U)
    (hB_card : B.card = 10) 
    (hB_C_card : (B ∩ C).card = 4) 
    (hB_C_P_card : (B ∩ C ∩ P).card = 3) 
    (hAll_atleast_one : ∀ x ∈ U, x ∈ B ∨ x ∈ C ∨ x ∈ P) :
    C.card = 6 := 
by 
  sorry

end NUMINAMATH_GPT_Greenwood_High_School_chemistry_students_l848_84804


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l848_84864

theorem equilateral_triangle_perimeter (s : ℕ) (b : ℕ) (h1 : 40 = 2 * s + b) (h2 : b = 10) : 3 * s = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l848_84864


namespace NUMINAMATH_GPT_lucas_fraction_of_money_left_l848_84899

theorem lucas_fraction_of_money_left (m p n : ℝ) (h1 : (1 / 4) * m = (1 / 2) * n * p) :
  (m - n * p) / m = 1 / 2 :=
by 
  -- Sorry is used to denote that we are skipping the proof
  sorry

end NUMINAMATH_GPT_lucas_fraction_of_money_left_l848_84899


namespace NUMINAMATH_GPT_total_loads_l848_84852

def shirts_per_load := 3
def sweaters_per_load := 2
def socks_per_load := 4

def white_shirts := 9
def colored_shirts := 12
def white_sweaters := 18
def colored_sweaters := 20
def white_socks := 16
def colored_socks := 24

def white_shirt_loads : ℕ := white_shirts / shirts_per_load
def white_sweater_loads : ℕ := white_sweaters / sweaters_per_load
def white_sock_loads : ℕ := white_socks / socks_per_load

def colored_shirt_loads : ℕ := colored_shirts / shirts_per_load
def colored_sweater_loads : ℕ := colored_sweaters / sweaters_per_load
def colored_sock_loads : ℕ := colored_socks / socks_per_load

def max_white_loads := max (max white_shirt_loads white_sweater_loads) white_sock_loads
def max_colored_loads := max (max colored_shirt_loads colored_sweater_loads) colored_sock_loads

theorem total_loads : max_white_loads + max_colored_loads = 19 := by
  sorry

end NUMINAMATH_GPT_total_loads_l848_84852


namespace NUMINAMATH_GPT_intersection_M_N_l848_84820

open Set

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {x | x^2 < 1}

theorem intersection_M_N : M ∩ N = Ico 0 1 := 
sorry

end NUMINAMATH_GPT_intersection_M_N_l848_84820


namespace NUMINAMATH_GPT_coats_from_high_schools_l848_84845

-- Define the total number of coats collected.
def total_coats_collected : ℕ := 9437

-- Define the number of coats collected from elementary schools.
def coats_from_elementary : ℕ := 2515

-- Goal: Prove that the number of coats collected from high schools is 6922.
theorem coats_from_high_schools : (total_coats_collected - coats_from_elementary) = 6922 := by
  sorry

end NUMINAMATH_GPT_coats_from_high_schools_l848_84845


namespace NUMINAMATH_GPT_ellipse_C_properties_l848_84881

open Real

noncomputable def ellipse_eq (b : ℝ) : Prop :=
  (∀ (x y : ℝ), (x = 1 ∧ y = sqrt 3 / 2) → (x^2 / 4 + y^2 / b^2 = 1))

theorem ellipse_C_properties : 
  (∀ (C : ℝ → ℝ → Prop), 
    (C 0 0) ∧ 
    (∀ x y, C x y → (x = 0 ↔ y = 0)) ∧ 
    (∀ x, C x 0) ∧ 
    (∃ x y, C x y ∧ x = 1 ∧ y = sqrt 3 / 2) →
    (∃ b, b > 0 ∧ b^2 = 1 ∧ ellipse_eq b)) ∧
  (∀ P A B : ℝ × ℝ, 
    (P.1 = P.1 ∧ P.1 ≠ 0 ∧ P.2 = 0 ∧ -2 ≤ P.1 ∧ P.1 ≤ 2) →
    (A.2 = 1/2 * (A.1 - P.1) ∧ B.2 = 1/2 * (B.1 - P.1)) →
    ((P.1 - A.1)^2 + A.2^2 + (P.1 - B.1)^2 + B.2^2 = 5)) :=
by sorry

end NUMINAMATH_GPT_ellipse_C_properties_l848_84881


namespace NUMINAMATH_GPT_jason_picked_pears_l848_84803

def jason_picked (total_picked keith_picked mike_picked jason_picked : ℕ) : Prop :=
  jason_picked + keith_picked + mike_picked = total_picked

theorem jason_picked_pears:
  jason_picked 105 47 12 46 :=
by 
  unfold jason_picked
  sorry

end NUMINAMATH_GPT_jason_picked_pears_l848_84803


namespace NUMINAMATH_GPT_calculate_expression_l848_84883

theorem calculate_expression : 
  (π - 3.14) ^ 0 - 8 ^ (2 / 3) + (1 / 5) ^ 2 * (Real.logb 2 32) + 5 ^ (Real.logb 5 3) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l848_84883


namespace NUMINAMATH_GPT_chef_pillsbury_flour_l848_84822

theorem chef_pillsbury_flour (x : ℕ) (h : 7 / 2 = 28 / x) : x = 8 := sorry

end NUMINAMATH_GPT_chef_pillsbury_flour_l848_84822


namespace NUMINAMATH_GPT_kelly_games_left_l848_84849

-- Definitions based on conditions
def original_games := 80
def additional_games := 31
def games_to_give_away := 105

-- Total games after finding more games
def total_games := original_games + additional_games

-- Number of games left after giving away
def games_left := total_games - games_to_give_away

-- Theorem statement
theorem kelly_games_left : games_left = 6 :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_kelly_games_left_l848_84849


namespace NUMINAMATH_GPT_solve_inequality_l848_84816

theorem solve_inequality (a : ℝ) (ha_pos : 0 < a) :
  (if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
   else if a = 1 then ∅
   else {x : ℝ | 1 / a < x ∧ x < 1}) =
  {x : ℝ | ax^2 - (a + 1) * x + 1 < 0} :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l848_84816


namespace NUMINAMATH_GPT_smallest_rat_num_l848_84870

theorem smallest_rat_num (a b c d : ℚ) (ha : a = -6 / 7) (hb : b = 2) (hc : c = 0) (hd : d = -1) :
  min (min a (min b c)) d = -1 :=
sorry

end NUMINAMATH_GPT_smallest_rat_num_l848_84870


namespace NUMINAMATH_GPT_table_height_l848_84815

-- Definitions
def height_of_table (h l x: ℕ): ℕ := h 
def length_of_block (l: ℕ): ℕ := l 
def width_of_block (w x: ℕ): ℕ := x + 6
def overlap_in_first_arrangement (x : ℕ) : ℕ := x 

-- Conditions
axiom h_conditions (h l x: ℕ): 
  (l + h - x = 42) ∧ (x + 6 + h - l = 36)

-- Proof statement
theorem table_height (h l x : ℕ) (h_conditions : (l + h - x = 42) ∧ (x + 6 + h - l = 36)) :
  height_of_table h l x = 36 := sorry

end NUMINAMATH_GPT_table_height_l848_84815


namespace NUMINAMATH_GPT_smallest_positive_period_of_y_l848_84866

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.sin (-x / 2 + Real.pi / 4)

-- Statement we need to prove
theorem smallest_positive_period_of_y :
  ∃ T > 0, ∀ x : ℝ, y (x + T) = y x ∧ T = 4 * Real.pi := sorry

end NUMINAMATH_GPT_smallest_positive_period_of_y_l848_84866


namespace NUMINAMATH_GPT_find_A_max_min_l848_84871

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end NUMINAMATH_GPT_find_A_max_min_l848_84871


namespace NUMINAMATH_GPT_range_of_m_l848_84827

theorem range_of_m (m : ℝ) : (∀ x, 0 ≤ x ∧ x ≤ m → -6 ≤ x^2 - 4 * x - 2 ∧ x^2 - 4 * x - 2 ≤ -2) → 2 ≤ m ∧ m ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l848_84827


namespace NUMINAMATH_GPT_measure_AB_l848_84859

noncomputable def segment_measure (a b : ℝ) : ℝ :=
  a + (2 / 3) * b

theorem measure_AB (a b : ℝ) (parallel_AB_CD : true) (angle_B_three_times_angle_D : true) (measure_AD_eq_a : true) (measure_CD_eq_b : true) :
  segment_measure a b = a + (2 / 3) * b :=
by
  sorry

end NUMINAMATH_GPT_measure_AB_l848_84859


namespace NUMINAMATH_GPT_min_value_m_n_l848_84885

theorem min_value_m_n (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : a * b = 4)
    (m n : ℝ) (h_m : m = b + 1 / a) (h_n : n = a + 1 / b) : m + n ≥ 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_m_n_l848_84885


namespace NUMINAMATH_GPT_log_addition_l848_84802

theorem log_addition (log_base_10 : ℝ → ℝ) (a b : ℝ) (h_base_10_log : log_base_10 10 = 1) :
  log_base_10 2 + log_base_10 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_log_addition_l848_84802


namespace NUMINAMATH_GPT_euler_school_voting_problem_l848_84853

theorem euler_school_voting_problem :
  let U := 198
  let A := 149
  let B := 119
  let AcBc := 29
  U - AcBc = 169 → 
  A + B - (U - AcBc) = 99 :=
by
  intros h₁
  sorry

end NUMINAMATH_GPT_euler_school_voting_problem_l848_84853


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_10_l848_84854

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_10 (a_1 a_3 a_7 a_9 : ℤ)
    (h1 : ∃ a_1, a_3 = a_1 - 4)
    (h2 : a_7 = a_1 - 12)
    (h3 : a_9 = a_1 - 16)
    (h4 : a_7 * a_7 = a_3 * a_9)
    : sum_of_first_n_terms a_1 (-2) 10 = 110 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_10_l848_84854


namespace NUMINAMATH_GPT_complex_multiplication_l848_84895

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (hi : i^2 = -1) : 
  ((3 : ℂ) - 4 * i) * ((-7 : ℂ) + 6 * i) = (3 : ℂ) + 46 * i := 
  by
    sorry

end NUMINAMATH_GPT_complex_multiplication_l848_84895


namespace NUMINAMATH_GPT_even_composite_fraction_l848_84833

theorem even_composite_fraction : 
  ((4 * 6 * 8 * 10 * 12) : ℚ) / (14 * 16 * 18 * 20 * 22) = 1 / 42 :=
by 
  sorry

end NUMINAMATH_GPT_even_composite_fraction_l848_84833


namespace NUMINAMATH_GPT_inequality_conditions_l848_84828

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + B * C + C * A)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_conditions_l848_84828


namespace NUMINAMATH_GPT_find_B_value_l848_84862

theorem find_B_value (A C B : ℕ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_find_B_value_l848_84862


namespace NUMINAMATH_GPT_max_sum_a_b_l848_84861

theorem max_sum_a_b (a b : ℝ) (ha : 4 * a + 3 * b ≤ 10) (hb : 3 * a + 6 * b ≤ 12) : a + b ≤ 22 / 7 :=
sorry

end NUMINAMATH_GPT_max_sum_a_b_l848_84861


namespace NUMINAMATH_GPT_henry_total_fee_8_bikes_l848_84836

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end NUMINAMATH_GPT_henry_total_fee_8_bikes_l848_84836


namespace NUMINAMATH_GPT_solve_for_x_l848_84801

theorem solve_for_x : ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 3.391 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l848_84801


namespace NUMINAMATH_GPT_hard_candy_food_colouring_l848_84878

noncomputable def food_colouring_per_hard_candy (lollipop_use : ℕ) (gummy_use : ℕ)
    (lollipops_per_day : ℕ) (gummies_per_day : ℕ) (hard_candies_per_day : ℕ)
    (total_food_colouring : ℕ) : ℕ := 
by
  -- Let ml_lollipops be the total amount needed for lollipops
  let ml_lollipops := lollipop_use * lollipops_per_day
  -- Let ml_gummy be the total amount needed for gummy candies
  let ml_gummy := gummy_use * gummies_per_day
  -- Let ml_non_hard be the amount for lollipops and gummy candies combined
  let ml_non_hard := ml_lollipops + ml_gummy
  -- Let ml_hard be the amount used for hard candies alone
  let ml_hard := total_food_colouring - ml_non_hard
  -- Compute the food colouring used per hard candy
  exact ml_hard / hard_candies_per_day

theorem hard_candy_food_colouring :
  food_colouring_per_hard_candy 8 3 150 50 20 1950 = 30 :=
by
  unfold food_colouring_per_hard_candy
  sorry

end NUMINAMATH_GPT_hard_candy_food_colouring_l848_84878


namespace NUMINAMATH_GPT_range_of_m_l848_84856

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + Real.log x

theorem range_of_m (e l : ℝ) (alpha : ℝ) :
  (∀ (α : ℝ), α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 1 ^ 2) → 
  ∀ (x : ℝ), x ∈ Set.Icc l e → f alpha x < m) →
  m ∈ Set.Ioi (1 + 2 * Real.exp 1 ^ 2) := sorry

end NUMINAMATH_GPT_range_of_m_l848_84856


namespace NUMINAMATH_GPT_sum_of_cubes_l848_84813

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l848_84813


namespace NUMINAMATH_GPT_emily_furniture_assembly_time_l848_84860

def num_chairs : Nat := 4
def num_tables : Nat := 2
def num_shelves : Nat := 3
def num_wardrobe : Nat := 1

def time_per_chair : Nat := 8
def time_per_table : Nat := 15
def time_per_shelf : Nat := 10
def time_per_wardrobe : Nat := 45

def total_time : Nat := 
  num_chairs * time_per_chair + 
  num_tables * time_per_table + 
  num_shelves * time_per_shelf + 
  num_wardrobe * time_per_wardrobe

theorem emily_furniture_assembly_time : total_time = 137 := by
  unfold total_time
  sorry

end NUMINAMATH_GPT_emily_furniture_assembly_time_l848_84860


namespace NUMINAMATH_GPT_inv_seq_not_arith_seq_l848_84872

theorem inv_seq_not_arith_seq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_arith : ∃ d : ℝ, d ≠ 0 ∧ b = a + d ∧ c = a + 2 * d) :
  ¬ ∃ d' : ℝ, ∀ i j k : ℝ, i = 1 / a → j = 1 / b → k = 1 / c → j - i = d' ∧ k - j = d' :=
sorry

end NUMINAMATH_GPT_inv_seq_not_arith_seq_l848_84872


namespace NUMINAMATH_GPT_distance_of_third_point_on_trip_l848_84876

theorem distance_of_third_point_on_trip (D : ℝ) (h1 : D + 2 * D + (1/2) * D + 7 * D = 560) :
  (1/2) * D = 27 :=
by
  sorry

end NUMINAMATH_GPT_distance_of_third_point_on_trip_l848_84876


namespace NUMINAMATH_GPT_sticky_strips_used_l848_84896

theorem sticky_strips_used 
  (total_decorations : ℕ) 
  (nails_used : ℕ) 
  (decorations_hung_with_nails_fraction : ℚ) 
  (decorations_hung_with_thumbtacks_fraction : ℚ) 
  (nails_used_eq : nails_used = 50)
  (decorations_hung_with_nails_fraction_eq : decorations_hung_with_nails_fraction = 2/3)
  (decorations_hung_with_thumbtacks_fraction_eq : decorations_hung_with_thumbtacks_fraction = 2/5)
  (total_decorations_eq : total_decorations = nails_used / decorations_hung_with_nails_fraction)
  : (total_decorations - nails_used - decorations_hung_with_thumbtacks_fraction * (total_decorations - nails_used)) = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_sticky_strips_used_l848_84896


namespace NUMINAMATH_GPT_fermats_little_theorem_l848_84825

theorem fermats_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : (a^p - a) % p = 0 := 
by sorry

end NUMINAMATH_GPT_fermats_little_theorem_l848_84825


namespace NUMINAMATH_GPT_inverse_B_squared_l848_84829

-- Defining the inverse matrix B_inv
def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 0, 1]

-- Theorem to prove that the inverse of B^2 is a specific matrix
theorem inverse_B_squared :
  (B_inv * B_inv) = !![9, -6; 0, 1] :=
  by sorry


end NUMINAMATH_GPT_inverse_B_squared_l848_84829


namespace NUMINAMATH_GPT_list_price_is_40_l848_84818

open Real

def list_price (x : ℝ) : Prop :=
  0.15 * (x - 15) = 0.25 * (x - 25)

theorem list_price_is_40 : list_price 40 :=
by
  unfold list_price
  sorry

end NUMINAMATH_GPT_list_price_is_40_l848_84818


namespace NUMINAMATH_GPT_unit_digit_hundred_digit_difference_l848_84811

theorem unit_digit_hundred_digit_difference :
  ∃ (A B C : ℕ), 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000 ∧
    99 * (A - C) = 198 ∧ 0 ≤ A ∧ A < 10 ∧ 0 ≤ C ∧ C < 10 ∧ 0 ≤ B ∧ B < 10 → 
  A - C = 2 :=
by 
  -- we only need to state the theorem, actual proof is not required.
  sorry

end NUMINAMATH_GPT_unit_digit_hundred_digit_difference_l848_84811


namespace NUMINAMATH_GPT_area_of_one_trapezoid_l848_84838

theorem area_of_one_trapezoid (outer_area inner_area : ℝ) (num_trapezoids : ℕ) (h_outer : outer_area = 36) (h_inner : inner_area = 4) (h_num_trapezoids : num_trapezoids = 3) : (outer_area - inner_area) / num_trapezoids = 32 / 3 :=
by
  rw [h_outer, h_inner, h_num_trapezoids]
  norm_num

end NUMINAMATH_GPT_area_of_one_trapezoid_l848_84838


namespace NUMINAMATH_GPT_weight_of_five_bowling_balls_l848_84839

theorem weight_of_five_bowling_balls (b c : ℕ) (hb : 9 * b = 4 * c) (hc : c = 36) : 5 * b = 80 := by
  sorry

end NUMINAMATH_GPT_weight_of_five_bowling_balls_l848_84839


namespace NUMINAMATH_GPT_resistance_parallel_l848_84857

theorem resistance_parallel (x y r : ℝ) (hy : y = 6) (hr : r = 2.4) 
  (h : 1 / r = 1 / x + 1 / y) : x = 4 :=
  sorry

end NUMINAMATH_GPT_resistance_parallel_l848_84857


namespace NUMINAMATH_GPT_train_length_l848_84844

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (length_meters : ℕ) : speed_kmph = 90 → time_sec = 4 → length_meters = 100 :=
by
  intros h₁ h₂
  have speed_mps : ℕ := speed_kmph * 1000 / 3600
  have speed_mps_val : speed_mps = 25 := sorry
  have distance : ℕ := speed_mps * time_sec
  have distance_val : distance = 100 := sorry
  exact sorry

end NUMINAMATH_GPT_train_length_l848_84844


namespace NUMINAMATH_GPT_foreign_exchange_decline_l848_84812

theorem foreign_exchange_decline (x : ℝ) (h1 : 200 * (1 - x)^2 = 98) : 
  200 * (1 - x)^2 = 98 :=
by
  sorry

end NUMINAMATH_GPT_foreign_exchange_decline_l848_84812


namespace NUMINAMATH_GPT_a4_minus_b4_l848_84898

theorem a4_minus_b4 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : a^4 - b^4 = -1 := by
  sorry

end NUMINAMATH_GPT_a4_minus_b4_l848_84898


namespace NUMINAMATH_GPT_sin_solution_set_l848_84877

open Real

theorem sin_solution_set (x : ℝ) : 
  (3 * sin x = 1 + cos (2 * x)) ↔ ∃ k : ℤ, x = k * π + (-1) ^ k * (π / 6) :=
by
  sorry

end NUMINAMATH_GPT_sin_solution_set_l848_84877


namespace NUMINAMATH_GPT_one_thirds_in_fraction_l848_84832

theorem one_thirds_in_fraction : (9 / 5) / (1 / 3) = 27 / 5 := by
  sorry

end NUMINAMATH_GPT_one_thirds_in_fraction_l848_84832


namespace NUMINAMATH_GPT_tyler_meal_choices_l848_84873

-- Define the total number of different meals Tyler can choose given the conditions.
theorem tyler_meal_choices : 
    (3 * (Nat.choose 5 3) * 4 * 4 = 480) := 
by
    -- Using the built-in combination function and the fact that meat, dessert, and drink choices are directly multiplied.
    sorry

end NUMINAMATH_GPT_tyler_meal_choices_l848_84873


namespace NUMINAMATH_GPT_initial_amount_l848_84847

theorem initial_amount (X : ℚ) (F : ℚ) :
  (∀ (X F : ℚ), F = X * (3/4)^3 → F = 37 → X = 37 * 64 / 27) :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_l848_84847
