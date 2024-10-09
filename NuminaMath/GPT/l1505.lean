import Mathlib

namespace sum_of_three_numbers_l1505_150533

theorem sum_of_three_numbers (a b c : ℝ) :
  a + b = 35 → b + c = 47 → c + a = 58 → a + b + c = 70 :=
by
  intros h1 h2 h3
  sorry

end sum_of_three_numbers_l1505_150533


namespace june_initial_stickers_l1505_150506

theorem june_initial_stickers (J b g t : ℕ) (h_b : b = 63) (h_g : g = 25) (h_t : t = 189) : 
  (J + g) + (b + g) = t → J = 76 :=
by
  sorry

end june_initial_stickers_l1505_150506


namespace addilynn_eggs_initial_l1505_150505

theorem addilynn_eggs_initial (E : ℕ) (H1 : ∃ (E : ℕ), (E / 2) - 15 = 21) : E = 72 :=
by
  sorry

end addilynn_eggs_initial_l1505_150505


namespace number_of_buckets_l1505_150587

-- Defining the conditions
def total_mackerels : ℕ := 27
def mackerels_per_bucket : ℕ := 3

-- The theorem to prove
theorem number_of_buckets :
  total_mackerels / mackerels_per_bucket = 9 :=
sorry

end number_of_buckets_l1505_150587


namespace james_speed_is_16_l1505_150582

theorem james_speed_is_16
  (distance : ℝ)
  (time : ℝ)
  (distance_eq : distance = 80)
  (time_eq : time = 5) :
  (distance / time = 16) :=
by
  rw [distance_eq, time_eq]
  norm_num

end james_speed_is_16_l1505_150582


namespace sum_of_exponents_l1505_150518

theorem sum_of_exponents (n : ℕ) (h : n = 2^11 + 2^10 + 2^5 + 2^4 + 2^2) : 11 + 10 + 5 + 4 + 2 = 32 :=
by {
  -- The proof could be written here
  sorry
}

end sum_of_exponents_l1505_150518


namespace mass_percentage_of_carbon_in_ccl4_l1505_150575

-- Define the atomic masses
def atomic_mass_c : Float := 12.01
def atomic_mass_cl : Float := 35.45

-- Define the molecular composition of Carbon Tetrachloride (CCl4)
def mol_mass_ccl4 : Float := (1 * atomic_mass_c) + (4 * atomic_mass_cl)

-- Theorem to prove the mass percentage of carbon in Carbon Tetrachloride is 7.81%
theorem mass_percentage_of_carbon_in_ccl4 : 
  (atomic_mass_c / mol_mass_ccl4) * 100 = 7.81 := by
  sorry

end mass_percentage_of_carbon_in_ccl4_l1505_150575


namespace solve_equation1_solve_equation2_l1505_150565

theorem solve_equation1 :
  ∀ x : ℝ, ((x-1) * (x-1) = 3 * (x-1)) ↔ (x = 1 ∨ x = 4) :=
by
  intro x
  sorry

theorem solve_equation2 :
  ∀ x : ℝ, (x^2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by
  intro x
  sorry

end solve_equation1_solve_equation2_l1505_150565


namespace series_sum_l1505_150588

variable (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : b < a)

noncomputable def infinite_series : ℝ := 
∑' n, 1 / ( ((n - 1) * a^2 - (n - 2) * b^2) * (n * a^2 - (n - 1) * b^2) )

theorem series_sum : infinite_series a b = 1 / ((a^2 - b^2) * b^2) := 
by 
  sorry

end series_sum_l1505_150588


namespace problem_I_problem_II_l1505_150591

-- Problem I statement
theorem problem_I (a b c : ℝ) (h : a + b + c = 1) : (a + 1)^2 + (b + 1)^2 + (c + 1)^2 ≥ 16 / 3 := 
by
  sorry

-- Problem II statement
theorem problem_II (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (2 * x - 1) ≥ 2) →
  a ∈ Set.Iic (-3/2) ∪ Set.Ici (5/2) :=
by 
  sorry

end problem_I_problem_II_l1505_150591


namespace cisco_spots_difference_l1505_150590

theorem cisco_spots_difference :
  ∃ C G R : ℕ, R = 46 ∧ G = 5 * C ∧ G + C = 108 ∧ (23 - C) = 5 :=
by
  sorry

end cisco_spots_difference_l1505_150590


namespace exponential_inequality_l1505_150520

-- Define the problem conditions and the proof goal
theorem exponential_inequality (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := 
sorry

end exponential_inequality_l1505_150520


namespace find_k_eq_l1505_150510

theorem find_k_eq (n : ℝ) (k m : ℤ) (h : ∀ n : ℝ, n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2) : k = 3 := 
sorry

end find_k_eq_l1505_150510


namespace estimate_y_value_at_x_equals_3_l1505_150540

noncomputable def estimate_y (x : ℝ) (a : ℝ) : ℝ :=
  (1 / 3) * x + a

theorem estimate_y_value_at_x_equals_3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ) (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ),
    (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 2 * (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8)) →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 8 →
    estimate_y 3 (1 / 6) = 7 / 6 := by
  intro x1 x2 x3 x4 x5 x6 x7 x8 y1 y2 y3 y4 y5 y6 y7 y8 h_sum hx
  sorry

end estimate_y_value_at_x_equals_3_l1505_150540


namespace original_number_increase_l1505_150592

theorem original_number_increase (x : ℝ) (h : 1.20 * x = 1800) : x = 1500 :=
by
  sorry

end original_number_increase_l1505_150592


namespace compute_expression_l1505_150580

theorem compute_expression : 85 * 1500 + (1 / 2) * 1500 = 128250 :=
by
  sorry

end compute_expression_l1505_150580


namespace speed_difference_l1505_150581

theorem speed_difference :
  let distance : ℝ := 8
  let zoe_time_hours : ℝ := 2 / 3
  let john_time_hours : ℝ := 1
  let zoe_speed : ℝ := distance / zoe_time_hours
  let john_speed : ℝ := distance / john_time_hours
  zoe_speed - john_speed = 4 :=
by
  sorry

end speed_difference_l1505_150581


namespace the_inequality_l1505_150544

theorem the_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (a / (1 + b)) + (b / (1 + c)) + (c / (1 + a)) ≥ 3 / 2 :=
by sorry

end the_inequality_l1505_150544


namespace remainder_of_12_factorial_mod_13_l1505_150501

open Nat

theorem remainder_of_12_factorial_mod_13 : (factorial 12) % 13 = 12 := by
  -- Wilson's Theorem: For a prime number \( p \), \( (p-1)! \equiv -1 \pmod{p} \)
  -- Given \( p = 13 \), we have \( 12! \equiv -1 \pmod{13} \)
  -- Thus, it follows that the remainder is 12
  sorry

end remainder_of_12_factorial_mod_13_l1505_150501


namespace total_lives_l1505_150549

theorem total_lives :
  ∀ (num_friends num_new_players lives_per_friend lives_per_new_player : ℕ),
  num_friends = 2 →
  lives_per_friend = 6 →
  num_new_players = 2 →
  lives_per_new_player = 6 →
  (num_friends * lives_per_friend + num_new_players * lives_per_new_player) = 24 :=
by
  intros num_friends num_new_players lives_per_friend lives_per_new_player
  intro h1 h2 h3 h4
  sorry

end total_lives_l1505_150549


namespace find_c_for_same_solution_l1505_150525

theorem find_c_for_same_solution (c : ℝ) (x : ℝ) :
  (3 * x + 5 = 1) ∧ (c * x + 15 = -5) → c = 15 :=
by
  sorry

end find_c_for_same_solution_l1505_150525


namespace exponent_property_l1505_150526

theorem exponent_property : (-2)^2004 + 3 * (-2)^2003 = -2^2003 :=
by 
  sorry

end exponent_property_l1505_150526


namespace find_y_given_x_inverse_square_l1505_150513

theorem find_y_given_x_inverse_square (x y : ℚ) : 
  (∀ k, (3 * y = k / x^2) ∧ (3 * 5 = k / 2^2)) → (x = 6) → y = 5 / 9 :=
by
  sorry

end find_y_given_x_inverse_square_l1505_150513


namespace number_of_beavers_l1505_150504

-- Definitions of the problem conditions
def total_workers : Nat := 862
def number_of_spiders : Nat := 544

-- The statement we need to prove
theorem number_of_beavers : (total_workers - number_of_spiders) = 318 := 
by 
  sorry

end number_of_beavers_l1505_150504


namespace nine_digit_number_l1505_150514

-- Conditions as definitions
def highest_digit (n : ℕ) : Prop :=
  (n / 100000000) = 6

def million_place (n : ℕ) : Prop :=
  (n / 1000000) % 10 = 1

def hundred_place (n : ℕ) : Prop :=
  n % 1000 / 100 = 1

def rest_digits_zero (n : ℕ) : Prop :=
  (n % 1000000 / 1000) % 10 = 0 ∧ 
  (n % 1000000 / 10000) % 10 = 0 ∧ 
  (n % 1000000 / 100000) % 10 = 0 ∧ 
  (n % 100000000 / 10000000) % 10 = 0 ∧ 
  (n % 100000000 / 100000000) % 10 = 0 ∧ 
  (n % 1000000000 / 100000000) % 10 = 6

-- The nine-digit number
def given_number : ℕ := 6001000100

-- Prove number == 60,010,001,00 and approximate to 6 billion
theorem nine_digit_number :
  ∃ n : ℕ, highest_digit n ∧ million_place n ∧ hundred_place n ∧ rest_digits_zero n ∧ n = 6001000100 ∧ (n / 1000000000) = 6 :=
sorry

end nine_digit_number_l1505_150514


namespace perpendicular_lines_l1505_150535

theorem perpendicular_lines (m : ℝ) :
  (∃ k l : ℝ, k * m + (1 - m) * l = 3 ∧ (m - 1) * k + (2 * m + 3) * l = 2) → m = -3 ∨ m = 1 :=
by sorry

end perpendicular_lines_l1505_150535


namespace total_amount_paid_correct_l1505_150566

-- Definitions of quantities and rates
def quantity_grapes := 3
def rate_grapes := 70
def quantity_mangoes := 9
def rate_mangoes := 55

-- Total amount calculation
def total_amount_paid := quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes

-- Theorem to prove total amount paid is 705
theorem total_amount_paid_correct : total_amount_paid = 705 :=
by
  sorry

end total_amount_paid_correct_l1505_150566


namespace largest_of_five_consecutive_non_primes_under_40_l1505_150553

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n 

theorem largest_of_five_consecutive_non_primes_under_40 :
  ∃ x, (x > 9) ∧ (x + 4 < 40) ∧ 
       (¬ is_prime x) ∧
       (¬ is_prime (x + 1)) ∧
       (¬ is_prime (x + 2)) ∧
       (¬ is_prime (x + 3)) ∧
       (¬ is_prime (x + 4)) ∧
       (x + 4 = 36) :=
sorry

end largest_of_five_consecutive_non_primes_under_40_l1505_150553


namespace vectors_parallel_solution_l1505_150570

theorem vectors_parallel_solution (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (2, x)) (h2 : b = (x, 8)) (h3 : ∃ k, b = (k * 2, k * x)) : x = 4 ∨ x = -4 :=
by
  sorry

end vectors_parallel_solution_l1505_150570


namespace prove_m_set_l1505_150539

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}

-- Define set B as dependent on m
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}

-- The main proof statement
theorem prove_m_set : {m : ℝ | B m ∩ A = B m} = {0, 1, 2} :=
by
  -- Code here would prove the above theorem
  sorry

end prove_m_set_l1505_150539


namespace value_of_x_minus_y_l1505_150537

theorem value_of_x_minus_y (x y : ℝ) (h1 : x = -(-3)) (h2 : |y| = 5) (h3 : x * y < 0) : x - y = 8 := 
sorry

end value_of_x_minus_y_l1505_150537


namespace sub_neg_eq_add_pos_l1505_150571

theorem sub_neg_eq_add_pos : 0 - (-2) = 2 := 
by
  sorry

end sub_neg_eq_add_pos_l1505_150571


namespace problem_part_1_problem_part_2_l1505_150545

theorem problem_part_1 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  |c| ≤ 1 :=
by
  sorry

theorem problem_part_2 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → |g x| ≤ 2 :=
by
  sorry

end problem_part_1_problem_part_2_l1505_150545


namespace find_m_value_l1505_150554

theorem find_m_value (m : ℤ) (h1 : m - 2 ≠ 0) (h2 : |m| = 2) : m = -2 :=
by {
  sorry
}

end find_m_value_l1505_150554


namespace men_in_club_l1505_150530

-- Definitions
variables (M W : ℕ) -- Number of men and women

-- Conditions
def club_members := M + W = 30
def event_participation := W / 3 + M = 18

-- Goal
theorem men_in_club : club_members M W → event_participation M W → M = 12 :=
sorry

end men_in_club_l1505_150530


namespace sqrt_double_sqrt_four_l1505_150502

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem sqrt_double_sqrt_four :
  sqrt (sqrt 4) = sqrt 2 ∨ sqrt (sqrt 4) = -sqrt 2 :=
by
  sorry

end sqrt_double_sqrt_four_l1505_150502


namespace angle_supplement_complement_l1505_150500

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l1505_150500


namespace smallest_n_l1505_150594

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def condition_for_n (n : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ n → ∀ x : ℕ, x ∈ M k → ∃ y : ℕ, y ∈ M k ∧ y ≠ x ∧ is_perfect_square (x + y)
  where M (k : ℕ) := { m : ℕ | m > 0 ∧ m ≤ k }

theorem smallest_n : ∃ n : ℕ, (condition_for_n n) ∧ (∀ m < n, ¬ condition_for_n m) :=
  sorry

end smallest_n_l1505_150594


namespace remainder_of_13_pow_13_plus_13_div_14_l1505_150528

theorem remainder_of_13_pow_13_plus_13_div_14 : ((13 ^ 13 + 13) % 14) = 12 :=
by
  sorry

end remainder_of_13_pow_13_plus_13_div_14_l1505_150528


namespace find_a_l1505_150593

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

def monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

def valid_interval (a : ℝ) : Prop :=
  monotonically_decreasing f (Set.Icc (a-1) (a+1))

theorem find_a :
  {a : ℝ | valid_interval a} = {a : ℝ | 1 < a ∧ a ≤ 2} :=
by
  sorry

end find_a_l1505_150593


namespace ratio_sheila_purity_l1505_150563

theorem ratio_sheila_purity (rose_share : ℕ) (total_rent : ℕ) (purity_share : ℕ) (sheila_share : ℕ) 
  (h1 : rose_share = 1800) 
  (h2 : total_rent = 5400) 
  (h3 : rose_share = 3 * purity_share)
  (h4 : total_rent = purity_share + rose_share + sheila_share) : 
  sheila_share / purity_share = 5 :=
by
  -- Proof will be here
  sorry

end ratio_sheila_purity_l1505_150563


namespace total_daisies_l1505_150516

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end total_daisies_l1505_150516


namespace field_division_l1505_150515

theorem field_division
  (total_area : ℕ)
  (part_area : ℕ)
  (diff : ℕ → ℕ)
  (X : ℕ)
  (h_total : total_area = 900)
  (h_part : part_area = 405)
  (h_diff : diff (total_area - part_area - part_area) = (1 / 5 : ℚ) * X)
  : X = 450 := 
sorry

end field_division_l1505_150515


namespace constructible_triangle_and_area_bound_l1505_150536

noncomputable def triangle_inequality_sine (α β γ : ℝ) : Prop :=
  (Real.sin α + Real.sin β > Real.sin γ) ∧
  (Real.sin β + Real.sin γ > Real.sin α) ∧
  (Real.sin γ + Real.sin α > Real.sin β)

theorem constructible_triangle_and_area_bound 
  (α β γ : ℝ) (h_pos : 0 < α) (h_pos_β : 0 < β) (h_pos_γ : 0 < γ)
  (h_sum : α + β + γ < Real.pi)
  (h_ineq1 : α + β > γ)
  (h_ineq2 : β + γ > α)
  (h_ineq3 : γ + α > β) :
  triangle_inequality_sine α β γ ∧
  (Real.sin α * Real.sin β * Real.sin γ) / 4 ≤ (1 / 8) * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) :=
sorry

end constructible_triangle_and_area_bound_l1505_150536


namespace washing_time_is_45_l1505_150509

-- Definitions based on conditions
variables (x : ℕ) -- time to wash one load
axiom h1 : 2 * x + 75 = 165 -- total laundry time equation

-- The statement to prove: washing one load takes 45 minutes
theorem washing_time_is_45 : x = 45 :=
by
  sorry

end washing_time_is_45_l1505_150509


namespace flowers_not_roses_percentage_l1505_150534

def percentage_non_roses (roses tulips daisies : Nat) : Nat :=
  let total := roses + tulips + daisies
  let non_roses := total - roses
  (non_roses * 100) / total

theorem flowers_not_roses_percentage :
  percentage_non_roses 25 40 35 = 75 :=
by
  sorry

end flowers_not_roses_percentage_l1505_150534


namespace area_of_trapezoid_l1505_150531

-- Define the parameters as given in the problem
def PQ : ℝ := 40
def RS : ℝ := 25
def h : ℝ := 10
def PR : ℝ := 20

-- Assert the quadrilateral is a trapezoid with bases PQ and RS parallel
def isTrapezoid (PQ RS : ℝ) (h : ℝ) (PR : ℝ) : Prop := true -- this is just a placeholder to state that it's a trapezoid

-- The main statement for the area of the trapezoid
theorem area_of_trapezoid (h : ℝ) (PQ RS : ℝ) (h : ℝ) (PR : ℝ) (is_trapezoid : isTrapezoid PQ RS h PR) : (1/2) * (PQ + RS) * h = 325 :=
by
  sorry

end area_of_trapezoid_l1505_150531


namespace maximize_volume_l1505_150550

theorem maximize_volume
  (R H A : ℝ) (K : ℝ) (hA : 2 * π * R * H + 2 * π * R * (Real.sqrt (R ^ 2 + H ^ 2)) = A)
  (hK : K = A / (2 * π)) :
  R = (A / (π * Real.sqrt 5)) ^ (1 / 3) :=
sorry

end maximize_volume_l1505_150550


namespace Sasha_added_digit_l1505_150559

noncomputable def Kolya_number : Nat := 45 -- Sum of all digits 0 to 9

theorem Sasha_added_digit (d x : Nat) (h : 0 ≤ d ∧ d ≤ 9) (h1 : 0 ≤ x ∧ x ≤ 9) (condition : Kolya_number - d + x ≡ 0 [MOD 9]) : x = 0 ∨ x = 9 := 
sorry

end Sasha_added_digit_l1505_150559


namespace solution_set_of_inequality_l1505_150532

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - 1) / (x + 2) > 1 ↔ x < -2 ∨ x > 3 :=
by
  sorry

end solution_set_of_inequality_l1505_150532


namespace michael_passes_donovan_l1505_150561

noncomputable def track_length : ℕ := 600
noncomputable def donovan_lap_time : ℕ := 45
noncomputable def michael_lap_time : ℕ := 40

theorem michael_passes_donovan :
  ∃ n : ℕ, michael_lap_time * n > donovan_lap_time * (n - 1) ∧ n = 9 :=
by
  sorry

end michael_passes_donovan_l1505_150561


namespace jesse_max_correct_answers_l1505_150596

theorem jesse_max_correct_answers :
  ∃ a b c : ℕ, a + b + c = 60 ∧ 5 * a - 2 * c = 150 ∧ a ≤ 38 :=
sorry

end jesse_max_correct_answers_l1505_150596


namespace geom_seq_not_necessary_sufficient_l1505_150508

theorem geom_seq_not_necessary_sufficient (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ¬(∀ n, a n > a (n + 1) → false) ∨ ¬(∀ n, a (n + 1) > a n) :=
sorry

end geom_seq_not_necessary_sufficient_l1505_150508


namespace regular_polygon_sides_l1505_150552

theorem regular_polygon_sides (n : ℕ) (h : ∀ (i : ℕ), i < n → (160 : ℝ) = (180 * (n - 2)) / n) : n = 18 := 
by 
  sorry

end regular_polygon_sides_l1505_150552


namespace boards_nailing_l1505_150584

variables {x y a b : ℕ}

theorem boards_nailing (h1 : 2 * x + 3 * y = 87)
                       (h2 : 3 * a + 5 * b = 94) :
                       x + y = 30 ∧ a + b = 30 :=
sorry

end boards_nailing_l1505_150584


namespace no_solution_exists_l1505_150562

theorem no_solution_exists : ¬ ∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by sorry

end no_solution_exists_l1505_150562


namespace rectangle_width_l1505_150538

/-- Given the conditions:
    - length of a rectangle is 5.4 cm
    - area of the rectangle is 48.6 cm²
    Prove that the width of the rectangle is 9 cm.
-/
theorem rectangle_width (length width area : ℝ) 
  (h_length : length = 5.4) 
  (h_area : area = 48.6) 
  (h_area_eq : area = length * width) : 
  width = 9 := 
by
  sorry

end rectangle_width_l1505_150538


namespace solution_to_power_tower_l1505_150527

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solution_to_power_tower : ∃ x : ℝ, infinite_power_tower x = 4 ∧ x = Real.sqrt 2 := sorry

end solution_to_power_tower_l1505_150527


namespace mark_bench_press_correct_l1505_150547

def dave_weight : ℝ := 175
def dave_bench_press : ℝ := 3 * dave_weight

def craig_bench_percentage : ℝ := 0.20
def craig_bench_press : ℝ := craig_bench_percentage * dave_bench_press

def emma_bench_percentage : ℝ := 0.75
def emma_initial_bench_press : ℝ := emma_bench_percentage * dave_bench_press
def emma_actual_bench_press : ℝ := emma_initial_bench_press + 15

def combined_craig_emma : ℝ := craig_bench_press + emma_actual_bench_press

def john_bench_factor : ℝ := 2
def john_bench_press : ℝ := john_bench_factor * combined_craig_emma

def mark_reduction : ℝ := 50
def mark_bench_press : ℝ := combined_craig_emma - mark_reduction

theorem mark_bench_press_correct : mark_bench_press = 463.75 := by
  sorry

end mark_bench_press_correct_l1505_150547


namespace charity_delivered_100_plates_l1505_150597

variables (cost_rice_per_plate cost_chicken_per_plate total_amount_spent : ℝ)
variable (P : ℝ)

-- Conditions provided
def rice_cost : ℝ := 0.10
def chicken_cost : ℝ := 0.40
def total_spent : ℝ := 50
def total_cost_per_plate : ℝ := rice_cost + chicken_cost

-- Lean 4 statement to prove:
theorem charity_delivered_100_plates :
  total_spent = 50 →
  total_cost_per_plate = rice_cost + chicken_cost →
  rice_cost = 0.10 →
  chicken_cost = 0.40 →
  P = total_spent / total_cost_per_plate →
  P = 100 :=
by
  sorry

end charity_delivered_100_plates_l1505_150597


namespace complex_exchange_of_apartments_in_two_days_l1505_150507

theorem complex_exchange_of_apartments_in_two_days :
  ∀ (n : ℕ) (p : Fin n → Fin n), ∃ (day1 day2 : Fin n → Fin n),
    (∀ x : Fin n, p (day1 x) = day2 x ∨ day1 (p x) = day2 x) ∧
    (∀ x : Fin n, day1 x ≠ x) ∧
    (∀ x : Fin n, day2 x ≠ x) :=
by
  sorry

end complex_exchange_of_apartments_in_two_days_l1505_150507


namespace vlecks_in_straight_angle_l1505_150548

theorem vlecks_in_straight_angle (V : Type) [LinearOrderedField V] (full_circle_vlecks : V) (h1 : full_circle_vlecks = 600) :
  (full_circle_vlecks / 2) = 300 :=
by
  sorry

end vlecks_in_straight_angle_l1505_150548


namespace sum_first_2500_terms_eq_zero_l1505_150522

theorem sum_first_2500_terms_eq_zero
  (b : ℕ → ℤ)
  (h1 : ∀ n ≥ 3, b n = b (n - 1) - b (n - 2))
  (h2 : (Finset.range 1800).sum b = 2023)
  (h3 : (Finset.range 2023).sum b = 1800) :
  (Finset.range 2500).sum b = 0 :=
sorry

end sum_first_2500_terms_eq_zero_l1505_150522


namespace trip_savings_l1505_150512

theorem trip_savings :
  let ticket_cost := 10
  let combo_cost := 10
  let ticket_discount := 0.20
  let combo_discount := 0.50
  (ticket_discount * ticket_cost + combo_discount * combo_cost) = 7 := 
by
  sorry

end trip_savings_l1505_150512


namespace drop_in_water_level_l1505_150541

theorem drop_in_water_level (rise_level : ℝ) (drop_level : ℝ) 
  (h : rise_level = 1) : drop_level = -2 :=
by
  sorry

end drop_in_water_level_l1505_150541


namespace find_constant_a_l1505_150572

theorem find_constant_a (a : ℚ) (S : ℕ → ℚ) (hS : ∀ n, S n = (a - 2) * 3^(n + 1) + 2) : a = 4 / 3 :=
by
  sorry

end find_constant_a_l1505_150572


namespace ratio_of_volumes_l1505_150578

theorem ratio_of_volumes (rC hC rD hD : ℝ) (h1 : rC = 10) (h2 : hC = 25) (h3 : rD = 25) (h4 : hD = 10) : 
  (1/3 * Real.pi * rC^2 * hC) / (1/3 * Real.pi * rD^2 * hD) = 2 / 5 :=
by
  sorry

end ratio_of_volumes_l1505_150578


namespace jill_has_1_more_peach_than_jake_l1505_150599

theorem jill_has_1_more_peach_than_jake
    (jill_peaches : ℕ)
    (steven_peaches : ℕ)
    (jake_peaches : ℕ)
    (h1 : jake_peaches = steven_peaches - 16)
    (h2 : steven_peaches = jill_peaches + 15)
    (h3 : jill_peaches = 12) :
    12 - (steven_peaches - 16) = 1 := 
sorry

end jill_has_1_more_peach_than_jake_l1505_150599


namespace negation_of_forall_implies_exists_l1505_150542

theorem negation_of_forall_implies_exists :
  (¬ ∀ x : ℝ, x^2 > 1) = (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_forall_implies_exists_l1505_150542


namespace flour_to_add_l1505_150519

-- Define the conditions
def total_flour_required : ℕ := 9
def flour_already_added : ℕ := 2

-- Define the proof statement
theorem flour_to_add : total_flour_required - flour_already_added = 7 := 
by {
    sorry
}

end flour_to_add_l1505_150519


namespace determine_missing_digits_l1505_150555

theorem determine_missing_digits :
  (237 * 0.31245 = 7430.65) := 
by 
  sorry

end determine_missing_digits_l1505_150555


namespace arithmetic_sequence_sum_product_l1505_150567

noncomputable def a := 13 / 2
def d := 3 / 2

theorem arithmetic_sequence_sum_product (a d : ℚ) (h1 : 4 * a = 26) (h2 : a^2 - d^2 = 40) :
  (a - 3 * d, a - d, a + d, a + 3 * d) = (2, 5, 8, 11) ∨
  (a - 3 * d, a - d, a + d, a + 3 * d) = (11, 8, 5, 2) :=
  sorry

end arithmetic_sequence_sum_product_l1505_150567


namespace incorrect_statement_C_l1505_150595

theorem incorrect_statement_C :
  (∀ b h : ℕ, (2 * b) * h = 2 * (b * h)) ∧
  (∀ b h : ℕ, (1 / 2) * b * (2 * h) = 2 * ((1 / 2) * b * h)) ∧
  (∀ r : ℕ, (π * (2 * r) ^ 2 ≠ 2 * (π * r ^ 2))) ∧
  (∀ a b : ℕ, (a / 2) / (2 * b) ≠ a / b) ∧
  (∀ x : ℤ, x < 0 -> 2 * x < x) →
  false :=
by
  intros h
  sorry

end incorrect_statement_C_l1505_150595


namespace speed_of_water_l1505_150558

theorem speed_of_water (v : ℝ) :
  (∀ (distance time : ℝ), distance = 16 ∧ time = 8 → distance = (4 - v) * time) → 
  v = 2 :=
by
  intro h
  have h1 : 16 = (4 - v) * 8 := h 16 8 (by simp)
  sorry

end speed_of_water_l1505_150558


namespace fraction_of_students_on_trip_are_girls_l1505_150543

variable (b g : ℕ)
variable (H1 : g = 2 * b) -- twice as many girls as boys
variable (fraction_girls_on_trip : ℚ := 2 / 3)
variable (fraction_boys_on_trip : ℚ := 1 / 2)

def fraction_of_girls_on_trip (b g : ℕ) (H1 : g = 2 * b) (fraction_girls_on_trip : ℚ) (fraction_boys_on_trip : ℚ) :=
  let girls_on_trip := fraction_girls_on_trip * g
  let boys_on_trip := fraction_boys_on_trip * b
  let total_on_trip := girls_on_trip + boys_on_trip
  girls_on_trip / total_on_trip

theorem fraction_of_students_on_trip_are_girls (b g : ℕ) (H1 : g = 2 * b) : 
  fraction_of_girls_on_trip b g H1 (2 / 3) (1 / 2) = 8 / 11 := 
by sorry

end fraction_of_students_on_trip_are_girls_l1505_150543


namespace xyz_inequality_l1505_150503

theorem xyz_inequality (x y z : ℝ) (h : x + y + z > 0) : x^3 + y^3 + z^3 > 3 * x * y * z :=
by
  sorry

end xyz_inequality_l1505_150503


namespace point_in_at_least_15_circles_l1505_150589

theorem point_in_at_least_15_circles
  (C : Fin 100 → Set (ℝ × ℝ))
  (h1 : ∀ i j, ∃ p, p ∈ C i ∧ p ∈ C j)
  : ∃ p, ∃ S : Finset (Fin 100), S.card ≥ 15 ∧ ∀ i ∈ S, p ∈ C i :=
sorry

end point_in_at_least_15_circles_l1505_150589


namespace correct_quotient_l1505_150529

theorem correct_quotient (D Q : ℕ) (h1 : 21 * Q = 12 * 56) : Q = 32 :=
by {
  -- Proof to be provided
  sorry
}

end correct_quotient_l1505_150529


namespace david_has_15_shells_l1505_150586

-- Definitions from the conditions
def mia_shells (david_shells : ℕ) : ℕ := 4 * david_shells
def ava_shells (david_shells : ℕ) : ℕ := mia_shells david_shells + 20
def alice_shells (david_shells : ℕ) : ℕ := (ava_shells david_shells) / 2

-- Total number of shells
def total_shells (david_shells : ℕ) : ℕ := david_shells + mia_shells david_shells + ava_shells david_shells + alice_shells david_shells

-- Proving the number of shells David has is 15 given the total number of shells is 195
theorem david_has_15_shells : total_shells 15 = 195 :=
by
  sorry

end david_has_15_shells_l1505_150586


namespace output_sequence_value_l1505_150523

theorem output_sequence_value (x y : Int) (seq : List (Int × Int))
  (h : (x, y) ∈ seq) (h_y : y = -10) : x = 32 :=
by
  sorry

end output_sequence_value_l1505_150523


namespace compute_LM_length_l1505_150576

-- Definitions of lengths and equidistant property
variables (GH JK LM : ℝ) 
variables (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
variables (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK)

-- State the theorem to prove lengths
theorem compute_LM_length (GH JD LM : ℝ) (equidistant : GH * 2 = 120 ∧ JK * 2 = 80)
  (parallel : GH = 120 ∧ JK = 80 ∧ GH = JK) :
  LM = (2 / 3) * 80 := 
sorry

end compute_LM_length_l1505_150576


namespace two_roses_more_than_three_carnations_l1505_150524

variable {x y : ℝ}

theorem two_roses_more_than_three_carnations
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y := 
by 
  sorry

end two_roses_more_than_three_carnations_l1505_150524


namespace probability_of_difference_three_l1505_150517

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end probability_of_difference_three_l1505_150517


namespace cylinder_has_no_triangular_cross_section_l1505_150560

inductive GeometricSolid
  | cylinder
  | cone
  | triangularPrism
  | cube

open GeometricSolid

-- Define the cross section properties
def can_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cone ∨ s = triangularPrism ∨ s = cube

-- Define the property where a solid cannot have a triangular cross-section
def cannot_have_triangular_cross_section (s : GeometricSolid) : Prop :=
  s = cylinder

theorem cylinder_has_no_triangular_cross_section :
  cannot_have_triangular_cross_section cylinder ∧
  ¬ can_have_triangular_cross_section cylinder :=
by
  -- This is where we state the proof goal
  sorry

end cylinder_has_no_triangular_cross_section_l1505_150560


namespace find_x_given_y_l1505_150569

theorem find_x_given_y (x y : ℤ) (h1 : 16 * (4 : ℝ)^x = 3^(y + 2)) (h2 : y = -2) : x = -2 := by
  sorry

end find_x_given_y_l1505_150569


namespace surface_area_is_726_l1505_150551

def edge_length : ℝ := 11

def surface_area_of_cube (e : ℝ) : ℝ := 6 * (e * e)

theorem surface_area_is_726 (h : edge_length = 11) : surface_area_of_cube edge_length = 726 := by
  sorry

end surface_area_is_726_l1505_150551


namespace complex_number_in_second_quadrant_l1505_150598

open Complex

theorem complex_number_in_second_quadrant (z : ℂ) :
  (Complex.abs z = Real.sqrt 7) →
  (z.re < 0 ∧ z.im > 0) →
  z = -2 + Real.sqrt 3 * Complex.I :=
by
  intros h1 h2
  sorry

end complex_number_in_second_quadrant_l1505_150598


namespace integer_solution_inequalities_l1505_150574

theorem integer_solution_inequalities (x : ℤ) (h1 : x + 12 > 14) (h2 : -3 * x > -9) : x = 2 :=
by
  sorry

end integer_solution_inequalities_l1505_150574


namespace transport_tax_correct_l1505_150556

def engine_power : ℕ := 250
def tax_rate : ℕ := 75
def months_owned : ℕ := 2
def months_in_year : ℕ := 12
def annual_tax : ℕ := engine_power * tax_rate
def adjusted_tax : ℕ := (annual_tax * months_owned) / months_in_year

theorem transport_tax_correct :
  adjusted_tax = 3125 :=
by
  sorry

end transport_tax_correct_l1505_150556


namespace smallest_integer_b_gt_4_base_b_perfect_square_l1505_150583

theorem smallest_integer_b_gt_4_base_b_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ n : ℕ, 2 * b + 5 = n^2 ∧ b = 10 :=
by
  sorry

end smallest_integer_b_gt_4_base_b_perfect_square_l1505_150583


namespace susan_spent_75_percent_l1505_150511

variables (B b s : ℝ)

-- Conditions
def condition1 : Prop := b = 0.25 * (B - 3 * s)
def condition2 : Prop := s = 0.10 * (B - 2 * b)

-- Theorem
theorem susan_spent_75_percent (h1 : condition1 B b s) (h2 : condition2 B b s) : b + s = 0.75 * B := 
sorry

end susan_spent_75_percent_l1505_150511


namespace third_smallest_abc_sum_l1505_150585

-- Define the necessary conditions and properties
def isIntegerRoots (a b c : ℕ) : Prop :=
  ∃ r1 r2 r3 r4 : ℤ, 
    a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 - c = 0 ∧ 
    a * r3^2 - b * r3 + c = 0 ∧ a * r4^2 - b * r4 - c = 0

-- State the main theorem
theorem third_smallest_abc_sum : ∃ a b c : ℕ, 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ isIntegerRoots a b c ∧ 
  (a + b + c = 35 ∧ a = 1 ∧ b = 10 ∧ c = 24) :=
by sorry

end third_smallest_abc_sum_l1505_150585


namespace proof_equivalence_l1505_150573

variable {x y : ℝ}

theorem proof_equivalence (h : x - y = 1) : x^3 - 3 * x * y - y^3 = 1 := by
  sorry

end proof_equivalence_l1505_150573


namespace gain_percentage_is_66_67_l1505_150579

variable (C S : ℝ)
variable (cost_price_eq : 20 * C = 12 * S)

theorem gain_percentage_is_66_67 (h : 20 * C = 12 * S) : (((5 / 3) * C - C) / C) * 100 = 66.67 := by
  sorry

end gain_percentage_is_66_67_l1505_150579


namespace find_number_of_hens_l1505_150546

def hens_and_cows_problem (H C : ℕ) : Prop :=
  (H + C = 50) ∧ (2 * H + 4 * C = 144)

theorem find_number_of_hens (H C : ℕ) (hc : hens_and_cows_problem H C) : H = 28 :=
by {
  -- We assume the problem conditions and skip the proof using sorry
  sorry
}

end find_number_of_hens_l1505_150546


namespace assignment_plans_l1505_150521

theorem assignment_plans {students towns : ℕ} (h_students : students = 5) (h_towns : towns = 3) :
  ∃ plans : ℕ, plans = 150 :=
by
  -- Given conditions
  have h1 : students = 5 := h_students
  have h2 : towns = 3 := h_towns
  
  -- The required number of assignment plans
  existsi 150
  -- Proof is not supplied
  sorry

end assignment_plans_l1505_150521


namespace inequality_proof_l1505_150577

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)^2 + bc / (b + c)^2 + ca / (c + a)^2) + (3 * (a^2 + b^2 + c^2)) / (a + b + c)^2 ≥ 7 / 4 := 
by
  sorry

end inequality_proof_l1505_150577


namespace findPositiveRealSolutions_l1505_150568

noncomputable def onlySolutions (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (a^2 - b * d) / (b + 2 * c + d) +
  (b^2 - c * a) / (c + 2 * d + a) +
  (c^2 - d * b) / (d + 2 * a + b) +
  (d^2 - a * c) / (a + 2 * b + c) = 0

theorem findPositiveRealSolutions :
  ∀ a b c d : ℝ,
  onlySolutions a b c d →
  ∃ k m : ℝ, k > 0 ∧ m > 0 ∧ a = k ∧ b = m ∧ c = k ∧ d = m :=
by
  intros a b c d h
  -- proof steps (if required) go here
  sorry

end findPositiveRealSolutions_l1505_150568


namespace bill_cooking_time_l1505_150564

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end bill_cooking_time_l1505_150564


namespace evaluate_f_l1505_150557

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem evaluate_f : f (f (f (-1))) = Real.pi + 1 :=
by
  -- Proof goes here
  sorry

end evaluate_f_l1505_150557
