import Mathlib

namespace john_burritos_left_l186_186322

def total_burritos (b1 b2 b3 b4 : ℕ) : ℕ :=
  b1 + b2 + b3 + b4

def burritos_left_after_giving_away (total : ℕ) (fraction : ℕ) : ℕ :=
  total - (total / fraction)

def burritos_left_after_eating (burritos_left : ℕ) (burritos_per_day : ℕ) (days : ℕ) : ℕ :=
  burritos_left - (burritos_per_day * days)

theorem john_burritos_left :
  let b1 := 15
  let b2 := 20
  let b3 := 25
  let b4 := 5
  let total := total_burritos b1 b2 b3 b4
  let burritos_after_give_away := burritos_left_after_giving_away total 3
  let burritos_after_eating := burritos_left_after_eating burritos_after_give_away 3 10
  burritos_after_eating = 14 :=
by
  sorry

end john_burritos_left_l186_186322


namespace path_to_tile_ratio_l186_186396

theorem path_to_tile_ratio
  (t p : ℝ) 
  (tiles : ℕ := 400)
  (grid_size : ℕ := 20)
  (total_tile_area : ℝ := (tiles : ℝ) * t^2)
  (total_courtyard_area : ℝ := (grid_size * (t + 2 * p))^2) 
  (tile_area_fraction : ℝ := total_tile_area / total_courtyard_area) : 
  tile_area_fraction = 0.25 → 
  p / t = 0.5 :=
by
  intro h
  sorry

end path_to_tile_ratio_l186_186396


namespace rectangle_length_width_l186_186114

theorem rectangle_length_width 
  (x y : ℚ)
  (h1 : x - 5 = y + 2)
  (h2 : x * y = (x - 5) * (y + 2)) :
  x = 25 / 3 ∧ y = 4 / 3 :=
by
  sorry

end rectangle_length_width_l186_186114


namespace second_number_is_12_l186_186827

noncomputable def expression := (26.3 * 12 * 20) / 3 + 125

theorem second_number_is_12 :
  expression = 2229 → 12 = 12 :=
by sorry

end second_number_is_12_l186_186827


namespace arcsin_half_eq_pi_six_l186_186862

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l186_186862


namespace tim_movie_marathon_duration_is_9_l186_186455

-- Define the conditions:
def first_movie_duration : ℕ := 2
def second_movie_duration : ℕ := first_movie_duration + (first_movie_duration / 2)
def combined_duration_first_two_movies : ℕ := first_movie_duration + second_movie_duration
def third_movie_duration : ℕ := combined_duration_first_two_movies - 1
def total_marathon_duration : ℕ := first_movie_duration + second_movie_duration + third_movie_duration

-- The theorem to prove the marathon duration is 9 hours
theorem tim_movie_marathon_duration_is_9 :
  total_marathon_duration = 9 :=
by sorry

end tim_movie_marathon_duration_is_9_l186_186455


namespace find_ff_of_five_half_l186_186898

noncomputable def f (x : ℝ) : ℝ :=
if x <= 1 then 2^x - 2 else Real.log x / Real.log 2

theorem find_ff_of_five_half : f (f (5/2)) = -1/2 := by
  sorry

end find_ff_of_five_half_l186_186898


namespace seven_n_form_l186_186903

theorem seven_n_form (n : ℤ) (a b : ℤ) (h : 7 * n = a^2 + 3 * b^2) : 
  ∃ c d : ℤ, n = c^2 + 3 * d^2 :=
by {
  sorry
}

end seven_n_form_l186_186903


namespace total_age_proof_l186_186934

variable (K : ℕ) -- Kaydence's age
variable (T : ℕ) -- Total age of people in the gathering

def Kaydence_father_age : ℕ := 60
def Kaydence_mother_age : ℕ := Kaydence_father_age - 2
def Kaydence_brother_age : ℕ := Kaydence_father_age / 2
def Kaydence_sister_age : ℕ := 40
def elder_cousin_age : ℕ := Kaydence_brother_age + 2 * Kaydence_sister_age
def younger_cousin_age : ℕ := elder_cousin_age / 2 + 3
def grandmother_age : ℕ := 3 * Kaydence_mother_age - 5

theorem total_age_proof (K : ℕ) : T = 525 + K :=
by 
  sorry

end total_age_proof_l186_186934


namespace products_B_correct_l186_186481

-- Define the total number of products
def total_products : ℕ := 4800

-- Define the sample size and the number of pieces from equipment A in the sample
def sample_size : ℕ := 80
def sample_A : ℕ := 50

-- Define the number of products produced by equipment A and B
def products_A : ℕ := 3000
def products_B : ℕ := total_products - products_A

-- The target number of products produced by equipment B
def target_products_B : ℕ := 1800

-- The theorem we need to prove
theorem products_B_correct :
  products_B = target_products_B := by
  sorry

end products_B_correct_l186_186481


namespace avg_growth_rate_eq_l186_186310

variable (x : ℝ)

theorem avg_growth_rate_eq :
  (560 : ℝ) * (1 + x)^2 = 830 :=
sorry

end avg_growth_rate_eq_l186_186310


namespace range_of_a_l186_186265

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry

end range_of_a_l186_186265


namespace discount_percent_l186_186710

theorem discount_percent (CP MP SP : ℝ) (markup profit: ℝ) (h1 : CP = 100) (h2 : MP = CP + (markup * CP))
  (h3 : SP = CP + (profit * CP)) (h4 : markup = 0.75) (h5 : profit = 0.225) : 
  (MP - SP) / MP * 100 = 30 :=
by
  sorry

end discount_percent_l186_186710


namespace intersection_of_A_and_B_l186_186505

-- Define set A
def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

-- Define set B
def B : Set ℤ := {2, 4, 6, 8}

-- Prove that the intersection of set A and set B is {2, 4}.
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
by
  sorry

end intersection_of_A_and_B_l186_186505


namespace proof_a_l186_186847

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (y - 3) / (x - 2) = 3}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ a * x + 2 * y + a = 0}

-- Given conditions that M ∩ N = ∅, prove that a = -6 or a = -2
theorem proof_a (h : ∃ a : ℝ, (N a ∩ M = ∅)) : ∃ a : ℝ, a = -6 ∨ a = -2 :=
  sorry

end proof_a_l186_186847


namespace weighted_average_score_l186_186288

def weight (subject_mark : Float) (weight_percentage : Float) : Float :=
    subject_mark * weight_percentage

theorem weighted_average_score :
    (weight 61 0.2) + (weight 65 0.25) + (weight 82 0.3) + (weight 67 0.15) + (weight 85 0.1) = 71.6 := by
    sorry

end weighted_average_score_l186_186288


namespace probability_square_not_touching_vertex_l186_186602

theorem probability_square_not_touching_vertex :
  let total_squares := 64
  let squares_touching_vertices := 16
  let squares_not_touching_vertices := total_squares - squares_touching_vertices
  let probability := (squares_not_touching_vertices : ℚ) / total_squares
  probability = 3 / 4 :=
by
  sorry

end probability_square_not_touching_vertex_l186_186602


namespace find_number_l186_186305

theorem find_number (x : ℝ) (h : (1/3) * x = 12) : x = 36 :=
sorry

end find_number_l186_186305


namespace leftover_coverage_l186_186937

variable (bagCoverage lawnLength lawnWidth bagsPurchased : ℕ)

def area_of_lawn (length width : ℕ) : ℕ :=
  length * width

def total_coverage (bagCoverage bags : ℕ) : ℕ :=
  bags * bagCoverage

theorem leftover_coverage :
  let lawnLength := 22
  let lawnWidth := 36
  let bagCoverage := 250
  let bagsPurchased := 4
  let lawnArea := area_of_lawn lawnLength lawnWidth
  let totalSeedCoverage := total_coverage bagCoverage bagsPurchased
  totalSeedCoverage - lawnArea = 208 := by
  sorry

end leftover_coverage_l186_186937


namespace range_of_m_l186_186412

variable {f : ℝ → ℝ}

theorem range_of_m 
  (even_f : ∀ x : ℝ, f x = f (-x))
  (mono_f : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h : f (m + 1) < f (3 * m - 1)) :
  m > 1 ∨ m < 0 :=
sorry

end range_of_m_l186_186412


namespace probability_not_face_card_l186_186420

-- Definitions based on the conditions
def total_cards : ℕ := 52
def face_cards  : ℕ := 12
def non_face_cards : ℕ := total_cards - face_cards

-- Statement of the theorem
theorem probability_not_face_card : (non_face_cards : ℚ) / (total_cards : ℚ) = 10 / 13 := by
  sorry

end probability_not_face_card_l186_186420


namespace oranges_in_bin_l186_186532

variable (n₀ n_throw n_new : ℕ)

theorem oranges_in_bin (h₀ : n₀ = 50) (h_throw : n_throw = 40) (h_new : n_new = 24) : 
  n₀ - n_throw + n_new = 34 := 
by 
  sorry

end oranges_in_bin_l186_186532


namespace min_value_frac_l186_186416

theorem min_value_frac (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) (h3 : a * c = 4) : 
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, y = (1 / c + 9 / a) → y ≥ x :=
by sorry

end min_value_frac_l186_186416


namespace euler_school_voting_problem_l186_186964

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

end euler_school_voting_problem_l186_186964


namespace grain_spilled_l186_186603

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled : (original_grain - remaining_grain) = 49952 :=
by
  -- Proof goes here
  sorry

end grain_spilled_l186_186603


namespace total_walking_time_l186_186038

open Nat

def walking_time (distance speed : ℕ) : ℕ :=
distance / speed

def number_of_rests (distance : ℕ) : ℕ :=
(distance / 10) - 1

def resting_time_in_minutes (rests : ℕ) : ℕ :=
rests * 5

def resting_time_in_hours (rest_time : ℕ) : ℚ :=
rest_time / 60

def total_time (walking_time resting_time : ℚ) : ℚ :=
walking_time + resting_time

theorem total_walking_time (distance speed : ℕ) (rest_per_10 : ℕ) (rest_time : ℕ) :
  speed = 10 →
  rest_per_10 = 10 →
  rest_time = 5 →
  distance = 50 →
  total_time (walking_time distance speed) (resting_time_in_hours (resting_time_in_minutes (number_of_rests distance))) = 5 + 1 / 3 :=
sorry

end total_walking_time_l186_186038


namespace guess_probability_l186_186121

-- Definitions based on the problem conditions
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

def possible_attempts : ℕ := (5 * 4) -- A^2_5

def favorable_outcomes : ℕ := (4 * 2) -- C^1_4 * A^2_2

noncomputable def probability_correct_guess : ℝ :=
  (favorable_outcomes : ℝ) / (possible_attempts : ℝ)

-- Lean statement for the proof problem
theorem guess_probability : probability_correct_guess = 2 / 5 := by
  sorry

end guess_probability_l186_186121


namespace maximize_S_n_l186_186673

-- Define the general term of the sequence and the sum of the first n terms.
def a_n (n : ℕ) : ℤ := -2 * n + 25

def S_n (n : ℕ) : ℤ := 24 * n - n^2

-- The main statement to prove
theorem maximize_S_n : ∃ (n : ℕ), n = 11 ∧ ∀ m, S_n m ≤ S_n 11 :=
  sorry

end maximize_S_n_l186_186673


namespace range_of_x_plus_y_l186_186080

theorem range_of_x_plus_y (x y : ℝ) (h : x^2 + 2 * x * y - 1 = 0) : (x + y ≤ -1 ∨ x + y ≥ 1) :=
by
  sorry

end range_of_x_plus_y_l186_186080


namespace hearts_total_shaded_area_l186_186377

theorem hearts_total_shaded_area (A B C D : ℕ) (hA : A = 1) (hB : B = 4) (hC : C = 9) (hD : D = 16) :
  (D - C) + (B - A) = 10 := 
by 
  sorry

end hearts_total_shaded_area_l186_186377


namespace intersection_complement_correct_l186_186090

open Set

def A : Set ℕ := {x | (x + 4) * (x - 5) ≤ 0}
def B : Set ℕ := {x | x < 2}
def U : Set ℕ := { x | True }

theorem intersection_complement_correct :
  (A ∩ (U \ B)) = {x | x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5} :=
by
  sorry

end intersection_complement_correct_l186_186090


namespace polygon_with_given_angle_sums_is_hexagon_l186_186586

theorem polygon_with_given_angle_sums_is_hexagon
  (n : ℕ)
  (h_interior : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end polygon_with_given_angle_sums_is_hexagon_l186_186586


namespace sqrt_eq_cond_l186_186081

theorem sqrt_eq_cond (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (not_perfect_square_a : ¬(∃ n : ℕ, n * n = a)) (not_perfect_square_b : ¬(∃ n : ℕ, n * n = b))
    (not_perfect_square_c : ¬(∃ n : ℕ, n * n = c)) :
    (Real.sqrt a + Real.sqrt b = Real.sqrt c) →
    (2 * Real.sqrt (a * b) = c - (a + b) ∧ (∃ k : ℕ, a * b = k * k)) :=
sorry

end sqrt_eq_cond_l186_186081


namespace Greenwood_High_School_chemistry_students_l186_186860

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

end Greenwood_High_School_chemistry_students_l186_186860


namespace negation_necessary_but_not_sufficient_l186_186192

def P (x : ℝ) : Prop := |x - 2| ≥ 1
def Q (x : ℝ) : Prop := x^2 - 3 * x + 2 ≥ 0

theorem negation_necessary_but_not_sufficient (x : ℝ) :
  (¬ P x → ¬ Q x) ∧ ¬ (¬ Q x → ¬ P x) :=
by
  sorry

end negation_necessary_but_not_sufficient_l186_186192


namespace smallest_rat_num_l186_186942

theorem smallest_rat_num (a b c d : ℚ) (ha : a = -6 / 7) (hb : b = 2) (hc : c = 0) (hd : d = -1) :
  min (min a (min b c)) d = -1 :=
sorry

end smallest_rat_num_l186_186942


namespace sample_capacity_l186_186836

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ)
  (h1 : frequency = 30)
  (h2 : frequency_rate = 25 / 100) :
  n = 120 :=
by
  sorry

end sample_capacity_l186_186836


namespace number_of_grouping_methods_l186_186385

theorem number_of_grouping_methods : 
  let males := 5
  let females := 3
  let groups := 2
  let select_males := Nat.choose males groups
  let select_females := Nat.choose females groups
  let permute := Nat.factorial groups
  select_males * select_females * permute * permute = 60 :=
by 
  sorry

end number_of_grouping_methods_l186_186385


namespace range_of_m_l186_186992

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x + α / x + Real.log x

theorem range_of_m (e l : ℝ) (alpha : ℝ) :
  (∀ (α : ℝ), α ∈ Set.Icc (1 / Real.exp 1) (2 * Real.exp 1 ^ 2) → 
  ∀ (x : ℝ), x ∈ Set.Icc l e → f alpha x < m) →
  m ∈ Set.Ioi (1 + 2 * Real.exp 1 ^ 2) := sorry

end range_of_m_l186_186992


namespace second_train_length_is_correct_l186_186043

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kmph : ℝ) (speed_second_train_kmph : ℝ) (time_crossing_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kmph * (1000 / 3600)
  let speed_second_train_mps := speed_second_train_kmph * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_crossing_seconds
  total_distance - length_first_train

theorem second_train_length_is_correct : length_of_second_train 360 120 80 9 = 139.95 :=
by
  sorry

end second_train_length_is_correct_l186_186043


namespace third_number_hcf_lcm_l186_186150

theorem third_number_hcf_lcm (N : ℕ) 
  (HCF : Nat.gcd (Nat.gcd 136 144) N = 8)
  (LCM : Nat.lcm (Nat.lcm 136 144) N = 2^4 * 3^2 * 17 * 7) : 
  N = 7 := 
  sorry

end third_number_hcf_lcm_l186_186150


namespace range_f_l186_186891

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2 * x + 1)

theorem range_f : (Set.range f) = Set.univ := by
  sorry

end range_f_l186_186891


namespace comparison_17_pow_14_31_pow_11_l186_186169

theorem comparison_17_pow_14_31_pow_11 : 17^14 > 31^11 :=
by
  sorry

end comparison_17_pow_14_31_pow_11_l186_186169


namespace inequality_proof_l186_186064

open Real

-- Define the conditions
def conditions (a b c : ℝ) := (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b * c = 1)

-- Express the inequality we need to prove
def inequality (a b c : ℝ) :=
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1

-- Statement of the theorem
theorem inequality_proof (a b c : ℝ) (h : conditions a b c) : inequality a b c :=
by {
  sorry
}

end inequality_proof_l186_186064


namespace random_variable_prob_l186_186057

theorem random_variable_prob (n : ℕ) (h : (3 : ℝ) / n = 0.3) : n = 10 :=
sorry

end random_variable_prob_l186_186057


namespace red_toys_removed_l186_186912

theorem red_toys_removed (R W : ℕ) (h1 : R + W = 134) (h2 : 2 * W = 88) (h3 : R - 2 * W / 2 = 88) : R - 88 = 2 :=
by {
  sorry
}

end red_toys_removed_l186_186912


namespace sum_of_sixth_powers_l186_186168

theorem sum_of_sixth_powers (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 0) 
  (h2 : α₁^2 + α₂^2 + α₃^2 = 2) 
  (h3 : α₁^3 + α₂^3 + α₃^3 = 4) : 
  α₁^6 + α₂^6 + α₃^6 = 7 :=
sorry

end sum_of_sixth_powers_l186_186168


namespace num_mappings_from_A_to_A_is_4_l186_186918

-- Define the number of elements in set A
def set_A_card := 2

-- Define the proof problem
theorem num_mappings_from_A_to_A_is_4 (h : set_A_card = 2) : (set_A_card ^ set_A_card) = 4 :=
by
  sorry

end num_mappings_from_A_to_A_is_4_l186_186918


namespace jane_exercises_40_hours_l186_186023

-- Define the conditions
def hours_per_day : ℝ := 1
def days_per_week : ℝ := 5
def weeks : ℝ := 8

-- Define total_hours using the conditions
def total_hours : ℝ := (hours_per_day * days_per_week) * weeks

-- The theorem stating the result
theorem jane_exercises_40_hours :
  total_hours = 40 := by
  sorry

end jane_exercises_40_hours_l186_186023


namespace two_abc_square_l186_186691

variable {R : Type*} [Ring R] [Fintype R]

-- Given condition: For any a, b ∈ R, ∃ c ∈ R such that a^2 + b^2 = c^2.
axiom ring_property (a b : R) : ∃ c : R, a^2 + b^2 = c^2

-- We need to prove: For any a, b, c ∈ R, ∃ d ∈ R such that 2abc = d^2.
theorem two_abc_square (a b c : R) : ∃ d : R, 2 * (a * b * c) = d^2 :=
by
  sorry

end two_abc_square_l186_186691


namespace derivative_at_one_l186_186926

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : deriv f 1 = 2 * Real.exp 1 := by
sorry

end derivative_at_one_l186_186926


namespace water_needed_in_pints_l186_186641

-- Define the input data
def parts_water : ℕ := 5
def parts_lemon : ℕ := 2
def pints_per_gallon : ℕ := 8
def total_gallons : ℕ := 3

-- Define the total parts of the mixture
def total_parts : ℕ := parts_water + parts_lemon

-- Define the total pints of lemonade
def total_pints : ℕ := total_gallons * pints_per_gallon

-- Define the pints per part of the mixture
def pints_per_part : ℚ := total_pints / total_parts

-- Define the total pints of water needed
def pints_water : ℚ := parts_water * pints_per_part

-- The theorem stating what we need to prove
theorem water_needed_in_pints : pints_water = 17 + 1 / 7 := by
  sorry

end water_needed_in_pints_l186_186641


namespace hypotenuse_is_correct_l186_186929

noncomputable def hypotenuse_of_right_triangle (a b : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_is_correct :
  hypotenuse_of_right_triangle 140 210 = 70 * Real.sqrt 13 :=
by
  sorry

end hypotenuse_is_correct_l186_186929


namespace chickens_and_rabbits_l186_186792

-- Let x be the number of chickens and y be the number of rabbits
variables (x y : ℕ)

-- Conditions: There are 35 heads and 94 feet in total
def heads_eq : Prop := x + y = 35
def feet_eq : Prop := 2 * x + 4 * y = 94

-- Proof statement (no proof is required, so we use sorry)
theorem chickens_and_rabbits :
  (heads_eq x y) ∧ (feet_eq x y) ↔ (x + y = 35 ∧ 2 * x + 4 * y = 94) :=
by
  sorry

end chickens_and_rabbits_l186_186792


namespace solution_set_f_cos_x_l186_186004

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x < 3 then -(x-2)^2 + 1
else if x = 0 then 0
else if -3 < x ∧ x < 0 then (x+2)^2 - 1
else 0 -- Defined as 0 outside the given interval for simplicity

theorem solution_set_f_cos_x :
  {x : ℝ | f x * Real.cos x < 0} = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 3)} :=
sorry

end solution_set_f_cos_x_l186_186004


namespace A_subset_B_l186_186386

def A : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 4 * k + 1 }
def B : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 }

theorem A_subset_B : A ⊆ B :=
  sorry

end A_subset_B_l186_186386


namespace set_equality_l186_186578

def P : Set ℝ := { x | x^2 = 1 }

theorem set_equality : P = {-1, 1} :=
by
  sorry

end set_equality_l186_186578


namespace combined_degrees_l186_186678

-- Definitions based on conditions
def summer_degrees : ℕ := 150
def jolly_degrees (summer_degrees : ℕ) : ℕ := summer_degrees - 5

-- Theorem stating the combined degrees
theorem combined_degrees : summer_degrees + jolly_degrees summer_degrees = 295 :=
by
  sorry

end combined_degrees_l186_186678


namespace min_distance_convex_lens_l186_186649

theorem min_distance_convex_lens (t k f : ℝ) (hf : f > 0) (ht : t ≥ f)
    (h_lens: 1 / t + 1 / k = 1 / f) :
  t = 2 * f → t + k = 4 * f :=
by
  sorry

end min_distance_convex_lens_l186_186649


namespace necessary_condition_not_sufficient_condition_main_l186_186264

example (x : ℝ) : (x^2 - 3 * x > 0) → (x > 4) ∨ (x < 0 ∧ x > 0) := by
  sorry

theorem necessary_condition (x : ℝ) :
  (x^2 - 3 * x > 0) → (x > 4) :=
by
  sorry

theorem not_sufficient_condition (x : ℝ) :
  ¬ (x > 4) → (x^2 - 3 * x > 0) :=
by
  sorry

theorem main (x : ℝ) :
  (x^2 - 3 * x > 0) ↔ ¬ (x > 4) :=
by
  sorry

end necessary_condition_not_sufficient_condition_main_l186_186264


namespace enclosed_area_correct_l186_186316

noncomputable def enclosed_area : ℝ :=
  ∫ x in (1/2)..2, (-x + 5/2 - 1/x)

theorem enclosed_area_correct :
  enclosed_area = (15/8) - 2 * Real.log 2 :=
by
  sorry

end enclosed_area_correct_l186_186316


namespace determinant_of_A_l186_186612

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![ -5, 8],
    ![ 3, -4]]

theorem determinant_of_A : A.det = -4 := by
  sorry

end determinant_of_A_l186_186612


namespace polynomial_simplification_l186_186593

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 :=
by
  sorry

end polynomial_simplification_l186_186593


namespace problem1_sol_l186_186894

noncomputable def problem1 :=
  let total_people := 200
  let avg_feelings_total := 70
  let female_total := 100
  let a := 30 -- derived from 2a + (70 - a) = 100
  let chi_square := 200 * (70 * 40 - 30 * 60) ^ 2 / (130 * 70 * 100 * 100)
  let k_95 := 3.841 -- critical value for 95% confidence
  let p_xi_2 := (1 / 3)
  let p_xi_3 := (1 / 2)
  let p_xi_4 := (1 / 6)
  let exi := (2 * (1 / 3)) + (3 * (1 / 2)) + (4 * (1 / 6))
  chi_square < k_95 ∧ exi = 17 / 6

theorem problem1_sol : problem1 :=
  by {
    sorry
  }

end problem1_sol_l186_186894


namespace maximum_value_of_f_l186_186285

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem maximum_value_of_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = (Real.pi / 6) + Real.sqrt 3 ∧ 
  ∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f (Real.pi / 6) :=
by
  sorry

end maximum_value_of_f_l186_186285


namespace interval_contains_root_l186_186158

noncomputable def f (x : ℝ) : ℝ := 3^x - x^2

theorem interval_contains_root : ∃ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), f x = 0 :=
by
  have f_neg : f (-1) < 0 := by sorry
  have f_zero : f 0 > 0 := by sorry
  sorry

end interval_contains_root_l186_186158


namespace diameter_outer_boundary_correct_l186_186498

noncomputable def diameter_outer_boundary 
  (D_fountain : ℝ)
  (w_gardenRing : ℝ)
  (w_innerPath : ℝ)
  (w_outerPath : ℝ) : ℝ :=
  let R_fountain := D_fountain / 2
  let R_innerPath := R_fountain + w_gardenRing
  let R_outerPathInner := R_innerPath + w_innerPath
  let R_outerPathOuter := R_outerPathInner + w_outerPath
  2 * R_outerPathOuter

theorem diameter_outer_boundary_correct :
  diameter_outer_boundary 10 12 3 4 = 48 := by
  -- skipping proof
  sorry

end diameter_outer_boundary_correct_l186_186498


namespace patrick_purchased_pencils_l186_186275

theorem patrick_purchased_pencils 
  (S : ℝ) -- selling price of one pencil
  (C : ℝ) -- cost price of one pencil
  (P : ℕ) -- number of pencils purchased
  (h1 : C = 1.3333333333333333 * S) -- condition 1: cost of pencils is 1.3333333 times the selling price
  (h2 : (P : ℝ) * C - (P : ℝ) * S = 20 * S) -- condition 2: loss equals selling price of 20 pencils
  : P = 60 := 
sorry

end patrick_purchased_pencils_l186_186275


namespace problem_ACD_l186_186049

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

theorem problem_ACD (a : ℝ) :
  (f a 0 = (2/3) ∧
  ¬(∀ x, f a x ≥ 0 → ((a ≥ 1) ∨ (a ≤ -1))) ∧
  (∃ x1 x2, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ∧
  (∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0)) :=
sorry

end problem_ACD_l186_186049


namespace jill_vs_jack_arrival_time_l186_186338

def distance_to_park : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem jill_vs_jack_arrival_time :
  let jill_time := distance_to_park / jill_speed
  let jack_time := distance_to_park / jack_speed
  let jill_time_minutes := jill_time * 60
  let jack_time_minutes := jack_time * 60
  jill_time_minutes < jack_time_minutes ∧ jack_time_minutes - jill_time_minutes = 5.4 :=
by
  sorry

end jill_vs_jack_arrival_time_l186_186338


namespace division_equals_fraction_l186_186506

theorem division_equals_fraction:
  180 / (8 + 9 * 3 - 4) = 180 / 31 := 
by
  sorry

end division_equals_fraction_l186_186506


namespace greatest_value_of_a_l186_186875

theorem greatest_value_of_a (a : ℝ) : a^2 - 12 * a + 32 ≤ 0 → a ≤ 8 :=
by
  sorry

end greatest_value_of_a_l186_186875


namespace laura_running_speed_l186_186482

theorem laura_running_speed (x : ℝ) (hx : 3 * x + 1 > 0) : 
    (30 / (3 * x + 1)) + (10 / x) = 31 / 12 → x = 7.57 := 
by 
  sorry

end laura_running_speed_l186_186482


namespace original_days_to_finish_work_l186_186003

theorem original_days_to_finish_work : 
  ∀ (D : ℕ), 
  (∃ (W : ℕ), 15 * D * W = 25 * (D - 3) * W) → 
  D = 8 :=
by
  intros D h
  sorry

end original_days_to_finish_work_l186_186003


namespace service_center_location_l186_186146

theorem service_center_location : 
  ∀ (milepost4 milepost9 : ℕ), 
  milepost4 = 30 → milepost9 = 150 → 
  (∃ milepost_service_center : ℕ, milepost_service_center = milepost4 + ((milepost9 - milepost4) / 2)) → 
  milepost_service_center = 90 :=
by
  intros milepost4 milepost9 h4 h9 hsc
  sorry

end service_center_location_l186_186146


namespace solution_set_of_inequality_l186_186941

theorem solution_set_of_inequality (x : ℝ) : (x^2 - |x| > 0) ↔ (x < -1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l186_186941


namespace solve_equation_l186_186507

theorem solve_equation (x : ℝ) (h : x > 0) :
  25^(Real.log x / Real.log 4) - 5^(Real.log (x^2) / Real.log 16 + 1) = Real.log (9 * Real.sqrt 3) / Real.log (Real.sqrt 3) - 25^(Real.log x / Real.log 16) ->
  x = 4 :=
by
  sorry

end solve_equation_l186_186507


namespace abs_p_minus_1_ge_2_l186_186997

theorem abs_p_minus_1_ge_2 (p : ℝ) (a : ℕ → ℝ) 
  (h₀ : a 0 = 1)
  (h₁ : a 1 = p)
  (h₂ : a 2 = p * (p - 1))
  (h₃ : ∀ n : ℕ, a (n + 3) = p * a (n + 2) - p * a (n + 1) + a n)
  (h₄ : ∀ n : ℕ, a n > 0)
  (h₅ : ∀ m n : ℕ, m ≥ n → a m * a n > a (m + 1) * a (n - 1)) :
  |p - 1| ≥ 2 :=
sorry

end abs_p_minus_1_ge_2_l186_186997


namespace infinite_sum_equals_l186_186086

theorem infinite_sum_equals :
  10 * (79 * (1 / 7)) + (∑' n : ℕ, if n % 2 = 0 then (if n = 0 then 0 else 2 / 7 ^ n) else (1 / 7 ^ n)) = 3 / 16 :=
by
  sorry

end infinite_sum_equals_l186_186086


namespace factor_polynomial_l186_186244

theorem factor_polynomial :
  ∀ x : ℝ, 
  (x^2 + 4 * x + 3) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) 
  = (x^2 + 6 * x + 1) * (x^2 + 6 * x + 37) :=
by
  intros x
  sorry

end factor_polynomial_l186_186244


namespace number_of_friends_l186_186726

theorem number_of_friends (n : ℕ) (total_bill : ℕ) :
  (total_bill = 12 * (n + 2)) → (total_bill = 16 * n) → n = 6 :=
by
  sorry

end number_of_friends_l186_186726


namespace sticker_arrangement_l186_186640

theorem sticker_arrangement : 
  ∀ (n : ℕ), n = 35 → 
  (∀ k : ℕ, k = 8 → 
    ∃ m : ℕ, m = 5 ∧ (n + m) % k = 0) := 
by sorry

end sticker_arrangement_l186_186640


namespace sample_size_l186_186988

theorem sample_size (k n : ℕ) (h_ratio : 4 * k + k + 5 * k = n) 
  (h_middle_aged : 10 * (4 + 1 + 5) = n) : n = 100 := 
by
  sorry

end sample_size_l186_186988


namespace egg_price_l186_186619

theorem egg_price (num_eggs capital_remaining : ℕ) (total_cost price_per_egg : ℝ)
  (h1 : num_eggs = 30)
  (h2 : capital_remaining = 5)
  (h3 : total_cost = 5)
  (h4 : num_eggs - capital_remaining = 25)
  (h5 : 25 * price_per_egg = total_cost) :
  price_per_egg = 0.20 := sorry

end egg_price_l186_186619


namespace approximate_number_of_fish_in_pond_l186_186106

-- Define the conditions as hypotheses.
def tagged_fish_caught_first : ℕ := 50
def total_fish_caught_second : ℕ := 50
def tagged_fish_found_second : ℕ := 5

-- Define total fish in the pond.
def total_fish_in_pond (N : ℝ) : Prop :=
  tagged_fish_found_second / total_fish_caught_second = tagged_fish_caught_first / N

-- The statement to be proved.
theorem approximate_number_of_fish_in_pond (N : ℝ) (h : total_fish_in_pond N) : N = 500 :=
sorry

end approximate_number_of_fish_in_pond_l186_186106


namespace grasshopper_jump_l186_186186

theorem grasshopper_jump :
  ∃ (x y : ℤ), 80 * x - 50 * y = 170 ∧ x + y ≤ 7 := by
  sorry

end grasshopper_jump_l186_186186


namespace coefficient_x3_expansion_l186_186156

open Finset -- To use binomial coefficients and summation

theorem coefficient_x3_expansion (x : ℝ) : 
  (2 + x) ^ 3 = 8 + 12 * x + 6 * x^2 + 1 * x^3 :=
by
  sorry

end coefficient_x3_expansion_l186_186156


namespace question1_question2_question3_l186_186758

-- Define the scores and relevant statistics for seventh and eighth grades
def seventh_grade_scores : List ℕ := [96, 86, 96, 86, 99, 96, 90, 100, 89, 82]
def eighth_grade_C_scores : List ℕ := [94, 90, 92]
def total_eighth_grade_students : ℕ := 800

def a := 40
def b := 93
def c := 96

-- Define given statistics from the table
def seventh_grade_mean := 92
def seventh_grade_variance := 34.6
def eighth_grade_mean := 91
def eighth_grade_median := 93
def eighth_grade_mode := 100
def eighth_grade_variance := 50.4

-- Proof for question 1
theorem question1 : (a = 40) ∧ (b = 93) ∧ (c = 96) :=
by sorry

-- Proof for question 2 (stability comparison)
theorem question2 : seventh_grade_variance < eighth_grade_variance :=
by sorry

-- Proof for question 3 (estimating number of excellent students)
theorem question3 : (7 / 10 : ℝ) * total_eighth_grade_students = 560 :=
by sorry

end question1_question2_question3_l186_186758


namespace cylinder_base_area_l186_186471

-- Definitions: Adding variables and hypotheses based on the problem statement.
variable (A_c A_r : ℝ) -- Base areas of the cylinder and the rectangular prism
variable (h1 : 8 * A_c = 6 * A_r) -- Condition from the rise in water levels
variable (h2 : A_c + A_r = 98) -- Sum of the base areas
variable (h3 : A_c / A_r = 3 / 4) -- Ratio of the base areas

-- Statement: The goal is to prove that the base area of the cylinder is 42.
theorem cylinder_base_area : A_c = 42 :=
by
  sorry

end cylinder_base_area_l186_186471


namespace find_Sum_4n_l186_186522

variable {a : ℕ → ℕ} -- Define a sequence a_n

-- Define our conditions about the sums Sn and S3n
axiom Sum_n : ℕ → ℕ 
axiom Sum_3n : ℕ → ℕ 
axiom Sum_4n : ℕ → ℕ 

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a n + a 0)) / 2

axiom h1 : is_arithmetic_sequence a
axiom h2 : Sum_n 1 = 2
axiom h3 : Sum_3n 3 = 12

theorem find_Sum_4n : Sum_4n 4 = 20 :=
sorry

end find_Sum_4n_l186_186522


namespace mass_percentage_Al_in_AlI3_l186_186048

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

theorem mass_percentage_Al_in_AlI3 : 
  (molar_mass_Al / molar_mass_AlI3) * 100 = 6.62 := 
  sorry

end mass_percentage_Al_in_AlI3_l186_186048


namespace complete_square_transform_l186_186456

theorem complete_square_transform (x : ℝ) :
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := 
sorry

end complete_square_transform_l186_186456


namespace find_the_number_l186_186606

theorem find_the_number (x : ℕ) : (220040 = (x + 445) * (2 * (x - 445)) + 40) → x = 555 :=
by
  intro h
  sorry

end find_the_number_l186_186606


namespace find_n_l186_186494

theorem find_n (n : ℕ) (h : n > 0) :
  (n * (n - 1) * (n - 2)) / (6 * n^3) = 1 / 16 ↔ n = 4 :=
by sorry

end find_n_l186_186494


namespace two_digit_number_is_54_l186_186568

theorem two_digit_number_is_54 
    (n : ℕ) 
    (h1 : 10 ≤ n ∧ n < 100) 
    (h2 : n % 2 = 0) 
    (h3 : ∃ (a b : ℕ), a * b = 20 ∧ 10 * a + b = n) : 
    n = 54 := 
by
  sorry

end two_digit_number_is_54_l186_186568


namespace triangle_side_length_l186_186107

theorem triangle_side_length (P Q R : Type) (cos_Q : ℝ) (PQ QR : ℝ) 
  (sin_Q : ℝ) (h_cos_Q : cos_Q = 0.6) (h_PQ : PQ = 10) (h_sin_Q : sin_Q = 0.8) : 
  QR = 50 / 3 :=
by
  sorry

end triangle_side_length_l186_186107


namespace find_A_max_min_l186_186999

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

end find_A_max_min_l186_186999


namespace least_integer_nk_l186_186379

noncomputable def min_nk (k : ℕ) : ℕ :=
  (5 * k + 1) / 2

theorem least_integer_nk (k : ℕ) (S : Fin 5 → Finset ℕ) :
  (∀ j : Fin 5, (S j).card = k) →
  (∀ i : Fin 4, (S i ∩ S (i + 1)).card = 0) →
  (S 4 ∩ S 0).card = 0 →
  (∃ nk, (∃ (U : Finset ℕ), (∀ j : Fin 5, S j ⊆ U) ∧ U.card = nk) ∧ nk = min_nk k) :=
by
  sorry

end least_integer_nk_l186_186379


namespace train_speed_platform_man_l186_186299

theorem train_speed_platform_man (t_man t_platform : ℕ) (platform_length : ℕ) (v_train_mps : ℝ) (v_train_kmph : ℝ) 
  (h1 : t_man = 18) 
  (h2 : t_platform = 32) 
  (h3 : platform_length = 280)
  (h4 : v_train_mps = (platform_length / (t_platform - t_man)))
  (h5 : v_train_kmph = v_train_mps * 3.6) :
  v_train_kmph = 72 := 
sorry

end train_speed_platform_man_l186_186299


namespace weight_of_B_l186_186332

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by
  sorry

end weight_of_B_l186_186332


namespace assembly_shortest_time_l186_186484

-- Define the times taken for each assembly path
def time_ACD : ℕ := 3 + 4
def time_EDF : ℕ := 4 + 2

-- State the theorem for the shortest time required to assemble the product
theorem assembly_shortest_time : max time_ACD time_EDF + 4 = 13 :=
by {
  -- Introduction of the given conditions and simplified value calculation
  sorry
}

end assembly_shortest_time_l186_186484


namespace john_total_distance_traveled_l186_186742

theorem john_total_distance_traveled :
  let d1 := 45 * 2.5
  let d2 := 60 * 3.5
  let d3 := 40 * 2
  let d4 := 55 * 3
  d1 + d2 + d3 + d4 = 567.5 := by
  sorry

end john_total_distance_traveled_l186_186742


namespace question1_question2_question3_l186_186671

-- Define probabilities of renting and returning bicycles at different stations
def P (X Y : Char) : ℝ :=
  if X = 'A' ∧ Y = 'A' then 0.3 else
  if X = 'A' ∧ Y = 'B' then 0.2 else
  if X = 'A' ∧ Y = 'C' then 0.5 else
  if X = 'B' ∧ Y = 'A' then 0.7 else
  if X = 'B' ∧ Y = 'B' then 0.1 else
  if X = 'B' ∧ Y = 'C' then 0.2 else
  if X = 'C' ∧ Y = 'A' then 0.4 else
  if X = 'C' ∧ Y = 'B' then 0.5 else
  if X = 'C' ∧ Y = 'C' then 0.1 else 0

-- Question 1: Prove P(CC) = 0.1
theorem question1 : P 'C' 'C' = 0.1 := by
  sorry

-- Question 2: Prove P(AC) * P(CB) = 0.25
theorem question2 : P 'A' 'C' * P 'C' 'B' = 0.25 := by
  sorry

-- Question 3: Prove the probability P = 0.43
theorem question3 : P 'A' 'A' * P 'A' 'A' + P 'A' 'B' * P 'B' 'A' + P 'A' 'C' * P 'C' 'A' = 0.43 := by
  sorry

end question1_question2_question3_l186_186671


namespace find_B_value_l186_186986

theorem find_B_value (A C B : ℕ) (h1 : A = 634) (h2 : A = C + 593) (h3 : B = C + 482) : B = 523 :=
by {
  -- Proof would go here
  sorry
}

end find_B_value_l186_186986


namespace jigsaw_puzzle_completion_l186_186793

theorem jigsaw_puzzle_completion (p : ℝ) :
  let total_pieces := 1000
  let pieces_first_day := total_pieces * 0.10
  let remaining_after_first_day := total_pieces - pieces_first_day

  let pieces_second_day := remaining_after_first_day * (p / 100)
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day

  let pieces_third_day := remaining_after_second_day * 0.30
  let remaining_after_third_day := remaining_after_second_day - pieces_third_day

  remaining_after_third_day = 504 ↔ p = 20 := 
by {
    sorry
}

end jigsaw_puzzle_completion_l186_186793


namespace angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l186_186857

-- Problem part (a)
theorem angles_in_arithmetic_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (arithmetic_progression : ∃ (d : ℝ) (α : ℝ), β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0):
  (∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0) :=
sorry

-- Problem part (b)
theorem angles_not_in_geometric_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (geometric_progression : ∃ (r : ℝ) (α : ℝ), β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1 ∧ r > 0):
  ¬(∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1) :=
sorry

end angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l186_186857


namespace watermelon_cost_100_l186_186546

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l186_186546


namespace find_second_candy_cost_l186_186185

theorem find_second_candy_cost :
  ∃ (x : ℝ), 
    (15 * 8 + 30 * x = 45 * 6) ∧
    x = 5 := by
  sorry

end find_second_candy_cost_l186_186185


namespace fermats_little_theorem_l186_186966

theorem fermats_little_theorem (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : (a^p - a) % p = 0 := 
by sorry

end fermats_little_theorem_l186_186966


namespace lines_perpendicular_l186_186592

noncomputable def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem lines_perpendicular {m : ℝ} :
  is_perpendicular (m + 2) (1 - m) (m - 1) (2 * m + 3) ↔ m = 1 :=
by
  sorry

end lines_perpendicular_l186_186592


namespace min_sum_a1_a2_l186_186050

-- Define the condition predicate for the sequence
def satisfies_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2009) / (1 + a (n + 1))

-- State the main problem as a theorem in Lean 4
theorem min_sum_a1_a2 (a : ℕ → ℕ) (h_seq : satisfies_seq a) (h_pos : ∀ n, a n > 0) :
  a 1 * a 2 = 2009 → a 1 + a 2 = 90 :=
sorry

end min_sum_a1_a2_l186_186050


namespace find_pairs_l186_186853

theorem find_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔ ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) :=
by sorry

end find_pairs_l186_186853


namespace mary_needs_6_cups_of_flour_l186_186190

-- Define the necessary constants according to the conditions.
def flour_needed : ℕ := 6
def sugar_needed : ℕ := 13
def flour_more_than_sugar : ℕ := 8

-- Define the number of cups of flour Mary needs to add.
def flour_to_add (flour_put_in : ℕ) : ℕ := flour_needed - flour_put_in

-- Prove that Mary needs to add 6 more cups of flour.
theorem mary_needs_6_cups_of_flour (flour_put_in : ℕ) (h : flour_more_than_sugar = 8): flour_to_add flour_put_in = 6 :=
by {
  sorry -- the proof is omitted.
}

end mary_needs_6_cups_of_flour_l186_186190


namespace f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l186_186778

open Real

noncomputable def f : ℝ → ℝ :=
sorry

axiom func_prop : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom pos_x_gt_1 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_1 : f 1 = 2

-- Prove that f(0) = 1
theorem f_0_eq_1 : f 0 = 1 :=
sorry

-- Prove that f(-1) ≠ 1 (and direct derivation showing f(-1) = 0)
theorem f_neg_1_ne_1 : f (-1) ≠ 1 ∧ f (-1) = 0 :=
sorry

-- Prove that f(x) is increasing
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ > f x₁ :=
sorry

-- Prove minimum value of f on [-3, 3] is -2
theorem min_f_neg3_3 : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -2 :=
sorry

end f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l186_186778


namespace what_to_do_first_l186_186366

-- Definition of the conditions
def eat_or_sleep_to_survive (days_without_eat : ℕ) (days_without_sleep : ℕ) : Prop :=
  (days_without_eat = 7 → days_without_sleep ≠ 7) ∨ (days_without_sleep = 7 → days_without_eat ≠ 7)

-- Theorem statement based on the problem and its conditions
theorem what_to_do_first (days_without_eat days_without_sleep : ℕ) :
  days_without_eat = 7 ∨ days_without_sleep = 7 →
  eat_or_sleep_to_survive days_without_eat days_without_sleep :=
by sorry

end what_to_do_first_l186_186366


namespace domain_of_function_l186_186213

def domain_of_f (x: ℝ) : Prop :=
x >= -1 ∧ x <= 48

theorem domain_of_function :
  ∀ x, (x + 1 >= 0 ∧ 7 - Real.sqrt (x + 1) >= 0 ∧ 4 - Real.sqrt (7 - Real.sqrt (x + 1)) >= 0)
  ↔ domain_of_f x := by
  sorry

end domain_of_function_l186_186213


namespace sum_of_digits_base2_315_l186_186425

theorem sum_of_digits_base2_315 :
  let b2_expr := 100111011 -- base-2 representation of 315
  let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1) -- sum of its digits
  digit_sum = 6 := by 
    let b2_expr := 100111011
    let digit_sum := (1 + 0 + 0 + 1 + 1 + 1 + 0 + 1 + 1)
    sorry

end sum_of_digits_base2_315_l186_186425


namespace angela_january_additional_sleep_l186_186911

-- Definitions corresponding to conditions in part a)
def december_sleep_hours : ℝ := 6.5
def january_sleep_hours : ℝ := 8.5
def days_in_january : ℕ := 31

-- The proof statement, proving the January's additional sleep hours
theorem angela_january_additional_sleep :
  (january_sleep_hours - december_sleep_hours) * days_in_january = 62 :=
by
  -- Since the focus is only on the statement, we skip the actual proof.
  sorry

end angela_january_additional_sleep_l186_186911


namespace cone_generatrix_length_is_2sqrt2_l186_186383

noncomputable def cone_generatrix_length (r : ℝ) : ℝ :=
  let C := 2 * Real.pi * r
  let l := (2 * Real.pi * r) / Real.pi
  l

theorem cone_generatrix_length_is_2sqrt2 :
  cone_generatrix_length (Real.sqrt 2) = 2 * Real.sqrt 2 :=
by
  sorry

end cone_generatrix_length_is_2sqrt2_l186_186383


namespace find_k_l186_186634

theorem find_k (x y k : ℝ) (h1 : x = 1) (h2 : y = 4) (h3 : k * x + y = 3) : k = -1 :=
by
  sorry

end find_k_l186_186634


namespace opposite_of_neg_five_l186_186668

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l186_186668


namespace waiter_earning_correct_l186_186182

-- Definitions based on the conditions
def tip1 : ℝ := 25 * 0.15
def tip2 : ℝ := 22 * 0.18
def tip3 : ℝ := 35 * 0.20
def tip4 : ℝ := 30 * 0.10

def total_tips : ℝ := tip1 + tip2 + tip3 + tip4
def commission : ℝ := total_tips * 0.05
def net_tips : ℝ := total_tips - commission

-- Theorem statement
theorem waiter_earning_correct : net_tips = 16.82 := by
  sorry

end waiter_earning_correct_l186_186182


namespace bottles_of_regular_soda_l186_186296

theorem bottles_of_regular_soda (R : ℕ) : 
  let apples := 36 
  let diet_soda := 54
  let total_bottles := apples + 98 
  R + diet_soda = total_bottles → R = 80 :=
by
  sorry

end bottles_of_regular_soda_l186_186296


namespace no_two_digit_prime_with_digit_sum_9_l186_186976

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

end no_two_digit_prime_with_digit_sum_9_l186_186976


namespace domain_of_f_l186_186781

-- Define the conditions for the function
def condition1 (x : ℝ) : Prop := 1 - x > 0
def condition2 (x : ℝ) : Prop := 3 * x + 1 > 0

-- Define the domain interval
def domain (x : ℝ) : Prop := -1 / 3 < x ∧ x < 1

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 / (Real.sqrt (1 - x)) + Real.log (3 * x + 1)

-- The main theorem to prove
theorem domain_of_f : 
  (∀ x : ℝ, condition1 x ∧ condition2 x ↔ domain x) :=
by {
  sorry
}

end domain_of_f_l186_186781


namespace percentage_increase_l186_186467

-- Defining the problem constants
def price (P : ℝ) : ℝ := P
def assets_A (A : ℝ) : ℝ := A
def assets_B (B : ℝ) : ℝ := B
def percentage (X : ℝ) : ℝ := X

-- Conditions
axiom price_company_B_double_assets : ∀ (P B: ℝ), price P = 2 * assets_B B
axiom price_seventy_five_percent_combined_assets : ∀ (P A B: ℝ), price P = 0.75 * (assets_A A + assets_B B)
axiom price_percentage_more_than_A : ∀ (P A X: ℝ), price P = assets_A A * (1 + percentage X / 100)

-- Theorem to prove
theorem percentage_increase : ∀ (P A B X : ℝ)
  (h1 : price P = 2 * assets_B B)
  (h2 : price P = 0.75 * (assets_A A + assets_B B))
  (h3 : price P = assets_A A * (1 + percentage X / 100)),
  percentage X = 20 :=
by
  intros P A B X h1 h2 h3
  -- Proof steps would go here
  sorry

end percentage_increase_l186_186467


namespace total_increase_percentage_l186_186865

-- Define the conditions: original speed S, first increase by 30%, then another increase by 10%
def original_speed (S : ℝ) := S
def first_increase (S : ℝ) := S * 1.30
def second_increase (S : ℝ) := (S * 1.30) * 1.10

-- Prove that the total increase in speed is 43% of the original speed
theorem total_increase_percentage (S : ℝ) :
  (second_increase S - original_speed S) / original_speed S * 100 = 43 :=
by
  sorry

end total_increase_percentage_l186_186865


namespace find_surface_area_of_ball_l186_186223

noncomputable def surface_area_of_ball : ℝ :=
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area

theorem find_surface_area_of_ball :
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  surface_area = (2 / 3) * Real.pi :=
by
  let tetrahedron_edge := 4
  let tetrahedron_volume := (1 / 3) * (Real.sqrt 3 / 4) * tetrahedron_edge ^ 2 * (Real.sqrt 16 - 16 / 3)
  let water_volume := (7 / 8) * tetrahedron_volume
  let remaining_volume := (1 / 8) * tetrahedron_volume
  let remaining_edge := 2
  let ball_radius := Real.sqrt 6 / 6
  let surface_area := 4 * Real.pi * ball_radius ^ 2
  sorry

end find_surface_area_of_ball_l186_186223


namespace circle_count_2012_l186_186018

/-
The pattern is defined as follows: 
○●, ○○●, ○○○●, ○○○○●, …
We need to prove that the number of ● in the first 2012 circles is 61.
-/

-- Define the pattern sequence
def circlePattern (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Total number of circles in the first k segments:
def totalCircles (k : ℕ) : ℕ :=
  k * (k + 1) / 2 + k

theorem circle_count_2012 : 
  ∃ (n : ℕ), totalCircles n ≤ 2012 ∧ 2012 < totalCircles (n + 1) ∧ n = 61 :=
by
  sorry

end circle_count_2012_l186_186018


namespace geometric_progression_theorem_l186_186137

variables {a b c : ℝ} {n : ℕ} {q : ℝ}

-- Define the terms in the geometric progression
def nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^n
def second_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(2 * n)
def fourth_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(4 * n)

-- Conditions
axiom nth_term_def : b = nth_term a q n
axiom second_nth_term_def : b = second_nth_term a q n
axiom fourth_nth_term_def : c = fourth_nth_term a q n

-- Statement to prove
theorem geometric_progression_theorem :
  b * (b^2 - a^2) = a^2 * (c - b) :=
sorry

end geometric_progression_theorem_l186_186137


namespace area_D_meets_sign_l186_186450

-- Definition of conditions as given in the question
def condition_A (mean median : ℝ) : Prop := mean = 3 ∧ median = 4
def condition_B (mean : ℝ) (variance_pos : Prop) : Prop := mean = 1 ∧ variance_pos
def condition_C (median mode : ℝ) : Prop := median = 2 ∧ mode = 3
def condition_D (mean variance : ℝ) : Prop := mean = 2 ∧ variance = 3

-- Theorem stating that Area D satisfies the condition to meet the required sign
theorem area_D_meets_sign (mean variance : ℝ) (h : condition_D mean variance) : 
  (∀ day_increase, day_increase ≤ 7) :=
sorry

end area_D_meets_sign_l186_186450


namespace dish_heats_up_by_5_degrees_per_minute_l186_186075

theorem dish_heats_up_by_5_degrees_per_minute
  (final_temperature initial_temperature : ℕ)
  (time_taken : ℕ)
  (h1 : final_temperature = 100)
  (h2 : initial_temperature = 20)
  (h3 : time_taken = 16) :
  (final_temperature - initial_temperature) / time_taken = 5 :=
by
  sorry

end dish_heats_up_by_5_degrees_per_minute_l186_186075


namespace teddy_bear_cost_l186_186399

-- Definitions for the given conditions
def num_toys : ℕ := 28
def toy_price : ℕ := 10
def num_teddy_bears : ℕ := 20
def total_money : ℕ := 580

-- The theorem we want to prove
theorem teddy_bear_cost :
  (num_teddy_bears * 15 + num_toys * toy_price = total_money) :=
by
  sorry

end teddy_bear_cost_l186_186399


namespace correct_quotient_division_l186_186687

variable (k : Nat) -- the unknown original number

def mistaken_division := k = 7 * 12 + 4

theorem correct_quotient_division (h : mistaken_division k) : 
  (k / 3) = 29 :=
by
  sorry

end correct_quotient_division_l186_186687


namespace positive_difference_l186_186478

theorem positive_difference (a b : ℕ) (h1 : a = (6^2 + 6^2) / 6) (h2 : b = (6^2 * 6^2) / 6) : a < b ∧ b - a = 204 :=
by
  sorry

end positive_difference_l186_186478


namespace range_of_m_l186_186224

variable {x m : ℝ}

def condition_p (x : ℝ) : Prop := |x - 3| ≤ 2
def condition_q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m (m : ℝ) :
  (∀ x, ¬(condition_p x) → ¬(condition_q x m)) ∧ ¬(∀ x, ¬(condition_q x m) → ¬(condition_p x)) →
  2 < m ∧ m < 4 := 
sorry

end range_of_m_l186_186224


namespace rhombus_diagonals_sum_squares_l186_186598

-- Definition of the rhombus side length condition
def is_rhombus_side_length (side_length : ℝ) : Prop :=
  side_length = 2

-- Lean 4 statement for the proof problem
theorem rhombus_diagonals_sum_squares (side_length : ℝ) (d1 d2 : ℝ) 
  (h : is_rhombus_side_length side_length) :
  side_length = 2 → (d1^2 + d2^2 = 16) :=
by
  sorry

end rhombus_diagonals_sum_squares_l186_186598


namespace range_of_m_l186_186022

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x - m)/2 ≥ 2 ∧ x - 4 ≤ 3 * (x - 2)) →
  ∃ x : ℝ, x = 2 ∧ -3 < m ∧ m ≤ -2 :=
by
  sorry

end range_of_m_l186_186022


namespace solution_set_inequality_l186_186733

theorem solution_set_inequality (x : ℝ) : (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) := 
sorry

end solution_set_inequality_l186_186733


namespace person_age_l186_186495

theorem person_age (A : ℕ) (h : 6 * (A + 6) - 6 * (A - 6) = A) : A = 72 := 
by
  sorry

end person_age_l186_186495


namespace total_cookies_prepared_l186_186384

-- State the conditions as definitions
def num_guests : ℕ := 10
def cookies_per_guest : ℕ := 18

-- The theorem stating the problem
theorem total_cookies_prepared (num_guests cookies_per_guest : ℕ) : 
  num_guests * cookies_per_guest = 180 := 
by 
  -- Here, we would have the proof, but we're using sorry to skip it
  sorry

end total_cookies_prepared_l186_186384


namespace bananas_to_mush_l186_186483

theorem bananas_to_mush (x : ℕ) (h1 : 3 * (20 / x) = 15) : x = 4 :=
by
  sorry

end bananas_to_mush_l186_186483


namespace hard_candy_food_colouring_l186_186990

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

end hard_candy_food_colouring_l186_186990


namespace probability_of_sum_at_least_10_l186_186374

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 6

theorem probability_of_sum_at_least_10 :
  (favorable_outcomes : ℝ) / (total_outcomes : ℝ) = 1 / 6 := by
  sorry

end probability_of_sum_at_least_10_l186_186374


namespace table_height_l186_186858

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

end table_height_l186_186858


namespace Malik_yards_per_game_l186_186721

-- Definitions of the conditions
def number_of_games : ℕ := 4
def josiah_yards_per_game : ℕ := 22
def darnell_average_yards_per_game : ℕ := 11
def total_yards_all_athletes : ℕ := 204

-- The statement to prove
theorem Malik_yards_per_game (M : ℕ) 
  (H1 : number_of_games = 4) 
  (H2 : josiah_yards_per_game = 22) 
  (H3 : darnell_average_yards_per_game = 11) 
  (H4 : total_yards_all_athletes = 204) :
  4 * M + 4 * 22 + 4 * 11 = 204 → M = 18 :=
by
  intros h
  sorry

end Malik_yards_per_game_l186_186721


namespace carmina_coins_l186_186956

-- Define the conditions related to the problem
variables (n d : ℕ) -- number of nickels and dimes

theorem carmina_coins (h1 : 5 * n + 10 * d = 360) (h2 : 10 * n + 5 * d = 540) : n + d = 60 :=
sorry

end carmina_coins_l186_186956


namespace find_a9_l186_186363

variable (a : ℕ → ℤ)  -- Arithmetic sequence
variable (S : ℕ → ℤ)  -- Sum of the first n terms

-- Conditions provided in the problem
axiom Sum_condition : S 8 = 4 * a 3
axiom Term_condition : a 7 = -2
axiom Sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2

-- Hypothesis for common difference
def common_diff (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Proving that a_9 equals -6 given the conditions
theorem find_a9 (d : ℤ) : common_diff a d → a 9 = -6 :=
by
  intros h
  sorry

end find_a9_l186_186363


namespace arithmetic_mean_is_five_sixths_l186_186880

theorem arithmetic_mean_is_five_sixths :
  let a := 3 / 4
  let b := 5 / 6
  let c := 7 / 8
  (a + c) / 2 = b := sorry

end arithmetic_mean_is_five_sixths_l186_186880


namespace tan_neg_210_eq_neg_sqrt_3_div_3_l186_186376

theorem tan_neg_210_eq_neg_sqrt_3_div_3 : Real.tan (-210 * Real.pi / 180) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_neg_210_eq_neg_sqrt_3_div_3_l186_186376


namespace proportion_solution_l186_186646

theorem proportion_solution (x : ℝ) (h : x / 6 = 4 / 0.39999999999999997) : x = 60 := sorry

end proportion_solution_l186_186646


namespace solve_for_y_l186_186313

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l186_186313


namespace number_of_cars_parked_l186_186233

-- Definitions for the given conditions
def total_area (length width : ℕ) : ℕ := length * width
def usable_area (total : ℕ) : ℕ := (8 * total) / 10
def cars_parked (usable : ℕ) (area_per_car : ℕ) : ℕ := usable / area_per_car

-- Given conditions
def length : ℕ := 400
def width : ℕ := 500
def area_per_car : ℕ := 10
def expected_cars : ℕ := 16000 -- correct answer from solution

-- Define a proof statement
theorem number_of_cars_parked : cars_parked (usable_area (total_area length width)) area_per_car = expected_cars := by
  sorry

end number_of_cars_parked_l186_186233


namespace union_eq_l186_186557

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

theorem union_eq : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_eq_l186_186557


namespace option_D_is_correct_l186_186834

variable (a b : ℝ)

theorem option_D_is_correct :
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^2 + 3 * a ≠ 4 * a^2) ∧
  ((a + 2) * (a - 2) ≠ a^2 - 2) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end option_D_is_correct_l186_186834


namespace percentage_dried_fruit_of_combined_mix_l186_186434

theorem percentage_dried_fruit_of_combined_mix :
  ∀ (weight_sue weight_jane : ℝ),
  (weight_sue * 0.3 + weight_jane * 0.6) / (weight_sue + weight_jane) = 0.45 →
  100 * (weight_sue * 0.7) / (weight_sue + weight_jane) = 35 :=
by
  intros weight_sue weight_jane H
  sorry

end percentage_dried_fruit_of_combined_mix_l186_186434


namespace fixed_point_exists_l186_186885

noncomputable def fixed_point : Prop := ∀ d : ℝ, ∃ (p q : ℝ), (p = -3) ∧ (q = 45) ∧ (q = 5 * p^2 + d * p + 3 * d)

theorem fixed_point_exists : fixed_point :=
by
  sorry

end fixed_point_exists_l186_186885


namespace total_gold_cost_l186_186693

-- Given conditions
def gary_grams : ℕ := 30
def gary_cost_per_gram : ℕ := 15
def anna_grams : ℕ := 50
def anna_cost_per_gram : ℕ := 20

-- Theorem statement to prove
theorem total_gold_cost :
  (gary_grams * gary_cost_per_gram + anna_grams * anna_cost_per_gram) = 1450 := 
by
  sorry

end total_gold_cost_l186_186693


namespace rain_probability_tel_aviv_l186_186821

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l186_186821


namespace golu_distance_travelled_l186_186130

theorem golu_distance_travelled 
  (b : ℝ) (c : ℝ) (h : c^2 = x^2 + b^2) : x = 8 := by
  sorry

end golu_distance_travelled_l186_186130


namespace chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l186_186774

-- Part a: Prove that with 40 chips, exactly one chip cannot remain after both players have made two moves.
theorem chips_removal_even_initial_40 
  (initial_chips : Nat)
  (num_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 40 → 
  num_moves = 4 → 
  remaining_chips = 1 → 
  False :=
by
  sorry

-- Part b: Prove that with 1000 chips, the minimum number of moves to reduce to one chip is 8.
theorem chips_removal_minimum_moves_1000
  (initial_chips : Nat)
  (min_moves : Nat)
  (remaining_chips : Nat) :
  initial_chips = 1000 → 
  remaining_chips = 1 → 
  min_moves = 8 :=
by
  sorry

end chips_removal_even_initial_40_chips_removal_minimum_moves_1000_l186_186774


namespace prime_n_if_power_of_prime_l186_186772

theorem prime_n_if_power_of_prime (n : ℕ) (h1 : n ≥ 2) (b : ℕ) (h2 : b > 0) (p : ℕ) (k : ℕ) 
  (hk : k > 0) (hb : (b^n - 1) / (b - 1) = p^k) : Nat.Prime n :=
sorry

end prime_n_if_power_of_prime_l186_186772


namespace original_side_length_l186_186217

theorem original_side_length (x : ℝ) (h1 : (x - 6) * (x - 5) = 120) : x = 15 :=
sorry

end original_side_length_l186_186217


namespace neither_sufficient_nor_necessary_l186_186565

-- For given real numbers x and y
-- Prove the statement "at least one of x and y is greater than 1" is not necessary and not sufficient for x^2 + y^2 > 2.
noncomputable def at_least_one_gt_one (x y : ℝ) : Prop := (x > 1) ∨ (y > 1)
def sum_of_squares_gt_two (x y : ℝ) : Prop := x^2 + y^2 > 2

theorem neither_sufficient_nor_necessary (x y : ℝ) :
  ¬(at_least_one_gt_one x y → sum_of_squares_gt_two x y) ∧ ¬(sum_of_squares_gt_two x y → at_least_one_gt_one x y) :=
by
  sorry

end neither_sufficient_nor_necessary_l186_186565


namespace fuel_consumption_l186_186616

-- Define the initial conditions based on the problem
variable (s Q : ℝ)

-- Distance and fuel data points
def data_points : List (ℝ × ℝ) := [(0, 50), (100, 42), (200, 34), (300, 26), (400, 18)]

-- Define the function Q and required conditions
theorem fuel_consumption :
  (∀ p ∈ data_points, ∃ k b, Q = k * s + b ∧
    ((p.1 = 0 → b = 50) ∧
     (p.1 = 100 → Q = 42 → k = -0.08))) :=
by
  sorry

end fuel_consumption_l186_186616


namespace does_not_determine_shape_l186_186391

-- Definition of a function that checks whether given data determine the shape of a triangle
def determines_shape (data : Type) : Prop := sorry

-- Various conditions about data
def ratio_two_angles_included_side : Type := sorry
def ratios_three_angle_bisectors : Type := sorry
def ratios_three_side_lengths : Type := sorry
def ratio_angle_bisector_opposite_side : Type := sorry
def three_angles : Type := sorry

-- The main theorem stating that the ratio of an angle bisector to its corresponding opposite side does not uniquely determine the shape of a triangle.
theorem does_not_determine_shape :
  ¬determines_shape ratio_angle_bisector_opposite_side := sorry

end does_not_determine_shape_l186_186391


namespace find_c_l186_186163

-- Define the polynomial P(x)
def P (c : ℚ) (x : ℚ) : ℚ := x^3 + 4 * x^2 + c * x + 20

-- Given that x - 3 is a factor of P(x), prove that c = -83/3
theorem find_c (c : ℚ) (h : P c 3 = 0) : c = -83 / 3 :=
by
  sorry

end find_c_l186_186163


namespace solution_l186_186059

def problem_statement : Prop :=
  (3025 - 2880) ^ 2 / 225 = 93

theorem solution : problem_statement :=
by {
  sorry
}

end solution_l186_186059


namespace area_of_one_trapezoid_l186_186996

theorem area_of_one_trapezoid (outer_area inner_area : ℝ) (num_trapezoids : ℕ) (h_outer : outer_area = 36) (h_inner : inner_area = 4) (h_num_trapezoids : num_trapezoids = 3) : (outer_area - inner_area) / num_trapezoids = 32 / 3 :=
by
  rw [h_outer, h_inner, h_num_trapezoids]
  norm_num

end area_of_one_trapezoid_l186_186996


namespace determine_b_from_inequality_l186_186066

theorem determine_b_from_inequality (b : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - b * x + 6 < 0) → b = 5 :=
by
  intro h
  -- Proof can be added here
  sorry

end determine_b_from_inequality_l186_186066


namespace remainder_of_expression_l186_186165

theorem remainder_of_expression :
  let a := 2^206 + 206
  let b := 2^103 + 2^53 + 1
  a % b = 205 := 
sorry

end remainder_of_expression_l186_186165


namespace original_number_of_people_l186_186124

/-- Initially, one-third of the people in a room left.
Then, one-fourth of those remaining started to dance.
There were then 18 people who were not dancing.
What was the original number of people in the room? -/
theorem original_number_of_people (x : ℕ) 
  (h_one_third_left : ∀ y : ℕ, 2 * y / 3 = x) 
  (h_one_fourth_dancing : ∀ y : ℕ, y / 4 = x) 
  (h_non_dancers : x / 2 = 18) : 
  x = 36 :=
sorry

end original_number_of_people_l186_186124


namespace product_divisible_by_49_l186_186131

theorem product_divisible_by_49 (a b : ℕ) (h : (a^2 + b^2) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end product_divisible_by_49_l186_186131


namespace color_circles_with_four_colors_l186_186178

theorem color_circles_with_four_colors (n : ℕ) (circles : Fin n → (ℝ × ℝ)) (radius : ℝ):
  (∀ i j, i ≠ j → dist (circles i) (circles j) ≥ 2 * radius) →
  ∃ f : Fin n → Fin 4, ∀ i j, dist (circles i) (circles j) < 2 * radius → f i ≠ f j :=
by
  sorry

end color_circles_with_four_colors_l186_186178


namespace directrix_of_parabola_l186_186755

theorem directrix_of_parabola (a : ℝ) (P : ℝ × ℝ)
  (h1 : 3 * P.1 ^ 2 - P.2 ^ 2 = 3 * a ^ 2)
  (h2 : P.2 ^ 2 = 8 * a * P.1)
  (h3 : a > 0)
  (h4 : abs ((P.1 - 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) + abs ((P.1 + 2 * a) ^ 2 + P.2 ^ 2) ^ (1 / 2) = 12) :
  (a = 1) → P.1 = 6 - 3 * a → P.2 ^ 2 = 8 * a * (6 - 3 * a) → -2 * a = -2 := 
by
  sorry

end directrix_of_parabola_l186_186755


namespace even_function_implies_f2_eq_neg5_l186_186301

def f (x a : ℝ) : ℝ := (x - a) * (x + 3)

theorem even_function_implies_f2_eq_neg5 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) :
  f 2 a = -5 :=
by
  sorry

end even_function_implies_f2_eq_neg5_l186_186301


namespace transformation_1_transformation_2_l186_186325

theorem transformation_1 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq1 : 5 * x + 2 * y = 0) : 
  5 * x' + 3 * y' = 0 := 
sorry

theorem transformation_2 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq2 : x^2 + y^2 = 1) : 
  4 * x' ^ 2 + 9 * y' ^ 2 = 1 := 
sorry

end transformation_1_transformation_2_l186_186325


namespace inverse_B_squared_l186_186944

-- Defining the inverse matrix B_inv
def B_inv : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 0, 1]

-- Theorem to prove that the inverse of B^2 is a specific matrix
theorem inverse_B_squared :
  (B_inv * B_inv) = !![9, -6; 0, 1] :=
  by sorry


end inverse_B_squared_l186_186944


namespace evaluate_fraction_l186_186509

theorem evaluate_fraction : (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = (8 / 21) :=
by
  sorry

end evaluate_fraction_l186_186509


namespace initial_innings_count_l186_186179

theorem initial_innings_count (n T L : ℕ) 
  (h1 : T = 50 * n)
  (h2 : 174 = L + 172)
  (h3 : (T - 174 - L) = 48 * (n - 2)) :
  n = 40 :=
by 
  sorry

end initial_innings_count_l186_186179


namespace inverse_function_correct_l186_186115

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) ^ 2 + 1

noncomputable def f_inv (y : ℝ) : ℝ :=
  1 - Real.sqrt (y - 1)

theorem inverse_function_correct (x : ℝ) (hx : x ≥ 2) :
  f_inv x = 1 - Real.sqrt (x - 1) ∧ ∀ y : ℝ, (y ≤ 0) → f y = x → y = f_inv x :=
by {
  sorry
}

end inverse_function_correct_l186_186115


namespace nearest_integer_to_x_plus_2y_l186_186095

theorem nearest_integer_to_x_plus_2y
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : |x| + 2 * y = 6)
  (h2 : |x| * y + x^3 = 2) :
  Int.floor (x + 2 * y + 0.5) = 6 :=
by sorry

end nearest_integer_to_x_plus_2y_l186_186095


namespace one_cow_one_bag_in_39_days_l186_186642

-- Definitions
def cows : ℕ := 52
def husks : ℕ := 104
def days : ℕ := 78

-- Problem: Given that 52 cows eat 104 bags of husk in 78 days,
-- Prove that one cow will eat one bag of husk in 39 days.
theorem one_cow_one_bag_in_39_days (cows_cons : cows = 52) (husks_cons : husks = 104) (days_cons : days = 78) :
  ∃ d : ℕ, d = 39 :=
by
  -- Placeholder for the proof.
  sorry

end one_cow_one_bag_in_39_days_l186_186642


namespace rectangle_distances_sum_l186_186323

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem rectangle_distances_sum :
  let A : (ℝ × ℝ) := (0, 0)
  let B : (ℝ × ℝ) := (3, 0)
  let C : (ℝ × ℝ) := (3, 4)
  let D : (ℝ × ℝ) := (0, 4)

  let M : (ℝ × ℝ) := ((B.1 + A.1) / 2, (B.2 + A.2) / 2)
  let N : (ℝ × ℝ) := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let O : (ℝ × ℝ) := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let P : (ℝ × ℝ) := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)

  distance A.1 A.2 M.1 M.2 + distance A.1 A.2 N.1 N.2 + distance A.1 A.2 O.1 O.2 + distance A.1 A.2 P.1 P.2 = 7.77 + Real.sqrt 13 :=
sorry

end rectangle_distances_sum_l186_186323


namespace horse_buying_problem_l186_186134

variable (x y z : ℚ)

theorem horse_buying_problem :
  (x + 1/2 * y + 1/2 * z = 12) →
  (y + 1/3 * x + 1/3 * z = 12) →
  (z + 1/4 * x + 1/4 * y = 12) →
  x = 60/17 ∧ y = 136/17 ∧ z = 156/17 :=
by
  sorry

end horse_buying_problem_l186_186134


namespace james_marbles_left_l186_186256

theorem james_marbles_left (initial_marbles : ℕ) (total_bags : ℕ) (marbles_per_bag : ℕ) (bags_given_away : ℕ) : 
  initial_marbles = 28 → total_bags = 4 → marbles_per_bag = initial_marbles / total_bags → bags_given_away = 1 → 
  initial_marbles - marbles_per_bag * bags_given_away = 21 :=
by
  intros h_initial h_total h_each h_given
  sorry

end james_marbles_left_l186_186256


namespace regular_polygon_sides_l186_186125

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end regular_polygon_sides_l186_186125


namespace find_C_value_l186_186423

theorem find_C_value (A B C : ℕ) 
  (cond1 : A + B + C = 10) 
  (cond2 : B + A = 9)
  (cond3 : A + 1 = 3) :
  C = 1 :=
by
  sorry

end find_C_value_l186_186423


namespace expand_and_simplify_product_l186_186395

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := 4 * x^4 + 7 * x^2 + 16

theorem expand_and_simplify_product (x : ℝ) : initial_expr x = simplified_expr x := by
  -- We would provide the proof steps here
  sorry

end expand_and_simplify_product_l186_186395


namespace a_formula_b_formula_T_formula_l186_186583

variable {n : ℕ}

def S (n : ℕ) := 2 * n^2

def a (n : ℕ) : ℕ := 
  if n = 1 then S 1 else S n - S (n - 1)

def b (n : ℕ) : ℕ := 
  if n = 1 then 2 else 2 * (1 / 4 ^ (n - 1))

def c (n : ℕ) : ℕ := (4 * n - 2) / (2 * 4 ^ (n - 1))

def T (n : ℕ) : ℕ := 
  (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5)

theorem a_formula :
  ∀ n, a n = 4 * n - 2 := 
sorry

theorem b_formula :
  ∀ n, b n = 2 / (4 ^ (n - 1)) :=
sorry

theorem T_formula :
  ∀ n, T n = (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5) :=
sorry

end a_formula_b_formula_T_formula_l186_186583


namespace bill_buys_125_bouquets_to_make_1000_l186_186763

-- Defining the conditions
def cost_per_bouquet : ℕ := 20
def roses_per_bouquet_buy : ℕ := 7
def roses_per_bouquet_sell : ℕ := 5
def target_difference : ℕ := 1000

-- To be demonstrated: number of bouquets Bill needs to buy to get a profit difference of $1000
theorem bill_buys_125_bouquets_to_make_1000 : 
  let total_cost_to_buy (n : ℕ) := n * cost_per_bouquet
  let total_roses (n : ℕ) := n * roses_per_bouquet_buy
  let bouquets_sell_from_roses (roses : ℕ) := roses / roses_per_bouquet_sell
  let total_revenue (bouquets : ℕ) := bouquets * cost_per_bouquet
  let profit (n : ℕ) := total_revenue (bouquets_sell_from_roses (total_roses n)) - total_cost_to_buy n
  profit (125 * 5) = target_difference := 
sorry

end bill_buys_125_bouquets_to_make_1000_l186_186763


namespace paul_runs_41_miles_l186_186491

-- Conditions as Definitions
def movie1_length : ℕ := (1 * 60) + 36
def movie2_length : ℕ := (2 * 60) + 18
def movie3_length : ℕ := (1 * 60) + 48
def movie4_length : ℕ := (2 * 60) + 30
def total_watch_time : ℕ := movie1_length + movie2_length + movie3_length + movie4_length
def time_per_mile : ℕ := 12

-- Proof Statement
theorem paul_runs_41_miles : total_watch_time / time_per_mile = 41 :=
by
  -- Proof would be provided here
  sorry 

end paul_runs_41_miles_l186_186491


namespace quadratic_eq_transformed_l186_186394

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 2 * x - 7 = 0

-- Define the form to transform to using completing the square method
def transformed_eq (x : ℝ) : Prop := (x - 1)^2 = 8

-- The theorem to be proved
theorem quadratic_eq_transformed (x : ℝ) :
  quadratic_eq x → transformed_eq x :=
by
  intros h
  -- here we would use steps of completing the square to transform the equation
  sorry

end quadratic_eq_transformed_l186_186394


namespace commercials_per_hour_l186_186029

theorem commercials_per_hour (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : ∃ x : ℝ, x = (1 - p) * 60 := 
sorry

end commercials_per_hour_l186_186029


namespace solve_absolute_value_equation_l186_186933

theorem solve_absolute_value_equation (x : ℝ) :
  |2 * x - 3| = x + 1 → (x = 4 ∨ x = 2 / 3) := by
  sorry

end solve_absolute_value_equation_l186_186933


namespace dogs_with_no_accessories_l186_186981

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

end dogs_with_no_accessories_l186_186981


namespace number_of_even_ones_matrices_l186_186172

noncomputable def count_even_ones_matrices (m n : ℕ) : ℕ :=
if m = 0 ∨ n = 0 then 1 else 2^((m-1)*(n-1))

theorem number_of_even_ones_matrices (m n : ℕ) : 
  count_even_ones_matrices m n = 2^((m-1)*(n-1)) := sorry

end number_of_even_ones_matrices_l186_186172


namespace product_expansion_l186_186739

theorem product_expansion (x : ℝ) : 2 * (x + 3) * (x + 4) = 2 * x^2 + 14 * x + 24 := 
by
  sorry

end product_expansion_l186_186739


namespace arithmetic_sequence_problem_l186_186082

variable (a : ℕ → ℤ) -- defining the sequence {a_n}
variable (S : ℕ → ℤ) -- defining the sum of the first n terms S_n

theorem arithmetic_sequence_problem (m : ℕ) (h1 : m > 1) 
  (h2 : a (m - 1) + a (m + 1) - a m ^ 2 = 0) 
  (h3 : S (2 * m - 1) = 38) 
  : m = 10 :=
sorry

end arithmetic_sequence_problem_l186_186082


namespace inverse_h_l186_186632

-- Define the functions f, g, and h as given in the conditions
def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := 3 * x + 2
def h (x : ℝ) := f (g x)

-- State the problem of proving the inverse of h
theorem inverse_h : ∀ x, h⁻¹ (x : ℝ) = (x - 5) / 12 :=
sorry

end inverse_h_l186_186632


namespace Anne_height_l186_186469

-- Define the conditions
variables (S : ℝ)   -- Height of Anne's sister
variables (A : ℝ)   -- Height of Anne
variables (B : ℝ)   -- Height of Bella

-- Define the relations according to the problem's conditions
def condition1 (S : ℝ) := A = 2 * S
def condition2 (S : ℝ) := B = 3 * A
def condition3 (S : ℝ) := B - S = 200

-- Theorem statement to prove Anne's height
theorem Anne_height (S : ℝ) (A : ℝ) (B : ℝ)
(h1 : A = 2 * S) (h2 : B = 3 * A) (h3 : B - S = 200) : A = 80 :=
by sorry

end Anne_height_l186_186469


namespace intersection_A_B_l186_186214

def A : Set ℝ := { x | 2 * x^2 - 5 * x < 0 }
def B : Set ℝ := { x | 3^(x - 1) ≥ Real.sqrt 3 }

theorem intersection_A_B : A ∩ B = Set.Ico (3 / 2) (5 / 2) := 
by
  sorry

end intersection_A_B_l186_186214


namespace even_increasing_function_inequality_l186_186537

theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ {x₁ x₂ : ℝ}, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_increasing_function_inequality_l186_186537


namespace range_f_l186_186010

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) * (Real.cos x) + (Real.cos x) ^ 4 

theorem range_f : Set.Icc (0 : ℝ) (1 : ℝ) = {y : ℝ | ∃ x : ℝ, f x = y} :=
by
  sorry

end range_f_l186_186010


namespace smallest_positive_period_of_y_l186_186982

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.sin (-x / 2 + Real.pi / 4)

-- Statement we need to prove
theorem smallest_positive_period_of_y :
  ∃ T > 0, ∀ x : ℝ, y (x + T) = y x ∧ T = 4 * Real.pi := sorry

end smallest_positive_period_of_y_l186_186982


namespace max_sum_a_b_l186_186959

theorem max_sum_a_b (a b : ℝ) (ha : 4 * a + 3 * b ≤ 10) (hb : 3 * a + 6 * b ≤ 12) : a + b ≤ 22 / 7 :=
sorry

end max_sum_a_b_l186_186959


namespace amount_of_bill_is_1575_l186_186770

noncomputable def time_in_years := (9 : ℝ) / 12

noncomputable def true_discount := 189
noncomputable def rate := 16

noncomputable def face_value (TD : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (TD * 100) / (R * T)

theorem amount_of_bill_is_1575 :
  face_value true_discount rate time_in_years = 1575 := by
  sorry

end amount_of_bill_is_1575_l186_186770


namespace find_a_with_constraints_l186_186920

theorem find_a_with_constraints (x y a : ℝ) 
  (h1 : 2 * x - y + 2 ≥ 0) 
  (h2 : x - 3 * y + 1 ≤ 0)
  (h3 : x + y - 2 ≤ 0)
  (h4 : a > 0)
  (h5 : ∃ (x1 x2 x3 y1 y2 y3 : ℝ), 
    ((x1, y1) = (1, 1) ∨ (x1, y1) = (5 / 3, 1 / 3) ∨ (x1, y1) = (2, 0)) ∧ 
    ((x2, y2) = (1, 1) ∨ (x2, y2) = (5 / 3, 1 / 3) ∨ (x2, y2) = (2, 0)) ∧ 
    ((x3, y3) = (1, 1) ∨ (x3, y3) = (5 / 3, 1 / 3) ∨ (x3, y3) = (2, 0)) ∧ 
    (ax1 - y1 = ax2 - y2) ∧ (ax2 - y2 = ax3 - y3)) :
  a = 1 / 3 :=
sorry

end find_a_with_constraints_l186_186920


namespace resistance_parallel_l186_186993

theorem resistance_parallel (x y r : ℝ) (hy : y = 6) (hr : r = 2.4) 
  (h : 1 / r = 1 / x + 1 / y) : x = 4 :=
  sorry

end resistance_parallel_l186_186993


namespace terminative_decimal_of_45_div_72_l186_186723

theorem terminative_decimal_of_45_div_72 :
  (45 / 72 : ℚ) = 0.625 :=
sorry

end terminative_decimal_of_45_div_72_l186_186723


namespace percentage_problem_l186_186773

theorem percentage_problem (P : ℕ) : (P / 100 * 400 = 20 / 100 * 700) → P = 35 :=
by
  intro h
  sorry

end percentage_problem_l186_186773


namespace eunsung_sungmin_menu_cases_l186_186144

theorem eunsung_sungmin_menu_cases :
  let kinds_of_chicken := 4
  let kinds_of_pizza := 3
  let same_chicken_different_pizza :=
    kinds_of_chicken * (kinds_of_pizza * (kinds_of_pizza - 1))
  let same_pizza_different_chicken :=
    kinds_of_pizza * (kinds_of_chicken * (kinds_of_chicken - 1))
  same_chicken_different_pizza + same_pizza_different_chicken = 60 :=
by
  sorry

end eunsung_sungmin_menu_cases_l186_186144


namespace solution_set_equiv_l186_186375

def solution_set (x : ℝ) : Prop := 2 * x - 6 < 0

theorem solution_set_equiv (x : ℝ) : solution_set x ↔ x < 3 := by
  sorry

end solution_set_equiv_l186_186375


namespace cycle_price_reduction_l186_186663

theorem cycle_price_reduction (original_price : ℝ) :
  let price_after_first_reduction := original_price * 0.75
  let price_after_second_reduction := price_after_first_reduction * 0.60
  (original_price - price_after_second_reduction) / original_price = 0.55 :=
by
  sorry

end cycle_price_reduction_l186_186663


namespace judah_crayons_l186_186153

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end judah_crayons_l186_186153


namespace multiplication_integer_multiple_l186_186786

theorem multiplication_integer_multiple (a b n : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
(h_eq : 10000 * a + b = n * (a * b)) : n = 73 := 
sorry

end multiplication_integer_multiple_l186_186786


namespace arithmetic_sequence_sum_l186_186762

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h1 : a 1 + a 3 + a 5 = 9) (h2 : a 2 + a 4 + a 6 = 15) : a 3 + a 4 = 8 := 
by 
  sorry

end arithmetic_sequence_sum_l186_186762


namespace number_of_boys_l186_186110

-- Define the conditions given in the problem
def total_people := 41
def total_amount := 460
def boy_amount := 12
def girl_amount := 8

-- Define the proof statement that needs to be proven
theorem number_of_boys (B G : ℕ) (h1 : B + G = total_people) (h2 : boy_amount * B + girl_amount * G = total_amount) : B = 33 := 
by {
  -- The actual proof will go here
  sorry
}

end number_of_boys_l186_186110


namespace no_solution_exists_l186_186664

theorem no_solution_exists (f : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, f (f x + 2 * y) = 3 * x + f (f (f y) - x)) :=
sorry

end no_solution_exists_l186_186664


namespace probability_is_one_third_l186_186588

noncomputable def probability_four_of_a_kind_or_full_house : ℚ :=
  let total_outcomes := 6
  let probability_triplet_match := 1 / total_outcomes
  let probability_pair_match := 1 / total_outcomes
  probability_triplet_match + probability_pair_match

theorem probability_is_one_third :
  probability_four_of_a_kind_or_full_house = 1 / 3 :=
by
  -- sorry
  trivial

end probability_is_one_third_l186_186588


namespace min_trips_required_l186_186127

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end min_trips_required_l186_186127


namespace range_of_m_l186_186800

theorem range_of_m (f : ℝ → ℝ) 
  (Hmono : ∀ x y, -2 ≤ x → x ≤ 2 → -2 ≤ y → y ≤ 2 → x ≤ y → f x ≤ f y)
  (Hineq : ∀ m, f (Real.log m / Real.log 2) < f (Real.log (m + 2) / Real.log 4))
  : ∀ m, (1 / 4 : ℝ) ≤ m ∧ m < 2 :=
sorry

end range_of_m_l186_186800


namespace circumference_in_scientific_notation_l186_186867

noncomputable def circumference_m : ℝ := 4010000

noncomputable def scientific_notation (m: ℝ) : Prop :=
  m = 4.01 * 10^6

theorem circumference_in_scientific_notation : scientific_notation circumference_m :=
by
  sorry

end circumference_in_scientific_notation_l186_186867


namespace path_length_of_dot_l186_186660

-- Define the dimensions of the rectangular prism
def prism_width := 1 -- cm
def prism_height := 1 -- cm
def prism_length := 2 -- cm

-- Define the condition that the dot is marked at the center of the top face
def dot_position := (0.5, 1)

-- Define the condition that the prism starts with the 1 cm by 2 cm face on the table
def initial_face_on_table := (prism_length, prism_height)

-- Define the statement to prove the length of the path followed by the dot
theorem path_length_of_dot: 
  ∃ length_of_path : ℝ, length_of_path = 2 * Real.pi :=
sorry

end path_length_of_dot_l186_186660


namespace area_of_square_l186_186895

-- Definitions
def radius_ratio (r R : ℝ) : Prop := R = 7 / 3 * r
def small_circle_circumference (r : ℝ) : Prop := 2 * Real.pi * r = 8
def square_side_length (R side : ℝ) : Prop := side = 2 * R
def square_area (side area : ℝ) : Prop := area = side * side

-- Problem statement
theorem area_of_square (r R side area : ℝ) 
    (h1 : radius_ratio r R)
    (h2 : small_circle_circumference r)
    (h3 : square_side_length R side)
    (h4 : square_area side area) :
    area = 3136 / (9 * Real.pi^2) := 
  by sorry

end area_of_square_l186_186895


namespace cos_225_eq_neg_sqrt2_div_2_l186_186294

theorem cos_225_eq_neg_sqrt2_div_2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_225_eq_neg_sqrt2_div_2_l186_186294


namespace relationship_between_M_and_N_l186_186016

theorem relationship_between_M_and_N (a b : ℝ) (M N : ℝ) 
  (hM : M = a^2 - a * b) 
  (hN : N = a * b - b^2) : M ≥ N :=
by sorry

end relationship_between_M_and_N_l186_186016


namespace range_of_x_l186_186558

theorem range_of_x (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 := 
  sorry

end range_of_x_l186_186558


namespace range_of_m_l186_186712

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 :=
by
  intro h
  sorry

end range_of_m_l186_186712


namespace find_triples_l186_186459

theorem find_triples (a b c : ℕ) :
  (∃ n : ℕ, 2^a + 2^b + 2^c + 3 = n^2) ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end find_triples_l186_186459


namespace min_washes_l186_186556

theorem min_washes (x : ℕ) :
  (1 / 4)^x ≤ 1 / 100 → x ≥ 4 :=
by sorry

end min_washes_l186_186556


namespace least_wins_to_40_points_l186_186907

theorem least_wins_to_40_points 
  (points_per_victory : ℕ)
  (points_per_draw : ℕ)
  (points_per_defeat : ℕ)
  (total_matches : ℕ)
  (initial_points : ℕ)
  (matches_played : ℕ)
  (target_points : ℕ) :
  points_per_victory = 3 →
  points_per_draw = 1 →
  points_per_defeat = 0 →
  total_matches = 20 →
  initial_points = 12 →
  matches_played = 5 →
  target_points = 40 →
  ∃ wins_needed : ℕ, wins_needed = 10 :=
by
  sorry

end least_wins_to_40_points_l186_186907


namespace fraction_of_work_left_l186_186950

theorem fraction_of_work_left (a_days b_days : ℕ) (together_days : ℕ) 
    (h_a : a_days = 15) (h_b : b_days = 20) (h_together : together_days = 4) : 
    (1 - together_days * ((1/a_days : ℚ) + (1/b_days))) = 8/15 := by
  sorry

end fraction_of_work_left_l186_186950


namespace jerry_removed_figures_l186_186708

-- Definitions based on conditions
def initialFigures : ℕ := 3
def addedFigures : ℕ := 4
def currentFigures : ℕ := 6

-- Total figures after adding
def totalFigures := initialFigures + addedFigures

-- Proof statement defining how many figures were removed
theorem jerry_removed_figures : (totalFigures - currentFigures) = 1 := by
  sorry

end jerry_removed_figures_l186_186708


namespace smallest_x_value_l186_186262

theorem smallest_x_value (x : ℝ) (h : 3 * (8 * x^2 + 10 * x + 12) = x * (8 * x - 36)) : x = -3 :=
sorry

end smallest_x_value_l186_186262


namespace variance_of_set_l186_186826

theorem variance_of_set (x : ℝ) (h : (-1 + x + 0 + 1 - 1)/5 = 0) : 
  (1/5) * ( (-1)^2 + (x)^2 + 0^2 + 1^2 + (-1)^2 ) = 0.8 :=
by
  -- placeholder for the proof
  sorry

end variance_of_set_l186_186826


namespace no_positive_integer_solutions_l186_186031

theorem no_positive_integer_solutions (x y : ℕ) (h : x > 0 ∧ y > 0) : x^2 + (x+1)^2 ≠ y^4 + (y+1)^4 :=
by
  intro h1
  sorry

end no_positive_integer_solutions_l186_186031


namespace area_AOC_is_1_l186_186428

noncomputable def point := (ℝ × ℝ) -- Define a point in 2D space

def vector_add (v1 v2 : point) : point :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_zero : point := (0, 0)

def scalar_mul (r : ℝ) (v : point) : point :=
  (r * v.1, r * v.2)

def vector_eq (v1 v2 : point) : Prop := 
  v1.1 = v2.1 ∧ v1.2 = v2.2

variables (A B C O : point)
variable (area_ABC : ℝ)

-- Conditions:
-- Point O is a point inside triangle ABC with an area of 4
-- \(\overrightarrow {OA} + \overrightarrow {OB} + 2\overrightarrow {OC} = \overrightarrow {0}\)
axiom condition_area : area_ABC = 4
axiom condition_vector : vector_eq (vector_add (vector_add O A) (vector_add O B)) (scalar_mul (-2) O)

-- Theorem to prove: the area of triangle AOC is 1
theorem area_AOC_is_1 : (area_ABC / 4) = 1 := 
sorry

end area_AOC_is_1_l186_186428


namespace Jenny_wants_to_read_three_books_l186_186785

noncomputable def books : Nat := 3

-- Definitions based on provided conditions
def reading_speed : Nat := 100 -- words per hour
def book1_words : Nat := 200 
def book2_words : Nat := 400
def book3_words : Nat := 300
def daily_reading_minutes : Nat := 54 
def days : Nat := 10

-- Derived definitions for the proof
def total_words : Nat := book1_words + book2_words + book3_words
def total_hours_needed : ℚ := total_words / reading_speed
def daily_reading_hours : ℚ := daily_reading_minutes / 60
def total_reading_hours : ℚ := daily_reading_hours * days

theorem Jenny_wants_to_read_three_books :
  total_reading_hours = total_hours_needed → books = 3 :=
by
  -- Proof goes here
  sorry

end Jenny_wants_to_read_three_books_l186_186785


namespace arithmetic_lemma_l186_186371

theorem arithmetic_lemma : 45 * 52 + 48 * 45 = 4500 := by
  sorry

end arithmetic_lemma_l186_186371


namespace total_short_trees_after_planting_l186_186711

def current_short_oak_trees := 3
def current_short_pine_trees := 4
def current_short_maple_trees := 5
def new_short_oak_trees := 9
def new_short_pine_trees := 6
def new_short_maple_trees := 4

theorem total_short_trees_after_planting :
  current_short_oak_trees + current_short_pine_trees + current_short_maple_trees +
  new_short_oak_trees + new_short_pine_trees + new_short_maple_trees = 31 := by
  sorry

end total_short_trees_after_planting_l186_186711


namespace cannot_form_set_l186_186623

/-- Define the set of non-negative real numbers not exceeding 20 --/
def setA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 20}

/-- Define the set of solutions of the equation x^2 - 9 = 0 within the real numbers --/
def setB : Set ℝ := {x | x^2 - 9 = 0}

/-- Define the set of all students taller than 170 cm enrolled in a certain school in the year 2013 --/
def setC : Type := sorry

/-- Define the (pseudo) set of all approximate values of sqrt(3) --/
def pseudoSetD : Set ℝ := {x | x = Real.sqrt 3}

/-- Main theorem stating that setD cannot form a mathematically valid set --/
theorem cannot_form_set (x : ℝ) : x ∈ pseudoSetD → False := sorry

end cannot_form_set_l186_186623


namespace problem1_eval_problem2_eval_l186_186776

-- Problem 1 equivalent proof problem
theorem problem1_eval : |(-2 + 1/4)| - (-3/4) + 1 - |(1 - 1/2)| = 3 + 1/2 := 
by
  sorry

-- Problem 2 equivalent proof problem
theorem problem2_eval : -3^2 - (8 / (-2)^3 - 1) + 3 / 2 * (1 / 2) = -6 + 1/4 :=
by
  sorry

end problem1_eval_problem2_eval_l186_186776


namespace inconsistent_conditions_l186_186614

-- Definitions based on the given conditions
def B : Nat := 59
def C : Nat := 27
def D : Nat := 31
def A := B * C + D

theorem inconsistent_conditions (A_is_factor : ∃ k : Nat, 4701 = k * A) : false := by
  sorry

end inconsistent_conditions_l186_186614


namespace original_savings_l186_186807

theorem original_savings (tv_cost : ℝ) (furniture_fraction : ℝ) (total_fraction : ℝ) (original_savings : ℝ) :
  tv_cost = 300 → furniture_fraction = 3 / 4 → total_fraction = 1 → 
  (total_fraction - furniture_fraction) * original_savings = tv_cost →
  original_savings = 1200 :=
by 
  intros htv hfurniture htotal hsavings_eq
  sorry

end original_savings_l186_186807


namespace constant_term_in_binomial_expansion_l186_186722

theorem constant_term_in_binomial_expansion 
  (a b : ℕ) (n : ℕ)
  (sum_of_coefficients : (1 + 1)^n = 4)
  (A B : ℕ)
  (sum_A_B : A + B = 72) 
  (A_value : A = 4) :
  (b^2 = 9) :=
by sorry

end constant_term_in_binomial_expansion_l186_186722


namespace units_digit_division_l186_186551

theorem units_digit_division (a b c d e denom : ℕ)
  (h30 : a = 30) (h31 : b = 31) (h32 : c = 32) (h33 : d = 33) (h34 : e = 34)
  (h120 : denom = 120) :
  ((a * b * c * d * e) / denom) % 10 = 4 :=
by
  sorry

end units_digit_division_l186_186551


namespace inverse_proportion_quadrants_l186_186752

theorem inverse_proportion_quadrants (k b : ℝ) (h1 : b > 0) (h2 : k < 0) :
  ∀ x : ℝ, (x > 0 → (y = kb / x) → y < 0) ∧ (x < 0 → (y = kb / x) → y > 0) :=
by
  sorry

end inverse_proportion_quadrants_l186_186752


namespace at_least_2_boys_and_1_girl_l186_186413

noncomputable def probability_at_least_2_boys_and_1_girl (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_with_0_boys := Nat.choose girls committee_size
  let ways_with_1_boy := Nat.choose boys 1 * Nat.choose girls (committee_size - 1)
  let ways_with_fewer_than_2_boys := ways_with_0_boys + ways_with_1_boy
  1 - (ways_with_fewer_than_2_boys / total_ways)

theorem at_least_2_boys_and_1_girl :
  probability_at_least_2_boys_and_1_girl 32 14 18 6 = 767676 / 906192 :=
by
  sorry

end at_least_2_boys_and_1_girl_l186_186413


namespace marbles_exceed_200_on_sunday_l186_186388

theorem marbles_exceed_200_on_sunday:
  ∃ n : ℕ, 3 * 2^n > 200 ∧ (n % 7) = 0 :=
by
  sorry

end marbles_exceed_200_on_sunday_l186_186388


namespace floor_problem_solution_l186_186116

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end floor_problem_solution_l186_186116


namespace sum_of_numbers_l186_186856

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : (a + b + c) / 3 = a + 20) 
  (h₂ : (a + b + c) / 3 = c - 30) 
  (h₃ : b = 10) : 
  a + b + c = 60 := 
by
  sorry

end sum_of_numbers_l186_186856


namespace unit_digit_hundred_digit_difference_l186_186839

theorem unit_digit_hundred_digit_difference :
  ∃ (A B C : ℕ), 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000 ∧
    99 * (A - C) = 198 ∧ 0 ≤ A ∧ A < 10 ∧ 0 ≤ C ∧ C < 10 ∧ 0 ≤ B ∧ B < 10 → 
  A - C = 2 :=
by 
  -- we only need to state the theorem, actual proof is not required.
  sorry

end unit_digit_hundred_digit_difference_l186_186839


namespace find_k_l186_186024

-- Define the problem conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- Define the dot product for 2D vectors
def dot_prod (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- State the theorem
theorem find_k (k : ℝ) (h : dot_prod b (c k) = 0) : k = -3/2 :=
by
  sorry

end find_k_l186_186024


namespace fraction_operation_correct_l186_186925

theorem fraction_operation_correct 
  (a b : ℝ) : 
  (0.2 * (3 * a + 10 * b) = 6 * a + 20 * b) → 
  (0.1 * (2 * a + 5 * b) = 2 * a + 5 * b) →
  (∀ c : ℝ, c ≠ 0 → (a / b = (a * c) / (b * c))) ∨
  (∀ x y : ℝ, ((x - y) / (x + y) ≠ (y - x) / (x - y))) ∨
  (∀ x : ℝ, (x + x * x * x + x * y ≠ 1 / x * x)) →
  ((0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b)) :=
sorry

end fraction_operation_correct_l186_186925


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l186_186368

theorem option_A_incorrect (a : ℝ) : (a^2) * (a^3) ≠ a^6 :=
by sorry

theorem option_B_incorrect (a : ℝ) : (a^2)^3 ≠ a^5 :=
by sorry

theorem option_C_incorrect (a : ℝ) : (a^6) / (a^2) ≠ a^3 :=
by sorry

theorem option_D_correct (a b : ℝ) : (a + 2 * b) * (a - 2 * b) = a^2 - 4 * b^2 :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l186_186368


namespace power_identity_l186_186026

theorem power_identity (x : ℕ) (h : 2^x = 16) : 2^(x + 3) = 128 := 
sorry

end power_identity_l186_186026


namespace lighter_boxes_weight_l186_186159

noncomputable def weight_lighter_boxes (W L H : ℕ) : Prop :=
  L + H = 30 ∧
  (L * W + H * 20) / 30 = 18 ∧
  (H - 15) = 0 ∧
  (15 + L - H = 15 ∧ 15 * 16 = 15 * W)

theorem lighter_boxes_weight :
  ∃ W, ∀ L H, weight_lighter_boxes W L H → W = 16 :=
by sorry

end lighter_boxes_weight_l186_186159


namespace compute_expression_l186_186410

theorem compute_expression : 11 * (1 / 17) * 34 - 3 = 19 :=
by
  sorry

end compute_expression_l186_186410


namespace circle_radius_l186_186798

theorem circle_radius {r : ℤ} (center: ℝ × ℝ) (inside_pt: ℝ × ℝ) (outside_pt: ℝ × ℝ)
  (h_center: center = (2, 1))
  (h_inside: dist center inside_pt < r)
  (h_outside: dist center outside_pt > r)
  (h_inside_pt: inside_pt = (-2, 1))
  (h_outside_pt: outside_pt = (2, -5))
  (h_integer: r > 0) :
  r = 5 :=
by
  sorry

end circle_radius_l186_186798


namespace determine_a_l186_186381

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem determine_a : (∃ a: ℝ, (∀ x: ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) ∧ ∀ x: ℝ, f x a ≤ 6 → -2 ≤ x ∧ x ≤ 3) ↔ a = 1 :=
by
  sorry

end determine_a_l186_186381


namespace comparison1_comparison2_comparison3_l186_186187

theorem comparison1 : -3.2 > -4.3 :=
by sorry

theorem comparison2 : (1 : ℚ) / 2 > -(1 / 3) :=
by sorry

theorem comparison3 : (1 : ℚ) / 4 > 0 :=
by sorry

end comparison1_comparison2_comparison3_l186_186187


namespace part1_part2_l186_186955

def custom_operation (a b : ℝ) : ℝ := a^2 + 2*a*b

theorem part1 : custom_operation 2 3 = 16 :=
by sorry

theorem part2 (x : ℝ) (h : custom_operation (-2) x = -2 + x) : x = 6 / 5 :=
by sorry

end part1_part2_l186_186955


namespace perimeter_of_square_from_quadratic_roots_l186_186703

theorem perimeter_of_square_from_quadratic_roots :
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  4 * side_length = 40 := by
  let r1 := 1
  let r2 := 10
  let larger_root := if r1 > r2 then r1 else r2
  let area := larger_root * larger_root
  let side_length := Real.sqrt area
  sorry

end perimeter_of_square_from_quadratic_roots_l186_186703


namespace no_nat_n_for_9_pow_n_minus_7_is_product_l186_186815

theorem no_nat_n_for_9_pow_n_minus_7_is_product :
  ¬ ∃ (n k : ℕ), 9 ^ n - 7 = k * (k + 1) :=
by
  sorry

end no_nat_n_for_9_pow_n_minus_7_is_product_l186_186815


namespace length_of_parallelepiped_l186_186842

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l186_186842


namespace tangent_line_intercept_l186_186737

theorem tangent_line_intercept:
  ∃ (m b : ℚ), 
    m > 0 ∧ 
    b = 135 / 28 ∧ 
    (∀ x y : ℚ, (y - 3)^2 + (x - 1)^2 ≥ 3^2 → (y - 8)^2 + (x - 10)^2 ≥ 6^2 → y = m * x + b) := 
sorry

end tangent_line_intercept_l186_186737


namespace stone_length_l186_186263

theorem stone_length (hall_length_m : ℕ) (hall_breadth_m : ℕ) (number_of_stones : ℕ) (stone_width_dm : ℕ) 
    (length_in_dm : 10 > 0) :
    hall_length_m = 36 → hall_breadth_m = 15 → number_of_stones = 2700 → stone_width_dm = 5 →
    ∀ L : ℕ, 
    (10 * hall_length_m) * (10 * hall_breadth_m) = number_of_stones * (L * stone_width_dm) → 
    L = 4 :=
by
  intros h1 h2 h3 h4
  simp at *
  sorry

end stone_length_l186_186263


namespace B_and_C_mutually_exclusive_l186_186900

-- Defining events in terms of products being defective or not
def all_not_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, ¬x

def all_defective (products : List Bool) : Prop := 
  ∀ x ∈ products, x

def not_all_defective (products : List Bool) : Prop := 
  ∃ x ∈ products, ¬x

-- Given a batch of three products, define events A, B, and C
def A (products : List Bool) : Prop := all_not_defective products
def B (products : List Bool) : Prop := all_defective products
def C (products : List Bool) : Prop := not_all_defective products

-- The theorem to prove that B and C are mutually exclusive
theorem B_and_C_mutually_exclusive (products : List Bool) (h : products.length = 3) : 
  ¬ (B products ∧ C products) :=
by
  sorry

end B_and_C_mutually_exclusive_l186_186900


namespace solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l186_186058

-- Problem 1: Prove the solutions to x^2 = 2
theorem solve_quad_eq1 : ∃ x : ℝ, x^2 = 2 ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) :=
by
  sorry

-- Problem 2: Prove the solutions to 4x^2 - 1 = 0
theorem solve_quad_eq2 : ∃ x : ℝ, 4 * x^2 - 1 = 0 ∧ (x = 1/2 ∨ x = -1/2) :=
by
  sorry

-- Problem 3: Prove the solutions to (x-1)^2 - 4 = 0
theorem solve_quad_eq3 : ∃ x : ℝ, (x - 1)^2 - 4 = 0 ∧ (x = 3 ∨ x = -1) :=
by
  sorry

-- Problem 4: Prove the solutions to 12 * (3 - x)^2 - 48 = 0
theorem solve_quad_eq4 : ∃ x : ℝ, 12 * (3 - x)^2 - 48 = 0 ∧ (x = 1 ∨ x = 5) :=
by
  sorry

end solve_quad_eq1_solve_quad_eq2_solve_quad_eq3_solve_quad_eq4_l186_186058


namespace sum_lent_is_300_l186_186825

-- Define the conditions
def interest_rate : ℕ := 4
def time_period : ℕ := 8
def interest_amounted_less : ℕ := 204

-- Prove that the sum lent P is 300 given the conditions
theorem sum_lent_is_300 (P : ℕ) : 
  (P * interest_rate * time_period / 100 = P - interest_amounted_less) -> P = 300 := by
  sorry

end sum_lent_is_300_l186_186825


namespace digit_H_value_l186_186102

theorem digit_H_value (E F G H : ℕ) (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (cond1 : 10 * E + F + 10 * G + E = 10 * H + E)
  (cond2 : 10 * E + F - (10 * G + E) = E)
  (cond3 : E + G = H + 1) : H = 8 :=
sorry

end digit_H_value_l186_186102


namespace sequence_problem_l186_186030

open Nat

theorem sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n : ℕ, 0 < n → S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ ∀ n : ℕ, 0 < n → a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_problem_l186_186030


namespace count_real_solutions_l186_186300

theorem count_real_solutions :
  ∃ x1 x2 : ℝ, (|x1-1| = |x1-2| + |x1-3| + |x1-4| ∧ |x2-1| = |x2-2| + |x2-3| + |x2-4|)
  ∧ (x1 ≠ x2) :=
sorry

end count_real_solutions_l186_186300


namespace maximum_n_for_sequence_l186_186913

theorem maximum_n_for_sequence :
  ∃ (n : ℕ), 
  (∀ a S : ℕ → ℝ, 
    a 1 = 1 → 
    (∀ n : ℕ, n > 0 → 2 * a (n + 1) + S n = 2) → 
    (1001 / 1000 < S (2 * n) / S n ∧ S (2 * n) / S n < 11 / 10)) →
  n = 9 :=
sorry

end maximum_n_for_sequence_l186_186913


namespace find_a_l186_186431

variable (A B : Set ℤ) (a : ℤ)
variable (elem1 : 0 ∈ A) (elem2 : 1 ∈ A)
variable (elem3 : -1 ∈ B) (elem4 : 0 ∈ B) (elem5 : a + 3 ∈ B)

theorem find_a (h : A ⊆ B) : a = -2 := sorry

end find_a_l186_186431


namespace largest_base4_to_base10_l186_186014

theorem largest_base4_to_base10 : 
  (3 * 4^2 + 3 * 4^1 + 3 * 4^0) = 63 := 
by
  -- sorry to skip the proof steps
  sorry

end largest_base4_to_base10_l186_186014


namespace problem_l186_186347

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (M m : ℕ)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 1 ≥ 1
axiom h3 : a 2 ≤ 5
axiom h4 : a 5 ≥ 8

-- Sum function for arithmetic sequence
axiom h5 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

-- Definition of M and m based on S_15
axiom hM : M = max (S 15)
axiom hm : m = min (S 15)

theorem problem (h : S 15 = M + m) : M + m = 600 :=
  sorry

end problem_l186_186347


namespace moles_NaHCO3_combined_l186_186189

-- Define conditions as given in the problem
def moles_HNO3_combined := 1
def moles_NaNO3_result := 1

-- The chemical equation as a definition
def balanced_reaction (moles_NaHCO3 moles_HNO3 moles_NaNO3 : ℕ) : Prop :=
  moles_HNO3 = moles_NaNO3 ∧ moles_NaHCO3 = moles_HNO3

-- The proof problem statement
theorem moles_NaHCO3_combined :
  balanced_reaction 1 moles_HNO3_combined moles_NaNO3_result → 1 = 1 :=
by 
  sorry

end moles_NaHCO3_combined_l186_186189


namespace distance_along_stream_1_hour_l186_186514

noncomputable def boat_speed_still_water : ℝ := 4
noncomputable def stream_speed : ℝ := 2
noncomputable def effective_speed_against_stream : ℝ := boat_speed_still_water - stream_speed
noncomputable def effective_speed_along_stream : ℝ := boat_speed_still_water + stream_speed

theorem distance_along_stream_1_hour : 
  effective_speed_agains_stream = 2 → effective_speed_along_stream * 1 = 6 :=
by
  sorry

end distance_along_stream_1_hour_l186_186514


namespace find_A_in_terms_of_B_and_C_l186_186083

noncomputable def f (A B : ℝ) (x : ℝ) := A * x - 3 * B^2
noncomputable def g (B C : ℝ) (x : ℝ) := B * x + C

theorem find_A_in_terms_of_B_and_C (A B C : ℝ) (h : B ≠ 0) (h1 : f A B (g B C 1) = 0) : A = 3 * B^2 / (B + C) :=
by sorry

end find_A_in_terms_of_B_and_C_l186_186083


namespace PB_length_l186_186253

/-- In a square ABCD with area 1989 cm², with the center O, and
a point P inside such that ∠OPB = 45° and PA : PB = 5 : 14,
prove that PB = 42 cm. -/
theorem PB_length (s PA PB : ℝ) (h₁ : s^2 = 1989) 
(h₂ : PA / PB = 5 / 14) 
(h₃ : 25 * (PA / PB)^2 + 196 * (PB / PA)^2 = s^2) :
  PB = 42 := 
by sorry

end PB_length_l186_186253


namespace angle_bisectors_triangle_l186_186893

theorem angle_bisectors_triangle
  (A B C I D K E : Type)
  (triangle : ∀ (A B C : Type), Prop)
  (is_incenter : ∀ (I A B C : Type), Prop)
  (is_on_arc_centered_at : ∀ (X Y : Type), Prop)
  (is_altitude_intersection : ∀ (X Y : Type), Prop)
  (angle_BIC : ∀ (B C : Type), ℝ)
  (angle_DKE : ∀ (D K E : Type), ℝ)
  (α β γ : ℝ)
  (h_sum_ang : α + β + γ = 180) :
  is_incenter I A B C →
  is_on_arc_centered_at D A → is_on_arc_centered_at K A → is_on_arc_centered_at E A →
  is_altitude_intersection E A →
  angle_BIC B C = 180 - (β + γ) / 2 →
  angle_DKE D K E = (360 - α) / 2 →
  angle_BIC B C + angle_DKE D K E = 270 :=
by sorry

end angle_bisectors_triangle_l186_186893


namespace max_complete_bouquets_l186_186040

-- Definitions based on conditions
def total_roses := 20
def total_lilies := 15
def total_daisies := 10

def wilted_roses := 12
def wilted_lilies := 8
def wilted_daisies := 5

def roses_per_bouquet := 3
def lilies_per_bouquet := 2
def daisies_per_bouquet := 1

-- Calculation of remaining flowers
def remaining_roses := total_roses - wilted_roses
def remaining_lilies := total_lilies - wilted_lilies
def remaining_daisies := total_daisies - wilted_daisies

-- Proof statement
theorem max_complete_bouquets : 
  min
    (remaining_roses / roses_per_bouquet)
    (min (remaining_lilies / lilies_per_bouquet) (remaining_daisies / daisies_per_bouquet)) = 2 :=
by
  sorry

end max_complete_bouquets_l186_186040


namespace find_difference_l186_186171

-- Define the necessary constants and variables
variables (u v : ℝ)

-- Define the conditions
def condition1 := u + v = 360
def condition2 := u = (1/1.1) * v

-- Define the theorem to prove
theorem find_difference (h1 : condition1 u v) (h2 : condition2 u v) : v - u = 17 := 
sorry

end find_difference_l186_186171


namespace train_length_l186_186644

theorem train_length :
  (∃ (L : ℝ), (L / 30 = (L + 2500) / 120) ∧ L = 75000 / 90) :=
sorry

end train_length_l186_186644


namespace percentage_of_class_taking_lunch_l186_186047

theorem percentage_of_class_taking_lunch 
  (total_students : ℕ)
  (boys_ratio : ℕ := 6)
  (girls_ratio : ℕ := 4)
  (boys_percentage_lunch : ℝ := 0.60)
  (girls_percentage_lunch : ℝ := 0.40) :
  total_students = 100 →
  (6 / (6 + 4) * 100) = 60 →
  (4 / (6 + 4) * 100) = 40 →
  (boys_percentage_lunch * 60 + girls_percentage_lunch * 40) = 52 →
  ℝ :=
    by
      intros
      sorry

end percentage_of_class_taking_lunch_l186_186047


namespace range_of_a_l186_186490

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end range_of_a_l186_186490


namespace three_term_inequality_l186_186421

theorem three_term_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) (h_abc : a * b * c = 1) :
  (a^3 / (a^3 + 2 * b^2)) + (b^3 / (b^3 + 2 * c^2)) + (c^3 / (c^3 + 2 * a^2)) ≥ 1 :=
by
  sorry

end three_term_inequality_l186_186421


namespace unique_solution_condition_l186_186497

theorem unique_solution_condition (p q : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by
  sorry

end unique_solution_condition_l186_186497


namespace example_problem_l186_186529

-- Definitions and conditions derived from the original problem statement
def smallest_integer_with_two_divisors (m : ℕ) : Prop := m = 2
def second_largest_integer_with_three_divisors_less_than_100 (n : ℕ) : Prop := n = 25

theorem example_problem (m n : ℕ) 
    (h1 : smallest_integer_with_two_divisors m) 
    (h2 : second_largest_integer_with_three_divisors_less_than_100 n) : 
    m + n = 27 :=
by sorry

end example_problem_l186_186529


namespace fg_of_2_eq_0_l186_186398

def f (x : ℝ) : ℝ := 4 - x^2
def g (x : ℝ) : ℝ := 3 * x - x^3

theorem fg_of_2_eq_0 : f (g 2) = 0 := by
  sorry

end fg_of_2_eq_0_l186_186398


namespace capacity_of_smaller_bucket_l186_186353

theorem capacity_of_smaller_bucket (x : ℕ) (h1 : x < 5) (h2 : 5 - x = 2) : x = 3 := by
  sorry

end capacity_of_smaller_bucket_l186_186353


namespace A_squared_plus_B_squared_eq_one_l186_186535

theorem A_squared_plus_B_squared_eq_one
  (A B : ℝ) (h1 : A ≠ B)
  (h2 : ∀ x : ℝ, (A * (B * x ^ 2 + A) ^ 2 + B - (B * (A * x ^ 2 + B) ^ 2 + A)) = B ^ 2 - A ^ 2) :
  A ^ 2 + B ^ 2 = 1 :=
sorry

end A_squared_plus_B_squared_eq_one_l186_186535


namespace max_ratio_is_99_over_41_l186_186470

noncomputable def max_ratio (x y : ℕ) (h1 : x > y) (h2 : x + y = 140) : ℚ :=
  if h : y ≠ 0 then (x / y : ℚ) else 0

theorem max_ratio_is_99_over_41 : ∃ (x y : ℕ), x > y ∧ x + y = 140 ∧ max_ratio x y (by sorry) (by sorry) = (99 / 41 : ℚ) :=
by
  sorry

end max_ratio_is_99_over_41_l186_186470


namespace min_students_with_same_score_l186_186462

noncomputable def highest_score : ℕ := 83
noncomputable def lowest_score : ℕ := 30
noncomputable def total_students : ℕ := 8000
noncomputable def range_scores : ℕ := (highest_score - lowest_score + 1)

theorem min_students_with_same_score :
  ∃ k : ℕ, k = Nat.ceil (total_students / range_scores) ∧ k = 149 :=
by
  sorry

end min_students_with_same_score_l186_186462


namespace necessary_but_not_sufficient_condition_l186_186747

theorem necessary_but_not_sufficient_condition :
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬ (x > 2)) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l186_186747


namespace quadratic_expression_value_l186_186464

theorem quadratic_expression_value (x₁ x₂ : ℝ) (h₁ : x₁^2 - 3 * x₁ + 1 = 0) (h₂ : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^2 + 3 * x₂ + x₁ * x₂ - 2 = 7 :=
by
  sorry

end quadratic_expression_value_l186_186464


namespace factor_difference_of_squares_l186_186813
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l186_186813


namespace triangle_at_most_one_obtuse_l186_186112

theorem triangle_at_most_one_obtuse 
  (A B C : ℝ)
  (h_sum : A + B + C = 180) 
  (h_obtuse_A : A > 90) 
  (h_obtuse_B : B > 90) 
  (h_obtuse_C : C > 90) :
  false :=
by 
  sorry

end triangle_at_most_one_obtuse_l186_186112


namespace common_area_of_rectangle_and_circle_l186_186684

theorem common_area_of_rectangle_and_circle (r : ℝ) (a b : ℝ) (h_center : r = 5) (h_dim : a = 10 ∧ b = 4) :
  let sector_area := (25 * Real.pi) / 2 
  let triangle_area := 4 * Real.sqrt 21 
  let result := sector_area + triangle_area 
  result = (25 * Real.pi) / 2 + 4 * Real.sqrt 21 := 
by
  sorry

end common_area_of_rectangle_and_circle_l186_186684


namespace second_year_associates_l186_186304

theorem second_year_associates (not_first_year : ℝ) (more_than_two_years : ℝ) 
  (h1 : not_first_year = 0.75) (h2 : more_than_two_years = 0.5) : 
  (not_first_year - more_than_two_years) = 0.25 :=
by 
  sorry

end second_year_associates_l186_186304


namespace chad_bbq_people_l186_186028

theorem chad_bbq_people (ice_cost_per_pack : ℝ) (packs_included : ℕ) (total_money_spent : ℝ) (pounds_needed_per_person : ℝ) :
  total_money_spent = 9 → 
  ice_cost_per_pack = 3 → 
  packs_included = 10 → 
  pounds_needed_per_person = 2 → 
  ∃ (people : ℕ), people = 15 :=
by intros; sorry

end chad_bbq_people_l186_186028


namespace sqrt3_times_3_minus_sqrt3_bound_l186_186943

theorem sqrt3_times_3_minus_sqrt3_bound : 2 < (Real.sqrt 3) * (3 - (Real.sqrt 3)) ∧ (Real.sqrt 3) * (3 - (Real.sqrt 3)) < 3 := 
by 
  sorry

end sqrt3_times_3_minus_sqrt3_bound_l186_186943


namespace Brenda_mice_left_l186_186503

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end Brenda_mice_left_l186_186503


namespace sum_of_ages_l186_186635

variable (P_years Q_years : ℝ) (D_years : ℝ)

-- conditions
def condition_1 : Prop := Q_years = 37.5
def condition_2 : Prop := P_years = 3 * (Q_years - D_years)
def condition_3 : Prop := P_years - Q_years = D_years

-- statement to prove
theorem sum_of_ages (h1 : condition_1 Q_years) (h2 : condition_2 P_years Q_years D_years) (h3 : condition_3 P_years Q_years D_years) :
  P_years + Q_years = 93.75 :=
by sorry

end sum_of_ages_l186_186635


namespace arithmetic_sequence_sum_10_l186_186989

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

end arithmetic_sequence_sum_10_l186_186989


namespace find_x_l186_186327

variable (a b x : ℝ)

def condition1 : Prop := a / b = 5 / 4
def condition2 : Prop := (4 * a + x * b) / (4 * a - x * b) = 4

theorem find_x (h1 : condition1 a b) (h2 : condition2 a b x) : x = 3 :=
  sorry

end find_x_l186_186327


namespace plane_speeds_l186_186954

theorem plane_speeds (v : ℕ) 
    (h1 : ∀ (t : ℕ), t = 5 → 20 * v = 4800): 
  v = 240 ∧ 3 * v = 720 := by
  sorry

end plane_speeds_l186_186954


namespace at_least_one_neg_l186_186274

theorem at_least_one_neg (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
sorry

end at_least_one_neg_l186_186274


namespace totalCostOfAllPuppies_l186_186062

noncomputable def goldenRetrieverCost : ℕ :=
  let numberOfGoldenRetrievers := 3
  let puppiesPerGoldenRetriever := 4
  let shotsPerPuppy := 2
  let costPerShot := 5
  let vitaminCostPerMonth := 12
  let monthsOfSupplements := 6
  numberOfGoldenRetrievers * puppiesPerGoldenRetriever *
  (shotsPerPuppy * costPerShot + vitaminCostPerMonth * monthsOfSupplements)

noncomputable def germanShepherdCost : ℕ :=
  let numberOfGermanShepherds := 2
  let puppiesPerGermanShepherd := 5
  let shotsPerPuppy := 3
  let costPerShot := 8
  let microchipCost := 25
  let toyCost := 15
  numberOfGermanShepherds * puppiesPerGermanShepherd *
  (shotsPerPuppy * costPerShot + microchipCost + toyCost)

noncomputable def bulldogCost : ℕ :=
  let numberOfBulldogs := 4
  let puppiesPerBulldog := 3
  let shotsPerPuppy := 4
  let costPerShot := 10
  let collarCost := 20
  let chewToyCost := 18
  numberOfBulldogs * puppiesPerBulldog *
  (shotsPerPuppy * costPerShot + collarCost + chewToyCost)

theorem totalCostOfAllPuppies : goldenRetrieverCost + germanShepherdCost + bulldogCost = 2560 :=
by
  sorry

end totalCostOfAllPuppies_l186_186062


namespace candies_problem_max_children_l186_186140

theorem candies_problem_max_children (u v : ℕ → ℕ) (n : ℕ) :
  (∀ i : ℕ, u i = v i + 2) →
  (∀ i : ℕ, u i + 2 = u (i + 1)) →
  (u (n - 1) / u 0 = 13) →
  n = 25 :=
by
  -- Proof not required as per the instructions.
  sorry

end candies_problem_max_children_l186_186140


namespace instantaneous_velocity_at_t_3_l186_186675

variable (t : ℝ)
def s (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_t_3 : 
  ∃ v, v = -1 + 2 * 3 ∧ v = 5 :=
by
  sorry

end instantaneous_velocity_at_t_3_l186_186675


namespace minimum_additional_small_bottles_needed_l186_186401

-- Definitions from the problem conditions
def small_bottle_volume : ℕ := 45
def large_bottle_total_volume : ℕ := 600
def initial_volume_in_large_bottle : ℕ := 90

-- The proof problem: How many more small bottles does Jasmine need to fill the large bottle?
theorem minimum_additional_small_bottles_needed : 
  (large_bottle_total_volume - initial_volume_in_large_bottle + small_bottle_volume - 1) / small_bottle_volume = 12 := 
by 
  sorry

end minimum_additional_small_bottles_needed_l186_186401


namespace cashier_overestimation_l186_186720

def nickel_value := 5
def dime_value := 10
def quarter_value := 25
def half_dollar_value := 50

def nickels_counted_as_dimes := 15
def quarters_counted_as_half_dollars := 10

noncomputable def overestimation_due_to_nickels_as_dimes : Nat := 
  (dime_value - nickel_value) * nickels_counted_as_dimes

noncomputable def overestimation_due_to_quarters_as_half_dollars : Nat := 
  (half_dollar_value - quarter_value) * quarters_counted_as_half_dollars

noncomputable def total_overestimation : Nat := 
  overestimation_due_to_nickels_as_dimes + overestimation_due_to_quarters_as_half_dollars

theorem cashier_overestimation : total_overestimation = 325 := by
  sorry

end cashier_overestimation_l186_186720


namespace triangle_inequality_l186_186569

variable {α β γ a b c: ℝ}

theorem triangle_inequality (h1 : α + β + γ = π)
  (h2 : α > 0) (h3 : β > 0) (h4 : γ > 0)
  (h5 : a > 0) (h6 : b > 0) (h7 : c > 0)
  (h8 : (α > β ∧ a > b) ∨ (α = β ∧ a = b) ∨ (α < β ∧ a < b))
  (h9 : (β > γ ∧ b > c) ∨ (β = γ ∧ b = c) ∨ (β < γ ∧ b < c))
  (h10 : (γ > α ∧ c > a) ∨ (γ = α ∧ c = a) ∨ (γ < α ∧ c < a)) :
  (π / 3) ≤ (a * α + b * β + c * γ) / (a + b + c) ∧
  (a * α + b * β + c * γ) / (a + b + c) < (π / 2) :=
sorry

end triangle_inequality_l186_186569


namespace kids_go_to_camp_l186_186658

-- Define the total number of kids in Lawrence County
def total_kids : ℕ := 1059955

-- Define the number of kids who stay home
def stay_home : ℕ := 495718

-- Define the expected number of kids who go to camp
def expected_go_to_camp : ℕ := 564237

-- The theorem to prove the number of kids who go to camp
theorem kids_go_to_camp :
  total_kids - stay_home = expected_go_to_camp :=
by
  -- Proof is omitted
  sorry

end kids_go_to_camp_l186_186658


namespace solve_equation_l186_186577

def equation_solution (x : ℝ) : Prop :=
  (x^2 + x + 1) / (x + 1) = x + 3

theorem solve_equation :
  ∃ x : ℝ, equation_solution x ∧ x = -2 / 3 :=
by
  sorry

end solve_equation_l186_186577


namespace sin_thirty_degree_l186_186938

theorem sin_thirty_degree : Real.sin (30 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end sin_thirty_degree_l186_186938


namespace ben_weekly_eggs_l186_186021

-- Definitions for the conditions
def weekly_saly_eggs : ℕ := 10
def weekly_ben_eggs (B : ℕ) : ℕ := B
def weekly_ked_eggs (B : ℕ) : ℕ := B / 2

def weekly_production (B : ℕ) : ℕ :=
  weekly_saly_eggs + weekly_ben_eggs B + weekly_ked_eggs B

def monthly_production (B : ℕ) : ℕ := 4 * weekly_production B

-- Theorem for the proof
theorem ben_weekly_eggs (B : ℕ) (h : monthly_production B = 124) : B = 14 :=
sorry

end ben_weekly_eggs_l186_186021


namespace remainder_when_abc_divided_by_7_l186_186892

theorem remainder_when_abc_divided_by_7 (a b c : ℕ) (h0 : a < 7) (h1 : b < 7) (h2 : c < 7)
  (h3 : (a + 2 * b + 3 * c) % 7 = 0)
  (h4 : (2 * a + 3 * b + c) % 7 = 4)
  (h5 : (3 * a + b + 2 * c) % 7 = 4) :
  (a * b * c) % 7 = 6 := 
sorry

end remainder_when_abc_divided_by_7_l186_186892


namespace compute_fraction_power_l186_186564

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l186_186564


namespace marbles_ratio_l186_186240

theorem marbles_ratio (miriam_current_marbles miriam_initial_marbles marbles_brother marbles_sister marbles_total_given marbles_savanna : ℕ)
  (h1 : miriam_current_marbles = 30)
  (h2 : marbles_brother = 60)
  (h3 : marbles_sister = 2 * marbles_brother)
  (h4 : miriam_initial_marbles = 300)
  (h5 : marbles_total_given = miriam_initial_marbles - miriam_current_marbles)
  (h6 : marbles_savanna = marbles_total_given - (marbles_brother + marbles_sister)) :
  (marbles_savanna : ℚ) / miriam_current_marbles = 3 :=
by
  sorry

end marbles_ratio_l186_186240


namespace number_of_keepers_l186_186465

theorem number_of_keepers
  (h₁ : 50 * 2 = 100)
  (h₂ : 45 * 4 = 180)
  (h₃ : 8 * 4 = 32)
  (h₄ : 12 * 8 = 96)
  (h₅ : 6 * 8 = 48)
  (h₆ : 100 + 180 + 32 + 96 + 48 = 456)
  (h₇ : 50 + 45 + 8 + 12 + 6 = 121)
  (h₈ : ∀ K : ℕ, (2 * (K - 5) + 6 + 2 = 2 * K - 2))
  (h₉ : ∀ K : ℕ, 121 + K + 372 = 456 + (2 * K - 2)) :
  ∃ K : ℕ, K = 39 :=
by
  sorry

end number_of_keepers_l186_186465


namespace solve_for_x_l186_186567

theorem solve_for_x (x : ℝ) (h : (1/3 : ℝ) * (x + 8 + 5*x + 3 + 3*x + 4) = 4*x + 1) : x = 4 :=
by {
  sorry
}

end solve_for_x_l186_186567


namespace cos_150_eq_neg_sqrt3_over_2_l186_186622

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l186_186622


namespace mul_binom_expansion_l186_186732

variable (a : ℝ)

theorem mul_binom_expansion : (a + 1) * (a - 1) = a^2 - 1 :=
by
  sorry

end mul_binom_expansion_l186_186732


namespace one_thirds_in_fraction_l186_186969

theorem one_thirds_in_fraction : (9 / 5) / (1 / 3) = 27 / 5 := by
  sorry

end one_thirds_in_fraction_l186_186969


namespace money_sum_l186_186626

theorem money_sum (A B C : ℕ) (h1 : A + C = 300) (h2 : B + C = 600) (h3 : C = 200) : A + B + C = 700 :=
by
  sorry

end money_sum_l186_186626


namespace P_not_77_for_all_integers_l186_186513

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end P_not_77_for_all_integers_l186_186513


namespace fernanda_total_time_eq_90_days_l186_186326

-- Define the conditions
def num_audiobooks : ℕ := 6
def hours_per_audiobook : ℕ := 30
def hours_listened_per_day : ℕ := 2

-- Define the total time calculation
def total_time_to_finish_audiobooks (a h r : ℕ) : ℕ :=
  (h / r) * a

-- The assertion we need to prove
theorem fernanda_total_time_eq_90_days :
  total_time_to_finish_audiobooks num_audiobooks hours_per_audiobook hours_listened_per_day = 90 :=
by sorry

end fernanda_total_time_eq_90_days_l186_186326


namespace min_value_of_sum_squares_l186_186202

theorem min_value_of_sum_squares (a b : ℝ) (h : (9 / a^2) + (4 / b^2) = 1) : a^2 + b^2 ≥ 25 :=
sorry

end min_value_of_sum_squares_l186_186202


namespace largest_integral_solution_l186_186679

theorem largest_integral_solution (x : ℤ) : (1 / 4 : ℝ) < (x / 7 : ℝ) ∧ (x / 7 : ℝ) < (3 / 5 : ℝ) → x = 4 :=
by {
  sorry
}

end largest_integral_solution_l186_186679


namespace find_n_l186_186533

-- Define the operation ø
def op (x w : ℕ) : ℕ := (2 ^ x) / (2 ^ w)

-- Prove that n operating with 2 and then 1 equals 8 implies n = 3
theorem find_n (n : ℕ) (H : op (op n 2) 1 = 8) : n = 3 :=
by
  -- Proof will be provided later
  sorry

end find_n_l186_186533


namespace triangle_area_range_l186_186745

theorem triangle_area_range (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : x₁ < 1) (h₂ : 1 < x₂) (h₃ : x₁ * x₂ = 1) :
  0 < (2 / (x₁ + 1 / x₁)) ∧ (2 / (x₁ + 1 / x₁)) < 1 :=
by
  sorry

end triangle_area_range_l186_186745


namespace carl_first_to_roll_six_l186_186628

-- Definitions based on problem conditions
def prob_six := 1 / 6
def prob_not_six := 5 / 6

-- Define geometric series sum formula for the given context
theorem carl_first_to_roll_six :
  ∑' n : ℕ, (prob_not_six^(3*n+1) * prob_six) = 25 / 91 :=
by
  sorry

end carl_first_to_roll_six_l186_186628


namespace cylindrical_container_depth_l186_186309

theorem cylindrical_container_depth :
    ∀ (L D A : ℝ), 
      L = 12 ∧ D = 8 ∧ A = 48 → (∃ h : ℝ, h = 4 - 2 * Real.sqrt 3) :=
by
  intros L D A h_cond
  obtain ⟨hL, hD, hA⟩ := h_cond
  sorry

end cylindrical_container_depth_l186_186309


namespace sheila_picnic_probability_l186_186339

theorem sheila_picnic_probability :
  let P_rain := 0.5
  let P_go_given_rain := 0.3
  let P_go_given_sunny := 0.9
  let P_remember := 0.9  -- P(remember) = 1 - P(forget)
  let P_sunny := 1 - P_rain
  
  P_rain * P_go_given_rain * P_remember + P_sunny * P_go_given_sunny * P_remember = 0.54 :=
by
  sorry

end sheila_picnic_probability_l186_186339


namespace vertex_property_l186_186812

theorem vertex_property (a b c m k : ℝ) (h : a ≠ 0)
  (vertex_eq : k = a * m^2 + b * m + c)
  (point_eq : m = a * k^2 + b * k + c) : a * (m - k) > 0 :=
sorry

end vertex_property_l186_186812


namespace nicholas_paid_more_than_kenneth_l186_186538

def price_per_yard : ℝ := 40
def kenneth_yards : ℝ := 700
def nicholas_multiplier : ℝ := 6
def discount_rate : ℝ := 0.15

def kenneth_total_cost : ℝ := price_per_yard * kenneth_yards
def nicholas_yards : ℝ := nicholas_multiplier * kenneth_yards
def nicholas_original_cost : ℝ := price_per_yard * nicholas_yards
def discount_amount : ℝ := discount_rate * nicholas_original_cost
def nicholas_discounted_cost : ℝ := nicholas_original_cost - discount_amount
def difference_in_cost : ℝ := nicholas_discounted_cost - kenneth_total_cost

theorem nicholas_paid_more_than_kenneth :
  difference_in_cost = 114800 := by
  sorry

end nicholas_paid_more_than_kenneth_l186_186538


namespace greatest_possible_value_y_l186_186203

theorem greatest_possible_value_y
  (x y : ℤ)
  (h : x * y + 3 * x + 2 * y = -6) : 
  y ≤ 3 :=
by sorry

end greatest_possible_value_y_l186_186203


namespace problem_statement_l186_186345

noncomputable def f : ℝ → ℝ := sorry

variable (α : ℝ)

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 3) = -f x
axiom tan_alpha : Real.tan α = 2

theorem problem_statement : f (15 * Real.sin α * Real.cos α) = 0 := 
by {
  sorry
}

end problem_statement_l186_186345


namespace max_value_of_abs_z_plus_4_l186_186510

open Complex
noncomputable def max_abs_z_plus_4 {z : ℂ} (h : abs (z + 3 * I) = 5) : ℝ :=
sorry

theorem max_value_of_abs_z_plus_4 (z : ℂ) (h : abs (z + 3 * I) = 5) : abs (z + 4) ≤ 10 :=
sorry

end max_value_of_abs_z_plus_4_l186_186510


namespace ten_percent_of_x_is_17_85_l186_186270

-- Define the conditions and the proof statement
theorem ten_percent_of_x_is_17_85 :
  ∃ x : ℝ, (3 - (1/4) * 2 - (1/3) * 3 - (1/7) * x = 27) ∧ (0.10 * x = 17.85) := sorry

end ten_percent_of_x_is_17_85_l186_186270


namespace problem_proof_l186_186167

theorem problem_proof (x : ℝ) (h : x + 1/x = 3) : (x - 3) ^ 2 + 36 / (x - 3) ^ 2 = 12 :=
sorry

end problem_proof_l186_186167


namespace count_japanese_stamps_l186_186764

theorem count_japanese_stamps (total_stamps : ℕ) (perc_chinese perc_us : ℕ) (h1 : total_stamps = 100) 
  (h2 : perc_chinese = 35) (h3 : perc_us = 20): 
  total_stamps - ((perc_chinese * total_stamps / 100) + (perc_us * total_stamps / 100)) = 45 :=
by
  sorry

end count_japanese_stamps_l186_186764


namespace Julie_simple_interest_l186_186212

variable (S : ℝ) (r : ℝ) (A : ℝ) (C : ℝ)

def initially_savings (S : ℝ) := S = 784
def half_savings_in_each_account (S A : ℝ) := A = S / 2
def compound_interest_after_two_years (A r : ℝ) := A * (1 + r)^2 - A = 120

theorem Julie_simple_interest
  (S : ℝ) (r : ℝ) (A : ℝ)
  (h1 : initially_savings S)
  (h2 : half_savings_in_each_account S A)
  (h3 : compound_interest_after_two_years A r) :
  A * r * 2 = 112 :=
by 
  sorry

end Julie_simple_interest_l186_186212


namespace range_of_m_l186_186978

theorem range_of_m (m : ℝ) : (∀ x, 0 ≤ x ∧ x ≤ m → -6 ≤ x^2 - 4 * x - 2 ∧ x^2 - 4 * x - 2 ≤ -2) → 2 ≤ m ∧ m ≤ 4 :=
by
  sorry

end range_of_m_l186_186978


namespace optimal_roof_angle_no_friction_l186_186242

theorem optimal_roof_angle_no_friction {g x : ℝ} (hg : 0 < g) (hx : 0 < x) :
  ∃ α : ℝ, α = 45 :=
by
  sorry

end optimal_roof_angle_no_friction_l186_186242


namespace nine_a_plus_a_plus_nine_l186_186818

theorem nine_a_plus_a_plus_nine (A : Nat) (hA : 0 < A) : 
  10 * A + 9 = 9 * A + (A + 9) := 
by 
  sorry

end nine_a_plus_a_plus_nine_l186_186818


namespace reciprocal_of_2023_l186_186142

theorem reciprocal_of_2023 : (2023 : ℝ)⁻¹ = 1 / 2023 :=
by
  sorry

end reciprocal_of_2023_l186_186142


namespace max_min_f_l186_186995

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

end max_min_f_l186_186995


namespace phoenix_hike_length_l186_186157

theorem phoenix_hike_length (a b c d : ℕ)
  (h1 : a + b = 22)
  (h2 : b + c = 26)
  (h3 : c + d = 30)
  (h4 : a + c = 26) :
  a + b + c + d = 52 :=
sorry

end phoenix_hike_length_l186_186157


namespace range_of_a_for_inequality_solutions_to_equation_l186_186797

noncomputable def f (x a : ℝ) := x^2 + 2 * a * x + 1
noncomputable def f_prime (x a : ℝ) := 2 * x + 2 * a

theorem range_of_a_for_inequality :
  (∀ x, -2 ≤ x ∧ x ≤ -1 → f x a ≤ f_prime x a) → a ≥ 3 / 2 :=
sorry

theorem solutions_to_equation (a : ℝ) (x : ℝ) :
  f x a = |f_prime x a| ↔ 
  (if a < -1 then x = -1 ∨ x = 1 - 2 * a 
  else if -1 ≤ a ∧ a ≤ 1 then x = 1 ∨ x = -1 ∨ x = 1 - 2 * a ∨ x = -(1 + 2 * a)
  else x = 1 ∨ x = -(1 + 2 * a)) :=
sorry

end range_of_a_for_inequality_solutions_to_equation_l186_186797


namespace inequality_on_abc_l186_186364

variable (a b c : ℝ)

theorem inequality_on_abc (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  (a^4 + b^4) / (a^6 + b^6) + (b^4 + c^4) / (b^6 + c^6) + (c^4 + a^4) / (c^6 + a^6) ≤ 1 / (a * b * c) :=
by
  sorry

end inequality_on_abc_l186_186364


namespace negation_equiv_l186_186430

theorem negation_equiv (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l186_186430


namespace algebra_expression_value_l186_186701

variable (x : ℝ)

theorem algebra_expression_value (h : x^2 - 3 * x - 12 = 0) : 3 * x^2 - 9 * x + 5 = 41 := 
sorry

end algebra_expression_value_l186_186701


namespace foreign_exchange_decline_l186_186831

theorem foreign_exchange_decline (x : ℝ) (h1 : 200 * (1 - x)^2 = 98) : 
  200 * (1 - x)^2 = 98 :=
by
  sorry

end foreign_exchange_decline_l186_186831


namespace sufficient_not_necessary_condition_l186_186269

variable {a : ℝ}

theorem sufficient_not_necessary_condition (ha : a > 1 / a^2) :
  a^2 > 1 / a ∧ ∃ a, a^2 > 1 / a ∧ ¬(a > 1 / a^2) :=
by
  sorry

end sufficient_not_necessary_condition_l186_186269


namespace linear_func_is_direct_proportion_l186_186414

theorem linear_func_is_direct_proportion (m : ℝ) : (∀ x : ℝ, (y : ℝ) → y = m * x + m - 2 → (m - 2 = 0) → y = 0) → m = 2 :=
by
  intros h
  have : m - 2 = 0 := sorry
  exact sorry

end linear_func_is_direct_proportion_l186_186414


namespace year_when_mother_age_is_twice_jack_age_l186_186756

noncomputable def jack_age_2010 := 12
noncomputable def mother_age_2010 := 3 * jack_age_2010

theorem year_when_mother_age_is_twice_jack_age :
  ∃ x : ℕ, mother_age_2010 + x = 2 * (jack_age_2010 + x) ∧ (2010 + x = 2022) :=
by
  sorry

end year_when_mother_age_is_twice_jack_age_l186_186756


namespace lcm_18_24_l186_186008

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  -- Sorry is place-holder for the actual proof.
  sorry

end lcm_18_24_l186_186008


namespace distance_between_a_and_c_l186_186287

-- Given conditions
variables (a : ℝ)

-- Statement to prove
theorem distance_between_a_and_c : |a + 1| = |a - (-1)| :=
by sorry

end distance_between_a_and_c_l186_186287


namespace range_of_a_l186_186667

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, 2 * a * (x : ℝ)^2 - 4 * (x : ℝ) < a * (x : ℝ) - 2 → ∃! x₀ : ℤ, x₀ = x) → 1 ≤ a ∧ a < 2 :=
sorry

end range_of_a_l186_186667


namespace vehicle_value_this_year_l186_186208

variable (V_last_year : ℝ) (V_this_year : ℝ)

-- Conditions
def last_year_value : ℝ := 20000
def this_year_value : ℝ := 0.8 * last_year_value

theorem vehicle_value_this_year :
  V_last_year = last_year_value →
  V_this_year = this_year_value →
  V_this_year = 16000 := sorry

end vehicle_value_this_year_l186_186208


namespace pizza_problem_l186_186027

theorem pizza_problem :
  ∃ (x : ℕ), x = 20 ∧ (3 * x ^ 2 = 3 * 14 ^ 2 * 2 + 49) :=
by
  let small_pizza_side := 14
  let large_pizza_cost := 20
  let pool_cost := 60
  let individually_cost := 30
  have total_individual_area := 2 * 3 * (small_pizza_side ^ 2)
  have extra_area := 49
  sorry

end pizza_problem_l186_186027


namespace smallest_solution_floor_equation_l186_186307

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l186_186307


namespace ways_to_stand_on_staircase_l186_186243

theorem ways_to_stand_on_staircase (A B C : Type) (steps : Fin 7) : 
  ∃ ways : Nat, ways = 336 := by sorry

end ways_to_stand_on_staircase_l186_186243


namespace f_odd_and_increasing_l186_186631

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := sorry

end f_odd_and_increasing_l186_186631


namespace monica_tiles_l186_186013

-- Define the dimensions of the living room
def living_room_length : ℕ := 20
def living_room_width : ℕ := 15

-- Define the size of the border tiles and inner tiles
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

-- Prove the number of tiles used is 44
theorem monica_tiles (border_tile_count inner_tile_count total_tiles : ℕ)
  (h_border : border_tile_count = ((2 * ((living_room_length - 4) / border_tile_size) + 2 * ((living_room_width - 4) / border_tile_size) - 4)))
  (h_inner : inner_tile_count = (176 / (inner_tile_size * inner_tile_size)))
  (h_total : total_tiles = border_tile_count + inner_tile_count) :
  total_tiles = 44 :=
by
  sorry

end monica_tiles_l186_186013


namespace unique_solution_triple_l186_186542

theorem unique_solution_triple {a b c : ℝ} (h₀ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h₁ : a^2 + b^2 + c^2 = 3) (h₂ : (a + b + c) * (a^2 * b + b^2 * c + c^2 * a) = 9) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ c = 1 ∧ b = 1) ∨ (b = 1 ∧ a = 1 ∧ c = 1) ∨ (b = 1 ∧ c = 1 ∧ a = 1) ∨ (c = 1 ∧ a = 1 ∧ b = 1) ∨ (c = 1 ∧ b = 1 ∧ a = 1) :=
sorry

end unique_solution_triple_l186_186542


namespace negation_of_universal_statement_l186_186861

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_of_universal_statement_l186_186861


namespace sampling_is_systematic_l186_186643

-- Conditions
def production_line (units_per_day : ℕ) : Prop := units_per_day = 128

def sampling_inspection (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  samples_per_day = 8 ∧ inspection_time = 30 ∧ inspection_days = 7

-- Question
def sampling_method (method : String) (units_per_day : ℕ) (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  production_line units_per_day ∧ sampling_inspection samples_per_day inspection_time inspection_days → method = "systematic sampling"

-- Theorem stating the question == answer given conditions
theorem sampling_is_systematic : sampling_method "systematic sampling" 128 8 30 7 :=
by
  sorry

end sampling_is_systematic_l186_186643


namespace find_x_l186_186657

theorem find_x (x : ℤ) (h_pos : x > 0) 
  (n := x^2 + 2 * x + 17) 
  (d := 2 * x + 5)
  (h_div : n = d * x + 7) : x = 2 := 
sorry

end find_x_l186_186657


namespace equal_real_roots_implies_m_l186_186314

theorem equal_real_roots_implies_m (m : ℝ) : (∃ (x : ℝ), x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) → m = 1/4 :=
by
  sorry

end equal_real_roots_implies_m_l186_186314


namespace basketball_team_free_throws_l186_186681

theorem basketball_team_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a - 1)
  (h3 : 2 * a + 3 * b + x = 89) : 
  x = 29 :=
by
  sorry

end basketball_team_free_throws_l186_186681


namespace percentage_students_school_A_l186_186527

theorem percentage_students_school_A
  (A B : ℝ)
  (h1 : A + B = 100)
  (h2 : 0.30 * A + 0.40 * B = 34) :
  A = 60 :=
sorry

end percentage_students_school_A_l186_186527


namespace find_x_minus_y_l186_186580

theorem find_x_minus_y (x y n : ℤ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x > y) (h4 : n / 10 < 10 ∧ n / 10 ≥ 1) 
  (h5 : 2 * n = x + y) 
  (h6 : ∃ m : ℤ, m^2 = x * y ∧ m = (n % 10) * 10 + n / 10) 
  : x - y = 66 :=
sorry

end find_x_minus_y_l186_186580


namespace total_distance_covered_l186_186361

theorem total_distance_covered :
  ∀ (r j w total : ℝ),
    r = 40 →
    j = (3 / 5) * r →
    w = 5 * j →
    total = r + j + w →
    total = 184 := by
  sorry

end total_distance_covered_l186_186361


namespace unique_function_satisfying_conditions_l186_186250

noncomputable def f : (ℝ → ℝ) := sorry

axiom condition1 : f 1 = 1
axiom condition2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem unique_function_satisfying_conditions : ∀ x : ℝ, f x = x := sorry

end unique_function_satisfying_conditions_l186_186250


namespace square_perimeter_l186_186236

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : perimeter_of_square side_length = 20 := by
  sorry

end square_perimeter_l186_186236


namespace unique_positive_x_eq_3_l186_186841

theorem unique_positive_x_eq_3 (x : ℝ) (h_pos : 0 < x) (h_eq : x + 17 = 60 * (1 / x)) : x = 3 :=
by
  sorry

end unique_positive_x_eq_3_l186_186841


namespace exponent_property_l186_186822

variable {a : ℝ} {m n : ℕ}

theorem exponent_property (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 :=
sorry

end exponent_property_l186_186822


namespace employee_b_pay_l186_186191

theorem employee_b_pay (total_pay : ℝ) (ratio_ab : ℝ) (pay_b : ℝ) 
  (h1 : total_pay = 570)
  (h2 : ratio_ab = 1.5 * pay_b)
  (h3 : total_pay = ratio_ab + pay_b) :
  pay_b = 228 := 
sorry

end employee_b_pay_l186_186191


namespace scallops_cost_l186_186036

-- define the conditions
def scallops_per_pound : ℝ := 8
def cost_per_pound : ℝ := 24
def scallops_per_person : ℝ := 2
def number_of_people : ℝ := 8

-- the question
theorem scallops_cost : (scallops_per_person * number_of_people / scallops_per_pound) * cost_per_pound = 48 := by 
  sorry

end scallops_cost_l186_186036


namespace distinct_arrangements_l186_186843

-- Define the conditions: 7 books, 3 are identical
def total_books : ℕ := 7
def identical_books : ℕ := 3

-- Statement that the number of distinct arrangements is 840
theorem distinct_arrangements : (Nat.factorial total_books) / (Nat.factorial identical_books) = 840 := 
by
  sorry

end distinct_arrangements_l186_186843


namespace bricks_in_chimney_l186_186605

-- Define the conditions
def brenda_rate (h : ℕ) : ℚ := h / 8
def brandon_rate (h : ℕ) : ℚ := h / 12
def combined_rate (h : ℕ) : ℚ := (brenda_rate h + brandon_rate h) - 15
def total_bricks_in_6_hours (h : ℕ) : ℚ := 6 * combined_rate h

-- The proof statement
theorem bricks_in_chimney : ∃ h : ℕ, total_bricks_in_6_hours h = h ∧ h = 360 :=
by
  -- Proof goes here
  sorry

end bricks_in_chimney_l186_186605


namespace units_digit_17_pow_35_l186_186151

theorem units_digit_17_pow_35 : (17 ^ 35) % 10 = 3 := by
sorry

end units_digit_17_pow_35_l186_186151


namespace am_gm_inequality_l186_186757

variable {x y z : ℝ}

theorem am_gm_inequality (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) :
  (x + y + z) / 3 ≥ Real.sqrt (Real.sqrt (x * y) * Real.sqrt z) :=
by
  sorry

end am_gm_inequality_l186_186757


namespace width_of_rect_prism_l186_186052

theorem width_of_rect_prism (w : ℝ) 
  (h : ℝ := 8) (l : ℝ := 5) (diagonal : ℝ := 17) 
  (h_diag : l^2 + w^2 + h^2 = diagonal^2) :
  w = 10 * Real.sqrt 2 :=
by
  sorry

end width_of_rect_prism_l186_186052


namespace part1_min_value_part2_find_b_part3_range_b_div_a_l186_186849

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - abs (a*x - b)

-- Part (1)
theorem part1_min_value : f 1 1 1 = -5/4 :=
by 
  sorry

-- Part (2)
theorem part2_find_b (b : ℝ) (h : b ≥ 2) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b) (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) : 
  b = 2 :=
by 
  sorry

-- Part (3)
theorem part3_range_b_div_a (a b : ℝ) (h_distinct : (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x a b = 1 ∧ ∀ y : ℝ, 0 < y ∧ y < 2 ∧ f y a b = 1 ∧ x ≠ y)) : 
  1 < b / a ∧ b / a < 2 :=
by 
  sorry

end part1_min_value_part2_find_b_part3_range_b_div_a_l186_186849


namespace Dan_running_speed_is_10_l186_186904

noncomputable def running_speed
  (d : ℕ)
  (S : ℕ)
  (avg : ℚ) : ℚ :=
  let total_distance := 2 * d
  let total_time := d / (avg * 60) 
  let swim_time := d / S
  let run_time := total_time - swim_time
  total_distance / run_time

theorem Dan_running_speed_is_10
  (d S : ℕ)
  (avg : ℚ)
  (h1 : d = 4)
  (h2 : S = 6)
  (h3 : avg = 0.125) :
  running_speed d S (avg * 60) = 10 := by 
  sorry

end Dan_running_speed_is_10_l186_186904


namespace sufficient_condition_for_odd_l186_186713

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log (Real.sqrt (x^2 + a^2) - x)

theorem sufficient_condition_for_odd (a : ℝ) :
  (∀ x : ℝ, f 1 (-x) = -f 1 x) ∧
  (∀ x : ℝ, f (-1) (-x) = -f (-1) x) → 
  (a = 1 → ∀ x : ℝ, f a (-x) = -f a x) ∧ 
  (a ≠ 1 → ∃ x : ℝ, f a (-x) ≠ -f a x) :=
by
  sorry

end sufficient_condition_for_odd_l186_186713


namespace bruce_paid_amount_l186_186984

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

end bruce_paid_amount_l186_186984


namespace christina_payment_l186_186001

theorem christina_payment :
  let pay_flowers_per_flower := (8 : ℚ) / 3
  let pay_lawn_per_meter := (5 : ℚ) / 2
  let num_flowers := (9 : ℚ) / 4
  let area_lawn := (7 : ℚ) / 3
  let total_payment := pay_flowers_per_flower * num_flowers + pay_lawn_per_meter * area_lawn
  total_payment = 71 / 6 :=
by
  sorry

end christina_payment_l186_186001


namespace gcd_18_24_l186_186749

theorem gcd_18_24 : Int.gcd 18 24 = 6 :=
by
  sorry

end gcd_18_24_l186_186749


namespace sin_C_value_l186_186729

theorem sin_C_value (A B C : ℝ) (a b c : ℝ) 
  (h_a : a = 1) 
  (h_b : b = 1/2) 
  (h_cos_A : Real.cos A = (Real.sqrt 3) / 2) 
  (h_angles : A + B + C = Real.pi) 
  (h_sides : Real.sin A / a = Real.sin B / b) :
  Real.sin C = (Real.sqrt 15 + Real.sqrt 3) / 8 :=
by 
  sorry

end sin_C_value_l186_186729


namespace proof_l186_186177

-- Define proposition p
def p : Prop := ∀ x : ℝ, x < 0 → 2^x > x

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

theorem proof : p ∨ q :=
by
  have hp : p := 
    -- Here, you would provide the proof of p being true.
    sorry
  have hq : ¬ q :=
    -- Here, you would provide the proof of q being false, 
    -- i.e., showing that ∀ x, x^2 + x + 1 ≥ 0.
    sorry
  exact Or.inl hp

end proof_l186_186177


namespace find_rope_costs_l186_186500

theorem find_rope_costs (x y : ℕ) (h1 : 10 * x + 5 * y = 175) (h2 : 15 * x + 10 * y = 300) : x = 10 ∧ y = 15 :=
    sorry

end find_rope_costs_l186_186500


namespace cricket_scores_l186_186238

-- Define the conditions
variable (X : ℝ) (A B C D E average10 average6 : ℝ)
variable (matches10 matches6 : ℕ)

-- Set the given constants
axiom average_runs_10 : average10 = 38.9
axiom matches_10 : matches10 = 10
axiom average_runs_6 : average6 = 42
axiom matches_6 : matches6 = 6

-- Define the equations based on the conditions
axiom eq1 : X = average10 * matches10
axiom eq2 : A + B + C + D = X - (average6 * matches6)
axiom eq3 : E = (A + B + C + D) / 4

-- The target statement
theorem cricket_scores : X = 389 ∧ A + B + C + D = 137 ∧ E = 34.25 :=
  by
    sorry

end cricket_scores_l186_186238


namespace adam_final_amount_l186_186076

def initial_savings : ℝ := 1579.37
def money_received_monday : ℝ := 21.85
def money_received_tuesday : ℝ := 33.28
def money_spent_wednesday : ℝ := 87.41

def total_money_received : ℝ := money_received_monday + money_received_tuesday
def new_total_after_receiving : ℝ := initial_savings + total_money_received
def final_amount : ℝ := new_total_after_receiving - money_spent_wednesday

theorem adam_final_amount : final_amount = 1547.09 := by
  -- proof omitted
  sorry

end adam_final_amount_l186_186076


namespace abs_x_plus_1_plus_abs_x_minus_3_ge_a_l186_186945

theorem abs_x_plus_1_plus_abs_x_minus_3_ge_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end abs_x_plus_1_plus_abs_x_minus_3_ge_a_l186_186945


namespace dylans_mom_hotdogs_l186_186674

theorem dylans_mom_hotdogs (hotdogs_total : ℕ) (helens_mom_hotdogs : ℕ) (dylans_mom_hotdogs : ℕ) 
  (h1 : hotdogs_total = 480) (h2 : helens_mom_hotdogs = 101) (h3 : hotdogs_total = helens_mom_hotdogs + dylans_mom_hotdogs) :
dylans_mom_hotdogs = 379 :=
by
  sorry

end dylans_mom_hotdogs_l186_186674


namespace trey_uses_47_nails_l186_186215

variable (D : ℕ) -- total number of decorations
variable (nails thumbtacks sticky_strips : ℕ)

-- Conditions
def uses_nails := nails = (5 * D) / 8
def uses_thumbtacks := thumbtacks = (9 * D) / 80
def uses_sticky_strips := sticky_strips = 20
def total_decorations := (21 * D) / 80 = 20

-- Question: Prove that Trey uses 47 nails when the conditions hold
theorem trey_uses_47_nails (D : ℕ) (h1 : uses_nails D nails) (h2 : uses_thumbtacks D thumbtacks) (h3 : uses_sticky_strips sticky_strips) (h4 : total_decorations D) : nails = 47 :=  
by
  sorry

end trey_uses_47_nails_l186_186215


namespace computer_program_X_value_l186_186281

theorem computer_program_X_value : 
  ∃ (n : ℕ), (let X := 5 + 3 * (n - 1) 
               let S := (3 * n^2 + 7 * n) / 2 
               S ≥ 10500) ∧ X = 251 :=
sorry

end computer_program_X_value_l186_186281


namespace find_a_l186_186303

theorem find_a (a : ℕ) (h_pos : a > 0) (h_quadrant : 2 - a > 0) : a = 1 := by
  sorry

end find_a_l186_186303


namespace remaining_pictures_l186_186308

theorem remaining_pictures (first_book : ℕ) (second_book : ℕ) (third_book : ℕ) (colored_pictures : ℕ) :
  first_book = 23 → second_book = 32 → third_book = 45 → colored_pictures = 44 →
  (first_book + second_book + third_book - colored_pictures) = 56 :=
by
  sorry

end remaining_pictures_l186_186308


namespace sum_of_coefficients_l186_186051

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℕ) (h₁ : (1 + x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 := by
  sorry

end sum_of_coefficients_l186_186051


namespace find_abc_solutions_l186_186436

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end find_abc_solutions_l186_186436


namespace files_more_than_apps_l186_186193

-- Defining the initial conditions
def initial_apps : ℕ := 11
def initial_files : ℕ := 3

-- Defining the conditions after some changes
def apps_left : ℕ := 2
def files_left : ℕ := 24

-- Statement to prove
theorem files_more_than_apps : (files_left - apps_left) = 22 := 
by
  sorry

end files_more_than_apps_l186_186193


namespace B_subset_A_implies_range_m_l186_186746

variable {x m : ℝ}

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | -m < x ∧ x < m}

theorem B_subset_A_implies_range_m (m : ℝ) (h : B m ⊆ A) : m ≤ 1 := by
  sorry

end B_subset_A_implies_range_m_l186_186746


namespace sum_of_interior_angles_n_plus_4_l186_186181

    noncomputable def sum_of_interior_angles (sides : ℕ) : ℝ :=
      180 * (sides - 2)

    theorem sum_of_interior_angles_n_plus_4 (n : ℕ) (h : sum_of_interior_angles n = 2340) :
      sum_of_interior_angles (n + 4) = 3060 :=
    by
      sorry
    
end sum_of_interior_angles_n_plus_4_l186_186181


namespace product_of_fractions_l186_186637

theorem product_of_fractions (a b c d e f : ℚ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) 
  (h_d : d = 2) (h_e : e = 3) (h_f : f = 4) :
  (a / b) * (d / e) * (c / f) = 1 / 4 :=
by
  sorry

end product_of_fractions_l186_186637


namespace vacation_cost_l186_186730

theorem vacation_cost (C : ℝ) (h : C / 6 - C / 8 = 120) : C = 2880 :=
by
  sorry

end vacation_cost_l186_186730


namespace hyperbola_A_asymptote_l186_186295

-- Define the hyperbola and asymptote conditions
def hyperbola_A (x y : ℝ) : Prop := x^2 - (y^2 / 4) = 1
def asymptote_eq (y x : ℝ) : Prop := y = 2 * x ∨ y = -2 * x

-- Statement of the proof problem in Lean 4
theorem hyperbola_A_asymptote :
  ∀ (x y : ℝ), hyperbola_A x y → asymptote_eq y x :=
sorry

end hyperbola_A_asymptote_l186_186295


namespace one_div_lt_one_div_of_gt_l186_186661

theorem one_div_lt_one_div_of_gt {a b : ℝ} (hab : a > b) (hb0 : b > 0) : (1 / a) < (1 / b) :=
sorry

end one_div_lt_one_div_of_gt_l186_186661


namespace jason_earnings_l186_186480

theorem jason_earnings :
  let fred_initial := 49
  let jason_initial := 3
  let emily_initial := 25
  let fred_increase := 1.5 
  let jason_increase := 0.625 
  let emily_increase := 0.40 
  let fred_new := fred_initial * fred_increase
  let jason_new := jason_initial * (1 + jason_increase)
  let emily_new := emily_initial * (1 + emily_increase)
  fred_new = fred_initial * fred_increase ->
  jason_new = jason_initial * (1 + jason_increase) ->
  emily_new = emily_initial * (1 + emily_increase) ->
  jason_new - jason_initial == 1.875 :=
by
  intros
  sorry

end jason_earnings_l186_186480


namespace clock_hands_meeting_duration_l186_186382

noncomputable def angle_between_clock_hands (h m : ℝ) : ℝ :=
  abs ((30 * h + m / 2) - (6 * m) % 360)

theorem clock_hands_meeting_duration : 
  ∃ n m : ℝ, 0 <= n ∧ n < m ∧ m < 60 ∧ angle_between_clock_hands 5 n = 120 ∧ angle_between_clock_hands 5 m = 120 ∧ m - n = 44 :=
sorry

end clock_hands_meeting_duration_l186_186382


namespace min_deliveries_l186_186126

theorem min_deliveries (cost_per_delivery_income: ℕ) (cost_per_delivery_gas: ℕ) (van_cost: ℕ) (d: ℕ) : 
  (d * (cost_per_delivery_income - cost_per_delivery_gas) ≥ van_cost) ↔ (d ≥ van_cost / (cost_per_delivery_income - cost_per_delivery_gas)) :=
by
  sorry

def john_deliveries : ℕ := 7500 / (15 - 5)

example : john_deliveries = 750 :=
by
  sorry

end min_deliveries_l186_186126


namespace weight_of_five_bowling_balls_l186_186983

theorem weight_of_five_bowling_balls (b c : ℕ) (hb : 9 * b = 4 * c) (hc : c = 36) : 5 * b = 80 := by
  sorry

end weight_of_five_bowling_balls_l186_186983


namespace monotonic_increasing_on_interval_l186_186039

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - a * Real.log x

theorem monotonic_increasing_on_interval (a : ℝ) :
  (∀ x > 1, 2 * x - a / x ≥ 0) → a ≤ 2 :=
sorry

end monotonic_increasing_on_interval_l186_186039


namespace largest_c_value_l186_186292

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 3 * x + c

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, f x c = -2) ↔ c ≤ 1/4 := by
sorry

end largest_c_value_l186_186292


namespace solve_for_N_l186_186231

theorem solve_for_N (N : ℤ) (h1 : N < 0) (h2 : 2 * N * N + N = 15) : N = -3 :=
sorry

end solve_for_N_l186_186231


namespace ratio_unit_price_l186_186247

theorem ratio_unit_price (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let vX := 1.25 * v
  let pX := 0.85 * p
  (pX / vX) / (p / v) = 17 / 25 := by
{
  sorry
}

end ratio_unit_price_l186_186247


namespace max_min_values_of_f_l186_186874

-- Define the function f(x) and the conditions about its coefficients
def f (x : ℝ) (p q : ℝ) : ℝ := x^3 - p * x^2 - q * x

def intersects_x_axis_at_1 (p q : ℝ) : Prop :=
  f 1 p q = 0

-- Define the maximum and minimum values on the interval [-1, 1]
theorem max_min_values_of_f (p q : ℝ) 
  (h1 : f 1 p q = 0) :
  (p = 2) ∧ (q = -1) ∧ (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x 2 (-1) ≤ f (1/3) 2 (-1)) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-1) 2 (-1) ≤ f x 2 (-1)) :=
sorry

end max_min_values_of_f_l186_186874


namespace coats_from_high_schools_l186_186962

-- Define the total number of coats collected.
def total_coats_collected : ℕ := 9437

-- Define the number of coats collected from elementary schools.
def coats_from_elementary : ℕ := 2515

-- Goal: Prove that the number of coats collected from high schools is 6922.
theorem coats_from_high_schools : (total_coats_collected - coats_from_elementary) = 6922 := by
  sorry

end coats_from_high_schools_l186_186962


namespace bet_strategy_possible_l186_186518

def betting_possibility : Prop :=
  (1 / 6 + 1 / 2 + 1 / 9 + 1 / 8 <= 1)

theorem bet_strategy_possible : betting_possibility :=
by
  -- Proof is intentionally omitted
  sorry

end bet_strategy_possible_l186_186518


namespace binomial_coefficient_sum_l186_186354

theorem binomial_coefficient_sum {n : ℕ} (h : (1 : ℝ) + 1 = 128) : n = 7 :=
by
  sorry

end binomial_coefficient_sum_l186_186354


namespace calculate_k_l186_186094

theorem calculate_k (β : ℝ) (hβ : (Real.tan β + 1 / Real.tan β) ^ 2 = k + 1) : k = 1 := by
  sorry

end calculate_k_l186_186094


namespace tangent_function_property_l186_186356

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.tan (ϕ - x)

theorem tangent_function_property 
  (ϕ a : ℝ) 
  (h1 : π / 2 < ϕ) 
  (h2 : ϕ < 3 * π / 2) 
  (h3 : f 0 ϕ = 0) 
  (h4 : f (-a) ϕ = 1/2) : 
  f (a + π / 4) ϕ = -3 := by
  sorry

end tangent_function_property_l186_186356


namespace bob_needs_50_planks_l186_186148

-- Define the raised bed dimensions and requirements
structure RaisedBedDimensions where
  height : ℕ -- in feet
  width : ℕ  -- in feet
  length : ℕ -- in feet

def plank_length : ℕ := 8  -- length of each plank in feet
def plank_width : ℕ := 1  -- width of each plank in feet
def num_beds : ℕ := 10

def planks_needed (bed : RaisedBedDimensions) : ℕ :=
  let long_sides := 2  -- 2 long sides per bed
  let short_sides := 2 * (bed.width / plank_length)  -- 1/4 plank per short side if width is 2 feet
  let total_sides := long_sides + short_sides
  let stacked_sides := total_sides * (bed.height / plank_width)  -- stacked to match height
  stacked_sides

def raised_bed : RaisedBedDimensions := {height := 2, width := 2, length := 8}

theorem bob_needs_50_planks : planks_needed raised_bed * num_beds = 50 := by
  sorry

end bob_needs_50_planks_l186_186148


namespace average_weight_of_a_and_b_is_40_l186_186653

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := (A + B + C) / 3 = 42
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 40

-- Theorem statement
theorem average_weight_of_a_and_b_is_40 (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : 
    (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_is_40_l186_186653


namespace coefficient_x6_in_expansion_l186_186795

theorem coefficient_x6_in_expansion :
  (∃ c : ℕ, c = 81648 ∧ (3 : ℝ) ^ 6 * c * 2 ^ 2  = c * (3 : ℝ) ^ 6 * 4) :=
sorry

end coefficient_x6_in_expansion_l186_186795


namespace max_value_pq_qr_rs_sp_l186_186545

variable (p q r s : ℕ)

theorem max_value_pq_qr_rs_sp :
  (p = 1 ∨ p = 3 ∨ p = 5 ∨ p = 7) →
  (q = 1 ∨ q = 3 ∨ q = 5 ∨ q = 7) →
  (r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) →
  (s = 1 ∨ s = 3 ∨ s = 5 ∨ s = 7) →
  (p ≠ q) →
  (p ≠ r) →
  (p ≠ s) →
  (q ≠ r) →
  (q ≠ s) →
  (r ≠ s) →
  pq + qr + rs + sp ≤ 64 :=
sorry

end max_value_pq_qr_rs_sp_l186_186545


namespace quadratic_root_four_times_another_l186_186638

theorem quadratic_root_four_times_another (a : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + a * x + 2 * a = 0 ∧ x2 = 4 * x1) → a = 25 / 2 :=
by
  sorry

end quadratic_root_four_times_another_l186_186638


namespace magnitude_a_eq_3sqrt2_l186_186499

open Real

def a (x: ℝ) : ℝ × ℝ := (3, x)
def b : ℝ × ℝ := (-1, 1)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem magnitude_a_eq_3sqrt2 (x : ℝ) (h : perpendicular (a x) b) :
  ‖a 3‖ = 3 * sqrt 2 := by
  sorry

end magnitude_a_eq_3sqrt2_l186_186499


namespace average_mark_of_second_class_l186_186183

/-- 
There is a class of 30 students with an average mark of 40. 
Another class has 50 students with an unknown average mark. 
The average marks of all students combined is 65. 
Prove that the average mark of the second class is 80.
-/
theorem average_mark_of_second_class (x : ℝ) (h1 : 30 * 40 + 50 * x = 65 * (30 + 50)) : x = 80 := 
sorry

end average_mark_of_second_class_l186_186183


namespace original_price_l186_186584

theorem original_price (P : ℝ) (final_price : ℝ) (percent_increase : ℝ) (h1 : final_price = 450) (h2 : percent_increase = 0.50) : 
  P + percent_increase * P = final_price → P = 300 :=
by
  sorry

end original_price_l186_186584


namespace distance_to_fourth_buoy_l186_186647

theorem distance_to_fourth_buoy
  (buoy_interval_distance : ℕ)
  (total_distance_to_third_buoy : ℕ)
  (h : total_distance_to_third_buoy = buoy_interval_distance * 3) :
  (buoy_interval_distance * 4 = 96) :=
by
  sorry

end distance_to_fourth_buoy_l186_186647


namespace problem_statement_l186_186613

def f : ℝ → ℝ :=
  sorry

lemma even_function (x : ℝ) : f (-x) = f x :=
  sorry

lemma periodicity (x : ℝ) (hx : 0 ≤ x) : f (x + 2) = -f x :=
  sorry

lemma value_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 2) : f x = Real.log (x + 1) :=
  sorry

theorem problem_statement : f (-2001) + f 2012 = 1 :=
  sorry

end problem_statement_l186_186613


namespace expand_expression_l186_186447

variable (y : ℝ)

theorem expand_expression : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end expand_expression_l186_186447


namespace neg_p_l186_186508

-- Let's define the original proposition p
def p : Prop := ∃ x : ℝ, x ≥ 2 ∧ x^2 - 2 * x - 2 > 0

-- Now, we state the problem in Lean as requiring the proof of the negation of p
theorem neg_p : ¬p ↔ ∀ x : ℝ, x ≥ 2 → x^2 - 2 * x - 2 ≤ 0 :=
by
  sorry

end neg_p_l186_186508


namespace probability_red_then_green_l186_186829

-- Total number of balls and their representation
def total_balls : ℕ := 3
def red_balls : ℕ := 2
def green_balls : ℕ := 1

-- The total number of outcomes when drawing two balls with replacement
def total_outcomes : ℕ := total_balls * total_balls

-- The desired outcomes: drawing a red ball first and a green ball second
def desired_outcomes : ℕ := 2 -- (1,3) and (2,3)

-- Calculating the probability of drawing a red ball first and a green ball second
def probability_drawing_red_then_green : ℚ := desired_outcomes / total_outcomes

-- The theorem we need to prove
theorem probability_red_then_green :
  probability_drawing_red_then_green = 2 / 9 :=
by 
  sorry

end probability_red_then_green_l186_186829


namespace parallel_condition_l186_186923

theorem parallel_condition (a : ℝ) : (a = -1) ↔ (¬ (a = -1 ∧ a ≠ 1)) ∧ (¬ (a ≠ -1 ∧ a = 1)) :=
by
  sorry

end parallel_condition_l186_186923


namespace compare_magnitudes_l186_186302

theorem compare_magnitudes (a b c d e : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) (h₅ : e < 0) :
  (e / (a - c)) > (e / (b - d)) :=
  sorry

end compare_magnitudes_l186_186302


namespace A_inter_B_empty_iff_l186_186489

variable (m : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem A_inter_B_empty_iff : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by
  sorry

end A_inter_B_empty_iff_l186_186489


namespace cole_drive_time_l186_186662

noncomputable def T_work (D : ℝ) : ℝ := D / 75
noncomputable def T_home (D : ℝ) : ℝ := D / 105

theorem cole_drive_time (v1 v2 T : ℝ) (D : ℝ) 
  (h_v1 : v1 = 75) (h_v2 : v2 = 105) (h_T : T = 4)
  (h_round_trip : T_work D + T_home D = T) : 
  T_work D = 140 / 60 :=
sorry

end cole_drive_time_l186_186662


namespace no_green_ball_in_bag_l186_186914

theorem no_green_ball_in_bag (bag : Set String) (h : bag = {"red", "yellow", "white"}): ¬ ("green" ∈ bag) :=
by
  sorry

end no_green_ball_in_bag_l186_186914


namespace sum_of_odd_integers_from_13_to_53_l186_186809

-- Definition of the arithmetic series summing from 13 to 53 with common difference 2
def sum_of_arithmetic_series (a l d : ℕ) (n : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Main theorem
theorem sum_of_odd_integers_from_13_to_53 :
  sum_of_arithmetic_series 13 53 2 21 = 693 := 
sorry

end sum_of_odd_integers_from_13_to_53_l186_186809


namespace tom_tim_typing_ratio_l186_186145

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) :
  M / T = 5 :=
by
  -- Proof to be completed
  sorry

end tom_tim_typing_ratio_l186_186145


namespace length_PR_l186_186111

variable (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R]
variable {xPR xQR xsinR : ℝ}
variable (hypotenuse_opposite_ratio : xsinR = (3/5))
variable (sideQR : xQR = 9)
variable (rightAngle : ∀ (P Q R : Type), P ≠ Q → Q ∈ line_through Q R)

theorem length_PR : (∃ xPR : ℝ, xPR = 15) :=
by
  sorry

end length_PR_l186_186111


namespace average_male_students_score_l186_186947

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

end average_male_students_score_l186_186947


namespace find_N_l186_186342

/-- Given a row: [a, b, c, d, 2, f, g], 
    first column: [15, h, i, 14, j, k, l, 10],
    second column: [N, m, n, o, p, q, r, -21],
    where h=i+4 and i=j+4,
    b=15 and d = (2 - 15) / 3.
    The common difference c_n = -2.5.
    Prove N = -13.5.
-/
theorem find_N (a b c d f g h i j k l m n o p q r : ℝ) (N : ℝ) :
  b = 15 ∧ j = 14 ∧ l = 10 ∧ r = -21 ∧
  h = i + 4 ∧ i = j + 4 ∧
  c = (2 - 15) / 3 ∧
  g = b + 6 * c ∧
  N = g + 1 * (-2.5) →
  N = -13.5 :=
by
  intros h1
  sorry

end find_N_l186_186342


namespace find_y_l186_186799

theorem find_y (x y : ℝ) (h1 : x^2 - 4 * x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end find_y_l186_186799


namespace exists_pair_satisfying_system_l186_186228

theorem exists_pair_satisfying_system (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) ↔ m ≠ 1 :=
by
  sorry

end exists_pair_satisfying_system_l186_186228


namespace balloon_difference_l186_186967

theorem balloon_difference (x y : ℝ) (h1 : x = 2 * y - 3) (h2 : y = x / 4 + 1) : x - y = -2.5 :=
by 
  sorry

end balloon_difference_l186_186967


namespace arithmetic_seq_product_of_first_two_terms_l186_186837

theorem arithmetic_seq_product_of_first_two_terms
    (a d : ℤ)
    (h1 : a + 4 * d = 17)
    (h2 : d = 2) :
    (a * (a + d) = 99) := 
by
    -- Proof to be done
    sorry

end arithmetic_seq_product_of_first_two_terms_l186_186837


namespace inequality_conditions_l186_186979

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + B * C + C * A)) :=
by
  sorry

end inequality_conditions_l186_186979


namespace train_length_l186_186961

theorem train_length (speed_kmph : ℕ) (time_sec : ℕ) (length_meters : ℕ) : speed_kmph = 90 → time_sec = 4 → length_meters = 100 :=
by
  intros h₁ h₂
  have speed_mps : ℕ := speed_kmph * 1000 / 3600
  have speed_mps_val : speed_mps = 25 := sorry
  have distance : ℕ := speed_mps * time_sec
  have distance_val : distance = 100 := sorry
  exact sorry

end train_length_l186_186961


namespace missing_number_l186_186561

theorem missing_number 
  (a : ℕ) (b : ℕ) (x : ℕ)
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * x * b) 
  (h3 : b = 147) : 
  x = 3 :=
sorry

end missing_number_l186_186561


namespace set_complement_intersection_l186_186960

variable (U : Set ℕ) (M N : Set ℕ)

theorem set_complement_intersection
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {1, 4, 5})
  (hN : N = {2, 3}) :
  ((U \ N) ∩ M) = {1, 4, 5} :=
by
  sorry

end set_complement_intersection_l186_186960


namespace tom_reads_pages_l186_186676

-- Definition of conditions
def initial_speed : ℕ := 12   -- pages per hour
def speed_factor : ℕ := 3
def time_period : ℕ := 2     -- hours

-- Calculated speeds
def increased_speed (initial_speed speed_factor : ℕ) : ℕ := initial_speed * speed_factor
def total_pages (increased_speed time_period : ℕ) : ℕ := increased_speed * time_period

-- Theorem statement
theorem tom_reads_pages :
  total_pages (increased_speed initial_speed speed_factor) time_period = 72 :=
by
  -- Omitting proof as only theorem statement is required
  sorry

end tom_reads_pages_l186_186676


namespace Mika_stickers_l186_186176

theorem Mika_stickers
  (initial_stickers : ℕ)
  (bought_stickers : ℕ)
  (received_stickers : ℕ)
  (given_stickers : ℕ)
  (used_stickers : ℕ)
  (final_stickers : ℕ) :
  initial_stickers = 45 →
  bought_stickers = 53 →
  received_stickers = 35 →
  given_stickers = 19 →
  used_stickers = 86 →
  final_stickers = initial_stickers + bought_stickers + received_stickers - given_stickers - used_stickers →
  final_stickers = 28 :=
by
  intros
  sorry

end Mika_stickers_l186_186176


namespace value_of_b_minus_d_squared_l186_186260

theorem value_of_b_minus_d_squared
  (a b c d : ℤ)
  (h1 : a - b - c + d = 13)
  (h2 : a + b - c - d = 9) :
  (b - d) ^ 2 = 4 :=
sorry

end value_of_b_minus_d_squared_l186_186260


namespace Hulk_jump_more_than_500_l186_186871

theorem Hulk_jump_more_than_500 :
  ∀ n : ℕ, 2 * 3^(n - 1) > 500 → n = 7 :=
by
  sorry

end Hulk_jump_more_than_500_l186_186871


namespace line_tangent_to_ellipse_l186_186888

theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x : ℝ, x^2 + 4 * (m * x + 1)^2 = 1) → m^2 = 3 / 4 :=
by
  sorry

end line_tangent_to_ellipse_l186_186888


namespace students_without_favorite_subject_l186_186621

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end students_without_favorite_subject_l186_186621


namespace totalNutsInCar_l186_186987

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

end totalNutsInCar_l186_186987


namespace initial_candies_l186_186706

theorem initial_candies (x : ℕ) (h1 : x % 4 = 0) (h2 : x / 4 * 3 / 3 * 2 / 2 - 24 ≥ 6) (h3 : x / 4 * 3 / 3 * 2 / 2 - 24 ≤ 9) :
  x = 64 :=
sorry

end initial_candies_l186_186706


namespace find_actual_balance_l186_186759

-- Define the given conditions
def current_balance : ℝ := 90000
def rate : ℝ := 0.10

-- Define the target
def actual_balance_before_deduction (X : ℝ) : Prop :=
  (X * (1 - rate) = current_balance)

-- Statement of the theorem
theorem find_actual_balance : ∃ X : ℝ, actual_balance_before_deduction X :=
  sorry

end find_actual_balance_l186_186759


namespace problem_statement_l186_186435

def U : Set Int := {x | |x| < 5}
def A : Set Int := {-2, 1, 3, 4}
def B : Set Int := {0, 2, 4}

theorem problem_statement : (A ∩ (U \ B)) = {-2, 1, 3} := by
  sorry

end problem_statement_l186_186435


namespace vector_condition_l186_186811

open Real

def acute_angle (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.1 + a.2 * b.2) > 0

def not_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 ≠ 0

theorem vector_condition (x : ℝ) :
  acute_angle (2, x + 1) (x + 2, 6) ∧ not_collinear (2, x + 1) (x + 2, 6) ↔ x > -5/4 ∧ x ≠ 2 :=
by
  sorry

end vector_condition_l186_186811


namespace rod_mass_equilibrium_l186_186084

variable (g : ℝ) (m1 : ℝ) (l : ℝ) (S : ℝ)

-- Given conditions
axiom m1_value : m1 = 1
axiom l_value  : l = 0.5
axiom S_value  : S = 0.1

-- The goal is to find m2 such that the equilibrium condition holds
theorem rod_mass_equilibrium (m2 : ℝ) :
  (m1 * S = m2 * l) → m2 = 0.2 :=
by
  sorry

end rod_mass_equilibrium_l186_186084


namespace eccentricity_range_l186_186278

section EllipseEccentricity

variables {F1 F2 : ℝ × ℝ}
variable (M : ℝ × ℝ)

-- Conditions from a)
def is_orthogonal (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def is_inside_ellipse (F1 F2 M : ℝ × ℝ) : Prop :=
  is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) ∧ 
  -- other conditions to assert M is inside could be defined but this is unspecified
  true

-- Statement from c)
theorem eccentricity_range {a b c e : ℝ}
  (h : ∀ (M: ℝ × ℝ), is_orthogonal (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) → is_inside_ellipse F1 F2 M)
  (h1 : c^2 < a^2 - c^2)
  (h2 : e^2 = c^2 / a^2) :
  0 < e ∧ e < (Real.sqrt 2) / 2 := 
sorry

end EllipseEccentricity

end eccentricity_range_l186_186278


namespace abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l186_186549

theorem abs_x_minus_one_eq_one_minus_x_implies_x_le_one (x : ℝ) (h : |x - 1| = 1 - x) : x ≤ 1 :=
by
  sorry

end abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l186_186549


namespace area_of_field_l186_186437

theorem area_of_field (L W A : ℝ) (hL : L = 20) (hP : L + 2 * W = 25) : A = 50 :=
by
  sorry

end area_of_field_l186_186437


namespace sum_of_radii_tangent_circles_l186_186501

theorem sum_of_radii_tangent_circles :
  ∃ (r1 r2 : ℝ), 
  (∀ r, (r = (6 + 2*Real.sqrt 6) ∨ r = (6 - 2*Real.sqrt 6)) → (r = r1 ∨ r = r2)) ∧ 
  ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
  ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧ 
  (r1 + r2 = 12) :=
by
  sorry

end sum_of_radii_tangent_circles_l186_186501


namespace Sheila_attends_picnic_probability_l186_186155

theorem Sheila_attends_picnic_probability :
  let P_rain := 0.5
  let P_no_rain := 0.5
  let P_Sheila_goes_if_rain := 0.3
  let P_Sheila_goes_if_no_rain := 0.7
  let P_friend_agrees := 0.5
  (P_rain * P_Sheila_goes_if_rain + P_no_rain * P_Sheila_goes_if_no_rain) * P_friend_agrees = 0.25 := 
by
  sorry

end Sheila_attends_picnic_probability_l186_186155


namespace decreasing_interval_l186_186570

noncomputable def func (x : ℝ) := 2 * x^3 - 6 * x^2 + 11

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv func x < 0 :=
by
  sorry

end decreasing_interval_l186_186570


namespace find_b_l186_186611

theorem find_b (a b c : ℕ) (h1 : a + b + c = 99) (h2 : a + 6 = b - 6) (h3 : b - 6 = 5 * c) : b = 51 :=
sorry

end find_b_l186_186611


namespace elvins_fixed_charge_l186_186289

theorem elvins_fixed_charge (F C : ℝ) 
  (h1 : F + C = 40) 
  (h2 : F + 2 * C = 76) : F = 4 := 
by 
  sorry

end elvins_fixed_charge_l186_186289


namespace problem_fraction_eq_l186_186360

theorem problem_fraction_eq (x : ℝ) :
  (x * (3 / 4) * (1 / 2) * 5060 = 759.0000000000001) ↔ (x = 0.4) :=
by
  sorry

end problem_fraction_eq_l186_186360


namespace find_principal_amount_l186_186020

-- Given conditions
def SI : ℝ := 4016.25
def R : ℝ := 0.14
def T : ℕ := 5

-- Question: What is the principal amount P?
theorem find_principal_amount : (SI / (R * T) = 5737.5) :=
sorry

end find_principal_amount_l186_186020


namespace f_neg_one_l186_186528

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1/x else - (x^2 + 1/(-x))

theorem f_neg_one : f (-1) = -2 :=
by
  -- This is where the proof would go, but it is left as a sorry
  sorry

end f_neg_one_l186_186528


namespace exists_a_bc_l186_186816

-- Definitions & Conditions
def satisfies_conditions (a b c : ℤ) : Prop :=
  - (b + c) - 10 = a ∧ (b + 10) * (c + 10) = 1

-- Theorem Statement
theorem exists_a_bc : ∃ (a b c : ℤ), satisfies_conditions a b c := by
  -- Substitute the correct proof below
  sorry

end exists_a_bc_l186_186816


namespace new_total_weight_correct_l186_186922

-- Definitions based on the problem statement
variables (R S k : ℝ)
def ram_original_weight : ℝ := 2 * k
def shyam_original_weight : ℝ := 5 * k
def ram_new_weight : ℝ := 1.10 * (ram_original_weight k)
def shyam_new_weight : ℝ := 1.17 * (shyam_original_weight k)

-- Definition for total original weight and increased weight
def total_original_weight : ℝ := ram_original_weight k + shyam_original_weight k
def total_weight_increased : ℝ := 1.15 * total_original_weight k
def new_total_weight : ℝ := ram_new_weight k + shyam_new_weight k

-- The proof statement
theorem new_total_weight_correct :
  new_total_weight k = total_weight_increased k :=
by
  sorry

end new_total_weight_correct_l186_186922


namespace fraction_division_l186_186139

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end fraction_division_l186_186139


namespace y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l186_186400

def y := 96 + 144 + 200 + 300 + 600 + 720 + 4800

theorem y_is_multiple_of_4 : y % 4 = 0 := 
by sorry

theorem y_is_not_multiple_of_8 : y % 8 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_16 : y % 16 ≠ 0 := 
by sorry

theorem y_is_not_multiple_of_32 : y % 32 ≠ 0 := 
by sorry

end y_is_multiple_of_4_y_is_not_multiple_of_8_y_is_not_multiple_of_16_y_is_not_multiple_of_32_l186_186400


namespace probability_different_colors_l186_186147

theorem probability_different_colors :
  let total_chips := 16
  let prob_blue := (7 : ℚ) / total_chips
  let prob_yellow := (5 : ℚ) / total_chips
  let prob_red := (4 : ℚ) / total_chips
  let prob_blue_then_nonblue := prob_blue * ((prob_yellow + prob_red) : ℚ)
  let prob_yellow_then_non_yellow := prob_yellow * ((prob_blue + prob_red) : ℚ)
  let prob_red_then_non_red := prob_red * ((prob_blue + prob_yellow) : ℚ)
  let total_prob := prob_blue_then_nonblue + prob_yellow_then_non_yellow + prob_red_then_non_red
  total_prob = (83 : ℚ) / 128 := 
by
  sorry

end probability_different_colors_l186_186147


namespace infinitely_many_n_l186_186563

theorem infinitely_many_n (p : ℕ) (hp : p.Prime) (hp2 : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ n * 2^n + 1 :=
sorry

end infinitely_many_n_l186_186563


namespace sum_of_cubes_l186_186438

def cubic_eq (x : ℝ) : Prop := x^3 - 2 * x^2 + 3 * x - 4 = 0

variables (a b c : ℝ)

axiom a_root : cubic_eq a
axiom b_root : cubic_eq b
axiom c_root : cubic_eq c

axiom sum_roots : a + b + c = 2
axiom sum_products_roots : a * b + a * c + b * c = 3
axiom product_roots : a * b * c = 4

theorem sum_of_cubes : a^3 + b^3 + c^3 = 2 :=
by
  sorry

end sum_of_cubes_l186_186438


namespace xy_addition_equals_13_l186_186122

theorem xy_addition_equals_13 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt_15 : x < 15) (hy_lt_15 : y < 15) (hxy : x + y + x * y = 49) : x + y = 13 :=
by
  sorry

end xy_addition_equals_13_l186_186122


namespace roots_reciprocal_sum_eq_three_halves_l186_186645

theorem roots_reciprocal_sum_eq_three_halves
  {a b : ℝ}
  (h1 : a^2 - 6 * a + 4 = 0)
  (h2 : b^2 - 6 * b + 4 = 0)
  (h_roots : a ≠ b) :
  1/a + 1/b = 3/2 := by
  sorry

end roots_reciprocal_sum_eq_three_halves_l186_186645


namespace Brandon_can_still_apply_l186_186719

-- Definitions based on the given conditions
def total_businesses : ℕ := 72
def fired_businesses : ℕ := total_businesses / 2
def quit_businesses : ℕ := total_businesses / 3
def businesses_restricted : ℕ := fired_businesses + quit_businesses

-- The final proof statement
theorem Brandon_can_still_apply : total_businesses - businesses_restricted = 12 :=
by
  -- Note: Proof is omitted; replace sorry with detailed proof in practice.
  sorry

end Brandon_can_still_apply_l186_186719


namespace number_of_people_in_room_l186_186879

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end number_of_people_in_room_l186_186879


namespace original_selling_price_l186_186788

theorem original_selling_price:
  ∀ (P : ℝ), (1.17 * P - 1.10 * P = 56) → (P > 0) → 1.10 * P = 880 :=
by
  intro P h₁ h₂
  sorry

end original_selling_price_l186_186788


namespace eq_sin_intersect_16_solutions_l186_186099

theorem eq_sin_intersect_16_solutions :
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 50 ∧ (x / 50 = Real.sin x)) ∧ (S.card = 16) :=
  sorry

end eq_sin_intersect_16_solutions_l186_186099


namespace conference_hall_initial_people_l186_186032

theorem conference_hall_initial_people (x : ℕ)  
  (h1 : 3 ∣ x) 
  (h2 : 4 ∣ (2 * x / 3))
  (h3 : (x / 2) = 27) : 
  x = 54 := 
by 
  sorry

end conference_hall_initial_people_l186_186032


namespace unique_non_zero_b_for_unique_x_solution_l186_186670

theorem unique_non_zero_b_for_unique_x_solution (c : ℝ) (hc : c ≠ 0) :
  c = 3 / 2 ↔ ∃! b : ℝ, b ≠ 0 ∧ ∃ x : ℝ, (x^2 + (b + 3 / b) * x + c = 0) ∧ 
  ∀ x1 x2 : ℝ, (x1^2 + (b + 3 / b) * x1 + c = 0) ∧ (x2^2 + (b + 3 / b) * x2 + c = 0) → x1 = x2 :=
sorry

end unique_non_zero_b_for_unique_x_solution_l186_186670


namespace tangent_line_at_P_eq_2x_l186_186868

noncomputable def tangentLineEq (f : ℝ → ℝ) (P : ℝ × ℝ) : ℝ → ℝ :=
  let slope := deriv f P.1
  fun x => slope * (x - P.1) + P.2

theorem tangent_line_at_P_eq_2x : 
  ∀ (f : ℝ → ℝ) (x y : ℝ),
    f x = x^2 + 1 → 
    (x = 1) → (y = 2) →
    tangentLineEq f (x, y) x = 2 * x :=
by
  intros f x y f_eq hx hy
  sorry

end tangent_line_at_P_eq_2x_l186_186868


namespace solution_set_inequalities_l186_186921

theorem solution_set_inequalities (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 2 * x) / 3 > x - 1) → (x ≤ 1) :=
by
  intros h
  sorry

end solution_set_inequalities_l186_186921


namespace correct_proposition_l186_186562

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_proposition :
  ¬ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  ¬ (∀ h : ℝ, f (-Real.pi / 6 + h) = f (-Real.pi / 6 - h)) ∧
  (∀ h : ℝ, f (-5 * Real.pi / 12 + h) = f (-5 * Real.pi / 12 - h)) :=
by sorry

end correct_proposition_l186_186562


namespace complex_number_problem_l186_186451

theorem complex_number_problem (a b : ℝ) (i : ℂ) (hi : i * i = -1) 
  (h : (a - 2 * i) * i = b - i) : a + b * i = -1 + 2 * i :=
by {
  -- provide proof here
  sorry
}

end complex_number_problem_l186_186451


namespace belle_rawhide_bones_per_evening_l186_186227

theorem belle_rawhide_bones_per_evening 
  (cost_rawhide_bone : ℝ)
  (cost_dog_biscuit : ℝ)
  (num_dog_biscuits_per_evening : ℕ)
  (total_weekly_cost : ℝ)
  (days_per_week : ℕ)
  (rawhide_bones_per_evening : ℕ)
  (h1 : cost_rawhide_bone = 1)
  (h2 : cost_dog_biscuit = 0.25)
  (h3 : num_dog_biscuits_per_evening = 4)
  (h4 : total_weekly_cost = 21)
  (h5 : days_per_week = 7)
  (h6 : rawhide_bones_per_evening * cost_rawhide_bone * (days_per_week : ℝ) = total_weekly_cost - num_dog_biscuits_per_evening * cost_dog_biscuit * (days_per_week : ℝ)) :
  rawhide_bones_per_evening = 2 := 
sorry

end belle_rawhide_bones_per_evening_l186_186227


namespace chess_tournament_game_count_l186_186906

theorem chess_tournament_game_count (n : ℕ) (h1 : ∃ n, ∀ i j, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ i ≠ j → ∃ games_between, games_between = n ∧ games_between * (Nat.choose 6 2) = 30) : n = 2 :=
by
  sorry

end chess_tournament_game_count_l186_186906


namespace expression_factorization_l186_186576

variables (a b c : ℝ)

theorem expression_factorization :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3)
  = (a - b) * (b - c) * (c - a) * (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
sorry

end expression_factorization_l186_186576


namespace temperature_on_friday_l186_186488

theorem temperature_on_friday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 42) : 
  F = 34 :=
by
  sorry

end temperature_on_friday_l186_186488


namespace completing_the_square_l186_186133

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end completing_the_square_l186_186133


namespace proof_eq1_proof_eq2_l186_186866

variable (x : ℝ)

-- Proof problem for Equation (1)
theorem proof_eq1 (h : (1 - x) / 3 - 2 = x / 6) : x = -10 / 3 := sorry

-- Proof problem for Equation (2)
theorem proof_eq2 (h : (x + 1) / 0.25 - (x - 2) / 0.5 = 5) : x = -3 / 2 := sorry

end proof_eq1_proof_eq2_l186_186866


namespace initial_amount_l186_186946

theorem initial_amount (X : ℚ) (F : ℚ) :
  (∀ (X F : ℚ), F = X * (3/4)^3 → F = 37 → X = 37 * 64 / 27) :=
by
  sorry

end initial_amount_l186_186946


namespace johns_speed_l186_186237

theorem johns_speed :
  ∀ (v : ℝ), 
    (∀ (t : ℝ), 24 = 30 * (t + 4 / 60) → 24 = v * (t - 8 / 60)) → 
    v = 40 :=
by
  intros
  sorry

end johns_speed_l186_186237


namespace max_value_expression_l186_186046

variable (x y z : ℝ)

theorem max_value_expression (h : x^2 + y^2 + z^2 = 4) :
  (2*x - y)^2 + (2*y - z)^2 + (2*z - x)^2 ≤ 28 :=
sorry

end max_value_expression_l186_186046


namespace total_turtles_l186_186219

variable (Kristen_turtles Kris_turtles Trey_turtles : ℕ)

-- Kristen has 12 turtles
def Kristen_turtles_count : Kristen_turtles = 12 := sorry

-- Kris has 1/4 the number of turtles Kristen has
def Kris_turtles_count (hK : Kristen_turtles = 12) : Kris_turtles = Kristen_turtles / 4 := sorry

-- Trey has 5 times as many turtles as Kris
def Trey_turtles_count (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) : Trey_turtles = 5 * Kris_turtles := sorry

-- Total number of turtles
theorem total_turtles (hK : Kristen_turtles = 12) (hKr : Kris_turtles = Kristen_turtles / 4) 
  (hT : Trey_turtles = 5 * Kris_turtles) : Kristen_turtles + Kris_turtles + Trey_turtles = 30 := sorry

end total_turtles_l186_186219


namespace kelly_games_left_l186_186974

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

end kelly_games_left_l186_186974


namespace stuffed_animal_cost_l186_186511

theorem stuffed_animal_cost
  (M S A C : ℝ)
  (h1 : M = 3 * S)
  (h2 : M = (1/2) * A)
  (h3 : C = (1/2) * A)
  (h4 : C = 2 * S)
  (h5 : M = 6) :
  A = 8 :=
by
  sorry

end stuffed_animal_cost_l186_186511


namespace function_not_strictly_decreasing_l186_186311

theorem function_not_strictly_decreasing (b : ℝ)
  (h : ¬ ∀ x1 x2 : ℝ, x1 < x2 → (-x1^3 + b*x1^2 - (2*b + 3)*x1 + 2 - b > -x2^3 + b*x2^2 - (2*b + 3)*x2 + 2 - b)) : 
  b < -1 ∨ b > 3 :=
by
  sorry

end function_not_strictly_decreasing_l186_186311


namespace math_problem_l186_186585

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, a = b * b

theorem math_problem (a m : ℕ) (h1: m = 2992) (h2: a = m^2 + m^2 * (m+1)^2 + (m+1)^2) : is_perfect_square a :=
  sorry

end math_problem_l186_186585


namespace bird_probability_l186_186753

def uniform_probability (segment_count bird_count : ℕ) : ℚ :=
  if bird_count = segment_count then
    1 / (segment_count ^ bird_count)
  else
    0

theorem bird_probability :
  let wire_length := 10
  let birds := 10
  let distance := 1
  let segments := wire_length / distance
  segments = birds ->
  uniform_probability segments birds = 1 / (10 ^ 10) := by
  intros
  sorry

end bird_probability_l186_186753


namespace min_value_expression_l186_186357

theorem min_value_expression (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (min ((1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + (x * y * z)) 2) = 2 :=
by 
  sorry

end min_value_expression_l186_186357


namespace smallest_nat_div3_and_5_rem1_l186_186331

theorem smallest_nat_div3_and_5_rem1 : ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ ∀ M : ℕ, M > 1 ∧ (M % 3 = 1) ∧ (M % 5 = 1) → N ≤ M := 
by
  sorry

end smallest_nat_div3_and_5_rem1_l186_186331


namespace hilary_ears_per_stalk_l186_186973

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

end hilary_ears_per_stalk_l186_186973


namespace a_n_is_perfect_square_l186_186291

theorem a_n_is_perfect_square :
  ∀ (a b : ℕ → ℤ), a 0 = 1 → b 0 = 0 →
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∀ n, ∃ k : ℤ, a n = k * k :=
by
  sorry

end a_n_is_perfect_square_l186_186291


namespace number_of_distinct_intersections_of_curves_l186_186025

theorem number_of_distinct_intersections_of_curves (x y : ℝ) :
  (∀ x y, x^2 - 4*y^2 = 4) ∧ (∀ x y, 4*x^2 + y^2 = 16) → 
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), 
    ((x1, y1) ≠ (x2, y2)) ∧
    ((x1^2 - 4*y1^2 = 4) ∧ (4*x1^2 + y1^2 = 16)) ∧
    ((x2^2 - 4*y2^2 = 4) ∧ (4*x2^2 + y2^2 = 16)) ∧
    ∀ (x' y' : ℝ), 
      ((x'^2 - 4*y'^2 = 4) ∧ (4*x'^2 + y'^2 = 16)) → 
      ((x', y') = (x1, y1) ∨ (x', y') = (x2, y2)) := 
sorry

end number_of_distinct_intersections_of_curves_l186_186025


namespace intersection_points_l186_186791

theorem intersection_points (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  (∃ x1 x2, 0 ≤ x1 ∧ x1 ≤ 2 * Real.pi ∧ 
   0 ≤ x2 ∧ x2 ≤ 2 * Real.pi ∧ 
   x1 ≠ x2 ∧ 
   1 + Real.sin x1 = 3 / 2 ∧ 
   1 + Real.sin x2 = 3 / 2 ) ∧ 
  (∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 1 + Real.sin x = 3 / 2) → 
   (x = x1 ∨ x = x2)) :=
sorry

end intersection_points_l186_186791


namespace min_disks_required_for_files_l186_186804

theorem min_disks_required_for_files :
  ∀ (number_of_files : ℕ)
    (files_0_9MB : ℕ)
    (files_0_6MB : ℕ)
    (disk_capacity_MB : ℝ)
    (file_size_0_9MB : ℝ)
    (file_size_0_6MB : ℝ)
    (file_size_0_45MB : ℝ),
  number_of_files = 40 →
  files_0_9MB = 5 →
  files_0_6MB = 15 →
  disk_capacity_MB = 1.44 →
  file_size_0_9MB = 0.9 →
  file_size_0_6MB = 0.6 →
  file_size_0_45MB = 0.45 →
  ∃ (min_disks : ℕ), min_disks = 16 :=
by
  sorry

end min_disks_required_for_files_l186_186804


namespace direct_proportion_function_l186_186393

theorem direct_proportion_function (k : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = k * x) (h2 : f 3 = 6) : ∀ x, f x = 2 * x := by
  sorry

end direct_proportion_function_l186_186393


namespace price_per_postcard_is_correct_l186_186725

noncomputable def initial_postcards : ℕ := 18
noncomputable def sold_postcards : ℕ := initial_postcards / 2
noncomputable def price_per_postcard_sold : ℕ := 15
noncomputable def total_earned : ℕ := sold_postcards * price_per_postcard_sold
noncomputable def total_postcards_after : ℕ := 36
noncomputable def remaining_original_postcards : ℕ := initial_postcards - sold_postcards
noncomputable def new_postcards_bought : ℕ := total_postcards_after - remaining_original_postcards
noncomputable def price_per_new_postcard : ℕ := total_earned / new_postcards_bought

theorem price_per_postcard_is_correct:
  price_per_new_postcard = 5 :=
by
  sorry

end price_per_postcard_is_correct_l186_186725


namespace smallest_nat_number_l186_186232

theorem smallest_nat_number (x : ℕ) (h1 : 5 ∣ x) (h2 : 7 ∣ x) (h3 : x % 3 = 1) : x = 70 :=
sorry

end smallest_nat_number_l186_186232


namespace radio_cost_price_l186_186734

theorem radio_cost_price (SP : ℝ) (Loss : ℝ) (CP : ℝ) (h1 : SP = 1110) (h2 : Loss = 0.26) (h3 : SP = CP * (1 - Loss)) : CP = 1500 :=
  by
  sorry

end radio_cost_price_l186_186734


namespace determine_k_l186_186272

theorem determine_k (k : ℤ) : (∀ n : ℤ, gcd (4 * n + 1) (k * n + 1) = 1) ↔ 
  (∃ m : ℕ, k = 4 + 2 ^ m ∨ k = 4 - 2 ^ m) :=
by
  sorry

end determine_k_l186_186272


namespace solid_color_marble_percentage_l186_186625

theorem solid_color_marble_percentage (solid striped dotted swirl red blue green yellow purple : ℝ)
  (h_solid: solid = 0.7) (h_striped: striped = 0.1) (h_dotted: dotted = 0.1) (h_swirl: swirl = 0.1)
  (h_red: red = 0.25) (h_blue: blue = 0.25) (h_green: green = 0.2) (h_yellow: yellow = 0.15) (h_purple: purple = 0.15) :
  solid * (red + blue + green) * 100 = 49 :=
by
  sorry

end solid_color_marble_percentage_l186_186625


namespace min_value_5x_plus_6y_l186_186573

theorem min_value_5x_plus_6y (x y : ℝ) (h : 3 * x ^ 2 + 3 * y ^ 2 = 20 * x + 10 * y + 10) : 
  ∃ x y, (5 * x + 6 * y = 122) :=
by
  sorry

end min_value_5x_plus_6y_l186_186573


namespace calculate_expression_l186_186246

theorem calculate_expression (a b : ℝ) : (a - b) * (a + b) * (a^2 - b^2) = a^4 - 2 * a^2 * b^2 + b^4 := 
by
  sorry

end calculate_expression_l186_186246


namespace slipper_cost_l186_186601

def original_price : ℝ := 50.00
def discount_rate : ℝ := 0.10
def embroidery_rate_per_shoe : ℝ := 5.50
def number_of_shoes : ℕ := 2
def shipping_cost : ℝ := 10.00

theorem slipper_cost :
  (original_price - original_price * discount_rate) + 
  (embroidery_rate_per_shoe * number_of_shoes) + 
  shipping_cost = 66.00 :=
by sorry

end slipper_cost_l186_186601


namespace temperature_rise_per_hour_l186_186409

-- Define the conditions
variables (x : ℕ) -- temperature rise per hour

-- Assume the given conditions
axiom power_outage : (3 : ℕ) * x = (6 : ℕ) * 4

-- State the proposition
theorem temperature_rise_per_hour : x = 8 :=
sorry

end temperature_rise_per_hour_l186_186409


namespace students_in_class_l186_186883

theorem students_in_class (n S : ℕ) 
    (h1 : S = 15 * n)
    (h2 : (S + 56) / (n + 1) = 16) : n = 40 :=
by
  sorry

end students_in_class_l186_186883


namespace machine_a_produces_18_sprockets_per_hour_l186_186526

theorem machine_a_produces_18_sprockets_per_hour :
  ∃ (A : ℝ), (∀ (B C : ℝ),
  B = 1.10 * A ∧
  B = 1.20 * C ∧
  990 / A = 990 / B + 10 ∧
  990 / C = 990 / A - 5) →
  A = 18 :=
by { sorry }

end machine_a_produces_18_sprockets_per_hour_l186_186526


namespace markup_percentage_l186_186128

theorem markup_percentage {C : ℝ} (hC0: 0 < C) (h1: 0 < 1.125 * C) : 
  ∃ (x : ℝ), 0.75 * (1.20 * C * (1 + x / 100)) = 1.125 * C ∧ x = 25 := 
by
  have h2 : 1.20 = (6 / 5 : ℝ) := by norm_num
  have h3 : 0.75 = (3 / 4 : ℝ) := by norm_num
  sorry

end markup_percentage_l186_186128


namespace exists_circle_with_exactly_n_integer_points_l186_186317

noncomputable def circle_with_n_integer_points (n : ℕ) : Prop :=
  ∃ r : ℤ, ∃ (xs ys : List ℤ), 
    xs.length = n ∧ ys.length = n ∧
    ∀ (x y : ℤ), x ∈ xs → y ∈ ys → x^2 + y^2 = r^2

theorem exists_circle_with_exactly_n_integer_points (n : ℕ) : 
  circle_with_n_integer_points n := 
sorry

end exists_circle_with_exactly_n_integer_points_l186_186317


namespace theresa_sons_count_l186_186346

theorem theresa_sons_count (total_meat_left : ℕ) (meat_per_plate : ℕ) (frac_left : ℚ) (s : ℕ) :
  total_meat_left = meat_per_plate ∧ meat_per_plate * frac_left * s = 3 → s = 9 :=
by sorry

end theresa_sons_count_l186_186346


namespace exists_sum_of_two_squares_l186_186085

theorem exists_sum_of_two_squares (n : ℕ) (h₁ : n > 10000) : 
  ∃ m : ℕ, (∃ a b : ℕ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * Real.sqrt n := 
sorry

end exists_sum_of_two_squares_l186_186085


namespace bags_of_hammers_to_load_l186_186916

noncomputable def total_crate_capacity := 15 * 20
noncomputable def weight_of_nails := 4 * 5
noncomputable def weight_of_planks := 10 * 30
noncomputable def weight_to_be_left_out := 80
noncomputable def effective_capacity := total_crate_capacity - weight_to_be_left_out
noncomputable def weight_of_loaded_planks := 220

theorem bags_of_hammers_to_load : (effective_capacity - weight_of_nails - weight_of_loaded_planks = 0) :=
by
  sorry

end bags_of_hammers_to_load_l186_186916


namespace quadratic_two_distinct_real_roots_l186_186035

theorem quadratic_two_distinct_real_roots:
  ∃ (α β : ℝ), α ≠ β ∧ (∀ x : ℝ, x * (x - 2) = x - 2 ↔ x = α ∨ x = β) :=
by
  sorry

end quadratic_two_distinct_real_roots_l186_186035


namespace buns_cost_eq_1_50_l186_186908

noncomputable def meat_cost : ℝ := 2 * 3.50
noncomputable def tomato_cost : ℝ := 1.5 * 2.00
noncomputable def pickles_cost : ℝ := 2.50 - 1.00
noncomputable def lettuce_cost : ℝ := 1.00
noncomputable def total_other_items_cost : ℝ := meat_cost + tomato_cost + pickles_cost + lettuce_cost
noncomputable def total_amount_spent : ℝ := 20.00 - 6.00
noncomputable def buns_cost : ℝ := total_amount_spent - total_other_items_cost

theorem buns_cost_eq_1_50 : buns_cost = 1.50 := by
  sorry

end buns_cost_eq_1_50_l186_186908


namespace matrix_equation_l186_186402

open Matrix

-- Define matrix N and the identity matrix I
def N : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 8], ![-4, -2]]
def I : Matrix (Fin 2) (Fin 2) ℤ := 1

-- Scalars p and q
def p : ℤ := 1
def q : ℤ := -26

-- Theorem statement
theorem matrix_equation :
  N * N = p • N + q • I :=
  by
    sorry

end matrix_equation_l186_186402


namespace total_spending_in_CAD_proof_l186_186463

-- Define Jayda's spending
def Jayda_spending_stall1 : ℤ := 400
def Jayda_spending_stall2 : ℤ := 120
def Jayda_spending_stall3 : ℤ := 250

-- Define the factor by which Aitana spends more
def Aitana_factor : ℚ := 2 / 5

-- Define the sales tax rate
def sales_tax_rate : ℚ := 0.10

-- Define the exchange rate from USD to CAD
def exchange_rate : ℚ := 1.25

-- Calculate Jayda's total spending in USD before tax
def Jayda_total_spending : ℤ := Jayda_spending_stall1 + Jayda_spending_stall2 + Jayda_spending_stall3

-- Calculate Aitana's spending at each stall
def Aitana_spending_stall1 : ℚ := Jayda_spending_stall1 + (Aitana_factor * Jayda_spending_stall1)
def Aitana_spending_stall2 : ℚ := Jayda_spending_stall2 + (Aitana_factor * Jayda_spending_stall2)
def Aitana_spending_stall3 : ℚ := Jayda_spending_stall3 + (Aitana_factor * Jayda_spending_stall3)

-- Calculate Aitana's total spending in USD before tax
def Aitana_total_spending : ℚ := Aitana_spending_stall1 + Aitana_spending_stall2 + Aitana_spending_stall3

-- Calculate the combined total spending in USD before tax
def combined_total_spending_before_tax : ℚ := Jayda_total_spending + Aitana_total_spending

-- Calculate the sales tax amount
def sales_tax : ℚ := sales_tax_rate * combined_total_spending_before_tax

-- Calculate the total spending including sales tax
def total_spending_including_tax : ℚ := combined_total_spending_before_tax + sales_tax

-- Convert the total spending to Canadian dollars
def total_spending_in_CAD : ℚ := total_spending_including_tax * exchange_rate

-- The theorem to be proven
theorem total_spending_in_CAD_proof : total_spending_in_CAD = 2541 := sorry

end total_spending_in_CAD_proof_l186_186463


namespace tom_already_has_4_pounds_of_noodles_l186_186780

-- Define the conditions
def beef : ℕ := 10
def noodle_multiplier : ℕ := 2
def packages : ℕ := 8
def weight_per_package : ℕ := 2

-- Define the total noodles needed
def total_noodles_needed : ℕ := noodle_multiplier * beef

-- Define the total noodles bought
def total_noodles_bought : ℕ := packages * weight_per_package

-- Define the already owned noodles
def already_owned_noodles : ℕ := total_noodles_needed - total_noodles_bought

-- State the theorem to prove
theorem tom_already_has_4_pounds_of_noodles :
  already_owned_noodles = 4 :=
  sorry

end tom_already_has_4_pounds_of_noodles_l186_186780


namespace number_of_terms_in_sequence_l186_186184

theorem number_of_terms_in_sequence :
  ∃ n : ℕ, (1 + 4 * (n - 1) = 2025) ∧ n = 507 := by
  sorry

end number_of_terms_in_sequence_l186_186184


namespace find_n_infinitely_many_squares_find_n_no_squares_l186_186457

def is_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P (n k l m : ℕ) : ℕ := n^k + n^l + n^m

theorem find_n_infinitely_many_squares :
  ∃ k, ∃ l, ∃ m, is_square (P 7 k l m) :=
by
  sorry

theorem find_n_no_squares :
  ∀ (k l m : ℕ) n, n ∈ [5, 6] → ¬is_square (P n k l m) :=
by
  sorry

end find_n_infinitely_many_squares_find_n_no_squares_l186_186457


namespace unique_solution_l186_186566

def is_prime (n : ℕ) : Prop := Nat.Prime n

def eq_triple (m p q : ℕ) : Prop :=
  2 ^ m * p ^ 2 + 1 = q ^ 5

theorem unique_solution (m p q : ℕ) (h1 : m > 0) (h2 : is_prime p) (h3 : is_prime q) :
  eq_triple m p q ↔ (m, p, q) = (1, 11, 3) := by
  sorry

end unique_solution_l186_186566


namespace quadratic_roots_diff_by_2_l186_186372

theorem quadratic_roots_diff_by_2 (q : ℝ) (hq : 0 < q) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 - r2 = 2 ∨ r2 - r1 = 2) ∧ r1 ^ 2 + (2 * q - 1) * r1 + q = 0 ∧ r2 ^ 2 + (2 * q - 1) * r2 + q = 0) ↔
  q = 1 + (Real.sqrt 7) / 2 :=
sorry

end quadratic_roots_diff_by_2_l186_186372


namespace additional_discount_A_is_8_l186_186280

-- Define the problem conditions
def full_price_A : ℝ := 125
def full_price_B : ℝ := 130
def discount_B : ℝ := 0.10
def price_difference : ℝ := 2

-- Define the unknown additional discount of store A
def discount_A (x : ℝ) : Prop :=
  full_price_A - (full_price_A * (x / 100)) = (full_price_B - (full_price_B * discount_B)) - price_difference

-- Theorem stating that the additional discount offered by store A is 8%
theorem additional_discount_A_is_8 : discount_A 8 :=
by
  -- Proof can be filled in here
  sorry

end additional_discount_A_is_8_l186_186280


namespace slices_with_both_toppings_l186_186005

theorem slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices : ℕ)
    (all_have_topping : total_slices = 24)
    (pepperoni_cond: pepperoni_slices = 14)
    (mushroom_cond: mushroom_slices = 16)
    (at_least_one_topping : total_slices = pepperoni_slices + mushroom_slices - slices_with_both):
    slices_with_both = 6 := by
  sorry

end slices_with_both_toppings_l186_186005


namespace partial_fraction_decomposition_l186_186751

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ), 
  (∀ x : ℝ, x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10 ≠ 0 →
    (x^2 - 23) /
    (x^4 - 3 * x^3 - 7 * x^2 + 15 * x - 10) = 
    A / (x - 1) + B / (x + 2) + C / (x - 2)) →
  (A = 44 / 21 ∧ B = -5 / 2 ∧ C = -5 / 6 → A * B * C = 275 / 63)
  := by
  intros A B C h₁ h₂
  sorry

end partial_fraction_decomposition_l186_186751


namespace temperature_increase_per_century_l186_186017

def total_temperature_change_over_1600_years : ℕ := 64
def years_in_a_century : ℕ := 100
def years_overall : ℕ := 1600

theorem temperature_increase_per_century :
  total_temperature_change_over_1600_years / (years_overall / years_in_a_century) = 4 := by
  sorry

end temperature_increase_per_century_l186_186017


namespace find_d_l186_186525

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
∀ n, a n = a₁ + d * (n - 1)

theorem find_d
  (a : ℕ → ℝ)
  (a₁ d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h₁ : a₁ = 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4)
  (h_d_neq_zero : d ≠ 0):
  d = 1 :=
sorry

end find_d_l186_186525


namespace sculpture_cost_in_cny_l186_186802

-- Define the equivalence rates
def usd_to_nad : ℝ := 8
def usd_to_cny : ℝ := 8

-- Define the cost of the sculpture in Namibian dollars
def sculpture_cost_nad : ℝ := 160

-- Theorem: Given the conversion rates, the sculpture cost in Chinese yuan is 160
theorem sculpture_cost_in_cny : (sculpture_cost_nad / usd_to_nad) * usd_to_cny = 160 :=
by sorry

end sculpture_cost_in_cny_l186_186802


namespace solve_for_x_l186_186350

theorem solve_for_x (x : ℤ) : 27 - 5 = 4 + x → x = 18 :=
by
  intro h
  sorry

end solve_for_x_l186_186350


namespace correct_number_of_three_digit_numbers_l186_186618

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end correct_number_of_three_digit_numbers_l186_186618


namespace problem_I_problem_II_l186_186574

/-- Proof problem I: Given f(x) = |x - 1|, prove that the inequality f(x) ≥ 4 - |x - 1| implies x ≥ 3 or x ≤ -1 -/
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (h2 : f x ≥ 4 - |x - 1|) : x ≥ 3 ∨ x ≤ -1 :=
  sorry

/-- Proof problem II: Given f(x) = |x - 1| and 1/m + 1/(2*n) = 1 (m > 0, n > 0), prove that the minimum value of mn is 2 -/
theorem problem_II (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h2 : 1/m + 1/(2*n) = 1) : m*n ≥ 2 :=
  sorry

end problem_I_problem_II_l186_186574


namespace maximum_value_problem_l186_186370

theorem maximum_value_problem (x : ℝ) (h : 0 < x ∧ x < 4/3) : ∃ M, M = (4 / 3) ∧ ∀ y, 0 < y ∧ y < 4/3 → x * (4 - 3 * x) ≤ M :=
sorry

end maximum_value_problem_l186_186370


namespace zero_clever_numbers_l186_186633

def isZeroClever (n : Nat) : Prop :=
  ∃ a b c : Nat, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  n = 1000 * a + 10 * b + c ∧
  n = 9 * (100 * a + 10 * b + c)

theorem zero_clever_numbers :
  ∀ n : Nat, isZeroClever n → n = 2025 ∨ n = 4050 ∨ n = 6075 :=
by
  -- Proof to be provided
  sorry

end zero_clever_numbers_l186_186633


namespace part_I_part_II_l186_186118

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x) / (x + 1)

theorem part_I (a : ℝ) : (∀ x, f a 0 ≤ f a x) → a = 1 := by
  sorry

theorem part_II (a : ℝ) : (∀ x > 0, f a x > 0) → a ≤ 1 := by
  sorry

end part_I_part_II_l186_186118


namespace totalCats_l186_186422

def whiteCats : Nat := 2
def blackCats : Nat := 10
def grayCats : Nat := 3

theorem totalCats : whiteCats + blackCats + grayCats = 15 := by
  sorry

end totalCats_l186_186422


namespace ratio_of_profits_is_2_to_3_l186_186282

-- Conditions
def Praveen_initial_investment := 3220
def Praveen_investment_duration := 12
def Hari_initial_investment := 8280
def Hari_investment_duration := 7

-- Effective capital contributions
def Praveen_effective_capital : ℕ := Praveen_initial_investment * Praveen_investment_duration
def Hari_effective_capital : ℕ := Hari_initial_investment * Hari_investment_duration

-- Theorem statement to be proven
theorem ratio_of_profits_is_2_to_3 : (Praveen_effective_capital : ℚ) / Hari_effective_capital = 2 / 3 :=
by sorry

end ratio_of_profits_is_2_to_3_l186_186282


namespace circle_center_radius_l186_186104

theorem circle_center_radius : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = (2, 0) ∧ radius = 2 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4 * x = 0 ↔ (x - 2)^2 + y^2 = 4 :=
by
  sorry

end circle_center_radius_l186_186104


namespace parallel_vectors_sin_cos_l186_186427

theorem parallel_vectors_sin_cos (θ : ℝ) (a := (6, 3)) (b := (Real.sin θ, Real.cos θ))
  (h : (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2)) :
  Real.sin (2 * θ) - 2 * (Real.cos θ)^2 = 2 / 5 :=
by
  sorry

end parallel_vectors_sin_cos_l186_186427


namespace product_equality_l186_186656

theorem product_equality : (2.05 * 4.1 = 20.5 * 0.41) :=
by
  sorry

end product_equality_l186_186656


namespace sin_solution_set_l186_186972

open Real

theorem sin_solution_set (x : ℝ) : 
  (3 * sin x = 1 + cos (2 * x)) ↔ ∃ k : ℤ, x = k * π + (-1) ^ k * (π / 6) :=
by
  sorry

end sin_solution_set_l186_186972


namespace students_receiving_B_lee_l186_186665

def num_students_receiving_B (students_kipling: ℕ) (B_kipling: ℕ) (students_lee: ℕ) : ℕ :=
  let ratio := (B_kipling * students_lee) / students_kipling
  ratio

theorem students_receiving_B_lee (students_kipling B_kipling students_lee : ℕ) 
  (h : B_kipling = 8 ∧ students_kipling = 12 ∧ students_lee = 30) :
  num_students_receiving_B students_kipling B_kipling students_lee = 20 :=
by
  sorry

end students_receiving_B_lee_l186_186665


namespace fraction_addition_l186_186100

theorem fraction_addition : 
  (2 : ℚ) / 5 + (3 : ℚ) / 8 + 1 = 71 / 40 :=
by
  sorry

end fraction_addition_l186_186100


namespace total_loads_l186_186963

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

end total_loads_l186_186963


namespace probability_of_next_satisfied_customer_l186_186530

noncomputable def probability_of_satisfied_customer : ℝ :=
  let p := (0.8 : ℝ)
  let q := (0.15 : ℝ)
  let neg_reviews := (60 : ℝ)
  let pos_reviews := (20 : ℝ)
  p / (p + q) * (q / (q + p))

theorem probability_of_next_satisfied_customer :
  probability_of_satisfied_customer = 0.64 :=
sorry

end probability_of_next_satisfied_customer_l186_186530


namespace g_five_l186_186902

def g (x : ℝ) : ℝ := 4 * x + 2

theorem g_five : g 5 = 22 := by
  sorry

end g_five_l186_186902


namespace trajectory_of_T_l186_186486

-- Define coordinates for points A, T, and M
variables {x x0 y y0 : ℝ}
def A (x0: ℝ) (y0: ℝ) := (x0, y0)
def T (x: ℝ) (y: ℝ) := (x, y)
def M : ℝ × ℝ := (-2, 0)

-- Conditions
def curve (x : ℝ) (y : ℝ) := 4 * x^2 - y + 1 = 0
def vector_condition (x x0 y y0 : ℝ) := (x - x0, y - y0) = 2 * (-2 - x, -y)

theorem trajectory_of_T (x y x0 y0 : ℝ) (hA : curve x0 y0) (hV : vector_condition x x0 y y0) :
  4 * (3 * x + 4)^2 - 3 * y + 1 = 0 :=
by
  sorry

end trajectory_of_T_l186_186486


namespace min_value_of_expression_l186_186444

theorem min_value_of_expression (α β : ℝ) (h : α + β = π / 2) : 
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 65 := 
sorry

end min_value_of_expression_l186_186444


namespace max_stickers_one_student_l186_186056

def total_students : ℕ := 25
def mean_stickers : ℕ := 4
def total_stickers := total_students * mean_stickers
def minimum_stickers_per_student : ℕ := 1
def minimum_stickers_taken_by_24_students := (total_students - 1) * minimum_stickers_per_student

theorem max_stickers_one_student : 
  total_stickers - minimum_stickers_taken_by_24_students = 76 := by
  sorry

end max_stickers_one_student_l186_186056


namespace seven_digit_number_l186_186442

theorem seven_digit_number (a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℕ)
(h1 : a_1 + a_2 = 9)
(h2 : a_2 + a_3 = 7)
(h3 : a_3 + a_4 = 9)
(h4 : a_4 + a_5 = 2)
(h5 : a_5 + a_6 = 8)
(h6 : a_6 + a_7 = 11)
(h_digits : ∀ (i : ℕ), i ∈ [a_1, a_2, a_3, a_4, a_5, a_6, a_7] → i < 10) :
a_1 = 9 ∧ a_2 = 0 ∧ a_3 = 7 ∧ a_4 = 2 ∧ a_5 = 0 ∧ a_6 = 8 ∧ a_7 = 3 :=
by sorry

end seven_digit_number_l186_186442


namespace triangle_inequality_l186_186329

theorem triangle_inequality (a b c Δ : ℝ) (h_Δ: Δ = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt (3) * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 :=
by
  sorry

end triangle_inequality_l186_186329


namespace tyler_meal_choices_l186_186951

-- Define the total number of different meals Tyler can choose given the conditions.
theorem tyler_meal_choices : 
    (3 * (Nat.choose 5 3) * 4 * 4 = 480) := 
by
    -- Using the built-in combination function and the fact that meat, dessert, and drink choices are directly multiplied.
    sorry

end tyler_meal_choices_l186_186951


namespace right_triangle_condition_l186_186740

theorem right_triangle_condition (a b c : ℝ) :
  (a^3 + b^3 + c^3 = a*b*(a + b) - b*c*(b + c) + a*c*(a + c)) ↔ (a^2 = b^2 + c^2) ∨ (b^2 = a^2 + c^2) ∨ (c^2 = a^2 + b^2) :=
by
  sorry

end right_triangle_condition_l186_186740


namespace complete_square_solution_l186_186591

theorem complete_square_solution :
  ∀ x : ℝ, ∃ p q : ℝ, (5 * x^2 - 30 * x - 45 = 0) → ((x + p) ^ 2 = q) ∧ (p + q = 15) :=
by
  sorry

end complete_square_solution_l186_186591


namespace average_value_of_T_l186_186426

noncomputable def expected_value_T (B G : ℕ) : ℚ :=
  let total_pairs := 19
  let prob_bg := (B / (B + G)) * (G / (B + G))
  2 * total_pairs * prob_bg

theorem average_value_of_T 
  (B G : ℕ) (hB : B = 8) (hG : G = 12) : 
  expected_value_T B G = 9 :=
by
  rw [expected_value_T, hB, hG]
  norm_num
  sorry

end average_value_of_T_l186_186426


namespace find_a_range_l186_186034

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 1 then (a + 3) * x - 5 else 2 * a / x

theorem find_a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) → -2 ≤ a ∧ a < 0 :=
by
  sorry

end find_a_range_l186_186034


namespace total_mission_days_l186_186502

variable (initial_days_first_mission : ℝ := 5)
variable (percentage_longer : ℝ := 0.60)
variable (days_second_mission : ℝ := 3)

theorem total_mission_days : 
  let days_first_mission_extra := initial_days_first_mission * percentage_longer
  let total_days_first_mission := initial_days_first_mission + days_first_mission_extra
  (total_days_first_mission + days_second_mission) = 11 := by
  sorry

end total_mission_days_l186_186502


namespace larger_gate_width_is_10_l186_186060

-- Define the conditions as constants
def garden_length : ℝ := 225
def garden_width : ℝ := 125
def small_gate_width : ℝ := 3
def total_fencing_length : ℝ := 687

-- Define the perimeter function for a rectangle
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

-- Define the width of the larger gate
def large_gate_width : ℝ :=
  let total_perimeter := perimeter garden_length garden_width
  let remaining_fencing := total_perimeter - total_fencing_length
  remaining_fencing - small_gate_width

-- State the theorem
theorem larger_gate_width_is_10 : large_gate_width = 10 := by
  -- skipping proof part
  sorry

end larger_gate_width_is_10_l186_186060


namespace brenda_more_than_jeff_l186_186415

def emma_amount : ℕ := 8
def daya_amount : ℕ := emma_amount + (emma_amount * 25 / 100)
def jeff_amount : ℕ := (2 / 5) * daya_amount
def brenda_amount : ℕ := 8

theorem brenda_more_than_jeff :
  brenda_amount - jeff_amount = 4 :=
sorry

end brenda_more_than_jeff_l186_186415


namespace sum_of_squares_219_l186_186672

theorem sum_of_squares_219 :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 219 ∧ a + b + c = 21 := by
  sorry

end sum_of_squares_219_l186_186672


namespace complex_fraction_value_l186_186328

theorem complex_fraction_value :
  1 + (1 / (2 + (1 / (2 + 2)))) = 13 / 9 :=
by
  sorry

end complex_fraction_value_l186_186328


namespace ratio_length_to_breadth_l186_186738

theorem ratio_length_to_breadth (b l : ℕ) (A : ℕ) (h1 : b = 30) (h2 : A = 2700) (h3 : A = l * b) :
  l / b = 3 :=
by sorry

end ratio_length_to_breadth_l186_186738


namespace binary_to_decimal_l186_186543

-- Define the binary number 10011_2
def binary_10011 : ℕ := bit0 (bit1 (bit1 (bit0 (bit1 0))))

-- Define the expected decimal value
def decimal_19 : ℕ := 19

-- State the theorem to convert binary 10011 to decimal
theorem binary_to_decimal :
  binary_10011 = decimal_19 :=
sorry

end binary_to_decimal_l186_186543


namespace solve_for_y_l186_186196

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l186_186196


namespace merchant_articles_l186_186093

theorem merchant_articles (N CP SP : ℝ) (h1 : N * CP = 16 * SP) (h2 : SP = CP * 1.0625) (h3 : CP ≠ 0) : N = 17 :=
by
  sorry

end merchant_articles_l186_186093


namespace average_second_pair_l186_186136

theorem average_second_pair 
  (avg_six : ℝ) (avg_first_pair : ℝ) (avg_third_pair : ℝ) (avg_second_pair : ℝ) 
  (h1 : avg_six = 3.95) 
  (h2 : avg_first_pair = 4.2) 
  (h3 : avg_third_pair = 3.8000000000000007) : 
  avg_second_pair = 3.85 :=
by
  sorry

end average_second_pair_l186_186136


namespace multiplication_with_mixed_number_l186_186318

-- Define mixed numbers as rational numbers for proper calculation
def mixed_to_rational (whole : ℕ) (num : ℕ) (den : ℕ) : ℚ :=
  whole + num / den

-- 7 * (9 + 2/5)
def lhs : ℚ := 7 * mixed_to_rational 9 2 5

-- 65 + 4/5
def rhs : ℚ := mixed_to_rational 65 4 5

theorem multiplication_with_mixed_number : lhs = rhs := by
  sorry

end multiplication_with_mixed_number_l186_186318


namespace mackenzie_new_disks_l186_186175

noncomputable def price_new (U N : ℝ) : Prop := 6 * N + 2 * U = 127.92

noncomputable def disks_mackenzie_buys (U N x : ℝ) : Prop := x * N + 8 * U = 133.89

theorem mackenzie_new_disks (U N x : ℝ) (h1 : U = 9.99) (h2 : price_new U N) (h3 : disks_mackenzie_buys U N x) :
  x = 3 :=
by
  sorry

end mackenzie_new_disks_l186_186175


namespace counting_indistinguishable_boxes_l186_186424

def distinguishable_balls := 5
def indistinguishable_boxes := 3

theorem counting_indistinguishable_boxes :
  (∃ ways : ℕ, ways = 66) := sorry

end counting_indistinguishable_boxes_l186_186424


namespace nathan_correct_answers_l186_186492

theorem nathan_correct_answers (c w : ℤ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 := 
by sorry

end nathan_correct_answers_l186_186492


namespace train_length_correct_l186_186340

noncomputable def length_of_train (speed_km_per_hr : ℝ) (platform_length_m : ℝ) (time_s : ℝ) : ℝ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  let total_distance := speed_m_per_s * time_s
  total_distance - platform_length_m

theorem train_length_correct :
  length_of_train 55 520 43.196544276457885 = 140 :=
by
  unfold length_of_train
  -- The conversion and calculations would be verified here
  sorry

end train_length_correct_l186_186340


namespace Binkie_gemstones_l186_186917

-- Define the number of gemstones each cat has
variables (F S B : ℕ)

-- Conditions based on the problem statement
axiom Spaatz_has_one : S = 1
axiom Spaatz_equation : S = F / 2 - 2
axiom Binkie_equation : B = 4 * F

-- Theorem statement
theorem Binkie_gemstones : B = 24 :=
by
  -- Proof will be inserted here
  sorry

end Binkie_gemstones_l186_186917


namespace brick_wall_l186_186790

theorem brick_wall (x : ℕ) 
  (h1 : x / 9 * 9 = x)
  (h2 : x / 10 * 10 = x)
  (h3 : 5 * (x / 9 + x / 10 - 10) = x) :
  x = 900 := 
sorry

end brick_wall_l186_186790


namespace minimum_value_x_plus_y_l186_186351

theorem minimum_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y * (x - y)^2 = 1) : x + y ≥ 2 :=
sorry

end minimum_value_x_plus_y_l186_186351


namespace calculate_expression_l186_186630

theorem calculate_expression : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end calculate_expression_l186_186630


namespace min_prime_factor_sum_l186_186149

theorem min_prime_factor_sum (x y a b c d : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : 5 * x^7 = 13 * y^11)
  (h4 : x = 13^6 * 5^7) (h5 : a = 13) (h6 : b = 5) (h7 : c = 6) (h8 : d = 7) : 
  a + b + c + d = 31 :=
by
  sorry

end min_prime_factor_sum_l186_186149


namespace area_of_triangle_ABC_circumcenter_of_triangle_ABC_l186_186284

structure Point where
  x : ℚ
  y : ℚ

def A : Point := ⟨2, 1⟩
def B : Point := ⟨4, 7⟩
def C : Point := ⟨8, 3⟩

def triangle_area (A B C : Point) : ℚ := by
  -- area calculation will be filled here
  sorry

def circumcenter (A B C : Point) : Point := by
  -- circumcenter calculation will be filled here
  sorry

theorem area_of_triangle_ABC : triangle_area A B C = 16 :=
  sorry

theorem circumcenter_of_triangle_ABC : circumcenter A B C = ⟨9/2, 7/2⟩ :=
  sorry

end area_of_triangle_ABC_circumcenter_of_triangle_ABC_l186_186284


namespace valid_assignment_statement_l186_186065

theorem valid_assignment_statement (S a : ℕ) : (S = a + 1) ∧ ¬(a + 1 = S) ∧ ¬(S - 1 = a) ∧ ¬(S - a = 1) := by
  sorry

end valid_assignment_statement_l186_186065


namespace trig_identity_l186_186105

theorem trig_identity : 
  Real.sin (15 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) + 
  Real.cos (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 :=
by 
  sorry

end trig_identity_l186_186105


namespace initial_balls_in_bag_l186_186931

theorem initial_balls_in_bag (n : ℕ) 
  (h_add_white : ∀ x : ℕ, x = n + 1)
  (h_probability : (5 / 8) = 0.625):
  n = 7 :=
sorry

end initial_balls_in_bag_l186_186931


namespace shuffleboard_total_games_l186_186715

theorem shuffleboard_total_games
    (jerry_wins : ℕ)
    (dave_wins : ℕ)
    (ken_wins : ℕ)
    (h1 : jerry_wins = 7)
    (h2 : dave_wins = jerry_wins + 3)
    (h3 : ken_wins = dave_wins + 5) :
    jerry_wins + dave_wins + ken_wins = 32 := 
by
  sorry

end shuffleboard_total_games_l186_186715


namespace percentage_of_annual_decrease_is_10_l186_186453

-- Define the present population and future population
def P_present : ℕ := 500
def P_future : ℕ := 450 

-- Calculate the percentage decrease
def percentage_decrease (P_present P_future : ℕ) : ℕ :=
  ((P_present - P_future) * 100) / P_present

-- Lean statement to prove the percentage decrease is 10%
theorem percentage_of_annual_decrease_is_10 :
  percentage_decrease P_present P_future = 10 :=
by
  unfold percentage_decrease
  sorry

end percentage_of_annual_decrease_is_10_l186_186453


namespace arithmetic_sequence_a8_l186_186971

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 15 = 90) :
  a 8 = 6 :=
by
  sorry

end arithmetic_sequence_a8_l186_186971


namespace diagonal_rectangle_l186_186782

theorem diagonal_rectangle (l w : ℝ) (hl : l = 20 * Real.sqrt 5) (hw : w = 10 * Real.sqrt 3) :
    Real.sqrt (l^2 + w^2) = 10 * Real.sqrt 23 :=
by
  sorry

end diagonal_rectangle_l186_186782


namespace Tom_total_yearly_intake_l186_186910

def soda_weekday := 5 * 12
def water_weekday := 64
def juice_weekday := 3 * 8
def sports_drink_weekday := 2 * 16

def total_weekday_intake := soda_weekday + water_weekday + juice_weekday + sports_drink_weekday

def soda_weekend_holiday := 5 * 12
def water_weekend_holiday := 64
def juice_weekend_holiday := 3 * 8
def sports_drink_weekend_holiday := 1 * 16
def fruit_smoothie_weekend_holiday := 32

def total_weekend_holiday_intake := soda_weekend_holiday + water_weekend_holiday + juice_weekend_holiday + sports_drink_weekend_holiday + fruit_smoothie_weekend_holiday

def weekdays := 260
def weekend_days := 104
def holidays := 1

def total_yearly_intake := (weekdays * total_weekday_intake) + (weekend_days * total_weekend_holiday_intake) + (holidays * total_weekend_holiday_intake)

theorem Tom_total_yearly_intake :
  total_yearly_intake = 67380 := by
  sorry

end Tom_total_yearly_intake_l186_186910


namespace polynomial_roots_expression_l186_186109

theorem polynomial_roots_expression 
  (a b α β γ δ : ℝ)
  (h1 : α^2 - a*α - 1 = 0)
  (h2 : β^2 - a*β - 1 = 0)
  (h3 : γ^2 - b*γ - 1 = 0)
  (h4 : δ^2 - b*δ - 1 = 0) :
  ((α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2) = (b^2 - a^2)^2 :=
sorry

end polynomial_roots_expression_l186_186109


namespace susie_remaining_money_l186_186909

noncomputable def calculate_remaining_money : Float :=
  let weekday_hours := 4.0
  let weekday_rate := 12.0
  let weekdays := 5.0
  let weekend_hours := 2.5
  let weekend_rate := 15.0
  let weekends := 2.0
  let total_weekday_earnings := weekday_hours * weekday_rate * weekdays
  let total_weekend_earnings := weekend_hours * weekend_rate * weekends
  let total_earnings := total_weekday_earnings + total_weekend_earnings
  let spent_makeup := 3 / 8 * total_earnings
  let remaining_after_makeup := total_earnings - spent_makeup
  let spent_skincare := 2 / 5 * remaining_after_makeup
  let remaining_after_skincare := remaining_after_makeup - spent_skincare
  let spent_cellphone := 1 / 6 * remaining_after_skincare
  let final_remaining := remaining_after_skincare - spent_cellphone
  final_remaining

theorem susie_remaining_money : calculate_remaining_money = 98.4375 := by
  sorry

end susie_remaining_money_l186_186909


namespace coefficient_x2_in_expansion_l186_186341

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the problem: Given (2x + 1)^5, find the coefficient of x^2 term
theorem coefficient_x2_in_expansion : 
  binomial 5 3 * (2 ^ 2) = 40 := by 
  sorry

end coefficient_x2_in_expansion_l186_186341


namespace sum_of_ages_is_26_l186_186266

def Yoongi_aunt_age := 38
def Yoongi_age := Yoongi_aunt_age - 23
def Hoseok_age := Yoongi_age - 4
def sum_of_ages := Yoongi_age + Hoseok_age

theorem sum_of_ages_is_26 : sum_of_ages = 26 :=
by
  sorry

end sum_of_ages_is_26_l186_186266


namespace machine_probabilities_at_least_one_first_class_component_l186_186390

theorem machine_probabilities : 
  (∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3) 
:=
sorry

theorem at_least_one_first_class_component : 
  ∃ (PA PB PC : ℝ), 
  PA * (1 - PB) = 1/4 ∧ 
  PB * (1 - PC) = 1/12 ∧ 
  PA * PC = 2/9 ∧ 
  PA = 1/3 ∧ 
  PB = 1/4 ∧ 
  PC = 2/3 ∧ 
  1 - (1 - PA) * (1 - PB) * (1 - PC) = 5/6
:=
sorry

end machine_probabilities_at_least_one_first_class_component_l186_186390


namespace eval_F_at_4_f_5_l186_186348

def f (a : ℤ) : ℤ := 3 * a - 6
def F (a : ℤ) (b : ℤ) : ℤ := 2 * b ^ 2 + 3 * a

theorem eval_F_at_4_f_5 : F 4 (f 5) = 174 := by
  sorry

end eval_F_at_4_f_5_l186_186348


namespace negation_of_sum_of_squares_l186_186741

variables (a b : ℝ)

theorem negation_of_sum_of_squares:
  ¬(a^2 + b^2 = 0) → (a ≠ 0 ∨ b ≠ 0) := 
by
  sorry

end negation_of_sum_of_squares_l186_186741


namespace bus_sarah_probability_l186_186098

-- Define the probability of Sarah arriving while the bus is still there
theorem bus_sarah_probability :
  let total_minutes := 60
  let bus_waiting_time := 15
  let total_area := (total_minutes * total_minutes : ℕ)
  let triangle_area := (1 / 2 : ℝ) * 45 * 15
  let rectangle_area := 15 * 15
  let shaded_area := triangle_area + rectangle_area
  (shaded_area / total_area : ℝ) = (5 / 32 : ℝ) :=
by
  sorry

end bus_sarah_probability_l186_186098


namespace graphs_intersect_at_one_point_l186_186796

theorem graphs_intersect_at_one_point (m : ℝ) (e := Real.exp 1) :
  (∀ f g : ℝ → ℝ,
    (∀ x, f x = x + Real.log x - 2 / e) ∧ (∀ x, g x = m / x) →
    ∃! x, f x = g x) ↔ (m ≥ 0 ∨ m = - (e + 1) / (e ^ 2)) :=
by sorry

end graphs_intersect_at_one_point_l186_186796


namespace product_of_ab_l186_186761

theorem product_of_ab (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 :=
sorry

end product_of_ab_l186_186761


namespace inequality_division_by_two_l186_186369

theorem inequality_division_by_two (x y : ℝ) (h : x > y) : (x / 2) > (y / 2) := 
sorry

end inequality_division_by_two_l186_186369


namespace fraction_of_pizza_covered_by_pepperoni_l186_186160

theorem fraction_of_pizza_covered_by_pepperoni :
  ∀ (d_pizza d_pepperoni : ℝ) (n_pepperoni : ℕ) (overlap_fraction : ℝ),
  d_pizza = 16 ∧ d_pepperoni = d_pizza / 8 ∧ n_pepperoni = 32 ∧ overlap_fraction = 0.25 →
  (π * d_pepperoni^2 / 4 * (1 - overlap_fraction) * n_pepperoni) / (π * (d_pizza / 2)^2) = 3 / 8 :=
by
  intro d_pizza d_pepperoni n_pepperoni overlap_fraction
  intro h
  sorry

end fraction_of_pizza_covered_by_pepperoni_l186_186160


namespace positive_integer_solutions_l186_186952

theorem positive_integer_solutions (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) :
  1 + 2^x + 2^(2*x+1) = y^n ↔ 
  (x = 4 ∧ y = 23 ∧ n = 2) ∨ (∃ t : ℕ, 0 < t ∧ x = t ∧ y = 1 + 2^t + 2^(2*t+1) ∧ n = 1) :=
sorry

end positive_integer_solutions_l186_186952


namespace solve_for_s_l186_186254

theorem solve_for_s (s : ℝ) :
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 14) = (s^2 - 3 * s - 18) / (s^2 - 2 * s - 24) →
  s = -5 / 4 :=
by {
  sorry
}

end solve_for_s_l186_186254


namespace congruent_semicircles_span_diameter_l186_186919

theorem congruent_semicircles_span_diameter (N : ℕ) (r : ℝ) 
  (h1 : 2 * N * r = 2 * (N * r)) 
  (h2 : (N * (π * r^2 / 2)) / ((N^2 * (π * r^2 / 2)) - (N * (π * r^2 / 2))) = 1/4) 
  : N = 5 :=
by
  sorry

end congruent_semicircles_span_diameter_l186_186919


namespace horner_rule_example_l186_186889

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_rule_example : f 2 = 62 := by
  sorry

end horner_rule_example_l186_186889


namespace johns_payment_ratio_is_one_half_l186_186728

-- Define the initial conditions
def num_members := 4
def join_fee_per_person := 4000
def monthly_cost_per_person := 1000
def johns_payment_per_year := 32000

-- Calculate total cost for joining
def total_join_fee := num_members * join_fee_per_person

-- Calculate total monthly cost for a year
def total_monthly_cost := num_members * monthly_cost_per_person * 12

-- Calculate total cost for the first year
def total_cost_for_year := total_join_fee + total_monthly_cost

-- The ratio of John's payment to the total cost
def johns_ratio := johns_payment_per_year / total_cost_for_year

-- The statement to be proved
theorem johns_payment_ratio_is_one_half : johns_ratio = (1 / 2) := by sorry

end johns_payment_ratio_is_one_half_l186_186728


namespace investment_total_amount_l186_186512

noncomputable def compoundedInvestment (principal : ℝ) (rate : ℝ) (tax : ℝ) (years : ℕ) : ℝ :=
let yearlyNetInterest := principal * rate * (1 - tax)
let rec calculate (year : ℕ) (accumulated : ℝ) : ℝ :=
  if year = 0 then accumulated else
    let newPrincipal := accumulated + yearlyNetInterest
    calculate (year - 1) newPrincipal
calculate years principal

theorem investment_total_amount :
  let finalAmount := compoundedInvestment 15000 0.05 0.10 4
  round finalAmount = 17607 :=
by
  sorry

end investment_total_amount_l186_186512


namespace solve_for_x_l186_186851

theorem solve_for_x : ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 3.391 := 
by 
  sorry

end solve_for_x_l186_186851


namespace range_of_x_l186_186608

theorem range_of_x (a : ℝ) (x : ℝ) (h_a : 1 ≤ a ∧ a ≤ 3) (h : a * x^2 + (a - 2) * x - 2 > 0) :
  x < -1 ∨ x > 2 / 3 :=
sorry

end range_of_x_l186_186608


namespace negation_of_p_l186_186452

variable (x : ℝ)

def proposition_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

theorem negation_of_p : ¬ (∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end negation_of_p_l186_186452


namespace maximum_value_of_N_l186_186077

-- Define J_k based on the conditions given
def J (k : ℕ) : ℕ := 10^(k+3) + 128

-- Define the number of factors of 2 in the prime factorization of J_k
def N (k : ℕ) : ℕ := Nat.factorization (J k) 2

-- The proposition to be proved
theorem maximum_value_of_N (k : ℕ) (hk : k > 0) : N 4 = 7 :=
by
  sorry

end maximum_value_of_N_l186_186077


namespace no_fixed_points_range_l186_186078

def no_fixed_points (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 ≠ x

theorem no_fixed_points_range (a : ℝ) : no_fixed_points a ↔ -1 < a ∧ a < 3 := by
  sorry

end no_fixed_points_range_l186_186078


namespace girls_together_count_l186_186927

-- Define the problem conditions
def boys : ℕ := 4
def girls : ℕ := 2
def total_entities : ℕ := boys + (girls - 1) -- One entity for the two girls together

-- Calculate the factorial
noncomputable def factorial (n: ℕ) : ℕ :=
  if n = 0 then 1 else (List.range (n+1)).foldl (λx y => x * y) 1

-- Define the total number of ways girls can be together
noncomputable def ways_girls_together : ℕ :=
  factorial total_entities * factorial girls

-- State the theorem that needs to be proved
theorem girls_together_count : ways_girls_together = 240 := by
  sorry

end girls_together_count_l186_186927


namespace total_price_of_shoes_l186_186873

theorem total_price_of_shoes
  (S J : ℝ) 
  (h1 : 6 * S + 4 * J = 560) 
  (h2 : J = S / 4) :
  6 * S = 480 :=
by 
  -- Begin the proof environment
  sorry -- Placeholder for the actual proof

end total_price_of_shoes_l186_186873


namespace radar_arrangements_l186_186887

-- Define the number of letters in the word RADAR
def total_letters : Nat := 5

-- Define the number of times each letter is repeated
def repetition_R : Nat := 2
def repetition_A : Nat := 2

-- The expected number of unique arrangements
def expected_unique_arrangements : Nat := 30

theorem radar_arrangements :
  (Nat.factorial total_letters) / (Nat.factorial repetition_R * Nat.factorial repetition_A) = expected_unique_arrangements := by
  sorry

end radar_arrangements_l186_186887


namespace angle_in_fourth_quadrant_l186_186123

theorem angle_in_fourth_quadrant (θ : ℝ) (hθ : θ = 300) : 270 < θ ∧ θ < 360 :=
by
  -- theta equals 300
  have h1 : θ = 300 := hθ
  -- check that 300 degrees lies between 270 and 360
  sorry

end angle_in_fourth_quadrant_l186_186123


namespace bus_speed_l186_186053

noncomputable def radius : ℝ := 35 / 100  -- Radius in meters
noncomputable def rpm : ℝ := 500.4549590536851

noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def distance_in_one_minute : ℝ := circumference * rpm
noncomputable def distance_in_km_per_hour : ℝ := (distance_in_one_minute / 1000) * 60

theorem bus_speed :
  distance_in_km_per_hour = 66.037 :=
by
  -- The proof is skipped here as it is not required
  sorry

end bus_speed_l186_186053


namespace least_number_with_remainder_4_l186_186441

theorem least_number_with_remainder_4 : ∃ n : ℕ, n = 184 ∧ 
  (∀ d ∈ [5, 9, 12, 18], (n - 4) % d = 0) ∧
  (∀ m : ℕ, (∀ d ∈ [5, 9, 12, 18], (m - 4) % d = 0) → m ≥ n) :=
by
  sorry

end least_number_with_remainder_4_l186_186441


namespace measure_AB_l186_186957

noncomputable def segment_measure (a b : ℝ) : ℝ :=
  a + (2 / 3) * b

theorem measure_AB (a b : ℝ) (parallel_AB_CD : true) (angle_B_three_times_angle_D : true) (measure_AD_eq_a : true) (measure_CD_eq_b : true) :
  segment_measure a b = a + (2 / 3) * b :=
by
  sorry

end measure_AB_l186_186957


namespace total_percent_decrease_l186_186547

theorem total_percent_decrease (initial_value first_year_decrease second_year_decrease third_year_decrease : ℝ)
  (h₁ : first_year_decrease = 0.30)
  (h₂ : second_year_decrease = 0.10)
  (h₃ : third_year_decrease = 0.20) :
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let value_after_second_year := value_after_first_year * (1 - second_year_decrease)
  let value_after_third_year := value_after_second_year * (1 - third_year_decrease)
  let total_decrease := initial_value - value_after_third_year
  let total_percent_decrease := (total_decrease / initial_value) * 100
  total_percent_decrease = 49.60 := 
by
  sorry

end total_percent_decrease_l186_186547


namespace distance_between_trees_l186_186553

def yard_length : ℕ := 414
def number_of_trees : ℕ := 24

theorem distance_between_trees : yard_length / (number_of_trees - 1) = 18 := 
by sorry

end distance_between_trees_l186_186553


namespace jogging_track_circumference_l186_186869

theorem jogging_track_circumference 
  (deepak_speed : ℝ)
  (wife_speed : ℝ)
  (meeting_time : ℝ)
  (circumference : ℝ)
  (H1 : deepak_speed = 4.5)
  (H2 : wife_speed = 3.75)
  (H3 : meeting_time = 4.08) :
  circumference = 33.66 := sorry

end jogging_track_circumference_l186_186869


namespace pages_per_day_l186_186696

variable (P : ℕ) (D : ℕ)

theorem pages_per_day (hP : P = 66) (hD : D = 6) : P / D = 11 :=
by
  sorry

end pages_per_day_l186_186696


namespace no_fraternity_member_is_club_member_l186_186252

variable {U : Type} -- Domain of discourse, e.g., the set of all people at the school
variables (Club Member Student Honest Fraternity : U → Prop)

theorem no_fraternity_member_is_club_member
  (h1 : ∀ x, Club x → Student x)
  (h2 : ∀ x, Club x → ¬ Honest x)
  (h3 : ∀ x, Fraternity x → Honest x) :
  ∀ x, Fraternity x → ¬ Club x := 
sorry

end no_fraternity_member_is_club_member_l186_186252


namespace gas_total_cost_l186_186088

theorem gas_total_cost (x : ℝ) (h : (x/3) - 11 = x/5) : x = 82.5 :=
sorry

end gas_total_cost_l186_186088


namespace tank_capacity_l186_186994

noncomputable def leak_rate (C : ℝ) := C / 6
noncomputable def inlet_rate := 240
noncomputable def net_emptying_rate (C : ℝ) := C / 8

theorem tank_capacity : ∀ (C : ℝ), 
  (inlet_rate - leak_rate C = net_emptying_rate C) → 
  C = 5760 / 7 :=
by 
  sorry

end tank_capacity_l186_186994


namespace largest_natural_divisible_power_l186_186460

theorem largest_natural_divisible_power (p q : ℤ) (hp : p % 5 = 0) (hq : q % 5 = 0) (hdiscr : p^2 - 4*q > 0) :
  ∀ (α β : ℂ), (α^2 + p*α + q = 0 ∧ β^2 + p*β + q = 0) → (α^100 + β^100) % 5^50 = 0 :=
sorry

end largest_natural_divisible_power_l186_186460


namespace find_a_l186_186358

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + (a + 2)

def g (x a : ℝ) := (a + 1) * x
def h (x a : ℝ) := x^2 + a + 2

def p (a : ℝ) := ∀ x ≥ (a + 1)^2, f x a ≤ x
def q (a : ℝ) := ∀ x, g x a < 0

theorem find_a : 
  (¬p a) → (p a ∨ q a) → a ≥ -1 := sorry

end find_a_l186_186358


namespace find_g_six_l186_186932

noncomputable def g : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : g (x + y) = g x + g y
axiom g_five : g 5 = 6

theorem find_g_six : g 6 = 36/5 := 
by 
  -- proof to be filled in
  sorry

end find_g_six_l186_186932


namespace ratio_difference_l186_186823

theorem ratio_difference (x : ℕ) (h_largest : 7 * x = 70) : 70 - 3 * x = 40 := by
  sorry

end ratio_difference_l186_186823


namespace given_problem_l186_186897

theorem given_problem (x y : ℝ) (hx : x ≠ 0) (hx4 : x ≠ 4) (hy : y ≠ 0) (hy6 : y ≠ 6) :
  (2 / x + 3 / y = 1 / 2) ↔ (4 * y / (y - 6) = x) :=
sorry

end given_problem_l186_186897


namespace find_coordinates_of_B_l186_186714

-- Define points A and B, and vector a
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 5 }
def a : Point := { x := 2, y := 3 }

-- Define the proof problem
theorem find_coordinates_of_B (B : Point) 
  (h1 : B.x + 1 = 3 * a.x)
  (h2 : B.y - 5 = 3 * a.y) : 
  B = { x := 5, y := 14 } := 
sorry

end find_coordinates_of_B_l186_186714


namespace find_b_when_a_is_1600_l186_186594

theorem find_b_when_a_is_1600 :
  ∀ (a b : ℝ), (a * b = 400) ∧ ((2 * a) * b = 600) → (1600 * b = 600) → b = 0.375 :=
by
  intro a b
  intro h
  sorry

end find_b_when_a_is_1600_l186_186594


namespace solve_for_x_l186_186235

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 2 * x - 3 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end solve_for_x_l186_186235


namespace mother_age_when_harry_born_l186_186808

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_l186_186808


namespace average_of_four_l186_186433

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end average_of_four_l186_186433


namespace prob1_prob2_prob3_prob4_prob5_l186_186073

theorem prob1 : (1 - 27 + (-32) + (-8) + 27) = -40 := sorry

theorem prob2 : (2 * -5 + abs (-3)) = -2 := sorry

theorem prob3 (x y : Int) (h₁ : -x = 3) (h₂ : abs y = 5) : x + y = 2 ∨ x + y = -8 := sorry

theorem prob4 : ((-1 : Int) * (3 / 2) + (5 / 4) + (-5 / 2) - (-13 / 4) - (5 / 4)) = -3 / 4 := sorry

theorem prob5 (a b : Int) (h : abs (a - 4) + abs (b + 5) = 0) : a - b = 9 := sorry

end prob1_prob2_prob3_prob4_prob5_l186_186073


namespace solve_quadratic_eq_l186_186068

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic_eq :
  ∀ a b c x1 x2 : ℝ,
  a = 2 →
  b = -2 →
  c = -1 →
  quadratic_eq a b c x1 ∧ quadratic_eq a b c x2 →
  (x1 = (1 + Real.sqrt 3) / 2 ∧ x2 = (1 - Real.sqrt 3) / 2) :=
by
  intros a b c x1 x2 ha hb hc h
  sorry

end solve_quadratic_eq_l186_186068


namespace angles_with_same_terminal_side_as_15_degree_l186_186555

def condition1 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 90
def condition2 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 180
def condition3 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 360
def condition4 (β : ℝ) (k : ℤ) : Prop := β = 15 + 2 * k * 360

def has_same_terminal_side_as_15_degree (β : ℝ) : Prop :=
  ∃ k : ℤ, β = 15 + k * 360

theorem angles_with_same_terminal_side_as_15_degree (β : ℝ) :
  (∃ k : ℤ, condition1 β k)  ∨
  (∃ k : ℤ, condition2 β k)  ∨
  (∃ k : ℤ, condition3 β k)  ∨
  (∃ k : ℤ, condition4 β k) →
  has_same_terminal_side_as_15_degree β :=
by
  sorry

end angles_with_same_terminal_side_as_15_degree_l186_186555


namespace proof_x_plus_y_l186_186267

variables (x y : ℝ)

-- Definitions for the given conditions
def cond1 (x y : ℝ) : Prop := 2 * |x| + x + y = 18
def cond2 (x y : ℝ) : Prop := x + 2 * |y| - y = 14

theorem proof_x_plus_y (x y : ℝ) (h1 : cond1 x y) (h2 : cond2 x y) : x + y = 14 := by
  sorry

end proof_x_plus_y_l186_186267


namespace min_a_n_l186_186806

def a_n (n : ℕ) : ℤ := n^2 - 8 * n + 5

theorem min_a_n : ∃ n : ℕ, ∀ m : ℕ, a_n n ≤ a_n m ∧ a_n n = -11 :=
by
  sorry

end min_a_n_l186_186806


namespace part1_monotonicity_part2_inequality_l186_186940

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem part1_monotonicity (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x : ℝ, (x < Real.log (1 / a) → f a x > f a (x + 1)) ∧
  (x > Real.log (1 / a) → f a x < f a (x + 1))) := sorry

theorem part2_inequality (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + (3 / 2) := sorry

end part1_monotonicity_part2_inequality_l186_186940


namespace find_atomic_weight_of_Na_l186_186271

def atomic_weight_of_Na_is_correct : Prop :=
  ∃ (atomic_weight_of_Na : ℝ),
    (atomic_weight_of_Na + 35.45 + 16.00 = 74) ∧ (atomic_weight_of_Na = 22.55)

theorem find_atomic_weight_of_Na : atomic_weight_of_Na_is_correct :=
by
  sorry

end find_atomic_weight_of_Na_l186_186271


namespace open_spots_level4_correct_l186_186579

noncomputable def open_spots_level_4 (total_levels : ℕ) (spots_per_level : ℕ) (open_spots_level1 : ℕ) (open_spots_level2 : ℕ) (open_spots_level3 : ℕ) (full_spots_total : ℕ) : ℕ := 
  let total_spots := total_levels * spots_per_level
  let open_spots_total := total_spots - full_spots_total 
  let open_spots_first_three := open_spots_level1 + open_spots_level2 + open_spots_level3
  open_spots_total - open_spots_first_three

theorem open_spots_level4_correct :
  open_spots_level_4 4 100 58 (58 + 2) (58 + 2 + 5) 186 = 31 :=
by
  sorry

end open_spots_level4_correct_l186_186579


namespace sin_double_angle_log_simplification_l186_186727

-- Problem 1: Prove sin(2 * α) = 7 / 25 given sin(α - π / 4) = 3 / 5
theorem sin_double_angle (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 3 / 5) : Real.sin (2 * α) = 7 / 25 :=
by
  sorry

-- Problem 2: Prove 2 * log₅ 10 + log₅ 0.25 = 2
theorem log_simplification : 2 * Real.log 10 / Real.log 5 + Real.log (0.25) / Real.log 5 = 2 :=
by
  sorry

end sin_double_angle_log_simplification_l186_186727


namespace competition_winner_is_C_l186_186194

-- Define the type for singers
inductive Singer
| A | B | C | D
deriving DecidableEq

-- Assume each singer makes a statement
def statement (s : Singer) : Prop :=
  match s with
  | Singer.A => Singer.B ≠ Singer.C
  | Singer.B => Singer.A ≠ Singer.C
  | Singer.C => true
  | Singer.D => Singer.B ≠ Singer.D

-- Define that two and only two statements are true
def exactly_two_statements_are_true : Prop :=
  (statement Singer.A ∧ statement Singer.C ∧ ¬statement Singer.B ∧ ¬statement Singer.D) ∨
  (statement Singer.A ∧ statement Singer.D ∧ ¬statement Singer.B ∧ ¬statement Singer.C)

-- Define the winner
def winner : Singer := Singer.C

-- The main theorem to be proved
theorem competition_winner_is_C :
  exactly_two_statements_are_true → (winner = Singer.C) :=
by
  intro h
  exact sorry

end competition_winner_is_C_l186_186194


namespace five_student_committees_from_ten_select_two_committees_with_three_overlap_l186_186476

-- Lean statement for the first part: number of different five-student committees from ten students.
theorem five_student_committees_from_ten : 
  (Nat.choose 10 5) = 252 := 
by
  sorry

-- Lean statement for the second part: number of ways to choose two five-student committees with exactly three overlapping members.
theorem select_two_committees_with_three_overlap :
  ( (Nat.choose 10 5) * ( (Nat.choose 5 3) * (Nat.choose 5 2) ) ) / 2 = 12600 := 
by
  sorry

end five_student_committees_from_ten_select_two_committees_with_three_overlap_l186_186476


namespace ValleyFalcons_all_items_l186_186639

noncomputable def num_fans_receiving_all_items (capacity : ℕ) (tshirt_interval : ℕ) 
  (cap_interval : ℕ) (wristband_interval : ℕ) : ℕ :=
  (capacity / Nat.lcm (Nat.lcm tshirt_interval cap_interval) wristband_interval)

theorem ValleyFalcons_all_items:
  num_fans_receiving_all_items 3000 50 25 60 = 10 :=
by
  -- This is where the mathematical proof would go
  sorry

end ValleyFalcons_all_items_l186_186639


namespace distance_of_third_point_on_trip_l186_186958

theorem distance_of_third_point_on_trip (D : ℝ) (h1 : D + 2 * D + (1/2) * D + 7 * D = 560) :
  (1/2) * D = 27 :=
by
  sorry

end distance_of_third_point_on_trip_l186_186958


namespace logistics_company_freight_l186_186881

theorem logistics_company_freight :
  ∃ (x y : ℕ), 
    50 * x + 30 * y = 9500 ∧
    70 * x + 40 * y = 13000 ∧
    x = 100 ∧
    y = 140 :=
by
  -- The proof is skipped here
  sorry

end logistics_company_freight_l186_186881


namespace ratio_of_routes_l186_186092

-- Definitions of m and n
def m : ℕ := 2 
def n : ℕ := 6

-- Theorem statement
theorem ratio_of_routes (m_positive : m > 0) : n / m = 3 := by
  sorry

end ratio_of_routes_l186_186092


namespace believe_more_blue_l186_186754

-- Define the conditions
def total_people : ℕ := 150
def more_green : ℕ := 90
def both_more_green_and_more_blue : ℕ := 40
def neither : ℕ := 20

-- Theorem statement: Prove that the number of people who believe teal is "more blue" is 80
theorem believe_more_blue : 
  total_people - neither - (more_green - both_more_green_and_more_blue) = 80 :=
by
  sorry

end believe_more_blue_l186_186754


namespace max_profit_at_60_l186_186069

variable (x : ℕ) (y W : ℝ)

def charter_fee : ℝ := 15000
def max_group_size : ℕ := 75

def ticket_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 900
  else if 30 < x ∧ x ≤ max_group_size then -10 * (x - 30) + 900
  else 0

def profit (x : ℕ) : ℝ :=
  if x ≤ 30 then 900 * x - charter_fee
  else if 30 < x ∧ x ≤ max_group_size then (-10 * x + 1200) * x - charter_fee
  else 0

theorem max_profit_at_60 : x = 60 → profit x = 21000 := by
  sorry

end max_profit_at_60_l186_186069


namespace average_speed_round_trip_l186_186355

-- Define average speed calculation for round trip

open Real

theorem average_speed_round_trip (S : ℝ) (hS : S > 0) :
  let t1 := S / 6
  let t2 := S / 4
  let total_distance := 2 * S
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 4.8 :=
  by
    sorry

end average_speed_round_trip_l186_186355


namespace train_length_is_300_l186_186901

theorem train_length_is_300 (L V : ℝ)
    (h1 : L = V * 20)
    (h2 : L + 285 = V * 39) :
    L = 300 := by
  sorry

end train_length_is_300_l186_186901


namespace sum_of_youngest_and_oldest_nephews_l186_186844

theorem sum_of_youngest_and_oldest_nephews 
    (n1 n2 n3 n4 n5 n6 : ℕ) 
    (mean_eq : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = 10) 
    (median_eq : (n3 + n4) / 2 = 12) : 
    n1 + n6 = 12 := 
by 
    sorry

end sum_of_youngest_and_oldest_nephews_l186_186844


namespace calc_exponent_l186_186161

theorem calc_exponent (a b : ℕ) : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end calc_exponent_l186_186161


namespace find_slope_of_line_l186_186939

-- Define the parabola, point M, and the conditions leading to the slope k.
theorem find_slope_of_line (k : ℝ) :
  let C := {p : ℝ × ℝ | p.2^2 = 4 * p.1}
  let focus : ℝ × ℝ := (1, 0)
  let M : ℝ × ℝ := (-1, 1)
  let line (k : ℝ) (x : ℝ) := k * (x - 1)
  ∃ A B : (ℝ × ℝ), 
    A ∈ C ∧ B ∈ C ∧
    A ≠ B ∧
    A.1 + 1 = B.1 + 1 ∧ 
    A.2 - 1 = B.2 - 1 ∧
    ((A.1 + 1) * (B.1 + 1) + (A.2 - 1) * (B.2 - 1) = 0) -> k = 2 := 
by
  sorry

end find_slope_of_line_l186_186939


namespace total_guppies_l186_186418

noncomputable def initial_guppies : Nat := 7
noncomputable def baby_guppies_first_set : Nat := 3 * 12
noncomputable def baby_guppies_additional : Nat := 9

theorem total_guppies : initial_guppies + baby_guppies_first_set + baby_guppies_additional = 52 :=
by
  sorry

end total_guppies_l186_186418


namespace street_tree_fourth_point_l186_186534

theorem street_tree_fourth_point (a b : ℝ) (h_a : a = 0.35) (h_b : b = 0.37) :
  (a + 4 * ((b - a) / 4)) = b :=
by 
  rw [h_a, h_b]
  sorry

end street_tree_fourth_point_l186_186534


namespace determine_n_l186_186042

-- All the terms used in the conditions
variables (S C M : ℝ)
variables (n : ℝ)

-- Define the conditions as hypotheses
def condition1 := M = 1 / 3 * S
def condition2 := M = 1 / n * C

-- The main theorem statement
theorem determine_n (S C M : ℝ) (n : ℝ) (h1 : condition1 S M) (h2 : condition2 M n C) : n = 2 :=
by sorry

end determine_n_l186_186042


namespace four_pq_plus_four_qp_l186_186953

theorem four_pq_plus_four_qp (p q : ℝ) (h : p / q - q / p = 21 / 10) : 
  4 * p / q + 4 * q / p = 16.8 :=
sorry

end four_pq_plus_four_qp_l186_186953


namespace school_student_count_l186_186540

theorem school_student_count (pencils erasers pencils_per_student erasers_per_student students : ℕ) 
    (h1 : pencils = 195) 
    (h2 : erasers = 65) 
    (h3 : pencils_per_student = 3)
    (h4 : erasers_per_student = 1) :
    students = pencils / pencils_per_student ∧ students = erasers / erasers_per_student → students = 65 :=
by
  sorry

end school_student_count_l186_186540


namespace simplify_and_evaluate_l186_186273

-- Defining the conditions
def a : Int := -3
def b : Int := -2

-- Defining the expression
def expr (a b : Int) : Int := (3 * a^2 * b + 2 * a * b^2) - (2 * (a^2 * b - 1) + 3 * a * b^2 + 2)

-- Stating the theorem/proof problem
theorem simplify_and_evaluate : expr a b = -6 := by
  sorry

end simplify_and_evaluate_l186_186273


namespace area_of_large_hexagon_eq_270_l186_186397

noncomputable def area_large_hexagon (area_shaded : ℝ) (n_small_hexagons_shaded : ℕ) (n_small_hexagons_large : ℕ): ℝ :=
  let area_one_small_hexagon := area_shaded / n_small_hexagons_shaded
  area_one_small_hexagon * n_small_hexagons_large

theorem area_of_large_hexagon_eq_270 :
  area_large_hexagon 180 6 7 = 270 := by
  sorry

end area_of_large_hexagon_eq_270_l186_186397


namespace total_seats_l186_186581

-- Define the conditions
variable {S : ℝ} -- Total number of seats in the hall
variable {vacantSeats : ℝ} (h_vacant : vacantSeats = 240) -- Number of vacant seats
variable {filledPercentage : ℝ} (h_filled : filledPercentage = 0.60) -- Percentage of seats filled

-- Total seats in the hall
theorem total_seats (h : 0.40 * S = 240) : S = 600 :=
sorry

end total_seats_l186_186581


namespace runway_show_time_l186_186695

/-
Problem: Prove that it will take 60 minutes to complete all of the runway trips during the show, 
given the following conditions:
- There are 6 models in the show.
- Each model will wear two sets of bathing suits and three sets of evening wear clothes during the runway portion of the show.
- It takes a model 2 minutes to walk out to the end of the runway and back, and models take turns, one at a time.
-/

theorem runway_show_time 
    (num_models : ℕ) 
    (sets_bathing_suits_per_model : ℕ) 
    (sets_evening_wear_per_model : ℕ) 
    (time_per_trip : ℕ) 
    (total_time : ℕ) :
    num_models = 6 →
    sets_bathing_suits_per_model = 2 →
    sets_evening_wear_per_model = 3 →
    time_per_trip = 2 →
    total_time = num_models * (sets_bathing_suits_per_model + sets_evening_wear_per_model) * time_per_trip →
    total_time = 60 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5


end runway_show_time_l186_186695


namespace joan_paid_amount_l186_186810

theorem joan_paid_amount (J K : ℕ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end joan_paid_amount_l186_186810


namespace edge_length_of_cube_l186_186768

noncomputable def cost_per_quart : ℝ := 3.20
noncomputable def coverage_per_quart : ℕ := 120
noncomputable def total_cost : ℝ := 16
noncomputable def total_coverage : ℕ := 600 -- From 5 quarts * 120 square feet per quart
noncomputable def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)

theorem edge_length_of_cube :
  (∃ edge_length : ℝ, surface_area edge_length = total_coverage) → 
  ∃ edge_length : ℝ, edge_length = 10 :=
by
  sorry

end edge_length_of_cube_l186_186768


namespace tonya_hamburgers_to_beat_winner_l186_186975

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

end tonya_hamburgers_to_beat_winner_l186_186975


namespace minimum_rounds_l186_186392

-- Given conditions based on the problem statement
variable (m : ℕ) (hm : m ≥ 17)
variable (players : Fin (2 * m)) -- Representing 2m players
variable (rounds : Fin (2 * m - 1)) -- Representing 2m - 1 rounds
variable (pairs : Fin m → Fin (2 * m) × Fin (2 * m)) -- Pairing for each of the m pairs in each round

-- Statement of the proof problem
theorem minimum_rounds (h1 : ∀ i j, i ≠ j → ∃! (k : Fin m), pairs k = (i, j) ∨ pairs k = (j, i))
(h2 : ∀ k : Fin m, (pairs k).fst ≠ (pairs k).snd)
(h3 : ∀ i j, i ≠ j → ∃ r : Fin (2 * m - 1), (∃ k : Fin m, pairs k = (i, j)) ∧ (∃ k : Fin m, pairs k = (j, i))) :
∃ (n : ℕ), n = m - 1 ∧ ∀ s : Fin 4 → Fin (2 * m), (∀ i j, i ≠ j → ¬ ∃ r : Fin n, ∃ k : Fin m, pairs k = (s i, s j)) ∨ (∃ r1 r2 : Fin n, ∃ i j, i ≠ j ∧ ∃ k1 k2 : Fin m, pairs k1 = (s i, s j) ∧ pairs k2 = (s j, s i)) :=
sorry

end minimum_rounds_l186_186392


namespace solution_of_inequality_l186_186200

noncomputable def solutionSet (a x : ℝ) : Set ℝ :=
  if a > 0 then {x | -a < x ∧ x < 3 * a}
  else if a < 0 then {x | 3 * a < x ∧ x < -a}
  else ∅

theorem solution_of_inequality (a x : ℝ) :
  (x^2 - 2 * a * x - 3 * a^2 < 0 ↔ x ∈ solutionSet a x) :=
sorry

end solution_of_inequality_l186_186200


namespace minimum_ticket_cost_correct_l186_186446

noncomputable def minimum_ticket_cost : Nat :=
let adults := 8
let children := 4
let adult_ticket_price := 100
let child_ticket_price := 50
let group_ticket_price := 70
let group_size := 10
-- Calculate the cost of group tickets for 10 people and regular tickets for 2 children
let total_cost := (group_size * group_ticket_price) + (2 * child_ticket_price)
total_cost

theorem minimum_ticket_cost_correct :
  minimum_ticket_cost = 800 := by
  sorry

end minimum_ticket_cost_correct_l186_186446


namespace arithmetic_geometric_sequence_l186_186731

open Real

noncomputable def a_4 (a1 q : ℝ) : ℝ := a1 * q^3
noncomputable def sum_five_terms (a1 q : ℝ) : ℝ := a1 * (1 - q^5) / (1 - q)

theorem arithmetic_geometric_sequence :
  ∀ (a1 q : ℝ),
    (a1 + a1 * q^2 = 10) →
    (a1 * q^3 + a1 * q^5 = 5 / 4) →
    (a_4 a1 q = 1) ∧ (sum_five_terms a1 q = 31 / 2) :=
by
  intros a1 q h1 h2
  sorry

end arithmetic_geometric_sequence_l186_186731


namespace base_10_representation_l186_186748

-- Conditions
variables (C D : ℕ)
variables (hC : 0 ≤ C ∧ C ≤ 7)
variables (hD : 0 ≤ D ∧ D ≤ 5)
variables (hEq : 8 * C + D = 6 * D + C)

-- Goal
theorem base_10_representation : 8 * C + D = 0 := by
  sorry

end base_10_representation_l186_186748


namespace program_output_l186_186132

theorem program_output :
  ∃ a b : ℕ, a = 10 ∧ b = a - 8 ∧ a = a - b ∧ a = 8 :=
by
  let a := 10
  let b := a - 8
  let a := a - b
  use a
  use b
  sorry

end program_output_l186_186132


namespace stan_weighs_5_more_than_steve_l186_186417

theorem stan_weighs_5_more_than_steve
(S V J : ℕ) 
(h1 : J = 110)
(h2 : V = J - 8)
(h3 : S + V + J = 319) : 
(S - V = 5) :=
by
  sorry

end stan_weighs_5_more_than_steve_l186_186417


namespace compute_difference_l186_186120

def distinct_solutions (p q : ℝ) : Prop :=
  (p ≠ q) ∧ (∃ (x : ℝ), (x = p ∨ x = q) ∧ (x-3)*(x+3) = 21*x - 63) ∧
  (p > q)

theorem compute_difference (p q : ℝ) (h : distinct_solutions p q) : p - q = 15 :=
by
  sorry

end compute_difference_l186_186120


namespace remove_terms_for_desired_sum_l186_186230

theorem remove_terms_for_desired_sum :
  let series_sum := (1/3) + (1/5) + (1/7) + (1/9) + (1/11) + (1/13)
  series_sum - (1/11 + 1/13) = 11/20 :=
by
  sorry

end remove_terms_for_desired_sum_l186_186230


namespace max_m_value_real_roots_interval_l186_186838

theorem max_m_value_real_roots_interval :
  (∃ x ∈ (Set.Icc 0 1), x^3 - 3 * x - m = 0) → m ≤ 0 :=
by
  sorry 

end max_m_value_real_roots_interval_l186_186838


namespace solve_for_t_l186_186180

theorem solve_for_t (t : ℝ) (ht : (t^2 - 3*t - 70) / (t - 10) = 7 / (t + 4)) : 
  t = -3 := sorry

end solve_for_t_l186_186180


namespace circle_through_A_B_and_tangent_to_m_l186_186222

noncomputable def circle_equation (x y : ℚ) : Prop :=
  x^2 + (y - 1/3)^2 = 16/9

theorem circle_through_A_B_and_tangent_to_m :
  ∃ (c : ℚ × ℚ) (r : ℚ),
    (c = (0, 1/3)) ∧
    (r = 4/3) ∧
    (∀ (x y : ℚ),
      (x = 0 ∧ y = -1 ∨ x = 4/3 ∧ y = 1/3 → (x^2 + (y - 1/3)^2 = 16/9)) ∧
      (x = 4/3 → x = r)) :=
by
  sorry

end circle_through_A_B_and_tangent_to_m_l186_186222


namespace real_solutions_of_fraction_eqn_l186_186103

theorem real_solutions_of_fraction_eqn (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ 7) :
  ( x = 3 + Real.sqrt 3 ∨ x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5 ) ↔
    ((x - 1) * (x - 3) * (x - 5) * (x - 7) * (x - 3) * (x - 5) * (x - 1)) / ((x - 3) * (x - 7) * (x - 3)) = 1 :=
sorry

end real_solutions_of_fraction_eqn_l186_186103


namespace number_of_small_cubes_l186_186930

-- Definition of the conditions from the problem
def painted_cube (n : ℕ) :=
  6 * (n - 2) * (n - 2) = 54

-- The theorem we need to prove
theorem number_of_small_cubes (n : ℕ) (h : painted_cube n) : n^3 = 125 :=
by
  have h1 : 6 * (n - 2) * (n - 2) = 54 := h
  sorry

end number_of_small_cubes_l186_186930


namespace additional_people_needed_l186_186234

-- Definition of the conditions
def person_hours (people: ℕ) (hours: ℕ) : ℕ := people * hours

-- Assertion that 8 people can paint the fence in 3 hours
def eight_people_three_hours : Prop := person_hours 8 3 = 24

-- Definition of the additional people required
def additional_people (initial_people required_people: ℕ) : ℕ := required_people - initial_people

-- Main theorem stating the problem
theorem additional_people_needed : eight_people_three_hours → additional_people 8 12 = 4 :=
by
  sorry

end additional_people_needed_l186_186234


namespace total_copies_in_half_hour_l186_186229

-- Define the rates of the copy machines
def rate_machine1 : ℕ := 35
def rate_machine2 : ℕ := 65

-- Define the duration of time in minutes
def time_minutes : ℕ := 30

-- Define the total number of copies made by both machines in the given duration
def total_copies_made : ℕ := rate_machine1 * time_minutes + rate_machine2 * time_minutes

-- Prove that the total number of copies made is 3000
theorem total_copies_in_half_hour : total_copies_made = 3000 := by
  -- The proof is skipped with sorry for the demonstration purpose
  sorry

end total_copies_in_half_hour_l186_186229


namespace license_plates_count_l186_186002

def num_consonants : Nat := 20
def num_vowels : Nat := 6
def num_digits : Nat := 10
def num_symbols : Nat := 3

theorem license_plates_count : 
  num_consonants * num_vowels * num_consonants * num_digits * num_symbols = 72000 :=
by 
  sorry

end license_plates_count_l186_186002


namespace stream_speed_l186_186552

def boat_speed_still : ℝ := 30
def distance_downstream : ℝ := 80
def distance_upstream : ℝ := 40

theorem stream_speed (v : ℝ) (h : (distance_downstream / (boat_speed_still + v) = distance_upstream / (boat_speed_still - v))) :
  v = 10 :=
sorry

end stream_speed_l186_186552


namespace vertical_asymptotes_polynomial_l186_186924

theorem vertical_asymptotes_polynomial (a b : ℝ) (h₁ : -3 * 2 = b) (h₂ : -3 + 2 = a) : a + b = -5 := by
  sorry

end vertical_asymptotes_polynomial_l186_186924


namespace find_percentage_l186_186589

theorem find_percentage (P : ℝ) : 
  0.15 * P * (0.5 * 5600) = 126 → P = 0.3 := 
by 
  sorry

end find_percentage_l186_186589


namespace integer_product_l186_186743

open Real

theorem integer_product (P Q R S : ℕ) (h1 : P + Q + R + S = 48)
    (h2 : P + 3 = Q - 3) (h3 : P + 3 = R * 3) (h4 : P + 3 = S / 3) :
    P * Q * R * S = 5832 :=
sorry

end integer_product_l186_186743


namespace expected_value_is_0_point_25_l186_186216

-- Define the probabilities and earnings
def prob_roll_1 := 1/4
def earning_1 := 4
def prob_roll_2 := 1/4
def earning_2 := -3
def prob_roll_3_to_6 := 1/8
def earning_3_to_6 := 0

-- Define the expected value calculation
noncomputable def expected_value : ℝ := 
  (prob_roll_1 * earning_1) + 
  (prob_roll_2 * earning_2) + 
  (prob_roll_3_to_6 * earning_3_to_6) * 4  -- For 3, 4, 5, and 6

-- The theorem to be proved
theorem expected_value_is_0_point_25 : expected_value = 0.25 := by
  sorry

end expected_value_is_0_point_25_l186_186216


namespace largest_integer_divisor_l186_186006

theorem largest_integer_divisor (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end largest_integer_divisor_l186_186006


namespace ceil_minus_val_eq_one_minus_frac_l186_186694

variable (x : ℝ)

theorem ceil_minus_val_eq_one_minus_frac (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 ≤ f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f := 
sorry

end ceil_minus_val_eq_one_minus_frac_l186_186694


namespace remainder_2023_mul_7_div_45_l186_186312

/-- The remainder when the product of 2023 and 7 is divided by 45 is 31. -/
theorem remainder_2023_mul_7_div_45 : 
  (2023 * 7) % 45 = 31 := 
by
  sorry

end remainder_2023_mul_7_div_45_l186_186312


namespace num_valid_four_digit_numbers_l186_186336

theorem num_valid_four_digit_numbers :
  let N (a b c d : ℕ) := 1000 * a + 100 * b + 10 * c + d
  ∃ (a b c d : ℕ), 5000 ≤ N a b c d ∧ N a b c d < 7000 ∧ (N a b c d % 5 = 0) ∧ (2 ≤ b ∧ b < c ∧ c ≤ 7) ∧
                   (60 = (if a = 5 ∨ a = 6 then (if d = 0 ∨ d = 5 then 15 else 0) else 0)) :=
sorry

end num_valid_four_digit_numbers_l186_186336


namespace roots_product_l186_186864

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end roots_product_l186_186864


namespace no_nat_numbers_m_n_satisfy_eq_l186_186152

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end no_nat_numbers_m_n_satisfy_eq_l186_186152


namespace k1_k2_ratio_l186_186794

theorem k1_k2_ratio (a b k k1 k2 : ℝ)
  (h1 : a^2 * k - (k - 1) * a + 5 = 0)
  (h2 : b^2 * k - (k - 1) * b + 5 = 0)
  (h3 : (a / b) + (b / a) = 4/5)
  (h4 : k1^2 - 16 * k1 + 1 = 0)
  (h5 : k2^2 - 16 * k2 + 1 = 0) :
  (k1 / k2) + (k2 / k1) = 254 := by
  sorry

end k1_k2_ratio_l186_186794


namespace instantaneous_rate_of_change_at_e_l186_186129

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem instantaneous_rate_of_change_at_e : deriv f e = 0 := by
  sorry

end instantaneous_rate_of_change_at_e_l186_186129


namespace tangent_and_normal_lines_l186_186517

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end tangent_and_normal_lines_l186_186517


namespace book_weight_l186_186365

theorem book_weight (total_weight : ℕ) (num_books : ℕ) (each_book_weight : ℕ) 
  (h1 : total_weight = 42) (h2 : num_books = 14) :
  each_book_weight = total_weight / num_books :=
by
  sorry

end book_weight_l186_186365


namespace problem1_problem2_l186_186850

/-- Proof statement for the first mathematical problem -/
theorem problem1 (x : ℝ) (h : (x - 2) ^ 2 = 9) : x = 5 ∨ x = -1 :=
by {
  -- Proof goes here
  sorry
}

/-- Proof statement for the second mathematical problem -/
theorem problem2 (x : ℝ) (h : 27 * (x + 1) ^ 3 + 8 = 0) : x = -5 / 3 :=
by {
  -- Proof goes here
  sorry
}

end problem1_problem2_l186_186850


namespace find_other_asymptote_l186_186689

-- Define the conditions
def one_asymptote (x : ℝ) : ℝ := 3 * x
def foci_x_coordinate : ℝ := 5

-- Define the expected answer
def other_asymptote (x : ℝ) : ℝ := -3 * x + 30

-- Theorem statement to prove the equation of the other asymptote
theorem find_other_asymptote :
  (∀ x, y = one_asymptote x) →
  (∀ _x, _x = foci_x_coordinate) →
  (∀ x, y = other_asymptote x) :=
by
  intros h_one_asymptote h_foci_x
  sorry

end find_other_asymptote_l186_186689


namespace largest_non_sum_of_multiple_of_30_and_composite_l186_186928

theorem largest_non_sum_of_multiple_of_30_and_composite :
  ∃ (n : ℕ), n = 211 ∧ ∀ a b : ℕ, (a > 0) → (b > 0) → (b < 30) → 
  n ≠ 30 * a + b ∧ ¬ ∃ k : ℕ, k > 1 ∧ k < b ∧ b % k = 0 :=
sorry

end largest_non_sum_of_multiple_of_30_and_composite_l186_186928


namespace problem_g_eq_l186_186101

noncomputable def g : ℝ → ℝ := sorry

theorem problem_g_eq :
  (∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x + x) →
  g 3 = ( -31 - 3 * 3^(1/3)) / 8 :=
by
  intro h
  -- proof goes here
  sorry

end problem_g_eq_l186_186101


namespace michael_initial_money_l186_186334

theorem michael_initial_money (M : ℝ) 
  (half_give_away_to_brother : ∃ (m_half : ℝ), M / 2 = m_half)
  (brother_initial_money : ℝ := 17)
  (candy_cost : ℝ := 3)
  (brother_ends_up_with : ℝ := 35) :
  brother_initial_money + M / 2 - candy_cost = brother_ends_up_with ↔ M = 42 :=
sorry

end michael_initial_money_l186_186334


namespace revenue_after_fall_is_correct_l186_186033

variable (originalRevenue : ℝ) (percentageDecrease : ℝ)

theorem revenue_after_fall_is_correct :
    originalRevenue = 69 ∧ percentageDecrease = 39.130434782608695 →
    originalRevenue - (originalRevenue * (percentageDecrease / 100)) = 42 := by
  intro h
  rcases h with ⟨h1, h2⟩
  sorry

end revenue_after_fall_is_correct_l186_186033


namespace factorial_divisibility_l186_186404

theorem factorial_divisibility 
  {n : ℕ} 
  (hn : bit0 (n.bits.count 1) == 1995) : 
  (2^(n-1995)) ∣ n! := 
sorry

end factorial_divisibility_l186_186404


namespace combined_tax_rate_35_58_l186_186283

noncomputable def combined_tax_rate (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ) : ℝ :=
  let john_tax := john_tax_rate * john_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let total_tax := john_tax + ingrid_tax
  let total_income := john_income + ingrid_income
  (total_tax / total_income) * 100

theorem combined_tax_rate_35_58
  (john_income : ℝ) (john_tax_rate : ℝ) (ingrid_income : ℝ) (ingrid_tax_rate : ℝ)
  (h1 : john_income = 57000) (h2 : john_tax_rate = 0.3)
  (h3 : ingrid_income = 72000) (h4 : ingrid_tax_rate = 0.4) :
  combined_tax_rate john_income john_tax_rate ingrid_income ingrid_tax_rate = 35.58 :=
by
  simp [combined_tax_rate, h1, h2, h3, h4]
  sorry

end combined_tax_rate_35_58_l186_186283


namespace range_of_a_l186_186257

theorem range_of_a
  (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 6)
  (y : ℝ) (hy : 0 < y)
  (h : (y / 4 - 2 * (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) :
  a ≤ 3 :=
sorry

end range_of_a_l186_186257


namespace tickets_sold_in_total_l186_186899

def total_tickets
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ) : ℕ :=
  adult_tickets + student_tickets

theorem tickets_sold_in_total 
    (adult_price student_price : ℕ)
    (total_revenue adult_tickets student_tickets : ℕ)
    (h1 : adult_price = 6)
    (h2 : student_price = 3)
    (h3 : total_revenue = 3846)
    (h4 : adult_tickets = 410)
    (h5 : student_tickets = 436) :
  total_tickets adult_price student_price total_revenue adult_tickets student_tickets = 846 :=
by
  sorry

end tickets_sold_in_total_l186_186899


namespace quadratic_roots_l186_186466

theorem quadratic_roots (x : ℝ) : x^2 + 4 * x + 3 = 0 → x = -3 ∨ x = -1 :=
by
  intro h
  have h1 : (x + 3) * (x + 1) = 0 := by sorry
  have h2 : (x = -3 ∨ x = -1) := by sorry
  exact h2

end quadratic_roots_l186_186466


namespace find_percentage_l186_186968

theorem find_percentage (p : ℝ) (h : (p / 100) * 8 = 0.06) : p = 0.75 := 
by 
  sorry

end find_percentage_l186_186968


namespace final_population_l186_186164

theorem final_population (P0 : ℕ) (r1 r2 : ℝ) (P2 : ℝ) 
  (h0 : P0 = 1000)
  (h1 : r1 = 1.20)
  (h2 : r2 = 1.30)
  (h3 : P2 = P0 * r1 * r2) : 
  P2 = 1560 := 
sorry

end final_population_l186_186164


namespace number_of_pens_l186_186380

theorem number_of_pens (num_pencils : ℕ) (total_cost : ℝ) (avg_price_pencil : ℝ) (avg_price_pen : ℝ) : ℕ :=
  sorry

example : number_of_pens 75 690 2 18 = 30 :=
by 
  sorry

end number_of_pens_l186_186380


namespace solve_system_of_equations_l186_186405

theorem solve_system_of_equations : 
  ∃ (x y : ℝ), 
  (6 * x - 3 * y = -3) ∧ (5 * x - 9 * y = -35) ∧ (x = 2) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l186_186405


namespace terminating_decimals_count_l186_186286

theorem terminating_decimals_count :
  (∃ count : ℕ, count = 166 ∧ ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → (∃ m : ℕ, n = 3 * m)) :=
sorry

end terminating_decimals_count_l186_186286


namespace sodas_purchasable_l186_186015

namespace SodaPurchase

variable {D C : ℕ}

theorem sodas_purchasable (D C : ℕ) : (3 * (4 * D) / 5 + 5 * C / 15) = (36 * D + 5 * C) / 15 := 
  sorry

end SodaPurchase

end sodas_purchasable_l186_186015


namespace initial_average_marks_l186_186559

theorem initial_average_marks (A : ℝ) (h1 : 25 * A - 50 = 2450) : A = 100 :=
by
  sorry

end initial_average_marks_l186_186559


namespace emily_furniture_assembly_time_l186_186965

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

end emily_furniture_assembly_time_l186_186965


namespace find_a_l186_186407

open Set

theorem find_a (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {1, 2})
  (hB : B = {-a, a^2 + 3})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = -1 :=
sorry

end find_a_l186_186407


namespace ratio_15_to_1_l186_186448

theorem ratio_15_to_1 (x : ℕ) (h : 15 / 1 = x / 10) : x = 150 := 
by sorry

end ratio_15_to_1_l186_186448


namespace find_n_l186_186205

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n + 4

-- Define the condition a_n = 13
def condition (n : ℕ) : Prop := a n = 13

-- Prove that under this condition, n = 3
theorem find_n (n : ℕ) (h : condition n) : n = 3 :=
by {
  sorry
}

end find_n_l186_186205


namespace triangle_area_l186_186440

noncomputable def area_of_triangle (l1 l2 l3 : ℝ × ℝ → Prop) (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * A.1)

theorem triangle_area :
  let A := (1, 6)
  let B := (-1, 6)
  let C := (0, 4)
  ∀ x y : ℝ, 
    (y = 6 → l1 (x, y)) ∧ 
    (y = 2 * x + 4 → l2 (x, y)) ∧ 
    (y = -2 * x + 4 → l3 (x, y)) →
  area_of_triangle l1 l2 l3 A B C = 1 :=
by 
  intros
  unfold area_of_triangle
  sorry

end triangle_area_l186_186440


namespace intersection_M_N_l186_186846

open Set

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {x | x^2 < 1}

theorem intersection_M_N : M ∩ N = Ico 0 1 := 
sorry

end intersection_M_N_l186_186846


namespace common_sum_l186_186011

theorem common_sum (a l : ℤ) (n r c : ℕ) (S x : ℤ) 
  (h_a : a = -18) 
  (h_l : l = 30) 
  (h_n : n = 49) 
  (h_S : S = (n * (a + l)) / 2) 
  (h_r : r = 7) 
  (h_c : c = 7) 
  (h_sum_eq : r * x = S) :
  x = 42 := 
sorry

end common_sum_l186_186011


namespace children_to_add_l186_186617

def total_guests := 80
def men := 40
def women := men / 2
def adults := men + women
def children := total_guests - adults
def desired_children := 30

theorem children_to_add : (desired_children - children) = 10 := by
  sorry

end children_to_add_l186_186617


namespace billion_to_scientific_notation_l186_186636

theorem billion_to_scientific_notation : 
  (98.36 * 10^9) = 9.836 * 10^10 := 
by
  sorry

end billion_to_scientific_notation_l186_186636


namespace race_time_l186_186297

theorem race_time 
    (v_A v_B t_A t_B : ℝ)
    (h1 : v_A = 1000 / t_A) 
    (h2 : v_B = 940 / t_A)
    (h3 : v_B = 1000 / (t_A + 15)) 
    (h4 : t_B = t_A + 15) :
    t_A = 235 := 
  by
    sorry

end race_time_l186_186297


namespace sliding_window_sash_translation_l186_186521

def is_translation (movement : Type) : Prop := sorry

def ping_pong_ball_movement : Type := sorry
def sliding_window_sash_movement : Type := sorry
def kite_flight_movement : Type := sorry
def basketball_movement : Type := sorry

axiom ping_pong_not_translation : ¬ is_translation ping_pong_ball_movement
axiom kite_not_translation : ¬ is_translation kite_flight_movement
axiom basketball_not_translation : ¬ is_translation basketball_movement
axiom window_sash_is_translation : is_translation sliding_window_sash_movement

theorem sliding_window_sash_translation :
  is_translation sliding_window_sash_movement :=
by 
  exact window_sash_is_translation

end sliding_window_sash_translation_l186_186521


namespace exist_indices_l186_186315

-- Define the sequence and the conditions.
variable (x : ℕ → ℤ)
variable (H1 : x 1 = 1)
variable (H2 : ∀ n : ℕ, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n)

theorem exist_indices (k : ℕ) (hk : 0 < k) :
  ∃ r s : ℕ, x r - x s = k := 
sorry

end exist_indices_l186_186315


namespace general_formula_sum_b_l186_186113

-- Define the arithmetic sequence
def arithmetic_sequence (a d: ℕ) (n: ℕ) := a + (n - 1) * d

-- Given conditions
def a1 : ℕ := 1
def d : ℕ := 2
def a (n : ℕ) : ℕ := arithmetic_sequence a1 d n
def b (n : ℕ) : ℕ := 2 ^ a n

-- Formula for the arithmetic sequence
theorem general_formula (n : ℕ) : a n = 2 * n - 1 := 
by sorry

-- Sum of the first n terms of b_n
theorem sum_b (n : ℕ) : (Finset.range n).sum b = (2 / 3) * (4 ^ n - 1) :=
by sorry

end general_formula_sum_b_l186_186113


namespace geometric_product_Pi8_l186_186449

def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

variables {a : ℕ → ℝ}
variable (h_geom : geometric_sequence a)
variable (h_prod : a 4 * a 5 = 2)

theorem geometric_product_Pi8 :
  (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = 16 :=
by
  sorry

end geometric_product_Pi8_l186_186449


namespace ratio_of_80_pencils_l186_186769

theorem ratio_of_80_pencils (C S : ℝ)
  (CP : ℝ := 80 * C)
  (L : ℝ := 30 * S)
  (SP : ℝ := 80 * S)
  (h : CP = SP + L) :
  CP / SP = 11 / 8 :=
by
  -- Start the proof
  sorry

end ratio_of_80_pencils_l186_186769


namespace sufficient_but_not_necessary_condition_l186_186429

variables {a b : ℝ}

theorem sufficient_but_not_necessary_condition (h₁ : b < -4) : |a| + |b| > 4 :=
by {
    sorry
}

end sufficient_but_not_necessary_condition_l186_186429


namespace tricycles_in_garage_l186_186828

theorem tricycles_in_garage 
    (T : ℕ) 
    (total_bicycles : ℕ := 3) 
    (total_unicycles : ℕ := 7) 
    (bicycle_wheels : ℕ := 2) 
    (tricycle_wheels : ℕ := 3) 
    (unicycle_wheels : ℕ := 1) 
    (total_wheels : ℕ := 25) 
    (eq_wheels : total_bicycles * bicycle_wheels + total_unicycles * unicycle_wheels + T * tricycle_wheels = total_wheels) :
    T = 4 :=
by {
  sorry
}

end tricycles_in_garage_l186_186828


namespace inequality_solution_range_l186_186319

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℤ, 6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) ∧
  (∃ x1 x2 x3 : ℤ, (x1 = 3 ∧ x2 = 4 ∧ x3 = 5) ∧
   (6 - 3 * (x1 : ℝ) < 0 ∧ 2 * (x1 : ℝ) ≤ a) ∧
   (6 - 3 * (x2 : ℝ) < 0 ∧ 2 * (x2 : ℝ) ≤ a) ∧
   (6 - 3 * (x3 : ℝ) < 0 ∧ 2 * (x3 : ℝ) ≤ a) ∧
   (∀ x : ℤ, (6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) → 
     (x = 3 ∨ x = 4 ∨ x = 5)))
  → 10 ≤ a ∧ a < 12 :=
sorry

end inequality_solution_range_l186_186319


namespace population_in_scientific_notation_l186_186352

theorem population_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 1370540000 = a * 10^n ∧ a = 1.37054 ∧ n = 9 :=
by
  sorry

end population_in_scientific_notation_l186_186352


namespace geometric_progression_fourth_term_l186_186744

theorem geometric_progression_fourth_term :
  let a1 := 2^(1/2)
  let a2 := 2^(1/4)
  let a3 := 2^(1/8)
  a4 = 2^(1/16) :=
by
  sorry

end geometric_progression_fourth_term_l186_186744


namespace train_length_correct_l186_186840

noncomputable def length_bridge : ℝ := 300
noncomputable def time_to_cross : ℝ := 45
noncomputable def speed_train_kmh : ℝ := 44

-- Conversion from km/h to m/s
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

-- Total distance covered
noncomputable def total_distance_covered : ℝ := speed_train_ms * time_to_cross

-- Length of the train
noncomputable def length_train : ℝ := total_distance_covered - length_bridge

theorem train_length_correct : abs (length_train - 249.9) < 0.1 :=
by
  sorry

end train_length_correct_l186_186840


namespace distances_product_eq_l186_186408

-- Define the distances
variables (d_ab d_ac d_bc d_ba d_cb d_ca : ℝ)

-- State the theorem
theorem distances_product_eq : d_ab * d_bc * d_ca = d_ac * d_ba * d_cb :=
sorry

end distances_product_eq_l186_186408


namespace lcm_48_180_l186_186659

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  -- Here follows the proof, which is omitted
  sorry

end lcm_48_180_l186_186659


namespace tom_total_spent_correct_l186_186596

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end tom_total_spent_correct_l186_186596


namespace parallelogram_properties_l186_186777

variable {b h : ℕ}

theorem parallelogram_properties
  (hb : b = 20)
  (hh : h = 4) :
  (b * h = 80) ∧ ((b^2 + h^2) = 416) :=
by
  sorry

end parallelogram_properties_l186_186777


namespace log_expression_correct_l186_186654

-- The problem involves logarithms and exponentials
theorem log_expression_correct : 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + (Real.log 25) + Real.exp (Real.log 3) = 5 := 
  by 
    sorry

end log_expression_correct_l186_186654


namespace find_k_l186_186209

theorem find_k
  (S : ℝ)    -- Distance between the village and city
  (x : ℝ)    -- Speed of the truck in km/h
  (y : ℝ)    -- Speed of the car in km/h
  (H1 : 18 = 0.75 * x - 0.75 * x ^ 2 / (x + y))  -- Condition that truck leaving earlier meets 18 km closer to the city
  (H2 : 24 = x * y / (x + y))      -- Intermediate step from solving the first condition
  : (k = 8) :=    -- We need to show that k = 8
  sorry

end find_k_l186_186209


namespace second_number_in_pair_l186_186548

theorem second_number_in_pair (n m : ℕ) (h1 : (n, m) = (57, 58)) (h2 : ∃ (n m : ℕ), n < 1500 ∧ m < 1500 ∧ (n + m) % 5 = 0) : m = 58 :=
by {
  sorry
}

end second_number_in_pair_l186_186548


namespace value_of_abc_l186_186007

theorem value_of_abc : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (ab + c + 10 = 51) ∧ (bc + a + 10 = 51) ∧ (ac + b + 10 = 51) ∧ (a + b + c = 41) :=
by
  sorry

end value_of_abc_l186_186007


namespace frog_paths_l186_186998

theorem frog_paths (n : ℕ) : (∃ e_2n e_2n_minus_1 : ℕ,
  e_2n_minus_1 = 0 ∧
  e_2n = (1 / Real.sqrt 2) * ((2 + Real.sqrt 2) ^ (n - 1) - (2 - Real.sqrt 2) ^ (n - 1))) :=
by {
  sorry
}

end frog_paths_l186_186998


namespace stamp_problem_solution_l186_186765

theorem stamp_problem_solution : ∃ n : ℕ, n > 1 ∧ (∀ m : ℕ, m ≥ 2 * n + 2 → ∃ a b : ℕ, m = n * a + (n + 2) * b) ∧ ∀ x : ℕ, 1 < x ∧ (∀ m : ℕ, m ≥ 2 * x + 2 → ∃ a b : ℕ, m = x * a + (x + 2) * b) → x ≥ 3 :=
by
  sorry

end stamp_problem_solution_l186_186765


namespace no_real_roots_range_l186_186536

theorem no_real_roots_range (a : ℝ) : (¬ ∃ x : ℝ, x^2 + a * x - 4 * a = 0) ↔ (-16 < a ∧ a < 0) := by
  sorry

end no_real_roots_range_l186_186536


namespace hyperbola_condition_l186_186595

theorem hyperbola_condition (m : ℝ) : 
  (∃ a b : ℝ, a = m + 4 ∧ b = m - 3 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)) ↔ m > 3 :=
sorry

end hyperbola_condition_l186_186595


namespace arithmetic_mean_of_fractions_l186_186268

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  (a + b) / 2 = 11 / 16 :=
by 
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  show (a + b) / 2 = 11 / 16
  sorry

end arithmetic_mean_of_fractions_l186_186268


namespace side_length_S2_l186_186597

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end side_length_S2_l186_186597


namespace solve_inequality_l186_186824

theorem solve_inequality (a : ℝ) (ha_pos : 0 < a) :
  (if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
   else if a = 1 then ∅
   else {x : ℝ | 1 / a < x ∧ x < 1}) =
  {x : ℝ | ax^2 - (a + 1) * x + 1 < 0} :=
by sorry

end solve_inequality_l186_186824


namespace abs_x_gt_1_iff_x_sq_minus1_gt_0_l186_186226

theorem abs_x_gt_1_iff_x_sq_minus1_gt_0 (x : ℝ) : (|x| > 1) ↔ (x^2 - 1 > 0) := by
  sorry

end abs_x_gt_1_iff_x_sq_minus1_gt_0_l186_186226


namespace cubeRootThree_expression_value_l186_186458

-- Define the approximate value of cube root of 3
def cubeRootThree : ℝ := 1.442

-- Lean theorem statement
theorem cubeRootThree_expression_value :
  cubeRootThree - 3 * cubeRootThree - 98 * cubeRootThree = -144.2 := by
  sorry

end cubeRootThree_expression_value_l186_186458


namespace common_ratio_of_infinite_geometric_series_l186_186411

noncomputable def first_term : ℝ := 500
noncomputable def series_sum : ℝ := 3125

theorem common_ratio_of_infinite_geometric_series (r : ℝ) (h₀ : first_term / (1 - r) = series_sum) : 
  r = 0.84 := 
by
  sorry

end common_ratio_of_infinite_geometric_series_l186_186411


namespace pastries_sold_value_l186_186037

-- Define the number of cakes sold and the relationship between cakes and pastries
def number_of_cakes_sold := 78
def pastries_sold (C : Nat) := C + 76

-- State the theorem we want to prove
theorem pastries_sold_value : pastries_sold number_of_cakes_sold = 154 := by
  sorry

end pastries_sold_value_l186_186037


namespace juan_original_number_l186_186620

theorem juan_original_number (x : ℝ) (h : (3 * (x + 3) - 4) / 2 = 10) : x = 5 :=
by
  sorry

end juan_original_number_l186_186620


namespace floor_plus_x_eq_17_over_4_l186_186820

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end floor_plus_x_eq_17_over_4_l186_186820


namespace min_even_integers_six_l186_186666

theorem min_even_integers_six (x y a b m n : ℤ) 
  (h1 : x + y = 30) 
  (h2 : x + y + a + b = 50) 
  (h3 : x + y + a + b + m + n = 70) 
  (hm_even : Even m) 
  (hn_even: Even n) : 
  ∃ k, (0 ≤ k ∧ k ≤ 6) ∧ (∀ e, (e = m ∨ e = n) → ∃ j, (j = 2)) :=
by
  sorry

end min_even_integers_six_l186_186666


namespace more_knights_than_liars_l186_186487

theorem more_knights_than_liars 
  (k l : Nat)
  (h1 : (k + l) % 2 = 1)
  (h2 : ∀ i : Nat, i < k → ∃ j : Nat, j < l)
  (h3 : ∀ j : Nat, j < l → ∃ i : Nat, i < k) :
  k > l := 
sorry

end more_knights_than_liars_l186_186487


namespace percentage_of_number_l186_186277

theorem percentage_of_number (n : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * n = 16) : 0.4 * n = 192 :=
by 
  sorry

end percentage_of_number_l186_186277


namespace potato_bag_weight_l186_186652

theorem potato_bag_weight :
  ∃ w : ℝ, w = 16 / (w / 4) ∧ w = 16 := 
by
  sorry

end potato_bag_weight_l186_186652


namespace find_x_l186_186750

theorem find_x : 
  ∀ x : ℝ, (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1 / 2 := 
by 
  sorry

end find_x_l186_186750


namespace inequality_solution_l186_186554

theorem inequality_solution (x : ℝ) : (x^3 - 10 * x^2 > -25 * x) ↔ (0 < x ∧ x < 5) ∨ (5 < x) := 
sorry

end inequality_solution_l186_186554


namespace junior_score_is_95_l186_186560

theorem junior_score_is_95:
  ∀ (n j s : ℕ) (x avg_total avg_seniors : ℕ),
    n = 20 →
    j = n * 15 / 100 →
    s = n * 85 / 100 →
    avg_total = 78 →
    avg_seniors = 75 →
    (j * x + s * avg_seniors) / n = avg_total →
    x = 95 :=
by
  sorry

end junior_score_is_95_l186_186560


namespace number_of_cows_l186_186702

theorem number_of_cows (n : ℝ) (h1 : n / 2 + n / 4 + n / 5 + 7 = n) : n = 140 := 
sorry

end number_of_cows_l186_186702


namespace tangent_from_origin_l186_186188

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (6, 14)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a function that computes the length of the tangent from O to the circle passing through A, B, and C
noncomputable def tangent_length : ℝ :=
 sorry -- Placeholder for the actual calculation

-- The theorem we need to prove: The length of the tangent from O to the circle passing through A, B, and C is as calculated
theorem tangent_from_origin (L : ℕ) : 
  tangent_length = L := 
 sorry -- Placeholder for the proof

end tangent_from_origin_l186_186188


namespace find_other_cat_weight_l186_186779

variable (cat1 cat2 dog : ℕ)

def weight_of_other_cat (cat1 cat2 dog : ℕ) : Prop :=
  cat1 = 7 ∧
  dog = 34 ∧
  dog = 2 * (cat1 + cat2) ∧
  cat2 = 10

theorem find_other_cat_weight (cat1 : ℕ) (cat2 : ℕ) (dog : ℕ) :
  weight_of_other_cat cat1 cat2 dog := by
  sorry

end find_other_cat_weight_l186_186779


namespace circle_center_l186_186771

theorem circle_center (x y : ℝ) (h : x^2 - 4 * x + y^2 - 6 * y - 12 = 0) : (x, y) = (2, 3) :=
sorry

end circle_center_l186_186771


namespace hog_cat_problem_l186_186685

theorem hog_cat_problem (hogs cats : ℕ)
  (hogs_eq : hogs = 75)
  (hogs_cats_relation : hogs = 3 * cats)
  : 5 < (6 / 10) * cats - 5 := 
by
  sorry

end hog_cat_problem_l186_186685


namespace sqrt_49_times_sqrt_25_l186_186688

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 5 * Real.sqrt 7 :=
by
  sorry

end sqrt_49_times_sqrt_25_l186_186688


namespace hours_sunday_correct_l186_186012

-- Definitions of given conditions
def hours_saturday : ℕ := 6
def total_hours : ℕ := 9

-- The question translated to a proof problem
theorem hours_sunday_correct : total_hours - hours_saturday = 3 := 
by
  -- The proof is skipped and replaced by sorry
  sorry

end hours_sunday_correct_l186_186012


namespace maximize_root_product_l186_186886

theorem maximize_root_product :
  (∃ k : ℝ, ∀ x : ℝ, 6 * x^2 - 5 * x + k = 0 ∧ (25 - 24 * k ≥ 0)) →
  ∃ k : ℝ, k = 25 / 24 :=
by
  sorry

end maximize_root_product_l186_186886


namespace initial_roses_in_vase_l186_186683

theorem initial_roses_in_vase (current_roses : ℕ) (added_roses : ℕ) (total_garden_roses : ℕ) (initial_roses : ℕ) :
  current_roses = 20 → added_roses = 13 → total_garden_roses = 59 → initial_roses = current_roses - added_roses → 
  initial_roses = 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  sorry

end initial_roses_in_vase_l186_186683


namespace max_total_weight_of_chocolates_l186_186154

theorem max_total_weight_of_chocolates 
  (A B C : ℕ)
  (hA : A ≤ 100)
  (hBC : B - C ≤ 100)
  (hC : C ≤ 100)
  (h_distribute : A ≤ 100 ∧ (B - C) ≤ 100)
  : (A + B = 300) :=
by 
  sorry

end max_total_weight_of_chocolates_l186_186154


namespace parallel_to_l3_through_P_perpendicular_to_l3_through_P_l186_186373

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P := (1, 1)

-- Define the parallel line equation to l3 passing through P
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the perpendicular line equation to l3 passing through P
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Prove the parallel line through P is 2x + y - 3 = 0
theorem parallel_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (parallel_line 1 1) := 
by 
  sorry

-- Prove the perpendicular line through P is x - 2y + 1 = 0
theorem perpendicular_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (perpendicular_line 1 1) := 
by 
  sorry

end parallel_to_l3_through_P_perpendicular_to_l3_through_P_l186_186373


namespace carlos_marbles_l186_186609

theorem carlos_marbles:
  ∃ M, M > 1 ∧ 
       M % 5 = 1 ∧ 
       M % 7 = 1 ∧ 
       M % 11 = 1 ∧ 
       M % 4 = 2 ∧ 
       M = 386 := by
  sorry

end carlos_marbles_l186_186609


namespace grade_assignments_count_l186_186935

theorem grade_assignments_count (n : ℕ) (g : ℕ) (h : n = 15) (k : g = 4) : g^n = 1073741824 :=
by
  sorry

end grade_assignments_count_l186_186935


namespace multiple_of_eight_l186_186349

theorem multiple_of_eight (x y : ℤ) (h : ∀ (k : ℤ), 24 + 16 * k = 8) : ∃ (k : ℤ), x + 16 * y = 8 * k := 
by
  sorry

end multiple_of_eight_l186_186349


namespace math_problem_l186_186539

theorem math_problem :
  (Int.ceil ((18: ℚ) / 5 * (-25 / 4)) - Int.floor ((18 / 5) * Int.floor (-25 / 4))) = 4 := 
by
  sorry

end math_problem_l186_186539


namespace sum_of_cubes_l186_186832

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l186_186832


namespace real_roots_of_polynomial_l186_186096

theorem real_roots_of_polynomial :
  {x : ℝ | (x^4 - 4*x^3 + 5*x^2 - 2*x + 2) = 0} = {1, -1} :=
sorry

end real_roots_of_polynomial_l186_186096


namespace range_of_a_l186_186119

-- Define the propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 0 ≤ x → x ≤ 1 → a ≥ Real.exp x
def q (a : ℝ) := ∃ x : ℝ, x^2 + 4 * x + a = 0

-- The proof statement
theorem range_of_a (a : ℝ) : (p a ∧ q a) → a ∈ Set.Icc (Real.exp 1) 4 := by
  intro h
  sorry

end range_of_a_l186_186119


namespace problem_statement_l186_186359

theorem problem_statement (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6) :
  (a * b / c) + (b * c / a) + (c * a / b) = 49 / 6 := 
by sorry

end problem_statement_l186_186359


namespace alcohol_water_ratio_l186_186067

variable {r s V1 : ℝ}

theorem alcohol_water_ratio 
  (h1 : r > 0) 
  (h2 : s > 0) 
  (h3 : V1 > 0) :
  let alcohol_in_JarA := 2 * r * V1 / (r + 1) + V1
  let water_in_JarA := 2 * V1 / (r + 1)
  let alcohol_in_JarB := 3 * s * V1 / (s + 1)
  let water_in_JarB := 3 * V1 / (s + 1)
  let total_alcohol := alcohol_in_JarA + alcohol_in_JarB
  let total_water := water_in_JarA + water_in_JarB
  (total_alcohol / total_water) = 
  ((2 * r / (r + 1) + 1 + 3 * s / (s + 1)) / (2 / (r + 1) + 3 / (s + 1))) :=
by
  sorry

end alcohol_water_ratio_l186_186067


namespace bricks_needed_for_wall_l186_186775

noncomputable def brick_volume (length : ℝ) (height : ℝ) (thickness : ℝ) : ℝ :=
  length * height * thickness

noncomputable def wall_volume (length : ℝ) (height : ℝ) (average_thickness : ℝ) : ℝ :=
  length * height * average_thickness

noncomputable def number_of_bricks (wall_vol : ℝ) (brick_vol : ℝ) : ℝ :=
  wall_vol / brick_vol

theorem bricks_needed_for_wall : 
  let length_wall := 800
  let height_wall := 660
  let avg_thickness_wall := (25 + 22.5) / 2 -- in cm
  let length_brick := 25
  let height_brick := 11.25
  let thickness_brick := 6
  let mortar_thickness := 1

  let adjusted_length_brick := length_brick + mortar_thickness
  let adjusted_height_brick := height_brick + mortar_thickness

  let volume_wall := wall_volume length_wall height_wall avg_thickness_wall
  let volume_brick_with_mortar := brick_volume adjusted_length_brick adjusted_height_brick thickness_brick

  number_of_bricks volume_wall volume_brick_with_mortar = 6565 :=
by
  sorry

end bricks_needed_for_wall_l186_186775


namespace trader_profit_l186_186787

noncomputable def profit_percentage (P : ℝ) : ℝ :=
  let purchased_price := 0.72 * P
  let market_increase := 1.05 * purchased_price
  let expenses := 0.08 * market_increase
  let net_price := market_increase - expenses
  let first_sale_price := 1.50 * net_price
  let final_sale_price := 1.25 * first_sale_price
  let profit := final_sale_price - P
  (profit / P) * 100

theorem trader_profit
  (P : ℝ) 
  (hP : 0 < P) :
  profit_percentage P = 30.41 :=
by
  sorry

end trader_profit_l186_186787


namespace length_of_second_train_l186_186293

theorem length_of_second_train 
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_seconds : ℝ)
  (same_direction : Bool) : 
  length_first_train = 380 ∧ 
  speed_first_train_kmph = 72 ∧ 
  speed_second_train_kmph = 36 ∧ 
  time_seconds = 91.9926405887529 ∧ 
  same_direction = tt → 
  ∃ L2 : ℝ, L2 = 539.93 := by
  intro h
  rcases h with ⟨hf, sf, ss, ts, sd⟩
  use 539.926405887529 -- exact value obtained in the solution
  sorry

end length_of_second_train_l186_186293


namespace sum_of_digits_S_l186_186859

-- Define S as 10^2021 - 2021
def S : ℕ := 10^2021 - 2021

-- Define function to calculate sum of digits of a given number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum 

theorem sum_of_digits_S :
  sum_of_digits S = 18185 :=
sorry

end sum_of_digits_S_l186_186859


namespace log_addition_l186_186852

theorem log_addition (log_base_10 : ℝ → ℝ) (a b : ℝ) (h_base_10_log : log_base_10 10 = 1) :
  log_base_10 2 + log_base_10 5 = 1 :=
by
  sorry

end log_addition_l186_186852


namespace disproving_iff_l186_186905

theorem disproving_iff (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : (a^2 > b^2) ∧ ¬(a > b) :=
by
  sorry

end disproving_iff_l186_186905


namespace find_a_g_range_l186_186079

noncomputable def f (x a : ℝ) : ℝ := x^2 + 4 * a * x + 2 * a + 6
noncomputable def g (a : ℝ) : ℝ := 2 - a * |a - 1|

theorem find_a (x a : ℝ) :
  (∀ x, f x a ≥ 0) ∧ (∀ x, f x a = 0 → x^2 + 4 * a * x + 2 * a + 6 = 0) ↔ (a = -1 ∨ a = 3 / 2) :=
  sorry

theorem g_range :
  (∀ x, f x a ≥ 0) ∧ (-1 ≤ a ∧ a ≤ 3/2) → (∀ a, (5 / 4 ≤ g a ∧ g a ≤ 4)) :=
  sorry

end find_a_g_range_l186_186079


namespace composite_numbers_equal_l186_186789

-- Define composite natural number
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

-- Define principal divisors
def principal_divisors (n : ℕ) (principal1 principal2 : ℕ) : Prop :=
  is_composite n ∧ 
  (1 < principal1 ∧ principal1 < n) ∧ 
  (1 < principal2 ∧ principal2 < n) ∧
  principal1 * principal2 = n

-- Problem statement to prove
theorem composite_numbers_equal (a b p1 p2 : ℕ) :
  is_composite a → is_composite b →
  principal_divisors a p1 p2 → principal_divisors b p1 p2 →
  a = b :=
by
  sorry

end composite_numbers_equal_l186_186789


namespace investment_ratio_proof_l186_186210

noncomputable def investment_ratio {A_invest B_invest C_invest : ℝ} (profit total_profit : ℝ) (A_times_B : ℝ) : ℝ :=
  C_invest / (A_times_B * B_invest + B_invest + C_invest)

theorem investment_ratio_proof (A_invest B_invest C_invest : ℝ)
  (profit total_profit : ℝ) (A_times_B : ℝ) 
  (h_profit : total_profit = 55000)
  (h_C_share : profit = 15000.000000000002)
  (h_A_times_B : A_times_B = 3)
  (h_ratio_eq : A_times_B * B_invest + B_invest + C_invest = 11 * B_invest / 3) :
  (A_invest / C_invest = 2) :=
by
  sorry

end investment_ratio_proof_l186_186210


namespace common_tangents_l186_186335

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y = 0
def circle2_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

theorem common_tangents :
  ∃ (n : ℕ), n = 4 ∧ 
    (∀ (L : ℝ → ℝ → Prop), 
      (∀ x y, L x y → circle1_eqn x y ∧ circle2_eqn x y) → n = 4) := 
sorry

end common_tangents_l186_186335


namespace inequality_gt_zero_l186_186044

theorem inequality_gt_zero (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 :=
  sorry

end inequality_gt_zero_l186_186044


namespace percentage_increase_in_items_sold_l186_186515

-- Definitions
variables (P N M : ℝ)
-- Given conditions:
-- The new price of an item
def new_price := P * 0.90
-- The relationship between incomes
def income_increase := (P * 0.90) * M = P * N * 1.125

-- The problem statement
theorem percentage_increase_in_items_sold (h : income_increase P N M) :
  M = N * 1.25 :=
sorry

end percentage_increase_in_items_sold_l186_186515


namespace diameter_of_circle_l186_186705

theorem diameter_of_circle (A : ℝ) (h : A = 100 * Real.pi) : ∃ d : ℝ, d = 20 :=
by
  sorry

end diameter_of_circle_l186_186705


namespace total_blue_balloons_l186_186648

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l186_186648


namespace equilateral_triangle_perimeter_l186_186980

theorem equilateral_triangle_perimeter (s : ℕ) (b : ℕ) (h1 : 40 = 2 * s + b) (h2 : b = 10) : 3 * s = 45 :=
by {
  sorry
}

end equilateral_triangle_perimeter_l186_186980


namespace solution_set_f_l186_186330

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x^(1/2) - 1

theorem solution_set_f (x : ℝ) (hx_pos : x > 0) : 
  f x > f (2 * x - 4) ↔ 2 < x ∧ x < 4 :=
sorry

end solution_set_f_l186_186330


namespace required_equation_l186_186074

-- Define the given lines
def line1 (x y : ℝ) : Prop := 2 * x - y = 0
def line2 (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Define the equation to be proven for the line through the intersection point and perpendicular to perp_line
def required_line (x y : ℝ) : Prop := x - 2 * y + 6 = 0

-- Define the predicate that states a point (2, 4) lies on line1 and line2
def point_intersect (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The main theorem to be proven in Lean 4
theorem required_equation : 
  point_intersect 2 4 ∧ perp_line 2 4 → required_line 2 4 := by
  sorry

end required_equation_l186_186074


namespace chef_pillsbury_flour_l186_186878

theorem chef_pillsbury_flour (x : ℕ) (h : 7 / 2 = 28 / x) : x = 8 := sorry

end chef_pillsbury_flour_l186_186878


namespace gigi_mushrooms_l186_186443

-- Define the conditions
def pieces_per_mushroom := 4
def kenny_pieces := 38
def karla_pieces := 42
def remaining_pieces := 8

-- Main theorem
theorem gigi_mushrooms : (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry

end gigi_mushrooms_l186_186443


namespace corner_coloring_condition_l186_186531

theorem corner_coloring_condition 
  (n : ℕ) 
  (h1 : n ≥ 5) 
  (board : ℕ → ℕ → Prop) -- board(i, j) = true if cell (i, j) is black, false if white
  (h2 : ∀ i j, board i j = board (i + 1) j → board (i + 2) j = board (i + 1) j → ¬(board i j = board (i + 2) j)) -- row condition
  (h3 : ∀ i j, board i j = board i (j + 1) → board i (j + 2) = board i (j + 1) → ¬(board i j = board i (j + 2))) -- column condition
  (h4 : ∀ i j, board i j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board i j = board (i + 2) (j + 2))) -- diagonal condition
  (h5 : ∀ i j, board (i + 2) j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board (i + 2) j = board (i + 2) (j + 2))) -- anti-diagonal condition
  : ∀ i j, i + 2 < n ∧ j + 2 < n → ((board i j ∧ board (i + 2) (j + 2)) ∨ (board i (j + 2) ∧ board (i + 2) j)) :=
sorry

end corner_coloring_condition_l186_186531


namespace cos_relation_l186_186784

theorem cos_relation 
  (a b c A B C : ℝ)
  (h1 : a = b * Real.cos C + c * Real.cos B)
  (h2 : b = c * Real.cos A + a * Real.cos C)
  (h3 : c = a * Real.cos B + b * Real.cos A)
  (h_abc_nonzero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 + 2 * Real.cos A * Real.cos B * Real.cos C = 1 :=
sorry

end cos_relation_l186_186784


namespace tod_trip_time_l186_186204

noncomputable def total_time (d1 d2 d3 d4 s1 s2 s3 s4 : ℝ) : ℝ :=
  d1 / s1 + d2 / s2 + d3 / s3 + d4 / s4

theorem tod_trip_time :
  total_time 55 95 30 75 40 50 20 60 = 6.025 :=
by 
  sorry

end tod_trip_time_l186_186204


namespace common_root_polynomials_l186_186403

theorem common_root_polynomials (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end common_root_polynomials_l186_186403


namespace tank_min_cost_l186_186833

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

end tank_min_cost_l186_186833


namespace margin_in_terms_of_ratio_l186_186627

variable (S m : ℝ)

theorem margin_in_terms_of_ratio (h1 : M = (1/m) * S) (h2 : C = S - M) : M = (1/m) * S :=
sorry

end margin_in_terms_of_ratio_l186_186627


namespace fifth_largest_divisor_of_1209600000_is_75600000_l186_186882

theorem fifth_largest_divisor_of_1209600000_is_75600000 :
  let n : ℤ := 1209600000
  let fifth_largest_divisor : ℤ := 75600000
  n = 2^10 * 5^5 * 3 * 503 →
  fifth_largest_divisor = n / 2^5 :=
by
  sorry

end fifth_largest_divisor_of_1209600000_is_75600000_l186_186882


namespace ratio_of_rises_l186_186896

noncomputable def radius_narrower_cone : ℝ := 4
noncomputable def radius_wider_cone : ℝ := 8
noncomputable def sphere_radius : ℝ := 2

noncomputable def height_ratio (h1 h2 : ℝ) : Prop := h1 = 4 * h2

noncomputable def volume_displacement := (4 / 3) * Real.pi * (sphere_radius^3)

noncomputable def new_height_narrower (h1 : ℝ) : ℝ := h1 + (volume_displacement / ((Real.pi * (radius_narrower_cone^2))))

noncomputable def new_height_wider (h2 : ℝ) : ℝ := h2 + (volume_displacement / ((Real.pi * (radius_wider_cone^2))))

theorem ratio_of_rises (h1 h2 : ℝ) (hr : height_ratio h1 h2) :
  (new_height_narrower h1 - h1) / (new_height_wider h2 - h2) = 4 :=
sorry

end ratio_of_rises_l186_186896


namespace minimal_APR_bank_A_l186_186070

def nominal_interest_rate_A : Float := 0.05
def nominal_interest_rate_B : Float := 0.055
def nominal_interest_rate_C : Float := 0.06

def compounding_periods_A : ℕ := 4
def compounding_periods_B : ℕ := 2
def compounding_periods_C : ℕ := 12

def effective_annual_rate (nom_rate : Float) (n : ℕ) : Float :=
  (1 + nom_rate / n.toFloat)^n.toFloat - 1

def APR_A := effective_annual_rate nominal_interest_rate_A compounding_periods_A
def APR_B := effective_annual_rate nominal_interest_rate_B compounding_periods_B
def APR_C := effective_annual_rate nominal_interest_rate_C compounding_periods_C

theorem minimal_APR_bank_A :
  APR_A < APR_B ∧ APR_A < APR_C ∧ APR_A = 0.050945 :=
by
  sorry

end minimal_APR_bank_A_l186_186070


namespace fourth_person_height_l186_186855

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

end fourth_person_height_l186_186855


namespace y_decreases_as_x_increases_l186_186767

-- Define the function y = 7 - x
def my_function (x : ℝ) : ℝ := 7 - x

-- Prove that y decreases as x increases
theorem y_decreases_as_x_increases : ∀ x1 x2 : ℝ, x1 < x2 → my_function x1 > my_function x2 := by
  intro x1 x2 h
  unfold my_function
  sorry

end y_decreases_as_x_increases_l186_186767


namespace kelly_initial_apples_l186_186890

theorem kelly_initial_apples : ∀ (T P I : ℕ), T = 105 → P = 49 → I + P = T → I = 56 :=
by
  intros T P I ht hp h
  rw [ht, hp] at h
  linarith

end kelly_initial_apples_l186_186890


namespace polar_distance_l186_186629

/-
Problem:
In the polar coordinate system, it is known that A(2, π / 6), B(4, 5π / 6). Then, the distance between points A and B is 2√7.

Conditions:
- Point A in polar coordinates: A(2, π / 6)
- Point B in polar coordinates: B(4, 5π / 6)
-/

/-- The distance between two points in the polar coordinate system A(2, π / 6) and B(4, 5π / 6) is 2√7. -/
theorem polar_distance :
  let A_ρ := 2
  let A_θ := π / 6
  let B_ρ := 4
  let B_θ := 5 * π / 6
  let A_x := A_ρ * Real.cos A_θ
  let A_y := A_ρ * Real.sin A_θ
  let B_x := B_ρ * Real.cos B_θ
  let B_y := B_ρ * Real.sin B_θ
  let distance := Real.sqrt ((B_x - A_x)^2 + (B_y - A_y)^2)
  distance = 2 * Real.sqrt 7 := by
  sorry

end polar_distance_l186_186629


namespace james_coffee_weekdays_l186_186201

theorem james_coffee_weekdays :
  ∃ (c d : ℕ) (k : ℤ), (c + d = 5) ∧ 
                      (3 * c + 2 * d + 10 = k / 3) ∧ 
                      (k % 3 = 0) ∧ 
                      c = 2 :=
by 
  sorry

end james_coffee_weekdays_l186_186201


namespace expression_undefined_count_l186_186575

theorem expression_undefined_count (x : ℝ) :
  ∃! x, (x - 1) * (x + 3) * (x - 3) = 0 :=
sorry

end expression_undefined_count_l186_186575


namespace second_athlete_high_jump_eq_eight_l186_186138

theorem second_athlete_high_jump_eq_eight :
  let first_athlete_long_jump := 26
  let first_athlete_triple_jump := 30
  let first_athlete_high_jump := 7
  let second_athlete_long_jump := 24
  let second_athlete_triple_jump := 34
  let winner_average_jump := 22
  (first_athlete_long_jump + first_athlete_triple_jump + first_athlete_high_jump) / 3 < winner_average_jump →
  ∃ (second_athlete_high_jump : ℝ), 
    second_athlete_high_jump = 
    (winner_average_jump * 3 - (second_athlete_long_jump + second_athlete_triple_jump)) ∧ 
    second_athlete_high_jump = 8 :=
by
  intros 
  sorry

end second_athlete_high_jump_eq_eight_l186_186138


namespace units_digit_L_L_15_l186_186590

def Lucas (n : ℕ) : ℕ :=
match n with
| 0 => 2
| 1 => 1
| n + 2 => Lucas n + Lucas (n + 1)

theorem units_digit_L_L_15 : (Lucas (Lucas 15)) % 10 = 7 := by
  sorry

end units_digit_L_L_15_l186_186590


namespace sufficient_not_necessary_condition_l186_186519

-- Definition of the proposition p
def prop_p (m : ℝ) := ∀ x : ℝ, x^2 - 4 * x + 2 * m ≥ 0

-- Statement of the proof problem
theorem sufficient_not_necessary_condition (m : ℝ) : 
  (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 → m ≥ 2) ∧ (m ≥ 2 → prop_p m) → (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 ↔ prop_p m) :=
sorry

end sufficient_not_necessary_condition_l186_186519


namespace family_of_four_children_includes_one_boy_one_girl_l186_186461

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - ((1/2)^4 + (1/2)^4)

theorem family_of_four_children_includes_one_boy_one_girl :
  probability_at_least_one_boy_and_one_girl = 7 / 8 :=
by
  sorry

end family_of_four_children_includes_one_boy_one_girl_l186_186461


namespace energy_of_first_particle_l186_186948

theorem energy_of_first_particle
  (E_1 E_2 E_3 : ℤ)
  (h1 : E_1^2 - E_2^2 - E_3^2 + E_1 * E_2 = 5040)
  (h2 : E_1^2 + 2 * E_2^2 + 2 * E_3^2 - 2 * E_1 * E_2 - E_1 * E_3 - E_2 * E_3 = -4968)
  (h3 : 0 < E_3)
  (h4 : E_3 ≤ E_2)
  (h5 : E_2 ≤ E_1) : E_1 = 12 :=
by sorry

end energy_of_first_particle_l186_186948


namespace min_sum_of_integers_cauchy_schwarz_l186_186493

theorem min_sum_of_integers_cauchy_schwarz :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  (1 / x + 4 / y + 9 / z = 1) ∧ 
  ((x + y + z) = 36) :=
  sorry

end min_sum_of_integers_cauchy_schwarz_l186_186493


namespace workshop_participants_problem_l186_186009

variable (WorkshopSize : ℕ) 
variable (LeftHanded : ℕ) 
variable (RockMusicLovers : ℕ) 
variable (RightHandedDislikeRock : ℕ) 
variable (Under25 : ℕ)
variable (RightHandedUnder25RockMusicLovers : ℕ)
variable (y : ℕ)

theorem workshop_participants_problem
  (h1 : WorkshopSize = 30)
  (h2 : LeftHanded = 12)
  (h3 : RockMusicLovers = 18)
  (h4 : RightHandedDislikeRock = 5)
  (h5 : Under25 = 9)
  (h6 : RightHandedUnder25RockMusicLovers = 3)
  (h7 : WorkshopSize = LeftHanded + (WorkshopSize - LeftHanded))
  (h8 : WorkshopSize - LeftHanded = RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + (WorkshopSize - LeftHanded - RightHandedDislikeRock - RightHandedUnder25RockMusicLovers - y))
  (h9 : WorkshopSize - (RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + Under25 - y - (RockMusicLovers - y)) - (LeftHanded - y) = WorkshopSize) :
  y = 5 := by
  sorry

end workshop_participants_problem_l186_186009


namespace value_of_g_neg3_l186_186045

def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem value_of_g_neg3 : g (-3) = 4 :=
by
  sorry

end value_of_g_neg3_l186_186045


namespace min_y_ellipse_l186_186915

-- Defining the ellipse equation
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 49) + ((y - 3)^2 / 25) = 1

-- Problem statement: Prove that the smallest y-coordinate is -2
theorem min_y_ellipse : 
  ∀ x y, ellipse x y → y ≥ -2 :=
sorry

end min_y_ellipse_l186_186915


namespace houses_with_pools_l186_186000

theorem houses_with_pools (total G overlap N P : ℕ) 
  (h1 : total = 70) 
  (h2 : G = 50) 
  (h3 : overlap = 35) 
  (h4 : N = 15) 
  (h_eq : total = G + P - overlap + N) : 
  P = 40 := by
  sorry

end houses_with_pools_l186_186000


namespace value_of_n_l186_186207

theorem value_of_n (n : ℤ) :
  (∀ x : ℤ, (x + n) * (x + 2) = x^2 + 2 * x + n * x + 2 * n → 2 + n = 0) → n = -2 := 
by
  intro h
  have h1 := h 0
  sorry

end value_of_n_l186_186207


namespace negation_of_existence_l186_186071

theorem negation_of_existence (x : ℝ) (hx : 0 < x) : ¬ (∃ x_0 : ℝ, 0 < x_0 ∧ Real.log x_0 = x_0 - 1) 
  → ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by sorry

end negation_of_existence_l186_186071


namespace parallelogram_area_example_l186_186143

def point := (ℚ × ℚ)
def parallelogram_area (A B C D : point) : ℚ :=
  let base := B.1 - A.1
  let height := C.2 - A.2
  base * height

theorem parallelogram_area_example : 
  parallelogram_area (1, 1) (7, 1) (4, 9) (10, 9) = 48 := by
  sorry

end parallelogram_area_example_l186_186143


namespace min_distance_to_line_l186_186259

theorem min_distance_to_line : 
  let A := 5
  let B := -3
  let C := 4
  let d (x₀ y₀ : ℤ) := (abs (A * x₀ + B * y₀ + C) : ℝ) / (Real.sqrt (A ^ 2 + B ^ 2))
  ∃ (x₀ y₀ : ℤ), d x₀ y₀ = Real.sqrt 34 / 85 := 
by 
  sorry

end min_distance_to_line_l186_186259


namespace depth_of_tunnel_l186_186173

theorem depth_of_tunnel (a b area : ℝ) (h := (2 * area) / (a + b)) (ht : a = 15) (hb : b = 5) (ha : area = 400) :
  h = 40 :=
by
  sorry

end depth_of_tunnel_l186_186173


namespace cat_collars_needed_l186_186863

-- Define the given constants
def nylon_per_dog_collar : ℕ := 18
def nylon_per_cat_collar : ℕ := 10
def total_nylon : ℕ := 192
def dog_collars : ℕ := 9

-- Compute the number of cat collars needed
theorem cat_collars_needed : (total_nylon - (dog_collars * nylon_per_dog_collar)) / nylon_per_cat_collar = 3 :=
by
  sorry

end cat_collars_needed_l186_186863


namespace solve_for_x_l186_186830

theorem solve_for_x (x : ℝ) (h : 3375 = (1 / 4) * x + 144) : x = 12924 :=
by
  sorry

end solve_for_x_l186_186830


namespace find_f_l186_186387

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 2) 
  (h₁ : ∀ x y : ℝ, f (x * y) = f ((x^2 + y^2) / 2) + (x + y)^2) :
  ∀ x : ℝ, f x = 2 - 2 * x :=
sorry

end find_f_l186_186387


namespace speed_of_first_train_l186_186736

theorem speed_of_first_train
  (v : ℝ)
  (d : ℝ)
  (distance_between_stations : ℝ := 450)
  (speed_of_second_train : ℝ := 25)
  (additional_distance_first_train : ℝ := 50)
  (meet_time_condition : d / v = (d - additional_distance_first_train) / speed_of_second_train)
  (total_distance_condition : d + (d - additional_distance_first_train) = distance_between_stations) :
  v = 31.25 :=
by {
  sorry
}

end speed_of_first_train_l186_186736


namespace henry_total_fee_8_bikes_l186_186991

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l186_186991


namespace two_real_solutions_only_if_c_zero_l186_186717

theorem two_real_solutions_only_if_c_zero (x y c : ℝ) :
  (|x + y| = 99 ∧ |x - y| = c → (∃! (x y : ℝ), |x + y| = 99 ∧ |x - y| = c)) ↔ c = 0 :=
by
  sorry

end two_real_solutions_only_if_c_zero_l186_186717


namespace trigonometric_product_eq_l186_186704

open Real

theorem trigonometric_product_eq :
  3.420 * (sin (10 * pi / 180)) * (sin (20 * pi / 180)) * (sin (30 * pi / 180)) *
  (sin (40 * pi / 180)) * (sin (50 * pi / 180)) * (sin (60 * pi / 180)) *
  (sin (70 * pi / 180)) * (sin (80 * pi / 180)) = 3 / 256 := 
sorry

end trigonometric_product_eq_l186_186704


namespace zeros_of_g_l186_186362

theorem zeros_of_g (a b : ℝ) (h : 2 * a + b = 0) :
  (∃ x : ℝ, (b * x^2 - a * x = 0) ∧ (x = 0 ∨ x = -1 / 2)) :=
by
  sorry

end zeros_of_g_l186_186362


namespace circumradius_of_triangle_ABC_l186_186439

noncomputable def circumradius (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * K)

theorem circumradius_of_triangle_ABC :
  (circumradius 12 10 7 = 6) :=
by
  sorry

end circumradius_of_triangle_ABC_l186_186439


namespace units_digit_7_pow_1995_l186_186406

theorem units_digit_7_pow_1995 : 
  ∃ a : ℕ, a = 3 ∧ ∀ n : ℕ, (7^n % 10 = a) → ((n % 4) + 1 = 3) := 
by
  sorry

end units_digit_7_pow_1995_l186_186406


namespace triangle_angles_l186_186604

-- Defining a structure for a triangle with angles
structure Triangle :=
(angleA angleB angleC : ℝ)

-- Define the condition for the triangle mentioned in the problem
def triangle_condition (t : Triangle) : Prop :=
  ∃ (α : ℝ), α = 22.5 ∧ t.angleA = 90 ∧ t.angleB = α ∧ t.angleC = 67.5

theorem triangle_angles :
  ∃ (t : Triangle), triangle_condition t :=
by
  -- The proof outline
  -- We need to construct a triangle with the given angle conditions
  -- angleA = 90°, angleB = 22.5°, angleC = 67.5°
  sorry

end triangle_angles_l186_186604


namespace function_property_l186_186690

noncomputable def f (x : ℝ) : ℝ := sorry
variable (a x1 x2 : ℝ)

-- Conditions
axiom f_defined_on_R : ∀ x : ℝ, f x ≠ 0
axiom f_increasing_on_left_of_a : ∀ x y : ℝ, x < y → y < a → f x < f y
axiom f_even_shifted_by_a : ∀ x : ℝ, f (x + a) = f (-(x + a))
axiom ordering : x1 < a ∧ a < x2
axiom distance_comp : |x1 - a| < |x2 - a|

-- Proof Goal
theorem function_property : f (2 * a - x1) > f (2 * a - x2) :=
by
  sorry

end function_property_l186_186690


namespace solve_system_eq_l186_186735

theorem solve_system_eq (x y z b : ℝ) :
  (3 * x * y * z - x^3 - y^3 - z^3 = b^3) ∧ 
  (x + y + z = 2 * b) ∧ 
  (x^2 + y^2 - z^2 = b^2) → 
  ( ∃ t : ℝ, (x = (1 + t) * b) ∧ (y = (1 - t) * b) ∧ (z = 0) ∧ t^2 = -1/2 ) :=
by
  -- proof will be filled in here
  sorry

end solve_system_eq_l186_186735


namespace jason_picked_pears_l186_186854

def jason_picked (total_picked keith_picked mike_picked jason_picked : ℕ) : Prop :=
  jason_picked + keith_picked + mike_picked = total_picked

theorem jason_picked_pears:
  jason_picked 105 47 12 46 :=
by 
  unfold jason_picked
  sorry

end jason_picked_pears_l186_186854


namespace negation_of_proposition_l186_186872

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x < 1) ↔ ∀ x : ℝ, x ≥ 1 :=
by sorry

end negation_of_proposition_l186_186872


namespace value_of_p_l186_186261

theorem value_of_p (x y p : ℝ) 
  (h1 : 3 * x - 2 * y = 4 - p) 
  (h2 : 4 * x - 3 * y = 2 + p) 
  (h3 : x > y) : 
  p < -1 := 
sorry

end value_of_p_l186_186261


namespace ratio_of_solving_linear_equations_to_algebra_problems_l186_186324

theorem ratio_of_solving_linear_equations_to_algebra_problems:
  let total_problems := 140
  let algebra_percentage := 0.40
  let solving_linear_equations := 28
  let total_algebra_problems := algebra_percentage * total_problems
  let ratio := solving_linear_equations / total_algebra_problems
  ratio = 1 / 2 := by
  sorry

end ratio_of_solving_linear_equations_to_algebra_problems_l186_186324


namespace min_value_x_squared_plus_6x_l186_186337

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, x^2 + 6 * x ≥ -9 := 
by
  sorry

end min_value_x_squared_plus_6x_l186_186337


namespace expression_evaluation_l186_186700

theorem expression_evaluation :
  100 + (120 / 15) + (18 * 20) - 250 - (360 / 12) = 188 := by
  sorry

end expression_evaluation_l186_186700


namespace next_ring_together_l186_186544

def nextRingTime (libraryInterval : ℕ) (fireStationInterval : ℕ) (hospitalInterval : ℕ) (start : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm libraryInterval fireStationInterval) hospitalInterval + start

theorem next_ring_together : nextRingTime 18 24 30 (8 * 60) = 14 * 60 :=
by
  sorry

end next_ring_together_l186_186544


namespace class_size_is_10_l186_186651

theorem class_size_is_10 
  (num_92 : ℕ) (num_80 : ℕ) (last_score : ℕ) (target_avg : ℕ) (total_score : ℕ) 
  (h_num_92 : num_92 = 5) (h_num_80 : num_80 = 4) (h_last_score : last_score = 70) 
  (h_target_avg : target_avg = 85) (h_total_score : total_score = 85 * (num_92 + num_80 + 1)) 
  : (num_92 * 92 + num_80 * 80 + last_score = total_score) → 
    (num_92 + num_80 + 1 = 10) :=
by {
  sorry
}

end class_size_is_10_l186_186651


namespace shaded_area_square_l186_186206

theorem shaded_area_square (s : ℝ) (r : ℝ) (A : ℝ) :
  s = 4 ∧ r = 2 * Real.sqrt 2 → A = s^2 - 4 * (π * r^2 / 2) → A = 8 - 2 * π :=
by
  intros h₁ h₂
  sorry

end shaded_area_square_l186_186206


namespace problem_l186_186977

variable (f : ℝ → ℝ)

-- Given condition
axiom h : ∀ x : ℝ, f (1 / x) = 1 / (x + 1)

-- Prove that f(2) = 2/3
theorem problem : f 2 = 2 / 3 :=
sorry

end problem_l186_186977


namespace find_general_term_arithmetic_sequence_l186_186276

-- Definitions needed
variable {a_n : ℕ → ℚ}
variable {S_n : ℕ → ℚ}

-- The main theorem to prove
theorem find_general_term_arithmetic_sequence 
  (h1 : a_n 4 - a_n 2 = 4)
  (h2 : S_n 3 = 9)
  (h3 : ∀ n : ℕ, S_n n = n / 2 * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) :
  (∀ n : ℕ, a_n n = 2 * n - 1) :=
by
  sorry

end find_general_term_arithmetic_sequence_l186_186276


namespace gwen_total_books_l186_186680

def mystery_shelves : Nat := 6
def mystery_books_per_shelf : Nat := 7

def picture_shelves : Nat := 4
def picture_books_per_shelf : Nat := 5

def biography_shelves : Nat := 3
def biography_books_per_shelf : Nat := 3

def scifi_shelves : Nat := 2
def scifi_books_per_shelf : Nat := 9

theorem gwen_total_books :
    (mystery_books_per_shelf * mystery_shelves) +
    (picture_books_per_shelf * picture_shelves) +
    (biography_books_per_shelf * biography_shelves) +
    (scifi_books_per_shelf * scifi_shelves) = 89 := 
by 
    sorry

end gwen_total_books_l186_186680


namespace range_of_a_l186_186496

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a (a : ℝ) : (-1 ≤ a ∧ a < 1/2) ↔
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) :=
by
  sorry

end range_of_a_l186_186496


namespace tangent_line_eq_area_independent_of_a_l186_186432

open Real

section TangentLineAndArea

def curve (x : ℝ) := x^2 - 1

def tangentCurvey (x : ℝ) := x^2

noncomputable def tangentLine (a : ℝ) (ha : a > 0) : (ℝ → ℝ) :=
  if a > 1 then λ x => (2*(a + 1)) * x - (a+1)^2
  else λ x => (2*(a - 1)) * x - (a-1)^2

theorem tangent_line_eq (a : ℝ) (ha : a > 0) :
  ∃ (line : ℝ → ℝ), (line = tangentLine a ha) :=
sorry

theorem area_independent_of_a (a : ℝ) (ha : a > 0) :
  (∫ x in (a - 1)..a, (tangentCurvey x - tangentLine a ha x)) +
  (∫ x in a..(a + 1), (tangentCurvey x - tangentLine a ha x)) = (2 / 3 : Real) :=
sorry

end TangentLineAndArea

end tangent_line_eq_area_independent_of_a_l186_186432


namespace solve_for_x_l186_186258

theorem solve_for_x (x : ℝ) (h : (1 / 5) + (5 / x) = (12 / x) + (1 / 12)) : x = 60 := by
  sorry

end solve_for_x_l186_186258


namespace system_eq_solution_l186_186054

theorem system_eq_solution (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 4 * x - 2 * y = c) 
  (h2 : 6 * y - 12 * x = d) :
  c / d = -1 / 3 := 
by 
  sorry

end system_eq_solution_l186_186054


namespace find_solutions_l186_186936

theorem find_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + 4^y = 5^z ↔ (x = 3 ∧ y = 2 ∧ z = 2) ∨ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 11 ∧ y = 1 ∧ z = 3) :=
by sorry

end find_solutions_l186_186936


namespace intersecting_sets_a_eq_1_l186_186650

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := { x | a * x^2 - 1 = 0 }
def N : Set ℝ := { -1/2, 1/2, 1 }

-- Define the intersection condition
def sets_intersect (M N : Set ℝ) : Prop :=
  ∃ x, x ∈ M ∧ x ∈ N

-- Statement of the problem
theorem intersecting_sets_a_eq_1 (a : ℝ) (h_intersect : sets_intersect (M a) N) : a = 1 :=
  sorry

end intersecting_sets_a_eq_1_l186_186650


namespace regions_of_diagonals_formula_l186_186367

def regions_of_diagonals (n : ℕ) : ℕ :=
  ((n - 1) * (n - 2) * (n * n - 3 * n + 12)) / 24

theorem regions_of_diagonals_formula (n : ℕ) (h : 3 ≤ n) :
  ∃ (fn : ℕ), fn = regions_of_diagonals n := by
  sorry

end regions_of_diagonals_formula_l186_186367


namespace output_correct_l186_186298

-- Define the initial values and assignments
def initial_a : ℕ := 1
def initial_b : ℕ := 2
def initial_c : ℕ := 3

-- Perform the assignments in sequence
def after_c_assignment : ℕ := initial_b
def after_b_assignment : ℕ := initial_a
def after_a_assignment : ℕ := after_c_assignment

-- Final values after all assignments
def final_a := after_a_assignment
def final_b := after_b_assignment
def final_c := after_c_assignment

-- Theorem statement
theorem output_correct :
  final_a = 2 ∧ final_b = 1 ∧ final_c = 2 :=
by {
  -- Proof is omitted
  sorry
}

end output_correct_l186_186298


namespace number_of_common_terms_between_arithmetic_sequences_l186_186468

-- Definitions for the sequences
def seq1 (n : Nat) := 2 + 3 * n
def seq2 (n : Nat) := 4 + 5 * n

theorem number_of_common_terms_between_arithmetic_sequences
  (A : Finset Nat := Finset.range 673)  -- There are 673 terms in seq1 from 2 to 2015
  (B : Finset Nat := Finset.range 403)  -- There are 403 terms in seq2 from 4 to 2014
  (common_terms : Finset Nat := (A.image seq1) ∩ (B.image seq2)) :
  common_terms.card = 134 := by
  sorry

end number_of_common_terms_between_arithmetic_sequences_l186_186468


namespace newspaper_price_l186_186255

-- Define the conditions as variables
variables 
  (P : ℝ)                    -- Price per edition for Wednesday, Thursday, and Friday
  (total_cost : ℝ := 28)     -- Total cost over 8 weeks
  (sunday_cost : ℝ := 2)     -- Cost of Sunday edition
  (weeks : ℕ := 8)           -- Number of weeks
  (wednesday_thursday_friday_editions : ℕ := 3 * weeks) -- Total number of editions for Wednesday, Thursday, and Friday over 8 weeks

-- Math proof problem statement
theorem newspaper_price : 
  (total_cost - weeks * sunday_cost) / wednesday_thursday_friday_editions = 0.5 :=
  sorry

end newspaper_price_l186_186255


namespace PJ_approx_10_81_l186_186320

noncomputable def PJ_length (P Q R J : Type) (PQ PR QR : ℝ) : ℝ :=
  if PQ = 30 ∧ PR = 29 ∧ QR = 27 then 10.81 else 0

theorem PJ_approx_10_81 (P Q R J : Type) (PQ PR QR : ℝ):
  PQ = 30 ∧ PR = 29 ∧ QR = 27 → PJ_length P Q R J PQ PR QR = 10.81 :=
by sorry

end PJ_approx_10_81_l186_186320


namespace arithmetic_sequence_sum_l186_186699

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    (∀ n, a (n + 1) = a n + d) →
    (a 1 + a 4 + a 7 = 45) →
    (a 2 + a_5 + a_8 = 39) →
    (a 3 + a_6 + a_9 = 33) :=
by 
  intros a d h_arith_seq h_cond1 h_cond2
  sorry

end arithmetic_sequence_sum_l186_186699


namespace sidney_cats_l186_186220

theorem sidney_cats (A : ℕ) :
  (4 * 7 * (3 / 4) + A * 7 = 42) →
  A = 3 :=
by
  intro h
  sorry

end sidney_cats_l186_186220


namespace age_ratio_l186_186615

variable (Cindy Jan Marcia Greg: ℕ)

theorem age_ratio 
  (h1 : Cindy = 5)
  (h2 : Jan = Cindy + 2)
  (h3: Greg = 16)
  (h4 : Greg = Marcia + 2)
  (h5 : ∃ k : ℕ, Marcia = k * Jan) 
  : Marcia / Jan = 2 := 
    sorry

end age_ratio_l186_186615


namespace equal_numbers_l186_186321

theorem equal_numbers {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + da) :
  a = b ∧ b = c ∧ c = d :=
by
  sorry

end equal_numbers_l186_186321


namespace triangle_area_ratio_l186_186019

theorem triangle_area_ratio (a n m : ℕ) (h1 : 0 < a) (h2 : 0 < n) (h3 : 0 < m) :
  let area_A := (a^2 : ℝ) / (4 * n^2)
  let area_B := (a^2 : ℝ) / (4 * m^2)
  (area_A / area_B) = (m^2 : ℝ) / (n^2 : ℝ) :=
by
  sorry

end triangle_area_ratio_l186_186019


namespace probability_of_at_least_two_same_rank_approx_l186_186306

noncomputable def probability_at_least_two_same_rank (cards_drawn : ℕ) (total_cards : ℕ) : ℝ :=
  let ranks := 13
  let different_ranks_comb := Nat.choose ranks cards_drawn
  let rank_suit_combinations := different_ranks_comb * (4 ^ cards_drawn)
  let total_combinations := Nat.choose total_cards cards_drawn
  let p_complement := rank_suit_combinations / total_combinations
  1 - p_complement

theorem probability_of_at_least_two_same_rank_approx (h : 5 ≤ 52) : 
  abs (probability_at_least_two_same_rank 5 52 - 0.49) < 0.01 := 
by
  sorry

end probability_of_at_least_two_same_rank_approx_l186_186306


namespace sum_coefficients_l186_186211

theorem sum_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℚ) :
  (1 - 2 * (1 : ℚ))^5 = a_0 + a_1 * (1 : ℚ) + a_2 * (1 : ℚ)^2 + a_3 * (1 : ℚ)^3 + a_4 * (1 : ℚ)^4 + a_5 * (1 : ℚ)^5 →
  (1 - 2 * (0 : ℚ))^5 = a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -2 :=
by
  sorry

end sum_coefficients_l186_186211


namespace intersection_complement_l186_186624

open Set

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | x^2 + 2 * x < 3}
noncomputable def B := {x : ℝ | x - 2 ≤ 0 ∧ x ≠ 0}

theorem intersection_complement :
  A ∩ -B = {x : ℝ | -3 < x ∧ x ≤ 0} :=
sorry

end intersection_complement_l186_186624


namespace area_of_given_rhombus_l186_186241

open Real

noncomputable def area_of_rhombus_with_side_and_angle (side : ℝ) (angle : ℝ) : ℝ :=
  let half_diag1 := side * cos (angle / 2)
  let half_diag2 := side * sin (angle / 2)
  let diag1 := 2 * half_diag1
  let diag2 := 2 * half_diag2
  (diag1 * diag2) / 2

theorem area_of_given_rhombus :
  area_of_rhombus_with_side_and_angle 25 40 = 201.02 :=
by
  sorry

end area_of_given_rhombus_l186_186241


namespace original_number_is_85_l186_186504

theorem original_number_is_85
  (x : ℤ) (h_sum : 10 ≤ x ∧ x < 100) 
  (h_condition1 : (x / 10) + (x % 10) = 13)
  (h_condition2 : 10 * (x % 10) + (x / 10) = x - 27) :
  x = 85 :=
by
  sorry

end original_number_is_85_l186_186504


namespace calc_two_pow_a_mul_two_pow_b_l186_186677

theorem calc_two_pow_a_mul_two_pow_b {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : (2^a)^b = 2^2) :
  2^a * 2^b = 8 :=
sorry

end calc_two_pow_a_mul_two_pow_b_l186_186677


namespace complete_square_identity_l186_186870

theorem complete_square_identity (x d e : ℤ) (h : x^2 - 10 * x + 15 = 0) :
  (x + d)^2 = e → d + e = 5 :=
by
  intros hde
  sorry

end complete_square_identity_l186_186870


namespace no_unfenced_area_l186_186520

noncomputable def area : ℝ := 5000
noncomputable def cost_per_foot : ℝ := 30
noncomputable def budget : ℝ := 120000

theorem no_unfenced_area (area : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  (budget / cost_per_foot) >= 4 * (Real.sqrt (area)) → 0 = 0 :=
by
  intro h
  sorry

end no_unfenced_area_l186_186520


namespace floor_area_cannot_exceed_10_square_meters_l186_186524

theorem floor_area_cannot_exceed_10_square_meters
  (a b : ℝ)
  (h : 3 > 0)
  (floor_lt_wall1 : a * b < 3 * a)
  (floor_lt_wall2 : a * b < 3 * b) :
  a * b ≤ 9 :=
by
  -- This is where the proof would go
  sorry

end floor_area_cannot_exceed_10_square_meters_l186_186524


namespace rectangle_perimeter_126_l186_186516

/-- Define the sides of the rectangle in terms of a common multiplier -/
def sides (x : ℝ) : ℝ × ℝ := (4 * x, 3 * x)

/-- Define the area of the rectangle given the common multiplier -/
def area (x : ℝ) : ℝ := (4 * x) * (3 * x)

example : ∃ (x : ℝ), area x = 972 :=
by
  sorry

/-- Calculate the perimeter of the rectangle given the common multiplier -/
def perimeter (x : ℝ) : ℝ := 2 * ((4 * x) + (3 * x))

/-- The final proof statement, stating that the perimeter of the rectangle is 126 meters,
    given the ratio of its sides and its area. -/
theorem rectangle_perimeter_126 (x : ℝ) (h: area x = 972) : perimeter x = 126 :=
by
  sorry

end rectangle_perimeter_126_l186_186516


namespace find_ck_l186_186472

def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
def geometric_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
def c_seq (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (n : ℕ) := a_seq n + b_seq n

theorem find_ck (d r k : ℕ) (a_seq := arithmetic_seq d) (b_seq := geometric_seq r) :
  c_seq a_seq b_seq (k - 1) = 200 →
  c_seq a_seq b_seq (k + 1) = 400 →
  c_seq a_seq b_seq k = 322 :=
by
  sorry

end find_ck_l186_186472


namespace find_larger_number_l186_186686

theorem find_larger_number 
  (x y : ℚ) 
  (h1 : 4 * y = 9 * x) 
  (h2 : y - x = 12) : 
  y = 108 / 5 := 
sorry

end find_larger_number_l186_186686


namespace speed_boat_in_still_water_l186_186454

-- Define the conditions
def speed_of_current := 20
def speed_upstream := 30

-- Define the effective speed given conditions
def effective_speed (speed_in_still_water : ℕ) := speed_in_still_water - speed_of_current

-- Theorem stating the problem
theorem speed_boat_in_still_water : 
  ∃ (speed_in_still_water : ℕ), effective_speed speed_in_still_water = speed_upstream ∧ speed_in_still_water = 50 := 
by 
  -- Proof to be filled in
  sorry

end speed_boat_in_still_water_l186_186454


namespace triplet_solution_l186_186239

theorem triplet_solution (x y z : ℝ) 
  (h1 : y = (x^3 + 12 * x) / (3 * x^2 + 4))
  (h2 : z = (y^3 + 12 * y) / (3 * y^2 + 4))
  (h3 : x = (z^3 + 12 * z) / (3 * z^2 + 4)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ 
  (x = -2 ∧ y = -2 ∧ z = -2) :=
sorry

end triplet_solution_l186_186239


namespace no_product_equal_remainder_l186_186655

theorem no_product_equal_remainder (n : ℤ) : 
  ¬ (n = (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 1) = n * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 2) = n * (n + 1) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 3) = n * (n + 1) * (n + 2) * (n + 4) * (n + 5) ∨
     (n + 4) = n * (n + 1) * (n + 2) * (n + 3) * (n + 5) ∨
     (n + 5) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end no_product_equal_remainder_l186_186655


namespace kyoko_bought_three_balls_l186_186582

theorem kyoko_bought_three_balls
  (cost_per_ball : ℝ)
  (total_paid : ℝ)
  (number_of_balls : ℝ)
  (h_cost_per_ball : cost_per_ball = 1.54)
  (h_total_paid : total_paid = 4.62)
  (h_number_of_balls : number_of_balls = total_paid / cost_per_ball) :
  number_of_balls = 3 := by
  sorry

end kyoko_bought_three_balls_l186_186582


namespace triangle_height_l186_186475

theorem triangle_height (area base : ℝ) (h_area : area = 9.31) (h_base : base = 4.9) : (2 * area) / base = 3.8 :=
by
  sorry

end triangle_height_l186_186475


namespace find_m_for_even_function_l186_186097

def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + (m + 2) * m * x + 2

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem find_m_for_even_function :
  ∃ m : ℝ, is_even_function (quadratic_function m) ∧ m = -2 :=
by
  sorry

end find_m_for_even_function_l186_186097


namespace solve_abs_eq_l186_186479

theorem solve_abs_eq (x : ℝ) : (|x + 4| = 3 - x) → (x = -1/2) := by
  intro h
  sorry

end solve_abs_eq_l186_186479


namespace animal_products_sampled_l186_186814

theorem animal_products_sampled
  (grains : ℕ)
  (oils : ℕ)
  (animal_products : ℕ)
  (fruits_vegetables : ℕ)
  (total_sample : ℕ)
  (total_food_types : grains + oils + animal_products + fruits_vegetables = 100)
  (sample_size : total_sample = 20)
  : (animal_products * total_sample / 100) = 6 := by
  sorry

end animal_products_sampled_l186_186814


namespace list_price_is_40_l186_186876

open Real

def list_price (x : ℝ) : Prop :=
  0.15 * (x - 15) = 0.25 * (x - 25)

theorem list_price_is_40 : list_price 40 :=
by
  unfold list_price
  sorry

end list_price_is_40_l186_186876


namespace distance_between_joe_and_gracie_l186_186249

open Complex

noncomputable def joe_point : ℂ := 2 + 3 * I
noncomputable def gracie_point : ℂ := -2 + 2 * I
noncomputable def distance := abs (joe_point - gracie_point)

theorem distance_between_joe_and_gracie :
  distance = Real.sqrt 17 := by
  sorry

end distance_between_joe_and_gracie_l186_186249


namespace circle_area_pi_l186_186541

def circle_eq := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1

theorem circle_area_pi (h : ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1) :
  ∃ S : ℝ, S = π :=
by {
  sorry
}

end circle_area_pi_l186_186541


namespace root_of_equation_l186_186251

theorem root_of_equation : 
  ∀ (f : ℝ → ℝ), (∀ x : ℝ, f x = (x - 1) / x) →
  f (4 * (1 / 2)) = (1 / 2) :=
by
  sorry

end root_of_equation_l186_186251


namespace smallest_integer_equal_costs_l186_186055

-- Definitions based directly on conditions
def decimal_cost (n : ℕ) : ℕ :=
  (n.digits 10).sum * 2

def binary_cost (n : ℕ) : ℕ :=
  (n.digits 2).sum

-- The main statement to prove
theorem smallest_integer_equal_costs : ∃ n : ℕ, n < 2000 ∧ decimal_cost n = binary_cost n ∧ n = 255 :=
by 
  sorry

end smallest_integer_equal_costs_l186_186055


namespace solve_equation_l186_186141

theorem solve_equation (x : ℝ) (h : x ≠ -1) :
  (x = -1 / 2 ∨ x = 2) ↔ (∃ x : ℝ, x ≠ -1 ∧ (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2) :=
sorry

end solve_equation_l186_186141


namespace unique_solution_pairs_count_l186_186550

theorem unique_solution_pairs_count :
  ∃! (p : ℝ × ℝ), (p.1 + 2 * p.2 = 2 ∧ (|abs p.1 - 2 * abs p.2| = 2) ∧
       ∃! q, (q = (2, 0) ∨ q = (0, 1)) ∧ p = q) := 
sorry

end unique_solution_pairs_count_l186_186550


namespace base7_subtraction_correct_l186_186061

-- Define a function converting base 7 number to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  d3 * 7^3 + d2 * 7^2 + d1 * 7^1 + d0 * 7^0

-- Define the numbers in base 7
def a : Nat := 2456
def b : Nat := 1234

-- Define the expected result in base 7
def result_base7 : Nat := 1222

-- State the theorem: The difference of a and b in base 7 should equal result_base7
theorem base7_subtraction_correct :
  let diff_base10 := (base7_to_base10 a) - (base7_to_base10 b)
  let result_base10 := base7_to_base10 result_base7
  diff_base10 = result_base10 :=
by
  sorry

end base7_subtraction_correct_l186_186061


namespace minimum_value_of_f_l186_186716

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f x ≤ f y) ∧ f x = -1 / Real.exp 1 :=
by
  -- Proof to be provided
  sorry

end minimum_value_of_f_l186_186716


namespace combined_points_kjm_l186_186819

theorem combined_points_kjm {P B K J M H C E: ℕ} 
  (total_points : P + B + K + J + M = 81)
  (paige_points : P = 21)
  (brian_points : B = 20)
  (karen_jennifer_michael_sum : K + J + M = 40)
  (karen_scores : ∀ p, K = 2 * p + 5 * (H - p))
  (jennifer_scores : ∀ p, J = 2 * p + 5 * (C - p))
  (michael_scores : ∀ p, M = 2 * p + 5 * (E - p)) :
  K + J + M = 40 :=
by sorry

end combined_points_kjm_l186_186819


namespace increased_consumption_5_percent_l186_186817

theorem increased_consumption_5_percent (T C : ℕ) (h1 : ¬ (T = 0)) (h2 : ¬ (C = 0)) :
  (0.80 * (1 + x/100) = 0.84) → (x = 5) :=
by
  sorry

end increased_consumption_5_percent_l186_186817


namespace area_of_inscribed_square_l186_186572

-- Define the right triangle with segments m and n on the hypotenuse
variables {m n : ℝ}

-- Noncomputable setting for non-constructive aspects
noncomputable def inscribed_square_area (m n : ℝ) : ℝ :=
  (m * n)

-- Theorem stating that the area of the inscribed square is m * n
theorem area_of_inscribed_square (m n : ℝ) : inscribed_square_area m n = m * n :=
by sorry

end area_of_inscribed_square_l186_186572


namespace hilt_books_transaction_difference_l186_186162

noncomputable def total_cost_paid (original_price : ℝ) (num_first_books : ℕ) (discount1 : ℝ) (num_second_books : ℕ) (discount2 : ℝ) : ℝ :=
  let cost_first_books := num_first_books * original_price * (1 - discount1)
  let cost_second_books := num_second_books * original_price * (1 - discount2)
  cost_first_books + cost_second_books

noncomputable def total_sale_amount (sale_price : ℝ) (interest_rate : ℝ) (num_books : ℕ) : ℝ :=
  let compounded_price := sale_price * (1 + interest_rate) ^ 1
  compounded_price * num_books

theorem hilt_books_transaction_difference : 
  let original_price := 11
  let num_first_books := 10
  let discount1 := 0.20
  let num_second_books := 5
  let discount2 := 0.25
  let sale_price := 25
  let interest_rate := 0.05
  let num_books := 15
  total_sale_amount sale_price interest_rate num_books - total_cost_paid original_price num_first_books discount1 num_second_books discount2 = 264.50 :=
by
  sorry

end hilt_books_transaction_difference_l186_186162


namespace smallest_k_for_sequence_l186_186692

theorem smallest_k_for_sequence (a : ℕ → ℕ) (k : ℕ) (h₁ : a 1 = 1) (h₂ : a 2018 = 2020)
  (h₃ : ∀ n, n ≥ 2 → a (n+1) = k * (a n) / (a (n-1))) : k = 2020 :=
sorry

end smallest_k_for_sequence_l186_186692


namespace even_composite_fraction_l186_186970

theorem even_composite_fraction : 
  ((4 * 6 * 8 * 10 * 12) : ℚ) / (14 * 16 * 18 * 20 * 22) = 1 / 42 :=
by 
  sorry

end even_composite_fraction_l186_186970


namespace parallelogram_angles_l186_186218

theorem parallelogram_angles (x y : ℝ) (h_sub : y = x + 50) (h_sum : x + y = 180) : x = 65 :=
by
  sorry

end parallelogram_angles_l186_186218


namespace tangent_product_le_one_third_l186_186599

theorem tangent_product_le_one_third (α β : ℝ) (h : α + β = π / 3) (hα : 0 < α) (hβ : 0 < β) : 
  Real.tan α * Real.tan β ≤ 1 / 3 :=
sorry

end tangent_product_le_one_third_l186_186599


namespace sum_largest_smallest_5_6_7_l186_186600

/--
Given the digits 5, 6, and 7, if we form all possible three-digit numbers using each digit exactly once, 
then the sum of the largest and smallest of these numbers is 1332.
-/
theorem sum_largest_smallest_5_6_7 : 
  let d1 := 5
  let d2 := 6
  let d3 := 7
  let smallest := 100 * d1 + 10 * d2 + d3
  let largest := 100 * d3 + 10 * d2 + d1
  smallest + largest = 1332 := 
by
  sorry

end sum_largest_smallest_5_6_7_l186_186600


namespace problem_1_problem_2_l186_186587

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end problem_1_problem_2_l186_186587


namespace circle_equation_tangent_to_line_l186_186766

theorem circle_equation_tangent_to_line
  (h k : ℝ) (A B C : ℝ)
  (hxk : h = 2) (hyk : k = -1) 
  (hA : A = 3) (hB : B = -4) (hC : C = 5)
  (r_squared : ℝ := (|A * h + B * k + C| / Real.sqrt (A^2 + B^2))^2)
  (h_radius : r_squared = 9) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r_squared := 
by
  sorry

end circle_equation_tangent_to_line_l186_186766


namespace ratio_of_numbers_l186_186474

theorem ratio_of_numbers (a b : ℕ) (hHCF : Nat.gcd a b = 4) (hLCM : Nat.lcm a b = 48) : a / b = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l186_186474


namespace circle_radius_proof_l186_186170

def circle_radius : Prop :=
  let D := -2
  let E := 3
  let F := -3 / 4
  let r := 1 / 2 * Real.sqrt (D^2 + E^2 - 4 * F)
  r = 2

theorem circle_radius_proof : circle_radius :=
  sorry

end circle_radius_proof_l186_186170


namespace find_digits_l186_186571

theorem find_digits (x y z : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (10 * x + 5) * (3 * 100 + y * 10 + z) = 7850 ↔ (x = 2 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end find_digits_l186_186571


namespace probability_at_5_5_equals_1_over_243_l186_186697

-- Define the base probability function P
def P : ℕ → ℕ → ℚ
| 0, 0       => 1
| x+1, 0     => 0
| 0, y+1     => 0
| x+1, y+1   => (1/3 : ℚ) * P x (y+1) + (1/3 : ℚ) * P (x+1) y + (1/3 : ℚ) * P x y

-- Theorem statement that needs to be proved
theorem probability_at_5_5_equals_1_over_243 : P 5 5 = 1 / 243 :=
sorry

end probability_at_5_5_equals_1_over_243_l186_186697


namespace train_time_to_pass_platform_l186_186801

-- Definitions as per the conditions
def length_of_train : ℕ := 720 -- Length of train in meters
def speed_of_train_kmh : ℕ := 72 -- Speed of train in km/hr
def length_of_platform : ℕ := 280 -- Length of platform in meters

-- Conversion factor and utility functions
def kmh_to_ms (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

def time_to_pass (distance speed_ms : ℕ) : ℕ :=
  distance / speed_ms

-- Main statement to be proven
theorem train_time_to_pass_platform :
  time_to_pass (total_distance length_of_train length_of_platform) (kmh_to_ms speed_of_train_kmh) = 50 :=
by
  sorry

end train_time_to_pass_platform_l186_186801


namespace number_value_l186_186279

theorem number_value (x : ℝ) (h : x = 3 * (1/x * -x) + 5) : x = 2 :=
by
  sorry

end number_value_l186_186279


namespace sum_of_digits_l186_186378

theorem sum_of_digits (N : ℕ) (h : N * (N + 1) / 2 = 3003) : (7 + 7) = 14 := by
  sorry

end sum_of_digits_l186_186378


namespace connie_total_markers_l186_186848

theorem connie_total_markers :
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  red_markers + blue_markers + green_markers + purple_markers = 15225 :=
by
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  -- Proof would go here, but we use sorry to skip it for now
  sorry

end connie_total_markers_l186_186848


namespace exists_positive_integer_m_l186_186760

noncomputable def d (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r - 1)
noncomputable def a_n (n : ℕ) (a1 d : ℝ) : ℝ := a1 + (n - 1) * d
noncomputable def g_n (n : ℕ) (g1 : ℝ) (r : ℝ) : ℝ := g1 * (r ^ (n - 1))

theorem exists_positive_integer_m (a1 g1 : ℝ) (r : ℝ) (h0 : g1 ≠ 0) (h1 : a1 = g1) (h2 : a2 = g2)
(h3 : a_n 10 a1 (d g1 r) = g_n 3 g1 r) :
  ∀ (p : ℕ), ∃ (m : ℕ), g_n p g1 r = a_n m a1 (d g1 r) := by
  sorry

end exists_positive_integer_m_l186_186760


namespace cos_double_angle_l186_186089

theorem cos_double_angle (α : ℝ) (h : Real.cos (π - α) = -3/5) : Real.cos (2 * α) = -7/25 :=
  sorry

end cos_double_angle_l186_186089


namespace smallest_angle_between_lines_l186_186707

theorem smallest_angle_between_lines (r1 r2 r3 : ℝ) (S U : ℝ) (h1 : r1 = 4) (h2 : r2 = 3) 
  (h3 : r3 = 2) (total_area : ℝ := π * (r1^2 + r2^2 + r3^2)) 
  (h4 : S = (5 / 8) * U) (h5 : S + U = total_area) : 
  ∃ θ : ℝ, θ = (5 * π) / 13 :=
by
  sorry

end smallest_angle_between_lines_l186_186707


namespace brandy_used_0_17_pounds_of_chocolate_chips_l186_186290

def weight_of_peanuts : ℝ := 0.17
def weight_of_raisins : ℝ := 0.08
def total_weight_of_trail_mix : ℝ := 0.42

theorem brandy_used_0_17_pounds_of_chocolate_chips :
  total_weight_of_trail_mix - (weight_of_peanuts + weight_of_raisins) = 0.17 :=
by
  sorry

end brandy_used_0_17_pounds_of_chocolate_chips_l186_186290


namespace expand_product_l186_186199

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x ^ 2 + 52 * x + 84 := by
  sorry

end expand_product_l186_186199


namespace chess_tournament_l186_186607

theorem chess_tournament :
  ∀ (n : ℕ), (∃ (players : ℕ) (total_games : ℕ),
  players = 8 ∧ total_games = 56 ∧ total_games = (players * (players - 1) * n) / 2) →
  n = 2 :=
by
  intros n h
  rcases h with ⟨players, total_games, h_players, h_total_games, h_eq⟩
  have := h_eq
  sorry

end chess_tournament_l186_186607


namespace find_incorrect_observation_l186_186063

theorem find_incorrect_observation (n : ℕ) (initial_mean new_mean : ℝ) (correct_value incorrect_value : ℝ) (observations_count : ℕ)
  (h1 : observations_count = 50)
  (h2 : initial_mean = 36)
  (h3 : new_mean = 36.5)
  (h4 : correct_value = 44) :
  incorrect_value = 19 :=
by
  sorry

end find_incorrect_observation_l186_186063


namespace determine_b_l186_186333

theorem determine_b (N a b c : ℤ) (h1 : a > 1 ∧ b > 1 ∧ c > 1) (h2 : N ≠ 1)
  (h3 : (N : ℝ) ^ (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c ^ 2)) = N ^ (49 / 60)) :
  b = 4 :=
sorry

end determine_b_l186_186333


namespace least_value_of_x_l186_186805

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) (h3 : x = 11 * p * 2) : x = 44 := 
by
  sorry

end least_value_of_x_l186_186805


namespace pens_given_away_l186_186198

theorem pens_given_away (initial_pens : ℕ) (pens_left : ℕ) (n : ℕ) (h1 : initial_pens = 56) (h2 : pens_left = 34) (h3 : n = initial_pens - pens_left) : n = 22 := by
  -- The proof is omitted
  sorry

end pens_given_away_l186_186198


namespace gray_region_area_l186_186884

-- Definitions based on given conditions
def radius_inner (r : ℝ) := r
def radius_outer (r : ℝ) := r + 3

-- Statement to prove: the area of the gray region
theorem gray_region_area (r : ℝ) : 
  (π * (radius_outer r)^2 - π * (radius_inner r)^2) = 6 * π * r + 9 * π := by
  sorry

end gray_region_area_l186_186884


namespace find_a_l186_186835

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

end find_a_l186_186835


namespace solution_to_absolute_value_equation_l186_186669

theorem solution_to_absolute_value_equation (x : ℝ) : 
    abs x - 2 - abs (-1) = 2 ↔ x = 5 ∨ x = -5 :=
by
  sorry

end solution_to_absolute_value_equation_l186_186669


namespace tangent_line_properties_l186_186949

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

end tangent_line_properties_l186_186949


namespace problem1_problem2_problem3_problem4_l186_186135

theorem problem1 (x : ℝ) : x^2 - 2 * x + 1 = 0 ↔ x = 1 := 
by sorry

theorem problem2 (x : ℝ) : x^2 + 2 * x - 3 = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem problem3 (x : ℝ) : 2 * x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 33) / 4 ∨ x = (-5 - Real.sqrt 33) / 4 :=
by sorry

theorem problem4 (x : ℝ) : 2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 :=
by sorry

end problem1_problem2_problem3_problem4_l186_186135


namespace correct_choice_D_l186_186724

theorem correct_choice_D (a : ℝ) :
  (2 * a ^ 2) ^ 3 = 8 * a ^ 6 ∧ 
  (a ^ 10 * a ^ 2 ≠ a ^ 20) ∧ 
  (a ^ 10 / a ^ 2 ≠ a ^ 5) ∧ 
  ((Real.pi - 3) ^ 0 ≠ 0) :=
by {
  sorry
}

end correct_choice_D_l186_186724


namespace bob_questions_three_hours_l186_186682

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end bob_questions_three_hours_l186_186682


namespace find_other_number_l186_186221

theorem find_other_number (a b : ℤ) (h1 : 2 * a + 3 * b = 100) (h2 : a = 28 ∨ b = 28) : a = 8 ∨ b = 8 :=
sorry

end find_other_number_l186_186221


namespace math_problem_l186_186445

theorem math_problem :
  (50 - (4050 - 450)) * (4050 - (450 - 50)) = -12957500 := 
by
  sorry

end math_problem_l186_186445


namespace remainder_97_pow_103_mul_7_mod_17_l186_186523

theorem remainder_97_pow_103_mul_7_mod_17 :
  (97 ^ 103 * 7) % 17 = 13 := by
  have h1 : 97 % 17 = -3 % 17 := by sorry
  have h2 : 9 % 17 = -8 % 17 := by sorry
  have h3 : 64 % 17 = 13 % 17 := by sorry
  have h4 : -21 % 17 = 13 % 17 := by sorry
  sorry

end remainder_97_pow_103_mul_7_mod_17_l186_186523


namespace arithmetic_geometric_sequence_l186_186091

/-- Given:
  * 1, a₁, a₂, 4 form an arithmetic sequence
  * 1, b₁, b₂, b₃, 4 form a geometric sequence
Prove that:
  (a₁ + a₂) / b₂ = 5 / 2
-/
theorem arithmetic_geometric_sequence (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (h_arith : 2 * a₁ = 1 + a₂ ∧ 2 * a₂ = a₁ + 4)
  (h_geom : b₁ * b₁ = b₂ ∧ b₁ * b₂ = b₃ ∧ b₂ * b₂ = b₃ * 4) :
  (a₁ + a₂) / b₂ = 5 / 2 :=
sorry

end arithmetic_geometric_sequence_l186_186091


namespace abc_equal_l186_186245

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l186_186245


namespace student_council_profit_l186_186344

def boxes : ℕ := 48
def erasers_per_box : ℕ := 24
def price_per_eraser : ℝ := 0.75

theorem student_council_profit :
  boxes * erasers_per_box * price_per_eraser = 864 := 
by
  sorry

end student_council_profit_l186_186344


namespace min_ticket_gates_l186_186117

theorem min_ticket_gates (a x y : ℕ) (h_pos: a > 0) :
  (a = 30 * x) ∧ (y = 2 * x) → ∃ n : ℕ, (n ≥ 4) ∧ (a + 5 * x ≤ 5 * n * y) :=
by
  sorry

end min_ticket_gates_l186_186117


namespace lana_total_spending_l186_186174

noncomputable def general_admission_cost : ℝ := 6
noncomputable def vip_cost : ℝ := 10
noncomputable def premium_cost : ℝ := 15

noncomputable def num_general_admission_tickets : ℕ := 6
noncomputable def num_vip_tickets : ℕ := 2
noncomputable def num_premium_tickets : ℕ := 1

noncomputable def discount_general_admission : ℝ := 0.10
noncomputable def discount_vip : ℝ := 0.15

noncomputable def total_spending (gen_cost : ℝ) (vip_cost : ℝ) (prem_cost : ℝ) (gen_num : ℕ) (vip_num : ℕ) (prem_num : ℕ) (gen_disc : ℝ) (vip_disc : ℝ) : ℝ :=
  let general_cost := gen_cost * gen_num
  let general_discount := general_cost * gen_disc
  let discounted_general_cost := general_cost - general_discount
  let vip_cost_total := vip_cost * vip_num
  let vip_discount := vip_cost_total * vip_disc
  let discounted_vip_cost := vip_cost_total - vip_discount
  let premium_cost_total := prem_cost * prem_num
  discounted_general_cost + discounted_vip_cost + premium_cost_total

theorem lana_total_spending : total_spending general_admission_cost vip_cost premium_cost num_general_admission_tickets num_vip_tickets num_premium_tickets discount_general_admission discount_vip = 64.40 := 
sorry

end lana_total_spending_l186_186174


namespace complement_union_l186_186485

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union : (U \ (A ∪ B)) = {1, 2, 6} :=
by simp only [U, A, B, Set.mem_union, Set.mem_compl, Set.mem_diff];
   sorry

end complement_union_l186_186485


namespace gcd_sequence_l186_186197

theorem gcd_sequence (n : ℕ) : gcd ((7^n - 1)/6) ((7^(n+1) - 1)/6) = 1 := by
  sorry

end gcd_sequence_l186_186197


namespace hyperbola_equation_l186_186845

theorem hyperbola_equation (c a b : ℝ) (ecc : ℝ) (h_c : c = 3) (h_ecc : ecc = 3 / 2) (h_a : a = 2) (h_b : b^2 = c^2 - a^2) :
    (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 4 - y^2 / 5 = 1)) :=
by
  sorry

end hyperbola_equation_l186_186845


namespace part_a_l186_186610

theorem part_a (n : ℤ) (m : ℤ) (h : m = n + 2) : 
  n * m + 1 = (n + 1) ^ 2 := by
  sorry

end part_a_l186_186610


namespace find_k_l186_186108

theorem find_k (k : ℝ) : (∀ x y : ℝ, (x + k * y - 2 * k = 0) → (k * x - (k - 2) * y + 1 = 0) → x * k + y * (-1 / k) + y * 2 = 0) →
  (k = 0 ∨ k = 3) :=
by
  sorry

end find_k_l186_186108


namespace community_service_arrangements_l186_186709

noncomputable def total_arrangements : ℕ :=
  let case1 := Nat.choose 6 3
  let case2 := 2 * Nat.choose 6 2
  let case3 := case2
  case1 + case2 + case3

theorem community_service_arrangements :
  total_arrangements = 80 :=
by
  sorry

end community_service_arrangements_l186_186709


namespace similar_triangles_height_l186_186985

theorem similar_triangles_height (h₁ h₂ : ℝ) (a₁ a₂ : ℝ) 
  (ratio_area : a₁ / a₂ = 1 / 9) (height_small : h₁ = 4) :
  h₂ = 12 :=
sorry

end similar_triangles_height_l186_186985


namespace no_non_trivial_power_ending_222_l186_186698

theorem no_non_trivial_power_ending_222 (x y : ℕ) (hx : x > 1) (hy : y > 1) : ¬ (∃ n : ℕ, n % 1000 = 222 ∧ n = x^y) :=
by
  sorry

end no_non_trivial_power_ending_222_l186_186698


namespace incorrect_statement_A_l186_186419

def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def has_real_roots (a b c : ℝ) : Prop :=
  let delta := b^2 - 4 * a * c
  delta ≥ 0

theorem incorrect_statement_A (a b c : ℝ) (h₀ : a ≠ 0) :
  (∃ x : ℝ, parabola a b c x = 0) ∧ (parabola a b c (-b/(2*a)) < 0) → ¬ has_real_roots a b c := 
by
  sorry -- proof required here if necessary

end incorrect_statement_A_l186_186419


namespace area_outside_small_squares_l186_186803

theorem area_outside_small_squares (a b : ℕ) (ha : a = 10) (hb : b = 4) (n : ℕ) (hn: n = 2) :
  a^2 - n * b^2 = 68 :=
by
  rw [ha, hb, hn]
  sorry

end area_outside_small_squares_l186_186803


namespace complement_intersection_l186_186225

open Set

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_intersection :
  compl A ∩ B = {-2, -1} :=
by
  sorry

end complement_intersection_l186_186225


namespace percentage_decrease_of_b_l186_186473

variables (a b x m : ℝ) (p : ℝ)

-- Given conditions
def ratio_ab : Prop := a / b = 4 / 5
def expression_x : Prop := x = 1.25 * a
def expression_m : Prop := m = b * (1 - p / 100)
def ratio_mx : Prop := m / x = 0.6

-- The theorem to be proved
theorem percentage_decrease_of_b 
  (h1 : ratio_ab a b)
  (h2 : expression_x a x)
  (h3 : expression_m b m p)
  (h4 : ratio_mx m x) 
  : p = 40 :=
sorry

end percentage_decrease_of_b_l186_186473


namespace total_students_surveyed_l186_186783

variable (F E S FE FS ES FES N T : ℕ)

def only_one_language := 230
def exactly_two_languages := 190
def all_three_languages := 40
def no_language := 60

-- Summing up all categories
def total_students := only_one_language + exactly_two_languages + all_three_languages + no_language

theorem total_students_surveyed (h1 : F + E + S = only_one_language) 
    (h2 : FE + FS + ES = exactly_two_languages) 
    (h3 : FES = all_three_languages) 
    (h4 : N = no_language) 
    (h5 : T = F + E + S + FE + FS + ES + FES + N) : 
    T = total_students :=
by
  rw [total_students, only_one_language, exactly_two_languages, all_three_languages, no_language]
  sorry

end total_students_surveyed_l186_186783


namespace min_odd_integers_l186_186072

theorem min_odd_integers :
  ∀ (a b c d e f g h : ℤ),
  a + b + c = 30 →
  a + b + c + d + e + f = 58 →
  a + b + c + d + e + f + g + h = 73 →
  ∃ (odd_count : ℕ), odd_count = 1 :=
by
  sorry

end min_odd_integers_l186_186072


namespace greater_number_l186_186389

theorem greater_number (x: ℕ) (h1 : 3 * x + 4 * x = 21) : 4 * x = 12 := by
  sorry

end greater_number_l186_186389


namespace find_line_eq_l186_186041

noncomputable def line_perpendicular (p : ℝ × ℝ) (a b c: ℝ) : Prop :=
  ∃ (m: ℝ) (k: ℝ), k ≠ 0 ∧ (b * m = -a) ∧ p = (m, (c - a * m) / b) ∧
  (∀ x y : ℝ, y = m * x + ((c - a * m) / b) ↔ b * y = -a * x - c)

theorem find_line_eq (p : ℝ × ℝ) (a b c : ℝ) (p_eq : p = (-3, 0)) (perpendicular_eq : a = 2 ∧ b = -1 ∧ c = 3) :
  ∃ (m k : ℝ), (k ≠ 0 ∧ (-1 * (b / a)) = m ∧ line_perpendicular p a b c) ∧ (b * m = -a) ∧ ((k = (-a * m) / b) ∧ (b * k * 0 - (-a * 3)) = c) := sorry

end find_line_eq_l186_186041


namespace sum_of_reciprocals_eq_six_l186_186087

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  (1 / x + 1 / y) = 6 :=
by
  sorry

end sum_of_reciprocals_eq_six_l186_186087


namespace speed_of_A_l186_186477

theorem speed_of_A (V_B : ℝ) (h_VB : V_B = 4.555555555555555)
  (h_B_overtakes: ∀ (t_A t_B : ℝ), t_A = t_B + 0.5 → t_B = 1.8) 
  : ∃ V_A : ℝ, V_A = 3.57 :=
by
  sorry

end speed_of_A_l186_186477


namespace age_difference_is_24_l186_186343

theorem age_difference_is_24 (d f : ℕ) (h1 : d = f / 9) (h2 : f + 1 = 7 * (d + 1)) : f - d = 24 := sorry

end age_difference_is_24_l186_186343


namespace lcm_gcd_pairs_l186_186166

theorem lcm_gcd_pairs (a b : ℕ) :
  (lcm a b + gcd a b = (a * b) / 5) ↔
  (a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6) :=
sorry

end lcm_gcd_pairs_l186_186166


namespace cost_of_art_book_l186_186718

theorem cost_of_art_book
  (total_cost m_c s_c : ℕ)
  (m_b s_b a_b : ℕ)
  (hm : m_c = 3)
  (hs : s_c = 3)
  (ht : total_cost = 30)
  (hm_books : m_b = 2)
  (hs_books : s_b = 6)
  (ha_books : a_b = 3)
  : ∃ (a_c : ℕ), a_c = 2 := 
by
  sorry

end cost_of_art_book_l186_186718


namespace range_of_a_l186_186877

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

def sibling_point_pair (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = f a A.1 ∧ B.2 = f a B.1 ∧ A.1 = -B.1 ∧ A.2 = -B.2

theorem range_of_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, sibling_point_pair a A B) ↔ a > 1 :=
sorry

end range_of_a_l186_186877


namespace solve_for_x_l186_186248

theorem solve_for_x (x : ℝ) (h : 9 / (5 + x / 0.75) = 1) : x = 3 :=
by {
  sorry
}

end solve_for_x_l186_186248


namespace problem_area_of_circle_l186_186195

noncomputable def circleAreaPortion : ℝ :=
  let r := Real.sqrt 59
  let theta := 135 * Real.pi / 180
  (theta / (2 * Real.pi)) * (Real.pi * r^2)

theorem problem_area_of_circle :
  circleAreaPortion = (177 / 8) * Real.pi := by
  sorry

end problem_area_of_circle_l186_186195
