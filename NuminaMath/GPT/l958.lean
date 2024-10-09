import Mathlib

namespace f_f_is_even_l958_95809

-- Let f be a function from reals to reals
variables {f : ℝ → ℝ}

-- Given that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem to prove
theorem f_f_is_even (h : is_even f) : is_even (fun x => f (f x)) :=
by
  intros
  unfold is_even at *
  -- at this point, we assume the function f is even,
  -- follow from the assumption, we can prove the result
  sorry

end f_f_is_even_l958_95809


namespace desired_antifreeze_pct_in_colder_climates_l958_95891

-- Definitions for initial conditions
def initial_antifreeze_pct : ℝ := 0.10
def radiator_volume : ℝ := 4
def drained_volume : ℝ := 2.2857
def replacement_antifreeze_pct : ℝ := 0.80

-- Proof goal: Desired percentage of antifreeze in the mixture is 50%
theorem desired_antifreeze_pct_in_colder_climates :
  (drained_volume * replacement_antifreeze_pct + (radiator_volume - drained_volume) * initial_antifreeze_pct) / radiator_volume = 0.50 :=
by
  sorry

end desired_antifreeze_pct_in_colder_climates_l958_95891


namespace intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l958_95850

-- Define the solution sets A and B given conditions
def solution_set_A (a : ℝ) : Set ℝ :=
  { x | |x - 1| ≤ a }

def solution_set_B : Set ℝ :=
  { x | (x - 2) * (x + 2) > 0 }

theorem intersection_A_B_when_a_eq_2 :
  solution_set_A 2 ∩ solution_set_B = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ (a : ℝ), solution_set_A a ∩ solution_set_B = ∅ → 0 < a ∧ a ≤ 1 :=
by
  sorry

end intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l958_95850


namespace frustum_lateral_surface_area_l958_95824

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) (hh : h = 5) :
  let d := r1 - r2
  let s := Real.sqrt (h^2 + d^2)
  let A := Real.pi * s * (r1 + r2)
  A = 12 * Real.pi * Real.sqrt 41 :=
by
  -- hr1 and hr2 imply that r1 and r2 are constants, therefore d = 8 - 4 = 4
  -- h = 5 and d = 4 imply s = sqrt (5^2 + 4^2) = sqrt 41
  -- The area A is then pi * sqrt 41 * (8 + 4) = 12 * pi * sqrt 41
  sorry

end frustum_lateral_surface_area_l958_95824


namespace martha_correct_guess_probability_l958_95804

namespace MarthaGuess

-- Definitions for the conditions
def height_guess_child_accurate : ℚ := 4 / 5
def height_guess_adult_accurate : ℚ := 5 / 6
def weight_guess_tight_clothing_accurate : ℚ := 3 / 4
def weight_guess_loose_clothing_accurate : ℚ := 7 / 10

-- Probabilities of incorrect guesses
def height_guess_child_inaccurate : ℚ := 1 - height_guess_child_accurate
def height_guess_adult_inaccurate : ℚ := 1 - height_guess_adult_accurate
def weight_guess_tight_clothing_inaccurate : ℚ := 1 - weight_guess_tight_clothing_accurate
def weight_guess_loose_clothing_inaccurate : ℚ := 1 - weight_guess_loose_clothing_accurate

-- Combined probability of guessing incorrectly for each case
def incorrect_prob_child_loose : ℚ := height_guess_child_inaccurate * weight_guess_loose_clothing_inaccurate
def incorrect_prob_adult_tight : ℚ := height_guess_adult_inaccurate * weight_guess_tight_clothing_inaccurate
def incorrect_prob_adult_loose : ℚ := height_guess_adult_inaccurate * weight_guess_loose_clothing_inaccurate

-- Total probability of incorrect guesses for all three cases
def total_incorrect_prob : ℚ := incorrect_prob_child_loose * incorrect_prob_adult_tight * incorrect_prob_adult_loose

-- Probability of at least one correct guess
def correct_prob_at_least_once : ℚ := 1 - total_incorrect_prob

-- Main theorem stating the final result
theorem martha_correct_guess_probability : correct_prob_at_least_once = 7999 / 8000 := by
  sorry

end MarthaGuess

end martha_correct_guess_probability_l958_95804


namespace cubic_poly_sum_l958_95890

noncomputable def q (x : ℕ) : ℤ := sorry

axiom h0 : q 1 = 5
axiom h1 : q 6 = 24
axiom h2 : q 10 = 16
axiom h3 : q 15 = 34

theorem cubic_poly_sum :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) +
  (q 7) + (q 8) + (q 9) + (q 10) + (q 11) + (q 12) + (q 13) +
  (q 14) + (q 15) + (q 16) = 340 :=
by
  sorry

end cubic_poly_sum_l958_95890


namespace pyramid_volume_l958_95806

noncomputable def volume_pyramid (a b : ℝ) : ℝ :=
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2))

theorem pyramid_volume (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 < 4 * b^2) :
  volume_pyramid a b =
  18 * a^3 * b^3 / ((a^2 - b^2) * Real.sqrt (4 * b^2 - a^2)) :=
sorry

end pyramid_volume_l958_95806


namespace length_of_segment_AB_l958_95854

theorem length_of_segment_AB :
  ∀ A B : ℝ × ℝ,
  (∃ x y : ℝ, y^2 = 8 * x ∧ y = (y - 0) / (4 - 2) * (x - 2))
  ∧ (A.1 + B.1) / 2 = 4
  → dist A B = 12 := 
by
  sorry

end length_of_segment_AB_l958_95854


namespace clubs_equal_students_l958_95816

-- Define the concepts of Club and Student
variable (Club Student : Type)

-- Define the membership relations
variable (Members : Club → Finset Student)
variable (Clubs : Student → Finset Club)

-- Define the conditions
axiom club_membership (c : Club) : (Members c).card = 3
axiom student_club_membership (s : Student) : (Clubs s).card = 3

-- The goal is to prove that the number of clubs is equal to the number of students
theorem clubs_equal_students [Fintype Club] [Fintype Student] : Fintype.card Club = Fintype.card Student := by
  sorry

end clubs_equal_students_l958_95816


namespace max_rectangle_area_l958_95847

theorem max_rectangle_area (l w : ℕ) (h : 3 * l + 5 * w ≤ 50) : (l * w ≤ 35) :=
by sorry

end max_rectangle_area_l958_95847


namespace manager_salary_l958_95834

theorem manager_salary 
  (a : ℝ) (n : ℕ) (m_total : ℝ) (new_avg : ℝ) (m_avg_inc : ℝ)
  (h1 : n = 20) 
  (h2 : a = 1600) 
  (h3 : m_avg_inc = 100) 
  (h4 : new_avg = a + m_avg_inc)
  (h5 : m_total = n * a)
  (h6 : new_avg = (m_total + M) / (n + 1)) : 
  M = 3700 :=
by
  sorry

end manager_salary_l958_95834


namespace john_spends_on_memory_cards_l958_95892

theorem john_spends_on_memory_cards :
  (10 * (3 * 365)) / 50 * 60 = 13140 :=
by
  sorry

end john_spends_on_memory_cards_l958_95892


namespace regression_analysis_correct_statement_l958_95880

variables (x : Type) (y : Type)

def is_deterministic (v : Type) : Prop := sorry -- A placeholder definition
def is_random (v : Type) : Prop := sorry -- A placeholder definition

theorem regression_analysis_correct_statement :
  (is_deterministic x) → (is_random y) →
  ("The independent variable is a deterministic variable, and the dependent variable is a random variable" = "C") :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end regression_analysis_correct_statement_l958_95880


namespace speed_of_stream_l958_95814

theorem speed_of_stream (v : ℝ) : (13 + v) * 4 = 68 → v = 4 :=
by
  intro h
  sorry

end speed_of_stream_l958_95814


namespace remainder_div_l958_95883

theorem remainder_div (n : ℕ) : (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 + 
  90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 - 90^7 * Nat.choose 10 7 + 
  90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 + 90^10 * Nat.choose 10 10) % 88 = 1 := by
  sorry

end remainder_div_l958_95883


namespace find_k_l958_95853

def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

theorem find_k (k : ℝ) (h : f 5 - g 5 k = 24) : k = -16.36 := by
  sorry

end find_k_l958_95853


namespace simplify_expression_l958_95802

theorem simplify_expression (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2) ^ 2) + Real.sqrt ((a - 8) ^ 2) = 6 :=
by
  sorry

end simplify_expression_l958_95802


namespace solve_for_x_l958_95860

theorem solve_for_x (x : ℝ) (h : (x - 75) / 3 = (8 - 3 * x) / 4) : 
  x = 324 / 13 :=
sorry

end solve_for_x_l958_95860


namespace smallest_w_value_l958_95875

theorem smallest_w_value (x y z w : ℝ) 
    (hx : -2 ≤ x ∧ x ≤ 5) 
    (hy : -3 ≤ y ∧ y ≤ 7) 
    (hz : 4 ≤ z ∧ z ≤ 8) 
    (hw : w = x * y - z) : 
    w ≥ -23 :=
sorry

end smallest_w_value_l958_95875


namespace child_support_owed_l958_95881

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l958_95881


namespace g_triple_evaluation_l958_95896

def g (x : ℤ) : ℤ := 
if x < 8 then x ^ 2 - 6 
else x - 15

theorem g_triple_evaluation :
  g (g (g 20)) = 4 :=
by sorry

end g_triple_evaluation_l958_95896


namespace expected_rainfall_week_l958_95820

theorem expected_rainfall_week :
  let P_sun := 0.35
  let P_2 := 0.40
  let P_8 := 0.25
  let rainfall_2 := 2
  let rainfall_8 := 8
  let daily_expected := P_sun * 0 + P_2 * rainfall_2 + P_8 * rainfall_8
  let total_expected := 7 * daily_expected
  total_expected = 19.6 :=
by
  sorry

end expected_rainfall_week_l958_95820


namespace ratio_of_c_to_b_l958_95813

    theorem ratio_of_c_to_b (a b c : ℤ) (h0 : a = 0) (h1 : a < b) (h2 : b < c)
      (h3 : (a + b + c) / 3 = b / 2) : c / b = 1 / 2 :=
    by
      -- proof steps go here
      sorry
    
end ratio_of_c_to_b_l958_95813


namespace vegetable_difference_is_30_l958_95800

def initial_tomatoes : Int := 17
def initial_carrots : Int := 13
def initial_cucumbers : Int := 8
def initial_bell_peppers : Int := 15
def initial_radishes : Int := 0

def picked_tomatoes : Int := 5
def picked_carrots : Int := 6
def picked_cucumbers : Int := 3
def picked_bell_peppers : Int := 8

def given_neighbor1_tomatoes : Int := 3
def given_neighbor1_carrots : Int := 2

def exchanged_neighbor2_tomatoes : Int := 2
def exchanged_neighbor2_cucumbers : Int := 3
def exchanged_neighbor2_radishes : Int := 5

def given_neighbor3_bell_peppers : Int := 3

noncomputable def initial_total := 
  initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers + initial_radishes

noncomputable def remaining_after_picking :=
  (initial_tomatoes - picked_tomatoes) +
  (initial_carrots - picked_carrots) +
  (initial_cucumbers - picked_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers)

noncomputable def remaining_after_exchanges :=
  ((initial_tomatoes - picked_tomatoes - given_neighbor1_tomatoes - exchanged_neighbor2_tomatoes) +
  (initial_carrots - picked_carrots - given_neighbor1_carrots) +
  (initial_cucumbers - picked_cucumbers - exchanged_neighbor2_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers - given_neighbor3_bell_peppers) +
  exchanged_neighbor2_radishes)

noncomputable def remaining_total := remaining_after_exchanges

noncomputable def total_difference := initial_total - remaining_total

theorem vegetable_difference_is_30 : total_difference = 30 := by
  sorry

end vegetable_difference_is_30_l958_95800


namespace distance_between_poles_l958_95812

theorem distance_between_poles (length width : ℝ) (num_poles : ℕ) (h_length : length = 90)
  (h_width : width = 40) (h_num_poles : num_poles = 52) : 
  (2 * (length + width)) / (num_poles - 1) = 5.098 := 
by 
  -- Sorry to skip the proof
  sorry

end distance_between_poles_l958_95812


namespace find_m_l958_95811

theorem find_m (m : ℕ) (h1 : List ℕ := [27, 32, 39, m, 46, 47])
            (h2 : List ℕ := [30, 31, 34, 41, 42, 45])
            (h3 : (39 + m) / 2 = 42) :
            m = 45 :=
by {
  sorry
}

end find_m_l958_95811


namespace square_side_length_l958_95884

/-- Define OPEN as a square and T a point on side NO
    such that the areas of triangles TOP and TEN are 
    respectively 62 and 10. Prove that the side length 
    of the square is 12. -/
theorem square_side_length (s x y : ℝ) (T : x + y = s)
    (h1 : 0 < s) (h2 : 0 < x) (h3 : 0 < y)
    (a1 : 1 / 2 * x * s = 62)
    (a2 : 1 / 2 * y * s = 10) :
    s = 12 :=
by
    sorry

end square_side_length_l958_95884


namespace five_digit_sine_rule_count_l958_95846

theorem five_digit_sine_rule_count :
    ∃ (count : ℕ), 
        (∀ (a b c d e : ℕ), 
          (a <  b) ∧
          (b >  c) ∧
          (c >  d) ∧
          (d <  e) ∧
          (a >  d) ∧
          (b >  e) ∧
          (∃ (num : ℕ), num = 10000 * a + 1000 * b + 100 * c + 10 * d + e))
        →
        count = 2892 :=
sorry

end five_digit_sine_rule_count_l958_95846


namespace range_of_function_l958_95865

theorem range_of_function : 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 12) ∧ 
  (∃ x : ℝ, |x + 5| - |x - 3| + 4 = 18) ∧ 
  (∀ y : ℝ, (12 ≤ y ∧ y ≤ 18) → 
    ∃ x : ℝ, y = |x + 5| - |x - 3| + 4) :=
by
  sorry

end range_of_function_l958_95865


namespace primes_less_or_equal_F_l958_95886

-- Definition of F_n
def F (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- The main theorem statement
theorem primes_less_or_equal_F (n : ℕ) : ∃ S : Finset ℕ, S.card ≥ n + 1 ∧ ∀ p ∈ S, Nat.Prime p ∧ p ≤ F n := 
sorry

end primes_less_or_equal_F_l958_95886


namespace girls_in_class_l958_95832

theorem girls_in_class (k : ℕ) (n_girls n_boys total_students : ℕ)
  (h1 : n_girls = 3 * k) (h2 : n_boys = 4 * k) (h3 : total_students = 35) 
  (h4 : n_girls + n_boys = total_students) : 
  n_girls = 15 :=
by
  -- The proof would normally go here, but is omitted per instructions.
  sorry

end girls_in_class_l958_95832


namespace sequence_a10_l958_95898

theorem sequence_a10 : 
  (∃ (a : ℕ → ℤ), 
    a 1 = -1 ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n) - a (2*n - 1) = 2^(2*n-1)) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n + 1) - a (2*n) = 2^(2*n))) → 
  (∃ a : ℕ → ℤ, a 10 = 1021) :=
by
  intro h
  obtain ⟨a, h1, h2, h3⟩ := h
  sorry

end sequence_a10_l958_95898


namespace james_take_home_pay_l958_95821

theorem james_take_home_pay :
  let main_hourly_rate := 20
  let second_hourly_rate := main_hourly_rate - (main_hourly_rate * 0.20)
  let main_hours := 30
  let second_hours := main_hours / 2
  let side_gig_earnings := 100 * 2
  let overtime_hours := 5
  let overtime_rate := main_hourly_rate * 1.5
  let irs_tax_rate := 0.18
  let state_tax_rate := 0.05
  
  -- Main job earnings
  let main_regular_earnings := main_hours * main_hourly_rate
  let main_overtime_earnings := overtime_hours * overtime_rate
  let main_total_earnings := main_regular_earnings + main_overtime_earnings
  
  -- Second job earnings
  let second_total_earnings := second_hours * second_hourly_rate
  
  -- Total earnings before taxes
  let total_earnings := main_total_earnings + second_total_earnings + side_gig_earnings
  
  -- Tax calculations
  let federal_tax := total_earnings * irs_tax_rate
  let state_tax := total_earnings * state_tax_rate
  let total_taxes := federal_tax + state_tax

  -- Total take home pay after taxes
  let take_home_pay := total_earnings - total_taxes

  take_home_pay = 916.30 := 
sorry

end james_take_home_pay_l958_95821


namespace system_solution_l958_95867

theorem system_solution :
  ∃ x y : ℝ, (3 * x + y = 11 ∧ x - y = 1) ∧ (x = 3 ∧ y = 2) := 
by
  sorry

end system_solution_l958_95867


namespace gcf_450_144_l958_95861

theorem gcf_450_144 : Nat.gcd 450 144 = 18 := by
  sorry

end gcf_450_144_l958_95861


namespace redistribution_amount_l958_95856

theorem redistribution_amount
    (earnings : Fin 5 → ℕ)
    (h : earnings = ![18, 22, 30, 35, 45]) :
    (earnings 4 - ((earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5)) = 15 :=
by
  sorry

end redistribution_amount_l958_95856


namespace rectangle_perimeter_l958_95839

variable (L W : ℝ)

-- Conditions
def width := 70
def length := (7 / 5) * width

-- Perimeter calculation and proof goal
def perimeter (L W : ℝ) := 2 * (L + W)

theorem rectangle_perimeter : perimeter (length) (width) = 336 := by
  sorry

end rectangle_perimeter_l958_95839


namespace taylor_scores_l958_95866

/-
Conditions:
1. Taylor combines white and black scores in the ratio of 7:6.
2. She gets 78 yellow scores.

Question:
Prove that 2/3 of the difference between the number of black and white scores she used is 4.
-/

theorem taylor_scores (yellow_scores total_parts: ℕ) (ratio_white ratio_black: ℕ)
  (ratio_condition: ratio_white + ratio_black = total_parts)
  (yellow_scores_given: yellow_scores = 78)
  (ratio_white_given: ratio_white = 7)
  (ratio_black_given: ratio_black = 6)
   :
   (2 / 3) * (ratio_white * (yellow_scores / total_parts) - ratio_black * (yellow_scores / total_parts)) = 4 := 
by
  sorry

end taylor_scores_l958_95866


namespace kenya_peanuts_l958_95815

def jose_peanuts : ℕ := 85
def difference : ℕ := 48

theorem kenya_peanuts : jose_peanuts + difference = 133 := by
  sorry

end kenya_peanuts_l958_95815


namespace probability_same_suit_JQKA_l958_95862

theorem probability_same_suit_JQKA  : 
  let deck_size := 52 
  let prob_J := 4 / deck_size
  let prob_Q_given_J := 1 / (deck_size - 1) 
  let prob_K_given_JQ := 1 / (deck_size - 2)
  let prob_A_given_JQK := 1 / (deck_size - 3)
  prob_J * prob_Q_given_J * prob_K_given_JQ * prob_A_given_JQK = 1 / 1624350 :=
by
  sorry

end probability_same_suit_JQKA_l958_95862


namespace distribution_scheme_count_l958_95851

-- Definitions based on conditions
variable (village1 village2 village3 village4 : Type)
variables (quota1 quota2 quota3 quota4 : ℕ)

-- Conditions as given in the problem
def valid_distribution (v1 v2 v3 v4 : ℕ) : Prop :=
  v1 = 1 ∧ v2 = 2 ∧ v3 = 3 ∧ v4 = 4

-- The goal is to prove the number of permutations is equal to 24
theorem distribution_scheme_count :
  (∃ v1 v2 v3 v4 : ℕ, valid_distribution v1 v2 v3 v4) → 
  (4 * 3 * 2 * 1 = 24) :=
by 
  sorry

end distribution_scheme_count_l958_95851


namespace Tanya_accompanied_two_l958_95889

-- Define the number of songs sung by each girl
def Anya_songs : ℕ := 8
def Tanya_songs : ℕ := 6
def Olya_songs : ℕ := 3
def Katya_songs : ℕ := 7

-- Assume each song is sung by three girls
def total_songs : ℕ := (Anya_songs + Tanya_songs + Olya_songs + Katya_songs) / 3

-- Define the number of times Tanya accompanied
def Tanya_accompanied : ℕ := total_songs - Tanya_songs

-- Prove that Tanya accompanied 2 times
theorem Tanya_accompanied_two : Tanya_accompanied = 2 :=
by sorry

end Tanya_accompanied_two_l958_95889


namespace gcd_of_28430_and_39674_l958_95874

theorem gcd_of_28430_and_39674 : Nat.gcd 28430 39674 = 2 := 
by 
  sorry

end gcd_of_28430_and_39674_l958_95874


namespace find_triangle_angles_l958_95895

theorem find_triangle_angles (α β γ : ℝ)
  (h1 : (180 - α) / (180 - β) = 13 / 9)
  (h2 : β - α = 45)
  (h3 : α + β + γ = 180) :
  (α = 33.75) ∧ (β = 78.75) ∧ (γ = 67.5) :=
by
  sorry

end find_triangle_angles_l958_95895


namespace total_price_of_hats_l958_95873

-- Declare the conditions as Lean definitions
def total_hats : Nat := 85
def green_hats : Nat := 38
def blue_hat_cost : Nat := 6
def green_hat_cost : Nat := 7

-- The question becomes proving the total cost of the hats is $548
theorem total_price_of_hats :
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  total_blue_cost + total_green_cost = 548 := by
  let blue_hats := total_hats - green_hats
  let total_blue_cost := blue_hat_cost * blue_hats
  let total_green_cost := green_hat_cost * green_hats
  show total_blue_cost + total_green_cost = 548
  sorry

end total_price_of_hats_l958_95873


namespace find_perpendicular_slope_value_l958_95826

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value_l958_95826


namespace solve_for_s_l958_95844

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * 3^s) 
  (h2 : 45 = m * 9^s) : 
  s = 2 :=
sorry

end solve_for_s_l958_95844


namespace roots_quadratic_expression_l958_95868

theorem roots_quadratic_expression (m n : ℝ) (h1 : m^2 + m - 2023 = 0) (h2 : n^2 + n - 2023 = 0) :
  m^2 + 2 * m + n = 2022 :=
by
  -- proof steps would go here
  sorry

end roots_quadratic_expression_l958_95868


namespace find_original_manufacturing_cost_l958_95869

noncomputable def originalManufacturingCost (P : ℝ) : ℝ := 0.70 * P

theorem find_original_manufacturing_cost (P : ℝ) (currentCost : ℝ) 
  (h1 : currentCost = 50) 
  (h2 : currentCost = P - 0.50 * P) : originalManufacturingCost P = 70 :=
by
  -- The actual proof steps would go here, but we'll add sorry for now
  sorry

end find_original_manufacturing_cost_l958_95869


namespace find_a33_in_arithmetic_sequence_grid_l958_95842

theorem find_a33_in_arithmetic_sequence_grid 
  (matrix : ℕ → ℕ → ℕ)
  (rows_are_arithmetic : ∀ i, ∃ a b, ∀ j, matrix i j = a + b * (j - 1))
  (columns_are_arithmetic : ∀ j, ∃ c d, ∀ i, matrix i j = c + d * (i - 1))
  : matrix 3 3 = 31 :=
sorry

end find_a33_in_arithmetic_sequence_grid_l958_95842


namespace ozverin_concentration_after_5_times_l958_95835

noncomputable def ozverin_concentration (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ) : ℝ :=
  C₀ * (1 - v / V) ^ n

theorem ozverin_concentration_after_5_times :
  ∀ (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ), V = 0.5 → C₀ = 0.4 → v = 50 → n = 5 →
  ozverin_concentration V C₀ v n = 0.236196 :=
by
  intros V C₀ v n hV hC₀ hv hn
  rw [hV, hC₀, hv, hn]
  simp only [ozverin_concentration]
  norm_num
  sorry

end ozverin_concentration_after_5_times_l958_95835


namespace ratio_naomi_to_katherine_l958_95882

theorem ratio_naomi_to_katherine 
  (katherine_time : ℕ) 
  (naomi_total_time : ℕ) 
  (websites_naomi : ℕ)
  (hk : katherine_time = 20)
  (hn : naomi_total_time = 750)
  (wn : websites_naomi = 30) : 
  naomi_total_time / websites_naomi / katherine_time = 5 / 4 := 
by sorry

end ratio_naomi_to_katherine_l958_95882


namespace valid_S2_example_l958_95822

def satisfies_transformation (S1 S2 : List ℕ) : Prop :=
  S2 = S1.map (λ n => (S1.count n : ℕ))

theorem valid_S2_example : 
  ∃ S1 : List ℕ, satisfies_transformation S1 [1, 2, 1, 1, 2] :=
by
  sorry

end valid_S2_example_l958_95822


namespace power_mod_1000_l958_95857

theorem power_mod_1000 (N : ℤ) (h : Int.gcd N 10 = 1) : (N ^ 101 ≡ N [ZMOD 1000]) :=
  sorry

end power_mod_1000_l958_95857


namespace find_a_l958_95849

theorem find_a (a : ℝ) : (∃ k : ℝ, (x - 2) * (x + k) = x^2 + a * x - 5) ↔ a = 1 / 2 :=
by
  sorry

end find_a_l958_95849


namespace first_digit_l958_95831

-- Definitions and conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

def number (x y : ℕ) : ℕ := 653 * 100 + x * 10 + y

-- Main theorem
theorem first_digit (x y : ℕ) (h₁ : isDivisibleBy (number x y) 80) (h₂ : x + y = 2) : x = 2 :=
sorry

end first_digit_l958_95831


namespace value_of_x_l958_95817

theorem value_of_x (x : ℝ) (h : (0.7 * x) - ((1 / 3) * x) = 110) : x = 300 :=
sorry

end value_of_x_l958_95817


namespace sum_of_squares_l958_95825

theorem sum_of_squares :
  1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 = 5020030 :=
by
  sorry

end sum_of_squares_l958_95825


namespace ellipse_eccentricity_l958_95899

open Real

def ellipse_foci_x_axis (m : ℝ) : Prop :=
  ∃ a b c e,
    a = sqrt m ∧
    b = sqrt 6 ∧
    c = sqrt (m - 6) ∧
    e = c / a ∧
    e = 1 / 2

theorem ellipse_eccentricity (m : ℝ) (h : ellipse_foci_x_axis m) :
  m = 8 := by
  sorry

end ellipse_eccentricity_l958_95899


namespace solve_for_x_l958_95819

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  5 * y ^ 2 + 3 * y + 2 = 3 * (8 * x ^ 2 + y + 1) ↔ x = 1 / Real.sqrt 21 ∨ x = -1 / Real.sqrt 21 :=
by
  sorry

end solve_for_x_l958_95819


namespace max_sum_x_y_min_diff_x_y_l958_95888

def circle_points (x y : ℤ) : Prop := (x - 1)^2 + (y + 2)^2 = 36

theorem max_sum_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x + y ≥ x' + y') :=
  by sorry

theorem min_diff_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x - y ≤ x' - y') :=
  by sorry

end max_sum_x_y_min_diff_x_y_l958_95888


namespace inequality_and_equality_l958_95848

theorem inequality_and_equality (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c ∧ (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end inequality_and_equality_l958_95848


namespace candies_markus_l958_95845

theorem candies_markus (m k s : ℕ) (h_initial_m : m = 9) (h_initial_k : k = 5) (h_total_s : s = 10) :
  (m + s) / 2 = 12 := by
  sorry

end candies_markus_l958_95845


namespace min_A_max_B_l958_95833

-- Part (a): prove A = 15 is the smallest value satisfying the condition
theorem min_A (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : A = 15 := 
sorry

-- Part (b): prove B = 76 is the largest value satisfying the condition
theorem max_B (A B : ℕ) (h : 10 ≤ A ∧ A ≤ 99 ∧ 10 ≤ B ∧ B ≤ 99)
  (eq1 : (A - 5) / A + 4 / B = 1) : B = 76 := 
sorry

end min_A_max_B_l958_95833


namespace cubic_identity_l958_95829

variable (a b c : ℝ)
variable (h1 : a + b + c = 13)
variable (h2 : ab + ac + bc = 30)

theorem cubic_identity : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 :=
by 
  sorry

end cubic_identity_l958_95829


namespace distinct_pairs_reciprocal_sum_l958_95871

theorem distinct_pairs_reciprocal_sum : 
  ∃ (S : Finset (ℕ × ℕ)), (∀ (m n : ℕ), ((m, n) ∈ S) ↔ (m > 0 ∧ n > 0 ∧ (1/m + 1/n = 1/5))) ∧ S.card = 3 :=
sorry

end distinct_pairs_reciprocal_sum_l958_95871


namespace y_intercept_of_linear_function_l958_95887

theorem y_intercept_of_linear_function 
  (k : ℝ)
  (h : (∃ k: ℝ, ∀ x y: ℝ, y = k * (x - 1) ∧ (x, y) = (-1, -2))) : 
  ∃ y : ℝ, (0, y) = (0, -1) :=
by {
  -- Skipping the proof as per the instruction
  sorry
}

end y_intercept_of_linear_function_l958_95887


namespace prime_not_fourth_power_l958_95885

theorem prime_not_fourth_power (p : ℕ) (hp : p > 5) (prime : Prime p) : 
  ¬ ∃ a : ℕ, p = a^4 + 4 :=
by
  sorry

end prime_not_fourth_power_l958_95885


namespace part1_part2_l958_95872

def A := {x : ℝ | x^2 - 2 * x - 8 ≤ 0}
def B (m : ℝ) := {x : ℝ | x^2 - 2 * x + 1 - m^2 ≤ 0}

theorem part1 (m : ℝ) (hm : m = 2) :
  A ∩ {x : ℝ | x < -1 ∨ 3 < x} = {x : ℝ | -2 ≤ x ∧ x < -1 ∨ 3 < x ∧ x ≤ 4} :=
sorry

theorem part2 :
  (∀ x, x ∈ A → x ∈ B (m : ℝ)) ↔ (0 < m ∧ m ≤ 3) :=
sorry

end part1_part2_l958_95872


namespace Walter_receives_49_bananas_l958_95830

-- Definitions of the conditions
def Jefferson_bananas := 56
def Walter_bananas := Jefferson_bananas - 1/4 * Jefferson_bananas
def combined_bananas := Jefferson_bananas + Walter_bananas

-- Statement ensuring the number of bananas Walter gets after the split
theorem Walter_receives_49_bananas:
  combined_bananas / 2 = 49 := by
  sorry

end Walter_receives_49_bananas_l958_95830


namespace translation_coordinates_l958_95803

variable (A B A1 B1 : ℝ × ℝ)

theorem translation_coordinates
  (hA : A = (-1, 0))
  (hB : B = (1, 2))
  (hA1 : A1 = (2, -1))
  (translation_A : A1 = (A.1 + 3, A.2 - 1))
  (translation_B : B1 = (B.1 + 3, B.2 - 1)) :
  B1 = (4, 1) :=
sorry

end translation_coordinates_l958_95803


namespace fraction_covered_by_triangle_l958_95855

structure Point where
  x : ℤ
  y : ℤ

def area_of_triangle (A B C : Point) : ℚ :=
  (1/2 : ℚ) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_grid (length width : ℤ) : ℚ :=
  (length * width : ℚ)

def fraction_of_grid_covered (A B C : Point) (length width : ℤ) : ℚ :=
  (area_of_triangle A B C) / (area_of_grid length width)

theorem fraction_covered_by_triangle :
  fraction_of_grid_covered ⟨2, 4⟩ ⟨7, 2⟩ ⟨6, 5⟩ 8 6 = 13 / 96 :=
by
  sorry

end fraction_covered_by_triangle_l958_95855


namespace probability_at_least_four_8s_in_five_rolls_l958_95858

-- Definitions 
def prob_three_favorable : ℚ := 3 / 10

def prob_at_least_four_times_in_five_rolls : ℚ := 5 * (prob_three_favorable^4) * ((7 : ℚ)/10) + (prob_three_favorable)^5

-- The proof statement
theorem probability_at_least_four_8s_in_five_rolls : prob_at_least_four_times_in_five_rolls = 2859.3 / 10000 :=
by
  sorry

end probability_at_least_four_8s_in_five_rolls_l958_95858


namespace range_of_a_l958_95810

theorem range_of_a
  (a : ℝ)
  (h : ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ (x1 * x2 = 2 * a + 6)) :
  a < -3 :=
by
  sorry

end range_of_a_l958_95810


namespace people_left_gym_l958_95843

theorem people_left_gym (initial : ℕ) (additional : ℕ) (current : ℕ) (H1 : initial = 16) (H2 : additional = 5) (H3 : current = 19) : (initial + additional - current) = 2 :=
by
  sorry

end people_left_gym_l958_95843


namespace remainder_of_sum_of_squares_mod_8_l958_95859

theorem remainder_of_sum_of_squares_mod_8 :
  let a := 445876
  let b := 985420
  let c := 215546
  let d := 656452
  let e := 387295
  a % 8 = 4 → b % 8 = 4 → c % 8 = 6 → d % 8 = 4 → e % 8 = 7 →
  (a^2 + b^2 + c^2 + d^2 + e^2) % 8 = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end remainder_of_sum_of_squares_mod_8_l958_95859


namespace compute_expression_l958_95838

theorem compute_expression : 
  Real.sqrt 8 - (2017 - Real.pi)^0 - 4^(-1 : Int) + (-1/2)^2 = 2 * Real.sqrt 2 - 1 := 
by 
  sorry

end compute_expression_l958_95838


namespace solve_printer_problem_l958_95807

noncomputable def printer_problem : Prop :=
  let rate_A := 10
  let rate_B := rate_A + 8
  let rate_C := rate_B - 4
  let combined_rate := rate_A + rate_B + rate_C
  let total_minutes := 20
  let total_pages := combined_rate * total_minutes
  total_pages = 840

theorem solve_printer_problem : printer_problem :=
by
  sorry

end solve_printer_problem_l958_95807


namespace curve_cartesian_equation_chord_length_l958_95840
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * θ.cos, ρ * θ.sin)

noncomputable def line_parametric (t : ℝ) : ℝ × ℝ :=
  (2 + 1/2 * t, (Real.sqrt 3) / 2 * t)

theorem curve_cartesian_equation :
  ∀ (ρ θ : ℝ), 
    ρ * θ.sin * θ.sin = 8 * θ.cos →
    (ρ * θ.cos) ^ 2 + (ρ * θ.sin) ^ 2 = 
    8 * (ρ * θ.cos) :=
by sorry

theorem chord_length :
  ∀ (t₁ t₂ : ℝ),
    (3 * t₁^2 - 16 * t₁ - 64 = 0) →
    (3 * t₂^2 - 16 * t₂ - 64 = 0) →
    |t₁ - t₂| = (32 / 3) :=
by sorry

end curve_cartesian_equation_chord_length_l958_95840


namespace value_of_one_TV_mixer_blender_l958_95877

variables (M T B : ℝ)

-- The given conditions
def eq1 : Prop := 2 * M + T + B = 10500
def eq2 : Prop := T + M + 2 * B = 14700

-- The problem: find the combined value of one TV, one mixer, and one blender
theorem value_of_one_TV_mixer_blender :
  eq1 M T B → eq2 M T B → (T + M + B = 18900) :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end value_of_one_TV_mixer_blender_l958_95877


namespace horse_total_value_l958_95894

theorem horse_total_value (n : ℕ) (a r : ℕ) (h₁ : n = 32) (h₂ : a = 1) (h₃ : r = 2) :
  (a * (r ^ n - 1) / (r - 1)) = 4294967295 :=
by 
  rw [h₁, h₂, h₃]
  sorry

end horse_total_value_l958_95894


namespace biased_coin_probability_l958_95864

theorem biased_coin_probability :
  let P1 := 3 / 4
  let P2 := 1 / 2
  let P3 := 3 / 4
  let P4 := 2 / 3
  let P5 := 1 / 3
  let P6 := 2 / 5
  let P7 := 3 / 7
  P1 * P2 * P3 * P4 * P5 * P6 * P7 = 3 / 560 :=
by sorry

end biased_coin_probability_l958_95864


namespace water_speed_l958_95836

theorem water_speed (v : ℝ) (h1 : 4 - v > 0) (h2 : 6 * (4 - v) = 12) : v = 2 :=
by
  -- proof steps
  sorry

end water_speed_l958_95836


namespace profit_percentage_l958_95870

def cost_price : ℝ := 60
def selling_price : ℝ := 78

theorem profit_percentage : ((selling_price - cost_price) / cost_price) * 100 = 30 := 
by
  sorry

end profit_percentage_l958_95870


namespace number_divisible_by_23_and_29_l958_95837

theorem number_divisible_by_23_and_29 (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  23 ∣ (200100 * a + 20010 * b + 2001 * c) ∧ 29 ∣ (200100 * a + 20010 * b + 2001 * c) :=
by
  sorry

end number_divisible_by_23_and_29_l958_95837


namespace simplify_polynomial_expression_l958_95818

variable {R : Type*} [CommRing R]

theorem simplify_polynomial_expression (x : R) :
  (2 * x^6 + 3 * x^5 + 4 * x^4 + x^3 + x^2 + x + 20) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 2 * x^2 + 5) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - x^2 + 15 := 
by
  sorry

end simplify_polynomial_expression_l958_95818


namespace number_of_set_B_l958_95823

theorem number_of_set_B (U A B : Finset ℕ) (hU : U.card = 193) (hA_inter_B : (A ∩ B).card = 25) (hA : A.card = 110) (h_not_in_A_or_B : 193 - (A ∪ B).card = 59) : B.card = 49 := 
by
  sorry

end number_of_set_B_l958_95823


namespace pieces_per_box_l958_95876

theorem pieces_per_box (boxes : ℕ) (total_pieces : ℕ) (h_boxes : boxes = 7) (h_total : total_pieces = 21) : 
  total_pieces / boxes = 3 :=
by
  sorry

end pieces_per_box_l958_95876


namespace num_sets_satisfying_union_is_four_l958_95879

variable (M : Set ℕ) (N : Set ℕ)

def num_sets_satisfying_union : Prop :=
  M = {1, 2} ∧ (M ∪ N = {1, 2, 6} → (N = {6} ∨ N = {1, 6} ∨ N = {2, 6} ∨ N = {1, 2, 6}))

theorem num_sets_satisfying_union_is_four :
  (∃ M : Set ℕ, M = {1, 2}) →
  (∃ N : Set ℕ, M ∪ N = {1, 2, 6}) →
  (∃ (num_sets : ℕ), num_sets = 4) :=
by
  sorry

end num_sets_satisfying_union_is_four_l958_95879


namespace bicycle_cost_price_l958_95801

theorem bicycle_cost_price 
  (CP_A : ℝ) 
  (H : CP_A * (1.20 * 0.85 * 1.30 * 0.90) = 285) : 
  CP_A = 285 / (1.20 * 0.85 * 1.30 * 0.90) :=
sorry

end bicycle_cost_price_l958_95801


namespace number_of_terms_in_expansion_l958_95878

def first_factor : List Char := ['x', 'y']
def second_factor : List Char := ['u', 'v', 'w', 'z', 's']

theorem number_of_terms_in_expansion :
  first_factor.length * second_factor.length = 10 :=
by
  -- Lean expects a proof here, but the problem statement specifies to use sorry to skip the proof.
  sorry

end number_of_terms_in_expansion_l958_95878


namespace geometric_sequence_a4_l958_95897

theorem geometric_sequence_a4 {a_2 a_6 a_4 : ℝ} 
  (h1 : ∃ a_1 r : ℝ, a_2 = a_1 * r ∧ a_6 = a_1 * r^5) 
  (h2 : a_2 * a_6 = 64) 
  (h3 : a_2 = a_1 * r)
  (h4 : a_6 = a_1 * r^5)
  : a_4 = 8 :=
by
  sorry

end geometric_sequence_a4_l958_95897


namespace ratio_of_pentagon_to_rectangle_l958_95805

theorem ratio_of_pentagon_to_rectangle (p l : ℕ) 
  (h1 : 5 * p = 30) (h2 : 2 * l + 2 * 5 = 30) : 
  p / l = 3 / 5 :=
by {
  sorry 
}

end ratio_of_pentagon_to_rectangle_l958_95805


namespace initial_distance_between_fred_and_sam_l958_95863

-- Define the conditions as parameters
variables (initial_distance : ℝ)
          (fred_speed sam_speed meeting_distance : ℝ)
          (h_fred_speed : fred_speed = 5)
          (h_sam_speed : sam_speed = 5)
          (h_meeting_distance : meeting_distance = 25)

-- State the theorem
theorem initial_distance_between_fred_and_sam :
  initial_distance = meeting_distance + meeting_distance :=
by
  -- Inline proof structure (sorry means the proof is omitted here)
  sorry

end initial_distance_between_fred_and_sam_l958_95863


namespace correlation_graph_is_scatter_plot_l958_95827

/-- The definition of a scatter plot graph -/
def scatter_plot_graph (x y : ℝ → ℝ) : Prop := 
  ∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)

/-- Prove that the graph representing a set of data for two variables with a correlation is called a "scatter plot" -/
theorem correlation_graph_is_scatter_plot (x y : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, ∀ t : ℝ, (x t, y t) = (t, f t)) → 
  (scatter_plot_graph x y) :=
by
  sorry

end correlation_graph_is_scatter_plot_l958_95827


namespace non_neg_int_solutions_inequality_l958_95852

theorem non_neg_int_solutions_inequality :
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} :=
by
  sorry

end non_neg_int_solutions_inequality_l958_95852


namespace cost_per_meter_l958_95841

def length_of_plot : ℝ := 75
def cost_of_fencing : ℝ := 5300

-- Define breadth as a variable b
def breadth_of_plot (b : ℝ) : Prop := length_of_plot = b + 50

-- Calculate the perimeter given the known breadth
def perimeter (b : ℝ) : ℝ := 2 * length_of_plot + 2 * b

-- Define the proof problem
theorem cost_per_meter (b : ℝ) (hb : breadth_of_plot b) : 5300 / (perimeter b) = 26.5 := by
  -- Given hb: length_of_plot = b + 50, perimeter calculation follows
  sorry

end cost_per_meter_l958_95841


namespace count_valid_pairs_l958_95808

theorem count_valid_pairs : 
  ∃ n : ℕ, n = 3 ∧ ∀ (m n : ℕ), m > n → n ≥ 4 → (m + n) ≤ 40 → (m - n)^2 = m + n → (m, n) ∈ [(10, 6), (15, 10), (21, 15)] := 
by {
  sorry 
}

end count_valid_pairs_l958_95808


namespace tables_needed_l958_95828

open Nat

def base7_to_base10 (n : Nat) : Nat := 
  3 * 7^2 + 1 * 7^1 + 2 * 7^0

theorem tables_needed (attendees_base7 : Nat) (attendees_base10 : Nat) (tables : Nat) :
  attendees_base7 = 312 ∧ attendees_base10 = base7_to_base10 attendees_base7 ∧ attendees_base10 = 156 ∧ tables = attendees_base10 / 3 → tables = 52 := 
by
  intros
  sorry

end tables_needed_l958_95828


namespace problem_l958_95893

theorem problem (a b : ℤ) (h1 : |a - 2| = 5) (h2 : |b| = 9) (h3 : a + b < 0) :
  a - b = 16 ∨ a - b = 6 := 
sorry

end problem_l958_95893
