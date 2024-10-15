import Mathlib

namespace NUMINAMATH_GPT_race_order_count_l1742_174211

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_race_order_count_l1742_174211


namespace NUMINAMATH_GPT_percentage_passed_in_all_three_subjects_l1742_174222

-- Define the given failed percentages as real numbers
def A : ℝ := 0.25  -- 25%
def B : ℝ := 0.48  -- 48%
def C : ℝ := 0.35  -- 35%
def AB : ℝ := 0.27 -- 27%
def AC : ℝ := 0.20 -- 20%
def BC : ℝ := 0.15 -- 15%
def ABC : ℝ := 0.10 -- 10%

-- State the theorem to prove the percentage of students who passed in all three subjects
theorem percentage_passed_in_all_three_subjects : 
  1 - (A + B + C - AB - AC - BC + ABC) = 0.44 :=
by
  sorry

end NUMINAMATH_GPT_percentage_passed_in_all_three_subjects_l1742_174222


namespace NUMINAMATH_GPT_average_after_17th_inning_l1742_174229

variable (A : ℕ)

-- Definition of total runs before the 17th inning
def total_runs_before := 16 * A

-- Definition of new total runs after the 17th inning
def total_runs_after := total_runs_before A + 87

-- Definition of new average after the 17th inning
def new_average := A + 4

-- Definition of new total runs in terms of new average
def new_total_runs := 17 * new_average A

-- The statement we want to prove
theorem average_after_17th_inning : total_runs_after A = new_total_runs A → new_average A = 23 := by
  sorry

end NUMINAMATH_GPT_average_after_17th_inning_l1742_174229


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1742_174298

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h1 : a 1 = 1) (h3 : a 3 = 4) :
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1742_174298


namespace NUMINAMATH_GPT_cindy_correct_result_l1742_174217

theorem cindy_correct_result (x : ℝ) (h: (x - 7) / 5 = 27) : (x - 5) / 7 = 20 :=
by
  sorry

end NUMINAMATH_GPT_cindy_correct_result_l1742_174217


namespace NUMINAMATH_GPT_max_sum_product_l1742_174297

theorem max_sum_product (a b c d : ℝ) (h_nonneg: a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h_sum: a + b + c + d = 200) : 
  ab + bc + cd + da ≤ 10000 := 
sorry

end NUMINAMATH_GPT_max_sum_product_l1742_174297


namespace NUMINAMATH_GPT_gcd_three_numbers_l1742_174214

theorem gcd_three_numbers (a b c : ℕ) (h₁ : a = 13847) (h₂ : b = 21353) (h₃ : c = 34691) : Nat.gcd (Nat.gcd a b) c = 5 := by sorry

end NUMINAMATH_GPT_gcd_three_numbers_l1742_174214


namespace NUMINAMATH_GPT_number_of_real_solutions_l1742_174213

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 50).sum (λ n => (n + 1 : ℝ) / (x - (n + 1 : ℝ)))

theorem number_of_real_solutions : ∃ n : ℕ, n = 51 ∧ ∀ x : ℝ, f x = x + 1 ↔ n = 51 :=
by
  sorry

end NUMINAMATH_GPT_number_of_real_solutions_l1742_174213


namespace NUMINAMATH_GPT_braden_money_box_total_l1742_174280

def initial_money : ℕ := 400

def correct_predictions : ℕ := 3

def betting_rules (correct_predictions : ℕ) : ℕ :=
  match correct_predictions with
  | 1 => 25
  | 2 => 50
  | 3 => 75
  | 4 => 200
  | _ => 0

theorem braden_money_box_total:
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  initial_money + winnings = 700 := 
by
  let winnings := (betting_rules correct_predictions * initial_money) / 100
  show initial_money + winnings = 700
  sorry

end NUMINAMATH_GPT_braden_money_box_total_l1742_174280


namespace NUMINAMATH_GPT_exists_i_with_α_close_to_60_l1742_174291

noncomputable def α : ℕ → ℝ := sorry  -- Placeholder for the function α

theorem exists_i_with_α_close_to_60 :
  ∃ i : ℕ, abs (α i - 60) < 1
:= sorry

end NUMINAMATH_GPT_exists_i_with_α_close_to_60_l1742_174291


namespace NUMINAMATH_GPT_find_unknown_number_l1742_174201

theorem find_unknown_number (x n : ℚ) (h1 : n + 7/x = 6 - 5/x) (h2 : x = 12) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l1742_174201


namespace NUMINAMATH_GPT_total_cows_in_ranch_l1742_174223

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end NUMINAMATH_GPT_total_cows_in_ranch_l1742_174223


namespace NUMINAMATH_GPT_simplify_expression_l1742_174272

theorem simplify_expression :
  ((4 * 7) / (12 * 14)) * ((9 * 12 * 14) / (4 * 7 * 9)) ^ 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1742_174272


namespace NUMINAMATH_GPT_sams_trip_length_l1742_174278

theorem sams_trip_length (total_trip : ℚ) 
  (h1 : total_trip / 4 + 24 + total_trip / 6 = total_trip) : 
  total_trip = 288 / 7 :=
by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_sams_trip_length_l1742_174278


namespace NUMINAMATH_GPT_solve_equation_l1742_174220

theorem solve_equation :
  ∃ x : Real, (x = 2 ∨ x = (-(1:Real) - Real.sqrt 17) / 2) ∧ (x^2 - |x - 1| - 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1742_174220


namespace NUMINAMATH_GPT_distance_ran_by_Juan_l1742_174232

-- Definitions based on the condition
def speed : ℝ := 10 -- in miles per hour
def time : ℝ := 8 -- in hours

-- Theorem statement
theorem distance_ran_by_Juan : speed * time = 80 := by
  sorry

end NUMINAMATH_GPT_distance_ran_by_Juan_l1742_174232


namespace NUMINAMATH_GPT_marbles_count_l1742_174246

-- Define the condition variables
variable (M : ℕ) -- total number of marbles placed on Monday
variable (day2_marbles : ℕ) -- marbles remaining after second day
variable (day3_cleo_marbles : ℕ) -- marbles taken by Cleo on third day

-- Condition definitions
def condition1 : Prop := day2_marbles = 2 * M / 5
def condition2 : Prop := day3_cleo_marbles = (day2_marbles / 2)
def condition3 : Prop := day3_cleo_marbles = 15

-- The theorem to prove
theorem marbles_count : 
  condition1 M day2_marbles → 
  condition2 day2_marbles day3_cleo_marbles → 
  condition3 day3_cleo_marbles → 
  M = 75 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_marbles_count_l1742_174246


namespace NUMINAMATH_GPT_complement_of_union_l1742_174290

def A : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def B : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 2}
def U : Set ℤ := Set.univ

theorem complement_of_union (x : ℤ) : (x ∉ A ∪ B) ↔ ∃ k : ℤ, x = 3 * k := 
by {
  sorry
}

end NUMINAMATH_GPT_complement_of_union_l1742_174290


namespace NUMINAMATH_GPT_sampling_scheme_exists_l1742_174234

theorem sampling_scheme_exists : 
  ∃ (scheme : List ℕ → List (List ℕ)), 
    ∀ (p : List ℕ), p.length = 100 → (scheme p).length = 20 :=
by
  sorry

end NUMINAMATH_GPT_sampling_scheme_exists_l1742_174234


namespace NUMINAMATH_GPT_sector_angle_l1742_174245

theorem sector_angle (r : ℝ) (S_sector : ℝ) (h_r : r = 2) (h_S : S_sector = (2 / 5) * π) : 
  (∃ α : ℝ, S_sector = (1 / 2) * α * r^2 ∧ α = (π / 5)) :=
by
  use π / 5
  sorry

end NUMINAMATH_GPT_sector_angle_l1742_174245


namespace NUMINAMATH_GPT_probability_A_does_not_lose_l1742_174261

theorem probability_A_does_not_lose (p_tie p_A_win : ℚ) (h_tie : p_tie = 1 / 2) (h_A_win : p_A_win = 1 / 3) :
  p_tie + p_A_win = 5 / 6 :=
by sorry

end NUMINAMATH_GPT_probability_A_does_not_lose_l1742_174261


namespace NUMINAMATH_GPT_sylvia_time_to_complete_job_l1742_174283

theorem sylvia_time_to_complete_job (S : ℝ) (h₁ : 18 ≠ 0) (h₂ : 30 ≠ 0)
  (together_rate : (1 / S) + (1 / 30) = 1 / 18) :
  S = 45 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_sylvia_time_to_complete_job_l1742_174283


namespace NUMINAMATH_GPT_mrs_taylor_total_payment_l1742_174208

-- Declaring the price of items and discounts
def price_tv : ℝ := 750
def price_soundbar : ℝ := 300

def discount_tv : ℝ := 0.15
def discount_soundbar : ℝ := 0.10

-- Total number of each items
def num_tv : ℕ := 2
def num_soundbar : ℕ := 3

-- Total cost calculation after discounts
def total_cost_tv := num_tv * price_tv * (1 - discount_tv)
def total_cost_soundbar := num_soundbar * price_soundbar * (1 - discount_soundbar)
def total_cost := total_cost_tv + total_cost_soundbar

-- The theorem we want to prove
theorem mrs_taylor_total_payment : total_cost = 2085 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_mrs_taylor_total_payment_l1742_174208


namespace NUMINAMATH_GPT_solve_a_value_l1742_174253

theorem solve_a_value (a b k : ℝ) 
  (h1 : a^3 * b^2 = k)
  (h2 : a = 5)
  (h3 : b = 2) :
  ∃ a', b = 8 → a' = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_a_value_l1742_174253


namespace NUMINAMATH_GPT_novel_pages_l1742_174251

theorem novel_pages (x : ℕ) (pages_per_day_in_reality : ℕ) (planned_days actual_days : ℕ)
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : pages_per_day_in_reality = x + 20)
  (h4 : pages_per_day_in_reality * actual_days = x * planned_days) :
  x * planned_days = 1200 :=
by
  sorry

end NUMINAMATH_GPT_novel_pages_l1742_174251


namespace NUMINAMATH_GPT_equal_parallelogram_faces_are_rhombuses_l1742_174226

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_equal_parallelogram_faces_are_rhombuses_l1742_174226


namespace NUMINAMATH_GPT_sum_of_possible_a_l1742_174257

theorem sum_of_possible_a (a : ℤ) :
  (∃ x : ℕ, x - (2 - a * x) / 6 = x / 3 - 1) →
  a = -19 :=
sorry

end NUMINAMATH_GPT_sum_of_possible_a_l1742_174257


namespace NUMINAMATH_GPT_find_a_minus_c_l1742_174228

theorem find_a_minus_c (a b c : ℝ) (h1 : (a + b) / 2 = 110) (h2 : (b + c) / 2 = 170) : a - c = -120 :=
by
  sorry

end NUMINAMATH_GPT_find_a_minus_c_l1742_174228


namespace NUMINAMATH_GPT_even_function_odd_function_neither_even_nor_odd_function_l1742_174284

def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

theorem even_function : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

theorem neither_even_nor_odd_function : ∀ x : ℝ, (h (-x) ≠ h x) ∧ (h (-x) ≠ -h x) :=
by
  sorry

end NUMINAMATH_GPT_even_function_odd_function_neither_even_nor_odd_function_l1742_174284


namespace NUMINAMATH_GPT_fraction_of_total_students_l1742_174254

variables (G B T : ℕ) (F : ℚ)

-- Given conditions
axiom ratio_boys_to_girls : (7 : ℚ) / 3 = B / G
axiom total_students : T = B + G
axiom fraction_equals_two_thirds_girls : (2 : ℚ) / 3 * G = F * T

-- Proof goal
theorem fraction_of_total_students : F = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_total_students_l1742_174254


namespace NUMINAMATH_GPT_Jerry_walked_9_miles_l1742_174255

theorem Jerry_walked_9_miles (x : ℕ) (h : 2 * x = 18) : x = 9 := 
by
  sorry

end NUMINAMATH_GPT_Jerry_walked_9_miles_l1742_174255


namespace NUMINAMATH_GPT_functional_identity_l1742_174204

-- Define the set of non-negative integers
def S : Set ℕ := {n | n ≥ 0}

-- Define the function f with the required domain and codomain
def f (n : ℕ) : ℕ := n

-- The hypothesis: the functional equation satisfied by f
axiom functional_equation :
  ∀ m n : ℕ, f (m + f n) = f (f m) + f n

-- The theorem we want to prove
theorem functional_identity (n : ℕ) : f n = n :=
  sorry

end NUMINAMATH_GPT_functional_identity_l1742_174204


namespace NUMINAMATH_GPT_mother_age_l1742_174209

theorem mother_age (x : ℕ) (h1 : 3 * x + x = 40) : 3 * x = 30 :=
by
  -- Here we should provide the proof but for now we use sorry to skip it
  sorry

end NUMINAMATH_GPT_mother_age_l1742_174209


namespace NUMINAMATH_GPT_average_of_t_b_c_29_l1742_174267
-- Importing the entire Mathlib library

theorem average_of_t_b_c_29 (t b c : ℝ) 
  (h : (t + b + c + 14 + 15) / 5 = 12) : 
  (t + b + c + 29) / 4 = 15 :=
by 
  sorry

end NUMINAMATH_GPT_average_of_t_b_c_29_l1742_174267


namespace NUMINAMATH_GPT_find_angle_B_and_sin_ratio_l1742_174263

variable (A B C a b c : ℝ)
variable (h₁ : a * (Real.sin C - Real.sin A) / (Real.sin C + Real.sin B) = c - b)
variable (h₂ : Real.tan B / Real.tan A + Real.tan B / Real.tan C = 4)

theorem find_angle_B_and_sin_ratio :
  B = Real.pi / 3 ∧ Real.sin A / Real.sin C = (3 + Real.sqrt 5) / 2 ∨ Real.sin A / Real.sin C = (3 - Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_and_sin_ratio_l1742_174263


namespace NUMINAMATH_GPT_special_set_exists_l1742_174258

def exists_special_set : Prop :=
  ∃ S : Finset ℕ, S.card = 4004 ∧ 
  (∀ A : Finset ℕ, A ⊆ S ∧ A.card = 2003 → (A.sum id % 2003 ≠ 0))

-- statement with sorry to skip the proof
theorem special_set_exists : exists_special_set :=
sorry

end NUMINAMATH_GPT_special_set_exists_l1742_174258


namespace NUMINAMATH_GPT_remainder_of_M_mod_1000_l1742_174200

def M : ℕ := Nat.choose 9 8

theorem remainder_of_M_mod_1000 : M % 1000 = 9 := by
  sorry

end NUMINAMATH_GPT_remainder_of_M_mod_1000_l1742_174200


namespace NUMINAMATH_GPT_cube_expansion_l1742_174224

variable {a b : ℝ}

theorem cube_expansion (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 :=
  sorry

end NUMINAMATH_GPT_cube_expansion_l1742_174224


namespace NUMINAMATH_GPT_min_value_l1742_174292

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + 2*y = 2) : 
  ∃ c : ℝ, c = 2 ∧ ∀ z, (z = (x^2 / (2*y) + 4*(y^2) / x)) → z ≥ c :=
by
  sorry

end NUMINAMATH_GPT_min_value_l1742_174292


namespace NUMINAMATH_GPT_geometric_sequence_a9_l1742_174225

theorem geometric_sequence_a9
  (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 3 * a 6 = -32)
  (h2 : a 4 + a 5 = 4)
  (hq : ∃ n : ℤ, q = n)
  : a 10 = -256 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a9_l1742_174225


namespace NUMINAMATH_GPT_area_OMVK_l1742_174235

def AreaOfQuadrilateral (S_OKSL S_ONAM S_OMVK : ℝ) : ℝ :=
  let S_ABCD := 4 * (S_OKSL + S_ONAM)
  S_ABCD - S_OKSL - 24 - S_ONAM

theorem area_OMVK {S_OKSL S_ONAM : ℝ} (h_OKSL : S_OKSL = 6) (h_ONAM : S_ONAM = 12) : 
  AreaOfQuadrilateral S_OKSL S_ONAM 30 = 30 :=
by
  sorry

end NUMINAMATH_GPT_area_OMVK_l1742_174235


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1742_174270

theorem solve_quadratic_equation:
  (∀ x : ℝ, (8 * x^2 + 52 * x + 4) / (3 * x + 13) = 2 * x + 3 →
    x = ( -17 + Real.sqrt 569) / 4 ∨ x = ( -17 - Real.sqrt 569) / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1742_174270


namespace NUMINAMATH_GPT_find_extrema_l1742_174281

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem find_extrema :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 6) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 6) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 2 ≤ f x) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 2) :=
by sorry

end NUMINAMATH_GPT_find_extrema_l1742_174281


namespace NUMINAMATH_GPT_juan_stamp_cost_l1742_174288

-- Defining the prices of the stamps
def price_brazil : ℝ := 0.07
def price_peru : ℝ := 0.05

-- Defining the number of stamps from the 70s and 80s
def stamps_brazil_70s : ℕ := 12
def stamps_brazil_80s : ℕ := 15
def stamps_peru_70s : ℕ := 6
def stamps_peru_80s : ℕ := 12

-- Calculating total number of stamps from the 70s and 80s
def total_stamps_brazil : ℕ := stamps_brazil_70s + stamps_brazil_80s
def total_stamps_peru : ℕ := stamps_peru_70s + stamps_peru_80s

-- Calculating total cost
def total_cost_brazil : ℝ := total_stamps_brazil * price_brazil
def total_cost_peru : ℝ := total_stamps_peru * price_peru

def total_cost : ℝ := total_cost_brazil + total_cost_peru

-- Proof statement
theorem juan_stamp_cost : total_cost = 2.79 :=
by
  sorry

end NUMINAMATH_GPT_juan_stamp_cost_l1742_174288


namespace NUMINAMATH_GPT_solve_abs_eq_linear_l1742_174279

theorem solve_abs_eq_linear (x : ℝ) (h : |2 * x - 4| = x + 3) : x = 7 :=
sorry

end NUMINAMATH_GPT_solve_abs_eq_linear_l1742_174279


namespace NUMINAMATH_GPT_band_song_average_l1742_174215

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_band_song_average_l1742_174215


namespace NUMINAMATH_GPT_ellipse_standard_equation_and_point_l1742_174240
  
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2) / 25 + (y^2) / 9 = 1

def exists_dot_product_zero_point (P : ℝ × ℝ) : Prop :=
  let F1 := (-4, 0)
  let F2 := (4, 0)
  (P.1 + 4) * (P.1 - 4) + P.2 * P.2 = 0

theorem ellipse_standard_equation_and_point :
  ∃ (P : ℝ × ℝ), ellipse_equation P.1 P.2 ∧ exists_dot_product_zero_point P ∧ 
    ((P = ((5 * Real.sqrt 7) / 4, 9 / 4)) ∨ (P = (-(5 * Real.sqrt 7) / 4, 9 / 4)) ∨ 
    (P = ((5 * Real.sqrt 7) / 4, -(9 / 4))) ∨ (P = (-(5 * Real.sqrt 7) / 4, -(9 / 4)))) :=
by 
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_and_point_l1742_174240


namespace NUMINAMATH_GPT_percentage_of_60_eq_15_l1742_174206

-- Conditions provided in the problem
def percentage (p : ℚ) : ℚ := p / 100
def num : ℚ := 60
def fraction_of_num (p : ℚ) (n : ℚ) : ℚ := (percentage p) * n

-- Assertion to be proved
theorem percentage_of_60_eq_15 : fraction_of_num 25 num = 15 := 
by 
  show fraction_of_num 25 60 = 15
  sorry

end NUMINAMATH_GPT_percentage_of_60_eq_15_l1742_174206


namespace NUMINAMATH_GPT_fraction_sum_le_41_over_42_l1742_174250

theorem fraction_sum_le_41_over_42 (a b c : ℕ) (h : 1/a + 1/b + 1/c < 1) : 1/a + 1/b + 1/c ≤ 41/42 :=
sorry

end NUMINAMATH_GPT_fraction_sum_le_41_over_42_l1742_174250


namespace NUMINAMATH_GPT_complex_fourth_power_l1742_174271

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end NUMINAMATH_GPT_complex_fourth_power_l1742_174271


namespace NUMINAMATH_GPT_payment_to_z_l1742_174274

-- Definitions of the conditions
def x_work_rate := 1 / 15
def y_work_rate := 1 / 10
def total_payment := 720
def combined_work_rate_xy := x_work_rate + y_work_rate
def combined_work_rate_xyz := 1 / 5
def z_work_rate := combined_work_rate_xyz - combined_work_rate_xy
def z_contribution := z_work_rate * 5
def z_payment := z_contribution * total_payment

-- The statement to be proven
theorem payment_to_z : z_payment = 120 := by
  sorry

end NUMINAMATH_GPT_payment_to_z_l1742_174274


namespace NUMINAMATH_GPT_license_plate_combinations_l1742_174248

-- Definitions of the conditions
def num_consonants : ℕ := 20
def num_vowels : ℕ := 6
def num_digits : ℕ := 10

-- The theorem statement
theorem license_plate_combinations : num_consonants * num_vowels * num_vowels * num_digits = 7200 := by
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l1742_174248


namespace NUMINAMATH_GPT_max_digit_e_l1742_174299

theorem max_digit_e 
  (d e : ℕ) 
  (digits : ∀ (n : ℕ), n ≤ 9) 
  (even_e : e % 2 = 0) 
  (div_9 : (22 + d + e) % 9 = 0) 
  : e ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_digit_e_l1742_174299


namespace NUMINAMATH_GPT_video_game_price_l1742_174231

theorem video_game_price (total_games not_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 10) (h2 : not_working_games = 2) (h3 : total_earnings = 32) :
  ((total_games - not_working_games) > 0) →
  (total_earnings / (total_games - not_working_games)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_video_game_price_l1742_174231


namespace NUMINAMATH_GPT_points_on_opposite_sides_of_line_l1742_174242

theorem points_on_opposite_sides_of_line 
  (a : ℝ) 
  (h : (3 * -3 - 2 * -1 - a) * (3 * 4 - 2 * -6 - a) < 0) : 
  -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_GPT_points_on_opposite_sides_of_line_l1742_174242


namespace NUMINAMATH_GPT_average_speed_l1742_174275

theorem average_speed (D T : ℝ) (hD : D = 200) (hT : T = 6) : D / T = 33.33 := by
  -- Sorry is used to skip the proof, only the statement is provided as per instruction
  sorry

end NUMINAMATH_GPT_average_speed_l1742_174275


namespace NUMINAMATH_GPT_area_of_rectangle_is_588_l1742_174212

-- Define the conditions
def radius_of_circle := 7
def width_of_rectangle := 2 * radius_of_circle
def length_to_width_ratio := 3

-- Define the width and length of the rectangle based on the conditions
def width := width_of_rectangle
def length := length_to_width_ratio * width_of_rectangle

-- Define the area of the rectangle
def area_of_rectangle := length * width

-- The theorem to prove
theorem area_of_rectangle_is_588 : area_of_rectangle = 588 :=
by sorry -- Proof is not required

end NUMINAMATH_GPT_area_of_rectangle_is_588_l1742_174212


namespace NUMINAMATH_GPT_amy_height_l1742_174244

variable (A H N : ℕ)

theorem amy_height (h1 : A = 157) (h2 : A = H + 4) (h3 : H = N + 3) :
  N = 150 := sorry

end NUMINAMATH_GPT_amy_height_l1742_174244


namespace NUMINAMATH_GPT_midpoint_trajectory_extension_trajectory_l1742_174243

-- Define the conditions explicitly

def is_midpoint (M A O : ℝ × ℝ) : Prop :=
  M = ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + P.2 ^ 2 - 8 * P.1 = 0

-- First problem: Trajectory equation of the midpoint M
theorem midpoint_trajectory (M O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hM : is_midpoint M A O) :
  M.1 ^ 2 + M.2 ^ 2 - 4 * M.1 = 0 :=
sorry

-- Define the condition for N
def extension_point (O A N : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * 2 = N.1 - O.1 ∧ (A.2 - O.2) * 2 = N.2 - O.2

-- Second problem: Trajectory equation of the point N
theorem extension_trajectory (N O A : ℝ × ℝ) (hO : O = (0,0)) (hA : on_circle A) (hN : extension_point O A N) :
  N.1 ^ 2 + N.2 ^ 2 - 16 * N.1 = 0 :=
sorry

end NUMINAMATH_GPT_midpoint_trajectory_extension_trajectory_l1742_174243


namespace NUMINAMATH_GPT_y_intercept_of_line_l1742_174210

/-- Let m be the slope of a line and (x_intercept, 0) be the x-intercept of the same line.
    If the line passes through the point (3, 0) and has a slope of -3, then its y-intercept is (0, 9). -/
theorem y_intercept_of_line 
    (m : ℝ) (x_intercept : ℝ) (x1 y1 : ℝ)
    (h1 : m = -3)
    (h2 : (x_intercept, 0) = (3, 0)) :
    (0, -m * x_intercept) = (0, 9) :=
by sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1742_174210


namespace NUMINAMATH_GPT_solve_cubic_eq_l1742_174218

theorem solve_cubic_eq (z : ℂ) : z^3 = 27 ↔ (z = 3 ∨ z = - (3 / 2) + (3 / 2) * Complex.I * Real.sqrt 3 ∨ z = - (3 / 2) - (3 / 2) * Complex.I * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_cubic_eq_l1742_174218


namespace NUMINAMATH_GPT_solution_system_of_equations_solution_system_of_inequalities_l1742_174262

-- Part 1: System of Equations
theorem solution_system_of_equations (x y : ℚ) :
  (3 * x + 2 * y = 13) ∧ (2 * x + 3 * y = -8) ↔ (x = 11 ∧ y = -10) :=
by
  sorry

-- Part 2: System of Inequalities
theorem solution_system_of_inequalities (y : ℚ) :
  ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2) ∧ (2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_system_of_equations_solution_system_of_inequalities_l1742_174262


namespace NUMINAMATH_GPT_good_fractions_expression_l1742_174264

def is_good_fraction (n : ℕ) (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ a + b = n

theorem good_fractions_expression (n : ℕ) (a b : ℕ) :
  n > 1 →
  (∀ a b, b < n → is_good_fraction n a b → ∃ x y, x + y = a / b ∨ x - y = a / b) ↔
  Nat.Prime n :=
by
  sorry

end NUMINAMATH_GPT_good_fractions_expression_l1742_174264


namespace NUMINAMATH_GPT_sin_x_plus_pi_l1742_174293

theorem sin_x_plus_pi {x : ℝ} (hx : Real.sin x = -4 / 5) : Real.sin (x + Real.pi) = 4 / 5 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_sin_x_plus_pi_l1742_174293


namespace NUMINAMATH_GPT_probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l1742_174216

/- Define number of boys and girls -/
def num_boys : ℕ := 5
def num_girls : ℕ := 3

/- Define number of students selected -/
def num_selected : ℕ := 2

/- Define the total number of ways to select -/
def total_ways : ℕ := Nat.choose (num_boys + num_girls) num_selected

/- Define the number of ways to select exactly one girl -/
def ways_one_girl : ℕ := Nat.choose num_girls 1 * Nat.choose num_boys 1

/- Define the number of ways to select at least one girl -/
def ways_at_least_one_girl : ℕ := total_ways - Nat.choose num_boys num_selected

/- Define the first probability: exactly one girl participates -/
def prob_one_girl : ℚ := ways_one_girl / total_ways

/- Define the second probability: exactly one girl given at least one girl -/
def prob_one_girl_given_at_least_one : ℚ := ways_one_girl / ways_at_least_one_girl

theorem probability_of_one_girl : prob_one_girl = 15 / 28 := by
  sorry

theorem conditional_probability_of_one_girl_given_at_least_one : prob_one_girl_given_at_least_one = 5 / 6 := by
  sorry

end NUMINAMATH_GPT_probability_of_one_girl_conditional_probability_of_one_girl_given_at_least_one_l1742_174216


namespace NUMINAMATH_GPT_range_of_k_l1742_174207

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → (-2 < k ∧ k < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1742_174207


namespace NUMINAMATH_GPT_apples_in_third_basket_l1742_174294

theorem apples_in_third_basket (total_apples : ℕ) (x : ℕ) (y : ℕ) 
    (h_total : total_apples = 2014)
    (h_second_basket : 49 + x = total_apples - 2 * y - x - y)
    (h_first_basket : total_apples - 2 * y - x + y = 2 * y)
    : x + y = 655 :=
by
    sorry

end NUMINAMATH_GPT_apples_in_third_basket_l1742_174294


namespace NUMINAMATH_GPT_sum_of_cubes_l1742_174276

theorem sum_of_cubes
  (a b c : ℝ)
  (h₁ : a + b + c = 7)
  (h₂ : ab + ac + bc = 9)
  (h₃ : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l1742_174276


namespace NUMINAMATH_GPT_cone_volume_is_3_6_l1742_174239

-- Define the given conditions
def is_maximum_volume_cone_with_cutoff (cone_volume cutoff_volume : ℝ) : Prop :=
  cutoff_volume = 2 * cone_volume

def volume_difference (cutoff_volume cone_volume difference : ℝ) : Prop :=
  cutoff_volume - cone_volume = difference

-- The theorem to prove the volume of the cone
theorem cone_volume_is_3_6 
  (cone_volume cutoff_volume difference: ℝ)  
  (h1: is_maximum_volume_cone_with_cutoff cone_volume cutoff_volume)
  (h2: volume_difference cutoff_volume cone_volume 3.6) 
  : cone_volume = 3.6 :=
sorry

end NUMINAMATH_GPT_cone_volume_is_3_6_l1742_174239


namespace NUMINAMATH_GPT_vanessa_missed_days_l1742_174268

theorem vanessa_missed_days (V M S : ℕ) 
                           (h1 : V + M + S = 17) 
                           (h2 : V + M = 14) 
                           (h3 : M + S = 12) : 
                           V = 5 :=
sorry

end NUMINAMATH_GPT_vanessa_missed_days_l1742_174268


namespace NUMINAMATH_GPT_find_number_l1742_174265

theorem find_number (x y a : ℝ) (h₁ : x * y = 1) (h₂ : (a ^ ((x + y) ^ 2)) / (a ^ ((x - y) ^ 2)) = 1296) : a = 6 :=
sorry

end NUMINAMATH_GPT_find_number_l1742_174265


namespace NUMINAMATH_GPT_total_area_to_be_painted_l1742_174277

theorem total_area_to_be_painted (length width height partition_length partition_height : ℝ) 
(partition_along_length inside_outside both_sides : Bool)
(h1 : length = 15)
(h2 : width = 12)
(h3 : height = 6)
(h4 : partition_length = 15)
(h5 : partition_height = 6) 
(h_partition_along_length : partition_along_length = true)
(h_inside_outside : inside_outside = true)
(h_both_sides : both_sides = true) :
    let end_wall_area := 2 * 2 * width * height
    let side_wall_area := 2 * 2 * length * height
    let ceiling_area := length * width
    let partition_area := 2 * partition_length * partition_height
    (end_wall_area + side_wall_area + ceiling_area + partition_area) = 1008 :=
by
    sorry

end NUMINAMATH_GPT_total_area_to_be_painted_l1742_174277


namespace NUMINAMATH_GPT_y_intercept_tangent_line_l1742_174273

noncomputable def tangent_line_y_intercept (r1 r2 : ℝ) (c1 c2 : ℝ × ℝ) (htangent: Prop) : ℝ :=
  if r1 = 3 ∧ r2 = 2 ∧ c1 = (3, 0) ∧ c2 = (8, 0) ∧ htangent = true then 6 * Real.sqrt 6 else 0

theorem y_intercept_tangent_line (h : tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6) :
  tangent_line_y_intercept 3 2 (3, 0) (8, 0) true = 6 * Real.sqrt 6 :=
by
  exact h

end NUMINAMATH_GPT_y_intercept_tangent_line_l1742_174273


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1742_174221

theorem necessary_but_not_sufficient (x : ℝ) : (x < 0) -> (x^2 + x < 0 ↔ -1 < x ∧ x < 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1742_174221


namespace NUMINAMATH_GPT_prove_R_value_l1742_174227

noncomputable def geometric_series (Q : ℕ) : ℕ :=
  (2^(Q + 1) - 1)

noncomputable def R (F : ℕ) : ℝ :=
  Real.sqrt (Real.log (1 + F) / Real.log 2)

theorem prove_R_value :
  let F := geometric_series 120
  R F = 11 :=
by
  sorry

end NUMINAMATH_GPT_prove_R_value_l1742_174227


namespace NUMINAMATH_GPT_non_congruent_rectangles_count_l1742_174249

theorem non_congruent_rectangles_count (h w : ℕ) (P : ℕ) (multiple_of_4: ℕ → Prop) :
  P = 80 →
  w ≥ 1 ∧ h ≥ 1 →
  P = 2 * (w + h) →
  (multiple_of_4 w ∨ multiple_of_4 h) →
  (∀ k, multiple_of_4 k ↔ ∃ m, k = 4 * m) →
  ∃ n, n = 5 :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_rectangles_count_l1742_174249


namespace NUMINAMATH_GPT_trig_identity_l1742_174205

noncomputable def tan_eq_neg_4_over_3 (theta : ℝ) : Prop := 
  Real.tan theta = -4 / 3

theorem trig_identity (theta : ℝ) (h : tan_eq_neg_4_over_3 theta) : 
  (Real.cos (π / 2 + θ) - Real.sin (-π - θ)) / (Real.cos (11 * π / 2 - θ) + Real.sin (9 * π / 2 + θ)) = 8 / 7 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1742_174205


namespace NUMINAMATH_GPT_gcd_4320_2550_l1742_174203

-- Definitions for 4320 and 2550
def a : ℕ := 4320
def b : ℕ := 2550

-- Statement to prove the greatest common factor of a and b is 30
theorem gcd_4320_2550 : Nat.gcd a b = 30 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_4320_2550_l1742_174203


namespace NUMINAMATH_GPT_bags_le_40kg_l1742_174289

theorem bags_le_40kg (capacity boxes crates sacks box_weight crate_weight sack_weight bag_weight: ℕ)
  (h_capacity: capacity = 13500)
  (h_boxes: boxes = 100)
  (h_crates: crates = 10)
  (h_sacks: sacks = 50)
  (h_box_weight: box_weight = 100)
  (h_crate_weight: crate_weight = 60)
  (h_sack_weight: sack_weight = 50)
  (h_bag_weight: bag_weight = 40) :
  10 = (capacity - (boxes * box_weight + crates * crate_weight + sacks * sack_weight)) / bag_weight := by 
  sorry

end NUMINAMATH_GPT_bags_le_40kg_l1742_174289


namespace NUMINAMATH_GPT_Tim_driving_hours_l1742_174256

theorem Tim_driving_hours (D T : ℕ) (h1 : T = 2 * D) (h2 : D + T = 15) : D = 5 :=
by
  sorry

end NUMINAMATH_GPT_Tim_driving_hours_l1742_174256


namespace NUMINAMATH_GPT_problem_statement_l1742_174295

theorem problem_statement {x₁ x₂ : ℝ} (h1 : 3 * x₁^2 - 9 * x₁ - 21 = 0) (h2 : 3 * x₂^2 - 9 * x₂ - 21 = 0) :
  (3 * x₁ - 4) * (6 * x₂ - 8) = -202 := sorry

end NUMINAMATH_GPT_problem_statement_l1742_174295


namespace NUMINAMATH_GPT_minimum_value_of_nS_n_l1742_174236

noncomputable def a₁ (d : ℝ) : ℝ := -9/2 * d

noncomputable def S (n : ℕ) (d : ℝ) : ℝ :=
  n / 2 * (2 * a₁ d + (n - 1) * d)

theorem minimum_value_of_nS_n :
  S 10 (2/3) = 0 → S 15 (2/3) = 25 → ∃ (n : ℕ), (n * S n (2/3)) = -48 :=
by 
  intros h10 h15
  sorry

end NUMINAMATH_GPT_minimum_value_of_nS_n_l1742_174236


namespace NUMINAMATH_GPT_xy_difference_squared_l1742_174247

theorem xy_difference_squared (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_xy_difference_squared_l1742_174247


namespace NUMINAMATH_GPT_polynomial_is_positive_for_all_x_l1742_174296

noncomputable def P (x : ℝ) : ℝ := x^12 - x^9 + x^4 - x + 1

theorem polynomial_is_positive_for_all_x (x : ℝ) : P x > 0 := 
by
  dsimp [P]
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_polynomial_is_positive_for_all_x_l1742_174296


namespace NUMINAMATH_GPT_bolton_class_students_l1742_174282

theorem bolton_class_students 
  (S : ℕ) 
  (H1 : 2/5 < 1)
  (H2 : 1/3 < 1)
  (C1 : (2 / 5) * (S:ℝ) + (2 / 5) * (S:ℝ) = 20) : 
  S = 25 := 
by
  sorry

end NUMINAMATH_GPT_bolton_class_students_l1742_174282


namespace NUMINAMATH_GPT_farmer_cows_after_selling_l1742_174230

theorem farmer_cows_after_selling
  (initial_cows : ℕ) (new_cows : ℕ) (quarter_factor : ℕ)
  (h_initial : initial_cows = 51)
  (h_new : new_cows = 5)
  (h_quarter : quarter_factor = 4) :
  initial_cows + new_cows - (initial_cows + new_cows) / quarter_factor = 42 :=
by
  sorry

end NUMINAMATH_GPT_farmer_cows_after_selling_l1742_174230


namespace NUMINAMATH_GPT_rational_k_quadratic_solution_count_l1742_174219

theorem rational_k_quadratic_solution_count (N : ℕ) :
  (N = 98) ↔ 
  (∃ (k : ℚ) (x : ℤ), |k| < 500 ∧ (3 * x^2 + k * x + 7 = 0)) :=
sorry

end NUMINAMATH_GPT_rational_k_quadratic_solution_count_l1742_174219


namespace NUMINAMATH_GPT_find_triangle_sides_l1742_174285

-- Define the variables and conditions
noncomputable def k := 5
noncomputable def c := 12
noncomputable def d := 10

-- Assume the perimeters of the figures
def P1 : ℕ := 74
def P2 : ℕ := 84
def P3 : ℕ := 82

-- Define the equations based on the perimeters
def Equation1 := P2 = P1 + 2 * k
def Equation2 := P3 = P1 + 6 * c - 2 * k

-- The lean theorem proving that the sides of the triangle are as given
theorem find_triangle_sides : 
  (Equation1 ∧ Equation2) →
  (k = 5 ∧ c = 12 ∧ d = 10) :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_sides_l1742_174285


namespace NUMINAMATH_GPT_number_of_squares_in_grid_l1742_174233

-- Grid of size 6 × 6 composed entirely of squares.
def grid_size : Nat := 6

-- Definition of the function that counts the number of squares of a given size in an n × n grid.
def count_squares (n : Nat) (size : Nat) : Nat :=
  (n - size + 1) * (n - size + 1)

noncomputable def total_squares : Nat :=
  List.sum (List.map (count_squares grid_size) (List.range grid_size).tail)  -- Using tail to skip zero size

theorem number_of_squares_in_grid : total_squares = 86 := by
  sorry

end NUMINAMATH_GPT_number_of_squares_in_grid_l1742_174233


namespace NUMINAMATH_GPT_integer_solution_l1742_174266

theorem integer_solution (x : ℤ) (h : (Int.natAbs x - 1) * x ^ 2 - 9 = 1) : x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_l1742_174266


namespace NUMINAMATH_GPT_find_f_1998_l1742_174269

noncomputable def f : ℝ → ℝ := sorry -- Define f as a noncomputable function

theorem find_f_1998 (x : ℝ) (h1 : ∀ x, f (x +1) = f x - 1) (h2 : f 1 = 3997) : f 1998 = 2000 :=
  sorry

end NUMINAMATH_GPT_find_f_1998_l1742_174269


namespace NUMINAMATH_GPT_shop_owner_cheat_selling_percentage_l1742_174241

noncomputable def percentage_cheat_buying : ℝ := 12
noncomputable def profit_percentage : ℝ := 40
noncomputable def percentage_cheat_selling : ℝ := 20

theorem shop_owner_cheat_selling_percentage 
  (percentage_cheat_buying : ℝ := 12)
  (profit_percentage : ℝ := 40) :
  percentage_cheat_selling = 20 := 
sorry

end NUMINAMATH_GPT_shop_owner_cheat_selling_percentage_l1742_174241


namespace NUMINAMATH_GPT_correct_choices_l1742_174238

theorem correct_choices :
  (∃ u : ℝ × ℝ, (2 * u.1 + u.2 + 3 = 0) → u = (1, -2)) ∧
  ¬ (∀ a : ℝ, (a = -1 ↔ a^2 * x - y + 1 = 0 ∧ x - a * y - 2 = 0) → a = -1) ∧
  ((∃ (l : ℝ) (P : ℝ × ℝ), l = x + y - 6 → P = (2, 4) → 2 + 4 = l) → x + y - 6 = 0) ∧
  ((∃ (m b : ℝ), y = m * x + b → b = -2) → y = 3 * x - 2) :=
sorry

end NUMINAMATH_GPT_correct_choices_l1742_174238


namespace NUMINAMATH_GPT_similar_terms_solution_l1742_174259

theorem similar_terms_solution
  (a b : ℝ)
  (m n x y : ℤ)
  (h1 : m - 1 = n - 2 * m)
  (h2 : m + n = 3 * m + n - 4)
  (h3 : m * x + (n - 2) * y = 24)
  (h4 : 2 * m * x + n * y = 46) :
  x = 9 ∧ y = 2 := by
  sorry

end NUMINAMATH_GPT_similar_terms_solution_l1742_174259


namespace NUMINAMATH_GPT_number_of_girls_l1742_174202

theorem number_of_girls (num_vans : ℕ) (students_per_van : ℕ) (num_boys : ℕ) (total_students : ℕ) (num_girls : ℕ) 
(h1 : num_vans = 5) 
(h2 : students_per_van = 28) 
(h3 : num_boys = 60) 
(h4 : total_students = num_vans * students_per_van) 
(h5 : num_girls = total_students - num_boys) : 
num_girls = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1742_174202


namespace NUMINAMATH_GPT_deborah_total_cost_l1742_174237

-- Standard postage per letter
def stdPostage : ℝ := 1.08

-- Additional charge for international shipping per letter
def intlAdditional : ℝ := 0.14

-- Number of domestic and international letters
def numDomestic : ℕ := 2
def numInternational : ℕ := 2

-- Expected total cost for four letters
def expectedTotalCost : ℝ := 4.60

theorem deborah_total_cost :
  (numDomestic * stdPostage) + (numInternational * (stdPostage + intlAdditional)) = expectedTotalCost :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_deborah_total_cost_l1742_174237


namespace NUMINAMATH_GPT_S8_is_80_l1742_174260

variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of sequence

-- Conditions
variable (h_seq : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
variable (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- sum of the first n terms
variable (h_cond : a 3 = 20 - a 6) -- given condition

theorem S8_is_80 (h_seq : ∀ n, a (n + 1) = a n + d) (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) (h_cond : a 3 = 20 - a 6) :
  S 8 = 80 :=
sorry

end NUMINAMATH_GPT_S8_is_80_l1742_174260


namespace NUMINAMATH_GPT_overtime_hours_proof_l1742_174252

-- Define the conditions
variable (regular_pay_rate : ℕ := 3)
variable (regular_hours : ℕ := 40)
variable (overtime_multiplier : ℕ := 2)
variable (total_pay : ℕ := 180)

-- Calculate the regular pay for 40 hours
def regular_pay : ℕ := regular_pay_rate * regular_hours

-- Calculate the extra pay received beyond regular pay
def extra_pay : ℕ := total_pay - regular_pay

-- Calculate overtime pay rate
def overtime_pay_rate : ℕ := overtime_multiplier * regular_pay_rate

-- Calculate the number of overtime hours
def overtime_hours (extra_pay : ℕ) (overtime_pay_rate : ℕ) : ℕ :=
  extra_pay / overtime_pay_rate

-- The theorem to prove
theorem overtime_hours_proof :
  overtime_hours extra_pay overtime_pay_rate = 10 := by
  sorry

end NUMINAMATH_GPT_overtime_hours_proof_l1742_174252


namespace NUMINAMATH_GPT_number_of_boys_in_school_l1742_174287

theorem number_of_boys_in_school (B : ℝ) (h1 : 542.0 = B + 155) : B = 387 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_school_l1742_174287


namespace NUMINAMATH_GPT_hyperbola_focus_y_axis_l1742_174286

theorem hyperbola_focus_y_axis (m : ℝ) :
  (∀ x y : ℝ, (m + 1) * x^2 + (2 - m) * y^2 = 1) → m < -1 :=
sorry

end NUMINAMATH_GPT_hyperbola_focus_y_axis_l1742_174286
