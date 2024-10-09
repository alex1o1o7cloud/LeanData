import Mathlib

namespace convert_to_canonical_form_l1231_123107

def quadratic_eqn (x y : ℝ) : ℝ :=
  8 * x^2 + 4 * x * y + 5 * y^2 - 56 * x - 32 * y + 80

def canonical_form (x2 y2 : ℝ) : Prop :=
  (x2^2 / 4) + (y2^2 / 9) = 1

theorem convert_to_canonical_form (x y : ℝ) :
  quadratic_eqn x y = 0 → ∃ (x2 y2 : ℝ), canonical_form x2 y2 :=
sorry

end convert_to_canonical_form_l1231_123107


namespace part1_part2_part3_l1231_123152

def A (x y : ℝ) := 2*x^2 + 3*x*y + 2*y
def B (x y : ℝ) := x^2 - x*y + x

theorem part1 (x y : ℝ) : A x y - 2 * B x y = 5*x*y - 2*x + 2*y := by
  sorry

theorem part2 (x y : ℝ) (h1 : x^2 = 9) (h2 : |y| = 2) :
  A x y - 2 * B x y = 28 ∨ A x y - 2 * B x y = -40 ∨ A x y - 2 * B x y = -20 ∨ A x y - 2 * B x y = 32 := by
  sorry

theorem part3 (y : ℝ) : (∀ x : ℝ, A x y - 2 * B x y = A 0 y - 2 * B 0 y) → y = 2/5 := by
  sorry

end part1_part2_part3_l1231_123152


namespace arithmetic_mean_of_two_digit_multiples_of_9_l1231_123112

theorem arithmetic_mean_of_two_digit_multiples_of_9 :
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  M = 58.5 :=
by
  let a := 18
  let l := 99
  let d := 9
  let n := 10
  let S := (n / 2) * (a + l)
  let M := S / n
  sorry

end arithmetic_mean_of_two_digit_multiples_of_9_l1231_123112


namespace movie_ticket_vs_popcorn_difference_l1231_123192

variable (P : ℝ) -- cost of a bucket of popcorn
variable (d : ℝ) -- cost of a drink
variable (c : ℝ) -- cost of a candy
variable (t : ℝ) -- cost of a movie ticket

-- Given conditions
axiom h1 : t = 8
axiom h2 : d = P + 1
axiom h3 : c = (P + 1) / 2
axiom h4 : t + P + d + c = 22

-- Question rewritten: Prove that the difference between the normal cost of a movie ticket and the cost of a bucket of popcorn is 3.
theorem movie_ticket_vs_popcorn_difference : t - P = 3 :=
by
  sorry

end movie_ticket_vs_popcorn_difference_l1231_123192


namespace first_term_formula_correct_l1231_123162

theorem first_term_formula_correct
  (S n d a : ℝ) 
  (h_sum_formula : S = (n / 2) * (2 * a + (n - 1) * d)) :
  a = (S / n) + (n - 1) * (d / 2) := 
sorry

end first_term_formula_correct_l1231_123162


namespace perpendicular_lines_condition_l1231_123186

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0) ↔ (∀ x y : ℝ, (m - 3) * x + 2 * y - 5 = 0) →
  (m = 3 ∨ m = -2) :=
sorry

end perpendicular_lines_condition_l1231_123186


namespace cube_side_length_l1231_123141

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) (h0 : s ≠ 0) : s = 6 :=
sorry

end cube_side_length_l1231_123141


namespace remainder_of_N_mod_16_is_7_l1231_123197

-- Let N be the product of all odd primes less than 16
def odd_primes : List ℕ := [3, 5, 7, 11, 13]

-- Calculate the product N of these primes
def N : ℕ := odd_primes.foldr (· * ·) 1

-- Prove the remainder of N when divided by 16 is 7
theorem remainder_of_N_mod_16_is_7 : N % 16 = 7 := by
  sorry

end remainder_of_N_mod_16_is_7_l1231_123197


namespace find_x_l1231_123181

theorem find_x (x : ℝ) : 9 - (x / (1 / 3)) + 3 = 3 → x = 3 := by
  intro h
  sorry

end find_x_l1231_123181


namespace probability_of_divisibility_by_7_l1231_123199

noncomputable def count_valid_numbers : Nat :=
  -- Implementation of the count of all five-digit numbers 
  -- such that the sum of the digits is 30 
  sorry

noncomputable def count_divisible_by_7 : Nat :=
  -- Implementation of the count of numbers among these 
  -- which are divisible by 7
  sorry

theorem probability_of_divisibility_by_7 :
  count_divisible_by_7 * 5 = count_valid_numbers :=
sorry

end probability_of_divisibility_by_7_l1231_123199


namespace problem_l1231_123124

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (a b c : ℝ) (h0 : f a b c 0 = f a b c 4) (h1 : f a b c 0 > f a b c 1) :
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l1231_123124


namespace stored_energy_in_doubled_square_l1231_123194

noncomputable def energy (q : ℝ) (d : ℝ) : ℝ := q^2 / d

theorem stored_energy_in_doubled_square (q d : ℝ) (h : energy q d * 4 = 20) :
  energy q (2 * d) * 4 = 10 := by
  -- Add steps: Show that energy proportional to 1/d means energy at 2d is half compared to at d
  sorry

end stored_energy_in_doubled_square_l1231_123194


namespace mail_total_correct_l1231_123142

def Monday_mail : ℕ := 65
def Tuesday_mail : ℕ := Monday_mail + 10
def Wednesday_mail : ℕ := Tuesday_mail - 5
def Thursday_mail : ℕ := Wednesday_mail + 15
def total_mail : ℕ := Monday_mail + Tuesday_mail + Wednesday_mail + Thursday_mail

theorem mail_total_correct : total_mail = 295 := by
  sorry

end mail_total_correct_l1231_123142


namespace relay_race_length_correct_l1231_123143

def relay_race_length (num_members distance_per_member : ℕ) : ℕ := num_members * distance_per_member

theorem relay_race_length_correct :
  relay_race_length 5 30 = 150 :=
by
  -- The proof would go here
  sorry

end relay_race_length_correct_l1231_123143


namespace abs_c_eq_116_l1231_123150

theorem abs_c_eq_116 (a b c : ℤ) (h : Int.gcd a (Int.gcd b c) = 1) 
  (h_eq : a * (Complex.ofReal 3 + Complex.I) ^ 4 + 
          b * (Complex.ofReal 3 + Complex.I) ^ 3 + 
          c * (Complex.ofReal 3 + Complex.I) ^ 2 + 
          b * (Complex.ofReal 3 + Complex.I) + 
          a = 0) : 
  |c| = 116 :=
sorry

end abs_c_eq_116_l1231_123150


namespace function_is_one_l1231_123104

noncomputable def f : ℝ → ℝ := sorry

theorem function_is_one (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x*y) + f (x*z) ≥ 1 + f (x) * f (y*z))
  : ∀ x : ℝ, f x = 1 :=
sorry

end function_is_one_l1231_123104


namespace deriv_y1_deriv_y2_deriv_y3_l1231_123126

variable (x : ℝ)

-- Prove the derivative of y = 3x^3 - 4x is 9x^2 - 4
theorem deriv_y1 : deriv (λ x => 3 * x^3 - 4 * x) x = 9 * x^2 - 4 := by
sorry

-- Prove the derivative of y = (2x - 1)(3x + 2) is 12x + 1
theorem deriv_y2 : deriv (λ x => (2 * x - 1) * (3 * x + 2)) x = 12 * x + 1 := by
sorry

-- Prove the derivative of y = x^2 (x^3 - 4) is 5x^4 - 8x
theorem deriv_y3 : deriv (λ x => x^2 * (x^3 - 4)) x = 5 * x^4 - 8 * x := by
sorry


end deriv_y1_deriv_y2_deriv_y3_l1231_123126


namespace amount_paid_correct_l1231_123125

def initial_debt : ℕ := 100
def hourly_wage : ℕ := 15
def hours_worked : ℕ := 4
def amount_paid_before_work : ℕ := initial_debt - (hourly_wage * hours_worked)

theorem amount_paid_correct : amount_paid_before_work = 40 := by
  sorry

end amount_paid_correct_l1231_123125


namespace probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l1231_123177

variable {p q : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1)

theorem probability_A_miss_at_least_once :
  1 - p^4 = (1 - p^4) := by
sorry

theorem probability_A_2_hits_B_3_hits :
  24 * p^2 * q^3 * (1 - p)^2 * (1 - q) = 24 * p^2 * q^3 * (1 - p)^2 * (1 - q) := by
sorry

end probability_A_miss_at_least_once_probability_A_2_hits_B_3_hits_l1231_123177


namespace football_team_practiced_hours_l1231_123128

-- Define the daily practice hours and missed days as conditions
def daily_practice_hours : ℕ := 6
def missed_days : ℕ := 1

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define a function to calculate the total practiced hours in a week, 
-- given the daily practice hours, missed days, and total days in a week
def total_practiced_hours (daily_hours : ℕ) (missed : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - missed) * daily_hours

-- Prove that the total practiced hours is 36
theorem football_team_practiced_hours :
  total_practiced_hours daily_practice_hours missed_days days_in_week = 36 := 
sorry

end football_team_practiced_hours_l1231_123128


namespace find_xyz_l1231_123103

theorem find_xyz (x y z : ℝ)
  (h1 : x > 4)
  (h2 : y > 4)
  (h3 : z > 4)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  (x, y, z) = (11, 9, 7) :=
by {
  sorry
}

end find_xyz_l1231_123103


namespace initial_pieces_l1231_123113

-- Definitions based on given conditions
variable (left : ℕ) (used : ℕ)
axiom cond1 : left = 93
axiom cond2 : used = 4

-- The mathematical proof problem statement
theorem initial_pieces (left used : ℕ) (cond1 : left = 93) (cond2 : used = 4) : left + used = 97 :=
by
  sorry

end initial_pieces_l1231_123113


namespace percentage_change_area_l1231_123196

theorem percentage_change_area (L B : ℝ) :
  let A_original := L * B
  let A_new := (L / 2) * (3 * B)
  (A_new - A_original) / A_original * 100 = 50 := by
  sorry

end percentage_change_area_l1231_123196


namespace evaluate_expression_l1231_123133

theorem evaluate_expression : 
  (3 / 20 - 5 / 200 + 7 / 2000 : ℚ) = 0.1285 :=
by
  sorry

end evaluate_expression_l1231_123133


namespace eq1_solution_eq2_solution_l1231_123138

theorem eq1_solution (x : ℝ) : (x = 3 + 2 * Real.sqrt 2 ∨ x = 3 - 2 * Real.sqrt 2) ↔ (x^2 - 6 * x + 1 = 0) :=
by
  sorry

theorem eq2_solution (x : ℝ) : (x = 1 ∨ x = -5 / 2) ↔ (2 * x^2 + 3 * x - 5 = 0) :=
by
  sorry

end eq1_solution_eq2_solution_l1231_123138


namespace eating_time_l1231_123153

-- Define the eating rates of Mr. Fat, Mr. Thin, and Mr. Medium
def mrFat_rate := 1 / 15
def mrThin_rate := 1 / 35
def mrMedium_rate := 1 / 25

-- Define the combined eating rate
def combined_rate := mrFat_rate + mrThin_rate + mrMedium_rate

-- Define the amount of cereal to be eaten
def amount_cereal := 5

-- Prove that the time taken to eat the cereal is 2625 / 71 minutes
theorem eating_time : amount_cereal / combined_rate = 2625 / 71 :=
by 
  -- Here should be the proof, but it is skipped
  sorry

end eating_time_l1231_123153


namespace prove_a_ge_neg_one_fourth_l1231_123122

-- Lean 4 statement to reflect the problem
theorem prove_a_ge_neg_one_fourth
  (x y z a : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h1 : x * y - z = a)
  (h2 : y * z - x = a)
  (h3 : z * x - y = a) :
  a ≥ - (1 / 4) :=
sorry

end prove_a_ge_neg_one_fourth_l1231_123122


namespace rod_length_difference_l1231_123120

theorem rod_length_difference (L₁ L₂ : ℝ) (h1 : L₁ + L₂ = 33)
    (h2 : (∀ x : ℝ, x = (2 / 3) * L₁ ∧ x = (4 / 5) * L₂)) :
    abs (L₁ - L₂) = 3 := by
  sorry

end rod_length_difference_l1231_123120


namespace ab_range_l1231_123173

theorem ab_range (a b : ℝ) : (a + b = 1/2) → ab ≤ 1/16 :=
by
  sorry

end ab_range_l1231_123173


namespace Angelina_drive_time_equation_l1231_123123

theorem Angelina_drive_time_equation (t : ℝ) 
    (h_speed1 : ∀ t: ℝ, 70 * t = 70 * t)
    (h_stop : 0.5 = 0.5) 
    (h_speed2 : ∀ t: ℝ, 90 * t = 90 * t) 
    (h_total_distance : 300 = 300) 
    (h_total_time : 4 = 4) 
    : 70 * t + 90 * (3.5 - t) = 300 :=
by
  sorry

end Angelina_drive_time_equation_l1231_123123


namespace fraction_zero_implies_x_neg1_l1231_123146

theorem fraction_zero_implies_x_neg1 (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : x - 1 ≠ 0) : x = -1 := by
  sorry

end fraction_zero_implies_x_neg1_l1231_123146


namespace trains_crossing_time_correct_l1231_123158

def convert_kmph_to_mps (speed_kmph : ℕ) : ℚ := (speed_kmph * 5) / 18

def time_to_cross_each_other 
  (length_train1 length_train2 speed_kmph_train1 speed_kmph_train2 : ℕ) : ℚ :=
  let speed_train1 := convert_kmph_to_mps speed_kmph_train1
  let speed_train2 := convert_kmph_to_mps speed_kmph_train2
  let relative_speed := speed_train2 - speed_train1
  let total_distance := length_train1 + length_train2
  (total_distance : ℚ) / relative_speed

theorem trains_crossing_time_correct :
  time_to_cross_each_other 200 150 40 46 = 210 := by
  sorry

end trains_crossing_time_correct_l1231_123158


namespace total_people_expression_l1231_123159

variable {X : ℕ}

def men (X : ℕ) := 24 * X
def women (X : ℕ) := 12 * X
def teenagers (X : ℕ) := 4 * X
def children (X : ℕ) := X

def total_people (X : ℕ) := men X + women X + teenagers X + children X

theorem total_people_expression (X : ℕ) : total_people X = 41 * X :=
by 
  unfold total_people
  unfold men women teenagers children
  sorry

end total_people_expression_l1231_123159


namespace original_photo_dimensions_l1231_123176

theorem original_photo_dimensions (squares_before : ℕ) 
    (squares_after : ℕ) 
    (vertical_length : ℕ) 
    (horizontal_length : ℕ) 
    (side_length : ℕ)
    (h1 : squares_before = 1812)
    (h2 : squares_after = 2018)
    (h3 : side_length = 1) :
    vertical_length = 101 ∧ horizontal_length = 803 :=
by
    sorry

end original_photo_dimensions_l1231_123176


namespace slopes_of_line_intersecting_ellipse_l1231_123134

theorem slopes_of_line_intersecting_ellipse (m : ℝ) : 
  (m ∈ Set.Iic (-1 / Real.sqrt 624) ∨ m ∈ Set.Ici (1 / Real.sqrt 624)) ↔
  ∃ x y, y = m * x + 10 ∧ 4 * x^2 + 25 * y^2 = 100 :=
by
  sorry

end slopes_of_line_intersecting_ellipse_l1231_123134


namespace group_scores_analysis_l1231_123106

def group1_scores : List ℕ := [92, 90, 91, 96, 96]
def group2_scores : List ℕ := [92, 96, 90, 95, 92]

def median (l : List ℕ) : ℕ := sorry
def mode (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℕ := sorry
def variance (l : List ℕ) : ℕ := sorry

theorem group_scores_analysis :
  median group2_scores = 92 ∧
  mode group1_scores = 96 ∧
  mean group2_scores = 93 ∧
  variance group1_scores = 64 / 10 ∧
  variance group2_scores = 48 / 10 ∧
  variance group2_scores < variance group1_scores :=
by
  sorry

end group_scores_analysis_l1231_123106


namespace opposite_of_neg_3_is_3_l1231_123171

theorem opposite_of_neg_3_is_3 : ∀ (x : ℤ), x = -3 → -x = 3 :=
by
  intro x
  intro h
  rw [h]
  simp

end opposite_of_neg_3_is_3_l1231_123171


namespace reflected_curve_equation_l1231_123100

-- Define the original curve equation
def original_curve (x y : ℝ) : Prop :=
  2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of reflection
def line_of_reflection (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

-- Define the equation of the reflected curve
def reflected_curve (x y : ℝ) : Prop :=
  146 * x^2 - 44 * x * y + 29 * y^2 + 152 * x - 64 * y - 494 = 0

-- Problem: Prove the equation of the reflected curve is as given
theorem reflected_curve_equation (x y : ℝ) :
  (∃ x1 y1 : ℝ, original_curve x1 y1 ∧ line_of_reflection x1 y1 ∧ (x, y) = (x1, y1)) →
  reflected_curve x y :=
by
  intros
  sorry

end reflected_curve_equation_l1231_123100


namespace closest_ratio_l1231_123180

theorem closest_ratio (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : (x + y) / 2 = 3 * Real.sqrt (x * y)) :
  abs (x / y - 34) < abs (x / y - n) :=
by sorry

end closest_ratio_l1231_123180


namespace probability_of_even_distinct_digits_l1231_123160

noncomputable def probability_even_distinct_digits : ℚ :=
  let total_numbers := 9000
  let favorable_numbers := 2744
  favorable_numbers / total_numbers

theorem probability_of_even_distinct_digits : 
  probability_even_distinct_digits = 343 / 1125 :=
by
  sorry

end probability_of_even_distinct_digits_l1231_123160


namespace solve_system_nat_l1231_123174

open Nat

theorem solve_system_nat (x y z t : ℕ) :
  (x + y = z * t ∧ z + t = x * y) ↔ (x, y, z, t) = (1, 5, 2, 3) ∨ (x, y, z, t) = (2, 2, 2, 2) :=
by
  sorry

end solve_system_nat_l1231_123174


namespace total_marbles_in_bag_l1231_123184

theorem total_marbles_in_bag 
  (r b p : ℕ) 
  (h1 : 32 = r)
  (h2 : b = (7 * r) / 4) 
  (h3 : p = (3 * b) / 2) 
  : r + b + p = 172 := 
sorry

end total_marbles_in_bag_l1231_123184


namespace tutors_work_together_again_in_360_days_l1231_123156

theorem tutors_work_together_again_in_360_days :
  Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 :=
by
  sorry

end tutors_work_together_again_in_360_days_l1231_123156


namespace fundraiser_brownies_l1231_123157

-- Definitions derived from the conditions in the problem statement
def brownie_price := 2
def cookie_price := 2
def donut_price := 2

def students_bringing_brownies (B : Nat) := B
def students_bringing_cookies := 20
def students_bringing_donuts := 15

def brownies_per_student := 12
def cookies_per_student := 24
def donuts_per_student := 12

def total_amount_raised := 2040

theorem fundraiser_brownies (B : Nat) :
  24 * B + 20 * 24 * 2 + 15 * 12 * 2 = total_amount_raised → B = 30 :=
by
  sorry

end fundraiser_brownies_l1231_123157


namespace gcd_36_60_l1231_123135

theorem gcd_36_60 : Int.gcd 36 60 = 12 := by
  sorry

end gcd_36_60_l1231_123135


namespace number_of_glasses_l1231_123115

theorem number_of_glasses (oranges_per_glass total_oranges : ℕ) 
  (h1 : oranges_per_glass = 2) 
  (h2 : total_oranges = 12) : 
  total_oranges / oranges_per_glass = 6 := by
  sorry

end number_of_glasses_l1231_123115


namespace count_solutions_congruence_l1231_123116

theorem count_solutions_congruence (x : ℕ) (h1 : 0 < x ∧ x < 50) (h2 : x + 7 ≡ 45 [MOD 22]) : ∃ x1 x2, (x1 ≠ x2) ∧ (0 < x1 ∧ x1 < 50) ∧ (0 < x2 ∧ x2 < 50) ∧ (x1 + 7 ≡ 45 [MOD 22]) ∧ (x2 + 7 ≡ 45 [MOD 22]) ∧ (∀ y, (0 < y ∧ y < 50) ∧ (y + 7 ≡ 45 [MOD 22]) → (y = x1 ∨ y = x2)) :=
by {
  sorry
}

end count_solutions_congruence_l1231_123116


namespace total_students_correct_l1231_123170

-- Definitions based on the conditions
def students_germain : Nat := 13
def students_newton : Nat := 10
def students_young : Nat := 12
def overlap_germain_newton : Nat := 2
def overlap_germain_young : Nat := 1

-- Total distinct students (using inclusion-exclusion principle)
def total_distinct_students : Nat :=
  students_germain + students_newton + students_young - overlap_germain_newton - overlap_germain_young

-- The theorem we want to prove
theorem total_students_correct : total_distinct_students = 32 :=
  by
    -- We state the computation directly; proof is omitted
    sorry

end total_students_correct_l1231_123170


namespace total_children_in_school_l1231_123185

theorem total_children_in_school (B : ℕ) (C : ℕ) 
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 :=
by sorry

end total_children_in_school_l1231_123185


namespace domain_of_log2_function_l1231_123101

theorem domain_of_log2_function :
  {x : ℝ | 2 * x - 1 > 0} = {x : ℝ | x > 1 / 2} :=
by
  sorry

end domain_of_log2_function_l1231_123101


namespace problem_a_b_c_relationship_l1231_123172

theorem problem_a_b_c_relationship (u v a b c : ℝ)
  (h1 : u - v = a)
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) :
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end problem_a_b_c_relationship_l1231_123172


namespace solution_set_of_inequality_l1231_123190

-- Definitions for the problem
def inequality (x : ℝ) : Prop := (1 + x) * (2 - x) * (3 + x^2) > 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l1231_123190


namespace notebook_cost_l1231_123169

-- Define the conditions
def cost_pen := 1
def num_pens := 3
def num_notebooks := 4
def cost_folder := 5
def num_folders := 2
def initial_bill := 50
def change_back := 25

-- Calculate derived values
def total_spent := initial_bill - change_back
def total_cost_pens := num_pens * cost_pen
def total_cost_folders := num_folders * cost_folder
def total_cost_notebooks := total_spent - total_cost_pens - total_cost_folders

-- Calculate the cost per notebook
def cost_per_notebook := total_cost_notebooks / num_notebooks

-- Proof statement
theorem notebook_cost : cost_per_notebook = 3 := by
  sorry

end notebook_cost_l1231_123169


namespace triangles_satisfying_equation_l1231_123198

theorem triangles_satisfying_equation (a b c : ℝ) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  (c ^ 2 - a ^ 2) / b + (b ^ 2 - c ^ 2) / a = b - a →
  (a = b ∨ c ^ 2 = a ^ 2 + b ^ 2) := 
sorry

end triangles_satisfying_equation_l1231_123198


namespace gauss_company_percent_five_years_or_more_l1231_123179

def num_employees_less_1_year (x : ℕ) : ℕ := 5 * x
def num_employees_1_to_2_years (x : ℕ) : ℕ := 5 * x
def num_employees_2_to_3_years (x : ℕ) : ℕ := 8 * x
def num_employees_3_to_4_years (x : ℕ) : ℕ := 3 * x
def num_employees_4_to_5_years (x : ℕ) : ℕ := 2 * x
def num_employees_5_to_6_years (x : ℕ) : ℕ := 2 * x
def num_employees_6_to_7_years (x : ℕ) : ℕ := 2 * x
def num_employees_7_to_8_years (x : ℕ) : ℕ := x
def num_employees_8_to_9_years (x : ℕ) : ℕ := x
def num_employees_9_to_10_years (x : ℕ) : ℕ := x

def total_employees (x : ℕ) : ℕ :=
  num_employees_less_1_year x +
  num_employees_1_to_2_years x +
  num_employees_2_to_3_years x +
  num_employees_3_to_4_years x +
  num_employees_4_to_5_years x +
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

def employees_with_5_years_or_more (x : ℕ) : ℕ :=
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

theorem gauss_company_percent_five_years_or_more (x : ℕ) :
  (employees_with_5_years_or_more x : ℝ) / (total_employees x : ℝ) * 100 = 30 :=
by
  sorry

end gauss_company_percent_five_years_or_more_l1231_123179


namespace eve_total_spend_l1231_123166

def hand_mitts_cost : ℝ := 14.00
def apron_cost : ℝ := 16.00
def utensils_cost : ℝ := 10.00
def knife_cost : ℝ := 2 * utensils_cost
def discount_percent : ℝ := 0.25
def nieces_count : ℕ := 3

def total_cost_before_discount : ℝ :=
  (hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * nieces_count

def discount_amount : ℝ :=
  discount_percent * total_cost_before_discount

def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount_amount

theorem eve_total_spend : total_cost_after_discount = 135.00 := by
  sorry

end eve_total_spend_l1231_123166


namespace value_of_k_l1231_123154

theorem value_of_k :
  3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 :=
by sorry

end value_of_k_l1231_123154


namespace Jake_has_one_more_balloon_than_Allan_l1231_123111

-- Defining the given values
def A : ℕ := 6
def J_initial : ℕ := 3
def J_buy : ℕ := 4
def J_total : ℕ := J_initial + J_buy

-- The theorem statement
theorem Jake_has_one_more_balloon_than_Allan : J_total - A = 1 := 
by
  sorry -- proof goes here

end Jake_has_one_more_balloon_than_Allan_l1231_123111


namespace value_of_w_over_y_l1231_123195

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3) : w / y = 2 / 3 :=
by
  sorry

end value_of_w_over_y_l1231_123195


namespace area_PVZ_is_correct_l1231_123187

noncomputable def area_triangle_PVZ : ℝ :=
  let PQ : ℝ := 8
  let QR : ℝ := 4
  let RV : ℝ := 2
  let WS : ℝ := 3
  let VW : ℝ := PQ - (RV + WS)  -- VW is calculated as 3
  let base_PV : ℝ := PQ
  let height_PVZ : ℝ := QR
  1 / 2 * base_PV * height_PVZ

theorem area_PVZ_is_correct : area_triangle_PVZ = 16 :=
  sorry

end area_PVZ_is_correct_l1231_123187


namespace find_added_number_l1231_123148

theorem find_added_number (a : ℕ → ℝ) (x : ℝ) (h_init : a 1 = 2) (h_a3 : a 3 = 6)
  (h_arith : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- arithmetic sequence condition
  (h_geom : (a 4 + x)^2 = (a 1 + x) * (a 5 + x)) : 
  x = -11 := 
sorry

end find_added_number_l1231_123148


namespace find_r_l1231_123109

theorem find_r (r : ℝ) (AB AD BD : ℝ) (circle_radius : ℝ) (main_circle_radius : ℝ) :
  main_circle_radius = 2 →
  circle_radius = r →
  AB = 2 * r →
  AD = 2 * r →
  BD = 4 + 2 * r →
  (2 * r)^2 + (2 * r)^2 = (4 + 2 * r)^2 →
  r = 4 :=
by 
  intros h_main_radius h_circle_radius h_AB h_AD h_BD h_pythagorean
  sorry

end find_r_l1231_123109


namespace determine_b_value_l1231_123149

theorem determine_b_value 
  (a : ℝ) 
  (b : ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1) 
  (h₂ : 2 * a^(2 - b) + 1 = 3) : 
  b = 2 := 
by 
  sorry

end determine_b_value_l1231_123149


namespace smallest_positive_period_of_h_l1231_123127

-- Definitions of f and g with period 1
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ
axiom T1 : ℝ
axiom T2 : ℝ

-- Given conditions
@[simp] axiom f_periodic : ∀ x, f (x + T1) = f x
@[simp] axiom g_periodic : ∀ x, g (x + T2) = g x
@[simp] axiom T1_eq_one : T1 = 1
@[simp] axiom T2_eq_one : T2 = 1

-- Statement to prove the smallest positive period of h(x) = f(x) + g(x) is 1/k
theorem smallest_positive_period_of_h (k : ℕ) (h : ℝ → ℝ) (hk: k > 0) :
  (∀ x, h (x + 1) = h x) →
  (∀ T > 0, (∀ x, h (x + T) = h x) → (∃ k : ℕ, T = 1 / k)) :=
by sorry

end smallest_positive_period_of_h_l1231_123127


namespace sequence_of_perfect_squares_l1231_123155

theorem sequence_of_perfect_squares (A B C D: ℕ)
(h1: 10 ≤ 10 * A + B) 
(h2 : 10 * A + B < 100) 
(h3 : (10 * A + B) % 3 = 0 ∨ (10 * A + B) % 3 = 1)
(hC : 1 ≤ C ∧ C ≤ 9)
(hD : 1 ≤ D ∧ D ≤ 9)
(hCD : (C + D) % 3 = 0)
(hAB_square : ∃ k₁ : ℕ, k₁^2 = 10 * A + B) 
(hACDB_square : ∃ k₂ : ℕ, k₂^2 = 1000 * A + 100 * C + 10 * D + B) 
(hACCDDB_square : ∃ k₃ : ℕ, k₃^2 = 100000 * A + 10000 * C + 1000 * C + 100 * D + 10 * D + B) :
∀ n: ℕ, ∃ k : ℕ, k^2 = (10^n * A + (10^(n/2) * C) + (10^(n/2) * D) + B) := 
by
  sorry

end sequence_of_perfect_squares_l1231_123155


namespace solve_for_a_l1231_123117

open Complex

theorem solve_for_a (a : ℝ) (h : (2 + a * I) * (a - 2 * I) = -4 * I) : a = 0 :=
sorry

end solve_for_a_l1231_123117


namespace polynomial_divisible_by_a_plus_1_l1231_123193

theorem polynomial_divisible_by_a_plus_1 (a : ℤ) : (3 * a + 5) ^ 2 - 4 ∣ a + 1 := 
by
  sorry

end polynomial_divisible_by_a_plus_1_l1231_123193


namespace faith_work_days_per_week_l1231_123114

theorem faith_work_days_per_week 
  (hourly_wage : ℝ)
  (normal_hours_per_day : ℝ)
  (overtime_hours_per_day : ℝ)
  (weekly_earnings : ℝ)
  (overtime_rate_multiplier : ℝ) :
  hourly_wage = 13.50 → 
  normal_hours_per_day = 8 → 
  overtime_hours_per_day = 2 → 
  weekly_earnings = 675 →
  overtime_rate_multiplier = 1.5 →
  ∀ days_per_week : ℝ, days_per_week = 5 :=
sorry

end faith_work_days_per_week_l1231_123114


namespace sum_of_primes_eq_24_l1231_123164

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

variable (a b c : ℕ)

theorem sum_of_primes_eq_24 (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
    (h4 : a * b + b * c = 119) : a + b + c = 24 :=
sorry

end sum_of_primes_eq_24_l1231_123164


namespace cone_lateral_area_l1231_123110

-- Definitions from the conditions
def radius_base : ℝ := 1 -- in cm
def slant_height : ℝ := 2 -- in cm

-- Statement to be proved: The lateral area of the cone is 2π cm²
theorem cone_lateral_area : 
  1/2 * (2 * π * radius_base) * slant_height = 2 * π :=
by
  sorry

end cone_lateral_area_l1231_123110


namespace mean_of_numbers_is_10_l1231_123129

-- Define the list of numbers
def numbers : List ℕ := [6, 8, 9, 11, 16]

-- Define the length of the list
def n : ℕ := numbers.length

-- Define the sum of the list
def sum_numbers : ℕ := numbers.sum

-- Define the mean (average) calculation for the list
def average : ℕ := sum_numbers / n

-- Prove that the mean of the list is 10
theorem mean_of_numbers_is_10 : average = 10 := by
  sorry

end mean_of_numbers_is_10_l1231_123129


namespace alex_height_l1231_123189

theorem alex_height
  (tree_height: ℚ) (tree_shadow: ℚ) (alex_shadow_in_inches: ℚ)
  (h_tree: tree_height = 50)
  (h_shadow_tree: tree_shadow = 25)
  (h_shadow_alex: alex_shadow_in_inches = 20) :
  ∃ alex_height_in_feet: ℚ, alex_height_in_feet = 10 / 3 :=
by
  sorry

end alex_height_l1231_123189


namespace packs_in_each_set_l1231_123132

variable (cost_per_set cost_per_pack total_savings : ℝ)
variable (x : ℕ)

-- Objecting conditions
axiom cost_set : cost_per_set = 2.5
axiom cost_pack : cost_per_pack = 1.3
axiom savings : total_savings = 1

-- Main proof problem
theorem packs_in_each_set :
  10 * x * cost_per_pack = 10 * cost_per_set + total_savings → x = 2 :=
by
  -- sorry is a placeholder for the proof
  sorry

end packs_in_each_set_l1231_123132


namespace part_I_part_II_l1231_123144

-- Translate the conditions and questions to Lean definition statements.

-- First part of the problem: proving the value of a
theorem part_I (a : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = |a * x - 1|) 
(Hsol : ∀ x, f x ≤ 2 ↔ -6 ≤ x ∧ x ≤ 2) : a = -1 / 2 :=
sorry

-- Second part of the problem: proving the range of m
theorem part_II (m : ℝ) 
(H : ∃ x : ℝ, |4 * x + 1| - |2 * x - 3| ≤ 7 - 3 * m) : m ≤ 7 / 2 :=
sorry

end part_I_part_II_l1231_123144


namespace neg_p_sufficient_for_neg_q_l1231_123191

def p (a : ℝ) := a ≤ 2
def q (a : ℝ) := a * (a - 2) ≤ 0

theorem neg_p_sufficient_for_neg_q (a : ℝ) : ¬ p a → ¬ q a :=
sorry

end neg_p_sufficient_for_neg_q_l1231_123191


namespace cricket_team_members_l1231_123188

-- Define variables and conditions
variable (n : ℕ) -- let n be the number of team members
variable (T : ℕ) -- let T be the total age of the team
variable (average_team_age : ℕ := 24) -- given average age of the team
variable (wicket_keeper_age : ℕ := average_team_age + 3) -- wicket keeper is 3 years older
variable (remaining_players_average_age : ℕ := average_team_age - 1) -- remaining players' average age

-- Given condition which relates to the total age
axiom total_age_condition : T = average_team_age * n

-- Given condition for the total age of remaining players
axiom remaining_players_total_age : T - 24 - 27 = remaining_players_average_age * (n - 2)

-- Prove the number of members in the cricket team
theorem cricket_team_members : n = 5 :=
by
  sorry

end cricket_team_members_l1231_123188


namespace circle_radius_l1231_123119

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y + 1 = 0) : 
    ∃ r : ℝ, r = 2 ∧ (x - 2)^2 + (y - 1)^2 = r^2 :=
by
  sorry

end circle_radius_l1231_123119


namespace value_of_expression_l1231_123140

theorem value_of_expression (V E F t h : ℕ) (H T : ℕ) 
  (h1 : V - E + F = 2)
  (h2 : F = 42)
  (h3 : T = 3)
  (h4 : H = 2)
  (h5 : t + h = 42)
  (h6 : E = (3 * t + 6 * h) / 2) :
  100 * H + 10 * T + V = 328 :=
sorry

end value_of_expression_l1231_123140


namespace lcm_924_660_eq_4620_l1231_123105

theorem lcm_924_660_eq_4620 : Nat.lcm 924 660 = 4620 := 
by
  sorry

end lcm_924_660_eq_4620_l1231_123105


namespace cost_of_rice_l1231_123161

theorem cost_of_rice (x : ℝ) 
  (h : 5 * x + 3 * 5 = 25) : x = 2 :=
by {
  sorry
}

end cost_of_rice_l1231_123161


namespace gum_pack_size_is_5_l1231_123151
noncomputable def find_gum_pack_size (x : ℕ) : Prop :=
  let cherry_initial := 25
  let grape_initial := 40
  let cherry_lost := cherry_initial - 2 * x
  let grape_found := grape_initial + 4 * x
  (cherry_lost * grape_found) = (cherry_initial * grape_initial)

theorem gum_pack_size_is_5 : find_gum_pack_size 5 :=
by
  sorry

end gum_pack_size_is_5_l1231_123151


namespace base7_product_digit_sum_l1231_123168

noncomputable def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 350 => 3 * 7 + 5
  | 217 => 2 * 7 + 1
  | _ => 0

noncomputable def base10_to_base7 (n : Nat) : Nat := 
  if n = 390 then 1065 else 0

noncomputable def digit_sum_in_base7 (n : Nat) : Nat :=
  if n = 1065 then 1 + 0 + 6 + 5 else 0

noncomputable def sum_to_base7 (n : Nat) : Nat :=
  if n = 12 then 15 else 0

theorem base7_product_digit_sum :
  digit_sum_in_base7 (base10_to_base7 (base7_to_base10 350 * base7_to_base10 217)) = 15 :=
by
  sorry

end base7_product_digit_sum_l1231_123168


namespace cos_theta_neg_three_fifths_l1231_123163

theorem cos_theta_neg_three_fifths 
  (θ : ℝ)
  (h1 : Real.sin θ = -4 / 5)
  (h2 : Real.tan θ > 0) : 
  Real.cos θ = -3 / 5 := 
sorry

end cos_theta_neg_three_fifths_l1231_123163


namespace real_roots_m_range_find_value_of_m_l1231_123145

-- Part 1: Prove the discriminant condition for real roots
theorem real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - (2 * m + 3) * x + m^2 + 2 = 0) ↔ m ≥ -1/12 := 
sorry

-- Part 2: Prove the value of m given the condition on roots
theorem find_value_of_m (m : ℝ) (x1 x2 : ℝ) 
  (h : x1^2 + x2^2 = 3 * x1 * x2 - 14)
  (h_roots : x^2 - (2 * m + 3) * x + m^2 + 2 = 0 → (x = x1 ∨ x = x2)) :
  m = 13 := 
sorry

end real_roots_m_range_find_value_of_m_l1231_123145


namespace fraction_squares_sum_l1231_123165

theorem fraction_squares_sum (x a y b z c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : x / a + y / b + z / c = 3) (h2 : a / x + b / y + c / z = -3) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 15 := 
by 
  sorry

end fraction_squares_sum_l1231_123165


namespace solve_n_is_2_l1231_123137

noncomputable def problem_statement (n : ℕ) : Prop :=
  ∃ m : ℕ, 9 * n^2 + 5 * n - 26 = m * (m + 1)

theorem solve_n_is_2 : problem_statement 2 :=
  sorry

end solve_n_is_2_l1231_123137


namespace option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l1231_123131

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Definitions of the given options as natural numbers
def A := 3^3 * 4^4 * 5^5
def B := 3^4 * 4^5 * 5^6
def C := 3^6 * 4^4 * 5^6
def D := 3^5 * 4^6 * 5^5
def E := 3^6 * 4^6 * 5^4

-- Lean statements for each option being a perfect square
theorem option_B_is_perfect_square : is_perfect_square B := sorry
theorem option_C_is_perfect_square : is_perfect_square C := sorry
theorem option_E_is_perfect_square : is_perfect_square E := sorry

end option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l1231_123131


namespace MrsHiltTravelMiles_l1231_123102

theorem MrsHiltTravelMiles
  (one_book_miles : ℕ)
  (finished_books : ℕ)
  (total_miles : ℕ)
  (h1 : one_book_miles = 450)
  (h2 : finished_books = 15)
  (h3 : total_miles = one_book_miles * finished_books) :
  total_miles = 6750 :=
by
  sorry

end MrsHiltTravelMiles_l1231_123102


namespace necessary_and_sufficient_condition_l1231_123178

theorem necessary_and_sufficient_condition (a : ℝ) : (a > 1) ↔ ∀ x : ℝ, (x^2 - 2*x + a > 0) :=
by 
  sorry

end necessary_and_sufficient_condition_l1231_123178


namespace gate_distance_probability_correct_l1231_123139

-- Define the number of gates
def num_gates : ℕ := 15

-- Define the distance between adjacent gates
def distance_between_gates : ℕ := 80

-- Define the maximum distance Dave can walk
def max_distance : ℕ := 320

-- Define the function that calculates the probability
def calculate_probability (num_gates : ℕ) (distance_between_gates : ℕ) (max_distance : ℕ) : ℚ :=
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs :=
    2 * (4 + 5 + 6 + 7) + 7 * 8
  valid_pairs / total_pairs

-- Assert the relevant result and stated answer
theorem gate_distance_probability_correct :
  let m := 10
  let n := 21
  let probability := calculate_probability num_gates distance_between_gates max_distance
  m + n = 31 ∧ probability = (10 / 21 : ℚ) :=
by
  sorry

end gate_distance_probability_correct_l1231_123139


namespace ms_brown_expects_8100_tulips_l1231_123121

def steps_length := 3
def width_steps := 18
def height_steps := 25
def tulips_per_sqft := 2

def width_feet := width_steps * steps_length
def height_feet := height_steps * steps_length
def area_feet := width_feet * height_feet
def expected_tulips := area_feet * tulips_per_sqft

theorem ms_brown_expects_8100_tulips :
  expected_tulips = 8100 := by
  sorry

end ms_brown_expects_8100_tulips_l1231_123121


namespace friends_count_l1231_123130

-- Define the conditions
def num_kids : ℕ := 2
def shonda_present : Prop := True  -- Shonda is present, we may just incorporate it as part of count for clarity
def num_adults : ℕ := 7
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9

-- Define the total number of eggs
def total_eggs : ℕ := num_baskets * eggs_per_basket

-- Define the total number of people
def total_people : ℕ := total_eggs / eggs_per_person

-- Define the number of known people (Shonda, her kids, and the other adults)
def known_people : ℕ := num_kids + 1 + num_adults  -- 1 represents Shonda

-- Define the number of friends
def num_friends : ℕ := total_people - known_people

-- The theorem we need to prove
theorem friends_count : num_friends = 10 :=
by
  sorry

end friends_count_l1231_123130


namespace function_parity_l1231_123182

noncomputable def f : ℝ → ℝ := sorry

-- Condition: f satisfies the functional equation for all x, y in Real numbers
axiom functional_eqn (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y

-- Prove that the function could be either odd or even.
theorem function_parity : (∀ x, f (-x) = f x) ∨ (∀ x, f (-x) = -f x) := 
sorry

end function_parity_l1231_123182


namespace problem_statement_l1231_123136

theorem problem_statement (a n : ℕ) (h1 : 1 ≤ a) (h2 : n = 1) : ∃ m : ℤ, ((a + 1)^n - a^n) = m * n := by
  sorry

end problem_statement_l1231_123136


namespace inequality_holds_l1231_123175

theorem inequality_holds (a b : ℝ) (h1 : a > 1) (h2 : 1 > b) (h3 : b > -1) : a > b^2 :=
by
  sorry

end inequality_holds_l1231_123175


namespace david_money_left_l1231_123183

noncomputable def david_trip (S H : ℝ) : Prop :=
  S + H = 3200 ∧ H = 0.65 * S

theorem david_money_left : ∃ H, david_trip 1939.39 H ∧ |H - 1260.60| < 0.01 := by
  sorry

end david_money_left_l1231_123183


namespace math_problem_l1231_123118

theorem math_problem : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end math_problem_l1231_123118


namespace lines_are_skew_iff_l1231_123108

def line1 (s : ℝ) (b : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * s, 3 + 4 * s, b + 5 * s)

def line2 (v : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6 * v, 2 + 3 * v, 1 + 2 * v)

def lines_intersect (s v b : ℝ) : Prop :=
  line1 s b = line2 v

theorem lines_are_skew_iff (b : ℝ) : ¬ (∃ s v, lines_intersect s v b) ↔ b ≠ 9 :=
by
  sorry

end lines_are_skew_iff_l1231_123108


namespace combined_percentage_increase_l1231_123167

def initial_interval_days : ℝ := 50
def additive_A_effect : ℝ := 0.20
def additive_B_effect : ℝ := 0.30
def additive_C_effect : ℝ := 0.40

theorem combined_percentage_increase :
  ((1 + additive_A_effect) * (1 + additive_B_effect) * (1 + additive_C_effect) - 1) * 100 = 118.4 :=
by
  norm_num
  sorry

end combined_percentage_increase_l1231_123167


namespace solve_for_x_l1231_123147

theorem solve_for_x (x : ℝ) :
  (x - 5)^4 = (1/16)⁻¹ → x = 7 :=
by
  sorry

end solve_for_x_l1231_123147
