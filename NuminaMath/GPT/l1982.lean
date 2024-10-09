import Mathlib

namespace perimeter_of_similar_triangle_l1982_198234

theorem perimeter_of_similar_triangle (a b c d : ℕ) (h_iso : (a = 12) ∧ (b = 24) ∧ (c = 24)) (h_sim : d = 30) 
  : (d + 2 * b) = 150 := by
  sorry

end perimeter_of_similar_triangle_l1982_198234


namespace greatest_k_dividing_n_l1982_198265

noncomputable def num_divisors (n : ℕ) : ℕ :=
  n.divisors.card

theorem greatest_k_dividing_n (n : ℕ) (h_pos : n > 0)
  (h_n_divisors : num_divisors n = 120)
  (h_5n_divisors : num_divisors (5 * n) = 144) :
  ∃ k : ℕ, 5^k ∣ n ∧ (∀ m : ℕ, 5^m ∣ n → m ≤ k) ∧ k = 4 :=
by sorry

end greatest_k_dividing_n_l1982_198265


namespace base8_subtraction_l1982_198249

theorem base8_subtraction : (52 - 27 : ℕ) = 23 := by sorry

end base8_subtraction_l1982_198249


namespace batsman_average_increase_l1982_198279

theorem batsman_average_increase (A : ℝ) (X : ℝ) (runs_11th_inning : ℝ) (average_11th_inning : ℝ) 
  (h_runs_11th_inning : runs_11th_inning = 85) 
  (h_average_11th_inning : average_11th_inning = 35) 
  (h_eq : (10 * A + runs_11th_inning) / 11 = average_11th_inning) :
  X = 5 := 
by 
  sorry

end batsman_average_increase_l1982_198279


namespace range_of_a_plus_b_at_least_one_nonnegative_l1982_198275

-- Conditions
variable (x : ℝ) (a := x^2 - 1) (b := 2 * x + 2)

-- Proof Problem 1: Prove that the range of a + b is [0, +∞)
theorem range_of_a_plus_b : (a + b) ≥ 0 :=
by sorry

-- Proof Problem 2: Prove by contradiction that at least one of a or b is greater than or equal to 0
theorem at_least_one_nonnegative : ¬(a < 0 ∧ b < 0) :=
by sorry

end range_of_a_plus_b_at_least_one_nonnegative_l1982_198275


namespace total_trip_time_l1982_198217

-- Definitions: conditions from the problem
def time_in_first_country : Nat := 2
def time_in_second_country := 2 * time_in_first_country
def time_in_third_country := 2 * time_in_first_country

-- Statement: prove that the total time spent is 10 weeks
theorem total_trip_time : time_in_first_country + time_in_second_country + time_in_third_country = 10 := by
  sorry

end total_trip_time_l1982_198217


namespace range_of_f_l1982_198203

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - 2 * x)

theorem range_of_f : ∀ y, (∃ x, x ≤ (1 / 2) ∧ f x = y) ↔ y ∈ Set.Iic 1 := by
  sorry

end range_of_f_l1982_198203


namespace intersection_A_B_l1982_198289

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end intersection_A_B_l1982_198289


namespace find_third_sum_l1982_198292

def arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (a 1) + (a 4) + (a 7) = 39 ∧ (a 2) + (a 5) + (a 8) = 33

theorem find_third_sum (a : ℕ → ℝ)
                       (d : ℝ)
                       (h_seq : arithmetic_sequence_sum a d)
                       (a_1 : ℝ) :
  a 1 = a_1 ∧ a 2 = a_1 + d ∧ a 3 = a_1 + 2 * d ∧
  a 4 = a_1 + 3 * d ∧ a 5 = a_1 + 4 * d ∧ a 6 = a_1 + 5 * d ∧
  a 7 = a_1 + 6 * d ∧ a 8 = a_1 + 7 * d ∧ a 9 = a_1 + 8 * d →
  a 3 + a 6 + a 9 = 27 :=
by
  sorry

end find_third_sum_l1982_198292


namespace carl_typing_hours_per_day_l1982_198271

theorem carl_typing_hours_per_day (words_per_minute : ℕ) (total_words : ℕ) (days : ℕ) (hours_per_day : ℕ) :
  words_per_minute = 50 →
  total_words = 84000 →
  days = 7 →
  hours_per_day = (total_words / days) / (words_per_minute * 60) →
  hours_per_day = 4 :=
by
  intros h_word_rate h_total_words h_days h_hrs_formula
  rewrite [h_word_rate, h_total_words, h_days] at h_hrs_formula
  exact h_hrs_formula

end carl_typing_hours_per_day_l1982_198271


namespace expr_undefined_iff_l1982_198201

theorem expr_undefined_iff (x : ℝ) : (x^2 - 9 = 0) ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end expr_undefined_iff_l1982_198201


namespace new_percentage_of_girls_is_5_l1982_198252

theorem new_percentage_of_girls_is_5
  (initial_children : ℕ)
  (percentage_boys : ℕ)
  (added_boys : ℕ)
  (initial_total_boys : ℕ)
  (initial_total_girls : ℕ)
  (new_total_boys : ℕ)
  (new_total_children : ℕ)
  (new_percentage_girls : ℕ)
  (h1 : initial_children = 60)
  (h2 : percentage_boys = 90)
  (h3 : added_boys = 60)
  (h4 : initial_total_boys = (percentage_boys * initial_children / 100))
  (h5 : initial_total_girls = initial_children - initial_total_boys)
  (h6 : new_total_boys = initial_total_boys + added_boys)
  (h7 : new_total_children = initial_children + added_boys)
  (h8 : new_percentage_girls = (initial_total_girls * 100 / new_total_children)) :
  new_percentage_girls = 5 :=
by sorry

end new_percentage_of_girls_is_5_l1982_198252


namespace part1_part2_l1982_198283

def f (x a : ℝ) := x^2 + 4 * a * x + 2 * a + 6

theorem part1 (a : ℝ) : (∃ x : ℝ, f x a = 0) ↔ (a = -1 ∨ a = 3 / 2) := 
by 
  sorry

def g (a : ℝ) := 2 - a * |a + 3|

theorem part2 (a : ℝ) :
  (-1 ≤ a ∧ a ≤ 3 / 2) →
  -19 / 4 ≤ g a ∧ g a ≤ 4 :=
by 
  sorry

end part1_part2_l1982_198283


namespace subtract_digits_value_l1982_198250

theorem subtract_digits_value (A B : ℕ) (h1 : A ≠ B) (h2 : 2 * 1000 + A * 100 + 3 * 10 + 2 - (B * 100 + B * 10 + B) = 1 * 1000 + B * 100 + B * 10 + B) :
  B - A = 3 :=
by
  sorry

end subtract_digits_value_l1982_198250


namespace area_diminished_by_64_percent_l1982_198270

/-- Given a rectangular field where both the length and width are diminished by 40%, 
    prove that the area is diminished by 64%. -/
theorem area_diminished_by_64_percent (L W : ℝ) :
  let L' := 0.6 * L
  let W' := 0.6 * W
  let A := L * W
  let A' := L' * W'
  (A - A') / A * 100 = 64 :=
by
  sorry

end area_diminished_by_64_percent_l1982_198270


namespace arrange_6_books_l1982_198207

theorem arrange_6_books :
  Nat.factorial 6 = 720 :=
by
  sorry

end arrange_6_books_l1982_198207


namespace total_polled_votes_proof_l1982_198231

-- Define the conditions
variables (V : ℕ) -- total number of valid votes
variables (invalid_votes : ℕ) -- number of invalid votes
variables (total_polled_votes : ℕ) -- total polled votes
variables (candidateA_votes candidateB_votes : ℕ) -- votes for candidate A and B respectively

-- Assume the known conditions
variable (h1 : candidateA_votes = 45 * V / 100) -- candidate A got 45% of valid votes
variable (h2 : candidateB_votes = 55 * V / 100) -- candidate B got 55% of valid votes
variable (h3 : candidateB_votes - candidateA_votes = 9000) -- candidate A was defeated by 9000 votes
variable (h4 : invalid_votes = 83) -- there are 83 invalid votes
variable (h5 : total_polled_votes = V + invalid_votes) -- total polled votes is sum of valid and invalid votes

-- Define the theorem to prove
theorem total_polled_votes_proof : total_polled_votes = 90083 :=
by 
  -- Placeholder for the proof
  sorry

end total_polled_votes_proof_l1982_198231


namespace total_cost_correct_l1982_198206

/-- Define the base car rental cost -/
def rental_cost : ℝ := 150

/-- Define cost per mile -/
def cost_per_mile : ℝ := 0.5

/-- Define miles driven on Monday -/
def miles_monday : ℝ := 620

/-- Define miles driven on Thursday -/
def miles_thursday : ℝ := 744

/-- Define the total cost Zach spent -/
def total_cost : ℝ := rental_cost + (miles_monday * cost_per_mile) + (miles_thursday * cost_per_mile)

/-- Prove that the total cost Zach spent is 832 dollars -/
theorem total_cost_correct : total_cost = 832 := by
  sorry

end total_cost_correct_l1982_198206


namespace total_points_seven_players_l1982_198263

theorem total_points_seven_players (S : ℕ) (x : ℕ) 
  (hAlex : Alex_scored = S / 4)
  (hBen : Ben_scored = 2 * S / 7)
  (hCharlie : Charlie_scored = 15)
  (hTotal : S / 4 + 2 * S / 7 + 15 + x = S)
  (hMultiple : S = 56) : 
  x = 11 := 
sorry

end total_points_seven_players_l1982_198263


namespace sum_of_c_and_d_l1982_198274

theorem sum_of_c_and_d (c d : ℝ) 
  (h1 : ∀ x, x ≠ 2 ∧ x ≠ -1 → x^2 + c * x + d ≠ 0)
  (h_asymp_2 : 2^2 + c * 2 + d = 0)
  (h_asymp_neg1 : (-1)^2 + c * (-1) + d = 0) :
  c + d = -3 :=
by 
  -- Proof placeholder
  sorry

end sum_of_c_and_d_l1982_198274


namespace largest_digit_A_l1982_198232

theorem largest_digit_A (A : ℕ) (h1 : (31 + A) % 3 = 0) (h2 : 96 % 4 = 0) : 
  A ≤ 7 ∧ (∀ a, a > 7 → ¬((31 + a) % 3 = 0 ∧ 96 % 4 = 0)) :=
by
  sorry

end largest_digit_A_l1982_198232


namespace number_of_cheesecakes_in_fridge_l1982_198216

section cheesecake_problem

def cheesecakes_on_display : ℕ := 10
def cheesecakes_sold : ℕ := 7
def cheesecakes_left_to_be_sold : ℕ := 18

def cheesecakes_in_fridge (total_display : ℕ) (sold : ℕ) (left : ℕ) : ℕ :=
  left - (total_display - sold)

theorem number_of_cheesecakes_in_fridge :
  cheesecakes_in_fridge cheesecakes_on_display cheesecakes_sold cheesecakes_left_to_be_sold = 15 :=
by
  sorry

end cheesecake_problem

end number_of_cheesecakes_in_fridge_l1982_198216


namespace travel_distance_proof_l1982_198258

-- Definitions based on conditions
def total_distance : ℕ := 369
def amoli_speed : ℕ := 42
def amoli_time : ℕ := 3
def anayet_speed : ℕ := 61
def anayet_time : ℕ := 2

-- Calculate distances traveled
def amoli_distance : ℕ := amoli_speed * amoli_time
def anayet_distance : ℕ := anayet_speed * anayet_time

-- Calculate total distance covered
def total_distance_covered : ℕ := amoli_distance + anayet_distance

-- Define remaining distance to travel
def remaining_distance (total : ℕ) (covered : ℕ) : ℕ := total - covered

-- The theorem to prove
theorem travel_distance_proof : remaining_distance total_distance total_distance_covered = 121 := by
  -- Placeholder for the actual proof
  sorry

end travel_distance_proof_l1982_198258


namespace hyungjun_initial_paint_count_l1982_198257

theorem hyungjun_initial_paint_count (X : ℝ) (h1 : X / 2 - (X / 6 + 5) = 5) : X = 30 :=
sorry

end hyungjun_initial_paint_count_l1982_198257


namespace percentage_of_knives_l1982_198243

def initial_knives : Nat := 6
def initial_forks : Nat := 12
def initial_spoons : Nat := 3 * initial_knives
def traded_knives : Nat := 10
def traded_spoons : Nat := 6

theorem percentage_of_knives :
  100 * (initial_knives + traded_knives) / (initial_knives + initial_forks + initial_spoons - traded_spoons + traded_knives) = 40 := by
  sorry

end percentage_of_knives_l1982_198243


namespace isosceles_triangles_count_isosceles_triangles_l1982_198293

theorem isosceles_triangles (x : ℕ) (b : ℕ) : 
  (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14) → 
  (b = 1 ∧ x = 14 ∨ b = 3 ∧ x = 13 ∨ b = 5 ∧ x = 12 ∨ b = 7 ∧ x = 11 ∨ b = 9 ∧ x = 10) :=
by sorry

theorem count_isosceles_triangles : 
  (∃ x b, (2 * x + b = 29 ∧ b % 2 = 1) ∧ (b < 14)) → 
  (5 = 5) :=
by sorry

end isosceles_triangles_count_isosceles_triangles_l1982_198293


namespace brit_age_after_vacation_l1982_198255

-- Define the given conditions and the final proof question

-- Rebecca's age is 25 years
def rebecca_age : ℕ := 25

-- Brittany is older than Rebecca by 3 years
def brit_age_before_vacation (rebecca_age : ℕ) : ℕ := rebecca_age + 3

-- Brittany goes on a 4-year vacation
def vacation_duration : ℕ := 4

-- Prove that Brittany’s age when she returns from her vacation is 32
theorem brit_age_after_vacation (rebecca_age vacation_duration : ℕ) : brit_age_before_vacation rebecca_age + vacation_duration = 32 :=
by
  sorry

end brit_age_after_vacation_l1982_198255


namespace sum_of_three_consecutive_integers_l1982_198219

theorem sum_of_three_consecutive_integers (n m l : ℕ) (h1 : n + 1 = m) (h2 : m + 1 = l) (h3 : l = 13) : n + m + l = 36 := 
by sorry

end sum_of_three_consecutive_integers_l1982_198219


namespace rhombus_perimeter_l1982_198208

-- Define the lengths of the diagonals
def d1 : ℝ := 5  -- Length of the first diagonal
def d2 : ℝ := 12 -- Length of the second diagonal

-- Calculate the perimeter and state the theorem
theorem rhombus_perimeter : ((d1 / 2)^2 + (d2 / 2)^2).sqrt * 4 = 26 := by
  -- Sorry is placed here to denote the proof
  sorry

end rhombus_perimeter_l1982_198208


namespace license_plate_palindrome_probability_find_m_plus_n_l1982_198256

noncomputable section

open Nat

def is_palindrome {α : Type} (seq : List α) : Prop :=
  seq = seq.reverse

def number_of_three_digit_palindromes : ℕ :=
  10 * 10  -- explanation: 10 choices for the first and last digits, 10 for the middle digit

def total_three_digit_numbers : ℕ :=
  10^3  -- 1000

def prob_three_digit_palindrome : ℚ :=
  number_of_three_digit_palindromes / total_three_digit_numbers

def number_of_three_letter_palindromes : ℕ :=
  26 * 26  -- 26 choices for the first and last letters, 26 for the middle letter

def total_three_letter_combinations : ℕ :=
  26^3  -- 26^3

def prob_three_letter_palindrome : ℚ :=
  number_of_three_letter_palindromes / total_three_letter_combinations

def prob_either_palindrome : ℚ :=
  prob_three_digit_palindrome + prob_three_letter_palindrome - (prob_three_digit_palindrome * prob_three_letter_palindrome)

def m : ℕ := 7
def n : ℕ := 52

theorem license_plate_palindrome_probability :
  prob_either_palindrome = 7 / 52 := sorry

theorem find_m_plus_n :
  m + n = 59 := rfl

end license_plate_palindrome_probability_find_m_plus_n_l1982_198256


namespace kara_forgot_medication_times_l1982_198218

theorem kara_forgot_medication_times :
  let ounces_per_medication := 4
  let medication_times_per_day := 3
  let days_per_week := 7
  let total_weeks := 2
  let total_water_intaken := 160
  let expected_total_water := (ounces_per_medication * medication_times_per_day * days_per_week * total_weeks)
  let water_difference := expected_total_water - total_water_intaken
  let forget_times := water_difference / ounces_per_medication
  forget_times = 2 := by sorry

end kara_forgot_medication_times_l1982_198218


namespace digit_7_count_in_range_l1982_198288

def count_digit_7 : ℕ :=
  let units_place := (107 - 100) / 10 + 1
  let tens_place := 10
  units_place + tens_place

theorem digit_7_count_in_range : count_digit_7 = 20 := by
  sorry

end digit_7_count_in_range_l1982_198288


namespace parabola_focus_coordinates_parabola_distance_to_directrix_l1982_198202

-- Define constants and variables
def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x

noncomputable def focus_coordinates : ℝ × ℝ := (1, 0)

noncomputable def point : ℝ × ℝ := (4, 4)

noncomputable def directrix : ℝ := -1

noncomputable def distance_to_directrix : ℝ := 5

-- Proof statements
theorem parabola_focus_coordinates (x y : ℝ) (h : parabola_equation x y) : 
  focus_coordinates = (1, 0) :=
sorry

theorem parabola_distance_to_directrix (p : ℝ × ℝ) (d : ℝ) (h : p = point) (h_line : d = directrix) : 
  distance_to_directrix = 5 :=
  by
    -- Define and use the distance between point and vertical line formula
    sorry

end parabola_focus_coordinates_parabola_distance_to_directrix_l1982_198202


namespace determine_f_16_l1982_198212

theorem determine_f_16 (a : ℝ) (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x ^ α) →
  (∀ x, a ^ (x - 4) + 1 = 2) →
  f 4 = 2 →
  f 16 = 4 :=
by
  sorry

end determine_f_16_l1982_198212


namespace problem_inequality_l1982_198211

theorem problem_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^6 - a^2 + 4) * (b^6 - b^2 + 4) * (c^6 - c^2 + 4) * (d^6 - d^2 + 4) ≥ (a + b + c + d)^4 :=
by
  sorry

end problem_inequality_l1982_198211


namespace find_multiple_l1982_198229

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end find_multiple_l1982_198229


namespace minimum_additional_squares_needed_to_achieve_symmetry_l1982_198233

def initial_grid : List (ℕ × ℕ) := [(1, 4), (4, 1)] -- Initial shaded squares

def is_symmetric (grid : List (ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ), x ∈ grid → y ∈ grid →
    ((x.1 = 2 * 2 - y.1 ∧ x.2 = y.2) ∨
     (x.1 = y.1 ∧ x.2 = 5 - y.2) ∨
     (x.1 = 2 * 2 - y.1 ∧ x.2 = 5 - y.2))

def additional_squares_needed : ℕ :=
  6 -- As derived in the solution steps, 6 additional squares are needed to achieve symmetry

theorem minimum_additional_squares_needed_to_achieve_symmetry :
  ∀ (initial_shades : List (ℕ × ℕ)),
    initial_shades = initial_grid →
    ∃ (additional : List (ℕ × ℕ)),
      initial_shades ++ additional = symmetric_grid ∧
      additional.length = additional_squares_needed :=
by 
-- skip the proof
sorry

end minimum_additional_squares_needed_to_achieve_symmetry_l1982_198233


namespace largest_n_divides_l1982_198298

theorem largest_n_divides (n : ℕ) (h : 2^n ∣ 5^256 - 1) : n ≤ 10 := sorry

end largest_n_divides_l1982_198298


namespace total_cost_eq_16000_l1982_198227

theorem total_cost_eq_16000 (F M T : ℕ) (n : ℕ) (hF : F = 12000) (hM : M = 200) (hT : T = 16000) :
  T = F + M * n → n = 20 :=
by
  sorry

end total_cost_eq_16000_l1982_198227


namespace student_count_l1982_198272

theorem student_count (ratio : ℝ) (teachers : ℕ) (students : ℕ)
  (h1 : ratio = 27.5)
  (h2 : teachers = 42)
  (h3 : ratio * (teachers : ℝ) = students) :
  students = 1155 :=
sorry

end student_count_l1982_198272


namespace length_of_segment_AB_l1982_198248

noncomputable def speed_relation_first (x v1 v2 : ℝ) : Prop :=
  300 / v1 = (x - 300) / v2

noncomputable def speed_relation_second (x v1 v2 : ℝ) : Prop :=
  (x + 100) / v1 = (x - 100) / v2

theorem length_of_segment_AB :
  (∃ (x v1 v2 : ℝ),
    x > 0 ∧
    v1 > 0 ∧
    v2 > 0 ∧
    speed_relation_first x v1 v2 ∧
    speed_relation_second x v1 v2) →
  ∃ x : ℝ, x = 500 :=
by
  sorry

end length_of_segment_AB_l1982_198248


namespace fraction_of_boxes_loaded_by_day_crew_l1982_198244

-- Definitions based on the conditions
variables (D W : ℕ)  -- Day crew per worker boxes (D) and number of workers (W)

-- Helper Definitions
def boxes_day_crew : ℕ := D * W  -- Total boxes by day crew
def boxes_night_crew : ℕ := (3 * D / 4) * (3 * W / 4)  -- Total boxes by night crew
def total_boxes : ℕ := boxes_day_crew D W + boxes_night_crew D W  -- Total boxes by both crews

-- The main theorem
theorem fraction_of_boxes_loaded_by_day_crew :
  (boxes_day_crew D W : ℚ) / (total_boxes D W : ℚ) = 16/25 :=
by
  sorry

end fraction_of_boxes_loaded_by_day_crew_l1982_198244


namespace big_rectangle_width_l1982_198295

theorem big_rectangle_width
  (W : ℝ)
  (h₁ : ∃ l w : ℝ, l = 40 ∧ w = W)
  (h₂ : ∃ l' w' : ℝ, l' = l / 2 ∧ w' = w / 2)
  (h_area : 200 = l' * w') :
  W = 20 :=
by sorry

end big_rectangle_width_l1982_198295


namespace find_value_of_expression_l1982_198245

variable {a b c d x : ℝ}

-- Conditions
def opposites (a b : ℝ) : Prop := a + b = 0
def reciprocals (c d : ℝ) : Prop := c * d = 1
def abs_three (x : ℝ) : Prop := |x| = 3

-- Proof
theorem find_value_of_expression (h1 : opposites a b) (h2 : reciprocals c d) 
  (h3 : abs_three x) : ∃ res : ℝ, (res = 3 ∨ res = -3) ∧ res = 10 * a + 10 * b + c * d * x :=
by
  sorry

end find_value_of_expression_l1982_198245


namespace polygon_number_of_sides_l1982_198291

-- Define the given conditions
def each_interior_angle (n : ℕ) : ℕ := 120

-- Define the property to calculate the number of sides
def num_sides (each_exterior_angle : ℕ) : ℕ := 360 / each_exterior_angle

-- Statement of the problem
theorem polygon_number_of_sides : num_sides (180 - each_interior_angle 6) = 6 :=
by
  -- Proof is omitted
  sorry

end polygon_number_of_sides_l1982_198291


namespace list_price_of_article_l1982_198262

theorem list_price_of_article (P : ℝ) (h : 0.882 * P = 57.33) : P = 65 :=
by
  sorry

end list_price_of_article_l1982_198262


namespace equation_represents_lines_and_point_l1982_198299

theorem equation_represents_lines_and_point:
    (∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 0 → (x = 1 ∧ y = -2)) ∧
    (∀ x y : ℝ, x^2 - y^2 = 0 → (x = y) ∨ (x = -y)) → 
    (∀ x y : ℝ, ((x - 1)^2 + (y + 2)^2) * (x^2 - y^2) = 0 → 
    ((x = 1 ∧ y = -2) ∨ (x + y = 0) ∨ (x - y = 0))) :=
by
  intros h1 h2 h3
  sorry

end equation_represents_lines_and_point_l1982_198299


namespace last_third_speed_l1982_198268

-- Definitions based on the conditions in the problem statement
def first_third_speed : ℝ := 80
def second_third_speed : ℝ := 30
def average_speed : ℝ := 45

-- Definition of the distance covered variable (non-zero to avoid division by zero)
variable (D : ℝ) (hD : D ≠ 0)

-- The unknown speed during the last third of the distance
noncomputable def V : ℝ := 
  D / ((D / 3 / first_third_speed) + (D / 3 / second_third_speed) + (D / 3 / average_speed))

-- The theorem to prove
theorem last_third_speed : V = 48 :=
by
  sorry

end last_third_speed_l1982_198268


namespace geometric_sequence_a3_a5_l1982_198259

-- Define the geometric sequence condition using a function
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Define the given conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (h1 : is_geometric_seq a)
variable (h2 : a 1 > 0)
variable (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

-- The main goal is to prove: a 3 + a 5 = 5
theorem geometric_sequence_a3_a5 : a 3 + a 5 = 5 :=
by
  simp [is_geometric_seq] at h1
  obtain ⟨q, ⟨hq_pos, hq⟩⟩ := h1
  sorry

end geometric_sequence_a3_a5_l1982_198259


namespace ben_chairs_in_10_days_l1982_198230

noncomputable def chairs_built_per_day (hours_per_shift : ℕ) (hours_per_chair : ℕ) : ℕ :=
  hours_per_shift / hours_per_chair

theorem ben_chairs_in_10_days 
  (hours_per_shift : ℕ)
  (hours_per_chair : ℕ)
  (days: ℕ)
  (h_shift: hours_per_shift = 8)
  (h_chair: hours_per_chair = 5)
  (h_days: days = 10) : 
  chairs_built_per_day hours_per_shift hours_per_chair * days = 10 :=
by 
  -- We insert a placeholder 'sorry' to be replaced by an actual proof.
  sorry

end ben_chairs_in_10_days_l1982_198230


namespace expand_product_l1982_198241

theorem expand_product (x : ℝ) : 2 * (x + 3) * (x + 6) = 2 * x^2 + 18 * x + 36 :=
by
  sorry

end expand_product_l1982_198241


namespace probability_complement_l1982_198294

theorem probability_complement (p : ℝ) (h : p = 0.997) : 1 - p = 0.003 :=
by
  rw [h]
  norm_num

end probability_complement_l1982_198294


namespace find_some_number_l1982_198284

-- Conditions on operations
axiom plus_means_mult (a b : ℕ) : (a + b) = (a * b)
axiom minus_means_plus (a b : ℕ) : (a - b) = (a + b)
axiom mult_means_div (a b : ℕ) : (a * b) = (a / b)
axiom div_means_minus (a b : ℕ) : (a / b) = (a - b)

-- Problem statement
theorem find_some_number (some_number : ℕ) :
  (6 - 9 + some_number * 3 / 25 = 5 ↔
   6 + 9 * some_number / 3 - 25 = 5) ∧
  some_number = 8 := by
  sorry

end find_some_number_l1982_198284


namespace quotient_is_six_l1982_198224

def larger_number (L : ℕ) : Prop := L = 1620
def difference (L S : ℕ) : Prop := L - S = 1365
def division_remainder (L S Q : ℕ) : Prop := L = S * Q + 15

theorem quotient_is_six (L S Q : ℕ) 
  (hL : larger_number L) 
  (hdiff : difference L S) 
  (hdiv : division_remainder L S Q) : Q = 6 :=
sorry

end quotient_is_six_l1982_198224


namespace greatest_pq_plus_r_l1982_198225

theorem greatest_pq_plus_r (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h : p * q + q * r + r * p = 2016) : 
  pq + r ≤ 1008 :=
sorry

end greatest_pq_plus_r_l1982_198225


namespace vehicle_A_must_pass_B_before_B_collides_with_C_l1982_198276

theorem vehicle_A_must_pass_B_before_B_collides_with_C
  (V_A : ℝ) -- speed of vehicle A in mph
  (V_B : ℝ := 40) -- speed of vehicle B in mph
  (V_C : ℝ := 65) -- speed of vehicle C in mph
  (distance_AB : ℝ := 100) -- distance between A and B in ft
  (distance_BC : ℝ := 250) -- initial distance between B and C in ft
  : (V_A > (100 * 65 - 150 * 40) / 250) :=
by {
  sorry
}

end vehicle_A_must_pass_B_before_B_collides_with_C_l1982_198276


namespace ice_cream_sales_l1982_198260

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end ice_cream_sales_l1982_198260


namespace jerry_time_proof_l1982_198280

noncomputable def tom_walk_speed (step_length_tom : ℕ) (pace_tom : ℕ) : ℕ := 
  step_length_tom * pace_tom

noncomputable def tom_distance_to_office (walk_speed_tom : ℕ) (time_tom : ℕ) : ℕ :=
  walk_speed_tom * time_tom

noncomputable def jerry_walk_speed (step_length_jerry : ℕ) (pace_jerry : ℕ) : ℕ :=
  step_length_jerry * pace_jerry

noncomputable def jerry_time_to_office (distance_to_office : ℕ) (walk_speed_jerry : ℕ) : ℚ :=
  distance_to_office / walk_speed_jerry

theorem jerry_time_proof :
  let step_length_tom := 80
  let pace_tom := 85
  let time_tom := 20
  let step_length_jerry := 70
  let pace_jerry := 110
  let office_distance := tom_distance_to_office (tom_walk_speed step_length_tom pace_tom) time_tom
  let jerry_speed := jerry_walk_speed step_length_jerry pace_jerry
  jerry_time_to_office office_distance jerry_speed = 53/3 := 
by
  sorry

end jerry_time_proof_l1982_198280


namespace correct_exponentiation_l1982_198286

theorem correct_exponentiation (a : ℕ) : 
  (a^3 * a^2 = a^5) ∧ ¬(a^3 + a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(a^10 / a^2 = a^5) :=
by
  -- Proof steps and actual mathematical validation will go here.
  -- For now, we skip the actual proof due to the problem requirements.
  sorry

end correct_exponentiation_l1982_198286


namespace black_cars_in_parking_lot_l1982_198226

theorem black_cars_in_parking_lot :
  let total_cars := 3000
  let blue_percent := 0.40
  let red_percent := 0.25
  let green_percent := 0.15
  let yellow_percent := 0.10
  let black_percent := 1 - (blue_percent + red_percent + green_percent + yellow_percent)
  let number_of_black_cars := total_cars * black_percent
  number_of_black_cars = 300 :=
by
  sorry

end black_cars_in_parking_lot_l1982_198226


namespace find_x_plus_y_l1982_198239

variable (x y : ℝ)

theorem find_x_plus_y (h1 : |x| + x + y = 8) (h2 : x + |y| - y = 10) : x + y = 14 / 5 := 
by
  sorry

end find_x_plus_y_l1982_198239


namespace x_days_worked_l1982_198228

theorem x_days_worked (W : ℝ) :
  let x_work_rate := W / 20
  let y_work_rate := W / 24
  let y_days := 12
  let y_work_done := y_work_rate * y_days
  let total_work := W
  let work_done_by_x := (W - y_work_done) / x_work_rate
  work_done_by_x = 10 := 
by
  sorry

end x_days_worked_l1982_198228


namespace sum_of_roots_l1982_198254

open Real

theorem sum_of_roots (r s : ℝ) (P : ℝ → ℝ) (Q : ℝ × ℝ) (m : ℝ) :
  (∀ (x : ℝ), P x = x^2) → 
  Q = (20, 14) → 
  (∀ m : ℝ, (m^2 - 80 * m + 56 < 0) ↔ (r < m ∧ m < s)) →
  r + s = 80 :=
by {
  -- sketched proof goes here
  sorry
}

end sum_of_roots_l1982_198254


namespace isosceles_triangle_vertex_angle_l1982_198267

theorem isosceles_triangle_vertex_angle (α β γ : ℝ) (h_triangle : α + β + γ = 180)
  (h_isosceles : α = β ∨ β = α ∨ α = γ ∨ γ = α ∨ β = γ ∨ γ = β)
  (h_ratio : α / γ = 1 / 4 ∨ γ / α = 1 / 4) :
  (γ = 20 ∨ γ = 120) :=
sorry

end isosceles_triangle_vertex_angle_l1982_198267


namespace eval_at_3_l1982_198269

theorem eval_at_3 : (3^3)^(3^3) = 27^27 :=
by sorry

end eval_at_3_l1982_198269


namespace Diego_total_stamp_cost_l1982_198282

theorem Diego_total_stamp_cost :
  let price_brazil_colombia := 0.07
  let price_peru := 0.05
  let num_brazil_50s := 6
  let num_brazil_60s := 9
  let num_peru_50s := 8
  let num_peru_60s := 5
  let num_colombia_50s := 7
  let num_colombia_60s := 6
  let total_brazil := num_brazil_50s + num_brazil_60s
  let total_peru := num_peru_50s + num_peru_60s
  let total_colombia := num_colombia_50s + num_colombia_60s
  let cost_brazil := total_brazil * price_brazil_colombia
  let cost_peru := total_peru * price_peru
  let cost_colombia := total_colombia * price_brazil_colombia
  cost_brazil + cost_peru + cost_colombia = 2.61 :=
by
  sorry

end Diego_total_stamp_cost_l1982_198282


namespace range_of_b_l1982_198204

open Real

theorem range_of_b {b x x1 x2 : ℝ} 
  (h1 : ∀ x : ℝ, x^2 - b * x + 1 > 0 ↔ x < x1 ∨ x > x2)
  (h2 : x1 < 1)
  (h3 : x2 > 1) : 
  b > 2 := sorry

end range_of_b_l1982_198204


namespace sum_of_coefficients_l1982_198240

theorem sum_of_coefficients : 
  ∃ (a b c d e f g h j k : ℤ), 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) → 
    (a + b + c + d + e + f + g + h + j + k = 92) :=
sorry

end sum_of_coefficients_l1982_198240


namespace max_composite_rel_prime_set_l1982_198242

theorem max_composite_rel_prime_set : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 10 ≤ n ∧ n ≤ 99 ∧ ¬Nat.Prime n) ∧ 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) ∧ 
  S.card = 4 := by
sorry

end max_composite_rel_prime_set_l1982_198242


namespace can_form_triangle_8_6_4_l1982_198277

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle_8_6_4 : can_form_triangle 8 6 4 :=
by
  unfold can_form_triangle
  simp
  exact ⟨by linarith, by linarith, by linarith⟩

end can_form_triangle_8_6_4_l1982_198277


namespace max_profit_300_l1982_198247

noncomputable def total_cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def total_revenue (x : ℝ) : ℝ :=
if x ≤ 400 then (400 * x - (1 / 2) * x^2)
else 80000

noncomputable def total_profit (x : ℝ) : ℝ :=
total_revenue x - total_cost x

theorem max_profit_300 :
    ∃ x : ℝ, (total_profit x = (total_revenue 300 - total_cost 300)) := sorry

end max_profit_300_l1982_198247


namespace required_tiles_0_4m_l1982_198266

-- Defining given conditions
def num_tiles_0_3m : ℕ := 720
def side_length_0_3m : ℝ := 0.3
def side_length_0_4m : ℝ := 0.4

-- The problem statement translated to Lean 4
theorem required_tiles_0_4m : (side_length_0_4m ^ 2) * (405 : ℝ) = (side_length_0_3m ^ 2) * (num_tiles_0_3m : ℝ) := 
by
  -- Skipping the proof
  sorry

end required_tiles_0_4m_l1982_198266


namespace prop_converse_inverse_contrapositive_correct_statements_l1982_198253

-- Defining the proposition and its types
def prop (x : ℕ) : Prop := x > 0 → x^2 ≥ 0
def converse (x : ℕ) : Prop := x^2 ≥ 0 → x > 0
def inverse (x : ℕ) : Prop := ¬ (x > 0) → x^2 < 0
def contrapositive (x : ℕ) : Prop := x^2 < 0 → ¬ (x > 0)

-- The proof problem
theorem prop_converse_inverse_contrapositive_correct_statements :
  (∃! (p : Prop), p = (∀ x : ℕ, converse x) ∨ p = (∀ x : ℕ, inverse x) ∨ p = (∀ x : ℕ, contrapositive x) ∧ p = True) :=
sorry

end prop_converse_inverse_contrapositive_correct_statements_l1982_198253


namespace ratio_grass_area_weeded_l1982_198278

/-- Lucille earns six cents for every weed she pulls. -/
def earnings_per_weed : ℕ := 6

/-- There are eleven weeds in the flower bed. -/
def weeds_flower_bed : ℕ := 11

/-- There are fourteen weeds in the vegetable patch. -/
def weeds_vegetable_patch : ℕ := 14

/-- There are thirty-two weeds in the grass around the fruit trees. -/
def weeds_grass_total : ℕ := 32

/-- Lucille bought a soda for 99 cents on her break. -/
def soda_cost : ℕ := 99

/-- Lucille has 147 cents left after the break. -/
def cents_left : ℕ := 147

/-- Statement to prove: The ratio of the grass area Lucille weeded to the total grass area around the fruit trees is 1:2. -/
theorem ratio_grass_area_weeded :
  (earnings_per_weed * (weeds_flower_bed + weeds_vegetable_patch) + earnings_per_weed * (weeds_flower_bed + (weeds_grass_total - (earnings_per_weed + soda_cost)) / earnings_per_weed) = soda_cost + cents_left)
→ ((earnings_per_weed  * (32 - (147 + 99) / earnings_per_weed)) / weeds_grass_total) = 1 / 2 :=
by
  sorry

end ratio_grass_area_weeded_l1982_198278


namespace center_of_circle_l1982_198261

theorem center_of_circle (x y : ℝ) : 
  (x^2 + y^2 = 6 * x - 10 * y + 9) → 
  (∃ c : ℝ × ℝ, c = (3, -5) ∧ c.1 + c.2 = -2) :=
by
  sorry

end center_of_circle_l1982_198261


namespace rhombus_triangle_area_l1982_198287

theorem rhombus_triangle_area (d1 d2 : ℝ) (h_d1 : d1 = 15) (h_d2 : d2 = 20) :
  ∃ (area : ℝ), area = 75 := 
by
  sorry

end rhombus_triangle_area_l1982_198287


namespace original_bill_amount_l1982_198246

/-- 
If 8 people decided to split the restaurant bill evenly and each paid $314.15 after rounding
up to the nearest cent, then the original bill amount was $2513.20.
-/
theorem original_bill_amount (n : ℕ) (individual_share : ℝ) (total_amount : ℝ) 
  (h1 : n = 8) (h2 : individual_share = 314.15) 
  (h3 : total_amount = n * individual_share) : 
  total_amount = 2513.20 :=
by
  sorry

end original_bill_amount_l1982_198246


namespace a_range_condition_l1982_198215

theorem a_range_condition (a : ℝ) : 
  (∀ x y : ℝ, ((x + a)^2 + (y - a)^2 < 4) → (x = -1 ∧ y = -1)) → 
  -1 < a ∧ a < 1 :=
by
  sorry

end a_range_condition_l1982_198215


namespace possible_apple_counts_l1982_198238

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l1982_198238


namespace pentagon_triangle_ratio_l1982_198213

theorem pentagon_triangle_ratio (p t s : ℝ) 
  (h₁ : 5 * p = 30) 
  (h₂ : 3 * t = 30)
  (h₃ : 4 * s = 30) : 
  p / t = 3 / 5 := by
  sorry

end pentagon_triangle_ratio_l1982_198213


namespace cost_of_one_book_l1982_198221

theorem cost_of_one_book (x : ℝ) : 
  (9 * x = 11) ∧ (13 * x = 15) → x = 1.23 :=
by sorry

end cost_of_one_book_l1982_198221


namespace total_rides_correct_l1982_198236

-- Definitions based on the conditions:
def billy_rides : ℕ := 17
def john_rides : ℕ := 2 * billy_rides
def mother_rides : ℕ := john_rides + 10
def total_rides : ℕ := billy_rides + john_rides + mother_rides

-- The theorem to prove their total bike rides.
theorem total_rides_correct : total_rides = 95 := by
  sorry

end total_rides_correct_l1982_198236


namespace solution_set_of_inequality_l1982_198209

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem solution_set_of_inequality :
  { x : ℝ | f (x - 2) + f (x^2 - 4) < 0 } = Set.Ioo (-3 : ℝ) 2 :=
by
  sorry

end solution_set_of_inequality_l1982_198209


namespace height_of_each_step_l1982_198222

-- Define the number of steps in each staircase
def first_staircase_steps : ℕ := 20
def second_staircase_steps : ℕ := 2 * first_staircase_steps
def third_staircase_steps : ℕ := second_staircase_steps - 10

-- Define the total steps climbed
def total_steps_climbed : ℕ := first_staircase_steps + second_staircase_steps + third_staircase_steps

-- Define the total height climbed
def total_height_climbed : ℝ := 45

-- Prove the height of each step
theorem height_of_each_step : (total_height_climbed / total_steps_climbed) = 0.5 := by
  sorry

end height_of_each_step_l1982_198222


namespace min_groups_required_l1982_198214

/-!
  Prove that if a coach has 30 athletes and wants to arrange them into equal groups with no more than 12 athletes each, 
  then the minimum number of groups required is 3.
-/

theorem min_groups_required (total_athletes : ℕ) (max_athletes_per_group : ℕ) (h_total : total_athletes = 30) (h_max : max_athletes_per_group = 12) :
  ∃ (min_groups : ℕ), min_groups = total_athletes / 10 ∧ (total_athletes % 10 = 0) := by
  sorry

end min_groups_required_l1982_198214


namespace determine_number_on_reverse_side_l1982_198200

variable (n : ℕ) (k : ℕ) (shown_cards : ℕ → Prop)

theorem determine_number_on_reverse_side :
    -- Conditions
    (∀ i, 1 ≤ i ∧ i ≤ n → (shown_cards (i - 1) ↔ shown_cards i)) →
    -- Prove
    (k = 0 ∨ k = n ∨ (1 ≤ k ∧ k < n ∧ (shown_cards (k - 1) ∨ shown_cards (k + 1)))) →
    (∃ j, (j = 1 ∧ k = 0) ∨ (j = n - 1 ∧ k = n) ∨ 
          (j = k - 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k + 1)) ∨ 
          (j = k + 1 ∧ k > 0 ∧ k < n ∧ shown_cards (k - 1))) :=
by
  sorry

end determine_number_on_reverse_side_l1982_198200


namespace probability_of_specific_selection_l1982_198290

/-- 
Given a drawer with 8 forks, 10 spoons, and 6 knives, 
the probability of randomly choosing one fork, one spoon, and one knife when three pieces of silverware are removed equals 120/506.
-/
theorem probability_of_specific_selection :
  let total_pieces := 24
  let total_ways := Nat.choose total_pieces 3
  let favorable_ways := 8 * 10 * 6
  (favorable_ways : ℚ) / total_ways = 120 / 506 := 
by
  sorry

end probability_of_specific_selection_l1982_198290


namespace sum_of_corners_9x9_grid_l1982_198264

theorem sum_of_corners_9x9_grid : 
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  topLeft + topRight + bottomLeft + bottomRight = 164 :=
by {
  let topLeft := 1
  let topRight := 9
  let bottomLeft := 73
  let bottomRight := 81
  show topLeft + topRight + bottomLeft + bottomRight = 164
  sorry
}

end sum_of_corners_9x9_grid_l1982_198264


namespace actual_diameter_of_tissue_l1982_198273

theorem actual_diameter_of_tissue (magnification_factor : ℝ) (magnified_diameter : ℝ) (image_magnified : magnification_factor = 1000 ∧ magnified_diameter = 2) : (1 / magnification_factor) * magnified_diameter = 0.002 :=
by
  sorry

end actual_diameter_of_tissue_l1982_198273


namespace find_third_number_l1982_198251

theorem find_third_number (x : ℝ) : 3 + 33 + x + 3.33 = 369.63 → x = 330.30 :=
by
  intros h
  sorry

end find_third_number_l1982_198251


namespace phil_baseball_cards_left_l1982_198237

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end phil_baseball_cards_left_l1982_198237


namespace division_by_fraction_l1982_198297

theorem division_by_fraction :
  (12 : ℝ) / (1 / 6) = 72 :=
by
  sorry

end division_by_fraction_l1982_198297


namespace factor_quadratic_polynomial_l1982_198205

theorem factor_quadratic_polynomial :
  (∀ x : ℝ, x^4 - 36*x^2 + 25 = (x^2 - 6*x + 5) * (x^2 + 6*x + 5)) :=
by
  sorry

end factor_quadratic_polynomial_l1982_198205


namespace break_even_price_correct_l1982_198235

-- Conditions
def variable_cost_per_handle : ℝ := 0.60
def fixed_cost_per_week : ℝ := 7640
def handles_per_week : ℝ := 1910

-- Define the correct answer for the price per handle to break even
def break_even_price_per_handle : ℝ := 4.60

-- The statement to prove
theorem break_even_price_correct :
  fixed_cost_per_week + (variable_cost_per_handle * handles_per_week) / handles_per_week = break_even_price_per_handle :=
by
  -- The proof is omitted
  sorry

end break_even_price_correct_l1982_198235


namespace product_of_numbers_l1982_198285

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := 
sorry

end product_of_numbers_l1982_198285


namespace relationship_correct_l1982_198223

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem relationship_correct (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) :
  log_base a b < a^b ∧ a^b < b^a :=
by sorry

end relationship_correct_l1982_198223


namespace speeds_of_bodies_l1982_198281

theorem speeds_of_bodies 
  (v1 v2 : ℝ)
  (h1 : 21 * v1 + 10 * v2 = 270)
  (h2 : 51 * v1 + 40 * v2 = 540)
  (h3 : 5 * v2 = 3 * v1): 
  v1 = 10 ∧ v2 = 6 :=
by
  sorry

end speeds_of_bodies_l1982_198281


namespace find_ccb_l1982_198220

theorem find_ccb (a b c : ℕ) 
  (h1: a ≠ b) 
  (h2: a ≠ c) 
  (h3: b ≠ c) 
  (h4: b = 1) 
  (h5: (10 * a + b) ^ 2 = 100 * c + 10 * c + b) 
  (h6: 100 * c + 10 * c + b > 300) : 
  100 * c + 10 * c + b = 441 :=
sorry

end find_ccb_l1982_198220


namespace megan_final_balance_percentage_l1982_198296

noncomputable def initial_balance_usd := 125.0
noncomputable def increase_percentage_babysitting := 0.25
noncomputable def exchange_rate_usd_to_eur_1 := 0.85
noncomputable def decrease_percentage_shoes := 0.20
noncomputable def exchange_rate_eur_to_usd := 1.15
noncomputable def increase_percentage_stocks := 0.15
noncomputable def decrease_percentage_medical := 0.10
noncomputable def exchange_rate_usd_to_eur_2 := 0.88

theorem megan_final_balance_percentage :
  let new_balance_after_babysitting := initial_balance_usd * (1 + increase_percentage_babysitting)
  let balance_in_eur := new_balance_after_babysitting * exchange_rate_usd_to_eur_1
  let balance_after_shoes := balance_in_eur * (1 - decrease_percentage_shoes)
  let balance_back_to_usd := balance_after_shoes * exchange_rate_eur_to_usd
  let balance_after_stocks := balance_back_to_usd * (1 + increase_percentage_stocks)
  let balance_after_medical := balance_after_stocks * (1 - decrease_percentage_medical)
  let final_balance_in_eur := balance_after_medical * exchange_rate_usd_to_eur_2
  let initial_balance_in_eur := initial_balance_usd * exchange_rate_usd_to_eur_1
  (final_balance_in_eur / initial_balance_in_eur) * 100 = 104.75 := by
  sorry

end megan_final_balance_percentage_l1982_198296


namespace dan_money_left_l1982_198210

def money_left (initial : ℝ) (candy_bar : ℝ) (chocolate : ℝ) (soda : ℝ) (gum : ℝ) : ℝ :=
  initial - candy_bar - chocolate - soda - gum

theorem dan_money_left :
  money_left 10 2 3 1.5 1.25 = 2.25 :=
by
  sorry

end dan_money_left_l1982_198210
