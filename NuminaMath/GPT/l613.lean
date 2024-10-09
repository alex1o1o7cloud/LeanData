import Mathlib

namespace sourdough_cost_eq_nine_l613_61358

noncomputable def cost_per_visit (white_bread_cost baguette_cost croissant_cost: ℕ) : ℕ :=
  2 * white_bread_cost + baguette_cost + croissant_cost

noncomputable def total_spent (weekly_cost num_weeks: ℕ) : ℕ :=
  weekly_cost * num_weeks

noncomputable def total_sourdough_spent (total_spent weekly_cost num_weeks: ℕ) : ℕ :=
  total_spent - weekly_cost * num_weeks

noncomputable def total_sourdough_per_week (total_sourdough_spent num_weeks: ℕ) : ℕ :=
  total_sourdough_spent / num_weeks

theorem sourdough_cost_eq_nine (white_bread_cost baguette_cost croissant_cost total_spent_over_4_weeks: ℕ)
  (h₁: white_bread_cost = 350) (h₂: baguette_cost = 150) (h₃: croissant_cost = 200) (h₄: total_spent_over_4_weeks = 7800) :
  total_sourdough_per_week (total_sourdough_spent total_spent_over_4_weeks (cost_per_visit white_bread_cost baguette_cost croissant_cost) 4) 4 = 900 :=
by 
  sorry

end sourdough_cost_eq_nine_l613_61358


namespace reinforcement_size_l613_61363

theorem reinforcement_size (R : ℕ) : 
  2000 * 39 = (2000 + R) * 20 → R = 1900 :=
by
  intro h
  sorry

end reinforcement_size_l613_61363


namespace highest_power_of_3_dividing_N_is_1_l613_61353

-- Define the integer N as described in the problem
def N : ℕ := 313233515253

-- State the problem
theorem highest_power_of_3_dividing_N_is_1 : ∃ k : ℕ, (3^k ∣ N) ∧ ∀ m > 1, ¬ (3^m ∣ N) ∧ k = 1 :=
by
  -- Specific solution details and steps are not required here
  sorry

end highest_power_of_3_dividing_N_is_1_l613_61353


namespace total_women_attendees_l613_61314

theorem total_women_attendees 
  (adults : ℕ) (adult_women : ℕ) (student_offset : ℕ) (total_students : ℕ)
  (male_students : ℕ) :
  adults = 1518 →
  adult_women = 536 →
  student_offset = 525 →
  total_students = adults + student_offset →
  total_students = 2043 →
  male_students = 1257 →
  (adult_women + (total_students - male_students) = 1322) :=
by
  sorry

end total_women_attendees_l613_61314


namespace fifty_third_card_is_A_l613_61381

noncomputable def card_seq : List String := 
  ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

theorem fifty_third_card_is_A : card_seq[(53 % 13)] = "A" := 
by 
  simp [card_seq] 
  sorry

end fifty_third_card_is_A_l613_61381


namespace factorization_count_l613_61301

noncomputable def count_factors (n : ℕ) (a b c : ℕ) : ℕ :=
if 2 ^ a * 2 ^ b * 2 ^ c = n ∧ a + b + c = 10 ∧ a ≥ b ∧ b ≥ c then 1 else 0

noncomputable def total_factorizations : ℕ :=
Finset.sum (Finset.range 11) (fun c => 
  Finset.sum (Finset.Icc c 10) (fun b => 
    Finset.sum (Finset.Icc b 10) (fun a =>
      count_factors 1024 a b c)))

theorem factorization_count : total_factorizations = 14 :=
sorry

end factorization_count_l613_61301


namespace add_pure_acid_to_obtain_final_concentration_l613_61361

   variable (x : ℝ)

   def initial_solution_volume : ℝ := 60
   def initial_acid_concentration : ℝ := 0.10
   def final_acid_concentration : ℝ := 0.15

   axiom calculate_pure_acid (x : ℝ) :
     initial_acid_concentration * initial_solution_volume + x = final_acid_concentration * (initial_solution_volume + x)

   noncomputable def pure_acid_solution : ℝ := 3/0.85

   theorem add_pure_acid_to_obtain_final_concentration :
     x = pure_acid_solution := by
     sorry
   
end add_pure_acid_to_obtain_final_concentration_l613_61361


namespace jean_total_calories_l613_61378

-- Define the conditions
def pages_per_donut : ℕ := 2
def written_pages : ℕ := 12
def calories_per_donut : ℕ := 150

-- Define the question as a theorem
theorem jean_total_calories : (written_pages / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_total_calories_l613_61378


namespace quadratic_function_properties_l613_61346

theorem quadratic_function_properties
    (f : ℝ → ℝ)
    (h_vertex : ∀ x, f x = -(x - 2)^2 + 1)
    (h_point : f (-1) = -8) :
  (∀ x, f x = -(x - 2)^2 + 1) ∧
  (f 1 = 0) ∧ (f 3 = 0) ∧ (f 0 = 1) :=
  by
    sorry

end quadratic_function_properties_l613_61346


namespace problem_solution_l613_61380

theorem problem_solution
  (a b : ℝ)
  (h1 : a * b = 2)
  (h2 : a - b = 3) :
  a^3 * b - 2 * a^2 * b^2 + a * b^3 = 18 :=
by
  sorry

end problem_solution_l613_61380


namespace country_x_income_l613_61362

variable (income : ℝ)
variable (tax_paid : ℝ)
variable (income_first_40000_tax : ℝ := 40000 * 0.1)
variable (income_above_40000_tax_rate : ℝ := 0.2)
variable (total_tax_paid : ℝ := 8000)
variable (income_above_40000 : ℝ := (total_tax_paid - income_first_40000_tax) / income_above_40000_tax_rate)

theorem country_x_income : 
  income = 40000 + income_above_40000 → 
  total_tax_paid = tax_paid → 
  tax_paid = income_first_40000_tax + (income_above_40000 * income_above_40000_tax_rate) →
  income = 60000 :=
by sorry

end country_x_income_l613_61362


namespace probability_first_hearts_second_ace_correct_l613_61330

noncomputable def probability_first_hearts_second_ace : ℚ :=
  let total_cards := 104
  let total_aces := 8 -- 4 aces per deck, 2 decks
  let hearts_count := 2 * 13 -- 13 hearts per deck, 2 decks
  let ace_of_hearts_count := 2

  -- Case 1: the first is an ace of hearts
  let prob_first_ace_of_hearts := (ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_ace_of_hearts := (total_aces - 1 : ℚ) / (total_cards - 1)

  -- Case 2: the first is a hearts but not an ace
  let prob_first_hearts_not_ace := (hearts_count - ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_hearts_not_ace := total_aces / (total_cards - 1)

  -- Combined probability
  (prob_first_ace_of_hearts * prob_second_ace_given_first_ace_of_hearts) +
  (prob_first_hearts_not_ace * prob_second_ace_given_first_hearts_not_ace)

theorem probability_first_hearts_second_ace_correct : 
  probability_first_hearts_second_ace = 7 / 453 := 
sorry

end probability_first_hearts_second_ace_correct_l613_61330


namespace card_game_fairness_l613_61320

theorem card_game_fairness :
  let deck_size := 52
  let aces := 2
  let total_pairings := Nat.choose deck_size aces  -- Number of ways to choose 2 positions from 52
  let tie_cases := deck_size - 1                  -- Number of ways for consecutive pairs
  let non_tie_outcomes := total_pairings - tie_cases
  non_tie_outcomes / 2 = non_tie_outcomes / 2
:= sorry

end card_game_fairness_l613_61320


namespace lines_through_three_distinct_points_l613_61391

theorem lines_through_three_distinct_points : 
  ∃ n : ℕ, n = 54 ∧ (∀ (i j k : ℕ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 → 
  ∃ (a b c : ℤ), -- Direction vector (a, b, c)
  abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
  ((i + a > 0 ∧ i + a ≤ 3) ∧ (j + b > 0 ∧ j + b ≤ 3) ∧ (k + c > 0 ∧ k + c ≤ 3) ∧
  (i + 2 * a > 0 ∧ i + 2 * a ≤ 3) ∧ (j + 2 * b > 0 ∧ j + 2 * b ≤ 3) ∧ (k + 2 * c > 0 ∧ k + 2 * c ≤ 3))) := 
sorry

end lines_through_three_distinct_points_l613_61391


namespace divisibility_by_7_l613_61319

theorem divisibility_by_7 (n : ℕ) : (3^(2 * n + 1) + 2^(n + 2)) % 7 = 0 :=
by
  sorry

end divisibility_by_7_l613_61319


namespace volleyball_match_probabilities_l613_61309

noncomputable def probability_of_team_A_winning : ℚ := (2 / 3) ^ 3
noncomputable def probability_of_team_B_winning_3_0 : ℚ := 1 / 3
noncomputable def probability_of_team_B_winning_3_1 : ℚ := (2 / 3) * (1 / 3)
noncomputable def probability_of_team_B_winning_3_2 : ℚ := (2 / 3) ^ 2 * (1 / 3)

theorem volleyball_match_probabilities :
  probability_of_team_A_winning = 8 / 27 ∧
  probability_of_team_B_winning_3_0 = 1 / 3 ∧
  probability_of_team_B_winning_3_1 ≠ 1 / 9 ∧
  probability_of_team_B_winning_3_2 ≠ 4 / 9 :=
by
  sorry

end volleyball_match_probabilities_l613_61309


namespace factor_expression_l613_61367

theorem factor_expression (x : ℝ) : 
  3 * x^2 * (x - 5) + 4 * x * (x - 5) + 6 * (x - 5) = (3 * x^2 + 4 * x + 6) * (x - 5) :=
  sorry

end factor_expression_l613_61367


namespace square_area_example_l613_61343

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def square_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (distance x1 y1 x2 y2)^2

theorem square_area_example : square_area 1 3 5 6 = 25 :=
by
  sorry

end square_area_example_l613_61343


namespace factorial_fraction_simplification_l613_61333

-- Define necessary factorial function
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Define the problem
theorem factorial_fraction_simplification :
  (4 * fact 6 + 20 * fact 5) / fact 7 = 22 / 21 := by
  sorry

end factorial_fraction_simplification_l613_61333


namespace hamburger_varieties_l613_61373

-- Define the problem conditions as Lean definitions.
def condiments := 9  -- There are 9 condiments
def patty_choices := 3  -- Choices of 1, 2, or 3 patties

-- The goal is to prove that the number of different kinds of hamburgers is 1536.
theorem hamburger_varieties : (3 * 2^9) = 1536 := by
  sorry

end hamburger_varieties_l613_61373


namespace find_smallest_x_l613_61357

theorem find_smallest_x :
  ∃ x : ℕ, x > 0 ∧
  (45 * x + 9) % 25 = 3 ∧
  (2 * x) % 5 = 8 ∧
  x = 20 :=
by
  sorry

end find_smallest_x_l613_61357


namespace sequence_sum_l613_61371

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ∀ n : ℕ, S n + a n = 2 * n + 1) :
  ∀ n : ℕ, a n = 2 - (1 / 2^n) :=
by
  sorry

end sequence_sum_l613_61371


namespace sqrt_17_irrational_l613_61307

theorem sqrt_17_irrational : ¬ ∃ (q : ℚ), q * q = 17 := sorry

end sqrt_17_irrational_l613_61307


namespace simplify_power_of_power_l613_61325

theorem simplify_power_of_power (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end simplify_power_of_power_l613_61325


namespace parallelogram_proof_l613_61374

noncomputable def sin_angle_degrees (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

theorem parallelogram_proof (x : ℝ) (A : ℝ) (r : ℝ) (side1 side2 : ℝ) (P : ℝ):
  (A = 972) → (r = 4 / 3) → (sin_angle_degrees 45 = Real.sqrt 2 / 2) →
  (side1 = 4 * x) → (side2 = 3 * x) →
  (A = side1 * (side2 * (Real.sqrt 2 / 2 / 3))) →
  x = 9 * 2^(3/4) →
  side1 = 36 * 2^(3/4) →
  side2 = 27 * 2^(3/4) →
  (P = 2 * (side1 + side2)) →
  (P = 126 * 2^(3/4)) :=
by
  intros
  sorry

end parallelogram_proof_l613_61374


namespace volume_of_given_cuboid_l613_61315

-- Definition of the function to compute the volume of a cuboid
def volume_of_cuboid (length width height : ℝ) : ℝ :=
  length * width * height

-- Given conditions and the proof target
theorem volume_of_given_cuboid : volume_of_cuboid 2 5 3 = 30 :=
by
  sorry

end volume_of_given_cuboid_l613_61315


namespace area_square_EFGH_l613_61336

theorem area_square_EFGH (AB BE : ℝ) (h : BE = 2) (h2 : AB = 10) :
  ∃ s : ℝ, (s = 8 * Real.sqrt 6 - 2) ∧ s^2 = (8 * Real.sqrt 6 - 2)^2 := by
  sorry

end area_square_EFGH_l613_61336


namespace product_of_p_r_s_l613_61311

theorem product_of_p_r_s (p r s : ℕ) 
  (h1 : 4^p + 4^3 = 280)
  (h2 : 3^r + 29 = 56) 
  (h3 : 7^s + 6^3 = 728) : 
  p * r * s = 27 :=
by
  sorry

end product_of_p_r_s_l613_61311


namespace inradius_plus_circumradius_le_height_l613_61344

theorem inradius_plus_circumradius_le_height {α β γ : ℝ} 
    (h : ℝ) (r R : ℝ)
    (h_triangle : α ≥ β ∧ β ≥ γ ∧ γ ≥ 0 ∧ α + β + γ = π )
    (h_non_obtuse : π / 2 ≥ α ∧ π / 2 ≥ β ∧ π / 2 ≥ γ)
    (h_greatest_height : true) -- Assuming this condition holds as given
    :
    r + R ≤ h :=
sorry

end inradius_plus_circumradius_le_height_l613_61344


namespace tan_cos_identity_l613_61316

theorem tan_cos_identity :
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  (tanθ^2 - cosθ^2) / (tanθ^2 * cosθ^2) = -5 / 3 :=
by
  let θ := 30 * Real.pi / 180; -- Convert 30 degrees to radians
  let tanθ := Real.tan θ;
  let cosθ := Real.cos θ;
  have h_tan : tanθ^2 = (Real.sin θ)^2 / (Real.cos θ)^2 := by sorry; -- Given condition 1
  have h_cos : cosθ^2 = 3 / 4 := by sorry; -- Given condition 2
  -- Prove the statement
  sorry

end tan_cos_identity_l613_61316


namespace find_pairs_l613_61370

theorem find_pairs (x y : Nat) (h : 1 + x + x^2 + x^3 + x^4 = y^2) : (x, y) = (0, 1) ∨ (x, y) = (3, 11) := by
  sorry

end find_pairs_l613_61370


namespace geomSeriesSum_eq_683_l613_61349

/-- Define the first term, common ratio, and number of terms -/
def firstTerm : ℤ := -1
def commonRatio : ℤ := -2
def numTerms : ℕ := 11

/-- Function to calculate the sum of the geometric series -/
def geomSeriesSum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r^n - 1) / (r - 1))

/-- The main theorem stating that the sum of the series equals 683 -/
theorem geomSeriesSum_eq_683 :
  geomSeriesSum firstTerm commonRatio numTerms = 683 :=
by sorry

end geomSeriesSum_eq_683_l613_61349


namespace skittles_total_correct_l613_61308

def number_of_students : ℕ := 9
def skittles_per_student : ℕ := 3
def total_skittles : ℕ := 27

theorem skittles_total_correct : number_of_students * skittles_per_student = total_skittles := by
  sorry

end skittles_total_correct_l613_61308


namespace cube_volume_l613_61365

theorem cube_volume (a : ℕ) (h : a^3 - ((a - 2) * a * (a + 2)) = 16) : a^3 = 64 := by
  sorry

end cube_volume_l613_61365


namespace calories_left_for_dinner_l613_61398

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l613_61398


namespace which_calc_is_positive_l613_61337

theorem which_calc_is_positive :
  (-3 + 7 - 5 < 0) ∧
  ((1 - 2) * 3 < 0) ∧
  (-16 / (↑(-3)^2) < 0) ∧
  (-2^4 * (-6) > 0) :=
by
sorry

end which_calc_is_positive_l613_61337


namespace rashmi_speed_second_day_l613_61394

noncomputable def rashmi_speed (distance speed1 time_late time_early : ℝ) : ℝ :=
  let time1 := distance / speed1
  let on_time := time1 - time_late / 60
  let time2 := on_time - time_early / 60
  distance / time2

theorem rashmi_speed_second_day :
  rashmi_speed 9.999999999999993 5 10 10 = 6 := by
  sorry

end rashmi_speed_second_day_l613_61394


namespace initial_volume_of_solution_l613_61368

theorem initial_volume_of_solution (V : ℝ) (h0 : 0.10 * V = 0.08 * (V + 20)) : V = 80 :=
by
  sorry

end initial_volume_of_solution_l613_61368


namespace total_money_l613_61321

theorem total_money (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 330) (h3 : C = 30) : 
  A + B + C = 500 :=
by
  sorry

end total_money_l613_61321


namespace janet_pills_monthly_l613_61382

def daily_intake_first_two_weeks := 2 + 3 -- 2 multivitamins + 3 calcium supplements
def daily_intake_last_two_weeks := 2 + 1 -- 2 multivitamins + 1 calcium supplement
def days_in_two_weeks := 2 * 7

theorem janet_pills_monthly :
  (daily_intake_first_two_weeks * days_in_two_weeks) + (daily_intake_last_two_weeks * days_in_two_weeks) = 112 :=
by
  sorry

end janet_pills_monthly_l613_61382


namespace largest_angle_in_right_isosceles_triangle_l613_61331

theorem largest_angle_in_right_isosceles_triangle (X Y Z : Type) 
  (angle_X : ℝ) (angle_Y : ℝ) (angle_Z : ℝ) 
  (h1 : angle_X = 45) 
  (h2 : angle_Y = 90)
  (h3 : angle_Y + angle_X + angle_Z = 180) 
  (h4 : angle_X = angle_Z) : angle_Y = 90 := by 
  sorry

end largest_angle_in_right_isosceles_triangle_l613_61331


namespace tenth_term_arithmetic_sequence_l613_61389

theorem tenth_term_arithmetic_sequence :
  ∀ (a : ℕ → ℚ), a 1 = 5/6 ∧ a 16 = 7/8 →
  a 10 = 103/120 :=
by
  sorry

end tenth_term_arithmetic_sequence_l613_61389


namespace total_viewing_time_amaya_l613_61302

/-- The total viewing time Amaya spent, including rewinding, was 170 minutes. -/
theorem total_viewing_time_amaya 
  (u1 u2 u3 u4 u5 r1 r2 r3 r4 : ℕ)
  (h1 : u1 = 35)
  (h2 : u2 = 45)
  (h3 : u3 = 25)
  (h4 : u4 = 15)
  (h5 : u5 = 20)
  (hr1 : r1 = 5)
  (hr2 : r2 = 7)
  (hr3 : r3 = 10)
  (hr4 : r4 = 8) :
  u1 + u2 + u3 + u4 + u5 + r1 + r2 + r3 + r4 = 170 :=
by
  sorry

end total_viewing_time_amaya_l613_61302


namespace always_true_statements_l613_61306

variable (a b c : ℝ)

theorem always_true_statements (h1 : a < 0) (h2 : a < b ∧ b ≤ 0) (h3 : b < c) : 
  (a + b < b + c) ∧ (c / a < 1) :=
by 
  sorry

end always_true_statements_l613_61306


namespace kindergarten_children_l613_61345

theorem kindergarten_children (x y z n : ℕ) 
  (h1 : 2 * x + 3 * y + 4 * z = n)
  (h2 : x + y + z = 26)
  : n = 24 := 
sorry

end kindergarten_children_l613_61345


namespace min_value_expression_l613_61356

theorem min_value_expression : 
  ∃ (x y : ℝ), x^2 + 2 * x * y + 2 * y^2 + 3 * x - 5 * y = -8.5 := by
  sorry

end min_value_expression_l613_61356


namespace pilot_fish_speed_when_moved_away_l613_61366

/-- Conditions -/
def keanu_speed : ℕ := 20
def shark_new_speed (k : ℕ) : ℕ := 2 * k
def pilot_fish_increase_speed (k s_new : ℕ) : ℕ := k + (s_new - k) / 2

/-- The problem statement to prove -/
theorem pilot_fish_speed_when_moved_away (k : ℕ) (s_new : ℕ) (p_new : ℕ) 
  (h1 : k = 20) 
  (h2 : s_new = shark_new_speed k) 
  (h3 : p_new = pilot_fish_increase_speed k s_new) : 
  p_new = 30 :=
by
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3]
  sorry

end pilot_fish_speed_when_moved_away_l613_61366


namespace arithmetic_sequence__geometric_sequence__l613_61342

-- Part 1: Arithmetic Sequence
theorem arithmetic_sequence_
  (d : ℤ) (n : ℤ) (a_n : ℤ) (a_1 : ℤ) (S_n : ℤ)
  (h_d : d = 2) (h_n : n = 15) (h_a_n : a_n = -10)
  (h_a_1 : a_1 = -38) (h_S_n : S_n = -360) :
  a_n = a_1 + (n - 1) * d ∧ S_n = n * (a_1 + a_n) / 2 :=
by
  sorry

-- Part 2: Geometric Sequence
theorem geometric_sequence_
  (a_1 : ℝ) (q : ℝ) (S_10 : ℝ)
  (a_2 : ℝ) (a_3 : ℝ) (a_4 : ℝ)
  (h_a_2_3 : a_2 + a_3 = 6) (h_a_3_4 : a_3 + a_4 = 12)
  (h_a_1 : a_1 = 1) (h_q : q = 2) (h_S_10 : S_10 = 1023) :
  a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3 ∧ S_10 = a_1 * (1 - q^10) / (1 - q) :=
by
  sorry

end arithmetic_sequence__geometric_sequence__l613_61342


namespace triangle_inequality_inequality_l613_61310

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 > 0 := 
by
  sorry

end triangle_inequality_inequality_l613_61310


namespace negation_of_statement_l613_61338

theorem negation_of_statement (x : ℝ) :
  (¬ (x^2 = 1 → x = 1 ∨ x = -1)) ↔ (x^2 = 1 ∧ (x ≠ 1 ∧ x ≠ -1)) :=
sorry

end negation_of_statement_l613_61338


namespace solve_for_x_l613_61388

theorem solve_for_x (x : ℝ) (h1 : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) 
  (h2 : x ≠ -2) (h3 : x ≠ 3) : x = -1 / 6 :=
by
  sorry

end solve_for_x_l613_61388


namespace total_additions_in_2_hours_30_minutes_l613_61303

def additions_rate : ℕ := 15000

def time_in_seconds : ℕ := 2 * 3600 + 30 * 60

def total_additions : ℕ := additions_rate * time_in_seconds

theorem total_additions_in_2_hours_30_minutes :
  total_additions = 135000000 :=
by
  -- Non-trivial proof skipped
  sorry

end total_additions_in_2_hours_30_minutes_l613_61303


namespace remainder_modulus_l613_61387

theorem remainder_modulus :
  (9^4 + 8^5 + 7^6 + 5^3) % 3 = 2 :=
by
  sorry

end remainder_modulus_l613_61387


namespace problem_gets_solved_prob_l613_61317

-- Define conditions for probabilities
def P_A_solves := 2 / 3
def P_B_solves := 3 / 4

-- Calculate the probability that the problem is solved
theorem problem_gets_solved_prob :
  let P_A_not_solves := 1 - P_A_solves
  let P_B_not_solves := 1 - P_B_solves
  let P_both_not_solve := P_A_not_solves * P_B_not_solves
  let P_solved := 1 - P_both_not_solve
  P_solved = 11 / 12 :=
by
  -- Skip proof
  sorry

end problem_gets_solved_prob_l613_61317


namespace geometric_sequence_seventh_term_l613_61375

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end geometric_sequence_seventh_term_l613_61375


namespace rachel_speed_painting_video_time_l613_61335

theorem rachel_speed_painting_video_time :
  let num_videos := 4
  let setup_time := 1
  let cleanup_time := 1
  let painting_time_per_video := 1
  let editing_time_per_video := 1.5
  (setup_time + cleanup_time + painting_time_per_video * num_videos + editing_time_per_video * num_videos) / num_videos = 3 :=
by
  sorry

end rachel_speed_painting_video_time_l613_61335


namespace part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l613_61340

-- Part I
def is_relevant_number (n m : ℕ) : Prop :=
  ∀ {P : Finset ℕ}, (P ⊆ (Finset.range (2*n + 1)) ∧ P.card = m) →
  ∃ (a b c d : ℕ), a ∈ P ∧ b ∈ P ∧ c ∈ P ∧ d ∈ P ∧ a + b + c + d = 4*n + 1

theorem part_I_n_3_not_relevant :
  ¬ is_relevant_number 3 5 := sorry

theorem part_I_n_3_is_relevant :
  is_relevant_number 3 6 := sorry

-- Part II
theorem part_II (n m : ℕ) (h : is_relevant_number n m) : m - n - 3 ≥ 0 := sorry

-- Part III
theorem part_III_min_value_of_relevant_number (n : ℕ) : 
  ∃ m : ℕ, is_relevant_number n m ∧ ∀ k, is_relevant_number n k → m ≤ k := sorry

end part_I_n_3_not_relevant_part_I_n_3_is_relevant_part_II_part_III_min_value_of_relevant_number_l613_61340


namespace purely_imaginary_a_eq_2_l613_61355

theorem purely_imaginary_a_eq_2 (a : ℝ) (h : (2 - a) / 2 = 0) : a = 2 :=
sorry

end purely_imaginary_a_eq_2_l613_61355


namespace percentage_of_40_l613_61305

theorem percentage_of_40 (P : ℝ) (h1 : 8/100 * 24 = 1.92) (h2 : P/100 * 40 + 1.92 = 5.92) : P = 10 :=
sorry

end percentage_of_40_l613_61305


namespace min_value_of_xy_l613_61351

theorem min_value_of_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 4 * x * y - x - 2 * y = 4) : 
  xy >= 2 :=
sorry

end min_value_of_xy_l613_61351


namespace inequality_for_all_real_l613_61379

theorem inequality_for_all_real (a b c : ℝ) : 
  a^6 + b^6 + c^6 - 3 * a^2 * b^2 * c^2 ≥ 1/2 * (a - b)^2 * (b - c)^2 * (c - a)^2 :=
by 
  sorry

end inequality_for_all_real_l613_61379


namespace find_sum_A_B_C_l613_61384

theorem find_sum_A_B_C (A B C : ℤ)
  (h1 : ∀ x > 4, (x^2 : ℝ) / (A * x^2 + B * x + C) > 0.4)
  (h2 : A * (-2)^2 + B * (-2) + C = 0)
  (h3 : A * (3)^2 + B * (3) + C = 0)
  (h4 : 0.4 < 1 / (A : ℝ) ∧ 1 / (A : ℝ) < 1) :
  A + B + C = -12 :=
by
  sorry

end find_sum_A_B_C_l613_61384


namespace geometric_sequence_m_value_l613_61352

theorem geometric_sequence_m_value 
  (a : ℕ → ℝ) (q : ℝ) (m : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a n = a 1 * q^(n-1))
  (h3 : |q| ≠ 1) 
  (h4 : a m = a 1 * a 2 * a 3 * a 4 * a 5) : 
  m = 11 := by
  sorry

end geometric_sequence_m_value_l613_61352


namespace gcd_lcm_1365_910_l613_61354

theorem gcd_lcm_1365_910 :
  gcd 1365 910 = 455 ∧ lcm 1365 910 = 2730 :=
by
  sorry

end gcd_lcm_1365_910_l613_61354


namespace bellas_score_l613_61323

-- Definitions from the problem conditions
def n : Nat := 17
def x : Nat := 75
def new_n : Nat := n + 1
def y : Nat := 76

-- Assertion that Bella's score is 93
theorem bellas_score : (new_n * y) - (n * x) = 93 :=
by
  -- This is where the proof would go
  sorry

end bellas_score_l613_61323


namespace tangent_chord_equation_l613_61350

theorem tangent_chord_equation (x1 y1 x2 y2 : ℝ) :
  (x1^2 + y1^2 = 1) →
  (x2^2 + y2^2 = 1) →
  (2*x1 + 2*y1 + 1 = 0) →
  (2*x2 + 2*y2 + 1 = 0) →
  ∀ (x y : ℝ), 2*x + 2*y + 1 = 0 :=
by
  intros hx1 hy1 hx2 hy2 x y
  exact sorry

end tangent_chord_equation_l613_61350


namespace solve_eq1_solve_eq2_l613_61312

theorem solve_eq1 (x : ℝ) : 4 * x^2 = 12 * x ↔ x = 0 ∨ x = 3 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : x^2 + 4 * x + 3 = 0 ↔ x = -3 ∨ x = -1 :=
by
  sorry

end solve_eq1_solve_eq2_l613_61312


namespace area_percentage_decrease_l613_61359

theorem area_percentage_decrease {a b : ℝ} 
  (h1 : 2 * b = 0.1 * 4 * a) :
  ((b^2) / (a^2) * 100 = 4) :=
by
  sorry

end area_percentage_decrease_l613_61359


namespace oranges_in_second_group_l613_61399

namespace oranges_problem

-- Definitions coming from conditions
def cost_of_apple : ℝ := 0.21
def total_cost_1 : ℝ := 1.77
def total_cost_2 : ℝ := 1.27
def num_apples_group1 : ℕ := 6
def num_oranges_group1 : ℕ := 3
def num_apples_group2 : ℕ := 2
def cost_of_orange : ℝ := 0.17
def num_oranges_group2 : ℕ := 5 -- derived from the solution involving $0.85/$0.17.

-- Price calculation functions and conditions
def price_group1 (cost_of_orange : ℝ) : ℝ :=
  num_apples_group1 * cost_of_apple + num_oranges_group1 * cost_of_orange

def price_group2 (num_oranges_group2 cost_of_orange : ℝ) : ℝ :=
  num_apples_group2 * cost_of_apple + num_oranges_group2 * cost_of_orange

theorem oranges_in_second_group :
  (price_group1 cost_of_orange = total_cost_1) →
  (price_group2 num_oranges_group2 cost_of_orange = total_cost_2) →
  num_oranges_group2 = 5 :=
by
  intros h1 h2
  sorry

end oranges_problem

end oranges_in_second_group_l613_61399


namespace time_to_pass_telegraph_post_l613_61324

def conversion_factor_km_per_hour_to_m_per_sec := 1000 / 3600

noncomputable def train_length := 70
noncomputable def train_speed_kmph := 36

noncomputable def train_speed_m_per_sec := train_speed_kmph * conversion_factor_km_per_hour_to_m_per_sec

theorem time_to_pass_telegraph_post : (train_length / train_speed_m_per_sec) = 7 := by
  sorry

end time_to_pass_telegraph_post_l613_61324


namespace second_smallest_sum_l613_61372

theorem second_smallest_sum (a b c d : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
                           (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
                           (h7 : a + b + c = 180) (h8 : a + c + d = 197)
                           (h9 : b + c + d = 208) (h10 : a + b + d = 222) :
  208 ≠ 180 ∧ 208 ≠ 197 ∧ 208 ≠ 222 := 
sorry

end second_smallest_sum_l613_61372


namespace candles_to_new_five_oz_l613_61385

theorem candles_to_new_five_oz 
  (h_wax_percent: ℝ)
  (h_candles_20oz_count: ℕ) 
  (h_candles_5oz_count: ℕ) 
  (h_candles_1oz_count: ℕ) 
  (h_candles_20oz_wax: ℝ) 
  (h_candles_5oz_wax: ℝ)
  (h_candles_1oz_wax: ℝ):
  h_wax_percent = 0.10 →
  h_candles_20oz_count = 5 →
  h_candles_5oz_count = 5 → 
  h_candles_1oz_count = 25 →
  h_candles_20oz_wax = 20 →
  h_candles_5oz_wax = 5 →
  h_candles_1oz_wax = 1 →
  (h_wax_percent * h_candles_20oz_wax * h_candles_20oz_count + 
   h_wax_percent * h_candles_5oz_wax * h_candles_5oz_count + 
   h_wax_percent * h_candles_1oz_wax * h_candles_1oz_count) / 5 = 3 :=
by
  sorry

end candles_to_new_five_oz_l613_61385


namespace value_corresponds_l613_61322

-- Define the problem
def certain_number (x : ℝ) : Prop :=
  0.30 * x = 120

-- State the theorem to be proved
theorem value_corresponds (x : ℝ) (h : certain_number x) : 0.40 * x = 160 :=
by
  sorry

end value_corresponds_l613_61322


namespace train_travel_time_l613_61390

theorem train_travel_time 
  (speed : ℝ := 120) -- speed in kmph
  (distance : ℝ := 80) -- distance in km
  (minutes_in_hour : ℝ := 60) -- conversion factor
  : (distance / speed) * minutes_in_hour = 40 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end train_travel_time_l613_61390


namespace solution_y_eq_2_l613_61376

theorem solution_y_eq_2 (y : ℝ) (h_pos : y > 0) (h_eq : y^6 = 64) : y = 2 :=
sorry

end solution_y_eq_2_l613_61376


namespace jennifer_fruits_left_l613_61396

-- Definitions based on the conditions
def pears : ℕ := 15
def oranges : ℕ := 30
def apples : ℕ := 2 * pears
def cherries : ℕ := oranges / 2
def grapes : ℕ := 3 * apples
def pineapples : ℕ := pears + oranges + apples + cherries + grapes

-- Definitions for the number of fruits given to the sister
def pears_given : ℕ := 3
def oranges_given : ℕ := 5
def apples_given : ℕ := 5
def cherries_given : ℕ := 7
def grapes_given : ℕ := 3

-- Calculations based on the conditions for what's left after giving fruits
def pears_left : ℕ := pears - pears_given
def oranges_left : ℕ := oranges - oranges_given
def apples_left : ℕ := apples - apples_given
def cherries_left : ℕ := cherries - cherries_given
def grapes_left : ℕ := grapes - grapes_given

def remaining_pineapples : ℕ := pineapples - (pineapples / 2)

-- Total number of fruits left
def total_fruits_left : ℕ := pears_left + oranges_left + apples_left + cherries_left + grapes_left + remaining_pineapples

-- Theorem statement
theorem jennifer_fruits_left : total_fruits_left = 247 :=
by
  -- The detailed proof would go here
  sorry

end jennifer_fruits_left_l613_61396


namespace chocolate_bar_breaks_l613_61304

-- Definition of the problem as per the conditions
def chocolate_bar (rows : ℕ) (cols : ℕ) : ℕ := rows * cols

-- Statement of the proving problem
theorem chocolate_bar_breaks :
  ∀ (rows cols : ℕ), chocolate_bar rows cols = 40 → rows = 5 → cols = 8 → 
  (rows - 1) + (cols * (rows - 1)) = 39 :=
by
  intros rows cols h_bar h_rows h_cols
  sorry

end chocolate_bar_breaks_l613_61304


namespace division_of_pow_of_16_by_8_eq_2_pow_4041_l613_61334

theorem division_of_pow_of_16_by_8_eq_2_pow_4041 :
  (16^1011) / 8 = 2^4041 :=
by
  -- Assume m = 16^1011
  let m := 16^1011
  -- Then expressing m in base 2
  have h_m_base2 : m = 2^4044 := by sorry
  -- Dividing m by 8
  have h_division : m / 8 = 2^4041 := by sorry
  -- Conclusion
  exact h_division

end division_of_pow_of_16_by_8_eq_2_pow_4041_l613_61334


namespace geometric_seq_increasing_condition_l613_61364

theorem geometric_seq_increasing_condition (q : ℝ) (a : ℕ → ℝ): 
  (∀ n : ℕ, a (n + 1) = q * a n) → (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m) ∧ ¬ (¬ (∀ a : ℕ → ℝ, (∀ n : ℕ, a (n + 1) = q * a n) → ∀ n m : ℕ, n < m → a n < a m))) :=
sorry

end geometric_seq_increasing_condition_l613_61364


namespace avg_licks_l613_61332

theorem avg_licks (Dan Michael Sam David Lance : ℕ) 
  (hDan : Dan = 58) 
  (hMichael : Michael = 63) 
  (hSam : Sam = 70) 
  (hDavid : David = 70) 
  (hLance : Lance = 39) : 
  (Dan + Michael + Sam + David + Lance) / 5 = 60 :=
by 
  sorry

end avg_licks_l613_61332


namespace perfect_square_condition_l613_61329

noncomputable def isPerfectSquareQuadratic (m : ℤ) (x y : ℤ) :=
  ∃ (k : ℤ), (4 * x^2 + m * x * y + 25 * y^2) = k^2

theorem perfect_square_condition (m : ℤ) :
  (∀ x y : ℤ, isPerfectSquareQuadratic m x y) → (m = 20 ∨ m = -20) :=
by
  sorry

end perfect_square_condition_l613_61329


namespace greatest_value_x_plus_y_l613_61383

theorem greatest_value_x_plus_y (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end greatest_value_x_plus_y_l613_61383


namespace annual_income_correct_l613_61369

def investment (amount : ℕ) := 6800
def dividend_rate (rate : ℕ) := 20
def stock_price (price : ℕ) := 136
def face_value : ℕ := 100
def calculate_annual_income (amount rate price value : ℕ) : ℕ := 
  let shares := amount / price
  let annual_income_per_share := value * rate / 100
  shares * annual_income_per_share

theorem annual_income_correct : calculate_annual_income (investment 6800) (dividend_rate 20) (stock_price 136) face_value = 1000 :=
by
  sorry

end annual_income_correct_l613_61369


namespace mara_correct_answers_l613_61392

theorem mara_correct_answers :
  let math_total    := 30
  let science_total := 20
  let history_total := 50
  let math_percent  := 0.85
  let science_percent := 0.75
  let history_percent := 0.65
  let math_correct  := math_percent * math_total
  let science_correct := science_percent * science_total
  let history_correct := history_percent * history_total
  let total_correct := math_correct + science_correct + history_correct
  let total_problems := math_total + science_total + history_total
  let overall_percent := total_correct / total_problems
  overall_percent = 0.73 :=
by
  sorry

end mara_correct_answers_l613_61392


namespace ratio_of_additional_hours_james_danced_l613_61341

-- Definitions based on given conditions
def john_first_dance_time : ℕ := 3
def john_break_time : ℕ := 1
def john_second_dance_time : ℕ := 5
def combined_dancing_time_excluding_break : ℕ := 20

-- Calculations to be proved
def john_total_resting_dancing_time : ℕ :=
  john_first_dance_time + john_break_time + john_second_dance_time

def john_total_dancing_time : ℕ :=
  john_first_dance_time + john_second_dance_time

def james_dancing_time : ℕ :=
  combined_dancing_time_excluding_break - john_total_dancing_time

def additional_hours_james_danced : ℕ :=
  james_dancing_time - john_total_dancing_time

def desired_ratio : ℕ × ℕ :=
  (additional_hours_james_danced, john_total_resting_dancing_time)

-- Theorem to be proved according to the problem statement
theorem ratio_of_additional_hours_james_danced :
  desired_ratio = (4, 9) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_additional_hours_james_danced_l613_61341


namespace find_numbers_l613_61386

theorem find_numbers :
  ∃ (a b c d : ℕ), 
  (a + 2 = 22) ∧ 
  (b - 2 = 22) ∧ 
  (c * 2 = 22) ∧ 
  (d / 2 = 22) ∧ 
  (a + b + c + d = 99) :=
sorry

end find_numbers_l613_61386


namespace quadratic_increasing_l613_61360

noncomputable def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_increasing (a b c : ℝ) 
  (h1 : quadratic a b c 0 = quadratic a b c 6)
  (h2 : quadratic a b c 0 < quadratic a b c 7) :
  ∀ x, x > 3 → ∀ y, y > 3 → x < y → quadratic a b c x < quadratic a b c y :=
sorry

end quadratic_increasing_l613_61360


namespace AC_total_l613_61348

theorem AC_total (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 450) (h3 : C = 100) : A + C = 250 := by
  sorry

end AC_total_l613_61348


namespace inequality_solution_l613_61327

variable (a x : ℝ)

noncomputable def inequality_solutions :=
  if a = 0 then
    {x | x > 1}
  else if a > 1 then
    {x | (1 / a) < x ∧ x < 1}
  else if a = 1 then
    ∅
  else if 0 < a ∧ a < 1 then
    {x | 1 < x ∧ x < (1 / a)}
  else if a < 0 then
    {x | x < (1 / a) ∨ x > 1}
  else
    ∅

theorem inequality_solution (h : a ≠ 0) :
  if a = 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 → x > 1
  else if a > 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ ((1 / a) < x ∧ x < 1)
  else if a = 1 then
    ∀ x, ¬((a * x - 1) * (x - 1) < 0)
  else if 0 < a ∧ a < 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (1 < x ∧ x < (1 / a))
  else if a < 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (x < (1 / a) ∨ x > 1)
  else
    True := sorry

end inequality_solution_l613_61327


namespace min_value_frac_l613_61377

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  ∃ (x : ℝ), x = 16 ∧ (forall y, y = 9 / a + 1 / b → x ≤ y) :=
sorry

end min_value_frac_l613_61377


namespace find_alpha_l613_61300

noncomputable section

open Real 

def curve_C1 (x y : ℝ) : Prop := x + y = 1
def curve_C2 (x y φ : ℝ) : Prop := x = 2 + 2 * cos φ ∧ y = 2 * sin φ 

def polar_coordinate_eq1 (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = sqrt 2 / 2
def polar_coordinate_eq2 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

def line_l (ρ θ α : ℝ)  (hα: α > 0 ∧ α < π / 2) : Prop := θ = α ∧ ρ > 0 

def OB_div_OA_eq_4 (ρA ρB α : ℝ) : Prop := ρB / ρA = 4

theorem find_alpha (α : ℝ) (hα: α > 0 ∧ α < π / 2)
  (h₁: ∀ (x y ρ θ: ℝ), curve_C1 x y → polar_coordinate_eq1 ρ θ) 
  (h₂: ∀ (x y φ ρ θ: ℝ), curve_C2 x y φ → polar_coordinate_eq2 ρ θ) 
  (h₃: ∀ (ρ θ: ℝ), line_l ρ θ α hα) 
  (h₄: ∀ (ρA ρB : ℝ), OB_div_OA_eq_4 ρA ρB α → ρA = 1 / (cos α + sin α) ∧ ρB = 4 * cos α ): 
  α = 3 * π / 8 :=
by
  sorry

end find_alpha_l613_61300


namespace gcd_168_486_l613_61318

theorem gcd_168_486 : gcd 168 486 = 6 := 
by sorry

end gcd_168_486_l613_61318


namespace cos_identity_l613_61393

theorem cos_identity (α : ℝ) (h : Real.cos (π / 3 - α) = 3 / 5) : 
  Real.cos (2 * π / 3 + α) = -3 / 5 :=
by
  sorry

end cos_identity_l613_61393


namespace number_of_tulips_l613_61339

theorem number_of_tulips (T : ℕ) (roses : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (total_flowers : ℕ) (h1 : roses = 37) (h2 : used_flowers = 70) 
  (h3 : extra_flowers = 3) (h4: total_flowers = 73) 
  (h5 : T + roses = total_flowers) : T = 36 := 
by
  sorry

end number_of_tulips_l613_61339


namespace total_votes_l613_61326

theorem total_votes (Ben_votes Matt_votes total_votes : ℕ)
  (h_ratio : 2 * Matt_votes = 3 * Ben_votes)
  (h_Ben_votes : Ben_votes = 24) :
  total_votes = Ben_votes + Matt_votes :=
sorry

end total_votes_l613_61326


namespace min_number_of_participants_l613_61313

theorem min_number_of_participants :
  ∃ n : ℕ, 
    (∃ x : ℕ, (3 * x + 1 = n) ∧ 
    (∃ y : ℕ, (4 * y + 1 = n) ∧ 
    (∃ z : ℕ, (5 * z + 1 = n)))) ∧
    n = 61 :=
by
  sorry

end min_number_of_participants_l613_61313


namespace shifted_graph_sum_l613_61347

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 2 - 2 * x + 5

def shift_right (f : ℝ → ℝ) (h : ℝ) (x : ℝ) : ℝ := f (x - h)
def shift_up (f : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := f x + k

noncomputable def g (x : ℝ) : ℝ := shift_up (shift_right f 7) 3 x

theorem shifted_graph_sum : (∃ (a b c : ℝ), g x = a * x ^ 2 + b * x + c ∧ (a + b + c = 128)) :=
by
  sorry

end shifted_graph_sum_l613_61347


namespace find_n_l613_61328

theorem find_n (n : ℤ) (h : (n + 1999) / 2 = -1) : n = -2001 := 
sorry

end find_n_l613_61328


namespace value_of_r_when_n_is_3_l613_61395

def r (s : ℕ) : ℕ := 4^s - 2 * s
def s (n : ℕ) : ℕ := 3^n + 2
def n : ℕ := 3

theorem value_of_r_when_n_is_3 : r (s n) = 4^29 - 58 :=
by
  sorry

end value_of_r_when_n_is_3_l613_61395


namespace leadership_selection_ways_l613_61397

theorem leadership_selection_ways (M : ℕ) (chiefs : ℕ) (supporting_chiefs : ℕ) (officers_per_supporting_chief : ℕ) 
  (M_eq : M = 15) (chiefs_eq : chiefs = 1) (supporting_chiefs_eq : supporting_chiefs = 2) 
  (officers_eq : officers_per_supporting_chief = 3) : 
  (M * (M - 1) * (M - 2) * (Nat.choose (M - 3) officers_per_supporting_chief) * (Nat.choose (M - 6) officers_per_supporting_chief)) = 3243240 := by
  simp [M_eq, chiefs_eq, supporting_chiefs_eq, officers_eq]
  norm_num
  sorry

end leadership_selection_ways_l613_61397
