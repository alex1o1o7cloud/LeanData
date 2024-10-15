import Mathlib

namespace NUMINAMATH_GPT_suitable_for_systematic_sampling_l1921_192168

-- Define the given conditions as a structure
structure SamplingProblem where
  option_A : String
  option_B : String
  option_C : String
  option_D : String

-- Define the equivalence theorem to prove Option C is the most suitable
theorem suitable_for_systematic_sampling (p : SamplingProblem) 
(hA: p.option_A = "Randomly selecting 8 students from a class of 48 students to participate in an activity")
(hB: p.option_B = "A city has 210 department stores, including 20 large stores, 40 medium stores, and 150 small stores. To understand the business situation of each store, a sample of 21 stores needs to be drawn")
(hC: p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions")
(hD: p.option_D = "Randomly selecting 10 students from 1200 high school students participating in a mock exam to understand the situation") :
  p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions" := 
sorry

end NUMINAMATH_GPT_suitable_for_systematic_sampling_l1921_192168


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1921_192118

/-- 
In an election, a candidate won by a majority of 1040 votes out of a total of 5200 votes.
Prove that the winning candidate received 60% of the votes.
-/
theorem winning_candidate_percentage {P : ℝ} (h_majority : (P * 5200) - ((1 - P) * 5200) = 1040) : P = 0.60 := 
by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1921_192118


namespace NUMINAMATH_GPT_Tina_profit_l1921_192145

variables (x : ℝ) (profit_per_book : ℝ) (number_of_people : ℕ) (cost_per_book : ℝ)
           (books_per_customer : ℕ) (total_profit : ℝ) (total_cost : ℝ) (total_books_sold : ℕ)

theorem Tina_profit :
  (number_of_people = 4) →
  (cost_per_book = 5) →
  (books_per_customer = 2) →
  (total_profit = 120) →
  (books_per_customer * number_of_people = total_books_sold) →
  (cost_per_book * total_books_sold = total_cost) →
  (total_profit = total_books_sold * x - total_cost) →
  x = 20 :=
by
  intros
  sorry


end NUMINAMATH_GPT_Tina_profit_l1921_192145


namespace NUMINAMATH_GPT_fraction_books_sold_l1921_192174

theorem fraction_books_sold :
  (∃ B F : ℝ, 3.50 * (B - 40) = 280.00000000000006 ∧ B ≠ 0 ∧ F = ((B - 40) / B) ∧ B = 120) → (F = 2 / 3) :=
by
  intro h
  obtain ⟨B, F, h1, h2, e⟩ := h
  sorry

end NUMINAMATH_GPT_fraction_books_sold_l1921_192174


namespace NUMINAMATH_GPT_distance_apart_after_3_hours_l1921_192172

-- Definitions derived from conditions
def Ann_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 6 else if hour = 2 then 8 else 4

def Glenda_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 8 else if hour = 2 then 5 else 9

-- The total distance function for a given skater
def total_distance (speed : ℕ → ℕ) : ℕ :=
  speed 1 + speed 2 + speed 3

-- Ann's total distance skated
def Ann_total_distance : ℕ := total_distance Ann_speed

-- Glenda's total distance skated
def Glenda_total_distance : ℕ := total_distance Glenda_speed

-- The total distance between Ann and Glenda after 3 hours
def total_distance_apart : ℕ := Ann_total_distance + Glenda_total_distance

-- Proof statement (without the proof itself; just the goal declaration)
theorem distance_apart_after_3_hours : total_distance_apart = 40 := by
  sorry

end NUMINAMATH_GPT_distance_apart_after_3_hours_l1921_192172


namespace NUMINAMATH_GPT_donuts_Niraek_covers_l1921_192154

/- Define the radii of the donut holes -/
def radius_Niraek : ℕ := 5
def radius_Theo : ℕ := 9
def radius_Akshaj : ℕ := 10
def radius_Lily : ℕ := 7

/- Define the surface areas of the donut holes -/
def surface_area (r : ℕ) : ℕ := 4 * r * r

/- Compute the surface areas -/
def sa_Niraek := surface_area radius_Niraek
def sa_Theo := surface_area radius_Theo
def sa_Akshaj := surface_area radius_Akshaj
def sa_Lily := surface_area radius_Lily

/- Define a function to compute the LCM of a list of natural numbers -/
def lcm_of_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

/- Compute the lcm of the surface areas -/
def lcm_surface_areas := lcm_of_list [sa_Niraek, sa_Theo, sa_Akshaj, sa_Lily]

/- Compute the answer -/
def num_donuts_Niraek_covers := lcm_surface_areas / sa_Niraek

/- Prove the statement -/
theorem donuts_Niraek_covers : num_donuts_Niraek_covers = 63504 :=
by
  /- Skipping the proof for now -/
  sorry

end NUMINAMATH_GPT_donuts_Niraek_covers_l1921_192154


namespace NUMINAMATH_GPT_chocolate_bars_left_l1921_192179

noncomputable def chocolateBarsCount : ℕ :=
  let initial_bars := 800
  let thomas_friends_bars := (3 * initial_bars) / 8
  let adjusted_thomas_friends_bars := thomas_friends_bars + 1  -- Adjust for the extra bar rounding issue
  let piper_bars_taken := initial_bars / 4
  let piper_bars_returned := 8
  let adjusted_piper_bars := piper_bars_taken - piper_bars_returned
  let paul_club_bars := 9
  let polly_club_bars := 7
  let catherine_bars_returned := 15
  
  initial_bars
  - adjusted_thomas_friends_bars
  - adjusted_piper_bars
  - paul_club_bars
  - polly_club_bars
  + catherine_bars_returned

theorem chocolate_bars_left : chocolateBarsCount = 308 := by
  sorry

end NUMINAMATH_GPT_chocolate_bars_left_l1921_192179


namespace NUMINAMATH_GPT_complex_evaluation_l1921_192151

theorem complex_evaluation (a b : ℂ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a^2 + a * b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := 
by 
  sorry

end NUMINAMATH_GPT_complex_evaluation_l1921_192151


namespace NUMINAMATH_GPT_total_tiles_in_room_l1921_192103

theorem total_tiles_in_room (s : ℕ) (hs : 6 * s - 5 = 193) : s^2 = 1089 :=
by sorry

end NUMINAMATH_GPT_total_tiles_in_room_l1921_192103


namespace NUMINAMATH_GPT_ellipse_hyperbola_proof_l1921_192137

noncomputable def ellipse_and_hyperbola_condition (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (a^2 - b^2 = 5) ∧ (a^2 = 11 * b^2)

theorem ellipse_hyperbola_proof : 
  ∀ (a b : ℝ), ellipse_and_hyperbola_condition a b → b^2 = 0.5 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_proof_l1921_192137


namespace NUMINAMATH_GPT_problem_1_solution_problem_2_solution_l1921_192178

variables (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_ball : ℕ)

def probability_of_red_or_black_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (red_balls + black_balls : ℚ) / total_balls

def probability_of_at_least_one_red_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (((red_balls * (total_balls - red_balls)) + ((red_balls * (red_balls - 1)) / 2)) : ℚ)
  / ((total_balls * (total_balls - 1) / 2) : ℚ)

theorem problem_1_solution :
  probability_of_red_or_black_ball 12 5 4 2 1 = 3 / 4 :=
by
  sorry

theorem problem_2_solution :
  probability_of_at_least_one_red_ball 12 5 4 2 1 = 15 / 22 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_solution_problem_2_solution_l1921_192178


namespace NUMINAMATH_GPT_cash_price_of_television_l1921_192157

variable (DownPayment : ℕ := 120)
variable (MonthlyPayment : ℕ := 30)
variable (NumberOfMonths : ℕ := 12)
variable (Savings : ℕ := 80)

-- Define the total installment cost
def TotalInstallment := DownPayment + MonthlyPayment * NumberOfMonths

-- The main statement to prove
theorem cash_price_of_television : (TotalInstallment - Savings) = 400 := by
  sorry

end NUMINAMATH_GPT_cash_price_of_television_l1921_192157


namespace NUMINAMATH_GPT_possible_values_for_a_l1921_192126

def setM : Set ℝ := {x | x^2 + x - 6 = 0}
def setN (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_for_a (a : ℝ) : (∀ x, x ∈ setN a → x ∈ setM) ↔ (a = -1 ∨ a = 0 ∨ a = 2 / 3) := 
by
  sorry

end NUMINAMATH_GPT_possible_values_for_a_l1921_192126


namespace NUMINAMATH_GPT_find_k_l1921_192134

theorem find_k (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := 
sorry

end NUMINAMATH_GPT_find_k_l1921_192134


namespace NUMINAMATH_GPT_max_5_cent_coins_l1921_192163

theorem max_5_cent_coins :
  ∃ (x y z : ℕ), 
  x + y + z = 25 ∧ 
  x + 2*y + 5*z = 60 ∧
  (∀ y' z' : ℕ, y' + 4*z' = 35 → z' ≤ 8) ∧
  y + 4*z = 35 ∧ z = 8 := 
sorry

end NUMINAMATH_GPT_max_5_cent_coins_l1921_192163


namespace NUMINAMATH_GPT_more_boys_than_girls_l1921_192124

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_more_boys_than_girls_l1921_192124


namespace NUMINAMATH_GPT_find_multiplier_l1921_192125

theorem find_multiplier (x : ℝ) (y : ℝ) (h1 : x = 62.5) (h2 : (y * (x + 5)) / 5 - 5 = 22) : y = 2 :=
sorry

end NUMINAMATH_GPT_find_multiplier_l1921_192125


namespace NUMINAMATH_GPT_length_more_than_breadth_by_10_l1921_192166

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end NUMINAMATH_GPT_length_more_than_breadth_by_10_l1921_192166


namespace NUMINAMATH_GPT_total_students_is_100_l1921_192156

-- Definitions of the conditions
def largest_class_students : Nat := 24
def decrement : Nat := 2

-- Let n be the number of classes, which is given by 5
def num_classes : Nat := 5

-- The number of students in each class
def students_in_class (n : Nat) : Nat := 
  if n = 1 then largest_class_students
  else largest_class_students - decrement * (n - 1)

-- Total number of students in the school
def total_students : Nat :=
  List.sum (List.map students_in_class (List.range num_classes))

-- Theorem to prove that total_students equals 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end NUMINAMATH_GPT_total_students_is_100_l1921_192156


namespace NUMINAMATH_GPT_c_minus_3_eq_neg3_l1921_192117

variable (g : ℝ → ℝ)
variable (c : ℝ)

-- defining conditions
axiom invertible_g : Function.Injective g
axiom g_c_eq_3 : g c = 3
axiom g_3_eq_5 : g 3 = 5

-- The goal is to prove that c - 3 = -3
theorem c_minus_3_eq_neg3 : c - 3 = -3 :=
by
  sorry

end NUMINAMATH_GPT_c_minus_3_eq_neg3_l1921_192117


namespace NUMINAMATH_GPT_work_completion_days_l1921_192120

noncomputable def A_days : ℝ := 20
noncomputable def B_days : ℝ := 35
noncomputable def C_days : ℝ := 50

noncomputable def A_work_rate : ℝ := 1 / A_days
noncomputable def B_work_rate : ℝ := 1 / B_days
noncomputable def C_work_rate : ℝ := 1 / C_days

noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate + C_work_rate
noncomputable def total_days : ℝ := 1 / combined_work_rate

theorem work_completion_days : total_days = 700 / 69 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_work_completion_days_l1921_192120


namespace NUMINAMATH_GPT_collinear_points_x_value_l1921_192132

theorem collinear_points_x_value :
  (∀ A B C : ℝ × ℝ, A = (-1, 1) → B = (2, -4) → C = (x, -9) → 
                    (∃ x : ℝ, x = 5)) :=
by sorry

end NUMINAMATH_GPT_collinear_points_x_value_l1921_192132


namespace NUMINAMATH_GPT_find_sum_due_l1921_192127

variable (BD TD FV : ℝ)

-- given conditions
def condition_1 : Prop := BD = 80
def condition_2 : Prop := TD = 70
def condition_3 : Prop := BD = TD + (TD * BD / FV)

-- goal statement
theorem find_sum_due (h1 : condition_1 BD) (h2 : condition_2 TD) (h3 : condition_3 BD TD FV) : FV = 560 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_due_l1921_192127


namespace NUMINAMATH_GPT_units_digit_of_n_l1921_192110

theorem units_digit_of_n
  (m n : ℕ)
  (h1 : m * n = 23^7)
  (h2 : m % 10 = 9) : n % 10 = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_of_n_l1921_192110


namespace NUMINAMATH_GPT_economical_shower_heads_l1921_192142

theorem economical_shower_heads (x T : ℕ) (x_pos : 0 < x)
    (students : ℕ := 100)
    (preheat_time_per_shower : ℕ := 3)
    (shower_time_per_group : ℕ := 12) :
  (T = preheat_time_per_shower * x + shower_time_per_group * (students / x)) →
  (students * preheat_time_per_shower + shower_time_per_group * students / x = T) →
  x = 20 := by
  sorry

end NUMINAMATH_GPT_economical_shower_heads_l1921_192142


namespace NUMINAMATH_GPT_fraction_value_l1921_192148

theorem fraction_value
  (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l1921_192148


namespace NUMINAMATH_GPT_correct_regression_line_l1921_192115

theorem correct_regression_line (h_neg_corr: ∀ x: ℝ, ∀ y: ℝ, y = -10*x + 200 ∨ y = 10*x + 200 ∨ y = -10*x - 200 ∨ y = 10*x - 200) 
(h_slope_neg : ∀ a b: ℝ, a < 0) 
(h_y_intercept: ∀ x: ℝ, x = 0 → 200 > 0 → y = 200) : 
∃ y: ℝ, y = -10*x + 200 :=
by
-- the proof will go here
sorry

end NUMINAMATH_GPT_correct_regression_line_l1921_192115


namespace NUMINAMATH_GPT_prob1_prob2_prob3_l1921_192128

-- Define the sequences for rows ①, ②, and ③
def seq1 (n : ℕ) : ℤ := (-2) ^ n
def seq2 (m : ℕ) : ℤ := (-2) ^ (m - 1)
def seq3 (m : ℕ) : ℤ := (-2) ^ (m - 1) - 1

-- Prove the $n^{th}$ number in row ①
theorem prob1 (n : ℕ) : seq1 n = (-2) ^ n :=
by sorry

-- Prove the relationship between $m^{th}$ numbers in row ② and row ③
theorem prob2 (m : ℕ) : seq3 m = seq2 m - 1 :=
by sorry

-- Prove the value of $x + y + z$ where $x$, $y$, and $z$ are the $2019^{th}$ numbers in rows ①, ②, and ③, respectively
theorem prob3 : seq1 2019 + seq2 2019 + seq3 2019 = -1 :=
by sorry

end NUMINAMATH_GPT_prob1_prob2_prob3_l1921_192128


namespace NUMINAMATH_GPT_water_consumption_comparison_l1921_192152

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_water_consumption_comparison_l1921_192152


namespace NUMINAMATH_GPT_number_exceeds_its_3_over_8_part_by_20_l1921_192116

theorem number_exceeds_its_3_over_8_part_by_20 (x : ℝ) (h : x = (3 / 8) * x + 20) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_exceeds_its_3_over_8_part_by_20_l1921_192116


namespace NUMINAMATH_GPT_amount_kept_by_Tim_l1921_192164

-- Define the conditions
def totalAmount : ℝ := 100
def percentageGivenAway : ℝ := 0.2

-- Prove the question == answer
theorem amount_kept_by_Tim : totalAmount - totalAmount * percentageGivenAway = 80 :=
by
  -- Here the proof would take place
  sorry

end NUMINAMATH_GPT_amount_kept_by_Tim_l1921_192164


namespace NUMINAMATH_GPT_grouping_equal_products_l1921_192171

def group1 : List Nat := [12, 42, 95, 143]
def group2 : List Nat := [30, 44, 57, 91]

def product (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem grouping_equal_products :
  product group1 = product group2 := by
  sorry

end NUMINAMATH_GPT_grouping_equal_products_l1921_192171


namespace NUMINAMATH_GPT_sum_of_solutions_of_absolute_value_l1921_192198

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_of_absolute_value_l1921_192198


namespace NUMINAMATH_GPT_Mike_ride_distance_l1921_192107

theorem Mike_ride_distance 
  (M : ℕ)
  (total_cost_Mike : ℝ)
  (total_cost_Annie : ℝ)
  (h1 : total_cost_Mike = 4.50 + 0.30 * M)
  (h2: total_cost_Annie = 15.00)
  (h3: total_cost_Mike = total_cost_Annie) : 
  M = 35 := 
by
  sorry

end NUMINAMATH_GPT_Mike_ride_distance_l1921_192107


namespace NUMINAMATH_GPT_bobs_walking_rate_l1921_192144

theorem bobs_walking_rate (distance_XY : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_distance_when_met : ℕ) 
  (yolanda_extra_hour : ℕ)
  (meet_covered_distance : distance_XY = yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1 + bob_distance_when_met / bob_distance_when_met)) 
  (yolanda_distance_when_met : yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) + bob_distance_when_met = distance_XY) 
  : 
  (bob_distance_when_met / (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) = yolanda_rate) :=
  sorry

end NUMINAMATH_GPT_bobs_walking_rate_l1921_192144


namespace NUMINAMATH_GPT_scientific_notation_of_population_l1921_192111

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_population_l1921_192111


namespace NUMINAMATH_GPT_isosceles_triangle_same_area_l1921_192143

-- Given conditions of the original isosceles triangle
def original_base : ℝ := 10
def original_side : ℝ := 13

-- The problem states that an isosceles triangle has the base 10 cm and side lengths 13 cm, 
-- we need to show there's another isosceles triangle with a different base but the same area.
theorem isosceles_triangle_same_area : 
  ∃ (new_base : ℝ) (new_side : ℝ), 
    new_base ≠ original_base ∧ 
    (∃ (h1 h2: ℝ), 
      h1 = 12 ∧ 
      h2 = 5 ∧
      1/2 * original_base * h1 = 60 ∧ 
      1/2 * new_base * h2 = 60) := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_same_area_l1921_192143


namespace NUMINAMATH_GPT_log_evaluation_l1921_192197

theorem log_evaluation
  (x : ℝ)
  (h : x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3)) :
  Real.log x / Real.log 7 = -(Real.log 5 / Real.log 3) * (Real.log (Real.log 5 / Real.log 3) / Real.log 7) :=
by
  sorry

end NUMINAMATH_GPT_log_evaluation_l1921_192197


namespace NUMINAMATH_GPT_abs_sub_abs_eq_six_l1921_192185

theorem abs_sub_abs_eq_six
  (a b : ℝ)
  (h₁ : |a| = 4)
  (h₂ : |b| = 2)
  (h₃ : a * b < 0) :
  |a - b| = 6 :=
sorry

end NUMINAMATH_GPT_abs_sub_abs_eq_six_l1921_192185


namespace NUMINAMATH_GPT_product_of_good_numbers_is_good_l1921_192191

def is_good (n : ℕ) : Prop :=
  ∃ (a b c x y : ℤ), n = a * x * x + b * x * y + c * y * y ∧ b * b - 4 * a * c = -20

theorem product_of_good_numbers_is_good {n1 n2 : ℕ} (h1 : is_good n1) (h2 : is_good n2) : is_good (n1 * n2) :=
sorry

end NUMINAMATH_GPT_product_of_good_numbers_is_good_l1921_192191


namespace NUMINAMATH_GPT_triangle_converse_inverse_false_l1921_192184

variables {T : Type} (p q : T → Prop)

-- Condition: If a triangle is equilateral, then it is isosceles
axiom h : ∀ t, p t → q t

-- Conclusion: Neither the converse nor the inverse is true
theorem triangle_converse_inverse_false : 
  (∃ t, q t ∧ ¬ p t) ∧ (∃ t, ¬ p t ∧ q t) :=
sorry

end NUMINAMATH_GPT_triangle_converse_inverse_false_l1921_192184


namespace NUMINAMATH_GPT_coordinates_of_point_l1921_192106

theorem coordinates_of_point (x : ℝ) (P : ℝ × ℝ) (h : P = (1 - x, 2 * x + 1)) (y_axis : P.1 = 0) : P = (0, 3) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_l1921_192106


namespace NUMINAMATH_GPT_project_selection_l1921_192109

noncomputable def binomial : ℕ → ℕ → ℕ 
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binomial n k + binomial n (k+1)

theorem project_selection :
  (binomial 5 2 * binomial 3 2) + (binomial 3 1 * binomial 5 1) = 45 := 
sorry

end NUMINAMATH_GPT_project_selection_l1921_192109


namespace NUMINAMATH_GPT_star_7_3_l1921_192186

def star (a b : ℤ) : ℤ := 4 * a + 3 * b - a * b

theorem star_7_3 : star 7 3 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_star_7_3_l1921_192186


namespace NUMINAMATH_GPT_cloth_sold_l1921_192114

theorem cloth_sold (C S M : ℚ) (P : ℚ) (hP : P = 1 / 3) (hG : 10 * S = (1 / 3) * (M * C)) (hS : S = (4 / 3) * C) : M = 40 := by
  sorry

end NUMINAMATH_GPT_cloth_sold_l1921_192114


namespace NUMINAMATH_GPT_general_term_sum_first_n_terms_l1921_192133

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}
variable (d : ℝ) (h1 : d ≠ 0)
variable (a10 : a 10 = 19)
variable (geo_seq : ∀ {x y z}, x * z = y ^ 2 → x = 1 → y = a 2 → z = a 5)
variable (arith_seq : ∀ n, a n = a 1 + (n - 1) * d)

-- General term of the arithmetic sequence
theorem general_term (a_1 : ℝ) (h1 : a 1 = a_1) : a n = 2 * n - 1 :=
sorry

-- Sum of the first n terms of the sequence b_n
theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

end NUMINAMATH_GPT_general_term_sum_first_n_terms_l1921_192133


namespace NUMINAMATH_GPT_slices_leftover_is_9_l1921_192158

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end NUMINAMATH_GPT_slices_leftover_is_9_l1921_192158


namespace NUMINAMATH_GPT_coords_of_P_max_PA_distance_l1921_192131

open Real

noncomputable def A : (ℝ × ℝ) := (0, -5)

def on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, x = P.1 ∧ y = P.2 ∧ (x - 2)^2 + (y + 3)^2 = 2

def max_PA_distance (P : (ℝ × ℝ)) : Prop :=
  dist P A = max (dist (3, -2) A) (dist (1, -4) A)

theorem coords_of_P_max_PA_distance (P : (ℝ × ℝ)) :
  on_circle P →
  max_PA_distance P →
  P = (3, -2) :=
  sorry

end NUMINAMATH_GPT_coords_of_P_max_PA_distance_l1921_192131


namespace NUMINAMATH_GPT_cogs_produced_after_speed_increase_l1921_192159

-- Define the initial conditions of the problem
def initial_cogs := 60
def initial_rate := 15
def increased_rate := 60
def average_output := 24

-- Variables to represent the number of cogs produced after the speed increase and the total time taken for each phase
variable (x : ℕ)

-- Assuming the equations representing the conditions
def initial_time := initial_cogs / initial_rate
def increased_time := x / increased_rate

def total_cogs := initial_cogs + x
def total_time := initial_time + increased_time

-- Define the overall average output equation
def average_eq := average_output * total_time = total_cogs

-- The proposition we want to prove
theorem cogs_produced_after_speed_increase : x = 60 :=
by
  -- Using the equation from the conditions
  have h1 : average_eq := sorry
  sorry

end NUMINAMATH_GPT_cogs_produced_after_speed_increase_l1921_192159


namespace NUMINAMATH_GPT_janets_garden_area_l1921_192123

theorem janets_garden_area :
  ∃ (s l : ℕ), 2 * (s + l) = 24 ∧ (l + 1) = 3 * (s + 1) ∧ 6 * (s + 1 - 1) * 6 * (l + 1 - 1) = 576 := 
by
  sorry

end NUMINAMATH_GPT_janets_garden_area_l1921_192123


namespace NUMINAMATH_GPT_total_revenue_l1921_192192

theorem total_revenue (chips_sold : ℕ) (chips_price : ℝ) (hotdogs_sold : ℕ) (hotdogs_price : ℝ)
(drinks_sold : ℕ) (drinks_price : ℝ) (sodas_sold : ℕ) (lemonades_sold : ℕ) (sodas_ratio : ℕ)
(lemonades_ratio : ℕ) (h1 : chips_sold = 27) (h2 : chips_price = 1.50) (h3 : hotdogs_sold = chips_sold - 8)
(h4 : hotdogs_price = 3.00) (h5 : drinks_sold = hotdogs_sold + 12) (h6 : drinks_price = 2.00)
(h7 : sodas_ratio = 2) (h8 : lemonades_ratio = 3) (h9 : sodas_sold = (sodas_ratio * drinks_sold) / (sodas_ratio + lemonades_ratio))
(h10 : lemonades_sold = drinks_sold - sodas_sold) :
chips_sold * chips_price + hotdogs_sold * hotdogs_price + drinks_sold * drinks_price = 159.50 := 
by
  -- Proof is left as an exercise for the reader
  sorry

end NUMINAMATH_GPT_total_revenue_l1921_192192


namespace NUMINAMATH_GPT_johns_initial_playtime_l1921_192101

theorem johns_initial_playtime :
  ∃ (x : ℝ), (14 * x = 0.40 * (14 * x + 84)) → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_johns_initial_playtime_l1921_192101


namespace NUMINAMATH_GPT_min_xy_of_conditions_l1921_192153

open Real

theorem min_xy_of_conditions
  (x y : ℝ)
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) : 
  xy ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_min_xy_of_conditions_l1921_192153


namespace NUMINAMATH_GPT_bus_passengers_l1921_192183

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end NUMINAMATH_GPT_bus_passengers_l1921_192183


namespace NUMINAMATH_GPT_theta_in_fourth_quadrant_l1921_192187

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan θ < 0) : 
  (π < θ ∧ θ < 2 * π) :=
by
  sorry

end NUMINAMATH_GPT_theta_in_fourth_quadrant_l1921_192187


namespace NUMINAMATH_GPT_equal_cake_distribution_l1921_192195

theorem equal_cake_distribution (total_cakes : ℕ) (total_friends : ℕ) (h_cakes : total_cakes = 150) (h_friends : total_friends = 50) :
  total_cakes / total_friends = 3 := by
  sorry

end NUMINAMATH_GPT_equal_cake_distribution_l1921_192195


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1921_192141

theorem simplify_and_evaluate_expression (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -2) (hx3 : x ≠ 2) :
  ( ( (x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = x - 2 ) ∧ 
  ( (x = 1) → ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = -1 ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1921_192141


namespace NUMINAMATH_GPT_complement_intersection_l1921_192193

-- Define the universal set U and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {4, 7, 8}

-- Compute the complements
def complement_U (s : Set ℕ) : Set ℕ := U \ s
def comp_A : Set ℕ := complement_U A
def comp_B : Set ℕ := complement_U B

-- Define the intersection of the complements
def intersection_complements : Set ℕ := comp_A ∩ comp_B

-- The theorem to prove
theorem complement_intersection :
  intersection_complements = {1, 2, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1921_192193


namespace NUMINAMATH_GPT_cubic_geometric_sequence_conditions_l1921_192113

-- Conditions from the problem
def cubic_eq (a b c x : ℝ) : Prop := x^3 + a * x^2 + b * x + c = 0

-- The statement to be proven
theorem cubic_geometric_sequence_conditions (a b c : ℝ) :
  (∃ x q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ 
    cubic_eq a b c x ∧ cubic_eq a b c (x*q) ∧ cubic_eq a b c (x*q^2)) → 
  (b^3 = a^3 * c ∧ c ≠ 0 ∧ -a^3 < c ∧ c < a^3 / 27 ∧ a < m ∧ m < - a / 3) :=
by 
  sorry

end NUMINAMATH_GPT_cubic_geometric_sequence_conditions_l1921_192113


namespace NUMINAMATH_GPT_max_poly_l1921_192139

noncomputable def poly (a b : ℝ) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_poly (a b : ℝ) (h : a + b = 4) :
  ∃ (a b : ℝ) (h : a + b = 4), poly a b = (7225 / 56) :=
sorry

end NUMINAMATH_GPT_max_poly_l1921_192139


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1921_192135

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a - 1}

-- The main statement to prove
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_A_and_B_l1921_192135


namespace NUMINAMATH_GPT_eq_neg_one_fifth_l1921_192162

theorem eq_neg_one_fifth : 
  ((1 : ℝ) / ((-5) ^ 4) ^ 2 * (-5) ^ 7) = -1 / 5 := by
  sorry

end NUMINAMATH_GPT_eq_neg_one_fifth_l1921_192162


namespace NUMINAMATH_GPT_bottle_caps_given_l1921_192108

variable (initial_caps : ℕ) (final_caps : ℕ) (caps_given_by_rebecca : ℕ)

theorem bottle_caps_given (h1: initial_caps = 7) (h2: final_caps = 9) : caps_given_by_rebecca = 2 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_bottle_caps_given_l1921_192108


namespace NUMINAMATH_GPT_pyramid_volume_of_unit_cube_l1921_192136

noncomputable def volume_of_pyramid : ℝ :=
  let s := (Real.sqrt 2) / 2
  let base_area := (Real.sqrt 3) / 8
  let height := 1
  (1 / 3) * base_area * height

theorem pyramid_volume_of_unit_cube :
  volume_of_pyramid = (Real.sqrt 3) / 24 := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_of_unit_cube_l1921_192136


namespace NUMINAMATH_GPT_proof_strictly_increasing_sequence_l1921_192104

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end NUMINAMATH_GPT_proof_strictly_increasing_sequence_l1921_192104


namespace NUMINAMATH_GPT_find_A_from_eq_l1921_192182

theorem find_A_from_eq (A : ℕ) (h : 10 - A = 6) : A = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_A_from_eq_l1921_192182


namespace NUMINAMATH_GPT_chick_hits_at_least_five_l1921_192146

theorem chick_hits_at_least_five (x y z : ℕ) (h1 : 9 * x + 5 * y + 2 * z = 61) (h2 : x + y + z = 10) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : x ≥ 5 :=
sorry

end NUMINAMATH_GPT_chick_hits_at_least_five_l1921_192146


namespace NUMINAMATH_GPT_share_sheets_equally_l1921_192188

theorem share_sheets_equally (sheets friends : ℕ) (h_sheets : sheets = 15) (h_friends : friends = 3) : sheets / friends = 5 := by
  sorry

end NUMINAMATH_GPT_share_sheets_equally_l1921_192188


namespace NUMINAMATH_GPT_carlson_fraction_jam_l1921_192165

-- Definitions and conditions.
def total_time (T : ℕ) := T > 0
def time_maloish_cookies (t : ℕ) := t > 0
def equal_cookies (c : ℕ) := c > 0
def carlson_rate := 3

-- Let j_k and j_m be the amounts of jam eaten by Carlson and Maloish respectively.
def fraction_jam_carlson (j_k j_m : ℕ) : ℚ := j_k / (j_k + j_m)

-- The problem statement
theorem carlson_fraction_jam (T t c j_k j_m : ℕ)
  (hT : total_time T)
  (ht : time_maloish_cookies t)
  (hc : equal_cookies c)
  (h_carlson_rate : carlson_rate = 3)
  (h_equal_cookies : c > 0)  -- Both ate equal cookies
  (h_jam : j_k + j_m = j_k * 9 / 10 + j_m / 10) :
  fraction_jam_carlson j_k j_m = 9 / 10 :=
by
  sorry

end NUMINAMATH_GPT_carlson_fraction_jam_l1921_192165


namespace NUMINAMATH_GPT_hypotenuse_length_l1921_192121

theorem hypotenuse_length (x y h : ℝ)
  (hx : (1 / 3) * π * y * x^2 = 1620 * π)
  (hy : (1 / 3) * π * x * y^2 = 3240 * π) :
  h = Real.sqrt 507 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1921_192121


namespace NUMINAMATH_GPT_count_multiples_of_14_between_100_and_400_l1921_192180

theorem count_multiples_of_14_between_100_and_400 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (100 ≤ k ∧ k ≤ 400 ∧ 14 ∣ k) ↔ (∃ i : ℕ, k = 14 * i ∧ 8 ≤ i ∧ i ≤ 28)) :=
sorry

end NUMINAMATH_GPT_count_multiples_of_14_between_100_and_400_l1921_192180


namespace NUMINAMATH_GPT_max_n_for_factored_poly_l1921_192189

theorem max_n_for_factored_poly : 
  ∃ (n : ℤ), (∀ (A B : ℤ), 2 * B + A = n → A * B = 50) ∧ 
            (∀ (m : ℤ), (∀ (A B : ℤ), 2 * B + A = m → A * B = 50) → m ≤ 101) ∧ 
            n = 101 :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_factored_poly_l1921_192189


namespace NUMINAMATH_GPT_maximum_median_soda_shop_l1921_192130

noncomputable def soda_shop_median (total_cans : ℕ) (total_customers : ℕ) (min_cans_per_customer : ℕ) : ℝ :=
  if total_cans = 300 ∧ total_customers = 120 ∧ min_cans_per_customer = 1 then 3.5 else sorry

theorem maximum_median_soda_shop : soda_shop_median 300 120 1 = 3.5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_median_soda_shop_l1921_192130


namespace NUMINAMATH_GPT_T_n_formula_l1921_192150

-- Define the given sequence sum S_n
def S (n : ℕ) : ℚ := (n^2 : ℚ) / 2 + (3 * n : ℚ) / 2

-- Define the general term a_n for the sequence {a_n}
def a (n : ℕ) : ℚ := if n = 1 then 2 else n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 2) - a n + 1 / (a (n + 2) * a n)

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℚ := 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3))

-- Prove the equality of T_n with the given expression
theorem T_n_formula (n : ℕ) : T n = 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := sorry

end NUMINAMATH_GPT_T_n_formula_l1921_192150


namespace NUMINAMATH_GPT_money_made_march_to_august_l1921_192169

section
variable (H : ℕ)

-- Given conditions
def hoursMarchToAugust : ℕ := 23
def hoursSeptToFeb : ℕ := 8
def additionalHours : ℕ := 16
def totalCost : ℕ := 600 + 340
def totalHours : ℕ := hoursMarchToAugust + hoursSeptToFeb + additionalHours

-- Total money equation
def totalMoney : ℕ := totalHours * H

-- Theorem to prove the money made from March to August
theorem money_made_march_to_august : totalMoney = totalCost → hoursMarchToAugust * H = 460 :=
by
  intro h
  have hH : H = 20 := by
    sorry
  rw [hH]
  sorry
end

end NUMINAMATH_GPT_money_made_march_to_august_l1921_192169


namespace NUMINAMATH_GPT_triangle_inequality_l1921_192196

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1921_192196


namespace NUMINAMATH_GPT_fraction_shaded_in_cube_l1921_192181

theorem fraction_shaded_in_cube :
  let side_length := 2
  let face_area := side_length * side_length
  let total_surface_area := 6 * face_area
  let shaded_faces := 3
  let shaded_face_area := face_area / 2
  let total_shaded_area := shaded_faces * shaded_face_area
  total_shaded_area / total_surface_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_shaded_in_cube_l1921_192181


namespace NUMINAMATH_GPT_ratio_of_second_to_first_show_l1921_192170

-- Definitions based on conditions
def first_show_length : ℕ := 30
def total_show_time : ℕ := 150
def second_show_length := total_show_time - first_show_length

-- Proof problem in Lean 4 statement
theorem ratio_of_second_to_first_show : 
  (second_show_length / first_show_length) = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_second_to_first_show_l1921_192170


namespace NUMINAMATH_GPT_petya_friends_count_l1921_192105

-- Define the number of classmates
def total_classmates : ℕ := 28

-- Each classmate has a unique number of friends from 0 to 27
def unique_friends (n : ℕ) : Prop :=
  n ≥ 0 ∧ n < total_classmates

-- We state the problem where Petya's number of friends is to be proven as 14
theorem petya_friends_count (friends : ℕ) (h : unique_friends friends) : friends = 14 :=
sorry

end NUMINAMATH_GPT_petya_friends_count_l1921_192105


namespace NUMINAMATH_GPT_feeding_amount_per_horse_per_feeding_l1921_192119

-- Define the conditions as constants
def num_horses : ℕ := 25
def feedings_per_day : ℕ := 2
def half_ton_in_pounds : ℕ := 1000
def bags_needed : ℕ := 60
def days : ℕ := 60

-- Statement of the problem
theorem feeding_amount_per_horse_per_feeding :
  (bags_needed * half_ton_in_pounds / days / feedings_per_day) / num_horses = 20 := by
  -- Assume conditions are satisfied
  sorry

end NUMINAMATH_GPT_feeding_amount_per_horse_per_feeding_l1921_192119


namespace NUMINAMATH_GPT_percentage_runs_by_running_l1921_192176

theorem percentage_runs_by_running
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (eq_total_runs : total_runs = 120)
  (eq_boundaries : boundaries = 3)
  (eq_sixes : sixes = 8)
  (eq_runs_per_boundary : runs_per_boundary = 4)
  (eq_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100) = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_runs_by_running_l1921_192176


namespace NUMINAMATH_GPT_range_of_a_l1921_192194

theorem range_of_a (a : ℝ) (h : a ≥ 0) :
  ∃ a, (2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2) ↔
  (∀ x y : ℝ, 
    ((x - a)^2 + y^2 = 1) ∧ (x^2 + (y - 2)^2 = 25)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1921_192194


namespace NUMINAMATH_GPT_intersection_complement_l1921_192173

open Set

def UniversalSet := ℝ
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def CU_M : Set ℝ := compl M

theorem intersection_complement :
  N ∩ CU_M = {x | 1 < x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_GPT_intersection_complement_l1921_192173


namespace NUMINAMATH_GPT_interest_rate_for_first_part_l1921_192147

def sum_amount : ℝ := 2704
def part2 : ℝ := 1664
def part1 : ℝ := sum_amount - part2
def rate2 : ℝ := 0.05
def years2 : ℝ := 3
def interest2 : ℝ := part2 * rate2 * years2
def years1 : ℝ := 8

theorem interest_rate_for_first_part (r1 : ℝ) :
  part1 * r1 * years1 = interest2 → r1 = 0.03 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_for_first_part_l1921_192147


namespace NUMINAMATH_GPT_rod_volume_proof_l1921_192129

-- Definitions based on given conditions
def original_length : ℝ := 2
def increase_in_surface_area : ℝ := 0.6
def rod_volume : ℝ := 0.3

-- Problem statement
theorem rod_volume_proof
  (len : ℝ)
  (inc_surface_area : ℝ)
  (vol : ℝ)
  (h_len : len = original_length)
  (h_inc_surface_area : inc_surface_area = increase_in_surface_area) :
  vol = rod_volume :=
sorry

end NUMINAMATH_GPT_rod_volume_proof_l1921_192129


namespace NUMINAMATH_GPT_distance_from_apex_to_larger_cross_section_l1921_192177

namespace PyramidProof

variables (As Al : ℝ) (d h : ℝ)

theorem distance_from_apex_to_larger_cross_section 
  (As_eq : As = 256 * Real.sqrt 2) 
  (Al_eq : Al = 576 * Real.sqrt 2) 
  (d_eq : d = 12) :
  h = 36 := 
sorry

end PyramidProof

end NUMINAMATH_GPT_distance_from_apex_to_larger_cross_section_l1921_192177


namespace NUMINAMATH_GPT_point_on_x_axis_coord_l1921_192155

theorem point_on_x_axis_coord (m : ℝ) (h : (m - 1, 2 * m).snd = 0) : (m - 1, 2 * m) = (-1, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_coord_l1921_192155


namespace NUMINAMATH_GPT_triangle_altitude_l1921_192149

theorem triangle_altitude (A b : ℝ) (h : ℝ) 
  (hA : A = 750) 
  (hb : b = 50) 
  (area_formula : A = (1 / 2) * b * h) : 
  h = 30 :=
  sorry

end NUMINAMATH_GPT_triangle_altitude_l1921_192149


namespace NUMINAMATH_GPT_problem1_problem2_l1921_192122

-- Problem (1)
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^3 + b^3 >= a*b^2 + a^2*b := 
sorry

-- Problem (2)
theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1921_192122


namespace NUMINAMATH_GPT_train_pass_tree_in_time_l1921_192112

-- Definitions from the given conditions
def train_length : ℚ := 270  -- length in meters
def train_speed_km_per_hr : ℚ := 108  -- speed in km/hr

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (v : ℚ) : ℚ := v * (5 / 18)

-- Speed of the train in m/s
def train_speed_m_per_s : ℚ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Question translated into a proof problem
theorem train_pass_tree_in_time :
  train_length / train_speed_m_per_s = 9 :=
by
  sorry

end NUMINAMATH_GPT_train_pass_tree_in_time_l1921_192112


namespace NUMINAMATH_GPT_expression_value_l1921_192190

theorem expression_value : (8 * 6) - (4 / 2) = 46 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1921_192190


namespace NUMINAMATH_GPT_symmetric_sum_l1921_192175

theorem symmetric_sum (a b : ℤ) (h1 : a = -4) (h2 : b = -3) : a + b = -7 := by
  sorry

end NUMINAMATH_GPT_symmetric_sum_l1921_192175


namespace NUMINAMATH_GPT_find_mn_expression_l1921_192100

-- Define the conditions
variables (m n : ℤ)
axiom abs_m_eq_3 : |m| = 3
axiom abs_n_eq_2 : |n| = 2
axiom m_lt_n : m < n

-- State the problem
theorem find_mn_expression : m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_mn_expression_l1921_192100


namespace NUMINAMATH_GPT_baseball_card_decrease_l1921_192102

theorem baseball_card_decrease (x : ℝ) (h : (1 - x / 100) * (1 - x / 100) = 0.64) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_baseball_card_decrease_l1921_192102


namespace NUMINAMATH_GPT_ratio_elyse_to_rick_l1921_192167

-- Define the conditions
def Elyse_initial_gum : ℕ := 100
def Shane_leftover_gum : ℕ := 14
def Shane_chewed_gum : ℕ := 11

-- Theorem stating the ratio of pieces Elyse gave to Rick to the total number of pieces Elyse had
theorem ratio_elyse_to_rick :
  let total_gum := Elyse_initial_gum
  let Shane_initial_gum := Shane_leftover_gum + Shane_chewed_gum
  let Rick_initial_gum := 2 * Shane_initial_gum
  let Elyse_given_to_Rick := Rick_initial_gum
  (Elyse_given_to_Rick : ℚ) / total_gum = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_elyse_to_rick_l1921_192167


namespace NUMINAMATH_GPT_record_loss_of_300_l1921_192140

-- Definitions based on conditions
def profit (x : Int) : String := "+" ++ toString x
def loss (x : Int) : String := "-" ++ toString x

-- The theorem to prove that a loss of 300 is recorded as "-300" based on the recording system
theorem record_loss_of_300 : loss 300 = "-300" :=
by
  sorry

end NUMINAMATH_GPT_record_loss_of_300_l1921_192140


namespace NUMINAMATH_GPT_selling_price_to_achieve_profit_l1921_192160

theorem selling_price_to_achieve_profit (num_pencils : ℝ) (cost_per_pencil : ℝ) (desired_profit : ℝ) (selling_price : ℝ) :
  num_pencils = 1800 →
  cost_per_pencil = 0.15 →
  desired_profit = 100 →
  selling_price = 0.21 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_to_achieve_profit_l1921_192160


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1921_192161

open Set

def A := {x : ℝ | 2 + x ≥ 4}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1921_192161


namespace NUMINAMATH_GPT_range_of_a_l1921_192138

theorem range_of_a (a : ℝ) (x y : ℝ) (hxy : x * y > 0) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + a / y) ≥ 9 → a ≥ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1921_192138


namespace NUMINAMATH_GPT_Patricia_money_l1921_192199

theorem Patricia_money 
(P L C : ℝ)
(h1 : L = 5 * P)
(h2 : L = 2 * C)
(h3 : P + L + C = 51) :
P = 6.8 := 
by 
  sorry

end NUMINAMATH_GPT_Patricia_money_l1921_192199
