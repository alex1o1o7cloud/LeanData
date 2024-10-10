import Mathlib

namespace book_arrangement_l1358_135849

def arrange_books (total : ℕ) (group1 : ℕ) (group2 : ℕ) : Prop :=
  total = group1 + group2 ∧ 
  Nat.choose total group1 = Nat.choose total group2

theorem book_arrangement : 
  arrange_books 9 4 5 → Nat.choose 9 4 = 126 := by
sorry

end book_arrangement_l1358_135849


namespace program_count_proof_l1358_135822

/-- The number of thirty-minute programs in a television schedule where
    one-fourth of the airing time is spent on commercials and
    45 minutes are spent on commercials for the whole duration of these programs. -/
def number_of_programs : ℕ := 6

/-- The duration of each program in minutes. -/
def program_duration : ℕ := 30

/-- The fraction of airing time spent on commercials. -/
def commercial_fraction : ℚ := 1/4

/-- The total time spent on commercials for all programs in minutes. -/
def total_commercial_time : ℕ := 45

theorem program_count_proof :
  number_of_programs = total_commercial_time / (commercial_fraction * program_duration) :=
by sorry

end program_count_proof_l1358_135822


namespace equation_solutions_l1358_135831

theorem equation_solutions : 
  {x : ℝ | (47 - 2*x)^(1/4) + (35 + 2*x)^(1/4) = 4} = {23, -17} := by
sorry

end equation_solutions_l1358_135831


namespace tims_soda_cans_l1358_135838

theorem tims_soda_cans (S : ℕ) : 
  (S - 10) + (S - 10) / 2 + 10 = 34 → S = 26 :=
by sorry

end tims_soda_cans_l1358_135838


namespace subtraction_with_division_l1358_135841

theorem subtraction_with_division : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end subtraction_with_division_l1358_135841


namespace a_value_in_set_equality_l1358_135894

theorem a_value_in_set_equality (a b : ℝ) : 
  let A : Set ℝ := {a, b, 2}
  let B : Set ℝ := {2, b^2, 2*a}
  A ∩ B = A ∪ B → a = 0 ∨ a = 1/4 := by
  sorry

end a_value_in_set_equality_l1358_135894


namespace parallel_lines_condition_l1358_135804

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a / b = d / e) ∧ (b ≠ 0) ∧ (e ≠ 0)

theorem parallel_lines_condition (a : ℝ) :
  (a = 1 → are_parallel a 2 (-1) 1 (a + 1) (-4)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 (-1) 1 (b + 1) (-4)) :=
sorry

end parallel_lines_condition_l1358_135804


namespace y_derivative_l1358_135803

noncomputable def y (x : ℝ) : ℝ := 3 * Real.arcsin (3 / (4 * x + 1)) + 2 * Real.sqrt (4 * x^2 + 2 * x - 2)

theorem y_derivative (x : ℝ) (h : 4 * x + 1 > 0) :
  deriv y x = (7 * (4 * x + 1)) / (2 * Real.sqrt (4 * x^2 + 2 * x - 2)) := by
  sorry

end y_derivative_l1358_135803


namespace problem_solving_questions_count_l1358_135864

-- Define the total number of multiple-choice questions
def total_mc : ℕ := 35

-- Define the fraction of multiple-choice questions already written
def mc_written_fraction : ℚ := 2/5

-- Define the fraction of problem-solving questions already written
def ps_written_fraction : ℚ := 1/3

-- Define the total number of remaining questions to write
def remaining_questions : ℕ := 31

-- Theorem to prove
theorem problem_solving_questions_count :
  ∃ (total_ps : ℕ),
    (total_ps : ℚ) * (1 - ps_written_fraction) + 
    (total_mc : ℚ) * (1 - mc_written_fraction) = remaining_questions ∧
    total_ps = 15 := by
  sorry

end problem_solving_questions_count_l1358_135864


namespace inscribed_square_area_l1358_135871

theorem inscribed_square_area (circle_area : ℝ) (square_area : ℝ) : 
  circle_area = 25 * Real.pi → 
  ∃ (r : ℝ), 
    circle_area = Real.pi * r^2 ∧ 
    square_area = 2 * r^2 →
    square_area = 50 := by
  sorry

end inscribed_square_area_l1358_135871


namespace xy_value_l1358_135828

theorem xy_value (x y : ℕ+) (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by
  sorry

end xy_value_l1358_135828


namespace stamp_collection_theorem_l1358_135820

/-- The face value of Xiaoming's stamps in jiao -/
def xiaoming_stamp_value : ℕ := 16

/-- The face value of Xiaoliang's stamps in jiao -/
def xiaoliang_stamp_value : ℕ := 2

/-- The number of stamps Xiaoming exchanges -/
def xiaoming_exchange_count : ℕ := 2

/-- The ratio of Xiaoliang's stamps to Xiaoming's before exchange -/
def pre_exchange_ratio : ℕ := 5

/-- The ratio of Xiaoliang's stamps to Xiaoming's after exchange -/
def post_exchange_ratio : ℕ := 3

/-- The total number of stamps Xiaoming and Xiaoliang have -/
def total_stamps : ℕ := 168

theorem stamp_collection_theorem :
  let xiaoming_initial := xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value
  let xiaoming_final := xiaoming_initial + xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value - xiaoming_exchange_count
  let xiaoliang_initial := pre_exchange_ratio * xiaoming_initial
  let xiaoliang_final := xiaoliang_initial - xiaoming_exchange_count * xiaoming_stamp_value / xiaoliang_stamp_value + xiaoming_exchange_count
  (xiaoliang_final = post_exchange_ratio * xiaoming_final) →
  (xiaoming_initial + xiaoliang_initial = total_stamps) := by
  sorry

end stamp_collection_theorem_l1358_135820


namespace sum_bound_l1358_135846

theorem sum_bound (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 4) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  21 ≤ x + y + z ∧ x + y + z ≤ 45 := by
  sorry

end sum_bound_l1358_135846


namespace max_value_of_x2_plus_y2_l1358_135885

theorem max_value_of_x2_plus_y2 (x y : ℝ) (h : x^2 + y^2 = 2*x - 2*y + 2) :
  x^2 + y^2 ≤ 6 + 4 * Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 6 + 4 * Real.sqrt 2 ∧ x₀^2 + y₀^2 = 2*x₀ - 2*y₀ + 2 := by
  sorry

end max_value_of_x2_plus_y2_l1358_135885


namespace polynomial_simplification_l1358_135824

theorem polynomial_simplification (p : ℝ) :
  (5 * p^4 - 4 * p^3 + 3 * p + 2) + (-3 * p^4 + 2 * p^3 - 7 * p^2 + 8) =
  2 * p^4 - 2 * p^3 - 7 * p^2 + 3 * p + 10 := by sorry

end polynomial_simplification_l1358_135824


namespace escalator_time_l1358_135845

/-- Time taken to cover an escalator's length -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) : 
  escalator_speed = 9 →
  person_speed = 3 →
  escalator_length = 200 →
  (escalator_length / (escalator_speed + person_speed)) = 200 / (9 + 3) := by
  sorry

end escalator_time_l1358_135845


namespace identical_answers_possible_l1358_135853

/-- A person who either always tells the truth or always lies -/
inductive TruthTeller
  | Always
  | Never

/-- The response to a question, either Yes or No -/
inductive Response
  | Yes
  | No

/-- Given a question, determine the response of a TruthTeller -/
def respond (person : TruthTeller) (questionTruth : Bool) : Response :=
  match person, questionTruth with
  | TruthTeller.Always, true => Response.Yes
  | TruthTeller.Always, false => Response.No
  | TruthTeller.Never, true => Response.No
  | TruthTeller.Never, false => Response.Yes

theorem identical_answers_possible :
  ∃ (question : Bool),
    respond TruthTeller.Always question = respond TruthTeller.Never question :=
by sorry

end identical_answers_possible_l1358_135853


namespace depth_multiplier_is_fifteen_l1358_135891

/-- The depth of water in feet -/
def water_depth : ℕ := 255

/-- Ron's height in feet -/
def ron_height : ℕ := 13

/-- The difference between Dean's and Ron's heights in feet -/
def height_difference : ℕ := 4

/-- Dean's height in feet -/
def dean_height : ℕ := ron_height + height_difference

/-- The multiplier for Dean's height to find the depth of the water -/
def depth_multiplier : ℕ := water_depth / dean_height

theorem depth_multiplier_is_fifteen :
  depth_multiplier = 15 :=
by sorry

end depth_multiplier_is_fifteen_l1358_135891


namespace abs_x_minus_four_plus_x_l1358_135817

theorem abs_x_minus_four_plus_x (x : ℝ) (h : |x - 3| + x - 3 = 0) : |x - 4| + x = 4 := by
  sorry

end abs_x_minus_four_plus_x_l1358_135817


namespace square_sum_equals_34_l1358_135862

theorem square_sum_equals_34 (x y : ℕ+) 
  (h1 : x * y + x + y = 23)
  (h2 : x^2 * y + x * y^2 = 120) : 
  x^2 + y^2 = 34 := by
  sorry

end square_sum_equals_34_l1358_135862


namespace cubic_root_ratio_l1358_135837

theorem cubic_root_ratio (a b c d : ℝ) (h : a ≠ 0) :
  (∃ x y z : ℝ, x = 1 ∧ y = (1/2 : ℝ) ∧ z = 4 ∧
    ∀ t : ℝ, a * t^3 + b * t^2 + c * t + d = 0 ↔ t = x ∨ t = y ∨ t = z) →
  c / d = -(13/4 : ℝ) := by
sorry

end cubic_root_ratio_l1358_135837


namespace hundredthOddPositiveInteger_l1358_135816

-- Define the function for the nth odd positive integer
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

-- Theorem statement
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end hundredthOddPositiveInteger_l1358_135816


namespace set_A_equals_one_two_l1358_135811

def A : Set ℕ := {x | x^2 - 3*x < 0 ∧ x > 0}

theorem set_A_equals_one_two : A = {1, 2} := by
  sorry

end set_A_equals_one_two_l1358_135811


namespace determinant_of_specific_matrix_l1358_135855

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, -2; 4, 3]
  Matrix.det A = 23 := by sorry

end determinant_of_specific_matrix_l1358_135855


namespace cos_sum_equals_one_l1358_135800

theorem cos_sum_equals_one (x : ℝ) (h : Real.cos (x - Real.pi / 6) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := by
  sorry

end cos_sum_equals_one_l1358_135800


namespace parallel_line_length_l1358_135840

/-- A triangle with a parallel line dividing it into equal areas -/
structure DividedTriangle where
  base : ℝ
  height : ℝ
  parallel_line : ℝ
  h_base_positive : 0 < base
  h_height_positive : 0 < height
  h_parallel_positive : 0 < parallel_line
  h_parallel_less_than_base : parallel_line < base
  h_equal_areas : parallel_line^2 / base^2 = 1/4

/-- The theorem stating that for a triangle with base 20 and height 24,
    the parallel line dividing it into four equal areas has length 10 -/
theorem parallel_line_length (t : DividedTriangle)
    (h_base : t.base = 20)
    (h_height : t.height = 24) :
    t.parallel_line = 10 := by
  sorry

end parallel_line_length_l1358_135840


namespace average_speed_calculation_l1358_135802

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) 
  (h1 : total_distance = 60)
  (h2 : first_half_distance = 30)
  (h3 : second_half_distance = 30)
  (h4 : first_half_speed = 48)
  (h5 : second_half_speed = 24)
  (h6 : total_distance = first_half_distance + second_half_distance) :
  (total_distance / (first_half_distance / first_half_speed + second_half_distance / second_half_speed)) = 32 := by
  sorry

end average_speed_calculation_l1358_135802


namespace yellow_marble_probability_l1358_135858

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Defines the bags A, B, C, and D -/
def bagA : Bag := { white := 4, black := 5 }
def bagB : Bag := { yellow := 7, blue := 3 }
def bagC : Bag := { yellow := 3, blue := 6 }
def bagD : Bag := { yellow := 5, blue := 4 }

/-- Calculates the probability of drawing a yellow marble given the problem conditions -/
def yellowProbability : ℚ :=
  let pWhiteA := bagA.white / bagA.total
  let pBlackA := bagA.black / bagA.total
  let pYellowB := bagB.yellow / bagB.total
  let pBlueB := bagB.blue / bagB.total
  let pYellowC := bagC.yellow / bagC.total
  let pBlueC := bagC.blue / bagC.total
  let pYellowD := bagD.yellow / bagD.total
  pWhiteA * pYellowB + pBlackA * pYellowC + pWhiteA * pBlueB * pYellowD + pBlackA * pBlueC * pYellowD

/-- The main theorem stating that the probability of drawing a yellow marble is 1884/3645 -/
theorem yellow_marble_probability : yellowProbability = 1884 / 3645 := by
  sorry


end yellow_marble_probability_l1358_135858


namespace inequality_and_equality_condition_l1358_135810

theorem inequality_and_equality_condition (a b n : ℕ+) (h1 : a > b) (h2 : a * b - 1 = n ^ 2) :
  (a : ℝ) - b ≥ Real.sqrt (4 * n - 3) ∧
  (∃ m : ℕ, n = m ^ 2 + m + 1 ∧ a = (m + 1) ^ 2 + 1 ∧ b = m ^ 2 + 1 ↔ (a : ℝ) - b = Real.sqrt (4 * n - 3)) :=
by sorry

end inequality_and_equality_condition_l1358_135810


namespace first_year_after_2000_with_digit_sum_15_l1358_135832

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isAfter2000 (year : ℕ) : Prop :=
  year > 2000

theorem first_year_after_2000_with_digit_sum_15 :
  ∀ year : ℕ, isAfter2000 year → sumOfDigits year = 15 → year ≥ 2049 :=
sorry

end first_year_after_2000_with_digit_sum_15_l1358_135832


namespace swim_club_members_l1358_135818

theorem swim_club_members :
  ∀ (total_members : ℕ) 
    (passed_test : ℕ) 
    (not_passed_with_course : ℕ) 
    (not_passed_without_course : ℕ),
  passed_test = (30 * total_members) / 100 →
  not_passed_with_course = 5 →
  not_passed_without_course = 30 →
  total_members = passed_test + not_passed_with_course + not_passed_without_course →
  total_members = 50 := by
sorry

end swim_club_members_l1358_135818


namespace solution_set_inequality_l1358_135812

-- Define the set containing a
def S (a : ℝ) : Set ℝ := {a^2 - 2*a + 2, a - 1, 0}

-- Theorem statement
theorem solution_set_inequality (a : ℝ) 
  (h : {1, a} ⊆ S a) : 
  {x : ℝ | a*x^2 - 5*x + a > 0} = 
  {x : ℝ | x < 1/2 ∨ x > 2} := by
sorry

end solution_set_inequality_l1358_135812


namespace race_heartbeats_l1358_135834

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

theorem race_heartbeats :
  total_heartbeats 160 6 30 = 28800 := by
  sorry

end race_heartbeats_l1358_135834


namespace inequality_proof_l1358_135861

theorem inequality_proof (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) :
  a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := by
  sorry

end inequality_proof_l1358_135861


namespace largest_solution_and_fraction_l1358_135835

theorem largest_solution_and_fraction (x : ℝ) :
  (7 * x) / 4 + 2 = 8 / x →
  ∃ (a b c d : ℤ),
    x = (a + b * Real.sqrt c) / d ∧
    a = -4 ∧ b = 8 ∧ c = 15 ∧ d = 7 ∧
    x ≤ (-4 + 8 * Real.sqrt 15) / 7 ∧
    (a * c * d : ℚ) / b = -105/2 := by
  sorry

end largest_solution_and_fraction_l1358_135835


namespace dividend_calculation_l1358_135884

/-- Calculates the total dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.07) :
  let actual_price := face_value * (1 + premium_rate)
  let num_shares := investment / actual_price
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by
sorry

end dividend_calculation_l1358_135884


namespace bianca_carrots_l1358_135890

/-- Proves that Bianca threw out 10 carrots given the initial conditions -/
theorem bianca_carrots (initial : ℕ) (next_day : ℕ) (total : ℕ) (thrown_out : ℕ) 
  (h1 : initial = 23)
  (h2 : next_day = 47)
  (h3 : total = 60)
  (h4 : initial - thrown_out + next_day = total) : 
  thrown_out = 10 := by
  sorry

end bianca_carrots_l1358_135890


namespace range_of_a_l1358_135893

/-- Proposition p: The equation represents a hyperbola -/
def p (a : ℝ) : Prop := 2 - a > 0 ∧ a + 1 > 0

/-- Proposition q: The equation has real roots -/
def q (a : ℝ) : Prop := 16 + 4 * a ≥ 0

/-- The range of a given the negation of p and q is true -/
theorem range_of_a : ∀ a : ℝ, (¬p a ∧ q a) → (a ≤ -1 ∨ a ≥ 2) ∧ a ≥ -4 := by
  sorry

end range_of_a_l1358_135893


namespace mountain_climb_time_l1358_135806

/-- Represents a climber with ascending and descending speeds -/
structure Climber where
  ascendSpeed : ℝ
  descendSpeed : ℝ

/-- The mountain climbing scenario -/
structure MountainClimb where
  a : Climber
  b : Climber
  mountainHeight : ℝ
  meetingDistance : ℝ
  meetingTime : ℝ

theorem mountain_climb_time (mc : MountainClimb) : 
  mc.a.descendSpeed = 1.5 * mc.a.ascendSpeed →
  mc.b.descendSpeed = 1.5 * mc.b.ascendSpeed →
  mc.a.ascendSpeed > mc.b.ascendSpeed →
  mc.meetingTime = 1 →
  mc.meetingDistance = 600 →
  (mc.mountainHeight / mc.a.ascendSpeed + mc.mountainHeight / mc.a.descendSpeed = 1.5) :=
by sorry

end mountain_climb_time_l1358_135806


namespace wendy_distance_difference_l1358_135814

theorem wendy_distance_difference (ran walked : ℝ) 
  (h1 : ran = 19.83) (h2 : walked = 9.17) : 
  ran - walked = 10.66 := by sorry

end wendy_distance_difference_l1358_135814


namespace right_triangle_hypotenuse_l1358_135844

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 6 →
  b = 8 →
  c^2 = a^2 + b^2 →
  c = 10 := by
  sorry

end right_triangle_hypotenuse_l1358_135844


namespace classroom_lights_theorem_l1358_135851

/-- The number of lamps in the classroom -/
def num_lamps : ℕ := 4

/-- The total number of possible states for the lights -/
def total_states : ℕ := 2^num_lamps

/-- The number of ways to turn on the lights, excluding the all-off state -/
def ways_to_turn_on : ℕ := total_states - 1

theorem classroom_lights_theorem : ways_to_turn_on = 15 := by
  sorry

end classroom_lights_theorem_l1358_135851


namespace min_value_theorem_l1358_135879

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 3) :
  (a^2 + b^2 + 22) / (a + b) ≥ 8 ∧ ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ a' * b' = 3 ∧ (a'^2 + b'^2 + 22) / (a' + b') = 8 :=
sorry

end min_value_theorem_l1358_135879


namespace pencil_price_l1358_135869

theorem pencil_price (total_cost : ℝ) (num_pens num_pencils : ℕ) (avg_pen_price : ℝ) :
  total_cost = 690 →
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 18 →
  (total_cost - num_pens * avg_pen_price) / num_pencils = 2 := by
  sorry

end pencil_price_l1358_135869


namespace translation_theorem_l1358_135850

def f (x : ℝ) : ℝ := (x - 2)^2 + 2

def g (x : ℝ) : ℝ := (x - 1)^2 + 3

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + 1) + 1 := by
  sorry

end translation_theorem_l1358_135850


namespace least_number_divisible_l1358_135883

theorem least_number_divisible (n : ℕ) : n = 857 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m + 7 = 24 * k₁ ∧ 
    m + 7 = 32 * k₂ ∧ 
    m + 7 = 36 * k₃ ∧ 
    m + 7 = 54 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n + 7 = 24 * k₁ ∧ 
    n + 7 = 32 * k₂ ∧ 
    n + 7 = 36 * k₃ ∧ 
    n + 7 = 54 * k₄) :=
by sorry

end least_number_divisible_l1358_135883


namespace birdseed_box_content_l1358_135874

/-- The number of grams of seeds in each box of birdseed -/
def grams_per_box : ℕ := 225

/-- The number of boxes Leah bought -/
def boxes_bought : ℕ := 3

/-- The number of boxes Leah already had -/
def boxes_in_pantry : ℕ := 5

/-- The number of grams the parrot eats per week -/
def parrot_consumption : ℕ := 100

/-- The number of grams the cockatiel eats per week -/
def cockatiel_consumption : ℕ := 50

/-- The number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feeding : ℕ := 12

theorem birdseed_box_content :
  grams_per_box * (boxes_bought + boxes_in_pantry) = 
    (parrot_consumption + cockatiel_consumption) * weeks_of_feeding :=
by
  sorry

#eval grams_per_box

end birdseed_box_content_l1358_135874


namespace unique_solution_system_l1358_135830

theorem unique_solution_system (x y z : ℝ) : 
  x^3 = 3*x - 12*y + 50 ∧ 
  y^3 = 12*y + 3*z - 2 ∧ 
  z^3 = 27*z + 27*x → 
  x = 2 ∧ y = 4 ∧ z = 6 :=
by sorry

end unique_solution_system_l1358_135830


namespace part_a_part_b_part_b_max_exists_l1358_135860

-- Part (a)
def P (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x + 2

theorem part_a (k : ℝ) : P k 2 = 0 → k = 5 := by sorry

-- Part (b)
theorem part_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  2*a + b + 4/(a*b) = 10 → a ≤ 4 := by sorry

theorem part_b_max_exists :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b + 4/(a*b) = 10 ∧ a = 4 := by sorry

end part_a_part_b_part_b_max_exists_l1358_135860


namespace automobile_repair_cost_l1358_135801

/-- The cost of fixing Leila's automobile given her supermarket expenses and total spending -/
def cost_to_fix_automobile (supermarket_expense : ℝ) (total_spent : ℝ) : ℝ :=
  3 * supermarket_expense + 50

/-- Theorem: Given the conditions, the cost to fix Leila's automobile is $350 -/
theorem automobile_repair_cost :
  ∃ (supermarket_expense : ℝ),
    cost_to_fix_automobile supermarket_expense 450 + supermarket_expense = 450 ∧
    cost_to_fix_automobile supermarket_expense 450 = 350 := by
  sorry

end automobile_repair_cost_l1358_135801


namespace worker_y_fraction_l1358_135807

theorem worker_y_fraction (total : ℝ) (x y : ℝ) 
  (h1 : x + y = total) 
  (h2 : 0.005 * x + 0.008 * y = 0.0065 * total) 
  (h3 : total > 0) :
  y / total = 1 / 2 := by
sorry

end worker_y_fraction_l1358_135807


namespace euler_negative_two_i_in_third_quadrant_l1358_135865

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem euler_negative_two_i_in_third_quadrant :
  third_quadrant (cexp (-2 * Complex.I)) := by
  sorry

end euler_negative_two_i_in_third_quadrant_l1358_135865


namespace dad_steps_l1358_135876

theorem dad_steps (dad_masha_ratio : ℕ → ℕ → Prop)
                   (masha_yasha_ratio : ℕ → ℕ → Prop)
                   (masha_yasha_total : ℕ) :
  dad_masha_ratio 3 5 →
  masha_yasha_ratio 3 5 →
  masha_yasha_total = 400 →
  ∃ (dad_steps : ℕ), dad_steps = 90 := by
  sorry

end dad_steps_l1358_135876


namespace area_of_four_triangles_l1358_135872

/-- The combined area of four right triangles with legs of 4 and 3 units is 24 square units. -/
theorem area_of_four_triangles :
  let triangle_area := (1 / 2) * 4 * 3
  4 * triangle_area = 24 := by sorry

end area_of_four_triangles_l1358_135872


namespace work_completion_time_l1358_135827

/-- The number of days it takes for A to finish the work alone -/
def days_A : ℝ := 22.5

/-- The number of days it takes for B to finish the work alone -/
def days_B : ℝ := 15

/-- The total wage when A and B work together -/
def total_wage : ℝ := 3400

/-- A's wage when working together with B -/
def wage_A : ℝ := 2040

theorem work_completion_time :
  days_B = 15 ∧ 
  wage_A / total_wage = 2040 / 3400 →
  days_A = 22.5 := by sorry

end work_completion_time_l1358_135827


namespace cube_root_equation_solution_l1358_135867

theorem cube_root_equation_solution : 
  ∃! x : ℝ, (7 - x / 3) ^ (1/3 : ℝ) = 5 ∧ x = -354 := by sorry

end cube_root_equation_solution_l1358_135867


namespace arithmetic_sequence_sum_ratio_l1358_135825

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- arithmetic sequence sum formula
  (S 6 / S 3 = 4) →                                         -- given condition
  (S 9 / S 6 = 9 / 4) :=                                    -- conclusion to prove
by sorry

end arithmetic_sequence_sum_ratio_l1358_135825


namespace triangle_theorem_l1358_135863

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that for a triangle ABC with parallel vectors (a, √3b) and (cos A, sin B),
    where a = √7 and b = 2, the angle A is π/3 and the area is (3√3)/2. -/
theorem triangle_theorem (t : Triangle) 
    (h1 : t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A) -- Vectors are parallel
    (h2 : t.a = Real.sqrt 7)
    (h3 : t.b = 2) :
    t.A = π / 3 ∧ (1 / 2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 3 / 2) := by
  sorry


end triangle_theorem_l1358_135863


namespace inscribed_tangent_circle_exists_l1358_135813

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an angle -/
structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a circle is inscribed in an angle -/
def isInscribed (c : Circle) (a : Angle) : Prop := sorry

/-- Predicate to check if two circles are tangent -/
def isTangent (c1 c2 : Circle) : Prop := sorry

/-- Theorem stating that given an angle and a circle, there exists an inscribed circle tangent to the given circle -/
theorem inscribed_tangent_circle_exists (a : Angle) (c : Circle) :
  ∃ (inscribed_circle : Circle), isInscribed inscribed_circle a ∧ isTangent inscribed_circle c := by
  sorry

end inscribed_tangent_circle_exists_l1358_135813


namespace baker_cakes_l1358_135848

theorem baker_cakes (cakes_sold : ℕ) (cakes_remaining : ℕ) (initial_cakes : ℕ) : 
  cakes_sold = 10 → cakes_remaining = 139 → initial_cakes = cakes_sold + cakes_remaining → initial_cakes = 149 := by
  sorry

end baker_cakes_l1358_135848


namespace remainder_sum_l1358_135836

theorem remainder_sum (n : ℤ) (h : n % 18 = 11) :
  (n % 2) + (n % 3) + (n % 9) = 5 := by sorry

end remainder_sum_l1358_135836


namespace total_annual_interest_l1358_135875

def total_amount : ℝ := 1600
def interest_rate_x : ℝ := 0.06
def interest_rate_y : ℝ := 0.05
def lent_amount : ℝ := 1100
def lent_interest_rate : ℝ := 0.0500001

theorem total_annual_interest :
  ∀ x y : ℝ,
  x + y = total_amount →
  y = lent_amount →
  x * interest_rate_x + y * lent_interest_rate = 85.00011 := by
sorry

end total_annual_interest_l1358_135875


namespace expand_and_simplify_l1358_135880

theorem expand_and_simplify (x : ℝ) : 3 * (x - 4) * (x + 9) = 3 * x^2 + 15 * x - 108 := by
  sorry

end expand_and_simplify_l1358_135880


namespace gcd_lcm_336_1260_l1358_135886

theorem gcd_lcm_336_1260 : 
  (Nat.gcd 336 1260 = 84) ∧ (Nat.lcm 336 1260 = 5040) := by
  sorry

end gcd_lcm_336_1260_l1358_135886


namespace no_odd_multiples_of_6_or_8_up_to_60_l1358_135899

theorem no_odd_multiples_of_6_or_8_up_to_60 : 
  ¬∃ n : ℕ, n ≤ 60 ∧ n % 2 = 1 ∧ (n % 6 = 0 ∨ n % 8 = 0) :=
by sorry

end no_odd_multiples_of_6_or_8_up_to_60_l1358_135899


namespace min_value_sum_reciprocals_l1358_135808

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 1 := by
  sorry

end min_value_sum_reciprocals_l1358_135808


namespace shaded_area_fraction_l1358_135854

theorem shaded_area_fraction (total_squares : ℕ) (half_shaded : ℕ) (full_shaded : ℕ) :
  total_squares = 18 →
  half_shaded = 10 →
  full_shaded = 3 →
  (half_shaded / 2 + full_shaded : ℚ) / total_squares = 4 / 9 := by
  sorry

end shaded_area_fraction_l1358_135854


namespace sum_equality_in_subset_l1358_135856

theorem sum_equality_in_subset (S : Finset ℕ) :
  S ⊆ Finset.range 38 →
  S.card = 10 →
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b = c + d := by
  sorry

end sum_equality_in_subset_l1358_135856


namespace max_value_of_sum_of_squares_l1358_135859

theorem max_value_of_sum_of_squares (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (∃ (m : ℝ), ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 10 →
    (x - y)^2 + (x - z)^2 + (x - w)^2 + (y - z)^2 + (y - w)^2 + (z - w)^2 ≤ m) ∧
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 40 :=
by sorry

end max_value_of_sum_of_squares_l1358_135859


namespace interior_angle_sum_difference_l1358_135878

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The difference in sum of interior angles between an (n+1)-sided polygon and an n-sided polygon is 180° -/
theorem interior_angle_sum_difference (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles (n + 1) - sum_interior_angles n = 180 := by
  sorry

end interior_angle_sum_difference_l1358_135878


namespace soda_price_after_increase_l1358_135896

theorem soda_price_after_increase (candy_price : ℝ) (soda_price : ℝ) : 
  candy_price = 10 →
  candy_price + soda_price = 16 →
  9 = soda_price * 1.5 :=
by
  sorry

end soda_price_after_increase_l1358_135896


namespace quadratic_equation_solution_l1358_135843

theorem quadratic_equation_solution : 
  {x : ℝ | x^2 - 9 = 0} = {-3, 3} := by sorry

end quadratic_equation_solution_l1358_135843


namespace nested_function_ratio_l1358_135877

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

def g (x : ℝ) : ℝ := 2 * x^2 - x + 1

theorem nested_function_ratio :
  f (g (f 1)) / g (f (g 1)) = 6801 / 281 := by
  sorry

end nested_function_ratio_l1358_135877


namespace base_10_300_equals_base_6_1220_l1358_135815

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

/-- Theorem stating that 300 in base 10 is equal to 1220 in base 6 -/
theorem base_10_300_equals_base_6_1220 : 
  300 = to_decimal [0, 2, 2, 1] 6 := by
  sorry

end base_10_300_equals_base_6_1220_l1358_135815


namespace factorial_sum_remainder_mod_7_l1358_135889

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

theorem factorial_sum_remainder_mod_7 : sum_factorials 10 % 7 = 5 := by sorry

end factorial_sum_remainder_mod_7_l1358_135889


namespace katie_sold_four_bead_necklaces_l1358_135857

/-- The number of bead necklaces Katie sold at her garage sale. -/
def bead_necklaces : ℕ := sorry

/-- The number of gem stone necklaces Katie sold. -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars. -/
def cost_per_necklace : ℕ := 3

/-- The total earnings from the necklace sale in dollars. -/
def total_earnings : ℕ := 21

/-- Theorem stating that Katie sold 4 bead necklaces. -/
theorem katie_sold_four_bead_necklaces : 
  bead_necklaces = 4 :=
by sorry

end katie_sold_four_bead_necklaces_l1358_135857


namespace quadratic_factorization_sum_l1358_135892

theorem quadratic_factorization_sum (d e f : ℤ) : 
  (∀ x, x^2 + 13*x + 40 = (x + d) * (x + e)) →
  (∀ x, x^2 - 19*x + 88 = (x - e) * (x - f)) →
  d + e + f = 24 :=
by sorry

end quadratic_factorization_sum_l1358_135892


namespace max_ab_value_l1358_135882

/-- The function f(x) defined in the problem -/
def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 12 * x^2 - 2 * a * x - 2 * b

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f_deriv a b 1 = 0) :
  (∃ (max_ab : ℝ), max_ab = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f_deriv a' b' 1 = 0 → a' * b' ≤ max_ab) :=
sorry

end max_ab_value_l1358_135882


namespace berry_swap_difference_l1358_135833

/-- The number of blueberries in each blue box -/
def blueberries_per_box : ℕ := 20

/-- The increase in total berries when swapping one blue box for one red box -/
def berry_increase : ℕ := 10

/-- The number of strawberries in each red box -/
def strawberries_per_box : ℕ := blueberries_per_box + berry_increase

/-- The change in the difference between total strawberries and total blueberries -/
def difference_change : ℕ := strawberries_per_box + blueberries_per_box

theorem berry_swap_difference :
  difference_change = 50 :=
sorry

end berry_swap_difference_l1358_135833


namespace expected_black_balls_l1358_135888

/-- The expected number of black balls drawn when drawing 3 balls without replacement from a bag containing 5 red balls and 2 black balls. -/
theorem expected_black_balls (total : Nat) (red : Nat) (black : Nat) (drawn : Nat) 
  (h_total : total = 7)
  (h_red : red = 5)
  (h_black : black = 2)
  (h_drawn : drawn = 3)
  (h_sum : red + black = total) :
  (0 : ℚ) * (Nat.choose red drawn : ℚ) / (Nat.choose total drawn : ℚ) +
  (1 : ℚ) * (Nat.choose red (drawn - 1) * Nat.choose black 1 : ℚ) / (Nat.choose total drawn : ℚ) +
  (2 : ℚ) * (Nat.choose red (drawn - 2) * Nat.choose black 2 : ℚ) / (Nat.choose total drawn : ℚ) = 6 / 7 := by
  sorry

end expected_black_balls_l1358_135888


namespace jasmine_laps_l1358_135881

/-- Calculates the total number of laps swum over a period of weeks -/
def total_laps (laps_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * num_weeks

/-- Proves that Jasmine swims 300 laps in 5 weeks -/
theorem jasmine_laps : total_laps 12 5 5 = 300 := by
  sorry

end jasmine_laps_l1358_135881


namespace power_function_decreasing_n_l1358_135847

/-- A power function f(x) = ax^n where a and n are constants and x > 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), ∀ x > 0, f x = a * x^n

/-- A function f is decreasing on (0, +∞) if for all x, y in (0, +∞) with x < y, f(x) > f(y) -/
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x > f y

/-- The main theorem -/
theorem power_function_decreasing_n (n : ℝ) :
  (isPowerFunction (fun x ↦ (n^2 - n - 1) * x^n)) ∧
  (isDecreasingOn (fun x ↦ (n^2 - n - 1) * x^n)) ↔
  n = -1 := by
  sorry

end power_function_decreasing_n_l1358_135847


namespace sufficient_drivers_and_schedule_l1358_135868

-- Define the duration of trips and rest time
def one_way_trip_duration : ℕ := 160  -- in minutes
def round_trip_duration : ℕ := 320    -- in minutes
def min_rest_duration : ℕ := 60       -- in minutes

-- Define the schedule times (in minutes since midnight)
def driver_a_return : ℕ := 12 * 60 + 40
def driver_d_departure : ℕ := 13 * 60 + 5
def driver_b_return : ℕ := 16 * 60
def driver_a_second_departure : ℕ := 16 * 60 + 10
def driver_b_second_departure : ℕ := 17 * 60 + 30

-- Define the number of drivers
def num_drivers : ℕ := 4

-- Define the end time of the last trip
def last_trip_end : ℕ := 21 * 60 + 30

-- Theorem statement
theorem sufficient_drivers_and_schedule :
  (num_drivers = 4) ∧
  (driver_a_return + min_rest_duration ≤ driver_d_departure) ∧
  (driver_b_return + min_rest_duration ≤ driver_b_second_departure) ∧
  (driver_a_second_departure + round_trip_duration = last_trip_end) ∧
  (last_trip_end ≤ 24 * 60) → 
  (num_drivers ≥ 4) ∧ (last_trip_end = 21 * 60 + 30) := by
  sorry

end sufficient_drivers_and_schedule_l1358_135868


namespace tetrahedron_unique_large_angle_sum_l1358_135823

/-- A tetrahedron is a structure with four vertices and six edges. -/
structure Tetrahedron :=
  (A B C D : Point)

/-- The plane angle between two edges at a vertex of a tetrahedron. -/
def planeAngle (t : Tetrahedron) (v1 v2 v3 : Point) : ℝ := sorry

/-- The property that the sum of any two plane angles at a vertex is greater than 180°. -/
def hasLargeAngleSum (t : Tetrahedron) (v : Point) : Prop :=
  ∀ (v1 v2 v3 : Point), v1 ≠ v2 → v1 ≠ v3 → v2 ≠ v3 →
    planeAngle t v v1 v2 + planeAngle t v v1 v3 > 180

/-- Theorem: No more than one vertex of a tetrahedron can have the large angle sum property. -/
theorem tetrahedron_unique_large_angle_sum (t : Tetrahedron) :
  ¬∃ (v1 v2 : Point), v1 ≠ v2 ∧ hasLargeAngleSum t v1 ∧ hasLargeAngleSum t v2 :=
sorry

end tetrahedron_unique_large_angle_sum_l1358_135823


namespace ellipse_smallest_area_l1358_135898

/-- Given an ellipse that contains two specific circles, prove its smallest possible area -/
theorem ellipse_smallest_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  ∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
    ∀ a' b' : ℝ, (∀ x y : ℝ, x^2/a'^2 + y^2/b'^2 = 1 → 
      ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) →
    π * a' * b' ≥ k * π :=
by sorry

end ellipse_smallest_area_l1358_135898


namespace integer_solution_of_inequalities_l1358_135866

theorem integer_solution_of_inequalities :
  ∃! (x : ℤ), (3 * x - 4 ≤ 6 * x - 2) ∧ ((2 * x + 1) / 3 - 1 < (x - 1) / 2) ∧ x = 0 := by
  sorry

end integer_solution_of_inequalities_l1358_135866


namespace solution_set_equals_interval_l1358_135842

-- Define the solution set of |x-3| < 5
def solution_set : Set ℝ := {x : ℝ | |x - 3| < 5}

-- State the theorem
theorem solution_set_equals_interval : solution_set = Set.Ioo (-2) 8 := by
  sorry

end solution_set_equals_interval_l1358_135842


namespace no_k_with_prime_roots_l1358_135809

/-- A quadratic equation x^2 - 65x + k = 0 with prime roots -/
def has_prime_roots (k : ℤ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ 
  (p : ℤ) + (q : ℤ) = 65 ∧ (p : ℤ) * (q : ℤ) = k

/-- There are no integer values of k for which the quadratic equation has prime roots -/
theorem no_k_with_prime_roots : ¬∃ k : ℤ, has_prime_roots k := by
  sorry

end no_k_with_prime_roots_l1358_135809


namespace expression_simplification_l1358_135829

theorem expression_simplification :
  (((3 + 5 + 6 - 2) * 2) / 4) + ((3 * 4 + 6 - 4) / 3) = 32 / 3 := by
  sorry

end expression_simplification_l1358_135829


namespace condition_relationship_l1358_135852

theorem condition_relationship (x : ℝ) :
  (x > 1/3 → 1/x < 3) ∧ ¬(1/x < 3 → x > 1/3) :=
by sorry

end condition_relationship_l1358_135852


namespace simplest_fraction_with_conditions_l1358_135870

theorem simplest_fraction_with_conditions (a b : ℕ) : 
  (a : ℚ) / b = 45 / 56 →
  ∃ (x : ℕ), a = x^2 →
  ∃ (y : ℕ), b = y^3 →
  ∃ (c d : ℕ), (c : ℚ) / d = 1 ∧ 
    (∀ (e f : ℕ), (e : ℚ) / f = 45 / 56 → 
      (∃ (g : ℕ), e = g^2) → 
      (∃ (h : ℕ), f = h^3) → 
      (c : ℚ) / d ≤ (e : ℚ) / f) :=
by sorry

end simplest_fraction_with_conditions_l1358_135870


namespace jerome_contacts_l1358_135887

/-- The number of people on Jerome's contact list -/
def total_contacts (classmates out_of_school_friends parents sisters : ℕ) : ℕ :=
  classmates + out_of_school_friends + parents + sisters

/-- Theorem stating the total number of contacts on Jerome's list -/
theorem jerome_contacts : ∃ (classmates out_of_school_friends parents sisters : ℕ),
  classmates = 20 ∧
  out_of_school_friends = classmates / 2 ∧
  parents = 2 ∧
  sisters = 1 ∧
  total_contacts classmates out_of_school_friends parents sisters = 33 := by
  sorry

end jerome_contacts_l1358_135887


namespace point_in_second_quadrant_l1358_135819

/-- A point in the second quadrant with specific properties has coordinates (-2, 1) -/
theorem point_in_second_quadrant (P : ℝ × ℝ) :
  (P.1 < 0 ∧ P.2 > 0) →  -- Second quadrant condition
  (abs P.1 = 2) →        -- |x| = 2 condition
  (P.2^2 = 1) →          -- y is square root of 1 condition
  P = (-2, 1) :=
by sorry

end point_in_second_quadrant_l1358_135819


namespace sum_base7_and_base13_equals_1109_l1358_135839

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10, where 'C' represents 12 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 536 (base 7) and 4C5 (base 13) is 1109 in base 10 -/
theorem sum_base7_and_base13_equals_1109 : 
  base7ToBase10 536 + base13ToBase10 4125 = 1109 := by sorry

end sum_base7_and_base13_equals_1109_l1358_135839


namespace triangle_angle_contradiction_l1358_135826

theorem triangle_angle_contradiction (α β γ : ℝ) : 
  (α > 60 ∧ β > 60 ∧ γ > 60) → 
  (α + β + γ = 180) → 
  False :=
sorry

end triangle_angle_contradiction_l1358_135826


namespace phone_bill_increase_l1358_135805

theorem phone_bill_increase (original_bill : ℝ) (increase_percent : ℝ) (months : ℕ) : 
  original_bill = 50 ∧ 
  increase_percent = 10 ∧ 
  months = 12 → 
  original_bill * (1 + increase_percent / 100) * months = 660 := by
  sorry

end phone_bill_increase_l1358_135805


namespace isosceles_right_triangle_ratio_l1358_135873

/-- For an isosceles right triangle with legs of length a, 
    the ratio of twice a leg to the hypotenuse is √2 -/
theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : 
  (2 * a) / Real.sqrt (a^2 + a^2) = Real.sqrt 2 := by
  sorry

end isosceles_right_triangle_ratio_l1358_135873


namespace triangle_trigonometric_identities_l1358_135821

theorem triangle_trigonometric_identities (A B C : ℝ) 
  (h : A + B + C = π) : 
  (Real.sin A + Real.sin B + Real.sin C = 
    4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2)) ∧
  (Real.tan A + Real.tan B + Real.tan C = 
    Real.tan A * Real.tan B * Real.tan C) := by
  sorry

end triangle_trigonometric_identities_l1358_135821


namespace valid_triplets_are_solution_set_l1358_135897

def is_valid_triplet (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  (b + c + 1) % a = 0 ∧
  (c + a + 1) % b = 0 ∧
  (a + b + 1) % c = 0

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 1, 1), (1, 2, 2), (1, 1, 3), (2, 2, 5), (3, 3, 7), (1, 4, 6),
   (2, 6, 9), (3, 8, 12), (4, 10, 15), (5, 12, 18), (6, 14, 21)}

theorem valid_triplets_are_solution_set :
  ∀ a b c : ℕ, is_valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end valid_triplets_are_solution_set_l1358_135897


namespace inequality_proof_l1358_135895

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a * b + b * c + c * a = 1) : 
  Real.sqrt (a^3 + a) + Real.sqrt (b^3 + b) + Real.sqrt (c^3 + c) ≥ 2 * Real.sqrt (a + b + c) := by
  sorry

end inequality_proof_l1358_135895
