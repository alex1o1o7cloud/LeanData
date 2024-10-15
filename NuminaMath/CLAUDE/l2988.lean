import Mathlib

namespace NUMINAMATH_CALUDE_survey_III_participants_l2988_298825

/-- Represents the systematic sampling method for a school survey. -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_selected : ℕ
  survey_III_start : ℕ
  survey_III_end : ℕ

/-- The number of students participating in Survey III. -/
def students_in_survey_III (s : SystematicSampling) : ℕ :=
  let n_start := ((s.survey_III_start + s.first_selected - 1) + s.group_size - 1) / s.group_size
  let n_end := (s.survey_III_end + s.first_selected - 1) / s.group_size
  n_end - n_start + 1

/-- Theorem stating the number of students in Survey III for the given conditions. -/
theorem survey_III_participants (s : SystematicSampling) 
  (h1 : s.total_students = 1080)
  (h2 : s.sample_size = 90)
  (h3 : s.group_size = s.total_students / s.sample_size)
  (h4 : s.first_selected = 5)
  (h5 : s.survey_III_start = 847)
  (h6 : s.survey_III_end = 1080) :
  students_in_survey_III s = 19 := by
  sorry

#eval students_in_survey_III {
  total_students := 1080,
  sample_size := 90,
  group_size := 12,
  first_selected := 5,
  survey_III_start := 847,
  survey_III_end := 1080
}

end NUMINAMATH_CALUDE_survey_III_participants_l2988_298825


namespace NUMINAMATH_CALUDE_minimum_balls_in_box_l2988_298838

theorem minimum_balls_in_box (blue : ℕ) (white : ℕ) (total : ℕ) : 
  white = 8 * blue →
  total = blue + white →
  (∀ drawn : ℕ, drawn = 100 → drawn > white) →
  total ≥ 108 := by
sorry

end NUMINAMATH_CALUDE_minimum_balls_in_box_l2988_298838


namespace NUMINAMATH_CALUDE_one_integer_is_seventeen_l2988_298832

theorem one_integer_is_seventeen (a b c d : ℕ+) 
  (eq1 : (b.val + c.val + d.val) / 3 + 2 * a.val = 54)
  (eq2 : (a.val + c.val + d.val) / 3 + 2 * b.val = 50)
  (eq3 : (a.val + b.val + d.val) / 3 + 2 * c.val = 42)
  (eq4 : (a.val + b.val + c.val) / 3 + 2 * d.val = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 := by
sorry

end NUMINAMATH_CALUDE_one_integer_is_seventeen_l2988_298832


namespace NUMINAMATH_CALUDE_total_animals_theorem_l2988_298858

/-- Calculates the total number of animals seen given initial counts and changes --/
def total_animals_seen (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 4 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 20
  let afternoon_total := afternoon_beavers + afternoon_chipmunks
  morning_total + afternoon_total

/-- Theorem stating that given the specific initial counts and changes, the total animals seen is 410 --/
theorem total_animals_theorem : total_animals_seen 50 90 = 410 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_theorem_l2988_298858


namespace NUMINAMATH_CALUDE_probability_log_base_3_is_integer_l2988_298857

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_power_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

def count_three_digit_powers_of_three : ℕ := 2

def total_three_digit_numbers : ℕ := 900

theorem probability_log_base_3_is_integer :
  (count_three_digit_powers_of_three : ℚ) / (total_three_digit_numbers : ℚ) = 1 / 450 := by
  sorry

#check probability_log_base_3_is_integer

end NUMINAMATH_CALUDE_probability_log_base_3_is_integer_l2988_298857


namespace NUMINAMATH_CALUDE_odd_function_properties_l2988_298893

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_function_properties (f : ℝ → ℝ) 
    (h_odd : is_odd f) 
    (h_shift : ∀ x, f (x - 2) = -f x) : 
    (f 2 = 0) ∧ 
    (periodic f 4) ∧ 
    (∀ x, f (x + 2) = f (-x)) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_properties_l2988_298893


namespace NUMINAMATH_CALUDE_negation_equivalence_l2988_298847

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (3 : ℝ) ^ x + x < 0) ↔ (∀ x : ℝ, (3 : ℝ) ^ x + x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2988_298847


namespace NUMINAMATH_CALUDE_inequality_proof_l2988_298880

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (a^4 + b^4)/(a^2 + b^2) + (b^4 + c^4)/(b^2 + c^2) + (c^4 + d^4)/(c^2 + d^2) + (d^4 + a^4)/(d^2 + a^2) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2988_298880


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l2988_298872

/-- The speed of the man in still water -/
def man_speed : ℝ := 7

/-- The speed of the stream -/
def stream_speed : ℝ := 1

/-- The distance traveled downstream -/
def downstream_distance : ℝ := 40

/-- The distance traveled upstream -/
def upstream_distance : ℝ := 30

/-- The time taken for each journey -/
def journey_time : ℝ := 5

theorem man_speed_in_still_water :
  (downstream_distance / journey_time = man_speed + stream_speed) ∧
  (upstream_distance / journey_time = man_speed - stream_speed) →
  man_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l2988_298872


namespace NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l2988_298895

/-- The problem of calculating the additional amount needed for free shipping -/
def free_shipping_problem (free_shipping_threshold : ℚ) 
                          (shampoo_price : ℚ) 
                          (conditioner_price : ℚ) 
                          (lotion_price : ℚ) 
                          (lotion_quantity : ℕ) : ℚ :=
  let total_spent := shampoo_price + conditioner_price + lotion_price * (lotion_quantity : ℚ)
  max (free_shipping_threshold - total_spent) 0

/-- Theorem stating the correct additional amount needed for free shipping -/
theorem additional_amount_for_free_shipping :
  free_shipping_problem 50 10 10 6 3 = 12 :=
sorry

end NUMINAMATH_CALUDE_additional_amount_for_free_shipping_l2988_298895


namespace NUMINAMATH_CALUDE_irrational_numbers_have_square_roots_l2988_298820

theorem irrational_numbers_have_square_roots : ∃ (x : ℝ), Irrational x ∧ ∃ (y : ℝ), y^2 = x := by
  sorry

end NUMINAMATH_CALUDE_irrational_numbers_have_square_roots_l2988_298820


namespace NUMINAMATH_CALUDE_bella_roses_count_l2988_298891

/-- The number of roses in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of roses Bella received from her parents -/
def roses_from_parents_dozens : ℕ := 2

/-- The number of Bella's dancer friends -/
def number_of_friends : ℕ := 10

/-- The number of roses Bella received from each friend -/
def roses_per_friend : ℕ := 2

/-- The total number of roses Bella received -/
def total_roses : ℕ := roses_from_parents_dozens * dozen + number_of_friends * roses_per_friend

theorem bella_roses_count : total_roses = 44 := by
  sorry

end NUMINAMATH_CALUDE_bella_roses_count_l2988_298891


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2988_298846

/-- Given points P, Q, R, and S on a line in that order, with PQ = 3, QR = 6, and PS = 20,
    the ratio of PR to QS is 9/17. -/
theorem ratio_of_segments (P Q R S : ℝ) : 
  P < Q ∧ Q < R ∧ R < S →  -- Points are in order on the line
  Q - P = 3 →              -- PQ = 3
  R - Q = 6 →              -- QR = 6
  S - P = 20 →             -- PS = 20
  (R - P) / (S - Q) = 9 / 17 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2988_298846


namespace NUMINAMATH_CALUDE_sin_570_degrees_l2988_298861

theorem sin_570_degrees : 2 * Real.sin (570 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_570_degrees_l2988_298861


namespace NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l2988_298804

/-- The number of kids who stay home during the break in Lawrence county -/
def kids_staying_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Theorem stating the number of kids staying home during the break in Lawrence county -/
theorem lawrence_county_kids_at_home :
  kids_staying_home 313473 38608 = 274865 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_at_home_l2988_298804


namespace NUMINAMATH_CALUDE_league_games_l2988_298806

theorem league_games (num_teams : ℕ) (total_games : ℕ) (games_per_matchup : ℕ) : 
  num_teams = 50 → 
  total_games = 4900 → 
  games_per_matchup * (num_teams - 1) * num_teams = 2 * total_games → 
  games_per_matchup = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_league_games_l2988_298806


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2988_298850

/-- Given two lines in the form ax + by + c = 0, they are parallel if and only if they have the same slope (a/b ratio) -/
def parallel_lines (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂

/-- A point (x, y) lies on a line ax + by + c = 0 if and only if the equation is satisfied -/
def point_on_line (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

/-- The theorem states that the line 3x + 4y - 11 = 0 is parallel to 3x + 4y + 1 = 0 and passes through (1, 2) -/
theorem parallel_line_through_point :
  parallel_lines 3 4 (-11) 3 4 1 ∧ point_on_line 3 4 (-11) 1 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2988_298850


namespace NUMINAMATH_CALUDE_prime_cube_equation_solutions_l2988_298803

theorem prime_cube_equation_solutions :
  ∀ m n p : ℕ+,
    Nat.Prime p.val →
    (m.val^3 + n.val) * (n.val^3 + m.val) = p.val^3 →
    ((m = 2 ∧ n = 1 ∧ p = 3) ∨ (m = 1 ∧ n = 2 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_equation_solutions_l2988_298803


namespace NUMINAMATH_CALUDE_discount_percentage_l2988_298866

/-- Proves that the discount percentage is 10% given the costs and final paid amount -/
theorem discount_percentage (couch_cost sectional_cost other_cost paid : ℚ)
  (h1 : couch_cost = 2500)
  (h2 : sectional_cost = 3500)
  (h3 : other_cost = 2000)
  (h4 : paid = 7200) :
  (1 - paid / (couch_cost + sectional_cost + other_cost)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l2988_298866


namespace NUMINAMATH_CALUDE_xyz_inequality_l2988_298840

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^3 + y^3 + z^3 ≥ x*y + y*z + x*z := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2988_298840


namespace NUMINAMATH_CALUDE_equal_share_ratio_l2988_298892

def total_amount : ℕ := 5400
def num_children : ℕ := 3
def b_share : ℕ := 1800

theorem equal_share_ratio :
  ∃ (a_share c_share : ℕ),
    a_share + b_share + c_share = total_amount ∧
    a_share = c_share ∧
    a_share = b_share :=
by sorry

end NUMINAMATH_CALUDE_equal_share_ratio_l2988_298892


namespace NUMINAMATH_CALUDE_basketball_cards_cost_l2988_298842

/-- The cost of Mary's sunglasses -/
def sunglasses_cost : ℕ := 50

/-- The number of sunglasses Mary bought -/
def num_sunglasses : ℕ := 2

/-- The cost of Mary's jeans -/
def jeans_cost : ℕ := 100

/-- The cost of Rose's shoes -/
def shoes_cost : ℕ := 150

/-- The number of basketball card decks Rose bought -/
def num_card_decks : ℕ := 2

/-- Mary's total spending -/
def mary_total : ℕ := num_sunglasses * sunglasses_cost + jeans_cost

/-- Rose's total spending -/
def rose_total : ℕ := shoes_cost + num_card_decks * (mary_total - shoes_cost) / num_card_decks

theorem basketball_cards_cost (h : mary_total = rose_total) : 
  (mary_total - shoes_cost) / num_card_decks = 25 := by
  sorry

end NUMINAMATH_CALUDE_basketball_cards_cost_l2988_298842


namespace NUMINAMATH_CALUDE_power_two_mod_four_l2988_298819

theorem power_two_mod_four : 2^300 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_two_mod_four_l2988_298819


namespace NUMINAMATH_CALUDE_units_digit_17_to_17_l2988_298836

theorem units_digit_17_to_17 : (17^17 : ℕ) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_17_to_17_l2988_298836


namespace NUMINAMATH_CALUDE_photo_arrangements_l2988_298881

def team_size : ℕ := 6

theorem photo_arrangements (captain_positions : ℕ) (ab_arrangements : ℕ) (remaining_arrangements : ℕ) :
  captain_positions = 2 →
  ab_arrangements = 2 →
  remaining_arrangements = 24 →
  captain_positions * ab_arrangements * remaining_arrangements = 96 :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2988_298881


namespace NUMINAMATH_CALUDE_homework_difference_l2988_298848

theorem homework_difference (total : ℕ) (math : ℕ) (reading : ℕ)
  (h1 : total = 13)
  (h2 : math = 8)
  (h3 : total = math + reading) :
  math - reading = 3 :=
by sorry

end NUMINAMATH_CALUDE_homework_difference_l2988_298848


namespace NUMINAMATH_CALUDE_evaluate_expression_l2988_298817

theorem evaluate_expression (y : ℝ) (h : y = -3) : 
  (5 + y * (5 + y) - 5^2) / (y - 5 + y^2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2988_298817


namespace NUMINAMATH_CALUDE_fraction_simplification_l2988_298823

theorem fraction_simplification : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2988_298823


namespace NUMINAMATH_CALUDE_ladder_problem_l2988_298897

/-- Given a right triangle with hypotenuse 13 meters and one leg 12 meters,
    prove that the other leg is 5 meters. -/
theorem ladder_problem (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : b = 12) :
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_ladder_problem_l2988_298897


namespace NUMINAMATH_CALUDE_test_total_points_l2988_298851

-- Define the test structure
structure Test where
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

-- Define the function to calculate total points
def calculateTotalPoints (test : Test) : ℕ :=
  2 * test.two_point_questions + 4 * test.four_point_questions

-- Theorem statement
theorem test_total_points (test : Test) 
  (h1 : test.total_questions = 40)
  (h2 : test.two_point_questions = 30)
  (h3 : test.four_point_questions = 10)
  (h4 : test.total_questions = test.two_point_questions + test.four_point_questions) :
  calculateTotalPoints test = 100 := by
  sorry

-- Example usage
def exampleTest : Test := {
  total_questions := 40,
  two_point_questions := 30,
  four_point_questions := 10
}

#eval calculateTotalPoints exampleTest

end NUMINAMATH_CALUDE_test_total_points_l2988_298851


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_81_l2988_298853

theorem right_triangle_arithmetic_progression_81 :
  ∃ (a d : ℕ), 
    (a > 0) ∧ (d > 0) ∧
    (a - d)^2 + a^2 = (a + d)^2 ∧
    (81 = a - d ∨ 81 = a ∨ 81 = a + d) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_81_l2988_298853


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2988_298826

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2988_298826


namespace NUMINAMATH_CALUDE_correct_expansion_l2988_298821

theorem correct_expansion (x : ℝ) : (-3*x + 2) * (-3*x - 2) = 9*x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_expansion_l2988_298821


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2988_298801

theorem geometric_sequence_middle_term (y : ℝ) :
  (3^2 : ℝ) < y ∧ y < (3^4 : ℝ) ∧ 
  (y / (3^2 : ℝ)) = ((3^4 : ℝ) / y) →
  y = 27 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l2988_298801


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1234_l2988_298862

theorem base_seven_digits_of_1234 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 1234 ∧ 1234 < 7^n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1234_l2988_298862


namespace NUMINAMATH_CALUDE_fraction_sum_equals_two_thirds_l2988_298811

theorem fraction_sum_equals_two_thirds : 
  2 / 10 + 4 / 40 + 6 / 60 + 8 / 30 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_two_thirds_l2988_298811


namespace NUMINAMATH_CALUDE_gcd_12012_21021_l2988_298824

theorem gcd_12012_21021 : Nat.gcd 12012 21021 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_21021_l2988_298824


namespace NUMINAMATH_CALUDE_tan_negative_210_degrees_l2988_298809

theorem tan_negative_210_degrees : 
  Real.tan (-(210 * π / 180)) = -(Real.sqrt 3 / 3) := by sorry

end NUMINAMATH_CALUDE_tan_negative_210_degrees_l2988_298809


namespace NUMINAMATH_CALUDE_jones_trip_time_comparison_l2988_298835

theorem jones_trip_time_comparison 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (speed_multiplier : ℝ) 
  (h1 : distance1 = 50) 
  (h2 : distance2 = 300) 
  (h3 : speed_multiplier = 3) :
  let time1 := distance1 / (distance1 / time1)
  let time2 := distance2 / (speed_multiplier * (distance1 / time1))
  time2 = 2 * time1 := by
sorry

end NUMINAMATH_CALUDE_jones_trip_time_comparison_l2988_298835


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2988_298868

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  x_tangent : Point
  has_y_tangent : Bool

/-- Calculate the length of the major axis of an ellipse -/
def majorAxisLength (e : Ellipse) : ℝ :=
  -- Implementation details omitted
  sorry

/-- Theorem: The length of the major axis of the given ellipse is 4 -/
theorem ellipse_major_axis_length :
  let e : Ellipse := {
    focus1 := { x := 4, y := 2 + 2 * Real.sqrt 2 },
    focus2 := { x := 4, y := 2 - 2 * Real.sqrt 2 },
    x_tangent := { x := 4, y := 0 },
    has_y_tangent := true
  }
  majorAxisLength e = 4 := by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2988_298868


namespace NUMINAMATH_CALUDE_race_result_l2988_298815

/-- Represents a runner in the race -/
structure Runner where
  position : ℝ
  speed : ℝ

/-- The race setup and result -/
theorem race_result 
  (race_length : ℝ) 
  (a b : Runner) 
  (h1 : race_length = 3000)
  (h2 : a.position = race_length - 500)
  (h3 : b.position = race_length - 600)
  (h4 : a.speed > 0)
  (h5 : b.speed > 0) :
  let time_to_finish_a := (race_length - a.position) / a.speed
  let b_final_position := b.position + b.speed * time_to_finish_a
  race_length - b_final_position = 120 := by
sorry

end NUMINAMATH_CALUDE_race_result_l2988_298815


namespace NUMINAMATH_CALUDE_marble_combination_count_l2988_298818

def num_marbles_per_color : ℕ := 2
def num_colors : ℕ := 4
def total_marbles : ℕ := num_marbles_per_color * num_colors

def choose_two_same_color : ℕ := num_colors * (num_marbles_per_color.choose 2)
def choose_two_diff_colors : ℕ := (num_colors.choose 2) * num_marbles_per_color * num_marbles_per_color

theorem marble_combination_count :
  choose_two_same_color + choose_two_diff_colors = 28 := by
  sorry

end NUMINAMATH_CALUDE_marble_combination_count_l2988_298818


namespace NUMINAMATH_CALUDE_work_completion_time_l2988_298865

theorem work_completion_time 
  (a_rate : ℝ) (b_rate : ℝ) (work_left : ℝ) (days_worked : ℝ) : 
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  work_left = 0.41666666666666663 →
  (1 - work_left) = (a_rate + b_rate) * days_worked →
  days_worked = 5 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2988_298865


namespace NUMINAMATH_CALUDE_max_value_of_a_l2988_298896

-- Define the condition function
def condition (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

-- Define the main theorem
theorem max_value_of_a :
  (∀ x, x < a → condition x) ∧ 
  (∃ x, condition x ∧ x ≥ a) →
  ∀ b, (∀ x, x < b → condition x) ∧ 
       (∃ x, condition x ∧ x ≥ b) →
  b ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2988_298896


namespace NUMINAMATH_CALUDE_eliza_initial_rings_l2988_298869

/-- The number of ornamental rings Eliza initially bought -/
def initial_rings : ℕ := 100

/-- The total stock after Eliza's purchase -/
def total_stock : ℕ := 3 * initial_rings

/-- The remaining stock after selling 3/4 of the total -/
def remaining_after_sale : ℕ := (total_stock * 1) / 4

/-- The stock after mother's purchase -/
def stock_after_mother_purchase : ℕ := remaining_after_sale + 300

/-- The final stock -/
def final_stock : ℕ := stock_after_mother_purchase - 150

theorem eliza_initial_rings :
  final_stock = 225 :=
by sorry

end NUMINAMATH_CALUDE_eliza_initial_rings_l2988_298869


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2988_298849

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem greatest_five_digit_with_product_90 :
  ∃ M : ℕ, is_five_digit M ∧ 
    digit_product M = 90 ∧ 
    (∀ n : ℕ, is_five_digit n → digit_product n = 90 → n ≤ M) ∧
    digit_sum M = 17 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l2988_298849


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2988_298898

/-- Given Tom's current age t and Lily's current age l, prove that the smallest positive integer x
    that satisfies (t + x) / (l + x) = 3 is 22, where t and l satisfy the given conditions. -/
theorem age_ratio_problem (t l : ℕ) (h1 : t - 3 = 5 * (l - 3)) (h2 : t - 8 = 6 * (l - 8)) :
  (∃ x : ℕ, x > 0 ∧ (t + x : ℚ) / (l + x) = 3 ∧ ∀ y : ℕ, y > 0 → (t + y : ℚ) / (l + y) = 3 → x ≤ y) →
  (∃ x : ℕ, x = 22 ∧ x > 0 ∧ (t + x : ℚ) / (l + x) = 3 ∧ ∀ y : ℕ, y > 0 → (t + y : ℚ) / (l + y) = 3 → x ≤ y) :=
by
  sorry


end NUMINAMATH_CALUDE_age_ratio_problem_l2988_298898


namespace NUMINAMATH_CALUDE_maddie_total_cost_l2988_298837

/-- Calculates the total cost of Maddie's beauty products purchase --/
def total_cost (palette_price : ℚ) (palette_quantity : ℕ) 
               (lipstick_price : ℚ) (lipstick_quantity : ℕ)
               (hair_color_price : ℚ) (hair_color_quantity : ℕ) : ℚ :=
  palette_price * palette_quantity + 
  lipstick_price * lipstick_quantity + 
  hair_color_price * hair_color_quantity

/-- Theorem stating that Maddie's total cost is $67 --/
theorem maddie_total_cost : 
  total_cost 15 3 (5/2) 4 4 3 = 67 := by
  sorry

end NUMINAMATH_CALUDE_maddie_total_cost_l2988_298837


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2988_298800

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2988_298800


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l2988_298882

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  (Complex.re z₁ = -Complex.re z₂) →
  (Complex.im z₁ = Complex.im z₂) →
  (z₁ = 3 + Complex.I) →
  z₁ * z₂ = -10 := by
sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l2988_298882


namespace NUMINAMATH_CALUDE_train_speed_l2988_298829

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 360) (h2 : time = 30) :
  length / time = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2988_298829


namespace NUMINAMATH_CALUDE_candy_problem_l2988_298889

theorem candy_problem (x : ℕ) : 
  (x % 12 = 0) →
  (∃ c : ℕ, c ≥ 1 ∧ c ≤ 3 ∧ 
   ((3 * x / 4) * 2 / 3 - 20 - c = 5)) →
  (x = 52 ∨ x = 56) := by
sorry

end NUMINAMATH_CALUDE_candy_problem_l2988_298889


namespace NUMINAMATH_CALUDE_solve_cottage_problem_l2988_298802

def cottage_problem (hourly_rate : ℚ) (jack_paid : ℚ) (jill_paid : ℚ) : Prop :=
  let total_paid := jack_paid + jill_paid
  let hours_rented := total_paid / hourly_rate
  hours_rented = 8

theorem solve_cottage_problem :
  cottage_problem 5 20 20 := by sorry

end NUMINAMATH_CALUDE_solve_cottage_problem_l2988_298802


namespace NUMINAMATH_CALUDE_bookstore_sales_l2988_298879

theorem bookstore_sales (tuesday : ℕ) (total : ℕ) : 
  total = tuesday + 3 * tuesday + 9 * tuesday → 
  total = 91 → 
  tuesday = 7 := by
sorry

end NUMINAMATH_CALUDE_bookstore_sales_l2988_298879


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l2988_298870

def G (n : ℕ) : ℕ := 3^(2^n) + 2

theorem units_digit_G_1000 : G 1000 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l2988_298870


namespace NUMINAMATH_CALUDE_square_sum_division_theorem_l2988_298867

theorem square_sum_division_theorem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_division_theorem_l2988_298867


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2988_298828

theorem circle_area_with_diameter_10 :
  ∀ (d : ℝ) (A : ℝ), 
    d = 10 →
    A = π * (d / 2)^2 →
    A = 25 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2988_298828


namespace NUMINAMATH_CALUDE_factor_w4_minus_16_l2988_298874

theorem factor_w4_minus_16 (w : ℝ) : w^4 - 16 = (w-2)*(w+2)*(w^2+4) := by sorry

end NUMINAMATH_CALUDE_factor_w4_minus_16_l2988_298874


namespace NUMINAMATH_CALUDE_fibonacci_tetrahedron_volume_zero_l2988_298843

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def tetrahedron_vertex (n : ℕ) : ℕ × ℕ × ℕ :=
  (fibonacci n, fibonacci (n + 1), fibonacci (n + 2))

def tetrahedron_volume (n : ℕ) : ℝ :=
  let v1 := tetrahedron_vertex n
  let v2 := tetrahedron_vertex (n + 3)
  let v3 := tetrahedron_vertex (n + 6)
  let v4 := tetrahedron_vertex (n + 9)
  -- Volume calculation would go here
  0  -- Placeholder for the actual volume calculation

theorem fibonacci_tetrahedron_volume_zero (n : ℕ) :
  tetrahedron_volume n = 0 := by
  sorry

#check fibonacci_tetrahedron_volume_zero

end NUMINAMATH_CALUDE_fibonacci_tetrahedron_volume_zero_l2988_298843


namespace NUMINAMATH_CALUDE_triangle_problem_l2988_298808

/-- Triangle ABC with angles A, B, C opposite sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions and theorem -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.A = π / 6)
  (h2 : (1 + Real.sqrt 3) * t.c = 2 * t.b) :
  t.C = π / 4 ∧ 
  (t.b * t.a * Real.cos t.C = 1 + Real.sqrt 3 → 
    t.a = Real.sqrt 2 ∧ t.b = 1 + Real.sqrt 3 ∧ t.c = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2988_298808


namespace NUMINAMATH_CALUDE_function_properties_l2988_298864

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

theorem function_properties :
  ∃ m : ℝ,
    (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x m ≤ 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x m = 1) ∧
    m = -2 ∧
    (∀ x : ℝ, f x m ≥ -3) ∧
    (∀ k : ℤ, f ((2 * Real.pi / 3) + k * Real.pi) m = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2988_298864


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l2988_298810

/-- Represents a trapezoid ABCD with sides AB and CD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ

/-- The theorem stating the relationship between the sides of the trapezoid
    given the area ratio of triangles ABC and ADC -/
theorem trapezoid_side_length (ABCD : Trapezoid)
    (h1 : (ABCD.AB / ABCD.CD) = (7 : ℝ) / 3)
    (h2 : ABCD.AB + ABCD.CD = 210) :
    ABCD.AB = 147 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l2988_298810


namespace NUMINAMATH_CALUDE_real_roots_iff_k_leq_5_root_one_implies_k_values_l2988_298888

-- Define the quadratic equation
def quadratic (k x : ℝ) : ℝ := x^2 - 2*(k-3)*x + k^2 - 4*k - 1

-- Theorem 1: The equation has real roots iff k ≤ 5
theorem real_roots_iff_k_leq_5 (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0) ↔ k ≤ 5 := by sorry

-- Theorem 2: If 1 is a root, then k = 3 + √3 or k = 3 - √3
theorem root_one_implies_k_values (k : ℝ) : 
  quadratic k 1 = 0 → k = 3 + Real.sqrt 3 ∨ k = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_leq_5_root_one_implies_k_values_l2988_298888


namespace NUMINAMATH_CALUDE_right_triangle_angle_l2988_298855

theorem right_triangle_angle (α β : ℝ) : 
  α + β + 90 = 180 → β = 70 → α = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_l2988_298855


namespace NUMINAMATH_CALUDE_circle_containment_l2988_298877

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius --/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

theorem circle_containment (circles : Fin 6 → Circle) 
  (O : ℝ × ℝ) (h : ∀ i, is_inside O (circles i)) :
  ∃ i j, i ≠ j ∧ is_inside (circles j).center (circles i) := by
  sorry

end NUMINAMATH_CALUDE_circle_containment_l2988_298877


namespace NUMINAMATH_CALUDE_pen_cost_l2988_298871

theorem pen_cost (pen_price : ℝ) (briefcase_price : ℝ) : 
  briefcase_price = 5 * pen_price →
  pen_price + briefcase_price = 24 →
  pen_price = 4 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l2988_298871


namespace NUMINAMATH_CALUDE_cos_double_angle_special_l2988_298887

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point (3,4) on its terminal side, prove that cos 2α = -7/25 -/
theorem cos_double_angle_special (α : Real) 
  (h1 : ∃ (x y : Real), x = 3 ∧ y = 4 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
                        y = Real.sin α * Real.sqrt (x^2 + y^2)) : 
  Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_l2988_298887


namespace NUMINAMATH_CALUDE_power_relation_l2988_298833

theorem power_relation (x m n : ℝ) (h1 : x^m = 6) (h2 : x^n = 9) : x^(2*m - n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l2988_298833


namespace NUMINAMATH_CALUDE_tan_and_expression_values_l2988_298859

theorem tan_and_expression_values (α : Real) 
  (h_acute : 0 < α ∧ α < π / 2)
  (h_tan : Real.tan (π / 4 + α) = 2) :
  Real.tan α = 1 / 3 ∧ 
  (Real.sqrt 2 * Real.sin (2 * α + π / 4) * Real.cos α - Real.sin α) / Real.cos (2 * α) = (2 / 5) * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_and_expression_values_l2988_298859


namespace NUMINAMATH_CALUDE_exercise_239_theorem_existence_not_implied_l2988_298834

-- Define a property A for functions
def PropertyA (f : ℝ → ℝ) : Prop := sorry

-- Define periodicity for functions
def Periodic (f : ℝ → ℝ) : Prop := ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- The theorem from exercise 239
theorem exercise_239_theorem : ∀ f : ℝ → ℝ, PropertyA f → Periodic f := sorry

-- The statement we want to prove
theorem existence_not_implied :
  (∀ f : ℝ → ℝ, PropertyA f → Periodic f) →
  ¬(∃ f : ℝ → ℝ, PropertyA f) := sorry

end NUMINAMATH_CALUDE_exercise_239_theorem_existence_not_implied_l2988_298834


namespace NUMINAMATH_CALUDE_range_of_a_l2988_298827

theorem range_of_a (a : ℝ) (n : ℕ) (h1 : a > 1) (h2 : n ≥ 2) 
  (h3 : ∃! (s : Finset ℤ), s.card = n ∧ ∀ x ∈ s, ⌊a * x⌋ = x) :
  1 + 1 / n ≤ a ∧ a < 1 + 1 / (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2988_298827


namespace NUMINAMATH_CALUDE_train_distance_theorem_l2988_298841

/-- Represents the distance traveled by the second train -/
def x : ℝ := 400

/-- The speed of the first train in km/hr -/
def speed1 : ℝ := 50

/-- The speed of the second train in km/hr -/
def speed2 : ℝ := 40

/-- The additional distance traveled by the first train compared to the second train -/
def additional_distance : ℝ := 100

/-- The total distance between the starting points of the two trains -/
def total_distance : ℝ := x + (x + additional_distance)

theorem train_distance_theorem :
  speed1 > 0 ∧ speed2 > 0 ∧ 
  x / speed2 = (x + additional_distance) / speed1 →
  total_distance = 900 :=
sorry

end NUMINAMATH_CALUDE_train_distance_theorem_l2988_298841


namespace NUMINAMATH_CALUDE_alpha_values_l2988_298814

theorem alpha_values (α : ℂ) 
  (h1 : α ≠ Complex.I ∧ α ≠ -Complex.I)
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : (Complex.abs (α^4 - 1))^2 = 9 * (Complex.abs (α - 1))^2) :
  α = (1/2 : ℂ) + Complex.I * (Real.sqrt 35 / 2) ∨ 
  α = (1/2 : ℂ) - Complex.I * (Real.sqrt 35 / 2) :=
sorry

end NUMINAMATH_CALUDE_alpha_values_l2988_298814


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2988_298852

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {1, 3, 5, 7, 9}
def B : Set Nat := {1, 2, 5, 6, 8}

theorem intersection_complement_equality : A ∩ (U \ B) = {3, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2988_298852


namespace NUMINAMATH_CALUDE_solution_set_equals_given_values_l2988_298845

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The set of solutions to the equation n = 2S(n)³ + 8 -/
def SolutionSet : Set ℕ := {n : ℕ | n > 0 ∧ n = 2 * (S n)^3 + 8}

/-- The theorem stating that the solution set contains exactly 10, 2008, and 13726 -/
theorem solution_set_equals_given_values : 
  SolutionSet = {10, 2008, 13726} := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_given_values_l2988_298845


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l2988_298885

theorem trig_expression_equals_four :
  1 / Real.cos (80 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l2988_298885


namespace NUMINAMATH_CALUDE_total_legs_in_group_l2988_298875

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of humans in the group -/
def num_humans : ℕ := 2

/-- The number of dogs in the group -/
def num_dogs : ℕ := 2

/-- Theorem stating that the total number of legs in the group is 12 -/
theorem total_legs_in_group : 
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_group_l2988_298875


namespace NUMINAMATH_CALUDE_total_kids_signed_up_l2988_298884

/-- The number of girls signed up for the talent show. -/
def num_girls : ℕ := 28

/-- The difference between the number of girls and boys signed up. -/
def girl_boy_difference : ℕ := 22

/-- Theorem: The total number of kids signed up for the talent show is 34. -/
theorem total_kids_signed_up : 
  num_girls + (num_girls - girl_boy_difference) = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_signed_up_l2988_298884


namespace NUMINAMATH_CALUDE_quadratic_condition_l2988_298873

theorem quadratic_condition (m : ℝ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, m * x^2 - 4*x + 3 = a * x^2 + b * x + c) → m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_condition_l2988_298873


namespace NUMINAMATH_CALUDE_equation_holds_l2988_298899

theorem equation_holds (x y z : ℤ) (h1 : x = z) (h2 : y - 1 = x) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l2988_298899


namespace NUMINAMATH_CALUDE_trapezoid_EFGH_area_l2988_298813

/-- Trapezoid with vertices E(0,0), F(0,3), G(5,0), and H(5,7) -/
structure Trapezoid where
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  H : ℝ × ℝ

/-- The area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- The theorem stating the area of the specific trapezoid EFGH -/
theorem trapezoid_EFGH_area :
  let t : Trapezoid := {
    E := (0, 0),
    F := (0, 3),
    G := (5, 0),
    H := (5, 7)
  }
  trapezoidArea t = 25 := by sorry

end NUMINAMATH_CALUDE_trapezoid_EFGH_area_l2988_298813


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l2988_298807

theorem triangle_angle_proof 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a + b + c) * (a - b + c) = a * c)
  (h2 : Real.sin A * Real.sin C = (Real.sqrt 3 - 1) / 4) : 
  B = 2 * π / 3 ∧ (C = π / 12 ∨ C = π / 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_proof_l2988_298807


namespace NUMINAMATH_CALUDE_sequence_convergence_l2988_298856

theorem sequence_convergence (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, (a (n + 2))^2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l2988_298856


namespace NUMINAMATH_CALUDE_eightieth_number_is_eighty_l2988_298844

def game_sequence (n : ℕ) : ℕ := n

theorem eightieth_number_is_eighty : game_sequence 80 = 80 := by sorry

end NUMINAMATH_CALUDE_eightieth_number_is_eighty_l2988_298844


namespace NUMINAMATH_CALUDE_employment_agency_payroll_l2988_298805

/-- Calculates the total payroll for an employment agency given the number of employees,
    number of laborers, and pay rates for heavy operators and laborers. -/
theorem employment_agency_payroll
  (total_employees : ℕ)
  (num_laborers : ℕ)
  (heavy_operator_pay : ℕ)
  (laborer_pay : ℕ)
  (h1 : total_employees = 31)
  (h2 : num_laborers = 1)
  (h3 : heavy_operator_pay = 129)
  (h4 : laborer_pay = 82) :
  (total_employees - num_laborers) * heavy_operator_pay + num_laborers * laborer_pay = 3952 :=
by
  sorry


end NUMINAMATH_CALUDE_employment_agency_payroll_l2988_298805


namespace NUMINAMATH_CALUDE_surface_area_change_after_cube_removal_l2988_298890

/-- Represents a rectangular solid with length, width, and height -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Calculates the change in surface area after removing a cube from the center -/
def surfaceAreaChange (solid : RectangularSolid) (cubeSize : ℝ) : ℝ :=
  6 * cubeSize^2

/-- The theorem to be proved -/
theorem surface_area_change_after_cube_removal :
  let original := RectangularSolid.mk 4 3 2
  let cubeSize := 1
  surfaceAreaChange original cubeSize = 6 := by sorry

end NUMINAMATH_CALUDE_surface_area_change_after_cube_removal_l2988_298890


namespace NUMINAMATH_CALUDE_common_root_values_l2988_298860

theorem common_root_values (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hk1 : a * k^4 + b * k^3 + c * k^2 + d * k + a = 0)
  (hk2 : b * k^4 + c * k^3 + d * k^2 + a * k + b = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_common_root_values_l2988_298860


namespace NUMINAMATH_CALUDE_root_ratio_sum_squared_l2988_298854

theorem root_ratio_sum_squared (k₁ k₂ : ℝ) (a b : ℝ) : 
  (∀ x, k₁ * (x^2 - x) + x + 3 = 0 → (x = a ∨ x = b)) →
  (∀ x, k₂ * (x^2 - x) + x + 3 = 0 → (x = a ∨ x = b)) →
  a / b + b / a = 2 →
  k₁^2 + k₂^2 = 194 := by
sorry

end NUMINAMATH_CALUDE_root_ratio_sum_squared_l2988_298854


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2988_298816

/-- The speed of a boat in still water, given downstream and upstream speeds and current speed. -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h1 : current_speed = 17)
  (h2 : downstream_speed = 77)
  (h3 : upstream_speed = 43) :
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 60 ∧ 
    still_water_speed + current_speed = downstream_speed ∧ 
    still_water_speed - current_speed = upstream_speed :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2988_298816


namespace NUMINAMATH_CALUDE_theme_parks_calculation_l2988_298831

/-- The number of theme parks in three towns -/
def total_theme_parks (jamestown venice marina_del_ray : ℕ) : ℕ :=
  jamestown + venice + marina_del_ray

/-- Theorem stating the total number of theme parks in the three towns -/
theorem theme_parks_calculation :
  ∃ (jamestown venice marina_del_ray : ℕ),
    jamestown = 20 ∧
    venice = jamestown + 25 ∧
    marina_del_ray = jamestown + 50 ∧
    total_theme_parks jamestown venice marina_del_ray = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_theme_parks_calculation_l2988_298831


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2988_298883

/-- Given a positive real number r, prove that if the line x - y = r is tangent to the circle x^2 + y^2 = r, then r = 2 -/
theorem tangent_line_to_circle (r : ℝ) (hr : r > 0) : 
  (∀ x y : ℝ, x - y = r → x^2 + y^2 ≤ r) ∧ 
  (∃ x y : ℝ, x - y = r ∧ x^2 + y^2 = r) → 
  r = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2988_298883


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2988_298839

open Real

/-- Given a function f: ℝ → ℝ with a tangent line at x = 2 described by the equation 2x - y - 3 = 0,
    prove that f(2) + f'(2) = 3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y, y = f 2 → 2 * x - y - 3 = 0 ↔ y = 2 * x - 3) :
  f 2 + deriv f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2988_298839


namespace NUMINAMATH_CALUDE_expected_consecutive_reds_l2988_298876

/-- A bag containing one red, one yellow, and one blue ball -/
inductive Ball : Type
| Red : Ball
| Yellow : Ball
| Blue : Ball

/-- The process of drawing balls with replacement -/
def DrawProcess : Type := ℕ → Ball

/-- The probability of drawing each color is equal -/
axiom equal_probability (b : Ball) : ℝ

/-- The sum of probabilities is 1 -/
axiom prob_sum : equal_probability Ball.Red + equal_probability Ball.Yellow + equal_probability Ball.Blue = 1

/-- ξ is the number of draws until two consecutive red balls are drawn -/
def ξ (process : DrawProcess) : ℕ := sorry

/-- The expected value of ξ -/
def expected_ξ : ℝ := sorry

/-- Theorem: The expected value of ξ is 12 -/
theorem expected_consecutive_reds : expected_ξ = 12 := by sorry

end NUMINAMATH_CALUDE_expected_consecutive_reds_l2988_298876


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l2988_298822

theorem real_part_of_complex_number (z : ℂ) 
  (h1 : Complex.abs z = 1)
  (h2 : Complex.abs (z - 1.45) = 1.05) :
  z.re = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l2988_298822


namespace NUMINAMATH_CALUDE_amount_in_scientific_notation_l2988_298863

-- Define the amount in yuan
def amount : ℕ := 25000000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 2.5 * (10 ^ 10)

-- Theorem statement
theorem amount_in_scientific_notation :
  (amount : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_amount_in_scientific_notation_l2988_298863


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_l2988_298830

theorem mixed_number_multiplication : 
  (39 + 18 / 19) * (18 + 19 / 20) = 757 + 1 / 380 := by sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_l2988_298830


namespace NUMINAMATH_CALUDE_prob_product_divisible_l2988_298886

/-- Represents a standard 6-sided die --/
def StandardDie : Type := Fin 6

/-- The probability of rolling a number on a standard die --/
def prob_roll (n : Nat) : ℚ := if n ≥ 1 ∧ n ≤ 6 then 1 / 6 else 0

/-- The probability that a single die roll is not divisible by 2, 3, and 5 --/
def prob_not_divisible : ℚ := 5 / 18

/-- The number of dice rolled --/
def num_dice : Nat := 6

/-- The probability that the product of 6 dice rolls is divisible by 2, 3, or 5 --/
theorem prob_product_divisible :
  1 - prob_not_divisible ^ num_dice = 33996599 / 34012224 := by sorry

end NUMINAMATH_CALUDE_prob_product_divisible_l2988_298886


namespace NUMINAMATH_CALUDE_root_transformation_l2988_298878

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 5 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 5 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 5 = 0) → 
  ((3*r₁)^3 - 12*(3*r₁)^2 + 135 = 0) ∧ 
  ((3*r₂)^3 - 12*(3*r₂)^2 + 135 = 0) ∧ 
  ((3*r₃)^3 - 12*(3*r₃)^2 + 135 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l2988_298878


namespace NUMINAMATH_CALUDE_martha_points_l2988_298894

/-- Represents Martha's shopping trip and point system. -/
structure ShoppingTrip where
  /-- Points earned per $10 spent -/
  pointsPerTen : ℕ
  /-- Bonus points for spending over $100 -/
  overHundredBonus : ℕ
  /-- Bonus points for 5th visit -/
  fifthVisitBonus : ℕ
  /-- Price of beef per pound -/
  beefPrice : ℚ
  /-- Quantity of beef in pounds -/
  beefQuantity : ℕ
  /-- Discount on beef as a percentage -/
  beefDiscount : ℚ
  /-- Price of fruits and vegetables per pound -/
  fruitVegPrice : ℚ
  /-- Quantity of fruits and vegetables in pounds -/
  fruitVegQuantity : ℕ
  /-- Discount on fruits and vegetables as a percentage -/
  fruitVegDiscount : ℚ
  /-- Price of spices per jar -/
  spicePrice : ℚ
  /-- Quantity of spice jars -/
  spiceQuantity : ℕ
  /-- Discount on spices as a percentage -/
  spiceDiscount : ℚ
  /-- Price of other groceries before coupon -/
  otherGroceriesPrice : ℚ
  /-- Coupon value for other groceries -/
  otherGroceriesCoupon : ℚ

/-- Calculates the total points earned during the shopping trip. -/
def calculatePoints (trip : ShoppingTrip) : ℕ :=
  sorry

/-- Theorem stating that Martha earns 850 points given the specific shopping conditions. -/
theorem martha_points : ∃ (trip : ShoppingTrip),
  trip.pointsPerTen = 50 ∧
  trip.overHundredBonus = 250 ∧
  trip.fifthVisitBonus = 100 ∧
  trip.beefPrice = 11 ∧
  trip.beefQuantity = 3 ∧
  trip.beefDiscount = 1/10 ∧
  trip.fruitVegPrice = 4 ∧
  trip.fruitVegQuantity = 8 ∧
  trip.fruitVegDiscount = 2/25 ∧
  trip.spicePrice = 6 ∧
  trip.spiceQuantity = 3 ∧
  trip.spiceDiscount = 1/20 ∧
  trip.otherGroceriesPrice = 37 ∧
  trip.otherGroceriesCoupon = 3 ∧
  calculatePoints trip = 850 :=
sorry

end NUMINAMATH_CALUDE_martha_points_l2988_298894


namespace NUMINAMATH_CALUDE_unique_solution_l2988_298812

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (heq : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 7 ∧ y = 5 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l2988_298812
