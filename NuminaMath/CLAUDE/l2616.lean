import Mathlib

namespace square_sum_equals_37_l2616_261669

theorem square_sum_equals_37 (x y : ℝ) 
  (h1 : y + 3 = (x - 3)^2)
  (h2 : x + 3 = (y - 3)^2)
  (h3 : x ≠ y) :
  x^2 + y^2 = 37 := by
sorry

end square_sum_equals_37_l2616_261669


namespace propositions_correctness_l2616_261612

-- Proposition ①
def proposition_1 : Prop :=
  (¬∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 < 3*x)

-- Proposition ②
def proposition_2 : Prop :=
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

-- Proposition ③
def proposition_3 : Prop :=
  ∀ a : ℝ, (a > 3 → a > Real.pi) ∧ ¬(a > Real.pi → a > 3)

-- Proposition ④
def proposition_4 : Prop :=
  ∀ a : ℝ, (∀ x : ℝ, (x + 2) * (x + a) = (-x + 2) * (-x + a)) → a = -2

theorem propositions_correctness :
  ¬proposition_1 ∧ proposition_2 ∧ ¬proposition_3 ∧ proposition_4 :=
sorry

end propositions_correctness_l2616_261612


namespace sum_a_plus_d_l2616_261631

theorem sum_a_plus_d (a b c d : ℤ) 
  (eq1 : a + b = 12) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 9 := by
  sorry

end sum_a_plus_d_l2616_261631


namespace sqrt_eight_and_one_ninth_l2616_261687

theorem sqrt_eight_and_one_ninth (x : ℝ) : 
  x = Real.sqrt (8 + 1 / 9) → x = Real.sqrt 73 / 3 := by
  sorry

end sqrt_eight_and_one_ninth_l2616_261687


namespace prob_sum_8_twice_eq_l2616_261674

/-- The number of sides on each die -/
def num_sides : ℕ := 7

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The probability of rolling a sum of 8 with two dice -/
def prob_sum_8 : ℚ := 7 / 49

/-- The probability of rolling a sum of 8 twice in a row with two dice -/
def prob_sum_8_twice : ℚ := (prob_sum_8) * (prob_sum_8)

/-- Theorem: The probability of rolling a sum of 8 twice in a row
    with two 7-sided dice (numbered 1 to 7) is equal to 49/2401 -/
theorem prob_sum_8_twice_eq : prob_sum_8_twice = 49 / 2401 := by
  sorry

end prob_sum_8_twice_eq_l2616_261674


namespace geometric_number_difference_l2616_261634

/-- A 4-digit number is geometric if it has 4 distinct digits forming a geometric sequence from left to right. -/
def IsGeometric (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ a b c d r : ℕ,
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    b = a * r ∧ c = a * r^2 ∧ d = a * r^3

/-- The largest 4-digit geometric number -/
def LargestGeometric : ℕ := 9648

/-- The smallest 4-digit geometric number -/
def SmallestGeometric : ℕ := 1248

theorem geometric_number_difference :
  IsGeometric LargestGeometric ∧
  IsGeometric SmallestGeometric ∧
  (∀ n : ℕ, IsGeometric n → SmallestGeometric ≤ n ∧ n ≤ LargestGeometric) ∧
  LargestGeometric - SmallestGeometric = 8400 := by
  sorry

end geometric_number_difference_l2616_261634


namespace rate_increase_is_33_percent_l2616_261685

/-- Represents the work team's processing scenario -/
structure WorkScenario where
  initial_items : ℕ
  total_time : ℕ
  worked_time : ℕ
  additional_items : ℕ

/-- Calculates the required rate increase percentage -/
def required_rate_increase (scenario : WorkScenario) : ℚ :=
  let initial_rate := scenario.initial_items / scenario.total_time
  let processed_items := initial_rate * scenario.worked_time
  let remaining_items := scenario.initial_items - processed_items + scenario.additional_items
  let remaining_time := scenario.total_time - scenario.worked_time
  let new_rate := remaining_items / remaining_time
  (new_rate - initial_rate) / initial_rate * 100

/-- The main theorem stating that the required rate increase is 33% -/
theorem rate_increase_is_33_percent (scenario : WorkScenario) 
  (h1 : scenario.initial_items = 1250)
  (h2 : scenario.total_time = 10)
  (h3 : scenario.worked_time = 6)
  (h4 : scenario.additional_items = 165) :
  required_rate_increase scenario = 33 := by
  sorry

end rate_increase_is_33_percent_l2616_261685


namespace xy_sum_reciprocals_l2616_261692

theorem xy_sum_reciprocals (x y : ℝ) (θ : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_theta : ∀ (n : ℤ), θ ≠ π / 2 * n)
  (h_eq1 : Real.sin θ / x = Real.cos θ / y)
  (h_eq2 : Real.cos θ ^ 4 / x ^ 4 + Real.sin θ ^ 4 / y ^ 4 = 
           97 * Real.sin (2 * θ) / (x ^ 3 * y + y ^ 3 * x)) :
  x / y + y / x = 4 := by
  sorry

end xy_sum_reciprocals_l2616_261692


namespace julian_needs_80_more_legos_l2616_261637

/-- The number of additional legos Julian needs to complete two identical airplane models -/
def additional_legos_needed (total_legos : ℕ) (legos_per_model : ℕ) (num_models : ℕ) : ℕ :=
  max 0 (legos_per_model * num_models - total_legos)

/-- Proof that Julian needs 80 more legos -/
theorem julian_needs_80_more_legos :
  additional_legos_needed 400 240 2 = 80 := by
  sorry

#eval additional_legos_needed 400 240 2

end julian_needs_80_more_legos_l2616_261637


namespace parabola_coefficient_sum_l2616_261681

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := sorry

/-- Check if a point lies on the parabola -/
def containsPoint (p : Parabola) (x y : ℚ) : Prop := sorry

/-- Check if the parabola has a vertical axis of symmetry -/
def hasVerticalAxisOfSymmetry (p : Parabola) : Prop := sorry

theorem parabola_coefficient_sum 
  (p : Parabola) 
  (h1 : vertex p = (5, 3))
  (h2 : hasVerticalAxisOfSymmetry p)
  (h3 : containsPoint p 2 0) :
  p.a + p.b + p.c = -7/3 := by sorry

end parabola_coefficient_sum_l2616_261681


namespace acid_solution_concentration_l2616_261649

theorem acid_solution_concentration 
  (x : ℝ) -- original concentration
  (h1 : 0.5 * x + 0.5 * 30 = 40) -- mixing equation
  : x = 50 := by
  sorry

end acid_solution_concentration_l2616_261649


namespace chlorine_used_equals_chloromethane_formed_l2616_261645

/-- Represents the chemical reaction between Methane and Chlorine to form Chloromethane -/
structure ChemicalReaction where
  methane_initial : ℝ
  chloromethane_formed : ℝ

/-- Theorem stating that the moles of Chlorine used equals the moles of Chloromethane formed -/
theorem chlorine_used_equals_chloromethane_formed (reaction : ChemicalReaction)
  (h : reaction.methane_initial = reaction.chloromethane_formed) :
  reaction.chloromethane_formed = reaction.methane_initial :=
by sorry

end chlorine_used_equals_chloromethane_formed_l2616_261645


namespace quadratic_roots_l2616_261691

theorem quadratic_roots (m : ℝ) : 
  ((-5 : ℝ)^2 + m * (-5) - 10 = 0) → ((2 : ℝ)^2 + m * 2 - 10 = 0) := by
sorry

end quadratic_roots_l2616_261691


namespace upstream_downstream_time_relation_stream_speed_is_twelve_l2616_261656

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 36

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 12

/-- The time taken to row upstream is twice the time taken to row downstream -/
theorem upstream_downstream_time_relation (d : ℝ) (h : d > 0) :
  d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed)) :=
by sorry

/-- Proves that the stream speed is 12 kmph given the conditions -/
theorem stream_speed_is_twelve :
  stream_speed = 12 :=
by sorry

end upstream_downstream_time_relation_stream_speed_is_twelve_l2616_261656


namespace simplify_power_expression_l2616_261640

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by
  sorry

end simplify_power_expression_l2616_261640


namespace scarf_sales_with_new_price_and_tax_l2616_261684

/-- Represents the relationship between number of scarves sold and their price -/
def scarfRelation (k : ℝ) (p c : ℝ) : Prop := p * c = k

theorem scarf_sales_with_new_price_and_tax 
  (k : ℝ) 
  (initial_price initial_quantity new_price tax_rate : ℝ) : 
  scarfRelation k initial_quantity initial_price →
  initial_price = 10 →
  initial_quantity = 30 →
  new_price = 15 →
  tax_rate = 0.1 →
  ∃ (new_quantity : ℕ), 
    scarfRelation k (new_quantity : ℝ) (new_price * (1 + tax_rate)) ∧ 
    new_quantity = 18 := by
  sorry

end scarf_sales_with_new_price_and_tax_l2616_261684


namespace art_club_students_l2616_261688

/-- The number of students in the art club -/
def num_students : ℕ := 15

/-- The number of artworks each student makes per quarter -/
def artworks_per_quarter : ℕ := 2

/-- The number of quarters in a school year -/
def quarters_per_year : ℕ := 4

/-- The total number of artworks collected in two school years -/
def total_artworks : ℕ := 240

/-- Theorem stating that the number of students in the art club is 15 -/
theorem art_club_students :
  num_students * artworks_per_quarter * quarters_per_year * 2 = total_artworks :=
by sorry

end art_club_students_l2616_261688


namespace triangle_area_l2616_261693

/-- The area of a triangle with base 9 cm and height 12 cm is 54 cm² -/
theorem triangle_area : 
  let base : ℝ := 9
  let height : ℝ := 12
  (1/2 : ℝ) * base * height = 54
  := by sorry

end triangle_area_l2616_261693


namespace exponent_rule_l2616_261686

theorem exponent_rule (a : ℝ) (m : ℤ) : a^(2*m + 2) = a^(2*m) * a^2 := by
  sorry

end exponent_rule_l2616_261686


namespace sin_cos_identity_l2616_261630

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x ^ 6 + Real.cos x ^ 6 + Real.sin x ^ 2 = 2 * Real.sin x ^ 4 + Real.cos x ^ 4 := by
  sorry

end sin_cos_identity_l2616_261630


namespace classroom_gpa_problem_l2616_261629

theorem classroom_gpa_problem (class_size : ℝ) (h_class_size_pos : class_size > 0) :
  let third_size := class_size / 3
  let rest_size := class_size - third_size
  let third_gpa := 60
  let overall_gpa := 64
  let rest_gpa := (overall_gpa * class_size - third_gpa * third_size) / rest_size
  rest_gpa = 66 := by sorry

end classroom_gpa_problem_l2616_261629


namespace sequence_properties_l2616_261678

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) (k : ℝ) : ℝ := k * n^2 + n

/-- The nth term of the sequence -/
def a (n : ℕ+) (k : ℝ) : ℝ := k * (2 * n - 1) + 1

theorem sequence_properties (k : ℝ) :
  (∀ n : ℕ+, S n k - S (n-1) k = a n k) ∧
  (∀ m : ℕ+, (a (2*m) k)^2 = (a m k) * (a (4*m) k)) →
  k = 1/3 := by sorry

end sequence_properties_l2616_261678


namespace max_books_read_l2616_261620

def reading_speed : ℕ := 120
def pages_per_book : ℕ := 360
def reading_time : ℕ := 8

theorem max_books_read : 
  (reading_speed * reading_time) / pages_per_book = 2 :=
by sorry

end max_books_read_l2616_261620


namespace simple_interest_rate_l2616_261679

/-- Represents the rate of simple interest per annum -/
def rate : ℚ := 1 / 24

/-- The time period in years -/
def time : ℕ := 12

/-- The ratio of final amount to initial amount -/
def growth_ratio : ℚ := 9 / 6

theorem simple_interest_rate :
  (1 + rate * time) = growth_ratio := by sorry

end simple_interest_rate_l2616_261679


namespace james_writes_to_fourteen_people_l2616_261697

/-- Represents James' writing habits and calculates the number of people he writes to daily --/
def james_writing (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (hours_per_week : ℕ) : ℕ :=
  (pages_per_hour * hours_per_week) / pages_per_person_per_day

/-- Theorem stating that James writes to 14 people daily --/
theorem james_writes_to_fourteen_people :
  james_writing 10 5 7 = 14 := by
  sorry

end james_writes_to_fourteen_people_l2616_261697


namespace paint_cost_contribution_l2616_261648

-- Define the given conditions
def wall_area : ℝ := 1600
def paint_coverage : ℝ := 400
def paint_cost_per_gallon : ℝ := 45
def number_of_coats : ℕ := 2

-- Define the theorem
theorem paint_cost_contribution :
  let total_gallons := (wall_area / paint_coverage) * number_of_coats
  let total_cost := total_gallons * paint_cost_per_gallon
  let individual_contribution := total_cost / 2
  individual_contribution = 180 := by sorry

end paint_cost_contribution_l2616_261648


namespace colors_needed_l2616_261643

/-- The number of people coloring the planets -/
def num_people : ℕ := 3

/-- The number of planets to be colored -/
def num_planets : ℕ := 8

/-- The total number of colors needed -/
def total_colors : ℕ := num_people * num_planets

/-- Theorem stating that the total number of colors needed is 24 -/
theorem colors_needed : total_colors = 24 := by sorry

end colors_needed_l2616_261643


namespace g_of_3_equals_4_l2616_261622

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Theorem statement
theorem g_of_3_equals_4 : g 3 = 4 := by sorry

end g_of_3_equals_4_l2616_261622


namespace customer_satisfaction_probability_l2616_261682

/-- The probability that a dissatisfied customer leaves an angry review -/
def prob_angry_given_dissatisfied : ℝ := 0.8

/-- The probability that a satisfied customer leaves a positive review -/
def prob_positive_given_satisfied : ℝ := 0.15

/-- The number of angry reviews received -/
def angry_reviews : ℕ := 60

/-- The number of positive reviews received -/
def positive_reviews : ℕ := 20

/-- The probability that a customer is satisfied with the service -/
def prob_satisfied : ℝ := 0.64

theorem customer_satisfaction_probability :
  prob_satisfied = 0.64 :=
sorry

end customer_satisfaction_probability_l2616_261682


namespace expression_simplification_and_evaluation_l2616_261600

theorem expression_simplification_and_evaluation :
  ∀ x y : ℝ, (x - 2)^2 + |y + 1| = 0 →
  3 * x^2 * y - (2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 5 * x * y) = 6 := by
  sorry

end expression_simplification_and_evaluation_l2616_261600


namespace patients_per_doctor_l2616_261689

/-- Given a hospital with 400 patients and 16 doctors, prove that each doctor takes care of 25 patients. -/
theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) :
  total_patients = 400 → total_doctors = 16 →
  total_patients / total_doctors = 25 := by
  sorry

end patients_per_doctor_l2616_261689


namespace arithmetic_sequence_perfect_squares_l2616_261614

/-- An arithmetic sequence of natural numbers -/
def ArithmeticSequence (a d : ℕ) : ℕ → ℕ := fun n ↦ a + n * d

/-- A number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The theorem stating that if an arithmetic sequence of natural numbers contains
    one perfect square, it contains infinitely many perfect squares -/
theorem arithmetic_sequence_perfect_squares
  (a d : ℕ) -- First term and common difference of the arithmetic sequence
  (h : ∃ n : ℕ, IsPerfectSquare (ArithmeticSequence a d n)) :
  ∀ m : ℕ, ∃ k > m, IsPerfectSquare (ArithmeticSequence a d k) :=
sorry

end arithmetic_sequence_perfect_squares_l2616_261614


namespace ana_driving_problem_l2616_261666

theorem ana_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (additional_speed : ℝ) (target_average_speed : ℝ) (additional_distance : ℝ) :
  initial_distance = 20 →
  initial_speed = 40 →
  additional_speed = 70 →
  target_average_speed = 60 →
  additional_distance = 70 →
  (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / additional_speed)) = target_average_speed :=
by
  sorry

end ana_driving_problem_l2616_261666


namespace min_sum_abc_def_l2616_261664

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def are_distinct (a b c d e f : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def to_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem min_sum_abc_def :
  ∀ a b c d e f : ℕ,
    is_valid_digit a → is_valid_digit b → is_valid_digit c →
    is_valid_digit d → is_valid_digit e → is_valid_digit f →
    are_distinct a b c d e f →
    459 ≤ to_number a b c + to_number d e f :=
by sorry

end min_sum_abc_def_l2616_261664


namespace expand_product_l2616_261632

theorem expand_product (x : ℝ) : (3 * x - 2) * (2 * x + 4) = 6 * x^2 + 8 * x - 8 := by
  sorry

end expand_product_l2616_261632


namespace missing_number_is_eight_l2616_261603

-- Define the structure of the pyramid
def Pyramid (a b c d e : ℕ) : Prop :=
  b * c = d ∧ c * a = e ∧ d * e = 3360

-- Theorem statement
theorem missing_number_is_eight :
  ∃ (x : ℕ), Pyramid 8 6 7 42 x ∧ x > 0 :=
by sorry

end missing_number_is_eight_l2616_261603


namespace product_sum_fractions_l2616_261611

theorem product_sum_fractions : (2 * 3 * 4) * (1 / 2 + 1 / 3 + 1 / 4) = 26 := by
  sorry

end product_sum_fractions_l2616_261611


namespace nancy_homework_l2616_261615

def homework_problem (math_problems : ℝ) (problems_per_hour : ℝ) (total_hours : ℝ) : Prop :=
  let total_problems := problems_per_hour * total_hours
  let spelling_problems := total_problems - math_problems
  spelling_problems = 15.0

theorem nancy_homework :
  homework_problem 17.0 8.0 4.0 := by
  sorry

end nancy_homework_l2616_261615


namespace equal_difference_implies_square_equal_difference_equal_difference_and_equal_square_difference_implies_constant_l2616_261680

/-- A sequence is an "equal difference" sequence if the difference between consecutive terms is constant. -/
def IsEqualDifference (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is an "equal square difference" sequence if the difference between consecutive squared terms is constant. -/
def IsEqualSquareDifference (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ n : ℕ, (a (n + 1))^2 - (a n)^2 = p

/-- If a sequence is an "equal difference" sequence, then its square is also an "equal difference" sequence. -/
theorem equal_difference_implies_square_equal_difference (a : ℕ → ℝ) :
    IsEqualDifference a → IsEqualDifference (fun n ↦ (a n)^2) := by sorry

/-- If a sequence is both an "equal difference" sequence and an "equal square difference" sequence,
    then it is a constant sequence. -/
theorem equal_difference_and_equal_square_difference_implies_constant (a : ℕ → ℝ) :
    IsEqualDifference a → IsEqualSquareDifference a → ∃ c : ℝ, ∀ n : ℕ, a n = c := by sorry

end equal_difference_implies_square_equal_difference_equal_difference_and_equal_square_difference_implies_constant_l2616_261680


namespace quadratic_sum_l2616_261621

theorem quadratic_sum (a h k : ℝ) : 
  (∀ x, 8 * x^2 - 24 * x - 15 = a * (x - h)^2 + k) → 
  a + h + k = -23.5 := by
  sorry

end quadratic_sum_l2616_261621


namespace orange_savings_percentage_l2616_261676

-- Define the given conditions
def family_size : ℕ := 4
def orange_cost : ℚ := 3/2  -- $1.5 as a rational number
def planned_spending : ℚ := 15

-- Define the theorem
theorem orange_savings_percentage :
  let saved_amount := family_size * orange_cost
  let savings_ratio := saved_amount / planned_spending
  savings_ratio * 100 = 40 := by
  sorry

end orange_savings_percentage_l2616_261676


namespace arithmetic_and_geometric_sequence_equal_l2616_261625

/-- A sequence is both arithmetic and geometric if and only if all its terms are equal -/
theorem arithmetic_and_geometric_sequence_equal (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) ∧ 
  (∀ n : ℕ, a n ≠ 0 → a (n + 1) / a n = a 1 / a 0) ↔ 
  (∀ n m : ℕ, a n = a m) :=
sorry

end arithmetic_and_geometric_sequence_equal_l2616_261625


namespace commemorative_book_sales_l2616_261670

/-- Profit function for commemorative book sales -/
def profit (x : ℝ) : ℝ := (x - 20) * (-2 * x + 80)

/-- Theorem for commemorative book sales problem -/
theorem commemorative_book_sales 
  (x : ℝ) 
  (h1 : 20 ≤ x ∧ x ≤ 28) : 
  (∃ (x : ℝ), profit x = 150 ∧ x = 25) ∧ 
  (∀ (y : ℝ), 20 ≤ y ∧ y ≤ 28 → profit y ≤ profit 28) ∧
  profit 28 = 192 := by
  sorry


end commemorative_book_sales_l2616_261670


namespace class_test_results_l2616_261698

theorem class_test_results (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.60 := by
  sorry

end class_test_results_l2616_261698


namespace fathers_age_is_45_l2616_261642

/-- Proves that the father's age is 45 given the problem conditions -/
theorem fathers_age_is_45 (F C : ℕ) : 
  F = 3 * C →  -- Father's age is three times the sum of the ages of his two children
  F + 5 = 2 * (C + 10) →  -- After 5 years, father's age will be twice the sum of age of two children
  F = 45 := by
sorry

end fathers_age_is_45_l2616_261642


namespace inverse_composition_equals_neg_eight_ninths_l2616_261659

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g⁻¹
noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

-- Theorem statement
theorem inverse_composition_equals_neg_eight_ninths :
  g_inv (g_inv 20) = -8/9 := by
  sorry

end inverse_composition_equals_neg_eight_ninths_l2616_261659


namespace juice_cost_calculation_l2616_261647

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 0.30

/-- The total amount Lyle has in dollars -/
def total_amount : ℚ := 2.50

/-- The number of friends Lyle is buying for -/
def num_friends : ℕ := 4

/-- The cost of a pack of juice in dollars -/
def juice_cost : ℚ := 0.325

theorem juice_cost_calculation : 
  sandwich_cost * num_friends + juice_cost * num_friends = total_amount :=
by sorry

end juice_cost_calculation_l2616_261647


namespace sum_24_probability_l2616_261635

/-- The number of ways to achieve a sum of 24 with 10 fair standard 6-sided dice -/
def ways_to_sum_24 : ℕ := 817190

/-- The number of possible outcomes when throwing 10 fair standard 6-sided dice -/
def total_outcomes : ℕ := 6^10

/-- The probability of achieving a sum of 24 when throwing 10 fair standard 6-sided dice -/
def prob_sum_24 : ℚ := ways_to_sum_24 / total_outcomes

theorem sum_24_probability :
  ways_to_sum_24 = 817190 ∧
  total_outcomes = 6^10 ∧
  prob_sum_24 = ways_to_sum_24 / total_outcomes :=
sorry

end sum_24_probability_l2616_261635


namespace u_difference_divisible_l2616_261654

/-- Sequence u defined recursively -/
def u (a : ℕ+) : ℕ → ℕ
  | 0 => 1
  | n + 1 => a.val ^ u a n

/-- Theorem stating that n! divides u_{n+1} - u_n for all n ≥ 1 -/
theorem u_difference_divisible (a : ℕ+) (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℕ, u a (n + 1) - u a n = k * n! :=
sorry

end u_difference_divisible_l2616_261654


namespace dartboard_section_angle_l2616_261641

theorem dartboard_section_angle (p : ℝ) (θ : ℝ) : 
  p = 1 / 4 →  -- probability of dart landing in a section
  p = θ / 360 →  -- probability equals ratio of central angle to full circle
  θ = 90 :=  -- central angle is 90 degrees
by sorry

end dartboard_section_angle_l2616_261641


namespace fraction_zero_implies_x_equals_two_l2616_261638

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (2 - |x|) / (x + 2) = 0 → x = 2 :=
by
  sorry

end fraction_zero_implies_x_equals_two_l2616_261638


namespace weight_within_range_l2616_261699

/-- The labeled weight of the flour in kilograms -/
def labeled_weight : ℝ := 25

/-- The tolerance range for the flour weight in kilograms -/
def tolerance : ℝ := 0.2

/-- The actual weight of the flour in kilograms -/
def actual_weight : ℝ := 25.1

/-- Theorem stating that the actual weight is within the acceptable range -/
theorem weight_within_range : 
  labeled_weight - tolerance ≤ actual_weight ∧ actual_weight ≤ labeled_weight + tolerance :=
by sorry

end weight_within_range_l2616_261699


namespace ellipse_major_axis_length_l2616_261662

/-- An ellipse with focal points (-2, 0) and (2, 0) that intersects the line x + y + 4 = 0 at exactly one point has a major axis of length 8. -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
  (∀ (P : ℝ × ℝ), P ∈ E ↔ 
    Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) = 8) →
  (∃! (P : ℝ × ℝ), P ∈ E ∧ P.1 + P.2 + 4 = 0) →
  8 = 8 := by
sorry


end ellipse_major_axis_length_l2616_261662


namespace intersection_trajectory_l2616_261660

/-- 
Given points A(a,0) and B(b,0) on the x-axis and a point C(0,c) on the y-axis,
prove that the trajectory of the intersection point of line l (passing through O(0,0) 
and perpendicular to AC) and line BC is described by the given equation.
-/
theorem intersection_trajectory 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hab : a ≠ b) :
  ∃ (x y : ℝ → ℝ), 
    ∀ (c : ℝ), c ≠ 0 →
      (x c - b/2)^2 / (b^2/4) + (y c)^2 / (a*b/4) = 1 :=
sorry

end intersection_trajectory_l2616_261660


namespace little_john_sweets_expenditure_l2616_261655

/-- Proof of the amount spent on sweets by Little John --/
theorem little_john_sweets_expenditure 
  (initial_amount : ℚ)
  (amount_per_friend : ℚ)
  (num_friends : ℕ)
  (final_amount : ℚ)
  (h1 : initial_amount = 20.10)
  (h2 : amount_per_friend = 1)
  (h3 : num_friends = 2)
  (h4 : final_amount = 17.05) :
  initial_amount - (↑num_friends * amount_per_friend) - final_amount = 1.05 := by
  sorry

#check little_john_sweets_expenditure

end little_john_sweets_expenditure_l2616_261655


namespace speed_ratio_and_distance_l2616_261694

/-- Represents a traveler with a constant speed -/
structure Traveler where
  speed : ℝ
  startPosition : ℝ

/-- Represents the problem setup -/
structure TravelProblem where
  A : Traveler
  B : Traveler
  C : Traveler
  distanceAB : ℝ
  timeToCMeetA : ℝ
  timeAMeetsB : ℝ
  BPastMidpoint : ℝ
  CFromA : ℝ

/-- The main theorem that proves the speed ratio and distance -/
theorem speed_ratio_and_distance 
  (p : TravelProblem)
  (h1 : p.A.startPosition = 0)
  (h2 : p.B.startPosition = 0)
  (h3 : p.C.startPosition = p.distanceAB)
  (h4 : p.timeToCMeetA = 20)
  (h5 : p.timeAMeetsB = 10)
  (h6 : p.BPastMidpoint = 105)
  (h7 : p.CFromA = 315)
  : p.A.speed / p.B.speed = 3 ∧ p.distanceAB = 1890 := by
  sorry

#check speed_ratio_and_distance

end speed_ratio_and_distance_l2616_261694


namespace triangle_side_length_l2616_261608

/-- 
Given a triangle XYZ where:
- y = 7
- z = 3
- cos(Y - Z) = 40/41
Prove that x² = 56.1951
-/
theorem triangle_side_length (X Y Z : ℝ) (x y z : ℝ) :
  y = 7 →
  z = 3 →
  Real.cos (Y - Z) = 40 / 41 →
  x ^ 2 = 56.1951 := by
  sorry

end triangle_side_length_l2616_261608


namespace xyz_sum_l2616_261696

theorem xyz_sum (x y z : ℕ+) 
  (eq1 : x * y + z = 47)
  (eq2 : y * z + x = 47)
  (eq3 : z * x + y = 47) : 
  x + y + z = 48 := by
sorry

end xyz_sum_l2616_261696


namespace vector_dot_product_collinear_l2616_261624

/-- Given two vectors a and b in ℝ², prove that if they are collinear and have specific components, their dot product satisfies a certain equation. -/
theorem vector_dot_product_collinear (k : ℝ) : 
  let a : Fin 2 → ℝ := ![3/2, 1]
  let b : Fin 2 → ℝ := ![3, k]
  (∃ (t : ℝ), b = t • a) →   -- Collinearity condition
  (a - b) • (2 • a + b) = -13 := by
  sorry

end vector_dot_product_collinear_l2616_261624


namespace A_oxen_count_l2616_261651

/-- Represents the number of oxen A put for grazing -/
def X : ℕ := sorry

/-- Total rent of the pasture in Rs -/
def total_rent : ℕ := 175

/-- Number of months A's oxen grazed -/
def A_months : ℕ := 7

/-- Number of oxen B put for grazing -/
def B_oxen : ℕ := 12

/-- Number of months B's oxen grazed -/
def B_months : ℕ := 5

/-- Number of oxen C put for grazing -/
def C_oxen : ℕ := 15

/-- Number of months C's oxen grazed -/
def C_months : ℕ := 3

/-- C's share of rent in Rs -/
def C_share : ℕ := 45

/-- Theorem stating that A put 10 oxen for grazing -/
theorem A_oxen_count : X = 10 := by sorry

end A_oxen_count_l2616_261651


namespace break_difference_l2616_261650

def work_duration : ℕ := 240
def water_break_interval : ℕ := 20
def sitting_break_interval : ℕ := 120

def water_breaks : ℕ := work_duration / water_break_interval
def sitting_breaks : ℕ := work_duration / sitting_break_interval

theorem break_difference : water_breaks - sitting_breaks = 10 := by
  sorry

end break_difference_l2616_261650


namespace work_completion_theorem_l2616_261677

/-- Represents the number of days needed to complete the work -/
def total_days_x : ℝ := 30

/-- Represents the number of days needed to complete the work -/
def total_days_y : ℝ := 15

/-- Represents the number of days x needs to finish the remaining work -/
def remaining_days_x : ℝ := 10.000000000000002

/-- Represents the number of days y worked before leaving -/
def days_y_worked : ℝ := 10

theorem work_completion_theorem :
  days_y_worked * (1 / total_days_y) + remaining_days_x * (1 / total_days_x) = 1 := by
  sorry

end work_completion_theorem_l2616_261677


namespace brendas_mice_problem_l2616_261668

theorem brendas_mice_problem (total_litters : Nat) (mice_per_litter : Nat) 
  (fraction_to_robbie : Rat) (multiplier_to_pet_store : Nat) (fraction_to_feeder : Rat) :
  total_litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1 / 6 →
  multiplier_to_pet_store = 3 →
  fraction_to_feeder = 1 / 2 →
  (total_litters * mice_per_litter 
    - (total_litters * mice_per_litter : Rat) * fraction_to_robbie 
    - (total_litters * mice_per_litter : Rat) * fraction_to_robbie * multiplier_to_pet_store) 
    * (1 - fraction_to_feeder) = 4 := by
  sorry

end brendas_mice_problem_l2616_261668


namespace negative_cube_squared_l2616_261671

theorem negative_cube_squared (a : ℝ) : -(-3*a)^2 = -9*a^2 := by
  sorry

end negative_cube_squared_l2616_261671


namespace kiwi_apple_equivalence_l2616_261628

/-- The value of kiwis in terms of apples -/
def kiwi_value (k : ℚ) : ℚ := k * 2

theorem kiwi_apple_equivalence :
  kiwi_value (1/4 * 20) = 10 →
  kiwi_value (3/4 * 12) = 18 :=
by
  sorry

end kiwi_apple_equivalence_l2616_261628


namespace apples_per_pie_l2616_261690

theorem apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) :
  initial_apples = 62 →
  handed_out = 8 →
  num_pies = 6 →
  (initial_apples - handed_out) / num_pies = 9 := by
  sorry

end apples_per_pie_l2616_261690


namespace dvd_packs_total_cost_l2616_261605

/-- Calculates the total cost of purchasing two packs of DVDs with given prices, discounts, and an additional discount for buying both. -/
def total_cost (price1 price2 discount1 discount2 additional_discount : ℕ) : ℕ :=
  (price1 - discount1) + (price2 - discount2) - additional_discount

/-- Theorem stating that the total cost of purchasing the two DVD packs is 111 dollars. -/
theorem dvd_packs_total_cost : 
  total_cost 76 85 25 15 10 = 111 := by
  sorry

end dvd_packs_total_cost_l2616_261605


namespace solution_composition_l2616_261661

theorem solution_composition (solution1_percent : Real) (solution1_carbonated : Real) 
  (solution2_carbonated : Real) (mixture_carbonated : Real) :
  solution1_percent = 0.4 →
  solution2_carbonated = 0.55 →
  mixture_carbonated = 0.65 →
  solution1_percent * solution1_carbonated + (1 - solution1_percent) * solution2_carbonated = mixture_carbonated →
  solution1_carbonated = 0.8 := by
sorry

end solution_composition_l2616_261661


namespace unique_pin_l2616_261619

def is_valid_pin (pin : Nat) : Prop :=
  pin ≥ 1000 ∧ pin < 10000 ∧
  let first_digit := pin / 1000
  let last_three_digits := pin % 1000
  10 * last_three_digits + first_digit = 3 * pin - 6

theorem unique_pin : ∃! pin, is_valid_pin pin ∧ pin = 2856 := by
  sorry

end unique_pin_l2616_261619


namespace park_tree_count_l2616_261695

/-- Calculates the final number of trees in a park after cutting --/
def final_tree_count (initial_oak initial_maple oak_cut maple_cut : ℕ) : ℕ × ℕ × ℕ :=
  let final_oak := initial_oak - oak_cut
  let final_maple := initial_maple - maple_cut
  let total := final_oak + final_maple
  (final_oak, final_maple, total)

/-- Theorem stating the final tree count after cutting in the park --/
theorem park_tree_count :
  final_tree_count 57 43 13 8 = (44, 35, 79) := by
  sorry

end park_tree_count_l2616_261695


namespace equation_solution_l2616_261636

theorem equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 18 := by
  sorry

end equation_solution_l2616_261636


namespace power_of_two_plus_one_l2616_261673

theorem power_of_two_plus_one (a : ℤ) (b : ℝ) (h : 2^a = b) : 2^(a+1) = 2*b := by
  sorry

end power_of_two_plus_one_l2616_261673


namespace problem_1_l2616_261613

theorem problem_1 : 2 + (-5) - (-4) + |(-3)| = 4 := by sorry

end problem_1_l2616_261613


namespace intersection_of_A_and_B_l2616_261626

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1} := by
  sorry

end intersection_of_A_and_B_l2616_261626


namespace jake_initial_cats_jake_initial_cats_is_one_l2616_261646

/-- Proves that Jake initially had 1 cat given the conditions of the problem -/
theorem jake_initial_cats : ℝ → Prop :=
  fun initial_cats =>
    let food_per_cat : ℝ := 0.5
    let total_food_after : ℝ := 0.9
    let extra_food : ℝ := 0.4
    (initial_cats * food_per_cat + food_per_cat = total_food_after) ∧
    (food_per_cat = extra_food) →
    initial_cats = 1

/-- The theorem is true -/
theorem jake_initial_cats_is_one : jake_initial_cats 1 := by
  sorry

end jake_initial_cats_jake_initial_cats_is_one_l2616_261646


namespace sqrt_a_minus_b_is_natural_l2616_261633

theorem sqrt_a_minus_b_is_natural (a b : ℕ) (h : 2015 * a^2 + a = 2016 * b^2 + b) :
  ∃ k : ℕ, a - b = k^2 := by sorry

end sqrt_a_minus_b_is_natural_l2616_261633


namespace sqrt_two_plus_sqrt_l2616_261658

theorem sqrt_two_plus_sqrt : ∃ y : ℝ, y = Real.sqrt (2 + y) → y = 2 := by sorry

end sqrt_two_plus_sqrt_l2616_261658


namespace circumscribed_sphere_radius_rectangular_solid_l2616_261644

/-- For a rectangular solid with edges a, b, and c, the radius R of its circumscribed sphere
    satisfies the equation 4R² = a² + b² + c². -/
theorem circumscribed_sphere_radius_rectangular_solid
  (a b c R : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * R^2 = a^2 + b^2 + c^2 :=
sorry

end circumscribed_sphere_radius_rectangular_solid_l2616_261644


namespace smartphone_loss_percentage_l2616_261627

/-- Calculates the percentage loss when selling an item -/
def percentageLoss (initialCost sellPrice : ℚ) : ℚ :=
  (initialCost - sellPrice) / initialCost * 100

/-- Proves that selling a $300 item for $255 results in a 15% loss -/
theorem smartphone_loss_percentage :
  percentageLoss 300 255 = 15 := by
  sorry

end smartphone_loss_percentage_l2616_261627


namespace two_minus_i_in_fourth_quadrant_l2616_261657

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/-- The complex number 2 - i is in the fourth quadrant. -/
theorem two_minus_i_in_fourth_quadrant :
  in_fourth_quadrant (2 - I) := by
  sorry

end two_minus_i_in_fourth_quadrant_l2616_261657


namespace min_reciprocal_sum_l2616_261652

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 := by sorry

end min_reciprocal_sum_l2616_261652


namespace equal_roots_implies_k_eq_four_l2616_261616

/-- 
A quadratic equation ax^2 + bx + c = 0 has two equal real roots if and only if 
its discriminant b^2 - 4ac is equal to 0.
-/
def has_two_equal_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- 
Given a quadratic equation kx^2 - 2kx + 4 = 0 with two equal real roots,
prove that k = 4.
-/
theorem equal_roots_implies_k_eq_four :
  ∀ k : ℝ, k ≠ 0 → has_two_equal_real_roots k (-2*k) 4 → k = 4 :=
by sorry

end equal_roots_implies_k_eq_four_l2616_261616


namespace gcd_of_quadratic_and_linear_l2616_261606

theorem gcd_of_quadratic_and_linear (a : ℤ) (h : ∃ k : ℤ, a = 2100 * k) :
  Int.gcd (a^2 + 14*a + 49) (a + 7) = 7 := by
  sorry

end gcd_of_quadratic_and_linear_l2616_261606


namespace two_digit_number_theorem_l2616_261617

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem two_digit_number_theorem (x y : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ 
  (10 * x + y) - (10 * y + x) = 81 ∧ 
  is_prime (x + y) → 
  x - y = 7 := by sorry

end two_digit_number_theorem_l2616_261617


namespace emberly_walk_distance_l2616_261607

/-- Emberly's walking problem -/
theorem emberly_walk_distance :
  ∀ (total_days : ℕ) (days_not_walked : ℕ) (total_miles : ℕ),
    total_days = 31 →
    days_not_walked = 4 →
    total_miles = 108 →
    (total_miles : ℚ) / (total_days - days_not_walked : ℚ) = 4 := by
  sorry

end emberly_walk_distance_l2616_261607


namespace product_equals_120_l2616_261683

theorem product_equals_120 (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end product_equals_120_l2616_261683


namespace three_digit_factorial_sum_l2616_261672

theorem three_digit_factorial_sum : ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ 
  (0 ≤ y ∧ y ≤ 9) ∧ 
  (0 ≤ z ∧ z ≤ 9) ∧
  (100 * x + 10 * y + z = Nat.factorial x + Nat.factorial y + Nat.factorial z) ∧
  (x + y + z = 10) := by
  sorry

end three_digit_factorial_sum_l2616_261672


namespace same_distance_different_time_l2616_261663

/-- Proves that given Joann's average speed and time, Fran needs to ride at a specific speed to cover the same distance in less time -/
theorem same_distance_different_time (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 14)
  (h2 : joann_time = 4)
  (h3 : fran_time = 2) :
  joann_speed * joann_time = (joann_speed * joann_time / fran_time) * fran_time :=
by sorry

end same_distance_different_time_l2616_261663


namespace green_ball_fraction_l2616_261653

theorem green_ball_fraction (total : ℕ) (green blue yellow white : ℕ) :
  blue = total / 8 →
  yellow = total / 12 →
  white = 26 →
  blue = 6 →
  green + blue + yellow + white = total →
  green = total / 4 := by
  sorry

end green_ball_fraction_l2616_261653


namespace happy_valley_theorem_l2616_261609

/-- The number of ways to arrange animals in the Happy Valley Kennel -/
def happy_valley_arrangements : ℕ :=
  let num_chickens : ℕ := 3
  let num_dogs : ℕ := 4
  let num_cats : ℕ := 6
  let total_animals : ℕ := num_chickens + num_dogs + num_cats
  let group_arrangements : ℕ := 2  -- chicken-dog or dog-chicken around cats
  let chicken_arrangements : ℕ := Nat.factorial num_chickens
  let dog_arrangements : ℕ := Nat.factorial num_dogs
  let cat_arrangements : ℕ := Nat.factorial num_cats
  group_arrangements * chicken_arrangements * dog_arrangements * cat_arrangements

/-- Theorem stating the correct number of arrangements for the Happy Valley Kennel problem -/
theorem happy_valley_theorem : happy_valley_arrangements = 69120 := by
  sorry

end happy_valley_theorem_l2616_261609


namespace distinct_fm_pairs_count_l2616_261639

/-- Represents the gender of a person -/
inductive Gender
| Male
| Female

/-- Represents a seating arrangement of 5 people around a round table -/
def SeatingArrangement := Vector Gender 5

/-- Counts the number of people sitting next to at least one female -/
def count_next_to_female (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Counts the number of people sitting next to at least one male -/
def count_next_to_male (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Generates all distinct seating arrangements -/
def all_distinct_arrangements : List SeatingArrangement :=
  sorry

/-- The main theorem stating that there are exactly 8 distinct (f, m) pairs -/
theorem distinct_fm_pairs_count :
  (all_distinct_arrangements.map (λ arr => (count_next_to_female arr, count_next_to_male arr))).toFinset.card = 8 :=
  sorry

end distinct_fm_pairs_count_l2616_261639


namespace contrapositive_equivalence_l2616_261602

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) :=
by sorry

end contrapositive_equivalence_l2616_261602


namespace min_value_theorem_l2616_261665

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 4) :
  ∃ (m : ℝ), m = 16/7 ∧ ∀ (z : ℝ), z ≥ m ↔ z ≥ x^2/(x+1) + y^2/(y+2) := by
  sorry

end min_value_theorem_l2616_261665


namespace max_product_sum_l2616_261618

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({2, 3, 4, 5} : Set ℕ) → 
  b ∈ ({2, 3, 4, 5} : Set ℕ) → 
  c ∈ ({2, 3, 4, 5} : Set ℕ) → 
  d ∈ ({2, 3, 4, 5} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + b * c + c * d + d * a) ≤ 49 :=
by sorry

end max_product_sum_l2616_261618


namespace circle_equation_l2616_261667

/-- A circle C in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line 2x - y + 3 = 0 -/
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

/-- Circle C satisfies the given conditions -/
def satisfies_conditions (C : Circle) : Prop :=
  let (a, b) := C.center
  line a b ∧
  (1 - a)^2 + (3 - b)^2 = C.radius^2 ∧
  (3 - a)^2 + (5 - b)^2 = C.radius^2

/-- The standard equation of circle C -/
def standard_equation (C : Circle) (x y : ℝ) : Prop :=
  let (a, b) := C.center
  (x - a)^2 + (y - b)^2 = C.radius^2

theorem circle_equation :
  ∃ C : Circle, satisfies_conditions C ∧
    ∀ x y : ℝ, standard_equation C x y ↔ (x - 1)^2 + (y - 5)^2 = 4 :=
sorry

end circle_equation_l2616_261667


namespace total_flowers_collected_l2616_261623

/-- The maximum number of flowers each person can pick --/
def max_flowers : ℕ := 50

/-- The number of tulips Arwen picked --/
def arwen_tulips : ℕ := 20

/-- The number of roses Arwen picked --/
def arwen_roses : ℕ := 18

/-- The number of sunflowers Arwen picked --/
def arwen_sunflowers : ℕ := 6

/-- The number of tulips Elrond picked --/
def elrond_tulips : ℕ := 2 * arwen_tulips

/-- The number of roses Elrond picked --/
def elrond_roses : ℕ := min (3 * arwen_roses) (max_flowers - elrond_tulips)

/-- The number of tulips Galadriel picked --/
def galadriel_tulips : ℕ := min (3 * elrond_tulips) max_flowers

/-- The number of roses Galadriel picked --/
def galadriel_roses : ℕ := min (2 * arwen_roses) (max_flowers - galadriel_tulips)

/-- The number of sunflowers Legolas picked --/
def legolas_sunflowers : ℕ := arwen_sunflowers

/-- The number of roses Legolas picked --/
def legolas_roses : ℕ := (max_flowers - legolas_sunflowers) / 2

/-- The number of tulips Legolas picked --/
def legolas_tulips : ℕ := (max_flowers - legolas_sunflowers) / 2

theorem total_flowers_collected :
  arwen_tulips + arwen_roses + arwen_sunflowers +
  elrond_tulips + elrond_roses +
  galadriel_tulips + galadriel_roses +
  legolas_sunflowers + legolas_roses + legolas_tulips = 194 := by
  sorry

#eval arwen_tulips + arwen_roses + arwen_sunflowers +
  elrond_tulips + elrond_roses +
  galadriel_tulips + galadriel_roses +
  legolas_sunflowers + legolas_roses + legolas_tulips

end total_flowers_collected_l2616_261623


namespace investment_problem_l2616_261604

/-- Given two investors P and Q, where the profit is divided in the ratio 3:5
    and P invested 12000, prove that Q invested 20000. -/
theorem investment_problem (P Q : ℕ) (profit_ratio : ℚ) (P_investment : ℕ) :
  profit_ratio = 3 / 5 →
  P_investment = 12000 →
  Q = 20000 :=
by sorry

end investment_problem_l2616_261604


namespace slower_walk_delay_l2616_261610

/-- Proves that walking at 4/5 of the usual speed results in a 6-minute delay -/
theorem slower_walk_delay (usual_time : ℝ) (h : usual_time = 24) : 
  let slower_time := usual_time / (4/5)
  slower_time - usual_time = 6 := by
  sorry

#check slower_walk_delay

end slower_walk_delay_l2616_261610


namespace triangle_theorem_l2616_261601

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) 
  (ha : t.a = 2)
  (hc : t.c = 3)
  (hcosB : Real.cos t.angleB = 1/4) :
  t.b = Real.sqrt 10 ∧ Real.sin (2 * t.angleC) = (3 * Real.sqrt 15) / 16 := by
  sorry


end triangle_theorem_l2616_261601


namespace highest_salary_grade_is_six_l2616_261675

/-- The minimum salary grade -/
def min_grade : ℕ := 1

/-- Function to calculate hourly wage based on salary grade -/
def hourly_wage (s : ℕ) : ℝ := 7.50 + 0.25 * (s - 1)

/-- The difference in hourly wage between the highest and lowest grade -/
def wage_difference : ℝ := 1.25

theorem highest_salary_grade_is_six :
  ∃ (max_grade : ℕ),
    (∀ (s : ℕ), min_grade ≤ s ∧ s ≤ max_grade) ∧
    (hourly_wage max_grade = hourly_wage min_grade + wage_difference) ∧
    max_grade = 6 :=
by sorry

end highest_salary_grade_is_six_l2616_261675
