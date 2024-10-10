import Mathlib

namespace union_equals_reals_l1115_111546

def S : Set ℝ := {x | x < -1 ∨ x > 5}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

theorem union_equals_reals (a : ℝ) : 
  S ∪ T a = Set.univ ↔ -3 < a ∧ a < -1 := by sorry

end union_equals_reals_l1115_111546


namespace katherine_age_when_mel_21_l1115_111520

/-- Katherine's age when Mel is 21 years old -/
def katherines_age (mels_age : ℕ) (age_difference : ℕ) : ℕ :=
  mels_age + age_difference

/-- Theorem stating Katherine's age when Mel is 21 -/
theorem katherine_age_when_mel_21 :
  katherines_age 21 3 = 24 := by
  sorry

end katherine_age_when_mel_21_l1115_111520


namespace hash_2_3_4_l1115_111541

/-- The # operation defined on real numbers -/
def hash (a b c : ℝ) : ℝ := b^2 - 3*a*c

/-- Theorem stating that hash(2, 3, 4) = -15 -/
theorem hash_2_3_4 : hash 2 3 4 = -15 := by
  sorry

end hash_2_3_4_l1115_111541


namespace at_least_two_first_grade_products_l1115_111581

theorem at_least_two_first_grade_products (total : Nat) (first_grade : Nat) (second_grade : Nat) (third_grade : Nat) (drawn : Nat) 
  (h1 : total = 9)
  (h2 : first_grade = 4)
  (h3 : second_grade = 3)
  (h4 : third_grade = 2)
  (h5 : drawn = 4)
  (h6 : total = first_grade + second_grade + third_grade) :
  (Nat.choose total drawn) - (Nat.choose (second_grade + third_grade) drawn) - 
  (Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (drawn - 1)) = 81 := by
sorry

end at_least_two_first_grade_products_l1115_111581


namespace y_derivative_l1115_111514

noncomputable def y (x : ℝ) : ℝ := 3 * (Real.sin x / Real.cos x ^ 2) + 2 * (Real.sin x / Real.cos x ^ 4)

theorem y_derivative (x : ℝ) :
  deriv y x = (3 + 3 * Real.sin x ^ 2) / Real.cos x ^ 3 + (2 - 6 * Real.sin x ^ 2) / Real.cos x ^ 5 :=
by sorry

end y_derivative_l1115_111514


namespace correct_calculation_l1115_111586

theorem correct_calculation (original : ℤ) (incorrect_result : ℤ) 
  (incorrect_addition : ℤ) (correct_addition : ℤ) : 
  incorrect_result = original + incorrect_addition →
  original + correct_addition = 97 :=
by
  intro h
  sorry

#check correct_calculation 35 61 26 62

end correct_calculation_l1115_111586


namespace colored_paper_distribution_l1115_111562

/-- Proves that each female student receives 6 sheets of colored paper given the problem conditions -/
theorem colored_paper_distribution (total_students : ℕ) (total_paper : ℕ) (leftover : ℕ) :
  total_students = 24 →
  total_paper = 50 →
  leftover = 2 →
  ∃ (female_students : ℕ) (male_students : ℕ),
    female_students + male_students = total_students ∧
    male_students = 2 * female_students ∧
    (total_paper - leftover) % female_students = 0 ∧
    (total_paper - leftover) / female_students = 6 :=
by sorry

end colored_paper_distribution_l1115_111562


namespace cheburashka_krakozyabra_relation_num_cheburashkas_is_eleven_l1115_111595

/-- Represents the number of Cheburashkas in Katya's drawing -/
def num_cheburashkas : ℕ := 11

/-- Represents the total number of Krakozyabras in the final drawing -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of rows in Katya's drawing -/
def num_rows : ℕ := 2

/-- Theorem stating the relationship between Cheburashkas and Krakozyabras -/
theorem cheburashka_krakozyabra_relation :
  num_cheburashkas = (total_krakozyabras + num_rows) / 2 := by
  sorry

/-- Theorem proving that the number of Cheburashkas is 11 -/
theorem num_cheburashkas_is_eleven :
  num_cheburashkas = 11 := by
  sorry

end cheburashka_krakozyabra_relation_num_cheburashkas_is_eleven_l1115_111595


namespace bike_rental_problem_l1115_111532

/-- Calculates the number of hours a bike was rented given the total payment, fixed fee, and hourly rate. -/
def rentedHours (totalPayment fixedFee hourlyRate : ℚ) : ℚ :=
  (totalPayment - fixedFee) / hourlyRate

theorem bike_rental_problem :
  let totalPayment : ℚ := 80
  let fixedFee : ℚ := 17
  let hourlyRate : ℚ := 7
  rentedHours totalPayment fixedFee hourlyRate = 9 := by
sorry

#eval rentedHours 80 17 7

end bike_rental_problem_l1115_111532


namespace complex_equation_solution_l1115_111509

theorem complex_equation_solution (z : ℂ) : (1 - 3*I)*z = 2 + 4*I → z = -1 + I := by
  sorry

end complex_equation_solution_l1115_111509


namespace exists_same_color_configuration_l1115_111552

/-- A color type with two possible values -/
inductive Color
| Red
| Blue

/-- A grid of cells with colors -/
def Grid := Fin 5 → Fin 41 → Color

/-- A configuration of three rows and three columns -/
structure Configuration where
  rows : Fin 3 → Fin 5
  cols : Fin 3 → Fin 41

/-- Check if a configuration has all intersections of the same color -/
def Configuration.allSameColor (grid : Grid) (config : Configuration) : Prop :=
  ∃ c : Color, ∀ i j : Fin 3, grid (config.rows i) (config.cols j) = c

/-- Main theorem: There exists a configuration with all intersections of the same color -/
theorem exists_same_color_configuration (grid : Grid) :
  ∃ config : Configuration, config.allSameColor grid := by
  sorry


end exists_same_color_configuration_l1115_111552


namespace max_value_m_plus_n_l1115_111539

theorem max_value_m_plus_n (a b m n : ℝ) : 
  (a < 0 ∧ b < 0) →  -- a and b have the same sign (negative)
  (∀ x, ax^2 + 2*x + b < 0 ↔ x ≠ -1/a) →  -- solution set condition
  m = b + 1/a →  -- definition of m
  n = a + 1/b →  -- definition of n
  (∀ k, m + n ≤ k) → k = -4 :=
by sorry

end max_value_m_plus_n_l1115_111539


namespace angle_sum_theorem_l1115_111504

theorem angle_sum_theorem (θ φ : Real) (h1 : 0 < θ ∧ θ < π/2) (h2 : 0 < φ ∧ φ < π/2)
  (h3 : Real.tan θ = 2/5) (h4 : Real.cos φ = 1/2) :
  2 * θ + φ = π/4 := by
sorry

end angle_sum_theorem_l1115_111504


namespace weed_pulling_l1115_111585

theorem weed_pulling (day1 : ℕ) : 
  let day2 := 3 * day1
  let day3 := day2 / 5
  let day4 := day3 - 10
  day1 + day2 + day3 + day4 = 120 →
  day1 = 25 := by
sorry

end weed_pulling_l1115_111585


namespace jack_combinations_eq_44_l1115_111582

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of combinations of rolls Jack could purchase. -/
def jack_combinations : ℕ := distribute 10 4

theorem jack_combinations_eq_44 : jack_combinations = 44 := by sorry

end jack_combinations_eq_44_l1115_111582


namespace tims_lunch_cost_l1115_111575

theorem tims_lunch_cost (tip_percentage : ℝ) (total_spent : ℝ) (lunch_cost : ℝ) : 
  tip_percentage = 0.20 → 
  total_spent = 72.6 → 
  lunch_cost * (1 + tip_percentage) = total_spent → 
  lunch_cost = 60.5 := by
sorry

end tims_lunch_cost_l1115_111575


namespace debby_water_bottles_l1115_111537

/-- The number of water bottles Debby drinks per day -/
def bottles_per_day : ℕ := 6

/-- The number of days the water bottles would last -/
def days_lasting : ℕ := 2

/-- The number of water bottles Debby bought -/
def bottles_bought : ℕ := bottles_per_day * days_lasting

theorem debby_water_bottles : bottles_bought = 12 := by
  sorry

end debby_water_bottles_l1115_111537


namespace prob_A_B_together_is_two_thirds_l1115_111583

/-- The number of ways to arrange 3 students in a row -/
def total_arrangements : ℕ := 6

/-- The number of arrangements where A and B are together -/
def favorable_arrangements : ℕ := 4

/-- The probability that A and B stand together -/
def prob_A_B_together : ℚ := favorable_arrangements / total_arrangements

theorem prob_A_B_together_is_two_thirds : 
  prob_A_B_together = 2/3 := by sorry

end prob_A_B_together_is_two_thirds_l1115_111583


namespace function_inequality_l1115_111580

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, (x - 1) * (deriv (deriv f) x) ≤ 0) :
  f 0 + f 2 ≤ 2 * f 1 := by
  sorry

end function_inequality_l1115_111580


namespace common_root_quadratics_l1115_111547

theorem common_root_quadratics (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ a = -2 := by
  sorry

end common_root_quadratics_l1115_111547


namespace homework_students_l1115_111589

theorem homework_students (total : ℕ) (silent_reading : ℚ) (board_games : ℚ) (group_discussions : ℚ)
  (h_total : total = 120)
  (h_silent : silent_reading = 2 / 5)
  (h_board : board_games = 3 / 10)
  (h_group : group_discussions = 1 / 8) :
  total - (silent_reading * total + board_games * total + group_discussions * total).floor = 21 :=
by sorry

end homework_students_l1115_111589


namespace gcd_problem_l1115_111505

theorem gcd_problem (p : Nat) (h : Nat.Prime p) (hp : p = 107) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 := by
  sorry

end gcd_problem_l1115_111505


namespace kelly_initial_bracelets_l1115_111506

/-- Proves that Kelly initially had 16 bracelets given the problem conditions -/
theorem kelly_initial_bracelets :
  ∀ (k : ℕ), -- k represents Kelly's initial number of bracelets
  let b_initial : ℕ := 5 -- Bingley's initial number of bracelets
  let b_after_kelly : ℕ := b_initial + k / 4 -- Bingley's bracelets after receiving from Kelly
  let b_final : ℕ := b_after_kelly * 2 / 3 -- Bingley's final number of bracelets
  b_final = 6 → k = 16 := by
  sorry

end kelly_initial_bracelets_l1115_111506


namespace quadratic_roots_problem_l1115_111542

theorem quadratic_roots_problem (x y : ℝ) : 
  x + y = 6 → 
  |x - y| = 8 → 
  x^2 - 6*x - 7 = 0 ∧ y^2 - 6*y - 7 = 0 := by
sorry

end quadratic_roots_problem_l1115_111542


namespace combination_arrangement_equality_l1115_111535

theorem combination_arrangement_equality (m : ℕ) : (Nat.choose m 3) = (m * (m - 1)) → m = 8 := by
  sorry

end combination_arrangement_equality_l1115_111535


namespace triangle_area_l1115_111556

/-- Given a triangle with perimeter 35 cm and inradius 4.5 cm, its area is 78.75 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 35 → inradius = 4.5 → area = perimeter / 2 * inradius → area = 78.75 := by
  sorry

end triangle_area_l1115_111556


namespace hiking_problem_solution_l1115_111563

/-- Represents the hiking problem with given speeds and distances -/
structure HikingProblem where
  total_time : ℚ  -- in hours
  total_distance : ℚ  -- in km
  uphill_speed : ℚ  -- in km/h
  flat_speed : ℚ  -- in km/h
  downhill_speed : ℚ  -- in km/h

/-- Theorem stating the solution to the hiking problem -/
theorem hiking_problem_solution (p : HikingProblem) 
  (h1 : p.total_time = 221 / 60)  -- 3 hours and 41 minutes in decimal form
  (h2 : p.total_distance = 9)
  (h3 : p.uphill_speed = 4)
  (h4 : p.flat_speed = 5)
  (h5 : p.downhill_speed = 6) :
  ∃ (x : ℚ), x = 4 ∧ 
    (2 * x / p.flat_speed + 
     (5 * (p.total_distance - x)) / (12 : ℚ) = p.total_time) := by
  sorry


end hiking_problem_solution_l1115_111563


namespace trigonometric_identity_l1115_111573

theorem trigonometric_identity (α : ℝ) (h : Real.sin α + Real.cos α = 2/3) :
  (2 * Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) / (1 + Real.tan α) = -5/9 := by
  sorry

end trigonometric_identity_l1115_111573


namespace dollars_to_dozen_quarters_l1115_111590

theorem dollars_to_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (items_per_dozen : ℕ) :
  dollars = 9 →
  quarters_per_dollar = 4 →
  items_per_dozen = 12 →
  (dollars * quarters_per_dollar) / items_per_dozen = 3 :=
by sorry

end dollars_to_dozen_quarters_l1115_111590


namespace complement_intersection_A_B_l1115_111540

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_A_B :
  (A ∩ B)ᶜ = {1, 4, 5} := by
  sorry

end complement_intersection_A_B_l1115_111540


namespace fish_population_estimate_l1115_111554

theorem fish_population_estimate (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 60 →
  second_catch = 60 →
  tagged_in_second = 2 →
  (initial_tagged : ℚ) / (second_catch : ℚ) = (tagged_in_second : ℚ) / (initial_tagged : ℚ) →
  (initial_tagged * second_catch : ℚ) / tagged_in_second = 1800 :=
by
  sorry

#check fish_population_estimate

end fish_population_estimate_l1115_111554


namespace solution_range_l1115_111511

theorem solution_range (k : ℝ) : 
  (∃ x : ℝ, x + k = 2 * x - 1 ∧ x < 0) → k < -1 := by
  sorry

end solution_range_l1115_111511


namespace equal_roots_quadratic_l1115_111570

/-- The equation (2kx^2 + 4kx + 2) = 0 has equal roots when k = 1 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + 2 = 0) → 
  (∃! r : ℝ, 2 * x^2 + 4 * x + 2 = 0) := by
sorry

end equal_roots_quadratic_l1115_111570


namespace segments_form_triangle_l1115_111564

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three given lengths can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The line segments 4cm, 5cm, and 6cm can form a triangle. -/
theorem segments_form_triangle : can_form_triangle 4 5 6 := by
  sorry

end segments_form_triangle_l1115_111564


namespace coefficient_a4b3c2_in_expansion_l1115_111549

theorem coefficient_a4b3c2_in_expansion (a b c : ℕ) : 
  (Nat.choose 9 5) * (Nat.choose 5 2) = 1260 := by sorry

end coefficient_a4b3c2_in_expansion_l1115_111549


namespace arithmetic_sequence_k_value_l1115_111503

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_k_value 
  (seq : ArithmeticSequence) 
  (k : ℕ) 
  (h1 : seq.S (k - 2) = -4)
  (h2 : seq.S k = 0)
  (h3 : seq.S (k + 2) = 8) :
  k = 6 := by
  sorry

end arithmetic_sequence_k_value_l1115_111503


namespace simple_interest_rate_example_l1115_111569

/-- Calculate the simple interest rate given principal, amount, and time -/
def simple_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  let simple_interest := amount - principal
  (simple_interest * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is approximately 9.23% -/
theorem simple_interest_rate_example :
  let principal := 650
  let amount := 950
  let time := 5
  let rate := simple_interest_rate principal amount time
  (rate ≥ 9.23) ∧ (rate < 9.24) :=
by sorry

end simple_interest_rate_example_l1115_111569


namespace sqrt_neg_five_squared_l1115_111597

theorem sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end sqrt_neg_five_squared_l1115_111597


namespace exact_five_blue_probability_l1115_111571

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_five_blue_probability :
  (Nat.choose total_draws blue_draws : ℚ) *
  (probability_blue ^ blue_draws) *
  (probability_red ^ (total_draws - blue_draws)) =
  (56 : ℚ) * 32 / 6561 :=
sorry

end exact_five_blue_probability_l1115_111571


namespace concentric_circles_radii_difference_l1115_111501

theorem concentric_circles_radii_difference (r : ℝ) (h : r > 0) :
  let R := (4 * r ^ 2) ^ (1 / 2 : ℝ)
  R - r = r :=
by sorry

end concentric_circles_radii_difference_l1115_111501


namespace common_tangents_from_guiding_circles_l1115_111500

/-- Represents an ellipse with its foci and semi-major axis -/
structure Ellipse where
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  semiMajorAxis : ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Enum representing the possible number of common tangents -/
inductive NumCommonTangents
  | zero
  | one
  | two

/-- Function to determine the number of intersections between two circles -/
def circleIntersections (c1 c2 : Circle) : NumCommonTangents :=
  sorry

/-- Function to get the guiding circle of an ellipse for a given focus -/
def guidingCircle (e : Ellipse) (f : ℝ × ℝ) : Circle :=
  sorry

/-- Theorem stating that the number of common tangents between two ellipses
    sharing a focus is determined by the intersection of their guiding circles -/
theorem common_tangents_from_guiding_circles 
  (e1 e2 : Ellipse) 
  (h : e1.focus1 = e2.focus1) :
  ∃ (f : ℝ × ℝ), 
    let c1 := guidingCircle e1 f
    let c2 := guidingCircle e2 f
    circleIntersections c1 c2 = NumCommonTangents.zero ∨
    circleIntersections c1 c2 = NumCommonTangents.one ∨
    circleIntersections c1 c2 = NumCommonTangents.two :=
  sorry

end common_tangents_from_guiding_circles_l1115_111500


namespace symmetric_point_theorem_l1115_111522

/-- The symmetric point of P(-2, 1) with respect to the line y = x + 1 is (0, -1) -/
theorem symmetric_point_theorem : 
  let P : ℝ × ℝ := (-2, 1)
  let line (x y : ℝ) : Prop := y = x + 1
  let is_symmetric (P Q : ℝ × ℝ) : Prop := 
    let midpoint := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
    line midpoint.1 midpoint.2 ∧ 
    (Q.2 - P.2) * (Q.1 - P.1) = -1  -- Perpendicular condition
  ∃ Q : ℝ × ℝ, is_symmetric P Q ∧ Q = (0, -1) :=
by sorry

end symmetric_point_theorem_l1115_111522


namespace fraction_sum_l1115_111518

theorem fraction_sum (x : ℝ) (h1 : x ≠ 1) (h2 : 2*x ≠ -3) (h3 : 2*x^2 + 5*x - 3 ≠ 0) : 
  (6*x - 8) / (2*x^2 + 5*x - 3) = (-2/5) / (x - 1) + (34/5) / (2*x + 3) := by
sorry

end fraction_sum_l1115_111518


namespace triangle_angle_measure_l1115_111567

theorem triangle_angle_measure (P Q R : ℝ) : 
  P = 88 → 
  Q = 2 * R + 18 → 
  P + Q + R = 180 → 
  R = 74 / 3 := by
sorry

end triangle_angle_measure_l1115_111567


namespace cube_edge_increase_l1115_111524

theorem cube_edge_increase (e : ℝ) (f : ℝ) (h : e > 0) : (f * e)^3 = 8 * e^3 → f = 2 := by
  sorry

end cube_edge_increase_l1115_111524


namespace positive_reals_inequality_l1115_111510

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y^3 + y ≤ x - x^3) : y < x ∧ x < 1 ∧ x^2 + y^2 < 1 := by
  sorry

end positive_reals_inequality_l1115_111510


namespace valid_combinations_count_l1115_111526

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of vowels -/
def vowel_count : ℕ := 5

/-- The number of valid four-letter word combinations where the word begins and ends with the same letter, and the second letter is a vowel -/
def valid_combinations : ℕ := alphabet_size * vowel_count * alphabet_size

theorem valid_combinations_count : valid_combinations = 3380 := by
  sorry

end valid_combinations_count_l1115_111526


namespace probability_of_s_in_statistics_l1115_111515

def word : String := "statistics"

def count_letter (w : String) (c : Char) : Nat :=
  w.toList.filter (· = c) |>.length

theorem probability_of_s_in_statistics :
  (count_letter word 's' : ℚ) / word.length = 3 / 10 := by
  sorry

end probability_of_s_in_statistics_l1115_111515


namespace arthurs_wallet_l1115_111578

theorem arthurs_wallet (initial_amount : ℝ) : 
  (1 / 5 : ℝ) * initial_amount = 40 → initial_amount = 200 := by
  sorry

end arthurs_wallet_l1115_111578


namespace youtube_views_problem_l1115_111593

/-- Calculates the additional views after the fourth day given the initial views,
    increase factor, and total views after 6 days. -/
def additional_views_after_fourth_day (initial_views : ℕ) (increase_factor : ℕ) (total_views_after_six_days : ℕ) : ℕ :=
  total_views_after_six_days - (initial_views + increase_factor * initial_views)

/-- Theorem stating that given the specific conditions of the problem,
    the additional views after the fourth day is 50000. -/
theorem youtube_views_problem :
  additional_views_after_fourth_day 4000 10 94000 = 50000 := by
  sorry

end youtube_views_problem_l1115_111593


namespace baseball_team_groups_l1115_111543

theorem baseball_team_groups (new_players returning_players players_per_group : ℕ) 
  (h1 : new_players = 4)
  (h2 : returning_players = 6)
  (h3 : players_per_group = 5) :
  (new_players + returning_players) / players_per_group = 2 := by
  sorry

end baseball_team_groups_l1115_111543


namespace line_slope_l1115_111528

theorem line_slope (x y : ℝ) : 
  x - Real.sqrt 3 * y + 3 = 0 → 
  (y - Real.sqrt 3) / (x - (-3)) = Real.sqrt 3 / 3 := by
sorry

end line_slope_l1115_111528


namespace polynomial_roots_l1115_111508

theorem polynomial_roots : 
  let f : ℂ → ℂ := λ x => 3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3
  let r1 := (-1 + Real.sqrt (-171 + 12 * Real.sqrt 43)) / 6
  let r2 := (-1 - Real.sqrt (-171 + 12 * Real.sqrt 43)) / 6
  let r3 := (-1 + Real.sqrt (-171 - 12 * Real.sqrt 43)) / 6
  let r4 := (-1 - Real.sqrt (-171 - 12 * Real.sqrt 43)) / 6
  (f r1 = 0) ∧ (f r2 = 0) ∧ (f r3 = 0) ∧ (f r4 = 0) :=
by sorry

end polynomial_roots_l1115_111508


namespace village_revenue_comparison_l1115_111502

def village_a : List ℝ := [5, 6, 6, 7, 8, 16]
def village_b : List ℝ := [4, 6, 8, 9, 10, 17]

theorem village_revenue_comparison :
  (village_a.sum / village_a.length) < (village_b.sum / village_b.length) := by
  sorry

end village_revenue_comparison_l1115_111502


namespace stock_exchange_problem_l1115_111584

/-- The number of stocks that closed higher today -/
def higher_stocks : ℕ := 1080

/-- The number of stocks that closed lower today -/
def lower_stocks : ℕ := 900

/-- The total number of stocks on the stock exchange -/
def total_stocks : ℕ := higher_stocks + lower_stocks

theorem stock_exchange_problem :
  (higher_stocks = lower_stocks * 120 / 100) →
  (total_stocks = 1980) := by
  sorry

end stock_exchange_problem_l1115_111584


namespace chocolate_probabilities_l1115_111560

theorem chocolate_probabilities (w1 n1 w2 n2 : ℕ) 
  (h1 : w1 ≤ n1) (h2 : w2 ≤ n2) (h3 : n1 > 0) (h4 : n2 > 0) :
  ∃ (w1' n1' w2' n2' : ℕ),
    w1' ≤ n1' ∧ w2' ≤ n2' ∧ n1' > 0 ∧ n2' > 0 ∧
    (w1' : ℚ) / n1' = (w1 + w2 : ℚ) / (n1 + n2) ∧
  ∃ (w1'' n1'' w2'' n2'' : ℕ),
    w1'' ≤ n1'' ∧ w2'' ≤ n2'' ∧ n1'' > 0 ∧ n2'' > 0 ∧
    ¬((w1'' : ℚ) / n1'' < (w1'' + w2'' : ℚ) / (n1'' + n2'') ∧
      (w1'' + w2'' : ℚ) / (n1'' + n2'') < (w2'' : ℚ) / n2'') :=
by sorry

end chocolate_probabilities_l1115_111560


namespace cube_sum_reciprocal_l1115_111507

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 := by
  sorry

end cube_sum_reciprocal_l1115_111507


namespace wide_flags_count_l1115_111579

/-- Represents the flag-making scenario with given parameters -/
structure FlagScenario where
  totalFabric : ℕ
  squareFlagSide : ℕ
  wideRectFlagWidth : ℕ
  wideRectFlagHeight : ℕ
  tallRectFlagWidth : ℕ
  tallRectFlagHeight : ℕ
  squareFlagsMade : ℕ
  tallFlagsMade : ℕ
  fabricLeft : ℕ

/-- Calculates the number of wide rectangular flags made -/
def wideFlagsMade (scenario : FlagScenario) : ℕ :=
  let squareFlagArea := scenario.squareFlagSide * scenario.squareFlagSide
  let wideFlagArea := scenario.wideRectFlagWidth * scenario.wideRectFlagHeight
  let tallFlagArea := scenario.tallRectFlagWidth * scenario.tallRectFlagHeight
  let usedFabric := scenario.totalFabric - scenario.fabricLeft
  let squareAndTallFlagsArea := scenario.squareFlagsMade * squareFlagArea + scenario.tallFlagsMade * tallFlagArea
  let wideFlagsArea := usedFabric - squareAndTallFlagsArea
  wideFlagsArea / wideFlagArea

/-- Theorem stating that the number of wide flags made is 20 -/
theorem wide_flags_count (scenario : FlagScenario) 
  (h1 : scenario.totalFabric = 1000)
  (h2 : scenario.squareFlagSide = 4)
  (h3 : scenario.wideRectFlagWidth = 5)
  (h4 : scenario.wideRectFlagHeight = 3)
  (h5 : scenario.tallRectFlagWidth = 3)
  (h6 : scenario.tallRectFlagHeight = 5)
  (h7 : scenario.squareFlagsMade = 16)
  (h8 : scenario.tallFlagsMade = 10)
  (h9 : scenario.fabricLeft = 294) :
  wideFlagsMade scenario = 20 := by
  sorry


end wide_flags_count_l1115_111579


namespace circumcircle_tangency_l1115_111576

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary functions and relations
variable (circumcircle : Point → Point → Point → Circle)
variable (on_arc : Point → Point → Point → Point → Prop)
variable (incircle_center : Point → Point → Point → Point)
variable (touches : Circle → Circle → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem circumcircle_tangency
  (A B C D I_A I_B : Point) (k : Circle) :
  k = circumcircle A B C →
  on_arc A B C D →
  I_A = incircle_center A D C →
  I_B = incircle_center B D C →
  touches (circumcircle I_A I_B C) k ↔
    distance A D / distance B D =
    (distance A C + distance C D) / (distance B C + distance C D) :=
sorry

end circumcircle_tangency_l1115_111576


namespace range_of_m_l1115_111513

theorem range_of_m (x y m : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 2 3, y^2 - x*y - m*x^2 ≤ 0) →
  m ∈ Set.Ioi 6 :=
sorry

end range_of_m_l1115_111513


namespace quadratic_inequality_quadratic_inequality_negative_m_l1115_111545

theorem quadratic_inequality (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (1 - m) * x + m - 2 ≥ -2) ↔ m ≥ 1/3 := by sorry

theorem quadratic_inequality_negative_m (m : ℝ) (hm : m < 0) :
  (∀ x : ℝ, m * x^2 + (1 - m) * x + m - 2 < m - 1) ↔
  ((m ≤ -1 ∧ (x < -1/m ∨ x > 1)) ∨
   (-1 < m ∧ m < 0 ∧ (x < 1 ∨ x > -1/m))) := by sorry

end quadratic_inequality_quadratic_inequality_negative_m_l1115_111545


namespace isosceles_triangle_base_angle_l1115_111531

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  -- The triangle is isosceles
  (α = β ∨ β = γ ∨ γ = α) →
  -- The sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- One angle is 80°
  (α = 80 ∨ β = 80 ∨ γ = 80) →
  -- The base angle is either 50° or 80°
  (α = 50 ∨ α = 80 ∨ β = 50 ∨ β = 80 ∨ γ = 50 ∨ γ = 80) :=
by sorry

end isosceles_triangle_base_angle_l1115_111531


namespace trapezoid_area_in_regular_hexagon_l1115_111512

/-- The area of a trapezoid formed by connecting midpoints of non-adjacent sides in a regular hexagon -/
theorem trapezoid_area_in_regular_hexagon (side_length : ℝ) (h : side_length = 12) :
  let height := side_length * Real.sqrt 3 / 2
  let trapezoid_base := side_length / 2
  let trapezoid_area := (trapezoid_base + trapezoid_base) * height / 2
  trapezoid_area = 36 * Real.sqrt 3 := by
  sorry

end trapezoid_area_in_regular_hexagon_l1115_111512


namespace x_eq_2_sufficient_not_necessary_l1115_111516

-- Define the equation
def equation (x : ℝ) : Prop := (x - 2) * (x + 5) = 0

-- Define sufficient condition
def sufficient (p q : Prop) : Prop := p → q

-- Define necessary condition
def necessary (p q : Prop) : Prop := q → p

-- Theorem statement
theorem x_eq_2_sufficient_not_necessary :
  (sufficient (x = 2) (equation x)) ∧ ¬(necessary (x = 2) (equation x)) :=
sorry

end x_eq_2_sufficient_not_necessary_l1115_111516


namespace gcf_36_60_90_l1115_111559

theorem gcf_36_60_90 : Nat.gcd 36 (Nat.gcd 60 90) = 6 := by
  sorry

end gcf_36_60_90_l1115_111559


namespace equation_system_ratio_l1115_111551

theorem equation_system_ratio (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y) / (y^2 + z^2) = 7 := by
sorry

end equation_system_ratio_l1115_111551


namespace cylinder_max_volume_l1115_111592

/-- The maximum volume of a cylinder with total surface area 1 is achieved when 
    the radius and height are both equal to 1/√(6π) -/
theorem cylinder_max_volume (r h : ℝ) :
  r > 0 ∧ h > 0 ∧ 2 * Real.pi * r^2 + 2 * Real.pi * r * h = 1 →
  Real.pi * r^2 * h ≤ Real.pi * (1 / Real.sqrt (6 * Real.pi))^2 * (1 / Real.sqrt (6 * Real.pi)) :=
sorry

end cylinder_max_volume_l1115_111592


namespace parabolas_sum_l1115_111566

/-- Given two parabolas that intersect the coordinate axes at four points forming a rhombus -/
structure Parabolas where
  a : ℝ
  b : ℝ
  parabola1 : ℝ → ℝ
  parabola2 : ℝ → ℝ
  h_parabola1 : ∀ x, parabola1 x = a * x^2 - 2
  h_parabola2 : ∀ x, parabola2 x = 6 - b * x^2
  h_rhombus : ∃ x1 x2 y1 y2, 
    parabola1 x1 = 0 ∧ parabola1 x2 = 0 ∧
    parabola2 0 = y1 ∧ parabola2 0 = y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2
  h_area : (x2 - x1) * (y2 - y1) = 24
  h_b_eq_2a : b = 2 * a

/-- The sum of a and b is 6 -/
theorem parabolas_sum (p : Parabolas) : p.a + p.b = 6 := by
  sorry

end parabolas_sum_l1115_111566


namespace smallest_integer_satisfying_inequality_l1115_111529

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, y < 3 * y - 15 → y ≥ 8 :=
by
  sorry

end smallest_integer_satisfying_inequality_l1115_111529


namespace tempo_original_value_l1115_111577

/-- The original value of the tempo -/
def original_value : ℝ := 11083.33

/-- The insurance coverage percentage for all three years -/
def coverage_percentage : ℚ := 5/7

/-- The premium rate for the first year -/
def premium_rate_year1 : ℚ := 3/100

/-- The premium rate for the second year -/
def premium_rate_year2 : ℚ := 4/100

/-- The premium rate for the third year -/
def premium_rate_year3 : ℚ := 5/100

/-- The total premium paid for all three years -/
def total_premium : ℝ := 950

/-- Theorem stating that the original value of the tempo satisfies the given conditions -/
theorem tempo_original_value :
  (coverage_percentage * premium_rate_year1 * original_value +
   coverage_percentage * premium_rate_year2 * original_value +
   coverage_percentage * premium_rate_year3 * original_value) = total_premium := by
  sorry

end tempo_original_value_l1115_111577


namespace jenna_stamps_problem_l1115_111596

theorem jenna_stamps_problem (a b c : ℕ) 
  (ha : a = 924) (hb : b = 1260) (hc : c = 1386) : 
  Nat.gcd a (Nat.gcd b c) = 42 := by
  sorry

end jenna_stamps_problem_l1115_111596


namespace min_value_theorem_l1115_111538

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x) + (4 / (y + 1)) ≥ 9 / 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ (1 / x₀) + (4 / (y₀ + 1)) = 9 / 2 :=
by sorry

end min_value_theorem_l1115_111538


namespace not_prime_fourth_power_minus_four_l1115_111561

theorem not_prime_fourth_power_minus_four (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ¬∃ q : ℕ, Nat.Prime q ∧ p - 4 = q^4 := by
  sorry

end not_prime_fourth_power_minus_four_l1115_111561


namespace exists_diameter_points_l1115_111598

/-- A circle divided into 3k arcs by 3k points -/
structure CircleDivision (k : ℕ) where
  points : Fin (3 * k) → ℝ × ℝ
  is_on_circle : ∀ i, (points i).1^2 + (points i).2^2 = 1
  arc_lengths : Fin (3 * k) → ℝ
  unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 1
  double_unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 2
  triple_unit_arcs : ∃ (S : Finset (Fin (3 * k))), S.card = k ∧ ∀ i ∈ S, arc_lengths i = 3
  total_length : (Finset.univ.sum arc_lengths) = 2 * Real.pi

/-- Two points determine a diameter if they are opposite each other on the circle -/
def is_diameter {k : ℕ} (cd : CircleDivision k) (i j : Fin (3 * k)) : Prop :=
  (cd.points i).1 = -(cd.points j).1 ∧ (cd.points i).2 = -(cd.points j).2

/-- There exist two division points that determine a diameter -/
theorem exists_diameter_points {k : ℕ} (cd : CircleDivision k) :
  ∃ (i j : Fin (3 * k)), is_diameter cd i j :=
sorry

end exists_diameter_points_l1115_111598


namespace job1_rate_is_correct_l1115_111568

/-- Represents the hourly rate of job 1 -/
def job1_rate : ℝ := 7

/-- Represents the hourly rate of job 2 -/
def job2_rate : ℝ := 10

/-- Represents the hourly rate of job 3 -/
def job3_rate : ℝ := 12

/-- Represents the number of hours worked on job 1 per day -/
def job1_hours : ℝ := 3

/-- Represents the number of hours worked on job 2 per day -/
def job2_hours : ℝ := 2

/-- Represents the number of hours worked on job 3 per day -/
def job3_hours : ℝ := 4

/-- Represents the number of days worked -/
def days_worked : ℝ := 5

/-- Represents the total earnings for the period -/
def total_earnings : ℝ := 445

theorem job1_rate_is_correct : 
  days_worked * (job1_hours * job1_rate + job2_hours * job2_rate + job3_hours * job3_rate) = total_earnings := by
  sorry

end job1_rate_is_correct_l1115_111568


namespace parallel_line_m_value_l1115_111530

/-- A line passing through two points is parallel to another line -/
def is_parallel_line (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (y2 - y1) / (x2 - x1) = -a / b

/-- The value of m for which the line through (-2, m) and (m, 4) is parallel to 2x + y - 1 = 0 -/
theorem parallel_line_m_value :
  ∀ m : ℝ, is_parallel_line (-2) m m 4 2 1 (-1) → m = -8 := by
  sorry

end parallel_line_m_value_l1115_111530


namespace percentage_students_taking_music_l1115_111523

/-- Given a school with students taking electives, prove the percentage taking music. -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (h1 : total_students = 400)
  (h2 : dance_students = 120)
  (h3 : art_students = 200)
  : (total_students - dance_students - art_students) / total_students * 100 = 20 := by
  sorry

end percentage_students_taking_music_l1115_111523


namespace solve_bowtie_equation_l1115_111557

-- Define the operation ⊛
noncomputable def bowtie (a b : ℝ) : ℝ := a + 3 * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem solve_bowtie_equation (g : ℝ) : bowtie 5 g = 14 → g = 6 := by
  sorry

end solve_bowtie_equation_l1115_111557


namespace cement_bought_l1115_111599

/-- The amount of cement bought, given the total amount, original amount, and son's contribution -/
theorem cement_bought (total : ℕ) (original : ℕ) (son_contribution : ℕ) 
  (h1 : total = 450)
  (h2 : original = 98)
  (h3 : son_contribution = 137) :
  total - (original + son_contribution) = 215 := by
  sorry

end cement_bought_l1115_111599


namespace square_one_implies_plus_minus_one_l1115_111555

theorem square_one_implies_plus_minus_one (x : ℝ) : x^2 = 1 → x = 1 ∨ x = -1 := by
  sorry

end square_one_implies_plus_minus_one_l1115_111555


namespace boat_journey_time_l1115_111525

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time (river_speed : ℝ) (boat_speed : ℝ) (distance : ℝ) : 
  river_speed = 2 →
  boat_speed = 6 →
  distance = 56 →
  (distance / (boat_speed - river_speed) + distance / (boat_speed + river_speed)) = 21 := by
  sorry

#check boat_journey_time

end boat_journey_time_l1115_111525


namespace cylinder_radius_is_18_over_5_l1115_111536

/-- A right circular cone with a right circular cylinder inscribed within it. -/
structure ConeWithCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ

/-- The conditions for our specific cone and cylinder. -/
def cone_cylinder_conditions (c : ConeWithCylinder) : Prop :=
  c.cone_diameter = 12 ∧
  c.cone_altitude = 18 ∧
  c.cylinder_radius * 2 = c.cylinder_radius * 2

theorem cylinder_radius_is_18_over_5 (c : ConeWithCylinder) 
  (h : cone_cylinder_conditions c) : c.cylinder_radius = 18 / 5 := by
  sorry

end cylinder_radius_is_18_over_5_l1115_111536


namespace maggie_income_l1115_111533

def office_rate : ℝ := 10
def tractor_rate : ℝ := 12
def tractor_hours : ℝ := 13
def office_hours : ℝ := 2 * tractor_hours

def total_income : ℝ := office_rate * office_hours + tractor_rate * tractor_hours

theorem maggie_income : total_income = 416 := by
  sorry

end maggie_income_l1115_111533


namespace not_strictly_decreasing_cubic_function_l1115_111565

theorem not_strictly_decreasing_cubic_function (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (-x₁^3 + b*x₁^2 - (2*b + 3)*x₁ + 2 - b) ≤ (-x₂^3 + b*x₂^2 - (2*b + 3)*x₂ + 2 - b)) ↔ 
  (b < -1 ∨ b > 3) :=
by sorry

end not_strictly_decreasing_cubic_function_l1115_111565


namespace sqrt_5_irrational_l1115_111550

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define what it means for a number to be irrational
def IsIrrational (x : ℝ) : Prop := ¬(IsRational x)

-- State the theorem
theorem sqrt_5_irrational : IsIrrational (Real.sqrt 5) := by
  sorry

end sqrt_5_irrational_l1115_111550


namespace shirt_to_pants_ratio_l1115_111548

/-- Proves that the ratio of the price of the shirt to the price of the pants is 3:4 given the conditions of the problem. -/
theorem shirt_to_pants_ratio (total_cost pants_price shoes_price shirt_price : ℕ) : 
  total_cost = 340 →
  pants_price = 120 →
  shoes_price = pants_price + 10 →
  shirt_price = total_cost - pants_price - shoes_price →
  (shirt_price : ℚ) / pants_price = 3 / 4 := by
  sorry

end shirt_to_pants_ratio_l1115_111548


namespace tan_22_5_degrees_l1115_111587

theorem tan_22_5_degrees :
  Real.tan (22.5 * π / 180) = Real.sqrt 8 - Real.sqrt 0 - 2 := by sorry

end tan_22_5_degrees_l1115_111587


namespace expression_simplification_l1115_111558

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(x + 2) - 5*(3 - 2*x) = 19*x - 13 := by
  sorry

end expression_simplification_l1115_111558


namespace only_D_is_certain_l1115_111534

structure Event where
  description : String
  is_certain : Bool

def event_A : Event := { description := "It will definitely rain on a cloudy day", is_certain := false }
def event_B : Event := { description := "When tossing a fair coin, the head side faces up", is_certain := false }
def event_C : Event := { description := "A boy's height is definitely taller than a girl's", is_certain := false }
def event_D : Event := { description := "When oil is dropped into water, the oil will float on the surface of the water", is_certain := true }

def events : List Event := [event_A, event_B, event_C, event_D]

theorem only_D_is_certain : ∃! e : Event, e ∈ events ∧ e.is_certain := by sorry

end only_D_is_certain_l1115_111534


namespace max_garden_area_l1115_111517

def garden_area (width : ℝ) : ℝ := 2 * width * width

def garden_perimeter (width : ℝ) : ℝ := 6 * width

theorem max_garden_area :
  ∃ (w : ℝ), w > 0 ∧ garden_perimeter w = 480 ∧
  ∀ (x : ℝ), x > 0 ∧ garden_perimeter x = 480 → garden_area x ≤ garden_area w ∧
  garden_area w = 12800 :=
sorry

end max_garden_area_l1115_111517


namespace system_solution_equivalence_l1115_111519

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

-- Define the solution set
def solution_set (x y z : ℝ) : Prop :=
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2) ∨
  (x = -3 ∧ y = 2) ∨
  (x = 3 ∧ z = -1) ∨
  (x = -3 ∧ z = 1)

-- State the theorem
theorem system_solution_equivalence :
  ∀ x y z : ℝ, system x y z ↔ solution_set x y z :=
sorry

end system_solution_equivalence_l1115_111519


namespace sum_of_eleventh_powers_l1115_111553

/-- Given two real numbers a and b satisfying certain conditions, prove that a^11 + b^11 = 199 -/
theorem sum_of_eleventh_powers (a b : ℝ) : 
  (a + b = 1) →
  (a^2 + b^2 = 3) →
  (a^3 + b^3 = 4) →
  (a^4 + b^4 = 7) →
  (a^5 + b^5 = 11) →
  (∀ n ≥ 3, a^n + b^n = (a^(n-1) + b^(n-1)) + (a^(n-2) + b^(n-2))) →
  a^11 + b^11 = 199 := by
  sorry

end sum_of_eleventh_powers_l1115_111553


namespace f_derivative_at_one_l1115_111544

-- Define the function f
def f (x : ℝ) (f'₁ : ℝ) : ℝ := x^2 + 2*x*f'₁ - 6

-- State the theorem
theorem f_derivative_at_one :
  ∃ f'₁ : ℝ, (∀ x, deriv (f · f'₁) x = 2*x + 2*f'₁) ∧ f'₁ = -2 :=
sorry

end f_derivative_at_one_l1115_111544


namespace volume_bound_l1115_111527

/-- 
Given a body in 3D space, its volume does not exceed the square root of the product 
of the areas of its projections onto the coordinate planes.
-/
theorem volume_bound (S₁ S₂ S₃ V : ℝ) 
  (h₁ : S₁ > 0) (h₂ : S₂ > 0) (h₃ : S₃ > 0) (h₄ : V > 0)
  (h_S₁ : S₁ = area_projection_xy)
  (h_S₂ : S₂ = area_projection_yz)
  (h_S₃ : S₃ = area_projection_zx)
  (h_V : V = volume_of_body) : 
  V ≤ Real.sqrt (S₁ * S₂ * S₃) := by
  sorry

end volume_bound_l1115_111527


namespace arithmetic_sequence_tenth_term_l1115_111594

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 8 + a 9 = 32)
  (h_seventh : a 7 = 1) :
  a 10 = 31 :=
sorry

end arithmetic_sequence_tenth_term_l1115_111594


namespace copper_button_percentage_is_28_percent_l1115_111521

/-- Represents the composition of items in a basket --/
structure BasketComposition where
  pin_percentage : ℝ
  brass_button_percentage : ℝ
  copper_button_percentage : ℝ

/-- The percentage of copper buttons in the basket --/
def copper_button_percentage (b : BasketComposition) : ℝ :=
  b.copper_button_percentage

/-- Theorem stating the percentage of copper buttons in the basket --/
theorem copper_button_percentage_is_28_percent 
  (b : BasketComposition)
  (h1 : b.pin_percentage = 0.3)
  (h2 : b.brass_button_percentage = 0.42)
  (h3 : b.pin_percentage + b.brass_button_percentage + b.copper_button_percentage = 1) :
  copper_button_percentage b = 0.28 := by
  sorry

#check copper_button_percentage_is_28_percent

end copper_button_percentage_is_28_percent_l1115_111521


namespace intersection_complement_when_m_3_union_equality_iff_m_range_l1115_111591

-- Define the sets A and B
def A : Set ℝ := {x | |x| ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 2*m + 1}

-- Define the complement of A
def C_U_A : Set ℝ := {x | |x| > 3}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  (C_U_A ∩ B 3) = {x | 3 < x ∧ x < 7} := by sorry

-- Theorem 2
theorem union_equality_iff_m_range :
  ∀ m : ℝ, (A ∪ B m = A) ↔ (-2 ≤ m ∧ m ≤ 1) := by sorry

end intersection_complement_when_m_3_union_equality_iff_m_range_l1115_111591


namespace purchase_cost_l1115_111574

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of a single cookie in dollars -/
def cookie_cost : ℕ := 2

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 4

/-- The number of sodas purchased -/
def num_sodas : ℕ := 6

/-- The number of cookies purchased -/
def num_cookies : ℕ := 7

/-- The total cost of the purchase in dollars -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas + cookie_cost * num_cookies

theorem purchase_cost : total_cost = 48 := by
  sorry

end purchase_cost_l1115_111574


namespace quadratic_inequality_range_l1115_111572

theorem quadratic_inequality_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := by
  sorry

end quadratic_inequality_range_l1115_111572


namespace expand_cube_difference_l1115_111588

theorem expand_cube_difference (x y : ℝ) : (x + y) * (x^2 - x*y + y^2) = x^3 + y^3 := by
  sorry

end expand_cube_difference_l1115_111588
