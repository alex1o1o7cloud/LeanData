import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3226_322608

theorem no_solution_for_equation (x y : ℝ) : xy = 1 → ¬(Real.sqrt (x^2 + y^2) = x + y) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3226_322608


namespace NUMINAMATH_CALUDE_abs_eq_piecewise_l3226_322685

theorem abs_eq_piecewise (x : ℝ) : |x| = if x ≥ 0 then x else -x := by sorry

end NUMINAMATH_CALUDE_abs_eq_piecewise_l3226_322685


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_ratio_l3226_322677

theorem consecutive_odd_numbers_ratio (x : ℝ) (k m : ℝ) : 
  x = 4.2 →                             -- First number is 4.2
  9 * x = k * (x + 4) + m * (x + 2) + 9  -- Equation from the problem
    → (x + 4) / (x + 2) = 41 / 31        -- Ratio of third to second number
  := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_ratio_l3226_322677


namespace NUMINAMATH_CALUDE_books_left_l3226_322614

def initial_books : ℕ := 75
def borrowed_books : ℕ := 18

theorem books_left : initial_books - borrowed_books = 57 := by
  sorry

end NUMINAMATH_CALUDE_books_left_l3226_322614


namespace NUMINAMATH_CALUDE_q_age_is_40_l3226_322647

/-- Represents the ages of two people p and q --/
structure Ages where
  p : ℕ
  q : ℕ

/-- The condition stated by p --/
def age_condition (ages : Ages) : Prop :=
  ages.p = 3 * (ages.q - (ages.p - ages.q))

/-- The sum of their present ages is 100 --/
def age_sum (ages : Ages) : Prop :=
  ages.p + ages.q = 100

/-- Theorem stating that given the conditions, q's present age is 40 --/
theorem q_age_is_40 (ages : Ages) 
  (h1 : age_condition ages) 
  (h2 : age_sum ages) : 
  ages.q = 40 := by
  sorry

end NUMINAMATH_CALUDE_q_age_is_40_l3226_322647


namespace NUMINAMATH_CALUDE_bob_profit_l3226_322698

/-- Calculates the profit from breeding and selling show dogs -/
def dogBreedingProfit (numDogs : ℕ) (dogCost : ℕ) (numPuppies : ℕ) (puppyPrice : ℕ) 
                      (foodVaccinationCost : ℕ) (advertisingCost : ℕ) : ℤ :=
  (numPuppies * puppyPrice : ℤ) - (numDogs * dogCost + foodVaccinationCost + advertisingCost)

theorem bob_profit : 
  dogBreedingProfit 2 250 6 350 500 150 = 950 := by
  sorry

end NUMINAMATH_CALUDE_bob_profit_l3226_322698


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3226_322637

/-- The inequality x^2 - 2ax + a > 0 has ℝ as its solution set -/
def has_real_solution_set (a : ℝ) : Prop :=
  ∀ x, x^2 - 2*a*x + a > 0

/-- 0 < a < 1 -/
def a_in_open_unit_interval (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem sufficient_not_necessary :
  (∀ a : ℝ, has_real_solution_set a → a_in_open_unit_interval a) ∧
  (∃ a : ℝ, a_in_open_unit_interval a ∧ ¬has_real_solution_set a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3226_322637


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l3226_322656

theorem at_least_one_real_root (a b c : ℝ) : 
  (a - b)^2 - 4*(b - c) ≥ 0 ∨ 
  (b - c)^2 - 4*(c - a) ≥ 0 ∨ 
  (c - a)^2 - 4*(a - b) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l3226_322656


namespace NUMINAMATH_CALUDE_meal_center_allocation_l3226_322624

/-- Represents the meal center's soup can allocation problem -/
theorem meal_center_allocation (total_cans : ℕ) (adults_per_can children_per_can : ℕ) 
  (children_to_feed : ℕ) (adults_fed : ℕ) :
  total_cans = 10 →
  adults_per_can = 4 →
  children_per_can = 7 →
  children_to_feed = 21 →
  adults_fed = (total_cans - (children_to_feed / children_per_can)) * adults_per_can →
  adults_fed = 28 := by
sorry

end NUMINAMATH_CALUDE_meal_center_allocation_l3226_322624


namespace NUMINAMATH_CALUDE_milk_water_ratio_in_first_vessel_l3226_322620

-- Define the volumes of the vessels
def vessel1_volume : ℚ := 3
def vessel2_volume : ℚ := 5

-- Define the milk to water ratio in the second vessel
def vessel2_milk_ratio : ℚ := 6
def vessel2_water_ratio : ℚ := 4

-- Define the mixed ratio
def mixed_ratio : ℚ := 1

-- Define the unknown ratio for the first vessel
def vessel1_milk_ratio : ℚ := 1
def vessel1_water_ratio : ℚ := 2

theorem milk_water_ratio_in_first_vessel :
  (vessel1_milk_ratio / vessel1_water_ratio = 1 / 2) ∧
  (vessel1_milk_ratio * vessel1_volume + vessel2_milk_ratio * vessel2_volume) /
  (vessel1_water_ratio * vessel1_volume + vessel2_water_ratio * vessel2_volume) = mixed_ratio :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_in_first_vessel_l3226_322620


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3226_322609

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  b = 3 * a →
  a^2 + b^2 = c^2 →
  a^2 + b^2 + c^2 = 500 →
  c = 5 * Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3226_322609


namespace NUMINAMATH_CALUDE_reach_floor_pushups_l3226_322629

/-- Represents the number of push-up variations -/
def numVariations : ℕ := 5

/-- Represents the number of training days per week -/
def trainingDaysPerWeek : ℕ := 6

/-- Represents the number of reps added per day -/
def repsAddedPerDay : ℕ := 1

/-- Represents the target number of reps to progress to the next variation -/
def targetReps : ℕ := 25

/-- Calculates the number of weeks needed to progress through one variation -/
def weeksPerVariation : ℕ := 
  (targetReps + trainingDaysPerWeek - 1) / trainingDaysPerWeek

/-- The total number of weeks needed to reach floor push-ups -/
def totalWeeks : ℕ := numVariations * weeksPerVariation

theorem reach_floor_pushups : totalWeeks = 20 := by
  sorry

end NUMINAMATH_CALUDE_reach_floor_pushups_l3226_322629


namespace NUMINAMATH_CALUDE_at_most_one_root_l3226_322659

theorem at_most_one_root (f : ℝ → ℝ) (h : ∀ a b, a < b → f a < f b) :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_at_most_one_root_l3226_322659


namespace NUMINAMATH_CALUDE_x_cube_x_square_order_l3226_322660

theorem x_cube_x_square_order (x : ℝ) (h : -1 < x ∧ x < 0) : x < x^3 ∧ x^3 < x^2 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_x_square_order_l3226_322660


namespace NUMINAMATH_CALUDE_test_question_points_l3226_322673

theorem test_question_points (total_points total_questions two_point_questions : ℕ) 
  (h1 : total_points = 100)
  (h2 : total_questions = 40)
  (h3 : two_point_questions = 30) :
  (total_points - 2 * two_point_questions) / (total_questions - two_point_questions) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_test_question_points_l3226_322673


namespace NUMINAMATH_CALUDE_decimal_places_of_fraction_l3226_322600

theorem decimal_places_of_fraction : ∃ (n : ℕ), 
  (5^7 : ℚ) / (10^5 * 125) = (n : ℚ) / 10^5 ∧ 
  0 < n ∧ 
  n < 10^5 :=
by sorry

end NUMINAMATH_CALUDE_decimal_places_of_fraction_l3226_322600


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l3226_322690

theorem theater_ticket_sales (total_tickets : ℕ) (adult_price senior_price : ℚ) (total_receipts : ℚ) :
  total_tickets = 510 →
  adult_price = 21 →
  senior_price = 15 →
  total_receipts = 8748 →
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l3226_322690


namespace NUMINAMATH_CALUDE_difference_in_sums_l3226_322664

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_of_rounded_integers (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem difference_in_sums :
  sum_of_rounded_integers 200 - sum_of_integers 200 = 120 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_sums_l3226_322664


namespace NUMINAMATH_CALUDE_distance_from_origin_l3226_322622

theorem distance_from_origin (P : ℝ × ℝ) (h : P = (5, 12)) : 
  Real.sqrt ((P.1)^2 + (P.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l3226_322622


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3226_322665

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) :
  let e := Real.sqrt 5 / 2 - 1 / 2
  ∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the ellipse
    F₁.1 = -Real.sqrt (a^2 - b^2) ∧ F₁.2 = 0 ∧
    F₂.1 = Real.sqrt (a^2 - b^2) ∧ F₂.2 = 0 ∧
    -- P is on the ellipse
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
    -- PF₂ is perpendicular to the x-axis
    P.1 = F₂.1 ∧
    -- |F₁F₂| = 2|PF₂|
    (F₁.1 - F₂.1)^2 = 4 * P.2^2 ∧
    -- The eccentricity is e
    e = Real.sqrt (a^2 - b^2) / a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3226_322665


namespace NUMINAMATH_CALUDE_two_digit_number_equals_three_times_square_of_units_digit_l3226_322635

theorem two_digit_number_equals_three_times_square_of_units_digit :
  ∀ n : ℕ, 10 ≤ n ∧ n < 100 →
    (n = 3 * (n % 10)^2) ↔ (n = 12 ∨ n = 75) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_equals_three_times_square_of_units_digit_l3226_322635


namespace NUMINAMATH_CALUDE_school_trip_classrooms_l3226_322693

theorem school_trip_classrooms 
  (students_per_classroom : ℕ) 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (h1 : students_per_classroom = 66)
  (h2 : seats_per_bus = 6)
  (h3 : buses_needed = 737) :
  (buses_needed * seats_per_bus) / students_per_classroom = 67 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_classrooms_l3226_322693


namespace NUMINAMATH_CALUDE_trajectory_and_max_area_l3226_322631

noncomputable section

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := p.1^2 / 2 + p.2^2 = 1

-- Define the relation between P and M
def P_relation (P M : ℝ × ℝ) : Prop := P.1 = 2 * M.1 ∧ P.2 = 2 * M.2

-- Define the trajectory C
def on_trajectory (p : ℝ × ℝ) : Prop := p.1^2 / 8 + p.2^2 / 4 = 1

-- Define the line l
def on_line (p : ℝ × ℝ) (m : ℝ) : Prop := p.2 = p.1 + m

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2)) / 2

theorem trajectory_and_max_area 
  (M : ℝ × ℝ) (P : ℝ × ℝ) (m : ℝ) (A B : ℝ × ℝ) :
  on_ellipse M → 
  P_relation P M → 
  m ≠ 0 →
  on_line A m →
  on_line B m →
  on_trajectory A →
  on_trajectory B →
  A ≠ B →
  (∀ P, P_relation P M → on_trajectory P) ∧
  (∀ X Y, on_trajectory X → on_trajectory Y → on_line X m → on_line Y m → 
    triangle_area O X Y ≤ 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_max_area_l3226_322631


namespace NUMINAMATH_CALUDE_product_sum_theorem_l3226_322612

theorem product_sum_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 15) :
  a * b + c * d = 1002 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l3226_322612


namespace NUMINAMATH_CALUDE_range_of_a_l3226_322638

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- Define the property that ¬p is sufficient but not necessary for ¬q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x a)) ∧ (∃ x, ¬(q x a) ∧ p x)

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, sufficient_not_necessary a) → (∀ a : ℝ, a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3226_322638


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l3226_322607

theorem modulus_of_complex_expression :
  let z : ℂ := (1 : ℂ) / (1 + Complex.I) + Complex.I
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l3226_322607


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l3226_322648

/-- Given a function f(x) = ax³ + bx - 2 where f(2017) = 10, prove that f(-2017) = -14 -/
theorem cubic_function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 2
  f 2017 = 10 → f (-2017) = -14 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l3226_322648


namespace NUMINAMATH_CALUDE_division_problem_l3226_322650

theorem division_problem (A : ℕ) : A = 1 → 23 = 13 * A + 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3226_322650


namespace NUMINAMATH_CALUDE_butter_fraction_for_chocolate_chip_cookies_l3226_322694

theorem butter_fraction_for_chocolate_chip_cookies 
  (total_butter : ℝ)
  (peanut_butter_fraction : ℝ)
  (sugar_cookie_fraction : ℝ)
  (remaining_butter : ℝ)
  (h1 : total_butter = 10)
  (h2 : peanut_butter_fraction = 1/5)
  (h3 : sugar_cookie_fraction = 1/3)
  (h4 : remaining_butter = 2)
  : (total_butter - (peanut_butter_fraction * total_butter) - 
     sugar_cookie_fraction * (total_butter - peanut_butter_fraction * total_butter) - 
     remaining_butter) / total_butter = 1/3 := by
  sorry

#check butter_fraction_for_chocolate_chip_cookies

end NUMINAMATH_CALUDE_butter_fraction_for_chocolate_chip_cookies_l3226_322694


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3226_322654

def is_valid (n : ℕ+) : Prop :=
  (Finset.card (Nat.divisors n) = 144) ∧
  (∃ k : ℕ, ∀ i : Fin 10, (k + i) ∈ Nat.divisors n)

theorem smallest_valid_number : 
  (is_valid 110880) ∧ (∀ m : ℕ+, m < 110880 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3226_322654


namespace NUMINAMATH_CALUDE_ashok_marks_average_l3226_322633

/-- Given a student's average marks and the marks in the last subject, 
    calculate the average marks in the remaining subjects. -/
def average_remaining_subjects (total_subjects : ℕ) (overall_average : ℚ) (last_subject_marks : ℕ) : ℚ :=
  ((overall_average * total_subjects) - last_subject_marks) / (total_subjects - 1)

/-- Theorem stating that given the conditions in the problem, 
    the average of marks in the first 5 subjects is 74. -/
theorem ashok_marks_average : 
  let total_subjects : ℕ := 6
  let overall_average : ℚ := 75
  let last_subject_marks : ℕ := 80
  average_remaining_subjects total_subjects overall_average last_subject_marks = 74 := by
  sorry

#eval average_remaining_subjects 6 75 80

end NUMINAMATH_CALUDE_ashok_marks_average_l3226_322633


namespace NUMINAMATH_CALUDE_cost_of_bananas_l3226_322676

/-- The cost of bananas given the following conditions:
  * The cost of one banana is 800 won
  * The cost of one kiwi is 400 won
  * The total number of bananas and kiwis is 18
  * The total amount spent is 10,000 won
-/
theorem cost_of_bananas :
  let banana_cost : ℕ := 800
  let kiwi_cost : ℕ := 400
  let total_fruits : ℕ := 18
  let total_spent : ℕ := 10000
  ∃ (num_bananas : ℕ),
    num_bananas * banana_cost + (total_fruits - num_bananas) * kiwi_cost = total_spent ∧
    num_bananas * banana_cost = 5600 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_bananas_l3226_322676


namespace NUMINAMATH_CALUDE_no_integer_solution_l3226_322617

theorem no_integer_solution : ¬∃ (x y : ℤ), x * (x + 1) = 13 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3226_322617


namespace NUMINAMATH_CALUDE_volleyball_team_girls_l3226_322643

/-- Given a volleyball team with the following properties:
  * The total number of team members is 30
  * 20 members attended the last meeting
  * One-third of the girls and all boys attended the meeting
  Prove that the number of girls on the team is 15 -/
theorem volleyball_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 20 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = attended →
  girls = 15 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_girls_l3226_322643


namespace NUMINAMATH_CALUDE_common_divisors_count_l3226_322692

/-- The number of positive divisors that 9240, 7920, and 8800 have in common -/
theorem common_divisors_count : Nat.card {d : ℕ | d > 0 ∧ d ∣ 9240 ∧ d ∣ 7920 ∧ d ∣ 8800} = 32 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_count_l3226_322692


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l3226_322634

theorem cone_vertex_angle (l r : ℝ) (h : l > 0) (h2 : r > 0) : 
  (2 * π * l / 3 = 2 * π * r) → 
  (2 * Real.arcsin (1 / 3) : ℝ) = 2 * Real.arcsin (r / l) := by
sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l3226_322634


namespace NUMINAMATH_CALUDE_extreme_value_implies_zero_derivative_converse_not_always_true_l3226_322639

-- Define a function that has an extreme value at a point
def has_extreme_value (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x ≤ f x₀ ∨ f x ≥ f x₀

-- Theorem statement
theorem extreme_value_implies_zero_derivative
  (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) :
  has_extreme_value f x₀ → deriv f x₀ = 0 :=
sorry

-- Counter-example to show the converse is not always true
theorem converse_not_always_true :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ deriv f 0 = 0 ∧ ¬(has_extreme_value f 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_zero_derivative_converse_not_always_true_l3226_322639


namespace NUMINAMATH_CALUDE_first_month_sale_is_7435_l3226_322672

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale for 6 months -/
def first_month_sale (second_month : ℕ) (third_month : ℕ) (fourth_month : ℕ) (fifth_month : ℕ) (sixth_month : ℕ) (average_sale : ℕ) : ℕ :=
  6 * average_sale - (second_month + third_month + fourth_month + fifth_month + sixth_month)

/-- Theorem stating that the sale in the first month is 7435 given the specified conditions -/
theorem first_month_sale_is_7435 :
  first_month_sale 7927 7855 8230 7562 5991 7500 = 7435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_is_7435_l3226_322672


namespace NUMINAMATH_CALUDE_dart_game_equations_correct_l3226_322671

/-- Represents the dart throwing game scenario -/
structure DartGame where
  x : ℕ  -- number of times Xiao Hua hits the target
  y : ℕ  -- number of times the father hits the target

/-- The conditions of the dart throwing game -/
def validGame (game : DartGame) : Prop :=
  game.x + game.y = 30 ∧  -- total number of hits
  5 * game.x + 2 = 3 * game.y  -- score difference condition

/-- Theorem stating that the system of equations correctly represents the game -/
theorem dart_game_equations_correct (game : DartGame) :
  validGame game ↔ 
    (game.x + game.y = 30 ∧ 5 * game.x + 2 = 3 * game.y) :=
by sorry

end NUMINAMATH_CALUDE_dart_game_equations_correct_l3226_322671


namespace NUMINAMATH_CALUDE_pyramid_properties_l3226_322601

/-- Pyramid structure with given properties -/
structure Pyramid where
  -- Base is a rhombus
  base_is_rhombus : Prop
  -- Height of the pyramid
  height : ℝ
  -- K lies on diagonal AC
  k_on_diagonal : Prop
  -- KC = KA + AC
  kc_eq_ka_plus_ac : Prop
  -- Length of lateral edge TC
  tc_length : ℝ
  -- Angles of lateral faces to base
  angle1 : ℝ
  angle2 : ℝ

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_properties (p : Pyramid)
  (h_height : p.height = 1)
  (h_tc : p.tc_length = 2 * Real.sqrt 2)
  (h_angles : p.angle1 = π/6 ∧ p.angle2 = π/3) :
  ∃ (base_side angle_ta_tcd : ℝ),
    base_side = 7/6 ∧
    angle_ta_tcd = Real.arcsin (Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_pyramid_properties_l3226_322601


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l3226_322602

theorem cube_sum_of_roots (r s t : ℝ) : 
  (r - (20 : ℝ)^(1/3)) * (r - (60 : ℝ)^(1/3)) * (r - (120 : ℝ)^(1/3)) = 1 →
  (s - (20 : ℝ)^(1/3)) * (s - (60 : ℝ)^(1/3)) * (s - (120 : ℝ)^(1/3)) = 1 →
  (t - (20 : ℝ)^(1/3)) * (t - (60 : ℝ)^(1/3)) * (t - (120 : ℝ)^(1/3)) = 1 →
  r ≠ s → r ≠ t → s ≠ t →
  r^3 + s^3 + t^3 = 203 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l3226_322602


namespace NUMINAMATH_CALUDE_soldiers_on_first_side_l3226_322686

theorem soldiers_on_first_side (food_per_soldier_first : ℕ)
                               (food_difference : ℕ)
                               (soldier_difference : ℕ)
                               (total_food : ℕ) :
  food_per_soldier_first = 10 →
  food_difference = 2 →
  soldier_difference = 500 →
  total_food = 68000 →
  ∃ (x : ℕ), 
    x * food_per_soldier_first + 
    (x - soldier_difference) * (food_per_soldier_first - food_difference) = total_food ∧
    x = 4000 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_on_first_side_l3226_322686


namespace NUMINAMATH_CALUDE_plant_branches_theorem_l3226_322669

theorem plant_branches_theorem : ∃ (x : ℕ), x > 0 ∧ 1 + x + x^2 = 57 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_theorem_l3226_322669


namespace NUMINAMATH_CALUDE_constant_integral_equals_one_l3226_322684

theorem constant_integral_equals_one : ∫ x in (0:ℝ)..1, (1:ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_constant_integral_equals_one_l3226_322684


namespace NUMINAMATH_CALUDE_bread_inventory_l3226_322603

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_inventory : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end NUMINAMATH_CALUDE_bread_inventory_l3226_322603


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3226_322619

def U : Set Int := {x | (x + 1) * (x - 3) ≤ 0}

def A : Set Int := {0, 1, 2}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {-1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3226_322619


namespace NUMINAMATH_CALUDE_symmetric_points_y_axis_l3226_322663

/-- Given two points in R² that are symmetric about the y-axis, 
    prove that their x-coordinates are negatives of each other 
    and their y-coordinates are the same. -/
theorem symmetric_points_y_axis 
  (A B : ℝ × ℝ) 
  (h_symmetric : A.1 = -B.1 ∧ A.2 = B.2) 
  (h_A : A = (1, -2)) : 
  B = (-1, -2) := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_y_axis_l3226_322663


namespace NUMINAMATH_CALUDE_grandfather_grandson_age_relation_l3226_322632

theorem grandfather_grandson_age_relation :
  ∀ (grandfather_age grandson_age : ℕ) (years : ℕ),
    50 < grandfather_age →
    grandfather_age < 90 →
    grandfather_age = 31 * grandson_age →
    (grandfather_age + years = 7 * (grandson_age + years)) →
    years = 8 :=
by sorry

end NUMINAMATH_CALUDE_grandfather_grandson_age_relation_l3226_322632


namespace NUMINAMATH_CALUDE_investment_problem_l3226_322655

/-- Proves that given the conditions of the investment problem, b's investment amount is 1000. -/
theorem investment_problem (a b c total_profit c_share : ℚ) : 
  a = 800 →
  c = 1200 →
  total_profit = 1000 →
  c_share = 400 →
  c_share / total_profit = c / (a + b + c) →
  b = 1000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3226_322655


namespace NUMINAMATH_CALUDE_total_quantities_l3226_322679

theorem total_quantities (average : ℝ) (average_three : ℝ) (average_two : ℝ) : 
  average = 11 → average_three = 4 → average_two = 21.5 → 
  ∃ (n : ℕ), n = 5 ∧ 
    (n : ℝ) * average = 3 * average_three + 2 * average_two := by
  sorry

end NUMINAMATH_CALUDE_total_quantities_l3226_322679


namespace NUMINAMATH_CALUDE_tony_remaining_money_l3226_322645

def initial_amount : ℕ := 20
def ticket_cost : ℕ := 8
def hotdog_cost : ℕ := 3

theorem tony_remaining_money :
  initial_amount - ticket_cost - hotdog_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_tony_remaining_money_l3226_322645


namespace NUMINAMATH_CALUDE_oyster_consumption_l3226_322628

/-- The number of oysters Squido eats -/
def squido_oysters : ℕ := 200

/-- The number of oysters Crabby eats -/
def crabby_oysters : ℕ := 2 * squido_oysters

/-- The total number of oysters eaten by Crabby and Squido -/
def total_oysters : ℕ := squido_oysters + crabby_oysters

theorem oyster_consumption :
  total_oysters = 600 :=
by sorry

end NUMINAMATH_CALUDE_oyster_consumption_l3226_322628


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l3226_322653

theorem complex_power_magnitude : Complex.abs ((4/5 : ℂ) + (3/5 : ℂ) * Complex.I) ^ 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l3226_322653


namespace NUMINAMATH_CALUDE_prime_product_theorem_l3226_322667

theorem prime_product_theorem (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ →
  2*p₁ + 3*p₂ + 5*p₃ + 7*p₄ = 162 →
  11*p₁ + 7*p₂ + 5*p₃ + 4*p₄ = 162 →
  p₁ * p₂ * p₃ * p₄ = 570 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_theorem_l3226_322667


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l3226_322641

theorem largest_common_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∀ m : ℕ, m > 315 → ∃ k : ℕ, k > 0 ∧ Even k ∧ 
    ¬(m ∣ (k+1)*(k+3)*(k+5)*(k+7)*(k+9)*(k+11)*(k+13))) ∧
  (315 ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_consecutive_odd_product_l3226_322641


namespace NUMINAMATH_CALUDE_harmonious_triplet_from_intersections_l3226_322674

/-- Definition of a harmonious triplet -/
def is_harmonious_triplet (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
  (1/x = 1/y + 1/z ∨ 1/y = 1/x + 1/z ∨ 1/z = 1/x + 1/y)

/-- Theorem about harmonious triplets formed by intersections -/
theorem harmonious_triplet_from_intersections
  (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  let x₁ := -c / b
  let x₂ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₃ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  is_harmonious_triplet x₁ x₂ x₃ :=
by sorry

end NUMINAMATH_CALUDE_harmonious_triplet_from_intersections_l3226_322674


namespace NUMINAMATH_CALUDE_simplify_expression_l3226_322618

theorem simplify_expression : (3 * 2 + 4 + 6) / 3 - 2 / 3 = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3226_322618


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3226_322680

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := fun x ↦ (x - 1)^2 - 4
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -1 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3226_322680


namespace NUMINAMATH_CALUDE_album_pages_count_l3226_322627

theorem album_pages_count : ∃ (x : ℕ) (y : ℕ), 
  x > 0 ∧ 
  y > 0 ∧ 
  20 * x < y ∧ 
  23 * x > y ∧ 
  21 * x + y = 500 ∧ 
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_album_pages_count_l3226_322627


namespace NUMINAMATH_CALUDE_smallest_c_value_l3226_322658

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_c_value (a b c d e : ℕ) :
  a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧
  is_perfect_square (b + c + d) ∧
  is_perfect_cube (a + b + c + d + e) →
  c ≥ 675 ∧ ∃ (a' b' c' d' e' : ℕ),
    a' + 1 = b' ∧ b' + 1 = c' ∧ c' + 1 = d' ∧ d' + 1 = e' ∧
    is_perfect_square (b' + c' + d') ∧
    is_perfect_cube (a' + b' + c' + d' + e') ∧
    c' = 675 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3226_322658


namespace NUMINAMATH_CALUDE_second_quadrant_trig_simplification_l3226_322621

theorem second_quadrant_trig_simplification (α : Real) 
  (h : π/2 < α ∧ α < π) : 
  (Real.sqrt (1 + 2 * Real.sin (5 * π - α) * Real.cos (α - π))) / 
  (Real.sin (α - 3 * π / 2) - Real.sqrt (1 - Real.sin (3 * π / 2 + α)^2)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_simplification_l3226_322621


namespace NUMINAMATH_CALUDE_inequalities_from_sqrt_reciprocal_l3226_322681

theorem inequalities_from_sqrt_reciprocal (a b : ℝ) (h : 1 / Real.sqrt a > 1 / Real.sqrt b) :
  (b / (a + b) + a / (2 * b) ≥ (2 * Real.sqrt 2 - 1) / 2) ∧
  ((b + 1) / (a + 1) < b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sqrt_reciprocal_l3226_322681


namespace NUMINAMATH_CALUDE_find_k_l3226_322683

theorem find_k : ∃ k : ℚ, (32 / k = 4) ∧ (k = 8) := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3226_322683


namespace NUMINAMATH_CALUDE_not_square_or_cube_of_2pow_minus_1_l3226_322689

theorem not_square_or_cube_of_2pow_minus_1 (n : ℕ) (h : n > 1) :
  ¬∃ (a : ℤ), (2^n - 1 : ℤ) = a^2 ∨ (2^n - 1 : ℤ) = a^3 := by
  sorry

end NUMINAMATH_CALUDE_not_square_or_cube_of_2pow_minus_1_l3226_322689


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3226_322604

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

def geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3226_322604


namespace NUMINAMATH_CALUDE_sum_a_d_l3226_322615

theorem sum_a_d (a b c d : ℤ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l3226_322615


namespace NUMINAMATH_CALUDE_wedge_volume_l3226_322649

/-- The volume of a wedge that represents one-third of a cylindrical cheese log -/
theorem wedge_volume (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cylinder_volume := π * r^2 * h
  let wedge_volume := (1/3) * cylinder_volume
  h = 8 ∧ r = 5 → wedge_volume = (200 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l3226_322649


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3226_322661

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 13) (h₃ : a₃ = 23) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 143 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l3226_322661


namespace NUMINAMATH_CALUDE_sqrt_product_property_sqrt_40_in_terms_of_a_b_l3226_322626

theorem sqrt_product_property (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  Real.sqrt x * Real.sqrt y = Real.sqrt (x * y) := by sorry

theorem sqrt_40_in_terms_of_a_b (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 10) :
  Real.sqrt 40 = Real.sqrt 2 * a * b := by sorry

end NUMINAMATH_CALUDE_sqrt_product_property_sqrt_40_in_terms_of_a_b_l3226_322626


namespace NUMINAMATH_CALUDE_eight_people_arrangement_l3226_322651

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where 2 specific people are together -/
def arrangementsTwoTogether (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where 2 specific people are not together -/
def arrangementsNotTogether (n : ℕ) : ℕ :=
  totalArrangements n - arrangementsTwoTogether n

theorem eight_people_arrangement :
  arrangementsNotTogether 8 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_arrangement_l3226_322651


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3226_322640

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The arithmetic sequence -/
def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_sum :
  (S 7 = 28) → (S 11 = 66) → (S 9 = 45) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3226_322640


namespace NUMINAMATH_CALUDE_prime_square_mod_240_l3226_322678

theorem prime_square_mod_240 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ r₁ < 240 ∧ r₂ < 240 ∧
  ∀ (q : Nat), Nat.Prime q → q > 5 → (q^2 % 240 = r₁ ∨ q^2 % 240 = r₂) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_240_l3226_322678


namespace NUMINAMATH_CALUDE_average_of_numbers_l3226_322646

def numbers : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : ℝ) = 125830.8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3226_322646


namespace NUMINAMATH_CALUDE_museum_visitors_scientific_notation_l3226_322675

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem museum_visitors_scientific_notation :
  toScientificNotation 3300000 = ScientificNotation.mk 3.3 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_museum_visitors_scientific_notation_l3226_322675


namespace NUMINAMATH_CALUDE_inequality_proof_l3226_322695

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.sqrt a + Real.sqrt b = 2) :
  (a * Real.sqrt b + b * Real.sqrt a ≤ 2) ∧ (2 ≤ a^2 + b^2) ∧ (a^2 + b^2 < 16) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3226_322695


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3226_322642

theorem sqrt_inequality (n : ℕ+) : Real.sqrt (n + 1) - Real.sqrt n < 1 / (2 * Real.sqrt n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3226_322642


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3226_322625

def A : Set ℝ := {x | (x - 3) / (x - 7) < 0}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

theorem set_operations_and_subset :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) ∧
  (∀ a : ℝ, C a ⊆ (A ∪ B) ↔ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3226_322625


namespace NUMINAMATH_CALUDE_triangle_area_72_l3226_322611

theorem triangle_area_72 (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * (2*x) * x = 72) : x = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_72_l3226_322611


namespace NUMINAMATH_CALUDE_clara_quarters_problem_l3226_322616

theorem clara_quarters_problem : ∃! q : ℕ, 
  8 < q ∧ q < 80 ∧ 
  q % 3 = 1 ∧ 
  q % 4 = 1 ∧ 
  q % 5 = 1 ∧ 
  q = 61 := by
sorry

end NUMINAMATH_CALUDE_clara_quarters_problem_l3226_322616


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l3226_322657

/-- The number of days in a leap year -/
def leap_year_days : ℕ := 366

/-- The number of possible combinations for the two extra days in a leap year -/
def extra_day_combinations : ℕ := 7

/-- The number of combinations that result in 53 Sundays -/
def favorable_combinations : ℕ := 2

/-- The probability of a leap year having 53 Sundays -/
def prob_53_sundays : ℚ := favorable_combinations / extra_day_combinations

theorem leap_year_53_sundays_probability :
  prob_53_sundays = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_probability_l3226_322657


namespace NUMINAMATH_CALUDE_gage_skating_problem_l3226_322630

theorem gage_skating_problem (days_75min : ℕ) (days_90min : ℕ) (total_days : ℕ) (avg_minutes : ℕ) :
  days_75min = 5 →
  days_90min = 3 →
  total_days = days_75min + days_90min + 1 →
  avg_minutes = 85 →
  (days_75min * 75 + days_90min * 90 + (total_days * avg_minutes - (days_75min * 75 + days_90min * 90))) / total_days = avg_minutes :=
by sorry

end NUMINAMATH_CALUDE_gage_skating_problem_l3226_322630


namespace NUMINAMATH_CALUDE_intersection_empty_iff_m_range_union_equals_B_iff_m_range_l3226_322662

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}
def B : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem for part (I)
theorem intersection_empty_iff_m_range (m : ℝ) :
  A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 :=
sorry

-- Theorem for part (II)
theorem union_equals_B_iff_m_range (m : ℝ) :
  A m ∪ B = B ↔ m < -7 ∨ m > 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_m_range_union_equals_B_iff_m_range_l3226_322662


namespace NUMINAMATH_CALUDE_reciprocal_of_x_l3226_322644

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2*x^2 = 0) (h2 : x ≠ 0) : 1/x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_x_l3226_322644


namespace NUMINAMATH_CALUDE_bottle_lasts_eight_months_l3226_322668

/-- Represents the number of pills in a bottle -/
def bottle_pills : ℕ := 60

/-- Represents the fraction of a pill consumed daily -/
def daily_consumption : ℚ := 1/4

/-- Represents the number of days in a month (approximation) -/
def days_per_month : ℕ := 30

/-- Calculates the number of months a bottle will last -/
def bottle_duration : ℚ := (bottle_pills : ℚ) / daily_consumption / days_per_month

theorem bottle_lasts_eight_months :
  bottle_duration = 8 := by sorry

end NUMINAMATH_CALUDE_bottle_lasts_eight_months_l3226_322668


namespace NUMINAMATH_CALUDE_baseball_gear_cost_l3226_322682

/-- Calculates the total cost of baseball gear including tax -/
def total_cost (birthday_money : ℚ) (glove_price : ℚ) (glove_discount : ℚ) 
  (baseball_price : ℚ) (bat_price : ℚ) (bat_discount : ℚ) (cleats_price : ℚ) 
  (cap_price : ℚ) (tax_rate : ℚ) : ℚ :=
  let discounted_glove := glove_price * (1 - glove_discount)
  let discounted_bat := bat_price * (1 - bat_discount)
  let subtotal := discounted_glove + baseball_price + discounted_bat + cleats_price + cap_price
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating the total cost of baseball gear -/
theorem baseball_gear_cost : 
  total_cost 120 35 0.2 15 50 0.1 30 10 0.07 = 136.96 := by
  sorry


end NUMINAMATH_CALUDE_baseball_gear_cost_l3226_322682


namespace NUMINAMATH_CALUDE_determine_q_investment_l3226_322666

/-- Represents the investment and profit sharing of two business partners -/
structure BusinessPartnership where
  p_investment : ℕ
  q_investment : ℕ
  profit_ratio : Rat

/-- Theorem stating that given P's investment and the profit ratio, Q's investment can be determined -/
theorem determine_q_investment (bp : BusinessPartnership) 
  (h1 : bp.p_investment = 75000)
  (h2 : bp.profit_ratio = 5 / 1) :
  bp.q_investment = 15000 := by
  sorry

#check determine_q_investment

end NUMINAMATH_CALUDE_determine_q_investment_l3226_322666


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l3226_322605

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- Point P is on the hyperbola -/
def P_on_hyperbola (px py : ℝ) : Prop := hyperbola px py

/-- M is the midpoint of OP -/
def M_is_midpoint (mx my px py : ℝ) : Prop := mx = px / 2 ∧ my = py / 2

/-- The trajectory equation for point M -/
def trajectory (x y : ℝ) : Prop := x^2 - 4*y^2 = 1

theorem trajectory_of_midpoint (mx my px py : ℝ) :
  P_on_hyperbola px py → M_is_midpoint mx my px py → trajectory mx my :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l3226_322605


namespace NUMINAMATH_CALUDE_speed_equivalence_l3226_322688

/-- Proves that a speed of 0.8 km/h is equivalent to 8/36 m/s -/
theorem speed_equivalence : ∃ (speed : ℚ), 
  (speed = 8 / 36) ∧ 
  (speed * 3600 / 1000 = 0.8) := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l3226_322688


namespace NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l3226_322652

/-- The function f(x) = x^2 / (x^2 + 1) is increasing on the interval (0, +∞) -/
theorem f_increasing_on_positive_reals :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → 
    (x₁^2 / (x₁^2 + 1)) < (x₂^2 / (x₂^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_positive_reals_l3226_322652


namespace NUMINAMATH_CALUDE_mod_eleven_fifth_power_l3226_322613

theorem mod_eleven_fifth_power (n : ℕ) : 
  11^5 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_eleven_fifth_power_l3226_322613


namespace NUMINAMATH_CALUDE_train_speed_proof_l3226_322636

/-- Proves that the speed of a train is 23.4 km/hr given specific conditions -/
theorem train_speed_proof (train_length : Real) (crossing_time : Real) (total_length : Real) :
  train_length = 180 →
  crossing_time = 30 →
  total_length = 195 →
  (total_length / crossing_time) * 3.6 = 23.4 :=
by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_train_speed_proof_l3226_322636


namespace NUMINAMATH_CALUDE_apple_rate_is_70_l3226_322606

-- Define the given quantities
def apple_quantity : ℕ := 8
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 45
def total_paid : ℕ := 965

-- Define the unknown apple rate
def apple_rate : ℕ := sorry

-- Theorem statement
theorem apple_rate_is_70 :
  apple_quantity * apple_rate + mango_quantity * mango_rate = total_paid →
  apple_rate = 70 := by
  sorry

end NUMINAMATH_CALUDE_apple_rate_is_70_l3226_322606


namespace NUMINAMATH_CALUDE_fathers_age_l3226_322687

/-- Given information about Sebastian, his sister, and their father's ages, prove the father's current age. -/
theorem fathers_age (sebastian_age : ℕ) (age_difference : ℕ) (years_ago : ℕ) (fraction : ℚ) : 
  sebastian_age = 40 →
  age_difference = 10 →
  years_ago = 5 →
  fraction = 3/4 →
  (sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) : ℚ) = 
    fraction * (sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) + years_ago) →
  sebastian_age - years_ago + (sebastian_age - age_difference - years_ago) + years_ago = 85 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l3226_322687


namespace NUMINAMATH_CALUDE_sphere_surface_volume_relation_l3226_322691

theorem sphere_surface_volume_relation :
  ∀ (r r' : ℝ) (A A' V V' : ℝ),
  (A = 4 * Real.pi * r^2) →
  (A' = 4 * A) →
  (V = (4/3) * Real.pi * r^3) →
  (V' = (4/3) * Real.pi * r'^3) →
  (A' = 4 * Real.pi * r'^2) →
  (V' = 8 * V) :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_volume_relation_l3226_322691


namespace NUMINAMATH_CALUDE_expression_equality_l3226_322610

theorem expression_equality : 
  Real.sqrt 8 + Real.sqrt (1/2) + (Real.sqrt 3 - 1)^2 + Real.sqrt 6 / (1/2 * Real.sqrt 2) = 
  5/2 * Real.sqrt 2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3226_322610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3226_322623

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 3 = 1) →
  (a 10 + a 11 = 9) →
  (a 5 + a 6 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3226_322623


namespace NUMINAMATH_CALUDE_factor_decomposition_96_l3226_322696

theorem factor_decomposition_96 : 
  ∃ (x y : ℤ), x * y = 96 ∧ x^2 + y^2 = 208 := by
  sorry

end NUMINAMATH_CALUDE_factor_decomposition_96_l3226_322696


namespace NUMINAMATH_CALUDE_solve_equation_l3226_322670

theorem solve_equation (x : ℝ) (h : x ≠ 0) :
  (2 / x + (3 / x) / (6 / x) = 1.25) → x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3226_322670


namespace NUMINAMATH_CALUDE_union_A_B_when_m_neg_two_intersection_A_B_equals_B_iff_l3226_322699

-- Define sets A and B
def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 * m + 3 < x ∧ x < m^2}

-- Theorem for part 1
theorem union_A_B_when_m_neg_two :
  A ∪ B (-2) = {x | -1 < x ∧ x < 4} := by sorry

-- Theorem for part 2
theorem intersection_A_B_equals_B_iff (m : ℝ) :
  A ∩ B m = B m ↔ m ∈ Set.Icc (-Real.sqrt 2) 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_neg_two_intersection_A_B_equals_B_iff_l3226_322699


namespace NUMINAMATH_CALUDE_prime_quadratic_roots_l3226_322697

theorem prime_quadratic_roots (p : ℕ) : 
  Prime p → 
  (∃ x y : ℤ, x^2 + p*x - 222*p = 0 ∧ y^2 + p*y - 222*p = 0) → 
  31 < p ∧ p ≤ 41 := by
sorry

end NUMINAMATH_CALUDE_prime_quadratic_roots_l3226_322697
