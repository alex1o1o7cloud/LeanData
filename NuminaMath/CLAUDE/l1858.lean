import Mathlib

namespace restaurant_time_is_ten_l1858_185876

-- Define the times as natural numbers (in minutes)
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_journey_time : ℕ := 32

-- Define the time to Lake Park restaurant as a function
def time_to_restaurant : ℕ := total_journey_time - (time_to_hidden_lake + time_from_hidden_lake)

-- Theorem statement
theorem restaurant_time_is_ten : time_to_restaurant = 10 := by
  sorry

end restaurant_time_is_ten_l1858_185876


namespace paint_calculation_l1858_185823

/-- The total amount of paint needed for finishing touches -/
def total_paint_needed (initial : ℕ) (purchased : ℕ) (additional_needed : ℕ) : ℕ :=
  initial + purchased + additional_needed

/-- Theorem stating that the total paint needed is the sum of initial, purchased, and additional needed paint -/
theorem paint_calculation (initial : ℕ) (purchased : ℕ) (additional_needed : ℕ) :
  total_paint_needed initial purchased additional_needed =
  initial + purchased + additional_needed :=
by
  sorry

#eval total_paint_needed 36 23 11

end paint_calculation_l1858_185823


namespace f_decreasing_interval_l1858_185808

-- Define the function
def f (x : ℝ) : ℝ := (x - 3) * |x|

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

-- Theorem statement
theorem f_decreasing_interval :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  is_decreasing_on f a b ∧
  ∀ (c d : ℝ), c < a ∨ b < d → ¬(is_decreasing_on f c d) :=
sorry

end f_decreasing_interval_l1858_185808


namespace shortest_player_height_l1858_185845

theorem shortest_player_height (tallest_height : Float) (height_difference : Float) :
  tallest_height = 77.75 →
  height_difference = 9.5 →
  tallest_height - height_difference = 68.25 := by
sorry

end shortest_player_height_l1858_185845


namespace total_turnips_l1858_185885

theorem total_turnips (melanie_turnips benny_turnips : ℕ) 
  (h1 : melanie_turnips = 139)
  (h2 : benny_turnips = 113) :
  melanie_turnips + benny_turnips = 252 := by
  sorry

end total_turnips_l1858_185885


namespace opposite_to_83_l1858_185852

/-- Represents a circle with 100 equally spaced points -/
def Circle := Fin 100

/-- A function assigning numbers 1 to 100 to the points on the circle -/
def numbering : Circle → Nat :=
  sorry

/-- Predicate to check if a number is opposite to another on the circle -/
def is_opposite (a b : Circle) : Prop :=
  sorry

/-- Predicate to check if numbers less than k are evenly distributed -/
def evenly_distributed (k : Nat) : Prop :=
  sorry

theorem opposite_to_83 (h : ∀ k, evenly_distributed k) :
  ∃ n : Circle, numbering n = 84 ∧ is_opposite n (⟨82, sorry⟩ : Circle) :=
sorry

end opposite_to_83_l1858_185852


namespace g_neg_two_l1858_185805

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem g_neg_two : g (-2) = 15 := by
  sorry

end g_neg_two_l1858_185805


namespace slope_positive_for_a_in_open_unit_interval_l1858_185809

theorem slope_positive_for_a_in_open_unit_interval :
  ∀ a : ℝ, 0 < a ∧ a < 1 →
  let k := -(2^a - 1) / Real.log a
  k > 0 := by
sorry

end slope_positive_for_a_in_open_unit_interval_l1858_185809


namespace f_is_even_l1858_185851

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_is_even_l1858_185851


namespace line_slope_intercept_sum_l1858_185861

/-- A line with slope 4 passing through (2, -1) has m + b = -5 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℝ), 
    (m = 4) →  -- Given slope
    (-1 = 4 * 2 + b) →  -- Line passes through (2, -1)
    (m + b = -5) :=  -- Conclusion to prove
by
  sorry  -- Proof omitted

end line_slope_intercept_sum_l1858_185861


namespace inequality_proof_l1858_185807

theorem inequality_proof (n : ℕ+) (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c > 0) : 
  ((a * b + c)^n.val - c) / ((b + c)^n.val - c) ≤ a^n.val := by
  sorry

end inequality_proof_l1858_185807


namespace quadratic_fit_energy_production_l1858_185854

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Theorem: There exists a quadratic function that fits the given data points
    and predicts the correct value for 2007 -/
theorem quadratic_fit_energy_production : ∃ f : QuadraticFunction,
  f.evaluate 0 = 8.6 ∧
  f.evaluate 5 = 10.4 ∧
  f.evaluate 10 = 12.9 ∧
  f.evaluate 15 = 16.1 := by
  sorry

end quadratic_fit_energy_production_l1858_185854


namespace f_expression_l1858_185802

-- Define the function f
def f : ℝ → ℝ := λ x => 2 * (x - 1) - 1

-- Theorem statement
theorem f_expression : ∀ x : ℝ, f x = 2 * x - 3 := by
  sorry

end f_expression_l1858_185802


namespace new_people_total_weight_l1858_185883

/-- Proves that the total weight of five new people joining a group is 270kg -/
theorem new_people_total_weight (initial_count : ℕ) (first_replacement_count : ℕ) (second_replacement_count : ℕ)
  (initial_average_increase : ℝ) (second_average_decrease : ℝ) 
  (first_outgoing_weights : Fin 3 → ℝ) (second_outgoing_total : ℝ) :
  initial_count = 20 ∧ 
  first_replacement_count = 3 ∧
  second_replacement_count = 2 ∧
  initial_average_increase = 2.5 ∧
  second_average_decrease = 1.8 ∧
  first_outgoing_weights 0 = 36 ∧
  first_outgoing_weights 1 = 48 ∧
  first_outgoing_weights 2 = 62 ∧
  second_outgoing_total = 110 →
  (initial_count : ℝ) * initial_average_increase + (first_outgoing_weights 0 + first_outgoing_weights 1 + first_outgoing_weights 2) +
  (second_outgoing_total - (initial_count : ℝ) * second_average_decrease) = 270 := by
  sorry

end new_people_total_weight_l1858_185883


namespace parabola_vertex_equation_l1858_185874

/-- A parabola with vertex coordinates (-2, 0) is represented by the equation y = (x+2)^2 -/
theorem parabola_vertex_equation :
  ∀ (x y : ℝ), (∃ (a : ℝ), y = a * (x + 2)^2) ↔ 
  (y = (x + 2)^2 ∧ (∀ (x₀ y₀ : ℝ), y₀ = (x₀ + 2)^2 → y₀ ≥ 0 ∧ (y₀ = 0 → x₀ = -2))) :=
by sorry

end parabola_vertex_equation_l1858_185874


namespace sum_of_numbers_l1858_185812

theorem sum_of_numbers (A B C : ℚ) : 
  (A / B = 2 / 5) → 
  (B / C = 4 / 7) → 
  (A = 16) → 
  (A + B + C = 126) := by
sorry

end sum_of_numbers_l1858_185812


namespace regular_polygon_angle_relation_l1858_185830

theorem regular_polygon_angle_relation (n : ℕ) : n ≥ 3 →
  (120 : ℝ) = 5 * (360 / n) → n = 15 := by
  sorry

end regular_polygon_angle_relation_l1858_185830


namespace g_neg_two_l1858_185858

def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem g_neg_two : g (-2) = -1 := by sorry

end g_neg_two_l1858_185858


namespace product_of_base_nine_digits_7654_l1858_185844

/-- Represents a number in base 9 as a list of digits -/
def BaseNineRepresentation := List Nat

/-- Converts a base 10 number to its base 9 representation -/
def toBaseNine (n : Nat) : BaseNineRepresentation :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List Nat) : Nat :=
  sorry

theorem product_of_base_nine_digits_7654 :
  productOfList (toBaseNine 7654) = 12 := by
  sorry

end product_of_base_nine_digits_7654_l1858_185844


namespace quadratic_root_problem_l1858_185843

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 7 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 7 = 0 ∧ y = -7/3) :=
by sorry

end quadratic_root_problem_l1858_185843


namespace problem_1_problem_2_problem_3_l1858_185821

-- Problem 1
theorem problem_1 (a : ℝ) : (-a^2)^3 + 9*a^4*a^2 = 8*a^6 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2*a*b^2 + a^2*b + b^3 = b*(a+b)^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 2*y) :
  (1/(x-y) - 1/(x+y)) / ((x-2*y)/((x^2)-(y^2))) = 2*y/(x-2*y) := by sorry

end problem_1_problem_2_problem_3_l1858_185821


namespace least_subtraction_for_divisibility_l1858_185825

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 13 ∧ (7538 - x) % 14 = 0 ∧ ∀ y : ℕ, y < x → (7538 - y) % 14 ≠ 0 :=
by
  -- The proof goes here
  sorry

end least_subtraction_for_divisibility_l1858_185825


namespace prime_factorization_equality_l1858_185860

theorem prime_factorization_equality : 5 * 13 * 31 - 2 = 3 * 11 * 61 := by
  sorry

end prime_factorization_equality_l1858_185860


namespace negation_of_universal_nonnegative_square_l1858_185838

theorem negation_of_universal_nonnegative_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end negation_of_universal_nonnegative_square_l1858_185838


namespace no_solution_iff_k_geq_two_l1858_185815

theorem no_solution_iff_k_geq_two (k : ℝ) :
  (∀ x : ℝ, ¬(1 < x ∧ x ≤ 2 ∧ x > k)) ↔ k ≥ 2 := by
  sorry

end no_solution_iff_k_geq_two_l1858_185815


namespace basketball_game_equations_l1858_185855

/-- Represents a basketball team's game results -/
structure BasketballTeam where
  gamesWon : ℕ
  gamesLost : ℕ

/-- Calculates the total points earned by a basketball team -/
def totalPoints (team : BasketballTeam) : ℕ :=
  2 * team.gamesWon + team.gamesLost

theorem basketball_game_equations (team : BasketballTeam) 
  (h1 : team.gamesWon + team.gamesLost = 12) 
  (h2 : totalPoints team = 20) : 
  (team.gamesWon + team.gamesLost = 12) ∧ (2 * team.gamesWon + team.gamesLost = 20) := by
  sorry

end basketball_game_equations_l1858_185855


namespace inequality_proof_l1858_185820

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x^2019 + y = 1) :
  x + y^2019 > 1 - 1/300 := by
  sorry

end inequality_proof_l1858_185820


namespace village_population_equality_l1858_185875

theorem village_population_equality (t : ℝ) (G : ℝ) : ¬(t > 0 ∧ 
  78000 - 1200 * t = 42000 + 800 * t ∧
  78000 - 1200 * t = 65000 + G * t ∧
  42000 + 800 * t = 65000 + G * t) :=
sorry

end village_population_equality_l1858_185875


namespace a2023_coordinates_l1858_185846

def companion_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2) + 1, p.1 + 1)

def sequence_point (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (2, 4)
  | n + 1 => companion_point (sequence_point n)

theorem a2023_coordinates :
  sequence_point 2022 = (-2, -2) :=
sorry

end a2023_coordinates_l1858_185846


namespace certain_number_value_l1858_185839

theorem certain_number_value (t b c : ℝ) (x : ℝ) :
  (t + b + c + 14 + x) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 →
  x = 15 := by
sorry

end certain_number_value_l1858_185839


namespace hypotenuse_length_triangle_area_l1858_185892

-- Define a right triangle with legs 30 and 40
def right_triangle (a b c : ℝ) : Prop :=
  a = 30 ∧ b = 40 ∧ c^2 = a^2 + b^2

-- Theorem for the hypotenuse
theorem hypotenuse_length (a b c : ℝ) (h : right_triangle a b c) : c = 50 := by
  sorry

-- Theorem for the area
theorem triangle_area (a b : ℝ) (h : a = 30 ∧ b = 40) : (1/2) * a * b = 600 := by
  sorry

end hypotenuse_length_triangle_area_l1858_185892


namespace pradeep_failed_by_25_marks_l1858_185828

/-- Calculates the number of marks by which a student failed, given the total marks,
    passing percentage, and the student's marks. -/
def marksFailed (totalMarks passingPercentage studentMarks : ℕ) : ℕ :=
  let passingMarks := totalMarks * passingPercentage / 100
  if studentMarks ≥ passingMarks then 0
  else passingMarks - studentMarks

/-- Theorem stating that Pradeep failed by 25 marks -/
theorem pradeep_failed_by_25_marks :
  marksFailed 840 25 185 = 25 := by
  sorry

end pradeep_failed_by_25_marks_l1858_185828


namespace factorial_fraction_equals_zero_l1858_185869

theorem factorial_fraction_equals_zero : 
  (5 * Nat.factorial 7 - 35 * Nat.factorial 6) / Nat.factorial 8 = 0 := by
  sorry

end factorial_fraction_equals_zero_l1858_185869


namespace curve_properties_l1858_185865

-- Define the function f(x) = x^3 - x
def f (x : ℝ) : ℝ := x^3 - x

-- State the theorem
theorem curve_properties :
  -- Part I: f'(1) = 2
  (deriv f) 1 = 2 ∧
  -- Part II: Tangent line equation at P(1, f(1)) is 2x - y - 2 = 0
  (∃ (m b : ℝ), m = 2 ∧ b = -2 ∧ ∀ x y, y = m * (x - 1) + f 1 ↔ 2 * x - y - 2 = 0) ∧
  -- Part III: Extreme values
  (∃ (x1 x2 : ℝ), 
    x1 = -Real.sqrt 3 / 3 ∧ 
    x2 = Real.sqrt 3 / 3 ∧
    f x1 = -2 * Real.sqrt 3 / 9 ∧
    f x2 = -2 * Real.sqrt 3 / 9 ∧
    (∀ x, f x ≥ -2 * Real.sqrt 3 / 9) ∧
    (∀ x, (deriv f) x = 0 → x = x1 ∨ x = x2)) :=
by sorry

end curve_properties_l1858_185865


namespace unwatered_bushes_l1858_185832

def total_bushes : ℕ := 2006

def bushes_watered_by_vitya (n : ℕ) : ℕ := n / 2
def bushes_watered_by_anya (n : ℕ) : ℕ := n / 2
def bushes_watered_by_both : ℕ := 3

theorem unwatered_bushes :
  total_bushes - (bushes_watered_by_vitya total_bushes + bushes_watered_by_anya total_bushes - bushes_watered_by_both) = 3 := by
  sorry

end unwatered_bushes_l1858_185832


namespace bacteria_habitat_limits_l1858_185880

/-- Represents a bacterial colony with its growth characteristics -/
structure BacterialColony where
  growthFactor : ℕ       -- How much the colony multiplies in size
  growthPeriod : ℕ       -- Number of days between each growth
  totalDays : ℕ          -- Total number of days the colony grows

/-- Calculates the number of days it takes for a colony to reach its habitat limit -/
def daysToHabitatLimit (colony : BacterialColony) : ℕ :=
  colony.totalDays

/-- Colony A doubles every day for 22 days -/
def colonyA : BacterialColony :=
  { growthFactor := 2
  , growthPeriod := 1
  , totalDays := 22 }

/-- Colony B triples every 2 days for 30 days -/
def colonyB : BacterialColony :=
  { growthFactor := 3
  , growthPeriod := 2
  , totalDays := 30 }

theorem bacteria_habitat_limits :
  daysToHabitatLimit colonyA = 22 ∧ daysToHabitatLimit colonyB = 30 := by
  sorry

#eval daysToHabitatLimit colonyA
#eval daysToHabitatLimit colonyB

end bacteria_habitat_limits_l1858_185880


namespace cubic_poly_b_value_l1858_185882

/-- Represents a cubic polynomial of the form x^3 - ax^2 + bx - b --/
def cubic_poly (a b : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + b*x - b

/-- Predicate to check if all roots of the polynomial are real and positive --/
def all_roots_real_positive (a b : ℝ) : Prop :=
  ∀ x : ℝ, cubic_poly a b x = 0 → x > 0

/-- The main theorem stating the value of b --/
theorem cubic_poly_b_value :
  ∃ (a : ℝ), a > 0 ∧
  (∀ a' : ℝ, a' > 0 → all_roots_real_positive a' (a'^2/3) → a ≤ a') ∧
  all_roots_real_positive a (a^2/3) ∧
  a^2/3 = 3 := by
sorry

end cubic_poly_b_value_l1858_185882


namespace triangle_area_approx_l1858_185891

/-- The area of a triangle with sides 30, 28, and 10 is approximately 139.94 -/
theorem triangle_area_approx : ∃ (area : ℝ), 
  let a : ℝ := 30
  let b : ℝ := 28
  let c : ℝ := 10
  let s : ℝ := (a + b + c) / 2
  area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ 
  abs (area - 139.94) < 0.01 := by
sorry

end triangle_area_approx_l1858_185891


namespace smallest_angle_in_triangle_l1858_185840

theorem smallest_angle_in_triangle (angle1 angle2 y : ℝ) : 
  angle1 = 60 → 
  angle2 = 65 → 
  angle1 + angle2 + y = 180 → 
  min angle1 (min angle2 y) = 55 :=
by sorry

end smallest_angle_in_triangle_l1858_185840


namespace anayet_driving_time_l1858_185859

/-- Proves that Anayet drove for 2 hours given the conditions of the problem -/
theorem anayet_driving_time 
  (total_distance : ℝ)
  (amoli_speed : ℝ)
  (amoli_time : ℝ)
  (anayet_speed : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_speed = 61)
  (h5 : remaining_distance = 121)
  : ∃ (anayet_time : ℝ), anayet_time = 2 ∧ 
    total_distance = amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance :=
by
  sorry


end anayet_driving_time_l1858_185859


namespace student_ticket_price_l1858_185890

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (general_tickets : ℕ) 
  (general_price : ℕ) 
  (h1 : total_tickets = 525)
  (h2 : total_revenue = 2876)
  (h3 : general_tickets = 388)
  (h4 : general_price = 6) :
  ∃ (student_price : ℕ),
    student_price = 4 ∧
    (total_tickets - general_tickets) * student_price + general_tickets * general_price = total_revenue :=
by sorry

end student_ticket_price_l1858_185890


namespace rectangle_area_with_hole_l1858_185837

theorem rectangle_area_with_hole (x : ℝ) : 
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3) = -x^2 + 24*x + 36 := by
  sorry

end rectangle_area_with_hole_l1858_185837


namespace shoes_sold_shoes_sold_is_six_l1858_185884

theorem shoes_sold (shoe_price : ℕ) (shirt_price : ℕ) (num_shirts : ℕ) (individual_earnings : ℕ) : ℕ :=
  let total_earnings := 2 * individual_earnings
  let shirt_earnings := shirt_price * num_shirts
  let shoe_earnings := total_earnings - shirt_earnings
  shoe_earnings / shoe_price

theorem shoes_sold_is_six : shoes_sold 3 2 18 27 = 6 := by
  sorry

end shoes_sold_shoes_sold_is_six_l1858_185884


namespace fraction_equality_implies_x_equals_one_l1858_185886

theorem fraction_equality_implies_x_equals_one :
  ∀ x : ℚ, (5 + x) / (7 + x) = (2 + x) / (3 + x) → x = 1 := by
  sorry

end fraction_equality_implies_x_equals_one_l1858_185886


namespace standard_deviation_of_random_variable_l1858_185849

def random_variable (ξ : ℝ → ℝ) : Prop :=
  (ξ 1 = 0.4) ∧ (ξ 3 = 0.1) ∧ (∃ x, ξ 5 = x) ∧ (ξ 1 + ξ 3 + ξ 5 = 1)

def expected_value (ξ : ℝ → ℝ) : ℝ :=
  1 * ξ 1 + 3 * ξ 3 + 5 * ξ 5

def variance (ξ : ℝ → ℝ) : ℝ :=
  (1 - expected_value ξ)^2 * ξ 1 + 
  (3 - expected_value ξ)^2 * ξ 3 + 
  (5 - expected_value ξ)^2 * ξ 5

theorem standard_deviation_of_random_variable (ξ : ℝ → ℝ) :
  random_variable ξ → Real.sqrt (variance ξ) = Real.sqrt 3.56 := by
  sorry

end standard_deviation_of_random_variable_l1858_185849


namespace melanie_dimes_l1858_185888

theorem melanie_dimes (x : ℕ) : x + 8 + 4 = 19 → x = 7 := by
  sorry

end melanie_dimes_l1858_185888


namespace jerrys_books_l1858_185829

/-- Given Jerry's initial and additional books, prove the total number of books. -/
theorem jerrys_books (initial_books additional_books : ℕ) :
  initial_books = 9 → additional_books = 10 → initial_books + additional_books = 19 :=
by sorry

end jerrys_books_l1858_185829


namespace annie_hamburgers_l1858_185881

theorem annie_hamburgers (initial_amount : ℕ) (hamburger_cost : ℕ) (milkshake_cost : ℕ)
  (milkshakes_bought : ℕ) (amount_left : ℕ) :
  initial_amount = 120 →
  hamburger_cost = 4 →
  milkshake_cost = 3 →
  milkshakes_bought = 6 →
  amount_left = 70 →
  ∃ (hamburgers_bought : ℕ),
    hamburgers_bought = 8 ∧
    initial_amount = amount_left + hamburger_cost * hamburgers_bought + milkshake_cost * milkshakes_bought :=
by
  sorry

end annie_hamburgers_l1858_185881


namespace inequality_solution_set_l1858_185879

/-- The solution set of the inequality (x-2)(ax-2) > 0 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Iio 2
  else if a < 0 then Set.Ioo (2/a) 2
  else if 0 < a ∧ a < 1 then Set.Iio 2 ∪ Set.Ioi (2/a)
  else if a > 1 then Set.Iio (2/a) ∪ Set.Ioi 2
  else Set.Iio 2 ∪ Set.Ioi 2

theorem inequality_solution_set (a : ℝ) (x : ℝ) :
  (x - 2) * (a * x - 2) > 0 ↔ x ∈ solution_set a :=
sorry

end inequality_solution_set_l1858_185879


namespace greatest_value_quadratic_inequality_l1858_185863

theorem greatest_value_quadratic_inequality :
  ∃ (a_max : ℝ), a_max = 9 ∧
  (∀ a : ℝ, a^2 - 14*a + 45 ≤ 0 → a ≤ a_max) ∧
  (a_max^2 - 14*a_max + 45 ≤ 0) :=
by sorry

end greatest_value_quadratic_inequality_l1858_185863


namespace sqrt_equation_solution_l1858_185896

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (7 + 3 * z) = 13 :=
by
  -- Proof goes here
  sorry

end sqrt_equation_solution_l1858_185896


namespace ticket_sales_l1858_185866

theorem ticket_sales (adult_price children_price senior_price discount : ℕ)
  (total_receipts total_attendance : ℕ)
  (discounted_adults discounted_children : ℕ)
  (h1 : adult_price = 25)
  (h2 : children_price = 15)
  (h3 : senior_price = 20)
  (h4 : discount = 5)
  (h5 : discounted_adults = 50)
  (h6 : discounted_children = 30)
  (h7 : total_receipts = 7200)
  (h8 : total_attendance = 400) :
  ∃ (regular_adults regular_children senior : ℕ),
    regular_adults + discounted_adults = 2 * senior ∧
    regular_adults + discounted_adults + regular_children + discounted_children + senior = total_attendance ∧
    regular_adults * adult_price + discounted_adults * (adult_price - discount) +
    regular_children * children_price + discounted_children * (children_price - discount) +
    senior * senior_price = total_receipts ∧
    regular_adults = 102 ∧
    regular_children = 142 ∧
    senior = 76 := by
  sorry

end ticket_sales_l1858_185866


namespace smallest_factorization_coefficient_l1858_185850

theorem smallest_factorization_coefficient : ∃ (r s : ℕ+), 
  (r : ℤ) * s = 1620 ∧ 
  r + s = 84 ∧ 
  (∀ (r' s' : ℕ+), (r' : ℤ) * s' = 1620 → r' + s' ≥ 84) := by
  sorry

end smallest_factorization_coefficient_l1858_185850


namespace percentage_of_students_owning_only_cats_l1858_185848

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ)
  (dog_owners : ℕ)
  (cat_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : dog_owners = 150)
  (h3 : cat_owners = 80)
  (h4 : both_owners = 25) :
  (cat_owners - both_owners) / total_students * 100 = 11 := by
sorry

end percentage_of_students_owning_only_cats_l1858_185848


namespace baking_ingredient_calculation_l1858_185853

/-- Represents the ingredients needed for baking --/
structure BakingIngredients where
  flour_cake : ℝ
  flour_cookies : ℝ
  sugar_cake : ℝ
  sugar_cookies : ℝ

/-- Represents the available ingredients --/
structure AvailableIngredients where
  flour : ℝ
  sugar : ℝ

/-- Calculates the difference between available and needed ingredients --/
def ingredientDifference (needed : BakingIngredients) (available : AvailableIngredients) : 
  ℝ × ℝ :=
  let total_flour_needed := needed.flour_cake + needed.flour_cookies
  let total_sugar_needed := needed.sugar_cake + needed.sugar_cookies
  (available.flour - total_flour_needed, available.sugar - total_sugar_needed)

theorem baking_ingredient_calculation 
  (needed : BakingIngredients) 
  (available : AvailableIngredients) : 
  needed.flour_cake = 6 ∧ 
  needed.flour_cookies = 2 ∧ 
  needed.sugar_cake = 3.5 ∧ 
  needed.sugar_cookies = 1.5 ∧
  available.flour = 8 ∧ 
  available.sugar = 4 → 
  ingredientDifference needed available = (0, -1) := by
  sorry

end baking_ingredient_calculation_l1858_185853


namespace plane_equation_theorem_l1858_185842

/-- The equation of a plane given its normal vector and a point on the plane -/
def plane_equation (normal : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ :=
  sorry

/-- Check if the first coefficient is positive -/
def first_coeff_positive (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  sorry

/-- Calculate the GCD of the absolute values of all coefficients -/
def gcd_of_coeffs (coeffs : ℤ × ℤ × ℤ × ℤ) : ℕ :=
  sorry

theorem plane_equation_theorem :
  let normal : ℝ × ℝ × ℝ := (10, -5, 6)
  let point : ℝ × ℝ × ℝ := (10, -5, 6)
  let coeffs := plane_equation normal point
  first_coeff_positive coeffs ∧ gcd_of_coeffs coeffs = 1 ∧ coeffs = (10, -5, 6, -161) :=
by sorry

end plane_equation_theorem_l1858_185842


namespace cubs_cardinals_home_run_difference_l1858_185824

/-- Represents the number of home runs scored by a team in each inning -/
structure HomeRuns :=
  (third : ℕ)
  (fifth : ℕ)
  (eighth : ℕ)

/-- Represents the number of home runs scored by the opposing team in each inning -/
structure OpponentHomeRuns :=
  (second : ℕ)
  (fifth : ℕ)

/-- The difference in home runs between the Cubs and the Cardinals -/
def homRunDifference (cubs : HomeRuns) (cardinals : OpponentHomeRuns) : ℕ :=
  (cubs.third + cubs.fifth + cubs.eighth) - (cardinals.second + cardinals.fifth)

theorem cubs_cardinals_home_run_difference :
  ∀ (cubs : HomeRuns) (cardinals : OpponentHomeRuns),
    cubs.third = 2 → cubs.fifth = 1 → cubs.eighth = 2 →
    cardinals.second = 1 → cardinals.fifth = 1 →
    homRunDifference cubs cardinals = 3 :=
by
  sorry

#check cubs_cardinals_home_run_difference

end cubs_cardinals_home_run_difference_l1858_185824


namespace paint_combinations_l1858_185831

theorem paint_combinations (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (n.choose m) * k^m = 960 := by
  sorry

end paint_combinations_l1858_185831


namespace old_card_sale_amount_l1858_185872

def initial_cost : ℕ := 1200
def new_card_cost : ℕ := 500
def total_spent : ℕ := 1400

theorem old_card_sale_amount : 
  initial_cost + new_card_cost - total_spent = 300 :=
by sorry

end old_card_sale_amount_l1858_185872


namespace reflect_point_across_y_axis_l1858_185857

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  ⟨-p.x, p.y⟩

theorem reflect_point_across_y_axis :
  let P : Point := ⟨5, -1⟩
  reflectAcrossYAxis P = ⟨-5, -1⟩ := by
  sorry

end reflect_point_across_y_axis_l1858_185857


namespace mike_song_book_price_l1858_185868

/-- The amount Mike received from selling the song book, given the cost of the trumpet and the net amount spent. -/
def song_book_price (trumpet_cost net_spent : ℚ) : ℚ :=
  trumpet_cost - net_spent

/-- Theorem stating that Mike sold the song book for $5.84, given the cost of the trumpet and the net amount spent. -/
theorem mike_song_book_price :
  let trumpet_cost : ℚ := 145.16
  let net_spent : ℚ := 139.32
  song_book_price trumpet_cost net_spent = 5.84 := by
  sorry

#eval song_book_price 145.16 139.32

end mike_song_book_price_l1858_185868


namespace find_m_l1858_185889

theorem find_m : ∃ m : ℤ, (|m| = 2 ∧ m - 2 ≠ 0) → m = -2 := by
  sorry

end find_m_l1858_185889


namespace restaurant_bill_calculation_l1858_185878

/-- Calculates the total cost for a group at a restaurant given specific pricing and group composition. -/
theorem restaurant_bill_calculation 
  (adult_meal_cost : ℚ)
  (kid_meal_cost : ℚ)
  (adult_drink_cost : ℚ)
  (kid_drink_cost : ℚ)
  (dessert_cost : ℚ)
  (total_people : ℕ)
  (num_adults : ℕ)
  (num_children : ℕ)
  (h1 : adult_meal_cost = 12)
  (h2 : kid_meal_cost = 0)
  (h3 : adult_drink_cost = 5/2)
  (h4 : kid_drink_cost = 3/2)
  (h5 : dessert_cost = 4)
  (h6 : total_people = 11)
  (h7 : num_adults = 7)
  (h8 : num_children = 4)
  (h9 : total_people = num_adults + num_children) :
  (num_adults * adult_meal_cost) +
  (num_adults * adult_drink_cost) +
  (num_children * kid_drink_cost) +
  (total_people * dessert_cost) = 151.5 := by
    sorry

end restaurant_bill_calculation_l1858_185878


namespace gwen_birthday_money_l1858_185814

/-- Given that Gwen received 14 dollars for her birthday and spent 8 dollars,
    prove that she has 6 dollars left. -/
theorem gwen_birthday_money (received : ℕ) (spent : ℕ) (left : ℕ) : 
  received = 14 → spent = 8 → left = received - spent → left = 6 := by
  sorry

end gwen_birthday_money_l1858_185814


namespace min_throws_for_repeated_sum_l1858_185813

/-- The number of dice being thrown -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 4

/-- The minimum possible sum when throwing the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when throwing the dice -/
def max_sum : ℕ := num_dice * sides_per_die

/-- The number of possible unique sums -/
def unique_sums : ℕ := max_sum - min_sum + 1

/-- The minimum number of throws required to ensure a repeated sum -/
def min_throws : ℕ := unique_sums + 1

theorem min_throws_for_repeated_sum :
  min_throws = 14 :=
sorry

end min_throws_for_repeated_sum_l1858_185813


namespace prob_A_or_B_prob_A_prob_B_given_A_l1858_185819

-- Define the class composition
def total_officials : ℕ := 6
def male_officials : ℕ := 4
def female_officials : ℕ := 2
def selected_officials : ℕ := 3

-- Define the events
def event_A : Set (Fin total_officials) := sorry
def event_B : Set (Fin total_officials) := sorry

-- Define the probability measure
noncomputable def P : Set (Fin total_officials) → ℝ := sorry

-- Theorem statements
theorem prob_A_or_B : P (event_A ∪ event_B) = 4/5 := by sorry

theorem prob_A : P event_A = 1/2 := by sorry

theorem prob_B_given_A : P (event_B ∩ event_A) / P event_A = 2/5 := by sorry

end prob_A_or_B_prob_A_prob_B_given_A_l1858_185819


namespace intersection_of_M_and_N_l1858_185826

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x * (x + 2) ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end intersection_of_M_and_N_l1858_185826


namespace increase_by_percentage_increase_40_by_150_percent_l1858_185877

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial + (percentage / 100) * initial = initial * (1 + percentage / 100) := by sorry

theorem increase_40_by_150_percent :
  40 + (150 / 100) * 40 = 100 := by sorry

end increase_by_percentage_increase_40_by_150_percent_l1858_185877


namespace min_reciprocal_sum_l1858_185862

theorem min_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 1/a + 1/b + 1/c ≥ 3 := by
  sorry

end min_reciprocal_sum_l1858_185862


namespace idle_days_is_37_l1858_185804

/-- Represents the worker's payment scenario -/
structure WorkerPayment where
  totalDays : ℕ
  workPayRate : ℕ
  idleForfeitRate : ℕ
  totalReceived : ℕ

/-- Calculates the number of idle days given a WorkerPayment scenario -/
def calculateIdleDays (wp : WorkerPayment) : ℕ :=
  let totalEarning := wp.workPayRate * wp.totalDays
  let totalLoss := wp.totalReceived - totalEarning
  totalLoss / (wp.workPayRate + wp.idleForfeitRate)

/-- Theorem stating that for the given scenario, the number of idle days is 37 -/
theorem idle_days_is_37 (wp : WorkerPayment) 
    (h1 : wp.totalDays = 60)
    (h2 : wp.workPayRate = 30)
    (h3 : wp.idleForfeitRate = 5)
    (h4 : wp.totalReceived = 500) :
    calculateIdleDays wp = 37 := by
  sorry

#eval calculateIdleDays ⟨60, 30, 5, 500⟩

end idle_days_is_37_l1858_185804


namespace min_xy_value_l1858_185867

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : 
  x * y ≥ 180 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 10 * x₀ + 2 * y₀ + 60 = x₀ * y₀ ∧ x₀ * y₀ = 180 := by
  sorry

end min_xy_value_l1858_185867


namespace sin_cos_sum_20_40_l1858_185873

theorem sin_cos_sum_20_40 :
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) +
  Real.cos (20 * π / 180) * Real.sin (40 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_sum_20_40_l1858_185873


namespace xy_nonneg_iff_abs_sum_eq_sum_abs_l1858_185827

theorem xy_nonneg_iff_abs_sum_eq_sum_abs (x y : ℝ) : x * y ≥ 0 ↔ |x + y| = |x| + |y| := by
  sorry

end xy_nonneg_iff_abs_sum_eq_sum_abs_l1858_185827


namespace digits_of_8_power_10_times_3_power_15_l1858_185871

theorem digits_of_8_power_10_times_3_power_15 : ∃ (n : ℕ), 
  (10 ^ (n - 1) ≤ 8^10 * 3^15) ∧ (8^10 * 3^15 < 10^n) ∧ (n = 12) := by
  sorry

end digits_of_8_power_10_times_3_power_15_l1858_185871


namespace tape_recorder_cost_l1858_185870

theorem tape_recorder_cost : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧
  170 ≤ x * y ∧ x * y ≤ 195 ∧
  y = 2 * x + 2 ∧
  x * y = 180 := by
  sorry

end tape_recorder_cost_l1858_185870


namespace sum_of_fraction_and_decimal_l1858_185822

theorem sum_of_fraction_and_decimal : (1 : ℚ) / 25 + (25 : ℚ) / 100 = (29 : ℚ) / 100 := by
  sorry

end sum_of_fraction_and_decimal_l1858_185822


namespace absolute_value_of_w_l1858_185897

theorem absolute_value_of_w (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end absolute_value_of_w_l1858_185897


namespace line_tangent_to_circle_l1858_185818

theorem line_tangent_to_circle (m n : ℝ) : 
  (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 ≥ 1) ∧
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 ∧ (x - 1)^2 + (y - 1)^2 = 1) →
  m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end line_tangent_to_circle_l1858_185818


namespace count_even_positive_factors_l1858_185847

/-- The number of even positive factors of n, where n = 2^4 * 3^2 * 5^2 * 7 -/
def evenPositiveFactors (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of even positive factors of n is 72 -/
theorem count_even_positive_factors :
  ∃ n : ℕ, n = 2^4 * 3^2 * 5^2 * 7 ∧ evenPositiveFactors n = 72 :=
sorry

end count_even_positive_factors_l1858_185847


namespace at_op_difference_l1858_185817

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - x - 2 * y

-- State the theorem
theorem at_op_difference : (at_op 7 4) - (at_op 4 7) = 3 := by sorry

end at_op_difference_l1858_185817


namespace max_pairs_after_loss_l1858_185803

/-- Represents the number of matching pairs of shoes after losing some shoes. -/
def MaxPairsAfterLoss (totalPairs : ℕ) (colors : ℕ) (sizes : ℕ) (shoesLost : ℕ) : ℕ :=
  min (totalPairs - shoesLost) (colors * sizes)

/-- Theorem stating the maximum number of matching pairs after losing shoes. -/
theorem max_pairs_after_loss :
  MaxPairsAfterLoss 23 6 3 9 = 14 := by
  sorry

#eval MaxPairsAfterLoss 23 6 3 9

end max_pairs_after_loss_l1858_185803


namespace scale_length_90_inches_l1858_185833

/-- Given a scale divided into equal parts, calculates the total length of the scale. -/
def scale_length (num_parts : ℕ) (part_length : ℕ) : ℕ :=
  num_parts * part_length

/-- Theorem stating that a scale with 5 parts of 18 inches each has a total length of 90 inches. -/
theorem scale_length_90_inches :
  scale_length 5 18 = 90 := by
  sorry

end scale_length_90_inches_l1858_185833


namespace fish_distribution_l1858_185810

theorem fish_distribution (total_fish : ℕ) (num_bowls : ℕ) (fish_per_bowl : ℕ) :
  total_fish = 6003 →
  num_bowls = 261 →
  total_fish = num_bowls * fish_per_bowl →
  fish_per_bowl = 23 := by
  sorry

end fish_distribution_l1858_185810


namespace triangle_properties_l1858_185856

-- Define a triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) :
  (2 * Real.sin t.B * Real.cos t.A = Real.sin t.A * Real.cos t.C + Real.cos t.A * Real.sin t.C) →
  (t.A = π / 3) ∧
  (t.A = π / 3 ∧ t.a = 6) →
  ∃ p : Real, 12 < p ∧ p ≤ 18 ∧ p = t.a + t.b + t.c :=
by sorry

end triangle_properties_l1858_185856


namespace log_five_eighteen_l1858_185806

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_five_eighteen (a b : ℝ) 
  (h1 : log 10 2 = a) 
  (h2 : log 10 3 = b) : 
  log 5 18 = (a + 2*b) / (1 - a) := by
  sorry

end log_five_eighteen_l1858_185806


namespace expected_value_of_heads_l1858_185895

/-- Represents the different types of coins -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents -/
def coinValue (c : Coin) : ℚ :=
  match c with
  | .Penny => 1
  | .Nickel => 5
  | .Dime => 10
  | .Quarter => 25
  | .HalfDollar => 50

/-- Returns the probability of a coin landing heads -/
def headsProbability (c : Coin) : ℚ :=
  match c with
  | .HalfDollar => 1/3
  | _ => 1/2

/-- The set of all coins -/
def coinSet : List Coin := [Coin.Penny, Coin.Nickel, Coin.Dime, Coin.Quarter, Coin.HalfDollar]

/-- Calculates the expected value for a single coin -/
def expectedValue (c : Coin) : ℚ := (headsProbability c) * (coinValue c)

/-- Theorem: The expected value of the amount of money from coins that come up heads is 223/6 cents -/
theorem expected_value_of_heads : 
  (coinSet.map expectedValue).sum = 223/6 := by sorry

end expected_value_of_heads_l1858_185895


namespace marks_remaining_money_l1858_185893

def initial_money : ℕ := 85
def num_books : ℕ := 10
def book_cost : ℕ := 5

theorem marks_remaining_money :
  initial_money - (num_books * book_cost) = 35 := by
  sorry

end marks_remaining_money_l1858_185893


namespace recurring_decimal_difference_l1858_185864

theorem recurring_decimal_difference : 
  let x : ℚ := 8/11  -- 0.overline{72}
  let y : ℚ := 18/25 -- 0.72
  x - y = 2/275 := by
sorry

end recurring_decimal_difference_l1858_185864


namespace polygon_sides_l1858_185800

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end polygon_sides_l1858_185800


namespace vector_AB_l1858_185841

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define the vector OA
def OA : Vector2D := (2, 8)

-- Define the vector OB
def OB : Vector2D := (-7, 2)

-- Define vector subtraction
def vectorSub (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Theorem statement
theorem vector_AB (OA OB : Vector2D) (h1 : OA = (2, 8)) (h2 : OB = (-7, 2)) :
  vectorSub OB OA = (-9, -6) := by
  sorry

end vector_AB_l1858_185841


namespace fraction_zero_implies_x_zero_l1858_185898

theorem fraction_zero_implies_x_zero (x : ℝ) : 
  (x^2 - x) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = 0 := by
  sorry

end fraction_zero_implies_x_zero_l1858_185898


namespace parabola_parameter_l1858_185887

/-- A parabola with equation y = ax² and latus rectum y = -1/2 has a = 1/2 --/
theorem parabola_parameter (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x^2) ∧  -- Parabola equation
  (∃ (y : ℝ), y = -1/2) →       -- Latus rectum equation
  a = 1/2 := by sorry

end parabola_parameter_l1858_185887


namespace second_track_has_30_checkpoints_l1858_185836

/-- The number of checkpoints on the first track -/
def first_track_checkpoints : ℕ := 6

/-- The total number of ways to form triangles -/
def total_triangles : ℕ := 420

/-- The number of checkpoints on the second track -/
def second_track_checkpoints : ℕ := 30

/-- Theorem stating that the number of checkpoints on the second track is 30 -/
theorem second_track_has_30_checkpoints :
  (first_track_checkpoints * (second_track_checkpoints.choose 2) = total_triangles) →
  second_track_checkpoints = 30 := by
  sorry

end second_track_has_30_checkpoints_l1858_185836


namespace existence_of_c_l1858_185894

theorem existence_of_c (a b : ℝ) : ∃ c ∈ Set.Icc 0 1, |a * c + b + 1 / (c + 1)| ≥ 1 / 24 := by
  sorry

end existence_of_c_l1858_185894


namespace inverse_variation_problem_l1858_185899

/-- Given that 8y varies inversely as the cube of x, and y = 25 when x = 2,
    prove that y = 25/8 when x = 4. -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, 8 * y x = k / x^3) →  -- 8y varies inversely as the cube of x
  y 2 = 25 →                 -- y = 25 when x = 2
  y 4 = 25 / 8 :=             -- y = 25/8 when x = 4
by sorry

end inverse_variation_problem_l1858_185899


namespace vector_operations_l1858_185801

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (-1, 3)

theorem vector_operations :
  (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) = (1, 3) ∧
  (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) = (3, -3) := by
  sorry

end vector_operations_l1858_185801


namespace closest_year_to_target_population_l1858_185816

-- Define the population function
def population (initial : ℕ) (year : ℕ) : ℕ :=
  initial * 2^((year - 2010) / 20)

-- Define a function to calculate the difference from target population
def diff_from_target (target : ℕ) (year : ℕ) : ℕ :=
  let pop := population 500 year
  if pop ≥ target then pop - target else target - pop

-- State the theorem
theorem closest_year_to_target_population :
  (∀ y : ℕ, y ≥ 2010 → diff_from_target 10000 2090 ≤ diff_from_target 10000 y) :=
sorry

end closest_year_to_target_population_l1858_185816


namespace paper_size_problem_l1858_185811

theorem paper_size_problem (L : ℝ) :
  (L > 0) →
  (2 * (L * 11) = 2 * (5.5 * 11) + 100) →
  L = 10 := by
sorry

end paper_size_problem_l1858_185811


namespace news_program_selection_methods_l1858_185835

theorem news_program_selection_methods (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 8 → k = 4 → m = 2 →
  (n.choose k) * (k.choose m) * (m.factorial) = 840 := by
  sorry

end news_program_selection_methods_l1858_185835


namespace tan_sum_product_identity_l1858_185834

open Real

theorem tan_sum_product_identity (α β γ : ℝ) : 
  0 < α ∧ α < π/2 ∧ 
  0 < β ∧ β < π/2 ∧ 
  0 < γ ∧ γ < π/2 ∧ 
  α + β + γ = π/2 ∧ 
  (∀ k : ℤ, α ≠ k * π + π/2) ∧
  (∀ k : ℤ, β ≠ k * π + π/2) ∧
  (∀ k : ℤ, γ ≠ k * π + π/2) →
  tan α * tan β + tan β * tan γ + tan γ * tan α = 1 := by
  sorry

end tan_sum_product_identity_l1858_185834
