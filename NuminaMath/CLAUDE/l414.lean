import Mathlib

namespace largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l414_41412

theorem largest_integer_negative_quadratic : 
  ∀ n : ℤ, n^2 - 11*n + 24 < 0 → n ≤ 7 :=
by sorry

theorem seven_satisfies_inequality : 
  (7 : ℤ)^2 - 11*7 + 24 < 0 :=
by sorry

theorem eight_does_not_satisfy_inequality : 
  (8 : ℤ)^2 - 11*8 + 24 ≥ 0 :=
by sorry

end largest_integer_negative_quadratic_seven_satisfies_inequality_eight_does_not_satisfy_inequality_l414_41412


namespace polynomial_equality_l414_41463

theorem polynomial_equality (a b c : ℝ) : 
  (∀ x : ℝ, x * (x + 1) = a + b * x + c * x^2) → a + b + c = 2 := by
  sorry

end polynomial_equality_l414_41463


namespace unique_digit_divisibility_l414_41480

theorem unique_digit_divisibility : 
  ∃! n : ℕ, 0 < n ∧ n ≤ 9 ∧ 
  100 ≤ 25 * n ∧ 25 * n ≤ 999 ∧ 
  (25 * n) % n = 0 ∧ (25 * n) % 5 = 0 := by
sorry

end unique_digit_divisibility_l414_41480


namespace simple_interest_difference_l414_41495

/-- Calculate the simple interest and prove that it's Rs. 306 less than the principal -/
theorem simple_interest_difference (principal rate time : ℝ) : 
  principal = 450 → 
  rate = 4 → 
  time = 8 → 
  principal - (principal * rate * time / 100) = 306 := by
sorry

end simple_interest_difference_l414_41495


namespace new_car_cost_proof_l414_41470

/-- The monthly cost of renting a car -/
def rental_cost : ℕ := 20

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total difference in cost over a year between renting and buying -/
def total_difference : ℕ := 120

/-- The monthly cost of the new car -/
def new_car_cost : ℕ := 30

theorem new_car_cost_proof : 
  new_car_cost * months_in_year - rental_cost * months_in_year = total_difference := by
  sorry

end new_car_cost_proof_l414_41470


namespace probability_not_pair_is_four_fifths_l414_41486

def num_pairs : ℕ := 3
def num_shoes : ℕ := 2 * num_pairs

def probability_not_pair : ℚ :=
  1 - (num_pairs * 1 : ℚ) / (num_shoes.choose 2 : ℚ)

theorem probability_not_pair_is_four_fifths :
  probability_not_pair = 4/5 := by
  sorry

end probability_not_pair_is_four_fifths_l414_41486


namespace square_sum_xy_l414_41482

theorem square_sum_xy (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : 1 / (x^2) + 1 / (y^2) = a) : 
  (x + y)^2 = b * (a * b + 2) := by
sorry

end square_sum_xy_l414_41482


namespace student_competition_assignments_l414_41440

/-- The number of ways to assign students to competitions -/
def num_assignments (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- Theorem: For 4 students and 3 competitions, there are 3^4 different assignment outcomes -/
theorem student_competition_assignments :
  num_assignments 4 3 = 3^4 := by
  sorry

end student_competition_assignments_l414_41440


namespace sum_of_roots_is_negative_4015_l414_41422

/-- Represents the polynomial (x-1)^2009 + 3(x-2)^2008 + 5(x-3)^2007 + ⋯ + 4017(x-2009)^2 + 4019(x-4018) -/
def specialPolynomial : Polynomial ℝ := sorry

/-- The sum of the roots of the specialPolynomial -/
def sumOfRoots : ℝ := sorry

/-- Theorem stating that the sum of the roots of the specialPolynomial is -4015 -/
theorem sum_of_roots_is_negative_4015 : sumOfRoots = -4015 := by sorry

end sum_of_roots_is_negative_4015_l414_41422


namespace expression_evaluation_l414_41498

theorem expression_evaluation (x y z : ℚ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  (1/y + 1/z) / (1/x) = 35/12 := by
  sorry

end expression_evaluation_l414_41498


namespace parabola_focus_theorem_l414_41410

/-- Represents a parabola y² = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of the parabola lies on the line x + y - 1 = 0 -/
def focus_on_line (para : Parabola) (F : Point) : Prop :=
  F.x + F.y = 1

/-- The equation of the parabola is y² = 4x -/
def parabola_equation (para : Parabola) : Prop :=
  para.p = 2

/-- A line through the focus at 45° angle -/
def line_through_focus (F : Point) (A B : Point) : Prop :=
  (A.y - F.y) = (A.x - F.x) ∧ (B.y - F.y) = (B.x - F.x)

/-- A and B are on the parabola -/
def points_on_parabola (para : Parabola) (A B : Point) : Prop :=
  A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x

/-- The length of AB is 8 -/
def length_AB (A B : Point) : Prop :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8

theorem parabola_focus_theorem (para : Parabola) (F A B : Point) :
  focus_on_line para F →
  line_through_focus F A B →
  points_on_parabola para A B →
  parabola_equation para ∧ length_AB A B :=
sorry

end parabola_focus_theorem_l414_41410


namespace f_of_3_equals_29_l414_41427

/-- Given f(x) = x^2 + 4x + 8, prove that f(3) = 29 -/
theorem f_of_3_equals_29 :
  let f : ℝ → ℝ := λ x ↦ x^2 + 4*x + 8
  f 3 = 29 := by sorry

end f_of_3_equals_29_l414_41427


namespace complex_square_sum_l414_41452

theorem complex_square_sum (a b : ℝ) : 
  (Complex.mk a b = (1 + Complex.I)^2) → a + b = 2 := by
  sorry

end complex_square_sum_l414_41452


namespace discount_percentage_l414_41457

theorem discount_percentage (num_tickets : ℕ) (price_per_ticket : ℚ) (total_spent : ℚ) : 
  num_tickets = 24 →
  price_per_ticket = 7 →
  total_spent = 84 →
  (1 - total_spent / (num_tickets * price_per_ticket)) * 100 = 50 := by
  sorry

end discount_percentage_l414_41457


namespace bart_earnings_l414_41419

/-- The amount Bart earns per question in dollars -/
def earnings_per_question : ℚ := 0.2

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart completed on Monday -/
def monday_surveys : ℕ := 3

/-- The number of surveys Bart completed on Tuesday -/
def tuesday_surveys : ℕ := 4

/-- Theorem stating Bart's total earnings over two days -/
theorem bart_earnings : 
  (earnings_per_question * questions_per_survey * (monday_surveys + tuesday_surveys) : ℚ) = 14 := by
  sorry

end bart_earnings_l414_41419


namespace parabola_c_value_l414_41446

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1,5) and (5,5). 
    The value of c is 15. -/
theorem parabola_c_value : ∀ b c : ℝ, 
  (5 = 2 * (1 : ℝ)^2 + b * 1 + c) → 
  (5 = 2 * (5 : ℝ)^2 + b * 5 + c) → 
  c = 15 := by
  sorry

end parabola_c_value_l414_41446


namespace juan_has_498_marbles_l414_41406

/-- The number of marbles Connie has -/
def connies_marbles : ℕ := 323

/-- The number of additional marbles Juan has compared to Connie -/
def juans_additional_marbles : ℕ := 175

/-- The total number of marbles Juan has -/
def juans_marbles : ℕ := connies_marbles + juans_additional_marbles

/-- Theorem stating that Juan has 498 marbles -/
theorem juan_has_498_marbles : juans_marbles = 498 := by
  sorry

end juan_has_498_marbles_l414_41406


namespace work_completion_theorem_l414_41461

theorem work_completion_theorem (days_group1 days_group2 : ℕ) 
  (men_group2 : ℕ) (total_work : ℕ) :
  days_group1 = 18 →
  days_group2 = 8 →
  men_group2 = 81 →
  total_work = men_group2 * days_group2 →
  ∃ men_group1 : ℕ, men_group1 * days_group1 = total_work ∧ men_group1 = 36 :=
by
  sorry

end work_completion_theorem_l414_41461


namespace polynomial_division_remainder_l414_41487

-- Define the polynomial and the divisor
def f (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2
def divisor (x : ℝ) : ℝ := (x^2 - 1) * (x - 2)

-- Define the remainder
def remainder (x : ℝ) : ℝ := 9 * x^2 - 8

-- Theorem statement
theorem polynomial_division_remainder :
  ∃ q : ℝ → ℝ, ∀ x : ℝ, f x = divisor x * q x + remainder x := by
  sorry

end polynomial_division_remainder_l414_41487


namespace number_ratio_l414_41472

theorem number_ratio (f s t : ℝ) : 
  t = 2 * f →
  (f + s + t) / 3 = 77 →
  f = 33 →
  s / f = 4 := by
sorry

end number_ratio_l414_41472


namespace stratified_sampling_pine_saplings_l414_41443

/-- Calculates the expected number of pine saplings in a stratified sample -/
def expected_pine_saplings (total_saplings : ℕ) (pine_saplings : ℕ) (sample_size : ℕ) : ℕ :=
  (pine_saplings * sample_size) / total_saplings

theorem stratified_sampling_pine_saplings :
  expected_pine_saplings 30000 4000 150 = 20 := by
  sorry

#eval expected_pine_saplings 30000 4000 150

end stratified_sampling_pine_saplings_l414_41443


namespace triangle_condition_implies_isosceles_right_l414_41468

/-- A triangle with sides a, b, c and circumradius R -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ

/-- The condition R(b+c) = a√(bc) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.R * (t.b + t.c) = t.a * Real.sqrt (t.b * t.c)

/-- Definition of an isosceles right triangle -/
def isIsoscelesRight (t : Triangle) : Prop :=
  t.a = t.b ∧ t.a * t.a + t.b * t.b = t.c * t.c

/-- The main theorem -/
theorem triangle_condition_implies_isosceles_right (t : Triangle) :
  satisfiesCondition t → isIsoscelesRight t :=
sorry

end triangle_condition_implies_isosceles_right_l414_41468


namespace arithmetic_evaluation_l414_41444

theorem arithmetic_evaluation : 68 + (108 * 3) + (29^2) - 310 - (6 * 9) = 869 := by
  sorry

end arithmetic_evaluation_l414_41444


namespace sum_inequality_l414_41453

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end sum_inequality_l414_41453


namespace sum_of_squares_l414_41459

theorem sum_of_squares : (10 + 3)^2 + (7 - 5)^2 = 173 := by
  sorry

end sum_of_squares_l414_41459


namespace sequence_inequality_l414_41418

theorem sequence_inequality (k : ℝ) : 
  (∀ n : ℕ+, (n + 1)^2 + k*(n + 1) + 2 > n^2 + k*n + 2) → k > -3 := by
  sorry

end sequence_inequality_l414_41418


namespace train_speed_crossing_bridge_l414_41441

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : Real) 
  (bridge_length : Real) 
  (crossing_time : Real) 
  (h1 : train_length = 150) 
  (h2 : bridge_length = 320) 
  (h3 : crossing_time = 40) : 
  (train_length + bridge_length) / crossing_time = 11.75 := by
  sorry

end train_speed_crossing_bridge_l414_41441


namespace fraction_equality_l414_41431

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (1/a + 1/b) / (1/a - 1/b) = 1001) : 
  (a + b) / (a - b) = 1001 := by
sorry

end fraction_equality_l414_41431


namespace average_of_three_l414_41416

theorem average_of_three (y : ℝ) : (15 + 24 + y) / 3 = 20 → y = 21 := by
  sorry

end average_of_three_l414_41416


namespace bread_roll_combinations_l414_41448

theorem bread_roll_combinations :
  let total_rolls : ℕ := 10
  let num_types : ℕ := 4
  let min_rolls_type1 : ℕ := 2
  let min_rolls_type2 : ℕ := 2
  let min_rolls_type3 : ℕ := 1
  let min_rolls_type4 : ℕ := 1
  let remaining_rolls : ℕ := total_rolls - (min_rolls_type1 + min_rolls_type2 + min_rolls_type3 + min_rolls_type4)
  (Nat.choose (remaining_rolls + num_types - 1) (num_types - 1)) = 35 :=
by sorry

end bread_roll_combinations_l414_41448


namespace band_members_max_l414_41420

theorem band_members_max (m r x : ℕ) : 
  m < 100 →
  r * x + 3 = m →
  (r - 1) * (x + 2) = m →
  ∀ n : ℕ, n < 100 ∧ (∃ r' x' : ℕ, r' * x' + 3 = n ∧ (r' - 1) * (x' + 2) = n) → n ≤ 91 :=
by sorry

end band_members_max_l414_41420


namespace leaf_movement_l414_41454

theorem leaf_movement (forward_distance : ℕ) (backward_distance : ℕ) (total_distance : ℕ) : 
  forward_distance = 5 → 
  backward_distance = 2 → 
  total_distance = 33 → 
  (total_distance / (forward_distance - backward_distance) : ℕ) = 11 := by
  sorry

end leaf_movement_l414_41454


namespace adam_laundry_l414_41429

/-- Given a total number of loads and a number of washed loads, calculate the remaining loads to wash. -/
def remaining_loads (total : ℕ) (washed : ℕ) : ℕ :=
  total - washed

/-- Theorem stating that given 14 total loads and 8 washed loads, the remaining loads is 6. -/
theorem adam_laundry : remaining_loads 14 8 = 6 := by
  sorry

end adam_laundry_l414_41429


namespace euro_equation_solution_l414_41401

def euro (x y : ℝ) : ℝ := 2 * x * y

theorem euro_equation_solution :
  ∀ x : ℝ, euro 6 (euro 4 x) = 480 → x = 5 := by
  sorry

end euro_equation_solution_l414_41401


namespace adjacent_complex_numbers_max_sum_squares_l414_41464

theorem adjacent_complex_numbers_max_sum_squares :
  ∀ (a b : ℝ),
  let z1 : ℂ := a + Complex.I * Real.sqrt 3
  let z2 : ℂ := 1 + Complex.I * b
  Complex.abs (z1 - z2) = 1 →
  ∃ (max : ℝ), max = 9 ∧ a^2 + b^2 ≤ max :=
by sorry

end adjacent_complex_numbers_max_sum_squares_l414_41464


namespace number_division_problem_l414_41409

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 2) / y = 4) : 
  y = 13 := by
sorry

end number_division_problem_l414_41409


namespace rhombus_height_is_half_side_l414_41478

/-- A rhombus is a quadrilateral with four equal sides -/
structure Rhombus where
  side : ℝ
  diag1 : ℝ
  diag2 : ℝ
  side_positive : 0 < side
  diag1_positive : 0 < diag1
  diag2_positive : 0 < diag2
  geometric_mean : side ^ 2 = diag1 * diag2

/-- The height of a rhombus with side s that is the geometric mean of its diagonals is s/2 -/
theorem rhombus_height_is_half_side (r : Rhombus) : 
  r.side / 2 = (r.diag1 * r.diag2) / (4 * r.side) := by
  sorry

#check rhombus_height_is_half_side

end rhombus_height_is_half_side_l414_41478


namespace circle_area_from_polar_equation_l414_41494

/-- The area of the circle described by the polar equation r = -4 cos θ + 8 sin θ is equal to 20π. -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := fun θ ↦ -4 * Real.cos θ + 8 * Real.sin θ
  ∃ c : ℝ × ℝ, ∃ radius : ℝ,
    (∀ θ : ℝ, (r θ * Real.cos θ - c.1)^2 + (r θ * Real.sin θ - c.2)^2 = radius^2) ∧
    Real.pi * radius^2 = 20 * Real.pi :=
by sorry

end circle_area_from_polar_equation_l414_41494


namespace min_framing_for_specific_picture_l414_41435

/-- Calculates the minimum number of linear feet of framing required for a picture with given dimensions and border width. -/
def min_framing_feet (original_width original_height border_width : ℕ) : ℕ :=
  let enlarged_width := 2 * original_width
  let enlarged_height := 2 * original_height
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + 11) / 12

/-- The minimum number of linear feet of framing required for a 5-inch by 8-inch picture,
    doubled in size and surrounded by a 4-inch border, is 7 feet. -/
theorem min_framing_for_specific_picture :
  min_framing_feet 5 8 4 = 7 := by
  sorry

end min_framing_for_specific_picture_l414_41435


namespace pencil_count_l414_41414

/-- Represents the number of items in a stationery store -/
structure StationeryStore where
  pens : ℕ
  pencils : ℕ
  erasers : ℕ

/-- Conditions for the stationery store inventory -/
def validInventory (s : StationeryStore) : Prop :=
  ∃ (x : ℕ), 
    s.pens = 5 * x ∧
    s.pencils = 6 * x ∧
    s.erasers = 10 * x ∧
    s.pencils = s.pens + 6 ∧
    s.erasers = 2 * s.pens

theorem pencil_count (s : StationeryStore) (h : validInventory s) : s.pencils = 36 := by
  sorry

#check pencil_count

end pencil_count_l414_41414


namespace cherry_tomatoes_per_jar_l414_41474

theorem cherry_tomatoes_per_jar 
  (total_tomatoes : ℕ) 
  (num_jars : ℕ) 
  (h1 : total_tomatoes = 56) 
  (h2 : num_jars = 7) : 
  total_tomatoes / num_jars = 8 := by
  sorry

end cherry_tomatoes_per_jar_l414_41474


namespace average_ticket_cost_l414_41489

/-- Calculates the average cost of tickets per person given the specified conditions --/
theorem average_ticket_cost (full_price : ℕ) (total_people : ℕ) (half_price_tickets : ℕ) (free_tickets : ℕ) (full_price_tickets : ℕ) :
  full_price = 150 →
  total_people = 5 →
  half_price_tickets = 2 →
  free_tickets = 1 →
  full_price_tickets = 2 →
  (full_price * full_price_tickets + (full_price / 2) * half_price_tickets) / total_people = 90 :=
by sorry

end average_ticket_cost_l414_41489


namespace parabola_focal_chord_angle_l414_41426

/-- Given a parabola y^2 = 2px and a focal chord AB of length 8p, 
    the angle of inclination θ of AB satisfies sin θ = ±1/2 -/
theorem parabola_focal_chord_angle (p : ℝ) (θ : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- parabola equation
  (8*p = 2*p / (Real.sin θ)^2) →  -- focal chord length formula
  (Real.sin θ = 1/2 ∨ Real.sin θ = -1/2) :=
sorry

end parabola_focal_chord_angle_l414_41426


namespace number_problem_l414_41411

theorem number_problem (x : ℝ) : 0.3 * x = 0.6 * 50 + 30 → x = 200 := by
  sorry

end number_problem_l414_41411


namespace abc_inequality_l414_41437

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 ∧ 1/a + 1/b + 1/c ≥ 9 :=
by sorry

end abc_inequality_l414_41437


namespace min_value_of_expression_l414_41465

theorem min_value_of_expression (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 ∧
  (x*y/z + y*z/x + z*x/y = Real.sqrt 3 ↔ x = y ∧ y = z ∧ z = Real.sqrt 3 / 3) := by
  sorry

end min_value_of_expression_l414_41465


namespace common_chord_equation_l414_41485

/-- The equation of the line containing the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → x + 2*y = 0 := by sorry

end common_chord_equation_l414_41485


namespace fiftieth_term_of_sequence_l414_41432

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

/-- Theorem: The 50th term of the arithmetic sequence starting with 2 and incrementing by 5 is 247 -/
theorem fiftieth_term_of_sequence : arithmetic_sequence 2 5 50 = 247 := by
  sorry

end fiftieth_term_of_sequence_l414_41432


namespace theater_seats_count_l414_41425

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increase : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increase + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given conditions has 770 seats -/
theorem theater_seats_count :
  ∀ t : Theater,
    t.first_row_seats = 14 →
    t.seat_increase = 2 →
    t.last_row_seats = 56 →
    total_seats t = 770 :=
by
  sorry


end theater_seats_count_l414_41425


namespace equal_roots_implies_c_value_l414_41496

-- Define the quadratic equation
def quadratic (x c : ℝ) : ℝ := x^2 + 6*x - c

-- Define the discriminant of the quadratic equation
def discriminant (c : ℝ) : ℝ := 6^2 - 4*(1)*(-c)

-- Theorem statement
theorem equal_roots_implies_c_value :
  (∃ x : ℝ, quadratic x c = 0 ∧ 
    ∀ y : ℝ, quadratic y c = 0 → y = x) →
  c = -9 := by sorry

end equal_roots_implies_c_value_l414_41496


namespace sin_difference_of_inverse_trig_functions_l414_41484

theorem sin_difference_of_inverse_trig_functions :
  Real.sin (Real.arcsin (3/5) - Real.arctan (1/2)) = 2 * Real.sqrt 5 / 25 := by
  sorry

end sin_difference_of_inverse_trig_functions_l414_41484


namespace odd_functions_sum_sufficient_not_necessary_l414_41460

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_functions_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsOdd f ∧ IsOdd g → IsOdd (f + g)) ∧
  (∃ f g : ℝ → ℝ, IsOdd (f + g) ∧ ¬(IsOdd f ∧ IsOdd g)) :=
sorry

end odd_functions_sum_sufficient_not_necessary_l414_41460


namespace regression_lines_intersection_l414_41400

/-- A linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passes_through (line : RegressionLine) (point : ℝ × ℝ) : Prop :=
  let (x, y) := point
  y = line.slope * x + line.intercept

theorem regression_lines_intersection
  (t1 t2 : RegressionLine)
  (s t : ℝ)
  (h1 : passes_through t1 (s, t))
  (h2 : passes_through t2 (s, t)) :
  ∃ (x y : ℝ), passes_through t1 (x, y) ∧ passes_through t2 (x, y) ∧ x = s ∧ y = t :=
sorry

end regression_lines_intersection_l414_41400


namespace elastic_collision_momentum_exchange_l414_41473

/-- Represents a particle with mass and velocity -/
structure Particle where
  mass : ℝ
  velocity : ℝ

/-- Calculates the momentum of a particle -/
def momentum (p : Particle) : ℝ := p.mass * p.velocity

/-- Represents the state of two particles before and after a collision -/
structure CollisionState where
  particle1 : Particle
  particle2 : Particle

/-- Defines an elastic head-on collision between two identical particles -/
def elasticCollision (initial : CollisionState) (final : CollisionState) : Prop :=
  initial.particle1.mass = initial.particle2.mass ∧
  initial.particle1.mass = final.particle1.mass ∧
  initial.particle2.velocity = 0 ∧
  momentum initial.particle1 + momentum initial.particle2 = momentum final.particle1 + momentum final.particle2 ∧
  (momentum initial.particle1)^2 + (momentum initial.particle2)^2 = (momentum final.particle1)^2 + (momentum final.particle2)^2

theorem elastic_collision_momentum_exchange 
  (initial final : CollisionState)
  (h_elastic : elasticCollision initial final)
  (h_initial_momentum : momentum initial.particle1 = p ∧ momentum initial.particle2 = 0) :
  momentum final.particle1 = 0 ∧ momentum final.particle2 = p := by
  sorry

end elastic_collision_momentum_exchange_l414_41473


namespace calculate_income_before_tax_l414_41417

/-- Given tax rates and differential savings, calculate the annual income before tax -/
theorem calculate_income_before_tax 
  (original_rate : ℝ) 
  (new_rate : ℝ) 
  (differential_savings : ℝ) 
  (h1 : original_rate = 0.42)
  (h2 : new_rate = 0.32)
  (h3 : differential_savings = 4240) :
  ∃ (income : ℝ), income * (original_rate - new_rate) = differential_savings ∧ income = 42400 := by
  sorry

end calculate_income_before_tax_l414_41417


namespace fraction_equality_l414_41451

theorem fraction_equality (x : ℝ) : 
  (4 + 2*x) / (7 + 3*x) = (2 + 3*x) / (4 + 5*x) ↔ x = -1 ∨ x = -2 := by
  sorry

end fraction_equality_l414_41451


namespace rhombus_longest_diagonal_l414_41458

/-- Given a rhombus with area 200 square units and diagonal ratio 4:3, 
    prove that the length of the longest diagonal is 40√3/3 -/
theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℚ) (d1 d2 : ℝ) :
  area = 200 →
  ratio = 4 / 3 →
  d1 / d2 = ratio →
  area = (d1 * d2) / 2 →
  d1 > d2 →
  d1 = 40 * Real.sqrt 3 / 3 := by
  sorry

end rhombus_longest_diagonal_l414_41458


namespace dennis_loose_coins_l414_41428

def loose_coins_problem (initial_amount : ℕ) (shirt_cost : ℕ) (bill_value : ℕ) (num_bills : ℕ) : Prop :=
  let total_change := initial_amount - shirt_cost
  let bills_amount := bill_value * num_bills
  let loose_coins := total_change - bills_amount
  loose_coins = 3

theorem dennis_loose_coins : 
  loose_coins_problem 50 27 10 2 := by
  sorry

end dennis_loose_coins_l414_41428


namespace ellipse_intersection_theorem_l414_41404

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := y^2/4 + x^2/2 = 1

-- Define the line
def Line (x y m k : ℝ) : Prop := y = k*x + m

-- Define the intersection condition
def Intersects (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ), 
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧ 
  Line x₁ y₁ m (y₁ / x₁) ∧ Line x₂ y₂ m (y₂ / x₂) ∧
  x₁ ≠ x₂ ∧ y₁ ≠ y₂

-- Define the vector condition
def VectorCondition (m : ℝ) : Prop := ∃ (x₁ y₁ x₂ y₂ : ℝ),
  Ellipse x₁ y₁ ∧ Ellipse x₂ y₂ ∧
  Line x₁ y₁ m (y₁ / x₁) ∧ Line x₂ y₂ m (y₂ / x₂) ∧
  x₁ + 2*x₂ = 0 ∧ y₁ + 2*y₂ = 3*m

theorem ellipse_intersection_theorem (m : ℝ) : 
  Intersects m ∧ VectorCondition m → 
  (2/3 < m ∧ m < 2) ∨ (-2 < m ∧ m < -2/3) :=
sorry

end ellipse_intersection_theorem_l414_41404


namespace right_angled_triangles_with_special_property_l414_41450

theorem right_angled_triangles_with_special_property :
  {(a, b, c) : ℕ × ℕ × ℕ | 
    0 < a ∧ a < b ∧ b < c ∧
    a * b = 4 * (a + b + c) ∧
    a * a + b * b = c * c} =
  {(10, 24, 26), (12, 16, 20), (9, 40, 41)} :=
by sorry

end right_angled_triangles_with_special_property_l414_41450


namespace prime_1993_equations_l414_41447

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem prime_1993_equations (h : isPrime 1993) :
  (∃ x y : ℕ, x^2 - y^2 = 1993) ∧
  (¬∃ x y : ℕ, x^3 - y^3 = 1993) ∧
  (¬∃ x y : ℕ, x^4 - y^4 = 1993) :=
by sorry

end prime_1993_equations_l414_41447


namespace union_of_sets_l414_41433

theorem union_of_sets : 
  let M : Set ℕ := {0, 1, 3}
  let N : Set ℕ := {x | x ∈ ({0, 3, 9} : Set ℕ)}
  M ∪ N = {0, 1, 3, 9} := by sorry

end union_of_sets_l414_41433


namespace min_red_chips_l414_41434

theorem min_red_chips (w b r : ℕ) : 
  b ≥ w / 3 →
  b ≤ r / 4 →
  w + b ≥ 75 →
  ∀ r' : ℕ, (∃ w' b' : ℕ, b' ≥ w' / 3 ∧ b' ≤ r' / 4 ∧ w' + b' ≥ 75) → r' ≥ 76 :=
by sorry

end min_red_chips_l414_41434


namespace greatest_k_value_l414_41497

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = 10) →
  k ≤ 2 * Real.sqrt 33 :=
by sorry

end greatest_k_value_l414_41497


namespace scientific_notation_of_0_0000000033_l414_41442

/-- Expresses a given number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_0_0000000033 :
  scientific_notation 0.0000000033 = (3.3, -9) := by sorry

end scientific_notation_of_0_0000000033_l414_41442


namespace minimize_distance_l414_41449

/-- Given points P(-2,-3) and Q(5,3) in the xy-plane, and R(2,m) chosen such that PR+RQ is minimized, prove that m = 3/7 -/
theorem minimize_distance (P Q R : ℝ × ℝ) (m : ℝ) :
  P = (-2, -3) →
  Q = (5, 3) →
  R = (2, m) →
  (∀ m' : ℝ, dist P R + dist R Q ≤ dist P (2, m') + dist (2, m') Q) →
  m = 3/7 := by
  sorry


end minimize_distance_l414_41449


namespace principal_is_15000_l414_41438

/-- Represents the loan details and calculations -/
structure Loan where
  principal : ℝ
  interestRates : Fin 3 → ℝ
  totalInterest : ℝ

/-- Calculates the total interest paid over 3 years -/
def totalInterestPaid (loan : Loan) : ℝ :=
  (loan.interestRates 0 + loan.interestRates 1 + loan.interestRates 2) * loan.principal

/-- Theorem stating that given the conditions, the principal amount is 15000 -/
theorem principal_is_15000 (loan : Loan)
  (h1 : loan.interestRates 0 = 0.10)
  (h2 : loan.interestRates 1 = 0.12)
  (h3 : loan.interestRates 2 = 0.14)
  (h4 : loan.totalInterest = 5400)
  (h5 : totalInterestPaid loan = loan.totalInterest) :
  loan.principal = 15000 := by
  sorry

#check principal_is_15000

end principal_is_15000_l414_41438


namespace one_of_each_color_probability_l414_41405

def total_marbles : ℕ := 9
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 3
def selected_marbles : ℕ := 3

def probability_one_of_each_color : ℚ := 9 / 28

theorem one_of_each_color_probability :
  probability_one_of_each_color = 
    (red_marbles * blue_marbles * green_marbles : ℚ) / 
    (Nat.choose total_marbles selected_marbles) :=
by sorry

end one_of_each_color_probability_l414_41405


namespace abs_neg_three_l414_41456

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by
  sorry

end abs_neg_three_l414_41456


namespace election_winner_percentage_l414_41421

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 3744 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 52 / 100 := by
sorry

end election_winner_percentage_l414_41421


namespace star_arrangements_l414_41415

/-- The number of points on a regular ten-pointed star -/
def num_points : ℕ := 20

/-- The number of rotational symmetries of a regular ten-pointed star -/
def num_rotations : ℕ := 10

/-- The number of reflectional symmetries of a regular ten-pointed star -/
def num_reflections : ℕ := 2

/-- The total number of symmetries of a regular ten-pointed star -/
def total_symmetries : ℕ := num_rotations * num_reflections

/-- The number of distinct arrangements of objects on a regular ten-pointed star -/
def distinct_arrangements : ℕ := Nat.factorial num_points / total_symmetries

theorem star_arrangements :
  distinct_arrangements = Nat.factorial (num_points - 1) := by
  sorry

end star_arrangements_l414_41415


namespace smoothie_servings_l414_41479

/-- The number of servings that can be made from a given volume of smoothie mix -/
def number_of_servings (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ) : ℕ :=
  (watermelon_puree + cream) / serving_size

/-- Theorem: Given 500 ml of watermelon puree and 100 ml of cream, 
    the number of 150 ml servings that can be made is equal to 4 -/
theorem smoothie_servings : 
  number_of_servings 500 100 150 = 4 := by
  sorry

end smoothie_servings_l414_41479


namespace function_properties_l414_41439

noncomputable def f (a b m x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + m

noncomputable def f' (a b x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

theorem function_properties (a b m : ℝ) :
  (∀ x : ℝ, f' a b x = f' a b (-1 - x)) →  -- Symmetry about x = -1/2
  f' a b 1 = 0 →                           -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                        -- Values of a and b
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧     -- Exactly three zeros
    f a b m x₁ = 0 ∧ f a b m x₂ = 0 ∧ f a b m x₃ = 0 ∧
    (∀ x : ℝ, f a b m x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  -20 < m ∧ m < 7                          -- Range of m
  := by sorry

end function_properties_l414_41439


namespace f_increasing_and_no_negative_roots_l414_41491

noncomputable section

variable (a : ℝ) (h : a > 1)

def f (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_increasing_and_no_negative_roots :
  (∀ x y, -1 < x ∧ x < y → f a x < f a y) ∧
  (∀ x, x < 0 → f a x ≠ 0) := by sorry

end f_increasing_and_no_negative_roots_l414_41491


namespace largest_indecomposable_amount_l414_41488

/-- Represents the set of coin denominations in Limonia -/
def coin_denominations (n : ℕ) : List ℕ :=
  List.range (n + 1) |> List.map (fun k => 3^(n - k) * 5^k)

/-- Predicate to check if a number is decomposable using given coin denominations -/
def is_decomposable (s : ℕ) (n : ℕ) : Prop :=
  ∃ (coeffs : List ℕ), 
    coeffs.length = n + 1 ∧ 
    (List.zip coeffs (coin_denominations n) |> List.map (fun (c, d) => c * d) |> List.sum) = s

/-- The main theorem stating the largest indecomposable amount -/
theorem largest_indecomposable_amount (n : ℕ) : 
  ¬(is_decomposable (5^(n+1) - 2 * 3^(n+1)) n) ∧ 
  ∀ m : ℕ, m > (5^(n+1) - 2 * 3^(n+1)) → is_decomposable m n :=
by sorry

end largest_indecomposable_amount_l414_41488


namespace share_price_increase_l414_41476

theorem share_price_increase (P : ℝ) (h : P > 0) : 
  let first_quarter := P * 1.25
  let second_quarter := first_quarter * 1.24
  (second_quarter - P) / P * 100 = 55 := by
sorry

end share_price_increase_l414_41476


namespace gcd_lcm_sum_l414_41403

theorem gcd_lcm_sum : Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := by
  sorry

end gcd_lcm_sum_l414_41403


namespace block3_can_reach_target_l414_41445

-- Define the board
def Board := Fin 3 × Fin 7

-- Define a block
structure Block where
  label : Nat
  position : Board

-- Define the game state
structure GameState where
  blocks : List Block

-- Define a valid move
inductive Move
| Up : Block → Move
| Down : Block → Move
| Left : Block → Move
| Right : Block → Move

-- Define the initial game state
def initialState : GameState := {
  blocks := [
    { label := 1, position := ⟨2, 2⟩ },
    { label := 2, position := ⟨3, 5⟩ },
    { label := 3, position := ⟨1, 4⟩ }
  ]
}

-- Define the target position
def targetPosition : Board := ⟨2, 4⟩

-- Function to check if a move is valid
def isValidMove (state : GameState) (move : Move) : Bool := sorry

-- Function to apply a move to the game state
def applyMove (state : GameState) (move : Move) : GameState := sorry

-- Theorem: There exists a sequence of valid moves to bring Block 3 to the target position
theorem block3_can_reach_target :
  ∃ (moves : List Move), 
    let finalState := moves.foldl (λ s m => applyMove s m) initialState
    (finalState.blocks.find? (λ b => b.label = 3)).map (λ b => b.position) = some targetPosition :=
sorry

end block3_can_reach_target_l414_41445


namespace mike_grew_four_onions_l414_41413

/-- The number of onions grown by Mike given the number of onions grown by Nancy, Dan, and the total number of onions. -/
def mikes_onions (nancy_onions dan_onions total_onions : ℕ) : ℕ :=
  total_onions - (nancy_onions + dan_onions)

/-- Theorem stating that Mike grew 4 onions given the conditions. -/
theorem mike_grew_four_onions :
  mikes_onions 2 9 15 = 4 := by
  sorry

end mike_grew_four_onions_l414_41413


namespace refrigerator_price_l414_41490

theorem refrigerator_price (P : ℝ) 
  (selling_price : P + 0.1 * P = 23100)
  (discount : ℝ := 0.2)
  (transport_cost : ℝ := 125)
  (installation_cost : ℝ := 250) :
  P * (1 - discount) + transport_cost + installation_cost = 17175 := by
sorry

end refrigerator_price_l414_41490


namespace ones_digit_of_prime_arithmetic_sequence_l414_41499

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_prime_arithmetic_sequence (p q r s : ℕ) : 
  is_prime p → is_prime q → is_prime r → is_prime s →
  p > 5 →
  q = p + 4 →
  r = q + 4 →
  s = r + 4 →
  ones_digit p = 9 := by sorry

end ones_digit_of_prime_arithmetic_sequence_l414_41499


namespace complex_number_proof_l414_41423

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_number_proof (Z : ℂ) 
  (h1 : Complex.abs Z = 3)
  (h2 : is_pure_imaginary (Z + 3*I)) : 
  Z = 3*I := by sorry

end complex_number_proof_l414_41423


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l414_41408

/-- An isosceles triangle with two sides of length 12 and one side of length 17 has a perimeter of 41. -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (side1 side2 base : ℝ),
      side1 = 12 ∧
      side2 = 12 ∧
      base = 17 ∧
      perimeter = side1 + side2 + base ∧
      perimeter = 41

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 41 := by
  sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l414_41408


namespace children_savings_l414_41407

/-- The total savings of Josiah, Leah, and Megan -/
def total_savings (josiah_daily : ℚ) (josiah_days : ℕ) 
                  (leah_daily : ℚ) (leah_days : ℕ)
                  (megan_daily : ℚ) (megan_days : ℕ) : ℚ :=
  josiah_daily * josiah_days + leah_daily * leah_days + megan_daily * megan_days

/-- Theorem stating that the total savings of the three children is $28 -/
theorem children_savings : 
  total_savings 0.25 24 0.50 20 1.00 12 = 28 := by
  sorry

end children_savings_l414_41407


namespace nancy_finished_problems_l414_41481

/-- Given that Nancy had 101 homework problems initially, still has 6 pages of problems to do,
    and each page has 9 problems, prove that she finished 47 problems. -/
theorem nancy_finished_problems (total_problems : ℕ) (pages_left : ℕ) (problems_per_page : ℕ)
    (h1 : total_problems = 101)
    (h2 : pages_left = 6)
    (h3 : problems_per_page = 9) :
    total_problems - (pages_left * problems_per_page) = 47 := by
  sorry


end nancy_finished_problems_l414_41481


namespace sugar_amount_in_recipe_l414_41430

/-- Given a recipe that requires a total of 10 cups of flour, 
    with 2 cups already added, and the remaining flour needed 
    being 5 cups more than the amount of sugar, 
    prove that the recipe calls for 3 cups of sugar. -/
theorem sugar_amount_in_recipe 
  (total_flour : ℕ) 
  (added_flour : ℕ) 
  (sugar : ℕ) : 
  total_flour = 10 → 
  added_flour = 2 → 
  total_flour = added_flour + (sugar + 5) → 
  sugar = 3 := by
  sorry

end sugar_amount_in_recipe_l414_41430


namespace sequence_less_than_two_l414_41469

theorem sequence_less_than_two (a : ℕ → ℝ) :
  (∀ n, a n < 2) ↔ ¬(∃ k, a k ≥ 2) := by
  sorry

end sequence_less_than_two_l414_41469


namespace polar_to_cartesian_x_plus_y_bounds_l414_41467

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the circle in Cartesian coordinates
def cartesian_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Theorem stating the equivalence of polar and Cartesian equations
theorem polar_to_cartesian :
  ∀ (x y ρ θ : ℝ), 
    x = ρ * Real.cos θ → 
    y = ρ * Real.sin θ → 
    polar_circle ρ θ ↔ cartesian_circle x y :=
sorry

-- Theorem for the bounds of x + y
theorem x_plus_y_bounds :
  ∀ (x y : ℝ), cartesian_circle x y → 2 ≤ x + y ∧ x + y ≤ 6 :=
sorry

end polar_to_cartesian_x_plus_y_bounds_l414_41467


namespace max_distance_to_point_l414_41477

theorem max_distance_to_point (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (w : ℂ), Complex.abs w = 1 ∧ Complex.abs (w - (1 + Complex.I)) = Real.sqrt 2 + 1 :=
sorry

end max_distance_to_point_l414_41477


namespace right_triangle_side_length_l414_41483

theorem right_triangle_side_length : 
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  c = 13 → a = 12 →
  c^2 = a^2 + b^2 →
  b = 5 := by
sorry

end right_triangle_side_length_l414_41483


namespace correct_sampling_methods_l414_41493

/-- Represents a population with possible strata --/
structure Population where
  total : Nat
  strata : List Nat

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Stratified

/-- Represents a sampling problem --/
structure SamplingProblem where
  population : Population
  sampleSize : Nat

/-- Determines the appropriate sampling method for a given problem --/
def appropriateSamplingMethod (problem : SamplingProblem) : SamplingMethod :=
  sorry

theorem correct_sampling_methods
  (collegeProblem : SamplingProblem)
  (workshopProblem : SamplingProblem)
  (h1 : collegeProblem.population = { total := 300, strata := [150, 150] })
  (h2 : collegeProblem.sampleSize = 100)
  (h3 : workshopProblem.population = { total := 100, strata := [] })
  (h4 : workshopProblem.sampleSize = 10) :
  appropriateSamplingMethod collegeProblem = SamplingMethod.Stratified ∧
  appropriateSamplingMethod workshopProblem = SamplingMethod.SimpleRandom :=
sorry

end correct_sampling_methods_l414_41493


namespace science_fiction_total_pages_l414_41492

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages :
  total_pages = 3824 := by
  sorry

end science_fiction_total_pages_l414_41492


namespace parabola_equation_l414_41424

/-- A parabola with vertex at the origin, focus on the y-axis, and directrix y = 3 has the equation x² = 12y -/
theorem parabola_equation (p : ℝ) (h1 : p > 0) (h2 : p / 2 = 3) :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | x^2 = 2 * p * y} ↔ x^2 = 12 * y := by
  sorry

end parabola_equation_l414_41424


namespace systematic_sampling_probability_l414_41471

theorem systematic_sampling_probability (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 121) (h2 : sample_size = 20) :
  (sample_size : ℚ) / total_students = 20 / 121 :=
by sorry

end systematic_sampling_probability_l414_41471


namespace male_gerbil_fraction_l414_41475

theorem male_gerbil_fraction (total_pets : ℕ) (total_gerbils : ℕ) (total_males : ℕ) :
  total_pets = 90 →
  total_gerbils = 66 →
  total_males = 25 →
  (total_pets - total_gerbils) / 3 + (total_males - (total_pets - total_gerbils) / 3) = total_males →
  (total_males - (total_pets - total_gerbils) / 3) / total_gerbils = 17 / 66 := by
  sorry

end male_gerbil_fraction_l414_41475


namespace power_mod_five_l414_41455

theorem power_mod_five : 3^19 % 5 = 2 := by
  sorry

end power_mod_five_l414_41455


namespace complex_number_equation_l414_41436

theorem complex_number_equation (z : ℂ) 
  (h : 15 * Complex.normSq z = 5 * Complex.normSq (z + 1) + Complex.normSq (z^2 - 1) + 44) : 
  z^2 + 36 / z^2 = 60 := by sorry

end complex_number_equation_l414_41436


namespace nested_fraction_evaluation_l414_41402

theorem nested_fraction_evaluation :
  1 + (1 / (1 + (1 / (1 + (1 / 2))))) = 8 / 5 := by
  sorry

end nested_fraction_evaluation_l414_41402


namespace nonzero_real_equation_solution_l414_41466

theorem nonzero_real_equation_solution (x : ℝ) (h : x ≠ 0) :
  (9 * x)^18 = (18 * x)^9 ↔ x = 2/9 := by
  sorry

end nonzero_real_equation_solution_l414_41466


namespace area_eq_xy_l414_41462

/-- A right-angled triangle with an inscribed circle. -/
structure RightTriangleWithIncircle where
  /-- The length of one segment of the hypotenuse. -/
  x : ℝ
  /-- The length of the other segment of the hypotenuse. -/
  y : ℝ
  /-- The radius of the inscribed circle. -/
  r : ℝ
  /-- x and y are positive -/
  x_pos : 0 < x
  y_pos : 0 < y
  /-- r is positive -/
  r_pos : 0 < r

/-- The area of a right-angled triangle with an inscribed circle. -/
def area (t : RightTriangleWithIncircle) : ℝ :=
  t.x * t.y

/-- Theorem: The area of a right-angled triangle with an inscribed circle
    touching the hypotenuse at a point dividing it into segments of lengths x and y
    is equal to x * y. -/
theorem area_eq_xy (t : RightTriangleWithIncircle) : area t = t.x * t.y := by
  sorry

end area_eq_xy_l414_41462
