import Mathlib

namespace point_on_line_m_range_l3102_310233

-- Define the function f
def f (x m n : ℝ) : ℝ := |x - m| + |x + n|

-- Part 1
theorem point_on_line (m n : ℝ) (h1 : m + n > 0) (h2 : ∀ x, f x m n ≥ 2) 
  (h3 : ∃ x, f x m n = 2) : m + n = 2 := by
  sorry

-- Part 2
theorem m_range (m : ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x m 2 ≤ x + 5) : 
  m ∈ Set.Icc (-2) 3 := by
  sorry

end point_on_line_m_range_l3102_310233


namespace wilson_payment_l3102_310275

def hamburger_price : ℕ := 5
def cola_price : ℕ := 2
def hamburger_quantity : ℕ := 2
def cola_quantity : ℕ := 3
def discount : ℕ := 4

def total_cost : ℕ := hamburger_price * hamburger_quantity + cola_price * cola_quantity - discount

theorem wilson_payment : total_cost = 12 := by
  sorry

end wilson_payment_l3102_310275


namespace tan_alpha_half_implies_fraction_equals_negative_four_l3102_310212

theorem tan_alpha_half_implies_fraction_equals_negative_four (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 := by
  sorry

end tan_alpha_half_implies_fraction_equals_negative_four_l3102_310212


namespace cyclic_quadrilateral_decomposition_l3102_310285

/-- A cyclic quadrilateral is a quadrilateral that can be circumscribed about a circle. -/
def CyclicQuadrilateral : Type := sorry

/-- A decomposition of a quadrilateral into n smaller quadrilaterals. -/
def Decomposition (Q : CyclicQuadrilateral) (n : ℕ) : Type := sorry

/-- Predicate to check if all quadrilaterals in a decomposition are cyclic. -/
def AllCyclic (d : Decomposition Q n) : Prop := sorry

theorem cyclic_quadrilateral_decomposition (n : ℕ) (Q : CyclicQuadrilateral) 
  (h : n ≥ 4) : 
  ∃ (d : Decomposition Q n), AllCyclic d := by sorry

end cyclic_quadrilateral_decomposition_l3102_310285


namespace ned_good_games_l3102_310236

/-- Calculates the number of good games Ned ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - non_working_games

/-- Theorem: Ned ended up with 14 good games -/
theorem ned_good_games : good_games 11 22 19 = 14 := by
  sorry

end ned_good_games_l3102_310236


namespace equation_solution_l3102_310268

theorem equation_solution (n : ℚ) : 
  (2 / (n + 2) + 3 / (n + 2) + (2 * n) / (n + 2) = 4) → n = -3/2 := by
  sorry

end equation_solution_l3102_310268


namespace not_sufficient_not_necessary_l3102_310217

/-- Two lines ax + by + c = 0 and dx + ey + f = 0 are perpendicular if and only if ad + be = 0 -/
def are_perpendicular (a b c d e f : ℝ) : Prop :=
  a * d + b * e = 0

/-- The proposition "a = 2" is neither sufficient nor necessary for the line ax + 3y - 1 = 0
    to be perpendicular to the line 6x + 4y - 3 = 0 -/
theorem not_sufficient_not_necessary : 
  (∃ a : ℝ, a = 2 ∧ ¬(are_perpendicular a 3 (-1) 6 4 (-3))) ∧ 
  (∃ a : ℝ, are_perpendicular a 3 (-1) 6 4 (-3) ∧ a ≠ 2) :=
sorry

end not_sufficient_not_necessary_l3102_310217


namespace smallest_divisible_by_12_15_16_l3102_310249

theorem smallest_divisible_by_12_15_16 :
  ∃ (n : ℕ), n > 0 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n ∧
  ∀ (m : ℕ), m > 0 → 12 ∣ m → 15 ∣ m → 16 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end smallest_divisible_by_12_15_16_l3102_310249


namespace nearest_town_distance_l3102_310210

theorem nearest_town_distance (d : ℝ) : 
  (¬ (d ≥ 8)) ∧ 
  (¬ (d ≤ 7)) ∧ 
  (¬ (d ≤ 6)) ∧ 
  (¬ (d ≥ 9)) →
  d ∈ Set.Ioo 7 8 :=
by sorry

end nearest_town_distance_l3102_310210


namespace cube_sum_simplification_l3102_310287

theorem cube_sum_simplification (a b c : ℝ) :
  (a^3 + b^3) / (c^3 + b^3) = (a + b) / (c + b) ↔ a + c = b :=
by sorry

end cube_sum_simplification_l3102_310287


namespace fifteenth_prime_l3102_310231

theorem fifteenth_prime (p : ℕ → ℕ) (h : ∀ n, Prime (p n)) (h15 : p 7 = 15) : p 15 = 47 := by
  sorry

end fifteenth_prime_l3102_310231


namespace minimum_rows_needed_l3102_310262

structure School where
  students : ℕ
  h1 : 1 ≤ students
  h2 : students ≤ 39

def City := List School

def totalStudents (city : City) : ℕ :=
  city.map (λ s => s.students) |>.sum

theorem minimum_rows_needed (city : City) 
  (h_total : totalStudents city = 1990) 
  (h_seats_per_row : ℕ := 199) : ℕ :=
  12

#check minimum_rows_needed

end minimum_rows_needed_l3102_310262


namespace greatest_prime_factor_of_341_l3102_310278

theorem greatest_prime_factor_of_341 : ∃ (p : ℕ), p.Prime ∧ p ∣ 341 ∧ ∀ (q : ℕ), q.Prime → q ∣ 341 → q ≤ p :=
by sorry

end greatest_prime_factor_of_341_l3102_310278


namespace division_remainder_problem_l3102_310276

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1365 → 
  L = 1637 → 
  ∃ (q : ℕ), q = 6 ∧ L = q * S + (L % S) → 
  L % S = 5 := by
sorry

end division_remainder_problem_l3102_310276


namespace convention_handshakes_count_l3102_310238

/-- Represents the number of handshakes at a convention with twins and triplets -/
def convention_handshakes (twin_sets triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2) / 2
  let triplet_handshakes := triplets * (triplets - 3) / 2
  let cross_handshakes := twins * (2 * triplets / 3)
  twin_handshakes + triplet_handshakes + cross_handshakes

/-- The number of handshakes at the convention is 900 -/
theorem convention_handshakes_count : convention_handshakes 12 8 = 900 := by
  sorry

#eval convention_handshakes 12 8

end convention_handshakes_count_l3102_310238


namespace average_and_difference_l3102_310246

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 53 → |y - 45| = 16 := by
  sorry

end average_and_difference_l3102_310246


namespace boat_speed_l3102_310250

/-- Given a boat that travels 11 km/h along a stream and 5 km/h against the same stream,
    its speed in still water is 8 km/h. -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) (still_water : ℝ)
    (h1 : along_stream = 11)
    (h2 : against_stream = 5)
    (h3 : along_stream = still_water + (along_stream - still_water))
    (h4 : against_stream = still_water - (along_stream - still_water)) :
    still_water = 8 := by
  sorry

end boat_speed_l3102_310250


namespace greatest_b_quadratic_inequality_l3102_310273

theorem greatest_b_quadratic_inequality :
  ∃ b : ℝ, b^2 - 10*b + 24 ≤ 0 ∧ ∀ x : ℝ, x^2 - 10*x + 24 ≤ 0 → x ≤ b :=
by
  -- The proof goes here
  sorry

end greatest_b_quadratic_inequality_l3102_310273


namespace fred_change_l3102_310263

def movie_ticket_cost : ℝ := 5.92
def movie_tickets : ℕ := 3
def movie_rental : ℝ := 6.79
def snacks : ℝ := 10.50
def parking : ℝ := 3.25
def paid_amount : ℝ := 50

def total_cost : ℝ := movie_ticket_cost * movie_tickets + movie_rental + snacks + parking

def change : ℝ := paid_amount - total_cost

theorem fred_change : change = 11.70 := by sorry

end fred_change_l3102_310263


namespace orange_distribution_l3102_310235

theorem orange_distribution (total_oranges : ℕ) (pieces_per_orange : ℕ) (pieces_per_friend : ℕ) : 
  total_oranges = 80 → pieces_per_orange = 10 → pieces_per_friend = 4 →
  (total_oranges * pieces_per_orange) / pieces_per_friend = 200 := by
sorry

end orange_distribution_l3102_310235


namespace petyas_coins_l3102_310242

theorem petyas_coins (total : ℕ) (not_two : ℕ) (not_ten : ℕ) (not_one : ℕ) 
  (h_total : total = 25)
  (h_not_two : not_two = 19)
  (h_not_ten : not_ten = 20)
  (h_not_one : not_one = 16) :
  total - ((total - not_two) + (total - not_ten) + (total - not_one)) = 5 := by
  sorry

end petyas_coins_l3102_310242


namespace valid_distribution_example_l3102_310241

def is_valid_distribution (probs : List ℚ) : Prop :=
  (probs.sum = 1) ∧ (∀ p ∈ probs, 0 < p ∧ p ≤ 1)

theorem valid_distribution_example : 
  is_valid_distribution [1/2, 1/3, 1/6] := by
  sorry

end valid_distribution_example_l3102_310241


namespace f_value_at_7_5_l3102_310239

def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + 2) = -f x) ∧  -- f(x+2) = -f(x)
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)  -- f(x) = x for 0 ≤ x ≤ 1

theorem f_value_at_7_5 (f : ℝ → ℝ) (h : f_conditions f) : f 7.5 = -0.5 := by
  sorry

end f_value_at_7_5_l3102_310239


namespace equation_solution_l3102_310280

theorem equation_solution (c d : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + 15 = 27 ∧ (x = c ∨ x = d)) →
  c ≥ d →
  3*c - d = 6 + 4*Real.sqrt 21 := by
sorry

end equation_solution_l3102_310280


namespace triangle_area_l3102_310244

/-- Given a triangle with perimeter 28 cm and inradius 2.5 cm, its area is 35 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2.5 → area = inradius * (perimeter / 2) → area = 35 := by
  sorry

end triangle_area_l3102_310244


namespace game_cost_l3102_310223

theorem game_cost (initial_amount allowance final_amount : ℕ) : 
  initial_amount = 5 → 
  allowance = 26 → 
  final_amount = 29 → 
  initial_amount + allowance - final_amount = 2 := by
sorry

end game_cost_l3102_310223


namespace vehicle_speeds_l3102_310226

/-- Proves that given the conditions of the problem, the bus speed is 20 km/h and the car speed is 60 km/h -/
theorem vehicle_speeds (distance : ℝ) (bus_speed : ℝ) (car_speed : ℝ) (bus_departure : ℝ) (car_departure : ℝ) (arrival_difference : ℝ) :
  distance = 80 ∧
  car_departure = bus_departure + 3 ∧
  car_speed = 3 * bus_speed ∧
  arrival_difference = 1/3 ∧
  distance / bus_speed = distance / car_speed + (car_departure - bus_departure) - arrival_difference →
  bus_speed = 20 ∧ car_speed = 60 := by
sorry


end vehicle_speeds_l3102_310226


namespace correct_answer_l3102_310296

theorem correct_answer (x : ℤ) (h : x + 5 = 35) : x - 5 = 25 := by
  sorry

end correct_answer_l3102_310296


namespace three_digit_integers_with_7_no_4_l3102_310265

/-- The set of digits excluding 0, 4, and 7 -/
def digits_no_047 : Finset Nat := Finset.filter (fun d => d ≠ 0 ∧ d ≠ 4 ∧ d ≠ 7) (Finset.range 10)

/-- The set of digits excluding 0 and 4 -/
def digits_no_04 : Finset Nat := Finset.filter (fun d => d ≠ 0 ∧ d ≠ 4) (Finset.range 10)

/-- The set of digits excluding 4 -/
def digits_no_4 : Finset Nat := Finset.filter (fun d => d ≠ 4) (Finset.range 10)

/-- The number of three-digit integers without 7 and 4 -/
def count_no_47 : Nat := digits_no_047.card * digits_no_4.card * digits_no_4.card

/-- The number of three-digit integers without 4 -/
def count_no_4 : Nat := digits_no_04.card * digits_no_4.card * digits_no_4.card

theorem three_digit_integers_with_7_no_4 :
  count_no_4 - count_no_47 = 200 := by sorry

end three_digit_integers_with_7_no_4_l3102_310265


namespace triangle_case1_triangle_case2_l3102_310254

-- Case 1
theorem triangle_case1 (AB AD HM : ℝ) (h1 : AB = 10) (h2 : AD = 4) (h3 : HM = 6/5) :
  let BD := Real.sqrt (AB^2 - AD^2)
  let DH := (4 * Real.sqrt 21) / 5
  let DC := BD - HM
  DC = (8 * Real.sqrt 21 - 12) / 5 := by sorry

-- Case 2
theorem triangle_case2 (AB AD HM : ℝ) (h1 : AB = 8 * Real.sqrt 2) (h2 : AD = 4) (h3 : HM = Real.sqrt 2) :
  let BD := Real.sqrt (AB^2 - AD^2)
  let DH := Real.sqrt 14
  let DC := BD - HM
  DC = 2 * Real.sqrt 14 - 2 * Real.sqrt 2 := by sorry

end triangle_case1_triangle_case2_l3102_310254


namespace line_equation_60_degrees_l3102_310284

theorem line_equation_60_degrees (x y : ℝ) :
  let angle : ℝ := 60 * π / 180  -- Convert 60° to radians
  let slope : ℝ := Real.tan angle
  let y_intercept : ℝ := -1
  (slope * x - y - y_intercept = 0) ↔ (Real.sqrt 3 * x - y - 1 = 0) := by
  sorry

end line_equation_60_degrees_l3102_310284


namespace equidistant_point_y_coordinate_l3102_310293

/-- The y-coordinate of the point on the y-axis equidistant from A(-3, 1) and B(2, 5) is 19/8 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, ((-3 - 0)^2 + (1 - y)^2 = (2 - 0)^2 + (5 - y)^2) ∧ y = 19/8 := by
  sorry

end equidistant_point_y_coordinate_l3102_310293


namespace expand_product_l3102_310251

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end expand_product_l3102_310251


namespace line_equation_l3102_310255

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a given equation represents the line -/
def is_equation_of_line (l : Line) (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ y - l.point.snd = l.slope * (x - l.point.fst)

theorem line_equation (l : Line) :
  l.slope = 3 ∧ l.point = (1, 3) →
  is_equation_of_line l (fun x y ↦ y - 3 = 3 * (x - 1)) :=
sorry

end line_equation_l3102_310255


namespace integer_sum_of_fourth_powers_l3102_310283

theorem integer_sum_of_fourth_powers (a b c : ℤ) (h : a = b + c) :
  a^4 + b^4 + c^4 = 2 * (a^2 - b*c)^2 := by
  sorry

end integer_sum_of_fourth_powers_l3102_310283


namespace range_of_a_for_M_subset_N_l3102_310257

/-- The set of real numbers m for which x^2 - x - m = 0 has solutions in (-1, 1) -/
def M : Set ℝ :=
  {m : ℝ | ∃ x, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

/-- The solution set of (x - a)(x + a - 2) < 0 -/
def N (a : ℝ) : Set ℝ :=
  {x : ℝ | (x - a) * (x + a - 2) < 0}

/-- The theorem stating the range of a values for which M ⊆ N(a) -/
theorem range_of_a_for_M_subset_N :
  {a : ℝ | M ⊆ N a} = {a : ℝ | a < -1/4 ∨ a > 9/4} := by sorry

end range_of_a_for_M_subset_N_l3102_310257


namespace median_salary_is_25000_l3102_310208

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions and the total number of employees. -/
def medianSalary (positions : List Position) (totalEmployees : Nat) : Nat :=
  sorry

/-- The list of positions in the company. -/
def companyPositions : List Position := [
  { title := "President", count := 1, salary := 140000 },
  { title := "Vice-President", count := 4, salary := 95000 },
  { title := "Director", count := 11, salary := 78000 },
  { title := "Associate Director", count := 8, salary := 55000 },
  { title := "Administrative Specialist", count := 39, salary := 25000 }
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat := 63

/-- Theorem stating that the median salary of the company is $25,000. -/
theorem median_salary_is_25000 : 
  medianSalary companyPositions totalEmployees = 25000 := by
  sorry

end median_salary_is_25000_l3102_310208


namespace complete_square_quadratic_l3102_310232

theorem complete_square_quadratic : ∀ x : ℝ, x^2 - 4*x + 2 = 0 ↔ (x - 2)^2 = 2 := by sorry

end complete_square_quadratic_l3102_310232


namespace negation_equivalence_l3102_310261

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 + Real.sin x + 1 < 0) ↔ (∃ x : ℝ, x^2 + Real.sin x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l3102_310261


namespace number_puzzle_l3102_310258

theorem number_puzzle : ∃ x : ℝ, (20 * (x / 5) = 40) ∧ x = 10 := by
  sorry

end number_puzzle_l3102_310258


namespace thirty_percent_less_eighty_forty_percent_more_l3102_310211

theorem thirty_percent_less_eighty_forty_percent_more (x : ℝ) : 
  (x + 0.4 * x = 80 - 0.3 * 80) → x = 40 := by
  sorry

end thirty_percent_less_eighty_forty_percent_more_l3102_310211


namespace alice_current_age_l3102_310288

/-- Alice's current age -/
def alice_age : ℕ := 30

/-- Beatrice's current age -/
def beatrice_age : ℕ := 11

/-- In 8 years, Alice will be twice as old as Beatrice -/
axiom future_age_relation : alice_age + 8 = 2 * (beatrice_age + 8)

/-- Ten years ago, the sum of their ages was 21 -/
axiom past_age_sum : (alice_age - 10) + (beatrice_age - 10) = 21

theorem alice_current_age : alice_age = 30 := by
  sorry

end alice_current_age_l3102_310288


namespace yolanda_three_point_average_l3102_310282

theorem yolanda_three_point_average (total_points season_games free_throws_per_game two_point_baskets_per_game : ℕ)
  (h1 : total_points = 345)
  (h2 : season_games = 15)
  (h3 : free_throws_per_game = 4)
  (h4 : two_point_baskets_per_game = 5) :
  (total_points - (free_throws_per_game * 1 + two_point_baskets_per_game * 2) * season_games) / (3 * season_games) = 3 := by
  sorry

end yolanda_three_point_average_l3102_310282


namespace felix_drive_l3102_310218

theorem felix_drive (average_speed : ℝ) (drive_time : ℝ) : 
  average_speed = 66 → drive_time = 4 → (2 * average_speed) * drive_time = 528 := by
  sorry

end felix_drive_l3102_310218


namespace purely_imaginary_solution_l3102_310269

theorem purely_imaginary_solution (z : ℂ) :
  (∃ b : ℝ, z = Complex.I * b) →
  (∃ c : ℝ, (z - 2)^2 - Complex.I * 8 = Complex.I * c) →
  z = Complex.I * 2 := by
sorry

end purely_imaginary_solution_l3102_310269


namespace sibling_ages_l3102_310230

/-- Represents the ages of the siblings -/
structure SiblingAges where
  maria : ℕ
  ann : ℕ
  david : ℕ
  ethan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : SiblingAges) : Prop :=
  ages.maria = ages.ann - 3 ∧
  ages.maria - 4 = (ages.ann - 4) / 2 ∧
  ages.david = ages.maria + 2 ∧
  (ages.david - 2) + (ages.ann - 2) = 3 * (ages.maria - 2) ∧
  ages.ethan = ages.david - ages.maria ∧
  ages.ann - ages.ethan = 8

/-- The theorem stating the ages of the siblings -/
theorem sibling_ages : 
  ∃ (ages : SiblingAges), satisfiesConditions ages ∧ 
    ages.maria = 7 ∧ ages.ann = 10 ∧ ages.david = 9 ∧ ages.ethan = 2 := by
  sorry

end sibling_ages_l3102_310230


namespace g_g_is_odd_l3102_310295

def f (x : ℝ) := x^3

def g (x : ℝ) := f (f x)

theorem g_g_is_odd (h1 : ∀ x, f (-x) = -f x) : 
  ∀ x, g (g (-x)) = -(g (g x)) := by sorry

end g_g_is_odd_l3102_310295


namespace solve_fruit_salad_problem_l3102_310277

def fruit_salad_problem (alaya_salads : ℕ) (angel_multiplier : ℕ) : Prop :=
  let angel_salads := angel_multiplier * alaya_salads
  let total_salads := alaya_salads + angel_salads
  total_salads = 600

theorem solve_fruit_salad_problem :
  fruit_salad_problem 200 2 := by
  sorry

end solve_fruit_salad_problem_l3102_310277


namespace puzzle_solution_l3102_310259

theorem puzzle_solution (a b c d : ℕ+) 
  (h1 : a^6 = b^5) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 25) : 
  d - b = 561 := by
  sorry

end puzzle_solution_l3102_310259


namespace triangle_dot_product_l3102_310222

-- Define the triangle ABC
theorem triangle_dot_product (A B C : ℝ × ℝ) :
  -- Given conditions
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let S := abs ((A.1 - C.1) * (B.2 - C.2) - (A.2 - C.2) * (B.1 - C.1)) / 2
  -- Hypothesis
  AC = 8 →
  BC = 5 →
  S = 10 * Real.sqrt 3 →
  -- Conclusion
  ((B.1 - C.1) * (C.1 - A.1) + (B.2 - C.2) * (C.2 - A.2) = 20 ∨
   (B.1 - C.1) * (C.1 - A.1) + (B.2 - C.2) * (C.2 - A.2) = -20) :=
by
  sorry


end triangle_dot_product_l3102_310222


namespace speed_ratio_l3102_310270

/-- The race scenario where A and B run at different speeds and finish at the same time -/
structure RaceScenario where
  speed_A : ℝ
  speed_B : ℝ
  distance_A : ℝ
  distance_B : ℝ
  finish_time : ℝ

/-- The conditions of the race -/
def race_conditions (r : RaceScenario) : Prop :=
  r.distance_A = 84 ∧ 
  r.distance_B = 42 ∧ 
  r.finish_time = r.distance_A / r.speed_A ∧
  r.finish_time = r.distance_B / r.speed_B

/-- The theorem stating the ratio of A's speed to B's speed -/
theorem speed_ratio (r : RaceScenario) (h : race_conditions r) : 
  r.speed_A / r.speed_B = 2 := by
  sorry


end speed_ratio_l3102_310270


namespace square_root_of_sixteen_l3102_310279

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_sixteen_l3102_310279


namespace s_iff_q_r_iff_q_p_necessary_for_s_l3102_310267

-- Define the propositions
variable (p q r s : Prop)

-- Define the given conditions
axiom p_necessary_for_r : r → p
axiom q_necessary_for_r : r → q
axiom s_sufficient_for_r : s → r
axiom q_sufficient_for_s : q → s

-- Theorem statements
theorem s_iff_q : s ↔ q := by sorry

theorem r_iff_q : r ↔ q := by sorry

theorem p_necessary_for_s : s → p := by sorry

end s_iff_q_r_iff_q_p_necessary_for_s_l3102_310267


namespace opposite_of_negative_nine_l3102_310292

theorem opposite_of_negative_nine :
  ∃ x : ℤ, x + (-9) = 0 ∧ x = 9 :=
sorry

end opposite_of_negative_nine_l3102_310292


namespace custom_op_equation_solution_l3102_310234

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a^2 + b^2 - a*b

-- State the theorem
theorem custom_op_equation_solution :
  ∀ x : ℝ, custom_op x (x - 1) = 3 ↔ x = 2 ∨ x = -1 := by
  sorry

end custom_op_equation_solution_l3102_310234


namespace original_price_calculation_l3102_310264

theorem original_price_calculation (decreased_price : ℝ) (decrease_percentage : ℝ) :
  decreased_price = 620 ∧ decrease_percentage = 20 →
  (100 - decrease_percentage) / 100 * (100 / (100 - decrease_percentage) * decreased_price) = 775 := by
  sorry

end original_price_calculation_l3102_310264


namespace class_project_funding_l3102_310290

/-- Calculates the total amount gathered by a class for a project --/
def total_amount_gathered (total_students : ℕ) (full_payment : ℕ) (half_paying_students : ℕ) : ℕ :=
  let full_paying_students := total_students - half_paying_students
  let full_amount := full_paying_students * full_payment
  let half_amount := half_paying_students * (full_payment / 2)
  full_amount + half_amount

/-- Proves that the class gathered $1150 for their project --/
theorem class_project_funding :
  total_amount_gathered 25 50 4 = 1150 := by
  sorry

end class_project_funding_l3102_310290


namespace motorcyclist_speed_l3102_310281

theorem motorcyclist_speed 
  (hiker_speed : ℝ)
  (time_to_stop : ℝ)
  (time_to_catch_up : ℝ)
  (h1 : hiker_speed = 6)
  (h2 : time_to_stop = 0.2)
  (h3 : time_to_catch_up = 0.8) :
  ∃ (motorcyclist_speed : ℝ),
    motorcyclist_speed * time_to_stop = 
    hiker_speed * (time_to_stop + time_to_catch_up) ∧
    motorcyclist_speed = 30 := by
  sorry

end motorcyclist_speed_l3102_310281


namespace rainfall_problem_l3102_310204

/-- Rainfall problem -/
theorem rainfall_problem (monday_hours : ℝ) (monday_rate : ℝ) (tuesday_rate : ℝ)
  (wednesday_hours : ℝ) (total_rainfall : ℝ)
  (h1 : monday_hours = 7)
  (h2 : monday_rate = 1)
  (h3 : tuesday_rate = 2)
  (h4 : wednesday_hours = 2)
  (h5 : total_rainfall = 23)
  (h6 : wednesday_rate = 2 * tuesday_rate) :
  ∃ tuesday_hours : ℝ,
    tuesday_hours = 4 ∧
    total_rainfall = monday_hours * monday_rate +
                     tuesday_hours * tuesday_rate +
                     wednesday_hours * wednesday_rate :=
by sorry

end rainfall_problem_l3102_310204


namespace polynomial_coefficient_product_l3102_310203

theorem polynomial_coefficient_product (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x^2 - 2) * (x - 1)^7 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + 
                                     a₄*(x-1)^4 + a₅*(x-1)^5 + a₆*(x-1)^6 + 
                                     a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9) →
  (a₁ + a₃ + a₅ + a₇ + a₉ + 2) * (a₂ + a₄ + a₆ + a₈) = 4 := by
sorry

end polynomial_coefficient_product_l3102_310203


namespace scooter_profit_percentage_l3102_310289

/-- Calculates the profit percentage for a scooter sale given specific conditions -/
theorem scooter_profit_percentage 
  (initial_price : ℝ)
  (initial_repair_rate : ℝ)
  (additional_maintenance : ℝ)
  (safety_upgrade_rate : ℝ)
  (sales_tax_rate : ℝ)
  (selling_price : ℝ)
  (h1 : initial_price = 4700)
  (h2 : initial_repair_rate = 0.1)
  (h3 : additional_maintenance = 500)
  (h4 : safety_upgrade_rate = 0.05)
  (h5 : sales_tax_rate = 0.12)
  (h6 : selling_price = 5800) :
  let initial_repair := initial_price * initial_repair_rate
  let total_repair := initial_repair + additional_maintenance
  let safety_upgrade := total_repair * safety_upgrade_rate
  let total_cost := initial_price + total_repair + safety_upgrade
  let sales_tax := selling_price * sales_tax_rate
  let total_selling_price := selling_price + sales_tax
  let profit := total_selling_price - total_cost
  let profit_percentage := (profit / total_cost) * 100
  ∃ ε > 0, abs (profit_percentage - 13.60) < ε :=
by sorry

end scooter_profit_percentage_l3102_310289


namespace max_rice_plates_l3102_310237

def chapati_count : ℕ := 16
def chapati_cost : ℕ := 6
def mixed_veg_count : ℕ := 7
def mixed_veg_cost : ℕ := 70
def ice_cream_count : ℕ := 6
def rice_cost : ℕ := 45
def total_paid : ℕ := 985

theorem max_rice_plates (rice_count : ℕ) : 
  rice_count * rice_cost + 
  chapati_count * chapati_cost + 
  mixed_veg_count * mixed_veg_cost ≤ total_paid →
  rice_count ≤ 8 :=
by sorry

end max_rice_plates_l3102_310237


namespace matrix_power_four_l3102_310205

/-- The fourth power of a specific 2x2 matrix equals a specific result -/
theorem matrix_power_four : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]
  A^4 = !![(-8 : ℝ), 8 * Real.sqrt 3; -8 * Real.sqrt 3, (-8 : ℝ)] := by
  sorry

end matrix_power_four_l3102_310205


namespace log_product_interval_l3102_310214

theorem log_product_interval :
  1 < Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 ∧
  Real.log 6 / Real.log 5 * Real.log 7 / Real.log 6 * Real.log 8 / Real.log 7 < 2 :=
by sorry

end log_product_interval_l3102_310214


namespace hannah_total_spent_l3102_310240

def sweatshirt_count : ℕ := 3
def tshirt_count : ℕ := 2
def sweatshirt_price : ℕ := 15
def tshirt_price : ℕ := 10

theorem hannah_total_spent :
  sweatshirt_count * sweatshirt_price + tshirt_count * tshirt_price = 65 :=
by sorry

end hannah_total_spent_l3102_310240


namespace toy_sword_cost_l3102_310228

theorem toy_sword_cost (total_spent : ℕ) (lego_cost : ℕ) (play_dough_cost : ℕ)
  (lego_sets : ℕ) (toy_swords : ℕ) (play_doughs : ℕ) :
  total_spent = 1940 →
  lego_cost = 250 →
  play_dough_cost = 35 →
  lego_sets = 3 →
  toy_swords = 7 →
  play_doughs = 10 →
  ∃ (sword_cost : ℕ),
    sword_cost = 120 ∧
    total_spent = lego_cost * lego_sets + sword_cost * toy_swords + play_dough_cost * play_doughs :=
by sorry

end toy_sword_cost_l3102_310228


namespace salary_reduction_percentage_l3102_310245

theorem salary_reduction_percentage (initial_increase : Real) (net_increase : Real) (reduction : Real) : 
  initial_increase = 0.25 →
  net_increase = 0.0625 →
  (1 + initial_increase) * (1 - reduction) = 1 + net_increase →
  reduction = 0.15 := by
sorry

end salary_reduction_percentage_l3102_310245


namespace work_fraction_after_twenty_days_l3102_310297

/-- Proves that the fraction of work completed after 20 days is 15/64 -/
theorem work_fraction_after_twenty_days 
  (W : ℝ) -- Total work to be done
  (initial_workers : ℕ := 10) -- Initial number of workers
  (initial_duration : ℕ := 100) -- Initial planned duration in days
  (work_time : ℕ := 20) -- Time worked before firing workers
  (fired_workers : ℕ := 2) -- Number of workers fired
  (remaining_time : ℕ := 75) -- Time to complete the remaining work
  (F : ℝ) -- Fraction of work completed after 20 days
  (h1 : initial_workers * (W / initial_duration) = work_time * (F * W / work_time)) -- Work rate equality for first 20 days
  (h2 : (initial_workers - fired_workers) * ((1 - F) * W / remaining_time) = initial_workers * (W / initial_duration)) -- Work rate equality for remaining work
  : F = 15 / 64 := by
  sorry

end work_fraction_after_twenty_days_l3102_310297


namespace electrolysis_mass_proportionality_l3102_310252

/-- Represents the mass of metal deposited during electrolysis -/
noncomputable def mass_deposited (current : ℝ) (time : ℝ) (ion_charge : ℝ) : ℝ :=
  sorry

/-- The mass deposited is directly proportional to the current -/
axiom mass_prop_current (time : ℝ) (ion_charge : ℝ) (k : ℝ) :
  ∀ current₁ current₂ : ℝ, mass_deposited (k * current₁) time ion_charge = k * mass_deposited current₂ time ion_charge

/-- The mass deposited is directly proportional to the time -/
axiom mass_prop_time (current : ℝ) (ion_charge : ℝ) (k : ℝ) :
  ∀ time₁ time₂ : ℝ, mass_deposited current (k * time₁) ion_charge = k * mass_deposited current time₂ ion_charge

/-- The mass deposited is inversely proportional to the ion charge -/
axiom mass_inv_prop_charge (current : ℝ) (time : ℝ) (k : ℝ) :
  ∀ charge₁ charge₂ : ℝ, charge₁ ≠ 0 → charge₂ ≠ 0 →
    mass_deposited current time (k * charge₁) = (1 / k) * mass_deposited current time charge₂

theorem electrolysis_mass_proportionality :
  (∀ k current time charge, mass_deposited (k * current) time charge = k * mass_deposited current time charge) ∧
  (∀ k current time charge, mass_deposited current (k * time) charge = k * mass_deposited current time charge) ∧
  ¬(∀ k current time charge, charge ≠ 0 → mass_deposited current time (k * charge) = k * mass_deposited current time charge) :=
by sorry

end electrolysis_mass_proportionality_l3102_310252


namespace no_valid_numbers_l3102_310256

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧  -- 3-digit number
  (n / 100 + (n / 10) % 10 + n % 10 = 27) ∧  -- digit-sum is 27
  n % 2 = 0 ∧  -- even number
  n % 10 = 4  -- ends in 4

theorem no_valid_numbers : ¬∃ (n : ℕ), is_valid_number n := by
  sorry

end no_valid_numbers_l3102_310256


namespace horners_method_for_f_l3102_310229

def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

theorem horners_method_for_f :
  f 3 = 21324 := by
sorry

end horners_method_for_f_l3102_310229


namespace cube_root_problem_l3102_310209

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end cube_root_problem_l3102_310209


namespace labyrinth_paths_count_l3102_310247

/-- Represents a point in the labyrinth --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a direction of movement in the labyrinth --/
inductive Direction
  | Right
  | Down
  | Up

/-- Represents the labyrinth structure --/
structure Labyrinth where
  entrance : Point
  exit : Point
  branchPoints : List Point
  isValidMove : Point → Direction → Bool

/-- Counts the number of paths from a given point to the exit --/
def countPaths (lab : Labyrinth) (start : Point) : ℕ :=
  sorry

/-- The main theorem stating that there are 16 paths in the given labyrinth --/
theorem labyrinth_paths_count (lab : Labyrinth) : 
  countPaths lab lab.entrance = 16 :=
  sorry

end labyrinth_paths_count_l3102_310247


namespace age_difference_in_decades_l3102_310248

/-- Given that the sum of x's and y's ages is 18 years greater than the sum of y's and z's ages,
    prove that z is 1.8 decades younger than x. -/
theorem age_difference_in_decades (x y z : ℕ) (h : x + y = y + z + 18) :
  (x - z : ℚ) / 10 = 1.8 := by sorry

end age_difference_in_decades_l3102_310248


namespace point_in_fourth_quadrant_l3102_310220

theorem point_in_fourth_quadrant (a : ℝ) (h : a < -1) :
  let x := a^2 - 2*a - 1
  let y := (a + 1) / |a + 1|
  x > 0 ∧ y < 0 := by
  sorry

end point_in_fourth_quadrant_l3102_310220


namespace triangle_area_72_l3102_310253

/-- 
Given a right triangle with vertices (0, 0), (x, 3x), and (x, 0),
prove that if its area is 72 square units and x > 0, then x = 4√3.
-/
theorem triangle_area_72 (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end triangle_area_72_l3102_310253


namespace pool_filling_time_l3102_310266

/-- Represents the volume of water in a pool as a function of time -/
def water_volume (t : ℕ) : ℝ := sorry

/-- The full capacity of the pool -/
def full_capacity : ℝ := sorry

theorem pool_filling_time :
  (∀ t, water_volume (t + 1) = 2 * water_volume t) →  -- Volume doubles every hour
  (water_volume 8 = full_capacity) →                  -- Full capacity reached in 8 hours
  (water_volume 6 = full_capacity / 2) :=             -- Half capacity reached in 6 hours
by sorry

end pool_filling_time_l3102_310266


namespace ratio_equality_l3102_310243

theorem ratio_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hbc : b/c = 2005) (hcb : c/b = 2005) :
  (b + c) / (a + b) = 2005 := by
  sorry

end ratio_equality_l3102_310243


namespace crayon_distribution_l3102_310221

/-- The problem of distributing crayons among Fred, Benny, Jason, and Sarah. -/
theorem crayon_distribution (total : ℕ) (fred benny jason sarah : ℕ) : 
  total = 96 →
  fred = 2 * benny →
  jason = 3 * sarah →
  benny = 12 →
  total = fred + benny + jason + sarah →
  fred = 24 ∧ benny = 12 ∧ jason = 45 ∧ sarah = 15 := by
  sorry

#check crayon_distribution

end crayon_distribution_l3102_310221


namespace problem_statement_l3102_310291

theorem problem_statement (x₁ x₂ x₃ x₄ n : ℝ) 
  (h1 : x₁ ≠ x₂)
  (h2 : (x₁ + x₃) * (x₁ + x₄) = n - 10)
  (h3 : (x₂ + x₃) * (x₂ + x₄) = n - 10)
  (h4 : x₁ + x₂ + x₃ + x₄ = 0) :
  let p := (x₁ + x₃) * (x₂ + x₃) + (x₁ + x₄) * (x₂ + x₄)
  p = 2 * n - 20 := by
  sorry


end problem_statement_l3102_310291


namespace snake_length_problem_l3102_310200

theorem snake_length_problem (penny_snake : ℕ) (jake_snake : ℕ) : 
  jake_snake = penny_snake + 12 →
  jake_snake + penny_snake = 70 →
  jake_snake = 41 := by
sorry

end snake_length_problem_l3102_310200


namespace uniform_transformation_l3102_310213

theorem uniform_transformation (a₁ : ℝ) : 
  a₁ ∈ Set.Icc 0 1 → (8 * a₁ - 2) ∈ Set.Icc (-2) 6 := by
  sorry

end uniform_transformation_l3102_310213


namespace original_equals_scientific_l3102_310225

/-- Represent a number in scientific notation -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_range : 1 ≤ mantissa ∧ mantissa < 10

/-- The given number -/
def original_number : ℕ := 3010000000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation := {
  mantissa := 3.01,
  exponent := 9,
  mantissa_range := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.mantissa * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end original_equals_scientific_l3102_310225


namespace quadrilateral_propositions_l3102_310274

-- Define a quadrilateral
structure Quadrilateral :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

-- Define a property for quadrilaterals with four equal sides
def has_equal_sides (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4

-- Define a property for squares
def is_square (q : Quadrilateral) : Prop :=
  has_equal_sides q ∧ q.side1 = q.side2 -- This is a simplified definition

theorem quadrilateral_propositions :
  (∃ q : Quadrilateral, has_equal_sides q ∧ ¬is_square q) ∧
  (∀ q : Quadrilateral, is_square q → has_equal_sides q) ∧
  (∀ q : Quadrilateral, ¬is_square q → ¬has_equal_sides q) ∧
  (∃ q : Quadrilateral, ¬is_square q ∧ has_equal_sides q) :=
sorry

end quadrilateral_propositions_l3102_310274


namespace product_from_lcm_gcd_l3102_310202

theorem product_from_lcm_gcd (a b : ℤ) : 
  Int.lcm a b = 42 → Int.gcd a b = 7 → a * b = 294 := by
  sorry

end product_from_lcm_gcd_l3102_310202


namespace two_people_completion_time_l3102_310215

/-- Represents the amount of work done on a given day -/
def work_on_day (n : ℕ) : ℕ := 2^(n-1)

/-- Represents the total amount of work done up to and including a given day -/
def total_work (n : ℕ) : ℕ := 2^n - 1

/-- The number of days it takes one person to complete the job -/
def days_for_one_person : ℕ := 12

/-- The theorem stating that two people working together will complete the job in 11 days -/
theorem two_people_completion_time :
  ∃ (n : ℕ), n = 11 ∧ total_work n = total_work days_for_one_person := by
  sorry

end two_people_completion_time_l3102_310215


namespace unique_triple_l3102_310219

theorem unique_triple : 
  ∃! (a b c : ℕ), 
    (10 ≤ b ∧ b ≤ 99) ∧ 
    (10 ≤ c ∧ c ≤ 99) ∧ 
    (10^4 * a + 100 * b + c = (a + b + c)^3) ∧
    a = 9 ∧ b = 11 ∧ c = 25 := by
  sorry

end unique_triple_l3102_310219


namespace no_solution_for_2015_problems_l3102_310224

theorem no_solution_for_2015_problems : 
  ¬ ∃ (x y z : ℕ), (y - x = z - y) ∧ (x + y + z = 2015) := by
sorry

end no_solution_for_2015_problems_l3102_310224


namespace percent_greater_average_l3102_310298

theorem percent_greater_average (M N : ℝ) (h : M > N) :
  (M - N) / ((M + N) / 2) * 100 = 200 * (M - N) / (M + N) := by
  sorry

end percent_greater_average_l3102_310298


namespace total_ways_to_draw_balls_l3102_310272

-- Define the total number of balls
def total_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 2

-- Define the number of white balls
def white_balls : ℕ := 6

-- Define a function to calculate the number of ways to draw balls
def ways_to_draw_balls : ℕ := 
  -- Sum of ways for each possible draw (1st, 2nd, 3rd, and 4th)
  1 + 2 + 3 + 4

-- Theorem statement
theorem total_ways_to_draw_balls : 
  ways_to_draw_balls = 10 :=
by
  sorry -- Proof is omitted as per instructions

end total_ways_to_draw_balls_l3102_310272


namespace mens_wages_l3102_310260

/-- Proves that the total wages of men is Rs. 30 given the problem conditions -/
theorem mens_wages (W : ℕ) (total_earnings : ℕ) : 
  (5 : ℕ) = W →  -- 5 men are equal to W women
  W = 8 →        -- W women are equal to 8 boys
  total_earnings = 90 →  -- Total earnings of all people is Rs. 90
  (5 : ℕ) * (total_earnings / 15) = 30 := by
sorry

end mens_wages_l3102_310260


namespace reflection_matrix_conditions_l3102_310294

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, b; -3/4, 1/4]

theorem reflection_matrix_conditions (a b : ℚ) :
  reflection_matrix a b * reflection_matrix a b = 1 ↔ a = -1/4 ∧ b = -3/4 := by
  sorry

end reflection_matrix_conditions_l3102_310294


namespace teacher_age_l3102_310201

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (new_avg_age * (num_students + 1) - num_students * student_avg_age) = 26 := by
  sorry

end teacher_age_l3102_310201


namespace linear_function_properties_l3102_310207

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := k * (x + 3)

-- Define the shifted function
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_properties (k : ℝ) (h : k ≠ 0) :
  (f k (-3) = 0) ∧ 
  (g k 1 = -2 → k = -1) ∧
  (k < 0 → ∀ x₁ x₂ y₁ y₂ : ℝ, 
    f k x₁ = y₁ → f k x₂ = y₂ → y₁ < y₂ → x₁ > x₂) :=
by sorry


end linear_function_properties_l3102_310207


namespace jessica_purchases_total_cost_l3102_310216

/-- The cost of Jessica's cat toy in dollars -/
def cat_toy_cost : ℚ := 10.22

/-- The cost of Jessica's cage in dollars -/
def cage_cost : ℚ := 11.73

/-- The total cost of Jessica's purchases in dollars -/
def total_cost : ℚ := cat_toy_cost + cage_cost

/-- Theorem stating that the total cost of Jessica's purchases is $21.95 -/
theorem jessica_purchases_total_cost : total_cost = 21.95 := by
  sorry

end jessica_purchases_total_cost_l3102_310216


namespace binomial_16_13_l3102_310299

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by
  sorry

end binomial_16_13_l3102_310299


namespace peggy_record_profit_difference_l3102_310271

/-- Represents the profit difference between two offers for a record collection. -/
def profit_difference (total_records : ℕ) (sammy_price : ℕ) (bryan_price_high : ℕ) (bryan_price_low : ℕ) : ℕ :=
  let sammy_offer := total_records * sammy_price
  let bryan_offer := (total_records / 2) * bryan_price_high + (total_records / 2) * bryan_price_low
  sammy_offer - bryan_offer

/-- Theorem stating the profit difference for Peggy's record collection. -/
theorem peggy_record_profit_difference :
  profit_difference 200 4 6 1 = 100 := by
  sorry

end peggy_record_profit_difference_l3102_310271


namespace trig_identity_l3102_310206

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l3102_310206


namespace sum_of_odd_coefficients_l3102_310227

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end sum_of_odd_coefficients_l3102_310227


namespace sequence_formula_l3102_310286

theorem sequence_formula (a b : ℕ → ℝ) : 
  (∀ n, a n > 0 ∧ b n > 0) →  -- Each term is positive
  (∀ n, 2 * b n = a n + a (n + 1)) →  -- Arithmetic sequence condition
  (∀ n, (a (n + 1))^2 = b n * b (n + 1)) →  -- Geometric sequence condition
  a 1 = 1 →  -- Initial condition
  a 2 = 3 →  -- Initial condition
  (∀ n, a n = (n^2 + n) / 2) :=  -- General term formula
by sorry

end sequence_formula_l3102_310286
