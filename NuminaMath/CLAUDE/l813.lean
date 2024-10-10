import Mathlib

namespace safe_round_trip_exists_l813_81337

/-- Represents the cycle of a dragon's fire-breathing pattern -/
structure DragonCycle where
  active : ℕ
  sleep : ℕ

/-- Represents the travel times for the journey -/
structure TravelTimes where
  road : ℕ
  path : ℕ

/-- Checks if a given hour is safe from both dragons -/
def is_safe (h : ℕ) (d1 d2 : DragonCycle) : Prop :=
  h % (d1.active + d1.sleep) > d1.active ∧ 
  h % (d2.active + d2.sleep) > d2.active

/-- Checks if a round trip is possible within a given time frame -/
def round_trip_possible (start : ℕ) (t : TravelTimes) (d1 d2 : DragonCycle) : Prop :=
  ∀ h : ℕ, start ≤ h ∧ h < start + 2 * (t.road + t.path) → is_safe h d1 d2

/-- Main theorem: There exists a safe starting time for the round trip -/
theorem safe_round_trip_exists (t : TravelTimes) (d1 d2 : DragonCycle) : 
  ∃ start : ℕ, round_trip_possible start t d1 d2 :=
sorry

end safe_round_trip_exists_l813_81337


namespace k_range_for_specific_inequalities_l813_81362

/-- Given a real number k, this theorem states that if the system of inequalities
    x^2 - x - 2 > 0 and 2x^2 + (2k+5)x + 5k < 0 has {-2} as its only integer solution,
    then k must be in the range [-3, 2). -/
theorem k_range_for_specific_inequalities (k : ℝ) :
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 :=
sorry

end k_range_for_specific_inequalities_l813_81362


namespace range_of_a_range_of_a_for_local_minimum_l813_81372

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 2*a) * (x^2 + a^2*x + 2*a^3)

/-- Theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ x, x < 0 → (3*x^2 + 2*(a^2 - 2*a)*x < 0)) ↔ (a < 0 ∨ a > 2) :=
sorry

/-- Main theorem proving the range of a -/
theorem range_of_a_for_local_minimum :
  {a : ℝ | IsLocalMin (f a) 0} = {a : ℝ | a < 0 ∨ a > 2} :=
sorry

end range_of_a_range_of_a_for_local_minimum_l813_81372


namespace cubic_inverse_exists_l813_81346

noncomputable def k : ℝ := Real.sqrt 3

theorem cubic_inverse_exists (x y z : ℚ) (h : x + y * k + z * k^2 ≠ 0) :
  ∃ u v w : ℚ, (x + y * k + z * k^2) * (u + v * k + w * k^2) = 1 := by
  sorry

end cubic_inverse_exists_l813_81346


namespace eight_by_ten_grid_theorem_l813_81395

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the number of squares not intersected by diagonals in a grid -/
def squares_not_intersected (g : Grid) : ℕ :=
  sorry

/-- Theorem: In an 8 × 10 grid, 48 squares are not intersected by either diagonal -/
theorem eight_by_ten_grid_theorem : 
  let g : Grid := { rows := 8, cols := 10 }
  squares_not_intersected g = 48 := by
  sorry

end eight_by_ten_grid_theorem_l813_81395


namespace purchase_total_l813_81345

/-- The total amount spent on a vacuum cleaner and dishwasher after applying a coupon -/
theorem purchase_total (vacuum_cost dishwasher_cost coupon_value : ℕ) : 
  vacuum_cost = 250 → 
  dishwasher_cost = 450 → 
  coupon_value = 75 → 
  vacuum_cost + dishwasher_cost - coupon_value = 625 := by
sorry

end purchase_total_l813_81345


namespace expected_adjacent_pairs_l813_81343

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (n_boys n_girls : ℕ) : 
  let n_total := n_boys + n_girls
  let n_pairs := n_total - 1
  let p_boy := n_boys / n_total
  let p_girl := n_girls / n_total
  let p_adjacent := p_boy * p_girl + p_girl * p_boy
  n_boys = 10 → n_girls = 15 → n_pairs * p_adjacent = 12 := by
  sorry

end expected_adjacent_pairs_l813_81343


namespace tom_family_plates_l813_81312

/-- Calculates the total number of plates used by a family during a stay -/
def total_plates_used (family_size : ℕ) (days : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) : ℕ :=
  family_size * days * meals_per_day * plates_per_meal

/-- Theorem: The total number of plates used by Tom's family during their 4-day stay is 144 -/
theorem tom_family_plates : 
  total_plates_used 6 4 3 2 = 144 := by
  sorry


end tom_family_plates_l813_81312


namespace tina_brownies_l813_81386

theorem tina_brownies (total_brownies : ℕ) (days : ℕ) (husband_daily : ℕ) (shared_guests : ℕ) (leftover : ℕ) :
  total_brownies = 24 →
  days = 5 →
  husband_daily = 1 →
  shared_guests = 4 →
  leftover = 5 →
  (total_brownies - (days * husband_daily + shared_guests + leftover)) / (days * 2) = 1 := by
  sorry

end tina_brownies_l813_81386


namespace vector_sum_equality_l813_81331

/-- Given two 2D vectors a and b, prove that 2a + 3b equals the specified result. -/
theorem vector_sum_equality (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-1, 2) →
  2 • a + 3 • b = (1, 8) := by
  sorry

end vector_sum_equality_l813_81331


namespace subset_M_l813_81371

def M : Set ℝ := {x : ℝ | x > -1}

theorem subset_M : {0} ⊆ M := by sorry

end subset_M_l813_81371


namespace largest_negative_and_smallest_absolute_l813_81311

theorem largest_negative_and_smallest_absolute : ∃ (a b : ℤ),
  (∀ x : ℤ, x < 0 → x ≤ a) ∧
  (∀ y : ℤ, |b| ≤ |y|) ∧
  b - 4*a = 4 :=
sorry

end largest_negative_and_smallest_absolute_l813_81311


namespace power_function_value_l813_81317

/-- A power function is a function of the form f(x) = x^α for some real α -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem power_function_value (f : ℝ → ℝ) (h1 : IsPowerFunction f) (h2 : f 2 / f 4 = 1 / 2) :
  f 2 = 2 := by
  sorry

end power_function_value_l813_81317


namespace businessmen_neither_coffee_nor_tea_l813_81383

theorem businessmen_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 30)
  (h2 : coffee = 15)
  (h3 : tea = 12)
  (h4 : both = 7) :
  total - (coffee + tea - both) = 10 := by
  sorry

end businessmen_neither_coffee_nor_tea_l813_81383


namespace original_price_calculation_l813_81339

/-- The original price of a meal given the total amount paid and various fees and discounts -/
theorem original_price_calculation (total_paid : ℝ) (discount_rate : ℝ) (sales_tax_rate : ℝ) 
  (service_fee_rate : ℝ) (tip_rate : ℝ) (h_total : total_paid = 165) 
  (h_discount : discount_rate = 0.15) (h_sales_tax : sales_tax_rate = 0.10) 
  (h_service_fee : service_fee_rate = 0.05) (h_tip : tip_rate = 0.20) :
  ∃ (P : ℝ), P = total_paid / ((1 - discount_rate) * (1 + sales_tax_rate + service_fee_rate) * (1 + tip_rate)) := by
  sorry

#eval (165 : Float) / (0.85 * 1.15 * 1.20)

end original_price_calculation_l813_81339


namespace total_selections_is_57_l813_81357

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement where
  total_circles : Nat
  horizontal_rows : List Nat
  diagonal_length : Nat

/-- Calculates the number of ways to select three consecutive circles in a row -/
def consecutive_selections (row_length : Nat) : Nat :=
  max (row_length - 2) 0

/-- Calculates the total number of ways to select three consecutive circles in the figure -/
def total_selections (arrangement : CircleArrangement) : Nat :=
  let horizontal_selections := arrangement.horizontal_rows.map consecutive_selections |>.sum
  let diagonal_selections := List.range arrangement.diagonal_length |>.map consecutive_selections |>.sum
  horizontal_selections + 2 * diagonal_selections

/-- The main theorem stating that the total number of selections is 57 -/
theorem total_selections_is_57 (arrangement : CircleArrangement) :
  arrangement.total_circles = 33 →
  arrangement.horizontal_rows = [6, 5, 4, 3, 2, 1] →
  arrangement.diagonal_length = 6 →
  total_selections arrangement = 57 := by
  sorry

#eval total_selections { total_circles := 33, horizontal_rows := [6, 5, 4, 3, 2, 1], diagonal_length := 6 }

end total_selections_is_57_l813_81357


namespace triangle_problem_l813_81398

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →   -- A is acute
  0 < B ∧ B < π/2 →   -- B is acute
  0 < C ∧ C < π/2 →   -- C is acute
  Real.sqrt 3 * c = 2 * a * Real.sin C →  -- √3c = 2a sin C
  a = Real.sqrt 7 →  -- a = √7
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 →  -- Area of triangle ABC
  (A = π/3) ∧ (a + b + c = Real.sqrt 7 + 5) := by
  sorry

end triangle_problem_l813_81398


namespace percent_students_in_school_l813_81348

/-- Given that 40% of students are learning from home and the remaining students are equally divided
    into two groups with only one group attending school on any day, prove that the percent of
    students present in school is 30%. -/
theorem percent_students_in_school :
  let total_percent : ℚ := 100
  let home_percent : ℚ := 40
  let remaining_percent : ℚ := total_percent - home_percent
  let in_school_percent : ℚ := remaining_percent / 2
  in_school_percent = 30 := by sorry

end percent_students_in_school_l813_81348


namespace intersecting_lines_y_intercept_sum_l813_81340

/-- Given two lines that intersect at a specific point, prove their y-intercepts sum to zero -/
theorem intersecting_lines_y_intercept_sum (a b : ℝ) : 
  (3 = (1/3) * (-3) + a) ∧ (-3 = (1/3) * 3 + b) → a + b = 0 := by
  sorry

end intersecting_lines_y_intercept_sum_l813_81340


namespace ethanol_in_fuel_tank_l813_81344

/-- Calculates the total amount of ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- The theorem states that the total amount of ethanol in the specified fuel mixture is 30 gallons -/
theorem ethanol_in_fuel_tank :
  total_ethanol 208 82 0.12 0.16 = 30 := by
  sorry

end ethanol_in_fuel_tank_l813_81344


namespace prime_square_mod_180_l813_81352

theorem prime_square_mod_180 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ 
  (∀ (r : Nat), p^2 % 180 = r → (r = r₁ ∨ r = r₂)) :=
sorry

end prime_square_mod_180_l813_81352


namespace quadratic_vertex_l813_81396

/-- The quadratic function f(x) = 2(x-3)^2 + 1 has its vertex at (3, 1). -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * (x - 3)^2 + 1
  (∀ x, f x ≥ f 3) ∧ f 3 = 1 := by sorry

end quadratic_vertex_l813_81396


namespace four_circles_in_larger_circle_l813_81375

-- Define a circle with a center point and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the property of a circle being internally tangent to another circle
def internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

theorem four_circles_in_larger_circle (c1 c2 c3 c4 large : Circle) :
  c1.radius = 2 ∧ c2.radius = 2 ∧ c3.radius = 2 ∧ c4.radius = 2 →
  externally_tangent c1 c2 ∧ externally_tangent c1 c3 ∧ externally_tangent c1 c4 ∧
  externally_tangent c2 c3 ∧ externally_tangent c2 c4 ∧ externally_tangent c3 c4 →
  internally_tangent c1 large ∧ internally_tangent c2 large ∧
  internally_tangent c3 large ∧ internally_tangent c4 large →
  large.radius = 4 := by
  sorry

end four_circles_in_larger_circle_l813_81375


namespace green_sequins_per_row_l813_81318

theorem green_sequins_per_row (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_rows : Nat) (purple_per_row : Nat) (green_rows : Nat) (total_sequins : Nat)
  (h1 : blue_rows = 6) (h2 : blue_per_row = 8)
  (h3 : purple_rows = 5) (h4 : purple_per_row = 12)
  (h5 : green_rows = 9) (h6 : total_sequins = 162) :
  (total_sequins - (blue_rows * blue_per_row + purple_rows * purple_per_row)) / green_rows = 6 := by
  sorry

end green_sequins_per_row_l813_81318


namespace distance_EC_l813_81313

/-- Given five points A, B, C, D, E on a line, with known distances between consecutive points,
    prove that the distance between E and C is 150. -/
theorem distance_EC (A B C D E : ℝ) 
  (h_AB : |A - B| = 30)
  (h_BC : |B - C| = 80)
  (h_CD : |C - D| = 236)
  (h_DE : |D - E| = 86)
  (h_EA : |E - A| = 40)
  (h_line : ∃ (t : ℝ → ℝ), t A < t B ∧ t B < t C ∧ t C < t D ∧ t D < t E) :
  |E - C| = 150 := by
  sorry

end distance_EC_l813_81313


namespace annas_meal_cost_difference_l813_81393

/-- Represents the cost of Anna's meals -/
def annas_meals (bagel_price cream_cheese_price orange_juice_price orange_juice_discount
                 sandwich_price avocado_price milk_price milk_discount : ℚ) : ℚ :=
  let breakfast_cost := bagel_price + cream_cheese_price + orange_juice_price * (1 - orange_juice_discount)
  let lunch_cost := sandwich_price + avocado_price + milk_price * (1 - milk_discount)
  lunch_cost - breakfast_cost

/-- The difference between Anna's lunch and breakfast costs is $4.14 -/
theorem annas_meal_cost_difference :
  annas_meals 0.95 0.50 1.25 0.32 4.65 0.75 1.15 0.10 = 4.14 := by
  sorry

end annas_meal_cost_difference_l813_81393


namespace problem_2012_l813_81390

theorem problem_2012 (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (eq1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (eq2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c*d)^2012 - (a*b)^2012 = 2011 := by
sorry

end problem_2012_l813_81390


namespace num_ways_to_place_pawns_l813_81322

/-- Represents a chess board configuration -/
def ChessBoard := Fin 5 → Fin 5

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- Checks if a chess board configuration is valid (no more than one pawn per row and column) -/
def is_valid_configuration (board : ChessBoard) : Prop :=
  (∀ i j : Fin 5, i ≠ j → board i ≠ board j) ∧
  (∀ i : Fin 5, ∃ j : Fin 5, board j = i)

/-- The number of valid chess board configurations -/
def num_valid_configurations : ℕ := factorial 5

/-- The main theorem stating the number of ways to place five distinct pawns -/
theorem num_ways_to_place_pawns :
  (num_valid_configurations * factorial 5 : ℕ) = 14400 :=
sorry

end num_ways_to_place_pawns_l813_81322


namespace M_equals_N_l813_81320

def M : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def N : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1 ∨ x = 4 * k - 1}

theorem M_equals_N : M = N := by sorry

end M_equals_N_l813_81320


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l813_81341

theorem sum_of_roots_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2004*x + 2021
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = 2004) :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l813_81341


namespace perpendicular_vector_l813_81364

/-- Given three points A, B, C in ℝ³ and a vector a, if a is perpendicular to both AB and AC,
    then a = (1, 1, 1) -/
theorem perpendicular_vector (A B C a : ℝ × ℝ × ℝ) :
  A = (0, 2, 3) →
  B = (-2, 1, 6) →
  C = (1, -1, 5) →
  a.2.2 = 1 →
  (a.1 * (B.1 - A.1) + a.2.1 * (B.2.1 - A.2.1) + a.2.2 * (B.2.2 - A.2.2) = 0) →
  (a.1 * (C.1 - A.1) + a.2.1 * (C.2.1 - A.2.1) + a.2.2 * (C.2.2 - A.2.2) = 0) →
  a = (1, 1, 1) := by
  sorry

end perpendicular_vector_l813_81364


namespace investment_difference_l813_81303

def initial_investment : ℕ := 2000

def alice_multiplier : ℕ := 2
def bob_multiplier : ℕ := 5

def alice_final (initial : ℕ) : ℕ := initial * alice_multiplier
def bob_final (initial : ℕ) : ℕ := initial * bob_multiplier

theorem investment_difference : 
  bob_final initial_investment - alice_final initial_investment = 6000 := by
  sorry

end investment_difference_l813_81303


namespace min_rooms_for_departments_l813_81381

/-- Given two departments with student counts and room constraints, 
    calculate the minimum number of rooms required. -/
theorem min_rooms_for_departments (dept1_count dept2_count : ℕ) : 
  dept1_count = 72 →
  dept2_count = 5824 →
  ∃ (room_size : ℕ), 
    room_size > 0 ∧
    dept1_count % room_size = 0 ∧
    dept2_count % room_size = 0 ∧
    (dept1_count / room_size + dept2_count / room_size) = 737 := by
  sorry

end min_rooms_for_departments_l813_81381


namespace parabola_point_coordinates_l813_81370

theorem parabola_point_coordinates :
  ∀ (x y : ℝ),
  y^2 = 12*x →                           -- Point (x, y) is on the parabola y^2 = 12x
  (x - 3)^2 + y^2 = 9^2 →                -- Point is 9 units away from the focus (3, 0)
  (x = 6 ∧ (y = 6*Real.sqrt 2 ∨ y = -6*Real.sqrt 2)) := by
sorry

end parabola_point_coordinates_l813_81370


namespace largest_integer_with_remainder_l813_81394

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 9 = 7 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 7 → m ≤ n ↔ n = 97 := by
  sorry

end largest_integer_with_remainder_l813_81394


namespace contrapositive_absolute_value_l813_81377

theorem contrapositive_absolute_value (a b : ℝ) :
  (¬(|a| > |b|) → ¬(a > b)) ↔ (|a| ≤ |b| → a ≤ b) := by sorry

end contrapositive_absolute_value_l813_81377


namespace quadratic_root_relation_l813_81334

/-- Given a quadratic equation 3x^2 + 4x + 5 = 0 with roots r and s,
    if we construct a new quadratic equation x^2 + px + q = 0 with roots 2r and 2s,
    then p = 56/9 -/
theorem quadratic_root_relation (r s : ℝ) (p q : ℝ) : 
  (3 * r^2 + 4 * r + 5 = 0) →
  (3 * s^2 + 4 * s + 5 = 0) →
  ((2 * r)^2 + p * (2 * r) + q = 0) →
  ((2 * s)^2 + p * (2 * s) + q = 0) →
  p = 56 / 9 := by
sorry

end quadratic_root_relation_l813_81334


namespace imaginary_part_of_complex_expression_l813_81385

theorem imaginary_part_of_complex_expression (z : ℂ) (h : z = 3 + 4*I) : 
  Complex.im (z + Complex.abs z / z) = 16/5 := by
  sorry

end imaginary_part_of_complex_expression_l813_81385


namespace arithmetic_series_sum_base6_l813_81389

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 6 -/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- The sum of an arithmetic series -/
def arithmeticSeriesSum (a : ℕ) (l : ℕ) (n : ℕ) : ℕ :=
  n * (a + l) / 2

theorem arithmetic_series_sum_base6 :
  let first := 1
  let last := base6ToBase10 55
  let terms := base6ToBase10 55
  let sum := arithmeticSeriesSum first last terms
  (sum = 630) ∧ (base10ToBase6 sum = 2530) := by sorry

end arithmetic_series_sum_base6_l813_81389


namespace least_three_digit_11_heavy_l813_81314

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

theorem least_three_digit_11_heavy : ∀ n : ℕ, 100 ≤ n ∧ n < 108 → ¬(is_11_heavy n) ∧ is_11_heavy 108 := by
  sorry

#check least_three_digit_11_heavy

end least_three_digit_11_heavy_l813_81314


namespace exists_special_function_l813_81310

theorem exists_special_function : ∃ (s : ℚ → Int), 
  (∀ x, s x = 1 ∨ s x = -1) ∧ 
  (∀ x y, x ≠ y → (x * y = 1 ∨ x + y = 0 ∨ x + y = 1) → s x * s y = -1) := by
  sorry

end exists_special_function_l813_81310


namespace min_value_quadratic_min_value_quadratic_achieved_l813_81355

theorem min_value_quadratic (x : ℝ) : 
  7 * x^2 - 28 * x + 1702 ≥ 1674 := by
sorry

theorem min_value_quadratic_achieved : 
  ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 := by
sorry

end min_value_quadratic_min_value_quadratic_achieved_l813_81355


namespace sum_interior_angles_pentagon_l813_81392

-- Define a regular pentagon
def RegularPentagon : Type := Unit

-- Define the function to calculate the sum of interior angles of a polygon
def sumInteriorAngles (n : ℕ) : ℝ := (n - 2) * 180

-- Theorem: The sum of interior angles of a regular pentagon is 540°
theorem sum_interior_angles_pentagon (p : RegularPentagon) :
  sumInteriorAngles 5 = 540 := by sorry

end sum_interior_angles_pentagon_l813_81392


namespace particle_position_after_2023_minutes_l813_81326

/-- Represents the position of a particle -/
structure Position where
  x : ℕ
  y : ℕ

/-- Calculates the time taken to complete n squares -/
def time_for_squares (n : ℕ) : ℕ :=
  n^2 + 5*n

/-- Determines the position of the particle after a given time -/
def particle_position (time : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_after_2023_minutes :
  particle_position 2023 = Position.mk 43 43 := by
  sorry

end particle_position_after_2023_minutes_l813_81326


namespace fraction_before_simplification_l813_81315

theorem fraction_before_simplification
  (n d : ℕ)  -- n and d are the numerator and denominator before simplification
  (h1 : n + d = 80)  -- sum of numerator and denominator is 80
  (h2 : n / d = 3 / 7)  -- fraction simplifies to 3/7
  : n = 24 ∧ d = 56 := by
  sorry

end fraction_before_simplification_l813_81315


namespace at_least_one_non_negative_l813_81329

theorem at_least_one_non_negative (a b c d e f g h : ℝ) :
  (max (a*c + b*d) (max (a*e + b*f) (max (a*g + b*h) (max (c*e + d*f) (max (c*g + d*h) (e*g + f*h)))))) ≥ 0 :=
by sorry

end at_least_one_non_negative_l813_81329


namespace complement_of_A_relative_to_U_l813_81391

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}

theorem complement_of_A_relative_to_U :
  {x ∈ U | x ∉ A} = {2} := by sorry

end complement_of_A_relative_to_U_l813_81391


namespace alloy_mix_solvable_l813_81359

/-- Represents an alloy of copper and tin -/
structure Alloy where
  mass : ℝ
  copper_percentage : ℝ

/-- Represents the problem of mixing two alloys -/
def AlloyMixProblem (alloy1 alloy2 : Alloy) (target_mass : ℝ) (target_percentage : ℝ) :=
  alloy1.mass ≥ 0 ∧
  alloy2.mass ≥ 0 ∧
  alloy1.copper_percentage ≥ 0 ∧ alloy1.copper_percentage ≤ 100 ∧
  alloy2.copper_percentage ≥ 0 ∧ alloy2.copper_percentage ≤ 100 ∧
  target_mass > 0 ∧
  target_percentage ≥ 0 ∧ target_percentage ≤ 100

theorem alloy_mix_solvable (alloy1 alloy2 : Alloy) (target_mass : ℝ) (p : ℝ) :
  AlloyMixProblem alloy1 alloy2 target_mass p →
  (alloy1.mass = 3 ∧ 
   alloy2.mass = 7 ∧ 
   alloy1.copper_percentage = 40 ∧ 
   alloy2.copper_percentage = 30 ∧
   target_mass = 8) →
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ alloy1.mass ∧ 
            0 ≤ target_mass - x ∧ target_mass - x ≤ alloy2.mass ∧
            alloy1.copper_percentage * x / 100 + alloy2.copper_percentage * (target_mass - x) / 100 = target_mass * p / 100) ↔
  (31.25 ≤ p ∧ p ≤ 33.75) :=
by sorry

end alloy_mix_solvable_l813_81359


namespace number_difference_l813_81338

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : |x - y| = 3 := by
  sorry

end number_difference_l813_81338


namespace sound_speed_in_new_rod_l813_81365

/-- The speed of sound in a new rod given experimental data -/
theorem sound_speed_in_new_rod (a b l : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : l > 0) (h4 : b > a) : ∃ v : ℝ,
  v = 3 * l / (2 * (b - a)) ∧
  (∃ (t1 t2 t3 t4 : ℝ),
    t1 > 0 ∧ t2 > 0 ∧ t3 > 0 ∧ t4 > 0 ∧
    t1 + t2 + t3 = a ∧
    t1 = 2 * (t2 + t3) ∧
    t1 + t4 + t3 = b ∧
    t1 + t4 = 2 * t3 ∧
    v = l / t4) :=
by sorry

end sound_speed_in_new_rod_l813_81365


namespace min_rooms_sufficient_l813_81336

/-- The minimum number of hotel rooms required for 100 tourists given k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  let m := k / 2
  if k % 2 = 0 then 100 * (m + 1) else 100 * (m + 1) + 1

/-- Theorem stating that min_rooms provides sufficient rooms for 100 tourists -/
theorem min_rooms_sufficient (k : ℕ) :
  ∀ (arrangement : Fin k → Fin (min_rooms k)),
  ∃ (allocation : Fin 100 → Fin (min_rooms k)),
  (∀ i j, i ≠ j → allocation i ≠ allocation j) ∧
  (∀ i, allocation i ∉ Set.range arrangement) :=
sorry

end min_rooms_sufficient_l813_81336


namespace inverse_composition_l813_81351

-- Define the function f and its inverse
def f : ℝ → ℝ := sorry

def f_inv : ℝ → ℝ := sorry

-- Define the conditions
axiom f_4 : f 4 = 6
axiom f_6 : f 6 = 3
axiom f_3 : f 3 = 7
axiom f_7 : f 7 = 2

-- Define the inverse relationship
axiom f_inverse : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Theorem to prove
theorem inverse_composition :
  f_inv (f_inv 7 + f_inv 6) = 2 := by sorry

end inverse_composition_l813_81351


namespace floor_x_floor_x_eq_42_l813_81308

theorem floor_x_floor_x_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 := by
sorry

end floor_x_floor_x_eq_42_l813_81308


namespace x_value_l813_81328

theorem x_value (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) : x = 15 := by
  sorry

end x_value_l813_81328


namespace repeating_decimal_division_l813_81384

theorem repeating_decimal_division :
  let x : ℚ := 63 / 99
  let y : ℚ := 84 / 99
  x / y = 3 / 4 := by
  sorry

end repeating_decimal_division_l813_81384


namespace train_speed_problem_l813_81373

/-- Given a train that covers a distance in 3 hours at its initial speed,
    and covers the same distance in 1 hour at 450 kmph,
    prove that its initial speed is 150 kmph. -/
theorem train_speed_problem (distance : ℝ) (initial_speed : ℝ) : 
  distance = initial_speed * 3 → distance = 450 * 1 → initial_speed = 150 := by
  sorry

end train_speed_problem_l813_81373


namespace ceiling_floor_difference_l813_81321

theorem ceiling_floor_difference : 
  ⌈(20 : ℝ) / 9 * ⌈(-53 : ℝ) / 4⌉⌉ - ⌊(20 : ℝ) / 9 * ⌊(-53 : ℝ) / 4⌋⌋ = 4 := by
  sorry

end ceiling_floor_difference_l813_81321


namespace sequence_formula_l813_81304

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h : ∀ n : ℕ+, S n = 2 * a n - 1) : 
  ∀ n : ℕ+, a n = 2^(n.val - 1) := by
  sorry

end sequence_formula_l813_81304


namespace marbles_lost_l813_81309

theorem marbles_lost (doug : ℕ) (ed_initial ed_final : ℕ) 
  (h1 : ed_initial = doug + 29)
  (h2 : ed_final = doug + 12) :
  ed_initial - ed_final = 17 := by
sorry

end marbles_lost_l813_81309


namespace waiter_customers_l813_81387

/-- Given a number of tables and the number of women and men at each table,
    calculate the total number of customers. -/
def total_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) : ℕ :=
  num_tables * (women_per_table + men_per_table)

/-- Theorem: The waiter has 90 customers in total. -/
theorem waiter_customers :
  total_customers 9 7 3 = 90 := by
  sorry

#eval total_customers 9 7 3

end waiter_customers_l813_81387


namespace triangle_side_length_l813_81332

theorem triangle_side_length (a b c : ℝ) (area : ℝ) : 
  a = 1 → b = Real.sqrt 7 → area = Real.sqrt 3 / 2 → 
  c = 2 ∨ c = 2 * Real.sqrt 3 := by sorry

end triangle_side_length_l813_81332


namespace shopping_trip_expenditure_l813_81300

theorem shopping_trip_expenditure (total : ℝ) (other_percent : ℝ)
  (h1 : total > 0)
  (h2 : 0 ≤ other_percent ∧ other_percent ≤ 100)
  (h3 : 50 + 10 + other_percent = 100)
  (h4 : 0.04 * 50 + 0.08 * other_percent = 5.2) :
  other_percent = 40 := by sorry

end shopping_trip_expenditure_l813_81300


namespace divisibility_of_expression_l813_81302

theorem divisibility_of_expression (a b : ℤ) : 
  ∃ k : ℤ, (2*a + 3)^2 - (2*b + 1)^2 = 8 * k := by
sorry

end divisibility_of_expression_l813_81302


namespace quadratic_equation_properties_l813_81363

theorem quadratic_equation_properties (m : ℝ) :
  (∀ x, x^2 - (2*m - 3)*x + m^2 + 1 = 0 → x = m) →
    m = -1/3 ∧
  m < 0 →
    (2*m - 3)^2 - 4*(m^2 + 1) > 0 :=
by sorry

end quadratic_equation_properties_l813_81363


namespace intersection_of_A_and_B_l813_81319

open Set

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≤ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_B_l813_81319


namespace largest_sum_and_simplification_l813_81349

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/2, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/9, 1/3 + 1/10]
  (∀ x ∈ sums, x ≤ 1/3 + 1/2) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end largest_sum_and_simplification_l813_81349


namespace three_zeros_condition_l813_81335

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - a * x
  else -x^2 - (a + 2) * x + 1

/-- The number of zeros of f(x) -/
def number_of_zeros (a : ℝ) : ℕ := sorry

/-- Theorem stating the condition for f(x) to have exactly 3 zeros -/
theorem three_zeros_condition (a : ℝ) :
  number_of_zeros a = 3 ↔ a > Real.exp 1 :=
sorry

end three_zeros_condition_l813_81335


namespace remaining_trip_time_l813_81379

/-- Proves that the time to complete the second half of a 510 km journey at 85 km/h is 3 hours -/
theorem remaining_trip_time (total_distance : ℝ) (speed : ℝ) (h1 : total_distance = 510) (h2 : speed = 85) :
  (total_distance / 2) / speed = 3 := by
  sorry

end remaining_trip_time_l813_81379


namespace rectangular_solid_surface_area_l813_81342

/-- A rectangular solid with prime edge lengths and volume 273 has surface area 302 -/
theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → a * b * c = 273 → 2 * (a * b + b * c + c * a) = 302 := by
  sorry

end rectangular_solid_surface_area_l813_81342


namespace minimum_shots_for_high_probability_l813_81380

theorem minimum_shots_for_high_probability (p : ℝ) (n : ℕ) : 
  p = 1/2 → 
  (1 - (1 - p)^n > 0.9 ↔ n ≥ 4) :=
by sorry

end minimum_shots_for_high_probability_l813_81380


namespace instantaneous_velocity_at_3_seconds_l813_81361

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end instantaneous_velocity_at_3_seconds_l813_81361


namespace triangle_perimeter_l813_81356

theorem triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  b / a = 4 / 3 →          -- ratio of second to first side is 4:3
  c / a = 5 / 3 →          -- ratio of third to first side is 5:3
  c - a = 6 →              -- difference between longest and shortest side is 6
  a + b + c = 36 :=        -- perimeter is 36
by sorry

end triangle_perimeter_l813_81356


namespace barbara_age_when_mike_24_l813_81306

/-- Given that Mike is 16 years old and Barbara is half his age, 
    prove that Barbara will be 16 years old when Mike is 24. -/
theorem barbara_age_when_mike_24 (mike_current_age barbara_current_age mike_future_age : ℕ) : 
  mike_current_age = 16 →
  barbara_current_age = mike_current_age / 2 →
  mike_future_age = 24 →
  barbara_current_age + (mike_future_age - mike_current_age) = 16 :=
by sorry

end barbara_age_when_mike_24_l813_81306


namespace greatest_common_divisor_under_100_l813_81382

theorem greatest_common_divisor_under_100 : ∃ (n : ℕ), n = 90 ∧ 
  n ∣ 540 ∧ n < 100 ∧ n ∣ 180 ∧ 
  ∀ (m : ℕ), m ∣ 540 ∧ m < 100 ∧ m ∣ 180 → m ≤ n :=
by sorry

end greatest_common_divisor_under_100_l813_81382


namespace rihanna_remaining_money_l813_81323

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount mango_price juice_price mango_count juice_count : ℕ) : ℕ :=
  initial_amount - (mango_price * mango_count + juice_price * juice_count)

/-- Theorem: Rihanna's remaining money after shopping --/
theorem rihanna_remaining_money :
  remaining_money 50 3 3 6 6 = 14 := by
  sorry

#eval remaining_money 50 3 3 6 6

end rihanna_remaining_money_l813_81323


namespace cube_inequality_l813_81366

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_inequality_l813_81366


namespace inequality_solution_l813_81347

/-- Given constants p, q, and r satisfying the conditions, prove that p + 2q + 3r = 32 -/
theorem inequality_solution (p q r : ℝ) (h1 : p < q)
  (h2 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x > 5 ∨ (3 ≤ x ∧ x ≤ 7)) :
  p + 2*q + 3*r = 32 := by
  sorry

end inequality_solution_l813_81347


namespace triangle_point_trajectory_l813_81369

theorem triangle_point_trajectory (A B C D : ℝ × ℝ) : 
  B = (-2, 0) →
  C = (2, 0) →
  D = (0, 0) →
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = 3^2 →
  A.2 ≠ 0 →
  A.1^2 + A.2^2 = 9 :=
by sorry

end triangle_point_trajectory_l813_81369


namespace teacups_left_result_l813_81358

/-- Calculates the number of teacups left after arranging --/
def teacups_left (total_boxes : ℕ) (pan_boxes : ℕ) (rows_per_box : ℕ) (cups_per_row : ℕ) (broken_per_box : ℕ) : ℕ :=
  let remaining_boxes := total_boxes - pan_boxes
  let decoration_boxes := remaining_boxes / 2
  let teacup_boxes := remaining_boxes - decoration_boxes
  let cups_per_box := rows_per_box * cups_per_row
  let total_cups := teacup_boxes * cups_per_box
  let broken_cups := teacup_boxes * broken_per_box
  total_cups - broken_cups

/-- Theorem stating the number of teacups left after arranging --/
theorem teacups_left_result : teacups_left 26 6 5 4 2 = 180 := by
  sorry

end teacups_left_result_l813_81358


namespace angles_with_same_terminal_side_eq_l813_81367

/-- Given an angle α whose terminal side is the same as 8π/5, 
    this function returns the set of angles in [0, 2π] 
    whose terminal sides are the same as α/4 -/
def anglesWithSameTerminalSide (α : ℝ) : Set ℝ :=
  {x | x ∈ Set.Icc 0 (2 * Real.pi) ∧ 
       ∃ k : ℤ, α = 2 * k * Real.pi + 8 * Real.pi / 5 ∧ 
               x = (k * Real.pi / 2 + 2 * Real.pi / 5) % (2 * Real.pi)}

/-- Theorem stating that the set of angles with the same terminal side as α/4 
    is equal to the specific set of four angles -/
theorem angles_with_same_terminal_side_eq (α : ℝ) 
    (h : ∃ k : ℤ, α = 2 * k * Real.pi + 8 * Real.pi / 5) : 
  anglesWithSameTerminalSide α = {2 * Real.pi / 5, 9 * Real.pi / 10, 7 * Real.pi / 5, 19 * Real.pi / 10} := by
  sorry

end angles_with_same_terminal_side_eq_l813_81367


namespace point_distance_on_line_l813_81388

/-- Given two points on a line, prove that the horizontal distance between them is 3 -/
theorem point_distance_on_line (m n : ℝ) : 
  (m = n / 5 - 2 / 5) → 
  (m + 3 = (n + 15) / 5 - 2 / 5) := by
sorry

end point_distance_on_line_l813_81388


namespace expression_equality_l813_81307

/-- The base-10 logarithm -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Prove that 27^(1/3) + lg 4 + 2 * lg 5 - e^(ln 3) = 2 -/
theorem expression_equality : 27^(1/3) + lg 4 + 2 * lg 5 - Real.exp (Real.log 3) = 2 := by
  sorry

end expression_equality_l813_81307


namespace female_students_count_l813_81360

/-- Represents a school with male and female students. -/
structure School where
  total_students : ℕ
  sample_size : ℕ
  sample_boys_girls_diff : ℕ

/-- Calculates the number of female students in the school based on the given parameters. -/
def female_students (s : School) : ℕ :=
  let sampled_girls := (s.sample_size - s.sample_boys_girls_diff) / 2
  let ratio := s.total_students / s.sample_size
  sampled_girls * ratio

/-- Theorem stating that for the given school parameters, the number of female students is 760. -/
theorem female_students_count (s : School) 
  (h1 : s.total_students = 1600)
  (h2 : s.sample_size = 200)
  (h3 : s.sample_boys_girls_diff = 10) :
  female_students s = 760 := by
  sorry

end female_students_count_l813_81360


namespace charles_pictures_before_work_l813_81301

/-- The number of pictures Charles drew before going to work yesterday -/
def pictures_before_work : ℕ → ℕ → ℕ → ℕ → ℕ
  | total_papers, papers_left, pictures_today, pictures_after_work =>
    total_papers - papers_left - pictures_today - pictures_after_work

theorem charles_pictures_before_work :
  pictures_before_work 20 2 6 6 = 6 := by
  sorry

end charles_pictures_before_work_l813_81301


namespace divisible_by_six_l813_81376

theorem divisible_by_six (n : ℕ) : ∃ k : ℤ, (2 * n^3 + 9 * n^2 + 13 * n : ℤ) = 6 * k := by
  sorry

end divisible_by_six_l813_81376


namespace section_4_eight_times_section_1_l813_81354

/-- Represents a circular target divided into sections -/
structure CircularTarget where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  α : ℝ
  β : ℝ
  (r₁_pos : 0 < r₁)
  (r₂_pos : 0 < r₂)
  (r₃_pos : 0 < r₃)
  (r₁_lt_r₂ : r₁ < r₂)
  (r₂_lt_r₃ : r₂ < r₃)
  (α_pos : 0 < α)
  (β_pos : 0 < β)
  (section_equality : r₁^2 * β = α * (r₂^2 - r₁^2))
  (section_2_half_3 : β * (r₂^2 - r₁^2) = 2 * r₁^2 * β)

/-- The theorem stating that the area of section 4 is 8 times the area of section 1 -/
theorem section_4_eight_times_section_1 (t : CircularTarget) : 
  (t.β * (t.r₃^2 - t.r₂^2)) / (t.α * t.r₁^2) = 8 := by
  sorry

end section_4_eight_times_section_1_l813_81354


namespace atop_difference_l813_81325

-- Define the @ operation
def atop (x y : ℤ) : ℤ := x * y - 3 * x

-- State the theorem
theorem atop_difference : atop 8 5 - atop 5 8 = -9 := by
  sorry

end atop_difference_l813_81325


namespace quadratic_function_coefficients_l813_81368

/-- Given a quadratic function f(x) = ax^2 + bx + 7, 
    if f(x+1) - f(x) = 8x - 2 for all x, then a = 4 and b = -6 -/
theorem quadratic_function_coefficients 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + 7)
  (h2 : ∀ x, f (x + 1) - f x = 8 * x - 2) : 
  a = 4 ∧ b = -6 := by
sorry

end quadratic_function_coefficients_l813_81368


namespace range_of_x_l813_81350

theorem range_of_x (x : ℝ) : 
  (∀ m : ℝ, m ≠ 0 → |5*m - 3| + |3 - 4*m| ≥ |m| * (x - 2/x)) →
  x ∈ Set.Ici (-1) ∪ Set.Ioc 0 2 :=
sorry

end range_of_x_l813_81350


namespace fencing_required_l813_81397

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area = uncovered_side * (area / uncovered_side) →
  2 * (area / uncovered_side) + uncovered_side = 32 := by
  sorry

#check fencing_required 120 20

end fencing_required_l813_81397


namespace intersected_prisms_count_l813_81324

def small_prism_dimensions : Fin 3 → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 5

def cube_edge_length : ℕ := 90

def count_intersected_prisms (dimensions : Fin 3 → ℕ) (edge_length : ℕ) : ℕ :=
  sorry

theorem intersected_prisms_count :
  count_intersected_prisms small_prism_dimensions cube_edge_length = 66 := by sorry

end intersected_prisms_count_l813_81324


namespace integer_sum_l813_81305

theorem integer_sum (x y : ℕ+) : x - y = 14 → x * y = 48 → x + y = 20 := by
  sorry

end integer_sum_l813_81305


namespace remainder_theorem_l813_81327

-- Define the polynomial
def f (x : ℝ) : ℝ := x^5 - 3*x^3 + x + 5

-- Define the divisor
def g (x : ℝ) : ℝ := (x - 3)^2

-- Theorem statement
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, f x = g x * q x + 65 :=
sorry

end remainder_theorem_l813_81327


namespace adlai_total_animal_legs_l813_81378

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of dogs Adlai has -/
def adlai_dogs : ℕ := 2

/-- The number of chickens Adlai has -/
def adlai_chickens : ℕ := 1

/-- Theorem stating the total number of animal legs Adlai has -/
theorem adlai_total_animal_legs : 
  adlai_dogs * dog_legs + adlai_chickens * chicken_legs = 10 := by
  sorry

end adlai_total_animal_legs_l813_81378


namespace stratified_sampling_result_l813_81316

/-- Proves that the total number of students sampled is 135 given the conditions of the stratified sampling problem -/
theorem stratified_sampling_result (grade10 : ℕ) (grade11 : ℕ) (grade12 : ℕ) (sampled10 : ℕ) 
  (h1 : grade10 = 2000)
  (h2 : grade11 = 1500)
  (h3 : grade12 = 1000)
  (h4 : sampled10 = 60) :
  (grade10 + grade11 + grade12) * sampled10 / grade10 = 135 := by
  sorry

#check stratified_sampling_result

end stratified_sampling_result_l813_81316


namespace factor_expression_l813_81333

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := by
  sorry

end factor_expression_l813_81333


namespace lucy_groceries_weight_l813_81374

/-- The total weight of groceries Lucy bought -/
def total_weight (cookies_packs noodles_packs cookie_weight noodle_weight : ℕ) : ℕ :=
  cookies_packs * cookie_weight + noodles_packs * noodle_weight

/-- Theorem stating that the total weight of Lucy's groceries is 11000g -/
theorem lucy_groceries_weight :
  total_weight 12 16 250 500 = 11000 := by
  sorry

end lucy_groceries_weight_l813_81374


namespace larger_number_proof_l813_81399

theorem larger_number_proof (x y : ℝ) 
  (h1 : x - y = 1860)
  (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 := by
  sorry

end larger_number_proof_l813_81399


namespace maryville_population_increase_l813_81330

/-- The average annual population increase in Maryville between 2000 and 2005 -/
def average_annual_increase (pop_2000 pop_2005 : ℕ) : ℚ :=
  (pop_2005 - pop_2000 : ℚ) / 5

/-- Theorem stating the average annual population increase in Maryville between 2000 and 2005 -/
theorem maryville_population_increase :
  average_annual_increase 450000 467000 = 3400 := by
  sorry

end maryville_population_increase_l813_81330


namespace sum_remainder_mod_nine_l813_81353

theorem sum_remainder_mod_nine : 
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 := by
  sorry

end sum_remainder_mod_nine_l813_81353
