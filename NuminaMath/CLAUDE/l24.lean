import Mathlib

namespace walking_problem_solution_l24_2421

def walking_problem (total_distance : ℝ) (speed_R : ℝ) (speed_S_initial : ℝ) (speed_S_second : ℝ) : Prop :=
  ∃ (k : ℕ) (x : ℝ),
    -- The total distance is 76 miles
    total_distance = 76 ∧
    -- Speed of person at R is 4.5 mph
    speed_R = 4.5 ∧
    -- Initial speed of person at S is 3.25 mph
    speed_S_initial = 3.25 ∧
    -- Second hour speed of person at S is 3.75 mph
    speed_S_second = 3.75 ∧
    -- They meet after k hours (k is a natural number)
    k > 0 ∧
    -- Distance traveled by person from R
    speed_R * k + x = total_distance / 2 ∧
    -- Distance traveled by person from S (arithmetic sequence sum)
    k * (speed_S_initial + (speed_S_second - speed_S_initial) * (k - 1) / 2) - x = total_distance / 2 ∧
    -- x is the difference in distances, and it equals 4
    x = 4

theorem walking_problem_solution :
  walking_problem 76 4.5 3.25 3.75 :=
sorry

end walking_problem_solution_l24_2421


namespace heathers_weight_l24_2460

/-- Given that Emily weighs 9 pounds and Heather is 78 pounds heavier than Emily,
    prove that Heather weighs 87 pounds. -/
theorem heathers_weight (emily_weight : ℕ) (weight_difference : ℕ) :
  emily_weight = 9 →
  weight_difference = 78 →
  emily_weight + weight_difference = 87 :=
by sorry

end heathers_weight_l24_2460


namespace correlation_coefficient_measures_linear_relationship_l24_2492

/-- The correlation coefficient is a statistical measure. -/
def correlation_coefficient : Type := sorry

/-- A measure of the strength of a linear relationship between two variables. -/
def linear_relationship_strength : Type := sorry

/-- The correlation coefficient measures the strength of the linear relationship between two variables. -/
theorem correlation_coefficient_measures_linear_relationship :
  correlation_coefficient → linear_relationship_strength :=
sorry

end correlation_coefficient_measures_linear_relationship_l24_2492


namespace infinite_geometric_series_ratio_l24_2491

/-- For an infinite geometric series with first term a and sum S,
    the common ratio r can be calculated. -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 512) (h2 : S = 3072) :
  ∃ r : ℝ, r = 5 / 6 ∧ S = a / (1 - r) := by
sorry

end infinite_geometric_series_ratio_l24_2491


namespace factorial_combination_l24_2467

theorem factorial_combination : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 3) = 120 := by
  sorry

end factorial_combination_l24_2467


namespace sum_of_coefficients_l24_2405

/-- The polynomial x^3 - 8x^2 + 17x - 14 -/
def polynomial (x : ℝ) : ℝ := x^3 - 8*x^2 + 17*x - 14

/-- The sum of the kth powers of the roots -/
def s (k : ℕ) : ℝ := sorry

/-- The relation between consecutive s_k values -/
def relation (a b c : ℝ) : Prop :=
  ∀ k : ℕ, k ≥ 1 → s (k+1) = a * s k + b * s (k-1) + c * s (k-2)

theorem sum_of_coefficients :
  ∃ (a b c : ℝ),
    s 0 = 3 ∧ s 1 = 8 ∧ s 2 = 17 ∧
    relation a b c ∧
    a + b + c = 9 :=
sorry

end sum_of_coefficients_l24_2405


namespace function_is_even_l24_2429

/-- A function satisfying certain properties is even -/
theorem function_is_even (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 2) = f (2 - x))
  (h2 : ∀ x, f (1 + x) = -f x)
  (h3 : ¬ ∀ x y, f x = f y) : 
  ∀ x, f x = f (-x) := by
  sorry

end function_is_even_l24_2429


namespace m_range_l24_2494

/-- The statement "The equation x^2 + 2x + m = 0 has no real roots" -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

/-- The statement "The equation x^2/(m-1) + y^2 = 1 is an ellipse with foci on the x-axis" -/
def q (m : ℝ) : Prop := m > 2 ∧ ∀ x y : ℝ, x^2/(m-1) + y^2 = 1 → ∃ c : ℝ, c^2 = m - 1

theorem m_range (m : ℝ) : (¬(¬(p m)) ∧ ¬(p m ∧ q m)) → (1 < m ∧ m ≤ 2) :=
sorry

end m_range_l24_2494


namespace igor_travel_time_l24_2436

/-- Represents the ski lift system with its properties and functions -/
structure SkiLift where
  total_cabins : Nat
  igor_cabin : Nat
  first_alignment : Nat
  second_alignment : Nat
  alignment_time : Nat

/-- Calculates the time for Igor to reach the top of the mountain -/
def time_to_top (lift : SkiLift) : Nat :=
  let total_distance := lift.total_cabins - lift.igor_cabin + lift.second_alignment
  let speed := (lift.first_alignment - lift.second_alignment) / lift.alignment_time
  (total_distance / 2) * (1 / speed)

/-- Theorem stating that Igor will reach the top in 1035 seconds -/
theorem igor_travel_time (lift : SkiLift) 
  (h1 : lift.total_cabins = 99)
  (h2 : lift.igor_cabin = 42)
  (h3 : lift.first_alignment = 13)
  (h4 : lift.second_alignment = 12)
  (h5 : lift.alignment_time = 15) :
  time_to_top lift = 1035 := by
  sorry

end igor_travel_time_l24_2436


namespace most_advantageous_order_l24_2415

-- Define the probabilities
variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
variable (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1)
variable (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1)
variable (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
variable (h₄ : p₃ < p₁)
variable (h₅ : p₁ < p₂)

-- Define the probability of winning two games in a row with p₂ as the second opponent
def prob_p₂_second := p₂ * (p₁ + p₃ - p₁ * p₃)

-- Define the probability of winning two games in a row with p₁ as the second opponent
def prob_p₁_second := p₁ * (p₂ + p₃ - p₂ * p₃)

-- The theorem to prove
theorem most_advantageous_order :
  prob_p₂_second p₁ p₂ p₃ > prob_p₁_second p₁ p₂ p₃ :=
sorry

end most_advantageous_order_l24_2415


namespace quadratic_function_properties_l24_2426

/-- A quadratic function of the form y = x^2 + mx + m^2 - 3 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + m*x + m^2 - 3

theorem quadratic_function_properties :
  ∀ m : ℝ, m > 0 →
  quadratic_function m 2 = 4 →
  (m = 1 ∧ ∃ x y : ℝ, x ≠ y ∧ quadratic_function m x = 0 ∧ quadratic_function m y = 0) :=
by sorry

end quadratic_function_properties_l24_2426


namespace binomial_expansion_example_l24_2402

theorem binomial_expansion_example : 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 104060401 := by
  sorry

end binomial_expansion_example_l24_2402


namespace inequality_multiplication_l24_2428

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b := by
  sorry

end inequality_multiplication_l24_2428


namespace problem_statement_l24_2431

theorem problem_statement (x y : ℝ) (h : -x + 2*y = 5) : 
  5*(x - 2*y)^2 - 3*(x - 2*y) - 60 = 80 := by
sorry

end problem_statement_l24_2431


namespace initial_passengers_l24_2471

theorem initial_passengers (P : ℕ) : 
  P % 2 = 0 ∧ 
  (P : ℝ) + 0.08 * (P : ℝ) ≤ 70 ∧ 
  P % 25 = 0 → 
  P = 50 := by
sorry

end initial_passengers_l24_2471


namespace gala_handshakes_l24_2447

/-- Number of married couples at the gala -/
def num_couples : ℕ := 15

/-- Total number of people at the gala -/
def total_people : ℕ := 2 * num_couples

/-- Number of handshakes between men -/
def handshakes_men : ℕ := num_couples.choose 2

/-- Number of handshakes between men and women -/
def handshakes_men_women : ℕ := num_couples * num_couples

/-- Total number of handshakes at the gala -/
def total_handshakes : ℕ := handshakes_men + handshakes_men_women

theorem gala_handshakes : total_handshakes = 330 := by
  sorry

end gala_handshakes_l24_2447


namespace gas_supply_equilibrium_l24_2496

/-- The distance between points A and B in kilometers -/
def total_distance : ℝ := 500

/-- The amount of gas extracted from reservoir A in cubic meters per minute -/
def gas_from_A : ℝ := 10000

/-- The rate of gas leakage in cubic meters per kilometer -/
def leakage_rate : ℝ := 4

/-- The distance between point A and city C in kilometers -/
def distance_AC : ℝ := 100

theorem gas_supply_equilibrium :
  let gas_to_C_from_A := gas_from_A - leakage_rate * distance_AC
  let gas_to_C_from_B := (gas_from_A * 1.12) - leakage_rate * (total_distance - distance_AC)
  gas_to_C_from_A = gas_to_C_from_B :=
by sorry

end gas_supply_equilibrium_l24_2496


namespace george_marbles_count_l24_2466

/-- The total number of marbles George collected -/
def total_marbles : ℕ := 50

/-- The number of yellow marbles -/
def yellow_marbles : ℕ := 12

/-- The number of red marbles -/
def red_marbles : ℕ := 7

/-- The number of green marbles -/
def green_marbles : ℕ := yellow_marbles / 2

/-- The number of white marbles -/
def white_marbles : ℕ := total_marbles / 2

theorem george_marbles_count :
  total_marbles = white_marbles + yellow_marbles + green_marbles + red_marbles :=
by sorry

end george_marbles_count_l24_2466


namespace line_through_points_l24_2463

/-- Given a line with equation x = 3y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 2/3 -/
theorem line_through_points (m n p : ℝ) : 
  (m = 3 * n + 5) ∧ (m + 2 = 3 * (n + p) + 5) → p = 2/3 := by
  sorry

end line_through_points_l24_2463


namespace student_distribution_theorem_l24_2449

/-- The number of ways to distribute n students among k groups, with each student choosing exactly one group -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose m items from a set of n items -/
def choose (n : ℕ) (m : ℕ) : ℕ := sorry

theorem student_distribution_theorem :
  let total_students : ℕ := 4
  let total_groups : ℕ := 4
  let groups_to_fill : ℕ := 3
  distribute_students total_students groups_to_fill = 36 :=
by sorry

end student_distribution_theorem_l24_2449


namespace cylinder_side_diagonal_l24_2443

theorem cylinder_side_diagonal (h l d : ℝ) (h_height : h = 16) (h_length : l = 12) : 
  d = 20 → d^2 = h^2 + l^2 := by
  sorry

end cylinder_side_diagonal_l24_2443


namespace calculator_exam_duration_l24_2489

theorem calculator_exam_duration 
  (full_battery : ℝ) 
  (remaining_battery : ℝ) 
  (exam_duration : ℝ) :
  full_battery = 60 →
  remaining_battery = 13 →
  exam_duration = (1/4 * full_battery) - remaining_battery →
  exam_duration = 2 :=
by sorry

end calculator_exam_duration_l24_2489


namespace segment_construction_l24_2497

/-- Given positive real numbers a, b, c, d, and e, there exists a real number x
    such that x = (a * b * c) / (d * e). -/
theorem segment_construction (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hd : d > 0) (he : e > 0) : ∃ x : ℝ, x = (a * b * c) / (d * e) := by
  sorry

end segment_construction_l24_2497


namespace square_difference_of_sum_and_difference_l24_2476

theorem square_difference_of_sum_and_difference (x y : ℝ) 
  (h_sum : x + y = 20) (h_diff : x - y = 10) : x^2 - y^2 = 200 := by
  sorry

end square_difference_of_sum_and_difference_l24_2476


namespace jane_payment_l24_2451

/-- The amount Jane paid with, given the cost of the apple and the change received. -/
def amount_paid (apple_cost change : ℚ) : ℚ :=
  apple_cost + change

/-- Theorem stating that Jane paid with $5.00, given the conditions of the problem. -/
theorem jane_payment :
  let apple_cost : ℚ := 75 / 100
  let change : ℚ := 425 / 100
  amount_paid apple_cost change = 5 := by
  sorry

end jane_payment_l24_2451


namespace a_range_l24_2464

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - (1/2) * x^2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 4*x

theorem a_range (a : ℝ) :
  (∃ x_0 : ℝ, x_0 > 0 ∧ IsLocalMin (g a) x_0) →
  (∃ x_0 : ℝ, x_0 > 0 ∧ IsLocalMin (g a) x_0 ∧ g a x_0 - (1/2) * x_0^2 + 2*a > 0) →
  a ∈ Set.Ioo (-4/ℯ + 1/ℯ^2) 0 :=
by sorry

end a_range_l24_2464


namespace repeating_decimal_sum_l24_2450

-- Define repeating decimals
def repeating_234 : ℚ := 234 / 999
def repeating_567 : ℚ := 567 / 999
def repeating_891 : ℚ := 891 / 999

-- State the theorem
theorem repeating_decimal_sum : 
  repeating_234 - repeating_567 + repeating_891 = 186 / 333 := by
  sorry

end repeating_decimal_sum_l24_2450


namespace nancy_savings_l24_2495

-- Define the value of a dozen
def dozen : ℕ := 12

-- Define the value of a quarter in cents
def quarter_value : ℕ := 25

-- Define the number of cents in a dollar
def cents_per_dollar : ℕ := 100

-- Theorem statement
theorem nancy_savings (quarters : ℕ) : 
  quarters = dozen → (quarters * quarter_value) / cents_per_dollar = 3 := by
  sorry

end nancy_savings_l24_2495


namespace square_root_equation_l24_2477

theorem square_root_equation (x : ℝ) : (x + 1)^2 = 9 → x = 2 ∨ x = -4 := by
  sorry

end square_root_equation_l24_2477


namespace rectangular_solid_surface_area_l24_2498

-- Define a structure for a rectangular solid
structure RectangularSolid where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the properties of the rectangular solid
def isPrime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem rectangular_solid_surface_area 
  (solid : RectangularSolid) 
  (prime_edges : isPrime solid.length ∧ isPrime solid.width ∧ isPrime solid.height) 
  (volume_constraint : solid.length * solid.width * solid.height = 105) :
  2 * (solid.length * solid.width + solid.width * solid.height + solid.height * solid.length) = 142 := by
  sorry


end rectangular_solid_surface_area_l24_2498


namespace toys_per_day_l24_2439

/-- A factory produces toys according to the following conditions:
  1. The factory produces 4560 toys per week.
  2. The workers work 4 days a week.
  3. The same number of toys is made every day.
-/
def factory_production (toys_per_day : ℕ) : Prop :=
  toys_per_day * 4 = 4560 ∧ toys_per_day > 0

/-- The number of toys produced each day is 1140. -/
theorem toys_per_day : ∃ (n : ℕ), factory_production n ∧ n = 1140 :=
  sorry

end toys_per_day_l24_2439


namespace p_oplus_q_equals_result_l24_2413

def P : Set ℤ := {4, 5}
def Q : Set ℤ := {1, 2, 3}

def setDifference (P Q : Set ℤ) : Set ℤ :=
  {x | ∃ p ∈ P, ∃ q ∈ Q, x = p - q}

theorem p_oplus_q_equals_result : setDifference P Q = {1, 2, 3, 4} := by
  sorry

end p_oplus_q_equals_result_l24_2413


namespace container_capacity_prove_container_capacity_l24_2438

theorem container_capacity : ℝ → Prop :=
  fun capacity =>
    (0.5 * capacity + 20 = 0.75 * capacity) →
    capacity = 80

-- The proof of the theorem
theorem prove_container_capacity : container_capacity 80 := by
  sorry

end container_capacity_prove_container_capacity_l24_2438


namespace power_of_ten_square_l24_2470

theorem power_of_ten_square (k : ℕ) (N : ℕ) : 
  (10^(k-1) ≤ N) ∧ (N < 10^k) ∧ 
  (∃ m : ℕ, N^2 = N * 10^k + m ∧ m < N * 10^k) → 
  N = 10^(k-1) :=
by sorry

end power_of_ten_square_l24_2470


namespace product_of_difference_and_sum_of_squares_l24_2424

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 39) : 
  a * b = 15 := by
  sorry

end product_of_difference_and_sum_of_squares_l24_2424


namespace difference_of_squares_l24_2408

theorem difference_of_squares (x y : ℝ) : (y + x) * (y - x) = y^2 - x^2 := by
  sorry

end difference_of_squares_l24_2408


namespace first_number_proof_l24_2473

theorem first_number_proof : ∃ x : ℝ, x + 2.017 + 0.217 + 2.0017 = 221.2357 ∧ x = 217 := by
  sorry

end first_number_proof_l24_2473


namespace printer_problem_l24_2453

/-- Given a total of 42 pages, where every 7th page is crumpled and every 3rd page is blurred,
    the number of pages that are neither crumpled nor blurred is 24. -/
theorem printer_problem (total_pages : Nat) (crumple_interval : Nat) (blur_interval : Nat)
    (h1 : total_pages = 42)
    (h2 : crumple_interval = 7)
    (h3 : blur_interval = 3) :
    total_pages - (total_pages / crumple_interval + total_pages / blur_interval - total_pages / (crumple_interval * blur_interval)) = 24 :=
by sorry

end printer_problem_l24_2453


namespace inequality_theorem_l24_2455

theorem inequality_theorem (a b c : ℝ) (θ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a * Real.cos θ ^ 2 + b * Real.sin θ ^ 2 < c) : 
  Real.sqrt a * Real.cos θ ^ 2 + Real.sqrt b * Real.sin θ ^ 2 < Real.sqrt c := by
  sorry

end inequality_theorem_l24_2455


namespace min_value_problem_max_value_problem_min_sum_problem_l24_2457

-- Problem 1
theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y ≥ 8 := by sorry

-- Problem 2
theorem max_value_problem (x : ℝ) (h : x < 3) :
  4/(x - 3) + x ≤ -1 := by sorry

-- Problem 3
theorem min_sum_problem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 2) :
  n/m + 1/(2*n) ≥ 5/4 := by sorry

end min_value_problem_max_value_problem_min_sum_problem_l24_2457


namespace complex_purely_imaginary_l24_2420

theorem complex_purely_imaginary (a : ℝ) : 
  (Complex.I * (2 * a + 1) : ℂ) = (2 + Complex.I) * (1 + a * Complex.I) → a = 2 := by
  sorry

end complex_purely_imaginary_l24_2420


namespace power_function_through_point_l24_2404

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = x^a) →  -- f is a power function with exponent a
  f 2 = 16 →              -- f passes through the point (2, 16)
  a = 4 := by             -- prove that a = 4
sorry

end power_function_through_point_l24_2404


namespace library_visitors_l24_2474

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) 
  (h1 : sunday_avg = 510)
  (h2 : total_days = 30)
  (h3 : month_avg = 285) :
  let sundays : ℕ := total_days / 7 + 1
  let other_days : ℕ := total_days - sundays
  let other_days_avg : ℕ := (month_avg * total_days - sunday_avg * sundays) / other_days
  other_days_avg = 240 := by
  sorry

end library_visitors_l24_2474


namespace least_of_four_consecutive_integers_with_sum_two_l24_2483

theorem least_of_four_consecutive_integers_with_sum_two :
  ∀ n : ℤ, (n + (n + 1) + (n + 2) + (n + 3) = 2) → n = -1 :=
by
  sorry

end least_of_four_consecutive_integers_with_sum_two_l24_2483


namespace power_sum_ratio_l24_2441

theorem power_sum_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_zero : a + b + c = 0) :
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49/60 := by
  sorry

end power_sum_ratio_l24_2441


namespace no_real_solutions_to_equation_l24_2456

theorem no_real_solutions_to_equation : 
  ¬ ∃ (x : ℝ), x > 0 ∧ x^(Real.log x / Real.log 10) = x^3 / 1000 :=
sorry

end no_real_solutions_to_equation_l24_2456


namespace overlap_difference_l24_2418

/-- Represents the student population --/
def StudentPopulation : Set ℕ := {n : ℕ | 1000 ≤ n ∧ n ≤ 1200}

/-- Represents the number of students studying German --/
def GermanStudents (n : ℕ) : Set ℕ := {g : ℕ | (70 * n + 99) / 100 ≤ g ∧ g ≤ (75 * n) / 100}

/-- Represents the number of students studying Russian --/
def RussianStudents (n : ℕ) : Set ℕ := {r : ℕ | (35 * n + 99) / 100 ≤ r ∧ r ≤ (45 * n) / 100}

/-- The minimum number of students studying both languages --/
def m (n : ℕ) (g : ℕ) (r : ℕ) : ℕ := g + r - n

/-- The maximum number of students studying both languages --/
def M (n : ℕ) (g : ℕ) (r : ℕ) : ℕ := min g r

/-- Main theorem --/
theorem overlap_difference (n : StudentPopulation) 
  (g : GermanStudents n) (r : RussianStudents n) : 
  ∃ (m_val : ℕ) (M_val : ℕ), 
    m_val = m n g r ∧ 
    M_val = M n g r ∧ 
    M_val - m_val = 190 := by
  sorry

end overlap_difference_l24_2418


namespace ski_price_after_discounts_l24_2442

def original_price : ℝ := 200
def discount1 : ℝ := 0.40
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.10

theorem ski_price_after_discounts :
  let price1 := original_price * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let final_price := price2 * (1 - discount3)
  final_price = 86.40 := by sorry

end ski_price_after_discounts_l24_2442


namespace geometric_sequence_property_l24_2401

/-- A geometric sequence of positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_positive : ∀ n, a n > 0)
  (h_sum : a 2 * a 8 + a 3 * a 7 = 32) : 
  a 5 = 4 := by
sorry

end geometric_sequence_property_l24_2401


namespace orchard_sections_l24_2493

/-- Given the daily harvest from each orchard section and the total daily harvest,
    calculate the number of orchard sections. -/
theorem orchard_sections 
  (sacks_per_section : ℕ) 
  (total_sacks : ℕ) 
  (h1 : sacks_per_section = 45)
  (h2 : total_sacks = 360) :
  total_sacks / sacks_per_section = 8 := by
  sorry

end orchard_sections_l24_2493


namespace smallest_difference_in_triangle_l24_2410

theorem smallest_difference_in_triangle (a b c : ℕ) : 
  a + b + c = 2023 →
  a < b →
  b ≤ c →
  (∀ x y z : ℕ, x + y + z = 2023 → x < y → y ≤ z → b - a ≤ y - x) →
  b - a = 1 := by
sorry

end smallest_difference_in_triangle_l24_2410


namespace ratio_s5_s8_l24_2452

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_ratio : (4 * (2 * a 1 + 3 * (a 2 - a 1))) / (6 * (2 * a 1 + 5 * (a 2 - a 1))) = -2/3

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

/-- The main theorem -/
theorem ratio_s5_s8 (seq : ArithmeticSequence) : 
  (sum_n seq 5) / (sum_n seq 8) = 1 / 40.8 := by
  sorry

end ratio_s5_s8_l24_2452


namespace cat_dog_food_difference_l24_2419

theorem cat_dog_food_difference :
  let cat_packages : ℕ := 6
  let dog_packages : ℕ := 2
  let cans_per_cat_package : ℕ := 9
  let cans_per_dog_package : ℕ := 3
  let total_cat_cans := cat_packages * cans_per_cat_package
  let total_dog_cans := dog_packages * cans_per_dog_package
  total_cat_cans - total_dog_cans = 48 :=
by sorry

end cat_dog_food_difference_l24_2419


namespace all_cells_equal_l24_2485

/-- Represents an infinite grid of natural numbers -/
def Grid := ℤ → ℤ → ℕ

/-- The condition that each cell's value is greater than or equal to the arithmetic mean of its four neighboring cells -/
def ValidGrid (g : Grid) : Prop :=
  ∀ i j : ℤ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

/-- The theorem stating that all cells in a valid grid must contain the same number -/
theorem all_cells_equal (g : Grid) (h : ValidGrid g) : 
  ∀ i j k l : ℤ, g i j = g k l :=
sorry

end all_cells_equal_l24_2485


namespace angle_between_lines_l24_2468

theorem angle_between_lines (k₁ k₂ : ℝ) (h₁ : 6 * k₁^2 + k₁ - 1 = 0) (h₂ : 6 * k₂^2 + k₂ - 1 = 0) :
  let θ := Real.arctan ((k₁ - k₂) / (1 + k₁ * k₂))
  θ = π / 4 ∨ θ = -π / 4 :=
sorry

end angle_between_lines_l24_2468


namespace flour_for_one_loaf_l24_2461

/-- The amount of flour required for one loaf of bread -/
def flour_per_loaf (total_flour : ℕ) (num_loaves : ℕ) : ℕ := total_flour / num_loaves

/-- Theorem: Given 400g of total flour and the ability to make 2 loaves, 
    prove that one loaf requires 200g of flour -/
theorem flour_for_one_loaf : 
  flour_per_loaf 400 2 = 200 := by
sorry

end flour_for_one_loaf_l24_2461


namespace geometric_sequence_sum_l24_2487

/-- A geometric sequence with common ratio 2 and sum of first 3 terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The sum of the 3rd, 4th, and 5th terms of the geometric sequence is 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 + a 4 + a 5 = 84 := by
sorry

end geometric_sequence_sum_l24_2487


namespace quadratic_root_implies_k_l24_2469

theorem quadratic_root_implies_k (k : ℝ) : 
  ((k - 3) * (-1)^2 + 6 * (-1) + k^2 - k = 0) → k = -3 := by
  sorry

end quadratic_root_implies_k_l24_2469


namespace daughters_and_granddaughters_without_children_l24_2481

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  daughtersWithChildren : ℕ
  totalDescendants : ℕ

/-- The actual Bertha family configuration -/
def berthaActual : BerthaFamily :=
  { daughters := 8,
    daughtersWithChildren := 7,  -- This is derived, not given directly
    totalDescendants := 36 }

/-- Theorem stating the number of daughters and granddaughters without children -/
theorem daughters_and_granddaughters_without_children
  (b : BerthaFamily)
  (h1 : b.daughters = berthaActual.daughters)
  (h2 : b.totalDescendants = berthaActual.totalDescendants)
  (h3 : ∀ d, d ≤ b.daughters → (d = b.daughtersWithChildren ∨ d = b.daughters - b.daughtersWithChildren))
  (h4 : b.totalDescendants = b.daughters + 4 * b.daughtersWithChildren) :
  b.daughters - b.daughtersWithChildren + (b.totalDescendants - b.daughters) = 29 := by
  sorry

end daughters_and_granddaughters_without_children_l24_2481


namespace arithmetic_sequence_sum_l24_2488

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 7 + a 13 = 20 →
  a 9 + a 10 + a 11 = 30 := by
sorry

end arithmetic_sequence_sum_l24_2488


namespace present_cost_l24_2407

/-- Proves that the total amount paid for a present by 4 friends is $60, given specific conditions. -/
theorem present_cost (initial_contribution : ℝ) : 
  (4 : ℝ) > 0 → 
  0 < initial_contribution → 
  0.75 * (4 * initial_contribution) = 4 * (initial_contribution - 5) → 
  0.75 * (4 * initial_contribution) = 60 := by
  sorry

end present_cost_l24_2407


namespace problem_solving_probability_l24_2430

/-- The probability that Alex, Kyle, and Catherine solve a problem, but not Bella and David -/
theorem problem_solving_probability 
  (p_alex : ℚ) (p_bella : ℚ) (p_kyle : ℚ) (p_david : ℚ) (p_catherine : ℚ)
  (h_alex : p_alex = 1/4)
  (h_bella : p_bella = 3/5)
  (h_kyle : p_kyle = 1/3)
  (h_david : p_david = 2/7)
  (h_catherine : p_catherine = 5/9) :
  p_alex * p_kyle * p_catherine * (1 - p_bella) * (1 - p_david) = 25/378 := by
sorry

end problem_solving_probability_l24_2430


namespace imaginary_part_product_l24_2480

theorem imaginary_part_product : Complex.im ((1 + Complex.I) * (3 - Complex.I)) = 2 := by
  sorry

end imaginary_part_product_l24_2480


namespace pen_cost_is_four_l24_2444

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := 2

/-- The cost of a pen in dollars -/
def pen_cost : ℝ := 2 * pencil_cost

/-- The total cost of a pen and pencil in dollars -/
def total_cost : ℝ := 6

theorem pen_cost_is_four :
  pen_cost = 4 ∧ pencil_cost + pen_cost = total_cost :=
by sorry

end pen_cost_is_four_l24_2444


namespace greatest_integer_solution_greatest_integer_value_l24_2409

theorem greatest_integer_solution (x : ℤ) : (5 - 4*x > 17) ↔ (x < -3) :=
  sorry

theorem greatest_integer_value : ∃ (x : ℤ), (∀ (y : ℤ), (5 - 4*y > 17) → y ≤ x) ∧ (5 - 4*x > 17) ∧ x = -4 :=
  sorry

end greatest_integer_solution_greatest_integer_value_l24_2409


namespace f_not_mapping_l24_2406

-- Define the sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Theorem stating that f is not a mapping from A to B
theorem f_not_mapping : ¬(∀ x ∈ A, f x ∈ B) :=
sorry

end f_not_mapping_l24_2406


namespace complex_pure_imaginary_m_l24_2412

/-- A complex number is pure imaginary if its real part is zero and its imaginary part is nonzero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_m (m : ℝ) : 
  is_pure_imaginary ((m^2 - 5*m + 6 : ℝ) + (m^2 - 3*m : ℝ) * I) → m = 2 := by
  sorry

end complex_pure_imaginary_m_l24_2412


namespace circle_equation_with_diameter_mn_l24_2417

/-- Given points M (1, -1) and N (-1, 1), prove that the equation of the circle with diameter MN is x² + y² = 2 -/
theorem circle_equation_with_diameter_mn (x y : ℝ) : 
  let m : ℝ × ℝ := (1, -1)
  let n : ℝ × ℝ := (-1, 1)
  let center : ℝ × ℝ := ((m.1 + n.1) / 2, (m.2 + n.2) / 2)
  let radius : ℝ := Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 = 2 :=
by sorry


end circle_equation_with_diameter_mn_l24_2417


namespace ln_gt_one_sufficient_not_necessary_for_x_gt_one_l24_2422

theorem ln_gt_one_sufficient_not_necessary_for_x_gt_one :
  (∃ x : ℝ, x > 1 ∧ ¬(Real.log x > 1)) ∧
  (∀ x : ℝ, Real.log x > 1 → x > 1) :=
sorry

end ln_gt_one_sufficient_not_necessary_for_x_gt_one_l24_2422


namespace grading_ratio_l24_2423

/-- A grading method for a test with 100 questions. -/
structure GradingMethod where
  total_questions : Nat
  score : Nat
  correct_answers : Nat

/-- Theorem stating the ratio of points subtracted per incorrect answer
    to points given per correct answer is 2:1 -/
theorem grading_ratio (g : GradingMethod)
  (h1 : g.total_questions = 100)
  (h2 : g.score = 73)
  (h3 : g.correct_answers = 91) :
  (g.correct_answers - g.score) / (g.total_questions - g.correct_answers) = 2 := by
  sorry


end grading_ratio_l24_2423


namespace cross_section_area_l24_2440

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  AB : ℝ
  AD : ℝ
  BD : ℝ
  AA₁ : ℝ

-- Define the theorem
theorem cross_section_area (rp : RectangularParallelepiped)
  (h1 : rp.AB = 29)
  (h2 : rp.AD = 36)
  (h3 : rp.BD = 25)
  (h4 : rp.AA₁ = 48) :
  ∃ (area : ℝ), area = 1872 ∧ area = rp.AD * Real.sqrt (rp.AA₁^2 + (Real.sqrt (rp.AD^2 + rp.AB^2 - rp.BD^2))^2) :=
by sorry

end cross_section_area_l24_2440


namespace pen_notebook_cost_l24_2437

theorem pen_notebook_cost : ∃ (p n : ℕ), 
  p > 0 ∧ n > 0 ∧ 
  15 * p + 5 * n = 13000 ∧ 
  p > n ∧ 
  p + n = 10 :=
by sorry

end pen_notebook_cost_l24_2437


namespace smallest_covering_l24_2458

/-- A rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- A configuration of rectangles covering a larger rectangle -/
structure Configuration where
  covering : Rectangle
  tiles : List Rectangle

/-- The total area covered by a list of rectangles -/
def total_area (tiles : List Rectangle) : ℕ := tiles.foldl (fun acc r => acc + r.area) 0

/-- A valid configuration has no gaps or overhangs -/
def Configuration.valid (c : Configuration) : Prop :=
  c.covering.area = total_area c.tiles

/-- The smallest valid configuration for covering with 3x4 rectangles -/
def smallest_valid_configuration : Configuration :=
  { covering := { width := 6, height := 8 }
  , tiles := List.replicate 4 { width := 3, height := 4 } }

theorem smallest_covering :
  smallest_valid_configuration.valid ∧
  (∀ c : Configuration, c.valid → c.covering.area ≥ smallest_valid_configuration.covering.area) ∧
  smallest_valid_configuration.tiles.length = 4 := by
  sorry

end smallest_covering_l24_2458


namespace painted_cubes_count_l24_2454

/-- Represents a cube with side length n --/
structure Cube (n : ℕ) where
  side_length : ℕ := n

/-- Represents a painted cube --/
structure PaintedCube (n : ℕ) extends Cube n where
  painted_faces : ℕ := 6

/-- Represents a cube that has been cut into smaller cubes --/
structure CutCube (n m : ℕ) extends PaintedCube n where
  cut_size : ℕ := m

/-- The number of smaller cubes with at least two painted faces in a cut painted cube --/
def cubes_with_two_plus_painted_faces (c : CutCube 4 1) : ℕ := 32

/-- Theorem stating that a 4x4x4 painted cube cut into 1x1x1 cubes has 32 smaller cubes with at least two painted faces --/
theorem painted_cubes_count (c : CutCube 4 1) : 
  cubes_with_two_plus_painted_faces c = 32 := by sorry

end painted_cubes_count_l24_2454


namespace arrangements_with_separation_l24_2459

/-- The number of ways to arrange 5 people in a line. -/
def total_arrangements : ℕ := 120

/-- The number of ways to arrange 5 people in a line with A and B adjacent. -/
def adjacent_arrangements : ℕ := 48

/-- The number of people in the line. -/
def num_people : ℕ := 5

/-- Theorem: The number of ways to arrange 5 people in a line with at least one person between A and B is 72. -/
theorem arrangements_with_separation :
  total_arrangements - adjacent_arrangements = 72 :=
sorry

end arrangements_with_separation_l24_2459


namespace compound_interest_problem_l24_2434

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Total amount calculation --/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Main theorem --/
theorem compound_interest_problem (principal : ℝ) :
  compound_interest principal 0.1 2 = 420 →
  total_amount principal 420 = 2420 := by
  sorry

end compound_interest_problem_l24_2434


namespace f_neg_one_eq_neg_one_l24_2448

/-- Given a function f(x) = -2x^2 + 1, prove that f(-1) = -1 -/
theorem f_neg_one_eq_neg_one :
  let f : ℝ → ℝ := fun x ↦ -2 * x^2 + 1
  f (-1) = -1 := by sorry

end f_neg_one_eq_neg_one_l24_2448


namespace nested_fraction_evaluation_l24_2486

theorem nested_fraction_evaluation :
  1 + 3 / (4 + 5 / (6 + 7/8)) = 85/52 := by
  sorry

end nested_fraction_evaluation_l24_2486


namespace catering_weight_calculation_mason_catering_weight_l24_2478

/-- Calculates the total weight of silverware and plates for a catering event. -/
theorem catering_weight_calculation (silverware_weight plate_weight : ℕ)
  (silverware_per_setting plates_per_setting : ℕ)
  (tables settings_per_table backup_settings : ℕ) : ℕ :=
  let total_settings := tables * settings_per_table + backup_settings
  let weight_per_setting := silverware_per_setting * silverware_weight + plates_per_setting * plate_weight
  total_settings * weight_per_setting

/-- Proves that the total weight of all settings for Mason's catering event is 5040 ounces. -/
theorem mason_catering_weight :
  catering_weight_calculation 4 12 3 2 15 8 20 = 5040 := by
  sorry

end catering_weight_calculation_mason_catering_weight_l24_2478


namespace entrance_exam_score_l24_2425

theorem entrance_exam_score (total_questions : ℕ) 
  (correct_score incorrect_score unattempted_score : ℤ) 
  (total_score : ℤ) :
  total_questions = 70 ∧ 
  correct_score = 3 ∧ 
  incorrect_score = -1 ∧ 
  unattempted_score = -2 ∧
  total_score = 38 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct = 27 ∧
    incorrect = 43 := by
  sorry

end entrance_exam_score_l24_2425


namespace system_solution_l24_2482

-- Define the two equations
def equation1 (x y : ℝ) : Prop :=
  8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0

def equation2 (x y : ℝ) : Prop :=
  8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0

-- Define the solution set
def solutions : Set (ℝ × ℝ) :=
  {(0, 4), (-7.5, 1), (-4.5, 0)}

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions := by
  sorry

end system_solution_l24_2482


namespace triangle_max_area_l24_2433

/-- Given a triangle ABC with area S, prove that the maximum value of S is √3/4
    when 2S + √3(AB · AC) = 0 and |BC| = √3 -/
theorem triangle_max_area (A B C : ℝ × ℝ) (S : ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  2 * S + Real.sqrt 3 * (AB.1 * AC.1 + AB.2 * AC.2) = 0 →
  BC.1^2 + BC.2^2 = 3 →
  S ≤ Real.sqrt 3 / 4 :=
by sorry

end triangle_max_area_l24_2433


namespace tom_catches_jerry_l24_2414

/-- The time it takes for Tom to catch Jerry in the given scenario --/
def catch_time : ℝ → Prop := λ t =>
  let rectangle_width : ℝ := 15
  let rectangle_length : ℝ := 30
  let tom_speed : ℝ := 5
  let jerry_speed : ℝ := 3
  16 * t^2 - 45 * Real.sqrt 2 * t - 225 = 0

theorem tom_catches_jerry : ∃ t : ℝ, catch_time t := by sorry

end tom_catches_jerry_l24_2414


namespace quadratic_inequalities_l24_2499

/-- Given a quadratic inequality and its solution set, prove the value of the coefficient and the solution set of a related inequality -/
theorem quadratic_inequalities (a : ℝ) :
  (∀ x : ℝ, (a * x^2 + 3 * x - 1 > 0) ↔ (1/2 < x ∧ x < 1)) →
  (a = -2 ∧ 
   ∀ x : ℝ, (a * x^2 - 3 * x + a^2 + 1 > 0) ↔ (-5/2 < x ∧ x < 1)) :=
by sorry

end quadratic_inequalities_l24_2499


namespace x_plus_y_equals_fifteen_l24_2479

theorem x_plus_y_equals_fifteen (x y : ℝ) 
  (h1 : (3 : ℝ)^x = 27^(y + 1)) 
  (h2 : (16 : ℝ)^y = 4^(x - 6)) : 
  x + y = 15 := by
  sorry

end x_plus_y_equals_fifteen_l24_2479


namespace classroom_arrangements_l24_2490

theorem classroom_arrangements (n : Nat) (h : n = 6) : 
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k) - Nat.choose n 1 - Nat.choose n 0 = 57 := by
  sorry

end classroom_arrangements_l24_2490


namespace optimal_garden_dimensions_l24_2462

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular garden -/
def area (g : RectangularGarden) : ℝ := g.width * g.length

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ := 2 * (g.width + g.length)

/-- Theorem: Optimal dimensions for a 600 sq ft garden with length twice the width -/
theorem optimal_garden_dimensions :
  ∃ (g : RectangularGarden),
    area g = 600 ∧
    g.length = 2 * g.width ∧
    g.width = 10 * Real.sqrt 3 ∧
    g.length = 20 * Real.sqrt 3 ∧
    ∀ (h : RectangularGarden),
      area h = 600 → h.length = 2 * h.width → perimeter h ≥ perimeter g :=
by sorry

end optimal_garden_dimensions_l24_2462


namespace isabel_paper_count_l24_2475

/-- The number of pieces of paper Isabel used -/
def used : ℕ := 156

/-- The number of pieces of paper Isabel has left -/
def left : ℕ := 744

/-- The initial number of pieces of paper Isabel bought -/
def initial : ℕ := used + left

theorem isabel_paper_count : initial = 900 := by
  sorry

end isabel_paper_count_l24_2475


namespace problem_1_problem_2_l24_2435

-- Problem 1
theorem problem_1 (m : ℝ) : 
  let A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
  let B : Set ℝ := {x | x^2 + (m+1)*x + m = 0}
  A ∩ B = B → m = 1 ∨ m = 2 := by sorry

-- Problem 2
theorem problem_2 (n : ℝ) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
  let B : Set ℝ := {x | n+1 ≤ x ∧ x ≤ 2*n-1}
  B ⊆ A → n ≤ 3 := by sorry

end problem_1_problem_2_l24_2435


namespace sin_2phi_value_l24_2411

theorem sin_2phi_value (φ : ℝ) (h : 7/13 + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120/169 := by
  sorry

end sin_2phi_value_l24_2411


namespace arithmetic_sequence_problem_l24_2432

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the arithmetic sequence equals 120. -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

/-- The main theorem: If a is an arithmetic sequence satisfying the sum condition,
    then the difference between a_7 and one-third of a_5 is 16. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_sum : SumCondition a) : 
    a 7 - (1/3) * a 5 = 16 := by
  sorry

end arithmetic_sequence_problem_l24_2432


namespace molecular_weight_CH3COOH_is_60_l24_2446

/-- The molecular weight of CH3COOH in grams per mole -/
def molecular_weight_CH3COOH : ℝ := 60

/-- The number of moles in the given sample -/
def sample_moles : ℝ := 6

/-- The total weight of the sample in grams -/
def sample_weight : ℝ := 360

/-- Theorem stating that the molecular weight of CH3COOH is 60 grams/mole -/
theorem molecular_weight_CH3COOH_is_60 :
  molecular_weight_CH3COOH = sample_weight / sample_moles :=
by sorry

end molecular_weight_CH3COOH_is_60_l24_2446


namespace card_distribution_theorem_l24_2400

/-- Represents the number of cards -/
def num_cards : ℕ := 6

/-- Represents the number of envelopes -/
def num_envelopes : ℕ := 3

/-- Represents the number of cards per envelope -/
def cards_per_envelope : ℕ := 2

/-- Calculates the number of ways to distribute cards into envelopes -/
def distribute_cards : ℕ := sorry

theorem card_distribution_theorem : 
  distribute_cards = 18 := by sorry

end card_distribution_theorem_l24_2400


namespace second_expression_value_l24_2403

theorem second_expression_value (a x : ℝ) (h1 : ((2 * a + 16) + x) / 2 = 69) (h2 : a = 26) : x = 70 := by
  sorry

end second_expression_value_l24_2403


namespace circles_intersect_l24_2445

theorem circles_intersect : 
  let c1 : ℝ × ℝ := (-2, 0)
  let r1 : ℝ := 2
  let c2 : ℝ × ℝ := (2, 1)
  let r2 : ℝ := 3
  let d := Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)
  (abs (r1 - r2) < d) ∧ (d < r1 + r2) :=
by sorry

end circles_intersect_l24_2445


namespace infinitely_many_good_pairs_l24_2427

/-- A natural number is 'good' if every prime factor in its prime factorization appears with at least the power of 2. -/
def is_good (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (∃ k : ℕ, k ≥ 2 ∧ p^k ∣ n)

/-- Definition of the sequence a_n -/
def a : ℕ → ℕ
  | 0 => 8
  | n + 1 => 4 * a n * (a n + 1)

/-- The main theorem stating that there are infinitely many pairs of consecutive 'good' numbers -/
theorem infinitely_many_good_pairs :
  ∀ n : ℕ, is_good (a n) ∧ is_good (a n + 1) :=
sorry

end infinitely_many_good_pairs_l24_2427


namespace angle_B_measure_l24_2465

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the measure of an angle in a quadrilateral
def angle_measure (q : Quadrilateral) (v : Fin 4) : ℝ := sorry

-- Theorem statement
theorem angle_B_measure (q : Quadrilateral) :
  angle_measure q 0 + angle_measure q 2 = 100 →
  angle_measure q 1 = 130 := by
  sorry

end angle_B_measure_l24_2465


namespace angle_measure_from_point_l24_2416

/-- If a point P(sin 40°, 1 + cos 40°) is on the terminal side of an acute angle α, then α = 70°. -/
theorem angle_measure_from_point (α : Real) : 
  α > 0 ∧ α < 90 ∧ 
  ∃ (P : ℝ × ℝ), P.1 = Real.sin (40 * π / 180) ∧ P.2 = 1 + Real.cos (40 * π / 180) ∧
  P.2 / P.1 = Real.tan α → 
  α = 70 * π / 180 := by
sorry

end angle_measure_from_point_l24_2416


namespace heat_required_for_temperature_change_l24_2472

/-- Specific heat capacity as a function of temperature -/
def specific_heat_capacity (c₀ α t : ℝ) : ℝ := c₀ * (1 + α * t)

/-- Amount of heat required to change temperature -/
def heat_required (m c_avg Δt : ℝ) : ℝ := m * c_avg * Δt

theorem heat_required_for_temperature_change 
  (m : ℝ) 
  (c₀ : ℝ) 
  (α : ℝ) 
  (t_initial t_final : ℝ) 
  (h_m : m = 3) 
  (h_c₀ : c₀ = 200) 
  (h_α : α = 0.05) 
  (h_t_initial : t_initial = 30) 
  (h_t_final : t_final = 80) :
  heat_required m 
    ((specific_heat_capacity c₀ α t_initial + specific_heat_capacity c₀ α t_final) / 2) 
    (t_final - t_initial) = 112500 := by
  sorry

#check heat_required_for_temperature_change

end heat_required_for_temperature_change_l24_2472


namespace six_digit_number_exists_l24_2484

/-- A six-digit number is between 100000 and 999999 -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- A five-digit number is between 10000 and 99999 -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The result of removing one digit from a six-digit number -/
def remove_digit (n : ℕ) : ℕ := n / 10

theorem six_digit_number_exists : 
  ∃! n : ℕ, is_six_digit n ∧ 
    ∃ m : ℕ, is_five_digit m ∧ 
      m = remove_digit n ∧ 
      n - m = 654321 :=
sorry

end six_digit_number_exists_l24_2484
