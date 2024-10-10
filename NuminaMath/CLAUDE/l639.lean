import Mathlib

namespace fourth_degree_polynomial_roots_l639_63911

theorem fourth_degree_polynomial_roots : 
  let p (x : ℂ) := x^4 - 16*x^2 + 51
  ∀ r : ℂ, r^2 = 8 + Real.sqrt 13 → p r = 0 :=
sorry

end fourth_degree_polynomial_roots_l639_63911


namespace cyclic_quadrilaterals_count_l639_63973

/-- The number of points on the circle -/
def n : ℕ := 20

/-- The number of ways to choose 2 points from n points -/
def choose_diameter (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of ways to choose 2 points from the remaining n-2 points -/
def choose_remaining (n : ℕ) : ℕ := (n - 2) * (n - 3) / 2

/-- The total number of cyclic quadrilaterals with one right angle -/
def total_quadrilaterals (n : ℕ) : ℕ := choose_diameter n * choose_remaining n

theorem cyclic_quadrilaterals_count :
  total_quadrilaterals n = 29070 :=
sorry

end cyclic_quadrilaterals_count_l639_63973


namespace complex_modulus_l639_63925

theorem complex_modulus (z : ℂ) (h : (z - I) / (2 - I) = I) : Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l639_63925


namespace initial_tourists_l639_63980

theorem initial_tourists (T : ℕ) : 
  (T : ℚ) - 2 - (3/7) * ((T : ℚ) - 2) = 16 → T = 30 := by
  sorry

end initial_tourists_l639_63980


namespace trailingZeros_30_factorial_l639_63960

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ := 
  let factN := factorial n
  (Nat.digits 10 factN).reverse.takeWhile (·= 0) |>.length

theorem trailingZeros_30_factorial : trailingZeros 30 = 7 := by sorry

end trailingZeros_30_factorial_l639_63960


namespace paint_coverage_per_quart_l639_63948

/-- Represents the cost of paint per quart in dollars -/
def paint_cost_per_quart : ℝ := 3.20

/-- Represents the total cost to paint the cube in dollars -/
def total_paint_cost : ℝ := 192

/-- Represents the length of one edge of the cube in feet -/
def cube_edge_length : ℝ := 10

/-- Theorem stating the coverage of one quart of paint in square feet -/
theorem paint_coverage_per_quart : 
  (6 * cube_edge_length^2) / (total_paint_cost / paint_cost_per_quart) = 10 := by
  sorry

end paint_coverage_per_quart_l639_63948


namespace eight_towns_distances_l639_63918

/-- The number of unique distances needed to connect n towns -/
def uniqueDistances (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- Theorem: For 8 towns, the number of unique distances is 28 -/
theorem eight_towns_distances : uniqueDistances 8 = 28 := by
  sorry

end eight_towns_distances_l639_63918


namespace room_expansion_theorem_l639_63959

/-- Represents a rectangular room --/
structure Room where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a room --/
def perimeter (r : Room) : ℝ := 2 * (r.length + r.breadth)

/-- Theorem: If increasing the length and breadth of a rectangular room by y feet
    results in a perimeter increase of 16 feet, then y must equal 4 feet. --/
theorem room_expansion_theorem (r : Room) (y : ℝ) 
    (h : perimeter { length := r.length + y, breadth := r.breadth + y } - perimeter r = 16) : 
  y = 4 := by
  sorry

end room_expansion_theorem_l639_63959


namespace product_of_sines_equals_one_fourth_l639_63922

theorem product_of_sines_equals_one_fourth :
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) *
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 4 := by
  sorry

end product_of_sines_equals_one_fourth_l639_63922


namespace max_parts_five_lines_max_parts_recurrence_l639_63997

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_parts m + (m + 1)

/-- Theorem stating the maximum number of parts for 5 lines -/
theorem max_parts_five_lines :
  max_parts 5 = 16 :=
by
  -- The proof goes here
  sorry

/-- Lemma for one line -/
lemma one_line_two_parts :
  max_parts 1 = 2 :=
by
  -- The proof goes here
  sorry

/-- Lemma for two lines -/
lemma two_lines_four_parts :
  max_parts 2 = 4 :=
by
  -- The proof goes here
  sorry

/-- Theorem proving the recurrence relation -/
theorem max_parts_recurrence (n : ℕ) :
  max_parts (n + 1) = max_parts n + (n + 1) :=
by
  -- The proof goes here
  sorry

end max_parts_five_lines_max_parts_recurrence_l639_63997


namespace total_keys_for_tim_l639_63999

/-- Calculates the total number of keys needed for Tim's rental properties -/
def total_keys (apartment_complex_1 apartment_complex_2 apartment_complex_3 : ℕ)
  (individual_houses : ℕ)
  (keys_per_apartment keys_per_main_entrance keys_per_house : ℕ) : ℕ :=
  (apartment_complex_1 + apartment_complex_2 + apartment_complex_3) * keys_per_apartment +
  3 * keys_per_main_entrance +
  individual_houses * keys_per_house

/-- Theorem stating the total number of keys needed for Tim's rental properties -/
theorem total_keys_for_tim : 
  total_keys 16 20 24 4 4 10 6 = 294 := by
  sorry

end total_keys_for_tim_l639_63999


namespace rationalize_and_product_l639_63929

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (((2:ℝ) + Real.sqrt 5) / ((3:ℝ) - Real.sqrt 5) = A + B * Real.sqrt C) ∧
  (A * B * C = 275) := by
  sorry

end rationalize_and_product_l639_63929


namespace insurance_slogan_equivalence_l639_63904

-- Define the universe of people
variable (Person : Type)

-- Define predicates
variable (happy : Person → Prop)
variable (has_it : Person → Prop)

-- Theorem stating the logical equivalence
theorem insurance_slogan_equivalence :
  (∀ p : Person, happy p → has_it p) ↔ (∀ p : Person, ¬has_it p → ¬happy p) :=
sorry

end insurance_slogan_equivalence_l639_63904


namespace rotation_solutions_l639_63920

-- Define the basic geometric elements
def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Point → Prop
def Plane : Type := Point → Prop

-- Define the given elements
variable (v : Line) -- Second elevation line
variable (P : Point) -- Original point
variable (P₂'' : Point) -- Inverted point parallel to second elevation plane

-- Define the geometric constructions
def rotationCircle (v : Line) (P : Point) : Set Point := sorry
def firstBisectorPlane : Plane := sorry
def planeS (v : Line) (P : Point) : Plane := sorry
def lineH₁ (v : Line) (P : Point) : Line := sorry

-- Define the number of intersections
def numIntersections (circle : Set Point) (line : Line) : ℕ := sorry

-- Define the number of solutions
def numSolutions (v : Line) (P : Point) : ℕ := sorry

-- State the theorem
theorem rotation_solutions (v : Line) (P : Point) :
  numSolutions v P = numIntersections (rotationCircle v P) (lineH₁ v P) := by sorry

end rotation_solutions_l639_63920


namespace rabbit_carrot_problem_l639_63916

theorem rabbit_carrot_problem (rabbit_holes fox_holes : ℕ) : 
  rabbit_holes * 3 = fox_holes * 5 →
  fox_holes = rabbit_holes - 6 →
  rabbit_holes * 3 = 45 := by
  sorry

end rabbit_carrot_problem_l639_63916


namespace arithmetic_sequence_sum_l639_63987

/-- An arithmetic sequence with given first and third terms -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a → a 1 = 1 → a 3 = -3 →
  a 1 - a 2 - a 3 - a 4 - a 5 = 17 := by
  sorry

end arithmetic_sequence_sum_l639_63987


namespace enrollment_calculation_l639_63949

def final_enrollment (initial : ℕ) (new_interested : ℕ) (new_dropout_rate : ℚ)
  (additional_dropouts : ℕ) (increase_factor : ℕ) (schedule_dropouts : ℕ)
  (final_rally : ℕ) (later_dropout_rate : ℚ) (graduation_rate : ℚ) : ℕ :=
  sorry

theorem enrollment_calculation :
  final_enrollment 8 8 (1/4) 2 5 2 6 (1/2) (1/2) = 19 :=
sorry

end enrollment_calculation_l639_63949


namespace rhombus_longest_diagonal_l639_63950

/-- A rhombus with area 192 and diagonal ratio 4:3 has longest diagonal of length 16√2 -/
theorem rhombus_longest_diagonal (d₁ d₂ : ℝ) : 
  d₁ * d₂ / 2 = 192 →  -- Area formula
  d₁ / d₂ = 4 / 3 →    -- Diagonal ratio
  max d₁ d₂ = 16 * Real.sqrt 2 := by
sorry

end rhombus_longest_diagonal_l639_63950


namespace line_equation_l639_63903

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The given point
def given_point : Point := { x := 3, y := -2 }

-- Theorem stating the line equation
theorem line_equation : 
  ∃ (l1 l2 : Line), 
    (point_on_line given_point l1 ∧ equal_intercepts l1) ∧
    (point_on_line given_point l2 ∧ equal_intercepts l2) ∧
    ((l1.a = 2 ∧ l1.b = 3 ∧ l1.c = 0) ∨ (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -1)) :=
by sorry

end line_equation_l639_63903


namespace solve_equation_l639_63917

theorem solve_equation (x n : ℚ) (h1 : n * (x - 3) = 15) (h2 : x = 12) : n = 5/3 := by
  sorry

end solve_equation_l639_63917


namespace strawberry_sales_chloe_strawberry_sales_l639_63981

/-- Calculates the number of dozens of strawberries sold given the cost per dozen,
    selling price per half dozen, and total profit. -/
theorem strawberry_sales 
  (cost_per_dozen : ℚ) 
  (selling_price_per_half_dozen : ℚ) 
  (total_profit : ℚ) : ℚ :=
  let profit_per_half_dozen := selling_price_per_half_dozen - cost_per_dozen / 2
  let half_dozens_sold := total_profit / profit_per_half_dozen
  let dozens_sold := half_dozens_sold / 2
  dozens_sold

/-- Proves that given the specified conditions, Chloe sold 50 dozens of strawberries. -/
theorem chloe_strawberry_sales : 
  strawberry_sales 50 30 500 = 50 := by
  sorry

end strawberry_sales_chloe_strawberry_sales_l639_63981


namespace line_slope_intercept_sum_l639_63974

/-- 
Given a line passing through points (1, 3) and (3, 7),
prove that the sum of its slope (m) and y-intercept (b) is equal to 3.
-/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by
  sorry

end line_slope_intercept_sum_l639_63974


namespace largest_a_is_four_l639_63978

/-- The largest coefficient of x^4 in a polynomial that satisfies the given conditions -/
noncomputable def largest_a : ℝ := 4

/-- A polynomial of degree 4 with real coefficients -/
def polynomial (a b c d e : ℝ) (x : ℝ) : ℝ :=
  a * x^4 + b * x^3 + c * x^2 + d * x + e

/-- The condition that the polynomial is between 0 and 1 on [-1, 1] -/
def satisfies_condition (a b c d e : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → 0 ≤ polynomial a b c d e x ∧ polynomial a b c d e x ≤ 1

/-- The theorem stating that 4 is the largest possible value for a -/
theorem largest_a_is_four :
  (∃ b c d e : ℝ, satisfies_condition largest_a b c d e) ∧
  (∀ a : ℝ, a > largest_a → ¬∃ b c d e : ℝ, satisfies_condition a b c d e) :=
sorry

end largest_a_is_four_l639_63978


namespace johnny_savings_l639_63966

theorem johnny_savings (september : ℕ) (october : ℕ) (spent : ℕ) (left : ℕ) :
  september = 30 →
  october = 49 →
  spent = 58 →
  left = 67 →
  ∃ november : ℕ, november = 46 ∧ september + october + november - spent = left :=
by sorry

end johnny_savings_l639_63966


namespace hearts_to_diamonds_ratio_l639_63909

/-- Represents the number of cards of each suit in a player's hand -/
structure CardCounts where
  spades : ℕ
  diamonds : ℕ
  hearts : ℕ
  clubs : ℕ

/-- The conditions of the card counting problem -/
def validCardCounts (c : CardCounts) : Prop :=
  c.spades + c.diamonds + c.hearts + c.clubs = 13 ∧
  c.spades + c.clubs = 7 ∧
  c.diamonds + c.hearts = 6 ∧
  c.diamonds = 2 * c.spades ∧
  c.clubs = 6

theorem hearts_to_diamonds_ratio (c : CardCounts) 
  (h : validCardCounts c) : c.hearts = 2 * c.diamonds := by
  sorry

end hearts_to_diamonds_ratio_l639_63909


namespace hotel_expenditure_l639_63951

theorem hotel_expenditure (num_persons : ℕ) (regular_spend : ℕ) (extra_spend : ℕ) : 
  num_persons = 9 →
  regular_spend = 12 →
  extra_spend = 8 →
  (num_persons - 1) * regular_spend + 
  (((num_persons - 1) * regular_spend + (regular_spend + extra_spend)) / num_persons + extra_spend) = 117 := by
  sorry

end hotel_expenditure_l639_63951


namespace monotonic_increasing_intervals_inequality_solution_l639_63988

-- Define the function f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the properties of f
def f_properties (a b c d : ℝ) : Prop :=
  -- f is symmetrical about the origin
  (∀ x, f a b c d x = -f a b c d (-x)) ∧
  -- f takes minimum value of -2 when x = 1
  (f a b c d 1 = -2) ∧
  (∀ x, f a b c d x ≥ -2)

-- Theorem for monotonically increasing intervals
theorem monotonic_increasing_intervals (a b c d : ℝ) (h : f_properties a b c d) :
  (∀ x y, x < y ∧ x < -1 → f a b c d x < f a b c d y) ∧
  (∀ x y, x < y ∧ y > 1 → f a b c d x < f a b c d y) := by sorry

-- Theorem for inequality solution
theorem inequality_solution (a b c d m : ℝ) (h : f_properties a b c d) :
  (m = 0 → ∀ x, x > 0 → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) ∧
  (m > 0 → ∀ x, (x > 4*m ∨ (0 < x ∧ x < m)) → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) ∧
  (m < 0 → ∀ x, (x > 0 ∨ (4*m < x ∧ x < m)) → f a b c d x > 5 * m * x^2 - (4 * m^2 + 3) * x) := by sorry

end monotonic_increasing_intervals_inequality_solution_l639_63988


namespace triangle_problem_l639_63933

theorem triangle_problem (a b c A B C : ℝ) 
  (h1 : a - c = (Real.sqrt 6 / 6) * b) 
  (h2 : Real.sin B = Real.sqrt 6 * Real.sin C) : 
  Real.cos A = Real.sqrt 6 / 4 ∧ 
  Real.sin (2 * A + π / 6) = (3 * Real.sqrt 5 - 1) / 8 := by
sorry

end triangle_problem_l639_63933


namespace bisection_interval_valid_l639_63913

-- Define the function f(x) = x^3 + 5
def f (x : ℝ) : ℝ := x^3 + 5

-- Theorem statement
theorem bisection_interval_valid :
  ∃ (a b : ℝ), a = -2 ∧ b = 1 ∧ f a * f b < 0 :=
by sorry

end bisection_interval_valid_l639_63913


namespace investment_split_l639_63924

theorem investment_split (alice_share bob_share total : ℕ) : 
  alice_share = 5 →
  bob_share = 3 * (total / bob_share) →
  bob_share = 3 * alice_share + 3 →
  total = bob_share * (total / bob_share) + alice_share →
  total = 113 := by
sorry

end investment_split_l639_63924


namespace car_speed_increase_car_speed_increase_proof_l639_63962

/-- Calculates the increased speed of a car given initial conditions and final results -/
theorem car_speed_increase (v : ℝ) (initial_time stop_time delay additional_distance total_distance : ℝ) : ℝ :=
  let original_time := total_distance / v
  let actual_time := original_time + stop_time + delay
  let remaining_time := actual_time - initial_time
  let new_total_distance := total_distance + additional_distance
  let distance_after_stop := new_total_distance - (v * initial_time)
  distance_after_stop / remaining_time

/-- Proves that the increased speed of the car is approximately 34.91 km/hr given the problem conditions -/
theorem car_speed_increase_proof :
  let v := 32
  let initial_time := 3
  let stop_time := 0.25
  let delay := 0.5
  let additional_distance := 28
  let total_distance := 116
  abs (car_speed_increase v initial_time stop_time delay additional_distance total_distance - 34.91) < 0.01 := by
  sorry

end car_speed_increase_car_speed_increase_proof_l639_63962


namespace f_sum_equals_sqrt2_minus_1_l639_63945

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f x + f (-x) = 0) ∧
  (∀ x, f x = f (x + 2)) ∧
  (∀ x, 0 ≤ x ∧ x < 1 → f x = 2 * x - 1)

theorem f_sum_equals_sqrt2_minus_1 (f : ℝ → ℝ) (hf : f_properties f) :
  f (1/2) + f 1 + f (3/2) + f (5/2) = Real.sqrt 2 - 1 := by
  sorry

end f_sum_equals_sqrt2_minus_1_l639_63945


namespace quadratic_roots_expression_l639_63921

theorem quadratic_roots_expression (r s : ℝ) : 
  (2 * r^2 - 3 * r = 11) → 
  (2 * s^2 - 3 * s = 11) → 
  r ≠ s →
  (4 * r^3 - 4 * s^3) / (r - s) = 31 := by
sorry

end quadratic_roots_expression_l639_63921


namespace older_brother_height_l639_63993

theorem older_brother_height
  (younger_height : ℝ)
  (your_height : ℝ)
  (older_height : ℝ)
  (h1 : younger_height = 1.1)
  (h2 : your_height = younger_height + 0.2)
  (h3 : older_height = your_height + 0.1) :
  older_height = 1.4 := by
sorry

end older_brother_height_l639_63993


namespace square_root_problem_l639_63955

theorem square_root_problem (a : ℝ) :
  (∃ (x : ℝ), x > 0 ∧ (3*a + 2)^2 = x ∧ (a + 14)^2 = x) →
  (∃ (x : ℝ), x > 0 ∧ (3*a + 2)^2 = x ∧ (a + 14)^2 = x ∧ x = 100) :=
by sorry

end square_root_problem_l639_63955


namespace least_positive_angle_l639_63994

open Real

theorem least_positive_angle (θ : ℝ) : 
  (θ > 0 ∧ ∀ φ, φ > 0 ∧ (cos (10 * π / 180) = sin (40 * π / 180) + cos φ) → θ ≤ φ) →
  cos (10 * π / 180) = sin (40 * π / 180) + cos θ →
  θ = 70 * π / 180 := by
sorry

end least_positive_angle_l639_63994


namespace dividend_proof_l639_63998

theorem dividend_proof : 
  let dividend : ℕ := 11889708
  let divisor : ℕ := 12
  let quotient : ℕ := 990809
  dividend = divisor * quotient := by sorry

end dividend_proof_l639_63998


namespace grocery_shop_sales_l639_63938

theorem grocery_shop_sales (sale1 sale2 sale4 sale5 sale6 average_sale : ℕ) 
  (h1 : sale1 = 6235)
  (h2 : sale2 = 6927)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 5191)
  (h_avg : average_sale = 6500) :
  ∃ sale3 : ℕ, 
    sale3 = 6855 ∧ 
    (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = average_sale :=
sorry

end grocery_shop_sales_l639_63938


namespace product_of_third_and_fourth_primes_above_20_l639_63934

def third_prime_above_20 : ℕ := 31

def fourth_prime_above_20 : ℕ := 37

theorem product_of_third_and_fourth_primes_above_20 :
  third_prime_above_20 * fourth_prime_above_20 = 1147 := by
  sorry

end product_of_third_and_fourth_primes_above_20_l639_63934


namespace percentage_of_temporary_workers_l639_63989

theorem percentage_of_temporary_workers
  (total_workers : ℕ)
  (technician_ratio : ℚ)
  (non_technician_ratio : ℚ)
  (permanent_technician_ratio : ℚ)
  (permanent_non_technician_ratio : ℚ)
  (h1 : technician_ratio = 9/10)
  (h2 : non_technician_ratio = 1/10)
  (h3 : permanent_technician_ratio = 9/10)
  (h4 : permanent_non_technician_ratio = 1/10)
  (h5 : technician_ratio + non_technician_ratio = 1) :
  let permanent_workers := (technician_ratio * permanent_technician_ratio +
                            non_technician_ratio * permanent_non_technician_ratio) * total_workers
  let temporary_workers := total_workers - permanent_workers
  (temporary_workers : ℚ) / (total_workers : ℚ) = 18/100 := by
  sorry

end percentage_of_temporary_workers_l639_63989


namespace movie_theater_seating_l639_63961

def seat_arrangements (n : ℕ) : ℕ :=
  if n < 7 then 0
  else (n - 4).choose 3 * 2

theorem movie_theater_seating : seat_arrangements 10 = 40 := by
  sorry

end movie_theater_seating_l639_63961


namespace orange_count_l639_63906

theorem orange_count (initial : ℕ) : 
  initial - 9 + 38 = 60 → initial = 31 := by
sorry

end orange_count_l639_63906


namespace power_equation_exponent_l639_63932

theorem power_equation_exponent (x : ℝ) (n : ℝ) (h : x ≠ 0) : 
  x^3 / x = x^n → n = 2 :=
by sorry

end power_equation_exponent_l639_63932


namespace expression_evaluation_l639_63941

theorem expression_evaluation :
  let x : ℝ := 2
  (x^2 * (x - 1) - x * (x^2 + x - 1)) = -6 :=
by sorry

end expression_evaluation_l639_63941


namespace calculate_hourly_pay_l639_63970

/-- Calculates the hourly pay per employee given the company's workforce and payment information. -/
theorem calculate_hourly_pay
  (initial_employees : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (additional_employees : ℕ)
  (total_monthly_payment : ℕ)
  (h1 : initial_employees = 500)
  (h2 : hours_per_day = 10)
  (h3 : days_per_week = 5)
  (h4 : weeks_per_month = 4)
  (h5 : additional_employees = 200)
  (h6 : total_monthly_payment = 1680000) :
  let total_employees := initial_employees + additional_employees
  let hours_per_employee := hours_per_day * days_per_week * weeks_per_month
  let total_hours := total_employees * hours_per_employee
  (total_monthly_payment / total_hours : ℚ) = 12 :=
sorry

end calculate_hourly_pay_l639_63970


namespace median_is_eight_l639_63931

-- Define the daily production values and the number of workers for each value
def daily_production : List ℕ := [5, 6, 7, 8, 9, 10]
def worker_count : List ℕ := [4, 5, 8, 9, 6, 4]

-- Define a function to calculate the median
def median (production : List ℕ) (workers : List ℕ) : ℚ :=
  sorry

-- Theorem statement
theorem median_is_eight :
  median daily_production worker_count = 8 := by
  sorry

end median_is_eight_l639_63931


namespace problem_solution_l639_63992

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * Real.log x) / (3*x + 1)

def tangent_perpendicular (a : ℝ) : Prop :=
  let f' := deriv (f a) 1
  f' * (-1) = 1

def inequality_holds (m : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → (f 0 x) ≤ m * (x - 1)

theorem problem_solution :
  (∃ a : ℝ, tangent_perpendicular a ∧ a = 0) ∧
  (∃ m : ℝ, inequality_holds m ∧ ∀ m' : ℝ, m' ≥ m → inequality_holds m') :=
sorry

end problem_solution_l639_63992


namespace fraction_simplification_l639_63967

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 72) = (5 * Real.sqrt 2) / 34 := by
  sorry

end fraction_simplification_l639_63967


namespace solution_check_l639_63957

theorem solution_check (x : ℝ) : x = 2 ↔ -1/3 * x + 2/3 = 0 := by
  sorry

end solution_check_l639_63957


namespace rational_equation_solution_l639_63905

theorem rational_equation_solution : ∃ x : ℚ, 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) ∧ 
  x = 55/13 := by
  sorry

end rational_equation_solution_l639_63905


namespace prime_divisor_congruence_l639_63926

theorem prime_divisor_congruence (p q : ℕ) : 
  Prime p → 
  Prime q → 
  q ∣ ((p^p - 1) / (p - 1)) → 
  q ≡ 1 [ZMOD p] := by
sorry

end prime_divisor_congruence_l639_63926


namespace amazing_triangle_exists_l639_63900

theorem amazing_triangle_exists : ∃ (a b c : ℕ+), 
  (a.val ^ 2 + b.val ^ 2 = c.val ^ 2) ∧ 
  (∃ (d0 d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
    d0 < 10 ∧ d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ 
    d5 < 10 ∧ d6 < 10 ∧ d7 < 10 ∧ d8 < 10 ∧
    d0 ≠ d1 ∧ d0 ≠ d2 ∧ d0 ≠ d3 ∧ d0 ≠ d4 ∧ d0 ≠ d5 ∧ d0 ≠ d6 ∧ d0 ≠ d7 ∧ d0 ≠ d8 ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧
    d7 ≠ d8 ∧
    a.val = d0 * 100 + d1 * 10 + d2 ∧
    b.val = d3 * 100 + d4 * 10 + d5 ∧
    c.val = d6 * 100 + d7 * 10 + d8) :=
by sorry

end amazing_triangle_exists_l639_63900


namespace largest_solution_of_equation_l639_63975

theorem largest_solution_of_equation : 
  ∃ (y : ℝ), y = 5 ∧ 
  3 * y^2 + 30 * y - 90 = y * (y + 18) ∧
  ∀ (z : ℝ), 3 * z^2 + 30 * z - 90 = z * (z + 18) → z ≤ y :=
by sorry

end largest_solution_of_equation_l639_63975


namespace ben_and_brothers_pizza_order_l639_63940

/-- The number of small pizzas ordered for Ben and his brothers -/
def small_pizzas_ordered (num_people : ℕ) (slices_per_person : ℕ) (large_pizza_slices : ℕ) (small_pizza_slices : ℕ) (large_pizzas_ordered : ℕ) : ℕ :=
  let total_slices_needed := num_people * slices_per_person
  let slices_from_large := large_pizzas_ordered * large_pizza_slices
  let remaining_slices := total_slices_needed - slices_from_large
  (remaining_slices + small_pizza_slices - 1) / small_pizza_slices

theorem ben_and_brothers_pizza_order :
  small_pizzas_ordered 3 12 14 8 2 = 1 := by
  sorry

end ben_and_brothers_pizza_order_l639_63940


namespace x_value_is_six_l639_63936

def star_op (a b : ℝ) : ℝ := a * b + a + b

theorem x_value_is_six (x : ℝ) : star_op 3 x = 27 → x = 6 := by
  sorry

end x_value_is_six_l639_63936


namespace math_game_result_l639_63991

theorem math_game_result (a : ℚ) : 
  (1/2 : ℚ) * (-(- a) - 2) = -1/2 * a - 1 := by
  sorry

end math_game_result_l639_63991


namespace january_salary_l639_63979

/-- The average salary calculation problem -/
theorem january_salary 
  (avg_jan_to_apr : ℝ) 
  (avg_feb_to_may : ℝ) 
  (may_salary : ℝ) 
  (h1 : avg_jan_to_apr = 8000)
  (h2 : avg_feb_to_may = 8500)
  (h3 : may_salary = 6500) :
  let jan_salary := 4 * avg_jan_to_apr - (4 * avg_feb_to_may - may_salary)
  jan_salary = 4500 := by
sorry


end january_salary_l639_63979


namespace q_squared_minus_one_div_fifteen_l639_63977

/-- The largest prime number with 2009 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2009 digits -/
axiom q_digits : 10^2008 ≤ q ∧ q < 10^2009

/-- q is the largest prime with 2009 digits -/
axiom q_largest : ∀ p, Nat.Prime p → 10^2008 ≤ p ∧ p < 10^2009 → p ≤ q

/-- The theorem to be proved -/
theorem q_squared_minus_one_div_fifteen : 15 ∣ (q^2 - 1) := by sorry

end q_squared_minus_one_div_fifteen_l639_63977


namespace unique_prime_sum_and_difference_l639_63944

theorem unique_prime_sum_and_difference : 
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    (∃ q r : ℕ, Nat.Prime q ∧ Nat.Prime r ∧ p = q + r) ∧
    (∃ s t : ℕ, Nat.Prime s ∧ Nat.Prime t ∧ p = s - t) ∧
    p = 5 :=
by sorry

end unique_prime_sum_and_difference_l639_63944


namespace vacation_cost_l639_63943

/-- If a total cost C divided among 3 people is $40 more per person than if divided among 4 people, then C equals $480. -/
theorem vacation_cost (C : ℚ) : C / 3 - C / 4 = 40 → C = 480 := by
  sorry

end vacation_cost_l639_63943


namespace flower_shop_carnation_percentage_l639_63946

theorem flower_shop_carnation_percentage :
  let carnations : ℝ := 1  -- Arbitrary non-zero value for carnations
  let violets : ℝ := (1/3) * carnations
  let tulips : ℝ := (1/4) * violets
  let roses : ℝ := tulips
  let total : ℝ := carnations + violets + tulips + roses
  (carnations / total) * 100 = 200/3 := by
sorry

end flower_shop_carnation_percentage_l639_63946


namespace tomatoes_left_l639_63902

/-- Theorem: Given a farmer with 97 tomatoes who picks 83 tomatoes, the number of tomatoes left is equal to 14. -/
theorem tomatoes_left (total : ℕ) (picked : ℕ) (h1 : total = 97) (h2 : picked = 83) :
  total - picked = 14 := by
  sorry

end tomatoes_left_l639_63902


namespace sum_of_ages_l639_63928

/-- Given the present ages of Henry and Jill, prove that their sum is 48 years. -/
theorem sum_of_ages (henry_age jill_age : ℕ) 
  (henry_present : henry_age = 29)
  (jill_present : jill_age = 19)
  (past_relation : henry_age - 9 = 2 * (jill_age - 9)) :
  henry_age + jill_age = 48 := by
  sorry

end sum_of_ages_l639_63928


namespace all_trains_return_to_initial_positions_cityN_trains_return_to_initial_positions_l639_63968

/-- Represents a metro line with a specific round trip time -/
structure MetroLine where
  roundTripTime : ℕ

/-- Represents the metro system of city N -/
structure MetroSystem where
  redLine : MetroLine
  blueLine : MetroLine
  greenLine : MetroLine

/-- Calculates the least common multiple (LCM) of three natural numbers -/
def lcm3 (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

/-- Theorem: All trains return to their initial positions after 2016 minutes -/
theorem all_trains_return_to_initial_positions (system : MetroSystem) : 
  (2016 % lcm3 system.redLine.roundTripTime system.blueLine.roundTripTime system.greenLine.roundTripTime = 0) → 
  (∀ (line : MetroLine), 2016 % line.roundTripTime = 0) :=
by
  sorry

/-- The actual metro system of city N -/
def cityN : MetroSystem :=
  { redLine := { roundTripTime := 14 }
  , blueLine := { roundTripTime := 16 }
  , greenLine := { roundTripTime := 18 }
  }

/-- Proof that the trains in city N return to their initial positions after 2016 minutes -/
theorem cityN_trains_return_to_initial_positions : 
  (2016 % lcm3 cityN.redLine.roundTripTime cityN.blueLine.roundTripTime cityN.greenLine.roundTripTime = 0) ∧
  (∀ (line : MetroLine), line ∈ [cityN.redLine, cityN.blueLine, cityN.greenLine] → 2016 % line.roundTripTime = 0) :=
by
  sorry

end all_trains_return_to_initial_positions_cityN_trains_return_to_initial_positions_l639_63968


namespace parabola_point_focus_distance_l639_63915

/-- Theorem: Distance between a point on a parabola and its focus
For a parabola defined by y^2 = 3x, if a point M on the parabola is at a distance
of 1 from the y-axis, then the distance between point M and the focus of the
parabola is 7/4. -/
theorem parabola_point_focus_distance
  (M : ℝ × ℝ) -- Point M on the parabola
  (h_on_parabola : M.2^2 = 3 * M.1) -- M is on the parabola y^2 = 3x
  (h_distance_from_y_axis : M.1 = 1) -- M is at distance 1 from y-axis
  : ∃ F : ℝ × ℝ, -- There exists a focus F
    (F.1 = 3/4 ∧ F.2 = 0) ∧ -- The focus is at (3/4, 0)
    Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = 7/4 -- Distance between M and F is 7/4
  := by sorry

end parabola_point_focus_distance_l639_63915


namespace smallest_odd_factors_above_50_l639_63971

/-- A number has an odd number of positive factors if and only if it is a perfect square. -/
def has_odd_factors (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- The smallest whole number greater than 50 that has an odd number of positive factors is 64. -/
theorem smallest_odd_factors_above_50 : 
  (∀ m : ℕ, m > 50 ∧ m < 64 → ¬(has_odd_factors m)) ∧ 
  (64 > 50 ∧ has_odd_factors 64) :=
sorry

end smallest_odd_factors_above_50_l639_63971


namespace gcd_490_910_l639_63952

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end gcd_490_910_l639_63952


namespace cos_graph_shift_l639_63956

theorem cos_graph_shift (x : ℝ) :
  3 * Real.cos (2 * x - π / 3) = 3 * Real.cos (2 * (x - π / 6)) := by
  sorry

end cos_graph_shift_l639_63956


namespace gain_percentage_l639_63947

/-- Given an article sold for $110 with a gain of $10, prove that the gain percentage is 10%. -/
theorem gain_percentage (selling_price : ℝ) (gain : ℝ) (h1 : selling_price = 110) (h2 : gain = 10) :
  (gain / (selling_price - gain)) * 100 = 10 := by
sorry

end gain_percentage_l639_63947


namespace valid_m_values_l639_63976

-- Define the set A
def A (m : ℝ) : Set ℝ := {1, m + 2, m^2 + 4}

-- State the theorem
theorem valid_m_values :
  ∀ m : ℝ, 5 ∈ A m → (m = 1 ∨ m = 3) := by
  sorry

end valid_m_values_l639_63976


namespace square_difference_63_57_l639_63953

theorem square_difference_63_57 : 63^2 - 57^2 = 720 := by
  -- The proof would go here
  sorry

end square_difference_63_57_l639_63953


namespace hexagon_triangle_angle_sum_l639_63965

theorem hexagon_triangle_angle_sum : ∀ (P Q R s t : ℝ),
  P = 40 ∧ Q = 88 ∧ R = 30 →
  (720 : ℝ) = P + Q + R + (120 - t) + (130 - s) + s + t →
  s + t = 312 := by
sorry

end hexagon_triangle_angle_sum_l639_63965


namespace distance_by_car_l639_63939

/-- Proves that the distance traveled by car is 6 kilometers -/
theorem distance_by_car (total_distance : ℝ) (h1 : total_distance = 24) :
  total_distance - (1/2 * total_distance + 1/4 * total_distance) = 6 := by
  sorry

#check distance_by_car

end distance_by_car_l639_63939


namespace inverse_proposition_l639_63990

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := ¬(x^2 > 0) → ¬(x < 0)

-- Theorem stating that inverse_prop is the inverse of original_prop
theorem inverse_proposition :
  (∀ x : ℝ, original_prop x) ↔ (∀ x : ℝ, inverse_prop x) :=
sorry

end inverse_proposition_l639_63990


namespace linear_function_property_l639_63984

/-- A linear function f(x) = ax + b satisfying f(1) = 2 and f'(1) = 2 -/
def f (x : ℝ) : ℝ := 2 * x

theorem linear_function_property : f 2 = 4 := by
  sorry

end linear_function_property_l639_63984


namespace four_sets_gemstones_l639_63982

/-- Calculates the number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let earrings_per_set := 2
  let magnets_per_earring := 2
  let buttons_per_earring := magnets_per_earring / 2
  let gemstones_per_earring := buttons_per_earring * 3
  num_sets * earrings_per_set * gemstones_per_earring

/-- Theorem stating that 4 sets of earrings require 24 gemstones -/
theorem four_sets_gemstones : gemstones_needed 4 = 24 := by
  sorry

end four_sets_gemstones_l639_63982


namespace add_7777_seconds_to_11pm_l639_63969

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- Converts a 12-hour time (with PM indicator) to 24-hour format -/
def to24Hour (hours : Nat) (isPM : Bool) : Nat :=
  sorry

theorem add_7777_seconds_to_11pm :
  let startTime := Time.mk (to24Hour 11 true) 0 0
  let endTime := addSeconds startTime 7777
  endTime = Time.mk 1 9 37 :=
sorry

end add_7777_seconds_to_11pm_l639_63969


namespace prob_implies_n_l639_63907

/-- The probability of selecting a second number greater than a first number -/
def prob : ℚ := 4995 / 10000

/-- The highest number in the range -/
def n : ℕ := 1000

/-- Theorem stating that the given probability results in the correct highest number -/
theorem prob_implies_n : 
  (n : ℚ) - 1 = 2 * n * prob := by sorry

end prob_implies_n_l639_63907


namespace alkyne_ch_bond_polarization_l639_63972

-- Define the hybridization states
inductive Hybridization
| sp
| sp2
| sp3

-- Define a function to represent the s-character percentage
def sCharacter (h : Hybridization) : ℚ :=
  match h with
  | .sp  => 1/2
  | .sp2 => 1/3
  | .sp3 => 1/4

-- Define a function to represent electronegativity
def electronegativity (h : Hybridization) : ℝ := sorry

-- Define a function to represent bond polarization strength
def bondPolarizationStrength (h : Hybridization) : ℝ := sorry

-- Theorem statement
theorem alkyne_ch_bond_polarization :
  (∀ h : Hybridization, h ≠ Hybridization.sp → electronegativity Hybridization.sp > electronegativity h) ∧
  (∀ h : Hybridization, bondPolarizationStrength h = electronegativity h) ∧
  (bondPolarizationStrength Hybridization.sp > bondPolarizationStrength Hybridization.sp2) ∧
  (bondPolarizationStrength Hybridization.sp > bondPolarizationStrength Hybridization.sp3) := by
  sorry

end alkyne_ch_bond_polarization_l639_63972


namespace concert_cost_l639_63983

theorem concert_cost (ticket_price : ℚ) (processing_fee_rate : ℚ) 
  (parking_fee : ℚ) (entrance_fee : ℚ) (num_people : ℕ) :
  ticket_price = 50 ∧ 
  processing_fee_rate = 0.15 ∧ 
  parking_fee = 10 ∧ 
  entrance_fee = 5 ∧ 
  num_people = 2 → 
  (ticket_price + ticket_price * processing_fee_rate) * num_people + 
  parking_fee + entrance_fee * num_people = 135 :=
by sorry

end concert_cost_l639_63983


namespace correct_calculation_l639_63919

theorem correct_calculation (x : ℝ) : 2 * x^2 - x^2 = x^2 := by
  sorry

end correct_calculation_l639_63919


namespace sum_of_squares_values_l639_63986

theorem sum_of_squares_values (x y z : ℝ) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (eq1 : x^2 = 2 + y)
  (eq2 : y^2 = 2 + z)
  (eq3 : z^2 = 2 + x) :
  x^2 + y^2 + z^2 = 5 ∨ x^2 + y^2 + z^2 = 6 ∨ x^2 + y^2 + z^2 = 9 :=
by sorry

end sum_of_squares_values_l639_63986


namespace planes_parallel_if_perpendicular_to_same_line_l639_63996

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (a : Line) (α β : Plane) :
  perpendicular a α → perpendicular a β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l639_63996


namespace luncheon_invitees_l639_63910

theorem luncheon_invitees (no_shows : ℕ) (people_per_table : ℕ) (tables_needed : ℕ) :
  no_shows = 35 →
  people_per_table = 2 →
  tables_needed = 5 →
  no_shows + (people_per_table * tables_needed) = 45 :=
by sorry

end luncheon_invitees_l639_63910


namespace triangle_properties_l639_63995

theorem triangle_properties (a b c : ℝ) (A B C : Real) (S : ℝ) (D : ℝ × ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  a * Real.sin B = b * Real.sin (A + π / 3) →
  S = 2 * Real.sqrt 3 →
  S = (1 / 2) * b * c * Real.sin A →
  D.1 = (2 / 3) * b →
  D.2 = 0 →
  (∃ (AD : ℝ), AD ≥ (4 * Real.sqrt 3) / 3 ∧
    AD^2 = (1 / 9) * c^2 + (4 / 9) * b^2 + (16 / 9)) →
  A = π / 3 ∧ (∃ (AD_min : ℝ), AD_min = (4 * Real.sqrt 3) / 3 ∧
    ∀ (AD : ℝ), AD ≥ AD_min) :=
by sorry

end triangle_properties_l639_63995


namespace number_of_jeans_to_wash_l639_63954

/-- The number of shirts Alex has to wash -/
def shirts : ℕ := 18

/-- The number of pants Alex has to wash -/
def pants : ℕ := 12

/-- The number of sweaters Alex has to wash -/
def sweaters : ℕ := 17

/-- The maximum number of items the washing machine can wash per cycle -/
def items_per_cycle : ℕ := 15

/-- The time in minutes each washing cycle takes -/
def minutes_per_cycle : ℕ := 45

/-- The total time in hours it takes to wash all clothes -/
def total_wash_time : ℕ := 3

/-- Theorem stating the number of jeans Alex has to wash -/
theorem number_of_jeans_to_wash : 
  ∃ (jeans : ℕ), 
    (shirts + pants + sweaters + jeans) = 
      (total_wash_time * 60 / minutes_per_cycle) * items_per_cycle ∧
    jeans = 13 := by
  sorry

end number_of_jeans_to_wash_l639_63954


namespace solution_value_l639_63908

theorem solution_value (x a : ℝ) : x = 4 ∧ 5 * (x - 1) - 3 * a = -3 → a = 6 := by
  sorry

end solution_value_l639_63908


namespace jill_peach_count_jill_peach_count_proof_l639_63964

/-- Given the peach distribution among Jake, Steven, Jill, and Sam, prove that Jill has 6 peaches. -/
theorem jill_peach_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun jake steven jill sam =>
    (jake = steven - 18) →
    (steven = jill + 13) →
    (steven = 19) →
    (sam = 2 * jill) →
    (jill = 6)

/-- Proof of the theorem -/
theorem jill_peach_count_proof : ∃ jake steven jill sam, jill_peach_count jake steven jill sam :=
  sorry

end jill_peach_count_jill_peach_count_proof_l639_63964


namespace billiard_ball_weight_l639_63923

/-- Given an empty box weighing 0.5 kg and a box containing 6 identical billiard balls
    weighing 1.82 kg, prove that each billiard ball weighs 0.22 kg. -/
theorem billiard_ball_weight (empty_box_weight : ℝ) (full_box_weight : ℝ) :
  empty_box_weight = 0.5 →
  full_box_weight = 1.82 →
  (full_box_weight - empty_box_weight) / 6 = 0.22 := by
  sorry

end billiard_ball_weight_l639_63923


namespace martian_amoeba_nim_exists_l639_63937

-- Define the set of Martian amoebas
inductive MartianAmoeba
  | A
  | B
  | C

-- Define the function type
def AmoebaNim := MartianAmoeba → Nat

-- Define the bitwise XOR operation
def bxor (a b : Nat) : Nat :=
  Nat.xor a b

-- State the theorem
theorem martian_amoeba_nim_exists : ∃ (f : AmoebaNim),
  (bxor (f MartianAmoeba.A) (f MartianAmoeba.B) = f MartianAmoeba.C) ∧
  (bxor (f MartianAmoeba.A) (f MartianAmoeba.C) = f MartianAmoeba.B) ∧
  (bxor (f MartianAmoeba.B) (f MartianAmoeba.C) = f MartianAmoeba.A) :=
by
  sorry

end martian_amoeba_nim_exists_l639_63937


namespace candles_used_l639_63927

/-- Given a candle that lasts 8 nights when burned for 1 hour per night,
    calculate the number of candles used when burned for 2 hours per night for 24 nights -/
theorem candles_used
  (nights_per_candle : ℕ)
  (hours_per_night : ℕ)
  (total_nights : ℕ)
  (h1 : nights_per_candle = 8)
  (h2 : hours_per_night = 2)
  (h3 : total_nights = 24) :
  (total_nights * hours_per_night) / (nights_per_candle * 1) = 6 := by
  sorry

end candles_used_l639_63927


namespace difference_of_squares_special_case_l639_63963

theorem difference_of_squares_special_case : (23 * 2 + 15)^2 - (23 * 2 - 15)^2 = 2760 := by
  sorry

end difference_of_squares_special_case_l639_63963


namespace count_integers_with_5_or_6_l639_63930

/-- The number of integers among the first 729 positive integers in base 9 
    that contain either 5 or 6 (or both) as a digit -/
def count_with_5_or_6 : ℕ := 386

/-- The base of the number system we're working with -/
def base : ℕ := 9

/-- The number of smallest positive integers we're considering -/
def total_count : ℕ := 729

theorem count_integers_with_5_or_6 :
  count_with_5_or_6 = total_count - (base - 2)^3 ∧
  total_count = base^3 := by
  sorry

end count_integers_with_5_or_6_l639_63930


namespace complementary_angles_ratio_l639_63985

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  a > b →  -- a is the larger angle
  a = 50 :=  -- The larger angle measures 50°
by sorry

end complementary_angles_ratio_l639_63985


namespace art_kits_count_l639_63912

theorem art_kits_count (total_students : ℕ) (students_per_kit : ℕ) 
  (artworks_group1 : ℕ) (artworks_group2 : ℕ) (total_artworks : ℕ) : ℕ :=
  let num_kits := total_students / students_per_kit
  let half_students := total_students / 2
  let artworks_from_group1 := half_students * artworks_group1
  let artworks_from_group2 := half_students * artworks_group2
  by
    have h1 : total_students = 10 := by sorry
    have h2 : students_per_kit = 2 := by sorry
    have h3 : artworks_group1 = 3 := by sorry
    have h4 : artworks_group2 = 4 := by sorry
    have h5 : total_artworks = 35 := by sorry
    have h6 : artworks_from_group1 + artworks_from_group2 = total_artworks := by sorry
    exact num_kits

end art_kits_count_l639_63912


namespace total_marks_math_physics_l639_63901

/-- Proves that the total marks in mathematics and physics is 60 -/
theorem total_marks_math_physics (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 40 →
  M + P = 60 := by
sorry

end total_marks_math_physics_l639_63901


namespace pizza_delivery_time_l639_63914

theorem pizza_delivery_time (total_pizzas : ℕ) (double_order_stops : ℕ) (time_per_stop : ℕ) : 
  total_pizzas = 12 →
  double_order_stops = 2 →
  time_per_stop = 4 →
  (total_pizzas - 2 * double_order_stops + double_order_stops) * time_per_stop = 40 :=
by sorry

end pizza_delivery_time_l639_63914


namespace cube_root_inequality_l639_63942

theorem cube_root_inequality (a b : ℝ) (h : a > b) : a^(1/3) > b^(1/3) := by
  sorry

end cube_root_inequality_l639_63942


namespace beads_per_bracelet_l639_63958

/-- Given the following conditions:
    - Nancy has 40 metal beads and 20 more pearl beads than metal beads
    - Rose has 20 crystal beads and twice as many stone beads as crystal beads
    - They can make 20 bracelets
    Prove that the number of beads in each bracelet is 8. -/
theorem beads_per_bracelet :
  let nancy_metal : ℕ := 40
  let nancy_pearl : ℕ := nancy_metal + 20
  let rose_crystal : ℕ := 20
  let rose_stone : ℕ := 2 * rose_crystal
  let total_bracelets : ℕ := 20
  let total_beads : ℕ := nancy_metal + nancy_pearl + rose_crystal + rose_stone
  (total_beads / total_bracelets : ℕ) = 8 := by
sorry

end beads_per_bracelet_l639_63958


namespace quadratic_coefficient_l639_63935

theorem quadratic_coefficient (b m : ℝ) : 
  b < 0 → 
  (∀ x, x^2 + b*x + 1/5 = (x + m)^2 + 1/20) → 
  b = -2 * Real.sqrt (3/20) := by
sorry

end quadratic_coefficient_l639_63935
