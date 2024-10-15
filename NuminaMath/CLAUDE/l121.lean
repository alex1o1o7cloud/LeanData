import Mathlib

namespace NUMINAMATH_CALUDE_construction_company_gravel_purchase_l121_12129

theorem construction_company_gravel_purchase
  (total_material : ℝ)
  (sand : ℝ)
  (gravel : ℝ)
  (h1 : total_material = 14.02)
  (h2 : sand = 8.11)
  (h3 : total_material = sand + gravel) :
  gravel = 5.91 :=
by
  sorry

end NUMINAMATH_CALUDE_construction_company_gravel_purchase_l121_12129


namespace NUMINAMATH_CALUDE_pell_solution_valid_pell_recurrence_relation_l121_12125

/-- Pell's equation solution type -/
structure PellSolution (D : ℕ) where
  x : ℤ
  y : ℤ
  eq : x^2 - D * y^2 = 1

/-- Generate the kth Pell solution -/
def genPellSolution (D : ℕ) (x₀ y₀ : ℤ) (k : ℕ) : PellSolution D :=
  sorry

theorem pell_solution_valid (D : ℕ) (x₀ y₀ : ℤ) (h : ¬ ∃ n : ℕ, n^2 = D) 
    (h₀ : x₀^2 - D * y₀^2 = 1) (k : ℕ) :
  let sol := genPellSolution D x₀ y₀ k
  sol.x^2 - D * sol.y^2 = 1 :=
sorry

theorem pell_recurrence_relation (D : ℕ) (x₀ y₀ : ℤ) (h : ¬ ∃ n : ℕ, n^2 = D) 
    (h₀ : x₀^2 - D * y₀^2 = 1) (k : ℕ) :
  let x₁ := (genPellSolution D x₀ y₀ (k+1)).x
  let x₂ := (genPellSolution D x₀ y₀ (k+2)).x
  let x := (genPellSolution D x₀ y₀ k).x
  x₂ = 2 * x₀ * x₁ - x :=
sorry

end NUMINAMATH_CALUDE_pell_solution_valid_pell_recurrence_relation_l121_12125


namespace NUMINAMATH_CALUDE_multiple_with_binary_digits_l121_12126

theorem multiple_with_binary_digits (n : ℕ+) : 
  ∃ m : ℕ, 
    (n : ℕ) ∣ m ∧ 
    (Nat.digits 2 m).length ≤ n ∧ 
    ∀ d ∈ Nat.digits 2 m, d = 0 ∨ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiple_with_binary_digits_l121_12126


namespace NUMINAMATH_CALUDE_cookie_calories_is_250_l121_12196

/-- The number of calories in a cookie, given the total lunch calories,
    burger calories, carrot stick calories, and number of carrot sticks. -/
def cookie_calories (total_lunch_calories burger_calories carrot_stick_calories num_carrot_sticks : ℕ) : ℕ :=
  total_lunch_calories - (burger_calories + carrot_stick_calories * num_carrot_sticks)

/-- Theorem stating that each cookie has 250 calories under the given conditions. -/
theorem cookie_calories_is_250 :
  cookie_calories 750 400 20 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_cookie_calories_is_250_l121_12196


namespace NUMINAMATH_CALUDE_cosine_sum_identity_l121_12190

theorem cosine_sum_identity : 
  Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + 
  Real.sin (80 * π / 180) * Real.sin (20 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_identity_l121_12190


namespace NUMINAMATH_CALUDE_total_distance_is_thirteen_l121_12181

/-- Represents the time in minutes to travel one mile on a given day -/
def time_per_mile (day : Nat) : Nat :=
  12 + 6 * day

/-- Represents the distance traveled in miles on a given day -/
def distance (day : Nat) : Nat :=
  60 / (time_per_mile day)

/-- The total distance traveled over five days -/
def total_distance : Nat :=
  (List.range 5).map distance |>.sum

theorem total_distance_is_thirteen :
  total_distance = 13 := by sorry

end NUMINAMATH_CALUDE_total_distance_is_thirteen_l121_12181


namespace NUMINAMATH_CALUDE_line_vector_at_zero_l121_12156

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_at_zero : 
  (∀ (t : ℝ), line_vector t = line_vector 0 + t • (line_vector 1 - line_vector 0)) →
  line_vector (-2) = (2, 4, 10) →
  line_vector 1 = (-1, -3, -5) →
  line_vector 0 = (0, -2/3, 0) := by sorry

end NUMINAMATH_CALUDE_line_vector_at_zero_l121_12156


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_30_l121_12135

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ, 
    is_prime p ∧ 
    is_prime q ∧ 
    p > 30 ∧ 
    q > 30 ∧ 
    p ≠ q ∧ 
    p * q = 1147 ∧ 
    (∀ p' q' : ℕ, is_prime p' → is_prime q' → p' > 30 → q' > 30 → p' ≠ q' → p' * q' ≥ 1147) :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_30_l121_12135


namespace NUMINAMATH_CALUDE_greatest_multiple_of_9_l121_12177

def digits : List Nat := [3, 6, 7, 8, 9]

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

def list_to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ l1.toFinset = l2.toFinset

theorem greatest_multiple_of_9 :
  (∀ l : List Nat, l.length = 5 → is_permutation l digits →
    is_multiple_of_9 (list_to_number l) →
    list_to_number l ≤ 98763) ∧
  (list_to_number [9, 8, 7, 6, 3] = 98763) ∧
  (is_multiple_of_9 98763) :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_9_l121_12177


namespace NUMINAMATH_CALUDE_coefficient_is_three_l121_12185

/-- The derivative function for our equation -/
noncomputable def derivative (q : ℝ) : ℝ := 3 * q - 3

/-- The second derivative of 6 -/
def second_derivative_of_six : ℝ := 210

/-- The coefficient of q in the equation q' = 3q - 3 -/
def coefficient : ℝ := 3

theorem coefficient_is_three : coefficient = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_is_three_l121_12185


namespace NUMINAMATH_CALUDE_fraction_simplification_l121_12197

theorem fraction_simplification (a : ℚ) (h : a ≠ 2) :
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l121_12197


namespace NUMINAMATH_CALUDE_license_plate_count_l121_12138

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of odd digits --/
def num_odd_digits : ℕ := 5

/-- The number of even digits --/
def num_even_digits : ℕ := 5

/-- The total number of possible license plates --/
def total_plates : ℕ := num_letters ^ 3 * num_odd_digits * num_even_digits

theorem license_plate_count :
  total_plates = 439400 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l121_12138


namespace NUMINAMATH_CALUDE_term2017_is_one_sixty_fifth_l121_12183

/-- A proper fraction is a pair of natural numbers (n, d) where 0 < n < d -/
def ProperFraction := { p : ℕ × ℕ // 0 < p.1 ∧ p.1 < p.2 }

/-- The sequence of proper fractions arranged by increasing denominators and numerators -/
def properFractionSequence : ℕ → ProperFraction :=
  sorry

/-- The 2017th term of the proper fraction sequence -/
def term2017 : ProperFraction :=
  properFractionSequence 2017

theorem term2017_is_one_sixty_fifth :
  term2017 = ⟨(1, 65), sorry⟩ := by sorry

end NUMINAMATH_CALUDE_term2017_is_one_sixty_fifth_l121_12183


namespace NUMINAMATH_CALUDE_problem_solution_l121_12150

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 2 = y^2) 
  (h3 : x / 5 = 3*y) : 
  x = 112.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l121_12150


namespace NUMINAMATH_CALUDE_largest_square_tile_l121_12169

theorem largest_square_tile (board_width board_length tile_size : ℕ) : 
  board_width = 19 → 
  board_length = 29 → 
  tile_size > 0 →
  (∀ n : ℕ, n > 1 → (board_width % n = 0 ∧ board_length % n = 0) → False) →
  tile_size = 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_l121_12169


namespace NUMINAMATH_CALUDE_first_bus_students_l121_12123

theorem first_bus_students (total_buses : ℕ) (initial_avg : ℕ) (remaining_avg : ℕ) : 
  total_buses = 6 → 
  initial_avg = 28 → 
  remaining_avg = 26 → 
  (total_buses * initial_avg - (total_buses - 1) * remaining_avg) = 38 := by
sorry

end NUMINAMATH_CALUDE_first_bus_students_l121_12123


namespace NUMINAMATH_CALUDE_number_exceeding_45_percent_l121_12163

theorem number_exceeding_45_percent : ∃ x : ℝ, x = 0.45 * x + 1000 ∧ x = 1000 / 0.55 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_45_percent_l121_12163


namespace NUMINAMATH_CALUDE_base3_product_theorem_l121_12142

/-- Converts a base 3 number to decimal --/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- Converts a decimal number to base 3 --/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- Multiplies two base 3 numbers --/
def multiplyBase3 (a b : List Nat) : List Nat :=
  decimalToBase3 (base3ToDecimal a * base3ToDecimal b)

theorem base3_product_theorem :
  multiplyBase3 [2, 0, 1] [2, 1] = [2, 0, 2, 1] := by sorry

end NUMINAMATH_CALUDE_base3_product_theorem_l121_12142


namespace NUMINAMATH_CALUDE_drummer_drum_sticks_l121_12178

/-- Calculates the total number of drum stick sets used by a drummer over multiple performances. -/
def total_drum_sticks (sticks_per_show : ℕ) (tossed_after_show : ℕ) (num_nights : ℕ) : ℕ :=
  (sticks_per_show + tossed_after_show) * num_nights

/-- Theorem stating that a drummer using 5 sets per show, tossing 6 sets after each show, 
    for 30 nights, uses 330 sets of drum sticks in total. -/
theorem drummer_drum_sticks : total_drum_sticks 5 6 30 = 330 := by
  sorry

end NUMINAMATH_CALUDE_drummer_drum_sticks_l121_12178


namespace NUMINAMATH_CALUDE_expand_expression_l121_12124

theorem expand_expression (x y : ℝ) : 24 * (3 * x + 4 * y - 2) = 72 * x + 96 * y - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l121_12124


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l121_12172

theorem yellow_marbles_count (blue : ℕ) (red : ℕ) (yellow : ℕ) :
  blue = 7 →
  red = 11 →
  (yellow : ℚ) / (blue + red + yellow : ℚ) = 1/4 →
  yellow = 6 :=
by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l121_12172


namespace NUMINAMATH_CALUDE_solve_digit_equation_l121_12159

theorem solve_digit_equation (a b d v t r : ℕ) : 
  a + b = v →
  v + d = t →
  t + a = r →
  b + d + r = 18 →
  1 ≤ a ∧ a ≤ 9 →
  1 ≤ b ∧ b ≤ 9 →
  1 ≤ d ∧ d ≤ 9 →
  1 ≤ v ∧ v ≤ 9 →
  1 ≤ t ∧ t ≤ 9 →
  1 ≤ r ∧ r ≤ 9 →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_solve_digit_equation_l121_12159


namespace NUMINAMATH_CALUDE_red_bus_to_orange_car_ratio_l121_12188

/-- The lengths of buses and a car, measured in feet. -/
structure VehicleLengths where
  red_bus : ℝ
  orange_car : ℝ
  yellow_bus : ℝ

/-- The conditions of the problem. -/
def problem_conditions (v : VehicleLengths) : Prop :=
  ∃ (x : ℝ),
    v.red_bus = x * v.orange_car ∧
    v.yellow_bus = 3.5 * v.orange_car ∧
    v.yellow_bus = v.red_bus - 6 ∧
    v.red_bus = 48

/-- The theorem statement. -/
theorem red_bus_to_orange_car_ratio 
  (v : VehicleLengths) (h : problem_conditions v) :
  v.red_bus / v.orange_car = 4 := by
  sorry


end NUMINAMATH_CALUDE_red_bus_to_orange_car_ratio_l121_12188


namespace NUMINAMATH_CALUDE_permutations_count_l121_12109

theorem permutations_count (n : ℕ) : Nat.factorial n = 6227020800 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_permutations_count_l121_12109


namespace NUMINAMATH_CALUDE_triangle_problem_l121_12184

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) : 
  -- Given conditions
  (2 * t.b * (2 * t.b - t.c) * Real.cos t.A = t.a^2 + t.b^2 - t.c^2) →
  ((1/2) * t.b * t.c * Real.sin t.A = 25 * Real.sqrt 3 / 4) →
  (t.a = 5) →
  -- Conclusions
  (t.A = π/3 ∧ t.b + t.c = 10) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l121_12184


namespace NUMINAMATH_CALUDE_scientific_notation_320000_l121_12136

theorem scientific_notation_320000 : 
  320000 = 3.2 * (10 : ℝ) ^ 5 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_320000_l121_12136


namespace NUMINAMATH_CALUDE_concentric_polygons_inequality_l121_12137

theorem concentric_polygons_inequality (n : ℕ) (R r : ℝ) (h : Fin n → ℝ) :
  n ≥ 3 →
  R > 0 →
  r > 0 →
  r < R →
  (∀ i, h i > 0) →
  (∀ i, h i ≤ R) →
  R * Real.cos (π / n) ≥ (Finset.sum Finset.univ h) / n ∧ (Finset.sum Finset.univ h) / n ≥ r :=
by sorry

end NUMINAMATH_CALUDE_concentric_polygons_inequality_l121_12137


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l121_12134

/-- A straight line passing through (-3, -2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through (-3, -2) -/
  passes_through_point : slope * (-3) + y_intercept = -2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : ∃ (a : ℝ), a ≠ 0 ∧ slope * a + y_intercept = 0 ∧ a + y_intercept = 0

/-- The equation of the line is either 2x - 3y = 0 or x + y + 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.slope = 2/3 ∧ l.y_intercept = 0) ∨ (l.slope = -1 ∧ l.y_intercept = -5) := by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l121_12134


namespace NUMINAMATH_CALUDE_f_properties_l121_12193

def f (x : ℝ) : ℝ := x * (x + 1) * (x + 2) * (x + 3)

theorem f_properties :
  (∀ x : ℝ, f x ≥ -1) ∧
  (∃ x : ℝ, f x = -1) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-2) (-1) → f x ≤ -9/16) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-2) (-1) ∧ f x = -9/16) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l121_12193


namespace NUMINAMATH_CALUDE_max_e_value_l121_12100

def is_valid_number (d e : ℕ) : Prop :=
  d ≤ 9 ∧ e ≤ 9 ∧ 
  (600000 + d * 10000 + 28000 + e) % 18 = 0

theorem max_e_value :
  ∃ (d : ℕ), is_valid_number d 8 ∧
  ∀ (d' e' : ℕ), is_valid_number d' e' → e' ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_e_value_l121_12100


namespace NUMINAMATH_CALUDE_sophies_doughnuts_price_l121_12173

theorem sophies_doughnuts_price (cupcake_price cupcake_count doughnut_count
  pie_price pie_count cookie_price cookie_count total_spent : ℚ) :
  cupcake_price = 2 →
  cupcake_count = 5 →
  doughnut_count = 6 →
  pie_price = 2 →
  pie_count = 4 →
  cookie_price = 0.60 →
  cookie_count = 15 →
  total_spent = 33 →
  cupcake_price * cupcake_count + doughnut_count * 1 + pie_price * pie_count + cookie_price * cookie_count = total_spent :=
by
  sorry

#check sophies_doughnuts_price

end NUMINAMATH_CALUDE_sophies_doughnuts_price_l121_12173


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l121_12151

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (b^2 + a + 1) / (a * b) ≥ 2 * Real.sqrt 10 + 6 :=
by sorry

theorem min_value_achievable (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 1 ∧
    (b₀^2 + a₀ + 1) / (a₀ * b₀) = 2 * Real.sqrt 10 + 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l121_12151


namespace NUMINAMATH_CALUDE_conic_focal_distance_l121_12110

/-- The focal distance of a conic curve x^2 + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_focal_distance (m : ℝ) : 
  (m^2 = 2 * 8) →  -- m is the geometric mean between 2 and 8
  let focal_distance := 
    if m > 0 then 2 * Real.sqrt 3  -- Ellipse case
    else 2 * Real.sqrt 5           -- Hyperbola case
  (∃ (x y : ℝ), x^2 + y^2/m = 1) →  -- The conic curve exists
  focal_distance = 2 * Real.sqrt 3 ∨ focal_distance = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_conic_focal_distance_l121_12110


namespace NUMINAMATH_CALUDE_square_sum_value_l121_12157

theorem square_sum_value (x y : ℝ) 
  (h1 : y + 6 = (x - 3)^2)
  (h2 : x + 6 = (y - 3)^2)
  (h3 : x ≠ y) : 
  x^2 + y^2 = 43 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l121_12157


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l121_12118

/-- Calculate the total cost of cable for a neighborhood given the following conditions:
- 18 east-west streets, each 2 miles long
- 10 north-south streets, each 4 miles long
- 5 miles of cable needed to electrify 1 mile of street
- Cable costs $2000 per mile
-/
theorem neighborhood_cable_cost :
  let east_west_streets := 18
  let east_west_length := 2
  let north_south_streets := 10
  let north_south_length := 4
  let cable_per_mile := 5
  let cost_per_mile := 2000
  let total_street_length := east_west_streets * east_west_length + north_south_streets * north_south_length
  let total_cable_length := total_street_length * cable_per_mile
  let total_cost := total_cable_length * cost_per_mile
  total_cost = 760000 := by
  sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l121_12118


namespace NUMINAMATH_CALUDE_probability_five_successes_in_seven_trials_l121_12191

/-- The probability of getting exactly 5 successes in 7 trials with a success probability of 3/4 -/
theorem probability_five_successes_in_seven_trials :
  let n : ℕ := 7  -- number of trials
  let k : ℕ := 5  -- number of successes
  let p : ℚ := 3/4  -- probability of success on each trial
  Nat.choose n k * p^k * (1 - p)^(n - k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_successes_in_seven_trials_l121_12191


namespace NUMINAMATH_CALUDE_correct_elderly_sample_l121_12162

/-- Represents the composition of employees in a company and its sample --/
structure EmployeeSample where
  total : ℕ
  young : ℕ
  middleAged : ℕ
  elderly : ℕ
  sampledYoung : ℕ
  sampledElderly : ℕ

/-- Checks if the employee sample is valid according to the given conditions --/
def isValidSample (s : EmployeeSample) : Prop :=
  s.total = 430 ∧
  s.young = 160 ∧
  s.middleAged = 2 * s.elderly ∧
  s.total = s.young + s.middleAged + s.elderly ∧
  s.sampledYoung = 32

/-- Theorem stating that for a valid sample, the number of sampled elderly should be 18 --/
theorem correct_elderly_sample (s : EmployeeSample) 
  (h : isValidSample s) : s.sampledElderly = 18 := by
  sorry


end NUMINAMATH_CALUDE_correct_elderly_sample_l121_12162


namespace NUMINAMATH_CALUDE_area_enclosed_by_parabola_and_line_l121_12147

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2

-- Define the line function
def line (_ : ℝ) : ℝ := 1

-- Theorem statement
theorem area_enclosed_by_parabola_and_line :
  ∫ x in (-1)..1, (line x - parabola x) = 4/3 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_parabola_and_line_l121_12147


namespace NUMINAMATH_CALUDE_hyperbola_s_squared_l121_12158

/-- Represents a hyperbola with the equation (y^2 / a^2) - (x^2 / b^2) = 1 --/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point (x, y) lies on the hyperbola --/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  (y^2 / h.a^2) - (x^2 / h.b^2) = 1

theorem hyperbola_s_squared (h : Hyperbola) :
  h.a = 3 →
  h.contains 0 (-3) →
  h.contains 4 (-2) →
  ∃ s, h.contains 2 s ∧ s^2 = 441/36 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_s_squared_l121_12158


namespace NUMINAMATH_CALUDE_red_balls_count_l121_12167

theorem red_balls_count (total : ℕ) (p : ℚ) (h_total : total = 12) (h_p : p = 1 / 22) :
  ∃ (r : ℕ), r ≤ total ∧ 
    (r : ℚ) / total * ((r - 1) : ℚ) / (total - 1) = p ∧
    r = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l121_12167


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_bounds_l121_12148

theorem roots_quadratic_equation_bounds (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - a*x₁ + a^2 - a = 0 ∧ x₂^2 - a*x₂ + a^2 - a = 0) →
  (0 ≤ a ∧ a ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_bounds_l121_12148


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l121_12128

/-- The shortest distance from a point on the parabola y = x^2 to the line 2x - y = 4 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2}
  let line := {p : ℝ × ℝ | 2 * p.1 - p.2 = 4}
  let distance (p : ℝ × ℝ) := |2 * p.1 - p.2 - 4| / Real.sqrt 5
  ∃ (p : ℝ × ℝ), p ∈ parabola ∧
    (∀ (q : ℝ × ℝ), q ∈ parabola → distance p ≤ distance q) ∧
    distance p = 3 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_to_line_l121_12128


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l121_12139

def f (x : ℝ) := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y →
    (3 * x^2 + 1 = 4) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l121_12139


namespace NUMINAMATH_CALUDE_volume_ratio_l121_12176

/-- Represents a square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- Represents a pyramid formed by folding a square along its diagonal -/
structure Pyramid :=
  (base : Square)

/-- Represents a sphere circumscribing a pyramid -/
structure CircumscribedSphere :=
  (pyramid : Pyramid)

/-- The volume of a pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

/-- The volume of a circumscribed sphere -/
def sphere_volume (s : CircumscribedSphere) : ℝ := sorry

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio (s : CircumscribedSphere) :
  sphere_volume s / pyramid_volume s.pyramid = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_volume_ratio_l121_12176


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l121_12132

/-- Given that x varies inversely with y³ and x = 8 when y = 1, prove that x = 1 when y = 2 -/
theorem inverse_variation_problem (x y : ℝ) (h : ∀ y : ℝ, y ≠ 0 → ∃ k : ℝ, x * y^3 = k) :
  (∃ k : ℝ, 8 * 1^3 = k) → (∃ x : ℝ, x * 2^3 = 8) → (∃ x : ℝ, x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l121_12132


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l121_12195

/-- The capacity of a gasoline tank in gallons -/
def tank_capacity : ℚ := 100 / 3

/-- The amount of gasoline added to the tank in gallons -/
def added_gasoline : ℚ := 5

/-- The initial fill level of the tank as a fraction of its capacity -/
def initial_fill : ℚ := 3 / 4

/-- The final fill level of the tank as a fraction of its capacity -/
def final_fill : ℚ := 9 / 10

theorem tank_capacity_proof :
  (final_fill * tank_capacity - initial_fill * tank_capacity = added_gasoline) ∧
  (tank_capacity > 0) := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l121_12195


namespace NUMINAMATH_CALUDE_mothers_house_distance_l121_12121

/-- The distance between your house and your mother's house -/
def total_distance : ℝ := 234.0

/-- The distance you have traveled so far -/
def traveled_distance : ℝ := 156.0

/-- Theorem stating that the total distance to your mother's house is 234.0 miles -/
theorem mothers_house_distance :
  (traveled_distance = (2/3) * total_distance) →
  total_distance = 234.0 :=
by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_mothers_house_distance_l121_12121


namespace NUMINAMATH_CALUDE_max_regions_quadratic_trinomials_l121_12153

/-- The maximum number of regions into which the coordinate plane can be divided by n quadratic trinomials -/
def max_regions (n : ℕ) : ℕ := n^2 + 1

/-- Theorem stating that the maximum number of regions created by n quadratic trinomials is n^2 + 1 -/
theorem max_regions_quadratic_trinomials (n : ℕ) :
  max_regions n = n^2 + 1 := by sorry

end NUMINAMATH_CALUDE_max_regions_quadratic_trinomials_l121_12153


namespace NUMINAMATH_CALUDE_min_value_of_f_l121_12127

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x - 2| + 3 * |x - 3| + 4 * |x - 4|

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 8) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l121_12127


namespace NUMINAMATH_CALUDE_container_weights_l121_12111

theorem container_weights (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (w1 : x + y = 110) (w2 : y + z = 130) (w3 : z + x = 150) :
  x + y + z = 195 := by
  sorry

end NUMINAMATH_CALUDE_container_weights_l121_12111


namespace NUMINAMATH_CALUDE_squirrel_acorns_l121_12105

/-- Given 5 squirrels collecting 575 acorns in total, and each squirrel needing 130 acorns for winter,
    the number of additional acorns each squirrel needs to collect is 15. -/
theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (acorns_needed : ℕ) :
  num_squirrels = 5 →
  total_acorns = 575 →
  acorns_needed = 130 →
  acorns_needed - (total_acorns / num_squirrels) = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_squirrel_acorns_l121_12105


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l121_12182

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 3 / 8)
  (h2 : material2 = 1 / 3)
  (h3 : leftover = 15 / 40) :
  material1 + material2 - leftover = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l121_12182


namespace NUMINAMATH_CALUDE_cubic_roots_classification_l121_12108

/-- The discriminant of the cubic equation x³ + px + q = 0 -/
def discriminant (p q : ℝ) : ℝ := 4 * p^3 + 27 * q^2

/-- Theorem about the nature of roots for the cubic equation x³ + px + q = 0 -/
theorem cubic_roots_classification (p q : ℝ) :
  (discriminant p q > 0 → ∃ (x : ℂ), x^3 + p*x + q = 0 ∧ (∀ y : ℂ, y^3 + p*y + q = 0 → y = x ∨ y.im ≠ 0)) ∧
  (discriminant p q < 0 → ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^3 + p*x + q = 0 ∧ y^3 + p*y + q = 0 ∧ z^3 + p*z + q = 0) ∧
  (discriminant p q = 0 ∧ p = 0 ∧ q = 0 → ∃ (x : ℝ), ∀ y : ℝ, y^3 + p*y + q = 0 → y = x) ∧
  (discriminant p q = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ (x y : ℝ), x ≠ y ∧ x^3 + p*x + q = 0 ∧ y^3 + p*y + q = 0 ∧ 
    ∀ z : ℝ, z^3 + p*z + q = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_classification_l121_12108


namespace NUMINAMATH_CALUDE_equal_angles_on_curve_l121_12189

/-- Curve C defined by y² = 4x -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point on the x-axis -/
def xAxisPoint (x : ℝ) : ℝ × ℝ := (x, 0)

/-- Line passing through two points -/
def line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- Angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem equal_angles_on_curve (m n : ℝ) (hm : m > 0) (hmn : m + n = 0)
    (A B : ℝ × ℝ) (hA : A ∈ C) (hB : B ∈ C)
    (hline : A ∈ line (xAxisPoint m) B ∧ B ∈ line (xAxisPoint m) A) :
  angle (A - xAxisPoint n) (xAxisPoint m - xAxisPoint n) =
  angle (B - xAxisPoint n) (xAxisPoint m - xAxisPoint n) := by
  sorry

end NUMINAMATH_CALUDE_equal_angles_on_curve_l121_12189


namespace NUMINAMATH_CALUDE_equation_solution_l121_12149

theorem equation_solution :
  ∃! x : ℚ, (x + 5) / (x - 3) = (x - 2) / (x + 4) :=
by
  use -1
  sorry

end NUMINAMATH_CALUDE_equation_solution_l121_12149


namespace NUMINAMATH_CALUDE_percentage_selected_state_A_l121_12130

theorem percentage_selected_state_A (
  total_A : ℕ) (total_B : ℕ) (percent_B : ℚ) (diff : ℕ) :
  total_A = 8000 →
  total_B = 8000 →
  percent_B = 7 / 100 →
  (total_B : ℚ) * percent_B = (total_A : ℚ) * (7 / 100 : ℚ) + diff →
  diff = 80 →
  (7 / 100 : ℚ) * total_A = (total_A : ℚ) * (7 / 100 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_percentage_selected_state_A_l121_12130


namespace NUMINAMATH_CALUDE_exam_score_standard_deviation_l121_12180

/-- Given an exam with mean score 74 and standard deviation σ,
    prove that if 98 is 3σ above the mean and 58 is k⋅σ below the mean,
    then k = 2 -/
theorem exam_score_standard_deviation (σ : ℝ) (k : ℝ) 
    (h1 : 98 = 74 + 3 * σ)
    (h2 : 58 = 74 - k * σ) : 
    k = 2 := by sorry

end NUMINAMATH_CALUDE_exam_score_standard_deviation_l121_12180


namespace NUMINAMATH_CALUDE_polynomial_equality_l121_12104

theorem polynomial_equality (p : ℝ → ℝ) :
  (∀ x, p x + (x^5 + 3*x^3 + 9*x) = 7*x^3 + 24*x^2 + 25*x + 1) →
  (∀ x, p x = -x^5 + 4*x^3 + 24*x^2 + 16*x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l121_12104


namespace NUMINAMATH_CALUDE_one_contribution_before_john_l121_12107

-- Define the problem parameters
def john_donation : ℝ := 100
def new_average : ℝ := 75
def increase_percentage : ℝ := 0.5

-- Define the theorem
theorem one_contribution_before_john
  (n : ℝ) -- Initial number of contributions
  (A : ℝ) -- Initial average contribution
  (h1 : A + increase_percentage * A = new_average) -- New average is 50% higher
  (h2 : n * A + john_donation = (n + 1) * new_average) -- Total amount equality
  : n = 1 := by
  sorry


end NUMINAMATH_CALUDE_one_contribution_before_john_l121_12107


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l121_12164

/-- A triangle with integer side lengths and perimeter 24 has a maximum side length of 11 -/
theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧ 
  a + b + c = 24 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  c ≤ 11 := by
  sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l121_12164


namespace NUMINAMATH_CALUDE_container_capacity_l121_12141

theorem container_capacity (C : ℝ) : 
  C > 0 → 
  0.30 * C + 36 = 0.75 * C → 
  C = 80 := by
sorry

end NUMINAMATH_CALUDE_container_capacity_l121_12141


namespace NUMINAMATH_CALUDE_trampoline_jumps_l121_12103

theorem trampoline_jumps (ronald_jumps rupert_jumps : ℕ) : 
  ronald_jumps = 157 →
  rupert_jumps > ronald_jumps →
  rupert_jumps + ronald_jumps = 400 →
  rupert_jumps - ronald_jumps = 86 := by
sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l121_12103


namespace NUMINAMATH_CALUDE_infinite_series_sum_l121_12131

/-- The sum of the infinite series ∑(k=1 to ∞) k^2 / 3^k is equal to 4.5 -/
theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ)^2 / 3^k) = (9/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l121_12131


namespace NUMINAMATH_CALUDE_divisor_sum_l121_12160

theorem divisor_sum (k m : ℕ) 
  (h1 : 30^k ∣ 929260) 
  (h2 : 20^m ∣ 929260) : 
  (3^k - k^3) + (2^m - m^3) = 2 := by
sorry

end NUMINAMATH_CALUDE_divisor_sum_l121_12160


namespace NUMINAMATH_CALUDE_final_attendance_is_1166_l121_12145

/-- Calculates the final number of spectators after a series of changes in attendance at a football game. -/
def final_attendance (initial_total initial_boys initial_girls : ℕ) : ℕ :=
  let initial_adults := initial_total - (initial_boys + initial_girls)
  
  -- After first quarter
  let boys_after_q1 := initial_boys - (initial_boys / 4)
  let girls_after_q1 := initial_girls - (initial_girls / 8)
  let adults_after_q1 := initial_adults - (initial_adults / 5)
  
  -- After halftime
  let boys_after_half := boys_after_q1 + (boys_after_q1 * 5 / 100)
  let girls_after_half := girls_after_q1 + (girls_after_q1 * 7 / 100)
  let adults_after_half := adults_after_q1 + 50
  
  -- After third quarter
  let boys_after_q3 := boys_after_half - (boys_after_half * 3 / 100)
  let girls_after_q3 := girls_after_half - (girls_after_half * 4 / 100)
  let adults_after_q3 := adults_after_half + (adults_after_half * 2 / 100)
  
  -- Final numbers
  let final_boys := boys_after_q3 + 15
  let final_girls := girls_after_q3 + 25
  let final_adults := adults_after_q3 - (adults_after_q3 / 100)
  
  final_boys + final_girls + final_adults

/-- Theorem stating that given the initial conditions, the final attendance is 1166. -/
theorem final_attendance_is_1166 : final_attendance 1300 350 450 = 1166 := by
  sorry

end NUMINAMATH_CALUDE_final_attendance_is_1166_l121_12145


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l121_12154

/-- Three points in a 2D plane are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

theorem collinear_points_x_value :
  let p : ℝ × ℝ := (1, 1)
  let a : ℝ × ℝ := (2, -4)
  let b : ℝ × ℝ := (x, 9)
  collinear p a b → x = 3 := by
sorry


end NUMINAMATH_CALUDE_collinear_points_x_value_l121_12154


namespace NUMINAMATH_CALUDE_a_minus_b_values_l121_12166

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) :
  a - b = 10 ∨ a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l121_12166


namespace NUMINAMATH_CALUDE_inverse_function_sum_l121_12133

/-- Given two real numbers a and b, define f and its inverse --/
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b

def f_inv (a b : ℝ) : ℝ → ℝ := λ x ↦ b * x^2 + a

/-- Theorem stating that if f and f_inv are inverse functions, then a + b = 1 --/
theorem inverse_function_sum (a b : ℝ) : 
  (∀ x, f a b (f_inv a b x) = x) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_sum_l121_12133


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l121_12117

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 20)
  (h2 : selling_price = 35) :
  (selling_price - cost_price) / cost_price * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l121_12117


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l121_12186

open Complex

theorem z_in_third_quadrant : 
  let z : ℂ := (2 + I) / (I^5 - 1)
  (z.re < 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l121_12186


namespace NUMINAMATH_CALUDE_point_relationships_l121_12168

def A : Set (ℝ × ℝ) := {(x, y) | x + 2*y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2*x + y - 5 ≤ 0}

theorem point_relationships :
  (¬ ((0 : ℝ), (0 : ℝ)) ∈ A) ∧ ((1 : ℝ), (1 : ℝ)) ∈ A := by sorry

end NUMINAMATH_CALUDE_point_relationships_l121_12168


namespace NUMINAMATH_CALUDE_cyclic_iff_perpendicular_diagonals_l121_12106

-- Define the basic geometric objects
variable (A B C D P Q R S : Point)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the incircle and its tangency points
def has_incircle_with_tangent_points (A B C D P Q R S : Point) : Prop := sorry

-- Define cyclic quadrilateral
def is_cyclic (A B C D : Point) : Prop := sorry

-- Define perpendicularity
def perpendicular (P Q R S : Point) : Prop := sorry

-- The main theorem
theorem cyclic_iff_perpendicular_diagonals 
  (h_quad : is_quadrilateral A B C D)
  (h_incircle : has_incircle_with_tangent_points A B C D P Q R S) :
  is_cyclic A B C D ↔ perpendicular P R Q S := by sorry

end NUMINAMATH_CALUDE_cyclic_iff_perpendicular_diagonals_l121_12106


namespace NUMINAMATH_CALUDE_total_lives_for_eight_friends_l121_12122

/-- Calculates the total number of lives for a group of friends in a video game -/
def totalLives (numFriends : ℕ) (livesPerFriend : ℕ) : ℕ :=
  numFriends * livesPerFriend

/-- Proves that the total number of lives for 8 friends with 8 lives each is 64 -/
theorem total_lives_for_eight_friends : totalLives 8 8 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_lives_for_eight_friends_l121_12122


namespace NUMINAMATH_CALUDE_range_of_m_l121_12165

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x + 1 > 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (¬(p m ∨ q m)) → (m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l121_12165


namespace NUMINAMATH_CALUDE_evaluation_of_expression_l121_12170

theorem evaluation_of_expression : (4^4 - 4*(4-1)^4)^4 = 21381376 := by
  sorry

end NUMINAMATH_CALUDE_evaluation_of_expression_l121_12170


namespace NUMINAMATH_CALUDE_handshakes_count_l121_12199

/-- The number of people in the gathering -/
def total_people : ℕ := 30

/-- The number of people who know each other (Group A) -/
def group_a : ℕ := 20

/-- The number of people who know no one (Group B) -/
def group_b : ℕ := 10

/-- The number of handshakes between Group A and Group B -/
def handshakes_between : ℕ := group_a * group_b

/-- The number of handshakes within Group B -/
def handshakes_within : ℕ := group_b * (group_b - 1) / 2

/-- The total number of handshakes -/
def total_handshakes : ℕ := handshakes_between + handshakes_within

theorem handshakes_count : total_handshakes = 245 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l121_12199


namespace NUMINAMATH_CALUDE_custom_cartesian_product_of_A_and_B_l121_12143

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2*x - x^2)}
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define the custom cartesian product
def customCartesianProduct (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem custom_cartesian_product_of_A_and_B :
  customCartesianProduct A B = Set.Icc 0 1 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_custom_cartesian_product_of_A_and_B_l121_12143


namespace NUMINAMATH_CALUDE_count_numbers_greater_than_threshold_l121_12113

def numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
def threshold : ℚ := 1.1

theorem count_numbers_greater_than_threshold : 
  (numbers.filter (λ x => x > threshold)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_greater_than_threshold_l121_12113


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l121_12119

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_plus_one :
  f (f (f (-1))) = Real.pi + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l121_12119


namespace NUMINAMATH_CALUDE_gold_cube_value_scaling_l121_12140

-- Define the properties of the 4-inch gold cube
def gold_cube_4inch_value : ℝ := 500
def gold_cube_4inch_side : ℝ := 4

-- Define the side length of the 5-inch gold cube
def gold_cube_5inch_side : ℝ := 5

-- Function to calculate the volume of a cube
def cube_volume (side : ℝ) : ℝ := side ^ 3

-- Theorem statement
theorem gold_cube_value_scaling :
  let v4 := cube_volume gold_cube_4inch_side
  let v5 := cube_volume gold_cube_5inch_side
  let scale_factor := v5 / v4
  let scaled_value := gold_cube_4inch_value * scale_factor
  ⌊scaled_value + 0.5⌋ = 977 := by sorry

end NUMINAMATH_CALUDE_gold_cube_value_scaling_l121_12140


namespace NUMINAMATH_CALUDE_inequality_solution_l121_12116

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l121_12116


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l121_12115

theorem min_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (3 : ℝ) = Real.sqrt (3^a * 3^b) → 
  (∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l121_12115


namespace NUMINAMATH_CALUDE_binary_110_equals_6_l121_12102

def binary_to_decimal (b₂ b₁ b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_110_equals_6 :
  binary_to_decimal 1 1 0 = 6 := by sorry

end NUMINAMATH_CALUDE_binary_110_equals_6_l121_12102


namespace NUMINAMATH_CALUDE_no_integer_solutions_cube_equation_l121_12152

theorem no_integer_solutions_cube_equation :
  ¬ ∃ (x y z : ℤ), x^3 + y^3 = 9*z + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_cube_equation_l121_12152


namespace NUMINAMATH_CALUDE_terminal_side_quadrant_l121_12175

/-- Given an angle α that satisfies α = 45° + k · 180° where k is an integer,
    the terminal side of α falls in either the first or third quadrant. -/
theorem terminal_side_quadrant (k : ℤ) (α : Real) 
  (h : α = 45 + k * 180) : 
  (0 < α % 360 ∧ α % 360 < 90) ∨ (180 < α % 360 ∧ α % 360 < 270) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_quadrant_l121_12175


namespace NUMINAMATH_CALUDE_two_tailed_coin_probability_l121_12198

/-- The probability of drawing the 2-tailed coin given that the flip resulted in tails -/
def prob_two_tailed_given_tails (total_coins : ℕ) (fair_coins : ℕ) (p_tails_fair : ℚ) : ℚ :=
  let two_tailed_coins := total_coins - fair_coins
  let p_two_tailed := two_tailed_coins / total_coins
  let p_tails_two_tailed := 1
  let p_tails := p_two_tailed * p_tails_two_tailed + (fair_coins / total_coins) * p_tails_fair
  (p_tails_two_tailed * p_two_tailed) / p_tails

theorem two_tailed_coin_probability :
  prob_two_tailed_given_tails 10 9 (1/2) = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_two_tailed_coin_probability_l121_12198


namespace NUMINAMATH_CALUDE_pie_crust_flour_usage_l121_12171

/-- Given that 40 pie crusts each use 1/8 cup of flour, 
    prove that 25 larger pie crusts using the same total amount of flour 
    will each use 1/5 cup of flour. -/
theorem pie_crust_flour_usage 
  (initial_crusts : ℕ) 
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ) 
  (total_flour : ℚ) :
  initial_crusts = 40 →
  initial_flour_per_crust = 1/8 →
  new_crusts = 25 →
  total_flour = initial_crusts * initial_flour_per_crust →
  total_flour = new_crusts * (1/5 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_usage_l121_12171


namespace NUMINAMATH_CALUDE_book_area_calculation_l121_12120

/-- Calculates the area of a book given its length expression, width, and conversion factor. -/
theorem book_area_calculation (x : ℝ) (inch_to_cm : ℝ) : 
  x = 5 → 
  inch_to_cm = 2.54 → 
  (3 * x - 4) * ((5 / 2) * inch_to_cm) = 69.85 := by
  sorry

end NUMINAMATH_CALUDE_book_area_calculation_l121_12120


namespace NUMINAMATH_CALUDE_employee_salary_problem_l121_12192

/-- Given two employees M and N with a total weekly salary of $605,
    where M's salary is 120% of N's, prove that N's salary is $275 per week. -/
theorem employee_salary_problem (total_salary m_salary n_salary : ℝ) :
  total_salary = 605 →
  m_salary = 1.2 * n_salary →
  total_salary = m_salary + n_salary →
  n_salary = 275 := by
sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l121_12192


namespace NUMINAMATH_CALUDE_power_of_two_equality_l121_12144

theorem power_of_two_equality (x : ℕ) : (1 / 4 : ℝ) * (2 ^ 30) = 2 ^ x → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l121_12144


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l121_12179

theorem initial_mean_calculation (n : ℕ) (correct_value wrong_value : ℝ) (correct_mean : ℝ) :
  n = 20 ∧ 
  correct_value = 160 ∧ 
  wrong_value = 135 ∧ 
  correct_mean = 151.25 →
  (n * correct_mean - correct_value + wrong_value) / n = 152.5 := by
sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l121_12179


namespace NUMINAMATH_CALUDE_original_students_per_section_l121_12101

theorem original_students_per_section 
  (S : ℕ) -- Initial number of sections
  (x : ℕ) -- Initial number of students per section
  (h1 : S + 3 = 16) -- After admission, there are S + 3 sections, totaling 16
  (h2 : S * x + 24 = 16 * 21) -- Total students after admission equals 16 sections of 21 students each
  : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_original_students_per_section_l121_12101


namespace NUMINAMATH_CALUDE_probability_is_one_over_432_l121_12155

/-- Represents a fair die with 6 faces -/
def Die := Fin 6

/-- Represents the outcome of tossing four dice -/
def FourDiceOutcome := (Die × Die × Die × Die)

/-- Checks if a sequence of four numbers forms an arithmetic progression with common difference 1 -/
def isArithmeticProgression (a b c d : ℕ) : Prop :=
  b - a = 1 ∧ c - b = 1 ∧ d - c = 1

/-- The set of all possible outcomes when tossing four fair dice -/
def allOutcomes : Finset FourDiceOutcome := sorry

/-- The set of favorable outcomes (forming an arithmetic progression) -/
def favorableOutcomes : Finset FourDiceOutcome := sorry

/-- The probability of getting an arithmetic progression when tossing four fair dice -/
def probabilityOfArithmeticProgression : ℚ :=
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

/-- Theorem stating that the probability of getting an arithmetic progression is 1/432 -/
theorem probability_is_one_over_432 :
  probabilityOfArithmeticProgression = 1 / 432 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_over_432_l121_12155


namespace NUMINAMATH_CALUDE_expression_value_l121_12114

theorem expression_value : 
  let x : ℤ := -1
  let y : ℤ := 2
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l121_12114


namespace NUMINAMATH_CALUDE_two_digit_addition_l121_12187

theorem two_digit_addition (A : ℕ) : A < 10 → (10 * A + 7 + 30 = 77) ↔ A = 4 := by sorry

end NUMINAMATH_CALUDE_two_digit_addition_l121_12187


namespace NUMINAMATH_CALUDE_six_thirty_six_am_metric_l121_12161

/-- Represents a time in the metric system -/
structure MetricTime where
  hours : Nat
  minutes : Nat

/-- Converts normal time (in minutes since midnight) to metric time -/
def normalToMetric (normalMinutes : Nat) : MetricTime :=
  let totalMetricMinutes := normalMinutes * 25 / 36
  { hours := totalMetricMinutes / 100
  , minutes := totalMetricMinutes % 100 }

theorem six_thirty_six_am_metric :
  normalToMetric (6 * 60 + 36) = { hours := 2, minutes := 75 } := by
  sorry

#eval 100 * (normalToMetric (6 * 60 + 36)).hours +
      10 * ((normalToMetric (6 * 60 + 36)).minutes / 10) +
      (normalToMetric (6 * 60 + 36)).minutes % 10

end NUMINAMATH_CALUDE_six_thirty_six_am_metric_l121_12161


namespace NUMINAMATH_CALUDE_series_sum_l121_12112

/-- The sum of the series $\sum_{n=1}^{\infty} \frac{3^n}{9^n - 1}$ is equal to $\frac{1}{2}$ -/
theorem series_sum : ∑' n, (3^n : ℝ) / (9^n - 1) = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_l121_12112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l121_12174

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 1 + a 4 + a 7 = 39)
  (h_sum2 : a 2 + a 5 + a 8 = 33) :
  a 3 + a 6 + a 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l121_12174


namespace NUMINAMATH_CALUDE_dog_shampoo_time_l121_12146

theorem dog_shampoo_time (total_time hosing_time : ℕ) (shampoo_count : ℕ) : 
  total_time = 55 → 
  hosing_time = 10 → 
  shampoo_count = 3 → 
  (total_time - hosing_time) / shampoo_count = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_shampoo_time_l121_12146


namespace NUMINAMATH_CALUDE_intersection_line_slope_l121_12194

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10*x + 8*y + 24 = 0

-- Define the slope of the line passing through the intersection points
def slope_of_intersection_line (circle1 circle2 : ℝ → ℝ → Prop) : ℝ := 1

-- Theorem statement
theorem intersection_line_slope :
  slope_of_intersection_line circle1 circle2 = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l121_12194
