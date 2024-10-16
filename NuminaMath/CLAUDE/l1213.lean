import Mathlib

namespace NUMINAMATH_CALUDE_trig_identity_l1213_121316

theorem trig_identity : Real.sin (47 * π / 180) * Real.cos (17 * π / 180) + 
                        Real.cos (47 * π / 180) * Real.cos (107 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1213_121316


namespace NUMINAMATH_CALUDE_x_value_l1213_121354

theorem x_value (x : ℚ) (h : 1/4 - 1/6 = 4/x) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1213_121354


namespace NUMINAMATH_CALUDE_kylie_coins_to_laura_l1213_121379

/-- The number of coins Kylie collected from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie collected from her brother -/
def brother_coins : ℕ := 13

/-- The number of coins Kylie collected from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie had left after giving some to Laura -/
def coins_left : ℕ := 15

/-- The total number of coins Kylie collected -/
def total_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

/-- The number of coins Kylie gave to Laura -/
def coins_given_to_laura : ℕ := total_coins - coins_left

theorem kylie_coins_to_laura : coins_given_to_laura = 21 := by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_to_laura_l1213_121379


namespace NUMINAMATH_CALUDE_geometric_relations_l1213_121359

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometric_relations 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (perpendicular_plane m α ∧ perpendicular_plane n β ∧ perpendicular m n → perpendicular_planes α β) ∧
  (perpendicular_plane m α ∧ parallel n β ∧ parallel_planes α β → perpendicular m n) := by
  sorry

end NUMINAMATH_CALUDE_geometric_relations_l1213_121359


namespace NUMINAMATH_CALUDE_solve_equation_l1213_121365

theorem solve_equation : ∃ x : ℝ, 0.5 * x + (0.3 * 0.2) = 0.26 ∧ x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1213_121365


namespace NUMINAMATH_CALUDE_fraction_simplification_l1213_121335

theorem fraction_simplification 
  (b c d x y z : ℝ) :
  (c * x * (b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3) + 
   d * z * (b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3)) / 
  (c * x + d * z) = 
  b^2 * x^3 + 3 * c^2 * y^3 + c^2 * z^3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1213_121335


namespace NUMINAMATH_CALUDE_line_equation_and_x_intercept_l1213_121353

-- Define the points A and B
def A : ℝ × ℝ := (-2, -3)
def B : ℝ × ℝ := (3, 0)

-- Define the line l
def l (x y : ℝ) : Prop := 5 * x + 3 * y + 2 = 0

-- Define symmetry about a line
def symmetric_about_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), l M.1 M.2 ∧ 
  (M.1 = (A.1 + B.1) / 2) ∧ 
  (M.2 = (A.2 + B.2) / 2)

-- Theorem statement
theorem line_equation_and_x_intercept :
  symmetric_about_line A B l →
  (∀ x y, l x y ↔ 5 * x + 3 * y + 2 = 0) ∧
  (∃ x, l x 0 ∧ x = -2/5) :=
sorry

end NUMINAMATH_CALUDE_line_equation_and_x_intercept_l1213_121353


namespace NUMINAMATH_CALUDE_sin_product_45_deg_l1213_121303

theorem sin_product_45_deg (α β : Real) 
  (h1 : Real.sin (α + β) = 0.2) 
  (h2 : Real.cos (α - β) = 0.3) : 
  Real.sin (α + Real.pi/4) * Real.sin (β + Real.pi/4) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_45_deg_l1213_121303


namespace NUMINAMATH_CALUDE_dividend_calculation_l1213_121369

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 6) : 
  divisor * quotient + remainder = 159 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1213_121369


namespace NUMINAMATH_CALUDE_modular_inverse_15_l1213_121337

theorem modular_inverse_15 :
  (¬ ∃ x : ℤ, (15 * x) % 1105 = 1) ∧
  (∃ x : ℤ, (15 * x) % 221 = 1) ∧
  ((15 * 59) % 221 = 1) := by
sorry

end NUMINAMATH_CALUDE_modular_inverse_15_l1213_121337


namespace NUMINAMATH_CALUDE_factor_expression_l1213_121319

theorem factor_expression (b : ℝ) : 145 * b^2 + 29 * b = 29 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1213_121319


namespace NUMINAMATH_CALUDE_water_added_to_tank_l1213_121315

theorem water_added_to_tank (tank_capacity : ℚ) 
  (h1 : tank_capacity = 64) 
  (initial_fraction : ℚ) 
  (h2 : initial_fraction = 3/4) 
  (final_fraction : ℚ) 
  (h3 : final_fraction = 7/8) : 
  (final_fraction - initial_fraction) * tank_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_tank_l1213_121315


namespace NUMINAMATH_CALUDE_solve_for_y_l1213_121372

theorem solve_for_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 4) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1213_121372


namespace NUMINAMATH_CALUDE_competition_selection_count_l1213_121314

def male_count : ℕ := 5
def female_count : ℕ := 3
def selection_size : ℕ := 3

def selection_count : ℕ := 45

theorem competition_selection_count :
  (Nat.choose female_count 2 * Nat.choose male_count 1) +
  (Nat.choose female_count 1 * Nat.choose male_count 2) = selection_count :=
by sorry

end NUMINAMATH_CALUDE_competition_selection_count_l1213_121314


namespace NUMINAMATH_CALUDE_ap_square_identity_l1213_121383

/-- Three consecutive terms of an arithmetic progression -/
structure ArithmeticProgressionTerms (α : Type*) [Add α] [Sub α] where
  a : α
  b : α
  c : α
  is_ap : b - a = c - b

/-- Theorem: For any three consecutive terms of an arithmetic progression,
    a^2 + 8bc = (2b + c)^2 -/
theorem ap_square_identity {α : Type*} [CommRing α] (terms : ArithmeticProgressionTerms α) :
  terms.a ^ 2 + 8 * terms.b * terms.c = (2 * terms.b + terms.c) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ap_square_identity_l1213_121383


namespace NUMINAMATH_CALUDE_fraction_calculation_l1213_121328

theorem fraction_calculation : (1/3 + 1/6) * 4/7 * 5/9 = 10/63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1213_121328


namespace NUMINAMATH_CALUDE_modular_inverse_14_mod_1001_l1213_121326

theorem modular_inverse_14_mod_1001 :
  ∃ x : ℕ, x ≤ 1000 ∧ (14 * x) % 1001 = 1 :=
by
  use 143
  sorry

end NUMINAMATH_CALUDE_modular_inverse_14_mod_1001_l1213_121326


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1213_121347

/-- Represents the quantity ratio of products A, B, and C -/
def quantity_ratio : Fin 3 → ℕ
  | 0 => 2  -- Product A
  | 1 => 3  -- Product B
  | 2 => 5  -- Product C
  | _ => 0  -- Unreachable case

/-- The sample size of product A -/
def sample_size_A : ℕ := 10

/-- Calculates the total sample size based on the sample size of product A -/
def total_sample_size (sample_A : ℕ) : ℕ :=
  sample_A * (quantity_ratio 0 + quantity_ratio 1 + quantity_ratio 2) / quantity_ratio 0

theorem stratified_sample_size :
  total_sample_size sample_size_A = 50 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1213_121347


namespace NUMINAMATH_CALUDE_tricycle_count_l1213_121306

/-- Represents the number of wheels for each vehicle type -/
def wheels_per_vehicle : Fin 3 → ℕ
  | 0 => 2  -- bicycles
  | 1 => 3  -- tricycles
  | 2 => 2  -- scooters

/-- Proves that the number of tricycles is 4 given the conditions of the parade -/
theorem tricycle_count (vehicles : Fin 3 → ℕ) 
  (total_children : vehicles 0 + vehicles 1 + vehicles 2 = 10)
  (total_wheels : vehicles 0 * wheels_per_vehicle 0 + 
                  vehicles 1 * wheels_per_vehicle 1 + 
                  vehicles 2 * wheels_per_vehicle 2 = 27) :
  vehicles 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l1213_121306


namespace NUMINAMATH_CALUDE_additional_cars_problem_solution_l1213_121310

theorem additional_cars (front_initial : Nat) (back_initial : Nat) (total_end : Nat) : Nat :=
  let total_initial := front_initial + back_initial
  total_end - total_initial

theorem problem_solution : 
  let front_initial := 100
  let back_initial := 2 * front_initial
  let total_end := 700
  additional_cars front_initial back_initial total_end = 400 := by
sorry

end NUMINAMATH_CALUDE_additional_cars_problem_solution_l1213_121310


namespace NUMINAMATH_CALUDE_no_solution_iff_m_equals_five_l1213_121363

theorem no_solution_iff_m_equals_five :
  ∀ m : ℝ, (∀ x : ℝ, x ≠ 5 ∧ x ≠ 8 → (x - 2) / (x - 5) ≠ (x - m) / (x - 8)) ↔ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_equals_five_l1213_121363


namespace NUMINAMATH_CALUDE_problem_statement_l1213_121389

theorem problem_statement (a b : ℝ) (h : 2 * a^2 - 3 * b + 5 = 0) :
  9 * b - 6 * a^2 + 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1213_121389


namespace NUMINAMATH_CALUDE_christine_distance_l1213_121321

theorem christine_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 20 → time = 4 → distance = speed * time → distance = 80 := by
  sorry

end NUMINAMATH_CALUDE_christine_distance_l1213_121321


namespace NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l1213_121377

/-- The coefficient of x in the expansion of (1 + √x)^6 * (1 + √x)^4 -/
def coefficient_of_x : ℕ := 45

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_of_x_in_expansion :
  coefficient_of_x = 
    binomial 4 2 + binomial 6 2 + binomial 6 1 * binomial 4 1 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_in_expansion_l1213_121377


namespace NUMINAMATH_CALUDE_g_50_equals_zero_l1213_121352

-- Define φ(n) as the number of positive integers not exceeding n that are coprime to n
def phi (n : ℕ) : ℕ := sorry

-- Define g(n) that satisfies the given condition
def g (n : ℕ) : ℤ := sorry

-- Define the sum of g(d) over all positive divisors d of n
def sum_g_divisors (n : ℕ) : ℤ := sorry

-- State the condition that g(n) satisfies
axiom g_condition (n : ℕ) : sum_g_divisors n = phi n

-- Theorem to prove
theorem g_50_equals_zero : g 50 = 0 := by sorry

end NUMINAMATH_CALUDE_g_50_equals_zero_l1213_121352


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l1213_121368

theorem arithmetic_mean_of_four_numbers :
  let numbers : List ℝ := [17, 29, 41, 53]
  (numbers.sum / numbers.length : ℝ) = 35 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l1213_121368


namespace NUMINAMATH_CALUDE_extremum_condition_l1213_121336

def f (a b x : ℝ) := x^3 - a*x^2 - b*x + a^2

theorem extremum_condition (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f a b x ≤ f a b 1) ∧
  f a b 1 = 10 →
  a + b = 7 := by sorry

end NUMINAMATH_CALUDE_extremum_condition_l1213_121336


namespace NUMINAMATH_CALUDE_max_robot_A_l1213_121330

def robot_problem (transport_rate_A transport_rate_B : ℕ) 
                  (price_A price_B total_budget : ℕ) 
                  (total_robots : ℕ) : Prop :=
  (transport_rate_A = transport_rate_B + 30) ∧
  (1500 / transport_rate_A = 1000 / transport_rate_B) ∧
  (price_A = 50000) ∧
  (price_B = 30000) ∧
  (total_robots = 12) ∧
  (total_budget = 450000)

theorem max_robot_A (transport_rate_A transport_rate_B : ℕ) 
                    (price_A price_B total_budget : ℕ) 
                    (total_robots : ℕ) :
  robot_problem transport_rate_A transport_rate_B price_A price_B total_budget total_robots →
  ∀ m : ℕ, m ≤ total_robots ∧ 
           price_A * m + price_B * (total_robots - m) ≤ total_budget →
  m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_robot_A_l1213_121330


namespace NUMINAMATH_CALUDE_percentage_decrease_of_b_l1213_121305

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) : 
  a > 0 →
  b > 0 →
  a / b = 4 / 5 →
  x = a * (1 + 0.25) →
  m = b * (1 - p / 100) →
  m / x = 0.8 →
  p = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_decrease_of_b_l1213_121305


namespace NUMINAMATH_CALUDE_circle_equation_radius_l1213_121371

/-- Given a circle with equation x^2 - 8x + y^2 + 10y + d = 0 and radius 5, prove that d = 16 -/
theorem circle_equation_radius (d : ℝ) : 
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + d = 0 → (x - 4)^2 + (y + 5)^2 = 5^2) → 
  d = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l1213_121371


namespace NUMINAMATH_CALUDE_selection_problem_l1213_121332

theorem selection_problem (n_boys m_boys n_girls m_girls : ℕ) 
  (h1 : n_boys = 5) (h2 : m_boys = 3) (h3 : n_girls = 4) (h4 : m_girls = 2) : 
  (Nat.choose n_boys m_boys) * (Nat.choose n_girls m_girls) = 
  (Nat.choose 5 3) * (Nat.choose 4 2) := by
  sorry

end NUMINAMATH_CALUDE_selection_problem_l1213_121332


namespace NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l1213_121339

open Complex

theorem sum_of_complex_roots_of_unity : 
  let ω : ℂ := exp (Complex.I * Real.pi / 11)
  (ω + ω^3 + ω^5 + ω^7 + ω^9 + ω^11 + ω^13 + ω^15 + ω^17 + ω^19 + ω^21) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_roots_of_unity_l1213_121339


namespace NUMINAMATH_CALUDE_sum_of_s_and_t_l1213_121338

theorem sum_of_s_and_t (s t : ℕ+) (h : s * (s - t) = 29) : s + t = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_s_and_t_l1213_121338


namespace NUMINAMATH_CALUDE_point_on_line_equidistant_from_axes_in_first_quadrant_l1213_121362

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 4 * x + 3 * y = 12

-- Define the condition for a point being equidistant from coordinate axes
def equidistant_from_axes (x y : ℝ) : Prop := |x| = |y|

-- Define the condition for a point being in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem point_on_line_equidistant_from_axes_in_first_quadrant :
  ∃ (x y : ℝ), line_equation x y ∧ equidistant_from_axes x y ∧ in_first_quadrant x y ∧
  (∀ (x' y' : ℝ), line_equation x' y' ∧ equidistant_from_axes x' y' → in_first_quadrant x' y') :=
sorry

end NUMINAMATH_CALUDE_point_on_line_equidistant_from_axes_in_first_quadrant_l1213_121362


namespace NUMINAMATH_CALUDE_complement_of_A_l1213_121390

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the set of non-negative real numbers
def A : Set ℝ := {x : ℝ | x ≥ 0}

-- State the theorem
theorem complement_of_A : Set.compl A = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1213_121390


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l1213_121302

/-- Given a function f(x) = -x² + x and two points on its graph,
    prove that Δy/Δx = 3 - Δx -/
theorem delta_y_over_delta_x (f : ℝ → ℝ) (Δx Δy : ℝ) :
  (∀ x, f x = -x^2 + x) →
  f (-1) = -2 →
  f (-1 + Δx) = -2 + Δy →
  Δx ≠ 0 →
  Δy / Δx = 3 - Δx :=
by sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l1213_121302


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l1213_121387

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (outer1 outer2 shaded : Rectangle) 
  (h1 : outer1.width = 12 ∧ outer1.height = 9)
  (h2 : outer2.width = 5 ∧ outer2.height = 3)
  (h3 : shaded.width = 6 ∧ shaded.height = 3)
  (h4 : area outer1 + area outer2 = 117)
  (h5 : area shaded = 108) :
  ∃ (non_shaded : Rectangle), perimeter non_shaded = 12 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l1213_121387


namespace NUMINAMATH_CALUDE_museum_visitors_l1213_121322

theorem museum_visitors (V : ℕ) 
  (h1 : V = (3/4 : ℚ) * V + 130)
  (h2 : ∃ E U : ℕ, E = U ∧ E = (3/4 : ℚ) * V) : 
  V = 520 := by
  sorry

end NUMINAMATH_CALUDE_museum_visitors_l1213_121322


namespace NUMINAMATH_CALUDE_journey_distance_l1213_121340

/-- A journey with two parts -/
structure Journey where
  total_time : ℝ
  speed1 : ℝ
  time1 : ℝ
  speed2 : ℝ

/-- Calculate the total distance of a journey -/
def total_distance (j : Journey) : ℝ :=
  j.speed1 * j.time1 + j.speed2 * (j.total_time - j.time1)

/-- Theorem: The total distance of the given journey is 240 km -/
theorem journey_distance :
  ∃ (j : Journey),
    j.total_time = 5 ∧
    j.speed1 = 40 ∧
    j.time1 = 3 ∧
    j.speed2 = 60 ∧
    total_distance j = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1213_121340


namespace NUMINAMATH_CALUDE_g_solution_set_m_range_l1213_121393

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x - 8
def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem g_solution_set :
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_g_solution_set_m_range_l1213_121393


namespace NUMINAMATH_CALUDE_alfonso_solution_l1213_121399

/-- Alfonso's weekly earnings and financial goals --/
def alfonso_problem (weekday_earnings : ℕ) (weekend_earnings : ℕ) 
  (total_cost : ℕ) (current_savings : ℕ) (desired_remaining : ℕ) : Prop :=
  let weekly_earnings := 5 * weekday_earnings + 2 * weekend_earnings
  let total_needed := total_cost - current_savings + desired_remaining
  let weeks_needed := (total_needed + weekly_earnings - 1) / weekly_earnings
  weeks_needed = 10

/-- Theorem stating the solution to Alfonso's problem --/
theorem alfonso_solution : 
  alfonso_problem 6 8 460 40 20 := by sorry

end NUMINAMATH_CALUDE_alfonso_solution_l1213_121399


namespace NUMINAMATH_CALUDE_base3_addition_proof_l1213_121350

/-- Represents a single digit in base 3 -/
def Base3Digit := Fin 3

/-- Represents a three-digit number in base 3 -/
def Base3Number := Fin 27

def toBase3 (n : ℕ) : Base3Number :=
  Fin.ofNat (n % 27)

def fromBase3 (n : Base3Number) : ℕ :=
  n.val

def addBase3 (a b c : Base3Number) : Base3Number :=
  toBase3 (fromBase3 a + fromBase3 b + fromBase3 c)

theorem base3_addition_proof (C D : ℕ) 
  (h1 : C < 10 ∧ D < 10)
  (h2 : addBase3 (toBase3 (D * 10 + D)) (toBase3 (3 * 10 + 2)) (toBase3 (C * 100 + 2 * 10 + 4)) = 
        toBase3 (C * 100 + 2 * 10 + 4 + 1)) :
  toBase3 (if D > C then D - C else C - D) = toBase3 1 := by
  sorry

end NUMINAMATH_CALUDE_base3_addition_proof_l1213_121350


namespace NUMINAMATH_CALUDE_jeff_swimming_laps_l1213_121300

/-- The number of laps Jeff swam on Saturday -/
def saturday_laps : ℕ := 27

/-- The number of laps Jeff swam on Sunday morning -/
def sunday_morning_laps : ℕ := 15

/-- The number of laps Jeff had remaining after the break -/
def remaining_laps : ℕ := 56

/-- The total number of laps Jeff's coach required him to swim over the weekend -/
def total_required_laps : ℕ := saturday_laps + sunday_morning_laps + remaining_laps

theorem jeff_swimming_laps : total_required_laps = 98 := by
  sorry

end NUMINAMATH_CALUDE_jeff_swimming_laps_l1213_121300


namespace NUMINAMATH_CALUDE_equal_numbers_after_operations_l1213_121333

theorem equal_numbers_after_operations : ∃ (x a b : ℝ), 
  x > 0 ∧ a > 0 ∧ b > 0 ∧
  96 / a = x ∧
  28 - b = x ∧
  20 + b = x ∧
  6 * a = x := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_after_operations_l1213_121333


namespace NUMINAMATH_CALUDE_AB_length_l1213_121395

-- Define the points
variable (A B C D E F G : ℝ)

-- Define the conditions
axiom C_midpoint : C = (A + B) / 2
axiom D_midpoint : D = (A + C) / 2
axiom E_midpoint : E = (A + D) / 2
axiom F_midpoint : F = (A + E) / 2
axiom G_midpoint : G = (A + F) / 2
axiom AG_length : G - A = 2

-- State the theorem
theorem AB_length : B - A = 64 := by sorry

end NUMINAMATH_CALUDE_AB_length_l1213_121395


namespace NUMINAMATH_CALUDE_can_display_sequence_l1213_121324

/-- 
Given a sequence where:
- The first term is 2
- Each subsequent term increases by 3
- The 9th term is 26
Prove that this sequence exists and satisfies these conditions.
-/
theorem can_display_sequence : 
  ∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n, a (n + 1) = a n + 3) ∧ 
    a 9 = 26 := by
  sorry

end NUMINAMATH_CALUDE_can_display_sequence_l1213_121324


namespace NUMINAMATH_CALUDE_triangle_similarity_FC_length_l1213_121344

theorem triangle_similarity_FC_length
  (DC : ℝ) (CB : ℝ) (AD : ℝ) (AB : ℝ) (ED : ℝ) (FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 5)
  (h3 : AB = (1/3) * AD)
  (h4 : ED = (4/5) * AD)
  : FC = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_FC_length_l1213_121344


namespace NUMINAMATH_CALUDE_newer_train_distance_proof_l1213_121331

/-- The distance traveled by the newer train -/
def newer_train_distance (older_train_distance : ℝ) : ℝ :=
  older_train_distance * 1.5

theorem newer_train_distance_proof (older_train_distance : ℝ) 
  (h : older_train_distance = 300) :
  newer_train_distance older_train_distance = 450 := by
  sorry

end NUMINAMATH_CALUDE_newer_train_distance_proof_l1213_121331


namespace NUMINAMATH_CALUDE_pyramid_stack_height_l1213_121397

/-- Represents a stack of square blocks arranged in a stepped pyramid. -/
structure BlockStack where
  blockSideLength : ℝ
  numLayers : ℕ
  blocksPerLayer : ℕ → ℕ

/-- Calculates the total height of a block stack. -/
def totalHeight (stack : BlockStack) : ℝ :=
  stack.blockSideLength * stack.numLayers

/-- Theorem: The total height of a specific stepped pyramid stack is 30 cm. -/
theorem pyramid_stack_height :
  let stack : BlockStack := {
    blockSideLength := 10,
    numLayers := 3,
    blocksPerLayer := fun n => 3 - n + 1
  }
  totalHeight stack = 30 := by sorry

end NUMINAMATH_CALUDE_pyramid_stack_height_l1213_121397


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1213_121396

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log 3) + x^(Real.log 4) = x^(Real.log 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l1213_121396


namespace NUMINAMATH_CALUDE_puppy_sale_revenue_l1213_121345

/-- Calculates the total amount received from selling puppies --/
theorem puppy_sale_revenue (num_dogs : ℕ) (puppies_per_dog : ℕ) (sale_fraction : ℚ) (price_per_puppy : ℕ) : 
  num_dogs = 2 → 
  puppies_per_dog = 10 → 
  sale_fraction = 3/4 → 
  price_per_puppy = 200 → 
  (↑num_dogs * ↑puppies_per_dog : ℚ) * sale_fraction * ↑price_per_puppy = 3000 := by
  sorry

#check puppy_sale_revenue

end NUMINAMATH_CALUDE_puppy_sale_revenue_l1213_121345


namespace NUMINAMATH_CALUDE_fourth_person_height_l1213_121394

/-- Proves that given four people with heights in increasing order, where the difference
    between consecutive heights is 2, 2, and 6 inches respectively, and the average
    height is 77 inches, the height of the fourth person is 83 inches. -/
theorem fourth_person_height
  (h₁ h₂ h₃ h₄ : ℝ)
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_1_2 : h₂ - h₁ = 2)
  (diff_2_3 : h₃ - h₂ = 2)
  (diff_3_4 : h₄ - h₃ = 6)
  (avg_height : (h₁ + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l1213_121394


namespace NUMINAMATH_CALUDE_sum_of_sequences_is_300_l1213_121341

def sequence1 : List ℕ := [2, 13, 24, 35, 46]
def sequence2 : List ℕ := [4, 15, 26, 37, 48]

theorem sum_of_sequences_is_300 : 
  (sequence1.sum + sequence2.sum) = 300 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_is_300_l1213_121341


namespace NUMINAMATH_CALUDE_marble_probability_l1213_121386

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 20) (h2 : blue = 5) (h3 : red = 7) :
  let white := total - (blue + red)
  (red + white : ℚ) / total = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l1213_121386


namespace NUMINAMATH_CALUDE_largest_n_inequality_l1213_121370

theorem largest_n_inequality : 
  (∃ n : ℕ, (1/4 : ℚ) + 2*n/5 < 7/8 ∧ 
   ∀ m : ℕ, (1/4 : ℚ) + 2*m/5 < 7/8 → m ≤ n) → 
  (∃ n : ℕ, (1/4 : ℚ) + 2*n/5 < 7/8 ∧ 
   ∀ m : ℕ, (1/4 : ℚ) + 2*m/5 < 7/8 → m ≤ n) ∧ 
  (∀ n : ℕ, (1/4 : ℚ) + 2*n/5 < 7/8 → n ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_inequality_l1213_121370


namespace NUMINAMATH_CALUDE_point_inside_circle_l1213_121357

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a quadratic equation ax² + bx - c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The statement to be proved -/
theorem point_inside_circle (e : Ellipse) (q : QuadraticEquation) 
  (h_eccentricity : e.a * (1/2) = (q.c^2 + e.b^2)^(1/2))
  (h_equation : q.a = e.a ∧ q.b = e.b ∧ q.c = e.a * (1/2))
  (h_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ q.a * x₁^2 + q.b * x₁ - q.c = 0 ∧ q.a * x₂^2 + q.b * x₂ - q.c = 0) :
  ∃ x₁ x₂ : ℝ, x₁^2 + x₂^2 < 2 :=
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l1213_121357


namespace NUMINAMATH_CALUDE_remaining_denomination_is_500_l1213_121320

/-- Represents the denomination problem with given conditions -/
def DenominationProblem (total_amount : ℕ) (total_notes : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) : Prop :=
  ∃ (other_denom : ℕ),
    total_amount = fifty_notes * fifty_value + (total_notes - fifty_notes) * other_denom ∧
    total_notes > fifty_notes ∧
    other_denom > 0

/-- Theorem stating that the denomination of remaining notes is 500 -/
theorem remaining_denomination_is_500 :
  DenominationProblem 10350 126 117 50 → ∃ (other_denom : ℕ), other_denom = 500 :=
by sorry

end NUMINAMATH_CALUDE_remaining_denomination_is_500_l1213_121320


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1213_121309

theorem quadratic_always_positive (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) → 0 ≤ a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1213_121309


namespace NUMINAMATH_CALUDE_det_dilation_matrix_3d_det_dilation_matrix_3d_scale_5_l1213_121327

def dilation_matrix (scale : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => scale)

theorem det_dilation_matrix_3d (scale : ℝ) :
  Matrix.det (dilation_matrix scale) = scale ^ 3 := by
  sorry

theorem det_dilation_matrix_3d_scale_5 :
  Matrix.det (dilation_matrix 5) = 125 := by
  sorry

end NUMINAMATH_CALUDE_det_dilation_matrix_3d_det_dilation_matrix_3d_scale_5_l1213_121327


namespace NUMINAMATH_CALUDE_profit_percentage_is_40_percent_l1213_121325

/-- The percentage of puppies that can be sold for a greater profit -/
def profitable_puppies_percentage (total_puppies : ℕ) (puppies_with_more_than_4_spots : ℕ) : ℚ :=
  (puppies_with_more_than_4_spots : ℚ) / (total_puppies : ℚ) * 100

/-- Theorem stating that the percentage of puppies that can be sold for a greater profit is 40% -/
theorem profit_percentage_is_40_percent :
  profitable_puppies_percentage 20 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_40_percent_l1213_121325


namespace NUMINAMATH_CALUDE_tim_weekly_fluid_intake_l1213_121318

/-- Calculates Tim's weekly fluid intake in ounces -/
def weekly_fluid_intake : ℝ :=
  let water_bottles_per_day : ℝ := 2
  let water_quarts_per_bottle : ℝ := 1.5
  let orange_juice_oz_per_day : ℝ := 20
  let soda_liters_per_other_day : ℝ := 1.5
  let coffee_cups_per_week : ℝ := 4
  let quart_to_oz : ℝ := 32
  let liter_to_oz : ℝ := 33.814
  let cup_to_oz : ℝ := 8
  let days_per_week : ℝ := 7
  let soda_days_per_week : ℝ := 4

  let water_oz : ℝ := water_bottles_per_day * water_quarts_per_bottle * quart_to_oz * days_per_week
  let orange_juice_oz : ℝ := orange_juice_oz_per_day * days_per_week
  let soda_oz : ℝ := soda_liters_per_other_day * liter_to_oz * soda_days_per_week
  let coffee_oz : ℝ := coffee_cups_per_week * cup_to_oz

  water_oz + orange_juice_oz + soda_oz + coffee_oz

/-- Theorem stating Tim's weekly fluid intake -/
theorem tim_weekly_fluid_intake : weekly_fluid_intake = 1046.884 := by
  sorry

end NUMINAMATH_CALUDE_tim_weekly_fluid_intake_l1213_121318


namespace NUMINAMATH_CALUDE_calculation_proof_l1213_121349

theorem calculation_proof : (12 * 0.5 * 3 * 0.0625 - 1.5) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1213_121349


namespace NUMINAMATH_CALUDE_hybrid_rice_scientific_notation_l1213_121351

theorem hybrid_rice_scientific_notation :
  ∃ n : ℕ, 250000000 = (2.5 : ℝ) * (10 : ℝ) ^ n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_rice_scientific_notation_l1213_121351


namespace NUMINAMATH_CALUDE_empty_square_existence_l1213_121392

/-- Represents a chessboard with rooks -/
structure Chessboard :=
  (size : ℕ)
  (rooks : Finset (ℕ × ℕ))

/-- Defines a valid chessboard configuration -/
def is_valid_configuration (board : Chessboard) : Prop :=
  board.size = 50 ∧ 
  board.rooks.card = 50 ∧
  ∀ (r1 r2 : ℕ × ℕ), r1 ∈ board.rooks → r2 ∈ board.rooks → r1 ≠ r2 → 
    r1.1 ≠ r2.1 ∧ r1.2 ≠ r2.2

/-- Defines an empty square on the chessboard -/
def has_empty_square (board : Chessboard) (k : ℕ) : Prop :=
  ∃ (x y : ℕ), x + k ≤ board.size ∧ y + k ≤ board.size ∧
    ∀ (i j : ℕ), i < k → j < k → (x + i, y + j) ∉ board.rooks

/-- The main theorem -/
theorem empty_square_existence (board : Chessboard) (h : is_valid_configuration board) :
  ∀ k : ℕ, (k ≤ 7 ↔ ∀ (board : Chessboard), is_valid_configuration board → has_empty_square board k) :=
sorry

end NUMINAMATH_CALUDE_empty_square_existence_l1213_121392


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l1213_121388

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (rowing_speed : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (h1 : distance_to_shore = 2)
  (h2 : rowing_speed = 3)
  (h3 : water_intake_rate = 6)
  (h4 : boat_capacity = 60) :
  ∃ (min_bailing_rate : ℝ),
    min_bailing_rate = 4.5 ∧
    ∀ (bailing_rate : ℝ),
      bailing_rate ≥ min_bailing_rate →
      (water_intake_rate - bailing_rate) * (distance_to_shore / rowing_speed * 60) ≤ boat_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l1213_121388


namespace NUMINAMATH_CALUDE_tan_equality_solution_l1213_121380

theorem tan_equality_solution (n : ℤ) (h1 : -180 < n) (h2 : n < 180) 
  (h3 : Real.tan (n * π / 180) = Real.tan (123 * π / 180)) : 
  n = 123 ∨ n = -57 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l1213_121380


namespace NUMINAMATH_CALUDE_parabola_point_k_value_l1213_121301

/-- Given that the point (3,0) lies on the parabola y = 2x^2 + (k+2)x - k, prove that k = -12 -/
theorem parabola_point_k_value :
  ∀ k : ℝ, (2 * 3^2 + (k + 2) * 3 - k = 0) → k = -12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_k_value_l1213_121301


namespace NUMINAMATH_CALUDE_card_combination_proof_l1213_121361

theorem card_combination_proof : Nat.choose 60 13 = 7446680748480 := by
  sorry

end NUMINAMATH_CALUDE_card_combination_proof_l1213_121361


namespace NUMINAMATH_CALUDE_sugar_percentage_approx_l1213_121398

-- Define the initial solution volume
def initial_volume : ℝ := 500

-- Define the initial composition percentages
def water_percent : ℝ := 0.60
def cola_percent : ℝ := 0.08
def orange_percent : ℝ := 0.10
def lemon_percent : ℝ := 0.12

-- Define the added components
def added_sugar : ℝ := 4
def added_water : ℝ := 15
def added_cola : ℝ := 9
def added_orange : ℝ := 5
def added_lemon : ℝ := 7
def added_ice : ℝ := 8

-- Calculate the new total volume
def new_volume : ℝ := initial_volume + added_sugar + added_water + added_cola + added_orange + added_lemon + added_ice

-- Define the theorem
theorem sugar_percentage_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |added_sugar / new_volume - 0.0073| < ε :=
sorry

end NUMINAMATH_CALUDE_sugar_percentage_approx_l1213_121398


namespace NUMINAMATH_CALUDE_no_ab_term_l1213_121323

/-- The polynomial does not contain the term ab if and only if m = -2 -/
theorem no_ab_term (a b m : ℝ) : 
  2 * (a^2 + a*b - 5*b^2) - (a^2 - m*a*b + 2*b^2) = a^2 - 12*b^2 ↔ m = -2 :=
by sorry

end NUMINAMATH_CALUDE_no_ab_term_l1213_121323


namespace NUMINAMATH_CALUDE_quadratic_roots_arithmetic_sequence_l1213_121375

theorem quadratic_roots_arithmetic_sequence (a b : ℚ) : 
  a ≠ b →
  (∃ x₁ x₂ x₃ x₄ : ℚ, 
    (x₁^2 - x₁ + a = 0 ∧ x₂^2 - x₂ + a = 0) ∧ 
    (x₃^2 - x₃ + b = 0 ∧ x₄^2 - x₄ + b = 0) ∧
    (∃ d : ℚ, x₁ = 1/4 ∧ x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d)) →
  a + b = 31/72 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_arithmetic_sequence_l1213_121375


namespace NUMINAMATH_CALUDE_function_properties_l1213_121329

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log (x + b)) / x

noncomputable def g (a x : ℝ) : ℝ := x + 2 / x - a - 2

noncomputable def F (a b x : ℝ) : ℝ := f a b x + g a x

theorem function_properties (a b : ℝ) (ha : a ≤ 2) (ha_nonzero : a ≠ 0) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = f a b x) →
  (∃ m : ℝ, ∀ x : ℝ, f a b x - f a b 1 = m * (x - 1) → f a b 3 = 0) →
  (∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ F a b x = 0) →
  (b = 2 * a ∧ (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2))) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1213_121329


namespace NUMINAMATH_CALUDE_lucas_investment_l1213_121364

theorem lucas_investment (total_investment : ℝ) (alpha_rate beta_rate : ℝ) (final_amount : ℝ)
  (h1 : total_investment = 1500)
  (h2 : alpha_rate = 0.04)
  (h3 : beta_rate = 0.06)
  (h4 : final_amount = 1584.50) :
  ∃ (alpha_investment : ℝ),
    alpha_investment * (1 + alpha_rate) + (total_investment - alpha_investment) * (1 + beta_rate) = final_amount ∧
    alpha_investment = 275 :=
by sorry

end NUMINAMATH_CALUDE_lucas_investment_l1213_121364


namespace NUMINAMATH_CALUDE_line_equation_proof_l1213_121313

/-- Given two lines in the 2D plane, we define them as parallel if they have the same slope. -/
def parallel_lines (m1 b1 m2 b2 : ℝ) : Prop :=
  m1 = m2

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line given by y = mx + b -/
def point_on_line (p : Point) (m b : ℝ) : Prop :=
  p.y = m * p.x + b

theorem line_equation_proof :
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x - y + 3 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => 2 * x - y - 8 = 0
  let point_A : Point := ⟨2, -4⟩
  parallel_lines 2 3 2 (-8) ∧ point_on_line point_A 2 (-8) :=
by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1213_121313


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l1213_121385

theorem max_value_sin_cos (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  (∀ φ, 0 < φ ∧ φ < Real.pi → 
    Real.sin (φ / 2) * (1 + Real.cos φ) ≤ Real.sin (θ / 2) * (1 + Real.cos θ)) ↔ 
  Real.sin (θ / 2) * (1 + Real.cos θ) = 4 * Real.sqrt 3 / 9 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l1213_121385


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1213_121373

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  3 * x^2 - 12 * x * y + 12 * y^2 = 3 * (x - 2 * y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l1213_121373


namespace NUMINAMATH_CALUDE_only_prime_three_squared_plus_eight_prime_l1213_121308

theorem only_prime_three_squared_plus_eight_prime :
  ∀ p : ℕ, Prime p ∧ Prime (p^2 + 8) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_only_prime_three_squared_plus_eight_prime_l1213_121308


namespace NUMINAMATH_CALUDE_sum_first_last_33_l1213_121307

/-- A sequence of ten terms -/
def Sequence := Fin 10 → ℕ

/-- The property that C (the third term) is 7 -/
def third_is_seven (s : Sequence) : Prop := s 2 = 7

/-- The property that the sum of any three consecutive terms is 40 -/
def consecutive_sum_40 (s : Sequence) : Prop :=
  ∀ i, i < 8 → s i + s (i + 1) + s (i + 2) = 40

/-- The main theorem: If C is 7 and the sum of any three consecutive terms is 40,
    then A + J = 33 -/
theorem sum_first_last_33 (s : Sequence) 
  (h1 : third_is_seven s) (h2 : consecutive_sum_40 s) : s 0 + s 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_last_33_l1213_121307


namespace NUMINAMATH_CALUDE_only_set_C_is_right_triangle_l1213_121391

-- Define a function to check if three numbers can form a right triangle
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def set_A : List ℕ := [2, 3, 3]
def set_B : List ℕ := [4, 5, 6]
def set_C : List ℕ := [5, 12, 13]
def set_D : List ℕ := [7, 7, 7]

-- State the theorem
theorem only_set_C_is_right_triangle :
  (¬ is_right_triangle set_A[0] set_A[1] set_A[2]) ∧
  (¬ is_right_triangle set_B[0] set_B[1] set_B[2]) ∧
  (is_right_triangle set_C[0] set_C[1] set_C[2]) ∧
  (¬ is_right_triangle set_D[0] set_D[1] set_D[2]) :=
sorry

end NUMINAMATH_CALUDE_only_set_C_is_right_triangle_l1213_121391


namespace NUMINAMATH_CALUDE_cats_sold_l1213_121342

theorem cats_sold (ratio : ℚ) (dogs : ℕ) (cats : ℕ) : 
  ratio = 2 / 1 → dogs = 8 → cats = 16 := by
  sorry

end NUMINAMATH_CALUDE_cats_sold_l1213_121342


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l1213_121348

theorem rightmost_three_digits_of_7_to_1993 : 7^1993 % 1000 = 407 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l1213_121348


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l1213_121311

theorem average_of_four_numbers (r s t u : ℝ) 
  (h : (5 / 4) * (r + s + t + u) = 15) : 
  (r + s + t + u) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l1213_121311


namespace NUMINAMATH_CALUDE_difference_of_two_numbers_l1213_121356

theorem difference_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (product_eq : x * y = 140) : 
  |x - y| = Real.sqrt 340 := by sorry

end NUMINAMATH_CALUDE_difference_of_two_numbers_l1213_121356


namespace NUMINAMATH_CALUDE_equation_solutions_l1213_121346

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3 ∧ 
    x₁^2 - 2*x₁ - 2 = 0 ∧ x₂^2 - 2*x₂ - 2 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3/2 ∧ y₂ = 7/2 ∧ 
    2*(y₁ - 3)^2 = y₁ - 3 ∧ 2*(y₂ - 3)^2 = y₂ - 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1213_121346


namespace NUMINAMATH_CALUDE_inequality_proof_l1213_121378

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1213_121378


namespace NUMINAMATH_CALUDE_community_center_chairs_l1213_121382

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : Nat) : Nat :=
  (n / 100) * 25 + ((n % 100) / 10) * 5 + (n % 10)

/-- Calculates the number of chairs needed given a capacity in base 5 and people per chair -/
def chairsNeeded (capacityBase5 : Nat) (peoplePerChair : Nat) : Nat :=
  (base5ToBase10 capacityBase5) / peoplePerChair

theorem community_center_chairs :
  chairsNeeded 310 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_community_center_chairs_l1213_121382


namespace NUMINAMATH_CALUDE_line_through_points_l1213_121374

/-- A line passing through two points (1,2) and (5,14) has equation y = ax + b. This theorem proves that a - b = 4. -/
theorem line_through_points (a b : ℝ) : 
  (2 = a * 1 + b) → (14 = a * 5 + b) → a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l1213_121374


namespace NUMINAMATH_CALUDE_following_pierre_better_than_guessing_l1213_121376

-- Define the probability of Pierre giving correct information
def pierre_correct_prob : ℚ := 3/4

-- Define the probability of Pierre giving incorrect information
def pierre_incorrect_prob : ℚ := 1/4

-- Define the probability of Jean guessing correctly for one event
def jean_guess_prob : ℚ := 1/2

-- Define the probability of Jean getting both dates correct when following Pierre's advice
def jean_correct_following_pierre : ℚ :=
  pierre_correct_prob * (pierre_correct_prob * pierre_correct_prob) +
  pierre_incorrect_prob * (pierre_incorrect_prob * pierre_incorrect_prob)

-- Define the probability of Jean getting both dates correct when guessing randomly
def jean_correct_guessing : ℚ := jean_guess_prob * jean_guess_prob

-- Theorem stating that following Pierre's advice is better than guessing randomly
theorem following_pierre_better_than_guessing :
  jean_correct_following_pierre > jean_correct_guessing :=
by sorry

end NUMINAMATH_CALUDE_following_pierre_better_than_guessing_l1213_121376


namespace NUMINAMATH_CALUDE_tan_function_property_l1213_121312

noncomputable def f (a b x : ℝ) : ℝ := a * Real.tan (b * x)

theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b (x + π/5) = f a b x) →
  f a b (5*π/24) = 5 →
  a * b = 25 / Real.tan (π/24) := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l1213_121312


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1213_121343

theorem sum_of_three_numbers : 1.35 + 0.123 + 0.321 = 1.794 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1213_121343


namespace NUMINAMATH_CALUDE_area_is_zero_l1213_121367

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop :=
  x^2 - 10*x + 5*y + 50 = 25 + 15*y - y^2

-- Define the line
def line (x : ℝ) : ℝ := x - 4

-- Define the region above the line
def region_above_line (x y : ℝ) : Prop :=
  y > line x

-- Define the area of the region
def area_of_region : ℝ := 0

-- Theorem statement
theorem area_is_zero :
  area_of_region = 0 :=
sorry

end NUMINAMATH_CALUDE_area_is_zero_l1213_121367


namespace NUMINAMATH_CALUDE_group_size_proof_l1213_121317

/-- The number of men in a group where:
    1) The average age increases by 1 year
    2) Two men aged 21 and 23 are replaced by two men with an average age of 32 -/
def number_of_men : ℕ := 20

theorem group_size_proof :
  let original_average : ℝ := number_of_men
  let new_average : ℝ := original_average + 1
  let replaced_sum : ℝ := 21 + 23
  let new_sum : ℝ := 2 * 32
  number_of_men * original_average + new_sum - replaced_sum = number_of_men * new_average :=
by sorry

end NUMINAMATH_CALUDE_group_size_proof_l1213_121317


namespace NUMINAMATH_CALUDE_triangle_problem_l1213_121334

theorem triangle_problem (A B C a b c p : ℝ) :
  -- Triangle ABC with angles A, B, C corresponding to sides a, b, c
  (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0) → (b > 0) → (c > 0) →
  -- Given conditions
  (Real.sin A + Real.sin C = p * Real.sin B) →
  (a * c = (1/4) * b^2) →
  -- Part I
  (p = 5/4 ∧ b = 1) →
  ((a = 1 ∧ c = 1/4) ∨ (a = 1/4 ∧ c = 1)) ∧
  -- Part II
  (0 < B ∧ B < π/2) →
  (Real.sqrt 6 / 2 < p ∧ p < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1213_121334


namespace NUMINAMATH_CALUDE_counterexample_exists_l1213_121366

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ a*b := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1213_121366


namespace NUMINAMATH_CALUDE_f_minimum_value_l1213_121358

def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem f_minimum_value :
  (∀ x y : ℝ, f x y ≥ 21.2) ∧ f 8 1 = 21.2 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1213_121358


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l1213_121360

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 →
  (1 - x) * 0.75 * P * 1.5686274509803921 = P →
  x = 0.15 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l1213_121360


namespace NUMINAMATH_CALUDE_slope_of_MN_constant_sum_of_reciprocals_l1213_121355

/- Ellipse C₁ -/
def C₁ (b : ℝ) (x y : ℝ) : Prop := x^2/8 + y^2/b^2 = 1 ∧ b > 0

/- Parabola C₂ -/
def C₂ (x y : ℝ) : Prop := y^2 = 8*x

/- Right focus F₂ -/
def F₂ : ℝ × ℝ := (2, 0)

/- Theorem for the slope of line MN -/
theorem slope_of_MN (b : ℝ) (M N : ℝ × ℝ) :
  C₁ b M.1 M.2 → C₁ b N.1 N.2 → ((M.1 + N.1)/2, (M.2 + N.2)/2) = (1, 1) →
  (N.2 - M.2) / (N.1 - M.1) = -1/2 :=
sorry

/- Theorem for the constant sum of reciprocals -/
theorem constant_sum_of_reciprocals (b : ℝ) (A B C D : ℝ × ℝ) (m n : ℝ) :
  C₁ b A.1 A.2 → C₁ b B.1 B.2 → C₁ b C.1 C.2 → C₁ b D.1 D.2 →
  ((A.1 - F₂.1) * (B.1 - F₂.1) + (A.2 - F₂.2) * (B.2 - F₂.2) = 0) →
  ((C.1 - F₂.1) * (D.1 - F₂.1) + (C.2 - F₂.2) * (D.2 - F₂.2) = 0) →
  m = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) →
  n = Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) →
  1/m + 1/n = 3 * Real.sqrt 2 / 8 :=
sorry

end NUMINAMATH_CALUDE_slope_of_MN_constant_sum_of_reciprocals_l1213_121355


namespace NUMINAMATH_CALUDE_min_value_of_squared_differences_l1213_121381

theorem min_value_of_squared_differences (a α β : ℝ) : 
  (α^2 - 2*a*α + a + 6 = 0) →
  (β^2 - 2*a*β + a + 6 = 0) →
  α ≠ β →
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 
    (x^2 - 2*a*x + a + 6 = 0) → 
    (y^2 - 2*a*y + a + 6 = 0) → 
    x ≠ y → 
    (x - 1)^2 + (y - 1)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_squared_differences_l1213_121381


namespace NUMINAMATH_CALUDE_intersection_length_theorem_l1213_121304

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 - y^2/3 = 1 ∧ x < -1

-- Define a line through F₁
def line_through_F₁ (m : ℝ) (x y : ℝ) : Prop := x = m * y - 2

-- Theorem statement
theorem intersection_length_theorem 
  (A B P Q : ℝ × ℝ) 
  (m : ℝ) 
  (h₁ : C A.1 A.2) 
  (h₂ : C B.1 B.2) 
  (h₃ : F₂ P.1 P.2) 
  (h₄ : F₂ Q.1 Q.2) 
  (h₅ : line_through_F₁ m A.1 A.2) 
  (h₆ : line_through_F₁ m B.1 B.2) 
  (h₇ : line_through_F₁ m P.1 P.2) 
  (h₈ : line_through_F₁ m Q.1 Q.2) 
  (h₉ : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 2) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_length_theorem_l1213_121304


namespace NUMINAMATH_CALUDE_march_first_is_monday_l1213_121384

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => daysBefore (match d with
    | DayOfWeek.Sunday => DayOfWeek.Saturday
    | DayOfWeek.Monday => DayOfWeek.Sunday
    | DayOfWeek.Tuesday => DayOfWeek.Monday
    | DayOfWeek.Wednesday => DayOfWeek.Tuesday
    | DayOfWeek.Thursday => DayOfWeek.Wednesday
    | DayOfWeek.Friday => DayOfWeek.Thursday
    | DayOfWeek.Saturday => DayOfWeek.Friday) n

theorem march_first_is_monday (march13 : DayOfWeek) 
    (h : march13 = DayOfWeek.Saturday) : 
    daysBefore march13 12 = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_march_first_is_monday_l1213_121384
