import Mathlib

namespace NUMINAMATH_CALUDE_profit_maximization_l3743_374370

/-- Represents the sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -x + 40

/-- Represents the profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 10) * (sales_volume x)

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 25

/-- The maximum profit achieved at the optimal price -/
def max_profit : ℝ := 225

theorem profit_maximization :
  (∀ x : ℝ, profit x ≤ profit optimal_price) ∧
  profit optimal_price = max_profit :=
sorry

end NUMINAMATH_CALUDE_profit_maximization_l3743_374370


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3743_374384

/-- A circle passing through three given points -/
def CircleThroughPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + p.2^2 - 7*p.1 - 3*p.2 + 2) = 0}

/-- Theorem stating that the circle passes through the given points -/
theorem circle_passes_through_points :
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (4, -2)
  A ∈ CircleThroughPoints A B C ∧
  B ∈ CircleThroughPoints A B C ∧
  C ∈ CircleThroughPoints A B C := by
  sorry

#check circle_passes_through_points

end NUMINAMATH_CALUDE_circle_passes_through_points_l3743_374384


namespace NUMINAMATH_CALUDE_notebook_distribution_l3743_374314

theorem notebook_distribution (S : ℕ) : 
  (S / 8 : ℚ) = 16 → S * (S / 8 : ℚ) = 2048 := by
  sorry

end NUMINAMATH_CALUDE_notebook_distribution_l3743_374314


namespace NUMINAMATH_CALUDE_certain_number_problem_l3743_374365

theorem certain_number_problem :
  ∃! x : ℝ, ((7 * (x + 10)) / 5) - 5 = 88 / 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3743_374365


namespace NUMINAMATH_CALUDE_leader_secretary_selection_count_l3743_374381

/-- The number of ways to select a team leader and a secretary from a team of 5 members -/
def select_leader_and_secretary (team_size : ℕ) : ℕ :=
  team_size * (team_size - 1)

/-- Theorem: The number of ways to select a team leader and a secretary from a team of 5 members is 20 -/
theorem leader_secretary_selection_count :
  select_leader_and_secretary 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_leader_secretary_selection_count_l3743_374381


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3743_374338

theorem division_remainder_problem (L S R : ℝ) : 
  L - S = 1356 →
  S = 268.2 →
  L = 6 * S + R →
  R = 15 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3743_374338


namespace NUMINAMATH_CALUDE_max_value_theorem_l3743_374346

theorem max_value_theorem (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) : 
  (a * b) / (2 * (a + b)) + (a * c) / (2 * (a + c)) + (b * c) / (2 * (b + c)) ≤ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3743_374346


namespace NUMINAMATH_CALUDE_rectangle_area_error_percentage_l3743_374368

/-- Given a rectangle with actual length L and width W, if the measured length is 1.06L
    and the measured width is 0.95W, then the error percentage in the calculated area is 0.7%. -/
theorem rectangle_area_error_percentage (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 0.7 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percentage_l3743_374368


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3743_374362

/-- The arithmetic square root of a non-negative real number -/
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

/-- The arithmetic square root is non-negative -/
axiom arithmetic_sqrt_nonneg (x : ℝ) : x ≥ 0 → arithmetic_sqrt x ≥ 0

/-- The arithmetic square root of 9 is 3 -/
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l3743_374362


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3743_374349

theorem sum_of_reciprocals (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (sum_squares_condition : a^2 + b^2 + c^2 = 10)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 36) :
  1/a + 1/b + 1/c = 13/18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3743_374349


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_in_8_years_l3743_374344

/-- 
Given a sum of money that doubles itself in 8 years at simple interest,
this theorem proves that the interest rate is 12.5% per annum.
-/
theorem simple_interest_rate_for_doubling_in_8_years : 
  ∀ (P : ℝ), P > 0 → 
  ∃ (R : ℝ), 
    (P + (P * R * 8) / 100 = 2 * P) ∧ 
    R = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_in_8_years_l3743_374344


namespace NUMINAMATH_CALUDE_system_solution_l3743_374393

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop :=
  7 / (2 * x - 3) - 2 / (10 * z - 3 * y) + 3 / (3 * y - 8 * z) = 8

def equation2 (x y z : ℝ) : Prop :=
  2 / (2 * x - 3 * y) - 3 / (10 * z - 3 * y) + 1 / (3 * y - 8 * z) = 0

def equation3 (x y z : ℝ) : Prop :=
  5 / (2 * x - 3 * y) - 4 / (10 * z - 3 * y) + 7 / (3 * y - 8 * z) = 8

-- Define the solution
def solution : ℝ × ℝ × ℝ := (5, 3, 1)

-- Theorem statement
theorem system_solution :
  ∀ x y z : ℝ,
  2 * x ≠ 3 * y →
  10 * z ≠ 3 * y →
  8 * z ≠ 3 * y →
  equation1 x y z ∧ equation2 x y z ∧ equation3 x y z →
  (x, y, z) = solution :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3743_374393


namespace NUMINAMATH_CALUDE_total_pens_bought_l3743_374300

theorem total_pens_bought (pen_cost : ℕ) (masha_spent : ℕ) (olya_spent : ℕ) : 
  pen_cost > 10 ∧ 
  masha_spent = 357 ∧ 
  olya_spent = 441 ∧
  masha_spent % pen_cost = 0 ∧ 
  olya_spent % pen_cost = 0 →
  masha_spent / pen_cost + olya_spent / pen_cost = 38 := by
sorry

end NUMINAMATH_CALUDE_total_pens_bought_l3743_374300


namespace NUMINAMATH_CALUDE_sequence_problem_l3743_374377

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define a geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom : is_geometric_sequence b)
  (h_eq : 2 * a 5 - (a 8)^2 + 2 * a 11 = 0)
  (h_b8 : b 8 = a 8) :
  b 7 * b 9 = 4 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l3743_374377


namespace NUMINAMATH_CALUDE_diagonal_angle_in_rectangular_parallelepiped_l3743_374388

/-- Given a rectangular parallelepiped with two non-intersecting diagonals of adjacent faces
    inclined at angles α and β to the plane of the base, the angle γ between these diagonals
    is equal to arccos(sin α * sin β). -/
theorem diagonal_angle_in_rectangular_parallelepiped
  (α β : Real)
  (h_α : 0 < α ∧ α < π / 2)
  (h_β : 0 < β ∧ β < π / 2) :
  ∃ γ : Real, γ = Real.arccos (Real.sin α * Real.sin β) ∧
    0 ≤ γ ∧ γ ≤ π := by
  sorry

end NUMINAMATH_CALUDE_diagonal_angle_in_rectangular_parallelepiped_l3743_374388


namespace NUMINAMATH_CALUDE_annular_area_l3743_374331

/-- The area of an annular region formed by two concentric circles -/
theorem annular_area (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 10) :
  π * r₂^2 - π * r₁^2 = 84 * π := by
  sorry

end NUMINAMATH_CALUDE_annular_area_l3743_374331


namespace NUMINAMATH_CALUDE_right_triangle_area_floor_l3743_374342

theorem right_triangle_area_floor (perimeter : ℝ) (inscribed_circle_area : ℝ) : 
  perimeter = 2008 →
  inscribed_circle_area = 100 * Real.pi ^ 3 →
  ⌊(perimeter / 2) * (inscribed_circle_area / Real.pi) ^ (1/2)⌋ = 31541 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_floor_l3743_374342


namespace NUMINAMATH_CALUDE_cube_edge_15cm_l3743_374339

/-- The edge length of a cube that displaces a specific volume of water -/
def cube_edge_length (base_length : ℝ) (base_width : ℝ) (water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with the given properties has an edge length of 15 cm -/
theorem cube_edge_15cm :
  cube_edge_length 20 15 11.25 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_15cm_l3743_374339


namespace NUMINAMATH_CALUDE_smallest_k_and_exponent_l3743_374306

theorem smallest_k_and_exponent (k : ℕ) (h : k = 7) :
  64^k > 4^20 ∧ 64^k ≤ 4^21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_and_exponent_l3743_374306


namespace NUMINAMATH_CALUDE_ngon_construction_l3743_374325

/-- A line in 2D space -/
structure Line where
  -- Define a line using two points
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- An n-gon in 2D space -/
structure Polygon where
  -- List of vertices
  vertices : List (ℝ × ℝ)

/-- Function to check if a line is a perpendicular bisector of a polygon side -/
def isPerpBisector (l : Line) (p : Polygon) : Prop :=
  sorry

/-- Function to check if a line is an angle bisector of a polygon vertex -/
def isAngleBisector (l : Line) (p : Polygon) : Prop :=
  sorry

/-- Main theorem: Given n lines, there exists an n-gon such that these lines
    are either perpendicular bisectors of its sides or angle bisectors -/
theorem ngon_construction (n : ℕ) (lines : List Line) :
  (lines.length = n) →
  ∃ (p : Polygon),
    (p.vertices.length = n) ∧
    (∀ l ∈ lines, isPerpBisector l p ∨ isAngleBisector l p) :=
by sorry

end NUMINAMATH_CALUDE_ngon_construction_l3743_374325


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l3743_374327

def num_legs : ℕ := 10

def total_items : ℕ := 2 * num_legs

def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_legs)

theorem centipede_sock_shoe_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l3743_374327


namespace NUMINAMATH_CALUDE_power_sum_equals_three_l3743_374383

theorem power_sum_equals_three (a b c : ℝ) 
  (sum_condition : a + b + c = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3) :
  a^2008 + b^2008 + c^2008 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_three_l3743_374383


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3743_374315

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^351 * k = 15^702 - 6^351) ∧ 
  (∀ m : ℕ, m > 351 → ¬(∃ k : ℕ, 2^m * k = 15^702 - 6^351)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3743_374315


namespace NUMINAMATH_CALUDE_max_m_value_l3743_374363

theorem max_m_value (m : ℕ+) 
  (h : ∃ (k : ℕ), (m.val ^ 4 + 16 * m.val + 8 : ℕ) = (k * (k + 1))) : 
  m ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_m_value_l3743_374363


namespace NUMINAMATH_CALUDE_smallest_bob_number_l3743_374352

def alice_number : ℕ := 30

def has_all_prime_factors (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ a → p ∣ b)

theorem smallest_bob_number (bob_number : ℕ) 
  (h1 : has_all_prime_factors alice_number bob_number)
  (h2 : 5 ∣ bob_number) :
  bob_number ≥ 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l3743_374352


namespace NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l3743_374313

-- Define the universe of items
variable (Item : Type)

-- Define the properties
variable (not_cheap : Item → Prop)
variable (good_quality : Item → Prop)

-- Define the "You get what you pay for" principle
variable (you_get_what_you_pay_for : ∀ (x : Item), good_quality x → ¬(not_cheap x) → False)

-- Theorem: "not cheap" is a necessary condition for "good quality"
theorem not_cheap_necessary_for_good_quality :
  ∀ (x : Item), good_quality x → not_cheap x :=
by
  sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l3743_374313


namespace NUMINAMATH_CALUDE_problem_statement_l3743_374305

theorem problem_statement : 3^(1 + Real.log 2 / Real.log 3) + Real.log 5 + (Real.log 2 / Real.log 3) * (Real.log 3 / Real.log 2) * Real.log 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3743_374305


namespace NUMINAMATH_CALUDE_shaded_area_regular_octagon_l3743_374347

/-- The area of the shaded region in a regular octagon --/
theorem shaded_area_regular_octagon (s : ℝ) (h : s = 8) :
  let R := s / (2 * Real.sin (π / 8))
  let shaded_area := (R / 2) ^ 2
  shaded_area = 32 + 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_regular_octagon_l3743_374347


namespace NUMINAMATH_CALUDE_airline_rows_calculation_l3743_374351

/-- Represents an airline company with a fleet of airplanes -/
structure AirlineCompany where
  num_airplanes : ℕ
  rows_per_airplane : ℕ
  seats_per_row : ℕ
  flights_per_day : ℕ
  total_daily_capacity : ℕ

/-- Theorem: Given the airline company's specifications, prove that each airplane has 20 rows -/
theorem airline_rows_calculation (airline : AirlineCompany)
    (h1 : airline.num_airplanes = 5)
    (h2 : airline.seats_per_row = 7)
    (h3 : airline.flights_per_day = 2)
    (h4 : airline.total_daily_capacity = 1400) :
    airline.rows_per_airplane = 20 := by
  sorry

/-- Example airline company satisfying the given conditions -/
def example_airline : AirlineCompany :=
  { num_airplanes := 5
    rows_per_airplane := 20  -- This is what we're proving
    seats_per_row := 7
    flights_per_day := 2
    total_daily_capacity := 1400 }

end NUMINAMATH_CALUDE_airline_rows_calculation_l3743_374351


namespace NUMINAMATH_CALUDE_sine_intersection_theorem_l3743_374379

def M : Set ℝ := { y | ∃ x, y = Real.sin x }
def N : Set ℝ := {0, 1, 2}

theorem sine_intersection_theorem : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_sine_intersection_theorem_l3743_374379


namespace NUMINAMATH_CALUDE_triangle_property_l3743_374374

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition from the problem -/
def satisfies_condition (t : Triangle) : Prop :=
  1 - 2 * Real.sin t.B * Real.sin t.C = Real.cos (2 * t.B) + Real.cos (2 * t.C) - Real.cos (2 * t.A)

theorem triangle_property (t : Triangle) (h : satisfies_condition t) :
  t.A = Real.pi / 3 ∧ ∃ (x : ℝ), x ≤ Real.pi ∧ ∀ (y : ℝ), Real.sin t.B + Real.sin t.C ≤ Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l3743_374374


namespace NUMINAMATH_CALUDE_solve_candy_problem_l3743_374321

def candy_problem (total : ℕ) (snickers : ℕ) (mars : ℕ) : Prop :=
  ∃ butterfingers : ℕ, 
    total = snickers + mars + butterfingers ∧ 
    butterfingers = 7

theorem solve_candy_problem : candy_problem 12 3 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l3743_374321


namespace NUMINAMATH_CALUDE_total_animals_l3743_374345

theorem total_animals (L C P R Q : ℕ) : 
  L = 10 → 
  C = 2 * L + 4 → 
  ∃ G : ℕ, G = 2 * (L + 3) + Q → 
  (L + C + P) + ((L + 3) + R * (L + 3) + G) = 73 + P + R * 13 + Q :=
by sorry

end NUMINAMATH_CALUDE_total_animals_l3743_374345


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l3743_374328

/-- Given that B is a digit in base 5 and c is a base greater than 6, 
    if BBB₅ = 44ₖ, then the smallest possible sum of B + c is 34. -/
theorem smallest_sum_B_plus_c : 
  ∀ (B c : ℕ), 
    0 ≤ B ∧ B ≤ 4 →  -- B is a digit in base 5
    c > 6 →  -- c is a base greater than 6
    31 * B = 4 * c + 4 →  -- BBB₅ = 44ₖ
    ∀ (B' c' : ℕ), 
      0 ≤ B' ∧ B' ≤ 4 →
      c' > 6 →
      31 * B' = 4 * c' + 4 →
      B + c ≤ B' + c' →
      B + c = 34 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_c_l3743_374328


namespace NUMINAMATH_CALUDE_rainfall_ratio_l3743_374326

/-- Given the total rainfall over two weeks and the rainfall in the second week,
    calculate the ratio of the second week's rainfall to the first week's rainfall. -/
theorem rainfall_ratio (total : ℝ) (second_week : ℝ) :
  total = 30 →
  second_week = 18 →
  (second_week / (total - second_week) = 3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_rainfall_ratio_l3743_374326


namespace NUMINAMATH_CALUDE_unique_factor_pair_l3743_374376

theorem unique_factor_pair : ∃! (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ y ≥ x ∧ x + y ≤ 20 ∧ 
  (∃ (a b : ℕ), a ≠ x ∧ b ≠ y ∧ a * b = x * y) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → b ≥ a → a + b ≤ 20 → a * b ≠ x * y ∨ a + b ≠ 13) ∧
  x = 2 ∧ y = 11 := by
sorry

end NUMINAMATH_CALUDE_unique_factor_pair_l3743_374376


namespace NUMINAMATH_CALUDE_power_division_equals_729_l3743_374392

theorem power_division_equals_729 : 3^12 / 27^2 = 729 :=
by
  -- Define 27 as 3^3
  have h1 : 27 = 3^3 := by sorry
  
  -- Prove that 3^12 / 27^2 = 729
  sorry

end NUMINAMATH_CALUDE_power_division_equals_729_l3743_374392


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3743_374391

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem mans_age_to_sons_age_ratio :
  ∀ (man_age son_age : ℕ),
  man_age = son_age + 18 →
  son_age = 16 →
  ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l3743_374391


namespace NUMINAMATH_CALUDE_complement_of_60_18_l3743_374318

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (α : Angle) : Angle :=
  let total_minutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  ⟨total_minutes / 60, total_minutes % 60⟩

theorem complement_of_60_18 :
  let α : Angle := ⟨60, 18⟩
  complement α = ⟨29, 42⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_60_18_l3743_374318


namespace NUMINAMATH_CALUDE_factorization_equality_l3743_374350

theorem factorization_equality (x : ℝ) : 90 * x^2 + 60 * x + 30 = 30 * (3 * x^2 + 2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3743_374350


namespace NUMINAMATH_CALUDE_total_days_2010_to_2013_l3743_374382

/-- A year is a leap year if it's divisible by 4, except for century years,
    which must be divisible by 400 to be a leap year. -/
def isLeapYear (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- The number of days in a given year -/
def daysInYear (year : ℕ) : ℕ :=
  if isLeapYear year then 366 else 365

/-- The range of years we're considering -/
def yearRange : List ℕ := [2010, 2011, 2012, 2013]

theorem total_days_2010_to_2013 :
  (yearRange.map daysInYear).sum = 1461 := by
  sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2013_l3743_374382


namespace NUMINAMATH_CALUDE_min_value_problem_l3743_374369

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3/x + 1/y = 1) :
  3*x + 4*y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3/x₀ + 1/y₀ = 1 ∧ 3*x₀ + 4*y₀ = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3743_374369


namespace NUMINAMATH_CALUDE_gray_area_between_circles_l3743_374319

theorem gray_area_between_circles (r : ℝ) (R : ℝ) : 
  r > 0 → 
  R = 3 * r → 
  2 * r = 4 → 
  R^2 * π - r^2 * π = 32 * π := by
sorry

end NUMINAMATH_CALUDE_gray_area_between_circles_l3743_374319


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3743_374380

theorem partial_fraction_decomposition_product : 
  let f (x : ℝ) := (x^2 + 5*x - 14) / (x^3 + x^2 - 11*x - 13)
  let g (x : ℝ) (A B C : ℝ) := A / (x - 1) + B / (x + 1) + C / (x + 13)
  ∀ A B C : ℝ, (∀ x : ℝ, f x = g x A B C) → A * B * C = -360 / 343 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3743_374380


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3743_374394

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The value of C in base 13 -/
def C : ℕ := 12

/-- The first number in base 9 -/
def num1 : List ℕ := [7, 5, 2]

/-- The second number in base 13 -/
def num2 : List ℕ := [6, C, 3]

theorem base_conversion_sum :
  to_base_10 num1 9 + to_base_10 num2 13 = 1787 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_sum_l3743_374394


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3743_374308

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, x > 0 → x > -1) ∧ (∃ x, x > -1 ∧ ¬(x > 0)) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3743_374308


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l3743_374371

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l3743_374371


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3743_374340

theorem ceiling_floor_difference : ⌈(16 : ℝ) / 5 * (-34 : ℝ) / 4⌉ - ⌊(16 : ℝ) / 5 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3743_374340


namespace NUMINAMATH_CALUDE_intersection_point_l3743_374333

/-- A quadratic function of the form y = x^2 + px + q where 3p + q = 2023 -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : 3 * p + q = 2023

/-- The point (3, 2032) lies on all quadratic functions satisfying the given condition -/
theorem intersection_point (f : QuadraticFunction) : 
  3^2 + f.p * 3 + f.q = 2032 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3743_374333


namespace NUMINAMATH_CALUDE_ice_cream_cone_ratio_l3743_374336

def sugar_cones : ℕ := 45
def waffle_cones : ℕ := 36

theorem ice_cream_cone_ratio : 
  ∃ (a b : ℕ), a = 5 ∧ b = 4 ∧ sugar_cones * b = waffle_cones * a :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_cone_ratio_l3743_374336


namespace NUMINAMATH_CALUDE_power_sum_six_l3743_374320

theorem power_sum_six (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_six_l3743_374320


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3743_374364

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) →
  (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = -1) →
  -- Definitions of triangle
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  -- Law of cosines
  (Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) →
  -- Conclusions
  (C = π / 3) ∧ 
  (c = Real.sqrt 6) ∧ 
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3743_374364


namespace NUMINAMATH_CALUDE_first_player_win_probability_l3743_374337

-- Define the probability of winning in one roll
def prob_win_one_roll : ℚ := 21 / 36

-- Define the probability of not winning in one roll
def prob_not_win_one_roll : ℚ := 1 - prob_win_one_roll

-- Define the game
def dice_game_probability : ℚ :=
  prob_win_one_roll / (1 - prob_not_win_one_roll ^ 2)

-- Theorem statement
theorem first_player_win_probability :
  dice_game_probability = 12 / 17 := by sorry

end NUMINAMATH_CALUDE_first_player_win_probability_l3743_374337


namespace NUMINAMATH_CALUDE_cornelia_countries_l3743_374359

/-- The number of countries Cornelia visited in Europe -/
def europe_countries : ℕ := 20

/-- The number of countries Cornelia visited in South America -/
def south_america_countries : ℕ := 10

/-- The number of countries Cornelia visited in Asia -/
def asia_countries : ℕ := 6

/-- The total number of countries Cornelia visited -/
def total_countries : ℕ := europe_countries + south_america_countries + 2 * asia_countries

theorem cornelia_countries : total_countries = 42 := by
  sorry

end NUMINAMATH_CALUDE_cornelia_countries_l3743_374359


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l3743_374395

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_and_line_in_plane
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_and_line_in_plane_l3743_374395


namespace NUMINAMATH_CALUDE_fence_painting_time_l3743_374396

theorem fence_painting_time (taimour_time : ℝ) (h1 : taimour_time = 21) :
  let jamshid_time := taimour_time / 2
  let combined_rate := 1 / taimour_time + 1 / jamshid_time
  1 / combined_rate = 7 := by sorry

end NUMINAMATH_CALUDE_fence_painting_time_l3743_374396


namespace NUMINAMATH_CALUDE_saline_solution_concentration_l3743_374348

theorem saline_solution_concentration (x : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (a * x / 100 = (a + b) * 20 / 100) ∧ 
    ((a + b) * (1 - 20 / 100) = (a + 2*b) * (1 - 30 / 100))) →
  x = 70 / 3 := by
  sorry

end NUMINAMATH_CALUDE_saline_solution_concentration_l3743_374348


namespace NUMINAMATH_CALUDE_cost_calculation_l3743_374312

-- Define the number of caramel apples and ice cream cones
def caramel_apples : ℕ := 3
def ice_cream_cones : ℕ := 4

-- Define the price difference between a caramel apple and an ice cream cone
def price_difference : ℚ := 25 / 100

-- Define the total amount spent
def total_spent : ℚ := 2

-- Define the cost of an ice cream cone
def ice_cream_cost : ℚ := 125 / 700

-- Define the cost of a caramel apple
def caramel_apple_cost : ℚ := ice_cream_cost + price_difference

-- Theorem statement
theorem cost_calculation :
  (caramel_apples : ℚ) * caramel_apple_cost + (ice_cream_cones : ℚ) * ice_cream_cost = total_spent :=
sorry

end NUMINAMATH_CALUDE_cost_calculation_l3743_374312


namespace NUMINAMATH_CALUDE_Q_equals_G_l3743_374356

-- Define the sets
def P : Set ℝ := {y | ∃ x, y = x^2 + 1}
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def E : Set ℝ := {x | ∃ y, y = x^2 + 1}
def F : Set (ℝ × ℝ) := {(x, y) | y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_G : Q = G := by sorry

end NUMINAMATH_CALUDE_Q_equals_G_l3743_374356


namespace NUMINAMATH_CALUDE_inspector_examination_l3743_374361

/-- Given an inspector who rejects 0.02% of meters as defective and examined 10,000 meters to reject 2 meters,
    prove that to reject x meters, the inspector needs to examine 5000x meters. -/
theorem inspector_examination (x : ℝ) : 
  (2 / 10000 = x / (5000 * x)) → 5000 * x = (x * 10000) / 2 := by
sorry

end NUMINAMATH_CALUDE_inspector_examination_l3743_374361


namespace NUMINAMATH_CALUDE_locus_equals_homothety_image_l3743_374385

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- Represents a rotational homothety transformation -/
structure RotationalHomothety where
  center : Point
  angle : ℝ
  factor : ℝ

/-- The locus of points Y for a given semicircle and constant k -/
def locusOfY (s : Semicircle) (k : ℝ) : Set Point :=
  sorry

/-- The image of a semicircle under rotational homothety -/
def imageUnderHomothety (s : Semicircle) (h : RotationalHomothety) : Set Point :=
  sorry

/-- Main theorem: The locus of Y is the image of the semicircle under rotational homothety -/
theorem locus_equals_homothety_image (s : Semicircle) (k : ℝ) (h0 : k > 0) :
  locusOfY s k = imageUnderHomothety s ⟨s.center, Real.arctan k, Real.sqrt (k^2 + 1)⟩ :=
sorry

end NUMINAMATH_CALUDE_locus_equals_homothety_image_l3743_374385


namespace NUMINAMATH_CALUDE_max_digits_product_5digit_4digit_l3743_374360

theorem max_digits_product_5digit_4digit :
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 →
    1000 ≤ b ∧ b < 10000 →
    a * b < 10000000000 :=
by sorry

end NUMINAMATH_CALUDE_max_digits_product_5digit_4digit_l3743_374360


namespace NUMINAMATH_CALUDE_no_real_solutions_iff_k_in_range_l3743_374317

theorem no_real_solutions_iff_k_in_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 + Real.sqrt 2 * k * x + 2 ≥ 0) ↔ k ∈ Set.Icc 0 4 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_iff_k_in_range_l3743_374317


namespace NUMINAMATH_CALUDE_distinct_collections_l3743_374304

/-- Represents the count of each letter in MATHEMATICIAN --/
def letterCounts : Finset (Char × ℕ) := 
  {('M', 1), ('A', 3), ('T', 2), ('H', 1), ('E', 1), ('I', 3), ('C', 1), ('N', 1)}

/-- Represents the set of vowels in MATHEMATICIAN --/
def vowels : Finset Char := {'A', 'I', 'E'}

/-- Represents the set of consonants in MATHEMATICIAN --/
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'N'}

/-- Calculates the number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of distinct vowel selections --/
def vowelSelections : ℕ := 
  choose 3 3 + 3 * choose 3 2 + 3 * choose 3 1 + choose 3 0

/-- Calculates the number of distinct consonant selections --/
def consonantSelections : ℕ := 
  choose 4 3 + 2 * choose 4 2 + choose 4 1

/-- The main theorem stating the total number of distinct collections --/
theorem distinct_collections : 
  vowelSelections * consonantSelections = 112 := by sorry

end NUMINAMATH_CALUDE_distinct_collections_l3743_374304


namespace NUMINAMATH_CALUDE_union_equals_one_two_three_l3743_374387

def M : Set ℤ := {1, 3}
def N (a : ℤ) : Set ℤ := {1 - a, 3}

theorem union_equals_one_two_three (a : ℤ) : 
  M ∪ N a = {1, 2, 3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_one_two_three_l3743_374387


namespace NUMINAMATH_CALUDE_complex_equation_circle_l3743_374372

/-- The set of complex numbers z satisfying |z|^2 + |z| = 2 forms a circle in the complex plane. -/
theorem complex_equation_circle : 
  {z : ℂ | Complex.abs z ^ 2 + Complex.abs z = 2} = {z : ℂ | Complex.abs z = 1} := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_circle_l3743_374372


namespace NUMINAMATH_CALUDE_expression_evaluation_l3743_374330

theorem expression_evaluation (a b c : ℝ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 20) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3743_374330


namespace NUMINAMATH_CALUDE_sin_2theta_problem_l3743_374303

theorem sin_2theta_problem (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.cos (π/2 - θ) = 3/5) :
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_problem_l3743_374303


namespace NUMINAMATH_CALUDE_min_volleyballs_l3743_374310

/-- The price of 3 basketballs and 2 volleyballs -/
def price_3b_2v : ℕ := 520

/-- The price of 2 basketballs and 5 volleyballs -/
def price_2b_5v : ℕ := 640

/-- The total number of balls to be purchased -/
def total_balls : ℕ := 50

/-- The total budget in yuan -/
def total_budget : ℕ := 5500

/-- The price of a basketball in yuan -/
def basketball_price : ℕ := 120

/-- The price of a volleyball in yuan -/
def volleyball_price : ℕ := 80

theorem min_volleyballs (b v : ℕ) :
  3 * b + 2 * v = price_3b_2v →
  2 * b + 5 * v = price_2b_5v →
  b = basketball_price →
  v = volleyball_price →
  (∀ x y : ℕ, x + y = total_balls → basketball_price * x + volleyball_price * y ≤ total_budget →
    y ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_min_volleyballs_l3743_374310


namespace NUMINAMATH_CALUDE_min_abs_z_l3743_374390

/-- Given a complex number z satisfying |z - 10| + |z + 3i| = 15, the minimum value of |z| is 2. -/
theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15) : 
  ∃ (w : ℂ), Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15 ∧ Complex.abs w = 2 ∧ 
  ∀ (v : ℂ), Complex.abs (v - 10) + Complex.abs (v + 3*I) = 15 → Complex.abs w ≤ Complex.abs v :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l3743_374390


namespace NUMINAMATH_CALUDE_equation_solution_l3743_374302

theorem equation_solution : ∃ x : ℝ, (2 / 7) * (1 / 8) * x = 12 ∧ x = 336 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3743_374302


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3743_374378

/-- The distance between the foci of a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 is 2√(a^2 + b^2) -/
theorem hyperbola_foci_distance (a b : ℝ) (h : a > 0 ∧ b > 0) :
  let distance := 2 * Real.sqrt (a^2 + b^2)
  distance = 2 * Real.sqrt 34 ↔ a^2 = 25 ∧ b^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3743_374378


namespace NUMINAMATH_CALUDE_cone_radii_sum_l3743_374307

/-- Given a circle with radius 5 divided into three sectors with area ratios 1:2:3,
    when these sectors are used as lateral surfaces of three cones,
    the sum of the base radii of these cones equals 5. -/
theorem cone_radii_sum (r₁ r₂ r₃ : ℝ) : r₁ + r₂ + r₃ = 5 :=
  sorry

end NUMINAMATH_CALUDE_cone_radii_sum_l3743_374307


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l3743_374354

theorem integer_solutions_of_inequalities (x : ℤ) :
  -1 < x ∧ x ≤ 1 ∧ 4*(2*x-1) ≤ 3*x+1 ∧ 2*x > (x-3)/2 → x = 0 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequalities_l3743_374354


namespace NUMINAMATH_CALUDE_min_probability_theorem_l3743_374335

def closest_integer (m : ℤ) (k : ℤ) : ℤ := 
  sorry

def P (k : ℤ) : ℚ :=
  sorry

theorem min_probability_theorem :
  ∀ k : ℤ, k % 2 = 1 → 1 ≤ k → k ≤ 99 →
    P k ≥ 34/67 ∧ 
    ∃ k₀ : ℤ, k₀ % 2 = 1 ∧ 1 ≤ k₀ ∧ k₀ ≤ 99 ∧ P k₀ = 34/67 :=
  sorry

end NUMINAMATH_CALUDE_min_probability_theorem_l3743_374335


namespace NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l3743_374329

theorem a_fourth_plus_inverse_a_fourth (a : ℝ) (h : (a + 1/a)^3 = 7) :
  a^4 + 1/a^4 = 1519/81 := by
  sorry

end NUMINAMATH_CALUDE_a_fourth_plus_inverse_a_fourth_l3743_374329


namespace NUMINAMATH_CALUDE_tile_coverage_l3743_374332

theorem tile_coverage (original_count : ℕ) (original_side : ℝ) (new_side : ℝ) :
  original_count = 96 →
  original_side = 3 →
  new_side = 2 →
  (original_count * original_side * original_side) / (new_side * new_side) = 216 := by
  sorry

end NUMINAMATH_CALUDE_tile_coverage_l3743_374332


namespace NUMINAMATH_CALUDE_xy_values_l3743_374334

theorem xy_values (x y : ℝ) : (x + y + 2) * (x + y - 1) = 0 → x + y = -2 ∨ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_values_l3743_374334


namespace NUMINAMATH_CALUDE_star_operations_l3743_374373

-- Define the new operation
def star (x y : ℚ) : ℚ := x * y + |x - y| - 2

-- Theorem statement
theorem star_operations :
  (star 3 (-2) = -3) ∧ (star (star 2 5) (-4) = -31) := by
  sorry

end NUMINAMATH_CALUDE_star_operations_l3743_374373


namespace NUMINAMATH_CALUDE_w_range_l3743_374301

-- Define the function w(x)
def w (x : ℝ) : ℝ := x^4 - 6*x^2 + 9

-- Theorem stating the range of w(x)
theorem w_range :
  Set.range w = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_w_range_l3743_374301


namespace NUMINAMATH_CALUDE_exists_equal_boundary_interior_rectangle_l3743_374343

/-- Represents a rectangle in a triangular lattice grid -/
structure TriLatticeRectangle where
  m : Nat  -- horizontal side length in lattice units
  n : Nat  -- vertical side length in lattice units

/-- Calculates the number of lattice points on the boundary of the rectangle -/
def boundaryPoints (rect : TriLatticeRectangle) : Nat :=
  2 * (rect.m + rect.n)

/-- Calculates the number of lattice points inside the rectangle -/
def interiorPoints (rect : TriLatticeRectangle) : Nat :=
  2 * rect.m * rect.n - rect.m - rect.n + 1

/-- Theorem stating the existence of a rectangle with equal boundary and interior points -/
theorem exists_equal_boundary_interior_rectangle :
  ∃ (rect : TriLatticeRectangle), boundaryPoints rect = interiorPoints rect :=
sorry

end NUMINAMATH_CALUDE_exists_equal_boundary_interior_rectangle_l3743_374343


namespace NUMINAMATH_CALUDE_two_n_is_good_pair_exists_good_pair_greater_than_two_l3743_374389

/-- A pair (m,n) is good if, when erasing every m-th and then every n-th number, 
    and separately erasing every n-th and then every m-th number, 
    any number k that occurs in both resulting lists appears at the same position in both lists -/
def is_good_pair (m n : ℕ) : Prop :=
  ∀ k : ℕ, 
    let pos1 := (k - k / n) - (k - k / n) / m
    let pos2 := k / m - (k / m) / n
    (pos1 ≠ 0 ∧ pos2 ≠ 0) → pos1 = pos2

/-- For any positive integer n, (2,n) is a good pair -/
theorem two_n_is_good_pair : ∀ n : ℕ, n > 0 → is_good_pair 2 n := by sorry

/-- There exists a pair of positive integers (m,n) such that 2 < m < n and (m,n) is a good pair -/
theorem exists_good_pair_greater_than_two : 
  ∃ m n : ℕ, 2 < m ∧ m < n ∧ is_good_pair m n := by sorry

end NUMINAMATH_CALUDE_two_n_is_good_pair_exists_good_pair_greater_than_two_l3743_374389


namespace NUMINAMATH_CALUDE_eight_couples_handshakes_l3743_374353

/-- The number of handshakes in a gathering of married couples --/
def handshakes (n : ℕ) : ℕ :=
  (n * (2 * n - 1)) / 2 - n

/-- Theorem: In a gathering of 8 married couples, the total number of handshakes is 112 --/
theorem eight_couples_handshakes :
  handshakes 8 = 112 := by
  sorry

end NUMINAMATH_CALUDE_eight_couples_handshakes_l3743_374353


namespace NUMINAMATH_CALUDE_polygon_sides_l3743_374397

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : sum_angles = 900 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3743_374397


namespace NUMINAMATH_CALUDE_tower_heights_count_l3743_374311

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of distinct tower heights achievable -/
def distinctTowerHeights (num_bricks : ℕ) (brick_dims : BrickDimensions) : ℕ :=
  sorry

/-- Theorem stating the number of distinct tower heights for the given problem -/
theorem tower_heights_count :
  let brick_dims : BrickDimensions := ⟨20, 10, 6⟩
  distinctTowerHeights 100 brick_dims = 701 := by
  sorry

end NUMINAMATH_CALUDE_tower_heights_count_l3743_374311


namespace NUMINAMATH_CALUDE_weight_comparison_l3743_374309

def weights : List ℝ := [4, 4, 5, 7, 9, 120]

def median (l : List ℝ) : ℝ := sorry

def mean (l : List ℝ) : ℝ := sorry

theorem weight_comparison (h : weights = [4, 4, 5, 7, 9, 120]) : 
  mean weights - median weights = 19 := by sorry

end NUMINAMATH_CALUDE_weight_comparison_l3743_374309


namespace NUMINAMATH_CALUDE_circle_center_l3743_374398

-- Define the equation
def circle_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0

-- Define what it means for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), ∀ (x y : ℝ), 
    circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem circle_center : 
  ∃ (a : ℝ), is_circle a ∧ 
  ∀ (h k : ℝ), (∀ (x y : ℝ), circle_equation a x y ↔ (x - h)^2 + (y - k)^2 = 25) → 
  h = -2 ∧ k = -4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l3743_374398


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l3743_374316

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 11

theorem quadratic_function_proof :
  (∀ x : ℝ, f x ≤ 13) ∧  -- maximum value is 13
  f 3 = 5 ∧              -- f(3) = 5
  f (-1) = 5 ∧           -- f(-1) = 5
  (∀ x : ℝ, f x = -2 * x^2 + 4 * x + 11) -- explicit formula
  :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l3743_374316


namespace NUMINAMATH_CALUDE_jim_bakes_two_loaves_l3743_374386

/-- The amount of flour Jim can bake into loaves -/
def jim_loaves (cupboard kitchen_counter pantry loaf_requirement : ℕ) : ℕ :=
  (cupboard + kitchen_counter + pantry) / loaf_requirement

/-- Theorem: Jim can bake 2 loaves of bread -/
theorem jim_bakes_two_loaves :
  jim_loaves 200 100 100 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_bakes_two_loaves_l3743_374386


namespace NUMINAMATH_CALUDE_bakery_sales_theorem_l3743_374323

/-- Represents the bakery sales scenario -/
structure BakerySales where
  pumpkin_slices_per_pie : ℕ
  custard_slices_per_pie : ℕ
  pumpkin_price_per_slice : ℕ
  custard_price_per_slice : ℕ
  pumpkin_pies_sold : ℕ
  custard_pies_sold : ℕ

/-- Calculates the total sales from the bakery -/
def total_sales (s : BakerySales) : ℕ :=
  (s.pumpkin_slices_per_pie * s.pumpkin_pies_sold * s.pumpkin_price_per_slice) +
  (s.custard_slices_per_pie * s.custard_pies_sold * s.custard_price_per_slice)

/-- Theorem stating that given the specific conditions, the total sales equal $340 -/
theorem bakery_sales_theorem (s : BakerySales) 
  (h1 : s.pumpkin_slices_per_pie = 8)
  (h2 : s.custard_slices_per_pie = 6)
  (h3 : s.pumpkin_price_per_slice = 5)
  (h4 : s.custard_price_per_slice = 6)
  (h5 : s.pumpkin_pies_sold = 4)
  (h6 : s.custard_pies_sold = 5) :
  total_sales s = 340 := by
  sorry

#eval total_sales {
  pumpkin_slices_per_pie := 8,
  custard_slices_per_pie := 6,
  pumpkin_price_per_slice := 5,
  custard_price_per_slice := 6,
  pumpkin_pies_sold := 4,
  custard_pies_sold := 5
}

end NUMINAMATH_CALUDE_bakery_sales_theorem_l3743_374323


namespace NUMINAMATH_CALUDE_root_in_interval_l3743_374375

def f (x : ℝ) := x^3 + 2*x - 1

theorem root_in_interval :
  (f 0 < 0) →
  (f 1 > 0) →
  ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3743_374375


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l3743_374324

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The set of digits used to form the numbers. -/
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The number of digits in each formed number. -/
def number_length : ℕ := 7

/-- The digit that must be at the last position. -/
def last_digit : ℕ := 3

/-- The theorem stating the number of valid arrangements. -/
theorem count_valid_arrangements :
  (factorial (number_length - 1) / 2 : ℕ) = 360 := by sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l3743_374324


namespace NUMINAMATH_CALUDE_remainder_problem_l3743_374355

theorem remainder_problem (x y : ℕ) 
  (h1 : 1059 % x = y)
  (h2 : 1417 % x = y)
  (h3 : 2312 % x = y) :
  x - y = 15 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3743_374355


namespace NUMINAMATH_CALUDE_right_triangle_sum_of_squares_l3743_374358

theorem right_triangle_sum_of_squares (A B C : ℝ × ℝ) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 1 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.1 - C.1)^2 + (A.2 - C.2)^2 + (B.1 - C.1)^2 + (B.2 - C.2)^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_sum_of_squares_l3743_374358


namespace NUMINAMATH_CALUDE_percentage_problem_l3743_374322

theorem percentage_problem (x : ℝ) : (35 / 100) * x = 126 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3743_374322


namespace NUMINAMATH_CALUDE_complex_roots_condition_l3743_374367

theorem complex_roots_condition (p : ℝ) :
  (∀ x : ℝ, x^2 + p*x + 1 ≠ 0) →
  p < 2 ∧
  ¬(p < 2 → ∀ x : ℝ, x^2 + p*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_condition_l3743_374367


namespace NUMINAMATH_CALUDE_students_not_in_biology_l3743_374357

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) : 
  total_students = 880 →
  biology_percentage = 35 / 100 →
  total_students - (biology_percentage * total_students).floor = 572 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l3743_374357


namespace NUMINAMATH_CALUDE_range_of_c_l3743_374366

-- Define the propositions P and Q
def P (c : ℝ) : Prop := ∀ x : ℝ, Monotone (fun x => (c^2 - 5*c + 7)^x)

def Q (c : ℝ) : Prop := ∀ x : ℝ, |x - 1| + |x - 2*c| > 1

-- Define the theorem
theorem range_of_c :
  (∃! c : ℝ, P c ∨ Q c) →
  {c : ℝ | c ∈ Set.Icc 0 1 ∪ Set.Icc 2 3} = {c : ℝ | P c ∨ Q c} :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l3743_374366


namespace NUMINAMATH_CALUDE_ln_inequality_and_range_l3743_374341

open Real

theorem ln_inequality_and_range (x : ℝ) (hx : x > 0) :
  (∀ x > 0, Real.log x ≤ x - 1) ∧
  (∀ a : ℝ, (∀ x > 0, Real.log x ≤ a * x + (a - 1) / x - 1) ↔ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_ln_inequality_and_range_l3743_374341


namespace NUMINAMATH_CALUDE_inequality_solution_l3743_374399

theorem inequality_solution :
  ∀ x y : ℝ,
  (y^2)^2 < (x + 1)^2 ∧ (x + 1)^2 = y^4 + y^2 + 1 ∧ y^4 + y^2 + 1 ≤ (y^2 + 1)^2 →
  (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3743_374399
