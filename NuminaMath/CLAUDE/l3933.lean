import Mathlib

namespace NUMINAMATH_CALUDE_baker_cakes_total_l3933_393334

/-- Calculates the total number of full-size cakes given initial cakes, additional cakes, and half-cakes. -/
def totalFullSizeCakes (initialCakes additionalCakes halfCakes : ℕ) : ℕ :=
  initialCakes + additionalCakes + (halfCakes / 2)

/-- Theorem stating that given the specific numbers from the problem, the total full-size cakes is 512. -/
theorem baker_cakes_total :
  totalFullSizeCakes 350 125 75 = 512 := by
  sorry

#eval totalFullSizeCakes 350 125 75

end NUMINAMATH_CALUDE_baker_cakes_total_l3933_393334


namespace NUMINAMATH_CALUDE_monkey_climb_l3933_393343

/-- Monkey's climb on a greased pole -/
theorem monkey_climb (pole_height : ℝ) (ascent : ℝ) (total_minutes : ℕ) (slip : ℝ) : 
  pole_height = 10 →
  ascent = 2 →
  total_minutes = 17 →
  (total_minutes / 2 : ℝ) * ascent - ((total_minutes - 1) / 2 : ℝ) * slip = pole_height →
  slip = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_l3933_393343


namespace NUMINAMATH_CALUDE_balloon_arrangements_l3933_393340

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : Nat) (repeatedLetters : List (Nat)) : Nat :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

theorem balloon_arrangements :
  distinctArrangements 7 [2, 2] = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l3933_393340


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3933_393336

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_properties 
  (a : ℕ → ℚ) 
  (h_geometric : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3933_393336


namespace NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_is_correct_division_result_l3933_393372

/-- The largest multiple of 18 with digits 9 or 0 -/
def largest_multiple_18_with_9_0 : ℕ := 9990

/-- Check if a natural number consists only of digits 9 and 0 -/
def has_only_9_and_0_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 9 ∨ d = 0

theorem largest_multiple_18_with_9_0_is_correct :
  largest_multiple_18_with_9_0 % 18 = 0 ∧
  has_only_9_and_0_digits largest_multiple_18_with_9_0 ∧
  ∀ m : ℕ, m > largest_multiple_18_with_9_0 →
    m % 18 ≠ 0 ∨ ¬(has_only_9_and_0_digits m) :=
by sorry

theorem division_result :
  largest_multiple_18_with_9_0 / 18 = 555 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_18_with_9_0_is_correct_division_result_l3933_393372


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l3933_393383

/-- The number of books in the 'crazy silly school' series -/
def total_books : ℕ := 21

/-- The number of books that have been read -/
def books_read : ℕ := 13

/-- The number of books yet to be read -/
def books_unread : ℕ := 8

/-- Theorem: The total number of books is equal to the sum of read and unread books -/
theorem crazy_silly_school_books : total_books = books_read + books_unread := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l3933_393383


namespace NUMINAMATH_CALUDE_inequality_proof_l3933_393358

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * (a + b)) + Real.sqrt (b * c * (b + c)) + Real.sqrt (c * a * (c + a)) >
  Real.sqrt ((a + b) * (b + c) * (c + a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3933_393358


namespace NUMINAMATH_CALUDE_y_min_at_a_or_b_l3933_393344

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 * (x - b)^2

/-- Theorem stating that the minimum of y occurs at x = a or x = b -/
theorem y_min_at_a_or_b (a b : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b ≥ y x a b ∧ (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_y_min_at_a_or_b_l3933_393344


namespace NUMINAMATH_CALUDE_inequality_preservation_l3933_393304

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a + c^2 > b + c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3933_393304


namespace NUMINAMATH_CALUDE_maria_water_bottles_l3933_393379

/-- The number of bottles Maria initially had -/
def initial_bottles : ℕ := 14

/-- The number of bottles Maria drank -/
def bottles_drunk : ℕ := 8

/-- The number of bottles Maria bought -/
def bottles_bought : ℕ := 45

/-- The final number of bottles Maria has -/
def final_bottles : ℕ := 51

theorem maria_water_bottles : 
  initial_bottles - bottles_drunk + bottles_bought = final_bottles :=
by sorry

end NUMINAMATH_CALUDE_maria_water_bottles_l3933_393379


namespace NUMINAMATH_CALUDE_parabola_minimum_y_value_l3933_393389

/-- The minimum y-value of the parabola y = 3x^2 + 6x + 4 is 1 -/
theorem parabola_minimum_y_value :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 + 6 * x + 4
  ∃ x₀ : ℝ, ∀ x : ℝ, f x₀ ≤ f x ∧ f x₀ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_minimum_y_value_l3933_393389


namespace NUMINAMATH_CALUDE_l_shape_area_l3933_393367

/-- The area of an "L" shape formed by removing a smaller rectangle from a larger rectangle -/
theorem l_shape_area (large_length large_width small_length_diff small_width_diff : ℕ) : 
  large_length = 10 →
  large_width = 7 →
  small_length_diff = 3 →
  small_width_diff = 3 →
  (large_length * large_width) - ((large_length - small_length_diff) * (large_width - small_width_diff)) = 42 := by
sorry

end NUMINAMATH_CALUDE_l_shape_area_l3933_393367


namespace NUMINAMATH_CALUDE_fourth_vertex_exists_l3933_393384

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a quadrilateral
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

-- Define properties of the quadrilateral
def is_cyclic (q : Quadrilateral) : Prop := sorry

def is_tangential (q : Quadrilateral) : Prop := sorry

def is_convex (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem fourth_vertex_exists 
  (A B C : Point) 
  (h_convex : is_convex ⟨A, B, C, C⟩) 
  (h_cyclic : ∀ D, is_cyclic ⟨A, B, C, D⟩) 
  (h_tangential : ∀ D, is_tangential ⟨A, B, C, D⟩) : 
  ∃ D, is_cyclic ⟨A, B, C, D⟩ ∧ is_tangential ⟨A, B, C, D⟩ ∧ is_convex ⟨A, B, C, D⟩ :=
sorry

end NUMINAMATH_CALUDE_fourth_vertex_exists_l3933_393384


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3933_393329

-- Define sets A and B
def A : Set ℝ := {x | x - 1 ≤ 0}
def B : Set ℝ := {x | x^2 - 4*x ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3933_393329


namespace NUMINAMATH_CALUDE_exchange_rate_problem_l3933_393359

theorem exchange_rate_problem (x : ℕ) : 
  (8 * x / 5 : ℚ) - 80 = x →
  (x / 100 + (x % 100) / 10 + x % 10 : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_problem_l3933_393359


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l3933_393302

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  s : ℝ  -- radius of the sphere
  a : ℝ  -- length of the box
  b : ℝ  -- width of the box
  c : ℝ  -- height of the box

/-- The sum of the lengths of the 12 edges of the box -/
def edge_sum (box : InscribedBox) : ℝ := 4 * (box.a + box.b + box.c)

/-- The surface area of the box -/
def surface_area (box : InscribedBox) : ℝ := 2 * (box.a * box.b + box.b * box.c + box.c * box.a)

/-- The main theorem -/
theorem inscribed_box_radius (box : InscribedBox) 
  (h1 : edge_sum box = 72)
  (h2 : surface_area box = 216) :
  box.s = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l3933_393302


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3933_393342

theorem sqrt_sum_fractions : 
  Real.sqrt (2 * ((1 : ℝ) / 25 + (1 : ℝ) / 36)) = (Real.sqrt 122) / 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3933_393342


namespace NUMINAMATH_CALUDE_range_of_fraction_l3933_393319

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  ∃ (x : ℝ), -2 < x ∧ x < -1/2 ∧ (∃ (a' b' : ℝ), 1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < -1 ∧ x = a' / b') ∧
  (∀ (y : ℝ), (∃ (a' b' : ℝ), 1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < -1 ∧ y = a' / b') → -2 < y ∧ y < -1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3933_393319


namespace NUMINAMATH_CALUDE_remaining_money_proof_l3933_393370

def calculate_remaining_money (initial_amount apples_price milk_price oranges_price candy_price eggs_price apples_discount milk_discount : ℚ) : ℚ :=
  let discounted_apples_price := apples_price * (1 - apples_discount)
  let discounted_milk_price := milk_price * (1 - milk_discount)
  let total_spent := discounted_apples_price + discounted_milk_price + oranges_price + candy_price + eggs_price
  initial_amount - total_spent

theorem remaining_money_proof :
  calculate_remaining_money 95 25 8 14 6 12 (15/100) (10/100) = 6891/200 :=
by sorry

end NUMINAMATH_CALUDE_remaining_money_proof_l3933_393370


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l3933_393315

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given three points -/
def area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Angle in degrees between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem triangle_angle_theorem (t : Triangle) :
  let O := circumcenter t
  angle t.B t.C t.A = 75 →
  area O t.A t.B + area O t.B t.C = Real.sqrt 3 * area O t.C t.A →
  angle t.B t.A t.C = 45 := by
    sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l3933_393315


namespace NUMINAMATH_CALUDE_final_hair_length_l3933_393346

/-- Given initial hair length x, amount cut off y, and growth z,
    prove that the final hair length F is 17 inches. -/
theorem final_hair_length
  (x y z : ℝ)
  (hx : x = 16)
  (hy : y = 11)
  (hz : z = 12)
  (hF : F = (x - y) + z) :
  F = 17 :=
by sorry

end NUMINAMATH_CALUDE_final_hair_length_l3933_393346


namespace NUMINAMATH_CALUDE_younger_person_age_is_29_l3933_393357

/-- The age difference between Brittany and the other person -/
def age_difference : ℕ := 3

/-- The duration of Brittany's vacation -/
def vacation_duration : ℕ := 4

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation : ℕ := 32

/-- The age of the person who is younger than Brittany -/
def younger_person_age : ℕ := brittany_age_after_vacation - vacation_duration - age_difference

theorem younger_person_age_is_29 : younger_person_age = 29 := by
  sorry

end NUMINAMATH_CALUDE_younger_person_age_is_29_l3933_393357


namespace NUMINAMATH_CALUDE_scooter_gain_percent_l3933_393323

/-- Calculate the gain percent on a scooter sale -/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_costs : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 900)
  (h2 : repair_costs = 300)
  (h3 : selling_price = 1320) :
  (selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_scooter_gain_percent_l3933_393323


namespace NUMINAMATH_CALUDE_stating_pipeline_equation_l3933_393341

/-- Represents the total length of the pipeline in meters -/
def total_length : ℝ := 3000

/-- Represents the increase in daily work efficiency as a decimal -/
def efficiency_increase : ℝ := 0.25

/-- Represents the number of days the project is completed ahead of schedule -/
def days_ahead : ℝ := 20

/-- 
Theorem stating that the equation correctly represents the relationship 
between the original daily pipeline laying rate and the given conditions
-/
theorem pipeline_equation (x : ℝ) : 
  total_length / ((1 + efficiency_increase) * x) - total_length / x = days_ahead := by
  sorry

end NUMINAMATH_CALUDE_stating_pipeline_equation_l3933_393341


namespace NUMINAMATH_CALUDE_quadratic_equation_problem_l3933_393332

theorem quadratic_equation_problem (m : ℤ) (a : ℝ) 
  (h1 : ∃ x y : ℝ, x ≠ y ∧ (m^2 - m) * x^2 - 2*m*x + 1 = 0 ∧ (m^2 - m) * y^2 - 2*m*y + 1 = 0)
  (h2 : m < 3)
  (h3 : (m^2 - m) * a^2 - 2*m*a + 1 = 0) :
  m = 2 ∧ (2*a^2 - 3*a - 3 = (-6 + Real.sqrt 2) / 2 ∨ 2*a^2 - 3*a - 3 = (-6 - Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem_l3933_393332


namespace NUMINAMATH_CALUDE_total_vacations_and_classes_l3933_393347

/-- The number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- The number of vacations Grant has -/
def grant_vacations : ℕ := 4 * kelvin_classes

/-- The total number of vacations and classes Grant and Kelvin have altogether -/
def total : ℕ := grant_vacations + kelvin_classes

theorem total_vacations_and_classes : total = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_vacations_and_classes_l3933_393347


namespace NUMINAMATH_CALUDE_four_solutions_l3933_393398

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - y + 3) * (4 * x + y - 5) = 0
def equation2 (x y : ℝ) : Prop := (x + y - 3) * (3 * x - 4 * y + 6) = 0

-- Define a solution as a pair of real numbers satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- State the theorem
theorem four_solutions :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 4 ∧ (∀ p ∈ s, is_solution p) ∧
  (∀ p : ℝ × ℝ, is_solution p → p ∈ s) :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l3933_393398


namespace NUMINAMATH_CALUDE_q_is_zero_l3933_393369

/-- A cubic polynomial with roots at -2, 0, and 2, passing through (1, -3) -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem q_is_zero (p q r s : ℝ) :
  (∀ x, x = -2 ∨ x = 0 ∨ x = 2 → g p q r s x = 0) →
  g p q r s 1 = -3 →
  q = 0 :=
sorry

end NUMINAMATH_CALUDE_q_is_zero_l3933_393369


namespace NUMINAMATH_CALUDE_max_safe_destroyers_l3933_393365

/-- Represents the configuration of ships and torpedo boats --/
structure NavalSetup where
  total_ships : Nat
  destroyers : Nat
  small_boats : Nat
  torpedo_boats : Nat
  torpedoes_per_boat : Nat

/-- Represents the targeting capabilities of torpedo boats --/
inductive TargetingStrategy
  | Successive : TargetingStrategy  -- Can target 10 successive ships
  | NextByOne : TargetingStrategy   -- Can target 10 ships next by one

/-- Defines a valid naval setup based on the problem conditions --/
def valid_setup (s : NavalSetup) : Prop :=
  s.total_ships = 30 ∧
  s.destroyers = 10 ∧
  s.small_boats = 20 ∧
  s.torpedo_boats = 2 ∧
  s.torpedoes_per_boat = 10

/-- Defines the maximum number of destroyers that can be targeted --/
def max_targeted_destroyers (s : NavalSetup) : Nat :=
  7  -- Based on the solution analysis

/-- The main theorem to be proved --/
theorem max_safe_destroyers (s : NavalSetup) 
  (h_valid : valid_setup s) :
  ∃ (safe_destroyers : Nat),
    safe_destroyers = s.destroyers - max_targeted_destroyers s ∧
    safe_destroyers = 3 :=
  sorry


end NUMINAMATH_CALUDE_max_safe_destroyers_l3933_393365


namespace NUMINAMATH_CALUDE_other_pencil_length_l3933_393382

/-- Given two pencils with a total length of 24 cubes, where one pencil is 12 cubes long,
    the other pencil must be 12 cubes long. -/
theorem other_pencil_length (total_length : ℕ) (first_pencil : ℕ) (h1 : total_length = 24) (h2 : first_pencil = 12) :
  total_length - first_pencil = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_pencil_length_l3933_393382


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l3933_393373

/-- Given a triangle with two sides of length 51 and 67 units, and the third side being an integer,
    the minimum possible perimeter is 135 units. -/
theorem min_perimeter_triangle (a b x : ℕ) (ha : a = 51) (hb : b = 67) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b > y ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 135 := by
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l3933_393373


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3933_393305

theorem inequality_system_solution :
  ∀ x : ℝ, (x + 2 > -1 ∧ x - 5 < 3 * (x - 1)) ↔ x > -1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3933_393305


namespace NUMINAMATH_CALUDE_lisa_candies_on_specific_days_l3933_393390

/-- The number of candies Lisa eats on Mondays and Wednesdays -/
def candies_on_specific_days (total_candies : ℕ) (weeks : ℕ) (days_per_week : ℕ) 
  (specific_days : ℕ) : ℕ :=
  let candies_on_other_days := (days_per_week - specific_days) * weeks
  let remaining_candies := total_candies - candies_on_other_days
  remaining_candies / (specific_days * weeks)

/-- Theorem stating that Lisa eats 2 candies on Mondays and Wednesdays -/
theorem lisa_candies_on_specific_days : 
  candies_on_specific_days 36 4 7 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candies_on_specific_days_l3933_393390


namespace NUMINAMATH_CALUDE_x_coord_difference_at_y_10_l3933_393328

/-- Represents a line in 2D space -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- Calculates the x-coordinate for a given y-coordinate on a line -/
def xCoordAtY (l : Line) (y : ℚ) : ℚ :=
  (y - l.intercept) / l.slope

/-- Creates a line from two points -/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line where
  slope := (y2 - y1) / (x2 - x1)
  intercept := y1 - (y2 - y1) / (x2 - x1) * x1

theorem x_coord_difference_at_y_10 : 
  let p := lineFromPoints 0 3 4 0
  let q := lineFromPoints 0 1 8 0
  let xp := xCoordAtY p 10
  let xq := xCoordAtY q 10
  |xp - xq| = 188 / 3 := by
    sorry

end NUMINAMATH_CALUDE_x_coord_difference_at_y_10_l3933_393328


namespace NUMINAMATH_CALUDE_perfect_square_proof_l3933_393392

theorem perfect_square_proof (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  ((2 * l - n - k) * (2 * l - n + k)) / 2 = (l - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_proof_l3933_393392


namespace NUMINAMATH_CALUDE_bank_deposit_time_calculation_l3933_393363

/-- Proves that given two equal deposits at the same interest rate, 
    if the difference in interest is known, we can determine the time for the first deposit. -/
theorem bank_deposit_time_calculation 
  (deposit : ℝ) 
  (rate : ℝ) 
  (time_second : ℝ) 
  (interest_diff : ℝ) 
  (h1 : deposit = 640)
  (h2 : rate = 0.15)
  (h3 : time_second = 5)
  (h4 : interest_diff = 144) :
  ∃ (time_first : ℝ), 
    deposit * rate * time_second - deposit * rate * time_first = interest_diff ∧ 
    time_first = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_bank_deposit_time_calculation_l3933_393363


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3933_393327

theorem existence_of_special_number :
  ∃ N : ℕ, 
    (∃ a b : ℕ, a < 150 ∧ b < 150 ∧ b = a + 1 ∧ ¬(a ∣ N) ∧ ¬(b ∣ N)) ∧
    (∀ k : ℕ, k ≤ 150 → (k ∣ N) ∨ k = a ∨ k = b) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3933_393327


namespace NUMINAMATH_CALUDE_complex_subtraction_example_l3933_393303

theorem complex_subtraction_example : (6 - 2*I) - (3*I + 1) = 5 - 5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_example_l3933_393303


namespace NUMINAMATH_CALUDE_odd_sum_product_equivalence_l3933_393312

theorem odd_sum_product_equivalence (p q : ℕ) 
  (hp : p < 16 ∧ p % 2 = 1) 
  (hq : q < 16 ∧ q % 2 = 1) : 
  p * q + p + q = (p + 1) * (q + 1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_product_equivalence_l3933_393312


namespace NUMINAMATH_CALUDE_x_value_l3933_393374

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = 4 * w + 40)
  (hy : y = 3 * z + 15)
  (hx : x = 2 * y + 6) : 
  x = 2436 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3933_393374


namespace NUMINAMATH_CALUDE_complement_A_when_a_5_union_A_B_when_a_2_l3933_393313

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}
def B : Set ℝ := {x | x < 0 ∨ x > 5}

-- Theorem 1: Complement of A when a = 5
theorem complement_A_when_a_5 : 
  (A 5)ᶜ = {x : ℝ | x < 4 ∨ x > 11} := by sorry

-- Theorem 2: Union of A and B when a = 2
theorem union_A_B_when_a_2 : 
  A 2 ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_when_a_5_union_A_B_when_a_2_l3933_393313


namespace NUMINAMATH_CALUDE_star_three_five_l3933_393309

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

-- State the theorem
theorem star_three_five : star 3 5 = 64 := by sorry

end NUMINAMATH_CALUDE_star_three_five_l3933_393309


namespace NUMINAMATH_CALUDE_triangle_side_length_l3933_393396

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a = 5 →
  b = 7 →
  B = π / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3933_393396


namespace NUMINAMATH_CALUDE_triangle_angles_from_exterior_l3933_393386

theorem triangle_angles_from_exterior (A B C : ℝ) : 
  A + B + C = 180 →
  (180 - B) / (180 - C) = 12 / 7 →
  (180 - B) - (180 - C) = 50 →
  (A = 10 ∧ B = 60 ∧ C = 110) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angles_from_exterior_l3933_393386


namespace NUMINAMATH_CALUDE_target_hit_probability_l3933_393337

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 0.5) (h2 : p2 = 0.7) :
  1 - (1 - p1) * (1 - p2) = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l3933_393337


namespace NUMINAMATH_CALUDE_unique_valid_number_l3933_393307

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ 
  (n / 100 % 10 > 6) ∧ (n / 10 % 10 > 6) ∧ (n % 10 > 6) ∧
  n % 12 = 0

theorem unique_valid_number : ∃! n, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l3933_393307


namespace NUMINAMATH_CALUDE_xy_max_value_l3933_393331

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2 * y = 8) :
  x * y ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_xy_max_value_l3933_393331


namespace NUMINAMATH_CALUDE_difference_of_squares_l3933_393317

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3933_393317


namespace NUMINAMATH_CALUDE_max_common_ratio_geometric_sequence_l3933_393354

/-- Given a geometric sequence {a_n} satisfying a_1(a_2 + a_3) = 6a_1 - 9, 
    the maximum value of the common ratio q is (-1 + √5) / 2 -/
theorem max_common_ratio_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 * (a 2 + a 3) = 6 * a 1 - 9 →  -- given equation
  q ≤ (-1 + Real.sqrt 5) / 2 ∧
  ∃ (a : ℕ → ℝ), (∀ n, a (n + 1) = a n * q) ∧ 
    a 1 * (a 2 + a 3) = 6 * a 1 - 9 ∧ 
    q = (-1 + Real.sqrt 5) / 2 := by
  sorry

#check max_common_ratio_geometric_sequence

end NUMINAMATH_CALUDE_max_common_ratio_geometric_sequence_l3933_393354


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3933_393349

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5*x^2 + 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3933_393349


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l3933_393371

/-- Given 60 feet of fencing for a rectangular pen, the maximum possible area is 225 square feet -/
theorem max_area_rectangular_pen (perimeter : ℝ) (area : ℝ → ℝ → ℝ) :
  perimeter = 60 →
  (∀ x y, x > 0 → y > 0 → x + y = perimeter / 2 → area x y = x * y) →
  (∃ x y, x > 0 ∧ y > 0 ∧ x + y = perimeter / 2 ∧ area x y = 225) ∧
  (∀ x y, x > 0 → y > 0 → x + y = perimeter / 2 → area x y ≤ 225) :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l3933_393371


namespace NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l3933_393300

/-- Represents the width of each rectangle --/
def rectangle_width : ℝ := 4

/-- Represents the length of each rectangle --/
def rectangle_length : ℝ := 4 * rectangle_width

/-- Represents the side length of the original square --/
def square_side : ℝ := rectangle_width + rectangle_length

/-- Represents the perimeter of the "P" shape --/
def p_perimeter : ℝ := 56

theorem square_perimeter_from_p_shape :
  p_perimeter = 2 * (square_side) + rectangle_length →
  4 * square_side = 80 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_p_shape_l3933_393300


namespace NUMINAMATH_CALUDE_prob_ace_ten_jack_standard_deck_l3933_393330

/-- Represents a standard deck of 52 playing cards. -/
structure Deck :=
  (cards : Nat)
  (aces : Nat)
  (tens : Nat)
  (jacks : Nat)

/-- The probability of drawing an Ace, then a 10, then a Jack from a standard 52-card deck without replacement. -/
def prob_ace_ten_jack (d : Deck) : ℚ :=
  (d.aces : ℚ) / d.cards *
  (d.tens : ℚ) / (d.cards - 1) *
  (d.jacks : ℚ) / (d.cards - 2)

/-- Theorem stating that the probability of drawing an Ace, then a 10, then a Jack
    from a standard 52-card deck without replacement is 8/16575. -/
theorem prob_ace_ten_jack_standard_deck :
  prob_ace_ten_jack ⟨52, 4, 4, 4⟩ = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_ten_jack_standard_deck_l3933_393330


namespace NUMINAMATH_CALUDE_cross_arrangement_sum_l3933_393310

/-- A type representing digits from 0 to 9 -/
inductive Digit : Type
| zero | one | two | three | four | five | six | seven | eight | nine

/-- Convert a Digit to its natural number value -/
def digitToNat (d : Digit) : Nat :=
  match d with
  | Digit.zero => 0
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6
  | Digit.seven => 7
  | Digit.eight => 8
  | Digit.nine => 9

/-- The cross shape arrangement of digits -/
structure CrossArrangement :=
  (a b c d e f g : Digit)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
                   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
                   e ≠ f ∧ e ≠ g ∧
                   f ≠ g)
  (vertical_sum : digitToNat a + digitToNat b + digitToNat c = 25)
  (horizontal_sum : digitToNat d + digitToNat e + digitToNat f + digitToNat g = 17)

theorem cross_arrangement_sum (arr : CrossArrangement) :
  digitToNat arr.a + digitToNat arr.b + digitToNat arr.c +
  digitToNat arr.d + digitToNat arr.e + digitToNat arr.f + digitToNat arr.g = 33 :=
by sorry

end NUMINAMATH_CALUDE_cross_arrangement_sum_l3933_393310


namespace NUMINAMATH_CALUDE_add_inequality_preserves_order_l3933_393360

theorem add_inequality_preserves_order (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : a + c > b + d := by sorry

end NUMINAMATH_CALUDE_add_inequality_preserves_order_l3933_393360


namespace NUMINAMATH_CALUDE_shirt_cost_l3933_393380

theorem shirt_cost (j s : ℝ) 
  (eq1 : 3 * j + 2 * s = 69) 
  (eq2 : 2 * j + 3 * s = 81) : 
  s = 21 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l3933_393380


namespace NUMINAMATH_CALUDE_exponentiation_puzzle_l3933_393335

theorem exponentiation_puzzle : 3^(1^(0^2)) - ((3^1)^0)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponentiation_puzzle_l3933_393335


namespace NUMINAMATH_CALUDE_farm_animals_problem_l3933_393376

/-- Represents the farm animals problem --/
theorem farm_animals_problem (cows ducks pigs : ℕ) : 
  cows = 20 →
  ducks = (3 : ℕ) * cows / 2 →
  cows + ducks + pigs = 60 →
  pigs = (cows + ducks) / 5 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_problem_l3933_393376


namespace NUMINAMATH_CALUDE_tom_tickets_left_l3933_393397

/-- The number of tickets Tom has left after winning some and spending some -/
def tickets_left (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (spent_tickets : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - spent_tickets

/-- Theorem stating that Tom has 50 tickets left -/
theorem tom_tickets_left : tickets_left 32 25 7 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_tickets_left_l3933_393397


namespace NUMINAMATH_CALUDE_boxtimes_self_not_always_zero_l3933_393375

-- Define the ⊠ operation
def boxtimes (x y : ℝ) : ℝ := |x + y|

-- Statement to be proven false
theorem boxtimes_self_not_always_zero :
  ¬ (∀ x : ℝ, boxtimes x x = 0) := by
sorry

end NUMINAMATH_CALUDE_boxtimes_self_not_always_zero_l3933_393375


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l3933_393368

theorem divisibility_by_seven (a b : ℤ) : (10 * a + b) % 7 = 0 ↔ (a - 2 * b) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l3933_393368


namespace NUMINAMATH_CALUDE_pam_has_1200_apples_l3933_393388

/-- The number of apples in each of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- The number of Gerald's bags equivalent to one of Pam's bags -/
def gerald_to_pam_ratio : ℕ := 3

/-- The number of bags Pam has -/
def pams_bag_count : ℕ := 10

/-- The total number of apples Pam has -/
def pams_total_apples : ℕ := pams_bag_count * (gerald_to_pam_ratio * geralds_bag_count)

theorem pam_has_1200_apples : pams_total_apples = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pam_has_1200_apples_l3933_393388


namespace NUMINAMATH_CALUDE_percentage_less_than_y_l3933_393364

theorem percentage_less_than_y (y q w z : ℝ) 
  (hw : w = 0.6 * q) 
  (hq : q = 0.6 * y) 
  (hz : z = 1.5 * w) : 
  z = 0.54 * y := by sorry

end NUMINAMATH_CALUDE_percentage_less_than_y_l3933_393364


namespace NUMINAMATH_CALUDE_trouser_original_price_l3933_393320

/-- 
Given a trouser with a sale price of $70 after a 30% decrease,
prove that its original price was $100.
-/
theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 70 → 
  discount_percentage = 30 → 
  sale_price = (1 - discount_percentage / 100) * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_trouser_original_price_l3933_393320


namespace NUMINAMATH_CALUDE_frustum_max_volume_l3933_393394

/-- The maximum volume of a frustum within a sphere -/
theorem frustum_max_volume (r : ℝ) (r_top : ℝ) (r_bottom : ℝ) (h_r : r = 5) (h_top : r_top = 3) (h_bottom : r_bottom = 4) :
  ∃ v : ℝ, v = (259 / 3) * Real.pi ∧ 
  (∀ v' : ℝ, v' ≤ v ∧ 
    (∃ h : ℝ, v' = (1 / 3) * h * (r_top^2 * Real.pi + r_top * r_bottom * Real.pi + r_bottom^2 * Real.pi) ∧
              0 < h ∧ h ≤ 2 * (r^2 - r_top^2).sqrt)) :=
sorry

end NUMINAMATH_CALUDE_frustum_max_volume_l3933_393394


namespace NUMINAMATH_CALUDE_cottage_rental_cost_per_hour_l3933_393356

/-- Represents the cost of renting a cottage -/
structure CottageRental where
  hours : ℕ
  jack_payment : ℕ
  jill_payment : ℕ

/-- Calculates the cost per hour of renting a cottage -/
def cost_per_hour (rental : CottageRental) : ℚ :=
  (rental.jack_payment + rental.jill_payment : ℚ) / rental.hours

/-- Theorem: The cost per hour of the cottage rental is $5 -/
theorem cottage_rental_cost_per_hour :
  let rental := CottageRental.mk 8 20 20
  cost_per_hour rental = 5 := by
  sorry

end NUMINAMATH_CALUDE_cottage_rental_cost_per_hour_l3933_393356


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l3933_393325

/-- Given a square with perimeter 64 inches, prove that cutting a right triangle
    with hypotenuse equal to one side and translating it results in a new figure
    with perimeter 32 + 16√2 inches. -/
theorem square_cut_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 64) :
  let side_length : ℝ := square_perimeter / 4
  let triangle_leg : ℝ := side_length * Real.sqrt 2 / 2
  let new_perimeter : ℝ := 2 * side_length + 2 * triangle_leg
  new_perimeter = 32 + 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_square_cut_perimeter_l3933_393325


namespace NUMINAMATH_CALUDE_marbles_distribution_l3933_393345

theorem marbles_distribution (total_marbles : ℕ) (num_boys : ℕ) (marbles_per_boy : ℕ) :
  total_marbles = 35 →
  num_boys = 5 →
  marbles_per_boy = total_marbles / num_boys →
  marbles_per_boy = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3933_393345


namespace NUMINAMATH_CALUDE_least_common_multiple_addition_l3933_393338

theorem least_common_multiple_addition (a b c d : ℕ) (n m : ℕ) : 
  (∀ k : ℕ, k < m → ¬(a ∣ (n + k) ∧ b ∣ (n + k) ∧ c ∣ (n + k) ∧ d ∣ (n + k))) →
  (a ∣ (n + m) ∧ b ∣ (n + m) ∧ c ∣ (n + m) ∧ d ∣ (n + m)) →
  m = 7 ∧ n = 857 ∧ a = 24 ∧ b = 32 ∧ c = 36 ∧ d = 54 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_addition_l3933_393338


namespace NUMINAMATH_CALUDE_sum_of_next_five_even_integers_l3933_393333

theorem sum_of_next_five_even_integers (a : ℕ) (h : ∃ x : ℕ, a = 5 * x + 20 ∧ x > 0) :
  ∃ y : ℕ, y = a + 50 ∧ y = (5 * (x + 10) + 70) :=
sorry

end NUMINAMATH_CALUDE_sum_of_next_five_even_integers_l3933_393333


namespace NUMINAMATH_CALUDE_range_of_M_l3933_393318

theorem range_of_M (x y z : ℝ) (h1 : x + y + z = 30) (h2 : 3 * x + y - z = 50)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let M := 5 * x + 4 * y + 2 * z
  ∀ m, (m = M) → 120 ≤ m ∧ m ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_range_of_M_l3933_393318


namespace NUMINAMATH_CALUDE_payment_per_task_l3933_393322

/-- Calculates the payment per task for a contractor given their work schedule and total earnings -/
theorem payment_per_task (hours_per_task : ℝ) (hours_per_day : ℝ) (days_per_week : ℕ) (total_earnings : ℝ) :
  hours_per_task = 2 →
  hours_per_day = 10 →
  days_per_week = 5 →
  total_earnings = 1400 →
  (total_earnings / (days_per_week * (hours_per_day / hours_per_task))) = 56 := by
sorry

end NUMINAMATH_CALUDE_payment_per_task_l3933_393322


namespace NUMINAMATH_CALUDE_parabola_vertex_l3933_393321

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (3, 1)

/-- Theorem: The vertex coordinates of the parabola y = 2(x-3)^2 + 1 are (3, 1) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3933_393321


namespace NUMINAMATH_CALUDE_parallel_vectors_problem_l3933_393351

/-- Given two vectors a and b in R², where a is parallel to (2a + b), prove that the second component of b is 4 and m = 2. -/
theorem parallel_vectors_problem (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (m, 4) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • (2 • a + b) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_problem_l3933_393351


namespace NUMINAMATH_CALUDE_cars_produced_in_north_america_l3933_393378

theorem cars_produced_in_north_america :
  ∀ (total_cars europe_cars north_america_cars : ℕ),
    total_cars = 6755 →
    europe_cars = 2871 →
    total_cars = north_america_cars + europe_cars →
    north_america_cars = 3884 := by
  sorry

end NUMINAMATH_CALUDE_cars_produced_in_north_america_l3933_393378


namespace NUMINAMATH_CALUDE_larger_circle_radius_is_32_l3933_393353

/-- Two concentric circles with chord properties -/
structure ConcentricCircles where
  r : ℝ  -- radius of the smaller circle
  AB : ℝ  -- length of AB
  h_ratio : r > 0  -- radius is positive
  h_AB : AB = 16  -- given length of AB

/-- The radius of the larger circle in the concentric circles setup -/
def larger_circle_radius (c : ConcentricCircles) : ℝ := 4 * c.r

theorem larger_circle_radius_is_32 (c : ConcentricCircles) : 
  larger_circle_radius c = 32 := by
  sorry

#check larger_circle_radius_is_32

end NUMINAMATH_CALUDE_larger_circle_radius_is_32_l3933_393353


namespace NUMINAMATH_CALUDE_parallel_reasoning_is_deductive_l3933_393352

-- Define a type for lines
structure Line : Type :=
  (id : ℕ)

-- Define a parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the property of transitivity for parallel lines
axiom parallel_transitive : ∀ (x y z : Line), parallel x y → parallel y z → parallel x z

-- Given lines a, b, and c
variable (a b c : Line)

-- Given that a is parallel to b, and b is parallel to c
axiom a_parallel_b : parallel a b
axiom b_parallel_c : parallel b c

-- Define deductive reasoning
def is_deductive_reasoning (conclusion : Prop) : Prop := sorry

-- Theorem to prove
theorem parallel_reasoning_is_deductive : 
  is_deductive_reasoning (parallel a c) := sorry

end NUMINAMATH_CALUDE_parallel_reasoning_is_deductive_l3933_393352


namespace NUMINAMATH_CALUDE_root_implies_a_value_l3933_393326

theorem root_implies_a_value (a : ℝ) : (2 * (-1)^2 + a * (-1) - 1 = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_value_l3933_393326


namespace NUMINAMATH_CALUDE_mao_saying_moral_l3933_393361

/-- Represents the moral of a saying -/
inductive Moral
| KnowledgeDrivesPractice
| KnowledgeGuidesPractice
| PracticeSourceOfKnowledge
| PracticeSocialHistorical

/-- Represents a philosophical saying -/
structure Saying :=
(content : String)
(moral : Moral)

/-- Mao Zedong's saying about tasting a pear -/
def maoSaying : Saying :=
{ content := "If you want to know the taste of a pear, you must change the pear and taste it yourself",
  moral := Moral.PracticeSourceOfKnowledge }

/-- Theorem stating that the moral of Mao's saying is "Practice is the source of knowledge" -/
theorem mao_saying_moral :
  maoSaying.moral = Moral.PracticeSourceOfKnowledge :=
sorry

end NUMINAMATH_CALUDE_mao_saying_moral_l3933_393361


namespace NUMINAMATH_CALUDE_catch_up_point_l3933_393314

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℝ
  startTime : ℝ
  arrivalTime : ℝ

/-- The problem setup -/
def travelProblem (distanceAB : ℝ) (carA carB : Car) : Prop :=
  distanceAB > 0 ∧
  carA.startTime = carB.startTime + 1 ∧
  carA.arrivalTime + 1 = carB.arrivalTime ∧
  distanceAB = carA.speed * (carA.arrivalTime - carA.startTime) ∧
  distanceAB = carB.speed * (carB.arrivalTime - carB.startTime)

/-- The theorem to be proved -/
theorem catch_up_point (distanceAB : ℝ) (carA carB : Car) 
  (h : travelProblem distanceAB carA carB) : 
  ∃ (t : ℝ), carA.speed * (t - carA.startTime) = carB.speed * (t - carB.startTime) ∧ 
              carA.speed * (t - carA.startTime) = distanceAB - 150 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_point_l3933_393314


namespace NUMINAMATH_CALUDE_det_AB_eq_one_l3933_393387

open Matrix

variable {n : ℕ}

theorem det_AB_eq_one
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (hA : IsUnit A)
  (hB : IsUnit B)
  (h : (A + B⁻¹)⁻¹ = A⁻¹ + B) :
  det (A * B) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_AB_eq_one_l3933_393387


namespace NUMINAMATH_CALUDE_four_friends_same_group_probability_l3933_393395

/-- The total number of students -/
def total_students : ℕ := 900

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The number of students in each group -/
def students_per_group : ℕ := total_students / num_groups

/-- The probability of a single student being assigned to a specific group -/
def prob_single_student : ℚ := 1 / num_groups

theorem four_friends_same_group_probability :
  (prob_single_student ^ 3 : ℚ) = 1 / 64 :=
sorry

end NUMINAMATH_CALUDE_four_friends_same_group_probability_l3933_393395


namespace NUMINAMATH_CALUDE_triangle_angle_sum_triangle_angle_sum_is_540_l3933_393316

theorem triangle_angle_sum : ℝ → Prop :=
  fun total_sum =>
    ∃ (int_angles ext_angles : ℝ),
      (int_angles = 180) ∧
      (ext_angles = 360) ∧
      (total_sum = int_angles + ext_angles)

theorem triangle_angle_sum_is_540 : 
  triangle_angle_sum 540 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_triangle_angle_sum_is_540_l3933_393316


namespace NUMINAMATH_CALUDE_product_of_binary_numbers_l3933_393348

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

theorem product_of_binary_numbers :
  let a := [true, true, false, false, true, true]
  let b := [true, true, false, true]
  let result := [true, false, false, true, true, false, false, false, true, false, true]
  binary_to_nat a * binary_to_nat b = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_numbers_l3933_393348


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3933_393399

theorem expression_simplification_and_evaluation :
  let a : ℤ := -1
  let b : ℤ := -2
  let original_expression := (2*a + b)*(b - 2*a) - (a - 3*b)^2
  let simplified_expression := -5*a^2 + 6*a*b - 8*b^2
  original_expression = simplified_expression ∧ simplified_expression = -25 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3933_393399


namespace NUMINAMATH_CALUDE_gummies_cost_gummies_cost_proof_l3933_393350

theorem gummies_cost (lollipop_count : ℕ) (lollipop_price : ℚ) 
                      (gummies_count : ℕ) (initial_amount : ℚ) 
                      (remaining_amount : ℚ) : ℚ :=
  let total_spent := initial_amount - remaining_amount
  let lollipop_total := ↑lollipop_count * lollipop_price
  let gummies_total := total_spent - lollipop_total
  gummies_total / ↑gummies_count

#check gummies_cost 4 (3/2) 2 15 5 = 2

theorem gummies_cost_proof :
  gummies_cost 4 (3/2) 2 15 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gummies_cost_gummies_cost_proof_l3933_393350


namespace NUMINAMATH_CALUDE_interview_pass_probability_l3933_393391

-- Define the number of questions
def num_questions : ℕ := 3

-- Define the probability of answering a question correctly
def prob_correct : ℝ := 0.7

-- Define the number of attempts per question
def num_attempts : ℕ := 3

-- Theorem statement
theorem interview_pass_probability :
  1 - (1 - prob_correct) ^ num_attempts = 0.973 := by
  sorry

end NUMINAMATH_CALUDE_interview_pass_probability_l3933_393391


namespace NUMINAMATH_CALUDE_bob_probability_after_three_turns_l3933_393377

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The game state after a certain number of turns -/
structure GameState :=
  (current_player : Player)
  (turn : ℕ)

/-- The probability of a player having the ball after a certain number of turns -/
def probability_has_ball (player : Player) (turns : ℕ) : ℚ :=
  sorry

theorem bob_probability_after_three_turns :
  probability_has_ball Player.Bob 3 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_bob_probability_after_three_turns_l3933_393377


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3933_393355

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7*I
  let z₂ : ℂ := 4 - 7*I
  (z₁ / z₂) - (z₂ / z₁) = 112 * I / 65 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3933_393355


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3933_393381

theorem smaller_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 3 / 5 → x + y + 10 = 50 → min x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3933_393381


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3933_393385

/-- A trinomial is a perfect square if it can be expressed as (x - a)^2 for some real number a -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (x - k)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l3933_393385


namespace NUMINAMATH_CALUDE_sweet_bitter_fruits_problem_l3933_393324

/-- Represents the problem of buying sweet and bitter fruits --/
theorem sweet_bitter_fruits_problem 
  (x y : ℕ) -- x is the number of sweet fruits, y is the number of bitter fruits
  (h1 : x + y = 99) -- total number of fruits
  (h2 : 3 * x + (1/3) * y = 97) -- total cost in wen
  : 
  -- The system of equations correctly represents the problem
  (x + y = 99 ∧ 3 * x + (1/3) * y = 97) := by
  sorry


end NUMINAMATH_CALUDE_sweet_bitter_fruits_problem_l3933_393324


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3933_393362

theorem arithmetic_evaluation : (7 + 5 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3933_393362


namespace NUMINAMATH_CALUDE_sqrt_180_simplification_l3933_393339

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplification_l3933_393339


namespace NUMINAMATH_CALUDE_christine_wandering_time_l3933_393366

/-- Given a distance of 20 miles and a speed of 4 miles per hour, 
    the time taken is 5 hours. -/
theorem christine_wandering_time :
  let distance : ℝ := 20
  let speed : ℝ := 4
  let time := distance / speed
  time = 5 := by sorry

end NUMINAMATH_CALUDE_christine_wandering_time_l3933_393366


namespace NUMINAMATH_CALUDE_gas_volume_calculation_l3933_393301

/-- Calculate the volume of gas using the Mendeleev-Clapeyron equation -/
theorem gas_volume_calculation (m R T p M : ℝ) (h_m : m = 140) (h_R : R = 8.314) 
  (h_T : T = 305) (h_p : p = 283710) (h_M : M = 28) :
  let V := (m * R * T * 1000) / (p * M)
  ∃ ε > 0, |V - 44.7| < ε :=
sorry

end NUMINAMATH_CALUDE_gas_volume_calculation_l3933_393301


namespace NUMINAMATH_CALUDE_sum_difference_equals_3146_main_theorem_l3933_393393

theorem sum_difference_equals_3146 : ℕ → Prop :=
  fun n =>
    let even_sum := n * (n + 1)
    let multiples_of_3_sum := (n / 3) * ((n / 3) + 1) * 3 / 2
    let odd_sum := ((n - 1) / 2 + 1) ^ 2
    (even_sum - multiples_of_3_sum - odd_sum = 3146) ∧ (2 * n = 400)

theorem main_theorem : ∃ n : ℕ, sum_difference_equals_3146 n := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equals_3146_main_theorem_l3933_393393


namespace NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l3933_393311

/-- 
Given a point P with coordinates (2, -5), 
prove that its symmetric point P' with respect to the origin has coordinates (-2, 5).
-/
theorem symmetric_point_wrt_origin : 
  let P : ℝ × ℝ := (2, -5)
  let P' : ℝ × ℝ := (-P.1, -P.2)
  P' = (-2, 5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_wrt_origin_l3933_393311


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_l3933_393308

def is_prime (n : ℕ) : Prop := sorry

def is_square (n : ℕ) : Prop := sorry

def has_prime_factor_less_than (n k : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square : 
  ∀ n : ℕ, n < 3599 → 
    (is_prime n ∨ is_square n ∨ has_prime_factor_less_than n 55) ∧
    (¬ is_prime 3599 ∧ ¬ is_square 3599 ∧ ¬ has_prime_factor_less_than 3599 55) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_l3933_393308


namespace NUMINAMATH_CALUDE_problem_solution_l3933_393306

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x/3 = y^2) 
  (h3 : x/5 = 5*y + 2) : 
  x = (685 + 25 * Real.sqrt 745) / 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3933_393306
