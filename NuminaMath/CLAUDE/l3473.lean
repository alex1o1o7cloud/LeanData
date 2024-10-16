import Mathlib

namespace NUMINAMATH_CALUDE_longest_perimeter_l3473_347301

theorem longest_perimeter (x : ℝ) 
  (hx : x > 1)
  (perimeterA : ℝ := 4 + 6*x)
  (perimeterB : ℝ := 2 + 10*x)
  (perimeterC : ℝ := 7 + 5*x)
  (perimeterD : ℝ := 6 + 6*x)
  (perimeterE : ℝ := 1 + 11*x) :
  perimeterE > perimeterA ∧ 
  perimeterE > perimeterB ∧ 
  perimeterE > perimeterC ∧ 
  perimeterE > perimeterD :=
by
  sorry

end NUMINAMATH_CALUDE_longest_perimeter_l3473_347301


namespace NUMINAMATH_CALUDE_fraction_addition_theorem_l3473_347345

theorem fraction_addition_theorem : (5 * 8) / 10 + 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_theorem_l3473_347345


namespace NUMINAMATH_CALUDE_necessary_condition_for_positive_linear_function_l3473_347341

theorem necessary_condition_for_positive_linear_function
  (a b : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a * x + b)
  (h_positive : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x > 0) :
  a + 2 * b > 0 :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_for_positive_linear_function_l3473_347341


namespace NUMINAMATH_CALUDE_range_of_g_l3473_347344

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x - 5

-- Define the function g as a composition of f
def g (x : ℝ) : ℝ := f (f (f x))

-- Theorem statement
theorem range_of_g :
  ∀ y : ℝ, (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ g x = y) ↔ -41 ≤ y ∧ y ≤ 87 :=
sorry

end NUMINAMATH_CALUDE_range_of_g_l3473_347344


namespace NUMINAMATH_CALUDE_candy_distribution_l3473_347370

theorem candy_distribution (C n : ℕ) 
  (h1 : C = 8 * n + 4)
  (h2 : C = 11 * (n - 1)) : 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3473_347370


namespace NUMINAMATH_CALUDE_train_speed_l3473_347331

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 300) (h2 : time = 10) :
  length / time = 30 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3473_347331


namespace NUMINAMATH_CALUDE_closest_point_l3473_347337

def v (t : ℝ) : Fin 3 → ℝ := fun i => 
  match i with
  | 0 => 3 + 8*t
  | 1 => -1 + 2*t
  | 2 => -2 - 3*t

def a : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1
  | 1 => 7
  | 2 => 1

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 8
  | 1 => 2
  | 2 => -3

theorem closest_point (t : ℝ) : 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) ↔ t = -1/7 := by sorry

end NUMINAMATH_CALUDE_closest_point_l3473_347337


namespace NUMINAMATH_CALUDE_duty_roster_theorem_l3473_347348

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of arrangements where two specific people are adjacent -/
def adjacent_arrangements (n : ℕ) : ℕ := 2 * permutations (n - 1)

/-- The number of arrangements where both pairs of specific people are adjacent -/
def both_adjacent_arrangements (n : ℕ) : ℕ := 2 * 2 * permutations (n - 2)

/-- The number of valid arrangements for the duty roster problem -/
def duty_roster_arrangements (n : ℕ) : ℕ :=
  permutations n - 2 * adjacent_arrangements n + both_adjacent_arrangements n

theorem duty_roster_theorem :
  duty_roster_arrangements 6 = 336 := by sorry

end NUMINAMATH_CALUDE_duty_roster_theorem_l3473_347348


namespace NUMINAMATH_CALUDE_cookie_number_proof_l3473_347377

/-- The smallest positive integer satisfying the given conditions -/
def smallest_cookie_number : ℕ := 2549

/-- Proof that the smallest_cookie_number satisfies all conditions -/
theorem cookie_number_proof :
  smallest_cookie_number % 6 = 5 ∧
  smallest_cookie_number % 8 = 6 ∧
  smallest_cookie_number % 10 = 9 ∧
  ∃ k : ℕ, k * k = smallest_cookie_number ∧
  ∀ n : ℕ, n > 0 ∧ n < smallest_cookie_number →
    ¬(n % 6 = 5 ∧ n % 8 = 6 ∧ n % 10 = 9 ∧ ∃ m : ℕ, m * m = n) :=
by sorry

end NUMINAMATH_CALUDE_cookie_number_proof_l3473_347377


namespace NUMINAMATH_CALUDE_unique_prime_sum_digits_l3473_347336

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Primality test -/
def isPrime (n : ℕ) : Prop := sorry

theorem unique_prime_sum_digits : 
  ∃! (n : ℕ), isPrime n ∧ n + S n + S (S n) = 3005 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_digits_l3473_347336


namespace NUMINAMATH_CALUDE_units_digit_17_31_l3473_347393

theorem units_digit_17_31 : (17^31) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_31_l3473_347393


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3473_347397

/-- Expresses the repeating decimal 0.7̄8̄ as a rational number -/
theorem repeating_decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ (0.7 + 0.08 / (1 - 0.1) : ℚ) = n / d ∧ n = 781 ∧ d = 990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3473_347397


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l3473_347343

/-- Represents a 4x4x4 cube composed of unit cubes -/
structure LargeCube where
  size : ℕ
  size_eq : size = 4

/-- Represents a plane intersecting the large cube -/
structure IntersectingPlane where
  cube : LargeCube
  ratio : ℚ
  ratio_eq : ratio = 1 / 3

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (plane : IntersectingPlane) : ℕ := sorry

/-- Theorem stating that the plane intersects 32 unit cubes -/
theorem intersected_cubes_count (plane : IntersectingPlane) : 
  count_intersected_cubes plane = 32 := by sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l3473_347343


namespace NUMINAMATH_CALUDE_tan_sum_difference_pi_fourth_l3473_347300

theorem tan_sum_difference_pi_fourth (a : ℝ) : 
  Real.tan (a + π/4) - Real.tan (a - π/4) = 2 * Real.tan (2*a) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_difference_pi_fourth_l3473_347300


namespace NUMINAMATH_CALUDE_eight_digit_increasing_count_l3473_347388

theorem eight_digit_increasing_count : ∃ M : ℕ, 
  (M = Nat.choose 7 5) ∧ 
  (M % 1000 = 21) := by sorry

end NUMINAMATH_CALUDE_eight_digit_increasing_count_l3473_347388


namespace NUMINAMATH_CALUDE_number_multiplied_by_four_twice_l3473_347386

theorem number_multiplied_by_four_twice : ∃ x : ℝ, (4 * (4 * x) = 32) ∧ (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_number_multiplied_by_four_twice_l3473_347386


namespace NUMINAMATH_CALUDE_triangle_area_l3473_347378

theorem triangle_area (A B C : EuclideanSpace ℝ (Fin 2)) :
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let angle_A : ℝ := π / 4
  let area : ℝ := (1 / 2) * b * c * Real.sin angle_A
  area = Real.sqrt 6 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3473_347378


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3473_347383

/-- Given a triangle DEF where ∠E is congruent to ∠F, the measure of ∠F is three times 
    the measure of ∠D, and ∠D is one-third the measure of ∠E, 
    prove that the measure of ∠E is 540/7 degrees. -/
theorem triangle_angle_measure (D E F : ℝ) : 
  D > 0 → E > 0 → F > 0 →  -- Angles are positive
  D + E + F = 180 →  -- Sum of angles in a triangle
  E = F →  -- ∠E is congruent to ∠F
  F = 3 * D →  -- Measure of ∠F is three times the measure of ∠D
  D = E / 3 →  -- ∠D is one-third the measure of ∠E
  E = 540 / 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3473_347383


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3473_347373

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (F₁ F₂ P : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the hyperbola
    (F₁.1 < 0 ∧ F₁.2 = 0) ∧ 
    (F₂.1 > 0 ∧ F₂.2 = 0) ∧ 
    -- P is on the hyperbola in the first quadrant
    (P.1 > 0 ∧ P.2 > 0) ∧
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    -- P is on the circle with center O and radius OF₁
    (P.1^2 + P.2^2 = F₁.1^2) ∧
    -- The area of triangle PF₁F₂ is a²
    (abs (P.1 * F₂.1 - P.2 * F₂.2) / 2 = a^2) →
    -- The eccentricity is √2
    ((F₂.1 - F₁.1) / 2) / a = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3473_347373


namespace NUMINAMATH_CALUDE_lemonade_sales_l3473_347390

theorem lemonade_sales (katya ricky tina : ℕ) : 
  ricky = 9 →
  tina = 2 * (katya + ricky) →
  tina = katya + 26 →
  katya = 8 := by sorry

end NUMINAMATH_CALUDE_lemonade_sales_l3473_347390


namespace NUMINAMATH_CALUDE_c_is_largest_l3473_347376

-- Define the numbers as real numbers
def a : ℝ := 7.25678
def b : ℝ := 7.256777777777777 -- Approximation of 7.256̄7
def c : ℝ := 7.257676767676767 -- Approximation of 7.25̄76
def d : ℝ := 7.275675675675675 -- Approximation of 7.2̄756
def e : ℝ := 7.275627562756275 -- Approximation of 7.̄2756

-- Theorem stating that c (7.25̄76) is the largest
theorem c_is_largest : 
  c > a ∧ c > b ∧ c > d ∧ c > e :=
sorry

end NUMINAMATH_CALUDE_c_is_largest_l3473_347376


namespace NUMINAMATH_CALUDE_domain_of_w_l3473_347339

-- Define the function w(y)
def w (y : ℝ) : ℝ := (y - 3) ^ (1/3) + (15 - y) ^ (1/3)

-- State the theorem about the domain of w
theorem domain_of_w :
  ∀ y : ℝ, ∃ z : ℝ, w y = z :=
sorry

end NUMINAMATH_CALUDE_domain_of_w_l3473_347339


namespace NUMINAMATH_CALUDE_candy_expenditure_l3473_347354

theorem candy_expenditure (total_spent : ℚ) :
  total_spent = 75 →
  (1 / 2 : ℚ) + (1 / 3 : ℚ) + (1 / 10 : ℚ) + (candy_fraction : ℚ) = 1 →
  candy_fraction * total_spent = 5 :=
by sorry

end NUMINAMATH_CALUDE_candy_expenditure_l3473_347354


namespace NUMINAMATH_CALUDE_digit_sum_in_multiplication_l3473_347380

theorem digit_sum_in_multiplication (c d : ℕ) : 
  c < 10 → d < 10 → 
  (30 + c) * (10 * d + 5) = 185 →
  5 * c = 15 →
  c + d = 11 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_in_multiplication_l3473_347380


namespace NUMINAMATH_CALUDE_total_discount_savings_l3473_347329

def mangoes_per_box : ℕ := 10 * 12 -- 10 dozen

def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def total_boxes : ℕ := 36

def discount_rate (boxes : ℕ) : ℚ :=
  if boxes ≥ 30 then 15 / 100
  else if boxes ≥ 20 then 10 / 100
  else if boxes ≥ 10 then 5 / 100
  else 0

theorem total_discount_savings : 
  let total_cost := (prices_per_dozen.map (· * mangoes_per_box)).sum * total_boxes
  let discounted_cost := total_cost * (1 - discount_rate total_boxes)
  total_cost - discounted_cost = 5090 := by
  sorry

end NUMINAMATH_CALUDE_total_discount_savings_l3473_347329


namespace NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l3473_347313

/-- 
Given an arithmetic sequence with the first three terms 2/3, y-2, and 4y+1,
prove that y = -17/6.
-/
theorem arithmetic_sequence_y_value :
  ∀ y : ℚ,
  let a₁ : ℚ := 2/3
  let a₂ : ℚ := y - 2
  let a₃ : ℚ := 4*y + 1
  (a₂ - a₁ = a₃ - a₂) →
  y = -17/6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_y_value_l3473_347313


namespace NUMINAMATH_CALUDE_G_of_4_f_2_l3473_347340

-- Define the functions f and G
def f (a : ℝ) : ℝ := a^2 - 3
def G (a b : ℝ) : ℝ := b^2 - a

-- State the theorem
theorem G_of_4_f_2 : G 4 (f 2) = -3 := by sorry

end NUMINAMATH_CALUDE_G_of_4_f_2_l3473_347340


namespace NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2_l3473_347392

theorem unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2 :
  ∃! n : ℕ+, 24 ∣ n ∧ 8 < (n : ℝ) ^ (1/3) ∧ (n : ℝ) ^ (1/3) < 8.2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_divisible_by_24_with_cube_root_between_8_and_8_2_l3473_347392


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3473_347307

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x^2 + 4*x) = 9 ∧ x^2 + 4*x ≥ 0 :=
by
  -- The unique solution is x = -2 + √85
  use -2 + Real.sqrt 85
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3473_347307


namespace NUMINAMATH_CALUDE_perpendicular_distance_to_adjacent_plane_l3473_347374

/-- A rectangular parallelepiped with dimensions 5 × 5 × 4 -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 5
  width_eq : width = 5
  height_eq : height = 4

/-- A vertex of the parallelepiped -/
structure Vertex where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The perpendicular distance from a vertex to a plane -/
def perpendicularDistance (v : Vertex) (plane : Vertex → Vertex → Vertex → Prop) : ℝ :=
  sorry

theorem perpendicular_distance_to_adjacent_plane (p : Parallelepiped) 
  (h v1 v2 v3 : Vertex) 
  (adj1 : h.z = 0)
  (adj2 : v1.z = 0 ∧ v1.y = 0 ∧ v1.x = p.length)
  (adj3 : v2.z = 0 ∧ v2.x = 0 ∧ v2.y = p.width)
  (adj4 : v3.x = 0 ∧ v3.y = 0 ∧ v3.z = p.height) :
  perpendicularDistance h (fun a b c => True) = 4 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_distance_to_adjacent_plane_l3473_347374


namespace NUMINAMATH_CALUDE_max_attendance_l3473_347350

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Diana

-- Define the availability function
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => true
  | Person.Diana, Day.Monday => true
  | Person.Diana, Day.Tuesday => true
  | Person.Diana, Day.Wednesday => false
  | Person.Diana, Day.Thursday => true
  | Person.Diana, Day.Friday => false

-- Define the function to count available people on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Diana]).length

-- Theorem statement
theorem max_attendance :
  (∀ d : Day, countAvailable d ≤ 2) ∧
  (countAvailable Day.Monday = 2) ∧
  (countAvailable Day.Tuesday = 2) ∧
  (countAvailable Day.Wednesday = 2) ∧
  (countAvailable Day.Thursday < 2) ∧
  (countAvailable Day.Friday < 2) :=
sorry

end NUMINAMATH_CALUDE_max_attendance_l3473_347350


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3473_347316

theorem repeating_decimal_to_fraction : ∃ (x : ℚ), x = 4/11 ∧ (∀ (n : ℕ), x = (36 * (100^n - 1)) / (99 * 100^n)) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3473_347316


namespace NUMINAMATH_CALUDE_marble_ratio_l3473_347302

theorem marble_ratio (selma_marbles merill_marbles elliot_marbles : ℕ) : 
  selma_marbles = 50 →
  merill_marbles = 30 →
  merill_marbles + elliot_marbles = selma_marbles - 5 →
  merill_marbles / elliot_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l3473_347302


namespace NUMINAMATH_CALUDE_eggs_per_student_l3473_347338

theorem eggs_per_student (total_eggs : ℕ) (num_students : ℕ) (eggs_per_student : ℕ)
  (h1 : total_eggs = 56)
  (h2 : num_students = 7)
  (h3 : total_eggs = num_students * eggs_per_student) :
  eggs_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_student_l3473_347338


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3473_347308

/-- An isosceles triangle with a median dividing the perimeter into two parts -/
structure IsoscelesTriangleWithMedian where
  /-- Length of each equal side -/
  a : ℝ
  /-- Length of the base -/
  b : ℝ
  /-- The median divides the perimeter into two parts -/
  part1 : ℝ
  part2 : ℝ
  /-- Conditions for isosceles triangle with median -/
  is_isosceles : a > 0
  valid_base : b > 0
  valid_parts : part1 > 0 ∧ part2 > 0
  perimeter_division : (2 * a + b) = (part1 + part2)
  median_property : 2 * a + b / 2 = max part1 part2 ∧ a + b = min part1 part2

/-- The theorem stating the possible side lengths of the isosceles triangle -/
theorem isosceles_triangle_side_lengths 
  (t : IsoscelesTriangleWithMedian) 
  (h : t.part1 = 14 ∧ t.part2 = 18 ∨ t.part1 = 18 ∧ t.part2 = 14) : 
  (t.a = 28/3 ∧ t.b = 40/3) ∨ (t.a = 12 ∧ t.b = 8) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l3473_347308


namespace NUMINAMATH_CALUDE_sum_M_l3473_347387

def M : ℕ → ℕ
  | 0 => 0
  | 1 => 4^2 - 2^2
  | (n+2) => (2*n+5)^2 + (2*n+4)^2 - (2*n+3)^2 - (2*n+2)^2 + M n

theorem sum_M : M 50 = 5304 := by
  sorry

end NUMINAMATH_CALUDE_sum_M_l3473_347387


namespace NUMINAMATH_CALUDE_charlie_has_32_cards_l3473_347333

/-- The number of soccer cards Chris has -/
def chris_cards : ℕ := 18

/-- The difference in cards between Charlie and Chris -/
def card_difference : ℕ := 14

/-- Charlie's number of soccer cards -/
def charlie_cards : ℕ := chris_cards + card_difference

/-- Theorem stating that Charlie has 32 soccer cards -/
theorem charlie_has_32_cards : charlie_cards = 32 := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_32_cards_l3473_347333


namespace NUMINAMATH_CALUDE_divisible_by_four_sum_consecutive_odds_l3473_347353

theorem divisible_by_four_sum_consecutive_odds (a : ℤ) : ∃ (x y : ℤ), 
  4 * a = x + y ∧ Odd x ∧ Odd y ∧ y = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_sum_consecutive_odds_l3473_347353


namespace NUMINAMATH_CALUDE_first_meeting_turns_l3473_347304

/-- The number of points on the circle -/
def n : ℕ := 15

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- The relative clockwise movement per turn -/
def relative_move : ℕ := alice_move - (n - bob_move)

theorem first_meeting_turns : 
  (∃ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0) → 
  (∃ m : ℕ, m > 0 ∧ (m * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → l ≥ m) →
  (∃ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → k ≤ l) →
  (∀ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → k ≤ l) → k = 5 := by
  sorry

#eval relative_move -- Should output 3

end NUMINAMATH_CALUDE_first_meeting_turns_l3473_347304


namespace NUMINAMATH_CALUDE_parabola_point_distance_l3473_347398

theorem parabola_point_distance (m n : ℝ) : 
  n^2 = 4*m →                             -- P(m,n) is on the parabola y^2 = 4x
  (m + 1)^2 = (m - 5)^2 + n^2 →           -- Distance from P to x=-1 equals distance from P to A(5,0)
  m = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l3473_347398


namespace NUMINAMATH_CALUDE_boys_from_clay_l3473_347347

/-- Represents the number of students from each school and gender --/
structure StudentCounts where
  total : Nat
  boys : Nat
  girls : Nat
  jonas : Nat
  clay : Nat
  pine : Nat
  jonasGirls : Nat
  pineBoys : Nat

/-- Theorem stating that the number of boys from Clay Middle School is 40 --/
theorem boys_from_clay (s : StudentCounts)
  (h_total : s.total = 120)
  (h_boys : s.boys = 70)
  (h_girls : s.girls = 50)
  (h_jonas : s.jonas = 50)
  (h_clay : s.clay = 40)
  (h_pine : s.pine = 30)
  (h_jonasGirls : s.jonasGirls = 30)
  (h_pineBoys : s.pineBoys = 10)
  : s.clay - (s.girls - s.jonasGirls - (s.pine - s.pineBoys)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_from_clay_l3473_347347


namespace NUMINAMATH_CALUDE_union_equals_A_l3473_347349

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equals_A (m : ℝ) : 
  (A m ∪ B m = A m) → (m = 0 ∨ m = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3473_347349


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3473_347328

theorem division_remainder_proof :
  let dividend : ℕ := 165
  let divisor : ℕ := 18
  let quotient : ℕ := 9
  let remainder : ℕ := dividend - divisor * quotient
  remainder = 3 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3473_347328


namespace NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l3473_347323

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic sequence. -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_diff * (n - 1 : ℝ)

theorem eighth_term_of_specific_sequence :
  ∃ (seq : ArithmeticSequence),
    nth_term seq 4 = 23 ∧
    nth_term seq 6 = 47 ∧
    nth_term seq 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_specific_sequence_l3473_347323


namespace NUMINAMATH_CALUDE_intersection_when_m_is_5_subset_condition_l3473_347358

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

-- Theorem for part 1
theorem intersection_when_m_is_5 :
  A 5 ∩ B = {x | 6 ≤ x ∧ x ≤ 10} := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ m : ℝ, A m ⊆ B ↔ m ≤ 11/3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_5_subset_condition_l3473_347358


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l3473_347309

def number : ℝ := 3120000

theorem scientific_notation_proof :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ number = a * (10 : ℝ) ^ n ∧ a = 3.12 ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l3473_347309


namespace NUMINAMATH_CALUDE_roots_expression_l3473_347314

theorem roots_expression (r s : ℝ) (u v s t : ℂ) : 
  (u^2 + r*u + 1 = 0) → 
  (v^2 + r*v + 1 = 0) → 
  (s^2 + s*s + 1 = 0) → 
  (t^2 + s*t + 1 = 0) → 
  (u - s)*(v - s)*(u + t)*(v + t) = s^2 - r^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l3473_347314


namespace NUMINAMATH_CALUDE_roes_speed_l3473_347371

/-- Proves that Roe's speed is 40 miles per hour given the conditions of the problem -/
theorem roes_speed (teena_speed : ℝ) (initial_distance : ℝ) (time : ℝ) (final_distance : ℝ)
  (h1 : teena_speed = 55)
  (h2 : initial_distance = 7.5)
  (h3 : time = 1.5)
  (h4 : final_distance = 15)
  (h5 : teena_speed * time - initial_distance - final_distance = time * roe_speed) :
  roe_speed = 40 :=
by sorry

end NUMINAMATH_CALUDE_roes_speed_l3473_347371


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l3473_347355

/-- Given two cubic polynomials with two distinct common roots, prove that a = 7 and b = 8 -/
theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧
    (r^3 + a*r^2 + 13*r + 10 = 0) ∧
    (r^3 + b*r^2 + 16*r + 12 = 0) ∧
    (s^3 + a*s^2 + 13*s + 10 = 0) ∧
    (s^3 + b*s^2 + 16*s + 12 = 0)) →
  a = 7 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l3473_347355


namespace NUMINAMATH_CALUDE_expression_simplification_l3473_347382

theorem expression_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 - 4*x + 1) / ((x - 1) * (x + 3)) - (6*x - 5) / ((x - 1) * (x + 3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3473_347382


namespace NUMINAMATH_CALUDE_second_number_is_fifteen_l3473_347325

def has_exactly_three_common_factors_with_15 (n : ℕ) : Prop :=
  ∃ (f₁ f₂ f₃ : ℕ), 
    f₁ ≠ f₂ ∧ f₁ ≠ f₃ ∧ f₂ ≠ f₃ ∧
    f₁ > 1 ∧ f₂ > 1 ∧ f₃ > 1 ∧
    f₁ ∣ 15 ∧ f₂ ∣ 15 ∧ f₃ ∣ 15 ∧
    f₁ ∣ n ∧ f₂ ∣ n ∧ f₃ ∣ n ∧
    ∀ (k : ℕ), k > 1 → k ∣ 15 → k ∣ n → (k = f₁ ∨ k = f₂ ∨ k = f₃)

theorem second_number_is_fifteen (n : ℕ) 
  (h : has_exactly_three_common_factors_with_15 n)
  (h3 : 3 ∣ n) (h5 : 5 ∣ n) (h15 : 15 ∣ n) : n = 15 :=
by sorry

end NUMINAMATH_CALUDE_second_number_is_fifteen_l3473_347325


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3473_347394

theorem sin_alpha_value (α : Real) (h : Real.tan α = 3/4) : Real.sin α = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3473_347394


namespace NUMINAMATH_CALUDE_chord_intersection_probability_l3473_347399

/-- The probability that a chord intersects the inner circle when two points are chosen randomly
    on the outer circle of two concentric circles with radii 2 and 3 -/
theorem chord_intersection_probability (r₁ r₂ : ℝ) (h₁ : r₁ = 2) (h₂ : r₂ = 3) :
  let θ := 2 * Real.arctan (r₁ / Real.sqrt (r₂^2 - r₁^2))
  (θ / (2 * Real.pi)) = 0.2148 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_l3473_347399


namespace NUMINAMATH_CALUDE_all_triangles_present_l3473_347320

/-- A permissible triangle with angles represented as integers -/
structure PermissibleTriangle (p : ℕ) :=
  (a b c : ℕ)
  (sum_eq_p : a + b + c = p)
  (all_pos : 0 < a ∧ 0 < b ∧ 0 < c)

/-- The set of all permissible triangles for a given prime p -/
def AllPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | true}

/-- A function representing the division process -/
def DivideTriangle (p : ℕ) (t : PermissibleTriangle p) : Option (PermissibleTriangle p × PermissibleTriangle p) :=
  sorry

/-- The set of triangles after the division process is complete -/
def FinalTriangleSet (p : ℕ) : Set (PermissibleTriangle p) :=
  sorry

/-- The main theorem -/
theorem all_triangles_present (p : ℕ) (hp : Prime p) :
  FinalTriangleSet p = AllPermissibleTriangles p :=
sorry

end NUMINAMATH_CALUDE_all_triangles_present_l3473_347320


namespace NUMINAMATH_CALUDE_square_of_nine_ones_l3473_347318

theorem square_of_nine_ones : (111111111 : ℕ)^2 = 12345678987654321 := by
  sorry

end NUMINAMATH_CALUDE_square_of_nine_ones_l3473_347318


namespace NUMINAMATH_CALUDE_fair_coin_probability_l3473_347322

-- Define a fair coin
def fair_coin := { p : ℝ | 0 ≤ p ∧ p ≤ 1 ∧ p = 1 - p }

-- Define the number of tosses
def num_tosses : ℕ := 20

-- Define the number of heads
def num_heads : ℕ := 8

-- Define the number of tails
def num_tails : ℕ := 12

-- Theorem statement
theorem fair_coin_probability : 
  ∀ (p : ℝ), p ∈ fair_coin → p = 1/2 :=
sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l3473_347322


namespace NUMINAMATH_CALUDE_dimes_spent_l3473_347315

/-- Given Joan's initial and remaining dimes, calculate the number of dimes spent. -/
theorem dimes_spent (initial : ℕ) (remaining : ℕ) (h : remaining ≤ initial) :
  initial - remaining = initial - remaining :=
by sorry

end NUMINAMATH_CALUDE_dimes_spent_l3473_347315


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l3473_347346

/-- The number of ways to distribute n identical objects among k distinct groups,
    where each group receives at least one object. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- The number of ways to distribute 8 pencils among 4 friends,
    where each friend receives at least one pencil. -/
def pencil_distribution : ℕ :=
  distribute_with_minimum 8 4

theorem pencil_distribution_ways : pencil_distribution = 35 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_ways_l3473_347346


namespace NUMINAMATH_CALUDE_max_correct_is_38_l3473_347321

/-- Represents the scoring system and results of a math contest. -/
structure MathContest where
  total_questions : ℕ
  correct_points : ℤ
  blank_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions for a given contest. -/
def max_correct_answers (contest : MathContest) : ℕ :=
  sorry

/-- Theorem stating that for the given contest parameters, the maximum number of correct answers is 38. -/
theorem max_correct_is_38 :
  let contest := MathContest.mk 60 5 0 (-2) 150
  max_correct_answers contest = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_is_38_l3473_347321


namespace NUMINAMATH_CALUDE_monotonic_function_theorem_l3473_347330

/-- A monotonic function is either non-increasing or non-decreasing --/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f x ≥ f y)

/-- The main theorem --/
theorem monotonic_function_theorem (f : ℝ → ℝ) (hf : Monotonic f)
    (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) :
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = -x) := by
  sorry


end NUMINAMATH_CALUDE_monotonic_function_theorem_l3473_347330


namespace NUMINAMATH_CALUDE_unique_lottery_ticket_l3473_347384

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_lottery_ticket (ticket : ℕ) (neighbor_age : ℕ) :
  is_five_digit ticket →
  digit_sum ticket = neighbor_age →
  (∀ m : ℕ, is_five_digit m → digit_sum m = neighbor_age → m = ticket) →
  ticket = 99999 :=
by sorry

end NUMINAMATH_CALUDE_unique_lottery_ticket_l3473_347384


namespace NUMINAMATH_CALUDE_solve_laboratory_budget_l3473_347396

def laboratory_budget_problem (total_budget flask_cost : ℕ) : Prop :=
  let test_tube_cost := (2 * flask_cost) / 3
  let safety_gear_cost := test_tube_cost / 2
  let total_spent := flask_cost + test_tube_cost + safety_gear_cost
  let remaining_budget := total_budget - total_spent
  total_budget = 325 ∧ flask_cost = 150 → remaining_budget = 25

theorem solve_laboratory_budget :
  laboratory_budget_problem 325 150 := by
  sorry

end NUMINAMATH_CALUDE_solve_laboratory_budget_l3473_347396


namespace NUMINAMATH_CALUDE_james_total_spent_l3473_347311

def milk_price : ℚ := 3
def bananas_price : ℚ := 2
def bread_price : ℚ := 3/2
def cereal_price : ℚ := 4

def milk_tax_rate : ℚ := 1/5
def bananas_tax_rate : ℚ := 3/20
def bread_tax_rate : ℚ := 1/10
def cereal_tax_rate : ℚ := 1/4

def total_spent : ℚ := milk_price * (1 + milk_tax_rate) + 
                       bananas_price * (1 + bananas_tax_rate) + 
                       bread_price * (1 + bread_tax_rate) + 
                       cereal_price * (1 + cereal_tax_rate)

theorem james_total_spent : total_spent = 251/20 := by
  sorry

end NUMINAMATH_CALUDE_james_total_spent_l3473_347311


namespace NUMINAMATH_CALUDE_min_value_theorem_l3473_347367

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ z, z = (2 * x * y) / (x + y - 1) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3473_347367


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l3473_347362

theorem unique_solution_power_equation (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! x : ℝ, a^x + b^x = c^x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l3473_347362


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3473_347360

theorem sum_of_coefficients (a b c d : ℤ) :
  (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 11*x + 6) →
  a + b + c + d = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3473_347360


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3473_347372

/-- The sum of the infinite series ∑_{n=1}^∞ (3^n / (1 + 3^n + 3^{n+1} + 3^{2n+1})) is equal to 1/4 -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1)) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_infinite_series_sum_l3473_347372


namespace NUMINAMATH_CALUDE_candy_cost_620_l3473_347335

/-- Calculates the cost of buying candies given the pricing structure -/
def candy_cost (total_candies : ℕ) : ℕ :=
  let regular_price := 8
  let discount_price := 7
  let candies_per_box := 40
  let discount_threshold := 500
  let full_price_boxes := min (total_candies / candies_per_box) (discount_threshold / candies_per_box)
  let discounted_boxes := (total_candies - full_price_boxes * candies_per_box + candies_per_box - 1) / candies_per_box
  full_price_boxes * regular_price + discounted_boxes * discount_price

theorem candy_cost_620 : candy_cost 620 = 125 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_620_l3473_347335


namespace NUMINAMATH_CALUDE_two_thirds_in_nine_fourths_l3473_347395

theorem two_thirds_in_nine_fourths : (9 : ℚ) / 4 / ((2 : ℚ) / 3) = 27 / 8 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_in_nine_fourths_l3473_347395


namespace NUMINAMATH_CALUDE_decimal_to_binary_87_l3473_347363

theorem decimal_to_binary_87 : 
  (87 : ℕ).digits 2 = [1, 1, 1, 0, 1, 0, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_87_l3473_347363


namespace NUMINAMATH_CALUDE_largest_valid_number_l3473_347365

def is_valid (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → (p^2 - 1) ∣ n

theorem largest_valid_number : 
  (1944 < 2012) ∧ 
  is_valid 1944 ∧ 
  (∀ m : ℕ, 1944 < m → m < 2012 → ¬ is_valid m) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3473_347365


namespace NUMINAMATH_CALUDE_vote_count_theorem_l3473_347385

/-- The number of ways to count votes such that candidate A always leads candidate B -/
def vote_count_ways (a b : ℕ) : ℕ :=
  (Nat.factorial (a + b - 1)) / (Nat.factorial (a - 1) * Nat.factorial b) -
  (Nat.factorial (a + b - 1)) / (Nat.factorial a * Nat.factorial (b - 1))

/-- Theorem stating the number of ways for candidate A to maintain a lead throughout the counting process -/
theorem vote_count_theorem (a b : ℕ) (h : a > b) :
  vote_count_ways a b = (Nat.factorial (a + b - 1)) / (Nat.factorial (a - 1) * Nat.factorial b) -
                        (Nat.factorial (a + b - 1)) / (Nat.factorial a * Nat.factorial (b - 1)) :=
by sorry

end NUMINAMATH_CALUDE_vote_count_theorem_l3473_347385


namespace NUMINAMATH_CALUDE_complement_union_eq_five_l3473_347389

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_eq_five : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_eq_five_l3473_347389


namespace NUMINAMATH_CALUDE_original_group_size_l3473_347359

-- Define the work completion rate for a group
def work_rate (num_men : ℕ) (days : ℕ) : ℚ := 1 / (num_men * days)

-- Define the theorem
theorem original_group_size :
  ∃ (x : ℕ),
    -- Condition 1: Original group completes work in 20 days
    work_rate x 20 =
    -- Condition 2 & 3: Remaining group (x - 10) completes work in 40 days
    work_rate (x - 10) 40 ∧
    -- Answer: The original group size is 20
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l3473_347359


namespace NUMINAMATH_CALUDE_distance_from_point_to_line_l3473_347310

def point : ℝ × ℝ × ℝ := (2, 4, 5)

def line_point : ℝ × ℝ × ℝ := (4, 6, 8)
def line_direction : ℝ × ℝ × ℝ := (1, 1, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  distance_to_line point line_point line_direction = 2 * Real.sqrt 33 / 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_point_to_line_l3473_347310


namespace NUMINAMATH_CALUDE_angle_bisector_points_sum_l3473_347368

theorem angle_bisector_points_sum (a b : ℝ) : 
  ((-4 : ℝ) = -4 ∧ a = -4) → 
  ((-2 : ℝ) = -2 ∧ b = -2) → 
  a + b + a * b = 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_points_sum_l3473_347368


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3473_347352

-- Define the function f(x) = 2|x|
def f (x : ℝ) : ℝ := 2 * abs x

-- State the theorem
theorem f_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ -- f is even
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) -- f is increasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3473_347352


namespace NUMINAMATH_CALUDE_balloon_difference_l3473_347319

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l3473_347319


namespace NUMINAMATH_CALUDE_circle_through_three_points_l3473_347326

/-- A circle passing through three points -/
structure Circle3Points where
  O : ℝ × ℝ
  M₁ : ℝ × ℝ
  M₂ : ℝ × ℝ

/-- The equation of a circle in standard form -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The circle passing through O(0,0), M₁(1,1), and M₂(4,2) 
    has the equation (x-4)² + (y+3)² = 25, with center (4, -3) and radius 5 -/
theorem circle_through_three_points :
  let c := Circle3Points.mk (0, 0) (1, 1) (4, 2)
  ∃ (h k r : ℝ),
    h = 4 ∧ k = -3 ∧ r = 5 ∧
    (∀ (x y : ℝ), CircleEquation h k r x y ↔
      ((x = c.O.1 ∧ y = c.O.2) ∨
       (x = c.M₁.1 ∧ y = c.M₁.2) ∨
       (x = c.M₂.1 ∧ y = c.M₂.2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_through_three_points_l3473_347326


namespace NUMINAMATH_CALUDE_tangent_line_intercept_l3473_347357

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + a*x + 2

-- Define the derivative of f(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*x + a

theorem tangent_line_intercept (a : ℝ) : 
  (f a 0 = 2) ∧ 
  (∃ m : ℝ, ∀ x : ℝ, m*x + 2 = f_prime a 0 * x + 2) ∧
  (∃ t : ℝ, t = -2 ∧ f_prime a 0 * t + 2 = 0) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_intercept_l3473_347357


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3473_347375

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3473_347375


namespace NUMINAMATH_CALUDE_largest_six_digit_divisible_by_five_l3473_347361

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem largest_six_digit_divisible_by_five :
  ∃ (n : ℕ), is_six_digit n ∧ n % 5 = 0 ∧ ∀ (m : ℕ), is_six_digit m ∧ m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_six_digit_divisible_by_five_l3473_347361


namespace NUMINAMATH_CALUDE_multiple_of_nine_between_15_and_30_l3473_347391

theorem multiple_of_nine_between_15_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 225)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_between_15_and_30_l3473_347391


namespace NUMINAMATH_CALUDE_line_bisects_circle_l3473_347351

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The equation of a line in the xy-plane -/
def Line (x y : ℝ) : Prop :=
  x - y + 1 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 2)

/-- Theorem stating that the line bisects the circle -/
theorem line_bisects_circle :
  ∀ x y : ℝ, Circle x y → Line x y → (x, y) = center :=
sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l3473_347351


namespace NUMINAMATH_CALUDE_intersection_theorem_l3473_347324

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 2*x < 3}

def B : Set ℝ := {x | (x-2)/x ≤ 0}

theorem intersection_theorem :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3473_347324


namespace NUMINAMATH_CALUDE_find_x_l3473_347312

theorem find_x : ∃ x : ℝ, 3 * x = (26 - x) + 26 ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_find_x_l3473_347312


namespace NUMINAMATH_CALUDE_expression_evaluation_l3473_347342

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  ((x + 2*y) * (x - 2*y) + 4 * (x - y)^2) / (-x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3473_347342


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3473_347332

theorem expression_equals_zero :
  (π - 2023) ^ (0 : ℝ) - |1 - Real.sqrt 2| + 2 * Real.cos (π / 4) - (1 / 2) ^ (-1 : ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3473_347332


namespace NUMINAMATH_CALUDE_total_apples_picked_l3473_347334

theorem total_apples_picked (benny_apples dan_apples : ℕ) : 
  benny_apples = 2 → dan_apples = 9 → benny_apples + dan_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_picked_l3473_347334


namespace NUMINAMATH_CALUDE_sqrt_sum_power_equality_l3473_347306

theorem sqrt_sum_power_equality (m n : ℕ) : 
  ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_power_equality_l3473_347306


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3473_347327

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 120) (h2 : leg = 24) :
  ∃ (other_leg hypotenuse : ℝ),
    area = (1 / 2) * leg * other_leg ∧
    hypotenuse ^ 2 = leg ^ 2 + other_leg ^ 2 ∧
    leg + other_leg + hypotenuse = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3473_347327


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_l3473_347364

/-- The probability of getting at least one head when tossing five coins,
    each with a 3/4 chance of heads, is 1023/1024. -/
theorem prob_at_least_one_head :
  let n : ℕ := 5  -- number of coins
  let p : ℚ := 3/4  -- probability of heads for each coin
  1 - (1 - p)^n = 1023/1024 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_head_l3473_347364


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l3473_347379

theorem students_taking_no_subjects (total : ℕ) (music art dance : ℕ) 
  (music_and_art music_and_dance art_and_dance : ℕ) (all_three : ℕ) :
  total = 800 ∧ 
  music = 140 ∧ 
  art = 90 ∧ 
  dance = 75 ∧
  music_and_art = 50 ∧
  music_and_dance = 30 ∧
  art_and_dance = 25 ∧
  all_three = 20 →
  total - (music + art + dance - music_and_art - music_and_dance - art_and_dance + all_three) = 580 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_no_subjects_l3473_347379


namespace NUMINAMATH_CALUDE_pen_measurement_properties_l3473_347303

def measured_length : Float := 0.06250

-- Function to count significant figures
def count_significant_figures (x : Float) : Nat :=
  sorry

-- Function to determine the place of accuracy
def place_of_accuracy (x : Float) : String :=
  sorry

theorem pen_measurement_properties :
  (count_significant_figures measured_length = 4) ∧
  (place_of_accuracy measured_length = "hundred-thousandth") :=
by sorry

end NUMINAMATH_CALUDE_pen_measurement_properties_l3473_347303


namespace NUMINAMATH_CALUDE_total_amount_correct_l3473_347317

/-- The amount of money Mrs. Hilt needs to share -/
def total_amount : ℝ := 3.75

/-- The number of people sharing the money -/
def number_of_people : ℕ := 3

/-- The amount each person receives -/
def amount_per_person : ℝ := 1.25

/-- Theorem stating that the total amount is correct given the conditions -/
theorem total_amount_correct : 
  total_amount = (number_of_people : ℝ) * amount_per_person :=
by sorry

end NUMINAMATH_CALUDE_total_amount_correct_l3473_347317


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3473_347356

theorem product_from_hcf_lcm (a b : ℕ+) : 
  Nat.gcd a b = 22 → Nat.lcm a b = 2058 → a * b = 45276 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3473_347356


namespace NUMINAMATH_CALUDE_two_recess_breaks_l3473_347366

/-- Calculates the number of 15-minute recess breaks given the total time outside class,
    lunch duration, and additional recess duration. -/
def numberOfRecessBreaks (totalTimeOutside lunchDuration additionalRecessDuration : ℕ) : ℕ :=
  ((totalTimeOutside - lunchDuration - additionalRecessDuration) / 15)

/-- Proves that given the specified conditions, students get 2 fifteen-minute recess breaks. -/
theorem two_recess_breaks :
  let totalTimeOutside : ℕ := 80
  let lunchDuration : ℕ := 30
  let additionalRecessDuration : ℕ := 20
  numberOfRecessBreaks totalTimeOutside lunchDuration additionalRecessDuration = 2 := by
sorry


end NUMINAMATH_CALUDE_two_recess_breaks_l3473_347366


namespace NUMINAMATH_CALUDE_claudia_weekend_earnings_l3473_347381

/-- Calculates the total earnings from weekend art classes -/
def weekend_earnings (cost_per_class : ℚ) (saturday_attendees : ℕ) : ℚ :=
  let sunday_attendees := saturday_attendees / 2
  let total_attendees := saturday_attendees + sunday_attendees
  cost_per_class * total_attendees

/-- Proves that Claudia's total earnings from her weekend art classes are $300.00 -/
theorem claudia_weekend_earnings :
  weekend_earnings 10 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_claudia_weekend_earnings_l3473_347381


namespace NUMINAMATH_CALUDE_charlie_age_when_jenny_thrice_bobby_l3473_347305

/-- 
Given:
- Jenny is older than Charlie by 12 years
- Charlie is older than Bobby by 7 years

Prove that Charlie is 18 years old when Jenny's age is three times Bobby's age.
-/
theorem charlie_age_when_jenny_thrice_bobby (jenny charlie bobby : ℕ) 
  (h1 : jenny = charlie + 12)
  (h2 : charlie = bobby + 7) :
  (jenny = 3 * bobby) → charlie = 18 := by
  sorry

end NUMINAMATH_CALUDE_charlie_age_when_jenny_thrice_bobby_l3473_347305


namespace NUMINAMATH_CALUDE_actual_distance_walked_l3473_347369

/-- 
Given a person who walks at two different speeds for the same duration:
- At 5 km/hr, they cover a distance D
- At 15 km/hr, they would cover a distance D + 20 km
This theorem proves that the actual distance D is 10 km.
-/
theorem actual_distance_walked (D : ℝ) : 
  (D / 5 = (D + 20) / 15) → D = 10 := by sorry

end NUMINAMATH_CALUDE_actual_distance_walked_l3473_347369
