import Mathlib

namespace NUMINAMATH_CALUDE_jacks_sock_purchase_l77_7744

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  four_dollar : ℕ

/-- Checks if the given SockPurchase satisfies all conditions --/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.four_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 4 * p.four_dollar = 36 ∧
  p.two_dollar ≥ 1 ∧ p.three_dollar ≥ 1 ∧ p.four_dollar ≥ 1

/-- Theorem stating that the only valid purchase has 10 pairs of $2 socks --/
theorem jacks_sock_purchase :
  ∀ p : SockPurchase, is_valid_purchase p → p.two_dollar = 10 := by
  sorry

end NUMINAMATH_CALUDE_jacks_sock_purchase_l77_7744


namespace NUMINAMATH_CALUDE_seven_non_similar_triangles_l77_7743

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base : Point) (top : Point)

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Checks if all sides of a triangle are unequal -/
def hasUnequalSides (t : Triangle) : Prop :=
  sorry

/-- Checks if three lines intersect at a single point -/
def intersectAtPoint (a b c : Altitude) (H : Point) : Prop :=
  sorry

/-- Counts the number of non-similar triangle types in the figure -/
def countNonSimilarTriangles (t : Triangle) (AD BE CF : Altitude) (H : Point) : ℕ :=
  sorry

/-- The main theorem -/
theorem seven_non_similar_triangles 
  (ABC : Triangle) 
  (AD BE CF : Altitude) 
  (H : Point) 
  (h1 : isAcuteAngled ABC) 
  (h2 : hasUnequalSides ABC)
  (h3 : intersectAtPoint AD BE CF H) :
  countNonSimilarTriangles ABC AD BE CF H = 7 :=
sorry

end NUMINAMATH_CALUDE_seven_non_similar_triangles_l77_7743


namespace NUMINAMATH_CALUDE_seven_presenter_schedule_l77_7741

/-- The number of ways to schedule n presenters with one specific presenter following another --/
def schedule_presenters (n : ℕ) : ℕ :=
  Nat.factorial n / 2

/-- Theorem: For 7 presenters, with one following another, there are 2520 ways to schedule --/
theorem seven_presenter_schedule :
  schedule_presenters 7 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_seven_presenter_schedule_l77_7741


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l77_7786

theorem smallest_integer_with_given_remainders :
  let x : ℕ := 167
  (∀ y : ℕ, y > 0 →
    (y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7) → y ≥ x) ∧
  (x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l77_7786


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l77_7721

/-- Converts a list of binary digits to its decimal representation -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to its binary representation -/
def decimalToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBinary (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinary (m / 2)
  toBinary n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let product := [true, true, true, false, false, true, true]  -- 1100111₂
  binaryToDecimal a * binaryToDecimal b = binaryToDecimal product := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l77_7721


namespace NUMINAMATH_CALUDE_dessert_probability_l77_7768

theorem dessert_probability (p_dessert_and_coffee : ℝ) (p_no_coffee_given_dessert : ℝ) :
  p_dessert_and_coffee = 0.6 →
  p_no_coffee_given_dessert = 0.2 →
  1 - (p_dessert_and_coffee / (1 - p_no_coffee_given_dessert)) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_dessert_probability_l77_7768


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l77_7787

theorem volunteer_allocation_schemes (n : ℕ) (m : ℕ) (k : ℕ) : 
  n = 5 → m = 3 → k = 2 →
  (Nat.choose n 1) * (Nat.choose (n - 1) k / 2) * Nat.factorial m = 90 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l77_7787


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l77_7745

/-- Given vectors a and b in ℝ², prove that k = -1 makes k*a - b perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (-3, 1)) :
  ∃ k : ℝ, k = -1 ∧ (k • a - b) • a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l77_7745


namespace NUMINAMATH_CALUDE_homecoming_dance_tickets_l77_7798

/-- Represents the number of couple tickets sold at a homecoming dance. -/
def couple_tickets : ℕ := 56

/-- Represents the number of single tickets sold at a homecoming dance. -/
def single_tickets : ℕ := 128 - 2 * couple_tickets

/-- The cost of a single ticket in dollars. -/
def single_ticket_cost : ℕ := 20

/-- The cost of a couple ticket in dollars. -/
def couple_ticket_cost : ℕ := 35

/-- The total ticket sales in dollars. -/
def total_sales : ℕ := 2280

/-- The total number of attendees. -/
def total_attendees : ℕ := 128

theorem homecoming_dance_tickets :
  single_ticket_cost * single_tickets + couple_ticket_cost * couple_tickets = total_sales ∧
  single_tickets + 2 * couple_tickets = total_attendees := by
  sorry

end NUMINAMATH_CALUDE_homecoming_dance_tickets_l77_7798


namespace NUMINAMATH_CALUDE_freshman_psych_liberal_arts_percentage_l77_7784

def college_population (total : ℝ) : Prop :=
  let freshmen := 0.5 * total
  let int_freshmen := 0.3 * freshmen
  let dom_freshmen := 0.7 * freshmen
  let int_lib_arts := 0.4 * int_freshmen
  let dom_lib_arts := 0.35 * dom_freshmen
  let int_psych_lib_arts := 0.2 * int_lib_arts
  let dom_psych_lib_arts := 0.25 * dom_lib_arts
  let total_psych_lib_arts := int_psych_lib_arts + dom_psych_lib_arts
  total_psych_lib_arts / total = 0.04

theorem freshman_psych_liberal_arts_percentage :
  ∀ total : ℝ, total > 0 → college_population total :=
sorry

end NUMINAMATH_CALUDE_freshman_psych_liberal_arts_percentage_l77_7784


namespace NUMINAMATH_CALUDE_new_city_total_buildings_l77_7748

/-- Represents the number of buildings of each type in Pittsburgh -/
structure PittsburghBuildings where
  stores : ℕ
  hospitals : ℕ
  schools : ℕ
  police_stations : ℕ

/-- Calculates the number of buildings for the new city based on Pittsburgh's numbers -/
def new_city_buildings (p : PittsburghBuildings) : ℕ × ℕ × ℕ × ℕ :=
  (p.stores / 2, p.hospitals * 2, p.schools - 50, p.police_stations + 5)

/-- Theorem stating that the total number of buildings in the new city is 2175 -/
theorem new_city_total_buildings (p : PittsburghBuildings) 
  (h1 : p.stores = 2000)
  (h2 : p.hospitals = 500)
  (h3 : p.schools = 200)
  (h4 : p.police_stations = 20) :
  let (new_stores, new_hospitals, new_schools, new_police) := new_city_buildings p
  new_stores + new_hospitals + new_schools + new_police = 2175 := by
  sorry

#check new_city_total_buildings

end NUMINAMATH_CALUDE_new_city_total_buildings_l77_7748


namespace NUMINAMATH_CALUDE_triangle_problem_l77_7706

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  (a^2 + c^2 - b^2) * Real.tan B = Real.sqrt 3 * (b^2 + c^2 - a^2) →
  (1/2) * b * c * Real.sin A = 3/2 →
  (A = π/3 ∧ ((b*c - 4*Real.sqrt 3) * Real.cos A + a*c * Real.cos B) / (a^2 - b^2) = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l77_7706


namespace NUMINAMATH_CALUDE_balls_after_5000_steps_l77_7778

/-- Converts a natural number to its base-6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Sums the digits in a list of natural numbers --/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- Represents the ball placement process --/
def ballPlacement (steps : ℕ) : ℕ :=
  sumDigits (toBase6 steps)

/-- The main theorem stating that after 5000 steps, there are 13 balls in the boxes --/
theorem balls_after_5000_steps :
  ballPlacement 5000 = 13 := by
  sorry

end NUMINAMATH_CALUDE_balls_after_5000_steps_l77_7778


namespace NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l77_7782

theorem sqrt_x_plus_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_reciprocal_l77_7782


namespace NUMINAMATH_CALUDE_candy_distribution_l77_7735

theorem candy_distribution (x y n : ℕ) : 
  y + n = 4 * (x - n) →
  x + 90 = 5 * (y - 90) →
  y ≥ 115 →
  (∀ y' : ℕ, y' ≥ 115 → y' + n = 4 * (x - n) → x + 90 = 5 * (y' - 90) → y ≤ y') →
  y = 115 ∧ x = 35 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l77_7735


namespace NUMINAMATH_CALUDE_todds_initial_money_l77_7759

/-- Represents the problem of calculating Todd's initial amount of money -/
theorem todds_initial_money (num_candies : ℕ) (candy_cost : ℕ) (money_left : ℕ) : 
  num_candies = 4 → candy_cost = 2 → money_left = 12 → 
  num_candies * candy_cost + money_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_todds_initial_money_l77_7759


namespace NUMINAMATH_CALUDE_mans_speed_mans_speed_specific_l77_7780

/-- The speed of a man running in the same direction as a train, given the train's length, speed, and time to cross the man. -/
theorem mans_speed (train_length : Real) (train_speed_kmh : Real) (time_to_cross : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / time_to_cross
  let mans_speed_ms := train_speed_ms - relative_speed
  let mans_speed_kmh := mans_speed_ms * 3600 / 1000
  mans_speed_kmh

/-- Given the specific conditions, prove that the man's speed is approximately 8 km/hr. -/
theorem mans_speed_specific : 
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  |mans_speed 620 80 30.99752019838413 - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_mans_speed_mans_speed_specific_l77_7780


namespace NUMINAMATH_CALUDE_quadratic_function_point_comparison_l77_7703

/-- Given a quadratic function y = x² - 4x + k passing through (-1, y₁) and (3, y₂), prove y₁ > y₂ -/
theorem quadratic_function_point_comparison (k : ℝ) (y₁ y₂ : ℝ)
  (h₁ : y₁ = (-1)^2 - 4*(-1) + k)
  (h₂ : y₂ = 3^2 - 4*3 + k) :
  y₁ > y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_point_comparison_l77_7703


namespace NUMINAMATH_CALUDE_inequality_proofs_l77_7729

theorem inequality_proofs :
  (∀ a b : ℝ, a > 0 → b > 0 → (a + b) * (1 / a + 1 / b) ≥ 4) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proofs_l77_7729


namespace NUMINAMATH_CALUDE_awards_distribution_l77_7796

/-- Represents the number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution :
  distribute_awards 6 4 = 780 :=
by
  sorry

end NUMINAMATH_CALUDE_awards_distribution_l77_7796


namespace NUMINAMATH_CALUDE_existence_of_four_integers_l77_7731

theorem existence_of_four_integers : ∃ (a b c d : ℤ),
  (a > 1000000) ∧ (b > 1000000) ∧ (c > 1000000) ∧ (d > 1000000) ∧
  (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / (a * b * c * d) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_four_integers_l77_7731


namespace NUMINAMATH_CALUDE_first_term_value_l77_7777

/-- Given a sequence {aₙ} with sum Sₙ, prove that a₁ = 1/2 -/
theorem first_term_value (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, S n = (a 1 * (4^n - 1)) / 3) →   -- Condition 1
  a 4 = 32 →                             -- Condition 2
  a 1 = 1/2 :=                           -- Conclusion
by sorry

end NUMINAMATH_CALUDE_first_term_value_l77_7777


namespace NUMINAMATH_CALUDE_abs_inequality_l77_7753

theorem abs_inequality (a b : ℝ) (h : a^2 + b^2 ≤ 4) :
  |3 * a^2 - 8 * a * b - 3 * b^2| ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l77_7753


namespace NUMINAMATH_CALUDE_largest_angle_measure_l77_7750

def ConvexPentagon (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a + b + c + d + e = 540

theorem largest_angle_measure (a b c d e : ℝ) :
  ConvexPentagon a b c d e →
  c - 3 = a →
  e = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_measure_l77_7750


namespace NUMINAMATH_CALUDE_rectangle_area_l77_7751

/-- Given a rectangle with perimeter 40 feet and length-to-width ratio 3:2, its area is 96 square feet -/
theorem rectangle_area (length width : ℝ) : 
  (2 * (length + width) = 40) →  -- perimeter condition
  (length = 3/2 * width) →       -- ratio condition
  (length * width = 96) :=        -- area is 96 square feet
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l77_7751


namespace NUMINAMATH_CALUDE_f_minimum_l77_7701

/-- The quadratic function f(x) = x^2 - 14x + 40 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 40

/-- The value of x that minimizes f(x) -/
def x_min : ℝ := 7

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f x_min :=
sorry

end NUMINAMATH_CALUDE_f_minimum_l77_7701


namespace NUMINAMATH_CALUDE_commonMaterialChoices_eq_120_l77_7708

/-- The number of ways to choose r items from n items without regard to order -/
def binomial (n r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items out of n items -/
def permutation (n r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of ways two students can choose 2 materials each from 6 materials, 
    with exactly 1 material in common -/
def commonMaterialChoices : ℕ :=
  binomial 6 1 * permutation 5 2

theorem commonMaterialChoices_eq_120 : commonMaterialChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_commonMaterialChoices_eq_120_l77_7708


namespace NUMINAMATH_CALUDE_rectangular_field_area_l77_7737

theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 90 →
  area = width * length →
  area = 379.6875 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l77_7737


namespace NUMINAMATH_CALUDE_sum_of_variables_l77_7720

theorem sum_of_variables (a b c : ℚ) 
  (eq1 : b + c = 15 - 4*a)
  (eq2 : a + c = -18 - 4*b)
  (eq3 : a + b = 10 - 4*c) : 
  2*a + 2*b + 2*c = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_variables_l77_7720


namespace NUMINAMATH_CALUDE_kitchen_tiles_l77_7756

theorem kitchen_tiles (kitchen_length kitchen_width tile_area : ℝ) 
  (h1 : kitchen_length = 52)
  (h2 : kitchen_width = 79)
  (h3 : tile_area = 7.5) : 
  ⌈(kitchen_length * kitchen_width) / tile_area⌉ = 548 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_tiles_l77_7756


namespace NUMINAMATH_CALUDE_molecular_weight_7_moles_KBrO3_l77_7716

/-- The atomic weight of potassium in g/mol -/
def atomic_weight_K : ℝ := 39.10

/-- The atomic weight of bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of oxygen atoms in KBrO3 -/
def num_oxygen_atoms : ℕ := 3

/-- The molecular weight of one mole of KBrO3 in g/mol -/
def molecular_weight_KBrO3 : ℝ :=
  atomic_weight_K + atomic_weight_Br + (atomic_weight_O * num_oxygen_atoms)

/-- The number of moles of KBrO3 -/
def num_moles : ℕ := 7

/-- Theorem: The molecular weight of 7 moles of KBrO3 is 1169.00 grams -/
theorem molecular_weight_7_moles_KBrO3 :
  molecular_weight_KBrO3 * num_moles = 1169.00 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_7_moles_KBrO3_l77_7716


namespace NUMINAMATH_CALUDE_handshakes_in_specific_event_l77_7717

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group1_size : ℕ  -- people who know each other
  group2_size : ℕ  -- people who know no one
  h_total : total_people = group1_size + group2_size

/-- Calculates the number of handshakes in a social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  (event.group2_size * (event.total_people - 1)) / 2

/-- Theorem stating the number of handshakes in the specific social event -/
theorem handshakes_in_specific_event :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group1_size = 25 ∧
    event.group2_size = 15 ∧
    count_handshakes event = 292 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_event_l77_7717


namespace NUMINAMATH_CALUDE_fraction_equality_implies_constants_l77_7783

theorem fraction_equality_implies_constants (a b : ℝ) :
  (∀ x : ℝ, x ≠ -b → x ≠ -36 → x ≠ -30 → 
    (x - a) / (x + b) = (x^2 - 45*x + 504) / (x^2 + 66*x - 1080)) →
  a = 18 ∧ b = 30 ∧ a + b = 48 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_constants_l77_7783


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l77_7712

/-- A function f : [1, +∞) → [1, +∞) satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≥ 1) ∧
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (1 / x) * ((f x)^2 - 1))

/-- The theorem stating that x + 1 is the unique function satisfying the conditions -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ ∀ x ≥ 1, f x = x + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l77_7712


namespace NUMINAMATH_CALUDE_red_car_cost_is_three_l77_7766

/-- Represents the cost of renting a red car per minute -/
def red_car_cost : ℝ := sorry

/-- Represents the number of red cars -/
def num_red_cars : ℕ := 3

/-- Represents the number of white cars -/
def num_white_cars : ℕ := 2

/-- Represents the cost of renting a white car per minute -/
def white_car_cost : ℝ := 2

/-- Represents the rental duration in minutes -/
def rental_duration : ℕ := 3 * 60

/-- Represents the total earnings -/
def total_earnings : ℝ := 2340

theorem red_car_cost_is_three :
  red_car_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_red_car_cost_is_three_l77_7766


namespace NUMINAMATH_CALUDE_rock_collection_contest_l77_7709

theorem rock_collection_contest (sydney_start conner_start : ℕ) 
  (sydney_day1 conner_day2 conner_day3 : ℕ) : 
  sydney_start = 837 → 
  conner_start = 723 → 
  sydney_day1 = 4 → 
  conner_day2 = 123 → 
  conner_day3 = 27 → 
  ∃ (conner_day1 : ℕ), 
    conner_start + conner_day1 + conner_day2 + conner_day3 
    = sydney_start + sydney_day1 + 2 * conner_day1 
    ∧ conner_day1 / sydney_day1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_rock_collection_contest_l77_7709


namespace NUMINAMATH_CALUDE_decreasing_implies_a_leq_neg_three_l77_7757

/-- A quadratic function f(x) that is decreasing on (-∞, 4] -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The property that f is decreasing on (-∞, 4] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- Theorem stating that if f is decreasing on (-∞, 4], then a ≤ -3 -/
theorem decreasing_implies_a_leq_neg_three (a : ℝ) :
  is_decreasing_on_interval a → a ≤ -3 := by sorry

end NUMINAMATH_CALUDE_decreasing_implies_a_leq_neg_three_l77_7757


namespace NUMINAMATH_CALUDE_exact_payment_l77_7797

/-- The cost of the book in cents -/
def book_cost : ℕ := 4550

/-- The value of four $10 bills in cents -/
def bills_value : ℕ := 4000

/-- The value of ten nickels in cents -/
def nickels_value : ℕ := 50

/-- The minimum number of pennies needed -/
def min_pennies : ℕ := book_cost - bills_value - nickels_value

theorem exact_payment :
  min_pennies = 500 := by sorry

end NUMINAMATH_CALUDE_exact_payment_l77_7797


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l77_7767

theorem cafeteria_green_apples :
  let red_apples : ℕ := 43
  let students_wanting_fruit : ℕ := 2
  let extra_apples : ℕ := 73
  let green_apples : ℕ := red_apples + extra_apples + students_wanting_fruit - red_apples
  green_apples = 32 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l77_7767


namespace NUMINAMATH_CALUDE_calculation_proof_l77_7764

theorem calculation_proof :
  let expr1 := -1^4 - (1/6) * (2 - (-3)^2) / (-7)
  let expr2 := (1 + 1/2 - 5/8 + 7/12) / (-1/24) - 8 * (-1/2)^3
  expr1 = -7/6 ∧ expr2 = -34 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l77_7764


namespace NUMINAMATH_CALUDE_min_distance_between_sets_l77_7773

/-- The minimum distance between a point on the set defined by y² - 3x² - 2xy - 9 - 12x = 0
    and a point on the set defined by x² - 8y + 23 + 6x + y² = 0 -/
theorem min_distance_between_sets :
  let set1 := {(x, y) : ℝ × ℝ | y^2 - 3*x^2 - 2*x*y - 9 - 12*x = 0}
  let set2 := {(x, y) : ℝ × ℝ | x^2 - 8*y + 23 + 6*x + y^2 = 0}
  ∃ (min_dist : ℝ), min_dist = (7 * Real.sqrt 10) / 10 - Real.sqrt 2 ∧
    ∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ set1 → b ∈ set2 →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_sets_l77_7773


namespace NUMINAMATH_CALUDE_annual_yield_improvement_l77_7747

/-- The percentage improvement in annual yield given last year's and this year's ranges -/
theorem annual_yield_improvement (last_year_range this_year_range : ℝ) 
  (h1 : last_year_range = 10000)
  (h2 : this_year_range = 11500) :
  (this_year_range - last_year_range) / last_year_range * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_annual_yield_improvement_l77_7747


namespace NUMINAMATH_CALUDE_cloth_cost_price_l77_7733

theorem cloth_cost_price
  (total_length : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_length = 60)
  (h2 : selling_price = 8400)
  (h3 : profit_per_meter = 12) :
  (selling_price - total_length * profit_per_meter) / total_length = 128 :=
by sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l77_7733


namespace NUMINAMATH_CALUDE_half_radius_circle_y_l77_7725

theorem half_radius_circle_y (x y : Real) : 
  (2 * Real.pi * x = 10 * Real.pi) →  -- Circumference of circle x is 10π
  (Real.pi * x^2 = Real.pi * y^2) →   -- Areas of circles x and y are equal
  (1/2) * y = 2.5 := by               -- Half of the radius of circle y is 2.5
sorry

end NUMINAMATH_CALUDE_half_radius_circle_y_l77_7725


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l77_7739

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) - 15 = 8) ∧ 
  (-1^4 + (-2)^3 * (-1/2) - |(-1-5)| = -3) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l77_7739


namespace NUMINAMATH_CALUDE_cube_property_l77_7779

-- Define a cube type
structure Cube where
  side : ℝ
  volume_eq : volume = 8 * x
  area_eq : surfaceArea = x / 2

-- Define volume and surface area functions
def volume (c : Cube) : ℝ := c.side ^ 3
def surfaceArea (c : Cube) : ℝ := 6 * c.side ^ 2

-- State the theorem
theorem cube_property (x : ℝ) (c : Cube) : x = 110592 := by
  sorry

end NUMINAMATH_CALUDE_cube_property_l77_7779


namespace NUMINAMATH_CALUDE_exists_quadrilateral_no_triangle_l77_7769

/-- A convex quadrilateral with angles α, β, γ, and δ (in degrees) -/
structure ConvexQuadrilateral where
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  sum_360 : α + β + γ + δ = 360
  all_positive : 0 < α ∧ 0 < β ∧ 0 < γ ∧ 0 < δ
  all_less_180 : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180

/-- Check if three real numbers can form the sides of a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that there exists a convex quadrilateral where no three angles can form a triangle -/
theorem exists_quadrilateral_no_triangle : ∃ q : ConvexQuadrilateral, 
  ¬(canFormTriangle q.α q.β q.γ ∨ 
    canFormTriangle q.α q.β q.δ ∨ 
    canFormTriangle q.α q.γ q.δ ∨ 
    canFormTriangle q.β q.γ q.δ) := by
  sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_no_triangle_l77_7769


namespace NUMINAMATH_CALUDE_polynomial_sum_l77_7761

theorem polynomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l77_7761


namespace NUMINAMATH_CALUDE_squirrel_mushroom_theorem_l77_7723

theorem squirrel_mushroom_theorem (N : ℝ) (h : N > 0) :
  let initial_porcini := 0.85 * N
  let initial_saffron := 0.15 * N
  let eaten (x : ℝ) := x
  let remaining_porcini (x : ℝ) := initial_porcini - eaten x
  let remaining_total (x : ℝ) := N - eaten x
  let final_saffron_ratio (x : ℝ) := initial_saffron / remaining_total x
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ initial_porcini ∧ final_saffron_ratio x = 0.3 ∧ eaten x / N = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_mushroom_theorem_l77_7723


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l77_7792

/-- Given that x² varies inversely with ⁴√w, prove that if x = 3 when w = 16, then x = √6 when w = 81 -/
theorem inverse_variation_problem (x w : ℝ) (h : ∃ k : ℝ, ∀ x w, x^2 * w^(1/4) = k) :
  (x = 3 ∧ w = 16) → (w = 81 → x = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l77_7792


namespace NUMINAMATH_CALUDE_final_population_theorem_l77_7736

/-- Calculates the population after two years of change -/
def population_after_two_years (initial_population : ℕ) : ℕ :=
  let after_increase := initial_population * 130 / 100
  let after_decrease := after_increase * 70 / 100
  after_decrease

/-- Theorem stating the final population after two years of change -/
theorem final_population_theorem :
  population_after_two_years 15000 = 13650 := by
  sorry

end NUMINAMATH_CALUDE_final_population_theorem_l77_7736


namespace NUMINAMATH_CALUDE_recipe_scaling_l77_7742

def original_flour : ℚ := 20/3

theorem recipe_scaling :
  let scaled_flour : ℚ := (1/3) * original_flour
  let scaled_sugar : ℚ := (1/2) * scaled_flour
  scaled_flour = 20/9 ∧ scaled_sugar = 10/9 := by sorry

end NUMINAMATH_CALUDE_recipe_scaling_l77_7742


namespace NUMINAMATH_CALUDE_complement_of_union_l77_7763

theorem complement_of_union (U A B : Set ℕ) : 
  U = {x : ℕ | x > 0 ∧ x < 6} →
  A = {1, 3} →
  B = {3, 5} →
  (U \ (A ∪ B)) = {2, 4} := by
sorry

end NUMINAMATH_CALUDE_complement_of_union_l77_7763


namespace NUMINAMATH_CALUDE_length_of_AB_l77_7704

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x + y - 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 9 = 0

-- Define that l is the axis of symmetry of C
def is_axis_of_symmetry (k : ℝ) : Prop := 
  ∀ x y : ℝ, line_l k x y → (circle_C x y ↔ circle_C (2*3-x) (2*(-1)-y))

-- Define point A
def point_A (k : ℝ) : ℝ × ℝ := (0, k)

-- Define that there exists a point B on circle C such that AB is tangent to C
def exists_tangent_point (k : ℝ) : Prop :=
  ∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    ((B.1 - 0) * (B.2 - k) = 1 ∨ (B.1 - 0) * (B.2 - k) = -1)

-- Theorem statement
theorem length_of_AB (k : ℝ) :
  is_axis_of_symmetry k →
  exists_tangent_point k →
  ∃ B : ℝ × ℝ, circle_C B.1 B.2 ∧ 
    ((B.1 - 0) * (B.2 - k) = 1 ∨ (B.1 - 0) * (B.2 - k) = -1) ∧
    Real.sqrt ((B.1 - 0)^2 + (B.2 - k)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l77_7704


namespace NUMINAMATH_CALUDE_three_tangent_lines_l77_7728

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a line passing through (0,2)
def line_through_point (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b ∧ 2 = b

-- Define the condition for a line to intersect the parabola at exactly one point
def intersects_once (m b : ℝ) : Prop :=
  ∃! x y, parabola x y ∧ line_through_point m b x y

-- The main theorem
theorem three_tangent_lines :
  ∃ L1 L2 L3 : ℝ × ℝ,
    L1 ≠ L2 ∧ L1 ≠ L3 ∧ L2 ≠ L3 ∧
    (∀ m b, intersects_once m b ↔ (m, b) = L1 ∨ (m, b) = L2 ∨ (m, b) = L3) :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l77_7728


namespace NUMINAMATH_CALUDE_ages_sum_l77_7740

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l77_7740


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l77_7762

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := 5 * x + 2

theorem composite_function_evaluation : f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l77_7762


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l77_7711

/-- Represents the number of students in various subject combinations --/
structure ScienceClub where
  total : ℕ
  math : ℕ
  physics : ℕ
  chemistry : ℕ
  math_physics : ℕ
  physics_chemistry : ℕ
  math_chemistry : ℕ
  all_three : ℕ

/-- Theorem stating the number of students taking no subjects --/
theorem students_taking_no_subjects (club : ScienceClub)
  (h_total : club.total = 150)
  (h_math : club.math = 85)
  (h_physics : club.physics = 63)
  (h_chemistry : club.chemistry = 40)
  (h_math_physics : club.math_physics = 20)
  (h_physics_chemistry : club.physics_chemistry = 15)
  (h_math_chemistry : club.math_chemistry = 10)
  (h_all_three : club.all_three = 5) :
  club.total - (club.math + club.physics + club.chemistry
    - club.math_physics - club.physics_chemistry - club.math_chemistry
    + club.all_three) = 2 := by
  sorry

#check students_taking_no_subjects

end NUMINAMATH_CALUDE_students_taking_no_subjects_l77_7711


namespace NUMINAMATH_CALUDE_estimate_shadowed_area_l77_7775

/-- Estimate the area of a shadowed region in a square based on bean distribution --/
theorem estimate_shadowed_area (total_area : ℝ) (total_beans : ℕ) (outside_beans : ℕ) 
  (h1 : total_area = 10) 
  (h2 : total_beans = 200) 
  (h3 : outside_beans = 114) : 
  ∃ (estimated_area : ℝ), abs (estimated_area - (total_area - (outside_beans : ℝ) / (total_beans : ℝ) * total_area)) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_estimate_shadowed_area_l77_7775


namespace NUMINAMATH_CALUDE_chips_in_bag_is_24_l77_7746

/-- The number of chips in a bag, given the calorie and cost information --/
def chips_in_bag (calories_per_chip : ℕ) (cost_per_bag : ℕ) (total_calories : ℕ) (total_cost : ℕ) : ℕ :=
  (total_calories / calories_per_chip) / (total_cost / cost_per_bag)

/-- Theorem stating that there are 24 chips in a bag --/
theorem chips_in_bag_is_24 :
  chips_in_bag 10 2 480 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_chips_in_bag_is_24_l77_7746


namespace NUMINAMATH_CALUDE_anthony_transactions_percentage_l77_7730

/-- Proves that Anthony handled 10% more transactions than Mabel -/
theorem anthony_transactions_percentage (mabel_transactions cal_transactions jade_transactions : ℕ) 
  (h1 : mabel_transactions = 90)
  (h2 : jade_transactions = 81)
  (h3 : jade_transactions = cal_transactions + 15)
  (h4 : cal_transactions = (2 * anthony_transactions) / 3)
  : (anthony_transactions - mabel_transactions : ℚ) / mabel_transactions = 1 / 10 := by
  sorry

#check anthony_transactions_percentage

end NUMINAMATH_CALUDE_anthony_transactions_percentage_l77_7730


namespace NUMINAMATH_CALUDE_factor_implies_k_equals_8_l77_7727

theorem factor_implies_k_equals_8 (m k : ℝ) : 
  (∃ q : ℝ, m^3 - k*m^2 - 24*m + 16 = (m^2 - 8*m) * q) → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_k_equals_8_l77_7727


namespace NUMINAMATH_CALUDE_square_decomposition_l77_7702

theorem square_decomposition (a : ℤ) :
  a^2 + 5*a + 7 = (a + 3) * (a + 2)^2 + (a + 2) * 1^2 := by
  sorry

end NUMINAMATH_CALUDE_square_decomposition_l77_7702


namespace NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l77_7734

theorem unit_circle_point_x_coordinate 
  (P : ℝ × ℝ) (α : ℝ) 
  (h1 : P.1^2 + P.2^2 = 1) 
  (h2 : P.1 = Real.cos α) 
  (h3 : P.2 = Real.sin α) 
  (h4 : π/3 < α ∧ α < 5*π/6) 
  (h5 : Real.sin (α + π/6) = 3/5) : 
  P.1 = (3 - 4*Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l77_7734


namespace NUMINAMATH_CALUDE_triangle_ratio_l77_7791

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B + b * Real.cos A = 3 * a →
  c / a = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l77_7791


namespace NUMINAMATH_CALUDE_bear_food_in_victors_l77_7705

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_worth_of_food (bear_food_per_day : ℕ) (victor_weight : ℕ) (weeks : ℕ) : ℕ :=
  (bear_food_per_day * weeks * 7) / victor_weight

/-- Theorem stating that a bear eating 90 pounds of food per day would eat 15 "Victors" worth of food in 3 weeks, given that Victor weighs 126 pounds -/
theorem bear_food_in_victors : victors_worth_of_food 90 126 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bear_food_in_victors_l77_7705


namespace NUMINAMATH_CALUDE_train_speed_l77_7707

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 385 →
  bridge_length = 140 →
  time = 42 →
  (train_length + bridge_length) / time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l77_7707


namespace NUMINAMATH_CALUDE_log_product_equivalence_l77_7765

open Real

theorem log_product_equivalence (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy_neq_1 : y ≠ 1) :
  (log x / log (y^4)) * (log (y^6) / log (x^3)) * (log (x^2) / log (y^3)) *
  (log (y^3) / log (x^2)) * (log (x^3) / log (y^6)) = (3/4) * (log x / log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_equivalence_l77_7765


namespace NUMINAMATH_CALUDE_milkshakes_bought_l77_7722

def initial_amount : ℕ := 120
def hamburger_cost : ℕ := 4
def milkshake_cost : ℕ := 3
def hamburgers_bought : ℕ := 8
def final_amount : ℕ := 70

theorem milkshakes_bought :
  ∃ (m : ℕ), 
    initial_amount - (hamburger_cost * hamburgers_bought + milkshake_cost * m) = final_amount ∧
    m = 6 := by
  sorry

end NUMINAMATH_CALUDE_milkshakes_bought_l77_7722


namespace NUMINAMATH_CALUDE_line_through_circle_center_l77_7794

/-- The center of a circle given by the equation x² + y² + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop := 3 * x + y + a = 0

/-- The circle equation x² + y² + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem: If the line 3x + y + a = 0 passes through the center of the circle x² + y² + 2x - 4y = 0, then a = 1 -/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l77_7794


namespace NUMINAMATH_CALUDE_y1_greater_y2_l77_7790

/-- A line in the 2D plane represented by y = mx + b -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

theorem y1_greater_y2 (l : Line) (p1 p2 : Point) :
  l.m = -1 →
  l.b = 1 →
  p1.x = -2 →
  p2.x = 3 →
  p1.liesOn l →
  p2.liesOn l →
  p1.y > p2.y := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_y2_l77_7790


namespace NUMINAMATH_CALUDE_tangent_line_condition_l77_7785

theorem tangent_line_condition (a : ℝ) : 
  (∃ (x : ℝ), a * x + 1 - a = x^2 ∧ ∀ (y : ℝ), y ≠ x → a * y + 1 - a ≠ y^2) ↔ |a| = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l77_7785


namespace NUMINAMATH_CALUDE_factorization_proof_l77_7700

theorem factorization_proof (x y : ℝ) : 9*y - 25*x^2*y = y*(3+5*x)*(3-5*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l77_7700


namespace NUMINAMATH_CALUDE_sequence_modulo_l77_7718

/-- Given a prime number p > 3, we define a sequence a_n as follows:
    a_n = n for n ∈ {0, 1, ..., p-1}
    a_n = a_{n-1} + a_{n-p} for n ≥ p
    This theorem states that a_{p^3} ≡ p-1 (mod p) -/
theorem sequence_modulo (p : ℕ) (hp : p.Prime ∧ p > 3) : 
  ∃ a : ℕ → ℕ, 
    (∀ n < p, a n = n) ∧ 
    (∀ n ≥ p, a n = a (n-1) + a (n-p)) ∧ 
    a (p^3) ≡ p-1 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_sequence_modulo_l77_7718


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l77_7788

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum :
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l77_7788


namespace NUMINAMATH_CALUDE_keychain_arrangement_theorem_l77_7760

def number_of_arrangements (n : ℕ) : ℕ := n.factorial

def number_of_adjacent_arrangements (n : ℕ) : ℕ := 2 * (n - 1).factorial

theorem keychain_arrangement_theorem :
  let total_arrangements := number_of_arrangements 5
  let adjacent_arrangements := number_of_adjacent_arrangements 5
  total_arrangements - adjacent_arrangements = 72 := by
  sorry

end NUMINAMATH_CALUDE_keychain_arrangement_theorem_l77_7760


namespace NUMINAMATH_CALUDE_multiples_of_15_between_17_and_152_l77_7714

theorem multiples_of_15_between_17_and_152 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 17 ∧ n < 152) (Finset.range 152)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_17_and_152_l77_7714


namespace NUMINAMATH_CALUDE_polynomial_divisibility_existence_l77_7724

theorem polynomial_divisibility_existence : ∃ (r s : ℝ),
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 
    ((x - r)^2 * (x - s) * (x - 1)) * q x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_existence_l77_7724


namespace NUMINAMATH_CALUDE_school_classrooms_l77_7771

theorem school_classrooms (total_students : ℕ) (desks_type1 : ℕ) (desks_type2 : ℕ) :
  total_students = 400 →
  desks_type1 = 30 →
  desks_type2 = 25 →
  ∃ (num_classrooms : ℕ),
    num_classrooms > 0 ∧
    (num_classrooms / 3) * desks_type1 + (2 * num_classrooms / 3) * desks_type2 = total_students ∧
    num_classrooms = 15 := by
  sorry

end NUMINAMATH_CALUDE_school_classrooms_l77_7771


namespace NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l77_7738

/-- The number of games Gwen gave away -/
def games_given_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Gwen gave away 7 games -/
theorem gwen_gave_away_seven_games :
  let initial := 98
  let remaining := 91
  games_given_away initial remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l77_7738


namespace NUMINAMATH_CALUDE_find_k_value_l77_7710

/-- Represents a point on a line segment --/
structure SegmentPoint where
  position : ℝ
  min : ℝ
  max : ℝ
  h : min ≤ position ∧ position ≤ max

/-- The theorem stating the value of k --/
theorem find_k_value (AB CD : ℝ × ℝ) (h_AB : AB = (0, 6)) (h_CD : CD = (0, 9)) :
  ∃ k : ℝ, 
    (∀ (P : SegmentPoint) (Q : SegmentPoint), 
      P.min = 0 ∧ P.max = 6 ∧ Q.min = 0 ∧ Q.max = 9 →
      P.position = 3 * k → P.position + Q.position = 12 * k) →
    k = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l77_7710


namespace NUMINAMATH_CALUDE_people_in_line_l77_7719

theorem people_in_line (initial_people : ℕ) (additional_people : ℕ) : 
  initial_people = 61 → additional_people = 22 → initial_people + additional_people = 83 := by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l77_7719


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_simplification_l77_7770

theorem sqrt_sum_fractions_simplification :
  Real.sqrt ((25 : ℝ) / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_simplification_l77_7770


namespace NUMINAMATH_CALUDE_gcf_lcm_300_125_l77_7793

theorem gcf_lcm_300_125 :
  (Nat.gcd 300 125 = 25) ∧ (Nat.lcm 300 125 = 1500) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_300_125_l77_7793


namespace NUMINAMATH_CALUDE_not_perfect_cube_l77_7774

theorem not_perfect_cube (t : ℤ) : ¬ ∃ (k : ℤ), 7 * t + 3 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_cube_l77_7774


namespace NUMINAMATH_CALUDE_balloon_problem_l77_7726

-- Define the number of balloons each person has
def allan_initial : ℕ := sorry
def jake_initial : ℕ := 6
def allan_bought : ℕ := 3

-- Define the relationship between Allan's and Jake's balloons
theorem balloon_problem :
  allan_initial = 2 :=
by
  have h1 : jake_initial = (allan_initial + allan_bought) + 1 :=
    sorry
  sorry

end NUMINAMATH_CALUDE_balloon_problem_l77_7726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l77_7758

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 5 + a 6 = 27) →
  (a 1 + a 9 = 18) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l77_7758


namespace NUMINAMATH_CALUDE_unique_element_quadratic_l77_7715

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 4 * x + 4 = 0}

-- State the theorem
theorem unique_element_quadratic (a : ℝ) : 
  (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_unique_element_quadratic_l77_7715


namespace NUMINAMATH_CALUDE_sams_sandwich_count_l77_7755

/-- Represents the number of different types for each sandwich component -/
structure SandwichOptions where
  bread : Nat
  meat : Nat
  cheese : Nat

/-- Calculates the number of sandwiches Sam can order given the options and restrictions -/
def samsSandwichOptions (options : SandwichOptions) : Nat :=
  options.bread * options.meat * options.cheese - 
  options.bread - 
  options.cheese - 
  options.bread

/-- The theorem stating the number of sandwich options for Sam -/
theorem sams_sandwich_count :
  samsSandwichOptions ⟨5, 7, 6⟩ = 194 := by
  sorry

#eval samsSandwichOptions ⟨5, 7, 6⟩

end NUMINAMATH_CALUDE_sams_sandwich_count_l77_7755


namespace NUMINAMATH_CALUDE_f_nine_equals_zero_l77_7795

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 6) = f x + f 3

theorem f_nine_equals_zero (f : ℝ → ℝ) (h1 : is_even f) (h2 : satisfies_condition f) : 
  f 9 = 0 := by sorry

end NUMINAMATH_CALUDE_f_nine_equals_zero_l77_7795


namespace NUMINAMATH_CALUDE_work_completion_time_l77_7799

/-- Given that person B can complete 2/3 of a job in 12 days, 
    prove that B can complete the entire job in 18 days. -/
theorem work_completion_time (B_partial_time : ℕ) (B_partial_work : ℚ) 
  (h1 : B_partial_time = 12) 
  (h2 : B_partial_work = 2/3) : 
  ∃ (B_full_time : ℕ), B_full_time = 18 ∧ 
  B_partial_work / B_partial_time = 1 / B_full_time :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l77_7799


namespace NUMINAMATH_CALUDE_distance_sum_equals_radii_sum_l77_7781

/-- An acute-angled triangle with its circumscribed and inscribed circles -/
structure AcuteTriangle where
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance from the circumcenter to side a -/
  da : ℝ
  /-- The distance from the circumcenter to side b -/
  db : ℝ
  /-- The distance from the circumcenter to side c -/
  dc : ℝ
  /-- The triangle is acute-angled -/
  acute : R > 0
  /-- The radii and distances are positive -/
  positive : r > 0 ∧ da > 0 ∧ db > 0 ∧ dc > 0

/-- The sum of distances from the circumcenter to the sides equals the sum of circumradius and inradius -/
theorem distance_sum_equals_radii_sum (t : AcuteTriangle) : t.da + t.db + t.dc = t.R + t.r := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_equals_radii_sum_l77_7781


namespace NUMINAMATH_CALUDE_point_on_number_line_l77_7749

theorem point_on_number_line (a : ℝ) : 
  (∃ (A : ℝ), A = 2 * a + 1 ∧ |A| = 3) → (a = 1 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l77_7749


namespace NUMINAMATH_CALUDE_unique_number_not_in_range_l77_7772

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ -d/c → f x = (a*x + b)/(c*x + d))
  (h11 : f 11 = 11)
  (h41 : f 41 = 41)
  (hinv : ∀ x, x ≠ -d/c → f (f x) = x) :
  ∃! y, ∀ x, f x ≠ y ∧ y = a/12 :=
sorry

end NUMINAMATH_CALUDE_unique_number_not_in_range_l77_7772


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l77_7732

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  let z := (10 * i) / (3 + i)
  Complex.im z = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l77_7732


namespace NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l77_7776

theorem sin_negative_31pi_over_6 : Real.sin (-31 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_31pi_over_6_l77_7776


namespace NUMINAMATH_CALUDE_lucas_avocados_l77_7789

/-- Calculates the number of avocados bought given initial money, cost per avocado, and change --/
def avocados_bought (initial_money change cost_per_avocado : ℚ) : ℚ :=
  (initial_money - change) / cost_per_avocado

/-- Proves that Lucas bought 3 avocados --/
theorem lucas_avocados :
  let initial_money : ℚ := 20
  let change : ℚ := 14
  let cost_per_avocado : ℚ := 2
  avocados_bought initial_money change cost_per_avocado = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucas_avocados_l77_7789


namespace NUMINAMATH_CALUDE_parabola_focus_l77_7752

/-- The focus of a parabola with equation x = 4y^2 is at (1/16, 0) -/
theorem parabola_focus (x y : ℝ) : 
  (x = 4 * y^2) → (∃ p : ℝ, p > 0 ∧ x = y^2 / (4 * p) ∧ (1 / (16 : ℝ), 0) = (p, 0)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l77_7752


namespace NUMINAMATH_CALUDE_square_sum_inequality_l77_7754

theorem square_sum_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l77_7754


namespace NUMINAMATH_CALUDE_adjacent_pairs_difference_l77_7713

/-- Given a circular arrangement of symbols, this theorem proves that the difference
    between the number of adjacent pairs of one symbol and the number of adjacent pairs
    of another symbol equals the difference in the total count of these symbols. -/
theorem adjacent_pairs_difference (p q a b : ℕ) : 
  (p + q > 0) →  -- Ensure the circle is not empty
  (a ≤ p) →      -- Number of X pairs cannot exceed total X's
  (b ≤ q) →      -- Number of 0 pairs cannot exceed total 0's
  (a = 0 → p ≤ 1) →  -- If no X pairs, at most one X
  (b = 0 → q ≤ 1) →  -- If no 0 pairs, at most one 0
  a - b = p - q :=
by sorry

end NUMINAMATH_CALUDE_adjacent_pairs_difference_l77_7713
