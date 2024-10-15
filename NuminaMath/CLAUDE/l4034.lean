import Mathlib

namespace NUMINAMATH_CALUDE_postage_cost_theorem_l4034_403484

/-- The floor function, representing the greatest integer less than or equal to x -/
def floor (x : ℝ) : ℤ := sorry

/-- The cost in cents for mailing a letter weighing W ounces -/
def postageCost (W : ℝ) : ℤ := sorry

theorem postage_cost_theorem (W : ℝ) : 
  postageCost W = -6 * floor (-W) :=
sorry

end NUMINAMATH_CALUDE_postage_cost_theorem_l4034_403484


namespace NUMINAMATH_CALUDE_quadratic_minimum_l4034_403419

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

theorem quadratic_minimum :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l4034_403419


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l4034_403416

theorem cubic_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 94 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l4034_403416


namespace NUMINAMATH_CALUDE_complex_modulus_sum_l4034_403421

theorem complex_modulus_sum : Complex.abs (3 - 3*I) + Complex.abs (3 + 3*I) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sum_l4034_403421


namespace NUMINAMATH_CALUDE_factorization_theorem_l4034_403434

theorem factorization_theorem (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (b + c) * (c + a) :=
by sorry

end NUMINAMATH_CALUDE_factorization_theorem_l4034_403434


namespace NUMINAMATH_CALUDE_henry_initial_games_count_l4034_403475

/-- The number of games Henry had initially -/
def henry_initial_games : ℕ := 33

/-- The number of games Neil had initially -/
def neil_initial_games : ℕ := 2

/-- The number of games Henry gave to Neil -/
def games_given : ℕ := 5

theorem henry_initial_games_count : 
  henry_initial_games = 33 :=
by
  have h1 : henry_initial_games - games_given = 4 * (neil_initial_games + games_given) :=
    sorry
  sorry

#check henry_initial_games_count

end NUMINAMATH_CALUDE_henry_initial_games_count_l4034_403475


namespace NUMINAMATH_CALUDE_shaded_area_is_two_thirds_l4034_403465

/-- Square PQRS with shaded regions -/
structure ShadedSquare where
  /-- Side length of the square PQRS -/
  side_length : ℝ
  /-- Side length of the first shaded square region -/
  first_region : ℝ
  /-- Side length of the outer square in the second shaded region -/
  second_region_outer : ℝ
  /-- Side length of the inner square in the second shaded region -/
  second_region_inner : ℝ
  /-- Side length of the outer square in the third shaded region -/
  third_region_outer : ℝ
  /-- Side length of the inner square in the third shaded region -/
  third_region_inner : ℝ

/-- Theorem stating that the shaded area is 2/3 of the total area -/
theorem shaded_area_is_two_thirds (sq : ShadedSquare)
    (h1 : sq.side_length = 6)
    (h2 : sq.first_region = 1)
    (h3 : sq.second_region_outer = 4)
    (h4 : sq.second_region_inner = 2)
    (h5 : sq.third_region_outer = 6)
    (h6 : sq.third_region_inner = 5) :
    (sq.first_region^2 + (sq.second_region_outer^2 - sq.second_region_inner^2) +
     (sq.third_region_outer^2 - sq.third_region_inner^2)) / sq.side_length^2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_two_thirds_l4034_403465


namespace NUMINAMATH_CALUDE_alice_number_sum_l4034_403486

/-- Represents the process of subtracting the smallest prime divisor from a number -/
def subtractSmallestPrimeDivisor (n : ℕ) : ℕ := sorry

/-- Returns true if the number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

theorem alice_number_sum : 
  ∀ n : ℕ, 
  n > 0 → 
  (isPrime (n.iterate subtractSmallestPrimeDivisor 2022)) → 
  (n = 4046 ∨ n = 4047) ∧ 
  (4046 + 4047 = 8093) := 
sorry

end NUMINAMATH_CALUDE_alice_number_sum_l4034_403486


namespace NUMINAMATH_CALUDE_f_range_l4034_403427

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 5*x + 2) / (x + 1)

theorem f_range :
  Set.range f = {y : ℝ | y > 0} := by sorry

end NUMINAMATH_CALUDE_f_range_l4034_403427


namespace NUMINAMATH_CALUDE_sum_of_abc_l4034_403458

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_abc_l4034_403458


namespace NUMINAMATH_CALUDE_distance_between_squares_l4034_403423

/-- Given two squares where:
  * The smaller square has a perimeter of 8 cm
  * The larger square has an area of 64 cm²
  * The bottom left corner of the larger square is 2 cm to the right of the top right corner of the smaller square
  Prove that the distance between the top right corner of the larger square (A) and 
  the top left corner of the smaller square (B) is √136 cm -/
theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ) (horizontal_shift : ℝ) :
  small_perimeter = 8 →
  large_area = 64 →
  horizontal_shift = 2 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal_distance := horizontal_shift + large_side
  let vertical_distance := large_side - small_side
  Real.sqrt (horizontal_distance ^ 2 + vertical_distance ^ 2) = Real.sqrt 136 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_squares_l4034_403423


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l4034_403448

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 10 = 0 ∧ 2 * s^2 - 4 * s - 10 = 0 ∧
   ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l4034_403448


namespace NUMINAMATH_CALUDE_solution_set_f_leq_0_range_of_m_l4034_403471

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the solution set of f(x) ≤ 0
theorem solution_set_f_leq_0 :
  {x : ℝ | f x ≤ 0} = {x : ℝ | x ≥ 1/3 ∨ x ≤ -3} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x - 2*m^2 ≤ 4*m} = {m : ℝ | m ≤ -5/2 ∨ m ≥ 1/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_0_range_of_m_l4034_403471


namespace NUMINAMATH_CALUDE_bus_journey_distance_l4034_403413

/-- Represents the distance traveled by a bus after k hours, given a total journey of 100 km -/
def distance_traveled (k : ℕ) : ℚ :=
  (100 * k) / (k + 1)

/-- Theorem stating that after 6 hours, the distance traveled is 600/7 km -/
theorem bus_journey_distance :
  distance_traveled 6 = 600 / 7 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_distance_l4034_403413


namespace NUMINAMATH_CALUDE_work_completion_time_l4034_403449

/-- The time required for x to complete a work given the combined time of x and y, and the time for y alone. -/
theorem work_completion_time (combined_time y_time : ℝ) (h1 : combined_time > 0) (h2 : y_time > 0) :
  let x_rate := 1 / combined_time - 1 / y_time
  x_rate > 0 → 1 / x_rate = y_time := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4034_403449


namespace NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l4034_403401

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = (x+1)(x-A) -/
def f (A : ℝ) (x : ℝ) : ℝ :=
  (x + 1) * (x - A)

/-- If f(x) = (x+1)(x-A) is an even function, then A = 1 -/
theorem even_function_implies_A_equals_one :
  ∀ A : ℝ, IsEven (f A) → A = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_A_equals_one_l4034_403401


namespace NUMINAMATH_CALUDE_power_boat_travel_time_l4034_403493

/-- The time it takes for the power boat to travel from A to B -/
def travel_time_AB : ℝ := 4

/-- The distance between dock A and dock B in km -/
def distance_AB : ℝ := 20

/-- The original speed of the river current -/
def river_speed : ℝ := sorry

/-- The speed of the power boat relative to the river -/
def boat_speed : ℝ := sorry

/-- The total time of the journey in hours -/
def total_time : ℝ := 12

theorem power_boat_travel_time :
  let increased_river_speed := 1.5 * river_speed
  let downstream_speed := boat_speed + river_speed
  let upstream_speed := boat_speed - increased_river_speed
  distance_AB / downstream_speed = travel_time_AB ∧
  distance_AB + upstream_speed * (total_time - travel_time_AB) = river_speed * total_time :=
by sorry

end NUMINAMATH_CALUDE_power_boat_travel_time_l4034_403493


namespace NUMINAMATH_CALUDE_b_completion_time_l4034_403414

/-- The number of days A needs to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days A works before B joins -/
def a_solo_days : ℝ := 2

/-- The total number of days A and B work together to complete the job -/
def total_days : ℝ := 8

/-- The number of days B needs to complete the work alone -/
def b_days : ℝ := 18

/-- The theorem stating that given the conditions, B can complete the work alone in 18 days -/
theorem b_completion_time :
  (a_days = 12) →
  (a_solo_days = 2) →
  (total_days = 8) →
  (b_days = 18) →
  (1 / a_days * a_solo_days + (total_days - a_solo_days) * (1 / a_days + 1 / b_days) = 1) :=
by sorry

end NUMINAMATH_CALUDE_b_completion_time_l4034_403414


namespace NUMINAMATH_CALUDE_john_puppy_profit_l4034_403464

/-- Calculates the profit from selling puppies given the initial conditions --/
def puppy_profit (initial_puppies : ℕ) (sale_price : ℕ) (stud_fee : ℕ) : ℕ :=
  let remaining_after_giving_away := initial_puppies / 2
  let remaining_after_keeping_one := remaining_after_giving_away - 1
  let total_sales := remaining_after_keeping_one * sale_price
  total_sales - stud_fee

/-- Proves that John's profit from selling puppies is $1500 --/
theorem john_puppy_profit : puppy_profit 8 600 300 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_john_puppy_profit_l4034_403464


namespace NUMINAMATH_CALUDE_equilateral_condition_obtuse_condition_two_triangles_condition_l4034_403415

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the properties we need to prove
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

def hasTwoConfigurations (a b : ℝ) (B : ℝ) : Prop :=
  ∃ (C₁ C₂ : ℝ), C₁ ≠ C₂ ∧
    (∃ (t₁ t₂ : Triangle), 
      t₁.a = a ∧ t₁.b = b ∧ t₁.B = B ∧ t₁.C = C₁ ∧
      t₂.a = a ∧ t₂.b = b ∧ t₂.B = B ∧ t₂.C = C₂)

-- State the theorems
theorem equilateral_condition (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) (h2 : t.B = Real.pi / 3) : 
  isEquilateral t := by sorry

theorem obtuse_condition (t : Triangle) 
  (h : Real.cos t.A^2 + Real.sin t.B^2 + Real.sin t.C^2 < 1) : 
  isObtuse t := by sorry

theorem two_triangles_condition :
  hasTwoConfigurations 4 2 (25 * Real.pi / 180) := by sorry

end NUMINAMATH_CALUDE_equilateral_condition_obtuse_condition_two_triangles_condition_l4034_403415


namespace NUMINAMATH_CALUDE_remaining_value_probability_theorem_l4034_403425

/-- Represents a bag of bills -/
structure Bag where
  tens : ℕ
  fives : ℕ
  ones : ℕ

/-- Calculates the total value of bills in a bag -/
def bagValue (b : Bag) : ℕ := 10 * b.tens + 5 * b.fives + b.ones

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the probability of the remaining value in Bag A being greater than Bag B -/
def remainingValueProbability (bagA bagB : Bag) : ℚ :=
  let totalA := choose (bagA.tens + bagA.fives + bagA.ones) 2
  let totalB := choose (bagB.tens + bagB.fives + bagB.ones) 2
  let favorableA := choose bagA.ones 2
  let favorableB := totalB - choose bagB.ones 2
  (favorableA * favorableB : ℚ) / (totalA * totalB : ℚ)

theorem remaining_value_probability_theorem :
  let bagA : Bag := { tens := 2, fives := 0, ones := 3 }
  let bagB : Bag := { tens := 0, fives := 4, ones := 3 }
  remainingValueProbability bagA bagB = 9 / 35 := by
  sorry

end NUMINAMATH_CALUDE_remaining_value_probability_theorem_l4034_403425


namespace NUMINAMATH_CALUDE_souvenir_cost_problem_l4034_403476

theorem souvenir_cost_problem (total_souvenirs : ℕ) (total_cost : ℚ) 
  (cheap_souvenirs : ℕ) (cheap_cost : ℚ) (expensive_souvenirs : ℕ) :
  total_souvenirs = 1000 →
  total_cost = 220 →
  cheap_souvenirs = 400 →
  cheap_cost = 1/4 →
  expensive_souvenirs = total_souvenirs - cheap_souvenirs →
  (total_cost - cheap_souvenirs * cheap_cost) / expensive_souvenirs = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_cost_problem_l4034_403476


namespace NUMINAMATH_CALUDE_f_has_minimum_l4034_403490

def f (x : ℝ) := |2*x + 1| - |x - 4|

theorem f_has_minimum : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) := by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_l4034_403490


namespace NUMINAMATH_CALUDE_y_value_l4034_403450

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l4034_403450


namespace NUMINAMATH_CALUDE_base_conversion_1357_to_base_5_l4034_403432

theorem base_conversion_1357_to_base_5 :
  (2 * 5^4 + 0 * 5^3 + 4 * 5^2 + 1 * 5^1 + 2 * 5^0 : ℕ) = 1357 := by
  sorry

#eval 2 * 5^4 + 0 * 5^3 + 4 * 5^2 + 1 * 5^1 + 2 * 5^0

end NUMINAMATH_CALUDE_base_conversion_1357_to_base_5_l4034_403432


namespace NUMINAMATH_CALUDE_iron_bucket_area_l4034_403482

/-- The area of iron sheet needed for a rectangular bucket -/
def bucket_area (length width height : ℝ) : ℝ :=
  length * width + 2 * (length * height + width * height)

/-- Theorem: The area of iron sheet needed for the specified bucket is 1.24 square meters -/
theorem iron_bucket_area :
  let length : ℝ := 0.4
  let width : ℝ := 0.3
  let height : ℝ := 0.8
  bucket_area length width height = 1.24 := by
  sorry


end NUMINAMATH_CALUDE_iron_bucket_area_l4034_403482


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4034_403438

/-- An isosceles triangle with two sides of length 6 and one side of length 2 has a perimeter of 14. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 6 ∧ b = 6 ∧ c = 2 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  a + b + c = 14 :=
by
  sorry

#check isosceles_triangle_perimeter

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4034_403438


namespace NUMINAMATH_CALUDE_max_quotient_value_l4034_403405

theorem max_quotient_value (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) :
  (∀ x y, 300 ≤ x ∧ x ≤ 500 → 800 ≤ y ∧ y ≤ 1600 → 2 * y / x ≤ 32 / 3) ∧
  (∃ x y, 300 ≤ x ∧ x ≤ 500 ∧ 800 ≤ y ∧ y ≤ 1600 ∧ 2 * y / x = 32 / 3) :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_value_l4034_403405


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4034_403469

theorem sufficient_not_necessary (x : ℝ) : 
  (∃ (S T : Set ℝ), 
    S = {x | x > 2} ∧ 
    T = {x | x^2 - 3*x + 2 > 0} ∧ 
    S ⊂ T ∧ 
    ∃ y, y ∈ T ∧ y ∉ S) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4034_403469


namespace NUMINAMATH_CALUDE_clock_time_sum_l4034_403417

/-- Represents time on a 12-hour digital clock -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

def addTime (start : ClockTime) (hours minutes seconds : Nat) : ClockTime :=
  let totalSeconds := start.hours * 3600 + start.minutes * 60 + start.seconds +
                      hours * 3600 + minutes * 60 + seconds
  let newSeconds := totalSeconds % 86400  -- 24 hours in seconds
  { hours := (newSeconds / 3600) % 12,
    minutes := (newSeconds % 3600) / 60,
    seconds := newSeconds % 60 }

def sumDigits (time : ClockTime) : Nat :=
  time.hours + time.minutes + time.seconds

theorem clock_time_sum (startTime : ClockTime) :
  let endTime := addTime startTime 189 58 52
  sumDigits endTime = 122 := by
  sorry

end NUMINAMATH_CALUDE_clock_time_sum_l4034_403417


namespace NUMINAMATH_CALUDE_wire_length_proof_l4034_403481

theorem wire_length_proof (side_length : ℝ) (total_area : ℝ) (original_length : ℝ) : 
  side_length = 2 →
  total_area = 92 →
  original_length = (total_area / (side_length ^ 2)) * (4 * side_length) →
  original_length = 184 := by
  sorry

#check wire_length_proof

end NUMINAMATH_CALUDE_wire_length_proof_l4034_403481


namespace NUMINAMATH_CALUDE_f_min_max_l4034_403472

-- Define the function
def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- State the theorem
theorem f_min_max :
  (∃ x₁ : ℝ, f x₁ = -1 ∧ ∀ x : ℝ, f x ≥ -1) ∧
  (∃ x₂ : ℝ, f x₂ = 3 ∧ ∀ x : ℝ, f x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_f_min_max_l4034_403472


namespace NUMINAMATH_CALUDE_total_birds_in_store_l4034_403404

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 7

/-- Theorem: The total number of birds in the pet store is 54 -/
theorem total_birds_in_store : 
  num_cages * (parrots_per_cage + parakeets_per_cage) = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_store_l4034_403404


namespace NUMINAMATH_CALUDE_train_stoppage_time_l4034_403477

/-- Given a train with speeds excluding and including stoppages, 
    calculate the number of minutes the train stops per hour. -/
theorem train_stoppage_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 48 → speed_with_stops = 36 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 15 := by
  sorry

#check train_stoppage_time

end NUMINAMATH_CALUDE_train_stoppage_time_l4034_403477


namespace NUMINAMATH_CALUDE_circle_through_points_l4034_403442

/-- The general equation of a circle -/
def CircleEquation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle equation we want to prove -/
def SpecificCircle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the specific circle equation passes through the given points -/
theorem circle_through_points :
  (∀ D E F : ℝ, CircleEquation 0 0 D E F → CircleEquation 4 0 D E F → CircleEquation (-1) 1 D E F
    → ∀ x y : ℝ, CircleEquation x y D E F ↔ SpecificCircle x y) := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_l4034_403442


namespace NUMINAMATH_CALUDE_average_age_proof_l4034_403445

/-- Given three people a, b, and c, prove that if their average age is 25 years
    and b's age is 17 years, then the average age of a and c is 29 years. -/
theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 25 → b = 17 → (a + c) / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l4034_403445


namespace NUMINAMATH_CALUDE_welders_left_correct_l4034_403444

/-- The number of welders who left after the first day -/
def welders_who_left : ℕ := 9

/-- The initial number of welders -/
def initial_welders : ℕ := 12

/-- The number of days to complete the order with all welders -/
def initial_days : ℕ := 3

/-- The number of additional days needed after some welders left -/
def additional_days : ℕ := 8

theorem welders_left_correct :
  ∃ (r : ℝ), r > 0 ∧
  initial_welders * r * initial_days = (initial_welders - welders_who_left) * r * (1 + additional_days) :=
by sorry

end NUMINAMATH_CALUDE_welders_left_correct_l4034_403444


namespace NUMINAMATH_CALUDE_sector_arc_length_l4034_403441

theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = 90) (h2 : r = 6) :
  (θ / 360) * (2 * Real.pi * r) = 3 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_sector_arc_length_l4034_403441


namespace NUMINAMATH_CALUDE_banana_split_difference_l4034_403491

/-- The number of ice cream scoops in Oli's banana split -/
def oli_scoops : ℕ := 4

/-- The number of ice cream scoops in Victoria's banana split -/
def victoria_scoops : ℕ := 2 * oli_scoops + oli_scoops

/-- The number of ice cream scoops in Brian's banana split -/
def brian_scoops : ℕ := oli_scoops + 3

/-- The total difference in scoops of ice cream between Oli's, Victoria's, and Brian's banana splits -/
def total_difference : ℕ := 
  (victoria_scoops - oli_scoops) + (brian_scoops - oli_scoops) + (victoria_scoops - brian_scoops)

theorem banana_split_difference : total_difference = 16 := by
  sorry

end NUMINAMATH_CALUDE_banana_split_difference_l4034_403491


namespace NUMINAMATH_CALUDE_large_cups_sold_is_five_l4034_403431

/-- Represents the number of cups sold for each size --/
structure CupsSold where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total revenue based on the number of cups sold --/
def totalRevenue (cups : CupsSold) : ℕ :=
  cups.small + 2 * cups.medium + 3 * cups.large

theorem large_cups_sold_is_five :
  ∃ (cups : CupsSold),
    totalRevenue cups = 50 ∧
    cups.small = 11 ∧
    2 * cups.medium = 24 ∧
    cups.large = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_cups_sold_is_five_l4034_403431


namespace NUMINAMATH_CALUDE_time_after_2700_minutes_l4034_403499

-- Define a custom type for time
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

-- Define a function to add minutes to a given time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

-- Define the starting time (6:00 a.m.)
def startTime : Time := { hours := 6, minutes := 0 }

-- Define the number of minutes to add
def minutesToAdd : Nat := 2700

-- Define the expected end time (3:00 a.m. the next day)
def expectedEndTime : Time := { hours := 3, minutes := 0 }

-- Theorem statement
theorem time_after_2700_minutes :
  addMinutes startTime minutesToAdd = expectedEndTime := by
  sorry

end NUMINAMATH_CALUDE_time_after_2700_minutes_l4034_403499


namespace NUMINAMATH_CALUDE_fraction_equality_l4034_403457

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 9)
  (h2 : s / r = 6)
  (h3 : s / t = 1 / 2) :
  t / q = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l4034_403457


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l4034_403480

theorem negation_of_universal_quantifier (x : ℝ) :
  (¬ ∀ m : ℝ, m ∈ Set.Icc 0 1 → x + 1 / x ≥ 2^m) ↔
  (∃ m : ℝ, m ∈ Set.Icc 0 1 ∧ x + 1 / x < 2^m) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l4034_403480


namespace NUMINAMATH_CALUDE_pizza_slices_l4034_403435

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (slices_per_pizza : ℕ) 
  (h1 : total_pizzas = 7)
  (h2 : total_slices = 14)
  (h3 : total_slices = total_pizzas * slices_per_pizza) :
  slices_per_pizza = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l4034_403435


namespace NUMINAMATH_CALUDE_kayla_total_items_l4034_403495

/-- Represents the number of items bought by a person -/
structure Items :=
  (chocolate_bars : ℕ)
  (soda_cans : ℕ)

/-- The total number of items -/
def Items.total (i : Items) : ℕ := i.chocolate_bars + i.soda_cans

/-- Theresa bought twice the number of items as Kayla -/
def twice (kayla : Items) (theresa : Items) : Prop :=
  theresa.chocolate_bars = 2 * kayla.chocolate_bars ∧
  theresa.soda_cans = 2 * kayla.soda_cans

theorem kayla_total_items 
  (kayla theresa : Items)
  (h1 : twice kayla theresa)
  (h2 : theresa.chocolate_bars = 12)
  (h3 : theresa.soda_cans = 18) :
  kayla.total = 15 :=
by sorry

end NUMINAMATH_CALUDE_kayla_total_items_l4034_403495


namespace NUMINAMATH_CALUDE_min_value_of_M_l4034_403436

theorem min_value_of_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 →
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) ≥ 5 * Real.sqrt 34 / 12) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) = 5 * Real.sqrt 34 / 12) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_M_l4034_403436


namespace NUMINAMATH_CALUDE_vasya_initial_larger_l4034_403452

/-- Represents the initial investments and profit rates for Vasya and Petya --/
structure InvestmentScenario where
  vasya_initial : ℝ
  petya_initial : ℝ
  vasya_rate : ℝ
  petya_rate : ℝ
  exchange_rate_increase : ℝ

/-- Calculates the profit for a given initial investment and rate --/
def profit (initial : ℝ) (rate : ℝ) : ℝ := initial * rate

/-- Calculates Petya's effective rate considering exchange rate increase --/
def petya_effective_rate (petya_rate : ℝ) (exchange_rate_increase : ℝ) : ℝ :=
  1 + petya_rate + exchange_rate_increase + petya_rate * exchange_rate_increase

/-- Theorem stating that Vasya's initial investment is larger given equal profits --/
theorem vasya_initial_larger (scenario : InvestmentScenario) 
  (h1 : scenario.vasya_rate = 0.20)
  (h2 : scenario.petya_rate = 0.10)
  (h3 : scenario.exchange_rate_increase = 0.095)
  (h4 : profit scenario.vasya_initial scenario.vasya_rate = 
        profit scenario.petya_initial (petya_effective_rate scenario.petya_rate scenario.exchange_rate_increase)) :
  scenario.vasya_initial > scenario.petya_initial := by
  sorry


end NUMINAMATH_CALUDE_vasya_initial_larger_l4034_403452


namespace NUMINAMATH_CALUDE_min_value_inequality_l4034_403439

theorem min_value_inequality (r s t : ℝ) (h1 : 1 ≤ r) (h2 : r ≤ s) (h3 : s ≤ t) (h4 : t ≤ 4) :
  (r - 1)^2 + (s/r - 1)^2 + (t/s - 1)^2 + (4/t - 1)^2 ≥ 4 * (Real.sqrt 2 - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l4034_403439


namespace NUMINAMATH_CALUDE_adam_initial_amount_l4034_403492

/-- The cost of the airplane in dollars -/
def airplane_cost : ℚ := 4.28

/-- The change Adam receives after buying the airplane in dollars -/
def change_received : ℚ := 0.72

/-- Adam's initial amount of money in dollars -/
def initial_amount : ℚ := airplane_cost + change_received

theorem adam_initial_amount :
  initial_amount = 5 :=
by sorry

end NUMINAMATH_CALUDE_adam_initial_amount_l4034_403492


namespace NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l4034_403451

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 2)

-- Define line l1
def line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Define line l2
def line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y + 2 = 0

-- Theorem for parallel line l1
theorem parallel_line_theorem :
  (∀ x y : ℝ, line_l1 x y ↔ 3 * x + 4 * y - 11 = 0) ∧
  (line_l1 (point_A.1) (point_A.2)) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, line_l x y ↔ line_l1 (k * x) (k * y)) :=
sorry

-- Theorem for perpendicular line l2
theorem perpendicular_line_theorem :
  (∀ x y : ℝ, line_l2 x y ↔ 4 * x - 3 * y + 2 = 0) ∧
  (line_l2 (point_A.1) (point_A.2)) ∧
  (∀ x1 y1 x2 y2 : ℝ, line_l x1 y1 → line_l x2 y2 →
    3 * (x2 - x1) + 4 * (y2 - y1) = 0 →
    4 * (x2 - x1) - 3 * (y2 - y1) = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_theorem_perpendicular_line_theorem_l4034_403451


namespace NUMINAMATH_CALUDE_figure_area_is_74_l4034_403460

/-- Represents the dimensions of the composite rectangular figure -/
structure FigureDimensions where
  height : ℕ
  width1 : ℕ
  width2 : ℕ
  width3 : ℕ
  height2 : ℕ
  height3 : ℕ

/-- Calculates the area of the composite rectangular figure -/
def calculateArea (d : FigureDimensions) : ℕ :=
  d.height * d.width1 + 
  (d.height - d.height2) * d.width2 +
  d.height2 * d.width2 +
  (d.height - d.height3) * d.width3

/-- Theorem stating that the area of the figure with given dimensions is 74 square units -/
theorem figure_area_is_74 (d : FigureDimensions) 
  (h1 : d.height = 7)
  (h2 : d.width1 = 6)
  (h3 : d.width2 = 4)
  (h4 : d.width3 = 5)
  (h5 : d.height2 = 2)
  (h6 : d.height3 = 6) :
  calculateArea d = 74 := by
  sorry

#eval calculateArea { height := 7, width1 := 6, width2 := 4, width3 := 5, height2 := 2, height3 := 6 }

end NUMINAMATH_CALUDE_figure_area_is_74_l4034_403460


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l4034_403411

open Real

theorem solution_set_of_inequality (f : ℝ → ℝ) (f_diff : Differentiable ℝ f) :
  (f 0 = 2) →
  (∀ x, f x + deriv f x > 1) →
  (∀ x, (exp x * f x > exp x + 1) ↔ x > 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l4034_403411


namespace NUMINAMATH_CALUDE_race_head_start_l4034_403461

/-- Proves the head start distance in a race with given conditions -/
theorem race_head_start 
  (race_distance : ℝ) 
  (speed_ratio : ℝ) 
  (win_margin : ℝ) 
  (h1 : race_distance = 600)
  (h2 : speed_ratio = 5/4)
  (h3 : win_margin = 200) :
  ∃ (head_start : ℝ), 
    head_start = 100 ∧ 
    (race_distance - head_start) / speed_ratio = (race_distance - win_margin) / 1 :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l4034_403461


namespace NUMINAMATH_CALUDE_union_covers_reals_a_equals_complement_b_l4034_403497

open Set Real

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | -m ≤ x - 2 ∧ x - 2 ≤ m}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 4}

-- Part 1: A ∪ B = ℝ iff m ≥ 4
theorem union_covers_reals (m : ℝ) : A m ∪ B = univ ↔ m ≥ 4 := by sorry

-- Part 2: A = ℝ\B iff 0 < m < 2
theorem a_equals_complement_b (m : ℝ) : A m = Bᶜ ↔ 0 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_union_covers_reals_a_equals_complement_b_l4034_403497


namespace NUMINAMATH_CALUDE_rhombus_area_l4034_403454

/-- The area of a rhombus with diagonals of 6cm and 8cm is 24cm². -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) :
  (1 / 2) * d1 * d2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4034_403454


namespace NUMINAMATH_CALUDE_weight_replacement_l4034_403424

theorem weight_replacement (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 8 → 
  new_avg - old_avg = 2.5 →
  new_weight = 70 →
  (n * new_avg - new_weight + (n * old_avg - n * new_avg)) / (n - 1) = 50 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l4034_403424


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l4034_403487

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (l m : Line) (α : Plane) : 
  perpendicular l α → parallel l m → perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l4034_403487


namespace NUMINAMATH_CALUDE_candidate_a_votes_l4034_403494

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 80 / 100

theorem candidate_a_votes : 
  (1 - invalid_percentage) * candidate_a_percentage * total_votes = 380800 := by
  sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l4034_403494


namespace NUMINAMATH_CALUDE_stationery_difference_l4034_403468

def georgia_stationery : ℚ := 25

def lorene_stationery : ℚ := 3 * georgia_stationery

def bria_stationery : ℚ := georgia_stationery + 10

def darren_stationery : ℚ := bria_stationery / 2

theorem stationery_difference :
  lorene_stationery + bria_stationery + darren_stationery - georgia_stationery = 102.5 := by
  sorry

end NUMINAMATH_CALUDE_stationery_difference_l4034_403468


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l4034_403474

theorem complex_expression_simplification :
  3 / Real.sqrt 3 - (Real.sqrt 3)^2 - Real.sqrt 27 + |Real.sqrt 3 - 2| = -1 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l4034_403474


namespace NUMINAMATH_CALUDE_greatest_int_prime_abs_quadratic_l4034_403430

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def f (x : ℤ) : ℤ := |4*x^2 - 39*x + 21|

theorem greatest_int_prime_abs_quadratic : 
  ∀ x : ℤ, x > 8 → ¬(is_prime (f x).toNat) ∧ is_prime (f 8).toNat :=
sorry

end NUMINAMATH_CALUDE_greatest_int_prime_abs_quadratic_l4034_403430


namespace NUMINAMATH_CALUDE_weight_of_A_l4034_403453

theorem weight_of_A (A B C D : ℝ) : 
  (A + B + C) / 3 = 84 →
  (A + B + C + D) / 4 = 80 →
  (B + C + D + (D + 8)) / 4 = 79 →
  A = 80 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_A_l4034_403453


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4034_403440

theorem complex_fraction_simplification :
  (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4034_403440


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l4034_403402

def total_beads : ℕ := 6
def red_beads : ℕ := 3
def white_beads : ℕ := 2
def blue_beads : ℕ := 1

def total_arrangements : ℕ := (Nat.factorial total_beads) / ((Nat.factorial red_beads) * (Nat.factorial white_beads) * (Nat.factorial blue_beads))

def valid_arrangements : ℕ := 10

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l4034_403402


namespace NUMINAMATH_CALUDE_no_six_if_mean_and_median_two_l4034_403420

/-- Represents the result of 5 dice rolls -/
def DiceRolls := Fin 5 → Nat

/-- The mean of the dice rolls is 2 -/
def mean_is_2 (rolls : DiceRolls) : Prop :=
  (rolls 0 + rolls 1 + rolls 2 + rolls 3 + rolls 4) / 5 = 2

/-- The median of the dice rolls is 2 -/
def median_is_2 (rolls : DiceRolls) : Prop :=
  ∃ (p : Equiv (Fin 5) (Fin 5)), 
    rolls (p 2) = 2 ∧ 
    (∀ i < 2, rolls (p i) ≤ 2) ∧ 
    (∀ i > 2, rolls (p i) ≥ 2)

/-- The theorem stating that if the mean and median are 2, then 6 cannot appear in the rolls -/
theorem no_six_if_mean_and_median_two (rolls : DiceRolls) 
  (h_mean : mean_is_2 rolls) (h_median : median_is_2 rolls) : 
  ∀ i, rolls i ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_no_six_if_mean_and_median_two_l4034_403420


namespace NUMINAMATH_CALUDE_carbonic_acid_weight_l4034_403455

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- Number of Hydrogen atoms in Carbonic acid -/
def num_H : ℕ := 2

/-- Number of Carbon atoms in Carbonic acid -/
def num_C : ℕ := 1

/-- Number of Oxygen atoms in Carbonic acid -/
def num_O : ℕ := 3

/-- Number of moles of Carbonic acid -/
def num_moles : ℝ := 8

/-- Molecular weight of Carbonic acid in g/mol -/
def molecular_weight_H2CO3 : ℝ := 
  num_H * atomic_weight_H + num_C * atomic_weight_C + num_O * atomic_weight_O

/-- Total weight of given moles of Carbonic acid in grams -/
def total_weight : ℝ := num_moles * molecular_weight_H2CO3

theorem carbonic_acid_weight : total_weight = 496.192 := by
  sorry

end NUMINAMATH_CALUDE_carbonic_acid_weight_l4034_403455


namespace NUMINAMATH_CALUDE_total_songs_in_june_l4034_403406

def june_days : ℕ := 30
def weekend_days : ℕ := 8
def holiday_days : ℕ := 1
def vivian_songs_per_day : ℕ := 10
def clara_songs_per_day : ℕ := vivian_songs_per_day - 2
def lucas_songs_per_day : ℕ := vivian_songs_per_day + 5

theorem total_songs_in_june :
  let playing_days : ℕ := june_days - weekend_days - holiday_days
  let vivian_total : ℕ := playing_days * vivian_songs_per_day
  let clara_total : ℕ := playing_days * clara_songs_per_day
  let lucas_total : ℕ := playing_days * lucas_songs_per_day
  vivian_total + clara_total + lucas_total = 693 := by
  sorry

end NUMINAMATH_CALUDE_total_songs_in_june_l4034_403406


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l4034_403410

theorem magnitude_of_complex_number (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l4034_403410


namespace NUMINAMATH_CALUDE_pie_price_is_seven_l4034_403403

def number_of_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def number_of_pies : ℕ := 126
def total_earnings : ℕ := 6318

theorem pie_price_is_seven :
  ∃ (price_per_pie : ℕ),
    price_per_pie = 7 ∧
    price_per_pie * number_of_pies + price_per_cake * number_of_cakes = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_pie_price_is_seven_l4034_403403


namespace NUMINAMATH_CALUDE_mixed_number_multiplication_l4034_403447

theorem mixed_number_multiplication (a b c d e f : ℚ) :
  a + b / c = -3 ∧ b / c = 3 / 4 ∧ d / e = 5 / 7 →
  (a + b / c) * (d / e) = (a - b / c) * (d / e) := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_multiplication_l4034_403447


namespace NUMINAMATH_CALUDE_fitness_equipment_problem_l4034_403437

/-- Unit price of type A fitness equipment -/
def unit_price_A : ℝ := 360

/-- Unit price of type B fitness equipment -/
def unit_price_B : ℝ := 540

/-- Total number of fitness equipment to be purchased -/
def total_equipment : ℕ := 50

/-- Maximum total cost allowed -/
def max_total_cost : ℝ := 21000

/-- Theorem stating the conditions and conclusions of the fitness equipment problem -/
theorem fitness_equipment_problem :
  (unit_price_B = 1.5 * unit_price_A) ∧
  (7200 / unit_price_A - 5400 / unit_price_B = 10) ∧
  (∀ x : ℕ, x ≤ total_equipment →
    unit_price_A * x + unit_price_B * (total_equipment - x) ≤ max_total_cost →
    x ≥ 34) :=
sorry

end NUMINAMATH_CALUDE_fitness_equipment_problem_l4034_403437


namespace NUMINAMATH_CALUDE_weight_of_N2O3_l4034_403418

/-- The molar mass of nitrogen in g/mol -/
def molar_mass_N : ℝ := 14.01

/-- The molar mass of oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- The number of moles of N2O3 -/
def moles_N2O3 : ℝ := 7

/-- The molar mass of N2O3 in g/mol -/
def molar_mass_N2O3 : ℝ := 2 * molar_mass_N + 3 * molar_mass_O

/-- The weight of N2O3 in grams -/
def weight_N2O3 : ℝ := moles_N2O3 * molar_mass_N2O3

theorem weight_of_N2O3 : weight_N2O3 = 532.14 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_N2O3_l4034_403418


namespace NUMINAMATH_CALUDE_mean_score_is_74_9_l4034_403459

structure ScoreDistribution where
  score : ℕ
  num_students : ℕ

def total_students : ℕ := 100

def score_data : List ScoreDistribution := [
  ⟨100, 10⟩,
  ⟨90, 15⟩,
  ⟨80, 20⟩,
  ⟨70, 30⟩,
  ⟨60, 20⟩,
  ⟨50, 4⟩,
  ⟨40, 1⟩
]

def sum_scores : ℕ := (score_data.map (λ x => x.score * x.num_students)).sum

theorem mean_score_is_74_9 : 
  (sum_scores : ℚ) / total_students = 749 / 10 := by
  sorry

end NUMINAMATH_CALUDE_mean_score_is_74_9_l4034_403459


namespace NUMINAMATH_CALUDE_f_plus_g_positive_implies_m_bound_l4034_403470

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) = e^x / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

/-- The function g(x) = mx -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x

/-- Theorem stating that if f(x) + g(x) > 0 for all x > 0, then m > -e^2/4 -/
theorem f_plus_g_positive_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, x > 0 → f x + g m x > 0) →
  m > -(e^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_f_plus_g_positive_implies_m_bound_l4034_403470


namespace NUMINAMATH_CALUDE_cosine_sum_special_case_l4034_403488

theorem cosine_sum_special_case : 
  Real.cos (π/12) * Real.cos (π/6) - Real.sin (π/12) * Real.sin (π/6) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_special_case_l4034_403488


namespace NUMINAMATH_CALUDE_square_root_of_four_l4034_403498

theorem square_root_of_four : ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l4034_403498


namespace NUMINAMATH_CALUDE_cone_volume_with_inscribed_square_l4034_403466

/-- The volume of a cone with a square inscribed in its base --/
theorem cone_volume_with_inscribed_square (a α : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < π) :
  let r := a * Real.sqrt 2 / 2
  let h := a * Real.sqrt (Real.cos α) / (2 * Real.sin (α/2) ^ 2)
  π * r^2 * h / 3 = π * a^3 * Real.sqrt (Real.cos α) / (12 * Real.sin (α/2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_with_inscribed_square_l4034_403466


namespace NUMINAMATH_CALUDE_tan_2alpha_values_l4034_403422

theorem tan_2alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4 / 3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_values_l4034_403422


namespace NUMINAMATH_CALUDE_quadratic_one_root_l4034_403489

/-- If the quadratic x^2 + 6mx + 2m has exactly one real root, then m = 2/9 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 + 6*m*x + 2*m = 0) → m = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l4034_403489


namespace NUMINAMATH_CALUDE_sum_and_divide_l4034_403446

theorem sum_and_divide : (40 + 5) / 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_divide_l4034_403446


namespace NUMINAMATH_CALUDE_rare_coin_value_l4034_403462

/-- Given a collection of rare coins where 4 coins are worth 16 dollars, 
    prove that 20 coins of the same type are worth 80 dollars. -/
theorem rare_coin_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℚ) :
  total_coins = 20 →
  sample_coins = 4 →
  sample_value = 16 →
  (total_coins : ℚ) * (sample_value / sample_coins) = 80 :=
by sorry

end NUMINAMATH_CALUDE_rare_coin_value_l4034_403462


namespace NUMINAMATH_CALUDE_sasha_can_buy_everything_l4034_403429

-- Define the store's discount policy and item prices
def discount_threshold : ℝ := 1500
def discount_rate : ℝ := 0.26
def shashlik_price : ℝ := 350
def sauce_price : ℝ := 70

-- Define Sasha's budget and desired quantities
def budget : ℝ := 1800
def shashlik_quantity : ℝ := 5
def sauce_quantity : ℝ := 1

-- Define a function to calculate the discounted price
def discounted_price (price : ℝ) : ℝ := price * (1 - discount_rate)

-- Theorem: Sasha can buy everything he planned within his budget
theorem sasha_can_buy_everything :
  ∃ (first_shashlik second_shashlik first_sauce : ℝ),
    first_shashlik + second_shashlik = shashlik_quantity ∧
    first_sauce = sauce_quantity ∧
    first_shashlik * shashlik_price + first_sauce * sauce_price ≥ discount_threshold ∧
    (first_shashlik * shashlik_price + first_sauce * sauce_price) +
    (second_shashlik * (discounted_price shashlik_price)) ≤ budget :=
  sorry

end NUMINAMATH_CALUDE_sasha_can_buy_everything_l4034_403429


namespace NUMINAMATH_CALUDE_series_sum_l4034_403485

/-- The sum of the infinite series ∑(k=1 to ∞) k/4^k is equal to 4/9 -/
theorem series_sum : ∑' k, k / (4 : ℝ) ^ k = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l4034_403485


namespace NUMINAMATH_CALUDE_inequality_holds_iff_c_equals_one_l4034_403483

theorem inequality_holds_iff_c_equals_one (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ c : ℝ, c > 0 ∧ ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (x^3 * y + y^3 * z + z^3 * x) / (x + y + z) + 4 * c / (x * y * z) ≥ 2 * c + 2) ↔
  (∃ c : ℝ, c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_c_equals_one_l4034_403483


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l4034_403409

/-- Given a point M(3, -4), its symmetric point with respect to the x-axis has coordinates (3, 4) -/
theorem symmetric_point_x_axis : 
  let M : ℝ × ℝ := (3, -4)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point M = (3, 4) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l4034_403409


namespace NUMINAMATH_CALUDE_bus_empty_seats_l4034_403412

/-- Calculates the number of empty seats on a bus after a series of boarding and disembarking events -/
def empty_seats_after_events (rows : ℕ) (seats_per_row : ℕ) 
  (initial_boarding : ℕ) 
  (stop1_board : ℕ) (stop1_disembark : ℕ)
  (stop2_board : ℕ) (stop2_disembark : ℕ)
  (stop3_board : ℕ) (stop3_disembark : ℕ) : ℕ :=
  let total_seats := rows * seats_per_row
  let after_initial := total_seats - initial_boarding
  let after_stop1 := after_initial - (stop1_board - stop1_disembark)
  let after_stop2 := after_stop1 - (stop2_board - stop2_disembark)
  let after_stop3 := after_stop2 - (stop3_board - stop3_disembark)
  after_stop3

theorem bus_empty_seats : 
  empty_seats_after_events 23 4 16 15 3 17 10 12 8 = 53 := by
  sorry

end NUMINAMATH_CALUDE_bus_empty_seats_l4034_403412


namespace NUMINAMATH_CALUDE_stationary_rigid_body_l4034_403408

/-- A point in a two-dimensional plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A rigid body in a two-dimensional plane -/
structure RigidBody2D where
  points : Set Point2D

/-- Three points are non-collinear if they do not lie on the same straight line -/
def NonCollinear (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

/-- A rigid body is stationary if it has no translational or rotational motion -/
def IsStationary (body : RigidBody2D) : Prop :=
  ∃ (p1 p2 p3 : Point2D), p1 ∈ body.points ∧ p2 ∈ body.points ∧ p3 ∈ body.points ∧
    NonCollinear p1 p2 p3

theorem stationary_rigid_body (body : RigidBody2D) :
  IsStationary body ↔ ∃ (p1 p2 p3 : Point2D), p1 ∈ body.points ∧ p2 ∈ body.points ∧ p3 ∈ body.points ∧
    NonCollinear p1 p2 p3 :=
  sorry

end NUMINAMATH_CALUDE_stationary_rigid_body_l4034_403408


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l4034_403496

def front_seats : Nat := 4
def back_seats : Nat := 5
def people_to_seat : Nat := 2

def is_adjacent (row1 row2 seat1 seat2 : Nat) : Bool :=
  (row1 = row2 ∧ seat2 = seat1 + 1) ∨
  (row1 = 1 ∧ row2 = 2 ∧ (seat1 = seat2 ∨ seat1 + 1 = seat2))

def count_seating_arrangements : Nat :=
  let total_seats := front_seats + back_seats
  (total_seats.choose people_to_seat) -
  (front_seats - 1 + back_seats - 1 + front_seats)

theorem seating_arrangements_count :
  count_seating_arrangements = 58 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l4034_403496


namespace NUMINAMATH_CALUDE_multiple_problem_l4034_403426

theorem multiple_problem (x : ℝ) (m : ℝ) (h1 : x = -4.5) (h2 : 10 * x = m * x - 36) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l4034_403426


namespace NUMINAMATH_CALUDE_f_increasing_and_even_l4034_403428

-- Define the function f(x) = x^2
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_increasing_and_even :
  -- f is monotonically increasing on (0, +∞)
  (∀ x y : ℝ, 0 < x ∧ x < y → f x < f y) ∧
  -- f is an even function
  (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_and_even_l4034_403428


namespace NUMINAMATH_CALUDE_min_value_theorem_l4034_403478

/-- Given x > 0 and y > 0 satisfying ln(xy)^y = e^x, 
    the minimum value of x^2y - ln x - x is 1 -/
theorem min_value_theorem (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x * y) ^ y = Real.exp x) : 
  ∃ (z : ℝ), z = 1 ∧ ∀ (w : ℝ), x^2 * y - Real.log x - x ≥ w → z ≤ w :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4034_403478


namespace NUMINAMATH_CALUDE_initial_daily_consumption_l4034_403456

/-- Proves that the initial daily consumption per soldier is 3 kg -/
theorem initial_daily_consumption (initial_soldiers : ℕ) (initial_days : ℕ) 
  (new_soldiers : ℕ) (new_days : ℕ) (new_consumption : ℚ) : 
  initial_soldiers = 1200 →
  initial_days = 30 →
  new_soldiers = 528 →
  new_days = 25 →
  new_consumption = 5/2 →
  (initial_soldiers * initial_days * (3 : ℚ) = 
   (initial_soldiers + new_soldiers) * new_days * new_consumption) := by
  sorry

end NUMINAMATH_CALUDE_initial_daily_consumption_l4034_403456


namespace NUMINAMATH_CALUDE_total_apples_l4034_403433

theorem total_apples (cecile_apples diane_apples : ℕ) : 
  cecile_apples = 15 → 
  diane_apples = cecile_apples + 20 → 
  cecile_apples + diane_apples = 50 := by
sorry

end NUMINAMATH_CALUDE_total_apples_l4034_403433


namespace NUMINAMATH_CALUDE_two_valid_m_values_l4034_403407

theorem two_valid_m_values : 
  ∃! (s : Finset ℕ), 
    (∀ m ∈ s, m > 0 ∧ (3087 : ℤ) ∣ (m^2 - 3)) ∧ 
    (∀ m : ℕ, m > 0 ∧ (3087 : ℤ) ∣ (m^2 - 3) → m ∈ s) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_m_values_l4034_403407


namespace NUMINAMATH_CALUDE_circuit_equation_l4034_403473

/-- Given voltage and impedance, prove the current satisfies the equation V = IZ -/
theorem circuit_equation (V Z I : ℂ) (hV : V = 2 + 3*I) (hZ : Z = 2 - I) : 
  V = I * Z ↔ I = (1 : ℝ)/5 + (8 : ℝ)/5 * I :=
sorry

end NUMINAMATH_CALUDE_circuit_equation_l4034_403473


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l4034_403479

/-- The line x = my + 1 intersects the parabola y² = x at two distinct points for any real m -/
theorem line_parabola_intersection (m : ℝ) : 
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ 
  (y₁^2 = m * y₁ + 1) ∧ 
  (y₂^2 = m * y₂ + 1) := by
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l4034_403479


namespace NUMINAMATH_CALUDE_pump_problem_l4034_403443

theorem pump_problem (x y : ℝ) 
  (h1 : x / 4 + y / 12 = 11)  -- Four pumps fill first tanker and 1/3 of second in 11 hours
  (h2 : x / 3 + y / 4 = 18)   -- Three pumps fill first tanker, one fills 1/4 of second in 18 hours
  : y / 3 = 8 :=              -- Three pumps fill second tanker in 8 hours
by sorry

end NUMINAMATH_CALUDE_pump_problem_l4034_403443


namespace NUMINAMATH_CALUDE_semicircle_pattern_area_l4034_403400

/-- The area of the shaded region formed by semicircles in a pattern -/
theorem semicircle_pattern_area (pattern_length : ℝ) (semicircle_diameter : ℝ) :
  pattern_length = 18 →
  semicircle_diameter = 3 →
  let num_semicircles : ℝ := pattern_length / semicircle_diameter
  let num_full_circles : ℝ := num_semicircles / 2
  let circle_radius : ℝ := semicircle_diameter / 2
  pattern_length > 0 →
  semicircle_diameter > 0 →
  (num_full_circles * π * circle_radius^2) = (27 / 4) * π :=
by sorry

end NUMINAMATH_CALUDE_semicircle_pattern_area_l4034_403400


namespace NUMINAMATH_CALUDE_warden_citations_l4034_403467

theorem warden_citations (total : ℕ) (littering off_leash parking : ℕ) : 
  total = 24 ∧ 
  littering = off_leash ∧ 
  parking = 2 * (littering + off_leash) ∧ 
  total = littering + off_leash + parking →
  littering = 4 := by
sorry

end NUMINAMATH_CALUDE_warden_citations_l4034_403467


namespace NUMINAMATH_CALUDE_find_x_l4034_403463

theorem find_x (p q r x : ℝ) 
  (h1 : (p + q + r) / 3 = 4) 
  (h2 : (p + q + r + x) / 4 = 5) : 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_find_x_l4034_403463
