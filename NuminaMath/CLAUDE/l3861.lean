import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3861_386133

theorem complex_modulus_problem (z : ℂ) (h : (1 - Complex.I) * z = 1) : 
  Complex.abs (2 * z - 3) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3861_386133


namespace NUMINAMATH_CALUDE_security_system_connections_l3861_386157

/-- 
Given a security system with 25 switches where each switch is connected to exactly 4 other switches,
the total number of connections is 50.
-/
theorem security_system_connections (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n * k) / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_security_system_connections_l3861_386157


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3861_386119

/-- Given two geometric sequences with different common ratios s and t, 
    both starting with term m, if m s^2 - m t^2 = 3(m s - m t), then s + t = 3 -/
theorem sum_of_common_ratios_is_three 
  (m : ℝ) (s t : ℝ) (h_diff : s ≠ t) (h_m_nonzero : m ≠ 0) 
  (h_eq : m * s^2 - m * t^2 = 3 * (m * s - m * t)) : 
  s + t = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_three_l3861_386119


namespace NUMINAMATH_CALUDE_sibling_count_l3861_386122

theorem sibling_count (boys girls : ℕ) : 
  boys = 1 ∧ 
  boys - 1 = 0 ∧ 
  girls - 1 = boys → 
  boys + girls = 3 := by
sorry

end NUMINAMATH_CALUDE_sibling_count_l3861_386122


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3861_386167

/-- The profit percentage of a cricket bat sale -/
def profit_percentage (selling_price profit : ℚ) : ℚ :=
  (profit / (selling_price - profit)) * 100

/-- Theorem: The profit percentage is 36% when a cricket bat is sold for $850 with a profit of $225 -/
theorem cricket_bat_profit_percentage :
  profit_percentage 850 225 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_l3861_386167


namespace NUMINAMATH_CALUDE_sequence_explicit_formula_l3861_386187

theorem sequence_explicit_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 2 * a n + 1) →
  ∀ n : ℕ, n ≥ 1 → a n = (-2) ^ (n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_explicit_formula_l3861_386187


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_in_solution_set_l3861_386105

/-- A system of two linear equations in two variables with parameters a and b -/
structure LinearSystem (a b : ℝ) where
  eq1 : ∀ x y : ℝ, 3 * (a + b) * x + 12 * y = a
  eq2 : ∀ x y : ℝ, 4 * b * x + (a + b) * b * y = 1

/-- The condition for the system to have infinitely many solutions -/
def HasInfinitelySolutions (a b : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ 3 * (a + b) = 4 * b * k ∧ 12 = (a + b) * b * k ∧ a = k

/-- The set of pairs (a, b) that satisfy the condition -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(1, 3), (3, 1), (-2 - Real.sqrt 7, Real.sqrt 7 - 2), (Real.sqrt 7 - 2, -2 - Real.sqrt 7)}

/-- The main theorem stating the equivalence -/
theorem infinite_solutions_iff_in_solution_set (a b : ℝ) :
  HasInfinitelySolutions a b ↔ (a, b) ∈ SolutionSet := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_in_solution_set_l3861_386105


namespace NUMINAMATH_CALUDE_atom_particle_count_l3861_386155

/-- Represents an atom with a given number of protons and mass number -/
structure Atom where
  protons : ℕ
  massNumber : ℕ

/-- Calculates the total number of fundamental particles in an atom -/
def totalParticles (a : Atom) : ℕ :=
  a.protons + (a.massNumber - a.protons) + a.protons

/-- Theorem: The total number of fundamental particles in an atom with 9 protons and mass number 19 is 28 -/
theorem atom_particle_count :
  let a : Atom := { protons := 9, massNumber := 19 }
  totalParticles a = 28 := by
  sorry


end NUMINAMATH_CALUDE_atom_particle_count_l3861_386155


namespace NUMINAMATH_CALUDE_three_cones_theorem_l3861_386101

/-- Represents a cone with apex A -/
structure Cone where
  apexAngle : ℝ

/-- Represents a plane -/
structure Plane where

/-- Checks if a cone touches a plane -/
def touchesPlane (c : Cone) (p : Plane) : Prop :=
  sorry

/-- Checks if two cones are identical -/
def areIdentical (c1 c2 : Cone) : Prop :=
  c1.apexAngle = c2.apexAngle

/-- Checks if cones lie on the same side of a plane -/
def onSameSide (c1 c2 c3 : Cone) (p : Plane) : Prop :=
  sorry

theorem three_cones_theorem (c1 c2 c3 : Cone) (p : Plane) :
  areIdentical c1 c2 →
  c3.apexAngle = π / 2 →
  touchesPlane c1 p →
  touchesPlane c2 p →
  touchesPlane c3 p →
  onSameSide c1 c2 c3 p →
  c1.apexAngle = 2 * Real.arctan (4 / 5) :=
sorry

end NUMINAMATH_CALUDE_three_cones_theorem_l3861_386101


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l3861_386112

theorem crazy_silly_school_books (books_read books_unread : ℕ) 
  (h1 : books_read = 13) 
  (h2 : books_unread = 8) : 
  books_read + books_unread = 21 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l3861_386112


namespace NUMINAMATH_CALUDE_owl_cost_in_gold_harry_owl_cost_l3861_386135

/-- Calculates the cost of an owl given the total cost and the cost of other items. -/
theorem owl_cost_in_gold (spellbook_cost : ℕ) (spellbook_count : ℕ) 
  (potion_kit_cost : ℕ) (potion_kit_count : ℕ) (silver_per_gold : ℕ) (total_cost_silver : ℕ) : ℕ :=
  let spellbook_total_cost := spellbook_cost * spellbook_count * silver_per_gold
  let potion_kit_total_cost := potion_kit_cost * potion_kit_count
  let other_items_cost := spellbook_total_cost + potion_kit_total_cost
  let owl_cost_silver := total_cost_silver - other_items_cost
  owl_cost_silver / silver_per_gold

/-- Proves that the owl costs 28 gold given the specific conditions in Harry's purchase. -/
theorem harry_owl_cost : 
  owl_cost_in_gold 5 5 20 3 9 537 = 28 := by
  sorry

end NUMINAMATH_CALUDE_owl_cost_in_gold_harry_owl_cost_l3861_386135


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3861_386110

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, if they are parallel, then x = -4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, x)
  are_parallel a b → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3861_386110


namespace NUMINAMATH_CALUDE_similar_triangle_sum_l3861_386177

/-- Given a triangle with sides in ratio 3:5:7 and a similar triangle with longest side 21,
    the sum of the other two sides of the similar triangle is 24. -/
theorem similar_triangle_sum (a b c : ℝ) (x y z : ℝ) :
  a / b = 3 / 5 →
  b / c = 5 / 7 →
  a / c = 3 / 7 →
  x / y = a / b →
  y / z = b / c →
  x / z = a / c →
  z = 21 →
  x + y = 24 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_sum_l3861_386177


namespace NUMINAMATH_CALUDE_valid_parameterizations_l3861_386199

/-- The slope of the line -/
def m : ℚ := 5 / 3

/-- The y-intercept of the line -/
def b : ℚ := 1

/-- The line equation: y = mx + b -/
def line_equation (x y : ℚ) : Prop := y = m * x + b

/-- A parameterization of a line -/
structure Parameterization where
  initial_point : ℚ × ℚ
  direction_vector : ℚ × ℚ

/-- Check if a parameterization is valid for the given line -/
def is_valid_parameterization (p : Parameterization) : Prop :=
  let (x₀, y₀) := p.initial_point
  let (dx, dy) := p.direction_vector
  line_equation x₀ y₀ ∧ dy / dx = m

/-- The five given parameterizations -/
def param_A : Parameterization := ⟨(3, 6), (3, 5)⟩
def param_B : Parameterization := ⟨(0, 1), (5, 3)⟩
def param_C : Parameterization := ⟨(1, 8/3), (5, 3)⟩
def param_D : Parameterization := ⟨(-1, -2/3), (3, 5)⟩
def param_E : Parameterization := ⟨(1, 1), (5, 8)⟩

theorem valid_parameterizations :
  is_valid_parameterization param_A ∧
  ¬is_valid_parameterization param_B ∧
  ¬is_valid_parameterization param_C ∧
  is_valid_parameterization param_D ∧
  ¬is_valid_parameterization param_E :=
sorry

end NUMINAMATH_CALUDE_valid_parameterizations_l3861_386199


namespace NUMINAMATH_CALUDE_simplify_expression_l3861_386132

theorem simplify_expression : 110^2 - 109 * 111 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3861_386132


namespace NUMINAMATH_CALUDE_some_ounce_glass_size_l3861_386158

/-- Proves that the size of the some-ounce glasses is 5 ounces given the problem conditions. -/
theorem some_ounce_glass_size (total_water : ℕ) (S : ℕ) 
  (h1 : total_water = 122)
  (h2 : 6 * S + 4 * 8 + 15 * 4 = total_water) : S = 5 := by
  sorry

#check some_ounce_glass_size

end NUMINAMATH_CALUDE_some_ounce_glass_size_l3861_386158


namespace NUMINAMATH_CALUDE_bottle_caps_difference_l3861_386178

/-- Represents the number of bottle caps in various states of Danny's collection --/
structure BottleCaps where
  thrown_away : ℕ
  found : ℕ
  final_count : ℕ

/-- Theorem stating the difference between found and thrown away bottle caps --/
theorem bottle_caps_difference (caps : BottleCaps)
  (h1 : caps.thrown_away = 6)
  (h2 : caps.found = 50)
  (h3 : caps.final_count = 60)
  : caps.found - caps.thrown_away = 44 := by
  sorry

#check bottle_caps_difference

end NUMINAMATH_CALUDE_bottle_caps_difference_l3861_386178


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l3861_386142

/-- The equation is quadratic with respect to x if and only if m^2 - 2 = 2 -/
def is_quadratic (m : ℝ) : Prop := m^2 - 2 = 2

/-- The equation is not degenerate if and only if m - 2 ≠ 0 -/
def is_not_degenerate (m : ℝ) : Prop := m - 2 ≠ 0

theorem quadratic_equation_m_value :
  ∀ m : ℝ, is_quadratic m ∧ is_not_degenerate m → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l3861_386142


namespace NUMINAMATH_CALUDE_cricketer_specific_average_l3861_386162

/-- Represents the average score calculation for a cricketer's matches -/
def cricketer_average_score (total_matches : ℕ) (first_set_matches : ℕ) (second_set_matches : ℕ) 
  (first_set_average : ℚ) (second_set_average : ℚ) : ℚ :=
  ((first_set_matches : ℚ) * first_set_average + (second_set_matches : ℚ) * second_set_average) / (total_matches : ℚ)

/-- Theorem stating the average score calculation for a specific cricketer's performance -/
theorem cricketer_specific_average : 
  cricketer_average_score 25 10 15 60 70 = 66 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_specific_average_l3861_386162


namespace NUMINAMATH_CALUDE_two_digit_number_exchange_l3861_386159

theorem two_digit_number_exchange (A : ℕ) : 
  A < 10 →  -- Ensure A is a single digit
  (10 * A + 2) - (20 + A) = 9 →  -- Condition for digit exchange
  A = 3 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_exchange_l3861_386159


namespace NUMINAMATH_CALUDE_alpha_beta_not_perfect_square_l3861_386120

/-- A polynomial of degree 4 with roots 0, αβ, βγ, and γα -/
def f (α β γ : ℕ) (x : ℤ) : ℤ := x * (x - α * β) * (x - β * γ) * (x - γ * α)

/-- Theorem: Given positive integers α, β, γ, and an integer s such that
    f(-1) = f(s)², αβ is not a perfect square. -/
theorem alpha_beta_not_perfect_square (α β γ : ℕ) (s : ℤ) 
    (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
    (h_eq : f α β γ (-1) = (f α β γ s)^2) :
    ¬ ∃ (k : ℕ), α * β = k^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_not_perfect_square_l3861_386120


namespace NUMINAMATH_CALUDE_round_trip_percentage_l3861_386136

/-- Proves that 80% of passengers held round-trip tickets given the conditions -/
theorem round_trip_percentage (total_passengers : ℕ) 
  (h1 : (40 : ℝ) / 100 * total_passengers = (passengers_with_car : ℝ))
  (h2 : (50 : ℝ) / 100 * (passengers_with_roundtrip : ℝ) = passengers_with_roundtrip - passengers_with_car) :
  (80 : ℝ) / 100 * total_passengers = (passengers_with_roundtrip : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_round_trip_percentage_l3861_386136


namespace NUMINAMATH_CALUDE_triangle_area_l3861_386148

theorem triangle_area (a c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : c = 2 * Real.sqrt 3) (h3 : A = π / 6) :
  ∃ (area : ℝ), (area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3) ∧
  ∃ (B C : ℝ), 0 ≤ B ∧ B < 2 * π ∧ 0 ≤ C ∧ C < 2 * π ∧
  A + B + C = π ∧
  area = (1 / 2) * a * c * Real.sin B :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3861_386148


namespace NUMINAMATH_CALUDE_negation_of_every_prime_is_odd_l3861_386185

theorem negation_of_every_prime_is_odd :
  (¬ ∀ p : ℕ, Prime p → Odd p) ↔ (∃ p : ℕ, Prime p ∧ ¬ Odd p) :=
sorry

end NUMINAMATH_CALUDE_negation_of_every_prime_is_odd_l3861_386185


namespace NUMINAMATH_CALUDE_train_crossing_time_l3861_386173

/-- Given two trains moving in opposite directions, this theorem proves
    the time taken for them to cross each other. -/
theorem train_crossing_time
  (train_length : ℝ)
  (faster_speed : ℝ)
  (h1 : train_length = 100)
  (h2 : faster_speed = 48)
  (h3 : faster_speed > 0) :
  let slower_speed := faster_speed / 2
  let relative_speed := faster_speed + slower_speed
  let total_distance := 2 * train_length
  let time := total_distance / (relative_speed * (1000 / 3600))
  time = 10 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3861_386173


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l3861_386189

theorem inequality_implies_upper_bound (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ a) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l3861_386189


namespace NUMINAMATH_CALUDE_fibonacci_product_theorem_l3861_386144

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sum of squares of divisors -/
def sum_of_squares_of_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x * x)

/-- Main theorem -/
theorem fibonacci_product_theorem (N : ℕ) (h_pos : N > 0)
  (h_sum : sum_of_squares_of_divisors N = N * (N + 3)) :
  ∃ i j, N = fib i * fib j :=
sorry

end NUMINAMATH_CALUDE_fibonacci_product_theorem_l3861_386144


namespace NUMINAMATH_CALUDE_roots_and_d_values_l3861_386100

-- Define the polynomial p(x)
def p (c d x : ℝ) : ℝ := x^3 + c*x + d

-- Define the polynomial q(x)
def q (c d x : ℝ) : ℝ := x^3 + c*x + d + 144

-- State the theorem
theorem roots_and_d_values (u v c d : ℝ) :
  (p c d u = 0) ∧ (p c d v = 0) ∧ 
  (q c d (u + 3) = 0) ∧ (q c d (v - 2) = 0) →
  (d = 84 ∨ d = -15) := by
  sorry


end NUMINAMATH_CALUDE_roots_and_d_values_l3861_386100


namespace NUMINAMATH_CALUDE_rectangle_length_l3861_386128

-- Define the radius of the circle
def R : ℝ := 2.5

-- Define pi as an approximation
def π : ℝ := 3.14

-- Define the perimeter of the rectangle
def perimeter : ℝ := 20.7

-- Theorem stating that the length of the rectangle is 7.85 cm
theorem rectangle_length : (π * R) = 7.85 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3861_386128


namespace NUMINAMATH_CALUDE_motorcycle_speeds_correct_l3861_386196

/-- Two motorcyclists travel towards each other with uniform speed. -/
structure MotorcycleJourney where
  /-- Total distance between starting points A and B in km -/
  total_distance : ℝ
  /-- Distance traveled by the first motorcyclist when the second has traveled 200 km -/
  first_partial_distance : ℝ
  /-- Time difference in hours between arrivals -/
  time_difference : ℝ
  /-- Speed of the first motorcyclist in km/h -/
  speed_first : ℝ
  /-- Speed of the second motorcyclist in km/h -/
  speed_second : ℝ

/-- The speeds of the motorcyclists satisfy the given conditions -/
def satisfies_conditions (j : MotorcycleJourney) : Prop :=
  j.total_distance = 600 ∧
  j.first_partial_distance = 250 ∧
  j.time_difference = 3 ∧
  j.first_partial_distance / j.speed_first = 200 / j.speed_second ∧
  j.total_distance / j.speed_first + j.time_difference = j.total_distance / j.speed_second

/-- The theorem stating that the given speeds satisfy the conditions -/
theorem motorcycle_speeds_correct (j : MotorcycleJourney) :
  j.speed_first = 50 ∧ j.speed_second = 40 → satisfies_conditions j :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_speeds_correct_l3861_386196


namespace NUMINAMATH_CALUDE_total_stickers_is_60_l3861_386171

/-- Represents the number of folders --/
def num_folders : Nat := 3

/-- Represents the number of sheets in each folder --/
def sheets_per_folder : Nat := 10

/-- Represents the number of stickers per sheet in the red folder --/
def red_stickers : Nat := 3

/-- Represents the number of stickers per sheet in the green folder --/
def green_stickers : Nat := 2

/-- Represents the number of stickers per sheet in the blue folder --/
def blue_stickers : Nat := 1

/-- Calculates the total number of stickers used --/
def total_stickers : Nat :=
  sheets_per_folder * red_stickers +
  sheets_per_folder * green_stickers +
  sheets_per_folder * blue_stickers

/-- Theorem stating that the total number of stickers used is 60 --/
theorem total_stickers_is_60 : total_stickers = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_is_60_l3861_386171


namespace NUMINAMATH_CALUDE_polynomial_coefficient_properties_l3861_386149

theorem polynomial_coefficient_properties (a : Fin 6 → ℝ) :
  (∀ x : ℝ, x^5 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5) →
  (a 3 = -10 ∧ a 1 + a 3 + a 5 = -16) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_properties_l3861_386149


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3861_386192

theorem smallest_prime_dividing_sum : Nat.minFac (7^7 + 3^14) = 2 := by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l3861_386192


namespace NUMINAMATH_CALUDE_two_balls_picked_l3861_386116

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the total number of balls in the bag -/
def totalBalls (bag : BagContents) : Nat :=
  bag.red + bag.blue + bag.green

/-- Calculates the probability of picking two red balls -/
def probTwoRed (bag : BagContents) (picked : Nat) : Rat :=
  if picked ≠ 2 then 0
  else
    let total := totalBalls bag
    (bag.red : Rat) / total * ((bag.red - 1) : Rat) / (total - 1)

theorem two_balls_picked (bag : BagContents) (picked : Nat) :
  bag.red = 4 → bag.blue = 3 → bag.green = 2 →
  probTwoRed bag picked = 1/6 →
  picked = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_balls_picked_l3861_386116


namespace NUMINAMATH_CALUDE_jim_has_220_buicks_l3861_386194

/-- Represents the number of model cars of each brand Jim has. -/
structure ModelCars where
  ford : ℕ
  buick : ℕ
  chevy : ℕ

/-- The conditions of Jim's model car collection. -/
def jim_collection (cars : ModelCars) : Prop :=
  cars.ford + cars.buick + cars.chevy = 301 ∧
  cars.buick = 4 * cars.ford ∧
  cars.ford = 2 * cars.chevy + 3

/-- Theorem stating that Jim has 220 Buicks. -/
theorem jim_has_220_buicks :
  ∃ (cars : ModelCars), jim_collection cars ∧ cars.buick = 220 := by
  sorry

end NUMINAMATH_CALUDE_jim_has_220_buicks_l3861_386194


namespace NUMINAMATH_CALUDE_probability_of_selection_l3861_386113

def multiple_choice_count : ℕ := 12
def fill_in_blank_count : ℕ := 4
def open_ended_count : ℕ := 6
def total_questions : ℕ := multiple_choice_count + fill_in_blank_count + open_ended_count
def selection_count : ℕ := 3

theorem probability_of_selection (multiple_choice_count fill_in_blank_count open_ended_count total_questions selection_count : ℕ) 
  (h1 : multiple_choice_count = 12)
  (h2 : fill_in_blank_count = 4)
  (h3 : open_ended_count = 6)
  (h4 : total_questions = multiple_choice_count + fill_in_blank_count + open_ended_count)
  (h5 : selection_count = 3) :
  (Nat.choose multiple_choice_count 1 * Nat.choose open_ended_count 2 +
   Nat.choose multiple_choice_count 2 * Nat.choose open_ended_count 1 +
   Nat.choose multiple_choice_count 1 * Nat.choose open_ended_count 1 * Nat.choose fill_in_blank_count 1) /
  (Nat.choose total_questions selection_count - Nat.choose (fill_in_blank_count + open_ended_count) selection_count) = 43 / 71 := by
  sorry

#check probability_of_selection

end NUMINAMATH_CALUDE_probability_of_selection_l3861_386113


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3861_386130

/-- Represents a number in a given base with repeated digits -/
def repeatedDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid for a given base -/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (A C : Nat),
    isValidDigit A 8 ∧
    isValidDigit C 6 ∧
    repeatedDigitNumber A 8 = repeatedDigitNumber C 6 ∧
    repeatedDigitNumber A 8 = 19 ∧
    (∀ (A' C' : Nat),
      isValidDigit A' 8 →
      isValidDigit C' 6 →
      repeatedDigitNumber A' 8 = repeatedDigitNumber C' 6 →
      repeatedDigitNumber A' 8 ≥ 19) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3861_386130


namespace NUMINAMATH_CALUDE_apple_difference_l3861_386179

theorem apple_difference (adam_apples jackie_apples : ℕ) 
  (adam_has : adam_apples = 9) 
  (jackie_has : jackie_apples = 10) : 
  jackie_apples - adam_apples = 1 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l3861_386179


namespace NUMINAMATH_CALUDE_tan_sum_equals_one_l3861_386147

theorem tan_sum_equals_one (α β : ℝ) 
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equals_one_l3861_386147


namespace NUMINAMATH_CALUDE_andrew_payment_l3861_386182

/-- The total amount Andrew paid to the shopkeeper -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: Andrew paid 1055 to the shopkeeper -/
theorem andrew_payment : total_amount 8 70 9 55 = 1055 := by
  sorry

end NUMINAMATH_CALUDE_andrew_payment_l3861_386182


namespace NUMINAMATH_CALUDE_man_walking_distance_l3861_386166

theorem man_walking_distance (speed : ℝ) (time : ℝ) : 
  speed > 0 →
  time > 0 →
  (speed + 1/3) * (5/6 * time) = speed * time →
  (speed - 1/3) * (time + 3.5) = speed * time →
  speed * time = 35/96 := by
  sorry

end NUMINAMATH_CALUDE_man_walking_distance_l3861_386166


namespace NUMINAMATH_CALUDE_absolute_difference_of_factors_l3861_386108

theorem absolute_difference_of_factors (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_factors_l3861_386108


namespace NUMINAMATH_CALUDE_susan_money_l3861_386129

theorem susan_money (S : ℝ) : 
  S - S/5 - S/4 - 120 = 540 → S = 1200 := by
  sorry

end NUMINAMATH_CALUDE_susan_money_l3861_386129


namespace NUMINAMATH_CALUDE_ellipse_proof_l3861_386126

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := 9 * x^2 + 4 * y^2 = 36

-- Define the equation of the ellipse we want to prove
def target_ellipse (x y : ℝ) : Prop := y^2/16 + x^2/11 = 1

-- Theorem statement
theorem ellipse_proof :
  -- The target ellipse passes through (0, 4)
  target_ellipse 0 4 ∧
  -- The target ellipse has the same foci as the given ellipse
  ∃ (c : ℝ), c^2 = 5 ∧
    ∀ (x y : ℝ), given_ellipse x y ↔ 
      ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = 4 ∧ c^2 = a^2 - b^2 ∧ y^2/a^2 + x^2/b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_proof_l3861_386126


namespace NUMINAMATH_CALUDE_complex_product_real_l3861_386198

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of a complex number being real -/
def is_real (z : ℂ) : Prop := z.im = 0

theorem complex_product_real (m : ℝ) :
  is_real ((2 + i) * (m - 2*i)) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l3861_386198


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l3861_386114

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length given specific conditions -/
theorem bridge_length_proof :
  bridge_length 150 45 30 = 225 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_proof_l3861_386114


namespace NUMINAMATH_CALUDE_wider_can_radius_l3861_386104

/-- Given two cylindrical cans with the same volume, where the height of one can is double 
    the height of the other, and the radius of the narrower can is 8 units, 
    the radius of the wider can is 8√2 units. -/
theorem wider_can_radius (h : ℝ) (r : ℝ) (h_pos : h > 0) : 
  π * 8^2 * (2*h) = π * r^2 * h → r = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_wider_can_radius_l3861_386104


namespace NUMINAMATH_CALUDE_decimal_is_fraction_l3861_386195

def is_fraction (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

theorem decimal_is_fraction :
  let x : ℝ := 0.666
  is_fraction x :=
sorry

end NUMINAMATH_CALUDE_decimal_is_fraction_l3861_386195


namespace NUMINAMATH_CALUDE_system_solution_l3861_386146

theorem system_solution (x y z : ℝ) (eq1 : x + y + z = 0) (eq2 : 4 * x + 2 * y + z = 0) :
  y = -3 * x ∧ z = 2 * x := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3861_386146


namespace NUMINAMATH_CALUDE_total_seeds_calculation_l3861_386156

/-- The number of rows of potatoes planted -/
def rows : ℕ := 6

/-- The number of seeds in each row -/
def seeds_per_row : ℕ := 9

/-- The total number of potato seeds planted -/
def total_seeds : ℕ := rows * seeds_per_row

theorem total_seeds_calculation : total_seeds = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_calculation_l3861_386156


namespace NUMINAMATH_CALUDE_work_days_solution_l3861_386175

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person b -/
def days_b : ℕ := 9

/-- The number of days worked by person c -/
def days_c : ℕ := 4

/-- The daily wage of person c -/
def wage_c : ℕ := 100

/-- The total earnings of all three workers -/
def total_earnings : ℕ := 1480

/-- The ratio of daily wages for a, b, and c respectively -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

theorem work_days_solution :
  ∃ (wage_a wage_b : ℕ),
    wage_a = wage_ratio 0 * (wage_c / wage_ratio 2) ∧
    wage_b = wage_ratio 1 * (wage_c / wage_ratio 2) ∧
    wage_a * days_a + wage_b * days_b + wage_c * days_c = total_earnings :=
by sorry


end NUMINAMATH_CALUDE_work_days_solution_l3861_386175


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3861_386183

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3861_386183


namespace NUMINAMATH_CALUDE_committee_selection_count_l3861_386134

/-- The number of ways to choose a committee from a club -/
def choose_committee (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

/-- The size of the club -/
def club_size : ℕ := 10

/-- The size of the committee -/
def committee_size : ℕ := 5

/-- Theorem: The number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_selection_count : 
  choose_committee club_size committee_size = 252 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l3861_386134


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3861_386139

theorem arithmetic_simplification :
  -11 - (-8) + (-13) + 12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3861_386139


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3861_386154

/-- Proves that in a triangle where two angles sum to 7/6 of a right angle,
    and one of these angles is 36° larger than the other,
    the largest angle in the triangle is 75°. -/
theorem largest_angle_in_special_triangle : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  a + b = 105 →
  b = a + 36 →
  max a (max b c) = 75 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3861_386154


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l3861_386174

theorem root_sum_reciprocal (p q r : ℝ) : 
  (p^3 - p - 6 = 0) → 
  (q^3 - q - 6 = 0) → 
  (r^3 - r - 6 = 0) → 
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l3861_386174


namespace NUMINAMATH_CALUDE_inequality_proof_l3861_386106

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 
  Real.sqrt 3 + (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3861_386106


namespace NUMINAMATH_CALUDE_paper_I_max_mark_l3861_386165

/-- Represents a test with a maximum mark and a passing percentage -/
structure Test where
  maxMark : ℕ
  passingPercentage : ℚ

/-- Calculates the passing mark for a given test -/
def passingMark (test : Test) : ℚ :=
  test.passingPercentage * test.maxMark

theorem paper_I_max_mark :
  ∃ (test : Test),
    test.passingPercentage = 42 / 100 ∧
    passingMark test = 42 + 22 ∧
    test.maxMark = 152 := by
  sorry

end NUMINAMATH_CALUDE_paper_I_max_mark_l3861_386165


namespace NUMINAMATH_CALUDE_average_sale_is_3500_l3861_386127

def sales : List ℕ := [3435, 3920, 3855, 4230, 3560, 2000]

theorem average_sale_is_3500 : 
  (sales.sum / sales.length : ℚ) = 3500 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_is_3500_l3861_386127


namespace NUMINAMATH_CALUDE_age_ratio_future_l3861_386118

/-- Given Alan's current age a and Bella's current age b, prove that the number of years
    until their age ratio is 3:2 is 7, given the conditions on their past ages. -/
theorem age_ratio_future (a b : ℕ) (h1 : a - 3 = 2 * (b - 3)) (h2 : a - 8 = 3 * (b - 8)) :
  ∃ x : ℕ, x = 7 ∧ (a + x) * 2 = (b + x) * 3 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_future_l3861_386118


namespace NUMINAMATH_CALUDE_count_divisible_by_eight_l3861_386172

theorem count_divisible_by_eight (n : ℕ) : 
  (150 < n ∧ n ≤ 400 ∧ n % 8 = 0) → 
  (Finset.filter (λ x => 150 < x ∧ x ≤ 400 ∧ x % 8 = 0) (Finset.range 401)).card = 31 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_eight_l3861_386172


namespace NUMINAMATH_CALUDE_students_in_both_clubs_l3861_386153

def total_students : ℕ := 250
def drama_club : ℕ := 80
def science_club : ℕ := 120
def either_or_both : ℕ := 180

theorem students_in_both_clubs :
  ∃ (both : ℕ), both = drama_club + science_club - either_or_both ∧ both = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_l3861_386153


namespace NUMINAMATH_CALUDE_diagonals_are_space_l3861_386143

/-- A cube with diagonals forming a 60-degree angle --/
structure CubeWithDiagonals where
  /-- The measure of the angle between two diagonals --/
  angle : ℝ
  /-- The angle between the diagonals is 60 degrees --/
  angle_is_60 : angle = 60

/-- The types of diagonals in a cube --/
inductive DiagonalType
  | Face
  | Space

/-- Theorem: If the angle between two diagonals of a cube is 60 degrees,
    then these diagonals are space diagonals --/
theorem diagonals_are_space (c : CubeWithDiagonals) :
  ∃ (d : DiagonalType), d = DiagonalType.Space :=
sorry

end NUMINAMATH_CALUDE_diagonals_are_space_l3861_386143


namespace NUMINAMATH_CALUDE_straw_purchase_solution_l3861_386123

/-- Represents the cost and quantity of straws --/
structure StrawPurchase where
  costA : ℚ  -- Cost per pack of type A straws
  costB : ℚ  -- Cost per pack of type B straws
  maxA : ℕ   -- Maximum number of type A straws that can be purchased

/-- Verifies if the given costs satisfy the purchase scenarios --/
def satisfiesPurchaseScenarios (sp : StrawPurchase) : Prop :=
  12 * sp.costA + 15 * sp.costB = 171 ∧
  24 * sp.costA + 28 * sp.costB = 332

/-- Checks if the maximum number of type A straws satisfies the constraints --/
def satisfiesConstraints (sp : StrawPurchase) : Prop :=
  sp.maxA ≤ 100 ∧
  sp.costA * sp.maxA + sp.costB * (100 - sp.maxA) ≤ 600 ∧
  ∀ m : ℕ, m > sp.maxA → sp.costA * m + sp.costB * (100 - m) > 600

/-- Theorem stating the solution to the straw purchase problem --/
theorem straw_purchase_solution :
  ∃ sp : StrawPurchase,
    sp.costA = 8 ∧ sp.costB = 5 ∧ sp.maxA = 33 ∧
    satisfiesPurchaseScenarios sp ∧
    satisfiesConstraints sp := by
  sorry

end NUMINAMATH_CALUDE_straw_purchase_solution_l3861_386123


namespace NUMINAMATH_CALUDE_trig_identity_l3861_386160

theorem trig_identity (α : ℝ) : Real.sin α ^ 6 + Real.cos α ^ 6 + 3 * Real.sin α ^ 2 * Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3861_386160


namespace NUMINAMATH_CALUDE_ticket_sales_total_l3861_386103

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (advanced_price : ℕ) (door_price : ℕ) (total_tickets : ℕ) (advanced_tickets : ℕ) : ℕ :=
  advanced_price * advanced_tickets + door_price * (total_tickets - advanced_tickets)

/-- Proves that the total money collected is $1360 given the ticket prices and quantities -/
theorem ticket_sales_total : total_money_collected 8 14 140 100 = 1360 := by
  sorry

#eval total_money_collected 8 14 140 100

end NUMINAMATH_CALUDE_ticket_sales_total_l3861_386103


namespace NUMINAMATH_CALUDE_tomato_problem_l3861_386184

/-- The number of tomatoes produced by the first plant -/
def first_plant : ℕ := 19

/-- The number of tomatoes produced by the second plant -/
def second_plant (x : ℕ) : ℕ := x / 2 + 5

/-- The number of tomatoes produced by the third plant -/
def third_plant (x : ℕ) : ℕ := second_plant x + 2

/-- The total number of tomatoes produced by all three plants -/
def total_tomatoes : ℕ := 60

theorem tomato_problem :
  first_plant + second_plant first_plant + third_plant first_plant = total_tomatoes :=
by sorry

end NUMINAMATH_CALUDE_tomato_problem_l3861_386184


namespace NUMINAMATH_CALUDE_parabola_transformation_sum_of_zeros_l3861_386145

/-- Represents a parabola and its transformations -/
structure Parabola where
  a : ℝ  -- coefficient of x^2
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex

/-- Apply transformations to the parabola -/
def transform (p : Parabola) : Parabola :=
  { a := -p.a,  -- 180-degree rotation
    h := p.h + 4,  -- 4 units right shift
    k := p.k + 4 }  -- 4 units up shift

/-- Calculate the sum of zeros for a parabola -/
def sumOfZeros (p : Parabola) : ℝ := 2 * p.h

theorem parabola_transformation_sum_of_zeros :
  let original := Parabola.mk 1 2 3
  let transformed := transform original
  sumOfZeros transformed = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_transformation_sum_of_zeros_l3861_386145


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3861_386131

-- Define the variables and conditions
theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) :
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l3861_386131


namespace NUMINAMATH_CALUDE_orthogonal_medians_theorem_l3861_386137

/-- Given a triangle with side lengths a, b, c and medians ma, mb, mc,
    if ma is perpendicular to mb, then the medians form a right-angled triangle
    and the inequality 5(a^2 + b^2 - c^2) ≥ 8ab holds. -/
theorem orthogonal_medians_theorem (a b c ma mb mc : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_mb : mb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_mc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_perp : ma * mb = 0) : 
  ma^2 + mb^2 = mc^2 ∧ 5*(a^2 + b^2 - c^2) ≥ 8*a*b :=
sorry

end NUMINAMATH_CALUDE_orthogonal_medians_theorem_l3861_386137


namespace NUMINAMATH_CALUDE_trees_in_garden_l3861_386125

/-- Given a yard of length 600 meters with trees planted at equal distances,
    including one at each end, and a distance of 24 meters between consecutive trees,
    the total number of trees planted is 26. -/
theorem trees_in_garden (yard_length : ℕ) (tree_distance : ℕ) (h1 : yard_length = 600) (h2 : tree_distance = 24) :
  yard_length / tree_distance + 1 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_garden_l3861_386125


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3861_386181

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)

/-- The theorem stating that if certain conditions are met, the triangle is right-angled. -/
theorem right_angled_triangle (t : Triangle) 
  (h1 : (Real.sqrt 3 * t.c) / (t.a * Real.cos t.B) = Real.tan t.A + Real.tan t.B)
  (h2 : t.b - t.c = (Real.sqrt 3 * t.a) / 3) : 
  t.B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3861_386181


namespace NUMINAMATH_CALUDE_max_trailing_zeros_product_l3861_386164

/-- Given three natural numbers that sum to 1003, the maximum number of trailing zeros in their product is 7. -/
theorem max_trailing_zeros_product (a b c : ℕ) (h_sum : a + b + c = 1003) :
  (∃ n : ℕ, a * b * c = n * 10^7 ∧ n % 10 ≠ 0) ∧
  ¬(∃ m : ℕ, a * b * c = m * 10^8) :=
by sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_product_l3861_386164


namespace NUMINAMATH_CALUDE_calculation_proof_l3861_386124

theorem calculation_proof :
  (1) * (-3)^2 - (-1)^3 - (-2) - |(-12)| = 0 ∧
  -2^2 * 3 * (-3/2) / (2/3) - 4 * (-3/2)^2 = 18 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l3861_386124


namespace NUMINAMATH_CALUDE_uncertain_mushrooms_l3861_386170

theorem uncertain_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) (uncertain : ℕ) : 
  total = 32 → 
  safe = 9 → 
  poisonous = 2 * safe → 
  total = safe + poisonous + uncertain → 
  uncertain = 5 := by
sorry

end NUMINAMATH_CALUDE_uncertain_mushrooms_l3861_386170


namespace NUMINAMATH_CALUDE_parallelogram_height_l3861_386111

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 416)
  (h_base : base = 26)
  (h_formula : area = base * height) : 
  height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3861_386111


namespace NUMINAMATH_CALUDE_range_of_a_l3861_386115

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≤ 1}
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 3) ≤ 0}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, x ∈ M → x ∈ N a) ∧ 
  (∃ x : ℝ, x ∈ N a ∧ x ∉ M) → 
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3861_386115


namespace NUMINAMATH_CALUDE_units_digit_of_product_division_l3861_386193

theorem units_digit_of_product_division : 
  (15 * 16 * 17 * 18 * 19 * 20) / 500 % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_division_l3861_386193


namespace NUMINAMATH_CALUDE_cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six_square_minus_product_equals_five_plus_two_sqrt_three_l3861_386163

-- Part 1
theorem cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six :
  (8 : ℝ) ^ (1/3) - (Real.sqrt 12 * Real.sqrt 6) / Real.sqrt 3 = 2 - 2 * Real.sqrt 6 := by sorry

-- Part 2
theorem square_minus_product_equals_five_plus_two_sqrt_three :
  (Real.sqrt 3 + 1)^2 - (2 * Real.sqrt 2 + 3) * (2 * Real.sqrt 2 - 3) = 5 + 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_eight_minus_fraction_equals_two_minus_two_sqrt_six_square_minus_product_equals_five_plus_two_sqrt_three_l3861_386163


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3861_386152

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) :
  a = (3, 2) →
  b.1 = x →
  b.2 = 4 →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = -8/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3861_386152


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3861_386107

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  (∃ (p q : ℝ × ℝ), p = (1, -Real.sqrt 3) ∧ q = (Real.cos B, Real.sin B) ∧ p.1 * q.2 = p.2 * q.1) →
  b * Real.cos C + c * Real.cos B = 2 * a * Real.sin A →
  A + B + C = Real.pi →
  C = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3861_386107


namespace NUMINAMATH_CALUDE_first_digit_base_9_of_628_l3861_386117

/-- The first digit of the base 9 representation of a number -/
def first_digit_base_9 (n : ℕ) : ℕ :=
  if n < 9 then n else first_digit_base_9 (n / 9)

/-- The number in base 10 -/
def number : ℕ := 628

theorem first_digit_base_9_of_628 :
  first_digit_base_9 number = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base_9_of_628_l3861_386117


namespace NUMINAMATH_CALUDE_mixed_coffee_bag_weight_l3861_386180

/-- Proves that the total weight of a mixed coffee bag is 102.8 pounds given specific conditions --/
theorem mixed_coffee_bag_weight 
  (colombian_price : ℝ) 
  (peruvian_price : ℝ) 
  (mixed_price : ℝ) 
  (colombian_weight : ℝ) 
  (h1 : colombian_price = 5.50)
  (h2 : peruvian_price = 4.25)
  (h3 : mixed_price = 4.60)
  (h4 : colombian_weight = 28.8) :
  ∃ (total_weight : ℝ), total_weight = 102.8 ∧ 
  (colombian_price * colombian_weight + peruvian_price * (total_weight - colombian_weight)) / total_weight = mixed_price :=
by
  sorry

#check mixed_coffee_bag_weight

end NUMINAMATH_CALUDE_mixed_coffee_bag_weight_l3861_386180


namespace NUMINAMATH_CALUDE_combination_equality_l3861_386191

theorem combination_equality (n : ℕ) : 
  Nat.choose 5 2 = Nat.choose 5 n → n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l3861_386191


namespace NUMINAMATH_CALUDE_unique_condition_result_l3861_386169

theorem unique_condition_result : ∃ (a b c : ℕ),
  ({a, b, c} : Set ℕ) = {0, 1, 2} ∧
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨
   ((a ≠ 2) ∧ (b = 2) ∧ (c = 0)) ∨
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0))) →
  100 * a + 10 * b + c = 201 :=
by sorry

end NUMINAMATH_CALUDE_unique_condition_result_l3861_386169


namespace NUMINAMATH_CALUDE_similar_triangles_height_l3861_386190

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_small = 5 →
    h_large = 15 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l3861_386190


namespace NUMINAMATH_CALUDE_abs_even_and_decreasing_l3861_386150

def f (x : ℝ) := abs x

theorem abs_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_abs_even_and_decreasing_l3861_386150


namespace NUMINAMATH_CALUDE_unit_vectors_equality_iff_sum_magnitude_two_l3861_386151

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem unit_vectors_equality_iff_sum_magnitude_two
  (a b : E) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  a = b ↔ ‖a + b‖ = 2 := by sorry

end NUMINAMATH_CALUDE_unit_vectors_equality_iff_sum_magnitude_two_l3861_386151


namespace NUMINAMATH_CALUDE_simplify_expression_l3861_386140

theorem simplify_expression (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1)^2 = 7 * a - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3861_386140


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3861_386176

open Complex

/-- The analytic function f(z) that satisfies the given conditions -/
noncomputable def f (z : ℂ) : ℂ := z^3 - 2*I*z + (2 + 3*I)

/-- The real part of f(z) -/
def u (x y : ℝ) : ℝ := x^3 - 3*x*y^2 + 2*y

theorem f_satisfies_conditions :
  (∀ x y : ℝ, (f (x + y*I)).re = u x y) ∧
  f I = 2 := by sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3861_386176


namespace NUMINAMATH_CALUDE_salary_increase_with_manager_l3861_386138

/-- Proves that adding a manager's salary increases the average salary by 150 rupees. -/
theorem salary_increase_with_manager 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 15 → 
  avg_salary = 1800 → 
  manager_salary = 4200 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 150 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_with_manager_l3861_386138


namespace NUMINAMATH_CALUDE_unique_prime_product_l3861_386141

theorem unique_prime_product (p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q →
  p * q * r = 2014 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_product_l3861_386141


namespace NUMINAMATH_CALUDE_segment_is_definition_l3861_386197

-- Define the type for geometric statements
inductive GeometricStatement
  | TwoPointsLine
  | SegmentDefinition
  | ComplementaryAngles
  | AlternateInteriorAngles

-- Define a predicate to check if a statement is a definition
def isDefinition : GeometricStatement → Prop
  | GeometricStatement.SegmentDefinition => True
  | _ => False

-- Theorem statement
theorem segment_is_definition :
  (∃! s : GeometricStatement, isDefinition s) →
  isDefinition GeometricStatement.SegmentDefinition :=
by
  sorry

end NUMINAMATH_CALUDE_segment_is_definition_l3861_386197


namespace NUMINAMATH_CALUDE_r₂_bound_l3861_386102

/-- The function f(x) = x² - r₂x + r₃ -/
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂ * x + r₃

/-- The sequence g_n defined recursively -/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The property that g₂ᵢ < g₂ᵢ₊₁ and g₂ᵢ₊₁ > g₂ᵢ₊₂ for 0 ≤ i ≤ 2011 -/
def alternating_property (r₂ r₃ : ℝ) : Prop :=
  ∀ i, 0 ≤ i ∧ i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)

/-- The property that there exists j such that gᵢ₊₁ > gᵢ for all i > j -/
def eventually_increasing (r₂ r₃ : ℝ) : Prop :=
  ∃ j : ℕ, ∀ i, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i

/-- The property that the sequence is unbounded -/
def unbounded (r₂ r₃ : ℝ) : Prop :=
  ∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M

theorem r₂_bound (r₂ r₃ : ℝ) 
  (h₁ : alternating_property r₂ r₃)
  (h₂ : eventually_increasing r₂ r₃)
  (h₃ : unbounded r₂ r₃) :
  |r₂| > 2 ∧ ∀ ε > 0, ∃ r₂' r₃', 
    alternating_property r₂' r₃' ∧ 
    eventually_increasing r₂' r₃' ∧ 
    unbounded r₂' r₃' ∧ 
    |r₂'| < 2 + ε :=
sorry

end NUMINAMATH_CALUDE_r₂_bound_l3861_386102


namespace NUMINAMATH_CALUDE_escalator_time_to_cover_l3861_386121

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover its length -/
theorem escalator_time_to_cover (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 15 →
  person_speed = 3 →
  escalator_length = 180 →
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_to_cover_l3861_386121


namespace NUMINAMATH_CALUDE_odd_function_extension_l3861_386188

/-- An odd function on the real line. -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The main theorem -/
theorem odd_function_extension (f : ℝ → ℝ) (h : OddFunction f) 
    (h_neg : ∀ x < 0, f x = x * Real.exp (-x)) :
    ∀ x > 0, f x = x * Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3861_386188


namespace NUMINAMATH_CALUDE_x_twelve_equals_negative_one_l3861_386186

theorem x_twelve_equals_negative_one (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^12 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelve_equals_negative_one_l3861_386186


namespace NUMINAMATH_CALUDE_pancake_milk_calculation_l3861_386109

/-- Given the ratio of pancakes to quarts of milk for 18 pancakes,
    and the conversion rate of quarts to pints,
    prove that the number of pints needed for 9 pancakes is 3. -/
theorem pancake_milk_calculation (pancakes_18 : ℕ) (quarts_18 : ℚ) (pints_per_quart : ℚ) :
  pancakes_18 = 18 →
  quarts_18 = 3 →
  pints_per_quart = 2 →
  (9 : ℚ) * quarts_18 * pints_per_quart / pancakes_18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pancake_milk_calculation_l3861_386109


namespace NUMINAMATH_CALUDE_number_triples_satisfying_equation_l3861_386168

theorem number_triples_satisfying_equation :
  ∀ (a b c : ℕ), a^(b+20) * (c-1) = c^(b+21) - 1 ↔ 
    ((a = 1 ∧ c = 0) ∨ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_number_triples_satisfying_equation_l3861_386168


namespace NUMINAMATH_CALUDE_projection_coplanarity_l3861_386161

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with a quadrilateral base -/
structure Pyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D

/-- Checks if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersectionPoint (p1 p2 p3 p4 : Point3D) : Point3D := sorry

/-- Checks if a point is the height of a pyramid -/
def isHeight (p : Point3D) (pyr : Pyramid) : Prop := sorry

/-- Projects a point onto a plane defined by three points -/
def projectOntoPlane (p : Point3D) (p1 p2 p3 : Point3D) : Point3D := sorry

/-- Checks if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

theorem projection_coplanarity (pyr : Pyramid) : 
  let M := intersectionPoint pyr.A pyr.C pyr.B pyr.D
  isPerpendicular pyr.A pyr.C pyr.B pyr.D ∧ 
  isHeight (intersectionPoint pyr.E M pyr.A pyr.C) pyr →
  areCoplanar 
    (projectOntoPlane M pyr.E pyr.A pyr.B)
    (projectOntoPlane M pyr.E pyr.B pyr.C)
    (projectOntoPlane M pyr.E pyr.C pyr.D)
    (projectOntoPlane M pyr.E pyr.D pyr.A) := by
  sorry

end NUMINAMATH_CALUDE_projection_coplanarity_l3861_386161
