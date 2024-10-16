import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_division_problem_l434_43462

theorem polynomial_division_problem (a : ℤ) : 
  (∃ p : Polynomial ℤ, (X^2 - 2*X + a) * p = X^13 + 2*X + 180) ↔ a = 3 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_problem_l434_43462


namespace NUMINAMATH_CALUDE_leadership_arrangements_l434_43453

/-- Represents the number of teachers -/
def num_teachers : ℕ := 5

/-- Represents the number of extracurricular groups -/
def num_groups : ℕ := 3

/-- Represents the maximum number of leaders per group -/
def max_leaders_per_group : ℕ := 2

/-- Represents that teachers A and B cannot lead alone -/
def ab_cannot_lead_alone : Prop := True

/-- The number of different leadership arrangements -/
def num_arrangements : ℕ := 54

/-- Theorem stating that the number of different leadership arrangements
    for the given conditions is equal to 54 -/
theorem leadership_arrangements :
  num_teachers = 5 ∧
  num_groups = 3 ∧
  max_leaders_per_group = 2 ∧
  ab_cannot_lead_alone →
  num_arrangements = 54 := by
  sorry

end NUMINAMATH_CALUDE_leadership_arrangements_l434_43453


namespace NUMINAMATH_CALUDE_english_chinese_difference_l434_43432

/-- Represents the number of hours Ryan spends studying each subject on weekdays and weekends --/
structure StudyHours where
  english_weekday : ℕ
  chinese_weekday : ℕ
  english_weekend : ℕ
  chinese_weekend : ℕ

/-- Calculates the total hours spent on a subject in a week --/
def total_hours (hours : StudyHours) (weekday : ℕ) (weekend : ℕ) : ℕ :=
  hours.english_weekday * weekday + hours.chinese_weekday * weekday +
  hours.english_weekend * weekend + hours.chinese_weekend * weekend

/-- Theorem stating the difference in hours spent on English vs Chinese --/
theorem english_chinese_difference (hours : StudyHours) 
  (h1 : hours.english_weekday = 6)
  (h2 : hours.chinese_weekday = 3)
  (h3 : hours.english_weekend = 2)
  (h4 : hours.chinese_weekend = 1)
  : total_hours hours 5 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_english_chinese_difference_l434_43432


namespace NUMINAMATH_CALUDE_complement_of_A_l434_43422

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ -3} ∪ {x | x ≥ 0}

-- Theorem statement
theorem complement_of_A : Set.compl A = Set.Ioo (-3) 0 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l434_43422


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9911_l434_43488

theorem largest_prime_factor_of_9911 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 9911 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 9911 → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9911_l434_43488


namespace NUMINAMATH_CALUDE_exchange_75_cad_to_yen_l434_43478

/-- Represents the exchange rate between Japanese Yen (JPY) and Canadian Dollars (CAD) -/
def exchange_rate : ℚ := 5000 / 60

/-- Represents the amount of Canadian Dollars to be exchanged -/
def cad_to_exchange : ℚ := 75

/-- Calculates the amount of Japanese Yen received for a given amount of Canadian Dollars -/
def exchange (cad : ℚ) : ℚ := cad * exchange_rate

/-- Theorem stating that exchanging 75 CAD results in 6250 JPY -/
theorem exchange_75_cad_to_yen : exchange cad_to_exchange = 6250 := by
  sorry

end NUMINAMATH_CALUDE_exchange_75_cad_to_yen_l434_43478


namespace NUMINAMATH_CALUDE_min_buses_for_field_trip_l434_43406

def min_buses (total_students : ℕ) (bus_cap_1 bus_cap_2 : ℕ) (min_bus_2 : ℕ) : ℕ :=
  let x := ((total_students - bus_cap_2 * min_bus_2 + bus_cap_1 - 1) / bus_cap_1 : ℕ)
  x + min_bus_2

theorem min_buses_for_field_trip :
  min_buses 530 45 35 3 = 13 :=
sorry

end NUMINAMATH_CALUDE_min_buses_for_field_trip_l434_43406


namespace NUMINAMATH_CALUDE_coat_price_problem_l434_43401

theorem coat_price_problem (price_reduction : ℝ) (percentage_reduction : ℝ) :
  price_reduction = 200 →
  percentage_reduction = 0.40 →
  ∃ original_price : ℝ, 
    original_price * percentage_reduction = price_reduction ∧
    original_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_problem_l434_43401


namespace NUMINAMATH_CALUDE_solution_set_f_leq_15_max_a_for_inequality_l434_43470

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the solution set of f(x) ≤ 15
theorem solution_set_f_leq_15 :
  {x : ℝ | f x ≤ 15} = Set.Icc (-8 : ℝ) 7 := by sorry

-- Theorem for the maximum value of a
theorem max_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, -x^2 + a ≤ f x) ↔ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_15_max_a_for_inequality_l434_43470


namespace NUMINAMATH_CALUDE_right_triangle_fraction_zero_smallest_constant_zero_l434_43437

theorem right_triangle_fraction_zero (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2) = 0 := by
  sorry

theorem smallest_constant_zero :
  ∃ N, N = 0 ∧ ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + b^2 = c^2 →
    (a^2 + b^2 - c^2) / (a^2 + b^2 + c^2) < N := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_fraction_zero_smallest_constant_zero_l434_43437


namespace NUMINAMATH_CALUDE_division_theorem_l434_43490

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 125 →
  divisor = 15 →
  remainder = 5 →
  quotient = (dividend - remainder) / divisor →
  quotient = 8 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l434_43490


namespace NUMINAMATH_CALUDE_zero_last_to_appear_l434_43409

-- Define the Fibonacci sequence modulo 9
def fibMod9 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => (fibMod9 n + fibMod9 (n + 1)) % 9

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppeared (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fibMod9 k = d

-- Define a function to check if all digits from 0 to 8 have appeared
def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d, d ≤ 8 → digitAppeared d n

-- The main theorem
theorem zero_last_to_appear :
  ∃ n, allDigitsAppeared n ∧
    ¬(∃ k < n, allDigitsAppeared k) ∧
    fibMod9 n = 0 :=
  sorry

end NUMINAMATH_CALUDE_zero_last_to_appear_l434_43409


namespace NUMINAMATH_CALUDE_minimum_canvas_dimensions_l434_43403

/-- Represents the dimensions of a canvas -/
structure CanvasDimensions where
  width : ℝ
  height : ℝ

/-- Represents the constraints for the canvas -/
structure CanvasConstraints where
  miniatureArea : ℝ
  topBottomMargin : ℝ
  sideMargin : ℝ

/-- Calculates the total area of the canvas given its dimensions -/
def totalArea (d : CanvasDimensions) : ℝ :=
  d.width * d.height

/-- Checks if the given dimensions satisfy the constraints -/
def satisfiesConstraints (d : CanvasDimensions) (c : CanvasConstraints) : Prop :=
  (d.width - 2 * c.sideMargin) * (d.height - 2 * c.topBottomMargin) = c.miniatureArea

/-- Theorem stating that the minimum dimensions of the canvas are 10 cm × 20 cm -/
theorem minimum_canvas_dimensions :
  ∃ (d : CanvasDimensions),
    d.width = 10 ∧
    d.height = 20 ∧
    satisfiesConstraints d { miniatureArea := 72, topBottomMargin := 4, sideMargin := 2 } ∧
    (∀ (d' : CanvasDimensions),
      satisfiesConstraints d' { miniatureArea := 72, topBottomMargin := 4, sideMargin := 2 } →
      totalArea d ≤ totalArea d') :=
by
  sorry


end NUMINAMATH_CALUDE_minimum_canvas_dimensions_l434_43403


namespace NUMINAMATH_CALUDE_min_cards_for_even_product_l434_43434

/-- Represents a card with an integer value -/
structure Card where
  value : Int
  even : Bool

/-- The set of cards in the box -/
def cards : Finset Card :=
  sorry

/-- A valid sequence of drawn cards according to the rules -/
def ValidSequence : List Card → Prop :=
  sorry

/-- The product of the values of a list of cards -/
def product : List Card → Int :=
  sorry

/-- Theorem: The minimum number of cards to ensure an even product is 3 -/
theorem min_cards_for_even_product :
  ∀ (s : List Card), ValidSequence s → product s % 2 = 0 → s.length ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_cards_for_even_product_l434_43434


namespace NUMINAMATH_CALUDE_first_same_side_after_104_minutes_l434_43444

/-- Represents a person walking around a pentagonal square -/
structure Walker where
  start_point : Fin 5
  speed : ℝ

/-- The time when two walkers are first on the same side of a pentagonal square -/
def first_same_side_time (perimeter : ℝ) (walker_a walker_b : Walker) : ℝ :=
  sorry

/-- The main theorem -/
theorem first_same_side_after_104_minutes :
  let perimeter : ℝ := 2000
  let walker_a : Walker := { start_point := 0, speed := 50 }
  let walker_b : Walker := { start_point := 2, speed := 46 }
  first_same_side_time perimeter walker_a walker_b = 104 := by
  sorry

end NUMINAMATH_CALUDE_first_same_side_after_104_minutes_l434_43444


namespace NUMINAMATH_CALUDE_extra_parts_calculation_l434_43499

/-- The number of extra parts produced compared to the original plan -/
def extra_parts (initial_rate : ℕ) (initial_days : ℕ) (rate_increase : ℕ) (total_parts : ℕ) : ℕ :=
  let total_days := (total_parts - initial_rate * initial_days) / (initial_rate + rate_increase) + initial_days
  let actual_production := initial_rate * initial_days + (initial_rate + rate_increase) * (total_days - initial_days)
  let planned_production := initial_rate * total_days
  actual_production - planned_production

theorem extra_parts_calculation :
  extra_parts 25 3 5 675 = 100 := by
  sorry

end NUMINAMATH_CALUDE_extra_parts_calculation_l434_43499


namespace NUMINAMATH_CALUDE_extreme_value_implies_fourth_quadrant_l434_43402

def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x

theorem extreme_value_implies_fourth_quadrant (a b : ℝ) :
  (∃ (c : ℝ), f a b 1 = c ∧ c = 10 ∧ ∀ x, f a b x ≤ c) →
  a < 0 ∧ b > 0 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_fourth_quadrant_l434_43402


namespace NUMINAMATH_CALUDE_parallel_line_x_coordinate_l434_43441

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two points form a line parallel to the y-axis -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

/-- The theorem statement -/
theorem parallel_line_x_coordinate 
  (a : ℝ) 
  (P : Point) 
  (Q : Point) 
  (h1 : P = ⟨a, -5⟩) 
  (h2 : Q = ⟨4, 3⟩) 
  (h3 : parallelToYAxis P Q) : 
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_x_coordinate_l434_43441


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achieved_l434_43471

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 14400/29 := by
  sorry

theorem min_sum_squares_achieved (ε : ℝ) (h_pos : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
  2*x₁ + 3*x₂ + 4*x₃ = 120 ∧ 
  x₁^2 + x₂^2 + x₃^2 < 14400/29 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_achieved_l434_43471


namespace NUMINAMATH_CALUDE_solution_set_l434_43455

def f (x : ℝ) := 3 - 2*x

theorem solution_set (x : ℝ) : 
  x ∈ Set.Icc 0 3 ↔ |f (x + 1) + 2| ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l434_43455


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l434_43419

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  8 * p^4 + 18 * q^4 + 50 * r^4 + 1 / (8 * p * q * r) ≥ 6 :=
by sorry

theorem min_value_achieved (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ (p₀ q₀ r₀ : ℝ), p₀ > 0 ∧ q₀ > 0 ∧ r₀ > 0 ∧
    8 * p₀^4 + 18 * q₀^4 + 50 * r₀^4 + 1 / (8 * p₀ * q₀ * r₀) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l434_43419


namespace NUMINAMATH_CALUDE_probability_total_more_than_seven_l434_43423

/-- The number of faces on each die -/
def numFaces : Nat := 6

/-- The total number of possible outcomes when throwing two dice -/
def totalOutcomes : Nat := numFaces * numFaces

/-- The number of favorable outcomes (total > 7) -/
def favorableOutcomes : Nat := 14

/-- The probability of getting a total more than 7 -/
def probabilityTotalMoreThan7 : Rat := favorableOutcomes / totalOutcomes

theorem probability_total_more_than_seven :
  probabilityTotalMoreThan7 = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_total_more_than_seven_l434_43423


namespace NUMINAMATH_CALUDE_school_supplies_problem_l434_43425

/-- Represents the school supplies problem --/
theorem school_supplies_problem 
  (num_students : ℕ) 
  (pens_per_student : ℕ) 
  (notebooks_per_student : ℕ) 
  (binders_per_student : ℕ) 
  (pen_cost : ℚ) 
  (notebook_cost : ℚ) 
  (binder_cost : ℚ) 
  (highlighter_cost : ℚ) 
  (teacher_discount : ℚ) 
  (total_spent : ℚ) 
  (h1 : num_students = 30)
  (h2 : pens_per_student = 5)
  (h3 : notebooks_per_student = 3)
  (h4 : binders_per_student = 1)
  (h5 : pen_cost = 1/2)
  (h6 : notebook_cost = 5/4)
  (h7 : binder_cost = 17/4)
  (h8 : highlighter_cost = 3/4)
  (h9 : teacher_discount = 100)
  (h10 : total_spent = 260) :
  (total_spent - (num_students * (pens_per_student * pen_cost + 
   notebooks_per_student * notebook_cost + 
   binders_per_student * binder_cost) - teacher_discount)) / 
  (num_students * highlighter_cost) = 2 := by
sorry


end NUMINAMATH_CALUDE_school_supplies_problem_l434_43425


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l434_43408

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / x + 9 / y = 2) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 1 / a + 9 / b = 2 → x + y ≤ a + b ∧ 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 9 / y = 2 ∧ x + y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l434_43408


namespace NUMINAMATH_CALUDE_train_crossing_time_l434_43417

/-- The time taken for a faster train to cross a man in a slower train -/
theorem train_crossing_time (faster_speed slower_speed : ℝ) (train_length : ℝ) : 
  faster_speed = 72 → 
  slower_speed = 36 → 
  train_length = 100 → 
  (train_length / ((faster_speed - slower_speed) * (5/18))) = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l434_43417


namespace NUMINAMATH_CALUDE_center_value_theorem_l434_43466

/-- Represents a 6x6 matrix with arithmetic sequences in rows and columns -/
def ArithmeticMatrix := Matrix (Fin 6) (Fin 6) ℝ

/-- Checks if a sequence is arithmetic -/
def is_arithmetic_sequence (seq : Fin 6 → ℝ) : Prop :=
  ∀ i j k : Fin 6, i < j ∧ j < k → seq j - seq i = seq k - seq j

/-- The matrix has arithmetic sequences in all rows and columns -/
def matrix_arithmetic (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 6, is_arithmetic_sequence (λ j => M i j)) ∧
  (∀ j : Fin 6, is_arithmetic_sequence (λ i => M i j))

theorem center_value_theorem (M : ArithmeticMatrix) 
  (h_arithmetic : matrix_arithmetic M)
  (h_first_row : M 0 1 = 3 ∧ M 0 4 = 27)
  (h_last_row : M 5 1 = 25 ∧ M 5 4 = 85) :
  M 2 2 = 30 ∧ M 2 3 = 30 ∧ M 3 2 = 30 ∧ M 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_center_value_theorem_l434_43466


namespace NUMINAMATH_CALUDE_truck_speed_problem_l434_43405

/-- The average speed of Truck Y in miles per hour -/
def speed_y : ℝ := 63

/-- The time it takes for Truck Y to overtake Truck X in hours -/
def overtake_time : ℝ := 3

/-- The initial distance Truck X is ahead of Truck Y in miles -/
def initial_gap : ℝ := 14

/-- The distance Truck Y is ahead of Truck X after overtaking in miles -/
def final_gap : ℝ := 4

/-- The average speed of Truck X in miles per hour -/
def speed_x : ℝ := 57

theorem truck_speed_problem :
  speed_y * overtake_time = speed_x * overtake_time + initial_gap + final_gap := by
  sorry

#check truck_speed_problem

end NUMINAMATH_CALUDE_truck_speed_problem_l434_43405


namespace NUMINAMATH_CALUDE_square_side_length_l434_43463

-- Define the right triangle PQR
def triangle_PQR (PQ PR : ℝ) : Prop := PQ = 5 ∧ PR = 12

-- Define the square on the hypotenuse
def square_on_hypotenuse (s : ℝ) (PQ PR : ℝ) : Prop :=
  ∃ (x : ℝ), 
    s / (PQ^2 + PR^2).sqrt = x / PR ∧
    s / (PR - PQ * PR / (PQ^2 + PR^2).sqrt) = (PR - x) / PR

-- Theorem statement
theorem square_side_length (PQ PR s : ℝ) : 
  triangle_PQR PQ PR →
  square_on_hypotenuse s PQ PR →
  s = 96.205 / 20.385 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l434_43463


namespace NUMINAMATH_CALUDE_marbles_lost_l434_43467

theorem marbles_lost (initial : ℕ) (final : ℕ) (h1 : initial = 38) (h2 : final = 23) :
  initial - final = 15 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l434_43467


namespace NUMINAMATH_CALUDE_composite_plus_four_prime_l434_43413

/-- A number is composite if it has a factor between 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A number is prime if it's greater than 1 and its only factors are 1 and itself -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0

theorem composite_plus_four_prime :
  ∃ n : ℕ, IsComposite n ∧ IsPrime (n + 4) :=
sorry

end NUMINAMATH_CALUDE_composite_plus_four_prime_l434_43413


namespace NUMINAMATH_CALUDE_missing_number_is_1745_l434_43452

def known_numbers : List ℕ := [744, 747, 748, 749, 752, 752, 753, 755, 755]

theorem missing_number_is_1745 :
  let total_count : ℕ := 10
  let average : ℕ := 750
  let sum_known : ℕ := known_numbers.sum
  let missing_number : ℕ := total_count * average - sum_known
  missing_number = 1745 := by sorry

end NUMINAMATH_CALUDE_missing_number_is_1745_l434_43452


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l434_43431

open Set

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (9 - x^2)}
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (4*x - x^2)}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Bᶜ) = Ioo (-3) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l434_43431


namespace NUMINAMATH_CALUDE_sara_popsicle_consumption_l434_43495

/-- The number of Popsicles Sara can eat in a given time period -/
def popsicles_eaten (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / minutes_per_popsicle

theorem sara_popsicle_consumption :
  popsicles_eaten 20 340 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sara_popsicle_consumption_l434_43495


namespace NUMINAMATH_CALUDE_apartment_renovation_is_credence_good_decision_is_difficult_l434_43429

-- Define the types
structure Service where
  name : String
  is_credence_good : Bool
  has_info_asymmetry : Bool
  quality_hard_to_assess : Bool

-- Define the apartment renovation service
def apartment_renovation : Service where
  name := "Complete Apartment Renovation"
  is_credence_good := true
  has_info_asymmetry := true
  quality_hard_to_assess := true

-- Theorem statement
theorem apartment_renovation_is_credence_good :
  apartment_renovation.is_credence_good ∧
  apartment_renovation.has_info_asymmetry ∧
  apartment_renovation.quality_hard_to_assess :=
by sorry

-- Define the provider types
inductive Provider
| ConstructionCompany
| PrivateRepairCrew

-- Define a function to represent the decision-making process
def choose_provider (service : Service) : Provider → Bool
| Provider.ConstructionCompany => true  -- Simplified for demonstration
| Provider.PrivateRepairCrew => false   -- Simplified for demonstration

-- Theorem about the difficulty of the decision
theorem decision_is_difficult (service : Service) :
  ∃ (p1 p2 : Provider), p1 ≠ p2 ∧ choose_provider service p1 = choose_provider service p2 :=
by sorry

end NUMINAMATH_CALUDE_apartment_renovation_is_credence_good_decision_is_difficult_l434_43429


namespace NUMINAMATH_CALUDE_power_calculation_l434_43420

theorem power_calculation : 
  ((18^13 * 18^11)^2 / 6^8) * 3^4 = 2^40 * 3^92 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l434_43420


namespace NUMINAMATH_CALUDE_composition_equality_l434_43430

def f (a b x : ℝ) : ℝ := a * x + b
def g (c d x : ℝ) : ℝ := c * x + d

theorem composition_equality (a b c d : ℝ) :
  (∀ x, f a b (g c d x) = g c d (f a b x)) ↔ b * (1 - c) - d * (1 - a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l434_43430


namespace NUMINAMATH_CALUDE_right_triangle_with_tangent_circle_l434_43407

theorem right_triangle_with_tangent_circle (a b c r : ℕ) : 
  a^2 + b^2 = c^2 → -- right triangle
  Nat.gcd a (Nat.gcd b c) = 1 → -- side lengths have no common divisor greater than 1
  r = (a + b - c) / 2 → -- radius of circle tangent to hypotenuse
  r = 420 → -- given radius
  (a = 399 ∧ b = 40 ∧ c = 401) ∨ (a = 40 ∧ b = 399 ∧ c = 401) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_with_tangent_circle_l434_43407


namespace NUMINAMATH_CALUDE_mean_height_of_basketball_team_l434_43418

def heights : List ℝ := [48, 50, 51, 54, 56, 57, 57, 59, 60, 63, 64, 65, 67, 69, 69, 71, 72, 74]

theorem mean_height_of_basketball_team : 
  (heights.sum / heights.length : ℝ) = 61.444444444444445 := by sorry

end NUMINAMATH_CALUDE_mean_height_of_basketball_team_l434_43418


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l434_43449

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 -/
def linear_coefficient (a b c : ℚ) : ℚ := b

/-- The quadratic equation 2x^2 - 3x - 4 = 0 -/
def quadratic_equation (x : ℚ) : Prop := 2 * x^2 - 3 * x - 4 = 0

theorem linear_coefficient_of_quadratic :
  linear_coefficient 2 (-3) (-4) = -3 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l434_43449


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_199_l434_43493

theorem modular_inverse_of_3_mod_199 : ∃ x : ℕ, 0 < x ∧ x < 199 ∧ (3 * x) % 199 = 1 :=
by
  use 133
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_199_l434_43493


namespace NUMINAMATH_CALUDE_return_trip_time_l434_43492

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  time_against_wind : ℝ  -- time taken against wind
  time_diff : ℝ  -- time difference between still air and with wind

/-- The main theorem about the return trip time -/
theorem return_trip_time (fs : FlightScenario) 
  (h1 : fs.time_against_wind = 90)
  (h2 : fs.d = fs.time_against_wind * (fs.p - fs.w))
  (h3 : fs.d / (fs.p + fs.w) = fs.d / fs.p - fs.time_diff)
  (h4 : fs.time_diff = 12) :
  fs.d / (fs.p + fs.w) = 18 ∨ fs.d / (fs.p + fs.w) = 60 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l434_43492


namespace NUMINAMATH_CALUDE_existence_of_equal_differences_l434_43459

theorem existence_of_equal_differences (n : ℕ) (a : Fin (2 * n) → ℕ)
  (h_n : n ≥ 3)
  (h_a : ∀ i j : Fin (2 * n), i < j → a i < a j)
  (h_bounds : ∀ i : Fin (2 * n), 1 ≤ a i ∧ a i ≤ n^2) :
  ∃ i₁ i₂ i₃ i₄ i₅ i₆ : Fin (2 * n),
    i₁ < i₂ ∧ i₂ ≤ i₃ ∧ i₃ < i₄ ∧ i₄ ≤ i₅ ∧ i₅ < i₆ ∧
    a i₂ - a i₁ = a i₄ - a i₃ ∧ a i₄ - a i₃ = a i₆ - a i₅ :=
by sorry

end NUMINAMATH_CALUDE_existence_of_equal_differences_l434_43459


namespace NUMINAMATH_CALUDE_salary_restoration_l434_43427

theorem salary_restoration (original_salary : ℝ) (reduced_salary : ℝ) : 
  reduced_salary = original_salary * (1 - 0.5) →
  reduced_salary * 2 = original_salary :=
by
  sorry

end NUMINAMATH_CALUDE_salary_restoration_l434_43427


namespace NUMINAMATH_CALUDE_binary_sequence_equiv_powerset_nat_l434_43461

/-- The type of infinite binary sequences -/
def BinarySequence := ℕ → Bool

/-- The theorem stating the equinumerosity of binary sequences and subsets of naturals -/
theorem binary_sequence_equiv_powerset_nat :
  ∃ (f : BinarySequence → Set ℕ), Function.Bijective f :=
sorry

end NUMINAMATH_CALUDE_binary_sequence_equiv_powerset_nat_l434_43461


namespace NUMINAMATH_CALUDE_football_lineup_combinations_l434_43494

def total_members : ℕ := 12
def offensive_linemen : ℕ := 4
def positions : ℕ := 5

def lineup_combinations : ℕ :=
  offensive_linemen * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

theorem football_lineup_combinations :
  lineup_combinations = 31680 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_combinations_l434_43494


namespace NUMINAMATH_CALUDE_D_72_eq_27_l434_43482

/-- 
D(n) represents the number of ways to write a positive integer n as a product of 
integers greater than 1, where the order matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- 
factorizations(n) represents the list of all valid factorizations of n,
where each factorization is a list of integers greater than 1.
-/
def factorizations (n : ℕ+) : List (List ℕ+) := sorry

/-- 
is_valid_factorization(n, factors) checks if the given list of factors
is a valid factorization of n according to the problem's conditions.
-/
def is_valid_factorization (n : ℕ+) (factors : List ℕ+) : Prop :=
  factors.all (· > 1) ∧ factors.prod = n

theorem D_72_eq_27 : D 72 = 27 := by sorry

end NUMINAMATH_CALUDE_D_72_eq_27_l434_43482


namespace NUMINAMATH_CALUDE_quinary1234_equals_octal302_l434_43481

/-- Converts a quinary (base-5) number to decimal (base-10) --/
def quinaryToDecimal (q : ℕ) : ℕ := sorry

/-- Converts a decimal (base-10) number to octal (base-8) --/
def decimalToOctal (d : ℕ) : ℕ := sorry

/-- The quinary representation of 1234 --/
def quinary1234 : ℕ := 1234

/-- The octal representation of 302 --/
def octal302 : ℕ := 302

theorem quinary1234_equals_octal302 : 
  decimalToOctal (quinaryToDecimal quinary1234) = octal302 := by sorry

end NUMINAMATH_CALUDE_quinary1234_equals_octal302_l434_43481


namespace NUMINAMATH_CALUDE_coprime_iff_no_common_prime_factor_l434_43439

theorem coprime_iff_no_common_prime_factor (a b : ℕ) : 
  Nat.gcd a b = 1 ↔ ¬ ∃ (p : ℕ), Nat.Prime p ∧ p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_coprime_iff_no_common_prime_factor_l434_43439


namespace NUMINAMATH_CALUDE_board_numbers_theorem_l434_43465

theorem board_numbers_theorem (a b c : ℝ) : 
  ({a, b, c} : Set ℝ) = {a - 2, b + 2, c^2} → 
  a + b + c = 2005 → 
  a = 1003 ∨ a = 1002 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_theorem_l434_43465


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l434_43476

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + (1/2)*x + a - 2 = 0 ∧ x = 1) → 
  (∃ y : ℝ, y^2 + (1/2)*y + a - 2 = 0 ∧ y = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l434_43476


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l434_43414

theorem complex_expression_evaluation :
  let i : ℂ := Complex.I
  ((2 + i) * (3 + i)) / (1 + i) = 5 := by sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l434_43414


namespace NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l434_43457

/-- The fraction of paint remaining after two days of use -/
def paint_remaining (initial_amount : ℚ) (first_day_use : ℚ) (second_day_use : ℚ) : ℚ :=
  initial_amount - (first_day_use * initial_amount) - (second_day_use * (initial_amount - first_day_use * initial_amount))

/-- Theorem stating that the fraction of paint remaining after two days is 3/8 -/
theorem paint_remaining_is_three_eighths :
  paint_remaining 1 (1/4) (1/2) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_is_three_eighths_l434_43457


namespace NUMINAMATH_CALUDE_game_probability_specific_case_l434_43426

def game_probability (total_rounds : ℕ) 
  (alex_prob : ℚ) (chelsea_prob : ℚ) (mel_prob : ℚ)
  (alex_wins : ℕ) (chelsea_wins : ℕ) (mel_wins : ℕ) : ℚ :=
  (alex_prob ^ alex_wins) * 
  (chelsea_prob ^ chelsea_wins) * 
  (mel_prob ^ mel_wins) * 
  (Nat.choose total_rounds alex_wins).choose chelsea_wins

theorem game_probability_specific_case : 
  game_probability 8 (5/12) (1/3) (1/4) 3 4 1 = 625/9994 := by
  sorry

end NUMINAMATH_CALUDE_game_probability_specific_case_l434_43426


namespace NUMINAMATH_CALUDE_special_function_value_l434_43428

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value (f : ℝ → ℝ) 
  (h : special_function f) (h1000 : f 1000 = 2) : f 750 = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l434_43428


namespace NUMINAMATH_CALUDE_exam_failure_count_l434_43498

theorem exam_failure_count (total_students : ℕ) (pass_percentage : ℚ) 
  (h1 : total_students = 840)
  (h2 : pass_percentage = 35 / 100) :
  (total_students : ℚ) * (1 - pass_percentage) = 546 := by
  sorry

end NUMINAMATH_CALUDE_exam_failure_count_l434_43498


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l434_43435

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) 
  (h1 : q → p)  -- p is necessary for q
  (h2 : ¬(p → q))  -- p is not sufficient for q
  : (¬p → ¬q) ∧ ¬(¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l434_43435


namespace NUMINAMATH_CALUDE_sarah_father_age_double_l434_43483

/-- Given Sarah's age and her father's age in 2010, find the year when the father's age will be double Sarah's age -/
theorem sarah_father_age_double (sarah_age_2010 : ℕ) (father_age_2010 : ℕ) 
  (h1 : sarah_age_2010 = 10)
  (h2 : father_age_2010 = 6 * sarah_age_2010) :
  ∃ (year : ℕ), 
    year > 2010 ∧ 
    (father_age_2010 + (year - 2010)) = 2 * (sarah_age_2010 + (year - 2010)) ∧
    year = 2030 :=
by sorry

end NUMINAMATH_CALUDE_sarah_father_age_double_l434_43483


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2048_l434_43473

theorem tens_digit_of_8_pow_2048 : 8^2048 % 100 = 88 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2048_l434_43473


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l434_43458

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 1) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l434_43458


namespace NUMINAMATH_CALUDE_incorrect_inequality_l434_43445

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l434_43445


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l434_43485

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (2, m) (m, 2) → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l434_43485


namespace NUMINAMATH_CALUDE_lucy_age_l434_43404

/-- Given that Lucy's age is three times Helen's age and the sum of their ages is 60,
    prove that Lucy is 45 years old. -/
theorem lucy_age (lucy helen : ℕ) 
  (h1 : lucy = 3 * helen) 
  (h2 : lucy + helen = 60) : 
  lucy = 45 := by
  sorry

end NUMINAMATH_CALUDE_lucy_age_l434_43404


namespace NUMINAMATH_CALUDE_daily_class_schedule_l434_43446

theorem daily_class_schedule (n m : ℕ) (hn : n = 10) (hm : m = 6) :
  (n.factorial / (n - m).factorial) = 151200 :=
sorry

end NUMINAMATH_CALUDE_daily_class_schedule_l434_43446


namespace NUMINAMATH_CALUDE_inverse_square_relation_l434_43464

theorem inverse_square_relation (k : ℝ) (a b c : ℝ) :
  (∀ a b c, a^2 * b^2 / c = k) →
  (4^2 * 2^2 / 3 = k) →
  (a^2 * 4^2 / 6 = k) →
  a^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l434_43464


namespace NUMINAMATH_CALUDE_unique_A_for_3AA1_multiple_of_9_l434_43416

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def four_digit_3AA1 (A : ℕ) : ℕ := 3000 + 100 * A + 10 * A + 1

theorem unique_A_for_3AA1_multiple_of_9 :
  ∃! A : ℕ, A < 10 ∧ is_multiple_of_9 (four_digit_3AA1 A) ∧ A = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_A_for_3AA1_multiple_of_9_l434_43416


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l434_43400

/-- A geometric sequence with a_2 = 5 and a_6 = 33 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q ∧ a 2 = 5 ∧ a 6 = 33

/-- The product of a_3 and a_5 in the geometric sequence is 165 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 3 * a 5 = 165 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l434_43400


namespace NUMINAMATH_CALUDE_a_range_l434_43448

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- State the theorem
theorem a_range (a : ℝ) :
  (f a 3 < f a 4) ∧
  (∀ n : ℕ, n ≥ 8 → f a n > f a (n + 1)) →
  -1/7 < a ∧ a < -1/17 :=
sorry

end NUMINAMATH_CALUDE_a_range_l434_43448


namespace NUMINAMATH_CALUDE_square_property_l434_43484

theorem square_property (n : ℕ) :
  (∃ (d : Finset ℕ), d.card = 6 ∧ ∀ x ∈ d, x ∣ (n^5 + n^4 + 1)) →
  ∃ k : ℕ, n^3 - n + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_l434_43484


namespace NUMINAMATH_CALUDE_contrapositive_prop2_true_l434_43447

-- Proposition 1
axiom prop1 : ∀ a b : ℝ, a > b → (1 / a) < (1 / b)

-- Proposition 2
axiom prop2 : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x + 2) * (x - 3) ≤ 0

-- Theorem: The contrapositive of Proposition 2 is true
theorem contrapositive_prop2_true :
  ∀ x : ℝ, (x + 2) * (x - 3) > 0 → (x < -2 ∨ x > 0) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_prop2_true_l434_43447


namespace NUMINAMATH_CALUDE_symmetry_axis_condition_l434_43468

theorem symmetry_axis_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) ↔ x = (p * (-y) + q) / (r * (-y) + s)) →
  p + s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_condition_l434_43468


namespace NUMINAMATH_CALUDE_no_valid_coloring_200_points_l434_43497

/-- Represents a coloring of points and segments -/
structure Coloring (n : ℕ) (k : ℕ) where
  pointColor : Fin n → Fin k
  segmentColor : Fin n → Fin n → Fin k

/-- Predicate for a valid coloring -/
def isValidColoring (n : ℕ) (k : ℕ) (c : Coloring n k) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    c.pointColor i ≠ c.pointColor j ∧
    c.pointColor i ≠ c.segmentColor i j ∧
    c.pointColor j ≠ c.segmentColor i j

/-- Theorem stating the impossibility of valid coloring for 200 points with 7 or 10 colors -/
theorem no_valid_coloring_200_points :
  ¬ (∃ c : Coloring 200 7, isValidColoring 200 7 c) ∧
  ¬ (∃ c : Coloring 200 10, isValidColoring 200 10 c) := by
  sorry


end NUMINAMATH_CALUDE_no_valid_coloring_200_points_l434_43497


namespace NUMINAMATH_CALUDE_p_not_sufficient_p_not_necessary_p_neither_sufficient_nor_necessary_l434_43496

/-- Proposition p: x ≠ 2 and y ≠ 3 -/
def p (x y : ℝ) : Prop := x ≠ 2 ∧ y ≠ 3

/-- Proposition q: x + y ≠ 5 -/
def q (x y : ℝ) : Prop := x + y ≠ 5

/-- p is not a sufficient condition for q -/
theorem p_not_sufficient : ¬∀ x y : ℝ, p x y → q x y :=
sorry

/-- p is not a necessary condition for q -/
theorem p_not_necessary : ¬∀ x y : ℝ, q x y → p x y :=
sorry

/-- p is neither a sufficient nor a necessary condition for q -/
theorem p_neither_sufficient_nor_necessary : (¬∀ x y : ℝ, p x y → q x y) ∧ (¬∀ x y : ℝ, q x y → p x y) :=
sorry

end NUMINAMATH_CALUDE_p_not_sufficient_p_not_necessary_p_neither_sufficient_nor_necessary_l434_43496


namespace NUMINAMATH_CALUDE_polynomial_value_l434_43477

theorem polynomial_value : ∀ a b : ℝ, 
  (a * 1^3 + b * 1 + 1 = 2023) → 
  (a * (-1)^3 + b * (-1) - 2 = -2024) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l434_43477


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l434_43442

/-- The probability of picking two red balls from a bag containing 7 red, 5 blue, and 4 green balls -/
theorem probability_two_red_balls (red blue green : ℕ) (h1 : red = 7) (h2 : blue = 5) (h3 : green = 4) :
  let total := red + blue + green
  (red / total) * ((red - 1) / (total - 1)) = 7 / 40 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l434_43442


namespace NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l434_43450

theorem smallest_b_in_arithmetic_sequence (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- All terms are positive
  a = b - d →              -- a is the first term
  c = b + d →              -- c is the third term
  a * b * c = 125 →        -- Product of terms is 125
  ∀ x : ℝ, x > 0 ∧ x < b → ¬∃ y : ℝ, 
    (x - y) > 0 ∧ (x + y) > 0 ∧ (x - y) * x * (x + y) = 125 →
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_in_arithmetic_sequence_l434_43450


namespace NUMINAMATH_CALUDE_area_of_region_R_l434_43469

/-- A regular hexagon with side length 3 -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : ∀ i, dist (vertices i) (vertices ((i + 1) % 6)) = 3
  -- Additional properties of regularity could be added here

/-- The region R in the hexagon -/
def region_R (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {p | p ∈ interior h ∧ 
       ∀ i : Fin 6, i ≠ 0 → dist p (h.vertices 0) < dist p (h.vertices i)}
  where
  interior : RegularHexagon → Set (ℝ × ℝ) := sorry  -- Definition of hexagon interior

/-- The area of a set in ℝ² -/
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

theorem area_of_region_R (h : RegularHexagon) : 
  area (region_R h) = 27 * Real.sqrt 3 / 16 := by sorry

end NUMINAMATH_CALUDE_area_of_region_R_l434_43469


namespace NUMINAMATH_CALUDE_integer_divisibility_l434_43491

theorem integer_divisibility (m : ℕ) : 
  Prime m → 
  ∃ k : ℕ+, m = 13 * k + 1 → 
  m ≠ 8191 → 
  ∃ n : ℤ, (2^(m-1) - 1) = 8191 * m * n :=
sorry

end NUMINAMATH_CALUDE_integer_divisibility_l434_43491


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l434_43472

theorem fraction_sum_zero (a b c d : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l434_43472


namespace NUMINAMATH_CALUDE_min_sum_of_determinant_condition_l434_43421

theorem min_sum_of_determinant_condition (x y : ℤ) 
  (h : 1 < 6 - x * y ∧ 6 - x * y < 3) : 
  ∃ (a b : ℤ), a + b = -5 ∧ 
    (∀ (c d : ℤ), 1 < 6 - c * d ∧ 6 - c * d < 3 → a + b ≤ c + d) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_determinant_condition_l434_43421


namespace NUMINAMATH_CALUDE_double_inverse_g_10_l434_43487

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := Real.sqrt y - 1

-- Theorem statement
theorem double_inverse_g_10 :
  g_inv (g_inv 10) = Real.sqrt (Real.sqrt 10 - 1) - 1 :=
by sorry

end NUMINAMATH_CALUDE_double_inverse_g_10_l434_43487


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l434_43474

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (2 * x - 3) = 10 → x = 103 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l434_43474


namespace NUMINAMATH_CALUDE_cone_volume_increase_l434_43454

/-- The volume of a cone increases by a factor of 8 when its height and radius are doubled -/
theorem cone_volume_increase (r h V : ℝ) (r' h' V' : ℝ) : 
  V = (1/3) * π * r^2 * h →  -- Original volume
  r' = 2*r →                 -- New radius is doubled
  h' = 2*h →                 -- New height is doubled
  V' = (1/3) * π * r'^2 * h' →  -- New volume
  V' = 8*V := by
sorry


end NUMINAMATH_CALUDE_cone_volume_increase_l434_43454


namespace NUMINAMATH_CALUDE_mirror_area_l434_43415

/-- The area of a rectangular mirror that fits exactly inside a frame with given dimensions. -/
theorem mirror_area (frame_length frame_width frame_thickness : ℕ) : 
  frame_length = 70 ∧ frame_width = 90 ∧ frame_thickness = 15 → 
  (frame_length - 2 * frame_thickness) * (frame_width - 2 * frame_thickness) = 2400 :=
by
  sorry

#check mirror_area

end NUMINAMATH_CALUDE_mirror_area_l434_43415


namespace NUMINAMATH_CALUDE_condition_relationships_l434_43456

theorem condition_relationships (α β γ : Prop) 
  (h1 : β → α)  -- α is necessary for β
  (h2 : ¬(α → β))  -- α is not sufficient for β
  (h3 : γ ↔ β)  -- γ is necessary and sufficient for β
  : (γ → α) ∧ ¬(α → γ) := by sorry

end NUMINAMATH_CALUDE_condition_relationships_l434_43456


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_25_23_l434_43480

theorem half_abs_diff_squares_25_23 : (1 / 2 : ℝ) * |25^2 - 23^2| = 48 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_25_23_l434_43480


namespace NUMINAMATH_CALUDE_sequence_property_implies_rational_factor_l434_43436

/-- Two sequences are nonconstant if there exist two different terms -/
def Nonconstant (s : ℕ → ℚ) : Prop :=
  ∃ i j, i ≠ j ∧ s i ≠ s j

theorem sequence_property_implies_rational_factor
  (s t : ℕ → ℚ)
  (hs : Nonconstant s)
  (ht : Nonconstant t)
  (h : ∀ i j : ℕ, ∃ k : ℤ, (s i - s j) * (t i - t j) = k) :
  ∃ r : ℚ, (∀ i j : ℕ, ∃ m n : ℤ, (s i - s j) * r = m ∧ (t i - t j) / r = n) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_implies_rational_factor_l434_43436


namespace NUMINAMATH_CALUDE_vector_coordinates_proof_l434_43489

/-- Given points A, B, C in a 2D plane, and points M and N satisfying certain conditions,
    prove that M, N, and vector MN have specific coordinates. -/
theorem vector_coordinates_proof (A B C M N : ℝ × ℝ) : 
  A = (-2, 4) → 
  B = (3, -1) → 
  C = (-3, -4) → 
  M - C = 3 • (A - C) → 
  N - C = 2 • (B - C) → 
  M = (0, 20) ∧ 
  N = (9, 2) ∧ 
  N - M = (9, -18) := by
  sorry

end NUMINAMATH_CALUDE_vector_coordinates_proof_l434_43489


namespace NUMINAMATH_CALUDE_emily_age_proof_l434_43475

/-- Rachel's current age -/
def rachel_current_age : ℕ := 24

/-- Rachel's age when Emily was half her age -/
def rachel_past_age : ℕ := 8

/-- Emily's age when Rachel was 8 -/
def emily_past_age : ℕ := rachel_past_age / 2

/-- The constant age difference between Rachel and Emily -/
def age_difference : ℕ := rachel_past_age - emily_past_age

/-- Emily's current age -/
def emily_current_age : ℕ := rachel_current_age - age_difference

theorem emily_age_proof : emily_current_age = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_age_proof_l434_43475


namespace NUMINAMATH_CALUDE_area_AMDN_eq_area_ABC_l434_43433

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define that ABC is an acute triangle
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define points E and F on side BC
def E_on_BC (E B C : ℝ × ℝ) : Prop := sorry
def F_on_BC (F B C : ℝ × ℝ) : Prop := sorry

-- Define the angle equality
def angle_BAE_eq_CAF (A B C E F : ℝ × ℝ) : Prop := sorry

-- Define perpendicular lines
def FM_perp_AB (F M A B : ℝ × ℝ) : Prop := sorry
def FN_perp_AC (F N A C : ℝ × ℝ) : Prop := sorry

-- Define D as the intersection of extended AE and the circumcircle
def D_on_circumcircle (A B C D E : ℝ × ℝ) : Prop := sorry

-- Define area function
def area (points : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_AMDN_eq_area_ABC 
  (A B C D E F M N : ℝ × ℝ) 
  (h1 : is_acute_triangle A B C)
  (h2 : E_on_BC E B C)
  (h3 : F_on_BC F B C)
  (h4 : angle_BAE_eq_CAF A B C E F)
  (h5 : FM_perp_AB F M A B)
  (h6 : FN_perp_AC F N A C)
  (h7 : D_on_circumcircle A B C D E) :
  area [A, M, D, N] = area [A, B, C] := by sorry

end NUMINAMATH_CALUDE_area_AMDN_eq_area_ABC_l434_43433


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l434_43438

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 2 ∧ 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f x ≤ m) ∧
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ f x = m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l434_43438


namespace NUMINAMATH_CALUDE_colorings_count_l434_43460

/-- Represents the number of colors available --/
def num_colors : ℕ := 4

/-- Represents the number of triangles in the configuration --/
def num_triangles : ℕ := 4

/-- Represents the number of ways to color the first triangle --/
def first_triangle_colorings : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Represents the number of ways to color each subsequent triangle --/
def subsequent_triangle_colorings : ℕ := (num_colors - 1) * (num_colors - 2)

/-- The total number of possible colorings for the entire configuration --/
def total_colorings : ℕ := first_triangle_colorings * subsequent_triangle_colorings^(num_triangles - 1)

theorem colorings_count :
  total_colorings = 5184 :=
sorry

end NUMINAMATH_CALUDE_colorings_count_l434_43460


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l434_43412

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  -- The side length of the equilateral triangle
  s : ℝ
  -- The coordinates of the special point P
  p : ℝ × ℝ
  -- Condition that the triangle is equilateral
  equilateral : s > 0
  -- Condition that P is inside the triangle
  p_inside : p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 < s
  -- Conditions for distances from P to vertices
  dist_ap : Real.sqrt ((0 - p.1)^2 + (0 - p.2)^2) = 1
  dist_bp : Real.sqrt ((s - p.1)^2 + (0 - p.2)^2) = Real.sqrt 3
  dist_cp : Real.sqrt ((s/2 - p.1)^2 + (s*Real.sqrt 3/2 - p.2)^2) = 2

theorem special_triangle_side_length (t : SpecialTriangle) : t.s = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l434_43412


namespace NUMINAMATH_CALUDE_exists_initial_points_for_82_l434_43451

/-- The function that calculates the number of points after one application of the procedure -/
def points_after_one_procedure (n : ℕ) : ℕ := 3 * n - 2

/-- The function that calculates the number of points after two applications of the procedure -/
def points_after_two_procedures (n : ℕ) : ℕ := 9 * n - 8

/-- Theorem stating that there exists an initial number of points that results in 82 points after two procedures -/
theorem exists_initial_points_for_82 : ∃ n : ℕ, points_after_two_procedures n = 82 := by
  sorry

end NUMINAMATH_CALUDE_exists_initial_points_for_82_l434_43451


namespace NUMINAMATH_CALUDE_trajectory_equation_l434_43443

-- Define the property for a point (x, y)
def satisfiesProperty (x y : ℝ) : Prop :=
  2 * (|x| + |y|) = x^2 + y^2

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, satisfiesProperty x y ↔ x^2 + y^2 = 2 * |x| + 2 * |y| :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l434_43443


namespace NUMINAMATH_CALUDE_actual_average_height_l434_43486

/-- The number of boys in the class -/
def num_boys : ℕ := 35

/-- The initial average height in centimeters -/
def initial_avg : ℚ := 182

/-- The incorrectly recorded height in centimeters -/
def incorrect_height : ℚ := 166

/-- The correct height in centimeters -/
def correct_height : ℚ := 106

/-- The actual average height after correction -/
def actual_avg : ℚ := (num_boys * initial_avg - (incorrect_height - correct_height)) / num_boys

theorem actual_average_height :
  ∃ ε > 0, abs (actual_avg - 180.29) < ε :=
sorry

end NUMINAMATH_CALUDE_actual_average_height_l434_43486


namespace NUMINAMATH_CALUDE_train_length_l434_43411

/-- Given a train traveling at 45 km/hr that crosses a 220.03-meter bridge in 30 seconds,
    the length of the train is 154.97 meters. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 220.03 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 154.97 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l434_43411


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three_l434_43424

theorem sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three :
  Real.sqrt 9 - 2^(0 : ℕ) + |(-1 : ℝ)| = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three_l434_43424


namespace NUMINAMATH_CALUDE_bridge_length_l434_43440

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 130 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 245 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l434_43440


namespace NUMINAMATH_CALUDE_solubility_product_scientific_notation_l434_43479

theorem solubility_product_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.0000000028 = a * (10 : ℝ) ^ n :=
by sorry

end NUMINAMATH_CALUDE_solubility_product_scientific_notation_l434_43479


namespace NUMINAMATH_CALUDE_cabbage_production_l434_43410

theorem cabbage_production (last_year_side : ℕ) (this_year_side : ℕ) : 
  (this_year_side : ℤ)^2 - (last_year_side : ℤ)^2 = 127 →
  this_year_side^2 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_cabbage_production_l434_43410
