import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_l2494_249472

theorem square_difference (a b : ℝ) (h1 : a + b = 2) (h2 : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2494_249472


namespace NUMINAMATH_CALUDE_xena_head_start_l2494_249479

/-- Xena's running speed in feet per second -/
def xena_speed : ℝ := 15

/-- Dragon's flying speed in feet per second -/
def dragon_speed : ℝ := 30

/-- Time Xena has to reach the cave in seconds -/
def time_to_cave : ℝ := 32

/-- Minimum safe distance between Xena and the dragon in feet -/
def safe_distance : ℝ := 120

/-- Theorem stating Xena's head start distance -/
theorem xena_head_start : 
  xena_speed * time_to_cave + 360 = dragon_speed * time_to_cave - safe_distance := by
  sorry

end NUMINAMATH_CALUDE_xena_head_start_l2494_249479


namespace NUMINAMATH_CALUDE_f_of_3_equals_9_l2494_249481

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x + 1) + 1

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_9_l2494_249481


namespace NUMINAMATH_CALUDE_macaroon_solution_l2494_249431

/-- Represents the problem of calculating the remaining weight of macaroons --/
def macaroon_problem (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) : Prop :=
  let total_weight := total_macaroons * weight_per_macaroon
  let macaroons_per_bag := total_macaroons / num_bags
  let weight_per_bag := macaroons_per_bag * weight_per_macaroon
  let remaining_bags := num_bags - 1
  let remaining_weight := remaining_bags * weight_per_bag
  remaining_weight = 45

/-- Theorem stating the solution to the macaroon problem --/
theorem macaroon_solution : macaroon_problem 12 5 4 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_solution_l2494_249431


namespace NUMINAMATH_CALUDE_probability_of_asian_card_l2494_249457

-- Define the set of cards
inductive Card : Type
| China : Card
| USA : Card
| UK : Card
| SouthKorea : Card

-- Define a function to check if a card corresponds to an Asian country
def isAsian : Card → Bool
| Card.China => true
| Card.SouthKorea => true
| _ => false

-- Define the total number of cards
def totalCards : ℕ := 4

-- Define the number of Asian countries
def asianCards : ℕ := 2

-- Theorem statement
theorem probability_of_asian_card :
  (asianCards : ℚ) / totalCards = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_asian_card_l2494_249457


namespace NUMINAMATH_CALUDE_person_age_is_54_l2494_249469

/-- Represents the age of a person and their eldest son, satisfying given conditions --/
structure AgeRelation where
  Y : ℕ  -- Current age of the person
  S : ℕ  -- Current age of the eldest son
  age_relation_past : Y - 9 = 5 * (S - 9)  -- Relation 9 years ago
  age_relation_present : Y = 3 * S         -- Current relation

/-- Theorem stating that given the conditions, the person's current age is 54 --/
theorem person_age_is_54 (ar : AgeRelation) : ar.Y = 54 := by
  sorry

end NUMINAMATH_CALUDE_person_age_is_54_l2494_249469


namespace NUMINAMATH_CALUDE_case_cost_l2494_249432

theorem case_cost (pen ink case : ℝ) 
  (total_cost : pen + ink + case = 2.30)
  (pen_cost : pen = ink + 1.50)
  (case_cost : case = 0.5 * ink) :
  case = 0.1335 := by
sorry

end NUMINAMATH_CALUDE_case_cost_l2494_249432


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2494_249477

def set_A : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def set_B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2494_249477


namespace NUMINAMATH_CALUDE_nabla_calculation_l2494_249487

def nabla (a b : ℕ) : ℕ := 3 + (Nat.factorial b) ^ a

theorem nabla_calculation : nabla (nabla 2 3) 4 = 3 + 24 ^ 39 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l2494_249487


namespace NUMINAMATH_CALUDE_parallel_planes_perpendicular_line_l2494_249447

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_perpendicular_line 
  (α β : Plane) (m : Line) 
  (h1 : parallel α β) 
  (h2 : perpendicular m α) : 
  perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_perpendicular_line_l2494_249447


namespace NUMINAMATH_CALUDE_solution_characterization_l2494_249446

def solution_set (a : ℕ) : Set ℝ :=
  {x : ℝ | a * x^2 + 2 * |x - a| - 20 < 0}

def inequality1_set : Set ℝ :=
  {x : ℝ | x^2 + x - 2 < 0}

def inequality2_set : Set ℝ :=
  {x : ℝ | |2*x - 1| < x + 2}

theorem solution_characterization :
  ∀ a : ℕ, (inequality1_set ⊆ solution_set a ∧ inequality2_set ⊆ solution_set a) ↔ 
    a ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_solution_characterization_l2494_249446


namespace NUMINAMATH_CALUDE_store_inventory_l2494_249466

theorem store_inventory (ties belts black_shirts : ℕ) 
  (h1 : ties = 34)
  (h2 : belts = 40)
  (h3 : black_shirts = 63)
  (h4 : ∃ white_shirts : ℕ, 
    ∃ jeans : ℕ, 
    ∃ scarves : ℕ,
    jeans = (2 * (black_shirts + white_shirts)) / 3 ∧
    scarves = (ties + belts) / 2 ∧
    jeans = scarves + 33) :
  ∃ white_shirts : ℕ, white_shirts = 42 := by
sorry

end NUMINAMATH_CALUDE_store_inventory_l2494_249466


namespace NUMINAMATH_CALUDE_angle_bisector_exists_l2494_249456

/-- A ruler with constant width and parallel edges -/
structure ConstantWidthRuler where
  width : ℝ
  width_positive : width > 0

/-- An angle in a plane -/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- A line in a plane -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Predicate to check if a line bisects an angle -/
def bisects (l : Line) (a : Angle) : Prop :=
  sorry

/-- Predicate to check if a line can be constructed using a constant width ruler -/
def constructible_with_ruler (l : Line) (r : ConstantWidthRuler) : Prop :=
  sorry

/-- Theorem stating that for any angle, there exists a bisector constructible with a constant width ruler -/
theorem angle_bisector_exists (a : Angle) (r : ConstantWidthRuler) :
  ∃ l : Line, bisects l a ∧ constructible_with_ruler l r := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_exists_l2494_249456


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l2494_249443

theorem max_value_of_trigonometric_function :
  let y : ℝ → ℝ := λ x => Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)
  ∃ (max_y : ℝ), max_y = 11 / 6 * Real.sqrt 3 ∧
    ∀ x ∈ Set.Icc (-5 * Real.pi / 12) (-Real.pi / 3), y x ≤ max_y :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l2494_249443


namespace NUMINAMATH_CALUDE_smallest_bound_for_cubic_coefficient_smallest_k_is_four_l2494_249418

-- Define the set of polynomials M
def M : Set (ℝ → ℝ) :=
  {P | ∃ (a b c d : ℝ), ∀ x, P x = a * x^3 + b * x^2 + c * x + d ∧ 
                         ∀ x ∈ Set.Icc (-1 : ℝ) 1, |P x| ≤ 1}

-- State the theorem
theorem smallest_bound_for_cubic_coefficient :
  ∃ k, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) → |a| ≤ k) ∧
       (∀ k' < k, ∃ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| > k') :=
by
  -- The proof goes here
  sorry

-- State that the smallest k is 4
theorem smallest_k_is_four :
  ∃! k, (∀ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) → |a| ≤ k) ∧
       (∀ k' < k, ∃ P ∈ M, ∃ a b c d : ℝ, (∀ x, P x = a * x^3 + b * x^2 + c * x + d) ∧ |a| > k') ∧
       k = 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_bound_for_cubic_coefficient_smallest_k_is_four_l2494_249418


namespace NUMINAMATH_CALUDE_librarian_shelves_l2494_249412

/-- The number of books on the top shelf -/
def first_term : ℕ := 3

/-- The difference in the number of books between each consecutive shelf -/
def common_difference : ℕ := 3

/-- The total number of books on all shelves -/
def total_books : ℕ := 225

/-- The number of shelves used by the librarian -/
def num_shelves : ℕ := 15

theorem librarian_shelves :
  ∃ (n : ℕ), n = num_shelves ∧
  n * (2 * first_term + (n - 1) * common_difference) = 2 * total_books :=
sorry

end NUMINAMATH_CALUDE_librarian_shelves_l2494_249412


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l2494_249478

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_expectation_and_variance :
  ∃ (X : BinomialRV), X.n = 10 ∧ X.p = 0.6 ∧ expected_value X = 6 ∧ variance X = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l2494_249478


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2494_249402

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2494_249402


namespace NUMINAMATH_CALUDE_plane_flight_time_l2494_249461

/-- Given a plane flying between two cities, prove that the return trip takes 84 minutes -/
theorem plane_flight_time (d : ℝ) (p : ℝ) (w : ℝ) :
  d > 0 ∧ p > 0 ∧ w > 0 ∧ p > w → -- Positive distance, plane speed, and wind speed
  d / (p - w) = 96 → -- Trip against wind takes 96 minutes
  d / (p + w) = d / p - 6 → -- Return trip is 6 minutes less than in still air
  d / (p + w) = 84 := by
  sorry

end NUMINAMATH_CALUDE_plane_flight_time_l2494_249461


namespace NUMINAMATH_CALUDE_frank_meets_quota_l2494_249462

/-- Represents the sales data for Frank's car sales challenge -/
structure SalesData where
  quota : Nat
  days : Nat
  first3DaysSales : Nat
  next4DaysSales : Nat
  bonusCars : Nat
  remainingInventory : Nat
  oddDaySales : Nat
  evenDaySales : Nat

/-- Calculates the total sales and remaining inventory based on the given sales data -/
def calculateSales (data : SalesData) : (Nat × Nat) :=
  let initialSales := data.first3DaysSales * 3 + data.next4DaysSales * 4 + data.bonusCars
  let remainingDays := data.days - 7
  let oddDays := remainingDays / 2
  let evenDays := remainingDays - oddDays
  let potentialRemainingDaySales := data.oddDaySales * oddDays + data.evenDaySales * evenDays
  let actualRemainingDaySales := min potentialRemainingDaySales data.remainingInventory
  let totalSales := min (initialSales + actualRemainingDaySales) data.quota
  let remainingInventory := data.remainingInventory - (totalSales - initialSales)
  (totalSales, remainingInventory)

/-- Theorem stating that Frank will meet his quota and have 22 cars left in inventory -/
theorem frank_meets_quota (data : SalesData)
  (h1 : data.quota = 50)
  (h2 : data.days = 30)
  (h3 : data.first3DaysSales = 5)
  (h4 : data.next4DaysSales = 3)
  (h5 : data.bonusCars = 5)
  (h6 : data.remainingInventory = 40)
  (h7 : data.oddDaySales = 2)
  (h8 : data.evenDaySales = 3) :
  calculateSales data = (50, 22) := by
  sorry


end NUMINAMATH_CALUDE_frank_meets_quota_l2494_249462


namespace NUMINAMATH_CALUDE_assignment_count_l2494_249498

theorem assignment_count : 
  (∀ n : ℕ, n = 8 → ∀ k : ℕ, k = 4 → (k : ℕ) ^ n = 65536) := by sorry

end NUMINAMATH_CALUDE_assignment_count_l2494_249498


namespace NUMINAMATH_CALUDE_erroneous_product_theorem_l2494_249474

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem erroneous_product_theorem (a b : ℕ) (h1 : is_two_digit a) (h2 : reverse_digits a * b = 180) :
  a * b = 315 ∨ a * b = 810 := by
  sorry

end NUMINAMATH_CALUDE_erroneous_product_theorem_l2494_249474


namespace NUMINAMATH_CALUDE_video_game_price_l2494_249486

def lawn_price : ℕ := 15
def book_price : ℕ := 5
def lawns_mowed : ℕ := 35
def video_games_wanted : ℕ := 5
def books_bought : ℕ := 60

theorem video_game_price :
  (lawn_price * lawns_mowed - book_price * books_bought) / video_games_wanted = 45 := by
  sorry

end NUMINAMATH_CALUDE_video_game_price_l2494_249486


namespace NUMINAMATH_CALUDE_cheryl_prob_correct_l2494_249417

/-- Represents the number of marbles of each color in the box -/
def marbles_per_color : ℕ := 3

/-- Represents the number of colors of marbles in the box -/
def num_colors : ℕ := 4

/-- Represents the total number of marbles in the box -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- Represents the number of marbles each person draws -/
def marbles_drawn : ℕ := 3

/-- Represents the probability of Cheryl getting 3 marbles of the same color,
    given that Claudia did not draw 3 marbles of the same color -/
def cheryl_same_color_prob : ℚ := 55 / 1540

theorem cheryl_prob_correct :
  cheryl_same_color_prob =
    (num_colors - 1) * (Nat.choose total_marbles marbles_drawn) /
    (Nat.choose total_marbles marbles_drawn *
     (Nat.choose (total_marbles - marbles_drawn) marbles_drawn -
      num_colors * 1) * 1) :=
by sorry

end NUMINAMATH_CALUDE_cheryl_prob_correct_l2494_249417


namespace NUMINAMATH_CALUDE_cardinality_of_S_l2494_249435

/-- The number of elements in a set -/
def C (A : Set ℝ) : ℕ := sorry

/-- The operation * defined on sets -/
def star (A B : Set ℝ) : ℕ :=
  if C A ≥ C B then C A - C B else C B - C A

/-- The set B parameterized by a -/
def B (a : ℝ) : Set ℝ :=
  {x : ℝ | (x + a) * (x^3 + a*x^2 + 2*x) = 0}

/-- The set A -/
def A : Set ℝ := {1, 2}

/-- The set S of all possible values of a -/
def S : Set ℝ :=
  {a : ℝ | star A (B a) = 1 ∧ C A = 2}

theorem cardinality_of_S : C S = 3 := by sorry

end NUMINAMATH_CALUDE_cardinality_of_S_l2494_249435


namespace NUMINAMATH_CALUDE_disjoint_triangles_exist_l2494_249453

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- A triangle formed by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Checks if two triangles are disjoint -/
def disjoint (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∧ t1.a ≠ t2.b ∧ t1.a ≠ t2.c ∧
  t1.b ≠ t2.a ∧ t1.b ≠ t2.b ∧ t1.b ≠ t2.c ∧
  t1.c ≠ t2.a ∧ t1.c ≠ t2.b ∧ t1.c ≠ t2.c

/-- The main theorem -/
theorem disjoint_triangles_exist (n : ℕ) (points : Fin (3 * n) → Point) 
  (h : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ triangles : Fin n → Triangle, 
    (∀ i, ∃ j k l, triangles i = ⟨points j, points k, points l⟩) ∧ 
    (∀ i j, i ≠ j → disjoint (triangles i) (triangles j)) :=
  sorry


end NUMINAMATH_CALUDE_disjoint_triangles_exist_l2494_249453


namespace NUMINAMATH_CALUDE_divided_square_plot_area_l2494_249499

/-- Represents a rectangular plot -/
structure RectangularPlot where
  length : ℝ
  width : ℝ

/-- Represents a square plot divided into 8 equal rectangular parts -/
structure DividedSquarePlot where
  part : RectangularPlot
  perimeter : ℝ

/-- The perimeter of a rectangular plot -/
def RectangularPlot.perimeter (r : RectangularPlot) : ℝ :=
  2 * (r.length + r.width)

/-- The area of a square plot -/
def square_area (side : ℝ) : ℝ :=
  side * side

theorem divided_square_plot_area (d : DividedSquarePlot) 
    (h1 : d.part.length = 2 * d.part.width)
    (h2 : d.perimeter = d.part.perimeter) :
    square_area (4 * d.part.width) = (4 * d.perimeter^2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_divided_square_plot_area_l2494_249499


namespace NUMINAMATH_CALUDE_inequality_proof_l2494_249455

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2494_249455


namespace NUMINAMATH_CALUDE_kamal_chemistry_marks_l2494_249484

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  biology : ℕ
  chemistry : ℕ

/-- Calculates the average marks for a student -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.biology + marks.chemistry) / 5

/-- Theorem: Given Kamal's marks and average, his Chemistry marks must be 62 -/
theorem kamal_chemistry_marks :
  ∀ (kamal : StudentMarks),
    kamal.english = 66 →
    kamal.mathematics = 65 →
    kamal.physics = 77 →
    kamal.biology = 75 →
    average kamal = 69 →
    kamal.chemistry = 62 := by
  sorry

end NUMINAMATH_CALUDE_kamal_chemistry_marks_l2494_249484


namespace NUMINAMATH_CALUDE_discarded_number_proof_l2494_249430

theorem discarded_number_proof (numbers : Finset ℕ) (sum : ℕ) (x : ℕ) :
  Finset.card numbers = 50 →
  sum = Finset.sum numbers id →
  sum / 50 = 50 →
  55 ∈ numbers →
  x ∈ numbers →
  x ≠ 55 →
  (sum - 55 - x) / 48 = 50 →
  x = 45 :=
by sorry

end NUMINAMATH_CALUDE_discarded_number_proof_l2494_249430


namespace NUMINAMATH_CALUDE_product_digit_sum_l2494_249468

def digit_sum (n : ℕ) : ℕ := sorry

def repeated_digit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem product_digit_sum (n : ℕ) : 
  n ≥ 1 → digit_sum (5 * repeated_digit 5 n) ≥ 500 ↔ n ≥ 72 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2494_249468


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2494_249401

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → 
    Real.sqrt 2 = Real.sqrt (4^x * 2^y) → 1/x + 2/y ≥ 1/a + 2/b) →
  1/a + 2/b = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2494_249401


namespace NUMINAMATH_CALUDE_acme_cheaper_min_shirts_l2494_249450

def acme_cost (x : ℕ) : ℚ := 50 + 9 * x
def beta_cost (x : ℕ) : ℚ := 25 + 15 * x

theorem acme_cheaper_min_shirts : 
  ∀ n : ℕ, (∀ k : ℕ, k < n → acme_cost k ≥ beta_cost k) ∧ 
           (acme_cost n < beta_cost n) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_acme_cheaper_min_shirts_l2494_249450


namespace NUMINAMATH_CALUDE_units_digit_2137_power_753_l2494_249407

def units_digit (n : ℕ) : ℕ := n % 10

def power_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  units_digit (units_digit base ^ exp)

def cycle_of_7 (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 7
  | 2 => 9
  | 3 => 3
  | _ => 0  -- This case will never be reached

theorem units_digit_2137_power_753 :
  power_units_digit 2137 753 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_2137_power_753_l2494_249407


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l2494_249463

/-- Given a line with equation y - 6 = -2(x - 3), 
    the sum of its x-intercept and y-intercept is 18 -/
theorem line_intercepts_sum : 
  ∀ (x y : ℝ), y - 6 = -2 * (x - 3) → 
  ∃ (x_int y_int : ℝ), 
    (y_int - 6 = -2 * (x_int - 3) ∧ y_int = 0) ∧
    (0 - 6 = -2 * (0 - 3) ∧ y_int = 0) ∧
    x_int + y_int = 18 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l2494_249463


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2494_249408

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 15 →
  a^2 + b^2 = c^2 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2494_249408


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2494_249459

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 400)
  (h2 : rate = 22.5)
  (h3 : time = 2) :
  (principal * rate * time) / 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2494_249459


namespace NUMINAMATH_CALUDE_fruit_bowl_problem_l2494_249416

/-- Represents the number of fruits in a bowl -/
structure FruitBowl where
  apples : ℕ
  pears : ℕ
  bananas : ℕ

/-- Defines the conditions of the fruit bowl problem -/
def validFruitBowl (bowl : FruitBowl) : Prop :=
  bowl.pears = bowl.apples + 2 ∧
  bowl.bananas = bowl.pears + 3 ∧
  bowl.apples + bowl.pears + bowl.bananas = 19

/-- Theorem stating that a valid fruit bowl contains 9 bananas -/
theorem fruit_bowl_problem (bowl : FruitBowl) : 
  validFruitBowl bowl → bowl.bananas = 9 := by
  sorry


end NUMINAMATH_CALUDE_fruit_bowl_problem_l2494_249416


namespace NUMINAMATH_CALUDE_janna_weekly_sleep_l2494_249403

/-- The number of hours Janna sleeps in a week -/
def total_sleep_hours (weekday_sleep : ℕ) (weekend_sleep : ℕ) (weekdays : ℕ) (weekend_days : ℕ) : ℕ :=
  weekday_sleep * weekdays + weekend_sleep * weekend_days

/-- Theorem stating that Janna sleeps 51 hours in a week -/
theorem janna_weekly_sleep :
  total_sleep_hours 7 8 5 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_janna_weekly_sleep_l2494_249403


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2494_249414

def number_list : List ℚ := [-14, 7, 0, -2/3, -5/16]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2494_249414


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2494_249427

theorem quadratic_equation_roots :
  ∃ (r1 r2 : ℝ), r1 ≠ r2 ∧ 
  (∀ x : ℝ, x^2 + x - 1 = 0 ↔ x = r1 ∨ x = r2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2494_249427


namespace NUMINAMATH_CALUDE_pen_profit_calculation_l2494_249493

theorem pen_profit_calculation (total_pens : ℕ) (buy_price sell_price : ℚ) (target_profit : ℚ) :
  total_pens = 2000 →
  buy_price = 15/100 →
  sell_price = 30/100 →
  target_profit = 120 →
  ∃ (sold_pens : ℕ), 
    sold_pens ≤ total_pens ∧ 
    (↑sold_pens * sell_price) - (↑total_pens * buy_price) = target_profit ∧
    sold_pens = 1400 :=
by sorry

end NUMINAMATH_CALUDE_pen_profit_calculation_l2494_249493


namespace NUMINAMATH_CALUDE_ring_with_finite_zero_divisors_is_finite_l2494_249400

/-- A ring with at least one non-zero zero divisor and finitely many zero divisors is finite. -/
theorem ring_with_finite_zero_divisors_is_finite (R : Type*) [Ring R]
  (h1 : ∃ (x y : R), x ≠ 0 ∧ y ≠ 0 ∧ x * y = 0)
  (h2 : Set.Finite {x : R | ∃ y, y ≠ 0 ∧ x * y = 0 ∨ y * x = 0}) :
  Set.Finite (Set.univ : Set R) := by
  sorry

end NUMINAMATH_CALUDE_ring_with_finite_zero_divisors_is_finite_l2494_249400


namespace NUMINAMATH_CALUDE_sandy_clothes_cost_l2494_249411

-- Define the costs of individual items
def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def jacket_cost : ℚ := 7.43

-- Define the total cost
def total_cost : ℚ := shorts_cost + shirt_cost + jacket_cost

-- Theorem statement
theorem sandy_clothes_cost : total_cost = 33.56 := by
  sorry

end NUMINAMATH_CALUDE_sandy_clothes_cost_l2494_249411


namespace NUMINAMATH_CALUDE_cruise_ship_tourists_l2494_249464

theorem cruise_ship_tourists : ∃ (x : ℕ) (tourists : ℕ), 
  x > 1 ∧ 
  tourists = 12 * x + 1 ∧
  ∃ (y : ℕ), y ≤ 15 ∧ tourists = y * (x - 1) ∧
  tourists = 169 := by
  sorry

end NUMINAMATH_CALUDE_cruise_ship_tourists_l2494_249464


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2494_249426

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,7), then m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    -- The midpoint of the original and reflected points lies on the line
    y = m * x + b ∧ 
    x = (2 + 10) / 2 ∧ 
    y = (3 + 7) / 2 ∧
    -- The line is perpendicular to the line segment between the original and reflected points
    m * ((10 - 2) / (7 - 3)) = -1) → 
  m + b = 15 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2494_249426


namespace NUMINAMATH_CALUDE_power_function_through_2_4_l2494_249429

/-- A power function that passes through the point (2, 4) is equivalent to f(x) = x^2 -/
theorem power_function_through_2_4 (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = x^α) →  -- f is a power function with exponent α
  f 2 = 4 →           -- f passes through the point (2, 4)
  ∀ x, f x = x^2 :=   -- f is equivalent to x^2
by sorry

end NUMINAMATH_CALUDE_power_function_through_2_4_l2494_249429


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2494_249433

theorem inequality_system_solution (x : ℝ) : 
  ((-x + 3) / 2 < x) ∧ (2 * (x + 6) ≥ 5 * x) → 1 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2494_249433


namespace NUMINAMATH_CALUDE_fraction_value_given_condition_l2494_249471

theorem fraction_value_given_condition (a b : ℝ) 
  (h : |a + 2| + Real.sqrt (b - 4) = 0) : a^2 / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_given_condition_l2494_249471


namespace NUMINAMATH_CALUDE_special_function_value_at_neg_two_l2494_249475

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 4 * x * y

theorem special_function_value_at_neg_two
  (f : ℝ → ℝ) (h : special_function f) (h1 : f 1 = 2) :
  f (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_at_neg_two_l2494_249475


namespace NUMINAMATH_CALUDE_nancys_savings_in_euros_l2494_249442

/-- Calculates the amount of money Nancy has in euros given her savings and the exchange rate. -/
def nancys_euros_savings (quarters : ℕ) (five_dollar_bills : ℕ) (dimes : ℕ) (exchange_rate : ℚ) : ℚ :=
  let dollars : ℚ := (quarters * (1 / 4) + five_dollar_bills * 5 + dimes * (1 / 10))
  dollars / exchange_rate

/-- Proves that Nancy has €18.21 in euros given her savings and the exchange rate. -/
theorem nancys_savings_in_euros :
  nancys_euros_savings 12 3 24 (112 / 100) = 1821 / 100 := by
  sorry

end NUMINAMATH_CALUDE_nancys_savings_in_euros_l2494_249442


namespace NUMINAMATH_CALUDE_complex_subtraction_l2494_249495

theorem complex_subtraction : (6 : ℂ) + 2*I - (3 - 5*I) = 3 + 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2494_249495


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l2494_249492

/-- 
Given an arithmetic progression with sum of n terms equal to 220,
common difference 3, first term an integer, and n > 1,
prove that the sum of the first 10 terms is 215.
-/
theorem arithmetic_progression_sum (n : ℕ) (a : ℤ) :
  n > 1 →
  (n : ℝ) * (a + (n - 1) * 3 / 2) = 220 →
  10 * (a + (10 - 1) * 3 / 2) = 215 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l2494_249492


namespace NUMINAMATH_CALUDE_dave_apps_left_l2494_249467

/-- The number of apps Dave had left on his phone after adding and deleting apps. -/
def apps_left (initial_apps new_apps : ℕ) : ℕ :=
  initial_apps + new_apps - (new_apps + 1)

/-- Theorem stating that Dave had 14 apps left on his phone. -/
theorem dave_apps_left : apps_left 15 71 = 14 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_left_l2494_249467


namespace NUMINAMATH_CALUDE_best_estimate_on_number_line_l2494_249476

theorem best_estimate_on_number_line (x : ℝ) (h1 : x < 0) (h2 : -2 < x) (h3 : x < -1) :
  let options := [1.3, -1.3, -2.7, 0.7, -0.7]
  (-1.3 : ℝ) = options.argmin (fun y => |x - y|) := by
  sorry

end NUMINAMATH_CALUDE_best_estimate_on_number_line_l2494_249476


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l2494_249473

def grandmother_gift : ℕ := 25
def aunt_uncle_gift : ℕ := 20
def parents_gift : ℕ := 75
def total_money : ℕ := 279

theorem chris_money_before_birthday :
  total_money - (grandmother_gift + aunt_uncle_gift + parents_gift) = 159 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l2494_249473


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l2494_249445

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- Expected value of a random variable -/
def expected_value (ξ : NormalRandomVariable) : ℝ := ξ.μ

/-- Variance of a random variable -/
def variance (ξ : NormalRandomVariable) : ℝ := ξ.σ^2

/-- Probability of a random variable falling within a certain range -/
def probability (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (ξ : NormalRandomVariable) 
  (h1 : expected_value ξ = 3) 
  (h2 : variance ξ = 1) 
  (h3 : probability ξ (ξ.μ - ξ.σ) (ξ.μ + ξ.σ) = 0.683) : 
  probability ξ 2 4 = 0.683 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l2494_249445


namespace NUMINAMATH_CALUDE_second_order_eq_circle_iff_l2494_249425

/-- A general second-order equation in two variables -/
structure SecondOrderEquation where
  a11 : ℝ
  a12 : ℝ
  a22 : ℝ
  a13 : ℝ
  a23 : ℝ
  a33 : ℝ

/-- Predicate to check if a second-order equation represents a circle -/
def IsCircle (eq : SecondOrderEquation) : Prop :=
  eq.a11 = eq.a22 ∧ eq.a12 = 0

/-- Theorem stating the conditions for a second-order equation to represent a circle -/
theorem second_order_eq_circle_iff (eq : SecondOrderEquation) :
  IsCircle eq ↔ ∃ (h k : ℝ) (r : ℝ), r > 0 ∧
    ∀ (x y : ℝ), eq.a11 * x^2 + 2*eq.a12 * x*y + eq.a22 * y^2 + 2*eq.a13 * x + 2*eq.a23 * y + eq.a33 = 0 ↔
    (x - h)^2 + (y - k)^2 = r^2 :=
  sorry


end NUMINAMATH_CALUDE_second_order_eq_circle_iff_l2494_249425


namespace NUMINAMATH_CALUDE_forty_percent_of_number_l2494_249428

theorem forty_percent_of_number (n : ℝ) : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * n = 15 → (40/100 : ℝ) * n = 180 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_number_l2494_249428


namespace NUMINAMATH_CALUDE_facebook_bonus_percentage_l2494_249485

/-- Represents the Facebook employee bonus problem -/
theorem facebook_bonus_percentage (total_employees : ℕ) 
  (annual_earnings : ℝ) (non_mother_women : ℕ) (bonus_per_mother : ℝ) :
  total_employees = 3300 →
  annual_earnings = 5000000 →
  non_mother_women = 1200 →
  bonus_per_mother = 1250 →
  (((total_employees * 2 / 3 - non_mother_women) * bonus_per_mother) / annual_earnings) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_facebook_bonus_percentage_l2494_249485


namespace NUMINAMATH_CALUDE_percentage_of_singles_l2494_249436

def total_hits : ℕ := 45
def home_runs : ℕ := 2
def triples : ℕ := 2
def doubles : ℕ := 8

def non_singles : ℕ := home_runs + triples + doubles
def singles : ℕ := total_hits - non_singles

theorem percentage_of_singles :
  (singles : ℚ) / total_hits * 100 = 73 := by sorry

end NUMINAMATH_CALUDE_percentage_of_singles_l2494_249436


namespace NUMINAMATH_CALUDE_arithmetic_sequence_roots_iff_l2494_249440

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate for a cubic equation having three real roots in arithmetic sequence -/
def has_arithmetic_sequence_roots (eq : CubicEquation) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
    x₃ - x₂ = x₂ - x₁ ∧
    x₁^3 + eq.a * x₁^2 + eq.b * x₁ + eq.c = 0 ∧
    x₂^3 + eq.a * x₂^2 + eq.b * x₂ + eq.c = 0 ∧
    x₃^3 + eq.a * x₃^2 + eq.b * x₃ + eq.c = 0

/-- The necessary and sufficient conditions for a cubic equation to have three real roots in arithmetic sequence -/
theorem arithmetic_sequence_roots_iff (eq : CubicEquation) :
  has_arithmetic_sequence_roots eq ↔ 
  (2 * eq.a^3 - 9 * eq.a * eq.b + 27 * eq.c = 0) ∧ (eq.a^2 - 3 * eq.b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_roots_iff_l2494_249440


namespace NUMINAMATH_CALUDE_vector_perpendicular_to_sum_l2494_249405

/-- Given vectors a and b in ℝ², prove that a is perpendicular to (a + b) -/
theorem vector_perpendicular_to_sum (a b : ℝ × ℝ) (ha : a = (2, -1)) (hb : b = (1, 7)) :
  a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0 := by
  sorry

#check vector_perpendicular_to_sum

end NUMINAMATH_CALUDE_vector_perpendicular_to_sum_l2494_249405


namespace NUMINAMATH_CALUDE_tan_105_degrees_l2494_249421

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l2494_249421


namespace NUMINAMATH_CALUDE_evaluate_g_l2494_249483

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem evaluate_g : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l2494_249483


namespace NUMINAMATH_CALUDE_patsy_appetizers_needed_l2494_249451

def appetizers_per_guest : ℕ := 6
def number_of_guests : ℕ := 30
def deviled_eggs_dozens : ℕ := 3
def pigs_in_blanket_dozens : ℕ := 2
def kebabs_dozens : ℕ := 2
def items_per_dozen : ℕ := 12

theorem patsy_appetizers_needed : 
  (appetizers_per_guest * number_of_guests - 
   (deviled_eggs_dozens + pigs_in_blanket_dozens + kebabs_dozens) * items_per_dozen) / items_per_dozen = 8 := by
  sorry

end NUMINAMATH_CALUDE_patsy_appetizers_needed_l2494_249451


namespace NUMINAMATH_CALUDE_least_integer_with_leading_six_and_fraction_l2494_249465

theorem least_integer_with_leading_six_and_fraction (x : ℕ) : x ≥ 625 →
  (∃ n : ℕ, ∃ y : ℕ, 
    x = 6 * 10^n + y ∧ 
    y < 10^n ∧ 
    y = x / 25) →
  x = 625 :=
sorry

end NUMINAMATH_CALUDE_least_integer_with_leading_six_and_fraction_l2494_249465


namespace NUMINAMATH_CALUDE_distribute_five_identical_books_to_three_students_l2494_249488

/-- The number of ways to distribute n identical objects to k recipients, 
    where each recipient receives exactly one object. -/
def distribute_identical (n k : ℕ) : ℕ :=
  if n = k then 1 else 0

/-- Theorem: There is only one way to distribute 5 identical books to 3 students, 
    with each student receiving one book. -/
theorem distribute_five_identical_books_to_three_students :
  distribute_identical 5 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_identical_books_to_three_students_l2494_249488


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l2494_249437

theorem add_preserves_inequality (a b c : ℝ) (h : a > b) : a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l2494_249437


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2494_249482

/-- Given a simple interest loan where:
    - The interest after 10 years is 1500
    - The principal amount is 1250
    Prove that the interest rate is 12% --/
theorem simple_interest_rate : 
  ∀ (rate : ℝ),
  (1250 * rate * 10 / 100 = 1500) →
  rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2494_249482


namespace NUMINAMATH_CALUDE_smallest_b_value_l2494_249449

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 6) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 9) :
  b.val ≥ 3 ∧ ∃ (a' b' : ℕ+), b'.val = 3 ∧ a'.val - b'.val = 6 ∧ 
    Nat.gcd ((a'.val^3 + b'.val^3) / (a'.val + b'.val)) (a'.val * b'.val) = 9 :=
by sorry


end NUMINAMATH_CALUDE_smallest_b_value_l2494_249449


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2494_249413

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if one of its asymptotes is y = 3x/5, then a = 5. -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = 3 * x / 5) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2494_249413


namespace NUMINAMATH_CALUDE_complement_of_A_l2494_249460

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 2}

-- Statement to prove
theorem complement_of_A (x : Nat) : x ∈ (U \ A) ↔ x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2494_249460


namespace NUMINAMATH_CALUDE_average_speeding_percentage_l2494_249452

def zone_a_speeding_percentage : ℝ := 30
def zone_b_speeding_percentage : ℝ := 20
def zone_c_speeding_percentage : ℝ := 25

def number_of_zones : ℕ := 3

theorem average_speeding_percentage :
  (zone_a_speeding_percentage + zone_b_speeding_percentage + zone_c_speeding_percentage) / number_of_zones = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speeding_percentage_l2494_249452


namespace NUMINAMATH_CALUDE_total_vehicles_proof_l2494_249448

/-- The number of vehicles involved in accidents last year -/
def accidents : ℕ := 2000

/-- The number of vehicles per 100 million that are involved in accidents -/
def accident_rate : ℕ := 100

/-- The total number of vehicles that traveled on the highway last year -/
def total_vehicles : ℕ := 2000000000

/-- Theorem stating that the total number of vehicles is correct given the accident rate and number of accidents -/
theorem total_vehicles_proof :
  accidents * (100000000 / accident_rate) = total_vehicles :=
sorry

end NUMINAMATH_CALUDE_total_vehicles_proof_l2494_249448


namespace NUMINAMATH_CALUDE_water_speed_l2494_249470

/-- The speed of water given a swimmer's speed in still water and time taken to swim against the current -/
theorem water_speed (swim_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : swim_speed = 6)
  (h2 : distance = 14) (h3 : time = 3.5) :
  ∃ (water_speed : ℝ), water_speed = 2 ∧ distance = (swim_speed - water_speed) * time := by
  sorry

end NUMINAMATH_CALUDE_water_speed_l2494_249470


namespace NUMINAMATH_CALUDE_connie_blue_markers_l2494_249404

/-- Given that Connie has 2315 red markers and 3343 markers in total, 
    prove that she has 1028 blue markers. -/
theorem connie_blue_markers 
  (total_markers : ℕ) 
  (red_markers : ℕ) 
  (h1 : total_markers = 3343)
  (h2 : red_markers = 2315) :
  total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_CALUDE_connie_blue_markers_l2494_249404


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l2494_249489

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 + 2 * Real.sqrt 2 * m + 1 = 0) ∧ 
  (n^2 + 2 * Real.sqrt 2 * n + 1 = 0) → 
  Real.sqrt (m^2 + n^2 + 3*m*n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l2494_249489


namespace NUMINAMATH_CALUDE_unique_solution_2014_l2494_249496

theorem unique_solution_2014 (x : ℝ) (h : x > 0) :
  (x * 2014^(1/x) + (1/x) * 2014^x) / 2 = 2014 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_2014_l2494_249496


namespace NUMINAMATH_CALUDE_train_length_problem_l2494_249444

theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 46) (h2 : slower_speed = 36) 
  (h3 : passing_time = 18) : ∃ (train_length : ℝ), 
  train_length = 50 ∧ 
  train_length * 1000 = (faster_speed - slower_speed) * passing_time / 3600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l2494_249444


namespace NUMINAMATH_CALUDE_caterpillar_length_difference_l2494_249409

theorem caterpillar_length_difference :
  let green_length : ℝ := 3
  let orange_length : ℝ := 1.17
  green_length - orange_length = 1.83 := by
sorry

end NUMINAMATH_CALUDE_caterpillar_length_difference_l2494_249409


namespace NUMINAMATH_CALUDE_intersection_range_l2494_249419

/-- Hyperbola C centered at the origin with right focus at (2,0) and real axis length 2√3 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- Line l with equation y = kx + √2 -/
def line_l (k x : ℝ) (y : ℝ) : Prop := y = k * x + Real.sqrt 2

/-- Predicate to check if a point (x, y) is on the left branch of hyperbola C -/
def on_left_branch (x y : ℝ) : Prop := hyperbola_C x y ∧ x < 0

/-- Theorem stating the range of k for which line l intersects the left branch of hyperbola C at two points -/
theorem intersection_range (k : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    on_left_branch x₁ y₁ ∧ 
    on_left_branch x₂ y₂ ∧ 
    line_l k x₁ y₁ ∧ 
    line_l k x₂ y₂) ↔ 
  Real.sqrt 3 / 3 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l2494_249419


namespace NUMINAMATH_CALUDE_dart_score_is_75_l2494_249491

/-- The final score of three dart throws -/
def final_score (bullseye : ℕ) (half_bullseye : ℕ) (miss : ℕ) : ℕ :=
  bullseye + half_bullseye + miss

/-- Theorem stating that the final score is 75 points -/
theorem dart_score_is_75 :
  ∃ (bullseye half_bullseye miss : ℕ),
    bullseye = 50 ∧
    half_bullseye = bullseye / 2 ∧
    miss = 0 ∧
    final_score bullseye half_bullseye miss = 75 := by
  sorry

end NUMINAMATH_CALUDE_dart_score_is_75_l2494_249491


namespace NUMINAMATH_CALUDE_symmetry_points_l2494_249406

/-- Given points M, N, P, and Q in a 2D plane, prove that Q has coordinates (b,a) -/
theorem symmetry_points (a b : ℝ) : 
  let M : ℝ × ℝ := (a, b)
  let N : ℝ × ℝ := (a, -b)  -- M symmetric to N w.r.t. x-axis
  let P : ℝ × ℝ := (-a, -b) -- P symmetric to N w.r.t. y-axis
  let Q : ℝ × ℝ := (b, a)   -- Q symmetric to P w.r.t. line x+y=0
  Q = (b, a) := by sorry

end NUMINAMATH_CALUDE_symmetry_points_l2494_249406


namespace NUMINAMATH_CALUDE_min_time_for_all_flickers_l2494_249480

/-- The number of colored lights -/
def num_lights : ℕ := 5

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The time taken for one flicker (in seconds) -/
def flicker_time : ℕ := 5

/-- The interval time between flickers (in seconds) -/
def interval_time : ℕ := 5

/-- The total number of possible flickers -/
def total_flickers : ℕ := Nat.factorial num_lights

theorem min_time_for_all_flickers :
  (total_flickers * flicker_time) + ((total_flickers - 1) * interval_time) = 1195 := by
  sorry

end NUMINAMATH_CALUDE_min_time_for_all_flickers_l2494_249480


namespace NUMINAMATH_CALUDE_no_four_tangents_for_different_radii_l2494_249410

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles --/
def commonTangents (c1 c2 : Circle) : ℕ := sorry

/-- Two circles with different radii cannot have exactly 4 common tangents --/
theorem no_four_tangents_for_different_radii (c1 c2 : Circle) 
  (h : c1.radius ≠ c2.radius) : commonTangents c1 c2 ≠ 4 := by sorry

end NUMINAMATH_CALUDE_no_four_tangents_for_different_radii_l2494_249410


namespace NUMINAMATH_CALUDE_max_product_roots_quadratic_l2494_249439

/-- Given a quadratic equation 6x^2 - 12x + m = 0 with real roots,
    the maximum value of m that maximizes the product of the roots is 6. -/
theorem max_product_roots_quadratic :
  ∀ m : ℝ,
  (∃ x y : ℝ, 6 * x^2 - 12 * x + m = 0 ∧ 6 * y^2 - 12 * y + m = 0 ∧ x ≠ y) →
  (∀ k : ℝ, (∃ x y : ℝ, 6 * x^2 - 12 * x + k = 0 ∧ 6 * y^2 - 12 * y + k = 0 ∧ x ≠ y) →
    m / 6 ≥ k / 6) →
  m = 6 :=
by sorry

end NUMINAMATH_CALUDE_max_product_roots_quadratic_l2494_249439


namespace NUMINAMATH_CALUDE_problem_solution_l2494_249415

theorem problem_solution (a b x : ℝ) 
  (h1 : a * (x + 2) + b * (x + 2) = 60) 
  (h2 : a + b = 12) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2494_249415


namespace NUMINAMATH_CALUDE_triangle_coordinates_l2494_249420

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  P : Point
  Q : Point
  R : Point

/-- Predicate to check if a line segment is horizontal -/
def isHorizontal (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Predicate to check if a line segment is vertical -/
def isVertical (p1 p2 : Point) : Prop :=
  p1.x = p2.x

theorem triangle_coordinates (t : Triangle) 
  (h1 : isHorizontal t.P t.R)
  (h2 : isVertical t.P t.Q)
  (h3 : t.R.y = -2)
  (h4 : t.Q.x = -11) :
  t.P.x = -11 ∧ t.P.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_coordinates_l2494_249420


namespace NUMINAMATH_CALUDE_two_digit_product_equals_concatenation_l2494_249497

def has_same_digits (a b : ℕ) : Prop :=
  (Nat.log 10 a).succ = (Nat.log 10 b).succ

def concatenate (a b : ℕ) : ℕ :=
  a * (10 ^ ((Nat.log 10 b).succ)) + b

theorem two_digit_product_equals_concatenation :
  ∀ A B : ℕ,
    A > 0 ∧ B > 0 →
    has_same_digits A B →
    2 * A * B = concatenate A B →
    (A = 3 ∧ B = 6) ∨ (A = 13 ∧ B = 52) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_equals_concatenation_l2494_249497


namespace NUMINAMATH_CALUDE_complex_symmetry_l2494_249424

theorem complex_symmetry (z₁ z₂ : ℂ) : 
  (z₁ = 2 - 3*I) → (z₁ = -z₂) → (z₂ = -2 + 3*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_l2494_249424


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2494_249494

theorem trigonometric_identities :
  (((Real.tan (10 * π / 180)) * (Real.tan (70 * π / 180))) /
   ((Real.tan (70 * π / 180)) - (Real.tan (10 * π / 180)) + (Real.tan (120 * π / 180))) = Real.sqrt 3 / 3) ∧
  ((2 * (Real.cos (40 * π / 180)) + (Real.cos (10 * π / 180)) * (1 + Real.sqrt 3 * (Real.tan (10 * π / 180)))) /
   (Real.sqrt (1 + Real.cos (10 * π / 180))) = 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2494_249494


namespace NUMINAMATH_CALUDE_even_function_shift_l2494_249454

/-- Given a function f and a real number a, proves that if f(x+a) is even and a is in (0,π/2), then a = 5π/12 -/
theorem even_function_shift (f : ℝ → ℝ) (a : ℝ) : 
  (f = λ x => 3 * Real.sin (2 * x - π/3)) →
  (∀ x, f (x + a) = f (-x - a)) →
  (0 < a) →
  (a < π/2) →
  a = 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_even_function_shift_l2494_249454


namespace NUMINAMATH_CALUDE_base_n_representation_l2494_249441

theorem base_n_representation (n : ℕ) : 
  n > 0 ∧ 
  (∃ a b c : ℕ, 
    a < n ∧ b < n ∧ c < n ∧ 
    1998 = a * n^2 + b * n + c ∧ 
    a + b + c = 24) → 
  n = 15 ∨ n = 22 ∨ n = 43 := by
sorry

end NUMINAMATH_CALUDE_base_n_representation_l2494_249441


namespace NUMINAMATH_CALUDE_decimal_binary_equality_l2494_249490

-- Define a function to convert decimal to binary
def decimalToBinary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec toBinary (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

-- Define a function to convert binary to decimal
def binaryToDecimal (bits : List Nat) : Nat :=
  bits.foldl (fun acc bit => 2 * acc + bit) 0

-- Theorem statement
theorem decimal_binary_equality :
  (decimalToBinary 25 ≠ [1, 0, 1, 1, 0]) ∧
  (decimalToBinary 13 = [1, 1, 0, 1]) ∧
  (decimalToBinary 11 ≠ [1, 1, 0, 0]) ∧
  (decimalToBinary 10 ≠ [1, 0]) :=
by sorry

end NUMINAMATH_CALUDE_decimal_binary_equality_l2494_249490


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2494_249423

theorem line_circle_intersection (k : ℝ) : 
  (∃ x y : ℝ, x - y + k = 0 ∧ x^2 + y^2 = 1) ↔ 
  (k = 1 → ∃ x y : ℝ, x - y + k = 0 ∧ x^2 + y^2 = 1) ∧ 
  (∃ k' : ℝ, k' ≠ 1 ∧ ∃ x y : ℝ, x - y + k' = 0 ∧ x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2494_249423


namespace NUMINAMATH_CALUDE_spiral_notebook_cost_l2494_249434

theorem spiral_notebook_cost :
  let personal_planner_cost : ℝ := 10
  let discount_rate : ℝ := 0.2
  let total_cost_with_discount : ℝ := 112
  let spiral_notebook_cost : ℝ := 15
  (1 - discount_rate) * (4 * spiral_notebook_cost + 8 * personal_planner_cost) = total_cost_with_discount :=
by sorry

end NUMINAMATH_CALUDE_spiral_notebook_cost_l2494_249434


namespace NUMINAMATH_CALUDE_largest_three_digit_square_cube_l2494_249422

/-- The largest three-digit number that is both a perfect square and a perfect cube -/
def largest_square_cube : ℕ := 729

/-- A number is a three-digit number if it's between 100 and 999 inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number is a perfect square if there exists an integer whose square is that number -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A number is a perfect cube if there exists an integer whose cube is that number -/
def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n

theorem largest_three_digit_square_cube :
  is_three_digit largest_square_cube ∧
  is_perfect_square largest_square_cube ∧
  is_perfect_cube largest_square_cube ∧
  ∀ n : ℕ, is_three_digit n → is_perfect_square n → is_perfect_cube n → n ≤ largest_square_cube :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_square_cube_l2494_249422


namespace NUMINAMATH_CALUDE_max_digits_distinct_divisible_l2494_249438

/-- A function that checks if all digits in a natural number are different -/
def hasDistinctDigits (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is divisible by all of its digits -/
def isDivisibleByAllDigits (n : ℕ) : Prop := sorry

/-- A function that returns the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the maximum number of digits in a natural number
    with distinct digits and divisible by all its digits is 7 -/
theorem max_digits_distinct_divisible :
  ∃ (n : ℕ), hasDistinctDigits n ∧ isDivisibleByAllDigits n ∧ numDigits n = 7 ∧
  ∀ (m : ℕ), hasDistinctDigits m → isDivisibleByAllDigits m → numDigits m ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_digits_distinct_divisible_l2494_249438


namespace NUMINAMATH_CALUDE_yellow_beads_count_l2494_249458

theorem yellow_beads_count (blue_beads : ℕ) (total_parts : ℕ) (removed_per_part : ℕ) (final_per_part : ℕ) : 
  blue_beads = 23 →
  total_parts = 3 →
  removed_per_part = 10 →
  final_per_part = 6 →
  (∃ (yellow_beads : ℕ),
    let total_beads := blue_beads + yellow_beads
    let remaining_per_part := (total_beads / total_parts) - removed_per_part
    2 * remaining_per_part = final_per_part ∧
    yellow_beads = 16) :=
by
  sorry

#check yellow_beads_count

end NUMINAMATH_CALUDE_yellow_beads_count_l2494_249458
