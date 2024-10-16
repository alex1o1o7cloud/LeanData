import Mathlib

namespace NUMINAMATH_CALUDE_non_right_triangle_l1354_135460

theorem non_right_triangle : 
  let triangle_sets : List (ℝ × ℝ × ℝ) := 
    [(6, 8, 10), (1, Real.sqrt 3, 2), (5/4, 1, 3/4), (4, 5, 7)]
  ∀ (a b c : ℝ), (a, b, c) ∈ triangle_sets →
    (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ↔ (a, b, c) ≠ (4, 5, 7) :=
by sorry

end NUMINAMATH_CALUDE_non_right_triangle_l1354_135460


namespace NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l1354_135479

theorem smallest_denominator_between_fractions :
  ∃ (p q : ℕ), 
    q = 4027 ∧ 
    (1 : ℚ) / 2014 < (p : ℚ) / q ∧ 
    (p : ℚ) / q < (1 : ℚ) / 2013 ∧
    (∀ (p' q' : ℕ), 
      (1 : ℚ) / 2014 < (p' : ℚ) / q' ∧ 
      (p' : ℚ) / q' < (1 : ℚ) / 2013 → 
      q ≤ q') :=
by sorry

end NUMINAMATH_CALUDE_smallest_denominator_between_fractions_l1354_135479


namespace NUMINAMATH_CALUDE_correct_propositions_l1354_135421

theorem correct_propositions :
  -- Proposition 1
  (∀ P : Prop, (¬P ↔ P) → ¬P) ∧
  -- Proposition 2 (negation)
  ¬(∀ a : ℕ → ℝ, a 0 = 2 ∧ (∀ n : ℕ, a (n + 1) = a n + (a 2 - a 0) / 2) ∧
    (∃ q : ℝ, a 2 = a 0 * q ∧ a 3 = a 2 * q) →
    a 1 - a 0 = -1/2) ∧
  -- Proposition 3
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 →
    (2/a + 3/b ≥ 5 + 2 * Real.sqrt 6)) ∧
  -- Proposition 4 (negation)
  ¬(∀ A B C : ℝ, 0 ≤ A ∧ A ≤ π ∧ 0 ≤ B ∧ B ≤ π ∧ 0 ≤ C ∧ C ≤ π ∧ A + B + C = π →
    (Real.sin A)^2 < (Real.sin B)^2 + (Real.sin C)^2 →
    A < π/2 ∧ B < π/2 ∧ C < π/2) :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l1354_135421


namespace NUMINAMATH_CALUDE_curve_symmetry_l1354_135411

/-- The curve equation -/
def curve (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

/-- The line equation -/
def line (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

/-- Two points are symmetric with respect to a line -/
def symmetric (P Q : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (R : ℝ × ℝ), line m R.1 R.2 ∧ 
    R.1 = (P.1 + Q.1) / 2 ∧ 
    R.2 = (P.2 + Q.2) / 2

theorem curve_symmetry (m : ℝ) :
  (∃ (P Q : ℝ × ℝ), curve P.1 P.2 ∧ curve Q.1 Q.2 ∧ symmetric P Q m) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetry_l1354_135411


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1354_135471

-- Define propositions p and q
def p (x : ℝ) : Prop := 1 < x ∧ x < 2
def q (x : ℝ) : Prop := x > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1354_135471


namespace NUMINAMATH_CALUDE_servant_salary_l1354_135402

/-- Calculates the money received by a servant, excluding the turban -/
theorem servant_salary (annual_salary : ℝ) (turban_price : ℝ) (months_worked : ℝ) : 
  annual_salary = 90 →
  turban_price = 10 →
  months_worked = 9 →
  (months_worked / 12) * (annual_salary + turban_price) - turban_price = 65 :=
by sorry

end NUMINAMATH_CALUDE_servant_salary_l1354_135402


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1354_135439

theorem quadratic_factorization (x : ℝ) : 2 * x^2 + 12 * x + 18 = 2 * (x + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1354_135439


namespace NUMINAMATH_CALUDE_intersection_conditions_l1354_135441

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_conditions (a : ℝ) :
  (9 ∈ A a ∩ B a ↔ a = 5 ∨ a = -3) ∧
  ({9} = A a ∩ B a ↔ a = -3) :=
sorry

end NUMINAMATH_CALUDE_intersection_conditions_l1354_135441


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1354_135480

theorem solve_exponential_equation :
  ∃ x : ℝ, 2^(2*x - 1) = (1/4 : ℝ) ∧ x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1354_135480


namespace NUMINAMATH_CALUDE_trajectory_of_M_l1354_135470

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of the perpendicular bisector of AQ and CQ
def point_M (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), point_Q qx qy ∧
  (x - 1)^2 + y^2 = (x - qx)^2 + (y - qy)^2 ∧
  (x + qx = 1) ∧ (y + qy = 0)

-- Theorem statement
theorem trajectory_of_M :
  ∀ (x y : ℝ), point_M x y → x^2/4 + y^2/3 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l1354_135470


namespace NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_5_l1354_135440

def is_abundant (n : ℕ) : Prop :=
  (Finset.sum (Finset.range n) (λ i => if n % (i + 1) = 0 then i + 1 else 0)) > n

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem smallest_abundant_not_multiple_of_5 : 
  (∀ m : ℕ, m < 12 → (¬is_abundant m ∨ is_multiple_of_5 m)) ∧
  is_abundant 12 ∧ 
  ¬is_multiple_of_5 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_abundant_not_multiple_of_5_l1354_135440


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1354_135482

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1354_135482


namespace NUMINAMATH_CALUDE_ln_x_over_x_decreasing_l1354_135437

theorem ln_x_over_x_decreasing (a b c : ℝ) : 
  a = (Real.log 3) / 3 → 
  b = (Real.log 5) / 5 → 
  c = (Real.log 6) / 6 → 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ln_x_over_x_decreasing_l1354_135437


namespace NUMINAMATH_CALUDE_trapezoid_existence_l1354_135409

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A set of marked vertices in a polygon -/
def MarkedVertices (n : ℕ) (m : ℕ) := Fin m → Fin n

/-- Four points form a trapezoid if two sides are parallel and not all four points are collinear -/
def IsTrapezoid (a b c d : ℝ × ℝ) : Prop := sorry

theorem trapezoid_existence (polygon : RegularPolygon 2015) (marked : MarkedVertices 2015 64) :
  ∃ (a b c d : Fin 64), IsTrapezoid (polygon.vertices (marked a)) (polygon.vertices (marked b)) 
                                    (polygon.vertices (marked c)) (polygon.vertices (marked d)) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_existence_l1354_135409


namespace NUMINAMATH_CALUDE_study_tour_problem_l1354_135463

/-- Represents a bus type with seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the number of buses needed for a given number of participants and bus type -/
def busesNeeded (participants : ℕ) (busType : BusType) : ℕ :=
  (participants + busType.seats - 1) / busType.seats

/-- Calculates the total rental cost for a given number of participants and bus type -/
def rentalCost (participants : ℕ) (busType : BusType) : ℕ :=
  (busesNeeded participants busType) * busType.fee

theorem study_tour_problem (x y : ℕ) (typeA typeB : BusType)
    (h1 : 45 * y + 15 = x)
    (h2 : 60 * (y - 3) = x)
    (h3 : typeA.seats = 45)
    (h4 : typeA.fee = 200)
    (h5 : typeB.seats = 60)
    (h6 : typeB.fee = 300) :
    x = 600 ∧ y = 13 ∧ rentalCost x typeA < rentalCost x typeB := by
  sorry

end NUMINAMATH_CALUDE_study_tour_problem_l1354_135463


namespace NUMINAMATH_CALUDE_smallest_part_of_three_way_division_l1354_135473

theorem smallest_part_of_three_way_division (total : ℕ) (a b c : ℕ) : 
  total = 2340 →
  a + b + c = total →
  ∃ (x : ℕ), a = 5 * x ∧ b = 7 * x ∧ c = 11 * x →
  min a (min b c) = 510 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_three_way_division_l1354_135473


namespace NUMINAMATH_CALUDE_middle_legs_arrangements_adjacent_legs_arrangements_l1354_135486

/-- The number of athletes -/
def total_athletes : ℕ := 6

/-- The number of athletes needed for the relay -/
def relay_size : ℕ := 4

/-- The number of ways to arrange n items taken r at a time -/
def permutations (n r : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - r))

/-- The number of ways to choose r items from n items -/
def combinations (n r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

/-- Theorem for the number of arrangements with A and B running the middle two legs -/
theorem middle_legs_arrangements : 
  permutations 2 2 * permutations (total_athletes - 2) (relay_size - 2) = 24 := by sorry

/-- Theorem for the number of arrangements with A and B running adjacent legs -/
theorem adjacent_legs_arrangements : 
  permutations 2 2 * combinations (total_athletes - 2) (relay_size - 2) * permutations 3 3 = 72 := by sorry

end NUMINAMATH_CALUDE_middle_legs_arrangements_adjacent_legs_arrangements_l1354_135486


namespace NUMINAMATH_CALUDE_supermarket_profit_analysis_l1354_135405

/-- Represents a supermarket area with its operating income and net profit percentages -/
structure Area where
  name : String
  operatingIncomePercentage : Float
  netProfitPercentage : Float

/-- Calculates the operating profit rate for an area given the total operating profit rate -/
def calculateOperatingProfitRate (area : Area) (totalOperatingProfitRate : Float) : Float :=
  (area.netProfitPercentage / area.operatingIncomePercentage) * totalOperatingProfitRate

theorem supermarket_profit_analysis 
  (freshArea dailyNecessitiesArea deliArea dairyArea otherArea : Area)
  (totalOperatingProfitRate : Float) :
  freshArea.name = "Fresh Area" →
  freshArea.operatingIncomePercentage = 48.6 →
  freshArea.netProfitPercentage = 65.8 →
  dailyNecessitiesArea.name = "Daily Necessities Area" →
  dailyNecessitiesArea.operatingIncomePercentage = 10.8 →
  dailyNecessitiesArea.netProfitPercentage = 20.2 →
  deliArea.name = "Deli Area" →
  deliArea.operatingIncomePercentage = 15.8 →
  deliArea.netProfitPercentage = -4.3 →
  dairyArea.name = "Dairy Area" →
  dairyArea.operatingIncomePercentage = 20.1 →
  dairyArea.netProfitPercentage = 16.5 →
  otherArea.name = "Other Area" →
  otherArea.operatingIncomePercentage = 4.7 →
  otherArea.netProfitPercentage = 1.8 →
  totalOperatingProfitRate = 32.5 →
  (freshArea.netProfitPercentage > 50) ∧ 
  (calculateOperatingProfitRate dailyNecessitiesArea totalOperatingProfitRate > 
   max (calculateOperatingProfitRate freshArea totalOperatingProfitRate)
       (max (calculateOperatingProfitRate deliArea totalOperatingProfitRate)
            (max (calculateOperatingProfitRate dairyArea totalOperatingProfitRate)
                 (calculateOperatingProfitRate otherArea totalOperatingProfitRate)))) ∧
  (calculateOperatingProfitRate freshArea totalOperatingProfitRate > 40) := by
  sorry

end NUMINAMATH_CALUDE_supermarket_profit_analysis_l1354_135405


namespace NUMINAMATH_CALUDE_zeros_in_square_expansion_l1354_135413

theorem zeros_in_square_expansion (n : ℕ) : 
  (∃ k : ℕ, (10^15 - 3)^2 = k * 10^n ∧ k % 10 ≠ 0) → n = 29 :=
sorry

end NUMINAMATH_CALUDE_zeros_in_square_expansion_l1354_135413


namespace NUMINAMATH_CALUDE_luke_new_cards_l1354_135487

/-- The number of new baseball cards Luke had --/
def new_cards (cards_per_page old_cards total_pages : ℕ) : ℕ :=
  cards_per_page * total_pages - old_cards

/-- Theorem stating that Luke had 3 new cards --/
theorem luke_new_cards : new_cards 3 9 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_luke_new_cards_l1354_135487


namespace NUMINAMATH_CALUDE_discount_percentage_l1354_135490

def ticket_price : ℝ := 25
def sale_price : ℝ := 18.75

theorem discount_percentage : 
  (ticket_price - sale_price) / ticket_price * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l1354_135490


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l1354_135438

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem sixth_term_of_arithmetic_sequence :
  let a₁ := 2
  let d := 3
  arithmetic_sequence a₁ d 6 = 17 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l1354_135438


namespace NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l1354_135494

/-- Given a rectangle with opposite vertices at (2,-3) and (14,9),
    the point where the diagonals intersect has coordinates (8, 3) -/
theorem rectangle_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -3)
  let v2 : ℝ × ℝ := (14, 9)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_intersection_l1354_135494


namespace NUMINAMATH_CALUDE_sequence_inequality_l1354_135472

/-- The sequence a_n defined by n^2 + kn + 2 -/
def a (n : ℕ) (k : ℝ) : ℝ := n^2 + k * n + 2

/-- Theorem stating that if a_n ≥ a_4 for all n ≥ 4, then k is in [-9, -7] -/
theorem sequence_inequality (k : ℝ) :
  (∀ n : ℕ, n ≥ 4 → a n k ≥ a 4 k) →
  k ∈ Set.Icc (-9 : ℝ) (-7 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1354_135472


namespace NUMINAMATH_CALUDE_min_production_time_l1354_135457

/-- Represents the production process of ceramic items -/
structure CeramicProduction where
  shapingTime : ℕ := 15
  dryingTime : ℕ := 10
  firingTime : ℕ := 30
  totalItems : ℕ := 75
  totalWorkers : ℕ := 13

/-- Calculates the production time for a given stage -/
def stageTime (itemsPerWorker : ℕ) (timePerItem : ℕ) : ℕ :=
  (itemsPerWorker + 1) * timePerItem

/-- Theorem stating the minimum production time for ceramic items -/
theorem min_production_time (prod : CeramicProduction) :
  ∃ (shapers firers : ℕ),
    shapers + firers = prod.totalWorkers ∧
    (∀ (s f : ℕ),
      s + f = prod.totalWorkers →
      max (stageTime (prod.totalItems / s) prod.shapingTime)
          (stageTime (prod.totalItems / f) prod.firingTime)
      ≥ 325) :=
sorry

end NUMINAMATH_CALUDE_min_production_time_l1354_135457


namespace NUMINAMATH_CALUDE_product_bounds_l1354_135474

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ 1/4 + Real.sqrt 3/8 := by
  sorry

end NUMINAMATH_CALUDE_product_bounds_l1354_135474


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l1354_135416

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 1155) :
  ∃ (m : ℕ+), (∀ (n : ℕ+), Nat.gcd q r ≥ n → n ≤ m) ∧ Nat.gcd q r ≥ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l1354_135416


namespace NUMINAMATH_CALUDE_unique_q_value_l1354_135491

theorem unique_q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p*q = 4) : q = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_q_value_l1354_135491


namespace NUMINAMATH_CALUDE_count_nines_to_800_l1354_135410

/-- Count of digit 9 occurrences in integers from 1 to n -/
def count_nines (n : ℕ) : ℕ := sorry

/-- Theorem: The count of digit 9 occurrences in integers from 1 to 800 is 160 -/
theorem count_nines_to_800 : count_nines 800 = 160 := by sorry

end NUMINAMATH_CALUDE_count_nines_to_800_l1354_135410


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1354_135462

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1354_135462


namespace NUMINAMATH_CALUDE_tournament_games_per_pair_l1354_135428

/-- Represents a chess tournament --/
structure ChessTournament where
  num_players : ℕ
  total_games : ℕ
  games_per_pair : ℕ
  h_players : num_players = 19
  h_total_games : total_games = 342
  h_games_formula : total_games = (num_players * (num_players - 1) * games_per_pair) / 2

/-- Theorem stating that in the given tournament, each player plays against each opponent twice --/
theorem tournament_games_per_pair (t : ChessTournament) : t.games_per_pair = 2 := by
  sorry

#check tournament_games_per_pair

end NUMINAMATH_CALUDE_tournament_games_per_pair_l1354_135428


namespace NUMINAMATH_CALUDE_inequality_solution_l1354_135415

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then
    {x | x ≤ -1}
  else if a > 0 then
    {x | x ≥ 2/a ∨ x ≤ -1}
  else if -2 < a ∧ a < 0 then
    {x | 2/a ≤ x ∧ x ≤ -1}
  else if a = -2 then
    {x | x = -1}
  else
    {x | -1 ≤ x ∧ x ≤ 2/a}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | a*x^2 - 2 ≥ 2*x - a*x} = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1354_135415


namespace NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1354_135469

theorem tangent_point_x_coordinate
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^2 + 1)
  (h2 : ∃ x, HasDerivAt f 4 x) :
  ∃ x, HasDerivAt f 4 x ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_x_coordinate_l1354_135469


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_isosceles_trapezoid_area_is_768_l1354_135453

/-- An isosceles trapezoid with the given properties has an area of 768 sq cm. -/
theorem isosceles_trapezoid_area : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun leg_length diagonal_length longer_base area =>
    leg_length = 30 ∧
    diagonal_length = 40 ∧
    longer_base = 50 ∧
    area = 768 ∧
    ∃ (height shorter_base : ℝ),
      height > 0 ∧
      shorter_base > 0 ∧
      shorter_base < longer_base ∧
      leg_length^2 = height^2 + ((longer_base - shorter_base) / 2)^2 ∧
      diagonal_length^2 = height^2 + (longer_base^2 / 4) ∧
      area = (longer_base + shorter_base) * height / 2

/-- The isosceles trapezoid with the given properties has an area of 768 sq cm. -/
theorem isosceles_trapezoid_area_is_768 : isosceles_trapezoid_area 30 40 50 768 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_isosceles_trapezoid_area_is_768_l1354_135453


namespace NUMINAMATH_CALUDE_max_consecutive_sum_l1354_135465

/-- The sum of n consecutive integers starting from a -/
def sum_consecutive (a : ℤ) (n : ℕ) : ℤ := n * (2 * a + n - 1) / 2

/-- The target sum -/
def target_sum : ℕ := 528

/-- The maximum number of consecutive integers summing to the target -/
def max_consecutive : ℕ := 1056

theorem max_consecutive_sum :
  (∃ a : ℤ, sum_consecutive a max_consecutive = target_sum) ∧
  (∀ n : ℕ, n > max_consecutive → ¬∃ a : ℤ, sum_consecutive a n = target_sum) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_l1354_135465


namespace NUMINAMATH_CALUDE_bd_length_l1354_135422

/-- Given four points A, B, C, and D on a line in that order, prove that BD = 6 -/
theorem bd_length 
  (A B C D : ℝ) -- Points represented as real numbers on a line
  (h_order : A ≤ B ∧ B ≤ C ∧ C ≤ D) -- Order of points on the line
  (h_AB : B - A = 2) -- Length of AB
  (h_AC : C - A = 5) -- Length of AC
  (h_CD : D - C = 3) -- Length of CD
  : D - B = 6 := by
  sorry

end NUMINAMATH_CALUDE_bd_length_l1354_135422


namespace NUMINAMATH_CALUDE_max_value_expression_l1354_135499

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (2 * x^2 + 2 * y^2 + 2) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l1354_135499


namespace NUMINAMATH_CALUDE_empty_tank_weight_is_80_l1354_135418

/-- The weight of an empty water tank --/
def empty_tank_weight (tank_capacity : ℝ) (fill_percentage : ℝ) (water_weight : ℝ) (filled_weight : ℝ) : ℝ :=
  filled_weight - (tank_capacity * fill_percentage * water_weight)

/-- Theorem stating the weight of the empty tank --/
theorem empty_tank_weight_is_80 :
  empty_tank_weight 200 0.80 8 1360 = 80 := by
  sorry

end NUMINAMATH_CALUDE_empty_tank_weight_is_80_l1354_135418


namespace NUMINAMATH_CALUDE_jeans_discount_percentage_l1354_135488

theorem jeans_discount_percentage (original_price : ℝ) (coupon : ℝ) (card_discount : ℝ) (total_savings : ℝ) :
  original_price = 125 →
  coupon = 10 →
  card_discount = 0.1 →
  total_savings = 44 →
  ∃ (sale_discount : ℝ),
    sale_discount = 0.2 ∧
    (original_price - sale_discount * original_price - coupon) * (1 - card_discount) = original_price - total_savings :=
by sorry

end NUMINAMATH_CALUDE_jeans_discount_percentage_l1354_135488


namespace NUMINAMATH_CALUDE_prob_at_least_one_junior_l1354_135461

/-- The probability of selecting at least one junior when randomly choosing 4 people from a group of 8 seniors and 4 juniors -/
theorem prob_at_least_one_junior (total : ℕ) (seniors : ℕ) (juniors : ℕ) (selected : ℕ) : 
  total = seniors + juniors →
  seniors = 8 →
  juniors = 4 →
  selected = 4 →
  (1 - (seniors.choose selected : ℚ) / (total.choose selected : ℚ)) = 85 / 99 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_junior_l1354_135461


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1354_135483

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = -3) : 
  Complex.abs (a + b + c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1354_135483


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1354_135424

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 9*x + 20) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 6) * (x^2 + 6*x + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1354_135424


namespace NUMINAMATH_CALUDE_rod_cutting_l1354_135459

theorem rod_cutting (rod_length_m : ℝ) (piece_length_cm : ℝ) : 
  rod_length_m = 38.25 →
  piece_length_cm = 85 →
  ⌊(rod_length_m * 100) / piece_length_cm⌋ = 45 := by
sorry

end NUMINAMATH_CALUDE_rod_cutting_l1354_135459


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l1354_135464

theorem complex_magnitude_squared (z₁ z₂ : ℂ) :
  let z₁ : ℂ := 3 * Real.sqrt 2 - 5*I
  let z₂ : ℂ := 2 * Real.sqrt 5 + 4*I
  ‖z₁ * z₂‖^2 = 1548 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l1354_135464


namespace NUMINAMATH_CALUDE_seating_arrangements_l1354_135492

/-- The number of seats in the front row -/
def front_seats : ℕ := 11

/-- The number of seats in the back row -/
def back_seats : ℕ := 12

/-- The total number of seats -/
def total_seats : ℕ := front_seats + back_seats

/-- The number of restricted seats in the front row -/
def restricted_seats : ℕ := 3

/-- The number of people to be seated -/
def people : ℕ := 2

/-- The number of arrangements without restrictions -/
def arrangements_without_restrictions : ℕ := total_seats * (total_seats - 2)

/-- The number of arrangements with one person in restricted seats -/
def arrangements_with_one_restricted : ℕ := restricted_seats * (total_seats - 3)

/-- The number of arrangements with both people in restricted seats -/
def arrangements_both_restricted : ℕ := restricted_seats * (restricted_seats - 1)

theorem seating_arrangements :
  arrangements_without_restrictions - 2 * arrangements_with_one_restricted + arrangements_both_restricted = 346 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1354_135492


namespace NUMINAMATH_CALUDE_geometric_sequence_statements_l1354_135448

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Statement 1: One term of a geometric sequence can be 0. -/
def Statement1 : Prop :=
  ∃ (a : ℕ → ℝ) (n : ℕ), IsGeometricSequence a ∧ a n = 0

/-- Statement 2: The common ratio of a geometric sequence can take any real value. -/
def Statement2 : Prop :=
  ∀ r : ℝ, ∃ a : ℕ → ℝ, IsGeometricSequence a ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Statement 3: If b² = ac, then a, b, c form a geometric sequence. -/
def Statement3 : Prop :=
  ∀ a b c : ℝ, b^2 = a * c → ∃ r : ℝ, r ≠ 0 ∧ b = r * a ∧ c = r * b

/-- Statement 4: If a constant sequence is a geometric sequence, then its common ratio is 1. -/
def Statement4 : Prop :=
  ∀ (a : ℕ → ℝ), (∀ n m : ℕ, a n = a m) → IsGeometricSequence a → ∃ r : ℝ, r = 1 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_statements :
  ¬Statement1 ∧ ¬Statement2 ∧ ¬Statement3 ∧ Statement4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_statements_l1354_135448


namespace NUMINAMATH_CALUDE_systematic_sampling_removal_l1354_135419

theorem systematic_sampling_removal (total : Nat) (sample_size : Nat) (h : total = 162 ∧ sample_size = 16) :
  total % sample_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_removal_l1354_135419


namespace NUMINAMATH_CALUDE_fifth_term_is_negative_ten_l1354_135443

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ+ → ℤ) : Prop :=
  a 1 = -2 ∧ ∀ m n : ℕ+, a (m + n) = a m + a n

/-- The theorem stating that the 5th term of the sequence is -10 -/
theorem fifth_term_is_negative_ten (a : ℕ+ → ℤ) (h : special_sequence a) : 
  a 5 = -10 := by sorry

end NUMINAMATH_CALUDE_fifth_term_is_negative_ten_l1354_135443


namespace NUMINAMATH_CALUDE_square_pentagon_side_ratio_l1354_135497

theorem square_pentagon_side_ratio (perimeter : ℝ) (square_side : ℝ) (pentagon_side : ℝ)
  (h1 : perimeter > 0)
  (h2 : 4 * square_side = perimeter)
  (h3 : 5 * pentagon_side = perimeter) :
  pentagon_side / square_side = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_square_pentagon_side_ratio_l1354_135497


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1354_135403

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 20 → 
    b = 21 → 
    c^2 = a^2 + b^2 → 
    c = 29 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1354_135403


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1354_135432

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Ensure positive side lengths
  (a^2 + b^2 = c^2) →        -- Pythagorean theorem (right-angled triangle)
  (a^2 + b^2 + c^2 = 2450) → -- Sum of squares condition
  (b = a + 7) →              -- One leg is 7 units longer
  c = 35 := by               -- Conclusion: hypotenuse length is 35
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1354_135432


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1354_135493

theorem arithmetic_sequence_middle_term (a₁ a₃ y : ℤ) :
  a₁ = 3^2 →
  a₃ = 3^4 →
  y = (a₁ + a₃) / 2 →
  y = 45 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1354_135493


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_three_l1354_135452

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_expression_equals_three :
  log10 5^2 + 2/3 * log10 8 + log10 5 * log10 20 + (log10 2)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_three_l1354_135452


namespace NUMINAMATH_CALUDE_satellite_sensor_ratio_l1354_135450

theorem satellite_sensor_ratio :
  ∀ (S : ℝ) (N : ℝ),
    S > 0 →
    N > 0 →
    S = 0.2 * S + 24 * N →
    N / (0.2 * S) = 1 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_satellite_sensor_ratio_l1354_135450


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1354_135408

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (Complex.mk a b) = (1 : ℂ) / (1 - Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l1354_135408


namespace NUMINAMATH_CALUDE_factorization1_factorization2_l1354_135478

-- Define the expressions
def expr1 (x y : ℝ) : ℝ := 4 - 12 * (x - y) + 9 * (x - y)^2

def expr2 (a x : ℝ) : ℝ := 2 * a * (x^2 + 1)^2 - 8 * a * x^2

-- State the theorems
theorem factorization1 (x y : ℝ) : expr1 x y = (2 - 3*x + 3*y)^2 := by sorry

theorem factorization2 (a x : ℝ) : expr2 a x = 2 * a * (x - 1)^2 * (x + 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization1_factorization2_l1354_135478


namespace NUMINAMATH_CALUDE_vertex_x_coordinate_l1354_135412

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y

-- Theorem statement
theorem vertex_x_coordinate 
  (a b c : ℝ) 
  (h1 : passes_through (quadratic a b c) 2 5)
  (h2 : passes_through (quadratic a b c) 8 5)
  (h3 : passes_through (quadratic a b c) 10 16) :
  ∃ (x_vertex : ℝ), x_vertex = 5 ∧ 
    ∀ (x : ℝ), quadratic a b c x ≥ quadratic a b c x_vertex :=
sorry

end NUMINAMATH_CALUDE_vertex_x_coordinate_l1354_135412


namespace NUMINAMATH_CALUDE_right_triangle_set_l1354_135425

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that only one of the given sets forms a right triangle -/
theorem right_triangle_set :
  ¬(is_right_triangle 2 4 3) ∧
  ¬(is_right_triangle 6 8 9) ∧
  ¬(is_right_triangle 3 4 6) ∧
  is_right_triangle 1 1 (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_set_l1354_135425


namespace NUMINAMATH_CALUDE_opposite_of_seven_l1354_135495

/-- The opposite of a number is the number that, when added to the original number, results in zero. -/
def opposite (a : ℝ) : ℝ := -a

/-- Theorem: The opposite of 7 is -7. -/
theorem opposite_of_seven : opposite 7 = -7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_seven_l1354_135495


namespace NUMINAMATH_CALUDE_pages_per_notepad_l1354_135429

/-- Proves that given the total cost, cost per notepad, and total pages,
    the number of pages per notepad can be determined. -/
theorem pages_per_notepad
  (total_cost : ℝ)
  (cost_per_notepad : ℝ)
  (total_pages : ℕ)
  (h1 : total_cost = 10)
  (h2 : cost_per_notepad = 1.25)
  (h3 : total_pages = 480) :
  (total_pages : ℝ) / (total_cost / cost_per_notepad) = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_pages_per_notepad_l1354_135429


namespace NUMINAMATH_CALUDE_innings_count_l1354_135401

/-- Represents the batting statistics of a batsman -/
structure BattingStats where
  n : ℕ  -- number of innings
  T : ℕ  -- total runs
  H : ℕ  -- highest score
  L : ℕ  -- lowest score

/-- The conditions given in the problem -/
def batting_conditions (stats : BattingStats) : Prop :=
  stats.T = 63 * stats.n ∧  -- batting average is 63
  stats.H - stats.L = 150 ∧  -- difference between highest and lowest score
  stats.H = 248 ∧  -- highest score
  (stats.T - stats.H - stats.L) / (stats.n - 2) = 58  -- average excluding highest and lowest

/-- The theorem to prove -/
theorem innings_count (stats : BattingStats) : 
  batting_conditions stats → stats.n = 46 := by
  sorry


end NUMINAMATH_CALUDE_innings_count_l1354_135401


namespace NUMINAMATH_CALUDE_birds_count_l1354_135430

/-- The number of fish-eater birds Cohen saw over three days -/
def total_birds (initial : ℕ) : ℕ :=
  let day1 := initial
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3

/-- Theorem stating that the total number of birds seen over three days is 1300 -/
theorem birds_count : total_birds 300 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_birds_count_l1354_135430


namespace NUMINAMATH_CALUDE_garden_trees_l1354_135455

/-- The number of trees in a garden with given specifications -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem stating that the number of trees in the garden is 26 -/
theorem garden_trees : number_of_trees 400 16 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l1354_135455


namespace NUMINAMATH_CALUDE_base_7_65234_equals_16244_l1354_135467

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_65234_equals_16244 :
  base_7_to_10 [4, 3, 2, 5, 6] = 16244 := by
  sorry

end NUMINAMATH_CALUDE_base_7_65234_equals_16244_l1354_135467


namespace NUMINAMATH_CALUDE_orange_count_l1354_135423

theorem orange_count (initial_apples : ℕ) (initial_oranges : ℕ) : 
  initial_apples = 14 →
  (initial_apples : ℚ) / (initial_apples + initial_oranges - 14 : ℚ) = 70 / 100 →
  initial_oranges = 20 :=
by sorry

end NUMINAMATH_CALUDE_orange_count_l1354_135423


namespace NUMINAMATH_CALUDE_cone_base_radius_l1354_135449

/-- Given a circle of radius 16 divided into 4 equal parts, if one part forms the lateral surface of a cone, then the radius of the cone's base is 4. -/
theorem cone_base_radius (r : ℝ) (h1 : r = 16) (h2 : r > 0) : 
  (2 * Real.pi * r) / 4 = 2 * Real.pi * 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l1354_135449


namespace NUMINAMATH_CALUDE_cos_two_thirds_pi_l1354_135442

theorem cos_two_thirds_pi : Real.cos (2/3 * Real.pi) = -(1/2) := by sorry

end NUMINAMATH_CALUDE_cos_two_thirds_pi_l1354_135442


namespace NUMINAMATH_CALUDE_same_color_probability_is_71_288_l1354_135458

/-- Represents a 24-sided die with colored sides -/
structure ColoredDie :=
  (purple : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (sparkly : ℕ)
  (total : ℕ)
  (sum_sides : purple + green + blue + yellow + sparkly = total)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.purple^2 + d.green^2 + d.blue^2 + d.yellow^2 + d.sparkly^2) / d.total^2

/-- Our specific 24-sided die -/
def our_die : ColoredDie :=
  { purple := 5
  , green := 6
  , blue := 8
  , yellow := 4
  , sparkly := 1
  , total := 24
  , sum_sides := by rfl }

theorem same_color_probability_is_71_288 :
  same_color_probability our_die = 71 / 288 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_71_288_l1354_135458


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1354_135477

def vector_a (m : ℝ) : ℝ × ℝ := (3, m)
def vector_b : ℝ × ℝ := (2, -4)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (vector_a m) vector_b → m = -6 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1354_135477


namespace NUMINAMATH_CALUDE_nicks_sister_age_difference_l1354_135426

theorem nicks_sister_age_difference (nick_age : ℕ) (sister_age_diff : ℕ) : 
  nick_age = 13 →
  (nick_age + sister_age_diff) / 2 + 5 = 21 →
  sister_age_diff = 19 := by
  sorry

end NUMINAMATH_CALUDE_nicks_sister_age_difference_l1354_135426


namespace NUMINAMATH_CALUDE_f_composition_equality_l1354_135484

noncomputable def f (x : ℝ) : ℝ :=
  if x > 3 then Real.exp x else Real.log (x + 1)

theorem f_composition_equality : f (f (f 1)) = Real.log (Real.log (Real.log 2 + 1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equality_l1354_135484


namespace NUMINAMATH_CALUDE_onion_basket_change_l1354_135468

theorem onion_basket_change (x : ℤ) : x + 4 - 5 + 9 = x + 8 := by
  sorry

end NUMINAMATH_CALUDE_onion_basket_change_l1354_135468


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1354_135447

theorem fraction_equals_zero (x : ℝ) (h : (x - 3) / x = 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1354_135447


namespace NUMINAMATH_CALUDE_not_isosceles_if_distinct_sides_l1354_135496

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Theorem statement
theorem not_isosceles_if_distinct_sides (t : Triangle) 
  (distinct_sides : t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c) : 
  ¬(is_isosceles t) := by
  sorry

end NUMINAMATH_CALUDE_not_isosceles_if_distinct_sides_l1354_135496


namespace NUMINAMATH_CALUDE_jason_has_four_balloons_l1354_135489

/-- The number of violet balloons Jason has now, given his initial count and the number he lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason has 4 violet balloons now. -/
theorem jason_has_four_balloons : remaining_balloons 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_four_balloons_l1354_135489


namespace NUMINAMATH_CALUDE_final_shell_count_l1354_135431

def calculate_final_shells (initial : ℕ) 
  (vacation1_day1to3 : ℕ) (vacation1_day4 : ℕ) (vacation1_lost : ℕ)
  (vacation2_day1to2 : ℕ) (vacation2_day3 : ℕ) (vacation2_given : ℕ)
  (vacation3_day1 : ℕ) (vacation3_day2 : ℕ) (vacation3_day3to4 : ℕ) (vacation3_misplaced : ℕ) : ℕ :=
  initial + 
  (vacation1_day1to3 * 3 + vacation1_day4 - vacation1_lost) +
  (vacation2_day1to2 * 2 + vacation2_day3 - vacation2_given) +
  (vacation3_day1 + vacation3_day2 + vacation3_day3to4 * 2 - vacation3_misplaced)

theorem final_shell_count :
  calculate_final_shells 20 5 6 4 4 7 3 8 4 3 5 = 62 := by
  sorry

end NUMINAMATH_CALUDE_final_shell_count_l1354_135431


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l1354_135481

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the initial number of cookies -/
def initialCookies : ℕ := 60

/-- Represents the initial number of brownies -/
def initialBrownies : ℕ := 10

/-- Represents the number of cookies eaten per day -/
def cookiesPerDay : ℕ := 3

/-- Represents the number of brownies eaten per day -/
def browniesPerDay : ℕ := 1

/-- Calculates the remaining cookies after a week -/
def remainingCookies : ℕ := initialCookies - daysInWeek * cookiesPerDay

/-- Calculates the remaining brownies after a week -/
def remainingBrownies : ℕ := initialBrownies - daysInWeek * browniesPerDay

/-- Theorem stating the difference between remaining cookies and brownies after a week -/
theorem cookie_brownie_difference :
  remainingCookies - remainingBrownies = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l1354_135481


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1354_135485

theorem min_value_sum_squares (x₁ x₂ x₃ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ = 100) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 10000/29 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1354_135485


namespace NUMINAMATH_CALUDE_roots_of_equation_l1354_135436

/-- The equation for which we need to find roots -/
def equation (x : ℝ) : Prop :=
  15 / (x^2 - 4) - 2 / (x - 2) = 1

/-- Theorem stating that -3 and 5 are the roots of the equation -/
theorem roots_of_equation :
  equation (-3) ∧ equation 5 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1354_135436


namespace NUMINAMATH_CALUDE_vasya_upward_run_time_l1354_135434

/-- Represents the speed and time properties of Vasya's escalator run -/
structure EscalatorRun where
  -- Vasya's speed going down (in units per minute)
  speed_down : ℝ
  -- Vasya's speed going up (in units per minute)
  speed_up : ℝ
  -- Escalator's speed (in units per minute)
  escalator_speed : ℝ
  -- Time for stationary run (in minutes)
  time_stationary : ℝ
  -- Time for downward moving escalator run (in minutes)
  time_down : ℝ
  -- Constraint: Vasya runs down twice as fast as he runs up
  speed_constraint : speed_down = 2 * speed_up
  -- Constraint: Stationary run takes 6 minutes
  stationary_constraint : time_stationary = 6
  -- Constraint: Downward moving escalator run takes 13.5 minutes
  down_constraint : time_down = 13.5

/-- Theorem stating the time for Vasya's upward moving escalator run -/
theorem vasya_upward_run_time (run : EscalatorRun) :
  let time_up := (1 / (run.speed_down - run.escalator_speed) + 1 / (run.speed_up + run.escalator_speed)) * 60
  time_up = 324 := by
  sorry

end NUMINAMATH_CALUDE_vasya_upward_run_time_l1354_135434


namespace NUMINAMATH_CALUDE_dots_on_abc_l1354_135404

/-- Represents a die face with a number of dots -/
structure DieFace :=
  (dots : Nat)
  (h : dots ≥ 1 ∧ dots ≤ 6)

/-- Represents a die with six faces -/
structure Die :=
  (faces : Fin 6 → DieFace)
  (opposite_sum : ∀ i : Fin 3, (faces i).dots + (faces (i + 3)).dots = 7)
  (all_different : ∀ i j : Fin 6, i ≠ j → (faces i).dots ≠ (faces j).dots)

/-- Represents the configuration of four glued dice -/
structure GluedDice :=
  (dice : Fin 4 → Die)
  (glued_faces_same : ∀ i j : Fin 4, i ≠ j → ∃ fi fj : Fin 6, 
    (dice i).faces fi = (dice j).faces fj)

/-- The main theorem stating the number of dots on faces A, B, and C -/
theorem dots_on_abc (gd : GluedDice) : 
  ∃ (a b c : DieFace), 
    a.dots = 2 ∧ b.dots = 2 ∧ c.dots = 6 ∧
    (∃ (i j k : Fin 4) (fi fj fk : Fin 6), 
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      a = (gd.dice i).faces fi ∧
      b = (gd.dice j).faces fj ∧
      c = (gd.dice k).faces fk) :=
sorry

end NUMINAMATH_CALUDE_dots_on_abc_l1354_135404


namespace NUMINAMATH_CALUDE_line_vertical_shift_specific_line_shift_l1354_135433

/-- Given a line y = mx + b, moving it down by k units results in y = mx + (b - k) -/
theorem line_vertical_shift (m b k : ℝ) :
  let original_line := fun (x : ℝ) => m * x + b
  let shifted_line := fun (x : ℝ) => m * x + (b - k)
  (∀ x, shifted_line x = original_line x - k) :=
by sorry

/-- Moving the line y = 3x down 2 units results in y = 3x - 2 -/
theorem specific_line_shift :
  let original_line := fun (x : ℝ) => 3 * x
  let shifted_line := fun (x : ℝ) => 3 * x - 2
  (∀ x, shifted_line x = original_line x - 2) :=
by sorry

end NUMINAMATH_CALUDE_line_vertical_shift_specific_line_shift_l1354_135433


namespace NUMINAMATH_CALUDE_min_distance_to_line_l1354_135498

/-- Given a line 5x + 12y = 60, the minimum distance from the origin (0, 0) to any point (x, y) on this line is 60/13 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | 5 * x + 12 * y = 60}
  ∃ (d : ℝ), d = 60 / 13 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ line → 
      d ≤ Real.sqrt ((p.1 ^ 2) + (p.2 ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l1354_135498


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l1354_135400

theorem spencer_walk_distance (total : ℝ) (house_to_library : ℝ) (library_to_post : ℝ)
  (h1 : total = 0.8)
  (h2 : house_to_library = 0.3)
  (h3 : library_to_post = 0.1) :
  total - (house_to_library + library_to_post) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_spencer_walk_distance_l1354_135400


namespace NUMINAMATH_CALUDE_equilibrium_force_l1354_135456

/-- Given three forces in a 2D plane, prove that a specific fourth force is required for equilibrium. -/
theorem equilibrium_force (f₁ f₂ f₃ f₄ : ℝ × ℝ) : 
  f₁ = (-2, -1) → f₂ = (-3, 2) → f₃ = (4, -3) →
  (f₁.1 + f₂.1 + f₃.1 + f₄.1 = 0 ∧ f₁.2 + f₂.2 + f₃.2 + f₄.2 = 0) →
  f₄ = (1, 2) := by
sorry

end NUMINAMATH_CALUDE_equilibrium_force_l1354_135456


namespace NUMINAMATH_CALUDE_oddDigitSequence_157th_l1354_135444

/-- A function that generates the nth number in the sequence of positive integers formed only by odd digits -/
def oddDigitSequence (n : ℕ) : ℕ :=
  sorry

/-- The set of odd digits -/
def oddDigits : Set ℕ := {1, 3, 5, 7, 9}

/-- A predicate to check if a number consists only of odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ∈ oddDigits

/-- The main theorem stating that the 157th number in the sequence is 1113 -/
theorem oddDigitSequence_157th :
  oddDigitSequence 157 = 1113 ∧ hasOnlyOddDigits (oddDigitSequence 157) :=
sorry

end NUMINAMATH_CALUDE_oddDigitSequence_157th_l1354_135444


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l1354_135454

theorem ratio_a_to_b (a b c : ℝ) (h1 : c = 0.2 * a) (h2 : c = 0.1 * b) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l1354_135454


namespace NUMINAMATH_CALUDE_rectangles_count_l1354_135476

/-- The number of rectangles in an n×n square grid -/
def rectangles_in_square (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The number of rectangles in the given arrangement of three n×n square grids -/
def rectangles_in_three_squares (n : ℕ) : ℕ := 
  n^2 * (2*n + 1)^2 - n^4 - n^3*(n + 1) - (n * (n + 1) / 2)^2

theorem rectangles_count (n : ℕ) (h : n > 0) : 
  (rectangles_in_square n = (n * (n + 1) / 2) ^ 2) ∧ 
  (rectangles_in_three_squares n = n^2 * (2*n + 1)^2 - n^4 - n^3*(n + 1) - (n * (n + 1) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_rectangles_count_l1354_135476


namespace NUMINAMATH_CALUDE_complex_subtraction_l1354_135435

theorem complex_subtraction : (5 - 3*I) - (2 + 7*I) = 3 - 10*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1354_135435


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1354_135414

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x + 3 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1354_135414


namespace NUMINAMATH_CALUDE_students_who_just_passed_l1354_135427

theorem students_who_just_passed 
  (total_students : ℕ) 
  (first_division_percent : ℚ) 
  (second_division_percent : ℚ) 
  (h1 : total_students = 300)
  (h2 : first_division_percent = 27 / 100)
  (h3 : second_division_percent = 54 / 100)
  (h4 : first_division_percent + second_division_percent < 1) :
  total_students - (total_students * (first_division_percent + second_division_percent)).floor = 57 := by
sorry

end NUMINAMATH_CALUDE_students_who_just_passed_l1354_135427


namespace NUMINAMATH_CALUDE_cone_volume_l1354_135417

/-- The volume of a cone with slant height 5 and base radius 3 is 12π -/
theorem cone_volume (s h r : ℝ) (hs : s = 5) (hr : r = 3) 
  (height_eq : h^2 + r^2 = s^2) : 
  (1/3 : ℝ) * π * r^2 * h = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1354_135417


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1354_135407

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n.choose 2) - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1354_135407


namespace NUMINAMATH_CALUDE_triangle_problem_l1354_135475

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = π ∧
  -- Given equation
  c * Real.cos B + (Real.sqrt 3 / 3) * b * Real.sin C - a = 0 ∧
  -- Given side length
  c = 3 ∧
  -- Given area
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 →
  -- Conclusions
  C = π/3 ∧ a + b = 3 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l1354_135475


namespace NUMINAMATH_CALUDE_food_allocation_difference_l1354_135446

/-- Proves that the difference in food allocation between soldiers on the first and second sides is 2 pounds -/
theorem food_allocation_difference (
  soldiers_first : ℕ)
  (soldiers_second : ℕ)
  (food_per_soldier_first : ℝ)
  (total_food : ℝ)
  (h1 : soldiers_first = 4000)
  (h2 : soldiers_second = soldiers_first - 500)
  (h3 : food_per_soldier_first = 10)
  (h4 : total_food = 68000)
  (h5 : total_food = soldiers_first * food_per_soldier_first + 
    soldiers_second * (food_per_soldier_first - (food_per_soldier_first - food_per_soldier_second)))
  : food_per_soldier_first - food_per_soldier_second = 2 := by
  sorry

end NUMINAMATH_CALUDE_food_allocation_difference_l1354_135446


namespace NUMINAMATH_CALUDE_quotient_sum_difference_forty_percent_less_than_36_l1354_135451

-- Problem 1
theorem quotient_sum_difference : (0.4 + 1/3) / (0.4 - 1/3) = 11 := by sorry

-- Problem 2
theorem forty_percent_less_than_36 : ∃ x : ℝ, x - 0.4 * x = 36 ∧ x = 60 := by sorry

end NUMINAMATH_CALUDE_quotient_sum_difference_forty_percent_less_than_36_l1354_135451


namespace NUMINAMATH_CALUDE_sum_a_d_l1354_135406

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + b * c + c * a + d * b = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l1354_135406


namespace NUMINAMATH_CALUDE_alyssa_final_money_l1354_135445

def weekly_allowance : ℕ := 8
def movie_spending : ℕ := weekly_allowance / 2
def car_wash_earnings : ℕ := 8

theorem alyssa_final_money :
  weekly_allowance - movie_spending + car_wash_earnings = 12 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_final_money_l1354_135445


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l1354_135420

-- Define the function f(x) = -x²
def f (x : ℝ) : ℝ := -x^2

-- State the theorem
theorem f_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l1354_135420


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1354_135466

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 4*x - 5 < 0} = Set.Ioo (-5) 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1354_135466
