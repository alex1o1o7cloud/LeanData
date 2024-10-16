import Mathlib

namespace NUMINAMATH_CALUDE_complement_of_union_theorem_l2894_289482

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x ≤ 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_of_union_theorem :
  (U \ (A ∪ B)) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_theorem_l2894_289482


namespace NUMINAMATH_CALUDE_min_rolls_for_repeat_sum_l2894_289443

/-- Represents an eight-sided die -/
def Die8 := Fin 8

/-- The sum of two dice rolls -/
def DiceSum := Fin 15

/-- The number of possible sums when rolling two eight-sided dice -/
def NumPossibleSums : ℕ := 15

/-- The minimum number of rolls to guarantee a repeated sum -/
def MinRollsForRepeat : ℕ := NumPossibleSums + 1

theorem min_rolls_for_repeat_sum : 
  ∀ (rolls : ℕ), rolls ≥ MinRollsForRepeat → 
  ∃ (sum : DiceSum), (∃ (i j : Fin rolls), i ≠ j ∧ 
    ∃ (d1 d2 d3 d4 : Die8), 
      sum = ⟨d1.val + d2.val - 1, by sorry⟩ ∧
      sum = ⟨d3.val + d4.val - 1, by sorry⟩) :=
by sorry

end NUMINAMATH_CALUDE_min_rolls_for_repeat_sum_l2894_289443


namespace NUMINAMATH_CALUDE_complex_product_l2894_289428

def z₁ : ℂ := 1 + 2 * Complex.I
def z₂ : ℂ := 2 - Complex.I

theorem complex_product : z₁ * z₂ = 4 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_l2894_289428


namespace NUMINAMATH_CALUDE_original_population_l2894_289499

def population_change (p : ℕ) : ℝ :=
  0.85 * (p + 1500 : ℝ) - p

theorem original_population : 
  ∃ p : ℕ, population_change p = -50 ∧ p = 8833 := by
  sorry

end NUMINAMATH_CALUDE_original_population_l2894_289499


namespace NUMINAMATH_CALUDE_bryan_books_per_continent_l2894_289403

/-- The number of continents Bryan visited -/
def num_continents : ℕ := 4

/-- The total number of books Bryan collected -/
def total_books : ℕ := 488

/-- The number of books Bryan collected per continent -/
def books_per_continent : ℕ := total_books / num_continents

/-- Theorem stating that Bryan collected 122 books per continent -/
theorem bryan_books_per_continent : books_per_continent = 122 := by
  sorry

end NUMINAMATH_CALUDE_bryan_books_per_continent_l2894_289403


namespace NUMINAMATH_CALUDE_slope_MN_constant_l2894_289400

/-- Curve C defined by y² = 4x -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Point D on curve C -/
def D : ℝ × ℝ := (1, 2)

/-- Line with slope k passing through D -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - D.1) + D.2}

/-- Theorem: The slope of line MN is constant -/
theorem slope_MN_constant (k : ℝ) :
  k ≠ 0 →
  D ∈ C →
  ∃ (M N : ℝ × ℝ),
    M ∈ C ∧ M ∈ line k ∧
    N ∈ C ∧ N ∈ line (-1/k) ∧
    M ≠ D ∧ N ≠ D →
    (N.2 - M.2) / (N.1 - M.1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_MN_constant_l2894_289400


namespace NUMINAMATH_CALUDE_marbles_given_l2894_289401

theorem marbles_given (initial_marbles : ℕ) (remaining_marbles : ℕ) : 
  initial_marbles = 87 → remaining_marbles = 79 → initial_marbles - remaining_marbles = 8 := by
sorry

end NUMINAMATH_CALUDE_marbles_given_l2894_289401


namespace NUMINAMATH_CALUDE_pattern_proof_l2894_289463

theorem pattern_proof (n : ℕ+) : n * (n + 2) + 1 = (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l2894_289463


namespace NUMINAMATH_CALUDE_gcf_of_180_252_315_l2894_289438

theorem gcf_of_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_252_315_l2894_289438


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l2894_289453

theorem perfect_square_quadratic (n : ℤ) : ∃ m : ℤ, 4 * n^2 + 12 * n + 9 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l2894_289453


namespace NUMINAMATH_CALUDE_paco_cookies_bought_l2894_289421

-- Define the initial number of cookies
def initial_cookies : ℕ := 13

-- Define the number of cookies eaten
def cookies_eaten : ℕ := 2

-- Define the additional cookies compared to eaten ones
def additional_cookies : ℕ := 34

-- Define the function to calculate the number of cookies bought
def cookies_bought (initial : ℕ) (eaten : ℕ) (additional : ℕ) : ℕ :=
  additional + eaten

-- Theorem statement
theorem paco_cookies_bought :
  cookies_bought initial_cookies cookies_eaten additional_cookies = 36 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_bought_l2894_289421


namespace NUMINAMATH_CALUDE_exponential_simplification_l2894_289426

theorem exponential_simplification : 3 * ((-5)^2)^(3/4) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_exponential_simplification_l2894_289426


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_negative_l2894_289495

/-- A point is in the third quadrant if both its x and y coordinates are negative -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- If point A(a, a-1) is in the third quadrant, then a < 0 -/
theorem point_in_third_quadrant_implies_a_negative (a : ℝ) : 
  in_third_quadrant a (a - 1) → a < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_implies_a_negative_l2894_289495


namespace NUMINAMATH_CALUDE_negation_of_implication_l2894_289441

-- Define a triangle type
structure Triangle where
  -- Add any necessary fields here
  mk :: -- Constructor

-- Define properties for triangles
def isEquilateral (t : Triangle) : Prop := sorry
def interiorAnglesEqual (t : Triangle) : Prop := sorry

-- State the theorem
theorem negation_of_implication :
  (¬(∀ t : Triangle, isEquilateral t → interiorAnglesEqual t)) ↔
  (∀ t : Triangle, ¬isEquilateral t → ¬interiorAnglesEqual t) :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2894_289441


namespace NUMINAMATH_CALUDE_line_through_point_l2894_289420

/-- Given a line with equation 3 - 2kx = -4y that contains the point (5, -2),
    prove that the value of k is -0.5 -/
theorem line_through_point (k : ℝ) : 
  (3 - 2 * k * 5 = -4 * (-2)) → k = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2894_289420


namespace NUMINAMATH_CALUDE_progression_product_exceeds_100000_l2894_289406

theorem progression_product_exceeds_100000 (n : ℕ) : 
  (n ≥ 11 ∧ ∀ k < 11, k > 0 → 10^((k * (k + 1)) / 22) ≤ 10^5) ↔ 
  (∀ k ≤ n, 10^((k * (k + 1)) / 22) > 10^5 ↔ k ≥ 11) := by
  sorry

end NUMINAMATH_CALUDE_progression_product_exceeds_100000_l2894_289406


namespace NUMINAMATH_CALUDE_cylinder_line_distance_theorem_l2894_289470

/-- A cylinder with a square axial cross-section -/
structure SquareCylinder where
  /-- The side length of the square axial cross-section -/
  side : ℝ
  /-- Assertion that the side length is positive -/
  side_pos : 0 < side

/-- A line segment connecting points on the upper and lower bases of the cylinder -/
structure CylinderLineSegment (c : SquareCylinder) where
  /-- The length of the line segment -/
  length : ℝ
  /-- The angle the line segment makes with the base plane -/
  angle : ℝ
  /-- Assertion that the length is positive -/
  length_pos : 0 < length
  /-- Assertion that the angle is between 0 and π/2 -/
  angle_range : 0 < angle ∧ angle < Real.pi / 2

/-- The theorem stating the distance formula and angle range -/
theorem cylinder_line_distance_theorem (c : SquareCylinder) (l : CylinderLineSegment c) :
  ∃ (d : ℝ), d = (1 / 2) * l.length * Real.sqrt (-Real.cos (2 * l.angle)) ∧
  Real.pi / 4 < l.angle ∧ l.angle < 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_line_distance_theorem_l2894_289470


namespace NUMINAMATH_CALUDE_equation_solution_l2894_289422

theorem equation_solution (x : ℝ) : 
  (∀ y : ℝ, 6*x*y - 18*y + 3*x - 9 = 0) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2894_289422


namespace NUMINAMATH_CALUDE_log15_12_equals_fraction_l2894_289486

-- Define the logarithm base 10 (lg) and logarithm base 15
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10
noncomputable def log15 (x : ℝ) := Real.log x / Real.log 15

-- State the theorem
theorem log15_12_equals_fraction (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log15 12 = (2*a + b) / (1 - a + b) := by sorry

end NUMINAMATH_CALUDE_log15_12_equals_fraction_l2894_289486


namespace NUMINAMATH_CALUDE_city_population_dynamics_l2894_289451

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birthRate : ℝ  -- Average birth rate per second
  netIncrease : ℝ  -- Net population increase per second
  deathRate : ℝ  -- Average death rate per second

/-- Theorem stating the relationship between birth rate, net increase, and death rate --/
theorem city_population_dynamics (city : CityPopulation) 
  (h1 : city.birthRate = 3.5)
  (h2 : city.netIncrease = 2) :
  city.deathRate = 1.5 := by
  sorry

#check city_population_dynamics

end NUMINAMATH_CALUDE_city_population_dynamics_l2894_289451


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2894_289416

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2894_289416


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_l2894_289458

theorem raffle_ticket_sales (total_money : ℕ) (ticket_cost : ℕ) (num_tickets : ℕ) :
  total_money = 620 →
  ticket_cost = 4 →
  total_money = ticket_cost * num_tickets →
  num_tickets = 155 := by
sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_l2894_289458


namespace NUMINAMATH_CALUDE_units_digit_sum_cubes_l2894_289431

theorem units_digit_sum_cubes : (24^3 + 42^3) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_cubes_l2894_289431


namespace NUMINAMATH_CALUDE_test_questions_l2894_289418

theorem test_questions (total_questions : ℕ) 
  (h1 : total_questions / 2 = (13 : ℕ) + (total_questions - 20) / 4)
  (h2 : total_questions ≥ 20) : total_questions = 32 := by
  sorry

end NUMINAMATH_CALUDE_test_questions_l2894_289418


namespace NUMINAMATH_CALUDE_vector_definition_l2894_289456

/-- A vector in mathematics -/
structure MathVector where
  magnitude : ℝ
  direction : ℝ × ℝ × ℝ

/-- The definition of a vector in mathematics -/
theorem vector_definition :
  ∀ v : MathVector, ∃ m : ℝ, ∃ d : ℝ × ℝ × ℝ,
    v.magnitude = m ∧ v.direction = d :=
by
  sorry

end NUMINAMATH_CALUDE_vector_definition_l2894_289456


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2894_289437

/-- The polynomial we're considering -/
def p (x : ℝ) : ℝ := x^2 - 9

/-- The proposed factorization of the polynomial -/
def f (x : ℝ) : ℝ := (x - 3) * (x + 3)

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, p x = f x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2894_289437


namespace NUMINAMATH_CALUDE_temperature_calculation_l2894_289423

theorem temperature_calculation (T₁ T₂ : ℝ) : 
  2.24 * T₁ = 1.1 * 2 * 298 ∧ 1.76 * T₂ = 1.1 * 2 * 298 → 
  T₁ = 292.7 ∧ T₂ = 372.5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_calculation_l2894_289423


namespace NUMINAMATH_CALUDE_scores_mode_is_37_l2894_289404

def scores : List Nat := [35, 37, 39, 37, 38, 38, 37]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem scores_mode_is_37 : mode scores = 37 := by
  sorry

end NUMINAMATH_CALUDE_scores_mode_is_37_l2894_289404


namespace NUMINAMATH_CALUDE_total_miles_on_wednesdays_l2894_289481

/-- The total miles flown on Wednesdays over a 4-week period, given that a pilot flies
    the same number of miles each week and x miles each Wednesday. -/
theorem total_miles_on_wednesdays
  (x : ℕ)  -- Miles flown on Wednesday
  (h1 : ∀ week : Fin 4, ∃ miles : ℕ, miles = x)  -- Same miles flown each Wednesday for 4 weeks
  : ∃ total : ℕ, total = 4 * x :=
by sorry

end NUMINAMATH_CALUDE_total_miles_on_wednesdays_l2894_289481


namespace NUMINAMATH_CALUDE_unique_solution_implies_equal_or_opposite_l2894_289485

theorem unique_solution_implies_equal_or_opposite (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃! x : ℝ, a * (x - a)^2 + b * (x - b)^2 = 0) → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_equal_or_opposite_l2894_289485


namespace NUMINAMATH_CALUDE_three_primes_sum_to_86_l2894_289405

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem three_primes_sum_to_86 :
  ∃ (a b c : ℕ), isPrime a ∧ isPrime b ∧ isPrime c ∧ a + b + c = 86 ∧
  (∀ (x y z : ℕ), isPrime x ∧ isPrime y ∧ isPrime z ∧ x + y + z = 86 →
    (x = 2 ∧ y = 5 ∧ z = 79) ∨
    (x = 2 ∧ y = 11 ∧ z = 73) ∨
    (x = 2 ∧ y = 13 ∧ z = 71) ∨
    (x = 2 ∧ y = 17 ∧ z = 67) ∨
    (x = 2 ∧ y = 23 ∧ z = 61) ∨
    (x = 2 ∧ y = 31 ∧ z = 53) ∨
    (x = 2 ∧ y = 37 ∧ z = 47) ∨
    (x = 2 ∧ y = 41 ∧ z = 43) ∨
    (x = 5 ∧ y = 2 ∧ z = 79) ∨
    (x = 11 ∧ y = 2 ∧ z = 73) ∨
    (x = 13 ∧ y = 2 ∧ z = 71) ∨
    (x = 17 ∧ y = 2 ∧ z = 67) ∨
    (x = 23 ∧ y = 2 ∧ z = 61) ∨
    (x = 31 ∧ y = 2 ∧ z = 53) ∨
    (x = 37 ∧ y = 2 ∧ z = 47) ∨
    (x = 41 ∧ y = 2 ∧ z = 43) ∨
    (x = 79 ∧ y = 2 ∧ z = 5) ∨
    (x = 73 ∧ y = 2 ∧ z = 11) ∨
    (x = 71 ∧ y = 2 ∧ z = 13) ∨
    (x = 67 ∧ y = 2 ∧ z = 17) ∨
    (x = 61 ∧ y = 2 ∧ z = 23) ∨
    (x = 53 ∧ y = 2 ∧ z = 31) ∨
    (x = 47 ∧ y = 2 ∧ z = 37) ∨
    (x = 43 ∧ y = 2 ∧ z = 41)) :=
by sorry


end NUMINAMATH_CALUDE_three_primes_sum_to_86_l2894_289405


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2894_289462

/-- A trinomial ax^2 + bx + c is a perfect square if there exist p and q such that
    ax^2 + bx + c = (px + q)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℤ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℤ, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_k (k : ℤ) :
  IsPerfectSquareTrinomial 9 6 k → k = 1 := by
  sorry

#check perfect_square_trinomial_k

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l2894_289462


namespace NUMINAMATH_CALUDE_hair_growth_calculation_l2894_289402

theorem hair_growth_calculation (initial_length : ℝ) (growth : ℝ) (final_length : ℝ) : 
  initial_length = 24 →
  final_length = 14 →
  final_length = initial_length / 2 + growth - 2 →
  growth = 4 := by
sorry

end NUMINAMATH_CALUDE_hair_growth_calculation_l2894_289402


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2894_289467

theorem equal_roots_quadratic (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0 ∧ (∀ y : ℝ, y^2 - a*y + 1 = 0 → y = x)) →
  a = 2 ∨ a = -2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2894_289467


namespace NUMINAMATH_CALUDE_problem_1_l2894_289480

theorem problem_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (1) * ((-2*a)^3 * (-a*b^2)^3 - 4*a*b^2 * (2*a^5*b^4 + 1/2*a*b^3 - 5)) / (-2*a*b) = a*b^4 - 10*b :=
sorry

end NUMINAMATH_CALUDE_problem_1_l2894_289480


namespace NUMINAMATH_CALUDE_people_on_boats_l2894_289439

/-- Given 5 boats in a lake, each with 3 people, prove that the total number of people on boats is 15. -/
theorem people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) 
  (h1 : num_boats = 5) 
  (h2 : people_per_boat = 3) : 
  num_boats * people_per_boat = 15 := by
  sorry

end NUMINAMATH_CALUDE_people_on_boats_l2894_289439


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2894_289430

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 (a > 0) and distance between foci equal to 10,
    its eccentricity is 5/4 -/
theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ c : ℝ, 2 * c = 10) →
  (∃ e : ℝ, e = 5/4 ∧ e = c/a) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2894_289430


namespace NUMINAMATH_CALUDE_probability_wind_given_haze_l2894_289473

/-- Probability of moderate haze occurring -/
def P_A : ℝ := 0.25

/-- Probability of wind force four or above occurring -/
def P_B : ℝ := 0.4

/-- Probability of both moderate haze and wind force four or above occurring -/
def P_AB : ℝ := 0.02

/-- Theorem stating that the probability of wind force four or above given moderate haze is 0.08 -/
theorem probability_wind_given_haze :
  P_AB / P_A = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_probability_wind_given_haze_l2894_289473


namespace NUMINAMATH_CALUDE_cake_ratio_theorem_l2894_289464

/-- Proves that the ratio of cakes sold to total cakes baked is 1:2 --/
theorem cake_ratio_theorem (cakes_per_day : ℕ) (days_baked : ℕ) (cakes_left : ℕ) :
  cakes_per_day = 20 →
  days_baked = 9 →
  cakes_left = 90 →
  let total_cakes := cakes_per_day * days_baked
  let cakes_sold := total_cakes - cakes_left
  (cakes_sold : ℚ) / total_cakes = 1 / 2 :=
by
  sorry

#check cake_ratio_theorem

end NUMINAMATH_CALUDE_cake_ratio_theorem_l2894_289464


namespace NUMINAMATH_CALUDE_hotel_room_allocation_l2894_289424

theorem hotel_room_allocation (total_people : ℕ) (small_room_capacity : ℕ) 
  (num_small_rooms : ℕ) (h1 : total_people = 26) (h2 : small_room_capacity = 2) 
  (h3 : num_small_rooms = 1) :
  ∃ (large_room_capacity : ℕ),
    large_room_capacity = 12 ∧
    large_room_capacity > 0 ∧
    (total_people - num_small_rooms * small_room_capacity) % large_room_capacity = 0 ∧
    ∀ (x : ℕ), x > large_room_capacity → 
      (total_people - num_small_rooms * small_room_capacity) % x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_hotel_room_allocation_l2894_289424


namespace NUMINAMATH_CALUDE_remaining_payment_example_l2894_289475

/-- Given a deposit percentage and amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  let total_cost := deposit_amount / deposit_percentage
  total_cost - deposit_amount

/-- Theorem: Given a 10% deposit of $130, the remaining amount to be paid is $1170 -/
theorem remaining_payment_example : 
  remaining_payment (1/10) 130 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_example_l2894_289475


namespace NUMINAMATH_CALUDE_rational_division_l2894_289457

theorem rational_division (x : ℚ) : (-2 : ℚ) / x = 8 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_division_l2894_289457


namespace NUMINAMATH_CALUDE_yellow_ball_count_l2894_289407

theorem yellow_ball_count (r b g y : ℕ) : 
  r = 2 * b →
  b = 2 * g →
  y > 7 →
  r + b + g + y = 27 →
  y = 20 := by
sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l2894_289407


namespace NUMINAMATH_CALUDE_sparcs_characterization_l2894_289465

-- Define "grows to"
def grows_to (s r : ℝ) : Prop :=
  ∃ n : ℕ+, s ^ (n : ℝ) = r

-- Define "sparcs"
def sparcs (r : ℝ) : Prop :=
  {s : ℝ | grows_to s r}.Finite

-- Theorem statement
theorem sparcs_characterization (r : ℝ) :
  sparcs r ↔ r = -1 ∨ r = 0 ∨ r = 1 := by
  sorry

end NUMINAMATH_CALUDE_sparcs_characterization_l2894_289465


namespace NUMINAMATH_CALUDE_parabola_point_relation_l2894_289454

theorem parabola_point_relation (a : ℝ) (y₁ y₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : y₁ = a * (-2)^2) 
  (h3 : y₂ = a * 1^2) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relation_l2894_289454


namespace NUMINAMATH_CALUDE_nine_possible_scores_l2894_289498

/-- The number of baskets scored by the player -/
def total_baskets : ℕ := 8

/-- The possible point values for each basket -/
inductive BasketValue : Type
| one : BasketValue
| three : BasketValue

/-- A function to calculate the total score given a list of basket values -/
def total_score (baskets : List BasketValue) : ℕ :=
  baskets.foldl (fun acc b => acc + match b with
    | BasketValue.one => 1
    | BasketValue.three => 3) 0

/-- The theorem to be proved -/
theorem nine_possible_scores :
  ∃! (scores : Finset ℕ), 
    (∀ (score : ℕ), score ∈ scores ↔ 
      ∃ (baskets : List BasketValue), 
        baskets.length = total_baskets ∧ total_score baskets = score) ∧
    scores.card = 9 := by sorry

end NUMINAMATH_CALUDE_nine_possible_scores_l2894_289498


namespace NUMINAMATH_CALUDE_unique_functional_equation_l2894_289477

theorem unique_functional_equation (f : ℕ+ → ℕ+)
  (h : ∀ m n : ℕ+, f (f m + f n) = m + n) :
  f 1988 = 1988 := by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_l2894_289477


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2894_289494

theorem sqrt_inequality (C : ℝ) (h : C > 1) :
  Real.sqrt (C + 1) - Real.sqrt C < Real.sqrt C - Real.sqrt (C - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2894_289494


namespace NUMINAMATH_CALUDE_correct_adult_ticket_cost_l2894_289466

/-- The cost of an adult ticket in dollars -/
def adult_ticket_cost : ℕ := 19

/-- The cost of a child ticket in dollars -/
def child_ticket_cost : ℕ := adult_ticket_cost - 6

/-- The number of adults in the family -/
def num_adults : ℕ := 2

/-- The number of children in the family -/
def num_children : ℕ := 3

/-- The total cost of tickets for the family -/
def total_cost : ℕ := 77

theorem correct_adult_ticket_cost :
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_correct_adult_ticket_cost_l2894_289466


namespace NUMINAMATH_CALUDE_smallest_label_on_final_position_l2894_289460

/-- The number of points on the circle -/
def n : ℕ := 70

/-- The function that calculates the position of a label -/
def position (k : ℕ) : ℕ := (k * (k + 1) / 2) % n

/-- The final label we're interested in -/
def final_label : ℕ := 2014

/-- The smallest label we claim to be on the same point as the final label -/
def smallest_label : ℕ := 5

theorem smallest_label_on_final_position :
  position final_label = position smallest_label ∧
  ∀ m : ℕ, m < smallest_label → position final_label ≠ position m :=
sorry

end NUMINAMATH_CALUDE_smallest_label_on_final_position_l2894_289460


namespace NUMINAMATH_CALUDE_end_with_one_piece_l2894_289479

/-- Represents the state of the chessboard -/
structure ChessboardState :=
  (n : ℕ)
  (pieces : ℕ)

/-- Represents a valid move on the chessboard -/
inductive ValidMove : ChessboardState → ChessboardState → Prop
  | jump {s1 s2 : ChessboardState} :
      s1.n = s2.n ∧ s1.pieces = s2.pieces + 1 → ValidMove s1 s2

/-- Represents a sequence of valid moves -/
def ValidMoveSequence : ChessboardState → ChessboardState → Prop :=
  Relation.ReflTransGen ValidMove

/-- The main theorem stating the condition for ending with one piece -/
theorem end_with_one_piece (n : ℕ) :
  (∃ (final : ChessboardState),
    ValidMoveSequence (ChessboardState.mk n (n^2)) final ∧
    final.pieces = 1) ↔ n % 3 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_end_with_one_piece_l2894_289479


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l2894_289476

theorem tan_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (40 * π / 180) + Real.tan (50 * π / 180)) / Real.sin (30 * π / 180) = 
  2 * (Real.cos (40 * π / 180) * Real.cos (50 * π / 180) + 
       Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) / 
      (Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * 
       Real.cos (40 * π / 180) * Real.cos (50 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l2894_289476


namespace NUMINAMATH_CALUDE_trees_that_died_haley_trees_died_l2894_289414

theorem trees_that_died (total : ℕ) (survived_more : ℕ) : ℕ :=
  let died := (total - survived_more) / 2
  died

theorem haley_trees_died : trees_that_died 11 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trees_that_died_haley_trees_died_l2894_289414


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l2894_289497

theorem shaded_area_percentage (large_side small_side : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side = 4)
  (h3 : large_side > 0)
  (h4 : small_side > 0)
  (h5 : small_side < large_side) :
  (large_side^2 - small_side^2) / large_side^2 * 100 = 84 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l2894_289497


namespace NUMINAMATH_CALUDE_shelves_in_closet_l2894_289433

/-- Given the following conditions:
  - There are 12 cans in one row
  - There are 4 rows on one shelf
  - One closet can store 480 cans
  Prove that the number of shelves in one closet is 10. -/
theorem shelves_in_closet (cans_per_row : ℕ) (rows_per_shelf : ℕ) (cans_per_closet : ℕ) 
  (h1 : cans_per_row = 12)
  (h2 : rows_per_shelf = 4)
  (h3 : cans_per_closet = 480) :
  cans_per_closet / (cans_per_row * rows_per_shelf) = 10 := by
  sorry

end NUMINAMATH_CALUDE_shelves_in_closet_l2894_289433


namespace NUMINAMATH_CALUDE_property_price_reduction_l2894_289489

/-- Represents the price reduction scenario of a property over two years -/
theorem property_price_reduction (x : ℝ) : 
  (20000 : ℝ) * (1 - x)^2 = 16200 ↔ 
  (∃ (initial_price final_price : ℝ), 
    initial_price = 20000 ∧ 
    final_price = 16200 ∧ 
    final_price = initial_price * (1 - x)^2 ∧ 
    0 ≤ x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_property_price_reduction_l2894_289489


namespace NUMINAMATH_CALUDE_gala_hat_count_l2894_289492

theorem gala_hat_count (total_attendees : ℕ) 
  (women_fraction : ℚ) (women_hat_percent : ℚ) (men_hat_percent : ℚ)
  (h1 : total_attendees = 2400)
  (h2 : women_fraction = 2/3)
  (h3 : women_hat_percent = 30/100)
  (h4 : men_hat_percent = 12/100) : 
  ↑⌊women_fraction * total_attendees * women_hat_percent⌋ + 
  ↑⌊(1 - women_fraction) * total_attendees * men_hat_percent⌋ = 576 :=
by sorry

end NUMINAMATH_CALUDE_gala_hat_count_l2894_289492


namespace NUMINAMATH_CALUDE_tank_capacity_correct_l2894_289408

/-- Represents the tank filling problem -/
structure TankProblem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ
  cycle_duration : ℕ
  total_time : ℕ

/-- The specific tank problem instance -/
def tankInstance : TankProblem :=
  { capacity := 850,
    pipeA_rate := 40,
    pipeB_rate := 30,
    pipeC_rate := 20,
    cycle_duration := 3,
    total_time := 51 }

/-- Calculates the net amount filled in one cycle -/
def netFillPerCycle (t : TankProblem) : ℕ :=
  t.pipeA_rate + t.pipeB_rate - t.pipeC_rate

/-- Theorem stating that the given tank instance has the correct capacity -/
theorem tank_capacity_correct (t : TankProblem) : 
  t = tankInstance → 
  t.capacity = (t.total_time / t.cycle_duration) * netFillPerCycle t :=
by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_correct_l2894_289408


namespace NUMINAMATH_CALUDE_marias_initial_savings_l2894_289427

def sweater_price : ℕ := 30
def scarf_price : ℕ := 20
def num_sweaters : ℕ := 6
def num_scarves : ℕ := 6
def remaining_money : ℕ := 200

theorem marias_initial_savings :
  (sweater_price * num_sweaters + scarf_price * num_scarves + remaining_money) = 500 := by
  sorry

end NUMINAMATH_CALUDE_marias_initial_savings_l2894_289427


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2894_289432

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2894_289432


namespace NUMINAMATH_CALUDE_number_of_teams_is_twelve_l2894_289415

/-- The number of teams in the baseball league --/
def n : ℕ := sorry

/-- The number of games each team plays against every other team --/
def games_per_pair : ℕ := 6

/-- The total number of games played in the league --/
def total_games : ℕ := 396

/-- Theorem stating that the number of teams in the league is 12 --/
theorem number_of_teams_is_twelve :
  (n * (n - 1) / 2) * games_per_pair = total_games ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_number_of_teams_is_twelve_l2894_289415


namespace NUMINAMATH_CALUDE_costco_mayo_price_l2894_289413

/-- The cost of a gallon of mayo at Costco -/
def costco_gallon_cost : ℚ := 8

/-- The volume of a gallon in ounces -/
def gallon_ounces : ℕ := 128

/-- The volume of a standard bottle in ounces -/
def bottle_ounces : ℕ := 16

/-- The cost of a standard bottle at a normal store -/
def normal_store_bottle_cost : ℚ := 3

/-- The savings when buying at Costco -/
def costco_savings : ℚ := 16

theorem costco_mayo_price :
  costco_gallon_cost = 
    (gallon_ounces / bottle_ounces : ℚ) * normal_store_bottle_cost - costco_savings :=
by sorry

end NUMINAMATH_CALUDE_costco_mayo_price_l2894_289413


namespace NUMINAMATH_CALUDE_rocket_momentum_l2894_289459

/-- Given two rockets with masses m and 9m, subjected to the same constant force F 
    for the same distance d, if the rocket with mass m acquires momentum p, 
    then the rocket with mass 9m acquires momentum 3p. -/
theorem rocket_momentum 
  (m : ℝ) 
  (F : ℝ) 
  (d : ℝ) 
  (p : ℝ) 
  (h1 : m > 0) 
  (h2 : F > 0) 
  (h3 : d > 0) 
  (h4 : p = Real.sqrt (2 * d * m * F)) : 
  9 * m * Real.sqrt ((2 * F * d) / (9 * m)) = 3 * p := by
  sorry

end NUMINAMATH_CALUDE_rocket_momentum_l2894_289459


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2894_289440

/-- The volume of a sphere circumscribing a rectangular solid with dimensions 1, 2, and 3 -/
theorem sphere_volume_circumscribing_rectangular_solid :
  let l : Real := 1  -- length
  let w : Real := 2  -- width
  let h : Real := 3  -- height
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = (7 * Real.sqrt 14 / 3) * Real.pi := by
sorry


end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_solid_l2894_289440


namespace NUMINAMATH_CALUDE_female_fraction_is_four_fifths_l2894_289478

/-- Represents a corporation with male and female employees -/
structure Corporation where
  maleEmployees : ℕ
  femaleEmployees : ℕ

/-- The fraction of employees who are at least 35 years old -/
def atLeast35Fraction (c : Corporation) : ℚ :=
  (0.5 * c.maleEmployees + 0.4 * c.femaleEmployees) / (c.maleEmployees + c.femaleEmployees)

/-- The fraction of employees who are females -/
def femaleFraction (c : Corporation) : ℚ :=
  c.femaleEmployees / (c.maleEmployees + c.femaleEmployees)

theorem female_fraction_is_four_fifths (c : Corporation) 
    (h : atLeast35Fraction c = 0.42) : 
    femaleFraction c = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_female_fraction_is_four_fifths_l2894_289478


namespace NUMINAMATH_CALUDE_xy_value_l2894_289419

theorem xy_value (x y : ℝ) 
  (h1 : (8 : ℝ)^x / (4 : ℝ)^(x + y) = 16)
  (h2 : (27 : ℝ)^(x + y) / (9 : ℝ)^(5 * y) = 729) :
  x * y = 96 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l2894_289419


namespace NUMINAMATH_CALUDE_hex_20F_to_decimal_l2894_289435

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9
| A | B | C | D | E | F

/-- Converts a HexDigit to its decimal value --/
def hexToDecimal (d : HexDigit) : ℕ :=
  match d with
  | HexDigit.D0 => 0 | HexDigit.D1 => 1 | HexDigit.D2 => 2 | HexDigit.D3 => 3
  | HexDigit.D4 => 4 | HexDigit.D5 => 5 | HexDigit.D6 => 6 | HexDigit.D7 => 7
  | HexDigit.D8 => 8 | HexDigit.D9 => 9 | HexDigit.A => 10 | HexDigit.B => 11
  | HexDigit.C => 12 | HexDigit.D => 13 | HexDigit.E => 14 | HexDigit.F => 15

/-- Converts a list of HexDigits to its decimal value --/
def hexListToDecimal (digits : List HexDigit) : ℤ :=
  digits.enum.foldl (fun acc (i, d) => acc + (hexToDecimal d : ℤ) * 16^(digits.length - 1 - i)) 0

/-- The hexadecimal number -20F --/
def hex20F : List HexDigit := [HexDigit.D2, HexDigit.D0, HexDigit.F]

theorem hex_20F_to_decimal :
  -hexListToDecimal hex20F = -527 := by sorry

end NUMINAMATH_CALUDE_hex_20F_to_decimal_l2894_289435


namespace NUMINAMATH_CALUDE_distinct_roots_sum_l2894_289471

theorem distinct_roots_sum (a b c : ℂ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * (a - 6) = 7 →
  b * (b - 6) = 7 →
  c * (c - 6) = 7 →
  a + b + c = 5 := by
sorry

end NUMINAMATH_CALUDE_distinct_roots_sum_l2894_289471


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2894_289468

theorem quadratic_discriminant (a b c : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) →
  |x₂ - x₁| = 2 →
  b^2 - 4*a*c = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2894_289468


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_effect_l2894_289444

theorem price_reduction_sales_increase_effect 
  (original_price original_sales : ℝ) 
  (price_reduction_percent : ℝ) 
  (sales_increase_percent : ℝ) 
  (h1 : price_reduction_percent = 30)
  (h2 : sales_increase_percent = 80) :
  let new_price := original_price * (1 - price_reduction_percent / 100)
  let new_sales := original_sales * (1 + sales_increase_percent / 100)
  let original_revenue := original_price * original_sales
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.26 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_effect_l2894_289444


namespace NUMINAMATH_CALUDE_train_crossing_time_l2894_289436

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 100 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 2 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2894_289436


namespace NUMINAMATH_CALUDE_circle_equation_is_correct_l2894_289448

/-- A circle with center on the y-axis, radius 1, passing through (1, 3) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  point_on_circle : (passes_through.1 - center.1)^2 + (passes_through.2 - center.2)^2 = radius^2

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) (h : c.passes_through = (1, 3)) :
  ∀ x y : ℝ, circle_equation c x y ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_is_correct_l2894_289448


namespace NUMINAMATH_CALUDE_bus_stop_interval_l2894_289455

/-- Proves that the time interval between bus stops is 6 minutes -/
theorem bus_stop_interval (average_speed : ℝ) (total_distance : ℝ) (num_stops : ℕ) 
  (h1 : average_speed = 60)
  (h2 : total_distance = 30)
  (h3 : num_stops = 6) :
  (total_distance / average_speed) * 60 / (num_stops - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_interval_l2894_289455


namespace NUMINAMATH_CALUDE_intersection_and_subset_l2894_289417

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m)*(x - m - 9) < 0}

theorem intersection_and_subset :
  (A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6}) ∧
  (∀ m : ℝ, A ⊆ C m ↔ -3 ≤ m ∧ m ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_subset_l2894_289417


namespace NUMINAMATH_CALUDE_geometric_sequence_312th_term_l2894_289452

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a₁ : ℝ  -- first term
  r : ℝ   -- common ratio
  
/-- The nth term of a geometric sequence -/
def GeometricSequence.nthTerm (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a₁ * seq.r ^ (n - 1)

/-- Theorem: The 312th term of the specific geometric sequence -/
theorem geometric_sequence_312th_term :
  let seq : GeometricSequence := { a₁ := 12, r := -1/2 }
  seq.nthTerm 312 = -12 * (1/2)^311 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_312th_term_l2894_289452


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2894_289488

theorem complex_equation_solution (z : ℂ) : (z - 1) / (z + 1) = I → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2894_289488


namespace NUMINAMATH_CALUDE_airline_route_theorem_l2894_289410

/-- Represents a city in the country -/
structure City where
  id : Nat
  republic : Nat
  routes : Finset Nat

/-- The country with its cities and airline routes -/
structure Country where
  cities : Finset City
  total_cities : Nat
  num_republics : Nat

/-- A country satisfies the problem conditions -/
def satisfies_conditions (country : Country) : Prop :=
  country.total_cities = 100 ∧
  country.num_republics = 3 ∧
  (country.cities.filter (λ c => c.routes.card ≥ 70)).card ≥ 70

/-- There exists an airline route within the same republic -/
def exists_intra_republic_route (country : Country) : Prop :=
  ∃ c1 c2 : City, c1 ∈ country.cities ∧ c2 ∈ country.cities ∧
    c1.id ≠ c2.id ∧ c1.republic = c2.republic ∧ c2.id ∈ c1.routes

/-- The main theorem -/
theorem airline_route_theorem (country : Country) :
  satisfies_conditions country → exists_intra_republic_route country :=
by
  sorry


end NUMINAMATH_CALUDE_airline_route_theorem_l2894_289410


namespace NUMINAMATH_CALUDE_total_campers_rowing_l2894_289474

theorem total_campers_rowing (morning afternoon evening : ℕ) 
  (h1 : morning = 36) 
  (h2 : afternoon = 13) 
  (h3 : evening = 49) : 
  morning + afternoon + evening = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_l2894_289474


namespace NUMINAMATH_CALUDE_salary_ratio_degree_to_diploma_l2894_289483

/-- Represents the monthly salary of a diploma holder in dollars. -/
def diploma_monthly_salary : ℕ := 4000

/-- Represents the annual salary of a degree holder in dollars. -/
def degree_annual_salary : ℕ := 144000

/-- Represents the number of months in a year. -/
def months_per_year : ℕ := 12

/-- Theorem stating that the ratio of annual salaries between degree and diploma holders is 3:1. -/
theorem salary_ratio_degree_to_diploma :
  (degree_annual_salary : ℚ) / (diploma_monthly_salary * months_per_year) = 3 := by
  sorry

#check salary_ratio_degree_to_diploma

end NUMINAMATH_CALUDE_salary_ratio_degree_to_diploma_l2894_289483


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2894_289472

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x > 2 ∨ x < -1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2894_289472


namespace NUMINAMATH_CALUDE_symmetric_points_imply_sum_power_l2894_289469

-- Define the points P and Q
def P (m n : ℝ) : ℝ × ℝ := (m - 1, n + 2)
def Q (m : ℝ) : ℝ × ℝ := (2 * m - 4, 2)

-- Define the symmetry condition
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

-- Theorem statement
theorem symmetric_points_imply_sum_power (m n : ℝ) :
  symmetric_x_axis (P m n) (Q m) → (m + n)^2023 = -1 := by
  sorry

#check symmetric_points_imply_sum_power

end NUMINAMATH_CALUDE_symmetric_points_imply_sum_power_l2894_289469


namespace NUMINAMATH_CALUDE_courtier_cycle_odd_l2894_289434

/-- Represents a directed cycle graph of courtiers -/
structure CourtierCycle where
  n : ℕ
  next : Fin n → Fin n
  cycle : ∀ i : Fin n, (next^[n] i) = i

/-- Theorem: The number of courtiers in a CourtierCycle is odd -/
theorem courtier_cycle_odd (c : CourtierCycle) : Odd c.n := by
  sorry

end NUMINAMATH_CALUDE_courtier_cycle_odd_l2894_289434


namespace NUMINAMATH_CALUDE_points_collinear_opposite_collinear_k_l2894_289409

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-zero vectors a and b
variable (a b : V)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hnc : ¬ ∃ (r : ℝ), a = r • b)

-- Define vectors AB, BC, and CD
def AB : V := a + b
def BC : V := 2 • a + 8 • b
def CD : V := 3 • (a - b)

-- Define collinearity
def collinear (u v : V) : Prop := ∃ (r : ℝ), u = r • v

-- Theorem 1: Points A, B, D are collinear
theorem points_collinear : 
  ∃ (r : ℝ), AB a b = r • (AB a b + BC a b + CD a b) :=
sorry

-- Theorem 2: Value of k for opposite collinearity
theorem opposite_collinear_k : 
  ∃ (k : ℝ), k = -1 ∧ 
  (∃ (r : ℝ), r < 0 ∧ k • a + b = r • (a + k • b)) :=
sorry

end NUMINAMATH_CALUDE_points_collinear_opposite_collinear_k_l2894_289409


namespace NUMINAMATH_CALUDE_cube_root_27_times_sixth_root_64_times_sqrt_9_l2894_289429

theorem cube_root_27_times_sixth_root_64_times_sqrt_9 :
  (27 : ℝ) ^ (1/3) * (64 : ℝ) ^ (1/6) * (9 : ℝ) ^ (1/2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_27_times_sixth_root_64_times_sqrt_9_l2894_289429


namespace NUMINAMATH_CALUDE_inverse_f_at_135_l2894_289496

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^3 + 5

-- State the theorem
theorem inverse_f_at_135 :
  ∃ (y : ℝ), f y = 135 ∧ y = (26 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_135_l2894_289496


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_and_c_are_2_l2894_289490

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions given in the problem
def satisfiesCondition1 (t : Triangle) : Prop :=
  t.a^2 - t.c^2 = t.b^2 - t.b * t.c

def satisfiesCondition2 (t : Triangle) : Prop :=
  t.a = 2 ∧ t.b + t.c = 4

-- Theorem 1: If the first condition is satisfied, then angle A is 60°
theorem angle_A_is_60_degrees (t : Triangle) (h : satisfiesCondition1 t) :
  t.A = 60 * (π / 180) := by sorry

-- Theorem 2: If both conditions are satisfied, then b = 2 and c = 2
theorem b_and_c_are_2 (t : Triangle) (h1 : satisfiesCondition1 t) (h2 : satisfiesCondition2 t) :
  t.b = 2 ∧ t.c = 2 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_b_and_c_are_2_l2894_289490


namespace NUMINAMATH_CALUDE_simplify_expression_l2894_289411

theorem simplify_expression : 5 * (18 / 6) * (21 / -63) = -5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2894_289411


namespace NUMINAMATH_CALUDE_fred_dimes_l2894_289445

/-- Proves that if Fred has 90 cents and each dime is worth 10 cents, then Fred has 9 dimes -/
theorem fred_dimes (total_cents : ℕ) (dime_value : ℕ) (h1 : total_cents = 90) (h2 : dime_value = 10) :
  total_cents / dime_value = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_dimes_l2894_289445


namespace NUMINAMATH_CALUDE_want_is_correct_choice_l2894_289493

/-- Represents the possible word choices for the sentence --/
inductive WordChoice
  | hope
  | search
  | want
  | charge

/-- Represents the context of the situation --/
structure Situation where
  duration : Nat
  location : String
  isSnowstorm : Bool
  lackOfSupplies : Bool

/-- Defines the correct word choice given a situation --/
def correctWordChoice (s : Situation) : WordChoice :=
  if s.duration ≥ 5 && s.location = "station" && s.isSnowstorm && s.lackOfSupplies then
    WordChoice.want
  else
    WordChoice.hope  -- Default choice, not relevant for this problem

/-- Theorem stating that 'want' is the correct word choice for the given situation --/
theorem want_is_correct_choice (s : Situation) 
  (h1 : s.duration = 5)
  (h2 : s.location = "station")
  (h3 : s.isSnowstorm = true)
  (h4 : s.lackOfSupplies = true) :
  correctWordChoice s = WordChoice.want := by
  sorry


end NUMINAMATH_CALUDE_want_is_correct_choice_l2894_289493


namespace NUMINAMATH_CALUDE_inverse_sum_equals_root_difference_l2894_289442

-- Define the function g
def g (x : ℝ) : ℝ := x^3 * |x|

-- State the theorem
theorem inverse_sum_equals_root_difference :
  (∃ y₁ : ℝ, g y₁ = 8) ∧ (∃ y₂ : ℝ, g y₂ = -125) →
  (∃ y₁ y₂ : ℝ, g y₁ = 8 ∧ g y₂ = -125 ∧ y₁ + y₂ = 2^(1/2) - 5^(3/4)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_root_difference_l2894_289442


namespace NUMINAMATH_CALUDE_smallest_integer_in_special_set_l2894_289461

theorem smallest_integer_in_special_set : ∃ (n : ℤ),
  (n + 6 > 2 * ((7 * n + 21) / 7)) ∧
  (∀ (m : ℤ), m < n → ¬(m + 6 > 2 * ((7 * m + 21) / 7))) →
  n = -1 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_special_set_l2894_289461


namespace NUMINAMATH_CALUDE_intersection_M_N_l2894_289446

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2894_289446


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l2894_289447

/-- Calculate the cost of plastering a tank's walls and bottom -/
theorem tank_plastering_cost 
  (length width depth : ℝ)
  (cost_per_sqm_paise : ℝ)
  (h_length : length = 25)
  (h_width : width = 12)
  (h_depth : depth = 6)
  (h_cost : cost_per_sqm_paise = 75) :
  let surface_area := 2 * (length * depth + width * depth) + length * width
  let cost_rupees := surface_area * (cost_per_sqm_paise / 100)
  cost_rupees = 558 := by
sorry

end NUMINAMATH_CALUDE_tank_plastering_cost_l2894_289447


namespace NUMINAMATH_CALUDE_marble_solution_l2894_289487

/-- Represents the number of marbles each person has -/
structure Marbles where
  selma : ℕ
  merill : ℕ
  elliot : ℕ
  vivian : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  m.selma = 50 ∧
  m.merill = 2 * m.elliot ∧
  m.merill + m.elliot = m.selma - 5 ∧
  m.vivian = m.merill + m.elliot + 10

/-- The theorem stating the solution to the marble problem -/
theorem marble_solution (m : Marbles) (h : marble_conditions m) : 
  m.merill = 30 ∧ m.vivian = 55 := by
  sorry

#check marble_solution

end NUMINAMATH_CALUDE_marble_solution_l2894_289487


namespace NUMINAMATH_CALUDE_product_lower_bound_l2894_289425

theorem product_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) :
  a * b ≥ (7 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_lower_bound_l2894_289425


namespace NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l2894_289450

theorem modular_inverse_35_mod_36 : ∃ x : ℤ, (35 * x) % 36 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_35_mod_36_l2894_289450


namespace NUMINAMATH_CALUDE_therapy_cost_difference_l2894_289491

/-- Represents the pricing scheme of a psychologist -/
structure PricingScheme where
  firstHourCost : ℕ
  additionalHourCost : ℕ
  first_hour_more_expensive : firstHourCost > additionalHourCost

/-- Theorem: Given the conditions, the difference in cost between the first hour
    and each additional hour is $30 -/
theorem therapy_cost_difference (p : PricingScheme) 
  (five_hour_cost : p.firstHourCost + 4 * p.additionalHourCost = 400)
  (three_hour_cost : p.firstHourCost + 2 * p.additionalHourCost = 252) :
  p.firstHourCost - p.additionalHourCost = 30 := by
  sorry

end NUMINAMATH_CALUDE_therapy_cost_difference_l2894_289491


namespace NUMINAMATH_CALUDE_A_3_2_l2894_289449

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_A_3_2_l2894_289449


namespace NUMINAMATH_CALUDE_carl_driving_hours_l2894_289412

/-- The number of hours Carl drives per day before promotion -/
def hours_per_day : ℝ :=
  2

/-- The number of additional hours Carl drives per week after promotion -/
def additional_hours_per_week : ℝ :=
  6

/-- The number of hours Carl drives in two weeks after promotion -/
def hours_in_two_weeks_after : ℝ :=
  40

/-- The number of days in two weeks -/
def days_in_two_weeks : ℝ :=
  14

theorem carl_driving_hours :
  hours_per_day * days_in_two_weeks + additional_hours_per_week * 2 = hours_in_two_weeks_after :=
by sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l2894_289412


namespace NUMINAMATH_CALUDE_usb_drive_available_space_l2894_289484

theorem usb_drive_available_space (total_capacity : ℝ) (used_percentage : ℝ) 
  (h1 : total_capacity = 16)
  (h2 : used_percentage = 50)
  : total_capacity * (1 - used_percentage / 100) = 8 := by
  sorry

end NUMINAMATH_CALUDE_usb_drive_available_space_l2894_289484
