import Mathlib

namespace distance_swum_against_current_l33_3398

/-- The distance swum against the current given swimming speed, current speed, and time taken -/
theorem distance_swum_against_current 
  (swimming_speed : ℝ) 
  (current_speed : ℝ) 
  (time_taken : ℝ) 
  (h1 : swimming_speed = 4)
  (h2 : current_speed = 2)
  (h3 : time_taken = 6) : 
  (swimming_speed - current_speed) * time_taken = 12 := by
  sorry

#check distance_swum_against_current

end distance_swum_against_current_l33_3398


namespace absent_present_probability_l33_3317

theorem absent_present_probability (course_days : ℕ) (avg_absent_days : ℕ) : 
  course_days = 40 → 
  avg_absent_days = 1 → 
  (39 : ℚ) / 800 = (course_days - avg_absent_days) / (course_days^2) * 2 := by
  sorry

end absent_present_probability_l33_3317


namespace brad_daily_reading_l33_3385

/-- Brad's daily reading in pages -/
def brad_pages : ℕ := 26

/-- Greg's daily reading in pages -/
def greg_pages : ℕ := 18

/-- The difference in pages read between Brad and Greg -/
def page_difference : ℕ := 8

theorem brad_daily_reading :
  brad_pages = greg_pages + page_difference :=
by sorry

end brad_daily_reading_l33_3385


namespace right_triangle_7_24_25_l33_3347

theorem right_triangle_7_24_25 : 
  ∀ (a b c : ℝ), a = 7 ∧ b = 24 ∧ c = 25 → a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_7_24_25_l33_3347


namespace peanut_butter_jar_size_l33_3356

theorem peanut_butter_jar_size (total_ounces : ℕ) (jar_size_1 jar_size_3 : ℕ) (total_jars : ℕ) :
  total_ounces = 252 →
  jar_size_1 = 16 →
  jar_size_3 = 40 →
  total_jars = 9 →
  ∃ (jar_size_2 : ℕ),
    jar_size_2 = 28 ∧
    total_ounces = (total_jars / 3) * (jar_size_1 + jar_size_2 + jar_size_3) :=
by sorry

end peanut_butter_jar_size_l33_3356


namespace boatworks_production_l33_3333

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boatworks_production : geometric_sum 5 3 6 = 1820 := by
  sorry

end boatworks_production_l33_3333


namespace least_positive_integer_for_reducible_fraction_l33_3344

theorem least_positive_integer_for_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(∃ (k : ℕ), k > 1 ∧ k ∣ (m - 10) ∧ k ∣ (9*m + 11))) ∧
  (∃ (k : ℕ), k > 1 ∧ k ∣ (n - 10) ∧ k ∣ (9*n + 11)) ∧
  n = 111 :=
by sorry

end least_positive_integer_for_reducible_fraction_l33_3344


namespace books_sold_on_thursday_l33_3345

theorem books_sold_on_thursday (initial_stock : ℕ) (sold_monday : ℕ) (sold_tuesday : ℕ)
  (sold_wednesday : ℕ) (sold_friday : ℕ) (unsold : ℕ) :
  initial_stock = 800 →
  sold_monday = 60 →
  sold_tuesday = 10 →
  sold_wednesday = 20 →
  sold_friday = 66 →
  unsold = 600 →
  initial_stock - (sold_monday + sold_tuesday + sold_wednesday + sold_friday + unsold) = 44 :=
by sorry

end books_sold_on_thursday_l33_3345


namespace horner_method_f_2_l33_3303

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem horner_method_f_2 : f 2 = 62 := by
  sorry

end horner_method_f_2_l33_3303


namespace solve_transportation_problem_l33_3337

/-- Represents the daily transportation problem for building materials -/
structure TransportationProblem where
  daily_requirement : ℕ
  max_supply_A : ℕ
  max_supply_B : ℕ
  cost_scenario1 : ℕ
  cost_scenario2 : ℕ

/-- Represents the solution to the transportation problem -/
structure TransportationSolution where
  cost_per_ton_A : ℕ
  cost_per_ton_B : ℕ
  min_total_cost : ℕ
  optimal_tons_A : ℕ
  optimal_tons_B : ℕ

/-- Theorem stating the solution to the transportation problem -/
theorem solve_transportation_problem (p : TransportationProblem) 
  (h1 : p.daily_requirement = 120)
  (h2 : p.max_supply_A = 80)
  (h3 : p.max_supply_B = 90)
  (h4 : p.cost_scenario1 = 26000)
  (h5 : p.cost_scenario2 = 27000) :
  ∃ (s : TransportationSolution),
    s.cost_per_ton_A = 240 ∧
    s.cost_per_ton_B = 200 ∧
    s.min_total_cost = 25200 ∧
    s.optimal_tons_A = 30 ∧
    s.optimal_tons_B = 90 ∧
    s.optimal_tons_A + s.optimal_tons_B = p.daily_requirement ∧
    s.optimal_tons_A ≤ p.max_supply_A ∧
    s.optimal_tons_B ≤ p.max_supply_B ∧
    s.min_total_cost = s.cost_per_ton_A * s.optimal_tons_A + s.cost_per_ton_B * s.optimal_tons_B :=
by
  sorry


end solve_transportation_problem_l33_3337


namespace initial_capacity_proof_l33_3357

/-- The daily processing capacity of each machine before modernization. -/
def initial_capacity : ℕ := 1215

/-- The number of machines before modernization. -/
def initial_machines : ℕ := 32

/-- The daily processing capacity of each machine after modernization. -/
def new_capacity : ℕ := 1280

/-- The number of machines after modernization. -/
def new_machines : ℕ := initial_machines + 3

/-- The total daily processing before modernization. -/
def total_before : ℕ := 38880

/-- The total daily processing after modernization. -/
def total_after : ℕ := 44800

theorem initial_capacity_proof :
  initial_capacity * initial_machines = total_before ∧
  new_capacity * new_machines = total_after ∧
  initial_capacity < new_capacity :=
by sorry

end initial_capacity_proof_l33_3357


namespace clock_rotation_impossibility_l33_3377

/-- Represents a clock face with 12 numbers -/
def ClockFace : Type := Fin 12

/-- The sum of all numbers on the clock face -/
def clockSum : ℕ := (List.range 12).sum + 12

/-- The target number to be achieved on all positions of the blackboard -/
def target : ℕ := 1984

/-- The number of positions on the clock face and blackboard -/
def numPositions : ℕ := 12

theorem clock_rotation_impossibility : 
  ¬ ∃ (n : ℕ), n * clockSum = numPositions * target := by
  sorry

end clock_rotation_impossibility_l33_3377


namespace other_root_of_quadratic_l33_3315

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = -5) → 
  (3 : ℝ) ∈ {x : ℝ | 3 * x^2 + k * x = -5} → 
  (5/9 : ℝ) ∈ {x : ℝ | 3 * x^2 + k * x = -5} :=
by sorry

end other_root_of_quadratic_l33_3315


namespace line_segment_endpoint_l33_3325

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  Real.sqrt ((x - 2)^2 + (7 - 1)^2) = 8 → 
  x = 2 + 2 * Real.sqrt 7 := by
sorry

end line_segment_endpoint_l33_3325


namespace incircle_radius_inscribed_triangle_l33_3359

theorem incircle_radius_inscribed_triangle (r : ℝ) (α β γ : ℝ) (h1 : 0 < r) (h2 : 0 < α) (h3 : 0 < β) (h4 : 0 < γ) 
  (h5 : α + β + γ = π) (h6 : Real.tan α = 1/3) (h7 : Real.sin β * Real.sin γ = 1/Real.sqrt 10) : 
  ∃ ρ : ℝ, ρ = (r * Real.sqrt 10) / (1 + Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end incircle_radius_inscribed_triangle_l33_3359


namespace knight_reachability_l33_3339

/-- Represents a position on an infinite chessboard -/
structure Position where
  x : Int
  y : Int

/-- Represents a knight's move -/
inductive KnightMove (n : Nat)
  | horizontal : KnightMove n
  | vertical   : KnightMove n

/-- Applies a knight's move to a position -/
def applyMove (n : Nat) (p : Position) (m : KnightMove n) : Position :=
  match m with
  | KnightMove.horizontal => ⟨p.x + n, p.y + 1⟩
  | KnightMove.vertical   => ⟨p.x + 1, p.y + n⟩

/-- Defines reachability for a knight -/
def isReachable (n : Nat) (start finish : Position) : Prop :=
  ∃ (moves : List (KnightMove n)), finish = moves.foldl (applyMove n) start

/-- The main theorem: A knight can reach any position iff n is even -/
theorem knight_reachability (n : Nat) :
  (∀ (start finish : Position), isReachable n start finish) ↔ Even n := by
  sorry


end knight_reachability_l33_3339


namespace linear_function_not_in_third_quadrant_graph_not_in_third_quadrant_l33_3393

/-- A linear function f(x) = kx + b does not pass through the third quadrant
    if and only if k < 0 and b > 0 -/
theorem linear_function_not_in_third_quadrant (k b : ℝ) :
  k < 0 ∧ b > 0 → ∀ x y : ℝ, y = k * x + b → ¬(x < 0 ∧ y < 0) := by
  sorry

/-- The graph of y = -2x + 1 does not pass through the third quadrant -/
theorem graph_not_in_third_quadrant :
  ∀ x y : ℝ, y = -2 * x + 1 → ¬(x < 0 ∧ y < 0) := by
  sorry

end linear_function_not_in_third_quadrant_graph_not_in_third_quadrant_l33_3393


namespace pair_five_cows_four_pigs_seven_horses_l33_3372

/-- The number of ways to pair animals of different species -/
def pairAnimals (cows pigs horses : ℕ) : ℕ :=
  cows * pigs * (cows + pigs - 2).factorial

/-- Theorem stating the number of ways to pair 5 cows, 4 pigs, and 7 horses -/
theorem pair_five_cows_four_pigs_seven_horses :
  pairAnimals 5 4 7 = 100800 := by
  sorry

#eval pairAnimals 5 4 7

end pair_five_cows_four_pigs_seven_horses_l33_3372


namespace gcd_of_30_and_45_l33_3348

theorem gcd_of_30_and_45 : Nat.gcd 30 45 = 15 := by
  sorry

end gcd_of_30_and_45_l33_3348


namespace find_q_l33_3389

theorem find_q (a b m p q : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ((a + 2/b)^2 - p*(a + 2/b) + q = 0) → 
  ((b + 2/a)^2 - p*(b + 2/a) + q = 0) → 
  q = 25/3 :=
by sorry

end find_q_l33_3389


namespace cubic_sum_over_product_equals_three_l33_3336

theorem cubic_sum_over_product_equals_three
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_sum : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := by
sorry

end cubic_sum_over_product_equals_three_l33_3336


namespace least_addition_for_divisibility_problem_solution_l33_3369

theorem least_addition_for_divisibility (n m : ℕ) : 
  ∃ x : ℕ, x ≤ m - 1 ∧ (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ x : ℕ, x ≤ 22 ∧ (1054 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1054 + y) % 23 ≠ 0 ∧ x = 4 :=
by sorry

end least_addition_for_divisibility_problem_solution_l33_3369


namespace no_solution_iff_m_geq_two_thirds_l33_3322

theorem no_solution_iff_m_geq_two_thirds (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2*m < 0 ∧ x + m > 2)) ↔ m ≥ 2/3 :=
by sorry

end no_solution_iff_m_geq_two_thirds_l33_3322


namespace largest_not_expressible_l33_3342

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ k c, k > 0 ∧ is_composite c ∧ n = 37 * k + c

theorem largest_not_expressible :
  (∀ n > 66, is_expressible n) ∧ ¬is_expressible 66 :=
sorry

end largest_not_expressible_l33_3342


namespace compare_expressions_l33_3300

theorem compare_expressions : (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) := by
  sorry

end compare_expressions_l33_3300


namespace sum_of_four_consecutive_values_l33_3366

-- Define the properties of the function f
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_two_property (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) = -f x

-- State the theorem
theorem sum_of_four_consecutive_values (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period : has_period_two_property f) : 
  f 2008 + f 2009 + f 2010 + f 2011 = 0 := by
  sorry

end sum_of_four_consecutive_values_l33_3366


namespace candy_calculation_correct_l33_3381

/-- Calculates the number of candy pieces Haley's sister gave her. -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Proves that the calculation of candy pieces from Haley's sister is correct. -/
theorem candy_calculation_correct (initial eaten final : ℕ) 
  (h1 : initial ≥ eaten) 
  (h2 : final ≥ initial - eaten) : 
  candy_from_sister initial eaten final = final - (initial - eaten) :=
by sorry

end candy_calculation_correct_l33_3381


namespace arccos_negative_one_equals_pi_l33_3382

theorem arccos_negative_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end arccos_negative_one_equals_pi_l33_3382


namespace transformation_theorem_l33_3383

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the transformation g
def g : ℝ → ℝ := sorry

-- Theorem statement
theorem transformation_theorem :
  ∀ x : ℝ, g x = -f (x - 3) := by
  sorry

end transformation_theorem_l33_3383


namespace polynomial_division_theorem_l33_3364

theorem polynomial_division_theorem (x : ℝ) :
  let dividend := 12 * x^3 + 20 * x^2 - 7 * x + 4
  let divisor := 3 * x + 4
  let quotient := 4 * x^2 + (4/3) * x - 37/9
  let remainder := 74/9
  dividend = divisor * quotient + remainder := by sorry

end polynomial_division_theorem_l33_3364


namespace second_to_first_layer_ratio_l33_3386

/-- Given a three-layer cake recipe, this theorem proves the ratio of the second layer to the first layer. -/
theorem second_to_first_layer_ratio 
  (sugar_first_layer : ℝ) 
  (sugar_third_layer : ℝ) 
  (third_to_second_ratio : ℝ) 
  (h1 : sugar_first_layer = 2)
  (h2 : sugar_third_layer = 12)
  (h3 : third_to_second_ratio = 3) :
  (sugar_third_layer / sugar_first_layer) / third_to_second_ratio = 2 := by
  sorry

end second_to_first_layer_ratio_l33_3386


namespace worm_distance_after_15_days_l33_3328

/-- Represents the daily movement of a worm -/
structure WormMovement where
  forward : ℝ
  backward : ℝ

/-- Calculates the net daily distance traveled by the worm -/
def net_daily_distance (movement : WormMovement) : ℝ :=
  movement.forward - movement.backward

/-- Calculates the total distance traveled over a number of days -/
def total_distance (movement : WormMovement) (days : ℕ) : ℝ :=
  (net_daily_distance movement) * days

/-- The theorem to be proved -/
theorem worm_distance_after_15_days (worm_movement : WormMovement)
    (h1 : worm_movement.forward = 5)
    (h2 : worm_movement.backward = 3)
    : total_distance worm_movement 15 = 30 := by
  sorry

end worm_distance_after_15_days_l33_3328


namespace ExistsSpecialSequence_l33_3312

-- Define the sequence type
def InfiniteSequence := ℕ → ℕ

-- Define the properties of the sequence
def NoDivisibility (seq : InfiniteSequence) :=
  ∀ i j, i ≠ j → ¬(seq i ∣ seq j)

def CommonDivisorGreaterThanOne (seq : InfiniteSequence) :=
  ∀ i j, i ≠ j → ∃ k, k > 1 ∧ k ∣ seq i ∧ k ∣ seq j

def NoCommonDivisorGreaterThanOne (seq : InfiniteSequence) :=
  ¬∃ k, k > 1 ∧ (∀ i, k ∣ seq i)

-- Main theorem
theorem ExistsSpecialSequence :
  ∃ seq : InfiniteSequence,
    NoDivisibility seq ∧
    CommonDivisorGreaterThanOne seq ∧
    NoCommonDivisorGreaterThanOne seq :=
by sorry


end ExistsSpecialSequence_l33_3312


namespace p_q_ratio_equals_ways_ratio_l33_3330

/-- The number of balls -/
def n : ℕ := 20

/-- The number of bins -/
def k : ℕ := 4

/-- The probability of a 3-5-6-6 distribution -/
def p : ℚ := sorry

/-- The probability of a 5-5-5-5 distribution -/
def q : ℚ := sorry

/-- The number of ways to distribute n balls into k bins with a given distribution -/
def ways_to_distribute (n : ℕ) (k : ℕ) (distribution : List ℕ) : ℕ := sorry

/-- The ratio of p to q is equal to the ratio of the number of ways to achieve each distribution -/
theorem p_q_ratio_equals_ways_ratio : 
  p / q = (ways_to_distribute n k [3, 5, 6, 6] * 12) / ways_to_distribute n k [5, 5, 5, 5] := by
  sorry

end p_q_ratio_equals_ways_ratio_l33_3330


namespace rectangle_with_perpendicular_diagonals_is_square_l33_3316

-- Define a rectangle
structure Rectangle :=
  (a b : ℝ)
  (a_positive : a > 0)
  (b_positive : b > 0)

-- Define a property for perpendicular diagonals
def has_perpendicular_diagonals (r : Rectangle) : Prop :=
  r.a^2 = r.b^2

-- Define a square as a special case of rectangle
def is_square (r : Rectangle) : Prop :=
  r.a = r.b

-- Theorem statement
theorem rectangle_with_perpendicular_diagonals_is_square 
  (r : Rectangle) (h : has_perpendicular_diagonals r) : 
  is_square r :=
sorry

end rectangle_with_perpendicular_diagonals_is_square_l33_3316


namespace parallelepiped_surface_area_l33_3334

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  a : ℝ  -- First diagonal of the base
  b : ℝ  -- Second diagonal of the base
  sphere_inscribed : Bool  -- Indicator that a sphere is inscribed

-- Define the total surface area function
def total_surface_area (p : RectangularParallelepiped) : ℝ :=
  3 * p.a * p.b

-- Theorem statement
theorem parallelepiped_surface_area 
  (p : RectangularParallelepiped) 
  (h : p.sphere_inscribed = true) :
  total_surface_area p = 3 * p.a * p.b :=
by
  sorry


end parallelepiped_surface_area_l33_3334


namespace problem_solution_l33_3367

theorem problem_solution (x : ℂ) (h : x + 1/x = -1) : x^1994 + 1/x^1994 = -1 := by
  sorry

end problem_solution_l33_3367


namespace marble_difference_l33_3368

theorem marble_difference (pink orange purple : ℕ) : 
  pink = 13 →
  orange < pink →
  purple = 4 * orange →
  pink + orange + purple = 33 →
  pink - orange = 9 := by
sorry

end marble_difference_l33_3368


namespace polynomial_multiplication_l33_3332

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 50*x^2 + 625) * (x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := by
  sorry

end polynomial_multiplication_l33_3332


namespace movie_ticket_cost_l33_3340

/-- Theorem: Movie Ticket Cost
  Given:
  - Movie tickets cost M on Monday
  - Wednesday tickets cost 2M
  - Saturday tickets cost 5M
  - Total cost for Wednesday and Saturday is $35
  Prove: M = 5
-/
theorem movie_ticket_cost (M : ℚ) : 2 * M + 5 * M = 35 → M = 5 := by
  sorry

end movie_ticket_cost_l33_3340


namespace vector_b_values_l33_3319

/-- Given two vectors a and b in ℝ², where a = (2,1), |b| = 2√5, and a is parallel to b,
    prove that b is either (-4,-2) or (4,2) -/
theorem vector_b_values (a b : ℝ × ℝ) : 
  a = (2, 1) → 
  ‖b‖ = 2 * Real.sqrt 5 →
  ∃ (k : ℝ), b = k • a →
  b = (-4, -2) ∨ b = (4, 2) := by
sorry

end vector_b_values_l33_3319


namespace zoo_trip_remainder_is_24_l33_3310

/-- Calculates the amount left for lunch and snacks after a zoo trip for two people -/
def zoo_trip_remainder (zoo_ticket_price : ℚ) (bus_fare_one_way : ℚ) (total_money : ℚ) : ℚ :=
  let zoo_cost := 2 * zoo_ticket_price
  let bus_cost := 2 * 2 * bus_fare_one_way
  total_money - (zoo_cost + bus_cost)

/-- Theorem: Given the specified prices and total money, the remainder for lunch and snacks is $24 -/
theorem zoo_trip_remainder_is_24 :
  zoo_trip_remainder 5 1.5 40 = 24 := by
  sorry

end zoo_trip_remainder_is_24_l33_3310


namespace sum_of_squares_l33_3341

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 3 → 
  (a - 1)^3 + (b - 1)^3 + (c - 1)^3 = 0 → 
  a = 2 → 
  a^2 + b^2 + c^2 = 5 := by
sorry

end sum_of_squares_l33_3341


namespace cone_slant_height_l33_3313

/-- The slant height of a cone given its base radius and curved surface area -/
theorem cone_slant_height (r : ℝ) (csa : ℝ) (h1 : r = 5) (h2 : csa = 157.07963267948966) :
  csa / (Real.pi * r) = 10 := by
  sorry

end cone_slant_height_l33_3313


namespace rectangle_ratio_theorem_l33_3374

theorem rectangle_ratio_theorem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_le_b : a ≤ b) :
  (a / b = (a + b) / Real.sqrt (a^2 + b^2)) →
  (a / b)^2 = (Real.sqrt 5 - 1) / 2 := by
  sorry

end rectangle_ratio_theorem_l33_3374


namespace unique_special_number_l33_3387

/-- A three-digit number is represented by its digits a, b, and c -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a > 0
  h2 : a ≤ 9
  h3 : b ≤ 9
  h4 : c ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- A three-digit number is special if it equals 11 times the sum of its digits -/
def isSpecial (n : ThreeDigitNumber) : Prop :=
  value n = 11 * digitSum n

theorem unique_special_number :
  ∃! n : ThreeDigitNumber, isSpecial n ∧ value n = 198 :=
sorry

end unique_special_number_l33_3387


namespace repeating_decimal_sum_l33_3375

theorem repeating_decimal_sum (a b : ℕ) : 
  (5 : ℚ) / 13 = (a * 10 + b : ℚ) / 99 → a + b = 11 := by
  sorry

end repeating_decimal_sum_l33_3375


namespace unique_solution_linear_equation_l33_3329

theorem unique_solution_linear_equation (a b c : ℝ) (h1 : c ≠ 0) (h2 : b ≠ 2) :
  ∃! x : ℝ, 4 * x - 7 + a = 2 * b * x + c ∧ x = (c + 7 - a) / (4 - 2 * b) :=
by sorry

end unique_solution_linear_equation_l33_3329


namespace chemistry_is_other_subject_l33_3321

/-- Represents the scores in three subjects -/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The conditions of the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.physics = 110 ∧
  (s.physics + s.chemistry + s.mathematics) / 3 = 70 ∧
  (s.physics + s.mathematics) / 2 = 90 ∧
  (s.physics + s.chemistry) / 2 = 70

/-- The theorem to be proved -/
theorem chemistry_is_other_subject (s : Scores) :
  satisfiesConditions s → (s.physics + s.chemistry) / 2 = 70 := by
  sorry

end chemistry_is_other_subject_l33_3321


namespace max_rectangle_area_l33_3358

/-- Represents the length of a wire segment between two marks -/
def segment_length : ℕ := 3

/-- Represents the total length of the wire -/
def wire_length : ℕ := 78

/-- Represents the total number of segments in the wire -/
def total_segments : ℕ := wire_length / segment_length

/-- Represents the perimeter of the rectangle in terms of segments -/
def perimeter_segments : ℕ := total_segments / 2

/-- Calculates the area of a rectangle given its length and width in segments -/
def rectangle_area (length width : ℕ) : ℕ :=
  (length * segment_length) * (width * segment_length)

/-- Theorem stating that the maximum area of the rectangle is 378 square centimeters -/
theorem max_rectangle_area :
  (∃ length width : ℕ,
    length + width = perimeter_segments ∧
    rectangle_area length width = 378 ∧
    ∀ l w : ℕ, l + w = perimeter_segments → rectangle_area l w ≤ 378) :=
by sorry

end max_rectangle_area_l33_3358


namespace negation_of_existence_negation_of_specific_proposition_l33_3331

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, x < 0 ∧ P x) ↔ (∀ x, x < 0 → ¬ P x) :=
by sorry

-- The specific proposition
def proposition (x : ℝ) : Prop := 3 * x < 4 * x

theorem negation_of_specific_proposition :
  (¬ ∃ x, x < 0 ∧ proposition x) ↔ (∀ x, x < 0 → 3 * x ≥ 4 * x) :=
by sorry

end negation_of_existence_negation_of_specific_proposition_l33_3331


namespace min_value_reciprocal_sum_l33_3391

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) :
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_value_reciprocal_sum_l33_3391


namespace paving_stone_width_l33_3399

/-- Proves that the width of each paving stone is 2 meters given the courtyard dimensions,
    number of paving stones, and length of each paving stone. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (num_stones : ℕ)
  (stone_length : ℝ)
  (h1 : courtyard_length = 40)
  (h2 : courtyard_width = 33/2)
  (h3 : num_stones = 132)
  (h4 : stone_length = 5/2)
  : ∃ (stone_width : ℝ), stone_width = 2 ∧ 
    courtyard_length * courtyard_width = (stone_length * stone_width) * num_stones :=
by
  sorry


end paving_stone_width_l33_3399


namespace additional_seashells_is_8_l33_3394

/-- The number of additional seashells Carina puts in each week -/
def additional_seashells : ℕ := sorry

/-- The number of seashells in the jar this week -/
def initial_seashells : ℕ := 50

/-- The number of seashells in the jar after 4 weeks -/
def final_seashells : ℕ := 130

/-- The number of weeks -/
def weeks : ℕ := 4

/-- Formula for the total number of seashells after n weeks -/
def total_seashells (n : ℕ) : ℕ :=
  initial_seashells + n * additional_seashells + (n * (n - 1) / 2) * additional_seashells

/-- Theorem stating that the number of additional seashells per week is 8 -/
theorem additional_seashells_is_8 :
  additional_seashells = 8 ∧
  (∀ n : ℕ, n ≤ weeks → total_seashells n ≤ total_seashells (n + 1)) ∧
  total_seashells weeks = final_seashells :=
sorry

end additional_seashells_is_8_l33_3394


namespace gabes_original_seat_l33_3304

/-- Represents the seats in the movie theater --/
inductive Seat
| one
| two
| three
| four
| five
| six
| seven

/-- Represents the friends --/
inductive Friend
| gabe
| flo
| dan
| cal
| bea
| eva
| hal

/-- Represents the seating arrangement --/
def Arrangement := Friend → Seat

/-- Returns the seat to the right of the given seat --/
def seatToRight (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.two
  | Seat.two => Seat.three
  | Seat.three => Seat.four
  | Seat.four => Seat.five
  | Seat.five => Seat.six
  | Seat.six => Seat.seven
  | Seat.seven => Seat.seven

/-- Returns the seat to the left of the given seat --/
def seatToLeft (s : Seat) : Seat :=
  match s with
  | Seat.one => Seat.one
  | Seat.two => Seat.one
  | Seat.three => Seat.two
  | Seat.four => Seat.three
  | Seat.five => Seat.four
  | Seat.six => Seat.five
  | Seat.seven => Seat.six

/-- Theorem stating Gabe's original seat --/
theorem gabes_original_seat (initial : Arrangement) (final : Arrangement) :
  (∀ (f : Friend), initial f ≠ initial Friend.gabe) →
  (final Friend.flo = seatToRight (seatToRight (seatToRight (initial Friend.flo)))) →
  (final Friend.dan = seatToLeft (initial Friend.dan)) →
  (final Friend.cal = initial Friend.cal) →
  (final Friend.bea = initial Friend.eva ∧ final Friend.eva = initial Friend.bea) →
  (final Friend.hal = seatToRight (initial Friend.gabe)) →
  (final Friend.gabe = Seat.one ∨ final Friend.gabe = Seat.seven) →
  initial Friend.gabe = Seat.three :=
by sorry


end gabes_original_seat_l33_3304


namespace tom_tim_typing_ratio_l33_3376

/-- 
Given that Tim and Tom can type 12 pages in one hour together,
and 14 pages when Tom increases his speed by 25%,
prove that the ratio of Tom's normal typing speed to Tim's is 2:1
-/
theorem tom_tim_typing_ratio :
  ∀ (tim_speed tom_speed : ℝ),
    tim_speed + tom_speed = 12 →
    tim_speed + (1.25 * tom_speed) = 14 →
    tom_speed / tim_speed = 2 := by
  sorry

end tom_tim_typing_ratio_l33_3376


namespace order_of_abc_l33_3365

theorem order_of_abc (a b c : ℝ) (ha : a = 2^(1/10)) (hb : b = (1/2)^(4/5)) (hc : c = (1/2)^(1/2)) :
  a > c ∧ c > b :=
sorry

end order_of_abc_l33_3365


namespace dog_age_difference_l33_3351

/-- Proves that the 5th fastest dog is 20 years older than the 4th fastest dog --/
theorem dog_age_difference :
  let dog1_age : ℕ := 10
  let dog2_age : ℕ := dog1_age - 2
  let dog3_age : ℕ := dog2_age + 4
  let dog4_age : ℕ := dog3_age / 2
  let dog5_age : ℕ := dog4_age + 20
  (dog1_age + dog5_age) / 2 = 18 →
  dog5_age - dog4_age = 20 := by
sorry

end dog_age_difference_l33_3351


namespace function_properties_l33_3392

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

def f_derivative_symmetric (a b : ℝ) : Prop :=
  ∀ x : ℝ, (6 * x^2 + 2 * a * x + b) = (6 * (-x - 1)^2 + 2 * a * (-x - 1) + b)

theorem function_properties (a b : ℝ) 
  (h1 : f_derivative_symmetric a b)
  (h2 : 6 + 2 * a + b = 0) :
  (a = 3 ∧ b = -12) ∧
  (∀ x : ℝ, f a b x ≤ f a b (-2)) ∧
  (∀ x : ℝ, f a b x ≥ f a b 1) ∧
  (f a b (-2) = 21) ∧
  (f a b 1 = -6) := by sorry

end function_properties_l33_3392


namespace same_color_probability_l33_3370

/-- The probability of drawing two balls of the same color from a bag with replacement -/
theorem same_color_probability (total : ℕ) (blue : ℕ) (yellow : ℕ) 
  (h_total : total = blue + yellow)
  (h_blue : blue = 5)
  (h_yellow : yellow = 5) :
  (blue / total) * (blue / total) + (yellow / total) * (yellow / total) = 1 / 2 :=
sorry

end same_color_probability_l33_3370


namespace product_and_reciprocal_sum_l33_3353

theorem product_and_reciprocal_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y = 12 → (1 / x) = 5 * (1 / y) → x + y = (6 * Real.sqrt 60) / 5 := by
  sorry

end product_and_reciprocal_sum_l33_3353


namespace ln_third_derivative_value_l33_3308

open Real

theorem ln_third_derivative_value (f : ℝ → ℝ) (x₀ : ℝ) 
  (h1 : ∀ x, f x = log x)
  (h2 : deriv (deriv (deriv f)) x₀ = 1 / x₀^2) :
  x₀ = 1/2 := by
sorry

end ln_third_derivative_value_l33_3308


namespace minimize_sqrt_difference_l33_3338

theorem minimize_sqrt_difference (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (x y : ℕ), 
    (x > 0 ∧ y > 0) ∧
    (x ≤ y) ∧
    (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0) ∧
    (∀ (a b : ℕ), (a > 0 ∧ b > 0) → (a ≤ b) → 
      (Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0) →
      (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b)) ∧
    (x = (p - 1) / 2) ∧
    (y = (p + 1) / 2) := by
  sorry

end minimize_sqrt_difference_l33_3338


namespace perfect_square_triples_l33_3311

theorem perfect_square_triples :
  ∀ (a b c : ℕ),
    (∃ (x : ℕ), a^2 + 2*b + c = x^2) ∧
    (∃ (y : ℕ), b^2 + 2*c + a = y^2) ∧
    (∃ (z : ℕ), c^2 + 2*a + b = z^2) →
    ((a, b, c) = (0, 0, 0) ∨
     (a, b, c) = (1, 1, 1) ∨
     (a, b, c) = (127, 106, 43) ∨
     (a, b, c) = (106, 43, 127) ∨
     (a, b, c) = (43, 127, 106)) :=
by sorry

end perfect_square_triples_l33_3311


namespace abc_inequality_and_fraction_sum_l33_3360

theorem abc_inequality_and_fraction_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^2 + b^2 + c^2 = 9) : 
  a * b * c ≤ 3 * Real.sqrt 3 ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) > (a + b + c) / 3 :=
by sorry

end abc_inequality_and_fraction_sum_l33_3360


namespace nuts_left_l33_3397

theorem nuts_left (total : ℕ) (eaten_fraction : ℚ) (left : ℕ) : 
  total = 30 → eaten_fraction = 5/6 → left = total - (eaten_fraction * total) → left = 5 := by
  sorry

end nuts_left_l33_3397


namespace circle_radius_condition_l33_3371

theorem circle_radius_condition (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 4*x + y^2 + 8*y + c = 0 ↔ (x + 2)^2 + (y + 4)^2 = 25) → 
  c = -5 :=
by sorry

end circle_radius_condition_l33_3371


namespace parallelogram_area_l33_3307

/-- The area of a parallelogram with base 12 cm and height 48 cm is 576 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
    base = 12 → 
    height = 48 → 
    area = base * height → 
    area = 576 := by
  sorry

end parallelogram_area_l33_3307


namespace ap_terms_count_l33_3306

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  n % 2 = 0 ∧ 
  (n / 2 : ℚ) * (2 * a + (n - 2) * d) = 32 ∧ 
  (n / 2 : ℚ) * (2 * a + 2 * d + (n - 2) * d) = 40 ∧ 
  a + (n - 1) * d - a = 8 → 
  n = 16 := by sorry

end ap_terms_count_l33_3306


namespace negation_equivalence_l33_3378

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) :=
by sorry

end negation_equivalence_l33_3378


namespace pencils_per_box_correct_l33_3361

/-- Represents the number of pencils in each box -/
def pencils_per_box : ℕ := 80

/-- Represents the number of boxes of pencils ordered -/
def boxes : ℕ := 15

/-- Represents the cost of a single pencil in dollars -/
def pencil_cost : ℕ := 4

/-- Represents the cost of a single pen in dollars -/
def pen_cost : ℕ := 5

/-- Represents the total cost of all stationery in dollars -/
def total_cost : ℕ := 18300

/-- Theorem stating that the number of pencils per box satisfies the given conditions -/
theorem pencils_per_box_correct : 
  let total_pencils := pencils_per_box * boxes
  let total_pens := 2 * total_pencils + 300
  total_pencils * pencil_cost + total_pens * pen_cost = total_cost := by
  sorry


end pencils_per_box_correct_l33_3361


namespace min_value_equals_gcd_l33_3349

theorem min_value_equals_gcd (a b c : ℕ+) :
  (∃ (x y z : ℤ), ∀ (x' y' z' : ℤ), a * x + b * y + c * z ≤ a * x' + b * y' + c * z' ∧ 0 < a * x + b * y + c * z) →
  (∃ (x y z : ℤ), a * x + b * y + c * z = Nat.gcd a.val (Nat.gcd b.val c.val)) := by
  sorry

end min_value_equals_gcd_l33_3349


namespace prob_independent_of_trials_l33_3324

/-- A random event. -/
structure RandomEvent where
  /-- The probability of the event occurring in a single trial. -/
  probability : ℝ
  /-- Assumption that the probability is between 0 and 1. -/
  prob_nonneg : 0 ≤ probability
  prob_le_one : probability ≤ 1

/-- The probability of the event not occurring in n trials. -/
def prob_not_occur (E : RandomEvent) (n : ℕ) : ℝ :=
  (1 - E.probability) ^ n

/-- The probability of the event occurring at least once in n trials. -/
def prob_occur_at_least_once (E : RandomEvent) (n : ℕ) : ℝ :=
  1 - prob_not_occur E n

/-- Theorem stating that the probability of a random event occurring
    is independent of the number of trials. -/
theorem prob_independent_of_trials (E : RandomEvent) :
  ∀ n : ℕ, prob_occur_at_least_once E (n + 1) - prob_occur_at_least_once E n = E.probability * (prob_not_occur E n) :=
sorry


end prob_independent_of_trials_l33_3324


namespace voter_distribution_l33_3396

theorem voter_distribution (total_voters : ℝ) (dem_percent : ℝ) (rep_percent : ℝ) 
  (rep_vote_a : ℝ) (total_vote_a : ℝ) (dem_vote_a : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 1 - dem_percent →
  rep_vote_a = 0.2 →
  total_vote_a = 0.5 →
  dem_vote_a * dem_percent + rep_vote_a * rep_percent = total_vote_a →
  dem_vote_a = 0.7 := by
sorry

end voter_distribution_l33_3396


namespace no_solution_iff_m_special_l33_3301

/-- The equation has no solution if and only if m is -4, 6, or 1 -/
theorem no_solution_iff_m_special (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → 2 / (x - 2) + m * x / (x^2 - 4) ≠ 3 / (x + 2)) ↔ 
  (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end no_solution_iff_m_special_l33_3301


namespace rational_power_floor_theorem_l33_3318

theorem rational_power_floor_theorem (x : ℚ) : 
  (∃ (a : ℤ), a ≥ 1 ∧ x^(⌊x⌋) = a / 2) ↔ (∃ (n : ℤ), x = n) ∨ x = 3/2 := by
  sorry

end rational_power_floor_theorem_l33_3318


namespace simplify_G_l33_3388

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((2 * x + x^2) / (1 + 2 * x))

theorem simplify_G (x : ℝ) (h : x ≠ -1/2 ∧ x ≠ 1) : 
  G x = 2 * Real.log (1 + 2 * x) - F x :=
by sorry

end simplify_G_l33_3388


namespace arithmetic_square_root_sum_l33_3305

theorem arithmetic_square_root_sum (a b c : ℝ) : 
  a^(1/3) = 2 → 
  b = ⌊Real.sqrt 5⌋ → 
  c^2 = 16 → 
  (Real.sqrt (a + b + c) = Real.sqrt 14) ∨ (Real.sqrt (a + b + c) = Real.sqrt 6) := by
  sorry

end arithmetic_square_root_sum_l33_3305


namespace impossibleColoring_l33_3309

/-- Represents a color in the grid -/
inductive Color
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10

/-- Represents the grid of colors -/
def Grid := Fin 99 → Fin 99 → Color

/-- Checks if a 3x3 subgrid centered at (i, j) has exactly one match with the center -/
def validSubgrid (g : Grid) (i j : Fin 99) : Prop :=
  let center := g i j
  (∃! x y, x ∈ [i-1, i, i+1] ∧ y ∈ [j-1, j, j+1] ∧ (x, y) ≠ (i, j) ∧ g x y = center)

/-- The main theorem stating the impossibility of the described coloring -/
theorem impossibleColoring : ¬∃ g : Grid, ∀ i j : Fin 99, validSubgrid g i j := by
  sorry


end impossibleColoring_l33_3309


namespace count_negative_numbers_l33_3323

def number_list : List ℝ := [0, -2, 3, -0.1, -(-5)]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 2 := by sorry

end count_negative_numbers_l33_3323


namespace unique_prime_cube_l33_3346

theorem unique_prime_cube (p : ℕ) : 
  Prime p ∧ ∃ (a : ℕ), a > 0 ∧ 16 * p + 1 = a^3 ↔ p = 307 := by
  sorry

end unique_prime_cube_l33_3346


namespace height_is_four_l33_3314

/-- The configuration of squares with a small square of area 1 -/
structure SquareConfiguration where
  /-- The side length of the second square -/
  a : ℝ
  /-- The height to be determined -/
  h : ℝ
  /-- The small square has area 1 -/
  small_square_area : 1 = 1
  /-- The equation relating the squares and height -/
  square_relation : 1 + a + 3 = a + h

/-- The theorem stating that h = 4 in the given square configuration -/
theorem height_is_four (config : SquareConfiguration) : config.h = 4 := by
  sorry

end height_is_four_l33_3314


namespace spade_nested_calc_l33_3302

-- Define the spade operation
def spade (x y : ℚ) : ℚ := x - 1 / y

-- Theorem statement
theorem spade_nested_calc : spade 3 (spade 3 (3/2)) = 18/7 := by sorry

end spade_nested_calc_l33_3302


namespace hundredth_digit_of_seven_twenty_sixths_l33_3380

theorem hundredth_digit_of_seven_twenty_sixths (n : ℕ) : n = 100 → 
  (7 : ℚ) / 26 * 10^n % 10 = 9 := by sorry

end hundredth_digit_of_seven_twenty_sixths_l33_3380


namespace min_attempts_to_guarantee_two_charged_l33_3373

/-- Represents a set of batteries -/
def Battery := Fin 8

/-- Represents a pair of batteries -/
def BatteryPair := (Battery × Battery)

/-- The set of all possible battery pairs -/
def allPairs : Finset BatteryPair := sorry

/-- The set of charged batteries -/
def chargedBatteries : Finset Battery := sorry

/-- A function that determines if a set of battery pairs guarantees finding two charged batteries -/
def guaranteesTwoCharged (pairs : Finset BatteryPair) : Prop := sorry

/-- The minimum number of attempts required -/
def minAttempts : ℕ := sorry

theorem min_attempts_to_guarantee_two_charged :
  (minAttempts = 12) ∧
  (∃ (pairs : Finset BatteryPair), pairs.card = minAttempts ∧ guaranteesTwoCharged pairs) ∧
  (∀ (pairs : Finset BatteryPair), pairs.card < minAttempts → ¬guaranteesTwoCharged pairs) := by
  sorry

end min_attempts_to_guarantee_two_charged_l33_3373


namespace smallest_upper_bound_l33_3395

theorem smallest_upper_bound (a b : ℤ) (h1 : a > 6) (h2 : ∀ (x y : ℤ), x > 6 → x - y ≥ 4) : 
  ∃ N : ℤ, (a + b < N) ∧ (∀ M : ℤ, M < N → ¬(a + b < M)) :=
sorry

end smallest_upper_bound_l33_3395


namespace probability_theorem_l33_3320

def total_marbles : ℕ := 8
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 3
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 4

def probability_one_each_plus_red : ℚ :=
  (red_marbles.choose 2 * blue_marbles.choose 1 * green_marbles.choose 1) /
  total_marbles.choose selected_marbles

theorem probability_theorem :
  probability_one_each_plus_red = 9 / 35 := by
  sorry

end probability_theorem_l33_3320


namespace sum_of_reciprocal_equations_l33_3326

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : 1/x + 1/y = 4)
  (h2 : 1/x - 1/y = -3) :
  x + y = 16/7 := by
  sorry

end sum_of_reciprocal_equations_l33_3326


namespace sufficient_not_necessary_l33_3363

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Statement of the theorem
theorem sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → log_half (x + 2) < 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ log_half (x + 2) < 0) :=
by sorry

end sufficient_not_necessary_l33_3363


namespace garden_length_l33_3327

/-- Given a rectangular garden with perimeter 1200 m and breadth 240 m, prove its length is 360 m -/
theorem garden_length (perimeter : ℝ) (breadth : ℝ) (length : ℝ)
  (h1 : perimeter = 1200)
  (h2 : breadth = 240)
  (h3 : perimeter = 2 * length + 2 * breadth) :
  length = 360 :=
by sorry

end garden_length_l33_3327


namespace books_together_l33_3354

/-- The number of books Sandy, Tim, and Benny have together after Benny lost some books. -/
def remaining_books (sandy_books tim_books lost_books : ℕ) : ℕ :=
  sandy_books + tim_books - lost_books

/-- Theorem stating the number of books Sandy, Tim, and Benny have together. -/
theorem books_together : remaining_books 10 33 24 = 19 := by
  sorry

end books_together_l33_3354


namespace comic_book_frames_l33_3352

/-- The number of frames in Julian's comic book -/
def total_frames : ℕ := 143

/-- The number of frames per page if Julian puts them equally on 13 pages -/
def frames_per_page : ℕ := 11

/-- The number of pages if Julian puts 11 frames on each page -/
def number_of_pages : ℕ := 13

/-- Theorem stating that the total number of frames is correct -/
theorem comic_book_frames : 
  total_frames = frames_per_page * number_of_pages :=
by sorry

end comic_book_frames_l33_3352


namespace hundredth_term_difference_l33_3350

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  terms : ℕ
  min_value : ℝ
  max_value : ℝ
  sum : ℝ

/-- The properties of our specific arithmetic sequence -/
def our_sequence : ArithmeticSequence where
  terms := 350
  min_value := 5
  max_value := 150
  sum := 38500

/-- The 100th term of an arithmetic sequence -/
def hundredth_term (a d : ℝ) : ℝ := a + 99 * d

/-- Theorem stating the difference between max and min possible 100th terms -/
theorem hundredth_term_difference (seq : ArithmeticSequence) 
  (h_seq : seq = our_sequence) : 
  ∃ (L G : ℝ), 
    (∀ (a d : ℝ), 
      (seq.min_value ≤ a) ∧ 
      (a + (seq.terms - 1) * d ≤ seq.max_value) ∧
      (seq.sum = (seq.terms : ℝ) * (2 * a + (seq.terms - 1) * d) / 2) →
      (L ≤ hundredth_term a d ∧ hundredth_term a d ≤ G)) ∧
    (G - L = 60.225) := by
  sorry

end hundredth_term_difference_l33_3350


namespace product_sum_theorem_l33_3362

theorem product_sum_theorem (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c → 
  a * b * c = 5^3 → 
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by
sorry

end product_sum_theorem_l33_3362


namespace rollo_guinea_pigs_l33_3390

/-- The amount of food eaten by the first guinea pig -/
def first_guinea_pig_food : ℕ := 2

/-- The amount of food eaten by the second guinea pig -/
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food

/-- The amount of food eaten by the third guinea pig -/
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

/-- The total amount of food needed to feed all guinea pigs -/
def total_food_needed : ℕ := 13

/-- The number of guinea pigs Rollo has -/
def number_of_guinea_pigs : ℕ := 3

theorem rollo_guinea_pigs :
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = total_food_needed ∧
  number_of_guinea_pigs = 3 := by
  sorry

end rollo_guinea_pigs_l33_3390


namespace fruit_drink_volume_l33_3384

/-- Represents the composition of a fruit drink -/
structure FruitDrink where
  orange : ℝ
  watermelon : ℝ
  grape : ℝ
  apple : ℝ
  pineapple : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.orange = 0.1)
  (h2 : drink.watermelon = 0.4)
  (h3 : drink.grape = 0.2)
  (h4 : drink.apple = 0.15)
  (h5 : drink.pineapple = 0.15)
  (h6 : drink.orange + drink.watermelon + drink.grape + drink.apple + drink.pineapple = 1)
  (h7 : 24 / drink.grape = 36 / drink.apple) :
  24 / drink.grape = 240 := by
  sorry

end fruit_drink_volume_l33_3384


namespace prob_sum_7_or_11_l33_3343

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The set of possible sums we're interested in -/
def target_sums : Set ℕ := {7, 11}

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of ways to get a sum of 7 or 11 -/
def favorable_outcomes : ℕ := 8

/-- The probability of rolling a sum of 7 or 11 with two dice -/
def probability_sum_7_or_11 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_7_or_11 : probability_sum_7_or_11 = 2 / 9 := by
  sorry

end prob_sum_7_or_11_l33_3343


namespace product_greater_than_sum_plus_two_l33_3355

theorem product_greater_than_sum_plus_two 
  (a b c : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hab : a * b > a + b) 
  (hbc : b * c > b + c) 
  (hac : a * c > a + c) : 
  a * b * c > a + b + c + 2 := by
sorry

end product_greater_than_sum_plus_two_l33_3355


namespace difference_of_squares_form_l33_3379

theorem difference_of_squares_form (x y : ℝ) :
  ∃ (a b : ℝ), (2*x + y) * (y - 2*x) = -(a^2 - b^2) :=
sorry

end difference_of_squares_form_l33_3379


namespace student_assignment_l33_3335

theorem student_assignment (n : ℕ) (m : ℕ) (h1 : n = 4) (h2 : m = 3) :
  (Nat.choose n 2) * (Nat.factorial m) = 36 := by
  sorry

end student_assignment_l33_3335
