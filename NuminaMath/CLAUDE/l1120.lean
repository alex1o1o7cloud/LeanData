import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1120_112070

def f (x : ℝ) := |3*x - 2| + |x - 2|

theorem problem_solution :
  (∀ x : ℝ, f x ≤ 8 ↔ x ∈ Set.Icc (-1) 3) ∧
  (∀ m : ℝ, (∀ x : ℝ, x ≠ 0 → f x ≥ (m^2 - m + 2) * |x|) → m ∈ Set.Icc 0 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1120_112070


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1120_112024

-- Problem 1
theorem problem_1 : (1) - 2 + 3 - 4 + 5 = 3 := by sorry

-- Problem 2
theorem problem_2 : (-4/7) / (8/49) = -7/2 := by sorry

-- Problem 3
theorem problem_3 : (1/2 - 3/5 + 2/3) * (-15) = -17/2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1120_112024


namespace NUMINAMATH_CALUDE_total_toy_count_l1120_112001

def toy_count (jerry gabriel jaxon sarah emily : ℕ) : Prop :=
  jerry = gabriel + 8 ∧
  gabriel = 2 * jaxon ∧
  jaxon = 15 ∧
  sarah = jerry - 5 ∧
  sarah = emily + 3 ∧
  emily = 2 * gabriel

theorem total_toy_count :
  ∀ jerry gabriel jaxon sarah emily : ℕ,
  toy_count jerry gabriel jaxon sarah emily →
  jerry + gabriel + jaxon + sarah + emily = 176 :=
by
  sorry

end NUMINAMATH_CALUDE_total_toy_count_l1120_112001


namespace NUMINAMATH_CALUDE_workshop_workers_l1120_112093

theorem workshop_workers (total_average : ℝ) (tech_count : ℕ) (tech_average : ℝ) (nontech_average : ℝ) : 
  total_average = 9500 → 
  tech_count = 7 → 
  tech_average = 12000 → 
  nontech_average = 6000 → 
  ∃ (total_workers : ℕ), total_workers = 12 ∧ 
    (total_workers : ℝ) * total_average = 
      (tech_count : ℝ) * tech_average + 
      ((total_workers - tech_count) : ℝ) * nontech_average :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1120_112093


namespace NUMINAMATH_CALUDE_car_not_sold_probability_l1120_112036

/-- Given the odds of selling a car on a given day are 5:6, 
    the probability that the car is not sold on that day is 6/11 -/
theorem car_not_sold_probability (odds_success : ℚ) (odds_failure : ℚ) :
  odds_success = 5/6 → odds_failure = 6/5 →
  (odds_failure / (odds_success + 1)) = 6/11 := by
  sorry

end NUMINAMATH_CALUDE_car_not_sold_probability_l1120_112036


namespace NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l1120_112002

theorem restaurant_gratuity_calculation (dish_prices : List ℝ) (tip_percentage : ℝ) : 
  dish_prices = [10, 13, 17, 15, 20] → 
  tip_percentage = 0.18 → 
  (dish_prices.sum * tip_percentage) = 13.50 := by
sorry

end NUMINAMATH_CALUDE_restaurant_gratuity_calculation_l1120_112002


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1120_112014

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  isGeometricSequence a →
  (a 4 + a 8 = -11) →
  (a 4 * a 8 = 9) →
  a 6 = -3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1120_112014


namespace NUMINAMATH_CALUDE_intersection_M_N_l1120_112053

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1120_112053


namespace NUMINAMATH_CALUDE_ratio_problem_l1120_112003

theorem ratio_problem (a b c : ℕ+) (x m : ℚ) :
  (∃ (k : ℕ+), a = 4 * k ∧ b = 5 * k ∧ c = 6 * k) →
  x = a + (25 / 100) * a →
  m = b - (40 / 100) * b →
  Even c →
  (∀ (a' b' c' : ℕ+), (∃ (k' : ℕ+), a' = 4 * k' ∧ b' = 5 * k' ∧ c' = 6 * k') → 
    a + b + c ≤ a' + b' + c') →
  m / x = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1120_112003


namespace NUMINAMATH_CALUDE_ceiling_sqrt_225_l1120_112022

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_225_l1120_112022


namespace NUMINAMATH_CALUDE_type_a_cubes_count_l1120_112073

/-- Represents the dimensions of the rectangular solid -/
def solid_dimensions : Fin 3 → ℕ
  | 0 => 120
  | 1 => 350
  | 2 => 400
  | _ => 0

/-- Calculates the number of cubes traversed by the diagonal -/
def total_cubes_traversed : ℕ := sorry

/-- The number of type A cubes traversed by the diagonal -/
def type_a_cubes : ℕ := total_cubes_traversed / 2

theorem type_a_cubes_count : type_a_cubes = 390 := by sorry

end NUMINAMATH_CALUDE_type_a_cubes_count_l1120_112073


namespace NUMINAMATH_CALUDE_lifeguard_swim_time_l1120_112054

/-- Proves the time spent swimming front crawl given total distance, speeds, and total time -/
theorem lifeguard_swim_time 
  (total_distance : ℝ) 
  (front_crawl_speed : ℝ) 
  (breaststroke_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : total_distance = 500)
  (h2 : front_crawl_speed = 45)
  (h3 : breaststroke_speed = 35)
  (h4 : total_time = 12) :
  ∃ (front_crawl_time : ℝ), 
    front_crawl_time * front_crawl_speed + 
    (total_time - front_crawl_time) * breaststroke_speed = total_distance ∧ 
    front_crawl_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_lifeguard_swim_time_l1120_112054


namespace NUMINAMATH_CALUDE_sqrt_neg_three_squared_l1120_112061

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_three_squared_l1120_112061


namespace NUMINAMATH_CALUDE_books_read_l1120_112017

theorem books_read (total : ℕ) (unread : ℕ) (h1 : total = 13) (h2 : unread = 4) :
  total - unread = 9 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l1120_112017


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1120_112083

/-- Given a quadratic function f(x) = x^2 - 2ax + b where a and b are real numbers,
    and the solution set of f(x) ≤ 0 is [-1, 2], this theorem proves two statements about f. -/
theorem quadratic_function_properties (a b : ℝ) 
    (f : ℝ → ℝ) 
    (h_f : ∀ x, f x = x^2 - 2*a*x + b) 
    (h_solution_set : Set.Icc (-1 : ℝ) 2 = {x | f x ≤ 0}) : 
  (∀ x, b*x^2 - 2*a*x + 1 ≤ 0 ↔ x ≤ -1 ∨ x ≥ 1/2) ∧ 
  (b = a^2 → 
   (∀ x₁ ∈ Set.Icc 2 4, ∃ x₂ ∈ Set.Icc 2 4, f x₁ * f x₂ = 1) → 
   a = 3 + Real.sqrt 2 ∨ a = 3 - Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1120_112083


namespace NUMINAMATH_CALUDE_picture_frame_width_l1120_112059

theorem picture_frame_width 
  (height : ℝ) 
  (circumference : ℝ) 
  (h_height : height = 12) 
  (h_circumference : circumference = 38) : 
  let width := (circumference - 2 * height) / 2
  width = 7 := by
sorry

end NUMINAMATH_CALUDE_picture_frame_width_l1120_112059


namespace NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_solution_existence_l1120_112056

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| * |x - 3|

-- Part 1: Solution set of f(x) > 7-x
theorem solution_set_f_gt_7_minus_x :
  {x : ℝ | f x > 7 - x} = {x : ℝ | x < -6 ∨ x > 2} := by sorry

-- Part 2: Range of m for which f(x) ≤ |3m-2| has a solution
theorem range_of_m_for_solution_existence :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_7_minus_x_range_of_m_for_solution_existence_l1120_112056


namespace NUMINAMATH_CALUDE_maggie_fish_books_l1120_112094

/-- The number of fish books Maggie bought -/
def fish_books : ℕ := sorry

/-- The total amount Maggie spent -/
def total_spent : ℕ := 170

/-- The number of plant books Maggie bought -/
def plant_books : ℕ := 9

/-- The number of science magazines Maggie bought -/
def science_magazines : ℕ := 10

/-- The cost of each book -/
def book_cost : ℕ := 15

/-- The cost of each magazine -/
def magazine_cost : ℕ := 2

theorem maggie_fish_books : 
  fish_books = 1 := by sorry

end NUMINAMATH_CALUDE_maggie_fish_books_l1120_112094


namespace NUMINAMATH_CALUDE_karen_cake_days_l1120_112020

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of days Karen packs ham sandwiches -/
def ham_days : ℕ := 3

/-- The probability of packing a ham sandwich and cake on the same day -/
def ham_cake_prob : ℚ := 12 / 100

/-- The number of days Karen packs a piece of cake -/
def cake_days : ℕ := sorry

theorem karen_cake_days :
  (ham_days : ℚ) / school_days * cake_days / school_days = ham_cake_prob →
  cake_days = 1 := by sorry

end NUMINAMATH_CALUDE_karen_cake_days_l1120_112020


namespace NUMINAMATH_CALUDE_min_diff_composite_sum_96_l1120_112051

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem min_diff_composite_sum_96 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → c + d = 96 → c ≠ d →
  (max c d - min c d) ≥ (max a b - min a b) ∧ (max a b - min a b) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_diff_composite_sum_96_l1120_112051


namespace NUMINAMATH_CALUDE_logarithm_equation_solution_l1120_112008

theorem logarithm_equation_solution :
  ∃ (A B C : ℕ+), 
    (Nat.gcd A.val (Nat.gcd B.val C.val) = 1) ∧
    (A.val : ℝ) * (Real.log 5 / Real.log 300) + (B.val : ℝ) * (Real.log (2 * A.val) / Real.log 300) = C.val ∧
    A.val + B.val + C.val = 4 := by
  sorry

#check logarithm_equation_solution

end NUMINAMATH_CALUDE_logarithm_equation_solution_l1120_112008


namespace NUMINAMATH_CALUDE_min_distance_circle_line_l1120_112089

theorem min_distance_circle_line :
  let C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 5}
  let L : Set (ℝ × ℝ) := {q | q.1 - 2*q.2 + 4 = 0}
  (∀ p ∈ C, ∃ q ∈ L, ∀ r ∈ L, dist p q ≤ dist p r) →
  (∃ p ∈ C, ∃ q ∈ L, dist p q = 3 * Real.sqrt 5 / 5) ∧
  (∀ p ∈ C, ∀ q ∈ L, dist p q ≥ 3 * Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_circle_line_l1120_112089


namespace NUMINAMATH_CALUDE_sequence_ratio_proof_l1120_112077

/-- Given an arithmetic sequence and a geometric sequence with specific properties,
    prove that the common ratio of the geometric sequence is (√5 - 1) / 2. -/
theorem sequence_ratio_proof (d : ℚ) (q : ℚ) (h_d : d ≠ 0) (h_q : 0 < q ∧ q < 1) :
  let a : ℕ → ℚ := λ n => d * n
  let b : ℕ → ℚ := λ n => d^2 * q^(n-1)
  (∃ k : ℕ+, (a 1)^2 + (a 2)^2 + (a 3)^2 = k * (b 1 + b 2 + b 3)) →
  q = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_proof_l1120_112077


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1120_112058

theorem sum_of_solutions (x y : ℝ) 
  (hx : x^3 - 6*x^2 + 12*x = 13) 
  (hy : y^3 + 3*y - 3*y^2 = -4) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1120_112058


namespace NUMINAMATH_CALUDE_hyperbola_and_line_equations_l1120_112035

/-- Given a hyperbola with specified properties, prove its equation and the equation of a line intersecting it. -/
theorem hyperbola_and_line_equations
  (a b : ℝ)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_asymptote : ∀ x y : ℝ, y = 2 * x → (∃ t : ℝ, y = t * x ∧ y^2 / a^2 - x^2 / b^2 = 1))
  (h_focus_distance : ∃ F : ℝ × ℝ, ∀ x y : ℝ, y = 2 * x → Real.sqrt ((F.1 - x)^2 + (F.2 - y)^2) = 1)
  (h_midpoint : ∃ A B : ℝ × ℝ, A.1 ≠ B.1 ∧ A.2 ≠ B.2 ∧
    (A.2^2 / a^2 - A.1^2 / b^2 = 1) ∧
    (B.2^2 / a^2 - B.1^2 / b^2 = 1) ∧
    ((A.1 + B.1) / 2 = 1) ∧
    ((A.2 + B.2) / 2 = 4)) :
  (a = 2 ∧ b = 1) ∧
  (∀ x y : ℝ, y^2 / 4 - x^2 = 1 ↔ y^2 / a^2 - x^2 / b^2 = 1) ∧
  (∃ k m : ℝ, k = 1 ∧ m = 3 ∧ ∀ x y : ℝ, y^2 / 4 - x^2 = 1 → (x - y + m = 0 ↔ ∃ t : ℝ, x = 1 + t ∧ y = 4 + t)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_equations_l1120_112035


namespace NUMINAMATH_CALUDE_one_true_proposition_l1120_112068

theorem one_true_proposition :
  let P := (fun (a b : ℝ) => a + b = 1 → a * b ≤ 1/4)
  let converse := (fun (a b : ℝ) => a * b > 1/4 → a + b ≠ 1)
  let inverse := (fun (a b : ℝ) => a * b ≤ 1/4 → a + b = 1)
  let contrapositive := (fun (a b : ℝ) => a * b > 1/4 → a + b ≠ 1)
  (∀ a b, P a b) ∧ (∀ a b, contrapositive a b) ∧ (∃ a b, ¬(inverse a b)) ∧ (∃ a b, ¬(converse a b)) :=
by sorry

end NUMINAMATH_CALUDE_one_true_proposition_l1120_112068


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1120_112006

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_eight_ten : Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l1120_112006


namespace NUMINAMATH_CALUDE_inequality_solution_l1120_112030

theorem inequality_solution (x : ℝ) : 
  (1 / 6 : ℝ) + |x - 1 / 3| < 1 / 2 ↔ 0 < x ∧ x < 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1120_112030


namespace NUMINAMATH_CALUDE_different_counting_units_for_equal_decimals_l1120_112026

-- Define the concept of a decimal number
structure Decimal where
  value : ℚ
  decimalPlaces : ℕ

-- Define the concept of a counting unit
def countingUnit (d : Decimal) : ℚ := 1 / (10 ^ d.decimalPlaces)

-- Define equality for decimals based on their value
def decimalEqual (d1 d2 : Decimal) : Prop := d1.value = d2.value

-- Theorem statement
theorem different_counting_units_for_equal_decimals :
  ∃ (d1 d2 : Decimal), decimalEqual d1 d2 ∧ countingUnit d1 ≠ countingUnit d2 := by
  sorry

end NUMINAMATH_CALUDE_different_counting_units_for_equal_decimals_l1120_112026


namespace NUMINAMATH_CALUDE_stamp_problem_l1120_112012

/-- Returns true if postage can be formed with given denominations -/
def can_form_postage (d1 d2 d3 amount : ℕ) : Prop :=
  ∃ (x y z : ℕ), d1 * x + d2 * y + d3 * z = amount

/-- Returns true if n satisfies the stamp problem conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n > 0 ∧
  (∀ m : ℕ, m > 70 → can_form_postage 3 n (n+1) m) ∧
  ¬(can_form_postage 3 n (n+1) 70)

theorem stamp_problem :
  ∃! (n : ℕ), satisfies_conditions n ∧ n = 37 :=
sorry

end NUMINAMATH_CALUDE_stamp_problem_l1120_112012


namespace NUMINAMATH_CALUDE_machine_total_time_l1120_112092

/-- The total time a machine worked, including downtime, given its production rates and downtime -/
theorem machine_total_time
  (time_A : ℕ) (shirts_A : ℕ) (time_B : ℕ) (shirts_B : ℕ) (downtime : ℕ)
  (h_A : time_A = 75 ∧ shirts_A = 13)
  (h_B : time_B = 5 ∧ shirts_B = 3)
  (h_downtime : downtime = 120) :
  time_A + time_B + downtime = 200 := by
  sorry


end NUMINAMATH_CALUDE_machine_total_time_l1120_112092


namespace NUMINAMATH_CALUDE_shortest_path_length_is_28b_l1120_112049

/-- Represents a 3x3 grid of blocks with side length b -/
structure Grid :=
  (b : ℝ)
  (size : ℕ := 3)

/-- The number of street segments in the grid -/
def Grid.streetSegments (g : Grid) : ℕ := 24

/-- The number of intersections with odd degree -/
def Grid.oddDegreeIntersections (g : Grid) : ℕ := 8

/-- The extra segments that need to be traversed twice -/
def Grid.extraSegments (g : Grid) : ℕ := g.oddDegreeIntersections / 2

/-- The shortest path length to pave all streets in the grid -/
def Grid.shortestPathLength (g : Grid) : ℝ :=
  (g.streetSegments + g.extraSegments) * g.b

/-- Theorem stating that the shortest path length is 28b -/
theorem shortest_path_length_is_28b (g : Grid) :
  g.shortestPathLength = 28 * g.b := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_length_is_28b_l1120_112049


namespace NUMINAMATH_CALUDE_first_option_cost_is_68_l1120_112040

/-- Represents the car rental problem with given conditions -/
def CarRentalProblem (trip_distance : ℝ) (second_option_cost : ℝ) 
  (gas_efficiency : ℝ) (gas_cost_per_liter : ℝ) (savings : ℝ) : Prop :=
  let total_distance := 2 * trip_distance
  let gas_needed := total_distance / gas_efficiency
  let gas_cost := gas_needed * gas_cost_per_liter
  let first_option_cost := second_option_cost - savings
  first_option_cost = 68

/-- Theorem stating that the first option costs $68 per day -/
theorem first_option_cost_is_68 :
  CarRentalProblem 150 90 15 0.9 22 := by
  sorry

#check first_option_cost_is_68

end NUMINAMATH_CALUDE_first_option_cost_is_68_l1120_112040


namespace NUMINAMATH_CALUDE_frog_jumped_farther_l1120_112010

/-- The distance the grasshopper jumped in inches -/
def grasshopper_jump : ℕ := 36

/-- The distance the frog jumped in inches -/
def frog_jump : ℕ := 53

/-- The difference between the frog's jump and the grasshopper's jump -/
def jump_difference : ℕ := frog_jump - grasshopper_jump

theorem frog_jumped_farther : jump_difference = 17 := by
  sorry

end NUMINAMATH_CALUDE_frog_jumped_farther_l1120_112010


namespace NUMINAMATH_CALUDE_certain_number_problem_l1120_112025

theorem certain_number_problem (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 4) 
  (h3 : a * (a - 4) = b * (b - 4)) : a * (a - 4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1120_112025


namespace NUMINAMATH_CALUDE_animal_mortality_probability_l1120_112011

/-- The probability of an animal dying in each of the first 3 months, given survival data -/
theorem animal_mortality_probability (total : ℕ) (survivors : ℝ) (p : ℝ) 
  (h_total : total = 400)
  (h_survivors : survivors = 291.6)
  (h_survival_equation : survivors = total * (1 - p)^3) :
  p = 0.1 := by
sorry

end NUMINAMATH_CALUDE_animal_mortality_probability_l1120_112011


namespace NUMINAMATH_CALUDE_mary_spends_five_l1120_112039

/-- Proves that Mary spends $5 given the initial conditions and final state -/
theorem mary_spends_five (marco_initial : ℕ) (mary_initial : ℕ) 
  (h1 : marco_initial = 24)
  (h2 : mary_initial = 15)
  (marco_gives : ℕ := marco_initial / 2)
  (marco_final : ℕ := marco_initial - marco_gives)
  (mary_after_receiving : ℕ := mary_initial + marco_gives)
  (mary_final : ℕ)
  (h3 : mary_final = marco_final + 10) :
  mary_after_receiving - mary_final = 5 := by
sorry

end NUMINAMATH_CALUDE_mary_spends_five_l1120_112039


namespace NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1120_112046

-- Define a structure for a rectangle
structure Rectangle where
  width : ℝ
  length : ℝ

-- Define the area of a rectangle
def area (r : Rectangle) : ℝ := r.width * r.length

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

-- Theorem statement
theorem rectangle_area_equals_perimeter :
  ∀ (r : Rectangle),
    r.length = 3 * r.width →
    area r = perimeter r →
    r.width = 8/3 ∧ r.length = 8 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_equals_perimeter_l1120_112046


namespace NUMINAMATH_CALUDE_inverse_proportion_increasing_l1120_112062

theorem inverse_proportion_increasing (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → (m + 3) / x₁ < (m + 3) / x₂) → 
  m < -3 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_increasing_l1120_112062


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l1120_112079

/-- Hexagon ABCDEF with given side lengths -/
structure Hexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  AF : ℝ

/-- The perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.AF

/-- Theorem: The perimeter of the given hexagon is 7 + √10 -/
theorem hexagon_perimeter : 
  ∀ (h : Hexagon), 
  h.AB = 1 → h.BC = 1 → h.CD = 2 → h.DE = 2 → h.EF = 1 → h.AF = Real.sqrt 10 →
  perimeter h = 7 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l1120_112079


namespace NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l1120_112064

/-- The maximum number of regions created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- The number of additional regions created by m parallel lines intersecting n non-parallel lines -/
def parallel_regions (m n : ℕ) : ℕ :=
  m * (n + 1)

/-- The maximum number of regions created by n lines in a plane, where p of them are parallel -/
def max_regions_with_parallel (n p : ℕ) : ℕ :=
  max_regions (n - p) + parallel_regions p (n - p)

theorem max_regions_50_lines_20_parallel :
  max_regions_with_parallel 50 20 = 1086 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l1120_112064


namespace NUMINAMATH_CALUDE_aron_vacuuming_days_l1120_112097

/-- The number of days Aron spends vacuuming each week -/
def vacuuming_days : ℕ := sorry

/-- The time spent vacuuming per day in minutes -/
def vacuuming_time_per_day : ℕ := 30

/-- The time spent dusting per day in minutes -/
def dusting_time_per_day : ℕ := 20

/-- The number of days Aron spends dusting each week -/
def dusting_days : ℕ := 2

/-- The total cleaning time per week in minutes -/
def total_cleaning_time : ℕ := 130

theorem aron_vacuuming_days :
  vacuuming_days * vacuuming_time_per_day +
  dusting_days * dusting_time_per_day =
  total_cleaning_time ∧
  vacuuming_days = 3 :=
by sorry

end NUMINAMATH_CALUDE_aron_vacuuming_days_l1120_112097


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l1120_112031

theorem max_tickets_purchasable (ticket_price : ℕ) (budget : ℕ) : 
  ticket_price = 15 → budget = 150 → 
  (∀ n : ℕ, n * ticket_price ≤ budget → n ≤ 10) ∧ 
  10 * ticket_price ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l1120_112031


namespace NUMINAMATH_CALUDE_c_used_car_for_13_hours_l1120_112069

/-- Represents the car rental scenario -/
structure CarRental where
  totalCost : ℝ
  aHours : ℝ
  bHours : ℝ
  bPaid : ℝ
  cHours : ℝ

/-- Theorem stating that under the given conditions, c used the car for 13 hours -/
theorem c_used_car_for_13_hours (rental : CarRental) 
  (h1 : rental.totalCost = 720)
  (h2 : rental.aHours = 9)
  (h3 : rental.bHours = 10)
  (h4 : rental.bPaid = 225) :
  rental.cHours = 13 := by
  sorry

#check c_used_car_for_13_hours

end NUMINAMATH_CALUDE_c_used_car_for_13_hours_l1120_112069


namespace NUMINAMATH_CALUDE_acid_dilution_l1120_112075

theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m * m / 100 = (m - 5) / 100 * (m + x)) → x = 5 * m / (m - 5) := by
  sorry

end NUMINAMATH_CALUDE_acid_dilution_l1120_112075


namespace NUMINAMATH_CALUDE_office_canteen_chairs_l1120_112091

/-- The number of round tables in the office canteen -/
def num_round_tables : ℕ := 2

/-- The number of rectangular tables in the office canteen -/
def num_rectangular_tables : ℕ := 2

/-- The number of chairs at each round table -/
def chairs_per_round_table : ℕ := 6

/-- The number of chairs at each rectangular table -/
def chairs_per_rectangular_table : ℕ := 7

/-- The total number of chairs in the office canteen -/
def total_chairs : ℕ := num_round_tables * chairs_per_round_table + num_rectangular_tables * chairs_per_rectangular_table

theorem office_canteen_chairs : total_chairs = 26 := by
  sorry

end NUMINAMATH_CALUDE_office_canteen_chairs_l1120_112091


namespace NUMINAMATH_CALUDE_work_completion_fraction_l1120_112066

theorem work_completion_fraction (x_days y_days z_days total_days : ℕ) 
  (hx : x_days = 14) 
  (hy : y_days = 20) 
  (hz : z_days = 25) 
  (ht : total_days = 5) : 
  (total_days : ℚ) * ((1 : ℚ) / x_days + (1 : ℚ) / y_days + (1 : ℚ) / z_days) = 113 / 140 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_fraction_l1120_112066


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l1120_112081

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a^3 + t.b^3 - t.c^3) / (t.a + t.b - t.c) = t.c^2

def condition2 (t : Triangle) : Prop :=
  Real.sin t.α * Real.sin t.β = 3/4

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) :
  t.a = t.b ∧ t.b = t.c ∧ t.α = π/3 ∧ t.β = π/3 ∧ t.γ = π/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l1120_112081


namespace NUMINAMATH_CALUDE_math_reading_difference_l1120_112086

def reading_homework : ℕ := 4
def math_homework : ℕ := 7

theorem math_reading_difference : math_homework - reading_homework = 3 := by
  sorry

end NUMINAMATH_CALUDE_math_reading_difference_l1120_112086


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1120_112095

-- Define an equilateral triangle
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.side_length

-- Theorem statement
theorem equilateral_triangle_side_length 
  (t : EquilateralTriangle) 
  (h : perimeter t = 18) : 
  t.side_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l1120_112095


namespace NUMINAMATH_CALUDE_opposite_pairs_l1120_112076

theorem opposite_pairs : 
  (3^2 = -(-(3^2))) ∧ 
  (-4 ≠ -(-4)) ∧ 
  (-3 ≠ -(-|-3|)) ∧ 
  (-2^3 ≠ -((-2)^3)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_pairs_l1120_112076


namespace NUMINAMATH_CALUDE_survey_probability_l1120_112067

theorem survey_probability : 
  let n : ℕ := 14  -- Total number of questions
  let k : ℕ := 10  -- Number of correct answers
  let m : ℕ := 4   -- Number of possible answers per question
  (n.choose k * (m - 1)^(n - k)) / m^n = 1001 * 3^4 / 4^14 := by
  sorry

end NUMINAMATH_CALUDE_survey_probability_l1120_112067


namespace NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_two_l1120_112004

/-- Given a complex number z defined as z = 2/(1+i) + (1+i)^2, prove that its modulus |z| is equal to √2 -/
theorem modulus_of_z_is_sqrt_two : 
  let z : ℂ := 2 / (1 + Complex.I) + (1 + Complex.I)^2
  ‖z‖ = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_sqrt_two_l1120_112004


namespace NUMINAMATH_CALUDE_flower_difference_l1120_112074

def white_flowers : ℕ := 555
def red_flowers : ℕ := 347
def blue_flowers : ℕ := 498
def yellow_flowers : ℕ := 425

theorem flower_difference : 
  (red_flowers + blue_flowers + yellow_flowers) - white_flowers = 715 := by
  sorry

end NUMINAMATH_CALUDE_flower_difference_l1120_112074


namespace NUMINAMATH_CALUDE_photo_arrangements_l1120_112055

/-- The number of ways 7 students can stand in a line for a photo, 
    given specific constraints on their positions. -/
theorem photo_arrangements (n : Nat) (h1 : n = 7) : 
  (∃ (arrangement_count : Nat), 
    (∀ (A B C : Nat) (others : Finset Nat), 
      A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
      others.card = n - 3 ∧
      (∀ x ∈ others, x ≠ A ∧ x ≠ B ∧ x ≠ C) ∧
      (∀ perm : List Nat, perm.length = n →
        (perm.indexOf A).succ ≠ perm.indexOf B ∧
        (perm.indexOf A).pred ≠ perm.indexOf B ∧
        ((perm.indexOf B).succ = perm.indexOf C ∨
         (perm.indexOf B).pred = perm.indexOf C)) →
    arrangement_count = 1200)) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1120_112055


namespace NUMINAMATH_CALUDE_correct_observation_value_l1120_112019

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h_n : n = 20)
  (h_initial_mean : initial_mean = 36)
  (h_wrong_value : wrong_value = 40)
  (h_corrected_mean : corrected_mean = 34.9) :
  (n : ℝ) * initial_mean - wrong_value + (n : ℝ) * corrected_mean - ((n : ℝ) * initial_mean - wrong_value) = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l1120_112019


namespace NUMINAMATH_CALUDE_mascot_purchase_equations_l1120_112044

/-- Represents the purchase of mascot dolls and keychains --/
structure MascotPurchase where
  dolls : ℕ
  keychains : ℕ
  total_cost : ℕ
  doll_price : ℕ
  keychain_price : ℕ

/-- The correct system of equations for the mascot purchase --/
def correct_equations (p : MascotPurchase) : Prop :=
  p.keychains = 2 * p.dolls ∧ 
  p.total_cost = p.doll_price * p.dolls + p.keychain_price * p.keychains

/-- Theorem stating the correct system of equations for the given conditions --/
theorem mascot_purchase_equations :
  ∀ (p : MascotPurchase), 
    p.total_cost = 5000 ∧ 
    p.doll_price = 60 ∧ 
    p.keychain_price = 20 →
    correct_equations p :=
by
  sorry


end NUMINAMATH_CALUDE_mascot_purchase_equations_l1120_112044


namespace NUMINAMATH_CALUDE_friend_candy_purchase_l1120_112038

def feeding_allowance : ℚ := 4
def fraction_given : ℚ := 1/4
def candy_cost : ℚ := 1/5  -- 20 cents = 1/5 dollar

theorem friend_candy_purchase :
  (feeding_allowance * fraction_given) / candy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_candy_purchase_l1120_112038


namespace NUMINAMATH_CALUDE_ten_factorial_divided_by_four_factorial_l1120_112084

theorem ten_factorial_divided_by_four_factorial :
  (10 : ℕ).factorial / (4 : ℕ).factorial = 151200 := by
  have h1 : (10 : ℕ).factorial = 3628800 := by sorry
  sorry

end NUMINAMATH_CALUDE_ten_factorial_divided_by_four_factorial_l1120_112084


namespace NUMINAMATH_CALUDE_basketball_donations_l1120_112047

theorem basketball_donations (total_donations : ℕ) 
  (basketball_hoops : ℕ) (pool_floats : ℕ) (footballs : ℕ) (tennis_balls : ℕ) :
  total_donations = 300 →
  basketball_hoops = 60 →
  pool_floats = 120 →
  footballs = 50 →
  tennis_balls = 40 →
  ∃ (basketballs : ℕ),
    basketballs = total_donations - (basketball_hoops + (pool_floats - pool_floats / 4) + footballs + tennis_balls) + basketball_hoops / 2 ∧
    basketballs = 90 :=
by sorry

end NUMINAMATH_CALUDE_basketball_donations_l1120_112047


namespace NUMINAMATH_CALUDE_number_less_than_abs_is_negative_l1120_112042

theorem number_less_than_abs_is_negative (x : ℝ) : x < |x| → x < 0 := by
  sorry

end NUMINAMATH_CALUDE_number_less_than_abs_is_negative_l1120_112042


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1120_112013

theorem right_triangle_side_length 
  (P Q R : ℝ × ℝ) 
  (is_right_triangle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0) 
  (tan_R : (R.2 - P.2) / (R.1 - P.1) = 4/3) 
  (PQ_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3) : 
  Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1120_112013


namespace NUMINAMATH_CALUDE_world_cup_matches_l1120_112050

/-- The number of matches played in a group of teams where each pair plays twice -/
def number_of_matches (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a group of 6 teams where each pair plays twice, 30 matches are played -/
theorem world_cup_matches : number_of_matches 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_world_cup_matches_l1120_112050


namespace NUMINAMATH_CALUDE_y_divisibility_l1120_112034

def y : ℕ := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem y_divisibility :
  (∃ k : ℕ, y = 5 * k) ∧
  (∃ k : ℕ, y = 10 * k) ∧
  (∃ k : ℕ, y = 20 * k) ∧
  (∃ k : ℕ, y = 40 * k) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l1120_112034


namespace NUMINAMATH_CALUDE_max_books_theorem_l1120_112090

def single_book_cost : ℕ := 3
def four_pack_cost : ℕ := 10
def seven_pack_cost : ℕ := 15
def budget : ℕ := 32

def max_books_bought (budget single four seven : ℕ) : ℕ :=
  sorry

theorem max_books_theorem :
  max_books_bought budget single_book_cost four_pack_cost seven_pack_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_books_theorem_l1120_112090


namespace NUMINAMATH_CALUDE_range_of_a_l1120_112099

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x - a * x ≥ -x + Real.log (a * x)) ↔ (0 < a ∧ a ≤ Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1120_112099


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l1120_112018

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- Represents that no three vertices of the hexadecagon are collinear -/
axiom no_collinear_vertices : True

/-- The number of triangles that can be formed using the vertices of a regular hexadecagon -/
def num_triangles : ℕ := Nat.choose n 3

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_triangles_l1120_112018


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1120_112057

theorem simplify_and_evaluate (a : ℚ) (h : a = -3/2) :
  (a + 2)^2 - (a + 1)*(a - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1120_112057


namespace NUMINAMATH_CALUDE_exponential_inequality_l1120_112085

theorem exponential_inequality (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) :
  a^(Real.sqrt a) > a^(a^a) ∧ a^(a^a) > a := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1120_112085


namespace NUMINAMATH_CALUDE_total_plans_is_180_l1120_112015

def male_teachers : ℕ := 4
def female_teachers : ℕ := 3
def schools : ℕ := 3

-- Function to calculate the number of ways to select and assign teachers
def selection_and_assignment_plans : ℕ :=
  (male_teachers.choose 1 * female_teachers.choose 2 +
   male_teachers.choose 2 * female_teachers.choose 1) *
  schools.factorial

-- Theorem to prove
theorem total_plans_is_180 :
  selection_and_assignment_plans = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_plans_is_180_l1120_112015


namespace NUMINAMATH_CALUDE_impossibility_of_tiling_l1120_112009

/-- Represents a tetromino shape -/
inductive TetrominoShape
  | T
  | L
  | I

/-- Represents a 10x10 chessboard -/
def Chessboard := Fin 10 → Fin 10 → Bool

/-- Checks if a given tetromino shape can tile the chessboard -/
def can_tile (shape : TetrominoShape) (board : Chessboard) : Prop :=
  ∃ (tiling : Nat → Nat → Nat → Nat → Bool),
    ∀ (i j : Fin 10), board i j = true ↔ 
      ∃ (x y : Nat), tiling x y i j = true

theorem impossibility_of_tiling (shape : TetrominoShape) :
  ¬∃ (board : Chessboard), can_tile shape board := by
  sorry

#check impossibility_of_tiling TetrominoShape.T
#check impossibility_of_tiling TetrominoShape.L
#check impossibility_of_tiling TetrominoShape.I

end NUMINAMATH_CALUDE_impossibility_of_tiling_l1120_112009


namespace NUMINAMATH_CALUDE_power_function_decreasing_m_l1120_112078

/-- A power function y = ax^b where a and b are constants and x > 0 -/
def isPowerFunction (y : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, x > 0 → y x = a * x ^ b

/-- A decreasing function on (0, +∞) -/
def isDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₂ < f x₁

theorem power_function_decreasing_m (m : ℝ) :
  isPowerFunction (fun x => (m^2 - 2*m - 2) * x^(-4*m - 2)) →
  isDecreasingOn (fun x => (m^2 - 2*m - 2) * x^(-4*m - 2)) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_m_l1120_112078


namespace NUMINAMATH_CALUDE_division_value_proof_l1120_112028

theorem division_value_proof (x : ℝ) : (2.25 / x) * 12 = 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_value_proof_l1120_112028


namespace NUMINAMATH_CALUDE_soccer_team_win_percentage_l1120_112005

/-- Calculate the percentage of games won by a soccer team -/
theorem soccer_team_win_percentage 
  (total_games : ℕ) 
  (games_won : ℕ) 
  (h1 : total_games = 130) 
  (h2 : games_won = 78) : 
  (games_won : ℚ) / total_games * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_win_percentage_l1120_112005


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l1120_112000

/-- The y-intercept of the line 3x + 5y = 20 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) :
  3 * x + 5 * y = 20 → x = 0 → y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l1120_112000


namespace NUMINAMATH_CALUDE_inequality_implies_bound_l1120_112029

theorem inequality_implies_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, Real.exp x - x > a * x) → a < Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_bound_l1120_112029


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1120_112052

/-- Represents a right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ

/-- The condition that the cylinder's diameter equals its height -/
def cylinder_diameter_equals_height (c : InscribedCylinder) : Prop :=
  2 * c.cylinder_radius = 2 * c.cylinder_radius

/-- The condition that the cone's diameter is 15 -/
def cone_diameter_is_15 (c : InscribedCylinder) : Prop :=
  c.cone_diameter = 15

/-- The condition that the cone's altitude is 15 -/
def cone_altitude_is_15 (c : InscribedCylinder) : Prop :=
  c.cone_altitude = 15

/-- The main theorem: the radius of the inscribed cylinder is 15/4 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) 
  (h1 : cylinder_diameter_equals_height c)
  (h2 : cone_diameter_is_15 c)
  (h3 : cone_altitude_is_15 c) :
  c.cylinder_radius = 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l1120_112052


namespace NUMINAMATH_CALUDE_expression_evaluation_l1120_112087

theorem expression_evaluation : 
  1 / 2^2 + ((2 / 3^3 * (3 / 2)^2) + 4^(1/2)) - 8 / (4^2 - 3^2) = 107/84 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1120_112087


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l1120_112045

-- Define the equation
def equation (y : Real) : Prop :=
  4 * Real.sin y * (Real.cos y)^3 - 4 * (Real.sin y)^3 * Real.cos y = Real.cos y

-- Define the theorem
theorem smallest_positive_angle :
  ∃ (y : Real), y > 0 ∧ y < 360 ∧ equation y ∧
  ∀ (z : Real), z > 0 ∧ z < y → ¬(equation z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l1120_112045


namespace NUMINAMATH_CALUDE_box_height_minimum_l1120_112027

theorem box_height_minimum (x : ℝ) : 
  x > 0 →                           -- side length is positive
  2 * x^2 + 4 * x * (2 * x) ≥ 120 → -- surface area is at least 120
  2 * x ≥ 4 * Real.sqrt 3 :=        -- height (2x) is at least 4√3
by
  sorry

end NUMINAMATH_CALUDE_box_height_minimum_l1120_112027


namespace NUMINAMATH_CALUDE_science_class_end_time_l1120_112033

-- Define the schedule as a list of durations in minutes
def class_schedule : List ℕ := [60, 90, 25, 45, 15, 75]

-- Function to calculate the end time given a start time and a list of durations
def calculate_end_time (start_time : ℕ) (schedule : List ℕ) : ℕ :=
  start_time + schedule.sum

-- Theorem statement
theorem science_class_end_time :
  calculate_end_time 720 class_schedule = 1030 := by
  sorry

-- Note: 720 minutes is 12:00 pm, 1030 minutes is 5:10 pm

end NUMINAMATH_CALUDE_science_class_end_time_l1120_112033


namespace NUMINAMATH_CALUDE_triangle_theorem_triangle_max_sum_l1120_112065

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

theorem triangle_theorem (t : Triangle) (h : 2 * t.c - t.a = 2 * t.b * Real.cos t.A) :
  t.B = π / 3 := by sorry

theorem triangle_max_sum (t : Triangle) 
  (h1 : 2 * t.c - t.a = 2 * t.b * Real.cos t.A) 
  (h2 : t.b = 2 * Real.sqrt 3) :
  (∀ (s : Triangle), s.a + s.c ≤ 4 * Real.sqrt 3) ∧ 
  (∃ (s : Triangle), s.a + s.c = 4 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_triangle_theorem_triangle_max_sum_l1120_112065


namespace NUMINAMATH_CALUDE_equation_solution_l1120_112082

theorem equation_solution (y : ℚ) : 
  (8 * y^2 + 127 * y + 5) / (4 * y + 41) = 2 * y + 3 → y = 118 / 33 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1120_112082


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1120_112043

open Set

def I : Finset Nat := {1,2,3,4,5}
def A : Finset Nat := {2,3,5}
def B : Finset Nat := {1,2}

theorem complement_intersection_theorem :
  (I \ B) ∩ A = {3,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1120_112043


namespace NUMINAMATH_CALUDE_salt_calculation_l1120_112063

/-- Calculates the amount of salt Jack will have after water evaporation -/
def salt_after_evaporation (
  water_volume_day1 : ℝ)
  (water_volume_day2 : ℝ)
  (salt_concentration_day1 : ℝ)
  (salt_concentration_day2 : ℝ)
  (evaporation_rate_day1 : ℝ)
  (evaporation_rate_day2 : ℝ) : ℝ :=
  ((water_volume_day1 * salt_concentration_day1 +
    water_volume_day2 * salt_concentration_day2) * 1000)

theorem salt_calculation :
  salt_after_evaporation 4 4 0.18 0.22 0.30 0.40 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_salt_calculation_l1120_112063


namespace NUMINAMATH_CALUDE_range_of_f_l1120_112048

-- Define the function f
def f (x : ℝ) : ℝ := x + |x - 2|

-- State the theorem about the range of f
theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1120_112048


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l1120_112032

/-- The number of girls in a school, given certain sampling conditions -/
theorem number_of_girls_in_school :
  ∀ (total_students sample_size : ℕ) (girls_in_school : ℕ),
  total_students = 1600 →
  sample_size = 200 →
  girls_in_school ≤ total_students →
  (girls_in_school : ℚ) / (total_students - girls_in_school : ℚ) = 95 / 105 →
  girls_in_school = 760 := by
sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l1120_112032


namespace NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l1120_112007

def base_seven_representation (n : ℕ) : List ℕ := sorry

def digit_sum (digits : List ℕ) : ℕ := sorry

def is_valid_base_seven (digits : List ℕ) : Prop := sorry

theorem greatest_digit_sum_base_seven :
  ∃ (n : ℕ), n < 2890 ∧
    (∀ (m : ℕ), m < 2890 →
      digit_sum (base_seven_representation m) ≤ digit_sum (base_seven_representation n)) ∧
    digit_sum (base_seven_representation n) = 23 :=
  sorry

end NUMINAMATH_CALUDE_greatest_digit_sum_base_seven_l1120_112007


namespace NUMINAMATH_CALUDE_min_likes_mozart_and_beethoven_l1120_112021

/-- Given a survey of 150 people where 120 liked Mozart and 80 liked Beethoven,
    the minimum number of people who liked both Mozart and Beethoven is 50. -/
theorem min_likes_mozart_and_beethoven
  (total : ℕ) (likes_mozart : ℕ) (likes_beethoven : ℕ)
  (h_total : total = 150)
  (h_mozart : likes_mozart = 120)
  (h_beethoven : likes_beethoven = 80) :
  (likes_mozart + likes_beethoven - total : ℤ).natAbs ≥ 50 := by
  sorry


end NUMINAMATH_CALUDE_min_likes_mozart_and_beethoven_l1120_112021


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1120_112080

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = Real.sqrt 10) →
  (a * b = t - 3 * Complex.I) →
  (t > 0) →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1120_112080


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l1120_112016

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 75 →
  yellow_students = 105 →
  total_students = blue_students + yellow_students →
  total_pairs = 90 →
  blue_blue_pairs = 30 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 ∧ 
    yellow_yellow_pairs = (yellow_students - (total_students - 2 * blue_blue_pairs)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l1120_112016


namespace NUMINAMATH_CALUDE_prob_top_joker_modified_deck_l1120_112023

/-- A deck of cards with jokers -/
structure Deck :=
  (total_cards : ℕ)
  (standard_cards : ℕ)
  (jokers : ℕ)
  (h_total : total_cards = standard_cards + jokers)
  (h_jokers : jokers = 2)

/-- The probability of drawing a joker from the top of a shuffled deck -/
def prob_top_joker (d : Deck) : ℚ :=
  d.jokers / d.total_cards

/-- Theorem stating the probability of drawing a joker from a modified 54-card deck -/
theorem prob_top_joker_modified_deck :
  ∃ (d : Deck), d.total_cards = 54 ∧ d.standard_cards = 52 ∧ prob_top_joker d = 1 / 27 := by
  sorry


end NUMINAMATH_CALUDE_prob_top_joker_modified_deck_l1120_112023


namespace NUMINAMATH_CALUDE_five_seventeenths_repetend_l1120_112037

/-- The repetend of a rational number a/b is the repeating sequence of digits in its decimal expansion. -/
def repetend (a b : ℕ) : List ℕ := sorry

/-- Returns the first n digits of a list. -/
def firstNDigits (n : ℕ) (l : List ℕ) : List ℕ := sorry

theorem five_seventeenths_repetend :
  firstNDigits 6 (repetend 5 17) = [2, 9, 4, 1, 1, 7] := by sorry

end NUMINAMATH_CALUDE_five_seventeenths_repetend_l1120_112037


namespace NUMINAMATH_CALUDE_newer_car_distance_l1120_112096

theorem newer_car_distance (older_distance : ℝ) (percentage_increase : ℝ) 
  (h1 : older_distance = 150)
  (h2 : percentage_increase = 0.30) : 
  older_distance * (1 + percentage_increase) = 195 :=
by sorry

end NUMINAMATH_CALUDE_newer_car_distance_l1120_112096


namespace NUMINAMATH_CALUDE_sum_of_coefficients_for_factored_form_l1120_112041

theorem sum_of_coefficients_for_factored_form : ∃ (a b c d e f : ℤ),
  (2401 : ℤ) * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f) ∧
  a + b + c + d + e + f = 274 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_for_factored_form_l1120_112041


namespace NUMINAMATH_CALUDE_fraction_simplification_l1120_112072

theorem fraction_simplification (m : ℝ) (hm : m ≠ 0 ∧ m ≠ 1) : 
  (m - 1) / m / ((m - 1) / (m^2)) = m := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1120_112072


namespace NUMINAMATH_CALUDE_student_expected_score_l1120_112088

theorem student_expected_score :
  let total_questions : ℕ := 12
  let points_per_question : ℝ := 5
  let confident_questions : ℕ := 6
  let eliminate_one_questions : ℕ := 3
  let eliminate_two_questions : ℕ := 2
  let random_guess_questions : ℕ := 1
  let prob_correct_confident : ℝ := 1
  let prob_correct_eliminate_one : ℝ := 1/4
  let prob_correct_eliminate_two : ℝ := 1/3
  let prob_correct_random : ℝ := 1/4

  let expected_score : ℝ :=
    points_per_question * (
      confident_questions * prob_correct_confident +
      eliminate_one_questions * prob_correct_eliminate_one +
      eliminate_two_questions * prob_correct_eliminate_two +
      random_guess_questions * prob_correct_random
    )

  total_questions = confident_questions + eliminate_one_questions + eliminate_two_questions + random_guess_questions →
  expected_score = 41.25 :=
by sorry

end NUMINAMATH_CALUDE_student_expected_score_l1120_112088


namespace NUMINAMATH_CALUDE_expected_value_of_coins_l1120_112098

-- Define coin values in cents
def penny : ℚ := 1
def nickel : ℚ := 5
def dime : ℚ := 10
def quarter : ℚ := 25
def half_dollar : ℚ := 50

-- Define probability of heads
def prob_heads : ℚ := 1/2

-- Define function to calculate expected value for a single coin
def expected_value (coin_value : ℚ) : ℚ := prob_heads * coin_value

-- Theorem statement
theorem expected_value_of_coins : 
  expected_value penny + expected_value nickel + expected_value dime + 
  expected_value quarter + expected_value half_dollar = 45.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_coins_l1120_112098


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1120_112060

theorem binomial_coefficient_ratio (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) ↔ n = 43 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l1120_112060


namespace NUMINAMATH_CALUDE_race_percentage_l1120_112071

theorem race_percentage (v_Q : ℝ) (h : v_Q > 0) : 
  let v_P := v_Q * (1 + 25/100)
  (300 / v_P = (300 - 60) / v_Q) → 
  ∃ (p : ℝ), v_P = v_Q * (1 + p/100) ∧ p = 25 :=
by sorry

end NUMINAMATH_CALUDE_race_percentage_l1120_112071
