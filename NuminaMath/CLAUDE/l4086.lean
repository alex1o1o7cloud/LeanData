import Mathlib

namespace NUMINAMATH_CALUDE_larger_integer_proof_l4086_408669

theorem larger_integer_proof (x : ℤ) : 
  (x > 0) →  -- Ensure x is positive
  (x + 6 : ℚ) / (4 * x : ℚ) = 1 / 3 → 
  4 * x = 72 :=
by sorry

end NUMINAMATH_CALUDE_larger_integer_proof_l4086_408669


namespace NUMINAMATH_CALUDE_ellipse_foci_range_l4086_408652

/-- Given an ellipse with equation x²/9 + y²/m² = 1 where the foci are on the x-axis,
    prove that the range of m is (-3, 0) ∪ (0, 3) -/
theorem ellipse_foci_range (m : ℝ) : 
  (∃ x y : ℝ, x^2/9 + y^2/m^2 = 1 ∧ (∃ c : ℝ, c > 0 ∧ c < 3 ∧ 
    (∀ x y : ℝ, x^2/9 + y^2/m^2 = 1 → x^2 - y^2 = c^2))) ↔ 
  m ∈ Set.union (Set.Ioo (-3) 0) (Set.Ioo 0 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_range_l4086_408652


namespace NUMINAMATH_CALUDE_special_deck_probability_l4086_408684

/-- A deck of cards with specified properties -/
structure Deck :=
  (total : ℕ)
  (red_non_joker : ℕ)
  (black_or_joker : ℕ)

/-- The probability of drawing a red non-joker card first and a black or joker card second -/
def draw_probability (d : Deck) : ℚ :=
  (d.red_non_joker : ℚ) * (d.black_or_joker : ℚ) / ((d.total : ℚ) * (d.total - 1 : ℚ))

/-- Theorem stating the probability for the specific deck described in the problem -/
theorem special_deck_probability :
  let d := Deck.mk 60 26 40
  draw_probability d = 5 / 17 := by sorry

end NUMINAMATH_CALUDE_special_deck_probability_l4086_408684


namespace NUMINAMATH_CALUDE_optimal_pasture_length_l4086_408693

/-- Represents a rectangular cow pasture -/
structure Pasture where
  width : ℝ  -- Width of the pasture (perpendicular to the barn)
  length : ℝ  -- Length of the pasture (parallel to the barn)

/-- Calculates the area of the pasture -/
def Pasture.area (p : Pasture) : ℝ := p.width * p.length

/-- Theorem: The optimal length of the pasture that maximizes the area -/
theorem optimal_pasture_length (total_fence : ℝ) (barn_length : ℝ) :
  total_fence = 240 →
  barn_length = 600 →
  ∃ (optimal : Pasture),
    optimal.length = 120 ∧
    optimal.width = (total_fence - optimal.length) / 2 ∧
    ∀ (p : Pasture),
      p.length + 2 * p.width = total_fence →
      p.area ≤ optimal.area := by
  sorry

end NUMINAMATH_CALUDE_optimal_pasture_length_l4086_408693


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l4086_408625

theorem complex_number_opposite_parts (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (2 + Complex.I)
  (Complex.re z = -Complex.im z) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l4086_408625


namespace NUMINAMATH_CALUDE_sophias_book_length_l4086_408628

theorem sophias_book_length (total_pages : ℕ) : 
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 90 → 
  total_pages = 270 := by
sorry

end NUMINAMATH_CALUDE_sophias_book_length_l4086_408628


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l4086_408645

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a circle equation, returns its properties (center and radius) -/
def circle_properties (eq : CircleEquation) : CircleProperties :=
  sorry

theorem circle_center_and_radius 
  (eq : CircleEquation) 
  (h : eq = ⟨-6, 0, 0⟩) : 
  circle_properties eq = ⟨(3, 0), 3⟩ :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l4086_408645


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l4086_408664

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m.val n.val = 12) :
  ∃ (k : ℕ+), k.val = Nat.gcd (8 * m.val) (18 * n.val) ∧ 
  ∀ (l : ℕ+), l.val = Nat.gcd (8 * m.val) (18 * n.val) → k ≤ l ∧ k.val = 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l4086_408664


namespace NUMINAMATH_CALUDE_newspaper_prices_l4086_408694

theorem newspaper_prices :
  ∃ (x y : ℕ) (k : ℚ),
    x < 30 ∧ y < 30 ∧ 0 < k ∧ k < 1 ∧
    k * 30 = y ∧ k * x = 15 ∧
    ((x = 25 ∧ y = 18) ∨ (x = 18 ∧ y = 25)) ∧
    ∀ (x' y' : ℕ) (k' : ℚ),
      x' < 30 → y' < 30 → 0 < k' → k' < 1 →
      k' * 30 = y' → k' * x' = 15 →
      ((x' = 25 ∧ y' = 18) ∨ (x' = 18 ∧ y' = 25)) :=
by sorry

end NUMINAMATH_CALUDE_newspaper_prices_l4086_408694


namespace NUMINAMATH_CALUDE_ratio_of_segments_l4086_408602

-- Define the points on a line
variable (E F G H : ℝ)

-- Define the conditions
variable (h1 : E < F)
variable (h2 : F < G)
variable (h3 : G < H)
variable (h4 : F - E = 3)
variable (h5 : G - F = 8)
variable (h6 : H - E = 23)

-- Theorem statement
theorem ratio_of_segments :
  (G - E) / (H - F) = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l4086_408602


namespace NUMINAMATH_CALUDE_ball_ratio_problem_l4086_408695

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 3 / 2 →
  white_balls = 9 →
  red_balls = 6 := by
sorry

end NUMINAMATH_CALUDE_ball_ratio_problem_l4086_408695


namespace NUMINAMATH_CALUDE_archibalds_apples_l4086_408627

/-- Archibald's apple eating problem -/
theorem archibalds_apples :
  let apples_per_day_first_two_weeks : ℕ := 1
  let weeks_first_period : ℕ := 2
  let weeks_second_period : ℕ := 3
  let weeks_third_period : ℕ := 2
  let total_weeks : ℕ := weeks_first_period + weeks_second_period + weeks_third_period
  let average_apples_per_week : ℕ := 10
  let total_apples : ℕ := average_apples_per_week * total_weeks
  let apples_first_two_weeks : ℕ := apples_per_day_first_two_weeks * 7 * weeks_first_period
  let apples_next_three_weeks : ℕ := apples_first_two_weeks * weeks_second_period
  let apples_last_two_weeks : ℕ := total_apples - apples_first_two_weeks - apples_next_three_weeks
  apples_last_two_weeks / (7 * weeks_third_period) = 1 :=
by sorry


end NUMINAMATH_CALUDE_archibalds_apples_l4086_408627


namespace NUMINAMATH_CALUDE_geometric_sequence_determinant_l4086_408622

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_determinant
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a5 : a 5 = 3) :
  let det := a 2 * a 8 + a 7 * a 3
  det = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_determinant_l4086_408622


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l4086_408681

/-- Given two points on a line and another line equation, find the value of k that makes the lines parallel -/
theorem parallel_lines_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ 6 * x - 2 * y = -8) ∧ 
   (23 - (-4)) / (k - 5) = m) → 
  k = 14 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l4086_408681


namespace NUMINAMATH_CALUDE_linear_function_property_l4086_408638

/-- A linear function f where f(6) - f(2) = 12 -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) ∧ 
  (f 6 - f 2 = 12)

theorem linear_function_property (f : ℝ → ℝ) (h : LinearFunction f) : 
  f 12 - f 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l4086_408638


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l4086_408633

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg : ℚ) (correct_mark : ℕ) :
  n = 30 ∧ 
  initial_avg = 60 ∧ 
  correct_avg = 57.5 ∧ 
  correct_mark = 15 →
  ∃ wrong_mark : ℕ,
    (n * initial_avg - wrong_mark + correct_mark) / n = correct_avg ∧
    wrong_mark = 90 := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l4086_408633


namespace NUMINAMATH_CALUDE_min_xy_equals_nine_l4086_408613

theorem min_xy_equals_nine (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  ∀ z, z = x * y → z ≥ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 1 / (b + 1) = 1 / 2 ∧ a * b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_equals_nine_l4086_408613


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l4086_408675

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

/-- The period of a function -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the equation -/
def LeastCommonPeriod (p : ℝ) : Prop :=
  p > 0 ∧
  (∀ f : ℝ → ℝ, SatisfiesEquation f → IsPeriod f p) ∧
  (∀ q : ℝ, q > 0 → (∀ f : ℝ → ℝ, SatisfiesEquation f → IsPeriod f q) → p ≤ q)

theorem least_common_period_is_36 :
  LeastCommonPeriod 36 := by sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l4086_408675


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l4086_408672

def product : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 10000

theorem units_digit_of_fraction :
  (product / denominator) % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l4086_408672


namespace NUMINAMATH_CALUDE_inequality_condition_l4086_408617

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x - 5| + |x - 10| < b) ↔ b > 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l4086_408617


namespace NUMINAMATH_CALUDE_correct_freshmen_sample_l4086_408662

/-- Represents a stratified sampling scenario in a college -/
structure CollegeSampling where
  total_students : ℕ
  freshmen : ℕ
  sample_size : ℕ

/-- Calculates the number of freshmen to be sampled in a stratified sampling -/
def freshmen_in_sample (cs : CollegeSampling) : ℕ :=
  cs.sample_size * cs.freshmen / cs.total_students

/-- Theorem stating the correct number of freshmen to be sampled -/
theorem correct_freshmen_sample (cs : CollegeSampling) 
  (h1 : cs.total_students = 3000)
  (h2 : cs.freshmen = 800)
  (h3 : cs.sample_size = 300) :
  freshmen_in_sample cs = 80 :=
sorry

end NUMINAMATH_CALUDE_correct_freshmen_sample_l4086_408662


namespace NUMINAMATH_CALUDE_always_possible_to_sell_tickets_l4086_408660

/-- Represents the amount a child pays (5 or 10 yuan) -/
inductive Payment
| five : Payment
| ten : Payment

/-- A queue of children represented by their payments -/
def Queue := List Payment

/-- Counts the number of each type of payment in a queue -/
def countPayments (q : Queue) : ℕ × ℕ :=
  q.foldl (λ (five, ten) p => match p with
    | Payment.five => (five + 1, ten)
    | Payment.ten => (five, ten + 1)
  ) (0, 0)

/-- Checks if it's possible to give change at each step -/
def canGiveChange (q : Queue) : Prop :=
  q.foldl (λ acc p => match p with
    | Payment.five => acc + 1
    | Payment.ten => acc - 1
  ) 0 ≥ 0

/-- The main theorem stating that it's always possible to sell tickets without running out of change -/
theorem always_possible_to_sell_tickets (q : Queue) :
  let (fives, tens) := countPayments q
  fives = tens → q.length = 2 * fives → canGiveChange q :=
sorry

#check always_possible_to_sell_tickets

end NUMINAMATH_CALUDE_always_possible_to_sell_tickets_l4086_408660


namespace NUMINAMATH_CALUDE_total_prizes_l4086_408623

def stuffed_animals : ℕ := 14
def frisbees : ℕ := 18
def yo_yos : ℕ := 18

theorem total_prizes : stuffed_animals + frisbees + yo_yos = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_prizes_l4086_408623


namespace NUMINAMATH_CALUDE_minimum_red_chips_l4086_408616

/-- Represents the number of chips of each color in the box -/
structure ChipCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if the chip count satisfies all given conditions -/
def satisfiesConditions (c : ChipCount) : Prop :=
  c.blue ≥ (3 * c.white) / 4 ∧
  c.blue ≤ c.red / 4 ∧
  60 ≤ c.white + c.blue ∧
  c.white + c.blue ≤ 80

/-- The minimum number of red chips that satisfies all conditions -/
def minRedChips : ℕ := 108

theorem minimum_red_chips :
  ∀ c : ChipCount, satisfiesConditions c → c.red ≥ minRedChips :=
sorry


end NUMINAMATH_CALUDE_minimum_red_chips_l4086_408616


namespace NUMINAMATH_CALUDE_matthew_initial_cakes_l4086_408604

/-- The number of friends Matthew has -/
def num_friends : ℕ := 4

/-- The initial number of crackers Matthew has -/
def initial_crackers : ℕ := 10

/-- The number of cakes each person eats -/
def cakes_eaten_per_person : ℕ := 2

/-- The number of crackers given to each friend -/
def crackers_per_friend : ℕ := initial_crackers / num_friends

/-- The initial number of cakes Matthew had -/
def initial_cakes : ℕ := 2 * num_friends * crackers_per_friend

theorem matthew_initial_cakes :
  initial_cakes = 16 :=
sorry

end NUMINAMATH_CALUDE_matthew_initial_cakes_l4086_408604


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l4086_408647

-- Problem 1
theorem equation_one_solutions (x : ℝ) :
  4 * x^2 - 16 = 0 ↔ x = 2 ∨ x = -2 :=
sorry

-- Problem 2
theorem equation_two_solution (x : ℝ) :
  (2*x - 1)^3 + 64 = 0 ↔ x = -3/2 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l4086_408647


namespace NUMINAMATH_CALUDE_power_function_through_point_l4086_408644

/-- Given a power function f(x) = x^a that passes through the point (2, 4), prove that f(x) = x^2 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →
  f 2 = 4 →
  ∀ x, f x = x ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l4086_408644


namespace NUMINAMATH_CALUDE_ribbon_length_l4086_408632

theorem ribbon_length (A : ℝ) (π : ℝ) (h1 : A = 616) (h2 : π = 22 / 7) : 
  let r := Real.sqrt (A / π)
  let C := 2 * π * r
  C + 5 = 93 := by sorry

end NUMINAMATH_CALUDE_ribbon_length_l4086_408632


namespace NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l4086_408670

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder fits inside a crate -/
def cylinderFitsInCrate (c : Cylinder) (d : CrateDimensions) : Prop :=
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.width ∧ c.height ≤ d.height) ∨
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.height ∧ c.height ≤ d.width) ∨
  (2 * c.radius ≤ d.width ∧ 2 * c.radius ≤ d.height ∧ c.height ≤ d.length)

/-- The main theorem stating that the maximum radius of a cylinder that fits in the given crate is 1.5 feet -/
theorem max_cylinder_radius_in_crate :
  let d := CrateDimensions.mk 3 8 12
  ∀ c : Cylinder, cylinderFitsInCrate c d → c.radius ≤ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l4086_408670


namespace NUMINAMATH_CALUDE_no_solution_to_system_l4086_408634

theorem no_solution_to_system :
  ¬∃ (x : ℝ), 
    (|Real.log x / Real.log 2| + (4 * x^2 / 15) - (16/15) = 0) ∧ 
    (Real.log (x + 2/3) / Real.log 7 + 12*x - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l4086_408634


namespace NUMINAMATH_CALUDE_odd_and_div_by_5_probability_l4086_408677

/-- A set of digits to form a four-digit number -/
def digits : Finset Nat := {8, 5, 9, 7}

/-- Predicate for a number being odd and divisible by 5 -/
def is_odd_and_div_by_5 (n : Nat) : Prop :=
  n % 2 = 1 ∧ n % 5 = 0

/-- The total number of possible four-digit numbers -/
def total_permutations : Nat := Nat.factorial 4

/-- The number of valid permutations (odd and divisible by 5) -/
def valid_permutations : Nat := Nat.factorial 3

/-- The probability of forming a number that is odd and divisible by 5 -/
def probability : Rat := valid_permutations / total_permutations

theorem odd_and_div_by_5_probability :
  probability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_odd_and_div_by_5_probability_l4086_408677


namespace NUMINAMATH_CALUDE_chimney_bricks_total_bricks_correct_l4086_408649

-- Define the problem parameters
def brenda_time : ℝ := 8
def brandon_time : ℝ := 12
def combined_decrease : ℝ := 12
def combined_time : ℝ := 6

-- Define the theorem
theorem chimney_bricks : ∃ (h : ℝ),
  h > 0 ∧
  h / brenda_time + h / brandon_time - combined_decrease = h / combined_time :=
by
  -- The proof goes here
  sorry

-- Define the final answer
def total_bricks : ℕ := 288

-- Prove that the total_bricks satisfies the theorem
theorem total_bricks_correct : 
  ∃ (h : ℝ), h = total_bricks ∧
  h > 0 ∧
  h / brenda_time + h / brandon_time - combined_decrease = h / combined_time :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_chimney_bricks_total_bricks_correct_l4086_408649


namespace NUMINAMATH_CALUDE_opposite_faces_l4086_408612

/-- Represents the six faces of a cube -/
inductive Face : Type
  | xiao : Face  -- 小
  | xue  : Face  -- 学
  | xi   : Face  -- 希
  | wang : Face  -- 望
  | bei  : Face  -- 杯
  | sai  : Face  -- 赛

/-- Defines the adjacency relationship between faces -/
def adjacent : Face → Face → Prop :=
  sorry

/-- Defines the opposite relationship between faces -/
def opposite : Face → Face → Prop :=
  sorry

/-- The cube configuration satisfies the given conditions -/
axiom cube_config :
  adjacent Face.xue Face.xiao ∧
  adjacent Face.xue Face.xi ∧
  adjacent Face.xue Face.wang ∧
  adjacent Face.xue Face.sai

/-- Theorem stating the opposite face relationships -/
theorem opposite_faces :
  opposite Face.xi Face.sai ∧
  opposite Face.wang Face.xiao ∧
  opposite Face.bei Face.xue :=
by sorry

end NUMINAMATH_CALUDE_opposite_faces_l4086_408612


namespace NUMINAMATH_CALUDE_age_problem_l4086_408659

theorem age_problem (age : ℕ) : 5 * (age + 5) - 5 * (age - 5) = age → age = 50 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l4086_408659


namespace NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l4086_408658

theorem largest_n_for_quadratic_equation : ∃ (x y z : ℕ+),
  13^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 6*x + 6*y + 6*z - 12 ∧
  ∀ (n : ℕ+), n > 13 →
    ¬∃ (a b c : ℕ+), n^2 = a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a + 6*a + 6*b + 6*c - 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l4086_408658


namespace NUMINAMATH_CALUDE_die_roll_frequency_l4086_408699

/-- The frequency of a specific outcome in a series of trials -/
def frequency (successful_outcomes : ℕ) (total_trials : ℕ) : ℚ :=
  successful_outcomes / total_trials

/-- The number of times the die was rolled -/
def total_rolls : ℕ := 60

/-- The number of times six appeared -/
def six_appearances : ℕ := 10

theorem die_roll_frequency :
  frequency six_appearances total_rolls = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_frequency_l4086_408699


namespace NUMINAMATH_CALUDE_remainder_of_polynomial_l4086_408668

theorem remainder_of_polynomial (n : ℤ) (k : ℤ) : 
  n = 100 * k - 1 → (n^2 + 3*n + 4) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_polynomial_l4086_408668


namespace NUMINAMATH_CALUDE_cos_2x_value_l4086_408690

theorem cos_2x_value (x : Real) (h : 2 * Real.sin x + Real.cos (π / 2 - x) = 1) :
  Real.cos (2 * x) = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_cos_2x_value_l4086_408690


namespace NUMINAMATH_CALUDE_circle_properties_l4086_408641

/-- Given a circle with circumference 31.4 decimeters, prove its diameter, radius, and area -/
theorem circle_properties (C : Real) (h : C = 31.4) :
  ∃ (d r A : Real),
    d = 10 ∧ 
    r = 5 ∧ 
    A = 78.5 ∧
    C = 2 * Real.pi * r ∧
    d = 2 * r ∧
    A = Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l4086_408641


namespace NUMINAMATH_CALUDE_set_operation_result_l4086_408665

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2, 4}

-- Define set B
def B : Set Nat := {2, 3, 5}

-- Theorem statement
theorem set_operation_result :
  (U \ A) ∪ B = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l4086_408665


namespace NUMINAMATH_CALUDE_point_transformation_l4086_408692

def initial_point : ℝ × ℝ × ℝ := (2, 2, 2)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p |> rotate_z_90
    |> reflect_xy
    |> reflect_yz
    |> rotate_x_90
    |> reflect_xy

theorem point_transformation :
  transform initial_point = (2, -2, 2) := by sorry

end NUMINAMATH_CALUDE_point_transformation_l4086_408692


namespace NUMINAMATH_CALUDE_percentage_loss_calculation_l4086_408685

theorem percentage_loss_calculation (cost_price selling_price : ℝ) : 
  cost_price = 1400 →
  selling_price = 1120 →
  (cost_price - selling_price) / cost_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_loss_calculation_l4086_408685


namespace NUMINAMATH_CALUDE_tetrahedron_volume_prove_tetrahedron_volume_l4086_408614

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ
  /-- Conditions on the tetrahedron -/
  ab_length_eq : ab_length = 3
  abc_area_eq : abc_area = 15
  abd_area_eq : abd_area = 12
  face_angle_eq : face_angle = Real.pi / 6

/-- The volume of the tetrahedron is 20 cm³ -/
theorem tetrahedron_volume (t : Tetrahedron) : ℝ :=
  20

#check tetrahedron_volume

/-- Proof of the tetrahedron volume -/
theorem prove_tetrahedron_volume (t : Tetrahedron) :
  tetrahedron_volume t = 20 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_prove_tetrahedron_volume_l4086_408614


namespace NUMINAMATH_CALUDE_total_revenue_is_2176_l4086_408642

def kitten_price : ℕ := 80
def puppy_price : ℕ := 150
def rabbit_price : ℕ := 45
def guinea_pig_price : ℕ := 30

def kitten_count : ℕ := 10
def puppy_count : ℕ := 8
def rabbit_count : ℕ := 4
def guinea_pig_count : ℕ := 6

def discount_rate : ℚ := 1/10

def total_revenue : ℚ := 
  (kitten_count * kitten_price + 
   puppy_count * puppy_price + 
   rabbit_count * rabbit_price + 
   guinea_pig_count * guinea_pig_price : ℚ) - 
  (min kitten_count puppy_count * discount_rate * (kitten_price + puppy_price))

theorem total_revenue_is_2176 : total_revenue = 2176 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_2176_l4086_408642


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l4086_408631

/-- The equation x^2 - 16y^2 - 10x + 4y + 36 = 0 represents a hyperbola. -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (A B : ℝ),
    A > 0 ∧ B > 0 ∧
    ∀ x y : ℝ,
      x^2 - 16*y^2 - 10*x + 4*y + 36 = 0 ↔
      ((x - h)^2 / A - (y - k)^2 / B = 1 ∨ (x - h)^2 / A - (y - k)^2 / B = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l4086_408631


namespace NUMINAMATH_CALUDE_intersection_of_lines_l4086_408610

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) :
  A = (5, -6, 8) →
  B = (15, -16, 13) →
  C = (1, 4, -5) →
  D = (3, -4, 11) →
  ∃ t s : ℝ,
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (1 + 2*s, 4 - 8*s, -5 + 16*s) ∧
    (5 + 10*t, -6 - 10*t, 8 + 5*t) = (-10/3, 14/3, -1/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l4086_408610


namespace NUMINAMATH_CALUDE_triangle_side_length_range_l4086_408624

theorem triangle_side_length_range (a b c : ℝ) :
  (|a + b - 4| + (a - b + 2)^2 = 0) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (2 < c ∧ c < 4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_range_l4086_408624


namespace NUMINAMATH_CALUDE_factorization_problems_l4086_408643

theorem factorization_problems :
  (∀ x y : ℝ, xy - 1 - x + y = (y - 1) * (x + 1)) ∧
  (∀ a b : ℝ, (a^2 + b^2)^2 - 4 * a^2 * b^2 = (a + b)^2 * (a - b)^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_problems_l4086_408643


namespace NUMINAMATH_CALUDE_solution_pairs_l4086_408678

theorem solution_pairs : ∀ x y : ℝ, 
  (x + y + 4 = (12*x + 11*y) / (x^2 + y^2) ∧ 
   y - x + 3 = (11*x - 12*y) / (x^2 + y^2)) ↔ 
  ((x = 2 ∧ y = 1) ∨ (x = -2.5 ∧ y = -4.5)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l4086_408678


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l4086_408608

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 60) : 
  (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l4086_408608


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l4086_408609

theorem cone_lateral_surface_area 
  (r : ℝ) (h : ℝ) (l : ℝ) (S : ℝ) 
  (h_r : r = 2) 
  (h_h : h = 4 * Real.sqrt 2) 
  (h_l : l^2 = r^2 + h^2) 
  (h_S : S = π * r * l) : 
  S = 12 * π := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l4086_408609


namespace NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l4086_408605

theorem no_consecutive_integers_with_square_diff_2000 :
  ¬ ∃ (x : ℤ), (x + 1)^2 - x^2 = 2000 := by sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_with_square_diff_2000_l4086_408605


namespace NUMINAMATH_CALUDE_printing_presses_l4086_408640

theorem printing_presses (time1 time2 : ℝ) (newspapers1 newspapers2 : ℕ) (presses2 : ℕ) :
  time1 = 6 →
  time2 = 9 →
  newspapers1 = 8000 →
  newspapers2 = 6000 →
  presses2 = 2 →
  ∃ (presses1 : ℕ), 
    (presses1 : ℝ) * (newspapers2 : ℝ) / (time2 * presses2) = newspapers1 / time1 ∧
    presses1 = 4 :=
by sorry


end NUMINAMATH_CALUDE_printing_presses_l4086_408640


namespace NUMINAMATH_CALUDE_hex_B1C_equals_2844_l4086_408689

/-- Converts a hexadecimal digit to its decimal value -/
def hexToDecimal (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hexStringToDecimal (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hexToDecimal c) 0

/-- The hexadecimal number B1C is equal to 2844 in decimal -/
theorem hex_B1C_equals_2844 : hexStringToDecimal "B1C" = 2844 := by
  sorry

end NUMINAMATH_CALUDE_hex_B1C_equals_2844_l4086_408689


namespace NUMINAMATH_CALUDE_acid_concentration_increase_l4086_408673

theorem acid_concentration_increase (initial_volume initial_concentration water_removed : ℝ) :
  initial_volume = 18 →
  initial_concentration = 0.4 →
  water_removed = 6 →
  let acid_amount := initial_volume * initial_concentration
  let final_volume := initial_volume - water_removed
  let final_concentration := acid_amount / final_volume
  final_concentration = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_increase_l4086_408673


namespace NUMINAMATH_CALUDE_inequality_proof_l4086_408615

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l4086_408615


namespace NUMINAMATH_CALUDE_common_tangents_count_l4086_408691

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

-- Define the number of common tangents
def num_common_tangents (C₁ C₂ : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  num_common_tangents C₁ C₂ = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l4086_408691


namespace NUMINAMATH_CALUDE_divisor_property_l4086_408626

theorem divisor_property (k : ℕ) : 
  (30 ^ k : ℕ) ∣ 929260 → 3 ^ k - k ^ 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l4086_408626


namespace NUMINAMATH_CALUDE_propositions_true_l4086_408676

-- Define the propositions
def proposition1 (x y : ℝ) : Prop := x + y = 0 → (x = -y ∨ y = -x)
def proposition3 (q : ℝ) : Prop := q ≤ 1 → ∃ x : ℝ, x^2 + 2*x + q = 0

-- Theorem statement
theorem propositions_true :
  (∀ x y : ℝ, ¬(x + y = 0) → ¬(x = -y ∨ y = -x)) ∧
  (∀ q : ℝ, (¬∃ x : ℝ, x^2 + 2*x + q = 0) → ¬(q ≤ 1)) := by sorry

end NUMINAMATH_CALUDE_propositions_true_l4086_408676


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4086_408607

theorem sum_of_squares_of_roots (u v w : ℝ) : 
  (3 * u^3 - 7 * u^2 + 6 * u + 15 = 0) →
  (3 * v^3 - 7 * v^2 + 6 * v + 15 = 0) →
  (3 * w^3 - 7 * w^2 + 6 * w + 15 = 0) →
  u^2 + v^2 + w^2 = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4086_408607


namespace NUMINAMATH_CALUDE_power_of_three_equality_l4086_408601

theorem power_of_three_equality : 3^1999 - 3^1998 - 3^1997 + 3^1996 = 16 * 3^1996 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l4086_408601


namespace NUMINAMATH_CALUDE_consecutive_multiples_problem_l4086_408667

/-- Given a set of 50 consecutive multiples of a number, prove that the number is 2 -/
theorem consecutive_multiples_problem (n : ℕ) (s : Set ℕ) : 
  (∃ k : ℕ, s = {k * n | k ∈ Finset.range 50}) →  -- s is a set of 50 consecutive multiples of n
  (56 ∈ s) →  -- The smallest number in s is 56
  (154 ∈ s) →  -- The greatest number in s is 154
  (∀ x ∈ s, 56 ≤ x ∧ x ≤ 154) →  -- All elements in s are between 56 and 154
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_multiples_problem_l4086_408667


namespace NUMINAMATH_CALUDE_rejected_products_percentage_l4086_408618

theorem rejected_products_percentage
  (john_reject_rate : ℝ)
  (jane_reject_rate : ℝ)
  (jane_inspect_fraction : ℝ)
  (h1 : john_reject_rate = 0.007)
  (h2 : jane_reject_rate = 0.008)
  (h3 : jane_inspect_fraction = 0.5)
  : (john_reject_rate * (1 - jane_inspect_fraction) + jane_reject_rate * jane_inspect_fraction) * 100 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_rejected_products_percentage_l4086_408618


namespace NUMINAMATH_CALUDE_stone_arrangement_exists_l4086_408653

theorem stone_arrangement_exists (P : ℕ) (h : P = 23) : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ 
  F 1 = 1 ∧ 
  (∀ i : ℕ, i ≥ 2 → F i = 3 * F (i - 1) - F (i - 2)) ∧
  F 12 % P = 0 :=
by sorry

end NUMINAMATH_CALUDE_stone_arrangement_exists_l4086_408653


namespace NUMINAMATH_CALUDE_remainder_of_expression_l4086_408611

theorem remainder_of_expression (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^(p + t) + 11^t * 6^(p*t)) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_expression_l4086_408611


namespace NUMINAMATH_CALUDE_cafeteria_total_l4086_408657

/-- The total number of people in a cafeteria with checkered, horizontal, and vertical striped shirts -/
def total_people (checkered : ℕ) (horizontal : ℕ) (vertical : ℕ) : ℕ :=
  checkered + horizontal + vertical

/-- Theorem: The total number of people in the cafeteria is 40 -/
theorem cafeteria_total : 
  ∃ (checkered horizontal vertical : ℕ),
    checkered = 7 ∧ 
    horizontal = 4 * checkered ∧ 
    vertical = 5 ∧ 
    total_people checkered horizontal vertical = 40 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_total_l4086_408657


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_4_l4086_408683

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2010WithDigitSum4 (year : Nat) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 4 ∧
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 4

theorem first_year_after_2010_with_digit_sum_4 :
  isFirstYearAfter2010WithDigitSum4 2011 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_4_l4086_408683


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_university_sample_sizes_correct_l4086_408656

/-- Represents a stratum in a population -/
structure Stratum where
  size : ℕ

/-- Represents a population with stratified sampling -/
structure StratifiedPopulation where
  total : ℕ
  strata : List Stratum
  sample_size : ℕ

/-- Calculates the number of samples for a given stratum -/
def sample_size_for_stratum (pop : StratifiedPopulation) (stratum : Stratum) : ℕ :=
  (pop.sample_size * stratum.size) / pop.total

/-- Theorem: The sum of samples from all strata equals the total sample size -/
theorem stratified_sampling_theorem (pop : StratifiedPopulation) 
  (h : pop.total = (pop.strata.map Stratum.size).sum) :
  (pop.strata.map (sample_size_for_stratum pop)).sum = pop.sample_size := by
  sorry

/-- The university population -/
def university_pop : StratifiedPopulation :=
  { total := 5600
  , strata := [⟨1300⟩, ⟨3000⟩, ⟨1300⟩]
  , sample_size := 280 }

/-- Theorem: The calculated sample sizes for the university population are correct -/
theorem university_sample_sizes_correct :
  (university_pop.strata.map (sample_size_for_stratum university_pop)) = [65, 150, 65] := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_university_sample_sizes_correct_l4086_408656


namespace NUMINAMATH_CALUDE_object_height_properties_l4086_408686

-- Define the height function
def h (t : ℝ) : ℝ := -14 * (t - 3)^2 + 140

-- Theorem statement
theorem object_height_properties :
  (∀ t : ℝ, h t ≤ h 3) ∧ (h 5 = 84) := by
  sorry

end NUMINAMATH_CALUDE_object_height_properties_l4086_408686


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l4086_408635

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x^2 - |x+a| -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - |x + a|

/-- If f(x) = x^2 - |x+a| is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) :
  IsEven (f a) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l4086_408635


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l4086_408650

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧ 
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l4086_408650


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l4086_408697

-- Define the triangle
def triangle (x : ℝ) : Fin 3 → ℝ
| 0 => 8
| 1 => 2*x + 5
| 2 => 3*x - 1
| _ => 0  -- This case is never reached due to Fin 3

-- State the theorem
theorem longest_side_of_triangle :
  ∃ x : ℝ, 
    (triangle x 0 + triangle x 1 + triangle x 2 = 45) ∧ 
    (∀ i : Fin 3, triangle x i ≤ 18.8) ∧
    (∃ i : Fin 3, triangle x i = 18.8) :=
by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l4086_408697


namespace NUMINAMATH_CALUDE_lawn_mowing_time_l4086_408679

theorem lawn_mowing_time (mary_rate tom_rate : ℚ) (mary_time : ℚ) : 
  mary_rate = 1/3 →
  tom_rate = 1/6 →
  mary_time = 1 →
  (1 - mary_rate * mary_time) / tom_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_lawn_mowing_time_l4086_408679


namespace NUMINAMATH_CALUDE_expected_socks_taken_l4086_408696

/-- Represents a collection of socks -/
structure SockCollection where
  pairs : ℕ  -- number of pairs of socks
  nonIdentical : Bool  -- whether all pairs are non-identical

/-- Represents the process of selecting socks -/
def selectSocks (sc : SockCollection) : ℕ → ℕ
  | 0 => 0
  | n + 1 => n + 1  -- simplified representation of sock selection

/-- Expected number of socks taken until a pair is found -/
def expectedSocksTaken (sc : SockCollection) : ℝ :=
  2 * sc.pairs

/-- Theorem stating the expected number of socks taken is 2p -/
theorem expected_socks_taken (sc : SockCollection) (h1 : sc.nonIdentical = true) :
  expectedSocksTaken sc = 2 * sc.pairs := by
  sorry

#check expected_socks_taken

end NUMINAMATH_CALUDE_expected_socks_taken_l4086_408696


namespace NUMINAMATH_CALUDE_odd_digits_in_3_times_257_base4_l4086_408680

/-- Counts the number of odd digits in the base-4 representation of a natural number. -/
def countOddDigitsBase4 (n : ℕ) : ℕ := sorry

/-- Converts a natural number from base 10 to base 4. -/
def toBase4 (n : ℕ) : List ℕ := sorry

theorem odd_digits_in_3_times_257_base4 :
  countOddDigitsBase4 (3 * 257) = 1 := by sorry

end NUMINAMATH_CALUDE_odd_digits_in_3_times_257_base4_l4086_408680


namespace NUMINAMATH_CALUDE_negation_of_absolute_sine_bound_l4086_408630

theorem negation_of_absolute_sine_bound :
  (¬ ∀ x : ℝ, |Real.sin x| ≤ 1) ↔ (∃ x : ℝ, |Real.sin x| > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_sine_bound_l4086_408630


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l4086_408637

/-- Parabola intersecting a line --/
structure ParabolaIntersection where
  p : ℝ
  chord_length : ℝ
  h_p_pos : p > 0
  h_chord : chord_length = 3 * Real.sqrt 5

/-- The result of the intersection --/
def ParabolaIntersectionResult (pi : ParabolaIntersection) : Prop :=
  -- Part I: The equation of the parabola is y² = 4x
  (pi.p = 2) ∧
  -- Part II: The maximum distance from a point on the circumcircle of triangle ABF to line AB
  (∃ (max_distance : ℝ), max_distance = (9 * Real.sqrt 5) / 2)

/-- Main theorem --/
theorem parabola_intersection_theorem (pi : ParabolaIntersection) :
  ParabolaIntersectionResult pi :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l4086_408637


namespace NUMINAMATH_CALUDE_magic_square_solution_l4086_408619

/-- Represents a 3x3 magic square with some known entries -/
structure MagicSquare where
  x : ℤ
  sum : ℤ

/-- The magic square property: all rows, columns, and diagonals sum to the same value -/
def magic_square_property (m : MagicSquare) : Prop :=
  ∃ (d e f g h : ℤ),
    m.x + 21 + 50 = m.sum ∧
    m.x + 3 + f = m.sum ∧
    50 + e + h = m.sum ∧
    m.x + d + h = m.sum ∧
    3 + d + e = m.sum ∧
    f + g + h = m.sum

/-- The theorem stating that x must be 106 in the given magic square -/
theorem magic_square_solution (m : MagicSquare) 
  (h : magic_square_property m) : m.x = 106 := by
  sorry

#check magic_square_solution

end NUMINAMATH_CALUDE_magic_square_solution_l4086_408619


namespace NUMINAMATH_CALUDE_inequality_theorem_l4086_408639

theorem inequality_theorem (p : ℝ) : 
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + p^3 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p < (2 + 2 * Real.sqrt 13) / 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l4086_408639


namespace NUMINAMATH_CALUDE_point_symmetry_l4086_408651

/-- Given three points A, B, and P in a 2D Cartesian coordinate system,
    prove that if B has coordinates (1, 2), P is symmetric to A with respect to the x-axis,
    and P is symmetric to B with respect to the y-axis, then A has coordinates (-1, -2). -/
theorem point_symmetry (A B P : ℝ × ℝ) : 
  B = (1, 2) → 
  P.1 = A.1 ∧ P.2 = -A.2 →  -- P is symmetric to A with respect to x-axis
  P.1 = -B.1 ∧ P.2 = B.2 →  -- P is symmetric to B with respect to y-axis
  A = (-1, -2) := by
sorry

end NUMINAMATH_CALUDE_point_symmetry_l4086_408651


namespace NUMINAMATH_CALUDE_regular_hours_is_40_l4086_408620

/-- Represents the pay structure and work hours for Bob --/
structure PayStructure where
  regularRate : ℝ  -- Regular hourly rate
  overtimeRate : ℝ  -- Overtime hourly rate
  hoursWeek1 : ℝ  -- Hours worked in week 1
  hoursWeek2 : ℝ  -- Hours worked in week 2
  totalEarnings : ℝ  -- Total earnings for both weeks

/-- Calculates the number of regular hours in a week --/
def calculateRegularHours (p : PayStructure) : ℝ :=
  let regularHours := 40  -- The value we want to prove
  regularHours

/-- Theorem stating that the number of regular hours is 40 --/
theorem regular_hours_is_40 (p : PayStructure) 
    (h1 : p.regularRate = 5)
    (h2 : p.overtimeRate = 6)
    (h3 : p.hoursWeek1 = 44)
    (h4 : p.hoursWeek2 = 48)
    (h5 : p.totalEarnings = 472) :
    calculateRegularHours p = 40 := by
  sorry

#eval calculateRegularHours { regularRate := 5, overtimeRate := 6, hoursWeek1 := 44, hoursWeek2 := 48, totalEarnings := 472 }

end NUMINAMATH_CALUDE_regular_hours_is_40_l4086_408620


namespace NUMINAMATH_CALUDE_linear_function_quadrant_slope_l4086_408603

/-- A linear function passing through the first, second, and third quadrants has a slope between 0 and 2 -/
theorem linear_function_quadrant_slope (k : ℝ) :
  (∀ x y : ℝ, y = k * x + (2 - k)) →
  (∃ x₁ y₁ : ℝ, x₁ > 0 ∧ y₁ > 0 ∧ y₁ = k * x₁ + (2 - k)) →
  (∃ x₂ y₂ : ℝ, x₂ < 0 ∧ y₂ > 0 ∧ y₂ = k * x₂ + (2 - k)) →
  (∃ x₃ y₃ : ℝ, x₃ < 0 ∧ y₃ < 0 ∧ y₃ = k * x₃ + (2 - k)) →
  0 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_quadrant_slope_l4086_408603


namespace NUMINAMATH_CALUDE_unattainable_value_l4086_408621

theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  ¬∃ x, (2 - x) / (3 * x + 4) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_unattainable_value_l4086_408621


namespace NUMINAMATH_CALUDE_sum_of_coordinates_B_l4086_408629

/-- Given point A at (0, 0), point B on the line y = 6, and the slope of segment AB is 3/4,
    the sum of the x- and y-coordinates of point B is 14. -/
theorem sum_of_coordinates_B (B : ℝ × ℝ) : 
  B.2 = 6 ∧ (B.2 - 0) / (B.1 - 0) = 3/4 → B.1 + B.2 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_B_l4086_408629


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l4086_408663

/-- An isosceles triangle with two sides of length 12 and one side of length 17 has a perimeter of 41 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → Prop :=
  fun (equal_side : ℝ) (third_side : ℝ) =>
    equal_side = 12 ∧ third_side = 17 →
    2 * equal_side + third_side = 41

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 12 17 :=
by
  sorry

#check isosceles_triangle_perimeter_proof

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l4086_408663


namespace NUMINAMATH_CALUDE_parabola_f_value_l4086_408606

/-- A parabola with equation x = dy² + ey + f, vertex at (5, 3), and passing through (2, 6) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 5 = d * 3^2 + e * 3 + f
  point_condition : 2 = d * 6^2 + e * 6 + f

/-- The value of f for the given parabola is 2 -/
theorem parabola_f_value (p : Parabola) : p.f = 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_f_value_l4086_408606


namespace NUMINAMATH_CALUDE_solve_pet_sitting_problem_l4086_408655

def pet_sitting_problem (hourly_rate : ℝ) (hours_this_week : ℝ) (total_earnings : ℝ) : Prop :=
  let earnings_this_week := hourly_rate * hours_this_week
  let earnings_last_week := total_earnings - earnings_this_week
  let hours_last_week := earnings_last_week / hourly_rate
  hourly_rate = 5 ∧ hours_this_week = 30 ∧ total_earnings = 250 → hours_last_week = 20

theorem solve_pet_sitting_problem :
  pet_sitting_problem 5 30 250 := by
  sorry

end NUMINAMATH_CALUDE_solve_pet_sitting_problem_l4086_408655


namespace NUMINAMATH_CALUDE_cylindrical_cans_radius_l4086_408648

theorem cylindrical_cans_radius (h : ℝ) (h_pos : h > 0) :
  let r₁ : ℝ := 15 -- radius of the second can
  let h₁ : ℝ := h -- height of the second can
  let h₂ : ℝ := (4 * h^2) / 3 -- height of the first can
  let v₁ : ℝ := π * r₁^2 * h₁ -- volume of the second can
  let r₂ : ℝ := (15 * Real.sqrt 3) / 2 -- radius of the first can
  v₁ = π * r₂^2 * h₂ -- volumes are equal
  := by sorry

end NUMINAMATH_CALUDE_cylindrical_cans_radius_l4086_408648


namespace NUMINAMATH_CALUDE_frank_lamp_purchase_l4086_408687

theorem frank_lamp_purchase (cheapest_lamp : ℕ) (frank_money : ℕ) :
  cheapest_lamp = 20 →
  frank_money = 90 →
  frank_money - (3 * cheapest_lamp) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_lamp_purchase_l4086_408687


namespace NUMINAMATH_CALUDE_man_age_difference_l4086_408636

/-- Proves that a man is 22 years older than his son given certain conditions -/
theorem man_age_difference (man_age son_age : ℕ) : 
  son_age = 20 →
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_man_age_difference_l4086_408636


namespace NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l4086_408682

theorem min_sum_of_squares_on_line (m n : ℝ) (h : m + n = 1) : 
  m^2 + n^2 ≥ 4 ∧ ∃ (m₀ n₀ : ℝ), m₀ + n₀ = 1 ∧ m₀^2 + n₀^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_on_line_l4086_408682


namespace NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_bounds_l4086_408600

/-- Represents a point on the side of a rectangle -/
structure SidePoint where
  side : Fin 4  -- 0: top, 1: right, 2: bottom, 3: left
  position : ℝ
  h_position : 0 ≤ position ∧ position ≤ match side with
    | 0 | 2 => 3  -- top and bottom sides
    | 1 | 3 => 4  -- right and left sides

/-- The quadrilateral formed by four points on the sides of a 3x4 rectangle -/
def Quadrilateral (p₁ p₂ p₃ p₄ : SidePoint) : Prop :=
  p₁.side ≠ p₂.side ∧ p₂.side ≠ p₃.side ∧ p₃.side ≠ p₄.side ∧ p₄.side ≠ p₁.side

/-- The side length of the quadrilateral between two points -/
def sideLength (p₁ p₂ : SidePoint) : ℝ :=
  sorry  -- Definition of side length calculation

/-- The sum of squares of side lengths of the quadrilateral -/
def sumOfSquares (p₁ p₂ p₃ p₄ : SidePoint) : ℝ :=
  (sideLength p₁ p₂)^2 + (sideLength p₂ p₃)^2 + (sideLength p₃ p₄)^2 + (sideLength p₄ p₁)^2

/-- The main theorem -/
theorem quadrilateral_sum_of_squares_bounds
  (p₁ p₂ p₃ p₄ : SidePoint)
  (h : Quadrilateral p₁ p₂ p₃ p₄) :
  25 ≤ sumOfSquares p₁ p₂ p₃ p₄ ∧ sumOfSquares p₁ p₂ p₃ p₄ ≤ 50 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_bounds_l4086_408600


namespace NUMINAMATH_CALUDE_room_width_calculation_l4086_408654

/-- Given a rectangular room with specified length, total paving cost, and paving rate per square meter, 
    prove that the width of the room is as calculated. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ) : 
  length = 7 →
  total_cost = 29925 →
  rate_per_sqm = 900 →
  width = total_cost / rate_per_sqm / length →
  width = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l4086_408654


namespace NUMINAMATH_CALUDE_weight_replacement_l4086_408666

theorem weight_replacement (n : ℕ) (avg_increase : ℝ) (new_weight : ℝ) :
  n = 10 →
  avg_increase = 6.3 →
  new_weight = 128 →
  ∃ (old_weight : ℝ),
    old_weight = new_weight - n * avg_increase ∧
    old_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l4086_408666


namespace NUMINAMATH_CALUDE_tax_rate_on_remaining_income_l4086_408698

def total_earnings : ℝ := 100000
def deductions : ℝ := 30000
def first_bracket_limit : ℝ := 20000
def first_bracket_rate : ℝ := 0.1
def total_tax : ℝ := 12000

def taxable_income : ℝ := total_earnings - deductions

def tax_on_first_bracket : ℝ := first_bracket_limit * first_bracket_rate

def remaining_taxable_income : ℝ := taxable_income - first_bracket_limit

theorem tax_rate_on_remaining_income : 
  (total_tax - tax_on_first_bracket) / remaining_taxable_income = 0.2 := by sorry

end NUMINAMATH_CALUDE_tax_rate_on_remaining_income_l4086_408698


namespace NUMINAMATH_CALUDE_fraction_problem_l4086_408661

theorem fraction_problem (N : ℝ) (F : ℝ) (h : F * (1/3 * N) = 30) :
  ∃ G : ℝ, G * N = 75 ∧ G = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l4086_408661


namespace NUMINAMATH_CALUDE_sum_of_fractions_l4086_408674

theorem sum_of_fractions : (1/2 : ℚ) + 2/4 + 4/8 + 8/16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l4086_408674


namespace NUMINAMATH_CALUDE_chicks_increase_l4086_408671

theorem chicks_increase (first_day : ℕ) (second_day : ℕ) : first_day = 23 → second_day = 12 → first_day + second_day = 35 := by
  sorry

end NUMINAMATH_CALUDE_chicks_increase_l4086_408671


namespace NUMINAMATH_CALUDE_T_equiv_horizontal_lines_l4086_408688

/-- The set of points R forming a right triangle PQR with area 4, where P(2,0) and Q(-2,0) -/
def T : Set (ℝ × ℝ) :=
  {R | ∃ (x y : ℝ), R = (x, y) ∧ 
       ((x - 2)^2 + y^2) * ((x + 2)^2 + y^2) = 16 * (x^2 + y^2) ∧
       (abs ((x - 2) * y - (x + 2) * y)) = 8}

/-- The set of points with y-coordinate equal to 2 or -2 -/
def horizontal_lines : Set (ℝ × ℝ) :=
  {R | ∃ (x y : ℝ), R = (x, y) ∧ (y = 2 ∨ y = -2)}

theorem T_equiv_horizontal_lines : T = horizontal_lines := by
  sorry

end NUMINAMATH_CALUDE_T_equiv_horizontal_lines_l4086_408688


namespace NUMINAMATH_CALUDE_two_counterexamples_l4086_408646

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has any digit equal to 0 -/
def has_zero_digit (n : ℕ) : Bool := sorry

/-- The main theorem -/
theorem two_counterexamples : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, sum_of_digits n = 4 ∧ ¬has_zero_digit n ∧ ¬Nat.Prime n) ∧ 
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_counterexamples_l4086_408646
