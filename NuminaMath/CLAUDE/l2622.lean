import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_intercept_distance_l2622_262223

/-- Given a quadratic function f(x) = x² + ax + b, where the line from (0, b) to one x-intercept
    is perpendicular to y = x, prove that the distance from (0, 0) to the other x-intercept is 1. -/
theorem quadratic_intercept_distance (a b : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ x^2 + a*x + b
  let x₁ := -b  -- One x-intercept
  let x₂ := 1   -- The other x-intercept (to be proven)
  (∀ x, f x = 0 → x = x₁ ∨ x = x₂) →  -- x₁ and x₂ are the only roots
  (x₁ + x₂ = -a ∧ x₁ * x₂ = b) →      -- Vieta's formulas
  (b ≠ 0) →                           -- Ensuring non-zero y-intercept
  (∀ x y, y = -x + b → f x = y) →     -- Line from (0, b) to (x₁, 0) has equation y = -x + b
  x₂ = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intercept_distance_l2622_262223


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2622_262221

theorem sufficient_not_necessary : ∀ x : ℝ, 
  (∀ x, x > 5 → x > 3) ∧ 
  (∃ x, x > 3 ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2622_262221


namespace NUMINAMATH_CALUDE_stating_race_outcomes_count_l2622_262204

/-- Represents the number of participants in the race -/
def total_participants : ℕ := 6

/-- Represents the number of top positions we're considering -/
def top_positions : ℕ := 3

/-- Represents the number of participants eligible for top positions -/
def eligible_participants : ℕ := total_participants - 1

/-- 
Calculates the number of different outcomes for top positions in a race
given the number of eligible participants and the number of top positions,
assuming no ties.
-/
def race_outcomes (eligible : ℕ) (positions : ℕ) : ℕ :=
  (eligible - positions + 1).factorial / (eligible - positions).factorial

/-- 
Theorem stating that the number of different 1st-2nd-3rd place outcomes
in a race with 6 participants, where one participant cannot finish 
in the top three and there are no ties, is equal to 60.
-/
theorem race_outcomes_count : 
  race_outcomes eligible_participants top_positions = 60 := by
  sorry

end NUMINAMATH_CALUDE_stating_race_outcomes_count_l2622_262204


namespace NUMINAMATH_CALUDE_symmetric_point_l2622_262229

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Function to check if a point is the midpoint of two other points -/
def isMidpoint (m : Point2D) (p1 : Point2D) (p2 : Point2D) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Function to check if two points are symmetric with respect to the origin -/
def isSymmetricToOrigin (p1 : Point2D) (p2 : Point2D) : Prop :=
  isMidpoint origin p1 p2

/-- Theorem: The point (2,-3) is symmetric to (-2,3) with respect to the origin -/
theorem symmetric_point : 
  isSymmetricToOrigin ⟨-2, 3⟩ ⟨2, -3⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_l2622_262229


namespace NUMINAMATH_CALUDE_new_numbers_average_l2622_262291

theorem new_numbers_average (initial_count : ℕ) (initial_mean : ℝ) 
  (new_count : ℕ) (new_mean : ℝ) : 
  initial_count = 12 →
  initial_mean = 45 →
  new_count = 15 →
  new_mean = 60 →
  (new_count * new_mean - initial_count * initial_mean) / (new_count - initial_count) = 120 :=
by sorry

end NUMINAMATH_CALUDE_new_numbers_average_l2622_262291


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2622_262259

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - 3*m = 0) ∧ (m^2 - 5*m + 6 ≠ 0) → m = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2622_262259


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2622_262286

/-- Two points are symmetric about the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite in sign but equal in magnitude. -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given two points M(a,3) and N(4,b) symmetric about the x-axis,
    prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_about_x_axis (a, 3) (4, b)) : a + b = 1 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_points_sum_l2622_262286


namespace NUMINAMATH_CALUDE_x_minus_y_times_x_plus_y_equals_95_l2622_262228

theorem x_minus_y_times_x_plus_y_equals_95 (x y : ℤ) (h1 : x = 12) (h2 : y = 7) : 
  (x - y) * (x + y) = 95 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_times_x_plus_y_equals_95_l2622_262228


namespace NUMINAMATH_CALUDE_chess_team_photo_arrangements_l2622_262230

/-- The number of ways to arrange a chess team in a line -/
def chessTeamArrangements (numBoys numGirls : ℕ) : ℕ :=
  Nat.factorial numGirls * Nat.factorial numBoys

/-- Theorem: There are 12 ways to arrange 2 boys and 3 girls in a line
    with girls in the middle and boys on the ends -/
theorem chess_team_photo_arrangements :
  chessTeamArrangements 2 3 = 12 := by
  sorry

#eval chessTeamArrangements 2 3

end NUMINAMATH_CALUDE_chess_team_photo_arrangements_l2622_262230


namespace NUMINAMATH_CALUDE_knight_placement_exists_l2622_262250

/-- A position on the modified 6x6 board -/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)
  (valid : ¬((x < 2 ∧ y < 2) ∨ (x > 3 ∧ y < 2) ∨ (x < 2 ∧ y > 3) ∨ (x > 3 ∧ y > 3)))

/-- A knight's move -/
def knightMove (p q : Position) : Prop :=
  (abs (p.x - q.x) == 2 ∧ abs (p.y - q.y) == 1) ∨
  (abs (p.x - q.x) == 1 ∧ abs (p.y - q.y) == 2)

/-- A valid knight placement -/
structure KnightPlacement :=
  (positions : Fin 10 → Position × Position)
  (distinct : ∀ i j, i ≠ j → positions i ≠ positions j)
  (canAttack : ∀ i, knightMove (positions i).1 (positions i).2)

/-- The main theorem -/
theorem knight_placement_exists : ∃ (k : KnightPlacement), True :=
sorry

end NUMINAMATH_CALUDE_knight_placement_exists_l2622_262250


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l2622_262266

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (7 * x - 3 * y = 20) ∧ x = 11 ∧ y = 19 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l2622_262266


namespace NUMINAMATH_CALUDE_square_land_area_l2622_262225

/-- Given a square land with perimeter p and area A, prove that A = 81 --/
theorem square_land_area (p A : ℝ) : p = 36 ∧ 5 * A = 10 * p + 45 → A = 81 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l2622_262225


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_implies_a_eq_two_l2622_262236

/-- If (1 + ai) / (2 - i) is a pure imaginary number, then a = 2 -/
theorem pure_imaginary_fraction_implies_a_eq_two (a : ℝ) :
  (∃ b : ℝ, (1 + a * Complex.I) / (2 - Complex.I) = b * Complex.I) →
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_implies_a_eq_two_l2622_262236


namespace NUMINAMATH_CALUDE_bus_problem_l2622_262269

theorem bus_problem (initial_students : ℕ) (remaining_fraction : ℚ) : 
  initial_students = 64 →
  remaining_fraction = 2/3 →
  (initial_students : ℚ) * remaining_fraction^3 = 512/27 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l2622_262269


namespace NUMINAMATH_CALUDE_coin_division_problem_l2622_262297

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 8 = 6) →
  (n % 7 = 5) →
  (n % 9 = 0) := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l2622_262297


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2622_262224

def M : Set ℝ := {x | |x| < 1}
def N : Set ℝ := {x | x^2 - x < 0}

theorem set_intersection_equality : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2622_262224


namespace NUMINAMATH_CALUDE_editing_posting_time_is_zero_l2622_262227

/-- Represents the time in hours for various activities in video production -/
structure VideoProductionTime where
  setup : ℝ
  painting : ℝ
  cleanup : ℝ
  total : ℝ

/-- The time spent on editing and posting each video -/
def editingPostingTime (t : VideoProductionTime) : ℝ :=
  t.total - (t.setup + t.painting + t.cleanup)

/-- Theorem stating that the editing and posting time is 0 hours -/
theorem editing_posting_time_is_zero (t : VideoProductionTime)
  (h_setup : t.setup = 1)
  (h_painting : t.painting = 1)
  (h_cleanup : t.cleanup = 1)
  (h_total : t.total = 3) :
  editingPostingTime t = 0 := by
  sorry

end NUMINAMATH_CALUDE_editing_posting_time_is_zero_l2622_262227


namespace NUMINAMATH_CALUDE_laptop_price_theorem_l2622_262205

/-- The sticker price of the laptop. -/
def sticker_price : ℝ := 250

/-- The price at store A after discount and rebate. -/
def price_A (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at store B after discount and rebate. -/
def price_B (x : ℝ) : ℝ := 0.7 * x - 50

/-- Theorem stating that the sticker price satisfies the given conditions. -/
theorem laptop_price_theorem : 
  price_A sticker_price = price_B sticker_price - 25 := by
  sorry

#check laptop_price_theorem

end NUMINAMATH_CALUDE_laptop_price_theorem_l2622_262205


namespace NUMINAMATH_CALUDE_bus_seats_solution_l2622_262243

/-- Represents the seating arrangement in a bus -/
structure BusSeats where
  left : ℕ  -- Number of seats on the left side
  right : ℕ  -- Number of seats on the right side
  back : ℕ  -- Capacity of the back seat
  capacity_per_seat : ℕ  -- Number of people each regular seat can hold

/-- The total capacity of the bus -/
def total_capacity (bs : BusSeats) : ℕ :=
  bs.capacity_per_seat * (bs.left + bs.right) + bs.back

theorem bus_seats_solution :
  ∃ (bs : BusSeats),
    bs.right = bs.left - 3 ∧
    bs.capacity_per_seat = 3 ∧
    bs.back = 10 ∧
    total_capacity bs = 91 ∧
    bs.left = 15 := by
  sorry

end NUMINAMATH_CALUDE_bus_seats_solution_l2622_262243


namespace NUMINAMATH_CALUDE_keith_bought_22_cards_l2622_262246

/-- The number of baseball cards Keith bought -/
def cards_bought (initial_cards remaining_cards : ℕ) : ℕ :=
  initial_cards - remaining_cards

/-- Theorem stating that Keith bought 22 baseball cards -/
theorem keith_bought_22_cards : cards_bought 40 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_keith_bought_22_cards_l2622_262246


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2622_262212

/-- Given integers a, b, c satisfying the equation (x - a)(x - 5) + 1 = (x + b)(x + c)
    and either (b + 5)(c + 5) = 1 or (b + 5)(c + 5) = 4,
    prove that the possible values of a are 2, 3, 4, and 7. -/
theorem possible_values_of_a (a b c : ℤ) 
  (h1 : ∀ x, (x - a) * (x - 5) + 1 = (x + b) * (x + c))
  (h2 : (b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) :
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 := by
  sorry


end NUMINAMATH_CALUDE_possible_values_of_a_l2622_262212


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2622_262220

-- Define the vectors
def a : ℝ × ℝ := (4, -3)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2622_262220


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l2622_262293

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

def N : Set ℝ := {x | Real.log 2 ^ (1 - x) < 1}

theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.univ \ N) = Set.Icc 1 2 \ {2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l2622_262293


namespace NUMINAMATH_CALUDE_playhouse_siding_cost_l2622_262231

/-- Calculates the cost of siding for a playhouse with given dimensions --/
theorem playhouse_siding_cost
  (wall_width : ℝ)
  (wall_height : ℝ)
  (roof_width : ℝ)
  (roof_height : ℝ)
  (siding_width : ℝ)
  (siding_height : ℝ)
  (siding_cost : ℝ)
  (h_wall_width : wall_width = 10)
  (h_wall_height : wall_height = 7)
  (h_roof_width : roof_width = 10)
  (h_roof_height : roof_height = 6)
  (h_siding_width : siding_width = 10)
  (h_siding_height : siding_height = 15)
  (h_siding_cost : siding_cost = 35) :
  ⌈(wall_width * wall_height + 2 * roof_width * roof_height) / (siding_width * siding_height)⌉ * siding_cost = 70 :=
by sorry

end NUMINAMATH_CALUDE_playhouse_siding_cost_l2622_262231


namespace NUMINAMATH_CALUDE_library_books_remaining_l2622_262234

/-- The number of books remaining in a library after a series of events --/
def remaining_books (initial : ℕ) (taken_out : ℕ) (returned : ℕ) (withdrawn : ℕ) : ℕ :=
  initial - taken_out + returned - withdrawn

/-- Theorem stating that given the specific events, 150 books remain in the library --/
theorem library_books_remaining : remaining_books 250 120 35 15 = 150 := by
  sorry

end NUMINAMATH_CALUDE_library_books_remaining_l2622_262234


namespace NUMINAMATH_CALUDE_hexagon_diagonals_intersect_at_nine_point_center_l2622_262242

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a hexagon -/
structure Hexagon where
  vertices : Finset Point
  is_convex : Bool

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- The perpendicular bisector of a line segment -/
def perp_bisector (p1 p2 : Point) : Set Point := sorry

/-- The intersection points of a line with a triangle's sides -/
def intersections_with_triangle (line : Set Point) (t : Triangle) : Finset Point := sorry

/-- The hexagon formed by the intersections of perpendicular bisectors with triangle sides -/
def form_hexagon (t : Triangle) (h : Point) : Hexagon := sorry

/-- The main diagonals of a hexagon -/
def main_diagonals (h : Hexagon) : Finset (Set Point) := sorry

/-- The intersection point of the main diagonals of a hexagon -/
def diagonals_intersection (h : Hexagon) : Option Point := sorry

/-- The center of the nine-point circle of a triangle -/
def nine_point_center (t : Triangle) : Point := sorry

/-- The theorem to be proved -/
theorem hexagon_diagonals_intersect_at_nine_point_center 
  (t : Triangle) (is_acute : Bool) : 
  let h := orthocenter t
  let hexagon := form_hexagon t h
  diagonals_intersection hexagon = some (nine_point_center t) := by sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_intersect_at_nine_point_center_l2622_262242


namespace NUMINAMATH_CALUDE_board_pair_positive_l2622_262258

inductive BoardPair : ℚ × ℚ → Prop where
  | initial : BoardPair (1, 1)
  | trans1a (x y : ℚ) : BoardPair (x, y - 1) → BoardPair (x + y, y + 1)
  | trans1b (x y : ℚ) : BoardPair (x + y, y + 1) → BoardPair (x, y - 1)
  | trans2a (x y : ℚ) : BoardPair (x, x * y) → BoardPair (1 / x, y)
  | trans2b (x y : ℚ) : BoardPair (1 / x, y) → BoardPair (x, x * y)

theorem board_pair_positive (a b : ℚ) : BoardPair (a, b) → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_board_pair_positive_l2622_262258


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_k_range_l2622_262203

theorem ellipse_eccentricity_k_range (k : ℝ) (e : ℝ) :
  (∃ x y : ℝ, x^2 / k + y^2 / 4 = 1) →
  (1/2 < e ∧ e < 1) →
  (0 < k ∧ k < 3) ∨ (16/3 < k) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_k_range_l2622_262203


namespace NUMINAMATH_CALUDE_min_a_for_ln_inequality_l2622_262208

/-- The minimum value of a for which ln x ≤ ax + 1 holds for all x > 0 is 1/e^2 -/
theorem min_a_for_ln_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) ∧ 
  (∀ (a : ℝ), (∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) → a ≥ 1 / Real.exp 2) ∧
  (∃ (a : ℝ), a = 1 / Real.exp 2 ∧ ∀ (x : ℝ), x > 0 → Real.log x ≤ a * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_ln_inequality_l2622_262208


namespace NUMINAMATH_CALUDE_family_ages_solution_l2622_262281

/-- Represents the ages of Priya and her parents -/
structure FamilyAges where
  priya : ℕ
  father : ℕ
  mother : ℕ

/-- Conditions for the family ages problem -/
def FamilyAgesProblem (ages : FamilyAges) : Prop :=
  ages.father - ages.priya = 31 ∧
  ages.father + 8 + ages.priya + 8 = 69 ∧
  ages.father - ages.mother = 4 ∧
  ages.priya + 5 + ages.mother + 5 = 65

/-- Theorem stating the solution to the family ages problem -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), FamilyAgesProblem ages ∧ 
    ages.priya = 11 ∧ ages.father = 42 ∧ ages.mother = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l2622_262281


namespace NUMINAMATH_CALUDE_some_number_added_l2622_262260

theorem some_number_added (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁ + a)^2 / (3 * x₁ + 65) = 2 ∧ 
                (x₂ + a)^2 / (3 * x₂ + 65) = 2 ∧ 
                |x₁ - x₂| = 22) → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_some_number_added_l2622_262260


namespace NUMINAMATH_CALUDE_prob_reroll_two_dice_l2622_262277

/-- The number of possible outcomes when rolling three fair six-sided dice -/
def total_outcomes : ℕ := 6^3

/-- The number of ways to get a sum of 8 when rolling three fair six-sided dice -/
def sum_eight_outcomes : ℕ := 20

/-- The probability that the sum of three fair six-sided dice is not equal to 8 -/
def prob_not_eight : ℚ := (total_outcomes - sum_eight_outcomes) / total_outcomes

theorem prob_reroll_two_dice : prob_not_eight = 49 / 54 := by
  sorry

end NUMINAMATH_CALUDE_prob_reroll_two_dice_l2622_262277


namespace NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l2622_262257

/-- The standard equation of a circle with diameter endpoints A(-1, 2) and B(5, -6) -/
theorem circle_equation_with_given_diameter :
  ∃ (f : ℝ × ℝ → ℝ),
    (∀ x y : ℝ, f (x, y) = (x - 2)^2 + (y + 2)^2) ∧
    (∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | f p = 25} ↔ 
      ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
        x = -1 + 6*t ∧ 
        y = 2 - 8*t) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l2622_262257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2622_262290

theorem arithmetic_sequence_problem (a b c : ℝ) : 
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →  -- arithmetic sequence condition
  a + b + c = 9 →                       -- sum condition
  a * b = 6 * c →                       -- product condition
  a = 4 ∧ b = 3 ∧ c = 2 := by           -- conclusion
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2622_262290


namespace NUMINAMATH_CALUDE_discount_difference_l2622_262215

def original_bill : ℝ := 12000

def single_discount (bill : ℝ) : ℝ := bill * 0.7

def successive_discounts (bill : ℝ) : ℝ := bill * 0.75 * 0.95

theorem discount_difference :
  successive_discounts original_bill - single_discount original_bill = 150 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l2622_262215


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l2622_262232

theorem minimum_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  (1/x + 4/(1 - 2*x)) ≥ 6 + 4*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l2622_262232


namespace NUMINAMATH_CALUDE_angle_complement_l2622_262252

/-- Given an angle α of 63°21', its complement is 26°39' -/
theorem angle_complement (α : Real) : α = 63 + 21 / 60 → 90 - α = 26 + 39 / 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_l2622_262252


namespace NUMINAMATH_CALUDE_equation_solution_l2622_262254

/-- Given an equation y = a + b / x^2, where a and b are constants,
    if y = 2 when x = -2 and y = 4 when x = -4, then a + b = -6 -/
theorem equation_solution (a b : ℝ) : 
  (2 = a + b / (-2)^2) → 
  (4 = a + b / (-4)^2) → 
  a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2622_262254


namespace NUMINAMATH_CALUDE_complete_square_sum_l2622_262285

theorem complete_square_sum (x : ℝ) : 
  (∃ (a b c : ℤ), a > 0 ∧ 
   64 * x^2 + 96 * x - 128 = 0 ↔ (a * x + b)^2 = c) →
  (∃ (a b c : ℤ), a > 0 ∧ 
   64 * x^2 + 96 * x - 128 = 0 ↔ (a * x + b)^2 = c ∧
   a + b + c = 178) := by
sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2622_262285


namespace NUMINAMATH_CALUDE_inequality_proof_l2622_262294

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2622_262294


namespace NUMINAMATH_CALUDE_min_value_expression_l2622_262226

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  6 * Real.sqrt (a * b) + 3 / a + 3 / b ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2622_262226


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2622_262263

/-- Given an arithmetic sequence {a_n} where a₁ + 3a₈ + a₁₅ = 120, prove that a₂ + a₁₄ = 48. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 1 + 3 * a 8 + a 15 = 120 →                      -- given condition
  a 2 + a 14 = 48 := by                             -- conclusion to prove
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2622_262263


namespace NUMINAMATH_CALUDE_triangle_area_is_64_l2622_262238

/-- The area of the triangle bounded by y = x, y = -x, and y = 8 -/
def triangleArea : ℝ := 64

/-- The first bounding line of the triangle -/
def line1 (x : ℝ) : ℝ := x

/-- The second bounding line of the triangle -/
def line2 (x : ℝ) : ℝ := -x

/-- The third bounding line of the triangle -/
def line3 : ℝ := 8

theorem triangle_area_is_64 :
  triangleArea = (1/2) * (line3 - line1 0) * (line3 - line2 0) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_64_l2622_262238


namespace NUMINAMATH_CALUDE_equation_solution_l2622_262214

theorem equation_solution :
  let f (x : ℂ) := -x^2 * (x + 2) - (2 * x + 4)
  ∀ x : ℂ, x ≠ -2 → (f x = 0 ↔ x = -2 ∨ x = 2*I ∨ x = -2*I) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2622_262214


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l2622_262274

theorem roof_dimension_difference :
  ∀ (width length : ℝ),
  width > 0 →
  length = 4 * width →
  width * length = 768 →
  length - width = 24 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l2622_262274


namespace NUMINAMATH_CALUDE_pencil_cost_l2622_262216

theorem pencil_cost (total_students : Nat) (total_cost : Nat) 
  (h1 : total_students = 30)
  (h2 : total_cost = 1771)
  (h3 : ∃ (s n c : Nat), 
    s > total_students / 2 ∧ 
    n > 1 ∧ 
    c > n ∧ 
    s * n * c = total_cost) :
  ∃ (s n : Nat), s * n * 11 = total_cost :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l2622_262216


namespace NUMINAMATH_CALUDE_two_parts_problem_l2622_262249

theorem two_parts_problem (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_parts_problem_l2622_262249


namespace NUMINAMATH_CALUDE_same_last_digit_l2622_262264

theorem same_last_digit (a b : ℕ) : 
  (2 * a + b) % 10 = (2 * b + a) % 10 → a % 10 = b % 10 := by
  sorry

end NUMINAMATH_CALUDE_same_last_digit_l2622_262264


namespace NUMINAMATH_CALUDE_impossible_inequality_l2622_262222

theorem impossible_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_log : Real.log x / Real.log 2 = Real.log y / Real.log 3 ∧ 
           Real.log y / Real.log 3 = Real.log z / Real.log 5 ∧
           Real.log z / Real.log 5 > 0) :
  ¬(y / 3 < z / 5 ∧ z / 5 < x / 2) := by
sorry

end NUMINAMATH_CALUDE_impossible_inequality_l2622_262222


namespace NUMINAMATH_CALUDE_ray_walks_dog_three_times_daily_l2622_262273

/-- The number of times Ray walks his dog each day -/
def walks_per_day (route_length total_distance : ℕ) : ℕ :=
  total_distance / route_length

theorem ray_walks_dog_three_times_daily :
  let route_length : ℕ := 4 + 7 + 11
  let total_distance : ℕ := 66
  walks_per_day route_length total_distance = 3 := by
  sorry

end NUMINAMATH_CALUDE_ray_walks_dog_three_times_daily_l2622_262273


namespace NUMINAMATH_CALUDE_sum_of_sqrt_ratios_geq_two_l2622_262262

theorem sum_of_sqrt_ratios_geq_two (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x / (y + z)) + Real.sqrt (y / (z + x)) + Real.sqrt (z / (x + y)) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_ratios_geq_two_l2622_262262


namespace NUMINAMATH_CALUDE_units_digit_of_product_l2622_262235

theorem units_digit_of_product (n m : ℕ) : (5^7 * 6^4) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l2622_262235


namespace NUMINAMATH_CALUDE_circle_triangle_areas_l2622_262201

theorem circle_triangle_areas (a b c : ℝ) (A B C : ℝ) : 
  a = 15 → b = 20 → c = 25 →
  a^2 + b^2 = c^2 →
  A > 0 → B > 0 → C > 0 →
  C > A ∧ C > B →
  A + B + (1/2 * a * b) = C := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_areas_l2622_262201


namespace NUMINAMATH_CALUDE_cube_inequality_l2622_262270

theorem cube_inequality (n : ℕ+) : (n + 1)^3 ≠ n^3 + (n - 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2622_262270


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2622_262202

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2622_262202


namespace NUMINAMATH_CALUDE_shipping_cost_per_unit_l2622_262233

/-- Proves that the shipping cost per unit is $1.67 given the manufacturing conditions --/
theorem shipping_cost_per_unit 
  (production_cost : ℝ) 
  (fixed_cost : ℝ) 
  (units_sold : ℝ) 
  (selling_price : ℝ) 
  (h1 : production_cost = 80)
  (h2 : fixed_cost = 16500)
  (h3 : units_sold = 150)
  (h4 : selling_price = 191.67)
  : ∃ (shipping_cost : ℝ), 
    shipping_cost = 1.67 ∧ 
    units_sold * (production_cost + shipping_cost) + fixed_cost ≤ units_sold * selling_price ∧
    ∀ (s : ℝ), s < shipping_cost → 
      units_sold * (production_cost + s) + fixed_cost < units_sold * selling_price :=
by sorry

end NUMINAMATH_CALUDE_shipping_cost_per_unit_l2622_262233


namespace NUMINAMATH_CALUDE_segment_length_proof_l2622_262213

theorem segment_length_proof (A B O P M : Real) : 
  -- Conditions
  (0 ≤ A) ∧ (A < O) ∧ (O < M) ∧ (M < P) ∧ (P < B) ∧  -- Points lie on the line segment in order
  (O - A = 4/5 * (B - A)) ∧                           -- AO = 4/5 * AB
  (B - P = 2/3 * (B - A)) ∧                           -- BP = 2/3 * AB
  (M - A = 1/2 * (B - A)) ∧                           -- M is the midpoint of AB
  (M - O = 2) →                                       -- OM = 2
  (P - M = 10/9)                                      -- PM = 10/9

:= by sorry

end NUMINAMATH_CALUDE_segment_length_proof_l2622_262213


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_values_l2622_262255

theorem infinite_solutions_imply_values (a b : ℚ) : 
  (∀ x : ℚ, a * (2 * x + b) = 12 * x + 5) → 
  (a = 6 ∧ b = 5/6) := by
sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_values_l2622_262255


namespace NUMINAMATH_CALUDE_box_area_is_679_l2622_262272

/-- The surface area of the interior of a box formed by removing square corners from a rectangular sheet --/
def box_interior_area (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the interior of the box is 679 square units --/
theorem box_area_is_679 :
  box_interior_area 25 35 7 = 679 :=
by sorry

end NUMINAMATH_CALUDE_box_area_is_679_l2622_262272


namespace NUMINAMATH_CALUDE_A_intersect_B_l2622_262256

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2622_262256


namespace NUMINAMATH_CALUDE_polynomial_evaluation_and_subtraction_l2622_262284

theorem polynomial_evaluation_and_subtraction :
  let x : ℝ := 2
  20 - 2 * (3 * x^2 - 4 * x + 8) = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_and_subtraction_l2622_262284


namespace NUMINAMATH_CALUDE_dormitory_students_count_l2622_262289

theorem dormitory_students_count :
  ∃ (x y : ℕ),
    x > 0 ∧
    y > 0 ∧
    x * (x - 1) + x * y + y = 51 ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_students_count_l2622_262289


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l2622_262298

theorem distinct_prime_factors_count : 
  (Finset.card (Nat.factors (85 * 87 * 91 * 94)).toFinset) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l2622_262298


namespace NUMINAMATH_CALUDE_speed_change_problem_l2622_262296

theorem speed_change_problem :
  ∃! (x : ℝ), x > 0 ∧
  (1 - x / 100) * (1 + 0.5 * x / 100) = 1 - 0.6 * x / 100 ∧
  ∀ (V : ℝ), V > 0 →
    V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100) :=
by sorry

end NUMINAMATH_CALUDE_speed_change_problem_l2622_262296


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2622_262248

theorem smallest_number_with_remainders : ∃! N : ℕ,
  (N > 0) ∧
  (N % 13 = 2) ∧
  (N % 15 = 4) ∧
  (N % 17 = 6) ∧
  (N % 19 = 8) ∧
  (∀ M : ℕ, M > 0 ∧ M % 13 = 2 ∧ M % 15 = 4 ∧ M % 17 = 6 ∧ M % 19 = 8 → M ≥ N) ∧
  N = 1070747 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2622_262248


namespace NUMINAMATH_CALUDE_initial_speed_calculation_l2622_262280

/-- Proves that the initial speed of a person traveling a distance D in time T is 160/3 kmph -/
theorem initial_speed_calculation (D T : ℝ) (h1 : D > 0) (h2 : T > 0) : ∃ S : ℝ,
  (2 / 3 * D) / (1 / 3 * T) = S ∧
  (1 / 3 * D) / 40 = 2 / 3 * T ∧
  S = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_calculation_l2622_262280


namespace NUMINAMATH_CALUDE_relationship_abc_l2622_262279

theorem relationship_abc : 
  let a := Real.sqrt 2 / 2 * (Real.sin (17 * π / 180) + Real.cos (17 * π / 180))
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2622_262279


namespace NUMINAMATH_CALUDE_satisfying_polynomial_iff_quadratic_l2622_262278

/-- A polynomial that satisfies the given functional equation -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, P (a + b - 2*c) + P (b + c - 2*a) + P (a + c - 2*b) = 
               3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The theorem stating the equivalence between the functional equation and the quadratic form -/
theorem satisfying_polynomial_iff_quadratic :
  ∀ P : ℝ → ℝ, SatisfyingPolynomial P ↔ 
    ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
by sorry

end NUMINAMATH_CALUDE_satisfying_polynomial_iff_quadratic_l2622_262278


namespace NUMINAMATH_CALUDE_rectangular_garden_dimensions_l2622_262244

theorem rectangular_garden_dimensions (perimeter area fixed_side : ℝ) :
  perimeter = 60 →
  area = 200 →
  fixed_side = 10 →
  ∃ (adjacent_side : ℝ),
    adjacent_side = 20 ∧
    2 * (fixed_side + adjacent_side) = perimeter ∧
    fixed_side * adjacent_side = area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_garden_dimensions_l2622_262244


namespace NUMINAMATH_CALUDE_frank_bakes_for_five_days_l2622_262282

-- Define the problem parameters
def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def frank_eats_per_day : ℕ := 1
def ted_eats_last_day : ℕ := 4
def cookies_left : ℕ := 134

-- Define the function to calculate the number of days
def days_baking (cookies_per_tray trays_per_day frank_eats_per_day ted_eats_last_day cookies_left : ℕ) : ℕ :=
  (cookies_left + ted_eats_last_day) / (cookies_per_tray * trays_per_day - frank_eats_per_day)

-- Theorem statement
theorem frank_bakes_for_five_days :
  days_baking cookies_per_tray trays_per_day frank_eats_per_day ted_eats_last_day cookies_left = 5 :=
sorry

end NUMINAMATH_CALUDE_frank_bakes_for_five_days_l2622_262282


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l2622_262275

/-- The largest 2-digit prime factor of (300 choose 150) -/
def largest_two_digit_prime_factor_of_binom : ℕ := 97

/-- The binomial coefficient (300 choose 150) -/
def binom_300_150 : ℕ := Nat.choose 300 150

theorem largest_two_digit_prime_factor_of_binom_300_150 :
  largest_two_digit_prime_factor_of_binom = 97 ∧
  Nat.Prime largest_two_digit_prime_factor_of_binom ∧
  largest_two_digit_prime_factor_of_binom ≥ 10 ∧
  largest_two_digit_prime_factor_of_binom < 100 ∧
  (binom_300_150 % largest_two_digit_prime_factor_of_binom = 0) ∧
  ∀ p : ℕ, Nat.Prime p → p ≥ 10 → p < 100 → 
    (binom_300_150 % p = 0) → p ≤ largest_two_digit_prime_factor_of_binom :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_300_150_l2622_262275


namespace NUMINAMATH_CALUDE_triangle_area_l2622_262241

theorem triangle_area (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  (1/2) * a * b = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2622_262241


namespace NUMINAMATH_CALUDE_amp_specific_value_l2622_262283

/-- The operation & defined for real numbers -/
def amp (a b c d : ℝ) : ℝ := b^2 - 4*a*c + d

/-- Theorem stating that &(2, -3, 1, 5) = 6 -/
theorem amp_specific_value : amp 2 (-3) 1 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_amp_specific_value_l2622_262283


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2622_262219

-- Definition of opposite equations
def are_opposite_equations (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ ∃ x y : ℝ, a * x - b = 0 ∧ b * y - a = 0

-- Part (1)
theorem part_one : 
  are_opposite_equations 4 3 → are_opposite_equations 3 c → c = 4 :=
sorry

-- Part (2)
theorem part_two :
  are_opposite_equations 4 (-3 * m - 1) → are_opposite_equations 5 (n - 2) → m / n = -1/3 :=
sorry

-- Part (3)
theorem part_three :
  (∃ x : ℤ, 3 * x - c = 0) → (∃ y : ℤ, c * y - 3 = 0) → c = 3 ∨ c = -3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2622_262219


namespace NUMINAMATH_CALUDE_elderly_arrangement_theorem_l2622_262217

/-- The number of ways to arrange n distinct objects in a row -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then
    Nat.factorial n / Nat.factorial (n - k)
  else
    0

/-- The number of ways to arrange volunteers and elderly people with given constraints -/
def arrangement_count (volunteers elderly : ℕ) : ℕ :=
  permutations volunteers * arrangements (volunteers + 1) elderly

theorem elderly_arrangement_theorem :
  arrangement_count 4 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_elderly_arrangement_theorem_l2622_262217


namespace NUMINAMATH_CALUDE_image_equality_under_composition_condition_l2622_262287

universe u

theorem image_equality_under_composition_condition 
  {S : Type u} [Finite S] (f : S → S) :
  (∀ (g : S → S), g ≠ f → (f ∘ g ∘ f) ≠ (g ∘ f ∘ g)) →
  let T := Set.range f
  f '' T = T := by
  sorry

end NUMINAMATH_CALUDE_image_equality_under_composition_condition_l2622_262287


namespace NUMINAMATH_CALUDE_apple_vendor_problem_l2622_262276

theorem apple_vendor_problem (initial_apples : ℝ) (h_initial_positive : initial_apples > 0) :
  let first_day_sold := 0.6 * initial_apples
  let first_day_remainder := initial_apples - first_day_sold
  let x := (23 * initial_apples - 0.5 * first_day_remainder) / (0.5 * first_day_remainder)
  x = 0.15
  := by sorry

end NUMINAMATH_CALUDE_apple_vendor_problem_l2622_262276


namespace NUMINAMATH_CALUDE_star_calculation_l2622_262200

-- Define the ⋆ operation
def star (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem star_calculation : star (star 2 1) 4 = 259 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2622_262200


namespace NUMINAMATH_CALUDE_jack_payment_l2622_262207

/-- The amount Jack paid for sandwiches -/
def amount_paid : ℕ := sorry

/-- The number of sandwiches Jack ordered -/
def num_sandwiches : ℕ := 3

/-- The cost of each sandwich in dollars -/
def cost_per_sandwich : ℕ := 5

/-- The amount of change Jack received in dollars -/
def change_received : ℕ := 5

/-- Theorem stating that the amount Jack paid is $20 -/
theorem jack_payment : amount_paid = 20 := by
  sorry

end NUMINAMATH_CALUDE_jack_payment_l2622_262207


namespace NUMINAMATH_CALUDE_solar_systems_per_planet_l2622_262211

theorem solar_systems_per_planet (total_bodies : ℕ) (planets : ℕ) : 
  total_bodies = 200 → planets = 20 → (total_bodies - planets) / planets = 9 := by
sorry

end NUMINAMATH_CALUDE_solar_systems_per_planet_l2622_262211


namespace NUMINAMATH_CALUDE_multiplicative_inverse_7_mod_31_l2622_262268

theorem multiplicative_inverse_7_mod_31 : ∃ x : ℕ, x < 31 ∧ (7 * x) % 31 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_7_mod_31_l2622_262268


namespace NUMINAMATH_CALUDE_denver_birdhouse_profit_l2622_262267

/-- Represents the profit calculation for Denver's birdhouse business -/
theorem denver_birdhouse_profit :
  ∀ (wood_pieces : ℕ) (wood_cost : ℚ) (sale_price : ℚ),
    wood_pieces = 7 →
    wood_cost = 3/2 →
    sale_price = 32 →
    (sale_price / 2) - (wood_pieces : ℚ) * wood_cost = 11/2 :=
by
  sorry

end NUMINAMATH_CALUDE_denver_birdhouse_profit_l2622_262267


namespace NUMINAMATH_CALUDE_unique_b_solution_l2622_262209

theorem unique_b_solution (a b : ℕ) : 
  0 ≤ a → a < 2^2008 → 0 ≤ b → b < 8 → 
  (7 * (a + 2^2008 * b)) % 2^2011 = 1 → 
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_b_solution_l2622_262209


namespace NUMINAMATH_CALUDE_andrew_ate_77_donuts_l2622_262251

/-- The number of donuts Andrew ate on Monday -/
def monday_donuts : ℕ := 14

/-- The number of donuts Andrew ate on Tuesday -/
def tuesday_donuts : ℕ := monday_donuts / 2

/-- The number of donuts Andrew ate on Wednesday -/
def wednesday_donuts : ℕ := 4 * monday_donuts

/-- The total number of donuts Andrew ate in three days -/
def total_donuts : ℕ := monday_donuts + tuesday_donuts + wednesday_donuts

/-- Theorem stating that Andrew ate 77 donuts in total -/
theorem andrew_ate_77_donuts : total_donuts = 77 := by
  sorry

end NUMINAMATH_CALUDE_andrew_ate_77_donuts_l2622_262251


namespace NUMINAMATH_CALUDE_book_width_calculation_l2622_262295

theorem book_width_calculation (length width area : ℝ) : 
  length = 2 → area = 6 → area = length * width → width = 3 := by
  sorry

end NUMINAMATH_CALUDE_book_width_calculation_l2622_262295


namespace NUMINAMATH_CALUDE_trig_identity_l2622_262265

theorem trig_identity (α : Real) (h : 2 * Real.sin α + Real.cos α = 0) :
  2 * Real.sin α ^ 2 - 3 * Real.sin α * Real.cos α - 5 * Real.cos α ^ 2 = -12/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2622_262265


namespace NUMINAMATH_CALUDE_only_two_satisfies_condition_l2622_262247

def is_quadratic_residue (a p : ℕ) : Prop :=
  ∃ x, x^2 ≡ a [MOD p]

def all_quadratic_residues (p : ℕ) : Prop :=
  ∀ k ∈ Finset.range p, is_quadratic_residue (2 * (p / k) - 1) p

theorem only_two_satisfies_condition :
  ∀ p, Nat.Prime p → (all_quadratic_residues p ↔ p = 2) := by sorry

end NUMINAMATH_CALUDE_only_two_satisfies_condition_l2622_262247


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2622_262253

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : 4 * Real.sqrt 5 = 2 * Real.sqrt ((a^2 + b^2) : ℝ))
  (h_asymptote : b / a = 2) :
  a^2 = 4 ∧ b^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2622_262253


namespace NUMINAMATH_CALUDE_equation_system_solution_l2622_262206

theorem equation_system_solution : ∃! (a b c d e f : ℕ),
  (a ∈ Finset.range 10) ∧
  (b ∈ Finset.range 10) ∧
  (c ∈ Finset.range 10) ∧
  (d ∈ Finset.range 10) ∧
  (e ∈ Finset.range 10) ∧
  (f ∈ Finset.range 10) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧
  (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧
  (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧
  (d ≠ e) ∧ (d ≠ f) ∧
  (e ≠ f) ∧
  (20 * (a - 8) = 20) ∧
  (b / 2 + 17 = 20) ∧
  (c * 8 - 4 = 20) ∧
  ((d + 8) / 12 = 1) ∧
  (4 * e = 20) ∧
  (20 * (f - 2) = 100) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_system_solution_l2622_262206


namespace NUMINAMATH_CALUDE_inequality_proof_l2622_262261

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≥ Real.sqrt (3 / 2) * Real.sqrt (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2622_262261


namespace NUMINAMATH_CALUDE_value_of_k_l2622_262210

theorem value_of_k (k : ℝ) (h : 16 / k = 4) : k = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_k_l2622_262210


namespace NUMINAMATH_CALUDE_line_at_0_l2622_262239

/-- A line parameterized by t -/
def line (t : ℝ) : ℝ × ℝ := sorry

/-- The vector on the line at t = 1 is (2, 3) -/
axiom line_at_1 : line 1 = (2, 3)

/-- The vector on the line at t = 4 is (8, -5) -/
axiom line_at_4 : line 4 = (8, -5)

/-- The vector on the line at t = 5 is (10, -9) -/
axiom line_at_5 : line 5 = (10, -9)

/-- The vector on the line at t = 0 is (0, 17/3) -/
theorem line_at_0 : line 0 = (0, 17/3) := by sorry

end NUMINAMATH_CALUDE_line_at_0_l2622_262239


namespace NUMINAMATH_CALUDE_probability_yellow_second_marble_l2622_262240

-- Define the number of marbles in each bag
def bag_A_white : ℕ := 5
def bag_A_black : ℕ := 2
def bag_B_yellow : ℕ := 4
def bag_B_blue : ℕ := 5
def bag_C_yellow : ℕ := 3
def bag_C_blue : ℕ := 4
def bag_D_yellow : ℕ := 8
def bag_D_blue : ℕ := 2

-- Define the probabilities of drawing from each bag
def prob_white_A : ℚ := bag_A_white / (bag_A_white + bag_A_black)
def prob_black_A : ℚ := bag_A_black / (bag_A_white + bag_A_black)
def prob_yellow_B : ℚ := bag_B_yellow / (bag_B_yellow + bag_B_blue)
def prob_yellow_C : ℚ := bag_C_yellow / (bag_C_yellow + bag_C_blue)
def prob_yellow_D : ℚ := bag_D_yellow / (bag_D_yellow + bag_D_blue)

-- Assume equal probability of odd and even weight for black marbles
def prob_odd_weight : ℚ := 1/2
def prob_even_weight : ℚ := 1/2

-- Define the theorem
theorem probability_yellow_second_marble :
  prob_white_A * prob_yellow_B +
  prob_black_A * prob_odd_weight * prob_yellow_C +
  prob_black_A * prob_even_weight * prob_yellow_D = 211/245 := by
  sorry

end NUMINAMATH_CALUDE_probability_yellow_second_marble_l2622_262240


namespace NUMINAMATH_CALUDE_fraction_simplification_l2622_262288

/-- Proves that for x = 198719871987, the fraction 198719871987 / (x^2 - (x-1)(x+1)) simplifies to 1987 -/
theorem fraction_simplification (x : ℕ) (h : x = 198719871987) :
  (x : ℚ) / (x^2 - (x-1)*(x+1)) = 1987 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2622_262288


namespace NUMINAMATH_CALUDE_intersection_M_N_l2622_262299

def M : Set ℝ := {x : ℝ | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2622_262299


namespace NUMINAMATH_CALUDE_equation_satisfied_for_all_x_l2622_262237

theorem equation_satisfied_for_all_x (a b c x : ℝ) 
  (h : a / b = 2 ∧ b / c = 3/4) : 
  (a + b) * (c - x) / a^2 - (b + c) * (x - 2*c) / (b*c) - 
  (c + a) * (c - 2*x) / (a*c) = (a + b) * c / (a*b) + 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_for_all_x_l2622_262237


namespace NUMINAMATH_CALUDE_olivia_weekly_earnings_l2622_262245

/-- Olivia's weekly earnings calculation -/
theorem olivia_weekly_earnings 
  (hourly_wage : ℕ) 
  (monday_hours wednesday_hours friday_hours : ℕ) : 
  hourly_wage = 9 → 
  monday_hours = 4 → 
  wednesday_hours = 3 → 
  friday_hours = 6 → 
  hourly_wage * (monday_hours + wednesday_hours + friday_hours) = 117 := by
  sorry

end NUMINAMATH_CALUDE_olivia_weekly_earnings_l2622_262245


namespace NUMINAMATH_CALUDE_squirrel_climb_time_l2622_262218

/-- Represents the climbing behavior of a squirrel -/
structure SquirrelClimb where
  climb_rate : ℕ  -- metres climbed in odd minutes
  slip_rate : ℕ   -- metres slipped in even minutes
  total_height : ℕ -- total height of the pole to climb

/-- Calculates the time taken for a squirrel to climb a pole -/
def climb_time (s : SquirrelClimb) : ℕ :=
  sorry

/-- Theorem: A squirrel with given climbing behavior takes 17 minutes to climb 26 metres -/
theorem squirrel_climb_time :
  let s : SquirrelClimb := { climb_rate := 5, slip_rate := 2, total_height := 26 }
  climb_time s = 17 :=
by sorry

end NUMINAMATH_CALUDE_squirrel_climb_time_l2622_262218


namespace NUMINAMATH_CALUDE_rent_ratio_increase_l2622_262271

/-- The ratio of rent spent this year compared to last year, given changes in income and rent percentage --/
theorem rent_ratio_increase (last_year_rent_percent : ℝ) (income_increase_percent : ℝ) (this_year_rent_percent : ℝ) :
  last_year_rent_percent = 0.20 →
  income_increase_percent = 0.15 →
  this_year_rent_percent = 0.25 →
  (this_year_rent_percent * (1 + income_increase_percent)) / last_year_rent_percent = 1.4375 := by
  sorry

end NUMINAMATH_CALUDE_rent_ratio_increase_l2622_262271


namespace NUMINAMATH_CALUDE_even_function_implies_c_eq_neg_four_l2622_262292

/-- Given a function f and a constant c, we define g in terms of f and c. -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (c : ℝ) (x : ℝ) : ℝ := f x + c*x

/-- A function h is even if h(-x) = h(x) for all x. -/
def IsEven (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

/-- If g is an even function, then c must equal -4. -/
theorem even_function_implies_c_eq_neg_four :
  IsEven (g c) → c = -4 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_c_eq_neg_four_l2622_262292
