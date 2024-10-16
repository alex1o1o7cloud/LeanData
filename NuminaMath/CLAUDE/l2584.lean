import Mathlib

namespace NUMINAMATH_CALUDE_smallest_board_is_7x7_l2584_258488

/-- Represents a ship in the Battleship game -/
structure Ship :=
  (length : Nat)

/-- The complete set of ships for the Battleship game -/
def battleshipSet : List Ship := [
  ⟨4⟩,  -- One 1x4 ship
  ⟨3⟩, ⟨3⟩,  -- Two 1x3 ships
  ⟨2⟩, ⟨2⟩, ⟨2⟩,  -- Three 1x2 ships
  ⟨1⟩, ⟨1⟩, ⟨1⟩, ⟨1⟩  -- Four 1x1 ships
]

/-- Represents a square board -/
structure Board :=
  (size : Nat)

/-- Checks if a given board can fit all ships without touching -/
def canFitShips (board : Board) (ships : List Ship) : Prop :=
  sorry

/-- Theorem stating that 7x7 is the smallest square board that can fit all ships -/
theorem smallest_board_is_7x7 :
  (∀ b : Board, b.size < 7 → ¬(canFitShips b battleshipSet)) ∧
  (canFitShips ⟨7⟩ battleshipSet) :=
sorry

end NUMINAMATH_CALUDE_smallest_board_is_7x7_l2584_258488


namespace NUMINAMATH_CALUDE_bobby_chocolate_pieces_l2584_258414

/-- The number of chocolate pieces Bobby ate -/
def chocolate_pieces (initial_candy pieces_more_candy total_pieces : ℕ) : ℕ :=
  total_pieces - (initial_candy + pieces_more_candy)

theorem bobby_chocolate_pieces :
  chocolate_pieces 33 4 51 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bobby_chocolate_pieces_l2584_258414


namespace NUMINAMATH_CALUDE_max_non_managers_dept_A_l2584_258443

/-- Represents a department in the company -/
inductive Department
| A
| B
| C

/-- Represents the gender of an employee -/
inductive Gender
| Male
| Female

/-- Represents the status of a manager -/
inductive ManagerStatus
| Active
| OnVacation

/-- Represents the type of non-manager employee -/
inductive NonManagerType
| FullTime
| PartTime

/-- The company structure and policies -/
structure Company where
  /-- The ratio of managers to non-managers must be greater than this for all departments -/
  baseRatio : Rat
  /-- Department A's specific ratio requirement -/
  deptARatio : Rat
  /-- Department B's specific ratio requirement -/
  deptBRatio : Rat
  /-- The minimum gender ratio (male:female) for non-managers -/
  genderRatio : Rat

/-- Represents the workforce of a department -/
structure DepartmentWorkforce where
  department : Department
  totalManagers : Nat
  activeManagers : Nat
  nonManagersMale : Nat
  nonManagersFemale : Nat
  partTimeNonManagers : Nat

/-- Main theorem to prove -/
theorem max_non_managers_dept_A (c : Company) (dA : DepartmentWorkforce) :
  c.baseRatio = 7/32 ∧
  c.deptARatio = 9/33 ∧
  c.deptBRatio = 8/34 ∧
  c.genderRatio = 1/2 ∧
  dA.department = Department.A ∧
  dA.totalManagers = 8 ∧
  dA.activeManagers = 4 →
  dA.nonManagersMale + dA.nonManagersFemale + dA.partTimeNonManagers / 2 ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_non_managers_dept_A_l2584_258443


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2584_258416

/-- A continuous function satisfying the given functional equation is either constantly 0 or 1/2. -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : Continuous f)
  (h : ∀ x y : ℝ, f (x^2 - y^2) = f x^2 + f y^2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2584_258416


namespace NUMINAMATH_CALUDE_area_of_folded_rectangle_l2584_258462

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle -/
structure Rectangle :=
  (A B C D : Point)

/-- Represents the folded configuration -/
structure FoldedConfig :=
  (rect : Rectangle)
  (E F B' C' : Point)

/-- The main theorem -/
theorem area_of_folded_rectangle 
  (config : FoldedConfig) 
  (h1 : config.rect.A.x < config.E.x) -- E is on AB
  (h2 : config.rect.C.x > config.F.x) -- F is on CD
  (h3 : config.E.x - config.rect.B.x < config.rect.C.x - config.F.x) -- BE < CF
  (h4 : config.C'.y = config.rect.A.y) -- C' is on AD
  (h5 : (config.B'.x - config.rect.A.x) * (config.C'.y - config.E.y) = 
        (config.C'.x - config.rect.A.x) * (config.B'.y - config.E.y)) -- ∠AB'C' ≅ ∠B'EA
  (h6 : Real.sqrt ((config.B'.x - config.rect.A.x)^2 + (config.B'.y - config.rect.A.y)^2) = 7) -- AB' = 7
  (h7 : config.E.x - config.rect.B.x = 17) -- BE = 17
  : (config.rect.B.x - config.rect.A.x) * (config.rect.C.y - config.rect.A.y) = 
    (1372 + 833 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_folded_rectangle_l2584_258462


namespace NUMINAMATH_CALUDE_max_piles_660_l2584_258463

/-- The maximum number of piles that can be created from a given number of stones,
    where any two pile sizes differ by strictly less than 2 times. -/
def maxPiles (totalStones : ℕ) : ℕ :=
  30 -- The actual implementation is not provided, just the result

/-- The condition that any two pile sizes differ by strictly less than 2 times -/
def validPileSizes (piles : List ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ piles → b ∈ piles → (a : ℝ) < 2 * b ∧ (b : ℝ) < 2 * a

theorem max_piles_660 :
  let n := maxPiles 660
  ∃ (piles : List ℕ), 
    piles.length = n ∧ 
    validPileSizes piles ∧ 
    piles.sum = 660 ∧
    ∀ (m : ℕ), m > n → 
      ¬∃ (largerPiles : List ℕ), 
        largerPiles.length = m ∧ 
        validPileSizes largerPiles ∧ 
        largerPiles.sum = 660 :=
by
  sorry

#eval maxPiles 660

end NUMINAMATH_CALUDE_max_piles_660_l2584_258463


namespace NUMINAMATH_CALUDE_conditional_prob_B_given_A_l2584_258415

/-- The number of class officers -/
def total_officers : ℕ := 6

/-- The number of boys among the class officers -/
def num_boys : ℕ := 4

/-- The number of girls among the class officers -/
def num_girls : ℕ := 2

/-- The number of students selected -/
def num_selected : ℕ := 3

/-- Event A: "boy A being selected" -/
def event_A : Set (Fin total_officers) := sorry

/-- Event B: "girl B being selected" -/
def event_B : Set (Fin total_officers) := sorry

/-- The probability of event A -/
def prob_A : ℚ := 1 / 2

/-- The probability of both events A and B occurring -/
def prob_AB : ℚ := 1 / 5

/-- Theorem: The conditional probability P(B|A) is 2/5 -/
theorem conditional_prob_B_given_A : 
  (prob_AB / prob_A : ℚ) = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_conditional_prob_B_given_A_l2584_258415


namespace NUMINAMATH_CALUDE_problem_solution_l2584_258483

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^5 + 3*y^3) / 8 = 54.375 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2584_258483


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2584_258451

/-- Given two lines in the form ax + by + c = 0, this function returns true if they are perpendicular --/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- Given a line in the form ax + by + c = 0 and a point (x, y), this function returns true if the point lies on the line --/
def point_on_line (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

theorem perpendicular_line_through_point :
  are_perpendicular 1 (-2) 2 1 ∧
  point_on_line 1 (-2) 3 1 2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2584_258451


namespace NUMINAMATH_CALUDE_statement_B_is_incorrect_l2584_258495

-- Define the basic types
def Chromosome : Type := String
def Allele : Type := String
def Genotype : Type := List Allele

-- Define the meiosis process
def meiosis (g : Genotype) : List Genotype := sorry

-- Define the normal chromosome distribution
def normalChromosomeDistribution (g : Genotype) : Prop := sorry

-- Define the statement B
def statementB : Prop :=
  ∃ (parent : Genotype) (sperm : Genotype),
    parent = ["A", "a", "X^b", "Y"] ∧
    sperm ∈ meiosis parent ∧
    sperm = ["A", "A", "a", "Y"] ∧
    (∃ (other_sperms : List Genotype),
      other_sperms.length = 3 ∧
      (∀ s ∈ other_sperms, s ∈ meiosis parent) ∧
      other_sperms = [["a", "Y"], ["X^b"], ["X^b"]])

-- Theorem stating that B is incorrect
theorem statement_B_is_incorrect :
  ¬statementB :=
sorry

end NUMINAMATH_CALUDE_statement_B_is_incorrect_l2584_258495


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l2584_258475

theorem compare_negative_fractions : -2/3 > -3/4 := by sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l2584_258475


namespace NUMINAMATH_CALUDE_bagged_sugar_weight_recording_l2584_258470

/-- Represents the recording of a bag's weight difference from the standard -/
def weightDifference (standardWeight actual : ℕ) : ℤ :=
  (actual : ℤ) - (standardWeight : ℤ)

/-- Proves that a bag weighing 498 grams should be recorded as -3 grams when the standard is 501 grams -/
theorem bagged_sugar_weight_recording :
  let standardWeight : ℕ := 501
  let actualWeight : ℕ := 498
  weightDifference standardWeight actualWeight = -3 := by
sorry

end NUMINAMATH_CALUDE_bagged_sugar_weight_recording_l2584_258470


namespace NUMINAMATH_CALUDE_equation_solution_l2584_258418

theorem equation_solution : ∃ x : ℝ, 61 + 5 * 12 / (180 / x) = 62 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2584_258418


namespace NUMINAMATH_CALUDE_milk_processing_time_l2584_258458

/-- Given two milk processing plants with the following conditions:
    - They process equal amounts of milk
    - The second plant starts 'a' days later than the first
    - The second plant processes 'm' liters more per day than the first
    - After 5a/9 days of joint work, 1/3 of the task remains
    - The work finishes simultaneously
    - Each plant processes half of the total volume

    Prove that the total number of days required to complete the task is 2a
-/
theorem milk_processing_time (a m : ℝ) (a_pos : 0 < a) (m_pos : 0 < m) : 
  ∃ (n : ℝ), n > 0 ∧ 
  (∃ (x : ℝ), x > 0 ∧ 
    (n * x = (n - a) * (x + m)) ∧ 
    (a * x + (5 * a / 9) * (2 * x + m) = 2 / 3) ∧
    (n * x = 1 / 2)) ∧
  n = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_milk_processing_time_l2584_258458


namespace NUMINAMATH_CALUDE_chanhee_walking_distance_l2584_258487

/-- Calculates the total distance walked given step length, duration, and pace. -/
def total_distance (step_length : Real) (duration : Real) (pace : Real) : Real :=
  step_length * duration * pace

/-- Proves that Chanhee walked 526.5 meters given the specified conditions. -/
theorem chanhee_walking_distance :
  let step_length : Real := 0.45
  let duration : Real := 13
  let pace : Real := 90
  total_distance step_length duration pace = 526.5 := by
  sorry

end NUMINAMATH_CALUDE_chanhee_walking_distance_l2584_258487


namespace NUMINAMATH_CALUDE_phil_charlie_difference_l2584_258427

/-- Represents the number of games won by each player -/
structure GamesWon where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- Conditions for the golf game results -/
def golf_conditions (g : GamesWon) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie = g.dana - 2 ∧
  g.phil > g.charlie ∧
  g.phil = 12 ∧
  g.perry = g.phil + 4

/-- Theorem stating the difference between Phil's and Charlie's games -/
theorem phil_charlie_difference (g : GamesWon) (h : golf_conditions g) : 
  g.phil - g.charlie = 3 := by
  sorry

end NUMINAMATH_CALUDE_phil_charlie_difference_l2584_258427


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_l2584_258430

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  let x := (l1.b * l3.c - l3.b * l1.c) / (l1.a * l3.b - l3.a * l1.b)
  let y := (l3.a * l1.c - l1.a * l3.c) / (l1.a * l3.b - l3.a * l1.b)
  l2.a * x + l2.b * y = l2.c

/-- The main theorem -/
theorem lines_cannot_form_triangle (m : ℝ) : 
  let l1 : Line := ⟨4, 1, 4⟩
  let l2 : Line := ⟨m, 1, 0⟩
  let l3 : Line := ⟨2, -3, 4⟩
  (parallel l1 l2 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) →
  m = 4 ∨ m = 1/2 ∨ m = -2/3 :=
by sorry


end NUMINAMATH_CALUDE_lines_cannot_form_triangle_l2584_258430


namespace NUMINAMATH_CALUDE_pages_remaining_l2584_258467

/-- Given a book with 93 pages, if Jerry reads 30 pages on Saturday and 20 pages on Sunday,
    then the number of pages remaining to finish the book is 43. -/
theorem pages_remaining (total_pages : Nat) (pages_read_saturday : Nat) (pages_read_sunday : Nat)
    (h1 : total_pages = 93)
    (h2 : pages_read_saturday = 30)
    (h3 : pages_read_sunday = 20) :
    total_pages - pages_read_saturday - pages_read_sunday = 43 := by
  sorry

end NUMINAMATH_CALUDE_pages_remaining_l2584_258467


namespace NUMINAMATH_CALUDE_min_cans_needed_l2584_258448

/-- The capacity of each can in ounces -/
def can_capacity : ℕ := 15

/-- The minimum amount of soda needed in ounces -/
def min_soda_amount : ℕ := 192

/-- The minimum number of cans needed -/
def min_cans : ℕ := 13

theorem min_cans_needed : 
  (∀ n : ℕ, n * can_capacity ≥ min_soda_amount → n ≥ min_cans) ∧ 
  (min_cans * can_capacity ≥ min_soda_amount) := by
  sorry

end NUMINAMATH_CALUDE_min_cans_needed_l2584_258448


namespace NUMINAMATH_CALUDE_inequality_proof_l2584_258422

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  64 * (a * b * c * d + 1) / (a + b + c + d)^2 ≤ 
  a^2 + b^2 + c^2 + d^2 + 1/a^2 + 1/b^2 + 1/c^2 + 1/d^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2584_258422


namespace NUMINAMATH_CALUDE_dusty_change_l2584_258490

/-- Represents the price of a single layer cake slice in dollars -/
def single_layer_price : ℕ := 4

/-- Represents the price of a double layer cake slice in dollars -/
def double_layer_price : ℕ := 7

/-- Represents the number of single layer cake slices Dusty buys -/
def single_layer_quantity : ℕ := 7

/-- Represents the number of double layer cake slices Dusty buys -/
def double_layer_quantity : ℕ := 5

/-- Represents the amount Dusty pays with in dollars -/
def payment : ℕ := 100

/-- Theorem stating that Dusty's change is $37 -/
theorem dusty_change : 
  payment - (single_layer_price * single_layer_quantity + double_layer_price * double_layer_quantity) = 37 := by
  sorry

end NUMINAMATH_CALUDE_dusty_change_l2584_258490


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2584_258493

/-- A perfect square trinomial in x and y -/
def isPerfectSquareTrinomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, f x y = (x + k*y)^2 ∨ f x y = (x - k*y)^2

/-- The theorem stating that if x^2 + axy + y^2 is a perfect square trinomial, then a = 2 or a = -2 -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  isPerfectSquareTrinomial (fun x y => x^2 + a*x*y + y^2) → a = 2 ∨ a = -2 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l2584_258493


namespace NUMINAMATH_CALUDE_smallest_multiple_l2584_258453

theorem smallest_multiple (n : ℕ) : n = 1050 ↔ 
  n > 0 ∧ 
  50 ∣ n ∧ 
  75 ∣ n ∧ 
  ¬(18 ∣ n) ∧ 
  7 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 50 ∣ m → 75 ∣ m → ¬(18 ∣ m) → 7 ∣ m → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2584_258453


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2584_258449

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 2) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2584_258449


namespace NUMINAMATH_CALUDE_line_perpendicular_to_line_l2584_258404

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_line
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : subset_line_plane m β)
  (h3 : perpendicular_plane_plane α β) :
  perpendicular_line_line l m :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_line_l2584_258404


namespace NUMINAMATH_CALUDE_min_fountains_correct_l2584_258439

/-- Represents a water fountain on a grid -/
structure Fountain where
  row : Nat
  col : Nat

/-- Checks if a fountain can spray a given square -/
def can_spray (f : Fountain) (row col : Nat) : Bool :=
  (f.row = row && (f.col = col - 1 || f.col = col + 1)) ||
  (f.col = col && (f.row = row - 1 || f.row = row + 1 || f.row = row - 2))

/-- Calculates the minimum number of fountains required for a given grid size -/
def min_fountains (m n : Nat) : Nat :=
  if m = 4 then
    2 * ((n + 2) / 3)
  else if m = 3 then
    3 * ((n + 2) / 3)
  else
    0  -- undefined for other cases

theorem min_fountains_correct (m n : Nat) :
  (m = 4 || m = 3) →
  ∃ (fountains : List Fountain),
    (fountains.length = min_fountains m n) ∧
    (∀ row col, row < m ∧ col < n →
      ∃ f ∈ fountains, can_spray f row col) :=
by sorry

#eval min_fountains 4 10  -- Expected: 8
#eval min_fountains 3 10  -- Expected: 12

end NUMINAMATH_CALUDE_min_fountains_correct_l2584_258439


namespace NUMINAMATH_CALUDE_rational_cube_sum_zero_l2584_258471

theorem rational_cube_sum_zero (x y z : ℚ) 
  (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_zero_l2584_258471


namespace NUMINAMATH_CALUDE_grocery_store_soda_l2584_258437

theorem grocery_store_soda (diet_soda apples : ℕ) 
  (h1 : diet_soda = 32)
  (h2 : apples = 78)
  (h3 : ∃ regular_soda : ℕ, regular_soda + diet_soda = apples + 26) :
  ∃ regular_soda : ℕ, regular_soda = 72 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l2584_258437


namespace NUMINAMATH_CALUDE_oliver_spending_l2584_258408

theorem oliver_spending (initial_amount spent_amount received_amount final_amount : ℕ) :
  initial_amount = 33 →
  received_amount = 32 →
  final_amount = 61 →
  final_amount = initial_amount - spent_amount + received_amount →
  spent_amount = 4 := by
sorry

end NUMINAMATH_CALUDE_oliver_spending_l2584_258408


namespace NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l2584_258455

theorem sufficient_condition_for_f_less_than_one
  (a : ℝ) (h_a : a > 1) :
  ∃ (x : ℝ), -1 < x ∧ x < 0 ∧ a * x + 2 * x < 1 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l2584_258455


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2584_258411

theorem fractional_equation_solution :
  ∃ x : ℝ, (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2584_258411


namespace NUMINAMATH_CALUDE_sum_of_squares_of_factors_72_l2584_258486

def sum_of_squares_of_factors (n : ℕ) : ℕ := sorry

theorem sum_of_squares_of_factors_72 : sum_of_squares_of_factors 72 = 7735 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_factors_72_l2584_258486


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l2584_258421

theorem student_average_greater_than_true_average (x y z : ℝ) (h : x < y ∧ y < z) :
  (x + y) / 2 + z > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l2584_258421


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l2584_258474

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l2584_258474


namespace NUMINAMATH_CALUDE_all_dice_same_number_probability_l2584_258417

/-- The probability of a single die showing a specific number -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice being tossed -/
def num_dice : ℕ := 4

/-- The probability of all dice showing the same number -/
def all_same_prob : ℚ := (single_die_prob) ^ (num_dice - 1)

theorem all_dice_same_number_probability :
  all_same_prob = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_all_dice_same_number_probability_l2584_258417


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2584_258484

/-- Given a triangle with sides 13, 14, and 15, the shortest altitude has length 168/15 -/
theorem shortest_altitude_of_triangle (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h1 := 2 * area / a
  let h2 := 2 * area / b
  let h3 := 2 * area / c
  min h1 (min h2 h3) = 168 / 15 := by
sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l2584_258484


namespace NUMINAMATH_CALUDE_haley_zoo_pictures_l2584_258489

/-- The number of pictures Haley took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The total number of pictures Haley took before deleting any -/
def total_pictures : ℕ := zoo_pictures + 8

/-- The number of pictures Haley had after deleting some -/
def remaining_pictures : ℕ := total_pictures - 38

theorem haley_zoo_pictures :
  zoo_pictures = 50 ∧ remaining_pictures = 20 :=
sorry

end NUMINAMATH_CALUDE_haley_zoo_pictures_l2584_258489


namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l2584_258441

/-- Represents the number of items in each stratum -/
structure Strata :=
  (grains : ℕ)
  (vegetable_oils : ℕ)
  (animal_foods : ℕ)
  (fruits_and_vegetables : ℕ)

/-- Calculates the total number of items across all strata -/
def total_items (s : Strata) : ℕ :=
  s.grains + s.vegetable_oils + s.animal_foods + s.fruits_and_vegetables

/-- Calculates the number of items to sample from a stratum -/
def stratum_sample (total_sample : ℕ) (stratum_size : ℕ) (s : Strata) : ℕ :=
  (total_sample * stratum_size) / (total_items s)

/-- The main theorem to prove -/
theorem stratified_sampling_sum (s : Strata) (total_sample : ℕ) :
  s.grains = 40 →
  s.vegetable_oils = 10 →
  s.animal_foods = 30 →
  s.fruits_and_vegetables = 20 →
  total_sample = 20 →
  (stratum_sample total_sample s.vegetable_oils s +
   stratum_sample total_sample s.fruits_and_vegetables s) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sum_l2584_258441


namespace NUMINAMATH_CALUDE_cubic_fraction_simplification_l2584_258434

theorem cubic_fraction_simplification 
  (a b x : ℝ) 
  (h1 : x = a^3 / b^3) 
  (h2 : a ≠ b) 
  (h3 : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_simplification_l2584_258434


namespace NUMINAMATH_CALUDE_binary_1010101_equals_85_l2584_258442

def binaryToDecimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010101_equals_85 :
  binaryToDecimal [true, false, true, false, true, false, true] = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_equals_85_l2584_258442


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l2584_258423

theorem sqrt_sum_equality (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + b*c + c*a = 0 ∧ a + b + c ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l2584_258423


namespace NUMINAMATH_CALUDE_largest_quantity_l2584_258457

def A : ℚ := 3004 / 3003 + 3004 / 3005
def B : ℚ := 3006 / 3005 + 3006 / 3007
def C : ℚ := 3005 / 3004 + 3005 / 3006

theorem largest_quantity : A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_quantity_l2584_258457


namespace NUMINAMATH_CALUDE_pokemon_cards_bought_l2584_258426

theorem pokemon_cards_bought (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : final_cards = 900) :
  final_cards - initial_cards = 224 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_bought_l2584_258426


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2584_258494

theorem quadratic_equation_solution : 
  ∃ (a b : ℝ), 
    (a^2 - 6*a + 9 = 15) ∧ 
    (b^2 - 6*b + 9 = 15) ∧ 
    (a ≥ b) ∧ 
    (3*a - b = 6 + 4*Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2584_258494


namespace NUMINAMATH_CALUDE_factorization_equality_l2584_258482

theorem factorization_equality (x : ℝ) : x^3 - 2*x^2 + x = x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2584_258482


namespace NUMINAMATH_CALUDE_sum_of_sqrt_geq_sum_of_products_l2584_258450

theorem sum_of_sqrt_geq_sum_of_products (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 3) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + a * c := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_geq_sum_of_products_l2584_258450


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l2584_258420

/-- Given a circular piece of tissue with an actual diameter and a magnification factor,
    calculate the diameter of the magnified image. -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem: The diameter of a circular piece of tissue with an actual diameter of 0.001 cm,
    when magnified 1,000 times, is 1 cm. -/
theorem magnified_tissue_diameter :
  magnified_diameter 0.001 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l2584_258420


namespace NUMINAMATH_CALUDE_at_least_one_solution_l2584_258461

-- Define the polynomials
variable (P S T : ℂ → ℂ)

-- Define the properties of the polynomials
axiom P_degree : ∃ (a b c : ℂ), ∀ z, P z = z^3 + a*z^2 + b*z + 4
axiom S_degree : ∃ (a b c d : ℂ), ∀ z, S z = z^4 + a*z^3 + b*z^2 + c*z + 5
axiom T_degree : ∃ (a b c d e f g : ℂ), ∀ z, T z = z^7 + a*z^6 + b*z^5 + c*z^4 + d*z^3 + e*z^2 + f*z + 20

-- Theorem statement
theorem at_least_one_solution :
  ∃ z : ℂ, P z * S z = T z :=
sorry

end NUMINAMATH_CALUDE_at_least_one_solution_l2584_258461


namespace NUMINAMATH_CALUDE_g_2023_of_2_eq_2_l2584_258410

def g (x : ℚ) : ℚ := (2 - x) / (2 * x + 1)

def g_n : ℕ → ℚ → ℚ
  | 0, x => x
  | 1, x => g x
  | (n + 2), x => g (g_n (n + 1) x)

theorem g_2023_of_2_eq_2 : g_n 2023 2 = 2 := by sorry

end NUMINAMATH_CALUDE_g_2023_of_2_eq_2_l2584_258410


namespace NUMINAMATH_CALUDE_problem_solution_l2584_258459

theorem problem_solution (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 9) 
  (h3 : x = 0) : 
  y = 33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2584_258459


namespace NUMINAMATH_CALUDE_two_bagels_solution_l2584_258479

/-- Represents the number of items bought in a week -/
structure WeeklyPurchase where
  bagels : ℕ
  muffins : ℕ
  donuts : ℕ

/-- Checks if the weekly purchase is valid (totals to 6 days) -/
def isValidPurchase (wp : WeeklyPurchase) : Prop :=
  wp.bagels + wp.muffins + wp.donuts = 6

/-- Calculates the total cost in cents -/
def totalCost (wp : WeeklyPurchase) : ℕ :=
  60 * wp.bagels + 45 * wp.muffins + 30 * wp.donuts

/-- Checks if the total cost is a whole number of dollars -/
def isWholeDollarAmount (wp : WeeklyPurchase) : Prop :=
  totalCost wp % 100 = 0

/-- Main theorem: There exists a valid purchase with 2 bagels that costs a whole dollar amount -/
theorem two_bagels_solution :
  ∃ (wp : WeeklyPurchase), wp.bagels = 2 ∧ isValidPurchase wp ∧ isWholeDollarAmount wp :=
sorry

end NUMINAMATH_CALUDE_two_bagels_solution_l2584_258479


namespace NUMINAMATH_CALUDE_problem_proof_l2584_258492

theorem problem_proof (a b : ℝ) (ha : a > 0) (h : Real.exp a + Real.log b = 1) :
  a * b < 1 ∧ a + b > 1 ∧ Real.exp a + b > 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2584_258492


namespace NUMINAMATH_CALUDE_orange_juice_amount_l2584_258454

theorem orange_juice_amount (total ingredients : ℝ) 
  (strawberries yogurt : ℝ) (h1 : total = 0.5) 
  (h2 : strawberries = 0.2) (h3 : yogurt = 0.1) :
  total - (strawberries + yogurt) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_amount_l2584_258454


namespace NUMINAMATH_CALUDE_squared_sum_of_x_and_y_l2584_258440

theorem squared_sum_of_x_and_y (x y : ℝ) 
  (h : (2*x^2 + 2*y^2 + 3)*(2*x^2 + 2*y^2 - 3) = 27) : 
  x^2 + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_squared_sum_of_x_and_y_l2584_258440


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2584_258496

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 54 → s^3 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2584_258496


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2584_258491

theorem exponent_multiplication (a : ℝ) : a * a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2584_258491


namespace NUMINAMATH_CALUDE_badminton_medals_count_l2584_258460

/-- Proves that the number of badminton medals is 5 --/
theorem badminton_medals_count :
  ∀ (total_medals track_medals swimming_medals badminton_medals : ℕ),
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  badminton_medals = total_medals - (track_medals + swimming_medals) →
  badminton_medals = 5 := by
  sorry

end NUMINAMATH_CALUDE_badminton_medals_count_l2584_258460


namespace NUMINAMATH_CALUDE_tan_triple_inequality_l2584_258436

theorem tan_triple_inequality (x y : Real) 
  (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2)
  (h_tan : Real.tan x = 3 * Real.tan y) :
  x - y ≤ Real.pi / 6 ∧
  (x - y = Real.pi / 6 ↔ x = Real.pi / 3 ∧ y = Real.pi / 6) := by
sorry

end NUMINAMATH_CALUDE_tan_triple_inequality_l2584_258436


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2584_258444

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 540) : 
  1.2 * L * (0.8 * W) = 518.4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2584_258444


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2584_258403

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (k : ℝ) :
  ∀ f : RealFunction, 
    (∀ x y : ℝ, f (f x + f y + k * x * y) = x * f y + y * f x) →
    (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2584_258403


namespace NUMINAMATH_CALUDE_pencils_lost_l2584_258480

/-- Given an initial number of pencils and a final number of pencils,
    prove that the number of lost pencils is the difference between them. -/
theorem pencils_lost (initial final : ℕ) (h : initial ≥ final) :
  initial - final = initial - final :=
by sorry

end NUMINAMATH_CALUDE_pencils_lost_l2584_258480


namespace NUMINAMATH_CALUDE_books_returned_count_l2584_258447

/-- Represents the number of books Mary has at different stages --/
structure BookCount where
  initial : Nat
  after_first_return : Nat
  after_second_checkout : Nat
  final : Nat

/-- Represents Mary's library transactions --/
def library_transactions (x : Nat) : BookCount :=
  { initial := 5,
    after_first_return := 5 - x + 5,
    after_second_checkout := 5 - x + 5 - 2 + 7,
    final := 12 }

/-- Theorem stating the number of books Mary returned --/
theorem books_returned_count : ∃ x : Nat, 
  (library_transactions x).final = 12 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_books_returned_count_l2584_258447


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l2584_258429

def P : Set ℝ := {x | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x | 0 < Real.log x ∧ Real.log x ≤ Real.log 2}

theorem complement_P_intersect_Q : 
  (Set.compl P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l2584_258429


namespace NUMINAMATH_CALUDE_inequality_abc_at_least_one_positive_l2584_258446

-- Problem 1
theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) := by
  sorry

-- Problem 2
theorem at_least_one_positive (x y z : ℝ) :
  let a := x^2 - 2*y + π/2
  let b := y^2 - 2*z + π/3
  let c := z^2 - 2*x + π/6
  0 < a ∨ 0 < b ∨ 0 < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_at_least_one_positive_l2584_258446


namespace NUMINAMATH_CALUDE_distance_to_double_reflection_distance_C_to_C_l2584_258473

/-- The distance between a point and its reflection over both x and y axes --/
theorem distance_to_double_reflection (x y : ℝ) : 
  let C : ℝ × ℝ := (x, y)
  let C' : ℝ × ℝ := (-x, -y)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = Real.sqrt (4 * (x^2 + y^2)) :=
by sorry

/-- The specific case for point C(-3, 2) --/
theorem distance_C_to_C'_is_sqrt_52 : 
  let C : ℝ × ℝ := (-3, 2)
  let C' : ℝ × ℝ := (3, -2)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = Real.sqrt 52 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_double_reflection_distance_C_to_C_l2584_258473


namespace NUMINAMATH_CALUDE_weight_of_smaller_cube_l2584_258465

/-- Given two cubes of the same material, where the second cube has sides twice as long as the first
and weighs 64 pounds, the weight of the first cube is 8 pounds. -/
theorem weight_of_smaller_cube (s : ℝ) (weight_first : ℝ) (weight_second : ℝ) : 
  s > 0 → 
  weight_second = 64 → 
  (2 * s)^3 / s^3 * weight_first = weight_second → 
  weight_first = 8 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_smaller_cube_l2584_258465


namespace NUMINAMATH_CALUDE_isosceles_triangle_from_quadratic_perimeter_l2584_258406

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Checks if a quadratic equation has two real roots -/
def has_two_real_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq > 0

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.base + 2*t.leg

theorem isosceles_triangle_from_quadratic_perimeter
  (k : ℝ)
  (eq : QuadraticEquation)
  (t : IsoscelesTriangle)
  (h1 : eq = { a := 1, b := -4, c := 2*k })
  (h2 : has_two_real_roots eq)
  (h3 : t.base = 1)
  (h4 : t.leg = 2) :
  perimeter t = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_from_quadratic_perimeter_l2584_258406


namespace NUMINAMATH_CALUDE_committee_selection_count_club_committee_count_l2584_258477

theorem committee_selection_count : Nat → Nat → Nat
  | n, k => (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem club_committee_count :
  committee_selection_count 30 5 = 142506 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_club_committee_count_l2584_258477


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2584_258407

theorem arithmetic_sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 2 → a n - a (n - 1) = 2) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2584_258407


namespace NUMINAMATH_CALUDE_given_number_proof_l2584_258469

theorem given_number_proof : ∃ x : ℤ, (143 - 10 * x = 3 * x) ∧ (x = 11) := by sorry

end NUMINAMATH_CALUDE_given_number_proof_l2584_258469


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l2584_258476

theorem min_value_of_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

theorem lower_bound_achievable : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l2584_258476


namespace NUMINAMATH_CALUDE_expression_evaluation_l2584_258435

theorem expression_evaluation : 150 * (150 - 4) - (150 * 150 - 6 + 2) = -596 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2584_258435


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l2584_258485

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (5*x - 4)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l2584_258485


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l2584_258405

theorem polynomial_product_expansion (x : ℝ) : 
  (5 * x + 3) * (6 * x^2 + 2) = 30 * x^3 + 18 * x^2 + 10 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l2584_258405


namespace NUMINAMATH_CALUDE_complex_polynomial_root_implies_abs_c_165_l2584_258400

def complex_polynomial (a b c : ℤ) : ℂ → ℂ := fun z ↦ a * z^4 + b * z^3 + c * z^2 + b * z + a

theorem complex_polynomial_root_implies_abs_c_165 (a b c : ℤ) :
  complex_polynomial a b c (3 + I) = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 165 := by sorry

end NUMINAMATH_CALUDE_complex_polynomial_root_implies_abs_c_165_l2584_258400


namespace NUMINAMATH_CALUDE_min_value_of_some_expression_l2584_258432

/-- The minimum value of |some expression| given the conditions -/
theorem min_value_of_some_expression :
  ∃ (f : ℝ → ℝ),
    (∀ x, |x - 4| + |x + 7| + |f x| ≥ 12) ∧
    (∃ x₀, |x₀ - 4| + |x₀ + 7| + |f x₀| = 12) →
    ∃ x₁, |f x₁| = 1 ∧ ∀ x, |f x| ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_some_expression_l2584_258432


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l2584_258452

/-- Given vectors a and b that are parallel, prove that their sum has magnitude √5 -/
theorem parallel_vectors_sum_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, -4]
  (∃ (k : ℝ), a = k • b) → 
  ‖a + b‖ = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_magnitude_l2584_258452


namespace NUMINAMATH_CALUDE_annas_size_l2584_258497

theorem annas_size (anna_size : ℕ) 
  (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : becky_size = 3 * anna_size)
  (h2 : ginger_size = 2 * becky_size - 4)
  (h3 : ginger_size = 8) : 
  anna_size = 2 := by
  sorry

end NUMINAMATH_CALUDE_annas_size_l2584_258497


namespace NUMINAMATH_CALUDE_club_average_age_l2584_258499

theorem club_average_age (women : ℕ) (men : ℕ) (children : ℕ)
  (women_avg : ℝ) (men_avg : ℝ) (children_avg : ℝ)
  (h_women : women = 12)
  (h_men : men = 18)
  (h_children : children = 10)
  (h_women_avg : women_avg = 32)
  (h_men_avg : men_avg = 38)
  (h_children_avg : children_avg = 10) :
  (women * women_avg + men * men_avg + children * children_avg) / (women + men + children) = 29.2 := by
  sorry

end NUMINAMATH_CALUDE_club_average_age_l2584_258499


namespace NUMINAMATH_CALUDE_sum_x_coordinates_q3_l2584_258424

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Creates a new polygon from the midpoints of the sides of a given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Calculates the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- The main theorem -/
theorem sum_x_coordinates_q3 (q1 : Polygon) 
  (h1 : q1.vertices.length = 45)
  (h2 : sumXCoordinates q1 = 135) :
  let q2 := midpointPolygon q1
  let q3 := midpointPolygon q2
  sumXCoordinates q3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_q3_l2584_258424


namespace NUMINAMATH_CALUDE_magenta_opposite_cyan_l2584_258401

-- Define the colors
inductive Color
| Yellow
| Orange
| Blue
| Cyan
| Magenta
| Black

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the property of opposite faces
def opposite (c : Cube) (f1 f2 : Fin 6) : Prop :=
  (f1.val + f2.val) % 6 = 3

-- Define the given conditions
def cube_conditions (c : Cube) : Prop :=
  ∃ (top front right : Fin 6),
    c.faces top = Color.Cyan ∧
    c.faces right = Color.Blue ∧
    (c.faces front = Color.Yellow ∨ c.faces front = Color.Orange ∨ c.faces front = Color.Black)

-- Theorem statement
theorem magenta_opposite_cyan (c : Cube) :
  cube_conditions c →
  ∃ (magenta_face cyan_face : Fin 6),
    c.faces magenta_face = Color.Magenta ∧
    c.faces cyan_face = Color.Cyan ∧
    opposite c magenta_face cyan_face :=
by sorry

end NUMINAMATH_CALUDE_magenta_opposite_cyan_l2584_258401


namespace NUMINAMATH_CALUDE_range_of_m_l2584_258425

def p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ m - 2 > 0 ∧ 6 - m > 0 ∧
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (6 - m) = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m > 0

theorem range_of_m :
  ∃ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 < m ∧ m ≤ 2) ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2584_258425


namespace NUMINAMATH_CALUDE_not_divisible_by_1955_l2584_258431

theorem not_divisible_by_1955 : ∀ n : ℤ, ¬(1955 ∣ (n^2 + n + 1)) := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_1955_l2584_258431


namespace NUMINAMATH_CALUDE_percent_problem_l2584_258472

theorem percent_problem (x : ℝ) (h : 0.22 * x = 66) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l2584_258472


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2584_258409

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2, 4}

-- Define set N
def N : Set Nat := {2, 4, 5}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl M ∩ Set.compl N : Set Nat) = {3} :=
by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2584_258409


namespace NUMINAMATH_CALUDE_loop_structure_requirement_l2584_258456

/-- Represents a computational task that may or may not require a loop structure. -/
inductive ComputationalTask
  | SolveLinearSystem
  | CalculatePiecewiseFunction
  | CalculateFixedSum
  | FindSmallestNaturalNumber

/-- Determines if a given computational task requires a loop structure. -/
def requiresLoopStructure (task : ComputationalTask) : Prop :=
  match task with
  | ComputationalTask.FindSmallestNaturalNumber => true
  | _ => false

/-- The sum of natural numbers from 1 to n. -/
def sumUpTo (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- Theorem stating that finding the smallest natural number n such that 1+2+3+...+n > 100
    requires a loop structure, while other given tasks do not. -/
theorem loop_structure_requirement :
  (∀ n : ℕ, sumUpTo n ≤ 100 → sumUpTo (n + 1) > 100) →
  (requiresLoopStructure ComputationalTask.FindSmallestNaturalNumber ∧
   ¬requiresLoopStructure ComputationalTask.SolveLinearSystem ∧
   ¬requiresLoopStructure ComputationalTask.CalculatePiecewiseFunction ∧
   ¬requiresLoopStructure ComputationalTask.CalculateFixedSum) :=
by sorry


end NUMINAMATH_CALUDE_loop_structure_requirement_l2584_258456


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2584_258402

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2584_258402


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l2584_258445

theorem angle_sum_in_circle (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + x = 360) → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l2584_258445


namespace NUMINAMATH_CALUDE_prev_geng_yin_year_is_1950_l2584_258478

/-- The number of Heavenly Stems in the Ganzhi system -/
def heavenly_stems : ℕ := 10

/-- The number of Earthly Branches in the Ganzhi system -/
def earthly_branches : ℕ := 12

/-- The year we know to be a Geng-Yin year -/
def known_geng_yin_year : ℕ := 2010

/-- The function to calculate the previous Geng-Yin year -/
def prev_geng_yin_year (current_year : ℕ) : ℕ :=
  current_year - Nat.lcm heavenly_stems earthly_branches

theorem prev_geng_yin_year_is_1950 :
  prev_geng_yin_year known_geng_yin_year = 1950 := by
  sorry

#eval prev_geng_yin_year known_geng_yin_year

end NUMINAMATH_CALUDE_prev_geng_yin_year_is_1950_l2584_258478


namespace NUMINAMATH_CALUDE_scaling_factor_of_similar_cubes_l2584_258438

theorem scaling_factor_of_similar_cubes (v1 v2 : ℝ) (h1 : v1 = 343) (h2 : v2 = 2744) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_scaling_factor_of_similar_cubes_l2584_258438


namespace NUMINAMATH_CALUDE_smallest_perimeter_rectangle_l2584_258481

/-- A polygon made of unit squares -/
structure UnitSquarePolygon where
  area : ℕ

/-- The problem setup -/
def problem_setup (p1 p2 : UnitSquarePolygon) : Prop :=
  p1.area + p2.area = 16

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- A rectangle can contain the polygons if its area is at least the sum of the polygons' areas -/
def can_contain (length width : ℕ) (p1 p2 : UnitSquarePolygon) : Prop :=
  length * width ≥ p1.area + p2.area

/-- The theorem statement -/
theorem smallest_perimeter_rectangle (p1 p2 : UnitSquarePolygon) 
  (h : problem_setup p1 p2) :
  ∃ (length width : ℕ), 
    can_contain length width p1 p2 ∧ 
    (∀ (l w : ℕ), can_contain l w p1 p2 → rectangle_perimeter length width ≤ rectangle_perimeter l w) ∧
    rectangle_perimeter length width = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_rectangle_l2584_258481


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2584_258419

theorem triangle_sine_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : A < π)
  (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) (h7 : A + B + C = π) :
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 ∧
  (Real.sin A * Real.sin B * Real.sin C = 3 * Real.sqrt 3 / 8 ↔ A = π/3 ∧ B = π/3 ∧ C = π/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2584_258419


namespace NUMINAMATH_CALUDE_square_sum_of_differences_l2584_258466

theorem square_sum_of_differences (x y z : ℤ) : 
  ∃ (σ₂ : ℤ), (1/2 : ℚ) * ((x - y)^4 + (y - z)^4 + (z - x)^4) = (σ₂^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_differences_l2584_258466


namespace NUMINAMATH_CALUDE_trishaWalkDistance_l2584_258468

/-- The total distance Trisha walked during her vacation in New York City -/
def trishaWalk : ℝ :=
  let hotel_to_postcard : ℝ := 0.1111111111111111
  let postcard_to_tshirt : ℝ := 0.2222222222222222
  let tshirt_to_keychain : ℝ := 0.7777777777777778
  let keychain_to_toy : ℝ := 0.5555555555555556
  let toy_to_bookstore_meters : ℝ := 400
  let bookstore_to_hotel : ℝ := 0.6666666666666666
  let meters_to_miles : ℝ := 0.000621371

  hotel_to_postcard + postcard_to_tshirt + tshirt_to_keychain + keychain_to_toy +
  (toy_to_bookstore_meters * meters_to_miles) + bookstore_to_hotel

/-- Theorem stating that Trisha's total walk distance is approximately 1.5819 miles -/
theorem trishaWalkDistance : ∃ ε > 0, |trishaWalk - 1.5819| < ε :=
  sorry

end NUMINAMATH_CALUDE_trishaWalkDistance_l2584_258468


namespace NUMINAMATH_CALUDE_robert_reading_capacity_l2584_258433

/-- Represents the number of complete books that can be read given the reading speed, book length, and available time. -/
def booksRead (readingSpeed : ℕ) (bookLength : ℕ) (availableTime : ℕ) : ℕ :=
  (readingSpeed * availableTime) / bookLength

/-- Theorem stating that Robert can read 2 complete 360-page books in 8 hours at a speed of 120 pages per hour. -/
theorem robert_reading_capacity :
  booksRead 120 360 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_robert_reading_capacity_l2584_258433


namespace NUMINAMATH_CALUDE_square_diff_div_81_l2584_258464

theorem square_diff_div_81 : (2500 - 2409)^2 / 81 = 102 := by sorry

end NUMINAMATH_CALUDE_square_diff_div_81_l2584_258464


namespace NUMINAMATH_CALUDE_pet_ownership_percentages_l2584_258428

theorem pet_ownership_percentages (total_students : ℕ) (cat_owners : ℕ) (dog_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 75)
  (h3 : dog_owners = 125) :
  (cat_owners : ℚ) / total_students * 100 = 15 ∧
  (dog_owners : ℚ) / total_students * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_pet_ownership_percentages_l2584_258428


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l2584_258413

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l2584_258413


namespace NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2584_258412

theorem smallest_third_term_geometric_progression (d : ℝ) :
  (5 : ℝ) * (33 + 2 * d) = (8 + d) ^ 2 →
  ∃ (x : ℝ), x = 5 + 2 * d + 28 ∧
    ∀ (y : ℝ), (5 : ℝ) * (33 + 2 * y) = (8 + y) ^ 2 →
      5 + 2 * y + 28 ≥ x ∧
      x ≥ -21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_term_geometric_progression_l2584_258412


namespace NUMINAMATH_CALUDE_house_value_correct_l2584_258498

/-- Represents the inheritance distribution problem --/
structure InheritanceProblem where
  totalBrothers : Nat
  housesCount : Nat
  moneyPaidPerOlderBrother : Nat
  totalInheritance : Nat

/-- Calculates the value of one house given the inheritance problem --/
def houseValue (problem : InheritanceProblem) : Nat :=
  let olderBrothersCount := problem.housesCount
  let youngerBrothersCount := problem.totalBrothers - olderBrothersCount
  let totalMoneyPaid := olderBrothersCount * problem.moneyPaidPerOlderBrother
  let inheritancePerBrother := problem.totalInheritance / problem.totalBrothers
  (inheritancePerBrother * problem.totalBrothers - totalMoneyPaid) / problem.housesCount

/-- Theorem stating that the house value is correct for the given problem --/
theorem house_value_correct (problem : InheritanceProblem) :
  problem.totalBrothers = 5 →
  problem.housesCount = 3 →
  problem.moneyPaidPerOlderBrother = 2000 →
  problem.totalInheritance = 15000 →
  houseValue problem = 3000 := by
  sorry

#eval houseValue { totalBrothers := 5, housesCount := 3, moneyPaidPerOlderBrother := 2000, totalInheritance := 15000 }

end NUMINAMATH_CALUDE_house_value_correct_l2584_258498
