import Mathlib

namespace NUMINAMATH_CALUDE_centroid_division_weight_theorem_l2946_294691

/-- Represents a triangle with a given total weight -/
structure WeightedTriangle where
  totalWeight : ℝ
  weightProportionalToArea : Bool

/-- Represents a line passing through the centroid of a triangle -/
structure CentroidLine where
  triangle : WeightedTriangle

/-- Represents the two parts of a triangle divided by a centroid line -/
structure DividedTriangle where
  centroidLine : CentroidLine
  part1Weight : ℝ
  part2Weight : ℝ

/-- The theorem to be proved -/
theorem centroid_division_weight_theorem (t : WeightedTriangle) (l : CentroidLine) (d : DividedTriangle) :
  t.totalWeight = 900 ∧ t.weightProportionalToArea = true ∧ l.triangle = t ∧ d.centroidLine = l →
  d.part1Weight ≥ 400 ∧ d.part2Weight ≥ 400 :=
by sorry

end NUMINAMATH_CALUDE_centroid_division_weight_theorem_l2946_294691


namespace NUMINAMATH_CALUDE_assignment_schemes_with_girl_l2946_294636

theorem assignment_schemes_with_girl (num_boys num_girls : ℕ) 
  (h1 : num_boys = 4) 
  (h2 : num_girls = 3) 
  (total_people : ℕ := num_boys + num_girls) 
  (tasks : ℕ := 3) : 
  (total_people * (total_people - 1) * (total_people - 2)) - 
  (num_boys * (num_boys - 1) * (num_boys - 2)) = 186 := by
  sorry

#check assignment_schemes_with_girl

end NUMINAMATH_CALUDE_assignment_schemes_with_girl_l2946_294636


namespace NUMINAMATH_CALUDE_weight_of_BaBr2_l2946_294668

/-- The molecular weight of BaBr2 in grams per mole -/
def molecular_weight_BaBr2 : ℝ := 137.33 + 2 * 79.90

/-- The number of moles of BaBr2 -/
def moles_BaBr2 : ℝ := 8

/-- Calculates the total weight of a given number of moles of BaBr2 -/
def total_weight (mw : ℝ) (moles : ℝ) : ℝ := mw * moles

/-- Theorem stating that the total weight of 8 moles of BaBr2 is 2377.04 grams -/
theorem weight_of_BaBr2 : 
  total_weight molecular_weight_BaBr2 moles_BaBr2 = 2377.04 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaBr2_l2946_294668


namespace NUMINAMATH_CALUDE_order_silk_total_l2946_294638

/-- The total yards of silk dyed for an order, given the yards of green and pink silk. -/
def total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) : ℕ :=
  green_silk + pink_silk

/-- Theorem stating that the total yards of silk dyed for the order is 111421 yards. -/
theorem order_silk_total : 
  total_silk_dyed 61921 49500 = 111421 := by
  sorry

end NUMINAMATH_CALUDE_order_silk_total_l2946_294638


namespace NUMINAMATH_CALUDE_compound_mass_proof_l2946_294601

/-- The atomic mass of Carbon in g/mol -/
def atomic_mass_C : ℝ := 12.01

/-- The atomic mass of Hydrogen in g/mol -/
def atomic_mass_H : ℝ := 1.008

/-- The atomic mass of Oxygen in g/mol -/
def atomic_mass_O : ℝ := 16.00

/-- The atomic mass of Nitrogen in g/mol -/
def atomic_mass_N : ℝ := 14.01

/-- The atomic mass of Bromine in g/mol -/
def atomic_mass_Br : ℝ := 79.90

/-- The molecular formula of the compound -/
def compound_formula := "C8H10O2NBr2"

/-- The number of moles of the compound -/
def moles_compound : ℝ := 3

/-- The total mass of the compound in grams -/
def total_mass : ℝ := 938.91

/-- Theorem stating that the total mass of 3 moles of C8H10O2NBr2 is 938.91 grams -/
theorem compound_mass_proof :
  moles_compound * (8 * atomic_mass_C + 10 * atomic_mass_H + 2 * atomic_mass_O + atomic_mass_N + 2 * atomic_mass_Br) = total_mass := by
  sorry

end NUMINAMATH_CALUDE_compound_mass_proof_l2946_294601


namespace NUMINAMATH_CALUDE_brian_always_wins_l2946_294639

/-- Represents the game board -/
structure GameBoard :=
  (n : ℕ)

/-- Represents a player in the game -/
inductive Player := | Albus | Brian

/-- Represents a position on the game board -/
structure Position :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the state of the game -/
structure GameState :=
  (board : GameBoard)
  (position : Position)
  (current_player : Player)
  (move_distance : ℕ)

/-- Checks if a position is within the game board -/
def is_valid_position (board : GameBoard) (pos : Position) : Prop :=
  abs pos.x ≤ board.n ∧ abs pos.y ≤ board.n

/-- Defines the initial game state -/
def initial_state (n : ℕ) : GameState :=
  { board := { n := n },
    position := { x := 0, y := 0 },
    current_player := Player.Albus,
    move_distance := 1 }

/-- Theorem: Brian always has a winning strategy -/
theorem brian_always_wins (n : ℕ) :
  ∃ (strategy : GameState → Position),
    ∀ (game : GameState),
      game.current_player = Player.Brian →
      is_valid_position game.board (strategy game) →
      ¬is_valid_position game.board
        {x := 2 * game.position.x - (strategy game).x,
         y := 2 * game.position.y - (strategy game).y} :=
sorry

end NUMINAMATH_CALUDE_brian_always_wins_l2946_294639


namespace NUMINAMATH_CALUDE_store_profit_ratio_l2946_294677

/-- Represents the cost and sales information for a product. -/
structure Product where
  cost : ℝ
  markup : ℝ
  salesRatio : ℝ

/-- Represents the store's product lineup. -/
structure Store where
  peachSlices : Product
  riceCrispyTreats : Product
  sesameSnacks : Product

theorem store_profit_ratio (s : Store) : 
  s.peachSlices.cost = 2 * s.sesameSnacks.cost ∧
  s.peachSlices.markup = 0.2 ∧
  s.riceCrispyTreats.markup = 0.3 ∧
  s.sesameSnacks.markup = 0.2 ∧
  s.peachSlices.salesRatio = 1 ∧
  s.riceCrispyTreats.salesRatio = 3 ∧
  s.sesameSnacks.salesRatio = 2 ∧
  (s.peachSlices.markup * s.peachSlices.cost * s.peachSlices.salesRatio +
   s.riceCrispyTreats.markup * s.riceCrispyTreats.cost * s.riceCrispyTreats.salesRatio +
   s.sesameSnacks.markup * s.sesameSnacks.cost * s.sesameSnacks.salesRatio) = 
  0.25 * (s.peachSlices.cost * s.peachSlices.salesRatio +
          s.riceCrispyTreats.cost * s.riceCrispyTreats.salesRatio +
          s.sesameSnacks.cost * s.sesameSnacks.salesRatio) →
  s.riceCrispyTreats.cost / s.sesameSnacks.cost = 4 / 3 := by
sorry


end NUMINAMATH_CALUDE_store_profit_ratio_l2946_294677


namespace NUMINAMATH_CALUDE_max_ratio_APPE_l2946_294612

/-- A point in the Z × Z lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is inside a triangle -/
def isInside (P : LatticePoint) (T : LatticeTriangle) : Prop := sorry

/-- Checks if a point is the unique interior lattice point of a triangle -/
def isUniqueInteriorPoint (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  isInside P T ∧ ∀ Q : LatticePoint, isInside Q T → Q = P

/-- Intersection point of line AP and BC -/
def intersectionPoint (A P B C : LatticePoint) : LatticePoint := sorry

/-- Calculates the ratio AP/PE -/
def ratioAPPE (A P E : LatticePoint) : ℚ := sorry

/-- Main theorem: The maximum value of AP/PE is 5 -/
theorem max_ratio_APPE (T : LatticeTriangle) (P : LatticePoint) 
  (h : isUniqueInteriorPoint P T) :
  ∃ (M : ℚ), (∀ (A B C : LatticePoint),
    T = ⟨A, B, C⟩ → 
    let E := intersectionPoint A P B C
    ratioAPPE A P E ≤ M) ∧
  M = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_APPE_l2946_294612


namespace NUMINAMATH_CALUDE_pushkin_pension_is_survivor_l2946_294652

-- Define the types of pensions
inductive PensionType
| Retirement
| Disability
| Survivor

-- Define a structure for a pension
structure Pension where
  recipient : String
  year_assigned : Nat
  is_lifelong : Bool
  type : PensionType

-- Define Pushkin's family pension
def pushkin_family_pension : Pension :=
  { recipient := "Pushkin's wife and daughters"
  , year_assigned := 1837
  , is_lifelong := true
  , type := PensionType.Survivor }

-- Theorem statement
theorem pushkin_pension_is_survivor :
  pushkin_family_pension.type = PensionType.Survivor :=
by sorry

end NUMINAMATH_CALUDE_pushkin_pension_is_survivor_l2946_294652


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2946_294635

theorem min_value_of_expression :
  (∀ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 ≥ 2008) ∧
  (∃ x : ℝ, (x + 1) * (x + 3) * (x + 5) * (x + 7) + 2024 = 2008) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2946_294635


namespace NUMINAMATH_CALUDE_x_squared_is_quadratic_x_squared_eq_zero_is_quadratic_l2946_294678

/-- A quadratic equation in one variable is of the form ax² + bx + c = 0, where a, b, and c are constants, and a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation (λ x => x^2) := by
  sorry

/-- The equation x² = 0 is equivalent to the function f(x) = x² -/
theorem x_squared_eq_zero_is_quadratic : is_quadratic_equation (λ x => x^2 - 0) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_is_quadratic_x_squared_eq_zero_is_quadratic_l2946_294678


namespace NUMINAMATH_CALUDE_days_between_appointments_l2946_294629

/-- Represents the waiting periods for Mark's vaccine appointments -/
structure VaccineWaitingPeriod where
  totalWait : ℕ
  initialWait : ℕ
  finalWait : ℕ

/-- Theorem stating the number of days between first and second appointments -/
theorem days_between_appointments (mark : VaccineWaitingPeriod)
  (h1 : mark.totalWait = 38)
  (h2 : mark.initialWait = 4)
  (h3 : mark.finalWait = 14) :
  mark.totalWait - mark.initialWait - mark.finalWait = 20 := by
  sorry

#check days_between_appointments

end NUMINAMATH_CALUDE_days_between_appointments_l2946_294629


namespace NUMINAMATH_CALUDE_linear_system_solution_inequality_system_solution_l2946_294604

-- Part 1: System of linear equations
theorem linear_system_solution :
  let x : ℝ := 5
  let y : ℝ := 1
  (x - 5 * y = 0) ∧ (3 * x + 2 * y = 17) := by sorry

-- Part 2: System of inequalities
theorem inequality_system_solution :
  ∀ x : ℝ, x < -1/5 →
    (2 * (x - 2) ≤ 3 - x) ∧ (1 - (2 * x + 1) / 3 > x + 1) := by sorry

end NUMINAMATH_CALUDE_linear_system_solution_inequality_system_solution_l2946_294604


namespace NUMINAMATH_CALUDE_valentino_farm_birds_l2946_294632

/-- The number of birds on Mr. Valentino's farm -/
def total_birds (chickens ducks turkeys : ℕ) : ℕ := chickens + ducks + turkeys

/-- Theorem stating the total number of birds on Mr. Valentino's farm -/
theorem valentino_farm_birds :
  ∃ (chickens ducks turkeys : ℕ),
    chickens = 200 ∧
    ducks = 2 * chickens ∧
    turkeys = 3 * ducks ∧
    total_birds chickens ducks turkeys = 1800 :=
by sorry

end NUMINAMATH_CALUDE_valentino_farm_birds_l2946_294632


namespace NUMINAMATH_CALUDE_triangle_area_is_twelve_l2946_294655

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangular region -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_twelve :
  ∃ (x_intercept y_intercept : ℝ),
    lineEquation x_intercept 0 ∧
    lineEquation 0 y_intercept ∧
    triangleArea = (1 / 2) * x_intercept * y_intercept :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_is_twelve_l2946_294655


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2946_294670

-- Define a and b as real numbers
variable (a b : ℝ)

-- Theorem stating that the sum of a and b is equal to a + b
theorem sum_of_a_and_b : (a + b) = (a + b) := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2946_294670


namespace NUMINAMATH_CALUDE_sally_has_five_balloons_l2946_294650

/-- The number of blue balloons Sally has -/
def sallys_balloons (total joan jessica : ℕ) : ℕ :=
  total - joan - jessica

/-- Theorem stating that Sally has 5 blue balloons given the conditions -/
theorem sally_has_five_balloons :
  sallys_balloons 16 9 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sally_has_five_balloons_l2946_294650


namespace NUMINAMATH_CALUDE_ab_minus_a_minus_b_even_l2946_294606

def S : Set ℕ := {1, 3, 5, 7, 9}

theorem ab_minus_a_minus_b_even (a b : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  Even (a * b - a - b) :=
by
  sorry

end NUMINAMATH_CALUDE_ab_minus_a_minus_b_even_l2946_294606


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l2946_294666

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + Real.sqrt ((x - 4)^2 + (y - 3)^2) = 7

-- Define the foci
def F₁ : ℝ × ℝ := (0, 3)
def F₂ : ℝ × ℝ := (4, 0)

-- Theorem statement
theorem ellipse_x_intercept :
  ellipse 0 0 → -- The ellipse passes through (0,0)
  (∃ x : ℝ, x ≠ 0 ∧ ellipse x 0) → -- There exists another x-intercept
  (∃ x : ℝ, x = 56/11 ∧ ellipse x 0) -- The other x-intercept is (56/11, 0)
  := by sorry

end NUMINAMATH_CALUDE_ellipse_x_intercept_l2946_294666


namespace NUMINAMATH_CALUDE_find_number_l2946_294694

theorem find_number (x : ℝ) : (2 * x - 8 = -12) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2946_294694


namespace NUMINAMATH_CALUDE_school_boys_count_l2946_294627

theorem school_boys_count (total_girls : ℕ) (girl_boy_difference : ℕ) (boys : ℕ) : 
  total_girls = 697 →
  girl_boy_difference = 228 →
  total_girls = boys + girl_boy_difference →
  boys = 469 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l2946_294627


namespace NUMINAMATH_CALUDE_union_of_sets_l2946_294628

def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) :
  A a ∩ B a b = {1/4} →
  A a ∪ B a b = {-2, 1, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2946_294628


namespace NUMINAMATH_CALUDE_school_students_l2946_294659

/-- The number of students in a school -/
theorem school_students (boys : ℕ) (girls : ℕ) : 
  boys = 272 → girls = boys + 106 → boys + girls = 650 := by
  sorry

end NUMINAMATH_CALUDE_school_students_l2946_294659


namespace NUMINAMATH_CALUDE_proposition_relation_necessary_not_sufficient_l2946_294667

theorem proposition_relation (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

theorem necessary_not_sufficient :
  (∃ a : ℝ, (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ∧ (a ≤ 0 ∨ a ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relation_necessary_not_sufficient_l2946_294667


namespace NUMINAMATH_CALUDE_green_ball_count_l2946_294675

/-- Given a box of balls where the ratio of blue to green balls is 5:3 and there are 15 blue balls,
    prove that the number of green balls is 9. -/
theorem green_ball_count (blue : ℕ) (green : ℕ) (h1 : blue = 15) (h2 : blue * 3 = green * 5) : green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_count_l2946_294675


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_a_l2946_294634

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if the asymptote equations are 3x ± 2y = 0, then a = 2 -/
theorem hyperbola_asymptote_a (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1 →
    (3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0)) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_a_l2946_294634


namespace NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l2946_294680

/-- Represents the car race scenario between Karen and Tom -/
def CarRace (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) : Prop :=
  let race_time := (karen_delay * karen_speed + winning_margin) / (karen_speed - tom_speed)
  tom_speed * race_time = 24

/-- Theorem stating the distance Tom drives before Karen wins -/
theorem tom_distance_before_karen_wins :
  CarRace 60 45 (4/60) 4 :=
by sorry

end NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l2946_294680


namespace NUMINAMATH_CALUDE_max_value_of_f_l2946_294617

noncomputable def f (x : ℝ) : ℝ := x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x ^ 3)

theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 18 ∧
  f x = 2 * Real.sqrt 17 ∧
  ∀ y ∈ Set.Icc 0 18, f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2946_294617


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l2946_294618

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define evenness for a function
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define symmetry about a vertical line
def SymmetricAbout (g : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, g (a + x) = g (a - x)

-- Proposition ②
theorem prop_2 (h : IsEven (fun x ↦ f (x + 2))) : SymmetricAbout f 2 := by sorry

-- Proposition ④
theorem prop_4 : SymmetricAbout (fun x ↦ f (x - 2)) 2 ∧ SymmetricAbout (fun x ↦ f (2 - x)) 2 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l2946_294618


namespace NUMINAMATH_CALUDE_marigold_sale_ratio_l2946_294614

/-- Proves that the ratio of marigolds sold on the third day to the second day is 2:1 --/
theorem marigold_sale_ratio :
  ∀ (day3 : ℕ),
  14 + 25 + day3 = 89 →
  (day3 : ℚ) / 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_marigold_sale_ratio_l2946_294614


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2946_294690

/-- A geometric sequence with sum of first n terms S_n -/
def GeometricSequence (S : ℕ → ℝ) : Prop :=
  ∃ (a r : ℝ), ∀ n : ℕ, S n = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → ℝ) :
  GeometricSequence S →
  S 5 = 10 →
  S 10 = 50 →
  S 15 = 210 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2946_294690


namespace NUMINAMATH_CALUDE_circle_and_m_value_l2946_294641

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y m : ℝ) : Prop := x + y + m = 0

-- Define perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem circle_and_m_value :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (curve 0 3 ∧ curve 1 0 ∧ curve 3 0) ∧  -- Intersection points with axes
    (circle_C 0 3 ∧ circle_C 1 0 ∧ circle_C 3 0) ∧  -- These points lie on circle C
    (∃ m : ℝ, 
      line x₁ y₁ m ∧ line x₂ y₂ m ∧  -- A and B lie on the line
      circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧  -- A and B lie on circle C
      perpendicular x₁ y₁ x₂ y₂ ∧  -- OA is perpendicular to OB
      (m = -1 ∨ m = -3))  -- The value of m
  :=
sorry

end NUMINAMATH_CALUDE_circle_and_m_value_l2946_294641


namespace NUMINAMATH_CALUDE_badge_exchange_problem_l2946_294631

theorem badge_exchange_problem (V T : ℕ) :
  V = T + 5 →
  (V - V * 24 / 100 + T * 20 / 100) = (T - T * 20 / 100 + V * 24 / 100 - 1) →
  V = 50 ∧ T = 45 := by
sorry

end NUMINAMATH_CALUDE_badge_exchange_problem_l2946_294631


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2946_294643

/-- Given a complex number z satisfying (z - 3i)(2 + i) = 5i, prove that z = 2 + 5i -/
theorem complex_equation_solution (z : ℂ) (h : (z - 3*Complex.I)*(2 + Complex.I) = 5*Complex.I) : 
  z = 2 + 5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2946_294643


namespace NUMINAMATH_CALUDE_chessboard_selection_divisibility_l2946_294616

theorem chessboard_selection_divisibility (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) :
  ∃ k : ℕ, p! - p = p^5 * k := by
  sorry

end NUMINAMATH_CALUDE_chessboard_selection_divisibility_l2946_294616


namespace NUMINAMATH_CALUDE_program_size_calculation_l2946_294699

/-- Calculates the size of a downloaded program given the download speed and time -/
theorem program_size_calculation (download_speed : ℝ) (download_time : ℝ) : 
  download_speed = 50 → download_time = 2 → 
  download_speed * download_time * 60 * 60 / 1024 = 351.5625 := by
  sorry

#check program_size_calculation

end NUMINAMATH_CALUDE_program_size_calculation_l2946_294699


namespace NUMINAMATH_CALUDE_range_of_f_l2946_294625

noncomputable def f (x : ℝ) : ℝ := 3 * (x + 5) * (x - 4) / (x + 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2946_294625


namespace NUMINAMATH_CALUDE_range_of_a_l2946_294649

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1/2 then (1/2)^(x - 1/2) else Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Icc (Real.sqrt 2 / 2) 1 ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2946_294649


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2946_294692

theorem polynomial_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2946_294692


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2946_294673

theorem least_positive_integer_with_remainders : ∃! N : ℕ,
  N > 0 ∧
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  ∀ M : ℕ, (M > 0 ∧ M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6) → N ≤ M :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2946_294673


namespace NUMINAMATH_CALUDE_melanie_plum_count_l2946_294624

/-- The number of plums Melanie picked -/
def melanie_picked : ℝ := 7.0

/-- The number of plums Sam gave to Melanie -/
def sam_gave : ℝ := 3.0

/-- The total number of plums Melanie has now -/
def total_plums : ℝ := melanie_picked + sam_gave

theorem melanie_plum_count : total_plums = 10.0 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plum_count_l2946_294624


namespace NUMINAMATH_CALUDE_square_field_area_l2946_294637

/-- The area of a square field with side length 17 meters is 289 square meters. -/
theorem square_field_area :
  ∀ (side_length area : ℝ),
  side_length = 17 →
  area = side_length * side_length →
  area = 289 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l2946_294637


namespace NUMINAMATH_CALUDE_number_puzzle_l2946_294658

theorem number_puzzle : ∃! x : ℝ, 0.8 * x + 20 = x := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l2946_294658


namespace NUMINAMATH_CALUDE_interior_angle_sum_for_polygon_with_60_degree_exterior_angles_l2946_294671

theorem interior_angle_sum_for_polygon_with_60_degree_exterior_angles :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 60 →
    n * exterior_angle = 360 →
    (n - 2) * 180 = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_for_polygon_with_60_degree_exterior_angles_l2946_294671


namespace NUMINAMATH_CALUDE_original_acid_percentage_l2946_294620

theorem original_acid_percentage (x y : ℝ) :
  (y / (x + y + 1) = 1 / 5) →
  ((y + 1) / (x + y + 2) = 1 / 3) →
  (y / (x + y) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_original_acid_percentage_l2946_294620


namespace NUMINAMATH_CALUDE_ball_max_height_l2946_294679

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

/-- The maximum height reached by the ball -/
def max_height : ℝ := 81.25

theorem ball_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ h t :=
by sorry

end NUMINAMATH_CALUDE_ball_max_height_l2946_294679


namespace NUMINAMATH_CALUDE_min_value_theorem_l2946_294676

theorem min_value_theorem (a k b m n : ℝ) : 
  a > 0 → a ≠ 1 → 
  (∀ x, a^(x-1) + 1 = b → x = k) → 
  m + n = b - k → m > 0 → n > 0 → 
  (9/m + 1/n ≥ 16 ∧ ∃ m n, 9/m + 1/n = 16) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2946_294676


namespace NUMINAMATH_CALUDE_nails_per_plank_l2946_294665

theorem nails_per_plank (total_nails : ℕ) (total_planks : ℕ) (h1 : total_nails = 4) (h2 : total_planks = 2) :
  total_nails / total_planks = 2 := by
sorry

end NUMINAMATH_CALUDE_nails_per_plank_l2946_294665


namespace NUMINAMATH_CALUDE_largest_difference_in_grid_l2946_294693

/-- A type representing a 20x20 grid of integers -/
def Grid := Fin 20 → Fin 20 → Fin 400

/-- The property that a grid contains all integers from 1 to 400 -/
def contains_all_integers (g : Grid) : Prop :=
  ∀ n : Fin 400, ∃ i j : Fin 20, g i j = n

/-- The property that there exist two numbers in the same row or column with a difference of at least N -/
def has_difference_at_least (g : Grid) (N : ℕ) : Prop :=
  ∃ i j k : Fin 20, (g i j).val + N ≤ (g i k).val ∨ (g j i).val + N ≤ (g k i).val

/-- The main theorem: 209 is the largest N satisfying the condition -/
theorem largest_difference_in_grid :
  (∀ g : Grid, contains_all_integers g → has_difference_at_least g 209) ∧
  ¬(∀ g : Grid, contains_all_integers g → has_difference_at_least g 210) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_in_grid_l2946_294693


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l2946_294683

-- Statement B
theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x > 1 ∧ y > 1 → x + y > 2) ∧
  ¬(x + y > 2 → x > 1 ∧ y > 1) :=
sorry

-- Statement C
theorem necessary_not_sufficient_condition (a b : ℝ) :
  (a > b ∧ b > 0 → 1 / a < 1 / b) ∧
  ¬(1 / a < 1 / b → a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l2946_294683


namespace NUMINAMATH_CALUDE_darias_remaining_debt_l2946_294688

def savings : ℕ := 500
def couch_price : ℕ := 750
def table_price : ℕ := 100
def lamp_price : ℕ := 50

theorem darias_remaining_debt : 
  (couch_price + table_price + lamp_price) - savings = 400 := by
  sorry

end NUMINAMATH_CALUDE_darias_remaining_debt_l2946_294688


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2946_294672

theorem fraction_equivalence : 
  ∀ (n : ℚ), (3 + n) / (5 + n) = 5 / 6 → n = 7 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2946_294672


namespace NUMINAMATH_CALUDE_metaPopulation2050_l2946_294630

-- Define the initial population and year
def initialPopulation : ℕ := 150
def initialYear : ℕ := 2005

-- Define the doubling period and target year
def doublingPeriod : ℕ := 20
def targetYear : ℕ := 2050

-- Define the population growth function
def populationGrowth (years : ℕ) : ℕ :=
  initialPopulation * (2 ^ (years / doublingPeriod))

-- Theorem statement
theorem metaPopulation2050 :
  populationGrowth (targetYear - initialYear) = 600 := by
  sorry

end NUMINAMATH_CALUDE_metaPopulation2050_l2946_294630


namespace NUMINAMATH_CALUDE_ball_probability_l2946_294633

/-- Given a bag of 100 balls with specific color distributions, 
    prove that the probability of choosing a ball that is neither red nor purple is 0.8 -/
theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_yellow : yellow = 10)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.8 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l2946_294633


namespace NUMINAMATH_CALUDE_weight_of_scaled_object_l2946_294619

/-- Given two similar three-dimensional objects where one has all dimensions 3 times
    larger than the other, if the smaller object weighs 10 grams, 
    then the larger object weighs 270 grams. -/
theorem weight_of_scaled_object (weight_small : ℝ) (scale_factor : ℝ) :
  weight_small = 10 →
  scale_factor = 3 →
  weight_small * scale_factor^3 = 270 := by
sorry


end NUMINAMATH_CALUDE_weight_of_scaled_object_l2946_294619


namespace NUMINAMATH_CALUDE_quadratic_coefficient_for_specific_parabola_l2946_294657

/-- A quadratic function with vertex (h, k) passing through point (x₀, y₀) has coefficient a -/
def quadratic_coefficient (h k x₀ y₀ : ℚ) : ℚ :=
  (y₀ - k) / ((x₀ - h)^2)

theorem quadratic_coefficient_for_specific_parabola :
  quadratic_coefficient 2 (-3) 6 (-63) = -15/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_for_specific_parabola_l2946_294657


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2946_294621

theorem negative_fraction_comparison : -4/5 < -2/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2946_294621


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2946_294615

/-- Represents the value in billion yuan -/
def original_value : ℝ := 8450

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 8.45

/-- Represents the exponent in scientific notation -/
def exponent : ℤ := 3

/-- Theorem stating that the original value is equal to its scientific notation representation -/
theorem scientific_notation_equivalence :
  original_value = coefficient * (10 : ℝ) ^ exponent :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2946_294615


namespace NUMINAMATH_CALUDE_collinear_vectors_dot_product_l2946_294647

/-- Given two collinear vectors m and n, prove their dot product is -17/2 -/
theorem collinear_vectors_dot_product :
  ∀ (k : ℝ),
  let m : ℝ × ℝ := (2*k - 1, k)
  let n : ℝ × ℝ := (4, 1)
  (∃ (t : ℝ), m = t • n) →  -- collinearity condition
  m.1 * n.1 + m.2 * n.2 = -17/2 :=
by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_dot_product_l2946_294647


namespace NUMINAMATH_CALUDE_smallest_value_satisfying_equation_l2946_294661

theorem smallest_value_satisfying_equation :
  ∃ (x : ℝ), x = 3 ∧ ∀ (y : ℝ), (⌊y⌋ = 3 + 50 * (y - ⌊y⌋)) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_satisfying_equation_l2946_294661


namespace NUMINAMATH_CALUDE_perimeter_is_72_l2946_294609

/-- A geometric figure formed by six identical squares arranged into a larger rectangle,
    with two smaller identical squares placed inside. -/
structure GeometricFigure where
  /-- The side length of each of the six identical squares forming the larger rectangle -/
  side_length : ℝ
  /-- The total area of the figure -/
  total_area : ℝ
  /-- The area of the figure is equal to the area of six squares -/
  area_eq : total_area = 6 * side_length^2

/-- The perimeter of the geometric figure -/
def perimeter (fig : GeometricFigure) : ℝ :=
  2 * (3 * fig.side_length + 2 * fig.side_length) + 2 * fig.side_length

theorem perimeter_is_72 (fig : GeometricFigure) (h : fig.total_area = 216) :
  perimeter fig = 72 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_72_l2946_294609


namespace NUMINAMATH_CALUDE_school_demographics_l2946_294602

theorem school_demographics (total_students : ℕ) (avg_age_boys avg_age_girls avg_age_school : ℚ) : 
  total_students = 640 →
  avg_age_boys = 12 →
  avg_age_girls = 11 →
  avg_age_school = 47/4 →
  ∃ (num_girls : ℕ), num_girls = 160 ∧ 
    (total_students - num_girls) * avg_age_boys + num_girls * avg_age_girls = total_students * avg_age_school :=
by sorry

end NUMINAMATH_CALUDE_school_demographics_l2946_294602


namespace NUMINAMATH_CALUDE_money_remaining_l2946_294663

/-- Given an initial amount of money and an amount spent, 
    the remaining amount is the difference between the two. -/
theorem money_remaining (initial spent : ℕ) : 
  initial = 16 → spent = 8 → initial - spent = 8 := by
  sorry

end NUMINAMATH_CALUDE_money_remaining_l2946_294663


namespace NUMINAMATH_CALUDE_lcm_14_21_45_l2946_294685

theorem lcm_14_21_45 : Nat.lcm 14 (Nat.lcm 21 45) = 630 := by sorry

end NUMINAMATH_CALUDE_lcm_14_21_45_l2946_294685


namespace NUMINAMATH_CALUDE_rectangular_field_area_increase_l2946_294603

theorem rectangular_field_area_increase 
  (original_length : ℝ) 
  (original_width : ℝ) 
  (length_increase : ℝ) : 
  original_length = 20 →
  original_width = 5 →
  length_increase = 10 →
  (original_length + length_increase) * original_width - original_length * original_width = 50 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_increase_l2946_294603


namespace NUMINAMATH_CALUDE_henrys_classical_cds_l2946_294613

/-- Given Henry's CD collection, prove the number of classical CDs --/
theorem henrys_classical_cds :
  ∀ (country rock classical : ℕ),
    country = 23 →
    country = rock + 3 →
    rock = 2 * classical →
    classical = 10 := by
  sorry

end NUMINAMATH_CALUDE_henrys_classical_cds_l2946_294613


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l2946_294642

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, -3 < (x^2 + a*x - 2) / (x^2 - x + 1) ∧ (x^2 + a*x - 2) / (x^2 - x + 1) < 2) ↔ 
  (-1 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l2946_294642


namespace NUMINAMATH_CALUDE_x_twelfth_power_is_one_l2946_294698

theorem x_twelfth_power_is_one (x : ℂ) (h : x + 1/x = -1) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_is_one_l2946_294698


namespace NUMINAMATH_CALUDE_debate_club_green_teams_l2946_294626

theorem debate_club_green_teams 
  (total_members : ℕ) 
  (red_members : ℕ) 
  (green_members : ℕ) 
  (total_teams : ℕ) 
  (red_red_teams : ℕ) : 
  total_members = 132 → 
  red_members = 48 → 
  green_members = 84 → 
  total_teams = 66 → 
  red_red_teams = 15 → 
  ∃ (green_green_teams : ℕ), green_green_teams = 33 ∧ 
    green_green_teams = (green_members - (total_members - 2 * red_red_teams - red_members)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_debate_club_green_teams_l2946_294626


namespace NUMINAMATH_CALUDE_component_unqualified_l2946_294660

/-- Determines if a component is qualified based on its diameter -/
def is_qualified (measured_diameter : ℝ) (specified_diameter : ℝ) (tolerance : ℝ) : Prop :=
  measured_diameter ≥ specified_diameter - tolerance ∧ 
  measured_diameter ≤ specified_diameter + tolerance

/-- Theorem stating that the component with measured diameter 19.9 mm is unqualified -/
theorem component_unqualified : 
  ¬ is_qualified 19.9 20 0.02 := by
  sorry

end NUMINAMATH_CALUDE_component_unqualified_l2946_294660


namespace NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_ellipse_l2946_294607

theorem largest_y_coordinate_degenerate_ellipse :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ (x^2 / 49) + ((y - 3)^2 / 25)
  ∀ (x y : ℝ), f (x, y) = 0 → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_coordinate_degenerate_ellipse_l2946_294607


namespace NUMINAMATH_CALUDE_sufficient_condition_for_vector_norm_equality_l2946_294648

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- For non-zero vectors a and b, if a + 2b = 0, then |a - b| = |a| + |b| -/
theorem sufficient_condition_for_vector_norm_equality 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : a + 2 • b = 0) : 
  ‖a - b‖ = ‖a‖ + ‖b‖ := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_vector_norm_equality_l2946_294648


namespace NUMINAMATH_CALUDE_line_parameterization_l2946_294611

def is_valid_parameterization (x₀ y₀ u v : ℝ) : Prop :=
  y₀ = 3 * x₀ - 5 ∧ ∃ (k : ℝ), u = k * 1 ∧ v = k * 3

theorem line_parameterization 
  (x₀ y₀ u v : ℝ) : 
  is_valid_parameterization x₀ y₀ u v ↔ 
  (∀ t : ℝ, (3 * (x₀ + t * u) - 5 = y₀ + t * v)) :=
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2946_294611


namespace NUMINAMATH_CALUDE_bet_winnings_l2946_294656

theorem bet_winnings (initial_amount : ℚ) : 
  initial_amount > 0 →
  initial_amount + 2 * initial_amount = 1200 →
  initial_amount = 400 := by
sorry

end NUMINAMATH_CALUDE_bet_winnings_l2946_294656


namespace NUMINAMATH_CALUDE_cubic_function_property_l2946_294682

/-- Given a cubic function f(x) = ax³ + bx + 1, prove that if f(m) = 6, then f(-m) = -4 -/
theorem cubic_function_property (a b m : ℝ) : 
  (fun x => a * x^3 + b * x + 1) m = 6 →
  (fun x => a * x^3 + b * x + 1) (-m) = -4 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2946_294682


namespace NUMINAMATH_CALUDE_count_pairs_eq_five_l2946_294687

/-- The number of pairs of natural numbers (a, b) satisfying the given conditions -/
def count_pairs : ℕ := 5

/-- Predicate to check if a pair of natural numbers satisfies the equation -/
def satisfies_equation (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 6

/-- The main theorem stating that there are exactly 5 pairs satisfying the conditions -/
theorem count_pairs_eq_five :
  (∃! (s : Finset (ℕ × ℕ)), s.card = count_pairs ∧ 
    (∀ (p : ℕ × ℕ), p ∈ s ↔ 
      (p.1 ≥ p.2 ∧ satisfies_equation p.1 p.2))) :=
by sorry

end NUMINAMATH_CALUDE_count_pairs_eq_five_l2946_294687


namespace NUMINAMATH_CALUDE_starting_lineups_count_l2946_294686

def total_players : ℕ := 15
def lineup_size : ℕ := 6
def injured_players : ℕ := 1
def incompatible_players : ℕ := 2

theorem starting_lineups_count :
  (Nat.choose (total_players - incompatible_players - injured_players + 1) (lineup_size - 1)) * 2 +
  (Nat.choose (total_players - incompatible_players - injured_players) lineup_size) = 3498 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineups_count_l2946_294686


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l2946_294695

/-- The length of a bridge given train characteristics and crossing time -/
theorem bridge_length_calculation 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 120)
  (h2 : crossing_time = 26.997840172786177)
  (h3 : train_speed_kmph = 36) :
  ∃ bridge_length : ℝ, 
    bridge_length = 149.97840172786177 ∧ 
    bridge_length = (train_speed_kmph * 1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l2946_294695


namespace NUMINAMATH_CALUDE_givenVectorIsDirectionVector_l2946_294689

/-- A line in 2D space --/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space --/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The line x-3y+1=0 --/
def givenLine : Line2D :=
  { a := 1, b := -3, c := 1 }

/-- The vector (3,1) --/
def givenVector : Vector2D :=
  { x := 3, y := 1 }

/-- Definition: A vector is a direction vector of a line if it's parallel to the line --/
def isDirectionVector (v : Vector2D) (l : Line2D) : Prop :=
  v.x * l.b = -v.y * l.a

/-- Theorem: The vector (3,1) is a direction vector of the line x-3y+1=0 --/
theorem givenVectorIsDirectionVector : isDirectionVector givenVector givenLine := by
  sorry

end NUMINAMATH_CALUDE_givenVectorIsDirectionVector_l2946_294689


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2946_294684

/-- 
For a quadratic equation x^2 - mx - 1 = 0 to have two roots, 
one greater than 2 and the other less than 2, m must be in the range (3/2, +∞).
-/
theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x < 2 ∧ y > 2 ∧ x^2 - m*x - 1 = 0 ∧ y^2 - m*y - 1 = 0) ↔ 
  m > 3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2946_294684


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2946_294674

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2946_294674


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l2946_294662

theorem unique_positive_integer_solution : 
  ∃! (x : ℕ), (4 * x)^2 - 2 * x = 8062 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l2946_294662


namespace NUMINAMATH_CALUDE_inequality_range_l2946_294640

theorem inequality_range (a : ℝ) : 
  (∀ x > 1, x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2946_294640


namespace NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l2946_294696

-- Define a function to create the six-digit number XAXAXA
def makeNumber (X A : Nat) : Nat :=
  100000 * X + 10000 * A + 1000 * X + 100 * A + 10 * X + A

-- Theorem statement
theorem xaxaxa_divisible_by_seven (X A : Nat) (h1 : X < 10) (h2 : A < 10) :
  (makeNumber X A) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_xaxaxa_divisible_by_seven_l2946_294696


namespace NUMINAMATH_CALUDE_max_value_of_f_l2946_294646

/-- The quadratic function f(x) = -2x^2 + 4x - 6 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x - 6

/-- The maximum value of f(x) is -4 -/
theorem max_value_of_f :
  ∃ (m : ℝ), m = -4 ∧ ∀ (x : ℝ), f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2946_294646


namespace NUMINAMATH_CALUDE_cosine_value_l2946_294669

theorem cosine_value (α : Real) 
  (h1 : 0 < α) (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - π) = -1/6 := by
sorry

end NUMINAMATH_CALUDE_cosine_value_l2946_294669


namespace NUMINAMATH_CALUDE_white_tulips_multiple_of_seven_l2946_294697

/-- The number of red tulips -/
def red_tulips : ℕ := 91

/-- The number of identical bouquets that can be made -/
def num_bouquets : ℕ := 7

/-- The number of white tulips -/
def white_tulips : ℕ := sorry

/-- Proposition stating that the number of white tulips is a multiple of 7 -/
theorem white_tulips_multiple_of_seven :
  ∃ k : ℕ, white_tulips = 7 * k ∧ red_tulips % num_bouquets = 0 :=
sorry

end NUMINAMATH_CALUDE_white_tulips_multiple_of_seven_l2946_294697


namespace NUMINAMATH_CALUDE_square_pyramid_sum_l2946_294608

/-- A square pyramid is a polyhedron with a square base and triangular faces meeting at an apex. -/
structure SquarePyramid where
  -- We don't need to define the internal structure, just the concept

/-- The number of faces in a square pyramid -/
def num_faces (sp : SquarePyramid) : ℕ := 5

/-- The number of edges in a square pyramid -/
def num_edges (sp : SquarePyramid) : ℕ := 8

/-- The number of vertices in a square pyramid -/
def num_vertices (sp : SquarePyramid) : ℕ := 5

/-- The sum of faces, edges, and vertices in a square pyramid is 18 -/
theorem square_pyramid_sum (sp : SquarePyramid) : 
  num_faces sp + num_edges sp + num_vertices sp = 18 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_sum_l2946_294608


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l2946_294610

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_arithmetic_sequence (n : ℕ) :
  n > 0 ∧ 
  n + (n + 3) + (n + 4) = 3000 ∧ 
  (fib n < fib (n + 3) ∧ fib (n + 3) < fib (n + 4)) ∧
  (fib (n + 4) - fib (n + 3) = fib (n + 3) - fib n) →
  n = 997 := by
sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l2946_294610


namespace NUMINAMATH_CALUDE_power_of_product_l2946_294681

theorem power_of_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2946_294681


namespace NUMINAMATH_CALUDE_billboard_area_l2946_294653

/-- The area of a rectangular billboard with perimeter 46 feet and width 8 feet is 120 square feet. -/
theorem billboard_area (perimeter width : ℝ) (h1 : perimeter = 46) (h2 : width = 8) :
  let length := (perimeter - 2 * width) / 2
  width * length = 120 :=
by sorry

end NUMINAMATH_CALUDE_billboard_area_l2946_294653


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2946_294605

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.4)
  let final_salary := decreased_salary * (1 + 0.4)
  (initial_salary - final_salary) / initial_salary = 0.16 := by
sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2946_294605


namespace NUMINAMATH_CALUDE_trader_profit_percentage_l2946_294664

theorem trader_profit_percentage (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let discount_rate : ℝ := 0.20
  let purchase_price : ℝ := original_price * (1 - discount_rate)
  let markup_rate : ℝ := 0.60
  let selling_price : ℝ := purchase_price * (1 + markup_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 28 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_percentage_l2946_294664


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l2946_294654

theorem square_sum_equals_one (a b : ℝ) (h : a + b = -1) : a^2 + b^2 + 2*a*b = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l2946_294654


namespace NUMINAMATH_CALUDE_simplify_expression_l2946_294651

theorem simplify_expression (x : ℝ) : 105*x - 57*x + 8 = 48*x + 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2946_294651


namespace NUMINAMATH_CALUDE_parabola_through_point_l2946_294644

theorem parabola_through_point (a b c : ℤ) : 
  5 = a * 2^2 + b * 2 + c → 2 * a + b - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l2946_294644


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2946_294623

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l2946_294623


namespace NUMINAMATH_CALUDE_cube_coloring_theorem_l2946_294645

/-- Represents the symmetry group of a cube -/
def CubeSymmetryGroup : Type := Unit

/-- The order of the cube symmetry group -/
def symmetryGroupOrder : ℕ := 24

/-- The total number of ways to color a cube with 6 colors without considering rotations -/
def totalColorings : ℕ := 720

/-- The number of distinct colorings of a cube with 6 colors, considering rotational symmetries -/
def distinctColorings : ℕ := totalColorings / symmetryGroupOrder

theorem cube_coloring_theorem :
  distinctColorings = 30 :=
sorry

end NUMINAMATH_CALUDE_cube_coloring_theorem_l2946_294645


namespace NUMINAMATH_CALUDE_sum_of_differences_l2946_294600

def T : Finset ℕ := Finset.range 9

def M : ℕ := Finset.sum T (λ x => Finset.sum T (λ y => if x > y then 3^x - 3^y else 0))

theorem sum_of_differences (T : Finset ℕ) (M : ℕ) :
  T = Finset.range 9 →
  M = Finset.sum T (λ x => Finset.sum T (λ y => if x > y then 3^x - 3^y else 0)) →
  M = 68896 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_differences_l2946_294600


namespace NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_value_qin_jiushao_f_3_l2946_294622

-- Define the polynomial coefficients
def a₀ : ℝ := -0.8
def a₁ : ℝ := 1.7
def a₂ : ℝ := -2.6
def a₃ : ℝ := 3.5
def a₄ : ℝ := 2
def a₅ : ℝ := 4

-- Define Qin Jiushao's algorithm
def qin_jiushao (x : ℝ) : ℝ :=
  let v₀ := a₅
  let v₁ := v₀ * x + a₄
  let v₂ := v₁ * x + a₃
  let v₃ := v₂ * x + a₂
  let v₄ := v₃ * x + a₁
  v₄ * x + a₀

-- Define the polynomial function
def f (x : ℝ) : ℝ := a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

-- Theorem stating that Qin Jiushao's algorithm gives the correct result for f(3)
theorem qin_jiushao_correct : qin_jiushao 3 = f 3 := by sorry

-- Theorem stating that f(3) equals 1209.4
theorem f_3_value : f 3 = 1209.4 := by sorry

-- Main theorem combining the above results
theorem qin_jiushao_f_3 : qin_jiushao 3 = 1209.4 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_correct_f_3_value_qin_jiushao_f_3_l2946_294622
