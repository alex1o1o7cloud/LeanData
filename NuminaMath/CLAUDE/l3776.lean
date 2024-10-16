import Mathlib

namespace NUMINAMATH_CALUDE_proposition_equivalence_l3776_377654

theorem proposition_equivalence (p q : Prop) :
  (¬p → ¬q) ↔ (p → q) := by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l3776_377654


namespace NUMINAMATH_CALUDE_ramanujan_number_proof_l3776_377696

def hardy_number : ℂ := 4 + 6 * Complex.I

theorem ramanujan_number_proof (product : ℂ) (h : product = 40 - 24 * Complex.I) :
  ∃ (ramanujan_number : ℂ), 
    ramanujan_number * hardy_number = product ∧ 
    ramanujan_number = 76/13 - 36/13 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_proof_l3776_377696


namespace NUMINAMATH_CALUDE_three_heads_one_tail_probability_three_heads_one_tail_probability_proof_l3776_377644

/-- The probability of getting exactly three heads and one tail when four fair coins are tossed simultaneously -/
theorem three_heads_one_tail_probability : ℝ :=
  1 / 4

/-- Proof that the probability of getting exactly three heads and one tail when four fair coins are tossed simultaneously is 1/4 -/
theorem three_heads_one_tail_probability_proof :
  three_heads_one_tail_probability = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_one_tail_probability_three_heads_one_tail_probability_proof_l3776_377644


namespace NUMINAMATH_CALUDE_inverse_proportion_quadrants_l3776_377671

/-- An inverse proportion function passing through a specific point -/
structure InverseProportionFunction where
  k : ℝ
  a : ℝ
  point_condition : k / (3 * a) = a

/-- The quadrants where the graph of an inverse proportion function lies -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- The set of quadrants where the graph lies -/
def graph_quadrants (f : InverseProportionFunction) : Set Quadrant :=
  {Quadrant.I, Quadrant.III}

/-- Theorem: The graph of the inverse proportion function lies in Quadrants I and III -/
theorem inverse_proportion_quadrants (f : InverseProportionFunction) :
  graph_quadrants f = {Quadrant.I, Quadrant.III} := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_quadrants_l3776_377671


namespace NUMINAMATH_CALUDE_cookies_in_bags_l3776_377660

def total_cookies : ℕ := 75
def cookies_per_bag : ℕ := 3

theorem cookies_in_bags : total_cookies / cookies_per_bag = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_bags_l3776_377660


namespace NUMINAMATH_CALUDE_smallest_possible_students_l3776_377602

theorem smallest_possible_students : ∃ (n : ℕ), 
  (5 * n + 2 > 50) ∧ 
  (∀ m : ℕ, m < n → 5 * m + 2 ≤ 50) ∧
  (5 * n + 2 = 52) := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_students_l3776_377602


namespace NUMINAMATH_CALUDE_f_less_than_g_l3776_377641

/-- Represents a board arrangement -/
def Board (m n : ℕ+) := Fin m → Fin n → Bool

/-- Number of arrangements with at least one row or column of noughts -/
def f (m n : ℕ+) : ℕ := sorry

/-- Number of arrangements with at least one row of noughts or column of crosses -/
def g (m n : ℕ+) : ℕ := sorry

/-- The theorem stating that f(m,n) < g(m,n) for all positive m and n -/
theorem f_less_than_g (m n : ℕ+) : f m n < g m n := by sorry

end NUMINAMATH_CALUDE_f_less_than_g_l3776_377641


namespace NUMINAMATH_CALUDE_bertha_family_childless_count_l3776_377665

/-- Represents a family tree with three generations -/
structure FamilyTree where
  daughters : ℕ
  granddaughters : ℕ
  daughters_with_children : ℕ

/-- The Bertha family scenario -/
def bertha_family : FamilyTree :=
  { daughters := 10,
    granddaughters := 32,
    daughters_with_children := 4 }

theorem bertha_family_childless_count : 
  let f := bertha_family
  let total := f.daughters + f.granddaughters
  let childless_daughters := f.daughters - f.daughters_with_children
  childless_daughters + f.granddaughters = 38 ∧ 
  total = 42 ∧
  f.daughters_with_children * 8 = f.granddaughters :=
by sorry


end NUMINAMATH_CALUDE_bertha_family_childless_count_l3776_377665


namespace NUMINAMATH_CALUDE_ratio_transitive_l3776_377633

theorem ratio_transitive (a b c : ℝ) 
  (h1 : a / b = 7 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transitive_l3776_377633


namespace NUMINAMATH_CALUDE_original_number_proof_l3776_377691

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) :
  x = Real.sqrt 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l3776_377691


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3776_377632

def P : Set ℝ := {x | x^2 - x = 0}
def Q : Set ℝ := {x | x^2 + x = 0}

theorem intersection_P_Q : P ∩ Q = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3776_377632


namespace NUMINAMATH_CALUDE_fraction_always_positive_l3776_377684

theorem fraction_always_positive (x : ℝ) : 3 / (x^2 + 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_always_positive_l3776_377684


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3776_377621

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 1 + a 2 + a 3 + a 4 + a 5 = 3 →
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 15 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3776_377621


namespace NUMINAMATH_CALUDE_distance_XY_is_16_l3776_377615

-- Define the travel parameters
def travel_time_A : ℕ → Prop := λ t => t * t = 16

def travel_time_B : ℕ → Prop := λ t => 
  ∃ (rest : ℕ), t = 11 ∧ 2 * (t - rest) = 16 ∧ 4 * rest < 16 ∧ 4 * rest + 4 ≥ 16

-- Theorem statement
theorem distance_XY_is_16 : 
  (∃ t : ℕ, travel_time_A t ∧ travel_time_B t) → 
  (∃ d : ℕ, d = 16 ∧ ∀ t : ℕ, travel_time_A t → t * t = d) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_XY_is_16_l3776_377615


namespace NUMINAMATH_CALUDE_expression_evaluation_l3776_377661

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := 1/2
  2 * (a^2 - 2*a*b) - 3 * (a^2 - a*b - 4*b^2) = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3776_377661


namespace NUMINAMATH_CALUDE_complex_subtraction_simplify_complex_expression_l3776_377618

theorem complex_subtraction (z₁ z₂ : ℂ) : z₁ - z₂ = (z₁.re - z₂.re) + (z₁.im - z₂.im) * I := by sorry

theorem simplify_complex_expression : (3 - 2 * I) - (5 - 2 * I) = -2 := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplify_complex_expression_l3776_377618


namespace NUMINAMATH_CALUDE_root_product_is_root_l3776_377617

/-- Given that a and b are two of the four roots of x^4 + x^3 - 1,
    prove that ab is a root of x^6 + x^4 + x^3 - x^2 - 1 -/
theorem root_product_is_root (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_root_product_is_root_l3776_377617


namespace NUMINAMATH_CALUDE_percentage_problem_l3776_377659

theorem percentage_problem (x : ℝ) (h : x = 942.8571428571427) :
  ∃ P : ℝ, (P / 100) * x = (1 / 3) * x + 110 ∧ P = 45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3776_377659


namespace NUMINAMATH_CALUDE_triangle_max_side_length_range_l3776_377634

theorem triangle_max_side_length_range (P : ℝ) (a b c : ℝ) (h_triangle : a + b + c = P) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_inequality : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_max : c = max a (max b c)) : P / 3 ≤ c ∧ c < P / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_range_l3776_377634


namespace NUMINAMATH_CALUDE_painted_subcubes_count_l3776_377606

/-- Represents a cube with painted faces -/
structure PaintedCube :=
  (size : ℕ)
  (isPainted : ℕ → ℕ → ℕ → Bool)

/-- Counts subcubes with at least two painted faces -/
def countSubcubesWithTwoPaintedFaces (cube : PaintedCube) : ℕ :=
  sorry

/-- The main theorem -/
theorem painted_subcubes_count (cube : PaintedCube) 
  (h1 : cube.size = 4)
  (h2 : ∀ x y z, (x = 0 ∨ x = cube.size - 1 ∨ 
                  y = 0 ∨ y = cube.size - 1 ∨ 
                  z = 0 ∨ z = cube.size - 1) → 
                 cube.isPainted x y z = true) :
  countSubcubesWithTwoPaintedFaces cube = 32 :=
sorry

end NUMINAMATH_CALUDE_painted_subcubes_count_l3776_377606


namespace NUMINAMATH_CALUDE_total_fruits_is_fifteen_l3776_377652

-- Define the three types of fruit
inductive FruitType
| A
| B
| C

-- Define a function that returns the quantity of each fruit type
def fruitQuantity (t : FruitType) : Nat :=
  match t with
  | FruitType.A => 5
  | FruitType.B => 6
  | FruitType.C => 4

-- Theorem: The total number of fruits is 15
theorem total_fruits_is_fifteen :
  (fruitQuantity FruitType.A) + (fruitQuantity FruitType.B) + (fruitQuantity FruitType.C) = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fruits_is_fifteen_l3776_377652


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_ratio_l3776_377643

/-- A hexahedron with equilateral triangle faces congruent to those of a regular octahedron -/
structure SpecialHexahedron where
  -- The faces are equilateral triangles
  faces_equilateral : Bool
  -- The faces are congruent to those of a regular octahedron
  faces_congruent_to_octahedron : Bool

/-- A regular octahedron -/
structure RegularOctahedron where

/-- The radius of the inscribed sphere in a polyhedron -/
def inscribed_sphere_radius (P : Type) : ℝ := sorry

/-- The theorem stating the ratio of inscribed sphere radii -/
theorem inscribed_sphere_radius_ratio 
  (h : SpecialHexahedron) 
  (o : RegularOctahedron) 
  (h_valid : h.faces_equilateral ∧ h.faces_congruent_to_octahedron) :
  inscribed_sphere_radius SpecialHexahedron / inscribed_sphere_radius RegularOctahedron = 2/3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_ratio_l3776_377643


namespace NUMINAMATH_CALUDE_coefficient_x2y1_is_60_l3776_377639

/-- The coefficient of x^m y^n in the expansion of (1+x)^6(1+y)^4 -/
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

/-- The theorem stating that the coefficient of x^2y^1 in the expansion of (1+x)^6(1+y)^4 is 60 -/
theorem coefficient_x2y1_is_60 : f 2 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y1_is_60_l3776_377639


namespace NUMINAMATH_CALUDE_number_equality_l3776_377612

theorem number_equality : ∃ x : ℝ, x * 120 = 173 * 240 ∧ x = 346 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l3776_377612


namespace NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_calculation_l3776_377650

/-- The atomic weight of Nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of Oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of Dinitrogen trioxide (N2O3) in g/mol -/
def dinitrogen_trioxide_weight : ℝ := 
  nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

theorem dinitrogen_trioxide_weight_calculation : 
  dinitrogen_trioxide_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_trioxide_weight_calculation_l3776_377650


namespace NUMINAMATH_CALUDE_concrete_amount_l3776_377672

/-- The amount of bricks ordered in tons -/
def bricks : ℝ := 0.17

/-- The amount of stone ordered in tons -/
def stone : ℝ := 0.5

/-- The total amount of material ordered in tons -/
def total_material : ℝ := 0.83

/-- The amount of concrete ordered in tons -/
def concrete : ℝ := total_material - (bricks + stone)

theorem concrete_amount : concrete = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_concrete_amount_l3776_377672


namespace NUMINAMATH_CALUDE_square_cut_corners_l3776_377622

theorem square_cut_corners (s : ℝ) (h : (2 / 9) * s^2 = 288) :
  s - 2 * (1 / 3 * s) = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_corners_l3776_377622


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3776_377604

theorem perfect_square_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 2*(m-3)*x + 25 = y^2) → (m = 8 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3776_377604


namespace NUMINAMATH_CALUDE_square_roots_problem_l3776_377670

theorem square_roots_problem (n : ℝ) (a : ℝ) : 
  n > 0 ∧ (a - 7)^2 = n ∧ (2*a + 1)^2 = n → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l3776_377670


namespace NUMINAMATH_CALUDE_three_zeros_implies_m_in_open_interval_l3776_377668

/-- A cubic function with a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x + m

/-- The theorem stating that if f has three zeros, then m is in the open interval (-4, 4) -/
theorem three_zeros_implies_m_in_open_interval (m : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0) →
  m ∈ Set.Ioo (-4 : ℝ) 4 :=
by sorry

end NUMINAMATH_CALUDE_three_zeros_implies_m_in_open_interval_l3776_377668


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l3776_377648

theorem longest_side_of_triangle (y : ℚ) : 
  6 + (y + 3) + (3 * y - 2) = 40 →
  max 6 (max (y + 3) (3 * y - 2)) = 91 / 4 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l3776_377648


namespace NUMINAMATH_CALUDE_f_eq_2x_plus_7_l3776_377697

-- Define the functions g and f
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem f_eq_2x_plus_7 : ∀ x : ℝ, f x = 2 * x + 7 := by
  sorry

end NUMINAMATH_CALUDE_f_eq_2x_plus_7_l3776_377697


namespace NUMINAMATH_CALUDE_kelly_initial_games_l3776_377637

/-- The number of games Kelly needs to give away -/
def games_to_give : ℕ := 15

/-- The number of games Kelly will have left after giving away some games -/
def games_left : ℕ := 35

/-- Kelly's initial number of games -/
def initial_games : ℕ := games_left + games_to_give

theorem kelly_initial_games : initial_games = 50 := by sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l3776_377637


namespace NUMINAMATH_CALUDE_trains_at_initial_positions_l3776_377695

/-- Represents a metro line with a given number of stations -/
structure MetroLine where
  stations : ℕ
  roundTripTime : ℕ

/-- Theorem: After 2016 minutes, all trains are at their initial positions -/
theorem trains_at_initial_positions 
  (red : MetroLine) 
  (blue : MetroLine) 
  (green : MetroLine)
  (h_red : red.stations = 7 ∧ red.roundTripTime = 14)
  (h_blue : blue.stations = 8 ∧ blue.roundTripTime = 16)
  (h_green : green.stations = 9 ∧ green.roundTripTime = 18) :
  2016 % red.roundTripTime = 0 ∧ 
  2016 % blue.roundTripTime = 0 ∧ 
  2016 % green.roundTripTime = 0 := by
  sorry

#eval 2016 % 14  -- Should output 0
#eval 2016 % 16  -- Should output 0
#eval 2016 % 18  -- Should output 0

end NUMINAMATH_CALUDE_trains_at_initial_positions_l3776_377695


namespace NUMINAMATH_CALUDE_unique_number_l3776_377666

def is_valid_digit (d : Nat) : Bool :=
  d ∈ [0, 1, 6, 8, 9]

def rotate_digit (d : Nat) : Nat :=
  match d with
  | 6 => 9
  | 9 => 6
  | _ => d

def rotate_number (n : Nat) : Nat :=
  let tens := n / 10
  let ones := n % 10
  10 * (rotate_digit ones) + (rotate_digit tens)

def satisfies_condition (n : Nat) : Bool :=
  n >= 10 ∧ n < 100 ∧
  is_valid_digit (n / 10) ∧
  is_valid_digit (n % 10) ∧
  n - (rotate_number n) = 75

theorem unique_number : ∃! n, satisfies_condition n :=
  sorry

end NUMINAMATH_CALUDE_unique_number_l3776_377666


namespace NUMINAMATH_CALUDE_triangle_area_l3776_377601

theorem triangle_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_angle : a / c = 1 / 2) (h_hypotenuse : c = 40) : 
  (a * b) / 2 = 200 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3776_377601


namespace NUMINAMATH_CALUDE_polynomial_inequality_l3776_377642

theorem polynomial_inequality (x : ℝ) : x^4 + x^3 - 10*x^2 > -25*x ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l3776_377642


namespace NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l3776_377692

/-- Represents a rectangular board -/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a rectangle that can be placed on the board -/
structure Rectangle :=
  (length : Nat)
  (width : Nat)

/-- Function to calculate the minimum number of cells to mark -/
def minMarkedCells (b : Board) (r : Rectangle) : Nat :=
  sorry

/-- Theorem stating that 84 is the minimum number of cells to mark -/
theorem min_marked_cells_13x13_board (b : Board) (r : Rectangle) :
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  minMarkedCells b r = 84 :=
by sorry

end NUMINAMATH_CALUDE_min_marked_cells_13x13_board_l3776_377692


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3776_377689

theorem percentage_of_sikh_boys (total_boys : ℕ) (muslim_percentage hindu_percentage : ℚ) 
  (other_boys : ℕ) (h1 : total_boys = 850) (h2 : muslim_percentage = 40/100) 
  (h3 : hindu_percentage = 28/100) (h4 : other_boys = 187) : 
  (total_boys - (muslim_percentage * total_boys + hindu_percentage * total_boys + other_boys)) / total_boys = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l3776_377689


namespace NUMINAMATH_CALUDE_gcd_problem_l3776_377685

theorem gcd_problem : Int.gcd (123^2 + 235^2 - 347^2) (122^2 + 234^2 - 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3776_377685


namespace NUMINAMATH_CALUDE_tobys_remaining_amount_l3776_377674

/-- Calculates the remaining amount for Toby after sharing with his brothers -/
theorem tobys_remaining_amount (initial_amount : ℕ) (num_brothers : ℕ) 
  (h1 : initial_amount = 343)
  (h2 : num_brothers = 2) : 
  initial_amount - num_brothers * (initial_amount / 7) = 245 := by
  sorry

#eval 343 - 2 * (343 / 7)  -- Expected output: 245

end NUMINAMATH_CALUDE_tobys_remaining_amount_l3776_377674


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l3776_377645

/-- Given a point P on the terminal side of angle α with coordinates 
    (sin(2π/3), cos(2π/3)), prove that the smallest positive value of α is 11π/6 -/
theorem smallest_positive_angle (α : Real) : 
  (∃ (P : Real × Real), P.1 = Real.sin (2 * Real.pi / 3) ∧ 
                         P.2 = Real.cos (2 * Real.pi / 3) ∧ 
                         P ∈ {(x, y) | x = Real.sin α ∧ y = Real.cos α}) →
  (∀ β : Real, β > 0 ∧ β < α → β ≥ 11 * Real.pi / 6) ∧ 
  α = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l3776_377645


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3776_377623

theorem sum_of_fractions (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) : 
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3776_377623


namespace NUMINAMATH_CALUDE_brad_start_time_l3776_377626

theorem brad_start_time (distance : ℝ) (maxwell_speed : ℝ) (brad_speed : ℝ) (meet_time : ℝ) :
  distance = 54 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  meet_time = 6 →
  ∃ t : ℝ, t = 1 ∧ 
    distance = maxwell_speed * meet_time + brad_speed * (meet_time - t) :=
by
  sorry

end NUMINAMATH_CALUDE_brad_start_time_l3776_377626


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l3776_377620

theorem exponential_equation_solution :
  ∃ x : ℝ, 3^(3*x + 2) = (1:ℝ)/81 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l3776_377620


namespace NUMINAMATH_CALUDE_problem_statement_l3776_377630

-- Define the base 10 logarithm
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the problem statement
theorem problem_statement (a b : ℝ) :
  a > 0 ∧ b > 0 ∧
  (∃ (m n : ℕ), 
    Real.sqrt (log a) = m ∧
    Real.sqrt (log b) = n ∧
    m + 2*n + m^2 + (n^2/2) = 150) ∧
  a^2 * b = 10^81 →
  a * b = 10^85 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3776_377630


namespace NUMINAMATH_CALUDE_count_odd_numbers_less_than_400_l3776_377649

/-- The set of digits that can be used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4}

/-- A function that checks if a number is odd -/
def isOdd (n : Nat) : Bool := n % 2 = 1

/-- A function that checks if a three-digit number is less than 400 -/
def isLessThan400 (n : Nat) : Bool := n < 400 ∧ n ≥ 100

/-- The set of valid hundreds digits (1, 2, 3) -/
def validHundreds : Finset Nat := {1, 2, 3}

/-- The set of valid units digits for odd numbers (1, 3) -/
def validUnits : Finset Nat := {1, 3}

/-- The main theorem -/
theorem count_odd_numbers_less_than_400 :
  (validHundreds.card * digits.card * validUnits.card) = 24 := by
  sorry

#eval validHundreds.card * digits.card * validUnits.card

end NUMINAMATH_CALUDE_count_odd_numbers_less_than_400_l3776_377649


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_8_l3776_377631

theorem sqrt_2_times_sqrt_8 : Real.sqrt 2 * Real.sqrt 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_8_l3776_377631


namespace NUMINAMATH_CALUDE_greatest_line_segment_length_l3776_377694

/-- The greatest possible length of a line segment joining two points on a circle -/
theorem greatest_line_segment_length (r : ℝ) (h : r = 4) : 
  ∃ (d : ℝ), d = 2 * r ∧ ∀ (l : ℝ), l ≤ d := by sorry

end NUMINAMATH_CALUDE_greatest_line_segment_length_l3776_377694


namespace NUMINAMATH_CALUDE_even_binomial_coefficients_iff_power_of_two_l3776_377688

def is_power_of_two (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = 2^k

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem even_binomial_coefficients_iff_power_of_two (n : ℕ+) :
  (∀ k : ℕ, 1 ≤ k ∧ k < n → Even (binomial_coefficient n k)) ↔ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_even_binomial_coefficients_iff_power_of_two_l3776_377688


namespace NUMINAMATH_CALUDE_class_size_correct_l3776_377658

/-- The number of students in class A -/
def class_size : ℕ := 30

/-- The number of students who like social studies -/
def social_studies_fans : ℕ := 25

/-- The number of students who like music -/
def music_fans : ℕ := 32

/-- The number of students who like both social studies and music -/
def both_fans : ℕ := 27

/-- Theorem stating that the class size is correct given the conditions -/
theorem class_size_correct :
  class_size = social_studies_fans + music_fans - both_fans ∧
  class_size = social_studies_fans + music_fans - both_fans :=
by sorry

end NUMINAMATH_CALUDE_class_size_correct_l3776_377658


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3776_377610

theorem pure_imaginary_condition (x y : ℝ) : 
  (∀ z : ℂ, z.re = x ∧ z.im = y → (z.re = 0 ↔ z.im ≠ 0)) ↔
  (x = 0 → ∃ y : ℝ, y ≠ 0) ∧ (∃ x y : ℝ, x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3776_377610


namespace NUMINAMATH_CALUDE_quadratic_transform_sum_l3776_377619

/-- Given a quadratic equation 9x^2 - 54x - 81 = 0, when transformed into (x+q)^2 = p,
    the sum of q and p is 15 -/
theorem quadratic_transform_sum (q p : ℝ) : 
  (∀ x, 9*x^2 - 54*x - 81 = 0 ↔ (x + q)^2 = p) → q + p = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transform_sum_l3776_377619


namespace NUMINAMATH_CALUDE_executive_board_selection_l3776_377607

theorem executive_board_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_selection_l3776_377607


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l3776_377613

theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) → x = (3 + Real.sqrt 69) / 2 :=
by sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l3776_377613


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l3776_377635

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.sqrt (1 + Real.log (1 + x^2 * Real.sin (1/x))) - 1
  else
    0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l3776_377635


namespace NUMINAMATH_CALUDE_remainder_2345678901_mod_101_l3776_377676

theorem remainder_2345678901_mod_101 : 2345678901 % 101 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345678901_mod_101_l3776_377676


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3776_377662

/-- Theorem: A regular polygon with exterior angles of 40° has 9 sides. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 40 → n * exterior_angle = 360 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3776_377662


namespace NUMINAMATH_CALUDE_at_least_one_alarm_probability_l3776_377683

theorem at_least_one_alarm_probability (pA pB : ℝ) 
  (hpA : 0 ≤ pA ∧ pA ≤ 1) (hpB : 0 ≤ pB ∧ pB ≤ 1) :
  1 - (1 - pA) * (1 - pB) = pA + pB - pA * pB :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_alarm_probability_l3776_377683


namespace NUMINAMATH_CALUDE_sandy_carrots_l3776_377627

def carrots_problem (initial_carrots : ℕ) (carrots_taken : ℕ) (carrots_left : ℕ) : Prop :=
  initial_carrots = carrots_left + carrots_taken

theorem sandy_carrots : ∃ (initial_carrots : ℕ), carrots_problem initial_carrots 3 3 :=
  sorry

end NUMINAMATH_CALUDE_sandy_carrots_l3776_377627


namespace NUMINAMATH_CALUDE_min_marked_cells_l3776_377686

/-- Represents a marked cell on the board -/
structure MarkedCell :=
  (row : ℕ)
  (col : ℕ)

/-- Represents a board with marked cells -/
structure Board :=
  (size : ℕ)
  (marked_cells : List MarkedCell)

/-- Checks if a sub-board contains a marked cell on both diagonals -/
def subBoardContainsMarkedDiagonals (board : Board) (m : ℕ) (topLeft : MarkedCell) : Prop :=
  ∃ (c1 c2 : MarkedCell),
    c1 ∈ board.marked_cells ∧
    c2 ∈ board.marked_cells ∧
    c1.row - topLeft.row = c1.col - topLeft.col ∧
    c2.row - topLeft.row = topLeft.col + m - 1 - c2.col ∧
    c1.row ≥ topLeft.row ∧ c1.row < topLeft.row + m ∧
    c1.col ≥ topLeft.col ∧ c1.col < topLeft.col + m ∧
    c2.row ≥ topLeft.row ∧ c2.row < topLeft.row + m ∧
    c2.col ≥ topLeft.col ∧ c2.col < topLeft.col + m

/-- The main theorem stating the minimum number of marked cells -/
theorem min_marked_cells (n : ℕ) :
  ∃ (board : Board),
    board.size = n ∧
    board.marked_cells.length = n ∧
    (∀ (m : ℕ) (topLeft : MarkedCell),
      m > n / 2 →
      topLeft.row + m ≤ n →
      topLeft.col + m ≤ n →
      subBoardContainsMarkedDiagonals board m topLeft) ∧
    (∀ (board' : Board),
      board'.size = n →
      board'.marked_cells.length < n →
      ∃ (m : ℕ) (topLeft : MarkedCell),
        m > n / 2 ∧
        topLeft.row + m ≤ n ∧
        topLeft.col + m ≤ n ∧
        ¬subBoardContainsMarkedDiagonals board' m topLeft) := by
  sorry

end NUMINAMATH_CALUDE_min_marked_cells_l3776_377686


namespace NUMINAMATH_CALUDE_bird_migration_difference_l3776_377657

/-- The number of bird families that flew to Africa -/
def africa_birds : ℕ := 42

/-- The number of bird families that flew to Asia -/
def asia_birds : ℕ := 31

/-- The number of bird families living near the mountain -/
def mountain_birds : ℕ := 8

/-- Theorem stating the difference between bird families that flew to Africa and Asia -/
theorem bird_migration_difference : africa_birds - asia_birds = 11 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l3776_377657


namespace NUMINAMATH_CALUDE_simplify_expression_proof_l3776_377681

noncomputable def simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : ℝ :=
  (a / b * (b - 4 * a^6 / b^3)^(1/3) - a^2 * (b / a^6 - 4 / b^3)^(1/3) + 2 / (a * b) * (a^3 * b^4 - 4 * a^9)^(1/3)) / ((b^2 - 2 * a^3)^(1/3) / b^2)

theorem simplify_expression_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expression a b ha hb = (a + b) * (b^2 + 2 * a^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_proof_l3776_377681


namespace NUMINAMATH_CALUDE_rotation_result_l3776_377680

/-- Represents the four positions around a circle -/
inductive Position
| Top
| Right
| Bottom
| Left

/-- Represents the four figures on the circle -/
inductive Figure
| Triangle
| SmallerCircle
| Square
| Pentagon

/-- Initial configuration of figures on the circle -/
def initial_config : Figure → Position
| Figure.Triangle => Position.Top
| Figure.SmallerCircle => Position.Right
| Figure.Square => Position.Bottom
| Figure.Pentagon => Position.Left

/-- Rotates a position by 150 degrees clockwise -/
def rotate_150_clockwise : Position → Position
| Position.Top => Position.Left
| Position.Right => Position.Top
| Position.Bottom => Position.Right
| Position.Left => Position.Bottom

/-- Final configuration after 150 degree clockwise rotation -/
def final_config : Figure → Position :=
  λ f => rotate_150_clockwise (initial_config f)

/-- Theorem stating the final positions after rotation -/
theorem rotation_result :
  final_config Figure.Triangle = Position.Left ∧
  final_config Figure.SmallerCircle = Position.Top ∧
  final_config Figure.Square = Position.Right ∧
  final_config Figure.Pentagon = Position.Bottom :=
sorry

end NUMINAMATH_CALUDE_rotation_result_l3776_377680


namespace NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l3776_377675

/-- The number of diagonals from a single vertex in a polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

theorem hexagon_diagonals_from_vertex :
  diagonals_from_vertex hexagon_sides = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l3776_377675


namespace NUMINAMATH_CALUDE_exists_N_average_ten_l3776_377651

theorem exists_N_average_ten :
  ∃ N : ℝ, 9 < N ∧ N < 17 ∧ (6 + 10 + N) / 3 = 10 := by
sorry

end NUMINAMATH_CALUDE_exists_N_average_ten_l3776_377651


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3776_377646

/-- The surface area of a rectangular solid with edge lengths a, b, and c -/
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

/-- The edge lengths of the rectangular solid are prime numbers -/
axiom prime_3 : Nat.Prime 3
axiom prime_5 : Nat.Prime 5
axiom prime_17 : Nat.Prime 17

/-- The edge lengths are different -/
axiom different_edges : 3 ≠ 5 ∧ 3 ≠ 17 ∧ 5 ≠ 17

theorem rectangular_solid_surface_area :
  surface_area 3 5 17 = 302 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3776_377646


namespace NUMINAMATH_CALUDE_vasyas_numbers_l3776_377625

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l3776_377625


namespace NUMINAMATH_CALUDE_six_digit_divisible_by_eleven_l3776_377682

theorem six_digit_divisible_by_eleven (d : Nat) : 
  d < 10 → (67890 * 10 + d) % 11 = 0 ↔ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisible_by_eleven_l3776_377682


namespace NUMINAMATH_CALUDE_population_below_five_percent_in_five_years_population_below_five_percent_in_2011_l3776_377608

/-- Represents the yearly decrease rate of the sparrow population -/
def yearly_decrease_rate : ℝ := 0.5

/-- Represents the target percentage of the original population -/
def target_percentage : ℝ := 0.05

/-- Calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (yearly_decrease_rate ^ years)

/-- Theorem: It takes 5 years for the population to become less than 5% of the original -/
theorem population_below_five_percent_in_five_years (initial_population : ℝ) 
  (h : initial_population > 0) : 
  population_after_years initial_population 5 < target_percentage * initial_population ∧
  ∀ n : ℕ, n < 5 → population_after_years initial_population n ≥ target_percentage * initial_population :=
by sorry

/-- The year when the population becomes less than 5% of the original -/
def year_below_five_percent : ℕ := 2011

/-- Theorem: The population becomes less than 5% of the original in 2011 -/
theorem population_below_five_percent_in_2011 (initial_year : ℕ) (h : initial_year = 2006) :
  year_below_five_percent - initial_year = 5 :=
by sorry

end NUMINAMATH_CALUDE_population_below_five_percent_in_five_years_population_below_five_percent_in_2011_l3776_377608


namespace NUMINAMATH_CALUDE_converse_of_proposition_l3776_377603

theorem converse_of_proposition :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) →
  (∀ x : ℝ, -1 < x ∧ x < 1 → x^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_of_proposition_l3776_377603


namespace NUMINAMATH_CALUDE_cube_root_equation_l3776_377698

theorem cube_root_equation (x : ℝ) : (1 + Real.sqrt x)^(1/3 : ℝ) = 2 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_l3776_377698


namespace NUMINAMATH_CALUDE_decision_box_has_two_exits_l3776_377616

/-- Represents a decision box in a program flowchart -/
structure DecisionBox where
  entrance : Nat
  exits : Nat

/-- Represents a flowchart -/
structure Flowchart where
  endpoints : Nat

/-- Theorem: A decision box in a program flowchart has exactly 2 exits -/
theorem decision_box_has_two_exits (d : DecisionBox) (f : Flowchart) : 
  d.entrance = 1 ∧ f.endpoints ≥ 1 → d.exits = 2 := by
  sorry

end NUMINAMATH_CALUDE_decision_box_has_two_exits_l3776_377616


namespace NUMINAMATH_CALUDE_ship_journey_distance_l3776_377693

theorem ship_journey_distance : 
  let day1_distance : ℕ := 100
  let day2_distance : ℕ := 3 * day1_distance
  let day3_distance : ℕ := day2_distance + 110
  let total_distance : ℕ := day1_distance + day2_distance + day3_distance
  total_distance = 810 := by sorry

end NUMINAMATH_CALUDE_ship_journey_distance_l3776_377693


namespace NUMINAMATH_CALUDE_line_equation_l3776_377687

/-- The equation of a line with slope 2 passing through the point (0, 3) is y = 2x + 3 -/
theorem line_equation (l : Set (ℝ × ℝ)) (slope : ℝ) (point : ℝ × ℝ) : 
  slope = 2 → 
  point = (0, 3) → 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y = 2*x + 3) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3776_377687


namespace NUMINAMATH_CALUDE_total_bronze_needed_l3776_377679

/-- The weight of the first bell in pounds -/
def first_bell_weight : ℕ := 50

/-- The weight of the second bell in pounds -/
def second_bell_weight : ℕ := 2 * first_bell_weight

/-- The weight of the third bell in pounds -/
def third_bell_weight : ℕ := 4 * second_bell_weight

/-- The total weight of bronze needed for all three bells -/
def total_bronze_weight : ℕ := first_bell_weight + second_bell_weight + third_bell_weight

theorem total_bronze_needed :
  total_bronze_weight = 550 := by sorry

end NUMINAMATH_CALUDE_total_bronze_needed_l3776_377679


namespace NUMINAMATH_CALUDE_m_eq_two_iff_parallel_l3776_377690

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The lines l1 and l2 with parameter m -/
def l1 (m : ℝ) : Line := ⟨2, -m, -1⟩
def l2 (m : ℝ) : Line := ⟨m-1, -1, 1⟩

/-- The theorem stating that m=2 is a necessary and sufficient condition for l1 ∥ l2 -/
theorem m_eq_two_iff_parallel :
  ∀ m : ℝ, parallel (l1 m) (l2 m) ↔ m = 2 :=
sorry

end NUMINAMATH_CALUDE_m_eq_two_iff_parallel_l3776_377690


namespace NUMINAMATH_CALUDE_alchemist_safe_combinations_l3776_377628

/-- The number of different herbs available to the alchemist. -/
def num_herbs : ℕ := 4

/-- The number of different gems available to the alchemist. -/
def num_gems : ℕ := 6

/-- The number of unstable combinations of herbs and gems. -/
def num_unstable : ℕ := 3

/-- The total number of possible combinations of herbs and gems. -/
def total_combinations : ℕ := num_herbs * num_gems

/-- The number of safe combinations available for the alchemist's elixir. -/
def safe_combinations : ℕ := total_combinations - num_unstable

theorem alchemist_safe_combinations :
  safe_combinations = 21 :=
sorry

end NUMINAMATH_CALUDE_alchemist_safe_combinations_l3776_377628


namespace NUMINAMATH_CALUDE_number_equation_solution_l3776_377664

theorem number_equation_solution : 
  ∃ x : ℚ, (2 * x - 6 = (1/4) * x + 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3776_377664


namespace NUMINAMATH_CALUDE_same_color_probability_l3776_377609

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 7

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability : 
  (green_balls / total_balls) ^ 2 + (red_balls / total_balls) ^ 2 = 113 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3776_377609


namespace NUMINAMATH_CALUDE_equation_solution_l3776_377605

theorem equation_solution (x y : ℝ) : 9 * x^2 - 25 * y^2 = 0 ↔ x = (5/3) * y ∨ x = -(5/3) * y := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3776_377605


namespace NUMINAMATH_CALUDE_probability_A_timeliness_at_least_75_l3776_377655

/-- Represents the survey data for a company -/
structure SurveyData where
  total_questionnaires : ℕ
  excellent_timeliness : ℕ
  good_timeliness : ℕ
  fair_timeliness : ℕ

/-- Calculates the probability of timeliness rating at least 75 points -/
def probabilityAtLeast75 (data : SurveyData) : ℚ :=
  (data.excellent_timeliness + data.good_timeliness : ℚ) / data.total_questionnaires

/-- The survey data for company A -/
def companyA : SurveyData := {
  total_questionnaires := 120,
  excellent_timeliness := 29,
  good_timeliness := 47,
  fair_timeliness := 44
}

/-- Theorem stating the probability of company A's delivery timeliness being at least 75 points -/
theorem probability_A_timeliness_at_least_75 :
  probabilityAtLeast75 companyA = 19 / 30 := by sorry

end NUMINAMATH_CALUDE_probability_A_timeliness_at_least_75_l3776_377655


namespace NUMINAMATH_CALUDE_raft_existence_l3776_377663

-- Define the river shape
def RiverShape : Type := sorry

-- Define the path of the chip
def ChipPath (river : RiverShape) : Type := sorry

-- Define the raft shape
def RaftShape : Type := sorry

-- Function to check if a raft touches both banks at all points
def touchesBothBanks (river : RiverShape) (raft : RaftShape) (path : ChipPath river) : Prop := sorry

-- Theorem statement
theorem raft_existence (river : RiverShape) (chip_path : ChipPath river) :
  ∃ (raft : RaftShape), touchesBothBanks river raft chip_path := by
  sorry

end NUMINAMATH_CALUDE_raft_existence_l3776_377663


namespace NUMINAMATH_CALUDE_det_E_l3776_377656

/-- A 2x2 matrix representing a dilation centered at the origin with scale factor 5 -/
def E : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0],
    ![0, 5]]

/-- Theorem stating that the determinant of E is 25 -/
theorem det_E : Matrix.det E = 25 := by
  sorry

end NUMINAMATH_CALUDE_det_E_l3776_377656


namespace NUMINAMATH_CALUDE_total_toads_l3776_377667

theorem total_toads (in_pond : ℕ) (outside_pond : ℕ) 
  (h1 : in_pond = 12) (h2 : outside_pond = 6) : 
  in_pond + outside_pond = 18 := by
sorry

end NUMINAMATH_CALUDE_total_toads_l3776_377667


namespace NUMINAMATH_CALUDE_total_weight_calculation_l3776_377624

theorem total_weight_calculation (a b c d : ℝ) 
  (h1 : a + b = 156)
  (h2 : c + d = 195)
  (h3 : a + c = 174)
  (h4 : b + d = 186) :
  a + b + c + d = 355.5 := by
sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l3776_377624


namespace NUMINAMATH_CALUDE_value_of_a_is_one_l3776_377636

/-- Two circles with a common chord of length 2√3 -/
structure TwoCirclesWithCommonChord where
  a : ℝ
  h1 : a > 0
  h2 : ∃ (x y : ℝ), x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0
  h3 : ∃ (x1 y1 x2 y2 : ℝ),
    (x1^2 + y1^2 = 4 ∧ x1^2 + y1^2 + 2*a*y1 - 6 = 0) ∧
    (x2^2 + y2^2 = 4 ∧ x2^2 + y2^2 + 2*a*y2 - 6 = 0) ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 12

/-- The value of a is 1 for two circles with a common chord of length 2√3 -/
theorem value_of_a_is_one (c : TwoCirclesWithCommonChord) : c.a = 1 :=
  sorry

end NUMINAMATH_CALUDE_value_of_a_is_one_l3776_377636


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3776_377638

theorem quadratic_equation_roots : 
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 3
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3776_377638


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l3776_377647

/-- Given points P and Q, where P is at the origin and Q is on the line y = 6,
    if the slope of PQ is 3/4, then the sum of Q's coordinates is 14. -/
theorem point_coordinate_sum (x : ℝ) : 
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (x, 6)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  slope = 3/4 → Q.1 + Q.2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l3776_377647


namespace NUMINAMATH_CALUDE_correct_total_spent_l3776_377611

/-- The total amount Mike spent at the music store after applying the discount -/
def total_spent (trumpet_price songbook_price accessories_price discount_rate : ℝ) : ℝ :=
  let total_before_discount := trumpet_price + songbook_price + accessories_price
  let discount_amount := discount_rate * total_before_discount
  total_before_discount - discount_amount

/-- Theorem stating the correct total amount spent -/
theorem correct_total_spent :
  total_spent 145.16 5.84 18.50 0.12 = 149.16 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_spent_l3776_377611


namespace NUMINAMATH_CALUDE_min_distance_sum_l3776_377673

/-- Definition of circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Definition of circle C₂ -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 1

/-- Definition of the locus of point P -/
def P_locus (a b : ℝ) : Prop := b = -(1/2) * a + 5/2

/-- The main theorem -/
theorem min_distance_sum (a b : ℝ) : 
  C₁ a b → C₂ a b → P_locus a b → 
  Real.sqrt (a^2 + b^2) + Real.sqrt ((a - 5)^2 + (b + 1)^2) ≥ Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3776_377673


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3776_377669

theorem a_equals_one_sufficient_not_necessary_for_abs_a_equals_one :
  ∀ a : ℝ,
  (∀ a : ℝ, a = 1 → |a| = 1) ∧
  (∃ a : ℝ, |a| = 1 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_for_abs_a_equals_one_l3776_377669


namespace NUMINAMATH_CALUDE_geoff_total_spending_l3776_377614

/-- Geoff's spending on sneakers over three days -/
def sneaker_spending (monday_spend : ℝ) : ℝ :=
  let tuesday_spend := 4 * monday_spend
  let wednesday_spend := 5 * monday_spend
  monday_spend + tuesday_spend + wednesday_spend

/-- Theorem stating that Geoff's total spending over three days is $600 -/
theorem geoff_total_spending :
  sneaker_spending 60 = 600 := by
  sorry

end NUMINAMATH_CALUDE_geoff_total_spending_l3776_377614


namespace NUMINAMATH_CALUDE_min_value_theorem_l3776_377678

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 3) :
  x + 2 * y ≥ 3 + 6 * Real.sqrt 2 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 2) + 1 / (y₀ + 2) = 1 / 3 ∧
    x₀ + 2 * y₀ = 3 + 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3776_377678


namespace NUMINAMATH_CALUDE_morks_tax_rate_l3776_377629

theorem morks_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) : 
  mork_tax_rate > 0 →
  mork_income > 0 →
  let mindy_income := 3 * mork_income
  let mindy_tax_rate := 0.3
  let total_income := mork_income + mindy_income
  let total_tax := mork_tax_rate * mork_income + mindy_tax_rate * mindy_income
  let combined_tax_rate := total_tax / total_income
  combined_tax_rate = 0.325 →
  mork_tax_rate = 0.4 := by
sorry

end NUMINAMATH_CALUDE_morks_tax_rate_l3776_377629


namespace NUMINAMATH_CALUDE_leftover_bolts_count_l3776_377600

theorem leftover_bolts_count :
  let bolt_boxes : Nat := 7
  let bolts_per_box : Nat := 11
  let nut_boxes : Nat := 3
  let nuts_per_box : Nat := 15
  let used_bolts_and_nuts : Nat := 113
  let leftover_nuts : Nat := 6
  
  let total_bolts : Nat := bolt_boxes * bolts_per_box
  let total_nuts : Nat := nut_boxes * nuts_per_box
  let total_bolts_and_nuts : Nat := total_bolts + total_nuts
  let leftover_bolts_and_nuts : Nat := total_bolts_and_nuts - used_bolts_and_nuts
  let leftover_bolts : Nat := leftover_bolts_and_nuts - leftover_nuts

  leftover_bolts = 3 := by sorry

end NUMINAMATH_CALUDE_leftover_bolts_count_l3776_377600


namespace NUMINAMATH_CALUDE_min_value_expression_l3776_377677

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  1 / (x + y) + 1 / (x + z) + 1 / (y + z) - x * y * z ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3776_377677


namespace NUMINAMATH_CALUDE_charge_account_interest_l3776_377653

/-- Calculates the total amount owed after one year given an initial charge and simple annual interest rate -/
def total_amount_owed (initial_charge : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_charge * (1 + interest_rate)

/-- Proves that the total amount owed after one year for a $60 charge at 6% simple annual interest is $63.60 -/
theorem charge_account_interest :
  total_amount_owed 60 0.06 = 63.60 := by
  sorry

end NUMINAMATH_CALUDE_charge_account_interest_l3776_377653


namespace NUMINAMATH_CALUDE_sum_of_products_even_l3776_377640

/-- Represents a regular hexagon with natural numbers assigned to its vertices -/
structure Hexagon where
  vertices : Fin 6 → ℕ

/-- The sum of products of adjacent vertex pairs in a hexagon -/
def sum_of_products (h : Hexagon) : ℕ :=
  (h.vertices 0 * h.vertices 1) + (h.vertices 1 * h.vertices 2) +
  (h.vertices 2 * h.vertices 3) + (h.vertices 3 * h.vertices 4) +
  (h.vertices 4 * h.vertices 5) + (h.vertices 5 * h.vertices 0)

/-- A hexagon with opposite vertices having the same value -/
def opposite_same_hexagon (a b c : ℕ) : Hexagon :=
  { vertices := fun i => match i with
    | 0 | 3 => a
    | 1 | 4 => b
    | 2 | 5 => c }

theorem sum_of_products_even (a b c : ℕ) :
  Even (sum_of_products (opposite_same_hexagon a b c)) := by
  sorry

#check sum_of_products_even

end NUMINAMATH_CALUDE_sum_of_products_even_l3776_377640


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_decimal_l3776_377699

theorem scientific_notation_of_small_decimal (x : ℝ) :
  x = 0.000815 →
  ∃ (a : ℝ) (n : ℤ), x = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -4 ∧ a = 8.15 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_decimal_l3776_377699
