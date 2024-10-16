import Mathlib

namespace NUMINAMATH_CALUDE_blue_then_green_probability_l3706_370638

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  total_eq : sides = red + blue + yellow + green

/-- The probability of an event occurring -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The probability of two independent events occurring in sequence -/
def sequential_probability (p1 : ℚ) (p2 : ℚ) : ℚ :=
  p1 * p2

theorem blue_then_green_probability (d : ColoredDie) 
  (h : d = ⟨12, 5, 4, 2, 1, rfl⟩) : 
  sequential_probability (probability d.blue d.sides) (probability d.green d.sides) = 1/36 := by
  sorry

end NUMINAMATH_CALUDE_blue_then_green_probability_l3706_370638


namespace NUMINAMATH_CALUDE_muffin_selling_price_l3706_370621

/-- Represents the daily muffin order quantity -/
def daily_order : ℕ := 12

/-- Represents the cost of each muffin in cents -/
def cost_per_muffin : ℕ := 75

/-- Represents the weekly profit in cents -/
def weekly_profit : ℕ := 6300

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the selling price of each muffin in cents -/
def selling_price : ℕ :=
  let total_cost := daily_order * cost_per_muffin * days_per_week
  let total_revenue := total_cost + weekly_profit
  let total_muffins := daily_order * days_per_week
  total_revenue / total_muffins

theorem muffin_selling_price :
  selling_price = 150 := by sorry

end NUMINAMATH_CALUDE_muffin_selling_price_l3706_370621


namespace NUMINAMATH_CALUDE_triangle_properties_l3706_370613

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  sin (2 * C) = Real.sqrt 3 * sin C →
  b = 4 →
  (1 / 2) * a * b * sin C = 2 * Real.sqrt 3 →
  -- Conclusions
  C = π / 6 ∧
  a + b + c = 6 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3706_370613


namespace NUMINAMATH_CALUDE_acetone_molecular_weight_proof_l3706_370681

-- Define the isotopes and their properties
structure Isotope where
  mass : Float
  abundance : Float

-- Define the elements and their isotopes
def carbon_isotopes : List Isotope := [
  { mass := 12, abundance := 0.9893 },
  { mass := 13.003355, abundance := 0.0107 }
]

def hydrogen_isotopes : List Isotope := [
  { mass := 1.007825, abundance := 0.999885 },
  { mass := 2.014102, abundance := 0.000115 }
]

def oxygen_isotopes : List Isotope := [
  { mass := 15.994915, abundance := 0.99757 },
  { mass := 16.999132, abundance := 0.00038 },
  { mass := 17.999159, abundance := 0.00205 }
]

-- Function to calculate average atomic mass
def average_atomic_mass (isotopes : List Isotope) : Float :=
  isotopes.foldl (fun acc isotope => acc + isotope.mass * isotope.abundance) 0

-- Define the molecular formula of Acetone
def acetone_formula : List (Nat × List Isotope) := [
  (3, carbon_isotopes),
  (6, hydrogen_isotopes),
  (1, oxygen_isotopes)
]

-- Calculate the molecular weight of Acetone
def acetone_molecular_weight : Float :=
  acetone_formula.foldl (fun acc (n, isotopes) => acc + n.toFloat * average_atomic_mass isotopes) 0

-- Theorem statement
theorem acetone_molecular_weight_proof :
  (acetone_molecular_weight - 58.107055).abs < 0.000001 := by
  sorry


end NUMINAMATH_CALUDE_acetone_molecular_weight_proof_l3706_370681


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3706_370690

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3706_370690


namespace NUMINAMATH_CALUDE_smallest_solution_is_negative_85_l3706_370635

def floor_equation (x : ℤ) : Prop :=
  Int.floor (x / 2) + Int.floor (x / 3) + Int.floor (x / 7) = x

theorem smallest_solution_is_negative_85 :
  (∀ y < -85, ¬ floor_equation y) ∧ floor_equation (-85) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_is_negative_85_l3706_370635


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3706_370628

theorem trigonometric_inequality (φ : Real) (h : 0 < φ ∧ φ < Real.pi / 2) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3706_370628


namespace NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l3706_370667

theorem integral_3x_plus_sin_x (x : Real) :
  ∫ x in (0)..(π/2), (3 * x + Real.sin x) = (3/8) * π^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l3706_370667


namespace NUMINAMATH_CALUDE_equation_a_correct_equation_b_correct_equation_c_correct_equation_d_incorrect_l3706_370693

/-- Represents the principal amount in yuan -/
def principal : ℝ := sorry

/-- The annual interest rate -/
def interest_rate : ℝ := 0.03

/-- The total amount withdrawn after one year -/
def total_amount : ℝ := 20600

/-- Theorem stating that equation A is correct -/
theorem equation_a_correct : principal + interest_rate * principal = total_amount := by sorry

/-- Theorem stating that equation B is correct -/
theorem equation_b_correct : interest_rate * principal = total_amount - principal := by sorry

/-- Theorem stating that equation C is correct -/
theorem equation_c_correct : principal - total_amount = -(interest_rate * principal) := by sorry

/-- Theorem stating that equation D is incorrect -/
theorem equation_d_incorrect : principal + interest_rate ≠ total_amount := by sorry

end NUMINAMATH_CALUDE_equation_a_correct_equation_b_correct_equation_c_correct_equation_d_incorrect_l3706_370693


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l3706_370655

-- Define a type for points in a plane
variable (Point : Type*)

-- Define a type for lines in a plane
variable (Line : Type*)

-- Define a function to determine if three lines form a triangle
variable (form_triangle : Line → Line → Line → Prop)

-- Define a function to get the orthocenter of a triangle formed by three lines
variable (orthocenter : Line → Line → Line → Point)

-- Define a function to check if points are collinear
variable (collinear : List Point → Prop)

-- Theorem statement
theorem orthocenters_collinear 
  (l₁ l₂ l₃ l₄ : Line)
  (h : ∀ (a b c : Line), a ∈ [l₁, l₂, l₃, l₄] → b ∈ [l₁, l₂, l₃, l₄] → c ∈ [l₁, l₂, l₃, l₄] → 
     a ≠ b ∧ b ≠ c ∧ a ≠ c → form_triangle a b c) :
  collinear [
    orthocenter l₁ l₂ l₃,
    orthocenter l₁ l₂ l₄,
    orthocenter l₁ l₃ l₄,
    orthocenter l₂ l₃ l₄
  ] := by
  sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l3706_370655


namespace NUMINAMATH_CALUDE_find_correct_divisor_l3706_370632

theorem find_correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X / (D + 12) = 70) (h3 : X / D = 40) : D = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_correct_divisor_l3706_370632


namespace NUMINAMATH_CALUDE_pillow_average_price_l3706_370631

/-- Given 4 pillows with an average cost of $5 and an additional pillow costing $10,
    prove that the average price of all 5 pillows is $6 -/
theorem pillow_average_price (n : ℕ) (avg_cost : ℚ) (additional_cost : ℚ) :
  n = 4 ∧ avg_cost = 5 ∧ additional_cost = 10 →
  ((n : ℚ) * avg_cost + additional_cost) / ((n : ℚ) + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_pillow_average_price_l3706_370631


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l3706_370675

theorem vector_addition_and_scalar_multiplication :
  (3 : ℝ) • (⟨4, -2⟩ : ℝ × ℝ) + ⟨-3, 5⟩ = ⟨9, -1⟩ := by sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l3706_370675


namespace NUMINAMATH_CALUDE_maria_towels_l3706_370617

theorem maria_towels (green_towels white_towels given_away : ℕ) 
  (h1 : green_towels = 125)
  (h2 : white_towels = 130)
  (h3 : given_away = 180) :
  green_towels + white_towels - given_away = 75 := by
  sorry

end NUMINAMATH_CALUDE_maria_towels_l3706_370617


namespace NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l3706_370602

theorem x_cube_plus_reciprocal (φ : Real) (x : Real) 
  (h1 : 0 < φ) (h2 : φ < π) (h3 : x + 1/x = 2 * Real.cos (2 * φ)) : 
  x^3 + 1/x^3 = 2 * Real.cos (6 * φ) := by
  sorry

end NUMINAMATH_CALUDE_x_cube_plus_reciprocal_l3706_370602


namespace NUMINAMATH_CALUDE_second_division_percentage_l3706_370688

theorem second_division_percentage (total_students : ℕ) 
  (first_division_percentage : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  first_division_percentage = 30 / 100 →
  just_passed = 48 →
  (total_students : ℚ) * first_division_percentage + 
    (just_passed : ℚ) + 
    (total_students : ℚ) * (54 / 100) = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_second_division_percentage_l3706_370688


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l3706_370664

def concatenate_range (a b : ℕ) : ℕ :=
  sorry

def M : ℕ := concatenate_range 25 87

theorem highest_power_of_three_dividing_M :
  ∃ (k : ℕ), M % 3 = 0 ∧ M % (3^2) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l3706_370664


namespace NUMINAMATH_CALUDE_tracys_candies_l3706_370648

theorem tracys_candies (x : ℕ) : 
  (x % 3 = 0) →  -- x is divisible by 3
  (x % 2 = 0) →  -- x is divisible by 2
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 5) →  -- Tracy's brother took between 1 and 5 candies
  (x / 2 - 30 - k = 3) →  -- Tracy was left with 3 candies after all events
  x = 72 :=
by sorry

#check tracys_candies

end NUMINAMATH_CALUDE_tracys_candies_l3706_370648


namespace NUMINAMATH_CALUDE_log_xy_value_l3706_370644

theorem log_xy_value (x y : ℝ) 
  (h1 : Real.log (x^2 * y^2) = 1) 
  (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x * y) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l3706_370644


namespace NUMINAMATH_CALUDE_problem_solution_l3706_370683

theorem problem_solution :
  (∀ (a : ℝ), a ≠ 0 → (a^2)^3 / (-a)^2 = a^4) ∧
  (∀ (a b : ℝ), (a+2*b)*(a+b) - 3*a*(a+b) = -2*a^2 + 2*b^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3706_370683


namespace NUMINAMATH_CALUDE_c_investment_value_l3706_370685

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  a_profit_share : ℕ

/-- Theorem stating that under the given conditions, C's investment is 9600 -/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 2400)
  (h2 : p.b_investment = 7200)
  (h3 : p.total_profit = 9000)
  (h4 : p.a_profit_share = 1125)
  (h5 : p.a_profit_share * (p.a_investment + p.b_investment + p.c_investment) = p.a_investment * p.total_profit) :
  p.c_investment = 9600 := by
  sorry

#check c_investment_value

end NUMINAMATH_CALUDE_c_investment_value_l3706_370685


namespace NUMINAMATH_CALUDE_president_vice_president_election_committee_members_election_l3706_370677

-- Define the number of candidates
def num_candidates : ℕ := 4

-- Define the number of positions for the first question (president and vice president)
def num_positions_1 : ℕ := 2

-- Define the number of positions for the second question (committee members)
def num_positions_2 : ℕ := 3

-- Theorem for the first question
theorem president_vice_president_election :
  (num_candidates.choose num_positions_1) * num_positions_1.factorial = 12 := by
  sorry

-- Theorem for the second question
theorem committee_members_election :
  num_candidates.choose num_positions_2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_election_committee_members_election_l3706_370677


namespace NUMINAMATH_CALUDE_luncheon_cost_l3706_370661

/-- Given two luncheon bills, prove the cost of one sandwich, one coffee, and one pie --/
theorem luncheon_cost (s c p : ℚ) : 
  (5 * s + 8 * c + 2 * p = 510/100) →
  (6 * s + 11 * c + 2 * p = 645/100) →
  (s + c + p = 135/100) := by
  sorry

#check luncheon_cost

end NUMINAMATH_CALUDE_luncheon_cost_l3706_370661


namespace NUMINAMATH_CALUDE_problem_structure_surface_area_l3706_370615

/-- Represents the 3D structure composed of unit cubes -/
structure CubeStructure where
  base : Nat
  secondLayer : Nat
  column : Nat
  sideOne : Nat
  sideTwo : Nat

/-- Calculates the surface area of the given cube structure -/
def surfaceArea (s : CubeStructure) : Nat :=
  let frontBack := s.base + s.secondLayer + s.column + s.sideOne + s.sideTwo
  let top := (s.base - s.secondLayer) + s.secondLayer + 1 + s.sideOne + s.sideTwo
  let bottom := s.base
  2 * frontBack + top + bottom

/-- The specific cube structure described in the problem -/
def problemStructure : CubeStructure :=
  { base := 5
  , secondLayer := 3
  , column := 2
  , sideOne := 3
  , sideTwo := 2 }

/-- Theorem stating that the surface area of the problem structure is 62 -/
theorem problem_structure_surface_area :
  surfaceArea problemStructure = 62 := by sorry

end NUMINAMATH_CALUDE_problem_structure_surface_area_l3706_370615


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3706_370637

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with ratio q
  a 1 = 2 →                     -- a_1 = 2
  (a 1 + a 2 + a 3 = 6) →       -- S_3 = 6
  (q = 1 ∨ q = -2) :=           -- q = 1 or q = -2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3706_370637


namespace NUMINAMATH_CALUDE_students_without_eyewear_l3706_370678

/-- Given a population of students with specified percentages wearing glasses, contact lenses, or both,
    calculate the number of students not wearing any eyewear. -/
theorem students_without_eyewear
  (total_students : ℕ)
  (glasses_percent : ℚ)
  (contacts_percent : ℚ)
  (both_percent : ℚ)
  (h_total : total_students = 1200)
  (h_glasses : glasses_percent = 35 / 100)
  (h_contacts : contacts_percent = 25 / 100)
  (h_both : both_percent = 10 / 100) :
  (total_students : ℚ) * (1 - (glasses_percent + contacts_percent - both_percent)) = 600 :=
sorry

end NUMINAMATH_CALUDE_students_without_eyewear_l3706_370678


namespace NUMINAMATH_CALUDE_product_of_sums_l3706_370697

theorem product_of_sums (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  a * b + a + b = 99 ∧ b * c + b + c = 99 ∧ c * a + c + a = 99 →
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l3706_370697


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3706_370682

theorem min_distance_to_line : 
  ∃ (d : ℝ), d > 0 ∧ 
  (∀ a b : ℝ, a + 2*b = Real.sqrt 5 → Real.sqrt (a^2 + b^2) ≥ d) ∧
  (∃ a b : ℝ, a + 2*b = Real.sqrt 5 ∧ Real.sqrt (a^2 + b^2) = d) ∧
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3706_370682


namespace NUMINAMATH_CALUDE_two_valid_colorings_l3706_370660

/-- Represents the three possible colors for the hexagons -/
inductive Color
| Red
| Yellow
| Green

/-- Represents a position in the hexagonal grid -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the hexagonal grid -/
def HexagonalGrid := Position → Color

/-- Checks if two positions are adjacent -/
def are_adjacent (p1 p2 : Position) : Bool :=
  sorry

/-- Checks if a coloring is valid according to the rules -/
def is_valid_coloring (grid : HexagonalGrid) : Prop :=
  ∀ p1 p2 : Position, are_adjacent p1 p2 → grid p1 ≠ grid p2

/-- The central hexagon position -/
def central_position : Position :=
  ⟨0, 0⟩

/-- Theorem stating that there are exactly two valid colorings -/
theorem two_valid_colorings :
  ∃! (n : Nat), n = 2 ∧ 
    ∃ (colorings : Fin n → HexagonalGrid),
      (∀ i : Fin n, is_valid_coloring (colorings i)) ∧
      (∀ i : Fin n, colorings i central_position = Color.Red) ∧
      (∀ g : HexagonalGrid, is_valid_coloring g → g central_position = Color.Red →
        ∃ i : Fin n, g = colorings i) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_colorings_l3706_370660


namespace NUMINAMATH_CALUDE_symmetric_points_product_l3706_370600

/-- Given two points A(-2, a) and B(b, -3) symmetric about the y-axis, prove that ab = -6 -/
theorem symmetric_points_product (a b : ℝ) : 
  ((-2 : ℝ) = -b) → (a = -3) → ab = -6 := by sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l3706_370600


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3706_370658

theorem abs_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3706_370658


namespace NUMINAMATH_CALUDE_pizza_theorem_l3706_370652

def pizza_eaten (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1/3
  else pizza_eaten (n-1) + (1 - pizza_eaten (n-1)) / 2

theorem pizza_theorem :
  pizza_eaten 4 = 11/12 ∧ (1 - pizza_eaten 4) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l3706_370652


namespace NUMINAMATH_CALUDE_expression_evaluation_l3706_370665

theorem expression_evaluation (x y : ℤ) (hx : x = -1) (hy : y = 2) : 
  ((x + 2*y) * (x - 2*y) - (x - y)^2) = -24 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3706_370665


namespace NUMINAMATH_CALUDE_quadratic_points_comparison_l3706_370609

theorem quadratic_points_comparison (c : ℝ) (y₁ y₂ : ℝ) 
  (h1 : y₁ = (-1)^2 - 6*(-1) + c) 
  (h2 : y₂ = 2^2 - 6*2 + c) : 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_points_comparison_l3706_370609


namespace NUMINAMATH_CALUDE_cabbage_count_this_year_l3706_370684

/-- Represents the number of cabbages in a square garden --/
def CabbageCount (side : ℕ) : ℕ := side * side

/-- Theorem stating the number of cabbages this year given the conditions --/
theorem cabbage_count_this_year :
  ∀ (last_year_side : ℕ),
  (CabbageCount (last_year_side + 1) - CabbageCount last_year_side = 197) →
  (CabbageCount (last_year_side + 1) = 9801) :=
by
  sorry

end NUMINAMATH_CALUDE_cabbage_count_this_year_l3706_370684


namespace NUMINAMATH_CALUDE_father_age_twice_marika_correct_target_year_l3706_370657

/-- Marika's birth year -/
def marika_birth_year : ℕ := 1996

/-- The year when Marika's father's age was five times her age -/
def reference_year : ℕ := 2006

/-- Marika's father's age in the reference year -/
def father_age_reference : ℕ := 5 * (reference_year - marika_birth_year)

/-- The year when Marika's father's age will be twice her age -/
def target_year : ℕ := 2036

theorem father_age_twice_marika (year : ℕ) :
  year = target_year ↔
  (year - marika_birth_year) * 2 = (year - reference_year) + father_age_reference :=
by sorry

theorem correct_target_year : 
  (target_year - marika_birth_year) * 2 = (target_year - reference_year) + father_age_reference :=
by sorry

end NUMINAMATH_CALUDE_father_age_twice_marika_correct_target_year_l3706_370657


namespace NUMINAMATH_CALUDE_intersection_probability_in_decagon_l3706_370604

/-- A regular decagon is a 10-sided polygon -/
def RegularDecagon : ℕ := 10

/-- The number of diagonals in a regular decagon -/
def NumDiagonals : ℕ := (RegularDecagon.choose 2) - RegularDecagon

/-- The number of ways to choose two diagonals -/
def WaysToChooseTwoDiagonals : ℕ := NumDiagonals.choose 2

/-- The number of convex quadrilaterals that can be formed in a regular decagon -/
def NumConvexQuadrilaterals : ℕ := RegularDecagon.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the decagon -/
def ProbabilityIntersectionInside : ℚ := NumConvexQuadrilaterals / WaysToChooseTwoDiagonals

theorem intersection_probability_in_decagon :
  ProbabilityIntersectionInside = 42 / 119 := by sorry

end NUMINAMATH_CALUDE_intersection_probability_in_decagon_l3706_370604


namespace NUMINAMATH_CALUDE_max_product_value_l3706_370601

-- Define the functions h and k on ℝ
variable (h k : ℝ → ℝ)

-- Define the ranges of h and k
variable (h_range : Set.range h = Set.Icc (-3) 5)
variable (k_range : Set.range k = Set.Icc (-1) 3)

-- Theorem statement
theorem max_product_value :
  ∃ (x : ℝ), h x * k x = 15 ∧ ∀ (y : ℝ), h y * k y ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_max_product_value_l3706_370601


namespace NUMINAMATH_CALUDE_equation_solution_l3706_370616

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (5 * x) ^ 15 = (25 * x) ^ 5 ↔ x = Real.sqrt (1 / 5) ∨ x = -Real.sqrt (1 / 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3706_370616


namespace NUMINAMATH_CALUDE_calculate_units_produced_l3706_370691

/-- Given fixed cost, marginal cost, and total cost, calculate the number of units produced. -/
theorem calculate_units_produced 
  (fixed_cost : ℝ) 
  (marginal_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000) :
  (total_cost - fixed_cost) / marginal_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_calculate_units_produced_l3706_370691


namespace NUMINAMATH_CALUDE_x_varies_as_eighth_power_of_z_l3706_370662

/-- Given that x varies as the fourth power of y, and y varies as the square of z,
    prove that x varies as the 8th power of z. -/
theorem x_varies_as_eighth_power_of_z
  (k : ℝ) (j : ℝ) (x y z : ℝ → ℝ)
  (h1 : ∀ t, x t = k * (y t)^4)
  (h2 : ∀ t, y t = j * (z t)^2) :
  ∃ m : ℝ, ∀ t, x t = m * (z t)^8 := by
  sorry

end NUMINAMATH_CALUDE_x_varies_as_eighth_power_of_z_l3706_370662


namespace NUMINAMATH_CALUDE_last_digit_is_zero_l3706_370646

def number (last_digit : Nat) : Nat :=
  626840 + last_digit

theorem last_digit_is_zero :
  ∀ d : Nat, d < 10 →
  (number d % 8 = 0 ∧ number d % 5 = 0) →
  d = 0 := by
sorry

end NUMINAMATH_CALUDE_last_digit_is_zero_l3706_370646


namespace NUMINAMATH_CALUDE_frank_max_average_time_l3706_370633

/-- The maximum average time per maze Frank wants to maintain -/
def maxAverageTime (previousMazes : ℕ) (averagePreviousTime : ℕ) (currentTime : ℕ) (remainingTime : ℕ) : ℚ :=
  let totalPreviousTime := previousMazes * averagePreviousTime
  let totalCurrentTime := currentTime + remainingTime
  let totalTime := totalPreviousTime + totalCurrentTime
  let totalMazes := previousMazes + 1
  totalTime / totalMazes

/-- Theorem stating the maximum average time Frank wants to maintain -/
theorem frank_max_average_time :
  maxAverageTime 4 50 45 55 = 60 := by
  sorry

end NUMINAMATH_CALUDE_frank_max_average_time_l3706_370633


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3706_370656

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * ((8 / y) - 6 * y^2 + 3 * y) = 6 / y - (9 * y^2) / 2 + (9 * y) / 4 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3706_370656


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3706_370666

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- Define the universal set U
def U : Set ℝ := A ∪ B 3

-- Part 1
theorem part_one : A ∩ (U \ B 3) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : A ∩ B m = ∅ → m ≤ -2 := by sorry

-- Part 3
theorem part_three : A ∩ B m = A → m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3706_370666


namespace NUMINAMATH_CALUDE_fraction_multiplication_division_l3706_370618

theorem fraction_multiplication_division (a b c d e f g h : ℚ) 
  (h1 : c = 663 / 245)
  (h2 : f = 328 / 15) :
  a / b * c / d / f = g / h :=
by
  sorry

#check fraction_multiplication_division (145 : ℚ) (273 : ℚ) (663 / 245 : ℚ) (1 : ℚ) (1 : ℚ) (328 / 15 : ℚ) (7395 : ℚ) (112504 : ℚ)

end NUMINAMATH_CALUDE_fraction_multiplication_division_l3706_370618


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l3706_370679

/-- The number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- Formula for the number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex heptagon is 14 -/
theorem heptagon_diagonals : num_diagonals heptagon_sides = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l3706_370679


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l3706_370630

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : 
  a^3 + b^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l3706_370630


namespace NUMINAMATH_CALUDE_circle_sum_puzzle_l3706_370653

/-- A solution is a 6-tuple of natural numbers representing the values in circles A, B, C, D, E, F --/
def Solution := (Nat × Nat × Nat × Nat × Nat × Nat)

/-- Check if a solution satisfies all conditions --/
def is_valid_solution (s : Solution) : Prop :=
  let (a, b, c, d, e, f) := s
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  b + c + a = 22 ∧
  d + c + f = 11 ∧
  e + b + d = 19 ∧
  a + e + c = 22

theorem circle_sum_puzzle :
  ∃! (s1 s2 : Solution),
    is_valid_solution s1 ∧
    is_valid_solution s2 ∧
    (∀ s, is_valid_solution s → (s = s1 ∨ s = s2)) :=
sorry

end NUMINAMATH_CALUDE_circle_sum_puzzle_l3706_370653


namespace NUMINAMATH_CALUDE_odd_function_properties_l3706_370692

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_properties (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ x y : ℝ, x < y → f 2 1 x > f 2 1 y) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 1 → f 2 1 (k * 3^x) + f 2 1 (3^x - 9^x + 2) > 0) ↔ k < 4/3) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l3706_370692


namespace NUMINAMATH_CALUDE_valid_arrangement_iff_even_l3706_370663

/-- A valid grid arrangement for the problem -/
def ValidArrangement (n : ℕ) (grid : Fin n → Fin n → ℕ) : Prop :=
  (∀ i j, grid i j ∈ Finset.range (n^2)) ∧
  (∀ k : Fin (n^2 - 1), ∃ i j i' j', 
    grid i j = k ∧ grid i' j' = k + 1 ∧ 
    ((i = i' ∧ j.val + 1 = j'.val) ∨ 
     (j = j' ∧ i.val + 1 = i'.val))) ∧
  (∀ i j i' j', grid i j % n = grid i' j' % n → 
    (i ≠ i' ∧ j ≠ j'))

/-- The main theorem stating that a valid arrangement exists if and only if n is even -/
theorem valid_arrangement_iff_even (n : ℕ) (h : n > 1) :
  (∃ grid, ValidArrangement n grid) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_iff_even_l3706_370663


namespace NUMINAMATH_CALUDE_problem_solution_l3706_370689

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x * log x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |exp x - a| + a^2 / 2

theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (exp 1), Monotone (f a)) →
  (∃ M m : ℝ, (∀ x ∈ Set.Icc 0 (log 3), m ≤ g a x ∧ g a x ≤ M) ∧ M - m = 3/2) →
  a = 5/2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3706_370689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3706_370611

/-- An arithmetic sequence with first three terms a-1, a+1, and 2a+3 has general term 2n-3 -/
theorem arithmetic_sequence_general_term (a : ℝ) (n : ℕ) :
  let a₁ := a - 1
  let a₂ := a + 1
  let a₃ := 2 * a + 3
  let d := a₂ - a₁
  let aₙ := a₁ + (n - 1) * d
  (a₁ + a₃) / 2 = a₂ → aₙ = 2 * n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3706_370611


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3706_370610

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3706_370610


namespace NUMINAMATH_CALUDE_eliminate_denominators_l3706_370634

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  1 + 2 / (x - 1) = (x - 5) / (x - 3)

-- Define the result after eliminating denominators
def eliminated_denominators (x : ℝ) : Prop :=
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1)

-- Theorem stating that eliminating denominators in the original equation
-- results in the specified equation
theorem eliminate_denominators (x : ℝ) :
  original_equation x → eliminated_denominators x :=
by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3706_370634


namespace NUMINAMATH_CALUDE_circle_fraction_range_l3706_370650

theorem circle_fraction_range (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  -(Real.sqrt 3 / 3) ≤ y / (x + 2) ∧ y / (x + 2) ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_circle_fraction_range_l3706_370650


namespace NUMINAMATH_CALUDE_book_collection_average_l3706_370699

def arithmeticSequenceSum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def arithmeticSequenceAverage (a d n : ℕ) : ℚ :=
  (arithmeticSequenceSum a d n : ℚ) / n

theorem book_collection_average :
  arithmeticSequenceAverage 12 12 7 = 48 := by
  sorry

end NUMINAMATH_CALUDE_book_collection_average_l3706_370699


namespace NUMINAMATH_CALUDE_problem_solution_l3706_370614

theorem problem_solution (x y : ℝ) (hx : x = 3 + 2 * Real.sqrt 2) (hy : y = 3 - 2 * Real.sqrt 2) :
  (x + y = 6) ∧
  (x - y = 4 * Real.sqrt 2) ∧
  (x * y = 1) ∧
  (x^2 - 3*x*y + y^2 - x - y = 25) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3706_370614


namespace NUMINAMATH_CALUDE_difference_of_squares_650_350_l3706_370695

theorem difference_of_squares_650_350 : 650^2 - 350^2 = 300000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_650_350_l3706_370695


namespace NUMINAMATH_CALUDE_find_y_value_l3706_370639

theorem find_y_value (x y z : ℤ) 
  (eq1 : x + y + z = 270)
  (eq2 : x - y + z = 200)
  (eq3 : x + y - z = 150) :
  y = 35 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l3706_370639


namespace NUMINAMATH_CALUDE_range_of_a_l3706_370669

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) → 
  a ∈ Set.Icc 4 8 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3706_370669


namespace NUMINAMATH_CALUDE_winning_percentage_l3706_370670

theorem winning_percentage (total_votes : ℕ) (majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 4500 →
  majority = 900 →
  winning_percentage = 60 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = majority :=
by
  sorry

end NUMINAMATH_CALUDE_winning_percentage_l3706_370670


namespace NUMINAMATH_CALUDE_find_b_value_l3706_370687

theorem find_b_value (a b : ℝ) (eq1 : 3 * a + 2 = 2) (eq2 : b - 2 * a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l3706_370687


namespace NUMINAMATH_CALUDE_right_side_number_l3706_370626

theorem right_side_number (x : ℝ) (some_number : ℝ) 
  (h1 : x + 1 = some_number) (h2 : x = 1) : some_number = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_side_number_l3706_370626


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3706_370636

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + b^2 = 2 → 3*a + 4*b ≤ max) ∧ max = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3706_370636


namespace NUMINAMATH_CALUDE_digit2021_is_one_l3706_370649

/-- The sequence of digits formed by concatenating natural numbers starting from 1 -/
def digitSequence : ℕ → ℕ :=
  sorry

/-- The 2021st digit in the sequence -/
def digit2021 : ℕ := digitSequence 2021

theorem digit2021_is_one : digit2021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit2021_is_one_l3706_370649


namespace NUMINAMATH_CALUDE_cost_per_side_of_square_park_l3706_370624

/-- Represents the cost of fencing a square park -/
def CostOfFencing : Type :=
  { total : ℕ // total > 0 }

/-- Calculates the cost of fencing each side of a square park -/
def costPerSide (c : CostOfFencing) : ℕ :=
  c.val / 4

/-- Theorem: The cost of fencing each side of a square park is 43 dollars,
    given that the total cost of fencing is 172 dollars -/
theorem cost_per_side_of_square_park :
  ∀ (c : CostOfFencing), c.val = 172 → costPerSide c = 43 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_side_of_square_park_l3706_370624


namespace NUMINAMATH_CALUDE_bobs_age_l3706_370659

theorem bobs_age (alice_age bob_age : ℝ) : 
  bob_age = 3 * alice_age - 20 →
  bob_age + alice_age = 70 →
  bob_age = 47.5 := by
sorry

end NUMINAMATH_CALUDE_bobs_age_l3706_370659


namespace NUMINAMATH_CALUDE_house_transactions_result_l3706_370642

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  hasHouse : Bool

/-- Represents a house transaction between two people -/
def houseTransaction (buyer seller : FinancialState) (price : Int) : FinancialState × FinancialState :=
  (FinancialState.mk (buyer.cash - price) true, FinancialState.mk (seller.cash + price) false)

/-- The main theorem to prove -/
theorem house_transactions_result :
  let initialA := FinancialState.mk 15000 true
  let initialB := FinancialState.mk 16000 false
  let (a1, b1) := houseTransaction initialB initialA 16000
  let (a2, b2) := houseTransaction a1 b1 14000
  let (a3, b3) := houseTransaction b2 a2 17000
  a3.cash = 34000 ∧ b3.cash = -3000 := by
  sorry

#check house_transactions_result

end NUMINAMATH_CALUDE_house_transactions_result_l3706_370642


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3706_370674

/-- The minimum value of 2y^2 + x^2 given the system of equations and parameter range -/
theorem min_value_of_expression (a : ℝ) (x y : ℝ) 
  (h1 : a * x - y = 2 * a + 1)
  (h2 : -x + a * y = a)
  (h3 : -0.5 ≤ a ∧ a ≤ 2) :
  ∃ (min_val : ℝ), min_val = -2/9 ∧ 2 * y^2 + x^2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3706_370674


namespace NUMINAMATH_CALUDE_sum_of_vectors_is_zero_l3706_370698

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given vectors a, b, c in a real vector space V, prove that their sum is zero
    under the given conditions. -/
theorem sum_of_vectors_is_zero (a b c : V)
  (not_collinear_ab : ¬ Collinear ℝ ({0, a, b} : Set V))
  (not_collinear_bc : ¬ Collinear ℝ ({0, b, c} : Set V))
  (not_collinear_ca : ¬ Collinear ℝ ({0, c, a} : Set V))
  (collinear_ab_c : Collinear ℝ ({0, a + b, c} : Set V))
  (collinear_bc_a : Collinear ℝ ({0, b + c, a} : Set V)) :
  a + b + c = (0 : V) := by
sorry

end NUMINAMATH_CALUDE_sum_of_vectors_is_zero_l3706_370698


namespace NUMINAMATH_CALUDE_initial_conditions_squares_in_figure_100_l3706_370608

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 2 * n + 1

/-- The sequence satisfies the given initial conditions -/
theorem initial_conditions :
  f 0 = 1 ∧ f 1 = 6 ∧ f 2 = 17 ∧ f 3 = 34 := by sorry

/-- The number of squares in figure 100 is 30201 -/
theorem squares_in_figure_100 :
  f 100 = 30201 := by sorry

end NUMINAMATH_CALUDE_initial_conditions_squares_in_figure_100_l3706_370608


namespace NUMINAMATH_CALUDE_number_above_210_l3706_370645

/-- Calculates the tetrahedral number for a given k -/
def tetrahedralNumber (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6

/-- Calculates the starting number of the k-th row -/
def rowStart (k : ℕ) : ℕ := tetrahedralNumber (k - 1) + 1

/-- Calculates the ending number of the k-th row -/
def rowEnd (k : ℕ) : ℕ := tetrahedralNumber k

/-- The row in which 210 is located -/
def row210 : ℕ := 10

/-- The position of 210 in its row -/
def pos210 : ℕ := 210 - rowStart row210 + 1

theorem number_above_210 :
  rowEnd (row210 - 1) = 165 ∧ pos210 ≤ (row210 - 1) * row210 / 2 :=
by sorry

end NUMINAMATH_CALUDE_number_above_210_l3706_370645


namespace NUMINAMATH_CALUDE_surface_area_of_joined_cubes_l3706_370672

/-- The surface area of a cuboid formed by joining two cubes with side length b -/
def cuboid_surface_area (b : ℝ) : ℝ := 10 * b^2

/-- Theorem: The surface area of a cuboid formed by joining two cubes with side length b is 10b^2 -/
theorem surface_area_of_joined_cubes (b : ℝ) (h : b > 0) :
  cuboid_surface_area b = 2 * ((2*b * b) + (2*b * b) + (b * b)) :=
by sorry

end NUMINAMATH_CALUDE_surface_area_of_joined_cubes_l3706_370672


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l3706_370629

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x + 2) - (x + 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x + 2)*(x + 6) =
  6*x^3 + 20*x^2 + 6*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l3706_370629


namespace NUMINAMATH_CALUDE_circle_constraint_extrema_sum_l3706_370671

theorem circle_constraint_extrema_sum (x y : ℝ) :
  x^2 + y^2 = 1 →
  ∃ (min max : ℝ),
    (∀ x' y' : ℝ, x'^2 + y'^2 = 1 → 
      min ≤ (x'-3)^2 + (y'+4)^2 ∧ (x'-3)^2 + (y'+4)^2 ≤ max) ∧
    min + max = 52 := by
  sorry

end NUMINAMATH_CALUDE_circle_constraint_extrema_sum_l3706_370671


namespace NUMINAMATH_CALUDE_bunny_rate_is_three_l3706_370612

/-- The number of times a single bunny comes out of its burrow per minute. -/
def bunny_rate : ℕ := sorry

/-- The number of bunnies. -/
def num_bunnies : ℕ := 20

/-- The number of hours observed. -/
def observation_hours : ℕ := 10

/-- The total number of times bunnies come out in the observation period. -/
def total_exits : ℕ := 36000

/-- Proves that the bunny_rate is 3 given the conditions of the problem. -/
theorem bunny_rate_is_three : bunny_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_bunny_rate_is_three_l3706_370612


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l3706_370673

/-- Given a geometric sequence with first term a₁ = 1, 
    the minimum value of 6a₂ + 7a₃ is -9/7 -/
theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) : 
  a₁ = 1 → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) → 
  (∀ x y : ℝ, x = a₂ ∧ y = a₃ → 6*x + 7*y ≥ -9/7) ∧ 
  (∃ x y : ℝ, x = a₂ ∧ y = a₃ ∧ 6*x + 7*y = -9/7) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l3706_370673


namespace NUMINAMATH_CALUDE_expression_equals_one_l3706_370696

theorem expression_equals_one :
  |Real.sqrt 3 - 2| + (-1/2)⁻¹ + (2023 - Real.pi)^0 + 3 * Real.tan (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3706_370696


namespace NUMINAMATH_CALUDE_joyce_gave_six_pencils_l3706_370654

/-- The number of pencils Joyce gave to Eugene -/
def pencils_from_joyce (initial_pencils final_pencils : ℕ) : ℕ :=
  final_pencils - initial_pencils

/-- Theorem stating that Joyce gave Eugene 6 pencils -/
theorem joyce_gave_six_pencils (h1 : pencils_from_joyce 51 57 = 6) : 
  pencils_from_joyce 51 57 = 6 := by
  sorry

end NUMINAMATH_CALUDE_joyce_gave_six_pencils_l3706_370654


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3706_370676

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3706_370676


namespace NUMINAMATH_CALUDE_range_of_k_l3706_370640

noncomputable def h (x : ℝ) : ℝ := 5 * x + 3

noncomputable def k (x : ℝ) : ℝ := h (h (h x))

theorem range_of_k :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-32 : ℝ) 468,
  k x = y ∧
  ∀ z ∈ Set.Icc (-32 : ℝ) 468,
  ∃ w ∈ Set.Icc (-1 : ℝ) 3,
  k w = z :=
sorry

end NUMINAMATH_CALUDE_range_of_k_l3706_370640


namespace NUMINAMATH_CALUDE_config_7_3_1_wins_for_second_player_l3706_370668

/-- Represents the nim-value of a wall of bricks. -/
def nimValue (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | 6 => 3
  | 7 => 2
  | _ => 0  -- Default case, though not used in this problem

/-- Calculates the nim-sum (XOR) of a list of natural numbers. -/
def nimSum : List ℕ → ℕ
  | [] => 0
  | (x::xs) => x ^^^ (nimSum xs)

/-- Represents a configuration of walls in the game. -/
structure GameConfig where
  walls : List ℕ

/-- Determines if a given game configuration is a winning position for the second player. -/
def isWinningForSecondPlayer (config : GameConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The theorem stating that the configuration (7, 3, 1) is a winning position for the second player. -/
theorem config_7_3_1_wins_for_second_player :
  isWinningForSecondPlayer ⟨[7, 3, 1]⟩ := by sorry

end NUMINAMATH_CALUDE_config_7_3_1_wins_for_second_player_l3706_370668


namespace NUMINAMATH_CALUDE_tetrahedron_volume_order_l3706_370627

/-- Represents a triangle with side lengths a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2
  ordered : a > b ∧ b > c

/-- Represents the volumes of tetrahedrons formed by folding the triangle --/
structure TetrahedronVolumes where
  V₁ : ℝ
  V₂ : ℝ
  V₃ : ℝ

/-- Calculates the volumes of tetrahedrons formed by folding the triangle --/
def calculateVolumes (t : Triangle) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) : TetrahedronVolumes :=
  sorry

/-- Theorem: The volumes of tetrahedrons satisfy V₁ > V₂ > V₃ --/
theorem tetrahedron_volume_order (t : Triangle) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  let v := calculateVolumes t θ hθ
  v.V₁ > v.V₂ ∧ v.V₂ > v.V₃ := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_order_l3706_370627


namespace NUMINAMATH_CALUDE_inequality_chain_l3706_370622

theorem inequality_chain (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) :
  a > -b ∧ -b > b ∧ b > -a := by sorry

end NUMINAMATH_CALUDE_inequality_chain_l3706_370622


namespace NUMINAMATH_CALUDE_triangle_inequality_l3706_370623

theorem triangle_inequality (a b x : ℝ) : 
  (a = 3 ∧ b = 5) → (2 < x ∧ x < 8) → 
  (a + b > x ∧ b + x > a ∧ x + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3706_370623


namespace NUMINAMATH_CALUDE_paving_cost_l3706_370694

/-- The cost of paving a rectangular floor given its dimensions and the rate per square metre. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 8) (h2 : width = 4.75) (h3 : rate = 900) :
  length * width * rate = 34200 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l3706_370694


namespace NUMINAMATH_CALUDE_limit_sqrt_sum_to_infinity_l3706_370625

/-- The limit of n(√(n^2+1) + √(n^2-1)) as n approaches infinity is infinity. -/
theorem limit_sqrt_sum_to_infinity :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, n * (Real.sqrt (n^2 + 1) + Real.sqrt (n^2 - 1)) > ε :=
by sorry

end NUMINAMATH_CALUDE_limit_sqrt_sum_to_infinity_l3706_370625


namespace NUMINAMATH_CALUDE_f_8_equals_952_l3706_370620

def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 27*x^2 - 24*x - 72

theorem f_8_equals_952 : f 8 = 952 := by
  sorry

end NUMINAMATH_CALUDE_f_8_equals_952_l3706_370620


namespace NUMINAMATH_CALUDE_total_results_l3706_370619

theorem total_results (total_sum : ℕ) (total_count : ℕ) 
  (first_six_sum : ℕ) (last_six_sum : ℕ) (sixth_result : ℕ) :
  total_sum / total_count = 60 →
  first_six_sum = 6 * 58 →
  last_six_sum = 6 * 63 →
  sixth_result = 66 →
  total_sum = first_six_sum + last_six_sum - sixth_result →
  total_count = 11 := by
sorry

end NUMINAMATH_CALUDE_total_results_l3706_370619


namespace NUMINAMATH_CALUDE_intersection_area_theorem_l3706_370647

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- The area of intersection between a rectangle and a circle -/
def intersectionArea (rect : Rectangle) (circ : Circle) : ℝ := sorry

/-- The theorem stating the area of intersection between the specific rectangle and circle -/
theorem intersection_area_theorem :
  let rect : Rectangle := { x1 := 3, y1 := -3, x2 := 14, y2 := 10 }
  let circ : Circle := { center_x := 3, center_y := -3, radius := 4 }
  intersectionArea rect circ = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_intersection_area_theorem_l3706_370647


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3706_370680

/-- 
For a quadratic equation x^2 - 3x + c to have roots in the form x = (3 ± √(2c-3)) / 2, 
c must equal 2.
-/
theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ ∃ s : ℝ, s^2 = 2*c - 3 ∧ x = (3 + s) / 2 ∨ x = (3 - s) / 2) →
  c = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3706_370680


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3706_370603

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a4 : a 4 = 1) 
  (h_a7 : a 7 = 8) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3706_370603


namespace NUMINAMATH_CALUDE_first_12_average_l3706_370607

theorem first_12_average (total_count : Nat) (total_average : ℝ) (last_12_average : ℝ) (result_13 : ℝ) :
  total_count = 25 →
  total_average = 18 →
  last_12_average = 20 →
  result_13 = 90 →
  (((total_count : ℝ) * total_average - 12 * last_12_average - result_13) / 12 : ℝ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_12_average_l3706_370607


namespace NUMINAMATH_CALUDE_floor_composition_identity_l3706_370605

open Real

theorem floor_composition_identity (α : ℝ) (n : ℕ) (h : α > 1) :
  let β := 1 / α
  let fₐ (x : ℝ) := ⌊α * x + 1/2⌋
  let fᵦ (x : ℝ) := ⌊β * x + 1/2⌋
  fᵦ (fₐ n) = n := by
  sorry

end NUMINAMATH_CALUDE_floor_composition_identity_l3706_370605


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3706_370651

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def circularArrangements (n : ℕ) : ℕ := factorial (n - 1)

def adjacentPairArrangements (n : ℕ) : ℕ := 2 * factorial (n - 2)

theorem seating_arrangements_with_restriction (total : ℕ) (restricted_pair : ℕ) :
  total = 12 →
  restricted_pair = 2 →
  circularArrangements total - adjacentPairArrangements total = 32659200 := by
  sorry

#eval circularArrangements 12 - adjacentPairArrangements 12

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l3706_370651


namespace NUMINAMATH_CALUDE_cubic_coefficient_B_l3706_370641

/-- A cubic function with roots at -2 and 2, and value -1 at x = 0 -/
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

/-- Theorem stating that under given conditions, B = 1 -/
theorem cubic_coefficient_B (A B C D : ℝ) :
  g A B C D (-2) = 0 →
  g A B C D 0 = -1 →
  g A B C D 2 = 0 →
  B = 1 := by
    sorry

end NUMINAMATH_CALUDE_cubic_coefficient_B_l3706_370641


namespace NUMINAMATH_CALUDE_certain_number_exists_l3706_370686

theorem certain_number_exists : ∃ x : ℝ, 220050 = (555 + x) * (2 * (x - 555)) + 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l3706_370686


namespace NUMINAMATH_CALUDE_smallest_factors_for_square_and_cube_l3706_370606

theorem smallest_factors_for_square_and_cube (n : ℕ) (hn : n = 450) :
  ∃ (x y : ℕ),
    (∀ x' : ℕ, x' < x → ¬∃ k : ℕ, n * x' = k^2) ∧
    (∀ y' : ℕ, y' < y → ¬∃ k : ℕ, n * y' = k^3) ∧
    (∃ k : ℕ, n * x = k^2) ∧
    (∃ k : ℕ, n * y = k^3) ∧
    x = 2 ∧ y = 60 ∧ x + y = 62 :=
by sorry

end NUMINAMATH_CALUDE_smallest_factors_for_square_and_cube_l3706_370606


namespace NUMINAMATH_CALUDE_equation_equivalence_l3706_370643

theorem equation_equivalence (x y : ℝ) : 2 * x - y = 3 ↔ y = 2 * x - 3 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3706_370643
