import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_is_15_5_l1248_124801

/-- Triangle ABC inscribed in a rectangle --/
structure TriangleInRectangle where
  -- Rectangle dimensions
  width : ℝ
  height : ℝ
  -- Vertex positions
  a_height : ℝ
  b_distance : ℝ
  c_distance : ℝ
  -- Conditions
  width_positive : width > 0
  height_positive : height > 0
  a_height_valid : 0 < a_height ∧ a_height < height
  b_distance_valid : 0 < b_distance ∧ b_distance < width
  c_distance_valid : 0 < c_distance ∧ c_distance < height

/-- The area of triangle ABC --/
def triangleArea (t : TriangleInRectangle) : ℝ :=
  t.width * t.height - (0.5 * t.width * t.c_distance + 0.5 * (t.height - t.a_height) * t.width + 0.5 * t.b_distance * t.a_height)

/-- Theorem: The area of triangle ABC is 15.5 square units --/
theorem triangle_area_is_15_5 (t : TriangleInRectangle) 
    (h_width : t.width = 6)
    (h_height : t.height = 4)
    (h_a_height : t.a_height = 1)
    (h_b_distance : t.b_distance = 3)
    (h_c_distance : t.c_distance = 1) : 
  triangleArea t = 15.5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_is_15_5_l1248_124801


namespace NUMINAMATH_CALUDE_inequality_proof_l1248_124874

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow (a^2 / (b + c)^2) (1/3) + 
  Real.rpow (b^2 / (c + a)^2) (1/3) + 
  Real.rpow (c^2 / (a + b)^2) (1/3) ≥ 
  3 / Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1248_124874


namespace NUMINAMATH_CALUDE_percent_relation_l1248_124896

theorem percent_relation (x y z P : ℝ) 
  (hy : y = 0.75 * x) 
  (hz : z = 0.65 * x) 
  (hP : P / 100 * z = 0.39 * y) : 
  P = 45 := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1248_124896


namespace NUMINAMATH_CALUDE_valid_assignment_l1248_124825

/-- Represents the squares in the grid -/
inductive Square
| One | Nine | A | B | C | D | E | F | G

/-- Represents the direction of arrows -/
inductive Direction
| Right | RightUp | Up | Down

/-- Define the arrow directions for each square -/
def arrowDirection (s : Square) : Option Direction :=
  match s with
  | Square.One => some Direction.Right
  | Square.B => some Direction.RightUp
  | Square.E => some Direction.Right
  | Square.C => some Direction.Right
  | Square.D => some Direction.Up
  | Square.A => some Direction.Down
  | Square.G => some Direction.Right
  | Square.F => some Direction.Right
  | Square.Nine => none

/-- Define the next square based on the current square and arrow direction -/
def nextSquare (s : Square) : Option Square :=
  match s, arrowDirection s with
  | Square.One, some Direction.Right => some Square.B
  | Square.B, some Direction.RightUp => some Square.E
  | Square.E, some Direction.Right => some Square.C
  | Square.C, some Direction.Right => some Square.D
  | Square.D, some Direction.Up => some Square.A
  | Square.A, some Direction.Down => some Square.G
  | Square.G, some Direction.Right => some Square.F
  | Square.F, some Direction.Right => some Square.Nine
  | _, _ => none

/-- The assignment of numbers to squares -/
def assignment : Square → Nat
| Square.One => 1
| Square.Nine => 9
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

/-- Theorem stating that the assignment is valid -/
theorem valid_assignment :
  (∀ s : Square, s ≠ Square.One ∧ s ≠ Square.Nine →
    2 ≤ assignment s ∧ assignment s ≤ 8) ∧
  (∀ s : Square, s ≠ Square.Nine →
    ∃ next : Square, nextSquare s = some next ∧
    assignment next = assignment s + 1) :=
sorry

end NUMINAMATH_CALUDE_valid_assignment_l1248_124825


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1248_124888

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottomRadius : ℝ
  topRadius : ℝ
  sphereRadius : ℝ
  isTangent : Bool

/-- The theorem stating the radius of the sphere tangent to a specific truncated cone -/
theorem sphere_radius_in_truncated_cone
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottomRadius = 18)
  (h2 : cone.topRadius = 2)
  (h3 : cone.isTangent = true) :
  cone.sphereRadius = 6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1248_124888


namespace NUMINAMATH_CALUDE_count_solutions_eq_51_l1248_124864

/-- The number of distinct ordered pairs (a, b) of non-negative integers that satisfy a + b = 50 -/
def count_solutions : ℕ := 
  (Finset.range 51).card

/-- Theorem: There are 51 distinct ordered pairs (a, b) of non-negative integers that satisfy a + b = 50 -/
theorem count_solutions_eq_51 : count_solutions = 51 := by
  sorry

#eval count_solutions  -- This should output 51

end NUMINAMATH_CALUDE_count_solutions_eq_51_l1248_124864


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l1248_124890

/-- The cost price of the cupboard -/
def cost_price : ℝ := sorry

/-- The selling price of the cupboard -/
def selling_price : ℝ := 0.84 * cost_price

/-- The increased selling price -/
def increased_selling_price : ℝ := 1.16 * cost_price

theorem cupboard_cost_price : cost_price = 3750 := by
  have h1 : selling_price = 0.84 * cost_price := rfl
  have h2 : increased_selling_price = 1.16 * cost_price := rfl
  have h3 : increased_selling_price - selling_price = 1200 := sorry
  sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l1248_124890


namespace NUMINAMATH_CALUDE_total_expenditure_nine_persons_l1248_124800

/-- Given 9 persons, where 8 spend 30 Rs each and the 9th spends 20 Rs more than the average,
    prove that the total expenditure is 292.5 Rs -/
theorem total_expenditure_nine_persons :
  let num_persons : ℕ := 9
  let num_regular_spenders : ℕ := 8
  let regular_expenditure : ℚ := 30
  let extra_expenditure : ℚ := 20
  let total_expenditure : ℚ := num_regular_spenders * regular_expenditure +
    (((num_regular_spenders * regular_expenditure) / num_persons) + extra_expenditure)
  total_expenditure = 292.5 := by
sorry

end NUMINAMATH_CALUDE_total_expenditure_nine_persons_l1248_124800


namespace NUMINAMATH_CALUDE_correct_guesser_is_D_l1248_124876

-- Define the set of suspects
inductive Suspect : Type
| A | B | C | D | E | F

-- Define the set of passersby
inductive Passerby : Type
| A | B | C | D

-- Define a function to represent each passerby's guess
def guess (p : Passerby) (s : Suspect) : Prop :=
  match p with
  | Passerby.A => s = Suspect.D ∨ s = Suspect.E
  | Passerby.B => s ≠ Suspect.C
  | Passerby.C => s = Suspect.A ∨ s = Suspect.B ∨ s = Suspect.F
  | Passerby.D => s ≠ Suspect.D ∧ s ≠ Suspect.E ∧ s ≠ Suspect.F

-- Theorem statement
theorem correct_guesser_is_D :
  ∃! (thief : Suspect),
    ∃! (correct_passerby : Passerby),
      (∀ (p : Passerby), p ≠ correct_passerby → ¬guess p thief) ∧
      guess correct_passerby thief ∧
      correct_passerby = Passerby.D :=
by sorry

end NUMINAMATH_CALUDE_correct_guesser_is_D_l1248_124876


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l1248_124819

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l1248_124819


namespace NUMINAMATH_CALUDE_probability_r_successes_correct_l1248_124879

/-- The probability of exactly r successful shots by the time the nth shot is taken -/
def probability_r_successes (n r : ℕ) (p : ℝ) : ℝ :=
  Nat.choose (n - 1) (r - 1) * p ^ r * (1 - p) ^ (n - r)

/-- Theorem stating the probability of exactly r successful shots by the nth shot -/
theorem probability_r_successes_correct (n r : ℕ) (p : ℝ) 
    (h1 : 0 ≤ p) (h2 : p ≤ 1) (h3 : 1 ≤ r) (h4 : r ≤ n) : 
  probability_r_successes n r p = Nat.choose (n - 1) (r - 1) * p ^ r * (1 - p) ^ (n - r) :=
by sorry

end NUMINAMATH_CALUDE_probability_r_successes_correct_l1248_124879


namespace NUMINAMATH_CALUDE_sum_of_possible_intersection_counts_l1248_124893

/-- A configuration of five distinct lines in a plane -/
structure LineConfiguration where
  lines : Finset (Set ℝ × ℝ)
  distinct : lines.card = 5

/-- The number of distinct intersection points in a configuration -/
def intersectionPoints (config : LineConfiguration) : ℕ :=
  sorry

/-- The set of all possible values for the number of intersection points -/
def possibleIntersectionCounts : Finset ℕ :=
  sorry

/-- Theorem: The sum of all possible values for the number of intersection points is 53 -/
theorem sum_of_possible_intersection_counts :
  (possibleIntersectionCounts.sum id) = 53 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_intersection_counts_l1248_124893


namespace NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l1248_124833

-- Define the curve
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the point M
def M : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at M
def m : ℝ := (6 : ℝ) * M.1 - 4

-- Define the equation of the line
def line_equation (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Theorem statement
theorem line_passes_through_P_and_parallel_to_tangent :
  line_equation P.1 P.2 ∧
  (∀ x y : ℝ, line_equation x y → (y - P.2) = m * (x - P.1)) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_P_and_parallel_to_tangent_l1248_124833


namespace NUMINAMATH_CALUDE_solve_age_problem_l1248_124872

def age_problem (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9)

theorem solve_age_problem :
  ∃ a b : ℕ, age_problem a b ∧ b = 39 :=
sorry

end NUMINAMATH_CALUDE_solve_age_problem_l1248_124872


namespace NUMINAMATH_CALUDE_min_candies_removed_correct_l1248_124869

/-- Represents the number of candies of each flavor -/
structure CandyCounts where
  chocolate : Nat
  mint : Nat
  butterscotch : Nat

/-- The initial candy counts in the bag -/
def initialCandies : CandyCounts :=
  { chocolate := 4, mint := 6, butterscotch := 10 }

/-- The total number of candies in the bag -/
def totalCandies : Nat := 20

/-- The minimum number of candies that must be removed to ensure
    at least two of each flavor have been eaten -/
def minCandiesRemoved : Nat := 18

theorem min_candies_removed_correct :
  minCandiesRemoved = totalCandies - (initialCandies.chocolate - 2) - (initialCandies.mint - 2) - (initialCandies.butterscotch - 2) :=
by sorry

end NUMINAMATH_CALUDE_min_candies_removed_correct_l1248_124869


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1248_124860

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  neverNotOut : Bool

/-- Calculates the average runs per innings -/
def average (perf : BatsmanPerformance) : ℚ :=
  perf.totalRuns / perf.innings

/-- Represents the change in a batsman's performance after an additional innings -/
structure PerformanceChange where
  before : BatsmanPerformance
  runsScored : ℕ
  newAverage : ℚ

/-- Calculates the increase in average after an additional innings -/
def averageIncrease (change : PerformanceChange) : ℚ :=
  change.newAverage - average change.before

theorem batsman_average_increase :
  ∀ (perf : BatsmanPerformance) (change : PerformanceChange),
    perf.innings = 11 →
    perf.neverNotOut = true →
    change.before = perf →
    change.runsScored = 60 →
    change.newAverage = 38 →
    averageIncrease change = 2 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1248_124860


namespace NUMINAMATH_CALUDE_cost_of_12_pencils_9_notebooks_l1248_124822

/-- The cost of a single pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a single notebook -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 9 pencils and 6 notebooks cost $3.21 -/
axiom condition1 : 9 * pencil_cost + 6 * notebook_cost = 3.21

/-- The second given condition: 8 pencils and 5 notebooks cost $2.84 -/
axiom condition2 : 8 * pencil_cost + 5 * notebook_cost = 2.84

/-- Theorem: The cost of 12 pencils and 9 notebooks is $4.32 -/
theorem cost_of_12_pencils_9_notebooks : 
  12 * pencil_cost + 9 * notebook_cost = 4.32 := by sorry

end NUMINAMATH_CALUDE_cost_of_12_pencils_9_notebooks_l1248_124822


namespace NUMINAMATH_CALUDE_trapezium_side_length_l1248_124813

theorem trapezium_side_length (a b h : ℝ) (area : ℝ) : 
  a = 20 → h = 15 → area = 285 → area = (a + b) * h / 2 → b = 18 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l1248_124813


namespace NUMINAMATH_CALUDE_selection_schemes_eq_240_l1248_124881

-- Define the number of people and cities
def total_people : ℕ := 6
def total_cities : ℕ := 4

-- Define the function to calculate the number of selection schemes
def selection_schemes : ℕ :=
  -- Options for city A (excluding person A and B)
  (total_people - 2) *
  -- Options for city B
  (total_people - 1) *
  -- Options for city C
  (total_people - 2) *
  -- Options for city D
  (total_people - 3)

-- Theorem to prove
theorem selection_schemes_eq_240 : selection_schemes = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_eq_240_l1248_124881


namespace NUMINAMATH_CALUDE_share_percentage_problem_l1248_124849

theorem share_percentage_problem (total z y x : ℝ) : 
  total = 740 →
  z = 200 →
  y = 1.2 * z →
  x = total - y - z →
  (x - y) / y * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_share_percentage_problem_l1248_124849


namespace NUMINAMATH_CALUDE_x1_value_l1248_124820

theorem x1_value (x₁ x₂ x₃ : ℝ) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h2 : (1 - x₁)^2 + 2*(x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/2) :
  x₁ = (3 * Real.sqrt 2 - 3) / 7 := by
  sorry

end NUMINAMATH_CALUDE_x1_value_l1248_124820


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1248_124859

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem unique_solution_factorial_equation :
  ∃! n : ℕ, 3 * n * factorial n + 2 * factorial n = 40320 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l1248_124859


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1248_124815

theorem arithmetic_mean_of_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x^2 + a^2) / x^2 + (x^2 - a^2) / x^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1248_124815


namespace NUMINAMATH_CALUDE_simplify_expression_l1248_124807

theorem simplify_expression :
  81 * ((5 + 1/3) - (3 + 1/4)) / ((4 + 1/2) + (2 + 2/5)) = 225/92 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1248_124807


namespace NUMINAMATH_CALUDE_suwy_unique_product_l1248_124838

/-- Represents a letter with its corresponding value -/
structure Letter where
  value : Nat
  h : value ≥ 1 ∧ value ≤ 26

/-- Represents a four-letter list -/
structure FourLetterList where
  letters : Fin 4 → Letter

/-- Calculates the product of a four-letter list -/
def product (list : FourLetterList) : Nat :=
  (list.letters 0).value * (list.letters 1).value * (list.letters 2).value * (list.letters 3).value

theorem suwy_unique_product :
  ∀ (list : FourLetterList),
    product list = 19 * 21 * 23 * 25 →
    (list.letters 0).value = 19 ∧
    (list.letters 1).value = 21 ∧
    (list.letters 2).value = 23 ∧
    (list.letters 3).value = 25 :=
by sorry

end NUMINAMATH_CALUDE_suwy_unique_product_l1248_124838


namespace NUMINAMATH_CALUDE_red_spools_count_l1248_124889

-- Define the variables
def spools_per_beret : ℕ := 3
def black_spools : ℕ := 15
def blue_spools : ℕ := 6
def total_berets : ℕ := 11

-- Define the theorem
theorem red_spools_count : 
  ∃ (red_spools : ℕ), 
    red_spools + black_spools + blue_spools = spools_per_beret * total_berets ∧ 
    red_spools = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_spools_count_l1248_124889


namespace NUMINAMATH_CALUDE_twelfth_finger_number_l1248_124868

-- Define the function f
def f : ℕ → ℕ
| 4 => 7
| 7 => 8
| 8 => 3
| 3 => 5
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define a function to apply f n times
def apply_f_n_times (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => f (apply_f_n_times n x)

-- Theorem statement
theorem twelfth_finger_number : apply_f_n_times 11 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_finger_number_l1248_124868


namespace NUMINAMATH_CALUDE_perimeter_of_specific_hexagon_l1248_124832

-- Define the hexagon ABCDEF
structure RightAngledHexagon where
  AB : ℝ
  BC : ℝ
  EF : ℝ

-- Define the perimeter function
def perimeter (h : RightAngledHexagon) : ℝ :=
  2 * (h.AB + h.EF) + 2 * h.BC

-- Theorem statement
theorem perimeter_of_specific_hexagon :
  ∃ (h : RightAngledHexagon), h.AB = 8 ∧ h.BC = 15 ∧ h.EF = 5 ∧ perimeter h = 56 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_hexagon_l1248_124832


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_eighteen_l1248_124875

theorem arithmetic_expression_equals_eighteen :
  8 / 2 - 3 - 10 + 3 * 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_eighteen_l1248_124875


namespace NUMINAMATH_CALUDE_june_election_win_l1248_124856

theorem june_election_win (total_students : ℕ) (boy_percentage : ℚ) (male_vote_percentage : ℚ) :
  total_students = 200 →
  boy_percentage = 60 / 100 →
  male_vote_percentage = 675 / 1000 →
  ∃ (female_vote_percentage : ℚ),
    female_vote_percentage = 25 / 100 ∧
    (⌊total_students * boy_percentage⌋ : ℚ) * male_vote_percentage +
    (total_students - ⌊total_students * boy_percentage⌋ : ℚ) * female_vote_percentage >
    (total_students : ℚ) / 2 ∧
    ∀ (x : ℚ), x < female_vote_percentage →
      (⌊total_students * boy_percentage⌋ : ℚ) * male_vote_percentage +
      (total_students - ⌊total_students * boy_percentage⌋ : ℚ) * x ≤
      (total_students : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_june_election_win_l1248_124856


namespace NUMINAMATH_CALUDE_verbal_to_inequality_l1248_124884

/-- The inequality that represents "twice x plus 8 is less than five times x" -/
def twice_x_plus_8_less_than_5x (x : ℝ) : Prop :=
  2 * x + 8 < 5 * x

theorem verbal_to_inequality :
  ∀ x : ℝ, twice_x_plus_8_less_than_5x x ↔ (2 * x + 8 < 5 * x) :=
by
  sorry

#check verbal_to_inequality

end NUMINAMATH_CALUDE_verbal_to_inequality_l1248_124884


namespace NUMINAMATH_CALUDE_auditorium_sampling_is_systematic_l1248_124887

/-- Represents a sampling method --/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium with a given number of rows and seats per row --/
structure Auditorium where
  rows : ℕ
  seatsPerRow : ℕ

/-- Represents a sampling strategy --/
structure SamplingStrategy where
  interval : ℕ
  startingSeat : ℕ

/-- Determines if a sampling strategy is systematic --/
def isSystematicSampling (strategy : SamplingStrategy) : Prop :=
  strategy.interval > 0 ∧ strategy.startingSeat > 0 ∧ strategy.startingSeat ≤ strategy.interval

/-- The theorem to be proved --/
theorem auditorium_sampling_is_systematic 
  (auditorium : Auditorium) 
  (strategy : SamplingStrategy) : 
  auditorium.rows = 25 → 
  auditorium.seatsPerRow = 20 → 
  strategy.interval = auditorium.seatsPerRow → 
  strategy.startingSeat = 15 → 
  isSystematicSampling strategy ∧ 
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
sorry


end NUMINAMATH_CALUDE_auditorium_sampling_is_systematic_l1248_124887


namespace NUMINAMATH_CALUDE_truncated_pyramid_surface_area_l1248_124846

/-- The total surface area of a truncated right pyramid with given dimensions --/
theorem truncated_pyramid_surface_area
  (base_side : ℝ)
  (upper_side : ℝ)
  (height : ℝ)
  (h_base : base_side = 15)
  (h_upper : upper_side = 10)
  (h_height : height = 20) :
  let slant_height := Real.sqrt (height^2 + ((base_side - upper_side) / 2)^2)
  let lateral_area := 2 * (base_side + upper_side) * slant_height
  let base_area := base_side^2 + upper_side^2
  lateral_area + base_area = 1332.8 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_surface_area_l1248_124846


namespace NUMINAMATH_CALUDE_sum_a_c_equals_five_l1248_124897

theorem sum_a_c_equals_five 
  (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by sorry

end NUMINAMATH_CALUDE_sum_a_c_equals_five_l1248_124897


namespace NUMINAMATH_CALUDE_total_undeveloped_area_is_18750_l1248_124824

/-- The number of undeveloped land sections -/
def num_sections : ℕ := 5

/-- The area of each undeveloped land section in square feet -/
def area_per_section : ℕ := 3750

/-- The total area of undeveloped land in square feet -/
def total_undeveloped_area : ℕ := num_sections * area_per_section

/-- Theorem stating that the total area of undeveloped land is 18,750 square feet -/
theorem total_undeveloped_area_is_18750 : total_undeveloped_area = 18750 := by
  sorry

end NUMINAMATH_CALUDE_total_undeveloped_area_is_18750_l1248_124824


namespace NUMINAMATH_CALUDE_chocolates_bought_l1248_124847

theorem chocolates_bought (cost_price selling_price : ℝ) (num_bought : ℕ) : 
  (num_bought * cost_price = 21 * selling_price) →
  ((selling_price - cost_price) / cost_price * 100 = 66.67) →
  num_bought = 35 := by
sorry

end NUMINAMATH_CALUDE_chocolates_bought_l1248_124847


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l1248_124898

-- Define a monomial type
structure Monomial where
  coefficient : ℤ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
def our_monomial : Monomial := ⟨-3, 2, 1⟩

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l1248_124898


namespace NUMINAMATH_CALUDE_triangle_side_length_l1248_124880

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  A = π/4 →
  2*b*(Real.sin B) - c*(Real.sin C) = 2*a*(Real.sin A) →
  (1/2)*b*c*(Real.sin A) = 3 →
  c = 2*(Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1248_124880


namespace NUMINAMATH_CALUDE_last_two_digits_of_product_squared_l1248_124891

theorem last_two_digits_of_product_squared : 
  (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349)^2 % 100 = 76 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_product_squared_l1248_124891


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1248_124852

/-- The quadratic function f(x) = -x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc a 2, f x ≤ 15/4) ∧ 
  (∃ x ∈ Set.Icc a 2, f x = 15/4) →
  a = -1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1248_124852


namespace NUMINAMATH_CALUDE_single_female_fraction_l1248_124854

theorem single_female_fraction (total : ℕ) (h1 : total > 0) :
  let male_percent : ℚ := 70 / 100
  let married_percent : ℚ := 30 / 100
  let male_married_fraction : ℚ := 1 / 7
  let male_count := (male_percent * total).floor
  let female_count := total - male_count
  let married_count := (married_percent * total).floor
  let male_married_count := (male_married_fraction * male_count).floor
  let female_married_count := married_count - male_married_count
  let single_female_count := female_count - female_married_count
  (single_female_count : ℚ) / female_count = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_single_female_fraction_l1248_124854


namespace NUMINAMATH_CALUDE_presents_difference_l1248_124851

def ethan_presents : ℝ := 31.0
def alissa_presents : ℕ := 9

theorem presents_difference : ethan_presents - alissa_presents = 22 := by
  sorry

end NUMINAMATH_CALUDE_presents_difference_l1248_124851


namespace NUMINAMATH_CALUDE_highlight_film_average_time_l1248_124827

/-- The average time each player gets in the highlight film -/
def average_time (durations : List Nat) : Rat :=
  (durations.sum / 60) / durations.length

/-- Theorem: Given the video durations for 5 players, the average time each player gets is 2 minutes -/
theorem highlight_film_average_time :
  let durations := [130, 145, 85, 60, 180]
  average_time durations = 2 := by sorry

end NUMINAMATH_CALUDE_highlight_film_average_time_l1248_124827


namespace NUMINAMATH_CALUDE_helmet_safety_analysis_l1248_124886

-- Define the data types
structure YearData where
  year_number : ℕ
  not_wearing_helmets : ℕ

-- Define the data for 4 years
def year_data : List YearData := [
  ⟨1, 1250⟩,
  ⟨2, 1050⟩,
  ⟨3, 1000⟩,
  ⟨4, 900⟩
]

-- Define the contingency table
structure ContingencyTable where
  injured_not_wearing : ℕ
  injured_wearing : ℕ
  not_injured_not_wearing : ℕ
  not_injured_wearing : ℕ

def accident_data : ContingencyTable := ⟨7, 3, 13, 27⟩

-- Define the theorem
theorem helmet_safety_analysis :
  -- Regression line equation
  let b : ℚ := -110
  let a : ℚ := 1325
  let regression_line (x : ℚ) := b * x + a

  -- Estimated number of people not wearing helmets in 2022
  let estimate_2022 : ℕ := 775

  -- Chi-square statistic
  let chi_square : ℚ := 4.6875
  let critical_value : ℚ := 3.841

  -- Theorem statements
  (∀ (x : ℚ), regression_line x = b * x + a) ∧
  (regression_line 5 = estimate_2022) ∧
  (chi_square > critical_value) := by
  sorry


end NUMINAMATH_CALUDE_helmet_safety_analysis_l1248_124886


namespace NUMINAMATH_CALUDE_subcommittee_count_l1248_124892

theorem subcommittee_count (total_members : ℕ) (subcommittees_per_member : ℕ) (members_per_subcommittee : ℕ) :
  total_members = 360 →
  subcommittees_per_member = 3 →
  members_per_subcommittee = 6 →
  (total_members * subcommittees_per_member) / members_per_subcommittee = 180 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1248_124892


namespace NUMINAMATH_CALUDE_number_of_boys_l1248_124843

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 50 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1248_124843


namespace NUMINAMATH_CALUDE_added_number_problem_l1248_124828

theorem added_number_problem (initial_count : ℕ) (initial_avg : ℚ) (new_avg : ℚ) : 
  initial_count = 6 →
  initial_avg = 24 →
  new_avg = 25 →
  ∃ x : ℚ, (initial_count * initial_avg + x) / (initial_count + 1) = new_avg ∧ x = 31 :=
by sorry

end NUMINAMATH_CALUDE_added_number_problem_l1248_124828


namespace NUMINAMATH_CALUDE_fourth_term_is_seven_l1248_124866

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2

-- Define the sequence term function
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Theorem statement
theorem fourth_term_is_seven : a 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_seven_l1248_124866


namespace NUMINAMATH_CALUDE_blue_regular_polygon_l1248_124850

/-- A circle with some red points and the rest blue -/
structure ColoredCircle where
  redPoints : Finset ℝ
  (red_count : redPoints.card = 2016)

/-- A regular n-gon inscribed in a circle -/
structure RegularPolygon (n : ℕ) where
  vertices : Finset ℝ
  (vertex_count : vertices.card = n)

/-- The theorem statement -/
theorem blue_regular_polygon
  (circle : ColoredCircle)
  (n : ℕ)
  (h : n ≥ 3) :
  ∃ (poly : RegularPolygon n), poly.vertices ∩ circle.redPoints = ∅ :=
sorry

end NUMINAMATH_CALUDE_blue_regular_polygon_l1248_124850


namespace NUMINAMATH_CALUDE_fruit_cost_theorem_l1248_124839

def calculate_fruit_cost (quantity : ℝ) (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let cost_before_discount := quantity * price
  let discounted_cost := cost_before_discount * (1 - discount)
  let tax_amount := discounted_cost * tax
  discounted_cost + tax_amount

def grapes_cost := calculate_fruit_cost 8 70 0.1 0.05
def mangoes_cost := calculate_fruit_cost 9 65 0.05 0.06
def oranges_cost := calculate_fruit_cost 6 60 0 0.03
def apples_cost := calculate_fruit_cost 4 80 0.12 0.07

def total_cost := grapes_cost + mangoes_cost + oranges_cost + apples_cost

theorem fruit_cost_theorem : total_cost = 1790.407 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_theorem_l1248_124839


namespace NUMINAMATH_CALUDE_childs_running_speed_l1248_124830

/-- Proves that the child's running speed on a still sidewalk is 74 m/min given the problem conditions -/
theorem childs_running_speed 
  (speed_still : ℝ) 
  (sidewalk_speed : ℝ) 
  (distance_against : ℝ) 
  (time_against : ℝ) 
  (h1 : speed_still = 74) 
  (h2 : distance_against = 165) 
  (h3 : time_against = 3) 
  (h4 : (speed_still - sidewalk_speed) * time_against = distance_against) : 
  speed_still = 74 := by
sorry

end NUMINAMATH_CALUDE_childs_running_speed_l1248_124830


namespace NUMINAMATH_CALUDE_at_least_one_third_l1248_124855

theorem at_least_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_third_l1248_124855


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l1248_124848

theorem bottle_cap_distribution (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) : 
  total_caps = 35 → num_groups = 7 → total_caps = num_groups * caps_per_group → caps_per_group = 5 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l1248_124848


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1248_124863

theorem expression_equals_zero : 2 * 2^5 - 8^58 / 8^56 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1248_124863


namespace NUMINAMATH_CALUDE_smallest_value_w4_plus_z4_l1248_124811

theorem smallest_value_w4_plus_z4 (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 10) :
  Complex.abs (w^4 + z^4) = 82 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_w4_plus_z4_l1248_124811


namespace NUMINAMATH_CALUDE_employed_females_percentage_l1248_124858

theorem employed_females_percentage (total_employed_percent : ℝ) (employed_males_percent : ℝ)
  (h1 : total_employed_percent = 64)
  (h2 : employed_males_percent = 48) :
  (total_employed_percent - employed_males_percent) / total_employed_percent * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l1248_124858


namespace NUMINAMATH_CALUDE_solve_equation_l1248_124816

theorem solve_equation (x : ℚ) (h : x / 4 - x - 3 / 6 = 1) : x = -14/9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1248_124816


namespace NUMINAMATH_CALUDE_large_puzzle_cost_l1248_124870

theorem large_puzzle_cost (small large : ℝ) 
  (h1 : small + large = 23)
  (h2 : large + 3 * small = 39) : 
  large = 15 := by
sorry

end NUMINAMATH_CALUDE_large_puzzle_cost_l1248_124870


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1248_124853

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d = 3)
  (h3 : a 4 = 14) :
  ∀ n, a n = 3 * n + 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1248_124853


namespace NUMINAMATH_CALUDE_correct_share_distribution_l1248_124814

def total_amount : ℕ := 12000
def ratio : List ℕ := [2, 4, 6, 3, 5]

def share_amount (total : ℕ) (ratios : List ℕ) : List ℕ :=
  let total_parts := ratios.sum
  let part_value := total / total_parts
  ratios.map (· * part_value)

theorem correct_share_distribution :
  share_amount total_amount ratio = [1200, 2400, 3600, 1800, 3000] := by
  sorry

end NUMINAMATH_CALUDE_correct_share_distribution_l1248_124814


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1248_124812

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- The volume of a rectangular solid is the product of its length, width, and height. -/
def volume (l w h : ℕ) : ℕ := l * w * h

/-- The surface area of a rectangular solid is given by 2(lw + wh + hl). -/
def surfaceArea (l w h : ℕ) : ℕ := 2 * (l * w + w * h + h * l)

theorem rectangular_solid_surface_area 
  (l w h : ℕ) 
  (hl : isPrime l) 
  (hw : isPrime w) 
  (hh : isPrime h) 
  (hv : volume l w h = 437) : 
  surfaceArea l w h = 958 := by
  sorry

#check rectangular_solid_surface_area

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1248_124812


namespace NUMINAMATH_CALUDE_crop_planting_problem_l1248_124804

/-- Cost function for planting crops -/
def cost_function (x : ℝ) : ℝ := x^2 + 5*x + 10

/-- Revenue function for planting crops -/
def revenue_function (x : ℝ) : ℝ := 15*x

/-- Profit function for planting crops -/
def profit_function (x : ℝ) : ℝ := revenue_function x - cost_function x

theorem crop_planting_problem :
  (cost_function 1 = 16 ∧ cost_function 3 = 34) ∧
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ cost_function x₁ / x₁ = 12 ∧ cost_function x₂ / x₂ = 12) ∧
  (∃ x_max : ℝ, x_max = 5 ∧ 
    ∀ x : ℝ, profit_function x ≤ profit_function x_max ∧ 
    profit_function x_max = 15) :=
by sorry

#check crop_planting_problem

end NUMINAMATH_CALUDE_crop_planting_problem_l1248_124804


namespace NUMINAMATH_CALUDE_tinas_trip_distance_l1248_124834

-- Define the total distance of Tina's trip
def total_distance : ℝ := 120

-- Define the distance driven through the city
def city_distance : ℝ := 30

-- Theorem stating that the total distance satisfies the given conditions
theorem tinas_trip_distance : 
  total_distance / 2 + city_distance + total_distance / 4 = total_distance :=
by sorry

end NUMINAMATH_CALUDE_tinas_trip_distance_l1248_124834


namespace NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l1248_124894

theorem max_value_sum_of_square_roots (x : ℝ) (h : x ∈ Set.Icc (-36) 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_square_roots_l1248_124894


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1248_124802

theorem sphere_surface_area (V : ℝ) (S : ℝ) : 
  V = (32 / 3) * Real.pi → S = 4 * Real.pi * ((3 * V) / (4 * Real.pi))^(2/3) → S = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1248_124802


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1248_124844

/-- The speed of a train given the lengths of two trains, the speed of the other train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length1 : ℝ) (length2 : ℝ) (speed2 : ℝ) (cross_time : ℝ) :
  length1 = 270 →
  length2 = 230 →
  speed2 = 80 →
  cross_time = 9 / 3600 →
  (length1 + length2) / 1000 / cross_time - speed2 = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1248_124844


namespace NUMINAMATH_CALUDE_log_equation_solution_l1248_124810

theorem log_equation_solution (a b c x : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.log x = Real.log a + 3 * Real.log b - 5 * Real.log c →
  x = a * b^3 / c^5 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1248_124810


namespace NUMINAMATH_CALUDE_tan_810_degrees_undefined_l1248_124878

theorem tan_810_degrees_undefined : 
  ¬∃ (x : ℝ), Real.tan (810 * π / 180) = x :=
by
  sorry

end NUMINAMATH_CALUDE_tan_810_degrees_undefined_l1248_124878


namespace NUMINAMATH_CALUDE_one_intersection_point_condition_l1248_124817

open Real

noncomputable def f (x : ℝ) : ℝ := x + log x - 2 / Real.exp 1

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m / x

theorem one_intersection_point_condition (m : ℝ) :
  (∃! x, f x = g m x) →
  (m ≥ 0 ∨ m = -(Real.exp 1 + 1) / (Real.exp 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_one_intersection_point_condition_l1248_124817


namespace NUMINAMATH_CALUDE_a_plus_b_equals_five_l1248_124818

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem a_plus_b_equals_five (a b : ℝ) :
  (A ∪ B a b = Set.univ) →
  (A ∩ B a b = {x | 3 < x ∧ x ≤ 4}) →
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_five_l1248_124818


namespace NUMINAMATH_CALUDE_incorrect_statements_l1248_124842

theorem incorrect_statements : 
  let statement1 := (∃ a b : ℚ, a + b = 5 ∧ a + b = -3)
  let statement2 := (∀ x : ℝ, ∃ q : ℚ, x = q)
  let statement3 := (∀ x : ℝ, |x| > 0)
  let statement4 := (∀ x : ℝ, x * x = x → (x = 0 ∨ x = 1))
  let statement5 := (∀ a b : ℚ, a + b = 0 → (a > 0 ∨ b > 0))
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 ∧ ¬statement5) := by sorry

end NUMINAMATH_CALUDE_incorrect_statements_l1248_124842


namespace NUMINAMATH_CALUDE_junior_freshman_ratio_l1248_124841

theorem junior_freshman_ratio (f j : ℕ) (hf : f > 0) (hj : j > 0)
  (h_participants : (1 : ℚ) / 4 * f = (1 : ℚ) / 2 * j) :
  j / f = (1 : ℚ) / 2 := by
sorry

end NUMINAMATH_CALUDE_junior_freshman_ratio_l1248_124841


namespace NUMINAMATH_CALUDE_project_time_calculation_l1248_124877

theorem project_time_calculation (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  y = (3 * x) / 2 →
  z = 2 * x →
  z = x + 20 →
  x + y + z = 90 :=
by sorry

end NUMINAMATH_CALUDE_project_time_calculation_l1248_124877


namespace NUMINAMATH_CALUDE_dave_guitar_strings_l1248_124836

/-- The number of guitar strings Dave breaks per night -/
def strings_per_night : ℕ := 2

/-- The number of shows Dave performs per week -/
def shows_per_week : ℕ := 6

/-- The number of weeks Dave performs -/
def total_weeks : ℕ := 12

/-- The total number of guitar strings Dave needs to replace -/
def total_strings_replaced : ℕ := strings_per_night * shows_per_week * total_weeks

theorem dave_guitar_strings :
  total_strings_replaced = 144 :=
by sorry

end NUMINAMATH_CALUDE_dave_guitar_strings_l1248_124836


namespace NUMINAMATH_CALUDE_translation_problem_l1248_124861

def complex_translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) (h : t (1 + 3*I) = 4 + 7*I) :
  t (2 + 6*I) = 5 + 10*I :=
by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l1248_124861


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1248_124831

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1248_124831


namespace NUMINAMATH_CALUDE_product_unit_digit_l1248_124867

-- Define a function to get the unit digit of a number
def unitDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem product_unit_digit : 
  unitDigit (624 * 708 * 913 * 463) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_unit_digit_l1248_124867


namespace NUMINAMATH_CALUDE_prob_same_gender_two_schools_l1248_124899

/-- Represents a school with a certain number of male and female teachers -/
structure School where
  male_teachers : ℕ
  female_teachers : ℕ

/-- Calculates the total number of teachers in a school -/
def total_teachers (s : School) : ℕ := s.male_teachers + s.female_teachers

/-- Calculates the probability of selecting two teachers of the same gender from two schools -/
def prob_same_gender (s1 s2 : School) : ℚ :=
  let total_outcomes := (total_teachers s1) * (total_teachers s2)
  let same_gender_outcomes := s1.male_teachers * s2.male_teachers + s1.female_teachers * s2.female_teachers
  same_gender_outcomes / total_outcomes

theorem prob_same_gender_two_schools :
  let school_A : School := ⟨2, 1⟩
  let school_B : School := ⟨1, 2⟩
  prob_same_gender school_A school_B = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_gender_two_schools_l1248_124899


namespace NUMINAMATH_CALUDE_zeta_sum_sixth_power_l1248_124806

theorem zeta_sum_sixth_power (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h4 : ζ₁^4 + ζ₂^4 + ζ₃^4 = 29) :
  ζ₁^6 + ζ₂^6 + ζ₃^6 = 101.40625 := by
  sorry

end NUMINAMATH_CALUDE_zeta_sum_sixth_power_l1248_124806


namespace NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l1248_124808

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem decreasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/7 : ℝ) (1/3 : ℝ) ∧ a ≠ 1/3 := by
  sorry


end NUMINAMATH_CALUDE_decreasing_f_implies_a_range_l1248_124808


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1248_124865

-- Define the floor function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem unique_solution_floor_equation :
  ∃! x : ℝ, (floor (x - 1/2) : ℝ) = 3*x - 5 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1248_124865


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_interval_l1248_124845

/-- The function f(x) defined in terms of a real parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x * |2 * a - x| + 2 * x

/-- The theorem stating that f(x) is increasing on ℝ if and only if a ∈ [-1, 1] -/
theorem f_increasing_iff_a_in_interval (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_interval_l1248_124845


namespace NUMINAMATH_CALUDE_parabola_translation_correct_l1248_124895

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = x^2 -/
def originalParabola : Parabola := { a := 1, b := 0, c := 0 }

/-- The translation of 1 unit right and 2 units up -/
def givenTranslation : Translation := { dx := 1, dy := 2 }

/-- Function to apply a translation to a parabola -/
def applyTranslation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx + p.b
    c := p.a * t.dx^2 - p.b * t.dx + p.c + t.dy }

theorem parabola_translation_correct :
  applyTranslation originalParabola givenTranslation = { a := 1, b := -2, c := 3 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_correct_l1248_124895


namespace NUMINAMATH_CALUDE_club_participation_theorem_l1248_124826

universe u

def club_participation (n : ℕ) : Prop :=
  ∃ (U A B C : Finset ℕ),
    Finset.card U = 40 ∧
    Finset.card A = 22 ∧
    Finset.card B = 16 ∧
    Finset.card C = 20 ∧
    Finset.card (A ∩ B) = 8 ∧
    Finset.card (B ∩ C) = 6 ∧
    Finset.card (A ∩ C) = 10 ∧
    Finset.card (A ∩ B ∩ C) = 2 ∧
    Finset.card (A \ (B ∪ C) ∪ B \ (A ∪ C) ∪ C \ (A ∪ B)) = 16 ∧
    Finset.card (U \ (A ∪ B ∪ C)) = 4

theorem club_participation_theorem : club_participation 40 := by
  sorry

end NUMINAMATH_CALUDE_club_participation_theorem_l1248_124826


namespace NUMINAMATH_CALUDE_cube_coloring_count_l1248_124821

/-- Represents a coloring scheme for a cube -/
structure CubeColoring where
  /-- The number of faces on the cube -/
  faces : Nat
  /-- The number of available colors -/
  colors : Nat
  /-- The number of faces already colored -/
  colored_faces : Nat
  /-- Function to check if a coloring scheme is valid -/
  is_valid : (List Nat) → Bool

/-- Counts the number of valid coloring schemes for a cube -/
def count_valid_colorings (c : CubeColoring) : Nat :=
  sorry

/-- Theorem stating that there are exactly 13 valid coloring schemes for a cube
    with 6 faces, 5 colors, and 3 faces already colored -/
theorem cube_coloring_count :
  ∃ (c : CubeColoring),
    c.faces = 6 ∧
    c.colors = 5 ∧
    c.colored_faces = 3 ∧
    count_valid_colorings c = 13 :=
  sorry

end NUMINAMATH_CALUDE_cube_coloring_count_l1248_124821


namespace NUMINAMATH_CALUDE_bee_speed_solution_l1248_124883

/-- The speed of a honey bee's flight between flowers -/
def bee_speed_problem (time_daisy_rose time_rose_poppy : ℝ) 
  (distance_difference speed_difference : ℝ) : Prop :=
  let speed_daisy_rose : ℝ := 6.5
  let speed_rose_poppy : ℝ := speed_daisy_rose + speed_difference
  let distance_daisy_rose : ℝ := speed_daisy_rose * time_daisy_rose
  let distance_rose_poppy : ℝ := speed_rose_poppy * time_rose_poppy
  distance_daisy_rose = distance_rose_poppy + distance_difference ∧
  speed_daisy_rose = 6.5

theorem bee_speed_solution :
  bee_speed_problem 10 6 8 3 := by
  sorry

#check bee_speed_solution

end NUMINAMATH_CALUDE_bee_speed_solution_l1248_124883


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l1248_124862

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x < 0) ↔ (∀ x : ℝ, f x ≥ 0) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l1248_124862


namespace NUMINAMATH_CALUDE_max_fourth_term_arithmetic_seq_l1248_124882

/-- Given a sequence of six positive integers in arithmetic progression with a sum of 90,
    the maximum possible value of the fourth term is 17. -/
theorem max_fourth_term_arithmetic_seq : ∀ (a d : ℕ),
  a > 0 → d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 90 →
  a + 3*d ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_term_arithmetic_seq_l1248_124882


namespace NUMINAMATH_CALUDE_chord_length_is_four_l1248_124805

/-- A circle with center at (0, 1) and radius 2, tangent to the line y = -1 -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  center_eq : center = (0, 1)
  radius_eq : radius = 2
  tangent_to_line : ∀ (x y : ℝ), y = -1 → (x - center.1)^2 + (y - center.2)^2 ≥ radius^2

/-- The length of the chord intercepted by the circle on the y-axis -/
def chord_length (c : Circle) : ℝ :=
  let y₁ := c.center.2 + c.radius
  let y₂ := c.center.2 - c.radius
  y₁ - y₂

theorem chord_length_is_four (c : Circle) : chord_length c = 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l1248_124805


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1248_124829

/-- Given a rectangle with an inscribed circle of radius 7 and a length-to-width ratio of 3:1, 
    prove that its area is 588. -/
theorem rectangle_area_with_inscribed_circle (r w l : ℝ) : 
  r = 7 ∧ w = 2 * r ∧ l = 3 * w → l * w = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l1248_124829


namespace NUMINAMATH_CALUDE_latin_speakers_l1248_124809

/-- In a group of people, given the total number, the number of French speakers,
    the number of people speaking neither Latin nor French, and the number of people
    speaking both Latin and French, we can determine the number of Latin speakers. -/
theorem latin_speakers (total : ℕ) (french : ℕ) (neither : ℕ) (both : ℕ) :
  total = 25 →
  french = 15 →
  neither = 6 →
  both = 9 →
  ∃ latin : ℕ, latin = 13 ∧ latin + french - both = total - neither :=
by sorry

end NUMINAMATH_CALUDE_latin_speakers_l1248_124809


namespace NUMINAMATH_CALUDE_five_digit_palindromes_count_l1248_124885

/-- A function that counts the number of 5-digit palindromes -/
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10

/-- Theorem stating that the number of 5-digit palindromes is 900 -/
theorem five_digit_palindromes_count :
  count_five_digit_palindromes = 900 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_palindromes_count_l1248_124885


namespace NUMINAMATH_CALUDE_baker_baking_soda_l1248_124871

/-- The number of boxes of baking soda bought by the baker -/
def baking_soda_boxes : ℕ := sorry

/-- The cost of one box of flour in dollars -/
def flour_cost : ℕ := 3

/-- The number of boxes of flour bought -/
def flour_boxes : ℕ := 3

/-- The cost of one tray of eggs in dollars -/
def eggs_cost : ℕ := 10

/-- The number of trays of eggs bought -/
def eggs_trays : ℕ := 3

/-- The cost of one liter of milk in dollars -/
def milk_cost : ℕ := 5

/-- The number of liters of milk bought -/
def milk_liters : ℕ := 7

/-- The cost of one box of baking soda in dollars -/
def baking_soda_cost : ℕ := 3

/-- The total cost of all items in dollars -/
def total_cost : ℕ := 80

theorem baker_baking_soda : baking_soda_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_baker_baking_soda_l1248_124871


namespace NUMINAMATH_CALUDE_morios_current_age_l1248_124837

/-- Calculates Morio's current age given the ages of Teresa and Morio at different points in time. -/
theorem morios_current_age
  (teresa_current_age : ℕ)
  (morio_age_at_michikos_birth : ℕ)
  (teresa_age_at_michikos_birth : ℕ)
  (h1 : teresa_current_age = 59)
  (h2 : morio_age_at_michikos_birth = 38)
  (h3 : teresa_age_at_michikos_birth = 26) :
  morio_age_at_michikos_birth + (teresa_current_age - teresa_age_at_michikos_birth) = 71 :=
by sorry

end NUMINAMATH_CALUDE_morios_current_age_l1248_124837


namespace NUMINAMATH_CALUDE_megan_fourth_game_score_l1248_124840

/-- Represents Megan's basketball scores --/
structure MeganScores where
  threeGameAverage : ℝ
  fourGameAverage : ℝ

/-- Calculates Megan's score in the fourth game --/
def fourthGameScore (scores : MeganScores) : ℝ :=
  4 * scores.fourGameAverage - 3 * scores.threeGameAverage

/-- Theorem stating Megan's score in the fourth game --/
theorem megan_fourth_game_score :
  ∀ (scores : MeganScores),
    scores.threeGameAverage = 18 →
    scores.fourGameAverage = 17 →
    fourthGameScore scores = 14 := by
  sorry

#eval fourthGameScore { threeGameAverage := 18, fourGameAverage := 17 }

end NUMINAMATH_CALUDE_megan_fourth_game_score_l1248_124840


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1248_124873

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (100 - x) = 9 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1248_124873


namespace NUMINAMATH_CALUDE_brads_speed_l1248_124835

/-- Proves that Brad's running speed is 3 km/h given the conditions of the problem -/
theorem brads_speed (maxwell_speed : ℝ) (total_distance : ℝ) (maxwell_distance : ℝ) :
  maxwell_speed = 2 →
  total_distance = 65 →
  maxwell_distance = 26 →
  (total_distance - maxwell_distance) / (maxwell_distance / maxwell_speed) = 3 :=
by sorry

end NUMINAMATH_CALUDE_brads_speed_l1248_124835


namespace NUMINAMATH_CALUDE_double_area_right_triangle_l1248_124823

/-- The area of a triangle with double the area of a right-angled triangle -/
theorem double_area_right_triangle (a b : ℝ) : 
  let triangle_I_base : ℝ := a + b
  let triangle_I_height : ℝ := a + b
  let triangle_I_area : ℝ := (1 / 2) * triangle_I_base * triangle_I_height
  let triangle_II_area : ℝ := 2 * triangle_I_area
  triangle_II_area = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_double_area_right_triangle_l1248_124823


namespace NUMINAMATH_CALUDE_dogs_running_l1248_124857

theorem dogs_running (total : ℕ) (playing : ℕ) (barking : ℕ) (idle : ℕ) : 
  total = 88 →
  playing = total / 2 →
  barking = total / 4 →
  idle = 10 →
  total - playing - barking - idle = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_dogs_running_l1248_124857


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1248_124803

theorem sqrt_sum_comparison : Real.sqrt 2 + Real.sqrt 10 < 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1248_124803
