import Mathlib

namespace remove_32_toothpicks_eliminates_triangles_l1252_125240

/-- A triangular figure constructed with toothpicks -/
structure TriangularFigure where
  toothpicks : ℕ
  triangles : ℕ
  horizontal_toothpicks : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (f : TriangularFigure) : ℕ :=
  f.horizontal_toothpicks

/-- Theorem stating that removing 32 toothpicks is sufficient to eliminate all triangles 
    in a specific triangular figure -/
theorem remove_32_toothpicks_eliminates_triangles (f : TriangularFigure) 
  (h1 : f.toothpicks = 42)
  (h2 : f.triangles > 35)
  (h3 : f.horizontal_toothpicks = 32) :
  min_toothpicks_to_remove f = 32 := by
  sorry

end remove_32_toothpicks_eliminates_triangles_l1252_125240


namespace fewer_puzzles_than_kits_difference_is_nine_l1252_125272

/-- The Smart Mart sells educational toys -/
structure SmartMart where
  science_kits : ℕ
  puzzles : ℕ

/-- The number of science kits sold is 45 -/
def science_kits_sold : ℕ := 45

/-- The number of puzzles sold is 36 -/
def puzzles_sold : ℕ := 36

/-- The Smart Mart sold fewer puzzles than science kits -/
theorem fewer_puzzles_than_kits (sm : SmartMart) :
  sm.puzzles < sm.science_kits :=
sorry

/-- The difference between science kits and puzzles sold is 9 -/
theorem difference_is_nine (sm : SmartMart) 
  (h1 : sm.science_kits = science_kits_sold) 
  (h2 : sm.puzzles = puzzles_sold) : 
  sm.science_kits - sm.puzzles = 9 :=
sorry

end fewer_puzzles_than_kits_difference_is_nine_l1252_125272


namespace vector_collinear_same_direction_l1252_125247

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Two vectors have the same direction if one is a positive scalar multiple of the other -/
def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ b = (k * a.1, k * a.2)

/-- Main theorem: If vectors a = (-1, x) and b = (-x, 2) are collinear and have the same direction, then x = √2 -/
theorem vector_collinear_same_direction (x : ℝ) :
  let a : ℝ × ℝ := (-1, x)
  let b : ℝ × ℝ := (-x, 2)
  collinear a b → same_direction a b → x = Real.sqrt 2 := by
  sorry

end vector_collinear_same_direction_l1252_125247


namespace quadratic_equation_roots_l1252_125210

theorem quadratic_equation_roots (m : ℝ) :
  (2 * (2 : ℝ)^2 - 5 * 2 - m = 0) →
  (m = -2 ∧ ∃ (x : ℝ), x ≠ 2 ∧ 2 * x^2 - 5 * x - m = 0 ∧ x = 1/2) :=
by sorry

end quadratic_equation_roots_l1252_125210


namespace intersection_implies_a_value_l1252_125242

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A a ∩ B a = {-3} → a = -1 :=
by
  sorry

end intersection_implies_a_value_l1252_125242


namespace intersection_M_N_l1252_125223

def M : Set ℝ := {x | 3 * x - 6 ≥ 0}
def N : Set ℝ := {x | x^2 < 16}

theorem intersection_M_N : M ∩ N = Set.Icc 2 4 := by sorry

end intersection_M_N_l1252_125223


namespace complement_A_intersect_B_l1252_125287

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {0, 2}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = {0} := by sorry

end complement_A_intersect_B_l1252_125287


namespace unsold_books_percentage_l1252_125244

def initial_stock : ℕ := 700
def monday_sales : ℕ := 50
def tuesday_sales : ℕ := 82
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

theorem unsold_books_percentage :
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let unsold_books := initial_stock - total_sales
  (unsold_books : ℚ) / initial_stock * 100 = 60 := by
  sorry

end unsold_books_percentage_l1252_125244


namespace sue_age_l1252_125245

theorem sue_age (total_age kate_age maggie_age sue_age : ℕ) : 
  total_age = 48 → kate_age = 19 → maggie_age = 17 → 
  total_age = kate_age + maggie_age + sue_age →
  sue_age = 12 := by
sorry

end sue_age_l1252_125245


namespace perfect_square_function_characterization_l1252_125216

/-- A function from positive naturals to positive naturals -/
def PositiveNatFunction := ℕ+ → ℕ+

/-- The property that (m + g(n))(g(m) + n) is a perfect square for all m, n -/
def IsPerfectSquareProperty (g : PositiveNatFunction) : Prop :=
  ∀ m n : ℕ+, ∃ k : ℕ+, (m + g n) * (g m + n) = k * k

/-- The main theorem stating that if g satisfies the perfect square property,
    then it must be of the form g(n) = n + c for some constant c -/
theorem perfect_square_function_characterization (g : PositiveNatFunction) 
    (h : IsPerfectSquareProperty g) :
    ∃ c : ℕ, ∀ n : ℕ+, g n = n + c := by
  sorry


end perfect_square_function_characterization_l1252_125216


namespace quadratic_equation_with_root_difference_l1252_125268

theorem quadratic_equation_with_root_difference (c : ℝ) : 
  (∃ (r₁ r₂ : ℝ), 2 * r₁^2 + 5 * r₁ = c ∧ 
                   2 * r₂^2 + 5 * r₂ = c ∧ 
                   r₂ = r₁ + 5.5) → 
  c = 12 := by
sorry

end quadratic_equation_with_root_difference_l1252_125268


namespace collinear_probability_l1252_125224

/-- Represents a rectangular grid of dots -/
structure DotGrid :=
  (rows : ℕ)
  (columns : ℕ)

/-- Calculates the total number of dots in the grid -/
def DotGrid.total_dots (g : DotGrid) : ℕ := g.rows * g.columns

/-- Calculates the number of ways to choose 4 dots from n dots -/
def choose_four (n : ℕ) : ℕ := n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Represents the number of collinear sets in the grid -/
def collinear_sets (g : DotGrid) : ℕ := 
  g.rows * 2 + g.columns + 4

/-- The main theorem stating the probability of four randomly chosen dots being collinear -/
theorem collinear_probability (g : DotGrid) (h1 : g.rows = 4) (h2 : g.columns = 5) : 
  (collinear_sets g : ℚ) / (choose_four (g.total_dots) : ℚ) = 17 / 4845 := by
  sorry

#eval collinear_sets (DotGrid.mk 4 5)
#eval choose_four 20

end collinear_probability_l1252_125224


namespace point_positions_l1252_125207

def circle_equation (x y : ℝ) : ℝ := x^2 + y^2 - 2*x + 4*y - 4

def point_M : ℝ × ℝ := (2, -4)
def point_N : ℝ × ℝ := (-2, 1)

theorem point_positions :
  circle_equation point_M.1 point_M.2 < 0 ∧ 
  circle_equation point_N.1 point_N.2 > 0 := by
sorry

end point_positions_l1252_125207


namespace second_applicant_revenue_l1252_125258

/-- Represents the financial details of a job applicant -/
structure Applicant where
  salary : ℕ
  revenue : ℕ
  trainingMonths : ℕ
  trainingCostPerMonth : ℕ
  hiringBonusPercent : ℕ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) : ℕ :=
  a.revenue - a.salary - (a.trainingMonths * a.trainingCostPerMonth) - (a.salary * a.hiringBonusPercent / 100)

/-- The theorem to prove -/
theorem second_applicant_revenue
  (first : Applicant)
  (second : Applicant)
  (h1 : first.salary = 42000)
  (h2 : first.revenue = 93000)
  (h3 : first.trainingMonths = 3)
  (h4 : first.trainingCostPerMonth = 1200)
  (h5 : first.hiringBonusPercent = 0)
  (h6 : second.salary = 45000)
  (h7 : second.trainingMonths = 0)
  (h8 : second.trainingCostPerMonth = 0)
  (h9 : second.hiringBonusPercent = 1)
  (h10 : netGain second = netGain first + 850) :
  second.revenue = 93700 := by
  sorry

end second_applicant_revenue_l1252_125258


namespace quadratic_inequality_solution_set_l1252_125222

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 36*x + 323
  let solution_set := {x : ℝ | f x ≤ 5}
  let lower_bound := 18 - Real.sqrt 6
  let upper_bound := 18 + Real.sqrt 6
  solution_set = Set.Icc lower_bound upper_bound := by
sorry

end quadratic_inequality_solution_set_l1252_125222


namespace white_marbles_count_l1252_125252

theorem white_marbles_count (total : ℕ) (black red green : ℕ) 
  (h_total : total = 60)
  (h_black : black = 32)
  (h_red : red = 10)
  (h_green : green = 5)
  (h_sum : total = black + red + green + (total - (black + red + green))) :
  total - (black + red + green) = 13 := by
  sorry

end white_marbles_count_l1252_125252


namespace toy_factory_daily_production_l1252_125294

/-- A factory produces toys with the following conditions:
    - The factory produces 5500 toys per week
    - Workers work 5 days a week
    - The same number of toys is made every day -/
def ToyFactory (weekly_production : ℕ) (work_days : ℕ) (daily_production : ℕ) : Prop :=
  weekly_production = 5500 ∧ work_days = 5 ∧ daily_production * work_days = weekly_production

/-- Theorem: Given the conditions of the toy factory, the daily production is 1100 toys -/
theorem toy_factory_daily_production :
  ∀ (weekly_production work_days daily_production : ℕ),
  ToyFactory weekly_production work_days daily_production →
  daily_production = 1100 := by
  sorry

end toy_factory_daily_production_l1252_125294


namespace inequality_proof_l1252_125209

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a := by
  sorry

end inequality_proof_l1252_125209


namespace insufficient_info_to_determine_C_l1252_125265

/-- A line in the xy-plane defined by the equation x = 8y + C -/
structure Line where
  C : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  p.x = 8 * p.y + l.C

theorem insufficient_info_to_determine_C 
  (m n : ℝ) (l : Line) :
  let p1 : Point := ⟨m, n⟩
  let p2 : Point := ⟨m + 2, n + 0.25⟩
  p1.on_line l ∧ p2.on_line l →
  ∃ (C' : ℝ), C' ≠ l.C ∧ 
    (⟨m, n⟩ : Point).on_line ⟨C'⟩ ∧ 
    (⟨m + 2, n + 0.25⟩ : Point).on_line ⟨C'⟩ :=
sorry

end insufficient_info_to_determine_C_l1252_125265


namespace quadratic_roots_properties_l1252_125296

theorem quadratic_roots_properties (x₁ x₂ : ℝ) 
  (h : 2 * x₁^2 - 3 * x₁ - 1 = 0 ∧ 2 * x₂^2 - 3 * x₂ - 1 = 0) : 
  (1 / x₁ + 1 / x₂ = -3) ∧ 
  ((x₁^2 - x₂^2)^2 = 153 / 16) ∧ 
  (2 * x₁^2 + 3 * x₂ = 11 / 2) := by
  sorry

end quadratic_roots_properties_l1252_125296


namespace g_value_l1252_125255

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem g_value (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2) (h4 : f 1 + g (-1) = 4) : g 1 = 3 := by
  sorry

end g_value_l1252_125255


namespace cubic_root_sum_l1252_125263

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 3 + a * (Complex.I + 2) + b = 0 → a + b = 20 := by
  sorry

end cubic_root_sum_l1252_125263


namespace block_height_is_75_l1252_125253

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the properties of the cubes cut from the block -/
structure CubeProperties where
  sideLength : ℝ
  count : ℕ

/-- Checks if the given dimensions and cube properties satisfy the problem conditions -/
def satisfiesConditions (block : BlockDimensions) (cube : CubeProperties) : Prop :=
  block.length = 15 ∧
  block.width = 30 ∧
  cube.count = 10 ∧
  (cube.sideLength ∣ block.length) ∧
  (cube.sideLength ∣ block.width) ∧
  (cube.sideLength ∣ block.height) ∧
  block.length * block.width * block.height = cube.sideLength ^ 3 * cube.count

theorem block_height_is_75 (block : BlockDimensions) (cube : CubeProperties) :
  satisfiesConditions block cube → block.height = 75 := by
  sorry

end block_height_is_75_l1252_125253


namespace salary_problem_l1252_125278

-- Define the salaries and total
variable (A B : ℝ)
def total : ℝ := 6000

-- Define the spending percentages
def A_spend_percent : ℝ := 0.95
def B_spend_percent : ℝ := 0.85

-- Define the theorem
theorem salary_problem :
  A + B = total ∧ 
  (1 - A_spend_percent) * A = (1 - B_spend_percent) * B →
  A = 4500 := by
sorry

end salary_problem_l1252_125278


namespace complement_of_union_l1252_125235

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end complement_of_union_l1252_125235


namespace polynomial_roots_l1252_125218

theorem polynomial_roots (AT TB : ℝ) (h1 : AT + TB = 15) (h2 : AT * TB = 36) :
  ∃ (p : ℝ → ℝ), p = (fun x ↦ x^2 - 20*x + 75) ∧ 
  p (AT + 5) = 0 ∧ p TB = 0 := by
  sorry

end polynomial_roots_l1252_125218


namespace slope_angle_of_y_equals_1_l1252_125259

-- Define a line parallel to the x-axis
def parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y

-- Define the slope angle of a line
def slope_angle (f : ℝ → ℝ) : ℝ := sorry

-- Theorem: The slope angle of the line y = 1 is 0
theorem slope_angle_of_y_equals_1 :
  let f : ℝ → ℝ := λ x => 1
  parallel_to_x_axis f ∧ slope_angle f = 0 := by sorry

end slope_angle_of_y_equals_1_l1252_125259


namespace johnson_family_seating_l1252_125239

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def total_arrangements (n : ℕ) : ℕ := factorial n

def alternating_arrangements (n : ℕ) : ℕ := 2 * factorial n * factorial n

theorem johnson_family_seating (boys girls : ℕ) (h : boys = 4 ∧ girls = 4) :
  total_arrangements (boys + girls) - alternating_arrangements boys =
  39168 := by sorry

end johnson_family_seating_l1252_125239


namespace sin_double_angle_when_tan_is_half_l1252_125234

theorem sin_double_angle_when_tan_is_half (α : Real) (h : Real.tan α = 1/2) : 
  Real.sin (2 * α) = 4/5 := by
  sorry

end sin_double_angle_when_tan_is_half_l1252_125234


namespace middle_share_is_forty_l1252_125246

/-- Represents the distribution of marbles among three people -/
structure MarbleDistribution where
  total : ℕ
  ratio1 : ℕ
  ratio2 : ℕ
  ratio3 : ℕ

/-- Calculates the number of marbles for the person with the middle ratio -/
def middleShare (d : MarbleDistribution) : ℕ :=
  d.total * d.ratio2 / (d.ratio1 + d.ratio2 + d.ratio3)

/-- Theorem: In a distribution of 120 marbles with ratio 4:5:6, the middle share is 40 -/
theorem middle_share_is_forty : 
  let d : MarbleDistribution := ⟨120, 4, 5, 6⟩
  middleShare d = 40 := by sorry


end middle_share_is_forty_l1252_125246


namespace spinner_direction_final_direction_is_west_l1252_125202

-- Define the possible directions
inductive Direction
  | North
  | South
  | East
  | West

-- Define a function to rotate a direction
def rotate (d : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- State the theorem
theorem spinner_direction (initial : Direction) 
  (clockwise : ℚ) (counterclockwise : ℚ) : Direction :=
  by
  -- Assume the initial direction is south
  have h1 : initial = Direction.South := by sorry
  -- Assume clockwise rotation is 3½ revolutions
  have h2 : clockwise = 7/2 := by sorry
  -- Assume counterclockwise rotation is 1¾ revolutions
  have h3 : counterclockwise = 7/4 := by sorry
  -- Prove that the final direction is west
  sorry

-- The main theorem
theorem final_direction_is_west :
  spinner_direction Direction.South (7/2) (7/4) = Direction.West :=
  by sorry

end spinner_direction_final_direction_is_west_l1252_125202


namespace total_cost_of_all_lawns_l1252_125212

structure Lawn where
  length : ℕ
  breadth : ℕ
  lengthRoadWidth : ℕ
  breadthRoadWidth : ℕ
  costPerSqMeter : ℕ

def totalRoadArea (l : Lawn) : ℕ :=
  l.length * l.lengthRoadWidth + l.breadth * l.breadthRoadWidth

def totalCost (l : Lawn) : ℕ :=
  totalRoadArea l * l.costPerSqMeter

def lawnA : Lawn := ⟨80, 70, 8, 6, 3⟩
def lawnB : Lawn := ⟨120, 50, 12, 10, 4⟩
def lawnC : Lawn := ⟨150, 90, 15, 9, 5⟩

theorem total_cost_of_all_lawns :
  totalCost lawnA + totalCost lawnB + totalCost lawnC = 26240 := by
  sorry

#eval totalCost lawnA + totalCost lawnB + totalCost lawnC

end total_cost_of_all_lawns_l1252_125212


namespace intersection_complement_when_a_is_two_union_equality_iff_a_leq_two_l1252_125229

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part I
theorem intersection_complement_when_a_is_two :
  M ∩ (Set.univ \ N 2) = {x | -2 ≤ x ∧ x < 3} := by sorry

-- Theorem for part II
theorem union_equality_iff_a_leq_two (a : ℝ) :
  M ∪ N a = M ↔ a ≤ 2 := by sorry

end intersection_complement_when_a_is_two_union_equality_iff_a_leq_two_l1252_125229


namespace square_of_3y_plus_4_when_y_is_neg_2_l1252_125238

theorem square_of_3y_plus_4_when_y_is_neg_2 :
  let y : ℤ := -2
  (3 * y + 4)^2 = 4 := by sorry

end square_of_3y_plus_4_when_y_is_neg_2_l1252_125238


namespace triangle_point_collinearity_l1252_125280

/-- Given a triangle ABC and a point P on the same plane, 
    if BC + BA = 2BP, then P, A, and C are collinear -/
theorem triangle_point_collinearity 
  (A B C P : EuclideanSpace ℝ (Fin 2)) 
  (h : (C - B) + (A - B) = 2 • (P - B)) : 
  Collinear ℝ ({P, A, C} : Set (EuclideanSpace ℝ (Fin 2))) :=
sorry

end triangle_point_collinearity_l1252_125280


namespace A_subset_A_inter_B_iff_l1252_125254

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem A_subset_A_inter_B_iff (a : ℝ) : 
  (A a).Nonempty → (A a ⊆ A a ∩ B ↔ 6 ≤ a ∧ a ≤ 9) := by sorry

end A_subset_A_inter_B_iff_l1252_125254


namespace root_between_roots_l1252_125226

theorem root_between_roots (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hc : c ≠ 0) 
  (hr : a * r^2 + b * r + c = 0) 
  (hs : -a * s^2 + b * s + c = 0) : 
  ∃ t, (t > min r s ∧ t < max r s) ∧ (a / 2) * t^2 + b * t + c = 0 :=
sorry

end root_between_roots_l1252_125226


namespace irrational_zero_one_sequence_exists_l1252_125298

/-- A sequence representing the decimal digits of a number -/
def DecimalSequence := ℕ → Fin 10

/-- Checks if a decimal sequence contains only 0 and 1 -/
def OnlyZeroOne (s : DecimalSequence) : Prop :=
  ∀ n, s n = 0 ∨ s n = 1

/-- Checks if a decimal sequence has no two adjacent 1s -/
def NoAdjacentOnes (s : DecimalSequence) : Prop :=
  ∀ n, ¬(s n = 1 ∧ s (n + 1) = 1)

/-- Checks if a decimal sequence has no more than two adjacent 0s -/
def NoMoreThanTwoZeros (s : DecimalSequence) : Prop :=
  ∀ n, ¬(s n = 0 ∧ s (n + 1) = 0 ∧ s (n + 2) = 0)

/-- Checks if a decimal sequence represents an irrational number -/
def IsIrrational (s : DecimalSequence) : Prop :=
  ∀ k p, ∃ n ≥ k, s n ≠ s (n + p)

/-- There exists an irrational number whose decimal representation
    contains only 0 and 1, with no two adjacent 1s and no more than two adjacent 0s -/
theorem irrational_zero_one_sequence_exists : 
  ∃ s : DecimalSequence, 
    OnlyZeroOne s ∧ 
    NoAdjacentOnes s ∧ 
    NoMoreThanTwoZeros s ∧ 
    IsIrrational s := by
  sorry

end irrational_zero_one_sequence_exists_l1252_125298


namespace rice_division_l1252_125264

theorem rice_division (total_pounds : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) : 
  total_pounds = 25 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_pounds * ounces_per_pound) / num_containers = 50 := by
  sorry

end rice_division_l1252_125264


namespace negation_of_universal_statement_l1252_125215

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 < 0) :=
by sorry

end negation_of_universal_statement_l1252_125215


namespace average_permutation_sum_l1252_125225

def permutation_sum (b : Fin 8 → Fin 8) : ℕ :=
  |b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|

def all_permutations : Finset (Fin 8 → Fin 8) :=
  Finset.univ.filter (λ b ↦ Function.Bijective b)

theorem average_permutation_sum :
  (Finset.sum all_permutations permutation_sum) / all_permutations.card = 672 := by
  sorry

end average_permutation_sum_l1252_125225


namespace outfit_combinations_l1252_125293

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (ties : ℕ) (jackets : ℕ) :
  shirts = 8 →
  pants = 5 →
  ties = 4 →
  jackets = 3 →
  shirts * pants * (ties + 1) * (jackets + 1) = 800 :=
by sorry

end outfit_combinations_l1252_125293


namespace systematic_sampling_probability_l1252_125206

theorem systematic_sampling_probability 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_population = 120) 
  (h2 : sample_size = 20) :
  (sample_size : ℚ) / total_population = 1 / 6 := by
  sorry

end systematic_sampling_probability_l1252_125206


namespace polynomial_coefficient_sum_l1252_125295

theorem polynomial_coefficient_sum (a₄ a₃ a₂ a₁ a : ℝ) :
  (∀ x : ℝ, a₄ * (x + 1)^4 + a₃ * (x + 1)^3 + a₂ * (x + 1)^2 + a₁ * (x + 1) + a = x^4) →
  a₃ - a₂ + a₁ = -14 := by
sorry

end polynomial_coefficient_sum_l1252_125295


namespace german_team_goals_l1252_125284

def journalist1_statement (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2_statement (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3_statement (x : ℕ) : Prop := x % 2 = 1

def exactly_two_correct (x : ℕ) : Prop :=
  (journalist1_statement x ∧ journalist2_statement x ∧ ¬journalist3_statement x) ∨
  (journalist1_statement x ∧ ¬journalist2_statement x ∧ journalist3_statement x) ∨
  (¬journalist1_statement x ∧ journalist2_statement x ∧ journalist3_statement x)

theorem german_team_goals :
  {x : ℕ | exactly_two_correct x} = {11, 12, 14, 16, 17} := by sorry

end german_team_goals_l1252_125284


namespace b_is_criminal_l1252_125232

-- Define the set of suspects
inductive Suspect : Type
  | A | B | C | D

-- Define a function to represent the statements of each suspect
def statement (s : Suspect) (criminal : Suspect) : Prop :=
  match s with
  | Suspect.A => criminal ≠ Suspect.A
  | Suspect.B => criminal = Suspect.C
  | Suspect.C => criminal = Suspect.A ∨ criminal = Suspect.B
  | Suspect.D => criminal = Suspect.C

-- Define a function to check if a statement is true given the actual criminal
def is_true_statement (s : Suspect) (criminal : Suspect) : Prop :=
  statement s criminal

-- Theorem stating that B is the criminal
theorem b_is_criminal :
  ∃ (criminal : Suspect),
    criminal = Suspect.B ∧
    (∃ (t1 t2 l1 l2 : Suspect),
      t1 ≠ t2 ∧ l1 ≠ l2 ∧
      t1 ≠ l1 ∧ t1 ≠ l2 ∧ t2 ≠ l1 ∧ t2 ≠ l2 ∧
      is_true_statement t1 criminal ∧
      is_true_statement t2 criminal ∧
      ¬is_true_statement l1 criminal ∧
      ¬is_true_statement l2 criminal) :=
by
  sorry


end b_is_criminal_l1252_125232


namespace expression_evaluation_l1252_125220

theorem expression_evaluation : 
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end expression_evaluation_l1252_125220


namespace min_value_theorem_l1252_125279

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 3 * m + n = 1) :
  (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 := by
  sorry

end min_value_theorem_l1252_125279


namespace integral_equality_l1252_125297

open Real MeasureTheory

theorem integral_equality : ∫ (x : ℝ) in (0)..(1), 
  Real.exp (Real.sqrt ((1 - x) / (1 + x))) / ((1 + x) * Real.sqrt (1 - x^2)) = Real.exp 1 - 1 := by
  sorry

end integral_equality_l1252_125297


namespace rent_to_expenses_ratio_l1252_125277

/-- Given Kathryn's monthly finances, prove the ratio of rent to food and travel expenses -/
theorem rent_to_expenses_ratio 
  (rent : ℕ) 
  (salary : ℕ) 
  (remaining : ℕ) 
  (h1 : rent = 1200)
  (h2 : salary = 5000)
  (h3 : remaining = 2000) :
  (rent : ℚ) / ((salary - remaining) - rent) = 2 / 3 := by
  sorry

end rent_to_expenses_ratio_l1252_125277


namespace range_of_increasing_function_l1252_125249

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem range_of_increasing_function (f : ℝ → ℝ) (h : increasing_function f) :
  {m : ℝ | f (2 - m) < f (m^2)} = {m : ℝ | m < -2 ∨ m > 1} := by
  sorry

end range_of_increasing_function_l1252_125249


namespace jack_shoe_time_proof_l1252_125262

/-- The time it takes Jack to put on his shoes -/
def jack_shoe_time : ℝ := 4

/-- The time it takes Jack to help one toddler with their shoes -/
def toddler_shoe_time (j : ℝ) : ℝ := j + 3

/-- The total time for Jack and two toddlers to get ready -/
def total_time (j : ℝ) : ℝ := j + 2 * (toddler_shoe_time j)

theorem jack_shoe_time_proof :
  total_time jack_shoe_time = 18 :=
by sorry

end jack_shoe_time_proof_l1252_125262


namespace bacteriophage_and_transformation_principle_correct_biological_experiment_description_l1252_125286

/-- Represents a biological experiment --/
structure BiologicalExperiment where
  name : String
  description : String

/-- Represents the principle behind an experiment --/
inductive ExperimentPrinciple
  | GeneticContinuity
  | Other

/-- Function to determine the principle of an experiment --/
def experimentPrinciple (exp : BiologicalExperiment) : ExperimentPrinciple :=
  if exp.name = "Bacteriophage Infection" || exp.name = "Bacterial Transformation" then
    ExperimentPrinciple.GeneticContinuity
  else
    ExperimentPrinciple.Other

/-- Theorem stating that bacteriophage infection and bacterial transformation 
    experiments are based on the same principle of genetic continuity --/
theorem bacteriophage_and_transformation_principle :
  ∀ (exp1 exp2 : BiologicalExperiment),
    exp1.name = "Bacteriophage Infection" →
    exp2.name = "Bacterial Transformation" →
    experimentPrinciple exp1 = experimentPrinciple exp2 :=
by
  sorry

/-- Main theorem proving the correctness of the statement --/
theorem correct_biological_experiment_description :
  ∃ (exp1 exp2 : BiologicalExperiment),
    exp1.name = "Bacteriophage Infection" ∧
    exp2.name = "Bacterial Transformation" ∧
    experimentPrinciple exp1 = ExperimentPrinciple.GeneticContinuity ∧
    experimentPrinciple exp2 = ExperimentPrinciple.GeneticContinuity :=
by
  sorry

end bacteriophage_and_transformation_principle_correct_biological_experiment_description_l1252_125286


namespace kyle_driving_time_l1252_125285

/-- Given the conditions of Joseph and Kyle's driving, prove that Kyle's driving time is 2 hours. -/
theorem kyle_driving_time :
  let joseph_speed : ℝ := 50
  let joseph_time : ℝ := 2.5
  let kyle_speed : ℝ := 62
  let joseph_distance : ℝ := joseph_speed * joseph_time
  let kyle_distance : ℝ := joseph_distance - 1
  kyle_distance / kyle_speed = 2 := by sorry

end kyle_driving_time_l1252_125285


namespace matrix_equation_l1252_125276

theorem matrix_equation (A B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : A + B = A * B) 
  (h2 : A * B = !![5, 2; -3, 9]) : 
  B * A = !![5, 2; -3, 9] := by sorry

end matrix_equation_l1252_125276


namespace prop_1_prop_2_prop_3_l1252_125213

-- Define the function f
def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

-- Proposition 1
theorem prop_1 (b : ℝ) : 
  ∀ x, f b 0 (-x) = -(f b 0 x) := by sorry

-- Proposition 2
theorem prop_2 (c : ℝ) (h : c > 0) : 
  ∃! x, f 0 c x = 0 := by sorry

-- Proposition 3
theorem prop_3 (b c : ℝ) : 
  ∀ x, f b c (-x) = 2 * c - f b c x := by sorry

end prop_1_prop_2_prop_3_l1252_125213


namespace total_remaining_students_l1252_125200

def calculate_remaining_students (initial_a initial_b initial_c new_a new_b new_c transfer_rate_a transfer_rate_b transfer_rate_c : ℕ) : ℕ :=
  let total_a := initial_a + new_a
  let total_b := initial_b + new_b
  let total_c := initial_c + new_c
  let remaining_a := total_a - (total_a * transfer_rate_a / 100)
  let remaining_b := total_b - (total_b * transfer_rate_b / 100)
  let remaining_c := total_c - (total_c * transfer_rate_c / 100)
  remaining_a + remaining_b + remaining_c

theorem total_remaining_students :
  calculate_remaining_students 160 145 130 20 25 15 30 25 20 = 369 :=
by sorry

end total_remaining_students_l1252_125200


namespace divisible_by_nine_l1252_125233

theorem divisible_by_nine (k : ℕ+) : 
  (9 : ℤ) ∣ (3 * (2 + 7^(k : ℕ))) := by
  sorry

end divisible_by_nine_l1252_125233


namespace jeans_discount_rates_l1252_125214

def regular_price_moose : ℝ := 20
def regular_price_fox : ℝ := 15
def regular_price_pony : ℝ := 18

def num_moose : ℕ := 2
def num_fox : ℕ := 3
def num_pony : ℕ := 2

def total_savings : ℝ := 12.48

def sum_all_rates : ℝ := 0.32
def sum_fox_pony_rates : ℝ := 0.20

def discount_rate_moose : ℝ := 0.12
def discount_rate_fox : ℝ := 0.0533
def discount_rate_pony : ℝ := 0.1467

theorem jeans_discount_rates :
  (discount_rate_moose + discount_rate_fox + discount_rate_pony = sum_all_rates) ∧
  (discount_rate_fox + discount_rate_pony = sum_fox_pony_rates) ∧
  (num_moose * discount_rate_moose * regular_price_moose +
   num_fox * discount_rate_fox * regular_price_fox +
   num_pony * discount_rate_pony * regular_price_pony = total_savings) :=
by sorry

end jeans_discount_rates_l1252_125214


namespace cos_150_degrees_l1252_125271

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l1252_125271


namespace ratio_problem_l1252_125270

/-- Given ratios for x, y, and z, prove their values -/
theorem ratio_problem (x y z : ℚ) : 
  (x / 12 = 5 / 1) → 
  (y / 21 = 7 / 3) → 
  (z / 16 = 4 / 2) → 
  (x = 60 ∧ y = 49 ∧ z = 32) :=
by sorry

end ratio_problem_l1252_125270


namespace problem_one_problem_two_l1252_125274

-- Problem 1
theorem problem_one : |2 - Real.sqrt 3| - 2^0 - Real.sqrt 12 = 1 - 3 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_two : (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + (2 * Real.sqrt 3 + 1)^2 = 14 + 4 * Real.sqrt 3 := by
  sorry

end problem_one_problem_two_l1252_125274


namespace sum_squares_and_inverses_bound_l1252_125273

theorem sum_squares_and_inverses_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^2 + 1/a^2 + b^2 + 1/b^2 ≥ 4 := by
  sorry

end sum_squares_and_inverses_bound_l1252_125273


namespace min_value_sum_min_value_attained_min_value_is_1215_l1252_125201

theorem min_value_sum (x y z : ℕ+) (h : x^3 + y^3 + z^3 - 3*x*y*z = 607) :
  ∀ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 → x + 2*y + 3*z ≤ a + 2*b + 3*c :=
by sorry

theorem min_value_attained (x y z : ℕ+) (h : x^3 + y^3 + z^3 - 3*x*y*z = 607) :
  ∃ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 ∧ a + 2*b + 3*c = 1215 :=
by sorry

theorem min_value_is_1215 :
  ∃ (x y z : ℕ+), x^3 + y^3 + z^3 - 3*x*y*z = 607 ∧
  (∀ (a b c : ℕ+), a^3 + b^3 + c^3 - 3*a*b*c = 607 → x + 2*y + 3*z ≤ a + 2*b + 3*c) ∧
  x + 2*y + 3*z = 1215 :=
by sorry

end min_value_sum_min_value_attained_min_value_is_1215_l1252_125201


namespace fractional_equation_transformation_l1252_125269

theorem fractional_equation_transformation (x : ℝ) :
  (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x - 2 = 3 * (2 * x - 1)) :=
by sorry

end fractional_equation_transformation_l1252_125269


namespace f_1_eq_0_f_decreasing_f_abs_lt_neg_2_iff_l1252_125288

noncomputable section

variable (f : ℝ → ℝ)

axiom f_domain : ∀ x, x > 0 → f x ≠ 0

axiom f_functional_equation : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

axiom f_negative_when_gt_one : ∀ x, x > 1 → f x < 0

axiom f_3_eq_neg_1 : f 3 = -1

theorem f_1_eq_0 : f 1 = 0 := by sorry

theorem f_decreasing : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ > x₂ → f x₁ < f x₂ := by sorry

theorem f_abs_lt_neg_2_iff : ∀ x, f (|x|) < -2 ↔ x < -9 ∨ x > 9 := by sorry

end

end f_1_eq_0_f_decreasing_f_abs_lt_neg_2_iff_l1252_125288


namespace writer_productivity_l1252_125299

/-- Calculates the average words per hour for a writer given total words, total hours, and break hours. -/
def averageWordsPerHour (totalWords : ℕ) (totalHours : ℕ) (breakHours : ℕ) : ℚ :=
  totalWords / (totalHours - breakHours)

/-- Theorem stating that for a writer completing 60,000 words in 100 hours with 20 hours of breaks,
    the average words per hour when actually working is 750. -/
theorem writer_productivity : averageWordsPerHour 60000 100 20 = 750 := by
  sorry

end writer_productivity_l1252_125299


namespace min_value_when_a_is_one_range_of_a_when_f_geq_3_l1252_125248

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

-- Theorem 1
theorem min_value_when_a_is_one :
  ∀ x ∈ Set.Ioo 0 (Real.exp 1), f 1 x ≥ f 1 1 ∧ f 1 1 = 1 := by sorry

-- Theorem 2
theorem range_of_a_when_f_geq_3 :
  (∀ x ∈ Set.Ioo 0 (Real.exp 1), f a x ≥ 3) → a ≥ Real.exp 2 := by sorry

end

end min_value_when_a_is_one_range_of_a_when_f_geq_3_l1252_125248


namespace ternary_57_has_four_digits_l1252_125217

/-- Converts a natural number to its ternary (base 3) representation -/
def to_ternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The theorem stating that the ternary representation of 57 has exactly 4 digits -/
theorem ternary_57_has_four_digits : (to_ternary 57).length = 4 := by
  sorry

end ternary_57_has_four_digits_l1252_125217


namespace initial_toothbrushes_l1252_125257

/-- The number of toothbrushes given away in January -/
def january : ℕ := 53

/-- The number of toothbrushes given away in February -/
def february : ℕ := 67

/-- The number of toothbrushes given away in March -/
def march : ℕ := 46

/-- The difference between the busiest and slowest month -/
def difference : ℕ := 36

/-- The number of toothbrushes given away in April (equal to May) -/
def april_may : ℕ := february - difference

/-- The total number of toothbrushes Dr. Banks had initially -/
def total_toothbrushes : ℕ := january + february + march + 2 * april_may

theorem initial_toothbrushes : total_toothbrushes = 228 := by
  sorry

end initial_toothbrushes_l1252_125257


namespace quadratic_inequality_solution_l1252_125221

/-- Theorem: Solution of quadratic inequality ax^2 + bx + c < 0 -/
theorem quadratic_inequality_solution 
  (a b c : ℝ) (h : a ≠ 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a > 0 → {x : ℝ | x1 < x ∧ x < x2} = {x : ℝ | a*x^2 + b*x + c < 0}) ∧
  (a < 0 → {x : ℝ | x < x1 ∨ x2 < x} = {x : ℝ | a*x^2 + b*x + c < 0}) :=
by sorry


end quadratic_inequality_solution_l1252_125221


namespace nicky_card_trade_loss_l1252_125231

/-- Calculates the profit or loss from a card trade with tax -/
def card_trade_profit (
  cards_given_value1 : ℝ)
  (cards_given_count1 : ℕ)
  (cards_given_value2 : ℝ)
  (cards_given_count2 : ℕ)
  (cards_received_value1 : ℝ)
  (cards_received_count1 : ℕ)
  (cards_received_value2 : ℝ)
  (cards_received_count2 : ℕ)
  (tax_rate : ℝ) : ℝ :=
  let total_given := cards_given_value1 * cards_given_count1 + cards_given_value2 * cards_given_count2
  let total_received := cards_received_value1 * cards_received_count1 + cards_received_value2 * cards_received_count2
  let total_trade_value := total_given + total_received
  let tax := tax_rate * total_trade_value
  total_received - total_given - tax

theorem nicky_card_trade_loss :
  card_trade_profit 8 2 5 3 21 1 6 2 0.05 = -1.20 := by
  sorry

end nicky_card_trade_loss_l1252_125231


namespace extreme_value_and_range_l1252_125241

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x - (x + 1)^2

theorem extreme_value_and_range :
  (∃ x : ℝ, ∀ y : ℝ, f (-1) y ≤ f (-1) x ∧ f (-1) x = 1 / Real.exp 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≤ 0) ↔ a ∈ Set.Icc 0 (4 / Real.exp 1)) :=
sorry

end extreme_value_and_range_l1252_125241


namespace tetrahedron_volume_lower_bound_l1252_125205

-- Define a tetrahedron type
structure Tetrahedron where
  -- The volume of the tetrahedron
  volume : ℝ
  -- The distances between opposite edges
  d₁ : ℝ
  d₂ : ℝ
  d₃ : ℝ
  -- Ensure all distances are positive
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  d₃_pos : d₃ > 0

-- State the theorem
theorem tetrahedron_volume_lower_bound (t : Tetrahedron) : 
  t.volume ≥ (1/3) * t.d₁ * t.d₂ * t.d₃ := by
  sorry

end tetrahedron_volume_lower_bound_l1252_125205


namespace inequality_solution_set_l1252_125237

theorem inequality_solution_set (x : ℝ) : 
  1 - 7 / (2 * x - 1) < 0 ↔ 1/2 < x ∧ x < 4 := by sorry

end inequality_solution_set_l1252_125237


namespace dot_product_equals_negative_31_l1252_125292

def vector1 : Fin 2 → ℝ
  | 0 => -3
  | 1 => 2

def vector2 : Fin 2 → ℝ
  | 0 => 7
  | 1 => -5

theorem dot_product_equals_negative_31 :
  (Finset.univ.sum fun i => vector1 i * vector2 i) = -31 := by
  sorry

end dot_product_equals_negative_31_l1252_125292


namespace translated_line_through_point_l1252_125291

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

/-- Check if a point lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_through_point (m : ℝ) :
  let original_line : Line := { slope := 1, intercept := 0 }
  let translated_line := translate_line original_line 3
  point_on_line translated_line 2 m → m = 5 := by
  sorry

end translated_line_through_point_l1252_125291


namespace equation_solution_l1252_125250

theorem equation_solution : 
  ∀ x : ℝ, |2001*x - 2001| = 2001 ↔ x = 0 ∨ x = 2 := by sorry

end equation_solution_l1252_125250


namespace kelly_apples_l1252_125236

theorem kelly_apples (initial : ℕ) (second_day : ℕ) (third_day : ℕ) (eaten : ℕ) : 
  initial = 56 → second_day = 105 → third_day = 84 → eaten = 23 →
  initial + second_day + third_day - eaten = 222 :=
by
  sorry

end kelly_apples_l1252_125236


namespace smallest_green_points_l1252_125289

/-- The total number of points in the plane -/
def total_points : ℕ := 2020

/-- The distance between a black point and its two associated green points -/
def distance : ℕ := 2020

/-- The property that for each black point, there are exactly two green points at the specified distance -/
def black_point_property (n : ℕ) : Prop :=
  ∀ b : ℕ, b ≤ n * (n - 1)

/-- The theorem stating the smallest number of green points -/
theorem smallest_green_points :
  ∃ n : ℕ, n = 45 ∧ 
    black_point_property n ∧
    n + (total_points - n) = total_points ∧
    ∀ m : ℕ, m < n → ¬(black_point_property m ∧ m + (total_points - m) = total_points) :=
by sorry

end smallest_green_points_l1252_125289


namespace consecutive_sum_smallest_l1252_125281

theorem consecutive_sum_smallest (a : ℤ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 210) : a = 40 := by
  sorry

end consecutive_sum_smallest_l1252_125281


namespace new_people_calculation_l1252_125203

/-- The number of new people who moved into the town -/
def new_people : ℕ := 580

/-- The original population of the town -/
def original_population : ℕ := 780

/-- The number of people who moved out -/
def people_moved_out : ℕ := 400

/-- The population after 4 years -/
def final_population : ℕ := 60

/-- The number of years that passed -/
def years_passed : ℕ := 4

theorem new_people_calculation :
  (((original_population - people_moved_out + new_people : ℚ) / 2^years_passed) : ℚ) = final_population := by
  sorry

end new_people_calculation_l1252_125203


namespace factor_implies_k_value_l1252_125261

theorem factor_implies_k_value (k : ℚ) :
  (∀ x, (3 * x + 4) ∣ (9 * x^3 + k * x^2 + 16 * x + 64)) →
  k = -12 := by
  sorry

end factor_implies_k_value_l1252_125261


namespace wrong_observation_value_l1252_125208

theorem wrong_observation_value (n : ℕ) (initial_mean corrected_mean true_value : ℝ) : 
  n = 50 →
  initial_mean = 36 →
  corrected_mean = 36.54 →
  true_value = 48 →
  (n : ℝ) * corrected_mean - (n : ℝ) * initial_mean = true_value - (n : ℝ) * initial_mean + (n : ℝ) * corrected_mean - (n : ℝ) * initial_mean →
  true_value - ((n : ℝ) * corrected_mean - (n : ℝ) * initial_mean) = 21 := by
  sorry

end wrong_observation_value_l1252_125208


namespace arithmetic_calculations_l1252_125228

theorem arithmetic_calculations : 
  (8 / (-2) - (-4) * (-3) = 8) ∧ 
  ((-2)^3 / 4 * (5 - (-3)^2) = 8) := by
  sorry

end arithmetic_calculations_l1252_125228


namespace fraction_addition_l1252_125251

theorem fraction_addition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 / x + 2 / y = (3 * y + 2 * x) / (x * y) := by
  sorry

end fraction_addition_l1252_125251


namespace linear_function_proof_l1252_125267

theorem linear_function_proof (f : ℝ → ℝ) :
  (∀ x y : ℝ, ∃ k b : ℝ, f x = k * x + b) →
  (∀ x : ℝ, 3 * f (x + 1) - f x = 2 * x + 9) →
  (∀ x : ℝ, f x = x + 3) := by
sorry

end linear_function_proof_l1252_125267


namespace product_difference_square_l1252_125282

theorem product_difference_square : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end product_difference_square_l1252_125282


namespace series_convergence_l1252_125266

/-- The infinite series ∑(k=1 to ∞) [k(k+1)/(2*3^k)] converges to 3/2 -/
theorem series_convergence : 
  ∑' k, (k * (k + 1) : ℝ) / (2 * 3^k) = 3/2 := by sorry

end series_convergence_l1252_125266


namespace chess_probability_l1252_125256

theorem chess_probability (prob_A_win prob_draw : ℝ) 
  (h1 : prob_A_win = 0.3)
  (h2 : prob_draw = 0.5) :
  prob_A_win + prob_draw = 0.8 := by
sorry

end chess_probability_l1252_125256


namespace simplify_expression_find_value_evaluate_compound_l1252_125290

-- Part 1
theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

-- Part 2
theorem find_value (x y : ℝ) (h : x^2 - 2*y = 4) :
  3 * x^2 - 6 * y - 21 = -9 := by sorry

-- Part 3
theorem evaluate_compound (a b c d : ℝ) 
  (h1 : a - 2*b = 6) (h2 : 2*b - c = -8) (h3 : c - d = 9) :
  (a - c) + (2*b - d) - (2*b - c) = 7 := by sorry

end simplify_expression_find_value_evaluate_compound_l1252_125290


namespace find_x_l1252_125275

theorem find_x : ∃ x : ℝ, 
  (3 + 7 + 10 + 15) / 4 = 2 * ((x + 20 + 6) / 3) ∧ 
  x = -12.875 := by
  sorry

end find_x_l1252_125275


namespace new_acute_angle_l1252_125283

theorem new_acute_angle (initial_angle : ℝ) (net_rotation : ℝ) : 
  initial_angle = 60 → net_rotation = 90 → 
  (180 - (initial_angle + net_rotation)) % 180 = 30 := by
sorry

end new_acute_angle_l1252_125283


namespace quadratic_roots_properties_l1252_125230

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + 2*(m-1)*x + m^2 - 1

-- Theorem statement
theorem quadratic_roots_properties :
  -- The equation has two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 →
  -- The range of m is m < 1
  (∀ m : ℝ, (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0) → m < 1) ∧
  -- There exists a value of m such that the product of the roots is zero, and that value is m = -1
  (∃ m : ℝ, m = -1 ∧ ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁ * x₂ = 0) :=
by
  sorry


end quadratic_roots_properties_l1252_125230


namespace equidecomposable_transitivity_l1252_125227

-- Define the concept of a polygon
def Polygon : Type := sorry

-- Define the concept of equidecomposability between two polygons
def equidecomposable (P Q : Polygon) : Prop := sorry

-- Theorem statement
theorem equidecomposable_transitivity (P Q R : Polygon) :
  equidecomposable P R → equidecomposable Q R → equidecomposable P Q := by
  sorry

end equidecomposable_transitivity_l1252_125227


namespace ram_weight_increase_l1252_125260

theorem ram_weight_increase (ram_initial : ℝ) (shyam_initial : ℝ) 
  (h_ratio : ram_initial / shyam_initial = 4 / 5)
  (h_total_new : ram_initial * (1 + x / 100) + shyam_initial * 1.19 = 82.8)
  (h_total_increase : ram_initial * (1 + x / 100) + shyam_initial * 1.19 = (ram_initial + shyam_initial) * 1.15)
  : x = 10 := by
  sorry

end ram_weight_increase_l1252_125260


namespace vasya_lives_on_fifth_floor_l1252_125243

/-- The floor number on which Vasya lives -/
def vasya_floor (petya_steps : ℕ) (vasya_steps : ℕ) : ℕ :=
  1 + vasya_steps / (petya_steps / 2)

/-- Theorem stating that Vasya lives on the 5th floor -/
theorem vasya_lives_on_fifth_floor :
  vasya_floor 36 72 = 5 := by
  sorry

#eval vasya_floor 36 72

end vasya_lives_on_fifth_floor_l1252_125243


namespace committee_selection_l1252_125211

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  Nat.choose n k = 1140 := by
  sorry

end committee_selection_l1252_125211


namespace deepak_age_l1252_125204

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age -/
theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) 
  (years_ahead : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 3 →
  rahul_future_age = 38 →
  years_ahead = 6 →
  ∃ (x : ℕ), rahul_ratio * x + years_ahead = rahul_future_age ∧ 
             deepak_ratio * x = 24 :=
by
  sorry

end deepak_age_l1252_125204


namespace steve_pie_difference_l1252_125219

/-- Represents a baker's weekly pie production --/
structure BakerProduction where
  pies_per_day : ℕ
  apple_pie_days : ℕ
  cherry_pie_days : ℕ

/-- Calculates the difference between apple pies and cherry pies baked in a week --/
def pie_difference (bp : BakerProduction) : ℕ :=
  bp.pies_per_day * bp.apple_pie_days - bp.pies_per_day * bp.cherry_pie_days

/-- Theorem stating the difference in pie production for Steve's bakery --/
theorem steve_pie_difference :
  ∀ (bp : BakerProduction),
    bp.pies_per_day = 12 →
    bp.apple_pie_days = 3 →
    bp.cherry_pie_days = 2 →
    pie_difference bp = 12 := by
  sorry

end steve_pie_difference_l1252_125219
