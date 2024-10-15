import Mathlib

namespace NUMINAMATH_CALUDE_rational_numbers_four_units_from_origin_l2848_284883

theorem rational_numbers_four_units_from_origin :
  {x : ℚ | |x| = 4} = {-4, 4} := by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_four_units_from_origin_l2848_284883


namespace NUMINAMATH_CALUDE_second_box_capacity_l2848_284836

/-- Represents the dimensions and capacity of a rectangular box -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.height * b.width * b.length

theorem second_box_capacity (box1 box2 : Box) : 
  box1.height = 1.5 ∧ 
  box1.width = 4 ∧ 
  box1.length = 6 ∧ 
  box1.capacity = 72 ∧
  box2.height = 3 * box1.height ∧
  box2.width = 2 * box1.width ∧
  box2.length = 0.5 * box1.length →
  box2.capacity = 216 := by
  sorry

end NUMINAMATH_CALUDE_second_box_capacity_l2848_284836


namespace NUMINAMATH_CALUDE_age_sum_problem_l2848_284848

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ a * b * c = 256 → a + b + c = 20 :=
by sorry

end NUMINAMATH_CALUDE_age_sum_problem_l2848_284848


namespace NUMINAMATH_CALUDE_multiply_x_equals_5_l2848_284800

theorem multiply_x_equals_5 (x y : ℝ) (h1 : x * y ≠ 0) 
  (h2 : (1/5 * x) / (1/6 * y) = 0.7200000000000001) : 
  ∃ n : ℝ, n * x = 3 * y ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_x_equals_5_l2848_284800


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2848_284885

def f (x : ℝ) : ℝ := x*(x+1)*(x+2)*(x+3)*(x+4)*(x+5) + 6

theorem f_derivative_at_zero : 
  deriv f 0 = 120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2848_284885


namespace NUMINAMATH_CALUDE_complex_simplification_l2848_284887

theorem complex_simplification :
  (7 * (4 - 2 * Complex.I) + 4 * Complex.I * (7 - 3 * Complex.I)) = (40 : ℂ) + 14 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2848_284887


namespace NUMINAMATH_CALUDE_median_distance_product_sum_l2848_284847

/-- Given a triangle with medians of lengths s₁, s₂, s₃ and a point P with 
    distances d₁, d₂, d₃ to these medians respectively, prove that 
    s₁d₁ + s₂d₂ + s₃d₃ = 0 -/
theorem median_distance_product_sum (s₁ s₂ s₃ d₁ d₂ d₃ : ℝ) : 
  s₁ * d₁ + s₂ * d₂ + s₃ * d₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_median_distance_product_sum_l2848_284847


namespace NUMINAMATH_CALUDE_range_of_a_l2848_284858

theorem range_of_a (a : ℝ) : 
  ((∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
   (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0)) ↔ 
  (a ≤ -2 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2848_284858


namespace NUMINAMATH_CALUDE_maria_hardcover_volumes_l2848_284862

/-- Proof that Maria bought 9 hardcover volumes -/
theorem maria_hardcover_volumes :
  ∀ (h p : ℕ), -- h: number of hardcover volumes, p: number of paperback volumes
  h + p = 15 → -- total number of volumes
  10 * p + 30 * h = 330 → -- total cost equation
  h = 9 := by
sorry

end NUMINAMATH_CALUDE_maria_hardcover_volumes_l2848_284862


namespace NUMINAMATH_CALUDE_transformation_matrix_exists_and_unique_l2848_284820

open Matrix

theorem transformation_matrix_exists_and_unique :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ, 
    ∀ A : Matrix (Fin 2) (Fin 2) ℝ, 
      N * A = !![4 * A 0 0, 4 * A 0 1; A 1 0, A 1 1] := by
  sorry

end NUMINAMATH_CALUDE_transformation_matrix_exists_and_unique_l2848_284820


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2848_284844

def complex_to_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem z_in_fourth_quadrant (z : ℂ) 
  (h : (2 - 3*I)/(3 + 2*I) + z = 2 - 2*I) : 
  in_fourth_quadrant (complex_to_point z) := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2848_284844


namespace NUMINAMATH_CALUDE_more_trucks_than_buses_l2848_284841

/-- Given 17 trucks and 9 buses, prove that there are 8 more trucks than buses. -/
theorem more_trucks_than_buses :
  let num_trucks : ℕ := 17
  let num_buses : ℕ := 9
  num_trucks - num_buses = 8 :=
by sorry

end NUMINAMATH_CALUDE_more_trucks_than_buses_l2848_284841


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l2848_284819

theorem consecutive_odd_numbers_multiple (k : ℕ) : 
  let a := 7
  let b := a + 2
  let c := b + 2
  k * a = 3 * c + (2 * b + 5) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l2848_284819


namespace NUMINAMATH_CALUDE_sum_of_triangle_and_rectangle_edges_l2848_284888

/-- The number of edges in a triangle -/
def triangle_edges : ℕ := 3

/-- The number of edges in a rectangle -/
def rectangle_edges : ℕ := 4

/-- The sum of edges in a triangle and a rectangle -/
def total_edges : ℕ := triangle_edges + rectangle_edges

theorem sum_of_triangle_and_rectangle_edges :
  total_edges = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_and_rectangle_edges_l2848_284888


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l2848_284823

theorem divisibility_implies_equality (a b : ℕ+) (h : (a * b : ℕ) ∣ (a ^ 2 + b ^ 2 : ℕ)) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l2848_284823


namespace NUMINAMATH_CALUDE_impossible_equal_checkers_l2848_284865

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Represents an L-shape on the grid -/
inductive LShape
  | topLeft : LShape
  | topRight : LShape
  | bottomLeft : LShape
  | bottomRight : LShape

/-- Applies a move to the grid -/
def applyMove (grid : Grid) (shape : LShape) : Grid :=
  sorry

/-- Checks if all cells in the grid have the same non-zero value -/
def allCellsSame (grid : Grid) : Prop :=
  sorry

/-- Theorem stating the impossibility of reaching a state where all cells have the same non-zero value -/
theorem impossible_equal_checkers :
  ¬ ∃ (initial : Grid) (moves : List LShape),
    (∀ i j, initial i j = 0) ∧ 
    allCellsSame (moves.foldl applyMove initial) :=
  sorry

end NUMINAMATH_CALUDE_impossible_equal_checkers_l2848_284865


namespace NUMINAMATH_CALUDE_point_movement_l2848_284802

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moves a point on the number line -/
def move (p : Point) (units : ℤ) : Point :=
  ⟨p.value + units⟩

theorem point_movement (A B : Point) :
  A.value = -3 →
  B = move A 7 →
  B.value = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_point_movement_l2848_284802


namespace NUMINAMATH_CALUDE_quadruple_cylinder_volume_l2848_284851

/-- Theorem: Quadrupling Cylinder Dimensions -/
theorem quadruple_cylinder_volume (V : ℝ) (V' : ℝ) :
  V > 0 →  -- Assume positive initial volume
  V' = 64 * V →  -- Definition of V' based on problem conditions
  V' = (4^3) * V  -- Conclusion to prove
  := by sorry

end NUMINAMATH_CALUDE_quadruple_cylinder_volume_l2848_284851


namespace NUMINAMATH_CALUDE_buns_left_is_two_l2848_284898

/-- The number of buns initially on the plate -/
def initial_buns : ℕ := 15

/-- Karlsson takes three times as many buns as Little Boy -/
def karlsson_multiplier : ℕ := 3

/-- Bimbo takes three times fewer buns than Little Boy -/
def bimbo_divisor : ℕ := 3

/-- The number of buns Bimbo takes -/
def bimbo_buns : ℕ := 1

/-- The number of buns Little Boy takes -/
def little_boy_buns : ℕ := bimbo_buns * bimbo_divisor

/-- The number of buns Karlsson takes -/
def karlsson_buns : ℕ := little_boy_buns * karlsson_multiplier

/-- The total number of buns taken -/
def total_taken : ℕ := bimbo_buns + little_boy_buns + karlsson_buns

/-- The number of buns left on the plate -/
def buns_left : ℕ := initial_buns - total_taken

theorem buns_left_is_two : buns_left = 2 := by
  sorry

end NUMINAMATH_CALUDE_buns_left_is_two_l2848_284898


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2848_284890

/-- A game with vectors in a plane -/
structure VectorGame where
  n : ℕ
  vectors : Fin n → ℝ × ℝ

/-- The result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- A strategy for playing the game -/
def Strategy := (n : ℕ) → (remaining : Finset (Fin n)) → Fin n

/-- The game outcome given a strategy for the first player -/
def playGame (game : VectorGame) (strategy : Strategy) : GameResult :=
  sorry

/-- Theorem: The first player has a winning strategy -/
theorem first_player_winning_strategy (game : VectorGame) 
  (h : game.n = 2010) : 
  ∃ (strategy : Strategy), playGame game strategy = GameResult.FirstPlayerWins :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2848_284890


namespace NUMINAMATH_CALUDE_candy_bar_profit_l2848_284814

/-- Calculates the profit from selling candy bars given the number of boxes, bars per box, selling price, and cost price. -/
def calculate_profit (boxes : ℕ) (bars_per_box : ℕ) (selling_price : ℚ) (cost_price : ℚ) : ℚ :=
  let total_bars := boxes * bars_per_box
  let revenue := total_bars * selling_price
  let cost := total_bars * cost_price
  revenue - cost

/-- Proves that the profit from selling 5 boxes of candy bars, with 10 bars per box,
    selling price of $1.50 per bar, and cost price of $1 per bar, is equal to $25. -/
theorem candy_bar_profit :
  calculate_profit 5 10 (3/2) 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l2848_284814


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2848_284840

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 3 * a 7 = 64 →
  a 5 = 8 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2848_284840


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2848_284833

-- Define a geometric sequence
def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

-- Define the problem statement
theorem geometric_sequence_properties
  (a b : ℕ → ℝ)
  (ha : is_geometric_sequence a)
  (hb : is_geometric_sequence b) :
  (is_geometric_sequence (λ n => a n * b n)) ∧
  ¬(∀ x y : ℕ → ℝ, is_geometric_sequence x → is_geometric_sequence y →
    is_geometric_sequence (λ n => x n + y n)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2848_284833


namespace NUMINAMATH_CALUDE_roots_sum_less_than_two_l2848_284813

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the theorem
theorem roots_sum_less_than_two (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ = m → f x₂ = m → x₁ + x₂ < 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_less_than_two_l2848_284813


namespace NUMINAMATH_CALUDE_egg_distribution_l2848_284867

def crate_capacity : ℕ := 18
def abigail_eggs : ℕ := 58
def beatrice_eggs : ℕ := 76
def carson_eggs : ℕ := 27

def total_eggs : ℕ := abigail_eggs + beatrice_eggs + carson_eggs
def full_crates : ℕ := total_eggs / crate_capacity
def remaining_eggs : ℕ := total_eggs % crate_capacity

theorem egg_distribution :
  (remaining_eggs / 3 = 5) ∧
  (remaining_eggs % 3 = 2) ∧
  (abigail_eggs + 6 + beatrice_eggs + 6 + carson_eggs + 5 = total_eggs - full_crates * crate_capacity) := by
  sorry

end NUMINAMATH_CALUDE_egg_distribution_l2848_284867


namespace NUMINAMATH_CALUDE_euro_calculation_l2848_284845

-- Define the € operation
def euro (x y : ℝ) : ℝ := 3 * x * y

-- Theorem statement
theorem euro_calculation : euro 3 (euro 4 5) = 540 := by
  sorry

end NUMINAMATH_CALUDE_euro_calculation_l2848_284845


namespace NUMINAMATH_CALUDE_constant_c_value_l2848_284825

theorem constant_c_value (b c : ℝ) :
  (∀ x : ℝ, 4 * (x + 2) * (x + b) = x^2 + c*x + 12) →
  c = 14 := by
sorry

end NUMINAMATH_CALUDE_constant_c_value_l2848_284825


namespace NUMINAMATH_CALUDE_sixth_grade_boys_count_l2848_284891

/-- Represents the set of boys in the 6th "A" grade. -/
def Boys : Type := Unit

/-- Represents the set of girls in the 6th "A" grade. -/
inductive Girls : Type
  | tanya : Girls
  | dasha : Girls
  | katya : Girls

/-- Represents the friendship relation between boys and girls. -/
def IsFriend : Boys → Girls → Prop := sorry

/-- The number of boys in the 6th "A" grade. -/
def numBoys : ℕ := sorry

theorem sixth_grade_boys_count :
  (∀ (b1 b2 b3 : Boys), ∃ (g : Girls), IsFriend b1 g ∨ IsFriend b2 g ∨ IsFriend b3 g) →
  (∃ (boys : Finset Boys), Finset.card boys = 12 ∧ ∀ b ∈ boys, IsFriend b Girls.tanya) →
  (∃ (boys : Finset Boys), Finset.card boys = 12 ∧ ∀ b ∈ boys, IsFriend b Girls.dasha) →
  (∃ (boys : Finset Boys), Finset.card boys = 13 ∧ ∀ b ∈ boys, IsFriend b Girls.katya) →
  numBoys = 13 ∨ numBoys = 14 := by
  sorry

end NUMINAMATH_CALUDE_sixth_grade_boys_count_l2848_284891


namespace NUMINAMATH_CALUDE_periodic_function_l2848_284895

def isPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

theorem periodic_function (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  isPeriodic f := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_l2848_284895


namespace NUMINAMATH_CALUDE_triangle_area_l2848_284864

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  c^2 = (a - b)^2 + 6 →
  C = π / 3 →
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l2848_284864


namespace NUMINAMATH_CALUDE_mans_walking_speed_l2848_284877

-- Define the given conditions
def walking_time : ℝ := 8
def running_time : ℝ := 2
def running_speed : ℝ := 36

-- Define the walking speed as a variable
def walking_speed : ℝ := sorry

-- Theorem statement
theorem mans_walking_speed :
  walking_speed * walking_time = running_speed * running_time →
  walking_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_mans_walking_speed_l2848_284877


namespace NUMINAMATH_CALUDE_sector_area_proof_l2848_284830

/-- Given a circle where a central angle of 2 radians corresponds to an arc length of 2 cm,
    prove that the area of the sector formed by this central angle is 1 cm². -/
theorem sector_area_proof (r : ℝ) (θ : ℝ) (l : ℝ) (A : ℝ) : 
  θ = 2 → l = 2 → l = r * θ → A = (1/2) * r^2 * θ → A = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_proof_l2848_284830


namespace NUMINAMATH_CALUDE_expression_simplification_l2848_284829

theorem expression_simplification 
  (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) 
  (h4 : c ≠ 0) : 
  a * (1/b + 1/c) + b * (1/a + 1/c) + c * (1/a + 1/b) = -3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2848_284829


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l2848_284859

theorem triangle_is_equilateral (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 + b^2 = c^2 + a*b →
  Real.cos A * Real.cos B = 1/4 →
  A = B ∧ B = C ∧ C = π/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l2848_284859


namespace NUMINAMATH_CALUDE_shift_theorem_l2848_284809

/-- Represents a quadratic function of the form a(x-h)^2 + k --/
structure QuadraticFunction where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a quadratic function horizontally --/
def horizontal_shift (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h + d, k := f.k }

/-- Shifts a quadratic function vertically --/
def vertical_shift (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, h := f.h, k := f.k + d }

/-- The original quadratic function y = 2(x-2)^2 - 5 --/
def original_function : QuadraticFunction :=
  { a := 2, h := 2, k := -5 }

/-- The resulting function after shifts --/
def shifted_function : QuadraticFunction :=
  { a := 2, h := 4, k := -2 }

theorem shift_theorem :
  (vertical_shift (horizontal_shift original_function 2) 3) = shifted_function := by
  sorry

end NUMINAMATH_CALUDE_shift_theorem_l2848_284809


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2848_284863

/-- Given a geometric sequence {a_n} where a_1 = 3 and a_4 = 24, 
    prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                                  -- a_1 = 3
  a 4 = 24 →                                 -- a_4 = 24
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2848_284863


namespace NUMINAMATH_CALUDE_problem_solution_l2848_284834

theorem problem_solution : 
  ((-54 : ℚ) * (-1/2 + 2/3 - 4/9) = 15) ∧ 
  (-2 / (4/9) * (-2/3)^2 = -2) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2848_284834


namespace NUMINAMATH_CALUDE_annual_interest_rate_l2848_284889

/-- Calculate the annual interest rate given the borrowed amount and repayment amount after one year. -/
theorem annual_interest_rate (borrowed : ℝ) (repaid : ℝ) (h1 : borrowed = 150) (h2 : repaid = 165) :
  (repaid - borrowed) / borrowed * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_rate_l2848_284889


namespace NUMINAMATH_CALUDE_multiples_of_ten_range_l2848_284805

theorem multiples_of_ten_range (start : ℕ) : 
  (∃ n : ℕ, n = 991) ∧ 
  (start ≤ 10000) ∧
  (∀ x ∈ Set.Icc start 10000, x % 10 = 0 → x ∈ Finset.range 992) ∧
  (10000 ∈ Finset.range 992) →
  start = 90 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_ten_range_l2848_284805


namespace NUMINAMATH_CALUDE_business_profit_theorem_l2848_284886

def business_profit_distribution (total_profit : ℝ) : ℝ :=
  let majority_owner_share := 0.25 * total_profit
  let remaining_profit := total_profit - majority_owner_share
  let partner_share := 0.25 * remaining_profit
  majority_owner_share + 2 * partner_share

theorem business_profit_theorem :
  business_profit_distribution 80000 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_business_profit_theorem_l2848_284886


namespace NUMINAMATH_CALUDE_class_average_after_exclusion_l2848_284894

theorem class_average_after_exclusion 
  (total_students : ℕ) 
  (total_average : ℚ) 
  (excluded_students : ℕ) 
  (excluded_average : ℚ) : 
  total_students = 10 → 
  total_average = 70 → 
  excluded_students = 5 → 
  excluded_average = 50 → 
  let remaining_students := total_students - excluded_students
  let remaining_total := total_students * total_average - excluded_students * excluded_average
  remaining_total / remaining_students = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_after_exclusion_l2848_284894


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2848_284826

-- Define the logarithm base 3
noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem inequality_equivalence :
  ∀ x : ℝ, (0 < x ∧ |x + log3 x| < |x| + |log3 x|) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2848_284826


namespace NUMINAMATH_CALUDE_sum_of_integers_l2848_284835

theorem sum_of_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2848_284835


namespace NUMINAMATH_CALUDE_friend_ate_two_slices_l2848_284828

/-- Calculates the number of slices James's friend ate given the initial number of slices,
    the number James ate, and the fact that James ate half of the remaining slices. -/
def friend_slices (total : ℕ) (james_ate : ℕ) : ℕ :=
  total - 2 * james_ate

theorem friend_ate_two_slices :
  let total := 8
  let james_ate := 3
  friend_slices total james_ate = 2 := by
  sorry

end NUMINAMATH_CALUDE_friend_ate_two_slices_l2848_284828


namespace NUMINAMATH_CALUDE_remaining_money_proof_l2848_284811

def calculate_remaining_money (initial_amount : ℝ) (sparkling_water_count : ℕ) 
  (sparkling_water_price : ℝ) (sparkling_water_discount : ℝ) 
  (still_water_price : ℝ) (still_water_multiplier : ℕ) 
  (cheddar_weight : ℝ) (cheddar_price : ℝ) 
  (swiss_weight : ℝ) (swiss_price : ℝ) 
  (cheese_tax_rate : ℝ) : ℝ :=
  let sparkling_water_cost := sparkling_water_count * sparkling_water_price * (1 - sparkling_water_discount)
  let still_water_count := sparkling_water_count * still_water_multiplier
  let still_water_paid_bottles := (still_water_count / 3) * 2
  let still_water_cost := still_water_paid_bottles * still_water_price
  let cheese_cost := cheddar_weight * cheddar_price + swiss_weight * swiss_price
  let cheese_tax := cheese_cost * cheese_tax_rate
  let total_cost := sparkling_water_cost + still_water_cost + cheese_cost + cheese_tax
  initial_amount - total_cost

theorem remaining_money_proof :
  calculate_remaining_money 200 4 3 0.1 2.5 3 2.5 8.5 1.75 11 0.05 = 126.67 := by
  sorry

#eval calculate_remaining_money 200 4 3 0.1 2.5 3 2.5 8.5 1.75 11 0.05

end NUMINAMATH_CALUDE_remaining_money_proof_l2848_284811


namespace NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l2848_284854

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_segment : ℝ
  equal_area_segment : ℝ
  midpoint_area_ratio : ℝ × ℝ
  longer_base_diff : longer_base = shorter_base + 150
  midpoint_segment_def : midpoint_segment = shorter_base + 75
  midpoint_area_ratio_def : midpoint_area_ratio = (3, 4)

/-- The main theorem about the trapezoid -/
theorem trapezoid_equal_area_segment (t : Trapezoid) :
  ⌊(t.equal_area_segment^2) / 150⌋ = 187 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_equal_area_segment_l2848_284854


namespace NUMINAMATH_CALUDE_probability_same_color_is_15_364_l2848_284843

def total_marbles : ℕ := 14
def red_marbles : ℕ := 3
def white_marbles : ℕ := 4
def blue_marbles : ℕ := 5
def green_marbles : ℕ := 2

def probability_same_color : ℚ :=
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2))

theorem probability_same_color_is_15_364 :
  probability_same_color = 15 / 364 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_is_15_364_l2848_284843


namespace NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2848_284818

theorem largest_n_for_trig_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / m) ∧
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_trig_inequality_l2848_284818


namespace NUMINAMATH_CALUDE_inscribed_triangle_inequality_l2848_284838

/-- A triangle inscribed in a circle -/
structure InscribedTriangle :=
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (center : ℝ × ℝ)  -- Center of the circumscribed circle
  (radius : ℝ)  -- Radius of the circumscribed circle

/-- Ratio of internal angle bisector to its extension -/
def angle_bisector_ratio (t : InscribedTriangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Sine of an angle in the triangle -/
def triangle_angle_sin (t : InscribedTriangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem inscribed_triangle_inequality (t : InscribedTriangle) :
  let l_a := angle_bisector_ratio t t.A
  let l_b := angle_bisector_ratio t t.B
  let l_c := angle_bisector_ratio t t.C
  let sin_A := triangle_angle_sin t t.A
  let sin_B := triangle_angle_sin t t.B
  let sin_C := triangle_angle_sin t t.C
  l_a / (sin_A * sin_A) + l_b / (sin_B * sin_B) + l_c / (sin_C * sin_C) ≥ 3 ∧
  (l_a / (sin_A * sin_A) + l_b / (sin_B * sin_B) + l_c / (sin_C * sin_C) = 3 ↔ 
   t.A = t.B ∧ t.B = t.C) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_inequality_l2848_284838


namespace NUMINAMATH_CALUDE_travel_ways_proof_l2848_284881

/-- The number of roads from village A to village B -/
def roads_A_to_B : ℕ := 3

/-- The number of roads from village B to village C -/
def roads_B_to_C : ℕ := 2

/-- The total number of ways to travel from village A to village C via village B -/
def total_ways : ℕ := roads_A_to_B * roads_B_to_C

theorem travel_ways_proof : total_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_proof_l2848_284881


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l2848_284803

theorem smallest_integer_with_remainder_two : ∃ n : ℕ, 
  (n > 20) ∧ 
  (∀ m : ℕ, m > 20 → 
    ((m % 3 = 2) ∧ (m % 4 = 2) ∧ (m % 5 = 2) ∧ (m % 6 = 2)) → 
    (n ≤ m)) ∧
  (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) :=
by
  -- The proof goes here
  sorry

#eval Nat.lcm (Nat.lcm 3 4) (Nat.lcm 5 6)  -- This should output 60

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_two_l2848_284803


namespace NUMINAMATH_CALUDE_birds_in_tree_l2848_284873

/-- Given 179 initial birds in a tree and 38 additional birds joining them,
    the total number of birds in the tree is 217. -/
theorem birds_in_tree (initial_birds additional_birds : ℕ) 
  (h1 : initial_birds = 179)
  (h2 : additional_birds = 38) :
  initial_birds + additional_birds = 217 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l2848_284873


namespace NUMINAMATH_CALUDE_min_moves_for_target_vectors_l2848_284899

/-- A tuple of 31 integers -/
def Tuple31 := Fin 31 → ℤ

/-- The set of standard basis vectors -/
def StandardBasis : Set Tuple31 :=
  {v | ∃ i, ∀ j, v j = if i = j then 1 else 0}

/-- The set of target vectors -/
def TargetVectors : Set Tuple31 :=
  {v | ∀ i, v i = if i = 0 then 0 else 1} ∪
  {v | ∀ i, v i = if i = 1 then 0 else 1} ∪
  {v | ∀ i, v i = if i = 30 then 0 else 1}

/-- The operation of adding two vectors -/
def AddVectors (v w : Tuple31) : Tuple31 :=
  λ i => v i + w i

/-- The set of vectors that can be generated in n moves -/
def GeneratedVectors (n : ℕ) : Set Tuple31 :=
  sorry

/-- The theorem statement -/
theorem min_moves_for_target_vectors :
  (∃ n, TargetVectors ⊆ GeneratedVectors n) ∧
  (∀ m, m < 87 → ¬(TargetVectors ⊆ GeneratedVectors m)) :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_target_vectors_l2848_284899


namespace NUMINAMATH_CALUDE_square_remainder_mod_five_l2848_284878

theorem square_remainder_mod_five (n : ℤ) (h : n % 5 = 3) : n^2 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_five_l2848_284878


namespace NUMINAMATH_CALUDE_expression_evaluation_l2848_284832

theorem expression_evaluation :
  let x : ℚ := -1/4
  (x - 1)^2 - 3*x*(1 - x) - (2*x - 1)*(2*x + 1) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2848_284832


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l2848_284872

theorem infinitely_many_pairs_exist : 
  ∀ n : ℕ, ∃ a b : ℕ+, 
    a.val > n ∧ 
    b.val > n ∧ 
    (a.val * b.val) ∣ (a.val^2 + b.val^2 + a.val + b.val + 1) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l2848_284872


namespace NUMINAMATH_CALUDE_root_difference_theorem_l2848_284808

theorem root_difference_theorem (k : ℝ) : 
  (∃ α β : ℝ, (α^2 + k*α + 8 = 0 ∧ β^2 + k*β + 8 = 0) ∧
              ((α+3)^2 - k*(α+3) + 12 = 0 ∧ (β+3)^2 - k*(β+3) + 12 = 0)) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_theorem_l2848_284808


namespace NUMINAMATH_CALUDE_binomial_inequality_l2848_284801

theorem binomial_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) : 
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k : ℝ) * ((n-k)^(n-k) : ℝ)) < 
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n-k).factorial : ℝ)) ∧
  (n.factorial : ℝ) / ((k.factorial : ℝ) * ((n-k).factorial : ℝ)) < 
  (n^n : ℝ) / ((k^k : ℝ) * ((n-k)^(n-k) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_inequality_l2848_284801


namespace NUMINAMATH_CALUDE_train_length_l2848_284866

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 36 → time = 30 → speed * time * (5 / 18) = 300 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l2848_284866


namespace NUMINAMATH_CALUDE_price_per_chicken_l2848_284850

/-- Given Alan's market purchases, prove the price per chicken --/
theorem price_per_chicken (num_eggs : ℕ) (price_per_egg : ℕ) (num_chickens : ℕ) (total_spent : ℕ) :
  num_eggs = 20 →
  price_per_egg = 2 →
  num_chickens = 6 →
  total_spent = 88 →
  (total_spent - num_eggs * price_per_egg) / num_chickens = 8 := by
  sorry

end NUMINAMATH_CALUDE_price_per_chicken_l2848_284850


namespace NUMINAMATH_CALUDE_product_sum_base_k_l2848_284806

theorem product_sum_base_k (k : ℕ) (hk : k > 0) :
  (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5 →
  (3 * k + 14).digits k = [5, 0] :=
by sorry

end NUMINAMATH_CALUDE_product_sum_base_k_l2848_284806


namespace NUMINAMATH_CALUDE_unique_prime_perfect_square_l2848_284842

theorem unique_prime_perfect_square : 
  ∃! p : ℕ, Prime p ∧ ∃ n : ℕ, 5^p + 4*p^4 = n^2 ∧ p = 31 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_square_l2848_284842


namespace NUMINAMATH_CALUDE_min_adventurers_l2848_284846

structure AdventurerGroup where
  rubies : Finset Nat
  emeralds : Finset Nat
  sapphires : Finset Nat
  diamonds : Finset Nat

def AdventurerGroup.valid (g : AdventurerGroup) : Prop :=
  g.rubies.card = 5 ∧
  g.emeralds.card = 11 ∧
  g.sapphires.card = 10 ∧
  g.diamonds.card = 6 ∧
  (∀ a ∈ g.diamonds, (a ∈ g.emeralds ∨ a ∈ g.sapphires) ∧ ¬(a ∈ g.emeralds ∧ a ∈ g.sapphires)) ∧
  (∀ a ∈ g.emeralds, (a ∈ g.rubies ∨ a ∈ g.diamonds) ∧ ¬(a ∈ g.rubies ∧ a ∈ g.diamonds))

theorem min_adventurers (g : AdventurerGroup) (h : g.valid) :
  (g.rubies ∪ g.emeralds ∪ g.sapphires ∪ g.diamonds).card ≥ 16 := by
  sorry

#check min_adventurers

end NUMINAMATH_CALUDE_min_adventurers_l2848_284846


namespace NUMINAMATH_CALUDE_linear_function_properties_l2848_284831

-- Define the linear function
def f (k x : ℝ) : ℝ := (3 - k) * x - 2 * k^2 + 18

theorem linear_function_properties :
  -- Part 1: The function passes through (0, -2) when k = ±√10
  (∃ k : ℝ, k^2 = 10 ∧ f k 0 = -2) ∧
  -- Part 2: The function is parallel to y = -x when k = 4
  (f 4 1 - f 4 0 = -1) ∧
  -- Part 3: The function decreases as x increases when k > 3
  (∀ k : ℝ, k > 3 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f k x₁ > f k x₂) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2848_284831


namespace NUMINAMATH_CALUDE_min_pieces_same_color_l2848_284874

theorem min_pieces_same_color (total_pieces : ℕ) (pieces_per_color : ℕ) (h1 : total_pieces = 60) (h2 : pieces_per_color = 15) :
  ∃ (min_pieces : ℕ), 
    (∀ (n : ℕ), n < min_pieces → ∃ (selection : Finset ℕ), selection.card = n ∧ 
      ∀ (i j : ℕ), i ∈ selection → j ∈ selection → i ≠ j → (i / pieces_per_color) ≠ (j / pieces_per_color)) ∧
    (∃ (selection : Finset ℕ), selection.card = min_pieces ∧ 
      ∃ (i j : ℕ), i ∈ selection ∧ j ∈ selection ∧ i ≠ j ∧ (i / pieces_per_color) = (j / pieces_per_color)) ∧
    min_pieces = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_pieces_same_color_l2848_284874


namespace NUMINAMATH_CALUDE_not_square_expression_l2848_284856

theorem not_square_expression (n : ℕ) (a : ℕ) (h1 : n > 2) (h2 : Odd a) (h3 : a > 0) : 
  let b := 2^(2^n)
  a ≤ b ∧ b ≤ 2*a → ¬ ∃ (k : ℕ), a^2 + b^2 - a*b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_expression_l2848_284856


namespace NUMINAMATH_CALUDE_train_length_l2848_284882

/-- The length of a train given its speed and time to pass an observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 6 → speed_kmh * (1000 / 3600) * time_s = 240 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2848_284882


namespace NUMINAMATH_CALUDE_unique_prime_p_squared_plus_two_prime_l2848_284871

theorem unique_prime_p_squared_plus_two_prime : 
  ∃! p : ℕ, Prime p ∧ Prime (p^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_p_squared_plus_two_prime_l2848_284871


namespace NUMINAMATH_CALUDE_imo_1993_function_exists_l2848_284869

/-- A strictly increasing function from positive integers to positive integers -/
def StrictlyIncreasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m < n → f m < f n

/-- The existence of a function satisfying the IMO 1993 conditions -/
theorem imo_1993_function_exists : ∃ f : ℕ+ → ℕ+, 
  f 1 = 2 ∧ 
  StrictlyIncreasing f ∧ 
  ∀ n : ℕ+, f (f n) = f n + n :=
sorry

end NUMINAMATH_CALUDE_imo_1993_function_exists_l2848_284869


namespace NUMINAMATH_CALUDE_correct_speeds_l2848_284817

/-- Two points moving uniformly along a circumference -/
structure MovingPoints where
  circumference : ℝ
  time_difference : ℝ
  coincidence_interval : ℝ

/-- The speeds of the two points -/
def speeds (mp : MovingPoints) : ℝ × ℝ :=
  sorry

/-- Theorem stating the correct speeds for the given conditions -/
theorem correct_speeds (mp : MovingPoints) 
  (h1 : mp.circumference = 60)
  (h2 : mp.time_difference = 5)
  (h3 : mp.coincidence_interval = 60) :
  speeds mp = (3, 4) :=
sorry

end NUMINAMATH_CALUDE_correct_speeds_l2848_284817


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2848_284884

theorem quadratic_factorization (x : ℝ) : 6*x^2 - 24*x + 18 = 6*(x - 1)*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2848_284884


namespace NUMINAMATH_CALUDE_cylinder_from_equation_l2848_284807

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def isCylinder (S : Set CylindricalPoint) (d : ℝ) : Prop :=
  d > 0 ∧ S = {p : CylindricalPoint | p.r = d}

/-- The main theorem: the set of points satisfying r = d forms a cylinder -/
theorem cylinder_from_equation (d : ℝ) :
  let S := {p : CylindricalPoint | p.r = d}
  d > 0 → isCylinder S d := by
  sorry


end NUMINAMATH_CALUDE_cylinder_from_equation_l2848_284807


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l2848_284821

theorem function_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l2848_284821


namespace NUMINAMATH_CALUDE_oldest_babysat_age_l2848_284824

/-- Represents Jane's babysitting career and age information -/
structure BabysittingCareer where
  current_age : ℕ
  years_since_stopped : ℕ
  start_age : ℕ

/-- Calculates the maximum age of a child Jane could babysit at a given time -/
def max_child_age (jane_age : ℕ) : ℕ :=
  jane_age / 2

/-- Theorem stating the current age of the oldest person Jane could have babysat -/
theorem oldest_babysat_age (jane : BabysittingCareer)
  (h1 : jane.current_age = 34)
  (h2 : jane.years_since_stopped = 10)
  (h3 : jane.start_age = 18) :
  jane.current_age - jane.years_since_stopped - max_child_age (jane.current_age - jane.years_since_stopped) + jane.years_since_stopped = 22 :=
by
  sorry

#check oldest_babysat_age

end NUMINAMATH_CALUDE_oldest_babysat_age_l2848_284824


namespace NUMINAMATH_CALUDE_median_mean_difference_l2848_284896

structure ArticleData where
  frequencies : List (Nat × Nat)
  total_students : Nat
  sum_articles : Nat

def median (data : ArticleData) : Rat := 2

def mean (data : ArticleData) : Rat := data.sum_articles / data.total_students

theorem median_mean_difference (data : ArticleData) 
  (h1 : data.frequencies = [(0, 4), (1, 3), (2, 2), (3, 2), (4, 3), (5, 4)])
  (h2 : data.total_students = 18)
  (h3 : data.sum_articles = 45) :
  mean data - median data = 1/2 := by sorry

end NUMINAMATH_CALUDE_median_mean_difference_l2848_284896


namespace NUMINAMATH_CALUDE_frank_jim_speed_difference_l2848_284857

theorem frank_jim_speed_difference : 
  ∀ (jim_distance frank_distance : ℝ) (time : ℝ),
    jim_distance = 16 →
    frank_distance = 20 →
    time = 2 →
    (frank_distance / time) - (jim_distance / time) = 2 := by
  sorry

end NUMINAMATH_CALUDE_frank_jim_speed_difference_l2848_284857


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l2848_284893

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) 
  (h1 : total_packages = 27) 
  (h2 : total_pieces = 486) : 
  total_pieces / total_packages = 18 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l2848_284893


namespace NUMINAMATH_CALUDE_cosine_even_and_decreasing_l2848_284849

-- Define the properties of evenness and decreasing for a function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def IsDecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem cosine_even_and_decreasing :
  IsEven Real.cos ∧ IsDecreasingOn Real.cos 0 3 := by sorry

end NUMINAMATH_CALUDE_cosine_even_and_decreasing_l2848_284849


namespace NUMINAMATH_CALUDE_smallest_divisor_after_323_l2848_284861

theorem smallest_divisor_after_323 (n : ℕ) (h1 : 1000 ≤ n ∧ n < 10000) 
  (h2 : Even n) (h3 : n % 323 = 0) :
  (∃ k : ℕ, k > 323 ∧ n % k = 0 ∧ ∀ m : ℕ, m > 323 ∧ n % m = 0 → k ≤ m) ∧
  (∀ k : ℕ, k > 323 ∧ n % k = 0 ∧ (∀ m : ℕ, m > 323 ∧ n % m = 0 → k ≤ m) → k = 340) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_after_323_l2848_284861


namespace NUMINAMATH_CALUDE_factor_expression_l2848_284815

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2848_284815


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2848_284837

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_complement_equality : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2848_284837


namespace NUMINAMATH_CALUDE_park_is_square_l2848_284860

/-- A shape with a certain number of 90-degree angles -/
structure Shape :=
  (angles : ℕ)

/-- Definition of a square -/
def is_square (s : Shape) : Prop := s.angles = 4

theorem park_is_square (park : Shape) (square_field : Shape)
  (h1 : is_square square_field)
  (h2 : park.angles + square_field.angles = 8) :
  is_square park :=
sorry

end NUMINAMATH_CALUDE_park_is_square_l2848_284860


namespace NUMINAMATH_CALUDE_crossing_over_result_l2848_284868

/-- Represents a chromatid with its staining pattern -/
structure Chromatid where
  staining : ℕ → Bool  -- True for darker staining, False for lighter

/-- Represents a chromosome with two sister chromatids -/
structure Chromosome where
  chromatid1 : Chromatid
  chromatid2 : Chromatid

/-- Represents the process of DNA replication with BrdU -/
def dnaReplication (c : Chromosome) : Chromosome :=
  { chromatid1 := { staining := fun _ => true },
    chromatid2 := c.chromatid1 }

/-- Represents the process of crossing over between sister chromatids -/
def crossingOver (c : Chromosome) : Chromosome :=
  { chromatid1 := { staining := fun n => if n % 2 = 0 then c.chromatid1.staining n else c.chromatid2.staining n },
    chromatid2 := { staining := fun n => if n % 2 = 0 then c.chromatid2.staining n else c.chromatid1.staining n } }

/-- Theorem stating the result of the experiment -/
theorem crossing_over_result (initialChromosome : Chromosome) :
  ∃ (n m : ℕ), 
    let finalChromosome := crossingOver (dnaReplication (dnaReplication initialChromosome))
    finalChromosome.chromatid1.staining n ≠ finalChromosome.chromatid1.staining m ∧
    finalChromosome.chromatid2.staining n ≠ finalChromosome.chromatid2.staining m :=
  sorry


end NUMINAMATH_CALUDE_crossing_over_result_l2848_284868


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2848_284839

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ 
  m ∈ Set.Ioo (-1/5 : ℝ) 3 ∪ {3} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2848_284839


namespace NUMINAMATH_CALUDE_max_value_on_parabola_l2848_284804

/-- The maximum value of m + n where (m,n) lies on y = -x^2 + 3 is 13/4 -/
theorem max_value_on_parabola :
  ∀ m n : ℝ, n = -m^2 + 3 → (∀ x y : ℝ, y = -x^2 + 3 → m + n ≥ x + y) → m + n = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_parabola_l2848_284804


namespace NUMINAMATH_CALUDE_sqrt_sum_ge_product_sum_l2848_284870

theorem sqrt_sum_ge_product_sum {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ a * b + b * c + c * a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_ge_product_sum_l2848_284870


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2848_284875

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- Conditions for the ages -/
def age_conditions (a : Ages) : Prop :=
  a.roy = a.julia + 6 ∧
  a.roy + 2 = 2 * (a.julia + 2) ∧
  (a.roy + 2) * (a.kelly + 2) = 108

/-- The theorem to be proved -/
theorem age_ratio_is_two_to_one (a : Ages) :
  age_conditions a →
  (a.roy - a.julia) / (a.roy - a.kelly) = 2 := by
  sorry

#check age_ratio_is_two_to_one

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l2848_284875


namespace NUMINAMATH_CALUDE_given_number_scientific_notation_l2848_284810

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number to be converted -/
def given_number : ℝ := 0.00000164

/-- The expected scientific notation representation -/
def expected_notation : ScientificNotation := {
  coefficient := 1.64,
  exponent := -6,
  is_valid := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_scientific_notation : given_number = expected_notation.coefficient * (10 : ℝ) ^ expected_notation.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_scientific_notation_l2848_284810


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_necessary_not_sufficient_l2848_284812

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 2*x - 8 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Theorem for the first part of the problem
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) → x ∈ Set.Ioo 2 4 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_not_p_necessary_not_sufficient :
  ∀ a : ℝ, (∀ x : ℝ, ¬(q x) → ¬(p x a)) ∧ (∃ x : ℝ, ¬(p x a) ∧ q x) → a ∈ Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_for_not_p_necessary_not_sufficient_l2848_284812


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_on_unit_circle_l2848_284879

/-- Given complex numbers on the unit circle represented by their real and imaginary parts,
    prove that the cosine of the sum of their arguments is as specified. -/
theorem cos_sum_of_complex_on_unit_circle
  (γ δ : ℝ)
  (h1 : Complex.exp (Complex.I * γ) = Complex.ofReal (8/17) + Complex.I * (15/17))
  (h2 : Complex.exp (Complex.I * δ) = Complex.ofReal (3/5) - Complex.I * (4/5)) :
  Real.cos (γ + δ) = 84/85 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_on_unit_circle_l2848_284879


namespace NUMINAMATH_CALUDE_randys_trip_distance_l2848_284876

theorem randys_trip_distance :
  ∀ y : ℝ,
  (y / 4 : ℝ) + 30 + (y / 3 : ℝ) = y →
  y = 72 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_distance_l2848_284876


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2848_284816

theorem solution_set_equivalence (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (h : ∀ x : ℝ, mx + n > 0 ↔ x > 2/5) : 
  ∀ x : ℝ, nx - m < 0 ↔ x > -5/2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2848_284816


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2848_284897

theorem sqrt_sum_squares_eq_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a + b + c ≥ 0 ∧ a*b + a*c + b*c = 0 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l2848_284897


namespace NUMINAMATH_CALUDE_sarah_molly_groups_l2848_284852

def chess_club_size : ℕ := 12
def group_size : ℕ := 6

theorem sarah_molly_groups (sarah molly : Fin chess_club_size) 
  (h_distinct : sarah ≠ molly) : 
  (Finset.univ.filter (λ s : Finset (Fin chess_club_size) => 
    s.card = group_size ∧ sarah ∈ s ∧ molly ∈ s)).card = 210 := by
  sorry

end NUMINAMATH_CALUDE_sarah_molly_groups_l2848_284852


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_twenty_l2848_284853

def sequence_term (n : ℕ) : ℤ := (-1)^(n+1) * n * (n+1)

theorem fourth_term_is_negative_twenty : sequence_term 4 = -20 := by sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_twenty_l2848_284853


namespace NUMINAMATH_CALUDE_divisible_by_nine_l2848_284827

theorem divisible_by_nine : ∃ (B : ℕ), B < 10 ∧ (7000 + 600 + 20 + B) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l2848_284827


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2848_284855

theorem rectangular_solid_diagonal (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 54)
  (h2 : 4 * (a + b + c) = 40)
  (h3 : c = a + b) :
  a^2 + b^2 + c^2 = 46 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2848_284855


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l2848_284892

/-- Given two lines in the real plane, determine if a specific value of a parameter is sufficient but not necessary for their parallelism. -/
theorem parallel_lines_condition (a : ℝ) : 
  (∃ (x y : ℝ), a * x + 2 * y - 1 = 0) →  -- l₁ exists
  (∃ (x y : ℝ), x + (a + 1) * y + 4 = 0) →  -- l₂ exists
  (a = 1 → (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    a * x₁ + 2 * y₁ - 1 = 0 → 
    x₂ + (a + 1) * y₂ + 4 = 0 → 
    (y₂ - y₁) * (1 - 0) = (x₂ - x₁) * (2 - (a + 1)))) ∧ 
  (∃ b : ℝ, b ≠ 1 ∧ 
    (∀ (x₁ y₁ x₂ y₂ : ℝ), 
      b * x₁ + 2 * y₁ - 1 = 0 → 
      x₂ + (b + 1) * y₂ + 4 = 0 → 
      (y₂ - y₁) * (1 - 0) = (x₂ - x₁) * (2 - (b + 1)))) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l2848_284892


namespace NUMINAMATH_CALUDE_problem_solution_l2848_284822

theorem problem_solution (x : ℚ) : x = (1 / x) * (-x) - 3 * x + 4 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2848_284822


namespace NUMINAMATH_CALUDE_ravi_coin_value_l2848_284880

/-- Represents the number of coins of each type Ravi has -/
structure CoinCounts where
  nickels : ℕ
  quarters : ℕ
  dimes : ℕ
  half_dollars : ℕ
  pennies : ℕ

/-- Calculates the total value of coins in cents -/
def total_value (counts : CoinCounts) : ℕ :=
  counts.nickels * 5 +
  counts.quarters * 25 +
  counts.dimes * 10 +
  counts.half_dollars * 50 +
  counts.pennies * 1

/-- Theorem stating that Ravi's coin collection is worth $12.51 -/
theorem ravi_coin_value : ∃ (counts : CoinCounts),
  counts.nickels = 6 ∧
  counts.quarters = counts.nickels + 2 ∧
  counts.dimes = counts.quarters + 4 ∧
  counts.half_dollars = counts.dimes + 5 ∧
  counts.pennies = counts.half_dollars * 3 ∧
  total_value counts = 1251 := by
  sorry

end NUMINAMATH_CALUDE_ravi_coin_value_l2848_284880
