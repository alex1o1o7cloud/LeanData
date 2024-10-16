import Mathlib

namespace NUMINAMATH_CALUDE_democrat_count_l3105_310540

theorem democrat_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 810 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (1 / 3 : ℚ) * total →
  female / 2 = 135 :=
by
  sorry

end NUMINAMATH_CALUDE_democrat_count_l3105_310540


namespace NUMINAMATH_CALUDE_ones_digit_of_prime_arithmetic_sequence_l3105_310535

theorem ones_digit_of_prime_arithmetic_sequence (a b c d : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧  -- Four prime numbers
  a > 5 ∧                                  -- a is greater than 5
  b = a + 6 ∧ c = b + 6 ∧ d = c + 6 ∧      -- Arithmetic sequence with common difference 6
  a < b ∧ b < c ∧ c < d →                  -- Increasing sequence
  a % 10 = 1 :=                            -- The ones digit of a is 1
by sorry

end NUMINAMATH_CALUDE_ones_digit_of_prime_arithmetic_sequence_l3105_310535


namespace NUMINAMATH_CALUDE_divisible_by_five_l3105_310594

theorem divisible_by_five (B : Nat) : B < 10 → (647 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l3105_310594


namespace NUMINAMATH_CALUDE_expression_value_l3105_310543

theorem expression_value (x y : ℝ) (h : x - 2*y = 3) : 5 - 2*x + 4*y = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3105_310543


namespace NUMINAMATH_CALUDE_supermarket_spending_l3105_310531

theorem supermarket_spending (total : ℚ) :
  (1 / 4 : ℚ) * total +
  (1 / 3 : ℚ) * total +
  (1 / 6 : ℚ) * total +
  6 = total →
  total = 24 := by
sorry

end NUMINAMATH_CALUDE_supermarket_spending_l3105_310531


namespace NUMINAMATH_CALUDE_tiling_count_is_96_l3105_310580

/-- Represents a tile with width and height -/
structure Tile :=
  (width : Nat)
  (height : Nat)

/-- Represents a rectangle with width and height -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Represents a set of tiles -/
def TileSet := List Tile

/-- Counts the number of ways to tile a rectangle with a given set of tiles -/
def tileCount (r : Rectangle) (ts : TileSet) : Nat :=
  sorry

/-- The set of tiles for our problem -/
def problemTiles : TileSet :=
  [⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 3⟩, ⟨1, 4⟩, ⟨1, 5⟩]

/-- The main theorem stating that the number of tilings is 96 -/
theorem tiling_count_is_96 :
  tileCount ⟨5, 3⟩ problemTiles = 96 :=
sorry

end NUMINAMATH_CALUDE_tiling_count_is_96_l3105_310580


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l3105_310550

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 246 → ¬(∃ k : ℤ, n^3 + 150 = k * (n + 12)) ∧
  ∃ k : ℤ, 246^3 + 150 = k * (246 + 12) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l3105_310550


namespace NUMINAMATH_CALUDE_smallBase_altitude_ratio_l3105_310507

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  smallBase : ℝ
  /-- Length of the larger base -/
  largeBase : ℝ
  /-- Length of the diagonal -/
  diagonal : ℝ
  /-- Length of the altitude -/
  altitude : ℝ
  /-- The larger base is twice the smaller base -/
  largeBase_eq : largeBase = 2 * smallBase
  /-- The diagonal is 1.5 times the larger base -/
  diagonal_eq : diagonal = 1.5 * largeBase
  /-- The altitude equals the smaller base -/
  altitude_eq : altitude = smallBase

/-- Theorem: The ratio of the smaller base to the altitude is 1:1 -/
theorem smallBase_altitude_ratio (t : IsoscelesTrapezoid) : t.smallBase / t.altitude = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallBase_altitude_ratio_l3105_310507


namespace NUMINAMATH_CALUDE_tan_two_simplification_l3105_310593

theorem tan_two_simplification (x : ℝ) (h : Real.tan x = 2) :
  (2 * Real.sin x + Real.cos x) / (2 * Real.sin x - Real.cos x) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_simplification_l3105_310593


namespace NUMINAMATH_CALUDE_cos_squared_165_minus_sin_squared_15_l3105_310526

theorem cos_squared_165_minus_sin_squared_15 :
  Real.cos (165 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_165_minus_sin_squared_15_l3105_310526


namespace NUMINAMATH_CALUDE_min_value_part1_min_value_part2_l3105_310541

-- Part 1
theorem min_value_part1 (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) + 6 ≥ 9 ∧
  (x + 4 / (x + 1) + 6 = 9 ↔ x = 1) :=
sorry

-- Part 2
theorem min_value_part2 (x : ℝ) (h : x > 1) :
  (x^2 + 8) / (x - 1) ≥ 8 ∧
  ((x^2 + 8) / (x - 1) = 8 ↔ x = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_part1_min_value_part2_l3105_310541


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l3105_310564

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocks (box : Dimensions) (block : Dimensions) : ℕ :=
  (box.height / block.height) * (box.width / block.width) * (box.length / block.length)

/-- The box dimensions -/
def boxDim : Dimensions := ⟨8, 10, 12⟩

/-- Type A block dimensions -/
def blockADim : Dimensions := ⟨3, 2, 4⟩

/-- Type B block dimensions -/
def blockBDim : Dimensions := ⟨4, 3, 5⟩

theorem max_blocks_in_box :
  max (maxBlocks boxDim blockADim) (maxBlocks boxDim blockBDim) = 30 := by
  sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l3105_310564


namespace NUMINAMATH_CALUDE_remainder_of_binary_number_mod_4_l3105_310574

def binary_number : ℕ := 111100010111

theorem remainder_of_binary_number_mod_4 :
  binary_number % 4 = 3 := by sorry

end NUMINAMATH_CALUDE_remainder_of_binary_number_mod_4_l3105_310574


namespace NUMINAMATH_CALUDE_midpoint_x_sum_eq_vertex_x_sum_l3105_310572

/-- Given a triangle in the Cartesian plane, the sum of the x-coordinates of the midpoints
    of its sides is equal to the sum of the x-coordinates of its vertices. -/
theorem midpoint_x_sum_eq_vertex_x_sum (a b c : ℝ) : 
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (a + c) / 2 + (b + c) / 2
  midpoint_sum = vertex_sum :=
by sorry

end NUMINAMATH_CALUDE_midpoint_x_sum_eq_vertex_x_sum_l3105_310572


namespace NUMINAMATH_CALUDE_horner_method_v2_l3105_310512

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 - x + 5

def horner_v2 (a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₅ * x + a₄) * x + a₃) * x + a₂

theorem horner_method_v2 :
  horner_v2 2 0 (-3) 2 (-1) 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l3105_310512


namespace NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l3105_310534

theorem real_part_of_inverse_one_minus_z_squared (z : ℂ) 
  (h1 : z ≠ (z.re : ℂ)) -- z is nonreal
  (h2 : Complex.abs z = 1) :
  Complex.re (1 / (1 - z^2)) = (1 - z.re^2) / 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l3105_310534


namespace NUMINAMATH_CALUDE_rectangle_y_value_l3105_310528

/-- Given a rectangle with vertices at (-2, y), (6, y), (-2, 2), and (6, 2),
    if the area is 80 square units, then y = 12 -/
theorem rectangle_y_value (y : ℝ) : 
  let vertices : List (ℝ × ℝ) := [(-2, y), (6, y), (-2, 2), (6, 2)]
  let width : ℝ := 6 - (-2)
  let height : ℝ := y - 2
  let area : ℝ := width * height
  (∀ v ∈ vertices, v.1 = -2 ∨ v.1 = 6) ∧
  (∀ v ∈ vertices, v.2 = y ∨ v.2 = 2) ∧
  area = 80 →
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l3105_310528


namespace NUMINAMATH_CALUDE_lg_100_equals_2_l3105_310568

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_100_equals_2 : lg 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_100_equals_2_l3105_310568


namespace NUMINAMATH_CALUDE_boat_upstream_speed_l3105_310516

/-- The speed of a boat upstream given its speed in still water and the speed of the current. -/
def speed_upstream (speed_still : ℝ) (speed_current : ℝ) : ℝ :=
  speed_still - speed_current

/-- Theorem: Given a boat with speed 50 km/h in still water and a current with speed 20 km/h,
    the speed of the boat upstream is 30 km/h. -/
theorem boat_upstream_speed :
  let speed_still : ℝ := 50
  let speed_current : ℝ := 20
  speed_upstream speed_still speed_current = 30 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_speed_l3105_310516


namespace NUMINAMATH_CALUDE_circle_O1_equation_constant_sum_of_squares_l3105_310566

-- Define the circles and points
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_O1 (x y m : ℝ) : Prop := (x - m)^2 + y^2 = (m - 3)^2 + 4^2

-- Define the intersection point P
def P : ℝ × ℝ := (3, 4)

-- Define the line l
def line_l (x y k : ℝ) : Prop := y - 4 = k * (x - 3)

-- Define the perpendicular line l1
def line_l1 (x y k : ℝ) : Prop := y - 4 = (-1/k) * (x - 3)

-- Theorem 1
theorem circle_O1_equation (m : ℝ) : 
  (∃ B : ℝ × ℝ, circle_O1 B.1 B.2 m ∧ line_l B.1 B.2 1 ∧ (B.1 - 3)^2 + (B.2 - 4)^2 = 98) →
  (∀ x y : ℝ, circle_O1 x y m ↔ (x - 14)^2 + y^2 = 137) :=
sorry

-- Theorem 2
theorem constant_sum_of_squares (m : ℝ) :
  ∀ k : ℝ, k ≠ 0 →
  (∃ A B C D : ℝ × ℝ, 
    circle_O A.1 A.2 ∧ circle_O1 B.1 B.2 m ∧ line_l A.1 A.2 k ∧ line_l B.1 B.2 k ∧
    circle_O C.1 C.2 ∧ circle_O1 D.1 D.2 m ∧ line_l1 C.1 C.2 k ∧ line_l1 D.1 D.2 k) →
  (∃ A B C D : ℝ × ℝ, 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = 4 * m^2) :=
sorry

end NUMINAMATH_CALUDE_circle_O1_equation_constant_sum_of_squares_l3105_310566


namespace NUMINAMATH_CALUDE_f_inequality_l3105_310523

open Real

-- Define a derivable function f on ℝ
variable (f : ℝ → ℝ)

-- Define the condition that f is twice differentiable
variable (hf : TwiceDifferentiable ℝ f)

-- Define the condition 3f(x) > f''(x) for all x ∈ ℝ
variable (h1 : ∀ x : ℝ, 3 * f x > (deriv^[2] f) x)

-- Define the condition f(1) = e^3
variable (h2 : f 1 = exp 3)

-- State the theorem
theorem f_inequality : f 2 < exp 6 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l3105_310523


namespace NUMINAMATH_CALUDE_factors_of_1320_l3105_310562

theorem factors_of_1320 : Finset.card (Nat.divisors 1320) = 24 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l3105_310562


namespace NUMINAMATH_CALUDE_floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1_l3105_310529

theorem floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1 : 
  Int.floor (Real.sqrt 45)^2 + 2 * Int.floor (Real.sqrt 45) + 1 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_45_squared_plus_twice_floor_sqrt_45_plus_1_l3105_310529


namespace NUMINAMATH_CALUDE_no_solution_exists_l3105_310513

theorem no_solution_exists : ¬∃ (a b c d : ℕ), 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧ 
  71 * a + 72 * b + 73 * c + 74 * d = 2014 :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3105_310513


namespace NUMINAMATH_CALUDE_pen_price_relationship_l3105_310581

/-- Given a box of pens with a selling price of $16 and containing 10 pens,
    prove that the relationship between the selling price of one pen (y)
    and the number of pens (x) is y = 1.6x. -/
theorem pen_price_relationship (box_price : ℝ) (pens_per_box : ℕ) (x y : ℝ) :
  box_price = 16 →
  pens_per_box = 10 →
  y = (box_price / pens_per_box) * x →
  y = 1.6 * x :=
by
  sorry


end NUMINAMATH_CALUDE_pen_price_relationship_l3105_310581


namespace NUMINAMATH_CALUDE_max_value_difference_l3105_310504

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define a as the point where f(x) reaches its maximum value
def a : ℝ := 1

-- Define b as the maximum value of f(x)
def b : ℝ := f a

-- Theorem statement
theorem max_value_difference (x : ℝ) : a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_difference_l3105_310504


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_main_theorem_l3105_310539

/-- Represents a rhombus with given properties -/
structure Rhombus where
  diagonal1 : ℝ
  perimeter : ℝ
  diagonal2 : ℝ

/-- Theorem stating the relationship between the diagonals and perimeter of a specific rhombus -/
theorem rhombus_diagonal_length (r : Rhombus) 
    (h1 : r.diagonal1 = 10)
    (h2 : r.perimeter = 52) : 
    r.diagonal2 = 24 := by
  sorry

/-- Main theorem to be proved -/
theorem main_theorem : ∃ r : Rhombus, r.diagonal1 = 10 ∧ r.perimeter = 52 ∧ r.diagonal2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_main_theorem_l3105_310539


namespace NUMINAMATH_CALUDE_equilateral_triangle_min_rotation_angle_l3105_310532

/-- An equilateral triangle is rotationally symmetric -/
structure EquilateralTriangle where
  is_rotationally_symmetric : Bool

/-- The minimum rotation angle of a rotationally symmetric figure -/
def minimum_rotation_angle (figure : EquilateralTriangle) : ℝ :=
  sorry

/-- Theorem: The minimum rotation angle of an equilateral triangle is 120 degrees -/
theorem equilateral_triangle_min_rotation_angle (t : EquilateralTriangle) :
  minimum_rotation_angle t = 120 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_min_rotation_angle_l3105_310532


namespace NUMINAMATH_CALUDE_quadratic_absolute_inequality_l3105_310596

theorem quadratic_absolute_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_absolute_inequality_l3105_310596


namespace NUMINAMATH_CALUDE_square_division_theorem_l3105_310502

/-- A type representing a square division -/
structure SquareDivision where
  n : ℕ
  is_valid : Bool

/-- Function that checks if a square can be divided into n smaller squares -/
def can_divide_square (n : ℕ) : Prop :=
  ∃ (sd : SquareDivision), sd.n = n ∧ sd.is_valid = true

theorem square_division_theorem :
  (∀ n : ℕ, n > 5 → can_divide_square n) ∧
  ¬(can_divide_square 2) ∧
  ¬(can_divide_square 3) := by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l3105_310502


namespace NUMINAMATH_CALUDE_cocktail_theorem_l3105_310508

def cocktail_proof (initial_volume : ℝ) (jasmine_percent : ℝ) (rose_percent : ℝ) (mint_percent : ℝ)
  (added_jasmine : ℝ) (added_rose : ℝ) (added_mint : ℝ) (added_plain : ℝ) : Prop :=
  let initial_jasmine := initial_volume * jasmine_percent
  let initial_rose := initial_volume * rose_percent
  let initial_mint := initial_volume * mint_percent
  let new_jasmine := initial_jasmine + added_jasmine
  let new_rose := initial_rose + added_rose
  let new_mint := initial_mint + added_mint
  let new_volume := initial_volume + added_jasmine + added_rose + added_mint + added_plain
  let new_percent := (new_jasmine + new_rose + new_mint) / new_volume * 100
  new_percent = 21.91

theorem cocktail_theorem :
  cocktail_proof 150 0.03 0.05 0.02 12 9 3 4 := by
  sorry

end NUMINAMATH_CALUDE_cocktail_theorem_l3105_310508


namespace NUMINAMATH_CALUDE_double_factorial_properties_l3105_310571

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def units_digit (n : ℕ) : ℕ := n % 10

theorem double_factorial_properties :
  (double_factorial 2003 * double_factorial 2002 = Nat.factorial 2003) ∧
  (double_factorial 2002 = 2^1001 * Nat.factorial 1001) ∧
  (units_digit (double_factorial 2002) = 0) ∧
  (units_digit (double_factorial 2003) = 5) := by
  sorry

#check double_factorial_properties

end NUMINAMATH_CALUDE_double_factorial_properties_l3105_310571


namespace NUMINAMATH_CALUDE_group_size_l3105_310506

theorem group_size (num_children : ℕ) (num_women : ℕ) (num_men : ℕ) : 
  num_children = 30 →
  num_women = 3 * num_children →
  num_men = 2 * num_women →
  num_children + num_women + num_men = 300 := by
  sorry

#check group_size

end NUMINAMATH_CALUDE_group_size_l3105_310506


namespace NUMINAMATH_CALUDE_diophantine_approximation_l3105_310525

theorem diophantine_approximation (α : ℝ) (C : ℝ) (h_α : α > 0) (h_C : C > 1) :
  ∃ (x : ℕ) (y : ℤ), (x : ℝ) < C ∧ |x * α - y| ≤ 1 / C := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l3105_310525


namespace NUMINAMATH_CALUDE_frog_final_position_probability_l3105_310591

noncomputable def frog_jump_probability : ℝ := 
  let n : ℕ := 4  -- number of jumps
  let jump_length : ℝ := 1  -- length of each jump
  let max_distance : ℝ := 1.5  -- maximum distance from starting point
  1/3  -- probability

theorem frog_final_position_probability :
  frog_jump_probability = 1/3 :=
sorry

end NUMINAMATH_CALUDE_frog_final_position_probability_l3105_310591


namespace NUMINAMATH_CALUDE_bijection_probability_l3105_310542

-- Define sets A and B
def A : Set (Fin 2) := Set.univ
def B : Set (Fin 3) := Set.univ

-- Define the total number of mappings from A to B
def total_mappings : ℕ := 3^2

-- Define the number of bijective mappings from A to B
def bijective_mappings : ℕ := 3 * 2

-- Define the probability of a random mapping being bijective
def prob_bijective : ℚ := bijective_mappings / total_mappings

-- Theorem statement
theorem bijection_probability :
  prob_bijective = 2/3 := by sorry

end NUMINAMATH_CALUDE_bijection_probability_l3105_310542


namespace NUMINAMATH_CALUDE_ahn_max_number_l3105_310586

theorem ahn_max_number : ∃ (m : ℕ), m = 870 ∧ 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ahn_max_number_l3105_310586


namespace NUMINAMATH_CALUDE_solve_rain_problem_l3105_310544

def rain_problem (x : ℝ) : Prop :=
  let monday_total := x + 1
  let tuesday := 2 * monday_total
  let wednesday := 0
  let thursday := 1
  let friday := monday_total + tuesday + wednesday + thursday
  let total_rain := monday_total + tuesday + wednesday + thursday + friday
  let daily_average := 4
  total_rain = 7 * daily_average ∧ x > 0

theorem solve_rain_problem :
  ∃ x : ℝ, rain_problem x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_rain_problem_l3105_310544


namespace NUMINAMATH_CALUDE_desired_average_sale_l3105_310567

theorem desired_average_sale (sales : List ℕ) (desired_sixth : ℕ) : 
  sales = [6235, 6927, 6855, 7230, 6562] → 
  desired_sixth = 5191 → 
  (sales.sum + desired_sixth) / 6 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_desired_average_sale_l3105_310567


namespace NUMINAMATH_CALUDE_number_added_to_x_l3105_310585

theorem number_added_to_x (x : ℝ) (some_number : ℝ) : 
  x + some_number = 2 → x = 1 → some_number = 1 := by
sorry

end NUMINAMATH_CALUDE_number_added_to_x_l3105_310585


namespace NUMINAMATH_CALUDE_squared_differences_inequality_l3105_310582

theorem squared_differences_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  min ((a - b)^2) (min ((b - c)^2) ((c - a)^2)) ≤ (a^2 + b^2 + c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_squared_differences_inequality_l3105_310582


namespace NUMINAMATH_CALUDE_scientific_notation_of_600000_l3105_310536

theorem scientific_notation_of_600000 : ∃ (a : ℝ) (n : ℤ), 
  1 ≤ a ∧ a < 10 ∧ 600000 = a * (10 : ℝ) ^ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_600000_l3105_310536


namespace NUMINAMATH_CALUDE_function_composition_equality_l3105_310538

/-- Given two functions p and q, where p(x) = 5x - 4 and q(x) = 4x - b,
    prove that if p(q(5)) = 16, then b = 16. -/
theorem function_composition_equality (b : ℝ) : 
  (let p : ℝ → ℝ := λ x => 5 * x - 4
   let q : ℝ → ℝ := λ x => 4 * x - b
   p (q 5) = 16) → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3105_310538


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3105_310554

/-- Given a line with equation y - 7 = -3(x - 5), 
    the sum of its x-intercept and y-intercept is 88/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 7 = -3 * (x - 5)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 7 = -3 * (x_int - 5)) ∧ 
    (0 - 7 = -3 * (x_int - 5)) ∧ 
    (y_int - 7 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 88 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3105_310554


namespace NUMINAMATH_CALUDE_biscuit_dimensions_l3105_310530

theorem biscuit_dimensions (sheet_side : ℝ) (num_biscuits : ℕ) (biscuit_side : ℝ) : 
  sheet_side = 12 →
  num_biscuits = 16 →
  (sheet_side * sheet_side) = (biscuit_side * biscuit_side * num_biscuits) →
  biscuit_side = 3 := by
sorry

end NUMINAMATH_CALUDE_biscuit_dimensions_l3105_310530


namespace NUMINAMATH_CALUDE_g_of_one_equals_fifteen_l3105_310576

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_one_equals_fifteen :
  (∀ x : ℝ, g (2 * x - 3) = 3 * x + 9) →
  g 1 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_one_equals_fifteen_l3105_310576


namespace NUMINAMATH_CALUDE_log_54883_between_consecutive_integers_l3105_310578

theorem log_54883_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 54883 / Real.log 10 ∧ Real.log 54883 / Real.log 10 < (d : ℝ) → c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_54883_between_consecutive_integers_l3105_310578


namespace NUMINAMATH_CALUDE_davids_pushups_l3105_310561

theorem davids_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) :
  zachary_pushups = 47 →
  david_extra_pushups = 15 →
  zachary_pushups + david_extra_pushups = 62 :=
by sorry

end NUMINAMATH_CALUDE_davids_pushups_l3105_310561


namespace NUMINAMATH_CALUDE_valentines_day_treats_cost_l3105_310595

/-- The cost of Valentine's Day treats for two dogs -/
def total_cost (heart_biscuit_cost puppy_boots_cost : ℕ) : ℕ :=
  let dog_a_cost := 5 * heart_biscuit_cost + puppy_boots_cost
  let dog_b_cost := 7 * heart_biscuit_cost + 2 * puppy_boots_cost
  dog_a_cost + dog_b_cost

/-- Theorem stating the total cost of Valentine's Day treats for two dogs -/
theorem valentines_day_treats_cost :
  total_cost 2 15 = 69 := by
  sorry

end NUMINAMATH_CALUDE_valentines_day_treats_cost_l3105_310595


namespace NUMINAMATH_CALUDE_max_value_of_sum_over_square_n_l3105_310598

theorem max_value_of_sum_over_square_n (n : ℕ+) : 
  let S : ℕ+ → ℚ := fun k => (k * (k + 1)) / 2
  (S n) / (n^2 : ℚ) ≤ 9/16 := by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_over_square_n_l3105_310598


namespace NUMINAMATH_CALUDE_nabla_calculation_l3105_310573

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- Theorem statement
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l3105_310573


namespace NUMINAMATH_CALUDE_sin_neg_seven_pi_thirds_l3105_310521

theorem sin_neg_seven_pi_thirds : Real.sin (-7 * Real.pi / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_seven_pi_thirds_l3105_310521


namespace NUMINAMATH_CALUDE_six_digit_numbers_at_least_two_zeros_l3105_310587

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 900000

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zeros : ℕ := 531441

/-- The number of 6-digit numbers with exactly one zero -/
def six_digit_numbers_one_zero : ℕ := 295245

/-- Theorem: The number of 6-digit numbers with at least two zeros is 73,314 -/
theorem six_digit_numbers_at_least_two_zeros :
  total_six_digit_numbers - (six_digit_numbers_no_zeros + six_digit_numbers_one_zero) = 73314 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_at_least_two_zeros_l3105_310587


namespace NUMINAMATH_CALUDE_baseball_team_grouping_l3105_310597

/-- Given the number of new players, returning players, and groups, 
    calculate the number of players in each group -/
def players_per_group (new_players returning_players groups : ℕ) : ℕ :=
  (new_players + returning_players) / groups

/-- Theorem stating that with 48 new players, 6 returning players, and 9 groups,
    there are 6 players in each group -/
theorem baseball_team_grouping :
  players_per_group 48 6 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_grouping_l3105_310597


namespace NUMINAMATH_CALUDE_angle_between_vectors_l3105_310500

def vector1 : Fin 3 → ℝ := ![1, 3, -2]
def vector2 : Fin 3 → ℝ := ![4, -2, 1]

theorem angle_between_vectors :
  let dot_product := (vector1 0) * (vector2 0) + (vector1 1) * (vector2 1) + (vector1 2) * (vector2 2)
  let magnitude1 := Real.sqrt ((vector1 0)^2 + (vector1 1)^2 + (vector1 2)^2)
  let magnitude2 := Real.sqrt ((vector2 0)^2 + (vector2 1)^2 + (vector2 2)^2)
  dot_product / (magnitude1 * magnitude2) = -2 / (7 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l3105_310500


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l3105_310583

theorem absolute_value_not_positive (x : ℚ) : |4*x - 8| ≤ 0 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l3105_310583


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3105_310509

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^2*(b+c) + b^2*(c+a) + c^2*(a+b)) / (a^3 + b^3 + c^3 - 2*a*b*c)
  A ≤ 6 ∧ (A = 6 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3105_310509


namespace NUMINAMATH_CALUDE_power_fraction_equality_l3105_310590

theorem power_fraction_equality : (1 : ℚ) / ((-5^4)^2) * (-5)^9 = -5 := by sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l3105_310590


namespace NUMINAMATH_CALUDE_pears_transport_l3105_310524

/-- Prove that given 8 tons of apples and the amount of pears being 7 times the amount of apples,
    the total amount of pears transported is 56 tons. -/
theorem pears_transport (apple_tons : ℕ) (pear_multiplier : ℕ) : 
  apple_tons = 8 → pear_multiplier = 7 → apple_tons * pear_multiplier = 56 := by
  sorry

end NUMINAMATH_CALUDE_pears_transport_l3105_310524


namespace NUMINAMATH_CALUDE_magnitude_of_b_l3105_310558

def a : ℝ × ℝ := (2, 1)

theorem magnitude_of_b (b : ℝ × ℝ) 
  (h1 : a.fst * b.fst + a.snd * b.snd = 10)
  (h2 : (a.fst + 2 * b.fst)^2 + (a.snd + 2 * b.snd)^2 = 50) :
  Real.sqrt (b.fst^2 + b.snd^2) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_b_l3105_310558


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3105_310565

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) →  -- Perimeter condition
  (l * w ≤ 100) ∧         -- Area is at most 100
  (∃ l' w' : ℕ, 2 * l' + 2 * w' = 40 ∧ l' * w' = 100) -- Maximum area exists
  :=
sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3105_310565


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l3105_310599

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x > 0, f x > 0) ↔ (∃ x > 0, f x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l3105_310599


namespace NUMINAMATH_CALUDE_domain_of_g_l3105_310510

-- Define the function f with domain (-3, 6)
def f : {x : ℝ // -3 < x ∧ x < 6} → ℝ := sorry

-- Define the function g(x) = f(2x)
def g (x : ℝ) : ℝ := f ⟨2*x, sorry⟩

-- Theorem statement
theorem domain_of_g :
  ∀ x : ℝ, (∃ y : ℝ, g x = y) ↔ -3/2 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3105_310510


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3105_310592

/-- An arithmetic progression with the first three terms 2x - 2, 2x + 2, and 4x + 6 has x = 0 --/
theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), 
  let a₁ : ℝ := 2 * x - 2
  let a₂ : ℝ := 2 * x + 2
  let a₃ : ℝ := 4 * x + 6
  (a₂ - a₁ = a₃ - a₂) → x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l3105_310592


namespace NUMINAMATH_CALUDE_special_sequence_bijective_l3105_310559

/-- A sequence of integers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, ∃ k > n, a k > 0) ∧  -- Infinite positive values
  (∀ n : ℕ, ∃ k > n, a k < 0) ∧  -- Infinite negative values
  (∀ n : ℕ+, ∀ i j, i ≠ j → i ≤ n → j ≤ n → a i % n ≠ a j % n)  -- Distinct modulo n

/-- The theorem stating that every integer appears exactly once in the sequence -/
theorem special_sequence_bijective (a : ℕ → ℤ) (h : SpecialSequence a) :
  Function.Bijective a :=
sorry

end NUMINAMATH_CALUDE_special_sequence_bijective_l3105_310559


namespace NUMINAMATH_CALUDE_jordan_running_time_l3105_310517

theorem jordan_running_time (steve_time steve_distance jordan_distance_1 jordan_distance_2 : ℚ)
  (h1 : steve_time = 32)
  (h2 : steve_distance = 4)
  (h3 : jordan_distance_1 = 3)
  (h4 : jordan_distance_2 = 7)
  (h5 : jordan_distance_1 / (steve_time / 2) = steve_distance / steve_time) :
  jordan_distance_2 / (jordan_distance_1 / (steve_time / 2)) = 112 / 3 := by sorry

end NUMINAMATH_CALUDE_jordan_running_time_l3105_310517


namespace NUMINAMATH_CALUDE_angle_bisectors_may_not_form_triangle_l3105_310553

/-- Given a triangle with sides a = 2, b = 3, and c < 5, 
    prove that its angle bisectors may not satisfy the triangle inequality -/
theorem angle_bisectors_may_not_form_triangle :
  ∃ (c : ℝ), c < 5 ∧ 
  ∃ (ℓa ℓb ℓc : ℝ),
    (ℓa + ℓb ≤ ℓc ∨ ℓa + ℓc ≤ ℓb ∨ ℓb + ℓc ≤ ℓa) ∧
    ℓa = 3 / (1 + 2 / 7 * 3) ∧
    ℓb = 2 / (2 + 3 / 8 * 2) ∧
    ℓc = 0 := by
  sorry


end NUMINAMATH_CALUDE_angle_bisectors_may_not_form_triangle_l3105_310553


namespace NUMINAMATH_CALUDE_circle_pattern_proof_l3105_310579

theorem circle_pattern_proof : 
  ∀ n : ℕ, (n * (n + 1)) / 2 ≤ 120 ∧ ((n + 1) * (n + 2)) / 2 > 120 → n = 14 :=
by sorry

end NUMINAMATH_CALUDE_circle_pattern_proof_l3105_310579


namespace NUMINAMATH_CALUDE_alto_saxophone_ratio_l3105_310511

/-- The ratio of alto saxophone players to total saxophone players in a high school band -/
theorem alto_saxophone_ratio (total_students : ℕ) 
  (h1 : total_students = 600)
  (marching_band : ℕ) 
  (h2 : marching_band = total_students / 5)
  (brass_players : ℕ) 
  (h3 : brass_players = marching_band / 2)
  (saxophone_players : ℕ) 
  (h4 : saxophone_players = brass_players / 5)
  (alto_saxophone_players : ℕ) 
  (h5 : alto_saxophone_players = 4) :
  (alto_saxophone_players : ℚ) / saxophone_players = 1 / 3 := by
sorry


end NUMINAMATH_CALUDE_alto_saxophone_ratio_l3105_310511


namespace NUMINAMATH_CALUDE_max_r_value_exists_max_r_unique_max_r_l3105_310584

open Set Real

/-- The set T parameterized by r -/
def T (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 7)^2 ≤ r^2}

/-- The set S -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∀ θ : ℝ, cos (2 * θ) + p.1 * cos θ + p.2 ≥ 0}

/-- The main theorem stating the maximum value of r -/
theorem max_r_value (r : ℝ) (h_pos : r > 0) (h_subset : T r ⊆ S) : r ≤ 4 * sqrt 2 := by
  sorry

/-- The existence of the maximum r value -/
theorem exists_max_r : ∃ r : ℝ, r > 0 ∧ T r ⊆ S ∧ ∀ s : ℝ, s > 0 ∧ T s ⊆ S → s ≤ r := by
  sorry

/-- The uniqueness of the maximum r value -/
theorem unique_max_r (r s : ℝ) (hr : r > 0) (hs : s > 0)
    (h_max_r : T r ⊆ S ∧ ∀ t : ℝ, t > 0 ∧ T t ⊆ S → t ≤ r)
    (h_max_s : T s ⊆ S ∧ ∀ t : ℝ, t > 0 ∧ T t ⊆ S → t ≤ s) : r = s := by
  sorry

end NUMINAMATH_CALUDE_max_r_value_exists_max_r_unique_max_r_l3105_310584


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l3105_310555

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 + 3 * Complex.I
  let z₂ : ℂ := -2 + Real.sqrt 2 * Complex.I
  Complex.abs (z₁ - z₂) = Real.sqrt (36 - 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l3105_310555


namespace NUMINAMATH_CALUDE_smith_family_mean_age_l3105_310546

def smith_family_ages : List ℕ := [5, 5, 5, 12, 13, 16]

theorem smith_family_mean_age :
  (smith_family_ages.sum : ℚ) / smith_family_ages.length = 9.33 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_mean_age_l3105_310546


namespace NUMINAMATH_CALUDE_gcd_85_357_is_1_l3105_310563

theorem gcd_85_357_is_1 : Nat.gcd 85 357 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_85_357_is_1_l3105_310563


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3105_310501

theorem sum_with_radical_conjugate : 
  let x : ℝ := 15 - Real.sqrt 500
  let y : ℝ := 15 + Real.sqrt 500
  x + y = 30 := by sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3105_310501


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_16_l3105_310557

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference
  first_term_eq : a 1 = a 1  -- Placeholder for the first term
  diff_eq : ∀ n, a (n + 1) - a n = d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem arithmetic_sequence_sum_16 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 12 = -8) 
  (h2 : sum_n seq 9 = -9) : 
  sum_n seq 16 = -72 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_16_l3105_310557


namespace NUMINAMATH_CALUDE_negative_double_negation_l3105_310514

theorem negative_double_negation (x : ℝ) (h : -x = 2) : -(-(-x)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_double_negation_l3105_310514


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_l3105_310569

-- Define set A
def A : Set ℕ := {0, 1, 2}

-- Define set B
def B : Set ℕ := {x : ℕ | (x + 1) / (x - 2 : ℝ) ≤ 0}

-- Theorem statement
theorem A_intersect_B_eq_zero_one : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_zero_one_l3105_310569


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3105_310575

theorem arithmetic_mean_of_special_set (n : ℕ) (hn : n > 2) :
  let set := List.replicate (n - 2) 1 ++ List.replicate 2 (1 - 1 / n)
  (List.sum set) / n = 1 - 2 / n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3105_310575


namespace NUMINAMATH_CALUDE_total_payment_equals_car_cost_l3105_310570

/-- Represents the car purchase scenario -/
structure CarPurchase where
  carCost : ℕ             -- Cost of the car in euros
  initialPayment : ℕ      -- Initial payment in euros
  installments : ℕ        -- Number of installments
  installmentAmount : ℕ   -- Amount per installment in euros

/-- Theorem stating that the total amount paid equals the car's cost -/
theorem total_payment_equals_car_cost (purchase : CarPurchase) 
  (h1 : purchase.carCost = 18000)
  (h2 : purchase.initialPayment = 3000)
  (h3 : purchase.installments = 6)
  (h4 : purchase.installmentAmount = 2500) :
  purchase.initialPayment + purchase.installments * purchase.installmentAmount = purchase.carCost :=
by sorry

end NUMINAMATH_CALUDE_total_payment_equals_car_cost_l3105_310570


namespace NUMINAMATH_CALUDE_smartphone_price_reduction_l3105_310589

theorem smartphone_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 7500)
  (h2 : final_price = 4800)
  (h3 : ∃ (x : ℝ), final_price = original_price * (1 - x)^2 ∧ 0 < x ∧ x < 1) :
  ∃ (x : ℝ), final_price = original_price * (1 - x)^2 ∧ x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_smartphone_price_reduction_l3105_310589


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3105_310551

theorem geometric_sequence_first_term (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ 16 = c * r ∧ 32 = 16 * r) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3105_310551


namespace NUMINAMATH_CALUDE_carriage_sharing_problem_l3105_310548

theorem carriage_sharing_problem (x : ℝ) : 
  (x > 0) →                            -- Ensure positive number of people
  (x / 3 + 2 = (x - 9) / 2) →           -- The equation to be proved
  (∃ n : ℕ, x = n) →                    -- Ensure x is a natural number
  (x / 3 + 2 : ℝ) = (x - 9) / 2 :=      -- The equation represents the problem
by
  sorry

end NUMINAMATH_CALUDE_carriage_sharing_problem_l3105_310548


namespace NUMINAMATH_CALUDE_salary_change_percentage_l3105_310533

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l3105_310533


namespace NUMINAMATH_CALUDE_bakers_cakes_l3105_310577

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes sold_cakes final_cakes : ℕ) 
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111) :
  final_cakes - (initial_cakes - sold_cakes) = 76 := by
  sorry

end NUMINAMATH_CALUDE_bakers_cakes_l3105_310577


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3105_310519

/-- The equation of a line tangent to a unit circle that intersects a specific ellipse -/
theorem tangent_line_equation (k b : ℝ) (h_b_pos : b > 0) 
  (h_tangent : b^2 = k^2 + 1)
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + b ∧ 
    y₂ = k * x₂ + b ∧ 
    x₁^2 / 2 + y₁^2 = 1 ∧ 
    x₂^2 / 2 + y₂^2 = 1)
  (h_dot_product : 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      y₁ = k * x₁ + b → 
      y₂ = k * x₂ + b → 
      x₁^2 / 2 + y₁^2 = 1 → 
      x₂^2 / 2 + y₂^2 = 1 → 
      x₁ * x₂ + y₁ * y₂ = 2/3) :
  (k = 1 ∧ b = Real.sqrt 2) ∨ (k = -1 ∧ b = Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3105_310519


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3105_310505

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 7) :
  ∃ (k : ℕ+), k = Nat.gcd (8 * m) (6 * n) ∧ ∀ (l : ℕ+), l = Nat.gcd (8 * m) (6 * n) → k ≤ l :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3105_310505


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l3105_310537

/-- Represents the number of workers in the workshop -/
def total_workers : ℕ := 26

/-- Represents the number of screws a worker can produce per day -/
def screws_per_worker : ℕ := 800

/-- Represents the number of nuts a worker can produce per day -/
def nuts_per_worker : ℕ := 1000

/-- Represents the number of nuts needed to match one screw -/
def nuts_per_screw : ℕ := 2

/-- Theorem stating the correct system of equations for matching screws and nuts -/
theorem correct_system_of_equations (x y : ℕ) :
  (x + y = total_workers) ∧
  (nuts_per_worker * y = nuts_per_screw * screws_per_worker * x) →
  (x + y = total_workers) ∧
  (1000 * y = 2 * 800 * x) :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l3105_310537


namespace NUMINAMATH_CALUDE_vector_subtraction_l3105_310527

/-- Given two planar vectors a and b, prove that a - 2b equals the expected result. -/
theorem vector_subtraction (a b : ℝ × ℝ) (h1 : a = (5, 3)) (h2 : b = (1, -2)) :
  a - 2 • b = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3105_310527


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l3105_310547

/-- Given points A, B, and C on the inverse proportion function y = 3/x,
    prove that y₂ < y₁ < y₃ -/
theorem inverse_proportion_ordering (y₁ y₂ y₃ : ℝ) :
  y₁ = 3 / (-5) →
  y₂ = 3 / (-3) →
  y₃ = 3 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l3105_310547


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3105_310522

/-- The eccentricity of an ellipse with equation x^2 + ky^2 = 3k (k > 0) is √3/2,
    given that one of its foci coincides with the focus of the parabola y^2 = 12x. -/
theorem ellipse_eccentricity (k : ℝ) (h_k : k > 0) : 
  let ellipse := {(x, y) : ℝ × ℝ | x^2 + k*y^2 = 3*k}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 12*x}
  let parabola_focus : ℝ × ℝ := (3, 0)
  ∃ (ellipse_focus : ℝ × ℝ), 
    ellipse_focus ∈ ellipse ∧ 
    ellipse_focus = parabola_focus →
    let a := Real.sqrt (3*k)
    let b := Real.sqrt 3
    let c := 3
    let eccentricity := c / a
    eccentricity = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3105_310522


namespace NUMINAMATH_CALUDE_division_problem_l3105_310518

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 100 →
  divisor = 11 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3105_310518


namespace NUMINAMATH_CALUDE_boxwoods_shaped_into_spheres_l3105_310503

/-- Calculates the number of boxwoods shaped into spheres given the total number of boxwoods,
    costs for trimming and shaping, and the total charge. -/
theorem boxwoods_shaped_into_spheres
  (total_boxwoods : ℕ)
  (trim_cost : ℚ)
  (shape_cost : ℚ)
  (total_charge : ℚ)
  (h1 : total_boxwoods = 30)
  (h2 : trim_cost = 5)
  (h3 : shape_cost = 15)
  (h4 : total_charge = 210) :
  (total_charge - (total_boxwoods * trim_cost)) / shape_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_boxwoods_shaped_into_spheres_l3105_310503


namespace NUMINAMATH_CALUDE_pizza_combinations_l3105_310520

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  1 + n + n.choose 2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3105_310520


namespace NUMINAMATH_CALUDE_olympic_rings_area_l3105_310549

/-- Olympic Ring -/
structure OlympicRing where
  outer_diameter : ℝ
  inner_diameter : ℝ

/-- Olympic Emblem -/
structure OlympicEmblem where
  rings : Fin 5 → OlympicRing
  hypotenuse : ℝ

/-- The area covered by the Olympic rings -/
def area_covered (e : OlympicEmblem) : ℝ :=
  sorry

/-- The theorem statement -/
theorem olympic_rings_area (e : OlympicEmblem)
  (h1 : ∀ i : Fin 5, (e.rings i).outer_diameter = 22)
  (h2 : ∀ i : Fin 5, (e.rings i).inner_diameter = 18)
  (h3 : e.hypotenuse = 24) :
  ∃ ε > 0, |area_covered e - 592| < ε :=
sorry

end NUMINAMATH_CALUDE_olympic_rings_area_l3105_310549


namespace NUMINAMATH_CALUDE_pyramid_volume_l3105_310560

/-- The volume of a pyramid with a square base of side length 2 and height 2 is 8/3 cubic units -/
theorem pyramid_volume (base_side_length height : ℝ) (h1 : base_side_length = 2) (h2 : height = 2) :
  (1 / 3 : ℝ) * base_side_length^2 * height = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_volume_l3105_310560


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l3105_310552

/-- Represents the daily consumption of cookies and brownies --/
structure DailyConsumption where
  cookies : Nat
  brownies : Nat

/-- Calculates the remaining items after a week of consumption --/
def remainingItems (initial : Nat) (daily : List Nat) : Nat :=
  max (initial - daily.sum) 0

theorem cookie_brownie_difference :
  let initialCookies : Nat := 60
  let initialBrownies : Nat := 10
  let weeklyConsumption : List DailyConsumption := [
    ⟨2, 1⟩, ⟨4, 2⟩, ⟨3, 1⟩, ⟨5, 1⟩, ⟨4, 3⟩, ⟨3, 2⟩, ⟨2, 1⟩
  ]
  let cookiesLeft := remainingItems initialCookies (weeklyConsumption.map DailyConsumption.cookies)
  let browniesLeft := remainingItems initialBrownies (weeklyConsumption.map DailyConsumption.brownies)
  cookiesLeft - browniesLeft = 37 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l3105_310552


namespace NUMINAMATH_CALUDE_total_molecular_weight_l3105_310545

-- Define atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Fe : ℝ := 55.845
def atomic_weight_S : ℝ := 32.07
def atomic_weight_Mn : ℝ := 54.938

-- Define molecular weights
def molecular_weight_K2Cr2O7 : ℝ :=
  2 * atomic_weight_K + 2 * atomic_weight_Cr + 7 * atomic_weight_O

def molecular_weight_Fe2SO43 : ℝ :=
  2 * atomic_weight_Fe + 3 * (atomic_weight_S + 4 * atomic_weight_O)

def molecular_weight_KMnO4 : ℝ :=
  atomic_weight_K + atomic_weight_Mn + 4 * atomic_weight_O

-- Define the theorem
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 +
  3 * molecular_weight_Fe2SO43 +
  5 * molecular_weight_KMnO4 = 3166.658 :=
by sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l3105_310545


namespace NUMINAMATH_CALUDE_chocolate_distribution_l3105_310588

theorem chocolate_distribution (total_pieces : ℕ) (num_boxes : ℕ) (pieces_per_box : ℕ) :
  total_pieces = 3000 →
  num_boxes = 6 →
  total_pieces = num_boxes * pieces_per_box →
  pieces_per_box = 500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_distribution_l3105_310588


namespace NUMINAMATH_CALUDE_trapezoid_sides_for_given_circle_l3105_310556

/-- Represents a trapezoid formed by tangents to a circle -/
structure CircleTrapezoid where
  radius : ℝ
  chord_length : ℝ

/-- Calculates the sides of the trapezoid -/
def trapezoid_sides (t : CircleTrapezoid) : (ℝ × ℝ × ℝ × ℝ) :=
  sorry

/-- Theorem stating the correct sides of the trapezoid for the given circle -/
theorem trapezoid_sides_for_given_circle :
  let t : CircleTrapezoid := { radius := 5, chord_length := 8 }
  trapezoid_sides t = (12.5, 5, 12.5, 20) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_sides_for_given_circle_l3105_310556


namespace NUMINAMATH_CALUDE_candy_solution_l3105_310515

def candy_problem (f b j : ℕ) : Prop :=
  f = 12 ∧ b = f + 6 ∧ j = 10 * (f + b)

theorem candy_solution : 
  ∀ f b j : ℕ, candy_problem f b j → (40 * j^2) / 100 = 36000 := by
  sorry

end NUMINAMATH_CALUDE_candy_solution_l3105_310515
