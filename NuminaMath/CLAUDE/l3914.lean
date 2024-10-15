import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l3914_391408

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≤ (a*b/c + b*c/a + c*a/b) / 3 ∧
  ((a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
   Real.sqrt ((a^2 + b^2 + c^2) / 3) = (a*b/c + b*c/a + c*a/b) / 3) ↔ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3914_391408


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l3914_391436

/-- The number of visitors to Buckingham Palace on different days --/
structure PalaceVisitors where
  dayOfVisit : ℕ
  previousDay : ℕ
  twoDaysPrior : ℕ

/-- Calculates the difference in visitors between the day of visit and the sum of the previous two days --/
def visitorDifference (v : PalaceVisitors) : ℕ :=
  v.dayOfVisit - (v.previousDay + v.twoDaysPrior)

/-- Theorem stating the difference in visitors for the given data --/
theorem buckingham_palace_visitors :
  ∃ (v : PalaceVisitors),
    v.dayOfVisit = 8333 ∧
    v.previousDay = 3500 ∧
    v.twoDaysPrior = 2500 ∧
    visitorDifference v = 2333 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l3914_391436


namespace NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l3914_391443

theorem cube_sum_minus_product_eq_2003 :
  ∀ x y z : ℤ, x^3 + y^3 + z^3 - 3*x*y*z = 2003 ↔ 
  ((x = 668 ∧ y = 668 ∧ z = 667) ∨ 
   (x = 668 ∧ y = 667 ∧ z = 668) ∨ 
   (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_minus_product_eq_2003_l3914_391443


namespace NUMINAMATH_CALUDE_largest_square_area_l3914_391480

-- Define the triangle and its properties
structure RightTriangle where
  XY : ℝ
  YZ : ℝ
  XZ : ℝ
  right_angle : XZ^2 = XY^2 + YZ^2
  hypotenuse_relation : XZ^2 = 2 * XY^2

-- Define the theorem
theorem largest_square_area
  (triangle : RightTriangle)
  (total_area : ℝ)
  (h_total_area : XY^2 + YZ^2 + XZ^2 = total_area)
  (h_total_area_value : total_area = 450) :
  XZ^2 = 225 := by
  sorry

#check largest_square_area

end NUMINAMATH_CALUDE_largest_square_area_l3914_391480


namespace NUMINAMATH_CALUDE_cube_root_equal_self_l3914_391453

theorem cube_root_equal_self (a : ℝ) : a^(1/3) = a → a = 1 ∨ a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equal_self_l3914_391453


namespace NUMINAMATH_CALUDE_length_BC_l3914_391428

-- Define the centers and radii of the circles
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
def radius_A : ℝ := 7
def radius_B : ℝ := 4

-- Define the distance between centers A and B
def AB : ℝ := radius_A + radius_B

-- Define point C
def C : ℝ × ℝ := sorry

-- Define the distance AC
def AC : ℝ := AB + 2

-- Theorem to prove
theorem length_BC : ∃ (BC : ℝ), BC = 52 / 7 := by
  sorry

end NUMINAMATH_CALUDE_length_BC_l3914_391428


namespace NUMINAMATH_CALUDE_polyhedron_sum_l3914_391482

/-- A convex polyhedron with triangular and hexagonal faces -/
structure ConvexPolyhedron where
  faces : ℕ
  triangles : ℕ
  hexagons : ℕ
  vertices : ℕ
  edges : ℕ
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex
  faces_sum : faces = triangles + hexagons
  faces_20 : faces = 20
  edges_formula : edges = (3 * triangles + 6 * hexagons) / 2
  euler_formula : vertices - edges + faces = 2

/-- The theorem to be proved -/
theorem polyhedron_sum (p : ConvexPolyhedron) : 100 * p.H + 10 * p.T + p.vertices = 227 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l3914_391482


namespace NUMINAMATH_CALUDE_guarantee_target_color_count_l3914_391468

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  purple : Nat

/-- The initial ball counts in the box -/
def initialBalls : BallCounts :=
  { red := 30, green := 24, yellow := 16, blue := 14, white := 12, purple := 4 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : Nat := 12

/-- The number of balls we claim will guarantee the target count -/
def claimedDrawCount : Nat := 60

/-- Theorem stating that drawing the claimed number of balls guarantees
    at least the target count of a single color -/
theorem guarantee_target_color_count :
  ∀ (drawn : Nat),
    drawn ≥ claimedDrawCount →
    ∃ (color : Fin 6),
      (match color with
       | 0 => initialBalls.red
       | 1 => initialBalls.green
       | 2 => initialBalls.yellow
       | 3 => initialBalls.blue
       | 4 => initialBalls.white
       | 5 => initialBalls.purple) -
      (claimedDrawCount - drawn) ≥ targetCount :=
by sorry

end NUMINAMATH_CALUDE_guarantee_target_color_count_l3914_391468


namespace NUMINAMATH_CALUDE_farm_has_two_fields_l3914_391454

/-- Represents a corn field -/
structure CornField where
  rows : ℕ
  cobs_per_row : ℕ

/-- Calculates the total number of corn cobs in a field -/
def total_cobs (field : CornField) : ℕ :=
  field.rows * field.cobs_per_row

/-- Represents the farm's corn production -/
structure FarmProduction where
  field1 : CornField
  field2 : CornField
  total_cobs : ℕ

/-- Theorem: The farm is growing corn in 2 fields -/
theorem farm_has_two_fields (farm : FarmProduction) : 
  farm.field1.rows = 13 ∧ 
  farm.field2.rows = 16 ∧ 
  farm.field1.cobs_per_row = 4 ∧ 
  farm.field2.cobs_per_row = 4 ∧ 
  farm.total_cobs = 116 → 
  2 = (if total_cobs farm.field1 + total_cobs farm.field2 = farm.total_cobs then 2 else 1) :=
by sorry


end NUMINAMATH_CALUDE_farm_has_two_fields_l3914_391454


namespace NUMINAMATH_CALUDE_no_prime_divisible_by_35_l3914_391497

/-- A number is prime if it has exactly two distinct positive divisors -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_35 :
  ¬∃ p : ℕ, isPrime p ∧ 35 ∣ p :=
by sorry

end NUMINAMATH_CALUDE_no_prime_divisible_by_35_l3914_391497


namespace NUMINAMATH_CALUDE_playground_slide_total_l3914_391460

theorem playground_slide_total (boys_first_10min : ℕ) (boys_next_5min : ℕ) (boys_last_20min : ℕ)
  (h1 : boys_first_10min = 22)
  (h2 : boys_next_5min = 13)
  (h3 : boys_last_20min = 35) :
  boys_first_10min + boys_next_5min + boys_last_20min = 70 :=
by sorry

end NUMINAMATH_CALUDE_playground_slide_total_l3914_391460


namespace NUMINAMATH_CALUDE_domain_f_l3914_391434

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_f (h : ∀ x, f (x + 1) ∈ domain_f_plus_one ↔ x ∈ Set.Icc (-2) 3) :
  ∀ x, f x ∈ Set.Icc (-3) 2 ↔ x ∈ Set.Icc (-3) 2 :=
sorry

end NUMINAMATH_CALUDE_domain_f_l3914_391434


namespace NUMINAMATH_CALUDE_modulus_of_z_l3914_391432

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3914_391432


namespace NUMINAMATH_CALUDE_zero_in_interval_l3914_391403

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 4, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3914_391403


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3914_391493

-- Define the sets A and B
def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | |x| < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3914_391493


namespace NUMINAMATH_CALUDE_other_triangle_rectangle_area_ratio_l3914_391486

/-- Represents a right triangle with a point on its hypotenuse -/
structure RightTriangleWithPoint where
  /-- Length of the side of the rectangle along the hypotenuse -/
  side_along_hypotenuse : ℝ
  /-- Length of the side of the rectangle perpendicular to the hypotenuse -/
  side_perpendicular : ℝ
  /-- Ratio of the area of one small right triangle to the area of the rectangle -/
  area_ratio : ℝ
  /-- Condition: The side along the hypotenuse has length 1 -/
  hypotenuse_side_length : side_along_hypotenuse = 1
  /-- Condition: The area of one small right triangle is n times the area of the rectangle -/
  area_ratio_condition : area_ratio > 0

/-- Theorem: The ratio of the area of the other small right triangle to the area of the rectangle -/
theorem other_triangle_rectangle_area_ratio 
  (t : RightTriangleWithPoint) : 
  ∃ (ratio : ℝ), ratio = t.side_perpendicular / t.area_ratio := by
  sorry

end NUMINAMATH_CALUDE_other_triangle_rectangle_area_ratio_l3914_391486


namespace NUMINAMATH_CALUDE_triangle_area_and_fixed_point_l3914_391470

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 8

-- Define the family of lines
def family_of_lines (m x y : ℝ) : Prop := m * x + y + m = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-1, 0)

theorem triangle_area_and_fixed_point :
  (∀ x y, line_equation x y → 
    (x = 0 ∨ y = 0) → triangle_area = 8) ∧
  (∀ m x y, family_of_lines m x y → 
    (x, y) = fixed_point) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_and_fixed_point_l3914_391470


namespace NUMINAMATH_CALUDE_binomial_1500_1_l3914_391437

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1500_1_l3914_391437


namespace NUMINAMATH_CALUDE_ratio_problem_l3914_391481

theorem ratio_problem (x y : ℤ) : 
  (y = 4 * x) →  -- The two integers are in the ratio of 1 to 4
  (x + 12 = y) →  -- Adding 12 to the smaller number makes the ratio 1 to 1
  y = 16 :=  -- The larger integer is 16
by sorry

end NUMINAMATH_CALUDE_ratio_problem_l3914_391481


namespace NUMINAMATH_CALUDE_fence_building_time_l3914_391455

/-- The time in minutes to build one fence -/
def time_per_fence : ℕ := 30

/-- The number of fences initially built -/
def initial_fences : ℕ := 10

/-- The total number of fences after additional work -/
def total_fences : ℕ := 26

/-- The additional work time in hours -/
def additional_work_time : ℕ := 8

/-- Theorem stating that the time per fence is 30 minutes -/
theorem fence_building_time :
  time_per_fence = 30 ∧
  initial_fences = 10 ∧
  total_fences = 26 ∧
  additional_work_time = 8 ∧
  (total_fences - initial_fences) * time_per_fence = additional_work_time * 60 :=
by sorry

end NUMINAMATH_CALUDE_fence_building_time_l3914_391455


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3914_391450

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : Real.log (a + b) = 0) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Real.log (x + y) = 0 ∧ 1 / x + 1 / y = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3914_391450


namespace NUMINAMATH_CALUDE_total_earnings_is_4350_l3914_391483

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  invest_ratio_A : ℕ
  invest_ratio_B : ℕ
  invest_ratio_C : ℕ
  return_ratio_A : ℕ
  return_ratio_B : ℕ
  return_ratio_C : ℕ

/-- Calculates the total earnings given investment data and the earnings difference between B and A -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff_B_A : ℕ) : ℕ :=
  let earnings_A := data.invest_ratio_A * data.return_ratio_A
  let earnings_B := data.invest_ratio_B * data.return_ratio_B
  let earnings_C := data.invest_ratio_C * data.return_ratio_C
  let total_ratio := earnings_A + earnings_B + earnings_C
  (total_ratio * earnings_diff_B_A) / (earnings_B - earnings_A)

/-- Theorem stating that given the specific investment ratios and conditions, the total earnings is 4350 -/
theorem total_earnings_is_4350 : 
  let data : InvestmentData := {
    invest_ratio_A := 3,
    invest_ratio_B := 4,
    invest_ratio_C := 5,
    return_ratio_A := 6,
    return_ratio_B := 5,
    return_ratio_C := 4
  }
  calculate_total_earnings data 150 = 4350 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_4350_l3914_391483


namespace NUMINAMATH_CALUDE_inequality_region_l3914_391446

theorem inequality_region (x y : ℝ) : 
  ((x*y + 1) / (x + y))^2 < 1 ↔ 
  ((-1 < x ∧ x < 1 ∧ (y < -1 ∨ y > 1)) ∨ ((x < -1 ∨ x > 1) ∧ -1 < y ∧ y < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_l3914_391446


namespace NUMINAMATH_CALUDE_equation_solution_l3914_391473

theorem equation_solution :
  ∃! x : ℚ, (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 :=
by
  use -11/6
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3914_391473


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3914_391457

theorem partial_fraction_decomposition :
  let f (x : ℝ) := (2*x + 7) / (x^2 - 2*x - 63)
  let g (x : ℝ) := 25 / (16 * (x - 9)) + 7 / (16 * (x + 7))
  ∀ x : ℝ, x ≠ 9 ∧ x ≠ -7 → f x = g x :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3914_391457


namespace NUMINAMATH_CALUDE_fitness_center_membership_ratio_l3914_391425

theorem fitness_center_membership_ratio :
  ∀ (f m c : ℕ), 
  (f > 0) → (m > 0) → (c > 0) →
  (35 * f + 30 * m + 10 * c : ℝ) / (f + m + c : ℝ) = 25 →
  ∃ (k : ℕ), f = 3 * k ∧ m = 6 * k ∧ c = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_membership_ratio_l3914_391425


namespace NUMINAMATH_CALUDE_inequality_not_preserved_after_subtraction_of_squares_l3914_391464

theorem inequality_not_preserved_after_subtraction_of_squares : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ (a - a^2) ≤ (b - b^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_preserved_after_subtraction_of_squares_l3914_391464


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3914_391484

/-- A 2x2 matrix is a projection matrix if and only if P² = P -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P ^ 2 = P

/-- The specific 2x2 matrix we're working with -/
def P (b d : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![b, 12/25; d, 13/25]

/-- The theorem stating the values of b and d for the projection matrix -/
theorem projection_matrix_values :
  ∀ b d : ℚ, is_projection_matrix (P b d) → b = 37/50 ∧ d = 19/50 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3914_391484


namespace NUMINAMATH_CALUDE_remaining_length_is_23_l3914_391400

/-- Represents a figure with perpendicular sides -/
structure PerpendicularFigure where
  left_perimeter : ℝ
  right_perimeter : ℝ
  top_side : ℝ
  bottom_left : ℝ
  bottom_right : ℝ

/-- Calculates the total length of remaining segments after removal -/
def remaining_length (fig : PerpendicularFigure) : ℝ :=
  fig.left_perimeter + fig.right_perimeter + fig.bottom_left + fig.bottom_right

/-- Theorem stating the total length of remaining segments is 23 units -/
theorem remaining_length_is_23 (fig : PerpendicularFigure)
  (h1 : fig.left_perimeter = 10)
  (h2 : fig.right_perimeter = 7)
  (h3 : fig.top_side = 3)
  (h4 : fig.bottom_left = 2)
  (h5 : fig.bottom_right = 1) :
  remaining_length fig = 23 := by
  sorry

#eval remaining_length { left_perimeter := 10, right_perimeter := 7, top_side := 3, bottom_left := 2, bottom_right := 1 }

end NUMINAMATH_CALUDE_remaining_length_is_23_l3914_391400


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l3914_391435

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ice cream flavors that can be created -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l3914_391435


namespace NUMINAMATH_CALUDE_function_extrema_condition_l3914_391495

def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ f a max ∧ f a min ≤ f a x) →
  (a < 1/3 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_condition_l3914_391495


namespace NUMINAMATH_CALUDE_pool_filling_rate_l3914_391475

/-- Given a pool filled by four hoses, this theorem proves the rate of two unknown hoses. -/
theorem pool_filling_rate 
  (pool_volume : ℝ) 
  (fill_time : ℝ) 
  (known_hose_rate : ℝ) 
  (h_volume : pool_volume = 15000)
  (h_time : fill_time = 25 * 60)  -- Convert hours to minutes
  (h_known_rate : known_hose_rate = 2)
  : ∃ (unknown_hose_rate : ℝ), 
    2 * known_hose_rate + 2 * unknown_hose_rate = pool_volume / fill_time ∧ 
    unknown_hose_rate = 3 :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_rate_l3914_391475


namespace NUMINAMATH_CALUDE_shift_down_three_units_l3914_391459

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := 2 * x

theorem shift_down_three_units (x : ℝ) : f x - 3 = g x := by
  sorry

end NUMINAMATH_CALUDE_shift_down_three_units_l3914_391459


namespace NUMINAMATH_CALUDE_sector_angle_l3914_391463

/-- Given a sector with area 1 and perimeter 4, its central angle in radians is 2 -/
theorem sector_angle (r : ℝ) (α : ℝ) 
  (h_area : (1/2) * α * r^2 = 1) 
  (h_perim : 2*r + α*r = 4) : 
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_angle_l3914_391463


namespace NUMINAMATH_CALUDE_smallest_four_digit_with_product_512_l3914_391417

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem smallest_four_digit_with_product_512 :
  ∀ n : ℕ, is_four_digit n → digit_product n = 512 → n ≥ 1888 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_with_product_512_l3914_391417


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3914_391466

-- Define the number of beads
def n : ℕ := 8

-- Define the function to calculate the number of distinct arrangements
def bracelet_arrangements (m : ℕ) : ℕ :=
  (Nat.factorial m) / (m * 2)

-- Theorem statement
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3914_391466


namespace NUMINAMATH_CALUDE_typing_speed_ratio_l3914_391488

/-- Represents the typing speeds of Tim and Tom -/
structure TypingSpeed where
  tim : ℝ
  tom : ℝ

/-- The total pages typed by Tim and Tom in one hour -/
def totalPages (speed : TypingSpeed) : ℝ := speed.tim + speed.tom

/-- The total pages typed when Tom increases his speed by 25% -/
def increasedTotalPages (speed : TypingSpeed) : ℝ := speed.tim + 1.25 * speed.tom

theorem typing_speed_ratio 
  (speed : TypingSpeed) 
  (h1 : totalPages speed = 12)
  (h2 : increasedTotalPages speed = 14) :
  speed.tom / speed.tim = 2 := by
  sorry

#check typing_speed_ratio

end NUMINAMATH_CALUDE_typing_speed_ratio_l3914_391488


namespace NUMINAMATH_CALUDE_harry_joe_fish_ratio_l3914_391447

/-- Proves that Harry has 4 times as many fish as Joe given the conditions -/
theorem harry_joe_fish_ratio :
  ∀ (harry joe sam : ℕ),
  joe = 8 * sam →
  sam = 7 →
  harry = 224 →
  harry = 4 * joe :=
by
  sorry

end NUMINAMATH_CALUDE_harry_joe_fish_ratio_l3914_391447


namespace NUMINAMATH_CALUDE_certain_number_existence_l3914_391402

theorem certain_number_existence : ∃ N : ℕ, 
  N % 127 = 10 ∧ 2045 % 127 = 13 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_existence_l3914_391402


namespace NUMINAMATH_CALUDE_day_before_yesterday_is_sunday_l3914_391477

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to get the previous day
def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

theorem day_before_yesterday_is_sunday 
  (h : nextDay (nextDay DayOfWeek.Sunday) = DayOfWeek.Monday) : 
  prevDay (prevDay DayOfWeek.Sunday) = DayOfWeek.Sunday := by
  sorry


end NUMINAMATH_CALUDE_day_before_yesterday_is_sunday_l3914_391477


namespace NUMINAMATH_CALUDE_total_frogs_is_18_l3914_391485

/-- The number of frogs inside the pond -/
def frogs_inside : ℕ := 12

/-- The number of frogs outside the pond -/
def frogs_outside : ℕ := 6

/-- The total number of frogs -/
def total_frogs : ℕ := frogs_inside + frogs_outside

/-- Theorem stating that the total number of frogs is 18 -/
theorem total_frogs_is_18 : total_frogs = 18 := by sorry

end NUMINAMATH_CALUDE_total_frogs_is_18_l3914_391485


namespace NUMINAMATH_CALUDE_four_distinct_roots_iff_q_16_l3914_391441

/-- The function f(x) = x^2 + 8x + q -/
def f (q : ℝ) (x : ℝ) : ℝ := x^2 + 8*x + q

/-- The composition of f with itself -/
def f_comp (q : ℝ) (x : ℝ) : ℝ := f q (f q x)

/-- The number of distinct real roots of f(f(x)) -/
noncomputable def num_distinct_roots (q : ℝ) : ℕ := sorry

theorem four_distinct_roots_iff_q_16 :
  ∀ q : ℝ, num_distinct_roots q = 4 ↔ q = 16 :=
sorry

end NUMINAMATH_CALUDE_four_distinct_roots_iff_q_16_l3914_391441


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3914_391424

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | a * x^2 - (a + 2) * x + 2 < 0}
  (a = 0 → S = {x : ℝ | x > 1}) ∧
  (0 < a ∧ a < 2 → S = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
  (a = 2 → S = ∅) ∧
  (a > 2 → S = {x : ℝ | 2/a < x ∧ x < 1}) ∧
  (a < 0 → S = {x : ℝ | x < 2/a ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3914_391424


namespace NUMINAMATH_CALUDE_marbles_left_l3914_391409

theorem marbles_left (initial_red : ℕ) (initial_blue : ℕ) (red_taken : ℕ) (blue_taken : ℕ) : 
  initial_red = 20 →
  initial_blue = 30 →
  red_taken = 3 →
  blue_taken = 4 * red_taken →
  initial_red - red_taken + initial_blue - blue_taken = 35 := by
sorry

end NUMINAMATH_CALUDE_marbles_left_l3914_391409


namespace NUMINAMATH_CALUDE_vowel_word_count_l3914_391462

/-- The number of vowels available (including Y) -/
def num_vowels : ℕ := 6

/-- The number of times each vowel appears, except A -/
def vowel_count : ℕ := 5

/-- The number of times A appears -/
def a_count : ℕ := 3

/-- The length of each word -/
def word_length : ℕ := 5

/-- The number of five-letter words that can be formed using vowels A, E, I, O, U, Y,
    where each vowel appears 5 times except A which appears 3 times -/
def num_words : ℕ := 7750

theorem vowel_word_count : 
  (vowel_count ^ word_length) + 
  (word_length.choose 1 * vowel_count ^ (word_length - 1)) +
  (word_length.choose 2 * vowel_count ^ (word_length - 2)) +
  (word_length.choose 3 * vowel_count ^ (word_length - 3)) = num_words :=
sorry

end NUMINAMATH_CALUDE_vowel_word_count_l3914_391462


namespace NUMINAMATH_CALUDE_coin_order_l3914_391449

/-- Represents the relative position of coins -/
inductive Position
| Above
| Below
| Same
| Unknown

/-- Represents a coin -/
inductive Coin
| F
| A
| B
| C
| D
| E

/-- Defines the relative position between two coins -/
def relative_position (c1 c2 : Coin) : Position := sorry

/-- Defines whether a coin is directly above another -/
def is_directly_above (c1 c2 : Coin) : Prop := 
  relative_position c1 c2 = Position.Above ∧ 
  ∀ c, c ≠ c1 ∧ c ≠ c2 → relative_position c1 c = Position.Above ∨ relative_position c c2 = Position.Above

/-- The main theorem to prove -/
theorem coin_order :
  (∀ c, c ≠ Coin.F → relative_position Coin.F c = Position.Above) ∧
  (is_directly_above Coin.A Coin.B) ∧
  (is_directly_above Coin.A Coin.C) ∧
  (relative_position Coin.A Coin.D = Position.Unknown) ∧
  (relative_position Coin.A Coin.E = Position.Unknown) ∧
  (is_directly_above Coin.D Coin.E) ∧
  (is_directly_above Coin.E Coin.B) ∧
  (∀ c, c ≠ Coin.F ∧ c ≠ Coin.A → relative_position c Coin.C = Position.Below ∨ relative_position c Coin.C = Position.Unknown) →
  (relative_position Coin.F Coin.A = Position.Above) ∧
  (relative_position Coin.A Coin.D = Position.Above) ∧
  (relative_position Coin.D Coin.E = Position.Above) ∧
  (relative_position Coin.E Coin.C = Position.Above ∨ relative_position Coin.E Coin.C = Position.Unknown) ∧
  (relative_position Coin.C Coin.B = Position.Above) := by
  sorry

end NUMINAMATH_CALUDE_coin_order_l3914_391449


namespace NUMINAMATH_CALUDE_squirrel_walnuts_l3914_391444

/-- The number of walnuts left in the squirrels' burrow after their gathering and eating activities. -/
def walnuts_left (initial : ℕ) (boy_gathered : ℕ) (boy_dropped : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) : ℕ :=
  initial + (boy_gathered - boy_dropped) + girl_brought - girl_ate

/-- Theorem stating that given the specific conditions of the problem, the number of walnuts left is 20. -/
theorem squirrel_walnuts : walnuts_left 12 6 1 5 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_walnuts_l3914_391444


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l3914_391406

theorem consecutive_integers_average (n : ℤ) : 
  (n * (n + 6) = 391) → 
  (((n + n + 1 + n + 2 + n + 3 + n + 4 + n + 5 + n + 6) : ℚ) / 7 = 20) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l3914_391406


namespace NUMINAMATH_CALUDE_middle_building_height_l3914_391498

/-- The height of the middle building in feet -/
def middle_height : ℝ := sorry

/-- The height of the left building in feet -/
def left_height : ℝ := 0.8 * middle_height

/-- The height of the right building in feet -/
def right_height : ℝ := middle_height + left_height - 20

/-- The total height of all three buildings in feet -/
def total_height : ℝ := 340

theorem middle_building_height :
  middle_height + left_height + right_height = total_height →
  middle_height = 340 / 5.2 :=
by sorry

end NUMINAMATH_CALUDE_middle_building_height_l3914_391498


namespace NUMINAMATH_CALUDE_stating_max_girls_in_class_l3914_391413

/-- Represents the number of students in the class -/
def total_students : ℕ := 25

/-- Represents the maximum number of girls in the class -/
def max_girls : ℕ := 13

/-- 
Theorem stating that given a class of 25 students where no two girls 
have the same number of boy friends, the maximum number of girls is 13.
-/
theorem max_girls_in_class :
  ∀ (girls boys : ℕ),
  girls + boys = total_students →
  (∀ (g₁ g₂ : ℕ), g₁ < girls → g₂ < girls → g₁ ≠ g₂ → 
    ∃ (b₁ b₂ : ℕ), b₁ ≤ boys ∧ b₂ ≤ boys ∧ b₁ ≠ b₂) →
  girls ≤ max_girls :=
by sorry

end NUMINAMATH_CALUDE_stating_max_girls_in_class_l3914_391413


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3914_391472

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 1 → 
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3914_391472


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l3914_391438

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 given boat and stream speeds -/
theorem rowing_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 60) 
  (h2 : stream_speed = 20) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check rowing_time_ratio

end NUMINAMATH_CALUDE_rowing_time_ratio_l3914_391438


namespace NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_sixth_l3914_391416

theorem arccos_difference_equals_negative_pi_sixth : 
  Real.arccos ((Real.sqrt 6 + 1) / (2 * Real.sqrt 3)) - Real.arccos (Real.sqrt (2/3)) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_difference_equals_negative_pi_sixth_l3914_391416


namespace NUMINAMATH_CALUDE_circle_points_theorem_l3914_391465

/-- The number of points on the circumference of the circle -/
def n : ℕ := 9

/-- Represents the property that no three points are collinear along any line passing through the circle's center -/
def no_three_collinear (points : Fin n → ℝ × ℝ) : Prop := sorry

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := Nat.choose n 3

/-- The number of distinct straight lines that can be drawn -/
def num_lines : ℕ := Nat.choose n 2

/-- Main theorem stating the number of triangles and lines -/
theorem circle_points_theorem (points : Fin n → ℝ × ℝ) 
  (h : no_three_collinear points) : 
  num_triangles = 84 ∧ num_lines = 36 := by sorry

end NUMINAMATH_CALUDE_circle_points_theorem_l3914_391465


namespace NUMINAMATH_CALUDE_function_decreasing_interval_l3914_391456

/-- Given a function f(x) = kx³ - 3(k+1)x² - k² + 1 where k > 0,
    if the decreasing interval of f(x) is (0, 4), then k = 1. -/
theorem function_decreasing_interval (k : ℝ) (h₁ : k > 0) :
  let f : ℝ → ℝ := λ x => k * x^3 - 3 * (k + 1) * x^2 - k^2 + 1
  (∀ x ∈ Set.Ioo 0 4, ∀ y ∈ Set.Ioo 0 4, x < y → f x > f y) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_interval_l3914_391456


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_AX_length_l3914_391404

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point X on BC
def X (t : Triangle) : ℝ × ℝ := sorry

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem angle_bisector_theorem (t : Triangle) :
  -- CX bisects ∠ACB
  (length (t.A) (X t)) * (length (t.B) (t.C)) =
  (length (t.A) (t.C)) * (length (t.B) (X t)) :=
sorry

-- State the main theorem
theorem AX_length (t : Triangle) :
  -- Conditions
  length (t.B) (t.C) = 50 →
  length (t.A) (t.C) = 40 →
  length (t.B) (X t) = 35 →
  -- CX bisects ∠ACB (using angle_bisector_theorem)
  (length (t.A) (X t)) * (length (t.B) (t.C)) =
  (length (t.A) (t.C)) * (length (t.B) (X t)) →
  -- Conclusion
  length (t.A) (X t) = 28 :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_AX_length_l3914_391404


namespace NUMINAMATH_CALUDE_jean_trips_l3914_391405

theorem jean_trips (total : ℕ) (extra : ℕ) (h1 : total = 40) (h2 : extra = 6) :
  ∃ (bill : ℕ) (jean : ℕ), bill + jean = total ∧ jean = bill + extra ∧ jean = 23 :=
by sorry

end NUMINAMATH_CALUDE_jean_trips_l3914_391405


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l3914_391407

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 693 :=
by sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_13_l3914_391407


namespace NUMINAMATH_CALUDE_expression_simplification_l3914_391414

theorem expression_simplification (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) (h3 : x ≠ 3) :
  (((x^2 - 2*x) / (x^2 - 4*x + 4) - 3 / (x - 2)) / ((x - 3) / (x^2 - 4))) = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3914_391414


namespace NUMINAMATH_CALUDE_repair_cost_is_correct_l3914_391487

/-- Calculates the total cost of car repair given the following conditions:
  * Two mechanics work on the car
  * First mechanic: $60/hour, 8 hours/day, 14 days
  * Second mechanic: $75/hour, 6 hours/day, 10 days
  * 15% discount on first mechanic's labor cost
  * 10% discount on second mechanic's labor cost
  * Parts cost: $3,200
  * 7% sales tax on final bill after discounts
-/
def totalRepairCost (
  mechanic1_rate : ℝ)
  (mechanic1_hours : ℝ)
  (mechanic1_days : ℝ)
  (mechanic2_rate : ℝ)
  (mechanic2_hours : ℝ)
  (mechanic2_days : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (parts_cost : ℝ)
  (sales_tax_rate : ℝ) : ℝ :=
  let mechanic1_cost := mechanic1_rate * mechanic1_hours * mechanic1_days
  let mechanic2_cost := mechanic2_rate * mechanic2_hours * mechanic2_days
  let discounted_mechanic1_cost := mechanic1_cost * (1 - discount1)
  let discounted_mechanic2_cost := mechanic2_cost * (1 - discount2)
  let total_before_tax := discounted_mechanic1_cost + discounted_mechanic2_cost + parts_cost
  total_before_tax * (1 + sales_tax_rate)

/-- Theorem stating that the total repair cost is $13,869.34 given the specific conditions -/
theorem repair_cost_is_correct :
  totalRepairCost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_is_correct_l3914_391487


namespace NUMINAMATH_CALUDE_smallest_value_of_a_l3914_391440

def a (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 10 * n.val + 3

theorem smallest_value_of_a (n : ℕ+) :
  a n ≥ a 2 ∧ a n ≥ a 3 ∧ (a 2 = a 3) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_of_a_l3914_391440


namespace NUMINAMATH_CALUDE_polygon_sides_l3914_391499

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 3420 → ∃ n : ℕ, n = 21 ∧ sum_interior_angles = 180 * (n - 2) := by
  sorry

#check polygon_sides

end NUMINAMATH_CALUDE_polygon_sides_l3914_391499


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3914_391433

theorem greatest_divisor_with_remainders : Nat.gcd (1442 - 12) (1816 - 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3914_391433


namespace NUMINAMATH_CALUDE_candy_difference_l3914_391401

/-- Represents the number of boxes -/
def num_boxes : ℕ := 10

/-- Represents the total number of candies in all boxes -/
def total_candies : ℕ := 320

/-- Represents the number of candies in the second box -/
def second_box_candies : ℕ := 11

/-- Calculates the sum of an arithmetic progression -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℕ) * d) / 2

/-- Theorem stating the common difference between consecutive boxes -/
theorem candy_difference : 
  ∃ (d : ℕ), 
    arithmetic_sum (second_box_candies - d : ℚ) d num_boxes = total_candies ∧ 
    d = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l3914_391401


namespace NUMINAMATH_CALUDE_sum_of_roots_l3914_391419

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y z : ℝ, x^2 - x - 34 = (x - y) * (x - z) ∧ y + z = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3914_391419


namespace NUMINAMATH_CALUDE_max_value_abc_l3914_391426

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  2*a*b*Real.sqrt 2 + 2*a*c + 2*b*c ≤ 1 / Real.sqrt 2 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a'^2 + b'^2 + c'^2 = 1 ∧
  2*a'*b'*Real.sqrt 2 + 2*a'*c' + 2*b'*c' = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_abc_l3914_391426


namespace NUMINAMATH_CALUDE_least_repeating_block_seven_thirteenths_l3914_391410

/-- The least number of digits in a repeating block of 7/13 -/
def repeating_block_length : ℕ := 6

/-- 7/13 is a repeating decimal -/
axiom seven_thirteenths_repeats : ∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n + (↑k : ℚ) / (10^repeating_block_length - 1)

theorem least_repeating_block_seven_thirteenths :
  ∀ m : ℕ, m < repeating_block_length → ¬∃ (n : ℕ) (k : ℕ+), (7 : ℚ) / 13 = ↑n + (↑k : ℚ) / (10^m - 1) :=
sorry

end NUMINAMATH_CALUDE_least_repeating_block_seven_thirteenths_l3914_391410


namespace NUMINAMATH_CALUDE_school_referendum_l3914_391476

theorem school_referendum (U : Finset ℕ) (A B : Finset ℕ) : 
  Finset.card U = 198 →
  Finset.card A = 149 →
  Finset.card B = 119 →
  Finset.card (U \ (A ∪ B)) = 29 →
  Finset.card (A ∩ B) = 99 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_l3914_391476


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l3914_391427

theorem overall_gain_percentage (cost_A cost_B cost_C gain_A gain_B gain_C : ℚ) :
  cost_A = 700 ∧ cost_B = 500 ∧ cost_C = 300 ∧
  gain_A = 70 ∧ gain_B = 50 ∧ gain_C = 30 →
  (gain_A + gain_B + gain_C) / (cost_A + cost_B + cost_C) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_l3914_391427


namespace NUMINAMATH_CALUDE_child_attraction_fee_is_two_l3914_391467

/-- Represents the cost of various tickets and the family composition --/
structure ParkCosts where
  entrance_fee : ℕ
  adult_attraction_fee : ℕ
  child_attraction_fee : ℕ
  num_children : ℕ
  num_parents : ℕ
  num_grandparents : ℕ
  total_cost : ℕ

/-- Theorem stating that given the conditions, the child attraction fee is $2 --/
theorem child_attraction_fee_is_two (c : ParkCosts)
  (h1 : c.entrance_fee = 5)
  (h2 : c.adult_attraction_fee = 4)
  (h3 : c.num_children = 4)
  (h4 : c.num_parents = 2)
  (h5 : c.num_grandparents = 1)
  (h6 : c.total_cost = 55)
  (h7 : c.total_cost = (c.num_children + c.num_parents + c.num_grandparents) * c.entrance_fee +
                       (c.num_parents + c.num_grandparents) * c.adult_attraction_fee +
                       c.num_children * c.child_attraction_fee) :
  c.child_attraction_fee = 2 :=
by sorry

end NUMINAMATH_CALUDE_child_attraction_fee_is_two_l3914_391467


namespace NUMINAMATH_CALUDE_solve_equation_l3914_391411

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (b : ℝ) : Prop :=
  (2 - i) * (4 * i) = 4 - b * i

-- State the theorem
theorem solve_equation : ∃ b : ℝ, equation b ∧ b = -8 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3914_391411


namespace NUMINAMATH_CALUDE_min_sum_of_weights_l3914_391430

theorem min_sum_of_weights (S : ℕ) : 
  S > 280 ∧ S % 70 = 30 → S ≥ 310 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_weights_l3914_391430


namespace NUMINAMATH_CALUDE_marble_remainder_l3914_391448

theorem marble_remainder (r p : ℕ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by sorry

end NUMINAMATH_CALUDE_marble_remainder_l3914_391448


namespace NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3914_391415

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℚ
  y_intercept : ℚ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℚ := -l.y_intercept / l.slope

/-- Two lines are perpendicular if their slopes are negative reciprocals -/
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

theorem x_intercept_of_perpendicular_line (given_line perp_line : Line) :
  given_line.slope = -5/3 →
  perpendicular given_line perp_line →
  perp_line.y_intercept = -4 →
  x_intercept perp_line = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_perpendicular_line_l3914_391415


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3914_391469

theorem product_from_hcf_lcm (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 128) :
  a * b = 2560 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3914_391469


namespace NUMINAMATH_CALUDE_exists_k_for_all_m_unique_k_characterization_l3914_391492

/-- The number of elements in {k+1, k+2, ..., 2k} with exactly 3 ones in binary representation -/
def f (k : ℕ+) : ℕ := sorry

/-- There exists a k for every m such that f(k) = m -/
theorem exists_k_for_all_m (m : ℕ+) : ∃ k : ℕ+, f k = m := by sorry

/-- Characterization of m for which there's exactly one k satisfying f(k) = m -/
theorem unique_k_characterization (m : ℕ+) : 
  (∃! k : ℕ+, f k = m) ↔ ∃ n : ℕ, n ≥ 2 ∧ m = n * (n - 1) / 2 + 1 := by sorry

end NUMINAMATH_CALUDE_exists_k_for_all_m_unique_k_characterization_l3914_391492


namespace NUMINAMATH_CALUDE_complex_division_fourth_quadrant_l3914_391494

theorem complex_division_fourth_quadrant : 
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 + i
  let z₂ : ℂ := 1 + 2*i
  (z₁ / z₂).re > 0 ∧ (z₁ / z₂).im < 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_division_fourth_quadrant_l3914_391494


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l3914_391471

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := (fun (x y : ℝ) => x^2 - 2*x + y^2 + 6*y + 2 = 0)
  let circle2 := (fun (x y : ℝ) => x^2 + 6*x + y^2 - 2*y + 9 = 0)
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (p1 p2 : ℝ × ℝ),
    circle1 p1.1 p1.2 → circle2 p2.1 p2.2 →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l3914_391471


namespace NUMINAMATH_CALUDE_some_number_value_l3914_391423

theorem some_number_value (some_number : ℝ) : 
  (40 / some_number) * (40 / 80) = 1 → some_number = 80 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l3914_391423


namespace NUMINAMATH_CALUDE_angle_measure_l3914_391421

theorem angle_measure (x : ℝ) : 
  (90 - x) = 3 * (180 - x) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l3914_391421


namespace NUMINAMATH_CALUDE_sine_of_supplementary_angles_l3914_391474

theorem sine_of_supplementary_angles (VPQ VPS : Real) 
  (h1 : VPS + VPQ = Real.pi)  -- Supplementary angles
  (h2 : Real.sin VPQ = 3/5) : 
  Real.sin VPS = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_supplementary_angles_l3914_391474


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l3914_391420

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_9_15_under_100 :
  ∃ (k : ℕ), k < 100 ∧ is_common_multiple 9 15 k ∧
  ∀ (m : ℕ), m < 100 → is_common_multiple 9 15 m → m ≤ k :=
by
  use 90
  sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_100_l3914_391420


namespace NUMINAMATH_CALUDE_train_passing_time_l3914_391496

theorem train_passing_time (fast_length slow_length : ℝ) (time_slow_observes : ℝ) :
  fast_length = 150 →
  slow_length = 200 →
  time_slow_observes = 6 →
  ∃ time_fast_observes : ℝ,
    time_fast_observes = 8 ∧
    fast_length / time_slow_observes = slow_length / time_fast_observes :=
by sorry

end NUMINAMATH_CALUDE_train_passing_time_l3914_391496


namespace NUMINAMATH_CALUDE_vacation_pictures_l3914_391418

theorem vacation_pictures (zoo_pics : ℕ) (museum_pics : ℕ) (deleted_pics : ℕ)
  (h1 : zoo_pics = 41)
  (h2 : museum_pics = 29)
  (h3 : deleted_pics = 15) :
  zoo_pics + museum_pics - deleted_pics = 55 := by
  sorry

end NUMINAMATH_CALUDE_vacation_pictures_l3914_391418


namespace NUMINAMATH_CALUDE_egyptian_341_correct_l3914_391412

/-- Represents an Egyptian numeral symbol -/
inductive EgyptianSymbol
  | hundreds
  | tens
  | ones

/-- Converts an Egyptian symbol to its numeric value -/
def symbolValue (s : EgyptianSymbol) : ℕ :=
  match s with
  | EgyptianSymbol.hundreds => 100
  | EgyptianSymbol.tens => 10
  | EgyptianSymbol.ones => 1

/-- Represents a list of Egyptian symbols -/
def EgyptianNumber := List EgyptianSymbol

/-- Converts an Egyptian number to its decimal value -/
def egyptianToDecimal (en : EgyptianNumber) : ℕ :=
  en.foldl (fun acc s => acc + symbolValue s) 0

/-- The Egyptian representation of 234 -/
def egyptian234 : EgyptianNumber :=
  [EgyptianSymbol.hundreds, EgyptianSymbol.tens, EgyptianSymbol.tens,
   EgyptianSymbol.ones, EgyptianSymbol.ones, EgyptianSymbol.ones, EgyptianSymbol.ones]

/-- The Egyptian representation of 123 -/
def egyptian123 : EgyptianNumber :=
  [EgyptianSymbol.tens, EgyptianSymbol.tens, EgyptianSymbol.tens,
   EgyptianSymbol.ones, EgyptianSymbol.ones, EgyptianSymbol.ones]

/-- The proposed Egyptian representation of 341 -/
def egyptian341 : EgyptianNumber :=
  [EgyptianSymbol.hundreds, EgyptianSymbol.hundreds, EgyptianSymbol.hundreds,
   EgyptianSymbol.tens, EgyptianSymbol.tens, EgyptianSymbol.tens, EgyptianSymbol.tens,
   EgyptianSymbol.ones]

theorem egyptian_341_correct :
  egyptianToDecimal egyptian234 = 234 ∧
  egyptianToDecimal egyptian123 = 123 →
  egyptianToDecimal egyptian341 = 341 :=
by sorry

end NUMINAMATH_CALUDE_egyptian_341_correct_l3914_391412


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3914_391451

theorem complex_fraction_simplification :
  (I : ℂ) / (Real.sqrt 7 + 3 * I) = (3 : ℂ) / 16 + (Real.sqrt 7 / 16) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3914_391451


namespace NUMINAMATH_CALUDE_not_necessarily_similar_remaining_parts_l3914_391461

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define a function to split a triangle into two smaller triangles
def split (t : Triangle) : Triangle × Triangle := sorry

theorem not_necessarily_similar_remaining_parts 
  (T1 T2 : Triangle) 
  (h_similar : similar T1 T2) 
  (T1_split : Triangle × Triangle) 
  (T2_split : Triangle × Triangle)
  (h_T1_split : T1_split = split T1)
  (h_T2_split : T2_split = split T2)
  (h_part_similar : similar T1_split.1 T2_split.1) :
  ¬ (∀ (T1 T2 : Triangle) (h_similar : similar T1 T2) 
      (T1_split T2_split : Triangle × Triangle)
      (h_T1_split : T1_split = split T1)
      (h_T2_split : T2_split = split T2)
      (h_part_similar : similar T1_split.1 T2_split.1),
    similar T1_split.2 T2_split.2) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_similar_remaining_parts_l3914_391461


namespace NUMINAMATH_CALUDE_leo_commute_cost_l3914_391429

theorem leo_commute_cost (total_cost : ℕ) (working_days : ℕ) (trips_per_day : ℕ) 
  (h1 : total_cost = 960)
  (h2 : working_days = 20)
  (h3 : trips_per_day = 2) :
  total_cost / (working_days * trips_per_day) = 24 := by
sorry

end NUMINAMATH_CALUDE_leo_commute_cost_l3914_391429


namespace NUMINAMATH_CALUDE_wallpaper_removal_time_l3914_391490

/-- Time to remove wallpaper from one wall in hours -/
def time_per_wall : ℕ := 2

/-- Number of walls in the dining room -/
def dining_room_walls : ℕ := 4

/-- Number of walls in the living room -/
def living_room_walls : ℕ := 4

/-- Number of walls already completed in the dining room -/
def completed_walls : ℕ := 1

/-- Calculates the total time to remove remaining wallpaper -/
def total_time : ℕ :=
  time_per_wall * (dining_room_walls - completed_walls) +
  time_per_wall * living_room_walls

theorem wallpaper_removal_time : total_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_removal_time_l3914_391490


namespace NUMINAMATH_CALUDE_model_A_better_fit_l3914_391478

-- Define the R² values for models A and B
def R_squared_A : ℝ := 0.96
def R_squared_B : ℝ := 0.85

-- Define a function to compare fitting effects based on R²
def better_fit (r1 r2 : ℝ) : Prop := r1 > r2

-- Theorem statement
theorem model_A_better_fit :
  better_fit R_squared_A R_squared_B :=
by sorry

end NUMINAMATH_CALUDE_model_A_better_fit_l3914_391478


namespace NUMINAMATH_CALUDE_opposite_seven_is_nine_or_eleven_l3914_391479

def DieNumbers : Finset ℕ := {6, 7, 8, 9, 10, 11}

def isValidOpposite (n : ℕ) : Prop :=
  n ∈ DieNumbers ∧ n ≠ 7 ∧
  ∃ (a b c d e : ℕ),
    {a, b, c, d, e, 7} = DieNumbers ∧
    (a + b + c + d = 33 ∨ a + b + c + d = 35) ∧
    (e + 7 = 16 ∨ e + 7 = 17 ∨ e + 7 = 18)

theorem opposite_seven_is_nine_or_eleven :
  ∀ n, isValidOpposite n → n = 9 ∨ n = 11 := by sorry

end NUMINAMATH_CALUDE_opposite_seven_is_nine_or_eleven_l3914_391479


namespace NUMINAMATH_CALUDE_expression_evaluation_l3914_391458

theorem expression_evaluation : (3^2 - 5) / (0.08 * 7 + 2) = 1.5625 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3914_391458


namespace NUMINAMATH_CALUDE_max_value_abs_sum_l3914_391442

theorem max_value_abs_sum (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) : 
  ∃ (m : ℝ), m = 5 ∧ ∀ x, 0 ≤ x ∧ x ≤ 4 → |x - 2| + |3 - x| ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_abs_sum_l3914_391442


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3914_391431

/-- Given two cars driving toward each other, prove that they meet in 4 hours -/
theorem cars_meeting_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : 
  speed1 = 100 →
  speed1 = 1.25 * speed2 →
  distance = 720 →
  distance / (speed1 + speed2) = 4 := by
sorry


end NUMINAMATH_CALUDE_cars_meeting_time_l3914_391431


namespace NUMINAMATH_CALUDE_line_y_coordinate_l3914_391489

/-- Given a line in a rectangular coordinate system passing through points (-2, y), (10, 3),
    and having an x-intercept of 4, prove that the y-coordinate of the point with x-coordinate -2 is -3. -/
theorem line_y_coordinate (y : ℝ) : 
  ∃ (m b : ℝ), 
    (∀ x, y = m * x + b) ∧  -- Line equation
    (y = m * (-2) + b) ∧    -- Line passes through (-2, y)
    (3 = m * 10 + b) ∧      -- Line passes through (10, 3)
    (0 = m * 4 + b) →       -- Line has x-intercept at 4
  y = -3 := by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l3914_391489


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l3914_391439

/-- Represents a binary digit (0 or 1) -/
inductive BinaryDigit
| zero : BinaryDigit
| one : BinaryDigit

/-- Represents a binary number as a list of binary digits -/
def BinaryNumber := List BinaryDigit

/-- Converts a binary number to its decimal equivalent -/
def binaryToDecimal (bin : BinaryNumber) : ℕ :=
  bin.enum.foldl (fun acc (i, digit) =>
    acc + match digit with
      | BinaryDigit.zero => 0
      | BinaryDigit.one => 2^i
  ) 0

/-- The binary representation of 1101 -/
def bin1101 : BinaryNumber :=
  [BinaryDigit.one, BinaryDigit.one, BinaryDigit.zero, BinaryDigit.one]

theorem binary_1101_equals_13 :
  binaryToDecimal bin1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l3914_391439


namespace NUMINAMATH_CALUDE_colored_pencils_count_l3914_391491

def number_of_packs : ℕ := 7
def pencils_per_pack : ℕ := 10
def difference : ℕ := 3

theorem colored_pencils_count :
  let total_pencils := number_of_packs * pencils_per_pack
  let colored_pencils := total_pencils + difference
  colored_pencils = 73 := by
  sorry

end NUMINAMATH_CALUDE_colored_pencils_count_l3914_391491


namespace NUMINAMATH_CALUDE_four_line_theorem_l3914_391422

-- Define the type for lines in space
variable (Line : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (para : Line → Line → Prop)

-- State the theorem
theorem four_line_theorem (a b c d : Line) 
  (h1 : perp a b) (h2 : perp b c) (h3 : perp c d) (h4 : perp d a) :
  para b d ∨ para a c :=
sorry

end NUMINAMATH_CALUDE_four_line_theorem_l3914_391422


namespace NUMINAMATH_CALUDE_correct_average_l3914_391445

theorem correct_average (n : ℕ) (initial_avg : ℚ) (wrong_num correct_num : ℚ) :
  n = 10 ∧ initial_avg = 18 ∧ wrong_num = 26 ∧ correct_num = 36 →
  (n * initial_avg - wrong_num + correct_num) / n = 19 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l3914_391445


namespace NUMINAMATH_CALUDE_median_inequality_exists_l3914_391452

/-- A dataset is represented as a list of real numbers -/
def Dataset := List ℝ

/-- The median of a dataset -/
def median (d : Dataset) : ℝ := sorry

/-- Count of values in a dataset less than a given value -/
def count_less_than (d : Dataset) (x : ℝ) : ℕ := sorry

/-- Count of values in a dataset greater than a given value -/
def count_greater_than (d : Dataset) (x : ℝ) : ℕ := sorry

/-- Theorem: There exists a dataset where the number of values greater than 
    the median is not equal to the number of values less than the median -/
theorem median_inequality_exists : 
  ∃ (d : Dataset), count_greater_than d (median d) ≠ count_less_than d (median d) := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_exists_l3914_391452
