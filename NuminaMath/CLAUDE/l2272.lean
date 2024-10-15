import Mathlib

namespace NUMINAMATH_CALUDE_red_stamp_price_l2272_227202

theorem red_stamp_price (simon_stamps : ℕ) (peter_stamps : ℕ) (white_stamp_price : ℚ) (price_difference : ℚ) :
  simon_stamps = 30 →
  peter_stamps = 80 →
  white_stamp_price = 0.20 →
  price_difference = 1 →
  ∃ (red_stamp_price : ℚ), 
    red_stamp_price * simon_stamps - white_stamp_price * peter_stamps = price_difference ∧
    red_stamp_price = 17 / 30 := by
  sorry

end NUMINAMATH_CALUDE_red_stamp_price_l2272_227202


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l2272_227208

noncomputable def y (k : ℝ) (x : ℝ) : ℝ :=
  if k ≠ 0 then
    1/2 * ((1/(1 - k*x))^(1/k) + (1 - k*x)^(1/k))
  else
    Real.cosh x

noncomputable def z (k : ℝ) (x : ℝ) : ℝ :=
  if k ≠ 0 then
    1/2 * ((1/(1 - k*x))^(1/k) - (1 - k*x)^(1/k))
  else
    Real.sinh x

theorem solution_satisfies_conditions (k : ℝ) :
  (∀ x, (deriv (y k)) x = (z k x) * ((y k x) + (z k x))^k) ∧
  (∀ x, (deriv (z k)) x = (y k x) * ((y k x) + (z k x))^k) ∧
  y k 0 = 1 ∧
  z k 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_conditions_l2272_227208


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l2272_227216

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

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l2272_227216


namespace NUMINAMATH_CALUDE_line_parametric_equation_l2272_227293

/-- The standard parametric equation of a line passing through a point with a given slope angle. -/
theorem line_parametric_equation (P : ℝ × ℝ) (θ : ℝ) :
  P = (1, -1) → θ = π / 3 →
  ∃ f g : ℝ → ℝ, 
    (∀ t, f t = 1 + (1/2) * t) ∧ 
    (∀ t, g t = -1 + (Real.sqrt 3 / 2) * t) ∧
    (∀ t, (f t, g t) ∈ {(x, y) | y - P.2 = Real.tan θ * (x - P.1)}) :=
sorry

end NUMINAMATH_CALUDE_line_parametric_equation_l2272_227293


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2272_227272

theorem inequality_solution_set (x : ℝ) : 
  (∃ y, y > 1 ∧ y < x) ↔ (x^2 - x) * (Real.exp x - 1) > 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2272_227272


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2272_227251

def A : Set ℝ := {x : ℝ | -x^2 + x + 6 > 0}
def B : Set ℝ := {x : ℝ | x^2 + 2*x - 8 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2272_227251


namespace NUMINAMATH_CALUDE_board_numbers_product_l2272_227274

def pairwise_sums (a b c d e : ℤ) : Finset ℤ :=
  {a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e}

theorem board_numbers_product (a b c d e : ℤ) :
  pairwise_sums a b c d e = {-1, 4, 6, 9, 10, 11, 15, 16, 20, 22} →
  a * b * c * d * e = -4914 := by
  sorry

end NUMINAMATH_CALUDE_board_numbers_product_l2272_227274


namespace NUMINAMATH_CALUDE_supermarket_max_profit_l2272_227285

/-- Represents the daily profit function for a supermarket selling daily necessities -/
def daily_profit (x : ℝ) : ℝ :=
  (200 - 10 * (x - 50)) * (x - 40)

/-- The maximum daily profit achievable by the supermarket -/
def max_daily_profit : ℝ := 2250

theorem supermarket_max_profit :
  ∃ (x : ℝ), daily_profit x = max_daily_profit ∧
  ∀ (y : ℝ), daily_profit y ≤ max_daily_profit :=
by sorry

end NUMINAMATH_CALUDE_supermarket_max_profit_l2272_227285


namespace NUMINAMATH_CALUDE_smallest_solution_is_negative_85_l2272_227254

def floor_equation (x : ℤ) : Prop :=
  Int.floor (x / 2) + Int.floor (x / 3) + Int.floor (x / 7) = x

theorem smallest_solution_is_negative_85 :
  (∀ y < -85, ¬ floor_equation y) ∧ floor_equation (-85) := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_is_negative_85_l2272_227254


namespace NUMINAMATH_CALUDE_floor_composition_identity_l2272_227297

open Real

theorem floor_composition_identity (α : ℝ) (n : ℕ) (h : α > 1) :
  let β := 1 / α
  let fₐ (x : ℝ) := ⌊α * x + 1/2⌋
  let fᵦ (x : ℝ) := ⌊β * x + 1/2⌋
  fᵦ (fₐ n) = n := by
  sorry

end NUMINAMATH_CALUDE_floor_composition_identity_l2272_227297


namespace NUMINAMATH_CALUDE_complex_symmetry_product_l2272_227264

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.re = 1 ∧ z₁.im = 2) → 
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) → 
  z₁ * z₂ = -5 := by
sorry

end NUMINAMATH_CALUDE_complex_symmetry_product_l2272_227264


namespace NUMINAMATH_CALUDE_exists_cube_with_2014_prime_points_l2272_227232

/-- A point in 3D space with integer coordinates -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Predicate to check if a number is prime -/
def isPrime (n : ℤ) : Prop := sorry

/-- Predicate to check if a point is in the first octant -/
def isFirstOctant (p : Point3D) : Prop :=
  p.x > 0 ∧ p.y > 0 ∧ p.z > 0

/-- Predicate to check if a point has all prime coordinates -/
def isPrimePoint (p : Point3D) : Prop :=
  isPrime p.x ∧ isPrime p.y ∧ isPrime p.z

/-- Definition of a cube in 3D space -/
structure Cube where
  corner : Point3D
  edgeLength : ℤ

/-- Predicate to check if a point is inside a cube -/
def isInsideCube (p : Point3D) (c : Cube) : Prop :=
  c.corner.x ≤ p.x ∧ p.x < c.corner.x + c.edgeLength ∧
  c.corner.y ≤ p.y ∧ p.y < c.corner.y + c.edgeLength ∧
  c.corner.z ≤ p.z ∧ p.z < c.corner.z + c.edgeLength

/-- The main theorem to be proved -/
theorem exists_cube_with_2014_prime_points :
  ∃ (c : Cube), c.edgeLength = 2014 ∧
    isFirstOctant c.corner ∧
    (∃ (points : Finset Point3D),
      points.card = 2014 ∧
      (∀ p ∈ points, isPrimePoint p ∧ isInsideCube p c) ∧
      (∀ p : Point3D, isPrimePoint p ∧ isInsideCube p c → p ∈ points)) :=
sorry

end NUMINAMATH_CALUDE_exists_cube_with_2014_prime_points_l2272_227232


namespace NUMINAMATH_CALUDE_train_speeds_l2272_227210

theorem train_speeds (distance : ℝ) (time : ℝ) (speed_difference : ℝ) 
  (h1 : distance = 450)
  (h2 : time = 5)
  (h3 : speed_difference = 6) :
  ∃ (speed1 speed2 : ℝ),
    speed2 = speed1 + speed_difference ∧
    distance = (speed1 + speed2) * time ∧
    speed1 = 42 ∧
    speed2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_train_speeds_l2272_227210


namespace NUMINAMATH_CALUDE_invitation_ways_l2272_227219

def number_of_teachers : ℕ := 10
def teachers_to_invite : ℕ := 6

def ways_to_invite (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem invitation_ways : 
  ways_to_invite number_of_teachers teachers_to_invite - 
  ways_to_invite (number_of_teachers - 2) (teachers_to_invite - 2) = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_invitation_ways_l2272_227219


namespace NUMINAMATH_CALUDE_max_value_on_circle_l2272_227288

theorem max_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + b^2 = 2 → 3*a + 4*b ≤ max) ∧ max = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l2272_227288


namespace NUMINAMATH_CALUDE_circle_constraint_extrema_sum_l2272_227250

theorem circle_constraint_extrema_sum (x y : ℝ) :
  x^2 + y^2 = 1 →
  ∃ (min max : ℝ),
    (∀ x' y' : ℝ, x'^2 + y'^2 = 1 → 
      min ≤ (x'-3)^2 + (y'+4)^2 ∧ (x'-3)^2 + (y'+4)^2 ≤ max) ∧
    min + max = 52 := by
  sorry

end NUMINAMATH_CALUDE_circle_constraint_extrema_sum_l2272_227250


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2272_227215

theorem cubic_equation_solutions :
  ∀ x : ℝ, (x^3 - 3*x^2*(Real.sqrt 3) + 9*x - 3*(Real.sqrt 3)) + (x - Real.sqrt 3)^2 = 0 ↔ 
  x = Real.sqrt 3 ∨ x = -1 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2272_227215


namespace NUMINAMATH_CALUDE_trajectory_and_line_m_l2272_227262

/-- The distance ratio condition for point P -/
def distance_ratio (x y : ℝ) : Prop :=
  (((x - 3 * Real.sqrt 3)^2 + y^2).sqrt) / (|x - 4 * Real.sqrt 3|) = Real.sqrt 3 / 2

/-- The equation of the ellipse -/
def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

/-- The equation of line m -/
def on_line_m (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

/-- The midpoint condition -/
def is_midpoint (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = 4 ∧ (y₁ + y₂) / 2 = 2

theorem trajectory_and_line_m :
  (∀ x y : ℝ, distance_ratio x y ↔ on_ellipse x y) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    on_ellipse x₁ y₁ ∧ on_ellipse x₂ y₂ ∧ is_midpoint x₁ y₁ x₂ y₂ →
    on_line_m x₁ y₁ ∧ on_line_m x₂ y₂) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_line_m_l2272_227262


namespace NUMINAMATH_CALUDE_incorrect_average_calculation_l2272_227201

theorem incorrect_average_calculation (n : ℕ) (correct_avg : ℚ) (error : ℚ) :
  n = 10 ∧ correct_avg = 17 ∧ error = 10 →
  (n * correct_avg - error) / n = 16 := by
sorry

end NUMINAMATH_CALUDE_incorrect_average_calculation_l2272_227201


namespace NUMINAMATH_CALUDE_chessboard_cover_l2272_227214

def coverWays (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | k + 3 => coverWays (k + 2) + coverWays (k + 1)

theorem chessboard_cover : coverWays 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_chessboard_cover_l2272_227214


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2272_227235

/-- Given a line segment with midpoint (2, 3) and one endpoint (5, -1), prove that the other endpoint is (-1, 7) -/
theorem other_endpoint_of_line_segment (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (2, 3) → endpoint1 = (5, -1) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, 7) := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l2272_227235


namespace NUMINAMATH_CALUDE_problem_solution_l2272_227218

theorem problem_solution :
  (∀ (a : ℝ), a ≠ 0 → (a^2)^3 / (-a)^2 = a^4) ∧
  (∀ (a b : ℝ), (a+2*b)*(a+b) - 3*a*(a+b) = -2*a^2 + 2*b^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2272_227218


namespace NUMINAMATH_CALUDE_cole_total_students_l2272_227211

/-- The number of students in Ms. Cole's math classes -/
structure ColeMathClasses where
  sixth_level : ℕ
  fourth_level : ℕ
  seventh_level : ℕ

/-- The conditions for Ms. Cole's math classes -/
def cole_math_class_conditions (c : ColeMathClasses) : Prop :=
  c.sixth_level = 40 ∧
  c.fourth_level = 4 * c.sixth_level ∧
  c.seventh_level = 2 * c.fourth_level

/-- The theorem stating the total number of students Ms. Cole teaches -/
theorem cole_total_students (c : ColeMathClasses) 
  (h : cole_math_class_conditions c) : 
  c.sixth_level + c.fourth_level + c.seventh_level = 520 := by
  sorry


end NUMINAMATH_CALUDE_cole_total_students_l2272_227211


namespace NUMINAMATH_CALUDE_blue_then_green_probability_l2272_227290

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

end NUMINAMATH_CALUDE_blue_then_green_probability_l2272_227290


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2272_227295

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a4 : a 4 = 1) 
  (h_a7 : a 7 = 8) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2272_227295


namespace NUMINAMATH_CALUDE_schur_like_inequality_l2272_227237

theorem schur_like_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) ≥ a + b + c :=
sorry

end NUMINAMATH_CALUDE_schur_like_inequality_l2272_227237


namespace NUMINAMATH_CALUDE_total_value_is_18_60_l2272_227271

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a half-dollar coin in dollars -/
def half_dollar_value : ℚ := 0.50

/-- The value of a dollar coin in dollars -/
def dollar_coin_value : ℚ := 1.00

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 25

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 15

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 12

/-- The number of half-dollar coins Tom found -/
def num_half_dollars : ℕ := 7

/-- The number of dollar coins Tom found -/
def num_dollar_coins : ℕ := 3

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 375

/-- The total value of the coins Tom found -/
def total_value : ℚ :=
  num_quarters * quarter_value +
  num_dimes * dime_value +
  num_nickels * nickel_value +
  num_half_dollars * half_dollar_value +
  num_dollar_coins * dollar_coin_value +
  num_pennies * penny_value

theorem total_value_is_18_60 : total_value = 18.60 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_18_60_l2272_227271


namespace NUMINAMATH_CALUDE_apple_distribution_exists_and_unique_l2272_227230

/-- Represents the last names of the children -/
inductive LastName
| Smith
| Brown
| Jones
| Robinson

/-- Represents a child with their name and number of apples -/
structure Child where
  firstName : String
  lastName : LastName
  apples : Nat

/-- The problem statement -/
theorem apple_distribution_exists_and_unique :
  ∃! (distribution : List Child),
    (distribution.length = 8) ∧
    (distribution.map (λ c => c.apples)).sum = 32 ∧
    (∃ ann ∈ distribution, ann.firstName = "Ann" ∧ ann.apples = 1) ∧
    (∃ mary ∈ distribution, mary.firstName = "Mary" ∧ mary.apples = 2) ∧
    (∃ jane ∈ distribution, jane.firstName = "Jane" ∧ jane.apples = 3) ∧
    (∃ kate ∈ distribution, kate.firstName = "Kate" ∧ kate.apples = 4) ∧
    (∃ ned ∈ distribution, ned.firstName = "Ned" ∧ ned.lastName = LastName.Smith ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Smith ∧ sister.apples = ned.apples) ∧
    (∃ tom ∈ distribution, tom.firstName = "Tom" ∧ tom.lastName = LastName.Brown ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Brown ∧ tom.apples = 2 * sister.apples) ∧
    (∃ bill ∈ distribution, bill.firstName = "Bill" ∧ bill.lastName = LastName.Jones ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Jones ∧ bill.apples = 3 * sister.apples) ∧
    (∃ jack ∈ distribution, jack.firstName = "Jack" ∧ jack.lastName = LastName.Robinson ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Robinson ∧ jack.apples = 4 * sister.apples) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_exists_and_unique_l2272_227230


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l2272_227255

/-- A hexagon that shares three sides with a rectangle and has the other three sides
    each equal to one of the rectangle's dimensions. -/
structure SpecialHexagon where
  rect_side1 : ℕ
  rect_side2 : ℕ

/-- The perimeter of the special hexagon. -/
def perimeter (h : SpecialHexagon) : ℕ :=
  2 * h.rect_side1 + 2 * h.rect_side2 + h.rect_side1 + h.rect_side2

/-- Theorem stating that the perimeter of the special hexagon with sides 7 and 5 is 36. -/
theorem special_hexagon_perimeter :
  ∃ (h : SpecialHexagon), h.rect_side1 = 7 ∧ h.rect_side2 = 5 ∧ perimeter h = 36 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l2272_227255


namespace NUMINAMATH_CALUDE_inequality_proof_l2272_227240

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / (a * b) + (b^2 + c^2) / (b * c) + (c^2 + a^2) / (c * a) ≥ 6 ∧
  (a + b) / 2 * (b + c) / 2 * (c + a) / 2 ≥ a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2272_227240


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2272_227236

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3) :
  (total_families * total_average) / (total_families - childless_families) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2272_227236


namespace NUMINAMATH_CALUDE_roses_remaining_proof_l2272_227278

def dozen : ℕ := 12

def initial_roses : ℕ := 3 * dozen

def roses_given_away : ℕ := initial_roses / 2

def roses_in_vase : ℕ := initial_roses - roses_given_away

def wilted_roses : ℕ := roses_in_vase / 3

def remaining_roses : ℕ := roses_in_vase - wilted_roses

theorem roses_remaining_proof :
  remaining_roses = 12 := by sorry

end NUMINAMATH_CALUDE_roses_remaining_proof_l2272_227278


namespace NUMINAMATH_CALUDE_divisibility_property_l2272_227226

theorem divisibility_property (n p q : ℕ) : 
  n > 0 → 
  Prime p → 
  q ∣ ((n + 1)^p - n^p) → 
  p ∣ (q - 1) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2272_227226


namespace NUMINAMATH_CALUDE_center_cell_is_seven_l2272_227203

/-- Represents a 3x3 grid with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are adjacent in the grid -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Checks if two numbers are consecutive -/
def consecutive (a b : Fin 9) : Prop :=
  a.val + 1 = b.val ∨ b.val + 1 = a.val

/-- The main theorem -/
theorem center_cell_is_seven (g : Grid)
  (all_nums : ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n)
  (consec_adjacent : ∀ i₁ j₁ i₂ j₂ : Fin 3,
    consecutive (g i₁ j₁) (g i₂ j₂) → adjacent (i₁, j₁) (i₂, j₂))
  (corner_sum : g 0 0 + g 0 2 + g 2 0 + g 2 2 = 18) :
  g 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_center_cell_is_seven_l2272_227203


namespace NUMINAMATH_CALUDE_incenter_vector_sum_implies_right_angle_l2272_227291

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a vector from a point to another point
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

-- Define vector addition
def add_vectors (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define scalar multiplication of a vector
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define the angle at a vertex of a triangle
def angle_at_vertex (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incenter_vector_sum_implies_right_angle (t : Triangle) :
  let I := incenter t
  let IA := vector I t.A
  let IB := vector I t.B
  let IC := vector I t.C
  add_vectors (scalar_mult 3 IA) (add_vectors (scalar_mult 4 IB) (scalar_mult 5 IC)) = (0, 0) →
  angle_at_vertex t t.C = 90 :=
sorry

end NUMINAMATH_CALUDE_incenter_vector_sum_implies_right_angle_l2272_227291


namespace NUMINAMATH_CALUDE_digit2021_is_one_l2272_227277

/-- The sequence of digits formed by concatenating natural numbers starting from 1 -/
def digitSequence : ℕ → ℕ :=
  sorry

/-- The 2021st digit in the sequence -/
def digit2021 : ℕ := digitSequence 2021

theorem digit2021_is_one : digit2021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit2021_is_one_l2272_227277


namespace NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l2272_227217

theorem integral_3x_plus_sin_x (x : Real) :
  ∫ x in (0)..(π/2), (3 * x + Real.sin x) = (3/8) * π^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_3x_plus_sin_x_l2272_227217


namespace NUMINAMATH_CALUDE_pepper_remaining_l2272_227227

theorem pepper_remaining (initial_amount used_amount : ℝ) 
  (h1 : initial_amount = 0.25)
  (h2 : used_amount = 0.16) : 
  initial_amount - used_amount = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_pepper_remaining_l2272_227227


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l2272_227284

def original_price : ℝ := 250
def first_discount : ℝ := 0.60
def second_discount : ℝ := 0.25

theorem final_price_after_discounts :
  (original_price * (1 - first_discount) * (1 - second_discount)) = 75 := by
sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l2272_227284


namespace NUMINAMATH_CALUDE_adqr_is_cyclic_l2272_227256

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point lies on a line segment between two other points -/
def point_on_segment (P Q R : Point) : Prop := sorry

/-- Checks if two line segments have equal length -/
def segments_equal (A B C D : Point) : Prop := sorry

/-- Checks if a quadrilateral is cyclic (can be inscribed in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Main theorem -/
theorem adqr_is_cyclic 
  (A B C D P Q R T : Point)
  (h_convex : is_convex ⟨A, B, C, D⟩)
  (h_equal1 : segments_equal A P P T)
  (h_equal2 : segments_equal P T T D)
  (h_equal3 : segments_equal Q B B C)
  (h_equal4 : segments_equal B C C R)
  (h_on_AB1 : point_on_segment A P B)
  (h_on_AB2 : point_on_segment A Q B)
  (h_on_CD1 : point_on_segment C R D)
  (h_on_CD2 : point_on_segment C T D)
  (h_bctp_cyclic : is_cyclic ⟨B, C, T, P⟩) :
  is_cyclic ⟨A, D, Q, R⟩ :=
sorry

end NUMINAMATH_CALUDE_adqr_is_cyclic_l2272_227256


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2272_227294

theorem functional_equation_solution (f : ℕ → ℝ) 
  (h : ∀ x y : ℕ, f (x + y) + f (x - y) = f (3 * x)) : 
  ∀ x : ℕ, f x = 0 := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2272_227294


namespace NUMINAMATH_CALUDE_sequence_properties_and_sum_l2272_227233

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem sequence_properties_and_sum (a b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence b →
  a 1 = 2 →
  (∃ r : ℝ, r = 2 ∧ ∀ n : ℕ, b (n + 1) = r * b n) →
  a 2 + b 3 = 7 →
  a 4 + b 5 = 21 →
  (∀ n : ℕ, c n = a n / b n) →
  (∀ n : ℕ, a n = n + 1) ∧
  (∀ n : ℕ, b n = 2^(n - 1)) ∧
  (∀ n : ℕ, S n = 6 - (n + 3) / 2^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_and_sum_l2272_227233


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l2272_227263

theorem arctan_sum_equals_pi_over_four :
  ∃ n : ℕ+, 
    Real.arctan (1/3) + Real.arctan (1/4) + Real.arctan (1/6) + Real.arctan (1/(n : ℝ)) = π/4 ∧
    n = 56 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_four_l2272_227263


namespace NUMINAMATH_CALUDE_expression_equals_one_l2272_227212

theorem expression_equals_one :
  |Real.sqrt 3 - 2| + (-1/2)⁻¹ + (2023 - Real.pi)^0 + 3 * Real.tan (30 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2272_227212


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l2272_227260

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_is_220 :
  rectangle_area 3025 10 = 220 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_220_l2272_227260


namespace NUMINAMATH_CALUDE_next_skipped_perfect_square_l2272_227224

theorem next_skipped_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, m^2 = n) ∧
  (∀ y : ℕ, y > x ∧ y < n → ¬∃ m : ℕ, m^2 = y) ∧
  (∃ m : ℕ, m^2 = x + 4 * Real.sqrt x + 4) :=
sorry

end NUMINAMATH_CALUDE_next_skipped_perfect_square_l2272_227224


namespace NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l2272_227228

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 116 / 85) 
  (h2 : Real.cos x + Real.cos y = 42 / 85) : 
  Real.tan x + Real.tan y = -232992832 / 5705296111 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l2272_227228


namespace NUMINAMATH_CALUDE_f_lipschitz_implies_m_bounded_l2272_227270

theorem f_lipschitz_implies_m_bounded (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ [-2, 2] → x₂ ∈ [-2, 2] →
    |((fun x => Real.exp (m * x) + x^4 - m * x) x₁) -
     ((fun x => Real.exp (m * x) + x^4 - m * x) x₂)| ≤ Real.exp 4 + 11) →
  m ∈ [-2, 2] := by
sorry

end NUMINAMATH_CALUDE_f_lipschitz_implies_m_bounded_l2272_227270


namespace NUMINAMATH_CALUDE_divisor_property_l2272_227276

theorem divisor_property (k : ℕ) : 
  (15 ^ k) ∣ 759325 → 3 ^ k - 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_property_l2272_227276


namespace NUMINAMATH_CALUDE_tons_to_kilograms_l2272_227292

-- Define the mass units
def ton : ℝ := 1000
def kilogram : ℝ := 1

-- State the theorem
theorem tons_to_kilograms : 24 * ton = 24000 * kilogram := by sorry

end NUMINAMATH_CALUDE_tons_to_kilograms_l2272_227292


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2272_227206

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x^2 + 4y^2 = 16 -/
def Ellipse (P : Point) : Prop :=
  P.x^2 + 4 * P.y^2 = 16

/-- Represents the distance between two points -/
def distance (P Q : Point) : ℝ :=
  ((P.x - Q.x)^2 + (P.y - Q.y)^2)^(1/2)

/-- Theorem: For a point P on the ellipse x^2 + 4y^2 = 16 with foci F1 and F2,
    if the distance from P to F1 is 7, then the distance from P to F2 is 1 -/
theorem ellipse_foci_distance (P F1 F2 : Point) :
  Ellipse P →
  distance P F1 = 7 →
  distance P F2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2272_227206


namespace NUMINAMATH_CALUDE_system_solution_l2272_227234

theorem system_solution (x y : ℝ) (dot star : ℝ) : 
  (2 * x + y = dot ∧ 2 * x - y = 12 ∧ x = 5 ∧ y = star) → 
  (dot = 8 ∧ star = -2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2272_227234


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l2272_227238

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (different : Line → Line → Prop)
variable (non_coincident : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) 
  (h1 : different m n) 
  (h2 : non_coincident α β) 
  (h3 : perpendicular m α) 
  (h4 : parallel m β) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l2272_227238


namespace NUMINAMATH_CALUDE_complex_product_QED_l2272_227280

theorem complex_product_QED (Q E D : ℂ) : 
  Q = 4 + 3*I ∧ E = 2*I ∧ D = 4 - 3*I → Q * E * D = 50*I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_QED_l2272_227280


namespace NUMINAMATH_CALUDE_intersection_S_T_l2272_227259

def S : Set ℝ := {x | x ≥ 1}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l2272_227259


namespace NUMINAMATH_CALUDE_complement_A_B_l2272_227281

def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

theorem complement_A_B : (A \ B) = {0, 2, 6, 10} := by sorry

end NUMINAMATH_CALUDE_complement_A_B_l2272_227281


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2272_227268

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := 2*x - 2
  let a₂ := 2*x + 2
  let a₃ := 4*x + 4
  (a₂ - a₁ = a₃ - a₂) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2272_227268


namespace NUMINAMATH_CALUDE_furniture_reimbursement_l2272_227222

/-- Calculates the reimbursement amount for an overcharged furniture purchase -/
theorem furniture_reimbursement
  (num_pieces : ℕ)
  (amount_paid : ℚ)
  (cost_per_piece : ℚ)
  (h1 : num_pieces = 150)
  (h2 : amount_paid = 20700)
  (h3 : cost_per_piece = 134) :
  amount_paid - (num_pieces : ℚ) * cost_per_piece = 600 := by
  sorry

end NUMINAMATH_CALUDE_furniture_reimbursement_l2272_227222


namespace NUMINAMATH_CALUDE_car_wheels_count_l2272_227279

theorem car_wheels_count (cars : ℕ) (motorcycles : ℕ) (total_wheels : ℕ) 
  (h1 : cars = 19)
  (h2 : motorcycles = 11)
  (h3 : total_wheels = 117)
  (h4 : ∀ m : ℕ, m ≤ motorcycles → 2 * m ≤ total_wheels) :
  ∃ (wheels_per_car : ℕ), wheels_per_car * cars + 2 * motorcycles = total_wheels ∧ wheels_per_car = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_wheels_count_l2272_227279


namespace NUMINAMATH_CALUDE_z_remainder_when_z_plus_3_div_9_is_integer_l2272_227275

theorem z_remainder_when_z_plus_3_div_9_is_integer (z : ℤ) :
  (∃ k : ℤ, (z + 3) / 9 = k) → z ≡ 6 [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_z_remainder_when_z_plus_3_div_9_is_integer_l2272_227275


namespace NUMINAMATH_CALUDE_range_of_piecewise_function_l2272_227243

/-- Given two linear functions f and g, and a piecewise function r,
    prove that the range of r is [a/2 + b, c + d] -/
theorem range_of_piecewise_function
  (a b c d : ℝ)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (r : ℝ → ℝ)
  (ha : a < 0)
  (hc : c > 0)
  (hf : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = a * x + b)
  (hg : ∀ x, 0 ≤ x ∧ x ≤ 1 → g x = c * x + d)
  (hr : ∀ x, 0 ≤ x ∧ x ≤ 1 → r x = if x ≤ 0.5 then f x else g x) :
  Set.range r = Set.Icc (a / 2 + b) (c + d) :=
sorry

end NUMINAMATH_CALUDE_range_of_piecewise_function_l2272_227243


namespace NUMINAMATH_CALUDE_function_property_l2272_227286

-- Define the function f and its property
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y - x * y

-- Define α
def α (f : ℝ → ℝ) : ℝ := f 0

-- Theorem statement
theorem function_property (f : ℝ → ℝ) (h : f_property f) :
  (f (α f) * f (-(α f)) = 0) ∧
  (α f = 0) ∧
  (∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l2272_227286


namespace NUMINAMATH_CALUDE_other_sales_percentage_l2272_227282

/-- Represents the sales distribution of the Dreamy Bookstore for April -/
structure SalesDistribution where
  notebooks : ℝ
  bookmarks : ℝ
  other : ℝ

/-- The sales distribution for the Dreamy Bookstore in April -/
def april_sales : SalesDistribution where
  notebooks := 45
  bookmarks := 25
  other := 100 - (45 + 25)

/-- Theorem stating that the percentage of sales that were neither notebooks nor bookmarks is 30% -/
theorem other_sales_percentage (s : SalesDistribution) 
  (h1 : s.notebooks = 45)
  (h2 : s.bookmarks = 25)
  (h3 : s.notebooks + s.bookmarks + s.other = 100) :
  s.other = 30 := by
  sorry

#eval april_sales.other

end NUMINAMATH_CALUDE_other_sales_percentage_l2272_227282


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l2272_227209

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + a

-- Define the theorem
theorem solution_set_implies_a_equals_one :
  (∃ (a : ℝ), ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) →
  (∃ (a : ℝ), a = 1 ∧ ∀ (x : ℝ), f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l2272_227209


namespace NUMINAMATH_CALUDE_intersection_probability_in_decagon_l2272_227296

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

end NUMINAMATH_CALUDE_intersection_probability_in_decagon_l2272_227296


namespace NUMINAMATH_CALUDE_gcf_of_60_180_150_l2272_227225

theorem gcf_of_60_180_150 : Nat.gcd 60 (Nat.gcd 180 150) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_60_180_150_l2272_227225


namespace NUMINAMATH_CALUDE_proportion_and_equation_imply_c_value_l2272_227242

theorem proportion_and_equation_imply_c_value 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 7*k) 
  (h2 : a - b + 3 = c - 2*b) : 
  c = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_and_equation_imply_c_value_l2272_227242


namespace NUMINAMATH_CALUDE_modified_fibonacci_sum_l2272_227244

-- Define the modified Fibonacci sequence
def F : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => F (n + 1) + F n

-- Define the sum of the series
noncomputable def S : ℝ := ∑' n, (F n : ℝ) / 5^n

-- Theorem statement
theorem modified_fibonacci_sum : S = 10 / 19 := by sorry

end NUMINAMATH_CALUDE_modified_fibonacci_sum_l2272_227244


namespace NUMINAMATH_CALUDE_min_value_expression_l2272_227239

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ 6 * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b = 6 * Real.sqrt 3 ↔ 
    a^2 = 1/6 ∧ b = -1/(2*a) ∧ c = 2*a) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2272_227239


namespace NUMINAMATH_CALUDE_odd_decreasing_properties_l2272_227249

-- Define an odd, decreasing function on ℝ
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y)

-- Theorem statement
theorem odd_decreasing_properties {f : ℝ → ℝ} {a b : ℝ} 
  (h_f : odd_decreasing_function f) (h_sum : a + b ≤ 0) : 
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_odd_decreasing_properties_l2272_227249


namespace NUMINAMATH_CALUDE_same_solution_implies_a_equals_one_l2272_227221

theorem same_solution_implies_a_equals_one :
  (∃ x : ℝ, 2 - a - x = 0 ∧ 2*x + 1 = 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_equals_one_l2272_227221


namespace NUMINAMATH_CALUDE_simplify_expressions_l2272_227252

theorem simplify_expressions :
  let exp1 := ((0.064 ^ (1/5)) ^ (-2.5)) ^ (2/3) - (3 * (3/8)) ^ (1/3) - π ^ 0
  let exp2 := (2 * Real.log 2 + Real.log 3) / (1 + (1/2) * Real.log 0.36 + (1/4) * Real.log 16)
  (exp1 = 0) ∧ (exp2 = (2 * Real.log 2 + Real.log 3) / Real.log 24) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2272_227252


namespace NUMINAMATH_CALUDE_infinite_zeros_or_nines_in_difference_l2272_227287

/-- Represents an infinite decimal fraction -/
def InfiniteDecimalFraction := ℕ → Fin 10

/-- Given a set of 11 infinite decimal fractions, there exist two fractions
    whose difference has either infinite zeros or infinite nines -/
theorem infinite_zeros_or_nines_in_difference 
  (fractions : Fin 11 → InfiniteDecimalFraction) :
  ∃ i j : Fin 11, i ≠ j ∧ 
    (∀ k : ℕ, (fractions i k - fractions j k) % 10 = 0 ∨
              (fractions i k - fractions j k) % 10 = 9) :=
sorry

end NUMINAMATH_CALUDE_infinite_zeros_or_nines_in_difference_l2272_227287


namespace NUMINAMATH_CALUDE_semicircle_in_right_triangle_l2272_227257

/-- Given a right-angled triangle with an inscribed semicircle, where:
    - The semicircle has radius r
    - The shorter edges of the triangle are tangent to the semicircle and have lengths a and b
    - The diameter of the semicircle lies on the hypotenuse of the triangle
    Then: 1/r = 1/a + 1/b -/
theorem semicircle_in_right_triangle (r a b : ℝ) 
    (hr : r > 0) (ha : a > 0) (hb : b > 0)
    (h_right_triangle : ∃ c, a^2 + b^2 = c^2)
    (h_tangent : ∃ p q : ℝ × ℝ, 
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (2*r)^2 ∧
      (p.1 - 0)^2 + (p.2 - 0)^2 = a^2 ∧
      (q.1 - 0)^2 + (q.2 - 0)^2 = b^2) :
  1/r = 1/a + 1/b := by
    sorry

end NUMINAMATH_CALUDE_semicircle_in_right_triangle_l2272_227257


namespace NUMINAMATH_CALUDE_cos_180_degrees_l2272_227258

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_180_degrees_l2272_227258


namespace NUMINAMATH_CALUDE_shaded_area_is_7pi_l2272_227229

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of circles in the problem -/
structure CircleConfiguration where
  smallCircles : List Circle
  largeCircle : Circle
  allIntersectAtTangency : Bool

/-- Calculates the area of the shaded region given a circle configuration -/
def shadedArea (config : CircleConfiguration) : ℝ :=
  sorry

/-- The main theorem stating the shaded area for the given configuration -/
theorem shaded_area_is_7pi (config : CircleConfiguration)
  (h1 : config.smallCircles.length = 13)
  (h2 : ∀ c ∈ config.smallCircles, c.radius = 1)
  (h3 : config.allIntersectAtTangency = true) :
  shadedArea config = 7 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_7pi_l2272_227229


namespace NUMINAMATH_CALUDE_least_number_divisibility_l2272_227261

theorem least_number_divisibility (x : ℕ) : 
  (x > 0) →
  (x / 5 = (x % 34) + 8) →
  (∀ y : ℕ, y > 0 → y / 5 = (y % 34) + 8 → y ≥ x) →
  x = 160 := by
sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l2272_227261


namespace NUMINAMATH_CALUDE_valid_arrangement_iff_even_l2272_227220

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

end NUMINAMATH_CALUDE_valid_arrangement_iff_even_l2272_227220


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l2272_227265

theorem similar_triangle_perimeter : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  ∃ (k : ℝ), k > 0 ∧ 
  (k * a = 18 ∨ k * b = 18) →
  k * (a + b + c) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l2272_227265


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2272_227298

theorem quadratic_rewrite (x : ℝ) :
  ∃ m : ℝ, 4 * x^2 - 16 * x - 448 = (x + m)^2 - 116 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2272_227298


namespace NUMINAMATH_CALUDE_nine_digit_integer_count_l2272_227200

/-- The number of digits in the integers we're counting -/
def num_digits : ℕ := 9

/-- The count of possible digits for the first position (1-9) -/
def first_digit_choices : ℕ := 9

/-- The count of possible digits for each remaining position (0-9) -/
def other_digit_choices : ℕ := 10

/-- The number of 9-digit positive integers that do not start with 0 -/
def count_9digit_integers : ℕ := first_digit_choices * (other_digit_choices ^ (num_digits - 1))

theorem nine_digit_integer_count :
  count_9digit_integers = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_integer_count_l2272_227200


namespace NUMINAMATH_CALUDE_landscape_ratio_is_8_to_1_l2272_227247

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playgroundArea : ℝ

/-- The ratio of breadth to length as a pair of integers -/
def BreadthLengthRatio := ℕ × ℕ

/-- Calculates the ratio of breadth to length -/
def calculateRatio (l : Landscape) : BreadthLengthRatio :=
  sorry

theorem landscape_ratio_is_8_to_1 (l : Landscape) 
  (h1 : ∃ n : ℝ, l.breadth = n * l.length)
  (h2 : l.playgroundArea = 3200)
  (h3 : l.length * l.breadth = 9 * l.playgroundArea)
  (h4 : l.breadth = 480) : 
  calculateRatio l = (8, 1) :=
sorry

end NUMINAMATH_CALUDE_landscape_ratio_is_8_to_1_l2272_227247


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l2272_227223

theorem x_squared_plus_y_squared (x y : ℚ) 
  (h : 2002 * (x - 1)^2 + |x - 12*y + 1| = 0) : 
  x^2 + y^2 = 37 / 36 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l2272_227223


namespace NUMINAMATH_CALUDE_original_dish_price_l2272_227205

theorem original_dish_price (price : ℝ) : 
  (price * 0.9 + price * 0.15 = price * 0.9 + price * 0.9 * 0.15 + 1.26) → 
  price = 84 :=
by sorry

end NUMINAMATH_CALUDE_original_dish_price_l2272_227205


namespace NUMINAMATH_CALUDE_eliminate_denominators_l2272_227253

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

end NUMINAMATH_CALUDE_eliminate_denominators_l2272_227253


namespace NUMINAMATH_CALUDE_c_investment_value_l2272_227299

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

end NUMINAMATH_CALUDE_c_investment_value_l2272_227299


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2272_227207

theorem quadratic_equation_roots : ∃ x₁ x₂ : ℝ, 
  (x₁ = -3 ∧ x₂ = -1) ∧ 
  (x₁^2 + 4*x₁ + 3 = 0) ∧ 
  (x₂^2 + 4*x₂ + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2272_227207


namespace NUMINAMATH_CALUDE_hyperbola_construction_uniqueness_l2272_227269

/-- A tangent line to a hyperbola at its vertex -/
structure Tangent where
  line : Line

/-- An asymptote of a hyperbola -/
structure Asymptote where
  line : Line

/-- Linear eccentricity of a hyperbola -/
def LinearEccentricity : Type := ℝ

/-- A hyperbola -/
structure Hyperbola where
  -- Define necessary components of a hyperbola

/-- Two hyperbolas are congruent if they have the same shape and size -/
def congruent (h1 h2 : Hyperbola) : Prop := sorry

/-- Two hyperbolas are parallel translations if one can be obtained from the other by a translation -/
def parallel_translation (h1 h2 : Hyperbola) (dir : Vec) : Prop := sorry

/-- Main theorem: Given a tangent, an asymptote, and linear eccentricity, 
    there exist exactly two congruent hyperbolas satisfying these conditions -/
theorem hyperbola_construction_uniqueness 
  (t : Tangent) (a₁ : Asymptote) (c : LinearEccentricity) :
  ∃! (h1 h2 : Hyperbola), 
    (∃ (dir : Vec), parallel_translation h1 h2 dir) ∧ 
    congruent h1 h2 ∧
    -- Additional conditions to ensure h1 and h2 satisfy t, a₁, and c
    sorry := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_construction_uniqueness_l2272_227269


namespace NUMINAMATH_CALUDE_savings_calculation_l2272_227266

/-- The amount saved per month in dollars -/
def monthly_savings : ℕ := 3000

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total amount saved after one year -/
def total_savings : ℕ := monthly_savings * months_in_year

theorem savings_calculation : total_savings = 36000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l2272_227266


namespace NUMINAMATH_CALUDE_eight_stairs_climb_ways_l2272_227231

-- Define the function for the number of ways to climb n stairs
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 1
  | m + 4 => climbStairs (m + 2) + climbStairs (m + 1)

-- Theorem stating that there are 4 ways to climb 8 stairs
theorem eight_stairs_climb_ways : climbStairs 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_eight_stairs_climb_ways_l2272_227231


namespace NUMINAMATH_CALUDE_revenue_change_l2272_227248

theorem revenue_change 
  (T : ℝ) -- Original tax
  (C : ℝ) -- Original consumption
  (tax_reduction : ℝ) -- Tax reduction percentage
  (consumption_increase : ℝ) -- Consumption increase percentage
  (h1 : tax_reduction = 0.20) -- 20% tax reduction
  (h2 : consumption_increase = 0.15) -- 15% consumption increase
  : 
  (1 - tax_reduction) * (1 + consumption_increase) * T * C - T * C = -0.08 * T * C :=
by sorry

end NUMINAMATH_CALUDE_revenue_change_l2272_227248


namespace NUMINAMATH_CALUDE_max_baseball_hits_percentage_l2272_227267

theorem max_baseball_hits_percentage (total_hits : ℕ) (home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 4)
  (h4 : doubles = 10) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 68 := by
  sorry

end NUMINAMATH_CALUDE_max_baseball_hits_percentage_l2272_227267


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2272_227241

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    prove that the eccentricity of the corresponding hyperbola is sqrt(5)/2 -/
theorem ellipse_hyperbola_eccentricity 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : (a^2 - b^2) / a^2 = 3/4) : 
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l2272_227241


namespace NUMINAMATH_CALUDE_volleyball_team_score_l2272_227283

/-- Volleyball team scoring problem -/
theorem volleyball_team_score (lizzie_score : ℕ) (team_total : ℕ) : 
  lizzie_score = 4 →
  team_total = 50 →
  17 = team_total - (lizzie_score + (lizzie_score + 3) + 2 * (lizzie_score + (lizzie_score + 3))) :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_score_l2272_227283


namespace NUMINAMATH_CALUDE_log_xy_value_l2272_227246

theorem log_xy_value (x y : ℝ) 
  (h1 : Real.log (x^2 * y^2) = 1) 
  (h2 : Real.log (x^3 * y) = 2) : 
  Real.log (x * y) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_log_xy_value_l2272_227246


namespace NUMINAMATH_CALUDE_circle_fraction_range_l2272_227245

theorem circle_fraction_range (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  -(Real.sqrt 3 / 3) ≤ y / (x + 2) ∧ y / (x + 2) ≤ Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_circle_fraction_range_l2272_227245


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2272_227289

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with ratio q
  a 1 = 2 →                     -- a_1 = 2
  (a 1 + a 2 + a 3 = 6) →       -- S_3 = 6
  (q = 1 ∨ q = -2) :=           -- q = 1 or q = -2
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2272_227289


namespace NUMINAMATH_CALUDE_bees_count_l2272_227213

theorem bees_count (first_day_count : ℕ) (second_day_count : ℕ) : 
  (second_day_count = 3 * first_day_count) → 
  (second_day_count = 432) → 
  (first_day_count = 144) := by
sorry

end NUMINAMATH_CALUDE_bees_count_l2272_227213


namespace NUMINAMATH_CALUDE_intersection_M_N_l2272_227273

-- Define set M
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 2*x + 8)}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = abs x + 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | -2 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2272_227273


namespace NUMINAMATH_CALUDE_base_6_sum_theorem_l2272_227204

/-- Represents a base-6 number with three digits --/
def Base6Number (a b c : Nat) : Nat :=
  a * 36 + b * 6 + c

/-- Checks if a number is a valid base-6 digit (1-5) --/
def IsValidBase6Digit (n : Nat) : Prop :=
  0 < n ∧ n < 6

/-- The main theorem --/
theorem base_6_sum_theorem (A B C : Nat) 
  (h1 : IsValidBase6Digit A)
  (h2 : IsValidBase6Digit B)
  (h3 : IsValidBase6Digit C)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : Base6Number A B C + Base6Number B C 0 = Base6Number A C A) :
  A + B + C = Base6Number 1 5 0 := by
  sorry

#check base_6_sum_theorem

end NUMINAMATH_CALUDE_base_6_sum_theorem_l2272_227204
