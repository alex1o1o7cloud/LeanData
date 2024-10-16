import Mathlib

namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3443_344366

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/8)
  (h2 : 3*x - 3*y = 3/8) : 
  x^2 - y^2 = 5/64 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3443_344366


namespace NUMINAMATH_CALUDE_no_intersection_point_l3443_344360

theorem no_intersection_point :
  ¬ ∃ (x y : ℝ), 
    (3 * x + 4 * y - 12 = 0) ∧ 
    (5 * x - 4 * y - 10 = 0) ∧ 
    (x = 3) ∧ 
    (y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_point_l3443_344360


namespace NUMINAMATH_CALUDE_molecular_weight_C6H8O7_moles_l3443_344342

/-- The molecular weight of a single molecule of C6H8O7 in g/mol -/
def molecular_weight_C6H8O7 : ℝ := 192.124

/-- The total molecular weight in grams -/
def total_weight : ℝ := 960

/-- Theorem stating that the molecular weight of a certain number of moles of C6H8O7 is equal to the total weight -/
theorem molecular_weight_C6H8O7_moles : 
  ∃ (n : ℝ), n * molecular_weight_C6H8O7 = total_weight :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_C6H8O7_moles_l3443_344342


namespace NUMINAMATH_CALUDE_susan_works_four_days_per_week_l3443_344326

/-- Represents Susan's work schedule and vacation details -/
structure WorkSchedule where
  hourlyRate : ℚ
  hoursPerDay : ℕ
  vacationDays : ℕ
  paidVacationDays : ℕ
  missedPay : ℚ

/-- Calculates the number of days Susan works per week -/
def daysWorkedPerWeek (schedule : WorkSchedule) : ℚ :=
  let totalVacationDays := 2 * 7
  let unpaidVacationDays := totalVacationDays - schedule.paidVacationDays
  unpaidVacationDays / 2

/-- Theorem stating that Susan works 4 days a week -/
theorem susan_works_four_days_per_week (schedule : WorkSchedule)
  (h1 : schedule.hourlyRate = 15)
  (h2 : schedule.hoursPerDay = 8)
  (h3 : schedule.vacationDays = 14)
  (h4 : schedule.paidVacationDays = 6)
  (h5 : schedule.missedPay = 480) :
  daysWorkedPerWeek schedule = 4 := by
  sorry


end NUMINAMATH_CALUDE_susan_works_four_days_per_week_l3443_344326


namespace NUMINAMATH_CALUDE_base4_calculation_l3443_344308

/-- Converts a base-4 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base-10 number to base-4 --/
def toBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem base4_calculation :
  let a := toBase10 [1, 3, 2]  -- 231₄
  let b := toBase10 [1, 2]     -- 21₄
  let c := toBase10 [2, 3]     -- 32₄
  let d := toBase10 [2]        -- 2₄
  toBase4 (a * b + c / d) = [0, 3, 1, 6] := by
  sorry

end NUMINAMATH_CALUDE_base4_calculation_l3443_344308


namespace NUMINAMATH_CALUDE_complement_A_union_B_l3443_344302

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem complement_A_union_B : 
  (Set.univ \ A) ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l3443_344302


namespace NUMINAMATH_CALUDE_min_moves_for_identical_contents_l3443_344347

/-- Represents a ball color -/
inductive BallColor
| White
| Black

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  black : Nat

/-- Represents a move: taking a ball from a box and either discarding it or transferring it -/
inductive Move
| Discard : BallColor → Move
| Transfer : BallColor → Move

/-- The initial state of the boxes -/
def initialState : (Box × Box) :=
  ({white := 4, black := 6}, {white := 0, black := 10})

/-- Predicate to check if two boxes have identical contents -/
def identicalContents (box1 box2 : Box) : Prop :=
  box1.white = box2.white ∧ box1.black = box2.black

/-- The minimum number of moves required to guarantee identical contents -/
def minMovesForIdenticalContents : Nat := 15

theorem min_moves_for_identical_contents :
  ∀ (sequence : List Move),
  (∃ (finalState : Box × Box),
    finalState.1.white + finalState.1.black + finalState.2.white + finalState.2.black ≤ 
      initialState.1.white + initialState.1.black + initialState.2.white + initialState.2.black ∧
    identicalContents finalState.1 finalState.2) →
  sequence.length ≥ minMovesForIdenticalContents :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_identical_contents_l3443_344347


namespace NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l3443_344328

/-- Represents the number of free ends after k iterations of extending segments -/
def num_free_ends (k : ℕ) : ℕ := 1 + 4 * k

/-- Theorem stating that there exists a number of iterations that results in 1001 free ends -/
theorem exists_k_for_1001_free_ends : ∃ k : ℕ, num_free_ends k = 1001 := by
  sorry

end NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l3443_344328


namespace NUMINAMATH_CALUDE_total_price_of_hats_l3443_344301

/-- Calculates the total price of hats given the conditions --/
theorem total_price_of_hats :
  let total_hats : ℕ := 85
  let green_hats : ℕ := 30
  let blue_hats : ℕ := total_hats - green_hats
  let price_green : ℕ := 7
  let price_blue : ℕ := 6
  (green_hats * price_green + blue_hats * price_blue) = 540 := by
  sorry

end NUMINAMATH_CALUDE_total_price_of_hats_l3443_344301


namespace NUMINAMATH_CALUDE_solve_equation_l3443_344390

theorem solve_equation : ∃ x : ℝ, (10 - x = 15) ∧ (x = -5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3443_344390


namespace NUMINAMATH_CALUDE_product_equals_one_l3443_344331

theorem product_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 4)
  (eq2 : y + 1/z = 1)
  (eq3 : z + 1/x = 7/3) :
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l3443_344331


namespace NUMINAMATH_CALUDE_cubic_factorization_l3443_344341

theorem cubic_factorization (x : ℝ) : x^3 - 2*x^2 + x = x*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3443_344341


namespace NUMINAMATH_CALUDE_equation_solution_l3443_344333

theorem equation_solution :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ x = 2 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3443_344333


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3443_344399

theorem smallest_solution_abs_equation :
  ∃ x : ℝ, x * |x| = 3 * x - 2 ∧
  ∀ y : ℝ, y * |y| = 3 * y - 2 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l3443_344399


namespace NUMINAMATH_CALUDE_liam_money_left_l3443_344373

/-- Calculates the amount of money Liam has left after paying his bills -/
def money_left_after_bills (
  savings_rate : ℕ
) (
  savings_duration_months : ℕ
) (
  bills_cost : ℕ
) : ℕ :=
  savings_rate * savings_duration_months - bills_cost

/-- Proves that Liam will have $8,500 left after paying his bills -/
theorem liam_money_left :
  money_left_after_bills 500 24 3500 = 8500 := by
  sorry

#eval money_left_after_bills 500 24 3500

end NUMINAMATH_CALUDE_liam_money_left_l3443_344373


namespace NUMINAMATH_CALUDE_problem_statement_l3443_344353

theorem problem_statement (a b c d e : ℕ+) 
  (h1 : a * b + a + b = 624)
  (h2 : b * c + b + c = 234)
  (h3 : c * d + c + d = 156)
  (h4 : d * e + d + e = 80)
  (h5 : a * b * c * d * e = 3628800) : -- 3628800 is 10!
  a - e = 22 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3443_344353


namespace NUMINAMATH_CALUDE_pictures_per_album_l3443_344359

/-- Given the number of pictures uploaded from phone and camera, and the number of albums,
    prove that the number of pictures in each album is correct. -/
theorem pictures_per_album
  (phone_pics : ℕ)
  (camera_pics : ℕ)
  (num_albums : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : num_albums = 5)
  (h4 : num_albums > 0) :
  (phone_pics + camera_pics) / num_albums = 8 := by
sorry

#eval (35 + 5) / 5  -- Expected output: 8

end NUMINAMATH_CALUDE_pictures_per_album_l3443_344359


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l3443_344391

theorem cos_alpha_plus_pi_sixth (α : Real) (h : Real.sin (α - π/3) = 1/3) : 
  Real.cos (α + π/6) = -1/3 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_sixth_l3443_344391


namespace NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l3443_344398

theorem walk_a_thon_miles_difference 
  (last_year_rate : ℝ) 
  (this_year_rate : ℝ) 
  (last_year_amount : ℝ) : 
  last_year_rate = 4 →
  this_year_rate = 2.75 →
  last_year_amount = 44 →
  (last_year_amount / this_year_rate) - (last_year_amount / last_year_rate) = 5 :=
by sorry

end NUMINAMATH_CALUDE_walk_a_thon_miles_difference_l3443_344398


namespace NUMINAMATH_CALUDE_isosceles_triangle_34_perimeter_l3443_344335

/-- An isosceles triangle with sides 3 and 4 -/
structure IsoscelesTriangle34 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : (side1 = side2 ∧ side3 = 3) ∨ (side1 = side3 ∧ side2 = 3)
  has_side_4 : side1 = 4 ∨ side2 = 4 ∨ side3 = 4

/-- The perimeter of a triangle -/
def perimeter (t : IsoscelesTriangle34) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem: The perimeter of an isosceles triangle with sides 3 and 4 is either 10 or 11 -/
theorem isosceles_triangle_34_perimeter (t : IsoscelesTriangle34) : 
  perimeter t = 10 ∨ perimeter t = 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_34_perimeter_l3443_344335


namespace NUMINAMATH_CALUDE_baseball_soccer_difference_l3443_344310

def total_balls : ℕ := 145
def soccer_balls : ℕ := 20
def volleyball_balls : ℕ := 30

def basketball_balls : ℕ := soccer_balls + 5
def tennis_balls : ℕ := 2 * soccer_balls

def baseball_balls : ℕ := total_balls - (soccer_balls + basketball_balls + tennis_balls + volleyball_balls)

theorem baseball_soccer_difference :
  baseball_balls - soccer_balls = 10 :=
by sorry

end NUMINAMATH_CALUDE_baseball_soccer_difference_l3443_344310


namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3443_344386

/-- Given a cube with edge length 5 and a square-based pyramid with base edge length 10,
    prove that the height of the pyramid is 3.75 when their volumes are equal. -/
theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) : 
  cube_edge = 5 →
  pyramid_base = 10 →
  (cube_edge ^ 3) = (1 / 3) * (pyramid_base ^ 2) * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3443_344386


namespace NUMINAMATH_CALUDE_marble_box_count_l3443_344383

theorem marble_box_count (blue : ℕ) (red : ℕ) : 
  red = blue + 12 →
  (blue : ℚ) / (blue + red : ℚ) = 1 / 4 →
  blue + red = 24 :=
by sorry

end NUMINAMATH_CALUDE_marble_box_count_l3443_344383


namespace NUMINAMATH_CALUDE_product_inequality_l3443_344365

theorem product_inequality (a b c : ℝ) (d : ℝ) 
  (sum_zero : a + b + c = 0)
  (d_def : d = max (|a|) (max (|b|) (|c|))) :
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3443_344365


namespace NUMINAMATH_CALUDE_total_pencils_is_52_l3443_344346

/-- The number of pencils in a pack -/
def pencils_per_pack : ℕ := 12

/-- The number of packs Jimin has -/
def jimin_packs : ℕ := 2

/-- The number of individual pencils Jimin has -/
def jimin_individual : ℕ := 7

/-- The number of packs Yuna has -/
def yuna_packs : ℕ := 1

/-- The number of individual pencils Yuna has -/
def yuna_individual : ℕ := 9

/-- The total number of pencils Jimin and Yuna have -/
def total_pencils : ℕ := 
  jimin_packs * pencils_per_pack + jimin_individual +
  yuna_packs * pencils_per_pack + yuna_individual

theorem total_pencils_is_52 : total_pencils = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_52_l3443_344346


namespace NUMINAMATH_CALUDE_total_surveyed_is_185_l3443_344376

/-- Represents the total number of students surveyed in a stratified sampling method -/
def total_surveyed (grade10_total : ℕ) (grade11_total : ℕ) (grade12_total : ℕ) (grade12_surveyed : ℕ) : ℕ :=
  let grade10_surveyed := (grade10_total * grade12_surveyed) / grade12_total
  let grade11_surveyed := (grade11_total * grade12_surveyed) / grade12_total
  grade10_surveyed + grade11_surveyed + grade12_surveyed

/-- Theorem stating that the total number of students surveyed is 185 given the problem conditions -/
theorem total_surveyed_is_185 :
  total_surveyed 1000 1200 1500 75 = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_surveyed_is_185_l3443_344376


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3443_344372

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop := m * y^2 - x^2 = 1

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2 / 5 + x^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, hyperbola m x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
   -- The foci of the hyperbola and ellipse are the same
   (x₁ = x₂ ∧ y₁ = y₂)) →
  (∀ x y : ℝ, hyperbola m x y → asymptotes x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3443_344372


namespace NUMINAMATH_CALUDE_inequality_proof_l3443_344332

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) : 
  (π/2) + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3443_344332


namespace NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3443_344355

theorem largest_consecutive_even_integer : ∃ n : ℕ,
  (n - 8) + (n - 6) + (n - 4) + (n - 2) + n = 2 * (25 * 26 / 2) ∧
  n % 2 = 0 ∧
  n = 134 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3443_344355


namespace NUMINAMATH_CALUDE_prize_plan_optimal_l3443_344318

/-- Represents the prices and quantities of prizes A and B -/
structure PrizePlan where
  priceA : ℕ
  priceB : ℕ
  quantityA : ℕ
  quantityB : ℕ

/-- Conditions for the prize plan -/
def validPrizePlan (p : PrizePlan) : Prop :=
  3 * p.priceA + 2 * p.priceB = 130 ∧
  5 * p.priceA + 4 * p.priceB = 230 ∧
  p.quantityA + p.quantityB = 20 ∧
  p.quantityA ≥ 2 * p.quantityB

/-- Total cost of the prize plan -/
def totalCost (p : PrizePlan) : ℕ :=
  p.priceA * p.quantityA + p.priceB * p.quantityB

/-- The theorem to be proved -/
theorem prize_plan_optimal (p : PrizePlan) (h : validPrizePlan p) :
  p.priceA = 30 ∧ p.priceB = 20 ∧ p.quantityA = 14 ∧ p.quantityB = 6 ∧ totalCost p = 560 := by
  sorry

end NUMINAMATH_CALUDE_prize_plan_optimal_l3443_344318


namespace NUMINAMATH_CALUDE_max_length_sequence_l3443_344315

def sequence_term (n : ℕ) (x : ℕ) : ℤ :=
  match n with
  | 0 => 5000
  | 1 => x
  | n + 2 => sequence_term n x - sequence_term (n + 1) x

def is_positive (n : ℤ) : Prop := n > 0

theorem max_length_sequence (x : ℕ) : 
  (∀ n : ℕ, n < 11 → is_positive (sequence_term n x)) ∧ 
  ¬(is_positive (sequence_term 11 x)) ↔ 
  x = 3089 :=
sorry

end NUMINAMATH_CALUDE_max_length_sequence_l3443_344315


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l3443_344304

/-- Given that the average of a and b is 40, and the average of b and c is 60,
    prove that the difference between c and a is 40. -/
theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 60) :
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l3443_344304


namespace NUMINAMATH_CALUDE_reflection_distance_A_l3443_344370

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_distance (x y : ℝ) : ℝ :=
  2 * |y|

/-- Theorem: The length of the segment from A(2, 4) to its reflection A' over the x-axis is 8 --/
theorem reflection_distance_A : reflection_distance 2 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_reflection_distance_A_l3443_344370


namespace NUMINAMATH_CALUDE_tons_to_pounds_l3443_344337

-- Define the basic units
def ounces_per_pound : ℕ := 16

-- Define the packet weight in ounces
def packet_weight_ounces : ℕ := 16 * ounces_per_pound + 4

-- Define the number of packets
def num_packets : ℕ := 1840

-- Define the capacity of the gunny bag in tons
def bag_capacity_tons : ℕ := 13

-- Define the weight of all packets in ounces
def total_weight_ounces : ℕ := num_packets * packet_weight_ounces

-- Define the relation between tons and pounds
def pounds_per_ton : ℕ := 2000

-- Theorem statement
theorem tons_to_pounds : 
  total_weight_ounces = bag_capacity_tons * pounds_per_ton * ounces_per_pound :=
sorry

end NUMINAMATH_CALUDE_tons_to_pounds_l3443_344337


namespace NUMINAMATH_CALUDE_sum_coordinates_of_B_l3443_344311

/-- Given that M(6,8) is the midpoint of AB and A has coordinates (10,8), 
    prove that the sum of the coordinates of B is 10. -/
theorem sum_coordinates_of_B (A B M : ℝ × ℝ) : 
  M = (6, 8) → 
  A = (10, 8) → 
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  B.1 + B.2 = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_B_l3443_344311


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3443_344321

/-- Given two points A and B in a plane, and vectors a and b, 
    prove that if a is perpendicular to b, then m = 1. -/
theorem perpendicular_vectors_m_value 
  (A B : ℝ × ℝ) 
  (h_A : A = (0, 2)) 
  (h_B : B = (3, -1)) 
  (a b : ℝ × ℝ) 
  (h_a : a = B - A) 
  (h_b : b = (1, m)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3443_344321


namespace NUMINAMATH_CALUDE_ellipse_properties_l3443_344363

/-- Given an ellipse W with equation x²/4 + y²/b² = 1 (b > 0) and a focus at (√3, 0),
    prove its equation, eccentricity, and a geometric property. -/
theorem ellipse_properties (b : ℝ) (hb : b > 0) :
  let W : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 / b^2 = 1}
  let focus : ℝ × ℝ := (Real.sqrt 3, 0)
  ∃ (e : ℝ),
    (∀ p, p ∈ W ↔ p.1^2 / 4 + p.2^2 = 1) ∧  -- Equation of W
    (e = Real.sqrt 3 / 2) ∧  -- Eccentricity
    (∀ (M : ℝ × ℝ) (hM : M ∈ W) (hMx : M.1 ≠ 0),
      let A : ℝ × ℝ := (0, 1)
      let B : ℝ × ℝ := (0, -1)
      let N : ℝ × ℝ := (0, M.2)
      let E : ℝ × ℝ := (M.1 / 2, M.2)
      let C : ℝ × ℝ := (M.1 / (1 - M.2), -1)
      let G : ℝ × ℝ := ((C.1 + B.1) / 2, (C.2 + B.2) / 2)
      let O : ℝ × ℝ := (0, 0)
      (E - O) • (G - E) = 0) -- ∠OEG = 90°
:= by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3443_344363


namespace NUMINAMATH_CALUDE_dispersion_measures_l3443_344356

-- Define a sample as a list of real numbers
def Sample := List Real

-- Define the statistics
def standardDeviation (s : Sample) : Real := sorry
def median (s : Sample) : Real := sorry
def range (s : Sample) : Real := sorry
def mean (s : Sample) : Real := sorry

-- Define a measure of dispersion
def measuresDispersion (f : Sample → Real) : Prop := sorry

-- Theorem stating which statistics measure dispersion
theorem dispersion_measures (s : Sample) :
  (measuresDispersion (standardDeviation)) ∧
  (measuresDispersion (range)) ∧
  (¬ measuresDispersion (median)) ∧
  (¬ measuresDispersion (mean)) :=
sorry

end NUMINAMATH_CALUDE_dispersion_measures_l3443_344356


namespace NUMINAMATH_CALUDE_largest_angle_of_triangle_l3443_344351

theorem largest_angle_of_triangle (a b c : ℝ) : 
  a = 70 → b = 80 → c = 180 - a - b → a + b + c = 180 → max a (max b c) = 80 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_of_triangle_l3443_344351


namespace NUMINAMATH_CALUDE_no_intersection_l3443_344384

theorem no_intersection : ¬∃ x : ℝ, |3 * x + 6| = -2 * |2 * x - 1| := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_l3443_344384


namespace NUMINAMATH_CALUDE_theater_eye_color_ratio_l3443_344378

theorem theater_eye_color_ratio :
  let total_people : ℕ := 100
  let blue_eyes : ℕ := 19
  let brown_eyes : ℕ := total_people / 2
  let green_eyes : ℕ := 6
  let black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)
  (black_eyes : ℚ) / total_people = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_theater_eye_color_ratio_l3443_344378


namespace NUMINAMATH_CALUDE_general_equation_l3443_344357

theorem general_equation (n : ℕ+) :
  (n + 1 : ℚ) / ((n + 1)^2 - 1) - 1 / (n * (n + 1) * (n + 2)) = 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_general_equation_l3443_344357


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3443_344354

theorem complex_fraction_simplification :
  (3 + 4 * Complex.I) / (1 - 2 * Complex.I) = -1 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3443_344354


namespace NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3443_344307

theorem algebraic_expression_symmetry (a b c : ℝ) :
  (a * (-5)^4 + b * (-5)^2 + c = 3) →
  (a * 5^4 + b * 5^2 + c = 3) := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_symmetry_l3443_344307


namespace NUMINAMATH_CALUDE_central_angle_unchanged_l3443_344309

theorem central_angle_unchanged (r₁ r₂ arc_length₁ arc_length₂ angle₁ angle₂ : ℝ) :
  r₁ > 0 →
  r₂ = 2 * r₁ →
  arc_length₂ = 2 * arc_length₁ →
  angle₁ = arc_length₁ / r₁ →
  angle₂ = arc_length₂ / r₂ →
  angle₂ = angle₁ :=
by sorry

end NUMINAMATH_CALUDE_central_angle_unchanged_l3443_344309


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l3443_344345

/-- A game played on a complete graph -/
structure Graph :=
  (n : ℕ)  -- number of vertices
  (is_complete : n > 0)

/-- A player in the game -/
inductive Player
| Alice
| Bob

/-- A move in the game -/
structure Move :=
  (player : Player)
  (edges_oriented : ℕ)

/-- The game state -/
structure GameState :=
  (graph : Graph)
  (moves : List Move)
  (remaining_edges : ℕ)

/-- Alice's strategy -/
def alice_strategy (state : GameState) : Move :=
  { player := Player.Alice, edges_oriented := 1 }

/-- Bob's strategy -/
def bob_strategy (state : GameState) (m : ℕ) : Move :=
  { player := Player.Bob, edges_oriented := m }

/-- The winning condition for Alice -/
def alice_wins (final_state : GameState) : Prop :=
  ∃ (cycle : List ℕ), cycle.length > 0 ∧ cycle.Nodup

/-- The main theorem -/
theorem alice_has_winning_strategy :
  ∀ (g : Graph),
    g.n = 2014 →
    ∀ (bob_moves : GameState → ℕ),
      (∀ (state : GameState), 1 ≤ bob_moves state ∧ bob_moves state ≤ 1000) →
      ∃ (final_state : GameState),
        final_state.graph = g ∧
        final_state.remaining_edges = 0 ∧
        alice_wins final_state :=
  sorry

end NUMINAMATH_CALUDE_alice_has_winning_strategy_l3443_344345


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_product_l3443_344339

/-- The coefficient of x^3 in the product of two specific polynomials -/
theorem coefficient_x_cubed_in_product : ∃ (p q : Polynomial ℤ),
  p = 3 * X^3 + 2 * X^2 + 4 * X + 5 ∧
  q = 4 * X^3 + 6 * X^2 + 5 * X + 2 ∧
  (p * q).coeff 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_product_l3443_344339


namespace NUMINAMATH_CALUDE_monic_quadratic_root_l3443_344348

theorem monic_quadratic_root (x : ℂ) :
  let p : ℂ → ℂ := λ x => x^2 + 6*x + 12
  p (-3 - Complex.I * Real.sqrt 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_root_l3443_344348


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l3443_344393

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / z = 5 / 3)
  (hz : z / x = 1 / 6) :
  w / y = 9 / 1 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l3443_344393


namespace NUMINAMATH_CALUDE_remainder_theorem_l3443_344389

theorem remainder_theorem : (43^43 + 43) % 44 = 42 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3443_344389


namespace NUMINAMATH_CALUDE_triangle_side_length_l3443_344395

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let cos_C := (AB^2 + AC^2 - BC^2) / (2 * AB * AC)
  AB = Real.sqrt 5 ∧ AC = 5 ∧ cos_C = 9/10 →
  BC = 4 ∨ BC = 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3443_344395


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l3443_344324

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 12*a

-- Define a function to check if a number is an integer
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Define a function to count the number of real a values that satisfy the condition
def count_a_values : ℕ := sorry

-- Theorem statement
theorem quadratic_integer_roots :
  count_a_values = 15 := by sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l3443_344324


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l3443_344375

theorem prime_pairs_dividing_sum_of_powers (p q : ℕ) : 
  Prime p → Prime q → (p * q) ∣ (2^p + 2^q) → 
  ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l3443_344375


namespace NUMINAMATH_CALUDE_quincy_peter_picture_difference_l3443_344368

theorem quincy_peter_picture_difference :
  ∀ (peter_pictures randy_pictures quincy_pictures total_pictures : ℕ),
    peter_pictures = 8 →
    randy_pictures = 5 →
    total_pictures = 41 →
    total_pictures = peter_pictures + randy_pictures + quincy_pictures →
    quincy_pictures - peter_pictures = 20 := by
  sorry

end NUMINAMATH_CALUDE_quincy_peter_picture_difference_l3443_344368


namespace NUMINAMATH_CALUDE_min_equal_triangles_is_18_l3443_344369

/-- A non-convex hexagon representing a chessboard with one corner square cut out. -/
structure CutoutChessboard :=
  (area : ℝ)
  (is_non_convex : Bool)

/-- The minimum number of equal triangles into which the cutout chessboard can be divided. -/
def min_equal_triangles (board : CutoutChessboard) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of equal triangles is 18 for a cutout chessboard with area 63. -/
theorem min_equal_triangles_is_18 (board : CutoutChessboard) 
  (h1 : board.area = 63)
  (h2 : board.is_non_convex = true) : 
  min_equal_triangles board = 18 :=
sorry

end NUMINAMATH_CALUDE_min_equal_triangles_is_18_l3443_344369


namespace NUMINAMATH_CALUDE_power_three_seventeen_mod_seven_l3443_344362

theorem power_three_seventeen_mod_seven :
  (3 : ℤ) ^ 17 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_power_three_seventeen_mod_seven_l3443_344362


namespace NUMINAMATH_CALUDE_unique_divisibility_l3443_344350

def is_divisible_by_only_one_small_prime (n : ℕ) : Prop :=
  ∃! p, p < 10 ∧ Nat.Prime p ∧ n % p = 0

def number_form (B : ℕ) : ℕ := 404300 + B

theorem unique_divisibility :
  ∃! B, B < 10 ∧ is_divisible_by_only_one_small_prime (number_form B) ∧ number_form B = 404304 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisibility_l3443_344350


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3443_344392

theorem solve_exponential_equation (y : ℝ) : 5^(2*y) = Real.sqrt 125 → y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3443_344392


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_implies_m_bound_l3443_344385

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∨ (∀ x y : ℝ, x ≤ y → f y ≤ f x)

/-- The main theorem: If f(x) = x³ + x² + mx + 1 is monotonic on ℝ, then m ≥ 1/3. -/
theorem monotonic_cubic_function_implies_m_bound (m : ℝ) :
  Monotonic (fun x : ℝ => x^3 + x^2 + m*x + 1) → m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_implies_m_bound_l3443_344385


namespace NUMINAMATH_CALUDE_initials_count_l3443_344320

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}
def vowels : Finset Char := {'A', 'E', 'I'}

def valid_initials (s : Finset (Char × Char × Char)) : Prop :=
  ∀ (a b c : Char), (a, b, c) ∈ s → 
    (a ∈ alphabet ∧ b ∈ alphabet ∧ c ∈ alphabet) ∧
    (a ∈ vowels ∨ b ∈ vowels ∨ c ∈ vowels)

theorem initials_count :
  ∃ (s : Finset (Char × Char × Char)), valid_initials s ∧ Finset.card s = 657 :=
sorry

end NUMINAMATH_CALUDE_initials_count_l3443_344320


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3443_344394

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧  -- nine-digit number
  (∃ (digits : List ℕ), 
    digits.length = 9 ∧
    digits.count 3 = 8 ∧
    digits.count 0 = 1 ∧
    digits.foldl (λ acc d => acc * 10 + d) 0 = n)

def leaves_remainder_one (n : ℕ) : Prop :=
  n % 4 = 1

theorem count_valid_numbers : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_number n ∧ leaves_remainder_one n) ∧
    S.card = 7 ∧
    (∀ n, is_valid_number n ∧ leaves_remainder_one n → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3443_344394


namespace NUMINAMATH_CALUDE_division_problem_l3443_344319

theorem division_problem (n : ℤ) : 
  (n / 6 = 124 ∧ n % 6 = 4) → ((n + 24) / 8 : ℚ) = 96.5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3443_344319


namespace NUMINAMATH_CALUDE_boat_speed_calculation_l3443_344380

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 15

/-- The speed of the stream -/
def stream_speed : ℝ := 3

/-- The time taken to travel downstream -/
def downstream_time : ℝ := 1

/-- The time taken to travel upstream -/
def upstream_time : ℝ := 1.5

/-- The distance traveled (same for both directions) -/
def distance : ℝ := boat_speed + stream_speed

theorem boat_speed_calculation :
  (distance = (boat_speed + stream_speed) * downstream_time) ∧
  (distance = (boat_speed - stream_speed) * upstream_time) →
  boat_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_calculation_l3443_344380


namespace NUMINAMATH_CALUDE_least_n_with_k_ge_10_M_mod_500_l3443_344325

/-- Sum of digits in base 6 representation -/
def h (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 10 representation of h(n) -/
def j (n : ℕ) : ℕ := sorry

/-- Sum of squares of digits in base 12 representation of j(n) -/
def k (n : ℕ) : ℕ := sorry

/-- The least value of n such that k(n) ≥ 10 -/
def M : ℕ := sorry

theorem least_n_with_k_ge_10 : M = 31 := by sorry

theorem M_mod_500 : M % 500 = 31 := by sorry

end NUMINAMATH_CALUDE_least_n_with_k_ge_10_M_mod_500_l3443_344325


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l3443_344361

theorem sqrt_sum_squares_eq_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + a*c + b*c = 0 ∧ a + b + c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l3443_344361


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3443_344377

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 2)
  (h3 : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3443_344377


namespace NUMINAMATH_CALUDE_ratio_calculation_l3443_344338

theorem ratio_calculation (X Y Z : ℚ) (h : X / Y = 3 / 2 ∧ Y / Z = 1 / 3) :
  (4 * X + 3 * Y) / (5 * Z - 2 * X) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3443_344338


namespace NUMINAMATH_CALUDE_N_is_perfect_square_l3443_344343

/-- Constructs the number N with n ones and n+1 twos, ending with 5 -/
def constructN (n : ℕ) : ℕ :=
  (10^(2*n+2) + 10^(n+2) + 25) / 9

/-- Theorem stating that N is a perfect square for any natural number n -/
theorem N_is_perfect_square (n : ℕ) : ∃ m : ℕ, (constructN n) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_N_is_perfect_square_l3443_344343


namespace NUMINAMATH_CALUDE_always_possible_largest_to_smallest_exists_impossible_smallest_to_largest_l3443_344374

-- Define the grid
def Grid := Fin 10 → Fin 10 → Bool

-- Define ship sizes
inductive ShipSize
| one
| two
| three
| four

-- Define the list of ships to be placed
def ships : List ShipSize :=
  [ShipSize.four] ++ List.replicate 2 ShipSize.three ++
  List.replicate 3 ShipSize.two ++ List.replicate 4 ShipSize.one

-- Define a valid placement
def isValidPlacement (g : Grid) (s : ShipSize) (x y : Fin 10) (horizontal : Bool) : Prop :=
  sorry

-- Define the theorem for part a
theorem always_possible_largest_to_smallest :
  ∀ (g : Grid),
  ∃ (g' : Grid),
    (∀ s ∈ ships, ∃ x y h, isValidPlacement g' s x y h) ∧
    (∀ x y, g' x y → g x y) :=
  sorry

-- Define the theorem for part b
theorem exists_impossible_smallest_to_largest :
  ∃ (g : Grid),
    (∀ s ∈ (ships.reverse.take (ships.length - 1)),
      ∃ x y h, isValidPlacement g s x y h) ∧
    (∀ x y h, ¬isValidPlacement g ShipSize.four x y h) :=
  sorry

end NUMINAMATH_CALUDE_always_possible_largest_to_smallest_exists_impossible_smallest_to_largest_l3443_344374


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_one_l3443_344344

theorem points_three_units_from_negative_one : 
  ∀ x : ℝ, abs (x - (-1)) = 3 ↔ x = 2 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_one_l3443_344344


namespace NUMINAMATH_CALUDE_science_fiction_total_pages_l3443_344382

/-- The number of books in the science fiction section -/
def num_books : ℕ := 8

/-- The number of pages in each book -/
def pages_per_book : ℕ := 478

/-- The total number of pages in the science fiction section -/
def total_pages : ℕ := num_books * pages_per_book

theorem science_fiction_total_pages :
  total_pages = 3824 := by sorry

end NUMINAMATH_CALUDE_science_fiction_total_pages_l3443_344382


namespace NUMINAMATH_CALUDE_tea_milk_problem_l3443_344379

/-- Represents the amount of liquid in a mug -/
structure Mug where
  tea : ℚ
  milk : ℚ

/-- Calculates the fraction of milk in a mug -/
def milkFraction (m : Mug) : ℚ :=
  m.milk / (m.tea + m.milk)

theorem tea_milk_problem : 
  let mug1_initial := Mug.mk 5 0
  let mug2_initial := Mug.mk 0 3
  let mug1_after_first_transfer := Mug.mk (mug1_initial.tea - 2) 0
  let mug2_after_first_transfer := Mug.mk 2 3
  let tea_fraction_in_mug2 := mug2_after_first_transfer.tea / 
    (mug2_after_first_transfer.tea + mug2_after_first_transfer.milk)
  let milk_fraction_in_mug2 := mug2_after_first_transfer.milk / 
    (mug2_after_first_transfer.tea + mug2_after_first_transfer.milk)
  let tea_returned := 3 * tea_fraction_in_mug2
  let milk_returned := 3 * milk_fraction_in_mug2
  let mug1_final := Mug.mk (mug1_after_first_transfer.tea + tea_returned) milk_returned
  milkFraction mug1_final = 3/10 := by
sorry

end NUMINAMATH_CALUDE_tea_milk_problem_l3443_344379


namespace NUMINAMATH_CALUDE_third_circle_radius_value_l3443_344352

/-- A sequence of six circles tangent to each other and to two parallel lines -/
structure TangentCircles where
  radii : Fin 6 → ℝ
  smallest_radius : radii 0 = 10
  largest_radius : radii 5 = 20
  tangent : ∀ i : Fin 5, radii i < radii (i + 1)

/-- The radius of the third circle from the smallest in the sequence -/
def third_circle_radius (tc : TangentCircles) : ℝ := tc.radii 2

/-- The theorem stating that the radius of the third circle is 10 · ⁵√4 -/
theorem third_circle_radius_value (tc : TangentCircles) :
  third_circle_radius tc = 10 * (4 : ℝ) ^ (1/5) :=
sorry

end NUMINAMATH_CALUDE_third_circle_radius_value_l3443_344352


namespace NUMINAMATH_CALUDE_g_theorem_l3443_344329

-- Define the function g
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y

-- Theorem statement
theorem g_theorem (g : ℝ → ℝ) (h : g_property g) :
  ∃ (a b : ℝ), (∀ x : ℝ, g x = a ∨ g x = b) ∧
  g 2 = a ∨ g 2 = b ∧
  a + b = 14/3 :=
sorry

end NUMINAMATH_CALUDE_g_theorem_l3443_344329


namespace NUMINAMATH_CALUDE_equation_solution_l3443_344349

theorem equation_solution (x : ℚ) : 
  x ≠ 2/3 →
  ((7*x + 3) / (3*x^2 + 7*x - 6) = 3*x / (3*x - 2)) ↔ (x = 1/3 ∨ x = -3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3443_344349


namespace NUMINAMATH_CALUDE_negation_of_exp_inequality_l3443_344358

theorem negation_of_exp_inequality :
  (¬ ∀ x : ℝ, Real.exp x > x) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exp_inequality_l3443_344358


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3443_344323

theorem inequality_system_solution (p : ℝ) :
  (19 * p < 10 ∧ p > 1/2) ↔ (1/2 < p ∧ p < 10/19) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3443_344323


namespace NUMINAMATH_CALUDE_gain_percent_for_80_and_58_l3443_344305

/-- Calculates the gain percent given the number of articles at cost price and selling price that are equal in total value -/
def gainPercent (costArticles sellingArticles : ℕ) : ℚ :=
  let ratio : ℚ := costArticles / sellingArticles
  (ratio - 1) / ratio * 100

theorem gain_percent_for_80_and_58 :
  gainPercent 80 58 = 11 / 29 * 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_for_80_and_58_l3443_344305


namespace NUMINAMATH_CALUDE_coefficient_of_y_l3443_344317

theorem coefficient_of_y (x y a : ℝ) : 
  5 * x + y = 19 →
  x + a * y = 1 →
  3 * x + 2 * y = 10 →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l3443_344317


namespace NUMINAMATH_CALUDE_q_of_one_equals_zero_l3443_344300

/-- Given a function q: ℝ → ℝ, prove that q(1) = 0 -/
theorem q_of_one_equals_zero (q : ℝ → ℝ) 
  (h1 : (1, 0) ∈ Set.range (λ x => (x, q x))) 
  (h2 : ∃ n : ℤ, q 1 = n) : 
  q 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_q_of_one_equals_zero_l3443_344300


namespace NUMINAMATH_CALUDE_selectBooks_eq_1041_l3443_344312

/-- The number of ways to select books for three children -/
def selectBooks : ℕ :=
  let smallBooks := 6
  let largeBooks := 3
  let children := 3

  -- Case 1: All children take large books
  let case1 := largeBooks.factorial

  -- Case 2: 1 child takes small books, 2 take large books
  let case2 := Nat.choose children 1 * Nat.choose smallBooks 2 * Nat.choose largeBooks 2

  -- Case 3: 2 children take small books, 1 takes large book
  let case3 := Nat.choose children 2 * Nat.choose smallBooks 2 * Nat.choose (smallBooks - 2) 2 * Nat.choose largeBooks 1

  -- Case 4: All children take small books
  let case4 := Nat.choose smallBooks 2 * Nat.choose (smallBooks - 2) 2 * Nat.choose (smallBooks - 4) 2

  case1 + case2 + case3 + case4

theorem selectBooks_eq_1041 : selectBooks = 1041 := by
  sorry

end NUMINAMATH_CALUDE_selectBooks_eq_1041_l3443_344312


namespace NUMINAMATH_CALUDE_expand_expression_l3443_344334

theorem expand_expression (x : ℝ) : 2 * (x + 3) * (x + 6) + x = 2 * x^2 + 19 * x + 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3443_344334


namespace NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l3443_344387

/-- Represents a type of algorithm statement -/
inductive AlgorithmStatement
  | INPUT
  | PRINT
  | IF_THEN
  | DO
  | END
  | WHILE
  | END_IF

/-- Defines the set of basic algorithm statements -/
def BasicAlgorithmStatements : Set AlgorithmStatement :=
  {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT, AlgorithmStatement.IF_THEN,
   AlgorithmStatement.DO, AlgorithmStatement.WHILE}

/-- Theorem stating that the set of basic algorithm statements is correct -/
theorem basic_algorithm_statements_correct :
  BasicAlgorithmStatements = {AlgorithmStatement.INPUT, AlgorithmStatement.PRINT,
    AlgorithmStatement.IF_THEN, AlgorithmStatement.DO, AlgorithmStatement.WHILE} := by
  sorry

end NUMINAMATH_CALUDE_basic_algorithm_statements_correct_l3443_344387


namespace NUMINAMATH_CALUDE_problem_solution_l3443_344327

theorem problem_solution :
  ∀ (A B C : ℝ) (a n b c : ℕ) (d : ℕ+),
    A^2 + B^2 + C^2 = 3 →
    A * B + B * C + C * A = 3 →
    a = A^2 →
    29 * n + 42 * b = a →
    5 < b →
    b < 10 →
    (Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) = 
      (c * Real.sqrt 21 - 18 * Real.sqrt 15 - 2 * Real.sqrt 35 + b) / 59 →
    d = (Nat.factors c).length →
    a = 1 ∧ b = 9 ∧ c = 20 ∧ d = 6 := by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l3443_344327


namespace NUMINAMATH_CALUDE_quadratic_properties_l3443_344306

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 12 * x + 10

-- State the theorem
theorem quadratic_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 0 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3443_344306


namespace NUMINAMATH_CALUDE_custom_op_result_l3443_344322

/-- Custom operation € -/
def custom_op (x y : ℝ) : ℝ := 3 * x * y - x - y

/-- Theorem stating the result of the custom operation -/
theorem custom_op_result : 
  let x : ℝ := 6
  let y : ℝ := 4
  let z : ℝ := 2
  custom_op x (custom_op y z) = 300 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_result_l3443_344322


namespace NUMINAMATH_CALUDE_only_fourteen_satisfies_l3443_344371

-- Define a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the operation of increasing digits
def increase_digits (n : ℕ) : Set ℕ :=
  { m : ℕ | ∃ (a b : ℕ), n = 10 * a + b ∧ 
    m = 10 * (a + 2) + (b + 2) ∨ 
    m = 10 * (a + 2) + (b + 4) ∨ 
    m = 10 * (a + 4) + (b + 2) ∨ 
    m = 10 * (a + 4) + (b + 4) }

-- The main theorem
theorem only_fourteen_satisfies : 
  ∃! (n : ℕ), is_two_digit n ∧ (4 * n) ∈ increase_digits n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_only_fourteen_satisfies_l3443_344371


namespace NUMINAMATH_CALUDE_total_sum_l3443_344364

/-- Represents the shares of money for each person -/
structure Shares where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The problem statement -/
def problem (s : Shares) : Prop :=
  s.b = 0.75 * s.a ∧
  s.c = 0.60 * s.a ∧
  s.d = 0.50 * s.a ∧
  s.e = 0.40 * s.a ∧
  s.e = 84

/-- The theorem to prove -/
theorem total_sum (s : Shares) : problem s → s.a + s.b + s.c + s.d + s.e = 682.50 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_l3443_344364


namespace NUMINAMATH_CALUDE_pyramid_scheme_characterization_l3443_344367

/-- Represents a financial scheme -/
structure FinancialScheme where
  returns : ℝ
  information_completeness : ℝ
  advertising_aggressiveness : ℝ

/-- Defines the average market return -/
def average_market_return : ℝ := sorry

/-- Defines the threshold for complete information -/
def complete_information_threshold : ℝ := sorry

/-- Defines the threshold for aggressive advertising -/
def aggressive_advertising_threshold : ℝ := sorry

/-- Determines if a financial scheme is a pyramid scheme -/
def is_pyramid_scheme (scheme : FinancialScheme) : Prop :=
  scheme.returns > average_market_return ∧
  scheme.information_completeness < complete_information_threshold ∧
  scheme.advertising_aggressiveness > aggressive_advertising_threshold

theorem pyramid_scheme_characterization (scheme : FinancialScheme) :
  is_pyramid_scheme scheme ↔
    scheme.returns > average_market_return ∧
    scheme.information_completeness < complete_information_threshold ∧
    scheme.advertising_aggressiveness > aggressive_advertising_threshold := by
  sorry

end NUMINAMATH_CALUDE_pyramid_scheme_characterization_l3443_344367


namespace NUMINAMATH_CALUDE_sams_pen_collection_l3443_344313

/-- The number of pens in Sam's collection -/
def total_pens (black blue red pencils : ℕ) : ℕ := black + blue + red

/-- The problem statement -/
theorem sams_pen_collection :
  ∀ (black blue red pencils : ℕ),
  black = blue + 10 →
  blue = 2 * pencils →
  pencils = 8 →
  red = pencils - 2 →
  total_pens black blue red pencils = 48 := by
sorry

end NUMINAMATH_CALUDE_sams_pen_collection_l3443_344313


namespace NUMINAMATH_CALUDE_particle_max_elevation_l3443_344397

/-- The elevation function for a vertically projected particle -/
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 20

/-- The maximum elevation achieved by the particle -/
def max_elevation : ℝ := 520

/-- Theorem stating that the maximum elevation is 520 feet -/
theorem particle_max_elevation :
  ∃ t : ℝ, elevation t = max_elevation ∧ ∀ u : ℝ, elevation u ≤ max_elevation := by
  sorry

end NUMINAMATH_CALUDE_particle_max_elevation_l3443_344397


namespace NUMINAMATH_CALUDE_sandwich_cost_l3443_344388

-- Define the given values
def num_sandwiches : ℕ := 3
def num_energy_bars : ℕ := 3
def num_drinks : ℕ := 2
def drink_cost : ℚ := 4
def energy_bar_cost : ℚ := 3
def energy_bar_discount : ℚ := 0.2
def total_spent : ℚ := 40.80

-- Define the theorem
theorem sandwich_cost :
  let drink_total : ℚ := num_drinks * drink_cost
  let energy_bar_total : ℚ := num_energy_bars * energy_bar_cost * (1 - energy_bar_discount)
  let sandwich_total : ℚ := total_spent - drink_total - energy_bar_total
  sandwich_total / num_sandwiches = 8.53 := by sorry

end NUMINAMATH_CALUDE_sandwich_cost_l3443_344388


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3443_344303

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 2)^2 + 3^2)^(1/2) = 6 → 
  x = 2 + 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3443_344303


namespace NUMINAMATH_CALUDE_calories_per_shake_johns_shake_calories_l3443_344340

/-- Calculates the calories in each shake given John's daily meal plan. -/
theorem calories_per_shake (breakfast : ℕ) (total_daily : ℕ) : ℕ :=
  let lunch := breakfast + breakfast / 4
  let dinner := 2 * lunch
  let meals_total := breakfast + lunch + dinner
  let shakes_total := total_daily - meals_total
  shakes_total / 3

/-- Proves that each shake contains 300 calories given John's meal plan. -/
theorem johns_shake_calories :
  calories_per_shake 500 3275 = 300 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_shake_johns_shake_calories_l3443_344340


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l3443_344330

theorem chinese_chess_draw_probability 
  (p_xiao_ming_not_lose : ℚ) 
  (p_xiao_dong_lose : ℚ) 
  (h1 : p_xiao_ming_not_lose = 3/4) 
  (h2 : p_xiao_dong_lose = 1/2) : 
  p_xiao_ming_not_lose - p_xiao_dong_lose = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l3443_344330


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3443_344336

/-- A plane in 3D space represented by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- A point in 3D space -/
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z + plane.D = 0

/-- Check if two planes are parallel -/
def Plane.isParallelTo (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

theorem plane_equation_proof (given_plane : Plane) (point : Point3D) :
  given_plane.A = -2 ∧ given_plane.B = 1 ∧ given_plane.C = -3 ∧ given_plane.D = 7 →
  point.x = 1 ∧ point.y = 4 ∧ point.z = -2 →
  ∃ (result_plane : Plane),
    result_plane.A = 2 ∧ 
    result_plane.B = -1 ∧ 
    result_plane.C = 3 ∧ 
    result_plane.D = 8 ∧
    point.liesOn result_plane ∧
    result_plane.isParallelTo given_plane :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3443_344336


namespace NUMINAMATH_CALUDE_discount_calculation_l3443_344316

/-- Calculates the discount amount given the cost of a suit, shoes, and the final payment -/
theorem discount_calculation (suit_cost shoes_cost final_payment : ℕ) :
  suit_cost = 430 →
  shoes_cost = 190 →
  final_payment = 520 →
  suit_cost + shoes_cost - final_payment = 100 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3443_344316


namespace NUMINAMATH_CALUDE_no_eulerian_path_in_picture_graph_l3443_344314

/-- A graph representing the regions in the picture --/
structure PictureGraph where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  adjacent : (a b : Fin 6) → (a, b) ∈ edges → a ≠ b

/-- The degree of a vertex in the graph --/
def degree (G : PictureGraph) (v : Fin 6) : Nat :=
  (G.edges.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- An Eulerian path visits each edge exactly once --/
def hasEulerianPath (G : PictureGraph) : Prop :=
  ∃ path : List (Fin 6), path.length = G.edges.card + 1 ∧
    (∀ e ∈ G.edges, ∃ i, path[i]? = some e.1 ∧ path[i+1]? = some e.2)

/-- The main theorem: No Eulerian path exists in this graph --/
theorem no_eulerian_path_in_picture_graph (G : PictureGraph) 
  (h1 : ∃ v1 v2 v3 : Fin 6, v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ 
    degree G v1 = 5 ∧ degree G v2 = 5 ∧ degree G v3 = 9) :
  ¬ hasEulerianPath G := by
  sorry


end NUMINAMATH_CALUDE_no_eulerian_path_in_picture_graph_l3443_344314


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3443_344396

theorem complete_square_quadratic (x : ℝ) : ∃ (a b : ℝ), (x^2 + 10*x - 1 = 0) ↔ ((x + a)^2 = b) ∧ b = 26 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3443_344396


namespace NUMINAMATH_CALUDE_length_NM_is_3_l3443_344381

-- Define the points and segments
variable (A B M N : ℝ × ℝ)
variable (AB AM NM : ℝ)

-- State the given conditions
axiom length_AB : AB = 12
axiom M_midpoint_AB : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
axiom N_midpoint_AM : N = ((A.1 + M.1) / 2, (A.2 + M.2) / 2)

-- Define the theorem
theorem length_NM_is_3 : NM = 3 :=
sorry

end NUMINAMATH_CALUDE_length_NM_is_3_l3443_344381
