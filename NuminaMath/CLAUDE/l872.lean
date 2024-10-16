import Mathlib

namespace NUMINAMATH_CALUDE_rani_cycling_speed_difference_l872_87236

/-- Rani's cycling speed as a girl in miles per minute -/
def girl_speed : ℚ := 20 / (2 * 60 + 45)

/-- Rani's cycling speed as an older woman in miles per minute -/
def woman_speed : ℚ := 12 / (3 * 60)

/-- The difference in minutes per mile between Rani's cycling speed as an older woman and as a girl -/
def speed_difference : ℚ := (1 / woman_speed) - (1 / girl_speed)

theorem rani_cycling_speed_difference :
  speed_difference = 6.75 := by sorry

end NUMINAMATH_CALUDE_rani_cycling_speed_difference_l872_87236


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l872_87245

theorem unique_solution_cube_equation :
  ∃! (y : ℝ), y ≠ 0 ∧ (3 * y)^6 = (9 * y)^5 :=
by
  use 81
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l872_87245


namespace NUMINAMATH_CALUDE_pinwheel_area_l872_87242

-- Define the constants
def π : ℝ := 3.14
def large_diameter : ℝ := 20

-- Define the theorem
theorem pinwheel_area : 
  let large_radius : ℝ := large_diameter / 2
  let small_radius : ℝ := large_radius / 2
  let large_area : ℝ := π * large_radius ^ 2
  let small_area : ℝ := π * small_radius ^ 2
  let cut_out_area : ℝ := 2 * small_area
  let remaining_area : ℝ := large_area - cut_out_area
  remaining_area = 157 := by
sorry

end NUMINAMATH_CALUDE_pinwheel_area_l872_87242


namespace NUMINAMATH_CALUDE_revenue_difference_l872_87244

/-- Represents the sales data for a season -/
structure SeasonData where
  packsPerHour : ℕ
  pricePerPack : ℕ
  salesHours : ℕ

/-- Calculates the revenue for a given season -/
def calculateRevenue (data : SeasonData) : ℕ :=
  data.packsPerHour * data.pricePerPack * data.salesHours

/-- The peak season data -/
def peakSeason : SeasonData := {
  packsPerHour := 8,
  pricePerPack := 70,
  salesHours := 17
}

/-- The low season data -/
def lowSeason : SeasonData := {
  packsPerHour := 5,
  pricePerPack := 50,
  salesHours := 14
}

/-- The theorem stating the difference in revenue between peak and low seasons -/
theorem revenue_difference : 
  calculateRevenue peakSeason - calculateRevenue lowSeason = 6020 := by
  sorry


end NUMINAMATH_CALUDE_revenue_difference_l872_87244


namespace NUMINAMATH_CALUDE_f_properties_l872_87281

def f (x : ℝ) : ℝ := |2*x - 1| + 1

theorem f_properties :
  (∀ x, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m, (∃ n, f n ≤ m - f (-n)) → m ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l872_87281


namespace NUMINAMATH_CALUDE_cindy_friends_l872_87233

/-- Calculates the number of friends Cindy gives envelopes to -/
def num_friends (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (remaining_envelopes : ℕ) : ℕ :=
  (initial_envelopes - remaining_envelopes) / envelopes_per_friend

/-- Proves that Cindy gives envelopes to 5 friends -/
theorem cindy_friends : num_friends 37 3 22 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cindy_friends_l872_87233


namespace NUMINAMATH_CALUDE_scientific_notation_of_35000000_l872_87225

theorem scientific_notation_of_35000000 :
  (35000000 : ℝ) = 3.5 * (10 ^ 7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_35000000_l872_87225


namespace NUMINAMATH_CALUDE_missing_ratio_l872_87272

theorem missing_ratio (x y : ℚ) (h : x / y * (6 / 11) * (11 / 2) = 2) : x / y = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_ratio_l872_87272


namespace NUMINAMATH_CALUDE_village_population_equality_l872_87215

/-- The number of years it takes for two villages' populations to be equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (y_increase + x_decrease)

theorem village_population_equality :
  years_until_equal_population 78000 1200 42000 800 = 18 := by
  sorry

end NUMINAMATH_CALUDE_village_population_equality_l872_87215


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l872_87294

theorem ratio_x_to_y (x y : ℝ) (h : (8*x - 5*y) / (11*x - 3*y) = 2/7) : 
  x/y = 29/34 := by sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l872_87294


namespace NUMINAMATH_CALUDE_range_of_a_l872_87265

-- Define the propositions p and q
def p (x : ℝ) : Prop := (2*x - 1) / (x - 1) ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

-- Define the set of x that satisfy p
def P : Set ℝ := {x | p x}

-- Define the set of x that satisfy q
def Q (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (P ⊆ Q a ∧ ¬(Q a ⊆ P))) → 
  {a : ℝ | 0 ≤ a ∧ a < 1/2} = {a : ℝ | ∃ x, q x a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l872_87265


namespace NUMINAMATH_CALUDE_cos_three_pi_halves_l872_87280

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_halves_l872_87280


namespace NUMINAMATH_CALUDE_f_of_f_2_l872_87254

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.log x / Real.log 2
  else if x ≥ 1 then 1 / x^2
  else 0  -- This case is added to make the function total

theorem f_of_f_2 : f (f 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_2_l872_87254


namespace NUMINAMATH_CALUDE_quadratic_root_implies_q_l872_87277

theorem quadratic_root_implies_q (p q : ℝ) : 
  (∃ (x : ℂ), 3 * x^2 + p * x + q = 0 ∧ x = 3 + 4*I) → q = 75 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_q_l872_87277


namespace NUMINAMATH_CALUDE_rectangle_division_theorem_l872_87282

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottomLeft : Point
  topRight : Point

/-- Checks if a point is inside a rectangle -/
def pointInRectangle (p : Point) (r : Rectangle) : Prop :=
  r.bottomLeft.x ≤ p.x ∧ p.x ≤ r.topRight.x ∧
  r.bottomLeft.y ≤ p.y ∧ p.y ≤ r.topRight.y

/-- Theorem: Given a rectangle with 4 points, it can be divided into 4 equal rectangles, each containing one point -/
theorem rectangle_division_theorem 
  (r : Rectangle) 
  (p1 p2 p3 p4 : Point) 
  (h1 : pointInRectangle p1 r)
  (h2 : pointInRectangle p2 r)
  (h3 : pointInRectangle p3 r)
  (h4 : pointInRectangle p4 r)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4) :
  ∃ (r1 r2 r3 r4 : Rectangle),
    -- The four rectangles are equal in area
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r2.topRight.x - r2.bottomLeft.x) * (r2.topRight.y - r2.bottomLeft.y) ∧
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r3.topRight.x - r3.bottomLeft.x) * (r3.topRight.y - r3.bottomLeft.y) ∧
    (r1.topRight.x - r1.bottomLeft.x) * (r1.topRight.y - r1.bottomLeft.y) =
    (r4.topRight.x - r4.bottomLeft.x) * (r4.topRight.y - r4.bottomLeft.y) ∧
    -- Each smaller rectangle contains exactly one point
    (pointInRectangle p1 r1 ∧ pointInRectangle p2 r2 ∧ pointInRectangle p3 r3 ∧ pointInRectangle p4 r4) ∧
    -- The union of the smaller rectangles is the original rectangle
    (r1.bottomLeft = r.bottomLeft) ∧ (r4.topRight = r.topRight) ∧
    (r1.topRight.x = r2.bottomLeft.x) ∧ (r2.topRight.x = r.topRight.x) ∧
    (r1.topRight.y = r3.bottomLeft.y) ∧ (r3.topRight.y = r.topRight.y) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_theorem_l872_87282


namespace NUMINAMATH_CALUDE_valid_positions_count_l872_87275

/-- Represents a 6x6 chess board -/
def Board := Fin 6 → Fin 6 → Bool

/-- Represents a position of 4 chips on the board -/
def ChipPosition := Fin 4 → Fin 6 × Fin 6

/-- Checks if four points are collinear -/
def areCollinear (p₁ p₂ p₃ p₄ : Fin 6 × Fin 6) : Bool :=
  sorry

/-- Checks if a square is attacked by at least one chip -/
def isAttacked (board : Board) (pos : ChipPosition) (x y : Fin 6) : Bool :=
  sorry

/-- Checks if all squares are attacked by at least one chip -/
def allSquaresAttacked (board : Board) (pos : ChipPosition) : Bool :=
  sorry

/-- Checks if a chip position is valid (chips are collinear and all squares are attacked) -/
def isValidPosition (board : Board) (pos : ChipPosition) : Bool :=
  sorry

/-- Counts the number of valid chip positions, including rotations and reflections -/
def countValidPositions (board : Board) : Nat :=
  sorry

/-- The main theorem: there are exactly 48 valid chip positions -/
theorem valid_positions_count :
  ∀ (board : Board), countValidPositions board = 48 :=
sorry

end NUMINAMATH_CALUDE_valid_positions_count_l872_87275


namespace NUMINAMATH_CALUDE_girls_fraction_l872_87293

theorem girls_fraction (T G B : ℚ) 
  (h1 : G > 0) 
  (h2 : T > 0) 
  (h3 : ∃ X : ℚ, X * G = (1/5) * T) 
  (h4 : B / G = 7/3) 
  (h5 : T = B + G) : 
  ∃ X : ℚ, X * G = (1/5) * T ∧ X = 2/3 := by
sorry

end NUMINAMATH_CALUDE_girls_fraction_l872_87293


namespace NUMINAMATH_CALUDE_carpenter_problem_solution_l872_87251

/-- Represents the carpenter problem -/
def CarpenterProblem (x : ℝ) : Prop :=
  let first_carpenter_rate := 1 / (x + 4)
  let second_carpenter_rate := 1 / 5
  let combined_rate := first_carpenter_rate + second_carpenter_rate
  2 * combined_rate = 4 * first_carpenter_rate

/-- The solution to the carpenter problem is 1 day -/
theorem carpenter_problem_solution :
  ∃ (x : ℝ), CarpenterProblem x ∧ x = 1 :=
sorry

end NUMINAMATH_CALUDE_carpenter_problem_solution_l872_87251


namespace NUMINAMATH_CALUDE_football_game_attendance_l872_87260

/-- Football game attendance problem -/
theorem football_game_attendance 
  (saturday_attendance : ℕ)
  (monday_attendance : ℕ)
  (wednesday_attendance : ℕ)
  (friday_attendance : ℕ)
  (expected_total : ℕ)
  (actual_total : ℕ)
  (h1 : saturday_attendance = 80)
  (h2 : monday_attendance = saturday_attendance - 20)
  (h3 : wednesday_attendance > monday_attendance)
  (h4 : friday_attendance = saturday_attendance + monday_attendance)
  (h5 : expected_total = 350)
  (h6 : actual_total = expected_total + 40)
  (h7 : actual_total = saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance) :
  wednesday_attendance - monday_attendance = 50 := by
  sorry

end NUMINAMATH_CALUDE_football_game_attendance_l872_87260


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_sum_l872_87204

-- Define the vectors a and b
def a (m n : ℝ) : Fin 3 → ℝ := ![2*m - 3, n + 2, 3]
def b (m n : ℝ) : Fin 3 → ℝ := ![2*m + 1, 3*n - 2, 6]

-- Define parallel vectors
def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, v i = k * u i)

-- Theorem statement
theorem parallel_vectors_imply_sum (m n : ℝ) :
  parallel (a m n) (b m n) → 2*m + n = 13 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_sum_l872_87204


namespace NUMINAMATH_CALUDE_math_voters_l872_87250

theorem math_voters (total_students : ℕ) (math_percentage : ℚ) : 
  total_students = 480 → math_percentage = 40 / 100 →
  (math_percentage * total_students.cast) = 192 := by
sorry

end NUMINAMATH_CALUDE_math_voters_l872_87250


namespace NUMINAMATH_CALUDE_cyclicInequality_l872_87239

theorem cyclicInequality (x y z p q : ℝ) (n : ℕ) (hn : n = 2 ∨ n = 2010) 
  (h1 : y = x^n + p*x + q) 
  (h2 : z = y^n + p*y + q) 
  (h3 : x = z^n + p*z + q) : 
  x^2 * y + y^2 * z + z^2 * x ≥ x^2 * z + y^2 * x + z^2 * y := by
  sorry

#check cyclicInequality

end NUMINAMATH_CALUDE_cyclicInequality_l872_87239


namespace NUMINAMATH_CALUDE_solve_equation_l872_87248

theorem solve_equation (x : ℝ) (h : x^2 - 3*x - 1 = 0) : -3*x^2 + 9*x + 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l872_87248


namespace NUMINAMATH_CALUDE_geometry_propositions_l872_87205

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the basic relations
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Plane → Line → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelPL : Plane → Line → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (((intersect α β = m) ∧ (contains α n) ∧ (perpendicular n m)) → (perpendicularPP α β)) ∧
  ((perpendicularPL α m) ∧ (perpendicularPL β m) → (parallelPP α β)) ∧
  ((perpendicularPL α m) ∧ (perpendicularPL β n) ∧ (perpendicular m n) → (perpendicularPP α β)) ∧
  (∃ (m n : Line) (α β : Plane), (parallelPL α m) ∧ (parallelPL β n) ∧ (parallel m n) ∧ ¬(parallelPP α β)) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l872_87205


namespace NUMINAMATH_CALUDE_smallest_a_for_96a_squared_equals_b_cubed_l872_87247

theorem smallest_a_for_96a_squared_equals_b_cubed :
  ∀ a : ℕ+, a < 12 → ¬∃ b : ℕ+, 96 * a^2 = b^3 ∧ 
  ∃ b : ℕ+, 96 * 12^2 = b^3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_96a_squared_equals_b_cubed_l872_87247


namespace NUMINAMATH_CALUDE_exists_circumcircle_equation_l872_87211

/-- Triangle with side lengths 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- Circumcircle of a triangle -/
structure Circumcircle (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  is_valid : radius^2 = (t.c / 2)^2

theorem exists_circumcircle_equation (t : RightTriangle) :
  ∃ (cc : Circumcircle t), ∃ (x y : ℝ), (x - cc.center.1)^2 + (y - cc.center.2)^2 = cc.radius^2 ∧
  cc.center = (0, 0) ∧ cc.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_exists_circumcircle_equation_l872_87211


namespace NUMINAMATH_CALUDE_journey_with_four_encounters_takes_five_days_l872_87249

/-- A train journey with daily departures -/
structure TrainJourney where
  /-- The number of trains encountered during the journey -/
  trains_encountered : ℕ

/-- The duration of a train journey in days -/
def journey_duration (j : TrainJourney) : ℕ :=
  j.trains_encountered + 1

/-- Theorem: A train journey where 4 trains are encountered takes 5 days -/
theorem journey_with_four_encounters_takes_five_days (j : TrainJourney) 
    (h : j.trains_encountered = 4) : journey_duration j = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_with_four_encounters_takes_five_days_l872_87249


namespace NUMINAMATH_CALUDE_office_canteen_round_tables_l872_87261

theorem office_canteen_round_tables :
  let rectangular_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let chairs_per_rectangular_table : ℕ := 7
  let total_chairs : ℕ := 26
  
  ∃ (round_tables : ℕ),
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table = total_chairs ∧
    round_tables = 2 :=
by sorry

end NUMINAMATH_CALUDE_office_canteen_round_tables_l872_87261


namespace NUMINAMATH_CALUDE_equation_solution_l872_87241

theorem equation_solution (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 4/3) :
  ∃! x : ℝ, (Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x) ∧
            (x = (4 - p) / Real.sqrt (8 * (2 - p))) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l872_87241


namespace NUMINAMATH_CALUDE_olympic_production_l872_87286

/-- The number of sets of Olympic logo and mascots that can be produced -/
theorem olympic_production : ∃ (x y : ℕ), 
  4 * x + 5 * y = 20000 ∧ 
  3 * x + 10 * y = 30000 ∧ 
  x = 2000 ∧ 
  y = 2400 := by
  sorry

end NUMINAMATH_CALUDE_olympic_production_l872_87286


namespace NUMINAMATH_CALUDE_contractor_absence_l872_87279

theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_amount : ℚ) :
  total_days = 30 ∧
  daily_pay = 25 ∧
  daily_fine = (15/2) ∧
  total_amount = 685 →
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    daily_pay * days_worked - daily_fine * days_absent = total_amount ∧
    days_absent = 2 :=
by sorry

end NUMINAMATH_CALUDE_contractor_absence_l872_87279


namespace NUMINAMATH_CALUDE_total_amount_theorem_l872_87257

/-- The total amount spent on cows and goats -/
def total_amount_spent (num_cows num_goats avg_price_cow avg_price_goat : ℕ) : ℕ :=
  num_cows * avg_price_cow + num_goats * avg_price_goat

/-- Theorem: The total amount spent on 2 cows and 10 goats is 1500 rupees -/
theorem total_amount_theorem :
  total_amount_spent 2 10 400 70 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l872_87257


namespace NUMINAMATH_CALUDE_divisibility_by_24_l872_87271

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) : 
  (p^2 - 1) % 24 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l872_87271


namespace NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessity_not_sufficiency_l872_87206

theorem a_squared_gt_b_squared_necessity_not_sufficiency (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) :=
sorry

end NUMINAMATH_CALUDE_a_squared_gt_b_squared_necessity_not_sufficiency_l872_87206


namespace NUMINAMATH_CALUDE_continuity_at_3_l872_87210

def f (x : ℝ) : ℝ := -2 * x^2 - 4

theorem continuity_at_3 :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 12 ∧
  ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_continuity_at_3_l872_87210


namespace NUMINAMATH_CALUDE_rectangle_area_l872_87267

/-- Given a rectangle where the length is 3 times the width and the width is 6 inches,
    prove that the area is 108 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 6 →
  length = 3 * width →
  area = length * width →
  area = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l872_87267


namespace NUMINAMATH_CALUDE_inequality_proof_l872_87243

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + a * c + b * c = a + b + c) :
  a + b + c + 1 ≥ 4 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l872_87243


namespace NUMINAMATH_CALUDE_inverse_true_converse_false_l872_87253

-- Define the universe of shapes
variable (Shape : Type)

-- Define predicates for being a circle and having corners
variable (is_circle : Shape → Prop)
variable (has_corners : Shape → Prop)

-- Given statement
axiom circle_no_corners : ∀ s : Shape, is_circle s → ¬(has_corners s)

-- Theorem to prove
theorem inverse_true_converse_false :
  (∀ s : Shape, ¬(is_circle s) → has_corners s) ∧
  ¬(∀ s : Shape, ¬(has_corners s) → is_circle s) :=
sorry

end NUMINAMATH_CALUDE_inverse_true_converse_false_l872_87253


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l872_87202

theorem simplify_trig_expression :
  (Real.sin (25 * π / 180) + Real.sin (35 * π / 180)) /
  (Real.cos (25 * π / 180) + Real.cos (35 * π / 180)) =
  Real.tan (30 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l872_87202


namespace NUMINAMATH_CALUDE_marble_redistribution_l872_87223

theorem marble_redistribution (dilan martha phillip veronica : ℕ) 
  (h1 : dilan = 14)
  (h2 : martha = 20)
  (h3 : phillip = 19)
  (h4 : veronica = 7) :
  (dilan + martha + phillip + veronica) / 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marble_redistribution_l872_87223


namespace NUMINAMATH_CALUDE_smallest_independent_after_reorganization_l872_87269

/-- Represents a faction of deputies -/
structure Faction :=
  (size : ℕ)

/-- Represents the parliament configuration -/
structure Parliament :=
  (factions : List Faction)
  (independent : ℕ)

def initialParliament : Parliament :=
  { factions := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14].map (λ n => ⟨n⟩),
    independent := 0 }

def totalDeputies (p : Parliament) : ℕ :=
  p.factions.foldl (λ acc f => acc + f.size) p.independent

def isValidReorganization (initial final : Parliament) : Prop :=
  totalDeputies initial = totalDeputies final ∧
  final.factions.all (λ f => f.size ≤ initial.factions.length) ∧
  final.factions.all (λ f => f.size ≥ 5)

theorem smallest_independent_after_reorganization :
  ∀ (final : Parliament),
    isValidReorganization initialParliament final →
    final.independent ≥ 50 :=
sorry

end NUMINAMATH_CALUDE_smallest_independent_after_reorganization_l872_87269


namespace NUMINAMATH_CALUDE_replaced_men_age_sum_l872_87235

/-- Given a group of 8 men where replacing two of them with two women increases the average age by 2 years,
    and the average age of the women is 32 years, prove that the combined age of the two replaced men is 48 years. -/
theorem replaced_men_age_sum (n : ℕ) (A : ℝ) (women_avg_age : ℝ) :
  n = 8 ∧ women_avg_age = 32 →
  ∃ (older_man_age younger_man_age : ℝ),
    n * (A + 2) = (n - 2) * A + 2 * women_avg_age ∧
    older_man_age + younger_man_age = 48 :=
by sorry

end NUMINAMATH_CALUDE_replaced_men_age_sum_l872_87235


namespace NUMINAMATH_CALUDE_gcd_840_1764_l872_87291

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l872_87291


namespace NUMINAMATH_CALUDE_lunks_needed_for_apples_l872_87209

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (4 / 7) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5 / 3) * k

/-- Number of apples to be purchased -/
def apples_to_buy : ℕ := 24

/-- Theorem stating that at least 27 lunks are needed to buy 24 apples -/
theorem lunks_needed_for_apples :
  ∀ l : ℚ, kunks_to_apples (lunks_to_kunks l) ≥ apples_to_buy → l ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_apples_l872_87209


namespace NUMINAMATH_CALUDE_problem_statement_l872_87299

theorem problem_statement (x y : ℝ) (h : Real.sqrt (x - 1) + (y + 2)^2 = 0) :
  (x + y)^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l872_87299


namespace NUMINAMATH_CALUDE_quadratic_minimum_l872_87207

/-- The quadratic function f(x) = x^2 - 12x + 35 attains its minimum value when x = 6. -/
theorem quadratic_minimum (x : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 12*x + 35
  ∀ y, f 6 ≤ f y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l872_87207


namespace NUMINAMATH_CALUDE_n_has_24_digits_l872_87270

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 12 -/
axiom n_div_12 : 12 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n^4 is a perfect fifth power -/
axiom n_fourth_fifth : ∃ k : ℕ, n^4 = k^5

/-- n is the smallest positive integer satisfying all conditions -/
axiom n_smallest : ∀ m : ℕ, m > 0 → (12 ∣ m) → (∃ k : ℕ, m^2 = k^3) → 
  (∃ k : ℕ, m^3 = k^2) → (∃ k : ℕ, m^4 = k^5) → m ≥ n

/-- Function to count the number of digits in a natural number -/
def digit_count (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 24 digits -/
theorem n_has_24_digits : digit_count n = 24 := sorry

end NUMINAMATH_CALUDE_n_has_24_digits_l872_87270


namespace NUMINAMATH_CALUDE_jungkook_has_larger_number_l872_87285

theorem jungkook_has_larger_number (yoongi_number jungkook_number : ℕ) : 
  yoongi_number = 4 → jungkook_number = 6 * 3 → jungkook_number > yoongi_number := by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_larger_number_l872_87285


namespace NUMINAMATH_CALUDE_square_mod_nine_not_five_l872_87218

theorem square_mod_nine_not_five (n : ℤ) : n^2 % 9 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_nine_not_five_l872_87218


namespace NUMINAMATH_CALUDE_fifteen_point_figures_l872_87220

def points : ℕ := 15

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Number of quadrilaterals
def quadrilaterals : ℕ := choose points 4

-- Number of triangles
def triangles : ℕ := choose points 3

-- Total number of figures
def total_figures : ℕ := quadrilaterals + triangles

-- Theorem statement
theorem fifteen_point_figures : total_figures = 1820 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_point_figures_l872_87220


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l872_87284

theorem gcd_from_lcm_and_ratio (X Y : ℕ) (h_lcm : Nat.lcm X Y = 180) (h_ratio : 5 * X = 2 * Y) : 
  Nat.gcd X Y = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l872_87284


namespace NUMINAMATH_CALUDE_lisas_lasagna_consumption_l872_87203

/-- The number of pieces Lisa eats from a lasagna, given the eating habits of her friends. -/
def lisas_lasagna_pieces (total_pieces manny_pieces aaron_pieces : ℚ) : ℚ :=
  let kai_pieces := 2 * manny_pieces
  let raphael_pieces := manny_pieces / 2
  total_pieces - (manny_pieces + kai_pieces + raphael_pieces + aaron_pieces)

/-- Theorem stating that Lisa will eat 2.5 pieces of lasagna given the specific conditions. -/
theorem lisas_lasagna_consumption :
  lisas_lasagna_pieces 6 1 0 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_lisas_lasagna_consumption_l872_87203


namespace NUMINAMATH_CALUDE_unique_rectangle_l872_87297

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area_eq : length * width = 100
  perimeter_eq : length + width = 24

/-- Two rectangles are considered distinct if they are not congruent -/
def distinct (r1 r2 : Rectangle) : Prop :=
  (r1.length ≠ r2.length ∧ r1.length ≠ r2.width) ∨
  (r1.width ≠ r2.length ∧ r1.width ≠ r2.width)

/-- There is exactly one distinct rectangle with area 100 and perimeter 24 -/
theorem unique_rectangle : ∃! r : Rectangle, ∀ s : Rectangle, ¬(distinct r s) :=
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_l872_87297


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l872_87219

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (1/a + 1/b) ≥ (3 + 2*Real.sqrt 2) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 1/b₀ = (3 + 2*Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l872_87219


namespace NUMINAMATH_CALUDE_square_of_hypotenuse_18_10_square_distance_18_east_10_north_l872_87237

/-- The square of the hypotenuse of a right triangle with sides 18 and 10 is 424 -/
theorem square_of_hypotenuse_18_10 : 18^2 + 10^2 = 424 := by
  sorry

/-- The square of the distance between two points, where one point is 18 km east
    and the other is 10 km north of the origin, is 424 km² -/
theorem square_distance_18_east_10_north : 18^2 + 10^2 = 424 := by
  sorry

end NUMINAMATH_CALUDE_square_of_hypotenuse_18_10_square_distance_18_east_10_north_l872_87237


namespace NUMINAMATH_CALUDE_m_range_l872_87234

-- Define the propositions p and q
def p (m : ℝ) : Prop := m + 1 ≤ 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Theorem statement
theorem m_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (m ≤ -2 ∨ (-1 < m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l872_87234


namespace NUMINAMATH_CALUDE_derek_car_increase_l872_87232

/-- Represents the number of dogs and cars Derek owns at a given time --/
structure DereksPets where
  dogs : ℕ
  cars : ℕ

/-- The change in Derek's pet ownership over 10 years --/
def petsChange (initial final : DereksPets) : ℕ := final.cars - initial.cars

/-- Theorem stating the increase in cars Derek owns over 10 years --/
theorem derek_car_increase :
  ∀ (initial final : DereksPets),
  initial.dogs = 90 →
  initial.dogs = 3 * initial.cars →
  final.dogs = 120 →
  final.cars = 2 * final.dogs →
  petsChange initial final = 210 := by
  sorry

end NUMINAMATH_CALUDE_derek_car_increase_l872_87232


namespace NUMINAMATH_CALUDE_mouse_cheese_distance_sum_l872_87278

/-- The point where the mouse begins moving away from the cheese -/
def mouse_turn_point (c d : ℝ) : Prop :=
  ∃ (k : ℝ), 
    d = -3 * c + 18 ∧  -- Mouse path
    d - 5 = k * (c - 20) ∧  -- Perpendicular line
    -3 * k = -1  -- Perpendicular condition

/-- The theorem stating the sum of coordinates where the mouse turns -/
theorem mouse_cheese_distance_sum : 
  ∃ (c d : ℝ), mouse_turn_point c d ∧ c + d = 9.4 :=
sorry

end NUMINAMATH_CALUDE_mouse_cheese_distance_sum_l872_87278


namespace NUMINAMATH_CALUDE_fraction_simplification_l872_87262

theorem fraction_simplification :
  (36 : ℚ) / 19 * 57 / 40 * 95 / 171 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l872_87262


namespace NUMINAMATH_CALUDE_expected_value_8_sided_die_l872_87264

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  let outcomes := standard_8_sided_die
  let prob (n : ℕ) := if n ∈ outcomes then (1 : ℚ) / 8 else 0
  let value (n : ℕ) := n + 1
  Finset.sum outcomes (λ n ↦ prob n * value n) = (9 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_8_sided_die_l872_87264


namespace NUMINAMATH_CALUDE_triangle_equilateral_l872_87256

theorem triangle_equilateral (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides opposite to angles
  a = 2 * Real.sin (A / 2) ∧
  b = 2 * Real.sin (B / 2) ∧
  c = 2 * Real.sin (C / 2) →
  -- Arithmetic sequence condition
  2 * b = a + c →
  -- Geometric sequence condition
  (Real.sin B)^2 = (Real.sin A) * (Real.sin C) →
  -- Conclusion: triangle is equilateral
  a = b ∧ b = c := by
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l872_87256


namespace NUMINAMATH_CALUDE_equality_except_two_l872_87201

theorem equality_except_two (x : ℝ) : 
  x ≠ 2 → (x^2 - 4*x + 4) / (x - 2) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_equality_except_two_l872_87201


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l872_87263

def is_arithmetic_progression (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def is_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |>.sum

theorem arithmetic_progression_first_term (a : ℕ → ℤ) :
  is_arithmetic_progression a →
  is_increasing a →
  let S := sum_first_n_terms a 10
  (a 6 * a 12 > S + 1) →
  (a 7 * a 11 < S + 17) →
  a 1 ∈ ({-6, -5, -4, -2, -1, 0} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l872_87263


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l872_87273

/-- Given a geometric sequence {a_n} where a₄ = 7 and a₈ = 63, prove that a₆ = 21 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 = 7 →                                  -- given a₄ = 7
  a 8 = 63 →                                 -- given a₈ = 63
  a 6 = 21 :=                                -- prove a₆ = 21
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l872_87273


namespace NUMINAMATH_CALUDE_intercept_sum_mod_17_l872_87283

theorem intercept_sum_mod_17 :
  ∃! (x₀ y₀ : ℕ), x₀ < 17 ∧ y₀ < 17 ∧
  (5 * x₀ ≡ 2 [MOD 17]) ∧
  (3 * y₀ + 2 ≡ 0 [MOD 17]) ∧
  x₀ + y₀ = 19 :=
by sorry

end NUMINAMATH_CALUDE_intercept_sum_mod_17_l872_87283


namespace NUMINAMATH_CALUDE_sum_of_powers_l872_87289

theorem sum_of_powers : (-3)^4 + (-3)^2 + (-3)^0 + 3^0 + 3^2 + 3^4 = 182 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l872_87289


namespace NUMINAMATH_CALUDE_bridge_length_l872_87229

/-- Given a train of length 120 meters traveling at 45 km/hr that crosses a bridge in 30 seconds,
    the length of the bridge is 255 meters. -/
theorem bridge_length (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 120 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_l872_87229


namespace NUMINAMATH_CALUDE_marble_probability_difference_l872_87224

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1500

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1500

/-- The number of white marbles in the box -/
def white_marbles : ℕ := 1

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles + white_marbles

/-- The probability of drawing two marbles of the same color (including white pairings) -/
def Ps : ℚ := (red_marbles * (red_marbles - 1) + black_marbles * (black_marbles - 1) + 2 * (red_marbles + black_marbles) * white_marbles) / (total_marbles * (total_marbles - 1))

/-- The probability of drawing two marbles of different colors (excluding white pairings) -/
def Pd : ℚ := (2 * red_marbles * black_marbles) / (total_marbles * (total_marbles - 1))

/-- The theorem stating that the absolute difference between Ps and Pd is 1/3 -/
theorem marble_probability_difference : |Ps - Pd| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l872_87224


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l872_87216

/-- The value of 'a' for which the line ax - y + 2 = 0 is tangent to the circle
    x = 2 + 2cos(θ), y = 1 + 2sin(θ) -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ θ : ℝ, (a * (2 + 2 * Real.cos θ) - (1 + 2 * Real.sin θ) + 2 = 0) →
   ∃ θ' : ℝ, (a * (2 + 2 * Real.cos θ') - (1 + 2 * Real.sin θ') + 2 = 0 ∧
              ∀ θ'' : ℝ, θ'' ≠ θ' → 
                a * (2 + 2 * Real.cos θ'') - (1 + 2 * Real.sin θ'') + 2 ≠ 0)) →
  a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l872_87216


namespace NUMINAMATH_CALUDE_ethans_work_hours_l872_87222

/-- Proves that Ethan works 8 hours per day given his earnings and work schedule --/
theorem ethans_work_hours 
  (hourly_rate : ℝ) 
  (days_per_week : ℕ) 
  (total_earnings : ℝ) 
  (total_weeks : ℕ) 
  (h1 : hourly_rate = 18)
  (h2 : days_per_week = 5)
  (h3 : total_earnings = 3600)
  (h4 : total_weeks = 5) :
  (total_earnings / total_weeks) / days_per_week / hourly_rate = 8 := by
  sorry

#check ethans_work_hours

end NUMINAMATH_CALUDE_ethans_work_hours_l872_87222


namespace NUMINAMATH_CALUDE_gigi_cookies_theorem_l872_87227

/-- Represents the number of cups of flour per batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the initial amount of flour in cups -/
def initial_flour : ℕ := 20

/-- Represents the number of additional batches that can be made with remaining flour -/
def additional_batches : ℕ := 7

/-- Calculates the number of batches Gigi baked initially -/
def batches_baked : ℕ := (initial_flour - additional_batches * flour_per_batch) / flour_per_batch

theorem gigi_cookies_theorem : batches_baked = 3 := by
  sorry

end NUMINAMATH_CALUDE_gigi_cookies_theorem_l872_87227


namespace NUMINAMATH_CALUDE_gervais_mileage_proof_l872_87296

/-- Gervais' average daily mileage --/
def gervais_average_mileage : ℝ := 315

/-- Number of days Gervais drove --/
def gervais_days : ℕ := 3

/-- Total miles Henri drove in a week --/
def henri_total_miles : ℝ := 1250

/-- Difference in miles between Henri and Gervais --/
def miles_difference : ℝ := 305

theorem gervais_mileage_proof :
  gervais_average_mileage * gervais_days = henri_total_miles - miles_difference :=
by sorry

end NUMINAMATH_CALUDE_gervais_mileage_proof_l872_87296


namespace NUMINAMATH_CALUDE_exists_proportion_with_means_less_than_extremes_l872_87258

/-- A proportion is represented by four real numbers a, b, c, d such that a : b = c : d -/
def IsProportion (a b c d : ℝ) : Prop := a * d = b * c

/-- Theorem: There exists a proportion where both means are less than both extremes -/
theorem exists_proportion_with_means_less_than_extremes :
  ∃ (a b c d : ℝ), IsProportion a b c d ∧ b < a ∧ b < d ∧ c < a ∧ c < d := by
  sorry

end NUMINAMATH_CALUDE_exists_proportion_with_means_less_than_extremes_l872_87258


namespace NUMINAMATH_CALUDE_equation_solution_l872_87288

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ (1 / x + (3 / x) / (6 / x) = 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l872_87288


namespace NUMINAMATH_CALUDE_cubic_difference_l872_87274

theorem cubic_difference (x y : ℝ) 
  (h1 : x + y - x * y = 155) 
  (h2 : x^2 + y^2 = 325) : 
  |x^3 - y^3| = 4375 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l872_87274


namespace NUMINAMATH_CALUDE_sum_of_powers_l872_87200

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l872_87200


namespace NUMINAMATH_CALUDE_xiaopang_problem_l872_87231

theorem xiaopang_problem (a : ℕ) (d : ℕ) (n : ℕ) : 
  a = 1 → d = 2 → n = 8 → (n / 2) * (2 * a + (n - 1) * d) = 64 := by
  sorry

end NUMINAMATH_CALUDE_xiaopang_problem_l872_87231


namespace NUMINAMATH_CALUDE_line_circle_intersection_range_l872_87259

/-- The range of m for a line intersecting a circle under specific conditions -/
theorem line_circle_intersection_range (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧ 
    A.1 + A.2 + m = 0 ∧ 
    B.1 + B.2 + m = 0 ∧ 
    A.1^2 + A.2^2 = 2 ∧ 
    B.1^2 + B.2^2 = 2 ∧ 
    ‖(A.1, A.2)‖ + ‖(B.1, B.2)‖ ≥ ‖(A.1 - B.1, A.2 - B.2)‖) →
  m ∈ Set.Ioo (-2 : ℝ) (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 2) 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_range_l872_87259


namespace NUMINAMATH_CALUDE_power_sum_integer_l872_87290

theorem power_sum_integer (x : ℝ) (h : ∃ (a : ℤ), x + 1/x = a) :
  ∀ (n : ℕ), ∃ (b : ℤ), x^n + 1/(x^n) = b :=
by sorry

end NUMINAMATH_CALUDE_power_sum_integer_l872_87290


namespace NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l872_87255

theorem ceiling_neg_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l872_87255


namespace NUMINAMATH_CALUDE_sport_to_standard_ratio_l872_87298

/-- Represents the ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
  | 0 => 1
  | 1 => 12
  | 2 => 30

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 105

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
def sport_flavoring_water_ratio : ℚ := 1/2

theorem sport_to_standard_ratio :
  let sport_flavoring := sport_water * (1 / (2 * standard_ratio 2))
  let sport_ratio := sport_flavoring / sport_corn_syrup
  let standard_ratio := (standard_ratio 0) / (standard_ratio 1)
  sport_ratio / standard_ratio = 1/3 := by sorry

end NUMINAMATH_CALUDE_sport_to_standard_ratio_l872_87298


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l872_87268

/-- Given a hyperbola with equation x²/8 - y²/2 = 1, its asymptotes have the equations y = ±(1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 8 - y^2 / 2 = 1 →
  ∃ (k : ℝ), k = 1/2 ∧ (y = k * x ∨ y = -k * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l872_87268


namespace NUMINAMATH_CALUDE_paint_mixer_days_to_make_drums_l872_87213

/-- Given a paint mixer who makes an equal number of drums each day,
    prove that if it takes 3 days to make 18 drums of paint,
    it will take 60 days to make 360 drums of paint. -/
theorem paint_mixer_days_to_make_drums
  (daily_production : ℕ → ℕ)  -- Function representing daily production
  (h1 : ∀ n : ℕ, daily_production n = daily_production 1)  -- Equal production each day
  (h2 : (daily_production 1) * 3 = 18)  -- 18 drums in 3 days
  : (daily_production 1) * 60 = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_mixer_days_to_make_drums_l872_87213


namespace NUMINAMATH_CALUDE_complex_square_equality_l872_87295

theorem complex_square_equality (x y : ℕ+) : 
  (x + y * Complex.I) ^ 2 = 7 + 24 * Complex.I → x + y * Complex.I = 4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l872_87295


namespace NUMINAMATH_CALUDE_jerry_tips_goal_l872_87238

def jerry_tips : List ℝ := [20, 60, 15, 40]
def work_days : ℕ := 5
def desired_average : ℝ := 50

theorem jerry_tips_goal (tips_needed : ℝ) : 
  (tips_needed + jerry_tips.sum) / work_days = desired_average →
  tips_needed = 115 := by
sorry

end NUMINAMATH_CALUDE_jerry_tips_goal_l872_87238


namespace NUMINAMATH_CALUDE_shorter_pipe_length_l872_87266

/-- Given a pipe of 177 inches cut into two pieces, where one piece is twice the length of the other,
    prove that the length of the shorter piece is 59 inches. -/
theorem shorter_pipe_length (total_length : ℝ) (short_length : ℝ) :
  total_length = 177 →
  total_length = short_length + 2 * short_length →
  short_length = 59 := by
  sorry

end NUMINAMATH_CALUDE_shorter_pipe_length_l872_87266


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_count_l872_87276

/-- The number of ways to distribute n indistinguishable balls among k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls among 4 distinguishable boxes -/
def five_balls_four_boxes : ℕ := distribute_balls 5 4

theorem five_balls_four_boxes_count : five_balls_four_boxes = 56 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_count_l872_87276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l872_87240

theorem arithmetic_sequence_sum (a b c : ℝ) : 
  (∃ d : ℝ, a = 3 + d ∧ b = a + d ∧ c = b + d ∧ 15 = c + d) → 
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l872_87240


namespace NUMINAMATH_CALUDE_number_problem_l872_87217

theorem number_problem (N : ℚ) : 
  (4/15 * 5/7 * N) - (4/9 * 2/5 * N) = 24 → N/2 = 945 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l872_87217


namespace NUMINAMATH_CALUDE_popcorn_yield_two_tablespoons_yield_l872_87292

/-- Represents the ratio of cups of popcorn to tablespoons of kernels -/
def popcorn_ratio (cups : ℚ) (tablespoons : ℚ) : Prop :=
  cups / tablespoons = 2

theorem popcorn_yield (cups : ℚ) (tablespoons : ℚ) 
  (h : popcorn_ratio 16 8) : 
  popcorn_ratio cups tablespoons → cups = 2 * tablespoons :=
by
  sorry

/-- Shows that 2 tablespoons of kernels make 4 cups of popcorn -/
theorem two_tablespoons_yield (h : popcorn_ratio 16 8) : 
  popcorn_ratio 4 2 :=
by
  sorry

end NUMINAMATH_CALUDE_popcorn_yield_two_tablespoons_yield_l872_87292


namespace NUMINAMATH_CALUDE_reading_difference_l872_87230

/-- The number of pages Janet reads per day -/
def janet_pages_per_day : ℕ := 80

/-- The number of pages Belinda reads per day -/
def belinda_pages_per_day : ℕ := 30

/-- The number of weeks in the reading period -/
def reading_weeks : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem reading_difference :
  (janet_pages_per_day - belinda_pages_per_day) * (reading_weeks * days_per_week) = 2100 := by
  sorry

end NUMINAMATH_CALUDE_reading_difference_l872_87230


namespace NUMINAMATH_CALUDE_earliest_retirement_year_l872_87252

/-- Rule of 70 retirement eligibility function -/
def eligible_to_retire (current_year : ℕ) (hire_year : ℕ) (hire_age : ℕ) : Prop :=
  (current_year - hire_year) + (hire_age + (current_year - hire_year)) ≥ 70

/-- Theorem: The earliest retirement year for an employee hired in 1989 at age 32 is 2008 -/
theorem earliest_retirement_year :
  ∀ year : ℕ, year ≥ 1989 →
  (eligible_to_retire year 1989 32 ↔ year ≥ 2008) :=
by sorry

end NUMINAMATH_CALUDE_earliest_retirement_year_l872_87252


namespace NUMINAMATH_CALUDE_smallest_square_area_l872_87208

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

/-- Checks if two rectangles can fit in a square of given side length without overlapping -/
def can_fit_in_square (r1 r2 : Rectangle) (side : ℝ) : Prop :=
  (min r1.width r1.height + min r2.width r2.height ≤ side) ∧
  (max r1.width r1.height + max r2.width r2.height ≤ side)

theorem smallest_square_area (r1 r2 : Rectangle) : 
  r1.width = 3 ∧ r1.height = 4 ∧ r2.width = 4 ∧ r2.height = 5 →
  ∃ (side : ℝ), 
    can_fit_in_square r1 r2 side ∧ 
    square_area side = 49 ∧
    ∀ (s : ℝ), can_fit_in_square r1 r2 s → square_area s ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l872_87208


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l872_87221

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l872_87221


namespace NUMINAMATH_CALUDE_second_tank_fish_length_is_two_l872_87226

/-- Represents the fish tank system with given conditions -/
structure FishTankSystem where
  first_tank_size : ℝ
  second_tank_size : ℝ
  first_tank_water : ℝ
  first_tank_fish_length : ℝ
  fish_difference_after_eating : ℕ
  (size_relation : first_tank_size = 2 * second_tank_size)
  (first_tank_water_amount : first_tank_water = 48)
  (first_tank_fish_size : first_tank_fish_length = 3)
  (fish_difference : fish_difference_after_eating = 3)

/-- The length of fish in the second tank -/
def second_tank_fish_length (system : FishTankSystem) : ℝ :=
  2

/-- Theorem stating that the length of fish in the second tank is 2 inches -/
theorem second_tank_fish_length_is_two (system : FishTankSystem) :
  second_tank_fish_length system = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_tank_fish_length_is_two_l872_87226


namespace NUMINAMATH_CALUDE_fraction_product_equals_reciprocal_of_2835_l872_87214

theorem fraction_product_equals_reciprocal_of_2835 :
  (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) * (1 / 7 : ℚ) = 1 / 2835 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_reciprocal_of_2835_l872_87214


namespace NUMINAMATH_CALUDE_pepperjack_probability_l872_87212

/-- The probability of picking a pepperjack cheese stick from a pack containing
    15 cheddar, 30 mozzarella, and 45 pepperjack sticks is 50%. -/
theorem pepperjack_probability (cheddar mozzarella pepperjack : ℕ) 
    (h1 : cheddar = 15)
    (h2 : mozzarella = 30)
    (h3 : pepperjack = 45) :
    (pepperjack : ℚ) / (cheddar + mozzarella + pepperjack) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_pepperjack_probability_l872_87212


namespace NUMINAMATH_CALUDE_parkway_fifth_grade_count_l872_87287

/-- The number of students in the fifth grade at Parkway Elementary School -/
def total_students : ℕ := sorry

/-- The number of boys in the fifth grade -/
def boys : ℕ := 312

/-- The number of students playing soccer -/
def soccer_players : ℕ := 250

/-- The percentage of soccer players who are boys -/
def boys_soccer_percentage : ℚ := 82 / 100

/-- The number of girls not playing soccer -/
def girls_not_soccer : ℕ := 63

theorem parkway_fifth_grade_count :
  total_students = 420 :=
by sorry

end NUMINAMATH_CALUDE_parkway_fifth_grade_count_l872_87287


namespace NUMINAMATH_CALUDE_projections_on_concentric_circles_imply_parallelogram_l872_87228

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in 2D space -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A quadrilateral in 2D space -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Projection of a point onto a line segment -/
def project (p : Point) (a b : Point) : Point :=
  sorry

/-- Check if four points form an inscribed quadrilateral in a circle -/
def is_inscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

/-- Check if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Main theorem -/
theorem projections_on_concentric_circles_imply_parallelogram 
  (q : Quadrilateral) (p1 p2 : Point) (c1 c2 : Circle) :
  c1.center = c2.center →
  c1.radius ≠ c2.radius →
  is_inscribed (Quadrilateral.mk 
    (project p1 q.a q.b) (project p1 q.b q.c) 
    (project p1 q.c q.d) (project p1 q.d q.a)) c1 →
  is_inscribed (Quadrilateral.mk 
    (project p2 q.a q.b) (project p2 q.b q.c) 
    (project p2 q.c q.d) (project p2 q.d q.a)) c2 →
  is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_projections_on_concentric_circles_imply_parallelogram_l872_87228


namespace NUMINAMATH_CALUDE_trapezoid_area_l872_87246

/-- Given an outer equilateral triangle with area 36, an inner equilateral triangle
    with area 4, and three congruent trapezoids between them, the area of one
    trapezoid is 32/3. -/
theorem trapezoid_area
  (outer_triangle_area : ℝ)
  (inner_triangle_area : ℝ)
  (num_trapezoids : ℕ)
  (h1 : outer_triangle_area = 36)
  (h2 : inner_triangle_area = 4)
  (h3 : num_trapezoids = 3) :
  (outer_triangle_area - inner_triangle_area) / num_trapezoids = 32 / 3 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l872_87246
