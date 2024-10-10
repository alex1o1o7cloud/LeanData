import Mathlib

namespace ceiling_minus_y_l1440_144075

theorem ceiling_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - (y - ⌊y⌋) := by
  sorry

end ceiling_minus_y_l1440_144075


namespace skylar_donation_l1440_144087

/-- Calculates the total donation amount given starting age, current age, and annual donation amount. -/
def totalDonation (startAge currentAge annualDonation : ℕ) : ℕ :=
  (currentAge - startAge) * annualDonation

/-- Proves that Skylar's total donation is 100k -/
theorem skylar_donation :
  let startAge : ℕ := 13
  let currentAge : ℕ := 33
  let annualDonation : ℕ := 5000
  totalDonation startAge currentAge annualDonation = 100000 := by
  sorry

end skylar_donation_l1440_144087


namespace students_present_l1440_144059

theorem students_present (total : ℕ) (absent_percent : ℚ) : 
  total = 100 → absent_percent = 14/100 → 
  (total : ℚ) * (1 - absent_percent) = 86 := by
  sorry

end students_present_l1440_144059


namespace non_equilateral_triangle_coverage_l1440_144051

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  sorry

-- Define coverage of a triangle by two other triangles
def covers (t1 t2 t : Triangle) : Prop :=
  sorry

-- Define non-equilateral triangle
def nonEquilateral (t : Triangle) : Prop :=
  sorry

-- Define smaller triangle
def smaller (t1 t2 : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem non_equilateral_triangle_coverage (t : Triangle) :
  nonEquilateral t →
  ∃ (t1 t2 : Triangle), smaller t1 t ∧ smaller t2 t ∧ similar t1 t ∧ similar t2 t ∧ covers t1 t2 t :=
sorry

end non_equilateral_triangle_coverage_l1440_144051


namespace chestnut_collection_l1440_144039

/-- The chestnut collection problem -/
theorem chestnut_collection
  (a b c : ℝ)
  (mean_ab_c : (a + b) / 2 = c + 10)
  (mean_ac_b : (a + c) / 2 = b - 3) :
  (b + c) / 2 - a = -7 := by
  sorry

end chestnut_collection_l1440_144039


namespace closest_axis_of_symmetry_l1440_144012

theorem closest_axis_of_symmetry (ω : ℝ) (h1 : 0 < ω) (h2 : ω < π) :
  let f := fun x ↦ Real.sin (ω * x + 5 * π / 6)
  (f 0 = 1 / 2) →
  (f (1 / 2) = 0) →
  (∃ k : ℤ, -1 = 3 * k - 1 ∧ 
    ∀ m : ℤ, m ≠ k → |3 * m - 1| > |3 * k - 1|) :=
by sorry

end closest_axis_of_symmetry_l1440_144012


namespace train_length_calculation_l1440_144086

-- Define the given constants
def train_speed : Real := 72 -- km/hr
def bridge_length : Real := 142 -- meters
def crossing_time : Real := 12.598992080633549 -- seconds

-- Define the theorem
theorem train_length_calculation :
  let speed_ms : Real := train_speed * 1000 / 3600
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 110 := by sorry

end train_length_calculation_l1440_144086


namespace number_ordering_l1440_144049

theorem number_ordering : 
  let a := Real.log 0.32
  let b := Real.log 0.33
  let c := 20.3
  let d := 0.32
  b < a ∧ a < d ∧ d < c := by sorry

end number_ordering_l1440_144049


namespace gcd_18_30_l1440_144028

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l1440_144028


namespace sum_product_bound_l1440_144025

theorem sum_product_bound (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α*β + β*γ + γ*α ∧ α*β + β*γ + γ*α ≤ 1 := by
  sorry

end sum_product_bound_l1440_144025


namespace perpendicular_lines_l1440_144098

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 ∧ 4 * y + b * x - 6 = 0 → 
   (2 / 3) * (b / 4) = 1) → 
  b = -6 := by
sorry

end perpendicular_lines_l1440_144098


namespace add_base_seven_example_l1440_144026

/-- Represents a number in base 7 --/
def BaseSevenNum (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Addition in base 7 --/
def addBaseSeven (a b : List Nat) : List Nat :=
  sorry

theorem add_base_seven_example :
  addBaseSeven [2, 1] [2, 5, 4] = [5, 0, 5] :=
by sorry

end add_base_seven_example_l1440_144026


namespace ellipse_major_axis_length_l1440_144027

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The major axis length of the ellipse is 7.2 -/
theorem ellipse_major_axis_length :
  major_axis_length 2 0.8 = 7.2 := by sorry

end ellipse_major_axis_length_l1440_144027


namespace red_peaches_count_l1440_144060

theorem red_peaches_count (green_peaches : ℕ) (red_peaches : ℕ) : 
  green_peaches = 16 → red_peaches = green_peaches + 1 → red_peaches = 17 := by
  sorry

end red_peaches_count_l1440_144060


namespace min_value_expression_l1440_144004

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 2)^2 + (z/y - 2)^2 + (5/z - 2)^2 ≥ 4 * (Real.rpow 5 (1/4) - 2)^2 := by
  sorry

end min_value_expression_l1440_144004


namespace altitude_sum_less_than_side_sum_l1440_144005

/-- Triangle structure with sides and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_positive : 0 < h_a ∧ 0 < h_b ∧ 0 < h_c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The sum of altitudes is less than the sum of sides in any triangle -/
theorem altitude_sum_less_than_side_sum (t : Triangle) :
  t.h_a + t.h_b + t.h_c < t.a + t.b + t.c := by
  sorry

end altitude_sum_less_than_side_sum_l1440_144005


namespace sector_area_l1440_144007

/-- Given a circular sector with circumference 8 and central angle 2 radians, its area is 4. -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) :
  circumference = 8 →
  central_angle = 2 →
  area = (1/2) * central_angle * ((circumference - central_angle) / 2)^2 →
  area = 4 := by
  sorry


end sector_area_l1440_144007


namespace intersection_A_complement_B_l1440_144080

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l1440_144080


namespace stratified_sampling_theorem_l1440_144050

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : Nat
  middleAged : Nat
  elderly : Nat

/-- Calculates the total number of employees -/
def totalEmployees (ec : EmployeeCount) : Nat :=
  ec.young + ec.middleAged + ec.elderly

/-- Represents the sample size for each age group -/
structure SampleSize where
  young : Nat
  middleAged : Nat
  elderly : Nat

/-- Calculates the total sample size -/
def totalSampleSize (ss : SampleSize) : Nat :=
  ss.young + ss.middleAged + ss.elderly

theorem stratified_sampling_theorem (ec : EmployeeCount) (ss : SampleSize) :
  totalEmployees ec = 750 ∧
  ec.young = 350 ∧
  ec.middleAged = 250 ∧
  ec.elderly = 150 ∧
  ss.young = 7 →
  totalSampleSize ss = 15 :=
sorry


end stratified_sampling_theorem_l1440_144050


namespace worker_completion_time_l1440_144046

/-- Given two workers A and B, where A is thrice as fast as B, 
    and together they can complete a job in 18 days,
    prove that A alone can complete the job in 24 days. -/
theorem worker_completion_time 
  (speed_A speed_B : ℝ) 
  (combined_time : ℝ) :
  speed_A = 3 * speed_B →
  1 / speed_A + 1 / speed_B = 1 / combined_time →
  combined_time = 18 →
  1 / speed_A = 1 / 24 :=
by sorry

end worker_completion_time_l1440_144046


namespace perpendicular_lines_from_perpendicular_planes_l1440_144064

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the non-coincident property for lines
variable (non_coincident_lines : Line → Line → Prop)

-- Define the non-coincident property for planes
variable (non_coincident_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (a b : Line) (α β : Plane)
  (h_non_coincident_lines : non_coincident_lines a b)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h_a_perp_α : perp_line_plane a α)
  (h_b_perp_β : perp_line_plane b β)
  (h_α_perp_β : perp_plane_plane α β) :
  perp_line_line a b :=
sorry

end perpendicular_lines_from_perpendicular_planes_l1440_144064


namespace percentage_calculation_l1440_144009

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 150 → 
  (3 / 5) * N = 90 → 
  (P / 100) * 90 = 36 → 
  P = 40 := by
sorry

end percentage_calculation_l1440_144009


namespace base_salary_per_week_l1440_144019

def past_week_incomes : List ℝ := [406, 413, 420, 436, 395]
def num_past_weeks : ℕ := 5
def num_future_weeks : ℕ := 2
def total_weeks : ℕ := num_past_weeks + num_future_weeks
def average_commission_future : ℝ := 345
def average_weekly_income : ℝ := 500

def total_past_income : ℝ := past_week_incomes.sum
def total_income : ℝ := average_weekly_income * total_weeks
def total_future_income : ℝ := total_income - total_past_income
def total_future_commission : ℝ := average_commission_future * num_future_weeks
def total_future_base_salary : ℝ := total_future_income - total_future_commission

theorem base_salary_per_week : 
  total_future_base_salary / num_future_weeks = 370 := by sorry

end base_salary_per_week_l1440_144019


namespace counterexample_exists_l1440_144095

theorem counterexample_exists : ∃ (a b : ℕ), 
  (∃ (k : ℕ), a^7 = b^3 * k) ∧ 
  ¬(∃ (m : ℕ), a^2 = b * m) := by
  sorry

end counterexample_exists_l1440_144095


namespace multiply_mixed_number_l1440_144013

theorem multiply_mixed_number : (7 : ℚ) * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l1440_144013


namespace max_popsicles_for_eight_dollars_l1440_144036

/-- Represents the number of popsicles in a box -/
inductive BoxSize
  | Single : BoxSize
  | Three : BoxSize
  | Five : BoxSize

/-- Returns the cost of a box given its size -/
def boxCost (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 2
  | BoxSize.Five => 3

/-- Returns the number of popsicles in a box given its size -/
def boxCount (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 3
  | BoxSize.Five => 5

/-- Represents a purchase of popsicle boxes -/
structure Purchase where
  singles : ℕ
  threes : ℕ
  fives : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.singles * boxCost BoxSize.Single +
  p.threes * boxCost BoxSize.Three +
  p.fives * boxCost BoxSize.Five

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.singles * boxCount BoxSize.Single +
  p.threes * boxCount BoxSize.Three +
  p.fives * boxCount BoxSize.Five

/-- Theorem: The maximum number of popsicles that can be purchased with $8 is 13 -/
theorem max_popsicles_for_eight_dollars :
  (∃ p : Purchase, totalCost p = 8 ∧ totalPopsicles p = 13) ∧
  (∀ p : Purchase, totalCost p ≤ 8 → totalPopsicles p ≤ 13) := by
  sorry

end max_popsicles_for_eight_dollars_l1440_144036


namespace three_intersections_iff_a_in_open_interval_l1440_144082

/-- The function f(x) = x^3 - 3x -/
def f (x : ℝ) : ℝ := x^3 - 3*x

/-- The proposition that the line y = a intersects the graph of f(x) at three distinct points -/
def has_three_distinct_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = a ∧ f x₂ = a ∧ f x₃ = a

/-- The theorem stating that the line y = a intersects the graph of f(x) at three distinct points
    if and only if a is in the open interval (-2, 2) -/
theorem three_intersections_iff_a_in_open_interval :
  ∀ a : ℝ, has_three_distinct_intersections a ↔ -2 < a ∧ a < 2 :=
sorry

end three_intersections_iff_a_in_open_interval_l1440_144082


namespace barrel_leak_percentage_l1440_144058

theorem barrel_leak_percentage (initial_volume : ℝ) (remaining_volume : ℝ) : 
  initial_volume = 220 →
  remaining_volume = 198 →
  (initial_volume - remaining_volume) / initial_volume * 100 = 10 := by
sorry

end barrel_leak_percentage_l1440_144058


namespace apples_processed_equals_stems_l1440_144099

/-- A machine that processes apples and cuts stems -/
structure AppleProcessor where
  stems_after_2_hours : ℕ
  apples_processed : ℕ

/-- The number of stems after 2 hours is equal to the number of apples processed -/
axiom stems_equal_apples (m : AppleProcessor) : m.stems_after_2_hours = m.apples_processed

/-- Theorem: The number of apples processed is equal to the number of stems observed after 2 hours -/
theorem apples_processed_equals_stems (m : AppleProcessor) :
  m.apples_processed = m.stems_after_2_hours := by sorry

end apples_processed_equals_stems_l1440_144099


namespace floor_equation_solution_l1440_144093

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊2 * x⌋ - (1/2 : ℝ)⌋ = ⌊x + 3⌋ ↔ 3.5 ≤ x ∧ x < 4.5 := by sorry

end floor_equation_solution_l1440_144093


namespace intersection_of_complex_circles_l1440_144035

theorem intersection_of_complex_circles (k : ℝ) :
  (∃! z : ℂ, Complex.abs (z - 3) = 2 * Complex.abs (z + 3) ∧ Complex.abs z = k) →
  k = 1 ∨ k = 9 :=
sorry

end intersection_of_complex_circles_l1440_144035


namespace prop_2_prop_3_l1440_144053

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (containedIn : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem prop_2 
  (m : Line) (α β : Plane) 
  (h1 : planeParallel α β) 
  (h2 : containedIn m α) : 
  parallel m β :=
sorry

-- Theorem for proposition ③
theorem prop_3 
  (m n : Line) (α β : Plane)
  (h1 : perpendicular n α)
  (h2 : perpendicular n β)
  (h3 : perpendicular m α) :
  perpendicular m β :=
sorry

end prop_2_prop_3_l1440_144053


namespace stewart_farm_horse_food_l1440_144085

theorem stewart_farm_horse_food (sheep_count : ℕ) (total_horse_food : ℕ) 
  (sheep_to_horse_ratio : ℚ) :
  sheep_count = 48 →
  total_horse_food = 12880 →
  sheep_to_horse_ratio = 6 / 7 →
  (total_horse_food / (sheep_count * (1 / sheep_to_horse_ratio))) = 230 := by
  sorry

end stewart_farm_horse_food_l1440_144085


namespace calculation_proof_inequalities_solution_l1440_144030

-- Problem 1
theorem calculation_proof :
  Real.pi ^ 0 + |3 - Real.sqrt 2| - (1/3)⁻¹ = 1 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem inequalities_solution (x : ℝ) :
  (2*x > x - 2 ∧ x + 1 < 2) ↔ (-2 < x ∧ x < 1) := by
  sorry

end calculation_proof_inequalities_solution_l1440_144030


namespace no_zero_root_l1440_144081

theorem no_zero_root : 
  (∀ x : ℝ, 4 * x^2 - 3 = 49 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 2)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - x - 20 = 0 → x ≠ 0) := by
  sorry

end no_zero_root_l1440_144081


namespace ellipse_min_sum_l1440_144077

/-- Given an ellipse x²/m² + y²/n² = 1 passing through point P(a, b),
    prove that the minimum value of m + n is (a²/³ + b²/³)¹/³ -/
theorem ellipse_min_sum (a b m n : ℝ) (hm : m > 0) (hn : n > 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : abs a ≠ abs b)
  (h_ellipse : a^2 / m^2 + b^2 / n^2 = 1) :
  ∃ (min_sum : ℝ), min_sum = (a^(2/3) + b^(2/3))^(1/3) ∧
    ∀ (m' n' : ℝ), m' > 0 → n' > 0 → a^2 / m'^2 + b^2 / n'^2 = 1 →
      m' + n' ≥ min_sum :=
sorry

end ellipse_min_sum_l1440_144077


namespace perfect_square_implies_zero_a_l1440_144029

theorem perfect_square_implies_zero_a (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) → a = 0 := by
  sorry

end perfect_square_implies_zero_a_l1440_144029


namespace f_properties_l1440_144079

def f (x : ℝ) := x^3 - 12*x

theorem f_properties :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧ 
  (∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y) ∧ 
  (∀ x y, 2 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ f (-2)) ∧
  (∀ x, f 2 ≤ f x) ∧
  (f (-2) = 16) ∧
  (f 2 = -16) :=
sorry

end f_properties_l1440_144079


namespace probability_is_two_thirds_l1440_144084

/-- Given four evenly spaced points A, B, C, D on a number line with an interval of 1,
    this function calculates the probability that a randomly chosen point E on AD
    has a sum of distances to B and C less than 2. -/
def probability_sum_distances_less_than_two (A B C D : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the probability is 2/3 -/
theorem probability_is_two_thirds (A B C D : ℝ) 
  (h1 : B - A = 1) 
  (h2 : C - B = 1) 
  (h3 : D - C = 1) : 
  probability_sum_distances_less_than_two A B C D = 2/3 :=
sorry

end probability_is_two_thirds_l1440_144084


namespace polynomial_divisibility_l1440_144066

/-- A polynomial of degree 3 with a parameter k -/
def f (k : ℝ) (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + k * x - 12

/-- The divisor x - 3 -/
def g (x : ℝ) : ℝ := x - 3

/-- The potential divisor 3x^2 + 4 -/
def h (x : ℝ) : ℝ := 3 * x^2 + 4

theorem polynomial_divisibility (k : ℝ) :
  (∃ q : ℝ → ℝ, ∀ x, f k x = g x * q x) →
  (∃ r : ℝ → ℝ, ∀ x, f k x = h x * r x) :=
by sorry

end polynomial_divisibility_l1440_144066


namespace chord_rotation_in_unit_circle_l1440_144073

/-- Chord rotation in a unit circle -/
theorem chord_rotation_in_unit_circle :
  -- Define the circle
  let circle_radius : ℝ := 1
  -- Define the chord length (side of inscribed equilateral triangle)
  let chord_length : ℝ := Real.sqrt 3
  -- Define the rotation angle (90 degrees in radians)
  let rotation_angle : ℝ := π / 2
  -- Define the area of the full circle
  let circle_area : ℝ := π * circle_radius ^ 2

  -- Statement 1: Area swept by chord during 90° rotation
  let area_swept : ℝ := (7 * π / 16) - 1 / 4

  -- Statement 2: Angle to sweep half of circle's area
  let angle_half_area : ℝ := (4 * π + 6 * Real.sqrt 3) / 9

  -- Prove the following:
  True →
    -- 1. The area swept by the chord during a 90° rotation
    (area_swept = (7 * π / 16) - 1 / 4) ∧
    -- 2. The angle required to sweep exactly half of the circle's area
    (angle_half_area = (4 * π + 6 * Real.sqrt 3) / 9) ∧
    -- Additional verification: the swept area at angle_half_area is indeed half the circle's area
    (2 * ((angle_half_area / (2 * π)) * circle_area - 
     (Real.sqrt (1 - (chord_length / 2) ^ 2) * (chord_length / 2))) = circle_area) :=
by
  sorry

end chord_rotation_in_unit_circle_l1440_144073


namespace max_vertical_distance_is_sqrt2_over_2_l1440_144037

/-- Represents a square with side length 1 inch -/
structure UnitSquare where
  center : ℝ × ℝ

/-- Represents the configuration of four squares -/
structure SquareConfiguration where
  squares : List UnitSquare
  rotated_square : UnitSquare

/-- The maximum vertical distance from the original line to any point on the rotated square -/
def max_vertical_distance (config : SquareConfiguration) : ℝ :=
  sorry

/-- Theorem stating the maximum vertical distance is √2/2 -/
theorem max_vertical_distance_is_sqrt2_over_2 (config : SquareConfiguration) :
  max_vertical_distance config = Real.sqrt 2 / 2 :=
sorry

end max_vertical_distance_is_sqrt2_over_2_l1440_144037


namespace max_m_value_l1440_144067

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 2/a + 1/b = 1/4) (h_ineq : ∀ m : ℝ, 2*a + b ≥ 4*m) : 
  ∃ m_max : ℝ, m_max = 9 ∧ ∀ m : ℝ, (∀ x : ℝ, 2*a + b ≥ 4*x → m ≤ x) → m ≤ m_max :=
sorry

end max_m_value_l1440_144067


namespace fathers_age_multiplier_l1440_144072

theorem fathers_age_multiplier (father_age son_age : ℕ) (h_sum : father_age + son_age = 75)
  (h_son : son_age = 27) (h_father : father_age = 48) :
  ∃ (M : ℕ), M * (son_age - (father_age - son_age)) = father_age ∧ M = 8 := by
  sorry

end fathers_age_multiplier_l1440_144072


namespace exactly_two_valid_numbers_l1440_144015

def is_valid_number (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 6 ∧
  digits.toFinset = {1, 2, 3, 4, 5, 6} ∧
  (n / 10000) % 2 = 0 ∧
  (n / 1000) % 3 = 0 ∧
  (n / 100) % 4 = 0 ∧
  (n / 10) % 5 = 0 ∧
  n % 6 = 0

theorem exactly_two_valid_numbers : 
  ∃! (s : Finset Nat), s.card = 2 ∧ ∀ n ∈ s, is_valid_number n :=
sorry

end exactly_two_valid_numbers_l1440_144015


namespace mass_of_third_metal_l1440_144018

/-- Given an alloy of four metals with specific mass ratios, prove the mass of the third metal -/
theorem mass_of_third_metal (m₁ m₂ m₃ m₄ : ℝ) 
  (h_total : m₁ + m₂ + m₃ + m₄ = 25)
  (h_ratio1 : m₁ = 1.5 * m₂)
  (h_ratio2 : m₂ / m₃ = 3 / 4)
  (h_ratio3 : m₃ / m₄ = 5 / 6) :
  m₃ = 375 / 78 := by
  sorry

#check mass_of_third_metal

end mass_of_third_metal_l1440_144018


namespace wind_velocity_problem_l1440_144091

/-- Represents the relationship between pressure, area, and velocity -/
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

theorem wind_velocity_problem (k : ℝ) :
  pressure_relation k 2 8 = 4 →
  pressure_relation k 4.5 (40/3) = 25 :=
by sorry

end wind_velocity_problem_l1440_144091


namespace winnie_balloon_distribution_l1440_144022

theorem winnie_balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 220) (h2 : num_friends = 9) :
  total_balloons % num_friends = 4 := by sorry

end winnie_balloon_distribution_l1440_144022


namespace smallest_digit_divisible_by_11_l1440_144070

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def digit_sum_odd (n : ℕ) : ℕ :=
  (n / 10000000) + ((n / 100000) % 10) + ((n / 1000) % 10) + ((n / 10) % 10)

def digit_sum_even (n : ℕ) : ℕ :=
  ((n / 1000000) % 10) + ((n / 10000) % 10) + ((n / 100) % 10) + (n % 10)

theorem smallest_digit_divisible_by_11 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_11 (85210000 + d * 1000 + 784) ↔ d = 1) ∧
    (∀ d' : ℕ, d' < d → ¬is_divisible_by_11 (85210000 + d' * 1000 + 784)) :=
by sorry

end smallest_digit_divisible_by_11_l1440_144070


namespace bridge_arch_ratio_l1440_144034

-- Define the variables for the original design
variable (r₁ : ℝ) -- radius of the original design
variable (v₁ : ℝ) -- height of the arch in the original design

-- Define the variables for the built bridge
variable (r₂ : ℝ) -- radius of the built bridge
variable (v₂ : ℝ) -- height of the arch in the built bridge

-- Define the length of the bridge
variable (l : ℝ)

-- State the theorem
theorem bridge_arch_ratio :
  (v₁ = 3 * v₂) → -- Condition 1
  (r₂ = 2 * r₁) → -- Condition 2
  (2 * r₁ * v₁ - v₁^2 = 2 * r₂ * v₂ - v₂^2) → -- Condition 3 (constant bridge length)
  (v₁ / r₁ = 3 / 4) ∧ (v₂ / r₂ = 1 / 8) := by
  sorry


end bridge_arch_ratio_l1440_144034


namespace jacobs_february_bill_l1440_144024

/-- Calculates the total cell phone bill given the plan details and usage --/
def calculate_bill (base_cost : ℚ) (included_hours : ℚ) (cost_per_text : ℚ) 
  (cost_per_extra_minute : ℚ) (texts_sent : ℚ) (hours_talked : ℚ) : ℚ :=
  let text_cost := texts_sent * cost_per_text
  let extra_hours := max (hours_talked - included_hours) 0
  let extra_minutes := extra_hours * 60
  let extra_cost := extra_minutes * cost_per_extra_minute
  base_cost + text_cost + extra_cost

/-- Theorem stating that Jacob's cell phone bill for February is $83.80 --/
theorem jacobs_february_bill :
  calculate_bill 25 25 0.08 0.13 150 31 = 83.80 := by
  sorry

end jacobs_february_bill_l1440_144024


namespace arithmetic_sequence_property_l1440_144057

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_property (b : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence b)
  (h_incr : ∀ n : ℕ, b n < b (n + 1))
  (h_prod : b 4 * b 7 = 24) :
  b 3 * b 8 = 200 / 9 := by
sorry

end arithmetic_sequence_property_l1440_144057


namespace five_books_three_bins_l1440_144090

-- Define the Stirling number of the second kind
def stirling2 (n k : ℕ) : ℕ := sorry

-- State the theorem
theorem five_books_three_bins : stirling2 5 3 = 25 := by sorry

end five_books_three_bins_l1440_144090


namespace box_volume_count_l1440_144078

theorem box_volume_count : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun x => x > 2 ∧ (x + 3) * (x - 2) * (x^2 + 10) < 500) 
    (Finset.range 100)).card :=
by
  sorry

end box_volume_count_l1440_144078


namespace line_x_coordinate_indeterminate_l1440_144052

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Given a line passing through (4, 0), (x₁, 3), and (-12, y₂), 
    prove that x₁ cannot be uniquely determined --/
theorem line_x_coordinate_indeterminate 
  (line : Line)
  (h1 : line.slope * 4 + line.y_intercept = 0)
  (h2 : ∃ x₁, line.slope * x₁ + line.y_intercept = 3)
  (h3 : ∃ y₂, line.slope * (-12) + line.y_intercept = y₂) :
  ¬(∃! x₁, line.slope * x₁ + line.y_intercept = 3) :=
sorry

end line_x_coordinate_indeterminate_l1440_144052


namespace tetrahedron_volume_l1440_144016

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume (S₁ S₂ S₃ S₄ r : ℝ) (h₁ : 0 < S₁) (h₂ : 0 < S₂) (h₃ : 0 < S₃) (h₄ : 0 < S₄) (hr : 0 < r) :
  ∃ V : ℝ, V = (1 / 3) * (S₁ + S₂ + S₃ + S₄) * r ∧ V > 0 :=
by sorry

end tetrahedron_volume_l1440_144016


namespace calculation_proof_l1440_144089

theorem calculation_proof : 70 + 5 * 12 / (180 / 3) = 71 := by
  sorry

end calculation_proof_l1440_144089


namespace min_value_expression_min_value_attained_l1440_144043

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) ≥ 16 := by
  sorry

theorem min_value_attained (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  ∃ p q r, 4 * p^3 + 6 * q^3 + 24 * r^3 + 8 / (3 * p * q * r) = 16 := by
  sorry

end min_value_expression_min_value_attained_l1440_144043


namespace equation_one_solution_l1440_144076

theorem equation_one_solution (x : ℝ) : 3 * x * (x - 1) = 1 - x → x = 1 ∨ x = -1/3 := by
  sorry

end equation_one_solution_l1440_144076


namespace fixed_point_on_tangency_line_l1440_144008

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency types
inductive TangencyType
  | ExternalExternal
  | ExternalInternal
  | InternalExternal
  | InternalInternal

-- Define the similarity point
def similarityPoint (k₁ k₂ : Circle) (t : TangencyType) : ℝ × ℝ :=
  sorry

-- Define the line connecting tangency points
def tangencyLine (k k₁ k₂ : Circle) : Set (ℝ × ℝ) :=
  sorry

-- Main theorem
theorem fixed_point_on_tangency_line
  (k₁ k₂ : Circle)
  (h : k₁.radius ≠ k₂.radius)
  (t : TangencyType) :
  ∃ (p : ℝ × ℝ), ∀ (k : Circle),
    p ∈ tangencyLine k k₁ k₂ ∧ p = similarityPoint k₁ k₂ t :=
  sorry

end fixed_point_on_tangency_line_l1440_144008


namespace fair_game_conditions_l1440_144088

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  total : Nat
  white : Nat
  green : Nat
  black : Nat

/-- Checks if the game is fair given the bag contents -/
def isFairGame (bag : BagContents) : Prop :=
  bag.green = bag.black

/-- Theorem stating the conditions for a fair game -/
theorem fair_game_conditions (x : Nat) :
  let bag : BagContents := {
    total := 15,
    white := x,
    green := 2 * x,
    black := 15 - x - 2 * x
  }
  isFairGame bag ↔ x = 3 := by
  sorry

end fair_game_conditions_l1440_144088


namespace min_movie_audience_l1440_144083

/-- Represents the number of people in the movie theater -/
structure MovieTheater where
  adults : ℕ
  children : ℕ

/-- Conditions for the movie theater audience -/
class MovieTheaterConditions (t : MovieTheater) where
  adult_men : t.adults * 4 = t.adults * 5
  male_children : t.children * 2 = t.adults * 2
  boy_children : t.children * 1 = t.children * 5

/-- The theorem stating the minimum number of people in the movie theater -/
theorem min_movie_audience (t : MovieTheater) [MovieTheaterConditions t] :
  t.adults + t.children ≥ 55 := by
  sorry

#check min_movie_audience

end min_movie_audience_l1440_144083


namespace correct_change_marys_change_l1440_144010

def change_calculation (cost_berries : ℚ) (cost_peaches : ℚ) (amount_paid : ℚ) : ℚ :=
  amount_paid - (cost_berries + cost_peaches)

theorem correct_change (cost_berries cost_peaches amount_paid : ℚ) :
  change_calculation cost_berries cost_peaches amount_paid =
  amount_paid - (cost_berries + cost_peaches) :=
by
  sorry

-- Example with Mary's specific values
theorem marys_change :
  change_calculation 7.19 6.83 20 = 5.98 :=
by
  sorry

end correct_change_marys_change_l1440_144010


namespace cubic_parabola_x_intercepts_l1440_144068

theorem cubic_parabola_x_intercepts :
  ∃! x : ℝ, x = -3 * 0^3 + 2 * 0^2 - 0 + 2 :=
sorry

end cubic_parabola_x_intercepts_l1440_144068


namespace frictional_force_is_10N_l1440_144094

/-- The acceleration due to gravity (m/s²) -/
def g : ℝ := 9.8

/-- Mass of the tank (kg) -/
def m₁ : ℝ := 2

/-- Mass of the cart (kg) -/
def m₂ : ℝ := 10

/-- Acceleration of the cart (m/s²) -/
def a : ℝ := 5

/-- Coefficient of friction between the tank and cart -/
def μ : ℝ := 0.6

/-- The frictional force acting on the tank from the cart (N) -/
def frictional_force : ℝ := m₁ * a

theorem frictional_force_is_10N : frictional_force = 10 := by
  sorry

#check frictional_force_is_10N

end frictional_force_is_10N_l1440_144094


namespace revenue_change_after_price_and_sales_change_l1440_144000

theorem revenue_change_after_price_and_sales_change
  (initial_price initial_sales : ℝ)
  (price_increase : ℝ)
  (sales_decrease : ℝ)
  (h1 : price_increase = 0.3)
  (h2 : sales_decrease = 0.2)
  : (((initial_price * (1 + price_increase)) * (initial_sales * (1 - sales_decrease)) - initial_price * initial_sales) / (initial_price * initial_sales)) * 100 = 4 := by
sorry

end revenue_change_after_price_and_sales_change_l1440_144000


namespace quadratic_equation_roots_l1440_144047

theorem quadratic_equation_roots (a b c : ℤ) : 
  a ≠ 0 → 
  (∃ x : ℚ, a * x^2 + b * x + c = 0) → 
  ¬(Odd a ∧ Odd b ∧ Odd c) :=
by sorry

end quadratic_equation_roots_l1440_144047


namespace ellipse_intersection_range_l1440_144006

-- Define the ellipse G
def G (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for point M
def M_condition (m : ℝ) (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  (xA - m)^2 + yA^2 = (xB - m)^2 + yB^2

-- Main theorem
theorem ellipse_intersection_range :
  ∀ (k : ℝ) (A B : ℝ × ℝ) (m : ℝ),
  (∃ (xA yA xB yB : ℝ), A = (xA, yA) ∧ B = (xB, yB) ∧
    G xA yA ∧ G xB yB ∧
    line k xA yA ∧ line k xB yB ∧
    A ≠ B ∧
    M_condition m A B) →
  m ∈ Set.Icc (- Real.sqrt 6 / 12) (Real.sqrt 6 / 12) :=
by sorry

end ellipse_intersection_range_l1440_144006


namespace pq_length_l1440_144001

/-- Two similar triangles PQR and STU with given side lengths and angles -/
structure SimilarTriangles where
  -- Side lengths of triangle PQR
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  -- Side lengths of triangle STU
  ST : ℝ
  TU : ℝ
  SU : ℝ
  -- Angles
  angle_P : ℝ
  angle_S : ℝ
  -- Conditions
  h1 : angle_P = 120
  h2 : angle_S = 120
  h3 : PR = 15
  h4 : SU = 15
  h5 : ST = 4.5
  h6 : TU = 10.5

/-- The length of PQ in similar triangles PQR and STU is 9 -/
theorem pq_length (t : SimilarTriangles) : t.PQ = 9 := by
  sorry

end pq_length_l1440_144001


namespace scientific_notation_of_1_3_million_l1440_144074

theorem scientific_notation_of_1_3_million :
  1300000 = 1.3 * (10 : ℝ)^6 := by sorry

end scientific_notation_of_1_3_million_l1440_144074


namespace triangle_yz_length_l1440_144044

/-- Given a triangle XYZ where cos(2X-Z) + sin(X+Y) = 2 and XY = 6, prove that YZ = 6√2 -/
theorem triangle_yz_length (X Y Z : Real) (h1 : Real.cos (2*X - Z) + Real.sin (X + Y) = 2) 
  (h2 : 0 < X ∧ X < π) (h3 : 0 < Y ∧ Y < π) (h4 : 0 < Z ∧ Z < π) 
  (h5 : X + Y + Z = π) (h6 : XY = 6) : 
  let YZ := Real.sqrt ((XY^2) * 2)
  YZ = 6 * Real.sqrt 2 := by
  sorry

end triangle_yz_length_l1440_144044


namespace fortran_program_141_l1440_144017

/-- Calculates the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ := sorry

/-- Generates the next number in the sequence -/
def nextNumber (n : ℕ) : ℕ := sorry

/-- Represents the sequence of numbers generated by the process -/
def numberSequence (start : ℕ) : ℕ → ℕ := sorry

theorem fortran_program_141 :
  ∀ k : ℕ, k > 0 → numberSequence 141 k ≠ 141 := by sorry

end fortran_program_141_l1440_144017


namespace karen_donald_children_count_l1440_144021

/-- Represents the number of children Karen and Donald have -/
def karen_donald_children : ℕ := sorry

/-- Represents the number of children Tom and Eva have -/
def tom_eva_children : ℕ := 4

/-- Represents the total number of legs in the pool -/
def legs_in_pool : ℕ := 16

/-- Represents the number of people not in the pool -/
def people_not_in_pool : ℕ := 6

/-- Proves that Karen and Donald have 6 children given the conditions -/
theorem karen_donald_children_count :
  karen_donald_children = 6 := by sorry

end karen_donald_children_count_l1440_144021


namespace ratio_problem_l1440_144042

theorem ratio_problem (x y z : ℚ) 
  (h1 : x / y = 4 / 7) 
  (h2 : z / x = 3 / 5) : 
  (x + y) / (z + x) = 55 / 32 := by
  sorry

end ratio_problem_l1440_144042


namespace alice_bob_distance_difference_l1440_144055

/-- The difference in distance traveled between two bikers after a given time -/
def distance_difference (speed_a : ℝ) (speed_b : ℝ) (time : ℝ) : ℝ :=
  (speed_a - speed_b) * time

/-- Theorem: Alice bikes 30 miles more than Bob after 6 hours -/
theorem alice_bob_distance_difference :
  distance_difference 15 10 6 = 30 := by
  sorry

end alice_bob_distance_difference_l1440_144055


namespace smallest_common_factor_l1440_144065

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 3 → gcd (12*m - 3) (8*m + 9) = 1) ∧ 
  gcd (12*3 - 3) (8*3 + 9) > 1 := by
  sorry

end smallest_common_factor_l1440_144065


namespace production_equation_proof_l1440_144069

/-- Represents a furniture production scenario -/
structure ProductionScenario where
  total : ℕ              -- Total sets to produce
  increase : ℕ           -- Daily production increase
  days_saved : ℕ         -- Days saved due to increase
  original_rate : ℕ      -- Original daily production rate

/-- Theorem stating the correct equation for the production scenario -/
theorem production_equation_proof (s : ProductionScenario) 
  (h1 : s.total = 540)
  (h2 : s.increase = 2)
  (h3 : s.days_saved = 3) :
  (s.total : ℝ) / s.original_rate - (s.total : ℝ) / (s.original_rate + s.increase) = s.days_saved := by
  sorry

#check production_equation_proof

end production_equation_proof_l1440_144069


namespace min_packages_required_l1440_144096

/-- Represents a floor in the apartment building -/
inductive Floor
| First
| Second
| Third

/-- Calculates the number of times a specific digit appears on a floor -/
def digit_count (floor : Floor) (digit : ℕ) : ℕ :=
  match floor with
  | Floor.First => if digit = 1 then 52 else 0
  | Floor.Second => if digit = 2 then 52 else 0
  | Floor.Third => if digit = 3 then 52 else 0

/-- Theorem stating the minimum number of packages required -/
theorem min_packages_required : 
  (∀ (floor : Floor) (digit : ℕ), digit_count floor digit ≤ 52) ∧ 
  (∃ (floor : Floor) (digit : ℕ), digit_count floor digit = 52) → 
  (∀ n : ℕ, n < 52 → ¬(∀ (floor : Floor) (digit : ℕ), digit_count floor digit ≤ n)) :=
by sorry

end min_packages_required_l1440_144096


namespace min_n_plus_d_l1440_144032

/-- An arithmetic sequence with positive integer terms -/
structure ArithmeticSequence where
  a : ℕ → ℕ
  d : ℕ
  first_term : a 1 = 1949
  nth_term : ∃ n : ℕ, a n = 2009
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- The minimum value of n + d for the given arithmetic sequence -/
theorem min_n_plus_d (seq : ArithmeticSequence) : 
  ∃ n d : ℕ, seq.d = d ∧ (∃ k, seq.a k = 2009) ∧ 
  (∀ m e : ℕ, seq.d = e ∧ (∃ j, seq.a j = 2009) → n + d ≤ m + e) ∧
  n + d = 17 := by
  sorry

end min_n_plus_d_l1440_144032


namespace kayak_production_sum_l1440_144054

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem kayak_production_sum :
  let a := 5  -- Initial production in February
  let r := 3  -- Growth ratio
  let n := 6  -- Number of months (February to July)
  geometric_sum a r n = 1820 := by
sorry

end kayak_production_sum_l1440_144054


namespace second_largest_prime_factor_of_sum_of_divisors_450_l1440_144033

def sum_of_divisors (n : ℕ) : ℕ := sorry

def second_largest_prime_factor (n : ℕ) : ℕ := sorry

theorem second_largest_prime_factor_of_sum_of_divisors_450 :
  second_largest_prime_factor (sum_of_divisors 450) = 13 := by sorry

end second_largest_prime_factor_of_sum_of_divisors_450_l1440_144033


namespace A_B_mutually_exclusive_A_C_independent_l1440_144071

-- Define the sample space
def S : Set (ℕ × ℕ) := {p | p.1 ∈ Finset.range 6 ∧ p.2 ∈ Finset.range 6}

-- Define events A, B, and C
def A : Set (ℕ × ℕ) := {p ∈ S | p.1 + p.2 = 7}
def B : Set (ℕ × ℕ) := {p ∈ S | Odd (p.1 * p.2)}
def C : Set (ℕ × ℕ) := {p ∈ S | p.1 > 3}

-- Define probability measure
noncomputable def P : Set (ℕ × ℕ) → ℝ := sorry

-- Theorem statements
theorem A_B_mutually_exclusive : A ∩ B = ∅ := by sorry

theorem A_C_independent : P (A ∩ C) = P A * P C := by sorry

end A_B_mutually_exclusive_A_C_independent_l1440_144071


namespace manicure_total_cost_l1440_144062

-- Define the cost of the manicure
def manicure_cost : ℝ := 30

-- Define the tip percentage
def tip_percentage : ℝ := 0.30

-- Define the function to calculate the total amount paid
def total_amount_paid (cost tip_percent : ℝ) : ℝ :=
  cost + (cost * tip_percent)

-- Theorem to prove
theorem manicure_total_cost :
  total_amount_paid manicure_cost tip_percentage = 39 := by
  sorry

end manicure_total_cost_l1440_144062


namespace nonzero_real_number_problem_l1440_144038

theorem nonzero_real_number_problem (x : ℝ) (h1 : x ≠ 0) :
  (x + x^2) / 2 = 5 * x → x = 9 := by
  sorry

end nonzero_real_number_problem_l1440_144038


namespace range_of_a_for_inequality_l1440_144040

theorem range_of_a_for_inequality : 
  {a : ℝ | ∃ x : ℝ, |x + 2| + |x - a| < 5} = Set.Ioo (-7 : ℝ) 3 := by
  sorry

end range_of_a_for_inequality_l1440_144040


namespace polynomial_efficient_evaluation_l1440_144063

/-- The polynomial 6x^5+5x^4+4x^3+3x^2+2x+2002 can be evaluated using 5 multiplications and 5 additions -/
theorem polynomial_efficient_evaluation :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = 6*x^5 + 5*x^4 + 4*x^3 + 3*x^2 + 2*x + 2002) ∧
    (∃ (g : ℝ → ℝ) (a b c d e : ℝ → ℝ),
      (∀ x, f x = g x + 2002) ∧
      (∀ x, g x = (((a x * x + b x) * x + c x) * x + d x) * x + e x) ∧
      (∀ x, a x = 6*x + 5) ∧
      (∀ x, b x = 4) ∧
      (∀ x, c x = 3) ∧
      (∀ x, d x = 2) ∧
      (∀ x, e x = 0)) :=
by sorry

end polynomial_efficient_evaluation_l1440_144063


namespace expression_evaluation_l1440_144041

theorem expression_evaluation :
  let x : ℤ := -1
  let y : ℤ := 2
  let expr := -2 * (-x^2*y + x*y^2) - (-3*x^2*y^2 + 3*x^2*y + (3*x^2*y^2 - 3*x*y^2))
  expr = -6 := by
  sorry

end expression_evaluation_l1440_144041


namespace parking_lot_spaces_l1440_144020

/-- Represents a parking lot with full-sized and compact car spaces. -/
structure ParkingLot where
  full_sized : ℕ
  compact : ℕ

/-- Calculates the total number of spaces in a parking lot. -/
def total_spaces (lot : ParkingLot) : ℕ :=
  lot.full_sized + lot.compact

/-- Represents the ratio of full-sized to compact car spaces. -/
structure SpaceRatio where
  full_sized : ℕ
  compact : ℕ

/-- Theorem: Given a parking lot with 330 full-sized car spaces and a ratio of 11:4
    for full-sized to compact car spaces, the total number of spaces is 450. -/
theorem parking_lot_spaces (ratio : SpaceRatio) 
    (h1 : ratio.full_sized = 11)
    (h2 : ratio.compact = 4)
    (lot : ParkingLot)
    (h3 : lot.full_sized = 330)
    (h4 : lot.full_sized * ratio.compact = lot.compact * ratio.full_sized) :
    total_spaces lot = 450 := by
  sorry

#check parking_lot_spaces

end parking_lot_spaces_l1440_144020


namespace a_5_value_l1440_144014

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem a_5_value (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 7) (h_9 : a 9 = 19) : a 5 = 11 := by
  sorry

end a_5_value_l1440_144014


namespace prob_at_least_six_heads_in_eight_flips_l1440_144011

/-- The probability of getting at least 6 heads in 8 fair coin flips -/
theorem prob_at_least_six_heads_in_eight_flips :
  let n : ℕ := 8  -- number of coin flips
  let k : ℕ := 6  -- minimum number of heads
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  Finset.sum (Finset.range (n - k + 1)) (λ i => (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - (k + i))) = 37/256 :=
by sorry

end prob_at_least_six_heads_in_eight_flips_l1440_144011


namespace absolute_value_equation_l1440_144002

theorem absolute_value_equation (x y : ℝ) : 
  |x^2 - Real.log y| = x^2 + Real.log y → x * (y - 1) = 0 := by
sorry

end absolute_value_equation_l1440_144002


namespace roots_cubic_equation_l1440_144097

theorem roots_cubic_equation (x₁ x₂ : ℝ) (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  x₁^3 - 4*x₂^2 + 19 = 0 := by
  sorry

end roots_cubic_equation_l1440_144097


namespace log_inequality_l1440_144092

theorem log_inequality : ∀ x : ℝ, x > 0 → x + 1/x > 2 := by sorry

end log_inequality_l1440_144092


namespace quadratic_inequality_solution_l1440_144031

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x : ℝ | x < 1 ∨ x > b}

-- Define the inequality function
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - (c + 2) * x + 2 * x

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ (a b : ℝ), (∀ x, f a x > 0 ↔ x ∈ solution_set a b) →
    (a = 1 ∧ b = 2) ∧
    (∀ c : ℝ,
      (c > 0 → {x | g c x < 0} = Set.Ioo 0 c) ∧
      (c = 0 → {x | g c x < 0} = ∅) ∧
      (c < 0 → {x | g c x < 0} = Set.Ioo c 0)) :=
by sorry

end quadratic_inequality_solution_l1440_144031


namespace sum_b_plus_d_l1440_144048

theorem sum_b_plus_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 42) 
  (h2 : a + c = 7) : 
  b + d = 6 := by
sorry

end sum_b_plus_d_l1440_144048


namespace probability_both_primary_l1440_144056

/-- Represents the types of schools in the area -/
inductive SchoolType
| Primary
| Middle
| University

/-- Represents the total number of schools of each type -/
def totalSchools : SchoolType → Nat
| SchoolType.Primary => 21
| SchoolType.Middle => 14
| SchoolType.University => 7

/-- Represents the number of schools selected in stratified sampling -/
def selectedSchools : SchoolType → Nat
| SchoolType.Primary => 3
| SchoolType.Middle => 2
| SchoolType.University => 1

/-- The total number of schools selected -/
def totalSelected : Nat := 6

/-- The number of ways to choose 2 schools from the selected primary schools -/
def waysToChoosePrimary : Nat := 3

/-- The total number of ways to choose 2 schools from all selected schools -/
def totalWaysToChoose : Nat := 15

theorem probability_both_primary :
  (waysToChoosePrimary : Rat) / totalWaysToChoose = 1 / 5 := by
  sorry


end probability_both_primary_l1440_144056


namespace last_date_2011_divisible_by_101_l1440_144003

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2011 ∧ 
  1 ≤ month ∧ month ≤ 12 ∧
  1 ≤ day ∧ day ≤ 31 ∧
  (month ∈ [4, 6, 9, 11] → day ≤ 30) ∧
  (month = 2 → day ≤ 28)

def date_to_number (year month day : ℕ) : ℕ :=
  year * 10000 + month * 100 + day

theorem last_date_2011_divisible_by_101 :
  ∀ year month day : ℕ,
    is_valid_date year month day →
    date_to_number year month day ≤ 20111221 →
    date_to_number year month day % 101 = 0 →
    date_to_number year month day = 20111221 :=
sorry

end last_date_2011_divisible_by_101_l1440_144003


namespace line_passes_through_fixed_point_l1440_144045

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + n - 1 = 0) :
  ∃ (x y : ℝ), x = 1 ∧ y = -1 ∧ m * x + y + n = 0 :=
by
  sorry

end line_passes_through_fixed_point_l1440_144045


namespace point_on_line_l1440_144023

/-- A point represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line : 
  let p1 : Point := ⟨0, 4⟩
  let p2 : Point := ⟨-6, 1⟩
  let p3 : Point := ⟨6, 7⟩
  collinear p1 p2 p3 := by
  sorry

end point_on_line_l1440_144023


namespace simple_interest_rate_calculation_l1440_144061

/-- Simple interest rate calculation -/
theorem simple_interest_rate_calculation (P : ℝ) (P_pos : P > 0) : ∃ R : ℝ,
  R > 0 ∧ R < 100 ∧ (P * R * 10) / 100 = P / 5 ∧ R = 2 := by
  sorry

end simple_interest_rate_calculation_l1440_144061
