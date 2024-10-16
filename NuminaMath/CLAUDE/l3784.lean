import Mathlib

namespace NUMINAMATH_CALUDE_positive_integer_solutions_l3784_378416

theorem positive_integer_solutions : 
  ∀ x y z : ℕ+, 
    (x + y = z ∧ x^2 * y = z^2 + 1) ↔ 
    ((x = 5 ∧ y = 2 ∧ z = 7) ∨ (x = 5 ∧ y = 3 ∧ z = 8)) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_solutions_l3784_378416


namespace NUMINAMATH_CALUDE_exists_monochromatic_congruent_triangle_l3784_378477

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Define congruence for triangles
def Congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of having all vertices of the same color
def SameColor (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- The main theorem
theorem exists_monochromatic_congruent_triangle :
  ∃ (T : Triangle), ∀ (coloring : Coloring),
    ∃ (T' : Triangle), Congruent T T' ∧ SameColor T' coloring := by sorry

end NUMINAMATH_CALUDE_exists_monochromatic_congruent_triangle_l3784_378477


namespace NUMINAMATH_CALUDE_log_equation_solution_l3784_378483

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 2 = 9 → x = 2^(27/10) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3784_378483


namespace NUMINAMATH_CALUDE_lowest_unique_score_above_100_unique_solution_for_105_l3784_378448

/-- Represents the score calculation function for the math examination. -/
def score (c w : ℕ) : ℕ := 50 + 5 * c - 2 * w

/-- Theorem stating that 105 is the lowest score above 100 with a unique solution. -/
theorem lowest_unique_score_above_100 : 
  ∀ s : ℕ, s > 100 → s < 105 → 
  (∃ c w : ℕ, c + w ≤ 50 ∧ score c w = s) → 
  (∃ c₁ w₁ c₂ w₂ : ℕ, 
    c₁ + w₁ ≤ 50 ∧ c₂ + w₂ ≤ 50 ∧ 
    score c₁ w₁ = s ∧ score c₂ w₂ = s ∧ 
    (c₁ ≠ c₂ ∨ w₁ ≠ w₂)) :=
by sorry

/-- Theorem stating that 105 has a unique solution for c and w. -/
theorem unique_solution_for_105 : 
  ∃! c w : ℕ, c + w ≤ 50 ∧ score c w = 105 :=
by sorry

end NUMINAMATH_CALUDE_lowest_unique_score_above_100_unique_solution_for_105_l3784_378448


namespace NUMINAMATH_CALUDE_weak_to_strong_ratio_l3784_378454

/-- Represents the amount of coffee used for different strengths --/
structure CoffeeUsage where
  weak_per_cup : ℕ
  strong_per_cup : ℕ
  cups_each : ℕ
  total_tablespoons : ℕ

/-- Theorem stating the ratio of weak to strong coffee usage --/
theorem weak_to_strong_ratio (c : CoffeeUsage) 
  (h1 : c.weak_per_cup = 1)
  (h2 : c.strong_per_cup = 2)
  (h3 : c.cups_each = 12)
  (h4 : c.total_tablespoons = 36) :
  (c.weak_per_cup * c.cups_each) / (c.strong_per_cup * c.cups_each) = 1 / 2 := by
  sorry

#check weak_to_strong_ratio

end NUMINAMATH_CALUDE_weak_to_strong_ratio_l3784_378454


namespace NUMINAMATH_CALUDE_grocery_cost_l3784_378497

/-- The cost of groceries problem -/
theorem grocery_cost (mango_price rice_price flour_price : ℝ) 
  (h1 : 10 * mango_price = 24 * rice_price)
  (h2 : flour_price = 2 * rice_price)
  (h3 : flour_price = 20.50) : 
  4 * mango_price + 3 * rice_price + 5 * flour_price = 231.65 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_l3784_378497


namespace NUMINAMATH_CALUDE_percentage_problem_l3784_378471

theorem percentage_problem (p : ℝ) : p * 50 = 0.15 → p = 0.003 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3784_378471


namespace NUMINAMATH_CALUDE_charging_pile_equation_l3784_378467

/-- Represents the growth of smart charging piles over two months -/
def charging_pile_growth (initial : ℕ) (growth_rate : ℝ) : ℝ :=
  initial * (1 + growth_rate)^2

/-- Theorem stating the relationship between the number of charging piles
    in the first and third months, given the monthly average growth rate -/
theorem charging_pile_equation (x : ℝ) : charging_pile_growth 301 x = 500 := by
  sorry

end NUMINAMATH_CALUDE_charging_pile_equation_l3784_378467


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_x_coord_l3784_378413

/-- Given vectors a and b in R², if a is perpendicular to (a - b), then the x-coordinate of b is 9. -/
theorem perpendicular_vectors_imply_x_coord (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.2 = -2 →
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) →
  b.1 = 9 := by
  sorry

#check perpendicular_vectors_imply_x_coord

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_x_coord_l3784_378413


namespace NUMINAMATH_CALUDE_impossible_coloring_l3784_378499

theorem impossible_coloring (R G B : Set ℤ) : 
  (∀ (x y : ℤ), (x ∈ G ∧ y ∈ B) ∨ (x ∈ R ∧ y ∈ B) ∨ (x ∈ R ∧ y ∈ G) → x + y ∈ R) →
  (R ∪ G ∪ B = Set.univ) →
  (R ∩ G = ∅ ∧ R ∩ B = ∅ ∧ G ∩ B = ∅) →
  (R ≠ ∅ ∧ G ≠ ∅ ∧ B ≠ ∅) →
  False :=
by sorry

end NUMINAMATH_CALUDE_impossible_coloring_l3784_378499


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l3784_378480

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l3784_378480


namespace NUMINAMATH_CALUDE_triangle_must_be_obtuse_l3784_378464

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  (t.a = 2 * t.b ∨ t.b = 2 * t.c ∨ t.c = 2 * t.a) ∧
  (t.A = Real.pi / 6 ∨ t.B = Real.pi / 6 ∨ t.C = Real.pi / 6)

-- Define an obtuse triangle
def IsObtuseTriangle (t : Triangle) : Prop :=
  t.A > Real.pi / 2 ∨ t.B > Real.pi / 2 ∨ t.C > Real.pi / 2

-- Theorem statement
theorem triangle_must_be_obtuse (t : Triangle) (h : TriangleProperties t) : IsObtuseTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_must_be_obtuse_l3784_378464


namespace NUMINAMATH_CALUDE_min_value_expression_l3784_378465

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (6 * c) / (3 * a + b) + (6 * a) / (b + 3 * c) + (2 * b) / (a + c) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3784_378465


namespace NUMINAMATH_CALUDE_circle_equation_l3784_378452

-- Define the circle ⊙C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

def center_on_line (c : Circle) : Prop :=
  3 * c.center.1 = c.center.2

def intersects_line (c : Circle) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    A.1 - A.2 = 0 ∧ B.1 - B.2 = 0 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2

def triangle_area (c : Circle) : Prop :=
  ∃ (A B : ℝ × ℝ), A ≠ B ∧
    A.1 - A.2 = 0 ∧ B.1 - B.2 = 0 ∧
    (A.1 - c.center.1)^2 + (A.2 - c.center.2)^2 = c.radius^2 ∧
    (B.1 - c.center.1)^2 + (B.2 - c.center.2)^2 = c.radius^2 ∧
    1/2 * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * 
    (|c.center.1 - c.center.2| / Real.sqrt 2) = Real.sqrt 14

-- Theorem statement
theorem circle_equation (c : Circle) :
  tangent_to_x_axis c →
  center_on_line c →
  intersects_line c →
  triangle_area c →
  ((c.center.1 = 1 ∧ c.center.2 = 3 ∧ c.radius = 3) ∨
   (c.center.1 = -1 ∧ c.center.2 = -3 ∧ c.radius = 3)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3784_378452


namespace NUMINAMATH_CALUDE_sister_watermelons_count_l3784_378440

/-- The number of watermelons Danny brings -/
def danny_watermelons : ℕ := 3

/-- The number of slices Danny cuts each watermelon into -/
def danny_slices_per_watermelon : ℕ := 10

/-- The number of slices Danny's sister cuts each watermelon into -/
def sister_slices_per_watermelon : ℕ := 15

/-- The total number of watermelon slices at the picnic -/
def total_slices : ℕ := 45

/-- The number of watermelons Danny's sister brings -/
def sister_watermelons : ℕ := 1

theorem sister_watermelons_count : sister_watermelons = 
  (total_slices - danny_watermelons * danny_slices_per_watermelon) / sister_slices_per_watermelon := by
  sorry

end NUMINAMATH_CALUDE_sister_watermelons_count_l3784_378440


namespace NUMINAMATH_CALUDE_zachary_pushups_l3784_378498

theorem zachary_pushups (zachary : ℕ) (david : ℕ) : 
  david = zachary + 58 → 
  zachary + david = 146 → 
  zachary = 44 := by
sorry

end NUMINAMATH_CALUDE_zachary_pushups_l3784_378498


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l3784_378488

theorem complex_absolute_value_product : 
  Complex.abs ((7 - 4 * Complex.I) * (5 + 12 * Complex.I)) = 13 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l3784_378488


namespace NUMINAMATH_CALUDE_athena_snack_spending_l3784_378450

/-- Calculates the total amount spent by Athena on snacks -/
def total_spent (sandwich_price : ℚ) (sandwich_qty : ℕ)
                (drink_price : ℚ) (drink_qty : ℕ)
                (cookie_price : ℚ) (cookie_qty : ℕ)
                (chips_price : ℚ) (chips_qty : ℕ) : ℚ :=
  sandwich_price * sandwich_qty +
  drink_price * drink_qty +
  cookie_price * cookie_qty +
  chips_price * chips_qty

/-- Proves that Athena spent $33.95 on snacks -/
theorem athena_snack_spending :
  total_spent (325/100) 4 (275/100) 3 (150/100) 6 (185/100) 2 = 3395/100 := by
  sorry

end NUMINAMATH_CALUDE_athena_snack_spending_l3784_378450


namespace NUMINAMATH_CALUDE_distinct_numbers_ratio_l3784_378414

theorem distinct_numbers_ratio (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : (b - a)^2 - 4*(b - c)*(c - a) = 0) : 
  (b - c) / (c - a) = -1 := by
  sorry

end NUMINAMATH_CALUDE_distinct_numbers_ratio_l3784_378414


namespace NUMINAMATH_CALUDE_equilateral_triangle_probability_l3784_378437

/-- Given a circle divided into 30 equal parts, the probability of randomly selecting
    3 different points that form an equilateral triangle is 1/406. -/
theorem equilateral_triangle_probability (n : ℕ) (h : n = 30) :
  let total_combinations := n.choose 3
  let equilateral_triangles := n / 3
  (equilateral_triangles : ℚ) / total_combinations = 1 / 406 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_probability_l3784_378437


namespace NUMINAMATH_CALUDE_ellipse_chord_through_focus_l3784_378491

/-- The x-coordinate of point A on the ellipse satisfies a specific quadratic equation --/
theorem ellipse_chord_through_focus (x y : ℝ) : 
  (x^2 / 36 + y^2 / 16 = 1) →  -- ellipse equation
  ((x - 2 * Real.sqrt 5)^2 + y^2 = 9) →  -- AF = 3
  (84 * x^2 - 400 * x + 552 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_through_focus_l3784_378491


namespace NUMINAMATH_CALUDE_interview_room_occupancy_l3784_378409

/-- The number of people initially in the waiting room -/
def initial_waiting : ℕ := 22

/-- The number of additional people arriving -/
def additional_arrivals : ℕ := 3

/-- The ratio of people in the waiting room to the interview room after arrivals -/
def room_ratio : ℕ := 5

theorem interview_room_occupancy :
  ∃ (interview : ℕ), 
    (initial_waiting + additional_arrivals) = room_ratio * interview ∧
    interview = 5 := by
  sorry

end NUMINAMATH_CALUDE_interview_room_occupancy_l3784_378409


namespace NUMINAMATH_CALUDE_cube_root_of_product_l3784_378466

theorem cube_root_of_product (x y z : ℕ) : 
  (5^9 * 7^6 * 13^3 : ℝ)^(1/3) = 79625 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l3784_378466


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3784_378430

theorem triangle_perimeter : ∀ x : ℝ,
  x^2 - 8*x + 12 = 0 →
  x > 0 →
  4 + x > 7 ∧ 7 + x > 4 ∧ 4 + 7 > x →
  4 + 7 + x = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3784_378430


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l3784_378420

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a / b = 5) :
  (6 * a^2) / (6 * b^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l3784_378420


namespace NUMINAMATH_CALUDE_group_b_sample_size_l3784_378424

/-- Represents the number of cities in each group and the total sample size -/
structure CityGroups where
  total : Nat
  groupA : Nat
  groupB : Nat
  groupC : Nat
  sampleSize : Nat

/-- Calculates the number of cities to be selected from a specific group in stratified sampling -/
def stratifiedSampleSize (cg : CityGroups) (groupSize : Nat) : Nat :=
  (groupSize * cg.sampleSize) / cg.total

/-- Theorem stating that for the given city groups, the stratified sample size for Group B is 3 -/
theorem group_b_sample_size (cg : CityGroups) 
  (h1 : cg.total = 24)
  (h2 : cg.groupA = 4)
  (h3 : cg.groupB = 12)
  (h4 : cg.groupC = 8)
  (h5 : cg.sampleSize = 6)
  : stratifiedSampleSize cg cg.groupB = 3 := by
  sorry

end NUMINAMATH_CALUDE_group_b_sample_size_l3784_378424


namespace NUMINAMATH_CALUDE_robins_hair_length_l3784_378489

/-- Given Robin's initial hair length and the length he cut off, calculate his final hair length -/
theorem robins_hair_length (initial_length cut_length : ℕ) 
  (h1 : initial_length = 14)
  (h2 : cut_length = 13) :
  initial_length - cut_length = 1 := by
  sorry

end NUMINAMATH_CALUDE_robins_hair_length_l3784_378489


namespace NUMINAMATH_CALUDE_apple_eating_contest_l3784_378456

def classroom (n : ℕ) (total_apples : ℕ) (aaron_apples : ℕ) (zeb_apples : ℕ) : Prop :=
  n = 8 ∧
  total_apples > 20 ∧
  ∀ student, student ≠ aaron_apples → aaron_apples ≥ student ∧
  ∀ student, student ≠ zeb_apples → student ≥ zeb_apples

theorem apple_eating_contest (n : ℕ) (total_apples : ℕ) (aaron_apples : ℕ) (zeb_apples : ℕ) 
  (h : classroom n total_apples aaron_apples zeb_apples) : 
  aaron_apples - zeb_apples = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_eating_contest_l3784_378456


namespace NUMINAMATH_CALUDE_force_system_ratio_l3784_378490

/-- Two forces acting on a material point at a right angle -/
structure ForceSystem where
  f1 : ℝ
  f2 : ℝ
  resultant : ℝ

/-- The magnitudes form an arithmetic progression -/
def is_arithmetic_progression (fs : ForceSystem) : Prop :=
  ∃ (d : ℝ), fs.f2 = fs.f1 + d ∧ fs.resultant = fs.f1 + 2*d

/-- The forces act at a right angle -/
def forces_at_right_angle (fs : ForceSystem) : Prop :=
  fs.resultant^2 = fs.f1^2 + fs.f2^2

/-- The ratio of the magnitudes of the forces is 3:4 -/
def force_ratio_is_3_to_4 (fs : ForceSystem) : Prop :=
  3 * fs.f2 = 4 * fs.f1

theorem force_system_ratio (fs : ForceSystem) 
  (h1 : is_arithmetic_progression fs) 
  (h2 : forces_at_right_angle fs) : 
  force_ratio_is_3_to_4 fs :=
sorry

end NUMINAMATH_CALUDE_force_system_ratio_l3784_378490


namespace NUMINAMATH_CALUDE_cookie_sharing_proof_l3784_378415

/-- The number of people sharing cookies baked by Beth --/
def number_of_people : ℕ :=
  let batches : ℕ := 4
  let dozens_per_batch : ℕ := 2
  let cookies_per_dozen : ℕ := 12
  let cookies_per_person : ℕ := 6
  let total_cookies : ℕ := batches * dozens_per_batch * cookies_per_dozen
  total_cookies / cookies_per_person

/-- Proof that the number of people sharing the cookies is 16 --/
theorem cookie_sharing_proof : number_of_people = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sharing_proof_l3784_378415


namespace NUMINAMATH_CALUDE_tv_purchase_months_l3784_378432

/-- Calculates the number of months required to purchase a TV given income and expenses -/
def monthsToTV (monthlyIncome : ℕ) (foodExpense : ℕ) (utilitiesExpense : ℕ) (otherExpenses : ℕ)
                (currentSavings : ℕ) (tvCost : ℕ) : ℕ :=
  let totalExpenses := foodExpense + utilitiesExpense + otherExpenses
  let disposableIncome := monthlyIncome - totalExpenses
  let amountNeeded := tvCost - currentSavings
  (amountNeeded + disposableIncome - 1) / disposableIncome

theorem tv_purchase_months :
  monthsToTV 30000 15000 5000 2500 10000 25000 = 2 :=
sorry

end NUMINAMATH_CALUDE_tv_purchase_months_l3784_378432


namespace NUMINAMATH_CALUDE_rectangle_area_l3784_378482

/-- The area of a rectangle with the given conditions -/
theorem rectangle_area (l w : ℝ) (h1 : (l + 2) * w - l * w = 10) (h2 : l * w - l * (w - 3) = 18) : l * w = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3784_378482


namespace NUMINAMATH_CALUDE_theater_revenue_l3784_378439

theorem theater_revenue (n : ℕ) (cost total_revenue actual_revenue : ℝ) :
  (total_revenue = cost * 1.2) →
  (actual_revenue = total_revenue * 0.95) →
  (actual_revenue = cost * 1.14) :=
by sorry

end NUMINAMATH_CALUDE_theater_revenue_l3784_378439


namespace NUMINAMATH_CALUDE_sequence_sum_l3784_378476

theorem sequence_sum (a b c d : ℕ) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧  -- increasing positive integers
  b - a = c - b ∧                  -- arithmetic progression
  c * c = b * d ∧                  -- geometric progression
  d - a = 42                       -- difference between first and fourth terms
  → a + b + c + d = 123 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l3784_378476


namespace NUMINAMATH_CALUDE_root_magnitude_l3784_378473

theorem root_magnitude (a b : ℝ) (z : ℂ) (h : z = 1 + b * Complex.I) 
  (h_root : z ^ 2 + a * z + 3 = 0) : Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_root_magnitude_l3784_378473


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l3784_378433

/-- Given four square regions with specified perimeters and a relation between sides,
    prove that the ratio of areas of region III to region IV is 9/4. -/
theorem area_ratio_of_squares (perimeter_I perimeter_II perimeter_IV : ℝ) 
    (h1 : perimeter_I = 16)
    (h2 : perimeter_II = 20)
    (h3 : perimeter_IV = 32)
    (h4 : ∀ s : ℝ, s > 0 → perimeter_I = 4 * s → 3 * s = side_length_III) :
    (side_length_III ^ 2) / ((perimeter_IV / 4) ^ 2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l3784_378433


namespace NUMINAMATH_CALUDE_least_jumps_to_19999_l3784_378444

/-- Represents the total distance jumped after n jumps -/
def totalDistance (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the distance of the nth jump -/
def nthJump (n : ℕ) : ℕ := n

theorem least_jumps_to_19999 :
  ∀ k : ℕ, (totalDistance k ≥ 19999 → k ≥ 201) ∧
  (∃ (adjustedJump : ℤ), 
    totalDistance 201 + nthJump 201 + adjustedJump = 19999 ∧ 
    adjustedJump.natAbs < nthJump 201) := by
  sorry

end NUMINAMATH_CALUDE_least_jumps_to_19999_l3784_378444


namespace NUMINAMATH_CALUDE_complex_problem_l3784_378431

-- Define complex numbers z₁ and z₂
def z₁ : ℂ := Complex.mk 1 2
def z₂ : ℂ := Complex.mk (-1) 1

-- Define the equation for z
def equation (a : ℝ) (z : ℂ) : Prop :=
  2 * z^2 + a * z + 10 = 0

-- Define the relationship between z and z₁z₂
def z_condition (z : ℂ) : Prop :=
  z.re = (z₁ * z₂).im

-- Main theorem
theorem complex_problem :
  ∃ (a : ℝ) (z : ℂ),
    Complex.abs (z₁ - z₂) = Real.sqrt 5 ∧
    a = 4 ∧
    (z = Complex.mk (-1) 2 ∨ z = Complex.mk (-1) (-2)) ∧
    equation a z ∧
    z_condition z := by sorry

end NUMINAMATH_CALUDE_complex_problem_l3784_378431


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3784_378435

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 23^7 → 
  m % 10 = 9 → 
  n % 10 = 3 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3784_378435


namespace NUMINAMATH_CALUDE_conditional_inequality_l3784_378400

theorem conditional_inequality (a b c : ℝ) (h1 : c > 0) (h2 : a * c^2 > b * c^2) : a > b := by
  sorry

end NUMINAMATH_CALUDE_conditional_inequality_l3784_378400


namespace NUMINAMATH_CALUDE_unique_lock_code_satisfies_conditions_unique_lock_code_is_unique_l3784_378410

/-- Represents a seven-digit lock code -/
structure LockCode where
  digits : Fin 7 → Nat
  first_three_same : ∀ i j, i < 3 → j < 3 → digits i = digits j
  last_four_same : ∀ i j, 3 ≤ i → i < 7 → 3 ≤ j → j < 7 → digits i = digits j
  all_digits : ∀ i, digits i < 10

/-- The sum of digits in a lock code -/
def digit_sum (code : LockCode) : Nat :=
  (Finset.range 7).sum (λ i => code.digits i)

/-- The unique lock code satisfying all conditions -/
def unique_lock_code : LockCode where
  digits := λ i => if i < 3 then 3 else 7
  first_three_same := by sorry
  last_four_same := by sorry
  all_digits := by sorry

theorem unique_lock_code_satisfies_conditions :
  let s := digit_sum unique_lock_code
  (10 ≤ s ∧ s < 100) ∧
  (s / 10 = unique_lock_code.digits 0) ∧
  (s % 10 = unique_lock_code.digits 6) :=
by sorry

theorem unique_lock_code_is_unique (code : LockCode) :
  let s := digit_sum code
  (10 ≤ s ∧ s < 100) →
  (s / 10 = code.digits 0) →
  (s % 10 = code.digits 6) →
  code = unique_lock_code :=
by sorry

end NUMINAMATH_CALUDE_unique_lock_code_satisfies_conditions_unique_lock_code_is_unique_l3784_378410


namespace NUMINAMATH_CALUDE_rectangle_area_l3784_378478

theorem rectangle_area (p : ℝ) (p_small : ℝ) (h1 : p = 30) (h2 : p_small = 16) :
  let w := (p - p_small) / 2
  let l := p_small / 2 - w + w
  w * l = 56 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3784_378478


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3784_378457

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 5}
def B : Set Nat := {1, 3, 4}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {3, 4} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3784_378457


namespace NUMINAMATH_CALUDE_total_stamps_l3784_378405

theorem total_stamps (a b : ℕ) (h1 : a * 4 = b * 5) (h2 : (a - 5) * 5 = (b + 5) * 4) : a + b = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l3784_378405


namespace NUMINAMATH_CALUDE_chloe_carrots_theorem_l3784_378421

/-- Calculates the total number of carrots Chloe has after throwing some out and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Chloe's total carrots is correct given the initial conditions. -/
theorem chloe_carrots_theorem (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) 
  (h1 : initial ≥ thrown_out) : 
  total_carrots initial thrown_out new_picked = initial - thrown_out + new_picked :=
by sorry

end NUMINAMATH_CALUDE_chloe_carrots_theorem_l3784_378421


namespace NUMINAMATH_CALUDE_olivia_payment_l3784_378438

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := 4

/-- Represents the number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- Calculates the total amount Olivia pays in dollars -/
def total_payment : ℚ :=
  (quarters_for_chips + quarters_for_soda) / quarters_per_dollar

theorem olivia_payment :
  total_payment = 4 := by sorry

end NUMINAMATH_CALUDE_olivia_payment_l3784_378438


namespace NUMINAMATH_CALUDE_shooter_scores_equal_l3784_378459

/-- The expected value of a binomial distribution -/
def binomialExpectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The score of shooter A -/
def X₁ : ℝ := binomialExpectation 10 0.9

/-- The score of shooter Y (intermediate for shooter B) -/
def Y : ℝ := binomialExpectation 5 0.8

/-- The score of shooter B -/
def X₂ : ℝ := 2 * Y + 1

theorem shooter_scores_equal : X₁ = X₂ := by sorry

end NUMINAMATH_CALUDE_shooter_scores_equal_l3784_378459


namespace NUMINAMATH_CALUDE_watertown_marching_band_max_size_l3784_378472

theorem watertown_marching_band_max_size :
  ∀ n : ℕ,
  (25 * n < 1200) →
  (25 * n % 29 = 6) →
  (∀ m : ℕ, (25 * m < 1200) → (25 * m % 29 = 6) → m ≤ n) →
  25 * n = 1050 :=
by sorry

end NUMINAMATH_CALUDE_watertown_marching_band_max_size_l3784_378472


namespace NUMINAMATH_CALUDE_divide_by_three_l3784_378423

theorem divide_by_three (n : ℚ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_three_l3784_378423


namespace NUMINAMATH_CALUDE_find_T_l3784_378443

theorem find_T : ∃ T : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * T = (1/2 : ℚ) * (1/6 : ℚ) * 72 ∧ T = 64 := by
  sorry

end NUMINAMATH_CALUDE_find_T_l3784_378443


namespace NUMINAMATH_CALUDE_line_equation_l3784_378406

-- Define the point M
def M : ℝ × ℝ := (1, -2)

-- Define the line l
def l : Set (ℝ × ℝ) := sorry

-- Define points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- State that P is on the x-axis
axiom P_on_x_axis : P.2 = 0

-- State that Q is on the y-axis
axiom Q_on_y_axis : Q.1 = 0

-- State that M is the midpoint of PQ
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- State that P, Q, and M are on the line l
axiom P_on_l : P ∈ l
axiom Q_on_l : Q ∈ l
axiom M_on_l : M ∈ l

-- Theorem: The equation of line PQ is 2x - y - 4 = 0
theorem line_equation : ∀ (x y : ℝ), (x, y) ∈ l ↔ 2 * x - y - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l3784_378406


namespace NUMINAMATH_CALUDE_joanna_reading_speed_l3784_378442

theorem joanna_reading_speed :
  ∀ (total_pages : ℕ) (monday_hours tuesday_hours remaining_hours : ℝ) (pages_per_hour : ℝ),
    total_pages = 248 →
    monday_hours = 3 →
    tuesday_hours = 6.5 →
    remaining_hours = 6 →
    (monday_hours + tuesday_hours + remaining_hours) * pages_per_hour = total_pages →
    pages_per_hour = 16 := by
  sorry

end NUMINAMATH_CALUDE_joanna_reading_speed_l3784_378442


namespace NUMINAMATH_CALUDE_cubic_polynomial_factor_l3784_378417

/-- Given a cubic polynomial of the form 3x^3 - dx + 18 with a quadratic factor x^2 + qx + 2,
    prove that d = -6 -/
theorem cubic_polynomial_factor (d : ℝ) : 
  (∃ q : ℝ, ∃ m : ℝ, ∀ x : ℝ, 
    3 * x^3 - d * x + 18 = (x^2 + q * x + 2) * (m * x)) → 
  d = -6 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_factor_l3784_378417


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l3784_378404

theorem shirt_price_calculation (num_shirts : ℕ) (discount_rate : ℚ) (total_paid : ℚ) :
  num_shirts = 6 →
  discount_rate = 1/5 →
  total_paid = 240 →
  ∃ (regular_price : ℚ), regular_price = 50 ∧ 
    num_shirts * (regular_price * (1 - discount_rate)) = total_paid := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l3784_378404


namespace NUMINAMATH_CALUDE_staircase_covering_l3784_378487

/-- A staircase tile with dimensions 6 × 1 -/
structure StaircaseTile where
  length : Nat := 6
  width : Nat := 1

/-- Predicate to check if a field can be covered with staircase tiles -/
def canCoverField (m n : Nat) : Prop :=
  ∃ (a b c d : Nat), 
    ((m = 12 * a ∧ n ≥ b ∧ b ≥ 6) ∨ 
     (n = 12 * a ∧ m ≥ b ∧ b ≥ 6) ∨
     (m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) ∨
     (n = 3 * c ∧ m = 4 * d ∧ c ≥ 2 ∧ d ≥ 3))

theorem staircase_covering (m n : Nat) (hm : m ≥ 6) (hn : n ≥ 6) :
  canCoverField m n ↔ 
    ∃ (tiles : List StaircaseTile), 
      (tiles.length * 6 = m * n) ∧ 
      (∀ t ∈ tiles, t.length = 6 ∧ t.width = 1) :=
by sorry

end NUMINAMATH_CALUDE_staircase_covering_l3784_378487


namespace NUMINAMATH_CALUDE_factor_expression_l3784_378446

theorem factor_expression (x : ℝ) : 4 * x * (x + 1) + 9 * (x + 1) = (x + 1) * (4 * x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3784_378446


namespace NUMINAMATH_CALUDE_exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals_l3784_378425

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle. -/
structure CyclicQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_cyclic : ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i, dist center (vertices i) = radius

/-- The perimeter of a quadrilateral. -/
def perimeter (q : CyclicQuadrilateral) : ℝ :=
  (dist (q.vertices 0) (q.vertices 1)) +
  (dist (q.vertices 1) (q.vertices 2)) +
  (dist (q.vertices 2) (q.vertices 3)) +
  (dist (q.vertices 3) (q.vertices 0))

/-- The area of a quadrilateral. -/
def area (q : CyclicQuadrilateral) : ℝ := sorry

/-- Two quadrilaterals are congruent if there exists a rigid transformation that maps one to the other. -/
def congruent (q1 q2 : CyclicQuadrilateral) : Prop := sorry

theorem exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals :
  ∃ (q1 q2 : CyclicQuadrilateral),
    perimeter q1 = perimeter q2 ∧
    area q1 = area q2 ∧
    ¬congruent q1 q2 := by
  sorry

end NUMINAMATH_CALUDE_exist_non_congruent_equal_perimeter_area_cyclic_quadrilaterals_l3784_378425


namespace NUMINAMATH_CALUDE_xiao_ming_foot_length_l3784_378419

/-- The relationship between a person's height and foot length -/
def height_foot_relation (h d : ℝ) : Prop := h = 7 * d

/-- Xiao Ming's height in cm -/
def xiao_ming_height : ℝ := 171.5

theorem xiao_ming_foot_length :
  ∃ d : ℝ, height_foot_relation xiao_ming_height d ∧ d = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_foot_length_l3784_378419


namespace NUMINAMATH_CALUDE_inscribed_box_dimension_l3784_378486

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  sphere_radius : ℝ
  surface_area : ℝ
  edge_sum : ℝ
  sphere_constraint : x^2 + y^2 + z^2 = 4 * sphere_radius^2
  surface_area_constraint : 2*x*y + 2*y*z + 2*x*z = surface_area
  edge_sum_constraint : 4*(x + y + z) = edge_sum

/-- Theorem: For a rectangular box inscribed in a sphere of radius 10,
    with surface area 416 and sum of edge lengths 120,
    one of its dimensions is 10 -/
theorem inscribed_box_dimension (Q : InscribedBox)
    (h_radius : Q.sphere_radius = 10)
    (h_surface : Q.surface_area = 416)
    (h_edges : Q.edge_sum = 120) :
    Q.x = 10 ∨ Q.y = 10 ∨ Q.z = 10 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_dimension_l3784_378486


namespace NUMINAMATH_CALUDE_graduates_parents_l3784_378474

theorem graduates_parents (graduates : ℕ) (teachers : ℕ) (total_chairs : ℕ)
  (h_graduates : graduates = 50)
  (h_teachers : teachers = 20)
  (h_total_chairs : total_chairs = 180) :
  (total_chairs - (graduates + teachers + teachers / 2)) / graduates = 2 := by
  sorry

end NUMINAMATH_CALUDE_graduates_parents_l3784_378474


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3784_378412

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  A = Real.pi / 3 →
  0 < B →
  B < 2 * Real.pi / 3 →
  0 < C →
  C < 2 * Real.pi / 3 →
  A + B + C = Real.pi →
  b = 2 * Real.sin B →
  c = 2 * Real.sin C →
  a + b + c ≤ 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3784_378412


namespace NUMINAMATH_CALUDE_probability_diamond_spade_heart_value_l3784_378447

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- Probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def probability_diamond_spade_heart : ℚ :=
  (CardsPerSuit / StandardDeck) *
  (CardsPerSuit / (StandardDeck - 1)) *
  (CardsPerSuit / (StandardDeck - 2))

/-- Theorem stating the probability of drawing a diamond, then a spade, then a heart -/
theorem probability_diamond_spade_heart_value :
  probability_diamond_spade_heart = 169 / 10200 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_spade_heart_value_l3784_378447


namespace NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l3784_378495

theorem percentage_of_women_in_non_union (total : ℝ) (h1 : total > 0) : 
  let men := 0.48 * total
  let unionized := 0.60 * total
  let non_unionized := total - unionized
  let women_non_union := 0.85 * non_unionized
  women_non_union / non_unionized = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l3784_378495


namespace NUMINAMATH_CALUDE_num_regions_correct_l3784_378470

/-- A structure representing a collection of planes in 3D space -/
structure PlaneCollection where
  n : ℕ
  intersection_of_two : ∀ p q : Fin n, p ≠ q → Line
  intersection_of_three : ∀ p q r : Fin n, p ≠ q ∧ q ≠ r ∧ p ≠ r → Point
  no_four_intersect : ∀ p q r s : Fin n, p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ p ≠ r ∧ p ≠ s ∧ q ≠ s → ¬ Point

/-- The number of non-overlapping regions created by n planes -/
def num_regions (pc : PlaneCollection) : ℕ :=
  (pc.n^3 + 5*pc.n + 6) / 6

/-- Theorem stating that the number of regions is correct -/
theorem num_regions_correct (pc : PlaneCollection) :
  num_regions pc = (pc.n^3 + 5*pc.n + 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_num_regions_correct_l3784_378470


namespace NUMINAMATH_CALUDE_largest_base_8_three_digit_in_base_10_l3784_378461

/-- The largest three-digit number in a given base -/
def largest_three_digit (base : ℕ) : ℕ :=
  (base - 1) * base^2 + (base - 1) * base^1 + (base - 1) * base^0

/-- Theorem: The largest three-digit base-8 number in base-10 is 511 -/
theorem largest_base_8_three_digit_in_base_10 :
  largest_three_digit 8 = 511 := by sorry

end NUMINAMATH_CALUDE_largest_base_8_three_digit_in_base_10_l3784_378461


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3784_378403

theorem fraction_decomposition (x A B : ℚ) : 
  (7 * x - 15) / (3 * x^2 + 2 * x - 8) = A / (x + 2) + B / (3 * x - 4) → 
  A = 29 / 10 ∧ B = -17 / 10 := by
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3784_378403


namespace NUMINAMATH_CALUDE_pushup_problem_l3784_378462

/-- Given that David did 30 more push-ups than Zachary, Sarah completed twice as many push-ups as Zachary,
    and David did 37 push-ups, prove that Zachary and Sarah did 21 push-ups combined. -/
theorem pushup_problem (david zachary sarah : ℕ) : 
  david = zachary + 30 →
  sarah = 2 * zachary →
  david = 37 →
  zachary + sarah = 21 := by
  sorry

end NUMINAMATH_CALUDE_pushup_problem_l3784_378462


namespace NUMINAMATH_CALUDE_johns_money_ratio_l3784_378479

/-- The ratio of money John got from his grandma to his grandpa -/
theorem johns_money_ratio :
  ∀ (x : ℚ), 
  (30 : ℚ) + 30 * x = 120 →
  (30 * x) / 30 = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_johns_money_ratio_l3784_378479


namespace NUMINAMATH_CALUDE_min_value_and_curve_intersection_l3784_378408

theorem min_value_and_curve_intersection (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/m + 16/n ≤ 1/x + 16/y) →
  m = 1/5 →
  n = 4/5 →
  ∃! α : ℝ, (1/5 : ℝ) = (1/25 : ℝ)^α :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_curve_intersection_l3784_378408


namespace NUMINAMATH_CALUDE_amoeba_growth_after_five_days_l3784_378481

def amoeba_population (initial_count : ℕ) (growth_factor : ℕ) (days : ℕ) : ℕ :=
  initial_count * growth_factor ^ days

theorem amoeba_growth_after_five_days :
  amoeba_population 1 3 5 = 243 :=
by sorry

end NUMINAMATH_CALUDE_amoeba_growth_after_five_days_l3784_378481


namespace NUMINAMATH_CALUDE_f_property_l3784_378428

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- This case should never occur in our problem

-- State the theorem
theorem f_property (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : f a = f (a + 1)) :
  f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l3784_378428


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_point_l3784_378427

theorem exponential_function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_point_l3784_378427


namespace NUMINAMATH_CALUDE_medium_sized_fir_trees_l3784_378460

theorem medium_sized_fir_trees (total : ℕ) (oaks : ℕ) (saplings : ℕ) 
  (h1 : total = 96) 
  (h2 : oaks = 15) 
  (h3 : saplings = 58) : 
  total - oaks - saplings = 23 := by
  sorry

end NUMINAMATH_CALUDE_medium_sized_fir_trees_l3784_378460


namespace NUMINAMATH_CALUDE_solution_y_composition_l3784_378449

/-- Represents a chemical solution --/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions --/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

def is_valid_solution (s : Solution) : Prop :=
  s.a + s.b = 100 ∧ s.a ≥ 0 ∧ s.b ≥ 0

def is_valid_mixture (m : Mixture) : Prop :=
  m.x_ratio ≥ 0 ∧ m.x_ratio ≤ 1

theorem solution_y_composition 
  (x : Solution)
  (y : Solution)
  (m : Mixture)
  (hx : is_valid_solution x)
  (hy : is_valid_solution y)
  (hm : is_valid_mixture m)
  (hx_comp : x.a = 40 ∧ x.b = 60)
  (hy_comp : y.a = y.b)
  (hm_comp : m.x = x ∧ m.y = y)
  (hm_ratio : m.x_ratio = 0.3)
  (hm_a : m.x_ratio * x.a + (1 - m.x_ratio) * y.a = 47) :
  y.a = 50 := by
    sorry

end NUMINAMATH_CALUDE_solution_y_composition_l3784_378449


namespace NUMINAMATH_CALUDE_probability_multiple_of_7_l3784_378429

def is_multiple_of_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def count_pairs (n : ℕ) : ℕ := n.choose 2

theorem probability_multiple_of_7 : 
  let total_pairs := count_pairs 100
  let valid_pairs := total_pairs - count_pairs (100 - 14)
  (valid_pairs : ℚ) / total_pairs = 259 / 990 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_7_l3784_378429


namespace NUMINAMATH_CALUDE_total_distance_of_trip_l3784_378436

-- Define the triangle XYZ
def Triangle (XY YZ ZX : ℝ) : Prop :=
  XY > 0 ∧ YZ > 0 ∧ ZX > 0 ∧ XY^2 = YZ^2 + ZX^2

-- Theorem statement
theorem total_distance_of_trip (XY YZ ZX : ℝ) 
  (h1 : Triangle XY YZ ZX) (h2 : XY = 5000) (h3 : ZX = 4000) : 
  XY + YZ + ZX = 12000 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_of_trip_l3784_378436


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3784_378468

/-- Represents the possible outcomes of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- The probability of getting heads in a single toss -/
def heads_prob : ℚ := 2/3

/-- The probability of getting tails in a single toss -/
def tails_prob : ℚ := 1/3

/-- The number of coin tosses -/
def num_tosses : ℕ := 10

/-- The target position to reach -/
def target_pos : ℤ := 6

/-- The position to avoid -/
def avoid_pos : ℤ := -3

/-- A function that calculates the probability of reaching the target position
    without hitting the avoid position in the given number of tosses -/
def prob_reach_target (heads_prob : ℚ) (tails_prob : ℚ) (num_tosses : ℕ) 
                      (target_pos : ℤ) (avoid_pos : ℤ) : ℚ :=
  sorry

theorem coin_toss_probability : 
  prob_reach_target heads_prob tails_prob num_tosses target_pos avoid_pos = 5120/59049 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3784_378468


namespace NUMINAMATH_CALUDE_min_cost_verification_l3784_378426

/-- Represents a set of weights -/
def WeightSet := List Nat

/-- Cost of using a weight once -/
def weighing_cost : Nat := 100

/-- The range of possible diamond masses -/
def diamond_range : Set Nat := Finset.range 15

/-- Checks if a set of weights can measure all masses in the given range -/
def can_measure_all (weights : WeightSet) (range : Set Nat) : Prop :=
  ∀ n ∈ range, ∃ subset : List Nat, subset.toFinset ⊆ weights.toFinset ∧ subset.sum = n

/-- Calculates the minimum number of weighings needed for a given set of weights -/
def min_weighings (weights : WeightSet) : Nat :=
  weights.length + 1

/-- Calculates the total cost for a given number of weighings -/
def total_cost (num_weighings : Nat) : Nat :=
  num_weighings * weighing_cost

/-- The optimal set of weights for measuring masses from 1 to 15 -/
def optimal_weights : WeightSet := [1, 2, 4, 8]

theorem min_cost_verification :
  can_measure_all optimal_weights diamond_range →
  total_cost (min_weighings optimal_weights) = 800 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_verification_l3784_378426


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l3784_378401

theorem sum_of_squares_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_of_roots_specific :
  let a : ℝ := 10
  let b : ℝ := 15
  let c : ℝ := -20
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 25/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_roots_specific_l3784_378401


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3784_378434

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_value 
  (l1 : ℝ → ℝ → Prop) 
  (l2 : ℝ → ℝ → Prop)
  (a : ℝ) 
  (h1 : ∀ x y, l1 x y ↔ x + 2*a*y - 1 = 0)
  (h2 : ∀ x y, l2 x y ↔ x - 4*y = 0)
  (h3 : perpendicular (2*a) (1/4)) :
  a = 1/8 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3784_378434


namespace NUMINAMATH_CALUDE_sum_product_theorem_l3784_378484

theorem sum_product_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 213) 
  (h2 : a + b + c = 15) : 
  a*b + b*c + c*a = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l3784_378484


namespace NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l3784_378402

theorem least_integer_square_72_more_than_double :
  ∃ (x : ℤ), x^2 = 2*x + 72 ∧ ∀ (y : ℤ), y^2 = 2*y + 72 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_72_more_than_double_l3784_378402


namespace NUMINAMATH_CALUDE_correct_people_left_l3784_378445

/-- Calculates the number of people left on a train after two stops -/
def peopleLeftOnTrain (initialPeople : ℕ) (peopleGotOff : ℕ) (peopleGotOn : ℕ) : ℕ :=
  initialPeople - peopleGotOff + peopleGotOn

theorem correct_people_left : peopleLeftOnTrain 123 58 37 = 102 := by
  sorry

end NUMINAMATH_CALUDE_correct_people_left_l3784_378445


namespace NUMINAMATH_CALUDE_modulus_of_z_is_5_l3784_378455

-- Define the complex number z
def z : ℂ := (2 - Complex.I) ^ 2

-- Theorem stating that the modulus of z is 5
theorem modulus_of_z_is_5 : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_is_5_l3784_378455


namespace NUMINAMATH_CALUDE_fruit_basket_combinations_l3784_378418

def num_apple_options : ℕ := 7
def num_orange_options : ℕ := 13

def total_combinations : ℕ := num_apple_options * num_orange_options

theorem fruit_basket_combinations :
  total_combinations - 1 = 90 := by sorry

end NUMINAMATH_CALUDE_fruit_basket_combinations_l3784_378418


namespace NUMINAMATH_CALUDE_cube_and_square_root_problem_l3784_378441

theorem cube_and_square_root_problem (a b : ℝ) :
  (2*b - 2*a)^(1/3) = -2 →
  (4*a + 3*b)^(1/2) = 3 →
  (a = 3 ∧ b = -1) ∧ ((5*a - b)^(1/2) = 4 ∨ (5*a - b)^(1/2) = -4) :=
by sorry

end NUMINAMATH_CALUDE_cube_and_square_root_problem_l3784_378441


namespace NUMINAMATH_CALUDE_number_of_observations_l3784_378494

theorem number_of_observations
  (initial_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (h1 : initial_mean = 41)
  (h2 : incorrect_value = 23)
  (h3 : correct_value = 48)
  (h4 : corrected_mean = 41.5) :
  ∃ n : ℕ, n * initial_mean - incorrect_value + correct_value = n * corrected_mean ∧ n = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_of_observations_l3784_378494


namespace NUMINAMATH_CALUDE_distribute_5_3_l3784_378496

/-- The number of ways to distribute n distinct objects into k identical containers,
    allowing empty containers. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 51 ways to distribute 5 distinct objects into 3 identical containers,
    allowing empty containers. -/
theorem distribute_5_3 : distribute 5 3 = 51 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3784_378496


namespace NUMINAMATH_CALUDE_josh_pencils_l3784_378422

theorem josh_pencils (pencils_given : ℕ) (pencils_left : ℕ) 
  (h1 : pencils_given = 31) 
  (h2 : pencils_left = 111) : 
  pencils_given + pencils_left = 142 := by
  sorry

end NUMINAMATH_CALUDE_josh_pencils_l3784_378422


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_three_l3784_378463

theorem three_digit_divisible_by_three :
  ∀ n : ℕ,
  (n ≥ 100 ∧ n < 1000) →  -- Three-digit number
  (n % 10 = 4) →  -- Units digit is 4
  (n / 100 = 4) →  -- Hundreds digit is 4
  (n % 3 = 0) →  -- Divisible by 3
  (n = 414 ∨ n = 444 ∨ n = 474) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_three_l3784_378463


namespace NUMINAMATH_CALUDE_inequalities_with_squares_and_roots_l3784_378458

theorem inequalities_with_squares_and_roots (a b : ℝ) : 
  (a > 0 ∧ b > 0 ∧ a^2 - b^2 = 1 → a - b ≤ 1) ∧
  (a > 0 ∧ b > 0 ∧ Real.sqrt a - Real.sqrt b = 1 → a - b ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_with_squares_and_roots_l3784_378458


namespace NUMINAMATH_CALUDE_work_time_ratio_l3784_378453

-- Define the time taken by A to finish the work
def time_A : ℝ := 4

-- Define the combined work rate of A and B
def combined_work_rate : ℝ := 0.75

-- Define the time taken by B to finish the work
def time_B : ℝ := 2

-- Theorem statement
theorem work_time_ratio :
  (1 / time_A + 1 / time_B = combined_work_rate) →
  (time_B / time_A = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_work_time_ratio_l3784_378453


namespace NUMINAMATH_CALUDE_cubic_inequality_l3784_378451

theorem cubic_inequality (p q x : ℝ) : x^3 + p*x + q = 0 → 4*q*x ≤ p^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3784_378451


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3784_378411

theorem imaginary_part_of_z (z : ℂ) : (1 + z) * (1 - Complex.I) = 2 → Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3784_378411


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3784_378493

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property of forming an increasing arithmetic sequence
def is_increasing_arithmetic_seq (a b c : ℕ) : Prop :=
  fib b - fib a = fib c - fib b ∧ fib a < fib b ∧ fib b < fib c

-- State the theorem
theorem fibonacci_arithmetic_sequence :
  ∃ (a b c : ℕ),
    is_increasing_arithmetic_seq a b c ∧
    a + b + c = 2000 ∧
    a = 665 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l3784_378493


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3784_378407

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 3 + a 5 + a 7 + a 9 = 10)
  (h_diff : a 8 ^ 2 - a 2 ^ 2 = 36) :
  a 11 = 11 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l3784_378407


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l3784_378485

def algae_population (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_500 :
  (∀ k : ℕ, k < 5 → algae_population k ≤ 500) ∧
  algae_population 5 > 500 :=
sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l3784_378485


namespace NUMINAMATH_CALUDE_percentage_problem_l3784_378469

theorem percentage_problem (P : ℝ) : 
  (0.20 * 30 = P / 100 * 16 + 2) → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3784_378469


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3784_378475

/-- 
Given an infinite geometric series with common ratio 1/4 and sum 40,
the second term of the sequence is 7.5.
-/
theorem second_term_of_geometric_series (a : ℝ) : 
  (∑' n, a * (1/4)^n = 40) → a * (1/4) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3784_378475


namespace NUMINAMATH_CALUDE_sequence_2003_l3784_378492

theorem sequence_2003 (a : ℕ → ℕ) (h1 : a 1 = 0) (h2 : ∀ n : ℕ, a (n + 1) = a n + 2 * n) : 
  a 2003 = 2003 * 2002 := by
sorry

end NUMINAMATH_CALUDE_sequence_2003_l3784_378492
