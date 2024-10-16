import Mathlib

namespace NUMINAMATH_CALUDE_race_distance_l1652_165255

theorem race_distance (time_A time_B : ℝ) (lead : ℝ) (distance : ℝ) : 
  time_A = 36 →
  time_B = 45 →
  lead = 26 →
  (distance / time_B) * time_A = distance - lead →
  distance = 130 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l1652_165255


namespace NUMINAMATH_CALUDE_max_value_implies_a_l1652_165212

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + a

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∃ x ∈ Set.Icc 0 4, f a x = 3) ∧
  (∀ x ∈ Set.Icc 0 4, f a x ≤ 3) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l1652_165212


namespace NUMINAMATH_CALUDE_inequality_solution_l1652_165299

theorem inequality_solution (x : ℝ) : 
  3 - 2 / (3 * x + 4) ≤ 5 ↔ x < -4/3 ∨ x > -5/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1652_165299


namespace NUMINAMATH_CALUDE_wendy_full_face_time_l1652_165224

/-- Calculates the total time for Wendy's "full face" routine -/
def fullFaceTime (numProducts : ℕ) (waitTime : ℕ) (makeupTime : ℕ) : ℕ :=
  numProducts * waitTime + makeupTime

/-- Theorem: Wendy's "full face" routine takes 55 minutes -/
theorem wendy_full_face_time :
  fullFaceTime 5 5 30 = 55 := by
  sorry

end NUMINAMATH_CALUDE_wendy_full_face_time_l1652_165224


namespace NUMINAMATH_CALUDE_tiffany_bags_total_l1652_165274

/-- The number of bags Tiffany had on Monday -/
def monday_bags : ℕ := 4

/-- The number of bags Tiffany found the next day -/
def next_day_bags : ℕ := 8

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := monday_bags + next_day_bags

theorem tiffany_bags_total : total_bags = 12 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bags_total_l1652_165274


namespace NUMINAMATH_CALUDE_traffic_class_total_l1652_165226

/-- The number of drunk drivers in the traffic class -/
def drunk_drivers : ℕ := 6

/-- The number of speeders in the traffic class -/
def speeders : ℕ := 7 * drunk_drivers - 3

/-- The total number of students in the traffic class -/
def total_students : ℕ := drunk_drivers + speeders

/-- Theorem stating that the total number of students in the traffic class is 45 -/
theorem traffic_class_total : total_students = 45 := by sorry

end NUMINAMATH_CALUDE_traffic_class_total_l1652_165226


namespace NUMINAMATH_CALUDE_area_circle_inscribed_equilateral_triangle_l1652_165215

theorem area_circle_inscribed_equilateral_triangle (p : ℝ) (h : p > 0) :
  ∃ (R : ℝ), R > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  p = 3 * s ∧
  R = s / Real.sqrt 3 ∧
  π * R^2 = π * p^2 / 27 :=
by sorry

end NUMINAMATH_CALUDE_area_circle_inscribed_equilateral_triangle_l1652_165215


namespace NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l1652_165247

/-- The polar equation of the curve -/
def polar_equation (ρ θ : ℝ) : Prop := ρ * (Real.cos θ)^2 = 4 * Real.sin θ

/-- The Cartesian equation of the curve -/
def cartesian_equation (x y : ℝ) : Prop := x^2 = 4 * y

/-- Theorem stating that the polar equation represents a parabola -/
theorem polar_to_cartesian_parabola :
  ∀ (x y ρ θ : ℝ), 
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  polar_equation ρ θ →
  cartesian_equation x y :=
sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_parabola_l1652_165247


namespace NUMINAMATH_CALUDE_rosa_flower_count_l1652_165227

/-- Given Rosa's initial flower count and the number of flowers Andre gave her,
    prove that the total number of flowers Rosa has now is equal to the sum of these two quantities. -/
theorem rosa_flower_count (initial_flowers andre_flowers : ℕ) :
  initial_flowers = 67 →
  andre_flowers = 23 →
  initial_flowers + andre_flowers = 90 :=
by sorry

end NUMINAMATH_CALUDE_rosa_flower_count_l1652_165227


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1652_165242

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  f 0 = 1 ∧ ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2

/-- The main theorem stating that any function satisfying the functional equation
    must be of the form f(x) = x + 1 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
    ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1652_165242


namespace NUMINAMATH_CALUDE_product_of_complex_sets_l1652_165259

theorem product_of_complex_sets : ∃ (z₁ z₂ : ℂ), 
  (Complex.I * z₁ = 1) ∧ 
  (z₂ + Complex.I = 1) ∧ 
  (z₁ * z₂ = -1 - Complex.I) := by sorry

end NUMINAMATH_CALUDE_product_of_complex_sets_l1652_165259


namespace NUMINAMATH_CALUDE_pages_read_sunday_l1652_165220

def average_pages_per_day : ℕ := 50
def days_in_week : ℕ := 7
def pages_monday : ℕ := 65
def pages_tuesday : ℕ := 28
def pages_wednesday : ℕ := 0
def pages_thursday : ℕ := 70
def pages_friday : ℕ := 56
def pages_saturday : ℕ := 88

def total_pages_week : ℕ := average_pages_per_day * days_in_week
def pages_monday_to_friday : ℕ := pages_monday + pages_tuesday + pages_wednesday + pages_thursday + pages_friday
def pages_monday_to_saturday : ℕ := pages_monday_to_friday + pages_saturday

theorem pages_read_sunday : 
  total_pages_week - pages_monday_to_saturday = 43 := by sorry

end NUMINAMATH_CALUDE_pages_read_sunday_l1652_165220


namespace NUMINAMATH_CALUDE_triangle_inequality_l1652_165282

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  t.a^2 * (t.b + t.c - t.a) + t.b^2 * (t.c + t.a - t.b) + t.c^2 * (t.a + t.b - t.c) ≤ 3 * t.a * t.b * t.c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1652_165282


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1652_165280

-- Define the fixed circle F
def F (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Define the fixed line L
def L (x : ℝ) : Prop := x = 1

-- Define the trajectory of the center M
def trajectory (x y : ℝ) : Prop := y^2 = -8*x

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y : ℝ),
  (∃ (r : ℝ), r > 0 ∧
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 →
      (∃ (x_f y_f : ℝ), F x_f y_f ∧ (x' - x_f)^2 + (y' - y_f)^2 = (r + 1)^2) ∧
      (∃ (x_l : ℝ), L x_l ∧ |x' - x_l| = r))) →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1652_165280


namespace NUMINAMATH_CALUDE_optimal_renovation_solution_l1652_165238

/-- Represents a renovation team -/
structure Team where
  dailyRate : ℕ
  daysAlone : ℕ

/-- The renovation scenario -/
structure RenovationScenario where
  teamA : Team
  teamB : Team
  jointDays : ℕ
  jointCost : ℕ
  mixedDaysA : ℕ
  mixedDaysB : ℕ
  mixedCost : ℕ

/-- Theorem stating the optimal solution for the renovation scenario -/
theorem optimal_renovation_solution (scenario : RenovationScenario) 
  (h1 : scenario.jointDays * (scenario.teamA.dailyRate + scenario.teamB.dailyRate) = scenario.jointCost)
  (h2 : scenario.mixedDaysA * scenario.teamA.dailyRate + scenario.mixedDaysB * scenario.teamB.dailyRate = scenario.mixedCost)
  (h3 : scenario.teamA.daysAlone = 12)
  (h4 : scenario.teamB.daysAlone = 24)
  (h5 : scenario.jointDays = 8)
  (h6 : scenario.jointCost = 3520)
  (h7 : scenario.mixedDaysA = 6)
  (h8 : scenario.mixedDaysB = 12)
  (h9 : scenario.mixedCost = 3480) :
  scenario.teamA.dailyRate = 300 ∧ 
  scenario.teamB.dailyRate = 140 ∧ 
  scenario.teamB.daysAlone * scenario.teamB.dailyRate < scenario.teamA.daysAlone * scenario.teamA.dailyRate :=
by sorry

end NUMINAMATH_CALUDE_optimal_renovation_solution_l1652_165238


namespace NUMINAMATH_CALUDE_cube_surface_area_l1652_165279

theorem cube_surface_area (x d : ℝ) (h_volume : x^3 > 0) (h_diagonal : d > 0) : 
  ∃ (s : ℝ), s > 0 ∧ s^3 = x^3 ∧ d^2 = 3 * s^2 ∧ 6 * s^2 = 2 * d^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1652_165279


namespace NUMINAMATH_CALUDE_min_value_inequality_l1652_165289

theorem min_value_inequality (a b c : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) : 
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1652_165289


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l1652_165252

theorem cylinder_radius_proof (r : ℝ) : 
  (r > 0) →                            -- r is positive (radius)
  (2 > 0) →                            -- original height is positive
  (π * (r + 6)^2 * 2 = π * r^2 * 8) →  -- volumes are equal when increased
  r = 6 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l1652_165252


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1652_165210

-- Problem 1
theorem problem_1 : 3 * (13 / 15) - 2 * (13 / 14) + 5 * (2 / 15) - 1 * (1 / 14) = 5 := by sorry

-- Problem 2
theorem problem_2 : (1 / 9) / (2 / (3 / 4 - 2 / 3)) = 1 / 216 := by sorry

-- Problem 3
theorem problem_3 : 99 * 78.6 + 786 * 0.3 - 7.86 * 20 = 7860 := by sorry

-- Problem 4
theorem problem_4 : 2015 / (2015 * 2015 / 2016) = 2016 / 2017 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1652_165210


namespace NUMINAMATH_CALUDE_relationship_abc_l1652_165293

theorem relationship_abc : ∀ a b c : ℝ,
  a = -1/3 * 9 →
  b = 2 - 4 →
  c = 2 / (-1/2) →
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1652_165293


namespace NUMINAMATH_CALUDE_intersection_implies_sin_2α_l1652_165297

noncomputable section

-- Define the line l
def line_l (α : Real) (t : Real) : Real × Real :=
  (-1 + t * Real.cos α, -3 + t * Real.sin α)

-- Define the curve C
def curve_C (θ : Real) : Real × Real :=
  let ρ := 4 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance between two points
def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem intersection_implies_sin_2α (α : Real) :
  ∃ (t1 t2 θ1 θ2 : Real),
    let A := line_l α t1
    let B := line_l α t2
    curve_C θ1 = A ∧
    curve_C θ2 = B ∧
    distance A B = 2 →
    Real.sin (2 * α) = 2/3 := by
  sorry

end

end NUMINAMATH_CALUDE_intersection_implies_sin_2α_l1652_165297


namespace NUMINAMATH_CALUDE_sandy_shopping_percentage_l1652_165218

/-- The percentage of money Sandy spent on shopping -/
def shopping_percentage (initial_amount spent_amount : ℚ) : ℚ :=
  (spent_amount / initial_amount) * 100

/-- Proof that Sandy spent 30% of her money on shopping -/
theorem sandy_shopping_percentage :
  let initial_amount : ℚ := 200
  let remaining_amount : ℚ := 140
  let spent_amount : ℚ := initial_amount - remaining_amount
  shopping_percentage initial_amount spent_amount = 30 := by
sorry

end NUMINAMATH_CALUDE_sandy_shopping_percentage_l1652_165218


namespace NUMINAMATH_CALUDE_dissected_rectangle_perimeter_l1652_165207

/-- A rectangle dissected into nine non-overlapping squares -/
structure DissectedRectangle where
  width : ℕ+
  height : ℕ+
  squares : Fin 9 → ℕ+
  sum_squares : width * height = (squares 0).val + (squares 1).val + (squares 2).val + (squares 3).val + 
                                 (squares 4).val + (squares 5).val + (squares 6).val + (squares 7).val + 
                                 (squares 8).val

/-- The perimeter of a rectangle -/
def perimeter (rect : DissectedRectangle) : ℕ :=
  2 * (rect.width + rect.height)

/-- The theorem to be proved -/
theorem dissected_rectangle_perimeter (rect : DissectedRectangle) 
  (h_coprime : Nat.Coprime rect.width rect.height) : 
  perimeter rect = 260 := by
  sorry

end NUMINAMATH_CALUDE_dissected_rectangle_perimeter_l1652_165207


namespace NUMINAMATH_CALUDE_product_remainder_l1652_165237

theorem product_remainder (a b c d : ℕ) (ha : a = 1492) (hb : b = 1776) (hc : c = 1812) (hd : d = 1996) :
  (a * b * c * d) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1652_165237


namespace NUMINAMATH_CALUDE_carpenters_for_chairs_l1652_165262

/-- Represents the number of carpenters needed to make a certain number of chairs in a given number of days. -/
def carpenters_needed (initial_carpenters : ℕ) (initial_chairs : ℕ) (target_chairs : ℕ) : ℕ :=
  (initial_carpenters * target_chairs + initial_chairs - 1) / initial_chairs

/-- Proves that 12 carpenters are needed to make 75 chairs in 10 days, given that 8 carpenters can make 50 chairs in 10 days. -/
theorem carpenters_for_chairs : carpenters_needed 8 50 75 = 12 := by
  sorry

end NUMINAMATH_CALUDE_carpenters_for_chairs_l1652_165262


namespace NUMINAMATH_CALUDE_no_mutually_exclusive_sets_l1652_165223

/-- Represents the color of a ball -/
inductive BallColor
  | White
  | Yellow

/-- Represents the outcome of drawing two balls -/
def TwoBallDraw := (BallColor × BallColor)

/-- The set of all possible outcomes when drawing two balls from a bag with two white and two yellow balls -/
def SampleSpace : Set TwoBallDraw := sorry

/-- Event: At least one white ball -/
def AtLeastOneWhite (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.White ∨ draw.2 = BallColor.White

/-- Event: At least one yellow ball -/
def AtLeastOneYellow (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.Yellow ∨ draw.2 = BallColor.Yellow

/-- Event: Both balls are yellow -/
def BothYellow (draw : TwoBallDraw) : Prop := 
  draw.1 = BallColor.Yellow ∧ draw.2 = BallColor.Yellow

/-- Event: Exactly one white ball and one yellow ball -/
def OneWhiteOneYellow (draw : TwoBallDraw) : Prop := 
  (draw.1 = BallColor.White ∧ draw.2 = BallColor.Yellow) ∨
  (draw.1 = BallColor.Yellow ∧ draw.2 = BallColor.White)

/-- The three sets of events -/
def EventSet1 := {draw : TwoBallDraw | AtLeastOneWhite draw ∧ AtLeastOneYellow draw}
def EventSet2 := {draw : TwoBallDraw | AtLeastOneYellow draw ∧ BothYellow draw}
def EventSet3 := {draw : TwoBallDraw | OneWhiteOneYellow draw}

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (A B : Set TwoBallDraw) : Prop := A ∩ B = ∅

theorem no_mutually_exclusive_sets : 
  ¬(MutuallyExclusive EventSet1 EventSet2) ∧ 
  ¬(MutuallyExclusive EventSet1 EventSet3) ∧ 
  ¬(MutuallyExclusive EventSet2 EventSet3) := by sorry

end NUMINAMATH_CALUDE_no_mutually_exclusive_sets_l1652_165223


namespace NUMINAMATH_CALUDE_animals_per_aquarium_l1652_165256

theorem animals_per_aquarium 
  (total_animals : ℕ) 
  (num_aquariums : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : num_aquariums = 20) 
  (h3 : total_animals % num_aquariums = 0) : 
  total_animals / num_aquariums = 2 := by
sorry

end NUMINAMATH_CALUDE_animals_per_aquarium_l1652_165256


namespace NUMINAMATH_CALUDE_pradeep_marks_l1652_165294

theorem pradeep_marks (total_marks : ℕ) (pass_percentage : ℚ) (fail_margin : ℕ) : 
  total_marks = 840 → 
  pass_percentage = 1/4 → 
  (total_marks * pass_percentage).floor - fail_margin = 185 :=
by sorry

end NUMINAMATH_CALUDE_pradeep_marks_l1652_165294


namespace NUMINAMATH_CALUDE_common_term_formula_l1652_165269

def x (n : ℕ) : ℕ := 2 * n - 1
def y (n : ℕ) : ℕ := n ^ 2

def is_common_term (m : ℕ) : Prop :=
  ∃ n k : ℕ, x n = m ∧ y k = m

def c (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem common_term_formula :
  ∀ n : ℕ, is_common_term (c n) ∧
  (∀ m : ℕ, m < c n → is_common_term m → ∃ k < n, c k = m) :=
sorry

end NUMINAMATH_CALUDE_common_term_formula_l1652_165269


namespace NUMINAMATH_CALUDE_bens_age_l1652_165292

theorem bens_age (b j : ℕ) : 
  b = 3 * j + 10 →  -- Ben's age is 10 years more than thrice Jane's age
  b + j = 70 →      -- The sum of their ages is 70
  b = 55 :=         -- Ben's age is 55
by sorry

end NUMINAMATH_CALUDE_bens_age_l1652_165292


namespace NUMINAMATH_CALUDE_specific_prism_triangle_perimeter_l1652_165298

/-- A right prism with equilateral triangle bases -/
structure RightPrism where
  base_side_length : ℝ
  height : ℝ

/-- Midpoints of edges in the prism -/
structure PrismMidpoints (prism : RightPrism) where
  V : ℝ × ℝ × ℝ  -- Midpoint of PR
  W : ℝ × ℝ × ℝ  -- Midpoint of RQ
  X : ℝ × ℝ × ℝ  -- Midpoint of QT

/-- The perimeter of triangle VWX in the prism -/
def triangle_perimeter (prism : RightPrism) (midpoints : PrismMidpoints prism) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle VWX in the specific prism -/
theorem specific_prism_triangle_perimeter :
  let prism : RightPrism := { base_side_length := 10, height := 20 }
  let midpoints : PrismMidpoints prism := sorry
  triangle_perimeter prism midpoints = 5 + 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_prism_triangle_perimeter_l1652_165298


namespace NUMINAMATH_CALUDE_opposite_number_l1652_165257

theorem opposite_number (x : ℤ) : (- x = 2016) → (x = -2016) := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_l1652_165257


namespace NUMINAMATH_CALUDE_square_difference_l1652_165287

theorem square_difference (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) :
  (x - y)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l1652_165287


namespace NUMINAMATH_CALUDE_alice_bracelet_profit_l1652_165214

def bracelet_profit (initial_bracelets : ℕ) (material_cost : ℚ) 
                    (given_away : ℕ) (selling_price : ℚ) : ℚ :=
  let remaining_bracelets := initial_bracelets - given_away
  let revenue := (remaining_bracelets : ℚ) * selling_price
  revenue - material_cost

theorem alice_bracelet_profit :
  bracelet_profit 52 3 8 (1/4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_alice_bracelet_profit_l1652_165214


namespace NUMINAMATH_CALUDE_second_number_value_l1652_165291

theorem second_number_value (A B C : ℝ) (h1 : A + B + C = 98) 
  (h2 : A / B = 2 / 3) (h3 : B / C = 5 / 8) (h4 : A > 0) (h5 : B > 0) (h6 : C > 0) : 
  B = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1652_165291


namespace NUMINAMATH_CALUDE_library_shelves_l1652_165268

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
sorry

end NUMINAMATH_CALUDE_library_shelves_l1652_165268


namespace NUMINAMATH_CALUDE_f_properties_l1652_165240

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (x - 1) * abs (x - a)

theorem f_properties :
  (∀ x : ℝ, (f (-1) x = 1 ↔ x ≤ -1 ∨ x = 1)) ∧
  (∀ a : ℝ, (StrictMono (f a) ↔ a ≥ 1/3)) ∧
  (∀ a : ℝ, a < 1 → (∀ x : ℝ, f a x ≥ 2*x - 3) ↔ a ∈ Set.Icc (-3) 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1652_165240


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1652_165288

theorem sqrt_product_simplification : Real.sqrt 12 * Real.sqrt 75 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1652_165288


namespace NUMINAMATH_CALUDE_f_negative_five_equals_negative_five_l1652_165205

/-- Given a function f(x) = a * sin(x) + b * tan(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem f_negative_five_equals_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 1)
  (h2 : f 5 = 7) :
  f (-5) = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_five_equals_negative_five_l1652_165205


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1652_165221

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → 
  b > 0 → 
  c > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (b * c / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2 * c) →
  c / a = 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1652_165221


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1652_165283

theorem arithmetic_sequence_count :
  let first : ℤ := 162
  let last : ℤ := 42
  let diff : ℤ := -3
  let count := (last - first) / diff + 1
  count = 41 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1652_165283


namespace NUMINAMATH_CALUDE_least_possible_value_a_2008_l1652_165285

theorem least_possible_value_a_2008 (a : ℕ → ℤ) 
  (h_increasing : ∀ n : ℕ, n ≥ 1 → a n < a (n + 1))
  (h_inequality : ∀ i j k l : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ k ∧ k < l ∧ i + l = j + k → 
    a i + a l > a j + a k) :
  a 2008 ≥ 2015029 := by
sorry

end NUMINAMATH_CALUDE_least_possible_value_a_2008_l1652_165285


namespace NUMINAMATH_CALUDE_petya_cannot_equalize_coins_l1652_165200

/-- Represents the state of Petya's coins -/
structure CoinState where
  two_kopeck : ℕ
  ten_kopeck : ℕ

/-- Represents a single transaction with the machine -/
inductive Transaction
  | insert_two
  | insert_ten

/-- Applies a single transaction to the current coin state -/
def apply_transaction (state : CoinState) (t : Transaction) : CoinState :=
  match t with
  | Transaction.insert_two => CoinState.mk (state.two_kopeck - 1) (state.ten_kopeck + 5)
  | Transaction.insert_ten => CoinState.mk (state.two_kopeck + 5) (state.ten_kopeck - 1)

/-- Applies a sequence of transactions to the initial state -/
def apply_transactions (initial : CoinState) (ts : List Transaction) : CoinState :=
  ts.foldl apply_transaction initial

/-- The theorem stating that Petya cannot end up with equal coins -/
theorem petya_cannot_equalize_coins :
  ∀ (ts : List Transaction),
    let final_state := apply_transactions (CoinState.mk 1 0) ts
    final_state.two_kopeck ≠ final_state.ten_kopeck :=
by sorry


end NUMINAMATH_CALUDE_petya_cannot_equalize_coins_l1652_165200


namespace NUMINAMATH_CALUDE_negation_equivalence_l1652_165204

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1652_165204


namespace NUMINAMATH_CALUDE_square_and_cube_roots_l1652_165266

theorem square_and_cube_roots : 
  (∀ x : ℝ, x^2 = 81 → x = 3 ∨ x = -3) ∧ 
  (∀ y : ℝ, y^3 = -64/125 → y = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_square_and_cube_roots_l1652_165266


namespace NUMINAMATH_CALUDE_library_visitors_l1652_165250

theorem library_visitors (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) :
  sunday_avg = 540 →
  total_days = 30 →
  month_avg = 290 →
  let sundays : ℕ := 5
  let other_days : ℕ := total_days - sundays
  let other_days_avg : ℕ := (total_days * month_avg - sundays * sunday_avg) / other_days
  other_days_avg = 240 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l1652_165250


namespace NUMINAMATH_CALUDE_josiah_cookie_expense_l1652_165202

/-- The amount Josiah spent on cookies in March -/
def cookie_expense : ℕ → ℕ → ℕ → ℕ
| cookies_per_day, cookie_cost, days_in_march =>
  cookies_per_day * cookie_cost * days_in_march

/-- Theorem: Josiah spent 992 dollars on cookies in March -/
theorem josiah_cookie_expense :
  cookie_expense 2 16 31 = 992 := by
  sorry

end NUMINAMATH_CALUDE_josiah_cookie_expense_l1652_165202


namespace NUMINAMATH_CALUDE_acute_angles_trig_identities_l1652_165275

theorem acute_angles_trig_identities (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_tan_α : Real.tan α = 4/3)
  (h_cos_sum : Real.cos (α + β) = -Real.sqrt 5 / 5) :
  Real.cos (2*α) = -7/25 ∧ Real.tan (α - β) = -2/11 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_trig_identities_l1652_165275


namespace NUMINAMATH_CALUDE_last_four_average_l1652_165211

/-- Given a list of seven real numbers where the average of all seven is 62
    and the average of the first three is 58, prove that the average of the
    last four numbers is 65. -/
theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 58 →
  (list.drop 3).sum / 4 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l1652_165211


namespace NUMINAMATH_CALUDE_factor_expression_1_factor_expression_2_l1652_165261

-- For the first expression
theorem factor_expression_1 (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (5 * x - 4 * y) * (x + 2 * y) := by
  sorry

-- For the second expression
theorem factor_expression_2 (x a : ℝ) :
  x^2 + 2 * x - 15 - a * x - 5 * a = (x + 5) * (x - 3 - a) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_1_factor_expression_2_l1652_165261


namespace NUMINAMATH_CALUDE_cake_distribution_l1652_165270

theorem cake_distribution (total_pieces : ℕ) (eaten_percentage : ℚ) (num_sisters : ℕ) : 
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  num_sisters = 3 →
  (total_pieces - (eaten_percentage * total_pieces).floor) / num_sisters = 32 := by
sorry

end NUMINAMATH_CALUDE_cake_distribution_l1652_165270


namespace NUMINAMATH_CALUDE_marlon_lollipops_l1652_165239

/-- The number of lollipops Marlon had initially -/
def initial_lollipops : ℕ := 42

/-- The fraction of lollipops Marlon gave to Emily -/
def emily_fraction : ℚ := 2/3

/-- The number of lollipops Marlon kept for himself -/
def marlon_kept : ℕ := 4

/-- The number of lollipops Lou received -/
def lou_received : ℕ := 10

theorem marlon_lollipops :
  (initial_lollipops : ℚ) * (1 - emily_fraction) = (marlon_kept + lou_received : ℚ) := by
  sorry

#check marlon_lollipops

end NUMINAMATH_CALUDE_marlon_lollipops_l1652_165239


namespace NUMINAMATH_CALUDE_bat_lifespan_solution_l1652_165296

def bat_lifespan_problem (bat_lifespan : ℕ) : Prop :=
  let hamster_lifespan := bat_lifespan - 6
  let frog_lifespan := 4 * hamster_lifespan
  bat_lifespan + hamster_lifespan + frog_lifespan = 30

theorem bat_lifespan_solution :
  ∃ (bat_lifespan : ℕ), bat_lifespan_problem bat_lifespan ∧ bat_lifespan = 10 := by
  sorry

end NUMINAMATH_CALUDE_bat_lifespan_solution_l1652_165296


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l1652_165235

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 - x + 2 < 0 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l1652_165235


namespace NUMINAMATH_CALUDE_cheezits_bag_weight_l1652_165206

/-- Proves that the weight of each bag of Cheezits is 2 ounces, given the problem conditions. -/
theorem cheezits_bag_weight :
  ∀ (bag_weight : ℝ),
    (3 * bag_weight * 150 - 40 * 12 = 420) →
    bag_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_cheezits_bag_weight_l1652_165206


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l1652_165260

theorem consecutive_odd_numbers (x : ℤ) : 
  (∃ y z : ℤ, y = x + 2 ∧ z = x + 4 ∧ 
   Odd x ∧ Odd y ∧ Odd z ∧
   11 * x = 3 * z + 4 * y + 16) → 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l1652_165260


namespace NUMINAMATH_CALUDE_properties_of_A_l1652_165241

def A : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2^k}

theorem properties_of_A :
  (∀ a ∈ A, ∀ b : ℕ, b < 2*a - 1 → ¬(2*a ∣ b*(b+1))) ∧
  (∀ a : ℕ, a ∉ A → a ≠ 1 → ∃ b : ℕ, b < 2*a - 1 ∧ (2*a ∣ b*(b+1))) :=
sorry

end NUMINAMATH_CALUDE_properties_of_A_l1652_165241


namespace NUMINAMATH_CALUDE_boys_ratio_l1652_165208

theorem boys_ratio (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = boys + girls) 
  (h2 : boys > 0 ∧ girls > 0) 
  (h3 : (boys : ℚ) / total = 3/5 * (girls : ℚ) / total) : 
  (boys : ℚ) / total = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_boys_ratio_l1652_165208


namespace NUMINAMATH_CALUDE_bus_passengers_l1652_165233

theorem bus_passengers (initial_students : ℕ) (num_stops : ℕ) : 
  initial_students = 64 →
  num_stops = 4 →
  (initial_students : ℚ) * (2/3)^num_stops = 1024/81 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l1652_165233


namespace NUMINAMATH_CALUDE_nickel_chocolates_l1652_165277

theorem nickel_chocolates (robert : ℕ) (difference : ℕ) (nickel : ℕ) : 
  robert = 13 → 
  robert = nickel + difference → 
  difference = 9 → 
  nickel = 4 := by sorry

end NUMINAMATH_CALUDE_nickel_chocolates_l1652_165277


namespace NUMINAMATH_CALUDE_square_difference_theorem_l1652_165263

theorem square_difference_theorem (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 1/105) : 
  x^2 - y^2 = 8/1575 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l1652_165263


namespace NUMINAMATH_CALUDE_longest_pole_in_stadium_l1652_165290

theorem longest_pole_in_stadium (l w h : ℝ) (hl : l = 24) (hw : w = 18) (hh : h = 16) :
  Real.sqrt (l^2 + w^2 + h^2) = 34 := by
  sorry

end NUMINAMATH_CALUDE_longest_pole_in_stadium_l1652_165290


namespace NUMINAMATH_CALUDE_cosine_is_periodic_l1652_165278

-- Define the property of being a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop := sorry

-- Define the cosine function
def cos : ℝ → ℝ := sorry

theorem cosine_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric cos →
  IsPeriodic cos := by
  sorry

end NUMINAMATH_CALUDE_cosine_is_periodic_l1652_165278


namespace NUMINAMATH_CALUDE_banana_apple_worth_l1652_165276

theorem banana_apple_worth (banana_worth : ℚ) :
  (3 / 4 * 12 : ℚ) * banana_worth = 6 →
  (1 / 4 * 8 : ℚ) * banana_worth = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_apple_worth_l1652_165276


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1652_165203

theorem triangle_angle_B (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 1/2 * b ∧
  a > b →
  B = π/6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1652_165203


namespace NUMINAMATH_CALUDE_buffet_combinations_l1652_165295

def num_meat_options : ℕ := 4
def num_vegetable_options : ℕ := 5
def num_vegetables_to_choose : ℕ := 3
def num_dessert_options : ℕ := 4
def num_desserts_to_choose : ℕ := 2

def choose (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem buffet_combinations :
  (num_meat_options) *
  (choose num_vegetable_options num_vegetables_to_choose) *
  (choose num_dessert_options num_desserts_to_choose) = 240 := by
  sorry

end NUMINAMATH_CALUDE_buffet_combinations_l1652_165295


namespace NUMINAMATH_CALUDE_residual_plot_ordinate_l1652_165267

/-- Represents a residual plot used in residual analysis -/
structure ResidualPlot where
  /-- The ordinate of the residual plot -/
  ordinate : ℝ
  /-- The abscissa of the residual plot (could be sample number, height data, or estimated weight) -/
  abscissa : ℝ

/-- Represents a residual in statistical analysis -/
def Residual : Type := ℝ

/-- Theorem stating that the ordinate of a residual plot represents the residual -/
theorem residual_plot_ordinate (plot : ResidualPlot) : 
  ∃ (r : Residual), plot.ordinate = r :=
sorry

end NUMINAMATH_CALUDE_residual_plot_ordinate_l1652_165267


namespace NUMINAMATH_CALUDE_point_transformation_sum_l1652_165258

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 90° counterclockwise around (2, 3) -/
def rotate90 (p : Point) : Point :=
  { x := -p.y + 5, y := p.x + 1 }

/-- Reflects a point about the line y = -x -/
def reflectAboutNegativeX (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- The final transformation applied to point P -/
def finalTransform (p : Point) : Point :=
  reflectAboutNegativeX (rotate90 p)

/-- Theorem statement -/
theorem point_transformation_sum (a b : ℝ) :
  let p := Point.mk a b
  finalTransform p = Point.mk (-3) 2 → a + b = 11 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_sum_l1652_165258


namespace NUMINAMATH_CALUDE_isabel_homework_l1652_165225

/-- Given the total number of problems, completed problems, and problems per page,
    calculate the number of remaining pages. -/
def remaining_pages (total : ℕ) (completed : ℕ) (per_page : ℕ) : ℕ :=
  (total - completed) / per_page

/-- Theorem stating that given Isabel's homework conditions, 
    the number of remaining pages is 5. -/
theorem isabel_homework : 
  remaining_pages 72 32 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_isabel_homework_l1652_165225


namespace NUMINAMATH_CALUDE_letter_F_transformation_l1652_165201

-- Define the position of the letter F
structure LetterFPosition where
  base : (ℝ × ℝ) -- Base endpoint
  top : (ℝ × ℝ)  -- Top endpoint

-- Define the transformations
def reflectXAxis (p : LetterFPosition) : LetterFPosition :=
  { base := (p.base.1, -p.base.2), top := (p.top.1, -p.top.2) }

def rotateCounterClockwise90 (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.2, p.base.1), top := (-p.top.2, p.top.1) }

def rotate180 (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.1, -p.base.2), top := (-p.top.1, -p.top.2) }

def reflectYAxis (p : LetterFPosition) : LetterFPosition :=
  { base := (-p.base.1, p.base.2), top := (-p.top.1, p.top.2) }

-- Define the initial position
def initialPosition : LetterFPosition :=
  { base := (0, -1), top := (1, 0) }

-- Define the final position
def finalPosition : LetterFPosition :=
  { base := (1, 0), top := (0, 1) }

-- Theorem statement
theorem letter_F_transformation :
  (reflectYAxis ∘ rotate180 ∘ rotateCounterClockwise90 ∘ reflectXAxis) initialPosition = finalPosition := by
  sorry

end NUMINAMATH_CALUDE_letter_F_transformation_l1652_165201


namespace NUMINAMATH_CALUDE_adjacent_roll_probability_adjacent_roll_probability_proof_l1652_165229

/-- The probability that no two adjacent people roll the same number on an eight-sided die
    when six people sit around a circular table. -/
theorem adjacent_roll_probability : ℚ :=
  117649 / 262144

/-- The number of people sitting around the circular table. -/
def num_people : ℕ := 6

/-- The number of sides on the die. -/
def die_sides : ℕ := 8

/-- The probability of rolling a different number than the previous person. -/
def diff_roll_prob : ℚ := 7 / 8

theorem adjacent_roll_probability_proof :
  adjacent_roll_probability = diff_roll_prob ^ num_people :=
sorry

end NUMINAMATH_CALUDE_adjacent_roll_probability_adjacent_roll_probability_proof_l1652_165229


namespace NUMINAMATH_CALUDE_continuous_finite_preimage_implies_smp_l1652_165228

open Set

/-- Definition of "smp" property for a function -/
def IsSmp (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (n : ℕ) (c : Fin (n + 1) → ℝ),
    c 0 = a ∧ c (Fin.last n) = b ∧
    (∀ i : Fin n, c i < c (i + 1)) ∧
    (∀ i : Fin n, ∀ x ∈ Ioo (c i) (c (i + 1)),
      (f (c i) < f x ∧ f x < f (c (i + 1))) ∨
      (f (c i) > f x ∧ f x > f (c (i + 1))))

/-- Main theorem statement -/
theorem continuous_finite_preimage_implies_smp
  (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b))
  (h_finite : ∀ v : ℝ, Set.Finite {x ∈ Icc a b | f x = v}) :
  IsSmp f a b :=
sorry

end NUMINAMATH_CALUDE_continuous_finite_preimage_implies_smp_l1652_165228


namespace NUMINAMATH_CALUDE_one_pair_probability_l1652_165264

/-- The number of colors of socks --/
def num_colors : ℕ := 5

/-- The number of socks per color --/
def socks_per_color : ℕ := 2

/-- The total number of socks --/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn --/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks of the same color --/
theorem one_pair_probability : 
  (Nat.choose num_colors 4 * 4 * (2^3)) / Nat.choose total_socks socks_drawn = 40 / 63 :=
sorry

end NUMINAMATH_CALUDE_one_pair_probability_l1652_165264


namespace NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l1652_165251

/-- The probability that Xavier and Yvonne solve a problem but Zelda does not, 
    given their individual probabilities of success. -/
theorem xavier_yvonne_not_zelda_probability 
  (p_xavier : ℚ) (p_yvonne : ℚ) (p_zelda : ℚ)
  (h_xavier : p_xavier = 1/6)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/32 :=
sorry

end NUMINAMATH_CALUDE_xavier_yvonne_not_zelda_probability_l1652_165251


namespace NUMINAMATH_CALUDE_shanghai_score_is_75_l1652_165272

/-- The score of Yao Ming in the basketball game -/
def yao_ming_score : ℕ := 30

/-- The winning margin of the Shanghai team over the Beijing team -/
def shanghai_margin : ℕ := 10

/-- Calculates the total score of both teams based on Yao Ming's score -/
def total_score (yao_score : ℕ) : ℕ := 5 * yao_score - 10

/-- The score of the Shanghai team -/
def shanghai_score : ℕ := 75

/-- The score of the Beijing team -/
def beijing_score : ℕ := shanghai_score - shanghai_margin

theorem shanghai_score_is_75 :
  shanghai_score = 75 ∧
  shanghai_score - beijing_score = shanghai_margin ∧
  shanghai_score + beijing_score = total_score yao_ming_score :=
by sorry

end NUMINAMATH_CALUDE_shanghai_score_is_75_l1652_165272


namespace NUMINAMATH_CALUDE_triangle_problem_l1652_165286

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : (Real.sin t.B) / (2 * Real.sin t.A - Real.sin t.C) = 1 / (2 * Real.cos t.C))
  (h2 : t.a = 1)
  (h3 : t.b = Real.sqrt 7) :
  t.B = π / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1652_165286


namespace NUMINAMATH_CALUDE_lizzies_group_area_l1652_165248

/-- The area covered by Lizzie's group given the total area, area covered by another group, and remaining area to be cleaned. -/
theorem lizzies_group_area (total_area other_group_area remaining_area : ℕ) 
  (h1 : total_area = 900)
  (h2 : other_group_area = 265)
  (h3 : remaining_area = 385) :
  total_area - other_group_area - remaining_area = 250 := by
  sorry

end NUMINAMATH_CALUDE_lizzies_group_area_l1652_165248


namespace NUMINAMATH_CALUDE_inequality_proof_l1652_165284

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1652_165284


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l1652_165273

theorem fraction_product_simplification :
  (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l1652_165273


namespace NUMINAMATH_CALUDE_indefinite_stick_shortening_l1652_165216

theorem indefinite_stick_shortening :
  ∃ (f : ℕ → ℝ × ℝ × ℝ),
    (∀ n : ℕ, (f n).1 > 0 ∧ (f n).2.1 > 0 ∧ (f n).2.2 > 0) ∧
    (∀ n : ℕ, max (f n).1 (max (f n).2.1 (f n).2.2) > (f n).1 + (f n).2.1 + (f n).2.2 - max (f n).1 (max (f n).2.1 (f n).2.2)) ∧
    (∀ n : ℕ, 
      let (a, b, c) := f n
      let m := max a (max b c)
      f (n + 1) = (if m = a then (b + c, b, c) else if m = b then (a, a + c, c) else (a, b, a + b))) :=
sorry

end NUMINAMATH_CALUDE_indefinite_stick_shortening_l1652_165216


namespace NUMINAMATH_CALUDE_solve_system_l1652_165245

theorem solve_system (p q : ℝ) (eq1 : 2 * p + 5 * q = 7) (eq2 : 5 * p + 2 * q = 16) : q = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1652_165245


namespace NUMINAMATH_CALUDE_triangle_side_length_l1652_165271

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  A = π / 3 →  -- 60° in radians
  B = π / 4 →  -- 45° in radians
  a = 3 →
  b = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1652_165271


namespace NUMINAMATH_CALUDE_black_go_stones_l1652_165234

theorem black_go_stones (total : ℕ) (difference : ℕ) (black : ℕ) (white : ℕ) : 
  total = 1256 → 
  difference = 408 → 
  total = black + white → 
  white = black + difference → 
  black = 424 := by
sorry

end NUMINAMATH_CALUDE_black_go_stones_l1652_165234


namespace NUMINAMATH_CALUDE_markers_per_box_l1652_165244

theorem markers_per_box (initial_markers : ℕ) (new_boxes : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32)
  (h2 : new_boxes = 6)
  (h3 : total_markers = 86) :
  (total_markers - initial_markers) / new_boxes = 9 :=
by sorry

end NUMINAMATH_CALUDE_markers_per_box_l1652_165244


namespace NUMINAMATH_CALUDE_tissue_box_price_l1652_165253

-- Define the quantities and prices
def toilet_paper_rolls : ℕ := 10
def paper_towel_rolls : ℕ := 7
def tissue_boxes : ℕ := 3
def toilet_paper_price : ℚ := 1.5
def paper_towel_price : ℚ := 2
def total_cost : ℚ := 35

-- Theorem to prove
theorem tissue_box_price : 
  (total_cost - (toilet_paper_rolls * toilet_paper_price + paper_towel_rolls * paper_towel_price)) / tissue_boxes = 2 := by
  sorry

end NUMINAMATH_CALUDE_tissue_box_price_l1652_165253


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l1652_165222

theorem least_three_digit_multiple_of_nine : ∃ n : ℕ, 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  n % 9 = 0 ∧
  ∀ m : ℕ, (m ≥ 100 ∧ m ≤ 999 ∧ m % 9 = 0) → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l1652_165222


namespace NUMINAMATH_CALUDE_factorization_theorem_l1652_165230

theorem factorization_theorem (x y : ℝ) :
  4 * (x + y)^2 - (x^2 - y^2)^2 = (x + y)^2 * (2 + x - y) * (2 - x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l1652_165230


namespace NUMINAMATH_CALUDE_statement_holds_only_in_specific_cases_l1652_165265

-- Define the basic types
inductive GeometricObject
| Line
| Plane

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the statement
def statement (x y z : GeometricObject) : Prop :=
  perpendicular x z → perpendicular y z → parallel x y

-- Theorem to prove
theorem statement_holds_only_in_specific_cases 
  (x y z : GeometricObject) : 
  statement x y z ↔ 
    ((x = GeometricObject.Line ∧ y = GeometricObject.Line ∧ z = GeometricObject.Plane) ∨
     (x = GeometricObject.Plane ∧ y = GeometricObject.Plane ∧ z = GeometricObject.Line)) :=
by sorry

end NUMINAMATH_CALUDE_statement_holds_only_in_specific_cases_l1652_165265


namespace NUMINAMATH_CALUDE_particle_position_at_1989_l1652_165209

/-- Represents the position of a particle on a 2D plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Defines the movement pattern of the particle -/
def move (t : ℕ) : Position :=
  sorry

/-- The theorem to be proved -/
theorem particle_position_at_1989 :
  move 1989 = Position.mk 0 0 := by
  sorry

end NUMINAMATH_CALUDE_particle_position_at_1989_l1652_165209


namespace NUMINAMATH_CALUDE_unique_rectangle_pieces_l1652_165231

theorem unique_rectangle_pieces :
  ∀ (a b : ℕ),
    a < b →
    (49 * 51) % (a * b) = 0 →
    (99 * 101) % (a * b) = 0 →
    a = 1 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_pieces_l1652_165231


namespace NUMINAMATH_CALUDE_antonella_remaining_money_l1652_165254

/-- Represents the types of Canadian coins -/
inductive CanadianCoin
  | Loonie
  | Toonie

/-- The value of a Canadian coin in dollars -/
def coin_value : CanadianCoin → ℕ
  | CanadianCoin.Loonie => 1
  | CanadianCoin.Toonie => 2

/-- Calculates the total value of coins -/
def total_value (coins : List CanadianCoin) : ℕ :=
  coins.map coin_value |>.sum

theorem antonella_remaining_money :
  let total_coins : ℕ := 10
  let toonie_count : ℕ := 4
  let loonie_count : ℕ := total_coins - toonie_count
  let initial_coins : List CanadianCoin := 
    List.replicate toonie_count CanadianCoin.Toonie ++ List.replicate loonie_count CanadianCoin.Loonie
  let frappuccino_cost : ℕ := 3
  total_value initial_coins - frappuccino_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_antonella_remaining_money_l1652_165254


namespace NUMINAMATH_CALUDE_vodka_alcohol_consumption_l1652_165217

/-- Calculates the amount of pure alcohol consumed by one person when splitting vodka -/
theorem vodka_alcohol_consumption 
  (total_shots : ℕ) 
  (ounces_per_shot : ℚ) 
  (alcohol_percentage : ℚ) 
  (num_people : ℕ) 
  (h1 : total_shots = 8)
  (h2 : ounces_per_shot = 3/2)
  (h3 : alcohol_percentage = 1/2)
  (h4 : num_people = 2) :
  (total_shots : ℚ) * ounces_per_shot * alcohol_percentage / num_people = 3 := by
  sorry

end NUMINAMATH_CALUDE_vodka_alcohol_consumption_l1652_165217


namespace NUMINAMATH_CALUDE_pipeline_equation_l1652_165249

/-- Represents the equation for a pipeline project with increased construction speed --/
theorem pipeline_equation (x : ℝ) (h : x > 0) :
  (35 / x) - (35 / ((1 + 0.2) * x)) = 7 :=
sorry

end NUMINAMATH_CALUDE_pipeline_equation_l1652_165249


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1652_165236

theorem inequality_solution_set (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1652_165236


namespace NUMINAMATH_CALUDE_distinct_centroids_count_l1652_165219

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℚ
  height : ℚ

/-- Represents the distribution of points on the perimeter of a rectangle -/
structure PerimeterPoints where
  total : ℕ
  long_side : ℕ
  short_side : ℕ

/-- Calculates the number of distinct centroid positions for triangles formed by
    any three non-collinear points from the specified points on the rectangle's perimeter -/
def count_distinct_centroids (rect : Rectangle) (points : PerimeterPoints) : ℕ :=
  sorry

/-- The main theorem stating that for a 12x8 rectangle with 48 equally spaced points
    on its perimeter, there are 925 distinct centroid positions -/
theorem distinct_centroids_count :
  let rect := Rectangle.mk 12 8
  let points := PerimeterPoints.mk 48 16 8
  count_distinct_centroids rect points = 925 := by
  sorry

end NUMINAMATH_CALUDE_distinct_centroids_count_l1652_165219


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1652_165243

theorem abs_sum_inequality (x : ℝ) : |x + 3| + |x - 4| < 8 ↔ 4 ≤ x ∧ x < 4.5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1652_165243


namespace NUMINAMATH_CALUDE_replaced_person_age_l1652_165213

theorem replaced_person_age 
  (n : ℕ) 
  (original_total_age : ℕ) 
  (new_person_age : ℕ) 
  (average_decrease : ℕ) :
  n = 10 →
  new_person_age = 10 →
  average_decrease = 3 →
  (original_total_age : ℚ) / n - average_decrease = 
    (original_total_age - (original_total_age / n * n - new_person_age) : ℚ) / n →
  (original_total_age / n * n - new_person_age : ℚ) / n = 40 :=
by sorry

end NUMINAMATH_CALUDE_replaced_person_age_l1652_165213


namespace NUMINAMATH_CALUDE_system_solution_l1652_165246

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 17 - 2*x)
  (eq2 : x + z = 1 - 2*y)
  (eq3 : x + y = 8 - 2*z) : 
  x + y + z = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1652_165246


namespace NUMINAMATH_CALUDE_prob_all_players_five_coins_l1652_165281

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor
| White : BallColor

/-- Represents the state of the game after each round -/
structure GameState :=
(coins : Player → ℕ)
(round : ℕ)

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
sorry

/-- The probability of drawing both green and blue balls by the same player in a single round -/
def prob_green_blue_same_player : ℚ :=
1 / 20

/-- The number of rounds in the game -/
def num_rounds : ℕ := 5

/-- The initial number of coins for each player -/
def initial_coins : ℕ := 5

/-- The probability that each player has exactly 5 coins after 5 rounds -/
theorem prob_all_players_five_coins :
  (prob_green_blue_same_player ^ num_rounds : ℚ) = 1 / 3200000 :=
sorry

end NUMINAMATH_CALUDE_prob_all_players_five_coins_l1652_165281


namespace NUMINAMATH_CALUDE_bottles_per_case_l1652_165232

theorem bottles_per_case (april_cases : ℕ) (may_cases : ℕ) (total_bottles : ℕ) : 
  april_cases = 20 → may_cases = 30 → total_bottles = 1000 →
  ∃ (bottles_per_case : ℕ), bottles_per_case * (april_cases + may_cases) = total_bottles ∧ bottles_per_case = 20 :=
by sorry

end NUMINAMATH_CALUDE_bottles_per_case_l1652_165232
