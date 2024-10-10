import Mathlib

namespace two_digit_number_sum_l3_386

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 3 * (a + b) →
  (10 * a + b) + (10 * b + a) = 33 := by
sorry

end two_digit_number_sum_l3_386


namespace triangle_area_arithmetic_angles_l3_382

/-- Given a triangle ABC where angles A, B, and C form an arithmetic sequence,
    and sides a = 1 and b = √3, the area of the triangle is √3/2. -/
theorem triangle_area_arithmetic_angles (A B C : ℝ) (a b c : ℝ) : 
  A + C = 2 * B → -- angles form arithmetic sequence
  A + B + C = π → -- sum of angles in a triangle
  a = 1 → -- given side length
  b = Real.sqrt 3 → -- given side length
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_arithmetic_angles_l3_382


namespace standard_deviation_proof_l3_365

/-- The standard deviation of a test score distribution. -/
def standard_deviation : ℝ := 20

/-- The mean score of the test. -/
def mean_score : ℝ := 60

/-- The lowest possible score within 2 standard deviations of the mean. -/
def lowest_score : ℝ := 20

/-- Theorem stating that the standard deviation is correct given the conditions. -/
theorem standard_deviation_proof :
  lowest_score = mean_score - 2 * standard_deviation :=
by sorry

end standard_deviation_proof_l3_365


namespace inequality_system_solution_l3_369

theorem inequality_system_solution :
  {x : ℝ | (5 * x + 3 > 3 * (x - 1)) ∧ ((8 * x + 2) / 9 > x)} = {x : ℝ | -3 < x ∧ x < 2} := by
  sorry

end inequality_system_solution_l3_369


namespace max_intersections_circle_line_parabola_l3_320

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in a 2D plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The maximum number of intersections between a circle and a line --/
def max_intersections_circle_line : ℕ := 2

/-- The maximum number of intersections between a parabola and a line --/
def max_intersections_parabola_line : ℕ := 2

/-- The maximum number of intersections between a circle and a parabola --/
def max_intersections_circle_parabola : ℕ := 4

/-- Theorem: The maximum number of intersections between a circle, a line, and a parabola is 8 --/
theorem max_intersections_circle_line_parabola 
  (c : Circle) (l : Line) (p : Parabola) : 
  max_intersections_circle_line + 
  max_intersections_parabola_line + 
  max_intersections_circle_parabola = 8 := by
  sorry

end max_intersections_circle_line_parabola_l3_320


namespace min_additional_squares_for_symmetry_l3_361

/-- Represents a point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- Represents the grid and its shaded squares --/
structure Grid where
  size : Nat
  shaded : List Point

/-- Checks if a grid has horizontal, vertical, and diagonal symmetry --/
def hasSymmetry (g : Grid) : Bool := sorry

/-- Counts the number of additional squares needed for symmetry --/
def additionalSquaresForSymmetry (g : Grid) : Nat := sorry

/-- The initial grid configuration --/
def initialGrid : Grid := {
  size := 6,
  shaded := [⟨2, 5⟩, ⟨3, 3⟩, ⟨4, 2⟩, ⟨6, 1⟩]
}

theorem min_additional_squares_for_symmetry :
  additionalSquaresForSymmetry initialGrid = 9 := by sorry

end min_additional_squares_for_symmetry_l3_361


namespace sqrt_equation_solutions_l3_380

theorem sqrt_equation_solutions (x : ℝ) : 
  (Real.sqrt (9 * x - 4) + 18 / Real.sqrt (9 * x - 4) = 10) ↔ (x = 85 / 9 ∨ x = 8 / 9) :=
by sorry

end sqrt_equation_solutions_l3_380


namespace polygon_sides_from_angle_sum_l3_383

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 900 → (n - 2) * 180 = angle_sum → n = 7 := by
  sorry

end polygon_sides_from_angle_sum_l3_383


namespace fraction_simplification_implies_even_difference_l3_367

theorem fraction_simplification_implies_even_difference 
  (a b c d : ℕ) (h1 : ∀ n : ℕ, c * n + d ≠ 0) 
  (h2 : ∀ n : ℕ, ∃ k : ℕ, a * n + b = 2 * k ∧ c * n + d = 2 * k) : 
  Even (a * d - b * c) := by
  sorry

end fraction_simplification_implies_even_difference_l3_367


namespace shelf_books_count_l3_311

/-- The number of books on a shelf after adding more books -/
def total_books (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of books on the shelf is 48 -/
theorem shelf_books_count : total_books 38 10 = 48 := by
  sorry

end shelf_books_count_l3_311


namespace friday_occurs_five_times_in_september_l3_327

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents months of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June
  | July
  | August
  | September
  | October
  | November
  | December

/-- Structure representing a year with specific properties -/
structure Year where
  julySundayCount : Nat
  februaryLeap : Bool
  septemberDayCount : Nat

/-- Function to determine the day that occurs five times in September -/
def dayOccurringFiveTimesInSeptember (y : Year) : DayOfWeek :=
  sorry

/-- Theorem stating that Friday occurs five times in September under given conditions -/
theorem friday_occurs_five_times_in_september (y : Year) 
    (h1 : y.julySundayCount = 5)
    (h2 : y.februaryLeap = true)
    (h3 : y.septemberDayCount = 30) :
    dayOccurringFiveTimesInSeptember y = DayOfWeek.Friday := by
  sorry


end friday_occurs_five_times_in_september_l3_327


namespace divisibility_of_square_l3_350

theorem divisibility_of_square (n : ℕ) (h1 : n > 0) (h2 : ∀ d : ℕ, d > 0 → d ∣ n → d ≤ 30) :
  900 ∣ n^2 := by
  sorry

end divisibility_of_square_l3_350


namespace only_one_divides_power_minus_one_l3_349

theorem only_one_divides_power_minus_one : 
  ∀ n : ℕ, n > 0 → n ∣ (2^n - 1) → n = 1 := by
  sorry

end only_one_divides_power_minus_one_l3_349


namespace chord_count_l3_393

/-- The number of chords formed by connecting any two of n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- There are 9 points on the circumference of a circle -/
def num_points : ℕ := 9

theorem chord_count : num_chords num_points = 36 := by
  sorry

end chord_count_l3_393


namespace stars_count_theorem_l3_324

theorem stars_count_theorem (east : ℕ) (west_percent : ℕ) : 
  east = 120 → west_percent = 473 → 
  east + (east * (west_percent : ℚ) / 100).ceil = 688 := by
sorry

end stars_count_theorem_l3_324


namespace taxi_fare_formula_l3_385

/-- Represents the taxi fare function for distances greater than 3 km -/
def taxiFare (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

/-- Theorem stating that the taxi fare function is equivalent to 2x + 4 for x > 3 -/
theorem taxi_fare_formula (x : ℝ) (h : x > 3) :
  taxiFare x = 2 * x + 4 := by
  sorry

end taxi_fare_formula_l3_385


namespace share_of_y_l3_346

theorem share_of_y (total : ℝ) (x y z : ℝ) : 
  total = 273 →
  y = (45/100) * x →
  z = (50/100) * x →
  total = x + y + z →
  y = 63 := by
sorry

end share_of_y_l3_346


namespace system_solution_l3_336

theorem system_solution : 
  ∃! (x y z : ℝ), 
    x * (y + z) * (x + y + z) = 1170 ∧ 
    y * (z + x) * (x + y + z) = 1008 ∧ 
    z * (x + y) * (x + y + z) = 1458 ∧ 
    x = 5 ∧ y = 4 ∧ z = 9 := by
  sorry

end system_solution_l3_336


namespace optimal_plan_is_best_three_valid_plans_l3_341

/-- Represents a purchasing plan for machines --/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a purchase plan is valid according to the given conditions --/
def isValidPlan (p : PurchasePlan) : Prop :=
  p.typeA + p.typeB = 6 ∧
  7 * p.typeA + 5 * p.typeB ≤ 34 ∧
  100 * p.typeA + 60 * p.typeB ≥ 380

/-- Calculates the total cost of a purchase plan --/
def totalCost (p : PurchasePlan) : ℕ :=
  7 * p.typeA + 5 * p.typeB

/-- The optimal purchase plan --/
def optimalPlan : PurchasePlan :=
  { typeA := 1, typeB := 5 }

/-- Theorem stating that the optimal plan is valid and minimizes cost --/
theorem optimal_plan_is_best :
  isValidPlan optimalPlan ∧
  ∀ p : PurchasePlan, isValidPlan p → totalCost optimalPlan ≤ totalCost p :=
sorry

/-- Theorem stating that there are exactly 3 valid purchase plans --/
theorem three_valid_plans :
  ∃! (plans : List PurchasePlan),
    plans.length = 3 ∧
    ∀ p : PurchasePlan, isValidPlan p ↔ p ∈ plans :=
sorry

end optimal_plan_is_best_three_valid_plans_l3_341


namespace problem_solution_l3_397

theorem problem_solution (a b c d x y : ℝ) 
  (h1 : a = -b)
  (h2 : c * d = 1)
  (h3 : (x + 3)^2 + |y - 2| = 0) :
  2*(a + b) - 2*(c*d)^4 + (x + y)^2022 = -1 := by
  sorry

end problem_solution_l3_397


namespace catch_up_time_l3_384

/-- Two people walk in opposite directions at the same speed for 10 minutes,
    then one increases speed by 5 times and chases the other. -/
theorem catch_up_time (s : ℝ) (h : s > 0) : 
  let initial_distance := 2 * 10 * s
  let relative_speed := 5 * s - s
  initial_distance / relative_speed = 5 :=
by sorry

end catch_up_time_l3_384


namespace simplify_rational_expression_l3_399

theorem simplify_rational_expression (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 2) (h4 : x ≠ 5) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 4) * (x - 2)) := by
  sorry

end simplify_rational_expression_l3_399


namespace toys_remaining_l3_348

theorem toys_remaining (initial_stock : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) 
  (h1 : initial_stock = 83) 
  (h2 : sold_week1 = 38) 
  (h3 : sold_week2 = 26) :
  initial_stock - (sold_week1 + sold_week2) = 19 :=
by
  sorry

end toys_remaining_l3_348


namespace homework_completion_l3_396

/-- Fraction of homework done on Monday night -/
def monday_fraction : ℚ := sorry

/-- Fraction of homework done on Tuesday night -/
def tuesday_fraction (x : ℚ) : ℚ := (1 - x) / 3

/-- Fraction of homework done on Wednesday night -/
def wednesday_fraction : ℚ := 4 / 15

theorem homework_completion (x : ℚ) :
  x + tuesday_fraction x + wednesday_fraction = 1 →
  x = 3 / 5 := by sorry

end homework_completion_l3_396


namespace gold_coins_percentage_l3_364

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  beads_percent : ℝ
  papers_percent : ℝ
  silver_coins_percent : ℝ
  gold_coins_percent : ℝ

/-- Theorem stating the percentage of gold coins in the urn -/
theorem gold_coins_percentage (u : UrnComposition) 
  (h1 : u.beads_percent = 15)
  (h2 : u.papers_percent = 10)
  (h3 : u.silver_coins_percent + u.gold_coins_percent = 75)
  (h4 : u.silver_coins_percent = 0.3 * 75) :
  u.gold_coins_percent = 52.5 := by
  sorry

#check gold_coins_percentage

end gold_coins_percentage_l3_364


namespace divisibility_implies_multiple_of_three_l3_395

theorem divisibility_implies_multiple_of_three (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 
  3 ∣ n := by
  sorry

end divisibility_implies_multiple_of_three_l3_395


namespace parabola_coefficient_l3_332

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, h) and y-intercept at (0, -2h),
    where h ≠ 0, prove that b = 6 -/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + h) →
  a * h^2 + h = -2 * h →
  b = 6 := by
  sorry

end parabola_coefficient_l3_332


namespace division_problem_l3_322

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 725 →
  divisor = 36 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 20 := by
  sorry

end division_problem_l3_322


namespace sector_central_angle_l3_328

/-- Given a circular sector with circumference 10 and area 4, 
    prove that its central angle in radians is 1/2 -/
theorem sector_central_angle (r l : ℝ) : 
  (2 * r + l = 10) →  -- circumference condition
  ((1 / 2) * l * r = 4) →  -- area condition
  (l / r = 1 / 2) :=  -- central angle in radians
by sorry

end sector_central_angle_l3_328


namespace train_problem_l3_388

/-- The length of the longer train given the conditions of the problem -/
def longer_train_length : ℝ := 319.96

theorem train_problem (train1_length train1_speed train2_speed clearing_time : ℝ) 
  (h1 : train1_length = 160)
  (h2 : train1_speed = 42)
  (h3 : train2_speed = 30)
  (h4 : clearing_time = 23.998) : 
  longer_train_length = 319.96 := by
  sorry

#check train_problem

end train_problem_l3_388


namespace max_rectangles_in_modified_grid_l3_321

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular cut-out --/
structure Cutout :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a grid --/
def gridArea (g : Grid) : ℕ :=
  g.rows * g.cols

/-- Calculates the area of a cutout --/
def cutoutArea (c : Cutout) : ℕ :=
  c.width * c.height

/-- Calculates the remaining area after cutouts --/
def remainingArea (g : Grid) (cutouts : List Cutout) : ℕ :=
  gridArea g - (cutouts.map cutoutArea).sum

/-- Theorem: Maximum number of 1x3 rectangles in modified 8x8 grid --/
theorem max_rectangles_in_modified_grid :
  let initial_grid : Grid := ⟨8, 8⟩
  let cutouts : List Cutout := [⟨2, 2⟩, ⟨2, 2⟩, ⟨2, 2⟩]
  let remaining_cells := remainingArea initial_grid cutouts
  (remaining_cells / 3 : ℕ) = 17 :=
by sorry

end max_rectangles_in_modified_grid_l3_321


namespace janes_number_exists_and_unique_l3_316

theorem janes_number_exists_and_unique :
  ∃! n : ℕ,
    200 ∣ n ∧
    45 ∣ n ∧
    500 < n ∧
    n < 2500 ∧
    Even n :=
by
  sorry

end janes_number_exists_and_unique_l3_316


namespace sphere_surface_area_l3_375

theorem sphere_surface_area (C : ℝ) (h : C = 4 * Real.pi) :
  ∃ (S : ℝ), S = 16 * Real.pi ∧ S = 4 * Real.pi * (C / (2 * Real.pi))^2 := by
  sorry

end sphere_surface_area_l3_375


namespace sum_of_constants_l3_302

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, (x - a) / (x + b) = (x^2 - 50*x + 621) / (x^2 + 75*x - 3400)) → 
  a + b = 112 := by
sorry

end sum_of_constants_l3_302


namespace sheets_per_day_l3_305

def sheets_per_pad : ℕ := 60
def working_days_per_week : ℕ := 5

theorem sheets_per_day :
  sheets_per_pad / working_days_per_week = 12 := by
  sorry

end sheets_per_day_l3_305


namespace acute_triangle_properties_l3_313

/-- Properties of an acute triangle ABC -/
structure AcuteTriangle where
  -- Sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angle_sum : A + B + C = π
  side_angle_relation : Real.sqrt 3 * Real.sin C - Real.cos B = Real.cos (A - C)
  side_a : a = 2 * Real.sqrt 3
  area : 1/2 * b * c * Real.sin A = 3 * Real.sqrt 3

/-- Theorem about the properties of the specified acute triangle -/
theorem acute_triangle_properties (t : AcuteTriangle) : 
  t.A = π/3 ∧ t.b + t.c = 4 * Real.sqrt 3 := by
  sorry

end acute_triangle_properties_l3_313


namespace vector_relations_l3_331

def vector_a : Fin 2 → ℝ := ![2, 3]
def vector_b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem vector_relations :
  (∃ x : ℝ, parallel vector_a (vector_b x) ↔ x = -4) ∧
  (∃ x : ℝ, perpendicular vector_a (vector_b x) ↔ x = 9) := by
  sorry

end vector_relations_l3_331


namespace tangent_parallel_x_axis_tangent_parallel_line_l3_372

-- Define the curve
def x (t : ℝ) : ℝ := t - 1
def y (t : ℝ) : ℝ := t^3 - 12*t + 1

-- Define the derivative of y with respect to x
def dy_dx (t : ℝ) : ℝ := 3*t^2 - 12

-- Define the slope of the line 9x + y + 3 = 0
def m : ℝ := -9

-- Theorem for points where tangent is parallel to x-axis
theorem tangent_parallel_x_axis :
  ∃ t₁ t₂ : ℝ, 
    t₁ ≠ t₂ ∧
    dy_dx t₁ = 0 ∧ dy_dx t₂ = 0 ∧
    x t₁ = 1 ∧ y t₁ = -15 ∧
    x t₂ = -3 ∧ y t₂ = 17 :=
sorry

-- Theorem for points where tangent is parallel to 9x + y + 3 = 0
theorem tangent_parallel_line :
  ∃ t₁ t₂ : ℝ,
    t₁ ≠ t₂ ∧
    dy_dx t₁ = m ∧ dy_dx t₂ = m ∧
    x t₁ = 0 ∧ y t₁ = -10 ∧
    x t₂ = -2 ∧ y t₂ = 12 :=
sorry

end tangent_parallel_x_axis_tangent_parallel_line_l3_372


namespace area_bounded_by_curves_l3_353

/-- The area between the parabola y = x^2 - x and the line y = mx from x = 0 to their intersection point. -/
def area_under_curve (m : ℤ) : ℚ :=
  (m + 1)^3 / 6

/-- The theorem statement -/
theorem area_bounded_by_curves (m n : ℤ) (h1 : m > n) (h2 : n > 0) :
  area_under_curve m - area_under_curve n = 37 / 6 → m = 3 ∧ n = 2 := by
  sorry

end area_bounded_by_curves_l3_353


namespace divisibility_conditions_l3_303

theorem divisibility_conditions :
  (∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1) → n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → Odd n → (n ∣ 3^n + 1) → n = 1) := by
  sorry

end divisibility_conditions_l3_303


namespace weeks_to_afford_laptop_l3_356

/-- The minimum number of whole weeks needed to afford a laptop -/
def weeks_needed (laptop_cost birthday_money weekly_earnings : ℕ) : ℕ :=
  (laptop_cost - birthday_money + weekly_earnings - 1) / weekly_earnings

/-- Proof that 34 weeks are needed to afford the laptop -/
theorem weeks_to_afford_laptop :
  weeks_needed 800 125 20 = 34 := by
  sorry

end weeks_to_afford_laptop_l3_356


namespace sin_30_degrees_l3_319

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end sin_30_degrees_l3_319


namespace problem_statement_l3_312

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : (3 : ℝ)^x = (4 : ℝ)^y) (h2 : 2 * x = a * y) : 
  a = 4 * (Real.log 2 / Real.log 3) := by
  sorry

end problem_statement_l3_312


namespace percentage_students_taking_music_l3_394

/-- The percentage of students taking music, given the total number of students
    and the number of students taking dance and art. -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (h1 : total_students = 400)
  (h2 : dance_students = 120)
  (h3 : art_students = 200) :
  (((total_students - dance_students - art_students) : ℚ) / total_students) * 100 = 20 := by
  sorry

end percentage_students_taking_music_l3_394


namespace patio_tile_count_l3_392

/-- Represents a square patio with red tiles along its diagonals -/
structure SquarePatio where
  side_length : ℕ
  red_tiles : ℕ

/-- The number of red tiles on a square patio with given side length -/
def red_tiles_count (s : ℕ) : ℕ := 2 * s - 1

/-- The total number of tiles on a square patio with given side length -/
def total_tiles_count (s : ℕ) : ℕ := s * s

/-- Theorem stating that if a square patio has 61 red tiles, it has 961 total tiles -/
theorem patio_tile_count (p : SquarePatio) (h : p.red_tiles = 61) :
  total_tiles_count p.side_length = 961 := by
  sorry

end patio_tile_count_l3_392


namespace exitCell_l3_318

/-- Represents a cell on the 4x4 grid --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the four possible directions --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The state of the game at any point --/
structure GameState :=
  (position : Cell)
  (arrows : Cell → Direction)

/-- Applies a single move to the game state --/
def move (state : GameState) : GameState :=
  sorry

/-- Checks if a cell is on the boundary of the grid --/
def isBoundary (cell : Cell) : Bool :=
  sorry

/-- Plays the game until the piece exits the grid --/
def playUntilExit (initialState : GameState) : Cell :=
  sorry

theorem exitCell :
  let initialArrows : Cell → Direction := sorry
  let initialState : GameState := {
    position := ⟨2, 1⟩,  -- C2 in 0-based indexing
    arrows := initialArrows
  }
  playUntilExit initialState = ⟨0, 1⟩  -- A2 in 0-based indexing
:= by sorry

end exitCell_l3_318


namespace total_chips_count_l3_360

def plain_chips : ℕ := 4
def bbq_chips : ℕ := 5
def probability_3_bbq : ℚ := 5/42

theorem total_chips_count : 
  let total_chips := plain_chips + bbq_chips
  (Nat.choose bbq_chips 3 : ℚ) / (Nat.choose total_chips 3 : ℚ) = probability_3_bbq →
  total_chips = 9 := by sorry

end total_chips_count_l3_360


namespace modulus_of_fraction_l3_347

def z : ℂ := -1 + Complex.I

theorem modulus_of_fraction : Complex.abs ((z + 3) / (z + 2)) = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_fraction_l3_347


namespace assignment_count_is_correct_l3_337

/-- The number of ways to assign 4 people to 3 offices with at least one person in each office -/
def assignmentCount : ℕ := 36

/-- The number of people to be assigned -/
def numPeople : ℕ := 4

/-- The number of offices -/
def numOffices : ℕ := 3

theorem assignment_count_is_correct :
  assignmentCount = (numPeople.choose 2) * numOffices * 2 :=
sorry

end assignment_count_is_correct_l3_337


namespace coordinates_of_N_l3_325

/-- Given a point M and a line segment MN parallel to the x-axis, 
    this function returns the possible coordinates of point N -/
def possible_coordinates_of_N (M : ℝ × ℝ) (length_MN : ℝ) : Set (ℝ × ℝ) :=
  let (x, y) := M
  { (x - length_MN, y), (x + length_MN, y) }

/-- Theorem stating that given M(2, -4) and MN of length 5 parallel to x-axis,
    N has coordinates either (-3, -4) or (7, -4) -/
theorem coordinates_of_N : 
  possible_coordinates_of_N (2, -4) 5 = {(-3, -4), (7, -4)} := by
  sorry


end coordinates_of_N_l3_325


namespace base_eight_31_equals_25_l3_334

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-eight number 31 is equal to the base-ten number 25 -/
theorem base_eight_31_equals_25 : base_eight_to_ten 3 1 = 25 := by
  sorry

end base_eight_31_equals_25_l3_334


namespace rectangle_ratio_l3_307

theorem rectangle_ratio (w : ℚ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 32 → w / 10 = 3 / 5 := by
  sorry

end rectangle_ratio_l3_307


namespace inverse_of_f_l3_309

noncomputable def f (x : ℝ) := Real.log x + 1

theorem inverse_of_f (x : ℝ) :
  x > 0 → f (Real.exp (x - 1)) = x ∧ Real.exp (f x - 1) = x := by
  sorry

end inverse_of_f_l3_309


namespace andrew_brought_40_chicken_nuggets_l3_373

/-- Represents the number of appetizer portions Andrew brought -/
def total_appetizers : ℕ := 90

/-- Represents the number of hotdogs on sticks Andrew brought -/
def hotdogs : ℕ := 30

/-- Represents the number of bite-sized cheese pops Andrew brought -/
def cheese_pops : ℕ := 20

/-- Represents the number of chicken nuggets Andrew brought -/
def chicken_nuggets : ℕ := total_appetizers - hotdogs - cheese_pops

/-- Theorem stating that Andrew brought 40 pieces of chicken nuggets -/
theorem andrew_brought_40_chicken_nuggets : chicken_nuggets = 40 := by
  sorry

end andrew_brought_40_chicken_nuggets_l3_373


namespace complex_number_in_fourth_quadrant_l3_362

/-- The complex number z = (2-i)/(1+i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l3_362


namespace race_outcomes_eq_210_l3_363

/-- The number of participants in the race -/
def num_participants : ℕ := 7

/-- The number of podium positions (1st, 2nd, 3rd) -/
def podium_positions : ℕ := 3

/-- Calculates the number of permutations of k elements from n elements -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else List.range k |>.foldl (fun acc i => acc * (n - i)) 1

/-- The number of different 1st-2nd-3rd place outcomes in a race with no ties -/
def race_outcomes : ℕ := permutations num_participants podium_positions

/-- Theorem: The number of different 1st-2nd-3rd place outcomes in a race
    with 7 participants and no ties is equal to 210 -/
theorem race_outcomes_eq_210 : race_outcomes = 210 := by
  sorry

end race_outcomes_eq_210_l3_363


namespace set_operations_l3_377

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -2 < x ∧ x < 3}

def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

theorem set_operations :
  (A ∩ B = {x | -2 < x ∧ x ≤ 2}) ∧
  ((Set.compl A) ∪ B = {x | x ≤ 2 ∨ x ≥ 3}) := by sorry

end set_operations_l3_377


namespace decagon_diagonals_l3_381

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  num_diagonals decagon_sides = 35 := by sorry

end decagon_diagonals_l3_381


namespace subtract_fractions_l3_391

theorem subtract_fractions : (3 : ℚ) / 4 - (1 : ℚ) / 6 = (7 : ℚ) / 12 := by sorry

end subtract_fractions_l3_391


namespace sqrt_point_three_six_equals_point_six_l3_304

theorem sqrt_point_three_six_equals_point_six : Real.sqrt 0.36 = 0.6 := by
  sorry

end sqrt_point_three_six_equals_point_six_l3_304


namespace algorithm_output_l3_338

def algorithm (n : ℕ) : ℤ :=
  let init := (0 : ℤ)
  init - 3 * n

theorem algorithm_output : algorithm 3 = -9 := by
  sorry

end algorithm_output_l3_338


namespace percentage_of_percentage_l3_390

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (30 / 100) * (60 / 100) * y = (18 / 100) * y :=
by sorry

end percentage_of_percentage_l3_390


namespace simplify_cube_root_exponent_sum_l3_389

theorem simplify_cube_root_exponent_sum (a b c : ℝ) : 
  ∃ (k : ℝ) (x y z : ℕ), 
    (∀ t : ℝ, t > 0 → (k * a^x * b^y * c^z)^3 * t = 40 * a^6 * b^9 * c^14) ∧ 
    x + y + z = 7 :=
sorry

end simplify_cube_root_exponent_sum_l3_389


namespace find_m_l3_335

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

theorem find_m : ∃ m : ℝ, (U \ A m = {1, 2}) → m = -3 := by
  sorry

end find_m_l3_335


namespace prism_faces_count_l3_368

/-- Represents a polygonal prism -/
structure Prism where
  base_sides : ℕ
  edges : ℕ := 3 * base_sides
  faces : ℕ := 2 + base_sides

/-- Represents a polygonal pyramid -/
structure Pyramid where
  base_sides : ℕ
  edges : ℕ := 2 * base_sides

/-- Theorem stating that a prism has 8 faces given the conditions -/
theorem prism_faces_count (p : Prism) (py : Pyramid) 
  (h1 : p.base_sides = py.base_sides) 
  (h2 : p.edges + py.edges = 30) : p.faces = 8 := by
  sorry

end prism_faces_count_l3_368


namespace chemists_sons_ages_l3_329

theorem chemists_sons_ages (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  a * b * c = 36 →  -- product is 36
  a + b + c = 13 →  -- sum is 13
  (a ≥ b ∧ a ≥ c) ∨ (b ≥ a ∧ b ≥ c) ∨ (c ≥ a ∧ c ≥ b) →  -- unique oldest son
  (a = 2 ∧ b = 2 ∧ c = 9) ∨ (a = 2 ∧ b = 9 ∧ c = 2) ∨ (a = 9 ∧ b = 2 ∧ c = 2) :=
by sorry

end chemists_sons_ages_l3_329


namespace no_right_triangle_with_integer_side_l3_359

theorem no_right_triangle_with_integer_side : 
  ¬ ∃ (x : ℤ), 
    (12 < x ∧ x < 30) ∧ 
    (x^2 = 12^2 + 30^2 ∨ 30^2 = 12^2 + x^2 ∨ 12^2 = 30^2 + x^2) :=
by sorry

#check no_right_triangle_with_integer_side

end no_right_triangle_with_integer_side_l3_359


namespace cards_distribution_l3_301

/-- Given 60 cards dealt to 7 people as evenly as possible, 
    exactly 3 people will have fewer than 9 cards. -/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 7) :
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end cards_distribution_l3_301


namespace scientific_notation_3080000_l3_339

theorem scientific_notation_3080000 :
  (3080000 : ℝ) = 3.08 * (10 ^ 6) :=
sorry

end scientific_notation_3080000_l3_339


namespace sams_friend_points_l3_342

theorem sams_friend_points (sam_points friend_points total_points : ℕ) :
  sam_points = 75 →
  total_points = 87 →
  total_points = sam_points + friend_points →
  friend_points = 12 := by
sorry

end sams_friend_points_l3_342


namespace cube_edge_length_l3_355

/-- Given a cube with surface area 216 cm², prove that the length of its edge is 6 cm. -/
theorem cube_edge_length (surface_area : ℝ) (edge_length : ℝ) 
  (h1 : surface_area = 216)
  (h2 : surface_area = 6 * edge_length^2) : 
  edge_length = 6 := by
  sorry

end cube_edge_length_l3_355


namespace ad_bc_ratio_l3_314

-- Define the triangle ABC
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define the triangle BCD
structure IsoscelesTriangle :=
  (side : ℝ)
  (angle : ℝ)
  (side_positive : side > 0)
  (angle_value : angle = 2 * Real.pi / 3)  -- 120° in radians

-- Define the configuration
structure TriangleConfiguration :=
  (abc : EquilateralTriangle)
  (bcd : IsoscelesTriangle)
  (shared_side : abc.side = bcd.side)

-- State the theorem
theorem ad_bc_ratio (config : TriangleConfiguration) :
  ∃ (ad bc : ℝ), ad / bc = 1 + Real.sqrt 3 :=
sorry

end ad_bc_ratio_l3_314


namespace square_sum_of_difference_and_product_l3_323

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a * b = 6) : 
  a^2 + b^2 = 48 := by
  sorry

end square_sum_of_difference_and_product_l3_323


namespace only_five_regular_polyhedra_five_platonic_solids_l3_352

/-- A regular polyhedron with n-gon faces and m faces meeting at each vertex -/
structure RegularPolyhedron where
  n : ℕ  -- number of sides of each face
  m : ℕ  -- number of faces meeting at each vertex
  n_ge_3 : n ≥ 3
  m_ge_3 : m ≥ 3

/-- The set of all possible (m, n) pairs for regular polyhedra -/
def valid_regular_polyhedra : Set (ℕ × ℕ) :=
  {(3, 3), (3, 4), (4, 3), (3, 5), (5, 3)}

/-- Theorem stating that only five regular polyhedra exist -/
theorem only_five_regular_polyhedra :
  ∀ p : RegularPolyhedron, (p.m, p.n) ∈ valid_regular_polyhedra := by
  sorry

/-- Corollary: There are exactly five types of regular polyhedra -/
theorem five_platonic_solids :
  ∃! (s : Set (ℕ × ℕ)), s = valid_regular_polyhedra ∧ (∀ p : RegularPolyhedron, (p.m, p.n) ∈ s) := by
  sorry

end only_five_regular_polyhedra_five_platonic_solids_l3_352


namespace simplify_sqrt_expression_l3_374

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 72) - (Real.sqrt 294 / Real.sqrt 98) = Real.sqrt 10 - Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l3_374


namespace last_two_digits_of_2006_factorial_l3_366

theorem last_two_digits_of_2006_factorial (n : ℕ) (h : n = 2006) : n! % 100 = 0 := by
  sorry

end last_two_digits_of_2006_factorial_l3_366


namespace circular_seating_pairs_l3_398

/-- The number of adjacent pairs in a circular seating arrangement --/
def adjacentPairs (n : ℕ) : ℕ := n

/-- Theorem: In a circular seating arrangement with n people,
    the number of different sets of two people sitting next to each other is n --/
theorem circular_seating_pairs (n : ℕ) (h : n > 0) :
  adjacentPairs n = n := by
  sorry

end circular_seating_pairs_l3_398


namespace greatest_common_divisor_630_90_under_35_l3_378

theorem greatest_common_divisor_630_90_under_35 : 
  ∀ n : ℕ, n ∣ 630 ∧ n < 35 ∧ n ∣ 90 → n ≤ 30 :=
by
  sorry

end greatest_common_divisor_630_90_under_35_l3_378


namespace unique_positive_integer_l3_354

theorem unique_positive_integer : ∃! (x : ℕ), x > 0 ∧ (4 * x)^2 - 2 * x = 2652 := by
  sorry

end unique_positive_integer_l3_354


namespace inequality_equivalence_l3_308

theorem inequality_equivalence (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by
  sorry

end inequality_equivalence_l3_308


namespace record_storage_cost_l3_370

-- Define the box dimensions
def box_length : ℝ := 15
def box_width : ℝ := 12
def box_height : ℝ := 10

-- Define the total occupied space in cubic inches
def total_space : ℝ := 1080000

-- Define the storage cost per box per month
def cost_per_box : ℝ := 0.5

-- Theorem to prove
theorem record_storage_cost :
  let box_volume : ℝ := box_length * box_width * box_height
  let num_boxes : ℝ := total_space / box_volume
  let total_cost : ℝ := num_boxes * cost_per_box
  total_cost = 300 := by
sorry


end record_storage_cost_l3_370


namespace salary_problem_l3_333

theorem salary_problem (a b : ℝ) 
  (h1 : a + b = 3000)
  (h2 : a * 0.05 = b * 0.15) : 
  a = 2250 := by
sorry

end salary_problem_l3_333


namespace successive_integers_product_l3_358

theorem successive_integers_product (n : ℤ) : 
  n * (n + 1) = 7832 → n = 88 := by
  sorry

end successive_integers_product_l3_358


namespace total_situps_is_110_l3_357

/-- The number of situps Diana did -/
def diana_situps : ℕ := 40

/-- The rate at which Diana did situps (situps per minute) -/
def diana_rate : ℕ := 4

/-- The difference in situps per minute between Hani and Diana -/
def hani_extra_rate : ℕ := 3

/-- Theorem stating that the total number of situps Hani and Diana did together is 110 -/
theorem total_situps_is_110 : 
  diana_situps + (diana_rate + hani_extra_rate) * (diana_situps / diana_rate) = 110 := by
  sorry

end total_situps_is_110_l3_357


namespace largest_negative_congruent_to_two_mod_seventeen_l3_340

theorem largest_negative_congruent_to_two_mod_seventeen :
  ∀ n : ℤ, -9999 ≤ n ∧ n < -999 ∧ n ≡ 2 [ZMOD 17] → n ≤ -1001 :=
by sorry

end largest_negative_congruent_to_two_mod_seventeen_l3_340


namespace range_of_m_l3_379

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (2*x + 5)/3 - 1 ≤ 2 - x → 3*(x - 1) + 5 > 5*x + 2*(m + x)) → 
  m < -3/5 := by
sorry

end range_of_m_l3_379


namespace min_value_a_l3_330

theorem min_value_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2004)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2004) :
  a ≥ 503 ∧ ∃ (a₀ b₀ c₀ d₀ : ℕ+), 
    a₀ = 503 ∧ 
    a₀ > b₀ ∧ b₀ > c₀ ∧ c₀ > d₀ ∧
    a₀ + b₀ + c₀ + d₀ = 2004 ∧
    a₀^2 - b₀^2 + c₀^2 - d₀^2 = 2004 :=
by sorry

end min_value_a_l3_330


namespace translation_result_l3_371

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a translation in 2D space
structure Translation2D where
  dx : ℝ
  dy : ℝ

-- Define a function to apply a translation to a point
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_result :
  let A : Point2D := { x := -3, y := 2 }
  let right_translation : Translation2D := { dx := 4, dy := 0 }
  let down_translation : Translation2D := { dx := 0, dy := -3 }
  let A' := applyTranslation (applyTranslation A right_translation) down_translation
  A'.x = 1 ∧ A'.y = -1 := by
sorry

end translation_result_l3_371


namespace shortest_distance_parabola_line_l3_376

/-- The shortest distance between a point on the parabola y = x^2 - 6x + 15 
    and a point on the line y = 2x - 7 -/
theorem shortest_distance_parabola_line : 
  let parabola := fun x : ℝ => x^2 - 6*x + 15
  let line := fun x : ℝ => 2*x - 7
  ∃ (min_dist : ℝ), 
    (∀ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) → 
      (q.2 = line q.1) → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ min_dist) ∧
    (∃ (p q : ℝ × ℝ), 
      (p.2 = parabola p.1) ∧ 
      (q.2 = line q.1) ∧ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = min_dist) ∧
    min_dist = 6 * Real.sqrt 5 / 5 := by
  sorry


end shortest_distance_parabola_line_l3_376


namespace larger_group_size_l3_351

/-- Given that 36 men can complete a piece of work in 18 days, and a larger group
    of men can complete the same work in 6 days, prove that the larger group
    consists of 108 men. -/
theorem larger_group_size (work : ℕ) (small_group : ℕ) (large_group : ℕ)
    (small_days : ℕ) (large_days : ℕ)
    (h1 : small_group = 36)
    (h2 : small_days = 18)
    (h3 : large_days = 6)
    (h4 : small_group * small_days = work)
    (h5 : large_group * large_days = work) :
    large_group = 108 := by
  sorry

#check larger_group_size

end larger_group_size_l3_351


namespace evaluate_expression_l3_300

theorem evaluate_expression (x y z : ℝ) (hx : x = 5) (hy : y = 10) (hz : z = 3) :
  z * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l3_300


namespace solution_triples_l3_326

theorem solution_triples (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + b + c = 1/a + 1/b + 1/c) ∧ (a^2 + b^2 + c^2 = 1/a^2 + 1/b^2 + 1/c^2) →
  ((a = 1 ∧ c = 1/b) ∨ (b = 1/a ∧ c = 1) ∨ (b = 1 ∧ c = 1/a) ∨
   (a = -1 ∧ c = 1/b) ∨ (b = -1 ∧ c = 1/a) ∨ (b = 1/a ∧ c = -1)) :=
by sorry

end solution_triples_l3_326


namespace mango_count_proof_l3_345

/-- Calculates the total number of mangoes in multiple boxes -/
def total_mangoes (mangoes_per_dozen : ℕ) (dozens_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  mangoes_per_dozen * dozens_per_box * num_boxes

/-- Proves that 36 boxes of 10 dozen mangoes each contain 4,320 mangoes in total -/
theorem mango_count_proof : total_mangoes 12 10 36 = 4320 := by
  sorry

#eval total_mangoes 12 10 36

end mango_count_proof_l3_345


namespace hyperbola_equation_l3_315

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := (2 * Real.sqrt 3) / 3
  let line_distance := Real.sqrt 3 / 2
  let eccentricity_eq := e^2 = 1 + b^2 / a^2
  let distance_eq := (a * b)^2 / (a^2 + b^2) = line_distance^2
  eccentricity_eq ∧ distance_eq →
  a^2 = 3 ∧ b^2 = 1 :=
by sorry


end hyperbola_equation_l3_315


namespace function_properties_l3_310

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * sin (ω * x) + cos (ω * x + π / 3) + cos (ω * x - π / 3) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * sin (2 * x - π / 6) - 1

theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∀ x, f ω x = 2 * sin (2 * x + π / 6) - 1) ∧
  (∀ x, g x = 2 * sin (2 * x - π / 6) - 1) ∧
  (Set.Icc 0 (π / 2)).image g = Set.Icc (-2) 1 :=
by sorry

end function_properties_l3_310


namespace rectangular_garden_width_l3_387

theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 675 →
  width = 15 := by
sorry

end rectangular_garden_width_l3_387


namespace markeesha_cracker_sales_l3_317

theorem markeesha_cracker_sales :
  ∀ (friday saturday sunday : ℕ),
    friday = 30 →
    saturday = 2 * friday →
    friday + saturday + sunday = 135 →
    saturday - sunday = 15 :=
by
  sorry

end markeesha_cracker_sales_l3_317


namespace initial_oak_trees_l3_343

theorem initial_oak_trees (initial : ℕ) (planted : ℕ) (total : ℕ) : 
  planted = 2 → total = 11 → initial + planted = total → initial = 9 := by
  sorry

end initial_oak_trees_l3_343


namespace f_2002_equals_96_l3_344

/-- A function satisfying the given property -/
def special_function (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a + b = 2^n → f a + f b = n^2

/-- The theorem to be proved -/
theorem f_2002_equals_96 (f : ℕ → ℝ) (h : special_function f) : f 2002 = 96 := by
  sorry

end f_2002_equals_96_l3_344


namespace star_three_five_l3_306

def star (a b : ℕ) : ℕ := (a + b) ^ 3

theorem star_three_five : star 3 5 = 512 := by
  sorry

end star_three_five_l3_306
