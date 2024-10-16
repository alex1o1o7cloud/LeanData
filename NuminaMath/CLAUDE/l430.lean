import Mathlib

namespace NUMINAMATH_CALUDE_ben_baseball_cards_l430_43029

/-- The number of baseball cards in each box given to Ben by his mother -/
def baseball_cards_per_box : ℕ := sorry

theorem ben_baseball_cards :
  let basketball_boxes : ℕ := 4
  let basketball_cards_per_box : ℕ := 10
  let baseball_boxes : ℕ := 5
  let cards_given_away : ℕ := 58
  let cards_remaining : ℕ := 22
  
  basketball_boxes * basketball_cards_per_box + 
  baseball_boxes * baseball_cards_per_box = 
  cards_given_away + cards_remaining →
  
  baseball_cards_per_box = 8 := by sorry

end NUMINAMATH_CALUDE_ben_baseball_cards_l430_43029


namespace NUMINAMATH_CALUDE_jose_profit_share_l430_43028

/-- Calculates the share of profit for an investor given their investment amount, duration, and the total profit and investment-months. -/
def shareOfProfit (investment : ℕ) (duration : ℕ) (totalProfit : ℕ) (totalInvestmentMonths : ℕ) : ℕ :=
  (investment * duration * totalProfit) / totalInvestmentMonths

theorem jose_profit_share (tomInvestment jose_investment : ℕ) (tomDuration joseDuration : ℕ) (totalProfit : ℕ)
    (h1 : tomInvestment = 30000)
    (h2 : jose_investment = 45000)
    (h3 : tomDuration = 12)
    (h4 : joseDuration = 10)
    (h5 : totalProfit = 72000) :
  shareOfProfit jose_investment joseDuration totalProfit (tomInvestment * tomDuration + jose_investment * joseDuration) = 40000 := by
  sorry

#eval shareOfProfit 45000 10 72000 (30000 * 12 + 45000 * 10)

end NUMINAMATH_CALUDE_jose_profit_share_l430_43028


namespace NUMINAMATH_CALUDE_rectangle_side_length_l430_43050

theorem rectangle_side_length (a b c d : ℝ) : 
  a / c = 3 / 4 → 
  b / d = 3 / 4 → 
  c = 4 → 
  d = 8 → 
  b = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l430_43050


namespace NUMINAMATH_CALUDE_system_solution_l430_43002

theorem system_solution (a b c : ℝ) : 
  (a^3 + 3*a*b^2 + 3*a*c^2 - 6*a*b*c = 1) ∧ 
  (b^3 + 3*b*a^2 + 3*b*c^2 - 6*a*b*c = 1) ∧ 
  (c^3 + 3*c*a^2 + 3*c*b^2 - 6*a*b*c = 1) → 
  (a = 1 ∧ b = 1 ∧ c = 1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l430_43002


namespace NUMINAMATH_CALUDE_power_inequality_l430_43035

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ a^b * b^a :=
by sorry

end NUMINAMATH_CALUDE_power_inequality_l430_43035


namespace NUMINAMATH_CALUDE_equation_solution_l430_43056

theorem equation_solution : ∃ x : ℚ, (5 * x + 9 * x = 570 - 12 * (x - 5)) ∧ (x = 315 / 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l430_43056


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l430_43024

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, x^2 - x < 0 → -1 < x ∧ x < 1) ∧
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ ¬(x^2 - x < 0)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l430_43024


namespace NUMINAMATH_CALUDE_polynomial_factorization_l430_43079

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - 35 = (x - 5) * (x + 7)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l430_43079


namespace NUMINAMATH_CALUDE_smallest_integer_of_three_l430_43005

theorem smallest_integer_of_three (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 100 →
  2 * b = 3 * a →
  2 * c = 5 * a →
  a = 20 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_of_three_l430_43005


namespace NUMINAMATH_CALUDE_exponent_division_simplification_l430_43027

theorem exponent_division_simplification (a b : ℝ) :
  (-a * b)^5 / (-a * b)^3 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_simplification_l430_43027


namespace NUMINAMATH_CALUDE_equation_solution_l430_43080

theorem equation_solution (y : ℝ) : 
  (y^2 - 11*y + 24)/(y - 1) + (4*y^2 + 20*y - 25)/(4*y - 5) = 5 → y = 3 ∨ y = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l430_43080


namespace NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l430_43009

/-- The number of real values of b for which the line y = 2x + b intersects
    the parabola y = x^2 - 4x + b^2 at its vertex -/
theorem line_intersects_parabola_vertex : 
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
  ∀ b ∈ s, ∃ x y : ℝ, 
    (y = 2 * x + b) ∧ 
    (y = x^2 - 4 * x + b^2) ∧
    (∀ x' y' : ℝ, (y' = x'^2 - 4 * x' + b^2) → y' ≤ y) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_parabola_vertex_l430_43009


namespace NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l430_43064

theorem triangle_side_ratio_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_geom_seq : b^2 = a*c) :
  2 ≤ (b/a + a/b) ∧ (b/a + a/b) < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_bounds_l430_43064


namespace NUMINAMATH_CALUDE_fewest_printers_equal_spend_l430_43052

/-- The cost of the first type of printer -/
def cost1 : ℕ := 400

/-- The cost of the second type of printer -/
def cost2 : ℕ := 350

/-- The function to calculate the total number of printers purchased -/
def total_printers (n1 n2 : ℕ) : ℕ := n1 + n2

/-- The function to calculate the total cost for each type of printer -/
def total_cost (cost quantity : ℕ) : ℕ := cost * quantity

/-- The theorem stating the fewest number of printers that can be purchased
    while spending equal amounts on both types -/
theorem fewest_printers_equal_spend :
  ∃ (n1 n2 : ℕ),
    total_cost cost1 n1 = total_cost cost2 n2 ∧
    ∀ (m1 m2 : ℕ),
      total_cost cost1 m1 = total_cost cost2 m2 →
      total_printers n1 n2 ≤ total_printers m1 m2 ∧
      total_printers n1 n2 = 15 :=
sorry

end NUMINAMATH_CALUDE_fewest_printers_equal_spend_l430_43052


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l430_43007

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = Real.sqrt 194 ∧
    (∀ y : ℝ, y < x → ⌊y^2⌋ - ⌊y⌋^2 ≠ 25) ∧
    ⌊x^2⌋ - ⌊x⌋^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l430_43007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l430_43077

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + 2 * a 6 + a 10 = 120) →
  (a 3 + a 9 = 60) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l430_43077


namespace NUMINAMATH_CALUDE_percentage_difference_theorem_l430_43072

theorem percentage_difference_theorem (x : ℝ) : 
  (0.35 * x = 0.50 * x - 24) → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_theorem_l430_43072


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l430_43049

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for the given cost and selling prices is 17% -/
theorem radio_loss_percentage : 
  loss_percentage 1500 1245 = 17 := by sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l430_43049


namespace NUMINAMATH_CALUDE_weighted_average_fish_per_day_l430_43000

-- Define the daily catch for each person
def aang_catch : List Nat := [5, 7, 9]
def sokka_catch : List Nat := [8, 5, 6]
def toph_catch : List Nat := [10, 12, 8]
def zuko_catch : List Nat := [6, 7, 10]

-- Define the number of people and days
def num_people : Nat := 4
def num_days : Nat := 3

-- Define the total fish caught by the group
def total_fish : Nat := aang_catch.sum + sokka_catch.sum + toph_catch.sum + zuko_catch.sum

-- Define the total days fished by the group
def total_days : Nat := num_people * num_days

-- Theorem to prove
theorem weighted_average_fish_per_day :
  (total_fish : Rat) / total_days = 93/12 := by sorry

end NUMINAMATH_CALUDE_weighted_average_fish_per_day_l430_43000


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l430_43068

/-- Two planes are mutually perpendicular -/
def mutually_perpendicular (α β : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (n : Line) (β : Plane) : Prop := sorry

/-- Two planes intersect at a line -/
def planes_intersect_at (α β : Plane) (l : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perp_line (n l : Line) : Prop := sorry

/-- Main theorem -/
theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (l m n : Line) 
  (h1 : mutually_perpendicular α β)
  (h2 : planes_intersect_at α β l)
  (h3 : line_parallel_plane m α)
  (h4 : line_perp_plane n β) :
  line_perp_line n l := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l430_43068


namespace NUMINAMATH_CALUDE_system_solution_l430_43073

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = 1) ∧ (y + z = 2) ∧ (z + x = 3) ∧ (x = 1) ∧ (y = 0) ∧ (z = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l430_43073


namespace NUMINAMATH_CALUDE_base_nine_representation_l430_43066

theorem base_nine_representation (b : ℕ) : 
  (777 : ℕ) = 1 * b^3 + 0 * b^2 + 5 * b^1 + 3 * b^0 ∧ 
  b > 1 ∧ 
  b^3 ≤ 777 ∧ 
  777 < b^4 ∧
  (∃ (A C : ℕ), A ≠ C ∧ A < b ∧ C < b ∧ 
    777 = A * b^3 + C * b^2 + A * b^1 + C * b^0) →
  b = 9 := by
sorry

end NUMINAMATH_CALUDE_base_nine_representation_l430_43066


namespace NUMINAMATH_CALUDE_smallest_multiple_of_11_23_37_l430_43055

theorem smallest_multiple_of_11_23_37 : ∃ (n : ℕ), n > 0 ∧ 11 ∣ n ∧ 23 ∣ n ∧ 37 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 11 ∣ m ∧ 23 ∣ m ∧ 37 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_11_23_37_l430_43055


namespace NUMINAMATH_CALUDE_combinations_equal_twenty_l430_43089

/-- The number of paint colors available. -/
def num_colors : ℕ := 5

/-- The number of painting methods available. -/
def num_methods : ℕ := 4

/-- The total number of combinations of paint colors and painting methods. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 20. -/
theorem combinations_equal_twenty : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twenty_l430_43089


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l430_43038

/-- A 4x4 grid of points separated by unit distances -/
def Grid := Fin 5 × Fin 5

/-- A rectangle on the grid is defined by two vertical lines and two horizontal lines -/
def Rectangle := (Fin 5 × Fin 5) × (Fin 5 × Fin 5)

/-- The number of rectangles on a 4x4 grid -/
def num_rectangles : ℕ := sorry

theorem rectangles_on_4x4_grid : num_rectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l430_43038


namespace NUMINAMATH_CALUDE_quadI_area_less_than_quadII_area_l430_43096

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

/-- Quadrilateral I with vertices (0,0), (2,0), (2,2), and (0,1) -/
def quadI : List (ℝ × ℝ) := [(0,0), (2,0), (2,2), (0,1)]

/-- Quadrilateral II with vertices (0,0), (3,0), (3,1), and (0,2) -/
def quadII : List (ℝ × ℝ) := [(0,0), (3,0), (3,1), (0,2)]

theorem quadI_area_less_than_quadII_area :
  quadrilateralArea quadI.head! quadI.tail!.head! quadI.tail!.tail!.head! quadI.tail!.tail!.tail!.head! <
  quadrilateralArea quadII.head! quadII.tail!.head! quadII.tail!.tail!.head! quadII.tail!.tail!.tail!.head! :=
by sorry

end NUMINAMATH_CALUDE_quadI_area_less_than_quadII_area_l430_43096


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l430_43039

/-- Given an arithmetic sequence {a_n} where a_3 + a_4 + a_5 = 12, 
    the sum of the first seven terms is 28. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 3 + a 4 + a 5 = 12 →                    -- given condition
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l430_43039


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l430_43013

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) * Real.sqrt (36 - y^2) = 12) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l430_43013


namespace NUMINAMATH_CALUDE_calculation_result_l430_43031

theorem calculation_result : (0.0077 : ℝ) * 4.5 / (0.05 * 0.1 * 0.007) = 990 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l430_43031


namespace NUMINAMATH_CALUDE_fruit_drink_total_volume_l430_43019

/-- A fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_ounces : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_total_volume (drink : FruitDrink) 
  (h1 : drink.orange_percent = 0.15)
  (h2 : drink.watermelon_percent = 0.60)
  (h3 : drink.grape_ounces = 30) :
  (drink.grape_ounces / (1 - drink.orange_percent - drink.watermelon_percent)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_fruit_drink_total_volume_l430_43019


namespace NUMINAMATH_CALUDE_grid_black_probability_l430_43003

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The probability of a single cell being black initially -/
def initial_black_prob : ℚ := 1/2

/-- Rotates the grid 90 degrees clockwise -/
def rotate (g : Grid) : Grid := sorry

/-- Applies the repainting rule after rotation -/
def repaint (g : Grid) : Grid := sorry

/-- The probability that the entire grid becomes black after rotation and repainting -/
def prob_all_black_after_process : ℚ := sorry

/-- Theorem stating the probability of the grid becoming entirely black -/
theorem grid_black_probability : 
  prob_all_black_after_process = 1 / 65536 := by sorry

end NUMINAMATH_CALUDE_grid_black_probability_l430_43003


namespace NUMINAMATH_CALUDE_exam_result_proof_l430_43008

/-- Represents the result of an examination --/
structure ExamResult where
  total_questions : ℕ
  correct_score : ℤ
  wrong_score : ℤ
  unanswered_score : ℤ
  total_score : ℤ
  correct_answers : ℕ
  wrong_answers : ℕ
  unanswered : ℕ

/-- Theorem stating the correct number of answers for the given exam conditions --/
theorem exam_result_proof (exam : ExamResult) : 
  exam.total_questions = 75 ∧ 
  exam.correct_score = 5 ∧ 
  exam.wrong_score = -2 ∧ 
  exam.unanswered_score = -1 ∧ 
  exam.total_score = 215 ∧
  exam.correct_answers + exam.wrong_answers + exam.unanswered = exam.total_questions ∧
  exam.correct_score * exam.correct_answers + exam.wrong_score * exam.wrong_answers + exam.unanswered_score * exam.unanswered = exam.total_score →
  exam.correct_answers = 52 ∧ exam.wrong_answers = 23 ∧ exam.unanswered = 0 := by
  sorry

end NUMINAMATH_CALUDE_exam_result_proof_l430_43008


namespace NUMINAMATH_CALUDE_select_and_assign_volunteers_eq_30_l430_43044

/-- The number of ways to select and assign volunteers for a two-day event -/
def select_and_assign_volunteers : ℕ :=
  let total_volunteers : ℕ := 5
  let selected_volunteers : ℕ := 4
  let days : ℕ := 2
  let volunteers_per_day : ℕ := 2
  (total_volunteers.choose selected_volunteers) *
  ((selected_volunteers.choose volunteers_per_day) * (days.factorial))

/-- Theorem stating that the number of ways to select and assign volunteers is 30 -/
theorem select_and_assign_volunteers_eq_30 :
  select_and_assign_volunteers = 30 := by
  sorry

end NUMINAMATH_CALUDE_select_and_assign_volunteers_eq_30_l430_43044


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l430_43069

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 135 → 
  (n - 2) * 180 = n * interior_angle → 
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l430_43069


namespace NUMINAMATH_CALUDE_leifeng_pagoda_height_l430_43006

/-- The height of the Leifeng Pagoda problem -/
theorem leifeng_pagoda_height 
  (AC : ℝ) 
  (α β : ℝ) 
  (h1 : AC = 62 * Real.sqrt 2)
  (h2 : α = 45 * π / 180)
  (h3 : β = 15 * π / 180) :
  ∃ BC : ℝ, BC = 62 :=
sorry

end NUMINAMATH_CALUDE_leifeng_pagoda_height_l430_43006


namespace NUMINAMATH_CALUDE_f_composition_of_i_l430_43065

noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then 2 * z^2 + 1 else -z^2 - 1

theorem f_composition_of_i : f (f (f (f Complex.I))) = -26 := by sorry

end NUMINAMATH_CALUDE_f_composition_of_i_l430_43065


namespace NUMINAMATH_CALUDE_average_equation_solution_l430_43094

theorem average_equation_solution (x : ℝ) : 
  ((2*x + 12) + (3*x + 3) + (5*x - 8)) / 3 = 3*x + 2 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_solution_l430_43094


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l430_43012

theorem simplify_and_evaluate (m : ℝ) (h : m = -2) :
  m / (m^2 - 9) / (1 + 3 / (m - 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l430_43012


namespace NUMINAMATH_CALUDE_sara_jim_savings_equality_l430_43086

def sara_weekly_savings (S : ℚ) : ℚ := S

theorem sara_jim_savings_equality (S : ℚ) : 
  (4100 : ℚ) + 820 * sara_weekly_savings S = 15 * 820 → S = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_jim_savings_equality_l430_43086


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l430_43090

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A tetrahedron defined by four points -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Check if a point is inside a triangle -/
def isInside (p : Point3D) (t : Tetrahedron) : Prop :=
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = Point3D.mk (a * t.A.x + b * t.B.x + c * t.C.x)
                   (a * t.A.y + b * t.B.y + c * t.C.y)
                   (a * t.A.z + b * t.B.z + c * t.C.z)

/-- Calculate the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Find the intersection point of a line parallel to DD₁ passing through a vertex -/
noncomputable def intersectionPoint (t : Tetrahedron) (D₁ : Point3D) (vertex : Point3D) : Point3D := sorry

/-- The main theorem -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) (D₁ : Point3D) :
  isInside D₁ t →
  let A₁ := intersectionPoint t D₁ t.A
  let B₁ := intersectionPoint t D₁ t.B
  let C₁ := intersectionPoint t D₁ t.C
  let t₁ := Tetrahedron.mk A₁ B₁ C₁ D₁
  volume t = (1/3) * volume t₁ := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l430_43090


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l430_43061

structure IsoscelesTrapezoid where
  smaller_base : ℝ
  larger_base : ℝ
  diagonal : ℝ
  altitude : ℝ
  is_isosceles : True
  smaller_base_half_diagonal : smaller_base = diagonal / 2
  altitude_half_larger_base : altitude = larger_base / 2

theorem isosceles_trapezoid_base_ratio 
  (t : IsoscelesTrapezoid) : t.smaller_base / t.larger_base = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_base_ratio_l430_43061


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l430_43022

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), x * 10^(2*n + 2) - (x * 10^(2*n + 2)).floor = 0.36) ∧ x = 4/11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l430_43022


namespace NUMINAMATH_CALUDE_additional_rotations_needed_l430_43051

/-- Calculates the number of additional wheel rotations needed to reach a goal distance --/
theorem additional_rotations_needed
  (rotations_per_block : ℕ)
  (goal_blocks : ℕ)
  (current_rotations : ℕ)
  (h1 : rotations_per_block = 200)
  (h2 : goal_blocks = 8)
  (h3 : current_rotations = 600) :
  rotations_per_block * goal_blocks - current_rotations = 1000 :=
by sorry

end NUMINAMATH_CALUDE_additional_rotations_needed_l430_43051


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_three_answer_is_199999_l430_43070

theorem largest_n_divisible_by_three (n : ℕ) : 
  n < 200000 → 
  (3 ∣ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36)) → 
  n ≤ 199999 :=
by sorry

theorem answer_is_199999 : 
  199999 < 200000 ∧ 
  (3 ∣ (10 * (199999 - 3)^5 - 2 * 199999^2 + 20 * 199999 - 36)) ∧
  ∀ m : ℕ, m > 199999 → m < 200000 → 
    ¬(3 ∣ (10 * (m - 3)^5 - 2 * m^2 + 20 * m - 36)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_three_answer_is_199999_l430_43070


namespace NUMINAMATH_CALUDE_count_numbers_with_5_or_7_up_to_700_l430_43084

def count_numbers_with_5_or_7 (n : ℕ) : ℕ :=
  n - (
    -- Three-digit numbers without 5 or 7
    6 * 8 * 8 +
    -- Two-digit numbers without 5 or 7
    8 * 8 +
    -- One-digit numbers without 5 or 7
    7 +
    -- Special case: 700
    1
  )

theorem count_numbers_with_5_or_7_up_to_700 :
  count_numbers_with_5_or_7 700 = 244 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_5_or_7_up_to_700_l430_43084


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l430_43020

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℤ) (h_d : d ≠ 0) :
  (∃ r : ℚ, (arithmetic_sequence a₁ d 2 + 1) ^ 2 = (arithmetic_sequence a₁ d 1 + 1) * (arithmetic_sequence a₁ d 4 + 1)) →
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = -12 →
  ∀ n : ℕ, arithmetic_sequence a₁ d n = -2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l430_43020


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l430_43046

/-- A parabola with equation y = dx^2 + ex + f, vertex (-3, 2), and passing through (-5, 10) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : 2 = d * (-3)^2 + e * (-3) + f
  point_condition : 10 = d * (-5)^2 + e * (-5) + f

/-- The sum of coefficients d, e, and f equals 10 -/
theorem parabola_coefficient_sum (p : Parabola) : p.d + p.e + p.f = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l430_43046


namespace NUMINAMATH_CALUDE_triangle_problem_l430_43057

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  (2 * c - b) / (Real.sqrt 3 * Real.sin C - Real.cos C) = a →
  -- b = 1
  b = 1 →
  -- Area condition
  (1 / 2) * b * c * Real.sin A = (3 / 4) * Real.tan A →
  -- Prove A = π/3 and a = √7
  A = π / 3 ∧ a = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l430_43057


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_fraction_sum_l430_43071

/-- Two prime numbers that are roots of a quadratic equation --/
def QuadraticPrimeRoots (a b : ℕ) : Prop :=
  Prime a ∧ Prime b ∧ ∃ t : ℤ, a^2 - 21*a + t = 0 ∧ b^2 - 21*b + t = 0

/-- The main theorem --/
theorem quadratic_prime_roots_fraction_sum
  (a b : ℕ) (h : QuadraticPrimeRoots a b) :
  (b : ℚ) / a + (a : ℚ) / b = 365 / 38 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_fraction_sum_l430_43071


namespace NUMINAMATH_CALUDE_baseball_card_packs_l430_43092

/-- The number of packs of baseball cards for a group of people -/
def total_packs (num_people : ℕ) (cards_per_person : ℕ) (cards_per_pack : ℕ) : ℕ :=
  num_people * (cards_per_person / cards_per_pack)

/-- Theorem: Four people buying 540 cards each, with 20 cards per pack, have 108 packs in total -/
theorem baseball_card_packs :
  total_packs 4 540 20 = 108 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_packs_l430_43092


namespace NUMINAMATH_CALUDE_function_property_l430_43014

theorem function_property (f : ℝ → ℝ) (h : ¬(∀ x > 0, f x > 0)) : ∃ x > 0, f x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l430_43014


namespace NUMINAMATH_CALUDE_single_point_conic_section_l430_43078

/-- If the graph of 3x^2 + y^2 + 6x - 6y + d = 0 consists of a single point, then d = 12 -/
theorem single_point_conic_section (d : ℝ) :
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 6 * p.2 + d = 0) →
  d = 12 := by
  sorry

end NUMINAMATH_CALUDE_single_point_conic_section_l430_43078


namespace NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l430_43023

theorem gold_copper_alloy_ratio 
  (G : ℝ) 
  (h_G : G > 9) : 
  let x := 9 / (G - 9)
  x * G + (1 - x) * 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gold_copper_alloy_ratio_l430_43023


namespace NUMINAMATH_CALUDE_sum_of_squares_square_of_sum_sum_of_three_squares_sum_of_fourth_powers_l430_43045

-- Part 1
theorem sum_of_squares (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) :
  a^2 + b^2 = 5 := by sorry

-- Part 2
theorem square_of_sum (a b c : ℝ) :
  (a + b + c)^2 = a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c := by sorry

-- Part 3
theorem sum_of_three_squares (a b c : ℝ) (h1 : a + b + c = 6) (h2 : a*b + b*c + a*c = 11) :
  a^2 + b^2 + c^2 = 14 := by sorry

-- Part 4
theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  a^4 + b^4 + c^4 = 18 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_square_of_sum_sum_of_three_squares_sum_of_fourth_powers_l430_43045


namespace NUMINAMATH_CALUDE_total_marbles_is_72_l430_43076

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of yellow:blue:green marbles -/
def marbleRatio : MarbleCounts := ⟨2, 3, 4⟩

/-- The actual number of green marbles in the bag -/
def greenMarbleCount : ℕ := 32

/-- Calculate the total number of marbles in the bag -/
def totalMarbles (mc : MarbleCounts) : ℕ :=
  mc.yellow + mc.blue + mc.green

/-- Theorem stating that the total number of marbles is 72 -/
theorem total_marbles_is_72 :
  ∃ (factor : ℕ), 
    factor * marbleRatio.green = greenMarbleCount ∧
    totalMarbles (MarbleCounts.mk 
      (factor * marbleRatio.yellow)
      (factor * marbleRatio.blue)
      greenMarbleCount) = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_72_l430_43076


namespace NUMINAMATH_CALUDE_price_increase_percentage_l430_43091

theorem price_increase_percentage (initial_price : ℝ) : 
  initial_price > 0 →
  let new_egg_price := initial_price * 1.1
  let new_apple_price := initial_price * 1.02
  let initial_total := initial_price * 2
  let new_total := new_egg_price + new_apple_price
  (new_total - initial_total) / initial_total = 0.04 :=
by
  sorry

#check price_increase_percentage

end NUMINAMATH_CALUDE_price_increase_percentage_l430_43091


namespace NUMINAMATH_CALUDE_simplify_expression_l430_43083

theorem simplify_expression (x : ℝ) : (3 * x + 30) + (150 * x - 45) = 153 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l430_43083


namespace NUMINAMATH_CALUDE_floor_plus_half_l430_43043

theorem floor_plus_half (x : ℝ) : 
  ⌊x + 0.5⌋ = ⌊x⌋ ∨ ⌊x + 0.5⌋ = ⌊x⌋ + 1 := by sorry

end NUMINAMATH_CALUDE_floor_plus_half_l430_43043


namespace NUMINAMATH_CALUDE_square_sum_equals_25_l430_43074

theorem square_sum_equals_25 (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) :
  x^2 + y^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_25_l430_43074


namespace NUMINAMATH_CALUDE_racecar_repair_discount_l430_43088

/-- Calculates the discount percentage on a racecar repair --/
theorem racecar_repair_discount (original_cost prize keep_percentage profit : ℝ) :
  original_cost = 20000 →
  prize = 70000 →
  keep_percentage = 0.9 →
  profit = 47000 →
  (original_cost - (keep_percentage * prize - profit)) / original_cost = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_racecar_repair_discount_l430_43088


namespace NUMINAMATH_CALUDE_odd_function_positive_range_l430_43015

open Set

def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_positive_range
  (f : ℝ → ℝ)
  (hf_odd : isOdd f)
  (hf_neg_one : f (-1) = 0)
  (hf_deriv : ∀ x > 0, x * (deriv^[2] f x) - deriv f x > 0) :
  {x : ℝ | f x > 0} = Ioo (-1) 0 ∪ Ioi 1 := by sorry

end NUMINAMATH_CALUDE_odd_function_positive_range_l430_43015


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l430_43010

theorem sum_of_squares_theorem (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 2)
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l430_43010


namespace NUMINAMATH_CALUDE_tournament_rounds_l430_43081

/-- Represents a table tennis tournament with the given rules --/
structure TableTennisTournament where
  players : ℕ
  champion_losses : ℕ

/-- Calculates the number of rounds in the tournament --/
def rounds (t : TableTennisTournament) : ℕ :=
  2 * (t.players - 1) + t.champion_losses

/-- Theorem stating that a tournament with 15 players and a champion who lost once has 29 rounds --/
theorem tournament_rounds :
  ∀ t : TableTennisTournament,
    t.players = 15 →
    t.champion_losses = 1 →
    rounds t = 29 :=
by
  sorry

#check tournament_rounds

end NUMINAMATH_CALUDE_tournament_rounds_l430_43081


namespace NUMINAMATH_CALUDE_franks_money_l430_43018

theorem franks_money (initial_money : ℚ) : 
  (3/4 : ℚ) * ((4/5 : ℚ) * initial_money) = 360 → initial_money = 600 := by
  sorry

end NUMINAMATH_CALUDE_franks_money_l430_43018


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l430_43021

-- Define the sets A and B
def A : Set ℝ := {x | x < 3/2}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l430_43021


namespace NUMINAMATH_CALUDE_ellen_yogurt_amount_l430_43087

/-- The amount of yogurt used in Ellen's smoothie -/
def yogurt_amount (strawberries orange_juice total : ℝ) : ℝ :=
  total - (strawberries + orange_juice)

/-- Theorem: Ellen used 0.1 cup of yogurt in her smoothie -/
theorem ellen_yogurt_amount :
  yogurt_amount 0.2 0.2 0.5 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ellen_yogurt_amount_l430_43087


namespace NUMINAMATH_CALUDE_cake_fraction_eaten_l430_43053

/-- Proves that the fraction of cake eaten by visitors is 1/4 given the conditions -/
theorem cake_fraction_eaten (total_slices : ℕ) (kept_slices : ℕ) 
  (h1 : total_slices = 12) 
  (h2 : kept_slices = 9) : 
  (total_slices - kept_slices : ℚ) / total_slices = 1/4 := by
  sorry

#check cake_fraction_eaten

end NUMINAMATH_CALUDE_cake_fraction_eaten_l430_43053


namespace NUMINAMATH_CALUDE_ellipse_and_triangle_properties_l430_43026

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about the ellipse and triangle properties -/
theorem ellipse_and_triangle_properties
  (e : Ellipse)
  (focus : Point)
  (pass_through : Point)
  (p : Point)
  (h_focus : focus = ⟨2 * Real.sqrt 2, 0⟩)
  (h_pass : pass_through = ⟨3, 1⟩)
  (h_p : p = ⟨-3, 2⟩)
  (h_on_ellipse : pass_through.x^2 / e.a^2 + pass_through.y^2 / e.b^2 = 1)
  (h_focus_prop : e.a^2 - e.b^2 = 8)
  (h_intersect : ∃ (a b : Point), a ≠ b ∧
    a.x^2 / e.a^2 + a.y^2 / e.b^2 = 1 ∧
    b.x^2 / e.a^2 + b.y^2 / e.b^2 = 1 ∧
    a.y - b.y = a.x - b.x)
  (h_isosceles : ∃ (a b : Point), 
    (a.x - p.x)^2 + (a.y - p.y)^2 = (b.x - p.x)^2 + (b.y - p.y)^2) :
  e.a^2 = 12 ∧ e.b^2 = 4 ∧
  (∃ (a b : Point), 
    (1/2) * Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2) * 
    (3 / Real.sqrt 2) = 9/2) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_triangle_properties_l430_43026


namespace NUMINAMATH_CALUDE_equation_solutions_l430_43059

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 12*x - 9) + 1 / (x^2 + 3*x - 9) + 1 / (x^2 - 14*x - 9) = 0

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x = -9 ∨ x = -3 ∨ x = 3 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l430_43059


namespace NUMINAMATH_CALUDE_snowball_distribution_l430_43060

/-- Represents the number of snowballs each person has -/
structure Snowballs :=
  (charlie : ℕ)
  (lucy : ℕ)
  (linus : ℕ)

/-- The initial state of snowballs -/
def initial_state : Snowballs :=
  { charlie := 19 + 31,  -- Lucy's snowballs + difference
    lucy := 19,
    linus := 0 }

/-- The final state after Charlie gives half his snowballs to Linus -/
def final_state : Snowballs :=
  { charlie := (19 + 31) / 2,
    lucy := 19,
    linus := (19 + 31) / 2 }

/-- Theorem stating the correct distribution of snowballs after sharing -/
theorem snowball_distribution :
  final_state.charlie = 25 ∧
  final_state.lucy = 19 ∧
  final_state.linus = 25 :=
by sorry

end NUMINAMATH_CALUDE_snowball_distribution_l430_43060


namespace NUMINAMATH_CALUDE_line_m_equation_l430_43063

/-- Two distinct lines in the xy-plane that intersect at the origin -/
structure IntersectingLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m
  intersect_origin : (0, 0) ∈ ℓ ∩ m

/-- Reflection of a point about a line -/
def reflect (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The equation of line ℓ is 2x - y = 0 -/
def line_ℓ_eq (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 = 0

theorem line_m_equation (lines : IntersectingLines) 
  (h_ℓ_eq : ∀ p ∈ lines.ℓ, line_ℓ_eq p)
  (h_Q : reflect (reflect (-2, 3) lines.ℓ) lines.m = (3, -1)) :
  ∀ p ∈ lines.m, 3 * p.1 + p.2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l430_43063


namespace NUMINAMATH_CALUDE_functional_equation_solution_l430_43095

-- Define the property that the function must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f x + f y) = f x ^ 2 + y

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l430_43095


namespace NUMINAMATH_CALUDE_trig_identity_l430_43011

theorem trig_identity (α : ℝ) :
  (Real.sin (6 * α) + Real.sin (7 * α) + Real.sin (8 * α) + Real.sin (9 * α)) /
  (Real.cos (6 * α) + Real.cos (7 * α) + Real.cos (8 * α) + Real.cos (9 * α)) =
  Real.tan (15 * α / 2) := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l430_43011


namespace NUMINAMATH_CALUDE_polynomial_minimum_value_l430_43040

theorem polynomial_minimum_value : 
  (∀ a b : ℝ, a^2 + 2*b^2 + 2*a + 4*b + 2008 ≥ 2005) ∧ 
  (∃ a b : ℝ, a^2 + 2*b^2 + 2*a + 4*b + 2008 = 2005) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_minimum_value_l430_43040


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l430_43036

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -5 + 7*I
  let z₂ : ℂ := 9 - 3*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = 2 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l430_43036


namespace NUMINAMATH_CALUDE_winnie_repetitions_l430_43062

/-- Calculates the number of repetitions completed today given yesterday's
    repetitions and the difference in performance. -/
def repetitions_today (yesterday : ℕ) (difference : ℕ) : ℕ :=
  yesterday - difference

/-- Proves that Winnie completed 73 repetitions today given the conditions. -/
theorem winnie_repetitions :
  repetitions_today 86 13 = 73 := by
  sorry

end NUMINAMATH_CALUDE_winnie_repetitions_l430_43062


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l430_43001

/-- Represents the height of a tree that quadruples each year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (4 ^ years)

/-- The problem statement -/
theorem tree_height_after_two_years 
  (h : tree_height 1 4 = 256) : 
  tree_height 1 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l430_43001


namespace NUMINAMATH_CALUDE_x_value_on_line_k_l430_43034

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem x_value_on_line_k (x y : ℝ) :
  line_k x 6 → 
  line_k 10 y → 
  x * y = 60 →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_value_on_line_k_l430_43034


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l430_43085

theorem largest_angle_in_triangle (a b c : ℝ) (h1 : a + 2*b + 2*c = a^2) (h2 : a + 2*b - 2*c = -3) :
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A ≤ 120 ∧ B ≤ 120 ∧ C = 120 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l430_43085


namespace NUMINAMATH_CALUDE_consecutive_naturals_integer_quotient_l430_43032

theorem consecutive_naturals_integer_quotient :
  ∃! (n : ℕ), (n + 1 : ℚ) / n = ⌊(n + 1 : ℚ) / n⌋ ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_naturals_integer_quotient_l430_43032


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l430_43030

/-- Diamond operation -/
def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem stating that if a ◇ 4 = 21, then a = 53/3 -/
theorem diamond_equation_solution :
  ∀ a : ℝ, diamond a 4 = 21 → a = 53/3 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l430_43030


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l430_43004

/-- Given a line 2ax + by - 2 = 0 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b*2 = 2 → (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y*2 = 2 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l430_43004


namespace NUMINAMATH_CALUDE_sqrt_172_01_l430_43099

theorem sqrt_172_01 (h1 : Real.sqrt 1.7201 = 1.311) (h2 : Real.sqrt 17.201 = 4.147) :
  Real.sqrt 172.01 = 13.11 ∨ Real.sqrt 172.01 = -13.11 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_172_01_l430_43099


namespace NUMINAMATH_CALUDE_city_population_l430_43067

/-- If 96% of a city's population is 23040, then the total population is 24000. -/
theorem city_population (population : ℕ) : 
  (96 : ℚ) / 100 * population = 23040 → population = 24000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_l430_43067


namespace NUMINAMATH_CALUDE_quadrilateral_area_l430_43098

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first smaller triangle -/
  area1 : ℝ
  /-- Area of the second smaller triangle -/
  area2 : ℝ
  /-- Area of the third smaller triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ
  /-- The sum of all areas equals the area of the original triangle -/
  area_sum : area1 + area2 + area3 + areaQuad > 0

/-- The main theorem about the area of the quadrilateral -/
theorem quadrilateral_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 5) (h2 : t.area2 = 9) (h3 : t.area3 = 9) : 
  t.areaQuad = 45 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_l430_43098


namespace NUMINAMATH_CALUDE_circle_radius_c_value_l430_43016

theorem circle_radius_c_value :
  ∀ (c : ℝ),
  (∀ (x y : ℝ), x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 36) →
  c = -19 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_c_value_l430_43016


namespace NUMINAMATH_CALUDE_correct_student_activities_and_championships_l430_43093

/-- The number of ways for students to sign up for activities and the number of possible championship outcomes -/
def student_activities_and_championships 
  (num_students : ℕ) 
  (num_activities : ℕ) 
  (num_championships : ℕ) : ℕ × ℕ :=
  (num_activities ^ num_students, num_students ^ num_championships)

/-- Theorem stating the correct number of ways for 4 students to sign up for 3 activities and compete in 3 championships -/
theorem correct_student_activities_and_championships :
  student_activities_and_championships 4 3 3 = (3^4, 4^3) := by
  sorry

end NUMINAMATH_CALUDE_correct_student_activities_and_championships_l430_43093


namespace NUMINAMATH_CALUDE_nested_expression_value_l430_43041

/-- The nested expression that needs to be evaluated -/
def nestedExpression : ℕ := 2*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4)))))))))

/-- Theorem stating that the nested expression equals 699050 -/
theorem nested_expression_value : nestedExpression = 699050 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l430_43041


namespace NUMINAMATH_CALUDE_f_g_properties_l430_43047

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

def g (a b : ℝ) (x : ℝ) : ℝ := f a x + 1/2 * x^2 - b * x

def tangent_perpendicular (a : ℝ) : Prop :=
  (deriv (f a) 1) * (-1/2) = -1

def has_decreasing_interval (a b : ℝ) : Prop :=
  ∃ x y, x < y ∧ ∀ z ∈ Set.Ioo x y, (deriv (g a b) z) < 0

def extreme_points (a b : ℝ) (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ (deriv (g a b) x₁) = 0 ∧ (deriv (g a b) x₂) = 0

theorem f_g_properties (a b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : tangent_perpendicular a)
  (h2 : extreme_points a b x₁ x₂)
  (h3 : b ≥ 7/2) :
  a = 1 ∧ 
  (has_decreasing_interval a b → b > 3) ∧
  (g a b x₁ - g a b x₂ ≥ 15/8 - 2 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_f_g_properties_l430_43047


namespace NUMINAMATH_CALUDE_garden_area_increase_l430_43037

theorem garden_area_increase : 
  let original_length : ℝ := 40
  let original_width : ℝ := 10
  let original_perimeter : ℝ := 2 * (original_length + original_width)
  let new_side_length : ℝ := original_perimeter / 4
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_side_length ^ 2
  new_area - original_area = 225 := by sorry

end NUMINAMATH_CALUDE_garden_area_increase_l430_43037


namespace NUMINAMATH_CALUDE_train_stations_distance_l430_43082

/-- The distance between two stations given train meeting points -/
theorem train_stations_distance
  (first_meet_offset : ℝ)  -- Distance from midpoint to first meeting point
  (second_meet_distance : ℝ)  -- Distance from eastern station to second meeting point
  (h1 : first_meet_offset = 10)  -- First meeting 10 km west of midpoint
  (h2 : second_meet_distance = 40)  -- Second meeting 40 km from eastern station
  : ℝ :=
by
  -- The distance between the stations
  let distance : ℝ := 140
  -- Proof goes here
  sorry

#check train_stations_distance

end NUMINAMATH_CALUDE_train_stations_distance_l430_43082


namespace NUMINAMATH_CALUDE_block_of_flats_floors_l430_43058

theorem block_of_flats_floors (n : ℕ) (h_even : Even n) :
  (n / 2 * 6 + n / 2 * 5) * 4 = 264 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_block_of_flats_floors_l430_43058


namespace NUMINAMATH_CALUDE_minimum_cents_to_win_l430_43025

/-- Represents the state of the game -/
structure GameState where
  beans : ℕ
  cents : ℕ

/-- Applies the penny rule: multiply beans by 5 and add 1 cent -/
def applyPenny (state : GameState) : GameState :=
  { beans := state.beans * 5, cents := state.cents + 1 }

/-- Applies the nickel rule: add 1 bean and 5 cents -/
def applyNickel (state : GameState) : GameState :=
  { beans := state.beans + 1, cents := state.cents + 5 }

/-- Checks if the game is won -/
def isWinningState (state : GameState) : Prop :=
  state.beans > 2008 ∧ state.beans % 100 = 42

/-- Represents a sequence of moves in the game -/
inductive GameMove
  | penny
  | nickel

def applyMove (state : GameState) (move : GameMove) : GameState :=
  match move with
  | GameMove.penny => applyPenny state
  | GameMove.nickel => applyNickel state

def applyMoves (state : GameState) (moves : List GameMove) : GameState :=
  moves.foldl applyMove state

theorem minimum_cents_to_win :
  ∃ (moves : List GameMove),
    let finalState := applyMoves { beans := 0, cents := 0 } moves
    isWinningState finalState ∧
    finalState.cents = 35 ∧
    (∀ (otherMoves : List GameMove),
      let otherFinalState := applyMoves { beans := 0, cents := 0 } otherMoves
      isWinningState otherFinalState → otherFinalState.cents ≥ 35) :=
by sorry

end NUMINAMATH_CALUDE_minimum_cents_to_win_l430_43025


namespace NUMINAMATH_CALUDE_triangle_area_isosceles_l430_43042

/-- The area of a triangle with two sides of length 30 and one side of length 40 -/
theorem triangle_area_isosceles (a b c : ℝ) (h1 : a = 30) (h2 : b = 30) (h3 : c = 40) : 
  ∃ area : ℝ, abs (area - Real.sqrt (50 * (50 - a) * (50 - b) * (50 - c))) < 0.01 ∧ 
  446.99 < area ∧ area < 447.01 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_isosceles_l430_43042


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l430_43075

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l430_43075


namespace NUMINAMATH_CALUDE_zeros_of_f_shifted_l430_43097

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem zeros_of_f_shifted (x : ℝ) : 
  f (x - 1) = 0 ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_f_shifted_l430_43097


namespace NUMINAMATH_CALUDE_max_valid_sequence_length_l430_43017

/-- A sequence of integers satisfying the given property -/
def ValidSequence (x : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → i + j ≤ n →
    (3 ∣ x i - x j) → (3 ∣ x (i + j) + x i + x j + 1)

/-- The maximum length of a valid sequence -/
def MaxValidSequenceLength : ℕ := 8

/-- Theorem stating that the maximum length of a valid sequence is 8 -/
theorem max_valid_sequence_length :
  (∃ x, ValidSequence x MaxValidSequenceLength) ∧
  (∀ n > MaxValidSequenceLength, ¬∃ x, ValidSequence x n) :=
sorry

end NUMINAMATH_CALUDE_max_valid_sequence_length_l430_43017


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l430_43048

/-- Represents the seating capacity of a bus with specific seat arrangements. -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  back_seat_capacity : Nat

/-- Calculates the total number of people who can sit in the bus. -/
def total_seating_capacity (bus : BusSeating) : Nat :=
  (bus.left_seats + bus.right_seats) * bus.people_per_seat + bus.back_seat_capacity

/-- Theorem stating the total seating capacity of the bus with given conditions. -/
theorem bus_seating_capacity : 
  ∀ (bus : BusSeating), 
    bus.left_seats = 15 → 
    bus.right_seats = bus.left_seats - 3 →
    bus.people_per_seat = 3 →
    bus.back_seat_capacity = 11 →
    total_seating_capacity bus = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l430_43048


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l430_43054

theorem negative_fraction_comparison : -4/3 < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l430_43054


namespace NUMINAMATH_CALUDE_jake_and_kendra_weight_l430_43033

/-- Jake's current weight in pounds -/
def jake_weight : ℕ := 198

/-- The weight Jake would lose in pounds -/
def weight_loss : ℕ := 8

/-- Kendra's weight in pounds -/
def kendra_weight : ℕ := (jake_weight - weight_loss) / 2

/-- The combined weight of Jake and Kendra in pounds -/
def combined_weight : ℕ := jake_weight + kendra_weight

/-- Theorem stating the combined weight of Jake and Kendra -/
theorem jake_and_kendra_weight : combined_weight = 293 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_kendra_weight_l430_43033
