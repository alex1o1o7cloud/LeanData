import Mathlib

namespace NUMINAMATH_CALUDE_sequence_formula_l4084_408463

theorem sequence_formula (a : ℕ+ → ℚ) :
  (∀ n : ℕ+, a (n + 1) / a n = (n + 2) / n) →
  a 1 = 1 →
  ∀ n : ℕ+, a n = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l4084_408463


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l4084_408420

/-- 
Given a subtraction problem where:
- The tens digit 7 was mistaken for 9
- The ones digit 3 was mistaken for 8
- The mistaken subtraction resulted in a difference of 76

Prove that the correct difference is 51.
-/
theorem correct_subtraction_result : 
  ∀ (original_tens original_ones mistaken_tens mistaken_ones mistaken_difference : ℕ),
  original_tens = 7 →
  original_ones = 3 →
  mistaken_tens = 9 →
  mistaken_ones = 8 →
  mistaken_difference = 76 →
  (mistaken_tens * 10 + mistaken_ones) - (original_tens * 10 + original_ones) = mistaken_difference →
  (original_tens * 10 + original_ones) - 
    ((mistaken_tens * 10 + mistaken_ones) - (original_tens * 10 + original_ones)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l4084_408420


namespace NUMINAMATH_CALUDE_whitewashing_cost_calculation_l4084_408496

/-- Calculates the cost of white washing a room with given specifications. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ)
                     (doorCount doorLength doorWidth : ℝ)
                     (windowCount windowLength windowWidth : ℝ)
                     (costPerSqFt additionalPaintPercentage : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength + roomWidth) * roomHeight
  let doorArea := doorCount * doorLength * doorWidth
  let windowArea := windowCount * windowLength * windowWidth
  let paintableArea := wallArea - doorArea - windowArea
  let totalPaintArea := paintableArea * (1 + additionalPaintPercentage)
  totalPaintArea * costPerSqFt

/-- Theorem stating the cost of white washing the room with given specifications. -/
theorem whitewashing_cost_calculation :
  whitewashingCost 25 15 12 2 6 3 5 4 3 7 0.1 = 6652.8 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_calculation_l4084_408496


namespace NUMINAMATH_CALUDE_max_value_implies_a_l4084_408465

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^2 + 4 * a * x - 4 * a - a^2

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ -5) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = -5) →
  a = 5/4 ∨ a = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l4084_408465


namespace NUMINAMATH_CALUDE_incorrect_permutations_of_good_l4084_408421

-- Define a structure for our word
structure Word where
  length : Nat
  repeated_letter_count : Nat

-- Define our specific word "good"
def good : Word := { length := 4, repeated_letter_count := 2 }

-- Theorem statement
theorem incorrect_permutations_of_good (w : Word) (h1 : w = good) : 
  (w.length.factorial / w.repeated_letter_count.factorial) - 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_of_good_l4084_408421


namespace NUMINAMATH_CALUDE_milk_leftover_problem_l4084_408454

/-- Calculates the amount of milk left over from yesterday given today's milk production and sales --/
def milk_leftover (morning_milk : ℕ) (evening_milk : ℕ) (sold_milk : ℕ) (total_left : ℕ) : ℕ :=
  total_left - ((morning_milk + evening_milk) - sold_milk)

/-- Theorem stating that given the problem conditions, the milk leftover from yesterday is 15 gallons --/
theorem milk_leftover_problem : milk_leftover 365 380 612 148 = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_leftover_problem_l4084_408454


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l4084_408489

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l4084_408489


namespace NUMINAMATH_CALUDE_smallest_x_value_l4084_408495

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (240 + x)) :
  x ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l4084_408495


namespace NUMINAMATH_CALUDE_triangle_inequality_l4084_408436

/-- For a triangle with side lengths a, b, and c, area 1/4, and circumradius 1,
    √a + √b + √c < 1/a + 1/b + 1/c -/
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (harea : a * b * c / 4 = 1/4) (hcircum : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1/a + 1/b + 1/c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4084_408436


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4084_408432

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4084_408432


namespace NUMINAMATH_CALUDE_bus_count_l4084_408444

theorem bus_count (total_students : ℕ) (students_per_bus : ℕ) (h1 : total_students = 360) (h2 : students_per_bus = 45) :
  total_students / students_per_bus = 8 :=
by sorry

end NUMINAMATH_CALUDE_bus_count_l4084_408444


namespace NUMINAMATH_CALUDE_negative_of_negative_five_l4084_408453

theorem negative_of_negative_five : -(- 5) = 5 := by sorry

end NUMINAMATH_CALUDE_negative_of_negative_five_l4084_408453


namespace NUMINAMATH_CALUDE_quadratic_equation_has_solution_l4084_408498

theorem quadratic_equation_has_solution (a b : ℝ) :
  ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_has_solution_l4084_408498


namespace NUMINAMATH_CALUDE_trig_equation_iff_equal_l4084_408447

theorem trig_equation_iff_equal (a b : Real) 
  (ha : 0 ≤ a ∧ a ≤ π/2) (hb : 0 ≤ b ∧ b ≤ π/2) : 
  (Real.sin a)^6 + 3*(Real.sin a)^2*(Real.cos b)^2 + (Real.cos b)^6 = 1 ↔ a = b := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_iff_equal_l4084_408447


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l4084_408466

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 7 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 28 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l4084_408466


namespace NUMINAMATH_CALUDE_system_of_equations_range_l4084_408481

theorem system_of_equations_range (x y m : ℝ) : 
  x + 2*y = 1 + m →
  2*x + y = 3 →
  x + y > 0 →
  m > -4 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_range_l4084_408481


namespace NUMINAMATH_CALUDE_system_solution_l4084_408478

theorem system_solution (a : ℂ) (x y z : ℝ) (k l : ℤ) :
  Complex.abs (a + 1 / a) = 2 →
  Real.tan x = 1 ∨ Real.tan x = -1 →
  Real.sin y = 1 ∨ Real.sin y = -1 →
  Real.cos z = 0 →
  x = Real.pi / 2 + k * Real.pi ∧
  y = Real.pi / 2 + k * Real.pi ∧
  z = Real.pi / 2 + l * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4084_408478


namespace NUMINAMATH_CALUDE_prism_volume_l4084_408464

/-- The volume of a right rectangular prism with face areas 15, 10, and 30 square inches is 30√5 cubic inches. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 15) (h2 : b * c = 10) (h3 : c * a = 30) :
  a * b * c = 30 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l4084_408464


namespace NUMINAMATH_CALUDE_ellipse_origin_inside_l4084_408499

theorem ellipse_origin_inside (k : ℝ) : 
  (∀ x y : ℝ, k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0 → x^2 + y^2 > 0) →
  (k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) →
  0 < |k| ∧ |k| < 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_origin_inside_l4084_408499


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l4084_408440

theorem quadratic_one_solution (k : ℚ) : 
  (∃! x, 2 * x^2 - 5 * x + k = 0) ↔ k = 25/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l4084_408440


namespace NUMINAMATH_CALUDE_operation_proof_l4084_408479

theorem operation_proof (v : ℝ) : (v - v / 3) - (v - v / 3) / 3 = 12 → v = 27 := by
  sorry

end NUMINAMATH_CALUDE_operation_proof_l4084_408479


namespace NUMINAMATH_CALUDE_pizza_fraction_eaten_l4084_408485

/-- The fraction of pizza eaten after n trips, where each trip consumes one-third of the remaining pizza -/
def fractionEaten (n : ℕ) : ℚ :=
  1 - (2/3)^n

/-- The number of trips to the refrigerator -/
def numTrips : ℕ := 6

theorem pizza_fraction_eaten :
  fractionEaten numTrips = 364 / 729 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_eaten_l4084_408485


namespace NUMINAMATH_CALUDE_widgets_in_shipping_box_is_300_l4084_408471

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.width * d.length * d.height

/-- Represents the problem setup -/
structure WidgetProblem where
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions
  widgetsPerCarton : ℕ

/-- Calculates the number of widgets in a shipping box -/
def widgetsInShippingBox (p : WidgetProblem) : ℕ :=
  let cartonsInBox := (boxVolume p.shippingBoxDimensions) / (boxVolume p.cartonDimensions)
  cartonsInBox * p.widgetsPerCarton

/-- The main theorem to prove -/
theorem widgets_in_shipping_box_is_300 (p : WidgetProblem) : 
  p.cartonDimensions = ⟨4, 4, 5⟩ ∧ 
  p.shippingBoxDimensions = ⟨20, 20, 20⟩ ∧ 
  p.widgetsPerCarton = 3 → 
  widgetsInShippingBox p = 300 := by
  sorry


end NUMINAMATH_CALUDE_widgets_in_shipping_box_is_300_l4084_408471


namespace NUMINAMATH_CALUDE_grassy_plot_width_l4084_408476

/-- Proves that the width of a rectangular grassy plot is 55 meters, given specific conditions -/
theorem grassy_plot_width : 
  ∀ (length width path_width : ℝ) (cost_per_sq_meter cost_total : ℝ),
  length = 110 →
  path_width = 2.5 →
  cost_per_sq_meter = 0.5 →
  cost_total = 425 →
  ((length + 2 * path_width) * (width + 2 * path_width) - length * width) * cost_per_sq_meter = cost_total →
  width = 55 := by
sorry

end NUMINAMATH_CALUDE_grassy_plot_width_l4084_408476


namespace NUMINAMATH_CALUDE_log_expression_equals_twelve_l4084_408409

theorem log_expression_equals_twelve : 
  (4 - (Real.log 4 / Real.log 36) - (Real.log 18 / Real.log 6)) / (Real.log 3 / Real.log 4) * 
  ((Real.log 27 / Real.log 8) + (Real.log 9 / Real.log 2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_twelve_l4084_408409


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4084_408486

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if a line through one vertex on the imaginary axis and perpendicular to the y-axis
    forms an equilateral triangle with the other vertex on the imaginary axis and
    the two points where it intersects the hyperbola, then the eccentricity is √10/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let B₁ := (0, b)
  let B₂ := (0, -b)
  let line := fun (x : ℝ) ↦ b
  let P := (-Real.sqrt 2 * a, b)
  let Q := (Real.sqrt 2 * a, b)
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2 ∧
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1 - B₂.1)^2 + (P.2 - B₂.2)^2 ∧
  (P.1 - B₂.1)^2 + (P.2 - B₂.2)^2 = (Q.1 - B₂.1)^2 + (Q.2 - B₂.2)^2 →
  Real.sqrt (1 + b^2 / a^2) = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4084_408486


namespace NUMINAMATH_CALUDE_complex_number_properties_l4084_408492

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem to prove the four statements
theorem complex_number_properties :
  (i^2017 = i) ∧
  ((i + 1) * i = -1 + i) ∧
  ((1 - i) / (1 + i) = -i) ∧
  (Complex.abs (2 + i) = Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l4084_408492


namespace NUMINAMATH_CALUDE_max_xy_value_l4084_408469

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (m : ℝ), m = 1/4 ∧ ∀ (z : ℝ), z = x * y → z ≤ m := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l4084_408469


namespace NUMINAMATH_CALUDE_average_of_four_l4084_408457

theorem average_of_four (total : ℕ) (avg_all : ℚ) (avg_two : ℚ) :
  total = 6 →
  avg_all = 8 →
  avg_two = 14 →
  (total * avg_all - 2 * avg_two) / (total - 2) = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_four_l4084_408457


namespace NUMINAMATH_CALUDE_equation_solutions_l4084_408451

theorem equation_solutions : 
  ∃! (s : Set ℝ), 
    (∀ x ∈ s, |x - 2| = |x - 1| + |x - 3| + |x - 4|) ∧ 
    s = {2, 2.25} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4084_408451


namespace NUMINAMATH_CALUDE_prob_king_then_ten_l4084_408487

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of 10s in a standard deck -/
def NumTens : ℕ := 4

/-- Probability of drawing a King first and then a 10 from a standard deck -/
theorem prob_king_then_ten : 
  (NumKings : ℚ) / StandardDeck * NumTens / (StandardDeck - 1) = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_then_ten_l4084_408487


namespace NUMINAMATH_CALUDE_smallest_multiple_l4084_408442

theorem smallest_multiple (x : ℕ) : x = 48 ↔ (
  x > 0 ∧
  (∃ k : ℕ, 600 * x = 1152 * k) ∧
  (∀ y : ℕ, y > 0 → y < x → ¬∃ k : ℕ, 600 * y = 1152 * k)
) := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l4084_408442


namespace NUMINAMATH_CALUDE_ethan_present_count_l4084_408449

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := 53

/-- The difference in presents between Alissa and Ethan -/
def present_difference : ℕ := 22

theorem ethan_present_count : ethan_presents = alissa_presents - present_difference := by
  sorry

end NUMINAMATH_CALUDE_ethan_present_count_l4084_408449


namespace NUMINAMATH_CALUDE_area_of_triangle_l4084_408491

/-- Two externally tangent circles with centers O and O' and radii 1 and 2 -/
structure TangentCircles where
  O : ℝ × ℝ
  O' : ℝ × ℝ
  radius_C : ℝ
  radius_C' : ℝ
  tangent_externally : (O.1 - O'.1)^2 + (O.2 - O'.2)^2 = (radius_C + radius_C')^2
  radius_C_eq_1 : radius_C = 1
  radius_C'_eq_2 : radius_C' = 2

/-- Point P is on circle C, and P' is on circle C' -/
def TangentPoints (tc : TangentCircles) :=
  {P : ℝ × ℝ | (P.1 - tc.O.1)^2 + (P.2 - tc.O.2)^2 = tc.radius_C^2} ×
  {P' : ℝ × ℝ | (P'.1 - tc.O'.1)^2 + (P'.2 - tc.O'.2)^2 = tc.radius_C'^2}

/-- X is the intersection point of O'P and OP' -/
def IntersectionPoint (tc : TangentCircles) (tp : TangentPoints tc) : ℝ × ℝ :=
  sorry -- Definition of X as the intersection point

/-- The area of triangle OXO' -/
def TriangleArea (tc : TangentCircles) (tp : TangentPoints tc) : ℝ :=
  let X := IntersectionPoint tc tp
  sorry -- Definition of the area of triangle OXO'

/-- Main theorem: The area of triangle OXO' is (4√2 - √5) / 3 -/
theorem area_of_triangle (tc : TangentCircles) (tp : TangentPoints tc) :
  TriangleArea tc tp = (4 * Real.sqrt 2 - Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_l4084_408491


namespace NUMINAMATH_CALUDE_complex_square_l4084_408477

theorem complex_square (z : ℂ) : z = 2 + 5*I → z^2 = -21 + 20*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_l4084_408477


namespace NUMINAMATH_CALUDE_total_pages_is_62_l4084_408483

/-- The number of pages Jairus read -/
def jairus_pages : ℕ := 20

/-- The number of pages Arniel read -/
def arniel_pages : ℕ := 2 * jairus_pages + 2

/-- The total number of pages read by Jairus and Arniel -/
def total_pages : ℕ := jairus_pages + arniel_pages

/-- Theorem stating that the total number of pages read is 62 -/
theorem total_pages_is_62 : total_pages = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_is_62_l4084_408483


namespace NUMINAMATH_CALUDE_lisa_likes_one_last_digit_l4084_408418

def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem lisa_likes_one_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, last_digit n = d → (divisible_by_5 n ∧ divisible_by_2 n) :=
by
  sorry

end NUMINAMATH_CALUDE_lisa_likes_one_last_digit_l4084_408418


namespace NUMINAMATH_CALUDE_function_equality_implies_m_equals_one_l4084_408448

/-- Given functions f and g, and a condition on their values at x = -1, 
    prove that the parameter m in g equals 1. -/
theorem function_equality_implies_m_equals_one :
  let f : ℝ → ℝ := λ x ↦ 3 * x^3 - 1/x + 5
  let g : ℝ → ℝ := λ x ↦ 3 * x^2 - m
  f (-1) - g (-1) = 1 →
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_equals_one_l4084_408448


namespace NUMINAMATH_CALUDE_cos_4theta_from_complex_exp_l4084_408405

theorem cos_4theta_from_complex_exp (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 4 →
  Real.cos (4 * θ) = -287 / 256 := by
  sorry

end NUMINAMATH_CALUDE_cos_4theta_from_complex_exp_l4084_408405


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_l4084_408407

theorem no_real_solution_for_equation : 
  ∀ x : ℝ, ¬(5 * (2*x)^2 - 3*(2*x) + 7 = 2*(8*x^2 - 2*x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_l4084_408407


namespace NUMINAMATH_CALUDE_drawn_games_in_specific_tournament_l4084_408406

/-- Represents a chess tournament. -/
structure ChessTournament where
  participants : Nat
  total_matches : Nat
  wins_per_participant : Nat
  has_growing_lists : Bool

/-- Calculates the number of drawn games in a chess tournament. -/
def drawn_games (tournament : ChessTournament) : Nat :=
  tournament.total_matches - (tournament.participants * tournament.wins_per_participant)

/-- Theorem stating the number of drawn games in the specific tournament. -/
theorem drawn_games_in_specific_tournament :
  ∀ (t : ChessTournament),
    t.participants = 12 ∧
    t.total_matches = (12 * 11) / 2 ∧
    t.wins_per_participant = 1 ∧
    t.has_growing_lists = true →
    drawn_games t = 54 := by
  sorry

end NUMINAMATH_CALUDE_drawn_games_in_specific_tournament_l4084_408406


namespace NUMINAMATH_CALUDE_sequence_difference_equals_170000_l4084_408488

/-- The sum of an arithmetic sequence with first term a, last term l, and n terms -/
def arithmetic_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The difference between two sums of arithmetic sequences -/
def sequence_difference : ℕ :=
  arithmetic_sum 2001 2100 100 - arithmetic_sum 301 400 100

theorem sequence_difference_equals_170000 : sequence_difference = 170000 := by
  sorry

end NUMINAMATH_CALUDE_sequence_difference_equals_170000_l4084_408488


namespace NUMINAMATH_CALUDE_initial_oranges_count_l4084_408437

/-- The number of oranges initially in the basket -/
def initial_oranges : ℕ := sorry

/-- The number of oranges taken from the basket -/
def oranges_taken : ℕ := 5

/-- The number of oranges remaining in the basket -/
def oranges_remaining : ℕ := 3

/-- Theorem stating that the initial number of oranges is 8 -/
theorem initial_oranges_count : initial_oranges = 8 := by sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l4084_408437


namespace NUMINAMATH_CALUDE_rational_coloring_exists_l4084_408458

theorem rational_coloring_exists : ∃ (f : ℚ → Bool), 
  (∀ x : ℚ, x ≠ 0 → f x ≠ f (-x)) ∧ 
  (∀ x : ℚ, x ≠ 1/2 → f x ≠ f (1 - x)) ∧ 
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → f x ≠ f (1 / x)) := by
  sorry

end NUMINAMATH_CALUDE_rational_coloring_exists_l4084_408458


namespace NUMINAMATH_CALUDE_place_left_l4084_408450

/-- A two-digit number is between 10 and 99, inclusive. -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A one-digit number is between 1 and 9, inclusive. -/
def is_one_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

/-- Placing a one-digit number b to the left of a two-digit number a results in 100b + a. -/
theorem place_left (a b : ℕ) (ha : is_two_digit a) (hb : is_one_digit b) :
  100 * b + a = (100 * b + a) := by sorry

end NUMINAMATH_CALUDE_place_left_l4084_408450


namespace NUMINAMATH_CALUDE_suit_cost_problem_l4084_408431

theorem suit_cost_problem (x : ℝ) (h1 : x + (3 * x + 200) = 1400) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_suit_cost_problem_l4084_408431


namespace NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l4084_408404

-- Define a decreasing function on an open interval
def DecreasingOnInterval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

-- Main theorem
theorem range_of_m_for_decreasing_function 
  (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingOnInterval f (-2) 2)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  0 < m ∧ m < 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_decreasing_function_l4084_408404


namespace NUMINAMATH_CALUDE_square_perimeter_l4084_408443

/-- Theorem: A square with an area of 625 cm² has a perimeter of 100 cm. -/
theorem square_perimeter (s : ℝ) (h_area : s^2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4084_408443


namespace NUMINAMATH_CALUDE_correct_calculation_l4084_408427

def correct_sum (mistaken_sum : ℕ) (original_tens : ℕ) (mistaken_tens : ℕ) 
                (original_units : ℕ) (mistaken_units : ℕ) : ℕ :=
  mistaken_sum - (mistaken_units - original_units) + (original_tens - mistaken_tens) * 10

theorem correct_calculation (mistaken_sum : ℕ) (original_tens : ℕ) (mistaken_tens : ℕ) 
                            (original_units : ℕ) (mistaken_units : ℕ) : 
  mistaken_sum = 111 ∧ 
  original_tens = 7 ∧ 
  mistaken_tens = 4 ∧ 
  original_units = 5 ∧ 
  mistaken_units = 8 → 
  correct_sum mistaken_sum original_tens mistaken_tens original_units mistaken_units = 138 := by
  sorry

#eval correct_sum 111 7 4 5 8

end NUMINAMATH_CALUDE_correct_calculation_l4084_408427


namespace NUMINAMATH_CALUDE_arithmetic_increasing_iff_positive_difference_l4084_408441

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: An arithmetic sequence is increasing if and only if its common difference is positive -/
theorem arithmetic_increasing_iff_positive_difference (a : ℕ → ℝ) :
  ArithmeticSequence a → (IncreasingSequence a ↔ ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_increasing_iff_positive_difference_l4084_408441


namespace NUMINAMATH_CALUDE_child_ticket_price_soccer_match_l4084_408426

/-- The price of a child's ticket at a soccer match -/
def child_ticket_price (num_adults num_children : ℕ) (adult_ticket_price total_bill : ℚ) : ℚ :=
  (total_bill - num_adults * adult_ticket_price) / num_children

theorem child_ticket_price_soccer_match :
  child_ticket_price 25 32 12 450 = 469/100 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_price_soccer_match_l4084_408426


namespace NUMINAMATH_CALUDE_system_solution_l4084_408460

theorem system_solution : ∃ (x y : ℝ), (7 * x - 3 * y = 2) ∧ (2 * x + y = 8) := by
  use 2, 4
  sorry

end NUMINAMATH_CALUDE_system_solution_l4084_408460


namespace NUMINAMATH_CALUDE_smallest_three_digit_square_base_seven_l4084_408494

/-- The smallest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 7

/-- Converts a natural number to its base 7 representation -/
def to_base_seven (n : ℕ) : List ℕ := sorry

/-- Checks if a number has exactly 3 digits when written in base 7 -/
def has_three_digits_base_seven (n : ℕ) : Prop :=
  (to_base_seven n).length = 3

theorem smallest_three_digit_square_base_seven :
  (M ^ 2 ≥ 7^2) ∧
  (M ^ 2 < 7^3) ∧
  (∀ k : ℕ, k < M → ¬(has_three_digits_base_seven (k^2))) ∧
  (to_base_seven M = [1, 0]) := by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_square_base_seven_l4084_408494


namespace NUMINAMATH_CALUDE_reciprocal_sum_inequality_l4084_408468

theorem reciprocal_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≤ 3) : 1/x + 1/y + 1/z ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_inequality_l4084_408468


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_l4084_408401

/-- The fixed cost to run the molding machine per week -/
def fixed_cost : ℝ := 7640

/-- The cost to mold each handle -/
def mold_cost : ℝ := 0.60

/-- The selling price per handle -/
def selling_price : ℝ := 4.60

/-- The number of handles needed to break even -/
def break_even_quantity : ℕ := 1910

/-- Theorem stating that the fixed cost is correct given the conditions -/
theorem fixed_cost_calculation :
  fixed_cost = (selling_price - mold_cost) * break_even_quantity := by
  sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_l4084_408401


namespace NUMINAMATH_CALUDE_g_prime_symmetry_l4084_408434

open Function Real

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the derivatives of f and g
variable (f' g' : ℝ → ℝ)

-- Assume f' is the derivative of f and g' is the derivative of g
variable (hf : ∀ x, HasDerivAt f (f' x) x)
variable (hg : ∀ x, HasDerivAt g (g' x) x)

-- Define the conditions
variable (h1 : ∀ x, f x + g' x = 5)
variable (h2 : ∀ x, f (2 - x) - g' (2 + x) = 5)
variable (h3 : Odd g)

-- State the theorem
theorem g_prime_symmetry (x : ℝ) : g' (8 - x) = g' x := sorry

end NUMINAMATH_CALUDE_g_prime_symmetry_l4084_408434


namespace NUMINAMATH_CALUDE_equation_solutions_l4084_408470

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧ x₂ = (5 - Real.sqrt 21) / 2 ∧
    x₁^2 - 5*x₁ + 1 = 0 ∧ x₂^2 - 5*x₂ + 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 5 ∧ y₂ = 10/3 ∧
    2*(y₁-5)^2 + y₁*(y₁-5) = 0 ∧ 2*(y₂-5)^2 + y₂*(y₂-5) = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l4084_408470


namespace NUMINAMATH_CALUDE_larger_number_problem_l4084_408493

theorem larger_number_problem (s l : ℝ) : 
  s = 48 → l - s = (1 / 3) * l → l = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4084_408493


namespace NUMINAMATH_CALUDE_sin_pi_12_function_value_l4084_408446

theorem sin_pi_12_function_value
  (f : ℝ → ℝ)
  (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_sin_pi_12_function_value_l4084_408446


namespace NUMINAMATH_CALUDE_polynomial_bound_l4084_408438

theorem polynomial_bound (a b c : ℝ) :
  (∀ x : ℝ, abs x ≤ 1 → abs (a * x^2 + b * x + c) ≤ 1) →
  (∀ x : ℝ, abs x ≤ 1 → abs (c * x^2 + b * x + a) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_bound_l4084_408438


namespace NUMINAMATH_CALUDE_smaller_number_proof_l4084_408484

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 14 → y = 3 * x → x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l4084_408484


namespace NUMINAMATH_CALUDE_recipe_sugar_amount_l4084_408413

/-- The amount of sugar in cups already added to the recipe -/
def sugar_added : ℕ := 4

/-- The amount of sugar in cups still needed to be added to the recipe -/
def sugar_needed : ℕ := 3

/-- The total amount of sugar in cups required by the recipe -/
def total_sugar : ℕ := sugar_added + sugar_needed

theorem recipe_sugar_amount : total_sugar = 7 := by sorry

end NUMINAMATH_CALUDE_recipe_sugar_amount_l4084_408413


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l4084_408429

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 1000)
  (h2 : interest = 400)
  (h3 : time = 4)
  : (interest * 100) / (principal * time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l4084_408429


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l4084_408402

/-- Given that the solution set of the inequality (ax-1)(x+1)<0 is (-∞, -1) ∪ (-1/2, +∞),
    prove that a = -2 -/
theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, (a*x - 1)*(x + 1) < 0 ↔ x < -1 ∨ -1/2 < x) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_value_l4084_408402


namespace NUMINAMATH_CALUDE_percentage_less_than_third_l4084_408408

theorem percentage_less_than_third (n1 n2 n3 : ℝ) : 
  n1 = 0.7 * n3 →  -- First number is 30% less than third number
  n2 = 0.9 * n1 →  -- Second number is 10% less than first number
  n2 = 0.63 * n3   -- Second number is 37% less than third number
:= by sorry

end NUMINAMATH_CALUDE_percentage_less_than_third_l4084_408408


namespace NUMINAMATH_CALUDE_ninas_ants_l4084_408412

theorem ninas_ants (spider_count : ℕ) (spider_eyes : ℕ) (ant_eyes : ℕ) (total_eyes : ℕ) :
  spider_count = 3 →
  spider_eyes = 8 →
  ant_eyes = 2 →
  total_eyes = 124 →
  (total_eyes - spider_count * spider_eyes) / ant_eyes = 50 := by
  sorry

end NUMINAMATH_CALUDE_ninas_ants_l4084_408412


namespace NUMINAMATH_CALUDE_a_5_value_l4084_408452

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = r * (a n - a (n - 1))

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 - a 0 = 1 →
  a 5 = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_a_5_value_l4084_408452


namespace NUMINAMATH_CALUDE_cost_for_36_people_l4084_408410

/-- The cost to feed a group of people with chicken combos -/
def cost_to_feed (people : ℕ) (combo_cost : ℚ) (people_per_combo : ℕ) : ℚ :=
  (people / people_per_combo : ℚ) * combo_cost

/-- Theorem: The cost to feed 36 people is $72.00 -/
theorem cost_for_36_people :
  cost_to_feed 36 12 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_36_people_l4084_408410


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_l4084_408428

/-- Proves that the percentage of alcohol in the first vessel is 25% --/
theorem alcohol_percentage_in_first_vessel : 
  ∀ (x : ℝ),
  -- Vessel capacities and total liquid
  let vessel1_capacity : ℝ := 2
  let vessel2_capacity : ℝ := 6
  let total_liquid : ℝ := 8
  let final_vessel_capacity : ℝ := 10
  -- Alcohol percentages
  let vessel2_alcohol_percentage : ℝ := 50
  let final_mixture_percentage : ℝ := 35
  -- Condition: total alcohol in final mixture
  (x / 100) * vessel1_capacity + (vessel2_alcohol_percentage / 100) * vessel2_capacity = 
    (final_mixture_percentage / 100) * final_vessel_capacity →
  -- Conclusion: alcohol percentage in first vessel is 25%
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_first_vessel_l4084_408428


namespace NUMINAMATH_CALUDE_expression_equality_l4084_408400

theorem expression_equality : (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l4084_408400


namespace NUMINAMATH_CALUDE_torn_sheets_count_l4084_408430

/-- Represents a book with numbered pages -/
structure Book where
  /-- The number of the first torn-out page -/
  first_torn_page : ℕ
  /-- The number of the last torn-out page -/
  last_torn_page : ℕ

/-- Calculates the number of torn-out sheets given a book -/
def torn_sheets (b : Book) : ℕ :=
  (b.last_torn_page - b.first_torn_page + 1) / 2

/-- The main theorem stating that 167 sheets were torn out -/
theorem torn_sheets_count (b : Book) 
  (h1 : b.first_torn_page = 185)
  (h2 : b.last_torn_page = 518) :
  torn_sheets b = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l4084_408430


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4084_408480

theorem simplify_and_evaluate (a b : ℝ) (h : a = -b) :
  2 * (3 * a^2 + a - 2*b) - 6 * (a^2 - b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4084_408480


namespace NUMINAMATH_CALUDE_stationery_shop_sales_l4084_408424

theorem stationery_shop_sales (total_sales percent_pens percent_pencils : ℝ) 
  (h_total : total_sales = 100)
  (h_pens : percent_pens = 38)
  (h_pencils : percent_pencils = 35) :
  total_sales - percent_pens - percent_pencils = 27 := by
  sorry

end NUMINAMATH_CALUDE_stationery_shop_sales_l4084_408424


namespace NUMINAMATH_CALUDE_problem_solution_l4084_408423

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (x^4 + 3*y^3 + 10) / 7 = 283/7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4084_408423


namespace NUMINAMATH_CALUDE_real_roots_condition_l4084_408462

theorem real_roots_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ 0 ≤ a ∧ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_condition_l4084_408462


namespace NUMINAMATH_CALUDE_oldest_babysat_age_l4084_408474

theorem oldest_babysat_age (jane_start_age : ℕ) (jane_current_age : ℕ) (years_since_stopped : ℕ) :
  jane_start_age = 18 →
  jane_current_age = 34 →
  years_since_stopped = 12 →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_age ≥ jane_start_age →
    jane_age ≤ jane_current_age - years_since_stopped →
    child_age ≤ jane_age / 2) →
  (jane_current_age - years_since_stopped - jane_start_age) + years_since_stopped + 
    ((jane_current_age - years_since_stopped) / 2) = 23 :=
by sorry

end NUMINAMATH_CALUDE_oldest_babysat_age_l4084_408474


namespace NUMINAMATH_CALUDE_servant_cash_payment_l4084_408422

-- Define the problem parameters
def annual_cash_salary : ℕ := 90
def turban_price : ℕ := 70
def months_worked : ℕ := 9
def months_per_year : ℕ := 12

-- Define the theorem
theorem servant_cash_payment :
  let total_annual_salary := annual_cash_salary + turban_price
  let proportion_worked := months_worked / months_per_year
  let earned_amount := (proportion_worked * total_annual_salary : ℚ).floor
  earned_amount - turban_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_servant_cash_payment_l4084_408422


namespace NUMINAMATH_CALUDE_A_equals_B_l4084_408411

/-- The number of ways to pair r girls with r boys in town A -/
def A (n r : ℕ) : ℕ := (n.choose r)^2 * r.factorial

/-- The number of ways to pair r girls with r boys in town B -/
def B : ℕ → ℕ → ℕ
| 0, _ => 0
| _, 0 => 1
| n+1, r+1 => (2*n+1 - r) * B n r + B n (r+1)

/-- The theorem stating that A(n,r) equals B(n,r) for all valid n and r -/
theorem A_equals_B (n r : ℕ) (h : r ≤ n) : A n r = B n r := by
  sorry

end NUMINAMATH_CALUDE_A_equals_B_l4084_408411


namespace NUMINAMATH_CALUDE_plane_flight_distance_l4084_408461

/-- Given a plane that flies with and against the wind, prove the distance flown against the wind -/
theorem plane_flight_distance 
  (distance_with_wind : ℝ) 
  (wind_speed : ℝ) 
  (plane_speed : ℝ) 
  (h1 : distance_with_wind = 420) 
  (h2 : wind_speed = 23) 
  (h3 : plane_speed = 253) : 
  (distance_with_wind * (plane_speed - wind_speed)) / (plane_speed + wind_speed) = 350 := by
  sorry

end NUMINAMATH_CALUDE_plane_flight_distance_l4084_408461


namespace NUMINAMATH_CALUDE_product_mod_six_l4084_408472

theorem product_mod_six : (2015 * 2016 * 2017 * 2018) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_six_l4084_408472


namespace NUMINAMATH_CALUDE_zeros_before_last_digit_2009_pow_2011_l4084_408473

theorem zeros_before_last_digit_2009_pow_2011 :
  ∃ n : ℕ, n > 0 ∧ (2009^2011 % 10^(n+1)) / 10^n = 0 ∧ (2009^2011 % 10^n) / 10^(n-1) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_last_digit_2009_pow_2011_l4084_408473


namespace NUMINAMATH_CALUDE_decagon_ratio_l4084_408414

/-- A decagon made up of unit squares with specific properties -/
structure Decagon where
  /-- The total number of unit squares in the decagon -/
  num_squares : ℕ
  /-- The total area of the decagon in square units -/
  total_area : ℝ
  /-- LZ is a line segment intersecting the left and right vertices of the decagon -/
  lz : ℝ × ℝ
  /-- XZ is a segment from LZ to a vertex -/
  xz : ℝ
  /-- ZY is a segment from LZ to another vertex -/
  zy : ℝ
  /-- The number of unit squares is 12 -/
  h_num_squares : num_squares = 12
  /-- The total area is 12 square units -/
  h_total_area : total_area = 12
  /-- LZ bisects the area of the decagon -/
  h_bisects : lz.1 = total_area / 2

/-- The ratio of XZ to ZY is 1 -/
theorem decagon_ratio (d : Decagon) : d.xz / d.zy = 1 := by
  sorry


end NUMINAMATH_CALUDE_decagon_ratio_l4084_408414


namespace NUMINAMATH_CALUDE_area_of_square_II_l4084_408419

/-- Given a square I with diagonal 3(a+b), where a and b are positive real numbers,
    the area of a square II that is three times the area of square I is equal to 27(a+b)^2/2. -/
theorem area_of_square_II (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let diagonal_I := 3 * (a + b)
  let area_I := (diagonal_I ^ 2) / 2
  let area_II := 3 * area_I
  area_II = 27 * (a + b)^2 / 2 := by sorry

end NUMINAMATH_CALUDE_area_of_square_II_l4084_408419


namespace NUMINAMATH_CALUDE_triangle_DEF_is_right_angled_and_isosceles_l4084_408403

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the properties of the given triangles
def is_midpoint (F : Point) (B C : Point) : Prop :=
  F.x = (B.x + C.x) / 2 ∧ F.y = (B.y + C.y) / 2

def is_isosceles_right_triangle (A B D : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = (A.x - D.x)^2 + (A.y - D.y)^2 ∧
  (A.x - D.x) * (B.x - D.x) + (A.y - D.y) * (B.y - D.y) = 0

-- Define the theorem
theorem triangle_DEF_is_right_angled_and_isosceles 
  (ABC : Triangle) 
  (F D E : Point) 
  (h1 : is_midpoint F ABC.B ABC.C)
  (h2 : is_isosceles_right_triangle ABC.A ABC.B D)
  (h3 : is_isosceles_right_triangle ABC.A ABC.C E) :
  is_isosceles_right_triangle D E F := by
  sorry

end NUMINAMATH_CALUDE_triangle_DEF_is_right_angled_and_isosceles_l4084_408403


namespace NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l4084_408445

/-- The type representing the possible outcomes of rolling a standard 6-sided die. -/
inductive Die : Type
  | one | two | three | four | five | six

/-- The function that returns the numeric value of a die roll. -/
def dieValue : Die → Nat
  | Die.one => 1
  | Die.two => 2
  | Die.three => 3
  | Die.four => 4
  | Die.five => 5
  | Die.six => 6

/-- The type representing the outcome of rolling two dice. -/
def TwoDiceRoll : Type := Die × Die

/-- The function that calculates the sum of two dice rolls. -/
def rollSum (roll : TwoDiceRoll) : Nat :=
  dieValue roll.1 + dieValue roll.2

/-- The predicate that checks if a number is a multiple of 3. -/
def isMultipleOfThree (n : Nat) : Prop :=
  ∃ k, n = 3 * k

/-- The set of all possible outcomes when rolling two dice. -/
def allOutcomes : Finset TwoDiceRoll :=
  sorry

/-- The set of outcomes where the sum is a multiple of 3. -/
def favorableOutcomes : Finset TwoDiceRoll :=
  sorry

theorem probability_sum_multiple_of_three :
  (favorableOutcomes.card : ℚ) / (allOutcomes.card : ℚ) = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l4084_408445


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l4084_408497

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 30 / 3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l4084_408497


namespace NUMINAMATH_CALUDE_max_area_rectangular_garden_l4084_408439

/-- The maximum area of a rectangular garden with integer side lengths and a perimeter of 150 feet. -/
theorem max_area_rectangular_garden : ∃ (l w : ℕ), 
  (2 * l + 2 * w = 150) ∧ 
  (∀ (a b : ℕ), (2 * a + 2 * b = 150) → (a * b ≤ l * w)) ∧
  (l * w = 1406) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_garden_l4084_408439


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l4084_408456

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (orig_width orig_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := orig_width * enlarge_factor
  let enlarged_height := orig_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- Theorem stating that for the given picture dimensions and specifications, 10 feet of framing is needed. -/
theorem framing_for_enlarged_picture :
  min_framing_feet 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l4084_408456


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l4084_408455

def set_A (a : ℝ) : Set ℝ := {a^2, a+1, -1}
def set_B (a : ℝ) : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}

theorem intersection_implies_a_zero (a : ℝ) :
  set_A a ∩ set_B a = {-1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l4084_408455


namespace NUMINAMATH_CALUDE_range_of_m_l4084_408467

/-- The condition p -/
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0

/-- The condition q -/
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

/-- The statement that "not p" is sufficient but not necessary for "not q" -/
def not_p_suff_not_nec_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

/-- The main theorem -/
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ not_p_suff_not_nec_not_q m) ↔ (m > 0 ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l4084_408467


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l4084_408482

-- Define the polynomial function g(x)
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- Theorem statement
theorem polynomial_value_theorem (p q r s : ℝ) :
  g p q r s (-1) = 4 → 6*p - 3*q + r - 2*s = -24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l4084_408482


namespace NUMINAMATH_CALUDE_power_negative_cube_squared_l4084_408433

theorem power_negative_cube_squared (a : ℝ) (n : ℤ) : (-a^(3*n))^2 = a^(6*n) := by
  sorry

end NUMINAMATH_CALUDE_power_negative_cube_squared_l4084_408433


namespace NUMINAMATH_CALUDE_sum_greater_than_one_l4084_408415

theorem sum_greater_than_one
  (a b c d : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (hac : a > c)
  (hbd : b < d)
  (h1 : a + Real.sqrt b ≥ c + Real.sqrt d)
  (h2 : Real.sqrt a + b ≤ Real.sqrt c + d) :
  a + b + c + d > 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_greater_than_one_l4084_408415


namespace NUMINAMATH_CALUDE_polynomial_value_equals_one_l4084_408475

theorem polynomial_value_equals_one (x y : ℝ) (h : x + y = -1) :
  x^4 + 5*x^3*y + x^2*y + 8*x^2*y^2 + x*y^2 + 5*x*y^3 + y^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equals_one_l4084_408475


namespace NUMINAMATH_CALUDE_triangle_properties_l4084_408490

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumradius
def circumradius (t : Triangle) : ℝ := sorry

-- Define the length of a side
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define an angle in a triangle
def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_properties (t : Triangle) :
  side_length t.A t.B = Real.sqrt 10 →
  side_length t.A t.C = Real.sqrt 2 →
  circumradius t = Real.sqrt 5 →
  angle t t.C < Real.pi / 2 →
  side_length t.B t.C = 4 ∧ angle t t.C = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l4084_408490


namespace NUMINAMATH_CALUDE_jung_age_l4084_408459

/-- Proves Jung's age given the ages of Li and Zhang and their relationships -/
theorem jung_age (li_age : ℕ) (zhang_age : ℕ) (jung_age : ℕ)
  (h1 : zhang_age = 2 * li_age)
  (h2 : li_age = 12)
  (h3 : jung_age = zhang_age + 2) :
  jung_age = 26 := by
sorry

end NUMINAMATH_CALUDE_jung_age_l4084_408459


namespace NUMINAMATH_CALUDE_bennys_work_days_l4084_408416

/-- Given that Benny worked 3 hours a day for a total of 18 hours,
    prove that he worked for 6 days. -/
theorem bennys_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 → total_hours = 18 → days * hours_per_day = total_hours → days = 6 := by
  sorry


end NUMINAMATH_CALUDE_bennys_work_days_l4084_408416


namespace NUMINAMATH_CALUDE_a_plays_d_on_third_day_l4084_408417

-- Define the players
inductive Player : Type
| A : Player
| B : Player
| C : Player
| D : Player

-- Define a match as a pair of players
def Match := Player × Player

-- Define the schedule as a function from day to pair of matches
def Schedule := Nat → Match × Match

-- Define the condition that each player plays against each other exactly once
def playsAgainstEachOther (s : Schedule) : Prop :=
  ∀ p1 p2 : Player, p1 ≠ p2 → ∃ d : Nat, (s d).1 = (p1, p2) ∨ (s d).1 = (p2, p1) ∨ (s d).2 = (p1, p2) ∨ (s d).2 = (p2, p1)

-- Define the condition that each player plays only one match per day
def oneMatchPerDay (s : Schedule) : Prop :=
  ∀ d : Nat, ∀ p : Player, 
    ((s d).1.1 = p ∨ (s d).1.2 = p) → ((s d).2.1 ≠ p ∧ (s d).2.2 ≠ p)

-- Define the given conditions for the first two days
def givenConditions (s : Schedule) : Prop :=
  (s 1).1 = (Player.A, Player.C) ∨ (s 1).1 = (Player.C, Player.A) ∨ 
  (s 1).2 = (Player.A, Player.C) ∨ (s 1).2 = (Player.C, Player.A) ∧
  (s 2).1 = (Player.C, Player.D) ∨ (s 2).1 = (Player.D, Player.C) ∨ 
  (s 2).2 = (Player.C, Player.D) ∨ (s 2).2 = (Player.D, Player.C)

-- Theorem statement
theorem a_plays_d_on_third_day (s : Schedule) 
  (h1 : playsAgainstEachOther s) 
  (h2 : oneMatchPerDay s) 
  (h3 : givenConditions s) : 
  (s 3).1 = (Player.A, Player.D) ∨ (s 3).1 = (Player.D, Player.A) ∨ 
  (s 3).2 = (Player.A, Player.D) ∨ (s 3).2 = (Player.D, Player.A) :=
sorry

end NUMINAMATH_CALUDE_a_plays_d_on_third_day_l4084_408417


namespace NUMINAMATH_CALUDE_point_movement_l4084_408425

/-- Given a point A with coordinates (-3, -2), moving it up by 3 units
    and then left by 2 units results in a point B with coordinates (-5, 1). -/
theorem point_movement :
  let A : ℝ × ℝ := (-3, -2)
  let up_movement : ℝ := 3
  let left_movement : ℝ := 2
  let B : ℝ × ℝ := (A.1 - left_movement, A.2 + up_movement)
  B = (-5, 1) := by sorry

end NUMINAMATH_CALUDE_point_movement_l4084_408425


namespace NUMINAMATH_CALUDE_square_not_always_positive_l4084_408435

theorem square_not_always_positive : ¬ (∀ x : ℝ, x^2 > 0) := by
  sorry

end NUMINAMATH_CALUDE_square_not_always_positive_l4084_408435
