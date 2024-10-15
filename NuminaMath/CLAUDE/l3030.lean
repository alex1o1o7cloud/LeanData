import Mathlib

namespace NUMINAMATH_CALUDE_max_side_length_l3030_303007

/-- A triangle with three different integer side lengths and perimeter 30 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 30

/-- The maximum length of any side in a triangle with perimeter 30 and different integer side lengths is 14 -/
theorem max_side_length (t : Triangle) : t.a ≤ 14 ∧ t.b ≤ 14 ∧ t.c ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_max_side_length_l3030_303007


namespace NUMINAMATH_CALUDE_derivative_f_at_2_l3030_303041

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_f_at_2 : 
  deriv f 2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_2_l3030_303041


namespace NUMINAMATH_CALUDE_graph_connected_probability_l3030_303076

def n : ℕ := 20
def edges_removed : ℕ := 35

theorem graph_connected_probability :
  let total_edges := n * (n - 1) / 2
  let remaining_edges := total_edges - edges_removed
  let prob_disconnected := n * (Nat.choose remaining_edges (remaining_edges - n + 1)) / (Nat.choose total_edges edges_removed)
  (1 : ℚ) - prob_disconnected = 1 - (20 * Nat.choose 171 16) / Nat.choose 190 35 := by
  sorry

end NUMINAMATH_CALUDE_graph_connected_probability_l3030_303076


namespace NUMINAMATH_CALUDE_complete_square_k_value_l3030_303098

/-- A quadratic expression can be factored using the complete square formula if and only if
    it can be written in the form (x + a)^2 for some real number a. --/
def is_complete_square (k : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + k*x + 9 = (x + a)^2

/-- If x^2 + kx + 9 can be factored using the complete square formula,
    then k = 6 or k = -6. --/
theorem complete_square_k_value (k : ℝ) :
  is_complete_square k → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_k_value_l3030_303098


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l3030_303081

/-- Parabola E: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P -/
def P : ℝ × ℝ := (7, 3)

/-- Line with slope k passing through point P -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = k * (x - P.1)

/-- Line with slope 2/3 passing through point A -/
def line_AC (A : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - A.2 = (2/3) * (x - A.1)

theorem parabola_fixed_point :
  ∀ (k : ℝ) (A B C : ℝ × ℝ),
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola C.1 C.2 →
  line_through_P k A.1 A.2 →
  line_through_P k B.1 B.2 →
  line_AC A C.1 C.2 →
  ∃ (m : ℝ), y - C.2 = m * (x - C.1) ∧ y - B.2 = m * (x - B.1) →
  y - 3 = m * (x + 5/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l3030_303081


namespace NUMINAMATH_CALUDE_min_people_to_ask_l3030_303045

theorem min_people_to_ask (knights : ℕ) (civilians : ℕ) : 
  knights = 50 → civilians = 15 → 
  ∃ (n : ℕ), n > civilians ∧ n - civilians ≤ knights ∧ 
  ∀ (m : ℕ), m < n → (m - civilians ≤ knights → m ≤ civilians) :=
sorry

end NUMINAMATH_CALUDE_min_people_to_ask_l3030_303045


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l3030_303037

def total_products : ℕ := 60
def sample_size : ℕ := 5

def systematic_sample (start : ℕ) : List ℕ :=
  List.range sample_size |>.map (λ i => start + i * (total_products / sample_size))

theorem correct_systematic_sample :
  systematic_sample 5 = [5, 17, 29, 41, 53] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l3030_303037


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3030_303043

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 1) :
  ((x^2 - 2*x) / (x^2 - 4*x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3030_303043


namespace NUMINAMATH_CALUDE_tom_climbing_time_l3030_303092

/-- Given that Elizabeth takes 30 minutes to climb a hill and Tom takes four times as long,
    prove that Tom's climbing time is 2 hours. -/
theorem tom_climbing_time :
  let elizabeth_time : ℕ := 30 -- Elizabeth's climbing time in minutes
  let tom_factor : ℕ := 4 -- Tom takes four times as long as Elizabeth
  let tom_time : ℕ := elizabeth_time * tom_factor -- Tom's climbing time in minutes
  tom_time / 60 = 2 -- Tom's climbing time in hours
:= by sorry

end NUMINAMATH_CALUDE_tom_climbing_time_l3030_303092


namespace NUMINAMATH_CALUDE_honey_nights_l3030_303073

/-- Represents the number of servings of honey per cup of tea -/
def servings_per_cup : ℕ := 1

/-- Represents the number of cups of tea Tabitha drinks before bed each night -/
def cups_per_night : ℕ := 2

/-- Represents the size of the honey container in ounces -/
def container_size : ℕ := 16

/-- Represents the number of servings of honey per ounce -/
def servings_per_ounce : ℕ := 6

/-- Theorem stating how many nights Tabitha can enjoy honey in her tea -/
theorem honey_nights : 
  (container_size * servings_per_ounce) / (servings_per_cup * cups_per_night) = 48 := by
  sorry

end NUMINAMATH_CALUDE_honey_nights_l3030_303073


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3030_303004

theorem complex_equation_solution (z : ℂ) : (1 - I) * z = 2 * I → z = -1 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3030_303004


namespace NUMINAMATH_CALUDE_apple_juice_production_l3030_303082

theorem apple_juice_production (total_production : ℝ) 
  (mixed_percentage : ℝ) (juice_percentage : ℝ) :
  total_production = 5.5 →
  mixed_percentage = 0.2 →
  juice_percentage = 0.5 →
  (1 - mixed_percentage) * juice_percentage * total_production = 2.2 := by
sorry

end NUMINAMATH_CALUDE_apple_juice_production_l3030_303082


namespace NUMINAMATH_CALUDE_saroj_current_age_l3030_303020

/-- Represents the age of a person at different points in time -/
structure PersonAge where
  sixYearsAgo : ℕ
  current : ℕ
  fourYearsHence : ℕ

/-- The problem statement -/
theorem saroj_current_age 
  (vimal saroj : PersonAge)
  (h1 : vimal.sixYearsAgo * 5 = saroj.sixYearsAgo * 6)
  (h2 : vimal.fourYearsHence * 10 = saroj.fourYearsHence * 11)
  (h3 : vimal.current = vimal.sixYearsAgo + 6)
  (h4 : saroj.current = saroj.sixYearsAgo + 6)
  (h5 : vimal.fourYearsHence = vimal.current + 4)
  (h6 : saroj.fourYearsHence = saroj.current + 4)
  : saroj.current = 16 := by
  sorry

end NUMINAMATH_CALUDE_saroj_current_age_l3030_303020


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l3030_303026

theorem mixed_number_calculation : 
  26 * (2 + 4/7 - (3 + 1/3)) + (3 + 1/5 + 2 + 3/7) = -(14 + 223/735) := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l3030_303026


namespace NUMINAMATH_CALUDE_gratuity_calculation_l3030_303064

def dish_price_1 : ℝ := 10
def dish_price_2 : ℝ := 13
def dish_price_3 : ℝ := 17
def tip_percentage : ℝ := 0.1

theorem gratuity_calculation : 
  (dish_price_1 + dish_price_2 + dish_price_3) * tip_percentage = 4 := by
sorry

end NUMINAMATH_CALUDE_gratuity_calculation_l3030_303064


namespace NUMINAMATH_CALUDE_project_completion_time_l3030_303060

theorem project_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let time_together := (a * b) / (a + b)
  time_together > 0 ∧ 
  (1 / a + 1 / b) * time_together = 1 := by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l3030_303060


namespace NUMINAMATH_CALUDE_chess_problem_l3030_303080

/-- Represents a chess piece (rook or king) -/
inductive Piece
| Rook
| King

/-- Represents a position on the chess board -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the state of the chess board -/
structure ChessBoard :=
  (size : Nat)
  (whiteRooks : List Position)
  (blackKing : Position)

/-- Checks if a position is in check -/
def isInCheck (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can get into check after some finite number of moves -/
def canGetIntoCheck (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can always be in check after its move (excluding initial moves) -/
def canAlwaysBeInCheckAfterMove (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can always be in check (even after white's move, excluding initial moves) -/
def canAlwaysBeInCheck (board : ChessBoard) : Bool :=
  sorry

theorem chess_problem (board : ChessBoard) 
  (h1 : board.size = 1000) 
  (h2 : board.whiteRooks.length = 499) :
  (canGetIntoCheck board = true) ∧ 
  (canAlwaysBeInCheckAfterMove board = false) ∧
  (canAlwaysBeInCheck board = false) :=
  sorry

end NUMINAMATH_CALUDE_chess_problem_l3030_303080


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3030_303088

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ z : ℂ, z = Complex.mk (a^2 - a - 2) (a + 1) ∧ z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3030_303088


namespace NUMINAMATH_CALUDE_toy_store_fraction_l3030_303028

theorem toy_store_fraction (weekly_allowance : ℚ) 
  (arcade_fraction : ℚ) (candy_store_amount : ℚ) :
  weekly_allowance = 3 →
  arcade_fraction = 2/5 →
  candy_store_amount = 6/5 →
  let remaining_after_arcade := weekly_allowance - arcade_fraction * weekly_allowance
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by
sorry

end NUMINAMATH_CALUDE_toy_store_fraction_l3030_303028


namespace NUMINAMATH_CALUDE_area_of_M_l3030_303054

-- Define the region M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (|y| + |4 - y| ≤ 4) ∧
               ((y^2 + x - 4*y + 1) / (2*y + x - 7) ≤ 0)}

-- State the theorem
theorem area_of_M : MeasureTheory.volume M = 8 := by
  sorry

end NUMINAMATH_CALUDE_area_of_M_l3030_303054


namespace NUMINAMATH_CALUDE_equation_equivalence_l3030_303066

theorem equation_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : 2*b - a ≠ 0) :
  (a + 2*b) / a = b / (2*b - a) ↔ 
  (a = -b * ((1 + Real.sqrt 17) / 2) ∨ a = -b * ((1 - Real.sqrt 17) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3030_303066


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l3030_303034

theorem abc_divisibility_problem (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b ∣ a^3 + b^3 + c^3) ∧
  (b^2 * c ∣ a^3 + b^3 + c^3) ∧
  (c^2 * a ∣ a^3 + b^3 + c^3) →
  ∃ k : ℕ, a = k ∧ b = k ∧ c = k := by
sorry

end NUMINAMATH_CALUDE_abc_divisibility_problem_l3030_303034


namespace NUMINAMATH_CALUDE_add_point_four_to_fifty_six_point_seven_l3030_303046

theorem add_point_four_to_fifty_six_point_seven :
  0.4 + 56.7 = 57.1 := by sorry

end NUMINAMATH_CALUDE_add_point_four_to_fifty_six_point_seven_l3030_303046


namespace NUMINAMATH_CALUDE_midpoint_region_area_l3030_303086

/-- A regular hexagon with area 16 -/
structure RegularHexagon :=
  (area : ℝ)
  (is_regular : Bool)
  (area_eq_16 : area = 16)

/-- The midpoint of a side of the hexagon -/
structure Midpoint :=
  (hexagon : RegularHexagon)

/-- A region formed by connecting four consecutive midpoints -/
structure MidpointRegion :=
  (hexagon : RegularHexagon)
  (midpoints : Fin 4 → Midpoint)
  (consecutive : ∀ i : Fin 3, (midpoints i).hexagon = (midpoints (i + 1)).hexagon)

/-- The theorem statement -/
theorem midpoint_region_area (region : MidpointRegion) : 
  (region.hexagon.area / 2) = 8 :=
sorry

end NUMINAMATH_CALUDE_midpoint_region_area_l3030_303086


namespace NUMINAMATH_CALUDE_tangent_line_implies_sum_l3030_303040

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := (a * x - 1) * Real.log x + b

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := a * Real.log x + (a * x - 1) / x

theorem tangent_line_implies_sum (a b : ℝ) : 
  (∀ x, f_derivative a x = f a b x) →  -- f_derivative is the derivative of f
  f_derivative a 1 = -a →              -- Slope condition at x = 1
  f a b 1 = -a + 1 →                   -- Point condition at x = 1
  a + b = 1 := by sorry

end

end NUMINAMATH_CALUDE_tangent_line_implies_sum_l3030_303040


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3030_303099

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem sum_of_coefficients (a b : ℝ) :
  (A ∩ B a b = {x : ℝ | 0 < x ∧ x ≤ 2}) ∧
  (A ∪ B a b = {x : ℝ | x > -2}) →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3030_303099


namespace NUMINAMATH_CALUDE_businesspeople_neither_coffee_nor_tea_l3030_303023

theorem businesspeople_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 35)
  (h2 : coffee = 18)
  (h3 : tea = 15)
  (h4 : both = 6) :
  total - (coffee + tea - both) = 8 := by
sorry

end NUMINAMATH_CALUDE_businesspeople_neither_coffee_nor_tea_l3030_303023


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l3030_303050

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 26 → x - y = 8 → x * y = 153 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l3030_303050


namespace NUMINAMATH_CALUDE_polynomial_square_l3030_303029

theorem polynomial_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x, x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_square_l3030_303029


namespace NUMINAMATH_CALUDE_factors_of_x_fourth_plus_81_l3030_303002

theorem factors_of_x_fourth_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factors_of_x_fourth_plus_81_l3030_303002


namespace NUMINAMATH_CALUDE_expression_value_l3030_303058

theorem expression_value : (2 * Real.sqrt 2) ^ (2/3) * (0.1)⁻¹ - Real.log 2 / Real.log 10 - Real.log 5 / Real.log 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3030_303058


namespace NUMINAMATH_CALUDE_product_equals_20152015_l3030_303083

theorem product_equals_20152015 : 5 * 13 * 31 * 73 * 137 = 20152015 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_20152015_l3030_303083


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l3030_303084

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = -16286/16384 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l3030_303084


namespace NUMINAMATH_CALUDE_range_of_a_theorem_l3030_303035

-- Define the set A (condition p)
def A (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a^2 + 1}

-- Define the set B (condition q)
def B (a : ℝ) : Set ℝ := {x | x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0}

-- Define the range of a
def RangeOfA : Set ℝ := {a | 1 ≤ a ∧ a ≤ 3 ∨ a = -1}

-- Statement of the theorem
theorem range_of_a_theorem :
  (∀ a : ℝ, A a ⊆ B a) → 
  (∀ a : ℝ, a ∈ RangeOfA ↔ (A a ⊆ B a)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_theorem_l3030_303035


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l3030_303070

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat

/-- Axiom: The month has exactly 5 Fridays -/
axiom five_fridays (m : Month) : m.fridayCount = 5

/-- Axiom: The first day of the month is not a Friday -/
axiom first_not_friday (m : Month) : m.firstDay ≠ DayOfWeek.Friday

/-- Axiom: The last day of the month is not a Friday -/
axiom last_not_friday (m : Month) : m.lastDay ≠ DayOfWeek.Friday

/-- Function to get the day of week for a given day number -/
def getDayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The 12th day of the month is a Monday -/
theorem twelfth_day_is_monday (m : Month) :
  getDayOfWeek m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l3030_303070


namespace NUMINAMATH_CALUDE_man_walking_time_l3030_303067

theorem man_walking_time (usual_time : ℝ) (reduced_time : ℝ) : 
  reduced_time = usual_time + 24 →
  (1 : ℝ) / 0.4 = reduced_time / usual_time →
  usual_time = 16 := by
sorry

end NUMINAMATH_CALUDE_man_walking_time_l3030_303067


namespace NUMINAMATH_CALUDE_equation_value_l3030_303049

theorem equation_value (x y : ℝ) (eq1 : 2*x + y = 8) (eq2 : x + 2*y = 10) :
  8*x^2 + 10*x*y + 8*y^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_equation_value_l3030_303049


namespace NUMINAMATH_CALUDE_hannah_late_times_l3030_303000

/-- Represents the number of times Hannah was late to work in a week. -/
def times_late (hourly_rate : ℕ) (hours_worked : ℕ) (dock_amount : ℕ) (actual_pay : ℕ) : ℕ :=
  (hourly_rate * hours_worked - actual_pay) / dock_amount

/-- Theorem stating that Hannah was late 3 times given the problem conditions. -/
theorem hannah_late_times :
  times_late 30 18 5 525 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hannah_late_times_l3030_303000


namespace NUMINAMATH_CALUDE_marbles_distribution_l3030_303051

/-- Given a total number of marbles and a number of boxes, calculate the number of marbles per box -/
def marblesPerBox (totalMarbles : ℕ) (numBoxes : ℕ) : ℕ :=
  totalMarbles / numBoxes

theorem marbles_distribution (totalMarbles : ℕ) (numBoxes : ℕ) 
  (h1 : totalMarbles = 18) (h2 : numBoxes = 3) :
  marblesPerBox totalMarbles numBoxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3030_303051


namespace NUMINAMATH_CALUDE_uncle_dave_nieces_l3030_303027

theorem uncle_dave_nieces (total_sandwiches : ℕ) (sandwiches_per_niece : ℕ) (h1 : total_sandwiches = 143) (h2 : sandwiches_per_niece = 13) :
  total_sandwiches / sandwiches_per_niece = 11 := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_nieces_l3030_303027


namespace NUMINAMATH_CALUDE_pythagorean_linear_function_l3030_303095

theorem pythagorean_linear_function (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem
  ((-a/c + b/c)^2 = 1/3) →  -- Point (-1, √3/3) lies on y = (a/c)x + (b/c)
  (a * b / 2 = 4) →  -- Area of triangle is 4
  c = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_pythagorean_linear_function_l3030_303095


namespace NUMINAMATH_CALUDE_total_fat_served_l3030_303005

/-- The amount of fat in ounces for each type of fish --/
def herring_fat : ℕ := 40
def eel_fat : ℕ := 20
def pike_fat : ℕ := eel_fat + 10
def salmon_fat : ℕ := 35
def halibut_fat : ℕ := 50

/-- The number of each type of fish served --/
def herring_count : ℕ := 40
def eel_count : ℕ := 30
def pike_count : ℕ := 25
def salmon_count : ℕ := 20
def halibut_count : ℕ := 15

/-- The total amount of fat served --/
def total_fat : ℕ := 
  herring_fat * herring_count +
  eel_fat * eel_count +
  pike_fat * pike_count +
  salmon_fat * salmon_count +
  halibut_fat * halibut_count

theorem total_fat_served : total_fat = 4400 := by
  sorry

end NUMINAMATH_CALUDE_total_fat_served_l3030_303005


namespace NUMINAMATH_CALUDE_slower_train_speed_l3030_303074

/-- Calculates the speed of the slower train given the conditions of two trains moving in the same direction. -/
theorem slower_train_speed
  (faster_train_speed : ℝ)
  (faster_train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_train_speed = 72)
  (h2 : faster_train_length = 70)
  (h3 : crossing_time = 7)
  : ∃ (slower_train_speed : ℝ), slower_train_speed = 36 :=
by
  sorry

#check slower_train_speed

end NUMINAMATH_CALUDE_slower_train_speed_l3030_303074


namespace NUMINAMATH_CALUDE_tangent_line_smallest_slope_l3030_303036

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem tangent_line_smallest_slope :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, f x = y → a*x + b*y + c = 0) ∧ 
    (∀ x₀ y₀ : ℝ, f x₀ = y₀ → ∀ m : ℝ, (∃ x y : ℝ, f x = y ∧ m = f' x) → m ≥ a) ∧
    a = 3 ∧ b = -1 ∧ c = -11 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_smallest_slope_l3030_303036


namespace NUMINAMATH_CALUDE_factor_decomposition_l3030_303078

theorem factor_decomposition (a b : Int) : 
  a * b = 96 → a^2 + b^2 = 208 → 
  ((a = 8 ∧ b = 12) ∨ (a = -8 ∧ b = -12) ∨ (a = 12 ∧ b = 8) ∨ (a = -12 ∧ b = -8)) :=
by sorry

end NUMINAMATH_CALUDE_factor_decomposition_l3030_303078


namespace NUMINAMATH_CALUDE_star_value_l3030_303048

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

/-- Theorem: If a + b = 15 and a * b = 36, then a * b = 5/12 -/
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum : a + b = 15) (product : a * b = 36) : 
  star a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l3030_303048


namespace NUMINAMATH_CALUDE_pet_store_count_l3030_303079

/-- Represents the count of animals in a pet store -/
structure PetStore :=
  (birds : ℕ)
  (puppies : ℕ)
  (cats : ℕ)
  (spiders : ℕ)

/-- Calculates the total number of animals in the pet store -/
def totalAnimals (store : PetStore) : ℕ :=
  store.birds + store.puppies + store.cats + store.spiders

/-- Represents the changes in animal counts -/
structure Changes :=
  (birdsSold : ℕ)
  (puppiesAdopted : ℕ)
  (spidersLoose : ℕ)

/-- Applies changes to the pet store counts -/
def applyChanges (store : PetStore) (changes : Changes) : PetStore :=
  { birds := store.birds - changes.birdsSold,
    puppies := store.puppies - changes.puppiesAdopted,
    cats := store.cats,
    spiders := store.spiders - changes.spidersLoose }

theorem pet_store_count : 
  let initialStore : PetStore := { birds := 12, puppies := 9, cats := 5, spiders := 15 }
  let changes : Changes := { birdsSold := 6, puppiesAdopted := 3, spidersLoose := 7 }
  let finalStore := applyChanges initialStore changes
  totalAnimals finalStore = 25 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_count_l3030_303079


namespace NUMINAMATH_CALUDE_janet_ride_count_l3030_303003

theorem janet_ride_count (roller_coaster_tickets : ℕ) (giant_slide_tickets : ℕ) 
  (roller_coaster_rides : ℕ) (total_tickets : ℕ) :
  roller_coaster_tickets = 5 →
  giant_slide_tickets = 3 →
  roller_coaster_rides = 7 →
  total_tickets = 47 →
  ∃ (giant_slide_rides : ℕ), 
    roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides = total_tickets ∧
    giant_slide_rides = 4 := by
  sorry

end NUMINAMATH_CALUDE_janet_ride_count_l3030_303003


namespace NUMINAMATH_CALUDE_spinner_probability_l3030_303038

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 5/12 → p_A + p_B + p_C + p_D = 1 → p_D = 0 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3030_303038


namespace NUMINAMATH_CALUDE_brian_initial_cards_l3030_303015

def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def cards_left : ℕ := 17

theorem brian_initial_cards : initial_cards = cards_taken + cards_left := by
  sorry

end NUMINAMATH_CALUDE_brian_initial_cards_l3030_303015


namespace NUMINAMATH_CALUDE_product_of_solutions_l3030_303052

theorem product_of_solutions (x₁ x₂ : ℝ) 
  (h₁ : x₁ * Real.exp x₁ = Real.exp 2)
  (h₂ : x₂ * Real.log x₂ = Real.exp 2) :
  x₁ * x₂ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l3030_303052


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3030_303071

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- The seventh term of a geometric sequence with first term -4 and second term 8 is -256 -/
theorem seventh_term_of_geometric_sequence :
  let a₁ : ℝ := -4
  let a₂ : ℝ := 8
  let r : ℝ := a₂ / a₁
  geometric_sequence a₁ r 7 = -256 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l3030_303071


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3030_303039

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3030_303039


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l3030_303019

theorem cube_volume_surface_area (x : ℝ) :
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) →
  x = Real.sqrt 3 / 72 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l3030_303019


namespace NUMINAMATH_CALUDE_chord_slope_of_ellipse_l3030_303009

/-- Given an ellipse and a chord bisected by a point, prove the slope of the chord -/
theorem chord_slope_of_ellipse (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 4) →         -- Midpoint x-coordinate is 4
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  (y₁ - y₂) / (x₁ - x₂) = -1/2  -- Slope of the chord is -1/2
:= by sorry

end NUMINAMATH_CALUDE_chord_slope_of_ellipse_l3030_303009


namespace NUMINAMATH_CALUDE_count_points_is_ten_l3030_303087

def M : Finset Int := {1, -2, 3}
def N : Finset Int := {-4, 5, 6, -7}

def is_in_third_or_fourth_quadrant (p : Int × Int) : Bool :=
  p.2 < 0

def count_points : Nat :=
  (M.card * (N.filter (· < 0)).card) + (N.card * (M.filter (· < 0)).card)

theorem count_points_is_ten :
  count_points = 10 := by sorry

end NUMINAMATH_CALUDE_count_points_is_ten_l3030_303087


namespace NUMINAMATH_CALUDE_total_yards_run_l3030_303010

/-- Calculates the total yards run by three athletes given their individual performances -/
theorem total_yards_run (athlete1_yards athlete2_yards athlete3_avg_yards : ℕ) 
  (games : ℕ) (h1 : games = 4) (h2 : athlete1_yards = 18) (h3 : athlete2_yards = 22) 
  (h4 : athlete3_avg_yards = 11) : 
  athlete1_yards * games + athlete2_yards * games + athlete3_avg_yards * games = 204 :=
by sorry

end NUMINAMATH_CALUDE_total_yards_run_l3030_303010


namespace NUMINAMATH_CALUDE_election_winner_votes_l3030_303031

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 54/100 →
  vote_difference = 288 →
  ⌊(winner_percentage : ℝ) * total_votes⌋ - ⌊((1 - winner_percentage) : ℝ) * total_votes⌋ = vote_difference →
  ⌊(winner_percentage : ℝ) * total_votes⌋ = 1944 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3030_303031


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_count_l3030_303006

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_ratio_count
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 8 = 36)
  (h_sum : a 3 + a 7 = 15) :
  ∃ (S : Finset ℝ), (∀ q ∈ S, ∃ (a : ℕ → ℝ), is_geometric_sequence a ∧ a 2 * a 8 = 36 ∧ a 3 + a 7 = 15) ∧ S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_count_l3030_303006


namespace NUMINAMATH_CALUDE_hexagons_in_nth_ring_hexagons_in_100th_ring_l3030_303059

/-- The number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_ring (n : ℕ) : ℕ := 6 * n

/-- Theorem: The number of hexagons in the nth ring is 6n -/
theorem hexagons_in_nth_ring (n : ℕ) :
  hexagons_in_ring n = 6 * n := by sorry

/-- Corollary: The number of hexagons in the 100th ring is 600 -/
theorem hexagons_in_100th_ring :
  hexagons_in_ring 100 = 600 := by sorry

end NUMINAMATH_CALUDE_hexagons_in_nth_ring_hexagons_in_100th_ring_l3030_303059


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3030_303047

/-- Two arithmetic sequences and their sums -/
structure ArithmeticSequences where
  a : ℕ → ℚ
  b : ℕ → ℚ
  S : ℕ → ℚ
  T : ℕ → ℚ

/-- The main theorem -/
theorem arithmetic_sequences_ratio 
  (seq : ArithmeticSequences)
  (h : ∀ n, seq.S n / seq.T n = (3 * n - 1) / (2 * n + 3)) :
  seq.a 7 / seq.b 7 = 38 / 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3030_303047


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l3030_303063

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def M : ℚ := 2 * 19 * factorial 19 * (
  1 / (factorial 3 * factorial 18) +
  1 / (factorial 4 * factorial 17) +
  1 / (factorial 5 * factorial 16) +
  1 / (factorial 6 * factorial 15) +
  1 / (factorial 7 * factorial 14) +
  1 / (factorial 8 * factorial 13) +
  1 / (factorial 9 * factorial 12) +
  1 / (factorial 10 * factorial 11)
)

theorem greatest_integer_less_than_M_over_100 :
  ⌊M / 100⌋ = 499 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_M_over_100_l3030_303063


namespace NUMINAMATH_CALUDE_ivan_work_and_charity_l3030_303033

/-- Represents Ivan Petrovich's daily work and financial situation --/
structure IvanPetrovich where
  workDays : ℕ -- number of working days per month
  sleepHours : ℕ -- hours of sleep per day
  workHours : ℝ -- hours of work per day
  hobbyRatio : ℝ -- ratio of hobby time to work time
  hourlyRate : ℝ -- rubles earned per hour of work
  rentalIncome : ℝ -- monthly rental income in rubles
  charityRatio : ℝ -- ratio of charity donation to rest hours
  monthlyExpenses : ℝ -- monthly expenses excluding charity in rubles

/-- Theorem stating Ivan Petrovich's work hours and charity donation --/
theorem ivan_work_and_charity 
  (ivan : IvanPetrovich)
  (h1 : ivan.workDays = 21)
  (h2 : ivan.sleepHours = 8)
  (h3 : ivan.hobbyRatio = 2)
  (h4 : ivan.hourlyRate = 3000)
  (h5 : ivan.rentalIncome = 14000)
  (h6 : ivan.charityRatio = 1/3)
  (h7 : ivan.monthlyExpenses = 70000)
  (h8 : 24 = ivan.sleepHours + ivan.workHours + ivan.hobbyRatio * ivan.workHours + (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)))
  (h9 : ivan.workDays * (ivan.hourlyRate * ivan.workHours + ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000) + ivan.rentalIncome = ivan.monthlyExpenses + ivan.workDays * ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000) :
  ivan.workHours = 2 ∧ ivan.workDays * ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000 = 70000 := by
  sorry

end NUMINAMATH_CALUDE_ivan_work_and_charity_l3030_303033


namespace NUMINAMATH_CALUDE_arccos_cos_eq_three_halves_x_implies_x_zero_l3030_303093

theorem arccos_cos_eq_three_halves_x_implies_x_zero 
  (x : ℝ) 
  (h1 : -π ≤ x ∧ x ≤ π) 
  (h2 : Real.arccos (Real.cos x) = (3 * x) / 2) : 
  x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_three_halves_x_implies_x_zero_l3030_303093


namespace NUMINAMATH_CALUDE_pears_picked_total_l3030_303062

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 3

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 2

/-- The total number of pears picked -/
def total_pears : ℕ := keith_pears + jason_pears

theorem pears_picked_total :
  total_pears = 5 := by sorry

end NUMINAMATH_CALUDE_pears_picked_total_l3030_303062


namespace NUMINAMATH_CALUDE_lcm_count_l3030_303089

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (6^9) (Nat.lcm (9^9) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (6^9) (Nat.lcm (9^9) k) ≠ 18^18)) := by
  sorry

end NUMINAMATH_CALUDE_lcm_count_l3030_303089


namespace NUMINAMATH_CALUDE_pentagon_stack_exists_no_pentagon_stack_for_one_no_pentagon_stack_for_three_l3030_303042

/-- A regular pentagon with numbers from 1 to 5 at its vertices -/
def Pentagon : Type := Fin 5 → Fin 5

/-- A stack of pentagons -/
def PentagonStack : Type := List Pentagon

/-- The sum of numbers at a vertex in a stack of pentagons -/
def vertexSum (stack : PentagonStack) (vertex : Fin 5) : ℕ :=
  (stack.map (λ p => p vertex)).sum

/-- A predicate that checks if all vertex sums in a stack are equal -/
def allVertexSumsEqual (stack : PentagonStack) : Prop :=
  ∀ v1 v2 : Fin 5, vertexSum stack v1 = vertexSum stack v2

/-- Main theorem: For any natural number n ≠ 1 and n ≠ 3, there exists a valid pentagon stack of size n -/
theorem pentagon_stack_exists (n : ℕ) (h1 : n ≠ 1) (h3 : n ≠ 3) :
  ∃ (stack : PentagonStack), stack.length = n ∧ allVertexSumsEqual stack :=
sorry

/-- No valid pentagon stack exists for n = 1 -/
theorem no_pentagon_stack_for_one :
  ¬∃ (stack : PentagonStack), stack.length = 1 ∧ allVertexSumsEqual stack :=
sorry

/-- No valid pentagon stack exists for n = 3 -/
theorem no_pentagon_stack_for_three :
  ¬∃ (stack : PentagonStack), stack.length = 3 ∧ allVertexSumsEqual stack :=
sorry

end NUMINAMATH_CALUDE_pentagon_stack_exists_no_pentagon_stack_for_one_no_pentagon_stack_for_three_l3030_303042


namespace NUMINAMATH_CALUDE_solar_panel_height_P_l3030_303068

/-- Regular hexagon with side length 10 and pillars at vertices -/
structure SolarPanelSupport where
  -- Side length of the hexagon
  side_length : ℝ
  -- Heights of pillars at L, M, and N
  height_L : ℝ
  height_M : ℝ
  height_N : ℝ

/-- The height of the pillar at P in the solar panel support system -/
def height_P (s : SolarPanelSupport) : ℝ := sorry

/-- Theorem stating the height of pillar P given specific conditions -/
theorem solar_panel_height_P (s : SolarPanelSupport) 
  (h_side : s.side_length = 10)
  (h_L : s.height_L = 15)
  (h_M : s.height_M = 12)
  (h_N : s.height_N = 13) : 
  height_P s = 22 := by sorry

end NUMINAMATH_CALUDE_solar_panel_height_P_l3030_303068


namespace NUMINAMATH_CALUDE_sum_of_roots_l3030_303072

theorem sum_of_roots (c d : ℝ) 
  (hc : c^3 - 18*c^2 + 25*c - 75 = 0) 
  (hd : 9*d^3 - 72*d^2 - 345*d + 3060 = 0) : 
  c + d = 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3030_303072


namespace NUMINAMATH_CALUDE_expression_equals_one_l3030_303077

theorem expression_equals_one : 
  (105^2 - 8^2) / (80^2 - 13^2) * ((80 - 13) * (80 + 13)) / ((105 - 8) * (105 + 8)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3030_303077


namespace NUMINAMATH_CALUDE_product_congruence_l3030_303011

theorem product_congruence : 65 * 76 * 87 ≡ 5 [ZMOD 25] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l3030_303011


namespace NUMINAMATH_CALUDE_inequality_proof_l3030_303044

theorem inequality_proof (x₁ x₂ : ℝ) (h₁ : |x₁| ≤ 1) (h₂ : |x₂| ≤ 1) :
  Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) ≤ 2 * Real.sqrt (1 - ((x₁ + x₂) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3030_303044


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l3030_303030

/-- The sum of squares of the roots of x^2 - (m+1)x + (m-1) = 0 is minimized when m = 0 -/
theorem min_sum_squares_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ m => m^2 + 3
  let sum_squares := f m
  ∀ k : ℝ, f k ≥ f 0 := by sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l3030_303030


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l3030_303008

theorem sum_sqrt_inequality (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) 
  (sum_eq : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l3030_303008


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l3030_303024

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l3030_303024


namespace NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l3030_303001

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZeroes (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

theorem trailing_zeroes_sum_factorials :
  trailingZeroes (factorial 60 + factorial 120) = 14 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l3030_303001


namespace NUMINAMATH_CALUDE_eight_x_plus_y_value_l3030_303022

theorem eight_x_plus_y_value (x y z : ℝ) 
  (eq1 : x + 2*y - 3*z = 7) 
  (eq2 : 2*x - y + 2*z = 6) : 
  8*x + y = 32 := by sorry

end NUMINAMATH_CALUDE_eight_x_plus_y_value_l3030_303022


namespace NUMINAMATH_CALUDE_circle_area_is_one_l3030_303032

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (4 * (1 / (2 * Real.pi * r)) = 2 * r) → (Real.pi * r^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_is_one_l3030_303032


namespace NUMINAMATH_CALUDE_well_depth_equation_l3030_303053

theorem well_depth_equation (d : ℝ) (u : ℝ) (h : u = Real.sqrt d) : 
  d = 14 * (10 - d / 1200)^2 → 14 * u^2 + 1200 * u - 12000 * Real.sqrt 14 = 0 := by
  sorry

#check well_depth_equation

end NUMINAMATH_CALUDE_well_depth_equation_l3030_303053


namespace NUMINAMATH_CALUDE_divisible_integers_count_l3030_303065

-- Define the range of integers
def lower_bound : ℕ := 2000
def upper_bound : ℕ := 3000

-- Define the factors
def factor1 : ℕ := 30
def factor2 : ℕ := 45
def factor3 : ℕ := 75

-- Function to count integers in the range divisible by all factors
def count_divisible_integers : ℕ := sorry

-- Theorem statement
theorem divisible_integers_count : count_divisible_integers = 2 := by sorry

end NUMINAMATH_CALUDE_divisible_integers_count_l3030_303065


namespace NUMINAMATH_CALUDE_square_sum_value_l3030_303013

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l3030_303013


namespace NUMINAMATH_CALUDE_probability_three_fives_in_eight_rolls_l3030_303016

/-- A fair die has 6 sides -/
def die_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of times we want the specific outcome (5 in this case) to appear -/
def target_occurrences : ℕ := 3

/-- The probability of rolling exactly 3 fives in 8 rolls of a fair die -/
theorem probability_three_fives_in_eight_rolls :
  (Nat.choose num_rolls target_occurrences : ℚ) / (die_sides ^ num_rolls) = 56 / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_fives_in_eight_rolls_l3030_303016


namespace NUMINAMATH_CALUDE_mater_cost_percentage_l3030_303097

theorem mater_cost_percentage (lightning_cost sally_cost mater_cost : ℝ) :
  lightning_cost = 140000 →
  sally_cost = 3 * mater_cost →
  sally_cost = 42000 →
  mater_cost / lightning_cost = 0.1 := by
sorry

end NUMINAMATH_CALUDE_mater_cost_percentage_l3030_303097


namespace NUMINAMATH_CALUDE_divisibility_count_l3030_303021

theorem divisibility_count : ∃! n : ℕ, n > 0 ∧ n < 500 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 7 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_count_l3030_303021


namespace NUMINAMATH_CALUDE_farm_field_theorem_l3030_303014

/-- Represents the farm field and ploughing scenario -/
structure FarmField where
  totalArea : ℕ
  plannedRate : ℕ
  actualRate : ℕ
  extraDays : ℕ

/-- Calculates the area left to plough given the farm field scenario -/
def areaLeftToPlough (f : FarmField) : ℕ :=
  f.totalArea - f.actualRate * (f.totalArea / f.plannedRate + f.extraDays)

/-- Theorem stating that under the given conditions, 40 hectares are left to plough -/
theorem farm_field_theorem (f : FarmField) 
  (h1 : f.totalArea = 3780)
  (h2 : f.plannedRate = 90)
  (h3 : f.actualRate = 85)
  (h4 : f.extraDays = 2) :
  areaLeftToPlough f = 40 := by
  sorry

#eval areaLeftToPlough { totalArea := 3780, plannedRate := 90, actualRate := 85, extraDays := 2 }

end NUMINAMATH_CALUDE_farm_field_theorem_l3030_303014


namespace NUMINAMATH_CALUDE_tom_trip_cost_l3030_303025

/-- Calculates the total cost of Tom's trip to Barbados --/
def total_trip_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (flight_cost : ℚ) (num_nights : ℕ) (lodging_cost_per_night : ℚ) 
  (transportation_cost : ℚ) (food_cost_per_day : ℚ) (exchange_rate : ℚ) 
  (conversion_fee_rate : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  let local_expenses := (num_nights * lodging_cost_per_night + transportation_cost + 
    num_nights * food_cost_per_day)
  let conversion_fee := local_expenses * exchange_rate * conversion_fee_rate / exchange_rate
  out_of_pocket_medical + flight_cost + local_expenses + conversion_fee

/-- Theorem stating that the total cost of Tom's trip is $3060.10 --/
theorem tom_trip_cost : 
  total_trip_cost 10 45 250 0.8 1200 7 150 200 60 2 0.03 = 3060.1 := by
  sorry

#eval total_trip_cost 10 45 250 0.8 1200 7 150 200 60 2 0.03

end NUMINAMATH_CALUDE_tom_trip_cost_l3030_303025


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3030_303096

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3030_303096


namespace NUMINAMATH_CALUDE_nina_money_theorem_l3030_303069

theorem nina_money_theorem (x : ℝ) 
  (h1 : 6 * x = 8 * (x - 1.5)) : 6 * x = 36 := by
  sorry

end NUMINAMATH_CALUDE_nina_money_theorem_l3030_303069


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3030_303091

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n : ℤ) ≡ 409 [ZMOD 31] ∧ 
  ∀ m : ℕ+, (5 * m : ℤ) ≡ 409 [ZMOD 31] → n ≤ m → n = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3030_303091


namespace NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l3030_303075

theorem square_difference_from_sum_and_difference (a b : ℚ) 
  (h1 : a + b = 9 / 17) (h2 : a - b = 1 / 51) : 
  a^2 - b^2 = 3 / 289 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_from_sum_and_difference_l3030_303075


namespace NUMINAMATH_CALUDE_seven_plums_balance_one_pear_l3030_303057

-- Define the weights of fruits as real numbers
variable (apple pear plum : ℝ)

-- Condition 1: 3 apples and 1 pear weigh as much as 10 plums
def condition1 : Prop := 3 * apple + pear = 10 * plum

-- Condition 2: 1 apple and 6 plums balance 1 pear
def condition2 : Prop := apple + 6 * plum = pear

-- Condition 3: Fruits of the same kind have the same weight
-- (This is implicitly assumed by using single variables for each fruit type)

-- Theorem: 7 plums balance one pear
theorem seven_plums_balance_one_pear
  (h1 : condition1 apple pear plum)
  (h2 : condition2 apple pear plum) :
  7 * plum = pear := by sorry

end NUMINAMATH_CALUDE_seven_plums_balance_one_pear_l3030_303057


namespace NUMINAMATH_CALUDE_statement_d_not_always_true_l3030_303018

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

/-- Statement D is not always true -/
theorem statement_d_not_always_true 
  (α : Plane) (m n : Line) 
  (h1 : different_lines m n) 
  (h2 : lines_perpendicular m n) 
  (h3 : line_perp_plane m α) : 
  ¬ (line_parallel_plane n α) := sorry

end NUMINAMATH_CALUDE_statement_d_not_always_true_l3030_303018


namespace NUMINAMATH_CALUDE_triangulations_count_l3030_303085

/-- The number of triangulations of a convex n-gon with exactly two internal triangles -/
def triangulations_with_two_internal_triangles (n : ℕ) : ℕ :=
  n * Nat.choose (n - 4) 4 * 2^(n - 9)

/-- Theorem stating the number of triangulations of a convex n-gon with exactly two internal triangles -/
theorem triangulations_count (n : ℕ) (hn : n > 7) :
  triangulations_with_two_internal_triangles n =
    n * Nat.choose (n - 4) 4 * 2^(n - 9) := by
  sorry

end NUMINAMATH_CALUDE_triangulations_count_l3030_303085


namespace NUMINAMATH_CALUDE_daniels_improvement_l3030_303094

/-- Represents the jogging data for Daniel -/
structure JoggingData where
  initial_laps : ℕ
  initial_time : ℕ  -- in minutes
  final_laps : ℕ
  final_time : ℕ    -- in minutes

/-- Calculates the improvement in lap time (in seconds) given jogging data -/
def lapTimeImprovement (data : JoggingData) : ℕ :=
  let initial_lap_time := (data.initial_time * 60) / data.initial_laps
  let final_lap_time := (data.final_time * 60) / data.final_laps
  initial_lap_time - final_lap_time

/-- Theorem stating that Daniel's lap time improvement is 20 seconds -/
theorem daniels_improvement (data : JoggingData) 
  (h1 : data.initial_laps = 15) 
  (h2 : data.initial_time = 40)
  (h3 : data.final_laps = 18)
  (h4 : data.final_time = 42) : 
  lapTimeImprovement data = 20 := by
  sorry

end NUMINAMATH_CALUDE_daniels_improvement_l3030_303094


namespace NUMINAMATH_CALUDE_book_cost_prices_correct_l3030_303055

/-- Represents the cost and quantity information for a type of book -/
structure BookType where
  cost_per_book : ℝ
  total_cost : ℝ
  quantity : ℝ

/-- Proves that given the conditions, the cost prices for book types A and B are correct -/
theorem book_cost_prices_correct (book_a book_b : BookType)
  (h1 : book_a.cost_per_book = book_b.cost_per_book + 15)
  (h2 : book_a.total_cost = 675)
  (h3 : book_b.total_cost = 450)
  (h4 : book_a.quantity = book_b.quantity)
  (h5 : book_a.quantity = book_a.total_cost / book_a.cost_per_book)
  (h6 : book_b.quantity = book_b.total_cost / book_b.cost_per_book) :
  book_a.cost_per_book = 45 ∧ book_b.cost_per_book = 30 := by
  sorry

#check book_cost_prices_correct

end NUMINAMATH_CALUDE_book_cost_prices_correct_l3030_303055


namespace NUMINAMATH_CALUDE_function_inequality_l3030_303012

/-- Given a differentiable function f: ℝ → ℝ, if f(x) + f''(x) < 0 for all x,
    then f(1) < f(0)/e < f(-1)/(e^2) -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (hf'' : Differentiable ℝ (deriv (deriv f)))
    (h : ∀ x, f x + (deriv (deriv f)) x < 0) :
    f 1 < f 0 / Real.exp 1 ∧ f 0 / Real.exp 1 < f (-1) / (Real.exp 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3030_303012


namespace NUMINAMATH_CALUDE_range_of_x_l3030_303056

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_x (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, |a + b| + |a - b| ≥ |a| * f x) →
  ∃ x, x ∈ Set.Icc 0 4 ∧ ∀ y, (∀ z, |a + b| + |a - b| ≥ |a| * f z) → y ∈ Set.Icc 0 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_l3030_303056


namespace NUMINAMATH_CALUDE_walter_chores_l3030_303061

theorem walter_chores (total_days : ℕ) (regular_pay : ℕ) (exceptional_pay : ℕ) (total_earnings : ℕ) :
  total_days = 15 ∧ regular_pay = 4 ∧ exceptional_pay = 6 ∧ total_earnings = 78 →
  ∃ (regular_days exceptional_days : ℕ),
    regular_days + exceptional_days = total_days ∧
    regular_days * regular_pay + exceptional_days * exceptional_pay = total_earnings ∧
    exceptional_days = 9 :=
by sorry

end NUMINAMATH_CALUDE_walter_chores_l3030_303061


namespace NUMINAMATH_CALUDE_one_count_greater_than_zero_count_l3030_303017

/-- Represents the sequence of concatenated decimal representations of numbers from 1 to n -/
def concatenatedSequence (n : ℕ) : List ℕ := sorry

/-- Counts the occurrences of a specific digit in the concatenated sequence -/
def digitCount (d : ℕ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the count of '1' is always greater than the count of '0' in the sequence -/
theorem one_count_greater_than_zero_count (n : ℕ) : digitCount 1 n > digitCount 0 n := by sorry

end NUMINAMATH_CALUDE_one_count_greater_than_zero_count_l3030_303017


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3030_303090

/-- Given an ellipse and a hyperbola with shared foci, prove that the eccentricity of the ellipse is 1/2 -/
theorem ellipse_eccentricity (a b m n c : ℝ) : 
  a > 0 → b > 0 → m > 0 → n > 0 → a > b →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1) →  -- Ellipse equation
  (∀ x y : ℝ, x^2/m^2 - y^2/n^2 = 1) →  -- Hyperbola equation
  c^2 = a^2 - b^2 →                     -- Shared foci condition for ellipse
  c^2 = m^2 + n^2 →                     -- Shared foci condition for hyperbola
  c^2 = a * m →                         -- c is geometric mean of a and m
  n^2 = m^2 + c^2/2 →                   -- n^2 is arithmetic mean of 2m^2 and c^2
  c/a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3030_303090
