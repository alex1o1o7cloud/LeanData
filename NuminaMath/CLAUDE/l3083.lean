import Mathlib

namespace NUMINAMATH_CALUDE_prob_first_red_given_second_black_l3083_308307

/-- Represents the contents of an urn -/
structure Urn :=
  (white : ℕ)
  (red : ℕ)
  (black : ℕ)

/-- The probability of drawing a specific color from an urn -/
def prob_draw (u : Urn) (color : String) : ℚ :=
  match color with
  | "white" => u.white / (u.white + u.red + u.black)
  | "red" => u.red / (u.white + u.red + u.black)
  | "black" => u.black / (u.white + u.red + u.black)
  | _ => 0

/-- The contents of Urn A -/
def urn_A : Urn := ⟨4, 2, 0⟩

/-- The contents of Urn B -/
def urn_B : Urn := ⟨0, 3, 3⟩

/-- The probability of selecting an urn -/
def prob_select_urn : ℚ := 1/2

theorem prob_first_red_given_second_black :
  let p_red_and_black := 
    (prob_select_urn * prob_draw urn_A "red" * prob_select_urn * prob_draw urn_B "black") +
    (prob_select_urn * prob_draw urn_B "red" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black - 1) / (urn_B.red + urn_B.black - 1)))
  let p_second_black :=
    (prob_select_urn * prob_select_urn * prob_draw urn_B "black") +
    (prob_select_urn * prob_draw urn_B "red" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black) / (urn_B.red + urn_B.black - 1))) +
    (prob_select_urn * prob_draw urn_B "black" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black - 1) / (urn_B.red + urn_B.black - 1)))
  p_red_and_black / p_second_black = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_first_red_given_second_black_l3083_308307


namespace NUMINAMATH_CALUDE_man_speed_man_speed_result_l3083_308324

/-- Calculates the speed of a man running opposite to a train --/
theorem man_speed (train_speed : Real) (train_length : Real) (passing_time : Real) : Real :=
  let train_speed_mps := train_speed * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * 3600 / 1000
  man_speed_kmph

/-- The speed of the man is approximately 6.0024 km/h --/
theorem man_speed_result : 
  ∃ ε > 0, |man_speed 60 110 5.999520038396929 - 6.0024| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_result_l3083_308324


namespace NUMINAMATH_CALUDE_math_class_size_l3083_308320

theorem math_class_size :
  ∃! n : ℕ, 0 < n ∧ n < 50 ∧ n % 8 = 5 ∧ n % 6 = 1 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_math_class_size_l3083_308320


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l3083_308364

/-- The total cost of Amanda's kitchen upgrade --/
def kitchen_upgrade_cost (cabinet_knobs : ℕ) (knob_price : ℚ) (drawer_pulls : ℕ) (pull_price : ℚ) : ℚ :=
  (cabinet_knobs : ℚ) * knob_price + (drawer_pulls : ℚ) * pull_price

/-- Proof that Amanda's kitchen upgrade costs $77.00 --/
theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 (5/2) 8 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l3083_308364


namespace NUMINAMATH_CALUDE_square_root_of_square_l3083_308372

theorem square_root_of_square (x : ℝ) (h : x = 36) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_l3083_308372


namespace NUMINAMATH_CALUDE_optimal_bus_price_with_competition_optimal_bus_price_without_competition_decrease_in_passengers_l3083_308331

/-- Represents the demand function for transportation -/
def demand (p : ℝ) : ℝ := 3000 - 20 * p

/-- Represents the cost function for the bus company -/
def busCompanyCost (y : ℝ) : ℝ := y + 5

/-- Represents the train fare -/
def trainFare : ℝ := 10

/-- Represents the train capacity -/
def trainCapacity : ℝ := 1000

/-- Theorem stating the optimal bus price with train competition -/
theorem optimal_bus_price_with_competition :
  ∃ (p : ℝ), p = 50.5 ∧
  ∀ (p' : ℝ), p' ≠ p →
    let q := demand (min p' trainFare) - trainCapacity
    let revenue := p' * q
    let cost := busCompanyCost q
    revenue - cost ≤ p * (demand (min p trainFare) - trainCapacity) - busCompanyCost (demand (min p trainFare) - trainCapacity) :=
sorry

/-- Theorem stating the optimal bus price without train competition -/
theorem optimal_bus_price_without_competition :
  ∃ (p : ℝ), p = 75.5 ∧
  ∀ (p' : ℝ), p' ≠ p →
    let q := demand p'
    let revenue := p' * q
    let cost := busCompanyCost q
    revenue - cost ≤ p * demand p - busCompanyCost (demand p) :=
sorry

/-- Theorem stating the decrease in total passengers when train service is removed -/
theorem decrease_in_passengers :
  demand trainFare < demand 75.5 :=
sorry

end NUMINAMATH_CALUDE_optimal_bus_price_with_competition_optimal_bus_price_without_competition_decrease_in_passengers_l3083_308331


namespace NUMINAMATH_CALUDE_total_books_proof_l3083_308336

/-- The total number of books on two bookshelves -/
def total_books : ℕ := 30

/-- The number of books moved from the first shelf to the second shelf -/
def books_moved : ℕ := 5

theorem total_books_proof :
  (∃ (initial_books_per_shelf : ℕ),
    initial_books_per_shelf * 2 = total_books ∧
    (initial_books_per_shelf + books_moved) = 2 * (initial_books_per_shelf - books_moved)) :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_proof_l3083_308336


namespace NUMINAMATH_CALUDE_four_square_base_boxes_l3083_308328

/-- A box with a square base that can contain exactly 64 unit cubes. -/
structure SquareBaseBox where
  base : ℕ
  height : ℕ
  volume_eq_64 : base * base * height = 64

/-- The set of all possible SquareBaseBox configurations. -/
def all_square_base_boxes : Set SquareBaseBox :=
  { box | box.base * box.base * box.height = 64 }

/-- The theorem stating that there are exactly four possible SquareBaseBox configurations. -/
theorem four_square_base_boxes :
  all_square_base_boxes = {
    ⟨1, 64, rfl⟩,
    ⟨2, 16, rfl⟩,
    ⟨4, 4, rfl⟩,
    ⟨8, 1, rfl⟩
  } := by sorry

end NUMINAMATH_CALUDE_four_square_base_boxes_l3083_308328


namespace NUMINAMATH_CALUDE_factory_scrap_rate_l3083_308369

/-- The overall scrap rate of a factory with two machines -/
def overall_scrap_rate (output_a output_b scrap_rate_a scrap_rate_b : ℝ) : ℝ :=
  output_a * scrap_rate_a + output_b * scrap_rate_b

theorem factory_scrap_rate :
  overall_scrap_rate 0.45 0.55 0.02 0.03 = 0.0255 := by
  sorry

end NUMINAMATH_CALUDE_factory_scrap_rate_l3083_308369


namespace NUMINAMATH_CALUDE_log_equation_relationships_l3083_308394

/-- Given real numbers a and b satisfying log_(1/2)(a) = log_(1/3)(b), 
    exactly 2 out of 5 given relationships cannot hold true. -/
theorem log_equation_relationships (a b : ℝ) 
  (h : Real.log a / Real.log (1/2) = Real.log b / Real.log (1/3)) : 
  ∃! (s : Finset (Fin 5)), s.card = 2 ∧ 
  (∀ i ∈ s, match i with
    | 0 => ¬(a > b ∧ b > 1)
    | 1 => ¬(0 < b ∧ b < a ∧ a < 1)
    | 2 => ¬(b > a ∧ a > 1)
    | 3 => ¬(0 < a ∧ a < b ∧ b < 1)
    | 4 => ¬(a = b)
  ) ∧
  (∀ i ∉ s, match i with
    | 0 => (a > b ∧ b > 1)
    | 1 => (0 < b ∧ b < a ∧ a < 1)
    | 2 => (b > a ∧ a > 1)
    | 3 => (0 < a ∧ a < b ∧ b < 1)
    | 4 => (a = b)
  ) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_relationships_l3083_308394


namespace NUMINAMATH_CALUDE_min_value_of_f_l3083_308300

-- Define the function f(x)
def f (x : ℝ) : ℝ := 12 * x - x^3

-- State the theorem
theorem min_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-3) 3 ∧ f x = -16 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3083_308300


namespace NUMINAMATH_CALUDE_set_equality_l3083_308370

-- Define the sets M, N, and P
def M : Set ℝ := {x | ∃ n : ℤ, x = n}
def N : Set ℝ := {x | ∃ n : ℤ, x = n / 2}
def P : Set ℝ := {x | ∃ n : ℤ, x = n + 1 / 2}

-- Theorem statement
theorem set_equality : N = M ∪ P := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l3083_308370


namespace NUMINAMATH_CALUDE_cost_shop1_calculation_l3083_308312

-- Define the problem parameters
def books_shop1 : ℕ := 65
def books_shop2 : ℕ := 35
def cost_shop2 : ℕ := 2000
def avg_price : ℕ := 85

-- Theorem to prove
theorem cost_shop1_calculation :
  let total_books : ℕ := books_shop1 + books_shop2
  let total_cost : ℕ := total_books * avg_price
  let cost_shop1 : ℕ := total_cost - cost_shop2
  cost_shop1 = 6500 := by sorry

end NUMINAMATH_CALUDE_cost_shop1_calculation_l3083_308312


namespace NUMINAMATH_CALUDE_x_value_l3083_308387

theorem x_value (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3083_308387


namespace NUMINAMATH_CALUDE_stratified_sampling_geometric_sequence_l3083_308374

theorem stratified_sampling_geometric_sequence (total : ℕ) (ratio : ℕ) : 
  total = 140 → ratio = 2 → ∃ (x : ℕ), x + ratio * x + ratio^2 * x = total ∧ ratio * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_geometric_sequence_l3083_308374


namespace NUMINAMATH_CALUDE_probability_both_selected_l3083_308389

theorem probability_both_selected (prob_X prob_Y : ℚ) 
  (h1 : prob_X = 1/7) 
  (h2 : prob_Y = 2/5) : 
  prob_X * prob_Y = 2/35 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_selected_l3083_308389


namespace NUMINAMATH_CALUDE_sum_of_squares_ratio_l3083_308329

theorem sum_of_squares_ratio (a b c : ℚ) : 
  a + b + c = 14 → 
  b = 2 * a → 
  c = 3 * a → 
  a^2 + b^2 + c^2 = 686/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_ratio_l3083_308329


namespace NUMINAMATH_CALUDE_rectangle_segment_ratio_l3083_308392

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a rectangle ABCD -/
structure Rectangle :=
  (A B C D : Point)
  (AB_length : ℝ)
  (BC_length : ℝ)

/-- Represents the ratio of segments -/
structure Ratio :=
  (r s t u : ℕ)

def is_on_segment (P Q R : Point) : Prop := sorry

def intersect (P Q R S : Point) : Point := sorry

def parallel (P Q R S : Point) : Prop := sorry

theorem rectangle_segment_ratio 
  (ABCD : Rectangle)
  (E F G : Point)
  (P Q R : Point)
  (h1 : ABCD.AB_length = 8)
  (h2 : ABCD.BC_length = 4)
  (h3 : is_on_segment ABCD.B E ABCD.C)
  (h4 : is_on_segment ABCD.B F ABCD.C)
  (h5 : is_on_segment ABCD.C G ABCD.D)
  (h6 : (ABCD.B.x - E.x) / (E.x - ABCD.C.x) = 1 / 2)
  (h7 : (ABCD.B.x - F.x) / (F.x - ABCD.C.x) = 2 / 1)
  (h8 : P = intersect ABCD.A E ABCD.B ABCD.D)
  (h9 : Q = intersect ABCD.A F ABCD.B ABCD.D)
  (h10 : R = intersect ABCD.A G ABCD.B ABCD.D)
  (h11 : parallel ABCD.A G ABCD.B ABCD.C) :
  ∃ (ratio : Ratio), 
    ratio.r = 3 ∧ 
    ratio.s = 2 ∧ 
    ratio.t = 6 ∧ 
    ratio.u = 6 ∧
    ratio.r + ratio.s + ratio.t + ratio.u = 17 := by sorry

end NUMINAMATH_CALUDE_rectangle_segment_ratio_l3083_308392


namespace NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l3083_308388

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define what it means for two lines to be parallel
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Define a perpendicular segment between two lines
def PerpendicularSegment (l₁ l₂ : Line) : Type := sorry

-- Define the length of a perpendicular segment
def Length (seg : PerpendicularSegment l₁ l₂) : ℝ := sorry

-- Theorem statement
theorem perpendicular_segments_equal_length 
  (l₁ l₂ : Line) (h : Parallel l₁ l₂) :
  ∀ (seg₁ seg₂ : PerpendicularSegment l₁ l₂), 
  Length seg₁ = Length seg₂ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_segments_equal_length_l3083_308388


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3083_308379

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) : 
  total_votes = 460 → majority = 184 → 
  (70 : ℚ) = (100 * (total_votes / 2 + majority) : ℚ) / total_votes := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3083_308379


namespace NUMINAMATH_CALUDE_largest_number_in_sampling_l3083_308304

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  smallest_number : ℕ
  second_smallest : ℕ
  selected_count : ℕ
  common_difference : ℕ

/-- The largest number in a systematic sampling. -/
def largest_number (s : SystematicSampling) : ℕ :=
  s.smallest_number + (s.selected_count - 1) * s.common_difference

/-- Theorem stating the largest number in the given systematic sampling. -/
theorem largest_number_in_sampling :
  let s : SystematicSampling := {
    total_students := 80,
    smallest_number := 6,
    second_smallest := 14,
    selected_count := 10,
    common_difference := 8
  }
  largest_number s = 78 := by sorry

end NUMINAMATH_CALUDE_largest_number_in_sampling_l3083_308304


namespace NUMINAMATH_CALUDE_slope_condition_implies_m_zero_l3083_308314

theorem slope_condition_implies_m_zero (m : ℝ) : 
  (4 - m^2) / (m - (-2)) = 2 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_slope_condition_implies_m_zero_l3083_308314


namespace NUMINAMATH_CALUDE_f_greater_than_three_f_inequality_solution_range_l3083_308310

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Theorem 1: f(x) > 3 iff x > 0
theorem f_greater_than_three (x : ℝ) : f x > 3 ↔ x > 0 := by sorry

-- Theorem 2: f(x) + 1 ≤ 4^a - 5×2^a has a solution iff a ∈ (-∞,0] ∪ [2,+∞)
theorem f_inequality_solution_range (a : ℝ) : 
  (∃ x, f x + 1 ≤ 4^a - 5*2^a) ↔ (a ≤ 0 ∨ a ≥ 2) := by sorry

end NUMINAMATH_CALUDE_f_greater_than_three_f_inequality_solution_range_l3083_308310


namespace NUMINAMATH_CALUDE_selection_theorem_l3083_308396

/-- The number of ways to select one person from a department with n employees -/
def selectOne (n : ℕ) : ℕ := n

/-- The total number of ways to select one person from three departments -/
def totalWays (deptA deptB deptC : ℕ) : ℕ :=
  selectOne deptA + selectOne deptB + selectOne deptC

theorem selection_theorem :
  totalWays 2 4 3 = 9 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l3083_308396


namespace NUMINAMATH_CALUDE_simplify_expression_l3083_308334

theorem simplify_expression (s : ℝ) : 180 * s - 88 * s = 92 * s := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3083_308334


namespace NUMINAMATH_CALUDE_cost_price_is_100_l3083_308322

/-- The cost price of a clothing item, given specific price changes and profit. -/
def cost_price : ℝ → Prop :=
  fun x => 
    let price_after_increase := x * 1.2
    let final_price := price_after_increase * 0.9
    final_price - x = 8

/-- Theorem stating that the cost price is 100 yuan. -/
theorem cost_price_is_100 : cost_price 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_100_l3083_308322


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l3083_308344

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x - 3)^2 = (3*y + 4)^2 - 90

/-- Function to determine the type of conic section -/
def determine_conic_type (eq : (ℝ → ℝ → Prop)) : ConicType :=
  sorry

/-- Theorem stating that the given equation describes a hyperbola -/
theorem conic_is_hyperbola :
  determine_conic_type conic_equation = ConicType.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l3083_308344


namespace NUMINAMATH_CALUDE_average_pop_percentage_l3083_308349

/-- Calculates the percentage of popped kernels in a bag -/
def popPercentage (popped : ℕ) (total : ℕ) : ℚ :=
  (popped : ℚ) / (total : ℚ) * 100

/-- Theorem: The average percentage of popped kernels across three bags is 82% -/
theorem average_pop_percentage :
  let bag1 := popPercentage 60 75
  let bag2 := popPercentage 42 50
  let bag3 := popPercentage 82 100
  (bag1 + bag2 + bag3) / 3 = 82 := by
  sorry

end NUMINAMATH_CALUDE_average_pop_percentage_l3083_308349


namespace NUMINAMATH_CALUDE_multiple_calculation_l3083_308348

theorem multiple_calculation (number : ℝ) (value : ℝ) (multiple : ℝ) : 
  number = -4.5 →
  value = 36 →
  10 * number = value - multiple * number →
  multiple = -18 := by sorry

end NUMINAMATH_CALUDE_multiple_calculation_l3083_308348


namespace NUMINAMATH_CALUDE_circle_M_equation_l3083_308321

/-- A circle M with the following properties:
    1. Tangent to the y-axis
    2. Its center lies on the line y = 1/2x
    3. The chord it cuts on the x-axis is 2√3 long -/
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_y_axis : abs (center.1) = radius
  center_on_line : center.2 = 1/2 * center.1
  x_axis_chord : 2 * radius = 2 * Real.sqrt 3

/-- The standard equation of circle M is either (x-2)² + (y-1)² = 4 or (x+2)² + (y+1)² = 4 -/
theorem circle_M_equation (M : CircleM) :
  (∀ x y, (x - 2)^2 + (y - 1)^2 = 4) ∨ (∀ x y, (x + 2)^2 + (y + 1)^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_circle_M_equation_l3083_308321


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3083_308385

theorem trigonometric_identities (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) : 
  ((3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 8/9) ∧ 
  (Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 1 = 13/10) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3083_308385


namespace NUMINAMATH_CALUDE_pairwise_product_inequality_l3083_308380

theorem pairwise_product_inequality (a b c : ℕ+) : 
  (a * b : ℕ) + (b * c : ℕ) + (a * c : ℕ) ≤ 3 * (a * b * c : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_product_inequality_l3083_308380


namespace NUMINAMATH_CALUDE_expression_evaluation_l3083_308315

theorem expression_evaluation (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3083_308315


namespace NUMINAMATH_CALUDE_a_less_than_b_l3083_308376

theorem a_less_than_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (h : (1 - a) * b > 1/4) : a < b := by
  sorry

end NUMINAMATH_CALUDE_a_less_than_b_l3083_308376


namespace NUMINAMATH_CALUDE_wire_cutting_l3083_308339

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 14 →
  ratio = 2 / 5 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 4 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l3083_308339


namespace NUMINAMATH_CALUDE_sum_of_possible_distances_l3083_308358

/-- Given two points A and B on a number line, where the distance between A and B is 2,
    and the distance between A and the origin O is 3,
    the sum of all possible distances between B and the origin O is 12. -/
theorem sum_of_possible_distances (A B : ℝ) : 
  (|A - B| = 2) → (|A| = 3) → (|B| + |-B| + |B - 2| + |-(B - 2)| = 12) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_distances_l3083_308358


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3083_308338

theorem correct_quotient_proof (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 70) : N / 21 = 40 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3083_308338


namespace NUMINAMATH_CALUDE_tank_capacity_l3083_308386

theorem tank_capacity (initial_fraction : ℚ) (added_gallons : ℚ) (final_fraction : ℚ) :
  initial_fraction = 3/4 →
  added_gallons = 9 →
  final_fraction = 7/8 →
  initial_fraction * C + added_gallons = final_fraction * C →
  C = 72 :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3083_308386


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3083_308352

def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def IsIncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ IsIncreasingSequence a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3083_308352


namespace NUMINAMATH_CALUDE_tank_filling_ratio_l3083_308342

theorem tank_filling_ratio (tank_capacity : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) (filling_time : ℝ) :
  tank_capacity = 1 →
  inflow_rate = 0.5 →
  outflow_rate1 = 0.25 →
  outflow_rate2 = 1/6 →
  filling_time = 6 →
  (tank_capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * filling_time) / tank_capacity = 0.5 := by
  sorry

#check tank_filling_ratio

end NUMINAMATH_CALUDE_tank_filling_ratio_l3083_308342


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3083_308351

theorem quadratic_inequality_condition (x : ℝ) : 
  (((x < 1) ∨ (x > 4)) → (x^2 - 3*x + 2 > 0)) ∧ 
  (∃ x, (x^2 - 3*x + 2 > 0) ∧ ¬((x < 1) ∨ (x > 4))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3083_308351


namespace NUMINAMATH_CALUDE_school_location_minimizes_distance_l3083_308317

/-- Represents a town with a number of students -/
structure Town where
  name : String
  students : ℕ

/-- Calculates the total distance traveled by students -/
def totalDistance (schoolLocation : Town) (townA : Town) (townB : Town) (distance : ℕ) : ℕ :=
  if schoolLocation.name = townA.name then
    townB.students * distance
  else if schoolLocation.name = townB.name then
    townA.students * distance
  else
    (townA.students + townB.students) * distance

/-- Theorem: Building a school in the town with more students minimizes total distance -/
theorem school_location_minimizes_distance (townA townB : Town) (distance : ℕ) :
  townA.students < townB.students →
  totalDistance townB townA townB distance ≤ totalDistance townA townA townB distance :=
by
  sorry

end NUMINAMATH_CALUDE_school_location_minimizes_distance_l3083_308317


namespace NUMINAMATH_CALUDE_p_amount_l3083_308326

theorem p_amount (p : ℚ) : p = (1/4) * p + 42 → p = 56 := by
  sorry

end NUMINAMATH_CALUDE_p_amount_l3083_308326


namespace NUMINAMATH_CALUDE_papers_per_notepad_l3083_308325

/-- The number of folds applied to the paper -/
def num_folds : ℕ := 3

/-- The number of days a notepad lasts -/
def days_per_notepad : ℕ := 4

/-- The number of notes written per day -/
def notes_per_day : ℕ := 10

/-- The number of smaller pieces obtained from one letter-size paper after folding -/
def pieces_per_paper : ℕ := 2^num_folds

/-- The total number of notes in one notepad -/
def notes_per_notepad : ℕ := days_per_notepad * notes_per_day

/-- Theorem: The number of letter-size papers needed for one notepad is 5 -/
theorem papers_per_notepad : (notes_per_notepad + pieces_per_paper - 1) / pieces_per_paper = 5 := by
  sorry

end NUMINAMATH_CALUDE_papers_per_notepad_l3083_308325


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l3083_308332

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

theorem jordan_rectangle_width : 
  ∀ (carol jordan : Rectangle),
  carol.length = 15 ∧ 
  carol.width = 24 ∧ 
  jordan.length = 8 ∧
  area carol = area jordan →
  jordan.width = 45 := by sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l3083_308332


namespace NUMINAMATH_CALUDE_muffin_milk_calculation_l3083_308306

/-- Given that 24 muffins require 3 liters of milk and 1 liter equals 4 cups,
    prove that 6 muffins require 3 cups of milk. -/
theorem muffin_milk_calculation (muffins_large : ℕ) (milk_liters : ℕ) (cups_per_liter : ℕ) 
  (muffins_small : ℕ) :
  muffins_large = 24 →
  milk_liters = 3 →
  cups_per_liter = 4 →
  muffins_small = 6 →
  (milk_liters * cups_per_liter * muffins_small) / muffins_large = 3 :=
by
  sorry

#check muffin_milk_calculation

end NUMINAMATH_CALUDE_muffin_milk_calculation_l3083_308306


namespace NUMINAMATH_CALUDE_two_numbers_sum_difference_product_l3083_308363

theorem two_numbers_sum_difference_product 
  (x y : ℝ) 
  (sum_eq : x + y = 40) 
  (diff_eq : x - y = 16) : 
  x = 28 ∧ y = 12 ∧ x * y = 336 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_sum_difference_product_l3083_308363


namespace NUMINAMATH_CALUDE_recipe_liquid_sum_l3083_308327

/-- Given the amounts of oil and water used in a recipe, 
    prove that the total amount of liquid is their sum. -/
theorem recipe_liquid_sum (oil water : ℝ) 
  (h_oil : oil = 0.17) 
  (h_water : water = 1.17) : 
  oil + water = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_recipe_liquid_sum_l3083_308327


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l3083_308383

theorem quadratic_equation_conversion (x : ℝ) : 
  (∃ m n : ℝ, x^2 + 2*x - 3 = 0 ↔ (x + m)^2 = n) → 
  (∃ m n : ℝ, x^2 + 2*x - 3 = 0 ↔ (x + m)^2 = n ∧ m + n = 5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l3083_308383


namespace NUMINAMATH_CALUDE_problem_1_l3083_308341

theorem problem_1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3083_308341


namespace NUMINAMATH_CALUDE_squares_after_seven_dwarfs_l3083_308391

/-- Represents the process of a dwarf cutting a square --/
def dwarf_cut (n : ℕ) : ℕ := n + 3

/-- Calculates the number of squares after n dwarfs have performed their cuts --/
def squares_after_cuts (n : ℕ) : ℕ := 
  Nat.iterate dwarf_cut n 1

/-- The theorem stating that after 7 dwarfs, there are 22 squares --/
theorem squares_after_seven_dwarfs : 
  squares_after_cuts 7 = 22 := by sorry

end NUMINAMATH_CALUDE_squares_after_seven_dwarfs_l3083_308391


namespace NUMINAMATH_CALUDE_initial_red_marbles_l3083_308378

theorem initial_red_marbles (initial_blue : ℕ) (removed_red : ℕ) (total_remaining : ℕ) : 
  initial_blue = 30 →
  removed_red = 3 →
  total_remaining = 35 →
  ∃ initial_red : ℕ, 
    initial_red = 20 ∧ 
    total_remaining = (initial_red - removed_red) + (initial_blue - 4 * removed_red) :=
by sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l3083_308378


namespace NUMINAMATH_CALUDE_coupons_used_proof_l3083_308397

/-- Calculates the total number of coupons used in a store's promotion --/
def total_coupons_used (initial_stock : ℝ) (books_sold : ℝ) (coupons_per_book : ℝ) : ℝ :=
  (initial_stock - books_sold) * coupons_per_book

/-- Proves that the total number of coupons used is 80.0 --/
theorem coupons_used_proof (initial_stock : ℝ) (books_sold : ℝ) (coupons_per_book : ℝ)
  (h1 : initial_stock = 40.0)
  (h2 : books_sold = 20.0)
  (h3 : coupons_per_book = 4.0) :
  total_coupons_used initial_stock books_sold coupons_per_book = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_coupons_used_proof_l3083_308397


namespace NUMINAMATH_CALUDE_sugar_recipe_calculation_l3083_308356

theorem sugar_recipe_calculation (initial_required : ℚ) (available : ℚ) : 
  initial_required = 1/3 → available = 1/6 → 
  (initial_required - available = 1/6) ∧ (2 * (initial_required - available) = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_sugar_recipe_calculation_l3083_308356


namespace NUMINAMATH_CALUDE_bicycle_wheels_l3083_308377

theorem bicycle_wheels (num_bicycles : ℕ) (num_tricycles : ℕ) (tricycle_wheels : ℕ) (total_wheels : ℕ) :
  num_bicycles = 16 →
  num_tricycles = 7 →
  tricycle_wheels = 3 →
  total_wheels = 53 →
  ∃ (bicycle_wheels : ℕ), 
    bicycle_wheels = 2 ∧ 
    num_bicycles * bicycle_wheels + num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l3083_308377


namespace NUMINAMATH_CALUDE_tuna_sales_difference_l3083_308311

/-- Calculates the difference in daily revenue between peak and low seasons for tuna fish sales. -/
theorem tuna_sales_difference (peak_rate : ℕ) (low_rate : ℕ) (price : ℕ) (hours : ℕ) : 
  peak_rate = 6 → low_rate = 4 → price = 60 → hours = 15 →
  (peak_rate * price * hours) - (low_rate * price * hours) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_tuna_sales_difference_l3083_308311


namespace NUMINAMATH_CALUDE_equation_solutions_l3083_308305

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1 / 12 ↔ x = 5 + Real.sqrt 19 ∨ x = 5 - Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3083_308305


namespace NUMINAMATH_CALUDE_task_completion_time_l3083_308346

/-- The number of days A takes to complete the task -/
def days_A : ℚ := 12

/-- The efficiency ratio of B compared to A -/
def efficiency_B : ℚ := 1.75

/-- The number of days B takes to complete the task -/
def days_B : ℚ := 48 / 7

theorem task_completion_time :
  days_B = days_A / efficiency_B := by sorry

end NUMINAMATH_CALUDE_task_completion_time_l3083_308346


namespace NUMINAMATH_CALUDE_zhang_daily_distance_l3083_308367

/-- Given a one-way distance and number of round trips, calculates the total distance driven. -/
def total_distance (one_way_distance : ℕ) (num_round_trips : ℕ) : ℕ :=
  2 * one_way_distance * num_round_trips

/-- Proves that given a one-way distance of 33 kilometers and 5 round trips per day, 
    the total distance driven is 330 kilometers. -/
theorem zhang_daily_distance : total_distance 33 5 = 330 := by
  sorry

end NUMINAMATH_CALUDE_zhang_daily_distance_l3083_308367


namespace NUMINAMATH_CALUDE_parallel_resistance_l3083_308343

theorem parallel_resistance (x y r : ℝ) : 
  x = 4 → y = 5 → (1 / r = 1 / x + 1 / y) → r = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistance_l3083_308343


namespace NUMINAMATH_CALUDE_polygon_diagonals_l3083_308302

theorem polygon_diagonals (n : ℕ) (h : n ≥ 3) :
  (n * (n - 1)) / 2 - n = 20 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l3083_308302


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3083_308399

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2024 - 1) (2^2000 - 1) = 2^24 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3083_308399


namespace NUMINAMATH_CALUDE_problem_statement_l3083_308381

theorem problem_statement (a b : ℝ) (h : a + b = 1) :
  (a^3 + b^3 ≥ 1/4) ∧
  (∃ x : ℝ, |x - a| + |x - b| ≤ 5 → 0 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3083_308381


namespace NUMINAMATH_CALUDE_yumis_farm_chickens_l3083_308355

/-- The number of chickens on Yumi's farm -/
def num_chickens : ℕ := 6

/-- The number of pigs on Yumi's farm -/
def num_pigs : ℕ := 9

/-- The number of legs each pig has -/
def pig_legs : ℕ := 4

/-- The number of legs each chicken has -/
def chicken_legs : ℕ := 2

/-- The total number of legs of all animals on Yumi's farm -/
def total_legs : ℕ := 48

theorem yumis_farm_chickens :
  num_chickens * chicken_legs + num_pigs * pig_legs = total_legs :=
by sorry

end NUMINAMATH_CALUDE_yumis_farm_chickens_l3083_308355


namespace NUMINAMATH_CALUDE_uphill_divisible_by_25_count_l3083_308323

/-- A positive integer is uphill if every digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get! i < (n.digits 10).get! j

/-- A number is divisible by 25 if and only if it ends in 00 or 25. -/
def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

/-- The count of uphill integers divisible by 25 -/
def count_uphill_divisible_by_25 : ℕ := 3

theorem uphill_divisible_by_25_count :
  (∃ S : Finset ℕ, (∀ n ∈ S, is_uphill n ∧ divisible_by_25 n) ∧
                   (∀ n, is_uphill n → divisible_by_25 n → n ∈ S) ∧
                   S.card = count_uphill_divisible_by_25) :=
sorry

end NUMINAMATH_CALUDE_uphill_divisible_by_25_count_l3083_308323


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3083_308301

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the open interval (-∞, 2)
def A : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem complement_of_A_in_U : 
  U \ A = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3083_308301


namespace NUMINAMATH_CALUDE_intersection_when_m_3_m_value_for_given_intersection_l3083_308335

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | -x^2 + 2*x + m > 0}

-- Theorem 1: Intersection of A and B when m = 3
theorem intersection_when_m_3 :
  A ∩ B 3 = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem 2: Value of m when A ∩ B = {x | -1 < x < 4}
theorem m_value_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x : ℝ | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_m_value_for_given_intersection_l3083_308335


namespace NUMINAMATH_CALUDE_characterize_square_property_functions_l3083_308347

/-- A function f: ℕ → ℕ satisfies the square property if (f(m) + n)(m + f(n)) is a square for all m, n ∈ ℕ -/
def satisfies_square_property (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k * k

/-- The main theorem characterizing functions satisfying the square property -/
theorem characterize_square_property_functions :
  ∀ f : ℕ → ℕ, satisfies_square_property f ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
sorry

end NUMINAMATH_CALUDE_characterize_square_property_functions_l3083_308347


namespace NUMINAMATH_CALUDE_complex_equation_result_l3083_308365

theorem complex_equation_result (x y : ℝ) (h : x * Complex.I - y = -1 + Complex.I) :
  (1 - Complex.I) * (x - y * Complex.I) = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l3083_308365


namespace NUMINAMATH_CALUDE_rectangle_area_reduction_l3083_308393

/-- Given a rectangle with dimensions 5 and 7 inches, if reducing one side by 2 inches
    results in an area of 21 square inches, then reducing the other side by 2 inches
    will result in an area of 25 square inches. -/
theorem rectangle_area_reduction (w h : ℝ) : 
  w = 5 ∧ h = 7 ∧ (w - 2) * h = 21 → w * (h - 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_reduction_l3083_308393


namespace NUMINAMATH_CALUDE_youngest_not_first_or_last_l3083_308354

def number_of_people : ℕ := 5

-- Define a function to calculate the number of permutations
def permutations (n : ℕ) : ℕ := Nat.factorial n

-- Define a function to calculate the number of valid arrangements
def valid_arrangements (n : ℕ) : ℕ :=
  permutations n - 2 * permutations (n - 1)

-- Theorem statement
theorem youngest_not_first_or_last :
  valid_arrangements number_of_people = 72 := by
  sorry

end NUMINAMATH_CALUDE_youngest_not_first_or_last_l3083_308354


namespace NUMINAMATH_CALUDE_bruce_remaining_eggs_l3083_308353

def bruce_initial_eggs : ℕ := 75
def eggs_lost : ℕ := 70

theorem bruce_remaining_eggs :
  bruce_initial_eggs - eggs_lost = 5 :=
by sorry

end NUMINAMATH_CALUDE_bruce_remaining_eggs_l3083_308353


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l3083_308319

theorem sqrt_sum_equals_twelve :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l3083_308319


namespace NUMINAMATH_CALUDE_no_seven_divisible_ones_and_five_l3083_308333

theorem no_seven_divisible_ones_and_five : ¬∃ (n : ℕ), 7 ∣ ((10^(n+1) + 35) / 9) := by
  sorry

end NUMINAMATH_CALUDE_no_seven_divisible_ones_and_five_l3083_308333


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l3083_308361

/-- The maximum distance a point can be from the origin, given the constraints --/
def max_distance : ℝ := 10

/-- The coordinates of the post where the dog is tied --/
def post : ℝ × ℝ := (6, 8)

/-- The length of the rope --/
def rope_length : ℝ := 15

/-- The x-coordinate of the wall's end --/
def wall_end : ℝ := 10

/-- Theorem stating the maximum distance from the origin --/
theorem max_distance_from_origin :
  ∀ (p : ℝ × ℝ), 
    (p.1 ≤ wall_end) → -- point is not beyond the wall
    (p.2 ≥ 0) → -- point is not below the wall
    ((p.1 - post.1)^2 + (p.2 - post.2)^2 ≤ rope_length^2) → -- point is within or on the circle
    (p.1^2 + p.2^2 ≤ max_distance^2) := -- distance from origin is at most max_distance
by
  sorry


end NUMINAMATH_CALUDE_max_distance_from_origin_l3083_308361


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3083_308316

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3083_308316


namespace NUMINAMATH_CALUDE_pencils_per_box_l3083_308366

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℚ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4) : 
  (total_pencils : ℚ) / num_boxes = 648 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_box_l3083_308366


namespace NUMINAMATH_CALUDE_value_of_x_l3083_308340

theorem value_of_x (x y z : ℝ) 
  (h1 : x = (1/2) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : z = 100) : 
  x = 12.5 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l3083_308340


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l3083_308308

theorem line_hyperbola_intersection (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < -1 ∧ x₂ < -1 ∧
    (x₁^2 - (k*x₁ - 1)^2 = 1) ∧
    (x₂^2 - (k*x₂ - 1)^2 = 1)) ↔
  -Real.sqrt 2 < k ∧ k < -1 := by
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l3083_308308


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3083_308359

theorem fixed_point_on_line (k : ℝ) : k * 1 - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3083_308359


namespace NUMINAMATH_CALUDE_margin_relation_l3083_308371

theorem margin_relation (n : ℝ) (C S M : ℝ) 
  (h1 : M = (1/n) * C) 
  (h2 : S = C + M) : 
  M = (1/(n+1)) * S := by
sorry

end NUMINAMATH_CALUDE_margin_relation_l3083_308371


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l3083_308345

/-- Given a parabola y² = 2px (p > 0) with a point A(4, m) on it,
    if the distance from A to the focus is 17/4, then p = 1/2. -/
theorem parabola_focus_distance (p : ℝ) (m : ℝ) : 
  p > 0 → 
  m^2 = 2*p*4 → 
  (4 - p/2)^2 + m^2 = (17/4)^2 → 
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l3083_308345


namespace NUMINAMATH_CALUDE_arcsin_sin_eq_half_x_solutions_l3083_308373

theorem arcsin_sin_eq_half_x_solutions :
  {x : ℝ | x ∈ Set.Icc (-Real.pi) Real.pi ∧ Real.arcsin (Real.sin x) = x / 2} =
  {-2 * Real.pi / 3, 0, 2 * Real.pi / 3} := by
sorry

end NUMINAMATH_CALUDE_arcsin_sin_eq_half_x_solutions_l3083_308373


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3083_308318

/-- The solution set of the quadratic inequality (m^2-2m-3)x^2-(m-3)x-1<0 is ℝ if and only if -1/5 < m ≤ 3 -/
theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ (-1/5 < m ∧ m ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3083_308318


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l3083_308313

def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / (10 - a) + y^2 / (a - 2) = 1

theorem ellipse_focal_length (a : ℝ) : 
  (∃ x y : ℝ, ellipse_equation x y a) → 
  (∃ c : ℝ, c = 2) →
  (a = 4 ∨ a = 8) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l3083_308313


namespace NUMINAMATH_CALUDE_triangle_side_product_l3083_308382

noncomputable def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_product (a b c : ℝ) :
  Triangle a b c →
  (a + b)^2 - c^2 = 4 →
  Real.cos (60 * π / 180) = 1/2 →
  a * b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_product_l3083_308382


namespace NUMINAMATH_CALUDE_combined_class_average_weight_l3083_308357

/-- Calculates the average weight of a combined class given two sections -/
def averageWeightCombinedClass (studentsA : ℕ) (studentsB : ℕ) (avgWeightA : ℚ) (avgWeightB : ℚ) : ℚ :=
  (studentsA * avgWeightA + studentsB * avgWeightB) / (studentsA + studentsB)

/-- Theorem stating the average weight of the combined class -/
theorem combined_class_average_weight :
  averageWeightCombinedClass 26 34 50 30 = 2320 / 60 := by
  sorry

#eval averageWeightCombinedClass 26 34 50 30

end NUMINAMATH_CALUDE_combined_class_average_weight_l3083_308357


namespace NUMINAMATH_CALUDE_equation_solution_count_l3083_308384

theorem equation_solution_count : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, (n^2 - 2*n - 2)*n^2 + 47 = (n^2 - 2*n - 2)*16*n - 16) ∧ 
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_count_l3083_308384


namespace NUMINAMATH_CALUDE_grass_seed_coverage_l3083_308395

/-- Calculates the area covered by one bag of grass seed given the dimensions of a rectangular lawn
and the total area covered by a known number of bags. -/
theorem grass_seed_coverage 
  (lawn_length : ℝ) 
  (lawn_width : ℝ) 
  (extra_area : ℝ) 
  (num_bags : ℕ) 
  (h1 : lawn_length = 22)
  (h2 : lawn_width = 36)
  (h3 : extra_area = 208)
  (h4 : num_bags = 4) :
  (lawn_length * lawn_width + extra_area) / num_bags = 250 :=
by sorry

end NUMINAMATH_CALUDE_grass_seed_coverage_l3083_308395


namespace NUMINAMATH_CALUDE_B_age_is_18_l3083_308375

/-- Given three people A, B, and C with the following conditions:
  1. A is two years older than B
  2. B is twice as old as C
  3. The sum of their ages is 47
  Prove that B is 18 years old -/
theorem B_age_is_18 (A B C : ℕ) 
  (h1 : A = B + 2)
  (h2 : B = 2 * C)
  (h3 : A + B + C = 47) :
  B = 18 := by
  sorry

end NUMINAMATH_CALUDE_B_age_is_18_l3083_308375


namespace NUMINAMATH_CALUDE_number_percentage_equality_l3083_308330

theorem number_percentage_equality (x : ℚ) : 
  (30 / 100 : ℚ) * x = (40 / 100 : ℚ) * 40 → x = 160 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l3083_308330


namespace NUMINAMATH_CALUDE_f_of_five_equals_sixtytwo_l3083_308303

/-- Given a function f where f(x) = 2x² + y and f(2) = 20, prove that f(5) = 62 -/
theorem f_of_five_equals_sixtytwo (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : 
  f 5 = 62 := by
  sorry

end NUMINAMATH_CALUDE_f_of_five_equals_sixtytwo_l3083_308303


namespace NUMINAMATH_CALUDE_xia_initial_stickers_l3083_308368

/-- The number of stickers Xia shared with her friends -/
def shared_stickers : ℕ := 100

/-- The number of sheets of stickers Xia had left after sharing -/
def remaining_sheets : ℕ := 5

/-- The number of stickers on each sheet -/
def stickers_per_sheet : ℕ := 10

/-- Xia's initial number of stickers -/
def initial_stickers : ℕ := shared_stickers + remaining_sheets * stickers_per_sheet

theorem xia_initial_stickers : initial_stickers = 150 := by
  sorry

end NUMINAMATH_CALUDE_xia_initial_stickers_l3083_308368


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3083_308337

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 24) :
  |x - y| = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3083_308337


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3083_308362

theorem student_average_greater_than_true_average 
  (x y w z : ℝ) (h : x < y ∧ y < w ∧ w < z) : 
  (x + y + 2*w + 2*z) / 6 > (x + y + w + z) / 4 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3083_308362


namespace NUMINAMATH_CALUDE_derivative_problems_l3083_308309

open Real

theorem derivative_problems :
  (∀ x : ℝ, deriv (λ x => (2*x^2 + 3)*(3*x - 1)) x = 18*x^2 - 4*x + 9) ∧
  (∀ x : ℝ, deriv (λ x => x * exp x + 2*x + 1) x = exp x + x * exp x + 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_problems_l3083_308309


namespace NUMINAMATH_CALUDE_line_properties_l3083_308350

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def l₂ (a x y : ℝ) : Prop := 3 * x + (a - 1) * y + 3 - a = 0

-- Define perpendicularity of two lines
def perpendicular (a : ℝ) : Prop := (-a / 2) * (-3 / (a - 1)) = -1

theorem line_properties (a : ℝ) :
  (l₂ a (-2/3) 1) ∧
  (perpendicular a → a = 2/5) := by sorry

end NUMINAMATH_CALUDE_line_properties_l3083_308350


namespace NUMINAMATH_CALUDE_teds_overall_correct_percentage_l3083_308398

theorem teds_overall_correct_percentage
  (t : ℝ) -- total number of problems
  (h_t_pos : t > 0) -- ensure t is positive
  (independent_solving : ℝ := 0.4 * t) -- 40% of problems solved independently
  (collaborative_solving : ℝ := 0.6 * t) -- 60% of problems solved collaboratively
  (ned_independent_correct : ℝ := 0.7 * independent_solving) -- Ned's correct answers for independent solving
  (ned_overall_correct : ℝ := 0.82 * t) -- Ned's overall correct answers
  (ted_independent_correct : ℝ := 0.85 * independent_solving) -- Ted's correct answers for independent solving
  : (ted_independent_correct + (ned_overall_correct - ned_independent_correct)) / t = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_teds_overall_correct_percentage_l3083_308398


namespace NUMINAMATH_CALUDE_paiges_flowers_l3083_308360

theorem paiges_flowers (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) (remaining_bouquets : ℕ) :
  flowers_per_bouquet = 7 →
  wilted_flowers = 18 →
  remaining_bouquets = 5 →
  flowers_per_bouquet * remaining_bouquets + wilted_flowers = 53 :=
by sorry

end NUMINAMATH_CALUDE_paiges_flowers_l3083_308360


namespace NUMINAMATH_CALUDE_gcd_5800_14025_l3083_308390

theorem gcd_5800_14025 : Nat.gcd 5800 14025 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5800_14025_l3083_308390
