import Mathlib

namespace NUMINAMATH_CALUDE_boxes_per_carton_l2385_238580

/-- Given a number of cartons per case, prove that there is 1 box per carton -/
theorem boxes_per_carton (c : ℕ) (h1 : c > 0) : ∃ (b : ℕ), b = 1 ∧ b * c * 400 = 400 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_carton_l2385_238580


namespace NUMINAMATH_CALUDE_wire_cutting_l2385_238540

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 →
  ratio = 2 / 5 →
  shorter_piece + shorter_piece / ratio = total_length →
  shorter_piece = 120 / 7 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l2385_238540


namespace NUMINAMATH_CALUDE_lcm_of_8_and_15_l2385_238527

theorem lcm_of_8_and_15 : Nat.lcm 8 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_8_and_15_l2385_238527


namespace NUMINAMATH_CALUDE_modified_grid_perimeter_l2385_238570

/-- Represents a square grid with a hole and an additional row on top. -/
structure ModifiedGrid :=
  (side : ℕ)
  (hole_size : ℕ)
  (top_row : ℕ)

/-- Calculates the perimeter of the modified grid. -/
def perimeter (grid : ModifiedGrid) : ℕ :=
  2 * (grid.side + grid.top_row) + 2 * grid.side - 2 * grid.hole_size

/-- Theorem stating that the perimeter of the specific modified 3x3 grid is 9. -/
theorem modified_grid_perimeter :
  ∃ (grid : ModifiedGrid), grid.side = 3 ∧ grid.hole_size = 1 ∧ grid.top_row = 3 ∧ perimeter grid = 9 :=
sorry

end NUMINAMATH_CALUDE_modified_grid_perimeter_l2385_238570


namespace NUMINAMATH_CALUDE_missing_number_equation_l2385_238543

theorem missing_number_equation (x : ℤ) : 10010 - x * 3 * 2 = 9938 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l2385_238543


namespace NUMINAMATH_CALUDE_largest_divisor_of_p_squared_minus_q_squared_l2385_238569

theorem largest_divisor_of_p_squared_minus_q_squared (p q : ℤ) 
  (h_p_gt_q : p > q) 
  (h_p_odd : Odd p) 
  (h_q_even : Even q) : 
  (∀ (d : ℤ), d ∣ (p^2 - q^2) → d = 1 ∨ d = -1) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_p_squared_minus_q_squared_l2385_238569


namespace NUMINAMATH_CALUDE_prime_sum_product_l2385_238564

theorem prime_sum_product : ∃ p q : ℕ, 
  Nat.Prime p ∧ Nat.Prime q ∧ p + q = 97 ∧ p * q = 190 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l2385_238564


namespace NUMINAMATH_CALUDE_bus_meeting_problem_l2385_238524

theorem bus_meeting_problem (n k : ℕ) : n > 3 → 
  (n * (n - 1) * (2 * k - 1) = 600) → 
  ((n = 4 ∧ k = 13) ∨ (n = 5 ∧ k = 8)) := by
  sorry

end NUMINAMATH_CALUDE_bus_meeting_problem_l2385_238524


namespace NUMINAMATH_CALUDE_algebra_test_average_l2385_238583

theorem algebra_test_average (total_average : ℝ) (male_count : ℕ) (female_average : ℝ) (female_count : ℕ) :
  total_average = 90 →
  male_count = 8 →
  female_average = 92 →
  female_count = 28 →
  (total_average * (male_count + female_count) - female_average * female_count) / male_count = 83 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_average_l2385_238583


namespace NUMINAMATH_CALUDE_eastbound_plane_speed_l2385_238581

/-- Given two planes traveling in opposite directions, this theorem proves
    the speed of the eastbound plane given the conditions of the problem. -/
theorem eastbound_plane_speed
  (time : ℝ)
  (westbound_speed : ℝ)
  (total_distance : ℝ)
  (h_time : time = 3.5)
  (h_westbound : westbound_speed = 275)
  (h_distance : total_distance = 2100) :
  ∃ (eastbound_speed : ℝ),
    eastbound_speed = 325 ∧
    (eastbound_speed + westbound_speed) * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_eastbound_plane_speed_l2385_238581


namespace NUMINAMATH_CALUDE_towel_area_decrease_l2385_238526

theorem towel_area_decrease (length width : ℝ) (h1 : length > 0) (h2 : width > 0) :
  let new_length := 0.9 * length
  let new_width := 0.8 * width
  let original_area := length * width
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.28 := by
sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l2385_238526


namespace NUMINAMATH_CALUDE_balance_balls_l2385_238545

/-- Given balance conditions between different colored balls, prove the number of blue balls needed to balance a specific combination. -/
theorem balance_balls (red blue orange purple : ℚ) 
  (h1 : 4 * red = 8 * blue)
  (h2 : 3 * orange = 7 * blue)
  (h3 : 8 * blue = 6 * purple) :
  5 * red + 3 * orange + 4 * purple = 67/3 * blue := by
  sorry

end NUMINAMATH_CALUDE_balance_balls_l2385_238545


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2385_238528

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1,
    if they have the same asymptotes, then M = 225/16 -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) →
  (∀ x y : ℝ, y = 4/3 * x ↔ y = 5/Real.sqrt M * x) →
  M = 225/16 := by
sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2385_238528


namespace NUMINAMATH_CALUDE_line_through_origin_and_point_l2385_238504

/-- A line passing through two points (0,0) and (-4,3) has the function expression y = -3/4 * x -/
theorem line_through_origin_and_point (x y : ℝ) : 
  (0 : ℝ) = 0 * x + y ∧ 3 = -4 * (-3/4) + y → y = -3/4 * x := by
sorry

end NUMINAMATH_CALUDE_line_through_origin_and_point_l2385_238504


namespace NUMINAMATH_CALUDE_min_values_proof_l2385_238560

theorem min_values_proof (a b m x : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hx : x > 2) :
  (a + b ≥ 2 * Real.sqrt (a * b)) ∧
  (a + b = 2 * Real.sqrt (a * b) ↔ a = b) →
  ((m + 1 / m ≥ 2) ∧ (∃ m₀ > 0, m₀ + 1 / m₀ = 2)) ∧
  ((x^2 + x - 5) / (x - 2) ≥ 7 ∧ (∃ x₀ > 2, (x₀^2 + x₀ - 5) / (x₀ - 2) = 7)) := by
  sorry

end NUMINAMATH_CALUDE_min_values_proof_l2385_238560


namespace NUMINAMATH_CALUDE_fourier_series_sum_l2385_238523

open Real

noncomputable def y (x : ℝ) : ℝ := x * cos x

theorem fourier_series_sum : 
  ∃ (S : ℝ), S = ∑' k, (4 * k^2 + 1) / (4 * k^2 - 1)^2 ∧ S = π^2 / 8 + 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fourier_series_sum_l2385_238523


namespace NUMINAMATH_CALUDE_smallest_d_for_divisibility_l2385_238593

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def digit_sum (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

def number (d : ℕ) : ℕ := 547000 + d * 1000 + 6

theorem smallest_d_for_divisibility :
  ∃ (d : ℕ), d = 2 ∧ 
  is_divisible_by_3 (number d) ∧ 
  ∀ (k : ℕ), k < d → ¬is_divisible_by_3 (number k) := by
  sorry

#check smallest_d_for_divisibility

end NUMINAMATH_CALUDE_smallest_d_for_divisibility_l2385_238593


namespace NUMINAMATH_CALUDE_kayak_rental_cost_l2385_238542

/-- Represents the daily rental business for canoes and kayaks -/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_count : ℕ
  kayak_count : ℕ
  total_revenue : ℕ

/-- The rental business satisfies the given conditions -/
def valid_rental_business (rb : RentalBusiness) : Prop :=
  rb.canoe_cost = 9 ∧
  rb.canoe_count = rb.kayak_count + 6 ∧
  4 * rb.kayak_count = 3 * rb.canoe_count ∧
  rb.total_revenue = rb.canoe_cost * rb.canoe_count + rb.kayak_cost * rb.kayak_count ∧
  rb.total_revenue = 432

/-- The theorem stating that under the given conditions, the kayak rental cost is $12 per day -/
theorem kayak_rental_cost (rb : RentalBusiness) 
  (h : valid_rental_business rb) : rb.kayak_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_kayak_rental_cost_l2385_238542


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2385_238562

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/5 < 3/2 ↔ x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l2385_238562


namespace NUMINAMATH_CALUDE_november_december_revenue_ratio_l2385_238575

/-- Proves that the revenue in November is 2/5 of the revenue in December given the conditions --/
theorem november_december_revenue_ratio
  (revenue : Fin 3 → ℝ)  -- revenue function for 3 months (0: November, 1: December, 2: January)
  (h1 : revenue 2 = (1/5) * revenue 0)  -- January revenue is 1/5 of November revenue
  (h2 : revenue 1 = (25/6) * ((revenue 0 + revenue 2) / 2))  -- December revenue condition
  : revenue 0 = (2/5) * revenue 1 := by
  sorry

#check november_december_revenue_ratio

end NUMINAMATH_CALUDE_november_december_revenue_ratio_l2385_238575


namespace NUMINAMATH_CALUDE_f_properties_l2385_238537

noncomputable def f (b c x : ℝ) : ℝ := |x| * x + b * x + c

theorem f_properties (b c : ℝ) :
  (∀ x y : ℝ, x < y → b > 0 → f b c x < f b c y) ∧
  (∀ x : ℝ, f b c x = f b c (-x)) ∧
  (∃ b c : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f b c x₁ = 0 ∧ f b c x₂ = 0 ∧ f b c x₃ = 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2385_238537


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2385_238522

/-- Given three real numbers forming an arithmetic sequence with sum 12,
    and their translations forming a geometric sequence,
    prove that the only solutions are (1, 4, 7) and (10, 4, -2) -/
theorem arithmetic_geometric_sequence_problem (a b c : ℝ) : 
  (∃ d : ℝ, b - a = d ∧ c - b = d) →  -- arithmetic sequence condition
  (a + b + c = 12) →                  -- sum condition
  (∃ r : ℝ, (b + 2) = (a + 2) * r ∧ (c + 5) = (b + 2) * r) →  -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l2385_238522


namespace NUMINAMATH_CALUDE_dereks_age_l2385_238573

/-- Given that Charlie's age is four times Derek's age, Emily is five years older than Derek,
    and Charlie and Emily are twins, prove that Derek is 5/3 years old. -/
theorem dereks_age (charlie emily derek : ℝ)
    (h1 : charlie = 4 * derek)
    (h2 : emily = derek + 5)
    (h3 : charlie = emily) :
    derek = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_dereks_age_l2385_238573


namespace NUMINAMATH_CALUDE_pen_price_relationship_l2385_238530

/-- Given a box of pens with a selling price of $16 and containing 10 pens,
    prove that the relationship between the selling price of one pen (y)
    and the number of pens (x) is y = 1.6x. -/
theorem pen_price_relationship (box_price : ℝ) (pens_per_box : ℕ) (x y : ℝ) :
  box_price = 16 →
  pens_per_box = 10 →
  y = (box_price / pens_per_box) * x →
  y = 1.6 * x :=
by
  sorry


end NUMINAMATH_CALUDE_pen_price_relationship_l2385_238530


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2385_238505

theorem pure_imaginary_complex_number (a : ℝ) : 
  (a^3 - a = 0 ∧ a/(1-a) ≠ 0) → a = -1 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2385_238505


namespace NUMINAMATH_CALUDE_lunks_needed_for_apples_l2385_238587

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

end NUMINAMATH_CALUDE_lunks_needed_for_apples_l2385_238587


namespace NUMINAMATH_CALUDE_perimeter_is_64_l2385_238533

/-- A structure formed by nine congruent squares -/
structure SquareStructure where
  /-- The side length of each square in the structure -/
  side_length : ℝ
  /-- The total area of the structure is 576 square centimeters -/
  total_area_eq : side_length ^ 2 * 9 = 576

/-- The perimeter of the square structure -/
def perimeter (s : SquareStructure) : ℝ :=
  8 * s.side_length

/-- Theorem stating that the perimeter of the structure is 64 centimeters -/
theorem perimeter_is_64 (s : SquareStructure) : perimeter s = 64 := by
  sorry

#check perimeter_is_64

end NUMINAMATH_CALUDE_perimeter_is_64_l2385_238533


namespace NUMINAMATH_CALUDE_product_112_54_l2385_238502

theorem product_112_54 : 112 * 54 = 6048 := by
  sorry

end NUMINAMATH_CALUDE_product_112_54_l2385_238502


namespace NUMINAMATH_CALUDE_xiao_ming_correct_count_l2385_238512

/-- Represents a math problem with a given answer --/
structure MathProblem where
  given_answer : Int
  correct_answer : Int

/-- Checks if a math problem is answered correctly --/
def is_correct (problem : MathProblem) : Bool :=
  problem.given_answer = problem.correct_answer

/-- Counts the number of correctly answered problems --/
def count_correct (problems : List MathProblem) : Nat :=
  (problems.filter is_correct).length

/-- The list of math problems Xiao Ming solved --/
def xiao_ming_problems : List MathProblem := [
  { given_answer := 0, correct_answer := -4 },
  { given_answer := -4, correct_answer := 0 },
  { given_answer := -4, correct_answer := -4 }
]

theorem xiao_ming_correct_count :
  count_correct xiao_ming_problems = 1 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_correct_count_l2385_238512


namespace NUMINAMATH_CALUDE_continuity_at_3_l2385_238588

def f (x : ℝ) : ℝ := -2 * x^2 - 4

theorem continuity_at_3 :
  ∀ ε > 0, ∃ δ > 0, δ = ε / 12 ∧
  ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε := by
  sorry

end NUMINAMATH_CALUDE_continuity_at_3_l2385_238588


namespace NUMINAMATH_CALUDE_polygon_sides_l2385_238501

theorem polygon_sides (n : ℕ) : 
  (((n - 2) * 180) / 360 : ℚ) = 3 / 1 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2385_238501


namespace NUMINAMATH_CALUDE_union_complement_equals_B_l2385_238594

def U : Set ℕ := {x | x < 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem union_complement_equals_B : B ∪ (U \ A) = B := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_B_l2385_238594


namespace NUMINAMATH_CALUDE_product_of_fractions_l2385_238558

theorem product_of_fractions (p : ℝ) (hp : p ≠ 0) :
  (p^3 + 4*p^2 + 10*p + 12) / (p^3 - p^2 + 2*p + 16) *
  (p^3 - 3*p^2 + 8*p) / (p^2 + 2*p + 6) =
  ((p^3 + 4*p^2 + 10*p + 12) * (p^3 - 3*p^2 + 8*p)) /
  ((p^3 - p^2 + 2*p + 16) * (p^2 + 2*p + 6)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2385_238558


namespace NUMINAMATH_CALUDE_probability_highest_is_four_value_l2385_238500

def number_of_balls : ℕ := 5
def balls_drawn : ℕ := 3

def probability_highest_is_four : ℚ :=
  (Nat.choose (number_of_balls - 2) (balls_drawn - 1)) / (Nat.choose number_of_balls balls_drawn)

theorem probability_highest_is_four_value : 
  probability_highest_is_four = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_highest_is_four_value_l2385_238500


namespace NUMINAMATH_CALUDE_triangle_segment_relation_l2385_238541

/-- Given a triangle ABC with point D on AB and point E on AD, 
    prove the relation for FC where F is on AC. -/
theorem triangle_segment_relation 
  (A B C D E F : ℝ × ℝ) 
  (h1 : dist D C = 6)
  (h2 : dist C B = 9)
  (h3 : dist A B = 1/5 * dist A D)
  (h4 : dist E D = 2/3 * dist A D) :
  dist F C = (dist E D * dist C A) / dist D A :=
sorry

end NUMINAMATH_CALUDE_triangle_segment_relation_l2385_238541


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l2385_238553

theorem smallest_four_digit_divisible_by_37 : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧  -- four-digit number
  n % 37 = 0 ∧              -- divisible by 37
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) ∧ m % 37 = 0 → n ≤ m) ∧  -- smallest such number
  n = 1036 :=               -- the answer is 1036
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_37_l2385_238553


namespace NUMINAMATH_CALUDE_triangle_equilateral_l2385_238597

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

end NUMINAMATH_CALUDE_triangle_equilateral_l2385_238597


namespace NUMINAMATH_CALUDE_smaller_number_value_l2385_238571

theorem smaller_number_value (s l : ℤ) : 
  (l - s = 28) → 
  (l + 13 = 2 * (s + 13)) → 
  s = 15 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_value_l2385_238571


namespace NUMINAMATH_CALUDE_odd_shift_three_l2385_238585

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem odd_shift_three (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1))) 
  (h2 : is_odd (λ x => f (x - 1))) : 
  is_odd (λ x => f (x + 3)) := by
sorry

end NUMINAMATH_CALUDE_odd_shift_three_l2385_238585


namespace NUMINAMATH_CALUDE_tank_capacity_l2385_238535

theorem tank_capacity : 
  ∀ (initial_fraction final_fraction added_amount total_capacity : ℚ),
  initial_fraction = 1/4 →
  final_fraction = 2/3 →
  added_amount = 160 →
  (final_fraction - initial_fraction) * total_capacity = added_amount →
  total_capacity = 384 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_l2385_238535


namespace NUMINAMATH_CALUDE_max_oranges_removal_l2385_238578

/-- A triangular grid of length n -/
structure TriangularGrid (n : ℕ) where
  (n_pos : 0 < n)
  (n_not_div_3 : ¬ 3 ∣ n)

/-- The total number of oranges in a triangular grid -/
def totalOranges (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

/-- A good triple of oranges -/
structure GoodTriple (n : ℕ) where
  (isValid : Bool)

/-- The maximum number of oranges that can be removed -/
def maxRemovableOranges (n : ℕ) : ℕ := totalOranges n - 3

theorem max_oranges_removal (n : ℕ) (grid : TriangularGrid n) :
  maxRemovableOranges n = totalOranges n - 3 := by sorry

end NUMINAMATH_CALUDE_max_oranges_removal_l2385_238578


namespace NUMINAMATH_CALUDE_intercept_sum_l2385_238521

-- Define the two lines
def line1 (x y : ℝ) : Prop := 20 * x + 16 * y - 40 = 0
def line2 (x y : ℝ) : Prop := 20 * x + 16 * y - 64 = 0

-- Define x-intercept of line1
def x_intercept : ℝ := 2

-- Define y-intercept of line2
def y_intercept : ℝ := 4

-- Theorem statement
theorem intercept_sum : x_intercept + y_intercept = 6 := by sorry

end NUMINAMATH_CALUDE_intercept_sum_l2385_238521


namespace NUMINAMATH_CALUDE_negation_equivalence_l2385_238599

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 + 2*x + 2 ≤ 0) ↔ 
  (∀ x : ℝ, x > 1 → x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2385_238599


namespace NUMINAMATH_CALUDE_simplify_expression_l2385_238534

theorem simplify_expression (y : ℝ) : 8*y - 3 + 2*y + 15 = 10*y + 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2385_238534


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l2385_238579

-- Define the initial quantities and served items
def initial_hamburgers : ℕ := 25
def first_hour_hamburgers : ℕ := 12
def second_hour_hamburgers : ℕ := 6

-- Define the function to calculate remaining hamburgers
def remaining_hamburgers : ℕ := 
  initial_hamburgers - (first_hour_hamburgers + second_hour_hamburgers)

-- Theorem statement
theorem hamburgers_left_over : 
  remaining_hamburgers = 7 :=
by sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l2385_238579


namespace NUMINAMATH_CALUDE_kids_difference_l2385_238507

/-- The number of kids Julia played with on different days of the week. -/
structure KidsPlayed where
  monday : ℕ
  wednesday : ℕ

/-- Theorem stating the difference in number of kids played with between Monday and Wednesday. -/
theorem kids_difference (k : KidsPlayed) (h1 : k.monday = 6) (h2 : k.wednesday = 4) :
  k.monday - k.wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l2385_238507


namespace NUMINAMATH_CALUDE_absolute_value_not_positive_l2385_238532

theorem absolute_value_not_positive (x : ℚ) : |4*x - 8| ≤ 0 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_positive_l2385_238532


namespace NUMINAMATH_CALUDE_paco_cookies_l2385_238536

theorem paco_cookies (initial_cookies : ℕ) : initial_cookies = 7 :=
by
  -- Define the number of cookies eaten initially
  let initial_eaten : ℕ := 5
  -- Define the number of cookies bought
  let bought : ℕ := 3
  -- Define the number of cookies eaten after buying
  let later_eaten : ℕ := bought + 2
  -- Assert that all cookies were eaten
  have h : initial_cookies - initial_eaten + bought - later_eaten = 0 := by sorry
  -- Prove that initial_cookies = 7
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2385_238536


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2385_238508

theorem max_value_on_ellipse (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ellipse : 4 * x^2 + 9 * y^2 = 36) : 
  ∀ u v : ℝ, u > 0 → v > 0 → 4 * u^2 + 9 * v^2 = 36 → x + 2*y ≤ 5 ∧ x + 2*y = 5 → u + 2*v ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2385_238508


namespace NUMINAMATH_CALUDE_min_value_of_function_l2385_238555

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  4 * x + 25 / x ≥ 20 ∧ ∃ y > 0, 4 * y + 25 / y = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2385_238555


namespace NUMINAMATH_CALUDE_jackie_exercise_hours_l2385_238525

/-- Represents Jackie's daily schedule --/
structure DailySchedule where
  total_hours : ℕ
  work_hours : ℕ
  sleep_hours : ℕ
  free_hours : ℕ

/-- Calculates the number of hours Jackie spends exercising --/
def exercise_hours (schedule : DailySchedule) : ℕ :=
  schedule.total_hours - (schedule.work_hours + schedule.sleep_hours + schedule.free_hours)

/-- Theorem stating that Jackie spends 3 hours exercising --/
theorem jackie_exercise_hours :
  let schedule : DailySchedule := {
    total_hours := 24,
    work_hours := 8,
    sleep_hours := 8,
    free_hours := 5
  }
  exercise_hours schedule = 3 := by sorry

end NUMINAMATH_CALUDE_jackie_exercise_hours_l2385_238525


namespace NUMINAMATH_CALUDE_min_packs_needed_l2385_238516

/-- Represents the number of cans in each pack type -/
def pack_sizes : Fin 3 → ℕ
  | 0 => 8
  | 1 => 15
  | 2 => 18

/-- The total number of cans needed -/
def total_cans : ℕ := 95

/-- The maximum number of packs allowed for each type -/
def max_packs : ℕ := 4

/-- A function to calculate the total number of cans from a given combination of packs -/
def total_from_packs (x y z : ℕ) : ℕ :=
  x * pack_sizes 0 + y * pack_sizes 1 + z * pack_sizes 2

/-- The main theorem to prove -/
theorem min_packs_needed :
  ∃ (x y z : ℕ),
    x ≤ max_packs ∧ y ≤ max_packs ∧ z ≤ max_packs ∧
    total_from_packs x y z = total_cans ∧
    x + y + z = 6 ∧
    (∀ (a b c : ℕ),
      a ≤ max_packs → b ≤ max_packs → c ≤ max_packs →
      total_from_packs a b c = total_cans →
      a + b + c ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_min_packs_needed_l2385_238516


namespace NUMINAMATH_CALUDE_expression_eval_zero_l2385_238557

theorem expression_eval_zero (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 15 * a + 5) * (3 * a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_eval_zero_l2385_238557


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2385_238547

theorem cos_alpha_value (α : Real) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) :
  Real.cos α = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2385_238547


namespace NUMINAMATH_CALUDE_article_profit_percentage_l2385_238568

theorem article_profit_percentage (CP : ℝ) (G : ℝ) : 
  CP = 800 →
  (CP * 0.95) * 1.1 = CP * (1 + G / 100) - 4 →
  G = 5 := by
sorry

end NUMINAMATH_CALUDE_article_profit_percentage_l2385_238568


namespace NUMINAMATH_CALUDE_fraction_sum_power_six_l2385_238538

theorem fraction_sum_power_six : (5 / 3 : ℚ)^6 + (2 / 3 : ℚ)^6 = 15689 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_power_six_l2385_238538


namespace NUMINAMATH_CALUDE_sphere_radius_non_uniform_l2385_238550

/-- The radius of a sphere given its curved surface area in a non-uniform coordinate system -/
theorem sphere_radius_non_uniform (surface_area : ℝ) (k1 k2 k3 : ℝ) (h : surface_area = 64 * Real.pi) :
  ∃ (r : ℝ), r = 4 ∧ surface_area = 4 * Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_non_uniform_l2385_238550


namespace NUMINAMATH_CALUDE_lottery_winnings_l2385_238586

theorem lottery_winnings 
  (num_tickets : ℕ) 
  (winning_numbers_per_ticket : ℕ) 
  (total_winnings : ℕ) 
  (h1 : num_tickets = 3)
  (h2 : winning_numbers_per_ticket = 5)
  (h3 : total_winnings = 300) :
  total_winnings / (num_tickets * winning_numbers_per_ticket) = 20 :=
by sorry

end NUMINAMATH_CALUDE_lottery_winnings_l2385_238586


namespace NUMINAMATH_CALUDE_opaque_arrangements_count_l2385_238510

/-- Represents a glass piece with one painted triangular section -/
structure GlassPiece where
  rotation : Fin 4  -- 0, 1, 2, 3 representing 0°, 90°, 180°, 270°

/-- Represents a stack of glass pieces -/
def GlassStack := List GlassPiece

/-- Checks if a given stack of glass pieces is completely opaque -/
def is_opaque (stack : GlassStack) : Bool :=
  sorry

/-- Counts the number of opaque arrangements for 5 glass pieces -/
def count_opaque_arrangements : Nat :=
  sorry

/-- The main theorem stating the correct number of opaque arrangements -/
theorem opaque_arrangements_count :
  count_opaque_arrangements = 7200 :=
sorry

end NUMINAMATH_CALUDE_opaque_arrangements_count_l2385_238510


namespace NUMINAMATH_CALUDE_vine_paint_time_l2385_238517

/-- Time to paint different flowers and total painting time -/
def paint_problem (lily_time rose_time orchid_time vine_time : ℕ) 
  (total_time lily_count rose_count orchid_count vine_count : ℕ) : Prop :=
  lily_time * lily_count + rose_time * rose_count + 
  orchid_time * orchid_count + vine_time * vine_count = total_time

/-- Theorem stating the time to paint a vine -/
theorem vine_paint_time : 
  ∃ (vine_time : ℕ), 
    paint_problem 5 7 3 vine_time 213 17 10 6 20 ∧ 
    vine_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_vine_paint_time_l2385_238517


namespace NUMINAMATH_CALUDE_increasing_odd_sum_nonpositive_l2385_238567

/-- A function f: ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function f: ℝ → ℝ is odd if for all x ∈ ℝ, f(-x) = -f(x) -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- Theorem: If f is an increasing and odd function on ℝ, and a and b are real numbers
    such that a + b ≤ 0, then f(a) + f(b) ≤ 0 -/
theorem increasing_odd_sum_nonpositive
  (f : ℝ → ℝ) (hf_inc : IsIncreasing f) (hf_odd : IsOdd f)
  (a b : ℝ) (hab : a + b ≤ 0) :
  f a + f b ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_increasing_odd_sum_nonpositive_l2385_238567


namespace NUMINAMATH_CALUDE_women_count_l2385_238554

/-- Represents the work done by one woman in one day -/
def W : ℝ := sorry

/-- Represents the work done by one child in one day -/
def C : ℝ := sorry

/-- Represents the number of women working initially -/
def x : ℝ := sorry

/-- The total work to be completed -/
def total_work : ℝ := sorry

theorem women_count : x = 10 := by
  have h1 : 5 * x * W = total_work := sorry
  have h2 : 100 * C = total_work := sorry
  have h3 : 5 * (5 * W + 10 * C) = total_work := sorry
  sorry

end NUMINAMATH_CALUDE_women_count_l2385_238554


namespace NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l2385_238596

theorem ceiling_neg_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_seven_fourths_squared_l2385_238596


namespace NUMINAMATH_CALUDE_sum_of_integers_l2385_238513

theorem sum_of_integers (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = y.val * z.val + x.val)
  (h2 : y.val * z.val + x.val = x.val * z.val + y.val)
  (h3 : x.val * z.val + y.val = 55)
  (h4 : Even x.val ∨ Even y.val ∨ Even z.val) :
  x.val + y.val + z.val = 56 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2385_238513


namespace NUMINAMATH_CALUDE_power_of_power_l2385_238556

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2385_238556


namespace NUMINAMATH_CALUDE_ten_point_circle_triangles_l2385_238561

/-- The number of triangles that can be formed from n distinct points on a circle's circumference -/
def total_triangles (n : ℕ) : ℕ := Nat.choose n 3

/-- The number of triangles where one side subtends an arc greater than 180 degrees -/
def long_arc_triangles (n : ℕ) : ℕ := 2 * n

/-- The number of valid triangles that can be formed from n distinct points on a circle's circumference,
    where no side subtends an arc greater than 180 degrees -/
def valid_triangles (n : ℕ) : ℕ := total_triangles n - long_arc_triangles n

theorem ten_point_circle_triangles :
  valid_triangles 10 = 100 := by sorry

end NUMINAMATH_CALUDE_ten_point_circle_triangles_l2385_238561


namespace NUMINAMATH_CALUDE_no_solution_iff_v_eq_neg_one_l2385_238590

/-- The system of equations has no solution if and only if v = -1 -/
theorem no_solution_iff_v_eq_neg_one (v : ℝ) :
  (∀ x y z : ℝ, (x + y + z = v ∧ x + v*y + z = v ∧ x + y + v^2*z = v^2) → False) ↔ v = -1 :=
sorry

end NUMINAMATH_CALUDE_no_solution_iff_v_eq_neg_one_l2385_238590


namespace NUMINAMATH_CALUDE_coffee_lasts_40_days_l2385_238515

/-- The number of days coffee will last given the amount bought, brewing capacity, and daily consumption. -/
def coffee_duration (pounds_bought : ℕ) (cups_per_pound : ℕ) (cups_per_day : ℕ) : ℕ :=
  (pounds_bought * cups_per_pound) / cups_per_day

/-- Theorem stating that under the given conditions, the coffee will last 40 days. -/
theorem coffee_lasts_40_days :
  coffee_duration 3 40 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_coffee_lasts_40_days_l2385_238515


namespace NUMINAMATH_CALUDE_cindy_friends_l2385_238566

/-- Calculates the number of friends Cindy gives envelopes to -/
def num_friends (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (remaining_envelopes : ℕ) : ℕ :=
  (initial_envelopes - remaining_envelopes) / envelopes_per_friend

/-- Proves that Cindy gives envelopes to 5 friends -/
theorem cindy_friends : num_friends 37 3 22 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cindy_friends_l2385_238566


namespace NUMINAMATH_CALUDE_equal_even_odd_probability_l2385_238518

/-- The number of dice being rolled -/
def num_dice : ℕ := 8

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The probability of rolling 8 six-sided dice and getting an equal number of even and odd results -/
def prob_equal_even_odd : ℚ := 35/128

theorem equal_even_odd_probability :
  (Nat.choose num_dice (num_dice / 2)) * (prob_even ^ num_dice) = prob_equal_even_odd := by
  sorry

end NUMINAMATH_CALUDE_equal_even_odd_probability_l2385_238518


namespace NUMINAMATH_CALUDE_equal_interest_principal_second_amount_calculation_l2385_238506

/-- Given two investments with equal interest, calculate the principal of the second investment -/
theorem equal_interest_principal (p₁ r₁ t₁ r₂ t₂ : ℚ) (hp₁ : p₁ > 0) (hr₁ : r₁ > 0) (ht₁ : t₁ > 0) (hr₂ : r₂ > 0) (ht₂ : t₂ > 0) :
  p₁ * r₁ * t₁ = (p₁ * r₁ * t₁ / (r₂ * t₂)) * r₂ * t₂ :=
by sorry

/-- The second amount that produces the same interest as Rs 200 at 10% for 12 years, when invested at 12% for 5 years, is Rs 400 -/
theorem second_amount_calculation :
  let p₁ : ℚ := 200
  let r₁ : ℚ := 10 / 100
  let t₁ : ℚ := 12
  let r₂ : ℚ := 12 / 100
  let t₂ : ℚ := 5
  (p₁ * r₁ * t₁ / (r₂ * t₂)) = 400 :=
by sorry

end NUMINAMATH_CALUDE_equal_interest_principal_second_amount_calculation_l2385_238506


namespace NUMINAMATH_CALUDE_third_quiz_score_l2385_238511

theorem third_quiz_score (score1 score2 score3 : ℕ) : 
  score1 = 91 → 
  score2 = 90 → 
  (score1 + score2 + score3) / 3 = 91 → 
  score3 = 92 := by
sorry

end NUMINAMATH_CALUDE_third_quiz_score_l2385_238511


namespace NUMINAMATH_CALUDE_reflection_coordinates_sum_l2385_238565

/-- Given a point C at (3, y+4) and its reflection D over the y-axis, with y = 2,
    the sum of all four coordinates of C and D is equal to 12. -/
theorem reflection_coordinates_sum :
  let y : ℝ := 2
  let C : ℝ × ℝ := (3, y + 4)
  let D : ℝ × ℝ := (-C.1, C.2)  -- Reflection over y-axis
  C.1 + C.2 + D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_reflection_coordinates_sum_l2385_238565


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2385_238509

theorem sum_of_reciprocals (a b : ℝ) (h1 : a ≠ b) (h2 : a/b + a = b/a + b) : 1/a + 1/b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2385_238509


namespace NUMINAMATH_CALUDE_bart_mixtape_length_l2385_238551

/-- Calculates the total length of a mixtape in minutes -/
def mixtape_length (songs_side1 : ℕ) (songs_side2 : ℕ) (song_duration : ℕ) : ℕ :=
  (songs_side1 + songs_side2) * song_duration

/-- Proves that the total length of Bart's mixtape is 40 minutes -/
theorem bart_mixtape_length :
  mixtape_length 6 4 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bart_mixtape_length_l2385_238551


namespace NUMINAMATH_CALUDE_point_displacement_on_line_l2385_238559

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of the line x = (y / 2) - (2 / 5) -/
def onLine (p : Point) : Prop :=
  p.x = p.y / 2 - 2 / 5

theorem point_displacement_on_line (m n p : ℝ) :
  onLine ⟨m, n⟩ ∧ onLine ⟨m + p, n + 4⟩ → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_displacement_on_line_l2385_238559


namespace NUMINAMATH_CALUDE_total_average_marks_l2385_238544

/-- Given two classes with different numbers of students and average marks,
    calculate the total average marks of all students in both classes. -/
theorem total_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 > 0 → n2 > 0 →
  (n1 : ℚ) * avg1 + (n2 : ℚ) * avg2 / ((n1 : ℚ) + (n2 : ℚ)) =
  ((39 : ℚ) * 45 + (35 : ℚ) * 70) / ((39 : ℚ) + (35 : ℚ)) :=
by
  sorry

#eval ((39 : ℚ) * 45 + (35 : ℚ) * 70) / ((39 : ℚ) + (35 : ℚ))

end NUMINAMATH_CALUDE_total_average_marks_l2385_238544


namespace NUMINAMATH_CALUDE_rectangle_area_l2385_238574

theorem rectangle_area (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l^2 + w^2 = d^2) :
  l * w = (10 / 29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2385_238574


namespace NUMINAMATH_CALUDE_share_multiple_l2385_238552

theorem share_multiple (a b c k : ℚ) : 
  a + b + c = 585 →
  4 * a = 6 * b →
  4 * a = k * c →
  c = 260 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_share_multiple_l2385_238552


namespace NUMINAMATH_CALUDE_squared_differences_inequality_l2385_238531

theorem squared_differences_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  min ((a - b)^2) (min ((b - c)^2) ((c - a)^2)) ≤ (a^2 + b^2 + c^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_squared_differences_inequality_l2385_238531


namespace NUMINAMATH_CALUDE_alternating_squares_sum_l2385_238595

theorem alternating_squares_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 242 := by
  sorry

end NUMINAMATH_CALUDE_alternating_squares_sum_l2385_238595


namespace NUMINAMATH_CALUDE_teena_loe_distance_l2385_238582

theorem teena_loe_distance (teena_speed loe_speed : ℝ) (time : ℝ) (ahead_distance : ℝ) :
  teena_speed = 55 →
  loe_speed = 40 →
  time = 1.5 →
  ahead_distance = 15 →
  ∃ initial_distance : ℝ,
    initial_distance = (teena_speed * time - loe_speed * time - ahead_distance) ∧
    initial_distance = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_teena_loe_distance_l2385_238582


namespace NUMINAMATH_CALUDE_function_symmetry_l2385_238576

theorem function_symmetry (f : ℝ → ℝ) (h : f 0 = 1) : f (4 - 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l2385_238576


namespace NUMINAMATH_CALUDE_proposition_truth_values_l2385_238572

theorem proposition_truth_values (p q : Prop) (hp : p) (hq : ¬q) :
  (p ∨ q) ∧ ¬(¬p) ∧ ¬(p ∧ q) ∧ ¬(¬p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l2385_238572


namespace NUMINAMATH_CALUDE_tens_digit_of_square_even_for_odd_numbers_up_to_99_l2385_238584

/-- The tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- Predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

theorem tens_digit_of_square_even_for_odd_numbers_up_to_99 :
  ∀ n : ℕ, n ≤ 99 → isOdd n → Even (tensDigit (n^2)) := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_square_even_for_odd_numbers_up_to_99_l2385_238584


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2385_238514

/-- Four distinct positive real numbers form an arithmetic sequence -/
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

theorem arithmetic_sequence_inequality (a b c d : ℝ) 
  (h : is_arithmetic_sequence a b c d) : (a + d) / 2 > Real.sqrt (b * c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2385_238514


namespace NUMINAMATH_CALUDE_age_difference_l2385_238591

theorem age_difference (c d : ℕ) (hc : c < 10) (hd : d < 10) 
  (h : 10 * c + d + 10 = 3 * (10 * d + c + 10)) :
  (10 * c + d) - (10 * d + c) = 54 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2385_238591


namespace NUMINAMATH_CALUDE_rectangle_areas_sum_l2385_238577

theorem rectangle_areas_sum : 
  let rectangles : List (ℕ × ℕ) := [(2, 1), (2, 9), (2, 25), (2, 49), (2, 81), (2, 121)]
  let areas := rectangles.map (fun (w, l) => w * l)
  areas.sum = 572 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_areas_sum_l2385_238577


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l2385_238549

theorem cosine_sum_simplification (x : ℝ) (k : ℤ) :
  Real.cos ((6 * k + 1) * π / 3 + x) + Real.cos ((6 * k - 1) * π / 3 + x) = Real.cos x :=
by sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l2385_238549


namespace NUMINAMATH_CALUDE_office_chairs_probability_l2385_238529

theorem office_chairs_probability (black_chairs brown_chairs : ℕ) 
  (h1 : black_chairs = 15)
  (h2 : brown_chairs = 18) :
  let total_chairs := black_chairs + brown_chairs
  let prob_same_color := (black_chairs * (black_chairs - 1) + brown_chairs * (brown_chairs - 1)) / (total_chairs * (total_chairs - 1))
  prob_same_color = 43 / 88 := by
sorry

end NUMINAMATH_CALUDE_office_chairs_probability_l2385_238529


namespace NUMINAMATH_CALUDE_average_visitors_theorem_l2385_238592

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (daysInMonth : Nat) (sundayVisitors : Nat) (otherDayVisitors : Nat) : Nat :=
  let numSundays := (daysInMonth + 6) / 7
  let numOtherDays := daysInMonth - numSundays
  let totalVisitors := numSundays * sundayVisitors + numOtherDays * otherDayVisitors
  totalVisitors / daysInMonth

theorem average_visitors_theorem :
  averageVisitors 30 660 240 = 296 := by
  sorry

end NUMINAMATH_CALUDE_average_visitors_theorem_l2385_238592


namespace NUMINAMATH_CALUDE_symmetric_circle_l2385_238539

/-- Given a circle and a line of symmetry, find the equation of the symmetric circle -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, (x + 2)^2 + y^2 = 2016) →  -- Original circle equation
  (∀ x y, x - y + 1 = 0) →           -- Line of symmetry
  (∀ x y, (x + 1)^2 + (y + 1)^2 = 2016) -- Symmetric circle equation
:= by sorry

end NUMINAMATH_CALUDE_symmetric_circle_l2385_238539


namespace NUMINAMATH_CALUDE_total_amount_theorem_l2385_238598

/-- The total amount spent on cows and goats -/
def total_amount_spent (num_cows num_goats avg_price_cow avg_price_goat : ℕ) : ℕ :=
  num_cows * avg_price_cow + num_goats * avg_price_goat

/-- Theorem: The total amount spent on 2 cows and 10 goats is 1500 rupees -/
theorem total_amount_theorem :
  total_amount_spent 2 10 400 70 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l2385_238598


namespace NUMINAMATH_CALUDE_black_ribbon_count_l2385_238503

theorem black_ribbon_count (total : ℕ) (silver : ℕ) : 
  silver = 40 →
  (1 : ℚ) / 4 + 1 / 3 + 1 / 6 + 1 / 12 + (silver : ℚ) / total = 1 →
  (total : ℚ) / 12 = 20 :=
by sorry

end NUMINAMATH_CALUDE_black_ribbon_count_l2385_238503


namespace NUMINAMATH_CALUDE_range_of_m_l2385_238546

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/3)^(-x) - 2 else 2 * Real.log (-x) / Real.log 3

theorem range_of_m (m : ℝ) (h : f m > 1) :
  m ∈ Set.Ioi 1 ∪ Set.Iic (-Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2385_238546


namespace NUMINAMATH_CALUDE_exists_circumcircle_equation_l2385_238589

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

end NUMINAMATH_CALUDE_exists_circumcircle_equation_l2385_238589


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2385_238548

/-- The number of ways three people can sit in a row of six chairs -/
def seating_arrangements : ℕ := 6 * 5 * 4

/-- Theorem stating that the number of seating arrangements is 120 -/
theorem seating_arrangements_count : seating_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2385_238548


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2385_238563

theorem expand_and_simplify (x y : ℝ) : (-x + y) * (-x - y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2385_238563


namespace NUMINAMATH_CALUDE_derivative_of_f_l2385_238520

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 3 * x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2385_238520


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l2385_238519

/-- Represents the sum of elements in the nth set of a sequence of sets,
    where each set contains consecutive integers and has one more element than the previous set. -/
def S (n : ℕ) : ℕ :=
  let first := (n * (n - 1)) / 2 + 1
  let last := first + n - 1
  n * (first + last) / 2

/-- The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l2385_238519
