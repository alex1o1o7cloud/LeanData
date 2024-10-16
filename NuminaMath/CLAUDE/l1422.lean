import Mathlib

namespace NUMINAMATH_CALUDE_max_volume_corner_cut_box_l1422_142252

/-- The maximum volume of an open-top box formed by cutting identical squares from the corners of a rectangular cardboard -/
theorem max_volume_corner_cut_box (a b : ℝ) (ha : a = 10) (hb : b = 16) :
  let V := fun x => (a - 2*x) * (b - 2*x) * x
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧ x < b/2 ∧
    (∀ y, y > 0 → y < a/2 → y < b/2 → V y ≤ V x) ∧
    V x = 144 :=
sorry

end NUMINAMATH_CALUDE_max_volume_corner_cut_box_l1422_142252


namespace NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l1422_142291

-- Define the numbers
def a : Float := 45.378
def b : Float := 13.897
def c : Float := 29.4567

-- Define the sum
def sum : Float := a + b + c

-- Define a function to round to the nearest hundredth
def round_to_hundredth (x : Float) : Float :=
  (x * 100).round / 100

-- Theorem statement
theorem sum_rounded_to_hundredth :
  round_to_hundredth sum = 88.74 := by sorry

end NUMINAMATH_CALUDE_sum_rounded_to_hundredth_l1422_142291


namespace NUMINAMATH_CALUDE_unique_remainder_mod_10_l1422_142225

theorem unique_remainder_mod_10 : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_10_l1422_142225


namespace NUMINAMATH_CALUDE_isometric_figure_area_l1422_142250

/-- A horizontally placed figure with an isometric view -/
structure IsometricFigure where
  /-- The isometric view is an isosceles right triangle -/
  isIsoscelesRightTriangle : Prop
  /-- The legs of the isometric view triangle have length 1 -/
  legLength : ℝ
  /-- The area of the isometric view -/
  isometricArea : ℝ
  /-- The area of the original plane figure -/
  originalArea : ℝ

/-- 
  If a horizontally placed figure has an isometric view that is an isosceles right triangle 
  with legs of length 1, then the area of the original plane figure is √2.
-/
theorem isometric_figure_area 
  (fig : IsometricFigure) 
  (h1 : fig.isIsoscelesRightTriangle) 
  (h2 : fig.legLength = 1) 
  (h3 : fig.isometricArea = 1 / 2) : 
  fig.originalArea = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isometric_figure_area_l1422_142250


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l1422_142221

theorem johns_candy_store_spending (allowance : ℚ) :
  allowance = 4.8 →
  let arcade_spending := (3 / 5) * allowance
  let remaining_after_arcade := allowance - arcade_spending
  let toy_store_spending := (1 / 3) * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 1.28 := by
sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l1422_142221


namespace NUMINAMATH_CALUDE_chord_division_ratio_l1422_142212

/-- Given a circle with radius 11 and a chord of length 18 that intersects
    a diameter at a point 7 units from the center, prove that this point
    divides the chord in a ratio of either 2:1 or 1:2. -/
theorem chord_division_ratio (R : ℝ) (chord_length : ℝ) (center_to_intersection : ℝ)
    (h1 : R = 11)
    (h2 : chord_length = 18)
    (h3 : center_to_intersection = 7) :
    ∃ (x y : ℝ), (x + y = chord_length ∧ 
                 ((x / y = 2 ∧ y / x = 1/2) ∨ 
                  (x / y = 1/2 ∧ y / x = 2))) :=
by sorry

end NUMINAMATH_CALUDE_chord_division_ratio_l1422_142212


namespace NUMINAMATH_CALUDE_train_length_calculation_l1422_142237

/-- Given a train that crosses a platform and a signal pole, calculate its length. -/
theorem train_length_calculation
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (pole_crossing_time : ℝ)
  (h1 : platform_length = 200)
  (h2 : platform_crossing_time = 45)
  (h3 : pole_crossing_time = 30)
  : ∃ (train_length : ℝ),
    train_length / pole_crossing_time = (train_length + platform_length) / platform_crossing_time ∧
    train_length = 400 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1422_142237


namespace NUMINAMATH_CALUDE_real_y_condition_l1422_142215

theorem real_y_condition (x y : ℝ) : 
  (∃ y : ℝ, 3 * y^2 + 5 * x * y + x + 7 = 0) ↔ (x ≤ -6/5 ∨ x ≥ 14/5) :=
by sorry

end NUMINAMATH_CALUDE_real_y_condition_l1422_142215


namespace NUMINAMATH_CALUDE_volleyball_team_math_players_l1422_142245

theorem volleyball_team_math_players (total : ℕ) (physics : ℕ) (both : ℕ) (math : ℕ) : 
  total = 15 → 
  physics = 9 → 
  both = 4 → 
  physics + math - both = total → 
  math = 10 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_math_players_l1422_142245


namespace NUMINAMATH_CALUDE_point_identity_l1422_142297

/-- Given a point P(x, y) in the plane, prove that s^2 + c^2 = 1 where
    r is the distance from the origin to P,
    s = y/r,
    c = x/r,
    and c^2 = 4/9 -/
theorem point_identity (x y : ℝ) : 
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  c^2 = 4/9 →
  s^2 + c^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_identity_l1422_142297


namespace NUMINAMATH_CALUDE_flight_cost_a_to_b_l1422_142285

/-- Represents the cost of a flight between two cities -/
structure FlightCost where
  bookingFee : ℝ
  ratePerKm : ℝ

/-- Calculates the total cost of a flight -/
def calculateFlightCost (distance : ℝ) (cost : FlightCost) : ℝ :=
  cost.bookingFee + cost.ratePerKm * distance

/-- The problem statement -/
theorem flight_cost_a_to_b :
  let distanceAB : ℝ := 3500
  let flightCost : FlightCost := { bookingFee := 120, ratePerKm := 0.12 }
  calculateFlightCost distanceAB flightCost = 540 := by
  sorry


end NUMINAMATH_CALUDE_flight_cost_a_to_b_l1422_142285


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l1422_142258

theorem lcm_factor_proof (A B : ℕ+) (h_hcf : Nat.gcd A B = 25) 
  (h_lcm : ∃ X : ℕ+, Nat.lcm A B = 25 * X * 14) (h_A : A = 350) (h_order : A > B) : 
  ∃ X : ℕ+, Nat.lcm A B = 25 * X * 14 ∧ X = 1 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l1422_142258


namespace NUMINAMATH_CALUDE_not_proportional_l1422_142210

-- Define the properties of direct and inverse proportionality
def is_directly_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

def is_inversely_proportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x * x = k

-- Define the function representing y = 3x + 2
def f (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem not_proportional :
  ¬(is_directly_proportional f) ∧ ¬(is_inversely_proportional f) :=
by sorry

end NUMINAMATH_CALUDE_not_proportional_l1422_142210


namespace NUMINAMATH_CALUDE_edric_working_days_l1422_142251

/-- Calculates the number of working days per week given monthly salary, hours per day, and hourly rate -/
def working_days_per_week (monthly_salary : ℕ) (hours_per_day : ℕ) (hourly_rate : ℕ) : ℚ :=
  (monthly_salary : ℚ) / 4 / (hours_per_day * hourly_rate)

/-- Theorem: Given Edric's monthly salary, hours per day, and hourly rate, he works 6 days a week -/
theorem edric_working_days : working_days_per_week 576 8 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_edric_working_days_l1422_142251


namespace NUMINAMATH_CALUDE_richs_walk_total_distance_l1422_142294

/-- Calculates the total distance of Rich's walk --/
def richs_walk (segment1 segment2 segment5 : ℝ) : ℝ :=
  let segment3 := 2 * (segment1 + segment2)
  let segment4 := 1.5 * segment3
  let sum_to_5 := segment1 + segment2 + segment3 + segment4 + segment5
  let segment6 := 3 * sum_to_5
  let sum_to_6 := sum_to_5 + segment6
  let segment7 := 0.75 * sum_to_6
  let one_way := segment1 + segment2 + segment3 + segment4 + segment5 + segment6 + segment7
  2 * one_way

theorem richs_walk_total_distance :
  richs_walk 20 200 300 = 22680 := by
  sorry

end NUMINAMATH_CALUDE_richs_walk_total_distance_l1422_142294


namespace NUMINAMATH_CALUDE_document_delivery_equation_l1422_142230

theorem document_delivery_equation (x : ℝ) (h : x > 3) : 
  let distance : ℝ := 900
  let slow_horse_time : ℝ := x + 1
  let fast_horse_time : ℝ := x - 3
  let slow_horse_speed : ℝ := distance / slow_horse_time
  let fast_horse_speed : ℝ := distance / fast_horse_time
  fast_horse_speed = 2 * slow_horse_speed →
  (distance / slow_horse_time) * 2 = distance / fast_horse_time :=
by sorry


end NUMINAMATH_CALUDE_document_delivery_equation_l1422_142230


namespace NUMINAMATH_CALUDE_rectangle_border_problem_l1422_142208

theorem rectangle_border_problem : 
  (∃ (S : Finset (ℕ × ℕ)), 
    S.card = 4 ∧ 
    (∀ (a b : ℕ), (a, b) ∈ S ↔ 
      (b > a ∧ 
       a > 0 ∧ 
       b > 0 ∧ 
       (a - 4) * (b - 4) = 2 * (a * b) / 3))) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_border_problem_l1422_142208


namespace NUMINAMATH_CALUDE_percentage_less_problem_l1422_142299

theorem percentage_less_problem (C A B : ℝ) : 
  B = 0.58 * C →
  B = 0.8923076923076923 * A →
  ∃ (ε : ℝ), abs (A - 0.65 * C) < ε ∧ ε > 0 := by
sorry

end NUMINAMATH_CALUDE_percentage_less_problem_l1422_142299


namespace NUMINAMATH_CALUDE_f_derivative_at_neg_one_l1422_142246

noncomputable def f (f'_neg_one : ℝ) (x : ℝ) : ℝ := (1/2) * f'_neg_one * x^2 - 2*x + 3

theorem f_derivative_at_neg_one :
  ∃ (f'_neg_one : ℝ), (∀ x, f f'_neg_one x = (1/2) * f'_neg_one * x^2 - 2*x + 3) → f'_neg_one = -1 :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_at_neg_one_l1422_142246


namespace NUMINAMATH_CALUDE_sand_bag_fraction_l1422_142227

/-- Given a bag of sand weighing 50 kg, prove that after using 30 kg,
    the remaining sand accounts for 2/5 of the total bag. -/
theorem sand_bag_fraction (total_weight : ℝ) (used_weight : ℝ) 
  (h1 : total_weight = 50)
  (h2 : used_weight = 30) :
  (total_weight - used_weight) / total_weight = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sand_bag_fraction_l1422_142227


namespace NUMINAMATH_CALUDE_distinct_painting_methods_is_catalan_l1422_142266

/-- Represents a ball with a number and color -/
structure Ball where
  number : Nat
  color : Nat

/-- Represents a painting method for n balls -/
def PaintingMethod (n : Nat) := Fin n → Ball

/-- Checks if two painting methods are distinct -/
def is_distinct (n : Nat) (m1 m2 : PaintingMethod n) : Prop :=
  ∃ i : Fin n, (m1 i).color ≠ (m2 i).color

/-- The number of distinct painting methods for n balls -/
def distinct_painting_methods (n : Nat) : Nat :=
  (Nat.choose (2 * n - 2) (n - 1)) / n

/-- Theorem: The number of distinct painting methods is the (n-1)th Catalan number -/
theorem distinct_painting_methods_is_catalan (n : Nat) :
  distinct_painting_methods n = (Nat.choose (2 * n - 2) (n - 1)) / n :=
by sorry

end NUMINAMATH_CALUDE_distinct_painting_methods_is_catalan_l1422_142266


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1422_142239

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  ∃ q : ℝ, y = x * q ∧ z = y * q

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1 + 2) (a 5 + 5) (a 9 + 8) →
  ∃ q : ℝ, geometric_sequence (a 1 + 2) (a 5 + 5) (a 9 + 8) ∧ q = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1422_142239


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l1422_142268

/-- Proves that given a total lunch cost of $17 and one person spending $3 more than the other,
    the person who spent more paid $10. -/
theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 17 → difference = 3 → friend_cost = total / 2 + difference / 2 → friend_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l1422_142268


namespace NUMINAMATH_CALUDE_playground_area_not_covered_l1422_142293

theorem playground_area_not_covered (playground_side : ℝ) (building_length building_width : ℝ) : 
  playground_side = 12 →
  building_length = 8 →
  building_width = 5 →
  playground_side * playground_side - building_length * building_width = 104 := by
sorry

end NUMINAMATH_CALUDE_playground_area_not_covered_l1422_142293


namespace NUMINAMATH_CALUDE_four_students_same_group_probability_l1422_142253

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- The probability of four specific students being assigned to the same group -/
def prob_four_students_same_group : ℚ := prob_assigned_to_group ^ 3

theorem four_students_same_group_probability :
  prob_four_students_same_group = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_four_students_same_group_probability_l1422_142253


namespace NUMINAMATH_CALUDE_function_properties_l1422_142292

def f (x : ℝ) : ℝ := |2*x - 1| + |x - 2|

theorem function_properties :
  (∀ k : ℝ, (∀ x₀ : ℝ, f x₀ ≥ |k + 3| - |k - 2|) ↔ k ≤ 1/4) ∧
  (∀ m n : ℝ, (∀ x : ℝ, f x ≥ 1/m + 1/n) → m + n ≥ 8/3) ∧
  (∃ m n : ℝ, (∀ x : ℝ, f x ≥ 1/m + 1/n) ∧ m + n = 8/3) := by sorry

end NUMINAMATH_CALUDE_function_properties_l1422_142292


namespace NUMINAMATH_CALUDE_ribbon_leftover_l1422_142222

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) 
  (h1 : total_ribbon = 18)
  (h2 : num_gifts = 6)
  (h3 : ribbon_per_gift = 2) : 
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_leftover_l1422_142222


namespace NUMINAMATH_CALUDE_spinster_count_l1422_142257

theorem spinster_count : 
  ∀ (S C : ℕ),                      -- S: number of spinsters, C: number of cats
  (S : ℚ) / (C : ℚ) = 2 / 9 →       -- Ratio of spinsters to cats is 2:9
  C = S + 42 →                      -- There are 42 more cats than spinsters
  S = 12 :=                         -- Prove that the number of spinsters is 12
by
  sorry

end NUMINAMATH_CALUDE_spinster_count_l1422_142257


namespace NUMINAMATH_CALUDE_sum_of_smallest_solutions_l1422_142236

noncomputable def smallest_solutions (x : ℝ) : Prop :=
  x > 2017 ∧ 
  (Real.cos (9*x))^5 + (Real.cos x)^5 = 
    32 * (Real.cos (5*x))^5 * (Real.cos (4*x))^5 + 
    5 * (Real.cos (9*x))^2 * (Real.cos x)^2 * (Real.cos (9*x) + Real.cos x)

theorem sum_of_smallest_solutions :
  ∃ (x₁ x₂ : ℝ), 
    smallest_solutions x₁ ∧ 
    smallest_solutions x₂ ∧ 
    x₁ < x₂ ∧
    (∀ (y : ℝ), smallest_solutions y → y ≥ x₂ ∨ y = x₁) ∧
    x₁ + x₂ = 4064 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smallest_solutions_l1422_142236


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1422_142261

theorem square_area_from_diagonal : 
  ∀ (d s A : ℝ), 
  d = 8 * Real.sqrt 2 →  -- diagonal length
  d = s * Real.sqrt 2 →  -- relationship between diagonal and side
  A = s^2 →             -- area formula
  A = 64 := by           
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1422_142261


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l1422_142280

def A (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def B : Set ℝ := {1, 2}

theorem union_equality_implies_a_values (a : ℝ) : 
  A a ∪ B = B → a = 0 ∨ a = 1/2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l1422_142280


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l1422_142213

/-- The length of a rectangular metallic sheet that forms an open box with given dimensions and volume -/
theorem metallic_sheet_length : ∃ (L : ℝ),
  (L > 0) ∧ 
  (L - 2 * 8) * (36 - 2 * 8) * 8 = 5120 ∧ 
  L = 48 := by
  sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l1422_142213


namespace NUMINAMATH_CALUDE_dannys_bottle_caps_l1422_142278

theorem dannys_bottle_caps (park_caps : ℕ) (park_wrappers : ℕ) (collection_wrappers : ℕ) :
  park_caps = 58 →
  park_wrappers = 25 →
  collection_wrappers = 11 →
  ∃ (collection_caps : ℕ), collection_caps = collection_wrappers + 1 ∧ collection_caps = 12 :=
by sorry

end NUMINAMATH_CALUDE_dannys_bottle_caps_l1422_142278


namespace NUMINAMATH_CALUDE_triangle_properties_l1422_142262

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- The given equation
  (Real.sqrt 3 * Real.sin A * Real.cos A - Real.sin A ^ 2 = 0) ∧
  -- Collinearity condition
  (∃ (k : Real), k ≠ 0 ∧ k * 1 = 2 ∧ k * Real.sin C = Real.sin B) ∧
  -- Given side length
  (a = 3) →
  -- Prove angle A and perimeter
  (A = π / 3) ∧
  (a + b + c = 3 * (1 + Real.sqrt 3)) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1422_142262


namespace NUMINAMATH_CALUDE_triangle_side_values_l1422_142223

/-- Given a triangle ABC with area S, sides b and c, prove that the third side a
    has one of two specific values. -/
theorem triangle_side_values (S b c : ℝ) (h1 : S = 12 * Real.sqrt 3)
    (h2 : b * c = 48) (h3 : b - c = 2) :
    ∃ (a : ℝ), (a = 2 * Real.sqrt 13 ∨ a = 2 * Real.sqrt 37) ∧
               (S = 1/2 * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_values_l1422_142223


namespace NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l1422_142235

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ q > 0 ∧ ∀ n : ℕ, b (n + 1) = b n * q ∧ b n > 0

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq2 : a 2 = b 2)
  (h_eq10 : a 10 = b 10) :
  a 6 > b 6 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_comparison_l1422_142235


namespace NUMINAMATH_CALUDE_bridge_length_l1422_142281

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 160 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ bridge_length : ℝ,
    bridge_length = 215 ∧
    bridge_length = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l1422_142281


namespace NUMINAMATH_CALUDE_book_reading_rate_l1422_142202

/-- Calculates the number of pages read per day given the total number of pages and days spent reading. -/
def pages_per_day (total_pages : ℕ) (total_days : ℕ) : ℕ :=
  total_pages / total_days

/-- Theorem stating that reading 12518 pages over 569 days results in 22 pages per day. -/
theorem book_reading_rate :
  pages_per_day 12518 569 = 22 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_rate_l1422_142202


namespace NUMINAMATH_CALUDE_consecutive_integers_puzzle_l1422_142234

theorem consecutive_integers_puzzle (a b c d e : ℕ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  b = a + 1 → c = b + 1 → d = c + 1 → e = d + 1 →
  a < b → b < c → c < d → d < e →
  a + b = e - 1 →
  a * b = d + 1 →
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_puzzle_l1422_142234


namespace NUMINAMATH_CALUDE_july_green_tea_price_l1422_142264

/-- Represents the price of tea and coffee in June and July -/
structure PriceData where
  june_price : ℝ
  july_coffee_price : ℝ
  july_tea_price : ℝ

/-- Represents the mixture of tea and coffee -/
structure Mixture where
  tea_quantity : ℝ
  coffee_quantity : ℝ
  total_weight : ℝ
  total_cost : ℝ

/-- Theorem stating the price of green tea in July -/
theorem july_green_tea_price (p : PriceData) (m : Mixture) : 
  p.june_price > 0 ∧ 
  p.july_coffee_price = 2 * p.june_price ∧ 
  p.july_tea_price = 0.1 * p.june_price ∧
  m.tea_quantity = m.coffee_quantity ∧
  m.total_weight = 3 ∧
  m.total_cost = 3.15 ∧
  m.total_cost = m.tea_quantity * p.july_tea_price + m.coffee_quantity * p.july_coffee_price →
  p.july_tea_price = 0.1 := by
sorry


end NUMINAMATH_CALUDE_july_green_tea_price_l1422_142264


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_999_l1422_142219

/-- Sum of digits function for a single number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of all digits in numbers from 0 to n -/
def sumOfAllDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in decimal representations of integers from 0 to 999 is 13500 -/
theorem sum_of_digits_up_to_999 : sumOfAllDigits 999 = 13500 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_999_l1422_142219


namespace NUMINAMATH_CALUDE_product_difference_equality_l1422_142201

theorem product_difference_equality : 2012.25 * 2013.75 - 2010.25 * 2015.75 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equality_l1422_142201


namespace NUMINAMATH_CALUDE_racket_sales_revenue_l1422_142255

theorem racket_sales_revenue 
  (average_price : ℝ) 
  (pairs_sold : ℕ) 
  (h1 : average_price = 9.8) 
  (h2 : pairs_sold = 75) :
  average_price * (pairs_sold : ℝ) = 735 := by
  sorry

end NUMINAMATH_CALUDE_racket_sales_revenue_l1422_142255


namespace NUMINAMATH_CALUDE_midpoint_polygon_perimeter_bound_l1422_142248

/-- A convex polygon with n sides -/
structure ConvexPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry

/-- The perimeter of a polygon -/
def perimeter (P : ConvexPolygon) : ℝ :=
  sorry

/-- The polygon formed by connecting the midpoints of sides of another polygon -/
def midpoint_polygon (P : ConvexPolygon) : ConvexPolygon :=
  sorry

/-- Theorem: The perimeter of the midpoint polygon is at least half the perimeter of the original polygon -/
theorem midpoint_polygon_perimeter_bound (P : ConvexPolygon) :
  perimeter (midpoint_polygon P) ≥ (perimeter P) / 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_polygon_perimeter_bound_l1422_142248


namespace NUMINAMATH_CALUDE_mary_fruit_cost_l1422_142240

/-- Calculates the total cost of fruits with a discount applied -/
def fruitCost (applePrice orangePrice bananaPrice : ℚ) 
              (appleCount orangeCount bananaCount : ℕ) 
              (fruitPerDiscount : ℕ) (discountAmount : ℚ) : ℚ :=
  let totalFruits := appleCount + orangeCount + bananaCount
  let subtotal := applePrice * appleCount + orangePrice * orangeCount + bananaPrice * bananaCount
  let discountCount := totalFruits / fruitPerDiscount
  subtotal - (discountCount * discountAmount)

/-- Theorem stating that Mary will pay $15 for her fruits -/
theorem mary_fruit_cost : 
  fruitCost 1 2 3 5 3 2 5 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_cost_l1422_142240


namespace NUMINAMATH_CALUDE_rope_division_l1422_142296

/-- Given a rope of 1 meter length divided into two parts, where the second part is twice the length of the first part, prove that the length of the first part is 1/3 meter. -/
theorem rope_division (x : ℝ) (h1 : x > 0) (h2 : x + 2*x = 1) : x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l1422_142296


namespace NUMINAMATH_CALUDE_exactly_two_b_values_l1422_142290

-- Define the quadratic function
def f (b : ℤ) (x : ℤ) : ℤ := x^2 + b*x + 3

-- Define a predicate for when f(b,x) ≤ 0
def satisfies_inequality (b : ℤ) (x : ℤ) : Prop := f b x ≤ 0

-- Define a predicate for when b gives exactly three integer solutions
def has_three_solutions (b : ℤ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    satisfies_inequality b x₁ ∧
    satisfies_inequality b x₂ ∧
    satisfies_inequality b x₃ ∧
    ∀ x : ℤ, satisfies_inequality b x → (x = x₁ ∨ x = x₂ ∨ x = x₃)

-- The main theorem
theorem exactly_two_b_values :
  ∃! s : Finset ℤ, s.card = 2 ∧ ∀ b : ℤ, b ∈ s ↔ has_three_solutions b :=
sorry

end NUMINAMATH_CALUDE_exactly_two_b_values_l1422_142290


namespace NUMINAMATH_CALUDE_rectangle_area_l1422_142267

/-- The area of a rectangle with vertices at (-3, 6), (1, 1), and (1, -6), 
    where (1, -6) is 7 units away from (1, 1), is equal to 7√41. -/
theorem rectangle_area : 
  let v1 : ℝ × ℝ := (-3, 6)
  let v2 : ℝ × ℝ := (1, 1)
  let v3 : ℝ × ℝ := (1, -6)
  let side1 := Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)
  let side2 := |v2.2 - v3.2|
  side2 = 7 →
  side1 * side2 = 7 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1422_142267


namespace NUMINAMATH_CALUDE_appropriate_speech_lengths_l1422_142228

/-- Represents the duration of a speech in minutes -/
def SpeechDuration := { d : ℝ // 20 ≤ d ∧ d ≤ 40 }

/-- The recommended speech rate in words per minute -/
def SpeechRate : ℝ := 120

/-- Calculates the number of words for a given duration -/
def wordCount (d : SpeechDuration) : ℝ := d.val * SpeechRate

/-- Checks if a word count is appropriate for the speech -/
def isAppropriateLength (w : ℝ) : Prop :=
  ∃ (d : SpeechDuration), wordCount d = w

theorem appropriate_speech_lengths :
  isAppropriateLength 2500 ∧ 
  isAppropriateLength 3800 ∧ 
  isAppropriateLength 4600 := by sorry

end NUMINAMATH_CALUDE_appropriate_speech_lengths_l1422_142228


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l1422_142298

theorem multiplicative_inverse_modulo_million : 
  let A : ℕ := 123456
  let B : ℕ := 153846
  let N : ℕ := 500000
  let M : ℕ := 1000000
  (A * B * N) % M = 1 := by
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_million_l1422_142298


namespace NUMINAMATH_CALUDE_parallel_planes_k_value_l1422_142256

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then k = 4 -/
theorem parallel_planes_k_value (k : ℝ) :
  let α_normal : ℝ × ℝ × ℝ := (1, 2, -2)
  let β_normal : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (c : ℝ), c ≠ 0 ∧ α_normal = c • β_normal) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_k_value_l1422_142256


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1422_142200

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 4

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 2

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := red_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- The probability of drawing at least one white ball when randomly selecting 2 balls from a bag containing 4 red balls and 2 white balls -/
theorem probability_of_white_ball : 
  (1 : ℚ) - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1422_142200


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l1422_142243

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base-3 representation of the number -/
def base3Number : List Nat := [2, 0, 1, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 416 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l1422_142243


namespace NUMINAMATH_CALUDE_F_2017_composition_l1422_142238

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

def F_comp (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => x
  | n+1 => F (F_comp n x)

theorem F_2017_composition (x : ℝ) : F_comp 2017 x = (x + 1)^(3^2017) - 1 := by
  sorry

end NUMINAMATH_CALUDE_F_2017_composition_l1422_142238


namespace NUMINAMATH_CALUDE_fred_stickers_l1422_142287

theorem fred_stickers (jerry george fred : ℕ) 
  (h1 : jerry = 3 * george)
  (h2 : george = fred - 6)
  (h3 : jerry = 36) : 
  fred = 18 := by
sorry

end NUMINAMATH_CALUDE_fred_stickers_l1422_142287


namespace NUMINAMATH_CALUDE_toucan_count_l1422_142231

theorem toucan_count (initial : ℕ) (joined : ℕ) (total : ℕ) : 
  joined = 1 → total = 3 → initial = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l1422_142231


namespace NUMINAMATH_CALUDE_min_distance_is_sqrt_2_l1422_142233

/-- Two moving lines that intersect at point A -/
structure IntersectingLines where
  a : ℝ
  b : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y => a * x + a + b * y + 3 * b = 0
  l₂ : ℝ → ℝ → Prop := λ x y => b * x - 3 * b - a * y + a = 0

/-- The intersection point of the two lines -/
def intersectionPoint (lines : IntersectingLines) : ℝ × ℝ := sorry

/-- The origin point -/
def origin : ℝ × ℝ := (0, 0)

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The minimum value of the length of segment OA is √2 -/
theorem min_distance_is_sqrt_2 (lines : IntersectingLines) :
  ∃ (min_dist : ℝ), ∀ (a b : ℝ),
    let lines' := { a := a, b := b : IntersectingLines }
    min_dist = Real.sqrt 2 ∧
    distance origin (intersectionPoint lines') ≥ min_dist :=
  sorry

end NUMINAMATH_CALUDE_min_distance_is_sqrt_2_l1422_142233


namespace NUMINAMATH_CALUDE_land_area_proof_l1422_142284

theorem land_area_proof (original_side : ℝ) (cut_width : ℝ) (remaining_area : ℝ) :
  cut_width = 10 →
  remaining_area = 1575 →
  original_side * (original_side - cut_width) = remaining_area →
  original_side * cut_width = 450 :=
by
  sorry

end NUMINAMATH_CALUDE_land_area_proof_l1422_142284


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1422_142282

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (1 - z) = -1) :
  Complex.im z = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1422_142282


namespace NUMINAMATH_CALUDE_divisors_of_fermat_like_number_l1422_142217

-- Define a function to represent the product of the first n primes in a list
def primeProduct : List Nat → Nat
  | [] => 1
  | p::ps => p * primeProduct ps

-- Define the main theorem
theorem divisors_of_fermat_like_number (n : Nat) (primes : List Nat) 
  (h_distinct : List.Pairwise (·≠·) primes)
  (h_prime : ∀ p ∈ primes, Nat.Prime p)
  (h_greater_than_three : ∀ p ∈ primes, p > 3)
  (h_length : primes.length = n) :
  (Nat.divisors (2^(primeProduct primes) + 1)).card ≥ 4^n := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_fermat_like_number_l1422_142217


namespace NUMINAMATH_CALUDE_horse_fertilizer_production_l1422_142241

-- Define the given constants
def num_horses : ℕ := 80
def total_acres : ℕ := 20
def gallons_per_acre : ℕ := 400
def acres_per_day : ℕ := 4
def total_days : ℕ := 25

-- Define the function to calculate daily fertilizer production per horse
def daily_fertilizer_per_horse : ℚ :=
  (total_acres * gallons_per_acre : ℚ) / (num_horses * total_days)

-- Theorem statement
theorem horse_fertilizer_production :
  daily_fertilizer_per_horse = 20 := by
  sorry

end NUMINAMATH_CALUDE_horse_fertilizer_production_l1422_142241


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l1422_142203

-- Define the function f
def f (x : ℝ) : ℝ := 2*x - 3*x

-- Theorem statement
theorem f_has_two_zeros :
  ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ ∀ x, f x = 0 → x = a ∨ x = b :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l1422_142203


namespace NUMINAMATH_CALUDE_new_rope_length_l1422_142216

/-- Proves that given an initial rope length of 12 m and an additional grazing area of 565.7142857142857 m², 
    the new rope length that allows this additional grazing area is 18 m. -/
theorem new_rope_length 
  (initial_length : ℝ) 
  (additional_area : ℝ) 
  (h1 : initial_length = 12)
  (h2 : additional_area = 565.7142857142857) : 
  ∃ (new_length : ℝ), 
    new_length = 18 ∧ 
    π * new_length ^ 2 = π * initial_length ^ 2 + additional_area :=
by sorry

end NUMINAMATH_CALUDE_new_rope_length_l1422_142216


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l1422_142254

theorem relationship_between_x_and_y (x y m : ℝ) 
  (hx : x = 3 - m) (hy : y = 2*m + 1) : 2*x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l1422_142254


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l1422_142232

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  ∃ (k : ℕ), k = 82 ∧ (N + k) % 7 = 0 ∧ (N + k) % 12 = 0 ∧
  ∀ (m : ℕ), m < k → (N + m) % 7 ≠ 0 ∨ (N + m) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l1422_142232


namespace NUMINAMATH_CALUDE_stations_between_hyderabad_and_bangalore_l1422_142269

theorem stations_between_hyderabad_and_bangalore : 
  ∃ (n : ℕ), n > 2 ∧ (n * (n - 1)) / 2 = 306 ∧ n - 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_stations_between_hyderabad_and_bangalore_l1422_142269


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1422_142273

theorem polynomial_factorization (x : ℝ) :
  x^15 + x^10 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1422_142273


namespace NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l1422_142244

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_100_divisors :
  ∃ m : ℕ,
    m > 0 ∧
    is_multiple m 100 ∧
    num_divisors m = 100 ∧
    (∀ k : ℕ, k > 0 → is_multiple k 100 → num_divisors k = 100 → m ≤ k) ∧
    m / 100 = 324 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_100_divisors_l1422_142244


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l1422_142209

theorem isosceles_triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  (a = 40 ∧ b = c) ∨ (b = 40 ∧ a = c) ∨ (c = 40 ∧ a = b) →  -- One angle is 40° and it's an isosceles triangle
  ((b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40) ∨ (a = 100 ∧ c = 40)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l1422_142209


namespace NUMINAMATH_CALUDE_fifty_eight_prime_sum_l1422_142270

/-- A function that returns the number of ways to write n as the sum of two primes -/
def count_prime_pairs (n : ℕ) : ℕ :=
  (Finset.filter (fun p => Nat.Prime p ∧ Nat.Prime (n - p)) (Finset.range (n + 1))).card

/-- Theorem stating that 58 can be written as the sum of two primes in exactly 3 ways -/
theorem fifty_eight_prime_sum : count_prime_pairs 58 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifty_eight_prime_sum_l1422_142270


namespace NUMINAMATH_CALUDE_clear_denominators_l1422_142205

theorem clear_denominators (x : ℝ) : 
  (2*x + 1) / 3 - (10*x + 1) / 6 = 1 ↔ 4*x + 2 - 10*x - 1 = 6 := by
sorry

end NUMINAMATH_CALUDE_clear_denominators_l1422_142205


namespace NUMINAMATH_CALUDE_circle_radius_property_l1422_142224

theorem circle_radius_property (r : ℝ) : 
  r > 0 → r * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → 
  ∃ (radius : ℝ), radius > 0 ∧ radius * (2 * Real.pi * radius) = 2 * (Real.pi * radius^2) := by
sorry

end NUMINAMATH_CALUDE_circle_radius_property_l1422_142224


namespace NUMINAMATH_CALUDE_system_solution_l1422_142204

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 17 - 2*x)
  (eq2 : x + z = -11 - 2*y)
  (eq3 : x + y = 9 - 2*z) :
  3*x + 3*y + 3*z = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1422_142204


namespace NUMINAMATH_CALUDE_range_of_a_l1422_142211

-- Define the sets A, B, and C
def A : Set ℝ := {x | -5 < x ∧ x < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem range_of_a (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1422_142211


namespace NUMINAMATH_CALUDE_max_children_to_movies_l1422_142220

def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 3
def total_budget : ℕ := 35

theorem max_children_to_movies :
  (total_budget - adult_ticket_cost) / child_ticket_cost = 9 :=
sorry

end NUMINAMATH_CALUDE_max_children_to_movies_l1422_142220


namespace NUMINAMATH_CALUDE_banana_sharing_l1422_142272

theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  total_bananas = num_friends * bananas_per_friend →
  bananas_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l1422_142272


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1422_142259

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) ↔ n = 36 ∨ n = -36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1422_142259


namespace NUMINAMATH_CALUDE_households_with_only_bike_l1422_142226

/-- Given a neighborhood with the following properties:
  * There are 90 total households
  * 11 households have neither a car nor a bike
  * 18 households have both a car and a bike
  * 44 households have a car (including those with both)
  Then the number of households with only a bike is 35. -/
theorem households_with_only_bike
  (total : ℕ)
  (neither : ℕ)
  (both : ℕ)
  (with_car : ℕ)
  (h_total : total = 90)
  (h_neither : neither = 11)
  (h_both : both = 18)
  (h_with_car : with_car = 44) :
  total - neither - (with_car - both) - both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l1422_142226


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1422_142247

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c - v }

/-- The original parabola y = 4x^2 + 1 -/
def original_parabola : Parabola :=
  { a := 4, b := 0, c := 1 }

theorem parabola_shift_theorem :
  let shifted := shift_parabola original_parabola 3 2
  shifted = { a := 4, b := -24, c := 35 } := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1422_142247


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1422_142229

theorem perfect_square_binomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 10*x + k = (x + a)^2) ↔ k = 25 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1422_142229


namespace NUMINAMATH_CALUDE_min_product_of_prime_sum_l1422_142277

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → m ≠ n → n ≠ p → m ≠ p → m + n = p → 
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → m' ≠ n' → n' ≠ p' → m' ≠ p' → m' + n' = p' → m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_prime_sum_l1422_142277


namespace NUMINAMATH_CALUDE_a_annual_income_l1422_142295

/-- Proves that A's annual income is 470400 given the specified conditions. -/
theorem a_annual_income (c_monthly_income : ℕ) 
  (h1 : c_monthly_income = 14000)
  (h2 : ∃ (b_monthly_income : ℕ), b_monthly_income = c_monthly_income + c_monthly_income / 100 * 12)
  (h3 : ∃ (a_monthly_income : ℕ), 5 * b_monthly_income = 2 * a_monthly_income) :
  ∃ (a_annual_income : ℕ), a_annual_income = 470400 :=
by sorry

end NUMINAMATH_CALUDE_a_annual_income_l1422_142295


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1422_142214

/-- Two arithmetic sequences and their sums -/
def arithmetic_sequences (a b : ℕ → ℚ) (S T : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n > 0 → S n / T n = (2 * n + 1 : ℚ) / (n + 2 : ℚ)

theorem arithmetic_sequences_ratio 
  (a b : ℕ → ℚ) (S T : ℕ → ℚ) 
  (h : arithmetic_sequences a b S T) : 
  a 7 / b 7 = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l1422_142214


namespace NUMINAMATH_CALUDE_birth_interval_proof_l1422_142276

/-- Proves that the interval between births is 2 years given the conditions of the problem -/
theorem birth_interval_proof (num_children : ℕ) (youngest_age : ℕ) (total_age : ℕ) :
  num_children = 5 →
  youngest_age = 7 →
  total_age = 55 →
  (∃ interval : ℕ,
    total_age = youngest_age * num_children + interval * (num_children * (num_children - 1)) / 2 ∧
    interval = 2) := by
  sorry

end NUMINAMATH_CALUDE_birth_interval_proof_l1422_142276


namespace NUMINAMATH_CALUDE_soap_weight_calculation_l1422_142286

/-- Calculates the total weight of soap bars given the weights of other items and suitcase weights. -/
theorem soap_weight_calculation (initial_weight final_weight perfume_weight chocolate_weight jam_weight : ℝ) : 
  initial_weight = 5 →
  perfume_weight = 5 * 1.2 / 16 →
  chocolate_weight = 4 →
  jam_weight = 2 * 8 / 16 →
  final_weight = 11 →
  final_weight - initial_weight - (perfume_weight + chocolate_weight + jam_weight) = 0.625 := by
  sorry

#check soap_weight_calculation

end NUMINAMATH_CALUDE_soap_weight_calculation_l1422_142286


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l1422_142275

theorem max_sum_with_constraints (a b : ℝ) 
  (h1 : 4 * a + 3 * b ≤ 10) 
  (h2 : 3 * a + 6 * b ≤ 12) : 
  a + b ≤ 14/5 ∧ ∃ (a' b' : ℝ), 4 * a' + 3 * b' = 10 ∧ 3 * a' + 6 * b' = 12 ∧ a' + b' = 14/5 :=
by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l1422_142275


namespace NUMINAMATH_CALUDE_circle_through_points_on_line_equation_l1422_142249

/-- A circle passing through two points with its center on a given line -/
def CircleThroughPointsOnLine (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (a, b) := C
  (x₁ - a)^2 + (y₁ - b)^2 = (x₂ - a)^2 + (y₂ - b)^2 ∧ a + b = 2

/-- The standard equation of a circle -/
def StandardCircleEquation (C : ℝ × ℝ) (r : ℝ) (x y : ℝ) : Prop :=
  let (a, b) := C
  (x - a)^2 + (y - b)^2 = r^2

theorem circle_through_points_on_line_equation :
  ∀ (C : ℝ × ℝ),
  CircleThroughPointsOnLine (1, -1) (-1, 1) C →
  ∃ (x y : ℝ), StandardCircleEquation C 2 x y :=
by sorry

end NUMINAMATH_CALUDE_circle_through_points_on_line_equation_l1422_142249


namespace NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l1422_142218

theorem negation_of_exists_leq (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_leq_negation_of_proposition_l1422_142218


namespace NUMINAMATH_CALUDE_brandon_skittles_l1422_142207

/-- 
Given Brandon's initial number of Skittles and the number of Skittles he loses,
prove that his final number of Skittles is equal to the difference between
the initial number and the number lost.
-/
theorem brandon_skittles (initial : ℕ) (lost : ℕ) :
  initial ≥ lost → initial - lost = initial - lost :=
by sorry

end NUMINAMATH_CALUDE_brandon_skittles_l1422_142207


namespace NUMINAMATH_CALUDE_min_m_value_l1422_142206

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x - a|)

theorem min_m_value (a : ℝ) :
  (∀ x, f a (1 + x) = f a (1 - x)) →
  (∃ m, ∀ x y, m ≤ x → x < y → f a x < f a y) →
  (∀ m', (∀ x y, m' ≤ x → x < y → f a x < f a y) → 1 ≤ m') :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l1422_142206


namespace NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l1422_142271

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

-- Define the property of non-monotonicity in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (x y z : ℝ), a < x ∧ x < y ∧ y < z ∧ z < b ∧
    ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- Theorem statement
theorem non_monotonic_interval_implies_k_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) → 1 ≤ k ∧ k < 3/2 := by sorry

end NUMINAMATH_CALUDE_non_monotonic_interval_implies_k_range_l1422_142271


namespace NUMINAMATH_CALUDE_fraction_problem_l1422_142289

theorem fraction_problem :
  let f₁ : ℚ := 75 / 34
  let f₂ : ℚ := 70 / 51
  (f₁ - f₂ = 5 / 6) ∧
  (Nat.gcd 75 70 = 75 - 70) ∧
  (Nat.lcm 75 70 = 1050) ∧
  (∀ a b c d : ℕ, (a / b : ℚ) = f₁ ∧ (c / d : ℚ) = f₂ → Nat.gcd a c = 1 ∧ Nat.gcd b d = 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1422_142289


namespace NUMINAMATH_CALUDE_car_instantaneous_speed_l1422_142283

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - t^2 + 2

-- State the theorem
theorem car_instantaneous_speed : 
  (deriv s) 1 = 4 := by sorry

end NUMINAMATH_CALUDE_car_instantaneous_speed_l1422_142283


namespace NUMINAMATH_CALUDE_can_determine_coin_type_l1422_142260

/-- Represents the outcome of weighing two groups of coins -/
inductive WeighingResult
  | Even
  | Odd

/-- Represents the type of a coin -/
inductive CoinType
  | Genuine
  | Counterfeit

/-- Function to weigh two groups of coins -/
def weigh (group1 : Finset Nat) (group2 : Finset Nat) : WeighingResult :=
  sorry

theorem can_determine_coin_type 
  (total_coins : Nat)
  (counterfeit_coins : Nat)
  (weight_difference : Nat)
  (h1 : total_coins = 101)
  (h2 : counterfeit_coins = 50)
  (h3 : weight_difference = 1)
  : ∃ (f : Finset Nat → Finset Nat → WeighingResult → CoinType), 
    ∀ (selected_coin : Nat) (group1 group2 : Finset Nat),
    selected_coin ∉ group1 ∧ selected_coin ∉ group2 →
    group1.card = 50 ∧ group2.card = 50 →
    f group1 group2 (weigh group1 group2) = 
      if selected_coin ≤ counterfeit_coins then CoinType.Counterfeit else CoinType.Genuine :=
sorry

end NUMINAMATH_CALUDE_can_determine_coin_type_l1422_142260


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1422_142265

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 + 2*a - 3 = 0) → 
  (b^3 - 2*b^2 + 2*b - 3 = 0) → 
  (c^3 - 2*c^2 + 2*c - 3 = 0) → 
  a^3 + b^3 + c^3 = 5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1422_142265


namespace NUMINAMATH_CALUDE_outfit_choices_l1422_142288

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 8

/-- The number of shirts available -/
def num_shirts : ℕ := num_colors

/-- The number of pants available -/
def num_pants : ℕ := num_colors

/-- The number of hats available -/
def num_hats : ℕ := num_colors

/-- Calculate the total number of outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- Calculate the number of outfits where shirt and pants are the same color -/
def same_color_combinations : ℕ := num_colors * num_hats

/-- Calculate the number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - same_color_combinations

theorem outfit_choices : valid_outfits = 448 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_l1422_142288


namespace NUMINAMATH_CALUDE_tigers_season_games_l1422_142242

def total_games (games_won : ℕ) (games_lost : ℕ) : ℕ :=
  games_won + games_lost

theorem tigers_season_games :
  let games_won : ℕ := 18
  let games_lost : ℕ := games_won + 21
  total_games games_won games_lost = 57 := by
  sorry

end NUMINAMATH_CALUDE_tigers_season_games_l1422_142242


namespace NUMINAMATH_CALUDE_x_equals_four_l1422_142279

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  fun (a, b) (c, d) => (a - c, b + d)

/-- Theorem stating that x = 4 given the conditions -/
theorem x_equals_four :
  ∀ y : ℤ, star (4, 1) (1, -2) = star (x, y) (1, 4) → x = 4 :=
by
  sorry

#check x_equals_four

end NUMINAMATH_CALUDE_x_equals_four_l1422_142279


namespace NUMINAMATH_CALUDE_sum_problem_l1422_142263

/-- The sum of 38 and twice a number is a certain value. The number is 43. -/
theorem sum_problem (x : ℕ) (h : x = 43) : 38 + 2 * x = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_problem_l1422_142263


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l1422_142274

/-- Given two similar right triangles, where one has sides 7, 24, and 25 inches,
    and the other has a hypotenuse of 100 inches, the shortest side of the larger triangle is 28 inches. -/
theorem similar_triangles_shortest_side (a b c d e : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a^2 + b^2 = c^2 →
  a = 7 →
  b = 24 →
  c = 25 →
  e = 100 →
  d / a = e / c →
  d = 28 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l1422_142274
