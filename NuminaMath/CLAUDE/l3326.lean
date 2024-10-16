import Mathlib

namespace NUMINAMATH_CALUDE_probability_tropical_temperate_l3326_332696

/-- The number of tropical fruits -/
def tropical_fruits : ℕ := 3

/-- The number of temperate fruits -/
def temperate_fruits : ℕ := 2

/-- The total number of fruits -/
def total_fruits : ℕ := tropical_fruits + temperate_fruits

/-- The number of fruits to be selected -/
def selection_size : ℕ := 2

/-- The probability of selecting one tropical and one temperate fruit -/
theorem probability_tropical_temperate :
  (Nat.choose tropical_fruits 1 * Nat.choose temperate_fruits 1 : ℚ) / 
  Nat.choose total_fruits selection_size = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_tropical_temperate_l3326_332696


namespace NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l3326_332613

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 32 ways to distribute 4 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_four_balls_four_boxes : distribute_objects 4 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_balls_four_boxes_l3326_332613


namespace NUMINAMATH_CALUDE_largest_common_divisor_of_cube_minus_self_l3326_332611

def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = p^2

theorem largest_common_divisor_of_cube_minus_self (n : ℕ) (h : is_prime_square n) :
  (∀ d : ℕ, d > 30 → ¬(d ∣ (n^3 - n))) ∧
  (30 ∣ (n^3 - n)) :=
sorry

end NUMINAMATH_CALUDE_largest_common_divisor_of_cube_minus_self_l3326_332611


namespace NUMINAMATH_CALUDE_no_natural_solutions_for_cubic_equation_l3326_332618

theorem no_natural_solutions_for_cubic_equation :
  ¬∃ (x y z : ℕ), x^3 + 2*y^3 = 4*z^3 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_for_cubic_equation_l3326_332618


namespace NUMINAMATH_CALUDE_steves_salary_l3326_332616

theorem steves_salary (take_home_pay : ℝ) (tax_rate : ℝ) (healthcare_rate : ℝ) (union_dues : ℝ) 
  (h1 : take_home_pay = 27200)
  (h2 : tax_rate = 0.20)
  (h3 : healthcare_rate = 0.10)
  (h4 : union_dues = 800) :
  ∃ (original_salary : ℝ), 
    original_salary * (1 - tax_rate - healthcare_rate) - union_dues = take_home_pay ∧ 
    original_salary = 40000 := by
sorry

end NUMINAMATH_CALUDE_steves_salary_l3326_332616


namespace NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l3326_332674

theorem complex_subtraction_and_multiplication :
  (5 - 4*I : ℂ) - 2*(3 + 6*I) = -1 - 16*I :=
by sorry

end NUMINAMATH_CALUDE_complex_subtraction_and_multiplication_l3326_332674


namespace NUMINAMATH_CALUDE_system_solution_l3326_332646

theorem system_solution (x y : ℝ) : 
  x^3 + y^3 = 1 ∧ x^4 + y^4 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3326_332646


namespace NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l3326_332637

theorem baby_sea_turtles_on_sand (total : ℕ) (swept_fraction : ℚ) : total = 42 → swept_fraction = 1/3 → total - (swept_fraction * total).floor = 28 := by
  sorry

end NUMINAMATH_CALUDE_baby_sea_turtles_on_sand_l3326_332637


namespace NUMINAMATH_CALUDE_part_one_part_two_l3326_332619

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x ∈ (Set.Icc 1 2), x^2 - a ≥ 0
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem for part (1)
theorem part_one (a : ℝ) : P a → a ≤ 1 := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a > 1 ∨ (-2 < a ∧ a < 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3326_332619


namespace NUMINAMATH_CALUDE_circle_condition_l3326_332640

theorem circle_condition (f : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + f = 0 ↔ (x - 2)^2 + (y + 3)^2 = r^2) ↔
  f < 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3326_332640


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l3326_332651

theorem certain_fraction_proof :
  ∀ (x y : ℚ),
  (3 : ℚ) / 5 / ((6 : ℚ) / 7) = (7 : ℚ) / 15 / (x / y) →
  x / y = (2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l3326_332651


namespace NUMINAMATH_CALUDE_root_equality_implies_b_equals_three_l3326_332602

theorem root_equality_implies_b_equals_three
  (a b c N : ℝ)
  (ha : a > 1)
  (hb : b > 1)
  (hc : c > 1)
  (hN : N > 1)
  (h_int_a : ∃ k : ℤ, a = k)
  (h_int_b : ∃ k : ℤ, b = k)
  (h_int_c : ∃ k : ℤ, c = k)
  (h_eq : (N * (N^(1/b))^(1/c))^(1/a) = N^(25/36)) :
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_root_equality_implies_b_equals_three_l3326_332602


namespace NUMINAMATH_CALUDE_arrangement_pattern_sixtieth_number_is_eighteen_l3326_332675

/-- Represents the value in a specific position of the arrangement -/
def arrangementValue (position : ℕ) : ℕ :=
  let rowNum := (position - 1) / 3 + 1
  3 * rowNum

/-- The arrangement follows the specified pattern -/
theorem arrangement_pattern (n : ℕ) :
  ∀ k, k ≤ 3 * n → arrangementValue (3 * (n - 1) + k) = 3 * n :=
  sorry

/-- The 60th number in the arrangement is 18 -/
theorem sixtieth_number_is_eighteen :
  arrangementValue 60 = 18 :=
  sorry

end NUMINAMATH_CALUDE_arrangement_pattern_sixtieth_number_is_eighteen_l3326_332675


namespace NUMINAMATH_CALUDE_total_triangle_area_is_36_l3326_332693

/-- Represents a square in the grid -/
structure Square where
  x : Nat
  y : Nat
  deriving Repr

/-- Represents a triangle in a square -/
structure Triangle where
  square : Square
  deriving Repr

/-- The size of the grid -/
def gridSize : Nat := 6

/-- Calculate the area of a single triangle -/
def triangleArea : ℝ := 0.5

/-- Calculate the number of triangles in a square -/
def trianglesPerSquare : Nat := 2

/-- Calculate the total number of squares in the grid -/
def totalSquares : Nat := gridSize * gridSize

/-- Calculate the total area of all triangles in the grid -/
def totalTriangleArea : ℝ :=
  (totalSquares : ℝ) * (trianglesPerSquare : ℝ) * triangleArea

/-- Theorem stating that the total area of triangles in the grid is 36 -/
theorem total_triangle_area_is_36 : totalTriangleArea = 36 := by
  sorry

#eval totalTriangleArea

end NUMINAMATH_CALUDE_total_triangle_area_is_36_l3326_332693


namespace NUMINAMATH_CALUDE_perimeter_increase_theorem_l3326_332623

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  vertices : List (Real × Real)
  is_convex : Bool

/-- Result of moving sides of a polygon outward -/
structure TransformedPolygon where
  original : ConvexPolygon
  distance : Real

/-- Perimeter of a polygon -/
def perimeter (p : ConvexPolygon) : Real := sorry

/-- Perimeter increase after transformation -/
def perimeter_increase (tp : TransformedPolygon) : Real :=
  perimeter (ConvexPolygon.mk tp.original.vertices true) - perimeter tp.original

/-- Theorem: Perimeter increase is greater than 30 cm when sides are moved by 5 cm -/
theorem perimeter_increase_theorem (p : ConvexPolygon) :
  perimeter_increase (TransformedPolygon.mk p 5) > 30 := by sorry

end NUMINAMATH_CALUDE_perimeter_increase_theorem_l3326_332623


namespace NUMINAMATH_CALUDE_simple_interest_time_calculation_l3326_332661

theorem simple_interest_time_calculation 
  (principal : ℝ) 
  (simple_interest : ℝ) 
  (rate : ℝ) 
  (h1 : principal = 400) 
  (h2 : simple_interest = 160) 
  (h3 : rate = 20) : 
  (simple_interest * 100) / (principal * rate) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_time_calculation_l3326_332661


namespace NUMINAMATH_CALUDE_toms_marble_expense_l3326_332653

/-- Given Tom's expenses, prove the amount spent on marbles --/
theorem toms_marble_expense (skateboard_cost shorts_cost total_toys_cost : ℚ)
  (h1 : skateboard_cost = 9.46)
  (h2 : shorts_cost = 14.50)
  (h3 : total_toys_cost = 19.02) :
  total_toys_cost - skateboard_cost = 9.56 := by
  sorry

#check toms_marble_expense

end NUMINAMATH_CALUDE_toms_marble_expense_l3326_332653


namespace NUMINAMATH_CALUDE_final_x_value_l3326_332638

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ  -- Current value of X
  s : ℕ  -- Current sum S
  n : ℕ  -- Number of iterations

/-- Updates the state for the next iteration -/
def nextState (state : State) : State :=
  { x := state.x + 2,
    s := state.s + state.x + 2,
    n := state.n + 1 }

/-- Computes the final state when S ≥ 10000 -/
def finalState : State :=
  sorry

/-- The main theorem to prove -/
theorem final_x_value :
  finalState.x = 201 ∧ finalState.s ≥ 10000 ∧
  ∀ (prev : State), prev.n < finalState.n → prev.s < 10000 :=
sorry

end NUMINAMATH_CALUDE_final_x_value_l3326_332638


namespace NUMINAMATH_CALUDE_intersection_points_form_diameter_l3326_332666

/-- Two circles in a plane -/
structure TwoCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)

/-- Intersection points of the two circles -/
def intersection_points (tc : TwoCircles) : Set (ℝ × ℝ) :=
  tc.S₁ ∩ tc.S₂

/-- Tangent line to a circle at a point -/
def tangent_line (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Radius of a circle -/
def radius (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Inner arc of a circle -/
def inner_arc (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Line passing through two points -/
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- Diameter of a circle -/
def is_diameter (S : Set (ℝ × ℝ)) (p q : ℝ × ℝ) : Prop := sorry

/-- Main theorem -/
theorem intersection_points_form_diameter
  (tc : TwoCircles)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ intersection_points tc ∧ B ∈ intersection_points tc)
  (h_tangent : tangent_line tc.S₁ A = radius tc.S₂ ∧ tangent_line tc.S₁ B = radius tc.S₂)
  (C : ℝ × ℝ)
  (h_C : C ∈ inner_arc tc.S₁)
  (K L : ℝ × ℝ)
  (h_K : K ∈ line_through A C ∩ tc.S₂)
  (h_L : L ∈ line_through B C ∩ tc.S₂) :
  is_diameter tc.S₂ K L := by sorry

end NUMINAMATH_CALUDE_intersection_points_form_diameter_l3326_332666


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l3326_332652

theorem prime_equation_solutions (p : ℕ) (hp : Prime p) :
  ∀ x y : ℤ, p * (x + y) = x * y ↔
    (x = p * (p + 1) ∧ y = p + 1) ∨
    (x = 2 * p ∧ y = 2 * p) ∨
    (x = 0 ∧ y = 0) ∨
    (x = p * (1 - p) ∧ y = p - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l3326_332652


namespace NUMINAMATH_CALUDE_cultural_festival_talents_l3326_332606

theorem cultural_festival_talents (total_students : ℕ) 
  (cannot_sing cannot_dance cannot_act no_talents : ℕ) : ℕ :=
by
  -- Define the conditions
  have h1 : total_students = 150 := by sorry
  have h2 : cannot_sing = 75 := by sorry
  have h3 : cannot_dance = 95 := by sorry
  have h4 : cannot_act = 40 := by sorry
  have h5 : no_talents = 20 := by sorry
  
  -- Define the number of students with each talent
  let can_sing := total_students - cannot_sing
  let can_dance := total_students - cannot_dance
  let can_act := total_students - cannot_act
  
  -- Define the sum of students with at least one talent
  let with_talents := total_students - no_talents
  
  -- Define the sum of all talents (ignoring overlaps)
  let sum_talents := can_sing + can_dance + can_act
  
  -- Calculate the number of students with exactly two talents
  let two_talents := sum_talents - with_talents
  
  -- Prove that two_talents equals 90
  have h6 : two_talents = 90 := by sorry
  
  -- Return the result
  exact two_talents

-- The theorem states that given the conditions, 
-- the number of students with exactly two talents is 90

end NUMINAMATH_CALUDE_cultural_festival_talents_l3326_332606


namespace NUMINAMATH_CALUDE_x_values_l3326_332603

def S : Set ℤ := {1, -1}

theorem x_values (a b c d e f : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) (he : e ∈ S) (hf : f ∈ S) :
  {x | ∃ (a b c d e f : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ x = a - b + c - d + e - f} = {-6, -4, -2, 0, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_x_values_l3326_332603


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l3326_332609

/-- An arithmetic progression containing the squares of its first three terms consists of integers. -/
theorem arithmetic_progression_with_squares_is_integer (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic progression condition
  (∃ k l m : ℕ, a k = (a 1)^2 ∧ a l = (a 2)^2 ∧ a m = (a 3)^2) →  -- squares condition
  ∀ n, ∃ z : ℤ, a n = z :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l3326_332609


namespace NUMINAMATH_CALUDE_system_solutions_correct_l3326_332601

theorem system_solutions_correct :
  -- System (1)
  (∃ x y : ℚ, x - y = 2 ∧ x + 1 = 2 * (y - 1) ∧ x = 7 ∧ y = 5) ∧
  -- System (2)
  (∃ x y : ℚ, 2 * x + 3 * y = 1 ∧ (y - 1) / 4 = (x - 2) / 3 ∧ x = 1 ∧ y = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l3326_332601


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3326_332617

theorem simplify_sqrt_sum : 
  Real.sqrt 2 + Real.sqrt (2 + 4) + Real.sqrt (2 + 4 + 6) + Real.sqrt (2 + 4 + 6 + 8) = 
  Real.sqrt 2 + Real.sqrt 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3326_332617


namespace NUMINAMATH_CALUDE_log_base_10_of_7_l3326_332697

theorem log_base_10_of_7 (p q : ℝ) (hp : Real.log 3 / Real.log 4 = p) (hq : Real.log 7 / Real.log 3 = q) :
  Real.log 7 / Real.log 10 = 2 * p * q / (2 * p * q + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_base_10_of_7_l3326_332697


namespace NUMINAMATH_CALUDE_circle_properties_l3326_332641

def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*y - 16 = -y^2 + 26*x + 36

theorem circle_properties :
  ∃ (p q s : ℝ),
    (∀ x y, circle_equation x y ↔ (x - p)^2 + (y - q)^2 = s^2) ∧
    p = 13 ∧
    q = 2 ∧
    s = 15 ∧
    p + q + s = 30 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l3326_332641


namespace NUMINAMATH_CALUDE_apartment_complex_flashlights_joas_apartment_complex_flashlights_l3326_332604

/-- Calculates the total number of emergency flashlights in an apartment complex -/
theorem apartment_complex_flashlights (total_buildings : ℕ) 
  (stories_per_building : ℕ) (families_per_floor_type1 : ℕ) 
  (families_per_floor_type2 : ℕ) (flashlights_per_family : ℕ) : ℕ :=
  let half_buildings := total_buildings / 2
  let families_type1 := half_buildings * stories_per_building * families_per_floor_type1
  let families_type2 := half_buildings * stories_per_building * families_per_floor_type2
  let total_families := families_type1 + families_type2
  total_families * flashlights_per_family

/-- The number of emergency flashlights in Joa's apartment complex -/
theorem joas_apartment_complex_flashlights : 
  apartment_complex_flashlights 8 15 4 5 2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_apartment_complex_flashlights_joas_apartment_complex_flashlights_l3326_332604


namespace NUMINAMATH_CALUDE_power_of_1024_l3326_332698

theorem power_of_1024 : 
  (1024 : ℝ) ^ (0.25 : ℝ) * (1024 : ℝ) ^ (0.2 : ℝ) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_1024_l3326_332698


namespace NUMINAMATH_CALUDE_jim_investment_approx_l3326_332649

/-- Represents the investment ratios of John, James, Jim, and Jordan respectively -/
def investment_ratio : Fin 4 → ℕ
  | 0 => 8   -- John
  | 1 => 11  -- James
  | 2 => 15  -- Jim
  | 3 => 19  -- Jordan

/-- The total investment amount in dollars -/
def total_investment : ℚ := 127000

/-- Jim's index in the investment ratio -/
def jim_index : Fin 4 := 2

/-- Calculate Jim's investment amount -/
def jim_investment : ℚ :=
  (total_investment * investment_ratio jim_index) /
  (Finset.sum Finset.univ investment_ratio)

theorem jim_investment_approx :
  ∃ ε > 0, |jim_investment - 35943.40| < ε := by sorry

end NUMINAMATH_CALUDE_jim_investment_approx_l3326_332649


namespace NUMINAMATH_CALUDE_largest_value_l3326_332626

theorem largest_value (a b c d e : ℝ) 
  (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5 ∧ a - 2 = e - 6) :
  e = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l3326_332626


namespace NUMINAMATH_CALUDE_investment_proof_l3326_332699

/-- Represents the total amount invested -/
def total_investment : ℝ := 10000

/-- Represents the amount invested at 6% interest -/
def investment_at_6_percent : ℝ := 7200

/-- Represents the annual interest rate for the first part of the investment -/
def interest_rate_1 : ℝ := 0.06

/-- Represents the annual interest rate for the second part of the investment -/
def interest_rate_2 : ℝ := 0.09

/-- Represents the total interest received after one year -/
def total_interest : ℝ := 684

theorem investment_proof : 
  interest_rate_1 * investment_at_6_percent + 
  interest_rate_2 * (total_investment - investment_at_6_percent) = 
  total_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_proof_l3326_332699


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l3326_332685

theorem quadratic_equation_range :
  {a : ℝ | ∃ x : ℝ, x^2 - 4*x + a = 0} = Set.Iic 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l3326_332685


namespace NUMINAMATH_CALUDE_min_draw_to_ensure_20_l3326_332607

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  white : Nat
  blue : Nat

/-- The minimum number of balls to draw to ensure at least 20 of one color -/
def minDrawToEnsure20 (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw -/
theorem min_draw_to_ensure_20 :
  let counts : BallCounts := { red := 23, green := 24, white := 12, blue := 21 }
  minDrawToEnsure20 counts = 70 := by
  sorry

end NUMINAMATH_CALUDE_min_draw_to_ensure_20_l3326_332607


namespace NUMINAMATH_CALUDE_rice_weight_calculation_l3326_332655

theorem rice_weight_calculation (total : ℚ) : 
  (total * (1 - 3/10) * (1 - 2/5) = 210) → total = 500 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_calculation_l3326_332655


namespace NUMINAMATH_CALUDE_range_of_m_l3326_332614

-- Define the conditions
def condition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m*x + 3/2 > 0

def condition_q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3 -- Simplified condition for foci on x-axis

-- Define the theorem
theorem range_of_m (m : ℝ) :
  condition_p m ∧ condition_q m → 2 < m ∧ m < Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3326_332614


namespace NUMINAMATH_CALUDE_adam_action_figures_l3326_332688

/-- The total number of action figures Adam can fit on his shelves -/
def total_action_figures (initial_shelves : List Nat) (new_shelves : Nat) (new_shelf_capacity : Nat) : Nat :=
  (initial_shelves.sum) + (new_shelves * new_shelf_capacity)

/-- Theorem: Adam can fit 52 action figures on his shelves -/
theorem adam_action_figures :
  total_action_figures [9, 14, 7] 2 11 = 52 := by
  sorry

end NUMINAMATH_CALUDE_adam_action_figures_l3326_332688


namespace NUMINAMATH_CALUDE_spade_calculation_l3326_332671

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : (spade 5 (spade 3 10)) * (spade 2 4) = 4 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l3326_332671


namespace NUMINAMATH_CALUDE_consecutive_even_integers_l3326_332635

theorem consecutive_even_integers (a b c d e : ℤ) : 
  (∃ n : ℤ, a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8) →  -- five consecutive even integers
  (a + e = 204) →  -- sum of first and last is 204
  (a + b + c + d + e = 510) ∧ (a = 98)  -- sum is 510 and smallest is 98
  := by sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_l3326_332635


namespace NUMINAMATH_CALUDE_wine_bottle_prices_l3326_332636

-- Define the prices as real numbers
variable (A B C X Y : ℝ)

-- Define the conditions from the problem
def condition1 : Prop := A + X = 3.50
def condition2 : Prop := B + X = 4.20
def condition3 : Prop := C + Y = 6.10
def condition4 : Prop := A = X + 1.50
def condition5 : Prop := B = X + 2.20
def condition6 : Prop := C = Y + 3.40

-- State the theorem to be proved
theorem wine_bottle_prices 
  (h1 : condition1 A X)
  (h2 : condition2 B X)
  (h3 : condition3 C Y)
  (h4 : condition4 A X)
  (h5 : condition5 B X)
  (h6 : condition6 C Y) :
  A = 2.50 ∧ B = 3.20 ∧ C = 4.75 ∧ X = 1.00 ∧ Y = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_wine_bottle_prices_l3326_332636


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l3326_332610

/-- A line in the xy-plane can be represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Given that line b is parallel to y = 3x - 2 and passes through (3, 4), its y-intercept is -5. -/
theorem parallel_line_y_intercept :
  let reference_line : Line := { slope := 3, point := (0, -2) }
  let b : Line := { slope := reference_line.slope, point := (3, 4) }
  y_intercept b = -5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l3326_332610


namespace NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l3326_332644

theorem tens_digit_of_N_power_20 (N : ℕ) 
  (h1 : Even N) 
  (h2 : ¬ (10 ∣ N)) : 
  (N^20 / 10) % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_tens_digit_of_N_power_20_l3326_332644


namespace NUMINAMATH_CALUDE_pizza_slices_left_over_is_ten_l3326_332630

/-- Calculates the number of pizza slices left over given the conditions of the problem. -/
def pizza_slices_left_over : ℕ :=
  let small_pizza_slices : ℕ := 4
  let large_pizza_slices : ℕ := 8
  let small_pizzas_bought : ℕ := 3
  let large_pizzas_bought : ℕ := 2
  let george_slices : ℕ := 3
  let bob_slices : ℕ := george_slices + 1
  let susie_slices : ℕ := bob_slices / 2
  let bill_fred_mark_slices : ℕ := 3 * 3

  let total_slices : ℕ := small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought
  let total_eaten : ℕ := george_slices + bob_slices + susie_slices + bill_fred_mark_slices

  total_slices - total_eaten

theorem pizza_slices_left_over_is_ten : pizza_slices_left_over = 10 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_left_over_is_ten_l3326_332630


namespace NUMINAMATH_CALUDE_total_amount_in_euros_l3326_332686

/-- Proves that the total amount in euros is 172.55 given the specified conditions --/
theorem total_amount_in_euros : 
  ∀ (x y z w : ℝ),
  y = 0.8 * x →
  z = 0.7 * x →
  w = 0.6 * x →
  y = 42 →
  z = 49 →
  x + w = 120 →
  (x + y + z + w) * 0.85 = 172.55 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_in_euros_l3326_332686


namespace NUMINAMATH_CALUDE_bmw_sales_count_l3326_332687

-- Define the total number of cars sold
def total_cars : ℕ := 300

-- Define the percentages of non-BMW cars sold
def volkswagen_percent : ℚ := 10/100
def toyota_percent : ℚ := 25/100
def acura_percent : ℚ := 20/100

-- Define the theorem
theorem bmw_sales_count :
  let non_bmw_percent : ℚ := volkswagen_percent + toyota_percent + acura_percent
  let bmw_percent : ℚ := 1 - non_bmw_percent
  (bmw_percent * total_cars : ℚ) = 135 := by sorry

end NUMINAMATH_CALUDE_bmw_sales_count_l3326_332687


namespace NUMINAMATH_CALUDE_log_product_identity_l3326_332659

theorem log_product_identity (b c : ℝ) (hb_pos : b > 0) (hc_pos : c > 0) (hb_ne_one : b ≠ 1) (hc_ne_one : c ≠ 1) :
  Real.log b / Real.log (2 * c) * Real.log (2 * c) / Real.log b = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_identity_l3326_332659


namespace NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l3326_332621

/-- Represents the number of students in each grade -/
def students_per_grade : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := 2 * students_per_grade

/-- Represents the number of students to be selected -/
def selected_students : ℕ := 2

/-- Calculates the probability of selecting two students from different grades -/
def prob_different_grades : ℚ :=
  (students_per_grade * students_per_grade) / (total_students.choose selected_students)

theorem prob_different_grades_is_two_thirds :
  prob_different_grades = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_different_grades_is_two_thirds_l3326_332621


namespace NUMINAMATH_CALUDE_mariams_neighborhood_homes_l3326_332678

/-- The number of homes in Mariam's neighborhood -/
def total_homes (homes_one_side : ℕ) (multiplier : ℕ) : ℕ :=
  homes_one_side + multiplier * homes_one_side

/-- Theorem stating the total number of homes in Mariam's neighborhood -/
theorem mariams_neighborhood_homes :
  total_homes 40 3 = 160 := by
  sorry

end NUMINAMATH_CALUDE_mariams_neighborhood_homes_l3326_332678


namespace NUMINAMATH_CALUDE_inequality_proof_l3326_332634

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 2) :
  1/a + 1/b ≥ 2*(a^2 - a + 1)*(b^2 - b + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3326_332634


namespace NUMINAMATH_CALUDE_one_plus_x_geq_two_sqrt_x_l3326_332663

theorem one_plus_x_geq_two_sqrt_x (x : ℝ) (h : x ≥ 0) : 1 + x ≥ 2 * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_one_plus_x_geq_two_sqrt_x_l3326_332663


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3326_332633

theorem polynomial_remainder_theorem (x : ℝ) : 
  (x^51 + 51) % (x + 1) = 50 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l3326_332633


namespace NUMINAMATH_CALUDE_num_distinct_colorings_bound_l3326_332689

/-- Represents a coloring of a 5x5 two-sided paper --/
def Coloring (n : ℕ) := Fin 5 → Fin 5 → Fin n

/-- The group of symmetries for a square --/
inductive SquareSymmetry
| identity
| rotate90
| rotate180
| rotate270
| reflectHorizontal
| reflectVertical
| reflectDiagonal1
| reflectDiagonal2

/-- Applies a symmetry to a coloring --/
def applySymmetry (sym : SquareSymmetry) (c : Coloring n) : Coloring n :=
  sorry

/-- Checks if a coloring is fixed under a symmetry --/
def isFixed (sym : SquareSymmetry) (c : Coloring n) : Prop :=
  c = applySymmetry sym c

/-- The number of colorings fixed by a given symmetry --/
def numFixedColorings (sym : SquareSymmetry) (n : ℕ) : ℕ :=
  sorry

/-- The total number of distinct colorings --/
def numDistinctColorings (n : ℕ) : ℕ :=
  sorry

theorem num_distinct_colorings_bound (n : ℕ) :
  numDistinctColorings n ≤ (n^25 + 4*n^15 + n^13 + 2*n^7) / 8 :=
sorry

end NUMINAMATH_CALUDE_num_distinct_colorings_bound_l3326_332689


namespace NUMINAMATH_CALUDE_combined_distance_is_3480_l3326_332684

-- Define the speeds and times for each train
def speed_A : ℝ := 150
def time_A : ℝ := 8
def speed_B : ℝ := 180
def time_B : ℝ := 6
def speed_C : ℝ := 120
def time_C : ℝ := 10

-- Define the function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem combined_distance_is_3480 :
  distance speed_A time_A + distance speed_B time_B + distance speed_C time_C = 3480 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_3480_l3326_332684


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3326_332624

/-- A polynomial that takes integer values at integer points -/
def IntegerPolynomial := ℤ → ℤ

/-- Proposition: If a polynomial with integer coefficients takes the value 2 
    at three distinct integer points, it cannot take the value 3 at any integer point -/
theorem polynomial_value_theorem (P : IntegerPolynomial) 
  (h1 : ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 2 ∧ P b = 2 ∧ P c = 2) :
  ¬∃ x : ℤ, P x = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3326_332624


namespace NUMINAMATH_CALUDE_max_fraction_sum_l3326_332691

theorem max_fraction_sum (a b c d : ℕ) (h1 : a/b + c/d < 1) (h2 : a + c = 20) :
  ∃ (a₀ b₀ c₀ d₀ : ℕ), 
    a₀/b₀ + c₀/d₀ = 1385/1386 ∧ 
    a₀ + c₀ = 20 ∧
    a₀/b₀ + c₀/d₀ < 1 ∧
    ∀ (x y z w : ℕ), x + z = 20 → x/y + z/w < 1 → x/y + z/w ≤ 1385/1386 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l3326_332691


namespace NUMINAMATH_CALUDE_alison_lollipops_l3326_332657

theorem alison_lollipops :
  ∀ (alison henry diane : ℕ),
  henry = alison + 30 →
  alison = diane / 2 →
  alison + henry + diane = 45 * 6 →
  alison = 60 := by
sorry

end NUMINAMATH_CALUDE_alison_lollipops_l3326_332657


namespace NUMINAMATH_CALUDE_v_equation_l3326_332620

/-- Given that V = kZ - 6 and V = 14 when Z = 5, prove that V = 22 when Z = 7 -/
theorem v_equation (k : ℝ) : 
  (∀ Z, (k * Z - 6 = 14) → (Z = 5)) →
  (k * 7 - 6 = 22) :=
by sorry

end NUMINAMATH_CALUDE_v_equation_l3326_332620


namespace NUMINAMATH_CALUDE_perfect_square_prime_sum_l3326_332694

theorem perfect_square_prime_sum (x y : ℤ) : 
  (∃ k : ℤ, 2 * x * y = k^2) ∧ 
  (∃ p : ℕ, Nat.Prime p ∧ x^2 + y^2 = p) →
  ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_prime_sum_l3326_332694


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3326_332665

/-- Atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- Number of Carbon atoms in the compound -/
def carbon_count : ℕ := 4

/-- Number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 8

/-- Number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Calculation of molecular weight -/
def molecular_weight : ℝ :=
  (carbon_count : ℝ) * carbon_weight +
  (hydrogen_count : ℝ) * hydrogen_weight +
  (oxygen_count : ℝ) * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 88.104 g/mol -/
theorem compound_molecular_weight :
  molecular_weight = 88.104 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3326_332665


namespace NUMINAMATH_CALUDE_infinite_pairs_and_odd_sum_l3326_332628

theorem infinite_pairs_and_odd_sum :
  (∃ (S : Set (ℕ × ℕ)), Set.Infinite S ∧
    ∀ (p : ℕ × ℕ), p ∈ S →
      (⌊(4 + 2 * Real.sqrt 3) * p.1⌋ : ℤ) = ⌊(4 - 2 * Real.sqrt 3) * p.2⌋) ∧
  (∀ (m n : ℕ), 
    (⌊(4 + 2 * Real.sqrt 3) * m⌋ : ℤ) = ⌊(4 - 2 * Real.sqrt 3) * n⌋ →
    Odd (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_pairs_and_odd_sum_l3326_332628


namespace NUMINAMATH_CALUDE_exists_tetrahedron_no_triangle_l3326_332695

/-- A tetrahedron with an inscribed sphere -/
structure TangentialTetrahedron where
  /-- Lengths of tangents from vertices to points of contact with the inscribed sphere -/
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  /-- All lengths are positive -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0

/-- Predicate to check if three lengths can form a triangle -/
def canFormTriangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating that there exists a tangential tetrahedron where no combination
    of tangent lengths can form a triangle -/
theorem exists_tetrahedron_no_triangle :
  ∃ (t : TangentialTetrahedron),
    ¬(canFormTriangle t.a t.b t.c ∨
      canFormTriangle t.a t.b t.d ∨
      canFormTriangle t.a t.c t.d ∨
      canFormTriangle t.b t.c t.d) :=
sorry

end NUMINAMATH_CALUDE_exists_tetrahedron_no_triangle_l3326_332695


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3326_332669

/-- Given an arithmetic sequence {aₙ} with a₁ = 2 and S₃ = 12, prove that a₆ = 12 -/
theorem arithmetic_sequence_a6 (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                             -- a₁ = 2
  S 3 = 12 →                            -- S₃ = 12
  a 6 = 12 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_a6_l3326_332669


namespace NUMINAMATH_CALUDE_mary_ate_seven_slices_l3326_332672

/-- The number of slices in a large pizza -/
def slices_per_pizza : ℕ := 8

/-- The number of pizzas Mary ordered -/
def pizzas_ordered : ℕ := 2

/-- The number of slices Mary has remaining -/
def slices_remaining : ℕ := 9

/-- The number of slices Mary ate -/
def slices_eaten : ℕ := pizzas_ordered * slices_per_pizza - slices_remaining

theorem mary_ate_seven_slices : slices_eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_mary_ate_seven_slices_l3326_332672


namespace NUMINAMATH_CALUDE_coffee_ratio_problem_l3326_332647

/-- Given two types of coffee, p and v, mixed into two blends x and y, 
    prove that the ratio of p to v in y is 1 to 5. -/
theorem coffee_ratio_problem (total_p total_v x_p x_v y_p y_v : ℚ) : 
  total_p = 24 →
  total_v = 25 →
  x_p / x_v = 4 / 1 →
  x_p = 20 →
  total_p = x_p + y_p →
  total_v = x_v + y_v →
  y_p / y_v = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_coffee_ratio_problem_l3326_332647


namespace NUMINAMATH_CALUDE_worker_savings_l3326_332615

theorem worker_savings (P : ℝ) (f : ℝ) (h1 : P > 0) (h2 : 0 ≤ f ∧ f ≤ 1) : 
  12 * f * P = 2 * (1 - f) * P → f = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_worker_savings_l3326_332615


namespace NUMINAMATH_CALUDE_divisibility_by_48_l3326_332600

theorem divisibility_by_48 :
  (∀ (n : ℕ), n > 0 → ¬(48 ∣ (7^n + 1))) ∧
  (∀ (n : ℕ), n > 0 → (48 ∣ (7^n - 1) ↔ Even n)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_48_l3326_332600


namespace NUMINAMATH_CALUDE_exists_x_squared_minus_two_x_plus_one_nonpositive_l3326_332656

theorem exists_x_squared_minus_two_x_plus_one_nonpositive :
  ∃ x : ℝ, x^2 - 2*x + 1 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_squared_minus_two_x_plus_one_nonpositive_l3326_332656


namespace NUMINAMATH_CALUDE_makeup_palette_cost_l3326_332690

/-- The cost of a makeup palette given the following conditions:
  * There are 3 makeup palettes
  * 4 lipsticks cost $2.50 each
  * 3 boxes of hair color cost $4 each
  * The total cost is $67
-/
theorem makeup_palette_cost :
  let num_palettes : ℕ := 3
  let num_lipsticks : ℕ := 4
  let lipstick_cost : ℚ := 5/2
  let num_hair_color : ℕ := 3
  let hair_color_cost : ℚ := 4
  let total_cost : ℚ := 67
  (total_cost - (num_lipsticks * lipstick_cost + num_hair_color * hair_color_cost)) / num_palettes = 15 := by
  sorry

end NUMINAMATH_CALUDE_makeup_palette_cost_l3326_332690


namespace NUMINAMATH_CALUDE_function_characterization_l3326_332643

def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ n p, is_prime p → (f n)^p ≡ n [MOD f p]

def is_identity (f : ℕ → ℕ) : Prop :=
  ∀ n, f n = n

def is_constant_one_on_primes (f : ℕ → ℕ) : Prop :=
  ∀ p, is_prime p → f p = 1

def is_special_function (f : ℕ → ℕ) : Prop :=
  f 2 = 2 ∧
  (∀ p, is_prime p → p > 2 → f p = 1) ∧
  (∀ n, f n ≡ n [MOD 2])

theorem function_characterization (f : ℕ → ℕ) :
  satisfies_condition f →
  (is_identity f ∨ is_constant_one_on_primes f ∨ is_special_function f) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l3326_332643


namespace NUMINAMATH_CALUDE_yard_area_l3326_332680

/-- Given a rectangular yard where one side is 40 feet and the sum of the other three sides is 56 feet,
    the area of the yard is 320 square feet. -/
theorem yard_area (length width : ℝ) : 
  length = 40 →
  2 * width + length = 56 →
  length * width = 320 := by
  sorry

end NUMINAMATH_CALUDE_yard_area_l3326_332680


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l3326_332650

theorem intersection_point_of_lines (x y : ℚ) :
  (5 * x - 3 * y = 20) ∧ (3 * x + 4 * y = 6) ↔ x = 98/29 ∧ y = 87/58 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l3326_332650


namespace NUMINAMATH_CALUDE_unique_triangle_side_length_l3326_332639

open Real

theorem unique_triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  A = π / 4 →
  b = 2 * Real.sqrt 2 →
  0 < B →
  B < 3 * π / 4 →
  0 < C →
  C < 3 * π / 4 →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  (∀ B' C' b' c', 
    0 < B' → B' < 3 * π / 4 → 
    0 < C' → C' < 3 * π / 4 → 
    A + B' + C' = π → 
    2 / sin A = b' / sin B' → 
    2 / sin A = c' / sin C' → 
    (B' = B ∧ C' = C ∧ b' = b ∧ c' = c)) →
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triangle_side_length_l3326_332639


namespace NUMINAMATH_CALUDE_orange_distribution_l3326_332681

theorem orange_distribution (total_oranges : ℕ) (pieces_per_orange : ℕ) (pieces_per_friend : ℕ) :
  total_oranges = 80 →
  pieces_per_orange = 10 →
  pieces_per_friend = 4 →
  (total_oranges * pieces_per_orange) / pieces_per_friend = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l3326_332681


namespace NUMINAMATH_CALUDE_lottery_probability_l3326_332648

theorem lottery_probability (total_tickets : Nat) (winning_tickets : Nat) (people : Nat) :
  total_tickets = 10 →
  winning_tickets = 3 →
  people = 5 →
  (1 : ℚ) - (Nat.choose (total_tickets - winning_tickets) people : ℚ) / (Nat.choose total_tickets people : ℚ) = 11 / 12 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l3326_332648


namespace NUMINAMATH_CALUDE_circle_symmetry_l3326_332658

/-- The original circle -/
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y - 4 = 0

/-- The line of symmetry -/
def symmetry_line (x y : ℝ) : Prop :=
  x - y - 1 = 0

/-- The symmetric circle -/
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 2)^2 = 9

/-- Theorem stating that the symmetric circle is indeed symmetric to the original circle
    with respect to the given line of symmetry -/
theorem circle_symmetry :
  ∀ (x y : ℝ), original_circle x y ↔ 
  ∃ (x' y' : ℝ), symmetric_circle x' y' ∧ 
  ((x + x')/2 = (y + y')/2 + 1) ∧
  (y' - y)/(x' - x) = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3326_332658


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3326_332629

theorem algebraic_expression_value (a b : ℝ) (h : 2 * a - b = 2) :
  4 * a^2 - b^2 - 4 * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3326_332629


namespace NUMINAMATH_CALUDE_reinforcements_calculation_l3326_332622

/-- Calculates the number of reinforcements given initial garrison size, 
    initial provisions duration, time before reinforcement, and 
    remaining provisions duration after reinforcement. -/
def calculate_reinforcements (initial_garrison : ℕ) 
                              (initial_duration : ℕ) 
                              (time_before_reinforcement : ℕ) 
                              (remaining_duration : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that given the specific conditions of the problem,
    the number of reinforcements is 1600. -/
theorem reinforcements_calculation :
  calculate_reinforcements 2000 54 18 20 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_reinforcements_calculation_l3326_332622


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l3326_332662

theorem sin_alpha_for_point (α : Real) :
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = 1 ∧ 
   x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ 
   y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l3326_332662


namespace NUMINAMATH_CALUDE_vector_magnitude_l3326_332632

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (norm a = 1) → 
  (norm (a - 2 • b) = Real.sqrt 21) → 
  (a.1 * b.1 + a.2 * b.2 = - (1/2) * norm a * norm b) →
  norm b = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3326_332632


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l3326_332676

theorem sqrt_expression_equals_two :
  Real.sqrt 12 + Real.sqrt 4 * (Real.sqrt 5 - Real.pi) ^ 0 - |(-2 * Real.sqrt 3)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l3326_332676


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l3326_332692

theorem no_solution_for_equation : 
  ¬ ∃ (x : ℝ), x ≠ 3 ∧ (2 - x) / (x - 3) = 1 / (3 - x) - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l3326_332692


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l3326_332654

def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 9 * d + (n - 1 : ℝ) * d

theorem arithmetic_sequence_geometric_mean (d : ℝ) :
  d ≠ 0 →
  ∃ k : ℕ, k > 0 ∧ 
    (arithmetic_sequence d k) ^ 2 = 
    (arithmetic_sequence d 1) * (arithmetic_sequence d (2 * k)) ∧
    k = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l3326_332654


namespace NUMINAMATH_CALUDE_somu_age_problem_l3326_332683

theorem somu_age_problem (s f : ℕ) : 
  s = f / 4 →
  s - 12 = (f - 12) / 7 →
  s = 24 :=
by sorry

end NUMINAMATH_CALUDE_somu_age_problem_l3326_332683


namespace NUMINAMATH_CALUDE_hike_pace_proof_l3326_332645

/-- Proves that given the conditions of the hike, the pace to the destination is 4 miles per hour -/
theorem hike_pace_proof (distance : ℝ) (return_pace : ℝ) (total_time : ℝ) (pace_to : ℝ) : 
  distance = 12 → 
  return_pace = 6 → 
  total_time = 5 → 
  distance / pace_to + distance / return_pace = total_time → 
  pace_to = 4 := by
sorry

end NUMINAMATH_CALUDE_hike_pace_proof_l3326_332645


namespace NUMINAMATH_CALUDE_max_f_and_min_sum_l3326_332660

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 4|

-- Theorem statement
theorem max_f_and_min_sum :
  (∃ m : ℝ, ∀ x : ℝ, f x ≤ m ∧ ∃ x₀ : ℝ, f x₀ = m) ∧
  (∃ m : ℝ, m = 4 ∧
   ∀ a b : ℝ, a > 0 → b > 0 → a + 2*b = m →
   2/a + 9/b ≥ 8 ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = m ∧ 2/a₀ + 9/b₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_f_and_min_sum_l3326_332660


namespace NUMINAMATH_CALUDE_square_sum_product_l3326_332673

theorem square_sum_product (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) :
  (x^2 + y^2) * (x + y) = 803 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l3326_332673


namespace NUMINAMATH_CALUDE_M_eq_302_l3326_332664

/-- The number of ways to write 3010 as a sum of powers of 10 with restricted coefficients -/
def M : ℕ :=
  (Finset.filter (fun (b₃ : ℕ) =>
    (Finset.filter (fun (b₂ : ℕ) =>
      (Finset.filter (fun (b₁ : ℕ) =>
        (Finset.filter (fun (b₀ : ℕ) =>
          b₃ * 1000 + b₂ * 100 + b₁ * 10 + b₀ = 3010
        ) (Finset.range 100)).card > 0
      ) (Finset.range 100)).card > 0
    ) (Finset.range 100)).card > 0
  ) (Finset.range 100)).card

/-- The theorem stating that M equals 302 -/
theorem M_eq_302 : M = 302 := by
  sorry

end NUMINAMATH_CALUDE_M_eq_302_l3326_332664


namespace NUMINAMATH_CALUDE_expression_evaluation_l3326_332625

theorem expression_evaluation :
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3326_332625


namespace NUMINAMATH_CALUDE_max_value_inequality_l3326_332679

theorem max_value_inequality (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2*y + 1| ≤ 5 ∧ ∀ (z : ℝ), (∀ (a b : ℝ), |a - 1| ≤ 1 → |b - 2| ≤ 1 → |a - 2*b + 1| ≤ z) → 5 ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3326_332679


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3326_332631

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the lines
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the chord length
def chord_length (k m : ℝ) : ℝ := 
  2 * Real.sqrt 2 * (Real.sqrt (1 + k^2) * Real.sqrt (2 * k^2 - m^2 + 1)) / (1 + 2 * k^2)

-- Define the area of the quadrilateral
def quad_area (k m₁ : ℝ) : ℝ := 
  4 * Real.sqrt 2 * Real.sqrt ((2 * k^2 - m₁^2 + 1) * m₁^2) / (1 + 2 * k^2)

-- State the theorem
theorem ellipse_intersection_theorem (k m₁ m₂ : ℝ) 
  (h₁ : m₁ ≠ m₂) 
  (h₂ : chord_length k m₁ = chord_length k m₂) : 
  m₁ + m₂ = 0 ∧ 
  ∀ m, quad_area k m ≤ 2 * Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l3326_332631


namespace NUMINAMATH_CALUDE_max_value_of_function_l3326_332612

theorem max_value_of_function (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  ∃ (max_y : ℝ), max_y = 5/2 ∧ 
  ∀ y : ℝ, y = 2^(2*x - 1) - 3 * 2^x + 5 → y ≤ max_y :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3326_332612


namespace NUMINAMATH_CALUDE_constant_phi_forms_cone_l3326_332677

/-- A point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The set of points satisfying φ = d -/
def ConstantPhiSet (d : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ = d}

/-- Definition of a cone in spherical coordinates -/
def IsCone (s : Set SphericalPoint) : Prop :=
  ∃ d : ℝ, s = ConstantPhiSet d

/-- Theorem: The set of points with constant φ forms a cone -/
theorem constant_phi_forms_cone (d : ℝ) :
  IsCone (ConstantPhiSet d) := by
  sorry

end NUMINAMATH_CALUDE_constant_phi_forms_cone_l3326_332677


namespace NUMINAMATH_CALUDE_propositions_equivalent_l3326_332627

-- Define P as a set
variable (P : Set α)

-- Define the original proposition
def original_prop (a b : α) : Prop :=
  a ∈ P → b ∉ P

-- Define the equivalent proposition (option D)
def equivalent_prop (a b : α) : Prop :=
  b ∈ P → a ∉ P

-- Theorem stating the equivalence of the two propositions
theorem propositions_equivalent (a b : α) :
  original_prop P a b ↔ equivalent_prop P a b :=
sorry

end NUMINAMATH_CALUDE_propositions_equivalent_l3326_332627


namespace NUMINAMATH_CALUDE_cranberries_count_l3326_332642

/-- The number of cranberries picked by Iris's sister -/
def cranberries : ℕ := 20

/-- The number of blueberries picked by Iris -/
def blueberries : ℕ := 30

/-- The number of raspberries picked by Iris's brother -/
def raspberries : ℕ := 10

/-- The total number of berries picked -/
def total_berries : ℕ := blueberries + cranberries + raspberries

/-- The number of fresh berries -/
def fresh_berries : ℕ := (2 * total_berries) / 3

/-- The number of berries that can be sold -/
def sellable_berries : ℕ := fresh_berries / 2

theorem cranberries_count : sellable_berries = 20 := by sorry

end NUMINAMATH_CALUDE_cranberries_count_l3326_332642


namespace NUMINAMATH_CALUDE_fraction_equality_l3326_332682

theorem fraction_equality (x y : ℚ) (h : x / y = 7 / 3) :
  (x + y) / (x - y) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3326_332682


namespace NUMINAMATH_CALUDE_age_difference_and_future_relation_l3326_332667

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := {n : ℕ // n ≤ 9}

/-- Jack's age given two digits -/
def jack_age (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Bill's age given two digits -/
def bill_age (a b : Digit) : ℕ := b.val^2 + a.val

theorem age_difference_and_future_relation :
  ∃ (a b : Digit), 
    (jack_age a b - bill_age a b = 18) ∧ 
    (jack_age a b + 6 = 3 * (bill_age a b + 6)) := by
  sorry

end NUMINAMATH_CALUDE_age_difference_and_future_relation_l3326_332667


namespace NUMINAMATH_CALUDE_apple_ratio_simplified_l3326_332608

/-- Represents the number of apples picked by each person -/
structure ApplePickers where
  sarah : Nat
  brother : Nat
  cousin : Nat

/-- Represents a ratio as three natural numbers -/
structure Ratio where
  a : Nat
  b : Nat
  c : Nat

/-- Function to simplify a ratio by dividing all components by their GCD -/
def simplifyRatio (r : Ratio) : Ratio :=
  sorry

/-- The main theorem to prove -/
theorem apple_ratio_simplified (pickers : ApplePickers) 
  (h : pickers = ⟨45, 9, 27⟩) : 
  simplifyRatio ⟨pickers.sarah, pickers.brother, pickers.cousin⟩ = ⟨5, 1, 3⟩ :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_simplified_l3326_332608


namespace NUMINAMATH_CALUDE_midpoint_locus_is_annulus_l3326_332605

/-- Two non-intersecting circles in a plane --/
structure TwoCircles where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  h1 : radius1 > 0
  h2 : radius2 > 0
  h3 : radius1 > radius2
  h4 : dist center1 center2 > radius1 + radius2

/-- The locus of midpoints of segments with endpoints on two non-intersecting circles --/
def midpointLocus (c : TwoCircles) : Set (ℝ × ℝ) :=
  {p | ∃ (a b : ℝ × ℝ), 
    dist a c.center1 = c.radius1 ∧ 
    dist b c.center2 = c.radius2 ∧ 
    p = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)}

/-- An annulus (ring) in a plane --/
def annulus (center : ℝ × ℝ) (inner_radius outer_radius : ℝ) : Set (ℝ × ℝ) :=
  {p | inner_radius ≤ dist p center ∧ dist p center ≤ outer_radius}

/-- The main theorem: the locus of midpoints is an annulus --/
theorem midpoint_locus_is_annulus (c : TwoCircles) :
  ∃ (center : ℝ × ℝ),
    midpointLocus c = annulus center ((c.radius1 - c.radius2) / 2) ((c.radius1 + c.radius2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_is_annulus_l3326_332605


namespace NUMINAMATH_CALUDE_soda_crate_weight_l3326_332670

/-- Given the following conditions:
  - Bridge weight limit is 20,000 pounds
  - Empty truck weight is 12,000 pounds
  - There are 20 soda crates
  - There are 3 dryers
  - Each dryer weighs 3,000 pounds
  - Weight of produce is twice the weight of soda
  - Fully loaded truck weighs 24,000 pounds

  Prove that each soda crate weighs 50 pounds -/
theorem soda_crate_weight :
  ∀ (bridge_limit : ℕ) 
    (empty_truck_weight : ℕ) 
    (num_soda_crates : ℕ) 
    (num_dryers : ℕ) 
    (dryer_weight : ℕ) 
    (loaded_truck_weight : ℕ),
  bridge_limit = 20000 →
  empty_truck_weight = 12000 →
  num_soda_crates = 20 →
  num_dryers = 3 →
  dryer_weight = 3000 →
  loaded_truck_weight = 24000 →
  ∃ (soda_weight produce_weight : ℕ),
    produce_weight = 2 * soda_weight ∧
    loaded_truck_weight = empty_truck_weight + num_dryers * dryer_weight + soda_weight + produce_weight →
    soda_weight / num_soda_crates = 50 :=
by sorry

end NUMINAMATH_CALUDE_soda_crate_weight_l3326_332670


namespace NUMINAMATH_CALUDE_f_1989_value_l3326_332668

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2) * (1 - f x) = 1 + f x

theorem f_1989_value (f : ℝ → ℝ) 
    (h_eq : SatisfiesEquation f) 
    (h_f1 : f 1 = 2 + Real.sqrt 3) : 
    f 1989 = -2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_f_1989_value_l3326_332668
