import Mathlib

namespace NUMINAMATH_CALUDE_linda_purchase_theorem_l1009_100901

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars2 : ℕ
  dollars4 : ℕ

/-- Calculates the total cost in cents given the item counts -/
def totalCost (items : ItemCounts) : ℕ :=
  50 * items.cents50 + 200 * items.dollars2 + 400 * items.dollars4

/-- Theorem stating that given the conditions, Linda bought 40 50-cent items -/
theorem linda_purchase_theorem (items : ItemCounts) : 
  (items.cents50 + items.dollars2 + items.dollars4 = 50) →
  (totalCost items = 5000) →
  (items.cents50 = 40) := by
  sorry

#eval totalCost { cents50 := 40, dollars2 := 4, dollars4 := 6 }

end NUMINAMATH_CALUDE_linda_purchase_theorem_l1009_100901


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1009_100938

/-- 
If two lines given by the equations 4y + 3x + 6 = 0 and 6y + bx + 5 = 0 are perpendicular,
then b = -8.
-/
theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y, 4 * y + 3 * x + 6 = 0 ↔ y = -3/4 * x - 3/2) →
  (∀ x y, 6 * y + b * x + 5 = 0 ↔ y = -b/6 * x - 5/6) →
  ((-3/4) * (-b/6) = -1) →
  b = -8 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1009_100938


namespace NUMINAMATH_CALUDE_budget_food_percentage_l1009_100992

theorem budget_food_percentage (total_budget : ℝ) (accommodation_percent : ℝ) (entertainment_percent : ℝ) (coursework_materials : ℝ) :
  total_budget = 1000 →
  accommodation_percent = 15 →
  entertainment_percent = 25 →
  coursework_materials = 300 →
  (total_budget - (total_budget * accommodation_percent / 100 + total_budget * entertainment_percent / 100 + coursework_materials)) / total_budget * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_budget_food_percentage_l1009_100992


namespace NUMINAMATH_CALUDE_point_on_decreasing_linear_function_l1009_100961

/-- A linear function that decreases as x increases -/
def decreasingLinearFunction (k : ℝ) (x : ℝ) : ℝ :=
  k * (x - 2) + 4

/-- The slope of the linear function is negative -/
def isDecreasing (k : ℝ) : Prop :=
  k < 0

/-- The point (3, -1) lies on the graph of the function -/
def pointOnGraph (k : ℝ) : Prop :=
  decreasingLinearFunction k 3 = -1

/-- Theorem: If the linear function y = k(x-2) + 4 is decreasing,
    then the point (3, -1) lies on its graph -/
theorem point_on_decreasing_linear_function :
  ∀ k : ℝ, isDecreasing k → pointOnGraph k :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_decreasing_linear_function_l1009_100961


namespace NUMINAMATH_CALUDE_lighthouse_distance_l1009_100922

theorem lighthouse_distance (a : ℝ) (h : a > 0) :
  let A : ℝ × ℝ := (a * Real.cos (20 * π / 180), a * Real.sin (20 * π / 180))
  let B : ℝ × ℝ := (a * Real.cos (220 * π / 180), a * Real.sin (220 * π / 180))
  let C : ℝ × ℝ := (0, 0)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_lighthouse_distance_l1009_100922


namespace NUMINAMATH_CALUDE_marks_towers_count_l1009_100995

/-- The number of sandcastles on Mark's beach -/
def marks_castles : ℕ := 20

/-- The number of sandcastles on Jeff's beach -/
def jeffs_castles : ℕ := 3 * marks_castles

/-- The number of towers on each of Jeff's sandcastles -/
def jeffs_towers_per_castle : ℕ := 5

/-- The total number of sandcastles and towers on both beaches -/
def total_count : ℕ := 580

/-- The number of towers on each of Mark's sandcastles -/
def marks_towers_per_castle : ℕ := 10

theorem marks_towers_count : 
  marks_castles + (marks_castles * marks_towers_per_castle) + 
  jeffs_castles + (jeffs_castles * jeffs_towers_per_castle) = total_count := by
  sorry

end NUMINAMATH_CALUDE_marks_towers_count_l1009_100995


namespace NUMINAMATH_CALUDE_jorge_corn_yield_ratio_l1009_100904

/-- Represents the yield ratio problem for Jorge's corn fields --/
theorem jorge_corn_yield_ratio :
  let total_acres : ℚ := 60
  let good_soil_yield : ℚ := 400
  let clay_rich_proportion : ℚ := 1/3
  let total_yield : ℚ := 20000
  let clay_rich_acres : ℚ := total_acres * clay_rich_proportion
  let good_soil_acres : ℚ := total_acres - clay_rich_acres
  let good_soil_total_yield : ℚ := good_soil_acres * good_soil_yield
  let clay_rich_total_yield : ℚ := total_yield - good_soil_total_yield
  let clay_rich_yield : ℚ := clay_rich_total_yield / clay_rich_acres
  clay_rich_yield / good_soil_yield = 1/2 :=
by sorry


end NUMINAMATH_CALUDE_jorge_corn_yield_ratio_l1009_100904


namespace NUMINAMATH_CALUDE_max_triangle_area_l1009_100968

/-- Given a triangle ABC with side lengths a, b, c and internal angles A, B, C,
    this theorem states that the maximum area of the triangle is √2 when
    a = √2, b² - c² = 6, and angle A is at its maximum. -/
theorem max_triangle_area (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 2 →
  b^2 - c^2 = 6 →
  (∀ (a' b' c' : ℝ) (A' B' C' : ℝ),
    a' = Real.sqrt 2 →
    b'^2 - c'^2 = 6 →
    A' ≤ A) →
  (1/2 * b * c * Real.sin A) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1009_100968


namespace NUMINAMATH_CALUDE_smallest_number_l1009_100980

theorem smallest_number (a b c d : ℝ) 
  (ha : a = -Real.sqrt 2) 
  (hb : b = 0) 
  (hc : c = 3.14) 
  (hd : d = 2021) : 
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1009_100980


namespace NUMINAMATH_CALUDE_tangent_line_at_P_l1009_100934

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Define the slope of the tangent line at P
def m : ℝ := 3

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := m * (x - P.1) + P.2

theorem tangent_line_at_P : 
  ∀ x : ℝ, tangent_line x = 3*x - 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_P_l1009_100934


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1009_100949

theorem min_value_trigonometric_expression (A B C : Real) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sum : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2) :
  (1 / (Real.sin A ^ 2 * Real.cos B ^ 4) + 
   1 / (Real.sin B ^ 2 * Real.cos C ^ 4) + 
   1 / (Real.sin C ^ 2 * Real.cos A ^ 4)) ≥ 81/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1009_100949


namespace NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l1009_100998

/-- The maximum area of the projection of a rectangular parallelepiped with edge lengths √70, √99, and √126 onto any plane is 168. -/
theorem max_projection_area_parallelepiped :
  let a := Real.sqrt 70
  let b := Real.sqrt 99
  let c := Real.sqrt 126
  ∃ (proj : ℝ → ℝ → ℝ → ℝ), 
    (∀ x y z, proj x y z ≤ 168) ∧ 
    (∃ x y z, proj x y z = 168) :=
by sorry

end NUMINAMATH_CALUDE_max_projection_area_parallelepiped_l1009_100998


namespace NUMINAMATH_CALUDE_largest_four_digit_binary_is_15_l1009_100932

/-- A binary digit is either 0 or 1 -/
def BinaryDigit : Type := {n : Nat // n = 0 ∨ n = 1}

/-- A four-digit binary number -/
def FourDigitBinary : Type := BinaryDigit × BinaryDigit × BinaryDigit × BinaryDigit

/-- Convert a four-digit binary number to its decimal representation -/
def binaryToDecimal (b : FourDigitBinary) : Nat :=
  b.1.val * 8 + b.2.1.val * 4 + b.2.2.1.val * 2 + b.2.2.2.val

/-- The largest four-digit binary number -/
def largestFourDigitBinary : FourDigitBinary :=
  (⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩)

theorem largest_four_digit_binary_is_15 :
  binaryToDecimal largestFourDigitBinary = 15 := by
  sorry

#eval binaryToDecimal largestFourDigitBinary

end NUMINAMATH_CALUDE_largest_four_digit_binary_is_15_l1009_100932


namespace NUMINAMATH_CALUDE_complement_of_P_l1009_100920

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_of_P : 
  (Set.univ \ P) = {x | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l1009_100920


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l1009_100974

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: When x ∈ (0, +∞), f'(x) > 0
  (∀ x : ℝ, x > 0 → deriv f x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, deriv f (-x) = -(deriv f x)) :=
by
  sorry


end NUMINAMATH_CALUDE_f_satisfies_properties_l1009_100974


namespace NUMINAMATH_CALUDE_correct_phone_call_sequence_l1009_100962

-- Define the steps as an enumeration
inductive PhoneStep
  | Dial
  | WaitForDialTone
  | PickUpHandset
  | StartConversationOrHangUp
  | WaitForSignal
  | EndCall

-- Define the correct sequence of steps
def correctSequence : List PhoneStep :=
  [PhoneStep.PickUpHandset, PhoneStep.WaitForDialTone, PhoneStep.Dial,
   PhoneStep.WaitForSignal, PhoneStep.StartConversationOrHangUp, PhoneStep.EndCall]

-- Theorem statement
theorem correct_phone_call_sequence :
  correctSequence = [PhoneStep.PickUpHandset, PhoneStep.WaitForDialTone, PhoneStep.Dial,
                     PhoneStep.WaitForSignal, PhoneStep.StartConversationOrHangUp, PhoneStep.EndCall] :=
by sorry

end NUMINAMATH_CALUDE_correct_phone_call_sequence_l1009_100962


namespace NUMINAMATH_CALUDE_madhav_rank_from_last_l1009_100909

theorem madhav_rank_from_last (total_students : ℕ) (madhav_rank_start : ℕ) 
  (h1 : total_students = 31) (h2 : madhav_rank_start = 17) : 
  total_students - madhav_rank_start + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_madhav_rank_from_last_l1009_100909


namespace NUMINAMATH_CALUDE_solve_mushroom_problem_l1009_100931

def mushroom_problem (pieces_per_mushroom : ℕ) (total_mushrooms : ℕ) 
  (kenny_pieces : ℕ) (remaining_pieces : ℕ) : Prop :=
  let total_pieces := pieces_per_mushroom * total_mushrooms
  let karla_pieces := total_pieces - (kenny_pieces + remaining_pieces)
  karla_pieces = 42

theorem solve_mushroom_problem :
  mushroom_problem 4 22 38 8 := by sorry

end NUMINAMATH_CALUDE_solve_mushroom_problem_l1009_100931


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_union_B_l1009_100956

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {1, 3, 5, 7}

-- Define set B
def B : Set Nat := {3, 5}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {1, 3, 5, 7} := by sorry

-- Theorem for (∁ₐA) ∪ B
theorem complement_A_union_B : (U \ A) ∪ B = {2, 3, 4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_union_B_l1009_100956


namespace NUMINAMATH_CALUDE_ted_stick_count_l1009_100967

/-- Represents the number of objects thrown by a person -/
structure ThrowCount where
  sticks : ℕ
  rocks : ℕ

/-- The scenario of Bill and Ted throwing objects into the river -/
def river_throw_scenario (ted : ThrowCount) (bill : ThrowCount) : Prop :=
  bill.sticks = ted.sticks + 6 ∧
  ted.rocks = 2 * bill.rocks ∧
  bill.sticks + bill.rocks = 21

theorem ted_stick_count (ted : ThrowCount) (bill : ThrowCount) 
  (h : river_throw_scenario ted bill) : ted.sticks = 15 := by
  sorry

#check ted_stick_count

end NUMINAMATH_CALUDE_ted_stick_count_l1009_100967


namespace NUMINAMATH_CALUDE_perfect_cube_units_digits_l1009_100914

theorem perfect_cube_units_digits : ∀ d : Fin 10, ∃ n : ℤ, (n ^ 3 : ℤ) % 10 = d.val :=
sorry

end NUMINAMATH_CALUDE_perfect_cube_units_digits_l1009_100914


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l1009_100940

theorem wood_length_after_sawing (original_length sawing_length : ℝ) 
  (h1 : original_length = 8.9)
  (h2 : sawing_length = 2.3) : 
  original_length - sawing_length = 6.6 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l1009_100940


namespace NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l1009_100979

/-- The maximum number of non-intersecting diagonals in a convex n-gon --/
def max_non_intersecting_diagonals (n : ℕ) : ℕ := n - 3

/-- Theorem stating that the maximum number of non-intersecting diagonals in a convex n-gon is n-3 --/
theorem max_non_intersecting_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  max_non_intersecting_diagonals n = n - 3 :=
by sorry

end NUMINAMATH_CALUDE_max_non_intersecting_diagonals_correct_l1009_100979


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1009_100903

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The problem statement -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, m^2)
  parallel a b → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1009_100903


namespace NUMINAMATH_CALUDE_equal_numbers_product_l1009_100913

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 24 →
  a = 20 →
  b = 25 →
  c = 33 →
  d = e →
  d * e = 441 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l1009_100913


namespace NUMINAMATH_CALUDE_find_different_coins_possible_l1009_100978

/-- Represents the result of a weighing --/
inductive WeighResult
  | Equal : WeighResult
  | Left : WeighResult
  | Right : WeighResult

/-- Represents a set of coins --/
structure CoinSet where
  total : Nat
  heavy : Nat
  light : Nat
  h_equal_light : heavy = light
  h_total : total = heavy + light

/-- Represents a weighing operation --/
def weigh (left right : CoinSet) : WeighResult :=
  sorry

/-- Represents the process of finding two coins of different weights --/
def findDifferentCoins (coins : CoinSet) (maxWeighings : Nat) : Bool :=
  sorry

/-- The main theorem to be proved --/
theorem find_different_coins_possible :
  ∃ (strategy : CoinSet → Nat → Bool),
    let initialCoins : CoinSet := {
      total := 128,
      heavy := 64,
      light := 64,
      h_equal_light := rfl,
      h_total := rfl
    }
    strategy initialCoins 7 = true :=
  sorry

end NUMINAMATH_CALUDE_find_different_coins_possible_l1009_100978


namespace NUMINAMATH_CALUDE_store_clearance_sale_l1009_100960

/-- Calculates the amount owed to creditors after a store's clearance sale --/
theorem store_clearance_sale 
  (total_items : ℕ) 
  (original_price : ℝ) 
  (discount_percent : ℝ) 
  (sold_percent : ℝ) 
  (remaining_amount : ℝ) 
  (h1 : total_items = 2000)
  (h2 : original_price = 50)
  (h3 : discount_percent = 0.8)
  (h4 : sold_percent = 0.9)
  (h5 : remaining_amount = 3000) : 
  (total_items : ℝ) * sold_percent * (original_price * (1 - discount_percent)) - remaining_amount = 15000 := by
  sorry

end NUMINAMATH_CALUDE_store_clearance_sale_l1009_100960


namespace NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_a_greater_half_l1009_100927

/-- The intersection point of two lines is in the fourth quadrant if and only if
    its x-coordinate is positive and its y-coordinate is negative. -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Given two lines y = -x + 1 and y = x - 2a, their intersection point
    is in the fourth quadrant implies a > 1/2 -/
theorem intersection_in_fourth_quadrant_implies_a_greater_half (a : ℝ) :
  (∃ x y : ℝ, y = -x + 1 ∧ y = x - 2 * a ∧ in_fourth_quadrant x y) →
  a > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_fourth_quadrant_implies_a_greater_half_l1009_100927


namespace NUMINAMATH_CALUDE_min_value_sum_l1009_100906

theorem min_value_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) / (a * b) = 1) :
  a + 2*b ≥ 3 + 2*Real.sqrt 2 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ (a₀ + b₀) / (a₀ * b₀) = 1 ∧ a₀ + 2*b₀ = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1009_100906


namespace NUMINAMATH_CALUDE_frog_population_equality_l1009_100912

theorem frog_population_equality : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (5^(m+1) = 243 * 3^m → m ≥ n)) ∧ 
  5^(n+1) = 243 * 3^n ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_frog_population_equality_l1009_100912


namespace NUMINAMATH_CALUDE_david_orange_juice_purchase_l1009_100970

/-- Calculates the minimum cost to buy a given number of bottles -/
def min_cost (single_price : ℚ) (pack_price : ℚ) (total_bottles : ℕ) : ℚ :=
  let pack_count := total_bottles / 6
  let single_count := total_bottles % 6
  pack_count * pack_price + single_count * single_price

theorem david_orange_juice_purchase :
  min_cost (280/100) (1500/100) 22 = 5620/100 := by
  sorry

end NUMINAMATH_CALUDE_david_orange_juice_purchase_l1009_100970


namespace NUMINAMATH_CALUDE_intersection_empty_iff_union_equals_B_iff_l1009_100943

def A (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x > 1 ∨ x < -6}

theorem intersection_empty_iff (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Icc (-6) (-2) := by sorry

theorem union_equals_B_iff (a : ℝ) :
  A a ∪ B = B ↔ a ∈ Set.Iio (-9) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_union_equals_B_iff_l1009_100943


namespace NUMINAMATH_CALUDE_series_convergence_l1009_100919

theorem series_convergence 
  (u v : ℕ → ℝ) 
  (hu : Summable (fun i => (u i)^2))
  (hv : Summable (fun i => (v i)^2))
  (p : ℕ) 
  (hp : p ≥ 2) : 
  Summable (fun i => (u i - v i)^p) :=
by
  sorry

end NUMINAMATH_CALUDE_series_convergence_l1009_100919


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1009_100954

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b + 2*a*b = 8) :
  ∀ x y, x > 0 → y > 0 → x + 2*y + 2*x*y = 8 → a + 2*b ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1009_100954


namespace NUMINAMATH_CALUDE_complex_simplification_l1009_100985

theorem complex_simplification :
  (7 - 4 * Complex.I) - (2 + 6 * Complex.I) + (3 - 3 * Complex.I) = 8 - 13 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1009_100985


namespace NUMINAMATH_CALUDE_gcd_lcm_120_40_l1009_100971

theorem gcd_lcm_120_40 : 
  (Nat.gcd 120 40 = 40) ∧ (Nat.lcm 120 40 = 120) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_120_40_l1009_100971


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l1009_100948

theorem triangle_area_theorem (x : ℝ) (h1 : x > 0) :
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l1009_100948


namespace NUMINAMATH_CALUDE_floor_length_proof_l1009_100911

/-- Proves that the length of a rectangular floor is 24 meters given specific conditions -/
theorem floor_length_proof (width : ℝ) (square_size : ℝ) (total_cost : ℝ) (square_cost : ℝ) :
  width = 64 →
  square_size = 8 →
  total_cost = 576 →
  square_cost = 24 →
  (total_cost / square_cost) * square_size * square_size / width = 24 := by
sorry

end NUMINAMATH_CALUDE_floor_length_proof_l1009_100911


namespace NUMINAMATH_CALUDE_robies_boxes_given_away_l1009_100936

/-- Given information about Robie's hockey cards and boxes, prove the number of boxes he gave away. -/
theorem robies_boxes_given_away
  (total_cards : ℕ)
  (cards_per_box : ℕ)
  (cards_not_in_box : ℕ)
  (boxes_with_robie : ℕ)
  (h1 : total_cards = 75)
  (h2 : cards_per_box = 10)
  (h3 : cards_not_in_box = 5)
  (h4 : boxes_with_robie = 5) :
  total_cards / cards_per_box - boxes_with_robie = 2 :=
by sorry

end NUMINAMATH_CALUDE_robies_boxes_given_away_l1009_100936


namespace NUMINAMATH_CALUDE_factorization_of_expression_l1009_100916

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

axiom natural_prime_factorization : ∀ n : ℕ, n > 1 → ∃ (primes : List ℕ), (∀ p ∈ primes, is_prime p) ∧ n = primes.prod

theorem factorization_of_expression : 2^4 * 3^2 - 1 = 11 * 13 := by sorry

end NUMINAMATH_CALUDE_factorization_of_expression_l1009_100916


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1009_100944

/-- Calculates the total amount after simple interest --/
def simpleInterestTotal (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that given the initial conditions, the total amount after 7 years is $850 --/
theorem simple_interest_problem (initialSum : ℝ) (totalAfter2Years : ℝ) :
  initialSum = 500 →
  totalAfter2Years = 600 →
  ∃ (rate : ℝ),
    simpleInterestTotal initialSum rate 2 = totalAfter2Years ∧
    simpleInterestTotal initialSum rate 7 = 850 := by
  sorry

/-- The solution to the problem --/
def solution : ℝ := 850

end NUMINAMATH_CALUDE_simple_interest_problem_l1009_100944


namespace NUMINAMATH_CALUDE_percentage_problem_l1009_100935

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1600 - 15 → x = 900 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1009_100935


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_congruences_l1009_100958

theorem smallest_positive_integer_with_congruences : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 → m ≥ n) ∧
  n = 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_congruences_l1009_100958


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l1009_100933

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1

/-- Definition of point M -/
def M : ℝ × ℝ := (0, 2)

/-- Definition of the dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Statement of the theorem -/
theorem ellipse_dot_product_range :
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  ∃ (k : ℝ), P.2 - M.2 = k * (P.1 - M.1) ∧ Q.2 - M.2 = k * (Q.1 - M.1) →
  -20 ≤ dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2) ∧
  dot_product P Q + dot_product (P.1 - M.1, P.2 - M.2) (Q.1 - M.1, Q.2 - M.2) ≤ -52/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l1009_100933


namespace NUMINAMATH_CALUDE_work_completion_l1009_100915

theorem work_completion (x : ℕ) : 
  (x * 40 = (x - 5) * 60) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l1009_100915


namespace NUMINAMATH_CALUDE_prime_power_sum_product_l1009_100982

theorem prime_power_sum_product (p : ℕ) : 
  Prime p → 
  (∃ x y z : ℕ, ∃ q r s : ℕ, 
    Prime q ∧ Prime r ∧ Prime s ∧ 
    q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    x^p + y^p + z^p - x - y - z = q * r * s) ↔ 
  p = 2 ∨ p = 3 ∨ p = 5 := by
sorry

end NUMINAMATH_CALUDE_prime_power_sum_product_l1009_100982


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1009_100928

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1) / 2 = 153) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1009_100928


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1009_100993

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 30 → b = 40 → h^2 = a^2 + b^2 → h = 50 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1009_100993


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1009_100910

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ x : ℝ,
    (x^3 - 2*x^2 + x - 1) / (x^3 + 2*x^2 + x + 1) = 
    P / (x + 1) + (Q*x + R) / (x^2 + 1) ∧
    P = -2 ∧ Q = 0 ∧ R = 1 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1009_100910


namespace NUMINAMATH_CALUDE_polynomial_bound_l1009_100959

open Complex

theorem polynomial_bound (a b c : ℂ) :
  (∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (a * z^2 + b * z + c) ≤ 1) →
  (∀ z : ℂ, Complex.abs z ≤ 1 → 0 ≤ Complex.abs (a * z + b) ∧ Complex.abs (a * z + b) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_bound_l1009_100959


namespace NUMINAMATH_CALUDE_kola_solution_water_added_l1009_100999

/-- Represents the composition of a kola solution -/
structure KolaSolution where
  volume : ℝ
  water_percent : ℝ
  kola_percent : ℝ
  sugar_percent : ℝ

def initial_solution : KolaSolution :=
  { volume := 440
  , water_percent := 88
  , kola_percent := 8
  , sugar_percent := 100 - 88 - 8 }

def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percent : ℝ := 4.521739130434784

/-- The amount of water added to the solution -/
def water_added : ℝ := 10

theorem kola_solution_water_added :
  let initial_sugar := initial_solution.volume * initial_solution.sugar_percent / 100
  let total_sugar := initial_sugar + added_sugar
  let final_volume := total_sugar / (final_sugar_percent / 100)
  water_added = final_volume - initial_solution.volume - added_sugar - added_kola :=
by sorry

end NUMINAMATH_CALUDE_kola_solution_water_added_l1009_100999


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1009_100923

theorem least_number_for_divisibility : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((5918273 + y) % (41 * 71 * 139) = 0)) ∧ 
  ((5918273 + x) % (41 * 71 * 139) = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1009_100923


namespace NUMINAMATH_CALUDE_root_sum_cube_l1009_100941

theorem root_sum_cube (α β : ℝ) : 
  α^2 - 2*α - 4 = 0 →
  β^2 - 2*β - 4 = 0 →
  α^3 + 8*β + 6 = 30 := by
sorry

end NUMINAMATH_CALUDE_root_sum_cube_l1009_100941


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1009_100981

-- Define the inequality
def inequality (x : ℝ) : Prop := (x - 2) / (x - 3) > 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2 ∨ x > 3}

-- Theorem stating that the solution set is correct
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1009_100981


namespace NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l1009_100972

theorem count_integers_satisfying_conditions : 
  ∃! (S : Finset ℤ), 
    (∀ x ∈ S, ⌊Real.sqrt x⌋ = 8 ∧ x % 5 = 3) ∧ 
    (∀ x : ℤ, ⌊Real.sqrt x⌋ = 8 ∧ x % 5 = 3 → x ∈ S) ∧
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l1009_100972


namespace NUMINAMATH_CALUDE_fraction_simplification_l1009_100900

theorem fraction_simplification (x m n : ℝ) (hx : x ≠ 0) (hmn : m + n ≠ 0) :
  x / (x * (m + n)) = 1 / (m + n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1009_100900


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1009_100969

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  let f := fun x : ℝ => (a - 1) * x^2 - a * x + 1
  (a = 2 → {x : ℝ | f x > 0} = {x : ℝ | x ≠ 1}) ∧
  (1 < a ∧ a < 2 → {x : ℝ | f x > 0} = {x : ℝ | x < 1 ∨ x > 1/(a-1)}) ∧
  (a > 2 → {x : ℝ | f x > 0} = {x : ℝ | x < 1/(a-1) ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1009_100969


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l1009_100946

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (m : Line) (α β : Plane) 
  (h1 : l ≠ m) 
  (h2 : α ≠ β) 
  (h3 : subset l α) 
  (h4 : subset m β) :
  perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l1009_100946


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1009_100990

theorem quadratic_roots_sum (a b c : ℝ) (x₁ x₂ : ℂ) : 
  (∃ (s t : ℝ), x₁ = s + t * I ∧ t ≠ 0) →  -- x₁ is a complex number
  (a * x₁^2 + b * x₁ + c = 0) →  -- x₁ is a root of the quadratic equation
  (a * x₂^2 + b * x₂ + c = 0) →  -- x₂ is a root of the quadratic equation
  (∃ (r : ℝ), x₁^2 / x₂ = r) →  -- x₁²/x₂ is real
  let S := 1 + x₁/x₂ + (x₁/x₂)^2 + (x₁/x₂)^4 + (x₁/x₂)^8 + (x₁/x₂)^16 + (x₁/x₂)^32
  S = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1009_100990


namespace NUMINAMATH_CALUDE_david_rosy_age_difference_l1009_100937

theorem david_rosy_age_difference :
  ∀ (david_age rosy_age : ℕ),
    david_age > rosy_age →
    rosy_age = 8 →
    david_age + 4 = 2 * (rosy_age + 4) →
    david_age - rosy_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_david_rosy_age_difference_l1009_100937


namespace NUMINAMATH_CALUDE_fraction_squared_times_32_equals_8_l1009_100905

theorem fraction_squared_times_32_equals_8 : ∃ f : ℚ, f^2 * 32 = 2^3 ∧ f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_squared_times_32_equals_8_l1009_100905


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1009_100965

theorem cubic_equation_solutions : 
  ∃! (s : Finset Int), 
    (∀ x ∈ s, (x^3 - x - 1)^2015 = 1) ∧ 
    (∀ x : Int, (x^3 - x - 1)^2015 = 1 → x ∈ s) ∧ 
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1009_100965


namespace NUMINAMATH_CALUDE_area_of_six_square_figure_l1009_100963

/-- A figure consisting of 6 identical squares with a total perimeter of 84 cm has an area of 216 cm². -/
theorem area_of_six_square_figure (perimeter : ℝ) (num_squares : ℕ) :
  perimeter = 84 →
  num_squares = 6 →
  (perimeter / 14) ^ 2 * num_squares = 216 := by
  sorry

end NUMINAMATH_CALUDE_area_of_six_square_figure_l1009_100963


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1009_100983

theorem sqrt_sum_comparison : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1009_100983


namespace NUMINAMATH_CALUDE_line_through_points_l1009_100917

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) is
    (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

theorem line_through_points :
  ∀ x y : ℝ, line_equation 2 (-2) (-2) 6 x y ↔ 2 * x + y - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l1009_100917


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l1009_100976

theorem sum_and_reciprocal_sum (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_sum : a + b = 6 * x) (h_reciprocal_sum : 1 / a + 1 / b = 6) : x = a * b :=
by sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l1009_100976


namespace NUMINAMATH_CALUDE_buratino_malvina_equation_l1009_100918

theorem buratino_malvina_equation (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_buratino_malvina_equation_l1009_100918


namespace NUMINAMATH_CALUDE_equation_solutions_l1009_100925

theorem equation_solutions :
  (∃ x : ℚ, 5 * x - 9 = 3 * x - 16 ∧ x = -7/2) ∧
  (∃ x : ℚ, (3 * x - 1) / 3 = 1 - (x + 2) / 4 ∧ x = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1009_100925


namespace NUMINAMATH_CALUDE_equation_solution_l1009_100997

theorem equation_solution :
  ∃ y : ℚ, (2 * y + 3 * y = 500 - (4 * y + 5 * y)) ∧ (y = 250 / 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1009_100997


namespace NUMINAMATH_CALUDE_sum_of_monomials_l1009_100987

-- Define the monomials
def monomial1 (x y : ℝ) (m : ℕ) := x^2 * y^m
def monomial2 (x y : ℝ) (n : ℕ) := x^n * y^3

-- Define the condition that the sum is a monomial
def sum_is_monomial (x y : ℝ) (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), ∀ x y, monomial1 x y m + monomial2 x y n = x^a * y^b

-- State the theorem
theorem sum_of_monomials (m n : ℕ) :
  (∀ x y : ℝ, sum_is_monomial x y m n) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_monomials_l1009_100987


namespace NUMINAMATH_CALUDE_fraction_sum_l1009_100950

theorem fraction_sum (x y : ℚ) (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1009_100950


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1009_100988

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 2 → x ≠ 0 → (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1009_100988


namespace NUMINAMATH_CALUDE_alyssa_total_games_l1009_100986

/-- The total number of soccer games Alyssa attends over three years -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Theorem stating that Alyssa will attend 39 games in total -/
theorem alyssa_total_games :
  total_games 11 13 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_total_games_l1009_100986


namespace NUMINAMATH_CALUDE_equality_condition_l1009_100921

theorem equality_condition (a b c k : ℝ) : 
  a + b + c = 1 → (k * (a + b * c) = (a + b) * (a + c) ↔ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l1009_100921


namespace NUMINAMATH_CALUDE_betty_boxes_l1009_100939

def total_oranges : ℕ := 24
def oranges_per_box : ℕ := 8

theorem betty_boxes : 
  total_oranges / oranges_per_box = 3 := by sorry

end NUMINAMATH_CALUDE_betty_boxes_l1009_100939


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_4_to_7_l1009_100955

/-- The sum of the 4th to 7th terms of a geometric sequence with first term 1 and common ratio 3 is 1080 -/
theorem geometric_sequence_sum_4_to_7 :
  let a : ℕ → ℝ := λ n => 1 * (3 : ℝ) ^ (n - 1)
  (a 4) + (a 5) + (a 6) + (a 7) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_4_to_7_l1009_100955


namespace NUMINAMATH_CALUDE_min_value_of_function_l1009_100902

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 - 3*x + 3) + Real.sqrt (y^2 - 3*y + 3) + Real.sqrt (x^2 - Real.sqrt 3 * x * y + y^2) ≥ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l1009_100902


namespace NUMINAMATH_CALUDE_hannah_apple_pie_apples_l1009_100977

/-- Calculates the number of pounds of apples needed for Hannah's apple pie. -/
def apple_pie_apples (
  servings : ℕ)
  (cost_per_serving : ℚ)
  (apple_cost_per_pound : ℚ)
  (pie_crust_cost : ℚ)
  (lemon_cost : ℚ)
  (butter_cost : ℚ) : ℚ :=
  let total_cost := servings * cost_per_serving
  let apple_cost := total_cost - pie_crust_cost - lemon_cost - butter_cost
  apple_cost / apple_cost_per_pound

/-- Theorem stating that Hannah needs 2 pounds of apples for her pie. -/
theorem hannah_apple_pie_apples :
  apple_pie_apples 8 1 2 2 (1/2) (3/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hannah_apple_pie_apples_l1009_100977


namespace NUMINAMATH_CALUDE_square_roots_problem_l1009_100966

theorem square_roots_problem (a : ℝ) (x : ℝ) 
  (h1 : a > 0)
  (h2 : (2*x + 6)^2 = a)
  (h3 : (x - 18)^2 = a) :
  a = 196 := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l1009_100966


namespace NUMINAMATH_CALUDE_final_cat_count_l1009_100973

/-- Represents the number of cats of each breed -/
structure CatInventory where
  siamese : ℕ
  house : ℕ
  persian : ℕ
  sphynx : ℕ

/-- Calculates the total number of cats -/
def totalCats (inventory : CatInventory) : ℕ :=
  inventory.siamese + inventory.house + inventory.persian + inventory.sphynx

/-- Represents a sale event -/
structure SaleEvent where
  siamese : ℕ
  house : ℕ
  persian : ℕ
  sphynx : ℕ

/-- Applies a sale event to the inventory -/
def applySale (inventory : CatInventory) (sale : SaleEvent) : CatInventory where
  siamese := inventory.siamese - sale.siamese
  house := inventory.house - sale.house
  persian := inventory.persian - sale.persian
  sphynx := inventory.sphynx - sale.sphynx

/-- Adds new cats to the inventory -/
def addNewCats (inventory : CatInventory) (newSiamese newPersian : ℕ) : CatInventory where
  siamese := inventory.siamese + newSiamese
  house := inventory.house
  persian := inventory.persian + newPersian
  sphynx := inventory.sphynx

theorem final_cat_count (initialInventory : CatInventory)
    (sale1 sale2 : SaleEvent) (newSiamese newPersian : ℕ)
    (h1 : initialInventory = CatInventory.mk 12 20 8 18)
    (h2 : sale1 = SaleEvent.mk 6 4 5 0)
    (h3 : sale2 = SaleEvent.mk 0 15 0 10)
    (h4 : newSiamese = 5)
    (h5 : newPersian = 3) :
    totalCats (addNewCats (applySale (applySale initialInventory sale1) sale2) newSiamese newPersian) = 26 := by
  sorry


end NUMINAMATH_CALUDE_final_cat_count_l1009_100973


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l1009_100924

theorem probability_of_white_ball (p_red p_black p_white : ℝ) : 
  p_red = 0.3 → p_black = 0.5 → p_red + p_black + p_white = 1 → p_white = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l1009_100924


namespace NUMINAMATH_CALUDE_youngest_child_age_problem_l1009_100964

/-- The age of the youngest child given the conditions of the problem -/
def youngest_child_age (n : ℕ) (interval : ℕ) (sum : ℕ) : ℕ :=
  (sum - (n - 1) * n * interval / 2) / n

/-- Theorem stating the age of the youngest child under the given conditions -/
theorem youngest_child_age_problem :
  youngest_child_age 5 2 55 = 7 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_problem_l1009_100964


namespace NUMINAMATH_CALUDE_dinas_crayons_l1009_100951

theorem dinas_crayons (wanda_crayons : ℕ) (total_crayons : ℕ) (dina_crayons : ℕ) :
  wanda_crayons = 62 →
  total_crayons = 116 →
  total_crayons = wanda_crayons + dina_crayons + (dina_crayons - 2) →
  dina_crayons = 28 := by
  sorry

end NUMINAMATH_CALUDE_dinas_crayons_l1009_100951


namespace NUMINAMATH_CALUDE_oscar_review_questions_l1009_100926

/-- The total number of questions Professor Oscar must review -/
def total_questions (questions_per_exam : ℕ) (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  questions_per_exam * num_classes * students_per_class

/-- Proof that Professor Oscar must review 1750 questions -/
theorem oscar_review_questions :
  total_questions 10 5 35 = 1750 := by
  sorry

end NUMINAMATH_CALUDE_oscar_review_questions_l1009_100926


namespace NUMINAMATH_CALUDE_min_value_sum_product_l1009_100929

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l1009_100929


namespace NUMINAMATH_CALUDE_incorrect_to_correct_ratio_l1009_100945

theorem incorrect_to_correct_ratio (total : ℕ) (correct : ℕ) (incorrect : ℕ) :
  total = 75 →
  incorrect = 2 * correct →
  total = correct + incorrect →
  (incorrect : ℚ) / (correct : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_to_correct_ratio_l1009_100945


namespace NUMINAMATH_CALUDE_smallest_bob_number_l1009_100930

def alice_number : ℕ := 30

theorem smallest_bob_number (bob_number : ℕ) 
  (h1 : ∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ bob_number) 
  (h2 : ∀ n : ℕ, n ≥ bob_number → 
    (∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n)) : 
  bob_number = 30 := by
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l1009_100930


namespace NUMINAMATH_CALUDE_hyperbola_solution_is_three_halves_l1009_100996

/-- The set of all real numbers m that satisfy the conditions of the hyperbola problem -/
def hyperbola_solution : Set ℝ :=
  {m : ℝ | m > 0 ∧ 2 * m^2 + 3 * m = 9}

/-- The theorem stating that the solution set contains only 3/2 -/
theorem hyperbola_solution_is_three_halves : hyperbola_solution = {3/2} := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_solution_is_three_halves_l1009_100996


namespace NUMINAMATH_CALUDE_sqrt_equation_condition_l1009_100953

theorem sqrt_equation_condition (x y : ℝ) : 
  Real.sqrt (3 * x^2 + y^2) = 2 * x + y ↔ x * (x + 4 * y) = 0 ∧ 2 * x + y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_condition_l1009_100953


namespace NUMINAMATH_CALUDE_expression_simplification_l1009_100952

theorem expression_simplification (x : ℝ) (h : x = 4) :
  (x^2 - 4*x + 4) / (x^2 - 1) / (1 - 3 / (x + 1)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1009_100952


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1009_100957

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    where a > 0, b > 0, and one asymptote forms a 60° angle with the y-axis. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
    (h_asymptote : b / a = Real.sqrt 3 / 3) : 
    Real.sqrt (1 + (b / a)^2) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1009_100957


namespace NUMINAMATH_CALUDE_root_sum_quotient_l1009_100994

theorem root_sum_quotient (p p₁ p₂ a b : ℝ) : 
  (a^2 - a) * p + 2 * a + 7 = 0 →
  (b^2 - b) * p + 2 * b + 7 = 0 →
  a / b + b / a = 7 / 10 →
  (p₁^2 - p₁) * a + 2 * a + 7 = 0 →
  (p₁^2 - p₁) * b + 2 * b + 7 = 0 →
  (p₂^2 - p₂) * a + 2 * a + 7 = 0 →
  (p₂^2 - p₂) * b + 2 * b + 7 = 0 →
  p₁ / p₂ + p₂ / p₁ = 9.2225 := by
sorry

end NUMINAMATH_CALUDE_root_sum_quotient_l1009_100994


namespace NUMINAMATH_CALUDE_parking_lot_increase_l1009_100975

def initial_cars : ℕ := 24
def final_cars : ℕ := 48

def percentage_increase (initial final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

theorem parking_lot_increase :
  percentage_increase initial_cars final_cars = 100 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_increase_l1009_100975


namespace NUMINAMATH_CALUDE_original_fraction_l1009_100991

theorem original_fraction (n : ℚ) : 
  (n + 1) / (n + 6) = 7 / 12 → n / (n + 5) = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l1009_100991


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l1009_100908

-- Define the functions
def f (x p q : ℝ) : ℝ := -|x - p| + q
def g (x r s : ℝ) : ℝ := |x - r| + s

-- State the theorem
theorem intersection_implies_sum (p q r s : ℝ) :
  (f 3 p q = g 3 r s) ∧ 
  (f 5 p q = g 5 r s) ∧ 
  (f 3 p q = 6) ∧ 
  (f 5 p q = 2) ∧ 
  (g 3 r s = 6) ∧ 
  (g 5 r s = 2) →
  p + r = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l1009_100908


namespace NUMINAMATH_CALUDE_square_inscribed_problem_l1009_100989

theorem square_inscribed_problem (inner_perimeter outer_perimeter : ℝ) 
  (h1 : inner_perimeter = 32)
  (h2 : outer_perimeter = 40)
  (h3 : inner_perimeter > 0)
  (h4 : outer_perimeter > 0) :
  let inner_side := inner_perimeter / 4
  let outer_side := outer_perimeter / 4
  let third_side := 2 * inner_side
  (∃ (greatest_distance : ℝ), 
    greatest_distance = Real.sqrt 2 ∧ 
    greatest_distance = (outer_side * Real.sqrt 2 - inner_side * Real.sqrt 2) / 2) ∧
  third_side ^ 2 = 256 := by
sorry

end NUMINAMATH_CALUDE_square_inscribed_problem_l1009_100989


namespace NUMINAMATH_CALUDE_kids_at_home_l1009_100942

/-- The number of kids who went to camp -/
def camp_kids : ℕ := 819058

/-- The difference between kids who went to camp and kids who stayed home -/
def difference : ℕ := 150780

/-- The number of kids who stayed home -/
def home_kids : ℕ := camp_kids - difference

theorem kids_at_home : home_kids = 668278 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_home_l1009_100942


namespace NUMINAMATH_CALUDE_fraction_difference_equals_eight_sqrt_three_l1009_100907

theorem fraction_difference_equals_eight_sqrt_three :
  let a : ℝ := 2 + Real.sqrt 3
  let b : ℝ := 2 - Real.sqrt 3
  (a / b) - (b / a) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_equals_eight_sqrt_three_l1009_100907


namespace NUMINAMATH_CALUDE_square_roots_product_l1009_100947

theorem square_roots_product (a b : ℝ) : 
  (a * a = 9) ∧ (b * b = 9) ∧ (a ≠ b) → a * b = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_product_l1009_100947


namespace NUMINAMATH_CALUDE_locus_is_perpendicular_line_l1009_100984

/-- Two non-intersecting circles in a plane -/
structure TwoCircles where
  O₁ : ℝ × ℝ  -- Center of the first circle
  O₂ : ℝ × ℝ  -- Center of the second circle
  R₁ : ℝ      -- Radius of the first circle
  R₂ : ℝ      -- Radius of the second circle
  h₁ : R₁ > 0
  h₂ : R₂ > 0
  h₃ : ‖O₁ - O₂‖ > R₁ + R₂  -- Circles do not intersect

/-- A point X is on the locus if it's the center of a circle that intersects 
    both given circles at diametrically opposite points -/
def IsOnLocus (tc : TwoCircles) (X : ℝ × ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ 
    r^2 = ‖X - tc.O₁‖^2 + tc.R₁^2 ∧
    r^2 = ‖X - tc.O₂‖^2 + tc.R₂^2

/-- The locus of centers of circles that divide two given non-intersecting circles in half 
    is a straight line perpendicular to the line segment connecting the centers of the given circles -/
theorem locus_is_perpendicular_line (tc : TwoCircles) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ 
    (∀ X : ℝ × ℝ, IsOnLocus tc X ↔ a * X.1 + b * X.2 + c = 0) ∧
    a * (tc.O₂.1 - tc.O₁.1) + b * (tc.O₂.2 - tc.O₁.2) = 0 :=
  sorry

end NUMINAMATH_CALUDE_locus_is_perpendicular_line_l1009_100984
