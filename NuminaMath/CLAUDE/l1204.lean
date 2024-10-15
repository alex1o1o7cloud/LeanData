import Mathlib

namespace NUMINAMATH_CALUDE_F_is_even_l1204_120412

-- Define the property of being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function F
def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  |f x| + f (|x|)

-- Theorem statement
theorem F_is_even (f : ℝ → ℝ) (h : is_odd_function f) :
  ∀ x, F f (-x) = F f x :=
by sorry

end NUMINAMATH_CALUDE_F_is_even_l1204_120412


namespace NUMINAMATH_CALUDE_range_of_k_l1204_120479

theorem range_of_k (k : ℝ) : 
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*k*a*b) → k ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l1204_120479


namespace NUMINAMATH_CALUDE_max_product_on_line_l1204_120414

/-- Given points A(a,b) and B(4,2) on the line y = kx + 3 where k is a non-zero constant,
    the maximum value of the product ab is 9. -/
theorem max_product_on_line (a b : ℝ) (k : ℝ) : 
  k ≠ 0 → 
  b = k * a + 3 → 
  2 = k * 4 + 3 → 
  ∃ (max : ℝ), max = 9 ∧ ∀ (x y : ℝ), y = k * x + 3 → x * y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_product_on_line_l1204_120414


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1204_120428

theorem right_triangle_sides : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = 2 ∧ c = 2) ∨
   (a = 1 ∧ b = 1 ∧ c = Real.sqrt 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 4 ∧ b = 5 ∧ c = 6)) ∧
  (a^2 + b^2 = c^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1204_120428


namespace NUMINAMATH_CALUDE_fundraising_amount_scientific_notation_l1204_120492

/-- Represents the amount in yuan --/
def amount : ℝ := 2.175e9

/-- Represents the number of significant figures to preserve --/
def significant_figures : ℕ := 3

/-- Converts a number to scientific notation with a specified number of significant figures --/
noncomputable def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem fundraising_amount_scientific_notation :
  to_scientific_notation amount significant_figures = (2.18, 9) := by sorry

end NUMINAMATH_CALUDE_fundraising_amount_scientific_notation_l1204_120492


namespace NUMINAMATH_CALUDE_beta_value_l1204_120462

open Real

theorem beta_value (α β : ℝ) 
  (h1 : sin α = (4/7) * Real.sqrt 3)
  (h2 : cos (α + β) = -11/14)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) : 
  β = π/3 := by
sorry

end NUMINAMATH_CALUDE_beta_value_l1204_120462


namespace NUMINAMATH_CALUDE_four_balls_two_boxes_l1204_120429

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := 
  (k ^ n) / (Nat.factorial k)

/-- Theorem: There are 8 ways to put 4 distinguishable balls into 2 indistinguishable boxes -/
theorem four_balls_two_boxes : ways_to_put_balls_in_boxes 4 2 = 8 := by
  sorry

#eval ways_to_put_balls_in_boxes 4 2

end NUMINAMATH_CALUDE_four_balls_two_boxes_l1204_120429


namespace NUMINAMATH_CALUDE_max_stamps_proof_l1204_120477

/-- The price of a single stamp in cents -/
def stamp_price : ℕ := 50

/-- The discount rate applied when buying more than 100 stamps -/
def discount_rate : ℚ := 1/10

/-- The threshold number of stamps for applying the discount -/
def discount_threshold : ℕ := 100

/-- The total amount available in cents -/
def total_amount : ℕ := 10000

/-- The maximum number of stamps that can be purchased -/
def max_stamps : ℕ := 200

theorem max_stamps_proof :
  (∀ n : ℕ, n ≤ max_stamps → n * stamp_price ≤ total_amount) ∧
  (∀ n : ℕ, n > max_stamps → 
    (if n > discount_threshold 
     then n * stamp_price * (1 - discount_rate)
     else n * stamp_price) > total_amount) :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_proof_l1204_120477


namespace NUMINAMATH_CALUDE_different_graphs_l1204_120408

-- Define the three equations
def equation_I (x y : ℝ) : Prop := y = x + 3
def equation_II (x y : ℝ) : Prop := y = (x^2 - 1) / (x - 1)
def equation_III (x y : ℝ) : Prop := (x - 1) * y = x^2 - 1

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem statement
theorem different_graphs :
  ¬(same_graph equation_I equation_II) ∧
  ¬(same_graph equation_I equation_III) ∧
  ¬(same_graph equation_II equation_III) :=
sorry

end NUMINAMATH_CALUDE_different_graphs_l1204_120408


namespace NUMINAMATH_CALUDE_interest_calculation_l1204_120480

/-- Given a principal amount P, calculate the compound interest for 2 years at 5% per year -/
def compound_interest (P : ℝ) : ℝ :=
  P * (1 + 0.05)^2 - P

/-- Given a principal amount P, calculate the simple interest for 2 years at 5% per year -/
def simple_interest (P : ℝ) : ℝ :=
  P * 0.05 * 2

/-- Theorem stating that if the compound interest is $615, then the simple interest is $600 -/
theorem interest_calculation (P : ℝ) :
  compound_interest P = 615 → simple_interest P = 600 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l1204_120480


namespace NUMINAMATH_CALUDE_triangle_condition_line_through_intersection_l1204_120441

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y - 4 = 0
def l2 (x y : ℝ) : Prop := x - y + 2 = 0
def l3 (a x y : ℝ) : Prop := a * x - y + 1 - 4 * a = 0

-- Define point M
def M : ℝ × ℝ := (-1, 2)

-- Theorem for the range of a
theorem triangle_condition (a : ℝ) :
  (∃ x y z : ℝ, l1 x y ∧ l2 y z ∧ l3 a z x) ↔ 
  (a ≠ -2/3 ∧ a ≠ 1 ∧ a ≠ -1) :=
sorry

-- Theorem for the equation of line l
theorem line_through_intersection (x y : ℝ) :
  (∃ p q : ℝ, l1 p q ∧ l2 p q) ∧ 
  (abs (3*x + 4*y - 15) / Real.sqrt (3^2 + 4^2) = 2) ↔
  3*x + 4*y - 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_triangle_condition_line_through_intersection_l1204_120441


namespace NUMINAMATH_CALUDE_original_decimal_l1204_120423

theorem original_decimal (x : ℝ) : (x - x / 100 = 1.485) → x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_original_decimal_l1204_120423


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_of_100_l1204_120410

def is_multiple_of_100 (n : ℕ) : Prop := ∃ k : ℕ, n = 100 * k

def digit_product (n : ℕ) : ℕ := 
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

theorem least_multiple_with_digit_product_multiple_of_100 : 
  ∀ n : ℕ, is_multiple_of_100 n → n ≥ 100 → 
    (is_multiple_of_100 (digit_product n) → n ≥ 100) ∧
    (is_multiple_of_100 (digit_product 100)) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_of_100_l1204_120410


namespace NUMINAMATH_CALUDE_number_equal_to_its_opposite_l1204_120404

theorem number_equal_to_its_opposite : ∀ x : ℝ, x = -x ↔ x = 0 := by sorry

end NUMINAMATH_CALUDE_number_equal_to_its_opposite_l1204_120404


namespace NUMINAMATH_CALUDE_x_power_twenty_is_negative_one_l1204_120487

theorem x_power_twenty_is_negative_one (x : ℂ) (h : x + 1/x = Real.sqrt 2) : x^20 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_twenty_is_negative_one_l1204_120487


namespace NUMINAMATH_CALUDE_equation_equivalent_to_lines_l1204_120472

/-- The set of points satisfying the original equation -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The set of points on the first line -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The set of points on the second line -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- Theorem stating the equivalence of the sets -/
theorem equation_equivalent_to_lines : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalent_to_lines_l1204_120472


namespace NUMINAMATH_CALUDE_gravel_path_rate_l1204_120445

/-- Given a rectangular plot with an inner gravel path, calculate the rate per square meter for gravelling. -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 100)
  (h2 : width = 70)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 742.5) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_rate_l1204_120445


namespace NUMINAMATH_CALUDE_unique_f_2_l1204_120435

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x + y) = f x + f y - x * y

theorem unique_f_2 (f : ℝ → ℝ) (hf : special_function f) : 
  f 2 = 3 ∧ ∀ y : ℝ, f 2 = y → y = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_f_2_l1204_120435


namespace NUMINAMATH_CALUDE_joel_donation_l1204_120482

/-- The number of toys Joel donated -/
def joels_toys : ℕ := 22

/-- The number of toys Joel's sister donated -/
def sisters_toys : ℕ := 11

/-- The number of toys Joel's friends donated -/
def friends_toys : ℕ := 75

/-- The total number of donated toys -/
def total_toys : ℕ := 108

theorem joel_donation :
  (friends_toys + sisters_toys + joels_toys = total_toys) ∧
  (joels_toys = 2 * sisters_toys) ∧
  (friends_toys = 18 + 42 + 2 + 13) :=
by sorry

end NUMINAMATH_CALUDE_joel_donation_l1204_120482


namespace NUMINAMATH_CALUDE_F_simplification_and_range_l1204_120405

noncomputable def f (t : ℝ) : ℝ := Real.sqrt ((1 - t) / (1 + t))

noncomputable def F (x : ℝ) : ℝ := Real.sin x * f (Real.cos x) + Real.cos x * f (Real.sin x)

theorem F_simplification_and_range (x : ℝ) (h : π < x ∧ x < 3 * π / 2) :
  F x = Real.sqrt 2 * Real.sin (x + π / 4) - 2 ∧
  ∃ y ∈ Set.Icc (-2 - Real.sqrt 2) (-3), F x = y :=
sorry

end NUMINAMATH_CALUDE_F_simplification_and_range_l1204_120405


namespace NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l1204_120497

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem eighteenth_term_of_sequence : arithmetic_sequence 3 4 18 = 71 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l1204_120497


namespace NUMINAMATH_CALUDE_point_coordinates_l1204_120468

/-- A point in the two-dimensional plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane. -/
def fourthQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis. -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis. -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Theorem: If a point P is in the fourth quadrant, its distance to the x-axis is 4,
    and its distance to the y-axis is 2, then its coordinates are (2, -4). -/
theorem point_coordinates (P : Point) 
  (h1 : fourthQuadrant P) 
  (h2 : distanceToXAxis P = 4) 
  (h3 : distanceToYAxis P = 2) : 
  P.x = 2 ∧ P.y = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1204_120468


namespace NUMINAMATH_CALUDE_petrol_price_reduction_l1204_120425

def original_price : ℝ := 4.444444444444445

theorem petrol_price_reduction (budget : ℝ) (additional_gallons : ℝ) 
  (h1 : budget = 200) 
  (h2 : additional_gallons = 5) :
  let reduced_price := budget / (budget / original_price + additional_gallons)
  (original_price - reduced_price) / original_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_petrol_price_reduction_l1204_120425


namespace NUMINAMATH_CALUDE_purchase_decision_l1204_120476

/-- Represents the prices and conditions for the company's purchase decision --/
structure PurchaseScenario where
  tablet_price : ℝ
  speaker_price : ℝ
  total_items : ℕ
  discount_rate1 : ℝ
  discount_threshold2 : ℝ
  discount_rate2 : ℝ

/-- Theorem stating the correct prices and cost-effective decision based on the number of tablets --/
theorem purchase_decision (p : PurchaseScenario) 
  (h1 : 2 * p.tablet_price + 3 * p.speaker_price = 7600)
  (h2 : 3 * p.tablet_price = 5 * p.speaker_price)
  (h3 : p.total_items = 30)
  (h4 : p.discount_rate1 = 0.1)
  (h5 : p.discount_threshold2 = 24000)
  (h6 : p.discount_rate2 = 0.2) :
  p.tablet_price = 2000 ∧ 
  p.speaker_price = 1200 ∧ 
  (∀ a : ℕ, a < 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) < 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) ∧
  (∀ a : ℕ, a = 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) = 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) ∧
  (∀ a : ℕ, a > 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) > 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) :=
by sorry

end NUMINAMATH_CALUDE_purchase_decision_l1204_120476


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_two_plus_sqrt_three_l1204_120402

theorem sqrt_sum_equals_sqrt_of_two_plus_sqrt_three (a b : ℚ) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) →
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_two_plus_sqrt_three_l1204_120402


namespace NUMINAMATH_CALUDE_remaining_jet_bars_to_sell_l1204_120496

def weekly_goal : ℕ := 90
def monday_sales : ℕ := 45
def tuesday_sales_difference : ℕ := 16

theorem remaining_jet_bars_to_sell :
  weekly_goal - (monday_sales + (monday_sales - tuesday_sales_difference)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_jet_bars_to_sell_l1204_120496


namespace NUMINAMATH_CALUDE_union_of_M_and_N_N_is_possible_set_l1204_120453

def M : Set ℕ := {1, 2}
def N : Set ℕ := {1, 3}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3} := by sorry

theorem N_is_possible_set :
  M = {1, 2} → M ∪ N = {1, 2, 3} → N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_N_is_possible_set_l1204_120453


namespace NUMINAMATH_CALUDE_borrowed_amount_is_6800_l1204_120489

/-- Calculates the amount borrowed given interest rates and total interest paid -/
def calculate_borrowed_amount (total_interest : ℚ) (rate1 rate2 rate3 : ℚ) 
  (period1 period2 period3 : ℚ) : ℚ :=
  total_interest / (rate1 * period1 + rate2 * period2 + rate3 * period3)

/-- Proves that the amount borrowed is 6800, given the specified conditions -/
theorem borrowed_amount_is_6800 : 
  let total_interest : ℚ := 8160
  let rate1 : ℚ := 12 / 100
  let rate2 : ℚ := 9 / 100
  let rate3 : ℚ := 13 / 100
  let period1 : ℚ := 3
  let period2 : ℚ := 5
  let period3 : ℚ := 3
  calculate_borrowed_amount total_interest rate1 rate2 rate3 period1 period2 period3 = 6800 := by
  sorry

#eval calculate_borrowed_amount 8160 (12/100) (9/100) (13/100) 3 5 3

end NUMINAMATH_CALUDE_borrowed_amount_is_6800_l1204_120489


namespace NUMINAMATH_CALUDE_function_root_property_l1204_120499

/-- Given a function f(x) = m · 2^x + x^2 + nx, if the set of roots of f(x) is equal to 
    the set of roots of f(f(x)) and is non-empty, then m+n is in the interval [0, 4). -/
theorem function_root_property (m n : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ m * (2^x) + x^2 + n*x
  (∃ x, f x = 0) ∧ 
  (∀ x, f x = 0 ↔ f (f x) = 0) →
  0 ≤ m + n ∧ m + n < 4 := by
sorry

end NUMINAMATH_CALUDE_function_root_property_l1204_120499


namespace NUMINAMATH_CALUDE_pyramid_volume_l1204_120483

/-- The volume of a pyramid with a triangular base and lateral faces forming 45° dihedral angles with the base -/
theorem pyramid_volume (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5) : 
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := S / p
  let H := r
  let V := (1/3) * S * H
  V = 6 := by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1204_120483


namespace NUMINAMATH_CALUDE_count_quadratic_integer_solutions_l1204_120455

theorem count_quadratic_integer_solutions :
  ∃ (S : Finset ℕ), 
    (∀ a ∈ S, a > 0 ∧ a ≤ 40) ∧
    (∀ a ∈ S, ∃ x y : ℤ, x ≠ y ∧ x^2 + (3*a + 2)*x + a^2 = 0 ∧ y^2 + (3*a + 2)*y + a^2 = 0) ∧
    (∀ a : ℕ, a > 0 → a ≤ 40 →
      (∃ x y : ℤ, x ≠ y ∧ x^2 + (3*a + 2)*x + a^2 = 0 ∧ y^2 + (3*a + 2)*y + a^2 = 0) →
      a ∈ S) ∧
    Finset.card S = 5 :=
sorry

end NUMINAMATH_CALUDE_count_quadratic_integer_solutions_l1204_120455


namespace NUMINAMATH_CALUDE_equation_one_solution_l1204_120450

theorem equation_one_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ - 2)^2 - 5 = 0 ∧ (x₂ - 2)^2 - 5 = 0 ∧ x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_equation_one_solution_l1204_120450


namespace NUMINAMATH_CALUDE_reflection_line_l1204_120473

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by y = k (horizontal line) -/
structure HorizontalLine where
  k : ℝ

/-- Reflection of a point about a horizontal line -/
def reflect (p : Point) (l : HorizontalLine) : Point :=
  ⟨p.x, 2 * l.k - p.y⟩

theorem reflection_line (p q r p' q' r' : Point) (l : HorizontalLine) :
  p = Point.mk (-3) 1 ∧
  q = Point.mk 5 (-2) ∧
  r = Point.mk 2 7 ∧
  p' = Point.mk (-3) (-9) ∧
  q' = Point.mk 5 (-8) ∧
  r' = Point.mk 2 (-3) ∧
  reflect p l = p' ∧
  reflect q l = q' ∧
  reflect r l = r' →
  l = HorizontalLine.mk (-4) := by
sorry

end NUMINAMATH_CALUDE_reflection_line_l1204_120473


namespace NUMINAMATH_CALUDE_ball_purchase_equation_l1204_120465

/-- Represents the price difference between a basketball and a soccer ball -/
def price_difference : ℝ := 20

/-- Represents the budget for basketballs -/
def basketball_budget : ℝ := 1500

/-- Represents the budget for soccer balls -/
def soccer_ball_budget : ℝ := 800

/-- Represents the quantity difference between basketballs and soccer balls purchased -/
def quantity_difference : ℝ := 5

/-- Theorem stating the equation that represents the relationship between
    the price of soccer balls and the quantities of basketballs and soccer balls purchased -/
theorem ball_purchase_equation (x : ℝ) :
  x > 0 →
  (basketball_budget / (x + price_difference) - soccer_ball_budget / x = quantity_difference) ↔
  (1500 / (x + 20) - 800 / x = 5) :=
by sorry

end NUMINAMATH_CALUDE_ball_purchase_equation_l1204_120465


namespace NUMINAMATH_CALUDE_function_value_2012_l1204_120437

theorem function_value_2012 (m n α₁ α₂ : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hα₁ : α₁ ≠ 0) (hα₂ : α₂ ≠ 0) :
  let f : ℝ → ℝ := λ x => m * Real.sin (π * x + α₁) + n * Real.cos (π * x + α₂)
  f 2011 = 1 → f 2012 = -1 := by
sorry

end NUMINAMATH_CALUDE_function_value_2012_l1204_120437


namespace NUMINAMATH_CALUDE_boar_sausages_problem_l1204_120484

theorem boar_sausages_problem (S : ℕ) : 
  (S > 0) →  -- Ensure S is positive
  (3 / 40 : ℚ) * S = 45 → 
  S = 600 := by 
sorry

end NUMINAMATH_CALUDE_boar_sausages_problem_l1204_120484


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1204_120413

theorem simplify_complex_fraction (b : ℝ) 
  (h1 : b ≠ 1/2) (h2 : b ≠ 1) : 
  1 - 2 / (1 + b / (1 - 2*b)) = (3*b - 1) / (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1204_120413


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1204_120418

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem parallel_line_through_point 
  (P : Point)
  (L1 : Line)
  (L2 : Line)
  (h1 : P.x = -1 ∧ P.y = 2)
  (h2 : L1.a = 2 ∧ L1.b = 1 ∧ L1.c = -5)
  (h3 : L2.a = 2 ∧ L2.b = 1 ∧ L2.c = 0)
  : parallel L1 L2 ∧ pointOnLine P L2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1204_120418


namespace NUMINAMATH_CALUDE_product_98_102_l1204_120406

theorem product_98_102 : 98 * 102 = 9996 := by
  sorry

end NUMINAMATH_CALUDE_product_98_102_l1204_120406


namespace NUMINAMATH_CALUDE_g_geq_h_implies_a_leq_one_l1204_120438

noncomputable def g (x : ℝ) : ℝ := Real.exp x - Real.exp 1 * x - 1

noncomputable def h (a x : ℝ) : ℝ := a * Real.sin x - Real.exp 1 * x

theorem g_geq_h_implies_a_leq_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g x ≥ h a x) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_g_geq_h_implies_a_leq_one_l1204_120438


namespace NUMINAMATH_CALUDE_valid_paths_count_l1204_120431

/-- Represents a point in the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents the grid with its dimensions and blocked points -/
structure Grid where
  width : Nat
  height : Nat
  blockedPoints : List GridPoint

/-- Calculates the number of valid paths in the grid -/
def countValidPaths (g : Grid) : Nat :=
  sorry

/-- The specific grid from the problem -/
def problemGrid : Grid :=
  { width := 5
  , height := 3
  , blockedPoints := [⟨2, 1⟩, ⟨3, 1⟩] }

theorem valid_paths_count :
  countValidPaths problemGrid = 39 :=
by sorry

end NUMINAMATH_CALUDE_valid_paths_count_l1204_120431


namespace NUMINAMATH_CALUDE_shooting_probability_l1204_120459

theorem shooting_probability (accuracy : ℝ) (consecutive_hits : ℝ) 
  (h1 : accuracy = 9/10) 
  (h2 : consecutive_hits = 1/2) : 
  consecutive_hits / accuracy = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probability_l1204_120459


namespace NUMINAMATH_CALUDE_mixed_doubles_probability_l1204_120436

/-- The number of athletes -/
def total_athletes : ℕ := 6

/-- The number of male athletes -/
def male_athletes : ℕ := 3

/-- The number of female athletes -/
def female_athletes : ℕ := 3

/-- The number of coaches -/
def coaches : ℕ := 3

/-- The number of players each coach selects -/
def players_per_coach : ℕ := 2

/-- The probability of all coaches forming mixed doubles teams -/
def probability_mixed_doubles : ℚ := 2/5

theorem mixed_doubles_probability :
  let total_outcomes := (total_athletes.choose players_per_coach * 
                         (total_athletes - players_per_coach).choose players_per_coach * 
                         (total_athletes - 2*players_per_coach).choose players_per_coach) / coaches.factorial
  let favorable_outcomes := male_athletes.choose 1 * female_athletes.choose 1 * 
                            (male_athletes - 1).choose 1 * (female_athletes - 1).choose 1 * 
                            (male_athletes - 2).choose 1 * (female_athletes - 2).choose 1 * 
                            coaches.factorial
  (favorable_outcomes : ℚ) / total_outcomes = probability_mixed_doubles :=
sorry

end NUMINAMATH_CALUDE_mixed_doubles_probability_l1204_120436


namespace NUMINAMATH_CALUDE_computer_sales_ratio_l1204_120498

theorem computer_sales_ratio (total : ℕ) (netbook_fraction : ℚ) (desktops : ℕ) : 
  total = 72 → netbook_fraction = 1/3 → desktops = 12 → 
  (total - (netbook_fraction * total).num - desktops : ℚ) / total = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_computer_sales_ratio_l1204_120498


namespace NUMINAMATH_CALUDE_cylinder_radius_l1204_120452

/-- Given a cylinder with height 8 cm and surface area 130π cm², prove its base circle radius is 5 cm. -/
theorem cylinder_radius (h : ℝ) (S : ℝ) (r : ℝ) 
  (height_eq : h = 8)
  (surface_area_eq : S = 130 * Real.pi)
  (surface_area_formula : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 := by sorry

end NUMINAMATH_CALUDE_cylinder_radius_l1204_120452


namespace NUMINAMATH_CALUDE_pants_original_price_l1204_120475

theorem pants_original_price 
  (total_spent : ℝ)
  (jacket_discount : ℝ)
  (pants_discount : ℝ)
  (jacket_original : ℝ)
  (h1 : total_spent = 306)
  (h2 : jacket_discount = 0.7)
  (h3 : pants_discount = 0.8)
  (h4 : jacket_original = 300) :
  ∃ (pants_original : ℝ), 
    jacket_original * jacket_discount + pants_original * pants_discount = total_spent ∧ 
    pants_original = 120 :=
by sorry

end NUMINAMATH_CALUDE_pants_original_price_l1204_120475


namespace NUMINAMATH_CALUDE_tenth_student_score_l1204_120427

/-- Represents a valid arithmetic sequence of exam scores -/
structure ExamScores where
  scores : Fin 10 → ℕ
  is_arithmetic : ∀ i j k : Fin 10, i.val + k.val = j.val + j.val → scores i + scores k = scores j + scores j
  max_score : ∀ i : Fin 10, scores i ≤ 100
  sum_middle : scores 2 + scores 3 + scores 4 + scores 5 = 354
  contains_96 : ∃ i : Fin 10, scores i = 96

/-- The theorem stating the possible scores for the 10th student -/
theorem tenth_student_score (e : ExamScores) : e.scores 0 = 61 ∨ e.scores 0 = 72 := by
  sorry

end NUMINAMATH_CALUDE_tenth_student_score_l1204_120427


namespace NUMINAMATH_CALUDE_expression_simplification_l1204_120466

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x - 1) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1) + 1) = (Real.sqrt 2 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1204_120466


namespace NUMINAMATH_CALUDE_oplus_four_two_l1204_120407

def oplus (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem oplus_four_two : oplus 4 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_oplus_four_two_l1204_120407


namespace NUMINAMATH_CALUDE_function_domain_range_l1204_120400

/-- Given a function f(x) = √(-5 / (ax² + ax - 3)) with domain R, 
    prove that the range of values for the real number a is (-12, 0]. -/
theorem function_domain_range (a : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (-5 / (a * x^2 + a * x - 3))) →
  a ∈ Set.Ioc (-12) 0 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_range_l1204_120400


namespace NUMINAMATH_CALUDE_largest_integer_solution_inequality_l1204_120451

theorem largest_integer_solution_inequality (x : ℤ) :
  (∀ y : ℤ, -y ≥ 2*y + 3 → y ≤ -1) ∧ (-(-1) ≥ 2*(-1) + 3) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_solution_inequality_l1204_120451


namespace NUMINAMATH_CALUDE_four_propositions_l1204_120444

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + x - m

-- Define what it means for a function to have zero points
def has_zero_points (f : ℝ → ℝ) : Prop := ∃ x, f x = 0

-- Define what it means for four points to be coplanar
def coplanar (E F G H : ℝ × ℝ × ℝ) : Prop := sorry

-- Define what it means for two lines to intersect
def lines_intersect (E F G H : ℝ × ℝ × ℝ) : Prop := sorry

-- Define what it means for an equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop := sorry

theorem four_propositions :
  (∀ m > 0, has_zero_points (f m)) ∧ 
  (∀ E F G H, ¬coplanar E F G H → ¬lines_intersect E F G H) ∧
  (∃ E F G H, ¬lines_intersect E F G H ∧ coplanar E F G H) ∧
  (∀ a : ℝ, (∀ x : ℝ, |x+1| + |x-1| ≥ a) ↔ a < 2) ∧
  (∀ m : ℝ, (0 < m ∧ m < 1) ↔ is_hyperbola m) :=
by
  sorry

end NUMINAMATH_CALUDE_four_propositions_l1204_120444


namespace NUMINAMATH_CALUDE_binomial_n_equals_10_l1204_120401

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p_range : 0 ≤ p ∧ p ≤ 1

/-- The variance of a binomial distribution -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a binomial distribution with p = 0.8 and variance 1.6, n = 10 -/
theorem binomial_n_equals_10 :
  ∀ X : BinomialRV, X.p = 0.8 → variance X = 1.6 → X.n = 10 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_equals_10_l1204_120401


namespace NUMINAMATH_CALUDE_monomial_sum_l1204_120432

/-- Given two monomials that form a monomial when added together, prove that m + n = 4 -/
theorem monomial_sum (m n : ℕ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), 2 * x^(m-1) * y^2 + (1/3) * x^2 * y^(n+1) = a * x^2 * y^2) → 
  m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_monomial_sum_l1204_120432


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l1204_120420

/-- The number of candy pieces eaten on Halloween night -/
def candy_eaten (katie_candy sister_candy remaining_candy : ℕ) : ℕ :=
  katie_candy + sister_candy - remaining_candy

/-- Theorem: Given the conditions, the number of candy pieces eaten is 9 -/
theorem halloween_candy_theorem :
  candy_eaten 10 6 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l1204_120420


namespace NUMINAMATH_CALUDE_log_xy_l1204_120464

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_xy (x y : ℝ) (h1 : log (x * y^5) = 2) (h2 : log (x^3 * y) = 2) :
  log (x * y) = 6/7 := by sorry

end NUMINAMATH_CALUDE_log_xy_l1204_120464


namespace NUMINAMATH_CALUDE_prime_sequence_multiple_of_six_l1204_120461

theorem prime_sequence_multiple_of_six (a d : ℤ) : 
  (Prime a ∧ a > 3) ∧ 
  (Prime (a + d) ∧ (a + d) > 3) ∧ 
  (Prime (a + 2*d) ∧ (a + 2*d) > 3) → 
  ∃ k : ℤ, d = 6 * k :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_multiple_of_six_l1204_120461


namespace NUMINAMATH_CALUDE_farmer_wheat_harvest_l1204_120493

/-- The farmer's wheat harvest problem -/
theorem farmer_wheat_harvest (estimated : ℕ) (actual : ℕ) 
  (h1 : estimated = 48097) 
  (h2 : actual = 48781) : 
  actual - estimated = 684 := by
  sorry

end NUMINAMATH_CALUDE_farmer_wheat_harvest_l1204_120493


namespace NUMINAMATH_CALUDE_line_equation_l1204_120491

/-- Given two points A(x₁,y₁) and B(x₂,y₂) satisfying the equations 3x₁ - 4y₁ - 2 = 0 and 3x₂ - 4y₂ - 2 = 0,
    the line passing through these points has the equation 3x - 4y - 2 = 0. -/
theorem line_equation (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 3 * x₁ - 4 * y₁ - 2 = 0) 
  (h₂ : 3 * x₂ - 4 * y₂ - 2 = 0) : 
  ∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) → 3 * x - 4 * y - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1204_120491


namespace NUMINAMATH_CALUDE_modulus_of_purely_imaginary_complex_l1204_120422

/-- If z is a purely imaginary complex number of the form a^2 - 1 + (a + 1)i where a is real,
    then the modulus of z is 2. -/
theorem modulus_of_purely_imaginary_complex (a : ℝ) :
  let z : ℂ := a^2 - 1 + (a + 1) * I
  (z.re = 0) → Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_purely_imaginary_complex_l1204_120422


namespace NUMINAMATH_CALUDE_unique_root_and_sequence_l1204_120442

theorem unique_root_and_sequence : ∃! r : ℝ, 
  (2 * r^3 + 5 * r - 2 = 0) ∧ 
  ∃! (a : ℕ → ℕ), (∀ n, a n < a (n+1)) ∧ 
    (2/5 : ℝ) = ∑' n, r^(a n) ∧
    ∀ n, a n = 3*n - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_and_sequence_l1204_120442


namespace NUMINAMATH_CALUDE_marsh_birds_total_l1204_120470

theorem marsh_birds_total (initial_geese ducks swans herons : ℕ) 
  (h1 : initial_geese = 58)
  (h2 : ducks = 37)
  (h3 : swans = 15)
  (h4 : herons = 22) :
  initial_geese * 2 + ducks + swans + herons = 190 := by
  sorry

end NUMINAMATH_CALUDE_marsh_birds_total_l1204_120470


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1204_120416

theorem sqrt_equation_solution : 
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1204_120416


namespace NUMINAMATH_CALUDE_add_1873_minutes_to_noon_l1204_120471

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24

def add_minutes (start_hour : ℕ) (start_minute : ℕ) (minutes_to_add : ℕ) : (ℕ × ℕ) :=
  let total_minutes := start_hour * minutes_in_hour + start_minute + minutes_to_add
  let final_hour := (total_minutes / minutes_in_hour) % hours_in_day
  let final_minute := total_minutes % minutes_in_hour
  (final_hour, final_minute)

theorem add_1873_minutes_to_noon :
  add_minutes 12 0 1873 = (19, 13) :=
sorry

end NUMINAMATH_CALUDE_add_1873_minutes_to_noon_l1204_120471


namespace NUMINAMATH_CALUDE_expression_evaluation_l1204_120481

theorem expression_evaluation :
  let expr1 := (27 / 8) ^ (-2/3) - (49 / 9) ^ (1/2) + (0.2)^(-2) * (3 / 25)
  let expr2 := -5 * (Real.log 4 / Real.log 9) + (Real.log (32 / 9) / Real.log 3) - 5^(Real.log 3 / Real.log 5)
  (expr1 = 10/9) ∧ (expr2 = -5 * (Real.log 2 / Real.log 3) - 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1204_120481


namespace NUMINAMATH_CALUDE_parents_present_l1204_120463

theorem parents_present (total_people : ℕ) (pupils : ℕ) (h1 : total_people = 676) (h2 : pupils = 654) :
  total_people - pupils = 22 := by
  sorry

end NUMINAMATH_CALUDE_parents_present_l1204_120463


namespace NUMINAMATH_CALUDE_power_equation_solution_l1204_120447

theorem power_equation_solution : ∃ y : ℝ, (12 : ℝ) ^ y * 6 ^ 3 / 432 = 72 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1204_120447


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1204_120424

theorem sum_of_three_numbers (A B C : ℝ) 
  (sum_eq : A + B + C = 2017)
  (A_eq : A = 2 * B - 3)
  (B_eq : B = 3 * C + 20) :
  A = 1213 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1204_120424


namespace NUMINAMATH_CALUDE_cost_of_paving_floor_l1204_120460

/-- The cost of paving a rectangular floor -/
theorem cost_of_paving_floor (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 300) :
  length * width * rate = 6187.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paving_floor_l1204_120460


namespace NUMINAMATH_CALUDE_rachel_apple_trees_l1204_120467

/-- The number of apple trees Rachel has -/
def num_trees : ℕ := 3

/-- The number of apples picked from each tree -/
def apples_per_tree : ℕ := 8

/-- The total number of apples remaining after picking -/
def apples_remaining : ℕ := 9

/-- The initial total number of apples on all trees -/
def initial_apples : ℕ := 33

theorem rachel_apple_trees :
  num_trees * apples_per_tree + apples_remaining = initial_apples :=
sorry

end NUMINAMATH_CALUDE_rachel_apple_trees_l1204_120467


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l1204_120454

theorem students_playing_neither_sport
  (total : ℕ)
  (football : ℕ)
  (tennis : ℕ)
  (both : ℕ)
  (h1 : total = 50)
  (h2 : football = 32)
  (h3 : tennis = 28)
  (h4 : both = 22) :
  total - (football + tennis - both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l1204_120454


namespace NUMINAMATH_CALUDE_square_rectangle_area_l1204_120417

/-- A rectangle composed of four identical squares with a given perimeter --/
structure SquareRectangle where
  side : ℝ  -- Side length of each square
  perim : ℝ  -- Perimeter of the rectangle
  perim_eq : perim = 10 * side  -- Perimeter equation

/-- The area of a SquareRectangle --/
def SquareRectangle.area (r : SquareRectangle) : ℝ := 4 * r.side^2

/-- Theorem: A SquareRectangle with perimeter 160 has an area of 1024 --/
theorem square_rectangle_area (r : SquareRectangle) (h : r.perim = 160) : r.area = 1024 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_l1204_120417


namespace NUMINAMATH_CALUDE_parabola_through_point_l1204_120411

/-- A parabola passing through the point (4, -2) has either the equation y^2 = x or x^2 = -8y -/
theorem parabola_through_point (x y : ℝ) : 
  (x = 4 ∧ y = -2) → (y^2 = x ∨ x^2 = -8*y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l1204_120411


namespace NUMINAMATH_CALUDE_unique_number_l1204_120403

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  Even n ∧ 
  n % 11 = 0 ∧ 
  is_perfect_cube (digit_product n) ∧
  n = 88 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_l1204_120403


namespace NUMINAMATH_CALUDE_solution_sets_union_l1204_120433

-- Define the solution sets M and N
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 6 = 0}
def N (q : ℝ) : Set ℝ := {x | x^2 + 6*x - q = 0}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (∃ (x : ℝ), x ∈ M p ∧ x ∈ N q) ∧ (M p ∩ N q = {2}) →
  M p ∪ N q = {2, 3, -8} :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_union_l1204_120433


namespace NUMINAMATH_CALUDE_correct_assignment_count_l1204_120469

/-- The number of ways to assign 5 friends to 5 rooms with at most 2 friends per room -/
def assignmentWays : ℕ := 1620

/-- A function that calculates the number of ways to assign n friends to m rooms with at most k friends per room -/
def calculateAssignmentWays (n m k : ℕ) : ℕ :=
  sorry  -- The actual implementation is not provided

theorem correct_assignment_count :
  calculateAssignmentWays 5 5 2 = assignmentWays :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l1204_120469


namespace NUMINAMATH_CALUDE_function_properties_l1204_120458

/-- Given a function f(x) = x - a*exp(x) + b, where a > 0 and b is real,
    this theorem states two properties:
    1. The maximum value of f(x) is ln(1/a) - 1 + b
    2. If f has two distinct zeros x₁ and x₂, then x₁ + x₂ < -2*ln(a) -/
theorem function_properties (a b : ℝ) (ha : a > 0) :
  let f := fun x => x - a * Real.exp x + b
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = Real.log (1 / a) - 1 + b) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = 0 → f x₂ = 0 → x₁ + x₂ < -2 * Real.log a) := by
  sorry


end NUMINAMATH_CALUDE_function_properties_l1204_120458


namespace NUMINAMATH_CALUDE_carol_goal_impossible_l1204_120421

theorem carol_goal_impossible (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (quizzes_taken : ℕ) (as_earned : ℕ) : 
  total_quizzes = 60 → 
  goal_percentage = 85 / 100 → 
  quizzes_taken = 40 → 
  as_earned = 26 → 
  ¬ ∃ (future_as : ℕ), 
    (as_earned + future_as : ℚ) / total_quizzes ≥ goal_percentage ∧ 
    future_as ≤ total_quizzes - quizzes_taken :=
by sorry

end NUMINAMATH_CALUDE_carol_goal_impossible_l1204_120421


namespace NUMINAMATH_CALUDE_sqrt_27_minus_sqrt_3_equals_2_sqrt_3_l1204_120443

theorem sqrt_27_minus_sqrt_3_equals_2_sqrt_3 : 
  Real.sqrt 27 - Real.sqrt 3 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_minus_sqrt_3_equals_2_sqrt_3_l1204_120443


namespace NUMINAMATH_CALUDE_true_false_questions_count_l1204_120495

def number_of_multiple_choice_questions : ℕ := 2
def choices_per_multiple_choice_question : ℕ := 4
def total_answer_key_combinations : ℕ := 480

def valid_true_false_combinations (n : ℕ) : ℕ := 2^n - 2

theorem true_false_questions_count :
  ∃ n : ℕ, 
    n > 0 ∧
    valid_true_false_combinations n * 
    choices_per_multiple_choice_question ^ number_of_multiple_choice_questions = 
    total_answer_key_combinations ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_true_false_questions_count_l1204_120495


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l1204_120446

/-- In a rectangle with a zigzag line, given specific angles, prove that ∠CDE is 11° --/
theorem zigzag_angle_theorem (ABC BCD DEF EFG : ℝ) (h1 : ABC = 10) (h2 : BCD = 14) 
  (h3 : DEF = 26) (h4 : EFG = 33) : ∃ (CDE : ℝ), CDE = 11 := by
  sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l1204_120446


namespace NUMINAMATH_CALUDE_carries_revenue_l1204_120448

/-- Represents the harvest quantities of vegetables -/
structure Harvest where
  tomatoes : ℕ
  carrots : ℕ
  eggplants : ℕ
  cucumbers : ℕ

/-- Represents the selling prices of vegetables -/
structure Prices where
  tomato : ℚ
  carrot : ℚ
  eggplant : ℚ
  cucumber : ℚ

/-- Calculates the total revenue from selling all vegetables -/
def totalRevenue (h : Harvest) (p : Prices) : ℚ :=
  h.tomatoes * p.tomato +
  h.carrots * p.carrot +
  h.eggplants * p.eggplant +
  h.cucumbers * p.cucumber

/-- Theorem stating that Carrie's total revenue is $1156.25 -/
theorem carries_revenue :
  let h : Harvest := { tomatoes := 200, carrots := 350, eggplants := 120, cucumbers := 75 }
  let p : Prices := { tomato := 1, carrot := 3/2, eggplant := 5/2, cucumber := 7/4 }
  totalRevenue h p = 4625/4 := by
  sorry

#eval (4625/4 : ℚ)  -- This should evaluate to 1156.25

end NUMINAMATH_CALUDE_carries_revenue_l1204_120448


namespace NUMINAMATH_CALUDE_candle_burn_theorem_l1204_120456

theorem candle_burn_theorem (t : ℝ) (h : t > 0) :
  let rate_second : ℝ := (3 / 5) / t
  let rate_third : ℝ := (4 / 7) / t
  let time_second_remaining : ℝ := (2 / 5) / rate_second
  let third_burned_while_second_finishes : ℝ := time_second_remaining * rate_third
  (3 / 7) - third_burned_while_second_finishes = 1 / 21 := by
sorry

end NUMINAMATH_CALUDE_candle_burn_theorem_l1204_120456


namespace NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_real_iff_l1204_120485

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 3 < x ∧ x < a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- First part of the theorem
theorem union_when_a_is_3 :
  A 3 ∪ B = {x | x < -1 ∨ x > 0} :=
sorry

-- Second part of the theorem
theorem union_equals_real_iff :
  ∀ a : ℝ, A a ∪ B = Set.univ ↔ 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_union_when_a_is_3_union_equals_real_iff_l1204_120485


namespace NUMINAMATH_CALUDE_radical_equality_implies_c_equals_six_l1204_120430

theorem radical_equality_implies_c_equals_six 
  (a b c : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (h : ∀ M : ℝ, M ≠ 1 → M^(1/a + 1/(a*b) + 3/(a*b*c)) = M^(14/24)) : 
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_radical_equality_implies_c_equals_six_l1204_120430


namespace NUMINAMATH_CALUDE_calculate_daily_fine_l1204_120419

/-- Calculates the daily fine for absence given contract details -/
theorem calculate_daily_fine (total_days : ℕ) (daily_pay : ℚ) (absent_days : ℕ) (total_payment : ℚ) : 
  total_days = 30 →
  daily_pay = 25 →
  absent_days = 10 →
  total_payment = 425 →
  (total_days - absent_days) * daily_pay - absent_days * (daily_pay - total_payment / (total_days - absent_days)) = total_payment →
  daily_pay - total_payment / (total_days - absent_days) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_daily_fine_l1204_120419


namespace NUMINAMATH_CALUDE_fifth_coaster_speed_l1204_120434

def rollercoaster_problem (S₁ S₂ S₃ S₄ S₅ : ℝ) : Prop :=
  S₁ = 50 ∧ S₂ = 62 ∧ S₃ = 73 ∧ S₄ = 70 ∧ (S₁ + S₂ + S₃ + S₄ + S₅) / 5 = 59

theorem fifth_coaster_speed :
  ∀ S₁ S₂ S₃ S₄ S₅ : ℝ,
  rollercoaster_problem S₁ S₂ S₃ S₄ S₅ →
  S₅ = 40 := by
  sorry


end NUMINAMATH_CALUDE_fifth_coaster_speed_l1204_120434


namespace NUMINAMATH_CALUDE_cos_neg_sixty_degrees_l1204_120415

theorem cos_neg_sixty_degrees : Real.cos (-(60 * π / 180)) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_neg_sixty_degrees_l1204_120415


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l1204_120426

def medium_bottle_capacity : ℕ := 120
def jumbo_bottle_capacity : ℕ := 2000

theorem minimum_bottles_needed : 
  (Nat.ceil (jumbo_bottle_capacity / medium_bottle_capacity : ℚ) : ℕ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l1204_120426


namespace NUMINAMATH_CALUDE_max_value_is_nine_l1204_120486

def max_value (a b c : ℕ) : ℕ := c * b^a

theorem max_value_is_nine :
  ∃ (a b c : ℕ), a ∈ ({1, 2, 3} : Set ℕ) ∧ b ∈ ({1, 2, 3} : Set ℕ) ∧ c ∈ ({1, 2, 3} : Set ℕ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  max_value a b c = 9 ∧
  ∀ (x y z : ℕ), x ∈ ({1, 2, 3} : Set ℕ) → y ∈ ({1, 2, 3} : Set ℕ) → z ∈ ({1, 2, 3} : Set ℕ) →
  x ≠ y → y ≠ z → x ≠ z →
  max_value x y z ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_is_nine_l1204_120486


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l1204_120409

def number_of_players : ℕ := 12
def lineup_size : ℕ := 5
def number_of_twins : ℕ := 2

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_lineup_count : 
  (number_of_twins * choose (number_of_players - number_of_twins) (lineup_size - 1)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l1204_120409


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l1204_120494

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a trapezoid formed by the arrangement of squares -/
structure Trapezoid where
  squares : List Square
  connector : ℝ × ℝ  -- Represents the connecting line segment

/-- Calculates the area of the trapezoid formed by the arrangement of squares -/
noncomputable def calculateTrapezoidArea (t : Trapezoid) : ℝ :=
  sorry

/-- The main theorem stating the area of the trapezoid -/
theorem trapezoid_area_theorem (s1 s2 s3 s4 : Square) 
  (h1 : s1.sideLength = 3)
  (h2 : s2.sideLength = 5)
  (h3 : s3.sideLength = 7)
  (h4 : s4.sideLength = 7)
  (t : Trapezoid)
  (ht : t.squares = [s1, s2, s3, s4]) :
  abs (calculateTrapezoidArea t - 12.83325) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_theorem_l1204_120494


namespace NUMINAMATH_CALUDE_two_identical_solutions_l1204_120439

/-- The value of k for which the equations y = x^2 and y = 4x + k have two identical solutions -/
def k_value : ℝ := -4

/-- First equation: y = x^2 -/
def eq1 (x y : ℝ) : Prop := y = x^2

/-- Second equation: y = 4x + k -/
def eq2 (x y k : ℝ) : Prop := y = 4*x + k

/-- Two identical solutions exist when k = k_value -/
theorem two_identical_solutions (k : ℝ) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    eq1 x₁ y₁ ∧ eq2 x₁ y₁ k ∧ 
    eq1 x₂ y₂ ∧ eq2 x₂ y₂ k) ↔ 
  k = k_value :=
sorry

end NUMINAMATH_CALUDE_two_identical_solutions_l1204_120439


namespace NUMINAMATH_CALUDE_ratio_equality_l1204_120490

theorem ratio_equality {a b c d : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a / b = c / d) : a / c = b / d := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1204_120490


namespace NUMINAMATH_CALUDE_eighth_term_value_l1204_120440

/-- An arithmetic sequence with 30 terms, first term 3, and last term 87 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (87 - 3) / (30 - 1)
  3 + (n - 1) * d

/-- The 8th term of the arithmetic sequence -/
def eighth_term : ℚ := arithmetic_sequence 8

theorem eighth_term_value : eighth_term = 675 / 29 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1204_120440


namespace NUMINAMATH_CALUDE_triangle_area_l1204_120457

/-- The area of a triangle with vertices (0,4,13), (-2,3,9), and (-5,6,9) is (3√30)/4 -/
theorem triangle_area : 
  let A : ℝ × ℝ × ℝ := (0, 4, 13)
  let B : ℝ × ℝ × ℝ := (-2, 3, 9)
  let C : ℝ × ℝ × ℝ := (-5, 6, 9)
  let area := Real.sqrt (
    let s := (Real.sqrt 21 + 3 * Real.sqrt 2 + 3 * Real.sqrt 5) / 2
    s * (s - Real.sqrt 21) * (s - 3 * Real.sqrt 2) * (s - 3 * Real.sqrt 5)
  )
  area = 3 * Real.sqrt 30 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1204_120457


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l1204_120449

/-- Calculates the gain percentage for a dishonest shopkeeper using a false weight -/
theorem shopkeeper_gain_percentage (false_weight : ℝ) (true_weight : ℝ) : 
  false_weight = 960 →
  true_weight = 1000 →
  (true_weight - false_weight) / false_weight * 100 = (1000 - 960) / 960 * 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l1204_120449


namespace NUMINAMATH_CALUDE_removed_triangles_area_l1204_120488

/-- Given a square with side length x, from which isosceles right triangles
    are removed from each corner to form a rectangle with diagonal 15,
    prove that the total area of the four removed triangles is 112.5. -/
theorem removed_triangles_area (x : ℝ) (r s : ℝ) : 
  (x - r)^2 + (x - s)^2 = 15^2 →
  r + s = x →
  (4 : ℝ) * (1/2 * r * s) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l1204_120488


namespace NUMINAMATH_CALUDE_sports_club_probability_l1204_120478

/-- The probability of selecting two girls when randomly choosing two members from a group. -/
def probability_two_girls (total : ℕ) (girls : ℕ) : ℚ :=
  (girls.choose 2 : ℚ) / (total.choose 2 : ℚ)

/-- The theorem stating the probability of selecting two girls from the sports club. -/
theorem sports_club_probability :
  let total := 15
  let girls := 8
  probability_two_girls total girls = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_probability_l1204_120478


namespace NUMINAMATH_CALUDE_problem_solution_l1204_120474

theorem problem_solution (a b : ℕ+) 
  (h1 : Nat.lcm a b = 5040)
  (h2 : Nat.gcd a b = 24)
  (h3 : a = 240) :
  b = 504 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1204_120474
