import Mathlib

namespace vector_collinearity_l263_26303

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

theorem vector_collinearity (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  collinear a (a.1 - b.1, a.2 - b.2) → x = 1/2 := by
sorry

end vector_collinearity_l263_26303


namespace sons_age_l263_26301

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end sons_age_l263_26301


namespace square_factor_l263_26397

theorem square_factor (a b : ℝ) (square : ℝ) :
  square * (3 * a * b) = 3 * a^2 * b → square = a := by
  sorry

end square_factor_l263_26397


namespace max_a_for_monotonic_increasing_l263_26327

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_increasing (a : ℝ) :
  a > 0 →
  (∀ x ≥ 1, Monotone (fun x => f a x)) →
  a ≤ 3 := by
  sorry

end max_a_for_monotonic_increasing_l263_26327


namespace men_entered_room_l263_26371

theorem men_entered_room (initial_men : ℕ) (initial_women : ℕ) (men_entered : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  2 * (initial_women - 3) = 24 →
  initial_men + men_entered = 14 →
  men_entered = 2 := by
  sorry

end men_entered_room_l263_26371


namespace total_travel_ways_l263_26384

-- Define the number of services for each mode of transportation
def bus_services : ℕ := 8
def train_services : ℕ := 3
def ferry_services : ℕ := 2

-- Theorem statement
theorem total_travel_ways : bus_services + train_services + ferry_services = 13 := by
  sorry

end total_travel_ways_l263_26384


namespace stewart_farm_sheep_count_l263_26326

theorem stewart_farm_sheep_count :
  ∀ (num_sheep num_horses : ℕ),
    (num_sheep : ℚ) / num_horses = 4 / 7 →
    num_horses * 230 = 12880 →
    num_sheep = 32 := by
  sorry

end stewart_farm_sheep_count_l263_26326


namespace final_price_approximation_l263_26358

/-- Represents the price reduction scenario for oil --/
structure OilPriceReduction where
  initialPrice : ℝ
  week1Reduction : ℝ := 0.10
  week2Reduction : ℝ := 0.15
  week3Reduction : ℝ := 0.20
  additionalQuantity : ℝ := 5
  fixedCost : ℝ := 800

/-- Calculates the final price after three weeks of reductions --/
def finalPrice (opr : OilPriceReduction) : ℝ :=
  opr.initialPrice * (1 - opr.week1Reduction) * (1 - opr.week2Reduction) * (1 - opr.week3Reduction)

/-- Theorem stating the final reduced price is approximately 62.06 --/
theorem final_price_approximation (opr : OilPriceReduction) : 
  ∃ (initialQuantity : ℝ), 
    opr.fixedCost = initialQuantity * opr.initialPrice ∧
    opr.fixedCost = (initialQuantity + opr.additionalQuantity) * (finalPrice opr) ∧
    abs ((finalPrice opr) - 62.06) < 0.01 := by
  sorry

end final_price_approximation_l263_26358


namespace leastDivisorTheorem_l263_26316

/-- The least positive integer that divides 16800 to get a number that is both a perfect square and a perfect cube -/
def leastDivisor : ℕ := 8400

/-- 16800 divided by the least divisor is a perfect square -/
def isPerfectSquare : Prop :=
  ∃ m : ℕ, (16800 / leastDivisor) = m * m

/-- 16800 divided by the least divisor is a perfect cube -/
def isPerfectCube : Prop :=
  ∃ m : ℕ, (16800 / leastDivisor) = m * m * m

/-- The main theorem stating that leastDivisor is the smallest positive integer
    that divides 16800 to get both a perfect square and a perfect cube -/
theorem leastDivisorTheorem :
  isPerfectSquare ∧ isPerfectCube ∧
  ∀ n : ℕ, 0 < n ∧ n < leastDivisor →
    ¬(∃ m : ℕ, (16800 / n) = m * m) ∨ ¬(∃ m : ℕ, (16800 / n) = m * m * m) :=
sorry

end leastDivisorTheorem_l263_26316


namespace unique_valid_number_l263_26353

def is_valid_number (abc : ℕ) : Prop :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  let sum := a + b + c
  abc % sum = 1 ∧
  (c * 100 + b * 10 + a) % sum = 1 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > c

theorem unique_valid_number : ∃! abc : ℕ, 100 ≤ abc ∧ abc < 1000 ∧ is_valid_number abc :=
  sorry

end unique_valid_number_l263_26353


namespace carpet_cost_l263_26362

/-- The cost of a carpet with increased dimensions -/
theorem carpet_cost (b₁ : ℝ) (l₁_factor : ℝ) (l₂_increase : ℝ) (b₂_increase : ℝ) (rate : ℝ) :
  b₁ = 6 →
  l₁_factor = 1.44 →
  l₂_increase = 0.4 →
  b₂_increase = 0.25 →
  rate = 45 →
  (b₁ * (1 + b₂_increase)) * (b₁ * l₁_factor * (1 + l₂_increase)) * rate = 4082.4 := by
  sorry

end carpet_cost_l263_26362


namespace f_properties_l263_26310

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x)^2 - Real.sqrt 3

/-- The theorem stating the properties of function f -/
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → (f x ≥ 1 ↔ x ∈ Set.Icc 0 (Real.pi / 4) ∪ Set.Icc (11 * Real.pi / 12) Real.pi)) :=
by sorry

end f_properties_l263_26310


namespace arithmetic_sequence_properties_l263_26321

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

def count_terms (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) : ℕ :=
  ((aₙ - a₁) / d).toNat + 1

def sum_multiples_of_10 (lst : List ℤ) : ℤ :=
  lst.filter (λ x => x % 10 = 0) |>.sum

theorem arithmetic_sequence_properties :
  let a₁ := -45
  let d := 7
  let aₙ := 98
  let n := count_terms a₁ d aₙ
  let seq := arithmetic_sequence a₁ d n
  n = 21 ∧ sum_multiples_of_10 seq = 60 := by sorry

end arithmetic_sequence_properties_l263_26321


namespace g_derivative_l263_26336

noncomputable def g (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem g_derivative (x : ℝ) : 
  deriv g x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
by sorry

end g_derivative_l263_26336


namespace product_of_repeating_decimal_and_eleven_l263_26375

/-- The repeating decimal 0.4567 as a rational number -/
def repeating_decimal : ℚ := 4567 / 9999

theorem product_of_repeating_decimal_and_eleven :
  11 * repeating_decimal = 50237 / 9999 := by sorry

end product_of_repeating_decimal_and_eleven_l263_26375


namespace parallel_line_difference_l263_26378

/-- Given two points (-1, q) and (-3, r) on a line parallel to y = (3/2)x + 1, 
    prove that r - q = -3 -/
theorem parallel_line_difference (q r : ℝ) : 
  (∃ (m b : ℝ), m = 3/2 ∧ 
    (∀ (x y : ℝ), y = m * x + b ↔ (x = -1 ∧ y = q) ∨ (x = -3 ∧ y = r))) →
  r - q = -3 :=
by sorry

end parallel_line_difference_l263_26378


namespace tate_additional_tickets_l263_26341

/-- The number of additional tickets Tate bought -/
def additional_tickets : ℕ := sorry

theorem tate_additional_tickets : 
  let initial_tickets : ℕ := 32
  let total_tickets : ℕ := initial_tickets + additional_tickets
  let peyton_tickets : ℕ := total_tickets / 2
  51 = total_tickets + peyton_tickets →
  additional_tickets = 2 := by sorry

end tate_additional_tickets_l263_26341


namespace perfect_square_equation_l263_26355

theorem perfect_square_equation (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) : 
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := by
  sorry

end perfect_square_equation_l263_26355


namespace fish_count_after_21_days_l263_26315

/-- Represents the state of the aquarium --/
structure AquariumState where
  days : ℕ
  fish : ℕ

/-- Calculates the number of fish eaten in a given number of days --/
def fishEaten (days : ℕ) : ℕ :=
  (2 + 3) * days

/-- Calculates the number of fish born in a given number of days --/
def fishBorn (days : ℕ) : ℕ :=
  2 * (days / 3)

/-- Updates the aquarium state for a given number of days --/
def updateAquarium (state : AquariumState) (days : ℕ) : AquariumState :=
  let newFish := max 0 (state.fish - fishEaten days + fishBorn days)
  { days := state.days + days, fish := newFish }

/-- Adds a specified number of fish to the aquarium --/
def addFish (state : AquariumState) (amount : ℕ) : AquariumState :=
  { state with fish := state.fish + amount }

/-- The final state of the aquarium after 21 days --/
def finalState : AquariumState :=
  let initialState : AquariumState := { days := 0, fish := 60 }
  let afterTwoWeeks := updateAquarium initialState 14
  let withAddedFish := addFish afterTwoWeeks 8
  updateAquarium withAddedFish 7

/-- The theorem stating that the number of fish after 21 days is 4 --/
theorem fish_count_after_21_days :
  finalState.fish = 4 := by sorry

end fish_count_after_21_days_l263_26315


namespace milk_problem_l263_26382

theorem milk_problem (M : ℝ) : 
  M > 0 → 
  (1 - 2/3) * (1 - 2/5) * (1 - 1/6) * M = 120 → 
  M = 720 :=
by
  sorry

end milk_problem_l263_26382


namespace intersection_implies_a_value_l263_26338

def A : Set ℝ := {-1, 0, 1}
def B (a : ℝ) : Set ℝ := {a + 1, 2 * a}

theorem intersection_implies_a_value (a : ℝ) :
  A ∩ B a = {0} → a = -1 := by sorry

end intersection_implies_a_value_l263_26338


namespace point_in_second_quadrant_l263_26360

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The theorem states that if the point P(x-1, x+1) is in the second quadrant and x is an integer, then x must be 0 -/
theorem point_in_second_quadrant (x : ℤ) : in_second_quadrant (x - 1 : ℝ) (x + 1 : ℝ) → x = 0 := by
  sorry

end point_in_second_quadrant_l263_26360


namespace x_eight_plus_x_four_plus_one_eq_zero_l263_26335

theorem x_eight_plus_x_four_plus_one_eq_zero 
  (x : ℂ) (h : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 := by
  sorry

end x_eight_plus_x_four_plus_one_eq_zero_l263_26335


namespace molecular_weight_proof_l263_26385

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Fe : ℝ := 55.85
def atomic_weight_H : ℝ := 1.01

-- Define the number of atoms for each element
def num_Al : ℕ := 2
def num_O : ℕ := 3
def num_Fe : ℕ := 2
def num_H : ℕ := 4

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_Al : ℝ) * atomic_weight_Al +
  (num_O : ℝ) * atomic_weight_O +
  (num_Fe : ℝ) * atomic_weight_Fe +
  (num_H : ℝ) * atomic_weight_H

-- Theorem statement
theorem molecular_weight_proof :
  molecular_weight = 217.70 := by sorry

end molecular_weight_proof_l263_26385


namespace solve_equation_l263_26308

theorem solve_equation : ∃ x : ℕ, x * 12 = 173 * 240 ∧ x = 3460 := by
  sorry

end solve_equation_l263_26308


namespace price_difference_theorem_l263_26333

-- Define the discounted price
def discounted_price : ℝ := 71.4

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the price increase rate
def increase_rate : ℝ := 0.25

-- Theorem statement
theorem price_difference_theorem :
  let original_price := discounted_price / (1 - discount_rate)
  let final_price := discounted_price * (1 + increase_rate)
  final_price - original_price = 5.25 := by
  sorry

end price_difference_theorem_l263_26333


namespace sunzi_wood_measurement_problem_l263_26302

/-- Represents the wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement_problem 
  (x y : ℝ) -- x: length of rope, y: length of wood
  (h1 : x - y = 4.5) -- condition: 4.5 feet of rope left when measuring
  (h2 : (1/2) * x + 1 = y) -- condition: 1 foot left when rope is folded in half
  : (x - y = 4.5) ∧ ((1/2) * x + 1 = y) := by
  sorry

end sunzi_wood_measurement_problem_l263_26302


namespace uncle_jude_cookies_l263_26343

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 15

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 23

/-- The number of cookies kept in the fridge -/
def fridge_cookies : ℕ := 188

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * tim_cookies

theorem uncle_jude_cookies : 
  total_cookies = tim_cookies + mike_cookies + anna_cookies + fridge_cookies := by
  sorry

end uncle_jude_cookies_l263_26343


namespace rectangle_longer_side_l263_26357

/-- Given a rectangular plot with one side of 10 meters, where fence poles are placed 5 meters apart
    and 24 poles are needed in total, the length of the longer side is 40 meters. -/
theorem rectangle_longer_side (width : ℝ) (length : ℝ) (poles : ℕ) :
  width = 10 →
  poles = 24 →
  (2 * width + 2 * length) / 5 = poles →
  length = 40 :=
by sorry

end rectangle_longer_side_l263_26357


namespace base_7_321_equals_162_l263_26350

def base_7_to_10 (a b c : ℕ) : ℕ := a * 7^2 + b * 7 + c

theorem base_7_321_equals_162 : base_7_to_10 3 2 1 = 162 := by
  sorry

end base_7_321_equals_162_l263_26350


namespace extraction_of_geometric_from_arithmetic_l263_26380

-- Define the arithmetic progression
def arithmeticProgression (a b : ℤ) (k : ℤ) : ℤ := a + b * k

-- Define the geometric progression
def geometricProgression (a b : ℤ) (k : ℕ) : ℤ := a * (b + 1)^k

theorem extraction_of_geometric_from_arithmetic (a b : ℤ) :
  ∃ (f : ℕ → ℤ), (∀ k : ℕ, ∃ l : ℤ, geometricProgression a b k = arithmeticProgression a b l) :=
sorry

end extraction_of_geometric_from_arithmetic_l263_26380


namespace perfect_square_factors_count_l263_26392

/-- The number of positive factors of 450 that are perfect squares -/
def perfect_square_factors_of_450 : ℕ :=
  let prime_factorization : ℕ × ℕ × ℕ := (1, 2, 2)  -- Exponents of 2, 3, and 5 in 450's factorization
  2 * 2 * 2  -- Number of ways to choose even exponents for each prime factor

theorem perfect_square_factors_count :
  perfect_square_factors_of_450 = 8 :=
by sorry

end perfect_square_factors_count_l263_26392


namespace complement_A_intersect_B_l263_26305

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {1, 2}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2} := by sorry

end complement_A_intersect_B_l263_26305


namespace notebook_marker_cost_l263_26339

/-- Given the cost of notebooks and markers, prove the cost of a specific combination -/
theorem notebook_marker_cost (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30)
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 := by
  sorry

end notebook_marker_cost_l263_26339


namespace ellipse_properties_l263_26328

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  right_focus_to_vertex : ℝ

/-- The standard form of an ellipse equation -/
def standard_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- A line in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The dot product of two 2D vectors -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem ellipse_properties (C : Ellipse) 
  (h1 : C.center = (0, 0))
  (h2 : C.foci_on_x_axis = true)
  (h3 : C.eccentricity = 1/2)
  (h4 : C.right_focus_to_vertex = 1) :
  (∃ (x y : ℝ), standard_equation 4 3 x y) ∧
  (∃ (l : Line) (A B : ℝ × ℝ), 
    (standard_equation 4 3 A.1 A.2) ∧
    (standard_equation 4 3 B.1 B.2) ∧
    (A.2 = l.slope * A.1 + l.intercept) ∧
    (B.2 = l.slope * B.1 + l.intercept) ∧
    (dot_product A B = 0)) ∧
  (∀ (m : ℝ), (∃ (k : ℝ), 
    ∃ (A B : ℝ × ℝ),
      (standard_equation 4 3 A.1 A.2) ∧
      (standard_equation 4 3 B.1 B.2) ∧
      (A.2 = k * A.1 + m) ∧
      (B.2 = k * B.1 + m) ∧
      (dot_product A B = 0)) ↔ 
    (m ≤ -2 * Real.sqrt 21 / 7 ∨ m ≥ 2 * Real.sqrt 21 / 7)) :=
by sorry

end ellipse_properties_l263_26328


namespace arithmetic_expression_equality_l263_26337

theorem arithmetic_expression_equality : 5 * 7 - 6 + 2 * 12 + 2 * 6 + 7 * 3 = 86 := by
  sorry

end arithmetic_expression_equality_l263_26337


namespace equation_solution_l263_26323

theorem equation_solution : ∃ x : ℝ, (x - 6) ^ 4 = (1 / 16)⁻¹ ∧ x = 8 := by
  sorry

end equation_solution_l263_26323


namespace quadratic_always_nonnegative_implies_a_range_l263_26381

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) → -1 < a ∧ a < 3 := by
  sorry

end quadratic_always_nonnegative_implies_a_range_l263_26381


namespace t_shape_perimeter_l263_26372

/-- A geometric figure composed of five identical squares arranged in a 'T' shape -/
structure TShape where
  /-- The total area of the figure in square centimeters -/
  total_area : ℝ
  /-- The figure is composed of five identical squares -/
  num_squares : ℕ
  /-- Assumption that the total area is 125 cm² -/
  area_assumption : total_area = 125
  /-- Assumption that the number of squares is 5 -/
  squares_assumption : num_squares = 5

/-- The perimeter of the 'T' shaped figure -/
def perimeter (t : TShape) : ℝ :=
  sorry

/-- Theorem stating that the perimeter of the 'T' shaped figure is 35 cm -/
theorem t_shape_perimeter (t : TShape) : perimeter t = 35 :=
  sorry

end t_shape_perimeter_l263_26372


namespace segment_length_given_ratio_points_l263_26334

/-- Represents a point on a line segment -/
structure PointOnSegment (A B : ℝ) where
  position : ℝ
  h1 : A ≤ position
  h2 : position ≤ B

/-- The length of a line segment -/
def segmentLength (A B : ℝ) : ℝ := B - A

theorem segment_length_given_ratio_points 
  (A B : ℝ) 
  (P Q : PointOnSegment A B)
  (h_order : A < P.position ∧ P.position < Q.position ∧ Q.position < B)
  (h_P_ratio : P.position - A = 3/8 * (B - A))
  (h_Q_ratio : Q.position - A = 2/5 * (B - A))
  (h_PQ_length : Q.position - P.position = 3)
  : segmentLength A B = 120 := by
  sorry

end segment_length_given_ratio_points_l263_26334


namespace prob_two_even_dice_l263_26309

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The set of even numbers on an 8-sided die -/
def even_numbers : Finset ℕ := {2, 4, 6, 8}

/-- The probability of rolling an even number on a single 8-sided die -/
def prob_even : ℚ := (even_numbers.card : ℚ) / sides

/-- The probability of rolling two even numbers on two 8-sided dice -/
theorem prob_two_even_dice : prob_even * prob_even = 1/4 := by sorry

end prob_two_even_dice_l263_26309


namespace remainder_14_power_53_mod_7_l263_26344

theorem remainder_14_power_53_mod_7 : 14^53 % 7 = 0 := by
  sorry

end remainder_14_power_53_mod_7_l263_26344


namespace negation_of_nonnegative_product_l263_26342

theorem negation_of_nonnegative_product (a b : ℝ) :
  ¬(a ≥ 0 ∧ b ≥ 0 → a * b ≥ 0) ↔ (a < 0 ∨ b < 0 → a * b < 0) := by sorry

end negation_of_nonnegative_product_l263_26342


namespace inverse_composition_nonexistence_l263_26317

theorem inverse_composition_nonexistence 
  (f h : ℝ → ℝ) 
  (h_def : ∀ x, f⁻¹ (h x) = 7 * x^2 + 4) : 
  ¬∃ y, h⁻¹ (f (-3)) = y :=
sorry

end inverse_composition_nonexistence_l263_26317


namespace complex_multiplication_l263_26376

theorem complex_multiplication (i : ℂ) :
  i * i = -1 →
  (-1 + i) * (2 - i) = -1 + 3 * i :=
by sorry

end complex_multiplication_l263_26376


namespace company_dividend_percentage_l263_26300

/-- Calculates the dividend percentage paid by a company given the face value of a share,
    the investor's return on investment, and the investor's purchase price per share. -/
def dividend_percentage (face_value : ℚ) (roi : ℚ) (purchase_price : ℚ) : ℚ :=
  (roi * purchase_price / face_value) * 100

/-- Theorem stating that under the given conditions, the dividend percentage is 18.5% -/
theorem company_dividend_percentage :
  let face_value : ℚ := 50
  let roi : ℚ := 25 / 100
  let purchase_price : ℚ := 37
  dividend_percentage face_value roi purchase_price = 185 / 10 := by
  sorry

end company_dividend_percentage_l263_26300


namespace min_value_xyz_l263_26388

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 27) :
  2 * x + 3 * y + 6 * z ≥ 54 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' * y' * z' = 27 ∧ 2 * x' + 3 * y' + 6 * z' = 54 := by
sorry

end min_value_xyz_l263_26388


namespace zero_points_count_midpoint_derivative_negative_l263_26311

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 1

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x - a

-- Theorem for the number of zero points
theorem zero_points_count (a : ℝ) (h : a > 0) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, f a x = 0 → x = x₁ ∨ x = x₂) ∨
  (∃! x, f a x = 0) ∨
  (∀ x, f a x ≠ 0) :=
sorry

-- Theorem for f'(x₀) < 0
theorem midpoint_derivative_negative (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : a > 0) (h₂ : x₁ < x₂) (h₃ : f a x₁ = 0) (h₄ : f a x₂ = 0) :
  let x₀ := (x₁ + x₂) / 2
  f_deriv a x₀ < 0 :=
sorry

end zero_points_count_midpoint_derivative_negative_l263_26311


namespace sled_distance_l263_26352

/-- Represents the distance traveled by a sled in a given second -/
def distance_in_second (n : ℕ) : ℕ := 6 + (n - 1) * 8

/-- Calculates the total distance traveled by the sled over a given number of seconds -/
def total_distance (seconds : ℕ) : ℕ :=
  (seconds * (distance_in_second 1 + distance_in_second seconds)) / 2

/-- Theorem stating that a sled sliding for 20 seconds travels 1640 inches -/
theorem sled_distance : total_distance 20 = 1640 := by
  sorry

end sled_distance_l263_26352


namespace intersection_complement_equals_l263_26398

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}

theorem intersection_complement_equals : A ∩ (U \ B) = {1, 3} := by sorry

end intersection_complement_equals_l263_26398


namespace smallest_a_for_sqrt_50a_l263_26383

theorem smallest_a_for_sqrt_50a (a : ℕ) : (∃ k : ℕ, k^2 = 50 * a) → a ≥ 2 :=
sorry

end smallest_a_for_sqrt_50a_l263_26383


namespace gcd_459_357_l263_26351

def euclidean_gcd (a b : ℕ) : ℕ := sorry

def successive_subtraction_gcd (a b : ℕ) : ℕ := sorry

theorem gcd_459_357 : 
  euclidean_gcd 459 357 = 51 ∧ 
  successive_subtraction_gcd 459 357 = 51 := by sorry

end gcd_459_357_l263_26351


namespace arc_length_ln_sin_l263_26365

open Real MeasureTheory

/-- The arc length of the curve y = ln(sin x) from x = π/3 to x = π/2 is (1/2) ln 3 -/
theorem arc_length_ln_sin (f : ℝ → ℝ) (h : ∀ x, f x = Real.log (Real.sin x)) :
  ∫ x in Set.Icc (π/3) (π/2), sqrt (1 + (deriv f x)^2) = (1/2) * Real.log 3 := by
  sorry

end arc_length_ln_sin_l263_26365


namespace min_value_arithmetic_seq_l263_26396

/-- An arithmetic sequence with positive terms and a_4 = 5 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  a 4 = 5

/-- The minimum value of 1/a_2 + 16/a_6 for the given arithmetic sequence -/
theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h : ArithmeticSequence a) :
    (∀ n, a n > 0) → (1 / a 2 + 16 / a 6) ≥ 5/2 := by
  sorry

end min_value_arithmetic_seq_l263_26396


namespace matrix_equation_solution_l263_26318

theorem matrix_equation_solution :
  let M : ℂ → Matrix (Fin 2) (Fin 2) ℂ := λ x => !![3*x, 3; 2*x, x]
  ∀ x : ℂ, M x = (-6 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ) ↔ x = 1 + I ∨ x = 1 - I :=
by sorry

end matrix_equation_solution_l263_26318


namespace cookies_left_l263_26345

def cookies_per_tray : ℕ := 12

def daily_trays : List ℕ := [2, 3, 4, 5, 3, 4, 4]

def frank_daily_consumption : ℕ := 2

def ted_consumption : List (ℕ × ℕ) := [(2, 3), (4, 5)]

def jan_consumption : ℕ × ℕ := (3, 5)

def tom_consumption : ℕ × ℕ := (5, 8)

def neighbours_consumption : ℕ × ℕ := (6, 20)

def total_baked (trays : List ℕ) (cookies_per_tray : ℕ) : ℕ :=
  (trays.map (· * cookies_per_tray)).sum

def total_eaten (frank_daily : ℕ) (ted : List (ℕ × ℕ)) (jan : ℕ × ℕ) (tom : ℕ × ℕ) (neighbours : ℕ × ℕ) : ℕ :=
  7 * frank_daily + (ted.map Prod.snd).sum + jan.snd + tom.snd + neighbours.snd

theorem cookies_left : 
  total_baked daily_trays cookies_per_tray - 
  total_eaten frank_daily_consumption ted_consumption jan_consumption tom_consumption neighbours_consumption = 245 := by
  sorry

end cookies_left_l263_26345


namespace max_height_sphere_hemispheres_tower_l263_26369

/-- The maximum height of a tower consisting of a sphere and three hemispheres -/
theorem max_height_sphere_hemispheres_tower (r₀ : ℝ) (h : r₀ = 2017) : 
  ∃ (r₁ r₂ r₃ : ℝ), 
    r₀ ≥ r₁ ∧ r₁ ≥ r₂ ∧ r₂ ≥ r₃ ∧ r₃ > 0 ∧
    r₀ + Real.sqrt (4 * r₀^2) = 3 * r₀ ∧
    3 * r₀ = 6051 :=
by sorry

end max_height_sphere_hemispheres_tower_l263_26369


namespace class_group_size_l263_26387

theorem class_group_size (boys girls groups : ℕ) 
  (h_boys : boys = 9) 
  (h_girls : girls = 12) 
  (h_groups : groups = 7) : 
  (boys + girls) / groups = 3 := by
sorry

end class_group_size_l263_26387


namespace symmetry_center_x_value_l263_26312

/-- Given a function f(x) = 1/2 * sin(ω*x + π/6) with ω > 0, 
    if its graph is tangent to a line y = m with distance π between adjacent tangent points,
    and A(x₀, y₀) is a symmetry center of y = f(x) with x₀ ∈ [0, π/2],
    then x₀ = 5π/12 -/
theorem symmetry_center_x_value (ω : ℝ) (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  ω > 0 →
  (∃ (k : ℤ), x₀ = k * π - π / 12) →
  x₀ ∈ Set.Icc 0 (π / 2) →
  (∀ (x : ℝ), (1 / 2) * Real.sin (ω * x + π / 6) = m → 
    ∃ (n : ℤ), x = n * π / ω) →
  x₀ = 5 * π / 12 := by
  sorry

end symmetry_center_x_value_l263_26312


namespace x_gt_one_sufficient_not_necessary_for_x_gt_zero_l263_26307

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_gt_zero_l263_26307


namespace fraction_simplification_l263_26367

theorem fraction_simplification (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  (x^8 + 2*x^4*y^2 + y^4) / (x^4 + y^2) = 85 := by
  sorry

end fraction_simplification_l263_26367


namespace f_2_eq_0_f_positive_solution_set_l263_26374

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (3-a)*x + 2*(1-a)

-- Theorem for f(2) = 0
theorem f_2_eq_0 (a : ℝ) : f a 2 = 0 := by sorry

-- Define the solution set for f(x) > 0
def solution_set (a : ℝ) : Set ℝ :=
  if a < -1 then {x | x < 2 ∨ x > 1-a}
  else if a = -1 then ∅
  else {x | 1-a < x ∧ x < 2}

-- Theorem for the solution set of f(x) > 0
theorem f_positive_solution_set (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ f a x > 0 := by sorry

end f_2_eq_0_f_positive_solution_set_l263_26374


namespace prob_at_least_one_ace_value_l263_26320

/-- The number of cards in two standard decks -/
def total_cards : ℕ := 104

/-- The number of aces in two standard decks -/
def total_aces : ℕ := 8

/-- The probability of drawing at least one ace when two cards are chosen
    sequentially with replacement from a deck of two standard decks -/
def prob_at_least_one_ace : ℚ :=
  1 - (1 - total_aces / total_cards) ^ 2

theorem prob_at_least_one_ace_value :
  prob_at_least_one_ace = 25 / 169 := by
  sorry

end prob_at_least_one_ace_value_l263_26320


namespace count_nonzero_monomials_l263_26379

/-- The number of monomials with non-zero coefficients in the expansion of (x+y+z)^2028 + (x-y-z)^2028 -/
def num_nonzero_monomials : ℕ := 1030225

/-- The exponent in the given expression -/
def exponent : ℕ := 2028

theorem count_nonzero_monomials :
  num_nonzero_monomials = (exponent / 2 + 1)^2 := by sorry

end count_nonzero_monomials_l263_26379


namespace min_cost_45_ropes_l263_26322

/-- Represents the cost and quantity of ropes --/
structure RopePurchase where
  costA : ℝ  -- Cost of one rope A
  costB : ℝ  -- Cost of one rope B
  quantA : ℕ -- Quantity of rope A
  quantB : ℕ -- Quantity of rope B

/-- Calculates the total cost of a rope purchase --/
def totalCost (p : RopePurchase) : ℝ :=
  p.costA * p.quantA + p.costB * p.quantB

/-- Theorem stating the minimum cost for purchasing 45 ropes --/
theorem min_cost_45_ropes (p : RopePurchase) :
  p.quantA + p.quantB = 45 →
  10 * 10 + 5 * 15 = 175 →
  15 * 10 + 10 * 15 = 300 →
  548 ≤ totalCost p →
  totalCost p ≤ 560 →
  ∃ (q : RopePurchase), 
    q.costA = 10 ∧ 
    q.costB = 15 ∧ 
    q.quantA = 25 ∧ 
    q.quantB = 20 ∧ 
    totalCost q = 550 ∧ 
    totalCost q ≤ totalCost p :=
by
  sorry


end min_cost_45_ropes_l263_26322


namespace sin_seven_pi_sixths_l263_26314

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end sin_seven_pi_sixths_l263_26314


namespace petya_ice_cream_l263_26393

theorem petya_ice_cream (ice_cream_cost : ℕ) (petya_money : ℕ) : 
  ice_cream_cost = 2000 →
  petya_money = 400^5 - 399^2 * (400^3 + 2 * 400^2 + 3 * 400 + 4) →
  petya_money < ice_cream_cost :=
by
  sorry

end petya_ice_cream_l263_26393


namespace boys_camp_total_l263_26348

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))  -- 20% of boys are from school A
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))  -- 30% of boys from school A study science
  (h3 : (total : ℚ) * (1/5) * (7/10) = 49)  -- 49 boys are from school A but do not study science
  : total = 350 := by
sorry


end boys_camp_total_l263_26348


namespace find_A_value_l263_26373

theorem find_A_value : ∃ (A B : ℕ), 
  A < 10 ∧ B < 10 ∧ 
  10 * A + 8 + 30 + B = 99 ∧
  A = 6 := by
  sorry

end find_A_value_l263_26373


namespace solution_range_for_a_l263_26340

/-- The system of equations has a solution with distinct real x, y, z if and only if a is in (23/27, 1) -/
theorem solution_range_for_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 + y^2 + z^2 = a ∧
    x^2 + y^3 + z^2 = a ∧
    x^2 + y^2 + z^3 = a) ↔
  (23/27 < a ∧ a < 1) :=
sorry

end solution_range_for_a_l263_26340


namespace min_sum_squares_over_a_squared_l263_26366

theorem min_sum_squares_over_a_squared (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a ≠ 0) : 
  ((a + b)^2 + (b + c)^2 + (c + a)^2) / a^2 ≥ 4 := by
  sorry

end min_sum_squares_over_a_squared_l263_26366


namespace largest_k_for_distinct_roots_l263_26346

theorem largest_k_for_distinct_roots : 
  ∀ k : ℤ, 
  (∃ x y : ℝ, x ≠ y ∧ 
    (k - 2 : ℝ) * x^2 - 4 * x + 4 = 0 ∧ 
    (k - 2 : ℝ) * y^2 - 4 * y + 4 = 0) →
  k ≤ 1 :=
by sorry

end largest_k_for_distinct_roots_l263_26346


namespace gcf_of_45_135_90_l263_26370

theorem gcf_of_45_135_90 : Nat.gcd 45 (Nat.gcd 135 90) = 45 := by sorry

end gcf_of_45_135_90_l263_26370


namespace cameron_house_paintable_area_l263_26386

/-- Calculates the total paintable area of walls in multiple bedrooms -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Theorem stating that the total paintable area of walls in Cameron's house is 1840 square feet -/
theorem cameron_house_paintable_area :
  total_paintable_area 4 15 12 10 80 = 1840 := by
  sorry

#eval total_paintable_area 4 15 12 10 80

end cameron_house_paintable_area_l263_26386


namespace infinite_triples_with_gcd_one_l263_26354

theorem infinite_triples_with_gcd_one (m n : ℕ+) :
  ∃ (a b c : ℕ+),
    a = m^2 + m * n + n^2 ∧
    b = m^2 - m * n ∧
    c = n^2 - m * n ∧
    Nat.gcd a (Nat.gcd b c) = 1 ∧
    a^2 = b^2 + c^2 + b * c :=
by sorry

end infinite_triples_with_gcd_one_l263_26354


namespace inequality_equivalence_l263_26330

theorem inequality_equivalence :
  ∀ x : ℝ, |(7 - 2*x) / 4| < 3 ↔ -2.5 < x ∧ x < 9.5 := by
  sorry

end inequality_equivalence_l263_26330


namespace second_train_speed_l263_26331

/-- Proves that the speed of the second train is 60 km/h given the conditions of the problem -/
theorem second_train_speed
  (first_train_speed : ℝ)
  (time_difference : ℝ)
  (meeting_distance : ℝ)
  (h1 : first_train_speed = 40)
  (h2 : time_difference = 1)
  (h3 : meeting_distance = 120) :
  let second_train_speed := meeting_distance / (meeting_distance / first_train_speed - time_difference)
  second_train_speed = 60 := by
sorry

end second_train_speed_l263_26331


namespace division_problem_l263_26349

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 251 := by
sorry

end division_problem_l263_26349


namespace decagon_diagonals_from_vertex_l263_26391

/-- The number of diagonals from a vertex in a regular decagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals from a vertex in a regular decagon is 7 -/
theorem decagon_diagonals_from_vertex :
  diagonals_from_vertex decagon_sides = 7 := by
  sorry

end decagon_diagonals_from_vertex_l263_26391


namespace sum_of_A_and_B_l263_26319

theorem sum_of_A_and_B : ∀ A B : ℚ, 
  (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / (4 * A) ∧ 
  (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / B → 
  A + B = 40 := by
sorry

end sum_of_A_and_B_l263_26319


namespace probability_to_reach_target_is_correct_l263_26368

/-- Represents a point in the 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a step direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of taking a specific step -/
def stepProbability : ℚ := 1/4

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The target point -/
def targetPoint : Point := ⟨3, 3⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 8

/-- Calculate the probability of reaching the target point from the start point
    in at most the maximum number of steps -/
def probabilityToReachTarget : ℚ :=
  45/2048

theorem probability_to_reach_target_is_correct :
  probabilityToReachTarget = 45/2048 := by
  sorry

end probability_to_reach_target_is_correct_l263_26368


namespace rectangle_area_change_l263_26395

/-- Given a rectangle with initial dimensions 3 × 7 inches, if shortening one side by 2 inches 
    results in an area of 15 square inches, then shortening the other side by 2 inches 
    will result in an area of 7 square inches. -/
theorem rectangle_area_change (initial_width initial_length : ℝ) 
  (h1 : initial_width = 3)
  (h2 : initial_length = 7)
  (h3 : initial_width * (initial_length - 2) = 15 ∨ (initial_width - 2) * initial_length = 15) :
  (initial_width - 2) * initial_length = 7 ∨ initial_width * (initial_length - 2) = 7 :=
by sorry

end rectangle_area_change_l263_26395


namespace product_of_solutions_with_positive_real_part_l263_26394

theorem product_of_solutions_with_positive_real_part (x : ℂ) : 
  (x^8 = -256) → 
  (∃ (S : Finset ℂ), 
    (∀ z ∈ S, z^8 = -256 ∧ z.re > 0) ∧ 
    (∀ z, z^8 = -256 ∧ z.re > 0 → z ∈ S) ∧ 
    (S.prod id = 8)) :=
by sorry

end product_of_solutions_with_positive_real_part_l263_26394


namespace tangent_slope_at_two_l263_26399

/-- The function representing the curve y = x^2 + 3x -/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of the function f -/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_two :
  f' 2 = 7 := by sorry

end tangent_slope_at_two_l263_26399


namespace travel_ratio_l263_26325

-- Define variables for the number of countries each person traveled to
def george_countries : ℕ := 6
def zack_countries : ℕ := 18

-- Define functions for other travelers based on the given conditions
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := zack_countries / 2

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Theorem statement
theorem travel_ratio : ratio patrick_countries joseph_countries = 3 := by sorry

end travel_ratio_l263_26325


namespace quadratic_inequality_l263_26389

theorem quadratic_inequality (x : ℝ) : x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_l263_26389


namespace complex_equation_solution_l263_26356

theorem complex_equation_solution (z : ℂ) 
  (h1 : Complex.abs (1 - z) + z = 10 - 3*I) :
  ∃ (m n : ℝ), 
    z = 5 - 3*I ∧ 
    z^2 + m*z + n = 1 - 3*I ∧ 
    m = -9 ∧ 
    n = 30 := by
  sorry

end complex_equation_solution_l263_26356


namespace lens_discount_percentage_l263_26364

theorem lens_discount_percentage (original_price : ℝ) (discounted_price : ℝ) (saving : ℝ) :
  original_price = 300 ∧ 
  discounted_price = 220 ∧ 
  saving = 20 →
  (original_price - (discounted_price + saving)) / original_price * 100 = 20 := by
  sorry

end lens_discount_percentage_l263_26364


namespace quadratic_root_difference_l263_26304

theorem quadratic_root_difference (a b c : ℝ) (h : b^2 - 4*a*c ≥ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 ∧ a*x^2 + b*x + c = 0 ∧ r₁ * r₂ < 20 → |r₁ - r₂| = 2 :=
by
  sorry

#check quadratic_root_difference 1 (-8) 15

end quadratic_root_difference_l263_26304


namespace probability_at_least_one_male_doctor_l263_26361

/-- The probability of selecting at least one male doctor when choosing 3 doctors from 4 female and 3 male doctors. -/
theorem probability_at_least_one_male_doctor : 
  let total_doctors : ℕ := 7
  let female_doctors : ℕ := 4
  let male_doctors : ℕ := 3
  let doctors_to_select : ℕ := 3
  let total_combinations := Nat.choose total_doctors doctors_to_select
  let favorable_outcomes := 
    Nat.choose male_doctors 1 * Nat.choose female_doctors 2 +
    Nat.choose male_doctors 2 * Nat.choose female_doctors 1 +
    Nat.choose male_doctors 3
  (favorable_outcomes : ℚ) / total_combinations = 31 / 35 :=
by sorry

end probability_at_least_one_male_doctor_l263_26361


namespace group_size_l263_26329

theorem group_size (B F BF : ℕ) (h1 : B = 13) (h2 : F = 15) (h3 : BF = 18) : 
  B + F - BF + 3 = 13 := by
sorry

end group_size_l263_26329


namespace circle_area_ratio_l263_26347

theorem circle_area_ratio (C D : Real) (r_C r_D : ℝ) : 
  (60 / 360) * (2 * Real.pi * r_C) = (40 / 360) * (2 * Real.pi * r_D) →
  (Real.pi * r_C^2) / (Real.pi * r_D^2) = 9 / 4 := by
  sorry

end circle_area_ratio_l263_26347


namespace quadratic_touch_existence_l263_26363

theorem quadratic_touch_existence (p q r : ℤ) : 
  (∃ x : ℝ, p * x^2 + q * x + r = 0 ∧ ∀ y : ℝ, p * y^2 + q * y + r ≥ 0) →
  ∃ a b : ℤ, 
    (∃ x : ℝ, p * x^2 + q * x + r = (b : ℝ)) ∧
    (∃ x : ℝ, x^2 + (a : ℝ) * x + (b : ℝ) = 0 ∧ ∀ y : ℝ, y^2 + (a : ℝ) * y + (b : ℝ) ≥ 0) :=
by sorry

end quadratic_touch_existence_l263_26363


namespace problem_solution_l263_26390

theorem problem_solution (m n a : ℝ) : 
  (m^2 - 2*m - 1 = 0) →
  (n^2 - 2*n - 1 = 0) →
  (7*m^2 - 14*m + a)*(3*n^2 - 6*n - 7) = 8 →
  a = -9 := by sorry

end problem_solution_l263_26390


namespace simplify_expression_l263_26306

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x + y)⁻¹ * (x⁻¹ + y⁻¹) = x⁻¹ * y⁻¹ := by
  sorry

end simplify_expression_l263_26306


namespace car_stop_time_l263_26359

/-- The distance traveled by a car after braking -/
def S (t : ℝ) : ℝ := -3 * t^2 + 18 * t

/-- The time required for the car to stop after braking -/
theorem car_stop_time : ∃ t : ℝ, S t = 0 ∧ t = 6 := by
  sorry

end car_stop_time_l263_26359


namespace garden_ratio_l263_26377

/-- Represents a rectangular garden -/
structure RectangularGarden where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Theorem: For a rectangular garden with area 675 sq meters and width 15 meters, 
    the ratio of length to width is 3:1 -/
theorem garden_ratio (g : RectangularGarden) 
    (h1 : g.area = 675)
    (h2 : g.width = 15)
    (h3 : g.area = g.length * g.width) :
  g.length / g.width = 3 := by
  sorry

#check garden_ratio

end garden_ratio_l263_26377


namespace intersection_of_A_and_B_l263_26324

def set_A : Set ℝ := {y | ∃ x, y = Real.log x}
def set_B : Set ℝ := {x | x ≥ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ici (0 : ℝ) := by sorry

end intersection_of_A_and_B_l263_26324


namespace segment_length_product_l263_26332

theorem segment_length_product (b : ℝ) : 
  (∃ b₁ b₂ : ℝ, 
    (∀ b : ℝ, (((3*b - 7)^2 + (2*b + 1)^2 : ℝ) = 50) ↔ (b = b₁ ∨ b = b₂)) ∧ 
    (b₁ * b₂ = 0)) := by
  sorry

end segment_length_product_l263_26332


namespace even_digits_in_base9_567_l263_26313

/-- Converts a natural number from base 10 to base 9 -/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem even_digits_in_base9_567 : 
  countEvenDigits (toBase9 567) = 2 :=
sorry

end even_digits_in_base9_567_l263_26313
