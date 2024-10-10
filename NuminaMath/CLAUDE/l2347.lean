import Mathlib

namespace school_purchase_cost_l2347_234775

/-- Calculates the total cost of pencils and pens after all applicable discounts -/
def totalCostAfterDiscounts (pencilPrice penPrice : ℚ) (pencilCount penCount : ℕ) 
  (pencilDiscountThreshold penDiscountThreshold : ℕ) 
  (pencilDiscountRate penDiscountRate additionalDiscountRate : ℚ)
  (additionalDiscountThreshold : ℚ) : ℚ :=
  sorry

theorem school_purchase_cost : 
  let pencilPrice : ℚ := 2.5
  let penPrice : ℚ := 3.5
  let pencilCount : ℕ := 38
  let penCount : ℕ := 56
  let pencilDiscountThreshold : ℕ := 30
  let penDiscountThreshold : ℕ := 50
  let pencilDiscountRate : ℚ := 0.1
  let penDiscountRate : ℚ := 0.15
  let additionalDiscountRate : ℚ := 0.05
  let additionalDiscountThreshold : ℚ := 250

  totalCostAfterDiscounts pencilPrice penPrice pencilCount penCount 
    pencilDiscountThreshold penDiscountThreshold 
    pencilDiscountRate penDiscountRate additionalDiscountRate
    additionalDiscountThreshold = 239.5 := by
  sorry

end school_purchase_cost_l2347_234775


namespace angle_supplement_l2347_234739

theorem angle_supplement (angle : ℝ) : 
  (90 - angle = angle - 18) → (180 - angle = 126) := by
  sorry

end angle_supplement_l2347_234739


namespace modular_inverse_14_mod_1001_l2347_234746

theorem modular_inverse_14_mod_1001 :
  ∃ x : ℕ, x ≤ 1000 ∧ (14 * x) % 1001 = 1 :=
by
  use 143
  sorry

end modular_inverse_14_mod_1001_l2347_234746


namespace arithmetic_sequence_sum_l2347_234717

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 20 and S_20 = 15, prove S_30 = -15 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 10 = 20) 
  (h2 : a.S 20 = 15) : 
  a.S 30 = -15 := by
  sorry


end arithmetic_sequence_sum_l2347_234717


namespace remainder_101_35_mod_100_l2347_234770

theorem remainder_101_35_mod_100 : 101^35 % 100 = 1 := by
  sorry

end remainder_101_35_mod_100_l2347_234770


namespace candy_bar_cost_l2347_234776

/-- Given that Dan spent $13 in total on a candy bar and a chocolate, 
    and the chocolate costs $6, prove that the candy bar costs $7. -/
theorem candy_bar_cost (total_spent : ℕ) (chocolate_cost : ℕ) (candy_bar_cost : ℕ) : 
  total_spent = 13 → chocolate_cost = 6 → candy_bar_cost = 7 := by
  sorry

end candy_bar_cost_l2347_234776


namespace student_ranking_l2347_234705

theorem student_ranking (n : ℕ) 
  (rank_from_right : ℕ) 
  (rank_from_left : ℕ) 
  (h1 : rank_from_right = 17) 
  (h2 : rank_from_left = 5) : 
  n = rank_from_right + rank_from_left - 1 :=
by sorry

end student_ranking_l2347_234705


namespace movie_book_difference_l2347_234727

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 17

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 11

/-- Theorem: The difference between the number of movies and books in the 'crazy silly school' series is 6 -/
theorem movie_book_difference : num_movies - num_books = 6 := by
  sorry

end movie_book_difference_l2347_234727


namespace power_of_two_plus_one_l2347_234785

theorem power_of_two_plus_one (b m n : ℕ) : 
  b > 1 → 
  m > n → 
  (∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) → 
  ∃ k : ℕ, b + 1 = 2^k :=
sorry

end power_of_two_plus_one_l2347_234785


namespace mabel_transactions_l2347_234755

theorem mabel_transactions :
  ∀ (mabel anthony cal jade : ℕ),
    anthony = mabel + mabel / 10 →
    cal = (2 * anthony) / 3 →
    jade = cal + 17 →
    jade = 83 →
    mabel = 90 :=
by
  sorry

end mabel_transactions_l2347_234755


namespace midpoint_coordinate_product_l2347_234732

/-- The product of the coordinates of the midpoint of a line segment with endpoints (4, -1) and (-2, 7) is 3. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 4
  let y1 : ℝ := -1
  let x2 : ℝ := -2
  let y2 : ℝ := 7
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = 3 := by
  sorry

end midpoint_coordinate_product_l2347_234732


namespace absolute_value_equality_l2347_234723

theorem absolute_value_equality (x : ℝ) (h : x > 2) :
  |x - Real.sqrt ((x - 3)^2)| = 3 := by
  sorry

end absolute_value_equality_l2347_234723


namespace two_blue_marbles_probability_l2347_234769

def total_marbles : ℕ := 3 + 4 + 9

def blue_marbles : ℕ := 4

def probability_two_blue_marbles : ℚ :=
  (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1))

theorem two_blue_marbles_probability :
  probability_two_blue_marbles = 1 / 20 :=
sorry

end two_blue_marbles_probability_l2347_234769


namespace motion_analysis_l2347_234782

-- Define the motion function
def s (t : ℝ) : ℝ := t^2 + 2*t - 3

-- Define velocity as the derivative of s
def v (t : ℝ) : ℝ := 2*t + 2

-- Define acceleration as the derivative of v
def a : ℝ := 2

theorem motion_analysis :
  v 2 = 6 ∧ a = 2 :=
sorry

end motion_analysis_l2347_234782


namespace geometric_sequence_50th_term_l2347_234772

/-- The 50th term of a geometric sequence with first term 8 and second term -16 -/
theorem geometric_sequence_50th_term :
  let a₁ : ℝ := 8
  let a₂ : ℝ := -16
  let r : ℝ := a₂ / a₁
  let aₙ (n : ℕ) : ℝ := a₁ * r^(n - 1)
  aₙ 50 = -8 * 2^49 := by
  sorry

end geometric_sequence_50th_term_l2347_234772


namespace perfect_apples_l2347_234715

theorem perfect_apples (total : ℕ) (small_fraction : ℚ) (unripe_fraction : ℚ) :
  total = 30 →
  small_fraction = 1/6 →
  unripe_fraction = 1/3 →
  (total : ℚ) - small_fraction * total - unripe_fraction * total = 15 := by
  sorry

end perfect_apples_l2347_234715


namespace right_triangle_properties_l2347_234752

theorem right_triangle_properties (A B C : ℝ) (h_right : A = 90) (h_tan : Real.tan C = 5) (h_hypotenuse : A = 80) :
  let AB := 80 * (5 / Real.sqrt 26)
  let BC := 80 / Real.sqrt 26
  (AB = 80 * (5 / Real.sqrt 26)) ∧ (BC / AB = 1 / 5) := by
  sorry

end right_triangle_properties_l2347_234752


namespace daniel_noodles_left_l2347_234756

/-- Given that Daniel initially had 66 noodles and gave 12 noodles to William,
    prove that he now has 54 noodles. -/
theorem daniel_noodles_left (initial : ℕ) (given : ℕ) (h1 : initial = 66) (h2 : given = 12) :
  initial - given = 54 := by sorry

end daniel_noodles_left_l2347_234756


namespace problem_solution_l2347_234760

theorem problem_solution : ∃ N : ℕ, 
  (N / (555 + 445) = 2 * (555 - 445)) ∧ 
  (N % (555 + 445) = 50) ∧ 
  N = 220050 := by
  sorry

end problem_solution_l2347_234760


namespace tracy_book_collection_l2347_234704

theorem tracy_book_collection (first_week : ℕ) (total_books : ℕ) : 
  total_books = 99 → 
  first_week + 5 * (10 * first_week) = total_books →
  first_week = 9 := by
sorry

end tracy_book_collection_l2347_234704


namespace snow_probability_l2347_234724

theorem snow_probability (p : ℝ) (h : p = 3/4) : 
  1 - (1 - p)^5 = 1023/1024 := by
sorry

end snow_probability_l2347_234724


namespace target_destruction_probabilities_l2347_234787

/-- Represents the probability of a person hitting a target -/
def HitProbability := Fin 2 → Fin 2 → ℚ

/-- The probability of person A hitting targets -/
def probA : HitProbability := fun i j => 
  if i = j then 1/2 else 1/2

/-- The probability of person B hitting targets -/
def probB : HitProbability := fun i j => 
  if i = 0 ∧ j = 0 then 1/3
  else if i = 1 ∧ j = 1 then 2/5
  else 0

/-- The probability of a target being destroyed -/
def targetDestroyed (i : Fin 2) : ℚ :=
  probA i i * probB i i

/-- The probability of exactly one target being destroyed -/
def oneTargetDestroyed : ℚ :=
  (targetDestroyed 0) * (1 - probA 1 1) * (1 - probB 1 1) +
  (targetDestroyed 1) * (1 - probA 0 0) * (1 - probB 0 0)

theorem target_destruction_probabilities :
  (targetDestroyed 0 = 1/6) ∧
  (oneTargetDestroyed = 3/10) := by sorry

end target_destruction_probabilities_l2347_234787


namespace exists_valid_configuration_l2347_234792

/-- Represents a lamp in a room -/
structure Lamp where
  room : Nat
  state : Bool

/-- Represents a switch controlling a pair of lamps -/
structure Switch where
  lamp1 : Lamp
  lamp2 : Lamp

/-- Configuration of lamps and switches -/
structure Configuration (k : Nat) where
  lamps : Fin (6 * k) → Lamp
  switches : Fin (3 * k) → Switch
  rooms : Fin (2 * k)

/-- Predicate to check if a room has at least one lamp on and one off -/
def validRoom (config : Configuration k) (room : Fin (2 * k)) : Prop :=
  ∃ (l1 l2 : Fin (6 * k)), 
    (config.lamps l1).room = room ∧ 
    (config.lamps l2).room = room ∧ 
    (config.lamps l1).state = true ∧ 
    (config.lamps l2).state = false

/-- Main theorem statement -/
theorem exists_valid_configuration (k : Nat) (h : k > 0) : 
  ∃ (config : Configuration k), ∀ (room : Fin (2 * k)), validRoom config room :=
sorry

end exists_valid_configuration_l2347_234792


namespace total_seashells_l2347_234707

-- Define the variables
def seashells_given_to_tom : ℕ := 49
def seashells_left_with_mike : ℕ := 13

-- Define the theorem
theorem total_seashells :
  seashells_given_to_tom + seashells_left_with_mike = 62 := by
  sorry

end total_seashells_l2347_234707


namespace expression_equals_negative_one_l2347_234790

theorem expression_equals_negative_one (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : b ≠ a) (hd : b ≠ -a) :
  (a / (a + b) + b / (a - b)) / (b / (a + b) - a / (a - b)) = -1 := by
  sorry

end expression_equals_negative_one_l2347_234790


namespace determinant_transformation_l2347_234761

theorem determinant_transformation (x y z w : ℝ) :
  (x * w - y * z = 3) →
  (x * (5 * z + 4 * w) - z * (5 * x + 4 * y) = 12) :=
by sorry

end determinant_transformation_l2347_234761


namespace tree_height_after_four_years_l2347_234709

/-- The height of a tree after n years, given its initial height and growth rate -/
def treeHeight (initialHeight : ℝ) (growthRate : ℝ) (n : ℕ) : ℝ :=
  initialHeight * growthRate^(n - 1)

/-- Theorem stating the height of the tree after 4 years -/
theorem tree_height_after_four_years
  (h1 : treeHeight 2 2 7 = 64)
  (h2 : treeHeight 2 2 1 = 2) :
  treeHeight 2 2 4 = 8 := by
  sorry

end tree_height_after_four_years_l2347_234709


namespace tangent_line_at_origin_l2347_234786

-- Define the curve f(x) = 2x³ - 3x
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

-- Theorem statement
theorem tangent_line_at_origin :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y = m * x) →                   -- Equation of a line through (0,0)
    (∃ (t : ℝ), t ≠ 0 →
      y = f t ∧                     -- Point (t, f(t)) is on the curve
      (f t - 0) / (t - 0) = m) →    -- Slope of secant line
    m = -3                          -- Slope of the tangent line
    :=
sorry

end tangent_line_at_origin_l2347_234786


namespace remaining_denomination_is_500_l2347_234788

/-- Represents the denomination problem with given conditions -/
def DenominationProblem (total_amount : ℕ) (total_notes : ℕ) (fifty_notes : ℕ) (fifty_value : ℕ) : Prop :=
  ∃ (other_denom : ℕ),
    total_amount = fifty_notes * fifty_value + (total_notes - fifty_notes) * other_denom ∧
    total_notes > fifty_notes ∧
    other_denom > 0

/-- Theorem stating that the denomination of remaining notes is 500 -/
theorem remaining_denomination_is_500 :
  DenominationProblem 10350 126 117 50 → ∃ (other_denom : ℕ), other_denom = 500 :=
by sorry

end remaining_denomination_is_500_l2347_234788


namespace square_window_side_length_l2347_234741

/-- Given a square window opening formed by two rectangular frames, 
    prove that the side length of the square is 5 when the perimeter 
    of the left frame is 14 and the perimeter of the right frame is 16. -/
theorem square_window_side_length 
  (a : ℝ) -- side length of the square window
  (b : ℝ) -- width of the left rectangular frame
  (h1 : 2 * a + 2 * b = 14) -- perimeter of the left frame
  (h2 : 4 * a - 2 * b = 16) -- perimeter of the right frame
  : a = 5 := by
  sorry

end square_window_side_length_l2347_234741


namespace smallest_n_satisfying_conditions_l2347_234720

theorem smallest_n_satisfying_conditions : 
  ∃ n : ℕ, n > 20 ∧ n % 6 = 5 ∧ n % 7 = 3 ∧ 
  ∀ m : ℕ, m > 20 ∧ m % 6 = 5 ∧ m % 7 = 3 → n ≤ m :=
by
  use 59
  sorry

end smallest_n_satisfying_conditions_l2347_234720


namespace additive_implies_zero_and_odd_l2347_234762

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y is zero at 0 and odd -/
theorem additive_implies_zero_and_odd (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end additive_implies_zero_and_odd_l2347_234762


namespace sqrt_nine_factorial_over_126_l2347_234719

theorem sqrt_nine_factorial_over_126 :
  let nine_factorial : ℕ := 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  let one_twenty_six : ℕ := 2 * 7 * 9
  (nine_factorial / one_twenty_six : ℚ).sqrt = 12 * Real.sqrt 10 := by
  sorry

end sqrt_nine_factorial_over_126_l2347_234719


namespace electricity_bill_theorem_l2347_234753

/-- Represents a meter reading with three tariff zones -/
structure MeterReading where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Represents tariff rates for electricity -/
structure TariffRates where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Calculates the electricity bill based on meter readings and tariff rates -/
def calculate_bill (previous : MeterReading) (current : MeterReading) (rates : TariffRates) : ℝ :=
  (current.peak - previous.peak) * rates.peak +
  (current.night - previous.night) * rates.night +
  (current.half_peak - previous.half_peak) * rates.half_peak

/-- Theorem: Maximum additional payment and expected difference -/
theorem electricity_bill_theorem 
  (previous : MeterReading)
  (current : MeterReading)
  (rates : TariffRates)
  (actual_payment : ℝ)
  (h1 : rates.peak = 4.03)
  (h2 : rates.night = 1.01)
  (h3 : rates.half_peak = 3.39)
  (h4 : actual_payment = 660.72)
  (h5 : current.peak > previous.peak)
  (h6 : current.night > previous.night)
  (h7 : current.half_peak > previous.half_peak) :
  ∃ (max_additional_payment expected_difference : ℝ),
    max_additional_payment = 397.34 ∧
    expected_difference = 19.30 :=
sorry

end electricity_bill_theorem_l2347_234753


namespace problem_1_problem_2_l2347_234740

-- Problem 1
theorem problem_1 (a b : ℝ) : 
  (abs a = 5) → 
  (abs b = 3) → 
  (abs (a - b) = b - a) → 
  ((a - b = -8) ∨ (a - b = -2)) :=
sorry

-- Problem 2
theorem problem_2 (a b c d m : ℝ) :
  (a + b = 0) →
  (c * d = 1) →
  (abs m = 2) →
  (abs (a + b) / m - c * d + m^2 = 3) :=
sorry

end problem_1_problem_2_l2347_234740


namespace max_area_rectangle_l2347_234766

/-- The maximum area of a rectangle with a perimeter of 40 inches is 100 square inches. -/
theorem max_area_rectangle (x y : ℝ) (h_perimeter : x + y = 20) :
  x * y ≤ 100 ∧ ∃ (a b : ℝ), a + b = 20 ∧ a * b = 100 :=
by sorry

end max_area_rectangle_l2347_234766


namespace larger_integer_proof_l2347_234768

theorem larger_integer_proof (x y : ℕ+) 
  (h1 : y - x = 8)
  (h2 : x * y = 272) : 
  y = 20 := by
  sorry

end larger_integer_proof_l2347_234768


namespace friend_spent_ten_l2347_234779

def lunch_problem (total : ℝ) (difference : ℝ) : Prop :=
  ∃ (your_cost friend_cost : ℝ),
    your_cost + friend_cost = total ∧
    friend_cost = your_cost + difference ∧
    friend_cost = 10

theorem friend_spent_ten :
  lunch_problem 17 3 :=
sorry

end friend_spent_ten_l2347_234779


namespace square_of_difference_formula_l2347_234767

theorem square_of_difference_formula (m n : ℝ) : 
  ¬ ∃ (a b : ℝ), (m - n) * (-m + n) = (a - b) * (a + b) := by
  sorry

end square_of_difference_formula_l2347_234767


namespace isosceles_triangle_area_l2347_234742

/-- Represents an isosceles triangle with vertex angle 80°, leg length a, and base length b -/
structure IsoscelesTriangle where
  a : ℝ  -- length of the legs
  b : ℝ  -- length of the base
  h₁ : a > 0
  h₂ : b > 0

/-- Calculates the area of an isosceles triangle -/
noncomputable def triangleArea (t : IsoscelesTriangle) : ℝ :=
  (t.a^3 * t.b) / (4 * (t.b^2 - t.a^2))

/-- Theorem stating that the area of the isosceles triangle with vertex angle 80° is (a^3 * b) / (4 * (b^2 - a^2)) -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) :
  triangleArea t = (t.a^3 * t.b) / (4 * (t.b^2 - t.a^2)) := by
  sorry

end isosceles_triangle_area_l2347_234742


namespace inequality_solution_l2347_234745

def solution_set (a : ℝ) : Set ℝ :=
  if a > 1 then {x | x ≤ 1 ∨ x ≥ a}
  else if a = 1 then Set.univ
  else {x | x ≤ a ∨ x ≥ 1}

theorem inequality_solution (a : ℝ) :
  {x : ℝ | x^2 - (a + 1)*x + a ≥ 0} = solution_set a := by sorry

end inequality_solution_l2347_234745


namespace quadratic_intersection_l2347_234738

/-- A quadratic function f(x) = x^2 - 6x + c intersects the x-axis at only one point
    if and only if its discriminant is zero. -/
def intersects_once (c : ℝ) : Prop :=
  ((-6)^2 - 4*1*c) = 0

/-- The theorem states that if a quadratic function f(x) = x^2 - 6x + c
    intersects the x-axis at only one point, then c = 9. -/
theorem quadratic_intersection (c : ℝ) :
  intersects_once c → c = 9 := by
  sorry

end quadratic_intersection_l2347_234738


namespace unique_solution_pqr_l2347_234735

theorem unique_solution_pqr : 
  ∀ p q r : ℕ,
  Prime p → Prime q → Even r → r > 0 →
  p^3 + q^2 = 4*r^2 + 45*r + 103 →
  p = 7 ∧ q = 2 ∧ r = 4 :=
by sorry

end unique_solution_pqr_l2347_234735


namespace our_number_not_perfect_square_l2347_234793

-- Define a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Define the number we want to prove is not a perfect square
def our_number : ℕ := 4^2021

-- Theorem statement
theorem our_number_not_perfect_square : ¬ (is_perfect_square our_number) := by
  sorry

end our_number_not_perfect_square_l2347_234793


namespace units_digit_sum_base8_l2347_234718

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a base-8 number --/
def units_digit_base8 (n : ℕ) : ℕ := sorry

theorem units_digit_sum_base8 : 
  units_digit_base8 (base10_to_base8 (base8_to_base10 53 + base8_to_base10 64)) = 7 := by sorry

end units_digit_sum_base8_l2347_234718


namespace P_less_than_Q_l2347_234714

theorem P_less_than_Q (a : ℝ) (h : a ≥ 0) : 
  Real.sqrt a + Real.sqrt (a + 7) < Real.sqrt (a + 3) + Real.sqrt (a + 4) := by
sorry

end P_less_than_Q_l2347_234714


namespace impossibleConstruction_l2347_234712

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a function to check if a triangle is acute-angled
def isAcuteAngled (t : Triangle3D) : Prop := sorry

-- Define a function to check if a triangle is equilateral
def isEquilateral (t : Triangle3D) : Prop := sorry

-- Define a function to check if three lines intersect at a point
def linesIntersectAtPoint (A A' B B' C C' O : Point3D) : Prop := sorry

-- Main theorem
theorem impossibleConstruction (ABC : Triangle3D) (O : Point3D) :
  isAcuteAngled ABC →
  ¬∃ (A'B'C' : Triangle3D),
    isEquilateral A'B'C' ∧
    linesIntersectAtPoint ABC.A A'B'C'.A ABC.B A'B'C'.B ABC.C A'B'C'.C O :=
by sorry

end impossibleConstruction_l2347_234712


namespace expression_evaluation_l2347_234783

theorem expression_evaluation : (10^9) / ((2 * 10^6) * 3) = 500/3 := by
  sorry

end expression_evaluation_l2347_234783


namespace polynomial_identity_sum_l2347_234798

theorem polynomial_identity_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -1 := by
sorry

end polynomial_identity_sum_l2347_234798


namespace paint_mixture_ratio_l2347_234713

/-- Given a paint mixture with a ratio of 5:3:7 for red:yellow:white paint,
    if 21 quarts of white paint is used, then 9 quarts of yellow paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) :
  red / yellow = 5 / 3 →
  yellow / white = 3 / 7 →
  white = 21 →
  yellow = 9 := by
  sorry

end paint_mixture_ratio_l2347_234713


namespace garrison_provisions_duration_l2347_234758

/-- The number of days provisions last for a garrison given reinforcements --/
theorem garrison_provisions_duration 
  (initial_men : ℕ) 
  (reinforcement_men : ℕ) 
  (days_before_reinforcement : ℕ) 
  (days_after_reinforcement : ℕ) 
  (h1 : initial_men = 2000)
  (h2 : reinforcement_men = 1900)
  (h3 : days_before_reinforcement = 15)
  (h4 : days_after_reinforcement = 20) :
  ∃ (initial_days : ℕ), 
    initial_days * initial_men = 
      (initial_men + reinforcement_men) * days_after_reinforcement + 
      initial_men * days_before_reinforcement ∧
    initial_days = 54 := by
  sorry

end garrison_provisions_duration_l2347_234758


namespace john_study_time_for_average_75_l2347_234781

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  k : ℝ  -- Proportionality constant
  study_time : ℝ → ℝ  -- Function mapping score to study time
  score : ℝ → ℝ  -- Function mapping study time to score

/-- John's hypothesis about study time and test score -/
def john_hypothesis (r : StudyScoreRelation) : Prop :=
  ∀ t, r.score t = r.k * t

theorem john_study_time_for_average_75 
  (r : StudyScoreRelation)
  (h1 : john_hypothesis r)
  (h2 : r.score 3 = 60)  -- First exam result
  (h3 : r.k = 20)  -- Derived from first exam
  : r.study_time 90 = 4.5 := by
  sorry

end john_study_time_for_average_75_l2347_234781


namespace train_length_l2347_234777

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 12 → ∃ length : ℝ, 
  (abs (length - 200.04) < 0.01) ∧ (length = speed * (1000 / 3600) * time) := by
  sorry

end train_length_l2347_234777


namespace sum_cubes_quartics_bounds_l2347_234764

theorem sum_cubes_quartics_bounds (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) :
  let expr := 3 * (p^3 + q^3 + r^3 + s^3) - 5 * (p^4 + q^4 + r^4 + s^4)
  ∃ (min max : ℝ), min = 132 ∧ max = -20 ∧ min ≤ expr ∧ expr ≤ max := by
  sorry

end sum_cubes_quartics_bounds_l2347_234764


namespace xy_yz_xz_equals_60_l2347_234751

theorem xy_yz_xz_equals_60 
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 36)
  (eq3 : z^2 + x*z + x^2 = 111) :
  x*y + y*z + x*z = 60 := by
sorry

end xy_yz_xz_equals_60_l2347_234751


namespace alyssa_pears_l2347_234765

theorem alyssa_pears (total_pears nancy_pears : ℕ) 
  (h1 : total_pears = 59) 
  (h2 : nancy_pears = 17) : 
  total_pears - nancy_pears = 42 := by
  sorry

end alyssa_pears_l2347_234765


namespace immediate_boarding_probability_l2347_234784

/-- Represents the cycle time of a subway train in minutes -/
def cycletime : ℝ := 10

/-- Represents the time the train stops at the station in minutes -/
def stoptime : ℝ := 1

/-- Theorem: The probability of a passenger arriving at the platform 
    and immediately boarding the train is 1/10 -/
theorem immediate_boarding_probability : 
  stoptime / cycletime = 1 / 10 := by sorry

end immediate_boarding_probability_l2347_234784


namespace same_color_probability_l2347_234754

theorem same_color_probability (N : ℕ) : 
  (4 : ℚ) / 10 * 16 / (16 + N) + (6 : ℚ) / 10 * N / (16 + N) = 29 / 50 → N = 144 := by
  sorry

end same_color_probability_l2347_234754


namespace number_of_women_at_tables_l2347_234797

/-- Proves that the number of women at the tables is 7.0 -/
theorem number_of_women_at_tables 
  (num_tables : Float) 
  (num_men : Float) 
  (avg_customers_per_table : Float) 
  (h1 : num_tables = 9.0)
  (h2 : num_men = 3.0)
  (h3 : avg_customers_per_table = 1.111111111) : 
  Float.round ((num_tables * avg_customers_per_table) - num_men) = 7.0 := by
  sorry

end number_of_women_at_tables_l2347_234797


namespace exists_number_not_divisible_by_3_with_digit_product_3_l2347_234794

def numbers : List Nat := [4621, 4631, 4641, 4651, 4661]

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def is_divisible_by_3 (n : Nat) : Prop :=
  n % 3 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem exists_number_not_divisible_by_3_with_digit_product_3 :
  ∃ n ∈ numbers, ¬(is_divisible_by_3 n) ∧ (units_digit n) * (tens_digit n) = 3 := by
  sorry

end exists_number_not_divisible_by_3_with_digit_product_3_l2347_234794


namespace intersection_with_complement_l2347_234749

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {0, 3, 6, 9, 12}

theorem intersection_with_complement :
  A ∩ (Set.univ \ B) = {1, 5, 7} := by sorry

end intersection_with_complement_l2347_234749


namespace min_value_expression_l2347_234796

theorem min_value_expression (x y : ℝ) : (x*y + 1)^2 + (x^2 + y^2)^2 ≥ 1 := by
  sorry

end min_value_expression_l2347_234796


namespace beong_gun_number_l2347_234736

theorem beong_gun_number : ∃ x : ℚ, (x / 11 + 156 = 178) ∧ (x = 242) := by
  sorry

end beong_gun_number_l2347_234736


namespace mask_probability_l2347_234774

theorem mask_probability (regular_ratio surgical_ratio regular_ear_loop_ratio surgical_ear_loop_ratio : Real) 
  (h1 : regular_ratio = 0.8)
  (h2 : surgical_ratio = 0.2)
  (h3 : regular_ear_loop_ratio = 0.1)
  (h4 : surgical_ear_loop_ratio = 0.2)
  (h5 : regular_ratio + surgical_ratio = 1) :
  regular_ratio * regular_ear_loop_ratio + surgical_ratio * surgical_ear_loop_ratio = 0.12 := by
sorry

end mask_probability_l2347_234774


namespace ap_square_identity_l2347_234789

/-- Three consecutive terms of an arithmetic progression -/
structure ArithmeticProgressionTerms (α : Type*) [Add α] [Sub α] where
  a : α
  b : α
  c : α
  is_ap : b - a = c - b

/-- Theorem: For any three consecutive terms of an arithmetic progression,
    a^2 + 8bc = (2b + c)^2 -/
theorem ap_square_identity {α : Type*} [CommRing α] (terms : ArithmeticProgressionTerms α) :
  terms.a ^ 2 + 8 * terms.b * terms.c = (2 * terms.b + terms.c) ^ 2 := by
  sorry

end ap_square_identity_l2347_234789


namespace constant_product_l2347_234728

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define curve C
def curve_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

-- Define points A, P, and B on curve C
def point_on_C (p : ℝ × ℝ) : Prop := curve_C p.1 p.2

theorem constant_product (A P B S T : ℝ × ℝ) 
  (hA : point_on_C A) (hP : point_on_C P) (hB : point_on_C B)
  (hB_sym : B.1 = A.1 ∧ B.2 = -A.2)
  (hS : S.2 = 0 ∧ (P.2 - A.2) * (S.1 - A.1) = (P.1 - A.1) * (S.2 - A.2))
  (hT : T.2 = 0 ∧ (P.2 - B.2) * (T.1 - B.1) = (P.1 - B.1) * (T.2 - B.2))
  (hP_ne_A : P.1 ≠ A.1 ∨ P.2 ≠ A.2) :
  |S.1| * |T.1| = 9 :=
sorry

end constant_product_l2347_234728


namespace range_of_a_l2347_234708

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x a : ℝ) : Prop := x > a

-- Define what it means for q to be sufficient but not necessary for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
sorry

end range_of_a_l2347_234708


namespace min_value_expression_l2347_234743

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 = 0 := by
sorry

end min_value_expression_l2347_234743


namespace det_dilation_matrix_3d_det_dilation_matrix_3d_scale_5_l2347_234747

def dilation_matrix (scale : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => scale)

theorem det_dilation_matrix_3d (scale : ℝ) :
  Matrix.det (dilation_matrix scale) = scale ^ 3 := by
  sorry

theorem det_dilation_matrix_3d_scale_5 :
  Matrix.det (dilation_matrix 5) = 125 := by
  sorry

end det_dilation_matrix_3d_det_dilation_matrix_3d_scale_5_l2347_234747


namespace square_root_of_four_l2347_234722

theorem square_root_of_four : ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_four_l2347_234722


namespace tissue_diameter_calculation_l2347_234700

/-- Given a circular piece of tissue magnified by an electron microscope,
    calculate its actual diameter in millimeters. -/
theorem tissue_diameter_calculation
  (magnification : ℕ)
  (magnified_diameter_meters : ℝ)
  (h_magnification : magnification = 5000)
  (h_magnified_diameter : magnified_diameter_meters = 0.15) :
  magnified_diameter_meters * 1000 / magnification = 0.03 := by
  sorry

end tissue_diameter_calculation_l2347_234700


namespace divisibility_by_eleven_l2347_234725

theorem divisibility_by_eleven (n : ℤ) : 
  (11 : ℤ) ∣ ((n + 11)^2 - n^2) := by
  sorry

end divisibility_by_eleven_l2347_234725


namespace sqrt_2_3_5_not_arithmetic_progression_l2347_234711

theorem sqrt_2_3_5_not_arithmetic_progression : ¬ ∃ (d : ℝ), Real.sqrt 3 = Real.sqrt 2 + d ∧ Real.sqrt 5 = Real.sqrt 2 + 2 * d := by
  sorry

end sqrt_2_3_5_not_arithmetic_progression_l2347_234711


namespace min_distance_C1_to_C2_sum_distances_PA_PB_l2347_234733

-- Define the circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line C2
def C2 (x y : ℝ) : Prop := y = x + 2

-- Define the ellipse C1'
def C1' (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define point P
def P : ℝ × ℝ := (-1, 1)

-- Theorem for the minimum distance
theorem min_distance_C1_to_C2 :
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧
  (∀ (x y : ℝ), C1 x y → ∀ (x' y' : ℝ), C2 x' y' →
    Real.sqrt ((x - x')^2 + (y - y')^2) ≥ d) ∧
  (∃ (x y : ℝ), C1 x y ∧ ∃ (x' y' : ℝ), C2 x' y' ∧
    Real.sqrt ((x - x')^2 + (y - y')^2) = d) :=
sorry

-- Theorem for the sum of distances
theorem sum_distances_PA_PB :
  ∃ (A B : ℝ × ℝ), C1' A.1 A.2 ∧ C1' B.1 B.2 ∧
  C2 A.1 A.2 ∧ C2 B.1 B.2 ∧
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 12 * Real.sqrt 2 / 7 :=
sorry

end min_distance_C1_to_C2_sum_distances_PA_PB_l2347_234733


namespace right_triangle_inequality_l2347_234703

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 = b^2 + c^2) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 5 + 3 * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l2347_234703


namespace equation_solution_l2347_234731

theorem equation_solution :
  {x : ℂ | x^4 - 81 = 0} = {3, -3, 3*I, -3*I} := by sorry

end equation_solution_l2347_234731


namespace min_tan_sum_l2347_234702

theorem min_tan_sum (α β : Real) (h1 : 0 < α ∧ α < π) (h2 : 0 < β ∧ β < π) 
  (h3 : α + β < π) 
  (h4 : (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = Real.cos (2 * β)) :
  ∃ (m : Real), ∀ (α' β' : Real), 
    (0 < α' ∧ α' < π) → (0 < β' ∧ β' < π) → (α' + β' < π) → 
    ((Real.cos α' - Real.sin α') / (Real.cos α' + Real.sin α') = Real.cos (2 * β')) →
    (Real.tan α' + Real.tan β' ≥ m) ∧ 
    (∃ (α₀ β₀ : Real), Real.tan α₀ + Real.tan β₀ = m) ∧ 
    m = -1/4 := by
  sorry

end min_tan_sum_l2347_234702


namespace largest_sales_increase_2011_l2347_234771

-- Define the sales data for each year
def sales : Fin 8 → ℕ
  | 0 => 20
  | 1 => 24
  | 2 => 27
  | 3 => 26
  | 4 => 28
  | 5 => 33
  | 6 => 32
  | 7 => 35

-- Define the function to calculate the sales increase between two consecutive years
def salesIncrease (i : Fin 7) : ℤ :=
  (sales (i.succ : Fin 8) : ℤ) - (sales i : ℤ)

-- Define the theorem to prove
theorem largest_sales_increase_2011 :
  ∃ i : Fin 7, salesIncrease i = 5 ∧
  ∀ j : Fin 7, salesIncrease j ≤ 5 ∧
  (i : ℕ) + 2006 = 2011 :=
by sorry

end largest_sales_increase_2011_l2347_234771


namespace geometric_progression_properties_l2347_234780

/-- A geometric progression with given second and fifth terms -/
structure GeometricProgression where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- The third term of the geometric progression -/
def thirdTerm (gp : GeometricProgression) : ℝ := 49

/-- The sum of the first four terms of the geometric progression -/
def sumFirstFour (gp : GeometricProgression) : ℝ := 183.75

theorem geometric_progression_properties (gp : GeometricProgression) :
  thirdTerm gp = 49 ∧ sumFirstFour gp = 183.75 := by
  sorry

end geometric_progression_properties_l2347_234780


namespace coin_problem_l2347_234706

theorem coin_problem (p n : ℕ) : 
  p + n = 32 →  -- Total number of coins
  p + 5 * n = 100 →  -- Total value in cents
  n = 17 :=
by sorry

end coin_problem_l2347_234706


namespace range_of_m_l2347_234710

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def B : Set ℝ := {x | -1 < x ∧ x < 5}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem range_of_m (m : ℝ) :
  (Set.compl (A m) ∩ B).Nonempty → m < 1 :=
by sorry

end range_of_m_l2347_234710


namespace discount_calculation_l2347_234729

theorem discount_calculation (CP : ℝ) (CP_pos : CP > 0) : 
  let MP := 1.12 * CP
  let SP := 0.99 * CP
  MP - SP = 0.13 * CP := by sorry

end discount_calculation_l2347_234729


namespace star_calculation_l2347_234750

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ (4 ⋆ 6)) = -152877 -/
theorem star_calculation : star 2 (star 3 (star 4 6)) = -152877 := by sorry

end star_calculation_l2347_234750


namespace coefficient_of_x_in_expansion_l2347_234773

/-- The coefficient of x in the expansion of (1 + √x)^6 * (1 + √x)^4 -/
def coefficient_of_x : ℕ := 45

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_of_x_in_expansion :
  coefficient_of_x = 
    binomial 4 2 + binomial 6 2 + binomial 6 1 * binomial 4 1 :=
by sorry

end coefficient_of_x_in_expansion_l2347_234773


namespace triangle_area_sum_form_sum_of_coefficients_l2347_234701

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side : ℝ)
  (is_two : side = 2)

/-- Represents the sum of areas of all triangles with vertices on the cube -/
def triangle_area_sum (c : Cube) : ℝ := sorry

/-- The sum can be expressed as q + √r + √s where q, r, s are integers -/
theorem triangle_area_sum_form (c : Cube) :
  ∃ (q r s : ℤ), triangle_area_sum c = ↑q + Real.sqrt (↑r) + Real.sqrt (↑s) :=
sorry

/-- The sum of q, r, and s is 7728 -/
theorem sum_of_coefficients (c : Cube) :
  ∃ (q r s : ℤ),
    triangle_area_sum c = ↑q + Real.sqrt (↑r) + Real.sqrt (↑s) ∧
    q + r + s = 7728 :=
sorry

end triangle_area_sum_form_sum_of_coefficients_l2347_234701


namespace quadratic_always_nonnegative_implies_a_range_l2347_234778

theorem quadratic_always_nonnegative_implies_a_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (1 - a)*x + 1 ≥ 0) → a ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end quadratic_always_nonnegative_implies_a_range_l2347_234778


namespace fraction_expression_equality_l2347_234795

theorem fraction_expression_equality : 
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 9) = 1548 / 805 := by
  sorry

end fraction_expression_equality_l2347_234795


namespace parabola_intersection_distance_l2347_234748

theorem parabola_intersection_distance : 
  ∀ (p q r s : ℝ), 
  (∃ x y : ℝ, y = 3*x^2 - 6*x + 3 ∧ y = -x^2 - 3*x + 3 ∧ ((x = p ∧ y = q) ∨ (x = r ∧ y = s))) → 
  r ≥ p → 
  r - p = 3/4 := by
sorry

end parabola_intersection_distance_l2347_234748


namespace part_one_part_two_l2347_234721

-- Define the function f
def f (x a m : ℝ) : ℝ := |x - a| + m * |x + a|

-- Part I
theorem part_one : 
  {x : ℝ | f x (-1) (-1) ≥ x} = {x : ℝ | x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 2)} := by sorry

-- Part II
theorem part_two (m : ℝ) (h1 : 0 < m) (h2 : m < 1) 
  (h3 : ∀ x : ℝ, f x a m ≥ 2) 
  (h4 : a ≤ -3 ∨ a ≥ 3) : 
  m = 1/3 := by sorry

end part_one_part_two_l2347_234721


namespace officers_on_duty_l2347_234734

theorem officers_on_duty (total_female_officers : ℕ) 
  (female_on_duty_percentage : ℚ) (female_ratio_on_duty : ℚ) :
  total_female_officers = 300 →
  female_on_duty_percentage = 2/5 →
  female_ratio_on_duty = 1/2 →
  (female_on_duty_percentage * total_female_officers : ℚ) / female_ratio_on_duty = 240 := by
  sorry

end officers_on_duty_l2347_234734


namespace square_root_range_l2347_234791

theorem square_root_range (x : ℝ) : 3 - 2*x ≥ 0 → x ≤ 3/2 := by
  sorry

end square_root_range_l2347_234791


namespace f_extended_domain_l2347_234799

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_extended_domain (f : ℝ → ℝ) :
  is_even f →
  has_period f π →
  (∀ x ∈ Set.Icc 0 (π / 2), f x = 1 - Real.sin x) →
  ∀ x ∈ Set.Icc (5 * π / 2) (3 * π), f x = 1 - Real.sin x :=
by sorry

end f_extended_domain_l2347_234799


namespace marcy_serves_36_people_l2347_234716

/-- Represents the makeup supplies and application rates --/
structure MakeupSupplies where
  lip_gloss_per_tube : ℕ
  mascara_per_tube : ℕ
  lip_gloss_tubs : ℕ
  lip_gloss_tubes_per_tub : ℕ
  mascara_tubs : ℕ
  mascara_tubes_per_tub : ℕ

/-- Calculates the number of people that can be served with the given makeup supplies --/
def people_served (supplies : MakeupSupplies) : ℕ :=
  min
    (supplies.lip_gloss_tubs * supplies.lip_gloss_tubes_per_tub * supplies.lip_gloss_per_tube)
    (supplies.mascara_tubs * supplies.mascara_tubes_per_tub * supplies.mascara_per_tube)

/-- Theorem stating that Marcy can serve exactly 36 people with her makeup supplies --/
theorem marcy_serves_36_people :
  let supplies := MakeupSupplies.mk 3 5 6 2 4 3
  people_served supplies = 36 := by
  sorry

#eval people_served (MakeupSupplies.mk 3 5 6 2 4 3)

end marcy_serves_36_people_l2347_234716


namespace andrea_skating_schedule_l2347_234763

/-- Represents Andrea's skating schedule and target average -/
structure SkatingSchedule where
  days_schedule1 : ℕ  -- Number of days for schedule 1
  minutes_per_day1 : ℕ  -- Minutes skated per day in schedule 1
  days_schedule2 : ℕ  -- Number of days for schedule 2
  minutes_per_day2 : ℕ  -- Minutes skated per day in schedule 2
  total_days : ℕ  -- Total number of days
  target_average : ℕ  -- Target average minutes per day

/-- Calculates the required skating time for the last day to achieve the target average -/
def required_last_day_minutes (schedule : SkatingSchedule) : ℕ :=
  schedule.target_average * schedule.total_days -
  (schedule.days_schedule1 * schedule.minutes_per_day1 +
   schedule.days_schedule2 * schedule.minutes_per_day2)

/-- Theorem stating that given Andrea's skating schedule, 
    she needs to skate 175 minutes on the ninth day to achieve the target average -/
theorem andrea_skating_schedule :
  let schedule : SkatingSchedule := {
    days_schedule1 := 6,
    minutes_per_day1 := 80,
    days_schedule2 := 2,
    minutes_per_day2 := 100,
    total_days := 9,
    target_average := 95
  }
  required_last_day_minutes schedule = 175 := by
  sorry

end andrea_skating_schedule_l2347_234763


namespace marks_reading_time_l2347_234737

/-- Calculates Mark's new weekly reading time given his daily reading time,
    the number of days in a week, and his planned increase in weekly reading time. -/
def new_weekly_reading_time (daily_reading_time : ℕ) (days_in_week : ℕ) (weekly_increase : ℕ) : ℕ :=
  daily_reading_time * days_in_week + weekly_increase

/-- Proves that Mark's new weekly reading time is 18 hours -/
theorem marks_reading_time :
  new_weekly_reading_time 2 7 4 = 18 := by
  sorry

end marks_reading_time_l2347_234737


namespace chewbacca_gum_packs_l2347_234744

theorem chewbacca_gum_packs (y : ℚ) : 
  (∃ (orange_packs apple_packs : ℕ), 
    orange_packs * y + (25 : ℚ) % y = 25 ∧ 
    apple_packs * y + (35 : ℚ) % y = 35 ∧
    (25 - 2 * y) / 35 = 25 / (35 + 4 * y)) → 
  y = 15 / 4 := by
sorry

end chewbacca_gum_packs_l2347_234744


namespace range_of_m_l2347_234757

open Set Real

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m - 1}

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (A ∩ (U \ B m) = A) ↔ m ≤ 3 :=
sorry

end range_of_m_l2347_234757


namespace couplet_distribution_ways_l2347_234759

def num_widows : ℕ := 4
def num_long_couplets : ℕ := 4
def num_short_couplets : ℕ := 7

def long_couplets_per_widow : ℕ := 1
def short_couplets_for_one_widow : ℕ := 1
def short_couplets_for_three_widows : ℕ := 2

theorem couplet_distribution_ways :
  (Nat.choose num_long_couplets long_couplets_per_widow) *
  (Nat.choose num_short_couplets short_couplets_for_three_widows) *
  (Nat.choose (num_long_couplets - long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - short_couplets_for_three_widows) short_couplets_for_one_widow) *
  (Nat.choose (num_long_couplets - 2 * long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - short_couplets_for_three_widows - short_couplets_for_one_widow) short_couplets_for_three_widows) *
  (Nat.choose (num_long_couplets - 3 * long_couplets_per_widow) long_couplets_per_widow) *
  (Nat.choose (num_short_couplets - 2 * short_couplets_for_three_widows - short_couplets_for_one_widow) short_couplets_for_three_widows) = 15120 := by
  sorry

end couplet_distribution_ways_l2347_234759


namespace smallest_n_for_inequality_l2347_234730

def sequence_a (n : ℕ) : ℝ := 2^n

def sum_S (n : ℕ) : ℝ := 2 * sequence_a n - 2

def sequence_T (n : ℕ) : ℝ := n * 2^(n+1) - 2^(n+1) + 2

theorem smallest_n_for_inequality :
  ∀ n : ℕ, (∀ k < n, sequence_T k - k * 2^(k+1) + 50 ≥ 0) ∧
           (sequence_T n - n * 2^(n+1) + 50 < 0) →
  n = 5 := by sorry

end smallest_n_for_inequality_l2347_234730


namespace f_of_2_equals_12_l2347_234726

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem f_of_2_equals_12 : f 2 = 12 := by
  sorry

end f_of_2_equals_12_l2347_234726
