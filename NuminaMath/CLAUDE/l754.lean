import Mathlib

namespace NUMINAMATH_CALUDE_intermediate_circle_radius_l754_75415

theorem intermediate_circle_radius 
  (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 5)
  (h₂ : r₃ = 13)
  (h₃ : π * r₁^2 = π * r₃^2 - π * r₂^2) :
  r₂ = 12 := by
sorry

end NUMINAMATH_CALUDE_intermediate_circle_radius_l754_75415


namespace NUMINAMATH_CALUDE_fruit_prices_l754_75479

theorem fruit_prices (mango_price banana_price : ℝ) : 
  (3 * mango_price + 2 * banana_price = 40) →
  (2 * mango_price + 3 * banana_price = 35) →
  (mango_price = 10 ∧ banana_price = 5) := by
sorry

end NUMINAMATH_CALUDE_fruit_prices_l754_75479


namespace NUMINAMATH_CALUDE_all_equal_l754_75442

-- Define the sequence type
def Sequence := Fin 2020 → ℕ

-- Define the divisibility condition for six consecutive numbers
def DivisibleSix (a : Sequence) : Prop :=
  ∀ n : Fin 2015, a n ∣ a (n + 5)

-- Define the divisibility condition for nine consecutive numbers
def DivisibleNine (a : Sequence) : Prop :=
  ∀ n : Fin 2012, a (n + 8) ∣ a n

-- State the theorem
theorem all_equal (a : Sequence) (h1 : DivisibleSix a) (h2 : DivisibleNine a) :
  ∀ i j : Fin 2020, a i = a j :=
sorry

end NUMINAMATH_CALUDE_all_equal_l754_75442


namespace NUMINAMATH_CALUDE_range_of_m_l754_75456

/-- An odd function f: ℝ → ℝ with domain [-2,2] -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  (∀ x, f x ≠ 0 → x ∈ Set.Icc (-2) 2)

/-- f is monotonically decreasing on [0,2] -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x < y → f y < f x

/-- The main theorem -/
theorem range_of_m (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : MonoDecreasing f) :
  {m : ℝ | f (1 + m) + f m < 0} = Set.Ioo (-1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l754_75456


namespace NUMINAMATH_CALUDE_hannah_savings_l754_75457

theorem hannah_savings (first_week : ℝ) : first_week = 4 := by
  have total_goal : ℝ := 80
  have fifth_week : ℝ := 20
  have savings_sum : first_week + 2 * first_week + 4 * first_week + 8 * first_week + fifth_week = total_goal := by sorry
  sorry

end NUMINAMATH_CALUDE_hannah_savings_l754_75457


namespace NUMINAMATH_CALUDE_unseen_area_30_40_l754_75485

/-- Represents a rectangular room with guards in opposite corners. -/
structure GuardedRoom where
  length : ℝ
  width : ℝ
  guard1_pos : ℝ × ℝ
  guard2_pos : ℝ × ℝ

/-- Calculates the area of the room that neither guard can see. -/
def unseen_area (room : GuardedRoom) : ℝ :=
  sorry

/-- Theorem stating that for a room of 30m x 40m with guards in opposite corners,
    the unseen area is 225 m². -/
theorem unseen_area_30_40 :
  let room : GuardedRoom := {
    length := 30,
    width := 40,
    guard1_pos := (0, 0),
    guard2_pos := (30, 40)
  }
  unseen_area room = 225 := by sorry

end NUMINAMATH_CALUDE_unseen_area_30_40_l754_75485


namespace NUMINAMATH_CALUDE_household_expenses_equal_savings_l754_75425

/-- The number of years it takes to buy a house with all earnings -/
def years_to_buy : ℕ := 4

/-- The total number of years to buy the house -/
def total_years : ℕ := 24

/-- The number of years spent saving -/
def years_saving : ℕ := 12

/-- The number of years spent on household expenses -/
def years_household : ℕ := total_years - years_saving

theorem household_expenses_equal_savings : years_household = years_saving := by
  sorry

end NUMINAMATH_CALUDE_household_expenses_equal_savings_l754_75425


namespace NUMINAMATH_CALUDE_box_office_scientific_notation_l754_75435

/-- Converts a number in billions to scientific notation -/
def billionsToScientificNotation (x : ℝ) : ℝ × ℤ :=
  let mantissa := x * 10^(9 % 3)
  let exponent := 9 - (9 % 3)
  (mantissa, exponent)

/-- The box office revenue in billions of yuan -/
def boxOfficeRevenue : ℝ := 53.96

theorem box_office_scientific_notation :
  billionsToScientificNotation boxOfficeRevenue = (5.396, 9) := by
  sorry

end NUMINAMATH_CALUDE_box_office_scientific_notation_l754_75435


namespace NUMINAMATH_CALUDE_salary_increase_after_two_years_l754_75496

-- Define the raise percentage
def raise_percentage : ℝ := 0.05

-- Define the number of six-month periods in two years
def periods : ℕ := 4

-- Theorem stating the salary increase after two years
theorem salary_increase_after_two_years :
  let final_multiplier := (1 + raise_percentage) ^ periods
  abs (final_multiplier - 1 - 0.2155) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_two_years_l754_75496


namespace NUMINAMATH_CALUDE_prime_sum_product_l754_75434

theorem prime_sum_product (p q : ℕ) : 
  Prime p → Prime q → p + q = 85 → p * q = 166 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l754_75434


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l754_75428

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem chocolate_milk_probability : 
  binomial_probability 7 5 (1/2) = 21/128 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l754_75428


namespace NUMINAMATH_CALUDE_movies_watched_l754_75403

theorem movies_watched (total_movies : ℕ) (movies_left : ℕ) (h1 : total_movies = 8) (h2 : movies_left = 4) :
  total_movies - movies_left = 4 := by
  sorry

end NUMINAMATH_CALUDE_movies_watched_l754_75403


namespace NUMINAMATH_CALUDE_vector_c_value_l754_75455

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_c_value :
  ∀ c : ℝ × ℝ,
  (4 • a) + (3 • b - 2 • a) + c = (0, 0) →
  c = (4, -6) :=
by sorry

end NUMINAMATH_CALUDE_vector_c_value_l754_75455


namespace NUMINAMATH_CALUDE_mikes_remaining_nickels_l754_75429

/-- Represents the number of nickels Mike has after his dad's borrowing. -/
def mikesRemainingNickels (initialNickels : ℕ) (borrowedNickels : ℕ) : ℕ :=
  initialNickels - borrowedNickels

/-- Represents the total number of nickels borrowed by Mike's dad. -/
def totalBorrowedNickels (mikesBorrowed : ℕ) (sistersBorrowed : ℕ) : ℕ :=
  mikesBorrowed + sistersBorrowed

/-- Represents the relationship between nickels borrowed from Mike and his sister. -/
def borrowingPattern (mikesBorrowed : ℕ) (sistersBorrowed : ℕ) : Prop :=
  3 * sistersBorrowed = 2 * mikesBorrowed

theorem mikes_remaining_nickels :
  ∀ (mikesInitialNickels : ℕ) (mikesBorrowed : ℕ) (sistersBorrowed : ℕ),
    mikesInitialNickels = 87 →
    totalBorrowedNickels mikesBorrowed sistersBorrowed = 75 →
    borrowingPattern mikesBorrowed sistersBorrowed →
    mikesRemainingNickels mikesInitialNickels mikesBorrowed = 42 :=
by sorry

end NUMINAMATH_CALUDE_mikes_remaining_nickels_l754_75429


namespace NUMINAMATH_CALUDE_smallest_cube_sum_solution_l754_75474

/-- The smallest positive integer solution for the equation u³ + v³ + w³ = x³ -/
theorem smallest_cube_sum_solution :
  let P : ℕ → ℕ → ℕ → ℕ → Prop :=
    fun u v w x => u^3 + v^3 + w^3 = x^3 ∧ 
                   u < v ∧ v < w ∧ w < x ∧
                   v = u + 1 ∧ w = v + 1 ∧ x = w + 1
  ∀ u v w x, P u v w x → x ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_sum_solution_l754_75474


namespace NUMINAMATH_CALUDE_decimal_to_binary_nineteen_l754_75482

theorem decimal_to_binary_nineteen : 
  (1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 19 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_nineteen_l754_75482


namespace NUMINAMATH_CALUDE_entrance_fee_is_five_l754_75497

/-- The entrance fee per person for a concert, given the following conditions:
  * Tickets cost $50.00 each
  * There's a 15% processing fee for tickets
  * There's a $10.00 parking fee
  * The total cost for two people is $135.00
-/
def entrance_fee : ℝ := by
  sorry

theorem entrance_fee_is_five : entrance_fee = 5 := by
  sorry

end NUMINAMATH_CALUDE_entrance_fee_is_five_l754_75497


namespace NUMINAMATH_CALUDE_labourer_savings_is_30_l754_75464

/-- Calculates the savings of a labourer after 10 months, given specific spending patterns -/
def labourerSavings (monthlyIncome : ℕ) (expenseFirst6Months : ℕ) (expenseLast4Months : ℕ) : ℤ :=
  let totalIncome : ℕ := monthlyIncome * 10
  let totalExpense : ℕ := expenseFirst6Months * 6 + expenseLast4Months * 4
  (totalIncome : ℤ) - (totalExpense : ℤ)

/-- Theorem stating that the labourer's savings after 10 months is 30 -/
theorem labourer_savings_is_30 :
  labourerSavings 75 80 60 = 30 := by
  sorry

#eval labourerSavings 75 80 60

end NUMINAMATH_CALUDE_labourer_savings_is_30_l754_75464


namespace NUMINAMATH_CALUDE_marks_ratio_l754_75400

def total_marks : ℕ := 170
def science_marks : ℕ := 17

def english_math_ratio : ℚ := 1 / 4

theorem marks_ratio : 
  ∃ (english_marks math_marks : ℕ),
    english_marks + math_marks + science_marks = total_marks ∧
    english_marks / math_marks = english_math_ratio ∧
    english_marks / science_marks = 31 / 17 :=
by sorry

end NUMINAMATH_CALUDE_marks_ratio_l754_75400


namespace NUMINAMATH_CALUDE_inequality_proof_l754_75410

theorem inequality_proof (a₁ a₂ a₃ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃) : 
  let b₁ := (1 + a₁ * a₂ / (a₁ - a₂)) * (1 + a₁ * a₃ / (a₁ - a₃))
  let b₂ := (1 + a₂ * a₁ / (a₂ - a₁)) * (1 + a₂ * a₃ / (a₂ - a₃))
  let b₃ := (1 + a₃ * a₁ / (a₃ - a₁)) * (1 + a₃ * a₂ / (a₃ - a₂))
  1 + |a₁ * b₁ + a₂ * b₂ + a₃ * b₃| ≤ (1 + |a₁|) * (1 + |a₂|) * (1 + |a₃|) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l754_75410


namespace NUMINAMATH_CALUDE_correct_algorithm_statements_l754_75477

-- Define the set of algorithm statements
def AlgorithmStatements : Set ℕ := {1, 2, 3}

-- Define the property of being a correct statement about algorithms
def IsCorrectStatement : ℕ → Prop :=
  fun n => match n with
    | 1 => False  -- Statement 1 is incorrect
    | 2 => True   -- Statement 2 is correct
    | 3 => True   -- Statement 3 is correct
    | _ => False  -- Other numbers are not valid statements

-- Theorem: The set of correct statements is {2, 3}
theorem correct_algorithm_statements :
  {n ∈ AlgorithmStatements | IsCorrectStatement n} = {2, 3} := by
  sorry


end NUMINAMATH_CALUDE_correct_algorithm_statements_l754_75477


namespace NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_4_l754_75413

theorem remainder_of_n_squared_plus_2n_plus_4 (n : ℤ) (k : ℤ) 
  (h : n = 75 * k - 1) : 
  (n^2 + 2*n + 4) % 75 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_4_l754_75413


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l754_75412

/-- Represents a standard die with opposite faces summing to 7 -/
structure StandardDie :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ i : Fin 6, faces i + faces (5 - i) = 7)

/-- Represents the 4x4x4 cube constructed from standard dice -/
def LargeCube := Fin 4 → Fin 4 → Fin 4 → StandardDie

/-- Calculates the sum of visible faces on the large cube -/
def visibleSum (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the smallest possible sum of visible faces is 144 -/
theorem smallest_visible_sum (cube : LargeCube) : 
  ∃ (min_cube : LargeCube), visibleSum min_cube = 144 ∧ ∀ (c : LargeCube), visibleSum c ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l754_75412


namespace NUMINAMATH_CALUDE_F_bounded_and_amplitude_l754_75402

def F (a x : ℝ) : ℝ := x * |x - 2*a| + 3

theorem F_bounded_and_amplitude (a : ℝ) (h : a ≤ 1/2) :
  ∃ (m M : ℝ), (∀ x ∈ Set.Icc 1 2, m ≤ F a x ∧ F a x ≤ M) ∧
  (M - m = 3 - 2*a) := by sorry

end NUMINAMATH_CALUDE_F_bounded_and_amplitude_l754_75402


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l754_75458

def tank_length : ℝ := 4
def tank_width : ℝ := 5
def tank_height : ℝ := 3
def insulation_cost_per_sqft : ℝ := 20

def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

def total_cost (sa : ℝ) (cost_per_sqft : ℝ) : ℝ := sa * cost_per_sqft

theorem tank_insulation_cost :
  total_cost (surface_area tank_length tank_width tank_height) insulation_cost_per_sqft = 1880 := by
  sorry

end NUMINAMATH_CALUDE_tank_insulation_cost_l754_75458


namespace NUMINAMATH_CALUDE_positive_sum_inequality_sqrt_difference_inequality_l754_75471

-- Problem 1
theorem positive_sum_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by sorry

-- Problem 2
theorem sqrt_difference_inequality (n : ℕ+) :
  Real.sqrt (n + 1) - Real.sqrt n > Real.sqrt (n + 2) - Real.sqrt (n + 1) := by sorry

end NUMINAMATH_CALUDE_positive_sum_inequality_sqrt_difference_inequality_l754_75471


namespace NUMINAMATH_CALUDE_school_trip_distances_l754_75417

/-- Represents the problem of finding the distances in the school trip scenario. -/
theorem school_trip_distances 
  (total_distance : ℝ) 
  (walking_speed : ℝ) 
  (bus_speed : ℝ) 
  (rest_time : ℝ) : 
  total_distance = 21 ∧ 
  walking_speed = 4 ∧ 
  bus_speed = 60 ∧ 
  rest_time = 1/6 →
  ∃ (distance_to_A : ℝ) (distance_walked : ℝ),
    distance_to_A = 19 ∧
    distance_walked = 2 ∧
    distance_to_A + distance_walked = total_distance ∧
    distance_to_A / bus_speed + total_distance / bus_speed = 
      rest_time + distance_walked / walking_speed :=
by sorry

end NUMINAMATH_CALUDE_school_trip_distances_l754_75417


namespace NUMINAMATH_CALUDE_intersection_chord_length_l754_75407

-- Define the circles in polar coordinates
def circle_O₁ (ρ θ : ℝ) : Prop := ρ = 2

def circle_O₂ (ρ θ : ℝ) : Prop := ρ^2 - 2*Real.sqrt 2*ρ*(Real.cos (θ - Real.pi/4)) = 2

-- Define the circles in rectangular coordinates
def rect_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

def rect_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Theorem statement
theorem intersection_chord_length :
  ∀ A B : ℝ × ℝ,
  (rect_O₁ A.1 A.2 ∧ rect_O₁ B.1 B.2) →
  (rect_O₂ A.1 A.2 ∧ rect_O₂ B.1 B.2) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l754_75407


namespace NUMINAMATH_CALUDE_sin_fourth_sum_eighths_pi_l754_75433

theorem sin_fourth_sum_eighths_pi : 
  Real.sin (π / 8) ^ 4 + Real.sin (3 * π / 8) ^ 4 + 
  Real.sin (5 * π / 8) ^ 4 + Real.sin (7 * π / 8) ^ 4 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_fourth_sum_eighths_pi_l754_75433


namespace NUMINAMATH_CALUDE_arctan_sum_l754_75491

theorem arctan_sum : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_l754_75491


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l754_75454

theorem loan_principal_calculation (principal : ℝ) : 
  principal * 0.05 * 5 = principal - 2250 → principal = 3000 := by
  sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l754_75454


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l754_75439

/-- The product of the coordinates of the midpoint of a line segment with endpoints (4,7) and (-8,9) is -16. -/
theorem midpoint_coordinate_product : 
  let a : ℝ × ℝ := (4, 7)
  let b : ℝ × ℝ := (-8, 9)
  let midpoint := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)
  (midpoint.1 * midpoint.2 : ℝ) = -16 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l754_75439


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l754_75483

theorem square_minus_product_equals_one : 1999^2 - 2000 * 1998 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l754_75483


namespace NUMINAMATH_CALUDE_evaluate_expression_l754_75420

theorem evaluate_expression (h : π / 2 < 2 ∧ 2 < π) :
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π + 2)) = Real.sin 2 ^ 2 - Real.cos 2 ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l754_75420


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l754_75424

theorem sin_40_tan_10_minus_sqrt_3 :
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l754_75424


namespace NUMINAMATH_CALUDE_martha_lasagna_meat_amount_l754_75450

-- Define the constants
def cheese_amount : Real := 1.5
def cheese_price_per_kg : Real := 6
def meat_price_per_kg : Real := 8
def total_cost : Real := 13

-- Define the theorem
theorem martha_lasagna_meat_amount :
  let cheese_cost := cheese_amount * cheese_price_per_kg
  let meat_cost := total_cost - cheese_cost
  let meat_amount_kg := meat_cost / meat_price_per_kg
  let meat_amount_g := meat_amount_kg * 1000
  meat_amount_g = 500 := by
  sorry

end NUMINAMATH_CALUDE_martha_lasagna_meat_amount_l754_75450


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l754_75465

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y - 5) = 9 → y = 86 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l754_75465


namespace NUMINAMATH_CALUDE_fraction_ordering_l754_75468

theorem fraction_ordering : (6 : ℚ) / 29 < 8 / 25 ∧ 8 / 25 < 10 / 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l754_75468


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l754_75498

theorem cos_thirty_degrees : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l754_75498


namespace NUMINAMATH_CALUDE_trig_expression_equals_four_l754_75486

theorem trig_expression_equals_four : 
  (Real.sqrt 3 / Real.sin (20 * π / 180)) - (1 / Real.cos (20 * π / 180)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_four_l754_75486


namespace NUMINAMATH_CALUDE_walmart_cards_requested_l754_75452

def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def best_buy_cards_requested : ℕ := 6
def best_buy_cards_sent : ℕ := 1
def walmart_cards_sent : ℕ := 2
def remaining_gift_card_value : ℕ := 3900

def total_best_buy_value : ℕ := best_buy_card_value * best_buy_cards_requested
def sent_gift_card_value : ℕ := best_buy_card_value * best_buy_cards_sent + walmart_card_value * walmart_cards_sent

theorem walmart_cards_requested (walmart_cards : ℕ) : 
  walmart_cards * walmart_card_value + total_best_buy_value = 
  remaining_gift_card_value + sent_gift_card_value → walmart_cards = 9 := by
  sorry

end NUMINAMATH_CALUDE_walmart_cards_requested_l754_75452


namespace NUMINAMATH_CALUDE_antihomologous_properties_l754_75473

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Homothety center -/
def S : Point := sorry

/-- Given two circles satisfying the problem conditions -/
def circle1 : Circle := sorry
def circle2 : Circle := sorry

/-- Antihomologous points -/
def isAntihomologous (p q : Point) : Prop := sorry

/-- A point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop := sorry

/-- A circle is tangent to another circle -/
def isTangent (c1 c2 : Circle) : Prop := sorry

/-- Main theorem -/
theorem antihomologous_properties 
  (h1 : circle1.radius > circle2.radius)
  (h2 : isTangent circle1 circle2 ∨ 
        (circle1.center.1 - circle2.center.1)^2 + (circle1.center.2 - circle2.center.2)^2 
        > (circle1.radius + circle2.radius)^2) :
  (∀ (c : Circle) (p1 p2 p3 p4 : Point),
    isAntihomologous p1 p2 →
    onCircle p1 c ∧ onCircle p2 c →
    onCircle p3 circle1 ∧ onCircle p4 circle2 ∧ onCircle p3 c ∧ onCircle p4 c →
    isAntihomologous p3 p4) ∧
  (∀ (c : Circle),
    isTangent c circle1 ∧ isTangent c circle2 →
    ∃ (p1 p2 : Point),
      onCircle p1 circle1 ∧ onCircle p2 circle2 ∧
      onCircle p1 c ∧ onCircle p2 c ∧
      isAntihomologous p1 p2) :=
by sorry

end NUMINAMATH_CALUDE_antihomologous_properties_l754_75473


namespace NUMINAMATH_CALUDE_complex_solutions_count_l754_75451

theorem complex_solutions_count : ∃ (S : Finset ℂ), 
  (∀ z ∈ S, (z^3 - 1) / (z^2 - z - 2) = 0) ∧ 
  (∀ z : ℂ, (z^3 - 1) / (z^2 - z - 2) = 0 → z ∈ S) ∧ 
  Finset.card S = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_solutions_count_l754_75451


namespace NUMINAMATH_CALUDE_tan_alpha_value_l754_75431

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin (Real.pi - α) = 1/3)
  (h2 : Real.sin (2 * α) > 0) :
  Real.tan α = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l754_75431


namespace NUMINAMATH_CALUDE_work_completion_time_l754_75432

/-- The time it takes to complete a work given two workers with different rates and a specific work pattern. -/
theorem work_completion_time 
  (total_work : ℝ) 
  (p_rate : ℝ) 
  (q_rate : ℝ) 
  (p_alone_time : ℝ) :
  p_rate = total_work / 10 →
  q_rate = total_work / 6 →
  p_alone_time = 2 →
  let remaining_work := total_work - p_rate * p_alone_time
  let combined_rate := p_rate + q_rate
  total_work > 0 →
  p_rate > 0 →
  q_rate > 0 →
  p_alone_time + remaining_work / combined_rate = 5 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l754_75432


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l754_75422

/-- The lateral surface area of a cone with base radius 2 and height 1 is 2√5π -/
theorem cone_lateral_surface_area :
  let r : ℝ := 2  -- base radius
  let h : ℝ := 1  -- height
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height
  r * l * Real.pi = 2 * Real.sqrt 5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l754_75422


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l754_75421

theorem min_value_and_inequality (a b x₁ x₂ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hab : a + b = 1) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ / a + x₂ / b + 2 / (x₁ * x₂) ≥ 6) ∧
  (a * x₁ + b * x₂) * (a * x₂ + b * x₁) ≥ x₁ * x₂ := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l754_75421


namespace NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l754_75430

/-- The final price of a coffee and cheesecake set with a discount -/
theorem coffee_cheesecake_set_price 
  (coffee_price : ℝ) 
  (cheesecake_price : ℝ) 
  (discount_rate : ℝ) 
  (h1 : coffee_price = 6)
  (h2 : cheesecake_price = 10)
  (h3 : discount_rate = 0.25) :
  coffee_price + cheesecake_price - (coffee_price + cheesecake_price) * discount_rate = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l754_75430


namespace NUMINAMATH_CALUDE_circle_cartesian_and_center_l754_75469

-- Define the circle C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Theorem statement
theorem circle_cartesian_and_center :
  ∃ (x y : ℝ), 
    (∀ (ρ θ : ℝ), C ρ θ ↔ x^2 - 2*x + y^2 = 0) ∧
    (x = 1 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_cartesian_and_center_l754_75469


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l754_75453

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The asymptotic equations of the hyperbola are y = ± (slope * x) -/
  slope : ℝ
  /-- The focal length of the hyperbola -/
  focal_length : ℝ

/-- Checks if the given equation is a valid standard form for the hyperbola -/
def is_standard_equation (h : Hyperbola) (eq : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, eq x y = 0 ↔ x^2 / 4 - y^2 = 1) ∨
  (∀ x y, eq x y = 0 ↔ y^2 - x^2 / 4 = 1)

/-- The main theorem stating the standard equation of the hyperbola -/
theorem hyperbola_standard_equation (h : Hyperbola) 
  (h_slope : h.slope = 1/2) 
  (h_focal : h.focal_length = 2 * Real.sqrt 5) :
  ∃ eq : ℝ → ℝ → ℝ, is_standard_equation h eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l754_75453


namespace NUMINAMATH_CALUDE_negation_of_universal_square_geq_one_l754_75411

theorem negation_of_universal_square_geq_one :
  (¬ ∀ x : ℝ, x^2 ≥ 1) ↔ (∃ x : ℝ, x^2 < 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_square_geq_one_l754_75411


namespace NUMINAMATH_CALUDE_cubic_equation_with_geometric_roots_l754_75466

/-- Given a cubic equation x^3 - 14x^2 + ax - 27 = 0 with three distinct real roots in geometric progression, prove that a = 42 -/
theorem cubic_equation_with_geometric_roots (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧  -- distinct roots
    (∃ r : ℝ, r ≠ 0 ∧ x₂ = x₁ * r ∧ x₃ = x₂ * r) ∧  -- geometric progression
    (x₁^3 - 14*x₁^2 + a*x₁ - 27 = 0) ∧
    (x₂^3 - 14*x₂^2 + a*x₂ - 27 = 0) ∧
    (x₃^3 - 14*x₃^2 + a*x₃ - 27 = 0)) →
  a = 42 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_with_geometric_roots_l754_75466


namespace NUMINAMATH_CALUDE_children_ages_sum_l754_75495

theorem children_ages_sum (a b c d : ℕ) : 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 882 →
  a + b + c + d = 31 := by
  sorry

end NUMINAMATH_CALUDE_children_ages_sum_l754_75495


namespace NUMINAMATH_CALUDE_minimum_rent_is_36800_l754_75481

/-- Represents the minimum rent problem for a travel agency --/
def MinimumRentProblem (total_passengers : ℕ) (capacity_A capacity_B : ℕ) (rent_A rent_B : ℕ) (max_buses : ℕ) (max_B_diff : ℕ) : Prop :=
  ∃ (num_A num_B : ℕ),
    -- Total passengers condition
    num_A * capacity_A + num_B * capacity_B ≥ total_passengers ∧
    -- Maximum number of buses condition
    num_A + num_B ≤ max_buses ∧
    -- Condition on the difference between B and A buses
    num_B ≤ num_A + max_B_diff ∧
    -- Minimum rent calculation
    ∀ (other_A other_B : ℕ),
      other_A * capacity_A + other_B * capacity_B ≥ total_passengers →
      other_A + other_B ≤ max_buses →
      other_B ≤ other_A + max_B_diff →
      num_A * rent_A + num_B * rent_B ≤ other_A * rent_A + other_B * rent_B

/-- The minimum rent for the given problem is 36800 yuan --/
theorem minimum_rent_is_36800 :
  MinimumRentProblem 900 36 60 1600 2400 21 7 →
  ∃ (num_A num_B : ℕ), num_A * 1600 + num_B * 2400 = 36800 :=
sorry

end NUMINAMATH_CALUDE_minimum_rent_is_36800_l754_75481


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l754_75418

/-- A collection of stamps with various properties -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreignAndOld : ℕ
  neitherForeignNorOld : ℕ

/-- The number of foreign stamps in the collection -/
def foreignStamps (sc : StampCollection) : ℕ :=
  sc.total - sc.old + sc.foreignAndOld - sc.neitherForeignNorOld

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (sc : StampCollection)
    (h1 : sc.total = 200)
    (h2 : sc.old = 50)
    (h3 : sc.foreignAndOld = 20)
    (h4 : sc.neitherForeignNorOld = 90) :
    foreignStamps sc = 80 := by
  sorry

end NUMINAMATH_CALUDE_foreign_stamps_count_l754_75418


namespace NUMINAMATH_CALUDE_remaining_payment_theorem_l754_75488

def calculate_remaining_payment (deposit : ℚ) (percentage : ℚ) : ℚ :=
  deposit / percentage - deposit

def total_remaining_payment (deposit1 deposit2 deposit3 : ℚ) (percentage1 percentage2 percentage3 : ℚ) : ℚ :=
  calculate_remaining_payment deposit1 percentage1 +
  calculate_remaining_payment deposit2 percentage2 +
  calculate_remaining_payment deposit3 percentage3

theorem remaining_payment_theorem (deposit1 deposit2 deposit3 : ℚ) (percentage1 percentage2 percentage3 : ℚ)
  (h1 : deposit1 = 105)
  (h2 : deposit2 = 180)
  (h3 : deposit3 = 300)
  (h4 : percentage1 = 1/10)
  (h5 : percentage2 = 15/100)
  (h6 : percentage3 = 1/5) :
  total_remaining_payment deposit1 deposit2 deposit3 percentage1 percentage2 percentage3 = 3165 := by
  sorry

#eval total_remaining_payment 105 180 300 (1/10) (15/100) (1/5)

end NUMINAMATH_CALUDE_remaining_payment_theorem_l754_75488


namespace NUMINAMATH_CALUDE_smallest_b_value_l754_75499

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^4 + b.val^4) / (a.val + b.val)) (a.val * b.val) = 16) :
  b.val ≥ 4 ∧ ∃ (a₀ b₀ : ℕ+), b₀.val = 4 ∧ a₀.val - b₀.val = 8 ∧ 
    Nat.gcd ((a₀.val^4 + b₀.val^4) / (a₀.val + b₀.val)) (a₀.val * b₀.val) = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l754_75499


namespace NUMINAMATH_CALUDE_pablo_works_seven_hours_l754_75475

/-- Represents the puzzle-solving scenario for Pablo --/
structure PuzzleScenario where
  pieces_per_hour : ℕ
  small_puzzles : ℕ
  small_puzzle_pieces : ℕ
  large_puzzles : ℕ
  large_puzzle_pieces : ℕ
  days_to_complete : ℕ

/-- Calculates the hours Pablo works on puzzles each day --/
def hours_per_day (scenario : PuzzleScenario) : ℚ :=
  let total_pieces := scenario.small_puzzles * scenario.small_puzzle_pieces +
                      scenario.large_puzzles * scenario.large_puzzle_pieces
  let total_hours := total_pieces / scenario.pieces_per_hour
  total_hours / scenario.days_to_complete

/-- Theorem stating that Pablo works 7 hours per day on puzzles --/
theorem pablo_works_seven_hours (scenario : PuzzleScenario) 
  (h1 : scenario.pieces_per_hour = 100)
  (h2 : scenario.small_puzzles = 8)
  (h3 : scenario.small_puzzle_pieces = 300)
  (h4 : scenario.large_puzzles = 5)
  (h5 : scenario.large_puzzle_pieces = 500)
  (h6 : scenario.days_to_complete = 7) :
  hours_per_day scenario = 7 := by
  sorry

end NUMINAMATH_CALUDE_pablo_works_seven_hours_l754_75475


namespace NUMINAMATH_CALUDE_complex_absolute_value_l754_75423

theorem complex_absolute_value (x : ℝ) (h : x > 0) :
  Complex.abs (-3 + 2*x*Complex.I) = 5 * Real.sqrt 5 ↔ x = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l754_75423


namespace NUMINAMATH_CALUDE_correct_result_l754_75447

theorem correct_result (mistaken_result : ℕ) 
  (ones_digit_mistake : ℕ) (tens_digit_mistake : ℕ) : 
  mistaken_result = 387 ∧ 
  ones_digit_mistake = 8 - 3 ∧ 
  tens_digit_mistake = 90 - 50 → 
  mistaken_result - ones_digit_mistake + tens_digit_mistake = 422 :=
by sorry

end NUMINAMATH_CALUDE_correct_result_l754_75447


namespace NUMINAMATH_CALUDE_david_did_more_pushups_l754_75405

/-- The number of push-ups David did -/
def david_pushups : ℕ := 44

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 35

/-- The difference in push-ups between David and Zachary -/
def pushup_difference : ℕ := david_pushups - zachary_pushups

theorem david_did_more_pushups : pushup_difference = 9 := by
  sorry

end NUMINAMATH_CALUDE_david_did_more_pushups_l754_75405


namespace NUMINAMATH_CALUDE_line_point_ratio_l754_75489

/-- Given four points A, B, C, D on a directed line such that AC/CB + AD/DB = 0,
    prove that 1/AC + 1/AD = 2/AB -/
theorem line_point_ratio (A B C D : ℝ) (h : (C - A) / (B - C) + (D - A) / (B - D) = 0) :
  1 / (C - A) + 1 / (D - A) = 2 / (B - A) := by
  sorry

end NUMINAMATH_CALUDE_line_point_ratio_l754_75489


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l754_75416

theorem absolute_value_equation_solution :
  ∃ x : ℚ, (|x - 1| = |x - 2|) ∧ (x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l754_75416


namespace NUMINAMATH_CALUDE_roses_in_bouquet_l754_75444

/-- The number of bouquets -/
def num_bouquets : ℕ := 5

/-- The number of table decorations -/
def num_table_decorations : ℕ := 7

/-- The number of white roses in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed -/
def total_roses : ℕ := 109

/-- The number of white roses in each bouquet -/
def roses_per_bouquet : ℕ := (total_roses - num_table_decorations * roses_per_table_decoration) / num_bouquets

theorem roses_in_bouquet :
  roses_per_bouquet = 5 :=
by sorry

end NUMINAMATH_CALUDE_roses_in_bouquet_l754_75444


namespace NUMINAMATH_CALUDE_inverse_g_87_l754_75461

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 6

-- Theorem statement
theorem inverse_g_87 : g⁻¹ 87 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_87_l754_75461


namespace NUMINAMATH_CALUDE_median_length_right_triangle_l754_75487

theorem median_length_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let median := (1 / 2 : ℝ) * c
  median = 5 := by
sorry

end NUMINAMATH_CALUDE_median_length_right_triangle_l754_75487


namespace NUMINAMATH_CALUDE_circle_equation_l754_75446

/-- A circle C in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle at a given point -/
def Circle.tangentAt (c : Circle) (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  c.contains p ∧ 
  (c.center.2 - p.2) / (c.center.1 - p.1) = -1 / m ∧
  p.2 = m * p.1 + b

/-- The main theorem -/
theorem circle_equation (C : Circle) :
  C.center = (3, 0) ∧ C.radius = 2 →
  C.contains (4, 1) ∧
  C.tangentAt 1 (-2) (2, 1) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l754_75446


namespace NUMINAMATH_CALUDE_inequality_solution_set_l754_75437

theorem inequality_solution_set (a : ℝ) :
  let f := fun x : ℝ => (a^2 - 4) * x^2 + 4 * x - 1
  (∀ x, f x > 0 ↔ 
    (a = 2 ∨ a = -2 → x > 1/4) ∧
    (a > 2 → x > 1/(a+2) ∨ x < 1/(2-a)) ∧
    (a < -2 → x < 1/(a+2) ∨ x > 1/(2-a)) ∧
    (-2 < a ∧ a < 2 → 1/(a+2) < x ∧ x < 1/(2-a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l754_75437


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_difference_56_l754_75414

-- Define the property for two numbers to have the same last two digits in their squares
def SameLastTwoDigitsSquared (a b : ℕ) : Prop :=
  a ^ 2 % 100 = b ^ 2 % 100

-- Main theorem
theorem two_digit_numbers_with_difference_56 :
  ∀ x y : ℕ,
    10 ≤ x ∧ x < 100 →  -- x is a two-digit number
    10 ≤ y ∧ y < 100 →  -- y is a two-digit number
    x - y = 56 →        -- their difference is 56
    SameLastTwoDigitsSquared x y →  -- last two digits of their squares are the same
    (x = 78 ∧ y = 22) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_difference_56_l754_75414


namespace NUMINAMATH_CALUDE_parallelepiped_edge_lengths_l754_75484

/-- Given a rectangular parallelepiped with mass M and density ρ, and thermal power ratios of 1:2:8
    when connected to different pairs of faces, this theorem states the edge lengths of the parallelepiped. -/
theorem parallelepiped_edge_lengths (M ρ : ℝ) (hM : M > 0) (hρ : ρ > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a < b ∧ b < c ∧
    a * b * c = M / ρ ∧
    b^2 / a^2 = 2 ∧
    c^2 / b^2 = 4 ∧
    a = (M / (4 * ρ))^(1/3) ∧
    b = Real.sqrt 2 * (M / (4 * ρ))^(1/3) ∧
    c = 2 * Real.sqrt 2 * (M / (4 * ρ))^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_parallelepiped_edge_lengths_l754_75484


namespace NUMINAMATH_CALUDE_f_inequalities_l754_75492

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x - a

theorem f_inequalities (a : ℝ) :
  (a < -1 → {x : ℝ | f a x < 0} = Set.Ioo a (-1)) ∧
  (a = -1 → {x : ℝ | f a x < 0} = ∅) ∧
  (a > -1 → {x : ℝ | f a x < 0} = Set.Ioo (-1) a) ∧
  ({x : ℝ | x^3 * f 2 x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 2) :=
by sorry


end NUMINAMATH_CALUDE_f_inequalities_l754_75492


namespace NUMINAMATH_CALUDE_unhappy_redheads_ratio_l754_75419

theorem unhappy_redheads_ratio 
  (x y z : ℕ) -- x: happy subjects, y: redheads, z: total subjects
  (h1 : (40 : ℚ) / 100 * x = (60 : ℚ) / 100 * y) -- Condition 1
  (h2 : z = x + (40 : ℚ) / 100 * y) -- Condition 2
  : (y - ((40 : ℚ) / 100 * y).floor) / z = 4 / 19 := by
  sorry


end NUMINAMATH_CALUDE_unhappy_redheads_ratio_l754_75419


namespace NUMINAMATH_CALUDE_all_star_seating_l754_75494

/-- Represents the number of ways to seat 9 baseball All-Stars from 3 teams -/
def seating_arrangements : ℕ :=
  let num_teams : ℕ := 3
  let players_per_team : ℕ := 3
  let team_arrangements : ℕ := Nat.factorial num_teams
  let within_team_arrangements : ℕ := Nat.factorial players_per_team
  team_arrangements * (within_team_arrangements ^ num_teams)

/-- Theorem stating the number of seating arrangements for 9 baseball All-Stars -/
theorem all_star_seating :
  seating_arrangements = 1296 := by
  sorry

end NUMINAMATH_CALUDE_all_star_seating_l754_75494


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l754_75462

/-- Given two rectangles A and B, where A has a perimeter of 40 cm and its length is twice its width,
    and B has an area equal to one-half the area of A and its length is twice its width,
    prove that the perimeter of B is 20√2 cm. -/
theorem rectangle_perimeter (width_A : ℝ) (width_B : ℝ) : 
  (2 * (width_A + 2 * width_A) = 40) →  -- Perimeter of A is 40 cm
  (width_B * (2 * width_B) = (width_A * (2 * width_A)) / 2) →  -- Area of B is half of A
  (2 * (width_B + 2 * width_B) = 20 * Real.sqrt 2) :=  -- Perimeter of B is 20√2 cm
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l754_75462


namespace NUMINAMATH_CALUDE_madeline_free_time_l754_75476

/-- Calculates the number of hours Madeline has left over in a week --/
theorem madeline_free_time (class_hours week_days daily_hours homework_hours sleep_hours work_hours : ℕ) :
  class_hours = 18 →
  week_days = 7 →
  daily_hours = 24 →
  homework_hours = 4 →
  sleep_hours = 8 →
  work_hours = 20 →
  daily_hours * week_days - (class_hours + homework_hours * week_days + sleep_hours * week_days + work_hours) = 46 := by
  sorry

end NUMINAMATH_CALUDE_madeline_free_time_l754_75476


namespace NUMINAMATH_CALUDE_money_distribution_l754_75449

/-- Given that A, B, and C have a total of 500 Rs between them,
    B and C together have 320 Rs, and C has 20 Rs,
    prove that A and C together have 200 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 →
  B + C = 320 →
  C = 20 →
  A + C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l754_75449


namespace NUMINAMATH_CALUDE_bottom_row_bricks_l754_75440

/-- Represents a pyramidal brick wall -/
structure PyramidalWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the total number of bricks in a pyramidal wall -/
def calculateTotalBricks (wall : PyramidalWall) : ℕ :=
  (wall.rows : ℕ) * (2 * wall.bottomRowBricks - wall.rows + 1) / 2

theorem bottom_row_bricks (wall : PyramidalWall) 
  (h1 : wall.rows = 15)
  (h2 : wall.totalBricks = 300)
  (h3 : calculateTotalBricks wall = wall.totalBricks) :
  wall.bottomRowBricks = 27 := by
  sorry

end NUMINAMATH_CALUDE_bottom_row_bricks_l754_75440


namespace NUMINAMATH_CALUDE_power_sum_difference_l754_75480

theorem power_sum_difference : 3^(1+2+3+4) - (3^1 + 3^2 + 3^3 + 3^4) = 58929 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l754_75480


namespace NUMINAMATH_CALUDE_equilateral_triangle_theorem_l754_75472

open Real

/-- Triangle with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The equation given in the problem -/
def equation_holds (t : Triangle) : Prop :=
  (t.a * cos t.A + t.b * cos t.B + t.c * cos t.C) / 
  (t.a * sin t.A + t.b * sin t.B + t.c * sin t.C) = 
  (t.a + t.b + t.c) / (9 * circumradius t)

/-- The main theorem to prove -/
theorem equilateral_triangle_theorem (t : Triangle) :
  equation_holds t → t.a = t.b ∧ t.b = t.c := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_theorem_l754_75472


namespace NUMINAMATH_CALUDE_same_average_speed_exists_l754_75463

theorem same_average_speed_exists : ∃ y : ℝ, 
  (y^2 - 14*y + 45 = (y^2 - 2*y - 35) / (y - 5)) ∧ 
  (y^2 - 14*y + 45 = 6) := by
  sorry

end NUMINAMATH_CALUDE_same_average_speed_exists_l754_75463


namespace NUMINAMATH_CALUDE_whole_number_between_bounds_l754_75401

theorem whole_number_between_bounds (M : ℤ) :
  (9.5 < (M : ℚ) / 5 ∧ (M : ℚ) / 5 < 10.5) ↔ (M = 49 ∨ M = 50 ∨ M = 51) := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_bounds_l754_75401


namespace NUMINAMATH_CALUDE_logan_max_rent_l754_75408

def current_income : ℕ := 65000
def grocery_expenses : ℕ := 5000
def gas_expenses : ℕ := 8000
def desired_savings : ℕ := 42000
def income_increase : ℕ := 10000

def max_rent : ℕ := 20000

theorem logan_max_rent :
  max_rent = current_income + income_increase - desired_savings - grocery_expenses - gas_expenses :=
by sorry

end NUMINAMATH_CALUDE_logan_max_rent_l754_75408


namespace NUMINAMATH_CALUDE_segments_form_triangle_l754_75460

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (4, 5, 7) can form a triangle -/
theorem segments_form_triangle : can_form_triangle 4 5 7 := by
  sorry

end NUMINAMATH_CALUDE_segments_form_triangle_l754_75460


namespace NUMINAMATH_CALUDE_cone_volume_l754_75467

/-- The volume of a cone whose lateral surface unfolds to a semicircle with radius 2 -/
theorem cone_volume (r : Real) (h : Real) : 
  r = 1 → h = Real.sqrt 3 → (1/3 : Real) * π * r^2 * h = (Real.sqrt 3 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l754_75467


namespace NUMINAMATH_CALUDE_only_solution_is_one_l754_75448

theorem only_solution_is_one : 
  ∀ n : ℕ, (2 * n - 1 : ℚ) / (n^5 : ℚ) = 3 - 2 / (n : ℚ) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_only_solution_is_one_l754_75448


namespace NUMINAMATH_CALUDE_equation_solution_l754_75438

theorem equation_solution : ∃! x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l754_75438


namespace NUMINAMATH_CALUDE_isosceles_triangles_equal_perimeter_area_l754_75443

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equal_side : ℝ
  base : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equal_side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let h := Real.sqrt (t.equal_side^2 - (t.base/2)^2)
  (1/2) * t.base * h

/-- The theorem to be proved -/
theorem isosceles_triangles_equal_perimeter_area (t1 t2 : IsoscelesTriangle)
  (h1 : t1.equal_side = 6 ∧ t1.base = 10)
  (h2 : perimeter t1 = perimeter t2)
  (h3 : area t1 = area t2)
  (h4 : t2.base^2 / 2 = perimeter t2 / 2) :
  t2.base = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_equal_perimeter_area_l754_75443


namespace NUMINAMATH_CALUDE_work_completion_time_l754_75490

/-- The number of days it takes for the original number of ladies to complete the work -/
def completion_time (original_ladies : ℕ) : ℝ :=
  6

/-- The time it takes for twice the number of ladies to complete half the work -/
def half_work_time (original_ladies : ℕ) : ℝ :=
  3

theorem work_completion_time (original_ladies : ℕ) :
  completion_time original_ladies = 2 * half_work_time original_ladies :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l754_75490


namespace NUMINAMATH_CALUDE_fundraiser_total_l754_75445

theorem fundraiser_total (brownie_students : Nat) (brownie_per_student : Nat) (brownie_price : Real)
                         (cookie_students : Nat) (cookie_per_student : Nat) (cookie_price : Real)
                         (donut_students : Nat) (donut_per_student : Nat) (donut_price : Real)
                         (cupcake_students : Nat) (cupcake_per_student : Nat) (cupcake_price : Real) :
  brownie_students = 70 ∧ brownie_per_student = 20 ∧ brownie_price = 1.50 ∧
  cookie_students = 40 ∧ cookie_per_student = 30 ∧ cookie_price = 2.25 ∧
  donut_students = 35 ∧ donut_per_student = 18 ∧ donut_price = 3.00 ∧
  cupcake_students = 25 ∧ cupcake_per_student = 12 ∧ cupcake_price = 2.50 →
  (brownie_students * brownie_per_student * brownie_price +
   cookie_students * cookie_per_student * cookie_price +
   donut_students * donut_per_student * donut_price +
   cupcake_students * cupcake_per_student * cupcake_price) = 7440 :=
by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l754_75445


namespace NUMINAMATH_CALUDE_sum_not_zero_l754_75470

theorem sum_not_zero (a b c d : ℝ) 
  (eq1 : a * b * c * d - d = 1)
  (eq2 : b * c * d - a = 2)
  (eq3 : c * d * a - b = 3)
  (eq4 : d * a * b - c = -6) :
  a + b + c + d ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_sum_not_zero_l754_75470


namespace NUMINAMATH_CALUDE_train_length_l754_75459

theorem train_length (t : ℝ) 
  (h1 : (t + 100) / 15 = (t + 250) / 20) : t = 350 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l754_75459


namespace NUMINAMATH_CALUDE_inequality_preserved_by_halving_l754_75478

theorem inequality_preserved_by_halving {a b : ℝ} (h : a > b) : a / 2 > b / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preserved_by_halving_l754_75478


namespace NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_acute_l754_75409

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < Real.pi / 2

-- Define the statement we want to prove false
def sum_of_acute_angles_always_acute : Prop :=
  ∀ (a b : ℝ), is_acute_angle a → is_acute_angle b → is_acute_angle (a + b)

-- Theorem stating that the above statement is false
theorem sum_of_acute_angles_not_always_acute :
  ¬ sum_of_acute_angles_always_acute :=
sorry

end NUMINAMATH_CALUDE_sum_of_acute_angles_not_always_acute_l754_75409


namespace NUMINAMATH_CALUDE_usable_field_area_l754_75427

/-- Calculates the area of a usable rectangular field with an L-shaped obstacle -/
theorem usable_field_area
  (breadth : ℕ)
  (h1 : breadth + 30 = 150)  -- Length is 30 meters more than breadth
  (h2 : 2 * (breadth + (breadth + 30)) = 540)  -- Perimeter is 540 meters
  : (breadth - 5) * (breadth + 30 - 10) = 16100 :=
by sorry

end NUMINAMATH_CALUDE_usable_field_area_l754_75427


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l754_75493

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-2, -1, 0}
def B : Set Int := {0, 1, 2}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l754_75493


namespace NUMINAMATH_CALUDE_joggers_meeting_times_l754_75404

theorem joggers_meeting_times (road_length : ℝ) (speed_a : ℝ) (speed_b : ℝ) (duration : ℝ) :
  road_length = 400 ∧
  speed_a = 3 ∧
  speed_b = 2.5 ∧
  duration = 20 * 60 →
  ∃ n : ℕ, n = 8 ∧ 
    (road_length + (n - 1) * 2 * road_length) / (speed_a + speed_b) = duration :=
by sorry

end NUMINAMATH_CALUDE_joggers_meeting_times_l754_75404


namespace NUMINAMATH_CALUDE_ram_krish_efficiency_ratio_l754_75426

/-- Ram's efficiency -/
def ram_efficiency : ℝ := 1

/-- Krish's efficiency -/
def krish_efficiency : ℝ := 2

/-- Time taken by Ram alone to complete the task -/
def ram_alone_time : ℝ := 30

/-- Time taken by Ram and Krish together to complete the task -/
def combined_time : ℝ := 10

/-- The amount of work to be done -/
def work : ℝ := ram_efficiency * ram_alone_time

theorem ram_krish_efficiency_ratio :
  ram_efficiency / krish_efficiency = 1 / 2 ∧
  work = ram_efficiency * ram_alone_time ∧
  work = (ram_efficiency + krish_efficiency) * combined_time :=
by sorry

end NUMINAMATH_CALUDE_ram_krish_efficiency_ratio_l754_75426


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l754_75436

/-- Given a geometric sequence {a_n} with common ratio q > 1,
    if a_5 - a_1 = 15 and a_4 - a_2 = 6, then a_3 = 4 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 1 →  -- common ratio greater than 1
  a 5 - a 1 = 15 →  -- condition on a_5 and a_1
  a 4 - a 2 = 6 →  -- condition on a_4 and a_2
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l754_75436


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l754_75406

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l754_75406


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l754_75441

/-- The length of the major axis of an ellipse with equation x^2/25 + y^2/16 = 1 is 10 -/
theorem ellipse_major_axis_length : 
  ∀ x y : ℝ, x^2/25 + y^2/16 = 1 → 
  ∃ a b : ℝ, a ≥ b ∧ a^2 = 25 ∧ b^2 = 16 ∧ 2*a = 10 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l754_75441
