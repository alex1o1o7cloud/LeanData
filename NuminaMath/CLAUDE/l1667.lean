import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_expression_l1667_166710

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  (∃ m : ℝ, ∀ a b : ℝ, a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 = m) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ m ∧
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 = m) → 
  m = 625/4) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1667_166710


namespace NUMINAMATH_CALUDE_sum_of_squares_l1667_166720

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 2) (h2 : x^3 + y^3 = 3) : x^2 + y^2 = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1667_166720


namespace NUMINAMATH_CALUDE_baker_cheesecake_problem_l1667_166785

/-- Calculates the total number of cheesecakes left to be sold given the initial quantities and sales. -/
def cheesecakes_left_to_sell (display_initial : ℕ) (fridge_initial : ℕ) (sold_from_display : ℕ) : ℕ :=
  (display_initial - sold_from_display) + fridge_initial

/-- Theorem stating that given the specific initial quantities and sales, 18 cheesecakes are left to be sold. -/
theorem baker_cheesecake_problem :
  cheesecakes_left_to_sell 10 15 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_baker_cheesecake_problem_l1667_166785


namespace NUMINAMATH_CALUDE_elvis_matchsticks_l1667_166719

theorem elvis_matchsticks (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) 
  (ralph_per_square : ℕ) (leftover : ℕ) :
  total = 50 →
  elvis_squares = 5 →
  ralph_squares = 3 →
  ralph_per_square = 8 →
  leftover = 6 →
  ∃ (elvis_per_square : ℕ), 
    elvis_per_square * elvis_squares + ralph_per_square * ralph_squares + leftover = total ∧
    elvis_per_square = 4 :=
by sorry

end NUMINAMATH_CALUDE_elvis_matchsticks_l1667_166719


namespace NUMINAMATH_CALUDE_rectangle_center_sum_l1667_166770

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the conditions
def rectangle_conditions (rect : Rectangle) : Prop :=
  -- Rectangle is in the first quadrant
  rect.A.1 ≥ 0 ∧ rect.A.2 ≥ 0 ∧
  rect.B.1 ≥ 0 ∧ rect.B.2 ≥ 0 ∧
  rect.C.1 ≥ 0 ∧ rect.C.2 ≥ 0 ∧
  rect.D.1 ≥ 0 ∧ rect.D.2 ≥ 0 ∧
  -- Points on the lines
  (2 : ℝ) ∈ Set.Icc rect.D.1 rect.A.1 ∧
  (6 : ℝ) ∈ Set.Icc rect.C.1 rect.B.1 ∧
  (10 : ℝ) ∈ Set.Icc rect.A.1 rect.B.1 ∧
  (18 : ℝ) ∈ Set.Icc rect.C.1 rect.D.1 ∧
  -- Ratio of AB to BC is 2:1
  2 * (rect.B.1 - rect.C.1) = rect.B.1 - rect.A.1

-- Theorem statement
theorem rectangle_center_sum (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  (rect.A.1 + rect.C.1) / 2 + (rect.A.2 + rect.C.2) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_center_sum_l1667_166770


namespace NUMINAMATH_CALUDE_polygon_area_l1667_166735

-- Define a point in 2D space
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the polygon
def polygon : List Point := [
  ⟨0, 0⟩, ⟨12, 0⟩, ⟨24, 12⟩, ⟨24, 0⟩, ⟨36, 0⟩,
  ⟨36, 24⟩, ⟨24, 36⟩, ⟨12, 36⟩, ⟨0, 36⟩, ⟨0, 24⟩
]

-- Function to calculate the area of the polygon
def calculateArea (vertices : List Point) : ℤ :=
  sorry

-- Theorem stating that the area of the polygon is 1008 square units
theorem polygon_area : calculateArea polygon = 1008 :=
  sorry

end NUMINAMATH_CALUDE_polygon_area_l1667_166735


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1667_166741

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x - a > -3) → a ∈ Set.Ioo (-6) 2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l1667_166741


namespace NUMINAMATH_CALUDE_siblings_total_age_l1667_166729

/-- Given the age ratio of Halima, Beckham, and Michelle as 4:3:7, and the age difference
    between Halima and Beckham as 9 years, prove that the total age of the three siblings
    is 126 years. -/
theorem siblings_total_age
  (halima_ratio : ℕ) (beckham_ratio : ℕ) (michelle_ratio : ℕ)
  (age_ratio : halima_ratio = 4 ∧ beckham_ratio = 3 ∧ michelle_ratio = 7)
  (age_difference : ℕ) (halima_beckham_diff : age_difference = 9)
  : ∃ (x : ℕ), 
    halima_ratio * x - beckham_ratio * x = age_difference ∧
    halima_ratio * x + beckham_ratio * x + michelle_ratio * x = 126 := by
  sorry

end NUMINAMATH_CALUDE_siblings_total_age_l1667_166729


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l1667_166722

/-- Calculate the overall gain percentage for three articles -/
theorem overall_gain_percentage
  (cost_A cost_B cost_C : ℝ)
  (sell_A sell_B sell_C : ℝ)
  (h_cost_A : cost_A = 100)
  (h_cost_B : cost_B = 200)
  (h_cost_C : cost_C = 300)
  (h_sell_A : sell_A = 110)
  (h_sell_B : sell_B = 250)
  (h_sell_C : sell_C = 330) :
  (((sell_A + sell_B + sell_C) - (cost_A + cost_B + cost_C)) / (cost_A + cost_B + cost_C)) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_overall_gain_percentage_l1667_166722


namespace NUMINAMATH_CALUDE_two_hour_walk_distance_l1667_166740

/-- Calculates the total distance walked in two hours given the distance walked in the first hour -/
def total_distance (first_hour_distance : ℝ) : ℝ :=
  first_hour_distance + 2 * first_hour_distance

/-- Theorem stating that walking 2 km in the first hour and twice that in the second hour results in 6 km total -/
theorem two_hour_walk_distance :
  total_distance 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_walk_distance_l1667_166740


namespace NUMINAMATH_CALUDE_volunteer_distribution_l1667_166712

/-- The number of ways to distribute n girls and m boys into two groups,
    where each group must have at least one girl and one boy. -/
def distribution_schemes (n m : ℕ) : ℕ :=
  if n < 2 ∨ m < 2 then 0
  else (Nat.choose n 1 + Nat.choose n 2) * Nat.factorial m

/-- The problem statement -/
theorem volunteer_distribution : distribution_schemes 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l1667_166712


namespace NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1667_166778

-- Define the start time and quarter completion time
def start_time : Nat := 7 * 60  -- 7:00 AM in minutes
def quarter_completion_time : Nat := 10 * 60  -- 10:00 AM in minutes

-- Define the maintenance break duration
def maintenance_break : Nat := 30  -- 30 minutes

-- Theorem to prove
theorem doughnut_machine_completion_time : 
  -- Given conditions
  (quarter_completion_time - start_time = 3 * 60) →  -- 3 hours to complete 1/4 of the job
  -- Conclusion
  (start_time + 4 * (quarter_completion_time - start_time) + maintenance_break = 19 * 60 + 30) :=  -- 7:30 PM in minutes
by
  sorry

end NUMINAMATH_CALUDE_doughnut_machine_completion_time_l1667_166778


namespace NUMINAMATH_CALUDE_all_pairs_divisible_by_seven_l1667_166791

-- Define the type for pairs on the board
def BoardPair := ℤ × ℤ

-- Define the property that 2a - b is divisible by 7
def DivisibleBySeven (p : BoardPair) : Prop :=
  ∃ k : ℤ, 2 * p.1 - p.2 = 7 * k

-- Define the set of all pairs that can appear on the board
inductive ValidPair : BoardPair → Prop where
  | initial : ValidPair (1, 2)
  | negate (a b : ℤ) : ValidPair (a, b) → ValidPair (-a, -b)
  | rotate (a b : ℤ) : ValidPair (a, b) → ValidPair (-b, a + b)
  | add (a b c d : ℤ) : ValidPair (a, b) → ValidPair (c, d) → ValidPair (a + c, b + d)

-- Theorem statement
theorem all_pairs_divisible_by_seven :
  ∀ p : BoardPair, ValidPair p → DivisibleBySeven p :=
  sorry

end NUMINAMATH_CALUDE_all_pairs_divisible_by_seven_l1667_166791


namespace NUMINAMATH_CALUDE_min_value_theorem_l1667_166790

def is_arithmetic_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (d q : ℝ), q > 0 ∧ ∀ n, a (n + 1) = a n * q + d

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  is_arithmetic_geometric a →
  (∀ n, a n > 0) →
  a 7 = a 6 + 2 * a 5 →
  a m * a n = 4 * (a 1) ^ 2 →
  (1 : ℝ) / m + 4 / n ≥ 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1667_166790


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l1667_166701

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices (0, 0), (0, 3), (3, 3), and (3, 0) --/
def givenSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

/-- The condition x + y < 4 --/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_x_plus_y_less_than_4 :
  probability givenSquare condition = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l1667_166701


namespace NUMINAMATH_CALUDE_spade_nested_operation_l1667_166780

/-- The spade operation defined as the absolute difference between two numbers -/
def spade (a b : ℝ) : ℝ := |a - b|

/-- Theorem stating that 3 ♠ (5 ♠ (8 ♠ 11)) = 1 -/
theorem spade_nested_operation : spade 3 (spade 5 (spade 8 11)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_operation_l1667_166780


namespace NUMINAMATH_CALUDE_min_value_I_l1667_166798

theorem min_value_I (a b c x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x ≥ 0) (hy : y ≥ 0)
  (h_sum : a^6 + b^6 + c^6 = 3)
  (h_constraint : (x + 1)^2 + y^2 ≤ 2) : 
  let I := 1 / (2*a^3*x + b^3*y^2) + 1 / (2*b^3*x + c^3*y^2) + 1 / (2*c^3*x + a^3*y^2)
  ∀ I', I ≥ I' → I' ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_I_l1667_166798


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1667_166777

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, a * x - 7 + (b + 2) * x = 3) ↔ a ≠ -b - 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1667_166777


namespace NUMINAMATH_CALUDE_range_of_x_plus_inverse_x_l1667_166748

theorem range_of_x_plus_inverse_x (x : ℝ) (h : x < 0) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ -2 ∧ ∀ (z : ℝ), (∃ (w : ℝ), w < 0 ∧ z = w + 1/w) → z ≤ y :=
sorry

end NUMINAMATH_CALUDE_range_of_x_plus_inverse_x_l1667_166748


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1667_166716

theorem complex_fraction_simplification :
  (Complex.I * 3 - 1) / (1 + Complex.I * 3) = Complex.mk (4/5) (3/5) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1667_166716


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1667_166742

theorem trigonometric_expression_equality :
  let sin30 := (1 : ℝ) / 2
  let cos30 := Real.sqrt 3 / 2
  let tan60 := Real.sqrt 3
  2 * sin30 + cos30 * tan60 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1667_166742


namespace NUMINAMATH_CALUDE_sterilization_tank_solution_l1667_166734

/-- Represents the sterilization tank problem --/
def sterilization_tank_problem (initial_volume : ℝ) (drained_volume : ℝ) (final_concentration : ℝ) (initial_concentration : ℝ) : Prop :=
  let remaining_volume := initial_volume - drained_volume
  remaining_volume * initial_concentration + drained_volume = initial_volume * final_concentration

/-- Theorem stating the solution to the sterilization tank problem --/
theorem sterilization_tank_solution :
  sterilization_tank_problem 100 3.0612244898 0.05 0.02 := by
  sorry

end NUMINAMATH_CALUDE_sterilization_tank_solution_l1667_166734


namespace NUMINAMATH_CALUDE_birthday_paradox_l1667_166706

theorem birthday_paradox (n : ℕ) (h : n = 367) :
  ∃ (f : Fin n → Fin 366), ¬Function.Injective f :=
sorry

end NUMINAMATH_CALUDE_birthday_paradox_l1667_166706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1667_166793

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The theorem stating that if S_2 = 3 and S_3 = 3, then S_5 = 0 for an arithmetic sequence -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h2 : seq.S 2 = 3) (h3 : seq.S 3 = 3) : seq.S 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1667_166793


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1667_166765

/-- Proves the relationship between y-coordinates of three points on an inverse proportion function -/
theorem inverse_proportion_y_relationship :
  ∀ (y₁ y₂ y₃ : ℝ),
  (y₁ = -4 / (-1)) →  -- Point A(-1, y₁) lies on y = -4/x
  (y₂ = -4 / 2) →     -- Point B(2, y₂) lies on y = -4/x
  (y₃ = -4 / 3) →     -- Point C(3, y₃) lies on y = -4/x
  (y₁ > y₃ ∧ y₃ > y₂) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l1667_166765


namespace NUMINAMATH_CALUDE_expected_value_is_six_point_five_l1667_166768

/-- A function representing a fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of a roll of the 12-sided die -/
def expected_value : ℚ := (twelve_sided_die.sum id + twelve_sided_die.card) / (2 * twelve_sided_die.card)

/-- Theorem stating that the expected value of a roll of the 12-sided die is 6.5 -/
theorem expected_value_is_six_point_five : expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_six_point_five_l1667_166768


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l1667_166773

/-- Given a triangle and a trapezoid with the same altitude and equal areas,
    prove that the median of the trapezoid is 24 inches. -/
theorem trapezoid_median_length (h : ℝ) (triangle_area trapezoid_area : ℝ) : 
  triangle_area = (1/2) * 24 * h →
  trapezoid_area = ((15 + 33) / 2) * h →
  triangle_area = trapezoid_area →
  (15 + 33) / 2 = 24 := by
sorry


end NUMINAMATH_CALUDE_trapezoid_median_length_l1667_166773


namespace NUMINAMATH_CALUDE_enjoyable_gameplay_l1667_166757

theorem enjoyable_gameplay (total_hours : ℝ) (boring_percentage : ℝ) (expansion_hours : ℝ) :
  total_hours = 100 ∧ 
  boring_percentage = 80 ∧ 
  expansion_hours = 30 →
  (1 - boring_percentage / 100) * total_hours + expansion_hours = 50 := by
  sorry

end NUMINAMATH_CALUDE_enjoyable_gameplay_l1667_166757


namespace NUMINAMATH_CALUDE_square_diff_theorem_l1667_166756

theorem square_diff_theorem : (25 + 9)^2 - (25^2 + 9^2) = 450 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_theorem_l1667_166756


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l1667_166787

theorem sqrt_sum_problem (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) : 
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l1667_166787


namespace NUMINAMATH_CALUDE_min_value_theorem_l1667_166739

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  a + 4 * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ ∧ a₀ + 4 * b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1667_166739


namespace NUMINAMATH_CALUDE_second_shop_payment_l1667_166721

/-- The amount Rahim paid for the books from the second shop -/
def second_shop_amount (first_shop_books : ℕ) (second_shop_books : ℕ) (first_shop_amount : ℕ) (average_price : ℕ) : ℕ :=
  (first_shop_books + second_shop_books) * average_price - first_shop_amount

/-- Theorem stating the amount Rahim paid for the books from the second shop -/
theorem second_shop_payment :
  second_shop_amount 40 20 600 14 = 240 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_payment_l1667_166721


namespace NUMINAMATH_CALUDE_optimal_plan_l1667_166751

/-- Represents the unit price of type A prizes -/
def price_A : ℝ := 30

/-- Represents the unit price of type B prizes -/
def price_B : ℝ := 15

/-- The total number of prizes to purchase -/
def total_prizes : ℕ := 30

/-- Condition: Total cost of 3 type A and 2 type B prizes is 120 yuan -/
axiom condition1 : 3 * price_A + 2 * price_B = 120

/-- Condition: Total cost of 5 type A and 4 type B prizes is 210 yuan -/
axiom condition2 : 5 * price_A + 4 * price_B = 210

/-- Function to calculate the total cost given the number of type A prizes -/
def total_cost (num_A : ℕ) : ℝ :=
  price_A * num_A + price_B * (total_prizes - num_A)

/-- Theorem stating the most cost-effective plan and its total cost -/
theorem optimal_plan :
  ∃ (num_A : ℕ),
    num_A ≥ (total_prizes - num_A) / 3 ∧
    num_A = 8 ∧
    total_cost num_A = 570 ∧
    ∀ (other_num_A : ℕ),
      other_num_A ≥ (total_prizes - other_num_A) / 3 →
      total_cost other_num_A ≥ total_cost num_A :=
sorry

end NUMINAMATH_CALUDE_optimal_plan_l1667_166751


namespace NUMINAMATH_CALUDE_hyperbola_circle_intersection_sum_of_squares_l1667_166732

/-- A hyperbola centered at the origin -/
structure Hyperbola where
  a : ℝ
  equation : ∀ (x y : ℝ), x^2 - y^2 = a^2

/-- A circle with center at the origin -/
structure Circle where
  r : ℝ
  equation : ∀ (x y : ℝ), x^2 + y^2 = r^2

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the origin -/
def distance_from_origin (p : Point) : ℝ := p.x^2 + p.y^2

theorem hyperbola_circle_intersection_sum_of_squares 
  (h : Hyperbola) (c : Circle) (P Q R S : Point) :
  (P.x^2 - P.y^2 = h.a^2) →
  (Q.x^2 - Q.y^2 = h.a^2) →
  (R.x^2 - R.y^2 = h.a^2) →
  (S.x^2 - S.y^2 = h.a^2) →
  (P.x^2 + P.y^2 = c.r^2) →
  (Q.x^2 + Q.y^2 = c.r^2) →
  (R.x^2 + R.y^2 = c.r^2) →
  (S.x^2 + S.y^2 = c.r^2) →
  distance_from_origin P + distance_from_origin Q + 
  distance_from_origin R + distance_from_origin S = 4 * c.r^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_intersection_sum_of_squares_l1667_166732


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_given_difference_and_larger_l1667_166796

theorem sum_of_numbers_with_given_difference_and_larger (L S : ℤ) : 
  L = 35 → L - S = 15 → L + S = 55 := by sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_given_difference_and_larger_l1667_166796


namespace NUMINAMATH_CALUDE_set_equality_l1667_166746

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l1667_166746


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1667_166767

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1667_166767


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1667_166744

theorem binomial_coefficient_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1667_166744


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l1667_166702

theorem reciprocal_of_repeating_decimal_one_third (x : ℚ) : 
  (∀ n : ℕ, (10 * x - x) * 10^n = 3 * 10^n - 3) → 
  (1 / x = 3) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_one_third_l1667_166702


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l1667_166737

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 5

/-- The total number of cats sold during the sale -/
def cats_sold : ℕ := 10

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is 13 -/
theorem pet_store_siamese_cats :
  initial_siamese_cats = 13 ∧
  initial_siamese_cats + initial_house_cats = cats_sold + cats_remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l1667_166737


namespace NUMINAMATH_CALUDE_division_of_decimals_l1667_166760

theorem division_of_decimals : (0.36 : ℝ) / (0.004 : ℝ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1667_166760


namespace NUMINAMATH_CALUDE_simplify_expression_l1667_166766

theorem simplify_expression (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ b) :
  (1 - a) + (1 - b) = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1667_166766


namespace NUMINAMATH_CALUDE_base_problem_l1667_166717

theorem base_problem (b : ℕ) : (3 * b + 1)^2 = b^3 + 2 * b + 1 → b = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_problem_l1667_166717


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1667_166783

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 20 + (5 : ℚ) / 200 + (7 : ℚ) / 2000 = (1785 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1667_166783


namespace NUMINAMATH_CALUDE_problem_solution_l1667_166772

theorem problem_solution (p_A p_B : ℝ) (h1 : p_A = 0.4) (h2 : p_B = 0.5) 
  (h3 : 0 ≤ p_A ∧ p_A ≤ 1) (h4 : 0 ≤ p_B ∧ p_B ≤ 1) :
  1 - (1 - p_A) * (1 - p_B) = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1667_166772


namespace NUMINAMATH_CALUDE_initial_puppies_count_l1667_166779

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℚ := 8.5

/-- The number of puppies Alyssa kept -/
def puppies_kept : ℚ := 12.5

/-- The total number of puppies Alyssa had initially -/
def total_puppies : ℚ := puppies_given_away + puppies_kept

theorem initial_puppies_count : total_puppies = 21 := by
  sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l1667_166779


namespace NUMINAMATH_CALUDE_cube_edge_length_l1667_166728

theorem cube_edge_length (V : ℝ) (h : V = 32 / 3 * Real.pi) :
  ∃ (a : ℝ), a > 0 ∧ a = 4 * Real.sqrt 3 / 3 ∧
  V = 4 / 3 * Real.pi * (3 * a^2 / 4) ^ (3/2) :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1667_166728


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1667_166725

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group of 30 players, where each player plays every other player exactly once,
    and each game involves two players, the total number of games played is 435. --/
theorem chess_tournament_games :
  num_games 30 = 435 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1667_166725


namespace NUMINAMATH_CALUDE_one_root_quadratic_l1667_166794

theorem one_root_quadratic (k : ℝ) : 
  (∃! x : ℝ, k * x^2 - 8 * x + 16 = 0) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_root_quadratic_l1667_166794


namespace NUMINAMATH_CALUDE_problem_classification_l1667_166705

-- Define a type for problems
inductive Problem
  | EquilateralTrianglePerimeter
  | ArithmeticMean
  | SmallerOfTwo
  | PiecewiseFunction

-- Define a function to determine if a problem requires a conditional statement
def requiresConditionalStatement (p : Problem) : Prop :=
  match p with
  | Problem.EquilateralTrianglePerimeter => False
  | Problem.ArithmeticMean => False
  | Problem.SmallerOfTwo => True
  | Problem.PiecewiseFunction => True

-- Theorem statement
theorem problem_classification :
  (¬ requiresConditionalStatement Problem.EquilateralTrianglePerimeter) ∧
  (¬ requiresConditionalStatement Problem.ArithmeticMean) ∧
  (requiresConditionalStatement Problem.SmallerOfTwo) ∧
  (requiresConditionalStatement Problem.PiecewiseFunction) :=
by sorry

end NUMINAMATH_CALUDE_problem_classification_l1667_166705


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_factorial_l1667_166799

theorem sum_gcd_lcm_factorial : 
  Nat.gcd 48 180 + Nat.lcm 48 180 + Nat.factorial 4 = 756 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_factorial_l1667_166799


namespace NUMINAMATH_CALUDE_crazy_silly_school_movies_l1667_166730

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 11

/-- The difference between the number of movies and books -/
def movie_book_difference : ℕ := 6

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := num_books + movie_book_difference

theorem crazy_silly_school_movies :
  num_movies = 17 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_movies_l1667_166730


namespace NUMINAMATH_CALUDE_average_score_is_68_l1667_166711

/-- Represents a score and the number of students who received it -/
structure ScoreData where
  score : ℕ
  count : ℕ

/-- Calculates the average score given a list of ScoreData -/
def averageScore (data : List ScoreData) : ℚ :=
  let totalStudents := data.map (·.count) |>.sum
  let weightedSum := data.map (fun sd => sd.score * sd.count) |>.sum
  weightedSum / totalStudents

/-- The given score data from Mrs. Thompson's test -/
def testScores : List ScoreData := [
  ⟨95, 10⟩,
  ⟨85, 15⟩,
  ⟨75, 20⟩,
  ⟨65, 25⟩,
  ⟨55, 15⟩,
  ⟨45, 10⟩,
  ⟨35, 5⟩
]

theorem average_score_is_68 :
  averageScore testScores = 68 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_68_l1667_166711


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_5_l1667_166709

/-- Given sets A and B, prove that if B is a subset of A, then m ≤ 5 -/
theorem subset_implies_m_leq_5 (m : ℝ) : 
  let A : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 2}
  B ⊆ A → m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_5_l1667_166709


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l1667_166763

theorem difference_of_squares_example : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l1667_166763


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l1667_166708

noncomputable def triangle_side_b (a : ℝ) (A B : ℝ) : ℝ :=
  2 * Real.sqrt 6

theorem triangle_side_b_value (a : ℝ) (A B : ℝ) 
  (h1 : a = 3)
  (h2 : B = 2 * A)
  (h3 : Real.cos A = Real.sqrt 6 / 3) :
  triangle_side_b a A B = 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l1667_166708


namespace NUMINAMATH_CALUDE_circle_center_transformation_l1667_166718

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given distance -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem stating the transformation of the circle's center -/
theorem circle_center_transformation :
  let original_center : ℝ × ℝ := (-3, 4)
  let reflected_center := reflect_x original_center
  let final_center := translate_right reflected_center 5
  final_center = (2, -4) := by sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l1667_166718


namespace NUMINAMATH_CALUDE_geometric_progression_with_conditions_l1667_166764

/-- A geometric progression of four terms satisfying specific conditions -/
theorem geometric_progression_with_conditions :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    -- The sequence forms a geometric progression
    (∃ (q : ℝ), b₂ = b₁ * q ∧ b₃ = b₁ * q^2 ∧ b₄ = b₁ * q^3) ∧
    -- The third term is 9 greater than the first term
    b₃ - b₁ = 9 ∧
    -- The second term is 18 greater than the fourth term
    b₂ - b₄ = 18 ∧
    -- The sequence is (3, -6, 12, -24)
    b₁ = 3 ∧ b₂ = -6 ∧ b₃ = 12 ∧ b₄ = -24 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_with_conditions_l1667_166764


namespace NUMINAMATH_CALUDE_number_difference_theorem_l1667_166733

theorem number_difference_theorem (x : ℝ) : x - (3 / 5) * x = 64 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_theorem_l1667_166733


namespace NUMINAMATH_CALUDE_smaller_number_is_24_l1667_166776

theorem smaller_number_is_24 (x y : ℝ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) :
  min x y = 24 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_is_24_l1667_166776


namespace NUMINAMATH_CALUDE_complex_number_equality_l1667_166747

theorem complex_number_equality : (7 : ℂ) - 3*I - 3*(2 - 5*I) + 4*I = 1 + 16*I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l1667_166747


namespace NUMINAMATH_CALUDE_modular_arithmetic_properties_l1667_166738

theorem modular_arithmetic_properties (a b c d m : ℤ) 
  (h1 : a ≡ b [ZMOD m]) 
  (h2 : c ≡ d [ZMOD m]) : 
  (a + c ≡ b + d [ZMOD m]) ∧ (a * c ≡ b * d [ZMOD m]) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_properties_l1667_166738


namespace NUMINAMATH_CALUDE_crypto_deg_is_69_l1667_166771

/-- Represents the digits in the cryptographer's encoding -/
inductive CryptoDigit
| A | B | C | D | E | F | G

/-- Converts a CryptoDigit to its corresponding base 7 value -/
def cryptoToBase7 : CryptoDigit → Fin 7
| CryptoDigit.A => 0
| CryptoDigit.B => 1
| CryptoDigit.D => 3
| CryptoDigit.E => 2
| CryptoDigit.F => 5
| CryptoDigit.G => 6
| _ => 0  -- C is not used in this problem, so we assign it 0

/-- Represents a three-digit number in the cryptographer's encoding -/
structure CryptoNumber where
  hundreds : CryptoDigit
  tens : CryptoDigit
  ones : CryptoDigit

/-- Converts a CryptoNumber to its base 10 value -/
def cryptoToBase10 (n : CryptoNumber) : Nat :=
  (cryptoToBase7 n.hundreds).val * 49 +
  (cryptoToBase7 n.tens).val * 7 +
  (cryptoToBase7 n.ones).val

/-- The main theorem to prove -/
theorem crypto_deg_is_69 :
  let deg : CryptoNumber := ⟨CryptoDigit.D, CryptoDigit.E, CryptoDigit.G⟩
  cryptoToBase10 deg = 69 := by sorry

end NUMINAMATH_CALUDE_crypto_deg_is_69_l1667_166771


namespace NUMINAMATH_CALUDE_line_direction_vector_l1667_166753

/-- Given a line passing through two points and a direction vector, prove the value of b -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-3, 6) → p2 = (2, -1) → 
  (∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) →
  b = 5/7 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l1667_166753


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1667_166782

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I) / z = I) : z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1667_166782


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l1667_166775

/-- For the quadratic x^2 - 24x + 60, when written as (x+b)^2 + c, b+c equals -96 -/
theorem quadratic_form_sum (b c : ℝ) : 
  (∀ x, x^2 - 24*x + 60 = (x+b)^2 + c) → b + c = -96 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l1667_166775


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1667_166723

/-- Given an incident light ray and a reflecting line, find the equation of the reflected ray. -/
theorem reflected_ray_equation (x y : ℝ) : 
  (∃ (x₀ y₀ : ℝ), y₀ = 2 * x₀ + 1) → -- Incident ray equation
  (∃ (x₁ y₁ : ℝ), y₁ = x₁) →        -- Reflecting line equation
  (x - 2 * y - 1 = 0)               -- Reflected ray equation
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1667_166723


namespace NUMINAMATH_CALUDE_annas_cupcake_earnings_l1667_166784

/-- Calculates Anna's earnings from selling cupcakes -/
def annas_earnings (num_trays : ℕ) (cupcakes_per_tray : ℕ) (price_per_cupcake : ℚ) (sold_fraction : ℚ) : ℚ :=
  (num_trays * cupcakes_per_tray : ℚ) * sold_fraction * price_per_cupcake

theorem annas_cupcake_earnings :
  annas_earnings 10 30 (5/2) (7/10) = 525 := by
  sorry

end NUMINAMATH_CALUDE_annas_cupcake_earnings_l1667_166784


namespace NUMINAMATH_CALUDE_james_payment_l1667_166774

theorem james_payment (adoption_fee : ℝ) (friend_percentage : ℝ) (james_payment : ℝ) : 
  adoption_fee = 200 →
  friend_percentage = 0.25 →
  james_payment = adoption_fee - (adoption_fee * friend_percentage) →
  james_payment = 150 := by
sorry

end NUMINAMATH_CALUDE_james_payment_l1667_166774


namespace NUMINAMATH_CALUDE_a_upper_bound_l1667_166758

theorem a_upper_bound (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 3, 2*x > x^2 + a) → a < -8 := by
  sorry

end NUMINAMATH_CALUDE_a_upper_bound_l1667_166758


namespace NUMINAMATH_CALUDE_fish_count_proof_l1667_166769

/-- The number of fish Kendra caught -/
def kendras_catch : ℕ := 30

/-- The number of fish Ken caught -/
def kens_catch : ℕ := 2 * kendras_catch

/-- The number of fish Ken released -/
def kens_released : ℕ := 3

/-- The number of fish Ken brought home -/
def kens_brought_home : ℕ := kens_catch - kens_released

/-- The number of fish Kendra brought home (same as caught) -/
def kendras_brought_home : ℕ := kendras_catch

/-- The total number of fish brought home by Ken and Kendra -/
def total_brought_home : ℕ := kens_brought_home + kendras_brought_home

theorem fish_count_proof : total_brought_home = 87 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_proof_l1667_166769


namespace NUMINAMATH_CALUDE_eighth_term_is_128_l1667_166714

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n
  second_term : a 2 = 2
  product_condition : a 3 * a 4 = 32

/-- The 8th term of the geometric sequence is 128 -/
theorem eighth_term_is_128 (seq : GeometricSequence) : seq.a 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_is_128_l1667_166714


namespace NUMINAMATH_CALUDE_parabola_c_value_l1667_166795

/-- A parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 2 = -4 →  -- vertex condition
  p.x_coord 4 = -2 →  -- point (-2, 4) condition
  p.x_coord 0 = -2 →  -- point (-2, 0) condition
  p.c = -2 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1667_166795


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l1667_166727

/-- Given a geometric sequence {a_n} with common ratio q > 1,
    if a_5 - a_1 = 15 and a_4 - a_2 = 6, then a_3 = 4 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence definition
  q > 1 →  -- common ratio greater than 1
  a 5 - a 1 = 15 →  -- condition on a_5 and a_1
  a 4 - a 2 = 6 →  -- condition on a_4 and a_2
  a 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l1667_166727


namespace NUMINAMATH_CALUDE_juice_theorem_l1667_166704

def juice_problem (sam_initial ben_initial sam_consumed ben_consumed sam_received : ℚ) : Prop :=
  let sam_final := sam_consumed + sam_received
  let ben_final := ben_consumed - sam_received
  sam_initial = 12 ∧
  ben_initial = sam_initial + 8 ∧
  sam_consumed = 2 / 3 * sam_initial ∧
  ben_consumed = 2 / 3 * ben_initial ∧
  sam_received = (1 / 2 * (ben_initial - ben_consumed)) + 1 ∧
  sam_final = ben_final ∧
  sam_initial + ben_initial = 32

theorem juice_theorem :
  ∃ (sam_initial ben_initial sam_consumed ben_consumed sam_received : ℚ),
    juice_problem sam_initial ben_initial sam_consumed ben_consumed sam_received :=
by
  sorry

#check juice_theorem

end NUMINAMATH_CALUDE_juice_theorem_l1667_166704


namespace NUMINAMATH_CALUDE_smallest_value_of_3a_plus_2_l1667_166781

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ x, (∃ y, 8 * y^2 + 7 * y + 6 = 5 ∧ 3 * y + 2 = x) → min ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_3a_plus_2_l1667_166781


namespace NUMINAMATH_CALUDE_calculate_expression_l1667_166707

theorem calculate_expression : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 + 9000000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1667_166707


namespace NUMINAMATH_CALUDE_snow_probability_first_week_l1667_166761

def probability_of_snow (days : ℕ) (daily_prob : ℚ) : ℚ :=
  1 - (1 - daily_prob) ^ days

theorem snow_probability_first_week :
  let prob_first_four := probability_of_snow 4 (1/4)
  let prob_next_three := probability_of_snow 3 (1/3)
  1 - (1 - prob_first_four) * (1 - prob_next_three) = 29/32 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_first_week_l1667_166761


namespace NUMINAMATH_CALUDE_cubic_root_relation_l1667_166749

theorem cubic_root_relation (a b c d : ℝ) (h : a ≠ 0) :
  (∃ u v w : ℝ, a * u^3 + b * u^2 + c * u + d = 0 ∧
               a * v^3 + b * v^2 + c * v + d = 0 ∧
               a * w^3 + b * w^2 + c * w + d = 0 ∧
               u + v = u * v) →
  (c + d) * (b + c + d) = a * d :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_relation_l1667_166749


namespace NUMINAMATH_CALUDE_max_peak_consumption_theorem_l1667_166792

/-- Represents the electricity pricing and consumption parameters for a household. -/
structure ElectricityParams where
  originalPrice : ℝ
  peakPrice : ℝ
  offPeakPrice : ℝ
  totalConsumption : ℝ
  savingsPercentage : ℝ

/-- Calculates the maximum peak hour consumption given electricity parameters. -/
def maxPeakConsumption (params : ElectricityParams) : ℝ := by
  sorry

/-- Theorem stating the maximum peak hour consumption for the given scenario. -/
theorem max_peak_consumption_theorem (params : ElectricityParams) 
  (h1 : params.originalPrice = 0.52)
  (h2 : params.peakPrice = 0.55)
  (h3 : params.offPeakPrice = 0.35)
  (h4 : params.totalConsumption = 200)
  (h5 : params.savingsPercentage = 0.1) :
  maxPeakConsumption params = 118 := by
  sorry

end NUMINAMATH_CALUDE_max_peak_consumption_theorem_l1667_166792


namespace NUMINAMATH_CALUDE_tan_45_plus_half_inv_plus_abs_neg_two_equals_five_l1667_166736

theorem tan_45_plus_half_inv_plus_abs_neg_two_equals_five :
  Real.tan (π / 4) + (1 / 2)⁻¹ + |(-2)| = 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_plus_half_inv_plus_abs_neg_two_equals_five_l1667_166736


namespace NUMINAMATH_CALUDE_honey_servings_calculation_l1667_166715

/-- The number of servings in a container of honey -/
def number_of_servings (container_volume : ℚ) (serving_size : ℚ) : ℚ :=
  container_volume / serving_size

/-- Proof that a container with 37 2/3 tablespoons of honey contains 25 1/9 servings when each serving is 1 1/2 tablespoons -/
theorem honey_servings_calculation :
  let container_volume : ℚ := 113/3  -- 37 2/3 as an improper fraction
  let serving_size : ℚ := 3/2        -- 1 1/2 as an improper fraction
  number_of_servings container_volume serving_size = 226/9
  := by sorry

end NUMINAMATH_CALUDE_honey_servings_calculation_l1667_166715


namespace NUMINAMATH_CALUDE_box_office_scientific_notation_l1667_166726

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

end NUMINAMATH_CALUDE_box_office_scientific_notation_l1667_166726


namespace NUMINAMATH_CALUDE_factorization_of_expression_l1667_166797

theorem factorization_of_expression (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (b^2 * c^3) := by sorry

end NUMINAMATH_CALUDE_factorization_of_expression_l1667_166797


namespace NUMINAMATH_CALUDE_min_coefficient_value_l1667_166786

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 30 * x^2 + box * x + 30) →
  a ≤ 15 →
  b ≤ 15 →
  a * b = 30 →
  box = a^2 + b^2 →
  61 ≤ box :=
by sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l1667_166786


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1667_166743

theorem power_of_two_equality : (2^8)^5 = 2^8 * 2^32 := by sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1667_166743


namespace NUMINAMATH_CALUDE_inequality_proof_l1667_166789

theorem inequality_proof (x y z : ℝ) 
  (h1 : -2 ≤ x ∧ x ≤ 2) 
  (h2 : -2 ≤ y ∧ y ≤ 2) 
  (h3 : -2 ≤ z ∧ z ≤ 2) 
  (h4 : x^2 + y^2 + z^2 + x*y*z = 4) : 
  z * (x*z + y*z + y) / (x*y + y^2 + z^2 + 1) ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1667_166789


namespace NUMINAMATH_CALUDE_elizabeth_granola_profit_l1667_166752

/-- Calculate Elizabeth's net profit from selling granola bags --/
theorem elizabeth_granola_profit :
  let ingredient_cost_per_bag : ℚ := 3
  let total_bags : ℕ := 20
  let full_price : ℚ := 6
  let full_price_sales : ℕ := 15
  let discounted_price : ℚ := 4
  let discounted_sales : ℕ := 5

  let total_cost : ℚ := ingredient_cost_per_bag * total_bags
  let full_price_revenue : ℚ := full_price * full_price_sales
  let discounted_revenue : ℚ := discounted_price * discounted_sales
  let total_revenue : ℚ := full_price_revenue + discounted_revenue
  let net_profit : ℚ := total_revenue - total_cost

  net_profit = 50 := by sorry

end NUMINAMATH_CALUDE_elizabeth_granola_profit_l1667_166752


namespace NUMINAMATH_CALUDE_susan_weather_probability_l1667_166731

/-- The probability of having exactly 1 or 2 sunny days in a 3-day period -/
def prob_1_or_2_sunny (p : ℚ) : ℚ :=
  (3 : ℚ) * p * (1 - p)^2 + (3 : ℚ) * p^2 * (1 - p)

/-- The theorem stating the probability of Susan getting her desired weather -/
theorem susan_weather_probability :
  prob_1_or_2_sunny (2/5) = 18/25 := by
  sorry


end NUMINAMATH_CALUDE_susan_weather_probability_l1667_166731


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1667_166724

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1667_166724


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1667_166750

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of the r-th term in the expansion of (1+2x)^7 -/
def coefficient (r : ℕ) : ℕ := binomial 7 r * 2^r

theorem binomial_expansion_properties :
  (coefficient 2 = binomial 7 2 * 2^2) ∧
  (coefficient 2 = 24) := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1667_166750


namespace NUMINAMATH_CALUDE_function_composition_equality_l1667_166745

/-- Given a function f(x) = ax^2 - √3 where a > 0, prove that f(f(√3)) = -√3 implies a = √3/3 -/
theorem function_composition_equality (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - Real.sqrt 3
  f (f (Real.sqrt 3)) = -Real.sqrt 3 → a = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1667_166745


namespace NUMINAMATH_CALUDE_tricycle_wheels_count_l1667_166713

theorem tricycle_wheels_count :
  ∀ (tricycle_wheels : ℕ),
    3 * 2 + 4 * tricycle_wheels + 7 * 1 = 25 →
    tricycle_wheels = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tricycle_wheels_count_l1667_166713


namespace NUMINAMATH_CALUDE_jane_brown_sheets_l1667_166754

/-- The number of old, brown sheets of drawing paper Jane has -/
def brown_sheets (total : ℕ) (yellow : ℕ) : ℕ := total - yellow

/-- Proof that Jane has 28 old, brown sheets of drawing paper -/
theorem jane_brown_sheets : brown_sheets 55 27 = 28 := by
  sorry

end NUMINAMATH_CALUDE_jane_brown_sheets_l1667_166754


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1667_166762

theorem cubic_root_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 3*p - 1 = 0) ∧ 
  (q^3 - 2*q^2 + 3*q - 1 = 0) ∧ 
  (r^3 - 2*r^2 + 3*r - 1 = 0) →
  p^3 + q^3 + r^3 = -7 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1667_166762


namespace NUMINAMATH_CALUDE_positive_derivative_implies_increasing_exists_increasing_with_nonpositive_derivative_l1667_166703

open Set
open Function

-- Define a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Part 1: If f'(x) > 0 for all x, then f is monotonically increasing
theorem positive_derivative_implies_increasing :
  (∀ x, deriv f x > 0) → MonotonicallyIncreasing f :=
sorry

-- Part 2: There exists a monotonically increasing function with f'(x) ≤ 0 for some x
theorem exists_increasing_with_nonpositive_derivative :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ MonotonicallyIncreasing f ∧ ∃ x, deriv f x ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_positive_derivative_implies_increasing_exists_increasing_with_nonpositive_derivative_l1667_166703


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1667_166759

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_number_satisfying_conditions : 
  (∀ n : ℕ, n < 70 → ¬(is_multiple_of n 7 ∧ is_multiple_of n 5 ∧ is_prime (n + 9))) ∧ 
  (is_multiple_of 70 7 ∧ is_multiple_of 70 5 ∧ is_prime (70 + 9)) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l1667_166759


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1667_166788

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 1| = |x - 3| :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1667_166788


namespace NUMINAMATH_CALUDE_smallest_x_value_l1667_166700

theorem smallest_x_value (x y z : ℝ) 
  (sum_condition : x + y + z = 6)
  (product_condition : x * y + x * z + y * z = 10) :
  ∀ x' : ℝ, (∃ y' z' : ℝ, x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10) → x' ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1667_166700


namespace NUMINAMATH_CALUDE_boxes_per_carton_l1667_166755

/-- Proves that the number of boxes in each carton is 1 -/
theorem boxes_per_carton (c : ℕ) : c > 0 → ∃ b : ℕ, b > 0 ∧ b * c = 1 := by
  sorry

#check boxes_per_carton

end NUMINAMATH_CALUDE_boxes_per_carton_l1667_166755
