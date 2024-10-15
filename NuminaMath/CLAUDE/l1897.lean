import Mathlib

namespace NUMINAMATH_CALUDE_circle_area_ratio_l1897_189777

theorem circle_area_ratio (Q P R : ℝ) (hP : P = 0.5 * Q) (hR : R = 0.75 * Q) :
  (π * (R / 2)^2) / (π * (Q / 2)^2) = 0.140625 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1897_189777


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1897_189782

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l1897_189782


namespace NUMINAMATH_CALUDE_intersection_range_length_AB_l1897_189710

-- Define the hyperbola C: x^2 - y^2 = 1
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the line l: y = kx + 1
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for two distinct intersection points
def has_two_intersections (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂

-- Theorem for the range of k
theorem intersection_range :
  ∀ k : ℝ, has_two_intersections k ↔ 
    (k > -Real.sqrt 2 ∧ k < -1) ∨ 
    (k > -1 ∧ k < 1) ∨ 
    (k > 1 ∧ k < Real.sqrt 2) :=
sorry

-- Define the condition for the midpoint x-coordinate
def midpoint_x_is_sqrt2 (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    (x₁ + x₂) / 2 = Real.sqrt 2

-- Theorem for the length of AB
theorem length_AB (k : ℝ) :
  midpoint_x_is_sqrt2 k → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_length_AB_l1897_189710


namespace NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l1897_189768

theorem min_value_x2_minus_xy_plus_y2 :
  ∀ x y : ℝ, x^2 - x*y + y^2 ≥ 0 ∧ (x^2 - x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_min_value_x2_minus_xy_plus_y2_l1897_189768


namespace NUMINAMATH_CALUDE_number_of_girls_l1897_189760

/-- Given a group of kids with boys and girls, prove the number of girls. -/
theorem number_of_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : total = 9 ∧ boys = 6 → girls = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_l1897_189760


namespace NUMINAMATH_CALUDE_equation_solution_l1897_189730

theorem equation_solution : 
  ∃ x : ℚ, (5 * x - 2) / (6 * x - 6) = 3 / 4 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1897_189730


namespace NUMINAMATH_CALUDE_S_is_infinite_l1897_189744

/-- A point in the xy-plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- The set of points satisfying the given conditions -/
def S : Set RationalPoint :=
  {p : RationalPoint | p.x > 0 ∧ p.y > 0 ∧ p.x * p.y ≤ 12}

/-- Theorem stating that the set S is infinite -/
theorem S_is_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_S_is_infinite_l1897_189744


namespace NUMINAMATH_CALUDE_equal_interior_angles_decagon_l1897_189707

/-- The measure of an interior angle in a regular decagon -/
def regular_decagon_angle : ℝ := 144

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: In a decagon where all interior angles are equal, each interior angle measures 144° -/
theorem equal_interior_angles_decagon : 
  ∀ (angles : Fin decagon_sides → ℝ), 
    (∀ (i j : Fin decagon_sides), angles i = angles j) →
    (∀ (i : Fin decagon_sides), angles i = regular_decagon_angle) :=
by sorry

end NUMINAMATH_CALUDE_equal_interior_angles_decagon_l1897_189707


namespace NUMINAMATH_CALUDE_solution_count_l1897_189743

theorem solution_count (S : Finset ℝ) (p : ℝ) : 
  S.card = 12 → p = 1/6 → ∃ n : ℕ, n = 2 ∧ n = (S.card : ℝ) * p := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l1897_189743


namespace NUMINAMATH_CALUDE_michael_matchsticks_l1897_189722

theorem michael_matchsticks (total : ℕ) (houses : ℕ) (sticks_per_house : ℕ) : 
  houses = 30 →
  sticks_per_house = 10 →
  houses * sticks_per_house = total / 2 →
  total = 600 := by
  sorry

end NUMINAMATH_CALUDE_michael_matchsticks_l1897_189722


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l1897_189757

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l1897_189757


namespace NUMINAMATH_CALUDE_cube_edge_length_l1897_189758

theorem cube_edge_length (V : ℝ) (h : V = 32 / 3 * Real.pi) :
  ∃ (a : ℝ), a > 0 ∧ a = 4 * Real.sqrt 3 / 3 ∧
  V = 4 / 3 * Real.pi * (3 * a^2 / 4) ^ (3/2) :=
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1897_189758


namespace NUMINAMATH_CALUDE_value_of_x_l1897_189739

theorem value_of_x (w y z : ℚ) (h1 : w = 45) (h2 : z = 2 * w) (h3 : y = (1 / 6) * z) : (1 / 3) * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1897_189739


namespace NUMINAMATH_CALUDE_max_value_of_largest_integer_l1897_189719

theorem max_value_of_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 60 →
  e.val - a.val = 10 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  e.val ≤ 290 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_largest_integer_l1897_189719


namespace NUMINAMATH_CALUDE_larger_number_problem_l1897_189745

theorem larger_number_problem (L S : ℕ) (h1 : L > S) (h2 : L - S = 1365) (h3 : L = 6 * S + 15) : L = 1635 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1897_189745


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_optimal_price_satisfies_conditions_l1897_189737

/-- Represents the daily profit function for a merchant's goods -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- Represents the optimal selling price that maximizes daily profit -/
def optimal_price : ℝ := 14

theorem optimal_price_maximizes_profit :
  ∀ (x : ℝ), x ≠ optimal_price → profit_function x < profit_function optimal_price :=
by sorry

/-- Verifies that the optimal price satisfies the given conditions -/
theorem optimal_price_satisfies_conditions :
  let initial_price : ℝ := 10
  let initial_sales : ℝ := 100
  let cost_per_item : ℝ := 8
  let price_increase : ℝ := optimal_price - initial_price
  let sales_decrease : ℝ := 10 * price_increase
  (initial_sales - sales_decrease) * (optimal_price - cost_per_item) = profit_function optimal_price :=
by sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_optimal_price_satisfies_conditions_l1897_189737


namespace NUMINAMATH_CALUDE_correct_operation_l1897_189708

theorem correct_operation (a b : ℝ) : (a - b) * (2 * a + 2 * b) = 2 * a^2 - 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1897_189708


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l1897_189762

theorem inscribed_circle_radius_rhombus (d₁ d₂ : ℝ) (h₁ : d₁ = 8) (h₂ : d₂ = 30) :
  let a := Real.sqrt ((d₁/2)^2 + (d₂/2)^2)
  let r := (d₁ * d₂) / (8 * a)
  r = 30 / Real.sqrt 241 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l1897_189762


namespace NUMINAMATH_CALUDE_R_is_top_right_l1897_189711

/-- Represents a rectangle with integer labels at its corners -/
structure Rectangle where
  a : Int  -- left-top
  b : Int  -- right-top
  c : Int  -- right-bottom
  d : Int  -- left-bottom

/-- The set of four rectangles -/
def rectangles : Finset Rectangle := sorry

/-- P is one of the rectangles -/
def P : Rectangle := ⟨5, 1, 8, 2⟩

/-- Q is one of the rectangles -/
def Q : Rectangle := ⟨2, 8, 10, 4⟩

/-- R is one of the rectangles -/
def R : Rectangle := ⟨4, 5, 1, 7⟩

/-- S is one of the rectangles -/
def S : Rectangle := ⟨8, 3, 7, 5⟩

/-- The rectangles are arranged in a 2x2 matrix -/
def isArranged2x2 (rects : Finset Rectangle) : Prop := sorry

/-- A rectangle is at the top-right position -/
def isTopRight (rect : Rectangle) (rects : Finset Rectangle) : Prop := sorry

/-- Main theorem: R is at the top-right position -/
theorem R_is_top_right : isTopRight R rectangles := by sorry

end NUMINAMATH_CALUDE_R_is_top_right_l1897_189711


namespace NUMINAMATH_CALUDE_fraction_addition_l1897_189748

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1897_189748


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l1897_189789

/-- A six-digit number is between 100000 and 999999 -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- Function to reduce the first digit of a number by 3 and append 3 at the end -/
def transform (n : ℕ) : ℕ := (n - 300000) * 10 + 3

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_six_digit n ∧ 3 * n = transform n ∧ n = 428571 := by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l1897_189789


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1897_189718

theorem complex_modulus_equality (n : ℝ) :
  n > 0 → (Complex.abs (4 + n * Complex.I) = 4 * Real.sqrt 13 ↔ n = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1897_189718


namespace NUMINAMATH_CALUDE_tomato_field_area_l1897_189767

theorem tomato_field_area (length : ℝ) (width : ℝ) (tomato_area : ℝ) : 
  length = 3.6 →
  width = 2.5 * length →
  tomato_area = (length * width) / 2 →
  tomato_area = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_field_area_l1897_189767


namespace NUMINAMATH_CALUDE_smallest_triangle_angle_function_range_l1897_189749

theorem smallest_triangle_angle_function_range :
  ∀ x : Real,
  0 < x → x ≤ Real.pi / 3 →
  let y := (Real.sin x * Real.cos x + 1) / (Real.sin x + Real.cos x)
  ∃ (a b : Real), a = 3/2 ∧ b = 3 * Real.sqrt 2 / 4 ∧
  (∀ z, y = z → a < z ∧ z ≤ b) ∧
  (∀ ε > 0, ∃ z, y = z ∧ z < a + ε) ∧
  (∃ z, y = z ∧ z = b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_triangle_angle_function_range_l1897_189749


namespace NUMINAMATH_CALUDE_september_march_ratio_is_two_to_one_l1897_189792

/-- Vacation policy and Andrew's work record --/
structure VacationRecord where
  workRatio : ℕ  -- Number of work days required for 1 vacation day
  workDays : ℕ   -- Number of days worked
  marchDays : ℕ  -- Vacation days taken in March
  remainingDays : ℕ  -- Remaining vacation days

/-- Calculate the ratio of September vacation days to March vacation days --/
def septemberToMarchRatio (record : VacationRecord) : ℚ :=
  let totalVacationDays := record.workDays / record.workRatio
  let septemberDays := totalVacationDays - record.remainingDays - record.marchDays
  septemberDays / record.marchDays

/-- Theorem stating the ratio of September to March vacation days is 2:1 --/
theorem september_march_ratio_is_two_to_one 
  (record : VacationRecord)
  (h1 : record.workRatio = 10)
  (h2 : record.workDays = 300)
  (h3 : record.marchDays = 5)
  (h4 : record.remainingDays = 15) :
  septemberToMarchRatio record = 2 := by
  sorry

#eval septemberToMarchRatio ⟨10, 300, 5, 15⟩

end NUMINAMATH_CALUDE_september_march_ratio_is_two_to_one_l1897_189792


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1897_189765

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 6 = 3) ∧ 
  (n % 8 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m ≥ n) ∧
  (n = 21) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1897_189765


namespace NUMINAMATH_CALUDE_base_equation_solution_l1897_189774

/-- Represents a number in a given base -/
def toBase (n : ℕ) (base : ℕ) : ℕ → ℕ 
| 0 => 0
| (d+1) => (toBase n base d) * base + n % base

/-- The main theorem -/
theorem base_equation_solution (A B : ℕ) (h1 : B = A + 2) 
  (h2 : toBase 216 A 3 + toBase 52 B 2 = toBase 75 (A + B + 1) 2) : 
  A + B + 1 = 15 := by
  sorry

#eval toBase 216 6 3  -- Should output 90
#eval toBase 52 8 2   -- Should output 42
#eval toBase 75 15 2  -- Should output 132

end NUMINAMATH_CALUDE_base_equation_solution_l1897_189774


namespace NUMINAMATH_CALUDE_rectangle_area_l1897_189703

-- Define the rectangle ABCD
structure Rectangle :=
  (A B C D : ℝ × ℝ)

-- Define the diagonal BD
def diagonal (rect : Rectangle) : ℝ × ℝ := (rect.B.1 - rect.D.1, rect.B.2 - rect.D.2)

-- Define points E and F on the diagonal
structure PerpendicularPoints (rect : Rectangle) :=
  (E F : ℝ × ℝ)

-- Define the perpendicularity condition
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- State the theorem
theorem rectangle_area (rect : Rectangle) (perp : PerpendicularPoints rect) :
  isPerpendicular (rect.A.1 - perp.E.1, rect.A.2 - perp.E.2) (diagonal rect) →
  isPerpendicular (rect.C.1 - perp.F.1, rect.C.2 - perp.F.2) (diagonal rect) →
  (perp.E.1 - rect.B.1)^2 + (perp.E.2 - rect.B.2)^2 = 1 →
  (perp.F.1 - perp.E.1)^2 + (perp.F.2 - perp.E.2)^2 = 4 →
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2) = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1897_189703


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1897_189756

def A : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1897_189756


namespace NUMINAMATH_CALUDE_min_value_theorem_l1897_189738

theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 0) (heq : a + 2*b = 2) :
  ∃ (min_val : ℝ), min_val = 4*(1 + Real.sqrt 2) ∧
  ∀ (x : ℝ), x = 2/(a - 1) + a/b → x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1897_189738


namespace NUMINAMATH_CALUDE_average_score_is_68_l1897_189772

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

end NUMINAMATH_CALUDE_average_score_is_68_l1897_189772


namespace NUMINAMATH_CALUDE_third_year_compound_interest_l1897_189759

/-- Calculates compound interest for a given principal, rate, and number of years -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

theorem third_year_compound_interest (P : ℝ) (r : ℝ) :
  r = 0.06 →
  compoundInterest P r 2 = 1200 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |compoundInterest P r 3 - 1858.03| < ε :=
sorry

end NUMINAMATH_CALUDE_third_year_compound_interest_l1897_189759


namespace NUMINAMATH_CALUDE_smallest_a_for_two_zeros_l1897_189753

/-- The function f(x) = x^2 - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

/-- The function g(x) = (a-2)*x -/
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

/-- The function F(x) = f(x) - g(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

/-- The theorem stating that 3 is the smallest positive integer value of a 
    for which F(x) has exactly two zeros -/
theorem smallest_a_for_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ F 3 x₁ = 0 ∧ F 3 x₂ = 0 ∧
  ∀ (a : ℕ), a < 3 → ¬∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ F (a : ℝ) y₁ = 0 ∧ F (a : ℝ) y₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_two_zeros_l1897_189753


namespace NUMINAMATH_CALUDE_smaller_prime_factor_l1897_189797

theorem smaller_prime_factor : ∃ p : ℕ, 
  Prime p ∧ 
  p > 4002001 ∧ 
  316990099009901 = 4002001 * p ∧
  316990099009901 = 32016000000000001 / 101 := by
sorry

end NUMINAMATH_CALUDE_smaller_prime_factor_l1897_189797


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l1897_189769

theorem complex_magnitude_proof : Complex.abs (8/7 + 3*I) = Real.sqrt 505 / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l1897_189769


namespace NUMINAMATH_CALUDE_vishal_investment_percentage_l1897_189794

/-- Represents the investment amounts in rupees -/
structure Investment where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The given conditions of the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.raghu = 2300 ∧
  i.trishul = 0.9 * i.raghu ∧
  i.vishal + i.trishul + i.raghu = 6647

/-- The theorem stating that Vishal invested 10% more than Trishul -/
theorem vishal_investment_percentage (i : Investment) 
  (h : investment_conditions i) : 
  (i.vishal - i.trishul) / i.trishul = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_vishal_investment_percentage_l1897_189794


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l1897_189726

theorem gas_station_candy_boxes : 
  let chocolate_boxes : ℕ := 2
  let sugar_boxes : ℕ := 5
  let gum_boxes : ℕ := 2
  chocolate_boxes + sugar_boxes + gum_boxes = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l1897_189726


namespace NUMINAMATH_CALUDE_expand_product_l1897_189752

theorem expand_product (x : ℝ) : 3 * (2 * x - 7) * (x + 9) = 6 * x^2 + 33 * x - 189 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1897_189752


namespace NUMINAMATH_CALUDE_volunteer_distribution_l1897_189773

/-- The number of ways to distribute n girls and m boys into two groups,
    where each group must have at least one girl and one boy. -/
def distribution_schemes (n m : ℕ) : ℕ :=
  if n < 2 ∨ m < 2 then 0
  else (Nat.choose n 1 + Nat.choose n 2) * Nat.factorial m

/-- The problem statement -/
theorem volunteer_distribution : distribution_schemes 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l1897_189773


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1897_189705

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 8*m^2 + 17*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1897_189705


namespace NUMINAMATH_CALUDE_pond_length_l1897_189799

theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) : 
  field_length = 48 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 18 →
  Real.sqrt pond_area = 8 := by
sorry

end NUMINAMATH_CALUDE_pond_length_l1897_189799


namespace NUMINAMATH_CALUDE_overall_gain_percentage_l1897_189717

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

end NUMINAMATH_CALUDE_overall_gain_percentage_l1897_189717


namespace NUMINAMATH_CALUDE_clara_alice_pen_ratio_l1897_189706

def alice_pens : ℕ := 60
def alice_age : ℕ := 20
def clara_future_age : ℕ := 61
def years_to_future : ℕ := 5

theorem clara_alice_pen_ratio :
  ∃ (clara_pens : ℕ) (clara_age : ℕ),
    clara_age > alice_age ∧
    clara_age + years_to_future = clara_future_age ∧
    clara_age - alice_age = alice_pens - clara_pens ∧
    clara_pens * 5 = alice_pens * 2 :=
by sorry

end NUMINAMATH_CALUDE_clara_alice_pen_ratio_l1897_189706


namespace NUMINAMATH_CALUDE_n_pow_half_n_eq_eight_l1897_189716

theorem n_pow_half_n_eq_eight (n : ℝ) : n = 2^Real.sqrt 6 → n^(n/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_n_pow_half_n_eq_eight_l1897_189716


namespace NUMINAMATH_CALUDE_weekly_average_rainfall_l1897_189712

/-- Calculates the daily average rainfall for a week given specific conditions. -/
theorem weekly_average_rainfall : 
  let monday_rain : ℝ := 2 + 1
  let tuesday_rain : ℝ := 2 * monday_rain
  let wednesday_rain : ℝ := 0
  let thursday_rain : ℝ := 1
  let friday_rain : ℝ := monday_rain + tuesday_rain + wednesday_rain + thursday_rain
  let total_rainfall : ℝ := monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain
  let days_in_week : ℕ := 7
  total_rainfall / days_in_week = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_weekly_average_rainfall_l1897_189712


namespace NUMINAMATH_CALUDE_sum_of_ages_l1897_189702

/-- Given the age relationships between Paula, Karl, and Jane at different points in time, 
    prove that the sum of their current ages is 63 years. -/
theorem sum_of_ages (P K J : ℚ) : 
  (P - 7 = 4 * (K - 7)) →  -- 7 years ago, Paula was 4 times as old as Karl
  (J - 7 = (P - 7) / 2) →  -- 7 years ago, Jane was half as old as Paula
  (P + 8 = 2 * (K + 8)) →  -- In 8 years, Paula will be twice as old as Karl
  (J + 8 = K + 5) →        -- In 8 years, Jane will be 3 years younger than Karl
  P + K + J = 63 :=        -- The sum of their current ages is 63
by sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1897_189702


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1897_189735

theorem arithmetic_calculation : 80 + 5 * 12 / (180 / 3) = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1897_189735


namespace NUMINAMATH_CALUDE_systematic_sampling_proof_l1897_189742

theorem systematic_sampling_proof (N n : ℕ) (hN : N = 92) (hn : n = 30) :
  let k := N / n
  (k = 3) ∧ (k - 1 = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_proof_l1897_189742


namespace NUMINAMATH_CALUDE_probability_not_snow_l1897_189784

theorem probability_not_snow (p : ℚ) (h : p = 2/5) : 1 - p = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_snow_l1897_189784


namespace NUMINAMATH_CALUDE_derivative_sqrt_at_one_l1897_189751

theorem derivative_sqrt_at_one :
  let f : ℝ → ℝ := λ x => Real.sqrt x
  HasDerivAt f (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_derivative_sqrt_at_one_l1897_189751


namespace NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l1897_189701

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

def consecutive_pair (a b : ℕ) : Prop := b = a + 1

theorem least_integer_with_divisibility_pattern : 
  ∃ (p q : ℕ), 18 ≤ p ∧ p < 25 ∧ consecutive_pair p q ∧
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 ∧ k ≠ p ∧ k ≠ q → is_divisible 659375723440 k) ∧
  ¬(is_divisible 659375723440 p) ∧ ¬(is_divisible 659375723440 q) ∧
  (∀ (n : ℕ), n < 659375723440 → 
    ¬(∃ (r s : ℕ), 18 ≤ r ∧ r < 25 ∧ consecutive_pair r s ∧
    (∀ (k : ℕ), 1 ≤ k ∧ k ≤ 30 ∧ k ≠ r ∧ k ≠ s → is_divisible n k) ∧
    ¬(is_divisible n r) ∧ ¬(is_divisible n s))) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_with_divisibility_pattern_l1897_189701


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1897_189720

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I * 2 + 1) * a + b = Complex.I * 2 → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1897_189720


namespace NUMINAMATH_CALUDE_fruit_selection_problem_l1897_189728

theorem fruit_selection_problem (apple_price orange_price : ℚ)
  (initial_avg_price new_avg_price : ℚ) (oranges_removed : ℕ) :
  apple_price = 40 / 100 →
  orange_price = 60 / 100 →
  initial_avg_price = 54 / 100 →
  new_avg_price = 48 / 100 →
  oranges_removed = 5 →
  ∃ (apples oranges : ℕ),
    (apple_price * apples + orange_price * oranges) / (apples + oranges) = initial_avg_price ∧
    (apple_price * apples + orange_price * (oranges - oranges_removed)) / (apples + oranges - oranges_removed) = new_avg_price ∧
    apples + oranges = 10 :=
by sorry

end NUMINAMATH_CALUDE_fruit_selection_problem_l1897_189728


namespace NUMINAMATH_CALUDE_problem_solution_l1897_189727

/-- The set A as defined in the problem -/
def A : Set ℝ := {x | 12 - 5*x - 2*x^2 > 0}

/-- The set B as defined in the problem -/
def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b ≤ 0}

/-- The theorem statement -/
theorem problem_solution :
  ∃ (a b : ℝ),
    (A ∩ B a b = ∅) ∧
    (A ∪ B a b = Set.Ioo (-4) 8) ∧
    (a = 19/2) ∧
    (b = 12) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1897_189727


namespace NUMINAMATH_CALUDE_batsman_average_after_19th_inning_l1897_189766

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRunsBefore : ℕ
  scoreInLastInning : ℕ
  averageIncrease : ℚ

/-- Calculates the new average of a batsman after their latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRunsBefore + b.scoreInLastInning : ℚ) / b.innings

theorem batsman_average_after_19th_inning 
  (b : Batsman) 
  (h1 : b.innings = 19) 
  (h2 : b.scoreInLastInning = 100) 
  (h3 : b.averageIncrease = 2) :
  newAverage b = 64 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_19th_inning_l1897_189766


namespace NUMINAMATH_CALUDE_rectangle_with_hole_area_l1897_189763

/-- The area of a rectangle with dimensions (2x+14) and (2x+10), minus the area of a rectangular hole
    with dimensions y and x, where (y+1) = (x-2) and x = (2y+3), is equal to 2x^2 + 57x + 131. -/
theorem rectangle_with_hole_area (x y : ℝ) : 
  (y + 1 = x - 2) → 
  (x = 2*y + 3) → 
  (2*x + 14) * (2*x + 10) - y * x = 2*x^2 + 57*x + 131 := by
sorry

end NUMINAMATH_CALUDE_rectangle_with_hole_area_l1897_189763


namespace NUMINAMATH_CALUDE_first_year_payment_is_20_l1897_189732

/-- Represents the payment structure over four years -/
structure PaymentStructure where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  fourth_year : ℕ

/-- Defines the conditions of the payment structure -/
def valid_payment_structure (p : PaymentStructure) : Prop :=
  p.second_year = p.first_year + 2 ∧
  p.third_year = p.second_year + 3 ∧
  p.fourth_year = p.third_year + 4 ∧
  p.first_year + p.second_year + p.third_year + p.fourth_year = 96

/-- Theorem stating that the first year's payment is 20 rupees -/
theorem first_year_payment_is_20 :
  ∀ (p : PaymentStructure), valid_payment_structure p → p.first_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_year_payment_is_20_l1897_189732


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_prime_l1897_189746

theorem unique_solution_cube_difference_prime (x y z : ℕ+) : 
  Nat.Prime y.val ∧ 
  ¬(3 ∣ z.val) ∧ 
  ¬(y.val ∣ z.val) ∧ 
  x.val^3 - y.val^3 = z.val^2 →
  x = 8 ∧ y = 7 ∧ z = 13 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_prime_l1897_189746


namespace NUMINAMATH_CALUDE_production_scaling_l1897_189721

/-- Given that x men working x hours a day for x days produce x^2 articles,
    prove that z men working z hours a day for z days produce z^3/x articles. -/
theorem production_scaling (x z : ℝ) (hx : x > 0) :
  (x * x * x * x^2 = x^3 * x^2) →
  (z * z * z * (z^3 / x) = z^3 * (z^3 / x)) :=
by sorry

end NUMINAMATH_CALUDE_production_scaling_l1897_189721


namespace NUMINAMATH_CALUDE_cubic_root_sum_power_l1897_189714

theorem cubic_root_sum_power (p q r t : ℝ) : 
  (p + q + r = 7) → 
  (p * q + q * r + r * p = 8) → 
  (p * q * r = 1) → 
  (t = Real.sqrt p + Real.sqrt q + Real.sqrt r) → 
  t^4 - 14 * t^2 - 8 * t = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_power_l1897_189714


namespace NUMINAMATH_CALUDE_interest_rate_cut_l1897_189761

theorem interest_rate_cut (x : ℝ) : 
  (2.25 / 100 : ℝ) * (1 - x)^2 = (1.98 / 100 : ℝ) → 
  (∃ (initial_rate final_rate : ℝ), 
    initial_rate = 2.25 / 100 ∧ 
    final_rate = 1.98 / 100 ∧ 
    final_rate = initial_rate * (1 - x)^2) :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_cut_l1897_189761


namespace NUMINAMATH_CALUDE_triangle_area_product_l1897_189779

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (1/2) * (8/a) * (8/b) = 8 → a * b = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_product_l1897_189779


namespace NUMINAMATH_CALUDE_red_crayon_boxes_l1897_189723

/-- The number of boxes of red crayons given the following conditions:
  * 6 boxes of 8 orange crayons each
  * 7 boxes of 5 blue crayons each
  * Each box of red crayons contains 11 crayons
  * Total number of crayons is 94
-/
theorem red_crayon_boxes : ℕ := by
  sorry

#check red_crayon_boxes

end NUMINAMATH_CALUDE_red_crayon_boxes_l1897_189723


namespace NUMINAMATH_CALUDE_coplanar_condition_l1897_189788

open Vector

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the points
variable (O E F G H : V)

-- Define the condition for coplanarity
def are_coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (B - A) + b • (C - A) + c • (D - A) = 0

-- Define the theorem
theorem coplanar_condition (m : ℝ) :
  (4 • (E - O) - 3 • (F - O) + 6 • (G - O) + m • (H - O) = 0) →
  (are_coplanar E F G H ↔ m = -7) :=
by sorry

end NUMINAMATH_CALUDE_coplanar_condition_l1897_189788


namespace NUMINAMATH_CALUDE_plane_speed_problem_l1897_189793

theorem plane_speed_problem (speed1 : ℝ) (time : ℝ) (total_distance : ℝ) (speed2 : ℝ) : 
  speed1 = 75 →
  time = 4.84848484848 →
  total_distance = 800 →
  (speed1 + speed2) * time = total_distance →
  speed2 = 90 := by
sorry

end NUMINAMATH_CALUDE_plane_speed_problem_l1897_189793


namespace NUMINAMATH_CALUDE_kite_side_lengths_l1897_189791

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length. -/
structure Kite where
  a : ℝ  -- First diagonal
  b : ℝ  -- Second diagonal
  k : ℝ  -- Half of the perimeter
  x : ℝ  -- Length of one side
  y : ℝ  -- Length of the other side

/-- Properties of the kite based on the given conditions -/
def kite_properties (q : Kite) : Prop :=
  q.a = 6 ∧ q.b = 25/4 ∧ q.k = 35/4 ∧ q.x + q.y = q.k

/-- The theorem stating the side lengths of the kite -/
theorem kite_side_lengths (q : Kite) (h : kite_properties q) :
  q.x = 5 ∧ q.y = 15/4 :=
sorry

end NUMINAMATH_CALUDE_kite_side_lengths_l1897_189791


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1897_189780

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2023) ↔ x ≥ 2023 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1897_189780


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1897_189798

theorem weight_of_new_person (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 3.7 →
  replaced_weight = 57.3 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101.7 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1897_189798


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1897_189786

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 3/7)
  (h2 : Real.tan (β - π/4) = -1/3) :
  Real.tan (α + π/4) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l1897_189786


namespace NUMINAMATH_CALUDE_total_friends_l1897_189778

def friends_in_line (front : ℕ) (back : ℕ) : ℕ :=
  (front - 1) + 1 + (back - 1)

theorem total_friends (seokjin_front : ℕ) (seokjin_back : ℕ) 
  (h1 : seokjin_front = 8) (h2 : seokjin_back = 6) : 
  friends_in_line seokjin_front seokjin_back = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_friends_l1897_189778


namespace NUMINAMATH_CALUDE_cake_change_calculation_l1897_189713

/-- Calculates the change received when buying cake slices -/
theorem cake_change_calculation (single_price double_price single_quantity double_quantity payment : ℕ) :
  single_price = 4 →
  double_price = 7 →
  single_quantity = 7 →
  double_quantity = 5 →
  payment = 100 →
  payment - (single_price * single_quantity + double_price * double_quantity) = 37 := by
  sorry

#check cake_change_calculation

end NUMINAMATH_CALUDE_cake_change_calculation_l1897_189713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_prime_factors_l1897_189796

theorem arithmetic_sequence_with_prime_factors
  (n d : ℕ+) :
  ∃ (a : ℕ → ℕ+),
    (∀ i j : ℕ, i < n ∧ j < n → a (i + 1) - a j = d * (i - j)) ∧
    (∀ i : ℕ, i < n → ∃ p : ℕ, p.Prime ∧ p ≥ i + 1 ∧ p ∣ a (i + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_prime_factors_l1897_189796


namespace NUMINAMATH_CALUDE_triangle_centroid_property_l1897_189754

/-- Triangle with centroid -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  h_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

/-- Distance squared between two points -/
def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Sum of squared distances from a point to triangle vertices -/
def sum_dist_sq (t : Triangle) (M : ℝ × ℝ) : ℝ :=
  dist_sq M t.A + dist_sq M t.B + dist_sq M t.C

/-- Theorem statement -/
theorem triangle_centroid_property (t : Triangle) :
  (∀ M : ℝ × ℝ, sum_dist_sq t M ≥ sum_dist_sq t t.G) ∧
  (∀ M : ℝ × ℝ, sum_dist_sq t M = sum_dist_sq t t.G ↔ M = t.G) ∧
  (∀ k : ℝ, k > sum_dist_sq t t.G →
    ∃ r : ℝ, r = Real.sqrt ((k - sum_dist_sq t t.G) / 3) ∧
      {M : ℝ × ℝ | sum_dist_sq t M = k} = {M : ℝ × ℝ | dist_sq M t.G = r^2}) :=
by sorry

end NUMINAMATH_CALUDE_triangle_centroid_property_l1897_189754


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l1897_189771

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l1897_189771


namespace NUMINAMATH_CALUDE_shaded_area_is_thirty_l1897_189775

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The large triangle partitioned into 25 congruent smaller triangles -/
def num_partitions : ℕ := 25

/-- The number of shaded smaller triangles -/
def num_shaded : ℕ := 15

/-- The theorem to be proved -/
theorem shaded_area_is_thirty 
  (t : IsoscelesRightTriangle) 
  (h_partitions : num_partitions = 25) 
  (h_shaded : num_shaded = 15) : 
  (t.leg_length * t.leg_length / 2) * (num_shaded / num_partitions) = 30 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_thirty_l1897_189775


namespace NUMINAMATH_CALUDE_square_sequence_properties_l1897_189764

/-- A quadratic sequence of unit squares -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The theorem stating the properties of the sequence -/
theorem square_sequence_properties :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 ∧ f 150 = 67951 := by
  sorry

#check square_sequence_properties

end NUMINAMATH_CALUDE_square_sequence_properties_l1897_189764


namespace NUMINAMATH_CALUDE_bank_savings_exceed_target_l1897_189755

/-- Geometric sequence sum function -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- Starting amount in cents -/
def initial_deposit : ℚ := 2

/-- Daily multiplication factor -/
def daily_factor : ℚ := 2

/-- Target amount in cents -/
def target_amount : ℚ := 400

theorem bank_savings_exceed_target :
  ∀ n : ℕ, n < 8 → geometric_sum initial_deposit daily_factor n < target_amount ∧
  geometric_sum initial_deposit daily_factor 8 ≥ target_amount :=
by sorry

end NUMINAMATH_CALUDE_bank_savings_exceed_target_l1897_189755


namespace NUMINAMATH_CALUDE_apple_count_l1897_189741

theorem apple_count (apples oranges : ℕ) : 
  oranges = 20 → 
  (apples : ℚ) / (apples + (oranges - 14 : ℚ)) = 7/10 → 
  apples = 14 := by
sorry

end NUMINAMATH_CALUDE_apple_count_l1897_189741


namespace NUMINAMATH_CALUDE_cone_intersection_volume_ratio_l1897_189729

/-- A cone with a circular base -/
structure Cone :=
  (radius : ℝ)
  (height : ℝ)

/-- A plane passing through the vertex of the cone -/
structure IntersectingPlane :=
  (chord_length : ℝ)

/-- The theorem stating the ratio of volumes when a plane intersects a cone -/
theorem cone_intersection_volume_ratio
  (c : Cone)
  (p : IntersectingPlane)
  (h1 : p.chord_length = c.radius) :
  ∃ (v1 v2 : ℝ),
    v1 > 0 ∧ v2 > 0 ∧
    (v1 / v2 = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_cone_intersection_volume_ratio_l1897_189729


namespace NUMINAMATH_CALUDE_scale_division_l1897_189783

/-- Given a scale of length 7 feet and 12 inches divided into 4 equal parts,
    the length of each part is 24 inches. -/
theorem scale_division (scale_length_feet : ℕ) (scale_length_inches : ℕ) (num_parts : ℕ) :
  scale_length_feet = 7 →
  scale_length_inches = 12 →
  num_parts = 4 →
  (scale_length_feet * 12 + scale_length_inches) / num_parts = 24 :=
by sorry

end NUMINAMATH_CALUDE_scale_division_l1897_189783


namespace NUMINAMATH_CALUDE_parallelogram_vertex_C_l1897_189733

/-- Represents a parallelogram in the complex plane -/
structure ComplexParallelogram where
  O : ℂ
  A : ℂ
  B : ℂ
  C : ℂ
  is_origin : O = 0
  is_parallelogram : C - O = B - A

/-- The complex number corresponding to vertex C in the given parallelogram -/
def vertex_C (p : ComplexParallelogram) : ℂ := p.B + p.A

/-- Theorem stating that for the given parallelogram, vertex C corresponds to 3+5i -/
theorem parallelogram_vertex_C :
  ∀ (p : ComplexParallelogram),
    p.O = 0 ∧ p.A = 1 - 3*I ∧ p.B = 4 + 2*I →
    vertex_C p = 3 + 5*I := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_C_l1897_189733


namespace NUMINAMATH_CALUDE_roberts_extra_chocolates_l1897_189734

/-- Given that Robert ate 12 chocolates and Nickel ate 3 chocolates,
    prove that Robert ate 9 more chocolates than Nickel. -/
theorem roberts_extra_chocolates (robert : Nat) (nickel : Nat)
    (h1 : robert = 12) (h2 : nickel = 3) :
    robert - nickel = 9 := by
  sorry

end NUMINAMATH_CALUDE_roberts_extra_chocolates_l1897_189734


namespace NUMINAMATH_CALUDE_nancy_zoo_pictures_nancy_zoo_pictures_proof_l1897_189781

theorem nancy_zoo_pictures : ℕ → Prop :=
  fun zoo_pictures =>
    let museum_pictures := 8
    let deleted_pictures := 38
    let remaining_pictures := 19
    zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures →
    zoo_pictures = 49

-- Proof
theorem nancy_zoo_pictures_proof : nancy_zoo_pictures 49 := by
  sorry

end NUMINAMATH_CALUDE_nancy_zoo_pictures_nancy_zoo_pictures_proof_l1897_189781


namespace NUMINAMATH_CALUDE_hash_four_negative_three_l1897_189731

-- Define the # operation
def hash (x y : Int) : Int := x * (y - 1) + x * y

-- Theorem statement
theorem hash_four_negative_three : hash 4 (-3) = -28 := by
  sorry

end NUMINAMATH_CALUDE_hash_four_negative_three_l1897_189731


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1897_189704

open Set

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {3, 4}

-- Define set B
def B : Set ℕ := {1, 4, 5}

-- Theorem statement
theorem union_of_A_and_complement_of_B : A ∪ (U \ B) = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1897_189704


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l1897_189736

def A (m : ℝ) : Set ℝ := {3, m^2}
def B (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}

theorem subset_implies_m_equals_one (m : ℝ) : A m ⊆ B m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l1897_189736


namespace NUMINAMATH_CALUDE_xyz_sum_l1897_189700

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : x * z + y = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l1897_189700


namespace NUMINAMATH_CALUDE_cloak_change_theorem_l1897_189725

/-- Represents the price of an invisibility cloak and the change received in different scenarios -/
structure CloakTransaction where
  silverPaid : ℕ
  goldChange : ℕ

/-- Calculates the number of silver coins received as change when buying a cloak with gold coins -/
def silverChangeForGoldPurchase (transaction1 transaction2 : CloakTransaction) (goldPaid : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct change in silver coins when buying a cloak for 14 gold coins -/
theorem cloak_change_theorem (transaction1 transaction2 : CloakTransaction) 
  (h1 : transaction1.silverPaid = 20 ∧ transaction1.goldChange = 4)
  (h2 : transaction2.silverPaid = 15 ∧ transaction2.goldChange = 1) :
  silverChangeForGoldPurchase transaction1 transaction2 14 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_change_theorem_l1897_189725


namespace NUMINAMATH_CALUDE_painters_rooms_theorem_l1897_189770

/-- Given that 3 painters can complete 3 rooms in 3 hours, 
    prove that 9 painters can complete 27 rooms in 9 hours. -/
theorem painters_rooms_theorem (painters_rate : ℕ → ℕ → ℕ → ℕ) 
  (h : painters_rate 3 3 3 = 3) : painters_rate 9 9 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_painters_rooms_theorem_l1897_189770


namespace NUMINAMATH_CALUDE_jack_euros_calculation_l1897_189776

/-- Calculates the number of euros Jack has given his total currency amounts -/
theorem jack_euros_calculation (pounds : ℕ) (yen : ℕ) (total_yen : ℕ) 
  (h1 : pounds = 42)
  (h2 : yen = 3000)
  (h3 : total_yen = 9400)
  (h4 : ∀ (e : ℕ), e * 2 * 100 + yen + pounds * 100 = total_yen) :
  ∃ (euros : ℕ), euros = 11 ∧ euros * 2 * 100 + yen + pounds * 100 = total_yen :=
by sorry

end NUMINAMATH_CALUDE_jack_euros_calculation_l1897_189776


namespace NUMINAMATH_CALUDE_overtaking_time_l1897_189787

/-- The problem of determining when person B starts walking to overtake person A --/
theorem overtaking_time (speed_A speed_B overtake_time : ℝ) (h1 : speed_A = 5)
  (h2 : speed_B = 5.555555555555555) (h3 : overtake_time = 1.8) :
  let start_time_diff := overtake_time * speed_B / speed_A - overtake_time
  start_time_diff = 0.2 := by sorry

end NUMINAMATH_CALUDE_overtaking_time_l1897_189787


namespace NUMINAMATH_CALUDE_max_profit_is_120_l1897_189750

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := -x^2 + 21*x

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2*x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := L₁ x + L₂ x

/-- Total sales volume constraint -/
def sales_constraint : ℝ := 15

theorem max_profit_is_120 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ sales_constraint ∧
  ∀ y : ℝ, y ≥ 0 ∧ y ≤ sales_constraint →
  total_profit x ≥ total_profit y ∧
  total_profit x = 120 :=
by sorry

end NUMINAMATH_CALUDE_max_profit_is_120_l1897_189750


namespace NUMINAMATH_CALUDE_machinery_expense_l1897_189747

/-- Proves that the amount spent on machinery is $1000 --/
theorem machinery_expense (total : ℝ) (raw_materials : ℝ) (cash_percentage : ℝ) :
  total = 5714.29 →
  raw_materials = 3000 →
  cash_percentage = 0.30 →
  ∃ (machinery : ℝ),
    machinery = 1000 ∧
    total = raw_materials + machinery + (cash_percentage * total) :=
by
  sorry


end NUMINAMATH_CALUDE_machinery_expense_l1897_189747


namespace NUMINAMATH_CALUDE_perpendicular_iff_m_eq_half_l1897_189709

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: 2x - y - 1 = 0 -/
def l1 : Line :=
  { a := 2, b := -1, c := -1 }

/-- The second line: mx + y + 1 = 0 -/
def l2 (m : ℝ) : Line :=
  { a := m, b := 1, c := 1 }

/-- The theorem stating the necessary and sufficient condition for perpendicularity -/
theorem perpendicular_iff_m_eq_half :
  ∀ m : ℝ, perpendicular l1 (l2 m) ↔ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_iff_m_eq_half_l1897_189709


namespace NUMINAMATH_CALUDE_rescue_possible_l1897_189790

/-- Represents the rescue mission parameters --/
structure RescueMission where
  distance : ℝ
  rover_air : ℝ
  ponchik_extra_air : ℝ
  dunno_tank_air : ℝ
  max_tanks : ℕ
  speed : ℝ

/-- Represents a rescue strategy --/
structure RescueStrategy where
  trips : ℕ
  air_drops : List ℝ
  meeting_point : ℝ

/-- Checks if a rescue strategy is valid for a given mission --/
def is_valid_strategy (mission : RescueMission) (strategy : RescueStrategy) : Prop :=
  -- Define the conditions for a valid strategy
  sorry

/-- Theorem stating that a valid rescue strategy exists --/
theorem rescue_possible (mission : RescueMission) 
  (h1 : mission.distance = 18)
  (h2 : mission.rover_air = 3)
  (h3 : mission.ponchik_extra_air = 1)
  (h4 : mission.dunno_tank_air = 2)
  (h5 : mission.max_tanks = 2)
  (h6 : mission.speed = 6) :
  ∃ (strategy : RescueStrategy), is_valid_strategy mission strategy :=
sorry

end NUMINAMATH_CALUDE_rescue_possible_l1897_189790


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1897_189785

/-- Given a regular pentagon and a rectangle with the same perimeter,
    where the rectangle's length is twice its width,
    prove that the ratio of the pentagon's side length to the rectangle's width is 6/5 -/
theorem pentagon_rectangle_ratio (p w : ℝ) (h1 : 5 * p = 30) (h2 : 6 * w = 30) : p / w = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1897_189785


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l1897_189795

theorem no_simultaneous_squares : ∀ n : ℤ, ¬(∃ a b c : ℤ, (10 * n - 1 = a^2) ∧ (13 * n - 1 = b^2) ∧ (85 * n - 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l1897_189795


namespace NUMINAMATH_CALUDE_linear_function_point_value_l1897_189724

theorem linear_function_point_value (m n : ℝ) : 
  n = 3 - 5 * m → 10 * m + 2 * n - 3 = 3 := by sorry

end NUMINAMATH_CALUDE_linear_function_point_value_l1897_189724


namespace NUMINAMATH_CALUDE_smallest_n_for_2005_angles_l1897_189740

/-- A function that, given a natural number n, returns the number of angles not exceeding 120° 
    between pairs of points when n points are placed on a circle. -/
def anglesNotExceeding120 (n : ℕ) : ℕ := sorry

/-- The proposition that 91 is the smallest natural number satisfying the condition -/
theorem smallest_n_for_2005_angles : 
  (∀ n : ℕ, n < 91 → anglesNotExceeding120 n < 2005) ∧ 
  (anglesNotExceeding120 91 ≥ 2005) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_2005_angles_l1897_189740


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1897_189715

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 < a + b) (h2 : a + b < 3) 
  (h3 : 2 < a - b) (h4 : a - b < 4) : 
  ∃ (x : ℝ), -9/2 < 2*a + 3*b ∧ 2*a + 3*b < 13/2 ∧ 
  ∀ (y : ℝ), -9/2 < y ∧ y < 13/2 → ∃ (a' b' : ℝ), 
    -1 < a' + b' ∧ a' + b' < 3 ∧ 
    2 < a' - b' ∧ a' - b' < 4 ∧ 
    2*a' + 3*b' = y :=
sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1897_189715
