import Mathlib

namespace square_root_of_factorial_fraction_l366_36654

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 84) = 24 * Real.sqrt 15 := by
  sorry

end square_root_of_factorial_fraction_l366_36654


namespace right_triangle_hypotenuse_l366_36692

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 → b = 36 → c^2 = a^2 + b^2 → c = 39 := by
  sorry

end right_triangle_hypotenuse_l366_36692


namespace ellipse_chord_slope_l366_36680

/-- The slope of a chord on an ellipse, given its midpoint -/
theorem ellipse_chord_slope (x₁ x₂ y₁ y₂ : ℝ) :
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 1) →
  ((y₁ + y₂) / 2 = 2) →
  (y₁ - y₂) / (x₁ - x₂) = -9 / 32 := by
  sorry

end ellipse_chord_slope_l366_36680


namespace abc_sum_l366_36664

theorem abc_sum (a b c : ℕ+) 
  (eq1 : a * b + c + 10 = 51)
  (eq2 : b * c + a + 10 = 51)
  (eq3 : a * c + b + 10 = 51) :
  a + b + c = 41 := by
  sorry

end abc_sum_l366_36664


namespace product_of_constrained_values_l366_36641

theorem product_of_constrained_values (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 10 := by
sorry

end product_of_constrained_values_l366_36641


namespace square_minimizes_diagonal_l366_36672

/-- A parallelogram with side lengths and angles -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  angle : ℝ
  area : ℝ

/-- The length of the larger diagonal of a parallelogram -/
def largerDiagonal (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem: Among all parallelograms with a given area, the square has the smallest larger diagonal -/
theorem square_minimizes_diagonal {A : ℝ} (h : A > 0) :
  ∀ p : Parallelogram, p.area = A →
    largerDiagonal p ≥ largerDiagonal { side1 := Real.sqrt A, side2 := Real.sqrt A, angle := π/2, area := A } :=
  sorry

end square_minimizes_diagonal_l366_36672


namespace cubic_root_equation_solution_l366_36623

theorem cubic_root_equation_solution :
  ∃! x : ℝ, 2.61 * (9 - Real.sqrt (x + 1))^(1/3) + (7 + Real.sqrt (x + 1))^(1/3) = 4 :=
by
  sorry

end cubic_root_equation_solution_l366_36623


namespace unicity_of_inverse_l366_36651

variable {G : Type*} [Group G]

theorem unicity_of_inverse (x y z : G) (h1 : 1 = x * y) (h2 : 1 = z * x) :
  y = z ∧ y = x⁻¹ := by
  sorry

end unicity_of_inverse_l366_36651


namespace set_containment_implies_a_bound_l366_36663

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- State the theorem
theorem set_containment_implies_a_bound (a : ℝ) :
  A ⊆ B a → a ≤ -2 := by
  sorry

-- The range of a is implicitly (-∞, -2] because a ≤ -2

end set_containment_implies_a_bound_l366_36663


namespace x_value_when_y_is_two_l366_36627

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by sorry

end x_value_when_y_is_two_l366_36627


namespace fraction_greater_than_one_implication_false_l366_36653

theorem fraction_greater_than_one_implication_false : 
  ¬(∀ a b : ℝ, a / b > 1 → a > b) := by sorry

end fraction_greater_than_one_implication_false_l366_36653


namespace min_value_of_a_l366_36676

theorem min_value_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∃ x, f x ≤ 0) →
  (∀ x, f x = Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x) →
  a ≥ 1 - 1 / Real.exp 1 :=
by sorry

end min_value_of_a_l366_36676


namespace washing_machine_cycle_time_l366_36635

theorem washing_machine_cycle_time 
  (total_items : ℕ) 
  (machine_capacity : ℕ) 
  (total_wash_time_minutes : ℕ) 
  (h1 : total_items = 60)
  (h2 : machine_capacity = 15)
  (h3 : total_wash_time_minutes = 180) :
  total_wash_time_minutes / (total_items / machine_capacity) = 45 :=
by
  sorry

end washing_machine_cycle_time_l366_36635


namespace park_outer_boundary_diameter_l366_36632

/-- Represents a circular park with concentric features -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  grassy_area_width : ℝ
  walking_path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.grassy_area_width + park.walking_path_width)

/-- Theorem stating that for a park with given measurements, the outer boundary diameter is 52 feet -/
theorem park_outer_boundary_diameter :
  let park : CircularPark := {
    pond_diameter := 12,
    garden_width := 10,
    grassy_area_width := 4,
    walking_path_width := 6
  }
  outer_boundary_diameter park = 52 := by
  sorry

end park_outer_boundary_diameter_l366_36632


namespace pet_store_dogs_l366_36621

theorem pet_store_dogs (cat_count : ℕ) (cat_ratio dog_ratio : ℕ) : 
  cat_count = 21 → cat_ratio = 3 → dog_ratio = 4 → 
  (cat_count * dog_ratio) / cat_ratio = 28 := by
sorry

end pet_store_dogs_l366_36621


namespace percent_profit_problem_l366_36698

/-- Given that the cost price of 60 articles equals the selling price of 50 articles,
    prove that the percent profit is 20%. -/
theorem percent_profit_problem (C S : ℝ) (h : 60 * C = 50 * S) :
  (S - C) / C * 100 = 20 := by
  sorry

end percent_profit_problem_l366_36698


namespace k_range_k_values_circle_origin_l366_36630

-- Define the line and hyperbola equations
def line (k x : ℝ) : ℝ := k * x + 1
def hyperbola (x y : ℝ) : Prop := 3 * x^2 - y^2 = 1

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁ < 0 ∧ x₂ > 0 ∧
    hyperbola x₁ y₁ ∧ hyperbola x₂ y₂ ∧
    y₁ = line k x₁ ∧ y₂ = line k x₂

-- Theorem for the range of k
theorem k_range :
  ∀ k : ℝ, intersection_points k ↔ -Real.sqrt 3 < k ∧ k < Real.sqrt 3 :=
sorry

-- Define the condition for the circle passing through the origin
def circle_through_origin (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    intersection_points k ∧
    x₁ * x₂ + y₁ * y₂ = 0

-- Theorem for the values of k when the circle passes through the origin
theorem k_values_circle_origin :
  ∀ k : ℝ, circle_through_origin k ↔ k = 1 ∨ k = -1 :=
sorry

end k_range_k_values_circle_origin_l366_36630


namespace boxwood_charge_theorem_l366_36609

/-- Calculates the total charge for trimming and shaping boxwoods -/
def total_charge (num_boxwoods : ℕ) (num_shaped : ℕ) (trim_cost : ℚ) (shape_cost : ℚ) : ℚ :=
  (num_boxwoods * trim_cost) + (num_shaped * shape_cost)

/-- Proves that the total charge for trimming 30 boxwoods and shaping 4 of them is $210.00 -/
theorem boxwood_charge_theorem :
  total_charge 30 4 5 15 = 210 := by
  sorry

#eval total_charge 30 4 5 15

end boxwood_charge_theorem_l366_36609


namespace ring_arrangement_correct_l366_36602

/-- The number of ways to arrange 5 rings out of 9 on 5 fingers -/
def ring_arrangements (total_rings : ℕ) (arranged_rings : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings arranged_rings * Nat.factorial arranged_rings * Nat.choose (total_rings - 1) (fingers - 1)

/-- The correct number of arrangements for 9 rings, 5 arranged, on 5 fingers -/
def correct_arrangement : ℕ := 1905120

/-- Theorem stating that the number of arrangements is correct -/
theorem ring_arrangement_correct :
  ring_arrangements 9 5 5 = correct_arrangement := by
  sorry

end ring_arrangement_correct_l366_36602


namespace root_sum_product_l366_36604

theorem root_sum_product (a b : ℝ) : 
  (a^4 - 4*a^2 - a - 1 = 0) → 
  (b^4 - 4*b^2 - b - 1 = 0) → 
  (a + b) * (a * b + 1) = -1/2 := by
sorry

end root_sum_product_l366_36604


namespace impossible_arrangement_l366_36667

/-- Represents a 3x3 grid of digits -/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- The set of digits used in the grid -/
def Digits : Finset (Fin 4) := {0, 1, 2, 3}

/-- Checks if a row contains three different digits -/
def row_valid (g : Grid) (i : Fin 3) : Prop :=
  (Finset.card {g i 0, g i 1, g i 2}) = 3

/-- Checks if a column contains three different digits -/
def col_valid (g : Grid) (j : Fin 3) : Prop :=
  (Finset.card {g 0 j, g 1 j, g 2 j}) = 3

/-- Checks if the main diagonal contains three different digits -/
def main_diag_valid (g : Grid) : Prop :=
  (Finset.card {g 0 0, g 1 1, g 2 2}) = 3

/-- Checks if the anti-diagonal contains three different digits -/
def anti_diag_valid (g : Grid) : Prop :=
  (Finset.card {g 0 2, g 1 1, g 2 0}) = 3

/-- Checks if the grid is valid according to all conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i : Fin 3, row_valid g i) ∧
  (∀ j : Fin 3, col_valid g j) ∧
  main_diag_valid g ∧
  anti_diag_valid g

theorem impossible_arrangement : ¬∃ (g : Grid), valid_grid g := by
  sorry

end impossible_arrangement_l366_36667


namespace line_l1_equation_range_of_b_l366_36679

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4*y + 4 = 0

-- Define the midpoint P of the chord intercepted by line l1
def midpoint_P : ℝ × ℝ := (5, 3)

-- Define line l2
def line_l2 (x y b : ℝ) : Prop := x + y + b = 0

-- Theorem for the equation of line l1
theorem line_l1_equation : 
  ∃ (l1 : ℝ → ℝ → Prop), 
  (∀ x y, l1 x y ↔ 2*x + y - 13 = 0) ∧ 
  (∀ x y, l1 x y → circle_C x y) ∧
  (l1 (midpoint_P.1) (midpoint_P.2)) := 
sorry

-- Theorem for the range of b
theorem range_of_b :
  ∀ b : ℝ, (∃ x y, circle_C x y ∧ line_l2 x y b) ↔ 
  (-3 * Real.sqrt 2 - 5 < b ∧ b < 3 * Real.sqrt 2 - 5) := 
sorry

end line_l1_equation_range_of_b_l366_36679


namespace trapezoid_in_square_l366_36648

theorem trapezoid_in_square (s : ℝ) (x : ℝ) : 
  s = 2 → -- Side length of the square
  (1/3) * s^2 = (1/2) * (s + x) * (s/2) → -- Area of trapezoid is 1/3 of square's area
  x = 2/3 := by
sorry

end trapezoid_in_square_l366_36648


namespace remainder_divisibility_l366_36652

theorem remainder_divisibility (x : ℤ) (h : x % 66 = 14) : x % 11 = 3 := by
  sorry

end remainder_divisibility_l366_36652


namespace power_three_inverse_exponent_l366_36675

theorem power_three_inverse_exponent (x y : ℕ) : 
  (2^x : ℕ) ∣ 900 ∧ 
  ∀ k > x, ¬((2^k : ℕ) ∣ 900) ∧ 
  (5^y : ℕ) ∣ 900 ∧ 
  ∀ l > y, ¬((5^l : ℕ) ∣ 900) → 
  (1/3 : ℚ)^(2*(y - x)) = 1 := by
sorry

end power_three_inverse_exponent_l366_36675


namespace last_day_of_second_quarter_in_common_year_l366_36638

/-- Represents a day in a month -/
structure DayInMonth where
  month : Nat
  day : Nat

/-- Definition of a common year -/
def isCommonYear (daysInYear : Nat) : Prop :=
  daysInYear = 365

/-- Definition of the last day of the second quarter -/
def isLastDayOfSecondQuarter (d : DayInMonth) : Prop :=
  d.month = 6 ∧ d.day = 30

/-- Theorem: In a common year, the last day of the second quarter is June 30 -/
theorem last_day_of_second_quarter_in_common_year 
  (daysInYear : Nat) 
  (h : isCommonYear daysInYear) :
  ∃ d : DayInMonth, isLastDayOfSecondQuarter d :=
by
  sorry

end last_day_of_second_quarter_in_common_year_l366_36638


namespace compound_inequality_l366_36618

theorem compound_inequality (x : ℝ) : 
  x > -1/2 → (3 - 1/(3*x + 4) < 5 ∧ 2*x + 1 > 0) :=
by sorry

end compound_inequality_l366_36618


namespace total_triangles_is_28_l366_36659

/-- Represents a triangular arrangement of equilateral triangles -/
structure TriangularArrangement where
  rows : ℕ
  -- Each row n contains n unit triangles
  unit_triangles_in_row : (n : ℕ) → n ≤ rows → ℕ
  unit_triangles_in_row_eq : ∀ n h, unit_triangles_in_row n h = n

/-- Counts the total number of equilateral triangles in the arrangement -/
def count_all_triangles (arrangement : TriangularArrangement) : ℕ :=
  sorry

/-- The main theorem: In a triangular arrangement with 6 rows, 
    the total number of equilateral triangles is 28 -/
theorem total_triangles_is_28 :
  ∀ (arrangement : TriangularArrangement),
  arrangement.rows = 6 →
  count_all_triangles arrangement = 28 :=
sorry

end total_triangles_is_28_l366_36659


namespace quadratic_equation_solution_l366_36606

theorem quadratic_equation_solution (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 4 * x + 20 = 0 ↔ x = (a : ℂ) + b * I ∨ x = (a : ℂ) - b * I) →
  a + b^2 = 394/25 := by
  sorry

end quadratic_equation_solution_l366_36606


namespace product_xy_in_parallelogram_l366_36665

/-- A parallelogram with given side lengths -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ → ℝ
  GH : ℝ → ℝ
  HE : ℝ
  is_parallelogram : EF = GH 1 ∧ FG 1 = HE

/-- The product of x and y in the given parallelogram is 18√3 -/
theorem product_xy_in_parallelogram (p : Parallelogram) 
    (h1 : p.EF = 42)
    (h2 : p.FG = fun y ↦ 4 * y^2 + 1)
    (h3 : p.GH = fun x ↦ 3 * x + 6)
    (h4 : p.HE = 28) :
    ∃ x y, p.GH x = p.EF ∧ p.FG y = p.HE ∧ x * y = 18 * Real.sqrt 3 := by
  sorry

end product_xy_in_parallelogram_l366_36665


namespace arithmetic_sequence_common_difference_l366_36649

/-- An arithmetic sequence with the given properties has a common difference of 1/3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- The arithmetic sequence
  (h1 : a 3 + a 5 = 2)  -- First condition
  (h2 : a 7 + a 10 + a 13 = 9)  -- Second condition
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  : ∃ d : ℚ, d = 1/3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end arithmetic_sequence_common_difference_l366_36649


namespace tank_circumference_l366_36605

/-- Given two right circular cylindrical tanks C and B, prove that the circumference of tank B is 10 meters. -/
theorem tank_circumference (h_C h_B r_C r_B : ℝ) : 
  h_C = 10 →  -- Height of tank C
  h_B = 8 →   -- Height of tank B
  2 * Real.pi * r_C = 8 →  -- Circumference of tank C
  (Real.pi * r_C^2 * h_C) = 0.8 * (Real.pi * r_B^2 * h_B) →  -- Volume relation
  2 * Real.pi * r_B = 10  -- Circumference of tank B
:= by sorry

end tank_circumference_l366_36605


namespace sqrt_difference_equals_two_sqrt_two_l366_36689

theorem sqrt_difference_equals_two_sqrt_two :
  Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_difference_equals_two_sqrt_two_l366_36689


namespace exponential_inequality_and_unique_a_l366_36637

open Real

theorem exponential_inequality_and_unique_a :
  (∀ x > -1, exp x > (x + 1)^2 / 2) ∧
  (∃! a : ℝ, a > 0 ∧ ∀ x > 0, exp (1 - x) + 2 * log x ≤ a * (x - 1) + 1) ∧
  (∃ a : ℝ, a > 0 ∧ ∀ x > 0, exp (1 - x) + 2 * log x ≤ a * (x - 1) + 1 ∧ a = 1) :=
by sorry

end exponential_inequality_and_unique_a_l366_36637


namespace difference_of_half_and_third_l366_36694

theorem difference_of_half_and_third : 1/2 - 1/3 = 1/6 := by
  sorry

end difference_of_half_and_third_l366_36694


namespace area_36_implies_a_plus_b_6_l366_36629

/-- A quadrilateral with vertices defined by a positive integer a -/
structure Quadrilateral (a : ℕ+) where
  P : ℝ × ℝ := (a, a)
  Q : ℝ × ℝ := (a, -a)
  R : ℝ × ℝ := (-a, -a)
  S : ℝ × ℝ := (-a, a)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral a) : ℝ := sorry

/-- Theorem: If the area of quadrilateral PQRS is 36, then a+b = 6 -/
theorem area_36_implies_a_plus_b_6 (a b : ℕ+) (q : Quadrilateral a) :
  area q = 36 → a + b = 6 := by sorry

end area_36_implies_a_plus_b_6_l366_36629


namespace triangle_abc_theorem_l366_36628

theorem triangle_abc_theorem (a b c : ℝ) (A B C : ℝ) :
  a * Real.sin (2 * B) = Real.sqrt 3 * b * Real.sin A →
  Real.cos A = 1 / 3 →
  B = π / 6 ∧ Real.sin C = (2 * Real.sqrt 6 + 1) / 6 :=
by sorry

end triangle_abc_theorem_l366_36628


namespace power_fraction_equality_l366_36688

theorem power_fraction_equality : (2^4 * 3^2 * 5^3 * 7^2) / 11 = 80182 := by
  sorry

end power_fraction_equality_l366_36688


namespace original_people_count_l366_36601

/-- The original number of people in the room. -/
def original_people : ℕ := 36

/-- The fraction of people who left initially. -/
def fraction_left : ℚ := 1 / 3

/-- The fraction of remaining people who started dancing. -/
def fraction_dancing : ℚ := 1 / 4

/-- The number of people who were not dancing. -/
def non_dancing_people : ℕ := 18

theorem original_people_count :
  (original_people : ℚ) * (1 - fraction_left) * (1 - fraction_dancing) = non_dancing_people := by
  sorry

end original_people_count_l366_36601


namespace angle_sum_l366_36620

theorem angle_sum (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (h3 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 2)
  (h4 : 4 * Real.cos (2 * a) + 3 * Real.cos (2 * b) = 1) :
  a + b = π/2 := by sorry

end angle_sum_l366_36620


namespace sin_power_sum_l366_36684

theorem sin_power_sum (φ : Real) (x : Real) (n : Nat) 
  (h1 : 0 < φ) (h2 : φ < π / 2) 
  (h3 : x + 1 / x = 2 * Real.sin φ) 
  (h4 : n > 0) : 
  x^n + 1 / x^n = 2 * Real.sin (n * φ) := by
  sorry

end sin_power_sum_l366_36684


namespace arithmetic_events_classification_l366_36639

/-- Represents the sign of a number -/
inductive Sign
| Positive
| Negative

/-- Represents the result of an arithmetic operation -/
inductive Result
| Positive
| Negative

/-- Represents an arithmetic event -/
structure ArithmeticEvent :=
  (operation : String)
  (sign1 : Sign)
  (sign2 : Sign)
  (result : Result)

/-- Defines the four events described in the problem -/
def events : List ArithmeticEvent :=
  [ ⟨"Addition", Sign.Positive, Sign.Negative, Result.Negative⟩
  , ⟨"Subtraction", Sign.Positive, Sign.Negative, Result.Positive⟩
  , ⟨"Multiplication", Sign.Positive, Sign.Negative, Result.Positive⟩
  , ⟨"Division", Sign.Positive, Sign.Negative, Result.Negative⟩ ]

/-- Predicate to determine if an event is certain -/
def isCertain (e : ArithmeticEvent) : Prop :=
  e.operation = "Division" ∧ 
  e.sign1 ≠ e.sign2 ∧ 
  e.result = Result.Negative

/-- Predicate to determine if an event is random -/
def isRandom (e : ArithmeticEvent) : Prop :=
  (e.operation = "Addition" ∨ e.operation = "Subtraction") ∧
  e.sign1 ≠ e.sign2

theorem arithmetic_events_classification :
  ∃ (certain : ArithmeticEvent) (random1 random2 : ArithmeticEvent),
    certain ∈ events ∧
    random1 ∈ events ∧
    random2 ∈ events ∧
    isCertain certain ∧
    isRandom random1 ∧
    isRandom random2 ∧
    random1 ≠ random2 :=
  sorry

end arithmetic_events_classification_l366_36639


namespace smaller_number_in_ratio_l366_36671

theorem smaller_number_in_ratio (a b c x y : ℝ) : 
  0 < a → a < b → x > 0 → y > 0 → x / y = a / (b ^ 2) → x + y = 2 * c → 
  min x y = (2 * a * c) / (a + b ^ 2) := by
  sorry

end smaller_number_in_ratio_l366_36671


namespace intersection_point_of_line_with_x_axis_l366_36608

/-- The intersection point of the line y = 2x - 4 with the x-axis is (2, 0). -/
theorem intersection_point_of_line_with_x_axis :
  let f : ℝ → ℝ := λ x ↦ 2 * x - 4
  ∃! x : ℝ, f x = 0 ∧ x = 2 := by sorry

end intersection_point_of_line_with_x_axis_l366_36608


namespace triangle_side_sum_l366_36693

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (3 * b - a) * Real.cos C = c * Real.cos A →
  c^2 = a * b →
  (1/2) * a * b * Real.sin C = 3 * Real.sqrt 2 →
  a + b = Real.sqrt 33 := by
sorry

end triangle_side_sum_l366_36693


namespace largest_area_polygon_E_l366_36634

/-- Represents a polygon composed of unit squares and right triangles -/
structure Polygon where
  unitSquares : ℕ
  rightTriangles : ℕ

/-- Calculates the area of a polygon -/
def areaOfPolygon (p : Polygon) : ℚ :=
  p.unitSquares + p.rightTriangles / 2

/-- The given polygons -/
def polygonA : Polygon := ⟨3, 2⟩
def polygonB : Polygon := ⟨6, 0⟩
def polygonC : Polygon := ⟨4, 3⟩
def polygonD : Polygon := ⟨5, 1⟩
def polygonE : Polygon := ⟨7, 0⟩

theorem largest_area_polygon_E :
  ∀ p ∈ [polygonA, polygonB, polygonC, polygonD, polygonE],
    areaOfPolygon p ≤ areaOfPolygon polygonE :=
by sorry

end largest_area_polygon_E_l366_36634


namespace triangle_angle_ratio_l366_36695

theorem triangle_angle_ratio (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles is 180°
  b = 4 * a →              -- ratio condition
  c = 7 * a →              -- ratio condition
  a = 15 ∧ b = 60 ∧ c = 105 := by
sorry

end triangle_angle_ratio_l366_36695


namespace complex_power_difference_l366_36624

theorem complex_power_difference (x : ℂ) (h : x - 1/x = 2*I) : x^2048 - 1/x^2048 = 0 := by
  sorry

end complex_power_difference_l366_36624


namespace total_wheels_at_park_l366_36613

-- Define the number of regular bikes
def regular_bikes : ℕ := 7

-- Define the number of children's bikes
def children_bikes : ℕ := 11

-- Define the number of wheels on a regular bike
def regular_bike_wheels : ℕ := 2

-- Define the number of wheels on a children's bike
def children_bike_wheels : ℕ := 4

-- Theorem: The total number of wheels Naomi saw at the park is 58
theorem total_wheels_at_park : 
  regular_bikes * regular_bike_wheels + children_bikes * children_bike_wheels = 58 := by
  sorry

end total_wheels_at_park_l366_36613


namespace equation_solution_l366_36650

theorem equation_solution (x : ℝ) :
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end equation_solution_l366_36650


namespace fraction_undefined_l366_36699

theorem fraction_undefined (x : ℝ) : (3 * x - 1) / (x + 3) = 0 / 0 ↔ x = -3 := by
  sorry

end fraction_undefined_l366_36699


namespace value_of_a_l366_36696

theorem value_of_a : ∀ a : ℕ, 
  (a * (9^3) = 3 * (15^5)) → 
  (a = 5^5) → 
  (a = 3125) := by
sorry

end value_of_a_l366_36696


namespace units_digit_of_7_19_l366_36619

theorem units_digit_of_7_19 : (7^19) % 10 = 3 := by
  sorry

end units_digit_of_7_19_l366_36619


namespace real_roots_quadratic_l366_36681

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x + 16*k = 0) ↔ k ≤ 0 ∨ k ≥ 64 := by
sorry

end real_roots_quadratic_l366_36681


namespace iggy_wednesday_miles_l366_36673

/-- Represents the days of the week Iggy runs --/
inductive RunDay
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Represents Iggy's running schedule --/
def IggySchedule : RunDay → ℕ
  | RunDay.Monday => 3
  | RunDay.Tuesday => 4
  | RunDay.Thursday => 8
  | RunDay.Friday => 3
  | RunDay.Wednesday => 0  -- We'll prove this should be 6

/-- Iggy's pace in minutes per mile --/
def IggyPace : ℕ := 10

/-- Total running time in hours --/
def TotalRunningTime : ℕ := 4

/-- Converts hours to minutes --/
def HoursToMinutes (hours : ℕ) : ℕ := hours * 60

theorem iggy_wednesday_miles :
  ∃ (wednesday_miles : ℕ),
    wednesday_miles = 6 ∧
    HoursToMinutes TotalRunningTime =
      (IggySchedule RunDay.Monday +
       IggySchedule RunDay.Tuesday +
       wednesday_miles +
       IggySchedule RunDay.Thursday +
       IggySchedule RunDay.Friday) * IggyPace :=
by sorry

end iggy_wednesday_miles_l366_36673


namespace common_area_of_30_60_90_triangles_l366_36626

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ

/-- The area common to two congruent 30-60-90 triangles with coinciding shorter legs -/
def commonArea (t : Triangle30_60_90) : ℝ := t.shortLeg ^ 2

/-- Theorem: The area common to two congruent 30-60-90 triangles with hypotenuse 16 and coinciding shorter legs is 64 square units -/
theorem common_area_of_30_60_90_triangles :
  ∀ t : Triangle30_60_90,
  t.hypotenuse = 16 →
  t.shortLeg = t.hypotenuse / 2 →
  t.longLeg = t.shortLeg * Real.sqrt 3 →
  commonArea t = 64 := by
  sorry

end common_area_of_30_60_90_triangles_l366_36626


namespace max_tank_volume_l366_36661

/-- A rectangular parallelepiped tank with the given properties -/
structure Tank where
  a : Real  -- length of the base
  b : Real  -- width of the base
  h : Real  -- height of the tank
  h_pos : h > 0
  a_pos : a > 0
  b_pos : b > 0
  side_area_condition : a * h ≥ a * b ∧ b * h ≥ a * b

/-- The theorem stating the maximum volume of the tank -/
theorem max_tank_volume (tank : Tank) (h_val : tank.h = 1.5) :
  (∀ t : Tank, t.h = 1.5 → t.a * t.b * t.h ≤ tank.a * tank.b * tank.h) →
  tank.a * tank.b * tank.h = 3.375 := by
  sorry

end max_tank_volume_l366_36661


namespace joan_gave_sam_seashells_l366_36670

/-- The number of seashells Joan gave to Sam -/
def seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) : ℕ :=
  initial_seashells - remaining_seashells

/-- Theorem: Joan gave Sam 43 seashells -/
theorem joan_gave_sam_seashells (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70)
  (h2 : remaining_seashells = 27) :
  seashells_given_to_sam initial_seashells remaining_seashells = 43 := by
  sorry

end joan_gave_sam_seashells_l366_36670


namespace portfolio_worth_calculation_l366_36610

/-- Calculates the final portfolio worth after two years given the initial investment,
    growth rates, and transactions. -/
def calculate_portfolio_worth (initial_investment : ℝ) 
                              (year1_growth_rate : ℝ) 
                              (year1_addition : ℝ) 
                              (year1_withdrawal : ℝ)
                              (year2_growth_rate1 : ℝ)
                              (year2_decline_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that given the specified conditions, 
    the final portfolio worth is approximately $115.59 -/
theorem portfolio_worth_calculation :
  let initial_investment : ℝ := 80
  let year1_growth_rate : ℝ := 0.15
  let year1_addition : ℝ := 28
  let year1_withdrawal : ℝ := 10
  let year2_growth_rate1 : ℝ := 0.10
  let year2_decline_rate : ℝ := 0.04
  
  abs (calculate_portfolio_worth initial_investment 
                                 year1_growth_rate
                                 year1_addition
                                 year1_withdrawal
                                 year2_growth_rate1
                                 year2_decline_rate - 115.59) < 0.01 := by
  sorry

end portfolio_worth_calculation_l366_36610


namespace school_event_water_drinkers_l366_36622

/-- Proves that 60 students chose water given the conditions of the school event -/
theorem school_event_water_drinkers (total : ℕ) (juice_percent soda_percent : ℚ) 
  (soda_count : ℕ) : 
  juice_percent = 1/2 →
  soda_percent = 3/10 →
  soda_count = 90 →
  total = soda_count / soda_percent →
  (1 - juice_percent - soda_percent) * total = 60 :=
by
  sorry

#check school_event_water_drinkers

end school_event_water_drinkers_l366_36622


namespace equation_solution_l366_36669

theorem equation_solution : 
  ∃ (x : ℝ), (4 * x - 5) / (5 * x - 10) = 3 / 4 ∧ x = -10 :=
by
  sorry

end equation_solution_l366_36669


namespace floor_equation_solution_l366_36686

theorem floor_equation_solution (x : ℚ) : 
  (⌊20 * x + 23⌋ = 20 + 23 * x) ↔ 
  (∃ k : ℕ, k ≤ 7 ∧ x = (23 - k : ℚ) / 23) :=
sorry

end floor_equation_solution_l366_36686


namespace dessert_preference_l366_36600

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) : 
  total = 50 → apple = 22 → chocolate = 20 → neither = 17 →
  apple + chocolate - (total - neither) = 9 := by
sorry

end dessert_preference_l366_36600


namespace circle_division_sum_integer_l366_36644

theorem circle_division_sum_integer :
  ∃ (a b c d e : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    ∃ (n : ℤ), (a / b : ℚ) + (b / c : ℚ) + (c / d : ℚ) + (d / e : ℚ) + (e / a : ℚ) = n :=
sorry

end circle_division_sum_integer_l366_36644


namespace total_books_count_l366_36683

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 8

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 5

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 4

/-- The total number of books -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem total_books_count : total_books = 72 := by
  sorry

end total_books_count_l366_36683


namespace composite_n4_plus_4_l366_36615

theorem composite_n4_plus_4 (n : ℕ) (h : n ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
sorry

end composite_n4_plus_4_l366_36615


namespace player_arrangement_count_l366_36690

/-- The number of ways to arrange players from three teams -/
def arrange_players (n : ℕ) : ℕ :=
  (n.factorial) * (n.factorial) * (n.factorial) * (n.factorial)

/-- Theorem: The number of ways to arrange 9 players from 3 teams is 1296 -/
theorem player_arrangement_count :
  arrange_players 3 = 1296 := by
  sorry

#eval arrange_players 3

end player_arrangement_count_l366_36690


namespace female_fraction_is_19_52_l366_36612

/-- Represents the chess club membership --/
structure ChessClub where
  males_last_year : ℕ
  total_increase_rate : ℚ
  male_increase_rate : ℚ
  female_increase_rate : ℚ

/-- Calculates the fraction of female participants in the chess club this year --/
def female_fraction (club : ChessClub) : ℚ :=
  sorry

/-- Theorem stating that the fraction of female participants is 19/52 --/
theorem female_fraction_is_19_52 (club : ChessClub) 
  (h1 : club.males_last_year = 30)
  (h2 : club.total_increase_rate = 15/100)
  (h3 : club.male_increase_rate = 10/100)
  (h4 : club.female_increase_rate = 25/100) :
  female_fraction club = 19/52 := by
  sorry

end female_fraction_is_19_52_l366_36612


namespace range_of_a_l366_36625

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ a ∧ x < 2 ↔ x < 2) ↔ a ≥ 2 := by
  sorry

end range_of_a_l366_36625


namespace symmetric_lines_b_value_l366_36697

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two lines are symmetric with respect to a given point -/
def are_symmetric (l1 l2 : Line) (p : Point) : Prop :=
  ∀ (x y : ℝ), l1.a * x + l1.b * y + l1.c = 0 →
    ∃ (x' y' : ℝ), l2.a * x' + l2.b * y' + l2.c = 0 ∧
      p.x = (x + x') / 2 ∧ p.y = (y + y') / 2

/-- The main theorem stating that given the conditions, b must equal 2 -/
theorem symmetric_lines_b_value :
  ∀ (a b : ℝ),
  let l1 : Line := ⟨1, 2, -3⟩
  let l2 : Line := ⟨a, 4, b⟩
  let p : Point := ⟨1, 0⟩
  are_symmetric l1 l2 p → b = 2 := by
  sorry

end symmetric_lines_b_value_l366_36697


namespace expected_democrat_votes_l366_36655

/-- Represents the percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.20

/-- Represents the total percentage of votes candidate A is expected to receive -/
def total_vote_percentage : ℝ := 0.53

/-- Represents the percentage of Democrat voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.75

theorem expected_democrat_votes :
  democrat_vote_percentage * democrat_percentage + 
  republican_vote_percentage * (1 - democrat_percentage) = 
  total_vote_percentage :=
sorry

end expected_democrat_votes_l366_36655


namespace f_monotonicity_and_range_l366_36643

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 5

theorem f_monotonicity_and_range :
  (∀ x y, -1 < x ∧ x < y → f x < f y) ∧
  (∀ x y, x < y ∧ y < -1 → f y < f x) ∧
  (∀ z ∈ Set.Icc 0 1, 5 ≤ f z ∧ f z ≤ Real.exp 1 + 5) ∧
  (f 0 = 5) ∧
  (f 1 = Real.exp 1 + 5) :=
sorry

end f_monotonicity_and_range_l366_36643


namespace plum_difference_l366_36656

def sharon_plums : ℕ := 7
def allan_plums : ℕ := 10

theorem plum_difference : allan_plums - sharon_plums = 3 := by
  sorry

end plum_difference_l366_36656


namespace inequality_solution_set_l366_36603

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 - x ≤ 0) ↔ (0 ≤ x ∧ x ≤ 1/2) := by sorry

end inequality_solution_set_l366_36603


namespace sets_in_borel_sigma_algebra_l366_36636

-- Define the type for infinite sequences of real numbers
def RealSequence := ℕ → ℝ

-- Define the Borel σ-algebra on ℝ^∞
def BorelSigmaAlgebra : Set (Set RealSequence) := sorry

-- Define the limsup of a sequence
def limsup (x : RealSequence) : ℝ := sorry

-- Define the limit of a sequence
def limit (x : RealSequence) : Option ℝ := sorry

-- Theorem statement
theorem sets_in_borel_sigma_algebra (a : ℝ) :
  {x : RealSequence | limsup x ≤ a} ∈ BorelSigmaAlgebra ∧
  {x : RealSequence | ∃ (l : ℝ), limit x = some l ∧ l > a} ∈ BorelSigmaAlgebra :=
sorry

end sets_in_borel_sigma_algebra_l366_36636


namespace eleanor_cookies_l366_36633

theorem eleanor_cookies (N : ℕ) : 
  N % 13 = 5 → N % 8 = 3 → N < 150 → N = 83 :=
by
  sorry

end eleanor_cookies_l366_36633


namespace H2O_formation_l366_36666

-- Define the molecules and their molar quantities
def HCl_moles : ℚ := 2
def CaCO3_moles : ℚ := 1

-- Define the balanced equation coefficients
def HCl_coeff : ℚ := 2
def CaCO3_coeff : ℚ := 1
def H2O_coeff : ℚ := 1

-- Define the function to calculate the amount of H2O formed
def H2O_formed (HCl : ℚ) (CaCO3 : ℚ) : ℚ :=
  min (HCl / HCl_coeff) (CaCO3 / CaCO3_coeff) * H2O_coeff

-- State the theorem
theorem H2O_formation :
  H2O_formed HCl_moles CaCO3_moles = 1 := by
  sorry

end H2O_formation_l366_36666


namespace remainder_98_power_50_mod_150_l366_36640

theorem remainder_98_power_50_mod_150 : 98^50 ≡ 74 [ZMOD 150] := by
  sorry

end remainder_98_power_50_mod_150_l366_36640


namespace problem_solution_l366_36647

theorem problem_solution (a b c : ℚ) 
  (sum_condition : a + b + c = 200)
  (equal_condition : a + 10 = b - 10 ∧ b - 10 = 10 * c) : 
  b = 2210 / 21 := by
  sorry

end problem_solution_l366_36647


namespace abs_a_plus_b_equals_three_minus_sqrt_two_l366_36657

theorem abs_a_plus_b_equals_three_minus_sqrt_two 
  (a b : ℝ) (h : Real.sqrt (2*a + 6) + |b - Real.sqrt 2| = 0) : 
  |a + b| = 3 - Real.sqrt 2 := by
  sorry

end abs_a_plus_b_equals_three_minus_sqrt_two_l366_36657


namespace min_sum_squares_distances_l366_36682

/-- An isosceles right triangle with leg length a -/
structure IsoscelesRightTriangle (a : ℝ) :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (legs_length : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = a ∧ B.2 = 0 ∧ C.1 = 0 ∧ C.2 = a)
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)

/-- The sum of squares of distances from a point to the vertices of the triangle -/
def sum_of_squares_distances (a : ℝ) (triangle : IsoscelesRightTriangle a) (point : ℝ × ℝ) : ℝ :=
  (point.1 - triangle.A.1)^2 + (point.2 - triangle.A.2)^2 +
  (point.1 - triangle.B.1)^2 + (point.2 - triangle.B.2)^2 +
  (point.1 - triangle.C.1)^2 + (point.2 - triangle.C.2)^2

/-- The theorem stating the minimum point and value -/
theorem min_sum_squares_distances (a : ℝ) (triangle : IsoscelesRightTriangle a) :
  ∃ (min_point : ℝ × ℝ),
    (∀ (point : ℝ × ℝ), sum_of_squares_distances a triangle min_point ≤ sum_of_squares_distances a triangle point) ∧
    min_point = (a/3, a/3) ∧
    sum_of_squares_distances a triangle min_point = (4*a^2)/3 :=
sorry

end min_sum_squares_distances_l366_36682


namespace total_weight_theorem_l366_36611

/-- The weight of the orange ring in ounces -/
def orange_ring_oz : ℚ := 1 / 12

/-- The weight of the purple ring in ounces -/
def purple_ring_oz : ℚ := 1 / 3

/-- The weight of the white ring in ounces -/
def white_ring_oz : ℚ := 5 / 12

/-- The weight of the blue ring in ounces -/
def blue_ring_oz : ℚ := 1 / 4

/-- The weight of the green ring in ounces -/
def green_ring_oz : ℚ := 1 / 6

/-- The weight of the red ring in ounces -/
def red_ring_oz : ℚ := 1 / 10

/-- The conversion factor from ounces to grams -/
def oz_to_g : ℚ := 28.3495

/-- The total weight of all rings in grams -/
def total_weight_g : ℚ :=
  (orange_ring_oz + purple_ring_oz + white_ring_oz + blue_ring_oz + green_ring_oz + red_ring_oz) * oz_to_g

theorem total_weight_theorem :
  total_weight_g = 38.271825 := by sorry

end total_weight_theorem_l366_36611


namespace max_k_inequality_l366_36614

theorem max_k_inequality (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∃ (k : ℝ), ∀ m, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) ∧ 
  (∀ k, (∀ m, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) → k ≤ 8) :=
sorry

end max_k_inequality_l366_36614


namespace extreme_value_and_tangent_line_l366_36677

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

/-- The derivative of f(x) with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

theorem extreme_value_and_tangent_line 
  (a : ℝ) 
  (h1 : f' a 3 = 0) -- f(x) has an extreme value at x = 3
  (h2 : f a 1 = 16) -- Point A(1,16) is on f(x)
  : 
  (∀ x, f a x = 2 * x^3 - 12 * x^2 + 18 * x + 8) ∧ 
  (f' a 1 = 0) := by 
  sorry

#check extreme_value_and_tangent_line

end extreme_value_and_tangent_line_l366_36677


namespace simplify_and_evaluate_l366_36674

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = 3) (h2 : b = -1) :
  (4 * a^2 * b - 5 * b^2) - 3 * (a^2 * b - 2 * b^2) = -8 := by
  sorry

end simplify_and_evaluate_l366_36674


namespace yellow_ball_probability_l366_36645

-- Define the number of black and yellow balls
def num_black_balls : ℕ := 4
def num_yellow_balls : ℕ := 6

-- Define the total number of balls
def total_balls : ℕ := num_black_balls + num_yellow_balls

-- Define the probability of drawing a yellow ball
def prob_yellow : ℚ := num_yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 3/5 := by
  sorry

end yellow_ball_probability_l366_36645


namespace largest_non_representable_integer_l366_36631

theorem largest_non_representable_integer : 
  (∀ n > 97, ∃ a b : ℕ, n = 8 * a + 15 * b) ∧ 
  (¬ ∃ a b : ℕ, 97 = 8 * a + 15 * b) := by
sorry

end largest_non_representable_integer_l366_36631


namespace apples_eaten_by_dog_l366_36668

theorem apples_eaten_by_dog (apples_on_tree : ℕ) (apples_on_ground : ℕ) (apples_remaining : ℕ) : 
  apples_on_tree = 5 → apples_on_ground = 8 → apples_remaining = 10 →
  apples_on_tree + apples_on_ground - apples_remaining = 3 := by
sorry

end apples_eaten_by_dog_l366_36668


namespace radical_conjugate_sum_product_l366_36691

theorem radical_conjugate_sum_product (a b : ℝ) :
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧
  (a + Real.sqrt b) * (a - Real.sqrt b) = 4 →
  a + b = -4 := by
sorry

end radical_conjugate_sum_product_l366_36691


namespace hummus_servings_thomas_hummus_servings_l366_36642

/-- Calculates the number of servings of hummus Thomas is making -/
theorem hummus_servings (recipe_cup : ℕ) (can_ounces : ℕ) (cup_ounces : ℕ) (cans_bought : ℕ) : ℕ :=
  let total_ounces := can_ounces * cans_bought
  let servings := total_ounces / cup_ounces
  servings

/-- Proves that Thomas is making 21 servings of hummus -/
theorem thomas_hummus_servings :
  hummus_servings 1 16 6 8 = 21 := by
  sorry

end hummus_servings_thomas_hummus_servings_l366_36642


namespace sum_equals_220_l366_36658

theorem sum_equals_220 : 145 + 33 + 29 + 13 = 220 := by
  sorry

end sum_equals_220_l366_36658


namespace vikki_take_home_pay_l366_36662

def weekly_pay_calculation (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : ℚ :=
  let total_earnings := hours_worked * hourly_rate
  let tax_deduction := total_earnings * tax_rate
  let insurance_deduction := total_earnings * insurance_rate
  let total_deductions := tax_deduction + insurance_deduction + union_dues
  total_earnings - total_deductions

theorem vikki_take_home_pay :
  weekly_pay_calculation 42 10 (20/100) (5/100) 5 = 310 := by
  sorry

end vikki_take_home_pay_l366_36662


namespace number_pyramid_result_l366_36617

theorem number_pyramid_result : 123456 * 9 + 7 = 1111111 := by
  sorry

end number_pyramid_result_l366_36617


namespace ellipse_a_plus_k_l366_36678

-- Define the ellipse
structure Ellipse where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  foci_correct : foci1 = (1, 2) ∧ foci2 = (1, 6)
  point_on_ellipse : point = (7, 4)
  equation : ∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    (x, y) ∈ {p : ℝ × ℝ | ∃ (t : ℝ), p.1 = h + a * Real.cos t ∧ p.2 = k + b * Real.sin t}

theorem ellipse_a_plus_k (e : Ellipse) : e.a + e.k = 10 := by
  sorry

#check ellipse_a_plus_k

end ellipse_a_plus_k_l366_36678


namespace similar_triangle_perimeter_l366_36687

theorem similar_triangle_perimeter (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 24) (h4 : b = 12) 
  (h5 : a = b * 2) (c : ℝ) (h6 : c = 30) :
  let scale := c / b
  let new_a := a * scale
  let new_b := b * scale
  2 * new_a + new_b = 150 := by sorry

end similar_triangle_perimeter_l366_36687


namespace probability_of_ANN9_l366_36607

/-- Represents the set of possible symbols for each position in the license plate --/
structure LicensePlateSymbols where
  vowels : Finset Char
  nonVowels : Finset Char
  digits : Finset Char

/-- Represents the rules for forming a license plate in Algebrica --/
structure LicensePlateRules where
  symbols : LicensePlateSymbols
  firstIsVowel : Char → Prop
  secondThirdAreIdenticalNonVowels : Char → Prop
  fourthIsDigit : Char → Prop

/-- Calculates the total number of possible license plates --/
def totalLicensePlates (rules : LicensePlateRules) : ℕ :=
  (rules.symbols.vowels.card) * (rules.symbols.nonVowels.card) * (rules.symbols.digits.card)

/-- Represents a specific license plate --/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char

/-- Checks if a license plate is valid according to the rules --/
def isValidLicensePlate (plate : LicensePlate) (rules : LicensePlateRules) : Prop :=
  rules.firstIsVowel plate.first ∧
  rules.secondThirdAreIdenticalNonVowels plate.second ∧
  plate.second = plate.third ∧
  rules.fourthIsDigit plate.fourth

/-- The main theorem to prove --/
theorem probability_of_ANN9 (rules : LicensePlateRules)
  (h_vowels : rules.symbols.vowels.card = 5)
  (h_nonVowels : rules.symbols.nonVowels.card = 21)
  (h_digits : rules.symbols.digits.card = 10)
  (plate : LicensePlate)
  (h_plate : plate = ⟨'A', 'N', 'N', '9'⟩)
  (h_valid : isValidLicensePlate plate rules) :
  (1 : ℚ) / (totalLicensePlates rules : ℚ) = 1 / 1050 :=
sorry

end probability_of_ANN9_l366_36607


namespace tangent_length_to_circle_l366_36616

/-- The length of the tangent segment from the origin to the circle passing through 
    the points (2,3), (4,6), and (3,9) is 3√5. -/
theorem tangent_length_to_circle : 
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (4, 6)
  let C : ℝ × ℝ := (3, 9)
  let O : ℝ × ℝ := (0, 0)
  ∃ (circle : Set (ℝ × ℝ)) (T : ℝ × ℝ),
    A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧
    T ∈ circle ∧
    (∀ P ∈ circle, dist O P ≥ dist O T) ∧
    dist O T = 3 * Real.sqrt 5 :=
by sorry

end tangent_length_to_circle_l366_36616


namespace largest_power_dividing_factorial_l366_36660

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  let n := 2006
  ∃ k : ℕ, k = 34 ∧
    (∀ m : ℕ, n^m ∣ factorial n → m ≤ k) ∧
    n^k ∣ factorial n ∧
    n = 2 * 17 * 59 :=
by sorry

end largest_power_dividing_factorial_l366_36660


namespace magical_tree_properties_l366_36646

structure FruitTree :=
  (bananas : Nat)
  (oranges : Nat)

inductive PickAction
  | PickOneBanana
  | PickOneOrange
  | PickTwoBananas
  | PickTwoOranges
  | PickBananaAndOrange

def applyAction (tree : FruitTree) (action : PickAction) : FruitTree :=
  match action with
  | PickAction.PickOneBanana => tree
  | PickAction.PickOneOrange => tree
  | PickAction.PickTwoBananas => 
      if tree.bananas ≥ 2 then { bananas := tree.bananas - 2, oranges := tree.oranges + 1 }
      else tree
  | PickAction.PickTwoOranges => 
      if tree.oranges ≥ 2 then { bananas := tree.bananas, oranges := tree.oranges - 1 }
      else tree
  | PickAction.PickBananaAndOrange =>
      if tree.bananas ≥ 1 && tree.oranges ≥ 1 then
        { bananas := tree.bananas, oranges := tree.oranges - 1 }
      else tree

def initialTree : FruitTree := { bananas := 15, oranges := 20 }

theorem magical_tree_properties :
  -- 1. It's possible to reach a state with exactly one fruit
  (∃ (actions : List PickAction), (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges = 1) ∧
  -- 2. If there's only one fruit left, it must be a banana
  (∀ (actions : List PickAction), 
    (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges = 1 →
    (actions.foldl applyAction initialTree).bananas = 1) ∧
  -- 3. It's impossible to reach a state with no fruits
  (∀ (actions : List PickAction), 
    (actions.foldl applyAction initialTree).bananas + (actions.foldl applyAction initialTree).oranges > 0) :=
by
  sorry

end magical_tree_properties_l366_36646


namespace even_function_property_l366_36685

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_positive : ∀ x > 0, f x = x) :
  ∀ x < 0, f x = -x :=
by sorry

end even_function_property_l366_36685
