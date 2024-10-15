import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_area_example_l3287_328729

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 10 meters and height 7 meters is 70 square meters -/
theorem parallelogram_area_example : parallelogram_area 10 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l3287_328729


namespace NUMINAMATH_CALUDE_intersection_point_l3287_328734

/-- The slope of the given line -/
def m : ℝ := 2

/-- The y-intercept of the given line -/
def b : ℝ := 5

/-- The x-coordinate of the point on the perpendicular line -/
def x₀ : ℝ := 5

/-- The y-coordinate of the point on the perpendicular line -/
def y₀ : ℝ := 5

/-- The x-coordinate of the claimed intersection point -/
def x_int : ℝ := 1

/-- The y-coordinate of the claimed intersection point -/
def y_int : ℝ := 7

/-- Theorem stating that (x_int, y_int) is the intersection point of the given line
    and its perpendicular line passing through (x₀, y₀) -/
theorem intersection_point :
  (y_int = m * x_int + b) ∧
  (y_int - y₀ = -(1/m) * (x_int - x₀)) ∧
  (∀ x y : ℝ, (y = m * x + b) ∧ (y - y₀ = -(1/m) * (x - x₀)) → x = x_int ∧ y = y_int) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l3287_328734


namespace NUMINAMATH_CALUDE_price_difference_l3287_328763

theorem price_difference (P : ℝ) (h : P > 0) :
  let new_price := P * 1.2
  let discounted_price := new_price * 0.8
  new_price - discounted_price = P * 0.24 := by
sorry

end NUMINAMATH_CALUDE_price_difference_l3287_328763


namespace NUMINAMATH_CALUDE_fish_value_in_dragon_scales_l3287_328742

/-- In a magical kingdom with given exchange rates, prove the value of a fish in dragon scales -/
theorem fish_value_in_dragon_scales 
  (fish_to_bread : ℚ) -- Exchange rate of fish to bread
  (bread_to_scales : ℚ) -- Exchange rate of bread to dragon scales
  (h1 : 2 * fish_to_bread = 3) -- Two fish can be exchanged for three loaves of bread
  (h2 : bread_to_scales = 2) -- One loaf of bread can be traded for two dragon scales
  : fish_to_bread * bread_to_scales = 3 := by sorry

end NUMINAMATH_CALUDE_fish_value_in_dragon_scales_l3287_328742


namespace NUMINAMATH_CALUDE_sphere_in_cone_l3287_328718

theorem sphere_in_cone (b d g : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * Real.sqrt d - g
  g = b + 6 →
  sphere_radius = (cone_height * cone_base_radius) / (cone_base_radius + Real.sqrt (cone_base_radius^2 + cone_height^2)) →
  b + d = 12.5 := by
sorry

end NUMINAMATH_CALUDE_sphere_in_cone_l3287_328718


namespace NUMINAMATH_CALUDE_average_score_proof_l3287_328782

def student_A_score : ℚ := 92
def student_B_score : ℚ := 75
def student_C_score : ℚ := 98

def number_of_students : ℚ := 3

def average_score : ℚ := (student_A_score + student_B_score + student_C_score) / number_of_students

theorem average_score_proof : average_score = 88.3333333333333 := by
  sorry

end NUMINAMATH_CALUDE_average_score_proof_l3287_328782


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3287_328768

theorem complex_product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := 1 - 2 * Complex.I
  let z₂ : ℂ := a + Complex.I
  (∃ (b : ℝ), z₁ * z₂ = b * Complex.I ∧ b ≠ 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l3287_328768


namespace NUMINAMATH_CALUDE_project_budget_equality_l3287_328728

/-- Represents the annual budget change for a project -/
structure BudgetChange where
  initial : ℕ  -- Initial budget in dollars
  annual : ℤ   -- Annual change in dollars (positive for increase, negative for decrease)

/-- Calculates the budget after a given number of years -/
def budget_after_years (bc : BudgetChange) (years : ℕ) : ℤ :=
  bc.initial + years * bc.annual

/-- The problem statement -/
theorem project_budget_equality (q v : BudgetChange) 
  (hq_initial : q.initial = 540000)
  (hv_initial : v.initial = 780000)
  (hq_annual : q.annual = 30000)
  (h_equal_after_4 : budget_after_years q 4 = budget_after_years v 4) :
  v.annual = -30000 := by
  sorry

end NUMINAMATH_CALUDE_project_budget_equality_l3287_328728


namespace NUMINAMATH_CALUDE_smallest_among_four_rationals_l3287_328739

theorem smallest_among_four_rationals :
  let a : ℚ := -2/3
  let b : ℚ := -1
  let c : ℚ := 0
  let d : ℚ := 1
  b < a ∧ b < c ∧ b < d := by sorry

end NUMINAMATH_CALUDE_smallest_among_four_rationals_l3287_328739


namespace NUMINAMATH_CALUDE_units_digit_of_17_times_27_l3287_328702

theorem units_digit_of_17_times_27 : (17 * 27) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_times_27_l3287_328702


namespace NUMINAMATH_CALUDE_bookstore_sales_ratio_l3287_328754

theorem bookstore_sales_ratio :
  let tuesday_sales : ℕ := 7
  let wednesday_sales : ℕ := 3 * tuesday_sales
  let total_sales : ℕ := 91
  let thursday_sales : ℕ := total_sales - (tuesday_sales + wednesday_sales)
  (thursday_sales : ℚ) / wednesday_sales = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_bookstore_sales_ratio_l3287_328754


namespace NUMINAMATH_CALUDE_sum_of_cosines_l3287_328744

theorem sum_of_cosines (z : ℂ) (α : ℝ) (h1 : z^7 = 1) (h2 : z ≠ 1) (h3 : z = Complex.exp (Complex.I * α)) :
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cosines_l3287_328744


namespace NUMINAMATH_CALUDE_derivative_of_composite_l3287_328788

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f'(3) = 9
variable (h : deriv f 3 = 9)

-- State the theorem
theorem derivative_of_composite (f : ℝ → ℝ) (h : deriv f 3 = 9) :
  deriv (fun x ↦ f (3 * x^2)) 1 = 54 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_composite_l3287_328788


namespace NUMINAMATH_CALUDE_square_of_linear_expression_l3287_328715

theorem square_of_linear_expression (x : ℝ) :
  x = -2 → (3*x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_linear_expression_l3287_328715


namespace NUMINAMATH_CALUDE_no_fermat_solutions_with_constraints_l3287_328725

theorem no_fermat_solutions_with_constraints (n : ℕ) (hn : n > 1) :
  ¬∃ (x y z : ℕ), x^n + y^n = z^n ∧ x ≤ n ∧ y ≤ n := by
  sorry

end NUMINAMATH_CALUDE_no_fermat_solutions_with_constraints_l3287_328725


namespace NUMINAMATH_CALUDE_triangle_side_length_l3287_328735

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  A = π / 3 →  -- 60 degrees in radians
  B = π / 4 →  -- 45 degrees in radians
  b = Real.sqrt 6 →
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3287_328735


namespace NUMINAMATH_CALUDE_base_3_division_theorem_l3287_328786

def base_3_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (3 ^ i)) 0

def decimal_to_base_3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

theorem base_3_division_theorem :
  let dividend := [1, 0, 2, 1]  -- 1021₃ in reverse order
  let divisor := [1, 1]         -- 11₃ in reverse order
  let quotient := [2, 2]        -- 22₃ in reverse order
  (base_3_to_decimal dividend) / (base_3_to_decimal divisor) = base_3_to_decimal quotient :=
by sorry

end NUMINAMATH_CALUDE_base_3_division_theorem_l3287_328786


namespace NUMINAMATH_CALUDE_number_operations_l3287_328764

theorem number_operations (x : ℝ) : ((x + 5) * 5 - 5) / 5 = 5 ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l3287_328764


namespace NUMINAMATH_CALUDE_cannot_visit_all_friends_l3287_328722

-- Define the building structure
structure Building where
  num_floors : ℕ
  start_floor : ℕ
  friend_floors : List ℕ
  elevator_moves : List ℕ

-- Define the problem specifics
def problem : Building :=
  { num_floors := 14
  , start_floor := 1
  , friend_floors := [12, 14]
  , elevator_moves := [3, 7]
  }

-- Define a single elevator trip
def elevator_trip (current : ℤ) (move : ℤ) : ℤ :=
  current + move

-- Define if a floor is reachable within given moves
def is_reachable (building : Building) (target : ℕ) (max_moves : ℕ) : Prop :=
  ∃ (moves : List ℤ),
    moves.length ≤ max_moves ∧
    moves.all (λ m => m.natAbs ∈ building.elevator_moves) ∧
    (moves.foldl elevator_trip building.start_floor : ℤ) = target

-- Theorem statement
theorem cannot_visit_all_friends :
  ¬∃ (moves : List ℤ),
    moves.length ≤ 6 ∧
    moves.all (λ m => m.natAbs ∈ problem.elevator_moves) ∧
    (∀ floor ∈ problem.friend_floors,
      ∃ (submoves : List ℤ),
        submoves ⊆ moves ∧
        (submoves.foldl elevator_trip problem.start_floor : ℤ) = floor) :=
sorry

end NUMINAMATH_CALUDE_cannot_visit_all_friends_l3287_328722


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3287_328721

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3287_328721


namespace NUMINAMATH_CALUDE_horatio_sonnets_l3287_328787

def sonnet_lines : ℕ := 16
def sonnets_read : ℕ := 9
def unread_lines : ℕ := 126

theorem horatio_sonnets :
  ∃ (total_sonnets : ℕ),
    total_sonnets * sonnet_lines = sonnets_read * sonnet_lines + unread_lines ∧
    total_sonnets = 16 := by
  sorry

end NUMINAMATH_CALUDE_horatio_sonnets_l3287_328787


namespace NUMINAMATH_CALUDE_angle_ratio_equality_l3287_328737

-- Define a triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define a point P inside the triangle
def PointInside (t : Triangle) (P : Point) : Prop := sorry

-- Define angle measure
def AngleMeasure (A B C : Point) : ℝ := sorry

-- Theorem statement
theorem angle_ratio_equality (t : Triangle) (P : Point) (x : ℝ) 
  (h_inside : PointInside t P)
  (h_ratio_AB_AC : AngleMeasure t.A P t.B / AngleMeasure t.A P t.C = x)
  (h_ratio_CA_CB : AngleMeasure t.C P t.A / AngleMeasure t.C P t.B = x)
  (h_ratio_BC_BA : AngleMeasure t.B P t.C / AngleMeasure t.B P t.A = x) :
  x = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_ratio_equality_l3287_328737


namespace NUMINAMATH_CALUDE_cost_price_is_four_l3287_328746

/-- The cost price of a bag of popcorn -/
def cost_price : ℝ := sorry

/-- The selling price of a bag of popcorn -/
def selling_price : ℝ := 8

/-- The number of bags sold -/
def bags_sold : ℝ := 30

/-- The total profit -/
def total_profit : ℝ := 120

/-- Theorem: The cost price of each bag of popcorn is $4 -/
theorem cost_price_is_four :
  cost_price = 4 :=
by
  have h1 : total_profit = bags_sold * (selling_price - cost_price) :=
    sorry
  sorry

end NUMINAMATH_CALUDE_cost_price_is_four_l3287_328746


namespace NUMINAMATH_CALUDE_milkshake_ice_cream_difference_l3287_328758

/-- Given the number of milkshakes and ice cream cones sold, prove the difference -/
theorem milkshake_ice_cream_difference (milkshakes ice_cream_cones : ℕ) 
  (h1 : milkshakes = 82) (h2 : ice_cream_cones = 67) : 
  milkshakes - ice_cream_cones = 15 := by
  sorry

end NUMINAMATH_CALUDE_milkshake_ice_cream_difference_l3287_328758


namespace NUMINAMATH_CALUDE_room_ratios_l3287_328759

/-- Represents a rectangular room with given length and width. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Represents a ratio as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem room_ratios (room : Rectangle) 
    (h1 : room.length = 24) 
    (h2 : room.width = 14) : 
    (∃ r1 : Ratio, r1.numerator = 6 ∧ r1.denominator = 19 ∧ 
      r1.numerator * perimeter room = r1.denominator * room.length) ∧
    (∃ r2 : Ratio, r2.numerator = 7 ∧ r2.denominator = 38 ∧ 
      r2.numerator * perimeter room = r2.denominator * room.width) := by
  sorry


end NUMINAMATH_CALUDE_room_ratios_l3287_328759


namespace NUMINAMATH_CALUDE_stock_price_return_l3287_328751

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) : 
  let increased_price := initial_price * 1.3
  let decrease_percentage := (1 - 1 / 1.3) * 100
  increased_price * (1 - decrease_percentage / 100) = initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_return_l3287_328751


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l3287_328724

theorem percentage_of_male_employees (total_employees : ℕ) 
  (males_below_50 : ℕ) (h1 : total_employees = 1800) 
  (h2 : males_below_50 = 756) : 
  (males_below_50 : ℝ) / (0.7 * total_employees) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l3287_328724


namespace NUMINAMATH_CALUDE_desired_circle_satisfies_conditions_l3287_328793

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The equation of the line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

/-- Theorem stating that the desired circle satisfies all conditions -/
theorem desired_circle_satisfies_conditions :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    (∃ (h k : ℝ), centerLine h k ∧ 
      ∀ (x y : ℝ), desiredCircle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 11)) :=
sorry

end NUMINAMATH_CALUDE_desired_circle_satisfies_conditions_l3287_328793


namespace NUMINAMATH_CALUDE_initial_deposit_l3287_328730

theorem initial_deposit (P R : ℝ) : 
  P + (P * R * 3) / 100 = 9200 →
  P + (P * (R + 0.5) * 3) / 100 = 9320 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_initial_deposit_l3287_328730


namespace NUMINAMATH_CALUDE_even_digits_finite_fissile_squares_odd_digits_infinite_fissile_squares_l3287_328711

/-- A fissile square is a positive integer which is a perfect square,
    and whose digits form two perfect squares in a row. -/
def is_fissile_square (n : ℕ) : Prop :=
  ∃ (x y r : ℕ) (d : ℕ), 
    n = x^2 ∧ 
    n = 10^d * y^2 + r^2 ∧ 
    y^2 ≠ 0 ∧ r^2 ≠ 0

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Every square with an even number of digits is the right square of only finitely many fissile squares -/
theorem even_digits_finite_fissile_squares (r : ℕ) (h : Even (num_digits (r^2))) :
  {x : ℕ | is_fissile_square (x^2) ∧ ∃ (y : ℕ) (d : ℕ), x^2 = 10^d * y^2 + r^2 ∧ Even d}.Finite :=
sorry

/-- Theorem: Every square with an odd number of digits is the right square of infinitely many fissile squares -/
theorem odd_digits_infinite_fissile_squares (r : ℕ) (h : Odd (num_digits (r^2))) :
  {x : ℕ | is_fissile_square (x^2) ∧ ∃ (y : ℕ) (d : ℕ), x^2 = 10^d * y^2 + r^2 ∧ Odd d}.Infinite :=
sorry

end NUMINAMATH_CALUDE_even_digits_finite_fissile_squares_odd_digits_infinite_fissile_squares_l3287_328711


namespace NUMINAMATH_CALUDE_parabola_properties_l3287_328780

/-- Given a parabola with equation x² = 8y, this theorem proves the equation of its directrix
    and the coordinates of its focus. -/
theorem parabola_properties (x y : ℝ) :
  x^2 = 8*y →
  (∃ (directrix : ℝ → Prop) (focus : ℝ × ℝ),
    directrix = λ y' => y' = -2 ∧
    focus = (0, 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3287_328780


namespace NUMINAMATH_CALUDE_sum_of_three_smallest_primes_l3287_328797

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def primes_between_1_and_50 : Set ℕ := {n : ℕ | 1 < n ∧ n ≤ 50 ∧ is_prime n}

theorem sum_of_three_smallest_primes :
  ∃ (a b c : ℕ), a ∈ primes_between_1_and_50 ∧
                 b ∈ primes_between_1_and_50 ∧
                 c ∈ primes_between_1_and_50 ∧
                 a < b ∧ b < c ∧
                 (∀ p ∈ primes_between_1_and_50, p ≥ c ∨ p = a ∨ p = b) ∧
                 a + b + c = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_smallest_primes_l3287_328797


namespace NUMINAMATH_CALUDE_average_price_is_18_l3287_328791

/-- The average price per book given two book purchases -/
def average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price per book is 18 for the given purchases -/
theorem average_price_is_18 :
  average_price_per_book 65 50 1150 920 = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_price_is_18_l3287_328791


namespace NUMINAMATH_CALUDE_correct_calculation_l3287_328743

theorem correct_calculation (x : ℤ) : 66 + x = 93 → (66 - x) + 21 = 60 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3287_328743


namespace NUMINAMATH_CALUDE_set_equality_l3287_328753

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define sets M and N
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {1, 3, 6}

-- Theorem statement
theorem set_equality : (U \ M) ∩ N = {1, 6} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3287_328753


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l3287_328757

/-- Calculates the rate of mixed oil per litre given volumes and prices of different oils. -/
theorem mixed_oil_rate (v1 v2 v3 v4 : ℚ) (p1 p2 p3 p4 : ℚ) :
  v1 = 10 ∧ v2 = 5 ∧ v3 = 3 ∧ v4 = 2 ∧
  p1 = 50 ∧ p2 = 66 ∧ p3 = 75 ∧ p4 = 85 →
  (v1 * p1 + v2 * p2 + v3 * p3 + v4 * p4) / (v1 + v2 + v3 + v4) = 61.25 := by
  sorry

#eval (10 * 50 + 5 * 66 + 3 * 75 + 2 * 85) / (10 + 5 + 3 + 2)

end NUMINAMATH_CALUDE_mixed_oil_rate_l3287_328757


namespace NUMINAMATH_CALUDE_histogram_group_width_l3287_328750

/-- Represents a group in a frequency histogram -/
structure HistogramGroup where
  a : ℝ
  b : ℝ
  m : ℝ  -- frequency
  h : ℝ  -- height
  h_pos : h > 0

/-- Theorem: The width of a histogram group is equal to its frequency divided by its height -/
theorem histogram_group_width (g : HistogramGroup) : |g.a - g.b| = g.m / g.h := by
  sorry

end NUMINAMATH_CALUDE_histogram_group_width_l3287_328750


namespace NUMINAMATH_CALUDE_sqrt_31_plus_3_tan_56_approx_7_l3287_328798

/-- Prove that the absolute difference between √31 + 3tan(56°) and 7.00 is less than 0.005 -/
theorem sqrt_31_plus_3_tan_56_approx_7 :
  |Real.sqrt 31 + 3 * Real.tan (56 * π / 180) - 7| < 0.005 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_31_plus_3_tan_56_approx_7_l3287_328798


namespace NUMINAMATH_CALUDE_expression_evaluation_l3287_328784

theorem expression_evaluation : 86 + (144 / 12) + (15 * 13) - 300 - (480 / 8) = -67 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3287_328784


namespace NUMINAMATH_CALUDE_emily_beads_count_l3287_328775

theorem emily_beads_count (necklaces : ℕ) (beads_per_necklace : ℕ) 
  (h1 : necklaces = 11) (h2 : beads_per_necklace = 28) : 
  necklaces * beads_per_necklace = 308 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l3287_328775


namespace NUMINAMATH_CALUDE_number_of_schools_l3287_328745

def students_per_school : ℕ := 247
def total_students : ℕ := 6175

theorem number_of_schools : (total_students / students_per_school : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_of_schools_l3287_328745


namespace NUMINAMATH_CALUDE_player_playing_time_l3287_328783

/-- Calculates the playing time for each player in a sports tournament. -/
theorem player_playing_time (total_players : ℕ) (players_on_field : ℕ) (match_duration : ℕ) :
  total_players = 10 →
  players_on_field = 8 →
  match_duration = 45 →
  (players_on_field * match_duration) / total_players = 36 := by
  sorry

end NUMINAMATH_CALUDE_player_playing_time_l3287_328783


namespace NUMINAMATH_CALUDE_house_ratio_l3287_328769

theorem house_ratio (houses_one_side : ℕ) (total_houses : ℕ) : 
  houses_one_side = 40 → 
  total_houses = 160 → 
  (total_houses - houses_one_side) / houses_one_side = 3 := by
sorry

end NUMINAMATH_CALUDE_house_ratio_l3287_328769


namespace NUMINAMATH_CALUDE_park_attendance_solution_l3287_328740

/-- Represents the number of people at Minewaska State Park --/
structure ParkAttendance where
  hikers : ℕ
  bikers : ℕ
  kayakers : ℕ

/-- The conditions of the park attendance problem --/
def parkProblem (p : ParkAttendance) : Prop :=
  p.hikers = p.bikers + 178 ∧
  p.kayakers * 2 = p.bikers ∧
  p.hikers + p.bikers + p.kayakers = 920

/-- The theorem stating the solution to the park attendance problem --/
theorem park_attendance_solution :
  ∃ p : ParkAttendance, parkProblem p ∧ p.hikers = 474 := by
  sorry

end NUMINAMATH_CALUDE_park_attendance_solution_l3287_328740


namespace NUMINAMATH_CALUDE_monday_count_l3287_328731

/-- The number of Mondays on which it rained -/
def num_mondays : ℕ := sorry

/-- The rainfall on each Monday in centimeters -/
def rainfall_per_monday : ℚ := 3/2

/-- The number of Tuesdays on which it rained -/
def num_tuesdays : ℕ := 9

/-- The rainfall on each Tuesday in centimeters -/
def rainfall_per_tuesday : ℚ := 5/2

/-- The difference in total rainfall between Tuesdays and Mondays in centimeters -/
def rainfall_difference : ℚ := 12

theorem monday_count : 
  num_mondays * rainfall_per_monday + rainfall_difference = 
  num_tuesdays * rainfall_per_tuesday ∧ num_mondays = 7 := by sorry

end NUMINAMATH_CALUDE_monday_count_l3287_328731


namespace NUMINAMATH_CALUDE_grid_value_bound_l3287_328752

/-- The value of a square in the grid -/
def square_value (is_filled : Bool) (filled_neighbors : Nat) : Nat :=
  if is_filled then 0 else filled_neighbors

/-- The maximum number of neighbors a square can have -/
def max_neighbors : Nat := 8

/-- The function f(m,n) representing the largest total value of squares in the grid -/
noncomputable def f (m n : Nat) : Nat :=
  sorry  -- Definition of f(m,n) is complex and depends on optimal grid configuration

/-- The theorem stating that 2 is the minimal constant C such that f(m,n) / (m*n) ≤ C -/
theorem grid_value_bound (m n : Nat) (hm : m > 0) (hn : n > 0) :
  (f m n : ℝ) / (m * n : ℝ) ≤ 2 ∧ ∀ C : ℝ, (∀ m' n' : Nat, m' > 0 → n' > 0 → (f m' n' : ℝ) / (m' * n' : ℝ) ≤ C) → C ≥ 2 :=
  sorry


end NUMINAMATH_CALUDE_grid_value_bound_l3287_328752


namespace NUMINAMATH_CALUDE_divisibility_reversal_implies_factor_of_99_l3287_328778

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisibility_reversal_implies_factor_of_99 (k : ℕ) :
  (∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) →
  k ∣ 99 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_reversal_implies_factor_of_99_l3287_328778


namespace NUMINAMATH_CALUDE_journey_gas_cost_l3287_328712

/-- Calculates the cost of gas for a journey given odometer readings, fuel efficiency, and gas price -/
def gas_cost (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  ((final_reading - initial_reading : ℚ) / fuel_efficiency) * gas_price

/-- Proves that the cost of gas for the given journey is $3.46 -/
theorem journey_gas_cost :
  gas_cost 85340 85368 32 (395/100) = 346/100 := by
  sorry

end NUMINAMATH_CALUDE_journey_gas_cost_l3287_328712


namespace NUMINAMATH_CALUDE_not_perfect_square_l3287_328760

theorem not_perfect_square (a b : ℕ+) : ¬∃ k : ℤ, (a : ℤ)^2 + Int.ceil ((4 * (a : ℤ)^2) / (b : ℤ)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3287_328760


namespace NUMINAMATH_CALUDE_fermat_number_prime_factor_l3287_328748

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_number_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ F n ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_prime_factor_l3287_328748


namespace NUMINAMATH_CALUDE_windows_preference_l3287_328708

theorem windows_preference (total_students : ℕ) (mac_preference : ℕ) (no_preference : ℕ) 
  (h1 : total_students = 210)
  (h2 : mac_preference = 60)
  (h3 : no_preference = 90) :
  total_students - (mac_preference + mac_preference / 3 + no_preference) = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_l3287_328708


namespace NUMINAMATH_CALUDE_polynomial_roots_problem_l3287_328790

/-- 
Given two real numbers u and v that are roots of the polynomial r(x) = x^3 + cx + d,
and u+3 and v-2 are roots of another polynomial s(x) = x^3 + cx + d + 153,
prove that the only possible value for d is 0.
-/
theorem polynomial_roots_problem (u v c d : ℝ) : 
  (u^3 + c*u + d = 0) →
  (v^3 + c*v + d = 0) →
  ((u+3)^3 + c*(u+3) + d + 153 = 0) →
  ((v-2)^3 + c*(v-2) + d + 153 = 0) →
  d = 0 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_roots_problem_l3287_328790


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l3287_328704

theorem foreign_stamps_count (total : ℕ) (old : ℕ) (foreign_and_old : ℕ) (neither : ℕ) :
  total = 200 →
  old = 60 →
  foreign_and_old = 20 →
  neither = 70 →
  ∃ foreign : ℕ, foreign = 90 ∧ 
    foreign + old - foreign_and_old = total - neither :=
by sorry

end NUMINAMATH_CALUDE_foreign_stamps_count_l3287_328704


namespace NUMINAMATH_CALUDE_special_divisor_form_l3287_328747

/-- A function that checks if a number is of the form a^r + 1 --/
def isOfForm (d : ℕ) : Prop :=
  ∃ (a r : ℕ), a > 0 ∧ r > 1 ∧ d = a^r + 1

/-- The main theorem --/
theorem special_divisor_form (n : ℕ) :
  n > 1 ∧ (∀ d : ℕ, 1 < d ∧ d ∣ n → isOfForm d) →
  n = 10 ∨ ∃ a : ℕ, n = a^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_special_divisor_form_l3287_328747


namespace NUMINAMATH_CALUDE_smallest_circle_area_l3287_328723

/-- The smallest area of a circle passing through two given points -/
theorem smallest_circle_area (x₁ y₁ x₂ y₂ : ℝ) : 
  let d := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  let r := d / 2
  let A := π * r^2
  x₁ = -3 ∧ y₁ = -2 ∧ x₂ = 2 ∧ y₂ = 4 →
  A = (61 * π) / 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_circle_area_l3287_328723


namespace NUMINAMATH_CALUDE_special_numbers_exist_l3287_328795

theorem special_numbers_exist : ∃ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (¬(3 ∣ a) ∧ ¬(4 ∣ a) ∧ ¬(5 ∣ a)) ∧
  (¬(3 ∣ b) ∧ ¬(4 ∣ b) ∧ ¬(5 ∣ b)) ∧
  (¬(3 ∣ c) ∧ ¬(4 ∣ c) ∧ ¬(5 ∣ c)) ∧
  (¬(3 ∣ d) ∧ ¬(4 ∣ d) ∧ ¬(5 ∣ d)) ∧
  (¬(3 ∣ e) ∧ ¬(4 ∣ e) ∧ ¬(5 ∣ e)) ∧
  (3 ∣ (a + b + c)) ∧ (3 ∣ (a + b + d)) ∧ (3 ∣ (a + b + e)) ∧
  (3 ∣ (a + c + d)) ∧ (3 ∣ (a + c + e)) ∧ (3 ∣ (a + d + e)) ∧
  (3 ∣ (b + c + d)) ∧ (3 ∣ (b + c + e)) ∧ (3 ∣ (b + d + e)) ∧
  (3 ∣ (c + d + e)) ∧
  (4 ∣ (a + b + c + d)) ∧ (4 ∣ (a + b + c + e)) ∧
  (4 ∣ (a + b + d + e)) ∧ (4 ∣ (a + c + d + e)) ∧
  (4 ∣ (b + c + d + e)) ∧
  (5 ∣ (a + b + c + d + e)) := by
sorry

end NUMINAMATH_CALUDE_special_numbers_exist_l3287_328795


namespace NUMINAMATH_CALUDE_largest_expression_l3287_328700

-- Define the expressions
def expr_A : ℝ := (7 * 8) ^ (1/6)
def expr_B : ℝ := (8 * 7^(1/3))^(1/2)
def expr_C : ℝ := (7 * 8^(1/3))^(1/2)
def expr_D : ℝ := (7 * 8^(1/2))^(1/3)
def expr_E : ℝ := (8 * 7^(1/2))^(1/3)

-- Theorem statement
theorem largest_expression :
  expr_B = max expr_A (max expr_B (max expr_C (max expr_D expr_E))) :=
by sorry

end NUMINAMATH_CALUDE_largest_expression_l3287_328700


namespace NUMINAMATH_CALUDE_product_simplification_l3287_328762

theorem product_simplification (y : ℝ) : (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3287_328762


namespace NUMINAMATH_CALUDE_special_isosceles_triangle_base_length_l3287_328717

/-- An isosceles triangle with side length a and a specific height property -/
structure SpecialIsoscelesTriangle (a : ℝ) where
  -- The triangle is isosceles with side length a
  side_length : ℝ
  is_isosceles : side_length = a
  -- The height dropped onto the base is equal to the segment connecting
  -- the midpoint of the base with the midpoint of the side
  height_property : ℝ → Prop

/-- The base length of the special isosceles triangle is a√3 -/
theorem special_isosceles_triangle_base_length 
  {a : ℝ} (t : SpecialIsoscelesTriangle a) : 
  ∃ (base : ℝ), base = a * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_special_isosceles_triangle_base_length_l3287_328717


namespace NUMINAMATH_CALUDE_polynomial_value_at_two_l3287_328710

theorem polynomial_value_at_two :
  let f : ℝ → ℝ := fun x ↦ x^2 - 3*x + 2
  f 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_two_l3287_328710


namespace NUMINAMATH_CALUDE_order_of_numbers_l3287_328796

theorem order_of_numbers (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) :
  a > -b ∧ -b > b ∧ b > -a := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3287_328796


namespace NUMINAMATH_CALUDE_range_of_a_min_value_of_g_l3287_328719

-- Define the quadratic function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + a

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * f a x - a^2 * (x + 1) - 2*x

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧
   f a x₁ = x₁ ∧ f a x₂ = x₂) →
  0 < a ∧ a < 3 - 2 * Real.sqrt 2 :=
sorry

-- Theorem for the minimum value of g
theorem min_value_of_g (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, 
    (a < 1 → g a x ≥ a - 2) ∧
    (a ≥ 1 → g a x ≥ -1/a)) ∧
  (∃ x ∈ Set.Icc 0 1, 
    (a < 1 → g a x = a - 2) ∧
    (a ≥ 1 → g a x = -1/a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_min_value_of_g_l3287_328719


namespace NUMINAMATH_CALUDE_pet_shop_birds_l3287_328732

theorem pet_shop_birds (total : ℕ) (kittens : ℕ) (hamsters : ℕ) (birds : ℕ) : 
  total = 77 → kittens = 32 → hamsters = 15 → birds = total - kittens - hamsters → birds = 30 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_birds_l3287_328732


namespace NUMINAMATH_CALUDE_largest_integer_less_than_sqrt5_plus_sqrt3_to_6th_l3287_328785

theorem largest_integer_less_than_sqrt5_plus_sqrt3_to_6th (n : ℕ) : 
  n = 3322 ↔ n = ⌊(Real.sqrt 5 + Real.sqrt 3)^6⌋ :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_sqrt5_plus_sqrt3_to_6th_l3287_328785


namespace NUMINAMATH_CALUDE_equation_solution_l3287_328789

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), 2 * (f x) - 16 = f (x - 6) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3287_328789


namespace NUMINAMATH_CALUDE_triangle_properties_l3287_328703

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle A is π/3 and the area is 4√3. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  Real.sin C / (Real.sin A + Real.sin B) + b / (a + c) = 1 →
  |b - a| = 4 →
  Real.cos B + Real.cos C = 1 →
  A = π / 3 ∧ a * c * (Real.sin B) / 2 = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3287_328703


namespace NUMINAMATH_CALUDE_school_student_count_l3287_328761

/-- The number of teachers in the school -/
def num_teachers : ℕ := 9

/-- The number of additional students needed for equal distribution -/
def additional_students : ℕ := 4

/-- The current number of students in the school -/
def current_students : ℕ := 23

/-- Theorem stating that the current number of students is correct -/
theorem school_student_count :
  ∃ (k : ℕ), (current_students + additional_students) = num_teachers * k ∧
             ∀ (m : ℕ), m < current_students →
               ¬(∃ (j : ℕ), (m + additional_students) = num_teachers * j) :=
by sorry

end NUMINAMATH_CALUDE_school_student_count_l3287_328761


namespace NUMINAMATH_CALUDE_probability_through_x_l3287_328774

structure DirectedGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)

def paths (g : DirectedGraph) (start finish : Char) : Nat :=
  sorry

theorem probability_through_x (g : DirectedGraph) :
  g.vertices = {'A', 'B', 'X', 'Y'} →
  paths g 'A' 'X' = 2 →
  paths g 'X' 'B' = 1 →
  paths g 'X' 'Y' = 1 →
  paths g 'Y' 'B' = 3 →
  paths g 'A' 'Y' = 3 →
  (paths g 'A' 'X' * paths g 'X' 'B' + paths g 'A' 'X' * paths g 'X' 'Y' * paths g 'Y' 'B') / 
  (paths g 'A' 'X' * paths g 'X' 'B' + paths g 'A' 'X' * paths g 'X' 'Y' * paths g 'Y' 'B' + paths g 'A' 'Y' * paths g 'Y' 'B') = 8 / 11 :=
sorry

end NUMINAMATH_CALUDE_probability_through_x_l3287_328774


namespace NUMINAMATH_CALUDE_inverse_f_sum_l3287_328771

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3*x - x^2

theorem inverse_f_sum : ∃ y₁ y₂ y₃ : ℝ, 
  f y₁ = -4 ∧ f y₂ = 1 ∧ f y₃ = 4 ∧ y₁ + y₂ + y₃ = 5 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l3287_328771


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l3287_328706

theorem other_root_of_quadratic (m : ℝ) : 
  ((-1)^2 + (-1) + m = 0) → 
  (∃ (x : ℝ), x ≠ -1 ∧ x^2 + x + m = 0 ∧ x = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l3287_328706


namespace NUMINAMATH_CALUDE_min_values_xy_l3287_328707

/-- Given positive real numbers x and y satisfying 2xy = x + 4y + a,
    prove the minimum values for xy and x + y + 2/x + 1/(2y) for different values of a. -/
theorem min_values_xy (x y a : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x * y = x + 4 * y + a) :
  (a = 16 → x * y ≥ 16) ∧
  (a = 0 → x + y + 2 / x + 1 / (2 * y) ≥ 11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_l3287_328707


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l3287_328773

/-- Given a circle C that is symmetric to the circle (x+2)^2+(y-1)^2=1 with respect to the origin,
    prove that the equation of circle C is (x-2)^2+(y+1)^2=1 -/
theorem symmetric_circle_equation (C : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (-x, -y) ∈ {(x, y) | (x + 2)^2 + (y - 1)^2 = 1}) →
  C = {(x, y) | (x - 2)^2 + (y + 1)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l3287_328773


namespace NUMINAMATH_CALUDE_remainder_theorem_l3287_328709

theorem remainder_theorem : (43^43 + 43) % 44 = 42 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3287_328709


namespace NUMINAMATH_CALUDE_book_sale_loss_percentage_l3287_328770

/-- Proves that when the cost price of 30 books equals the selling price of 40 books, the loss percentage is 25% -/
theorem book_sale_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h : 30 * cost_price = 40 * selling_price) : 
  (cost_price - selling_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_loss_percentage_l3287_328770


namespace NUMINAMATH_CALUDE_unique_complex_pair_l3287_328716

theorem unique_complex_pair : 
  ∃! (a b : ℂ), (a^4 * b^3 = 1) ∧ (a^6 * b^7 = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_complex_pair_l3287_328716


namespace NUMINAMATH_CALUDE_sin_five_pi_six_plus_two_alpha_l3287_328755

theorem sin_five_pi_six_plus_two_alpha (α : Real) 
  (h : Real.cos (α + π/6) = 1/3) : 
  Real.sin (5*π/6 + 2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_five_pi_six_plus_two_alpha_l3287_328755


namespace NUMINAMATH_CALUDE_positive_difference_of_numbers_l3287_328749

theorem positive_difference_of_numbers (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_squares_eq : a^2 - b^2 = 40) : 
  |a - b| = 4 := by
sorry

end NUMINAMATH_CALUDE_positive_difference_of_numbers_l3287_328749


namespace NUMINAMATH_CALUDE_factor_polynomial_l3287_328701

theorem factor_polynomial (x : ℝ) : 75 * x^7 - 300 * x^13 = 75 * x^7 * (1 - 4 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3287_328701


namespace NUMINAMATH_CALUDE_a_stops_implies_smudged_l3287_328705

-- Define the set of ladies
inductive Lady : Type
  | A
  | B
  | C

-- Define the state of a lady's face
inductive FaceState : Type
  | Clean
  | Smudged

-- Define the laughing state of a lady
inductive LaughState : Type
  | Laughing
  | NotLaughing

-- Function to get the face state of a lady
def faceState : Lady → FaceState
  | Lady.A => FaceState.Smudged
  | Lady.B => FaceState.Smudged
  | Lady.C => FaceState.Smudged

-- Function to get the initial laugh state of a lady
def initialLaughState : Lady → LaughState
  | _ => LaughState.Laughing

-- Function to determine if a lady can see another lady's smudged face
def canSeeSmugedFace (observer viewer : Lady) : Prop :=
  observer ≠ viewer ∧ faceState viewer = FaceState.Smudged

-- Theorem: If A stops laughing, it implies A must have a smudged face
theorem a_stops_implies_smudged :
  (initialLaughState Lady.A = LaughState.Laughing) →
  (∃ (newLaughState : Lady → LaughState),
    newLaughState Lady.A = LaughState.NotLaughing ∧
    (∀ l : Lady, l ≠ Lady.A → newLaughState l = LaughState.Laughing)) →
  faceState Lady.A = FaceState.Smudged :=
by
  sorry


end NUMINAMATH_CALUDE_a_stops_implies_smudged_l3287_328705


namespace NUMINAMATH_CALUDE_max_beauty_value_bound_l3287_328714

/-- Represents a figure with circles and segments arranged into pentagons -/
structure Figure where
  circles : Nat
  segments : Nat
  pentagons : Nat

/-- Represents a method of filling numbers in the circles -/
def FillingMethod := Fin 15 → Fin 3

/-- Calculates the beauty value of a filling method -/
def beautyValue (f : Figure) (m : FillingMethod) : Nat :=
  sorry

/-- The maximum possible beauty value -/
def maxBeautyValue (f : Figure) : Nat :=
  sorry

theorem max_beauty_value_bound (f : Figure) 
  (h1 : f.circles = 15) 
  (h2 : f.segments = 20) 
  (h3 : f.pentagons = 6) : 
  maxBeautyValue f ≤ 17 := by
  sorry

end NUMINAMATH_CALUDE_max_beauty_value_bound_l3287_328714


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l3287_328767

theorem sum_of_real_solutions (b : ℝ) (h : b > 0) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + 2*x)) = x ∧
  (∀ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + 2*y)) = y → y = x) ∧
  x = Real.sqrt (b - 1) - 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l3287_328767


namespace NUMINAMATH_CALUDE_simplify_powers_l3287_328766

theorem simplify_powers (x : ℝ) : x^5 * x^3 * 2 = 2 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_powers_l3287_328766


namespace NUMINAMATH_CALUDE_vector_equations_l3287_328733

def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

def a : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def b : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def c : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)

theorem vector_equations :
  (3 * a.1 + b.1, 3 * a.2 + b.2) = (9, -18) ∧
  a = (-b.1 - c.1, -b.2 - c.2) :=
sorry

end NUMINAMATH_CALUDE_vector_equations_l3287_328733


namespace NUMINAMATH_CALUDE_hospital_workers_count_l3287_328792

/-- The number of other workers at the hospital -/
def num_other_workers : ℕ := 2

/-- The total number of workers at the hospital -/
def total_workers : ℕ := num_other_workers + 2

/-- The probability of selecting both John and David when choosing 2 workers randomly -/
def prob_select_john_and_david : ℚ := 1 / 6

theorem hospital_workers_count :
  (prob_select_john_and_david = 1 / (total_workers.choose 2)) →
  num_other_workers = 2 := by
sorry

#eval num_other_workers

end NUMINAMATH_CALUDE_hospital_workers_count_l3287_328792


namespace NUMINAMATH_CALUDE_mittens_per_box_l3287_328736

/-- Given the conditions of Chloe's winter clothing boxes, prove the number of mittens per box -/
theorem mittens_per_box 
  (num_boxes : ℕ) 
  (scarves_per_box : ℕ) 
  (total_pieces : ℕ) 
  (h1 : num_boxes = 4) 
  (h2 : scarves_per_box = 2) 
  (h3 : total_pieces = 32) : 
  (total_pieces - num_boxes * scarves_per_box) / num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_mittens_per_box_l3287_328736


namespace NUMINAMATH_CALUDE_trig_identity_l3287_328741

theorem trig_identity : Real.cos (70 * π / 180) * Real.sin (50 * π / 180) - 
                        Real.cos (200 * π / 180) * Real.sin (40 * π / 180) = 
                        Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3287_328741


namespace NUMINAMATH_CALUDE_olivia_initial_money_l3287_328765

/-- The amount of money Olivia and Nigel spent on tickets -/
def ticket_cost : ℕ := 6 * 28

/-- The amount of money Nigel had initially -/
def nigel_money : ℕ := 139

/-- The amount of money Olivia and Nigel have left after buying tickets -/
def remaining_money : ℕ := 83

/-- The amount of money Olivia had initially -/
def olivia_money : ℕ := (ticket_cost + remaining_money) - nigel_money

theorem olivia_initial_money : olivia_money = 112 := by
  sorry

end NUMINAMATH_CALUDE_olivia_initial_money_l3287_328765


namespace NUMINAMATH_CALUDE_f_derivative_l3287_328726

noncomputable def f (x : ℝ) : ℝ := x * Real.cos (2 * x)

theorem f_derivative : 
  deriv f = fun x => Real.cos (2 * x) - 2 * x * Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_l3287_328726


namespace NUMINAMATH_CALUDE_machines_count_l3287_328779

/-- The number of machines that complete a job lot in 6 hours -/
def N : ℕ := 8

/-- The time taken by N machines to complete the job lot -/
def time_N : ℕ := 6

/-- The number of machines in the second scenario -/
def machines_2 : ℕ := 4

/-- The time taken by machines_2 to complete the job lot -/
def time_2 : ℕ := 12

/-- The work rate of a single machine (job lots per hour) -/
def work_rate : ℚ := 1 / 48

theorem machines_count :
  N * work_rate * time_N = 1 ∧
  machines_2 * work_rate * time_2 = 1 :=
sorry

#check machines_count

end NUMINAMATH_CALUDE_machines_count_l3287_328779


namespace NUMINAMATH_CALUDE_solution_system_equations_l3287_328772

theorem solution_system_equations :
  ∀ x y z : ℝ,
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4 →
  ((x = 1/3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2)) :=
by
  sorry

end NUMINAMATH_CALUDE_solution_system_equations_l3287_328772


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_max_area_difference_l3287_328738

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the vertices of the ellipse
def A : ℝ × ℝ := (-5, 0)
def B : ℝ × ℝ := (5, 0)

-- Define a line
def line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the slope ratio condition
def slope_ratio (k₁ k₂ : ℝ) : Prop := k₁ / k₂ = 1 / 9

-- Define the intersection points
def intersection_points (k m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line k m x₁ y₁ ∧ line k m x₂ y₂

-- Define the triangles' areas
def area_diff (S₁ S₂ : ℝ) : Prop := S₁ - S₂ ≤ 15

-- Theorem 1: The line passes through (4, 0)
theorem line_passes_through_fixed_point (k m : ℝ) :
  intersection_points k m →
  (∃ k₁ k₂ : ℝ, slope_ratio k₁ k₂) →
  line k m 4 0 :=
sorry

-- Theorem 2: Maximum value of S₁ - S₂
theorem max_area_difference :
  ∀ S₁ S₂ : ℝ,
  (∃ k m : ℝ, intersection_points k m ∧ 
   (∃ k₁ k₂ : ℝ, slope_ratio k₁ k₂)) →
  area_diff S₁ S₂ ∧ 
  (∀ S₁' S₂' : ℝ, area_diff S₁' S₂' → S₁ - S₂ ≥ S₁' - S₂') :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_max_area_difference_l3287_328738


namespace NUMINAMATH_CALUDE_little_john_initial_money_l3287_328727

theorem little_john_initial_money :
  let sweets_cost : ℚ := 1.25
  let friends_count : ℕ := 2
  let money_per_friend : ℚ := 1.20
  let money_left : ℚ := 4.85
  let initial_money : ℚ := sweets_cost + friends_count * money_per_friend + money_left
  initial_money = 8.50 := by sorry

end NUMINAMATH_CALUDE_little_john_initial_money_l3287_328727


namespace NUMINAMATH_CALUDE_third_quarter_gdp_l3287_328713

/-- Represents the GDP growth over quarters -/
def gdp_growth (initial_gdp : ℝ) (growth_rate : ℝ) (quarters : ℕ) : ℝ :=
  initial_gdp * (1 + growth_rate) ^ quarters

theorem third_quarter_gdp 
  (initial_gdp : ℝ) 
  (growth_rate : ℝ) :
  gdp_growth initial_gdp growth_rate 2 = initial_gdp * (1 + growth_rate)^2 :=
by sorry

end NUMINAMATH_CALUDE_third_quarter_gdp_l3287_328713


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l3287_328799

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  (∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b) ∧
  n^2 = (n / 10 + n % 10)^3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l3287_328799


namespace NUMINAMATH_CALUDE_interest_problem_l3287_328776

/-- Given a sum P at simple interest rate R for 3 years, if increasing the rate by 8%
    results in Rs. 120 more interest, then P = 500. -/
theorem interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 8) * 3) / 100 = (P * R * 3) / 100 + 120 →
  P = 500 := by
  sorry

end NUMINAMATH_CALUDE_interest_problem_l3287_328776


namespace NUMINAMATH_CALUDE_cube_sum_inequality_cube_sum_equality_l3287_328720

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (y*z) + y^3 / (z*x) + z^3 / (x*y) ≥ x + y + z :=
sorry

theorem cube_sum_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (y*z) + y^3 / (z*x) + z^3 / (x*y) = x + y + z ↔ x = y ∧ y = z :=
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_cube_sum_equality_l3287_328720


namespace NUMINAMATH_CALUDE_three_distinct_volumes_l3287_328781

/-- A triangular pyramid with specific face conditions -/
structure TriangularPyramid where
  /-- Two lateral faces are isosceles right triangles -/
  has_two_isosceles_right_faces : Bool
  /-- One face is an equilateral triangle with side length 1 -/
  has_equilateral_face : Bool
  /-- The side length of the equilateral face -/
  equilateral_side_length : ℝ

/-- The volume of a triangular pyramid -/
def volume (pyramid : TriangularPyramid) : ℝ := sorry

/-- The set of all possible volumes for triangular pyramids satisfying the conditions -/
def possible_volumes : Set ℝ := sorry

/-- Theorem stating that there are exactly three distinct volumes -/
theorem three_distinct_volumes :
  ∃ (v₁ v₂ v₃ : ℝ), v₁ ≠ v₂ ∧ v₁ ≠ v₃ ∧ v₂ ≠ v₃ ∧
  possible_volumes = {v₁, v₂, v₃} :=
sorry

end NUMINAMATH_CALUDE_three_distinct_volumes_l3287_328781


namespace NUMINAMATH_CALUDE_constant_distance_special_points_min_distance_to_origin_euclidean_vs_orthogonal_distance_l3287_328794

-- Define orthogonal distance
def orthogonal_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Proposition 1
theorem constant_distance_special_points :
  ∀ α : ℝ, orthogonal_distance 2 3 (Real.sin α ^ 2) (Real.cos α ^ 2) = 4 :=
sorry

-- Proposition 2 (negation)
theorem min_distance_to_origin :
  ∃ x y : ℝ, x - y + 1 = 0 ∧ |x| + |y| < 1 :=
sorry

-- Proposition 3
theorem euclidean_vs_orthogonal_distance :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ (Real.sqrt 2 / 2) * (|x₁ - x₂| + |y₁ - y₂|) :=
sorry

end NUMINAMATH_CALUDE_constant_distance_special_points_min_distance_to_origin_euclidean_vs_orthogonal_distance_l3287_328794


namespace NUMINAMATH_CALUDE_no_solution_iff_m_leq_three_l3287_328756

/-- Given a real number m, the system of inequalities {x - m > 2, x - 2m < -1} has no solution if and only if m ≤ 3. -/
theorem no_solution_iff_m_leq_three (m : ℝ) : 
  (∀ x : ℝ, ¬(x - m > 2 ∧ x - 2*m < -1)) ↔ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_leq_three_l3287_328756


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3287_328777

/-- Given a quadratic function f(x) = ax² + bx + c passing through specific points,
    prove that g(x) = cx² + 2bx + a has a specific vertex form -/
theorem quadratic_transformation (a b c : ℝ) 
  (h1 : c = 1)
  (h2 : a + b + c = -2)
  (h3 : a - b + c = 2) :
  let f := fun x => a * x^2 + b * x + c
  let g := fun x => c * x^2 + 2 * b * x + a
  ∀ x, g x = (x - 2)^2 - 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3287_328777
