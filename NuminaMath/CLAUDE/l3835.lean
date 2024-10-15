import Mathlib

namespace NUMINAMATH_CALUDE_triangle_third_side_l3835_383545

theorem triangle_third_side (a b area c : ℝ) : 
  a = 2 * Real.sqrt 2 →
  b = 3 →
  area = 3 →
  area = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) →
  (c = Real.sqrt 5 ∨ c = Real.sqrt 29) := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l3835_383545


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3835_383582

/-- Represents a modified cube with square holes cut through each face -/
structure ModifiedCube where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a modified cube including inside surfaces -/
def total_surface_area (cube : ModifiedCube) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let hole_area := 6 * cube.hole_side_length^2
  let new_exposed_area := 6 * 4 * cube.hole_side_length^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 4 and hole side length 2 has a total surface area of 168 -/
theorem modified_cube_surface_area :
  let cube : ModifiedCube := { edge_length := 4, hole_side_length := 2 }
  total_surface_area cube = 168 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3835_383582


namespace NUMINAMATH_CALUDE_bank_cash_increase_l3835_383575

/-- Represents a bank transaction --/
inductive Transaction
  | Deposit (amount : ℕ)
  | Withdrawal (amount : ℕ)

/-- Calculates the net change in cash after a series of transactions --/
def netChange (transactions : List Transaction) : ℤ :=
  transactions.foldl
    (fun acc t => match t with
      | Transaction.Deposit a => acc + a
      | Transaction.Withdrawal a => acc - a)
    0

/-- The list of transactions for the day --/
def dayTransactions : List Transaction := [
  Transaction.Withdrawal 960000,
  Transaction.Deposit 500000,
  Transaction.Withdrawal 700000,
  Transaction.Deposit 1200000,
  Transaction.Deposit 2200000,
  Transaction.Withdrawal 1025000,
  Transaction.Withdrawal 240000
]

theorem bank_cash_increase :
  netChange dayTransactions = 975000 := by
  sorry

end NUMINAMATH_CALUDE_bank_cash_increase_l3835_383575


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_200_l3835_383580

theorem largest_multiple_of_11_below_negative_200 :
  ∀ n : ℤ, n * 11 < -200 → n * 11 ≤ -209 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_below_negative_200_l3835_383580


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_97_l3835_383512

theorem first_nonzero_digit_after_decimal_1_97 : ∃ (n : ℕ) (d : ℕ), 
  0 < d ∧ d < 10 ∧ 
  (∃ (k : ℕ), 10^n ≤ k * 97 ∧ k * 97 < 10^(n+1) ∧ 
  (10^(n+1) * 1 - k * 97) / 97 = d) ∧
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_97_l3835_383512


namespace NUMINAMATH_CALUDE_fourth_square_dots_l3835_383586

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- The number of dots in the nth square -/
def num_dots (n : ℕ) : ℕ := (side_length n) ^ 2

theorem fourth_square_dots :
  num_dots 4 = 49 := by sorry

end NUMINAMATH_CALUDE_fourth_square_dots_l3835_383586


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3835_383591

theorem sum_of_x_and_y (x y : ℝ) (h : (2 : ℝ)^x = (18 : ℝ)^y ∧ (2 : ℝ)^x = (6 : ℝ)^(x*y)) :
  x + y = 0 ∨ x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3835_383591


namespace NUMINAMATH_CALUDE_hyperbola_parabola_same_foci_l3835_383564

-- Define the hyperbola equation
def hyperbola (k : ℝ) (x y : ℝ) : Prop :=
  y^2 / 5 - x^2 / k = 1

-- Define the parabola equation
def parabola (x y : ℝ) : Prop :=
  x^2 = 12 * y

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (0, 3)

-- Define the property of having the same foci
def same_foci (k : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 5 + (-k) ∧ c = 3

-- Theorem statement
theorem hyperbola_parabola_same_foci (k : ℝ) :
  same_foci k → k = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_same_foci_l3835_383564


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l3835_383516

theorem greatest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, 7 - 3 * y + 2 > 23 → y ≤ x) ↔ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l3835_383516


namespace NUMINAMATH_CALUDE_other_asymptote_equation_l3835_383539

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ

/-- Theorem: For a hyperbola with one asymptote y = 4x and foci on the line x = 3,
    the other asymptote has the equation y = -4x + 24 -/
theorem other_asymptote_equation (h : Hyperbola) 
    (h1 : h.asymptote1 = fun x => 4 * x) 
    (h2 : h.foci_x = 3) : 
    ∃ (asymptote2 : ℝ → ℝ), asymptote2 = fun x => -4 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_other_asymptote_equation_l3835_383539


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3835_383567

theorem smallest_four_digit_divisible_by_53 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1007 → ¬(53 ∣ n)) ∧ 53 ∣ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l3835_383567


namespace NUMINAMATH_CALUDE_art_exhibit_revenue_l3835_383510

/-- Calculates the total revenue from ticket sales for an art exhibit --/
theorem art_exhibit_revenue :
  let start_time : Nat := 9 * 60  -- 9:00 AM in minutes
  let end_time : Nat := 16 * 60 + 55  -- 4:55 PM in minutes
  let interval : Nat := 5  -- 5 minutes
  let group_size : Nat := 30
  let regular_price : Nat := 10
  let student_price : Nat := 6
  let regular_to_student_ratio : Nat := 3

  let total_intervals : Nat := (end_time - start_time) / interval + 1
  let total_tickets : Nat := total_intervals * group_size
  let student_tickets : Nat := total_tickets / (regular_to_student_ratio + 1)
  let regular_tickets : Nat := total_tickets - student_tickets

  let total_revenue : Nat := student_tickets * student_price + regular_tickets * regular_price

  total_revenue = 25652 := by sorry

end NUMINAMATH_CALUDE_art_exhibit_revenue_l3835_383510


namespace NUMINAMATH_CALUDE_reflected_ray_is_correct_l3835_383553

/-- The line of reflection --/
def reflection_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = -1}

/-- Point P --/
def P : ℝ × ℝ := (1, 1)

/-- Point Q --/
def Q : ℝ × ℝ := (2, 3)

/-- The reflected ray --/
def reflected_ray : Set (ℝ × ℝ) := {p : ℝ × ℝ | 5 * p.1 - 4 * p.2 + 2 = 0}

/-- Theorem stating that the reflected ray is correct --/
theorem reflected_ray_is_correct : 
  ∃ (M : ℝ × ℝ), 
    (M ∈ reflected_ray) ∧ 
    (Q ∈ reflected_ray) ∧
    (∀ (X : ℝ × ℝ), X ∈ reflection_line → (X.1 - P.1) * (X.1 - M.1) + (X.2 - P.2) * (X.2 - M.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_is_correct_l3835_383553


namespace NUMINAMATH_CALUDE_weight_difference_l3835_383554

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem weight_difference : mildred_weight - carol_weight = 50 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3835_383554


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3835_383532

/-- An isosceles triangle with two sides of lengths 2 and 4 has a perimeter of 10. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  (a = b ∨ b = c ∨ a = c) →
  ((a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2) ∨ (b = 2 ∧ c = 4) ∨ (b = 4 ∧ c = 2) ∨ (a = 2 ∧ c = 4) ∨ (a = 4 ∧ c = 2)) →
  a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3835_383532


namespace NUMINAMATH_CALUDE_ellipse_focus_m_value_l3835_383531

/-- Given an ellipse with equation x²/25 + y²/m² = 1 where m > 0,
    and left focus at (-4, 0), prove that m = 3 -/
theorem ellipse_focus_m_value (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) →
  (∃ x y : ℝ, x = -4 ∧ y = 0 ∧ x^2 / 25 + y^2 / m^2 = 1) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focus_m_value_l3835_383531


namespace NUMINAMATH_CALUDE_polar_bear_fish_consumption_l3835_383562

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish : ℝ := trout_amount + salmon_amount

theorem polar_bear_fish_consumption :
  total_fish = 0.6 := by sorry

end NUMINAMATH_CALUDE_polar_bear_fish_consumption_l3835_383562


namespace NUMINAMATH_CALUDE_function_shift_and_value_l3835_383520

theorem function_shift_and_value (φ : Real) 
  (h1 : 0 < φ ∧ φ < π / 2) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x, f x = 2 * Real.sin (x + φ)) 
  (g : ℝ → ℝ) 
  (h3 : ∀ x, g x = f (x + π / 3)) 
  (h4 : ∀ x, g x = g (-x)) : 
  f (π / 6) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_shift_and_value_l3835_383520


namespace NUMINAMATH_CALUDE_turtle_arrangement_l3835_383560

/-- The number of grid intersections in a rectangular arrangement of square tiles -/
def grid_intersections (width : ℕ) (height : ℕ) : ℕ :=
  (width + 1) * height

/-- Theorem: The number of grid intersections in a 20 × 21 rectangular arrangement of square tiles is 420 -/
theorem turtle_arrangement : grid_intersections 20 21 = 420 := by
  sorry

end NUMINAMATH_CALUDE_turtle_arrangement_l3835_383560


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l3835_383541

/-- Given a function g : ℝ → ℝ such that g(8) = 5, prove that (8/3, 14/9) is on the graph of 3y = g(3x)/3 + 3 and the sum of its coordinates is 38/9 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 8 = 5) :
  let f : ℝ → ℝ := λ x => (g (3 * x) / 3 + 3) / 3
  f (8/3) = 14/9 ∧ 8/3 + 14/9 = 38/9 := by
sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l3835_383541


namespace NUMINAMATH_CALUDE_polynomial_value_l3835_383584

def f (x : ℝ) (a₀ a₁ a₂ a₃ a₄ : ℝ) : ℝ :=
  a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + 7 * x^5

theorem polynomial_value (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  f 2004 a₀ a₁ a₂ a₃ a₄ = 72 ∧
  f 2005 a₀ a₁ a₂ a₃ a₄ = -30 ∧
  f 2006 a₀ a₁ a₂ a₃ a₄ = 32 ∧
  f 2007 a₀ a₁ a₂ a₃ a₄ = -24 ∧
  f 2008 a₀ a₁ a₂ a₃ a₄ = 24 →
  f 2009 a₀ a₁ a₂ a₃ a₄ = 847 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l3835_383584


namespace NUMINAMATH_CALUDE_horner_rule_operations_l3835_383558

/-- Horner's Rule evaluation steps for a polynomial -/
def hornerSteps (coeffs : List ℤ) (x : ℤ) : List ℤ :=
  match coeffs with
  | [] => []
  | a :: as => List.scanl (fun acc b => acc * x + b) a as

/-- Number of multiplications in Horner's Rule -/
def numMultiplications (coeffs : List ℤ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: _ => coeffs.length - 1

/-- Number of additions in Horner's Rule -/
def numAdditions (coeffs : List ℤ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: _ => coeffs.length - 1

/-- The polynomial f(x) = 5x^6 + 4x^5 + x^4 + 3x^3 - 81x^2 + 9x - 1 -/
def f : List ℤ := [5, 4, 1, 3, -81, 9, -1]

theorem horner_rule_operations :
  numMultiplications f = 6 ∧ numAdditions f = 6 :=
sorry

end NUMINAMATH_CALUDE_horner_rule_operations_l3835_383558


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_15_l3835_383578

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_sum_15 :
  ∀ n : ℕ, is_three_digit n → digit_sum n = 15 → n ≤ 960 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_15_l3835_383578


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3835_383515

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 21.875

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 35

/-- Theorem stating that the weight of one bowling ball is 21.875 pounds -/
theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 140) →
  bowling_ball_weight = 21.875 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3835_383515


namespace NUMINAMATH_CALUDE_fifi_closet_hangers_l3835_383529

theorem fifi_closet_hangers :
  let pink : ℕ := 7
  let green : ℕ := 4
  let blue : ℕ := green - 1
  let yellow : ℕ := blue - 1
  pink + green + blue + yellow = 16 :=
by sorry

end NUMINAMATH_CALUDE_fifi_closet_hangers_l3835_383529


namespace NUMINAMATH_CALUDE_combination_count_l3835_383503

/-- The number of different styles of backpacks -/
def num_backpacks : ℕ := 2

/-- The number of different styles of pencil cases -/
def num_pencil_cases : ℕ := 2

/-- A combination consists of one backpack and one pencil case -/
def combination := ℕ × ℕ

/-- The total number of possible combinations -/
def total_combinations : ℕ := num_backpacks * num_pencil_cases

theorem combination_count : total_combinations = 4 := by sorry

end NUMINAMATH_CALUDE_combination_count_l3835_383503


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l3835_383570

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ (p : ℚ) / q < 5 / 8 →
  q ≥ 13 ∧ (q = 13 → p = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l3835_383570


namespace NUMINAMATH_CALUDE_frequency_distribution_theorem_l3835_383538

-- Define the frequency of the first group
def f1 : ℕ := 6

-- Define the frequencies of the second and third groups based on the ratio
def f2 : ℕ := 2 * f1
def f3 : ℕ := 3 * f1

-- Define the sum of frequencies for the first three groups
def sum_first_three : ℕ := f1 + f2 + f3

-- Define the total number of students
def total_students : ℕ := 48

-- Theorem statement
theorem frequency_distribution_theorem :
  sum_first_three < total_students ∧ 
  total_students - sum_first_three > 0 ∧
  total_students - sum_first_three < f3 :=
by sorry

end NUMINAMATH_CALUDE_frequency_distribution_theorem_l3835_383538


namespace NUMINAMATH_CALUDE_enrique_commission_l3835_383599

/-- Calculates the commission for a given item --/
def calculate_commission (price : ℝ) (quantity : ℕ) (commission_rate : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 - discount_rate) * (1 + tax_rate) * quantity * commission_rate

/-- Calculates the total commission for all items sold --/
def total_commission (suit_price suit_quantity : ℕ) (shirt_price shirt_quantity : ℕ) 
                     (loafer_price loafer_quantity : ℕ) (tie_price tie_quantity : ℕ) 
                     (sock_price sock_quantity : ℕ) : ℝ :=
  let suit_commission := calculate_commission (suit_price : ℝ) suit_quantity 0.15 0.1 0
  let shirt_commission := calculate_commission (shirt_price : ℝ) shirt_quantity 0.15 0 0.05
  let loafer_commission := calculate_commission (loafer_price : ℝ) loafer_quantity 0.1 0 0.05
  let tie_commission := calculate_commission (tie_price : ℝ) tie_quantity 0.1 0 0.05
  let sock_commission := calculate_commission (sock_price : ℝ) sock_quantity 0.1 0 0.05
  suit_commission + shirt_commission + loafer_commission + tie_commission + sock_commission

theorem enrique_commission : 
  total_commission 700 2 50 6 150 2 30 4 10 5 = 285.60 := by
  sorry

end NUMINAMATH_CALUDE_enrique_commission_l3835_383599


namespace NUMINAMATH_CALUDE_parents_age_difference_l3835_383597

/-- The difference between Sobha's parents' ages -/
def age_difference (s : ℕ) : ℕ :=
  let f := s + 38  -- father's age
  let b := s - 4   -- brother's age
  let m := b + 36  -- mother's age
  f - m

/-- Theorem stating that the age difference between Sobha's parents is 6 years -/
theorem parents_age_difference (s : ℕ) (h : s ≥ 4) : age_difference s = 6 := by
  sorry

end NUMINAMATH_CALUDE_parents_age_difference_l3835_383597


namespace NUMINAMATH_CALUDE_certain_number_is_three_l3835_383542

theorem certain_number_is_three (x : ℝ) (n : ℝ) : 
  (4 / (1 + n / x) = 1) → (x = 1) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l3835_383542


namespace NUMINAMATH_CALUDE_lily_of_valley_price_increase_l3835_383555

/-- Calculates the percentage increase in selling price compared to buying price for Françoise's lily of the valley pots. -/
theorem lily_of_valley_price_increase 
  (buying_price : ℝ) 
  (num_pots : ℕ) 
  (amount_given_back : ℝ) 
  (h1 : buying_price = 12)
  (h2 : num_pots = 150)
  (h3 : amount_given_back = 450) :
  let total_cost := buying_price * num_pots
  let total_revenue := total_cost + amount_given_back
  let selling_price := total_revenue / num_pots
  (selling_price - buying_price) / buying_price * 100 = 25 := by
sorry


end NUMINAMATH_CALUDE_lily_of_valley_price_increase_l3835_383555


namespace NUMINAMATH_CALUDE_expression_evaluation_l3835_383540

theorem expression_evaluation : 11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3835_383540


namespace NUMINAMATH_CALUDE_diagonals_sum_bounds_l3835_383521

/-- A convex pentagon in a 2D plane -/
structure ConvexPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  convex : Bool

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the perimeter of the pentagon -/
def perimeter (p : ConvexPentagon) : ℝ :=
  distance p.A p.B + distance p.B p.C + distance p.C p.D + distance p.D p.E + distance p.E p.A

/-- Calculate the sum of diagonals of the pentagon -/
def sumDiagonals (p : ConvexPentagon) : ℝ :=
  distance p.A p.C + distance p.B p.D + distance p.C p.E + distance p.D p.A + distance p.B p.E

/-- Theorem: The sum of diagonals is greater than the perimeter but less than twice the perimeter -/
theorem diagonals_sum_bounds (p : ConvexPentagon) (h : p.convex = true) :
  perimeter p < sumDiagonals p ∧ sumDiagonals p < 2 * perimeter p := by sorry

end NUMINAMATH_CALUDE_diagonals_sum_bounds_l3835_383521


namespace NUMINAMATH_CALUDE_annies_crayons_l3835_383502

/-- Annie's crayon problem -/
theorem annies_crayons (initial : ℕ) (additional : ℕ) : 
  initial = 4 → additional = 36 → initial + additional = 40 := by
  sorry

end NUMINAMATH_CALUDE_annies_crayons_l3835_383502


namespace NUMINAMATH_CALUDE_hexagon_area_is_six_l3835_383509

/-- A point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- A polygon defined by its vertices --/
structure Polygon where
  vertices : List GridPoint

/-- Calculate the area of a polygon given its vertices --/
def calculateArea (p : Polygon) : Int :=
  sorry

/-- The 4x4 square on the grid --/
def square : Polygon :=
  { vertices := [
      { x := 0, y := 0 },
      { x := 0, y := 4 },
      { x := 4, y := 4 },
      { x := 4, y := 0 }
    ] }

/-- The hexagon formed by adding two points to the square --/
def hexagon : Polygon :=
  { vertices := [
      { x := 0, y := 0 },
      { x := 0, y := 4 },
      { x := 2, y := 4 },
      { x := 4, y := 4 },
      { x := 4, y := 0 },
      { x := 2, y := 0 }
    ] }

theorem hexagon_area_is_six :
  calculateArea hexagon = 6 :=
sorry

end NUMINAMATH_CALUDE_hexagon_area_is_six_l3835_383509


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3835_383588

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l3835_383588


namespace NUMINAMATH_CALUDE_conditional_probability_not_first_class_l3835_383524

def total_products : ℕ := 8
def first_class_products : ℕ := 6
def selected_products : ℕ := 2

theorem conditional_probability_not_first_class 
  (h1 : total_products = 8)
  (h2 : first_class_products = 6)
  (h3 : selected_products = 2)
  (h4 : first_class_products < total_products)
  (h5 : selected_products ≤ total_products) :
  (Nat.choose first_class_products 1 * Nat.choose (total_products - first_class_products) 1) / 
  (Nat.choose total_products selected_products - Nat.choose first_class_products selected_products) = 12 / 13 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_not_first_class_l3835_383524


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3835_383547

/-- Given that px³ + qx - 10 = 2006 when x = 1, prove that px³ + qx - 10 = -2026 when x = -1 -/
theorem cubic_expression_value (p q : ℝ) 
  (h : p * 1^3 + q * 1 - 10 = 2006) :
  p * (-1)^3 + q * (-1) - 10 = -2026 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3835_383547


namespace NUMINAMATH_CALUDE_sample_size_calculation_l3835_383534

/-- Given three districts with population ratios 2:3:5 and 100 people sampled from the largest district,
    the total sample size is 200. -/
theorem sample_size_calculation (ratio_a ratio_b ratio_c : ℕ) 
  (largest_district_sample : ℕ) :
  ratio_a = 2 → ratio_b = 3 → ratio_c = 5 → 
  largest_district_sample = 100 →
  (ratio_a + ratio_b + ratio_c : ℚ) * largest_district_sample / ratio_c = 200 :=
by sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l3835_383534


namespace NUMINAMATH_CALUDE_correct_representation_l3835_383544

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented -/
def number : ℕ := 91000

/-- The scientific notation representation of the number -/
def representation : ScientificNotation := {
  coefficient := 9.1
  exponent := 4
  h1 := by sorry
}

/-- Theorem: The given representation is correct for the number -/
theorem correct_representation : 
  (representation.coefficient * (10 : ℝ) ^ representation.exponent) = number := by sorry

end NUMINAMATH_CALUDE_correct_representation_l3835_383544


namespace NUMINAMATH_CALUDE_division_problem_l3835_383598

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 136 → 
  quotient = 9 → 
  remainder = 1 → 
  dividend = divisor * quotient + remainder → 
  divisor = 15 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3835_383598


namespace NUMINAMATH_CALUDE_total_missed_pitches_example_l3835_383556

/-- Represents a person's batting performance -/
structure BattingPerformance where
  tokens : Nat
  hits : Nat

/-- Calculates the total number of missed pitches for all players -/
def totalMissedPitches (pitchesPerToken : Nat) (performances : List BattingPerformance) : Nat :=
  performances.foldl (fun acc p => acc + p.tokens * pitchesPerToken - p.hits) 0

theorem total_missed_pitches_example :
  let pitchesPerToken := 15
  let macy := BattingPerformance.mk 11 50
  let piper := BattingPerformance.mk 17 55
  let quinn := BattingPerformance.mk 13 60
  let performances := [macy, piper, quinn]
  totalMissedPitches pitchesPerToken performances = 450 := by
  sorry

#eval totalMissedPitches 15 [BattingPerformance.mk 11 50, BattingPerformance.mk 17 55, BattingPerformance.mk 13 60]

end NUMINAMATH_CALUDE_total_missed_pitches_example_l3835_383556


namespace NUMINAMATH_CALUDE_sport_formulation_corn_syrup_l3835_383572

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio (standard : DrinkRatio) : DrinkRatio :=
  ⟨standard.flavoring,
   standard.corn_syrup / 3,
   standard.water * 2⟩

/-- Calculates the amount of corn syrup given the amount of water and the ratio -/
def corn_syrup_amount (water_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (ratio.corn_syrup * water_amount) / ratio.water

theorem sport_formulation_corn_syrup :
  corn_syrup_amount 30 (sport_ratio standard_ratio) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_corn_syrup_l3835_383572


namespace NUMINAMATH_CALUDE_complex_number_range_l3835_383559

theorem complex_number_range (x y : ℝ) : 
  let z : ℂ := x + y * Complex.I
  (Complex.abs (z - (3 + 4 * Complex.I)) = 1) →
  (16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_range_l3835_383559


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3835_383543

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  (∀ a : ℝ, C a ⊆ (A ∪ B) → 2 ≤ a ∧ a ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3835_383543


namespace NUMINAMATH_CALUDE_area_of_triangle_ABC_is_one_l3835_383566

/-- Given three unit squares arranged in a straight line, each sharing a side with the next,
    where A is the bottom left vertex of the first square,
    B is the top right vertex of the second square,
    and C is the top left vertex of the third square,
    prove that the area of triangle ABC is 1. -/
theorem area_of_triangle_ABC_is_one :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 1)
  let C : ℝ × ℝ := (2, 1)
  let triangle_area (p q r : ℝ × ℝ) : ℝ :=
    (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  triangle_area A B C = 1 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABC_is_one_l3835_383566


namespace NUMINAMATH_CALUDE_even_digit_sum_pairs_count_l3835_383504

/-- Given a natural number, returns true if its digit sum is even -/
def has_even_digit_sum (n : ℕ) : Bool :=
  sorry

/-- Returns the count of natural numbers less than 10^6 where both
    the number and its successor have even digit sums -/
def count_even_digit_sum_pairs : ℕ :=
  sorry

/-- The main theorem stating that the count of natural numbers less than 10^6
    where both the number and its successor have even digit sums is 45454 -/
theorem even_digit_sum_pairs_count :
  count_even_digit_sum_pairs = 45454 := by sorry

end NUMINAMATH_CALUDE_even_digit_sum_pairs_count_l3835_383504


namespace NUMINAMATH_CALUDE_remainder_sum_l3835_383573

theorem remainder_sum (c d : ℤ) (hc : c % 100 = 86) (hd : d % 150 = 144) :
  (c + d) % 50 = 30 := by sorry

end NUMINAMATH_CALUDE_remainder_sum_l3835_383573


namespace NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l3835_383596

/-- Given a line segment AB extended to P such that AP:PB = 10:3,
    prove that the position vector of P can be expressed as P = -3/7*A + 10/7*B,
    where A and B are the position vectors of points A and B respectively. -/
theorem extended_line_segment_vector_representation 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h : (dist A P) / (dist P B) = 10 / 3) -- AP:PB = 10:3
  : ∃ (t u : ℝ), P = t • A + u • B ∧ t = -3/7 ∧ u = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l3835_383596


namespace NUMINAMATH_CALUDE_complex_angle_90_degrees_l3835_383594

theorem complex_angle_90_degrees (z₁ z₂ : ℂ) (hz₁ : z₁ ≠ 0) (hz₂ : z₂ ≠ 0) 
  (h : Complex.abs (z₁ + z₂) = Complex.abs (z₁ - z₂)) : 
  Real.cos (Complex.arg z₁ - Complex.arg z₂) = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_angle_90_degrees_l3835_383594


namespace NUMINAMATH_CALUDE_josh_extracurricular_hours_l3835_383593

/-- Represents the number of days Josh has soccer practice in a week -/
def soccer_days : ℕ := 3

/-- Represents the number of hours Josh spends on soccer practice each day -/
def soccer_hours_per_day : ℝ := 2

/-- Represents the number of days Josh has band practice in a week -/
def band_days : ℕ := 2

/-- Represents the number of hours Josh spends on band practice each day -/
def band_hours_per_day : ℝ := 1.5

/-- Calculates the total hours Josh spends on extracurricular activities in a week -/
def total_extracurricular_hours : ℝ :=
  (soccer_days : ℝ) * soccer_hours_per_day + (band_days : ℝ) * band_hours_per_day

/-- Theorem stating that Josh spends 9 hours on extracurricular activities in a week -/
theorem josh_extracurricular_hours :
  total_extracurricular_hours = 9 := by
  sorry

end NUMINAMATH_CALUDE_josh_extracurricular_hours_l3835_383593


namespace NUMINAMATH_CALUDE_range_of_a_correct_l3835_383537

/-- Proposition p: For all x ∈ ℝ, ax^2 + ax + 1 > 0 always holds -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The function f(x) = 4x^2 - ax is monotonically increasing on [1, +∞) -/
def q (a : ℝ) : Prop := ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := {a : ℝ | a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)}

theorem range_of_a_correct (a : ℝ) : (p a ∨ q a) ∧ ¬(p a) → a ∈ range_of_a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_correct_l3835_383537


namespace NUMINAMATH_CALUDE_businessmen_one_beverage_businessmen_one_beverage_proof_l3835_383583

/-- The number of businessmen who drank only one type of beverage at a conference -/
theorem businessmen_one_beverage (total : ℕ) (coffee tea juice : ℕ) 
  (coffee_tea coffee_juice tea_juice : ℕ) (all_three : ℕ) : ℕ :=
  let total_businessmen : ℕ := 35
  let coffee_drinkers : ℕ := 18
  let tea_drinkers : ℕ := 15
  let juice_drinkers : ℕ := 8
  let coffee_and_tea : ℕ := 6
  let tea_and_juice : ℕ := 4
  let coffee_and_juice : ℕ := 3
  let all_beverages : ℕ := 2
  21

#check businessmen_one_beverage

/-- Proof that 21 businessmen drank only one type of beverage -/
theorem businessmen_one_beverage_proof : 
  businessmen_one_beverage 35 18 15 8 6 3 4 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_businessmen_one_beverage_businessmen_one_beverage_proof_l3835_383583


namespace NUMINAMATH_CALUDE_ant_population_growth_l3835_383577

/-- Represents the number of days passed -/
def days : ℕ := 4

/-- The growth factor for Species C per day -/
def growth_factor_C : ℕ := 2

/-- The growth factor for Species D per day -/
def growth_factor_D : ℕ := 4

/-- The total number of ants on Day 0 -/
def total_ants_day0 : ℕ := 35

/-- The total number of ants on Day 4 -/
def total_ants_day4 : ℕ := 3633

/-- The number of Species C ants on Day 0 -/
def species_C_day0 : ℕ := 22

/-- The number of Species D ants on Day 0 -/
def species_D_day0 : ℕ := 13

theorem ant_population_growth :
  species_C_day0 * growth_factor_C ^ days = 352 ∧
  species_C_day0 + species_D_day0 = total_ants_day0 ∧
  species_C_day0 * growth_factor_C ^ days + species_D_day0 * growth_factor_D ^ days = total_ants_day4 :=
by sorry

end NUMINAMATH_CALUDE_ant_population_growth_l3835_383577


namespace NUMINAMATH_CALUDE_bus_departure_interval_l3835_383517

/-- The departure interval of the bus -/
def x : ℝ := sorry

/-- The speed of the bus -/
def bus_speed : ℝ := sorry

/-- The speed of Xiao Hong -/
def xiao_hong_speed : ℝ := sorry

/-- The time interval between buses passing Xiao Hong from behind -/
def overtake_interval : ℝ := 6

/-- The time interval between buses approaching Xiao Hong head-on -/
def approach_interval : ℝ := 3

theorem bus_departure_interval :
  (overtake_interval * (bus_speed - xiao_hong_speed) = x * bus_speed) ∧
  (approach_interval * (bus_speed + xiao_hong_speed) = x * bus_speed) →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_bus_departure_interval_l3835_383517


namespace NUMINAMATH_CALUDE_music_class_participation_l3835_383514

theorem music_class_participation (jacob_total : ℕ) (jacob_participating : ℕ) (steve_total : ℕ)
  (h1 : jacob_total = 27)
  (h2 : jacob_participating = 18)
  (h3 : steve_total = 45) :
  (jacob_participating * steve_total) / jacob_total = 30 := by
  sorry

end NUMINAMATH_CALUDE_music_class_participation_l3835_383514


namespace NUMINAMATH_CALUDE_lcm_count_l3835_383533

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (9^9) (Nat.lcm (12^12) k) ≠ 18^18)) :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_count_l3835_383533


namespace NUMINAMATH_CALUDE_total_pizza_slices_l3835_383568

theorem total_pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 35) (h2 : slices_per_pizza = 12) : 
  num_pizzas * slices_per_pizza = 420 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_slices_l3835_383568


namespace NUMINAMATH_CALUDE_max_profit_at_70_l3835_383549

-- Define the linear function for weekly sales quantity
def sales_quantity (x : ℝ) : ℝ := -2 * x + 200

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - 40) * (sales_quantity x)

-- Theorem stating the maximum profit and the price at which it occurs
theorem max_profit_at_70 :
  ∃ (max_profit : ℝ), max_profit = 1800 ∧
  ∀ (x : ℝ), profit x ≤ max_profit ∧
  profit 70 = max_profit :=
sorry

#check max_profit_at_70

end NUMINAMATH_CALUDE_max_profit_at_70_l3835_383549


namespace NUMINAMATH_CALUDE_special_polygon_perimeter_l3835_383535

/-- A polygon with specific properties -/
structure SpecialPolygon where
  AB : ℝ
  AE : ℝ
  BD : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  angle_DBC : ℝ
  angle_BCD : ℝ
  angle_CDB : ℝ
  h_AB_eq_AE : AB = AE
  h_AB_val : AB = 120
  h_DE_val : DE = 226
  h_BD_val : BD = 115
  h_BD_eq_BC : BD = BC
  h_angle_DBC_eq_BCD : angle_DBC = angle_BCD
  h_triangle_BCD_equilateral : angle_DBC = 60 ∧ angle_BCD = 60 ∧ angle_CDB = 60
  h_CD_eq_BD : CD = BD

/-- The perimeter of the special polygon is 696 -/
theorem special_polygon_perimeter (p : SpecialPolygon) : 
  p.AB + p.AE + p.BD + p.BC + p.CD + p.DE = 696 := by
  sorry


end NUMINAMATH_CALUDE_special_polygon_perimeter_l3835_383535


namespace NUMINAMATH_CALUDE_sequence_general_term_l3835_383508

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n > 0, a (n - 1)^2 = a n^2 + 4) →
  a 1 = 1 →
  (∀ n > 0, a n > 0) →
  ∀ n > 0, a n = Real.sqrt (4 * n - 3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3835_383508


namespace NUMINAMATH_CALUDE_zero_exponent_is_one_l3835_383546

theorem zero_exponent_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_is_one_l3835_383546


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l3835_383571

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (distance_between : ℕ) : ℕ :=
  (num_trees - 1) * distance_between

/-- Theorem: The length of a yard with 26 trees planted at equal distances,
    with one tree at each end and 16 meters between consecutive trees, is 400 meters. -/
theorem yard_length_26_trees : yard_length 26 16 = 400 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l3835_383571


namespace NUMINAMATH_CALUDE_preferred_numbers_count_l3835_383500

/-- A function that counts the number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- A function that counts the number of four-digit "preferred" numbers. -/
def count_preferred_numbers : ℕ :=
  -- Numbers with two 8s, not in the first position
  choose 3 2 * 8 * 9 +
  -- Numbers with two 8s, including in the first position
  choose 3 1 * 9 * 9 +
  -- Numbers with four 8s
  1

/-- Theorem stating that the count of four-digit "preferred" numbers is 460. -/
theorem preferred_numbers_count : count_preferred_numbers = 460 := by sorry

end NUMINAMATH_CALUDE_preferred_numbers_count_l3835_383500


namespace NUMINAMATH_CALUDE_incorrect_factorization_l3835_383592

theorem incorrect_factorization (x : ℝ) : x^2 - 7*x + 12 ≠ x*(x - 7) + 12 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_factorization_l3835_383592


namespace NUMINAMATH_CALUDE_max_value_on_curve_l3835_383551

-- Define the curve
def on_curve (x y b : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the function to maximize
def f (x y : ℝ) : ℝ := x^2 + 2*y

-- State the theorem
theorem max_value_on_curve (b : ℝ) (h : b > 0) :
  (∃ (x y : ℝ), on_curve x y b ∧ 
    ∀ (x' y' : ℝ), on_curve x' y' b → f x y ≥ f x' y') →
  ((0 < b ∧ b ≤ 4 → ∃ (x y : ℝ), on_curve x y b ∧ f x y = b^2/4 + 4) ∧
   (b > 4 → ∃ (x y : ℝ), on_curve x y b ∧ f x y = 2*b)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l3835_383551


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l3835_383587

-- Define the function f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_satisfies_properties :
  -- Property 1: f(x₁x₂) = f(x₁)f(x₂)
  (∀ x₁ x₂ : ℝ, f (x₁ * x₂) = f x₁ * f x₂) ∧
  -- Property 2: f'(x) > 0 for x ∈ (0, +∞)
  (∀ x : ℝ, x > 0 → HasDerivAt f (2 * x) x) ∧
  (∀ x : ℝ, x > 0 → 2 * x > 0) ∧
  -- Property 3: f'(x) is an odd function
  (∀ x : ℝ, HasDerivAt f (2 * (-x)) (-x) ∧ HasDerivAt f (2 * x) x ∧ 2 * (-x) = -(2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_properties_l3835_383587


namespace NUMINAMATH_CALUDE_double_reflection_result_l3835_383505

def reflect_over_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_over_y_axis (reflect_over_y_equals_x p)

theorem double_reflection_result :
  double_reflection (7, -3) = (3, 7) := by sorry

end NUMINAMATH_CALUDE_double_reflection_result_l3835_383505


namespace NUMINAMATH_CALUDE_average_age_combined_l3835_383561

theorem average_age_combined (n_students : ℕ) (avg_age_students : ℝ)
                              (n_parents : ℕ) (avg_age_parents : ℝ)
                              (n_teachers : ℕ) (avg_age_teachers : ℝ) :
  n_students = 40 →
  avg_age_students = 10 →
  n_parents = 60 →
  avg_age_parents = 35 →
  n_teachers = 5 →
  avg_age_teachers = 45 →
  (n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers) /
  (n_students + n_parents + n_teachers : ℝ) = 26 := by
  sorry

#check average_age_combined

end NUMINAMATH_CALUDE_average_age_combined_l3835_383561


namespace NUMINAMATH_CALUDE_small_bottles_sold_percentage_l3835_383513

theorem small_bottles_sold_percentage 
  (initial_small : ℕ) 
  (initial_big : ℕ) 
  (big_sold_percent : ℚ) 
  (total_remaining : ℕ) :
  initial_small = 6000 →
  initial_big = 10000 →
  big_sold_percent = 15/100 →
  total_remaining = 13780 →
  ∃ (small_sold_percent : ℚ),
    small_sold_percent = 12/100 ∧
    (initial_small * (1 - small_sold_percent)).floor + 
    (initial_big * (1 - big_sold_percent)).floor = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_small_bottles_sold_percentage_l3835_383513


namespace NUMINAMATH_CALUDE_remainder_theorem_l3835_383574

theorem remainder_theorem (n m : ℤ) (q2 : ℤ) 
  (h1 : n % 11 = 1) 
  (h2 : m % 17 = 3) 
  (h3 : m = 17 * q2 + 3) : 
  (5 * n + 3 * m) % 11 = (3 + 7 * q2) % 11 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3835_383574


namespace NUMINAMATH_CALUDE_square_window_side_length_l3835_383519

/-- Represents the dimensions of a glass pane -/
structure GlassPane where
  height : ℝ
  width : ℝ
  ratio : height / width = 5 / 2

/-- Represents the dimensions of a square window -/
structure SquareWindow where
  pane : GlassPane
  border_width : ℝ
  side_length : ℝ

/-- Theorem stating the side length of the square window -/
theorem square_window_side_length 
  (window : SquareWindow)
  (h1 : window.border_width = 2)
  (h2 : window.side_length = 4 * window.pane.width + 5 * window.border_width)
  (h3 : window.side_length = 2 * window.pane.height + 3 * window.border_width) :
  window.side_length = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_window_side_length_l3835_383519


namespace NUMINAMATH_CALUDE_cruz_marbles_l3835_383563

/-- The number of marbles Atticus has -/
def atticus : ℕ := 4

/-- The number of marbles Jensen has -/
def jensen : ℕ := 2 * atticus

/-- The number of marbles Cruz has -/
def cruz : ℕ := 20 - (atticus + jensen)

/-- The total number of marbles -/
def total : ℕ := atticus + jensen + cruz

theorem cruz_marbles :
  (3 * total = 60) ∧ (atticus = 4) ∧ (jensen = 2 * atticus) → cruz = 8 := by
  sorry


end NUMINAMATH_CALUDE_cruz_marbles_l3835_383563


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3835_383523

theorem quadratic_equal_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + m - 1 = 0 ∧
    ∀ y : ℝ, y^2 - 4*y + m - 1 = 0 → y = x) →
  m = 5 ∧ ∃ x : ℝ, x^2 - 4*x + m - 1 = 0 ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3835_383523


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3835_383550

theorem unique_solution_exponential_equation (p q : ℝ) :
  (∀ x : ℝ, 2^(p*x + q) = p * 2^x + q) → p = 1 ∧ q = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l3835_383550


namespace NUMINAMATH_CALUDE_min_production_quantity_l3835_383518

def cost_function (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

def selling_price : ℝ := 25

theorem min_production_quantity :
  ∃ (min_x : ℝ), min_x = 150 ∧
  ∀ (x : ℝ), x ∈ Set.Ioo 0 240 →
    (selling_price * x ≥ cost_function x ↔ x ≥ min_x) :=
sorry

end NUMINAMATH_CALUDE_min_production_quantity_l3835_383518


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_problem_l3835_383525

-- Define the cyclic quadrilateral ABCD
def CyclicQuadrilateral (A B C D : Point) : Prop := sorry

-- Define the angle measure
def AngleMeasure (P Q R : Point) : ℝ := sorry

-- Define a point inside a triangle
def PointInsideTriangle (X A B C : Point) : Prop := sorry

-- Define angle bisector
def AngleBisector (A X B C : Point) : Prop := sorry

theorem cyclic_quadrilateral_angle_problem 
  (A B C D X : Point) 
  (h1 : CyclicQuadrilateral A B C D) 
  (h2 : AngleMeasure A D B = 48)
  (h3 : AngleMeasure B D C = 56)
  (h4 : PointInsideTriangle X A B C)
  (h5 : AngleMeasure B C X = 24)
  (h6 : AngleBisector A X B C) :
  AngleMeasure C B X = 38 := by
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_problem_l3835_383525


namespace NUMINAMATH_CALUDE_square_area_ratio_l3835_383501

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3835_383501


namespace NUMINAMATH_CALUDE_class_size_l3835_383536

theorem class_size (debate_only : ℕ) (singing_only : ℕ) (both : ℕ)
  (h1 : debate_only = 10)
  (h2 : singing_only = 18)
  (h3 : both = 17) :
  debate_only + singing_only + both - both = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_class_size_l3835_383536


namespace NUMINAMATH_CALUDE_evaluate_F_4_f_5_l3835_383579

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 3
def F (a b : ℝ) : ℝ := b^3 + a*b

-- State the theorem
theorem evaluate_F_4_f_5 : F 4 (f 5) = 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_F_4_f_5_l3835_383579


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3835_383511

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the focal length and real axis relationship
def focal_length_relation (a : ℝ) : Prop := 2 * (2 * a) = 4 * a

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem statement
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, hyperbola x y a b) →
  focal_length_relation a →
  (∀ x y, asymptote_equation x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3835_383511


namespace NUMINAMATH_CALUDE_x_equals_3_sufficient_not_necessary_for_x_squared_9_l3835_383576

theorem x_equals_3_sufficient_not_necessary_for_x_squared_9 :
  (∀ x : ℝ, x = 3 → x^2 = 9) ∧
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_x_equals_3_sufficient_not_necessary_for_x_squared_9_l3835_383576


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l3835_383507

theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.95 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (1 / 0.95 - 1) * 100 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l3835_383507


namespace NUMINAMATH_CALUDE_ashley_family_movie_cost_l3835_383585

/-- Calculates the total cost of a movie outing for Ashley's family --/
def movie_outing_cost (
  child_ticket_price : ℝ)
  (adult_ticket_price_diff : ℝ)
  (senior_ticket_price_diff : ℝ)
  (morning_discount : ℝ)
  (voucher_discount : ℝ)
  (popcorn_price : ℝ)
  (soda_price : ℝ)
  (candy_price : ℝ)
  (concession_discount : ℝ) : ℝ :=
  let adult_ticket_price := child_ticket_price + adult_ticket_price_diff
  let senior_ticket_price := adult_ticket_price - senior_ticket_price_diff
  let ticket_cost := 2 * adult_ticket_price + 4 * child_ticket_price + senior_ticket_price
  let discounted_ticket_cost := ticket_cost * (1 - morning_discount) - child_ticket_price - voucher_discount
  let concession_cost := 3 * popcorn_price + 2 * soda_price + candy_price
  let discounted_concession_cost := concession_cost * (1 - concession_discount)
  discounted_ticket_cost + discounted_concession_cost

/-- Theorem stating the total cost of Ashley's family's movie outing --/
theorem ashley_family_movie_cost :
  movie_outing_cost 4.25 3.50 1.75 0.10 4.00 5.25 3.50 4.00 0.10 = 50.47 := by
  sorry

end NUMINAMATH_CALUDE_ashley_family_movie_cost_l3835_383585


namespace NUMINAMATH_CALUDE_money_distribution_l3835_383526

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (ac_sum : A + C = 200)
  (c_amount : C = 30) :
  B + C = 330 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3835_383526


namespace NUMINAMATH_CALUDE_product_powers_equality_l3835_383527

theorem product_powers_equality (a : ℝ) : 
  let b := a - 1
  (a + b) * (a^2 + b^2) * (a^4 + b^4) * (a^8 + b^8) * (a^16 + b^16) * (a^32 + b^32) = a^64 - b^64 := by
  sorry

end NUMINAMATH_CALUDE_product_powers_equality_l3835_383527


namespace NUMINAMATH_CALUDE_tan_sixty_degrees_l3835_383565

theorem tan_sixty_degrees : Real.tan (π / 3) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_sixty_degrees_l3835_383565


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3835_383557

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3835_383557


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l3835_383506

-- Define the * operation
def star (a b : ℝ) : ℝ := a + b + a * b

-- Theorem statement
theorem no_distributive_laws_hold :
  ∃ x y z : ℝ,
    (star x (y + z) ≠ star (star x y) (star x z)) ∧
    (x + star y z ≠ star (x + y) (x + z)) ∧
    (star x (star y z) ≠ star (star x y) (star x z)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_distributive_laws_hold_l3835_383506


namespace NUMINAMATH_CALUDE_bills_naps_l3835_383569

theorem bills_naps (total_hours : ℕ) (work_hours : ℕ) (nap_duration : ℕ) : 
  total_hours = 96 → work_hours = 54 → nap_duration = 7 → 
  (total_hours - work_hours) / nap_duration = 6 := by
sorry

end NUMINAMATH_CALUDE_bills_naps_l3835_383569


namespace NUMINAMATH_CALUDE_opposite_vectors_properties_l3835_383589

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def are_opposite (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ b = -a

theorem opposite_vectors_properties {a b : V} (h : are_opposite a b) :
  (∃ (k : ℝ), b = k • a) ∧  -- a is parallel to b
  a ≠ b ∧                   -- a ≠ b
  ‖a‖ = ‖b‖ ∧              -- |a| = |b|
  b = -a :=                 -- b = -a
by sorry

end NUMINAMATH_CALUDE_opposite_vectors_properties_l3835_383589


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3835_383590

theorem larger_integer_problem (x y : ℤ) : 
  y - x = 8 → x * y = 272 → max x y = 17 := by sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3835_383590


namespace NUMINAMATH_CALUDE_monomial_product_l3835_383528

/-- Given two monomials 4x⁴y² and 3x²y³, prove that their product is 12x⁶y⁵ -/
theorem monomial_product :
  ∀ (x y : ℝ), (4 * x^4 * y^2) * (3 * x^2 * y^3) = 12 * x^6 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_monomial_product_l3835_383528


namespace NUMINAMATH_CALUDE_tangent_point_divides_equally_l3835_383530

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateralWithInscribedCircle where
  -- Sides of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Ensure all sides are positive
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  -- The quadrilateral is cyclic (inscribed in a circle)
  is_cyclic : True
  -- The quadrilateral has an inscribed circle
  has_inscribed_circle : True

/-- Theorem: In a cyclic quadrilateral with an inscribed circle, 
    if the consecutive sides have lengths 80, 120, 100, and 140, 
    then the point of tangency of the inscribed circle on the side 
    of length 100 divides it into two equal segments. -/
theorem tangent_point_divides_equally 
  (Q : CyclicQuadrilateralWithInscribedCircle) 
  (h1 : Q.a = 80) 
  (h2 : Q.b = 120) 
  (h3 : Q.c = 100) 
  (h4 : Q.d = 140) : 
  ∃ (x y : ℝ), x + y = 100 ∧ x = y :=
sorry

end NUMINAMATH_CALUDE_tangent_point_divides_equally_l3835_383530


namespace NUMINAMATH_CALUDE_max_abs_Z_on_circle_l3835_383595

open Complex

theorem max_abs_Z_on_circle (Z : ℂ) (h : abs (Z - (3 + 4*I)) = 1) :
  ∃ (M : ℝ), M = 6 ∧ ∀ (W : ℂ), abs (W - (3 + 4*I)) = 1 → abs W ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_Z_on_circle_l3835_383595


namespace NUMINAMATH_CALUDE_aisha_driving_problem_l3835_383581

theorem aisha_driving_problem (initial_distance : ℝ) (initial_speed : ℝ) (second_speed : ℝ) (average_speed : ℝ) :
  initial_distance = 18 →
  initial_speed = 36 →
  second_speed = 60 →
  average_speed = 48 →
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = average_speed ∧
    additional_distance = 30 :=
by sorry

end NUMINAMATH_CALUDE_aisha_driving_problem_l3835_383581


namespace NUMINAMATH_CALUDE_union_complement_equality_l3835_383552

def U : Finset Nat := {0, 1, 2, 4, 6, 8}
def M : Finset Nat := {0, 4, 6}
def N : Finset Nat := {0, 1, 6}

theorem union_complement_equality : M ∪ (U \ N) = {0, 2, 4, 6, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l3835_383552


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3835_383522

-- Define the expansion
def expansion (a : ℝ) (x : ℝ) : ℝ := (2 + a * x) * (1 + x)^5

-- Define the coefficient of x^2
def coeff_x2 (a : ℝ) : ℝ := 20 + 5 * a

-- Theorem statement
theorem sum_of_coefficients (a : ℝ) (h : coeff_x2 a = 15) : 
  ∃ (sum : ℝ), sum = expansion a 1 ∧ sum = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3835_383522


namespace NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_region_l3835_383548

/-- The perimeter of a region bounded by four quarter-circle arcs constructed at each corner of a square with sides measuring 4/π is equal to 4. -/
theorem perimeter_of_quarter_circle_bounded_region : 
  let square_side : ℝ := 4 / Real.pi
  let quarter_circle_radius : ℝ := square_side / 2
  let quarter_circle_perimeter : ℝ := Real.pi * quarter_circle_radius / 2
  let total_perimeter : ℝ := 4 * quarter_circle_perimeter
  total_perimeter = 4 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_quarter_circle_bounded_region_l3835_383548
