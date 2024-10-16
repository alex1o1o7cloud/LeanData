import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l3992_399296

/-- The value of k for which the asymptotes of the hyperbola x^2 - y^2/k^2 = 1 
    are tangent to the circle x^2 + (y-2)^2 = 1 -/
theorem hyperbola_asymptote_tangent_circle (k : ℝ) :
  k > 0 →
  (∀ x y : ℝ, x^2 - y^2/k^2 = 1 → 
    ∃ m : ℝ, (∀ t : ℝ, (x = t ∧ y = k*t) ∨ (x = t ∧ y = -k*t)) →
      (∃ x₀ y₀ : ℝ, x₀^2 + (y₀-2)^2 = 1 ∧
        (x₀ - x)^2 + (y₀ - y)^2 = 1)) →
  k = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_tangent_circle_l3992_399296


namespace NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l3992_399213

-- Define a quadrilateral in 2D space
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a function to check if a quadrilateral is a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  let BA := (q.B.1 - q.A.1, q.B.2 - q.A.2)
  let DC := (q.D.1 - q.C.1, q.D.2 - q.C.2)
  BA = DC

-- Theorem statement
theorem quadrilateral_is_parallelogram (q : Quadrilateral) (O : ℝ × ℝ) :
  (q.A.1 - O.1, q.A.2 - O.2) + (q.C.1 - O.1, q.C.2 - O.2) =
  (q.B.1 - O.1, q.B.2 - O.2) + (q.D.1 - O.1, q.D.2 - O.2) →
  is_parallelogram q :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l3992_399213


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l3992_399255

theorem max_sum_with_constraint (a b : ℝ) (h : a^2 - a*b + b^2 = 1) :
  a + b ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 - a₀*b₀ + b₀^2 = 1 ∧ a₀ + b₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l3992_399255


namespace NUMINAMATH_CALUDE_mary_added_candy_l3992_399234

/-- Proof that Mary added 10 pieces of candy to her collection --/
theorem mary_added_candy (megan_candy : ℕ) (mary_total : ℕ) (h1 : megan_candy = 5) (h2 : mary_total = 25) :
  mary_total - (3 * megan_candy) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_added_candy_l3992_399234


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l3992_399222

theorem solution_set_implies_a_value (a b : ℝ) : 
  (∀ x, |x - a| < b ↔ 2 < x ∧ x < 4) → a = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l3992_399222


namespace NUMINAMATH_CALUDE_problem_solution_l3992_399229

theorem problem_solution (a b c : ℚ) 
  (h1 : a + b + c = 72)
  (h2 : a + 4 = b - 8)
  (h3 : a + 4 = 4 * c) : 
  a = 236 / 9 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3992_399229


namespace NUMINAMATH_CALUDE_log_inequality_l3992_399284

theorem log_inequality (a b c : ℝ) (ha : a = Real.log 6 / Real.log 4)
  (hb : b = Real.log 0.2 / Real.log 4) (hc : c = Real.log 3 / Real.log 2) :
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l3992_399284


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l3992_399271

theorem simplify_radical_expression : 
  Real.sqrt 80 - 3 * Real.sqrt 20 + Real.sqrt 500 / Real.sqrt 5 + 2 * Real.sqrt 45 = 4 * Real.sqrt 5 + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l3992_399271


namespace NUMINAMATH_CALUDE_bank_account_withdrawal_l3992_399291

theorem bank_account_withdrawal (initial_balance : ℚ) : 
  (initial_balance > 0) →
  (initial_balance - 200 + (1/2) * (initial_balance - 200) = 450) →
  (200 / initial_balance = 2/5) := by
sorry

end NUMINAMATH_CALUDE_bank_account_withdrawal_l3992_399291


namespace NUMINAMATH_CALUDE_customer_count_is_twenty_l3992_399298

/-- The number of customers who bought marbles from Mr Julien's store -/
def number_of_customers (initial_marbles final_marbles marbles_per_customer : ℕ) : ℕ :=
  (initial_marbles - final_marbles) / marbles_per_customer

/-- Theorem stating that the number of customers who bought marbles is 20 -/
theorem customer_count_is_twenty :
  number_of_customers 400 100 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_customer_count_is_twenty_l3992_399298


namespace NUMINAMATH_CALUDE_point_division_theorem_l3992_399248

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q = (5/8)*C + (3/8)*D -/
theorem point_division_theorem (C D Q : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) 
  (h2 : ∃ k : ℝ, k > 0 ∧ (Q - C) = k • (3 • (D - C))) :
  Q = (5/8) • C + (3/8) • D :=
sorry

end NUMINAMATH_CALUDE_point_division_theorem_l3992_399248


namespace NUMINAMATH_CALUDE_maggie_tractor_hours_l3992_399202

/-- Represents Maggie's work schedule and income for a week. -/
structure WorkWeek where
  tractorHours : ℕ
  officeHours : ℕ
  deliveryHours : ℕ
  totalIncome : ℕ

/-- Checks if a work week satisfies the given conditions. -/
def isValidWorkWeek (w : WorkWeek) : Prop :=
  w.officeHours = 2 * w.tractorHours ∧
  w.deliveryHours = w.officeHours - 3 ∧
  w.totalIncome = 10 * w.officeHours + 12 * w.tractorHours + 15 * w.deliveryHours

/-- Theorem stating that given the conditions, Maggie spent 15 hours driving the tractor. -/
theorem maggie_tractor_hours :
  ∃ (w : WorkWeek), isValidWorkWeek w ∧ w.totalIncome = 820 → w.tractorHours = 15 :=
by sorry


end NUMINAMATH_CALUDE_maggie_tractor_hours_l3992_399202


namespace NUMINAMATH_CALUDE_expression_value_l3992_399274

theorem expression_value : (100 - (3000 - 300)) + (3000 - (300 - 100)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3992_399274


namespace NUMINAMATH_CALUDE_range_of_negative_values_l3992_399254

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on (-∞, 0] if
    for all x, y ∈ (-∞, 0], x ≤ y implies f(x) ≥ f(y) -/
def MonoDecreasingNonPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 0 → f x ≥ f y

/-- The main theorem -/
theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_mono : MonoDecreasingNonPositive f)
  (h_f1 : f 1 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l3992_399254


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3992_399212

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 11) → (x + x = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3992_399212


namespace NUMINAMATH_CALUDE_square_area_proof_l3992_399261

theorem square_area_proof (x : ℝ) : 
  (3 * x - 12 = 24 - 2 * x) → 
  ((3 * x - 12) ^ 2 = 92.16) := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l3992_399261


namespace NUMINAMATH_CALUDE_square_of_1035_l3992_399247

theorem square_of_1035 : (1035 : ℕ)^2 = 1071225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1035_l3992_399247


namespace NUMINAMATH_CALUDE_binary_263_ones_minus_zeros_l3992_399237

def binary_representation (n : Nat) : List Nat :=
  sorry

def count_zeros (l : List Nat) : Nat :=
  sorry

def count_ones (l : List Nat) : Nat :=
  sorry

theorem binary_263_ones_minus_zeros :
  let bin_263 := binary_representation 263
  let x := count_zeros bin_263
  let y := count_ones bin_263
  y - x = 0 := by sorry

end NUMINAMATH_CALUDE_binary_263_ones_minus_zeros_l3992_399237


namespace NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l3992_399206

theorem exists_solution_for_calendar_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l3992_399206


namespace NUMINAMATH_CALUDE_damage_proportion_l3992_399216

/-- The proportion of a 3x2 rectangle that can be reached by the midpoint of a 2-unit line segment
    rotating freely within the rectangle -/
theorem damage_proportion (rectangle_length : Real) (rectangle_width : Real) (log_length : Real) :
  rectangle_length = 3 ∧ rectangle_width = 2 ∧ log_length = 2 →
  (rectangle_length * rectangle_width - 4 * (Real.pi / 4 * (log_length / 2)^2)) / (rectangle_length * rectangle_width) = 1 - Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_damage_proportion_l3992_399216


namespace NUMINAMATH_CALUDE_oil_drop_probability_l3992_399205

theorem oil_drop_probability (circle_diameter : ℝ) (square_side : ℝ) 
  (h1 : circle_diameter = 3) 
  (h2 : square_side = 1) : 
  (square_side ^ 2) / (π * (circle_diameter / 2) ^ 2) = 4 / (9 * π) :=
sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l3992_399205


namespace NUMINAMATH_CALUDE_unknown_bill_value_l3992_399203

/-- Represents the contents of Ali's wallet -/
structure Wallet where
  five_dollar_bills : ℕ
  unknown_bill : ℕ
  total_amount : ℕ

/-- Theorem stating that given the conditions of Ali's wallet, the unknown bill is $10 -/
theorem unknown_bill_value (w : Wallet) 
  (h1 : w.five_dollar_bills = 7)
  (h2 : w.total_amount = 45) :
  w.unknown_bill = 10 := by
  sorry

#check unknown_bill_value

end NUMINAMATH_CALUDE_unknown_bill_value_l3992_399203


namespace NUMINAMATH_CALUDE_ellipse_circle_dot_product_range_l3992_399245

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 9) + (P.2^2 / 8) = 1

-- Define the circle
def is_on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 1)^2 + P.2^2 = 1

-- Define a diameter of the circle
def is_diameter (A B : ℝ × ℝ) : Prop :=
  is_on_circle A ∧ is_on_circle B ∧ (A.1 + B.1 = 2) ∧ (A.2 + B.2 = 0)

-- Define the dot product
def dot_product (P A B : ℝ × ℝ) : ℝ :=
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2)

theorem ellipse_circle_dot_product_range :
  ∀ (P A B : ℝ × ℝ),
    is_on_ellipse P →
    is_diameter A B →
    3 ≤ dot_product P A B ∧ dot_product P A B ≤ 15 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_circle_dot_product_range_l3992_399245


namespace NUMINAMATH_CALUDE_kevins_toad_feeding_l3992_399260

/-- Given Kevin's toad feeding scenario, prove the number of worms per toad. -/
theorem kevins_toad_feeding (num_toads : ℕ) (minutes_per_worm : ℕ) (total_hours : ℕ) 
  (h1 : num_toads = 8)
  (h2 : minutes_per_worm = 15)
  (h3 : total_hours = 6) :
  (total_hours * 60) / minutes_per_worm / num_toads = 3 := by
  sorry

#check kevins_toad_feeding

end NUMINAMATH_CALUDE_kevins_toad_feeding_l3992_399260


namespace NUMINAMATH_CALUDE_special_hexagon_perimeter_l3992_399289

/-- An equilateral hexagon with specific properties -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- The hexagon is equilateral
  equilateral : True
  -- Three nonadjacent acute interior angles measure 45°
  three_angles_45 : True
  -- The area of the hexagon is 12√2
  area : side^2 * (3 * Real.sqrt 2 / 4 + Real.sqrt 3 / 2 - Real.sqrt 2 / 2) = 12 * Real.sqrt 2

/-- The perimeter of a hexagon is 6 times its side length -/
def perimeter (h : SpecialHexagon) : ℝ := 6 * h.side

/-- Theorem: The perimeter of the special hexagon is 24√2 -/
theorem special_hexagon_perimeter (h : SpecialHexagon) : perimeter h = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_perimeter_l3992_399289


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l3992_399256

def number_of_players : ℕ := 12
def lineup_size : ℕ := 5
def number_of_twins : ℕ := 2

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem basketball_lineup_count : 
  (number_of_twins * choose (number_of_players - number_of_twins) (lineup_size - 1)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l3992_399256


namespace NUMINAMATH_CALUDE_log2_odd_and_increasing_l3992_399295

open Real

-- Define the function f(x) = log₂ x
noncomputable def f (x : ℝ) : ℝ := log x / log 2

-- Theorem statement
theorem log2_odd_and_increasing :
  (∀ x > 0, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_log2_odd_and_increasing_l3992_399295


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l3992_399210

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + k + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ↔ k ≤ 3 :=
by sorry

theorem specific_root_condition (k : ℝ) (x₁ x₂ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + k + 1
  (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ 3/x₁ + 3/x₂ = x₁*x₂ - 4) → k = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l3992_399210


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l3992_399215

-- Define the function y(x)
def y (x : ℝ) : ℝ := (x + 1)^2

-- State the theorem
theorem solution_satisfies_conditions :
  (∀ x, (deriv^[2] y) x = 2) ∧ 
  y 0 = 1 ∧ 
  (deriv y) 0 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_solution_satisfies_conditions_l3992_399215


namespace NUMINAMATH_CALUDE_point_on_axes_l3992_399279

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The coordinate axes, represented as a set of points -/
def CoordinateAxes : Set Point :=
  {p : Point | p.x = 0 ∨ p.y = 0}

/-- Theorem: If xy = 0, then the point is on the coordinate axes -/
theorem point_on_axes (p : Point) (h : p.x * p.y = 0) : p ∈ CoordinateAxes := by
  sorry

end NUMINAMATH_CALUDE_point_on_axes_l3992_399279


namespace NUMINAMATH_CALUDE_not_perfect_square_l3992_399236

-- Define a function to create a number with n ones
def ones (n : ℕ) : ℕ := 
  (10^n - 1) / 9

-- Define our specific number N
def N (k : ℕ) : ℕ := 
  ones 300 * 10^k

-- Theorem statement
theorem not_perfect_square (k : ℕ) : 
  ¬ ∃ (m : ℕ), N k = m^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3992_399236


namespace NUMINAMATH_CALUDE_specific_arrangement_surface_area_l3992_399268

/-- Represents a cube arrangement with two layers --/
structure CubeArrangement where
  totalCubes : Nat
  layerSize : Nat
  cubeEdgeLength : Real

/-- Calculates the exposed surface area of the cube arrangement --/
def exposedSurfaceArea (arrangement : CubeArrangement) : Real :=
  sorry

/-- Theorem stating that the exposed surface area of the specific arrangement is 49 square meters --/
theorem specific_arrangement_surface_area :
  let arrangement : CubeArrangement := {
    totalCubes := 18,
    layerSize := 9,
    cubeEdgeLength := 1
  }
  exposedSurfaceArea arrangement = 49 := by sorry

end NUMINAMATH_CALUDE_specific_arrangement_surface_area_l3992_399268


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3992_399214

theorem arithmetic_calculation : 8 + (-2)^3 / (-4) * (-7 + 5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3992_399214


namespace NUMINAMATH_CALUDE_expression_simplification_l3992_399290

theorem expression_simplification (x : ℝ) (h : x = -2) :
  (1 - 2 / (x + 1)) / ((x^2 - x) / (x^2 - 1)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3992_399290


namespace NUMINAMATH_CALUDE_max_profit_is_12250_l3992_399262

/-- Represents the profit function for selling humidifiers -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 300 * x + 10000

/-- Represents the selling price of a humidifier -/
def selling_price (x : ℝ) : ℝ := 100 + x

/-- Represents the daily sales volume -/
def daily_sales (x : ℝ) : ℝ := 500 - 10 * x

/-- Theorem stating that the maximum profit is 12250 yuan -/
theorem max_profit_is_12250 :
  ∃ x : ℝ, 
    (∀ y : ℝ, profit_function y ≤ profit_function x) ∧ 
    profit_function x = 12250 ∧
    selling_price x = 115 :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_12250_l3992_399262


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3992_399283

def sum_of_two_digit_pairs (n : ℕ) : ℕ :=
  (n % 100) + ((n / 100) % 100) + ((n / 10000) % 100)

def alternating_sum_of_three_digit_groups (n : ℕ) : ℤ :=
  (n % 1000 : ℤ) - ((n / 1000) % 1000 : ℤ)

theorem smallest_addition_for_divisibility (n : ℕ) (k : ℕ) :
  (∀ m < k, ¬(456 ∣ (987654 + m))) ∧
  (456 ∣ (987654 + k)) ∧
  (19 ∣ sum_of_two_digit_pairs (987654 + k)) ∧
  (8 ∣ alternating_sum_of_three_digit_groups (987654 + k)) →
  k = 22 := by
  sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l3992_399283


namespace NUMINAMATH_CALUDE_race_distance_l3992_399249

/-- The race problem -/
theorem race_distance (a_time b_time : ℕ) (beat_distance : ℕ) (total_distance : ℕ) : 
  a_time = 36 →
  b_time = 45 →
  beat_distance = 20 →
  (total_distance : ℚ) / a_time * b_time = total_distance + beat_distance →
  total_distance = 80 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3992_399249


namespace NUMINAMATH_CALUDE_impossible_arrangement_l3992_399244

/-- Represents a person at the table -/
structure Person :=
  (id : Nat)

/-- Represents the circular table with 40 people -/
def Table := Fin 40 → Person

/-- Returns the number of people between two positions on the table -/
def distanceBetween (table : Table) (p1 p2 : Fin 40) : Nat :=
  sorry

/-- Checks if two people have a common acquaintance -/
def haveCommonAcquaintance (table : Table) (p1 p2 : Fin 40) : Prop :=
  sorry

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement (table : Table) : 
  ¬(∀ (p1 p2 : Fin 40), 
    (distanceBetween table p1 p2 % 2 = 0 → haveCommonAcquaintance table p1 p2) ∧
    (distanceBetween table p1 p2 % 2 = 1 → ¬haveCommonAcquaintance table p1 p2)) :=
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l3992_399244


namespace NUMINAMATH_CALUDE_three_vectors_with_zero_sum_and_unit_difference_l3992_399278

theorem three_vectors_with_zero_sum_and_unit_difference (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :
  ∃ (a b c : α), 
    a + b + c = 0 ∧ 
    ‖a + b - c‖ = 1 ∧ 
    ‖b + c - a‖ = 1 ∧ 
    ‖c + a - b‖ = 1 ∧
    ‖a‖ = (1 : ℝ) / 2 ∧ 
    ‖b‖ = (1 : ℝ) / 2 ∧ 
    ‖c‖ = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_three_vectors_with_zero_sum_and_unit_difference_l3992_399278


namespace NUMINAMATH_CALUDE_bigger_part_is_34_l3992_399269

theorem bigger_part_is_34 (x y : ℝ) (h1 : x + y = 54) (h2 : 10 * x + 22 * y = 780) :
  max x y = 34 := by
  sorry

end NUMINAMATH_CALUDE_bigger_part_is_34_l3992_399269


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3992_399228

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_minus_x_squared_odd : ∀ x : ℝ, f (-x) - (-x)^2 = -(f x - x^2)
axiom f_plus_2_pow_x_even : ∀ x : ℝ, f (-x) + 2^(-x) = f x + 2^x

-- Define the interval
def interval : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ -1}

-- State the theorem
theorem min_value_of_f_on_interval :
  ∃ x₀ ∈ interval, ∀ x ∈ interval, f x₀ ≤ f x ∧ f x₀ = 7/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l3992_399228


namespace NUMINAMATH_CALUDE_earlier_usage_time_correct_l3992_399223

/-- Represents a beer barrel with two taps -/
structure BeerBarrel where
  capacity : ℕ
  midwayTapRate : ℕ  -- minutes per litre
  bottomTapRate : ℕ  -- minutes per litre

/-- Calculates how much earlier the lower tap was used than usual -/
def earlierUsageTime (barrel : BeerBarrel) (usageTime : ℕ) : ℕ :=
  let drawnAmount := usageTime / barrel.bottomTapRate
  let midwayAmount := barrel.capacity / 2
  let remainingAmount := barrel.capacity - drawnAmount
  let excessAmount := remainingAmount - midwayAmount
  excessAmount * barrel.midwayTapRate

theorem earlier_usage_time_correct (barrel : BeerBarrel) (usageTime : ℕ) :
  barrel.capacity = 36 ∧ 
  barrel.midwayTapRate = 6 ∧ 
  barrel.bottomTapRate = 4 ∧ 
  usageTime = 16 →
  earlierUsageTime barrel usageTime = 84 := by
  sorry

#eval earlierUsageTime ⟨36, 6, 4⟩ 16

end NUMINAMATH_CALUDE_earlier_usage_time_correct_l3992_399223


namespace NUMINAMATH_CALUDE_malcolm_followers_l3992_399209

def total_followers (instagram : ℕ) (facebook : ℕ) : ℕ :=
  let twitter := (instagram + facebook) / 2
  let tiktok := 3 * twitter
  let youtube := tiktok + 510
  instagram + facebook + twitter + tiktok + youtube

theorem malcolm_followers : total_followers 240 500 = 3840 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_followers_l3992_399209


namespace NUMINAMATH_CALUDE_rational_expression_evaluation_l3992_399204

theorem rational_expression_evaluation : 
  let x : ℝ := 8
  (x^4 - 18*x^2 + 81) / (x^2 - 9) = 55 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_evaluation_l3992_399204


namespace NUMINAMATH_CALUDE_selected_students_l3992_399285

-- Define the set of students
inductive Student : Type
| A | B | C | D | E

-- Define a type for the selection of students
def Selection := Student → Prop

-- Define the conditions
def valid_selection (s : Selection) : Prop :=
  -- 3 students are selected
  (∃ (x y z : Student), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ s x ∧ s y ∧ s z ∧
    ∀ (w : Student), s w → (w = x ∨ w = y ∨ w = z)) ∧
  -- If A is selected, then B is selected and E is not selected
  (s Student.A → s Student.B ∧ ¬s Student.E) ∧
  -- If B or E is selected, then D is not selected
  ((s Student.B ∨ s Student.E) → ¬s Student.D) ∧
  -- At least one of C or D must be selected
  (s Student.C ∨ s Student.D)

-- Theorem statement
theorem selected_students (s : Selection) :
  valid_selection s → s Student.A → s Student.B ∧ s Student.C :=
by sorry

end NUMINAMATH_CALUDE_selected_students_l3992_399285


namespace NUMINAMATH_CALUDE_trig_ratio_sum_l3992_399251

theorem trig_ratio_sum (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 3)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_sum_l3992_399251


namespace NUMINAMATH_CALUDE_second_odd_integer_l3992_399241

theorem second_odd_integer (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n - 1 ∧ b = 2*n + 1 ∧ c = 2*n + 3) →  -- consecutive odd integers
  (a + c = 128) →                                      -- sum of first and third is 128
  b = 64                                               -- second integer is 64
:= by sorry

end NUMINAMATH_CALUDE_second_odd_integer_l3992_399241


namespace NUMINAMATH_CALUDE_cricket_team_size_l3992_399258

-- Define the number of team members
variable (n : ℕ)

-- Define the average age of the team
def team_average : ℝ := 25

-- Define the wicket keeper's age
def wicket_keeper_age : ℝ := team_average + 3

-- Define the average age of remaining players after excluding two members
def remaining_average : ℝ := team_average - 1

-- Define the total age of the team
def total_age : ℝ := n * team_average

-- Define the total age of remaining players
def remaining_total_age : ℝ := (n - 2) * remaining_average

-- Define the total age of the two excluded members
def excluded_total_age : ℝ := wicket_keeper_age + team_average

-- Theorem stating that the number of team members is 5
theorem cricket_team_size : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3992_399258


namespace NUMINAMATH_CALUDE_max_value_expression_l3992_399218

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26*a*b*c) ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3992_399218


namespace NUMINAMATH_CALUDE_inequality_solution_l3992_399287

theorem inequality_solution (a : ℝ) : 
  (∀ x > 0, (a * x - 9) * Real.log (2 * a / x) ≤ 0) ↔ a = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3992_399287


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l3992_399293

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no points of intersection in the real plane. -/
theorem line_circle_no_intersection :
  ¬ ∃ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_no_intersection_l3992_399293


namespace NUMINAMATH_CALUDE_amy_biking_distance_l3992_399277

theorem amy_biking_distance (yesterday_distance today_distance : ℝ) : 
  yesterday_distance = 12 →
  yesterday_distance + today_distance = 33 →
  today_distance < 2 * yesterday_distance →
  2 * yesterday_distance - today_distance = 3 :=
by sorry

end NUMINAMATH_CALUDE_amy_biking_distance_l3992_399277


namespace NUMINAMATH_CALUDE_guam_stay_duration_l3992_399259

/-- Calculates the number of days spent in Guam given the regular plan cost, international data cost per day, and total charges for the month. -/
def days_in_guam (regular_plan : ℚ) (intl_data_cost : ℚ) (total_charges : ℚ) : ℚ :=
  (total_charges - regular_plan) / intl_data_cost

/-- Theorem stating that given the specific costs in the problem, the number of days in Guam is 10. -/
theorem guam_stay_duration :
  let regular_plan : ℚ := 175
  let intl_data_cost : ℚ := 3.5
  let total_charges : ℚ := 210
  days_in_guam regular_plan intl_data_cost total_charges = 10 := by
  sorry

end NUMINAMATH_CALUDE_guam_stay_duration_l3992_399259


namespace NUMINAMATH_CALUDE_principal_calculation_l3992_399221

/-- Calculates the principal given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  simple_interest * 100 / (rate * time)

/-- Proves that the given conditions result in the correct principal -/
theorem principal_calculation :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 9
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 8925 := by
  sorry

end NUMINAMATH_CALUDE_principal_calculation_l3992_399221


namespace NUMINAMATH_CALUDE_midpoint_sum_x_invariant_l3992_399263

/-- Represents a polygon in the Cartesian plane -/
structure Polygon :=
  (vertices : List (ℝ × ℝ))

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon (p : Polygon) : Polygon :=
  sorry

/-- Computes the sum of x-coordinates of a polygon's vertices -/
def sumXCoordinates (p : Polygon) : ℝ :=
  sorry

/-- Theorem: The sum of x-coordinates remains constant through midpoint constructions -/
theorem midpoint_sum_x_invariant (Q₁ : Polygon) :
  sumXCoordinates Q₁ = 120 →
  let Q₂ := midpointPolygon Q₁
  let Q₃ := midpointPolygon Q₂
  let Q₄ := midpointPolygon Q₃
  sumXCoordinates Q₄ = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_x_invariant_l3992_399263


namespace NUMINAMATH_CALUDE_intersection_points_range_l3992_399266

/-- The curve equation x^2 + (y+3)^2 = 4 -/
def curve (x y : ℝ) : Prop := x^2 + (y+3)^2 = 4

/-- The line equation y = k(x-2) -/
def line (k x y : ℝ) : Prop := y = k*(x-2)

/-- The theorem stating the range of k for which the curve and line have two distinct intersection points -/
theorem intersection_points_range :
  ∀ k : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    curve x₁ y₁ ∧ curve x₂ y₂ ∧ 
    line k x₁ y₁ ∧ line k x₂ y₂ ∧ 
    y₁ ≥ -3 ∧ y₂ ≥ -3) ↔ 
  (5/12 < k ∧ k ≤ 3/4) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_range_l3992_399266


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l3992_399264

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (skew : Line → Line → Prop)
variable (lies_on : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line)
variable (intersects : Line → Line → Prop)

-- State the theorem
theorem line_intersection_theorem 
  (l₁ l₂ l : Line) (α β : Plane)
  (h1 : skew l₁ l₂)
  (h2 : lies_on l₁ α)
  (h3 : lies_on l₂ β)
  (h4 : l = intersection α β) :
  intersects l l₁ ∨ intersects l l₂ :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l3992_399264


namespace NUMINAMATH_CALUDE_water_saved_in_june_john_water_savings_l3992_399275

/-- Calculates the water saved in June by replacing an inefficient toilet with a more efficient one. -/
theorem water_saved_in_june (old_toilet_usage : ℝ) (flushes_per_day : ℕ) (water_reduction_percentage : ℝ) (days_in_june : ℕ) : ℝ :=
  let new_toilet_usage := old_toilet_usage * (1 - water_reduction_percentage)
  let daily_old_usage := old_toilet_usage * flushes_per_day
  let daily_new_usage := new_toilet_usage * flushes_per_day
  let june_old_usage := daily_old_usage * days_in_june
  let june_new_usage := daily_new_usage * days_in_june
  june_old_usage - june_new_usage

/-- Proves that John saved 1800 gallons of water in June by replacing his old toilet. -/
theorem john_water_savings : water_saved_in_june 5 15 0.8 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_water_saved_in_june_john_water_savings_l3992_399275


namespace NUMINAMATH_CALUDE_leadership_team_selection_l3992_399243

theorem leadership_team_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) : 
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_leadership_team_selection_l3992_399243


namespace NUMINAMATH_CALUDE_jerry_feather_ratio_l3992_399220

def jerryFeatherProblem (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left : ℕ) : Prop :=
  hawk_feathers = 6 ∧
  eagle_feathers = 17 * hawk_feathers ∧
  total_feathers = hawk_feathers + eagle_feathers ∧
  feathers_given = 10 ∧
  feathers_left = 49

theorem jerry_feather_ratio 
  (hawk_feathers eagle_feathers total_feathers feathers_given feathers_left : ℕ) 
  (h : jerryFeatherProblem hawk_feathers eagle_feathers total_feathers feathers_given feathers_left) : 
  ∃ (feathers_after_giving feathers_sold : ℕ),
    feathers_after_giving = total_feathers - feathers_given ∧
    feathers_sold = feathers_after_giving - feathers_left ∧
    2 * feathers_sold = feathers_after_giving :=
sorry

end NUMINAMATH_CALUDE_jerry_feather_ratio_l3992_399220


namespace NUMINAMATH_CALUDE_complex_number_solution_l3992_399267

theorem complex_number_solution (x y z : ℂ) 
  (sum_eq : x + y + z = 3)
  (sum_sq_eq : x^2 + y^2 + z^2 = 3)
  (sum_cube_eq : x^3 + y^3 + z^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_solution_l3992_399267


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_l3992_399231

theorem smallest_resolvable_debt (pig_value goat_value : ℕ) 
  (h_pig : pig_value = 350) (h_goat : goat_value = 240) :
  ∃ (debt : ℕ), debt > 0 ∧ 
  (∀ (d : ℕ), d > 0 → (∃ (p g : ℤ), d = pig_value * p + goat_value * g) → d ≥ debt) ∧
  (∃ (p g : ℤ), debt = pig_value * p + goat_value * g) :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_l3992_399231


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l3992_399211

theorem ellipse_hyperbola_eccentricity (m n : ℝ) (e₁ e₂ : ℝ) : 
  m > 1 → 
  n > 0 → 
  (∀ x y : ℝ, x^2 / m^2 + y^2 = 1 ↔ x^2 / n^2 - y^2 = 1) → 
  e₁ = Real.sqrt (1 - 1 / m^2) → 
  e₂ = Real.sqrt (1 + 1 / n^2) → 
  m > n ∧ e₁ * e₂ > 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_eccentricity_l3992_399211


namespace NUMINAMATH_CALUDE_percent_difference_l3992_399288

theorem percent_difference (x y p : ℝ) (h : x = y * (1 + p / 100)) : 
  p = 100 * ((x - y) / y) := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l3992_399288


namespace NUMINAMATH_CALUDE_systematic_sampling_constant_difference_l3992_399265

/-- Represents a sequence of 5 numbers -/
structure Sequence :=
  (numbers : Fin 5 → Nat)

/-- Checks if a sequence has a constant difference between consecutive elements -/
def hasConstantDifference (s : Sequence) (d : Nat) : Prop :=
  ∀ i : Fin 4, s.numbers (i.succ) - s.numbers i = d

/-- Systematic sampling function -/
def systematicSample (totalStudents : Nat) (sampleSize : Nat) : Sequence :=
  sorry

theorem systematic_sampling_constant_difference :
  let totalStudents : Nat := 55
  let sampleSize : Nat := 5
  let sampledSequence := systematicSample totalStudents sampleSize
  hasConstantDifference sampledSequence (totalStudents / sampleSize) :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_constant_difference_l3992_399265


namespace NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_of_100_l3992_399257

def is_multiple_of_100 (n : ℕ) : Prop := ∃ k : ℕ, n = 100 * k

def digit_product (n : ℕ) : ℕ := 
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

theorem least_multiple_with_digit_product_multiple_of_100 : 
  ∀ n : ℕ, is_multiple_of_100 n → n ≥ 100 → 
    (is_multiple_of_100 (digit_product n) → n ≥ 100) ∧
    (is_multiple_of_100 (digit_product 100)) :=
sorry

end NUMINAMATH_CALUDE_least_multiple_with_digit_product_multiple_of_100_l3992_399257


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l3992_399238

/-- Calculates the total cost of power cable for a neighborhood grid --/
theorem neighborhood_cable_cost
  (ew_streets : ℕ) (ew_length : ℕ)
  (ns_streets : ℕ) (ns_length : ℕ)
  (cable_per_street : ℕ) (cable_cost : ℕ) :
  ew_streets = 18 →
  ew_length = 2 →
  ns_streets = 10 →
  ns_length = 4 →
  cable_per_street = 5 →
  cable_cost = 2000 →
  (ew_streets * ew_length + ns_streets * ns_length) * cable_per_street * cable_cost = 760000 :=
by sorry

end NUMINAMATH_CALUDE_neighborhood_cable_cost_l3992_399238


namespace NUMINAMATH_CALUDE_sum_product_equality_l3992_399226

theorem sum_product_equality : 1235 + 2346 + 3412 * 2 + 4124 = 15529 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_equality_l3992_399226


namespace NUMINAMATH_CALUDE_product_xyzw_l3992_399273

theorem product_xyzw (x y z w : ℝ) (h1 : x + 1/y = 1) (h2 : y + 1/z + w = 1) (h3 : w = 2) (h4 : y ≠ 0) :
  x * y * z * w = -2 * y^2 + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_product_xyzw_l3992_399273


namespace NUMINAMATH_CALUDE_greatest_value_of_fraction_l3992_399246

theorem greatest_value_of_fraction (y : ℝ) : 
  (∀ θ : ℝ, y ≥ 14 / (5 + 3 * Real.sin θ)) → y = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_value_of_fraction_l3992_399246


namespace NUMINAMATH_CALUDE_feta_cheese_price_per_pound_l3992_399297

/-- Given Teresa's shopping list and total spent, calculate the price per pound of feta cheese --/
theorem feta_cheese_price_per_pound 
  (sandwich_price : ℝ) 
  (sandwich_quantity : ℕ) 
  (salami_price : ℝ) 
  (olive_price_per_pound : ℝ) 
  (olive_quantity : ℝ) 
  (bread_price : ℝ) 
  (feta_quantity : ℝ) 
  (total_spent : ℝ) 
  (h1 : sandwich_price = 7.75)
  (h2 : sandwich_quantity = 2)
  (h3 : salami_price = 4)
  (h4 : olive_price_per_pound = 10)
  (h5 : olive_quantity = 0.25)
  (h6 : bread_price = 2)
  (h7 : feta_quantity = 0.5)
  (h8 : total_spent = 40) :
  (total_spent - (sandwich_price * sandwich_quantity + salami_price + 3 * salami_price + 
  olive_price_per_pound * olive_quantity + bread_price)) / feta_quantity = 8 := by
sorry

end NUMINAMATH_CALUDE_feta_cheese_price_per_pound_l3992_399297


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l3992_399242

/-- Pascal's triangle as a function from row and column to value -/
def pascal (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The set of all four-digit numbers in Pascal's triangle -/
def four_digit_pascal : Set ℕ :=
  {n | ∃ (i j : ℕ), pascal i j = n ∧ is_four_digit n}

/-- The third smallest element in a set of natural numbers -/
noncomputable def third_smallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal :
  third_smallest four_digit_pascal = 1002 := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l3992_399242


namespace NUMINAMATH_CALUDE_economy_to_luxury_ratio_l3992_399294

/-- Represents the ratio between two quantities -/
structure Ratio where
  antecedent : ℕ
  consequent : ℕ

/-- Represents the inventory of a car dealership -/
structure CarInventory where
  economy_to_suv : Ratio
  luxury_to_suv : Ratio

theorem economy_to_luxury_ratio (inventory : CarInventory) 
  (h1 : inventory.economy_to_suv = Ratio.mk 4 1)
  (h2 : inventory.luxury_to_suv = Ratio.mk 8 1) :
  Ratio.mk 1 2 = 
    Ratio.mk 
      (inventory.economy_to_suv.antecedent * inventory.luxury_to_suv.consequent)
      (inventory.economy_to_suv.consequent * inventory.luxury_to_suv.antecedent) :=
by sorry

end NUMINAMATH_CALUDE_economy_to_luxury_ratio_l3992_399294


namespace NUMINAMATH_CALUDE_opposite_of_two_l3992_399299

theorem opposite_of_two : 
  ∃ x : ℝ, x + 2 = 0 ∧ x = -2 :=
sorry

end NUMINAMATH_CALUDE_opposite_of_two_l3992_399299


namespace NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l3992_399224

/-- Represents a dodecahedron -/
structure Dodecahedron where
  vertices : Finset (Fin 20)
  faces : Finset (Fin 12)
  is_pentagonal : faces → Prop
  vertex_face_incidence : vertices → faces → Prop
  three_faces_per_vertex : ∀ v : vertices, ∃! (f1 f2 f3 : faces), 
    vertex_face_incidence v f1 ∧ vertex_face_incidence v f2 ∧ vertex_face_incidence v f3 ∧ f1 ≠ f2 ∧ f2 ≠ f3 ∧ f1 ≠ f3

/-- An interior diagonal in a dodecahedron -/
def interior_diagonal (d : Dodecahedron) (v1 v2 : d.vertices) : Prop :=
  v1 ≠ v2 ∧ ∀ f : d.faces, ¬(d.vertex_face_incidence v1 f ∧ d.vertex_face_incidence v2 f)

/-- The number of interior diagonals in a dodecahedron -/
def num_interior_diagonals (d : Dodecahedron) : ℕ :=
  (d.vertices.card * (d.vertices.card - 3)) / 2

/-- Theorem stating that a dodecahedron has 170 interior diagonals -/
theorem dodecahedron_interior_diagonals (d : Dodecahedron) : 
  num_interior_diagonals d = 170 := by sorry

end NUMINAMATH_CALUDE_dodecahedron_interior_diagonals_l3992_399224


namespace NUMINAMATH_CALUDE_cubic_identity_l3992_399253

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l3992_399253


namespace NUMINAMATH_CALUDE_average_and_difference_l3992_399201

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 53 → |45 - y| = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l3992_399201


namespace NUMINAMATH_CALUDE_expand_expression_1_expand_expression_2_expand_expression_3_simplified_calculation_l3992_399270

-- Problem 1
theorem expand_expression_1 (x y : ℝ) :
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
sorry

-- Problem 2
theorem expand_expression_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
sorry

-- Problem 3
theorem expand_expression_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
sorry

-- Problem 4
theorem simplified_calculation :
  2010^2 - 2011 * 2009 = 1 :=
sorry

end NUMINAMATH_CALUDE_expand_expression_1_expand_expression_2_expand_expression_3_simplified_calculation_l3992_399270


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3992_399272

theorem x_plus_y_value (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (25 : ℝ) ^ y = 5 ^ (x - 7)) : 
  x + y = 8.5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3992_399272


namespace NUMINAMATH_CALUDE_total_laundry_time_l3992_399235

/-- Represents the laundry times for a person -/
structure LaundryTimes where
  washingTimes : List Nat
  dryingTimes : List Nat

/-- Calculates the total laundry time for a person -/
def totalLaundryTime (lt : LaundryTimes) : Nat :=
  lt.washingTimes.sum + lt.dryingTimes.sum

/-- The laundry times for Carlos -/
def carlosLaundry : LaundryTimes :=
  { washingTimes := [30, 45, 40, 50, 35]
    dryingTimes := [85, 95] }

/-- The laundry times for Maria -/
def mariaLaundry : LaundryTimes :=
  { washingTimes := [25, 55, 40]
    dryingTimes := [80] }

/-- The laundry times for José -/
def joseLaundry : LaundryTimes :=
  { washingTimes := [20, 45, 35, 60]
    dryingTimes := [90] }

/-- Theorem stating the total laundry time for all family members -/
theorem total_laundry_time :
  totalLaundryTime carlosLaundry +
  totalLaundryTime mariaLaundry +
  totalLaundryTime joseLaundry = 830 := by
  sorry


end NUMINAMATH_CALUDE_total_laundry_time_l3992_399235


namespace NUMINAMATH_CALUDE_nuts_per_cookie_l3992_399200

theorem nuts_per_cookie (total_cookies : ℕ) (total_nuts : ℕ) 
  (h1 : total_cookies = 60)
  (h2 : total_nuts = 72)
  (h3 : (1 : ℚ) / 4 * total_cookies + (2 : ℚ) / 5 * total_cookies + 
        (total_cookies - (1 : ℚ) / 4 * total_cookies - (2 : ℚ) / 5 * total_cookies) = total_cookies) :
  total_nuts / ((1 : ℚ) / 4 * total_cookies + 
    (total_cookies - (1 : ℚ) / 4 * total_cookies - (2 : ℚ) / 5 * total_cookies)) = 2 := by
  sorry

#check nuts_per_cookie

end NUMINAMATH_CALUDE_nuts_per_cookie_l3992_399200


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3992_399217

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_of_angles : ℝ) : 
  sum_of_angles = 1080 → (n - 2) * 180 = sum_of_angles → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l3992_399217


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3992_399207

-- Problem 1
theorem problem_1 (x y z : ℝ) :
  2 * x^3 * y^2 * (-2 * x * y^2 * z)^2 = 8 * x^5 * y^6 * z^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) :
  (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3992_399207


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3992_399286

theorem complex_modulus_problem (θ : ℝ) (z : ℂ) : 
  z = (Complex.I * (Real.sin θ - Complex.I)) / Complex.I →
  Real.cos θ = 1/3 →
  Complex.abs z = Real.sqrt 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3992_399286


namespace NUMINAMATH_CALUDE_additional_earnings_is_correct_l3992_399239

/-- Represents the company's dividend policy and earnings information -/
structure CompanyData where
  expected_earnings : ℝ
  actual_earnings : ℝ
  base_dividend_ratio : ℝ
  extra_dividend_rate : ℝ
  shares_owned : ℕ
  total_dividend_paid : ℝ

/-- Calculates the additional earnings per share that triggers the extra dividend -/
def additional_earnings (data : CompanyData) : ℝ :=
  data.actual_earnings - data.expected_earnings

/-- Theorem stating that the additional earnings per share is $0.30 -/
theorem additional_earnings_is_correct (data : CompanyData) 
  (h1 : data.expected_earnings = 0.80)
  (h2 : data.actual_earnings = 1.10)
  (h3 : data.base_dividend_ratio = 0.5)
  (h4 : data.extra_dividend_rate = 0.04)
  (h5 : data.shares_owned = 400)
  (h6 : data.total_dividend_paid = 208) :
  additional_earnings data = 0.30 := by
  sorry

#eval additional_earnings {
  expected_earnings := 0.80,
  actual_earnings := 1.10,
  base_dividend_ratio := 0.5,
  extra_dividend_rate := 0.04,
  shares_owned := 400,
  total_dividend_paid := 208
}

end NUMINAMATH_CALUDE_additional_earnings_is_correct_l3992_399239


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3992_399280

theorem quadrilateral_diagonal_length 
  (area : ℝ) 
  (offset1 : ℝ) 
  (offset2 : ℝ) 
  (h1 : area = 300) 
  (h2 : offset1 = 9) 
  (h3 : offset2 = 6) : 
  area = (1/2) * (offset1 + offset2) * 40 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3992_399280


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_C_subset_B_implies_m_range_l3992_399230

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x ≤ 6}
def B : Set ℝ := {x | x^2 - 4*x < 0}
def C (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m - 1}

-- Theorem statements
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 4} := by sorry

theorem complement_union_A_B : 
  (Set.univ : Set ℝ) \ (A ∪ B) = {x : ℝ | x ≤ 0 ∨ x > 6} := by sorry

theorem C_subset_B_implies_m_range (m : ℝ) : 
  C m ⊆ B → m ≤ 5/2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_union_A_B_C_subset_B_implies_m_range_l3992_399230


namespace NUMINAMATH_CALUDE_average_bag_weight_l3992_399281

def bag_weights : List ℕ := [25, 30, 31, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 48]

theorem average_bag_weight :
  (bag_weights.sum : ℚ) / bag_weights.length = 71/2 := by sorry

end NUMINAMATH_CALUDE_average_bag_weight_l3992_399281


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3992_399250

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let set := List.replicate (n - 3) 1 ++ [1 + 1/n, 1 + 1/n, 1 - 1/n]
  (set.sum / n : ℚ) = 1 + 1/n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3992_399250


namespace NUMINAMATH_CALUDE_correct_equation_l3992_399233

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l3992_399233


namespace NUMINAMATH_CALUDE_min_value_binomial_distribution_l3992_399232

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 < p
  h2 : p < 1

/-- The expected value of a binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: The minimum value of 1/p + 1/q for a binomial distribution
    with E(X) = 4 and D(X) = q is 9/4 -/
theorem min_value_binomial_distribution 
  (X : BinomialDistribution) 
  (h_exp : expectedValue X = 4)
  (h_var : variance X = q)
  : (1 / X.p + 1 / q) ≥ 9/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_binomial_distribution_l3992_399232


namespace NUMINAMATH_CALUDE_A_investment_l3992_399227

-- Define the investments and profit shares
def investment_B : ℝ := 10000
def investment_C : ℝ := 12000
def profit_share_B : ℝ := 2500
def profit_difference_AC : ℝ := 999.9999999999998

-- Define the theorem
theorem A_investment (investment_A : ℝ) : 
  (investment_A / investment_B * profit_share_B) - 
  (investment_C / investment_B * profit_share_B) = profit_difference_AC → 
  investment_A = 16000 := by
sorry

end NUMINAMATH_CALUDE_A_investment_l3992_399227


namespace NUMINAMATH_CALUDE_moms_approach_is_sampling_survey_l3992_399292

/-- Represents a method of data collection. -/
inductive DataCollectionMethod
| Census
| SamplingSurvey

/-- Represents the action of tasting food. -/
structure TastingAction where
  dish : String
  portion : String

/-- Determines the data collection method based on the tasting action. -/
def determineMethod (action : TastingAction) : DataCollectionMethod :=
  if action.portion = "entire" then DataCollectionMethod.Census
  else DataCollectionMethod.SamplingSurvey

theorem moms_approach_is_sampling_survey :
  let momsTasting : TastingAction := { dish := "cooking dish", portion := "little bit" }
  determineMethod momsTasting = DataCollectionMethod.SamplingSurvey := by
  sorry


end NUMINAMATH_CALUDE_moms_approach_is_sampling_survey_l3992_399292


namespace NUMINAMATH_CALUDE_money_division_l3992_399276

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 2400 →
  r - q = 3000 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l3992_399276


namespace NUMINAMATH_CALUDE_last_digit_product_divisible_by_three_l3992_399282

theorem last_digit_product_divisible_by_three (n : ℕ) :
  let a := (2^n % 10)
  ∃ k : ℤ, a * (2^n - a) = 3 * k :=
sorry

end NUMINAMATH_CALUDE_last_digit_product_divisible_by_three_l3992_399282


namespace NUMINAMATH_CALUDE_max_value_trig_function_l3992_399208

theorem max_value_trig_function :
  ∃ M : ℝ, M = -1/2 ∧
  (∀ x : ℝ, 2 * Real.sin x ^ 2 + 2 * Real.cos x - 3 ≤ M) ∧
  ∀ ε > 0, ∃ x : ℝ, 2 * Real.sin x ^ 2 + 2 * Real.cos x - 3 > M - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_function_l3992_399208


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l3992_399225

theorem smallest_divisible_by_15_18_20 : 
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 18 ∣ n ∧ 20 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 18 ∣ m → 20 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_18_20_l3992_399225


namespace NUMINAMATH_CALUDE_smallest_a_is_390_l3992_399240

/-- A polynomial with three positive integer roots -/
structure PolynomialWithThreeIntegerRoots where
  a : ℕ
  b : ℕ
  root1 : ℕ+
  root2 : ℕ+
  root3 : ℕ+
  root_product : root1 * root2 * root3 = 2310
  root_sum : root1 + root2 + root3 = a

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a : ℕ := 390

/-- Theorem stating that 390 is the smallest possible value of a -/
theorem smallest_a_is_390 :
  ∀ p : PolynomialWithThreeIntegerRoots, p.a ≥ smallest_a :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_is_390_l3992_399240


namespace NUMINAMATH_CALUDE_rod_string_equilibrium_theorem_l3992_399252

/-- Represents the equilibrium conditions for a rod and string system --/
def rod_string_equilibrium (a b : ℝ) (θ : ℝ) : Prop :=
  (θ = 0 ∨ (θ.cos = (b^2 + 2*a^2) / (3*a*b) ∧ 1/2 * b < a ∧ a ≤ b)) ∧ a > 0 ∧ b > 0

/-- Theorem stating the equilibrium conditions for the rod and string system --/
theorem rod_string_equilibrium_theorem (a b : ℝ) (θ : ℝ) :
  a > 0 → b > 0 → rod_string_equilibrium a b θ ↔
    (θ = 0 ∨ (θ.cos = (b^2 + 2*a^2) / (3*a*b) ∧ 1/2 * b < a ∧ a ≤ b)) :=
by sorry

end NUMINAMATH_CALUDE_rod_string_equilibrium_theorem_l3992_399252


namespace NUMINAMATH_CALUDE_prudence_sleep_hours_l3992_399219

/-- The number of hours Prudence sleeps per night from Sunday to Thursday -/
def sleepHoursSundayToThursday : ℝ := 6

/-- The number of hours Prudence sleeps on Friday and Saturday nights -/
def sleepHoursFridaySaturday : ℝ := 9

/-- The number of hours Prudence naps on Saturday and Sunday -/
def napHours : ℝ := 1

/-- The total number of hours Prudence sleeps in 4 weeks -/
def totalSleepHours : ℝ := 200

/-- The number of weeks -/
def numWeeks : ℝ := 4

/-- The number of nights from Sunday to Thursday -/
def nightsSundayToThursday : ℝ := 5

/-- The number of nights for Friday and Saturday -/
def nightsFridaySaturday : ℝ := 2

/-- The number of nap days (Saturday and Sunday) -/
def napDays : ℝ := 2

theorem prudence_sleep_hours :
  sleepHoursSundayToThursday * nightsSundayToThursday +
  sleepHoursFridaySaturday * nightsFridaySaturday +
  napHours * napDays * numWeeks = totalSleepHours :=
by sorry

end NUMINAMATH_CALUDE_prudence_sleep_hours_l3992_399219
