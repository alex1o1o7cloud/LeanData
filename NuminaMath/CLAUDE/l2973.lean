import Mathlib

namespace NUMINAMATH_CALUDE_car_profit_percent_l2973_297337

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℝ) 
  (repair_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : repair_cost = 12000) 
  (h3 : selling_price = 64900) : 
  ∃ (profit_percent : ℝ), abs (profit_percent - 20.19) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_car_profit_percent_l2973_297337


namespace NUMINAMATH_CALUDE_certain_event_good_product_l2973_297385

/-- Represents the total number of products --/
def total_products : ℕ := 12

/-- Represents the number of good products --/
def good_products : ℕ := 10

/-- Represents the number of defective products --/
def defective_products : ℕ := 2

/-- Represents the number of products selected --/
def selected_products : ℕ := 3

/-- Represents a selection of products --/
def Selection := Fin selected_products → Fin total_products

/-- Predicate to check if a selection contains at least one good product --/
def contains_good_product (s : Selection) : Prop :=
  ∃ i, s i < good_products

/-- The main theorem stating that any selection contains at least one good product --/
theorem certain_event_good_product :
  ∀ s : Selection, contains_good_product s :=
sorry

end NUMINAMATH_CALUDE_certain_event_good_product_l2973_297385


namespace NUMINAMATH_CALUDE_unique_three_digit_number_exists_l2973_297341

/-- Represents a 3-digit number abc as 100a + 10b + c -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the number acb obtained by swapping the last two digits of abc -/
def swap_last_two_digits (a b c : ℕ) : ℕ := 100 * a + 10 * c + b

theorem unique_three_digit_number_exists :
  ∃! (a b c : ℕ),
    (100 ≤ three_digit_number a b c) ∧
    (three_digit_number a b c ≤ 999) ∧
    (1730 ≤ three_digit_number a b c + swap_last_two_digits a b c) ∧
    (three_digit_number a b c + swap_last_two_digits a b c ≤ 1739) ∧
    (three_digit_number a b c = 832) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_exists_l2973_297341


namespace NUMINAMATH_CALUDE_combined_work_time_l2973_297317

-- Define the work rates for A and B
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Theorem statement
theorem combined_work_time :
  (1 : ℚ) / combined_work_rate = 3 := by sorry

end NUMINAMATH_CALUDE_combined_work_time_l2973_297317


namespace NUMINAMATH_CALUDE_only_proposition4_is_correct_l2973_297314

-- Define the propositions
def proposition1 : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) - a n = 0) → (∃ r, ∀ n, a (n + 1) = r * a n)
def proposition2 : Prop := ∀ (a : ℕ → ℝ), (∀ n, a (n + 1) = (1/2) * a n) → (∀ n, a (n + 1) < a n)
def proposition3 : Prop := ∀ a b c : ℝ, (b^2 = a * c) ↔ (∃ r, b = a * r ∧ c = b * r)
def proposition4 : Prop := ∀ a b c : ℝ, (2 * b = a + c) ↔ (∃ d, b = a + d ∧ c = b + d)

-- Theorem statement
theorem only_proposition4_is_correct :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
sorry

end NUMINAMATH_CALUDE_only_proposition4_is_correct_l2973_297314


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l2973_297370

/-- A polynomial with integer coefficients where each coefficient is between 0 and 4 inclusive -/
def IntPolynomial (m : ℕ) := { b : Fin (m + 1) → ℤ // ∀ i, 0 ≤ b i ∧ b i < 5 }

/-- Evaluation of an IntPolynomial at a given value -/
def evalPoly {m : ℕ} (P : IntPolynomial m) (x : ℝ) : ℝ :=
  (Finset.range (m + 1)).sum (fun i => (P.val i : ℝ) * x ^ i)

theorem polynomial_evaluation (m : ℕ) (P : IntPolynomial m) :
  evalPoly P (Real.sqrt 5) = 23 + 19 * Real.sqrt 5 →
  evalPoly P 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l2973_297370


namespace NUMINAMATH_CALUDE_f_5_equals_18556_l2973_297327

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ :=
  horner_polynomial [5, 4, 3, 2, 1, 1] x

theorem f_5_equals_18556 : f 5 = 18556 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_18556_l2973_297327


namespace NUMINAMATH_CALUDE_multiplication_fraction_simplification_l2973_297313

theorem multiplication_fraction_simplification :
  8 * (2 / 17) * 34 * (1 / 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_fraction_simplification_l2973_297313


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l2973_297312

/-- Represents a rectangular grid with alternating painted columns and rows -/
structure PaintedGrid where
  rows : Nat
  cols : Nat
  unpaintedCells : Nat

/-- Checks if the grid dimensions are valid (odd number of rows and columns) -/
def PaintedGrid.isValid (grid : PaintedGrid) : Prop :=
  grid.rows % 2 = 1 ∧ grid.cols % 2 = 1

/-- Calculates the number of painted cells in the grid -/
def PaintedGrid.paintedCells (grid : PaintedGrid) : Nat :=
  grid.rows * grid.cols - grid.unpaintedCells

/-- Theorem: If a valid painted grid has 74 unpainted cells, 
    then the number of painted cells is either 301 or 373 -/
theorem painted_cells_theorem (grid : PaintedGrid) :
  grid.isValid ∧ grid.unpaintedCells = 74 →
  grid.paintedCells = 301 ∨ grid.paintedCells = 373 := by
  sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l2973_297312


namespace NUMINAMATH_CALUDE_wire_length_ratio_l2973_297361

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the total wire length needed for a cuboid frame -/
def wireLength (d : CuboidDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

theorem wire_length_ratio : 
  let bonnie := CuboidDimensions.mk 8 10 10
  let roark := CuboidDimensions.mk 1 2 2
  let bonnieVolume := cuboidVolume bonnie
  let roarkVolume := cuboidVolume roark
  let numRoarkCuboids := bonnieVolume / roarkVolume
  let bonnieWire := wireLength bonnie
  let roarkTotalWire := numRoarkCuboids * wireLength roark
  bonnieWire / roarkTotalWire = 9 / 250 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l2973_297361


namespace NUMINAMATH_CALUDE_exam_mean_score_l2973_297369

/-- Given an exam score distribution where 58 is 2 standard deviations below the mean
    and 98 is 3 standard deviations above the mean, the mean score is 74. -/
theorem exam_mean_score (μ σ : ℝ) 
  (h1 : 58 = μ - 2 * σ) 
  (h2 : 98 = μ + 3 * σ) : 
  μ = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2973_297369


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l2973_297347

theorem unique_solution_cube_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l2973_297347


namespace NUMINAMATH_CALUDE_product_congruence_l2973_297381

def product : ℕ → ℕ
| 0 => 2
| 1 => 13
| (n+2) => if n % 2 = 0 then (10 + n + 2) * 2 else (10 + n + 2) * 3

def big_product : ℕ := (product 0) * (product 1) * (product 2) * (product 3) * 
                       (product 4) * (product 5) * (product 6) * (product 7) * 
                       (product 8) * (product 9) * (product 10) * (product 11) * 
                       (product 12) * (product 13) * (product 14) * (product 15) * 
                       (product 16) * (product 17)

theorem product_congruence : big_product ≡ 1 [ZMOD 5] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l2973_297381


namespace NUMINAMATH_CALUDE_probability_second_math_given_first_math_l2973_297340

def total_questions : ℕ := 5
def math_questions : ℕ := 3
def physics_questions : ℕ := 2

theorem probability_second_math_given_first_math :
  let P : ℝ := (math_questions * (math_questions - 1)) / (total_questions * (total_questions - 1))
  let Q : ℝ := math_questions / total_questions
  P / Q = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_second_math_given_first_math_l2973_297340


namespace NUMINAMATH_CALUDE_max_xy_constraint_min_x_plus_y_constraint_l2973_297388

-- Part 1
theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y + x*y = 12) :
  x*y ≤ 4 ∧ (x*y = 4 → x = 4 ∧ y = 1) :=
sorry

-- Part 2
theorem min_x_plus_y_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y = x*y) :
  x + y ≥ 9 ∧ (x + y = 9 → x = 6 ∧ y = 3) :=
sorry

end NUMINAMATH_CALUDE_max_xy_constraint_min_x_plus_y_constraint_l2973_297388


namespace NUMINAMATH_CALUDE_fruit_cost_difference_l2973_297300

/-- Represents the cost and quantity of a fruit carton -/
structure FruitCarton where
  cost : ℚ  -- Cost in dollars
  quantity : ℚ  -- Quantity in ounces
  inv_mk : cost > 0 ∧ quantity > 0

/-- Calculates the number of cartons needed for a given amount of fruit -/
def cartonsNeeded (fruit : FruitCarton) (amount : ℚ) : ℚ :=
  amount / fruit.quantity

/-- Calculates the total cost for a given number of cartons -/
def totalCost (fruit : FruitCarton) (cartons : ℚ) : ℚ :=
  fruit.cost * cartons

/-- The main theorem to prove -/
theorem fruit_cost_difference 
  (blueberries : FruitCarton)
  (raspberries : FruitCarton)
  (batches : ℕ)
  (fruitPerBatch : ℚ)
  (h1 : blueberries.cost = 5)
  (h2 : blueberries.quantity = 6)
  (h3 : raspberries.cost = 3)
  (h4 : raspberries.quantity = 8)
  (h5 : batches = 4)
  (h6 : fruitPerBatch = 12) :
  totalCost blueberries (cartonsNeeded blueberries (batches * fruitPerBatch)) -
  totalCost raspberries (cartonsNeeded raspberries (batches * fruitPerBatch)) = 22 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_difference_l2973_297300


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l2973_297387

theorem cost_increase_percentage (initial_cost selling_price new_cost : ℝ) : 
  initial_cost > 0 →
  selling_price = 2.5 * initial_cost →
  new_cost > initial_cost →
  (selling_price - new_cost) / selling_price = 0.552 →
  (new_cost - initial_cost) / initial_cost = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l2973_297387


namespace NUMINAMATH_CALUDE_power_sum_theorem_l2973_297316

theorem power_sum_theorem (a : ℝ) (m : ℕ) (h : a^m = 2) : a^(2*m) + a^(3*m) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l2973_297316


namespace NUMINAMATH_CALUDE_nba_scheduling_impossibility_l2973_297315

theorem nba_scheduling_impossibility :
  ∀ (k : ℕ) (x y z : ℕ),
    k ≤ 30 ∧
    x + y + z = 1230 ∧
    82 * k = 2 * x + z →
    z ≠ (x + y + z) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_nba_scheduling_impossibility_l2973_297315


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l2973_297328

theorem binomial_floor_divisibility (p n : ℕ) (h_prime : Nat.Prime p) (h_n_ge_p : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) :=
sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l2973_297328


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l2973_297304

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem smallest_integer_with_divisibility_condition :
  ∃ (n : ℕ) (i j : ℕ),
    n > 0 ∧
    i < j ∧
    j - i = 1 ∧
    j ≤ 30 ∧
    (∀ k : ℕ, k ≤ 30 → k ≠ i → k ≠ j → is_divisible n k) ∧
    ¬(is_divisible n i) ∧
    ¬(is_divisible n j) ∧
    (∀ m : ℕ, m > 0 →
      (∃ (x y : ℕ), x < y ∧ y - x = 1 ∧ y ≤ 30 ∧
        (∀ k : ℕ, k ≤ 30 → k ≠ x → k ≠ y → is_divisible m k) ∧
        ¬(is_divisible m x) ∧
        ¬(is_divisible m y)) →
      m ≥ n) ∧
    n = 2230928700 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l2973_297304


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l2973_297336

theorem salary_reduction_percentage (S : ℝ) (R : ℝ) :
  S > 0 →
  (S - (R / 100 * S)) * (1 + 1 / 3) = S →
  R = 25 := by
sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l2973_297336


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2973_297329

theorem rationalize_denominator :
  ∃ (A B C : ℤ), (1 + Real.sqrt 3) / (1 - Real.sqrt 3) = A + B * Real.sqrt C ∧ A = -2 ∧ B = -1 ∧ C = 3 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2973_297329


namespace NUMINAMATH_CALUDE_common_roots_product_l2973_297332

theorem common_roots_product (C : ℝ) : 
  ∃ (p q r t : ℝ), 
    (p^3 + 2*p^2 + 15 = 0) ∧ 
    (q^3 + 2*q^2 + 15 = 0) ∧ 
    (r^3 + 2*r^2 + 15 = 0) ∧
    (p^3 + C*p + 30 = 0) ∧ 
    (q^3 + C*q + 30 = 0) ∧ 
    (t^3 + C*t + 30 = 0) ∧
    (p ≠ q) ∧ (p ≠ r) ∧ (q ≠ r) ∧ 
    (p ≠ t) ∧ (q ≠ t) →
    p * q = -5 * Real.rpow 2 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l2973_297332


namespace NUMINAMATH_CALUDE_cost_per_bag_first_is_24_l2973_297334

/-- The cost per bag of zongzi in the first batch -/
def cost_per_bag_first : ℝ := 24

/-- The total cost of the first batch of zongzi -/
def total_cost_first : ℝ := 3000

/-- The total cost of the second batch of zongzi -/
def total_cost_second : ℝ := 7500

/-- The number of bags in the second batch is three times the number in the first batch -/
def batch_ratio : ℝ := 3

/-- The cost difference per bag between the first and second batch -/
def cost_difference : ℝ := 4

theorem cost_per_bag_first_is_24 :
  cost_per_bag_first = 24 ∧
  total_cost_first = 3000 ∧
  total_cost_second = 7500 ∧
  batch_ratio = 3 ∧
  cost_difference = 4 →
  cost_per_bag_first = 24 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_bag_first_is_24_l2973_297334


namespace NUMINAMATH_CALUDE_fraction_equality_l2973_297368

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b * (a + b) = 1) :
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2973_297368


namespace NUMINAMATH_CALUDE_alice_pairs_l2973_297357

theorem alice_pairs (total_students : ℕ) (h1 : total_students = 12) : 
  (total_students - 2) = 11 := by
  sorry

#check alice_pairs

end NUMINAMATH_CALUDE_alice_pairs_l2973_297357


namespace NUMINAMATH_CALUDE_regular_septagon_interior_angle_measure_l2973_297331

/-- The number of sides in a septagon -/
def n : ℕ := 7

/-- A regular septagon is a polygon with 7 sides and all interior angles equal -/
structure RegularSeptagon where
  sides : Fin n → ℝ
  angles : Fin n → ℝ
  all_sides_equal : ∀ i j : Fin n, sides i = sides j
  all_angles_equal : ∀ i j : Fin n, angles i = angles j

/-- Theorem: The measure of each interior angle in a regular septagon is 900/7 degrees -/
theorem regular_septagon_interior_angle_measure (s : RegularSeptagon) :
  ∀ i : Fin n, s.angles i = 900 / 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_septagon_interior_angle_measure_l2973_297331


namespace NUMINAMATH_CALUDE_intersection_slope_l2973_297372

/-- Given two circles in the xy-plane, this theorem states that the slope of the line
    passing through their intersection points is 1/7. -/
theorem intersection_slope (x y : ℝ) :
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) →
  (x^2 + y^2 - 8*x + 18*y + 40 = 0) →
  (∃ (m : ℝ), m = 1/7 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) →
    (x₁^2 + y₁^2 - 8*x₁ + 18*y₁ + 40 = 0) →
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) →
    (x₂^2 + y₂^2 - 8*x₂ + 18*y₂ + 40 = 0) →
    x₁ ≠ x₂ →
    m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry


end NUMINAMATH_CALUDE_intersection_slope_l2973_297372


namespace NUMINAMATH_CALUDE_base_seven_sum_l2973_297359

/-- Given A, B, C are distinct digits in base 7 and ABC_7 + BCA_7 + CAB_7 = AAA1_7,
    prove that B + C = 6 (in base 7) if A = 1, or B + C = 12 (in base 7) if A = 2. -/
theorem base_seven_sum (A B C : ℕ) : 
  A < 7 → B < 7 → C < 7 → 
  A ≠ B → B ≠ C → A ≠ C →
  (7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B) = 
    7^3 * A + 7^2 * A + 7 * A + 1 →
  (A = 1 ∧ B + C = 6) ∨ (A = 2 ∧ B + C = 12) :=
by sorry

end NUMINAMATH_CALUDE_base_seven_sum_l2973_297359


namespace NUMINAMATH_CALUDE_existence_of_abcd_l2973_297393

theorem existence_of_abcd (n : ℕ) (h : n > 1) : ∃ (a b c d : ℕ),
  a = 3*n - 1 ∧
  b = n + 1 ∧
  c = 3*n + 1 ∧
  d = n - 1 ∧
  a + b = 4*n ∧
  c + d = 4*n ∧
  a * b - c * d = 4*n :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_abcd_l2973_297393


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2973_297311

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2973_297311


namespace NUMINAMATH_CALUDE_polygon_with_two_diagonals_has_five_sides_l2973_297378

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- The number of diagonals from any vertex in a polygon. -/
def diagonals_from_vertex (p : Polygon) : ℕ := p.sides - 3

/-- Theorem: A polygon with 2 diagonals from any vertex has 5 sides. -/
theorem polygon_with_two_diagonals_has_five_sides (p : Polygon) 
  (h : diagonals_from_vertex p = 2) : p.sides = 5 := by
  sorry

#check polygon_with_two_diagonals_has_five_sides

end NUMINAMATH_CALUDE_polygon_with_two_diagonals_has_five_sides_l2973_297378


namespace NUMINAMATH_CALUDE_prob_committee_with_both_genders_l2973_297375

-- Define the total number of members
def total_members : ℕ := 40

-- Define the number of boys
def num_boys : ℕ := 18

-- Define the number of girls
def num_girls : ℕ := 22

-- Define the committee size
def committee_size : ℕ := 6

-- Define the probability function
noncomputable def prob_at_least_one_boy_one_girl : ℚ :=
  1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) /
      (Nat.choose total_members committee_size : ℚ)

-- Theorem statement
theorem prob_committee_with_both_genders :
  prob_at_least_one_boy_one_girl = 2913683 / 3838380 :=
sorry

end NUMINAMATH_CALUDE_prob_committee_with_both_genders_l2973_297375


namespace NUMINAMATH_CALUDE_smaller_cube_side_length_l2973_297323

theorem smaller_cube_side_length (R : ℝ) (x : ℝ) : 
  R = Real.sqrt 3 →
  (1 + x)^2 + (x * Real.sqrt 2 / 2)^2 = R^2 →
  x = 2/3 :=
sorry

end NUMINAMATH_CALUDE_smaller_cube_side_length_l2973_297323


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2973_297391

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles at the same height in the first quadrant -/
def TangentLine (c1 c2 : Circle) :=
  ∃ (y : ℝ), y > 0 ∧
    (y = c1.radius ∨ y = c2.radius) ∧
    (c1.center.1 + c1.radius < c2.center.1 - c2.radius)

theorem tangent_line_y_intercept
  (c1 : Circle)
  (c2 : Circle)
  (h1 : c1 = ⟨(3, 0), 3⟩)
  (h2 : c2 = ⟨(8, 0), 2⟩)
  (h3 : TangentLine c1 c2) :
  ∃ (line : ℝ → ℝ), line 0 = 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2973_297391


namespace NUMINAMATH_CALUDE_rectangle_area_l2973_297310

theorem rectangle_area (length width diagonal : ℝ) : 
  length = 16 →
  length / diagonal = 4 / 5 →
  length ^ 2 + width ^ 2 = diagonal ^ 2 →
  length * width = 192 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2973_297310


namespace NUMINAMATH_CALUDE_max_value_abc_l2973_297392

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  a^2 * b^3 * c^2 ≤ 81/262144 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l2973_297392


namespace NUMINAMATH_CALUDE_min_value_sum_and_reciprocal_l2973_297302

theorem min_value_sum_and_reciprocal (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (((a^2 + b^2 : ℚ) / (a * b)) + ((a * b : ℚ) / (a^2 + b^2))) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' ≠ 0 ∧ b' ≠ 0 ∧ (((a'^2 + b'^2 : ℚ) / (a' * b')) + ((a' * b' : ℚ) / (a'^2 + b'^2))) = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_and_reciprocal_l2973_297302


namespace NUMINAMATH_CALUDE_arc_length_quarter_circle_l2973_297319

/-- Given a circle with circumference 120 feet and a central angle of 90°, 
    the length of the corresponding arc is 30 feet. -/
theorem arc_length_quarter_circle (D : Real) (EF : Real) (EOF : Real) : 
  D = 120 → EOF = 90 → EF = 30 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_quarter_circle_l2973_297319


namespace NUMINAMATH_CALUDE_specific_path_count_l2973_297376

/-- The number of paths on a grid with given dimensions and constraints -/
def numPaths (width height diagonalSteps : ℕ) : ℕ :=
  Nat.choose (width + height - diagonalSteps) diagonalSteps *
  Nat.choose (width + height - 2 * diagonalSteps) height

/-- Theorem stating the number of paths for the specific problem -/
theorem specific_path_count :
  numPaths 7 6 2 = 6930 := by
  sorry

end NUMINAMATH_CALUDE_specific_path_count_l2973_297376


namespace NUMINAMATH_CALUDE_binomial_expected_value_theorem_l2973_297389

/-- A random variable following a Binomial distribution -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  prob : ℝ
  property : prob = p

/-- The expected value of a Binomial distribution -/
def expectedValue (ξ : BinomialDistribution n p) : ℝ := n * p

theorem binomial_expected_value_theorem (ξ : BinomialDistribution 18 p) 
  (h : expectedValue ξ = 9) : p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expected_value_theorem_l2973_297389


namespace NUMINAMATH_CALUDE_horner_method_v2_l2973_297309

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

def horner_v2 (a b c d e f x : ℝ) : ℝ :=
  ((a * x + b) * x + c) * x + d

theorem horner_method_v2 :
  horner_v2 2 (-5) (-4) 3 (-6) 7 5 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_horner_method_v2_l2973_297309


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l2973_297342

-- Define the quadratic function
def f (a b x : ℝ) : ℝ := x^2 + a*x + (b - 2)

-- State the theorem
theorem range_of_a_minus_b (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ -1 < x₂ ∧ x₂ < 0 ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) →
  ∀ y : ℝ, y > -1 → ∃ a' b' : ℝ, a' - b' = y ∧
    ∃ x₁ x₂ : ℝ, x₁ < -1 ∧ -1 < x₂ ∧ x₂ < 0 ∧ f a' b' x₁ = 0 ∧ f a' b' x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l2973_297342


namespace NUMINAMATH_CALUDE_probability_drawing_white_ball_l2973_297351

theorem probability_drawing_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ)
  (h1 : total_balls = 15)
  (h2 : red_balls = 9)
  (h3 : white_balls = 6)
  (h4 : total_balls = red_balls + white_balls) :
  (white_balls : ℚ) / (total_balls - 1 : ℚ) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_drawing_white_ball_l2973_297351


namespace NUMINAMATH_CALUDE_trishas_total_distance_is_correct_l2973_297390

/-- The total distance Trisha walked during her vacation in New York City -/
def trishas_total_distance : ℝ :=
  let hotel_to_postcard := 0.11
  let postcard_to_hotel := 0.11
  let hotel_to_tshirt := 1.52
  let tshirt_to_hat := 0.45
  let hat_to_purse := 0.87
  let purse_to_hotel := 2.32
  hotel_to_postcard + postcard_to_hotel + hotel_to_tshirt + tshirt_to_hat + hat_to_purse + purse_to_hotel

/-- Theorem stating that the total distance Trisha walked is 5.38 miles -/
theorem trishas_total_distance_is_correct : trishas_total_distance = 5.38 := by
  sorry

end NUMINAMATH_CALUDE_trishas_total_distance_is_correct_l2973_297390


namespace NUMINAMATH_CALUDE_at_least_one_less_than_one_l2973_297306

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  a < 1 ∨ b < 1 ∨ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_one_l2973_297306


namespace NUMINAMATH_CALUDE_min_value_theorem_l2973_297356

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 1 → x + y = 2 → 4/x + 1/(y-1) ≥ 4/a + 1/(b-1)) ∧
  4/a + 1/(b-1) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2973_297356


namespace NUMINAMATH_CALUDE_remaining_pokemon_cards_l2973_297394

/-- Theorem: Calculating remaining Pokemon cards after a sale --/
theorem remaining_pokemon_cards 
  (initial_cards : ℕ) 
  (sold_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : sold_cards = 224) :
  initial_cards - sold_cards = 452 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_pokemon_cards_l2973_297394


namespace NUMINAMATH_CALUDE_third_group_men_count_l2973_297374

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the third group -/
def x : ℕ := sorry

theorem third_group_men_count : x = 5 := by
  have h1 : 3 * man_rate + 8 * woman_rate = 6 * man_rate + 2 * woman_rate := by sorry
  have h2 : x * man_rate + 2 * woman_rate = (6/7) * (3 * man_rate + 8 * woman_rate) := by sorry
  sorry

end NUMINAMATH_CALUDE_third_group_men_count_l2973_297374


namespace NUMINAMATH_CALUDE_vermont_clicked_68_ads_l2973_297364

/-- The number of ads Vermont clicked on -/
def ads_clicked (first_page : ℕ) : ℕ :=
  let second_page := 2 * first_page
  let third_page := second_page + 24
  let fourth_page := (3 * second_page) / 4
  let total_ads := first_page + second_page + third_page + fourth_page
  (2 * total_ads) / 3

/-- Theorem stating that Vermont clicked on 68 ads -/
theorem vermont_clicked_68_ads : ads_clicked 12 = 68 := by
  sorry

end NUMINAMATH_CALUDE_vermont_clicked_68_ads_l2973_297364


namespace NUMINAMATH_CALUDE_total_books_stu_and_albert_l2973_297335

/-- Given that Stu has 9 books and Albert has 4 times as many books as Stu,
    prove that the total number of books Stu and Albert have is 45. -/
theorem total_books_stu_and_albert :
  let stu_books : ℕ := 9
  let albert_books : ℕ := 4 * stu_books
  stu_books + albert_books = 45 := by
sorry

end NUMINAMATH_CALUDE_total_books_stu_and_albert_l2973_297335


namespace NUMINAMATH_CALUDE_triangle_area_l2973_297355

/-- Given a triangle with sides AC, BC, and BD, prove that its area is 14 -/
theorem triangle_area (AC BC BD : ℝ) (h1 : AC = 4) (h2 : BC = 3) (h3 : BD = 10) :
  (1 / 2 : ℝ) * (BD - BC) * AC = 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2973_297355


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2973_297366

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 3*Complex.I) / (3 - Complex.I) → z.im = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2973_297366


namespace NUMINAMATH_CALUDE_cucumber_problem_l2973_297395

theorem cucumber_problem (boxes : ℕ) (cucumbers_per_box : ℕ) (rotten : ℕ) (bags : ℕ) :
  boxes = 7 →
  cucumbers_per_box = 16 →
  rotten = 13 →
  bags = 8 →
  (boxes * cucumbers_per_box - rotten) % bags = 3 := by
sorry

end NUMINAMATH_CALUDE_cucumber_problem_l2973_297395


namespace NUMINAMATH_CALUDE_original_number_proof_l2973_297305

theorem original_number_proof : ∃ N : ℕ, 
  (∀ m : ℕ, m < N → ¬(m - 6 ≡ 3 [MOD 5] ∧ m - 6 ≡ 3 [MOD 11] ∧ m - 6 ≡ 3 [MOD 13])) ∧
  (N - 6 ≡ 3 [MOD 5] ∧ N - 6 ≡ 3 [MOD 11] ∧ N - 6 ≡ 3 [MOD 13]) ∧
  N = 724 :=
by sorry

#check original_number_proof

end NUMINAMATH_CALUDE_original_number_proof_l2973_297305


namespace NUMINAMATH_CALUDE_N_subset_M_l2973_297386

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 ≥ x}
def N : Set ℝ := {x : ℝ | Real.log (x + 1) / Real.log (1/2) > 0}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_N_subset_M_l2973_297386


namespace NUMINAMATH_CALUDE_successive_integers_product_l2973_297362

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 9506 → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l2973_297362


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2973_297360

theorem quadratic_roots_difference (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + 42*x + 384
  let roots := {x : ℝ | f x = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ |r₁ - r₂| = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2973_297360


namespace NUMINAMATH_CALUDE_points_on_parabola_l2973_297318

-- Define the function y = x^2
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem points_on_parabola :
  ∀ t : ℝ, ∃ p₁ p₂ : ℝ × ℝ,
    p₁ = (1, f 1) ∧
    p₂ = (t, f t) ∧
    (p₁.2 = f p₁.1) ∧
    (p₂.2 = f p₂.1) :=
by
  sorry


end NUMINAMATH_CALUDE_points_on_parabola_l2973_297318


namespace NUMINAMATH_CALUDE_waiting_room_problem_l2973_297383

theorem waiting_room_problem (initial_waiting : ℕ) (interview_room : ℕ) : 
  initial_waiting = 22 → interview_room = 5 → 
  ∃ (additional : ℕ), initial_waiting + additional = 5 * interview_room ∧ additional = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_waiting_room_problem_l2973_297383


namespace NUMINAMATH_CALUDE_secret_spread_reaches_target_target_day_minimal_l2973_297350

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when the secret reaches 3280 students -/
def target_day : ℕ := 7

theorem secret_spread_reaches_target :
  secret_spread target_day = 3280 :=
sorry

theorem target_day_minimal :
  ∀ k < target_day, secret_spread k < 3280 :=
sorry

end NUMINAMATH_CALUDE_secret_spread_reaches_target_target_day_minimal_l2973_297350


namespace NUMINAMATH_CALUDE_part1_range_of_m_part2_range_of_m_l2973_297308

-- Define the function f
def f (a m x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x + m

-- Part 1
theorem part1_range_of_m :
  ∀ m : ℝ, (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f 1 m x = 0 ∧ f 1 m y = 0 ∧ f 1 m z = 0) →
  -1 < m ∧ m < 5/27 :=
sorry

-- Part 2
theorem part2_range_of_m :
  ∀ m : ℝ, (∀ a : ℝ, 3 ≤ a ∧ a ≤ 6 →
    ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a m x ≤ 1) →
  m ≤ -87 :=
sorry

end NUMINAMATH_CALUDE_part1_range_of_m_part2_range_of_m_l2973_297308


namespace NUMINAMATH_CALUDE_jellybeans_theorem_l2973_297322

def jellybeans_problem (initial_jellybeans : ℕ) (normal_class_size : ℕ) (sick_children : ℕ) (jellybeans_per_child : ℕ) : Prop :=
  let attending_children := normal_class_size - sick_children
  let eaten_jellybeans := attending_children * jellybeans_per_child
  let remaining_jellybeans := initial_jellybeans - eaten_jellybeans
  remaining_jellybeans = 34

theorem jellybeans_theorem :
  jellybeans_problem 100 24 2 3 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_theorem_l2973_297322


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2973_297352

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos (π - α) = 1/3) :
  Real.sin (2 * α) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2973_297352


namespace NUMINAMATH_CALUDE_coefficient_of_x4_in_expansion_l2973_297398

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^4
def coefficient_x4 (a b : ℝ) (n : ℕ) : ℝ :=
  binomial n 2 * a^(n-2) * b^2

-- Theorem statement
theorem coefficient_of_x4_in_expansion :
  coefficient_x4 2 1 5 = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x4_in_expansion_l2973_297398


namespace NUMINAMATH_CALUDE_mia_bought_three_more_notebooks_l2973_297344

/-- Represents the price of a single notebook in cents -/
def notebook_price : ℕ := 50

/-- Represents the number of notebooks Colin bought -/
def colin_notebooks : ℕ := 5

/-- Represents the number of notebooks Mia bought -/
def mia_notebooks : ℕ := 8

/-- Represents Colin's total payment in cents -/
def colin_payment : ℕ := 250

/-- Represents Mia's total payment in cents -/
def mia_payment : ℕ := 400

theorem mia_bought_three_more_notebooks :
  mia_notebooks = colin_notebooks + 3 ∧
  notebook_price > 1 ∧
  notebook_price * colin_notebooks = colin_payment ∧
  notebook_price * mia_notebooks = mia_payment :=
by sorry

end NUMINAMATH_CALUDE_mia_bought_three_more_notebooks_l2973_297344


namespace NUMINAMATH_CALUDE_clothes_expenditure_fraction_l2973_297371

theorem clothes_expenditure_fraction (initial_amount : ℝ) (remaining_amount : ℝ) (F : ℝ) : 
  initial_amount = 499.9999999999999 →
  remaining_amount = 200 →
  remaining_amount = (3/5) * (1 - F) * initial_amount →
  F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_clothes_expenditure_fraction_l2973_297371


namespace NUMINAMATH_CALUDE_seven_eighths_of_64_l2973_297339

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_64_l2973_297339


namespace NUMINAMATH_CALUDE_system_solution_l2973_297345

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * y = |2 * x + 3| - |2 * x - 3|
def equation2 (x y : ℝ) : Prop := 4 * x = |y + 2| - |y - 2|

-- Define the solution set
def solutionSet (x y : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1 ∧ y = 2 * x

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y ↔ solutionSet x y :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2973_297345


namespace NUMINAMATH_CALUDE_prob_two_cards_sum_fifteen_l2973_297396

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face
| Ace

/-- A standard 52-card deck --/
def Deck : Finset Card := sorry

/-- The set of number cards (2 through 10) in the deck --/
def NumberCards : Finset Card := sorry

/-- The probability of drawing two specific cards from the deck --/
def drawTwoCardsProbability (card1 card2 : Card) : ℚ := sorry

/-- The sum of two cards --/
def cardSum (card1 card2 : Card) : ℕ := sorry

/-- The probability of drawing two number cards that sum to 15 --/
def probSumFifteen : ℚ := sorry

theorem prob_two_cards_sum_fifteen :
  probSumFifteen = 16 / 884 := by sorry

end NUMINAMATH_CALUDE_prob_two_cards_sum_fifteen_l2973_297396


namespace NUMINAMATH_CALUDE_power_product_equals_one_third_l2973_297330

theorem power_product_equals_one_third :
  (-3 : ℚ)^2022 * (1/3 : ℚ)^2023 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_power_product_equals_one_third_l2973_297330


namespace NUMINAMATH_CALUDE_lines_perpendicular_imply_parallel_l2973_297307

-- Define a type for lines in 3D space
structure Line3D where
  -- You might want to add more properties here, but for this problem, we only need the line itself
  line : Type

-- Define perpendicularity and parallelism for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

def parallel (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem lines_perpendicular_imply_parallel (a b c d : Line3D) 
  (h1 : perpendicular a c)
  (h2 : perpendicular b c)
  (h3 : perpendicular a d)
  (h4 : perpendicular b d) :
  parallel a b ∨ parallel c d := by
  sorry

end NUMINAMATH_CALUDE_lines_perpendicular_imply_parallel_l2973_297307


namespace NUMINAMATH_CALUDE_system_solution_l2973_297380

theorem system_solution : ∃ (x y : ℝ), 
  (7 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 32) ∧ (x = -3) ∧ (y = 4) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2973_297380


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l2973_297326

theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l2973_297326


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_alt_l2973_297321

/-- Given an equilateral triangle and an isosceles triangle sharing a side,
    prove that the perimeter of the equilateral triangle is 60 -/
theorem equilateral_triangle_perimeter
  (s : ℝ)  -- side length of the equilateral triangle
  (h1 : s > 0)  -- side length is positive
  (h2 : 2 * s + 5 = 45)  -- condition from isosceles triangle
  : 3 * s = 60 := by
  sorry

/-- Alternative formulation using more basic definitions -/
theorem equilateral_triangle_perimeter_alt
  (s : ℝ)  -- side length of the equilateral triangle
  (P_isosceles : ℝ)  -- perimeter of the isosceles triangle
  (b : ℝ)  -- base of the isosceles triangle
  (h1 : s > 0)  -- side length is positive
  (h2 : P_isosceles = 45)  -- given perimeter of isosceles triangle
  (h3 : b = 5)  -- given base of isosceles triangle
  (h4 : P_isosceles = 2 * s + b)  -- definition of isosceles triangle perimeter
  : 3 * s = 60 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_alt_l2973_297321


namespace NUMINAMATH_CALUDE_range_of_m_l2973_297349

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x m : ℝ) : Prop := x < m

-- Define the relationship between ¬p and ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, ¬(q x m) → ¬(p x) ∧ ∃ y, ¬(p y) ∧ q y m

-- Theorem statement
theorem range_of_m (m : ℝ) :
  not_p_necessary_not_sufficient_for_not_q m →
  m ∈ Set.Iic (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2973_297349


namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l2973_297303

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem perpendicular_line_to_plane 
  (α β γ : Plane) (m l : Line) 
  (h1 : perp α γ)
  (h2 : intersect γ α = m)
  (h3 : intersect γ β = l)
  (h4 : perp_line l m) :
  perp_line_plane l α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l2973_297303


namespace NUMINAMATH_CALUDE_middle_circle_radius_l2973_297379

/-- A configuration of five circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  /-- The radii of the five circles, from smallest to largest -/
  radii : Fin 5 → ℝ
  /-- The radii are positive -/
  radii_pos : ∀ i, 0 < radii i
  /-- The radii are in ascending order -/
  radii_ascending : ∀ i j, i < j → radii i < radii j

/-- The theorem stating that if the smallest and largest radii are 8 and 18, 
    then the middle radius is 12 -/
theorem middle_circle_radius (c : CircleConfiguration)
    (h_smallest : c.radii 0 = 8)
    (h_largest : c.radii 4 = 18) :
    c.radii 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_circle_radius_l2973_297379


namespace NUMINAMATH_CALUDE_grid_state_theorem_l2973_297338

/-- Represents the number of times a 2x2 square was picked -/
structure SquarePicks where
  topLeft : ℕ
  topRight : ℕ
  bottomLeft : ℕ
  bottomRight : ℕ

/-- Represents the state of the 3x3 grid -/
def GridState (p : SquarePicks) : Matrix (Fin 3) (Fin 3) ℕ :=
  fun i j =>
    match i, j with
    | 0, 0 => p.topLeft
    | 0, 2 => p.topRight
    | 2, 0 => p.bottomLeft
    | 2, 2 => p.bottomRight
    | 0, 1 => p.topLeft + p.topRight
    | 1, 0 => p.topLeft + p.bottomLeft
    | 1, 2 => p.topRight + p.bottomRight
    | 2, 1 => p.bottomLeft + p.bottomRight
    | 1, 1 => p.topLeft + p.topRight + p.bottomLeft + p.bottomRight

theorem grid_state_theorem (p : SquarePicks) :
  (GridState p 2 0 = 13) →
  (GridState p 0 1 = 18) →
  (GridState p 1 1 = 47) →
  (GridState p 2 2 = 16) := by
    sorry

end NUMINAMATH_CALUDE_grid_state_theorem_l2973_297338


namespace NUMINAMATH_CALUDE_calculation_proof_l2973_297353

theorem calculation_proof : Real.sqrt 4 - Real.sin (30 * π / 180) - (π - 1) ^ 0 + 2⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2973_297353


namespace NUMINAMATH_CALUDE_john_marble_weight_l2973_297384

/-- Represents a rectangular prism -/
structure RectangularPrism where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ
  density : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.height * prism.baseLength * prism.baseWidth

/-- Calculates the weight of a rectangular prism -/
def weight (prism : RectangularPrism) : ℝ :=
  prism.density * volume prism

/-- The main theorem stating the weight of John's marble prism -/
theorem john_marble_weight :
  let prism : RectangularPrism := {
    height := 8,
    baseLength := 2,
    baseWidth := 2,
    density := 2700
  }
  weight prism = 86400 := by
  sorry


end NUMINAMATH_CALUDE_john_marble_weight_l2973_297384


namespace NUMINAMATH_CALUDE_continuous_cauchy_solution_is_linear_l2973_297324

/-- Cauchy's functional equation -/
def CauchyEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

/-- The theorem stating that continuous solutions of Cauchy's equation are linear -/
theorem continuous_cauchy_solution_is_linear
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_cauchy : CauchyEquation f) :
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
sorry

end NUMINAMATH_CALUDE_continuous_cauchy_solution_is_linear_l2973_297324


namespace NUMINAMATH_CALUDE_min_value_expression_l2973_297363

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 10) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 11 ∧
  (∃ x₀ : ℝ, x₀ > 1 ∧ (x₀ + 10) / Real.sqrt (x₀ - 1) = 2 * Real.sqrt 11 ∧ x₀ = 12) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2973_297363


namespace NUMINAMATH_CALUDE_interval_length_implies_c_minus_three_l2973_297358

theorem interval_length_implies_c_minus_three (c : ℝ) : 
  (∃ x : ℝ, 3 ≤ 5*x - 4 ∧ 5*x - 4 ≤ c) →
  (∀ x : ℝ, 3 ≤ 5*x - 4 ∧ 5*x - 4 ≤ c → (7/5 : ℝ) ≤ x ∧ x ≤ (c + 4)/5) →
  ((c + 4)/5 - 7/5 = 15) →
  c - 3 = 75 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_c_minus_three_l2973_297358


namespace NUMINAMATH_CALUDE_meeting_size_l2973_297343

/-- Represents the number of people attending the meeting -/
def n : ℕ → ℕ := λ k => 12 * k

/-- Represents the number of handshakes each person makes -/
def handshakes : ℕ → ℕ := λ k => 3 * k + 6

/-- Represents the number of mutual handshakes between any two people -/
def mutual_handshakes : ℕ → ℚ := λ k => 
  ((3 * k + 6) * (3 * k + 5)) / (12 * k - 1)

theorem meeting_size : 
  ∃ k : ℕ, k > 0 ∧ 
    (∀ i j : Fin (n k), i ≠ j → 
      (mutual_handshakes k).num % (mutual_handshakes k).den = 0) ∧
    n k = 36 := by
  sorry

end NUMINAMATH_CALUDE_meeting_size_l2973_297343


namespace NUMINAMATH_CALUDE_odd_floor_time_building_floor_time_l2973_297354

theorem odd_floor_time (total_floors : ℕ) (even_floor_time : ℕ) (total_time : ℕ) : ℕ :=
  let odd_floors := (total_floors + 1) / 2
  let even_floors := total_floors / 2
  let even_total_time := even_floors * even_floor_time
  let odd_total_time := total_time - even_total_time
  odd_total_time / odd_floors

/-- 
Given a building with 10 floors, where:
- It takes 15 seconds to reach each even-numbered floor
- It takes 120 seconds (2 minutes) to reach the 10th floor
Prove that it takes 9 seconds to reach each odd-numbered floor
-/
theorem building_floor_time : odd_floor_time 10 15 120 = 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_floor_time_building_floor_time_l2973_297354


namespace NUMINAMATH_CALUDE_abs_sum_equality_l2973_297346

theorem abs_sum_equality (a b c : ℤ) (h : |a - b| + |c - a| = 1) :
  |a - c| + |c - b| + |b - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_equality_l2973_297346


namespace NUMINAMATH_CALUDE_cd_total_length_l2973_297365

theorem cd_total_length : 
  let cd1 : ℝ := 1.5
  let cd2 : ℝ := 1.5
  let cd3 : ℝ := 2 * cd1
  let cd4 : ℝ := 0.5 * cd2
  let cd5 : ℝ := cd1 + cd2
  cd1 + cd2 + cd3 + cd4 + cd5 = 9.75 := by
sorry

end NUMINAMATH_CALUDE_cd_total_length_l2973_297365


namespace NUMINAMATH_CALUDE_grid_toothpicks_l2973_297348

/-- The number of toothpicks needed to construct a grid of given length and width -/
def toothpicks_needed (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem stating that a grid of 50 toothpicks long and 40 toothpicks wide requires 4090 toothpicks -/
theorem grid_toothpicks : toothpicks_needed 50 40 = 4090 := by
  sorry

end NUMINAMATH_CALUDE_grid_toothpicks_l2973_297348


namespace NUMINAMATH_CALUDE_lining_fabric_cost_l2973_297367

/-- The cost of lining fabric per yard -/
def lining_cost : ℝ := 30.69

theorem lining_fabric_cost :
  let velvet_cost : ℝ := 24
  let pattern_cost : ℝ := 15
  let thread_cost : ℝ := 3 * 2
  let buttons_cost : ℝ := 14
  let trim_cost : ℝ := 19 * 3
  let velvet_yards : ℝ := 5
  let lining_yards : ℝ := 4
  let discount_rate : ℝ := 0.1
  let total_cost : ℝ := 310.50
  
  total_cost = (1 - discount_rate) * (velvet_cost * velvet_yards + lining_cost * lining_yards) +
               pattern_cost + thread_cost + buttons_cost + trim_cost :=
by sorry


end NUMINAMATH_CALUDE_lining_fabric_cost_l2973_297367


namespace NUMINAMATH_CALUDE_sequence_properties_l2973_297325

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ+) : ℚ := n^2 + 2*n

/-- The nth term of the sequence a_n -/
def a (n : ℕ+) : ℚ := 2*n + 1

/-- The nth term of the sequence b_n -/
def b (n : ℕ+) : ℚ := 1 / (a n * a (n + 1))

/-- The sum of the first n terms of the sequence b_n -/
def T (n : ℕ+) : ℚ := n / (3 * (2*n + 3))

theorem sequence_properties (n : ℕ+) :
  (∀ k : ℕ+, k ≤ n → S k = k^2 + 2*k) →
  (a n = 2*n + 1) ∧
  (T n = n / (3 * (2*n + 3))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2973_297325


namespace NUMINAMATH_CALUDE_cubic_quadratic_comparison_l2973_297373

theorem cubic_quadratic_comparison (n : ℝ) :
  (n > -1 → n^3 + 1 > n^2 + n) ∧ (n < -1 → n^3 + 1 < n^2 + n) := by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_comparison_l2973_297373


namespace NUMINAMATH_CALUDE_average_age_proof_l2973_297399

def average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age1 : ℕ) (leaving_age2 : ℕ) : ℚ :=
  let total_age := initial_people * initial_average
  let remaining_age := total_age - (leaving_age1 + leaving_age2)
  let remaining_people := initial_people - 2
  remaining_age / remaining_people

theorem average_age_proof :
  average_age_after_leaving 7 28 22 25 = 29.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_proof_l2973_297399


namespace NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l2973_297377

theorem six_digit_divisibility_by_seven (a b c d e f : Nat) 
  (h1 : a ≥ 1 ∧ a ≤ 9)  -- Ensure it's a six-digit number
  (h2 : b ≥ 0 ∧ b ≤ 9)
  (h3 : c ≥ 0 ∧ c ≤ 9)
  (h4 : d ≥ 0 ∧ d ≤ 9)
  (h5 : e ≥ 0 ∧ e ≤ 9)
  (h6 : f ≥ 0 ∧ f ≤ 9)
  (h7 : (100 * a + 10 * b + c) - (100 * d + 10 * e + f) ≡ 0 [MOD 7]) :
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ≡ 0 [MOD 7] := by
  sorry

#check six_digit_divisibility_by_seven

end NUMINAMATH_CALUDE_six_digit_divisibility_by_seven_l2973_297377


namespace NUMINAMATH_CALUDE_value_of_expression_l2973_297301

theorem value_of_expression (x y : ℤ) (h1 : x = -1) (h2 : y = 4) : 2 * (x + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2973_297301


namespace NUMINAMATH_CALUDE_road_renovation_equation_ahead_of_schedule_l2973_297382

/-- Proves that the given equation holds true for a road renovation scenario -/
theorem road_renovation_equation (x : ℝ) (h1 : x > 5) : 
  (1500 / (x - 5) : ℝ) - (1500 / x : ℝ) = 10 ↔ 
  (1500 / (x - 5) : ℝ) = (1500 / x : ℝ) + 10 := by sorry

/-- Proves that the equation represents completing 10 days ahead of schedule -/
theorem ahead_of_schedule (x : ℝ) (h1 : x > 5) :
  (1500 / (x - 5) : ℝ) - (1500 / x : ℝ) = 10 ↔
  (1500 / (x - 5) : ℝ) = (1500 / x : ℝ) + 10 := by sorry

end NUMINAMATH_CALUDE_road_renovation_equation_ahead_of_schedule_l2973_297382


namespace NUMINAMATH_CALUDE_x_power_2023_l2973_297397

theorem x_power_2023 (x : ℝ) (h : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) : x^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2023_l2973_297397


namespace NUMINAMATH_CALUDE_pencil_box_problem_l2973_297320

structure BoxOfPencils where
  blue : ℕ
  green : ℕ

def Vasya (box : BoxOfPencils) : Prop := box.blue ≥ 4
def Kolya (box : BoxOfPencils) : Prop := box.green ≥ 5
def Petya (box : BoxOfPencils) : Prop := box.blue ≥ 3 ∧ box.green ≥ 4
def Misha (box : BoxOfPencils) : Prop := box.blue ≥ 4 ∧ box.green ≥ 4

theorem pencil_box_problem :
  ∃ (box : BoxOfPencils),
    (Vasya box ∧ ¬Kolya box ∧ Petya box ∧ Misha box) ∧
    ¬∃ (other_box : BoxOfPencils),
      ((¬Vasya other_box ∧ Kolya other_box ∧ Petya other_box ∧ Misha other_box) ∨
       (Vasya other_box ∧ Kolya other_box ∧ ¬Petya other_box ∧ Misha other_box) ∨
       (Vasya other_box ∧ Kolya other_box ∧ Petya other_box ∧ ¬Misha other_box)) :=
by
  sorry

#check pencil_box_problem

end NUMINAMATH_CALUDE_pencil_box_problem_l2973_297320


namespace NUMINAMATH_CALUDE_crescent_lake_loop_length_l2973_297333

/-- Represents the distance walked on each day of the trip -/
structure DailyDistances where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (d : DailyDistances) : Prop :=
  d.day1 + d.day2 + d.day3 = 32 ∧
  (d.day2 + d.day3) / 2 = 12 ∧
  d.day3 + d.day4 + d.day5 = 45 ∧
  d.day1 + d.day4 = 30

/-- The theorem stating that if the conditions are satisfied, the total distance is 69 miles -/
theorem crescent_lake_loop_length 
  (d : DailyDistances) 
  (h : satisfies_conditions d) : 
  d.day1 + d.day2 + d.day3 + d.day4 + d.day5 = 69 := by
  sorry

end NUMINAMATH_CALUDE_crescent_lake_loop_length_l2973_297333
