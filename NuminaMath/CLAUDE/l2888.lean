import Mathlib

namespace NUMINAMATH_CALUDE_point_B_coordinates_l2888_288865

def point_A : ℝ × ℝ := (-3, 2)

def move_right (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1 + units, p.2)

def move_down (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - units)

def point_B : ℝ × ℝ :=
  move_down (move_right point_A 1) 2

theorem point_B_coordinates :
  point_B = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l2888_288865


namespace NUMINAMATH_CALUDE_digit_sum_property_l2888_288812

def digit_sum (n : ℕ) : ℕ := sorry

def all_digits_leq_7 (n : ℕ) : Prop := sorry

theorem digit_sum_property (k : ℕ) (h1 : digit_sum k = 2187) 
  (h2 : all_digits_leq_7 (k * 2)) : digit_sum (k * 2) = 4374 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l2888_288812


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2888_288820

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z - 4) = 3 + 2 * Complex.I) : 
  z.im = 3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2888_288820


namespace NUMINAMATH_CALUDE_sally_pen_distribution_l2888_288864

/-- Represents the problem of distributing pens to students --/
def pen_distribution (total_pens : ℕ) (num_students : ℕ) (pens_home : ℕ) : ℕ → Prop :=
  λ pens_per_student : ℕ =>
    let pens_given := pens_per_student * num_students
    let remainder := total_pens - pens_given
    let pens_in_locker := remainder / 2
    pens_in_locker + pens_home = remainder

theorem sally_pen_distribution :
  pen_distribution 342 44 17 7 := by
  sorry

#check sally_pen_distribution

end NUMINAMATH_CALUDE_sally_pen_distribution_l2888_288864


namespace NUMINAMATH_CALUDE_difference_2020th_2010th_term_l2888_288844

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℤ :=
  -10 + (n - 1) * 9

-- State the theorem
theorem difference_2020th_2010th_term :
  (arithmeticSequence 2020 - arithmeticSequence 2010).natAbs = 90 := by
  sorry

end NUMINAMATH_CALUDE_difference_2020th_2010th_term_l2888_288844


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l2888_288897

theorem mrs_hilt_reading (books : ℝ) (chapters_per_book : ℝ) 
  (h1 : books = 4.0) (h2 : chapters_per_book = 4.25) : 
  books * chapters_per_book = 17 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l2888_288897


namespace NUMINAMATH_CALUDE_consecutive_integers_right_triangle_l2888_288887

theorem consecutive_integers_right_triangle (m n : ℕ) (h : n^2 = 2*m + 1) :
  n^2 + m^2 = (m + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_right_triangle_l2888_288887


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l2888_288858

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' + 6 = x' * y' → 2 * x + y ≤ 2 * x' + y') ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ + 6 = x₀ * y₀ ∧ 2 * x₀ + y₀ = 12) :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l2888_288858


namespace NUMINAMATH_CALUDE_symmetric_points_solution_l2888_288839

/-- 
Given two points P and Q that are symmetric about the x-axis,
prove that their coordinates satisfy the given conditions and
result in specific values for a and b.
-/
theorem symmetric_points_solution :
  ∀ (a b : ℝ),
  let P : ℝ × ℝ := (-a + 3*b, 3)
  let Q : ℝ × ℝ := (-5, a - 2*b)
  -- P and Q are symmetric about the x-axis
  (P.1 = Q.1 ∧ P.2 = -Q.2) →
  (a = -19 ∧ b = -8) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_solution_l2888_288839


namespace NUMINAMATH_CALUDE_figure_placement_count_l2888_288898

/-- Represents a configuration of figure placements -/
structure FigurePlacement where
  pages : Fin 6 → Fin 3
  order_preserved : ∀ i j : Fin 4, i < j → pages i ≤ pages j

/-- The number of valid figure placements -/
def count_placements : ℕ := sorry

/-- Theorem stating the correct number of placements -/
theorem figure_placement_count : count_placements = 225 := by sorry

end NUMINAMATH_CALUDE_figure_placement_count_l2888_288898


namespace NUMINAMATH_CALUDE_electrician_wage_l2888_288884

/-- Given the following conditions:
  1. A bricklayer and an electrician worked for a total of 90 hours.
  2. The bricklayer's wage is $12 per hour.
  3. The total payment for both workers is $1350.
  4. The bricklayer worked for 67.5 hours.
Prove that the electrician's hourly wage is $24. -/
theorem electrician_wage (total_hours : ℝ) (bricklayer_wage : ℝ) (total_payment : ℝ) (bricklayer_hours : ℝ)
  (h1 : total_hours = 90)
  (h2 : bricklayer_wage = 12)
  (h3 : total_payment = 1350)
  (h4 : bricklayer_hours = 67.5) :
  (total_payment - bricklayer_wage * bricklayer_hours) / (total_hours - bricklayer_hours) = 24 := by
  sorry

end NUMINAMATH_CALUDE_electrician_wage_l2888_288884


namespace NUMINAMATH_CALUDE_polynomial_difference_divisibility_l2888_288874

/-- Given a polynomial P with integer coefficients, 
    (a-b) divides (P(a)-P(b)) for all integers a and b -/
theorem polynomial_difference_divisibility 
  (P : Polynomial ℤ) (a b : ℤ) : 
  (a - b) ∣ (P.eval a - P.eval b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_difference_divisibility_l2888_288874


namespace NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l2888_288809

def A (a : ℕ) : Set ℕ := {1, a}
def B : Set ℕ := {1, 2, 3}

theorem a_equals_3_sufficient_not_necessary :
  (∀ a : ℕ, a = 3 → A a ⊆ B) ∧
  (∃ a : ℕ, A a ⊆ B ∧ a ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_3_sufficient_not_necessary_l2888_288809


namespace NUMINAMATH_CALUDE_area_gray_region_l2888_288880

/-- The area of the gray region between two concentric circles -/
theorem area_gray_region (r : ℝ) (h1 : r > 0) (h2 : 2 * r = r + 3) : 
  π * (2 * r)^2 - π * r^2 = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_area_gray_region_l2888_288880


namespace NUMINAMATH_CALUDE_sum_in_base3_l2888_288876

/-- Represents a number in base 3 --/
def Base3 : Type := List (Fin 3)

/-- Converts a natural number to its base 3 representation --/
def toBase3 (n : ℕ) : Base3 := sorry

/-- Adds two Base3 numbers --/
def addBase3 (a b : Base3) : Base3 := sorry

/-- Theorem: The sum of 2₃, 21₃, 110₃, and 2202₃ in base 3 is 11000₃ --/
theorem sum_in_base3 :
  addBase3 (toBase3 2)
    (addBase3 (toBase3 7)
      (addBase3 (toBase3 12)
        (toBase3 72))) = [1, 1, 0, 0, 0] := by sorry

end NUMINAMATH_CALUDE_sum_in_base3_l2888_288876


namespace NUMINAMATH_CALUDE_neha_removed_amount_l2888_288805

/-- The amount removed from Neha's share in a money division problem -/
theorem neha_removed_amount (total : ℝ) (mahi_share : ℝ) (sabi_removed : ℝ) (mahi_removed : ℝ) :
  total = 1100 →
  mahi_share = 102 →
  sabi_removed = 8 →
  mahi_removed = 4 →
  ∃ (neha_share sabi_share neha_removed : ℝ),
    neha_share + sabi_share + mahi_share = total ∧
    neha_share - neha_removed = 2 * ((sabi_share - sabi_removed) / 8) ∧
    mahi_share - mahi_removed = 6 * ((sabi_share - sabi_removed) / 8) ∧
    neha_removed = 826.70 := by
  sorry

#eval (826.70 : Float)

end NUMINAMATH_CALUDE_neha_removed_amount_l2888_288805


namespace NUMINAMATH_CALUDE_vitamin_d_scientific_notation_l2888_288846

theorem vitamin_d_scientific_notation : 0.0000046 = 4.6 * 10^(-6) := by
  sorry

end NUMINAMATH_CALUDE_vitamin_d_scientific_notation_l2888_288846


namespace NUMINAMATH_CALUDE_A_is_irrational_l2888_288819

/-- The sequence of consecutive prime numbers -/
def consecutive_primes : ℕ → ℕ := sorry

/-- The decimal representation of our number -/
def A : ℝ := sorry

/-- Dirichlet's theorem on arithmetic progressions -/
axiom dirichlet_theorem : ∃ (infinitely_many : Set ℕ), ∀ p ∈ infinitely_many, 
  ∃ (n x : ℕ), p = 10^(n+1) * x + 1 ∧ Prime p

/-- The main theorem: A is irrational -/
theorem A_is_irrational : Irrational A := sorry

end NUMINAMATH_CALUDE_A_is_irrational_l2888_288819


namespace NUMINAMATH_CALUDE_marble_probability_difference_l2888_288826

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1001

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1001

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_same : ℚ := (red_marbles.choose 2 + black_marbles.choose 2 : ℚ) / total_marbles.choose 2

/-- The probability of drawing two marbles of different colors -/
def P_diff : ℚ := (red_marbles * black_marbles : ℚ) / total_marbles.choose 2

/-- The theorem stating that the absolute difference between P_same and P_diff is 1/2001 -/
theorem marble_probability_difference : |P_same - P_diff| = 1 / 2001 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l2888_288826


namespace NUMINAMATH_CALUDE_bookstore_shipment_problem_l2888_288818

theorem bookstore_shipment_problem :
  ∀ (B : ℕ), 
    (70 : ℚ) / 100 * B = 45 →
    B = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_problem_l2888_288818


namespace NUMINAMATH_CALUDE_horner_method_difference_l2888_288877

def f (x : ℝ) : ℝ := 1 - 5*x - 8*x^2 + 10*x^3 + 6*x^4 + 12*x^5 + 3*x^6

def v₀ : ℝ := 3
def v₁ (x : ℝ) : ℝ := v₀ * x + 12
def v₂ (x : ℝ) : ℝ := v₁ x * x + 6
def v₃ (x : ℝ) : ℝ := v₂ x * x + 10
def v₄ (x : ℝ) : ℝ := v₃ x * x - 8

theorem horner_method_difference (x : ℝ) (hx : x = -4) :
  (max v₀ (max (v₁ x) (max (v₂ x) (max (v₃ x) (v₄ x))))) -
  (min v₀ (min (v₁ x) (min (v₂ x) (min (v₃ x) (v₄ x))))) = 62 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_difference_l2888_288877


namespace NUMINAMATH_CALUDE_f_min_value_a_range_characterization_l2888_288833

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * abs (x - 2) - x + 5

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 := by
  sorry

-- Define the set of valid values for a
def valid_a_set : Set ℝ := {a | a ≤ -5 ∨ a ≥ 1}

-- Theorem for the range of a
theorem a_range_characterization (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) + abs (x + 2) ≥ 3) ↔ a ∈ valid_a_set := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_a_range_characterization_l2888_288833


namespace NUMINAMATH_CALUDE_expression_value_l2888_288817

theorem expression_value (α : Real) (h : Real.tan α = -3/4) :
  (3 * (Real.sin (α/2))^2 + 2 * Real.sin (α/2) * Real.cos (α/2) + (Real.cos (α/2))^2 - 2) /
  (Real.sin (π/2 + α) * Real.tan (-3*π + α) + Real.cos (6*π - α)) = -7 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2888_288817


namespace NUMINAMATH_CALUDE_largest_k_inequality_l2888_288893

theorem largest_k_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (∀ k : ℕ+, (1 / (a - b) + 1 / (b - c) ≥ k / (a - c)) → k ≤ 4) ∧ 
  (∃ a b c : ℝ, a > b ∧ b > c ∧ 1 / (a - b) + 1 / (b - c) = 4 / (a - c)) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_inequality_l2888_288893


namespace NUMINAMATH_CALUDE_expression_equality_l2888_288824

theorem expression_equality (x y z : ℝ) :
  (-5 * x^3 * y^2 * z^3)^2 / (5 * x * y^2) * (6 * x^4 * y)^0 = 5 * x^5 * y^2 * z^6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2888_288824


namespace NUMINAMATH_CALUDE_building_heights_properties_l2888_288867

/-- Heights of the buildings in meters -/
def burj_khalifa : ℕ := 828
def shanghai_tower : ℕ := 632
def one_world_trade_center : ℕ := 541
def willis_tower : ℕ := 527

/-- List of building heights -/
def building_heights : List ℕ := [burj_khalifa, shanghai_tower, one_world_trade_center, willis_tower]

/-- Theorem stating the total height and average height difference -/
theorem building_heights_properties :
  (building_heights.sum = 2528) ∧
  (((building_heights.map (λ h => h - willis_tower)).sum : ℚ) / 4 = 105) := by
  sorry

end NUMINAMATH_CALUDE_building_heights_properties_l2888_288867


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l2888_288849

def point_P (y : ℝ) : ℝ × ℝ := (-3, y)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

theorem product_of_y_coordinates (k : ℝ) (h : k > 0) :
  ∃ y1 y2 : ℝ,
    distance_squared (point_P y1) (5, 2) = k^2 ∧
    distance_squared (point_P y2) (5, 2) = k^2 ∧
    y1 * y2 = 68 - k^2 :=
  sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l2888_288849


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2888_288841

theorem quadratic_roots_sum_of_squares (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧
    x₁^2 + x₂^2 = 19) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l2888_288841


namespace NUMINAMATH_CALUDE_binary_10010_is_18_l2888_288875

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10010_is_18 : 
  binary_to_decimal [false, true, false, false, true] = 18 := by
  sorry

end NUMINAMATH_CALUDE_binary_10010_is_18_l2888_288875


namespace NUMINAMATH_CALUDE_floor_painting_theorem_l2888_288814

/-- The number of integer solutions to the floor painting problem -/
def floor_painting_solutions : Nat :=
  (Finset.filter
    (fun p : Nat × Nat =>
      let a := p.1
      let b := p.2
      b > a ∧ (a - 4) * (b - 4) = a * b / 2)
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The floor painting problem has exactly 3 solutions -/
theorem floor_painting_theorem : floor_painting_solutions = 3 := by
  sorry

end NUMINAMATH_CALUDE_floor_painting_theorem_l2888_288814


namespace NUMINAMATH_CALUDE_gravel_cost_theorem_l2888_288837

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 4

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 8

/-- The total cost of gravel for a given volume in cubic yards -/
def total_cost (volume_cubic_yards : ℝ) : ℝ :=
  volume_cubic_yards * cubic_yards_to_cubic_feet * gravel_cost_per_cubic_foot

theorem gravel_cost_theorem : total_cost gravel_volume_cubic_yards = 864 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_theorem_l2888_288837


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2888_288806

theorem solve_exponential_equation :
  ∃ x : ℝ, (4 : ℝ) ^ x * (4 : ℝ) ^ x * (4 : ℝ) ^ x = 256 ^ 3 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2888_288806


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l2888_288832

theorem factorization_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m - 2) * (m + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l2888_288832


namespace NUMINAMATH_CALUDE_point_distance_theorem_l2888_288888

/-- Given a point P with coordinates (x^2 - k, -5), where k is a positive constant,
    if the distance from P to the x-axis is half the distance from P to the y-axis,
    and the total distance from P to both axes is 15 units,
    then k = x^2 - 10. -/
theorem point_distance_theorem (x k : ℝ) (h1 : k > 0) :
  let P : ℝ × ℝ := (x^2 - k, -5)
  abs P.2 = (1/2) * abs P.1 →
  abs P.2 + abs P.1 = 15 →
  k = x^2 - 10 := by
sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l2888_288888


namespace NUMINAMATH_CALUDE_number_division_problem_l2888_288878

theorem number_division_problem (x : ℝ) : (x / 5 = 30 + x / 6) ↔ (x = 900) := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2888_288878


namespace NUMINAMATH_CALUDE_division_problem_l2888_288823

theorem division_problem (a b c d : ℚ) : 
  a + b + c + d = 5440 ∧
  a / b = 2 / 3 ∧
  b / c = 3 / 5 ∧
  c / d = 5 / 6 →
  a = 680 ∧ b = 1020 ∧ c = 1700 ∧ d = 2040 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2888_288823


namespace NUMINAMATH_CALUDE_three_X_four_equals_ten_l2888_288831

-- Define the operation X
def X (a b : ℤ) : ℤ := 2*b + 5*a - a^2 - b

-- Theorem statement
theorem three_X_four_equals_ten : X 3 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_three_X_four_equals_ten_l2888_288831


namespace NUMINAMATH_CALUDE_no_solution_range_l2888_288825

theorem no_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) → a ∈ Set.Iic 8 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_range_l2888_288825


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l2888_288827

theorem jelly_bean_problem (b c : ℕ) : 
  b = 3 * c →                  -- Initial ratio
  b - 5 = 5 * (c - 15) →       -- Ratio after eating jelly beans
  b = 105 :=                   -- Conclusion
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l2888_288827


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2888_288801

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℚ) (h1 : principal = 23) 
  (h2 : time = 3) (h3 : interest = 3.45) : 
  interest / (principal * time) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l2888_288801


namespace NUMINAMATH_CALUDE_not_fourth_ABE_l2888_288890

-- Define the set of runners
inductive Runner : Type
  | A | B | C | D | E | F

-- Define the ordering relation for runners
def beats : Runner → Runner → Prop := sorry

-- Define the race result as a function from position to runner
def raceResult : Nat → Runner := sorry

-- State the given conditions
axiom A_beats_B : beats Runner.A Runner.B
axiom A_beats_C : beats Runner.A Runner.C
axiom B_beats_D : beats Runner.B Runner.D
axiom B_beats_E : beats Runner.B Runner.E
axiom C_beats_F : beats Runner.C Runner.F
axiom E_after_B_before_C : beats Runner.B Runner.E ∧ beats Runner.E Runner.C

-- Define what it means to finish in a certain position
def finishesIn (r : Runner) (pos : Nat) : Prop :=
  raceResult pos = r

-- State the theorem
theorem not_fourth_ABE :
  ¬(finishesIn Runner.A 4) ∧ ¬(finishesIn Runner.B 4) ∧ ¬(finishesIn Runner.E 4) :=
by sorry

end NUMINAMATH_CALUDE_not_fourth_ABE_l2888_288890


namespace NUMINAMATH_CALUDE_six_people_arrangement_l2888_288892

/-- The number of ways to arrange 6 people in a line with two specific people not adjacent -/
def line_arrangement (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial (n - k) * (Nat.choose (n - k + 1) k)

theorem six_people_arrangement :
  line_arrangement 6 2 = 480 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l2888_288892


namespace NUMINAMATH_CALUDE_equation_solution_l2888_288842

theorem equation_solution : ∃ x : ℝ, (2 / (3 * x) = 1 / (x + 2)) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2888_288842


namespace NUMINAMATH_CALUDE_bankers_gain_interest_rate_l2888_288882

/-- Given a banker's gain, present worth, and time period, 
    calculate the annual interest rate. -/
theorem bankers_gain_interest_rate 
  (bankers_gain : ℝ) 
  (present_worth : ℝ) 
  (time_period : ℕ) 
  (h1 : bankers_gain = 36) 
  (h2 : present_worth = 400) 
  (h3 : time_period = 3) :
  ∃ r : ℝ, bankers_gain = present_worth * (1 + r)^time_period - present_worth :=
sorry

end NUMINAMATH_CALUDE_bankers_gain_interest_rate_l2888_288882


namespace NUMINAMATH_CALUDE_grouping_theorem_l2888_288866

/- Define the number of men and women -/
def num_men : ℕ := 4
def num_women : ℕ := 5

/- Define the size of each group -/
def group_size : ℕ := 3

/- Define the total number of groups -/
def num_groups : ℕ := 3

/- Define the function to calculate the number of ways to group people -/
def group_ways : ℕ :=
  let first_group_men := 1
  let first_group_women := 2
  let second_group_men := 2
  let second_group_women := 1
  (num_men.choose first_group_men * num_women.choose first_group_women) *
  ((num_men - first_group_men).choose second_group_men * (num_women - first_group_women).choose second_group_women)

/- Theorem statement -/
theorem grouping_theorem :
  group_ways = 360 :=
sorry

end NUMINAMATH_CALUDE_grouping_theorem_l2888_288866


namespace NUMINAMATH_CALUDE_log_50000_sum_consecutive_integers_l2888_288804

theorem log_50000_sum_consecutive_integers : ∃ (a b : ℕ), 
  (a + 1 = b) ∧ 
  (a : ℝ) < Real.log 50000 / Real.log 10 ∧ 
  Real.log 50000 / Real.log 10 < (b : ℝ) ∧ 
  a + b = 9 := by
sorry

end NUMINAMATH_CALUDE_log_50000_sum_consecutive_integers_l2888_288804


namespace NUMINAMATH_CALUDE_bowling_team_size_l2888_288871

theorem bowling_team_size (original_avg : ℝ) (new_player1_weight : ℝ) (new_player2_weight : ℝ) (new_avg : ℝ) :
  original_avg = 103 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  new_avg = 99 →
  ∃ n : ℕ, n > 0 ∧ n * original_avg + new_player1_weight + new_player2_weight = (n + 2) * new_avg ∧ n = 7 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_size_l2888_288871


namespace NUMINAMATH_CALUDE_min_cooking_time_is_15_l2888_288879

/-- Represents the duration of each cooking step in minutes -/
structure CookingSteps :=
  (washPot : ℕ)
  (washVegetables : ℕ)
  (prepareNoodles : ℕ)
  (boilWater : ℕ)
  (cookNoodles : ℕ)

/-- Calculates the minimum cooking time given the cooking steps -/
def minCookingTime (steps : CookingSteps) : ℕ :=
  max steps.boilWater (steps.washPot + steps.washVegetables + steps.prepareNoodles + steps.cookNoodles)

/-- Theorem stating that the minimum cooking time for the given steps is 15 minutes -/
theorem min_cooking_time_is_15 (steps : CookingSteps) 
  (h1 : steps.washPot = 2)
  (h2 : steps.washVegetables = 6)
  (h3 : steps.prepareNoodles = 2)
  (h4 : steps.boilWater = 10)
  (h5 : steps.cookNoodles = 3) :
  minCookingTime steps = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_cooking_time_is_15_l2888_288879


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_l2888_288835

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpendicular m α →
  parallel m n →
  subset n β →
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_l2888_288835


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2888_288815

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c ≥ a ∧ c ≥ b) : 
  a + b ≤ c * Real.sqrt 2 ∧ (a + b = c * Real.sqrt 2 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2888_288815


namespace NUMINAMATH_CALUDE_fraction_to_decimal_equivalence_l2888_288834

theorem fraction_to_decimal_equivalence : (1 : ℚ) / 4 = (25 : ℚ) / 100 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_equivalence_l2888_288834


namespace NUMINAMATH_CALUDE_johns_nap_hours_l2888_288802

/-- Calculates the total hours of naps taken over a given number of days -/
def total_nap_hours (naps_per_week : ℕ) (hours_per_nap : ℕ) (total_days : ℕ) : ℕ :=
  (total_days / 7) * naps_per_week * hours_per_nap

/-- Theorem: John's total nap hours in 70 days -/
theorem johns_nap_hours :
  total_nap_hours 3 2 70 = 60 := by
  sorry

end NUMINAMATH_CALUDE_johns_nap_hours_l2888_288802


namespace NUMINAMATH_CALUDE_smallest_a_for_polynomial_l2888_288847

theorem smallest_a_for_polynomial (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - a*x^2 + b*x - 30030 = (x - r₁)*(x - r₂)*(x - r₃)) →
  a = r₁ + r₂ + r₃ →
  r₁ * r₂ * r₃ = 30030 →
  a ≥ 184 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_polynomial_l2888_288847


namespace NUMINAMATH_CALUDE_function_equation_solution_l2888_288828

open Real

theorem function_equation_solution (f : ℝ → ℝ) (h : ∀ x ∈ Set.Ioo (-1) 1, 2 * f x - f (-x) = log (x + 1)) :
  ∀ x ∈ Set.Ioo (-1) 1, f x = (2/3) * log (x + 1) + (1/3) * log (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2888_288828


namespace NUMINAMATH_CALUDE_smallest_sum_is_134_l2888_288886

def digits : List Nat := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def sum_of_arrangement (a b c d : Nat) : Nat :=
  10 * a + b + 10 * c + d

theorem smallest_sum_is_134 :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    sum_of_arrangement a b c d ≥ 134 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_is_134_l2888_288886


namespace NUMINAMATH_CALUDE_triangle_perimeter_after_tripling_l2888_288830

theorem triangle_perimeter_after_tripling (a b c : ℝ) :
  a = 8 → b = 15 → c = 17 →
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) →
  (3 * a + 3 * b + 3 * c = 120) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_after_tripling_l2888_288830


namespace NUMINAMATH_CALUDE_candles_from_beehives_l2888_288845

/-- Given that 3 beehives can make enough wax for 12 candles,
    prove that 24 beehives can make enough wax for 96 candles. -/
theorem candles_from_beehives :
  ∀ (beehives candles : ℕ),
    beehives = 3 →
    candles = 12 →
    (24 : ℕ) * candles / beehives = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_candles_from_beehives_l2888_288845


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2888_288899

/-- An equilateral triangle with perimeter 15 meters has sides of length 5 meters. -/
theorem equilateral_triangle_side_length (triangle : Set ℝ) (perimeter : ℝ) : 
  perimeter = 15 → 
  (∃ side : ℝ, side > 0 ∧ 
    (∀ s : ℝ, s ∈ triangle → s = side) ∧ 
    3 * side = perimeter) → 
  (∃ side : ℝ, side = 5 ∧ 
    (∀ s : ℝ, s ∈ triangle → s = side)) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2888_288899


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l2888_288885

theorem quadratic_perfect_square (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 + 20 * x + 9 = (r * x + s)^2) →
  a = 100 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l2888_288885


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l2888_288810

theorem angle_terminal_side_point (α : Real) :
  let P : ℝ × ℝ := (4, -3)
  (P.1 = 4 ∧ P.2 = -3) →
  2 * Real.sin α + Real.cos α = -2/5 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l2888_288810


namespace NUMINAMATH_CALUDE_min_value_of_fraction_min_value_achieved_l2888_288836

theorem min_value_of_fraction (x : ℝ) (h : x > 9) : 
  (x^2) / (x - 9) ≥ 36 := by
sorry

theorem min_value_achieved (x : ℝ) (h : x > 9) : 
  (x^2) / (x - 9) = 36 ↔ x = 18 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_min_value_achieved_l2888_288836


namespace NUMINAMATH_CALUDE_percentage_problem_l2888_288838

theorem percentage_problem (x : ℝ) :
  (0.15 * 0.30 * (x / 100) * 5200 = 117) → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2888_288838


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l2888_288822

/-- Calculates the on-time departure rate given the number of on-time departures and total flights -/
def onTimeRate (onTime : ℕ) (total : ℕ) : ℚ :=
  onTime / total

/-- Proves that adding one more on-time flight after 3 on-time and 1 late flight 
    results in an on-time rate higher than 60% -/
theorem phoenix_airport_on_time_rate : 
  let initialOnTime : ℕ := 3
  let initialTotal : ℕ := 4
  let additionalOnTime : ℕ := 1
  onTimeRate (initialOnTime + additionalOnTime) (initialTotal + additionalOnTime) > 60 / 100 := by
  sorry

#eval onTimeRate 4 5 > 60 / 100

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l2888_288822


namespace NUMINAMATH_CALUDE_distance_after_two_hours_l2888_288861

-- Define Aaron's speed
def aaron_speed : ℚ := 1 / 20

-- Define Mia's speed
def mia_speed : ℚ := 3 / 40

-- Define the time period in hours
def time_period : ℚ := 2

-- Define the direction multiplier (opposite directions)
def direction_multiplier : ℚ := 2

-- Theorem statement
theorem distance_after_two_hours :
  (aaron_speed * (time_period * 60) + mia_speed * (time_period * 60)) * direction_multiplier = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_two_hours_l2888_288861


namespace NUMINAMATH_CALUDE_louise_boxes_needed_louise_needs_23_boxes_l2888_288853

/-- Represents the number of pencils a box can hold for each color --/
structure BoxCapacity where
  red : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- Represents the number of pencils Louise has for each color --/
structure PencilCount where
  red : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- Calculates the number of boxes needed for a given color --/
def boxesNeeded (pencils : Nat) (capacity : Nat) : Nat :=
  (pencils + capacity - 1) / capacity

/-- Theorem: Given the specified conditions, Louise needs 23 boxes in total --/
theorem louise_boxes_needed (capacity : BoxCapacity) (count : PencilCount) : Nat :=
  have red_boxes := boxesNeeded count.red capacity.red
  have blue_boxes := boxesNeeded count.blue capacity.blue
  have yellow_boxes := boxesNeeded count.yellow capacity.yellow
  have green_boxes := boxesNeeded count.green capacity.green
  red_boxes + blue_boxes + yellow_boxes + green_boxes

/-- Main theorem: Louise needs 23 boxes given the specific conditions --/
theorem louise_needs_23_boxes : 
  let capacity := BoxCapacity.mk 15 25 10 30
  let count := PencilCount.mk 45 (3 * 45) 80 (45 + 3 * 45)
  louise_boxes_needed capacity count = 23 := by
  sorry

end NUMINAMATH_CALUDE_louise_boxes_needed_louise_needs_23_boxes_l2888_288853


namespace NUMINAMATH_CALUDE_vector_calculation_l2888_288850

/-- Given vectors AB and BC in 2D space, prove that -1/2 * AC equals the specified vector -/
theorem vector_calculation (AB BC : Fin 2 → ℝ) 
  (h1 : AB = ![3, 7])
  (h2 : BC = ![-2, 3]) :
  (-1/2 : ℝ) • (AB + BC) = ![-1/2, -5] := by
  sorry

end NUMINAMATH_CALUDE_vector_calculation_l2888_288850


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l2888_288869

theorem square_root_equation_solution (P : ℝ) :
  Real.sqrt (3 - 2*P) + Real.sqrt (1 - 2*P) = 2 → P = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l2888_288869


namespace NUMINAMATH_CALUDE_triangle_properties_l2888_288860

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles of the triangle
variable (a b c : ℝ) -- Sides of the triangle opposite to A, B, C respectively

-- Define the conditions
axiom bc_cos_a : b * Real.cos A = 2
axiom area : (1/2) * b * c * Real.sin A = 2
axiom sin_relation : Real.sin B = 2 * Real.cos A * Real.sin C

-- Define the theorem
theorem triangle_properties :
  (Real.tan A = 2) ∧ (c = 5) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2888_288860


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2888_288857

/-- The number of men initially doing the work -/
def initial_men : ℕ := 50

/-- The number of days it takes for the initial number of men to complete the work -/
def initial_days : ℕ := 100

/-- The number of men needed to complete the work in 20 days -/
def men_for_20_days : ℕ := 250

/-- The number of days it takes for 250 men to complete the work -/
def days_for_250_men : ℕ := 20

theorem work_completion_theorem :
  initial_men * initial_days = men_for_20_days * days_for_250_men :=
by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l2888_288857


namespace NUMINAMATH_CALUDE_slope_determines_y_coordinate_l2888_288829

/-- Given two points R and S in a coordinate plane, if the slope of the line through R and S
    is -5/4, then the y-coordinate of S is -2. -/
theorem slope_determines_y_coordinate (x_R y_R x_S : ℚ) :
  let R : ℚ × ℚ := (x_R, y_R)
  let S : ℚ × ℚ := (x_S, y_S)
  x_R = -3 →
  y_R = 8 →
  x_S = 5 →
  (y_S - y_R) / (x_S - x_R) = -5/4 →
  y_S = -2 := by
sorry

end NUMINAMATH_CALUDE_slope_determines_y_coordinate_l2888_288829


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2888_288843

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (2 - 2*i) / (3 + 4*i) = -2/25 - 14/25*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2888_288843


namespace NUMINAMATH_CALUDE_error_percentage_limit_l2888_288873

theorem error_percentage_limit :
  ∀ ε > 0, ∃ N : ℝ, ∀ x ≥ N,
    x > 0 → |((7 * x + 8) / (8 * x)) * 100 - 87.5| < ε := by
  sorry

end NUMINAMATH_CALUDE_error_percentage_limit_l2888_288873


namespace NUMINAMATH_CALUDE_max_ac_value_l2888_288856

theorem max_ac_value (a c x y z m n : ℤ) : 
  x^2 + a*x + 48 = (x + y)*(x + z) →
  x^2 - 8*x + c = (x + m)*(x + n) →
  y ≥ -50 → y ≤ 50 →
  z ≥ -50 → z ≤ 50 →
  m ≥ -50 → m ≤ 50 →
  n ≥ -50 → n ≤ 50 →
  ∃ (a' c' : ℤ), a'*c' = 98441 ∧ ∀ (a'' c'' : ℤ), a''*c'' ≤ 98441 :=
by sorry

end NUMINAMATH_CALUDE_max_ac_value_l2888_288856


namespace NUMINAMATH_CALUDE_total_count_theorem_l2888_288852

/-- The total number of oysters and crabs counted over two days -/
def total_count (initial_oysters initial_crabs : ℕ) : ℕ :=
  let day1_total := initial_oysters + initial_crabs
  let day2_oysters := initial_oysters / 2
  let day2_crabs := initial_crabs * 2 / 3
  let day2_total := day2_oysters + day2_crabs
  day1_total + day2_total

/-- Theorem stating the total count of oysters and crabs over two days -/
theorem total_count_theorem (initial_oysters initial_crabs : ℕ) 
  (h1 : initial_oysters = 50) 
  (h2 : initial_crabs = 72) : 
  total_count initial_oysters initial_crabs = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_count_theorem_l2888_288852


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l2888_288848

theorem trig_expression_simplification (α : ℝ) :
  (Real.sin (π/2 + α) * Real.sin (π + α) * Real.tan (3*π + α)) /
  (Real.cos (3*π/2 + α) * Real.sin (-α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l2888_288848


namespace NUMINAMATH_CALUDE_punger_pages_needed_l2888_288895

/-- The number of pages needed to hold all baseball cards -/
def pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) : ℕ :=
  (packs * cards_per_pack + cards_per_page - 1) / cards_per_page

/-- Proof that 42 pages are needed for Punger's baseball cards -/
theorem punger_pages_needed :
  pages_needed 60 7 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_punger_pages_needed_l2888_288895


namespace NUMINAMATH_CALUDE_vector_simplification_l2888_288803

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_simplification (A B C D : V) :
  (B - A) + (C - B) - (D - A) = D - C := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l2888_288803


namespace NUMINAMATH_CALUDE_expression_evaluation_l2888_288854

theorem expression_evaluation : 
  (1728^2 : ℚ) / (137^3 - (137^2 - 11^2)) = 2985984 / 2552705 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2888_288854


namespace NUMINAMATH_CALUDE_zero_acceleration_in_quadrant_IV_l2888_288821

-- Define the disk and its properties
structure Disk where
  uniform : Bool
  rolling_smoothly : Bool
  pulled_by_force : Bool

-- Define the acceleration vectors
structure Acceleration where
  tangential : ℝ × ℝ
  centripetal : ℝ × ℝ
  horizontal : ℝ × ℝ

-- Define the quadrants of the disk
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Function to check if a point in a given quadrant can have zero total acceleration
def can_have_zero_acceleration (d : Disk) (q : Quadrant) (a : Acceleration) : Prop :=
  d.uniform ∧ d.rolling_smoothly ∧ d.pulled_by_force ∧
  match q with
  | Quadrant.IV => ∃ (x y : ℝ), 
      x > 0 ∧ y < 0 ∧
      a.tangential.1 + a.centripetal.1 + a.horizontal.1 = 0 ∧
      a.tangential.2 + a.centripetal.2 + a.horizontal.2 = 0
  | _ => False

-- Theorem statement
theorem zero_acceleration_in_quadrant_IV (d : Disk) (a : Acceleration) :
  d.uniform ∧ d.rolling_smoothly ∧ d.pulled_by_force →
  ∃ (q : Quadrant), can_have_zero_acceleration d q a :=
sorry

end NUMINAMATH_CALUDE_zero_acceleration_in_quadrant_IV_l2888_288821


namespace NUMINAMATH_CALUDE_female_contestant_probability_l2888_288883

theorem female_contestant_probability :
  let total_contestants : ℕ := 8
  let female_contestants : ℕ := 4
  let male_contestants : ℕ := 4
  let chosen_contestants : ℕ := 2
  
  (female_contestants.choose chosen_contestants : ℚ) / (total_contestants.choose chosen_contestants) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_female_contestant_probability_l2888_288883


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2888_288891

/-- Two hyperbolas have the same asymptotes if M = 4.5 -/
theorem hyperbolas_same_asymptotes :
  let h₁ : ℝ → ℝ → Prop := λ x y => x^2 / 9 - y^2 / 16 = 1
  let h₂ : ℝ → ℝ → ℝ → Prop := λ x y M => y^2 / 8 - x^2 / M = 1
  let asymptote₁ : ℝ → ℝ → Prop := λ x y => y = (4/3) * x ∨ y = -(4/3) * x
  let asymptote₂ : ℝ → ℝ → ℝ → Prop := λ x y M => y = Real.sqrt (8/M) * x ∨ y = -Real.sqrt (8/M) * x
  ∀ (M : ℝ), (∀ x y, asymptote₁ x y ↔ asymptote₂ x y M) → M = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l2888_288891


namespace NUMINAMATH_CALUDE_remaining_subtasks_l2888_288862

def total_problems : ℝ := 72.0
def completed_problems : ℝ := 32.0
def subtasks_per_problem : ℕ := 5

theorem remaining_subtasks : 
  (total_problems - completed_problems) * subtasks_per_problem = 200 := by
  sorry

end NUMINAMATH_CALUDE_remaining_subtasks_l2888_288862


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l2888_288894

/-- The height of a cone formed from a sector of a circular sheet -/
theorem cone_height_from_circular_sector (r : ℝ) (n : ℕ) (h : n > 0) : 
  let base_radius := r * Real.pi / (2 * n)
  let slant_height := r
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  (r = 10 ∧ n = 4) → height = Real.sqrt 93.75 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l2888_288894


namespace NUMINAMATH_CALUDE_point_on_intersection_line_l2888_288807

-- Define the sets and points
variable (α β m n l : Set Point)
variable (P : Point)

-- State the theorem
theorem point_on_intersection_line
  (h1 : α ∩ β = l)
  (h2 : m ⊆ α)
  (h3 : n ⊆ β)
  (h4 : m ∩ n = {P}) :
  P ∈ l := by
sorry

end NUMINAMATH_CALUDE_point_on_intersection_line_l2888_288807


namespace NUMINAMATH_CALUDE_henrys_cd_collection_l2888_288855

theorem henrys_cd_collection :
  ∀ (classical rock country : ℕ),
    classical = 10 →
    rock = 2 * classical →
    country = rock + 3 →
    country = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_henrys_cd_collection_l2888_288855


namespace NUMINAMATH_CALUDE_cube_sum_power_of_two_solutions_l2888_288808

theorem cube_sum_power_of_two_solutions (x y : ℤ) :
  x^3 + y^3 = 2^30 ↔ (x = 0 ∧ y = 2^10) ∨ (x = 2^10 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_power_of_two_solutions_l2888_288808


namespace NUMINAMATH_CALUDE_equation_conditions_l2888_288889

theorem equation_conditions (a b c d : ℝ) :
  (2*a + 3*b) / (b + 2*c) = (3*c + 2*d) / (d + 2*a) →
  (2*a = 3*c) ∨ (2*a + 3*b + d + 2*c = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_conditions_l2888_288889


namespace NUMINAMATH_CALUDE_triangle_squares_l2888_288872

theorem triangle_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let petya_square := a * b / (a + b)
  let vasya_square := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)
  -- Petya's square is larger than Vasya's square
  petya_square > vasya_square ∧ 
  -- Petya's square formula is correct
  (∃ (x : ℝ), x = petya_square ∧ x * (a + b) = a * b) ∧
  -- Vasya's square formula is correct
  (∃ (y : ℝ), y = vasya_square ∧ 
    y * (a^2 / b + b + a) = Real.sqrt (a^2 + b^2) * a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_squares_l2888_288872


namespace NUMINAMATH_CALUDE_even_composition_is_even_l2888_288851

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_composition_is_even (f : ℝ → ℝ) (h : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_even_composition_is_even_l2888_288851


namespace NUMINAMATH_CALUDE_correct_calculation_l2888_288859

theorem correct_calculation : ∃ (x : ℝ), x * 5 = 40 ∧ x * 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2888_288859


namespace NUMINAMATH_CALUDE_newspaper_ad_cost_newspaper_ad_cost_proof_l2888_288868

/-- The total cost for three companies purchasing ads in a newspaper -/
theorem newspaper_ad_cost (num_companies : ℕ) (num_ad_spaces : ℕ) 
  (ad_length : ℝ) (ad_width : ℝ) (cost_per_sqft : ℝ) : ℝ :=
  let ad_area := ad_length * ad_width
  let cost_per_ad := ad_area * cost_per_sqft
  let cost_per_company := cost_per_ad * num_ad_spaces
  num_companies * cost_per_company

/-- Proof that the total cost for three companies purchasing 10 ad spaces each, 
    where each ad space is a 12-foot by 5-foot rectangle and costs $60 per square foot, 
    is $108,000 -/
theorem newspaper_ad_cost_proof :
  newspaper_ad_cost 3 10 12 5 60 = 108000 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_ad_cost_newspaper_ad_cost_proof_l2888_288868


namespace NUMINAMATH_CALUDE_max_value_of_b_l2888_288863

theorem max_value_of_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : 
  b ≤ 1/3 ∧ ∃ (x : ℝ), x > 0 ∧ 2 * x * (1/3) = (2 * x - 1/3) / (2 * x + 1) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_b_l2888_288863


namespace NUMINAMATH_CALUDE_root_equation_solution_l2888_288816

theorem root_equation_solution (a b c : ℕ) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (∀ N : ℝ, N ≠ 1 → N^(1/a + 1/(a*b) + 1/(a*b*c)) = N^(25/36)) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_solution_l2888_288816


namespace NUMINAMATH_CALUDE_all_statements_correct_l2888_288811

theorem all_statements_correct :
  (∀ a b : ℕ, Odd a → Odd b → Even (a + b)) ∧
  (∀ p : ℕ, Prime p → p > 3 → ∃ k : ℕ, p^2 = 12*k + 1) ∧
  (∀ r : ℚ, ∀ i : ℝ, Irrational i → Irrational (r + i)) ∧
  (∀ n : ℕ, 2 ∣ n → 3 ∣ n → 6 ∣ n) ∧
  (∀ n : ℕ, n > 1 → Prime n ∨ ∃ (p : List ℕ), (∀ q ∈ p, Prime q) ∧ n = p.prod) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_correct_l2888_288811


namespace NUMINAMATH_CALUDE_ninth_term_is_nine_l2888_288800

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first six terms is 21 -/
  sum_first_six : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) + (a + 5*d) = 21
  /-- The seventh term is 7 -/
  seventh_term : a + 6*d = 7

/-- The ninth term of the arithmetic sequence is 9 -/
theorem ninth_term_is_nine (seq : ArithmeticSequence) : seq.a + 8*seq.d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_nine_l2888_288800


namespace NUMINAMATH_CALUDE_no_matrix_sin_B_l2888_288881

def B : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1996; 0, 1]

-- Define sin(A) using power series
noncomputable def matrix_sin (A : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  A - (A^3 / 6) + (A^5 / 120) - (A^7 / 5040) + (A^9 / 362880) - (A^11 / 39916800) + higher_order_terms
where
  higher_order_terms := sorry  -- Represents the rest of the infinite series

theorem no_matrix_sin_B : ¬ ∃ (A : Matrix (Fin 2) (Fin 2) ℝ), matrix_sin A = B := by
  sorry

end NUMINAMATH_CALUDE_no_matrix_sin_B_l2888_288881


namespace NUMINAMATH_CALUDE_solve_for_m_l2888_288896

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x - 8

-- State the theorem
theorem solve_for_m :
  ∃ m : ℝ, (f 8 - g m 8 = 20) ∧ (m = -25.5) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2888_288896


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2888_288870

theorem quadratic_solution_difference_squared :
  ∀ p q : ℝ,
  (5 * p^2 - 8 * p - 15 = 0) →
  (5 * q^2 - 8 * q - 15 = 0) →
  (p - q)^2 = 14.5924 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l2888_288870


namespace NUMINAMATH_CALUDE_quadratic_real_roots_iff_k_le_4_l2888_288840

/-- The quadratic function f(x) = (k - 3)x² + 2x + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 3) * x^2 + 2 * x + 1

/-- The discriminant of the quadratic function f -/
def discriminant (k : ℝ) : ℝ := 4 - 4 * k + 12

theorem quadratic_real_roots_iff_k_le_4 :
  ∀ k : ℝ, (∃ x : ℝ, f k x = 0) ↔ k ≤ 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_iff_k_le_4_l2888_288840


namespace NUMINAMATH_CALUDE_binomial_equation_implies_C_8_2_l2888_288813

def binomial (n m : ℕ) : ℕ := Nat.choose n m

theorem binomial_equation_implies_C_8_2 (m : ℕ) :
  (1 / binomial 5 m : ℚ) - (1 / binomial 6 m : ℚ) = (7 / (10 * binomial 7 m) : ℚ) →
  binomial 8 m = 28 :=
by sorry

end NUMINAMATH_CALUDE_binomial_equation_implies_C_8_2_l2888_288813
