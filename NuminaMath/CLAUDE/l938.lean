import Mathlib

namespace intersection_of_A_and_B_l938_93887

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | x^2 + x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end intersection_of_A_and_B_l938_93887


namespace no_xy_term_implies_k_eq_two_l938_93810

/-- Given a polynomial x^2 + kxy + 4x - 2xy + y^2 - 1, if it does not contain the term xy, then k = 2 -/
theorem no_xy_term_implies_k_eq_two (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x*y + 4*x - 2*x*y + y^2 - 1 = x^2 + 4*x + y^2 - 1) → k = 2 := by
  sorry

end no_xy_term_implies_k_eq_two_l938_93810


namespace simplified_and_rationalized_l938_93884

theorem simplified_and_rationalized (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplified_and_rationalized_l938_93884


namespace total_cookie_sales_l938_93851

/-- Represents the sales data for Robyn and Lucy's cookie selling adventure -/
structure CookieSales where
  /-- Sales in the first neighborhood -/
  neighborhood1 : Nat × Nat
  /-- Sales in the second neighborhood -/
  neighborhood2 : Nat × Nat
  /-- Sales in the third neighborhood -/
  neighborhood3 : Nat × Nat
  /-- Total sales in the first park -/
  park1_total : Nat
  /-- Total sales in the second park -/
  park2_total : Nat

/-- Theorem stating the total number of packs sold by Robyn and Lucy -/
theorem total_cookie_sales (sales : CookieSales)
  (h1 : sales.neighborhood1 = (15, 12))
  (h2 : sales.neighborhood2 = (23, 15))
  (h3 : sales.neighborhood3 = (17, 16))
  (h4 : sales.park1_total = 25)
  (h5 : ∃ x y : Nat, x = 2 * y ∧ x + y = sales.park1_total)
  (h6 : sales.park2_total = 35)
  (h7 : ∃ x y : Nat, y = x + 5 ∧ x + y = sales.park2_total) :
  (sales.neighborhood1.1 + sales.neighborhood1.2 +
   sales.neighborhood2.1 + sales.neighborhood2.2 +
   sales.neighborhood3.1 + sales.neighborhood3.2 +
   sales.park1_total + sales.park2_total) = 158 := by
  sorry

end total_cookie_sales_l938_93851


namespace black_squares_21st_row_l938_93872

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n

/-- Represents the number of black squares in a row of the stair-step figure -/
def black_squares_in_row (n : ℕ) : ℕ := 2 * (squares_in_row n / 4)

theorem black_squares_21st_row :
  black_squares_in_row 21 = 20 := by
  sorry

end black_squares_21st_row_l938_93872


namespace euler_line_equation_l938_93854

/-- The Euler line of a triangle ABC with vertices A(2,0), B(0,4), and AC = BC -/
def euler_line (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2 * p.2 + 3 = 0}

/-- Triangle ABC with given properties -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A_coord : A = (2, 0)
  B_coord : B = (0, 4)
  isosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem euler_line_equation (t : TriangleABC) :
  euler_line t.A t.B t.C = {p : ℝ × ℝ | p.1 - 2 * p.2 + 3 = 0} :=
by sorry

end euler_line_equation_l938_93854


namespace first_complete_column_coverage_l938_93862

theorem first_complete_column_coverage : 
  let triangular (n : ℕ) := n * (n + 1) / 2
  ∃ (k : ℕ), k > 0 ∧ 
    (∀ (r : ℕ), r < 8 → ∃ (n : ℕ), n ≤ k ∧ triangular n % 8 = r) ∧
    (∀ (m : ℕ), m < k → ¬(∀ (r : ℕ), r < 8 → ∃ (n : ℕ), n ≤ m ∧ triangular n % 8 = r)) ∧
  k = 15 :=
by sorry

end first_complete_column_coverage_l938_93862


namespace f_increasing_implies_k_nonpositive_l938_93883

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x - 8

theorem f_increasing_implies_k_nonpositive :
  ∀ k : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 14 → f k x < f k y) → k ≤ 0 :=
by sorry

end f_increasing_implies_k_nonpositive_l938_93883


namespace sqrt_fraction_simplification_l938_93803

theorem sqrt_fraction_simplification :
  Real.sqrt (7^2 + 24^2) / Real.sqrt (49 + 16) = (25 * Real.sqrt 65) / 65 := by
  sorry

end sqrt_fraction_simplification_l938_93803


namespace initial_bees_calculation_l938_93847

/-- Calculates the initial number of bees given the daily hatch rate, daily loss rate,
    number of days, and final number of bees. -/
def initialBees (hatchRate dailyLoss : ℕ) (days : ℕ) (finalBees : ℕ) : ℕ :=
  finalBees - (hatchRate - dailyLoss) * days

theorem initial_bees_calculation 
  (hatchRate dailyLoss days finalBees : ℕ) 
  (hatchRate_pos : hatchRate > dailyLoss) :
  initialBees hatchRate dailyLoss days finalBees = 
    finalBees - (hatchRate - dailyLoss) * days := by
  sorry

#eval initialBees 3000 900 7 27201

end initial_bees_calculation_l938_93847


namespace quadratic_equation_root_l938_93858

theorem quadratic_equation_root (k : ℚ) : 
  (∃ x : ℚ, x - x^2 = k*x^2 + 1) → 
  (2 - 2^2 = k*2^2 + 1) → 
  k = -3/4 := by
sorry

end quadratic_equation_root_l938_93858


namespace monotonicity_intervals_no_increasing_intervals_l938_93845

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + a^2

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1/x + 2*x - 2*a

theorem monotonicity_intervals (x : ℝ) (h : x > 0) :
  let a := 2
  (f_deriv a x > 0 ↔ (x < (2 - Real.sqrt 2) / 2 ∨ x > (2 + Real.sqrt 2) / 2)) ∧
  (f_deriv a x < 0 ↔ ((2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2)) :=
sorry

theorem no_increasing_intervals (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f_deriv a x ≤ 0) ↔ a ≥ 19/6 :=
sorry

end monotonicity_intervals_no_increasing_intervals_l938_93845


namespace max_tan_MPN_l938_93818

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4/25
def C2 (x y θ : ℝ) : Prop := (x - 3 - Real.cos θ)^2 + (y - Real.sin θ)^2 = 1/25

-- Define a point P on C2
def P_on_C2 (x y θ : ℝ) : Prop := C2 x y θ

-- Define tangent points M and N on C1
def tangent_points (xm ym xn yn : ℝ) : Prop := C1 xm ym ∧ C1 xn yn

-- Define the angle MPN
def angle_MPN (xp yp xm ym xn yn : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_tan_MPN :
  ∃ (xp yp θ xm ym xn yn : ℝ),
    P_on_C2 xp yp θ ∧
    tangent_points xm ym xn yn ∧
    (∀ (xp' yp' θ' xm' ym' xn' yn' : ℝ),
      P_on_C2 xp' yp' θ' →
      tangent_points xm' ym' xn' yn' →
      Real.tan (angle_MPN xp yp xm ym xn yn) ≥ Real.tan (angle_MPN xp' yp' xm' ym' xn' yn')) ∧
    Real.tan (angle_MPN xp yp xm ym xn yn) = 4 * Real.sqrt 2 / 7 :=
sorry

end max_tan_MPN_l938_93818


namespace eggs_per_group_l938_93824

/-- Given 15 eggs split into 3 equal groups, prove that each group contains 5 eggs. -/
theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (h1 : total_eggs = 15) (h2 : num_groups = 3) :
  total_eggs / num_groups = 5 := by
  sorry

end eggs_per_group_l938_93824


namespace division_rebus_proof_l938_93849

theorem division_rebus_proof :
  -- Given conditions
  let dividend : ℕ := 1089708
  let divisor : ℕ := 12
  let quotient : ℕ := 90809

  -- Divisor is a two-digit number
  (10 ≤ divisor) ∧ (divisor < 100) →
  
  -- When divisor is multiplied by 8, it results in a two-digit number
  (10 ≤ divisor * 8) ∧ (divisor * 8 < 100) →
  
  -- When divisor is multiplied by the first (or last) digit of quotient, it results in a three-digit number
  (100 ≤ divisor * (quotient / 10000)) ∧ (divisor * (quotient / 10000) < 1000) →
  
  -- Quotient has 5 digits
  (10000 ≤ quotient) ∧ (quotient < 100000) →
  
  -- Second and fourth digits of quotient are 0
  (quotient % 10000 / 1000 = 0) ∧ (quotient % 100 / 10 = 0) →
  
  -- The division problem has a unique solution
  ∃! (d q : ℕ), d * q = dividend ∧ q = quotient →
  
  -- Prove that the division is correct
  dividend / divisor = quotient ∧ dividend % divisor = 0 :=
by
  sorry

end division_rebus_proof_l938_93849


namespace midpoint_xy_product_l938_93817

/-- Given that C = (3, 5) is the midpoint of AB, where A = (1, 8) and B = (x, y), prove that xy = 10 -/
theorem midpoint_xy_product (x y : ℝ) : 
  (3 : ℝ) = (1 + x) / 2 ∧ (5 : ℝ) = (8 + y) / 2 → x * y = 10 := by
  sorry

end midpoint_xy_product_l938_93817


namespace parabola_with_directrix_x_eq_1_l938_93808

/-- A parabola is a set of points in a plane that are equidistant from a fixed point (focus) and a fixed line (directrix). -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- The standard equation of a parabola represents the set of points (x, y) that satisfy the parabola's definition. -/
def standard_equation (p : Parabola) : (ℝ × ℝ) → Prop :=
  sorry

theorem parabola_with_directrix_x_eq_1 (p : Parabola) (h : p.directrix = 1) :
  standard_equation p = fun (x, y) ↦ y^2 = -4*x := by
  sorry

end parabola_with_directrix_x_eq_1_l938_93808


namespace cube_root_equation_solutions_l938_93815

theorem cube_root_equation_solutions :
  ∀ x : ℝ, (x^(1/3) = 15 / (8 - x^(1/3))) ↔ (x = 27 ∨ x = 125) := by sorry

end cube_root_equation_solutions_l938_93815


namespace root_relationship_l938_93868

theorem root_relationship (p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ y = 2*x) → 
  2*p^2 = 9*q := by
sorry

end root_relationship_l938_93868


namespace distribute_8_balls_4_boxes_l938_93877

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that there are 139 ways to distribute 8 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_8_balls_4_boxes : distribute_balls 8 4 = 139 := by sorry

end distribute_8_balls_4_boxes_l938_93877


namespace parabola_vertex_l938_93865

/-- The parabola defined by y = -2(x-2)^2 - 5 has vertex (2, -5) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * (x - 2)^2 - 5 → (2, -5) = (x, y) := by
  sorry

end parabola_vertex_l938_93865


namespace smallest_integer_with_given_remainders_l938_93843

theorem smallest_integer_with_given_remainders : ∃ M : ℕ,
  (M > 0) ∧
  (M % 4 = 3) ∧
  (M % 5 = 4) ∧
  (M % 6 = 5) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (∀ n : ℕ, n > 0 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 → n ≥ M) ∧
  M = 2519 :=
by sorry

end smallest_integer_with_given_remainders_l938_93843


namespace fractional_equation_solution_l938_93828

theorem fractional_equation_solution :
  ∃ x : ℝ, x ≠ 3 ∧ (2 - x) / (x - 3) + 1 / (3 - x) = 1 ∧ x = 2 := by
  sorry

end fractional_equation_solution_l938_93828


namespace parabola_through_negative_x_l938_93861

/-- A parabola passing through the point (-2, 3) cannot have a standard equation of the form y^2 = 2px where p > 0 -/
theorem parabola_through_negative_x (p : ℝ) (h : p > 0) : ¬ (3^2 = 2 * p * (-2)) := by
  sorry

end parabola_through_negative_x_l938_93861


namespace part_one_part_two_part_three_l938_93897

/-- A function f(x) = ax - bx² where a > 0 and b > 0 -/
def f (a b x : ℝ) : ℝ := a * x - b * x^2

/-- Theorem for part I -/
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, f a b x ≤ 1) → a ≤ 2 * Real.sqrt b :=
sorry

/-- Theorem for part II -/
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
sorry

/-- Theorem for part III -/
theorem part_three (a b : ℝ) (ha : a > 0) (hb : 0 < b) (hb' : b ≤ 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ a ≤ b + 1 :=
sorry

end part_one_part_two_part_three_l938_93897


namespace squares_in_6x4_rectangle_l938_93880

/-- The number of unit squares that can fit in a rectangle -/
def squaresInRectangle (length width : ℕ) : ℕ := length * width

/-- Theorem: A 6x4 rectangle can fit 24 unit squares -/
theorem squares_in_6x4_rectangle :
  squaresInRectangle 6 4 = 24 := by
  sorry

end squares_in_6x4_rectangle_l938_93880


namespace power_sum_negative_two_l938_93864

theorem power_sum_negative_two : (-2)^2009 + (-2)^2010 = 2^2009 := by sorry

end power_sum_negative_two_l938_93864


namespace rain_probability_l938_93857

theorem rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.35)
  (h2 : p_sunday = 0.45)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.6425 := by
  sorry

end rain_probability_l938_93857


namespace unknown_number_proof_l938_93856

theorem unknown_number_proof (x : ℝ) : x + 5 * 12 / (180 / 3) = 41 → x = 40 := by
  sorry

end unknown_number_proof_l938_93856


namespace min_sum_given_product_l938_93881

theorem min_sum_given_product (x y : ℝ) : 
  x > 0 → y > 0 → (x - 1) * (y - 1) = 1 → x + y ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ (x - 1) * (y - 1) = 1 ∧ x + y = 4 := by
  sorry

end min_sum_given_product_l938_93881


namespace sqrt_product_simplification_l938_93890

theorem sqrt_product_simplification (x : ℝ) (hx : x ≥ 0) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) := by
  sorry

end sqrt_product_simplification_l938_93890


namespace expression_always_zero_l938_93879

theorem expression_always_zero (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ((x / |y| - |x| / y) * (y / |z| - |y| / z) * (z / |x| - |z| / x)) = 0 := by
  sorry

end expression_always_zero_l938_93879


namespace jane_albert_same_committee_l938_93829

/-- The number of second-year MBAs -/
def total_mbas : ℕ := 9

/-- The number of committees to be formed -/
def num_committees : ℕ := 3

/-- The number of members in each committee -/
def committee_size : ℕ := 4

/-- The probability that Jane and Albert are on the same committee -/
def probability_same_committee : ℚ := 1 / 6

theorem jane_albert_same_committee :
  let total_ways := (total_mbas.choose committee_size) * ((total_mbas - committee_size).choose committee_size)
  let ways_together := ((total_mbas - 2).choose (committee_size - 2)) * ((total_mbas - committee_size).choose committee_size)
  (ways_together : ℚ) / total_ways = probability_same_committee :=
sorry

end jane_albert_same_committee_l938_93829


namespace nested_fraction_equality_l938_93893

theorem nested_fraction_equality : 
  1 / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 3 / 4 := by
  sorry

end nested_fraction_equality_l938_93893


namespace non_tax_paying_percentage_is_six_percent_l938_93899

/-- The number of customers shopping per day -/
def customers_per_day : ℕ := 1000

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of customers who pay taxes per week -/
def tax_paying_customers_per_week : ℕ := 6580

/-- The percentage of customers who do not pay tax -/
def non_tax_paying_percentage : ℚ :=
  (customers_per_day * days_per_week - tax_paying_customers_per_week : ℚ) /
  (customers_per_day * days_per_week : ℚ) * 100

theorem non_tax_paying_percentage_is_six_percent :
  non_tax_paying_percentage = 6 := by
  sorry

end non_tax_paying_percentage_is_six_percent_l938_93899


namespace min_cut_area_for_given_board_l938_93895

/-- Represents a rectangular board with a damaged corner -/
structure Board :=
  (length : ℝ)
  (width : ℝ)
  (damaged_length : ℝ)
  (damaged_width : ℝ)

/-- Calculates the minimum area that needs to be cut off -/
def min_cut_area (b : Board) : ℝ :=
  2 + b.damaged_length * b.damaged_width

/-- Theorem stating the minimum area to be cut off for the given board -/
theorem min_cut_area_for_given_board :
  let b : Board := ⟨7, 5, 2, 1⟩
  min_cut_area b = 4 := by sorry

end min_cut_area_for_given_board_l938_93895


namespace unique_solution_for_abc_l938_93889

theorem unique_solution_for_abc (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a - 1)^2 / (b + c + 1) + (b + 1)^2 / (c + a - 1) + (c + 5)^2 / (a + b - 5) = 49) :
  a = 10.5 ∧ b = 10 ∧ c = 0.5 := by
  sorry

end unique_solution_for_abc_l938_93889


namespace arman_two_week_earnings_l938_93882

/-- Calculates Arman's earnings for two weeks given his work hours and rates --/
theorem arman_two_week_earnings : 
  let first_week_hours : ℕ := 35
  let first_week_rate : ℚ := 10
  let second_week_hours : ℕ := 40
  let second_week_rate_increase : ℚ := 0.5
  let second_week_rate : ℚ := first_week_rate + second_week_rate_increase
  let first_week_earnings : ℚ := first_week_hours * first_week_rate
  let second_week_earnings : ℚ := second_week_hours * second_week_rate
  let total_earnings : ℚ := first_week_earnings + second_week_earnings
  total_earnings = 770 := by sorry

end arman_two_week_earnings_l938_93882


namespace hyperbola_minimum_value_l938_93837

theorem hyperbola_minimum_value (x y : ℝ) :
  x^2 / 4 - y^2 = 1 →
  3 * x^2 - 2 * x * y ≥ 6 + 4 * Real.sqrt 2 :=
by sorry

end hyperbola_minimum_value_l938_93837


namespace quadratic_roots_less_than_one_l938_93859

theorem quadratic_roots_less_than_one (a b : ℝ) 
  (h1 : abs a + abs b < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → abs x < 1 := by
  sorry

end quadratic_roots_less_than_one_l938_93859


namespace parallel_line_divides_equally_l938_93812

-- Define the shaded area
def shaded_area : ℝ := 10

-- Define the distance from MO to the parallel line
def distance_from_MO : ℝ := 2.6

-- Define the function that calculates the area above the parallel line
def area_above (d : ℝ) : ℝ := sorry

-- Theorem statement
theorem parallel_line_divides_equally :
  area_above distance_from_MO = shaded_area / 2 := by sorry

end parallel_line_divides_equally_l938_93812


namespace sum_of_possible_x_values_l938_93827

theorem sum_of_possible_x_values (x z : ℝ) 
  (h1 : (x - z)^2 = 100) 
  (h2 : (z - 12)^2 = 36) : 
  ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 - z)^2 = 100 ∧ 
    (x2 - z)^2 = 100 ∧ 
    (x3 - z)^2 = 100 ∧ 
    (x4 - z)^2 = 100 ∧ 
    (z - 12)^2 = 36 ∧
    x1 + x2 + x3 + x4 = 48 :=
by sorry

end sum_of_possible_x_values_l938_93827


namespace coefficient_x_squared_in_binomial_expansion_l938_93869

theorem coefficient_x_squared_in_binomial_expansion :
  let binomial := (X - 1 / X : Polynomial ℚ)^6
  (binomial.coeff 2) = 15 := by sorry

end coefficient_x_squared_in_binomial_expansion_l938_93869


namespace susan_work_hours_l938_93866

/-- Susan's work problem -/
theorem susan_work_hours 
  (summer_weeks : ℕ) 
  (summer_hours_per_week : ℕ) 
  (summer_earnings : ℕ) 
  (school_weeks : ℕ) 
  (school_earnings : ℕ) 
  (h1 : summer_weeks = 10)
  (h2 : summer_hours_per_week = 60)
  (h3 : summer_earnings = 6000)
  (h4 : school_weeks = 50)
  (h5 : school_earnings = 6000) :
  ∃ (school_hours_per_week : ℕ),
    (summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week : ℚ) * 
    (school_weeks * school_hours_per_week : ℚ) = school_earnings ∧
    school_hours_per_week = 12 :=
by sorry

end susan_work_hours_l938_93866


namespace books_bought_from_first_shop_l938_93805

theorem books_bought_from_first_shop
  (total_first_shop : ℕ)
  (books_second_shop : ℕ)
  (total_second_shop : ℕ)
  (average_price : ℕ)
  (h1 : total_first_shop = 600)
  (h2 : books_second_shop = 20)
  (h3 : total_second_shop = 240)
  (h4 : average_price = 14)
  : ∃ (books_first_shop : ℕ),
    (total_first_shop + total_second_shop) / (books_first_shop + books_second_shop) = average_price ∧
    books_first_shop = 40 :=
by sorry

end books_bought_from_first_shop_l938_93805


namespace web_pages_scientific_notation_l938_93885

/-- The number of web pages found when searching for "Mount Fanjing" in "Sogou" -/
def web_pages : ℕ := 1630000

/-- The scientific notation representation of the number of web pages -/
def scientific_notation : ℝ := 1.63 * (10 : ℝ) ^ 6

/-- Theorem stating that the number of web pages is equal to its scientific notation representation -/
theorem web_pages_scientific_notation : (web_pages : ℝ) = scientific_notation := by
  sorry

end web_pages_scientific_notation_l938_93885


namespace sum_of_repeating_decimals_l938_93896

theorem sum_of_repeating_decimals : 
  (2 : ℚ) / 9 + (2 : ℚ) / 99 + (2 : ℚ) / 9999 = 224422 / 9999 := by
  sorry

end sum_of_repeating_decimals_l938_93896


namespace ten_cuts_eleven_pieces_l938_93836

/-- The number of pieces resulting from cutting a log -/
def num_pieces (cuts : ℕ) : ℕ := cuts + 1

/-- Theorem: 10 cuts on a log result in 11 pieces -/
theorem ten_cuts_eleven_pieces : num_pieces 10 = 11 := by
  sorry

end ten_cuts_eleven_pieces_l938_93836


namespace m_range_l938_93867

-- Define proposition p
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

-- Define proposition q
def q (x m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0

-- Define the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  m ≤ 3 :=
sorry

end m_range_l938_93867


namespace family_admission_price_l938_93822

/-- The total price for a family's admission to an amusement park --/
def total_price (adult_price child_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

/-- Theorem: The total price for a family of 2 adults and 2 children,
    with adult tickets costing $22 and child tickets costing $7, is $58 --/
theorem family_admission_price :
  total_price 22 7 2 2 = 58 := by
  sorry

end family_admission_price_l938_93822


namespace wrong_to_right_ratio_l938_93831

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) (h1 : total = 24) (h2 : correct = 8) :
  (total - correct) / correct = 2 := by
  sorry

end wrong_to_right_ratio_l938_93831


namespace sum_a_d_equals_five_l938_93800

theorem sum_a_d_equals_five 
  (a b c d : ℤ) 
  (eq1 : a + b = 11) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 5 := by
sorry

end sum_a_d_equals_five_l938_93800


namespace tomato_types_salad_bar_problem_l938_93806

theorem tomato_types (lettuce_types : Nat) (olive_types : Nat) (soup_types : Nat) 
  (total_options : Nat) : Nat :=
  let tomato_types := total_options / (lettuce_types * olive_types * soup_types)
  tomato_types

theorem salad_bar_problem :
  let lettuce_types := 2
  let olive_types := 4
  let soup_types := 2
  let total_options := 48
  tomato_types lettuce_types olive_types soup_types total_options = 3 := by
  sorry

end tomato_types_salad_bar_problem_l938_93806


namespace combined_fuel_efficiency_l938_93870

theorem combined_fuel_efficiency (d : ℝ) (h : d > 0) :
  let efficiency1 : ℝ := 50
  let efficiency2 : ℝ := 20
  let efficiency3 : ℝ := 15
  let total_distance : ℝ := 3 * d
  let total_fuel : ℝ := d / efficiency1 + d / efficiency2 + d / efficiency3
  total_distance / total_fuel = 900 / 41 :=
by sorry

end combined_fuel_efficiency_l938_93870


namespace cylinder_minimal_material_l938_93852

/-- For a cylindrical beverage can with a fixed volume, the material used is minimized when the base radius is half the height -/
theorem cylinder_minimal_material (V : ℝ) (h R : ℝ) (h_pos : h > 0) (R_pos : R > 0) :
  V = π * R^2 * h → (∀ R' h', V = π * R'^2 * h' → 2 * π * R^2 + 2 * π * R * h ≤ 2 * π * R'^2 + 2 * π * R' * h') ↔ R = h / 2 := by
  sorry

end cylinder_minimal_material_l938_93852


namespace fred_took_233_marbles_l938_93802

/-- The number of black marbles Fred took from Sara -/
def marbles_taken (initial_black_marbles remaining_black_marbles : ℕ) : ℕ :=
  initial_black_marbles - remaining_black_marbles

/-- Proof that Fred took 233 black marbles from Sara -/
theorem fred_took_233_marbles :
  marbles_taken 792 559 = 233 := by
  sorry

end fred_took_233_marbles_l938_93802


namespace percent_less_than_l938_93874

theorem percent_less_than (N M : ℝ) (h1 : 0 < N) (h2 : 0 < M) (h3 : N < M) :
  (M - N) / M * 100 = 100 * (1 - N / M) := by
  sorry

end percent_less_than_l938_93874


namespace overlapping_semicircles_area_l938_93855

/-- Given a pattern of overlapping semicircles, this theorem calculates the shaded area. -/
theorem overlapping_semicircles_area (diameter : ℝ) (overlap : ℝ) (total_length : ℝ) : 
  diameter = 3 ∧ overlap = 0.5 ∧ total_length = 12 →
  (∃ (shaded_area : ℝ), shaded_area = 5.625 * Real.pi) := by
  sorry

#check overlapping_semicircles_area

end overlapping_semicircles_area_l938_93855


namespace michaels_weight_loss_l938_93892

/-- Michael's weight loss problem -/
theorem michaels_weight_loss 
  (total_goal : ℝ) 
  (april_loss : ℝ) 
  (may_goal : ℝ) 
  (h1 : total_goal = 10) 
  (h2 : april_loss = 4) 
  (h3 : may_goal = 3) : 
  total_goal - (april_loss + may_goal) = 3 := by
sorry

end michaels_weight_loss_l938_93892


namespace prob_two_non_defective_pens_l938_93873

/-- The probability of selecting two non-defective pens without replacement from a box of 12 pens, where 3 are defective, is 6/11. -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) 
  (h1 : total_pens = 12) (h2 : defective_pens = 3) : 
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 6 / 11 := by
  sorry

#check prob_two_non_defective_pens

end prob_two_non_defective_pens_l938_93873


namespace otimes_composition_l938_93830

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y

-- State the theorem
theorem otimes_composition (h : ℝ) : otimes h (otimes h h) = 2 * h^2 + h := by
  sorry

end otimes_composition_l938_93830


namespace cube_increase_correct_l938_93820

/-- Represents the percentage increase in a cube's dimensions and properties -/
structure CubeIncrease where
  edge : ℝ
  surface_area : ℝ
  volume : ℝ

/-- The percentage increases when a cube's edge is increased by 60% -/
def cube_increase : CubeIncrease :=
  { edge := 60
  , surface_area := 156
  , volume := 309.6 }

theorem cube_increase_correct :
  let original_edge := 1
  let new_edge := original_edge * (1 + cube_increase.edge / 100)
  let original_surface_area := 6 * original_edge^2
  let new_surface_area := 6 * new_edge^2
  let original_volume := original_edge^3
  let new_volume := new_edge^3
  (new_surface_area / original_surface_area - 1) * 100 = cube_increase.surface_area ∧
  (new_volume / original_volume - 1) * 100 = cube_increase.volume :=
by sorry

end cube_increase_correct_l938_93820


namespace equal_cost_mileage_l938_93811

/-- Represents the cost function for a truck rental company -/
structure RentalCompany where
  baseCost : ℝ
  costPerMile : ℝ

/-- Calculates the total cost for a given mileage -/
def totalCost (company : RentalCompany) (miles : ℝ) : ℝ :=
  company.baseCost + company.costPerMile * miles

/-- Theorem: The mileage at which all three companies have the same cost is 150 miles, 
    and this common cost is $85.45 -/
theorem equal_cost_mileage 
  (safety : RentalCompany)
  (city : RentalCompany)
  (metro : RentalCompany)
  (h1 : safety.baseCost = 41.95 ∧ safety.costPerMile = 0.29)
  (h2 : city.baseCost = 38.95 ∧ city.costPerMile = 0.31)
  (h3 : metro.baseCost = 44.95 ∧ metro.costPerMile = 0.27) :
  ∃ (m : ℝ), 
    m = 150 ∧ 
    totalCost safety m = totalCost city m ∧
    totalCost city m = totalCost metro m ∧
    totalCost safety m = 85.45 :=
by sorry

end equal_cost_mileage_l938_93811


namespace function_properties_l938_93878

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem function_properties :
  (∃ a > 0, ∀ x > 0, f a x ≥ 0) ∧
  (∃ a > 0, ∃ x > 0, f a x ≤ 0) ∧
  (∀ a > 0, ∃ x > 0, f a x ≤ 0) :=
by sorry

end function_properties_l938_93878


namespace sum_of_integers_l938_93826

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
sorry

end sum_of_integers_l938_93826


namespace james_sodas_per_day_l938_93898

/-- Calculates the number of sodas James drinks per day given the following conditions:
  * James buys 5 packs of sodas
  * Each pack contains 12 sodas
  * James already had 10 sodas
  * He finishes all the sodas in 1 week
-/
def sodas_per_day (packs : ℕ) (sodas_per_pack : ℕ) (initial_sodas : ℕ) (days_in_week : ℕ) : ℕ :=
  ((packs * sodas_per_pack + initial_sodas) / days_in_week)

theorem james_sodas_per_day :
  sodas_per_day 5 12 10 7 = 10 := by
  sorry

end james_sodas_per_day_l938_93898


namespace extended_annuity_duration_l938_93825

/-- Calculates the number of years an annuity will last given initial conditions and a delay --/
def calculate_extended_annuity_years (initial_rate : ℝ) (initial_years : ℕ) (annual_payment : ℝ) (delay_years : ℕ) : ℕ :=
  sorry

/-- Theorem stating that under given conditions, the annuity will last for 34 years --/
theorem extended_annuity_duration :
  let initial_rate : ℝ := 0.045
  let initial_years : ℕ := 26
  let annual_payment : ℝ := 5000
  let delay_years : ℕ := 3
  calculate_extended_annuity_years initial_rate initial_years annual_payment delay_years = 34 :=
by sorry

end extended_annuity_duration_l938_93825


namespace apples_per_crate_value_l938_93875

/-- The number of apples in each crate -/
def apples_per_crate : ℕ := sorry

/-- The total number of crates -/
def total_crates : ℕ := 12

/-- The number of rotten apples -/
def rotten_apples : ℕ := 160

/-- The number of boxes filled with good apples -/
def filled_boxes : ℕ := 100

/-- The number of apples in each box -/
def apples_per_box : ℕ := 20

theorem apples_per_crate_value : apples_per_crate = 180 := by sorry

end apples_per_crate_value_l938_93875


namespace complex_modulus_problem_l938_93839

theorem complex_modulus_problem (z : ℂ) : z = (2 * Complex.I) / (1 - Complex.I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l938_93839


namespace subset_implies_a_leq_one_l938_93816

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- State the theorem
theorem subset_implies_a_leq_one (a : ℝ) : A ⊆ B a → a ≤ 1 := by
  sorry

end subset_implies_a_leq_one_l938_93816


namespace max_red_socks_l938_93860

theorem max_red_socks (r b : ℕ) : 
  let t := r + b
  r + b ≤ 2000 →
  (r * (r - 1) + b * (b - 1)) / (t * (t - 1)) = 5 / 12 →
  r ≤ 109 :=
by sorry

end max_red_socks_l938_93860


namespace nap_time_is_three_hours_l938_93807

-- Define flight duration in minutes
def flight_duration : ℕ := 11 * 60 + 20

-- Define durations of activities in minutes
def reading_time : ℕ := 2 * 60
def movie_time : ℕ := 4 * 60
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 1 * 60 + 10

-- Define total activity time
def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Define nap time in hours
def nap_time_hours : ℕ := (flight_duration - total_activity_time) / 60

-- Theorem statement
theorem nap_time_is_three_hours : nap_time_hours = 3 := by
  sorry

end nap_time_is_three_hours_l938_93807


namespace function_value_symmetry_l938_93801

/-- Given a function f(x) = ax^5 - bx^3 + cx where a, b, c are real numbers,
    if f(-3) = 7, then f(3) = -7 -/
theorem function_value_symmetry (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 - b * x^3 + c * x)
    (h2 : f (-3) = 7) : 
  f 3 = -7 := by
  sorry

end function_value_symmetry_l938_93801


namespace inequality_solution_implies_m_value_l938_93841

-- Define the inequality
def inequality (x m : ℝ) : Prop :=
  (x + 2) / 2 ≥ (2 * x + m) / 3 + 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := x ≤ 8

-- Theorem statement
theorem inequality_solution_implies_m_value :
  (∀ x, inequality x m ↔ solution_set x) → 2^m = (1 : ℝ) / 16 := by
  sorry

end inequality_solution_implies_m_value_l938_93841


namespace mark_election_votes_l938_93835

theorem mark_election_votes (first_area_voters : ℕ) (first_area_percentage : ℚ) :
  first_area_voters = 100000 →
  first_area_percentage = 70 / 100 →
  (first_area_voters * first_area_percentage).floor +
  2 * (first_area_voters * first_area_percentage).floor = 210000 :=
by sorry

end mark_election_votes_l938_93835


namespace cocktail_cost_per_litre_l938_93853

/-- Calculate the cost per litre of a superfruit juice cocktail --/
theorem cocktail_cost_per_litre 
  (mixed_fruit_cost : ℝ) 
  (acai_cost : ℝ) 
  (mixed_fruit_volume : ℝ) 
  (acai_volume : ℝ) 
  (h1 : mixed_fruit_cost = 262.85)
  (h2 : acai_cost = 3104.35)
  (h3 : mixed_fruit_volume = 32)
  (h4 : acai_volume = 21.333333333333332) : 
  ∃ (cost_per_litre : ℝ), abs (cost_per_litre - 1399.99) < 0.01 := by
  sorry

end cocktail_cost_per_litre_l938_93853


namespace quadratic_equation_unique_solution_l938_93894

theorem quadratic_equation_unique_solution (a : ℝ) : 
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2*a*x^2 - x - 1 = 0) → a > 1 := by
  sorry

end quadratic_equation_unique_solution_l938_93894


namespace arithmetic_sequence_ratio_l938_93819

/-- Given an arithmetic sequence {a_n} with non-zero common difference d,
    if a_2 + a_3 = a_6, then (a_1 + a_2) / (a_3 + a_4 + a_5) = 1/3 -/
theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 2 + a 3 = a 6) :
  (a 1 + a 2) / (a 3 + a 4 + a 5) = 1 / 3 := by
  sorry

end arithmetic_sequence_ratio_l938_93819


namespace melodys_dogs_eating_frequency_l938_93871

/-- Proves that each dog eats twice a day given the conditions of Melody's dog food problem -/
theorem melodys_dogs_eating_frequency :
  let num_dogs : ℕ := 3
  let food_per_meal : ℚ := 1/2
  let initial_food : ℕ := 30
  let remaining_food : ℕ := 9
  let days_in_week : ℕ := 7
  
  let total_food_eaten : ℕ := initial_food - remaining_food
  let food_per_day : ℚ := (total_food_eaten : ℚ) / days_in_week
  let meals_per_day : ℚ := food_per_day / (num_dogs * food_per_meal)
  
  meals_per_day = 2 := by sorry

end melodys_dogs_eating_frequency_l938_93871


namespace parallelogram_xy_product_l938_93823

-- Define the parallelogram EFGH
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ

-- Define the theorem
theorem parallelogram_xy_product 
  (EFGH : Parallelogram) 
  (h1 : EFGH.EF = 58) 
  (h2 : ∃ y, EFGH.FG = 4 * y^3) 
  (h3 : ∃ x, EFGH.GH = 3 * x + 5) 
  (h4 : EFGH.HE = 24) :
  ∃ x y, x * y = (53 * Real.rpow 6 (1/3)) / 3 := by
  sorry

end parallelogram_xy_product_l938_93823


namespace rectangle_area_scientific_notation_l938_93863

theorem rectangle_area_scientific_notation :
  let side1 : ℝ := 3 * 10^3
  let side2 : ℝ := 400
  let area : ℝ := side1 * side2
  area = 1.2 * 10^6 := by sorry

end rectangle_area_scientific_notation_l938_93863


namespace complex_power_to_rectangular_l938_93844

theorem complex_power_to_rectangular : 
  (3 * Complex.cos (Real.pi / 6) + 3 * Complex.I * Complex.sin (Real.pi / 6)) ^ 4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by
  sorry

end complex_power_to_rectangular_l938_93844


namespace quarter_count_l938_93833

/-- Given a sum of $3.35 consisting of quarters and dimes, with a total of 23 coins, 
    prove that the number of quarters is 7. -/
theorem quarter_count (total_value : ℚ) (total_coins : ℕ) (quarter_value dime_value : ℚ) 
  (h1 : total_value = 335/100)
  (h2 : total_coins = 23)
  (h3 : quarter_value = 25/100)
  (h4 : dime_value = 1/10)
  : ∃ (quarters dimes : ℕ), 
    quarters + dimes = total_coins ∧ 
    quarters * quarter_value + dimes * dime_value = total_value ∧
    quarters = 7 :=
by sorry

end quarter_count_l938_93833


namespace tangent_lines_theorem_l938_93891

/-- The function f(x) = x³ + x - 16 -/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_lines_theorem :
  /- Tangent lines with slope 4 -/
  (∃ x₀ y₀ : ℝ, f x₀ = y₀ ∧ f' x₀ = 4 ∧ (4*x₀ - y₀ - 18 = 0 ∨ 4*x₀ - y₀ - 14 = 0)) ∧
  /- Tangent line at point (2, -6) -/
  (f 2 = -6 ∧ f' 2 = 13 ∧ 13*2 - (-6) - 32 = 0) ∧
  /- Tangent line passing through origin -/
  (∃ x₀ : ℝ, f x₀ = f' x₀ * (-x₀) ∧ f' x₀ = 13) :=
by sorry

end tangent_lines_theorem_l938_93891


namespace sequence_inequality_l938_93834

theorem sequence_inequality (a : ℕ → ℝ) (k : ℕ) (h1 : ∀ n, a n > 0) 
  (h2 : k > 0) (h3 : ∀ n, a (n + 1) ≤ (a n)^k * (1 - a n)) :
  ∀ n ≥ 2, (1 / a n) ≥ ((k + 1 : ℝ)^(k + 1) / k^k) + (n - 2) :=
by sorry

end sequence_inequality_l938_93834


namespace trigonometric_expression_equals_one_l938_93848

theorem trigonometric_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (100 * π / 180)) / 
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) + 
   Real.cos (156 * π / 180) * Real.cos (96 * π / 180)) = 1 := by
sorry

end trigonometric_expression_equals_one_l938_93848


namespace exists_multiple_factorizations_l938_93821

-- Define the set V
def V (p : Nat) : Set Nat :=
  {n : Nat | ∃ k : Nat, (n = k * p + 1 ∨ n = k * p - 1) ∧ k > 0}

-- Define indecomposability in V
def isIndecomposable (p : Nat) (n : Nat) : Prop :=
  n ∈ V p ∧ ∀ k l : Nat, k ∈ V p → l ∈ V p → n ≠ k * l

-- Theorem statement
theorem exists_multiple_factorizations (p : Nat) (h : p > 5) :
  ∃ N : Nat, N ∈ V p ∧
    ∃ (factors1 factors2 : List Nat),
      factors1 ≠ factors2 ∧
      (∀ f ∈ factors1, isIndecomposable p f) ∧
      (∀ f ∈ factors2, isIndecomposable p f) ∧
      N = factors1.prod ∧
      N = factors2.prod :=
by sorry

end exists_multiple_factorizations_l938_93821


namespace linear_system_solution_l938_93809

theorem linear_system_solution (x y a : ℝ) : 
  (3 * x + y = a) → 
  (x - 2 * y = 1) → 
  (2 * x + 3 * y = 2) → 
  (a = 3) := by
sorry

end linear_system_solution_l938_93809


namespace bridge_brick_ratio_l938_93814

theorem bridge_brick_ratio (total_bricks : ℕ) (type_a_bricks : ℕ) (other_bricks : ℕ) : 
  total_bricks = 150 →
  type_a_bricks = 40 →
  other_bricks = 90 →
  ∃ (type_b_bricks : ℕ), 
    type_a_bricks + type_b_bricks + other_bricks = total_bricks ∧
    type_b_bricks * 2 = type_a_bricks :=
by
  sorry

end bridge_brick_ratio_l938_93814


namespace brad_reads_more_than_greg_l938_93842

/-- Greg's daily reading pages -/
def greg_pages : ℕ := 18

/-- Brad's daily reading pages -/
def brad_pages : ℕ := 26

/-- The difference in pages read between Brad and Greg -/
def page_difference : ℕ := brad_pages - greg_pages

theorem brad_reads_more_than_greg : page_difference = 8 := by
  sorry

end brad_reads_more_than_greg_l938_93842


namespace elephant_distribution_l938_93876

theorem elephant_distribution (union_members non_union_members : ℕ) 
  (h1 : union_members = 28)
  (h2 : non_union_members = 37) :
  let total_elephants := 2072
  let elephants_per_union := total_elephants / union_members
  let elephants_per_non_union := total_elephants / non_union_members
  (elephants_per_union * union_members = elephants_per_non_union * non_union_members) ∧
  (elephants_per_union ≥ 1) ∧
  (elephants_per_non_union ≥ 1) ∧
  (∀ n : ℕ, n > total_elephants → 
    ¬(n / union_members * union_members = n / non_union_members * non_union_members ∧
      n / union_members ≥ 1 ∧
      n / non_union_members ≥ 1)) :=
by sorry

end elephant_distribution_l938_93876


namespace complex_number_problem_l938_93832

theorem complex_number_problem (z : ℂ) (m n : ℝ) :
  (z.re > 0) →
  (Complex.abs z = 2 * Real.sqrt 5) →
  ((1 + 2 * Complex.I) * z).re = 0 →
  (z ^ 2 + m * z + n = 0) →
  (z = 4 + 2 * Complex.I ∧ m = -8 ∧ n = 20) :=
by sorry

end complex_number_problem_l938_93832


namespace sequence_problem_l938_93813

/-- Given a sequence {aₙ} where a₂ = 3, a₄ = 15, and {aₙ₊₁} is a geometric sequence, prove that a₆ = 63. -/
theorem sequence_problem (a : ℕ → ℝ) 
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) + 1 = (a n + 1) * q) :
  a 6 = 63 := by
  sorry

end sequence_problem_l938_93813


namespace sin_2theta_value_l938_93850

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/5) : 
  Real.sin (2 * θ) = -24/25 := by
  sorry

end sin_2theta_value_l938_93850


namespace no_common_solution_l938_93838

theorem no_common_solution : ¬∃ y : ℝ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0) := by
  sorry

end no_common_solution_l938_93838


namespace intersecting_line_theorem_l938_93840

/-- Given points A and B, and a line y = ax intersecting segment AB at point C,
    prove that if AC = 2CB, then a = 1 -/
theorem intersecting_line_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, 4)
  ∃ (C : ℝ × ℝ), 
    (C.2 = a * C.1) ∧  -- C is on the line y = ax
    (∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = (1 - t) • A + t • B) ∧  -- C is on segment AB
    ((C.1 - A.1, C.2 - A.2) = (2 * (B.1 - C.1), 2 * (B.2 - C.2)))  -- AC = 2CB
    → a = 1 := by
  sorry

end intersecting_line_theorem_l938_93840


namespace systematic_sampling_problem_l938_93846

/-- Systematic sampling selection function -/
def systematicSample (initialSelection : ℕ) (interval : ℕ) (groupNumber : ℕ) : ℕ :=
  initialSelection + interval * (groupNumber - 1)

/-- Theorem for the systematic sampling problem -/
theorem systematic_sampling_problem (totalStudents : ℕ) (sampleSize : ℕ) (interval : ℕ) 
    (initialSelection : ℕ) (targetGroupStart : ℕ) (targetGroupEnd : ℕ) :
    totalStudents = 800 →
    sampleSize = 50 →
    interval = 16 →
    initialSelection = 7 →
    targetGroupStart = 65 →
    targetGroupEnd = 80 →
    systematicSample initialSelection interval 
      ((targetGroupStart - 1) / interval + 1) = 71 := by
  sorry

end systematic_sampling_problem_l938_93846


namespace extreme_values_imply_a_range_l938_93886

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*x + a * Real.log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - 2 + a / x

/-- Theorem stating that if f(x) has two distinct extreme values, then 0 < a < 1 -/
theorem extreme_values_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) →
  0 < a ∧ a < 1 := by sorry

end extreme_values_imply_a_range_l938_93886


namespace sum_of_15th_set_l938_93888

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ :=
  1 + (n * (n - 1)) / 2

/-- The number of elements in the nth set -/
def set_size (n : ℕ) : ℕ := n

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ :=
  let a := first_element n
  let l := a + set_size n - 1
  (set_size n * (a + l)) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end sum_of_15th_set_l938_93888


namespace age_problem_l938_93804

theorem age_problem (a b : ℕ) (h1 : a - 10 = (b - 10) / 2) (h2 : 4 * a = 3 * b) :
  a + b = 35 := by sorry

end age_problem_l938_93804
