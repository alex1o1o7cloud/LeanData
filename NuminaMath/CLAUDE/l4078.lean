import Mathlib

namespace NUMINAMATH_CALUDE_larger_integer_value_l4078_407809

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (b : ℚ) / (a : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  (b : ℕ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l4078_407809


namespace NUMINAMATH_CALUDE_group_size_proof_l4078_407871

theorem group_size_proof (total_paise : ℕ) (n : ℕ) : 
  total_paise = 7744 →
  n * n = total_paise →
  n = 88 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l4078_407871


namespace NUMINAMATH_CALUDE_product_ab_l4078_407836

theorem product_ab (a b : ℝ) (h1 : a - b = 2) (h2 : a^2 + b^2 = 25) : a * b = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l4078_407836


namespace NUMINAMATH_CALUDE_equation_solutions_l4078_407878

theorem equation_solutions : 
  ∀ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ↔ 
  ((m = 5 ∧ n = -3) ∨ (m = -5 ∧ n = -3) ∨ (m = 5 ∧ n = 3) ∨ (m = -5 ∧ n = 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4078_407878


namespace NUMINAMATH_CALUDE_exponential_equation_and_inequality_l4078_407818

variables (a x : ℝ)

theorem exponential_equation_and_inequality 
  (h1 : a > 0) (h2 : a ≠ 1) :
  (a^(3*x + 5) = a^(-2*x) → x = -1) ∧
  (a^(3*x + 5) > a^(-2*x) → 
    ((a > 1 → x > -1) ∧ 
     (0 < a ∧ a < 1 → x < -1))) :=
by sorry

end NUMINAMATH_CALUDE_exponential_equation_and_inequality_l4078_407818


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l4078_407888

theorem intersection_points_form_line : 
  ∀ (s : ℝ), 
  ∃ (x y : ℝ), 
  (x + 3 * y = 10 * s + 4) ∧ 
  (2 * x - y = 3 * s - 5) → 
  y = (119 / 133) * x + (435 / 133) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l4078_407888


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_ratio_l4078_407814

/-- The ratio of volumes of two brands of vegetable ghee in a mixture -/
theorem vegetable_ghee_mixture_ratio :
  ∀ (Va Vb : ℝ),
  Va + Vb = 4 →
  900 * Va + 850 * Vb = 3520 →
  Va / Vb = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_ghee_mixture_ratio_l4078_407814


namespace NUMINAMATH_CALUDE_rhombus_area_l4078_407894

-- Define the vertices of the rhombus
def v1 : ℝ × ℝ := (1.2, 4.1)
def v2 : ℝ × ℝ := (7.3, 2.5)
def v3 : ℝ × ℝ := (1.2, -2.8)
def v4 : ℝ × ℝ := (-4.9, 2.5)

-- Define the vectors representing two adjacent sides of the rhombus
def vector1 : ℝ × ℝ := (v2.1 - v1.1, v2.2 - v1.2)
def vector2 : ℝ × ℝ := (v4.1 - v1.1, v4.2 - v1.2)

-- Theorem stating that the area of the rhombus is 19.52 square units
theorem rhombus_area : 
  abs ((vector1.1 * vector2.2) - (vector1.2 * vector2.1)) = 19.52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4078_407894


namespace NUMINAMATH_CALUDE_hyperbola_t_range_l4078_407886

-- Define the curve C
def curve_C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

-- Define what it means for a curve to be a hyperbola
def is_hyperbola (C : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem hyperbola_t_range (t : ℝ) :
  is_hyperbola (curve_C t) → t < 1 ∨ t > 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_t_range_l4078_407886


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l4078_407803

/-- An equilateral hexagon with specific properties -/
structure EquilateralHexagon where
  -- Side length of the hexagon
  side : ℝ
  -- Assertion that three nonadjacent interior angles are 60°
  angle_property : True
  -- The area of the hexagon is 9√3
  area_eq : side^2 * Real.sqrt 3 = 9 * Real.sqrt 3

/-- The perimeter of an equilateral hexagon is 18 given the specified conditions -/
theorem hexagon_perimeter (h : EquilateralHexagon) : h.side * 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l4078_407803


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4078_407831

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (3 * x + 5) / Real.sqrt (6 * x + 5) = Real.sqrt 5 / 3) → x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4078_407831


namespace NUMINAMATH_CALUDE_negative_division_equals_positive_division_negative_three_hundred_by_negative_twenty_five_l4078_407893

theorem negative_division_equals_positive (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (-x) / (-y) = x / y :=
sorry

theorem division_negative_three_hundred_by_negative_twenty_five :
  (-300) / (-25) = 12 :=
sorry

end NUMINAMATH_CALUDE_negative_division_equals_positive_division_negative_three_hundred_by_negative_twenty_five_l4078_407893


namespace NUMINAMATH_CALUDE_circle_inscribed_angles_sum_l4078_407896

theorem circle_inscribed_angles_sum (n : ℕ) (x y : ℝ) : 
  n = 16 → 
  x = 3 * (360 / n) / 2 → 
  y = 5 * (360 / n) / 2 → 
  x + y = 90 := by
sorry

end NUMINAMATH_CALUDE_circle_inscribed_angles_sum_l4078_407896


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l4078_407898

noncomputable def f (x : ℝ) := x * Real.exp x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    y = m * x + b ↔ 
    (∃ (h : ℝ → ℝ), (∀ t, t ≠ 1 → (h t - f 1) / (t - 1) = (f t - f 1) / (t - 1)) ∧
                     (h 1 = f 1) ∧
                     y = h x) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l4078_407898


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l4078_407812

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  (∀ p : ℕ, Nat.Prime p → p ≤ 31 → ¬(p ∣ n)) →
  (∀ p : ℕ, Nat.Prime p → p < Real.sqrt (n : ℝ) → ¬(p ∣ n)) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l4078_407812


namespace NUMINAMATH_CALUDE_concert_revenue_calculation_l4078_407816

/-- Calculates the total revenue from concert ticket sales given specific conditions --/
theorem concert_revenue_calculation (ticket_price : ℝ) 
  (first_ten_discount : ℝ) (next_twenty_discount : ℝ)
  (military_discount : ℝ) (student_discount : ℝ) (senior_discount : ℝ)
  (total_buyers : ℕ) (military_buyers : ℕ) (student_buyers : ℕ) (senior_buyers : ℕ) :
  ticket_price = 20 →
  first_ten_discount = 0.4 →
  next_twenty_discount = 0.15 →
  military_discount = 0.25 →
  student_discount = 0.2 →
  senior_discount = 0.1 →
  total_buyers = 85 →
  military_buyers = 8 →
  student_buyers = 12 →
  senior_buyers = 9 →
  (10 * (ticket_price * (1 - first_ten_discount)) +
   20 * (ticket_price * (1 - next_twenty_discount)) +
   military_buyers * (ticket_price * (1 - military_discount)) +
   student_buyers * (ticket_price * (1 - student_discount)) +
   senior_buyers * (ticket_price * (1 - senior_discount)) +
   (total_buyers - (10 + 20 + military_buyers + student_buyers + senior_buyers)) * ticket_price) = 1454 := by
  sorry


end NUMINAMATH_CALUDE_concert_revenue_calculation_l4078_407816


namespace NUMINAMATH_CALUDE_hyperbolic_and_linear_functions_l4078_407813

/-- The hyperbolic and linear functions with their properties -/
theorem hyperbolic_and_linear_functions (k : ℝ) (h : |k| < 1) :
  (∀ x y : ℝ, y = (k - 1) / x → (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) ∧
  (k * (-1 + 1) = 0) := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_and_linear_functions_l4078_407813


namespace NUMINAMATH_CALUDE_final_price_calculation_l4078_407825

def original_price : ℝ := 10.00
def increase_percent : ℝ := 0.40
def decrease_percent : ℝ := 0.30

def price_after_increase (p : ℝ) (i : ℝ) : ℝ := p * (1 + i)
def price_after_decrease (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

theorem final_price_calculation : 
  price_after_decrease (price_after_increase original_price increase_percent) decrease_percent = 9.80 := by
  sorry

end NUMINAMATH_CALUDE_final_price_calculation_l4078_407825


namespace NUMINAMATH_CALUDE_star_op_example_l4078_407879

/-- Custom binary operation ☼ defined for rational numbers -/
def star_op (a b : ℚ) : ℚ := a^3 - 2*a*b + 4

/-- Theorem stating that 4 ☼ (-9) = 140 -/
theorem star_op_example : star_op 4 (-9) = 140 := by
  sorry

end NUMINAMATH_CALUDE_star_op_example_l4078_407879


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l4078_407892

theorem largest_prime_factor_of_1001 : ∃ p : ℕ, p.Prime ∧ p ∣ 1001 ∧ ∀ q : ℕ, q.Prime → q ∣ 1001 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l4078_407892


namespace NUMINAMATH_CALUDE_range_of_a_l4078_407877

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ y : ℝ, y ≥ a ∧ |y - 1| ≥ 1) → 
  a ∈ Set.Iic 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4078_407877


namespace NUMINAMATH_CALUDE_rhombus_region_area_l4078_407861

/-- Represents a rhombus ABCD -/
structure Rhombus where
  side_length : ℝ
  angle_B : ℝ

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem rhombus_region_area (r : Rhombus) 
  (h1 : r.side_length = 3)
  (h2 : r.angle_B = π / 2) :
  area (region_R r) = 9 * π / 16 :=
sorry

end NUMINAMATH_CALUDE_rhombus_region_area_l4078_407861


namespace NUMINAMATH_CALUDE_min_segments_to_return_l4078_407841

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    and the measure of angle ABC is 80 degrees, prove that the minimum number of segments
    needed to return to the starting point is 18. -/
theorem min_segments_to_return (m_angle_ABC : ℝ) (n : ℕ) : 
  m_angle_ABC = 80 → 
  (∀ m : ℕ, 100 * n = 360 * m) → 
  n ≥ 18 ∧ 
  (∀ k < n, ¬(∀ m : ℕ, 100 * k = 360 * m)) := by
  sorry

end NUMINAMATH_CALUDE_min_segments_to_return_l4078_407841


namespace NUMINAMATH_CALUDE_ratio_odd_even_divisors_l4078_407881

def M : ℕ := 36 * 36 * 85 * 128

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors M) * 4094 = sum_even_divisors M :=
sorry

end NUMINAMATH_CALUDE_ratio_odd_even_divisors_l4078_407881


namespace NUMINAMATH_CALUDE_exists_nat_square_not_positive_exists_real_not_root_quadratic_always_positive_exists_prime_not_odd_l4078_407852

-- 1. There exists a natural number whose square is not positive.
theorem exists_nat_square_not_positive : ∃ n : ℕ, ¬(n^2 > 0) := by sorry

-- 2. There exists a real number x that is not a root of the equation 5x-12=0.
theorem exists_real_not_root : ∃ x : ℝ, 5*x - 12 ≠ 0 := by sorry

-- 3. For all x ∈ ℝ, x^2 - 3x + 3 > 0.
theorem quadratic_always_positive : ∀ x : ℝ, x^2 - 3*x + 3 > 0 := by sorry

-- 4. There exists a prime number that is not odd.
theorem exists_prime_not_odd : ∃ p : ℕ, Nat.Prime p ∧ ¬Odd p := by sorry

end NUMINAMATH_CALUDE_exists_nat_square_not_positive_exists_real_not_root_quadratic_always_positive_exists_prime_not_odd_l4078_407852


namespace NUMINAMATH_CALUDE_inequality_solutions_l4078_407890

theorem inequality_solutions (a : ℝ) (h1 : a < 0) (h2 : a ≤ -Real.rpow 2 (1/3)) :
  ∃ (w x y z : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  (a^2 * |a + (w : ℝ)/a^2| + |1 + w| ≤ 1 - a^3) ∧
  (a^2 * |a + (x : ℝ)/a^2| + |1 + x| ≤ 1 - a^3) ∧
  (a^2 * |a + (y : ℝ)/a^2| + |1 + y| ≤ 1 - a^3) ∧
  (a^2 * |a + (z : ℝ)/a^2| + |1 + z| ≤ 1 - a^3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l4078_407890


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4078_407870

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_perimeter (A B C a b c : ℝ) : 
  Triangle A B C a b c →
  Real.sin A + Real.sin C = 2 * Real.sin B →
  (2 * a) * (3 * b) = (3 * b) * (5 * c) →
  (a / Real.sin A) = 6 →
  a + b + c = 6 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l4078_407870


namespace NUMINAMATH_CALUDE_sum_of_number_and_its_square_l4078_407856

theorem sum_of_number_and_its_square (n : ℕ) : n = 11 → n + n^2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_number_and_its_square_l4078_407856


namespace NUMINAMATH_CALUDE_chocolate_bar_theorem_l4078_407811

theorem chocolate_bar_theorem (n m : ℕ) (h : n * m = 25) :
  (∃ (b w : ℕ), b + w = n * m ∧ b = w + 1 ∧ b = (25 * w) / 3) →
  n + m = 10 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_theorem_l4078_407811


namespace NUMINAMATH_CALUDE_find_divisor_l4078_407801

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 139 →
  quotient = 7 →
  remainder = 6 →
  dividend = divisor * quotient + remainder →
  divisor = 19 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l4078_407801


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l4078_407853

def vector_a : ℝ × ℝ := (4, -5)
def vector_b (b : ℝ) : ℝ × ℝ := (b, 3)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem perpendicular_vectors (b : ℝ) :
  dot_product vector_a (vector_b b) = 0 → b = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l4078_407853


namespace NUMINAMATH_CALUDE_selling_price_calculation_l4078_407891

/-- The selling price that yields a 4% higher gain than selling at 340, given a cost of 250 -/
def higher_selling_price (cost : ℝ) (lower_price : ℝ) : ℝ :=
  let lower_gain := lower_price - cost
  let higher_gain := lower_gain * 1.04
  cost + higher_gain

theorem selling_price_calculation :
  higher_selling_price 250 340 = 343.6 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l4078_407891


namespace NUMINAMATH_CALUDE_season_win_percentage_l4078_407867

/-- 
Given a team that:
- Won 70 percent of its first 100 games
- Played a total of 100 games

Prove that the percentage of games won for the entire season is 70%.
-/
theorem season_win_percentage 
  (total_games : ℕ) 
  (first_100_win_percentage : ℚ) 
  (h1 : total_games = 100)
  (h2 : first_100_win_percentage = 70/100) : 
  first_100_win_percentage = 70/100 := by
sorry

end NUMINAMATH_CALUDE_season_win_percentage_l4078_407867


namespace NUMINAMATH_CALUDE_divisibility_by_30_l4078_407885

theorem divisibility_by_30 (p : ℕ) (h1 : p.Prime) (h2 : p ≥ 7) : 30 ∣ p^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_30_l4078_407885


namespace NUMINAMATH_CALUDE_multiple_of_10_average_l4078_407839

theorem multiple_of_10_average (N : ℕ) : 
  N % 10 = 0 → -- N is a multiple of 10
  (10 + N) / 2 = 305 → -- The average of multiples of 10 from 10 to N inclusive is 305
  N = 600 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_10_average_l4078_407839


namespace NUMINAMATH_CALUDE_batsman_new_average_l4078_407834

/-- Represents a batsman's performance -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the new average after an additional inning -/
def newAverage (bp : BatsmanPerformance) : Nat :=
  (bp.totalRuns + 74) / (bp.innings + 1)

/-- Theorem: The batsman's new average is 26 runs -/
theorem batsman_new_average (bp : BatsmanPerformance) 
  (h1 : bp.innings = 16)
  (h2 : newAverage bp = bp.totalRuns / bp.innings + 3)
  : newAverage bp = 26 := by
  sorry

#check batsman_new_average

end NUMINAMATH_CALUDE_batsman_new_average_l4078_407834


namespace NUMINAMATH_CALUDE_toaster_cost_l4078_407842

def amazon_purchase : ℝ := 3000
def tv_cost : ℝ := 700
def returned_bike_cost : ℝ := 500
def sold_bike_cost : ℝ := returned_bike_cost * 1.2
def sold_bike_price : ℝ := sold_bike_cost * 0.8
def total_out_of_pocket : ℝ := 2020

theorem toaster_cost :
  let total_return := tv_cost + returned_bike_cost
  let out_of_pocket_before_toaster := amazon_purchase - total_return + sold_bike_price
  let toaster_cost := out_of_pocket_before_toaster - total_out_of_pocket
  toaster_cost = 260 :=
by sorry

end NUMINAMATH_CALUDE_toaster_cost_l4078_407842


namespace NUMINAMATH_CALUDE_least_bench_sections_l4078_407817

/-- Represents the capacity of a single bench section -/
structure BenchCapacity where
  adults : Nat
  children : Nat

/-- Proves that 6 is the least positive integer N such that N bench sections
    can hold an equal number of adults and children -/
theorem least_bench_sections (capacity : BenchCapacity)
    (h_adults : capacity.adults = 8)
    (h_children : capacity.children = 12) :
    (∃ N : Nat, N > 0 ∧
      ∃ x : Nat, x > 0 ∧
        N * capacity.adults = x ∧
        N * capacity.children = x ∧
        ∀ M : Nat, M > 0 → M < N →
          ¬∃ y : Nat, y > 0 ∧
            M * capacity.adults = y ∧
            M * capacity.children = y) →
    (∃ N : Nat, N = 6 ∧ N > 0 ∧
      ∃ x : Nat, x > 0 ∧
        N * capacity.adults = x ∧
        N * capacity.children = x ∧
        ∀ M : Nat, M > 0 → M < N →
          ¬∃ y : Nat, y > 0 ∧
            M * capacity.adults = y ∧
            M * capacity.children = y) :=
  sorry

end NUMINAMATH_CALUDE_least_bench_sections_l4078_407817


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_value_l4078_407843

/-- Given two lines l₁ and l₂ in the form ax + by + c = 0,
    this function returns true if they are perpendicular. -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The slope-intercept form of l₁: (m-2)x + 3y + 2m = 0 -/
def l1 (m : ℝ) (x y : ℝ) : Prop :=
  (m - 2) * x + 3 * y + 2 * m = 0

/-- The slope-intercept form of l₂: x + my + 6 = 0 -/
def l2 (m : ℝ) (x y : ℝ) : Prop :=
  x + m * y + 6 = 0

theorem perpendicular_lines_m_value :
  ∀ m : ℝ, (∀ x y : ℝ, are_perpendicular (m - 2) 3 1 m) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_value_l4078_407843


namespace NUMINAMATH_CALUDE_xyz_value_l4078_407833

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14)
  (h3 : (x + y + z)^2 = 25) :
  x * y * z = 31 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l4078_407833


namespace NUMINAMATH_CALUDE_simplify_expression_l4078_407883

theorem simplify_expression (x y : ℝ) : 3*y - 5*x + 2*y + 4*x = 5*y - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4078_407883


namespace NUMINAMATH_CALUDE_product_digit_sum_l4078_407868

def first_number : ℕ := 141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141
def second_number : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem product_digit_sum :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l4078_407868


namespace NUMINAMATH_CALUDE_spencer_journey_distance_l4078_407884

def walking_distances : List Float := [1.2, 0.4, 0.6, 1.5]
def biking_distances : List Float := [1.8, 2]
def bus_distance : Float := 3

def biking_to_walking_factor : Float := 0.5
def bus_to_walking_factor : Float := 0.8

def total_walking_equivalent (walking : List Float) (biking : List Float) (bus : Float) 
  (bike_factor : Float) (bus_factor : Float) : Float :=
  (walking.sum) + 
  (biking.sum * bike_factor) + 
  (bus * bus_factor)

theorem spencer_journey_distance :
  total_walking_equivalent walking_distances biking_distances bus_distance
    biking_to_walking_factor bus_to_walking_factor = 8 := by
  sorry

end NUMINAMATH_CALUDE_spencer_journey_distance_l4078_407884


namespace NUMINAMATH_CALUDE_problem_solution_l4078_407829

theorem problem_solution (x : ℕ+) : 
  x^2 + 4*x + 29 = x*(4*x + 9) + 13 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4078_407829


namespace NUMINAMATH_CALUDE_oplus_example_l4078_407862

-- Define the ⊕ operation
def oplus (a b : ℕ) : ℕ := a + b + a * b

-- Statement to prove
theorem oplus_example : oplus (oplus 2 3) 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_oplus_example_l4078_407862


namespace NUMINAMATH_CALUDE_range_of_3x_plus_y_l4078_407882

theorem range_of_3x_plus_y (x y : ℝ) :
  3 * x^2 + y^2 ≤ 1 →
  ∃ (max min : ℝ), max = 2 ∧ min = -2 ∧
    (3 * x + y ≤ max ∧ 3 * x + y ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3x_plus_y_l4078_407882


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l4078_407872

theorem smallest_n_for_probability_condition : 
  (∃ n : ℕ+, (1 : ℚ) / (n * (n + 1)) < 1 / 2020 ∧ 
    ∀ m : ℕ+, m < n → (1 : ℚ) / (m * (m + 1)) ≥ 1 / 2020) ∧
  (∀ n : ℕ+, (1 : ℚ) / (n * (n + 1)) < 1 / 2020 ∧ 
    ∀ m : ℕ+, m < n → (1 : ℚ) / (m * (m + 1)) ≥ 1 / 2020 → n = 45) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l4078_407872


namespace NUMINAMATH_CALUDE_max_principals_in_period_l4078_407895

/-- Represents the number of years in a principal's term -/
def term_length : ℕ := 4

/-- Represents the total period in years -/
def total_period : ℕ := 10

/-- Represents the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 4

/-- Theorem stating that given a 10-year period and principals serving 4-year terms,
    the maximum number of principals that can serve during this period is 4 -/
theorem max_principals_in_period :
  ∀ (n : ℕ), n ≤ max_principals →
  n * term_length > total_period →
  (n - 1) * term_length ≤ total_period :=
sorry

end NUMINAMATH_CALUDE_max_principals_in_period_l4078_407895


namespace NUMINAMATH_CALUDE_function_sum_theorem_l4078_407858

/-- Given a function f(x) = a^x + a^(-x) where a > 0 and a ≠ 1, 
    if f(1) = 3, then f(0) + f(1) + f(2) = 12 -/
theorem function_sum_theorem (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + a^(-x)
  f 1 = 3 → f 0 + f 1 + f 2 = 12 := by
sorry

end NUMINAMATH_CALUDE_function_sum_theorem_l4078_407858


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4078_407874

theorem sufficient_not_necessary (x : ℝ) :
  ((x + 1) * (x - 2) > 0 → abs x ≥ 1) ∧
  ¬(abs x ≥ 1 → (x + 1) * (x - 2) > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4078_407874


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l4078_407830

/-- Given a geometric sequence {a_n} with positive terms where a_2, (1/2)a_3, a_1 form an arithmetic sequence,
    the ratio (a_3 + a_4) / (a_4 + a_5) is equal to (√5 - 1) / 2. -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Geometric sequence with common ratio q
  (a 2 - (1/2) * a 3 = (1/2) * a 3 - a 1) →  -- Arithmetic sequence condition
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l4078_407830


namespace NUMINAMATH_CALUDE_num_correct_statements_is_zero_l4078_407897

/-- Definition of a frustum -/
structure Frustum where
  has_parallel_bases : Bool
  lateral_edges_converge : Bool

/-- The three statements about frustums -/
def statement1 (f : Frustum) : Prop :=
  true -- We don't need to define this precisely as it's always false

def statement2 (f : Frustum) : Prop :=
  f.has_parallel_bases

def statement3 (f : Frustum) : Prop :=
  f.has_parallel_bases

/-- Theorem: The number of correct statements is 0 -/
theorem num_correct_statements_is_zero : 
  (∀ f : Frustum, ¬statement1 f) ∧ 
  (∀ f : Frustum, f.has_parallel_bases ∧ f.lateral_edges_converge → statement2 f) ∧
  (∀ f : Frustum, f.has_parallel_bases ∧ f.lateral_edges_converge → statement3 f) →
  (¬∃ f : Frustum, statement1 f) ∧ 
  (¬∃ f : Frustum, statement2 f) ∧ 
  (¬∃ f : Frustum, statement3 f) :=
by
  sorry

#check num_correct_statements_is_zero

end NUMINAMATH_CALUDE_num_correct_statements_is_zero_l4078_407897


namespace NUMINAMATH_CALUDE_equation_solutions_l4078_407848

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (2 * (x₁ - 3) = 3 * x₁ * (x₁ - 3) ∧ x₁ = 3) ∧ 
                (2 * (x₂ - 3) = 3 * x₂ * (x₂ - 3) ∧ x₂ = 2/3)) ∧
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 - 3 * y₁ + 1 = 0 ∧ y₁ = 1) ∧ 
                (2 * y₂^2 - 3 * y₂ + 1 = 0 ∧ y₂ = 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l4078_407848


namespace NUMINAMATH_CALUDE_paint_containers_left_over_l4078_407804

/-- Calculates the number of paint containers left over after repainting a bathroom --/
theorem paint_containers_left_over 
  (initial_containers : ℕ) 
  (total_walls : ℕ) 
  (tiled_walls : ℕ) 
  (ceiling_containers : ℕ) 
  (gradient_containers_per_wall : ℕ) : 
  initial_containers = 16 →
  total_walls = 4 →
  tiled_walls = 1 →
  ceiling_containers = 1 →
  gradient_containers_per_wall = 1 →
  initial_containers - 
    (ceiling_containers + 
     (total_walls - tiled_walls) * (1 + gradient_containers_per_wall)) = 11 := by
  sorry


end NUMINAMATH_CALUDE_paint_containers_left_over_l4078_407804


namespace NUMINAMATH_CALUDE_percentage_of_men_speaking_french_l4078_407805

theorem percentage_of_men_speaking_french (E : ℝ) (E_pos : E > 0) :
  let men_percentage : ℝ := 70
  let french_speaking_percentage : ℝ := 40
  let women_not_speaking_french_percentage : ℝ := 83.33333333333331
  let men_count : ℝ := (men_percentage / 100) * E
  let french_speaking_count : ℝ := (french_speaking_percentage / 100) * E
  let women_count : ℝ := E - men_count
  let women_speaking_french_count : ℝ := (1 - women_not_speaking_french_percentage / 100) * women_count
  let men_speaking_french_count : ℝ := french_speaking_count - women_speaking_french_count
  (men_speaking_french_count / men_count) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_men_speaking_french_l4078_407805


namespace NUMINAMATH_CALUDE_distance_to_center_l4078_407821

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8*x - 4*y + 16

/-- The center of the circle -/
def circle_center : ℝ × ℝ := sorry

/-- The given point -/
def given_point : ℝ × ℝ := (3, -1)

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem distance_to_center : distance circle_center given_point = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_center_l4078_407821


namespace NUMINAMATH_CALUDE_cost_price_percentage_l4078_407823

/-- Proves that given a discount of 12% and a gain percent of 37.5%, 
    the cost price is approximately 64% of the marked price. -/
theorem cost_price_percentage (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : marked_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : (marked_price - cost_price) / cost_price = 0.375) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |cost_price / marked_price - 0.64| < ε :=
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l4078_407823


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4078_407827

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 3*x = 2) : 3*x^2 - 9*x - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4078_407827


namespace NUMINAMATH_CALUDE_expression_evaluation_l4078_407849

theorem expression_evaluation : 
  8^(1/4) * 42 + (32 * Real.sqrt 3)^6 + Real.log 2 / Real.log 3 * (Real.log (Real.log 27 / Real.log 3) / Real.log 2) = 111 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4078_407849


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l4078_407837

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the new line
def new_line (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given :
  (new_line point.1 point.2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), new_line x₁ y₁ → new_line x₂ y₂ → 
    (x₂ - x₁) * 3 = (y₂ - y₁) * 2) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line x₁ y₁ → given_line x₂ y₂ → 
    (x₂ - x₁) * 3 = (y₂ - y₁) * 2) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l4078_407837


namespace NUMINAMATH_CALUDE_square_of_102_l4078_407864

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_CALUDE_square_of_102_l4078_407864


namespace NUMINAMATH_CALUDE_general_admission_price_l4078_407820

/-- Proves that the price of a general admission seat is $21.85 -/
theorem general_admission_price : 
  ∀ (total_tickets : ℕ) 
    (total_revenue : ℚ) 
    (vip_price : ℚ) 
    (gen_price : ℚ),
  total_tickets = 320 →
  total_revenue = 7500 →
  vip_price = 45 →
  (∃ (vip_tickets gen_tickets : ℕ),
    vip_tickets + gen_tickets = total_tickets ∧
    vip_tickets = gen_tickets - 276 ∧
    vip_price * vip_tickets + gen_price * gen_tickets = total_revenue) →
  gen_price = 21.85 := by
sorry


end NUMINAMATH_CALUDE_general_admission_price_l4078_407820


namespace NUMINAMATH_CALUDE_french_toast_slices_per_loaf_l4078_407866

/-- The number of slices in each loaf of bread for Suzanne's french toast -/
def slices_per_loaf (days_per_week : ℕ) (slices_per_day : ℕ) (weeks : ℕ) (total_loaves : ℕ) : ℕ :=
  (days_per_week * slices_per_day * weeks) / total_loaves

/-- Proof that the number of slices in each loaf is 6 -/
theorem french_toast_slices_per_loaf :
  slices_per_loaf 2 3 52 26 = 6 := by
  sorry

end NUMINAMATH_CALUDE_french_toast_slices_per_loaf_l4078_407866


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4078_407807

theorem complex_equation_solution (z : ℂ) (h : z + Complex.abs z = 2 + 8 * I) : z = -15 + 8 * I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4078_407807


namespace NUMINAMATH_CALUDE_tan_half_angle_special_point_l4078_407835

/-- 
If the terminal side of angle α passes through the point (-1, 2),
then tan(α/2) = (1 + √5) / 2.
-/
theorem tan_half_angle_special_point (α : Real) :
  (Real.cos α = -1 / Real.sqrt 5 ∧ Real.sin α = 2 / Real.sqrt 5) →
  Real.tan (α / 2) = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_angle_special_point_l4078_407835


namespace NUMINAMATH_CALUDE_max_m_value_l4078_407859

theorem max_m_value (b a m : ℝ) (h_b : b > 0) :
  (∀ a, (b - (a - 2))^2 + (Real.log b - (a - 1))^2 ≥ m^2 - m) →
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l4078_407859


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l4078_407822

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Statement 1
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Statement 2
theorem A_subset_C_implies_a_geq_7 (a : ℝ) (h : A ⊆ C a) : a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_implies_a_geq_7_l4078_407822


namespace NUMINAMATH_CALUDE_product_sequence_equals_243_l4078_407869

theorem product_sequence_equals_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_equals_243_l4078_407869


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l4078_407838

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (Real.sqrt (n + 1) ≤ Real.sqrt (5*n - 7) ∧ Real.sqrt (5*n - 7) < Real.sqrt (3*n + 6))) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l4078_407838


namespace NUMINAMATH_CALUDE_crayons_per_box_l4078_407808

theorem crayons_per_box (total_boxes : ℕ) (crayons_to_mae : ℕ) (extra_crayons_to_lea : ℕ) (crayons_left : ℕ) :
  total_boxes = 4 →
  crayons_to_mae = 5 →
  extra_crayons_to_lea = 7 →
  crayons_left = 15 →
  ∃ (crayons_per_box : ℕ),
    crayons_per_box * total_boxes = crayons_to_mae + (crayons_to_mae + extra_crayons_to_lea) + crayons_left ∧
    crayons_per_box = 8 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_box_l4078_407808


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l4078_407857

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ (x : ℝ), x ≠ 4 ∧ x ≠ 2 →
    3 * x^2 + 2 * x = (x - 4) * (x - 2)^2 * (P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 14 ∧ Q = -11 ∧ R = -8 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l4078_407857


namespace NUMINAMATH_CALUDE_candy_problem_l4078_407826

theorem candy_problem (initial_candy : ℕ) (talitha_took : ℕ) (remaining_candy : ℕ) 
  (h1 : initial_candy = 349)
  (h2 : talitha_took = 108)
  (h3 : remaining_candy = 88) :
  initial_candy - talitha_took - remaining_candy = 153 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l4078_407826


namespace NUMINAMATH_CALUDE_floor_tiling_floor_covered_l4078_407875

/-- A square floor of size n × n can be completely covered by an equal number of 2 × 2 and 3 × 1 tiles if and only if n is a multiple of 7. -/
theorem floor_tiling (n : ℕ) : 
  (∃ (a : ℕ), n^2 = 7 * a) ↔ ∃ (k : ℕ), n = 7 * k :=
by sorry

/-- The number of tiles of each type needed to cover a square floor of size n × n, where n is a multiple of 7. -/
def num_tiles (n : ℕ) (h : ∃ (k : ℕ), n = 7 * k) : ℕ :=
  n^2 / 7

/-- Verification that the floor is completely covered using an equal number of 2 × 2 and 3 × 1 tiles. -/
theorem floor_covered (n : ℕ) (h : ∃ (k : ℕ), n = 7 * k) :
  let a := num_tiles n h
  4 * a + 3 * a = n^2 :=
by sorry

end NUMINAMATH_CALUDE_floor_tiling_floor_covered_l4078_407875


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l4078_407855

/-- If the equation 2ax^2 - x - 1 = 0 has exactly one solution in the interval (0,1), then a > 1 -/
theorem unique_solution_implies_a_greater_than_one (a : ℝ) : 
  (∃! x : ℝ, x ∈ Set.Ioo 0 1 ∧ 2*a*x^2 - x - 1 = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l4078_407855


namespace NUMINAMATH_CALUDE_fraction_arrangement_l4078_407802

theorem fraction_arrangement :
  (1 / 8 * 1 / 9 * 1 / 28 : ℚ) = 1 / 2016 ∨ ((1 / 8 - 1 / 9) * 1 / 28 : ℚ) = 1 / 2016 :=
by sorry

end NUMINAMATH_CALUDE_fraction_arrangement_l4078_407802


namespace NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l4078_407851

theorem smallest_gcd_of_multiples (a b : ℕ+) (h : Nat.gcd a b = 18) :
  (Nat.gcd (12 * a) (20 * b)).min = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_multiples_l4078_407851


namespace NUMINAMATH_CALUDE_square_difference_l4078_407847

theorem square_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x - y = 4) : x^2 - y^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l4078_407847


namespace NUMINAMATH_CALUDE_not_divisible_by_11599_l4078_407887

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def N : ℚ := (factorial 3400) / ((factorial 1700) ^ 2)

theorem not_divisible_by_11599 : ¬ (∃ (k : ℤ), N = k * (11599 : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_11599_l4078_407887


namespace NUMINAMATH_CALUDE_blocks_used_for_first_building_l4078_407860

/-- Given the number of building blocks Jesse started with, used for farmhouse and fenced-in area, and left at the end, 
    calculate the number of blocks used for the first building. -/
theorem blocks_used_for_first_building 
  (total_blocks : ℕ) 
  (farmhouse_blocks : ℕ) 
  (fenced_area_blocks : ℕ) 
  (blocks_left : ℕ) 
  (h1 : total_blocks = 344) 
  (h2 : farmhouse_blocks = 123) 
  (h3 : fenced_area_blocks = 57) 
  (h4 : blocks_left = 84) :
  total_blocks - farmhouse_blocks - fenced_area_blocks - blocks_left = 80 :=
by sorry

end NUMINAMATH_CALUDE_blocks_used_for_first_building_l4078_407860


namespace NUMINAMATH_CALUDE_basement_pump_time_l4078_407876

-- Define the constants
def basement_length : ℝ := 30
def basement_width : ℝ := 40
def water_depth : ℝ := 2
def initial_pumps : ℕ := 4
def pump_capacity : ℝ := 10
def breakdown_time : ℝ := 120
def cubic_foot_to_gallon : ℝ := 7.5

-- Define the theorem
theorem basement_pump_time :
  let initial_volume : ℝ := basement_length * basement_width * water_depth * cubic_foot_to_gallon
  let initial_pump_rate : ℝ := initial_pumps * pump_capacity
  let volume_pumped_before_breakdown : ℝ := initial_pump_rate * breakdown_time
  let remaining_volume : ℝ := initial_volume - volume_pumped_before_breakdown
  let remaining_pumps : ℕ := initial_pumps - 1
  let remaining_pump_rate : ℝ := remaining_pumps * pump_capacity
  let remaining_time : ℝ := remaining_volume / remaining_pump_rate
  breakdown_time + remaining_time = 560 := by
  sorry

end NUMINAMATH_CALUDE_basement_pump_time_l4078_407876


namespace NUMINAMATH_CALUDE_vector_sum_problem_l4078_407832

/-- Given two vectors a and b in ℝ³, prove that a + 2b equals the expected result. -/
theorem vector_sum_problem (a b : Fin 3 → ℝ) 
  (ha : a = ![1, 2, 3]) 
  (hb : b = ![-1, 0, 1]) : 
  a + 2 • b = ![-1, 2, 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_problem_l4078_407832


namespace NUMINAMATH_CALUDE_carol_savings_per_week_l4078_407873

/-- Proves that Carol saves $9 per week given the initial conditions and final equality of savings --/
theorem carol_savings_per_week (carol_initial : ℕ) (mike_initial : ℕ) (mike_savings : ℕ) (weeks : ℕ)
  (h1 : carol_initial = 60)
  (h2 : mike_initial = 90)
  (h3 : mike_savings = 3)
  (h4 : weeks = 5)
  (h5 : ∃ (carol_savings : ℕ), carol_initial + weeks * carol_savings = mike_initial + weeks * mike_savings) :
  ∃ (carol_savings : ℕ), carol_savings = 9 := by
  sorry

end NUMINAMATH_CALUDE_carol_savings_per_week_l4078_407873


namespace NUMINAMATH_CALUDE_odd_function_property_l4078_407819

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) :
  f (-2) = 11 → f 2 = -11 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l4078_407819


namespace NUMINAMATH_CALUDE_louis_lemon_heads_l4078_407865

/-- The number of Lemon Heads in a package -/
def lemon_heads_per_package : ℕ := 6

/-- The number of whole boxes Louis ate -/
def boxes_eaten : ℕ := 9

/-- The total number of Lemon Heads Louis ate -/
def total_lemon_heads : ℕ := lemon_heads_per_package * boxes_eaten

theorem louis_lemon_heads : total_lemon_heads = 54 := by
  sorry

end NUMINAMATH_CALUDE_louis_lemon_heads_l4078_407865


namespace NUMINAMATH_CALUDE_irrational_between_neg_one_and_two_l4078_407815

theorem irrational_between_neg_one_and_two :
  (-1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧
  ¬(-1 < -Real.sqrt 3 ∧ -Real.sqrt 3 < 2) ∧
  ¬(-1 < -Real.sqrt 5 ∧ -Real.sqrt 5 < 2) ∧
  ¬(-1 < Real.sqrt 5 ∧ Real.sqrt 5 < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_irrational_between_neg_one_and_two_l4078_407815


namespace NUMINAMATH_CALUDE_domain_range_equal_iff_l4078_407880

/-- The function f(x) = √(ax² + bx) where b > 0 -/
noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

/-- The domain of f -/
def domain (a b : ℝ) : Set ℝ :=
  if a > 0 then {x | x ≤ -b/a ∨ x ≥ 0}
  else if a < 0 then {x | 0 ≤ x ∧ x ≤ -b/a}
  else {x | x ≥ 0}

/-- The range of f -/
def range (a b : ℝ) : Set ℝ :=
  if a ≥ 0 then {y | y ≥ 0}
  else {y | 0 ≤ y ∧ y ≤ b / (2 * Real.sqrt (-a))}

theorem domain_range_equal_iff (b : ℝ) (hb : b > 0) :
  ∀ a : ℝ, domain a b = range a b ↔ a = -4 ∨ a = 0 := by sorry

end NUMINAMATH_CALUDE_domain_range_equal_iff_l4078_407880


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l4078_407846

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l4078_407846


namespace NUMINAMATH_CALUDE_max_value_theorem_l4078_407800

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*a*c*Real.sqrt 3 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l4078_407800


namespace NUMINAMATH_CALUDE_expand_product_l4078_407854

theorem expand_product (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l4078_407854


namespace NUMINAMATH_CALUDE_min_chord_length_proof_l4078_407845

/-- The circle equation: x^2 + y^2 - 6x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (1, 2)

/-- The minimum length of the chord -/
def min_chord_length : ℝ := 2

theorem min_chord_length_proof :
  ∀ (x y : ℝ), circle_equation x y →
  ∀ (chord_length : ℝ),
  (∃ (x1 y1 x2 y2 : ℝ), 
    circle_equation x1 y1 ∧ 
    circle_equation x2 y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = chord_length^2 ∧
    (x1 + x2) / 2 = point.1 ∧ 
    (y1 + y2) / 2 = point.2) →
  chord_length ≥ min_chord_length :=
by sorry

#check min_chord_length_proof

end NUMINAMATH_CALUDE_min_chord_length_proof_l4078_407845


namespace NUMINAMATH_CALUDE_problem_solution_l4078_407863

theorem problem_solution (x : ℝ) : (0.50 * x = 0.05 * 500 - 20) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4078_407863


namespace NUMINAMATH_CALUDE_john_money_left_l4078_407850

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_cost := 4 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_cost

/-- Theorem stating that John will have 50 - 10q dollars left -/
theorem john_money_left (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l4078_407850


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l4078_407844

/-- Initial percentage of alcohol in the solution -/
def initial_percentage : ℝ := 5

/-- Initial volume of the solution in liters -/
def initial_volume : ℝ := 40

/-- Volume of alcohol added in liters -/
def added_alcohol : ℝ := 3.5

/-- Volume of water added in liters -/
def added_water : ℝ := 6.5

/-- Final percentage of alcohol in the solution -/
def final_percentage : ℝ := 11

theorem initial_alcohol_percentage :
  initial_percentage = 5 :=
by
  have h1 : initial_volume + added_alcohol + added_water = 50 := by sorry
  have h2 : (initial_percentage / 100) * initial_volume + added_alcohol =
            (final_percentage / 100) * (initial_volume + added_alcohol + added_water) := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l4078_407844


namespace NUMINAMATH_CALUDE_intersection_line_equation_l4078_407810

/-- Circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def circle1 : Circle := { center := (-5, -3), radius := 15 }
def circle2 : Circle := { center := (4, 15), radius := 9 }

/-- The line passing through the intersection points of two circles -/
def intersectionLine (c1 c2 : Circle) : Line := sorry

theorem intersection_line_equation :
  let l := intersectionLine circle1 circle2
  l.a = 1 ∧ l.b = 1 ∧ l.c = -27/4 := by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l4078_407810


namespace NUMINAMATH_CALUDE_two_digit_subtraction_equality_l4078_407824

theorem two_digit_subtraction_equality (a b : Nat) : 
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a ≠ b → (70 * a - 7 * a) - (70 * b - 7 * b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_subtraction_equality_l4078_407824


namespace NUMINAMATH_CALUDE_determinant_equality_l4078_407840

theorem determinant_equality (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 7 → Matrix.det !![p - 3*r, q - 3*s; r, s] = 7 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equality_l4078_407840


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l4078_407889

-- Define the circles M₁ and M₂
def M₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def M₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the moving circle M
structure MovingCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency conditions
def externally_tangent (M : MovingCircle) : Prop :=
  M₁ (M.center.1 + M.radius) M.center.2

def internally_tangent (M : MovingCircle) : Prop :=
  M₂ (M.center.1 - M.radius) M.center.2

-- Define the trajectory of the center of M
def trajectory (x y : ℝ) : Prop :=
  x^2/4 + y^2/3 = 1 ∧ x ≠ -2

-- Theorem statement
theorem moving_circle_trajectory (M : MovingCircle) :
  externally_tangent M → internally_tangent M →
  trajectory M.center.1 M.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l4078_407889


namespace NUMINAMATH_CALUDE_equal_bills_at_120_minutes_l4078_407806

/-- The base rate for United Telephone service in dollars -/
def united_base_rate : ℝ := 6

/-- The per-minute charge for United Telephone in dollars -/
def united_per_minute : ℝ := 0.25

/-- The base rate for Atlantic Call service in dollars -/
def atlantic_base_rate : ℝ := 12

/-- The per-minute charge for Atlantic Call in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 120

theorem equal_bills_at_120_minutes :
  united_base_rate + united_per_minute * equal_minutes =
  atlantic_base_rate + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_bills_at_120_minutes_l4078_407806


namespace NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_exists_mutually_exclusive_not_complementary_l4078_407899

variable {Ω : Type} [MeasurableSpace Ω]
variable (A₁ A₂ : Set Ω)

-- Define mutually exclusive events
def mutually_exclusive (A₁ A₂ : Set Ω) : Prop := A₁ ∩ A₂ = ∅

-- Define complementary events
def complementary (A₁ A₂ : Set Ω) : Prop := A₁ ∪ A₂ = Ω ∧ A₁ ∩ A₂ = ∅

-- Theorem 1: Complementary events are mutually exclusive
theorem complementary_implies_mutually_exclusive :
  complementary A₁ A₂ → mutually_exclusive A₁ A₂ := by sorry

-- Theorem 2: Existence of mutually exclusive events that are not complementary
theorem exists_mutually_exclusive_not_complementary :
  ∃ A₁ A₂ : Set Ω, mutually_exclusive A₁ A₂ ∧ ¬complementary A₁ A₂ := by sorry

end NUMINAMATH_CALUDE_complementary_implies_mutually_exclusive_exists_mutually_exclusive_not_complementary_l4078_407899


namespace NUMINAMATH_CALUDE_all_expressions_zero_l4078_407828

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define points in the vector space
variable (A B C D O N M P Q : V)

-- Define the expressions
def expr1 (A B C : V) : V := (B - A) + (C - B) + (A - C)
def expr2 (A B C D : V) : V := (B - A) - (C - A) + (D - B) - (D - C)
def expr3 (O A D : V) : V := (A - O) - (D - O) + (D - A)
def expr4 (N Q P M : V) : V := (Q - N) + (P - Q) + (N - M) - (P - M)

-- Theorem stating that all expressions result in the zero vector
theorem all_expressions_zero (A B C D O N M P Q : V) : 
  expr1 A B C = 0 ∧ 
  expr2 A B C D = 0 ∧ 
  expr3 O A D = 0 ∧ 
  expr4 N Q P M = 0 :=
sorry

end NUMINAMATH_CALUDE_all_expressions_zero_l4078_407828
