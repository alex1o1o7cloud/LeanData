import Mathlib

namespace NUMINAMATH_CALUDE_function_inequality_l3877_387720

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem function_inequality (b c : ℝ) 
  (h : ∀ x : ℝ, f b c (-x) = f b c x) : 
  f b c 1 < f b c (-2) ∧ f b c (-2) < f b c 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3877_387720


namespace NUMINAMATH_CALUDE_fence_width_is_ten_l3877_387721

/-- A rectangular fence with specific properties -/
structure RectangularFence where
  circumference : ℝ
  length : ℝ
  width : ℝ
  circ_eq : circumference = 2 * (length + width)
  width_eq : width = 2 * length

/-- The width of a rectangular fence with circumference 30m and width twice the length is 10m -/
theorem fence_width_is_ten (fence : RectangularFence) 
    (h_circ : fence.circumference = 30) : fence.width = 10 := by
  sorry

end NUMINAMATH_CALUDE_fence_width_is_ten_l3877_387721


namespace NUMINAMATH_CALUDE_solve_for_y_l3877_387782

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 4) (h2 : x = 2) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3877_387782


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l3877_387710

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the tangent line at (1, -1)
def tangent_line (x : ℝ) : ℝ := -3*x + 2

-- Theorem statement
theorem tangent_triangle_area : 
  let x_intercept : ℝ := 2/3
  let y_intercept : ℝ := tangent_line 0
  let area : ℝ := (1/2) * x_intercept * y_intercept
  (f 1 = -1) ∧ (f' 1 = -3) → area = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_tangent_triangle_area_l3877_387710


namespace NUMINAMATH_CALUDE_small_parallelogram_area_l3877_387726

/-- Given a parallelogram ABCD with area 1, where sides AB and CD are divided into n equal parts,
    and sides AD and BC are divided into m equal parts, the area of each smaller parallelogram
    formed by connecting the division points is 1 / (mn - 1). -/
theorem small_parallelogram_area (n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let total_area : ℝ := 1
  let num_small_parallelograms : ℕ := n * m - 1
  let small_parallelogram_area : ℝ := total_area / num_small_parallelograms
  small_parallelogram_area = 1 / (n * m - 1) := by
  sorry

end NUMINAMATH_CALUDE_small_parallelogram_area_l3877_387726


namespace NUMINAMATH_CALUDE_no_chess_tournament_with_804_games_l3877_387709

theorem no_chess_tournament_with_804_games : ¬∃ (n : ℕ), n > 0 ∧ n * (n - 4) = 1608 := by
  sorry

end NUMINAMATH_CALUDE_no_chess_tournament_with_804_games_l3877_387709


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3877_387771

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 3 * a 5 = 4 →
  a 1 * a 2 * a 3 * a 4 * a 5 * a 6 * a 7 = 128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3877_387771


namespace NUMINAMATH_CALUDE_cough_ratio_l3877_387733

-- Define the number of coughs per minute for Georgia
def georgia_coughs_per_minute : ℕ := 5

-- Define the total number of coughs after 20 minutes
def total_coughs_after_20_minutes : ℕ := 300

-- Define Robert's coughs per minute
def robert_coughs_per_minute : ℕ := (total_coughs_after_20_minutes - georgia_coughs_per_minute * 20) / 20

-- Theorem stating the ratio of Robert's coughs to Georgia's coughs is 2:1
theorem cough_ratio :
  robert_coughs_per_minute / georgia_coughs_per_minute = 2 ∧
  robert_coughs_per_minute > georgia_coughs_per_minute :=
by sorry

end NUMINAMATH_CALUDE_cough_ratio_l3877_387733


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3877_387753

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2009 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3877_387753


namespace NUMINAMATH_CALUDE_stating_total_seats_is_680_l3877_387737

/-- 
Calculates the total number of seats in a theater given the following conditions:
- The first row has 15 seats
- Each row has 2 more seats than the previous row
- The last row has 53 seats
-/
def theaterSeats : ℕ := by
  -- Define the number of seats in the first row
  let firstRow : ℕ := 15
  -- Define the increase in seats per row
  let seatIncrease : ℕ := 2
  -- Define the number of seats in the last row
  let lastRow : ℕ := 53
  
  -- Calculate the number of rows
  let numRows : ℕ := (lastRow - firstRow) / seatIncrease + 1
  
  -- Calculate the total number of seats
  let totalSeats : ℕ := numRows * (firstRow + lastRow) / 2
  
  exact totalSeats

/-- 
Theorem stating that the total number of seats in the theater is 680
-/
theorem total_seats_is_680 : theaterSeats = 680 := by
  sorry

end NUMINAMATH_CALUDE_stating_total_seats_is_680_l3877_387737


namespace NUMINAMATH_CALUDE_equal_area_rectangle_width_l3877_387769

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: Given two rectangles of equal area, where one rectangle has dimensions 8 by x,
    and the other has dimensions 4 by 30, the value of x is 15 -/
theorem equal_area_rectangle_width :
  ∀ (x : ℝ),
  let r1 := Rectangle.mk 8 x
  let r2 := Rectangle.mk 4 30
  area r1 = area r2 → x = 15 := by
sorry


end NUMINAMATH_CALUDE_equal_area_rectangle_width_l3877_387769


namespace NUMINAMATH_CALUDE_jellybean_count_l3877_387727

theorem jellybean_count (total blue purple red orange : ℕ) : 
  total = 200 →
  blue = 14 →
  purple = 26 →
  red = 120 →
  total = blue + purple + red + orange →
  orange = 40 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_l3877_387727


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3877_387703

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ x y : ℝ, y = 3*x ∧ x^2/a^2 - y^2/b^2 = 1) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3877_387703


namespace NUMINAMATH_CALUDE_inequality_solution_l3877_387748

theorem inequality_solution (x : ℝ) : (1/2)^x - x + 1/2 > 0 → x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3877_387748


namespace NUMINAMATH_CALUDE_line_BC_equation_l3877_387739

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  angle_bisector_B : ℝ → ℝ → Prop
  angle_bisector_C : ℝ → ℝ → Prop

-- Define the specific triangle from the problem
def triangle_ABC : Triangle where
  A := (1, 4)
  angle_bisector_B := λ x y => x - 2*y = 0
  angle_bisector_C := λ x y => x + y - 1 = 0

-- Define the equation of line BC
def line_BC (x y : ℝ) : Prop := 4*x + 17*y + 12 = 0

-- Theorem statement
theorem line_BC_equation (t : Triangle) (h1 : t = triangle_ABC) :
  ∀ x y, t.angle_bisector_B x y ∧ t.angle_bisector_C x y → line_BC x y :=
by sorry

end NUMINAMATH_CALUDE_line_BC_equation_l3877_387739


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l3877_387791

theorem smallest_multiples_sum : ∃ (a b : ℕ),
  (a ≥ 10 ∧ a < 100 ∧ a % 5 = 0 ∧ ∀ x : ℕ, (x ≥ 10 ∧ x < 100 ∧ x % 5 = 0) → a ≤ x) ∧
  (b ≥ 100 ∧ b < 1000 ∧ b % 6 = 0 ∧ ∀ y : ℕ, (y ≥ 100 ∧ y < 1000 ∧ y % 6 = 0) → b ≤ y) ∧
  a + b = 112 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l3877_387791


namespace NUMINAMATH_CALUDE_tan_equality_proof_l3877_387773

theorem tan_equality_proof (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (225 * π / 180) → n = 45 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_proof_l3877_387773


namespace NUMINAMATH_CALUDE_vowels_on_board_l3877_387718

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 3

/-- The total number of alphabets written on the board -/
def total_written : ℕ := num_vowels * times_written

theorem vowels_on_board : total_written = 15 := by
  sorry

end NUMINAMATH_CALUDE_vowels_on_board_l3877_387718


namespace NUMINAMATH_CALUDE_segment_length_to_reflection_segment_length_F_to_F_l3877_387706

/-- The length of a segment from a point to its reflection over the x-axis -/
theorem segment_length_to_reflection (x y : ℝ) : 
  let F : ℝ × ℝ := (x, y)
  let F' : ℝ × ℝ := (x, -y)
  Real.sqrt ((F'.1 - F.1)^2 + (F'.2 - F.2)^2) = 2 * abs y :=
by sorry

/-- The specific case for F(-4, 3) -/
theorem segment_length_F_to_F'_is_6 : 
  let F : ℝ × ℝ := (-4, 3)
  let F' : ℝ × ℝ := (-4, -3)
  Real.sqrt ((F'.1 - F.1)^2 + (F'.2 - F.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_to_reflection_segment_length_F_to_F_l3877_387706


namespace NUMINAMATH_CALUDE_right_focus_of_hyperbola_l3877_387724

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- Theorem: The right focus of the hyperbola x^2/3 - y^2 = 1 is (2, 0) -/
theorem right_focus_of_hyperbola :
  ∀ (x y : ℝ), hyperbola x y → right_focus = (2, 0) := by
  sorry

end NUMINAMATH_CALUDE_right_focus_of_hyperbola_l3877_387724


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3877_387728

theorem smallest_marble_count : ∃ N : ℕ, 
  N > 1 ∧ 
  N % 9 = 1 ∧ 
  N % 10 = 1 ∧ 
  N % 11 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 9 = 1 ∧ m % 10 = 1 ∧ m % 11 = 1 → m ≥ N) ∧
  N = 991 := by
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3877_387728


namespace NUMINAMATH_CALUDE_two_valid_arrangements_l3877_387707

/-- Represents an arrangement of people in rows. -/
structure Arrangement where
  rows : ℕ
  front : ℕ

/-- Checks if an arrangement is valid according to the problem conditions. -/
def isValidArrangement (a : Arrangement) : Prop :=
  a.rows ≥ 3 ∧
  a.front * a.rows + a.rows * (a.rows - 1) / 2 = 100

/-- The main theorem stating that there are exactly two valid arrangements. -/
theorem two_valid_arrangements :
  ∃! (s : Finset Arrangement), (∀ a ∈ s, isValidArrangement a) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_arrangements_l3877_387707


namespace NUMINAMATH_CALUDE_max_skip_percentage_is_five_percent_l3877_387764

/-- The maximum percentage of school days a senior can miss and still skip final exams -/
def max_skip_percentage (total_days : ℕ) (max_skip_days : ℕ) : ℚ :=
  (max_skip_days : ℚ) / (total_days : ℚ) * 100

/-- Theorem stating the maximum percentage of school days a senior can miss in the given scenario -/
theorem max_skip_percentage_is_five_percent :
  max_skip_percentage 180 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_skip_percentage_is_five_percent_l3877_387764


namespace NUMINAMATH_CALUDE_favorite_numbers_exist_l3877_387794

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem favorite_numbers_exist : ∃ (a b c : ℕ), 
  a * b * c = 71668 ∧ 
  a * sum_of_digits a = 10 * a ∧
  b * sum_of_digits b = 10 * b ∧
  c * sum_of_digits c = 10 * c :=
sorry

end NUMINAMATH_CALUDE_favorite_numbers_exist_l3877_387794


namespace NUMINAMATH_CALUDE_good_carrots_count_l3877_387701

/-- Given that Carol picked 29 carrots, her mother picked 16 carrots, and they had 7 bad carrots,
    prove that the number of good carrots is 38. -/
theorem good_carrots_count (carol_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ)
    (h1 : carol_carrots = 29)
    (h2 : mom_carrots = 16)
    (h3 : bad_carrots = 7) :
    carol_carrots + mom_carrots - bad_carrots = 38 := by
  sorry

end NUMINAMATH_CALUDE_good_carrots_count_l3877_387701


namespace NUMINAMATH_CALUDE_bowl_delivery_fee_l3877_387746

/-- The problem of calculating the initial fee for a bowl delivery service -/
theorem bowl_delivery_fee
  (total_bowls : ℕ)
  (safe_delivery_pay : ℕ)
  (loss_penalty : ℕ)
  (lost_bowls : ℕ)
  (broken_bowls : ℕ)
  (total_payment : ℕ)
  (h1 : total_bowls = 638)
  (h2 : safe_delivery_pay = 3)
  (h3 : loss_penalty = 4)
  (h4 : lost_bowls = 12)
  (h5 : broken_bowls = 15)
  (h6 : total_payment = 1825) :
  ∃ (initial_fee : ℕ),
    initial_fee = 100 ∧
    total_payment = initial_fee +
      (total_bowls - lost_bowls - broken_bowls) * safe_delivery_pay -
      (lost_bowls + broken_bowls) * loss_penalty :=
by sorry

end NUMINAMATH_CALUDE_bowl_delivery_fee_l3877_387746


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3877_387745

theorem polynomial_factorization (a x y : ℝ) :
  3 * a * x^2 - 3 * a * y^2 = 3 * a * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3877_387745


namespace NUMINAMATH_CALUDE_shop_dimension_example_l3877_387715

/-- Calculates the dimension of a shop given its monthly rent and annual rent per square foot. -/
def shopDimension (monthlyRent : ℕ) (annualRentPerSqFt : ℕ) : ℕ :=
  (monthlyRent * 12) / annualRentPerSqFt

/-- Theorem stating that for a shop with a monthly rent of 1300 and an annual rent per square foot of 156, the dimension is 100 square feet. -/
theorem shop_dimension_example : shopDimension 1300 156 = 100 := by
  sorry

end NUMINAMATH_CALUDE_shop_dimension_example_l3877_387715


namespace NUMINAMATH_CALUDE_oliver_bumper_car_rides_l3877_387729

def carnival_rides (ferris_wheel_rides : ℕ) (tickets_per_ride : ℕ) (total_tickets : ℕ) : ℕ :=
  (total_tickets - ferris_wheel_rides * tickets_per_ride) / tickets_per_ride

theorem oliver_bumper_car_rides :
  carnival_rides 5 7 63 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_bumper_car_rides_l3877_387729


namespace NUMINAMATH_CALUDE_equivalence_conditions_l3877_387714

theorem equivalence_conditions (n : ℕ) :
  (∀ (a : ℕ+), n ∣ a^n - a) ↔
  (∀ (p : ℕ), Prime p → p ∣ n → (¬(p^2 ∣ n) ∧ (p - 1 ∣ n - 1))) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_conditions_l3877_387714


namespace NUMINAMATH_CALUDE_symmetry_example_l3877_387765

/-- Given two points in a 2D plane, this function checks if they are symmetric with respect to the origin. -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- Theorem stating that the point (2,-3) is symmetric to (-2,3) with respect to the origin. -/
theorem symmetry_example : symmetric_wrt_origin (-2, 3) (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_example_l3877_387765


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3877_387795

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 12^600 - 8^400 = 2^1204 * k ∧ k % 2 ≠ 0) ∧
  (∀ m : ℕ, m > 1204 → ¬(∃ l : ℕ, 12^600 - 8^400 = 2^m * l)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3877_387795


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l3877_387762

/-- Represents the total investment in yuan -/
def total_investment : ℝ := 82000000000

/-- The scientific notation representation of the total investment -/
def scientific_notation : ℝ := 8.2 * (10 ^ 10)

/-- Theorem stating that the total investment equals its scientific notation representation -/
theorem investment_scientific_notation : total_investment = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l3877_387762


namespace NUMINAMATH_CALUDE_total_food_for_three_months_l3877_387731

-- Define the number of days in each month
def december_days : ℕ := 31
def january_days : ℕ := 31
def february_days : ℕ := 28

-- Define the amount of food per feeding
def food_per_feeding : ℚ := 1/2

-- Define the number of feedings per day
def feedings_per_day : ℕ := 2

-- Theorem statement
theorem total_food_for_three_months :
  let total_days := december_days + january_days + february_days
  let daily_food := food_per_feeding * feedings_per_day
  total_days * daily_food = 90 := by sorry

end NUMINAMATH_CALUDE_total_food_for_three_months_l3877_387731


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3877_387763

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3877_387763


namespace NUMINAMATH_CALUDE_quiz_competition_arrangements_l3877_387735

/-- The number of permutations of k items chosen from n distinct items -/
def permutations (n k : ℕ) : ℕ :=
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- Theorem: There are 24 ways to arrange 3 out of 4 distinct items in order -/
theorem quiz_competition_arrangements : permutations 4 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_arrangements_l3877_387735


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l3877_387751

open Real

/-- Represents a curve in polar coordinates -/
structure PolarCurve where
  equation : ℝ → ℝ → Prop

/-- The first curve: ρ = 2sin θ -/
def C₁ : PolarCurve :=
  ⟨λ ρ θ ↦ ρ = 2 * sin θ⟩

/-- The second curve: ρ = 2cos θ -/
def C₂ : PolarCurve :=
  ⟨λ ρ θ ↦ ρ = 2 * cos θ⟩

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Finds the intersection points of two polar curves -/
def intersectionPoints (c₁ c₂ : PolarCurve) : Set PolarPoint :=
  {p | c₁.equation p.ρ p.θ ∧ c₂.equation p.ρ p.θ}

/-- The perpendicular bisector equation -/
def perpendicularBisector (ρ θ : ℝ) : Prop :=
  ρ = 1 / (sin θ + cos θ)

theorem perpendicular_bisector_of_intersection_points :
  ∀ (A B : PolarPoint), A ∈ intersectionPoints C₁ C₂ → B ∈ intersectionPoints C₁ C₂ → A ≠ B →
  ∀ ρ θ, perpendicularBisector ρ θ ↔ 
    (∃ t, ρ * cos θ = A.ρ * cos A.θ + t * (B.ρ * cos B.θ - A.ρ * cos A.θ) ∧
          ρ * sin θ = A.ρ * sin A.θ + t * (B.ρ * sin B.θ - A.ρ * sin A.θ) ∧
          0 < t ∧ t < 1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l3877_387751


namespace NUMINAMATH_CALUDE_pirate_treasure_l3877_387799

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l3877_387799


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l3877_387742

theorem cos_2alpha_plus_3pi_over_5 (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 4) :
  Real.cos (2 * α + 3 * π / 5) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_3pi_over_5_l3877_387742


namespace NUMINAMATH_CALUDE_cuboids_on_diagonal_of_90_cube_l3877_387770

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  side : ℕ

/-- Calculates the number of cuboids a diagonal of a cube passes through -/
def cuboids_on_diagonal (cube : Cube) (cuboid : Cuboid) : ℕ :=
  let n1 := cube.side / cuboid.height - 1
  let n2 := cube.side / cuboid.width - 1
  let n3 := cube.side / cuboid.length - 1
  let i12 := cube.side / (cuboid.height * cuboid.width) - 1
  let i23 := cube.side / (cuboid.width * cuboid.length) - 1
  let i13 := cube.side / (cuboid.height * cuboid.length) - 1
  let i123 := cube.side / (cuboid.height * cuboid.width * cuboid.length) - 1
  n1 + n2 + n3 - (i12 + i23 + i13) + i123

/-- The main theorem to be proved -/
theorem cuboids_on_diagonal_of_90_cube (c : Cube) (b : Cuboid) :
  c.side = 90 ∧ b.length = 2 ∧ b.width = 3 ∧ b.height = 5 →
  cuboids_on_diagonal c b = 65 := by
  sorry

end NUMINAMATH_CALUDE_cuboids_on_diagonal_of_90_cube_l3877_387770


namespace NUMINAMATH_CALUDE_percent_relation_l3877_387757

theorem percent_relation (x y : ℝ) (h : 0.2 * (x - y) = 0.14 * (x + y)) :
  y = (3 / 17) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3877_387757


namespace NUMINAMATH_CALUDE_selection_schemes_count_l3877_387736

-- Define the number of individuals and cities
def total_individuals : ℕ := 6
def total_cities : ℕ := 4

-- Define the number of restricted individuals (A and B)
def restricted_individuals : ℕ := 2

-- Function to calculate permutations
def permutations (n k : ℕ) : ℕ := (n.factorial) / (n - k).factorial

-- Theorem statement
theorem selection_schemes_count :
  (permutations total_individuals total_cities) -
  (restricted_individuals * permutations (total_individuals - 1) (total_cities - 1)) = 240 :=
sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l3877_387736


namespace NUMINAMATH_CALUDE_volleyball_ticket_sales_l3877_387749

theorem volleyball_ticket_sales (total_tickets : ℕ) (jude_tickets : ℕ) (left_tickets : ℕ) 
  (h1 : total_tickets = 100)
  (h2 : jude_tickets = 16)
  (h3 : left_tickets = 40)
  : total_tickets - left_tickets - 2 * jude_tickets - jude_tickets = jude_tickets - 4 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_ticket_sales_l3877_387749


namespace NUMINAMATH_CALUDE_total_income_is_139_80_l3877_387722

/-- Represents a pastry item with its original price, discount rate, and quantity sold. -/
structure Pastry where
  name : String
  originalPrice : Float
  discountRate : Float
  quantitySold : Nat

/-- Calculates the total income generated from selling pastries after applying discounts. -/
def calculateTotalIncome (pastries : List Pastry) : Float :=
  pastries.foldl (fun acc pastry =>
    let discountedPrice := pastry.originalPrice * (1 - pastry.discountRate)
    acc + discountedPrice * pastry.quantitySold.toFloat
  ) 0

/-- Theorem stating that the total income from the given pastries is $139.80. -/
theorem total_income_is_139_80 : 
  let pastries : List Pastry := [
    { name := "Cupcakes", originalPrice := 3.00, discountRate := 0.30, quantitySold := 25 },
    { name := "Cookies", originalPrice := 2.00, discountRate := 0.45, quantitySold := 18 },
    { name := "Brownies", originalPrice := 4.00, discountRate := 0.25, quantitySold := 15 },
    { name := "Macarons", originalPrice := 1.50, discountRate := 0.50, quantitySold := 30 }
  ]
  calculateTotalIncome pastries = 139.80 := by
  sorry

end NUMINAMATH_CALUDE_total_income_is_139_80_l3877_387722


namespace NUMINAMATH_CALUDE_rectangle_area_l3877_387786

/-- Given a rectangle where the sum of width and length is half of 28, and the width is 6,
    prove that its area is 48 square units. -/
theorem rectangle_area (w l : ℝ) : w = 6 → w + l = 28 / 2 → w * l = 48 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3877_387786


namespace NUMINAMATH_CALUDE_function_value_problem_l3877_387798

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (x/2 - 1) = 2*x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 := by sorry

end NUMINAMATH_CALUDE_function_value_problem_l3877_387798


namespace NUMINAMATH_CALUDE_constant_grid_function_l3877_387719

/-- A function from integer pairs to non-negative integers -/
def GridFunction := ℤ × ℤ → ℕ

/-- The property that each value is the average of its four neighbors -/
def IsAverageOfNeighbors (f : GridFunction) : Prop :=
  ∀ x y : ℤ, 4 * f (x, y) = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)

/-- Theorem stating that if a grid function satisfies the average property, it is constant -/
theorem constant_grid_function (f : GridFunction) (h : IsAverageOfNeighbors f) :
  ∀ x₁ y₁ x₂ y₂ : ℤ, f (x₁, y₁) = f (x₂, y₂) := by
  sorry


end NUMINAMATH_CALUDE_constant_grid_function_l3877_387719


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3877_387781

def p (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3877_387781


namespace NUMINAMATH_CALUDE_alex_cake_slices_l3877_387723

theorem alex_cake_slices (total_slices : ℕ) (cakes : ℕ) : 
  cakes = 2 →
  (total_slices / 4 : ℚ) + (3 * total_slices / 4 / 3 : ℚ) + 3 + 5 = total_slices →
  total_slices / cakes = 8 := by
sorry

end NUMINAMATH_CALUDE_alex_cake_slices_l3877_387723


namespace NUMINAMATH_CALUDE_complex_sum_squares_l3877_387700

theorem complex_sum_squares (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 3) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_squares_l3877_387700


namespace NUMINAMATH_CALUDE_third_side_length_l3877_387702

/-- A scalene triangle with integer side lengths satisfying certain conditions -/
structure ScaleneTriangle where
  a : ℤ
  b : ℤ
  c : ℤ
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c
  condition : (a - 3)^2 + (b - 2)^2 = 0

/-- The third side of the triangle is either 2, 3, or 4 -/
theorem third_side_length (t : ScaleneTriangle) : t.c = 2 ∨ t.c = 3 ∨ t.c = 4 :=
  sorry

end NUMINAMATH_CALUDE_third_side_length_l3877_387702


namespace NUMINAMATH_CALUDE_max_dominos_with_room_for_one_l3877_387775

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a domino -/
structure Domino :=
  (width : Nat)
  (height : Nat)

/-- Represents a placement of dominos on a chessboard -/
def DominoPlacement := List (Nat × Nat)

/-- Function to check if a domino placement is valid -/
def isValidPlacement (board : Chessboard) (domino : Domino) (placement : DominoPlacement) : Prop :=
  sorry

/-- Function to check if there's room for one more domino -/
def hasRoomForOne (board : Chessboard) (domino : Domino) (placement : DominoPlacement) : Prop :=
  sorry

/-- The main theorem -/
theorem max_dominos_with_room_for_one (board : Chessboard) (domino : Domino) :
  board.rows = 6 →
  board.cols = 6 →
  domino.width = 1 →
  domino.height = 2 →
  (∃ (n : Nat) (placement : DominoPlacement),
    n = 11 ∧
    isValidPlacement board domino placement ∧
    placement.length = n ∧
    hasRoomForOne board domino placement) ∧
  (∀ (m : Nat) (placement : DominoPlacement),
    m > 11 →
    isValidPlacement board domino placement →
    placement.length = m →
    ¬hasRoomForOne board domino placement) :=
  by sorry

end NUMINAMATH_CALUDE_max_dominos_with_room_for_one_l3877_387775


namespace NUMINAMATH_CALUDE_price_before_increase_l3877_387797

/-- Proves that the total price before the increase was 25 pounds, given the original prices and percentage increases. -/
theorem price_before_increase 
  (candy_price : ℝ) 
  (soda_price : ℝ) 
  (candy_increase : ℝ) 
  (soda_increase : ℝ) 
  (h1 : candy_price = 10)
  (h2 : soda_price = 15)
  (h3 : candy_increase = 0.25)
  (h4 : soda_increase = 0.50) :
  candy_price + soda_price = 25 := by
  sorry

#check price_before_increase

end NUMINAMATH_CALUDE_price_before_increase_l3877_387797


namespace NUMINAMATH_CALUDE_cosine_sine_equation_l3877_387725

theorem cosine_sine_equation (x : ℝ) :
  2 * Real.cos x - 3 * Real.sin x = 4 →
  3 * Real.sin x + 2 * Real.cos x = 0 ∨ 3 * Real.sin x + 2 * Real.cos x = 8/13 :=
by sorry

end NUMINAMATH_CALUDE_cosine_sine_equation_l3877_387725


namespace NUMINAMATH_CALUDE_min_value_expression_l3877_387759

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  ∃ (min : ℝ), min = 1 ∧ ∀ z, z = 2*x^2 + 3*y^2 - 4*x*y → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3877_387759


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3877_387776

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) 
  (h1 : a 3 * a 6 = 9) (h2 : a 2 * a 4 * a 5 = 27) : a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3877_387776


namespace NUMINAMATH_CALUDE_trivia_game_score_l3877_387712

/-- Represents the score distribution in a trivia game --/
structure TriviaGame where
  total_members : Float
  absent_members : Float
  total_points : Float

/-- Calculates the score per member for a given trivia game --/
def score_per_member (game : TriviaGame) : Float :=
  game.total_points / (game.total_members - game.absent_members)

/-- Theorem: In the given trivia game scenario, each member scores 2.0 points --/
theorem trivia_game_score :
  let game := TriviaGame.mk 5.0 2.0 6.0
  score_per_member game = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_trivia_game_score_l3877_387712


namespace NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l3877_387766

-- Define the points of the rectangle and rhombus
variable (I J K L E F G H : ℝ × ℝ)

-- Define the properties of the rectangle and rhombus
def is_rectangle (I J K L : ℝ × ℝ) : Prop := sorry
def is_rhombus (E F G H : ℝ × ℝ) : Prop := sorry
def inscribed (E F G H I J K L : ℝ × ℝ) : Prop := sorry
def interior_point (P Q R : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem rhombus_in_rectangle_perimeter 
  (h_rectangle : is_rectangle I J K L)
  (h_rhombus : is_rhombus E F G H)
  (h_inscribed : inscribed E F G H I J K L)
  (h_E : interior_point E I J)
  (h_F : interior_point F J K)
  (h_G : interior_point G K L)
  (h_H : interior_point H L I)
  (h_IE : distance I E = 12)
  (h_EJ : distance E J = 25)
  (h_EG : distance E G = 35)
  (h_FH : distance F H = 42) :
  distance I J + distance J K + distance K L + distance L I = 110 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_in_rectangle_perimeter_l3877_387766


namespace NUMINAMATH_CALUDE_solve_for_s_l3877_387774

theorem solve_for_s (s t : ℚ) 
  (eq1 : 8 * s + 6 * t = 120)
  (eq2 : t - 3 = s) : 
  s = 51 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_for_s_l3877_387774


namespace NUMINAMATH_CALUDE_sequence_increasing_l3877_387704

def a (n : ℕ) : ℚ := (n - 1) / (n + 1)

theorem sequence_increasing : ∀ n : ℕ, n ≥ 1 → a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_increasing_l3877_387704


namespace NUMINAMATH_CALUDE_square_root_equation_l3877_387743

theorem square_root_equation (n : ℝ) : 
  Real.sqrt (8 + n) = 9 → n = 73 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_l3877_387743


namespace NUMINAMATH_CALUDE_cindy_pens_l3877_387783

theorem cindy_pens (initial_pens : ℕ) (mike_gives : ℕ) (sharon_receives : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 5)
  (h2 : mike_gives = 20)
  (h3 : sharon_receives = 19)
  (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gives - sharon_receives + 25 :=
by sorry

end NUMINAMATH_CALUDE_cindy_pens_l3877_387783


namespace NUMINAMATH_CALUDE_sector_arc_length_l3877_387758

/-- Given a circular sector with area 4 cm² and central angle 2 radians, 
    the length of its arc is 4 cm. -/
theorem sector_arc_length (area : ℝ) (central_angle : ℝ) (arc_length : ℝ) : 
  area = 4 → central_angle = 2 → arc_length = area / central_angle * 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3877_387758


namespace NUMINAMATH_CALUDE_proportion_not_greater_than_30_proportion_as_percentage_l3877_387779

-- Define the sample size
def sample_size : ℕ := 50

-- Define the number of data points greater than 30
def data_greater_than_30 : ℕ := 3

-- Define the proportion calculation function
def calculate_proportion (total : ℕ) (part : ℕ) : ℚ :=
  (total - part : ℚ) / total

-- Theorem statement
theorem proportion_not_greater_than_30 :
  calculate_proportion sample_size data_greater_than_30 = 47/50 :=
by
  sorry

-- Additional theorem to show the decimal representation
theorem proportion_as_percentage :
  (calculate_proportion sample_size data_greater_than_30 * 100 : ℚ) = 94 :=
by
  sorry

end NUMINAMATH_CALUDE_proportion_not_greater_than_30_proportion_as_percentage_l3877_387779


namespace NUMINAMATH_CALUDE_beta_values_l3877_387792

theorem beta_values (β : ℂ) (h1 : β ≠ 1) 
  (h2 : Complex.abs (β^3 - 1) = 3 * Complex.abs (β - 1))
  (h3 : Complex.abs (β^6 - 1) = 6 * Complex.abs (β - 1)) :
  β = Complex.I * Real.sqrt 2 ∨ β = -Complex.I * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_beta_values_l3877_387792


namespace NUMINAMATH_CALUDE_num_chords_from_nine_points_l3877_387777

/-- The number of points on the circumference of the circle -/
def num_points : ℕ := 9

/-- A function to calculate the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that the number of distinct chords from 9 points is 36 -/
theorem num_chords_from_nine_points : 
  choose_two num_points = 36 := by sorry

end NUMINAMATH_CALUDE_num_chords_from_nine_points_l3877_387777


namespace NUMINAMATH_CALUDE_min_value_expression_l3877_387755

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3877_387755


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3877_387772

theorem complex_equation_sum (a t : ℝ) (i : ℂ) : 
  i * i = -1 → a + i = (1 + 2*i) * t*i → t + a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3877_387772


namespace NUMINAMATH_CALUDE_program_output_is_66_l3877_387756

/-- A simplified representation of the program output -/
def program_output : ℕ := 66

/-- The theorem stating that the program output is 66 -/
theorem program_output_is_66 : program_output = 66 := by sorry

end NUMINAMATH_CALUDE_program_output_is_66_l3877_387756


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l3877_387713

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l3877_387713


namespace NUMINAMATH_CALUDE_remainder_problem_l3877_387780

theorem remainder_problem (n : ℤ) (h : n % 22 = 12) : (2 * n) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3877_387780


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3877_387741

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x - 3) + Real.sqrt (x - 8) = 10 → x = 30.5625 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3877_387741


namespace NUMINAMATH_CALUDE_division_problem_l3877_387738

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) : 
  dividend = 127 → divisor = 14 → remainder = 1 → 
  dividend = divisor * quotient + remainder → quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3877_387738


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3877_387730

/-- Given that a and b satisfy a + 2*b = 1, prove that the line ax + 3y + b = 0 passes through the point (1/2, -1/6) -/
theorem line_passes_through_fixed_point (a b : ℝ) (h : a + 2*b = 1) :
  a * (1/2 : ℝ) + 3 * (-1/6 : ℝ) + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3877_387730


namespace NUMINAMATH_CALUDE_infinite_even_k_composite_sum_l3877_387790

theorem infinite_even_k_composite_sum (t : ℕ+) (p : ℕ) :
  let k := 30 * t + 26
  (∃ n : ℕ+, k = 2 * n) ∧ 
  (Nat.Prime p → ∃ (m n : ℕ+), p^2 + k = m * n ∧ m ≠ 1 ∧ n ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_even_k_composite_sum_l3877_387790


namespace NUMINAMATH_CALUDE_total_cost_is_87_60_l3877_387716

/-- Calculate the total cost of T-shirts bought by Dave -/
def calculate_total_cost : ℝ :=
  let white_packs := 3
  let blue_packs := 2
  let red_packs := 4
  let green_packs := 1

  let white_price := 12
  let blue_price := 8
  let red_price := 10
  let green_price := 6

  let white_discount := 0.10
  let blue_discount := 0.05
  let red_discount := 0.15
  let green_discount := 0

  let white_cost := white_packs * white_price * (1 - white_discount)
  let blue_cost := blue_packs * blue_price * (1 - blue_discount)
  let red_cost := red_packs * red_price * (1 - red_discount)
  let green_cost := green_packs * green_price * (1 - green_discount)

  white_cost + blue_cost + red_cost + green_cost

/-- The total cost of T-shirts bought by Dave is $87.60 -/
theorem total_cost_is_87_60 : calculate_total_cost = 87.60 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_87_60_l3877_387716


namespace NUMINAMATH_CALUDE_vladimir_investment_opportunity_l3877_387778

/-- Represents the value of 1 kg of buckwheat in rubles -/
def buckwheat_value : ℝ := 85

/-- Represents the initial price of 1 kg of buckwheat in rubles -/
def initial_price : ℝ := 70

/-- Calculates the value after a one-year deposit at the given rate -/
def one_year_deposit (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate)

/-- Calculates the value after a two-year deposit at the given rate -/
def two_year_deposit (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * (1 + rate) * (1 + rate)

/-- Represents the annual deposit rate for 2015 -/
def rate_2015 : ℝ := 0.16

/-- Represents the annual deposit rate for 2016 -/
def rate_2016 : ℝ := 0.10

/-- Represents the two-year deposit rate starting from 2015 -/
def rate_2015_2016 : ℝ := 0.15

theorem vladimir_investment_opportunity : 
  let option1 := one_year_deposit (one_year_deposit initial_price rate_2015) rate_2016
  let option2 := two_year_deposit initial_price rate_2015_2016
  max option1 option2 > buckwheat_value := by sorry

end NUMINAMATH_CALUDE_vladimir_investment_opportunity_l3877_387778


namespace NUMINAMATH_CALUDE_sum_first_tenth_l3877_387744

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  sum_4_7 : a 4 + a 7 = 2
  prod_5_6 : a 5 * a 6 = -8

/-- The sum of the first and tenth terms of the geometric sequence is -7 -/
theorem sum_first_tenth (seq : GeometricSequence) : seq.a 1 + seq.a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_tenth_l3877_387744


namespace NUMINAMATH_CALUDE_division_problem_l3877_387784

theorem division_problem : (-1/24) / (1/3 - 1/6 + 3/8) = -1/13 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3877_387784


namespace NUMINAMATH_CALUDE_songcheng_visitors_l3877_387717

/-- Calculates the total number of visitors to Hangzhou Songcheng on Sunday -/
def total_visitors (morning_visitors : ℕ) (noon_departures : ℕ) (afternoon_increase : ℕ) : ℕ :=
  morning_visitors + (noon_departures + afternoon_increase)

/-- Theorem stating the total number of visitors to Hangzhou Songcheng on Sunday -/
theorem songcheng_visitors :
  total_visitors 500 119 138 = 757 := by
  sorry

end NUMINAMATH_CALUDE_songcheng_visitors_l3877_387717


namespace NUMINAMATH_CALUDE_chocolate_count_l3877_387708

/-- The number of boxes of chocolates -/
def num_boxes : ℕ := 6

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 500

/-- The total number of chocolate pieces -/
def total_pieces : ℕ := num_boxes * pieces_per_box

theorem chocolate_count : total_pieces = 3000 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_count_l3877_387708


namespace NUMINAMATH_CALUDE_sequence_contains_24_l3877_387768

theorem sequence_contains_24 : ∃ n : ℕ+, n * (n + 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_sequence_contains_24_l3877_387768


namespace NUMINAMATH_CALUDE_steven_more_apples_l3877_387793

/-- The number of apples Steven has -/
def steven_apples : ℕ := 19

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 15

/-- The difference between Steven's apples and peaches -/
def apple_peach_difference : ℤ := steven_apples - steven_peaches

theorem steven_more_apples : apple_peach_difference = 4 := by
  sorry

end NUMINAMATH_CALUDE_steven_more_apples_l3877_387793


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3877_387789

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l3877_387789


namespace NUMINAMATH_CALUDE_count_not_divisible_9999_l3877_387760

def count_not_divisible (n : ℕ) : ℕ :=
  n + 1 - (
    (n / 3 + 1) + (n / 5 + 1) + (n / 7 + 1) -
    (n / 15 + 1) - (n / 21 + 1) - (n / 35 + 1) +
    (n / 105 + 1)
  )

theorem count_not_divisible_9999 :
  count_not_divisible 9999 = 4571 := by
sorry

end NUMINAMATH_CALUDE_count_not_divisible_9999_l3877_387760


namespace NUMINAMATH_CALUDE_quadratic_solution_and_max_product_l3877_387734

-- Define the quadratic inequality
def quadratic_inequality (x m : ℝ) : Prop := x^2 - 3*x + m < 0

-- Define the solution set
def solution_set (x n : ℝ) : Prop := 1 < x ∧ x < n

-- Define the constraint for a and b
def constraint (m n a b : ℝ) : Prop := m*a + 2*n*b = 3

-- Theorem statement
theorem quadratic_solution_and_max_product :
  ∃ (m n : ℝ),
    (∀ x, quadratic_inequality x m ↔ solution_set x n) ∧
    (m = 2 ∧ n = 2) ∧
    (∀ a b : ℝ, a > 0 → b > 0 → constraint m n a b →
      a * b ≤ 9/32 ∧ ∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ constraint m n a₀ b₀ ∧ a₀ * b₀ = 9/32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_and_max_product_l3877_387734


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3877_387787

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (a 1 + a 2 = 40) →
  (a 3 + a 4 = 60) →
  (a 7 + a 8 = 135) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3877_387787


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3877_387740

/-- For a quadratic equation with two equal real roots, the value of k is ±6 --/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x + 9 = 0 ∧ 
   ∀ y : ℝ, y^2 - k*y + 9 = 0 → y = x) →
  k = 6 ∨ k = -6 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3877_387740


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l3877_387754

theorem sqrt_difference_inequality (n : ℝ) (hn : n ≥ 0) :
  Real.sqrt (n + 2) - Real.sqrt (n + 1) < Real.sqrt (n + 1) - Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l3877_387754


namespace NUMINAMATH_CALUDE_last_disc_is_blue_l3877_387796

/-- Represents the color of a disc --/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Represents the state of the bag --/
structure BagState where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Initial state of the bag --/
def initial_state : BagState :=
  { red := 7, blue := 8, yellow := 9 }

/-- Represents the rules for drawing and replacing discs --/
def draw_and_replace (state : BagState) : BagState :=
  sorry

/-- Represents the process of repeatedly drawing and replacing discs until the end condition is met --/
def process (state : BagState) : BagState :=
  sorry

/-- Theorem stating that the last remaining disc(s) will be blue --/
theorem last_disc_is_blue :
  ∃ (final_state : BagState), process initial_state = final_state ∧ 
  final_state.blue > 0 ∧ final_state.red = 0 ∧ final_state.yellow = 0 :=
sorry

end NUMINAMATH_CALUDE_last_disc_is_blue_l3877_387796


namespace NUMINAMATH_CALUDE_bank_transfer_theorem_l3877_387785

def calculate_final_balance (initial_balance : ℚ) (transfer1 : ℚ) (transfer2 : ℚ) (service_charge_rate : ℚ) : ℚ :=
  let service_charge1 := transfer1 * service_charge_rate
  let service_charge2 := transfer2 * service_charge_rate
  initial_balance - (transfer1 + service_charge1) - service_charge2

theorem bank_transfer_theorem (initial_balance : ℚ) (transfer1 : ℚ) (transfer2 : ℚ) (service_charge_rate : ℚ) 
  (h1 : initial_balance = 400)
  (h2 : transfer1 = 90)
  (h3 : transfer2 = 60)
  (h4 : service_charge_rate = 2/100) :
  calculate_final_balance initial_balance transfer1 transfer2 service_charge_rate = 307 := by
  sorry

end NUMINAMATH_CALUDE_bank_transfer_theorem_l3877_387785


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_l3877_387752

theorem product_of_sums_equals_difference (n : ℕ) :
  (5 + 1) * (5^2 + 1^2) * (5^4 + 1^4) * (5^8 + 1^8) * (5^16 + 1^16) * (5^32 + 1^32) * (5^64 + 1^64) = 5^128 - 1^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_l3877_387752


namespace NUMINAMATH_CALUDE_marks_wage_proof_l3877_387732

/-- Mark's hourly wage before the raise -/
def pre_raise_wage : ℝ := 40

/-- Mark's weekly work hours -/
def weekly_hours : ℝ := 40

/-- Mark's raise percentage -/
def raise_percentage : ℝ := 0.05

/-- Mark's weekly expenses -/
def weekly_expenses : ℝ := 700

/-- Mark's leftover money per week -/
def weekly_leftover : ℝ := 980

theorem marks_wage_proof :
  pre_raise_wage * weekly_hours * (1 + raise_percentage) = weekly_expenses + weekly_leftover :=
by sorry

end NUMINAMATH_CALUDE_marks_wage_proof_l3877_387732


namespace NUMINAMATH_CALUDE_age_puzzle_l3877_387761

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 32) (h2 : 4 * (A + x) - 4 * (A - 4) = A) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3877_387761


namespace NUMINAMATH_CALUDE_smallest_value_complex_expression_l3877_387747

theorem smallest_value_complex_expression (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
    ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
      m ≤ Complex.abs (x + y*ω + z*ω^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_expression_l3877_387747


namespace NUMINAMATH_CALUDE_product_equals_simplified_fraction_l3877_387750

/-- The repeating decimal 0.456̅ as a rational number -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of 0.456̅ and 8 -/
def product : ℚ := repeating_decimal * 8

/-- Theorem stating that the product of 0.456̅ and 8 is equal to 1216/333 -/
theorem product_equals_simplified_fraction : product = 1216 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_simplified_fraction_l3877_387750


namespace NUMINAMATH_CALUDE_student_sequences_count_l3877_387711

/-- The number of ways to select 5 students from a group of 15 students,
    where order matters and no student can be selected more than once. -/
def student_sequences : ℕ :=
  Nat.descFactorial 15 5

theorem student_sequences_count : student_sequences = 360360 := by
  sorry

end NUMINAMATH_CALUDE_student_sequences_count_l3877_387711


namespace NUMINAMATH_CALUDE_birds_on_fence_l3877_387767

theorem birds_on_fence (num_birds : ℕ) (h : num_birds = 20) : 
  2 * num_birds + 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3877_387767


namespace NUMINAMATH_CALUDE_min_value_trig_function_min_value_trig_function_achievable_l3877_387705

theorem min_value_trig_function (α : Real) (h : α ∈ Set.Ioo 0 (π / 2)) :
  1 / (Real.sin α)^2 + 3 / (Real.cos α)^2 ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

theorem min_value_trig_function_achievable :
  ∃ α : Real, α ∈ Set.Ioo 0 (π / 2) ∧
  1 / (Real.sin α)^2 + 3 / (Real.cos α)^2 = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_min_value_trig_function_achievable_l3877_387705


namespace NUMINAMATH_CALUDE_exists_modular_inverse_l3877_387788

theorem exists_modular_inverse :
  ∃ n : ℤ, 21 * n ≡ 1 [ZMOD 74] := by
  sorry

end NUMINAMATH_CALUDE_exists_modular_inverse_l3877_387788
