import Mathlib

namespace NUMINAMATH_CALUDE_seven_double_prime_l3239_323918

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Theorem statement
theorem seven_double_prime : prime (prime 7) = 51 := by
  sorry

end NUMINAMATH_CALUDE_seven_double_prime_l3239_323918


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l3239_323999

/-- The number of ways to allocate volunteers to projects -/
def allocate_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * (k.factorial)

/-- Theorem stating that allocating 5 volunteers to 4 projects results in 240 schemes -/
theorem volunteer_allocation_schemes :
  allocate_volunteers 5 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l3239_323999


namespace NUMINAMATH_CALUDE_caiden_roofing_cost_l3239_323997

-- Define the parameters
def total_feet : ℕ := 300
def cost_per_foot : ℚ := 8
def discount_rate : ℚ := 0.1
def shipping_fee : ℚ := 150
def sales_tax_rate : ℚ := 0.05
def free_feet : ℕ := 250

-- Define the calculation steps
def paid_feet : ℕ := total_feet - free_feet
def base_cost : ℚ := paid_feet * cost_per_foot
def discounted_cost : ℚ := base_cost * (1 - discount_rate)
def cost_with_shipping : ℚ := discounted_cost + shipping_fee
def total_cost : ℚ := cost_with_shipping * (1 + sales_tax_rate)

-- Theorem to prove
theorem caiden_roofing_cost :
  total_cost = 535.5 := by sorry

end NUMINAMATH_CALUDE_caiden_roofing_cost_l3239_323997


namespace NUMINAMATH_CALUDE_shopping_trip_cost_l3239_323933

/-- Calculates the total cost of a shopping trip including discounts, taxes, and fees -/
def calculate_total_cost (items : List (ℕ × ℚ)) (discount_rate : ℚ) (sales_tax_rate : ℚ) (local_tax_rate : ℚ) (sustainability_fee : ℚ) : ℚ :=
  let total_before_discount := (items.map (λ (q, p) => q * p)).sum
  let discounted_total := total_before_discount * (1 - discount_rate)
  let total_tax_rate := sales_tax_rate + local_tax_rate
  let tax_amount := discounted_total * total_tax_rate
  let total_with_tax := discounted_total + tax_amount
  total_with_tax + sustainability_fee

theorem shopping_trip_cost :
  let items := [(3, 18), (2, 11), (4, 22), (6, 9), (5, 14), (2, 30), (3, 25)]
  let discount_rate := 0.15
  let sales_tax_rate := 0.05
  let local_tax_rate := 0.02
  let sustainability_fee := 5
  calculate_total_cost items discount_rate sales_tax_rate local_tax_rate sustainability_fee = 389.72 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_cost_l3239_323933


namespace NUMINAMATH_CALUDE_inequality_solution_l3239_323948

theorem inequality_solution (x : ℝ) : 
  -1 < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧ 
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < 1 ↔ 
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3239_323948


namespace NUMINAMATH_CALUDE_problem_solution_l3239_323906

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 4*x + 4/x + 1/x^2 = 35)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3239_323906


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3239_323901

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3239_323901


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3239_323998

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P
def P : ℝ × ℝ := (-2, -2)

-- Define a tangent point
def TangentPoint (x y : ℝ) : Prop :=
  Circle x y ∧ ((x + 2) * x + (y + 2) * y = 0)

-- Theorem statement
theorem tangent_line_equation :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    TangentPoint x₁ y₁ → TangentPoint x₂ y₂ →
    (2 * x₁ + 2 * y₁ + 1 = 0) ∧ (2 * x₂ + 2 * y₂ + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3239_323998


namespace NUMINAMATH_CALUDE_number_division_problem_l3239_323976

theorem number_division_problem (x : ℝ) : (x / 5 = 75 + x / 6) ↔ (x = 2250) := by sorry

end NUMINAMATH_CALUDE_number_division_problem_l3239_323976


namespace NUMINAMATH_CALUDE_first_week_rate_is_18_l3239_323929

/-- The daily rate for the first week in a student youth hostel -/
def first_week_rate : ℝ := 18

/-- The daily rate for additional weeks in a student youth hostel -/
def additional_week_rate : ℝ := 14

/-- The total number of days stayed -/
def total_days : ℕ := 23

/-- The total cost for the stay -/
def total_cost : ℝ := 350

/-- Theorem stating that the daily rate for the first week is $18.00 -/
theorem first_week_rate_is_18 :
  first_week_rate * 7 + additional_week_rate * (total_days - 7) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_first_week_rate_is_18_l3239_323929


namespace NUMINAMATH_CALUDE_relative_rate_of_change_cubic_parabola_l3239_323905

/-- For a point (x, y) on the cubic parabola 12y = x^3, the relative rate of change between y and x is x^2/4 -/
theorem relative_rate_of_change_cubic_parabola (x y : ℝ) (h : 12 * y = x^3) :
  ∃ (dx dy : ℝ), dy / dx = x^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_relative_rate_of_change_cubic_parabola_l3239_323905


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3239_323953

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 5 ∪ {5} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3239_323953


namespace NUMINAMATH_CALUDE_problem_solution_l3239_323954

theorem problem_solution : 
  (27 / 8) ^ (-1/3 : ℝ) + Real.log 3 / Real.log 2 * Real.log 4 / Real.log 3 + 
  Real.log 2 / Real.log 10 + Real.log 50 / Real.log 10 = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3239_323954


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3239_323922

theorem quadratic_one_solution_sum (b : ℝ) : 
  let equation := fun (x : ℝ) => 3 * x^2 + b * x + 6 * x + 14
  let discriminant := (b + 6)^2 - 4 * 3 * 14
  (∃! x, equation x = 0) → 
  (∃ b₁ b₂, b = b₁ ∨ b = b₂) ∧ (b₁ + b₂ = -12) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3239_323922


namespace NUMINAMATH_CALUDE_max_three_cards_l3239_323902

theorem max_three_cards (total_cards : ℕ) (sum : ℕ) (cards_chosen : ℕ) : 
  total_cards = 10 →
  sum = 31 →
  cards_chosen = 8 →
  ∃ (threes fours fives : ℕ),
    threes + fours + fives = cards_chosen ∧
    3 * threes + 4 * fours + 5 * fives = sum ∧
    threes ≤ 4 ∧
    ∀ (t f v : ℕ), 
      t + f + v = cards_chosen →
      3 * t + 4 * f + 5 * v = sum →
      t ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_three_cards_l3239_323902


namespace NUMINAMATH_CALUDE_value_of_S_l3239_323935

/-- Given S = 6 × 10000 + 5 × 1000 + 4 × 10 + 3 × 1, prove that S = 65043 -/
theorem value_of_S : 
  let S := 6 * 10000 + 5 * 1000 + 4 * 10 + 3 * 1
  S = 65043 := by
  sorry

end NUMINAMATH_CALUDE_value_of_S_l3239_323935


namespace NUMINAMATH_CALUDE_a_cubed_congruence_l3239_323988

theorem a_cubed_congruence (n : ℕ+) (a : ℤ) 
  (h1 : a * a ≡ 1 [ZMOD n])
  (h2 : a ≡ -1 [ZMOD n]) :
  a^3 ≡ -1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_a_cubed_congruence_l3239_323988


namespace NUMINAMATH_CALUDE_min_point_of_translated_abs_value_l3239_323970

-- Define the function representing the translated graph
def f (x : ℝ) : ℝ := |x - 3| - 1

-- Theorem stating that the minimum point of the graph is (3, -1)
theorem min_point_of_translated_abs_value :
  ∀ x : ℝ, f x ≥ f 3 ∧ f 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_point_of_translated_abs_value_l3239_323970


namespace NUMINAMATH_CALUDE_intersection_M_N_l3239_323931

def M : Set ℝ := {1, 3, 4}
def N : Set ℝ := {x : ℝ | x^2 - 4*x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3239_323931


namespace NUMINAMATH_CALUDE_question_selection_ways_eq_13838400_l3239_323978

/-- The number of ways to select questions from a question paper with three parts -/
def questionSelectionWays : ℕ :=
  let partA := Nat.choose 12 8
  let partB := Nat.choose 10 5
  let partC := Nat.choose 8 3
  partA * partB * partC

/-- Theorem stating the correct number of ways to select questions -/
theorem question_selection_ways_eq_13838400 : questionSelectionWays = 13838400 := by
  sorry

end NUMINAMATH_CALUDE_question_selection_ways_eq_13838400_l3239_323978


namespace NUMINAMATH_CALUDE_S_periodic_l3239_323968

def S (x y z : ℤ) : ℤ × ℤ × ℤ := (x*y - x*z, y*z - y*x, z*x - z*y)

def S_power (n : ℕ) (a b c : ℤ) : ℤ × ℤ × ℤ :=
  match n with
  | 0 => (a, b, c)
  | n + 1 => S (S_power n a b c).1 (S_power n a b c).2.1 (S_power n a b c).2.2

def congruent_triple (u v : ℤ × ℤ × ℤ) (m : ℤ) : Prop :=
  u.1 % m = v.1 % m ∧ u.2.1 % m = v.2.1 % m ∧ u.2.2 % m = v.2.2 % m

theorem S_periodic (a b c : ℤ) (h : a * b * c > 1) :
  ∃ (n₀ k : ℕ), 0 < k ∧ k ≤ a * b * c ∧
  ∀ n ≥ n₀, congruent_triple (S_power (n + k) a b c) (S_power n a b c) (a * b * c) :=
sorry

end NUMINAMATH_CALUDE_S_periodic_l3239_323968


namespace NUMINAMATH_CALUDE_quality_related_to_renovation_probability_two_qualified_l3239_323917

-- Define the data from the table
def qualified_before : ℕ := 60
def substandard_before : ℕ := 40
def qualified_after : ℕ := 80
def substandard_after : ℕ := 20
def total_sample : ℕ := 200

-- Define the K^2 statistic
def K_squared (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Theorem for part 1
theorem quality_related_to_renovation :
  K_squared qualified_before substandard_before qualified_after substandard_after > critical_value := by
  sorry

-- Theorem for part 2
theorem probability_two_qualified :
  (Nat.choose 3 2 : ℚ) / (Nat.choose 5 2 : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_quality_related_to_renovation_probability_two_qualified_l3239_323917


namespace NUMINAMATH_CALUDE_total_morning_afternoon_emails_l3239_323977

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 5

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- Theorem: The total number of emails Jack received in the morning and afternoon is 13 -/
theorem total_morning_afternoon_emails : morning_emails + afternoon_emails = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_morning_afternoon_emails_l3239_323977


namespace NUMINAMATH_CALUDE_boat_current_rate_l3239_323938

/-- Proves that the rate of the current is 4 km/hr given the conditions of the boat problem -/
theorem boat_current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 12 →
  downstream_distance = 4.8 →
  downstream_time = 18 / 60 →
  ∃ current_rate : ℝ,
    current_rate = 4 ∧
    downstream_distance = (boat_speed + current_rate) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_boat_current_rate_l3239_323938


namespace NUMINAMATH_CALUDE_factory_month_days_l3239_323920

/-- The number of days in a month for a computer factory -/
def days_in_month (computers_per_month : ℕ) (computers_per_half_hour : ℕ) : ℕ :=
  computers_per_month * 30 / (computers_per_half_hour * 24 * 2)

/-- Theorem: Given the production rate, the number of days in the month is 28 -/
theorem factory_month_days :
  days_in_month 4032 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_factory_month_days_l3239_323920


namespace NUMINAMATH_CALUDE_equilateral_triangle_exists_l3239_323947

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  vertex : Point
  base1 : Point
  base2 : Point

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if a triangle is equilateral
def isEquilateral (t : EquilateralTriangle) : Prop :=
  let d1 := ((t.vertex.x - t.base1.x)^2 + (t.vertex.y - t.base1.y)^2)
  let d2 := ((t.vertex.x - t.base2.x)^2 + (t.vertex.y - t.base2.y)^2)
  let d3 := ((t.base1.x - t.base2.x)^2 + (t.base1.y - t.base2.y)^2)
  d1 = d2 ∧ d2 = d3

-- Theorem statement
theorem equilateral_triangle_exists (P : Point) (l : Line) :
  ∃ (t : EquilateralTriangle), t.vertex = P ∧ 
    isPointOnLine t.base1 l ∧ isPointOnLine t.base2 l ∧ 
    isEquilateral t :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_exists_l3239_323947


namespace NUMINAMATH_CALUDE_seth_yogurt_purchase_l3239_323957

theorem seth_yogurt_purchase (ice_cream_cartons : ℕ) (ice_cream_cost : ℕ) (yogurt_cost : ℕ) (difference : ℕ) :
  ice_cream_cartons = 20 →
  ice_cream_cost = 6 →
  yogurt_cost = 1 →
  ice_cream_cartons * ice_cream_cost = difference + yogurt_cost * (ice_cream_cartons * ice_cream_cost - difference) / yogurt_cost →
  (ice_cream_cartons * ice_cream_cost - difference) / yogurt_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_seth_yogurt_purchase_l3239_323957


namespace NUMINAMATH_CALUDE_smoking_chronic_bronchitis_relationship_l3239_323961

-- Define the confidence level
def confidence_level : Real := 0.99

-- Define the relationship between smoking and chronic bronchitis
def smoking_related_to_chronic_bronchitis : Prop := True

-- Define a sample of smokers
def sample_size : Nat := 100

-- Define the possibility of no chronic bronchitis cases in the sample
def possible_no_cases : Prop := True

-- Theorem statement
theorem smoking_chronic_bronchitis_relationship 
  (h1 : confidence_level > 0.99)
  (h2 : smoking_related_to_chronic_bronchitis) :
  possible_no_cases := by
  sorry

end NUMINAMATH_CALUDE_smoking_chronic_bronchitis_relationship_l3239_323961


namespace NUMINAMATH_CALUDE_robot_path_area_l3239_323944

/-- A type representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a closed path on a 2D plane -/
structure ClosedPath where
  vertices : List Point

/-- Function to calculate the area of a closed path -/
noncomputable def areaOfClosedPath (path : ClosedPath) : ℝ :=
  sorry

/-- The specific closed path followed by the robot -/
def robotPath : ClosedPath :=
  sorry

/-- Theorem stating that the area of the robot's path is 13√3/4 -/
theorem robot_path_area :
  areaOfClosedPath robotPath = (13 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_robot_path_area_l3239_323944


namespace NUMINAMATH_CALUDE_ribbon_length_proof_l3239_323921

theorem ribbon_length_proof (length1 length2 : ℕ) : 
  length1 = 8 →
  (∃ (piece_length : ℕ), piece_length > 0 ∧ 
    length1 % piece_length = 0 ∧ 
    length2 % piece_length = 0 ∧
    ∀ (l : ℕ), l > piece_length → (length1 % l ≠ 0 ∨ length2 % l ≠ 0)) →
  length2 = 8 := by
sorry

end NUMINAMATH_CALUDE_ribbon_length_proof_l3239_323921


namespace NUMINAMATH_CALUDE_square_division_and_triangle_area_l3239_323951

/-- The area of the remaining part after cutting off squares from a unit square -/
def S (n : ℕ) : ℚ :=
  (n + 1 : ℚ) / (2 * n)

/-- The area of triangle ABP formed by the intersection of y = (1/2)x and y = 1/(2x) -/
def triangle_area (n : ℕ) : ℚ :=
  1 / 2 + (1 / 2) * (1 / 2)

theorem square_division_and_triangle_area (n : ℕ) (h : n ≥ 2) :
  S n = (n + 1 : ℚ) / (2 * n) ∧ 
  triangle_area n = 1 ∧
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |triangle_area n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_square_division_and_triangle_area_l3239_323951


namespace NUMINAMATH_CALUDE_M_inter_compl_N_l3239_323959

/-- The set M defined by the square root function -/
def M : Set ℝ := {x | ∃ y, y = Real.sqrt x}

/-- The set N defined by a quadratic inequality -/
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}

/-- The theorem stating the intersection of M and the complement of N -/
theorem M_inter_compl_N : M ∩ (Set.univ \ N) = {x | 0 ≤ x ∧ x < 2 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_M_inter_compl_N_l3239_323959


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3239_323987

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  B = π / 4 →
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b / Real.sin B = c / Real.sin C →
  A + B + C = π →
  A = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3239_323987


namespace NUMINAMATH_CALUDE_max_dot_product_in_trapezoid_l3239_323995

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a point is inside or on the boundary of a trapezoid -/
def isInTrapezoid (t : Trapezoid) (p : Point) : Prop := sorry

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Main theorem -/
theorem max_dot_product_in_trapezoid (t : Trapezoid) :
  t.A = Point.mk 0 0 →
  t.B = Point.mk 3 0 →
  t.C = Point.mk 2 2 →
  t.D = Point.mk 0 2 →
  let N := Point.mk 1 2
  ∀ M : Point, isInTrapezoid t M →
  dotProduct (Point.mk (M.x - t.A.x) (M.y - t.A.y)) (Point.mk (N.x - t.A.x) (N.y - t.A.y)) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_dot_product_in_trapezoid_l3239_323995


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3239_323989

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 48)
  (sum_ca : c + a = 60) :
  a + b + c = 71.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3239_323989


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_near_twin_primes_divisible_by_240_l3239_323928

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q = p + 2

theorem product_of_consecutive_integers_near_twin_primes_divisible_by_240 
  (p : ℕ) (h1 : p > 7) (h2 : are_twin_primes p (p + 2)) : 
  240 ∣ ((p - 1) * p * (p + 1)) :=
sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_near_twin_primes_divisible_by_240_l3239_323928


namespace NUMINAMATH_CALUDE_division_with_remainder_l3239_323907

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 127 * q + r ∧ 0 ≤ r ∧ r < 127 ∧ r = 51 := by
  sorry

end NUMINAMATH_CALUDE_division_with_remainder_l3239_323907


namespace NUMINAMATH_CALUDE_divisibility_condition_l3239_323926

theorem divisibility_condition (a b : ℕ+) :
  (a.val * b.val^2 + b.val + 7) ∣ (a.val^2 * b.val + a.val + b.val) →
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3239_323926


namespace NUMINAMATH_CALUDE_paiges_math_problems_l3239_323934

theorem paiges_math_problems (total_problems math_problems science_problems finished_problems left_problems : ℕ) :
  science_problems = 12 →
  finished_problems = 44 →
  left_problems = 11 →
  total_problems = math_problems + science_problems →
  total_problems = finished_problems + left_problems →
  math_problems = 43 := by
sorry

end NUMINAMATH_CALUDE_paiges_math_problems_l3239_323934


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3239_323963

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3239_323963


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3239_323950

theorem quadratic_two_roots (c : ℝ) (h : c < 4) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ + c = 0 ∧ x₂^2 - 4*x₂ + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3239_323950


namespace NUMINAMATH_CALUDE_polyhedron_volume_theorem_l3239_323981

/-- A polyhedron consisting of a prism and two pyramids -/
structure Polyhedron where
  prism_volume : ℝ
  pyramid_volume : ℝ
  prism_volume_eq : prism_volume = Real.sqrt 2 - 1
  pyramid_volume_eq : pyramid_volume = 1 / 6

/-- The total volume of the polyhedron -/
def total_volume (p : Polyhedron) : ℝ :=
  p.prism_volume + 2 * p.pyramid_volume

theorem polyhedron_volume_theorem (p : Polyhedron) :
  total_volume p = Real.sqrt 2 - 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_theorem_l3239_323981


namespace NUMINAMATH_CALUDE_sophie_total_spent_l3239_323990

def cupcake_quantity : ℕ := 5
def cupcake_price : ℚ := 2

def doughnut_quantity : ℕ := 6
def doughnut_price : ℚ := 1

def apple_pie_slice_quantity : ℕ := 4
def apple_pie_slice_price : ℚ := 2

def cookie_quantity : ℕ := 15
def cookie_price : ℚ := 0.60

def total_spent : ℚ := cupcake_quantity * cupcake_price + 
                        doughnut_quantity * doughnut_price + 
                        apple_pie_slice_quantity * apple_pie_slice_price + 
                        cookie_quantity * cookie_price

theorem sophie_total_spent : total_spent = 33 := by
  sorry

end NUMINAMATH_CALUDE_sophie_total_spent_l3239_323990


namespace NUMINAMATH_CALUDE_jason_current_is_sum_jason_has_63_dollars_l3239_323972

/-- Represents the money situation for Fred and Jason --/
structure MoneySituation where
  fred_initial : ℕ
  jason_initial : ℕ
  fred_current : ℕ
  jason_earned : ℕ

/-- Calculates Jason's current amount of money --/
def jason_current (s : MoneySituation) : ℕ := s.jason_initial + s.jason_earned

/-- Theorem stating that Jason's current amount is the sum of his initial and earned amounts --/
theorem jason_current_is_sum (s : MoneySituation) :
  jason_current s = s.jason_initial + s.jason_earned := by sorry

/-- The specific money situation from the problem --/
def problem_situation : MoneySituation :=
  { fred_initial := 49
  , jason_initial := 3
  , fred_current := 112
  , jason_earned := 60 }

/-- Theorem proving that Jason now has 63 dollars --/
theorem jason_has_63_dollars :
  jason_current problem_situation = 63 := by sorry

end NUMINAMATH_CALUDE_jason_current_is_sum_jason_has_63_dollars_l3239_323972


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3239_323969

theorem min_value_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + 3*c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ + 2*b₀ + 3*c₀ = 1 ∧ 1/a₀ + 2/b₀ + 3/c₀ = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3239_323969


namespace NUMINAMATH_CALUDE_second_player_wins_12_and_11_l3239_323915

/-- Represents the state of the daisy game -/
inductive DaisyState
  | petals (n : Nat)

/-- Represents a move in the daisy game -/
inductive DaisyMove
  | remove_one
  | remove_two

/-- Defines a valid move in the daisy game -/
def valid_move (state : DaisyState) (move : DaisyMove) : Prop :=
  match state, move with
  | DaisyState.petals n, DaisyMove.remove_one => n ≥ 1
  | DaisyState.petals n, DaisyMove.remove_two => n ≥ 2

/-- Applies a move to the current state -/
def apply_move (state : DaisyState) (move : DaisyMove) : DaisyState :=
  match state, move with
  | DaisyState.petals n, DaisyMove.remove_one => DaisyState.petals (n - 1)
  | DaisyState.petals n, DaisyMove.remove_two => DaisyState.petals (n - 2)

/-- Defines a winning strategy for the second player -/
def second_player_wins (initial_petals : Nat) : Prop :=
  ∀ (first_move : DaisyMove),
    valid_move (DaisyState.petals initial_petals) first_move →
    ∃ (strategy : DaisyState → DaisyMove),
      (∀ (state : DaisyState), valid_move state (strategy state)) ∧
      (∀ (game : Nat → DaisyState),
        game 0 = apply_move (DaisyState.petals initial_petals) first_move →
        (∀ n, game (n + 1) = apply_move (game n) (strategy (game n))) →
        ∃ k, ¬∃ move, valid_move (game k) move)

/-- The main theorem stating that the second player wins for both 12 and 11 initial petals -/
theorem second_player_wins_12_and_11 :
  second_player_wins 12 ∧ second_player_wins 11 := by sorry

end NUMINAMATH_CALUDE_second_player_wins_12_and_11_l3239_323915


namespace NUMINAMATH_CALUDE_rectangle_area_l3239_323979

/-- The area of a rectangle with perimeter 200 cm, which can be divided into five identical squares -/
theorem rectangle_area (side : ℝ) (h1 : side > 0) (h2 : 12 * side = 200) : 
  5 * side^2 = 12500 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3239_323979


namespace NUMINAMATH_CALUDE_largest_divisible_n_l3239_323974

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 8) ∣ (m^3 + 64))) ∧ 
  ((n + 8) ∣ (n^3 + 64)) ∧ 
  n = 440 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l3239_323974


namespace NUMINAMATH_CALUDE_right_triangle_semicircles_l3239_323949

theorem right_triangle_semicircles (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let pr := Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)
  let qr := Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2)
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 →  -- right angle at Q
  (1/2) * Real.pi * (pq/2)^2 = 50 * Real.pi →  -- area of semicircle on PQ
  Real.pi * (pr/2) = 18 * Real.pi →  -- circumference of semicircle on PR
  qr/2 = 20.6 ∧  -- radius of semicircle on QR
  ∃ (C : ℝ × ℝ), (C.1 - P.1)^2 + (C.2 - P.2)^2 = (pr/2)^2 ∧
                 (C.1 - R.1)^2 + (C.2 - R.2)^2 = (pr/2)^2 ∧
                 (C.1 - Q.1) * (R.1 - P.1) + (C.2 - Q.2) * (R.2 - P.2) = 0  -- 90° angle at Q in semicircle on PR
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_semicircles_l3239_323949


namespace NUMINAMATH_CALUDE_cubic_increasing_implies_positive_a_l3239_323993

/-- A cubic function f(x) = ax^3 + x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x

/-- The property of f being increasing on all real numbers -/
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: If f(x) = ax^3 + x is increasing on all real numbers, then a > 0 -/
theorem cubic_increasing_implies_positive_a (a : ℝ) :
  increasing_on_reals (f a) → a > 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_increasing_implies_positive_a_l3239_323993


namespace NUMINAMATH_CALUDE_cantaloupe_total_l3239_323956

theorem cantaloupe_total (fred_cantaloupes tim_cantaloupes : ℕ) 
  (h1 : fred_cantaloupes = 38) 
  (h2 : tim_cantaloupes = 44) : 
  fred_cantaloupes + tim_cantaloupes = 82 := by
sorry

end NUMINAMATH_CALUDE_cantaloupe_total_l3239_323956


namespace NUMINAMATH_CALUDE_events_A_D_independent_l3239_323955

-- Define the sample space
def Ω : Type := Fin 6 × Fin 6

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Define events A and D
def A : Set Ω := {ω | Odd ω.1}
def D : Set Ω := {ω | ω.1 + ω.2 = 7}

-- State the theorem
theorem events_A_D_independent : 
  P (A ∩ D) = P A * P D := by sorry

end NUMINAMATH_CALUDE_events_A_D_independent_l3239_323955


namespace NUMINAMATH_CALUDE_min_good_operations_2009_l3239_323930

/-- Represents the sum of digits in the binary representation of a natural number -/
def S₂ (n : ℕ) : ℕ := sorry

/-- Represents the minimum number of "good" operations required to split a rope of length n into unit lengths -/
def min_good_operations (n : ℕ) : ℕ := sorry

/-- Theorem stating that the minimum number of good operations for a rope of length 2009 
    is equal to S₂(2009) - 1 -/
theorem min_good_operations_2009 : 
  min_good_operations 2009 = S₂ 2009 - 1 := by sorry

end NUMINAMATH_CALUDE_min_good_operations_2009_l3239_323930


namespace NUMINAMATH_CALUDE_rational_term_count_is_seventeen_l3239_323924

/-- The number of terms with rational coefficients in the expansion of (√3x + ∛2)^100 -/
def rationalTermCount : ℕ := 17

/-- The exponent in the binomial expansion -/
def exponent : ℕ := 100

/-- Predicate to check if a number is a multiple of 2 -/
def isMultipleOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Predicate to check if a number is a multiple of 3 -/
def isMultipleOfThree (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Theorem stating that the number of terms with rational coefficients in the expansion of (√3x + ∛2)^100 is 17 -/
theorem rational_term_count_is_seventeen :
  (∀ r : ℕ, r ≤ exponent →
    (isMultipleOfTwo (exponent - r) ∧ isMultipleOfThree r) ↔
    (∃ n : ℕ, r = 6 * n ∧ n ≤ 16)) ∧
  rationalTermCount = 17 := by sorry

end NUMINAMATH_CALUDE_rational_term_count_is_seventeen_l3239_323924


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3239_323927

/-- A quadratic function with vertex at (-2, 3) passing through (3, -45) has a = -48/25 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Quadratic function definition
  (3 = a * (-2)^2 + b * (-2) + c) →       -- Vertex condition
  (-45 = a * 3^2 + b * 3 + c) →           -- Point condition
  a = -48/25 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3239_323927


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l3239_323946

theorem rational_solutions_quadratic (k : ℕ+) :
  (∃ x : ℚ, k * x^2 + 12 * x + k = 0) ↔ (k = 3 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l3239_323946


namespace NUMINAMATH_CALUDE_exists_x_iff_b_gt_min_sum_l3239_323937

/-- The minimum value of the sum of absolute differences -/
def min_sum : ℝ := 4

/-- The function representing the sum of absolute differences -/
def f (x : ℝ) : ℝ := |x - 5| + |x - 3| + |x - 2|

/-- Theorem stating the condition for the existence of x satisfying the inequality -/
theorem exists_x_iff_b_gt_min_sum (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, f x < b) ↔ b > min_sum :=
sorry

end NUMINAMATH_CALUDE_exists_x_iff_b_gt_min_sum_l3239_323937


namespace NUMINAMATH_CALUDE_tank_fill_time_l3239_323916

/-- Given three pipes with fill rates, calculates the time to fill a tank when all pipes are open -/
theorem tank_fill_time (p q r : ℝ) (hp : p = 1/3) (hq : q = 1/9) (hr : r = 1/18) :
  1 / (p + q + r) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3239_323916


namespace NUMINAMATH_CALUDE_base_8_5624_equals_2964_l3239_323962

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_5624_equals_2964 : 
  base_8_to_10 [4, 2, 6, 5] = 2964 := by
  sorry

end NUMINAMATH_CALUDE_base_8_5624_equals_2964_l3239_323962


namespace NUMINAMATH_CALUDE_computer_preference_ratio_l3239_323984

theorem computer_preference_ratio : 
  ∀ (total mac no_pref equal : ℕ),
    total = 210 →
    mac = 60 →
    no_pref = 90 →
    equal = total - (mac + no_pref) →
    equal = mac →
    (equal : ℚ) / mac = 1 := by
  sorry

end NUMINAMATH_CALUDE_computer_preference_ratio_l3239_323984


namespace NUMINAMATH_CALUDE_chocolate_ticket_value_l3239_323910

/-- Represents the value of a chocolate box ticket in terms of the box's cost -/
def ticket_value : ℚ := 1 / 9

/-- Represents the number of tickets needed to get a free box -/
def tickets_for_free_box : ℕ := 10

/-- Theorem stating the value of a single ticket -/
theorem chocolate_ticket_value :
  ticket_value = 1 / (tickets_for_free_box - 1) :=
by sorry

end NUMINAMATH_CALUDE_chocolate_ticket_value_l3239_323910


namespace NUMINAMATH_CALUDE_spinner_probabilities_l3239_323943

theorem spinner_probabilities : ∃ (x y : ℚ),
  (1 / 4 : ℚ) + (1 / 3 : ℚ) + x + y = 1 ∧
  x + y = 5 / 12 ∧
  x = 1 / 4 ∧
  y = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_spinner_probabilities_l3239_323943


namespace NUMINAMATH_CALUDE_next_simultaneous_occurrence_l3239_323967

def town_hall_interval : ℕ := 18
def fire_station_interval : ℕ := 24
def university_bell_interval : ℕ := 30

def simultaneous_occurrence (t : ℕ) : Prop :=
  t % town_hall_interval = 0 ∧
  t % fire_station_interval = 0 ∧
  t % university_bell_interval = 0

theorem next_simultaneous_occurrence :
  ∃ t : ℕ, t > 0 ∧ t ≤ 360 ∧ simultaneous_occurrence t ∧
  ∀ s : ℕ, 0 < s ∧ s < t → ¬simultaneous_occurrence s :=
sorry

end NUMINAMATH_CALUDE_next_simultaneous_occurrence_l3239_323967


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3239_323909

theorem diophantine_equation_solutions :
  ∀ a b c : ℕ+,
  a * b + b * c + c * a = 2 * (a + b + c) ↔
  ((a = 2 ∧ b = 2 ∧ c = 2) ∨
   (a = 1 ∧ b = 2 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 2) ∨
   (a = 2 ∧ b = 1 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 1) ∨
   (a = 4 ∧ b = 1 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3239_323909


namespace NUMINAMATH_CALUDE_element_value_l3239_323985

theorem element_value (a : ℕ) : 
  a ∈ ({0, 1, 2, 3} : Set ℕ) → 
  a ∉ ({0, 1, 2} : Set ℕ) → 
  a = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_element_value_l3239_323985


namespace NUMINAMATH_CALUDE_alipay_growth_rate_l3239_323960

theorem alipay_growth_rate (initial : ℕ) (final : ℕ) (years : ℕ) (rate : ℝ) : 
  initial = 45000 →
  final = 64800 →
  years = 2 →
  (initial : ℝ) * (1 + rate) ^ years = final →
  rate = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_alipay_growth_rate_l3239_323960


namespace NUMINAMATH_CALUDE_student_D_most_stable_l3239_323975

-- Define the students
inductive Student : Type
  | A
  | B
  | C
  | D

-- Define the variance function
def variance : Student → Real
  | Student.A => 2.1
  | Student.B => 3.5
  | Student.C => 9
  | Student.D => 0.7

-- Define the concept of stability
def most_stable (s : Student) : Prop :=
  ∀ t : Student, variance s ≤ variance t

-- Theorem statement
theorem student_D_most_stable :
  most_stable Student.D :=
by sorry

end NUMINAMATH_CALUDE_student_D_most_stable_l3239_323975


namespace NUMINAMATH_CALUDE_no_solution_l3239_323908

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem no_solution : ¬∃ x : ℕ, (137 + x = 435) ∧ (reverse_digits x = 672) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l3239_323908


namespace NUMINAMATH_CALUDE_class_test_percentages_l3239_323941

theorem class_test_percentages (total : ℝ) (first : ℝ) (second : ℝ) (both : ℝ) 
  (h_total : total = 100)
  (h_first : first = 75)
  (h_second : second = 30)
  (h_both : both = 25) :
  total - (first + second - both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_class_test_percentages_l3239_323941


namespace NUMINAMATH_CALUDE_dice_probability_l3239_323940

def probability_less_than_6 : ℚ := 1 / 2

def number_of_dice : ℕ := 6

def target_count : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem dice_probability : 
  (choose number_of_dice target_count : ℚ) * probability_less_than_6^number_of_dice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l3239_323940


namespace NUMINAMATH_CALUDE_prime_pair_from_quadratic_roots_l3239_323996

theorem prime_pair_from_quadratic_roots (p q : ℕ) (hp : p.Prime) (hq : q.Prime) 
  (x₁ x₂ : ℤ) (h_sum : x₁ + x₂ = -p) (h_prod : x₁ * x₂ = q) : p = 3 ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_pair_from_quadratic_roots_l3239_323996


namespace NUMINAMATH_CALUDE_river_speed_l3239_323923

/-- Proves that the speed of the river is 1.2 kmph given the conditions -/
theorem river_speed (rowing_speed : ℝ) (total_time : ℝ) (total_distance : ℝ)
  (h1 : rowing_speed = 8)
  (h2 : total_time = 1)
  (h3 : total_distance = 7.82) :
  ∃ v : ℝ, v = 1.2 ∧
  (total_distance / 2) / (rowing_speed - v) + (total_distance / 2) / (rowing_speed + v) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_speed_l3239_323923


namespace NUMINAMATH_CALUDE_nancy_hourly_wage_l3239_323986

/-- Calculates the hourly wage needed to cover remaining expenses for one semester --/
def hourly_wage_needed (tuition housing meal_plan textbooks merit_scholarship need_scholarship work_hours : ℕ) : ℚ :=
  let total_cost := tuition + housing + meal_plan + textbooks
  let parents_contribution := tuition / 2
  let student_loan := 2 * merit_scholarship
  let total_support := parents_contribution + merit_scholarship + need_scholarship + student_loan
  let remaining_expenses := total_cost - total_support
  (remaining_expenses : ℚ) / work_hours

/-- Theorem stating that Nancy needs to earn $49 per hour --/
theorem nancy_hourly_wage :
  hourly_wage_needed 22000 6000 2500 800 3000 1500 200 = 49 := by
  sorry

end NUMINAMATH_CALUDE_nancy_hourly_wage_l3239_323986


namespace NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3239_323939

theorem largest_consecutive_even_integer (n : ℕ) : 
  n % 2 = 0 ∧ 
  n * (n + 2) * (n + 4) * (n + 6) = 6720 →
  n + 6 = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_consecutive_even_integer_l3239_323939


namespace NUMINAMATH_CALUDE_ratio_problem_l3239_323992

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.75 * a) (h5 : m = b - 0.8 * b) : m / x = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3239_323992


namespace NUMINAMATH_CALUDE_polynomial_inequality_roots_l3239_323903

theorem polynomial_inequality_roots (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c*x - 8 < 0 ↔ x < 2 ∨ x > 6) → c = 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_roots_l3239_323903


namespace NUMINAMATH_CALUDE_cube_face_area_l3239_323958

/-- Given a cube with surface area 36 square centimeters, 
    prove that the area of one face is 6 square centimeters. -/
theorem cube_face_area (surface_area : ℝ) (h : surface_area = 36) :
  surface_area / 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_area_l3239_323958


namespace NUMINAMATH_CALUDE_f_neg_two_equals_six_l3239_323911

/-- The quadratic function f(x) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * b * x + c

/-- The quadratic function g(x) -/
def g (a b c : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 2 * (b + 2) * x + (c + 4)

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem f_neg_two_equals_six (a b c : ℝ) :
  discriminant (a + 1) (b + 2) (c + 4) - discriminant a b c = 24 →
  f a b c (-2) = 6 := by
  sorry

#eval f 1 2 3 (-2)  -- Example usage

end NUMINAMATH_CALUDE_f_neg_two_equals_six_l3239_323911


namespace NUMINAMATH_CALUDE_x_value_proof_l3239_323971

theorem x_value_proof (x : ℝ) (h1 : x^2 - 5*x = 0) (h2 : x ≠ 0) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3239_323971


namespace NUMINAMATH_CALUDE_branch_A_more_profitable_l3239_323925

/-- Represents the grades of products -/
inductive Grade
| A
| B
| C
| D

/-- Represents a branch of the factory -/
structure Branch where
  name : String
  processingCost : ℝ
  gradeDistribution : Grade → ℝ

/-- Calculates the processing fee for a given grade -/
def processingFee (g : Grade) : ℝ :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Calculates the average profit per 100 products for a given branch -/
def averageProfit (b : Branch) : ℝ :=
  (processingFee Grade.A - b.processingCost) * b.gradeDistribution Grade.A +
  (processingFee Grade.B - b.processingCost) * b.gradeDistribution Grade.B +
  (processingFee Grade.C - b.processingCost) * b.gradeDistribution Grade.C +
  (processingFee Grade.D - b.processingCost) * b.gradeDistribution Grade.D

/-- Branch A of the factory -/
def branchA : Branch :=
  { name := "A"
    processingCost := 25
    gradeDistribution := fun g => match g with
      | Grade.A => 0.4
      | Grade.B => 0.2
      | Grade.C => 0.2
      | Grade.D => 0.2 }

/-- Branch B of the factory -/
def branchB : Branch :=
  { name := "B"
    processingCost := 20
    gradeDistribution := fun g => match g with
      | Grade.A => 0.28
      | Grade.B => 0.17
      | Grade.C => 0.34
      | Grade.D => 0.21 }

theorem branch_A_more_profitable :
  averageProfit branchA > averageProfit branchB :=
sorry

end NUMINAMATH_CALUDE_branch_A_more_profitable_l3239_323925


namespace NUMINAMATH_CALUDE_cycling_distance_conversion_l3239_323942

/-- Converts a list of digits in base 9 to a number in base 10 -/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ (digits.length - 1 - i))) 0

/-- The cycling distance in base 9 -/
def cyclingDistanceBase9 : List Nat := [3, 6, 1, 8]

theorem cycling_distance_conversion :
  base9ToBase10 cyclingDistanceBase9 = 2690 := by
  sorry

end NUMINAMATH_CALUDE_cycling_distance_conversion_l3239_323942


namespace NUMINAMATH_CALUDE_exists_quadrilateral_with_adjacent_colors_l3239_323980

/-- Represents the color of a vertex -/
inductive Color
| Black
| White

/-- Represents a convex polygon -/
structure ConvexPolygon where
  vertices : ℕ
  coloring : ℕ → Color

/-- Represents a quadrilateral formed by dividing the polygon -/
structure Quadrilateral where
  v1 : ℕ
  v2 : ℕ
  v3 : ℕ
  v4 : ℕ

/-- The specific coloring pattern of the 2550-gon -/
def specific_coloring : ℕ → Color := sorry

/-- The 2550-gon with the specific coloring -/
def polygon_2550 : ConvexPolygon :=
  { vertices := 2550,
    coloring := specific_coloring }

/-- Predicate to check if a quadrilateral has two adjacent black vertices and two adjacent white vertices -/
def has_adjacent_colors (q : Quadrilateral) (p : ConvexPolygon) : Prop := sorry

/-- A division of the polygon into quadrilaterals -/
def division : List Quadrilateral := sorry

/-- Theorem stating that there exists a quadrilateral with the required color pattern -/
theorem exists_quadrilateral_with_adjacent_colors :
  ∃ q ∈ division, has_adjacent_colors q polygon_2550 := by sorry

end NUMINAMATH_CALUDE_exists_quadrilateral_with_adjacent_colors_l3239_323980


namespace NUMINAMATH_CALUDE_sequence_limit_l3239_323900

/-- The sequence defined by the recurrence relation -/
noncomputable def x : ℕ → ℝ
| 0 => sorry -- x₁ is not specified in the original problem
| n + 1 => Real.sqrt (2 * x n + 3)

/-- The theorem stating that the limit of the sequence is 3 -/
theorem sequence_limit : Filter.Tendsto x Filter.atTop (nhds 3) := by sorry

end NUMINAMATH_CALUDE_sequence_limit_l3239_323900


namespace NUMINAMATH_CALUDE_smallest_number_with_divisibility_property_l3239_323991

theorem smallest_number_with_divisibility_property : 
  ∀ n : ℕ, n > 0 → (n + 9) % 8 = 0 ∧ (n + 9) % 11 = 0 ∧ (∃ k : ℕ, k > 1 ∧ k ≠ 8 ∧ k ≠ 11 ∧ n % k = 0) → n ≥ 255 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_divisibility_property_l3239_323991


namespace NUMINAMATH_CALUDE_decimal_multiplication_division_l3239_323945

theorem decimal_multiplication_division : (0.5 * 0.6) / 0.2 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_division_l3239_323945


namespace NUMINAMATH_CALUDE_fraction_sum_and_divide_l3239_323932

theorem fraction_sum_and_divide : (3/20 + 5/200 + 7/2000) / 2 = 0.08925 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_divide_l3239_323932


namespace NUMINAMATH_CALUDE_intersection_orthogonal_l3239_323966

/-- The ellipse E with equation x²/8 + y²/4 = 1 -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}

/-- The line L with equation y = √5*x + 4 -/
def L : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 5 * p.1 + 4}

/-- The intersection points of E and L -/
def intersection := E ∩ L

/-- Theorem: If A and B are the intersection points of E and L, then OA ⊥ OB -/
theorem intersection_orthogonal (A B : ℝ × ℝ) 
  (hA : A ∈ intersection) (hB : B ∈ intersection) (hAB : A ≠ B) :
  (A.1 * B.1 + A.2 * B.2 = 0) := by sorry

end NUMINAMATH_CALUDE_intersection_orthogonal_l3239_323966


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l3239_323982

/-- Given that 45 cows eat 45 bags of husk in 45 days, 
    prove that 1 cow will eat 1 bag of husk in 45 days -/
theorem cow_husk_consumption 
  (cows : ℕ) (bags : ℕ) (days : ℕ) 
  (h : cows = 45 ∧ bags = 45 ∧ days = 45) : 
  1 * bags / cows = days :=
sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l3239_323982


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3239_323904

def base8_to_10 (n : ℕ) : ℕ := 2 * 8^2 + 5 * 8^1 + 4 * 8^0

def base2_to_10 (n : ℕ) : ℕ := 1 * 2^1 + 1 * 2^0

def base5_to_10 (n : ℕ) : ℕ := 1 * 5^2 + 4 * 5^1 + 4 * 5^0

def base4_to_10 (n : ℕ) : ℕ := 3 * 4^1 + 2 * 4^0

theorem base_conversion_sum :
  (base8_to_10 254 : ℚ) / (base2_to_10 11) + (base5_to_10 144 : ℚ) / (base4_to_10 32) = 57.4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3239_323904


namespace NUMINAMATH_CALUDE_match_rectangle_properties_l3239_323994

/-- Represents a rectangle made of matches -/
structure MatchRectangle where
  m : ℕ
  n : ℕ
  h : m > n

/-- Total number of matches used to construct the rectangle -/
def totalMatches (r : MatchRectangle) : ℕ :=
  2 * r.m * r.n + r.m + r.n

/-- Total number of possible rectangles in the figure -/
def totalRectangles (r : MatchRectangle) : ℚ :=
  (r.m * r.n * (r.m + 1) * (r.n + 1)) / 4

/-- Total number of possible squares in the figure -/
def totalSquares (r : MatchRectangle) : ℚ :=
  (r.n * (r.n + 1) * (3 * r.m - r.n + 1)) / 6

theorem match_rectangle_properties (r : MatchRectangle) :
  (totalMatches r = 2 * r.m * r.n + r.m + r.n) ∧
  (totalRectangles r = (r.m * r.n * (r.m + 1) * (r.n + 1)) / 4) ∧
  (totalSquares r = (r.n * (r.n + 1) * (3 * r.m - r.n + 1)) / 6) := by
  sorry

end NUMINAMATH_CALUDE_match_rectangle_properties_l3239_323994


namespace NUMINAMATH_CALUDE_cube_root_inequality_l3239_323952

theorem cube_root_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.rpow ((a + 1) * (b + 1) * (c + 1)) (1/3) ≥ Real.rpow (a * b * c) (1/3) + 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l3239_323952


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l3239_323983

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 - 8 * x + 36 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b h k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧
  ∀ x y, f x y ↔ (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l3239_323983


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l3239_323919

/-- Proves that given a man rowing downstream with a current speed of 3 kmph,
    covering 80 meters in 15.99872010239181 seconds, his speed in still water is 15 kmph. -/
theorem mans_speed_in_still_water
  (current_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : current_speed = 3)
  (h2 : distance = 80)
  (h3 : time = 15.99872010239181)
  : ∃ (speed_still_water : ℝ), speed_still_water = 15 := by
  sorry

#check mans_speed_in_still_water

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l3239_323919


namespace NUMINAMATH_CALUDE_geometric_inequalities_l3239_323913

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a point inside a triangle
def InsideTriangle (t : Triangle) (D : ℝ × ℝ) : Prop := sorry

-- Define a point inside a convex quadrilateral
def InsideConvexQuadrilateral (q : Quadrilateral) (E : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A of a triangle
def angle_A (t : Triangle) : ℝ := sorry

-- Define the ratio k
def ratio_k (q : Quadrilateral) (E : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem geometric_inequalities 
  (t : Triangle) 
  (D : ℝ × ℝ) 
  (q : Quadrilateral) 
  (E : ℝ × ℝ) 
  (h1 : InsideTriangle t D) 
  (h2 : InsideConvexQuadrilateral q E) : 
  (distance t.B t.C / min (distance t.A D) (min (distance t.B D) (distance t.C D)) ≥ 
    if angle_A t < π/2 then 2 * Real.sin (angle_A t) else 2) ∧
  (ratio_k q E ≥ 2 * Real.sin (70 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_inequalities_l3239_323913


namespace NUMINAMATH_CALUDE_divisibility_by_133_l3239_323973

theorem divisibility_by_133 (n : ℕ) : ∃ k : ℤ, 11^(n+2) + 12^(2*n+1) = 133 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_133_l3239_323973


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l3239_323964

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define a point inside a triangle
def PointInside (T : Triangle) (P : Point) : Prop :=
  -- Placeholder for the condition that P is inside triangle T
  sorry

-- Define a point on a side of a triangle
def PointOnSide (T : Triangle) (Q : Point) : Prop :=
  -- Placeholder for the condition that Q is on a side of triangle T
  sorry

-- Define the property of not sharing an entire side
def NotShareEntireSide (T1 T2 : Triangle) : Prop :=
  -- Placeholder for the condition that T1 and T2 do not share an entire side
  sorry

theorem triangle_division_theorem (T : Triangle) :
  ∃ (P Q : Point) (T1 T2 T3 T4 : Triangle),
    PointInside T P ∧
    PointOnSide T Q ∧
    NotShareEntireSide T1 T2 ∧
    NotShareEntireSide T1 T3 ∧
    NotShareEntireSide T1 T4 ∧
    NotShareEntireSide T2 T3 ∧
    NotShareEntireSide T2 T4 ∧
    NotShareEntireSide T3 T4 :=
  sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l3239_323964


namespace NUMINAMATH_CALUDE_imaginary_part_of_two_over_one_plus_i_l3239_323912

theorem imaginary_part_of_two_over_one_plus_i :
  Complex.im (2 / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_two_over_one_plus_i_l3239_323912


namespace NUMINAMATH_CALUDE_grass_cutting_expenditure_l3239_323965

/-- Represents the four seasons --/
inductive Season
  | Spring
  | Summer
  | Fall
  | Winter

/-- Growth rate of grass per month for each season (in inches) --/
def growth_rate (s : Season) : Real :=
  match s with
  | Season.Spring => 0.6
  | Season.Summer => 0.5
  | Season.Fall => 0.4
  | Season.Winter => 0.2

/-- Number of months in each season --/
def months_per_season : Nat := 3

/-- Initial height of grass after cutting (in inches) --/
def initial_height : Real := 2

/-- Height at which grass needs to be cut (in inches) --/
def cut_height : Real := 4

/-- Initial cost to cut grass --/
def initial_cost : Nat := 100

/-- Cost increase per cut --/
def cost_increase : Nat := 5

/-- Calculate the total growth of grass in a season --/
def season_growth (s : Season) : Real :=
  growth_rate s * months_per_season

/-- Calculate the number of cuts needed in a year --/
def cuts_per_year : Nat := 2

/-- Calculate the total expenditure for cutting grass in a year --/
def total_expenditure : Nat :=
  initial_cost + (initial_cost + cost_increase)

theorem grass_cutting_expenditure :
  total_expenditure = 205 := by
  sorry

end NUMINAMATH_CALUDE_grass_cutting_expenditure_l3239_323965


namespace NUMINAMATH_CALUDE_triangle_point_distance_l3239_323914

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AB = 13 ∧ AC = 13 ∧ BC = 10

-- Define the point P
def PointInside (P A B C : ℝ × ℝ) : Prop :=
  ∃ t u v : ℝ, t > 0 ∧ u > 0 ∧ v > 0 ∧ t + u + v = 1 ∧
  P = (t * A.1 + u * B.1 + v * C.1, t * A.2 + u * B.2 + v * C.2)

-- Define the distances PA and PB
def Distances (P A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 15 ∧
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 9

-- Define the angle equality
def AngleEquality (P A B C : ℝ × ℝ) : Prop :=
  let angle (X Y Z : ℝ × ℝ) := Real.arccos (
    ((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) /
    (Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) * Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2))
  )
  angle A P B = angle B P C ∧ angle B P C = angle C P A

-- Main theorem
theorem triangle_point_distance (A B C P : ℝ × ℝ) :
  Triangle A B C →
  PointInside P A B C →
  Distances P A B →
  AngleEquality P A B C →
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = (-9 + Real.sqrt 157) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_point_distance_l3239_323914


namespace NUMINAMATH_CALUDE_oplus_four_two_l3239_323936

-- Define the operation ⊕ for real numbers
def oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem oplus_four_two : oplus 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_oplus_four_two_l3239_323936
