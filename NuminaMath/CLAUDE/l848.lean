import Mathlib

namespace NUMINAMATH_CALUDE_placemat_length_l848_84876

theorem placemat_length (R : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  R = 5 → n = 8 → w = 1 → 
  y = (R^2 - (w/2)^2).sqrt - R * (2 - (2:ℝ).sqrt).sqrt / 2 := by
  sorry

end NUMINAMATH_CALUDE_placemat_length_l848_84876


namespace NUMINAMATH_CALUDE_ratio_difference_bound_l848_84812

theorem ratio_difference_bound (a : Fin 5 → ℝ) (h : ∀ i, 0 < a i) :
  ∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    |a i / a j - a k / a l| < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_bound_l848_84812


namespace NUMINAMATH_CALUDE_shaded_triangles_area_sum_l848_84885

/-- The sum of areas of shaded triangles in an infinite geometric series --/
theorem shaded_triangles_area_sum (x y z : ℝ) (h1 : x = 8) (h2 : y = 8) (h3 : z = 8) 
  (h4 : x^2 = y^2 + z^2) : 
  let initial_area := (1/2) * y * z
  let first_shaded_area := (1/4) * initial_area
  let ratio := (1/4 : ℝ)
  (initial_area * ratio) / (1 - ratio) = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_triangles_area_sum_l848_84885


namespace NUMINAMATH_CALUDE_correct_calculation_l848_84859

theorem correct_calculation (x y : ℝ) : 3 * x^4 * y / (x^2 * y) = 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l848_84859


namespace NUMINAMATH_CALUDE_katie_speed_calculation_l848_84834

-- Define the running speeds
def eugene_speed : ℚ := 4
def brianna_speed : ℚ := (2/3) * eugene_speed
def katie_speed : ℚ := (7/5) * brianna_speed

-- Theorem to prove
theorem katie_speed_calculation :
  katie_speed = 56/15 := by sorry

end NUMINAMATH_CALUDE_katie_speed_calculation_l848_84834


namespace NUMINAMATH_CALUDE_wire_length_problem_l848_84892

theorem wire_length_problem (total_wires : ℕ) (total_avg_length : ℝ) 
  (quarter_avg_length : ℝ) (third_avg_length : ℝ) :
  total_wires = 12 →
  total_avg_length = 95 →
  quarter_avg_length = 120 →
  third_avg_length = 75 →
  let quarter_wires := total_wires / 4
  let third_wires := total_wires / 3
  let remaining_wires := total_wires - quarter_wires - third_wires
  let total_length := total_wires * total_avg_length
  let quarter_length := quarter_wires * quarter_avg_length
  let third_length := third_wires * third_avg_length
  let remaining_length := total_length - quarter_length - third_length
  remaining_length / remaining_wires = 96 := by
sorry

end NUMINAMATH_CALUDE_wire_length_problem_l848_84892


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l848_84853

theorem circle_diameter_from_area (A : Real) (π : Real) (h : π > 0) :
  A = 225 * π → ∃ d : Real, d > 0 ∧ A = π * (d / 2)^2 ∧ d = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l848_84853


namespace NUMINAMATH_CALUDE_sum_assigned_values_zero_l848_84867

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first n prime numbers -/
def primeProduct (n : ℕ) : ℕ := sorry

/-- Assigns 1 or -1 to a number based on its prime factorization -/
def assignValue (n : ℕ) : Int := sorry

/-- The sum of assigned values for all divisors of a number -/
def sumAssignedValues (n : ℕ) : Int := sorry

/-- Theorem: The sum of assigned values for divisors of the product of first k primes is 0 -/
theorem sum_assigned_values_zero (k : ℕ) : sumAssignedValues (primeProduct k) = 0 := by sorry

end NUMINAMATH_CALUDE_sum_assigned_values_zero_l848_84867


namespace NUMINAMATH_CALUDE_functional_equation_zero_value_l848_84887

theorem functional_equation_zero_value 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) = f x + f y) : 
  f 0 = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_value_l848_84887


namespace NUMINAMATH_CALUDE_spending_limit_ratio_l848_84864

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

/-- The conditions of Sally's credit cards -/
def sally_cards_conditions (cards : SallysCards) : Prop :=
  cards.gold.balance = (1/3) * cards.gold.limit ∧
  cards.platinum.balance = (1/4) * cards.platinum.limit ∧
  cards.platinum.balance + cards.gold.balance = (5/12) * cards.platinum.limit

/-- The theorem stating the ratio of spending limits -/
theorem spending_limit_ratio (cards : SallysCards) 
  (h : sally_cards_conditions cards) : 
  cards.platinum.limit = (1/2) * cards.gold.limit := by
  sorry

#check spending_limit_ratio

end NUMINAMATH_CALUDE_spending_limit_ratio_l848_84864


namespace NUMINAMATH_CALUDE_unique_function_property_l848_84863

theorem unique_function_property (f : ℕ → ℕ) :
  (f 1 > 0) ∧
  (∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) →
  (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_property_l848_84863


namespace NUMINAMATH_CALUDE_product_zero_given_sum_and_seventh_power_sum_zero_l848_84833

theorem product_zero_given_sum_and_seventh_power_sum_zero 
  (w x y z : ℝ) 
  (sum_zero : w + x + y + z = 0) 
  (seventh_power_sum_zero : w^7 + x^7 + y^7 + z^7 = 0) : 
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_given_sum_and_seventh_power_sum_zero_l848_84833


namespace NUMINAMATH_CALUDE_profit_maximum_l848_84879

/-- The profit function for a product with selling price m -/
def profit (m : ℝ) : ℝ := (m - 8) * (900 - 15 * m)

/-- The expression claimed to represent the maximum profit -/
def maxProfitExpr (m : ℝ) : ℝ := -15 * (m - 34)^2 + 10140

theorem profit_maximum :
  ∃ m₀ : ℝ, 
    (∀ m : ℝ, profit m ≤ profit m₀) ∧ 
    (∀ m : ℝ, maxProfitExpr m = profit m) ∧
    (maxProfitExpr m₀ = profit m₀) :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l848_84879


namespace NUMINAMATH_CALUDE_sushil_marks_proof_l848_84896

def total_marks (english science maths : ℕ) : ℕ := english + science + maths

theorem sushil_marks_proof (english science maths : ℕ) :
  english = 3 * science →
  english = maths / 4 →
  science = 17 →
  total_marks english science maths = 272 :=
by
  sorry

end NUMINAMATH_CALUDE_sushil_marks_proof_l848_84896


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l848_84820

theorem least_number_divisible_by_five_primes : ∃ n : ℕ, 
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    p₁ ∣ n ∧ p₂ ∣ n ∧ p₃ ∣ n ∧ p₄ ∣ n ∧ p₅ ∣ n) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ, 
      Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      q₁ ∣ m ∧ q₂ ∣ m ∧ q₃ ∣ m ∧ q₄ ∣ m ∧ q₅ ∣ m)) ∧
  n = 2310 :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l848_84820


namespace NUMINAMATH_CALUDE_karthik_weight_average_l848_84845

def karthik_weight_lower_bound : ℝ := 56
def karthik_weight_upper_bound : ℝ := 57

theorem karthik_weight_average :
  let min_weight := karthik_weight_lower_bound
  let max_weight := karthik_weight_upper_bound
  (min_weight + max_weight) / 2 = 56.5 := by sorry

end NUMINAMATH_CALUDE_karthik_weight_average_l848_84845


namespace NUMINAMATH_CALUDE_terry_bottle_caps_l848_84835

def bottle_cap_collection (num_groups : ℕ) (caps_per_group : ℕ) : ℕ :=
  num_groups * caps_per_group

theorem terry_bottle_caps : 
  bottle_cap_collection 80 7 = 560 := by
  sorry

end NUMINAMATH_CALUDE_terry_bottle_caps_l848_84835


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l848_84848

def f (x : ℝ) := x^3 - 12*x^2 + 47*x - 60

theorem cubic_inequality_solution :
  ∀ x : ℝ, f x < 0 ↔ 3 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l848_84848


namespace NUMINAMATH_CALUDE_range_of_a_l848_84808

/-- A linear function y = (2a-3)x + a + 2 that is above the x-axis for -2 ≤ x ≤ 1 -/
def LinearFunction (a : ℝ) (x : ℝ) : ℝ := (2*a - 3)*x + a + 2

/-- The function is above the x-axis for -2 ≤ x ≤ 1 -/
def AboveXAxis (a : ℝ) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 1 → LinearFunction a x > 0

theorem range_of_a (a : ℝ) (h : AboveXAxis a) :
  1/3 < a ∧ a < 8/3 ∧ a ≠ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l848_84808


namespace NUMINAMATH_CALUDE_unique_special_parallelogram_l848_84868

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  B : ℤ × ℤ
  D : ℤ × ℤ
  C : ℚ × ℚ
  area_eq : abs (B.1 * D.2 + D.1 * C.2 + C.1 * 0 - (B.2 * D.1 + D.2 * C.1 + C.2 * 0)) / 2 = 2000000
  B_on_y_eq_x : B.2 = B.1
  D_on_y_eq_2x : D.2 = 2 * D.1
  C_on_y_eq_3x : C.2 = 3 * C.1
  first_quadrant : 0 < B.1 ∧ 0 < B.2 ∧ 0 < D.1 ∧ 0 < D.2 ∧ 0 < C.1 ∧ 0 < C.2
  parallelogram_condition : C.1 = B.1 + D.1 ∧ C.2 = B.2 + D.2

/-- There exists exactly one special parallelogram -/
theorem unique_special_parallelogram : ∃! p : SpecialParallelogram, True :=
  sorry

end NUMINAMATH_CALUDE_unique_special_parallelogram_l848_84868


namespace NUMINAMATH_CALUDE_worker_a_time_proof_l848_84882

/-- The time it takes for Worker A to complete the job alone -/
def worker_a_time : ℝ := 8.4

/-- The time it takes for Worker B to complete the job alone -/
def worker_b_time : ℝ := 6

/-- The time it takes for both workers to complete the job together -/
def combined_time : ℝ := 3.428571428571429

theorem worker_a_time_proof :
  (1 / worker_a_time) + (1 / worker_b_time) = (1 / combined_time) :=
sorry

end NUMINAMATH_CALUDE_worker_a_time_proof_l848_84882


namespace NUMINAMATH_CALUDE_rope_length_proof_l848_84840

theorem rope_length_proof : 
  ∀ (L : ℝ), 
    (L / 4 - L / 6 = 2) →  -- Difference between parts is 2 meters
    (2 * L = 48)           -- Total length of two ropes is 48 meters
  := by sorry

end NUMINAMATH_CALUDE_rope_length_proof_l848_84840


namespace NUMINAMATH_CALUDE_base8_to_base7_conversion_l848_84819

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 7
def base10ToBase7 (n : Nat) : Nat :=
  (n / 343) * 1000 + ((n / 49) % 7) * 100 + ((n / 7) % 7) * 10 + (n % 7)

theorem base8_to_base7_conversion :
  base10ToBase7 (base8ToBase10 563) = 1162 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base7_conversion_l848_84819


namespace NUMINAMATH_CALUDE_validArrangementCount_l848_84869

/-- Represents a seating arrangement around a rectangular table. -/
structure SeatingArrangement where
  chairs : Fin 15 → Person
  satisfiesConditions : Bool

/-- Represents a person to be seated. -/
inductive Person
  | Man : Person
  | Woman : Person
  | AdditionalPerson : Person

/-- Checks if two positions are adjacent or opposite on the table. -/
def areAdjacentOrOpposite (pos1 pos2 : Fin 15) : Bool := sorry

/-- Checks if the seating arrangement satisfies all conditions. -/
def satisfiesAllConditions (arrangement : SeatingArrangement) : Bool := sorry

/-- Counts the number of valid seating arrangements. -/
def countValidArrangements : Nat := sorry

/-- Theorem stating the number of valid seating arrangements. -/
theorem validArrangementCount : countValidArrangements = 3265920 := by sorry

end NUMINAMATH_CALUDE_validArrangementCount_l848_84869


namespace NUMINAMATH_CALUDE_prob_select_all_cocaptains_l848_84830

/-- Represents a math team with a given number of students and co-captains -/
structure MathTeam where
  num_students : ℕ
  num_cocaptains : ℕ

/-- Calculates the probability of selecting all co-captains from a given team -/
def prob_select_cocaptains (team : MathTeam) : ℚ :=
  1 / (team.num_students.choose 3)

/-- The set of math teams in the area -/
def math_teams : List MathTeam := [
  { num_students := 6, num_cocaptains := 3 },
  { num_students := 7, num_cocaptains := 3 },
  { num_students := 8, num_cocaptains := 3 },
  { num_students := 9, num_cocaptains := 3 }
]

/-- Theorem stating the probability of selecting all co-captains -/
theorem prob_select_all_cocaptains : 
  (1 / (math_teams.length : ℚ)) * (math_teams.map prob_select_cocaptains).sum = 91 / 6720 := by
  sorry


end NUMINAMATH_CALUDE_prob_select_all_cocaptains_l848_84830


namespace NUMINAMATH_CALUDE_steven_amanda_hike_difference_l848_84804

/-- The number of hikes Camila has gone on -/
def camila_hikes : ℕ := 7

/-- The number of times Amanda has gone hiking compared to Camila -/
def amanda_multiplier : ℕ := 8

/-- The number of hikes Amanda has gone on -/
def amanda_hikes : ℕ := camila_hikes * amanda_multiplier

/-- The number of hikes Camila plans to go on per week -/
def camila_weekly_plan : ℕ := 4

/-- The number of weeks Camila plans to hike to match Steven -/
def camila_weeks_plan : ℕ := 16

/-- The total number of hikes Camila aims for to match Steven -/
def steven_hikes : ℕ := camila_hikes + camila_weekly_plan * camila_weeks_plan

theorem steven_amanda_hike_difference :
  steven_hikes - amanda_hikes = 15 := by
  sorry

end NUMINAMATH_CALUDE_steven_amanda_hike_difference_l848_84804


namespace NUMINAMATH_CALUDE_sum_simplification_l848_84899

theorem sum_simplification :
  (-1)^2002 + (-1)^2003 + 2^2004 - 2^2003 = 2^2003 := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l848_84899


namespace NUMINAMATH_CALUDE_hyperbola_intersecting_line_l848_84800

/-- Given a hyperbola and an ellipse with specific properties, prove the equation of a line intersecting the hyperbola. -/
theorem hyperbola_intersecting_line 
  (a : ℝ) 
  (h_a_pos : a > 0)
  (C : Set (ℝ × ℝ)) 
  (h_C : C = {(x, y) | x^2/a^2 - y^2/4 = 1})
  (E : Set (ℝ × ℝ))
  (h_E : E = {(x, y) | x^2/16 + y^2/8 = 1})
  (h_foci : {(-4, 0), (4, 0)} ⊆ C)
  (A B : ℝ × ℝ)
  (h_AB : A ∈ C ∧ B ∈ C)
  (h_midpoint : (A.1 + B.1)/2 = 6 ∧ (A.2 + B.2)/2 = 1) :
  ∃ (k m : ℝ), k * A.1 + m * A.2 = 1 ∧ k * B.1 + m * B.2 = 1 ∧ k = 2 ∧ m = -1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_intersecting_line_l848_84800


namespace NUMINAMATH_CALUDE_intersection_points_imply_c_value_l848_84865

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem intersection_points_imply_c_value :
  ∀ c : ℝ, (∃! (a b : ℝ), a ≠ b ∧ f c a = 0 ∧ f c b = 0 ∧ 
    (∀ x : ℝ, f c x = 0 → x = a ∨ x = b)) →
  c = -2 ∨ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_imply_c_value_l848_84865


namespace NUMINAMATH_CALUDE_greatest_number_l848_84889

def octal_to_decimal (n : ℕ) : ℕ := 3 * 8^1 + 2 * 8^0

def base5_to_decimal (n : ℕ) : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0

def binary_to_decimal (n : ℕ) : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

def base6_to_decimal (n : ℕ) : ℕ := 5 * 6^1 + 4 * 6^0

theorem greatest_number : 
  binary_to_decimal 101010 > octal_to_decimal 32 ∧
  binary_to_decimal 101010 > base5_to_decimal 111 ∧
  binary_to_decimal 101010 > base6_to_decimal 54 := by
  sorry

end NUMINAMATH_CALUDE_greatest_number_l848_84889


namespace NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l848_84839

theorem cube_sphere_volume_ratio (a r : ℝ) (h : a > 0) (k : r > 0) :
  6 * a^2 = 4 * Real.pi * r^2 →
  (a^3) / ((4/3) * Real.pi * r^3) = Real.sqrt 6 / 6 := by
sorry

end NUMINAMATH_CALUDE_cube_sphere_volume_ratio_l848_84839


namespace NUMINAMATH_CALUDE_jesses_room_difference_l848_84810

/-- Jesse's room dimensions and length-width difference --/
theorem jesses_room_difference :
  ∀ (length width : ℝ),
  length = 20 →
  width = 19 →
  length - width = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_jesses_room_difference_l848_84810


namespace NUMINAMATH_CALUDE_cosine_sum_twenty_degrees_l848_84815

theorem cosine_sum_twenty_degrees : 
  Real.cos (20 * π / 180) + Real.cos (60 * π / 180) + 
  Real.cos (100 * π / 180) + Real.cos (140 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_twenty_degrees_l848_84815


namespace NUMINAMATH_CALUDE_constant_term_expansion_l848_84872

/-- The constant term in the expansion of (x - 1/x)^6 -/
def constantTerm : ℤ := -20

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_expansion :
  constantTerm = -binomial 6 3 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l848_84872


namespace NUMINAMATH_CALUDE_accounting_balance_l848_84891

/-- Given the equation 3q - x = 15000, where q = 7 and x = 7 + 75i, prove that p = 5005 + 25i -/
theorem accounting_balance (q x p : ℂ) : 
  3 * q - x = 15000 → q = 7 → x = 7 + 75 * Complex.I → p = 5005 + 25 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_accounting_balance_l848_84891


namespace NUMINAMATH_CALUDE_five_star_seven_l848_84807

/-- The star operation defined as (a + b + 3)^2 -/
def star (a b : ℕ) : ℕ := (a + b + 3)^2

/-- Theorem stating that 5 ★ 7 = 225 -/
theorem five_star_seven : star 5 7 = 225 := by
  sorry

end NUMINAMATH_CALUDE_five_star_seven_l848_84807


namespace NUMINAMATH_CALUDE_rotation_of_point_transformed_curve_equation_l848_84821

def rotation_pi_over_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), p.1)

def transformation_T2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.2)

def compose_transformations (f g : ℝ × ℝ → ℝ × ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  f (g p)

def parabola (x : ℝ) : ℝ := x^2

theorem rotation_of_point :
  rotation_pi_over_2 (2, 1) = (-1, 2) := by sorry

theorem transformed_curve_equation (x y : ℝ) :
  (∃ t : ℝ, compose_transformations transformation_T2 rotation_pi_over_2 (t, parabola t) = (x, y)) ↔
  y - x = y^2 := by sorry

end NUMINAMATH_CALUDE_rotation_of_point_transformed_curve_equation_l848_84821


namespace NUMINAMATH_CALUDE_expression_simplification_l848_84858

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 / (x - 1) - 1 / (x + 1)) / (2 / ((x - 1)^2)) = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l848_84858


namespace NUMINAMATH_CALUDE_largest_nested_root_l848_84823

theorem largest_nested_root : 
  let a := (7 : ℝ)^(1/4) * 8^(1/12)
  let b := 8^(1/2) * 7^(1/8)
  let c := 7^(1/2) * 8^(1/8)
  let d := 7^(1/3) * 8^(1/6)
  let e := 8^(1/3) * 7^(1/6)
  b > a ∧ b > c ∧ b > d ∧ b > e :=
by sorry

end NUMINAMATH_CALUDE_largest_nested_root_l848_84823


namespace NUMINAMATH_CALUDE_smallest_r_in_special_progression_l848_84817

theorem smallest_r_in_special_progression (p q r : ℤ) : 
  p < q → q < r → 
  q^2 = p * r →  -- Geometric progression condition
  2 * q = p + r →  -- Arithmetic progression condition
  ∀ (p' q' r' : ℤ), p' < q' → q' < r' → q'^2 = p' * r' → 2 * q' = p' + r' → r ≤ r' →
  r = 4 := by
sorry

end NUMINAMATH_CALUDE_smallest_r_in_special_progression_l848_84817


namespace NUMINAMATH_CALUDE_team_combinations_theorem_l848_84828

/-- The number of ways to select k elements from n elements --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid team combinations --/
def validCombinations (totalMale totalFemale teamSize : ℕ) : ℕ :=
  (choose totalMale 1 * choose totalFemale 2) +
  (choose totalMale 2 * choose totalFemale 1)

theorem team_combinations_theorem :
  validCombinations 5 4 3 = 70 := by sorry

end NUMINAMATH_CALUDE_team_combinations_theorem_l848_84828


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l848_84829

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ
  side : ℝ

/-- The theorem stating the relationship between the trapezoid's properties -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.base1 = 9) 
  (h2 : t.base2 = 15) 
  (h3 : t.area = 48) : 
  t.side = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l848_84829


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l848_84813

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l848_84813


namespace NUMINAMATH_CALUDE_min_max_sum_l848_84851

theorem min_max_sum (a b c d e f : ℕ+) (h : a + b + c + d + e + f = 1800) :
  361 ≤ max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) := by
  sorry

end NUMINAMATH_CALUDE_min_max_sum_l848_84851


namespace NUMINAMATH_CALUDE_tan_product_identity_l848_84847

theorem tan_product_identity : (1 + Real.tan (28 * π / 180)) * (1 + Real.tan (17 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_identity_l848_84847


namespace NUMINAMATH_CALUDE_line_translation_theorem_l848_84878

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line vertically and horizontally -/
def translateLine (l : Line) (vertical : ℝ) (horizontal : ℝ) : Line :=
  { slope := l.slope,
    yIntercept := l.yIntercept - vertical - l.slope * horizontal }

theorem line_translation_theorem :
  let initialLine : Line := { slope := 2, yIntercept := 1 }
  let translatedLine := translateLine initialLine 3 2
  translatedLine = { slope := 2, yIntercept := -6 } := by sorry

end NUMINAMATH_CALUDE_line_translation_theorem_l848_84878


namespace NUMINAMATH_CALUDE_exactly_two_valid_sets_l848_84838

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)  -- The first integer in the set
  (length : ℕ) -- The number of integers in the set

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set according to our conditions -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 3 ∧ sum_consecutive s = 18

/-- The main theorem to prove -/
theorem exactly_two_valid_sets :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ ∀ s ∈ sets, is_valid_set s :=
sorry

end NUMINAMATH_CALUDE_exactly_two_valid_sets_l848_84838


namespace NUMINAMATH_CALUDE_saloon_prices_l848_84861

/-- The cost of items in a saloon -/
structure SaloonPrices where
  sandwich : ℚ
  coffee : ℚ
  donut : ℚ

/-- The total cost of a purchase -/
def total_cost (p : SaloonPrices) (s c d : ℕ) : ℚ :=
  s * p.sandwich + c * p.coffee + d * p.donut

/-- The prices in the saloon satisfy the given conditions -/
def satisfies_conditions (p : SaloonPrices) : Prop :=
  total_cost p 4 1 10 = 169/100 ∧ total_cost p 3 1 7 = 126/100

theorem saloon_prices (p : SaloonPrices) (h : satisfies_conditions p) :
  total_cost p 1 1 1 = 40/100 := by
  sorry

end NUMINAMATH_CALUDE_saloon_prices_l848_84861


namespace NUMINAMATH_CALUDE_inequality_solution_l848_84856

def inequality (x : ℝ) : Prop :=
  2*x^4 + x^2 - 2*x - 3*x^2*|x-1| + 1 ≥ 0

def solution_set : Set ℝ :=
  {x | x ≤ -(1 + Real.sqrt 5)/2 ∨ 
       (-1 ≤ x ∧ x ≤ 1/2) ∨ 
       x ≥ (Real.sqrt 5 - 1)/2}

theorem inequality_solution : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l848_84856


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l848_84895

theorem solution_satisfies_system : ∃ (x y : ℝ), 
  (y = 2 - x ∧ 3 * x = 1 + 2 * y) ∧ (x = 1 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l848_84895


namespace NUMINAMATH_CALUDE_line_slopes_product_l848_84824

theorem line_slopes_product (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (∃ θ : ℝ, m = Real.tan (3 * θ) ∧ n = Real.tan θ) →
  m = 9 * n →
  m * n = 81 / 13 := by
  sorry

end NUMINAMATH_CALUDE_line_slopes_product_l848_84824


namespace NUMINAMATH_CALUDE_democrats_ratio_l848_84836

/-- Proves that the ratio of democrats to total participants is 1:3 -/
theorem democrats_ratio (total : ℕ) (female_democrats : ℕ) :
  total = 810 →
  female_democrats = 135 →
  let female := 2 * female_democrats
  let male := total - female
  let male_democrats := male / 4
  let total_democrats := female_democrats + male_democrats
  (total_democrats : ℚ) / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_democrats_ratio_l848_84836


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l848_84805

theorem art_gallery_pieces (total : ℕ) 
  (h1 : total / 3 = total / 3)  -- 1/3 of pieces are on display
  (h2 : (total / 3) / 6 = (total / 3) / 6)  -- 1/6 of displayed pieces are sculptures
  (h3 : (total * 2 / 3) / 3 = (total * 2 / 3) / 3)  -- 1/3 of non-displayed pieces are paintings
  (h4 : total * 2 / 3 * 2 / 3 = 800)  -- 800 sculptures are not on display
  : total = 1800 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l848_84805


namespace NUMINAMATH_CALUDE_equivalent_coin_value_l848_84852

theorem equivalent_coin_value : ∀ (quarter_value dime_value : ℕ),
  quarter_value = 25 →
  dime_value = 10 →
  30 * quarter_value + 20 * dime_value = 15 * quarter_value + 58 * dime_value :=
by
  sorry

end NUMINAMATH_CALUDE_equivalent_coin_value_l848_84852


namespace NUMINAMATH_CALUDE_square_area_is_49_l848_84809

-- Define the right triangle ABC
structure RightTriangle :=
  (AB : ℝ)
  (BC : ℝ)
  (is_right : True)  -- Placeholder for the right angle condition

-- Define the square BDEF
structure Square :=
  (side : ℝ)

-- Define the triangle EMN
structure TriangleEMN :=
  (EH : ℝ)

-- Main theorem
theorem square_area_is_49 
  (triangle : RightTriangle)
  (square : Square)
  (triangle_EMN : TriangleEMN)
  (h1 : triangle.AB = 15)
  (h2 : triangle.BC = 20)
  (h3 : triangle_EMN.EH = 2) :
  square.side ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_is_49_l848_84809


namespace NUMINAMATH_CALUDE_expression_evaluation_l848_84844

theorem expression_evaluation :
  let a : ℝ := 3 + Real.sqrt 5
  let b : ℝ := 3 - Real.sqrt 5
  ((a^2 - 2*a*b + b^2) / (a^2 - b^2)) * ((a*b) / (a - b)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l848_84844


namespace NUMINAMATH_CALUDE_original_proposition_contrapositive_converse_false_inverse_false_negation_false_l848_84814

-- Original proposition
theorem original_proposition : ∀ x : ℝ, x = 1 → x^2 = 1 := by sorry

-- Contrapositive
theorem contrapositive : ∀ x : ℝ, x^2 ≠ 1 → x ≠ 1 := by sorry

-- Converse (as a counterexample)
theorem converse_false : ∃ x : ℝ, x^2 = 1 ∧ x ≠ 1 := by sorry

-- Inverse (as a counterexample)
theorem inverse_false : ∃ x : ℝ, x ≠ 1 ∧ x^2 = 1 := by sorry

-- Negation (as false)
theorem negation_false : ¬(∀ x : ℝ, x = 1 → x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_original_proposition_contrapositive_converse_false_inverse_false_negation_false_l848_84814


namespace NUMINAMATH_CALUDE_equality_proof_l848_84873

theorem equality_proof : 2222 - 222 + 22 - 2 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_equality_proof_l848_84873


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l848_84886

-- Problem 1
theorem calculation_proof : 
  Real.sqrt 4 + 2 * Real.sin (45 * π / 180) - (π - 3)^0 + |Real.sqrt 2 - 2| = 3 := by sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) : 
  (2 * (x + 2) - x ≤ 5 ∧ (4 * x + 1) / 3 > x - 1) ↔ (-4 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l848_84886


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l848_84832

-- Define the parabola
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

-- Define a as the x-intercept
def a : ℝ := parabola 0

-- Define b and c as y-intercepts
noncomputable def b : ℝ := (9 - Real.sqrt 33) / 6
noncomputable def c : ℝ := (9 + Real.sqrt 33) / 6

-- Theorem statement
theorem parabola_intercepts_sum : a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l848_84832


namespace NUMINAMATH_CALUDE_lcm_of_6_8_10_l848_84841

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_6_8_10_l848_84841


namespace NUMINAMATH_CALUDE_odd_function_properties_l848_84877

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y
def has_min_value_on (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → v ≤ f x
def has_max_value_on (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v

-- State the theorem
theorem odd_function_properties (f : ℝ → ℝ) :
  is_odd f →
  is_increasing_on f 1 3 →
  has_min_value_on f 7 1 3 →
  is_increasing_on f (-3) (-1) ∧ has_max_value_on f (-7) (-3) (-1) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l848_84877


namespace NUMINAMATH_CALUDE_journey_meet_day_l848_84842

structure Journey where
  start_day : Nat
  meet_day : Nat
  end_day : Nat
  princess_solo_days : Nat
  together_days : Nat

def is_valid_journey (j : Journey) : Prop :=
  j.start_day = 5 ∧  -- Friday
  j.princess_solo_days = 2 ∧
  j.together_days = (j.princess_solo_days * 4) ∧
  j.end_day = j.meet_day + 11 ∧
  (j.meet_day - j.start_day) % 7 = j.princess_solo_days % 7

theorem journey_meet_day (j : Journey) (h : is_valid_journey j) : j.meet_day % 7 = 3 := by
  sorry

#check journey_meet_day

end NUMINAMATH_CALUDE_journey_meet_day_l848_84842


namespace NUMINAMATH_CALUDE_animal_sanctuary_l848_84854

theorem animal_sanctuary (total : ℕ) (difference : ℕ) : total = 450 ∧ difference = 75 → ∃ (dogs cats : ℕ), cats = dogs + difference ∧ dogs + cats = total ∧ cats = 262 := by
  sorry

end NUMINAMATH_CALUDE_animal_sanctuary_l848_84854


namespace NUMINAMATH_CALUDE_sum_c_n_d_n_over_8_n_l848_84803

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sequences c_n and d_n
def c_n_d_n (n : ℕ) : ℂ := (3 + 2 * i) ^ n

-- Define c_n as the real part of c_n_d_n
def c_n (n : ℕ) : ℝ := (c_n_d_n n).re

-- Define d_n as the imaginary part of c_n_d_n
def d_n (n : ℕ) : ℝ := (c_n_d_n n).im

-- State the theorem
theorem sum_c_n_d_n_over_8_n :
  ∑' n, (c_n n * d_n n) / (8 : ℝ) ^ n = 6 / 17 := by sorry

end NUMINAMATH_CALUDE_sum_c_n_d_n_over_8_n_l848_84803


namespace NUMINAMATH_CALUDE_log_simplification_l848_84894

theorem log_simplification (a b c d x y : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hx : 0 < x) (hy : 0 < y) : 
  Real.log (a / b) + Real.log (b / c) + Real.log (c / d) - Real.log ((a * y) / (d * x)) = Real.log (x / y) := by
  sorry

end NUMINAMATH_CALUDE_log_simplification_l848_84894


namespace NUMINAMATH_CALUDE_rogers_initial_money_l848_84875

theorem rogers_initial_money (game_cost toy_cost num_toys : ℕ) 
  (h1 : game_cost = 48)
  (h2 : toy_cost = 3)
  (h3 : num_toys = 5)
  (h4 : ∃ (remaining : ℕ), remaining = num_toys * toy_cost) :
  game_cost + num_toys * toy_cost = 63 := by
sorry

end NUMINAMATH_CALUDE_rogers_initial_money_l848_84875


namespace NUMINAMATH_CALUDE_proportion_equation_l848_84884

theorem proportion_equation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 2 * x = 3 * y) :
  x / 3 = y / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equation_l848_84884


namespace NUMINAMATH_CALUDE_find_x_l848_84850

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 7 = 17 ∧ x = 38 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l848_84850


namespace NUMINAMATH_CALUDE_average_of_abc_l848_84874

theorem average_of_abc (A B C : ℚ) : 
  A = 2 → 
  2002 * C - 1001 * A = 8008 → 
  2002 * B + 3003 * A = 7007 → 
  (A + B + C) / 3 = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_l848_84874


namespace NUMINAMATH_CALUDE_smallest_covering_segment_l848_84881

/-- Represents an equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_unit : side_length = 1

/-- Represents a sliding segment in the triangle -/
structure SlidingSegment where
  length : ℝ
  covers_triangle : Prop

/-- The smallest sliding segment that covers the triangle has length 2/3 -/
theorem smallest_covering_segment (triangle : EquilateralTriangle) :
  ∃ (d : ℝ), d = 2/3 ∧ 
  (∀ (s : SlidingSegment), s.covers_triangle → s.length ≥ d) ∧
  (∃ (s : SlidingSegment), s.covers_triangle ∧ s.length = d) :=
sorry

end NUMINAMATH_CALUDE_smallest_covering_segment_l848_84881


namespace NUMINAMATH_CALUDE_logarithm_equation_l848_84883

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem logarithm_equation : log10 5 * log10 50 - log10 2 * log10 20 - log10 625 = -2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equation_l848_84883


namespace NUMINAMATH_CALUDE_other_number_proof_l848_84827

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 24)
  (h2 : Nat.gcd a b = 4)
  (h3 : a = 12) : 
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l848_84827


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l848_84822

theorem fixed_point_on_line (m : ℝ) : 
  (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l848_84822


namespace NUMINAMATH_CALUDE_no_valid_A_l848_84818

theorem no_valid_A : ¬∃ (A : ℕ), A < 10 ∧ 81 % A = 0 ∧ (456200 + A * 10 + 4) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_A_l848_84818


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l848_84855

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (sum_eq_10 : x + y = 10) (sum_eq_5prod : x + y = 5 * x * y) : 
  1 / x + 1 / y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l848_84855


namespace NUMINAMATH_CALUDE_fran_ate_15_green_macaroons_l848_84898

/-- The number of green macaroons Fran ate -/
def green_eaten : ℕ := sorry

/-- The number of red macaroons Fran baked -/
def red_baked : ℕ := 50

/-- The number of green macaroons Fran baked -/
def green_baked : ℕ := 40

/-- The number of macaroons remaining -/
def remaining : ℕ := 45

theorem fran_ate_15_green_macaroons :
  green_eaten = 15 ∧
  red_baked = 50 ∧
  green_baked = 40 ∧
  remaining = 45 ∧
  red_baked + green_baked = green_eaten + 2 * green_eaten + remaining :=
sorry

end NUMINAMATH_CALUDE_fran_ate_15_green_macaroons_l848_84898


namespace NUMINAMATH_CALUDE_soccer_handshakes_l848_84849

/-- Calculates the total number of handshakes in a soccer match -/
theorem soccer_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 11 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size * (num_teams - 1) / 2) + (team_size * num_teams * num_referees) = 187 := by
  sorry

end NUMINAMATH_CALUDE_soccer_handshakes_l848_84849


namespace NUMINAMATH_CALUDE_hit_probability_random_gun_selection_l848_84802

/-- The probability of hitting a target when randomly selecting a gun from a set of calibrated and uncalibrated guns. -/
theorem hit_probability_random_gun_selection 
  (total_guns : ℕ) 
  (calibrated_guns : ℕ) 
  (uncalibrated_guns : ℕ) 
  (calibrated_accuracy : ℝ) 
  (uncalibrated_accuracy : ℝ) 
  (h1 : total_guns = 5)
  (h2 : calibrated_guns = 3)
  (h3 : uncalibrated_guns = 2)
  (h4 : calibrated_guns + uncalibrated_guns = total_guns)
  (h5 : calibrated_accuracy = 0.9)
  (h6 : uncalibrated_accuracy = 0.4) :
  (calibrated_guns : ℝ) / total_guns * calibrated_accuracy + 
  (uncalibrated_guns : ℝ) / total_guns * uncalibrated_accuracy = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_hit_probability_random_gun_selection_l848_84802


namespace NUMINAMATH_CALUDE_circle_radius_l848_84862

/-- The circle C is defined by the equation x^2 + y^2 - 4x - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y + 1 = 0

/-- The radius of a circle is the distance from its center to any point on the circle -/
def is_radius (r : ℝ) (center : ℝ × ℝ) (equation : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, equation x y → (x - center.1)^2 + (y - center.2)^2 = r^2

/-- The radius of the circle C defined by x^2 + y^2 - 4x - 2y + 1 = 0 is equal to 2 -/
theorem circle_radius : ∃ center, is_radius 2 center circle_equation := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l848_84862


namespace NUMINAMATH_CALUDE_smallest_k_for_f_divides_l848_84866

/-- The polynomial z^12 + z^11 + z^7 + z^6 + z^5 + z + 1 -/
def f (z : ℂ) : ℂ := z^12 + z^11 + z^7 + z^6 + z^5 + z + 1

/-- Proposition: 91 is the smallest positive integer k such that f(z) divides z^k - 1 -/
theorem smallest_k_for_f_divides : ∀ z : ℂ, z ≠ 0 →
  (∀ k : ℕ, k > 0 → k < 91 → ¬(f z ∣ z^k - 1)) ∧
  (f z ∣ z^91 - 1) := by
  sorry

#check smallest_k_for_f_divides

end NUMINAMATH_CALUDE_smallest_k_for_f_divides_l848_84866


namespace NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l848_84831

theorem smallest_bound_for_cubic_inequality :
  ∃ (M : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2) ∧
  (∀ (M' : ℝ), (∀ (a b c : ℝ),
    |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M' * (a^2 + b^2 + c^2)^2) → M ≤ M') ∧
  M = (9 * Real.sqrt 2) / 32 :=
sorry

end NUMINAMATH_CALUDE_smallest_bound_for_cubic_inequality_l848_84831


namespace NUMINAMATH_CALUDE_strawberry_pancakes_l848_84825

theorem strawberry_pancakes (total : ℕ) (blueberry : ℕ) (banana : ℕ) (chocolate : ℕ) 
  (h1 : total = 150)
  (h2 : blueberry = 45)
  (h3 : banana = 60)
  (h4 : chocolate = 25) :
  total - (blueberry + banana + chocolate) = 20 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_pancakes_l848_84825


namespace NUMINAMATH_CALUDE_identical_angular_acceleration_l848_84846

/-- Two wheels with identical masses and different radii have identical angular accelerations -/
theorem identical_angular_acceleration (m : ℝ) (R₁ R₂ F₁ F₂ : ℝ) 
  (h_m : m = 1)
  (h_R₁ : R₁ = 0.5)
  (h_R₂ : R₂ = 1)
  (h_F₁ : F₁ = 1)
  (h_positive : m > 0 ∧ R₁ > 0 ∧ R₂ > 0 ∧ F₁ > 0 ∧ F₂ > 0) :
  (F₁ * R₁ / (m * R₁^2) = F₂ * R₂ / (m * R₂^2)) → F₂ = 2 := by
  sorry

#check identical_angular_acceleration

end NUMINAMATH_CALUDE_identical_angular_acceleration_l848_84846


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l848_84880

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c - k }

theorem parabola_shift_theorem (x y : ℝ) :
  let original := Parabola.mk 3 0 0
  let shifted := shift_parabola original 1 2
  y = 3 * x^2 → y = 3 * (x - 1)^2 - 2 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l848_84880


namespace NUMINAMATH_CALUDE_pyramid_properties_l848_84860

/-- Represents a pyramid ABCD with given edge lengths -/
structure Pyramid where
  DA : ℝ
  DB : ℝ
  DC : ℝ
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- The specific pyramid from the problem -/
def specific_pyramid : Pyramid :=
  { DA := 15
    DB := 12
    DC := 12
    AB := 9
    AC := 9
    BC := 3 }

/-- Calculates the radius of the circumscribed sphere around the pyramid -/
def circumscribed_sphere_radius (p : Pyramid) : ℝ := sorry

/-- Calculates the volume of the pyramid -/
def pyramid_volume (p : Pyramid) : ℝ := sorry

theorem pyramid_properties :
  circumscribed_sphere_radius specific_pyramid = 7.5 ∧
  pyramid_volume specific_pyramid = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_properties_l848_84860


namespace NUMINAMATH_CALUDE_largest_percent_error_rectangular_plot_l848_84897

theorem largest_percent_error_rectangular_plot (length width : ℝ) 
  (h_length : length = 15)
  (h_width : width = 10)
  (h_error : ℝ) (h_error_bound : h_error = 0.1) : 
  let actual_area := length * width
  let max_length := length * (1 + h_error)
  let max_width := width * (1 + h_error)
  let max_area := max_length * max_width
  let max_percent_error := (max_area - actual_area) / actual_area * 100
  max_percent_error = 21 := by sorry

end NUMINAMATH_CALUDE_largest_percent_error_rectangular_plot_l848_84897


namespace NUMINAMATH_CALUDE_yah_to_bah_conversion_l848_84837

-- Define the exchange rates
def bah_to_rah_rate : ℚ := 30 / 18
def rah_to_yah_rate : ℚ := 25 / 10

-- Define the conversion function
def convert_yah_to_bah (yahs : ℚ) : ℚ :=
  yahs / (rah_to_yah_rate * bah_to_rah_rate)

-- Theorem statement
theorem yah_to_bah_conversion :
  convert_yah_to_bah 1250 = 300 := by
  sorry

end NUMINAMATH_CALUDE_yah_to_bah_conversion_l848_84837


namespace NUMINAMATH_CALUDE_expression_simplification_l848_84801

theorem expression_simplification (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a * b^2 = c / a - b) :
  let expr := (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c)) /
               (2 / (a * b) - 2 * a * b / c) /
               (101 / c)
  expr = -1 / 202 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l848_84801


namespace NUMINAMATH_CALUDE_chess_tournament_solution_l848_84816

/-- Chess tournament with n women and 2n men -/
structure ChessTournament (n : ℕ) where
  women : Fin n
  men : Fin (2 * n)

/-- The number of games played in the tournament -/
def total_games (n : ℕ) : ℕ :=
  n * (3 * n - 1) / 2

/-- The number of games won by women -/
def women_wins (n : ℕ) : ℚ :=
  (n * (n - 1) / 2) + (17 * n^2 - 3 * n) / 8

/-- The number of games won by men -/
def men_wins (n : ℕ) : ℚ :=
  (n * (2 * n - 1)) + (3 * n / 8)

/-- The theorem stating that n must equal 3 -/
theorem chess_tournament_solution : 
  ∃ (n : ℕ), n > 0 ∧ 
  7 * (men_wins n) = 5 * (women_wins n) ∧
  (women_wins n).isInt ∧ (men_wins n).isInt :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_solution_l848_84816


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l848_84857

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (∀ m : ℕ, m < n → ∃ j : ℕ, j ≤ 10 ∧ j > 0 ∧ m % j ≠ 0) ∧
  n = 2520 :=
by sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l848_84857


namespace NUMINAMATH_CALUDE_train_route_encoding_l848_84826

def encode_letter (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

def decode_digit (n : ℕ) : Char :=
  Char.ofNat (n + 'A'.toNat - 1)

def encode_city (s : String) : List ℕ :=
  s.toList.map encode_letter

theorem train_route_encoding :
  (encode_city "UFA" = [21, 6, 1]) ∧
  (encode_city "BAKU" = [2, 1, 11, 21]) →
  "21221-211221".splitOn "-" = ["21221", "211221"] →
  ∃ (departure arrival : String),
    departure = "UFA" ∧
    arrival = "BAKU" ∧
    encode_city departure = [21, 6, 1] ∧
    encode_city arrival = [2, 1, 11, 21] :=
by sorry

end NUMINAMATH_CALUDE_train_route_encoding_l848_84826


namespace NUMINAMATH_CALUDE_prime_sum_fraction_l848_84806

theorem prime_sum_fraction (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hdistinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (ha : ∃ (a : ℕ), a = (p + q) / r + (q + r) / p + (r + p) / q) :
  ∃ (a : ℕ), a = 7 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_fraction_l848_84806


namespace NUMINAMATH_CALUDE_prism_volume_is_400_l848_84871

/-- The volume of a right rectangular prism with face areas 40, 50, and 80 square centimeters -/
def prism_volume : ℝ := 400

/-- The areas of the three faces of the prism -/
def face_area_1 : ℝ := 40
def face_area_2 : ℝ := 50
def face_area_3 : ℝ := 80

/-- Theorem: The volume of the prism is 400 cubic centimeters -/
theorem prism_volume_is_400 :
  ∃ (a b c : ℝ),
    a * b = face_area_1 ∧
    a * c = face_area_2 ∧
    b * c = face_area_3 ∧
    a * b * c = prism_volume :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_is_400_l848_84871


namespace NUMINAMATH_CALUDE_min_value_expression_l848_84870

theorem min_value_expression (x y : ℝ) : (x^2*y + x*y^2 - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l848_84870


namespace NUMINAMATH_CALUDE_rectangle_side_greater_than_twelve_l848_84893

theorem rectangle_side_greater_than_twelve 
  (a b : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a > 0) 
  (h3 : b > 0) 
  (h4 : a * b = 3 * (2 * a + 2 * b)) : 
  a > 12 ∨ b > 12 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_greater_than_twelve_l848_84893


namespace NUMINAMATH_CALUDE_min_period_sin_squared_l848_84843

theorem min_period_sin_squared (f : ℝ → ℝ) (C : ℝ) :
  (∀ x : ℝ, f x = Real.sin x ^ 2) →
  (∀ x : ℝ, f x = f (x + C)) →
  C > 0 →
  (∀ D : ℝ, D > 0 → (∀ x : ℝ, f x = f (x + D)) → C ≤ D) →
  C = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_min_period_sin_squared_l848_84843


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l848_84811

/-- A regular tetrahedron in three-dimensional space -/
structure RegularTetrahedron where
  -- Define the properties of a regular tetrahedron here
  -- (We don't need to fully define it for this statement)

/-- A point inside a regular tetrahedron -/
structure InnerPoint (t : RegularTetrahedron) where
  -- Define the properties of an inner point here
  -- (We don't need to fully define it for this statement)

/-- The sum of distances from a point to all faces of a regular tetrahedron -/
def sum_of_distances_to_faces (t : RegularTetrahedron) (p : InnerPoint t) : ℝ :=
  sorry -- Definition would go here

/-- Theorem stating that the sum of distances from any point inside a regular tetrahedron to its faces is constant -/
theorem sum_of_distances_constant (t : RegularTetrahedron) :
  ∃ c : ℝ, ∀ p : InnerPoint t, sum_of_distances_to_faces t p = c :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_constant_l848_84811


namespace NUMINAMATH_CALUDE_dish_price_theorem_l848_84888

/-- The original price of a dish that satisfies the given conditions -/
def original_price : ℝ := 40

/-- John's total payment -/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's total payment -/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions -/
theorem dish_price_theorem : 
  john_payment original_price = jane_payment original_price + 0.60 := by
  sorry

#eval original_price

end NUMINAMATH_CALUDE_dish_price_theorem_l848_84888


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l848_84890

theorem power_mod_thirteen :
  5^2023 ≡ 8 [ZMOD 13] := by
sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l848_84890
