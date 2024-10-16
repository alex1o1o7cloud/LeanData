import Mathlib

namespace NUMINAMATH_CALUDE_positive_root_condition_negative_root_condition_zero_root_condition_l3611_361133

-- Define the equation ax = b - c
def equation (a b c x : ℝ) : Prop := a * x = b - c

-- Theorem for positive root condition
theorem positive_root_condition (a b c : ℝ) :
  (∃ x > 0, equation a b c x) ↔ (a > 0 ∧ b > c) ∨ (a < 0 ∧ b < c) :=
sorry

-- Theorem for negative root condition
theorem negative_root_condition (a b c : ℝ) :
  (∃ x < 0, equation a b c x) ↔ (a > 0 ∧ b < c) ∨ (a < 0 ∧ b > c) :=
sorry

-- Theorem for zero root condition
theorem zero_root_condition (a b c : ℝ) :
  (∃ x, x = 0 ∧ equation a b c x) ↔ (a ≠ 0 ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_positive_root_condition_negative_root_condition_zero_root_condition_l3611_361133


namespace NUMINAMATH_CALUDE_total_skittles_l3611_361185

theorem total_skittles (num_students : ℕ) (skittles_per_student : ℕ) 
  (h1 : num_students = 9)
  (h2 : skittles_per_student = 3) :
  num_students * skittles_per_student = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_skittles_l3611_361185


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l3611_361150

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l3611_361150


namespace NUMINAMATH_CALUDE_franks_work_days_l3611_361198

/-- Frank's work schedule problem -/
theorem franks_work_days 
  (hours_per_day : ℕ) 
  (total_hours : ℕ) 
  (h1 : hours_per_day = 8) 
  (h2 : total_hours = 32) : 
  total_hours / hours_per_day = 4 := by
  sorry

end NUMINAMATH_CALUDE_franks_work_days_l3611_361198


namespace NUMINAMATH_CALUDE_intersection_M_N_l3611_361114

def M : Set ℝ := {x | (x - 1)^2 < 4}
def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3611_361114


namespace NUMINAMATH_CALUDE_rays_grocery_bill_l3611_361104

/-- Calculates the total grocery bill for Ray's purchase with a rewards discount --/
theorem rays_grocery_bill :
  let hamburger_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let total_before_discount : ℚ := 
    hamburger_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  
  let discount_amount : ℚ := total_before_discount * discount_rate
  
  let final_bill : ℚ := total_before_discount - discount_amount

  final_bill = 18 := by sorry

end NUMINAMATH_CALUDE_rays_grocery_bill_l3611_361104


namespace NUMINAMATH_CALUDE_equilateral_triangle_height_and_area_l3611_361127

/-- Given an equilateral triangle with side length a, prove its height and area. -/
theorem equilateral_triangle_height_and_area (a : ℝ) (h : a > 0) :
  ∃ (height area : ℝ),
    height = (Real.sqrt 3 / 2) * a ∧
    area = (Real.sqrt 3 / 4) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_height_and_area_l3611_361127


namespace NUMINAMATH_CALUDE_connors_garage_wheels_l3611_361143

/-- Calculates the total number of wheels in Connor's garage -/
theorem connors_garage_wheels :
  let num_bicycles : ℕ := 20
  let num_cars : ℕ := 10
  let num_motorcycles : ℕ := 5
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  let wheels_per_motorcycle : ℕ := 2
  (num_bicycles * wheels_per_bicycle + 
   num_cars * wheels_per_car + 
   num_motorcycles * wheels_per_motorcycle) = 90 := by
  sorry

end NUMINAMATH_CALUDE_connors_garage_wheels_l3611_361143


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3611_361145

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 9) * (x^2 + 6*x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3611_361145


namespace NUMINAMATH_CALUDE_sequence_divisibility_l3611_361184

theorem sequence_divisibility (m n k : ℕ) (a : ℕ → ℕ) (hm : m > 1) (hn : n ≥ 0) :
  m^n ∣ a k → m^(n+1) ∣ (a (k+1))^m - (a (k-1))^m :=
by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l3611_361184


namespace NUMINAMATH_CALUDE_ramesh_investment_l3611_361125

def suresh_investment : ℕ := 24000
def total_profit : ℕ := 19000
def ramesh_profit_share : ℕ := 11875

theorem ramesh_investment :
  ∃ (ramesh_investment : ℕ),
    (ramesh_investment * suresh_investment * ramesh_profit_share
     = (total_profit - ramesh_profit_share) * suresh_investment * total_profit)
    ∧ ramesh_investment = 42000 :=
by sorry

end NUMINAMATH_CALUDE_ramesh_investment_l3611_361125


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l3611_361146

/-- A T-shaped figure composed of squares -/
structure TShape where
  side_length : ℝ
  is_t_shaped : Bool
  horizontal_squares : ℕ
  vertical_squares : ℕ

/-- Calculate the perimeter of a T-shaped figure -/
def perimeter (t : TShape) : ℝ :=
  sorry

/-- Theorem: The perimeter of the specific T-shaped figure is 18 -/
theorem t_shape_perimeter :
  ∃ (t : TShape),
    t.side_length = 2 ∧
    t.is_t_shaped = true ∧
    t.horizontal_squares = 3 ∧
    t.vertical_squares = 1 ∧
    perimeter t = 18 :=
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l3611_361146


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3611_361183

theorem geometric_sequence_seventh_term 
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a = 2) -- first term is 2
  (h2 : a * r^8 = 32) -- last term (9th term) is 32
  : a * r^6 = 128 := by -- seventh term is 128
sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l3611_361183


namespace NUMINAMATH_CALUDE_max_value_of_f_l3611_361199

/-- The quadratic function f(x) = -2x^2 + 4x + 3 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

/-- The maximum value of f(x) for x ∈ ℝ is 5 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3611_361199


namespace NUMINAMATH_CALUDE_initial_women_count_l3611_361157

def work_completion (women : ℕ) (children : ℕ) (days : ℕ) : Prop :=
  women * days + children * days = women * 7

theorem initial_women_count : ∃ x : ℕ,
  work_completion x 0 7 ∧
  work_completion 0 10 14 ∧
  work_completion 5 10 4 ∧
  x = 4 := by sorry

end NUMINAMATH_CALUDE_initial_women_count_l3611_361157


namespace NUMINAMATH_CALUDE_programming_methods_count_l3611_361126

/-- Represents the number of subprograms -/
def num_subprograms : ℕ := 6

/-- Represents the number of fixed positions (A, B, C, D in order) -/
def fixed_positions : ℕ := 4

/-- Represents the number of remaining subprograms to be placed -/
def remaining_subprograms : ℕ := 2

/-- Represents the number of possible positions for the first remaining subprogram -/
def positions_for_first : ℕ := fixed_positions + 1

/-- Represents the number of possible positions for the second remaining subprogram -/
def positions_for_second : ℕ := fixed_positions + 2

/-- The total number of programming methods -/
def total_methods : ℕ := positions_for_first * positions_for_second

theorem programming_methods_count :
  total_methods = 20 :=
sorry

end NUMINAMATH_CALUDE_programming_methods_count_l3611_361126


namespace NUMINAMATH_CALUDE_not_always_greater_slope_for_greater_angle_not_always_inclination_equals_arctan_slope_not_different_angles_for_equal_slopes_l3611_361151

-- Define a straight line
structure Line where
  slope : ℝ
  inclination_angle : ℝ

-- Statement 1
theorem not_always_greater_slope_for_greater_angle : 
  ¬ ∀ (l1 l2 : Line), l1.inclination_angle > l2.inclination_angle → l1.slope > l2.slope :=
sorry

-- Statement 2
theorem not_always_inclination_equals_arctan_slope :
  ¬ ∀ (l : Line), l.slope = Real.tan l.inclination_angle → l.inclination_angle = Real.arctan l.slope :=
sorry

-- Statement 3
theorem not_different_angles_for_equal_slopes :
  ¬ ∃ (l1 l2 : Line), l1.slope = l2.slope ∧ l1.inclination_angle ≠ l2.inclination_angle :=
sorry

end NUMINAMATH_CALUDE_not_always_greater_slope_for_greater_angle_not_always_inclination_equals_arctan_slope_not_different_angles_for_equal_slopes_l3611_361151


namespace NUMINAMATH_CALUDE_smallest_cut_length_five_is_smallest_smallest_integral_cut_l3611_361161

theorem smallest_cut_length (x : ℕ) : x ≥ 5 ↔ ¬(9 - x + 14 - x > 18 - x) :=
  sorry

theorem five_is_smallest : ∀ y : ℕ, y < 5 → (9 - y + 14 - y > 18 - y) :=
  sorry

theorem smallest_integral_cut : 
  ∃ x : ℕ, (x ≥ 5) ∧ (∀ y : ℕ, y < x → (9 - y + 14 - y > 18 - y)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_five_is_smallest_smallest_integral_cut_l3611_361161


namespace NUMINAMATH_CALUDE_cans_collected_l3611_361197

/-- Proves that the number of cans collected is 144 given the recycling rates and total money received -/
theorem cans_collected (can_rate : ℚ) (newspaper_rate : ℚ) (newspaper_collected : ℚ) (total_money : ℚ) :
  can_rate = 1/24 →
  newspaper_rate = 3/10 →
  newspaper_collected = 20 →
  total_money = 12 →
  ∃ (cans : ℚ), cans * can_rate + newspaper_collected * newspaper_rate = total_money ∧ cans = 144 := by
  sorry

end NUMINAMATH_CALUDE_cans_collected_l3611_361197


namespace NUMINAMATH_CALUDE_travel_ways_theorem_l3611_361191

/-- The number of ways to travel between two cities using exactly k buses -/
def travel_ways (n k : ℕ) : ℚ :=
  ((n - 1)^k - (-1)^k) / n

/-- Theorem stating the number of ways to travel between two cities using exactly k buses -/
theorem travel_ways_theorem (n k : ℕ) (hn : n ≥ 2) (hk : k ≥ 1) :
  travel_ways n k = ((n - 1)^k - (-1)^k) / n :=
by
  sorry

#check travel_ways_theorem

end NUMINAMATH_CALUDE_travel_ways_theorem_l3611_361191


namespace NUMINAMATH_CALUDE_marksman_probability_l3611_361147

theorem marksman_probability (p10 p9 p8 : ℝ) 
  (h1 : p10 = 0.20)
  (h2 : p9 = 0.30)
  (h3 : p8 = 0.10) :
  1 - (p10 + p9 + p8) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_marksman_probability_l3611_361147


namespace NUMINAMATH_CALUDE_continuous_function_with_three_preimages_l3611_361148

theorem continuous_function_with_three_preimages :
  ∃ f : ℝ → ℝ, Continuous f ∧
    ∀ y : ℝ, ∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      f x₁ = y ∧ f x₂ = y ∧ f x₃ = y ∧
      ∀ x : ℝ, f x = y → x = x₁ ∨ x = x₂ ∨ x = x₃ :=
by
  sorry

end NUMINAMATH_CALUDE_continuous_function_with_three_preimages_l3611_361148


namespace NUMINAMATH_CALUDE_min_perimeter_rectangle_is_square_l3611_361154

/-- For a rectangle with area S and sides a and b, the perimeter is minimized when it's a square -/
theorem min_perimeter_rectangle_is_square (S : ℝ) (h : S > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = S ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x * y = S →
  2 * (a + b) ≤ 2 * (x + y) ∧
  (2 * (a + b) = 2 * (x + y) → a = b) :=
by sorry


end NUMINAMATH_CALUDE_min_perimeter_rectangle_is_square_l3611_361154


namespace NUMINAMATH_CALUDE_cube_sum_equation_l3611_361107

theorem cube_sum_equation (a b : ℝ) 
  (h1 : a^5 - a^4*b - a^4 + a - b - 1 = 0)
  (h2 : 2*a - 3*b = 1) : 
  a^3 + b^3 = 9 := by sorry

end NUMINAMATH_CALUDE_cube_sum_equation_l3611_361107


namespace NUMINAMATH_CALUDE_max_power_under_500_l3611_361178

theorem max_power_under_500 :
  ∃ (a b : ℕ), 
    b > 1 ∧
    a^b < 500 ∧
    (∀ (c d : ℕ), d > 1 → c^d < 500 → c^d ≤ a^b) ∧
    a = 22 ∧
    b = 2 ∧
    a + b = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_power_under_500_l3611_361178


namespace NUMINAMATH_CALUDE_scooter_gain_percentage_l3611_361186

/-- Calculates the overall gain percentage for three scooters -/
def overall_gain_percentage (purchase_price_A purchase_price_B purchase_price_C : ℚ)
                            (repair_cost_A repair_cost_B repair_cost_C : ℚ)
                            (selling_price_A selling_price_B selling_price_C : ℚ) : ℚ :=
  let total_cost := purchase_price_A + purchase_price_B + purchase_price_C +
                    repair_cost_A + repair_cost_B + repair_cost_C
  let total_revenue := selling_price_A + selling_price_B + selling_price_C
  let total_gain := total_revenue - total_cost
  (total_gain / total_cost) * 100

/-- Theorem stating that the overall gain percentage for the given scooter transactions is 10% -/
theorem scooter_gain_percentage :
  overall_gain_percentage 4700 3500 5400 600 800 1000 5800 4800 7000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_scooter_gain_percentage_l3611_361186


namespace NUMINAMATH_CALUDE_divisibility_theorems_l3611_361167

def divisible_by_7 (n : Int) : Prop := ∃ k : Int, n = 7 * k

theorem divisibility_theorems :
  (∀ a b : Int, divisible_by_7 a ∧ divisible_by_7 b → divisible_by_7 (a + b)) ∧
  (∀ a b : Int, ¬divisible_by_7 (a + b) → ¬divisible_by_7 a ∨ ¬divisible_by_7 b) ∧
  ¬(∀ a b : Int, ¬divisible_by_7 a ∧ ¬divisible_by_7 b → ¬divisible_by_7 (a + b)) ∧
  ¬(∀ a b : Int, divisible_by_7 a ∨ divisible_by_7 b → divisible_by_7 (a + b)) ∧
  ¬(∀ a b : Int, divisible_by_7 (a + b) → divisible_by_7 a ∧ divisible_by_7 b) ∧
  ¬(∀ a b : Int, ¬divisible_by_7 (a + b) → ¬divisible_by_7 a ∧ ¬divisible_by_7 b) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_theorems_l3611_361167


namespace NUMINAMATH_CALUDE_determinant_zero_l3611_361124

theorem determinant_zero (x y z : ℝ) : 
  Matrix.det !![1, x, y+z; 1, x+y, z; 1, x+z, y] = 0 := by
  sorry

end NUMINAMATH_CALUDE_determinant_zero_l3611_361124


namespace NUMINAMATH_CALUDE_quadratic_root_triple_relation_l3611_361128

theorem quadratic_root_triple_relation (a b c : ℝ) :
  (∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ 
              a * y^2 + b * y + c = 0 ∧ 
              y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_triple_relation_l3611_361128


namespace NUMINAMATH_CALUDE_sin_theta_for_point_neg_two_three_l3611_361165

theorem sin_theta_for_point_neg_two_three (θ : Real) :
  (∃ (t : Real), t > 0 ∧ t * Real.cos θ = -2 ∧ t * Real.sin θ = 3) →
  Real.sin θ = 3 / Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_sin_theta_for_point_neg_two_three_l3611_361165


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3611_361123

/-- Proves that a bus stopping for 20 minutes per hour with an average speed of 36 kmph including stoppages has a speed of 54 kmph excluding stoppages. -/
theorem bus_speed_excluding_stoppages 
  (average_speed : ℝ) 
  (stopping_time : ℝ) 
  (h1 : average_speed = 36) 
  (h2 : stopping_time = 20) : ℝ := by
  sorry

#check bus_speed_excluding_stoppages

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3611_361123


namespace NUMINAMATH_CALUDE_counterexample_exists_l3611_361108

theorem counterexample_exists : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3611_361108


namespace NUMINAMATH_CALUDE_complex_number_properties_l3611_361164

theorem complex_number_properties (z : ℂ) (h : z - 2*I = z*I + 4) :
  z = 1 + 3*I ∧ Complex.abs z = Real.sqrt 10 ∧ ((z - 1) / 3)^2023 = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3611_361164


namespace NUMINAMATH_CALUDE_root_count_theorem_l3611_361142

def count_roots (f : ℝ → ℝ) (a b : ℝ) : ℕ :=
  sorry

theorem root_count_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = f (2 + x))
  (h2 : ∀ x, f (7 - x) = f (7 + x))
  (h3 : ∀ x ∈ Set.Icc 0 7, f x = 0 ↔ x = 1 ∨ x = 3) :
  count_roots f (-2005) 2005 = 802 :=
sorry

end NUMINAMATH_CALUDE_root_count_theorem_l3611_361142


namespace NUMINAMATH_CALUDE_ball_final_position_l3611_361153

/-- Represents the possible final positions of the ball -/
inductive FinalPosition
  | B
  | A
  | C

/-- Determines the final position of the ball based on the parity of m and n -/
def finalBallPosition (m n : ℕ) : FinalPosition :=
  if m % 2 = 1 ∧ n % 2 = 1 then FinalPosition.B
  else if m % 2 = 0 ∧ n % 2 = 1 then FinalPosition.A
  else FinalPosition.C

/-- Theorem stating the final position of the ball -/
theorem ball_final_position (m n : ℕ) :
  (m > 0 ∧ n > 0) →
  (finalBallPosition m n = FinalPosition.B ↔ m % 2 = 1 ∧ n % 2 = 1) ∧
  (finalBallPosition m n = FinalPosition.A ↔ m % 2 = 0 ∧ n % 2 = 1) ∧
  (finalBallPosition m n = FinalPosition.C ↔ m % 2 = 1 ∧ n % 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ball_final_position_l3611_361153


namespace NUMINAMATH_CALUDE_equation_has_real_root_l3611_361132

/-- The equation x^4 + 2px^3 + x^3 + 2px + 1 = 0 has at least one real root for all real p. -/
theorem equation_has_real_root (p : ℝ) : ∃ x : ℝ, x^4 + 2*p*x^3 + x^3 + 2*p*x + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l3611_361132


namespace NUMINAMATH_CALUDE_triangle_area_is_168_l3611_361110

-- Define the function representing the curve
def f (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define y-intercept
def y_intercept : ℝ := f 0

-- Theorem statement
theorem triangle_area_is_168 :
  let base : ℝ := x_intercept1 - x_intercept2
  let height : ℝ := y_intercept
  (1/2 : ℝ) * base * height = 168 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_168_l3611_361110


namespace NUMINAMATH_CALUDE_shortest_median_not_longer_than_longest_bisector_shortest_bisector_not_longer_than_longest_altitude_l3611_361140

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define median, angle bisector, and altitude
def median (t : Triangle) : ℝ := sorry
def angle_bisector (t : Triangle) : ℝ := sorry
def altitude (t : Triangle) : ℝ := sorry

-- Theorem 1: The shortest median is never longer than the longest angle bisector
theorem shortest_median_not_longer_than_longest_bisector (t : Triangle) :
  ∀ m b, median t ≤ m → angle_bisector t ≥ b → m ≤ b :=
sorry

-- Theorem 2: The shortest angle bisector is never longer than the longest altitude
theorem shortest_bisector_not_longer_than_longest_altitude (t : Triangle) :
  ∀ b h, angle_bisector t ≤ b → altitude t ≥ h → b ≤ h :=
sorry

end NUMINAMATH_CALUDE_shortest_median_not_longer_than_longest_bisector_shortest_bisector_not_longer_than_longest_altitude_l3611_361140


namespace NUMINAMATH_CALUDE_acid_concentration_proof_l3611_361152

/-- Represents the composition of an acid-water mixture -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The original mixture before any additions -/
def original_mixture : Mixture :=
  { acid := 0,  -- We don't know the initial acid amount
    water := 0 } -- We don't know the initial water amount

theorem acid_concentration_proof :
  -- The total volume of the original mixture is 10 ounces
  original_mixture.acid + original_mixture.water = 10 →
  -- After adding 1 ounce of water, the acid concentration becomes 25%
  original_mixture.acid / (original_mixture.acid + original_mixture.water + 1) = 1/4 →
  -- After adding 1 ounce of acid to the water-added mixture, the concentration becomes 40%
  (original_mixture.acid + 1) / (original_mixture.acid + original_mixture.water + 2) = 2/5 →
  -- Then the original acid concentration was 27.5%
  original_mixture.acid / (original_mixture.acid + original_mixture.water) = 11/40 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_proof_l3611_361152


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3611_361158

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ x y : ℝ, a^2 * x + y + 7 = 0 ∧ x - 2 * a * y + 1 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (a^2 * x₁ + y₁ + 7 = 0 ∧ x₁ - 2 * a * y₁ + 1 = 0) →
    (a^2 * x₂ + y₂ + 7 = 0 ∧ x₂ - 2 * a * y₂ + 1 = 0) →
    (x₂ - x₁) * (a^2 * (x₂ - x₁) + (y₂ - y₁)) = 0) →
  a = 0 ∨ a = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3611_361158


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3611_361188

theorem blue_marble_probability
  (total_marbles : ℕ)
  (red_prob : ℚ)
  (h1 : total_marbles = 30)
  (h2 : red_prob = 32/75) :
  ∃ (x y : ℕ) (r1 r2 : ℕ),
    x + y = total_marbles ∧
    (r1 : ℚ) * r2 / (x * y) = red_prob ∧
    ((x - r1) : ℚ) * (y - r2) / (x * y) = 3/25 := by
  sorry

#eval 3 + 25  -- Expected output: 28

end NUMINAMATH_CALUDE_blue_marble_probability_l3611_361188


namespace NUMINAMATH_CALUDE_proportion_problem_l3611_361168

theorem proportion_problem :
  ∀ x₁ x₂ x₃ x₄ : ℤ,
    (x₁ : ℚ) / x₂ = (x₃ : ℚ) / x₄ ∧
    x₁ = x₂ + 6 ∧
    x₃ = x₄ + 5 ∧
    x₁^2 + x₂^2 + x₃^2 + x₄^2 = 793 →
    ((x₁ = -12 ∧ x₂ = -18 ∧ x₃ = -10 ∧ x₄ = -15) ∨
     (x₁ = 18 ∧ x₂ = 12 ∧ x₃ = 15 ∧ x₄ = 10)) :=
by sorry

end NUMINAMATH_CALUDE_proportion_problem_l3611_361168


namespace NUMINAMATH_CALUDE_equation_solution_l3611_361119

theorem equation_solution : ∃! x : ℚ, (3 / 20 + 3 / x = 8 / x + 1 / 15) ∧ x = 60 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3611_361119


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3611_361121

/-- A right triangle with perimeter 40 and area 24 has a hypotenuse of length 18.8 -/
theorem right_triangle_hypotenuse : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  a + b + c = 40 ∧  -- Perimeter condition
  (1/2) * a * b = 24 ∧  -- Area condition
  a^2 + b^2 = c^2 ∧  -- Pythagorean theorem (right triangle condition)
  c = 18.8 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3611_361121


namespace NUMINAMATH_CALUDE_sixty_seven_in_one_row_l3611_361170

def pascal_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem sixty_seven_in_one_row :
  ∃! row : ℕ, ∃ k : ℕ, pascal_coefficient row k = 67 :=
by
  sorry

end NUMINAMATH_CALUDE_sixty_seven_in_one_row_l3611_361170


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3611_361118

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3611_361118


namespace NUMINAMATH_CALUDE_circle_op_equation_solution_l3611_361109

-- Define the € operation
def circle_op (x y : ℝ) : ℝ := 3 * x * y

-- State the theorem
theorem circle_op_equation_solution :
  ∀ z : ℝ, circle_op (circle_op 4 5) z = 540 → z = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_op_equation_solution_l3611_361109


namespace NUMINAMATH_CALUDE_students_present_l3611_361195

theorem students_present (total : ℕ) (absent_percentage : ℚ) : 
  total = 50 ∧ absent_percentage = 14/100 → 
  total - (total * absent_percentage).floor = 43 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l3611_361195


namespace NUMINAMATH_CALUDE_triangle_side_length_l3611_361134

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = 7 →
  c = 6 →
  Real.cos (B - C) = 31/32 →
  a = (Real.sqrt 299) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3611_361134


namespace NUMINAMATH_CALUDE_tan_sum_pi_24_and_7pi_24_l3611_361130

theorem tan_sum_pi_24_and_7pi_24 :
  Real.tan (π / 24) + Real.tan (7 * π / 24) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_24_and_7pi_24_l3611_361130


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l3611_361120

noncomputable def f (x : ℝ) := (x - 2) * Real.exp x

theorem monotonic_increasing_interval_f :
  {x : ℝ | ∀ y, x < y → f x < f y} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_f_l3611_361120


namespace NUMINAMATH_CALUDE_base_9_conversion_l3611_361100

/-- Converts a list of digits in base 9 to its decimal (base 10) representation -/
def base9ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The problem statement -/
theorem base_9_conversion :
  base9ToDecimal [1, 3, 3, 2] = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_9_conversion_l3611_361100


namespace NUMINAMATH_CALUDE_circle_equation_l3611_361159

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : ℝ := x - y - 1
def line2 (x y : ℝ) : ℝ := 4*x + 3*y + 14
def line3 (x y : ℝ) : ℝ := 3*x + 4*y + 10

-- State the theorem
theorem circle_equation (C : Circle) :
  (∀ x y, line1 x y = 0 → x = C.center.1 ∧ y = C.center.2) →
  (∃ x y, line2 x y = 0 ∧ (x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) →
  (∃ x1 y1 x2 y2, line3 x1 y1 = 0 ∧ line3 x2 y2 = 0 ∧
    (x1 - C.center.1)^2 + (y1 - C.center.2)^2 = C.radius^2 ∧
    (x2 - C.center.1)^2 + (y2 - C.center.2)^2 = C.radius^2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 36) →
  C.center = (2, 1) ∧ C.radius = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3611_361159


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l3611_361181

theorem min_value_complex_expression (a b c : ℤ) (ξ : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_fourth_root : ξ^4 = 1)
  (h_not_one : ξ ≠ 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    Complex.abs (↑x + ↑y * ξ + ↑z * ξ^3) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l3611_361181


namespace NUMINAMATH_CALUDE_sum_of_odd_and_five_times_odd_is_even_l3611_361179

theorem sum_of_odd_and_five_times_odd_is_even (m n : ℕ) 
  (hm : m % 2 = 1) (hn : n % 2 = 1) (hm_pos : 0 < m) (hn_pos : 0 < n) : 
  ∃ k : ℕ, m + 5 * n = 2 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_and_five_times_odd_is_even_l3611_361179


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l3611_361141

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l3611_361141


namespace NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3611_361155

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ 
  n + (n + 1) < 150 ∧ 
  (n + 1)^2 - n^2 = 149 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_difference_l3611_361155


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3611_361106

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3611_361106


namespace NUMINAMATH_CALUDE_min_k_good_is_two_l3611_361173

/-- A function f: ℕ+ → ℕ+ is k-good if for all m ≠ n in ℕ+, (f(m)+n, f(n)+m) ≤ k -/
def IsKGood (k : ℕ) (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → Nat.gcd (f m + n) (f n + m) ≤ k

/-- The minimum k for which a k-good function exists is 2 -/
theorem min_k_good_is_two :
  (∃ k : ℕ, k > 0 ∧ ∃ f : ℕ+ → ℕ+, IsKGood k f) ∧
  (∀ k : ℕ, k > 0 → (∃ f : ℕ+ → ℕ+, IsKGood k f) → k ≥ 2) :=
by sorry

#check min_k_good_is_two

end NUMINAMATH_CALUDE_min_k_good_is_two_l3611_361173


namespace NUMINAMATH_CALUDE_rectangle_area_l3611_361137

theorem rectangle_area (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3611_361137


namespace NUMINAMATH_CALUDE_no_solution_exists_l3611_361131

theorem no_solution_exists : ¬∃ x : ℝ, (81 : ℝ)^(3*x) = (27 : ℝ)^(4*x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3611_361131


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3611_361117

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a + Real.log b > b + Real.log a) ∧
  (∃ a b : ℝ, a + Real.log b > b + Real.log a ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3611_361117


namespace NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3611_361139

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_8_factorial_10_factorial : 
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 := by sorry

end NUMINAMATH_CALUDE_gcd_8_factorial_10_factorial_l3611_361139


namespace NUMINAMATH_CALUDE_range_of_a_l3611_361136

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |x - 2|

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f x a ≤ |x - 4|) ↔ a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3611_361136


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l3611_361187

theorem greatest_multiple_of_four (x : ℕ) : 
  x > 0 → x % 4 = 0 → x^3 < 500 → x ≤ 4 ∧ ∃ y : ℕ, y > 0 ∧ y % 4 = 0 ∧ y^3 < 500 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l3611_361187


namespace NUMINAMATH_CALUDE_pie_weight_theorem_l3611_361115

theorem pie_weight_theorem (total_weight : ℝ) (fridge_weight : ℝ) : 
  (5 / 6 : ℝ) * total_weight = fridge_weight → 
  (1 / 6 : ℝ) * total_weight = 240 :=
by
  sorry

#check pie_weight_theorem 1440 1200

end NUMINAMATH_CALUDE_pie_weight_theorem_l3611_361115


namespace NUMINAMATH_CALUDE_mn_value_l3611_361169

theorem mn_value (M N : ℝ) 
  (h1 : (Real.log N) / (2 * Real.log M) = 2 * Real.log M / Real.log N)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = Real.sqrt N :=
by sorry

end NUMINAMATH_CALUDE_mn_value_l3611_361169


namespace NUMINAMATH_CALUDE_rachel_video_game_score_l3611_361101

/-- Rachel's video game scoring problem -/
theorem rachel_video_game_score :
  let level1_treasures : ℕ := 5
  let level1_points : ℕ := 9
  let level2_treasures : ℕ := 2
  let level2_points : ℕ := 12
  let level3_treasures : ℕ := 8
  let level3_points : ℕ := 15
  let total_score := 
    level1_treasures * level1_points +
    level2_treasures * level2_points +
    level3_treasures * level3_points
  total_score = 189 := by sorry

end NUMINAMATH_CALUDE_rachel_video_game_score_l3611_361101


namespace NUMINAMATH_CALUDE_boys_who_left_l3611_361129

theorem boys_who_left (initial_children final_children girls_entered : ℕ) 
  (h1 : initial_children = 85)
  (h2 : girls_entered = 24)
  (h3 : final_children = 78) :
  initial_children + girls_entered - final_children = 31 := by
  sorry

end NUMINAMATH_CALUDE_boys_who_left_l3611_361129


namespace NUMINAMATH_CALUDE_female_salmon_count_l3611_361172

theorem female_salmon_count (total : Nat) (male : Nat) (h1 : total = 971639) (h2 : male = 712261) :
  total - male = 259378 := by
  sorry

end NUMINAMATH_CALUDE_female_salmon_count_l3611_361172


namespace NUMINAMATH_CALUDE_jens_height_l3611_361112

theorem jens_height (original_height : ℝ) (bakis_growth_rate : ℝ) (jens_growth_ratio : ℝ) (bakis_final_height : ℝ) :
  original_height > 0 ∧ 
  bakis_growth_rate = 0.25 ∧
  jens_growth_ratio = 2/3 ∧
  bakis_final_height = 75 ∧
  bakis_final_height = original_height * (1 + bakis_growth_rate) →
  original_height + jens_growth_ratio * (bakis_final_height - original_height) = 70 :=
by
  sorry

#check jens_height

end NUMINAMATH_CALUDE_jens_height_l3611_361112


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l3611_361113

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def move_first_digit_to_end (n : ℕ) : ℕ :=
  (n % 10000) * 10 + (n / 10000)

theorem unique_five_digit_number : ∃! n : ℕ,
  is_five_digit n ∧
  move_first_digit_to_end n = n + 34767 ∧
  move_first_digit_to_end n + n = 86937 ∧
  n = 26035 := by
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l3611_361113


namespace NUMINAMATH_CALUDE_distance_difference_l3611_361163

/-- The walking speed of Taehyung in meters per minute -/
def taehyung_speed : ℕ := 114

/-- The walking speed of Minyoung in meters per minute -/
def minyoung_speed : ℕ := 79

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem stating the difference in distance walked by Taehyung and Minyoung in an hour -/
theorem distance_difference : 
  taehyung_speed * minutes_per_hour - minyoung_speed * minutes_per_hour = 2100 := by
  sorry


end NUMINAMATH_CALUDE_distance_difference_l3611_361163


namespace NUMINAMATH_CALUDE_sin_570_degrees_l3611_361196

theorem sin_570_degrees : 2 * Real.sin (570 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_sin_570_degrees_l3611_361196


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3611_361189

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) / (x - 1) ≤ 0}
def B : Set ℝ := {x | Real.log x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3611_361189


namespace NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l3611_361156

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the probability of drawing two hearts
def prob_two_hearts : ℚ := (hearts_in_deck : ℚ) / standard_deck * (hearts_in_deck - 1) / (standard_deck - 1)

-- Theorem statement
theorem prob_two_hearts_is_one_seventeenth : 
  prob_two_hearts = 1 / 17 := by sorry

end NUMINAMATH_CALUDE_prob_two_hearts_is_one_seventeenth_l3611_361156


namespace NUMINAMATH_CALUDE_min_value_theorem_l3611_361111

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) :
  (9 / a) + (16 / b) + (25 / c) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3611_361111


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l3611_361175

theorem mystery_book_shelves :
  ∀ (books_per_shelf : ℕ) 
    (picture_book_shelves : ℕ) 
    (total_books : ℕ),
  books_per_shelf = 8 →
  picture_book_shelves = 4 →
  total_books = 72 →
  ∃ (mystery_book_shelves : ℕ),
    mystery_book_shelves * books_per_shelf + 
    picture_book_shelves * books_per_shelf = total_books ∧
    mystery_book_shelves = 5 :=
by sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l3611_361175


namespace NUMINAMATH_CALUDE_gcf_of_3150_and_7350_l3611_361144

theorem gcf_of_3150_and_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_3150_and_7350_l3611_361144


namespace NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l3611_361122

theorem permutations_of_eight_distinct_objects : 
  Nat.factorial 8 = 40320 := by sorry

end NUMINAMATH_CALUDE_permutations_of_eight_distinct_objects_l3611_361122


namespace NUMINAMATH_CALUDE_concert_attendance_l3611_361149

def ticket_price : ℕ := 20
def first_group_size : ℕ := 10
def second_group_size : ℕ := 20
def first_discount : ℚ := 40 / 100
def second_discount : ℚ := 15 / 100
def total_revenue : ℕ := 980

theorem concert_attendance : ∃ (full_price_tickets : ℕ),
  let discounted_revenue := 
    (first_group_size * (ticket_price * (1 - first_discount)).floor) +
    (second_group_size * (ticket_price * (1 - second_discount)).floor)
  let full_price_revenue := total_revenue - discounted_revenue
  full_price_tickets = full_price_revenue / ticket_price ∧
  first_group_size + second_group_size + full_price_tickets = 56 :=
sorry

end NUMINAMATH_CALUDE_concert_attendance_l3611_361149


namespace NUMINAMATH_CALUDE_opposite_of_neg_three_l3611_361192

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem stating that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_three_l3611_361192


namespace NUMINAMATH_CALUDE_isosceles_triangle_l3611_361182

theorem isosceles_triangle (A B C : ℝ) (h₁ : A + B + C = π) 
  (h₂ : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : 
  A = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l3611_361182


namespace NUMINAMATH_CALUDE_solution_ratio_l3611_361177

/-- Given a system of equations with solution (2, 5), prove that a/c = 3 -/
theorem solution_ratio (a c : ℝ) : 
  (∃ x y : ℝ, x = 2 ∧ y = 5 ∧ a * x + 2 * y = 16 ∧ 3 * x - y = c) →
  a / c = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_ratio_l3611_361177


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3611_361193

theorem polynomial_simplification (x : ℝ) :
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6) =
  2 * x^3 + 7 * x^2 - 3 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3611_361193


namespace NUMINAMATH_CALUDE_earbuds_cost_after_tax_l3611_361116

/-- The total amount paid after tax for an item with a given cost and tax rate. -/
def total_amount_after_tax (cost : ℝ) (tax_rate : ℝ) : ℝ :=
  cost * (1 + tax_rate)

/-- Theorem stating that the total amount paid after tax for an item costing $200 with a 15% tax rate is $230. -/
theorem earbuds_cost_after_tax :
  total_amount_after_tax 200 0.15 = 230 := by
  sorry

end NUMINAMATH_CALUDE_earbuds_cost_after_tax_l3611_361116


namespace NUMINAMATH_CALUDE_equation_solution_l3611_361105

theorem equation_solution (k : ℕ+) :
  (∃ (m n : ℕ+), m * (m + k) = n * (n + 1)) ↔ (k = 1 ∨ k ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3611_361105


namespace NUMINAMATH_CALUDE_min_value_theorem_l3611_361162

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∀ z : ℝ, z = 1 / x + 4 / y → z ≥ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3611_361162


namespace NUMINAMATH_CALUDE_f_has_one_root_l3611_361138

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3*x - 2

theorem f_has_one_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_one_root_l3611_361138


namespace NUMINAMATH_CALUDE_shop_width_calculation_l3611_361180

/-- Calculates the width of a shop given its monthly rent, length, and annual rent per square foot. -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) : 
  monthly_rent = 1440 → length = 18 → annual_rent_per_sqft = 48 → 
  (monthly_rent * 12) / (annual_rent_per_sqft * length) = 20 := by
  sorry

end NUMINAMATH_CALUDE_shop_width_calculation_l3611_361180


namespace NUMINAMATH_CALUDE_triangle_similarity_after_bisections_specific_triangle_similarity_l3611_361160

/-- Triangle with sides a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle bisector application --/
def angleBisector (t : Triangle) : Triangle :=
  sorry

/-- Repeated angle bisector application --/
def repeatedAngleBisector (t : Triangle) (n : ℕ) : Triangle :=
  sorry

/-- Similarity ratio between two triangles --/
def similarityRatio (t1 t2 : Triangle) : ℝ :=
  sorry

theorem triangle_similarity_after_bisections (n : ℕ) :
  let original := Triangle.mk 5 6 4
  let final := repeatedAngleBisector original n
  similarityRatio original final = (4/9)^n :=
sorry

theorem specific_triangle_similarity :
  let original := Triangle.mk 5 6 4
  let final := repeatedAngleBisector original 2021
  similarityRatio original final = (4/9)^2021 :=
sorry

end NUMINAMATH_CALUDE_triangle_similarity_after_bisections_specific_triangle_similarity_l3611_361160


namespace NUMINAMATH_CALUDE_sum_product_zero_l3611_361171

theorem sum_product_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_product_zero_l3611_361171


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3611_361194

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I + 1) * (a + 2 * Complex.I) * Complex.I = Complex.I * (a + 2 : ℝ) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3611_361194


namespace NUMINAMATH_CALUDE_five_Y_three_equals_four_l3611_361135

-- Define the Y operation
def Y (a b : ℤ) : ℤ := a^2 - 2*a*b + b^2

-- Theorem statement
theorem five_Y_three_equals_four : Y 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_five_Y_three_equals_four_l3611_361135


namespace NUMINAMATH_CALUDE_dog_park_theorem_l3611_361103

/-- The total number of dogs barking after a new group joins -/
def total_dogs (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial + multiplier * initial

/-- Theorem: Given 30 initial dogs and a new group triple the size, the total is 120 dogs -/
theorem dog_park_theorem :
  total_dogs 30 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_theorem_l3611_361103


namespace NUMINAMATH_CALUDE_granddaughter_mother_age_ratio_l3611_361102

/-- The ratio of a granddaughter's age to her mother's age, given the ages of three generations. -/
theorem granddaughter_mother_age_ratio
  (betty_age : ℕ)
  (daughter_age : ℕ)
  (granddaughter_age : ℕ)
  (h1 : betty_age = 60)
  (h2 : daughter_age = betty_age - (40 * betty_age / 100))
  (h3 : granddaughter_age = 12) :
  granddaughter_age / daughter_age = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_granddaughter_mother_age_ratio_l3611_361102


namespace NUMINAMATH_CALUDE_athletes_leaving_rate_l3611_361176

/-- The rate at which athletes left the camp per hour -/
def leaving_rate : ℝ := 24.5

/-- The initial number of athletes at the camp -/
def initial_athletes : ℕ := 300

/-- The number of hours athletes left the camp -/
def leaving_hours : ℕ := 4

/-- The rate at which new athletes entered the camp per hour -/
def entering_rate : ℕ := 15

/-- The number of hours new athletes entered the camp -/
def entering_hours : ℕ := 7

/-- The difference in total number of athletes over the two nights -/
def athlete_difference : ℕ := 7

theorem athletes_leaving_rate : 
  initial_athletes - leaving_rate * leaving_hours + entering_rate * entering_hours 
  = initial_athletes + athlete_difference :=
sorry

end NUMINAMATH_CALUDE_athletes_leaving_rate_l3611_361176


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3611_361166

theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 8) = -95 + m * x) ↔ 
  (m = -20 - 2 * Real.sqrt 189 ∨ m = -20 + 2 * Real.sqrt 189) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3611_361166


namespace NUMINAMATH_CALUDE_three_true_propositions_l3611_361174

theorem three_true_propositions (a b c d : ℝ) : 
  (∃ (p q r : Prop), 
    (p ∧ q → r) ∧ 
    (p ∧ r → q) ∧ 
    (q ∧ r → p) ∧
    (p = (a * b > 0)) ∧ 
    (q = (-c / a < -d / b)) ∧ 
    (r = (b * c > a * d))) :=
by sorry

end NUMINAMATH_CALUDE_three_true_propositions_l3611_361174


namespace NUMINAMATH_CALUDE_smallest_possible_a_for_parabola_l3611_361190

theorem smallest_possible_a_for_parabola :
  ∀ (a b c : ℚ),
    a > 0 →
    (∃ (n : ℤ), 2 * a + b + 3 * c = n) →
    (∀ (x : ℚ), a * (x - 3/5)^2 - 13/5 = a * x^2 + b * x + c) →
    (∀ (a' : ℚ), a' > 0 ∧ 
      (∃ (b' c' : ℚ) (n' : ℤ), 2 * a' + b' + 3 * c' = n' ∧
        (∀ (x : ℚ), a' * (x - 3/5)^2 - 13/5 = a' * x^2 + b' * x + c')) →
      a ≤ a') →
    a = 45/19 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_for_parabola_l3611_361190
