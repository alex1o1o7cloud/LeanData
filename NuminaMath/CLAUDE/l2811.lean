import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_bisector_intersection_equidistant_l2811_281122

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to find the intersection point of perpendicular bisectors
def intersectionOfPerpendicularBisectors (t : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem perpendicular_bisector_intersection_equidistant (t : Triangle) :
  let P := intersectionOfPerpendicularBisectors t
  distance P t.A = distance P t.B ∧ distance P t.B = distance P t.C := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_intersection_equidistant_l2811_281122


namespace NUMINAMATH_CALUDE_constant_function_property_l2811_281126

theorem constant_function_property (f : ℝ → ℝ) (h : ∀ x, f (4 * x) = 4) :
  ∀ x, f (2 * x) = 4 := by
sorry

end NUMINAMATH_CALUDE_constant_function_property_l2811_281126


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2811_281112

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (θ : ℝ) 
  (ha : a = 8) 
  (hb : b = 15) 
  (hθ : θ = 30 * π / 180) :
  c = Real.sqrt (289 - 120 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2811_281112


namespace NUMINAMATH_CALUDE_inequality_proof_l2811_281157

theorem inequality_proof (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  0 < (x - Real.sin x) / (Real.tan x - Real.sin x) ∧
  (x - Real.sin x) / (Real.tan x - Real.sin x) < 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2811_281157


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2811_281152

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times as long as the other, prove that the number
    of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2811_281152


namespace NUMINAMATH_CALUDE_docked_amount_is_five_l2811_281111

/-- Calculates the amount docked per late arrival given the hourly rate, weekly hours, 
    number of late arrivals, and actual pay. -/
def amount_docked_per_late_arrival (hourly_rate : ℚ) (weekly_hours : ℚ) 
  (late_arrivals : ℕ) (actual_pay : ℚ) : ℚ :=
  ((hourly_rate * weekly_hours) - actual_pay) / late_arrivals

/-- Proves that the amount docked per late arrival is $5 given the specific conditions. -/
theorem docked_amount_is_five :
  amount_docked_per_late_arrival 30 18 3 525 = 5 := by
  sorry

end NUMINAMATH_CALUDE_docked_amount_is_five_l2811_281111


namespace NUMINAMATH_CALUDE_calculate_expression_l2811_281148

theorem calculate_expression : 
  (-2)^2 + Real.sqrt 8 - abs (1 - Real.sqrt 2) + (2023 - Real.pi)^0 = 6 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2811_281148


namespace NUMINAMATH_CALUDE_infiniteContinuedFraction_eq_infiniteContinuedFraction_value_l2811_281178

/-- The value of the infinite continued fraction 1 / (1 + 1 / (1 + ...)) -/
noncomputable def infiniteContinuedFraction : ℝ :=
  Real.sqrt 5 / 2 + 1 / 2

/-- The infinite continued fraction satisfies the equation x = 1 + 1/x -/
theorem infiniteContinuedFraction_eq : 
  infiniteContinuedFraction = 1 + 1 / infiniteContinuedFraction := by
sorry

/-- The infinite continued fraction 1 / (1 + 1 / (1 + ...)) is equal to (√5 + 1) / 2 -/
theorem infiniteContinuedFraction_value : 
  infiniteContinuedFraction = Real.sqrt 5 / 2 + 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_infiniteContinuedFraction_eq_infiniteContinuedFraction_value_l2811_281178


namespace NUMINAMATH_CALUDE_gillian_spending_theorem_l2811_281108

/-- Calculates the total amount Gillian spent at the farmer's market after tax -/
def gillian_total_spending (sandi_initial: ℝ) (sandi_market_fraction: ℝ) (sandi_discount: ℝ) 
  (gillian_extra: ℝ) (gillian_tax: ℝ) : ℝ :=
  let sandi_market := sandi_initial * sandi_market_fraction
  let sandi_after_discount := sandi_market * (1 - sandi_discount)
  let gillian_before_tax := 3 * sandi_after_discount + gillian_extra
  gillian_before_tax * (1 + gillian_tax)

/-- Theorem stating that Gillian's total spending at the farmer's market after tax is $957 -/
theorem gillian_spending_theorem :
  gillian_total_spending 600 0.5 0.2 150 0.1 = 957 := by
  sorry

end NUMINAMATH_CALUDE_gillian_spending_theorem_l2811_281108


namespace NUMINAMATH_CALUDE_hypotenuse_length_l2811_281132

/-- Given a right triangle with an acute angle α and a circle of radius R
    touching the hypotenuse and the extensions of the two legs,
    the length of the hypotenuse is R * (1 - tan(α/2)) / cos(α) -/
theorem hypotenuse_length (α R : Real) (h1 : 0 < α ∧ α < π/2) (h2 : R > 0) :
  ∃ x, x > 0 ∧ x = R * (1 - Real.tan (α/2)) / Real.cos α :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l2811_281132


namespace NUMINAMATH_CALUDE_no_prime_cubic_polynomial_l2811_281185

theorem no_prime_cubic_polynomial :
  ¬ ∃ (n : ℕ), n > 0 ∧ Nat.Prime (n^3 - 9*n^2 + 27*n - 28) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_cubic_polynomial_l2811_281185


namespace NUMINAMATH_CALUDE_two_number_problem_l2811_281115

theorem two_number_problem :
  ∃ (x y : ℕ), x > y ∧ x - y = 4 ∧ x * y = 80 ∧ (Even x ∨ Even y) ∧ x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l2811_281115


namespace NUMINAMATH_CALUDE_sin_four_arcsin_l2811_281170

theorem sin_four_arcsin (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) :
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_four_arcsin_l2811_281170


namespace NUMINAMATH_CALUDE_spells_conversion_l2811_281176

/-- Converts a base-9 number to base-10 --/
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- The number of spells in each book in base-9 --/
def spellsPerBook : List Nat := [5, 3, 6]

theorem spells_conversion :
  base9ToBase10 spellsPerBook = 518 := by
  sorry

#eval base9ToBase10 spellsPerBook

end NUMINAMATH_CALUDE_spells_conversion_l2811_281176


namespace NUMINAMATH_CALUDE_tangent_intersection_monotonicity_intervals_m_range_l2811_281193

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + m / x

theorem tangent_intersection (m : ℝ) :
  (∃ y, y = f m 1 ∧ y - f m 1 = (1 - m) * (0 - 1) ∧ y = 1) → m = 1 := by sorry

theorem monotonicity_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) ∧
  (m > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < m → f m x₁ > f m x₂) ∧
           (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂)) := by sorry

theorem m_range (m : ℝ) :
  (∀ a b, 0 < a ∧ a < b → (f m b - f m a) / (b - a) < 1) → m ≥ 1/4 := by sorry

end NUMINAMATH_CALUDE_tangent_intersection_monotonicity_intervals_m_range_l2811_281193


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2811_281119

def U : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1}
def B : Set Nat := {1, 2, 3}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2811_281119


namespace NUMINAMATH_CALUDE_problem_statement_l2811_281137

theorem problem_statement :
  (∀ x : ℝ, |x| ≥ 0) ∧
  (1^2 + 1 + 1 ≠ 0) ∧
  ((∀ x : ℝ, |x| ≥ 0) ∧ (1^2 + 1 + 1 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2811_281137


namespace NUMINAMATH_CALUDE_carolyn_practice_ratio_l2811_281177

/-- Represents Carolyn's music practice schedule and calculates the ratio of violin to piano practice time -/
theorem carolyn_practice_ratio :
  let piano_daily := 20 -- minutes of piano practice per day
  let days_per_week := 6 -- number of practice days per week
  let weeks_per_month := 4 -- number of weeks in a month
  let total_monthly := 1920 -- total practice time in minutes per month

  let piano_monthly := piano_daily * days_per_week * weeks_per_month
  let violin_monthly := total_monthly - piano_monthly
  let violin_daily := violin_monthly / (days_per_week * weeks_per_month)

  (violin_daily : ℚ) / piano_daily = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_practice_ratio_l2811_281177


namespace NUMINAMATH_CALUDE_range_of_m_l2811_281180

def P (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m : ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ∈ Set.Ioc 1 2 ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2811_281180


namespace NUMINAMATH_CALUDE_system_solutions_l2811_281197

theorem system_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (x₁ + x₂ = x₃^2 ∧
   x₂ + x₃ = x₄^2 ∧
   x₃ + x₁ = x₅^2 ∧
   x₄ + x₅ = x₁^2 ∧
   x₅ + x₁ = x₂^2) →
  ((x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2) ∨
   (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2811_281197


namespace NUMINAMATH_CALUDE_distance_difference_l2811_281199

/-- Clara's travel rate in miles per hour -/
def clara_rate : ℝ := 3.75

/-- Daniel's travel rate in miles per hour -/
def daniel_rate : ℝ := 3

/-- Time period in hours -/
def time : ℝ := 5

/-- Theorem stating the difference in distance traveled -/
theorem distance_difference : clara_rate * time - daniel_rate * time = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2811_281199


namespace NUMINAMATH_CALUDE_x_1971_approximation_l2811_281147

/-- A sequence satisfying the given recurrence relation -/
def recurrence_sequence (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → 3 * x n - x (n - 1) = n

theorem x_1971_approximation
  (x : ℕ → ℝ)
  (h_recurrence : recurrence_sequence x)
  (h_x1_bound : |x 1| < 1971) :
  |x 1971 - 985.250000| < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_x_1971_approximation_l2811_281147


namespace NUMINAMATH_CALUDE_reflection_distance_A_l2811_281195

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflection_distance (x y : ℝ) : ℝ :=
  2 * |y|

/-- Theorem: The length of the segment from A(2, 4) to its reflection A' over the x-axis is 8 --/
theorem reflection_distance_A : reflection_distance 2 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_reflection_distance_A_l2811_281195


namespace NUMINAMATH_CALUDE_number_problem_l2811_281192

theorem number_problem : ∃ x : ℝ, (0.2 * x = 0.4 * 140 + 80) ∧ (x = 680) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2811_281192


namespace NUMINAMATH_CALUDE_sum_of_squares_given_means_l2811_281160

theorem sum_of_squares_given_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 2 * Real.sqrt 5 →
  a^2 + b^2 = 216 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_given_means_l2811_281160


namespace NUMINAMATH_CALUDE_clubsuit_not_commutative_l2811_281153

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := |x - y|

-- Define the clubsuit operation
def clubsuit (x y : ℝ) : ℝ := heartsuit x (y + 1)

-- Theorem stating that the equality is false
theorem clubsuit_not_commutative : ¬ (∀ x y : ℝ, clubsuit x y = clubsuit y x) := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_not_commutative_l2811_281153


namespace NUMINAMATH_CALUDE_school_purchase_cost_l2811_281139

theorem school_purchase_cost : 
  let projector_count : ℕ := 8
  let computer_count : ℕ := 32
  let projector_cost : ℕ := 7500
  let computer_cost : ℕ := 3600
  (projector_count * projector_cost + computer_count * computer_cost : ℕ) = 175200 := by
  sorry

end NUMINAMATH_CALUDE_school_purchase_cost_l2811_281139


namespace NUMINAMATH_CALUDE_add_36_15_l2811_281169

theorem add_36_15 : 36 + 15 = 51 := by
  sorry

end NUMINAMATH_CALUDE_add_36_15_l2811_281169


namespace NUMINAMATH_CALUDE_perfume_price_change_l2811_281136

-- Define the original price
def original_price : ℝ := 1200

-- Define the increase percentage
def increase_percent : ℝ := 10

-- Define the decrease percentage
def decrease_percent : ℝ := 15

-- Theorem statement
theorem perfume_price_change :
  let increased_price := original_price * (1 + increase_percent / 100)
  let final_price := increased_price * (1 - decrease_percent / 100)
  original_price - final_price = 78 := by
sorry

end NUMINAMATH_CALUDE_perfume_price_change_l2811_281136


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l2811_281141

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt ((x + 1)^2 + (y + 1)^2) - Real.sqrt (x^2 + y^2) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l2811_281141


namespace NUMINAMATH_CALUDE_final_position_total_consumption_l2811_281133

-- Define the list of mileage values
def mileage : List Int := [-6, -2, 8, -3, 6, -4, 6, 3]

-- Define the electricity consumption rate per kilometer
def consumption_rate : Float := 0.15

-- Theorem for the final position
theorem final_position (m : List Int := mileage) :
  m.sum = 8 := by sorry

-- Theorem for total electricity consumption
theorem total_consumption (m : List Int := mileage) (r : Float := consumption_rate) :
  (m.map Int.natAbs).sum.toFloat * r = 5.7 := by sorry

end NUMINAMATH_CALUDE_final_position_total_consumption_l2811_281133


namespace NUMINAMATH_CALUDE_chocolate_bars_to_sell_l2811_281187

theorem chocolate_bars_to_sell (initial : ℕ) (sold_week1 : ℕ) (sold_week2 : ℕ) 
  (h1 : initial = 18)
  (h2 : sold_week1 = 5)
  (h3 : sold_week2 = 7) :
  initial - (sold_week1 + sold_week2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_to_sell_l2811_281187


namespace NUMINAMATH_CALUDE_percentage_calculation_l2811_281165

theorem percentage_calculation (number : ℝ) (p : ℝ) 
  (h1 : (4/5) * (3/8) * number = 24) 
  (h2 : p * number / 100 = 199.99999999999997) : 
  p = 250 := by sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2811_281165


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2811_281145

theorem no_positive_integer_solutions (k n : ℕ+) (h : n > 2) :
  ¬∃ (x y : ℕ+), x^(n : ℕ) - y^(n : ℕ) = 2^(k : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2811_281145


namespace NUMINAMATH_CALUDE_X_inverse_of_A_l2811_281150

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -1, 0; -3, 5, 0; 0, 0, 2]

def X : Matrix (Fin 3) (Fin 3) ℚ := !![5/7, 1/7, 0; 3/7, 2/7, 0; 0, 0, 1/2]

theorem X_inverse_of_A : X * A = 1 := by sorry

end NUMINAMATH_CALUDE_X_inverse_of_A_l2811_281150


namespace NUMINAMATH_CALUDE_child_growth_l2811_281118

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) :
  current_height - previous_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_growth_l2811_281118


namespace NUMINAMATH_CALUDE_original_ratio_first_term_l2811_281129

theorem original_ratio_first_term 
  (original_first : ℚ) 
  (original_second : ℚ) 
  (added_number : ℚ) 
  (new_ratio_first : ℚ) 
  (new_ratio_second : ℚ) :
  original_first / original_second = 4 / 15 →
  added_number = 29 →
  (original_first + added_number) / (original_second + added_number) = new_ratio_first / new_ratio_second →
  new_ratio_first / new_ratio_second = 3 / 4 →
  original_first = 4 :=
by sorry

end NUMINAMATH_CALUDE_original_ratio_first_term_l2811_281129


namespace NUMINAMATH_CALUDE_binomial_30_3_l2811_281173

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by sorry

end NUMINAMATH_CALUDE_binomial_30_3_l2811_281173


namespace NUMINAMATH_CALUDE_max_distinct_pairs_l2811_281116

/-- Given a set of integers from 1 to 3000, we can choose at most 1199 pairs
    such that each pair sum is distinct and no greater than 3000. -/
theorem max_distinct_pairs : ∀ (k : ℕ) (a b : ℕ → ℕ),
  (∀ i, i < k → 1 ≤ a i ∧ a i < b i ∧ b i ≤ 3000) →
  (∀ i j, i < k → j < k → i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) →
  (∀ i j, i < k → j < k → i ≠ j → a i + b i ≠ a j + b j) →
  (∀ i, i < k → a i + b i ≤ 3000) →
  k ≤ 1199 :=
by sorry

end NUMINAMATH_CALUDE_max_distinct_pairs_l2811_281116


namespace NUMINAMATH_CALUDE_bike_ride_distance_l2811_281143

/-- Calculates the total distance traveled given the conditions of the bike ride --/
theorem bike_ride_distance (total_time : ℝ) (speed_out speed_back : ℝ) : 
  total_time = 7 ∧ speed_out = 24 ∧ speed_back = 18 →
  2 * (total_time / (1 / speed_out + 1 / speed_back)) = 144 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_distance_l2811_281143


namespace NUMINAMATH_CALUDE_real_part_of_z_l2811_281184

theorem real_part_of_z (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : 
  z.re = 3 / 25 := by
sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2811_281184


namespace NUMINAMATH_CALUDE_max_area_inscribed_rectangle_optimal_rectangle_sides_l2811_281151

/-- The maximum area of a rectangle inscribed in a right triangle -/
theorem max_area_inscribed_rectangle (h : Real) (α : Real) (x y : Real) :
  h = 24 →                   -- Hypotenuse is 24 cm
  α = π / 3 →                -- One angle is 60°
  0 < x →                    -- Length of rectangle is positive
  0 < y →                    -- Width of rectangle is positive
  y = h - (4 * x * Real.sqrt 3) / 3 →  -- Relationship between x and y
  x * y ≤ 12 * 3 * Real.sqrt 3 :=
by sorry

/-- The sides of the rectangle that achieve maximum area -/
theorem optimal_rectangle_sides (h : Real) (α : Real) (x y : Real) :
  h = 24 →                   -- Hypotenuse is 24 cm
  α = π / 3 →                -- One angle is 60°
  0 < x →                    -- Length of rectangle is positive
  0 < y →                    -- Width of rectangle is positive
  y = h - (4 * x * Real.sqrt 3) / 3 →  -- Relationship between x and y
  x * y = 12 * 3 * Real.sqrt 3 →       -- Maximum area condition
  x = 3 * Real.sqrt 3 ∧ y = 12 :=
by sorry

end NUMINAMATH_CALUDE_max_area_inscribed_rectangle_optimal_rectangle_sides_l2811_281151


namespace NUMINAMATH_CALUDE_quadratic_radical_simplification_l2811_281100

theorem quadratic_radical_simplification :
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x + y) = Real.sqrt x + Real.sqrt y → x = 0 ∨ y = 0) ∧
  (Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2) ∧
  (Real.sqrt (8 + 4 * Real.sqrt 3) = Real.sqrt 6 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radical_simplification_l2811_281100


namespace NUMINAMATH_CALUDE_probability_more_heads_l2811_281105

/-- 
Given two players A and B, where A flips a fair coin n+1 times and B flips a fair coin n times,
this theorem states that the probability of A having more heads than B is 1/2.
-/
theorem probability_more_heads (n : ℕ) : ℝ := by
  sorry

#check probability_more_heads

end NUMINAMATH_CALUDE_probability_more_heads_l2811_281105


namespace NUMINAMATH_CALUDE_jacoby_work_hours_l2811_281123

/-- The problem of calculating Jacoby's work hours -/
theorem jacoby_work_hours :
  let trip_cost : ℕ := 5000
  let hourly_wage : ℕ := 20
  let cookies_sold : ℕ := 24
  let cookie_price : ℕ := 4
  let lottery_ticket_cost : ℕ := 10
  let lottery_winnings : ℕ := 500
  let sister_gift : ℕ := 500
  let remaining_needed : ℕ := 3214

  let cookie_earnings := cookies_sold * cookie_price
  let gifts := sister_gift * 2
  let other_income := cookie_earnings + lottery_winnings + gifts - lottery_ticket_cost
  let total_earned := trip_cost - remaining_needed
  let job_earnings := total_earned - other_income
  let hours_worked := job_earnings / hourly_wage

  hours_worked = 10 := by sorry

end NUMINAMATH_CALUDE_jacoby_work_hours_l2811_281123


namespace NUMINAMATH_CALUDE_complex_distance_problem_l2811_281131

theorem complex_distance_problem (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^3 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^6 - 1) = 5 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 3 ∨ α = -Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_problem_l2811_281131


namespace NUMINAMATH_CALUDE_standard_notation_expression_l2811_281142

/-- A predicate to check if an expression conforms to standard algebraic notation -/
def is_standard_notation : String → Prop := sorry

/-- The set of given expressions -/
def expressions : Set String :=
  {"18 * b", "1 1/4 x", "-b / a^2", "m ÷ 2n"}

/-- Theorem stating that "-b / a^2" conforms to standard algebraic notation -/
theorem standard_notation_expression :
  ∃ e ∈ expressions, is_standard_notation e ∧ e = "-b / a^2" := by sorry

end NUMINAMATH_CALUDE_standard_notation_expression_l2811_281142


namespace NUMINAMATH_CALUDE_sarah_flour_amount_l2811_281154

/-- The amount of rye flour Sarah bought in pounds -/
def rye_flour : ℕ := 5

/-- The amount of whole-wheat bread flour Sarah bought in pounds -/
def wheat_bread_flour : ℕ := 10

/-- The amount of chickpea flour Sarah bought in pounds -/
def chickpea_flour : ℕ := 3

/-- The amount of whole-wheat pastry flour Sarah already had at home in pounds -/
def pastry_flour : ℕ := 2

/-- The total amount of flour Sarah now has in pounds -/
def total_flour : ℕ := rye_flour + wheat_bread_flour + chickpea_flour + pastry_flour

theorem sarah_flour_amount : total_flour = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_flour_amount_l2811_281154


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2811_281120

theorem child_ticket_cost (adult_price : ℕ) (total_sales : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) :
  adult_price = 5 →
  total_sales = 178 →
  total_tickets = 42 →
  child_tickets = 16 →
  ∃ (child_price : ℕ), child_price = 3 ∧
    total_sales = adult_price * (total_tickets - child_tickets) + child_price * child_tickets :=
by
  sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2811_281120


namespace NUMINAMATH_CALUDE_awards_distribution_l2811_281103

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution :
  distribute_awards 6 4 = 1560 :=
sorry

end NUMINAMATH_CALUDE_awards_distribution_l2811_281103


namespace NUMINAMATH_CALUDE_difference_largest_smallest_valid_numbers_l2811_281135

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  (n / 100 ≠ (n / 10) % 10) ∧
  (n / 100 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10) ∧
  (n / 100 - (n / 10) % 10 = (n / 10) % 10 - n % 10)

def largest_valid_number : ℕ := 951

def smallest_valid_number : ℕ := 159

theorem difference_largest_smallest_valid_numbers :
  largest_valid_number - smallest_valid_number = 792 ∧
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  ∀ n : ℕ, is_valid_number n → 
    smallest_valid_number ≤ n ∧ n ≤ largest_valid_number := by
  sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_valid_numbers_l2811_281135


namespace NUMINAMATH_CALUDE_total_amount_divided_l2811_281194

/-- The total amount divided among A, B, and C is 3366.00000000000006 given the conditions. -/
theorem total_amount_divided (a b c : ℝ) 
  (h1 : a = (2/3) * b)
  (h2 : b = (1/4) * c)
  (h3 : a = 396.00000000000006) : 
  a + b + c = 3366.00000000000006 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_divided_l2811_281194


namespace NUMINAMATH_CALUDE_at_least_two_equal_l2811_281101

theorem at_least_two_equal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : a^2 + b + c = 1/a)
  (eq2 : b^2 + c + a = 1/b)
  (eq3 : c^2 + a + b = 1/c) :
  ¬(a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l2811_281101


namespace NUMINAMATH_CALUDE_only_fourteen_satisfies_l2811_281196

-- Define a two-digit number
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define the operation of increasing digits
def increase_digits (n : ℕ) : Set ℕ :=
  { m : ℕ | ∃ (a b : ℕ), n = 10 * a + b ∧ 
    m = 10 * (a + 2) + (b + 2) ∨ 
    m = 10 * (a + 2) + (b + 4) ∨ 
    m = 10 * (a + 4) + (b + 2) ∨ 
    m = 10 * (a + 4) + (b + 4) }

-- The main theorem
theorem only_fourteen_satisfies : 
  ∃! (n : ℕ), is_two_digit n ∧ (4 * n) ∈ increase_digits n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_only_fourteen_satisfies_l2811_281196


namespace NUMINAMATH_CALUDE_absolute_difference_simplification_l2811_281163

theorem absolute_difference_simplification (a b : ℝ) 
  (ha : a < 0) (hab : a * b < 0) : 
  |a - b - 3| - |4 + b - a| = -1 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_simplification_l2811_281163


namespace NUMINAMATH_CALUDE_equation_solution_l2811_281156

theorem equation_solution : ∃ x : ℝ, (x - 5) ^ 4 = (1 / 16)⁻¹ ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2811_281156


namespace NUMINAMATH_CALUDE_triangle_division_regions_l2811_281159

/-- Given a triangle ABC and a positive integer n, with the sides divided into 2^n equal parts
    and cevians drawn as described, the number of regions into which the triangle is divided
    is equal to 3 · 2^(2n) - 6 · 2^n + 6. -/
theorem triangle_division_regions (n : ℕ+) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_triangle_division_regions_l2811_281159


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l2811_281128

theorem min_distance_to_origin (x y : ℝ) : 
  8 * x + 15 * y = 120 → x ≥ 0 → y ≥ 0 → 
  ∀ x' y' : ℝ, 8 * x' + 15 * y' = 120 → x' ≥ 0 → y' ≥ 0 → 
  Real.sqrt (x^2 + y^2) ≤ Real.sqrt (x'^2 + y'^2) → 
  Real.sqrt (x^2 + y^2) = 120 / 17 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l2811_281128


namespace NUMINAMATH_CALUDE_isosceles_triangle_unique_point_l2811_281106

-- Define the triangle and point
def Triangle (A B C P : ℝ × ℝ) : Prop :=
  ∃ (s t : ℝ),
    -- Triangle ABC is isosceles with AB = AC = s
    dist A B = s ∧ dist A C = s ∧
    -- BC = t
    dist B C = t ∧
    -- Point P is inside the triangle (simplified assumption)
    true ∧
    -- AP = 2
    dist A P = 2 ∧
    -- BP = √5
    dist B P = Real.sqrt 5 ∧
    -- CP = 3
    dist C P = 3

-- The theorem to prove
theorem isosceles_triangle_unique_point 
  (A B C P : ℝ × ℝ) 
  (h : Triangle A B C P) : 
  ∃ (s t : ℝ), s = 2 * Real.sqrt 3 ∧ t = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_unique_point_l2811_281106


namespace NUMINAMATH_CALUDE_r_six_times_thirty_l2811_281168

/-- The function r as defined in the problem -/
def r (θ : ℚ) : ℚ := 1 / (2 - θ)

/-- The composition of r with itself n times -/
def r_n (n : ℕ) (θ : ℚ) : ℚ :=
  match n with
  | 0 => θ
  | n + 1 => r (r_n n θ)

/-- The main theorem stating that applying r six times to 30 results in 22/23 -/
theorem r_six_times_thirty : r_n 6 30 = 22 / 23 := by
  sorry

end NUMINAMATH_CALUDE_r_six_times_thirty_l2811_281168


namespace NUMINAMATH_CALUDE_sum_of_sqrt_products_gt_sum_of_numbers_l2811_281186

theorem sum_of_sqrt_products_gt_sum_of_numbers 
  (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : |x - y| < 2) (hyz : |y - z| < 2) (hzx : |z - x| < 2) : 
  Real.sqrt (x * y + 1) + Real.sqrt (y * z + 1) + Real.sqrt (z * x + 1) > x + y + z := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_products_gt_sum_of_numbers_l2811_281186


namespace NUMINAMATH_CALUDE_ellipse_axis_ratio_l2811_281183

/-- Given an ellipse with equation x²/9 + y²/m² = 1 where 0 < m < 3,
    if the length of its major axis is twice that of its minor axis,
    then m = 3/2 -/
theorem ellipse_axis_ratio (m : ℝ) 
  (h1 : 0 < m) (h2 : m < 3) 
  (h3 : ∀ x y : ℝ, x^2/9 + y^2/m^2 = 1 → 6 = 2*(2*m)) : 
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_axis_ratio_l2811_281183


namespace NUMINAMATH_CALUDE_trig_values_for_point_l2811_281189

/-- Given a point P(-√3, m) on the terminal side of angle α, where m ≠ 0 and sin α = (√2 * m) / 4,
    prove the values of m, cos α, and tan α. -/
theorem trig_values_for_point (m : ℝ) (α : ℝ) (h1 : m ≠ 0) (h2 : Real.sin α = (Real.sqrt 2 * m) / 4) :
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  (m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
  (m < 0 → Real.tan α = Real.sqrt 15 / 3) := by
  sorry

end NUMINAMATH_CALUDE_trig_values_for_point_l2811_281189


namespace NUMINAMATH_CALUDE_no_rain_probability_l2811_281117

theorem no_rain_probability (pMonday pTuesday pBoth : ℝ) 
  (hMonday : pMonday = 0.6)
  (hTuesday : pTuesday = 0.55)
  (hBoth : pBoth = 0.4) :
  1 - (pMonday + pTuesday - pBoth) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2811_281117


namespace NUMINAMATH_CALUDE_x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y_l2811_281102

theorem x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y :
  ¬(∀ (x y : ℝ), x ≤ y → |x| ≤ |y|) ∧ ¬(∀ (x y : ℝ), |x| ≤ |y| → x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_x_leq_y_neither_necessary_nor_sufficient_for_abs_x_leq_abs_y_l2811_281102


namespace NUMINAMATH_CALUDE_min_value_of_f_min_value_is_zero_l2811_281127

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 3

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem min_value_of_f :
  ∀ x : ℝ, domain x → ∀ y : ℝ, domain y → f y ≥ f 3 := by
  sorry

-- The minimum value is f(3) = 0
theorem min_value_is_zero : f 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_min_value_is_zero_l2811_281127


namespace NUMINAMATH_CALUDE_min_value_of_function_l2811_281138

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  ∃ (y : ℝ), y = x + 4 / (x - 2) ∧ (∀ (z : ℝ), z = x + 4 / (x - 2) → y ≤ z) ∧ y = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2811_281138


namespace NUMINAMATH_CALUDE_soccer_team_theorem_l2811_281121

def soccer_team_problem (total_players starting_players first_half_subs : ℕ) : ℕ :=
  let second_half_subs := first_half_subs + (first_half_subs + 1) / 2
  let total_played := starting_players + first_half_subs + second_half_subs
  total_players - total_played

theorem soccer_team_theorem :
  soccer_team_problem 36 11 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_theorem_l2811_281121


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l2811_281124

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x > 0 ∧ x = 89/9 ∧
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 8 → x ≤ y) ∧
  ⌊x^2⌋ - x * ⌊x⌋ = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l2811_281124


namespace NUMINAMATH_CALUDE_negation_equivalence_l2811_281125

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2811_281125


namespace NUMINAMATH_CALUDE_top_quality_soccer_balls_l2811_281174

/-- Given a batch of soccer balls, calculate the number of top-quality balls -/
theorem top_quality_soccer_balls 
  (total : ℕ) 
  (frequency : ℝ) 
  (h_total : total = 10000)
  (h_frequency : frequency = 0.975) :
  ⌊(total : ℝ) * frequency⌋ = 9750 := by
  sorry

end NUMINAMATH_CALUDE_top_quality_soccer_balls_l2811_281174


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2811_281130

/-- Given that the solution set of ax^2 - bx + 2 < 0 is {x | 1 < x < 2}, prove that a + b = -2 -/
theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + 2 < 0 ↔ 1 < x ∧ x < 2) → 
  a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2811_281130


namespace NUMINAMATH_CALUDE_russian_players_pairing_probability_l2811_281110

/-- The probability of all Russian players pairing only with other Russian players in a random pairing -/
theorem russian_players_pairing_probability 
  (total_players : ℕ) 
  (russian_players : ℕ) 
  (h1 : total_players = 10) 
  (h2 : russian_players = 4) 
  (h3 : russian_players ≤ total_players) :
  (russian_players.choose 2 : ℚ) / total_players.choose 2 = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_russian_players_pairing_probability_l2811_281110


namespace NUMINAMATH_CALUDE_right_triangle_area_l2811_281149

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 26) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 120 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2811_281149


namespace NUMINAMATH_CALUDE_abc_sum_theorem_l2811_281109

theorem abc_sum_theorem (a b c : ℚ) (h : a * b * c > 0) :
  (|a| / a + |b| / b + |c| / c : ℚ) = 3 ∨ (|a| / a + |b| / b + |c| / c : ℚ) = -1 :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_theorem_l2811_281109


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l2811_281155

def isGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

theorem geometric_sequence_solution (a : ℝ) (h : a > 0) 
  (h_seq : isGeometricSequence 280 a (180/49)) : 
  a = Real.sqrt (50400/49) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l2811_281155


namespace NUMINAMATH_CALUDE_bus_journey_stoppage_time_l2811_281113

/-- Calculates the total stoppage time for a bus journey with three stops -/
def total_stoppage_time (stop1 stop2 stop3 : ℕ) : ℕ :=
  stop1 + stop2 + stop3

/-- Theorem stating that the total stoppage time for the given stop durations is 23 minutes -/
theorem bus_journey_stoppage_time :
  total_stoppage_time 5 8 10 = 23 :=
by sorry

end NUMINAMATH_CALUDE_bus_journey_stoppage_time_l2811_281113


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2811_281164

/-- Given two hyperbolas l and C, prove that the eccentricity of C is 3 -/
theorem hyperbola_eccentricity (k a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), k * x + y - Real.sqrt 2 * k = 0) →  -- Hyperbola l
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola C
  (abs k = b / a) →  -- Parallel asymptotes condition
  (Real.sqrt 2 * k / Real.sqrt (1 + k^2) = 4 / 3) →  -- Distance between asymptotes
  Real.sqrt (1 + b^2 / a^2) = 3 :=  -- Eccentricity of C
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2811_281164


namespace NUMINAMATH_CALUDE_expression_simplification_l2811_281162

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x^2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2811_281162


namespace NUMINAMATH_CALUDE_divisibility_by_2016_l2811_281158

theorem divisibility_by_2016 (n : ℕ) : 
  2016 ∣ ((n^2 + n)^2 - (n^2 - n)^2) * (n^6 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_2016_l2811_281158


namespace NUMINAMATH_CALUDE_diana_erasers_l2811_281190

/-- Given that Diana shares her erasers among 48 friends and each friend gets 80 erasers,
    prove that Diana has 3840 erasers. -/
theorem diana_erasers : ℕ → ℕ → ℕ → Prop :=
  fun num_friends erasers_per_friend total_erasers =>
    (num_friends = 48) →
    (erasers_per_friend = 80) →
    (total_erasers = num_friends * erasers_per_friend) →
    total_erasers = 3840

/-- Proof of the theorem -/
lemma diana_erasers_proof : diana_erasers 48 80 3840 := by
  sorry

end NUMINAMATH_CALUDE_diana_erasers_l2811_281190


namespace NUMINAMATH_CALUDE_expression_value_l2811_281134

theorem expression_value (x : ℝ) (h : x^2 - 3*x = 12) : 3*x^2 - 9*x + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2811_281134


namespace NUMINAMATH_CALUDE_factors_of_N_squared_not_dividing_N_l2811_281179

theorem factors_of_N_squared_not_dividing_N : ∃ (S : Finset ℕ), 
  (∀ d ∈ S, d ∣ (2019^2 - 1)^2 ∧ ¬(d ∣ (2019^2 - 1))) ∧ 
  (∀ d : ℕ, d ∣ (2019^2 - 1)^2 ∧ ¬(d ∣ (2019^2 - 1)) → d ∈ S) ∧ 
  S.card = 157 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_N_squared_not_dividing_N_l2811_281179


namespace NUMINAMATH_CALUDE_expression_evaluation_l2811_281188

theorem expression_evaluation (x y : ℚ) (hx : x = 5) (hy : y = 6) :
  (2 / y) / (2 / x) * 3 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2811_281188


namespace NUMINAMATH_CALUDE_joe_list_count_l2811_281171

/-- The number of balls in the bin -/
def n : ℕ := 15

/-- The number of times Joe draws a ball -/
def draws : ℕ := 4

/-- The number of numbers Joe selects for the final list -/
def selected : ℕ := 3

/-- The number of different possible lists Joe can create -/
def num_lists : ℕ := n^draws * (draws.choose selected)

theorem joe_list_count :
  num_lists = 202500 := by
  sorry

end NUMINAMATH_CALUDE_joe_list_count_l2811_281171


namespace NUMINAMATH_CALUDE_flat_transaction_l2811_281146

theorem flat_transaction (x y : ℝ) : 
  0.14 * x - 0.14 * y = 1.96 ↔ 
  ∃ (gain loss : ℝ), 
    gain = 0.14 * x ∧ 
    loss = 0.14 * y ∧ 
    gain - loss = 1.96 :=
sorry

end NUMINAMATH_CALUDE_flat_transaction_l2811_281146


namespace NUMINAMATH_CALUDE_gardening_project_total_cost_l2811_281107

/-- The cost of the gardening project -/
def gardening_project_cost (
  num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℕ)
  (gardener_hourly_rate : ℕ)
  (gardener_hours_per_day : ℕ)
  (gardener_days : ℕ)
  (soil_volume : ℕ)
  (soil_cost_per_unit : ℕ) : ℕ :=
  num_rose_bushes * cost_per_rose_bush +
  gardener_hourly_rate * gardener_hours_per_day * gardener_days +
  soil_volume * soil_cost_per_unit

/-- The theorem stating the total cost of the gardening project -/
theorem gardening_project_total_cost :
  gardening_project_cost 20 150 30 5 4 100 5 = 4100 := by
  sorry

end NUMINAMATH_CALUDE_gardening_project_total_cost_l2811_281107


namespace NUMINAMATH_CALUDE_total_surveyed_is_185_l2811_281182

/-- Represents the total number of students surveyed in a stratified sampling method -/
def total_surveyed (grade10_total : ℕ) (grade11_total : ℕ) (grade12_total : ℕ) (grade12_surveyed : ℕ) : ℕ :=
  let grade10_surveyed := (grade10_total * grade12_surveyed) / grade12_total
  let grade11_surveyed := (grade11_total * grade12_surveyed) / grade12_total
  grade10_surveyed + grade11_surveyed + grade12_surveyed

/-- Theorem stating that the total number of students surveyed is 185 given the problem conditions -/
theorem total_surveyed_is_185 :
  total_surveyed 1000 1200 1500 75 = 185 := by
  sorry

end NUMINAMATH_CALUDE_total_surveyed_is_185_l2811_281182


namespace NUMINAMATH_CALUDE_fishing_result_l2811_281140

/-- The total number of fishes Will and Henry have after fishing -/
def total_fishes (will_catfish : ℕ) (will_eels : ℕ) (henry_trout_ratio : ℕ) : ℕ :=
  let will_total := will_catfish + will_eels
  let henry_total := will_catfish * henry_trout_ratio
  let henry_kept := henry_total / 2
  will_total + henry_kept

/-- Theorem stating the total number of fishes Will and Henry have -/
theorem fishing_result : total_fishes 16 10 3 = 50 := by
  sorry

#eval total_fishes 16 10 3

end NUMINAMATH_CALUDE_fishing_result_l2811_281140


namespace NUMINAMATH_CALUDE_crayons_lost_theorem_l2811_281175

/-- The number of crayons lost or given away -/
def crayons_lost_or_given_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that the number of crayons lost or given away is correct -/
theorem crayons_lost_theorem (initial : ℕ) (remaining : ℕ) 
  (h : initial ≥ remaining) : 
  crayons_lost_or_given_away initial remaining = initial - remaining :=
by
  sorry

#eval crayons_lost_or_given_away 479 134

end NUMINAMATH_CALUDE_crayons_lost_theorem_l2811_281175


namespace NUMINAMATH_CALUDE_xyz_sum_of_squares_l2811_281166

theorem xyz_sum_of_squares (x y z : ℝ) 
  (h1 : (2*x + 2*y + 3*z) / 7 = 9)
  (h2 : (x^2 * y^2 * z^3)^(1/7) = 6)
  (h3 : 7 / ((2/x) + (2/y) + (3/z)) = 4) :
  x^2 + y^2 + z^2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_of_squares_l2811_281166


namespace NUMINAMATH_CALUDE_exists_a_greater_than_bound_l2811_281198

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 1/3
  | (n+2) => (2 * a (n+1)) / 3 - a n

theorem exists_a_greater_than_bound : ∃ n : ℕ, a n > 999/1000 := by
  sorry

end NUMINAMATH_CALUDE_exists_a_greater_than_bound_l2811_281198


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2811_281181

theorem prime_pairs_dividing_sum_of_powers (p q : ℕ) : 
  Prime p → Prime q → (p * q) ∣ (2^p + 2^q) → 
  ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2811_281181


namespace NUMINAMATH_CALUDE_homework_theorem_l2811_281161

/-- The number of possible homework situations for a given number of teachers and students -/
def homework_situations (num_teachers : ℕ) (num_students : ℕ) : ℕ :=
  num_teachers ^ num_students

/-- Theorem: With 3 teachers and 4 students, there are 3^4 possible homework situations -/
theorem homework_theorem :
  homework_situations 3 4 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_homework_theorem_l2811_281161


namespace NUMINAMATH_CALUDE_same_color_combination_probability_l2811_281191

def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8

def lucy_picks : ℕ := 2
def john_picks : ℕ := 2

theorem same_color_combination_probability :
  let probability_same_combination := (2 * (Nat.choose red_candies 2 * Nat.choose (red_candies - 2) 2 +
                                            Nat.choose blue_candies 2 * Nat.choose (blue_candies - 2) 2) +
                                       Nat.choose red_candies 2 * Nat.choose blue_candies 2 +
                                       Nat.choose blue_candies 2 * Nat.choose red_candies 2) /
                                      (Nat.choose total_candies 2 * Nat.choose (total_candies - 2) 2)
  probability_same_combination = 184 / 323 := by
  sorry

end NUMINAMATH_CALUDE_same_color_combination_probability_l2811_281191


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2811_281172

/-- Given a line parallel to 3x - 6y = 12, its slope is 1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℚ) (b : ℚ), (∃ (k : ℚ), 3 * x - 6 * (m * x + b) = k) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2811_281172


namespace NUMINAMATH_CALUDE_slope_is_plus_minus_two_l2811_281144

/-- The slope of a line passing through (-1,0) that intersects the parabola y^2 = 4x
    such that the midpoint of the intersection points lies on x = 3 -/
def slope_of_intersecting_line : ℝ → Prop :=
  λ k : ℝ => ∃ (x₁ x₂ y₁ y₂ : ℝ),
    -- Line equation
    y₁ = k * (x₁ + 1) ∧
    y₂ = k * (x₂ + 1) ∧
    -- Parabola equation
    y₁^2 = 4 * x₁ ∧
    y₂^2 = 4 * x₂ ∧
    -- Midpoint condition
    (x₁ + x₂) / 2 = 3

theorem slope_is_plus_minus_two :
  ∀ k : ℝ, slope_of_intersecting_line k ↔ k = 2 ∨ k = -2 :=
sorry

end NUMINAMATH_CALUDE_slope_is_plus_minus_two_l2811_281144


namespace NUMINAMATH_CALUDE_smallest_c_value_l2811_281114

-- Define the polynomial
def polynomial (c d x : ℤ) : ℤ := x^3 - c*x^2 + d*x - 2730

-- Define the property that the polynomial has three positive integer roots
def has_three_positive_integer_roots (c d : ℤ) : Prop :=
  ∃ (r₁ r₂ r₃ : ℤ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
    ∀ x, polynomial c d x = (x - r₁) * (x - r₂) * (x - r₃)

-- Theorem statement
theorem smallest_c_value (c d : ℤ) :
  has_three_positive_integer_roots c d → c ≥ 54 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2811_281114


namespace NUMINAMATH_CALUDE_calculate_expression_l2811_281104

theorem calculate_expression : (-2)^2 - (1/8 - 3/4 + 1/2) * (-24) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2811_281104


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2811_281167

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 1) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2811_281167
