import Mathlib

namespace NUMINAMATH_CALUDE_halloween_candy_count_l3761_376176

/-- Calculates the final candy count given initial count, eaten count, and received count. -/
def finalCandyCount (initial eaten received : ℕ) : ℕ :=
  initial - eaten + received

/-- Theorem stating that given the specific values from the problem, 
    the final candy count is 62. -/
theorem halloween_candy_count : 
  finalCandyCount 47 25 40 = 62 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_count_l3761_376176


namespace NUMINAMATH_CALUDE_vector_magnitude_l3761_376116

theorem vector_magnitude (a b : ℝ × ℝ) : 
  let angle := π / 6
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  dot_product = 3 ∧ 
  magnitude_a = 3 →
  Real.sqrt (b.1^2 + b.2^2) = (2 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3761_376116


namespace NUMINAMATH_CALUDE_probability_three_defective_shipment_l3761_376163

/-- The probability of selecting three defective smartphones from a shipment --/
def probability_three_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total *
  ((defective - 1) : ℚ) / (total - 1) *
  ((defective - 2) : ℚ) / (total - 2)

/-- Theorem stating the approximate probability of selecting three defective smartphones --/
theorem probability_three_defective_shipment :
  let total := 500
  let defective := 85
  abs (probability_three_defective total defective - 0.0047) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_defective_shipment_l3761_376163


namespace NUMINAMATH_CALUDE_range_of_a_l3761_376101

-- Define the function f
def f (x : ℝ) : ℝ := |3*x + 2|

-- State the theorem
theorem range_of_a (m n : ℝ) (h_mn : m + n = 1) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (∀ x : ℝ, |x - a| - f x ≤ 1/m + 1/n) → 0 < a ∧ a ≤ 10/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3761_376101


namespace NUMINAMATH_CALUDE_solution_set_is_closed_interval_l3761_376194

def system_solution (x : ℝ) : Prop :=
  -2 * (x - 3) > 10 ∧ x^2 + 7*x + 12 ≤ 0

theorem solution_set_is_closed_interval :
  {x : ℝ | system_solution x} = {x : ℝ | -4 ≤ x ∧ x ≤ -3} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_is_closed_interval_l3761_376194


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3761_376172

theorem polynomial_factorization (x : ℝ) :
  (∃ (a b c d : ℝ), x^2 - 1 = (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x + 1 ≠ (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x^2 + x + 1 ≠ (a*x + b) * (c*x + d)) ∧
  (∀ (a b c d : ℝ), x^2 + 4 ≠ (a*x + b) * (c*x + d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3761_376172


namespace NUMINAMATH_CALUDE_drought_periods_correct_max_water_storage_volume_l3761_376174

noncomputable def v (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 9 then
    (1 / 240) * (-(t^2) + 15*t - 51) * Real.exp t + 50
  else if 9 < t ∧ t ≤ 12 then
    4 * (t - 9) * (3*t - 41) + 50
  else
    0

def isDroughtPeriod (t : ℝ) : Prop := v t < 50

def monthToPeriod (m : ℕ) : Set ℝ := {t | m - 1 < t ∧ t ≤ m}

def droughtMonths : Set ℕ := {1, 2, 3, 4, 5, 10, 11, 12}

theorem drought_periods_correct (m : ℕ) (hm : m ∈ droughtMonths) :
  ∀ t ∈ monthToPeriod m, isDroughtPeriod t :=
sorry

theorem max_water_storage_volume :
  ∃ t ∈ Set.Icc (0 : ℝ) 12, v t = 150 ∧ ∀ s ∈ Set.Icc (0 : ℝ) 12, v s ≤ v t :=
sorry

axiom e_cubed_eq_20 : Real.exp 3 = 20

end NUMINAMATH_CALUDE_drought_periods_correct_max_water_storage_volume_l3761_376174


namespace NUMINAMATH_CALUDE_paint_calculation_l3761_376185

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCoverage where
  totalRooms : ℕ
  cansUsed : ℕ

/-- Calculates the number of cans used for a given number of rooms -/
def cansForRooms (initialCoverage finalCoverage : PaintCoverage) (roomsToPaint : ℕ) : ℕ :=
  let roomsPerCan := (initialCoverage.totalRooms - finalCoverage.totalRooms) / 
                     (initialCoverage.cansUsed - finalCoverage.cansUsed)
  roomsToPaint / roomsPerCan

theorem paint_calculation (initialCoverage finalCoverage : PaintCoverage) 
  (h1 : initialCoverage.totalRooms = 45)
  (h2 : finalCoverage.totalRooms = 36)
  (h3 : initialCoverage.cansUsed - finalCoverage.cansUsed = 4) :
  cansForRooms initialCoverage finalCoverage 36 = 16 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3761_376185


namespace NUMINAMATH_CALUDE_C₂_fixed_point_l3761_376115

/-- Parabola C₁ with vertex (√2-1, 1) and focus (√2-3/4, 1) -/
def C₁ : Set (ℝ × ℝ) :=
  {p | (p.2 - 1)^2 = 2 * (p.1 - (Real.sqrt 2 - 1))}

/-- Parabola C₂ with equation y² - ay + x + 2b = 0 -/
def C₂ (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2^2 - a * p.2 + p.1 + 2 * b = 0}

/-- The tangent line to C₁ at point p -/
def tangentC₁ (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | 2 * p.2 * q.2 - q.1 - 2 * (p.2 + 1) = 0}

/-- The tangent line to C₂ at point p -/
def tangentC₂ (a : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | (2 * p.2 - a) * q.2 + q.1 - a * p.2 + p.1 + 4 * ((a - 2) * p.2 - p.1 - Real.sqrt 2) / 4 = 0}

/-- Perpendicularity condition for tangent lines -/
def perpendicularTangents (p : ℝ × ℝ) (a : ℝ) : Prop :=
  (p.2 - 1) * (2 * p.2 - a) = -1

theorem C₂_fixed_point (a b : ℝ) :
  (∃ p, p ∈ C₁ ∧ p ∈ C₂ a b ∧ perpendicularTangents p a) →
  (Real.sqrt 2 - 1/2, 1) ∈ C₂ a b := by
  sorry

end NUMINAMATH_CALUDE_C₂_fixed_point_l3761_376115


namespace NUMINAMATH_CALUDE_circle_C_area_l3761_376140

-- Define the circle C
def circle_C (x y r : ℝ) : Prop := (x + 2)^2 + y^2 = r^2

-- Define the parabola D
def parabola_D (x y : ℝ) : Prop := y^2 = 20 * x

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := -5

-- Define the distance between the center of C and the axis of symmetry
def distance_to_axis : ℝ := 3

-- Define the length of AB
def length_AB : ℝ := 8

-- Theorem to prove
theorem circle_C_area :
  ∃ (r : ℝ), 
    (∀ x y, circle_C x y r → parabola_D x y → x = axis_of_symmetry) ∧
    length_AB = 8 ∧
    distance_to_axis = 3 →
    π * r^2 = 25 * π :=
sorry

end NUMINAMATH_CALUDE_circle_C_area_l3761_376140


namespace NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3761_376122

theorem similar_triangles_leg_sum 
  (A₁ A₂ : ℝ) 
  (h_areas : A₁ = 12 ∧ A₂ = 192) 
  (h_similar : ∃ (k : ℝ), k > 0 ∧ A₂ = k^2 * A₁) 
  (a b : ℝ) 
  (h_right : a^2 + b^2 = 10^2) 
  (h_leg_ratio : a = 2*b) 
  (h_area_small : A₁ = 1/2 * a * b) : 
  ∃ (c d : ℝ), c^2 + d^2 = (4*10)^2 ∧ A₂ = 1/2 * c * d ∧ c + d = 24 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_similar_triangles_leg_sum_l3761_376122


namespace NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l3761_376139

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem square_of_cube_of_third_smallest_prime :
  (nthSmallestPrime 3) ^ 3 ^ 2 = 15625 := by sorry

end NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l3761_376139


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3761_376144

theorem expression_simplification_and_evaluation :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := Real.sqrt 3 - 1
  ((3 * a) / (2 * a - b) - 1) / ((a + b) / (4 * a^2 - b^2)) = 3 * Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3761_376144


namespace NUMINAMATH_CALUDE_smallest_five_digit_cube_sum_l3761_376184

theorem smallest_five_digit_cube_sum (x : ℕ) : 
  x ≥ 10000 ∧ x < 100000 ∧ 
  (∃ k : ℕ, (343 * x) / 90 = k^3) ∧
  (∀ y : ℕ, y ≥ 10000 ∧ y < x → ¬(∃ k : ℕ, (343 * y) / 90 = k^3)) →
  x = 11250 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_cube_sum_l3761_376184


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l3761_376162

/-- The maximum area of a right-angled triangle with perimeter 2 is 3 - 2√2 -/
theorem max_area_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_perimeter : a + b + c = 2) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  (1/2) * a * b ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l3761_376162


namespace NUMINAMATH_CALUDE_increase_both_averages_l3761_376193

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem increase_both_averages :
  ∃ g ∈ group1,
    average (group1.filter (· ≠ g)) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_increase_both_averages_l3761_376193


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_2015029_l3761_376104

/-- The area of a quadrilateral with vertices at (2, 4), (2, 2), (3, 2), and (2010, 2011) -/
def quadrilateralArea : ℝ := 2015029

/-- The vertices of the quadrilateral -/
def vertices : List (ℝ × ℝ) := [(2, 4), (2, 2), (3, 2), (2010, 2011)]

/-- Theorem stating that the area of the quadrilateral with the given vertices is 2015029 square units -/
theorem quadrilateral_area_is_2015029 :
  let computeArea : List (ℝ × ℝ) → ℝ := sorry -- Function to compute area from vertices
  computeArea vertices = quadrilateralArea := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_2015029_l3761_376104


namespace NUMINAMATH_CALUDE_max_m_value_l3761_376148

theorem max_m_value (m : ℝ) : 
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  A ⊆ B → m ≤ 11 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l3761_376148


namespace NUMINAMATH_CALUDE_f_is_quadratic_l3761_376182

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 + 15x - 7 = 0 -/
def f (x : ℝ) : ℝ := x^2 + 15*x - 7

/-- Theorem: f is a quadratic equation in one variable -/
theorem f_is_quadratic : is_quadratic_equation_in_one_variable f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l3761_376182


namespace NUMINAMATH_CALUDE_travel_time_difference_l3761_376153

/-- Given a set of 5 numbers (x, y, 10, 11, 9) with an average of 10 and a variance of 2, |x-y| = 4 -/
theorem travel_time_difference (x y : ℝ) : 
  (x + y + 10 + 11 + 9) / 5 = 10 ∧ 
  ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2 →
  |x - y| = 4 := by
sorry


end NUMINAMATH_CALUDE_travel_time_difference_l3761_376153


namespace NUMINAMATH_CALUDE_total_fruits_l3761_376147

theorem total_fruits (total_baskets : ℕ) 
                     (apple_baskets orange_baskets : ℕ) 
                     (apples_per_basket oranges_per_basket pears_per_basket : ℕ) : 
  total_baskets = 127 →
  apple_baskets = 79 →
  orange_baskets = 30 →
  apples_per_basket = 75 →
  oranges_per_basket = 143 →
  pears_per_basket = 56 →
  (apple_baskets * apples_per_basket + 
   orange_baskets * oranges_per_basket + 
   (total_baskets - apple_baskets - orange_baskets) * pears_per_basket) = 11223 :=
by
  sorry

#check total_fruits

end NUMINAMATH_CALUDE_total_fruits_l3761_376147


namespace NUMINAMATH_CALUDE_roots_sum_product_l3761_376102

theorem roots_sum_product (x₁ x₂ : ℝ) : 
  (x₁ - 2)^2 = 3*(x₁ + 5) ∧ 
  (x₂ - 2)^2 = 3*(x₂ + 5) → 
  x₁*x₂ + x₁^2 + x₂^2 = 60 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_product_l3761_376102


namespace NUMINAMATH_CALUDE_ammonium_chloride_formed_l3761_376135

-- Define the reaction components
variable (NH3 : ℝ) -- Moles of Ammonia
variable (HCl : ℝ) -- Moles of Hydrochloric acid
variable (NH4Cl : ℝ) -- Moles of Ammonium chloride

-- Define the conditions
axiom ammonia_moles : NH3 = 3
axiom total_product : NH4Cl = 3

-- Theorem to prove
theorem ammonium_chloride_formed : NH4Cl = 3 :=
by sorry

end NUMINAMATH_CALUDE_ammonium_chloride_formed_l3761_376135


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3761_376155

theorem polynomial_division_remainder 
  (x : ℝ) : 
  ∃ (q : ℝ → ℝ), 
  x^4 - 8*x^3 + 18*x^2 - 27*x + 15 = 
  (x^2 - 3*x + 14/3) * q x + (2*x + 205/9) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3761_376155


namespace NUMINAMATH_CALUDE_road_repair_theorem_l3761_376179

/-- The number of persons in the first group -/
def first_group : ℕ := 42

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 14

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem road_repair_theorem : second_group = 30 := by
  sorry

end NUMINAMATH_CALUDE_road_repair_theorem_l3761_376179


namespace NUMINAMATH_CALUDE_two_thousand_fourteen_between_powers_of_ten_l3761_376196

theorem two_thousand_fourteen_between_powers_of_ten : 10^3 < 2014 ∧ 2014 < 10^4 := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_fourteen_between_powers_of_ten_l3761_376196


namespace NUMINAMATH_CALUDE_find_original_numbers_l3761_376152

theorem find_original_numbers (x y : ℤ) 
  (sum_condition : x + y = 2022)
  (modified_sum_condition : (x - 5) / 10 + 10 * y + 1 = 2252) :
  x = 1815 ∧ y = 207 := by
  sorry

end NUMINAMATH_CALUDE_find_original_numbers_l3761_376152


namespace NUMINAMATH_CALUDE_problem_statement_l3761_376199

theorem problem_statement (x : ℝ) :
  x = (Real.sqrt (6 + 2 * Real.sqrt 5) + Real.sqrt (6 - 2 * Real.sqrt 5)) / Real.sqrt 20 →
  (1 + x^5 - x^7)^(2012^(3^11)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3761_376199


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3761_376107

/-- Given a line with equation y = mx + b, where m = -2/3 and b = 3/2, prove that mb = -1 -/
theorem line_slope_intercept_product (m b : ℚ) : 
  m = -2/3 → b = 3/2 → m * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3761_376107


namespace NUMINAMATH_CALUDE_average_price_per_book_l3761_376154

-- Define the problem parameters
def books_shop1 : ℕ := 32
def cost_shop1 : ℕ := 1500
def books_shop2 : ℕ := 60
def cost_shop2 : ℕ := 340

-- Theorem to prove
theorem average_price_per_book :
  (cost_shop1 + cost_shop2) / (books_shop1 + books_shop2) = 20 := by
  sorry


end NUMINAMATH_CALUDE_average_price_per_book_l3761_376154


namespace NUMINAMATH_CALUDE_balls_per_bag_l3761_376110

theorem balls_per_bag (total_balls : ℕ) (num_bags : ℕ) (balls_per_bag : ℕ) : 
  total_balls = 36 → num_bags = 9 → total_balls = num_bags * balls_per_bag → balls_per_bag = 4 := by
  sorry

end NUMINAMATH_CALUDE_balls_per_bag_l3761_376110


namespace NUMINAMATH_CALUDE_slacks_percentage_is_25_percent_l3761_376195

/-- Represents the clothing items and their quantities -/
structure Wardrobe where
  blouses : ℕ
  skirts : ℕ
  slacks : ℕ

/-- Represents the percentages of clothing items in the hamper -/
structure HamperPercentages where
  blouses : ℚ
  skirts : ℚ
  slacks : ℚ

/-- Calculates the percentage of slacks in the hamper -/
def calculate_slacks_percentage (w : Wardrobe) (h : HamperPercentages) (total_in_washer : ℕ) : ℚ :=
  let blouses_in_hamper := (w.blouses : ℚ) * h.blouses
  let skirts_in_hamper := (w.skirts : ℚ) * h.skirts
  let slacks_in_hamper := (total_in_washer : ℚ) - blouses_in_hamper - skirts_in_hamper
  slacks_in_hamper / (w.slacks : ℚ)

/-- Theorem stating that the percentage of slacks in the hamper is 25% -/
theorem slacks_percentage_is_25_percent (w : Wardrobe) (h : HamperPercentages) :
  w.blouses = 12 →
  w.skirts = 6 →
  w.slacks = 8 →
  h.blouses = 3/4 →
  h.skirts = 1/2 →
  calculate_slacks_percentage w h 14 = 1/4 := by
  sorry

#eval (1 : ℚ) / 4  -- To verify that 1/4 is indeed 25%

end NUMINAMATH_CALUDE_slacks_percentage_is_25_percent_l3761_376195


namespace NUMINAMATH_CALUDE_celebrity_match_probability_l3761_376157

/-- The number of celebrities -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities with their pictures and hobbies -/
def correct_match_probability : ℚ := 1 / (n.factorial * n.factorial)

/-- Theorem: The probability of correctly matching all celebrities with their pictures and hobbies is 1/576 -/
theorem celebrity_match_probability :
  correct_match_probability = 1 / 576 := by sorry

end NUMINAMATH_CALUDE_celebrity_match_probability_l3761_376157


namespace NUMINAMATH_CALUDE_amritsar_bombay_encounters_l3761_376183

/-- Represents a train journey from Amritsar to Bombay -/
structure TrainJourney where
  startTime : Nat  -- Start time in minutes after midnight
  duration : Nat   -- Duration of journey in minutes
  dailyDepartures : Nat  -- Number of trains departing each day

/-- Calculates the number of trains encountered during the journey -/
def encountersCount (journey : TrainJourney) : Nat :=
  sorry

/-- Theorem stating that a train journey with given conditions encounters 5 other trains -/
theorem amritsar_bombay_encounters :
  ∀ (journey : TrainJourney),
    journey.startTime = 9 * 60 →  -- 9 am start time
    journey.duration = 3 * 24 * 60 + 30 →  -- 3 days and 30 minutes duration
    journey.dailyDepartures = 1 →  -- One train departs each day
    encountersCount journey = 5 :=
  sorry

end NUMINAMATH_CALUDE_amritsar_bombay_encounters_l3761_376183


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3761_376109

theorem solve_linear_equation (x : ℝ) (h : 3*x - 4*x + 7*x = 120) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3761_376109


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3761_376178

theorem complex_equation_solution (a b : ℝ) : 
  (a - 2 * Complex.I = (b + Complex.I) * Complex.I) → (a = -1 ∧ b = -2) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3761_376178


namespace NUMINAMATH_CALUDE_wind_velocity_theorem_l3761_376108

-- Define the relationship between pressure, area, and velocity
def pressure_relation (k : ℝ) (A : ℝ) (V : ℝ) : ℝ := k * A * V^2

-- Given initial condition
def initial_condition (k : ℝ) : Prop :=
  pressure_relation k 9 105 = 4

-- Theorem to prove
theorem wind_velocity_theorem (k : ℝ) (h : initial_condition k) :
  pressure_relation k 36 70 = 64 := by
  sorry

end NUMINAMATH_CALUDE_wind_velocity_theorem_l3761_376108


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_is_sqrt_6_over_36_l3761_376149

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (lateral_face_equilateral : Bool)

/-- A cube inscribed in a pyramid -/
structure InscribedCube :=
  (pyramid : Pyramid)
  (bottom_face_on_base : Bool)
  (top_face_edges_on_lateral_faces : Bool)

/-- The volume of an inscribed cube in a specific pyramid -/
noncomputable def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume_is_sqrt_6_over_36 
  (cube : InscribedCube) 
  (h1 : cube.pyramid.base_side = 1) 
  (h2 : cube.pyramid.lateral_face_equilateral = true)
  (h3 : cube.bottom_face_on_base = true)
  (h4 : cube.top_face_edges_on_lateral_faces = true) : 
  inscribed_cube_volume cube = Real.sqrt 6 / 36 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_is_sqrt_6_over_36_l3761_376149


namespace NUMINAMATH_CALUDE_annes_bottle_caps_l3761_376175

/-- Anne's initial number of bottle caps -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Anne finds -/
def found_caps : ℕ := 5

/-- Anne's final number of bottle caps -/
def final_caps : ℕ := 15

/-- Theorem stating that Anne's initial number of bottle caps plus the found caps equals her final number of caps -/
theorem annes_bottle_caps : initial_caps + found_caps = final_caps := by sorry

end NUMINAMATH_CALUDE_annes_bottle_caps_l3761_376175


namespace NUMINAMATH_CALUDE_magic_trick_minimum_cards_l3761_376127

/-- The number of possible colors for the cards -/
def num_colors : ℕ := 2017

/-- The strategy function type for the assistant -/
def Strategy : Type := Fin num_colors → Fin num_colors

/-- The minimum number of cards needed for the trick -/
def min_cards : ℕ := 2018

theorem magic_trick_minimum_cards :
  ∀ (n : ℕ), n < min_cards →
    ¬∃ (s : Strategy),
      ∀ (colors : Fin n → Fin num_colors),
        ∃ (i : Fin n),
          ∀ (j : Fin n),
            j ≠ i →
              s (colors j) = colors i := by sorry

end NUMINAMATH_CALUDE_magic_trick_minimum_cards_l3761_376127


namespace NUMINAMATH_CALUDE_unique_row_with_29_l3761_376113

def pascal_coeff (n k : ℕ) : ℕ := Nat.choose n k

def contains_29 (row : ℕ) : Prop :=
  ∃ k, k ≤ row ∧ pascal_coeff row k = 29

theorem unique_row_with_29 :
  ∃! row, contains_29 row :=
sorry

end NUMINAMATH_CALUDE_unique_row_with_29_l3761_376113


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3761_376186

theorem largest_divisor_of_n (n : ℕ+) (h : 650 ∣ n^3) : 130 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3761_376186


namespace NUMINAMATH_CALUDE_physics_marks_l3761_376143

theorem physics_marks (P C M : ℝ) 
  (total_avg : (P + C + M) / 3 = 85)
  (phys_math_avg : (P + M) / 2 = 90)
  (phys_chem_avg : (P + C) / 2 = 70) :
  P = 65 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l3761_376143


namespace NUMINAMATH_CALUDE_binomial_coeff_x8_eq_10_l3761_376105

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the function to get the exponent of x in the general term
def x_exponent (r : ℕ) : ℚ := 15 - (7 * r) / 2

-- Define the function to find the binomial coefficient for x^8
def binomial_coeff_x8 (n : ℕ) : ℕ :=
  let r := 2 -- r is 2 when x_exponent(r) = 8
  binomial_coeff n r

-- Theorem statement
theorem binomial_coeff_x8_eq_10 :
  binomial_coeff_x8 5 = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_coeff_x8_eq_10_l3761_376105


namespace NUMINAMATH_CALUDE_max_ab_l3761_376106

theorem max_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4*a + b = 1) :
  ab ≤ 1/16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4*a₀ + b₀ = 1 ∧ a₀*b₀ = 1/16 :=
sorry


end NUMINAMATH_CALUDE_max_ab_l3761_376106


namespace NUMINAMATH_CALUDE_quadratic_equation_k_value_l3761_376126

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := k * x^2 + x - 3 = 0

-- Define the condition for distinct real roots
def has_distinct_real_roots (k : ℝ) : Prop := k > -1/12 ∧ k ≠ 0

-- Define the condition for the roots
def roots_condition (x₁ x₂ : ℝ) : Prop := (x₁ + x₂)^2 + x₁ * x₂ = 4

-- Theorem statement
theorem quadratic_equation_k_value :
  ∀ k : ℝ, 
  (∃ x₁ x₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    quadratic_equation k x₁ ∧ 
    quadratic_equation k x₂ ∧
    has_distinct_real_roots k ∧
    roots_condition x₁ x₂) →
  k = 1/4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_k_value_l3761_376126


namespace NUMINAMATH_CALUDE_max_product_representation_l3761_376137

def representation_sum (n : ℕ) : List ℕ → Prop :=
  λ l => l.sum = n ∧ l.all (· > 0)

theorem max_product_representation (n : ℕ) :
  ∃ (l : List ℕ), representation_sum 2015 l ∧
    ∀ (m : List ℕ), representation_sum 2015 m →
      l.prod ≥ m.prod :=
by
  sorry

#check max_product_representation 2015

end NUMINAMATH_CALUDE_max_product_representation_l3761_376137


namespace NUMINAMATH_CALUDE_lines_skew_when_one_parallel_to_plane_other_in_plane_l3761_376100

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (We'll leave this abstract for now)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane in 3D space
  -- (We'll leave this abstract for now)

/-- Proposition that a line is parallel to a plane -/
def is_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define what it means for a line to be parallel to a plane
  sorry

/-- Proposition that a line is contained within a plane -/
def is_contained_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  -- Define what it means for a line to be contained in a plane
  sorry

/-- Proposition that two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Define what it means for two lines to be skew
  sorry

/-- Theorem statement -/
theorem lines_skew_when_one_parallel_to_plane_other_in_plane 
  (a b : Line3D) (α : Plane3D) 
  (h1 : is_parallel_to_plane a α) 
  (h2 : is_contained_in_plane b α) : 
  are_skew a b :=
sorry

end NUMINAMATH_CALUDE_lines_skew_when_one_parallel_to_plane_other_in_plane_l3761_376100


namespace NUMINAMATH_CALUDE_subset_intersection_condition_l3761_376120

theorem subset_intersection_condition (a : ℝ) : 
  let A := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 22}
  (∃ x, x ∈ A) →
  (A ⊆ A ∩ B) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_condition_l3761_376120


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3761_376119

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 10 ∧ |r₁ - r₂| = 2) ↔ (a = 1 ∧ b = -10 ∧ c = 24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3761_376119


namespace NUMINAMATH_CALUDE_sue_necklace_beads_l3761_376198

def necklace_beads (purple blue green red : ℕ) : Prop :=
  (blue = 2 * purple) ∧
  (green = blue + 11) ∧
  (red = green / 2) ∧
  (purple + blue + green + red = 58)

theorem sue_necklace_beads :
  ∃ (purple blue green red : ℕ),
    purple = 7 ∧
    necklace_beads purple blue green red :=
by sorry

end NUMINAMATH_CALUDE_sue_necklace_beads_l3761_376198


namespace NUMINAMATH_CALUDE_kids_at_camp_l3761_376197

theorem kids_at_camp (total : ℕ) (home : ℕ) (difference : ℕ) : 
  total = home + (home + difference) → 
  home = 668278 → 
  difference = 150780 → 
  home + difference = 409529 :=
by sorry

end NUMINAMATH_CALUDE_kids_at_camp_l3761_376197


namespace NUMINAMATH_CALUDE_power_function_value_l3761_376130

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x^α

-- Define the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 4 = 1/2 → f (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l3761_376130


namespace NUMINAMATH_CALUDE_equal_color_diagonals_l3761_376165

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → Point

/-- A coloring of vertices of a polygon -/
def VertexColoring (n : ℕ) := Fin n → Bool

/-- The number of diagonals with both endpoints of a given color -/
def numSameColorDiagonals (n : ℕ) (coloring : VertexColoring n) (color : Bool) : ℕ := sorry

theorem equal_color_diagonals 
  (polygon : RegularPolygon 20) 
  (coloring : VertexColoring 20)
  (h_black_count : (Finset.filter (fun i => coloring i = true) (Finset.univ : Finset (Fin 20))).card = 10)
  (h_white_count : (Finset.filter (fun i => coloring i = false) (Finset.univ : Finset (Fin 20))).card = 10) :
  numSameColorDiagonals 20 coloring true = numSameColorDiagonals 20 coloring false := by
  sorry


end NUMINAMATH_CALUDE_equal_color_diagonals_l3761_376165


namespace NUMINAMATH_CALUDE_boys_on_playground_l3761_376133

/-- The number of boys on a playground, given the total number of children and the number of girls. -/
def number_of_boys (total_children : ℕ) (number_of_girls : ℕ) : ℕ :=
  total_children - number_of_girls

/-- Theorem stating that the number of boys on the playground is 40. -/
theorem boys_on_playground : number_of_boys 117 77 = 40 := by
  sorry

end NUMINAMATH_CALUDE_boys_on_playground_l3761_376133


namespace NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l3761_376128

theorem smallest_positive_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 37 = 17 % 37 ∧ 
  ∀ (y : ℕ), y > 0 → (4 * y) % 37 = 17 % 37 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l3761_376128


namespace NUMINAMATH_CALUDE_photo_arrangement_probability_l3761_376160

/-- The number of boys -/
def num_boys : ℕ := 2

/-- The number of girls -/
def num_girls : ℕ := 5

/-- The total number of people -/
def total_people : ℕ := num_boys + num_girls

/-- The number of girls between the boys -/
def girls_between : ℕ := 3

/-- The probability of the specific arrangement -/
def probability : ℚ := 1 / 7

theorem photo_arrangement_probability :
  (num_boys = 2) →
  (num_girls = 5) →
  (girls_between = 3) →
  (probability = 1 / 7) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangement_probability_l3761_376160


namespace NUMINAMATH_CALUDE_cube_diff_of_squares_l3761_376112

theorem cube_diff_of_squares (a : ℕ+) : ∃ x y : ℤ, x^2 - y^2 = (a : ℤ)^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_of_squares_l3761_376112


namespace NUMINAMATH_CALUDE_new_circle_equation_l3761_376190

/-- Given a circle with equation x^2 + 2x + y^2 = 0, prove that a new circle 
    with the same center and radius 2 has the equation (x+1)^2 + y^2 = 4 -/
theorem new_circle_equation (x y : ℝ) : 
  (∀ x y, x^2 + 2*x + y^2 = 0 → ∃ h k, (x - h)^2 + (y - k)^2 = 1) →
  (∀ x y, (x + 1)^2 + y^2 = 4 ↔ (x - (-1))^2 + (y - 0)^2 = 2^2) :=
by sorry

end NUMINAMATH_CALUDE_new_circle_equation_l3761_376190


namespace NUMINAMATH_CALUDE_missing_number_is_four_l3761_376189

theorem missing_number_is_four : 
  ∃ x : ℤ, (x + 3) + (8 - 3 - 1) = 11 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_missing_number_is_four_l3761_376189


namespace NUMINAMATH_CALUDE_painted_faces_count_l3761_376159

/-- Represents a cube with a given side length -/
structure Cube :=
  (side_length : ℕ)

/-- Represents a painted cube with three adjacent painted faces -/
structure PaintedCube extends Cube :=
  (painted_faces : Fin 3)

/-- Counts the number of unit cubes with at least two painted faces when a painted cube is cut into unit cubes -/
def count_multi_painted_faces (c : PaintedCube) : ℕ :=
  sorry

/-- Theorem stating that a 4x4x4 cube painted on three adjacent faces, when cut into unit cubes, has 14 cubes with at least two painted faces -/
theorem painted_faces_count (c : PaintedCube) (h : c.side_length = 4) : 
  count_multi_painted_faces c = 14 :=
sorry

end NUMINAMATH_CALUDE_painted_faces_count_l3761_376159


namespace NUMINAMATH_CALUDE_power_sum_fifth_l3761_376151

theorem power_sum_fifth (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 25)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = 136.25 := by sorry

end NUMINAMATH_CALUDE_power_sum_fifth_l3761_376151


namespace NUMINAMATH_CALUDE_delta_triple_72_l3761_376158

-- Define the Δ function
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Theorem statement
theorem delta_triple_72 : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end NUMINAMATH_CALUDE_delta_triple_72_l3761_376158


namespace NUMINAMATH_CALUDE_imaginary_roots_condition_l3761_376111

/-- The quadratic equation kx^2 + mx + k = 0 (where k ≠ 0) has imaginary roots
    if and only if m^2 < 4k^2 -/
theorem imaginary_roots_condition (k m : ℝ) (hk : k ≠ 0) :
  (∀ x : ℝ, k * x^2 + m * x + k ≠ 0) ↔ m^2 < 4 * k^2 := by sorry

end NUMINAMATH_CALUDE_imaginary_roots_condition_l3761_376111


namespace NUMINAMATH_CALUDE_min_n_for_constant_term_l3761_376124

theorem min_n_for_constant_term (x : ℝ) : 
  (∃ (n : ℕ), n > 0 ∧ (∃ (r : ℕ), r ≤ n ∧ 3 * n = 7 * r)) ∧
  (∀ (m : ℕ), m > 0 → (∃ (r : ℕ), r ≤ m ∧ 3 * m = 7 * r) → m ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_min_n_for_constant_term_l3761_376124


namespace NUMINAMATH_CALUDE_max_gumdrops_l3761_376156

/-- Represents the candy purchasing problem with given constraints --/
def CandyProblem (total_budget : ℕ) (bulk_cost gummy_cost gumdrop_cost : ℕ) 
                 (min_bulk min_gummy : ℕ) : Prop :=
  let remaining_budget := total_budget - (min_bulk * bulk_cost + min_gummy * gummy_cost)
  remaining_budget / gumdrop_cost = 28

/-- Theorem stating the maximum number of gumdrops that can be purchased --/
theorem max_gumdrops : 
  CandyProblem 224 8 6 4 10 5 := by
  sorry

#check max_gumdrops

end NUMINAMATH_CALUDE_max_gumdrops_l3761_376156


namespace NUMINAMATH_CALUDE_mark_sprint_speed_l3761_376192

/-- Given a distance of 144 miles traveled in 24.0 hours, prove the speed is 6 miles per hour. -/
theorem mark_sprint_speed (distance : ℝ) (time : ℝ) (h1 : distance = 144) (h2 : time = 24.0) :
  distance / time = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_sprint_speed_l3761_376192


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3761_376169

theorem inequalities_theorem (a b c : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : c < 0) :
  (c / a > c / b) ∧ ((a - c)^c < (b - c)^c) ∧ (b * Real.exp a > a * Real.exp b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3761_376169


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3761_376125

/-- The asymptote of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let asymptote_parallel : ℝ → ℝ → Prop := λ x y => y = (b / a) * (x - c)
  ∃ (P : ℝ × ℝ), 
    C P.1 P.2 ∧ 
    asymptote_parallel P.1 P.2 ∧
    ((P.1 + c) * (P.1 - c) + P.2^2 = 0) →
    (∀ (x y : ℝ), y = 2 * x ∨ y = -2 * x ↔ x^2 / a^2 - y^2 / b^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3761_376125


namespace NUMINAMATH_CALUDE_equal_fractions_imply_equal_values_l3761_376161

theorem equal_fractions_imply_equal_values (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h1 : (a + b) / c = (c + b) / a) 
  (h2 : (c + b) / a = (a + c) / b) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_imply_equal_values_l3761_376161


namespace NUMINAMATH_CALUDE_problem_statement_l3761_376173

theorem problem_statement (n b : ℝ) : n = 2^(1/10) ∧ n^b = 16 → b = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3761_376173


namespace NUMINAMATH_CALUDE_simplify_expression_l3761_376166

theorem simplify_expression : 
  ((1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)))^2 = 1/4 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3761_376166


namespace NUMINAMATH_CALUDE_biased_coin_probability_l3761_376181

theorem biased_coin_probability (p : ℝ) : 
  p < (1 : ℝ) / 2 →
  (20 : ℝ) * p^3 * (1 - p)^3 = (5 : ℝ) / 32 →
  p = (1 - Real.sqrt ((32 - 4 * Real.rpow 5 (1/3)) / 8)) / 2 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l3761_376181


namespace NUMINAMATH_CALUDE_race_end_count_l3761_376171

/-- Represents the total number of people in all cars at the end of a race with given conditions. -/
def total_people_at_end (num_cars : ℕ) (initial_people_per_car : ℕ) 
  (first_quarter_gain : ℕ) (half_way_gain : ℕ) (three_quarter_gain : ℕ) : ℕ :=
  num_cars * (initial_people_per_car + first_quarter_gain + half_way_gain + three_quarter_gain)

/-- Theorem stating that under the given race conditions, the total number of people at the end is 450. -/
theorem race_end_count : 
  total_people_at_end 50 4 2 2 1 = 450 := by
  sorry

#eval total_people_at_end 50 4 2 2 1

end NUMINAMATH_CALUDE_race_end_count_l3761_376171


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l3761_376177

theorem at_least_one_not_less_than_six (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ¬(a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_six_l3761_376177


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3761_376164

/-- Two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def is_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are different -/
def are_different (l1 l2 : Line) : Prop := sorry

theorem line_perpendicular_to_plane 
  (m n : Line) (β : Plane) 
  (h1 : are_different m n)
  (h2 : are_parallel m n) 
  (h3 : is_perpendicular_to_plane n β) : 
  is_perpendicular_to_plane m β := by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3761_376164


namespace NUMINAMATH_CALUDE_f_three_l3761_376117

/-- A function satisfying the given property -/
def f (x : ℝ) : ℝ := sorry

/-- The property of the function f -/
axiom f_property (x y : ℝ) : f (x + y) = f x + f y + x * y

/-- The given condition that f(1) = 1 -/
axiom f_one : f 1 = 1

/-- Theorem stating that f(3) = 6 -/
theorem f_three : f 3 = 6 := by sorry

end NUMINAMATH_CALUDE_f_three_l3761_376117


namespace NUMINAMATH_CALUDE_unique_digit_product_l3761_376134

theorem unique_digit_product (n : ℕ) : n ≤ 9 → (n * (10 * n + n) = 176) ↔ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_product_l3761_376134


namespace NUMINAMATH_CALUDE_different_solution_set_l3761_376150

-- Define the equations
def eq0 (x : ℚ) := x - 3 = 3 * x + 4
def eqA := 79 - 4 = 59 - 11
def eqB (x : ℚ) := 1 / (x + 3) + 2 = 0
def eqC (x a : ℚ) := (a^2 + 1) * (x - 3) = (3 * x + 4) * (a^2 + 1)
def eqD (x : ℚ) := (7 * x - 4) * (x - 1) = (5 * x - 11) * (x - 1)

-- Define the solution set of eq0
def sol0 : Set ℚ := {x | eq0 x}

-- State the theorem
theorem different_solution_set : 
  (∃ x, eqD x ∧ x ∉ sol0) ∧ 
  (∀ x, eqB x → x ∈ sol0) ∧
  (∀ x a, eqC x a → x ∈ sol0) ∧
  (sol0 = {-7/2}) := by
  sorry

end NUMINAMATH_CALUDE_different_solution_set_l3761_376150


namespace NUMINAMATH_CALUDE_two_fifths_in_three_fourths_l3761_376123

theorem two_fifths_in_three_fourths : (3 : ℚ) / 4 / ((2 : ℚ) / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_two_fifths_in_three_fourths_l3761_376123


namespace NUMINAMATH_CALUDE_largest_unorderable_correct_l3761_376118

/-- The largest number of dumplings that cannot be ordered -/
def largest_unorderable : ℕ := 43

/-- The set of possible portion sizes for dumplings -/
def portion_sizes : Finset ℕ := {6, 9, 20}

/-- Predicate to check if a number can be expressed as a combination of portion sizes -/
def is_orderable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_unorderable_correct :
  (∀ n > largest_unorderable, is_orderable n) ∧
  ¬(is_orderable largest_unorderable) ∧
  (∀ m < largest_unorderable, ∃ n > m, ¬(is_orderable n)) :=
sorry

end NUMINAMATH_CALUDE_largest_unorderable_correct_l3761_376118


namespace NUMINAMATH_CALUDE_basketball_problem_l3761_376146

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descents := List.range (bounces + 1) |>.map (fun n => initialHeight * reboundFactor ^ n)
  let ascents := List.range bounces |>.map (fun n => initialHeight * reboundFactor ^ (n + 1))
  (descents.sum + ascents.sum)

/-- The basketball problem -/
theorem basketball_problem :
  totalDistance 150 (2/5) 5 = 347.952 := by
  sorry

end NUMINAMATH_CALUDE_basketball_problem_l3761_376146


namespace NUMINAMATH_CALUDE_number_of_blue_marbles_l3761_376188

/-- Given the number of yellow, red, and blue marbles, prove that the number of blue marbles is 108 -/
theorem number_of_blue_marbles
  (yellow red blue : ℕ)
  (h1 : blue = 3 * red)
  (h2 : red = yellow + 15)
  (h3 : yellow + red + blue = 165) :
  blue = 108 := by
  sorry

end NUMINAMATH_CALUDE_number_of_blue_marbles_l3761_376188


namespace NUMINAMATH_CALUDE_reception_hall_tables_l3761_376187

/-- The cost of a linen tablecloth -/
def tablecloth_cost : ℕ := 25

/-- The cost of a single place setting -/
def place_setting_cost : ℕ := 10

/-- The number of place settings per table -/
def place_settings_per_table : ℕ := 4

/-- The cost of a single rose -/
def rose_cost : ℕ := 5

/-- The number of roses per centerpiece -/
def roses_per_centerpiece : ℕ := 10

/-- The cost of a single lily -/
def lily_cost : ℕ := 4

/-- The number of lilies per centerpiece -/
def lilies_per_centerpiece : ℕ := 15

/-- The total decoration budget -/
def total_budget : ℕ := 3500

/-- The cost of decorations for a single table -/
def cost_per_table : ℕ :=
  tablecloth_cost +
  place_setting_cost * place_settings_per_table +
  rose_cost * roses_per_centerpiece +
  lily_cost * lilies_per_centerpiece

/-- The number of tables at the reception hall -/
def number_of_tables : ℕ := total_budget / cost_per_table

theorem reception_hall_tables :
  number_of_tables = 20 :=
sorry

end NUMINAMATH_CALUDE_reception_hall_tables_l3761_376187


namespace NUMINAMATH_CALUDE_three_digit_twice_divisible_by_1001_l3761_376103

theorem three_digit_twice_divisible_by_1001 (a : ℕ) : 
  100 ≤ a ∧ a < 1000 → ∃ k : ℕ, 1000 * a + a = 1001 * k := by
sorry

end NUMINAMATH_CALUDE_three_digit_twice_divisible_by_1001_l3761_376103


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_one_l3761_376142

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem f_neg_two_eq_neg_one : f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_one_l3761_376142


namespace NUMINAMATH_CALUDE_cello_count_l3761_376145

/-- Given a music store with cellos and violas, prove the number of cellos. -/
theorem cello_count (violas : ℕ) (matching_pairs : ℕ) (probability : ℚ) (cellos : ℕ) : 
  violas = 600 →
  matching_pairs = 70 →
  probability = 70 / (cellos * 600) →
  probability = 0.00014583333333333335 →
  cellos = 800 := by
sorry

#eval (70 : ℚ) / (800 * 600)  -- To verify the probability

end NUMINAMATH_CALUDE_cello_count_l3761_376145


namespace NUMINAMATH_CALUDE_minimum_framing_for_specific_photo_l3761_376168

/-- Calculates the minimum framing needed for a scaled photograph with a border -/
def minimum_framing (original_width original_height scale_factor border_width : ℕ) : ℕ :=
  let scaled_width := original_width * scale_factor
  let scaled_height := original_height * scale_factor
  let total_width := scaled_width + 2 * border_width
  let total_height := scaled_height + 2 * border_width
  let perimeter := 2 * (total_width + total_height)
  (perimeter + 11) / 12  -- Round up to the nearest foot

theorem minimum_framing_for_specific_photo :
  minimum_framing 5 7 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_minimum_framing_for_specific_photo_l3761_376168


namespace NUMINAMATH_CALUDE_equation_solutions_l3761_376141

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁^2 - 4*x₁ - 8 = 0 ∧ x₂^2 - 4*x₂ - 8 = 0) ∧
    x₁ = 2*Real.sqrt 3 + 2 ∧ x₂ = -2*Real.sqrt 3 + 2) ∧
  (∃ y₁ y₂ : ℝ, (3*y₁ - 6 = y₁*(y₁ - 2) ∧ 3*y₂ - 6 = y₂*(y₂ - 2)) ∧
    y₁ = 2 ∧ y₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3761_376141


namespace NUMINAMATH_CALUDE_vehicle_wheels_count_l3761_376170

/-- Proves that each vehicle has 4 wheels given the problem conditions -/
theorem vehicle_wheels_count (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 25)
  (h2 : total_wheels = 100) :
  total_wheels / total_vehicles = 4 := by
  sorry

#check vehicle_wheels_count

end NUMINAMATH_CALUDE_vehicle_wheels_count_l3761_376170


namespace NUMINAMATH_CALUDE_max_rooks_on_chessboard_sixteen_rooks_achievable_l3761_376191

/-- Represents a chessboard with rooks --/
structure Chessboard :=
  (size : ℕ)
  (white_rooks : ℕ)
  (black_rooks : ℕ)

/-- Predicate to check if the rook placement is valid --/
def valid_placement (board : Chessboard) : Prop :=
  board.white_rooks ≤ board.size * 2 ∧ 
  board.black_rooks ≤ board.size * 2 ∧
  board.white_rooks = board.black_rooks

/-- Theorem stating the maximum number of rooks of each color --/
theorem max_rooks_on_chessboard :
  ∀ (board : Chessboard),
    board.size = 8 →
    valid_placement board →
    board.white_rooks ≤ 16 ∧
    board.black_rooks ≤ 16 :=
by sorry

/-- Theorem stating that 16 rooks of each color is achievable --/
theorem sixteen_rooks_achievable :
  ∃ (board : Chessboard),
    board.size = 8 ∧
    valid_placement board ∧
    board.white_rooks = 16 ∧
    board.black_rooks = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_rooks_on_chessboard_sixteen_rooks_achievable_l3761_376191


namespace NUMINAMATH_CALUDE_grey_area_ratio_l3761_376138

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square piece of paper -/
structure Square where
  sideLength : ℝ
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a kite shape -/
structure Kite where
  a : Point
  e : Point
  c : Point
  f : Point

/-- Function to fold the paper along a line -/
def foldPaper (s : Square) (p : Point) : Kite :=
  sorry

/-- Theorem stating the ratio of grey area to total area of the kite -/
theorem grey_area_ratio (s : Square) (e f : Point) :
  let k := foldPaper s e
  let k' := foldPaper s f
  let greyArea := sorry
  let totalArea := sorry
  greyArea / totalArea = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_grey_area_ratio_l3761_376138


namespace NUMINAMATH_CALUDE_line_circle_no_intersection_l3761_376167

/-- The line and circle have no intersection points in the real plane -/
theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + 2 * y^2 = 2) → False :=
by
  sorry

#check line_circle_no_intersection

end NUMINAMATH_CALUDE_line_circle_no_intersection_l3761_376167


namespace NUMINAMATH_CALUDE_cody_purchase_price_l3761_376180

/-- The initial price of Cody's purchase before taxes -/
def initial_price : ℝ := 40

/-- The tax rate applied to the purchase -/
def tax_rate : ℝ := 0.05

/-- The discount applied after taxes -/
def discount : ℝ := 8

/-- Cody's share of the final payment -/
def cody_payment : ℝ := 17

theorem cody_purchase_price :
  let price_after_tax := initial_price * (1 + tax_rate)
  let price_after_discount := price_after_tax - discount
  let total_final_price := 2 * cody_payment
  price_after_discount = total_final_price :=
by sorry

end NUMINAMATH_CALUDE_cody_purchase_price_l3761_376180


namespace NUMINAMATH_CALUDE_horse_sale_problem_l3761_376131

theorem horse_sale_problem (x : ℝ) : 
  (x - x^2 / 100 = 24) → (x = 40 ∨ x = 60) :=
by
  sorry

end NUMINAMATH_CALUDE_horse_sale_problem_l3761_376131


namespace NUMINAMATH_CALUDE_Q_not_subset_P_l3761_376132

-- Define set P
def P : Set ℝ := {y | y ≥ 0}

-- Define set Q
def Q : Set ℝ := {y | ∃ x, y = Real.log x}

-- Theorem statement
theorem Q_not_subset_P : ¬(Q ⊆ P ∧ P ∩ Q = Q) := by
  sorry

end NUMINAMATH_CALUDE_Q_not_subset_P_l3761_376132


namespace NUMINAMATH_CALUDE_inequality_proof_l3761_376129

theorem inequality_proof (w x y z : ℝ) 
  (h_non_neg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : w * x + x * y + y * z + z * w = 1) : 
  w^3 / (x + y + z) + x^3 / (w + y + z) + y^3 / (w + x + z) + z^3 / (w + x + y) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3761_376129


namespace NUMINAMATH_CALUDE_unique_phone_number_l3761_376114

def is_valid_phone_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def first_upgrade (n : ℕ) : ℕ :=
  let d := n.div 100000
  let r := n.mod 100000
  d * 1000000 + 8 * 100000 + r

def second_upgrade (n : ℕ) : ℕ :=
  2000000000 + n

theorem unique_phone_number :
  ∃! n : ℕ, is_valid_phone_number n ∧ 
    second_upgrade (first_upgrade n) = 81 * n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_phone_number_l3761_376114


namespace NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_zero_l3761_376121

/-- The function f(x) = (x+a)e^x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

/-- The derivative of f(x) --/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  (x + a + 1) * Real.exp x

theorem tangent_perpendicular_implies_a_zero (a : ℝ) : 
  (f_derivative a 0 = 1) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_implies_a_zero_l3761_376121


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l3761_376136

/-- Given two intersecting chords in a circle, where one chord is divided into segments
    of 12 cm and 18 cm, and the other chord is divided in the ratio 3:8,
    prove that the length of the second chord is 33 cm. -/
theorem intersecting_chords_theorem (chord1_seg1 chord1_seg2 : ℝ)
  (chord2_ratio1 chord2_ratio2 : ℕ) :
  chord1_seg1 = 12 →
  chord1_seg2 = 18 →
  chord2_ratio1 = 3 →
  chord2_ratio2 = 8 →
  chord1_seg1 * chord1_seg2 = (chord2_ratio1 : ℝ) * (chord2_ratio2 : ℝ) * ((33 : ℝ) / (chord2_ratio1 + chord2_ratio2))^2 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l3761_376136
