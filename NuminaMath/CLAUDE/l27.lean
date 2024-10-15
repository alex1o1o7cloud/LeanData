import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l27_2723

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 7) :
  a 5 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l27_2723


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l27_2797

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 3*x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 3

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = -2*x + 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l27_2797


namespace NUMINAMATH_CALUDE_complex_product_equals_one_l27_2730

theorem complex_product_equals_one (x : ℂ) (h : x = Complex.exp (Complex.I * Real.pi / 7)) :
  (x^2 + x^4) * (x^4 + x^8) * (x^6 + x^12) * (x^8 + x^16) * (x^10 + x^20) * (x^12 + x^24) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_one_l27_2730


namespace NUMINAMATH_CALUDE_total_distance_traveled_l27_2796

/-- Calculates the total distance traveled given walking and running speeds and durations, with a break. -/
theorem total_distance_traveled
  (total_time : ℝ)
  (walking_time : ℝ)
  (walking_speed : ℝ)
  (running_time : ℝ)
  (running_speed : ℝ)
  (break_time : ℝ)
  (h1 : total_time = 2)
  (h2 : walking_time = 1)
  (h3 : walking_speed = 3.5)
  (h4 : running_time = 0.75)
  (h5 : running_speed = 8)
  (h6 : break_time = 0.25)
  (h7 : total_time = walking_time + running_time + break_time) :
  walking_time * walking_speed + running_time * running_speed = 9.5 := by
  sorry


end NUMINAMATH_CALUDE_total_distance_traveled_l27_2796


namespace NUMINAMATH_CALUDE_prob_white_second_is_half_l27_2771

/-- Represents the number of black balls initially in the bag -/
def initial_black_balls : ℕ := 4

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 3

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := initial_black_balls + initial_white_balls

/-- Represents the probability of drawing a white ball on the second draw,
    given that a black ball was drawn on the first draw -/
def prob_white_second_given_black_first : ℚ :=
  initial_white_balls / (total_balls - 1)

theorem prob_white_second_is_half :
  prob_white_second_given_black_first = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_white_second_is_half_l27_2771


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l27_2787

/-- Represents a repeating decimal of the form 0.abcabc... where abc is a finite sequence of digits -/
def RepeatingDecimal (numerator denominator : ℕ) : ℚ := numerator / denominator

theorem repeating_decimal_sum : 
  RepeatingDecimal 4 33 + RepeatingDecimal 2 999 + RepeatingDecimal 2 99999 = 12140120 / 99999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l27_2787


namespace NUMINAMATH_CALUDE_martin_financial_calculation_l27_2717

theorem martin_financial_calculation (g u q : ℂ) (h1 : g * q - u = 15000) (h2 : g = 10) (h3 : u = 10 + 200 * Complex.I) : q = 1501 + 20 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_martin_financial_calculation_l27_2717


namespace NUMINAMATH_CALUDE_final_sum_theorem_l27_2792

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  3 * (a + 4) + 3 * (b + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l27_2792


namespace NUMINAMATH_CALUDE_fraction_problem_l27_2764

theorem fraction_problem (x : ℚ) : (3/4 : ℚ) * x * (2/3 : ℚ) = (2/5 : ℚ) → x = (4/5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l27_2764


namespace NUMINAMATH_CALUDE_joan_payment_l27_2784

/-- Represents the purchase amounts for Joan, Karl, and Lea --/
structure Purchases where
  joan : ℝ
  karl : ℝ
  lea : ℝ

/-- Defines the conditions of the telescope purchase problem --/
def validPurchases (p : Purchases) : Prop :=
  p.joan + p.karl + p.lea = 600 ∧
  2 * p.joan = p.karl + 74 ∧
  p.lea - p.karl = 52

/-- Theorem stating that if the purchases satisfy the given conditions, 
    then Joan's payment is $139.20 --/
theorem joan_payment (p : Purchases) (h : validPurchases p) : 
  p.joan = 139.20 := by
  sorry

end NUMINAMATH_CALUDE_joan_payment_l27_2784


namespace NUMINAMATH_CALUDE_m_geq_n_l27_2756

theorem m_geq_n (a b : ℝ) : 
  let M := a^2 + 12*a - 4*b
  let N := 4*a - 20 - b^2
  M ≥ N := by
sorry

end NUMINAMATH_CALUDE_m_geq_n_l27_2756


namespace NUMINAMATH_CALUDE_july_production_l27_2755

/-- Calculates the mask production after a given number of months, 
    starting from an initial production and doubling each month. -/
def maskProduction (initialProduction : ℕ) (months : ℕ) : ℕ :=
  initialProduction * 2^months

/-- Theorem stating that the mask production in July (4 months after March) 
    is 48000, given an initial production of 3000 in March. -/
theorem july_production : maskProduction 3000 4 = 48000 := by
  sorry

end NUMINAMATH_CALUDE_july_production_l27_2755


namespace NUMINAMATH_CALUDE_no_solution_double_inequality_l27_2772

theorem no_solution_double_inequality :
  ¬∃ x : ℝ, (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_double_inequality_l27_2772


namespace NUMINAMATH_CALUDE_figure_to_square_partition_l27_2766

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a planar figure on a grid --/
def PlanarFigure := Set GridPoint

/-- Represents a transformation that can be applied to a set of points --/
structure Transformation where
  rotate : ℤ → GridPoint → GridPoint
  translate : ℤ → ℤ → GridPoint → GridPoint

/-- Checks if a set of points forms a square --/
def is_square (s : Set GridPoint) : Prop := sorry

/-- The main theorem --/
theorem figure_to_square_partition 
  (F : PlanarFigure) 
  (G : Set GridPoint) -- The grid
  (T : Transformation) -- Available transformations
  : 
  ∃ (S1 S2 S3 : Set GridPoint),
    (S1 ∪ S2 ∪ S3 = F) ∧ 
    (S1 ∩ S2 ≠ ∅) ∧ 
    (S2 ∩ S3 ≠ ∅) ∧ 
    (S3 ∩ S1 ≠ ∅) ∧
    ∃ (S : Set GridPoint), 
      is_square S ∧ 
      ∃ (f1 f2 f3 : Set GridPoint → Set GridPoint),
        (∀ p ∈ S1, ∃ q, q = T.rotate r1 (T.translate dx1 dy1 p) ∧ f1 {p} = {q}) ∧
        (∀ p ∈ S2, ∃ q, q = T.rotate r2 (T.translate dx2 dy2 p) ∧ f2 {p} = {q}) ∧
        (∀ p ∈ S3, ∃ q, q = T.rotate r3 (T.translate dx3 dy3 p) ∧ f3 {p} = {q}) ∧
        (f1 S1 ∪ f2 S2 ∪ f3 S3 = S)
  := by sorry

end NUMINAMATH_CALUDE_figure_to_square_partition_l27_2766


namespace NUMINAMATH_CALUDE_max_good_sequences_75_each_l27_2791

/-- Represents a string of beads -/
structure BeadString :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Defines a "good" sequence of beads -/
def is_good_sequence (seq : List Char) : Bool :=
  seq.length = 5 ∧ 
  seq.count 'G' = 3 ∧ 
  seq.count 'R' = 1 ∧ 
  seq.count 'B' = 1

/-- Calculates the maximum number of "good" sequences in a bead string -/
def max_good_sequences (s : BeadString) : ℕ :=
  min (s.green * 5 / 3) (min s.red s.blue)

/-- Theorem stating the maximum number of "good" sequences for the given bead string -/
theorem max_good_sequences_75_each (s : BeadString) 
  (h1 : s.blue = 75) (h2 : s.red = 75) (h3 : s.green = 75) : 
  max_good_sequences s = 123 := by
  sorry

end NUMINAMATH_CALUDE_max_good_sequences_75_each_l27_2791


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l27_2746

theorem greatest_value_quadratic_inequality :
  ∃ (a : ℝ), a^2 - 10*a + 21 ≤ 0 ∧ ∀ (x : ℝ), x^2 - 10*x + 21 ≤ 0 → x ≤ a :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l27_2746


namespace NUMINAMATH_CALUDE_angle_B_measure_max_area_l27_2754

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

/-- The given condition a^2 + c^2 = b^2 - ac -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a^2 + t.c^2 = t.b^2 - t.a * t.c

theorem angle_B_measure (t : Triangle) (h : satisfiesCondition t) :
  t.B = 2 * π / 3 := by sorry

theorem max_area (t : Triangle) (h1 : satisfiesCondition t) (h2 : t.b = 2 * Real.sqrt 3) :
  (t.a * t.c * Real.sin t.B) / 2 ≤ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_B_measure_max_area_l27_2754


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l27_2747

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 / b1 = a2 / b2

/-- Line l₁: 2x + my - 7 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop :=
  2 * x + m * y - 7 = 0

/-- Line l₂: mx + 8y - 14 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop :=
  m * x + 8 * y - 14 = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (parallel_lines 2 m (-7) m 8 (-14)) → m = -4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l27_2747


namespace NUMINAMATH_CALUDE_marble_distribution_l27_2760

/-- Represents a distribution of marbles into bags -/
def Distribution := List Nat

/-- Checks if a distribution is valid for a given number of children -/
def isValidDistribution (d : Distribution) (numChildren : Nat) : Prop :=
  d.sum = 77 ∧ d.length ≥ numChildren ∧ (77 % numChildren = 0)

/-- The minimum number of bags needed -/
def minBags : Nat := 17

theorem marble_distribution :
  (∀ d : Distribution, d.length < minBags → ¬(isValidDistribution d 7 ∧ isValidDistribution d 11)) ∧
  (∃ d : Distribution, d.length = minBags ∧ isValidDistribution d 7 ∧ isValidDistribution d 11) :=
sorry

#check marble_distribution

end NUMINAMATH_CALUDE_marble_distribution_l27_2760


namespace NUMINAMATH_CALUDE_simon_blueberry_theorem_l27_2785

/-- The number of blueberries Simon picked from his own bushes -/
def own_blueberries : ℕ := 100

/-- The number of blueberries needed for each pie -/
def blueberries_per_pie : ℕ := 100

/-- The number of pies Simon can make -/
def number_of_pies : ℕ := 3

/-- The number of blueberries Simon picked from nearby bushes -/
def nearby_blueberries : ℕ := number_of_pies * blueberries_per_pie - own_blueberries

theorem simon_blueberry_theorem : nearby_blueberries = 200 := by
  sorry

end NUMINAMATH_CALUDE_simon_blueberry_theorem_l27_2785


namespace NUMINAMATH_CALUDE_certain_number_problem_l27_2783

theorem certain_number_problem (x : ℝ) : 
  (0.3 * x) - (1/3) * (0.3 * x) = 36 → x = 180 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l27_2783


namespace NUMINAMATH_CALUDE_sample_size_is_80_l27_2704

/-- Represents the ratio of products A, B, and C in production -/
def productionRatio : Fin 3 → ℕ
| 0 => 2  -- Product A
| 1 => 3  -- Product B
| 2 => 5  -- Product C

/-- The number of products of type B selected in the sample -/
def selectedB : ℕ := 24

/-- The total sample size -/
def n : ℕ := 80

/-- Theorem stating that the given conditions lead to a sample size of 80 -/
theorem sample_size_is_80 : 
  (productionRatio 1 : ℚ) / (productionRatio 0 + productionRatio 1 + productionRatio 2) = selectedB / n :=
sorry

end NUMINAMATH_CALUDE_sample_size_is_80_l27_2704


namespace NUMINAMATH_CALUDE_complex_simplification_l27_2703

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  7 * (4 - 2*i) - 2*i * (3 - 4*i) = 20 - 20*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l27_2703


namespace NUMINAMATH_CALUDE_third_month_sale_l27_2765

theorem third_month_sale
  (average : ℕ)
  (month1 month2 month4 month5 month6 : ℕ)
  (h1 : average = 6800)
  (h2 : month1 = 6435)
  (h3 : month2 = 6927)
  (h4 : month4 = 7230)
  (h5 : month5 = 6562)
  (h6 : month6 = 6791)
  : ∃ month3 : ℕ, 
    month3 = 6855 ∧ 
    (month1 + month2 + month3 + month4 + month5 + month6) / 6 = average :=
by sorry

end NUMINAMATH_CALUDE_third_month_sale_l27_2765


namespace NUMINAMATH_CALUDE_f_not_odd_nor_even_f_minimum_value_l27_2732

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + |x - 2| - 1

-- Theorem for the parity of f(x)
theorem f_not_odd_nor_even :
  ¬(∀ x, f x = f (-x)) ∧ ¬(∀ x, f x = -f (-x)) :=
sorry

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∀ x, f x ≥ 3 ∧ ∃ y, f y = 3 :=
sorry

end NUMINAMATH_CALUDE_f_not_odd_nor_even_f_minimum_value_l27_2732


namespace NUMINAMATH_CALUDE_car_can_climb_slope_l27_2769

theorem car_can_climb_slope (car_max_angle : Real) (slope_gradient : Real) : 
  car_max_angle = 60 * Real.pi / 180 →
  slope_gradient = 1.5 →
  Real.tan car_max_angle > slope_gradient := by
  sorry

end NUMINAMATH_CALUDE_car_can_climb_slope_l27_2769


namespace NUMINAMATH_CALUDE_max_revenue_at_50_10_l27_2743

/-- Represents the parking lot problem -/
structure ParkingLot where
  carSpace : ℝ
  busSpace : ℝ
  carFee : ℝ
  busFee : ℝ
  totalArea : ℝ
  maxVehicles : ℕ

/-- Revenue function for the parking lot -/
def revenue (p : ParkingLot) (x y : ℝ) : ℝ :=
  p.carFee * x + p.busFee * y

/-- Theorem stating that (50, 10) maximizes revenue for the given parking lot problem -/
theorem max_revenue_at_50_10 (p : ParkingLot)
  (h1 : p.carSpace = 6)
  (h2 : p.busSpace = 30)
  (h3 : p.carFee = 2.5)
  (h4 : p.busFee = 7.5)
  (h5 : p.totalArea = 600)
  (h6 : p.maxVehicles = 60) :
  ∀ x y : ℝ,
  x ≥ 0 → y ≥ 0 →
  x + y ≤ p.maxVehicles →
  p.carSpace * x + p.busSpace * y ≤ p.totalArea →
  revenue p x y ≤ revenue p 50 10 := by
  sorry


end NUMINAMATH_CALUDE_max_revenue_at_50_10_l27_2743


namespace NUMINAMATH_CALUDE_work_completion_time_l27_2761

theorem work_completion_time 
  (a_time b_time c_time : ℝ) 
  (ha : a_time = 8) 
  (hb : b_time = 12) 
  (hc : c_time = 24) : 
  1 / (1 / a_time + 1 / b_time + 1 / c_time) = 4 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l27_2761


namespace NUMINAMATH_CALUDE_father_son_age_difference_l27_2701

theorem father_son_age_difference : ∀ (father_age son_age : ℕ),
  son_age = 33 →
  father_age + 2 = 2 * (son_age + 2) →
  father_age - son_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l27_2701


namespace NUMINAMATH_CALUDE_p_plus_q_value_l27_2782

theorem p_plus_q_value (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 25*p - 75 = 0) 
  (hq : 10*q^3 - 75*q^2 - 365*q + 3375 = 0) : 
  p + q = 39/4 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_value_l27_2782


namespace NUMINAMATH_CALUDE_subset_intersection_bound_l27_2793

theorem subset_intersection_bound (m n k : ℕ) (F : Fin k → Finset (Fin m)) :
  m ≥ n →
  n > 1 →
  (∀ i, (F i).card = n) →
  (∀ i j, i < j → (F i ∩ F j).card ≤ 1) →
  k ≤ m * (m - 1) / (n * (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_subset_intersection_bound_l27_2793


namespace NUMINAMATH_CALUDE_initial_amount_theorem_l27_2736

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def num_posters : ℕ := 2
def num_notebooks : ℕ := 3
def num_bookmarks : ℕ := 2

def leftover : ℕ := 14

def total_cost : ℕ := num_posters * poster_cost + num_notebooks * notebook_cost + num_bookmarks * bookmark_cost

theorem initial_amount_theorem : total_cost + leftover = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_theorem_l27_2736


namespace NUMINAMATH_CALUDE_function_relationship_l27_2779

def main (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 - x) = f x) ∧
  (∀ x, f (x + 2) = f (x - 2)) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0) →
  f 2016 = f 2014 ∧ f 2014 > f 2015

theorem function_relationship : main f :=
sorry

end NUMINAMATH_CALUDE_function_relationship_l27_2779


namespace NUMINAMATH_CALUDE_pattern_proof_l27_2722

theorem pattern_proof (n : ℕ) (h : n > 0) : 
  Real.sqrt (n - n / (n^2 + 1)) = n * Real.sqrt (n / (n^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l27_2722


namespace NUMINAMATH_CALUDE_small_cube_side_length_l27_2724

theorem small_cube_side_length (large_cube_side : ℝ) (num_small_cubes : ℕ) (small_cube_side : ℝ) :
  large_cube_side = 1 →
  num_small_cubes = 1000 →
  large_cube_side ^ 3 = num_small_cubes * small_cube_side ^ 3 →
  small_cube_side = 0.1 := by
sorry

end NUMINAMATH_CALUDE_small_cube_side_length_l27_2724


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l27_2790

theorem boys_neither_happy_nor_sad (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ)
  (total_boys : ℕ) (total_girls : ℕ) (happy_boys : ℕ) (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : total_boys = 22)
  (h5 : total_girls = 38)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : total_children = total_boys + total_girls)
  (h9 : sad_children ≥ sad_girls) :
  total_boys - happy_boys - (sad_children - sad_girls) = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l27_2790


namespace NUMINAMATH_CALUDE_taxi_charge_calculation_l27_2773

/-- Taxi service charge calculation -/
theorem taxi_charge_calculation 
  (initial_fee : ℝ) 
  (total_charge : ℝ) 
  (trip_distance : ℝ) 
  (segment_length : ℝ) : 
  initial_fee = 2.25 →
  total_charge = 4.5 →
  trip_distance = 3.6 →
  segment_length = 2/5 →
  (total_charge - initial_fee) / (trip_distance / segment_length) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_calculation_l27_2773


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l27_2768

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 15 45 = 59 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l27_2768


namespace NUMINAMATH_CALUDE_two_planes_division_l27_2781

/-- Represents the possible configurations of two planes in 3D space -/
inductive PlaneConfiguration
  | Parallel
  | Intersecting

/-- Represents the number of parts that two planes divide the space into -/
def spaceDivisions (config : PlaneConfiguration) : Nat :=
  match config with
  | PlaneConfiguration.Parallel => 3
  | PlaneConfiguration.Intersecting => 4

/-- Theorem stating that two planes divide space into either 3 or 4 parts -/
theorem two_planes_division :
  ∀ (config : PlaneConfiguration), 
    (spaceDivisions config = 3 ∨ spaceDivisions config = 4) :=
by sorry

end NUMINAMATH_CALUDE_two_planes_division_l27_2781


namespace NUMINAMATH_CALUDE_interest_rate_is_30_percent_l27_2716

-- Define the compound interest function
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

-- State the theorem
theorem interest_rate_is_30_percent 
  (P : ℝ) 
  (h1 : compound_interest P r 2 = 17640) 
  (h2 : compound_interest P r 3 = 22932) : 
  r = 0.3 := by
  sorry


end NUMINAMATH_CALUDE_interest_rate_is_30_percent_l27_2716


namespace NUMINAMATH_CALUDE_problem_statement_l27_2750

theorem problem_statement (a b : ℝ) (ha : a > 0) (hcond : Real.exp a + Real.log b = 1) :
  (a + Real.log b < 0) ∧ (Real.exp a + b > 2) ∧ (a + b > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l27_2750


namespace NUMINAMATH_CALUDE_area_equality_function_unique_l27_2786

/-- A function satisfying the given area equality property -/
def AreaEqualityFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ * f x₂ = (x₂ - x₁) * (f x₁ + f x₂)

theorem area_equality_function_unique
  (f : ℝ → ℝ)
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : AreaEqualityFunction f)
  (h₃ : f 1 = 4) :
  (∀ x, x > 0 → f x = 4 / x) ∧ f 4 = 1 := by
sorry

end NUMINAMATH_CALUDE_area_equality_function_unique_l27_2786


namespace NUMINAMATH_CALUDE_medication_C_consumption_l27_2726

def days_in_july : ℕ := 31

def doses_per_day_C : ℕ := 3

def missed_days_C : ℕ := 2

theorem medication_C_consumption :
  days_in_july * doses_per_day_C - missed_days_C * doses_per_day_C = 87 := by
  sorry

end NUMINAMATH_CALUDE_medication_C_consumption_l27_2726


namespace NUMINAMATH_CALUDE_average_weight_decrease_l27_2711

theorem average_weight_decrease (initial_average : ℝ) : 
  let initial_total : ℝ := 8 * initial_average
  let new_total : ℝ := initial_total - 86 + 46
  let new_average : ℝ := new_total / 8
  initial_average - new_average = 5 := by sorry

end NUMINAMATH_CALUDE_average_weight_decrease_l27_2711


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l27_2710

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a + a^2 > b + b^2) ∧
  (∃ a b, a + a^2 > b + b^2 ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l27_2710


namespace NUMINAMATH_CALUDE_class_size_is_50_l27_2751

def original_average : ℝ := 87.26
def incorrect_score : ℝ := 89
def correct_score : ℝ := 98
def new_average : ℝ := 87.44

theorem class_size_is_50 : 
  ∃ n : ℕ, n > 0 ∧ 
  (n : ℝ) * new_average = (n : ℝ) * original_average + (correct_score - incorrect_score) ∧
  n = 50 :=
sorry

end NUMINAMATH_CALUDE_class_size_is_50_l27_2751


namespace NUMINAMATH_CALUDE_lanas_nickels_l27_2745

theorem lanas_nickels (num_stacks : ℕ) (nickels_per_stack : ℕ) 
  (h1 : num_stacks = 9) (h2 : nickels_per_stack = 8) : 
  num_stacks * nickels_per_stack = 72 := by
  sorry

end NUMINAMATH_CALUDE_lanas_nickels_l27_2745


namespace NUMINAMATH_CALUDE_equilateral_triangle_figure_divisible_l27_2713

/-- A figure composed of equilateral triangles -/
structure EquilateralTriangleFigure where
  /-- The set of points in the figure -/
  points : Set ℝ × ℝ
  /-- Predicate asserting that the figure is composed of equal equilateral triangles -/
  is_composed_of_equilateral_triangles : Prop

/-- A straight line in 2D space -/
structure Line where
  /-- Slope of the line -/
  slope : ℝ
  /-- Y-intercept of the line -/
  intercept : ℝ

/-- Predicate asserting that a line divides a figure into two congruent parts -/
def divides_into_congruent_parts (f : EquilateralTriangleFigure) (l : Line) : Prop :=
  sorry

/-- Theorem stating that any figure composed of equal equilateral triangles
    can be divided into two congruent parts by a straight line -/
theorem equilateral_triangle_figure_divisible (f : EquilateralTriangleFigure) :
  ∃ l : Line, divides_into_congruent_parts f l :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_figure_divisible_l27_2713


namespace NUMINAMATH_CALUDE_bhanu_petrol_expenditure_l27_2780

theorem bhanu_petrol_expenditure (income : ℝ) 
  (h1 : income > 0)
  (h2 : 0.14 * (income - 0.3 * income) = 98) : 
  0.3 * income = 300 := by
  sorry

end NUMINAMATH_CALUDE_bhanu_petrol_expenditure_l27_2780


namespace NUMINAMATH_CALUDE_ellipse_inscribed_circle_max_area_l27_2759

/-- The ellipse with equation x²/4 + y²/3 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The left focus of the ellipse -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F₂ : ℝ × ℝ := (1, 0)

/-- A line passing through F₂ -/
def line_through_F₂ (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

/-- The area of the inscribed circle in triangle F₁MN -/
def inscribed_circle_area (m : ℝ) : ℝ := sorry

theorem ellipse_inscribed_circle_max_area :
  ∃ (max_area : ℝ),
    (∀ m : ℝ, inscribed_circle_area m ≤ max_area) ∧
    (max_area = 9 * Real.pi / 16) ∧
    (∀ m : ℝ, inscribed_circle_area m = max_area ↔ m = 0) :=
  sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_circle_max_area_l27_2759


namespace NUMINAMATH_CALUDE_no_linear_term_condition_l27_2715

theorem no_linear_term_condition (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x - 4) = a * x^2 + b) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_condition_l27_2715


namespace NUMINAMATH_CALUDE_recreation_spending_ratio_l27_2725

theorem recreation_spending_ratio : 
  ∀ (last_week_wages : ℝ),
  last_week_wages > 0 →
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.80 * last_week_wages
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_recreation_spending_ratio_l27_2725


namespace NUMINAMATH_CALUDE_intersection_equality_iff_t_range_l27_2789

/-- The set M -/
def M : Set ℝ := {x | -2 < x ∧ x < 5}

/-- The set N parameterized by t -/
def N (t : ℝ) : Set ℝ := {x | 2 - t < x ∧ x < 2 * t + 1}

/-- Theorem stating the equivalence between M ∩ N = N and t ∈ (-∞, 2] -/
theorem intersection_equality_iff_t_range :
  ∀ t : ℝ, (M ∩ N t = N t) ↔ t ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_equality_iff_t_range_l27_2789


namespace NUMINAMATH_CALUDE_no_real_solution_l27_2741

theorem no_real_solution :
  ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l27_2741


namespace NUMINAMATH_CALUDE_seungchan_book_pages_l27_2705

/-- The number of pages in Seungchan's children's book -/
def total_pages : ℝ := 250

/-- The fraction of the book Seungchan read until yesterday -/
def read_yesterday : ℝ := 0.2

/-- The fraction of the remaining part Seungchan read today -/
def read_today : ℝ := 0.35

/-- The number of pages left after today's reading -/
def pages_left : ℝ := 130

theorem seungchan_book_pages :
  (1 - read_yesterday) * (1 - read_today) * total_pages = pages_left :=
sorry

end NUMINAMATH_CALUDE_seungchan_book_pages_l27_2705


namespace NUMINAMATH_CALUDE_gnome_count_after_removal_l27_2770

/-- The number of gnomes in each forest and the total remaining after removal --/
theorem gnome_count_after_removal :
  let westerville : ℕ := 20
  let ravenswood : ℕ := 4 * westerville
  let greenwood : ℕ := ravenswood + ravenswood / 4
  let remaining_westerville : ℕ := westerville - westerville * 3 / 10
  let remaining_ravenswood : ℕ := ravenswood - ravenswood * 2 / 5
  let remaining_greenwood : ℕ := greenwood - greenwood / 2
  remaining_westerville + remaining_ravenswood + remaining_greenwood = 112 := by
sorry

end NUMINAMATH_CALUDE_gnome_count_after_removal_l27_2770


namespace NUMINAMATH_CALUDE_factoring_expression_l27_2737

theorem factoring_expression (x y : ℝ) :
  5 * x * (x + 4) + 2 * (x + 4) * (y + 2) = (x + 4) * (5 * x + 2 * y + 4) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l27_2737


namespace NUMINAMATH_CALUDE_dot_product_sum_equilateral_triangle_l27_2709

-- Define the equilateral triangle
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C A = 1)

-- Define vectors a, b, c
def a (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def b (A C : ℝ × ℝ) : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def c (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_sum_equilateral_triangle (A B C : ℝ × ℝ) 
  (h : EquilateralTriangle A B C) : 
  dot_product (a B C) (b A C) + dot_product (b A C) (c A B) + dot_product (c A B) (a B C) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_equilateral_triangle_l27_2709


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l27_2731

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 19 ↔ 
  (x = 20 ∧ y = 380) ∨ (x = 380 ∧ y = 20) ∨ 
  (x = 18 ∧ y = -342) ∨ (x = -342 ∧ y = 18) ∨ 
  (x = 38 ∧ y = 38) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l27_2731


namespace NUMINAMATH_CALUDE_strawberry_distribution_l27_2794

theorem strawberry_distribution (num_girls : ℕ) (strawberries_per_girl : ℕ) 
  (h1 : num_girls = 8) (h2 : strawberries_per_girl = 6) :
  num_girls * strawberries_per_girl = 48 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_distribution_l27_2794


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l27_2702

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 4 = 3 →
  a 5 = 7 →
  a 6 = 11 →
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l27_2702


namespace NUMINAMATH_CALUDE_emily_subtraction_l27_2775

theorem emily_subtraction : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by sorry

end NUMINAMATH_CALUDE_emily_subtraction_l27_2775


namespace NUMINAMATH_CALUDE_two_solutions_equation_l27_2767

/-- The value of a 2x2 matrix [[a, c], [d, b]] is defined as ab - cd -/
def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

/-- The equation 2x^2 - x = 3 has exactly two real solutions -/
theorem two_solutions_equation :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ matrix_value (2*x) x 1 x = 3 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_equation_l27_2767


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l27_2763

def is_valid_ratio (a r : ℕ+) : Prop :=
  a * r^2 + a * r^4 + a * r^6 = 819 * 6^2016

theorem geometric_progression_ratio :
  ∃ (a : ℕ+), is_valid_ratio a 1 ∧ is_valid_ratio a 2 ∧ is_valid_ratio a 3 ∧ is_valid_ratio a 4 ∧
  ∀ (r : ℕ+), r ≠ 1 ∧ r ≠ 2 ∧ r ≠ 3 ∧ r ≠ 4 → ¬(∃ (b : ℕ+), is_valid_ratio b r) :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l27_2763


namespace NUMINAMATH_CALUDE_trig_inequality_l27_2740

theorem trig_inequality : ∀ a b c : ℝ,
  a = Real.sin (21 * π / 180) →
  b = Real.cos (72 * π / 180) →
  c = Real.tan (23 * π / 180) →
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l27_2740


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l27_2778

/-- Definition of the sum function for the arithmetic progression -/
def S (n : ℕ) : ℝ := 4 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 10 * r - 1

/-- Theorem stating that a(r) is the rth term of the arithmetic progression -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  a r = S r - S (r - 1) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l27_2778


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l27_2762

theorem negation_of_all_squares_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, ¬(n^2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l27_2762


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l27_2734

/-- A complex number z is in the second quadrant if its real part is negative and its imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- Given that (2a+2i)/(1+i) is purely imaginary for some real a, 
    prove that 2a+2i is in the second quadrant -/
theorem complex_in_second_quadrant (a : ℝ) 
    (h : (Complex.I : ℂ).re * ((2*a + 2*Complex.I) / (1 + Complex.I)).im = 
         (Complex.I : ℂ).im * ((2*a + 2*Complex.I) / (1 + Complex.I)).re) : 
    in_second_quadrant (2*a + 2*Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_complex_in_second_quadrant_l27_2734


namespace NUMINAMATH_CALUDE_open_box_volume_l27_2799

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume (sheet_length sheet_width cut_size : ℝ) 
  (h1 : sheet_length = 100)
  (h2 : sheet_width = 50)
  (h3 : cut_size = 10) : 
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 24000 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l27_2799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l27_2748

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 80 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h1 : ArithmeticSequence a)
  (h2 : SumCondition a) :
  a 5 + (1/4) * a 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l27_2748


namespace NUMINAMATH_CALUDE_complex_power_equality_l27_2788

theorem complex_power_equality : (3 * Complex.cos (π / 6) + 3 * Complex.I * Complex.sin (π / 6)) ^ 4 = -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equality_l27_2788


namespace NUMINAMATH_CALUDE_relationship_abc_l27_2708

theorem relationship_abc : ∀ (a b c : ℝ), 
  a = Real.sqrt 0.5 → 
  b = 2^(0.5 : ℝ) → 
  c = 0.5^(0.2 : ℝ) → 
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l27_2708


namespace NUMINAMATH_CALUDE_dan_final_marbles_l27_2738

/-- The number of marbles Dan has after giving some away and receiving more. -/
def final_marbles (initial : ℕ) (given_mary : ℕ) (given_peter : ℕ) (received : ℕ) : ℕ :=
  initial - given_mary - given_peter + received

/-- Theorem stating that Dan has 98 marbles at the end. -/
theorem dan_final_marbles :
  final_marbles 128 24 16 10 = 98 := by
  sorry

end NUMINAMATH_CALUDE_dan_final_marbles_l27_2738


namespace NUMINAMATH_CALUDE_alloy_chromium_percentage_l27_2727

/-- The percentage of chromium in an alloy mixture -/
def chromium_percentage (m1 m2 p1 p2 p3 : ℝ) : Prop :=
  m1 * p1 / 100 + m2 * p2 / 100 = (m1 + m2) * p3 / 100

/-- The problem statement -/
theorem alloy_chromium_percentage :
  ∃ (x : ℝ),
    chromium_percentage 15 30 12 x 9.333333333333334 ∧
    x = 8 := by sorry

end NUMINAMATH_CALUDE_alloy_chromium_percentage_l27_2727


namespace NUMINAMATH_CALUDE_original_total_is_390_l27_2752

/-- Represents the number of movies in each format --/
structure MovieCollection where
  dvd : ℕ
  bluray : ℕ
  digital : ℕ

/-- The original collection of movies --/
def original : MovieCollection := sorry

/-- The updated collection after purchasing new movies --/
def updated : MovieCollection := sorry

/-- The ratio of the original collection --/
def original_ratio : MovieCollection := ⟨7, 2, 1⟩

/-- The ratio of the updated collection --/
def updated_ratio : MovieCollection := ⟨13, 4, 2⟩

/-- The number of new Blu-ray movies purchased --/
def new_bluray : ℕ := 5

/-- The number of new digital movies purchased --/
def new_digital : ℕ := 3

theorem original_total_is_390 :
  ∃ (x : ℕ),
    original.dvd = 7 * x ∧
    original.bluray = 2 * x ∧
    original.digital = x ∧
    updated.dvd = original.dvd ∧
    updated.bluray = original.bluray + new_bluray ∧
    updated.digital = original.digital + new_digital ∧
    (updated.dvd : ℚ) / updated_ratio.dvd = (updated.bluray : ℚ) / updated_ratio.bluray ∧
    (updated.dvd : ℚ) / updated_ratio.dvd = (updated.digital : ℚ) / updated_ratio.digital ∧
    original.dvd + original.bluray + original.digital = 390 :=
by
  sorry

end NUMINAMATH_CALUDE_original_total_is_390_l27_2752


namespace NUMINAMATH_CALUDE_greatest_x_value_l27_2742

theorem greatest_x_value (x : ℤ) (h : (2.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 21000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (2.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 21000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l27_2742


namespace NUMINAMATH_CALUDE_square_root_of_two_l27_2757

theorem square_root_of_two :
  ∀ x : ℝ, x^2 = 2 ↔ x = Real.sqrt 2 ∨ x = -Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_two_l27_2757


namespace NUMINAMATH_CALUDE_arithmetic_geometric_seq_l27_2777

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℤ) : Prop :=
  (a 3) ^ 2 = (a 1) * (a 4)

theorem arithmetic_geometric_seq (a : ℕ → ℤ) 
  (h1 : arithmetic_seq a) 
  (h2 : geometric_subseq a) : 
  a 2 = -9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_seq_l27_2777


namespace NUMINAMATH_CALUDE_reaction_stoichiometry_l27_2758

-- Define the chemical species
def CaO : Type := Unit
def H2O : Type := Unit
def Ca_OH_2 : Type := Unit

-- Define the reaction
def reaction (cao : CaO) (h2o : H2O) : Ca_OH_2 := sorry

-- Define the number of moles
def moles : Type → ℝ := sorry

-- Theorem statement
theorem reaction_stoichiometry :
  ∀ (cao : CaO) (h2o : H2O),
    moles CaO = 1 →
    moles Ca_OH_2 = 1 →
    moles H2O = 1 :=
by sorry

end NUMINAMATH_CALUDE_reaction_stoichiometry_l27_2758


namespace NUMINAMATH_CALUDE_chess_tournament_games_l27_2707

theorem chess_tournament_games (n : ℕ) (h : n = 9) : 
  (n * (n - 1)) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l27_2707


namespace NUMINAMATH_CALUDE_gcd_221_195_l27_2749

theorem gcd_221_195 : Nat.gcd 221 195 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_221_195_l27_2749


namespace NUMINAMATH_CALUDE_triangle_area_l27_2753

/-- The area of a triangle ABC with given side lengths and angle -/
theorem triangle_area (a b c : ℝ) (θ : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : θ = 2 * Real.pi / 3) :
  let area := (1/2) * a * b * Real.sin θ
  area = (3 * Real.sqrt 3) / 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l27_2753


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l27_2729

/-- The number of rows in Pascal's Triangle we're considering -/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1's in the first n rows of Pascal's Triangle -/
def ones_count (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of choosing a 1 from the first n rows of Pascal's Triangle -/
def probability_of_one (n : ℕ) : ℚ := ones_count n / total_elements n

theorem probability_of_one_in_20_rows : 
  probability_of_one n = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l27_2729


namespace NUMINAMATH_CALUDE_smartphone_price_problem_l27_2735

theorem smartphone_price_problem (store_a_price : ℝ) (store_a_discount : ℝ) (store_b_discount : ℝ) :
  store_a_price = 125 →
  store_a_discount = 0.08 →
  store_b_discount = 0.10 →
  store_a_price * (1 - store_a_discount) = store_b_price * (1 - store_b_discount) - 2 →
  store_b_price = 130 :=
by
  sorry

#check smartphone_price_problem

end NUMINAMATH_CALUDE_smartphone_price_problem_l27_2735


namespace NUMINAMATH_CALUDE_seashell_collection_l27_2774

/-- Calculates the total number of seashells after Leo gives away a quarter of his collection -/
theorem seashell_collection (henry paul total : ℕ) (h1 : henry = 11) (h2 : paul = 24) (h3 : total = 59) :
  let leo := total - henry - paul
  let leo_remaining := leo - (leo / 4)
  henry + paul + leo_remaining = 53 := by sorry

end NUMINAMATH_CALUDE_seashell_collection_l27_2774


namespace NUMINAMATH_CALUDE_car_rental_rates_equal_l27_2721

/-- The daily rate of Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate of Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The per-mile rate of the second company in dollars -/
def second_mile_rate : ℝ := 0.21

/-- The number of miles driven -/
def miles_driven : ℝ := 150

/-- The daily rate of the second company in dollars -/
def second_daily_rate : ℝ := 18.95

theorem car_rental_rates_equal :
  safety_daily_rate + safety_mile_rate * miles_driven =
  second_daily_rate + second_mile_rate * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rates_equal_l27_2721


namespace NUMINAMATH_CALUDE_complex_fourth_power_l27_2714

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fourth_power_l27_2714


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l27_2744

theorem largest_divisor_of_difference_of_squares (a b : ℤ) :
  let m : ℤ := 2*a + 3
  let n : ℤ := 2*b + 1
  (n < m) →
  (∃ k : ℤ, m^2 - n^2 = 4*k) ∧
  (∀ d : ℤ, d > 4 → ∃ a' b' : ℤ, 
    let m' : ℤ := 2*a' + 3
    let n' : ℤ := 2*b' + 1
    (n' < m') ∧ (m'^2 - n'^2) % d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_squares_l27_2744


namespace NUMINAMATH_CALUDE_ring_arrangements_l27_2795

theorem ring_arrangements (n k f : ℕ) (hn : n = 10) (hk : k = 7) (hf : f = 5) :
  (n.choose k) * k.factorial * ((k + f - 1).choose (f - 1)) = 200160000 :=
sorry

end NUMINAMATH_CALUDE_ring_arrangements_l27_2795


namespace NUMINAMATH_CALUDE_bush_height_after_two_years_l27_2739

/-- The height of a bush after a given number of years -/
def bush_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * 4^years

/-- Theorem stating the height of the bush after 2 years -/
theorem bush_height_after_two_years
  (h : bush_height (bush_height 1 0) 4 = 64) :
  bush_height (bush_height 1 0) 2 = 4 :=
by
  sorry

#check bush_height_after_two_years

end NUMINAMATH_CALUDE_bush_height_after_two_years_l27_2739


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l27_2712

theorem no_solution_implies_a_equals_one (a : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (a * x) / (x - 2) ≠ 4 / (x - 2) + 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l27_2712


namespace NUMINAMATH_CALUDE_probability_eight_distinct_rolls_l27_2718

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- The probability of rolling eight standard, eight-sided dice and getting eight distinct numbers -/
def probability_distinct_rolls : ℚ :=
  (Nat.factorial num_dice) / (num_sides ^ num_dice)

theorem probability_eight_distinct_rolls :
  probability_distinct_rolls = 5 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_probability_eight_distinct_rolls_l27_2718


namespace NUMINAMATH_CALUDE_managers_salary_l27_2733

def employee_count : ℕ := 24
def initial_average_salary : ℕ := 1500
def average_increase : ℕ := 400

theorem managers_salary (total_salary : ℕ) (managers_salary : ℕ) :
  total_salary = employee_count * initial_average_salary ∧
  (total_salary + managers_salary) / (employee_count + 1) = initial_average_salary + average_increase →
  managers_salary = 11500 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l27_2733


namespace NUMINAMATH_CALUDE_value_of_x_l27_2776

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l27_2776


namespace NUMINAMATH_CALUDE_strawberry_calories_is_4_l27_2728

/-- The number of strawberries Zoe ate -/
def num_strawberries : ℕ := 12

/-- The amount of yogurt Zoe ate in ounces -/
def yogurt_ounces : ℕ := 6

/-- The number of calories per ounce of yogurt -/
def yogurt_calories_per_ounce : ℕ := 17

/-- The total calories Zoe ate -/
def total_calories : ℕ := 150

/-- The number of calories in each strawberry -/
def strawberry_calories : ℕ := (total_calories - yogurt_ounces * yogurt_calories_per_ounce) / num_strawberries

theorem strawberry_calories_is_4 : strawberry_calories = 4 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_calories_is_4_l27_2728


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l27_2798

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_11th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_2 : a 2 = 3)
  (h_6 : a 6 = 7) :
  a 11 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l27_2798


namespace NUMINAMATH_CALUDE_square_side_length_l27_2706

/-- Given a square and a rectangle with specific properties, prove that the side length of the square is 15 cm. -/
theorem square_side_length (s : ℝ) : 
  s > 0 →  -- side length is positive
  4 * s = 2 * (18 + 216 / 18) →  -- perimeters are equal
  s = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l27_2706


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l27_2720

/-- Given that z = (1-mi)/(2+i) is a pure imaginary number, prove that m = 2 --/
theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 - m * Complex.I) / (2 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l27_2720


namespace NUMINAMATH_CALUDE_initial_water_percentage_is_70_l27_2700

/-- The percentage of liquid X in solution Y -/
def liquid_x_percentage : ℝ := 30

/-- The initial mass of solution Y in kg -/
def initial_mass : ℝ := 8

/-- The mass of water that evaporates in kg -/
def evaporated_water : ℝ := 3

/-- The mass of solution Y added after evaporation in kg -/
def added_solution : ℝ := 3

/-- The percentage of liquid X in the new solution -/
def new_liquid_x_percentage : ℝ := 41.25

/-- The initial percentage of water in solution Y -/
def initial_water_percentage : ℝ := 100 - liquid_x_percentage

theorem initial_water_percentage_is_70 :
  initial_water_percentage = 70 :=
sorry

end NUMINAMATH_CALUDE_initial_water_percentage_is_70_l27_2700


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l27_2719

/-- The height of a tree after n years, given its initial height and growth rate -/
def tree_height (initial_height : ℝ) (growth_rate : ℝ) (n : ℕ) : ℝ :=
  initial_height * growth_rate ^ n

/-- Theorem: A tree that triples its height every year and reaches 243 feet after 5 years
    will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years
  (h1 : ∃ initial_height : ℝ, tree_height initial_height 3 5 = 243)
  (h2 : ∀ n : ℕ, tree_height initial_height 3 (n + 1) = 3 * tree_height initial_height 3 n) :
  tree_height initial_height 3 2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l27_2719
