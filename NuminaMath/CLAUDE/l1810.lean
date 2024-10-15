import Mathlib

namespace NUMINAMATH_CALUDE_tan_equality_periodic_l1810_181051

theorem tan_equality_periodic (n : ℤ) : 
  -180 < n ∧ n < 180 → 
  Real.tan (n * π / 180) = Real.tan (1540 * π / 180) → 
  n = 40 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_periodic_l1810_181051


namespace NUMINAMATH_CALUDE_min_value_with_product_constraint_l1810_181075

theorem min_value_with_product_constraint (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (product_constraint : x * y * z = 32) : 
  x^2 + 4*x*y + 4*y^2 + 2*z^2 ≥ 68 ∧ 
  (x^2 + 4*x*y + 4*y^2 + 2*z^2 = 68 ↔ x = 4 ∧ y = 2 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_with_product_constraint_l1810_181075


namespace NUMINAMATH_CALUDE_parallelogram_area_l1810_181061

def v : Fin 2 → ℝ
| 0 => 7
| 1 => -5

def w : Fin 2 → ℝ
| 0 => 14
| 1 => -4

theorem parallelogram_area : 
  abs (v 0 * w 1 - v 1 * w 0) = 42 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1810_181061


namespace NUMINAMATH_CALUDE_angle_B_measure_max_perimeter_max_perimeter_achieved_l1810_181046

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.B + t.b * Real.cos t.C + t.c * Real.cos t.B = 0

-- Part I: Prove that angle B is 2π/3
theorem angle_B_measure (t : Triangle) (h : condition t) : t.B = 2 * Real.pi / 3 := by
  sorry

-- Part II: Prove the maximum perimeter
theorem max_perimeter (t : Triangle) (h : condition t) (hb : t.b = Real.sqrt 3) :
  t.a + t.b + t.c ≤ Real.sqrt 3 + 2 := by
  sorry

-- Prove that the maximum perimeter is achieved
theorem max_perimeter_achieved : ∃ (t : Triangle), condition t ∧ t.b = Real.sqrt 3 ∧ t.a + t.b + t.c = Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_max_perimeter_max_perimeter_achieved_l1810_181046


namespace NUMINAMATH_CALUDE_compound_composition_l1810_181089

def atomic_weight_N : ℕ := 14
def atomic_weight_H : ℕ := 1
def atomic_weight_Br : ℕ := 80
def molecular_weight : ℕ := 98

theorem compound_composition (n : ℕ) : 
  atomic_weight_N + n * atomic_weight_H + atomic_weight_Br = molecular_weight → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l1810_181089


namespace NUMINAMATH_CALUDE_probability_at_most_one_incorrect_l1810_181062

/-- The probability of at most one incorrect result in 10 hemoglobin tests -/
def prob_at_most_one_incorrect (p : ℝ) : ℝ :=
  p^9 * (10 - 9*p)

/-- Theorem: Given the accuracy of a hemoglobin test is p, 
    the probability of at most one incorrect result out of 10 tests 
    is equal to p^9 * (10 - 9p) -/
theorem probability_at_most_one_incorrect 
  (p : ℝ) 
  (h1 : 0 ≤ p) 
  (h2 : p ≤ 1) : 
  (p^10 + 10 * (1 - p) * p^9) = prob_at_most_one_incorrect p :=
sorry

end NUMINAMATH_CALUDE_probability_at_most_one_incorrect_l1810_181062


namespace NUMINAMATH_CALUDE_dartboard_sector_angle_l1810_181066

theorem dartboard_sector_angle (probability : ℝ) (angle : ℝ) : 
  probability = 1/4 → angle = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_dartboard_sector_angle_l1810_181066


namespace NUMINAMATH_CALUDE_truck_tunnel_time_l1810_181024

theorem truck_tunnel_time (truck_length : ℝ) (tunnel_length : ℝ) (speed_mph : ℝ) :
  truck_length = 66 →
  tunnel_length = 330 →
  speed_mph = 45 →
  let speed_fps := speed_mph * 5280 / 3600
  let total_distance := tunnel_length + truck_length
  let time := total_distance / speed_fps
  time = 6 := by sorry

end NUMINAMATH_CALUDE_truck_tunnel_time_l1810_181024


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1810_181060

theorem remainder_divisibility (n : ℤ) : n % 22 = 12 → (2 * n) % 22 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1810_181060


namespace NUMINAMATH_CALUDE_white_surface_fraction_of_given_cube_l1810_181083

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  -- Implementation details omitted
  0

/-- Theorem stating the fraction of white surface area for the given cube -/
theorem white_surface_fraction_of_given_cube :
  let c : CompositeCube := {
    edge_length := 4,
    small_cube_count := 64,
    white_cube_count := 44,
    black_cube_count := 20
  }
  white_surface_fraction c = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_white_surface_fraction_of_given_cube_l1810_181083


namespace NUMINAMATH_CALUDE_complex_division_problem_l1810_181034

theorem complex_division_problem (z : ℂ) (h : z = 4 + 3*I) : 
  Complex.abs z / z = 4/5 - 3/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_problem_l1810_181034


namespace NUMINAMATH_CALUDE_seventh_term_largest_implies_n_l1810_181056

/-- The binomial coefficient -/
def binomial_coefficient (n k : ℕ) : ℕ := sorry

/-- Predicate to check if the 7th term has the largest binomial coefficient -/
def seventh_term_largest (n : ℕ) : Prop :=
  ∀ k, k ≠ 6 → binomial_coefficient n 6 ≥ binomial_coefficient n k

/-- Theorem stating the possible values of n when the 7th term has the largest binomial coefficient -/
theorem seventh_term_largest_implies_n (n : ℕ) :
  seventh_term_largest n → n = 11 ∨ n = 12 ∨ n = 13 := by sorry

end NUMINAMATH_CALUDE_seventh_term_largest_implies_n_l1810_181056


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l1810_181007

theorem exam_maximum_marks :
  ∀ (passing_threshold : ℝ) (obtained_marks : ℝ) (failing_margin : ℝ),
    passing_threshold = 0.30 →
    obtained_marks = 30 →
    failing_margin = 36 →
    ∃ (max_marks : ℝ),
      max_marks = 220 ∧
      passing_threshold * max_marks = obtained_marks + failing_margin :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l1810_181007


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1810_181084

/-- The speed of a boat in still water, given its travel distances with and against a stream. -/
theorem boat_speed_in_still_water 
  (along_stream : ℝ) 
  (against_stream : ℝ) 
  (h1 : along_stream = 11) 
  (h2 : against_stream = 7) : 
  ∃ (boat_speed stream_speed : ℝ), 
    boat_speed + stream_speed = along_stream ∧ 
    boat_speed - stream_speed = against_stream ∧ 
    boat_speed = 9 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1810_181084


namespace NUMINAMATH_CALUDE_same_color_probability_l1810_181080

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 6

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls + blue_balls

/-- The probability of drawing two balls of the same color with replacement -/
theorem same_color_probability : 
  (green_balls^2 + red_balls^2 + blue_balls^2) / total_balls^2 = 101 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1810_181080


namespace NUMINAMATH_CALUDE_point_symmetry_l1810_181017

/-- Given a line l: 2x - y - 1 = 0 and two points A and A', 
    this theorem states that A' is symmetric to A about l. -/
theorem point_symmetry (x y : ℚ) : 
  let l := {(x, y) : ℚ × ℚ | 2 * x - y - 1 = 0}
  let A := (3, -2)
  let A' := (-13/5, 4/5)
  let midpoint := ((A'.1 + A.1) / 2, (A'.2 + A.2) / 2)
  (2 * midpoint.1 - midpoint.2 - 1 = 0) ∧ 
  ((A'.2 - A.2) / (A'.1 - A.1) * 2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_symmetry_l1810_181017


namespace NUMINAMATH_CALUDE_waste_fraction_for_park_l1810_181002

/-- A kite-shaped park with specific properties -/
structure KitePark where
  -- AB and BC lengths
  side_length : ℝ
  -- Ensure side_length is positive
  side_positive : side_length > 0

/-- The fraction of the park's area from which waste is brought to the longest diagonal -/
noncomputable def waste_fraction (park : KitePark) : ℝ :=
  7071 / 10000

/-- Theorem stating the waste fraction for a kite park with side length 100 -/
theorem waste_fraction_for_park (park : KitePark) 
  (h : park.side_length = 100) : 
  waste_fraction park = 7071 / 10000 :=
by sorry

end NUMINAMATH_CALUDE_waste_fraction_for_park_l1810_181002


namespace NUMINAMATH_CALUDE_johns_wife_notebooks_l1810_181055

/-- Proves the number of notebooks John's wife bought for each child --/
theorem johns_wife_notebooks (num_children : ℕ) (johns_notebooks_per_child : ℕ) (total_notebooks : ℕ) :
  num_children = 3 →
  johns_notebooks_per_child = 2 →
  total_notebooks = 21 →
  (total_notebooks - num_children * johns_notebooks_per_child) / num_children = 5 := by
sorry

end NUMINAMATH_CALUDE_johns_wife_notebooks_l1810_181055


namespace NUMINAMATH_CALUDE_difference_of_squares_75_25_l1810_181093

theorem difference_of_squares_75_25 : 75^2 - 25^2 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_75_25_l1810_181093


namespace NUMINAMATH_CALUDE_no_integer_roots_for_odd_coeff_quadratic_l1810_181006

/-- A quadratic function with odd coefficients has no integer roots -/
theorem no_integer_roots_for_odd_coeff_quadratic (a b c : ℤ) (ha : a ≠ 0) 
  (hodd : Odd a ∧ Odd b ∧ Odd c) :
  ¬∃ x : ℤ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_for_odd_coeff_quadratic_l1810_181006


namespace NUMINAMATH_CALUDE_certain_number_subtraction_l1810_181082

theorem certain_number_subtraction (x : ℤ) : 
  (3005 - x + 10 = 2705) → (x = 310) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_subtraction_l1810_181082


namespace NUMINAMATH_CALUDE_min_words_to_pass_l1810_181008

-- Define the exam parameters
def total_words : ℕ := 800
def passing_score : ℚ := 90 / 100
def guess_rate : ℚ := 10 / 100

-- Define the function to calculate the score based on words learned
def exam_score (words_learned : ℕ) : ℚ :=
  (words_learned : ℚ) / total_words + 
  guess_rate * ((total_words - words_learned) : ℚ) / total_words

-- Theorem statement
theorem min_words_to_pass : 
  ∀ n : ℕ, n < 712 → exam_score n < passing_score ∧ 
  exam_score 712 ≥ passing_score := by sorry

end NUMINAMATH_CALUDE_min_words_to_pass_l1810_181008


namespace NUMINAMATH_CALUDE_find_b_l1810_181015

theorem find_b (a b c : ℕ) 
  (h1 : 1 < a) (h2 : a < b) (h3 : b < c)
  (h4 : a + b + c = 111)
  (h5 : b^2 = a * c) :
  b = 36 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l1810_181015


namespace NUMINAMATH_CALUDE_article_cost_l1810_181033

/-- The cost of an article given specific profit conditions -/
theorem article_cost (C : ℝ) (S : ℝ) : 
  S = 1.25 * C → -- Original selling price (25% profit)
  (0.8 * C + 0.3 * (0.8 * C) = S - 6.3) → -- New cost and selling price with 30% profit
  C = 30 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l1810_181033


namespace NUMINAMATH_CALUDE_run_time_around_square_field_l1810_181021

/-- Calculates the time taken for a boy to run around a square field -/
theorem run_time_around_square_field (side_length : ℝ) (speed_kmh : ℝ) : 
  side_length = 60 → speed_kmh = 9 → 
  (4 * side_length) / (speed_kmh * 1000 / 3600) = 96 := by
  sorry

#check run_time_around_square_field

end NUMINAMATH_CALUDE_run_time_around_square_field_l1810_181021


namespace NUMINAMATH_CALUDE_shop_annual_rent_per_square_foot_l1810_181077

/-- Calculates the annual rent per square foot of a shop -/
theorem shop_annual_rent_per_square_foot
  (length : ℝ)
  (width : ℝ)
  (monthly_rent : ℝ)
  (h1 : length = 10)
  (h2 : width = 8)
  (h3 : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := by
  sorry

end NUMINAMATH_CALUDE_shop_annual_rent_per_square_foot_l1810_181077


namespace NUMINAMATH_CALUDE_tomato_basket_price_l1810_181090

-- Define the given values
def strawberry_plants : ℕ := 5
def tomato_plants : ℕ := 7
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def strawberry_basket_price : ℕ := 9
def total_revenue : ℕ := 186

-- Calculate total strawberries and tomatoes
def total_strawberries : ℕ := strawberry_plants * strawberries_per_plant
def total_tomatoes : ℕ := tomato_plants * tomatoes_per_plant

-- Calculate number of baskets
def strawberry_baskets : ℕ := total_strawberries / fruits_per_basket
def tomato_baskets : ℕ := total_tomatoes / fruits_per_basket

-- Define the theorem
theorem tomato_basket_price :
  (total_revenue - strawberry_baskets * strawberry_basket_price) / tomato_baskets = 6 :=
by sorry

end NUMINAMATH_CALUDE_tomato_basket_price_l1810_181090


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_digit_numbers_l1810_181081

def NumberSet : Finset Nat := {5, 6, 7, 8, 9}

def is_valid_pair (a b : Nat) : Prop :=
  a ∈ NumberSet ∧ b ∈ NumberSet ∧ a ≠ b ∧ a ≥ 10 ∧ a < 100 ∧ b ≥ 10 ∧ b < 100

def sum_of_pair (a b : Nat) : Nat := a + b

theorem smallest_sum_of_two_digit_numbers :
  ∃ (a b : Nat), is_valid_pair a b ∧
    sum_of_pair a b = 125 ∧
    (∀ (c d : Nat), is_valid_pair c d → sum_of_pair c d ≥ 125) :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_digit_numbers_l1810_181081


namespace NUMINAMATH_CALUDE_original_price_sum_l1810_181099

/-- The original price of all items before price increases -/
def original_total_price (candy_box soda_can chips_bag chocolate_bar : ℝ) : ℝ :=
  candy_box + soda_can + chips_bag + chocolate_bar

/-- Theorem stating that the original total price is 22 pounds -/
theorem original_price_sum :
  original_total_price 10 6 4 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_original_price_sum_l1810_181099


namespace NUMINAMATH_CALUDE_robins_initial_gum_pieces_robins_initial_gum_pieces_proof_l1810_181022

/-- Given that Robin now has 62 pieces of gum after receiving 44.0 pieces from her brother,
    prove that her initial number of gum pieces was 18. -/
theorem robins_initial_gum_pieces : ℝ → Prop :=
  fun initial_gum =>
    initial_gum + 44.0 = 62 →
    initial_gum = 18
    
/-- Proof of the theorem -/
theorem robins_initial_gum_pieces_proof : robins_initial_gum_pieces 18 := by
  sorry

end NUMINAMATH_CALUDE_robins_initial_gum_pieces_robins_initial_gum_pieces_proof_l1810_181022


namespace NUMINAMATH_CALUDE_product_no_x_squared_term_l1810_181027

theorem product_no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x^2 - 2*a*x + a^2) = x^3 + (a^2 - 2*a)*x + a^2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_no_x_squared_term_l1810_181027


namespace NUMINAMATH_CALUDE_exactly_one_true_l1810_181012

def X : Set Int := {x | -2 < x ∧ x ≤ 3}

def p (a : ℝ) : Prop := ∀ x ∈ X, (1/3 : ℝ) * x^2 < 2*a - 3

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a = 0

theorem exactly_one_true (a : ℝ) : 
  (p a ∧ ¬(q a)) ∨ (¬(p a) ∧ q a) ↔ a ≤ 1 ∨ a > 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l1810_181012


namespace NUMINAMATH_CALUDE_dog_park_theorem_l1810_181011

/-- The total number of dogs barking after a new group joins -/
def total_dogs (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial + multiplier * initial

/-- Theorem: Given 30 initial dogs and a new group triple the size, the total is 120 dogs -/
theorem dog_park_theorem :
  total_dogs 30 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_theorem_l1810_181011


namespace NUMINAMATH_CALUDE_equation_equiv_lines_l1810_181070

/-- The set of points satisfying the equation 2x^2 + y^2 + 3xy + 3x + y = 2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1^2 + p.2^2 + 3 * p.1 * p.2 + 3 * p.1 + p.2 = 2}

/-- The set of points on the line y = -x - 2 -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1 - 2}

/-- The set of points on the line y = -2x + 1 -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -2 * p.1 + 1}

/-- Theorem stating that S is equivalent to the union of L1 and L2 -/
theorem equation_equiv_lines : S = L1 ∪ L2 := by
  sorry

end NUMINAMATH_CALUDE_equation_equiv_lines_l1810_181070


namespace NUMINAMATH_CALUDE_kamal_average_marks_l1810_181042

/-- Calculates the average marks given a list of obtained marks and total marks -/
def averageMarks (obtained : List ℕ) (total : List ℕ) : ℚ :=
  (obtained.sum : ℚ) / (total.sum : ℚ) * 100

theorem kamal_average_marks : 
  let obtained := [76, 60, 82, 67, 85, 78]
  let total := [120, 110, 100, 90, 100, 95]
  averageMarks obtained total = 448 / 615 * 100 := by
  sorry

#eval (448 : ℚ) / 615 * 100

end NUMINAMATH_CALUDE_kamal_average_marks_l1810_181042


namespace NUMINAMATH_CALUDE_function_extrema_implies_interval_bounds_l1810_181004

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the theorem
theorem function_extrema_implies_interval_bounds
  (a : ℝ)
  (h_nonneg : 0 ≤ a)
  (h_max : ∀ x ∈ Set.Icc 0 a, f x ≤ 3)
  (h_min : ∀ x ∈ Set.Icc 0 a, 2 ≤ f x)
  (h_max_achieved : ∃ x ∈ Set.Icc 0 a, f x = 3)
  (h_min_achieved : ∃ x ∈ Set.Icc 0 a, f x = 2) :
  a ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_function_extrema_implies_interval_bounds_l1810_181004


namespace NUMINAMATH_CALUDE_fraction_sum_l1810_181069

theorem fraction_sum : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1810_181069


namespace NUMINAMATH_CALUDE_sum_of_digits_doubled_l1810_181029

/-- Sum of digits function -/
def S (k : ℕ+) : ℕ := sorry

/-- All digits less than or equal to 7 -/
def digits_le_7 (n : ℕ+) : Prop := sorry

theorem sum_of_digits_doubled (k : ℕ+) :
  S k = 2187 → digits_le_7 (2 * k) → S (2 * k) = 4374 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_doubled_l1810_181029


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l1810_181016

theorem roots_of_polynomial (x : ℝ) :
  let p : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | p x = 0} = {2, 3, -2} := by
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l1810_181016


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_120_l1810_181076

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) :
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_120_l1810_181076


namespace NUMINAMATH_CALUDE_angle_edc_measure_l1810_181032

theorem angle_edc_measure (y : ℝ) :
  let angle_bde : ℝ := 4 * y
  let angle_edc : ℝ := 3 * y
  angle_bde + angle_edc = 180 →
  angle_edc = 540 / 7 := by
sorry

end NUMINAMATH_CALUDE_angle_edc_measure_l1810_181032


namespace NUMINAMATH_CALUDE_max_product_sum_l1810_181053

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({6, 7, 8, 9} : Set ℕ) → 
  g ∈ ({6, 7, 8, 9} : Set ℕ) → 
  h ∈ ({6, 7, 8, 9} : Set ℕ) → 
  j ∈ ({6, 7, 8, 9} : Set ℕ) → 
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j → 
  (f * g + g * h + h * j + f * j) ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l1810_181053


namespace NUMINAMATH_CALUDE_apple_picking_multiple_l1810_181091

theorem apple_picking_multiple (K : ℕ) (M : ℕ) : 
  K + 274 = 340 → 
  274 = M * K + 10 →
  M = 4 := by sorry

end NUMINAMATH_CALUDE_apple_picking_multiple_l1810_181091


namespace NUMINAMATH_CALUDE_birds_in_tree_l1810_181071

theorem birds_in_tree (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) :
  initial_birds = 14 →
  new_birds = 21 →
  total_birds = initial_birds + new_birds →
  total_birds = 35 := by
sorry

end NUMINAMATH_CALUDE_birds_in_tree_l1810_181071


namespace NUMINAMATH_CALUDE_total_assignments_for_20_points_l1810_181048

def homework_assignments (n : ℕ) : ℕ :=
  if n ≤ 4 then n
  else if n ≤ 8 then 4 + 2 * (n - 4)
  else if n ≤ 12 then 12 + 3 * (n - 8)
  else if n ≤ 16 then 24 + 4 * (n - 12)
  else 40 + 5 * (n - 16)

theorem total_assignments_for_20_points :
  homework_assignments 20 = 60 :=
by sorry

end NUMINAMATH_CALUDE_total_assignments_for_20_points_l1810_181048


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l1810_181054

-- Define the rectangle's properties
def rectangle_area : ℝ := 54.3
def rectangle_width : ℝ := 6

-- Theorem statement
theorem rectangle_length_proof :
  let length := rectangle_area / rectangle_width
  length = 9.05 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l1810_181054


namespace NUMINAMATH_CALUDE_power_equation_solution_l1810_181063

theorem power_equation_solution (x y : ℕ+) (h : 2^(x.val + 1) * 4^y.val = 128) : 
  x.val + 2 * y.val = 6 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1810_181063


namespace NUMINAMATH_CALUDE_vegan_menu_fraction_l1810_181035

theorem vegan_menu_fraction (vegan_dishes : ℕ) (total_dishes : ℕ) (soy_dishes : ℕ) :
  vegan_dishes = 6 →
  vegan_dishes = total_dishes / 3 →
  soy_dishes = 4 →
  (vegan_dishes - soy_dishes : ℚ) / total_dishes = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_vegan_menu_fraction_l1810_181035


namespace NUMINAMATH_CALUDE_bee_multiple_l1810_181028

theorem bee_multiple (bees_day1 bees_day2 : ℕ) : 
  bees_day1 = 144 → bees_day2 = 432 → bees_day2 / bees_day1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bee_multiple_l1810_181028


namespace NUMINAMATH_CALUDE_bread_pieces_theorem_l1810_181072

/-- Number of pieces after tearing a slice of bread in half twice -/
def pieces_per_slice : ℕ := 4

/-- Number of initial bread slices -/
def initial_slices : ℕ := 2

/-- Total number of bread pieces after tearing -/
def total_pieces : ℕ := initial_slices * pieces_per_slice

theorem bread_pieces_theorem : total_pieces = 8 := by
  sorry

end NUMINAMATH_CALUDE_bread_pieces_theorem_l1810_181072


namespace NUMINAMATH_CALUDE_problem_solution_l1810_181074

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n - 2 else 3 * n

theorem problem_solution (m : ℤ) (h1 : m % 2 = 0) (h2 : g (g (g m)) = 54) : m = 60 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1810_181074


namespace NUMINAMATH_CALUDE_final_stamp_count_l1810_181064

def parkers_stamps (initial_stamps : ℕ) (addies_stamps : ℕ) : ℕ :=
  initial_stamps + (addies_stamps / 4)

theorem final_stamp_count : parkers_stamps 18 72 = 36 := by
  sorry

end NUMINAMATH_CALUDE_final_stamp_count_l1810_181064


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1810_181078

theorem min_value_reciprocal_sum (m n : ℝ) : 
  m > 0 → n > 0 → m * 1 + n * 1 = 2 → (1 / m + 1 / n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1810_181078


namespace NUMINAMATH_CALUDE_max_remainder_239_div_n_l1810_181088

theorem max_remainder_239_div_n (n : ℕ) (h : n < 135) :
  (Finset.range n).sup (λ m => 239 % m) = 119 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_239_div_n_l1810_181088


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1810_181014

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.36363636

/-- The fraction 40/99 -/
def fraction : ℚ := 40 / 99

/-- Theorem stating that the repeating decimal 0.363636... equals 40/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1810_181014


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1810_181079

theorem roots_sum_of_squares (α β : ℝ) : 
  (α^2 - 2*α - 1 = 0) → (β^2 - 2*β - 1 = 0) → α^2 + β^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1810_181079


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l1810_181031

/-- Given a school with more girls than boys, calculate the number of girls. -/
theorem number_of_girls_in_school 
  (total_pupils : ℕ) 
  (girl_boy_difference : ℕ) 
  (h1 : total_pupils = 926)
  (h2 : girl_boy_difference = 458) :
  ∃ (girls boys : ℕ), 
    girls = boys + girl_boy_difference ∧ 
    girls + boys = total_pupils ∧ 
    girls = 692 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l1810_181031


namespace NUMINAMATH_CALUDE_triangle_angle_45_degrees_l1810_181098

theorem triangle_angle_45_degrees (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- angles are positive
  A + B + C = 180 → -- sum of angles in a triangle is 180°
  B + C = 3 * A → -- given condition
  A = 45 ∨ B = 45 ∨ C = 45 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_45_degrees_l1810_181098


namespace NUMINAMATH_CALUDE_samantha_sleep_hours_l1810_181025

/-- Represents a time of day in 24-hour format -/
structure TimeOfDay where
  hour : Nat
  minute : Nat
  is_valid : hour < 24 ∧ minute < 60

/-- Calculates the number of hours between two times -/
def hoursBetween (t1 t2 : TimeOfDay) : Nat :=
  if t2.hour ≥ t1.hour then
    t2.hour - t1.hour
  else
    24 + t2.hour - t1.hour

/-- Samantha's bedtime -/
def bedtime : TimeOfDay := {
  hour := 19,
  minute := 0,
  is_valid := by simp
}

/-- Samantha's wake-up time -/
def wakeupTime : TimeOfDay := {
  hour := 11,
  minute := 0,
  is_valid := by simp
}

theorem samantha_sleep_hours :
  hoursBetween bedtime wakeupTime = 16 := by sorry

end NUMINAMATH_CALUDE_samantha_sleep_hours_l1810_181025


namespace NUMINAMATH_CALUDE_savings_calculation_l1810_181001

/-- Calculates the amount saved given sales, basic salary, commission rate, and savings rate -/
def calculate_savings (sales : ℝ) (basic_salary : ℝ) (commission_rate : ℝ) (savings_rate : ℝ) : ℝ :=
  let total_earnings := basic_salary + sales * commission_rate
  total_earnings * savings_rate

/-- Proves that given the specified conditions, the amount saved is $29 -/
theorem savings_calculation :
  let sales := 2500
  let basic_salary := 240
  let commission_rate := 0.02
  let savings_rate := 0.10
  calculate_savings sales basic_salary commission_rate savings_rate = 29 := by
sorry

#eval calculate_savings 2500 240 0.02 0.10

end NUMINAMATH_CALUDE_savings_calculation_l1810_181001


namespace NUMINAMATH_CALUDE_clarence_oranges_l1810_181026

/-- The number of oranges Clarence had initially -/
def initial_oranges : ℕ := sorry

/-- The number of oranges Clarence received from Joyce -/
def oranges_from_joyce : ℕ := 3

/-- The total number of oranges Clarence has after receiving oranges from Joyce -/
def total_oranges : ℕ := 8

/-- Theorem stating that the initial number of oranges plus those from Joyce equals the total -/
theorem clarence_oranges : initial_oranges + oranges_from_joyce = total_oranges := by sorry

end NUMINAMATH_CALUDE_clarence_oranges_l1810_181026


namespace NUMINAMATH_CALUDE_specific_theater_seats_l1810_181096

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row : ℕ
  seat_increase : ℕ
  last_row : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let num_rows := (t.last_row - t.first_row) / t.seat_increase + 1
  num_rows * (t.first_row + t.last_row) / 2

/-- Theorem stating that a theater with specific parameters has 570 seats -/
theorem specific_theater_seats :
  let t : Theater := { first_row := 12, seat_increase := 2, last_row := 48 }
  total_seats t = 570 := by sorry

end NUMINAMATH_CALUDE_specific_theater_seats_l1810_181096


namespace NUMINAMATH_CALUDE_horner_v₂_equals_4_l1810_181065

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 1 + x + x^2 + x^3 + 2x^4 -/
def f : List ℝ := [1, 1, 1, 1, 2]

/-- v₂ in Horner's method for f(x) at x = 1 -/
def v₂ : ℝ :=
  let v₁ := 2 * 1 + 1  -- a₄x + a₃
  v₁ * 1 + 1           -- v₁x + a₂

theorem horner_v₂_equals_4 :
  v₂ = 4 := by sorry

end NUMINAMATH_CALUDE_horner_v₂_equals_4_l1810_181065


namespace NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l1810_181041

/-- Perimeter and area of a rectangle with specific dimensions -/
theorem rectangle_perimeter_and_area :
  let l : ℝ := Real.sqrt 6 + 2 * Real.sqrt 5
  let w : ℝ := 2 * Real.sqrt 6 - Real.sqrt 5
  let perimeter : ℝ := 2 * (l + w)
  let area : ℝ := l * w
  (perimeter = 6 * Real.sqrt 6 + 2 * Real.sqrt 5) ∧
  (area = 2 + 3 * Real.sqrt 30) := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_and_area_l1810_181041


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l1810_181013

theorem no_positive_integer_solution :
  ¬∃ (x y z t : ℕ+), x^2 + 2*y^2 = z^2 ∧ 2*x^2 + y^2 = t^2 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l1810_181013


namespace NUMINAMATH_CALUDE_victor_final_books_l1810_181094

/-- The number of books Victor has after various transactions -/
def final_books (initial : ℝ) (bought : ℝ) (given : ℝ) (donated : ℝ) : ℝ :=
  initial + bought - given - donated

/-- Theorem stating that Victor ends up with 19.8 books -/
theorem victor_final_books :
  final_books 35.5 12.3 7.2 20.8 = 19.8 := by
  sorry

end NUMINAMATH_CALUDE_victor_final_books_l1810_181094


namespace NUMINAMATH_CALUDE_students_taking_music_l1810_181097

theorem students_taking_music (total : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) :
  total = 500 →
  art = 20 →
  both = 10 →
  neither = 470 →
  total - neither - art + both = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_students_taking_music_l1810_181097


namespace NUMINAMATH_CALUDE_solve_temperature_l1810_181052

def temperature_problem (temps : List ℝ) (avg : ℝ) : Prop :=
  temps.length = 6 ∧
  (temps.sum + (7 * avg - temps.sum)) / 7 = avg

theorem solve_temperature (temps : List ℝ) (avg : ℝ) 
  (h : temperature_problem temps avg) : ℝ :=
  7 * avg - temps.sum

#check solve_temperature

end NUMINAMATH_CALUDE_solve_temperature_l1810_181052


namespace NUMINAMATH_CALUDE_original_figure_area_l1810_181087

/-- The area of the original figure given the properties of its intuitive diagram --/
theorem original_figure_area (height : ℝ) (top_angle : ℝ) (area_ratio : ℝ) : 
  height = 2 → 
  top_angle = 120 * π / 180 → 
  area_ratio = 2 * Real.sqrt 2 → 
  (1 / 2) * (4 * height) * (4 * height) * Real.sin top_angle * area_ratio = 8 * Real.sqrt 6 := by
  sorry

#check original_figure_area

end NUMINAMATH_CALUDE_original_figure_area_l1810_181087


namespace NUMINAMATH_CALUDE_beverly_bottle_caps_l1810_181059

/-- The number of groups Beverly's bottle caps can be organized into -/
def num_groups : ℕ := 7

/-- The number of bottle caps in each group -/
def caps_per_group : ℕ := 5

/-- The total number of bottle caps in Beverly's collection -/
def total_caps : ℕ := num_groups * caps_per_group

/-- Theorem stating that the total number of bottle caps is 35 -/
theorem beverly_bottle_caps : total_caps = 35 := by
  sorry

end NUMINAMATH_CALUDE_beverly_bottle_caps_l1810_181059


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1810_181039

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

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1810_181039


namespace NUMINAMATH_CALUDE_davantes_boy_friends_l1810_181010

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define Davante's total number of friends
def total_friends : ℕ := 2 * days_in_week

-- Define the number of Davante's friends who are girls
def girl_friends : ℕ := 3

-- Theorem statement
theorem davantes_boy_friends :
  total_friends - girl_friends = 11 :=
sorry

end NUMINAMATH_CALUDE_davantes_boy_friends_l1810_181010


namespace NUMINAMATH_CALUDE_average_problem_l1810_181058

theorem average_problem (x : ℝ) : 
  (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750 → x = 1255 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1810_181058


namespace NUMINAMATH_CALUDE_test_scores_sum_l1810_181019

/-- Given the scores of Bill, John, and Sue on a test, prove that their total sum is 160 points. -/
theorem test_scores_sum (bill john sue : ℕ) : 
  bill = john + 20 →   -- Bill scored 20 more points than John
  bill = sue / 2 →     -- Bill scored half as many points as Sue
  bill = 45 →          -- Bill received 45 points
  bill + john + sue = 160 := by
sorry

end NUMINAMATH_CALUDE_test_scores_sum_l1810_181019


namespace NUMINAMATH_CALUDE_smallest_n_for_square_not_cube_l1810_181057

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y * y = x

def expression (n k : ℕ) : ℕ :=
  3^k + n^k + (3*n)^k + 2014^k

theorem smallest_n_for_square_not_cube :
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, is_perfect_square (expression n k)) ∧
    (∀ k : ℕ, ¬ is_perfect_cube (expression n k)) ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(∀ k : ℕ, is_perfect_square (expression m k)) ∨
      ¬(∀ k : ℕ, ¬ is_perfect_cube (expression m k))) ∧
    n = 2 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_not_cube_l1810_181057


namespace NUMINAMATH_CALUDE_derivative_cosh_l1810_181040

open Real

theorem derivative_cosh (x : ℝ) : 
  deriv (fun x => (1/2) * (exp x + exp (-x))) x = (1/2) * (exp x - exp (-x)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_cosh_l1810_181040


namespace NUMINAMATH_CALUDE_positive_roots_of_x_power_x_l1810_181067

theorem positive_roots_of_x_power_x (x : ℝ) : 
  x > 0 → (x^x = 1 / Real.sqrt 2 ↔ x = 1/2 ∨ x = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_positive_roots_of_x_power_x_l1810_181067


namespace NUMINAMATH_CALUDE_allocation_ways_l1810_181043

-- Define the number of doctors and nurses
def num_doctors : ℕ := 2
def num_nurses : ℕ := 4

-- Define the number of schools
def num_schools : ℕ := 2

-- Define the number of doctors and nurses needed per school
def doctors_per_school : ℕ := 1
def nurses_per_school : ℕ := 2

-- Theorem statement
theorem allocation_ways :
  (Nat.choose num_doctors doctors_per_school) * (Nat.choose num_nurses nurses_per_school) = 12 :=
sorry

end NUMINAMATH_CALUDE_allocation_ways_l1810_181043


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1810_181003

/-- An arithmetic sequence with first term 2 and 10th term 20 has common difference 2 -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 10 = 20 →                          -- 10th term is 20
  a 2 - a 1 = 2 :=                     -- common difference is 2
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1810_181003


namespace NUMINAMATH_CALUDE_shooting_training_probabilities_l1810_181023

/-- Shooting training probabilities -/
structure ShootingProbabilities where
  nine_or_above : ℝ
  eight_to_nine : ℝ
  seven_to_eight : ℝ
  six_to_seven : ℝ

/-- Theorem for shooting training probabilities -/
theorem shooting_training_probabilities
  (probs : ShootingProbabilities)
  (h1 : probs.nine_or_above = 0.18)
  (h2 : probs.eight_to_nine = 0.51)
  (h3 : probs.seven_to_eight = 0.15)
  (h4 : probs.six_to_seven = 0.09) :
  (probs.nine_or_above + probs.eight_to_nine = 0.69) ∧
  (probs.nine_or_above + probs.eight_to_nine + probs.seven_to_eight + probs.six_to_seven = 0.93) :=
by sorry

end NUMINAMATH_CALUDE_shooting_training_probabilities_l1810_181023


namespace NUMINAMATH_CALUDE_francis_family_violins_l1810_181005

/-- The number of ukuleles in Francis' family --/
def num_ukuleles : ℕ := 2

/-- The number of guitars in Francis' family --/
def num_guitars : ℕ := 4

/-- The number of strings on each ukulele --/
def strings_per_ukulele : ℕ := 4

/-- The number of strings on each guitar --/
def strings_per_guitar : ℕ := 6

/-- The number of strings on each violin --/
def strings_per_violin : ℕ := 4

/-- The total number of strings among all instruments --/
def total_strings : ℕ := 40

/-- The number of violins in Francis' family --/
def num_violins : ℕ := 2

theorem francis_family_violins :
  num_violins * strings_per_violin = 
    total_strings - (num_ukuleles * strings_per_ukulele + num_guitars * strings_per_guitar) :=
by sorry

end NUMINAMATH_CALUDE_francis_family_violins_l1810_181005


namespace NUMINAMATH_CALUDE_total_donation_l1810_181036

def charity_donation (cassandra james stephanie alex : ℕ) : Prop :=
  cassandra = 5000 ∧
  james = cassandra - 276 ∧
  stephanie = 2 * james ∧
  alex = (3 * (cassandra + stephanie)) / 4 ∧
  cassandra + james + stephanie + alex = 31008

theorem total_donation :
  ∃ (cassandra james stephanie alex : ℕ),
    charity_donation cassandra james stephanie alex :=
by
  sorry

end NUMINAMATH_CALUDE_total_donation_l1810_181036


namespace NUMINAMATH_CALUDE_prob_select_B_is_one_fourth_prob_select_B_and_C_is_one_sixth_l1810_181092

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the total number of students
def total_students : ℕ := 4

-- Define the probability of selecting one student
def prob_select_one (s : Student) : ℚ :=
  1 / total_students

-- Define the probability of selecting two specific students
def prob_select_two (s1 s2 : Student) : ℚ :=
  2 / (total_students * (total_students - 1))

-- Theorem for part 1
theorem prob_select_B_is_one_fourth :
  prob_select_one Student.B = 1 / 4 := by sorry

-- Theorem for part 2
theorem prob_select_B_and_C_is_one_sixth :
  prob_select_two Student.B Student.C = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_select_B_is_one_fourth_prob_select_B_and_C_is_one_sixth_l1810_181092


namespace NUMINAMATH_CALUDE_coefficient_sum_l1810_181095

theorem coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l1810_181095


namespace NUMINAMATH_CALUDE_power_two_congruence_l1810_181050

theorem power_two_congruence (n : ℕ) (a : ℤ) (hn : n ≥ 1) (ha : Odd a) :
  a ^ (2 ^ n) ≡ 1 [ZMOD (2 ^ (n + 2))] := by
  sorry

end NUMINAMATH_CALUDE_power_two_congruence_l1810_181050


namespace NUMINAMATH_CALUDE_smallest_set_size_l1810_181073

theorem smallest_set_size (n : ℕ) (hn : n > 0) :
  let S := {S : Finset ℕ | S ⊆ Finset.range n ∧
    ∀ β : ℝ, β > 0 → (∀ s ∈ S, ∃ m : ℕ, s = ⌊β * m⌋) →
      ∀ k ∈ Finset.range n, ∃ m : ℕ, k = ⌊β * m⌋}
  ∃ S₀ ∈ S, S₀.card = n / 2 + 1 ∧ ∀ S' ∈ S, S'.card ≥ S₀.card :=
sorry

end NUMINAMATH_CALUDE_smallest_set_size_l1810_181073


namespace NUMINAMATH_CALUDE_chord_ratio_l1810_181000

-- Define the circle and points
variable (circle : Type) (A B C D E P : circle)

-- Define the distance function
variable (dist : circle → circle → ℝ)

-- State the theorem
theorem chord_ratio (h1 : dist A P = 5)
                    (h2 : dist C P = 9)
                    (h3 : dist D E = 4) :
  dist B P / dist E P = 81 / 805 := by sorry

end NUMINAMATH_CALUDE_chord_ratio_l1810_181000


namespace NUMINAMATH_CALUDE_victor_deck_count_l1810_181049

theorem victor_deck_count (cost_per_deck : ℕ) (friend_deck_count : ℕ) (total_spent : ℕ) : ℕ :=
  let victor_deck_count := (total_spent - friend_deck_count * cost_per_deck) / cost_per_deck
  have h1 : cost_per_deck = 8 := by sorry
  have h2 : friend_deck_count = 2 := by sorry
  have h3 : total_spent = 64 := by sorry
  have h4 : victor_deck_count = 6 := by sorry
  victor_deck_count

#check victor_deck_count

end NUMINAMATH_CALUDE_victor_deck_count_l1810_181049


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1810_181020

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a = 4√3, b = 12, and B = 60°, then A = 30° -/
theorem triangle_angle_measure (a b c A B C : ℝ) : 
  a = 4 * Real.sqrt 3 → 
  b = 12 → 
  B = 60 * π / 180 → 
  A = 30 * π / 180 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1810_181020


namespace NUMINAMATH_CALUDE_opposite_of_one_half_l1810_181045

theorem opposite_of_one_half : -(1 / 2 : ℚ) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_one_half_l1810_181045


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l1810_181038

theorem intersection_points_form_line : 
  ∀ (s : ℝ), 
  ∃ (x y : ℝ), 
  (x + 3 * y = 10 * s + 4) ∧ 
  (2 * x - y = 3 * s - 5) → 
  y = (119 / 133) * x + (435 / 133) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l1810_181038


namespace NUMINAMATH_CALUDE_translation_theorem_l1810_181037

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B : Point2D) (A' : Point2D) :
  A = Point2D.mk 2 2 →
  B = Point2D.mk (-1) 1 →
  A' = Point2D.mk (-2) (-2) →
  let t : Translation2D := { dx := A'.x - A.x, dy := A'.y - A.y }
  applyTranslation B t = Point2D.mk (-5) (-3) := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l1810_181037


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_l1810_181044

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Statement for the correct option (A)
theorem correct_calculation : -cubeRoot 8 = -2 := by sorry

-- Statements for the incorrect options (B, C, D)
theorem incorrect_calculation_B : -abs (-3) ≠ 3 := by sorry

theorem incorrect_calculation_C : Real.sqrt 16 ≠ 4 ∧ Real.sqrt 16 ≠ -4 := by sorry

theorem incorrect_calculation_D : -(2^2) ≠ 4 := by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_l1810_181044


namespace NUMINAMATH_CALUDE_keiths_purchases_total_cost_l1810_181085

theorem keiths_purchases_total_cost : 
  let rabbit_toy_cost : ℚ := 651/100
  let pet_food_cost : ℚ := 579/100
  let cage_cost : ℚ := 1251/100
  rabbit_toy_cost + pet_food_cost + cage_cost = 2481/100 := by
sorry

end NUMINAMATH_CALUDE_keiths_purchases_total_cost_l1810_181085


namespace NUMINAMATH_CALUDE_tan_theta_eq_seven_l1810_181068

theorem tan_theta_eq_seven (θ : Real) 
  (h1 : θ > π/4 ∧ θ < π/2) 
  (h2 : Real.cos (θ - π/4) = 4/5) : 
  Real.tan θ = 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_eq_seven_l1810_181068


namespace NUMINAMATH_CALUDE_weeks_per_season_l1810_181086

def weekly_earnings : ℕ := 1357
def num_seasons : ℕ := 73
def total_earnings : ℕ := 22090603

theorem weeks_per_season : 
  (total_earnings / weekly_earnings) / num_seasons = 223 :=
sorry

end NUMINAMATH_CALUDE_weeks_per_season_l1810_181086


namespace NUMINAMATH_CALUDE_test_score_mode_l1810_181018

/-- Represents a stem-and-leaf plot entry -/
structure StemLeafEntry where
  stem : ℕ
  leaves : List ℕ

/-- Calculates the mode of a list of numbers -/
def mode (numbers : List ℕ) : ℕ := sorry

/-- The stem-and-leaf plot of the test scores -/
def testScores : List StemLeafEntry := [
  ⟨5, [0, 5, 5]⟩,
  ⟨6, [2, 2, 8]⟩,
  ⟨7, [0, 1, 5, 9]⟩,
  ⟨8, [1, 1, 3, 5, 5, 5]⟩,
  ⟨9, [2, 6, 6, 8]⟩,
  ⟨10, [0, 0]⟩
]

/-- Converts a stem-and-leaf plot to a list of scores -/
def stemLeafToScores (plot : List StemLeafEntry) : List ℕ := sorry

theorem test_score_mode :
  mode (stemLeafToScores testScores) = 85 := by
  sorry

end NUMINAMATH_CALUDE_test_score_mode_l1810_181018


namespace NUMINAMATH_CALUDE_log_equality_implies_value_l1810_181047

theorem log_equality_implies_value (p q : ℝ) (c : ℝ) (h : 0 < p ∧ 0 < q ∧ 0 < 5) :
  Real.log p / Real.log 5 = c - Real.log q / Real.log 5 → p = 5^c / q := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_value_l1810_181047


namespace NUMINAMATH_CALUDE_dennis_rocks_theorem_l1810_181030

/-- Calculates the number of rocks Dennis made the fish spit out -/
def rocks_spit_out (initial_rocks : ℕ) (eaten_rocks : ℕ) (final_rocks : ℕ) : ℕ :=
  final_rocks - (initial_rocks - eaten_rocks)

/-- Proves that Dennis made the fish spit out 2 rocks -/
theorem dennis_rocks_theorem (initial_rocks eaten_rocks final_rocks : ℕ) 
  (h1 : initial_rocks = 10)
  (h2 : eaten_rocks = initial_rocks / 2)
  (h3 : final_rocks = 7) :
  rocks_spit_out initial_rocks eaten_rocks final_rocks = 2 := by
sorry

end NUMINAMATH_CALUDE_dennis_rocks_theorem_l1810_181030


namespace NUMINAMATH_CALUDE_fraction_equality_l1810_181009

theorem fraction_equality (m n : ℝ) (h : 1/m + 1/n = 7) : 14*m*n/(m+n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1810_181009
