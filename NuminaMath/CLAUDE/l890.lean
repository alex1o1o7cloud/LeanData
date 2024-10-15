import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_area_with_tangent_circle_l890_89022

/-- Given a rectangle ABCD with a circle of radius r tangent to sides AB, AD, and CD,
    and passing through a point one-third the distance from A to C along diagonal AC,
    the area of the rectangle is (2√2)/3 * r^2. -/
theorem rectangle_area_with_tangent_circle (r : ℝ) (h : r > 0) :
  ∃ (w h : ℝ),
    w > 0 ∧ h > 0 ∧
    h = r ∧
    (w^2 + h^2) = 9 * r^2 ∧
    w * h = (2 * Real.sqrt 2 / 3) * r^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_tangent_circle_l890_89022


namespace NUMINAMATH_CALUDE_bottle_production_l890_89084

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 20 such machines will produce 3600 bottles in 4 minutes. -/
theorem bottle_production (rate : ℕ) (h1 : 6 * rate = 270) : 20 * rate * 4 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_bottle_production_l890_89084


namespace NUMINAMATH_CALUDE_books_per_shelf_l890_89075

theorem books_per_shelf (total_shelves : ℕ) (total_books : ℕ) (h1 : total_shelves = 8) (h2 : total_books = 32) :
  total_books / total_shelves = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l890_89075


namespace NUMINAMATH_CALUDE_positive_real_inequality_l890_89058

theorem positive_real_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y^2016 ≥ 1) :
  x^2016 + y > 1 - 1/100 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l890_89058


namespace NUMINAMATH_CALUDE_oplus_nested_equation_l890_89056

def oplus (x y : ℝ) : ℝ := x^2 + 2*y

theorem oplus_nested_equation (a : ℝ) : oplus a (oplus a a) = 3*a^2 + 4*a := by
  sorry

end NUMINAMATH_CALUDE_oplus_nested_equation_l890_89056


namespace NUMINAMATH_CALUDE_weight_loss_per_month_l890_89004

def initial_weight : ℝ := 250
def final_weight : ℝ := 154
def months_in_year : ℕ := 12

theorem weight_loss_per_month :
  (initial_weight - final_weight) / months_in_year = 8 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_per_month_l890_89004


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l890_89009

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 5 + 1
  let y : ℝ := Real.sqrt 5 - 1
  ((5 * x + 3 * y) / (x^2 - y^2) + (2 * x) / (y^2 - x^2)) / (1 / (x^2 * y - x * y^2)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l890_89009


namespace NUMINAMATH_CALUDE_adjacent_sides_equal_not_implies_parallelogram_l890_89000

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- Definition of a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.A.1 - q.B.1 = q.D.1 - q.C.1 ∧ q.A.2 - q.B.2 = q.D.2 - q.C.2) ∧
  (q.A.1 - q.D.1 = q.B.1 - q.C.1 ∧ q.A.2 - q.D.2 = q.B.2 - q.C.2)

/-- Definition of equality of two sides -/
def sides_equal (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 = (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2

/-- Theorem: Adjacent sides being equal does not imply parallelogram -/
theorem adjacent_sides_equal_not_implies_parallelogram :
  ¬∀ (q : Quadrilateral), 
    (sides_equal q.A q.B q.A q.D ∧ sides_equal q.B q.C q.C q.D) → 
    is_parallelogram q :=
sorry

end NUMINAMATH_CALUDE_adjacent_sides_equal_not_implies_parallelogram_l890_89000


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l890_89055

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - x - 2)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -33 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l890_89055


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l890_89095

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 2 + 35 / 99) ∧ (x = 233 / 99) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l890_89095


namespace NUMINAMATH_CALUDE_problem_1_l890_89096

theorem problem_1 : Real.sqrt 18 - 4 * Real.sqrt (1/2) + Real.sqrt 24 / Real.sqrt 3 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l890_89096


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l890_89028

/-- Represents the arrangement of numbers in the square with a center circle -/
structure Arrangement :=
  (top_left : ℕ)
  (top_right : ℕ)
  (bottom_left : ℕ)
  (bottom_right : ℕ)
  (center : ℕ)

/-- The set of numbers to be arranged -/
def numbers : Finset ℕ := {2, 4, 6, 8, 10}

/-- Checks if the given arrangement satisfies the diagonal and vertex sum condition -/
def is_valid_arrangement (a : Arrangement) : Prop :=
  a.top_left + a.center + a.bottom_right = 
  a.top_right + a.center + a.bottom_left ∧
  a.top_left + a.center + a.bottom_right = 
  a.top_left + a.top_right + a.bottom_left + a.bottom_right

/-- Checks if the given arrangement uses all the required numbers -/
def uses_all_numbers (a : Arrangement) : Prop :=
  {a.top_left, a.top_right, a.bottom_left, a.bottom_right, a.center} = numbers

/-- Theorem stating that a valid arrangement exists -/
theorem valid_arrangement_exists : 
  ∃ (a : Arrangement), is_valid_arrangement a ∧ uses_all_numbers a :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l890_89028


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l890_89027

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l890_89027


namespace NUMINAMATH_CALUDE_pollard_complexity_l890_89092

/-- Represents the state of the algorithm at each iteration -/
structure AlgorithmState where
  u : Nat
  v : Nat

/-- The update function for the algorithm -/
def update (state : AlgorithmState) : AlgorithmState := sorry

/-- The main loop of the algorithm -/
def mainLoop (n : Nat) (initialState : AlgorithmState) : Nat := sorry

theorem pollard_complexity {n p : Nat} (hprime : Nat.Prime p) (hfactor : p ∣ n) :
  ∃ (c : Nat), mainLoop n (AlgorithmState.mk 1 1) ≤ 2 * p ∧
  mainLoop n (AlgorithmState.mk 1 1) ≤ c * p * (Nat.log n)^2 := by
  sorry

end NUMINAMATH_CALUDE_pollard_complexity_l890_89092


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l890_89088

/-- Proves that given a bowl with 10 ounces of water, if 4% of the original amount
    evaporates over 50 days, then the amount of water that evaporates each day is 0.008 ounces. -/
theorem water_evaporation_rate (initial_water : ℝ) (days : ℕ) (evaporation_percent : ℝ) :
  initial_water = 10 →
  days = 50 →
  evaporation_percent = 4 →
  (evaporation_percent / 100 * initial_water) / days = 0.008 := by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l890_89088


namespace NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l890_89024

/-- The tax rate for properties in Township K -/
def tax_rate : ℝ := sorry

/-- The initial assessed value of the property -/
def initial_value : ℝ := 20000

/-- The final assessed value of the property -/
def final_value : ℝ := 28000

/-- The increase in property tax -/
def tax_increase : ℝ := 800

/-- Theorem stating that the tax rate is 10% of the assessed value -/
theorem tax_rate_is_ten_percent :
  tax_rate = 0.1 :=
by
  sorry

#check tax_rate_is_ten_percent

end NUMINAMATH_CALUDE_tax_rate_is_ten_percent_l890_89024


namespace NUMINAMATH_CALUDE_grape_difference_l890_89039

/-- The number of grapes in Rob's bowl -/
def robs_grapes : ℕ := 25

/-- The total number of grapes in all three bowls -/
def total_grapes : ℕ := 83

/-- The number of grapes in Allie's bowl -/
def allies_grapes : ℕ := (total_grapes - robs_grapes - 4) / 2

/-- The number of grapes in Allyn's bowl -/
def allyns_grapes : ℕ := allies_grapes + 4

theorem grape_difference : allies_grapes - robs_grapes = 2 := by
  sorry

end NUMINAMATH_CALUDE_grape_difference_l890_89039


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l890_89061

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℝ
  day : Nat

/-- Represents the volume of orangeade made on a given day -/
structure OrangeadeVolume where
  volume : ℝ
  day : Nat

/-- Calculates the revenue from selling orangeade -/
def revenue (price : OrangeadePrice) (volume : OrangeadeVolume) : ℝ :=
  price.price * volume.volume

theorem orangeade_price_day2 
  (price_day1 : OrangeadePrice)
  (price_day2 : OrangeadePrice)
  (volume_day1 : OrangeadeVolume)
  (volume_day2 : OrangeadeVolume)
  (h1 : price_day1.day = 1)
  (h2 : price_day2.day = 2)
  (h3 : volume_day1.day = 1)
  (h4 : volume_day2.day = 2)
  (h5 : price_day1.price = 0.82)
  (h6 : volume_day2.volume = (3/2) * volume_day1.volume)
  (h7 : revenue price_day1 volume_day1 = revenue price_day2 volume_day2) :
  price_day2.price = (2 * 0.82) / 3 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l890_89061


namespace NUMINAMATH_CALUDE_adjacent_roll_probability_adjacent_roll_probability_proof_l890_89082

/-- The probability that no two adjacent people roll the same number on an eight-sided die
    when six people sit around a circular table. -/
theorem adjacent_roll_probability : ℚ :=
  117649 / 262144

/-- The number of people sitting around the circular table. -/
def num_people : ℕ := 6

/-- The number of sides on the die. -/
def die_sides : ℕ := 8

/-- The probability of rolling a different number than the previous person. -/
def diff_roll_prob : ℚ := 7 / 8

theorem adjacent_roll_probability_proof :
  adjacent_roll_probability = diff_roll_prob ^ num_people :=
sorry

end NUMINAMATH_CALUDE_adjacent_roll_probability_adjacent_roll_probability_proof_l890_89082


namespace NUMINAMATH_CALUDE_median_room_number_l890_89064

/-- Given a list of integers from 1 to n with two consecutive numbers removed,
    this function returns the median of the remaining numbers. -/
def medianWithGap (n : ℕ) (gap_start : ℕ) : ℕ :=
  if gap_start ≤ (n + 1) / 2
  then (n + 1) / 2 + 1
  else (n + 1) / 2

theorem median_room_number :
  medianWithGap 23 14 = 13 :=
by sorry

end NUMINAMATH_CALUDE_median_room_number_l890_89064


namespace NUMINAMATH_CALUDE_average_string_length_l890_89076

theorem average_string_length :
  let string1 : ℚ := 2
  let string2 : ℚ := 5
  let string3 : ℚ := 3
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_string_length_l890_89076


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l890_89002

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = 172546/1048576 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l890_89002


namespace NUMINAMATH_CALUDE_no_integer_solution_for_cornelia_age_l890_89023

theorem no_integer_solution_for_cornelia_age :
  ∀ (C : ℕ) (K : ℕ),
    K = 30 →
    C + 20 = 2 * (K + 20) →
    (K - 5)^2 = 3 * (C - 5) →
    False :=
by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_cornelia_age_l890_89023


namespace NUMINAMATH_CALUDE_max_sides_equal_longest_diagonal_l890_89049

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- The longest diagonal of a convex polygon -/
def longest_diagonal (p : ConvexPolygon) : ℝ :=
  sorry

/-- The number of sides equal to the longest diagonal in a convex polygon -/
def num_sides_equal_longest_diagonal (p : ConvexPolygon) : ℕ :=
  sorry

/-- Theorem: The maximum number of sides that can be equal to the longest diagonal in a convex polygon is 2 -/
theorem max_sides_equal_longest_diagonal (p : ConvexPolygon) :
  num_sides_equal_longest_diagonal p ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_sides_equal_longest_diagonal_l890_89049


namespace NUMINAMATH_CALUDE_drug_price_reduction_l890_89016

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 60)
  (h2 : final_price = 48.6)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) : 
  x = 0.1 := by
sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l890_89016


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l890_89021

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 900 is 40√2 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 900}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ ‖f₁ - f₂‖ = 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l890_89021


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l890_89018

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π. -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1 / 3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l890_89018


namespace NUMINAMATH_CALUDE_zara_sheep_count_l890_89099

/-- The number of sheep Zara bought -/
def num_sheep : ℕ := 7

/-- The number of cows Zara bought -/
def num_cows : ℕ := 24

/-- The number of goats Zara bought -/
def num_goats : ℕ := 113

/-- The number of groups for transporting animals -/
def num_groups : ℕ := 3

/-- The number of animals in each group -/
def animals_per_group : ℕ := 48

theorem zara_sheep_count :
  num_sheep + num_cows + num_goats = num_groups * animals_per_group := by
  sorry

end NUMINAMATH_CALUDE_zara_sheep_count_l890_89099


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l890_89074

def num_white_balls : ℕ := 3
def num_red_balls : ℕ := 4
def num_yellow_balls : ℕ := 5
def num_boxes : ℕ := 3

def distribute_balls : ℕ := (Nat.choose (num_boxes + num_white_balls - 1) (num_boxes - 1)) *
                             (Nat.choose (num_boxes + num_red_balls - 1) (num_boxes - 1)) *
                             (Nat.choose (num_boxes + num_yellow_balls - 1) (num_boxes - 1))

theorem ball_distribution_theorem : distribute_balls = 3150 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_theorem_l890_89074


namespace NUMINAMATH_CALUDE_inequality_holds_iff_k_geq_four_l890_89087

theorem inequality_holds_iff_k_geq_four :
  ∀ k : ℝ, k > 0 →
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    a / (b + c) + b / (c + a) + k * c / (a + b) ≥ 2) ↔
  k ≥ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_k_geq_four_l890_89087


namespace NUMINAMATH_CALUDE_log_difference_cube_l890_89053

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_difference_cube (x y a : ℝ) (h : x > 0) (h' : y > 0) :
  lg x - lg y = a → lg ((x / 2) ^ 3) - lg ((y / 2) ^ 3) = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_log_difference_cube_l890_89053


namespace NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l890_89041

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total cycle duration of a traffic light -/
def cycleDuration (d : TrafficLightDurations) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of seeing a red light -/
def redLightProbability (d : TrafficLightDurations) : ℚ :=
  d.red / (cycleDuration d)

/-- Theorem stating that the probability of seeing a red light is 2/5 -/
theorem red_light_probability_is_two_fifths (d : TrafficLightDurations)
  (h_red : d.red = 30)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 40) :
  redLightProbability d = 2/5 := by
  sorry

#eval redLightProbability ⟨30, 5, 40⟩

end NUMINAMATH_CALUDE_red_light_probability_is_two_fifths_l890_89041


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l890_89054

/-- Proves that the percentage increase resulting in a $324 weekly salary is 8%,
    given that a 10% increase results in a $330 weekly salary. -/
theorem salary_increase_percentage (current_salary : ℝ) : 
  (current_salary * 1.1 = 330) →
  (current_salary * (1 + 0.08) = 324) := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l890_89054


namespace NUMINAMATH_CALUDE_existence_equivalence_l890_89093

theorem existence_equivalence (a b : ℤ) :
  (∃ c d : ℤ, a + b + c + d = 0 ∧ a * c + b * d = 0) ↔ (∃ k : ℤ, 2 * a * b = k * (a - b)) :=
by sorry

end NUMINAMATH_CALUDE_existence_equivalence_l890_89093


namespace NUMINAMATH_CALUDE_area_is_twenty_l890_89034

/-- The equation of the graph --/
def graph_equation (x y : ℝ) : Prop := abs (5 * x) + abs (2 * y) = 10

/-- The set of points satisfying the graph equation --/
def graph_set : Set (ℝ × ℝ) := {p | graph_equation p.1 p.2}

/-- The area enclosed by the graph --/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the area enclosed by the graph is 20 --/
theorem area_is_twenty : enclosed_area = 20 := by sorry

end NUMINAMATH_CALUDE_area_is_twenty_l890_89034


namespace NUMINAMATH_CALUDE_triangle_property_l890_89069

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a∙sin(A) + c∙sin(C) - √2∙a∙sin(C) = b∙sin(B) and cos(A) = 1/3,
    then B = π/4 and sin(C) = (4 + √2) / 6. -/
theorem triangle_property (a b c A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B →
  Real.cos A = 1 / 3 →
  B = π / 4 ∧ Real.sin C = (4 + Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l890_89069


namespace NUMINAMATH_CALUDE_repeating_decimal_quotient_l890_89060

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (n : ℕ) : ℚ :=
  n / 99

theorem repeating_decimal_quotient :
  (RepeatingDecimal 54) / (RepeatingDecimal 18) = 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_quotient_l890_89060


namespace NUMINAMATH_CALUDE_age_difference_l890_89006

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 23 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l890_89006


namespace NUMINAMATH_CALUDE_total_birds_is_148_l890_89042

/-- The number of birds seen on Monday -/
def monday_birds : ℕ := 70

/-- The number of birds seen on Tuesday -/
def tuesday_birds : ℕ := monday_birds / 2

/-- The number of birds seen on Wednesday -/
def wednesday_birds : ℕ := tuesday_birds + 8

/-- The total number of birds seen from Monday to Wednesday -/
def total_birds : ℕ := monday_birds + tuesday_birds + wednesday_birds

/-- Theorem stating that the total number of birds seen is 148 -/
theorem total_birds_is_148 : total_birds = 148 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_is_148_l890_89042


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l890_89047

theorem least_three_digit_multiple_of_nine : ∃ n : ℕ, 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  n % 9 = 0 ∧
  ∀ m : ℕ, (m ≥ 100 ∧ m ≤ 999 ∧ m % 9 = 0) → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_nine_l890_89047


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l890_89085

theorem sqrt_eight_and_nine_sixteenths (x : ℝ) : 
  x = Real.sqrt (8 + 9/16) → x = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l890_89085


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l890_89079

theorem factor_difference_of_squares (y : ℝ) :
  25 - 16 * y^2 = (5 - 4*y) * (5 + 4*y) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l890_89079


namespace NUMINAMATH_CALUDE_journey_length_is_70_l890_89057

-- Define the journey
def Journey (length : ℝ) : Prop :=
  -- Time taken at 40 kmph
  let time_at_40 := length / 40
  -- Time taken at 35 kmph
  let time_at_35 := length / 35
  -- The difference in time is 0.25 hours (15 minutes)
  time_at_35 - time_at_40 = 0.25

-- Theorem stating that the journey length is 70 km
theorem journey_length_is_70 : 
  ∃ (length : ℝ), Journey length ∧ length = 70 :=
sorry

end NUMINAMATH_CALUDE_journey_length_is_70_l890_89057


namespace NUMINAMATH_CALUDE_length_of_segment_l890_89005

/-- Given a line segment AB divided by points P and Q, prove that AB has length 25 -/
theorem length_of_segment (A B P Q : ℝ) : 
  (P - A) / (B - A) = 3 / 5 →  -- P divides AB in ratio 3:2
  (Q - A) / (B - A) = 2 / 5 →  -- Q divides AB in ratio 2:3
  Q - P = 5 →                  -- Distance between P and Q is 5 units
  B - A = 25 := by             -- Length of AB is 25 units
sorry

end NUMINAMATH_CALUDE_length_of_segment_l890_89005


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l890_89046

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) : 
  a > 0 → 
  b > 0 → 
  c > 0 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (b * c / Real.sqrt (a^2 + b^2) = Real.sqrt 3 / 2 * c) →
  c / a = 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l890_89046


namespace NUMINAMATH_CALUDE_continuous_finite_preimage_implies_smp_l890_89081

open Set

/-- Definition of "smp" property for a function -/
def IsSmp (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ (n : ℕ) (c : Fin (n + 1) → ℝ),
    c 0 = a ∧ c (Fin.last n) = b ∧
    (∀ i : Fin n, c i < c (i + 1)) ∧
    (∀ i : Fin n, ∀ x ∈ Ioo (c i) (c (i + 1)),
      (f (c i) < f x ∧ f x < f (c (i + 1))) ∨
      (f (c i) > f x ∧ f x > f (c (i + 1))))

/-- Main theorem statement -/
theorem continuous_finite_preimage_implies_smp
  (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b))
  (h_finite : ∀ v : ℝ, Set.Finite {x ∈ Icc a b | f x = v}) :
  IsSmp f a b :=
sorry

end NUMINAMATH_CALUDE_continuous_finite_preimage_implies_smp_l890_89081


namespace NUMINAMATH_CALUDE_prob_both_female_given_one_female_l890_89077

/-- Represents the number of male students -/
def num_male : ℕ := 3

/-- Represents the number of female students -/
def num_female : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := num_male + num_female

/-- Represents the number of students drawn -/
def students_drawn : ℕ := 2

/-- The probability of drawing both female students given that one female student is drawn -/
theorem prob_both_female_given_one_female :
  (students_drawn = 2) →
  (num_male = 3) →
  (num_female = 2) →
  (∃ (p : ℚ), p = 1 / 7 ∧ 
    p = (1 : ℚ) / (total_students.choose students_drawn - num_male.choose students_drawn)) :=
sorry

end NUMINAMATH_CALUDE_prob_both_female_given_one_female_l890_89077


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l890_89059

/-- A coloring of the plane using seven colors -/
def Coloring := ℝ × ℝ → Fin 7

/-- The property that no two points of the same color are exactly 1 unit apart -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ x y : ℝ × ℝ, c x = c y → (x.1 - y.1)^2 + (x.2 - y.2)^2 ≠ 1

/-- There exists a coloring of the plane using seven colors such that
    no two points of the same color are exactly 1 unit apart -/
theorem exists_valid_coloring : ∃ c : Coloring, ValidColoring c := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l890_89059


namespace NUMINAMATH_CALUDE_power_equality_l890_89083

theorem power_equality (p : ℕ) : 16^5 = 4^p → p = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l890_89083


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l890_89090

theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 289) 
  (h2 : total_cost = 3944) : 
  total_cost / (4 * Real.sqrt area) = 58 := by
sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l890_89090


namespace NUMINAMATH_CALUDE_quadratic_roots_negative_real_part_l890_89029

theorem quadratic_roots_negative_real_part (p q : ℝ) :
  (∃ x : ℂ, p * x^2 + (p^2 - q) * x - (2*p - q - 1) = 0 ∧ x.re < 0) ↔
  (p = 0 ∧ -1 < q ∧ q < 0) ∨ (p > 0 ∧ q < p^2 ∧ q > 2*p - 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_negative_real_part_l890_89029


namespace NUMINAMATH_CALUDE_maximum_marks_l890_89001

theorem maximum_marks (percentage : ℝ) (obtained_marks : ℝ) (max_marks : ℝ) : 
  percentage = 92 / 100 → 
  obtained_marks = 184 → 
  percentage * max_marks = obtained_marks → 
  max_marks = 200 := by
sorry

end NUMINAMATH_CALUDE_maximum_marks_l890_89001


namespace NUMINAMATH_CALUDE_trigonometric_identities_l890_89038

theorem trigonometric_identities (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -2) : 
  ((2 * Real.cos (Real.pi / 2 + α) - Real.cos (Real.pi - α)) / 
   (Real.sin (Real.pi / 2 - α) - 3 * Real.sin (Real.pi + α)) = -1) ∧
  (2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l890_89038


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l890_89097

theorem sum_of_reciprocals_of_quadratic_roots :
  let a := 1
  let b := -17
  let c := 8
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (1 / r₁ + 1 / r₂) = 17 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_quadratic_roots_l890_89097


namespace NUMINAMATH_CALUDE_jovana_shells_l890_89030

/-- The amount of shells added to a bucket -/
def shells_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Jovana added 12 pounds of shells to her bucket -/
theorem jovana_shells : shells_added 5 17 = 12 := by
  sorry

end NUMINAMATH_CALUDE_jovana_shells_l890_89030


namespace NUMINAMATH_CALUDE_triangle_problem_l890_89071

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.c = 5/2)
  (h2 : t.b = Real.sqrt 6)
  (h3 : 4 * t.a - 3 * Real.sqrt 6 * Real.cos t.A = 0) :
  t.a = 3/2 ∧ t.B = 2 * t.A := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l890_89071


namespace NUMINAMATH_CALUDE_phone_number_probability_l890_89026

theorem phone_number_probability :
  let first_three_options : ℕ := 2
  let last_four_arrangements : ℕ := 24
  let total_numbers : ℕ := first_three_options * last_four_arrangements
  let correct_numbers : ℕ := 1
  (correct_numbers : ℚ) / total_numbers = 1 / 48 := by
sorry

end NUMINAMATH_CALUDE_phone_number_probability_l890_89026


namespace NUMINAMATH_CALUDE_no_such_function_l890_89020

theorem no_such_function : ¬∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_l890_89020


namespace NUMINAMATH_CALUDE_cost_of_500_pencils_l890_89003

/-- The cost of n pencils in dollars, given the price of one pencil in cents and the number of cents in a dollar. -/
def cost_of_pencils (n : ℕ) (price_per_pencil : ℕ) (cents_per_dollar : ℕ) : ℚ :=
  (n * price_per_pencil : ℚ) / cents_per_dollar

/-- Theorem stating that the cost of 500 pencils is 10 dollars, given the specified conditions. -/
theorem cost_of_500_pencils : 
  cost_of_pencils 500 2 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_pencils_l890_89003


namespace NUMINAMATH_CALUDE_angle_D_measure_l890_89068

-- Define the pentagon and its properties
structure Pentagon where
  A : ℝ  -- Measure of angle A
  B : ℝ  -- Measure of angle B
  C : ℝ  -- Measure of angle C
  D : ℝ  -- Measure of angle D
  E : ℝ  -- Measure of angle E
  convex : A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 ∧ E > 0
  sum_angles : A + B + C + D + E = 540
  congruent_ABC : A = B ∧ B = C
  congruent_DE : D = E
  A_less_D : A = D - 40

-- Theorem statement
theorem angle_D_measure (p : Pentagon) : p.D = 132 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l890_89068


namespace NUMINAMATH_CALUDE_andreas_living_room_area_andreas_living_room_area_is_48_l890_89032

/-- The area of Andrea's living room floor given a carpet covering 75% of it -/
theorem andreas_living_room_area (carpet_width : ℝ) (carpet_length : ℝ) 
  (carpet_coverage_percentage : ℝ) : ℝ :=
  let carpet_area := carpet_width * carpet_length
  let floor_area := carpet_area / carpet_coverage_percentage
  floor_area

/-- Proof of Andrea's living room floor area -/
theorem andreas_living_room_area_is_48 :
  andreas_living_room_area 4 9 0.75 = 48 := by
  sorry

end NUMINAMATH_CALUDE_andreas_living_room_area_andreas_living_room_area_is_48_l890_89032


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l890_89011

def D (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 4

theorem sqrt_D_irrational : ∀ x : ℝ, Irrational (Real.sqrt (D x)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l890_89011


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l890_89052

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)) →  -- Definition of S_n
  a 1 = -2011 →                                            -- Given a_1
  (S 2010 / 2010) - (S 2008 / 2008) = 2 →                  -- Given condition
  S 2011 = -2011 :=                                        -- Conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l890_89052


namespace NUMINAMATH_CALUDE_rosa_flower_count_l890_89080

/-- Given Rosa's initial flower count and the number of flowers Andre gave her,
    prove that the total number of flowers Rosa has now is equal to the sum of these two quantities. -/
theorem rosa_flower_count (initial_flowers andre_flowers : ℕ) :
  initial_flowers = 67 →
  andre_flowers = 23 →
  initial_flowers + andre_flowers = 90 :=
by sorry

end NUMINAMATH_CALUDE_rosa_flower_count_l890_89080


namespace NUMINAMATH_CALUDE_ladies_walk_l890_89010

/-- The combined distance walked by two ladies in Central Park -/
theorem ladies_walk (distance_lady2 : ℝ) (h1 : distance_lady2 = 4) :
  let distance_lady1 : ℝ := 2 * distance_lady2
  distance_lady1 + distance_lady2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ladies_walk_l890_89010


namespace NUMINAMATH_CALUDE_particular_number_l890_89072

theorem particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_l890_89072


namespace NUMINAMATH_CALUDE_max_wins_l890_89043

/-- 
Given that the ratio of Chloe's wins to Max's wins is 8:3, and Chloe won 24 times,
prove that Max won 9 times.
-/
theorem max_wins (chloe_wins : ℕ) (max_wins : ℕ) 
  (h1 : chloe_wins = 24)
  (h2 : chloe_wins * 3 = max_wins * 8) : 
  max_wins = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_wins_l890_89043


namespace NUMINAMATH_CALUDE_max_sections_five_lines_l890_89073

/-- The maximum number of sections a rectangle can be divided into by n line segments -/
def max_sections (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of sections a rectangle can be divided into by 5 line segments is 16 -/
theorem max_sections_five_lines :
  max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_lines_l890_89073


namespace NUMINAMATH_CALUDE_rectangle_composition_l890_89015

/-- Given a rectangle ABCD composed of six identical smaller rectangles,
    prove that the length y is 20 -/
theorem rectangle_composition (x y : ℝ) : 
  (3 * y) * (2 * x) = 2400 →  -- Area of ABCD
  y = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_composition_l890_89015


namespace NUMINAMATH_CALUDE_f_properties_l890_89045

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b / x^2

theorem f_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := f a b
  ∃ (min_value : ℝ) (min_point : ℝ),
    (∀ x, x ≠ 0 → f x ≥ min_value) ∧
    (f min_point = min_value) ∧
    (min_value = 2 * Real.sqrt (a * b)) ∧
    (min_point = Real.sqrt (Real.sqrt (b / a))) ∧
    (∀ x, f (-x) = f x) ∧
    (∀ x y, 0 < x ∧ x < y ∧ y < min_point → f x > f y) ∧
    (∀ x y, min_point < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l890_89045


namespace NUMINAMATH_CALUDE_nabla_example_l890_89031

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_example : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l890_89031


namespace NUMINAMATH_CALUDE_receipts_change_after_price_reduction_and_sales_increase_l890_89067

/-- Calculates the percentage change in total receipts when price is reduced and sales increase -/
theorem receipts_change_after_price_reduction_and_sales_increase
  (original_price : ℝ)
  (original_sales : ℝ)
  (price_reduction_percent : ℝ)
  (sales_increase_percent : ℝ)
  (h1 : price_reduction_percent = 30)
  (h2 : sales_increase_percent = 50)
  : (((1 - price_reduction_percent / 100) * (1 + sales_increase_percent / 100) - 1) * 100 = 5) := by
  sorry

end NUMINAMATH_CALUDE_receipts_change_after_price_reduction_and_sales_increase_l890_89067


namespace NUMINAMATH_CALUDE_chess_draw_probability_l890_89098

theorem chess_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.9) :
  prob_A_not_lose - prob_A_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l890_89098


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_median_relation_l890_89013

/-- In a right triangle, the square of the hypotenuse is equal to four-fifths of the sum of squares of the medians to the other two sides. -/
theorem right_triangle_hypotenuse_median_relation (a b c k_a k_b : ℝ) :
  a > 0 → b > 0 → c > 0 → k_a > 0 → k_b > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  k_a^2 = (2*b^2 + 2*c^2 - a^2) / 4 →  -- Definition of k_a
  k_b^2 = (2*a^2 + 2*c^2 - b^2) / 4 →  -- Definition of k_b
  c^2 = (4/5) * (k_a^2 + k_b^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_median_relation_l890_89013


namespace NUMINAMATH_CALUDE_four_nested_s_of_6_l890_89051

-- Define the function s
def s (x : ℚ) : ℚ := 1 / (2 - x)

-- State the theorem
theorem four_nested_s_of_6 : s (s (s (s 6))) = 14 / 19 := by sorry

end NUMINAMATH_CALUDE_four_nested_s_of_6_l890_89051


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l890_89070

theorem absolute_value_inequality_solution_set :
  ∀ x : ℝ, |x - 2| < 1 ↔ x ∈ Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l890_89070


namespace NUMINAMATH_CALUDE_max_m_value_l890_89065

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  symmetry : ∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c
  inequality : ∀ x ∈ Set.Ioo 0 2, a * x^2 + b * x + c ≤ ((x + 1) / 2)^2
  min_value : ∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 0

/-- The theorem stating the maximum value of m -/
theorem max_m_value (f : QuadraticFunction) :
  ∃ m : ℝ, m = 9 ∧ m > 1 ∧
  (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) ∧
  (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f.a * (x + t)^2 + f.b * (x + t) + f.c ≤ x) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l890_89065


namespace NUMINAMATH_CALUDE_lattice_points_on_curve_l890_89017

theorem lattice_points_on_curve : 
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 15) ∧ 
    points.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_curve_l890_89017


namespace NUMINAMATH_CALUDE_two_numbers_difference_l890_89012

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 40 → 
  3 * y - 2 * x = 10 → 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l890_89012


namespace NUMINAMATH_CALUDE_constant_k_value_l890_89040

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4) → k = -17 := by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l890_89040


namespace NUMINAMATH_CALUDE_hotel_profit_equation_correct_l890_89089

/-- Represents a hotel's pricing and occupancy model -/
structure Hotel where
  baseRooms : ℕ
  basePrice : ℝ
  costPerRoom : ℝ
  vacancyRate : ℝ
  priceIncrease : ℝ
  desiredProfit : ℝ

/-- The profit equation for the hotel -/
def profitEquation (h : Hotel) : Prop :=
  (h.basePrice + h.priceIncrease - h.costPerRoom) * 
  (h.baseRooms - h.priceIncrease / h.vacancyRate) = h.desiredProfit

/-- Theorem stating that the given equation correctly represents the hotel's profit scenario -/
theorem hotel_profit_equation_correct (h : Hotel) 
  (hRooms : h.baseRooms = 50)
  (hBasePrice : h.basePrice = 180)
  (hCost : h.costPerRoom = 20)
  (hVacancy : h.vacancyRate = 10)
  (hProfit : h.desiredProfit = 10890) :
  profitEquation h := by sorry

end NUMINAMATH_CALUDE_hotel_profit_equation_correct_l890_89089


namespace NUMINAMATH_CALUDE_range_of_k_is_real_l890_89044

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an increasing function
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem range_of_k_is_real (h : IsIncreasing f) : 
  ∀ k : ℝ, ∃ x : ℝ, f x = k :=
sorry

end NUMINAMATH_CALUDE_range_of_k_is_real_l890_89044


namespace NUMINAMATH_CALUDE_no_square_on_four_circles_l890_89062

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a square in a plane -/
structure Square where
  vertices : Fin 4 → ℝ × ℝ

/-- Checks if four radii form a strictly increasing arithmetic progression -/
def is_strict_arithmetic_progression (r₁ r₂ r₃ r₄ : ℝ) : Prop :=
  ∃ (a d : ℝ), d > 0 ∧ r₁ = a ∧ r₂ = a + d ∧ r₃ = a + 2*d ∧ r₄ = a + 3*d

/-- Checks if a point lies on a circle -/
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Main theorem statement -/
theorem no_square_on_four_circles (c₁ c₂ c₃ c₄ : Circle) 
  (h_common_center : c₁.center = c₂.center ∧ c₂.center = c₃.center ∧ c₃.center = c₄.center)
  (h_radii : is_strict_arithmetic_progression c₁.radius c₂.radius c₃.radius c₄.radius) :
  ¬ ∃ (s : Square), 
    (point_on_circle (s.vertices 0) c₁) ∧
    (point_on_circle (s.vertices 1) c₂) ∧
    (point_on_circle (s.vertices 2) c₃) ∧
    (point_on_circle (s.vertices 3) c₄) :=
by sorry

end NUMINAMATH_CALUDE_no_square_on_four_circles_l890_89062


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l890_89008

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 46 years older than his son and the son's current age is 44. -/
theorem mans_age_to_sons_age_ratio :
  ∀ (son_age man_age : ℕ),
    son_age = 44 →
    man_age = son_age + 46 →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l890_89008


namespace NUMINAMATH_CALUDE_complex_number_modulus_l890_89050

theorem complex_number_modulus : 
  let z : ℂ := (-1 - 2*I) / (1 - I)^2
  ‖z‖ = Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l890_89050


namespace NUMINAMATH_CALUDE_specific_bill_amount_l890_89063

/-- Calculates the amount of a bill given its true discount, due time, and interest rate. -/
def bill_amount (true_discount : ℚ) (due_time : ℚ) (interest_rate : ℚ) : ℚ :=
  (true_discount * (100 + interest_rate * due_time)) / (interest_rate * due_time)

/-- Theorem stating that given the specific conditions, the bill amount is 1680. -/
theorem specific_bill_amount :
  let true_discount : ℚ := 180
  let due_time : ℚ := 9 / 12  -- 9 months expressed in years
  let interest_rate : ℚ := 16 -- 16% per annum
  bill_amount true_discount due_time interest_rate = 1680 :=
by sorry


end NUMINAMATH_CALUDE_specific_bill_amount_l890_89063


namespace NUMINAMATH_CALUDE_marcia_wardrobe_cost_l890_89014

/-- Calculates the total cost of Marcia's wardrobe --/
def wardrobeCost (skirtPrice blousePrice pantPrice : ℚ) 
                 (numSkirts numBlouses numPants : ℕ) : ℚ :=
  let skirtCost := skirtPrice * numSkirts
  let blouseCost := blousePrice * numBlouses
  let pantCost := pantPrice * (numPants - 1) + (pantPrice / 2)
  skirtCost + blouseCost + pantCost

/-- Proves that the total cost of Marcia's wardrobe is $180.00 --/
theorem marcia_wardrobe_cost :
  wardrobeCost 20 15 30 3 5 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_marcia_wardrobe_cost_l890_89014


namespace NUMINAMATH_CALUDE_platform_length_l890_89019

/-- Given a train crossing a platform and a signal pole, calculate the platform length -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_platform = 54) 
  (h3 : time_pole = 18) : 
  ∃ platform_length : ℝ, platform_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l890_89019


namespace NUMINAMATH_CALUDE_parabola_sum_is_vertical_l890_89094

/-- Original parabola function -/
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Reflected and left-translated parabola -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * (x + 3) + c

/-- Reflected and right-translated parabola -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * (x - 4) + c

/-- Sum of f and g -/
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_sum_is_vertical (a b c : ℝ) :
  ∃ A C : ℝ, ∀ x : ℝ, f_plus_g a b c x = A * x^2 + C :=
sorry

end NUMINAMATH_CALUDE_parabola_sum_is_vertical_l890_89094


namespace NUMINAMATH_CALUDE_problem_solution_l890_89086

def f (x : ℝ) := x^2 - 2*x

theorem problem_solution :
  (∀ x : ℝ, (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ x ≥ 3 ∨ x = 0)) ∧
  (∀ x a : ℝ, |x - a| < 1 → |f x - f a| < 2*|a| + 3) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l890_89086


namespace NUMINAMATH_CALUDE_light_ray_reflection_angle_l890_89066

/-- Regular hexagon with mirrored inner surface -/
structure RegularHexagon :=
  (side : ℝ)
  (A B C D E F : ℝ × ℝ)

/-- Light ray path in the hexagon -/
structure LightRayPath (hex : RegularHexagon) :=
  (M N : ℝ × ℝ)
  (start_at_A : M.1 = hex.A.1 ∨ M.2 = hex.A.2)
  (end_at_D : N.1 = hex.D.1 ∨ N.2 = hex.D.2)
  (on_sides : (M.1 = hex.A.1 ∨ M.1 = hex.B.1 ∨ M.2 = hex.A.2 ∨ M.2 = hex.B.2) ∧
              (N.1 = hex.B.1 ∨ N.1 = hex.C.1 ∨ N.2 = hex.B.2 ∨ N.2 = hex.C.2))

/-- Main theorem -/
theorem light_ray_reflection_angle (hex : RegularHexagon) (path : LightRayPath hex) :
  let tan_EAM := (hex.E.2 - hex.A.2) / (hex.E.1 - hex.A.1)
  tan_EAM = 1 / (3 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_light_ray_reflection_angle_l890_89066


namespace NUMINAMATH_CALUDE_factorization_of_four_minus_n_squared_l890_89007

theorem factorization_of_four_minus_n_squared (n : ℝ) : 4 - n^2 = (2 + n) * (2 - n) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_four_minus_n_squared_l890_89007


namespace NUMINAMATH_CALUDE_amys_house_height_l890_89091

/-- The height of Amy's house given shadow lengths -/
theorem amys_house_height (house_shadow : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h1 : house_shadow = 63)
  (h2 : tree_height = 14)
  (h3 : tree_shadow = 28)
  : ∃ (house_height : ℝ), 
    (house_height / tree_height = house_shadow / tree_shadow) ∧ 
    (round house_height = 32) := by
  sorry

end NUMINAMATH_CALUDE_amys_house_height_l890_89091


namespace NUMINAMATH_CALUDE_cos_alpha_value_l890_89033

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α - Real.pi/4) = -Real.sqrt 2 / 10)
  (h2 : 0 < α) (h3 : α < Real.pi/2) : 
  Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l890_89033


namespace NUMINAMATH_CALUDE_purple_shoes_count_l890_89078

theorem purple_shoes_count (total : ℕ) (blue : ℕ) (h1 : total = 1250) (h2 : blue = 540) :
  let remaining := total - blue
  let purple := remaining / 2
  purple = 355 := by
sorry

end NUMINAMATH_CALUDE_purple_shoes_count_l890_89078


namespace NUMINAMATH_CALUDE_eight_stairs_climbs_l890_89025

def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | n + 3 => climbStairs (n + 2) + climbStairs (n + 1) + climbStairs n

theorem eight_stairs_climbs : climbStairs 8 = 81 := by
  sorry

end NUMINAMATH_CALUDE_eight_stairs_climbs_l890_89025


namespace NUMINAMATH_CALUDE_triangle_angle_B_l890_89048

theorem triangle_angle_B (A B C : Real) (a b c : Real) : 
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a * Real.sin B * Real.cos C + c * Real.sin B * Real.cos A = 1/2 * b ∧
  a > b →
  B = π/6 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l890_89048


namespace NUMINAMATH_CALUDE_smallest_stair_count_l890_89035

theorem smallest_stair_count (n : ℕ) : 
  (n > 20 ∧ n % 3 = 1 ∧ n % 5 = 4) → 
  (∀ m : ℕ, m > 20 ∧ m % 3 = 1 ∧ m % 5 = 4 → m ≥ n) → 
  n = 34 := by
sorry

end NUMINAMATH_CALUDE_smallest_stair_count_l890_89035


namespace NUMINAMATH_CALUDE_collinear_vectors_magnitude_l890_89037

def a : ℝ × ℝ := (1, 2)
def b (k : ℝ) : ℝ × ℝ := (-2, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = t • w ∨ w = t • v

theorem collinear_vectors_magnitude (k : ℝ) :
  collinear a (b k) →
  ‖(3 • a) + (b k)‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_magnitude_l890_89037


namespace NUMINAMATH_CALUDE_two_male_two_female_selection_methods_at_least_one_male_one_female_selection_methods_l890_89036

-- Define the number of female and male students
def num_female : ℕ := 5
def num_male : ℕ := 4

-- Define the number of students to be selected
def num_selected : ℕ := 4

-- Theorem for scenario 1
theorem two_male_two_female_selection_methods : 
  (num_male.choose 2 * num_female.choose 2) * num_selected.factorial = 1440 := by sorry

-- Theorem for scenario 2
theorem at_least_one_male_one_female_selection_methods :
  (num_male.choose 1 * num_female.choose 3 + 
   num_male.choose 2 * num_female.choose 2 + 
   num_male.choose 3 * num_female.choose 1) * num_selected.factorial = 2880 := by sorry

end NUMINAMATH_CALUDE_two_male_two_female_selection_methods_at_least_one_male_one_female_selection_methods_l890_89036
