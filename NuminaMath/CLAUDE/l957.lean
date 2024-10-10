import Mathlib

namespace sufficient_not_necessary_condition_l957_95733

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (abs m < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → abs m < 1) :=
by sorry

end sufficient_not_necessary_condition_l957_95733


namespace circumscribed_circle_diameter_l957_95743

/-- The diameter of a triangle's circumscribed circle, given one side and its opposite angle. -/
theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h_side : side = 18) (h_angle : angle = π/4) :
  let diameter := side / Real.sin angle
  diameter = 18 * Real.sqrt 2 := by sorry

end circumscribed_circle_diameter_l957_95743


namespace line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l957_95729

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem for the first line
theorem line_through_P_and_origin : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x = P.1 ∧ y = P.2 ∨ x = 0 ∧ y = 0) → 
    y = m*x + b ∧ 
    (∀ (x y : ℝ), y = m*x + b ↔ x + y = 0) :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ : 
  ∃ (m b : ℝ), ∀ (x y : ℝ),
    (x = P.1 ∧ y = P.2) →
    y = m*x + b ∧
    m * (1/2) = -1 ∧
    (∀ (x y : ℝ), y = m*x + b ↔ 2*x + y + 2 = 0) :=
sorry

end line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l957_95729


namespace prob_condition_one_before_two_l957_95786

/-- Represents the state of ball draws as a sorted list of integers -/
def DrawState := List Nat

/-- The probability of reaching a certain draw state -/
def StateProbability := DrawState → ℚ

/-- Checks if some ball has been drawn at least three times -/
def conditionOne (state : DrawState) : Prop :=
  state.head! ≥ 3

/-- Checks if every ball has been drawn at least once -/
def conditionTwo (state : DrawState) : Prop :=
  state.length = 3 ∧ state.all (· > 0)

/-- The probability of condition one occurring before condition two -/
def probConditionOneBeforeTwo (probMap : StateProbability) : ℚ :=
  probMap [3, 0, 0] + probMap [3, 1, 0] + probMap [3, 2, 0]

theorem prob_condition_one_before_two :
  ∃ (probMap : StateProbability),
    (∀ state, conditionOne state → conditionTwo state → probMap state = 0) →
    (probMap [0, 0, 0] = 1) →
    (∀ state, probMap state ≥ 0) →
    (∀ state, probMap state ≤ 1) →
    probConditionOneBeforeTwo probMap = 13 / 27 := by
  sorry

end prob_condition_one_before_two_l957_95786


namespace painted_cells_theorem_l957_95727

/-- Represents a rectangular grid with painted and unpainted cells. -/
structure PaintedRectangle where
  rows : Nat
  cols : Nat
  painted_cells : Nat

/-- Checks if the given PaintedRectangle satisfies the problem conditions. -/
def is_valid_painting (rect : PaintedRectangle) : Prop :=
  ∃ k l : Nat,
    rect.rows = 2 * k + 1 ∧
    rect.cols = 2 * l + 1 ∧
    k * l = 74 ∧
    rect.painted_cells = (2 * k + 1) * (2 * l + 1) - 74

/-- The main theorem stating the only possible numbers of painted cells. -/
theorem painted_cells_theorem :
  ∀ rect : PaintedRectangle,
    is_valid_painting rect →
    (rect.painted_cells = 373 ∨ rect.painted_cells = 301) :=
by sorry

end painted_cells_theorem_l957_95727


namespace batting_cage_pitches_per_token_l957_95722

/-- The number of pitches per token at a batting cage -/
def pitches_per_token : ℕ := 15

/-- Macy's number of tokens -/
def macy_tokens : ℕ := 11

/-- Piper's number of tokens -/
def piper_tokens : ℕ := 17

/-- Macy's number of hits -/
def macy_hits : ℕ := 50

/-- Piper's number of hits -/
def piper_hits : ℕ := 55

/-- Total number of missed pitches -/
def total_misses : ℕ := 315

theorem batting_cage_pitches_per_token :
  (macy_tokens + piper_tokens) * pitches_per_token =
  macy_hits + piper_hits + total_misses :=
by sorry

end batting_cage_pitches_per_token_l957_95722


namespace geometric_series_ratio_l957_95781

theorem geometric_series_ratio (a r : ℝ) (hr : |r| < 1) :
  (a / (1 - r) = 16 * (a * r^2 / (1 - r))) → |r| = 1/4 := by
  sorry

end geometric_series_ratio_l957_95781


namespace shoe_shirt_cost_difference_is_three_l957_95787

/-- The cost difference between a pair of shoes and a shirt -/
def shoe_shirt_cost_difference : ℝ :=
  let shirt_cost : ℝ := 7
  let shoe_cost : ℝ := shirt_cost + shoe_shirt_cost_difference
  let bag_cost : ℝ := (2 * shirt_cost + shoe_cost) / 2
  let total_cost : ℝ := 2 * shirt_cost + shoe_cost + bag_cost
  shoe_shirt_cost_difference

/-- Theorem stating the cost difference between a pair of shoes and a shirt -/
theorem shoe_shirt_cost_difference_is_three :
  shoe_shirt_cost_difference = 3 := by
  sorry

#eval shoe_shirt_cost_difference

end shoe_shirt_cost_difference_is_three_l957_95787


namespace divisibility_n_plus_seven_l957_95732

theorem divisibility_n_plus_seven (n : ℕ+) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 := by
  sorry

end divisibility_n_plus_seven_l957_95732


namespace one_statement_implies_negation_l957_95740

theorem one_statement_implies_negation (p q r : Prop) : 
  let statement1 := p ∧ q ∧ ¬r
  let statement2 := p ∧ ¬q ∧ r
  let statement3 := ¬p ∧ q ∧ ¬r
  let statement4 := ¬p ∧ ¬q ∧ r
  let negation := ¬((p ∧ q) ∨ r)
  ∃! x : Fin 4, match x with
    | 0 => statement1 → negation
    | 1 => statement2 → negation
    | 2 => statement3 → negation
    | 3 => statement4 → negation
  := by sorry

end one_statement_implies_negation_l957_95740


namespace inequality_solution_range_l957_95795

theorem inequality_solution_range (a : ℝ) (h_a : a > 0) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ 
   Real.exp 2 * Real.log a - Real.exp 2 * x + x - Real.log a ≥ 2 * a / Real.exp x - 2) ↔
  a ∈ Set.Icc (1 / Real.exp 1) (Real.exp 4) :=
sorry

end inequality_solution_range_l957_95795


namespace diagonals_difference_octagon_heptagon_l957_95776

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem diagonals_difference_octagon_heptagon :
  num_diagonals octagon_sides - num_diagonals heptagon_sides = 6 := by
  sorry

end diagonals_difference_octagon_heptagon_l957_95776


namespace tangent_line_parallel_point_exists_l957_95704

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_parallel_point_exists :
  ∃ (x y : ℝ), f x = y ∧ f' x = 4 ∧ x = 1 := by sorry

end tangent_line_parallel_point_exists_l957_95704


namespace completing_square_transform_l957_95763

theorem completing_square_transform (x : ℝ) :
  (x^2 - 8*x + 5 = 0) ↔ ((x - 4)^2 = 11) :=
by sorry

end completing_square_transform_l957_95763


namespace min_surface_area_cubic_pile_l957_95744

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the surface area of a cube given its side length -/
def cubeSurfaceArea (sideLength : ℕ) : ℕ :=
  6 * sideLength * sideLength

/-- Theorem: The minimum surface area of a cubic pile of bricks -/
theorem min_surface_area_cubic_pile (brick : BrickDimensions)
  (h1 : brick.length = 25)
  (h2 : brick.width = 15)
  (h3 : brick.height = 5) :
  ∃ (sideLength : ℕ), cubeSurfaceArea sideLength = 33750 ∧
    ∀ (otherSideLength : ℕ), cubeSurfaceArea otherSideLength ≥ 33750 := by
  sorry

end min_surface_area_cubic_pile_l957_95744


namespace triangle_reflection_area_l957_95761

/-- The area of the union of a triangle and its reflection --/
theorem triangle_reflection_area : 
  let A : ℝ × ℝ := (3, 4)
  let B : ℝ × ℝ := (5, -2)
  let C : ℝ × ℝ := (6, 2)
  let reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.1, 2 - p.2)
  let A' := reflect A
  let B' := reflect B
  let C' := reflect C
  let area (p q r : ℝ × ℝ) : ℝ := 
    (1/2) * abs (p.1 * (q.2 - r.2) + q.1 * (r.2 - p.2) + r.1 * (p.2 - q.2))
  area A B C + area A' B' C' = 11 := by
sorry


end triangle_reflection_area_l957_95761


namespace parabola_sine_no_intersection_l957_95794

theorem parabola_sine_no_intersection :
  ∀ x : ℝ, x^2 - x + 5.35 > 2 * Real.sin x + 3 := by
sorry

end parabola_sine_no_intersection_l957_95794


namespace simplify_expressions_l957_95749

theorem simplify_expressions :
  (2 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + Real.sqrt 48 = -2 * Real.sqrt 3) ∧
  ((Real.sqrt 3 - Real.pi)^0 - (Real.sqrt 20 - Real.sqrt 15) / Real.sqrt 5 + (-1)^2017 = Real.sqrt 3 - 2) := by
  sorry

end simplify_expressions_l957_95749


namespace f_properties_l957_95766

def f (x y : ℝ) : ℝ × ℝ := (x - y, x + y)

theorem f_properties :
  (f 3 5 = (-2, 8)) ∧ (f 4 1 = (3, 5)) := by
  sorry

end f_properties_l957_95766


namespace erasers_remaining_l957_95751

/-- The number of erasers left in a box after some are removed -/
def erasers_left (initial : ℕ) (removed : ℕ) : ℕ := initial - removed

/-- Theorem: Given 69 initial erasers and 54 removed, 15 erasers are left -/
theorem erasers_remaining : erasers_left 69 54 = 15 := by
  sorry

end erasers_remaining_l957_95751


namespace hamburger_sales_average_l957_95754

theorem hamburger_sales_average (total_hamburgers : ℕ) (days_in_week : ℕ) 
  (h1 : total_hamburgers = 63)
  (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
sorry

end hamburger_sales_average_l957_95754


namespace complex_division_simplification_l957_95702

theorem complex_division_simplification : 
  (2 + Complex.I) / (1 - 2 * Complex.I) = Complex.I := by sorry

end complex_division_simplification_l957_95702


namespace max_difference_consecutive_means_l957_95709

theorem max_difference_consecutive_means (a b : ℕ) : 
  0 < a ∧ 0 < b ∧ a < 1000 ∧ b < 1000 →
  ∃ (k : ℕ), (a + b) / 2 = 2 * k + 1 ∧ Real.sqrt (a * b) = 2 * k - 1 →
  a - b ≤ 62 := by
sorry

end max_difference_consecutive_means_l957_95709


namespace sprint_team_distance_l957_95717

/-- Given a sprint team with a certain number of people, where each person runs a fixed distance,
    calculate the total distance run by the team. -/
def total_distance (team_size : ℝ) (distance_per_person : ℝ) : ℝ :=
  team_size * distance_per_person

/-- Theorem: A sprint team of 150.0 people, where each person runs 5.0 miles,
    will run a total of 750.0 miles. -/
theorem sprint_team_distance :
  total_distance 150.0 5.0 = 750.0 := by
  sorry

end sprint_team_distance_l957_95717


namespace sum_of_digits_seven_power_fifteen_l957_95789

/-- The sum of the tens digit and the ones digit of 7^15 is 7 -/
theorem sum_of_digits_seven_power_fifteen : ∃ (a b : ℕ), 
  7^15 % 100 = 10 * a + b ∧ a + b = 7 := by sorry

end sum_of_digits_seven_power_fifteen_l957_95789


namespace remainder_zero_l957_95773

/-- A polynomial of degree 5 with real coefficients -/
structure Poly5 (D E F G H : ℝ) where
  q : ℝ → ℝ
  eq : ∀ x, q x = D * x^5 + E * x^4 + F * x^3 + G * x^2 + H * x + 2

/-- The remainder theorem for polynomials -/
axiom remainder_theorem {p : ℝ → ℝ} {a r : ℝ} :
  (∀ x, ∃ q, p x = (x - a) * q + r) ↔ p a = r

/-- Main theorem: If the remainder of q(x) divided by (x - 4) is 15,
    then the remainder of q(x) divided by (x + 4) is 0 -/
theorem remainder_zero {D E F G H : ℝ} (p : Poly5 D E F G H) :
  p.q 4 = 15 → p.q (-4) = 0 := by
  sorry

end remainder_zero_l957_95773


namespace jane_daffodil_bulbs_l957_95718

/-- The number of daffodil bulbs Jane planted -/
def daffodil_bulbs : ℕ := 30

theorem jane_daffodil_bulbs :
  let tulip_bulbs : ℕ := 20
  let iris_bulbs : ℕ := tulip_bulbs / 2
  let crocus_bulbs : ℕ := 3 * daffodil_bulbs
  let total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs
  let earnings_per_bulb : ℚ := 1/2
  let total_earnings : ℚ := 75
  total_earnings = earnings_per_bulb * total_bulbs :=
by sorry


end jane_daffodil_bulbs_l957_95718


namespace smallest_number_with_distinct_sums_ending_in_two_l957_95715

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def digitSumSequence (x : Nat) : List Nat :=
  [x, sumOfDigits x, sumOfDigits (sumOfDigits x), sumOfDigits (sumOfDigits (sumOfDigits x))]

theorem smallest_number_with_distinct_sums_ending_in_two :
  ∀ y : Nat, y < 2999 →
    ¬(List.Pairwise (· ≠ ·) (digitSumSequence y) ∧
      (digitSumSequence y).getLast? = some 2) ∧
    (List.Pairwise (· ≠ ·) (digitSumSequence 2999) ∧
     (digitSumSequence 2999).getLast? = some 2) :=
by sorry

end smallest_number_with_distinct_sums_ending_in_two_l957_95715


namespace harriet_driving_speed_l957_95760

/-- Harriet's driving problem -/
theorem harriet_driving_speed 
  (total_time : ℝ) 
  (time_to_b : ℝ) 
  (speed_back : ℝ) : 
  total_time = 5 → 
  time_to_b = 192 / 60 → 
  speed_back = 160 → 
  (total_time - time_to_b) * speed_back / time_to_b = 90 := by
  sorry

end harriet_driving_speed_l957_95760


namespace stratified_sampling_most_appropriate_l957_95752

/-- Represents the different types of sampling methods. -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a school with different student groups. -/
structure School where
  elementaryStudents : ℕ
  juniorHighStudents : ℕ
  highSchoolStudents : ℕ

/-- Determines if there are significant differences between student groups. -/
def hasDifferences (s : School) : Prop :=
  sorry -- Definition of significant differences

/-- Determines the most appropriate sampling method given a school and sample size. -/
def mostAppropriateSamplingMethod (s : School) (sampleSize : ℕ) : SamplingMethod :=
  sorry -- Definition of most appropriate sampling method

/-- Theorem stating that stratified sampling is the most appropriate method for the given conditions. -/
theorem stratified_sampling_most_appropriate (s : School) (sampleSize : ℕ) :
  s.elementaryStudents = 125 →
  s.juniorHighStudents = 280 →
  s.highSchoolStudents = 95 →
  sampleSize = 100 →
  hasDifferences s →
  mostAppropriateSamplingMethod s sampleSize = SamplingMethod.Stratified :=
by sorry

end stratified_sampling_most_appropriate_l957_95752


namespace man_son_age_ratio_l957_95792

/-- The ratio of a man's age to his son's age after two years, given their current ages. -/
theorem man_son_age_ratio (son_age : ℕ) (man_age : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end man_son_age_ratio_l957_95792


namespace book_reading_theorem_l957_95797

def days_to_read (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem book_reading_theorem :
  let num_books := 18
  let start_day := 0  -- 0 represents Sunday
  let total_days := days_to_read num_books
  day_of_week start_day total_days = 3  -- 3 represents Wednesday
:= by sorry

end book_reading_theorem_l957_95797


namespace boxes_given_away_l957_95769

def total_cupcakes : ℕ := 53
def cupcakes_left_at_home : ℕ := 2
def cupcakes_per_box : ℕ := 3

theorem boxes_given_away : 
  (total_cupcakes - cupcakes_left_at_home) / cupcakes_per_box = 17 := by
  sorry

end boxes_given_away_l957_95769


namespace unique_number_with_special_divisor_property_l957_95723

theorem unique_number_with_special_divisor_property :
  ∃! (N : ℕ), 
    N > 0 ∧
    (∃ (m : ℕ), 
      m > 0 ∧ 
      m < N ∧
      N % m = 0 ∧
      (∀ (d : ℕ), d > 0 → d < N → N % d = 0 → d ≤ m) ∧
      (∃ (k : ℕ), N + m = 10^k)) ∧
    N = 75 :=
by sorry

end unique_number_with_special_divisor_property_l957_95723


namespace coffee_order_total_cost_l957_95735

/-- The total cost of a coffee order -/
def coffee_order_cost (drip_coffee_price : ℝ) (drip_coffee_quantity : ℕ)
                      (espresso_price : ℝ) (espresso_quantity : ℕ)
                      (latte_price : ℝ) (latte_quantity : ℕ)
                      (vanilla_syrup_price : ℝ) (vanilla_syrup_quantity : ℕ)
                      (cold_brew_price : ℝ) (cold_brew_quantity : ℕ)
                      (cappuccino_price : ℝ) (cappuccino_quantity : ℕ) : ℝ :=
  drip_coffee_price * drip_coffee_quantity +
  espresso_price * espresso_quantity +
  latte_price * latte_quantity +
  vanilla_syrup_price * vanilla_syrup_quantity +
  cold_brew_price * cold_brew_quantity +
  cappuccino_price * cappuccino_quantity

/-- The theorem stating that the given coffee order costs $25.00 -/
theorem coffee_order_total_cost :
  coffee_order_cost 2.25 2 3.50 1 4.00 2 0.50 1 2.50 2 3.50 1 = 25 := by
  sorry

end coffee_order_total_cost_l957_95735


namespace cubic_sum_problem_l957_95741

theorem cubic_sum_problem (a b c : ℝ) 
  (sum_condition : a + b + c = 7)
  (product_sum_condition : a * b + a * c + b * c = 9)
  (product_condition : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end cubic_sum_problem_l957_95741


namespace hotel_room_encoding_l957_95747

theorem hotel_room_encoding (x : ℕ) : 
  1 ≤ x ∧ x ≤ 30 → x % 5 = 3 → x % 7 = 6 → x = 13 := by
  sorry

end hotel_room_encoding_l957_95747


namespace triangle_max_value_l957_95736

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area is √3 and cos(C) / cos(B) = c / (2a - b),
    the maximum value of 1/(b+1) + 9/(a+9) is 3/5. -/
theorem triangle_max_value (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  Real.cos C / Real.cos B = c / (2*a - b) →
  (∃ (x : ℝ), (1/(b+1) + 9/(a+9) ≤ x) ∧ 
   (∀ (y : ℝ), 1/(b+1) + 9/(a+9) ≤ y → x ≤ y)) →
  (1/(b+1) + 9/(a+9)) ≤ 3/5 :=
by sorry

end triangle_max_value_l957_95736


namespace max_eggs_per_basket_l957_95798

def purple_eggs : ℕ := 30
def yellow_eggs : ℕ := 45
def min_eggs_per_basket : ℕ := 5

theorem max_eggs_per_basket : 
  ∃ (n : ℕ), n ≥ min_eggs_per_basket ∧ 
  purple_eggs % n = 0 ∧ 
  yellow_eggs % n = 0 ∧
  ∀ (m : ℕ), m > n → 
    (m ≥ min_eggs_per_basket ∧ 
     purple_eggs % m = 0 ∧ 
     yellow_eggs % m = 0) → False :=
by sorry

end max_eggs_per_basket_l957_95798


namespace complex_product_example_l957_95783

theorem complex_product_example : (1 + Complex.I) * (2 + Complex.I) = 1 + 3 * Complex.I := by
  sorry

end complex_product_example_l957_95783


namespace min_sum_of_parallel_vectors_l957_95784

-- Define the vectors
def m (a : ℝ) : ℝ × ℝ := (a, a - 4)
def n (b : ℝ) : ℝ × ℝ := (b, 1 - b)

-- Define parallelism condition
def are_parallel (a b : ℝ) : Prop :=
  a * (1 - b) = b * (a - 4)

theorem min_sum_of_parallel_vectors (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_parallel : are_parallel a b) :
  a + b ≥ 9/2 ∧ (a + b = 9/2 ↔ a = 4 ∧ b = 2) := by
  sorry


end min_sum_of_parallel_vectors_l957_95784


namespace perpendicular_vectors_m_value_l957_95708

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a = (-1, 2) ∧ b = (m, 1)) :
  a.1 * b.1 + a.2 * b.2 = 0 → m = 2 := by
  sorry

end perpendicular_vectors_m_value_l957_95708


namespace kate_pen_purchase_l957_95788

theorem kate_pen_purchase (pen_cost : ℝ) (kate_money : ℝ) : 
  pen_cost = 30 → kate_money = pen_cost / 3 → pen_cost - kate_money = 20 := by
  sorry

end kate_pen_purchase_l957_95788


namespace interesting_numbers_l957_95707

def is_interesting (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 90 ∧
  ∃ (p : ℕ) (k : ℕ), 
    Nat.Prime p ∧ 
    k ≥ 2 ∧ 
    n = p^k

theorem interesting_numbers : 
  {n : ℕ | is_interesting n} = {25, 27, 32, 49, 64, 81} :=
by sorry

end interesting_numbers_l957_95707


namespace bca_equals_341_l957_95785

def repeating_decimal_bc (b c : ℕ) : ℚ :=
  (10 * b + c : ℚ) / 99

def repeating_decimal_bcabc (b c a : ℕ) : ℚ :=
  (10000 * b + 1000 * c + 100 * a + 10 * b + c : ℚ) / 99999

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem bca_equals_341 (b c a : ℕ) 
  (hb : is_digit b) (hc : is_digit c) (ha : is_digit a)
  (h_eq : repeating_decimal_bc b c + repeating_decimal_bcabc b c a = 41 / 111) :
  100 * b + 10 * c + a = 341 := by
sorry

end bca_equals_341_l957_95785


namespace expression_values_l957_95706

theorem expression_values (a b : ℝ) : 
  (∀ x : ℝ, |a| ≤ |x|) → (b * b = 1) → 
  (|a - 2| - b^2023 = 1 ∨ |a - 2| - b^2023 = 3) := by
  sorry

end expression_values_l957_95706


namespace proportion_problem_l957_95791

theorem proportion_problem (x y : ℝ) : 
  x / 5 = 5 / 6 → x = 0.9 → y / x = 5 / 6 → y = 0.75 := by sorry

end proportion_problem_l957_95791


namespace total_sums_attempted_l957_95750

theorem total_sums_attempted (right_sums wrong_sums total_sums : ℕ) : 
  wrong_sums = 2 * right_sums →
  right_sums = 8 →
  total_sums = right_sums + wrong_sums →
  total_sums = 24 := by
sorry

end total_sums_attempted_l957_95750


namespace cyclists_meeting_time_l957_95742

/-- Two cyclists meeting problem -/
theorem cyclists_meeting_time
  (b : ℝ) -- distance between towns A and B in km
  (peter_speed : ℝ) -- Peter's speed in km/h
  (john_speed : ℝ) -- John's speed in km/h
  (h1 : peter_speed = 7) -- Peter's speed is 7 km/h
  (h2 : john_speed = 5) -- John's speed is 5 km/h
  : ∃ p : ℝ, p = b / (peter_speed + john_speed) ∧ p = b / 12 :=
by sorry

end cyclists_meeting_time_l957_95742


namespace max_value_trig_function_l957_95793

theorem max_value_trig_function :
  (∀ x : ℝ, 3 * Real.sin x - 3 * Real.sqrt 3 * Real.cos x ≤ 6) ∧
  (∃ x : ℝ, 3 * Real.sin x - 3 * Real.sqrt 3 * Real.cos x = 6) := by
  sorry

end max_value_trig_function_l957_95793


namespace problem_solution_l957_95721

theorem problem_solution (x : ℚ) : 5 * x - 8 = 12 * x + 15 → 5 * (x + 4) = 25 / 7 := by
  sorry

end problem_solution_l957_95721


namespace both_samples_stratified_l957_95755

/-- Represents a sample of students -/
structure Sample :=
  (numbers : List Nat)

/-- Represents the school population -/
structure School :=
  (total_students : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)

/-- Checks if a sample is valid for stratified sampling -/
def is_valid_stratified_sample (school : School) (sample : Sample) : Prop :=
  sample.numbers.length = 10 ∧
  sample.numbers.all (λ n => n > 0 ∧ n ≤ school.total_students) ∧
  sample.numbers.Nodup

/-- The given school configuration -/
def junior_high : School :=
  { total_students := 300
  , first_grade := 120
  , second_grade := 90
  , third_grade := 90 }

/-- Sample ① -/
def sample1 : Sample :=
  { numbers := [7, 37, 67, 97, 127, 157, 187, 217, 247, 277] }

/-- Sample ③ -/
def sample3 : Sample :=
  { numbers := [11, 41, 71, 101, 131, 161, 191, 221, 251, 281] }

theorem both_samples_stratified :
  is_valid_stratified_sample junior_high sample1 ∧
  is_valid_stratified_sample junior_high sample3 := by
  sorry

end both_samples_stratified_l957_95755


namespace three_correct_statements_l957_95765

theorem three_correct_statements : 
  (0 ∉ (∅ : Set ℕ)) ∧ 
  ((∅ : Set ℕ) ⊆ {1,2}) ∧ 
  ({(x,y) : ℝ × ℝ | 2*x + y = 10 ∧ 3*x - y = 5} ≠ {3,4}) ∧ 
  (∀ A B : Set α, A ⊆ B → A ∩ B = A) :=
by sorry

end three_correct_statements_l957_95765


namespace zeros_bound_l957_95799

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem zeros_bound (a : ℝ) :
  (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → (f a x ≠ 0 ∨ f a y ≠ 0 ∨ f a z ≠ 0)) →
  a ≤ 3 := by
  sorry

end zeros_bound_l957_95799


namespace rainfall_rate_l957_95739

/-- Rainfall problem statement -/
theorem rainfall_rate (monday_hours monday_rate tuesday_hours wednesday_hours total_rainfall : ℝ) 
  (h1 : monday_hours = 7)
  (h2 : monday_rate = 1)
  (h3 : tuesday_hours = 4)
  (h4 : wednesday_hours = 2)
  (h5 : total_rainfall = 23)
  : ∃ tuesday_rate : ℝ, 
    monday_hours * monday_rate + tuesday_hours * tuesday_rate + wednesday_hours * (2 * tuesday_rate) = total_rainfall ∧ 
    tuesday_rate = 2 := by
  sorry

end rainfall_rate_l957_95739


namespace cos_330_degrees_l957_95762

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l957_95762


namespace function_range_and_triangle_area_l957_95759

noncomputable def f (x : ℝ) := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem function_range_and_triangle_area 
  (A B C : ℝ) (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), f x ∈ Set.Icc 0 (Real.sqrt 3)) ∧
  (f (A / 2) = Real.sqrt 3 / 2) ∧
  (a = 4) ∧
  (b + c = 5) →
  (Set.range (fun x => f x) = Set.Icc 0 (Real.sqrt 3)) ∧
  (1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 4) :=
by sorry

end function_range_and_triangle_area_l957_95759


namespace stair_climbing_and_descending_l957_95713

def climbStairs (n : ℕ) : ℕ :=
  if n ≤ 2 then n else climbStairs (n - 1) + climbStairs (n - 2)

def descendStairs (n : ℕ) : ℕ := 2^(n - 1)

theorem stair_climbing_and_descending :
  (climbStairs 10 = 89) ∧ (descendStairs 10 = 512) := by
  sorry

end stair_climbing_and_descending_l957_95713


namespace seconds_in_day_l957_95730

/-- Represents the number of minutes in a day on the island of Misfortune -/
def minutes_per_day : ℕ := 77

/-- Represents the number of seconds in an hour on the island of Misfortune -/
def seconds_per_hour : ℕ := 91

/-- Theorem stating that there are 1001 seconds in a day on the island of Misfortune -/
theorem seconds_in_day : 
  ∃ (hours_per_day minutes_per_hour seconds_per_minute : ℕ), 
    hours_per_day * minutes_per_hour = minutes_per_day ∧
    minutes_per_hour * seconds_per_minute = seconds_per_hour ∧
    hours_per_day * minutes_per_hour * seconds_per_minute = 1001 :=
by
  sorry


end seconds_in_day_l957_95730


namespace intersection_parallel_to_l_l957_95703

structure GeometricSpace where
  Line : Type
  Plane : Type
  perpendicular : Line → Line → Prop
  perpendicular_line_plane : Line → Plane → Prop
  in_plane : Line → Plane → Prop
  skew : Line → Line → Prop
  intersect : Plane → Plane → Prop
  parallel : Line → Line → Prop
  intersection_line : Plane → Plane → Line

variable (S : GeometricSpace)

theorem intersection_parallel_to_l 
  (m n l : S.Line) (α β : S.Plane) :
  S.skew m n →
  S.perpendicular_line_plane m α →
  S.perpendicular_line_plane n β →
  S.perpendicular l m →
  S.perpendicular l n →
  ¬S.in_plane l α →
  ¬S.in_plane l β →
  S.intersect α β ∧ S.parallel l (S.intersection_line α β) :=
sorry

end intersection_parallel_to_l_l957_95703


namespace cards_distribution_l957_95768

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end cards_distribution_l957_95768


namespace greatest_value_quadratic_inequality_l957_95728

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 7 ∧
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 12*x_max + 35 ≤ 0) :=
by sorry

end greatest_value_quadratic_inequality_l957_95728


namespace xiao_ming_run_distance_l957_95779

/-- The distance between two adjacent trees in meters -/
def tree_spacing : ℕ := 6

/-- The number of the last tree Xiao Ming runs to -/
def last_tree : ℕ := 200

/-- The total distance Xiao Ming runs in meters -/
def total_distance : ℕ := (last_tree - 1) * tree_spacing

theorem xiao_ming_run_distance :
  total_distance = 1194 :=
sorry

end xiao_ming_run_distance_l957_95779


namespace endpoint_sum_l957_95780

/-- Given a line segment with one endpoint (1, -2) and midpoint (5, 4),
    the sum of coordinates of the other endpoint is 19. -/
theorem endpoint_sum (x y : ℝ) : 
  (1 + x) / 2 = 5 ∧ (-2 + y) / 2 = 4 → x + y = 19 := by
  sorry

end endpoint_sum_l957_95780


namespace profit_reached_l957_95771

/-- The number of pencils bought for 6 dollars -/
def pencils_bought : ℕ := 5

/-- The cost in dollars for buying pencils_bought pencils -/
def cost : ℚ := 6

/-- The number of pencils sold for 7 dollars -/
def pencils_sold : ℕ := 4

/-- The revenue in dollars for selling pencils_sold pencils -/
def revenue : ℚ := 7

/-- The target profit in dollars -/
def target_profit : ℚ := 80

/-- The minimum number of pencils that must be sold to reach the target profit -/
def min_pencils_to_sell : ℕ := 146

theorem profit_reached : 
  ∃ (n : ℕ), n ≥ min_pencils_to_sell ∧ 
  n * (revenue / pencils_sold - cost / pencils_bought) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_pencils_to_sell → 
  m * (revenue / pencils_sold - cost / pencils_bought) < target_profit :=
by sorry

end profit_reached_l957_95771


namespace factor_expression_l957_95767

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) := by sorry

end factor_expression_l957_95767


namespace set_b_forms_triangle_l957_95712

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of line segments can form a triangle if they satisfy the triangle inequality. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem set_b_forms_triangle :
  can_form_triangle 8 6 3 := by
  sorry

end set_b_forms_triangle_l957_95712


namespace inequality_range_l957_95778

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 := by sorry

end inequality_range_l957_95778


namespace election_votes_theorem_l957_95714

theorem election_votes_theorem (V : ℕ) (W L : ℕ) : 
  W + L = V →  -- Total votes
  W - L = V / 10 →  -- Initial margin
  (L + 1500) - (W - 1500) = V / 10 →  -- New margin after vote change
  V = 30000 := by
sorry

end election_votes_theorem_l957_95714


namespace odd_function_solution_set_l957_95758

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The solution set of an inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | x*f x - Real.exp (abs x) > 0}

theorem odd_function_solution_set
  (f : ℝ → ℝ)
  (hodd : OddFunction f)
  (hf1 : f 1 = Real.exp 1)
  (hineq : ∀ x ≥ 0, (x - 1) * f x < x * (deriv f x)) :
  SolutionSet f = Set.Ioi 1 ∪ Set.Iic (-1) :=
sorry

end odd_function_solution_set_l957_95758


namespace smallest_winning_number_l957_95700

def game_winner (M : ℕ) : Prop :=
  M ≤ 999 ∧ 
  2 * M < 1000 ∧ 
  6 * M < 1000 ∧ 
  12 * M < 1000 ∧ 
  36 * M > 999

theorem smallest_winning_number : 
  ∃ (M : ℕ), game_winner M ∧ ∀ (N : ℕ), N < M → ¬game_winner N :=
sorry

end smallest_winning_number_l957_95700


namespace johnny_attend_probability_l957_95753

-- Define the probabilities
def p_rain : ℝ := 0.3
def p_sunny : ℝ := 0.5
def p_cloudy : ℝ := 1 - p_rain - p_sunny

def p_attend_given_rain : ℝ := 0.5
def p_attend_given_sunny : ℝ := 0.9
def p_attend_given_cloudy : ℝ := 0.7

-- Define the theorem
theorem johnny_attend_probability :
  p_attend_given_rain * p_rain + p_attend_given_sunny * p_sunny + p_attend_given_cloudy * p_cloudy = 0.74 := by
  sorry


end johnny_attend_probability_l957_95753


namespace max_a_value_l957_95724

theorem max_a_value (a b : ℕ) (ha : 1 < a) (hb : a < b) :
  (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a| + |x - b|) →
  (∀ a' : ℕ, 1 < a' ∧ a' < b ∧ (∃! x : ℝ, -2 * x + 4033 = |x - 1| + |x + a'| + |x - b|) → a' ≤ a) ∧
  a = 4031 :=
by sorry

end max_a_value_l957_95724


namespace stating_max_remaining_coins_product_mod_l957_95719

/-- Represents the grid size -/
def gridSize : Nat := 418

/-- Represents the modulus for the final result -/
def modulus : Nat := 2007

/-- Represents the maximum number of coins that can remain in one quadrant -/
def maxCoinsPerQuadrant : Nat := (gridSize / 2) * (gridSize / 2)

/-- 
Theorem stating that the maximum value of bw (mod 2007) is 1999, 
where b and w are the number of remaining black and white coins respectively 
after applying the removal rules on a 418 × 418 grid.
-/
theorem max_remaining_coins_product_mod (b w : Nat) : 
  b ≤ maxCoinsPerQuadrant → 
  w ≤ maxCoinsPerQuadrant → 
  (b * w) % modulus ≤ 1999 ∧ 
  ∃ (b' w' : Nat), b' ≤ maxCoinsPerQuadrant ∧ w' ≤ maxCoinsPerQuadrant ∧ (b' * w') % modulus = 1999 := by
  sorry

end stating_max_remaining_coins_product_mod_l957_95719


namespace pure_imaginary_quadratic_l957_95782

theorem pure_imaginary_quadratic (a : ℝ) : 
  (Complex.mk (a^2 - 4*a + 3) (a - 1)).im ≠ 0 ∧ (Complex.mk (a^2 - 4*a + 3) (a - 1)).re = 0 → 
  a = 1 ∨ a = 3 := by
sorry

end pure_imaginary_quadratic_l957_95782


namespace real_part_of_reciprocal_l957_95725

theorem real_part_of_reciprocal (z : ℂ) : 
  z ≠ (1 : ℂ) →
  Complex.abs z = 1 → 
  z = Complex.exp (Complex.I * Real.pi / 3) →
  Complex.re (1 / (1 - z)) = 1 / 2 := by
  sorry

end real_part_of_reciprocal_l957_95725


namespace square_perimeter_l957_95756

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s ^ 2 = 625) : 4 * s = 100 := by
  sorry

end square_perimeter_l957_95756


namespace age_ratio_theorem_l957_95774

/-- Represents the ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions from the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.john - 3 = 2 * (a.mary - 3)) ∧ 
  (a.john - 7 = 3 * (a.mary - 7))

/-- The future condition we're looking for -/
def future_ratio (a : Ages) (years : ℕ) : Prop :=
  3 * (a.mary + years) = 2 * (a.john + years)

/-- The main theorem -/
theorem age_ratio_theorem (a : Ages) :
  age_conditions a → ∃ y : ℕ, y = 5 ∧ future_ratio a y := by
  sorry

end age_ratio_theorem_l957_95774


namespace triangle_area_is_two_l957_95775

/-- The first line bounding the triangle -/
def line1 (x y : ℝ) : Prop := y - 2*x = 4

/-- The second line bounding the triangle -/
def line2 (x y : ℝ) : Prop := 2*y - x = 6

/-- The x-axis -/
def x_axis (y : ℝ) : Prop := y = 0

/-- A point is in the triangle if it satisfies the equations of both lines and is above or on the x-axis -/
def in_triangle (x y : ℝ) : Prop :=
  line1 x y ∧ line2 x y ∧ y ≥ 0

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := 2

theorem triangle_area_is_two :
  triangle_area = 2 := by sorry

end triangle_area_is_two_l957_95775


namespace pizza_group_size_l957_95731

theorem pizza_group_size (slices_per_person : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)
  (h1 : slices_per_person = 3)
  (h2 : slices_per_pizza = 9)
  (h3 : num_pizzas = 6) :
  (num_pizzas * (slices_per_pizza / slices_per_person)) = 18 :=
by sorry

end pizza_group_size_l957_95731


namespace money_ratio_to_anna_l957_95737

def total_money : ℕ := 2000
def furniture_cost : ℕ := 400
def money_left : ℕ := 400

def money_after_furniture : ℕ := total_money - furniture_cost
def money_given_to_anna : ℕ := money_after_furniture - money_left

theorem money_ratio_to_anna : 
  (money_given_to_anna : ℚ) / (money_left : ℚ) = 3 := by sorry

end money_ratio_to_anna_l957_95737


namespace base7_divisibility_by_19_l957_95748

def base7ToDecimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem base7_divisibility_by_19 :
  ∃ (x : ℕ), x < 7 ∧ 19 ∣ (base7ToDecimal 2 5 x 3) ∧ x = 1 := by
  sorry

end base7_divisibility_by_19_l957_95748


namespace decimal_fraction_equality_l957_95772

theorem decimal_fraction_equality (b : ℕ+) : 
  (4 * b + 19 : ℚ) / (6 * b + 11) = 19 / 25 → b = 19 := by
  sorry

end decimal_fraction_equality_l957_95772


namespace alloy_mixture_l957_95701

/-- The amount of alloy A in kg -/
def alloy_A : ℝ := 130

/-- The ratio of lead to tin in alloy A -/
def ratio_A : ℚ := 2/3

/-- The ratio of tin to copper in alloy B -/
def ratio_B : ℚ := 3/4

/-- The amount of tin in the new alloy in kg -/
def tin_new : ℝ := 146.57

/-- The amount of alloy B mixed with alloy A in kg -/
def alloy_B : ℝ := 160.33

theorem alloy_mixture :
  alloy_B * (ratio_B / (1 + ratio_B)) + alloy_A * (ratio_A / (1 + ratio_A)) = tin_new := by
  sorry

end alloy_mixture_l957_95701


namespace four_at_three_equals_thirty_l957_95738

-- Define the operation @
def at_op (a b : ℤ) : ℤ := 3 * a^2 - 2 * b^2

-- Theorem statement
theorem four_at_three_equals_thirty : at_op 4 3 = 30 := by
  sorry

end four_at_three_equals_thirty_l957_95738


namespace tv_weather_forecast_is_random_l957_95705

/-- Represents an event in probability theory -/
structure Event where
  (description : String)

/-- Classifies an event as random, certain, or impossible -/
inductive EventClass
  | Random
  | Certain
  | Impossible

/-- An event is random if it can lead to different outcomes, doesn't have a guaranteed outcome, and is feasible to occur -/
def is_random_event (e : Event) : Prop :=
  (∃ (outcome1 outcome2 : String), outcome1 ≠ outcome2) ∧
  ¬(∃ (guaranteed_outcome : String), true) ∧
  (∃ (possible_occurrence : Bool), possible_occurrence = true)

/-- The main theorem: Turning on the TV and watching the weather forecast is a random event -/
theorem tv_weather_forecast_is_random :
  let e : Event := { description := "turning on the TV and watching the weather forecast" }
  is_random_event e → EventClass.Random = EventClass.Random :=
by
  sorry

end tv_weather_forecast_is_random_l957_95705


namespace a_divisibility_l957_95770

/-- Sequence a_n defined recursively -/
def a (k : ℤ) : ℕ → ℤ
  | 0 => 0
  | 1 => k
  | (n + 2) => k^2 * a k (n + 1) - a k n

/-- Theorem stating that a_{n+1} * a_n + 1 divides a_{n+1}^2 + a_n^2 for all n -/
theorem a_divisibility (k : ℤ) (n : ℕ) :
  ∃ m : ℤ, (a k (n + 1))^2 + (a k n)^2 = ((a k (n + 1)) * (a k n) + 1) * m := by
  sorry

end a_divisibility_l957_95770


namespace right_vertex_intersection_l957_95777

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1

-- Define the line
def line (x y a : ℝ) : Prop := y = x - a

-- State the theorem
theorem right_vertex_intersection (a : ℝ) :
  ellipse 3 0 ∧ line 3 0 a → a = 3 := by
  sorry

end right_vertex_intersection_l957_95777


namespace garden_max_area_exists_max_area_garden_l957_95726

/-- Represents a rectangular garden with fencing on three sides --/
structure Garden where
  width : ℝ
  length : ℝ
  fencing : ℝ
  fence_constraint : fencing = 2 * width + length

/-- The area of a rectangular garden --/
def Garden.area (g : Garden) : ℝ := g.width * g.length

/-- The maximum possible area of a garden with 400 feet of fencing --/
def max_garden_area : ℝ := 20000

/-- Theorem stating that the maximum area of a garden with 400 feet of fencing is 20000 square feet --/
theorem garden_max_area :
  ∀ g : Garden, g.fencing = 400 → g.area ≤ max_garden_area :=
by
  sorry

/-- Theorem stating that there exists a garden configuration achieving the maximum area --/
theorem exists_max_area_garden :
  ∃ g : Garden, g.fencing = 400 ∧ g.area = max_garden_area :=
by
  sorry

end garden_max_area_exists_max_area_garden_l957_95726


namespace data_analysis_l957_95711

def data : List ℕ := [11, 10, 11, 13, 11, 13, 15]

def mode (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def variance (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℕ := sorry

theorem data_analysis (d : List ℕ) (h : d = data) : 
  mode d = 11 ∧ 
  mean d = 12 ∧ 
  variance d = 18/7 ∧ 
  median d = 11 := by sorry

end data_analysis_l957_95711


namespace arithmetic_geometric_sequence_l957_95757

theorem arithmetic_geometric_sequence (a b : ℝ) : 
  (2 * a = 1 + b) →  -- arithmetic sequence condition
  (b^2 = a) →        -- geometric sequence condition
  (a ≠ b) →          -- given condition
  (a = 1/4) :=       -- conclusion to prove
by sorry

end arithmetic_geometric_sequence_l957_95757


namespace opposite_of_negative_two_l957_95734

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- Theorem stating that the opposite of -2 is 2
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end opposite_of_negative_two_l957_95734


namespace relationship_between_x_and_y_l957_95746

theorem relationship_between_x_and_y (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
sorry

end relationship_between_x_and_y_l957_95746


namespace calories_per_chip_l957_95710

/-- Represents the number of chips in a bag -/
def chips_per_bag : ℕ := 24

/-- Represents the cost of a bag in dollars -/
def cost_per_bag : ℚ := 2

/-- Represents the total calories Peter wants to consume -/
def total_calories : ℕ := 480

/-- Represents the total amount Peter needs to spend in dollars -/
def total_spent : ℚ := 4

/-- Theorem stating that each chip contains 10 calories -/
theorem calories_per_chip : 
  (total_calories : ℚ) / (total_spent / cost_per_bag * chips_per_bag) = 10 := by
  sorry

end calories_per_chip_l957_95710


namespace unique_prime_cube_plus_two_l957_95745

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The main theorem stating that there is exactly one positive integer n ≥ 2 
    such that n^3 + 2 is prime -/
theorem unique_prime_cube_plus_two :
  ∃! (n : ℕ), n ≥ 2 ∧ isPrime (n^3 + 2) :=
sorry

end unique_prime_cube_plus_two_l957_95745


namespace art_club_teams_l957_95716

theorem art_club_teams (n : ℕ) (h : n.choose 2 = 15) :
  n.choose 4 = 15 := by
  sorry

end art_club_teams_l957_95716


namespace quadratic_roots_property_l957_95790

theorem quadratic_roots_property : ∃ (x y : ℝ), 
  (x + y = 10) ∧ 
  (|x - y| = 6) ∧ 
  (∀ z : ℝ, z^2 - 10*z + 16 = 0 ↔ (z = x ∨ z = y)) :=
by sorry

end quadratic_roots_property_l957_95790


namespace weekly_calorie_allowance_is_11700_l957_95720

/-- Represents the weekly calorie allowance calculation for a person in their 60's --/
def weekly_calorie_allowance : ℕ :=
  let average_daily_allowance : ℕ := 2000
  let daily_reduction : ℕ := 500
  let reduced_daily_allowance : ℕ := average_daily_allowance - daily_reduction
  let intense_workout_days : ℕ := 2
  let moderate_exercise_days : ℕ := 3
  let rest_days : ℕ := 2
  let intense_workout_extra_calories : ℕ := 300
  let moderate_exercise_extra_calories : ℕ := 200
  
  (reduced_daily_allowance + intense_workout_extra_calories) * intense_workout_days +
  (reduced_daily_allowance + moderate_exercise_extra_calories) * moderate_exercise_days +
  reduced_daily_allowance * rest_days

/-- Theorem stating that the weekly calorie allowance is 11700 calories --/
theorem weekly_calorie_allowance_is_11700 : 
  weekly_calorie_allowance = 11700 := by
  sorry

end weekly_calorie_allowance_is_11700_l957_95720


namespace percentage_increase_l957_95796

theorem percentage_increase (x y z : ℝ) : 
  x = 1.25 * y →
  x + y + z = 925 →
  z = 250 →
  (y - z) / z = 0.2 :=
by
  sorry

end percentage_increase_l957_95796


namespace least_subtraction_for_divisibility_l957_95764

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 98 ∧ 
  (∀ (y : ℕ), y < x → ¬(769 ∣ (157673 - y))) ∧ 
  (769 ∣ (157673 - x)) := by
  sorry

end least_subtraction_for_divisibility_l957_95764
