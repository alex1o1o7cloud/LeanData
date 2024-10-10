import Mathlib

namespace four_variable_inequality_l3529_352937

theorem four_variable_inequality (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) 
  (h5 : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end four_variable_inequality_l3529_352937


namespace rectangle_area_l3529_352910

/-- Given a rectangle with width 42 inches, where ten such rectangles placed end to end
    reach a length of 390 inches, prove that its area is 1638 square inches. -/
theorem rectangle_area (width : ℝ) (total_length : ℝ) (h1 : width = 42)
    (h2 : total_length = 390) : width * (total_length / 10) = 1638 := by
  sorry

end rectangle_area_l3529_352910


namespace parking_lot_spaces_l3529_352944

/-- The number of spaces a single caravan occupies -/
def spaces_per_caravan : ℕ := 2

/-- The number of caravans currently parked -/
def number_of_caravans : ℕ := 3

/-- The number of spaces left for other vehicles -/
def spaces_left : ℕ := 24

/-- The total number of spaces in the parking lot -/
def total_spaces : ℕ := spaces_per_caravan * number_of_caravans + spaces_left

theorem parking_lot_spaces : total_spaces = 30 := by
  sorry

end parking_lot_spaces_l3529_352944


namespace journey_distance_l3529_352998

/-- Proves that given a journey of 9 hours, partly on foot at 4 km/hr for 16 km,
    and partly on bicycle at 9 km/hr, the total distance traveled is 61 km. -/
theorem journey_distance (total_time foot_speed bike_speed foot_distance : ℝ) :
  total_time = 9 ∧
  foot_speed = 4 ∧
  bike_speed = 9 ∧
  foot_distance = 16 →
  ∃ (bike_distance : ℝ),
    foot_distance / foot_speed + bike_distance / bike_speed = total_time ∧
    foot_distance + bike_distance = 61 := by
  sorry

end journey_distance_l3529_352998


namespace at_least_one_acute_angle_leq_45_l3529_352977

-- Define a right triangle
structure RightTriangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  is_right_triangle : angle1 + angle2 + angle3 = 180
  has_right_angle : angle3 = 90

-- Theorem statement
theorem at_least_one_acute_angle_leq_45 (t : RightTriangle) :
  t.angle1 ≤ 45 ∨ t.angle2 ≤ 45 :=
sorry

end at_least_one_acute_angle_leq_45_l3529_352977


namespace tic_tac_toe_tie_l3529_352930

theorem tic_tac_toe_tie (amy_win : ℚ) (lily_win : ℚ) (tie : ℚ) : 
  amy_win = 5/12 → lily_win = 1/4 → tie = 1 - (amy_win + lily_win) → tie = 1/3 := by
sorry

end tic_tac_toe_tie_l3529_352930


namespace division_4512_by_32_l3529_352993

theorem division_4512_by_32 : ∃ (q r : ℕ), 4512 = 32 * q + r ∧ r < 32 ∧ q = 141 ∧ r = 0 := by
  sorry

end division_4512_by_32_l3529_352993


namespace max_value_of_f_l3529_352997

/-- The function f(x) defined as sin(2x) - 2√3 * sin²(x) has a maximum value of 2 - √3 -/
theorem max_value_of_f (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (2 * x) - 2 * Real.sqrt 3 * (Real.sin x) ^ 2
  ∃ (M : ℝ), M = 2 - Real.sqrt 3 ∧ ∀ x, f x ≤ M := by
  sorry

end max_value_of_f_l3529_352997


namespace vertex_on_x_axis_segment_cut_on_x_axis_l3529_352929

-- Define the quadratic function
def f (k x : ℝ) : ℝ := (k + 2) * x^2 - 2 * k * x + 3 * k

-- Theorem for the vertex on x-axis
theorem vertex_on_x_axis (k : ℝ) :
  (∃ x, f k x = 0 ∧ ∀ y, f k y ≥ f k x) ↔ k = 0 ∨ k = -3 := by sorry

-- Theorem for the segment cut on x-axis
theorem segment_cut_on_x_axis (k : ℝ) :
  (∃ a b, a > b ∧ f k a = 0 ∧ f k b = 0 ∧ a - b = 4) ↔ k = -8/3 ∨ k = -1 := by sorry

end vertex_on_x_axis_segment_cut_on_x_axis_l3529_352929


namespace even_monotone_increasing_neg_implies_f1_gt_fneg2_l3529_352955

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y

-- Theorem statement
theorem even_monotone_increasing_neg_implies_f1_gt_fneg2
  (h_even : is_even f)
  (h_mono : monotone_increasing_on_neg f) :
  f 1 > f (-2) :=
sorry

end even_monotone_increasing_neg_implies_f1_gt_fneg2_l3529_352955


namespace car_distance_proof_l3529_352905

/-- Proves that the distance covered by a car is 540 kilometers under given conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) :
  initial_time = 8 →
  speed = 45 →
  (3 / 2 : ℝ) * initial_time * speed = 540 := by
  sorry

end car_distance_proof_l3529_352905


namespace ratatouille_cost_per_quart_l3529_352994

def eggplant_pounds : ℝ := 5
def eggplant_price : ℝ := 2
def zucchini_pounds : ℝ := 4
def zucchini_price : ℝ := 2
def tomato_pounds : ℝ := 4
def tomato_price : ℝ := 3.5
def onion_pounds : ℝ := 3
def onion_price : ℝ := 1
def basil_pounds : ℝ := 1
def basil_price : ℝ := 2.5
def basil_unit : ℝ := 0.5
def yield_quarts : ℝ := 4

theorem ratatouille_cost_per_quart :
  let total_cost := eggplant_pounds * eggplant_price +
                    zucchini_pounds * zucchini_price +
                    tomato_pounds * tomato_price +
                    onion_pounds * onion_price +
                    basil_pounds / basil_unit * basil_price
  total_cost / yield_quarts = 10 := by sorry

end ratatouille_cost_per_quart_l3529_352994


namespace triangle_area_inequality_l3529_352945

theorem triangle_area_inequality (a b c T : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hT : T > 0) (triangle_area : T^2 = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c) / 16) :
  T^2 ≤ a * b * c * (a + b + c) / 16 := by
sorry

end triangle_area_inequality_l3529_352945


namespace geometric_sequence_general_term_l3529_352906

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : IsGeometric a) 
    (h3 : a 3 = 3) (h10 : a 10 = 384) : 
  ∀ n : ℕ, a n = 3 * 2^(n - 3) := by
sorry

end geometric_sequence_general_term_l3529_352906


namespace salary_increase_proof_l3529_352919

theorem salary_increase_proof (S : ℝ) (P : ℝ) : 
  S > 0 →
  0.06 * S > 0 →
  0.10 * (S * (1 + P / 100)) = 1.8333333333333331 * (0.06 * S) →
  P = 10 := by
sorry

end salary_increase_proof_l3529_352919


namespace max_value_abc_l3529_352999

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → x^2 + y^2 + z^2 = 1 → 
  a * b * Real.sqrt 3 + 2 * a * c ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧ 
  a₀ * b₀ * Real.sqrt 3 + 2 * a₀ * c₀ = Real.sqrt 3 :=
sorry

end max_value_abc_l3529_352999


namespace hobbit_burrow_assignment_l3529_352968

-- Define the burrows
inductive Burrow
| A | B | C | D | E | F

-- Define the hobbits
inductive Hobbit
| Frodo | Sam | Merry | Pippin

-- Define the concept of distance between burrows
def closer_to (b1 b2 b3 : Burrow) : Prop := sorry

-- Define the concept of closeness to river and forest
def closer_to_river (b1 b2 : Burrow) : Prop := sorry
def farther_from_forest (b1 b2 : Burrow) : Prop := sorry

-- Define the assignment of hobbits to burrows
def assignment : Hobbit → Burrow
| Hobbit.Frodo => Burrow.E
| Hobbit.Sam => Burrow.A
| Hobbit.Merry => Burrow.C
| Hobbit.Pippin => Burrow.F

-- Theorem statement
theorem hobbit_burrow_assignment :
  (∀ h1 h2 : Hobbit, h1 ≠ h2 → assignment h1 ≠ assignment h2) ∧
  (∀ b : Burrow, b ≠ Burrow.B ∧ b ≠ Burrow.D → ∃ h : Hobbit, assignment h = b) ∧
  (closer_to Burrow.B Burrow.A Burrow.E) ∧
  (closer_to Burrow.D Burrow.A Burrow.E) ∧
  (closer_to_river Burrow.E Burrow.C) ∧
  (farther_from_forest Burrow.E Burrow.F) :=
by sorry

end hobbit_burrow_assignment_l3529_352968


namespace shaded_region_area_l3529_352933

/-- The number of congruent squares in the shaded region -/
def total_squares : ℕ := 20

/-- The number of shaded squares in the larger square -/
def squares_in_larger : ℕ := 4

/-- The length of the diagonal of the larger square in cm -/
def diagonal_length : ℝ := 10

/-- The area of the entire shaded region in square cm -/
def shaded_area : ℝ := 250

theorem shaded_region_area :
  ∀ (total_squares squares_in_larger : ℕ) (diagonal_length shaded_area : ℝ),
  total_squares = 20 →
  squares_in_larger = 4 →
  diagonal_length = 10 →
  shaded_area = total_squares * (diagonal_length / (2 * Real.sqrt 2))^2 →
  shaded_area = 250 := by
sorry

end shaded_region_area_l3529_352933


namespace simplify_expression_l3529_352970

theorem simplify_expression (c : ℝ) : ((3 * c + 6) - 6 * c) / 3 = -c + 2 := by
  sorry

end simplify_expression_l3529_352970


namespace expression_decrease_l3529_352913

theorem expression_decrease (x y : ℝ) (x_pos : x > 0) (y_pos : y > 0) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  (x' * y' ^ 2) / (x * y ^ 2) = 0.216 := by sorry

end expression_decrease_l3529_352913


namespace box_dimensions_sum_l3529_352963

theorem box_dimensions_sum (X Y Z : ℝ) : 
  X > 0 ∧ Y > 0 ∧ Z > 0 →
  X * Y = 24 →
  X * Z = 48 →
  Y * Z = 72 →
  X + Y + Z = 22 := by
sorry

end box_dimensions_sum_l3529_352963


namespace f_max_at_two_l3529_352951

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

/-- Theorem stating that f has a maximum value of 24 at x = 2 -/
theorem f_max_at_two :
  ∃ (x_max : ℝ), x_max = 2 ∧ f x_max = 24 ∧ ∀ (x : ℝ), f x ≤ f x_max :=
sorry

end f_max_at_two_l3529_352951


namespace cube_root_7200_simplification_l3529_352938

theorem cube_root_7200_simplification : 
  ∃ (c d : ℕ+), (c.val : ℝ) * (d.val : ℝ)^(1/3) = 7200^(1/3) ∧ 
  (∀ (c' d' : ℕ+), (c'.val : ℝ) * (d'.val : ℝ)^(1/3) = 7200^(1/3) → d'.val ≤ d.val) →
  c.val + d.val = 452 := by
sorry

end cube_root_7200_simplification_l3529_352938


namespace compare_large_powers_l3529_352940

theorem compare_large_powers : 100^100 > 50^50 * 150^50 := by
  sorry

end compare_large_powers_l3529_352940


namespace max_consecutive_interesting_integers_l3529_352942

/-- A function that returns the nth prime number -/
noncomputable def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns the product of the first n primes -/
noncomputable def productOfFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Definition of an interesting number -/
def isInteresting (k : ℕ) : Prop :=
  k > 0 ∧ (productOfFirstNPrimes k) % k = 0

/-- Theorem stating that the maximal number of consecutive interesting integers is 7 -/
theorem max_consecutive_interesting_integers :
  ∃ n : ℕ, n > 0 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ n → isInteresting k) ∧
  (∀ m : ℕ, m > n → ¬(∀ k : ℕ, k > 0 ∧ k ≤ m → isInteresting k)) ∧
  n = 7 := by sorry

end max_consecutive_interesting_integers_l3529_352942


namespace inequality_equivalence_l3529_352976

theorem inequality_equivalence (x : ℝ) : 2 - x > 3 + x ↔ x < -1/2 := by sorry

end inequality_equivalence_l3529_352976


namespace complex_cube_theorem_l3529_352920

theorem complex_cube_theorem : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / (1 - 2*i)
  z^3 = -i := by sorry

end complex_cube_theorem_l3529_352920


namespace coins_per_roll_is_25_l3529_352904

/-- Represents the number of coins in a single roll -/
def coins_per_roll : ℕ := sorry

/-- The number of rolls each bank teller has -/
def rolls_per_teller : ℕ := 10

/-- The number of bank tellers -/
def number_of_tellers : ℕ := 4

/-- The total number of coins among all tellers -/
def total_coins : ℕ := 1000

theorem coins_per_roll_is_25 : 
  coins_per_roll * rolls_per_teller * number_of_tellers = total_coins →
  coins_per_roll = 25 := by
  sorry

end coins_per_roll_is_25_l3529_352904


namespace largest_four_digit_divisible_by_2718_and_gcd_l3529_352954

theorem largest_four_digit_divisible_by_2718_and_gcd : ∃ (n : ℕ), n ≤ 9999 ∧ n % 2718 = 0 ∧ 
  (∀ m : ℕ, m ≤ 9999 ∧ m % 2718 = 0 → m ≤ n) ∧
  n = 8154 ∧
  Nat.gcd n 8640 = 6 := by
  sorry

end largest_four_digit_divisible_by_2718_and_gcd_l3529_352954


namespace points_per_recycled_bag_l3529_352950

theorem points_per_recycled_bag 
  (total_bags : ℕ) 
  (unrecycled_bags : ℕ) 
  (total_points : ℕ) 
  (h1 : total_bags = 11) 
  (h2 : unrecycled_bags = 2) 
  (h3 : total_points = 45) :
  total_points / (total_bags - unrecycled_bags) = 5 := by
  sorry

end points_per_recycled_bag_l3529_352950


namespace maria_boxes_count_l3529_352959

def eggs_per_box : ℕ := 7
def total_eggs : ℕ := 21

theorem maria_boxes_count : 
  total_eggs / eggs_per_box = 3 := by sorry

end maria_boxes_count_l3529_352959


namespace geometric_series_sum_l3529_352985

/-- The sum of the infinite series ∑(n=0 to ∞) (2^n / 5^n) is equal to 5/3 -/
theorem geometric_series_sum : 
  let a : ℕ → ℝ := λ n => (2 : ℝ)^n
  (∑' n, a n / 5^n) = 5/3 := by
  sorry

end geometric_series_sum_l3529_352985


namespace sufficient_but_not_necessary_l3529_352900

open Real

theorem sufficient_but_not_necessary (θ : ℝ) : 
  (∀ θ, |θ - π/12| < π/12 → sin θ < 1/2) ∧ 
  (∃ θ, sin θ < 1/2 ∧ |θ - π/12| ≥ π/12) := by
  sorry

end sufficient_but_not_necessary_l3529_352900


namespace square_area_from_perimeter_l3529_352947

/-- The area of a square with perimeter 32 cm is 64 cm² -/
theorem square_area_from_perimeter (perimeter : ℝ) (side : ℝ) (area : ℝ) :
  perimeter = 32 →
  side = perimeter / 4 →
  area = side * side →
  area = 64 := by
  sorry

end square_area_from_perimeter_l3529_352947


namespace budget_calculation_l3529_352946

/-- The total budget for purchasing a TV, computer, and fridge -/
def total_budget (tv_cost computer_cost fridge_extra_cost : ℕ) : ℕ :=
  tv_cost + computer_cost + (computer_cost + fridge_extra_cost)

/-- Theorem stating that the total budget for the given costs is 1600 -/
theorem budget_calculation :
  total_budget 600 250 500 = 1600 := by
  sorry

end budget_calculation_l3529_352946


namespace line_properties_l3529_352953

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 2

theorem line_properties :
  -- Part 1: The line passes through (1, 1) for all real m
  (∀ m : ℝ, line_equation m 1 1) ∧
  -- Part 2: When the line is tangent to the circle, m = -1
  (∃ m : ℝ, (∀ x y : ℝ, line_equation m x y → circle_equation x y → 
    (x - 0)^2 + (y - 0)^2 = (1 - m)^2 / (m^2 + 1)) ∧ m = -1) :=
sorry

end line_properties_l3529_352953


namespace base_difference_theorem_l3529_352948

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers
def num1 : List Nat := [3, 2, 7]
def base1 : Nat := 9
def num2 : List Nat := [2, 5, 3]
def base2 : Nat := 8

-- State the theorem
theorem base_difference_theorem : 
  to_base_10 num1 base1 - to_base_10 num2 base2 = 97 := by
  sorry

end base_difference_theorem_l3529_352948


namespace pencil_boxes_filled_l3529_352973

theorem pencil_boxes_filled (total_pencils : ℕ) (pencils_per_box : ℕ) (h1 : total_pencils = 7344) (h2 : pencils_per_box = 7) : 
  total_pencils / pencils_per_box = 1049 := by
  sorry

end pencil_boxes_filled_l3529_352973


namespace quadratic_other_intercept_l3529_352961

/-- For a quadratic function f(x) = ax^2 + bx + c with vertex (2, 10) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 3. -/
theorem quadratic_other_intercept 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 2 = 10 ∧ (∀ x, f x ≤ 10)) 
  (h3 : f 1 = 0) : 
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 3 :=
sorry

end quadratic_other_intercept_l3529_352961


namespace d_is_nonzero_l3529_352958

/-- A polynomial of degree 5 with six distinct x-intercepts, including 0 and -1 -/
def Q (a b c d : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x

/-- The property that Q has six distinct x-intercepts, including 0 and -1 -/
def has_six_distinct_intercepts (a b c d : ℝ) : Prop :=
  ∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
              p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧
              p ≠ -1 ∧ q ≠ -1 ∧ r ≠ -1 ∧ s ≠ -1 ∧
              ∀ x : ℝ, Q a b c d x = 0 ↔ x = 0 ∨ x = -1 ∨ x = p ∨ x = q ∨ x = r ∨ x = s

theorem d_is_nonzero (a b c d : ℝ) (h : has_six_distinct_intercepts a b c d) : d ≠ 0 :=
sorry

end d_is_nonzero_l3529_352958


namespace clock_angle_at_7_l3529_352972

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The time in hours -/
def time : ℕ := 7

/-- The angle between each hour mark on the clock -/
def angle_per_hour : ℚ := 360 / clock_hours

/-- The position of the hour hand in degrees -/
def hour_hand_position : ℚ := time * angle_per_hour

/-- The smaller angle between the hour and minute hands at the given time -/
def smaller_angle : ℚ := min hour_hand_position (360 - hour_hand_position)

/-- Theorem stating that the smaller angle between clock hands at 7 o'clock is 150 degrees -/
theorem clock_angle_at_7 : smaller_angle = 150 := by sorry

end clock_angle_at_7_l3529_352972


namespace percentage_equality_l3529_352912

theorem percentage_equality : (10 : ℚ) / 100 * 200 = (20 : ℚ) / 100 * 100 := by
  sorry

end percentage_equality_l3529_352912


namespace estimate_event_knowledge_chengdu_games_knowledge_estimate_l3529_352992

/-- Estimates the number of people in a population who know about an event,
    given a sample survey result. -/
theorem estimate_event_knowledge (total_population : ℕ) 
                                  (sample_size : ℕ) 
                                  (sample_positive : ℕ) : ℕ :=
  let estimate := (sample_positive * total_population) / sample_size
  estimate

/-- Proves that the estimated number of people who know about the event
    in a population of 10,000, given 125 out of 200 know in a sample, is 6250. -/
theorem chengdu_games_knowledge_estimate :
  estimate_event_knowledge 10000 200 125 = 6250 := by
  sorry

end estimate_event_knowledge_chengdu_games_knowledge_estimate_l3529_352992


namespace division_problem_l3529_352901

theorem division_problem (n : ℕ) : n / 20 = 10 ∧ n % 20 = 10 ↔ n = 210 := by
  sorry

end division_problem_l3529_352901


namespace magical_coin_expected_winnings_l3529_352986

/-- Represents the outcomes of the magical coin flip -/
inductive Outcome
  | Heads
  | Tails
  | Edge
  | Disappear

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Heads => 3/8
  | Outcome.Tails => 1/4
  | Outcome.Edge => 1/8
  | Outcome.Disappear => 1/4

/-- The winnings (or losses) for each outcome -/
def winnings (o : Outcome) : ℚ :=
  match o with
  | Outcome.Heads => 2
  | Outcome.Tails => 5
  | Outcome.Edge => -2
  | Outcome.Disappear => -6

/-- The expected winnings of flipping the magical coin -/
def expected_winnings : ℚ :=
  (probability Outcome.Heads * winnings Outcome.Heads) +
  (probability Outcome.Tails * winnings Outcome.Tails) +
  (probability Outcome.Edge * winnings Outcome.Edge) +
  (probability Outcome.Disappear * winnings Outcome.Disappear)

theorem magical_coin_expected_winnings :
  expected_winnings = 1/4 := by
  sorry

end magical_coin_expected_winnings_l3529_352986


namespace cos_fourteen_pi_thirds_l3529_352923

theorem cos_fourteen_pi_thirds : Real.cos (14 * π / 3) = -1 / 2 := by
  sorry

end cos_fourteen_pi_thirds_l3529_352923


namespace lateral_surface_area_theorem_l3529_352932

/-- Represents a regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The dihedral angle at the lateral edge -/
  dihedral_angle : ℝ
  /-- The area of the diagonal section -/
  diagonal_section_area : ℝ

/-- The lateral surface area of a regular quadrilateral pyramid -/
noncomputable def lateral_surface_area (p : RegularQuadrilateralPyramid) : ℝ :=
  4 * p.diagonal_section_area

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid is 4S,
    where S is the area of its diagonal section, given that the dihedral angle
    at the lateral edge is 120° -/
theorem lateral_surface_area_theorem (p : RegularQuadrilateralPyramid) 
  (h : p.dihedral_angle = 120) : 
  lateral_surface_area p = 4 * p.diagonal_section_area := by
  sorry

end lateral_surface_area_theorem_l3529_352932


namespace peters_socks_l3529_352939

theorem peters_socks (x y z : ℕ) : 
  x + y + z = 15 →
  2*x + 3*y + 5*z = 45 →
  x ≥ 1 →
  y ≥ 1 →
  z ≥ 1 →
  x = 6 :=
by sorry

end peters_socks_l3529_352939


namespace evan_book_difference_l3529_352987

/-- Represents the number of books Evan owns at different points in time -/
structure EvanBooks where
  twoYearsAgo : ℕ
  current : ℕ
  inFiveYears : ℕ

/-- The conditions of Evan's book collection -/
def evanBookConditions (books : EvanBooks) : Prop :=
  books.twoYearsAgo = 200 ∧
  books.current = books.twoYearsAgo - 40 ∧
  books.inFiveYears = 860

/-- The theorem stating the difference between Evan's books in five years
    and five times his current number of books -/
theorem evan_book_difference (books : EvanBooks) 
  (h : evanBookConditions books) : 
  books.inFiveYears - (5 * books.current) = 60 := by
  sorry

end evan_book_difference_l3529_352987


namespace same_parity_of_extrema_l3529_352925

/-- A set with certain properties related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest_element (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest_element (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_of_extrema :
  is_even (smallest_element A_P) ↔ is_even (largest_element A_P) := by
  sorry

end same_parity_of_extrema_l3529_352925


namespace tangent_sphere_radius_l3529_352966

/-- A truncated cone with horizontal bases of radii 12 and 4, and height 15 -/
structure TruncatedCone where
  largeRadius : ℝ := 12
  smallRadius : ℝ := 4
  height : ℝ := 15

/-- A sphere tangent to the inside surfaces of a truncated cone -/
structure TangentSphere (cone : TruncatedCone) where
  radius : ℝ

/-- The radius of the tangent sphere is √161/2 -/
theorem tangent_sphere_radius (cone : TruncatedCone) (sphere : TangentSphere cone) :
  sphere.radius = Real.sqrt 161 / 2 := by
  sorry

end tangent_sphere_radius_l3529_352966


namespace unique_n_with_divisor_sum_property_l3529_352915

theorem unique_n_with_divisor_sum_property (n : ℕ+) 
  (h1 : ∃ d₁ d₂ d₃ d₄ : ℕ+, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
        (∀ m : ℕ+, m ∣ n → m ≥ d₁) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m ≥ d₂) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m = d₂ ∨ m ≥ d₃) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m = d₂ ∨ m = d₃ ∨ m ≥ d₄))
  (h2 : ∃ d₁ d₂ d₃ d₄ : ℕ+, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
        d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
        d₁^2 + d₂^2 + d₃^2 + d₄^2 = n) :
  n = 130 := by
  sorry

end unique_n_with_divisor_sum_property_l3529_352915


namespace system_solution_iff_conditions_l3529_352957

-- Define the system of equations
def has_solution (n p x y z : ℕ) : Prop :=
  x + p * y = n ∧ x + y = p^z

-- Define the conditions
def conditions (n p : ℕ) : Prop :=
  p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ∀ k : ℕ, n ≠ p^k

-- Theorem statement
theorem system_solution_iff_conditions (n p : ℕ) :
  (∃! x y z : ℕ, has_solution n p x y z) ↔ conditions n p :=
sorry

end system_solution_iff_conditions_l3529_352957


namespace max_value_of_five_integers_with_mean_eleven_l3529_352995

theorem max_value_of_five_integers_with_mean_eleven (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  (a + b + c + d + e : ℚ) / 5 = 11 →
  max a (max b (max c (max d e))) ≤ 45 :=
sorry

end max_value_of_five_integers_with_mean_eleven_l3529_352995


namespace picnic_problem_l3529_352991

theorem picnic_problem (total : ℕ) (men_excess : ℕ) (adult_excess : ℕ) :
  total = 240 →
  men_excess = 80 →
  adult_excess = 80 →
  ∃ (men women adults children : ℕ),
    men = women + men_excess ∧
    adults = children + adult_excess ∧
    men + women = adults ∧
    adults + children = total ∧
    men = 120 := by
  sorry

end picnic_problem_l3529_352991


namespace cindy_calculation_l3529_352926

theorem cindy_calculation (x : ℚ) : (x - 7) / 5 = 53 → (x - 5) / 7 = 38 := by
  sorry

end cindy_calculation_l3529_352926


namespace positive_difference_l3529_352922

theorem positive_difference (x y w : ℝ) 
  (hx : 0 < x ∧ x < 0.5) 
  (hy : -0.5 < y ∧ y < 0) 
  (hw : 0.5 < w ∧ w < 1) : 
  w - y > 0 := by
  sorry

end positive_difference_l3529_352922


namespace cos_150_degrees_l3529_352952

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_150_degrees_l3529_352952


namespace functional_equation_problem_l3529_352911

/-- A function satisfying f(a+b) = f(a)f(b) for all a and b -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b, f (a + b) = f a * f b

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) 
  (h2 : f 1 = 2) : 
  (f 1)^2 / f 1 + f 2 / f 1 + 
  (f 2)^2 / f 3 + f 4 / f 3 + 
  (f 3)^2 / f 5 + f 6 / f 5 + 
  (f 4)^2 / f 7 + f 8 / f 7 = 16 := by
  sorry

end functional_equation_problem_l3529_352911


namespace extra_sweets_per_child_l3529_352931

theorem extra_sweets_per_child (total_children : ℕ) (absent_children : ℕ) (sweets_per_present_child : ℕ) :
  total_children = 190 →
  absent_children = 70 →
  sweets_per_present_child = 38 →
  (total_children - absent_children) * sweets_per_present_child / total_children - 
    ((total_children - absent_children) * sweets_per_present_child / total_children) = 14 := by
  sorry

end extra_sweets_per_child_l3529_352931


namespace perp_necessary_not_sufficient_l3529_352934

-- Define the plane α
variable (α : Plane)

-- Define lines l, m, and n
variable (l m n : Line)

-- Define the property that a line is in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between a line and a plane
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between two lines
def line_perp_line (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perp_necessary_not_sufficient :
  (line_in_plane m α ∧ line_in_plane n α) →
  (∀ l, line_perp_plane l α → (line_perp_line l m ∧ line_perp_line l n)) ∧
  (∃ l, line_perp_line l m ∧ line_perp_line l n ∧ ¬line_perp_plane l α) := by
  sorry

end perp_necessary_not_sufficient_l3529_352934


namespace min_five_dollar_frisbees_l3529_352935

/-- Given a total of 115 frisbees sold for $450, with prices of $3, $4, and $5,
    the minimum number of $5 frisbees sold is 1. -/
theorem min_five_dollar_frisbees :
  ∀ (x y z : ℕ),
    x + y + z = 115 →
    3 * x + 4 * y + 5 * z = 450 →
    z ≥ 1 ∧
    ∀ (a b c : ℕ),
      a + b + c = 115 →
      3 * a + 4 * b + 5 * c = 450 →
      c ≥ z :=
by sorry

end min_five_dollar_frisbees_l3529_352935


namespace hyperbola_eccentricity_l3529_352960

theorem hyperbola_eccentricity (m : ℝ) :
  (∃ e : ℝ, e > Real.sqrt 2 ∧
    ∀ x y : ℝ, x^2 - y^2/m = 1 → e = Real.sqrt (1 + m)) ↔
  m > 1 :=
by sorry

end hyperbola_eccentricity_l3529_352960


namespace parallel_line_through_point_l3529_352988

/-- Given two lines in the xy-plane, this function returns true if they are parallel --/
def are_parallel (m1 : ℝ) (m2 : ℝ) : Prop := m1 = m2

/-- Given a point (x, y) and a line equation y = mx + b, this function returns true if the point lies on the line --/
def point_on_line (x : ℝ) (y : ℝ) (m : ℝ) (b : ℝ) : Prop := y = m * x + b

theorem parallel_line_through_point (x0 y0 : ℝ) : 
  ∃ (m b : ℝ), 
    are_parallel m 2 ∧ 
    point_on_line x0 y0 m b ∧ 
    m = 2 ∧ 
    b = -5 :=
sorry

end parallel_line_through_point_l3529_352988


namespace quadratic_roots_property_l3529_352921

theorem quadratic_roots_property (f g : ℝ) : 
  (3 * f^2 + 5 * f - 8 = 0) → 
  (3 * g^2 + 5 * g - 8 = 0) → 
  (f - 2) * (g - 2) = 14/3 := by
sorry

end quadratic_roots_property_l3529_352921


namespace circuit_board_count_l3529_352967

theorem circuit_board_count (T P : ℕ) : 
  (64 + P = T) →  -- Total boards = Failed + Passed
  (64 + P / 8 = 456) →  -- Total faulty boards
  T = 3200 := by
  sorry

end circuit_board_count_l3529_352967


namespace divisible_by_21_with_sqrt_between_30_and_30_5_l3529_352965

theorem divisible_by_21_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (21 ∣ n) ∧ (30 < Real.sqrt n) ∧ (Real.sqrt n < 30.5) := by
  sorry

end divisible_by_21_with_sqrt_between_30_and_30_5_l3529_352965


namespace identical_coordinate_point_exists_l3529_352980

/-- Represents a 2D rectangular coordinate system -/
structure CoordinateSystem :=
  (origin : ℝ × ℝ)
  (xAxis : ℝ × ℝ)
  (yAxis : ℝ × ℝ)
  (unitLength : ℝ)

/-- Theorem: Existence of a point with identical coordinates in two different coordinate systems -/
theorem identical_coordinate_point_exists 
  (cs1 cs2 : CoordinateSystem) 
  (h1 : cs1.origin ≠ cs2.origin) 
  (h2 : ¬ (∃ k : ℝ, cs1.xAxis = k • cs2.xAxis)) 
  (h3 : cs1.unitLength ≠ cs2.unitLength) : 
  ∃ p : ℝ × ℝ, ∃ x y : ℝ, 
    (x, y) = p ∧ 
    (∃ x' y' : ℝ, (x', y') = p ∧ x = x' ∧ y = y') :=
sorry

end identical_coordinate_point_exists_l3529_352980


namespace line_equation_through_point_l3529_352903

/-- The equation of a line with slope 2 passing through the point (2, 3) is 2x - y - 1 = 0 -/
theorem line_equation_through_point (x y : ℝ) :
  let slope : ℝ := 2
  let point : ℝ × ℝ := (2, 3)
  (y - point.2 = slope * (x - point.1)) ↔ (2 * x - y - 1 = 0) := by
sorry

end line_equation_through_point_l3529_352903


namespace imaginary_part_of_z_l3529_352969

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = 5) : z.im = -2 := by
  sorry

end imaginary_part_of_z_l3529_352969


namespace field_trip_cost_theorem_l3529_352974

/-- Calculates the total cost of a field trip with a group discount --/
def field_trip_cost (num_classes : ℕ) (students_per_class : ℕ) (adults_per_class : ℕ)
  (student_fee : ℚ) (adult_fee : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_students := num_classes * students_per_class
  let total_adults := num_classes * adults_per_class
  let student_cost := total_students * student_fee
  let adult_cost := total_adults * adult_fee
  let total_cost := student_cost + adult_cost
  let discount := if total_students > discount_threshold then discount_rate * student_cost else 0
  total_cost - discount

/-- The total cost of the field trip is $987.60 --/
theorem field_trip_cost_theorem :
  field_trip_cost 4 42 6 (11/2) (13/2) (1/10) 40 = 9876/10 := by
  sorry

end field_trip_cost_theorem_l3529_352974


namespace f_smallest_positive_period_l3529_352978

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x) + 2^(|Real.sin (2 * x)|^2) + 5 * |Real.sin (2 * x)|

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem f_smallest_positive_period :
  is_smallest_positive_period f (Real.pi / 2) := by
  sorry

end f_smallest_positive_period_l3529_352978


namespace volunteer_arrangement_count_l3529_352983

theorem volunteer_arrangement_count : 
  (n : ℕ) → 
  (total : ℕ) → 
  (day1 : ℕ) → 
  (day2 : ℕ) → 
  (day3 : ℕ) → 
  n = 4 → 
  total = 5 → 
  day1 = 1 → 
  day2 = 2 → 
  day3 = 1 → 
  day1 + day2 + day3 = n →
  (total.choose day1) * ((total - day1).choose day2) * ((total - day1 - day2).choose day3) = 60 := by
sorry

end volunteer_arrangement_count_l3529_352983


namespace jane_change_l3529_352909

/-- Calculates the change received after a purchase -/
def calculate_change (num_skirts : ℕ) (price_skirt : ℕ) (num_blouses : ℕ) (price_blouse : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_skirts * price_skirt + num_blouses * price_blouse)

/-- Proves that Jane received $56 in change -/
theorem jane_change : calculate_change 2 13 3 6 100 = 56 := by
  sorry

end jane_change_l3529_352909


namespace inequality_solution_range_l3529_352981

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3) ↔ 3 * (x : ℝ) - m ≤ 0) →
  (9 ≤ m ∧ m < 12) :=
by sorry

end inequality_solution_range_l3529_352981


namespace bananas_bought_l3529_352971

/-- The number of times Ronald went to the store last month -/
def store_visits : ℕ := 2

/-- The number of bananas Ronald buys each time he goes to the store -/
def bananas_per_visit : ℕ := 10

/-- The total number of bananas Ronald bought last month -/
def total_bananas : ℕ := store_visits * bananas_per_visit

theorem bananas_bought : total_bananas = 20 := by
  sorry

end bananas_bought_l3529_352971


namespace geometric_arithmetic_sequence_ratio_sum_l3529_352908

theorem geometric_arithmetic_sequence_ratio_sum :
  ∀ (x y z : ℝ),
  (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) →
  (∃ (r : ℝ), r ≠ 0 ∧ 4*y = 3*x*r ∧ 5*z = 4*y*r) →
  (∃ (d : ℝ), 1/y - 1/x = d ∧ 1/z - 1/y = d) →
  x/z + z/x = 34/15 := by
sorry

end geometric_arithmetic_sequence_ratio_sum_l3529_352908


namespace sum_of_digits_equals_five_l3529_352927

/-- S(n) is the sum of digits in the decimal representation of 2^n -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that S(n) = 5 if and only if n = 5 -/
theorem sum_of_digits_equals_five (n : ℕ) : S n = 5 ↔ n = 5 := by sorry

end sum_of_digits_equals_five_l3529_352927


namespace cos_negative_nineteen_pi_sixths_tan_x_given_sin_x_l3529_352902

-- Part I
theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by sorry

-- Part II
theorem tan_x_given_sin_x (x : Real) :
  x ∈ Set.Icc (π / 2) (3 * π / 2) →
  Real.sin x = -3 / 5 →
  Real.tan x = 3 / 4 := by sorry

end cos_negative_nineteen_pi_sixths_tan_x_given_sin_x_l3529_352902


namespace min_value_of_f_l3529_352941

theorem min_value_of_f (a₁ a₂ a₃ a₄ : ℝ) (h : a₁ * a₄ - a₂ * a₃ = 1) : 
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₄ - x₂ * x₃ = 1 → 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₁*x₃ + x₂*x₄ ≥ m ∧
    ∃ (y₁ y₂ y₃ y₄ : ℝ), y₁ * y₄ - y₂ * y₃ = 1 ∧
      y₁^2 + y₂^2 + y₃^2 + y₄^2 + y₁*y₃ + y₂*y₄ = m :=
by sorry

end min_value_of_f_l3529_352941


namespace positive_real_solution_l3529_352956

theorem positive_real_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : 3 * Real.sqrt (x^2 + x) + 3 * Real.sqrt (x^2 - x) = 6 * Real.sqrt 2) : 
  x = 4 * Real.sqrt 7 / 7 := by
  sorry

end positive_real_solution_l3529_352956


namespace cos_squared_minus_sin_squared_three_pi_eighths_l3529_352979

theorem cos_squared_minus_sin_squared_three_pi_eighths :
  Real.cos (3 * Real.pi / 8) ^ 2 - Real.sin (3 * Real.pi / 8) ^ 2 = -Real.sqrt 2 / 2 := by
  sorry

end cos_squared_minus_sin_squared_three_pi_eighths_l3529_352979


namespace caitlin_age_caitlin_age_proof_l3529_352962

/-- Proves that Caitlin is 54 years old given the conditions in the problem -/
theorem caitlin_age : ℕ → ℕ → ℕ → Prop :=
  λ anna_age brianna_age caitlin_age =>
    anna_age = 48 ∧
    brianna_age = 2 * (anna_age - 18) ∧
    caitlin_age = brianna_age - 6 →
    caitlin_age = 54

/-- Proof of the theorem -/
theorem caitlin_age_proof : caitlin_age 48 60 54 := by
  sorry

end caitlin_age_caitlin_age_proof_l3529_352962


namespace remaining_money_l3529_352984

def initial_amount : ℚ := 100
def apple_price : ℚ := 1.5
def orange_price : ℚ := 2
def pear_price : ℚ := 2.25
def apple_quantity : ℕ := 5
def orange_quantity : ℕ := 10
def pear_quantity : ℕ := 4

theorem remaining_money :
  initial_amount - 
  (apple_price * apple_quantity + 
   orange_price * orange_quantity + 
   pear_price * pear_quantity) = 63.5 := by
  sorry

end remaining_money_l3529_352984


namespace integer_equation_solution_l3529_352916

theorem integer_equation_solution (x y : ℤ) : x^4 - 2*y^2 = 1 → (x = 1 ∨ x = -1) ∧ y = 0 := by
  sorry

end integer_equation_solution_l3529_352916


namespace arcsin_arccos_inequality_l3529_352928

theorem arcsin_arccos_inequality (x : ℝ) : 
  Real.arcsin ((5 / (2 * Real.pi)) * Real.arccos x) > Real.arccos ((10 / (3 * Real.pi)) * Real.arcsin x) ↔ 
  (x ∈ Set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪ 
   Set.Ioo (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5))) :=
by sorry

end arcsin_arccos_inequality_l3529_352928


namespace divisibility_proof_l3529_352989

theorem divisibility_proof (n : ℕ) (x : ℝ) :
  ∃ P : ℝ → ℝ, (x + 1)^(2*n) - x^(2*n) - 2*x - 1 = x * (x + 1) * (2*x + 1) * P x :=
by sorry

end divisibility_proof_l3529_352989


namespace innings_count_l3529_352949

/-- Represents the batting statistics of a batsman -/
structure BattingStats where
  n : ℕ                -- Total number of innings
  highest : ℕ          -- Highest score
  lowest : ℕ           -- Lowest score
  average : ℚ          -- Average score
  newAverage : ℚ       -- Average after excluding highest and lowest scores

/-- Theorem stating the conditions and the result to be proved -/
theorem innings_count (stats : BattingStats) : 
  stats.average = 50 ∧ 
  stats.highest - stats.lowest = 172 ∧
  stats.newAverage = stats.average - 2 ∧
  stats.highest = 174 →
  stats.n = 40 := by
  sorry


end innings_count_l3529_352949


namespace coral_reef_decrease_l3529_352943

def coral_decrease_rate : ℝ := 0.3
def target_percentage : ℝ := 0.05
def years_since_2010 : ℕ := 10

theorem coral_reef_decrease :
  (1 - coral_decrease_rate) ^ years_since_2010 < target_percentage := by
  sorry

end coral_reef_decrease_l3529_352943


namespace intersection_and_complement_eq_union_l3529_352924

/-- Given the universal set ℝ, prove that the intersection of M and the complement of N in ℝ
    is the union of {x | x < -2} and {x | x ≥ 3} -/
theorem intersection_and_complement_eq_union (M N : Set ℝ) : 
  M = {x : ℝ | x^2 > 4} →
  N = {x : ℝ | (x - 3) / (x + 1) < 0} →
  M ∩ (Set.univ \ N) = {x : ℝ | x < -2} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end intersection_and_complement_eq_union_l3529_352924


namespace sector_angle_l3529_352914

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, its central angle is 2 radians. -/
theorem sector_angle (r : ℝ) (θ : ℝ) : 
  (1/2 * θ * r^2 = 1) → (2*r + θ*r = 4) → θ = 2 := by
  sorry

end sector_angle_l3529_352914


namespace units_digit_of_3_pow_5_times_2_pow_3_l3529_352975

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 3^5 × 2^3 is 4 -/
theorem units_digit_of_3_pow_5_times_2_pow_3 :
  unitsDigit (3^5 * 2^3) = 4 := by sorry

end units_digit_of_3_pow_5_times_2_pow_3_l3529_352975


namespace house_prices_and_yields_l3529_352907

theorem house_prices_and_yields :
  ∀ (price1 price2 yield1 yield2 : ℝ),
  price1 > 0 ∧ price2 > 0 ∧ yield1 > 0 ∧ yield2 > 0 →
  425 = (yield1 / 100) * price1 →
  459 = (yield2 / 100) * price2 →
  price2 = (6 / 5) * price1 →
  yield2 = yield1 - (1 / 2) →
  price1 = 8500 ∧ price2 = 10200 ∧ yield1 = 5 ∧ yield2 = (9 / 2) :=
by sorry

end house_prices_and_yields_l3529_352907


namespace alex_savings_dimes_l3529_352936

/-- Proves that given $6.35 in dimes and quarters, with 5 more dimes than quarters, the number of dimes is 22 -/
theorem alex_savings_dimes : 
  ∀ (d q : ℕ), 
    (d : ℚ) * 0.1 + (q : ℚ) * 0.25 = 6.35 → -- Total value in dollars
    d = q + 5 → -- 5 more dimes than quarters
    d = 22 := by sorry

end alex_savings_dimes_l3529_352936


namespace x_squared_over_x_fourth_plus_x_squared_plus_one_l3529_352918

theorem x_squared_over_x_fourth_plus_x_squared_plus_one (x : ℝ) 
  (h1 : x^2 - 3*x - 1 = 0) (h2 : x ≠ 0) : x^2 / (x^4 + x^2 + 1) = 1/12 := by
  sorry

end x_squared_over_x_fourth_plus_x_squared_plus_one_l3529_352918


namespace parallel_vectors_l3529_352982

/-- Given vectors a and b in ℝ², prove that if a is parallel to b, then λ = 8/5 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : a.1 / b.1 = a.2 / b.2) :
  a = (2, 5) → b.2 = 4 → b.1 = 8/5 := by
  sorry

#check parallel_vectors

end parallel_vectors_l3529_352982


namespace complex_fraction_simplification_l3529_352917

theorem complex_fraction_simplification :
  (7 : ℂ) + 9*I / (3 : ℂ) - 4*I = 57/25 + 55/25*I :=
by sorry

end complex_fraction_simplification_l3529_352917


namespace pie_eating_contest_l3529_352996

theorem pie_eating_contest (first_participant second_participant : ℚ) : 
  first_participant = 5/6 → second_participant = 2/3 → 
  first_participant - second_participant = 1/6 := by
sorry

end pie_eating_contest_l3529_352996


namespace solve_for_m_l3529_352964

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (m : ℝ) : Prop :=
  (1 - m * i) / (i^3) = 1 + i

-- Theorem statement
theorem solve_for_m :
  ∃ (m : ℝ), equation m ∧ m = 1 :=
sorry

end solve_for_m_l3529_352964


namespace min_students_in_class_l3529_352990

theorem min_students_in_class (b g : ℕ) : 
  b = 2 * g →  -- ratio of boys to girls is 2:1
  (3 * b) / 5 = (5 * g) / 8 →  -- number of boys who passed equals number of girls who passed
  b + g ≥ 120 ∧ ∀ n < 120, ¬(∃ b' g', b' = 2 * g' ∧ (3 * b') / 5 = (5 * g') / 8 ∧ b' + g' = n) :=
by sorry

end min_students_in_class_l3529_352990
