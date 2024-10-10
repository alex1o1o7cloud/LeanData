import Mathlib

namespace sum_of_coefficients_l357_35729

theorem sum_of_coefficients (f : ℝ → ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, f x = (x - 5)^7 + (x - 8)^5) →
  (∀ x, f x = a₀ + a₁*(x - 6) + a₂*(x - 6)^2 + a₃*(x - 6)^3 + a₄*(x - 6)^4 + 
           a₅*(x - 6)^5 + a₆*(x - 6)^6 + a₇*(x - 6)^7) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 127 := by
sorry

end sum_of_coefficients_l357_35729


namespace problem_solution_l357_35726

theorem problem_solution (x y a : ℝ) 
  (h1 : Real.sqrt (3 * x + 4) + y^2 + 6 * y + 9 = 0)
  (h2 : a * x * y - 3 * x = y) : 
  a = -7/4 := by
  sorry

end problem_solution_l357_35726


namespace solution_concentration_l357_35700

def concentrate_solution (initial_volume : ℝ) (final_concentration : ℝ) (water_removed : ℝ) : Prop :=
  ∃ (initial_concentration : ℝ),
    0 < initial_concentration ∧
    initial_concentration < final_concentration ∧
    final_concentration < 1 ∧
    initial_volume > water_removed ∧
    -- The actual concentration calculation would go here, but we lack the initial concentration
    True

theorem solution_concentration :
  concentrate_solution 24 0.6 8 := by
  sorry

end solution_concentration_l357_35700


namespace car_speed_problem_l357_35778

/-- Given a car traveling for 2 hours with an average speed of 55 km/h,
    if its speed in the second hour is 60 km/h,
    then its speed in the first hour must be 50 km/h. -/
theorem car_speed_problem (x : ℝ) : 
  (x + 60) / 2 = 55 → x = 50 := by
  sorry

end car_speed_problem_l357_35778


namespace daily_fine_is_two_l357_35761

/-- Calculates the daily fine for absence given the total engagement period, daily wage, total amount received, and number of days absent. -/
def calculate_daily_fine (total_days : ℕ) (daily_wage : ℕ) (total_received : ℕ) (days_absent : ℕ) : ℕ :=
  let days_worked := total_days - days_absent
  let total_earned := days_worked * daily_wage
  let total_fine := total_earned - total_received
  total_fine / days_absent

/-- Theorem stating that the daily fine is 2 given the problem conditions. -/
theorem daily_fine_is_two :
  calculate_daily_fine 30 10 216 7 = 2 := by
  sorry

end daily_fine_is_two_l357_35761


namespace union_A_complement_B_eq_closed_interval_l357_35721

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

def B : Set ℝ := {x | |x - 3| > 1}

theorem union_A_complement_B_eq_closed_interval :
  A ∪ (U \ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 4} :=
by sorry

end union_A_complement_B_eq_closed_interval_l357_35721


namespace quadratic_minimum_min_value_is_zero_l357_35734

theorem quadratic_minimum (x : ℝ) : 
  (∀ y : ℝ, x^2 - 12*x + 36 ≤ y^2 - 12*y + 36) ↔ x = 6 :=
by sorry

theorem min_value_is_zero : 
  (6:ℝ)^2 - 12*(6:ℝ) + 36 = 0 :=
by sorry

end quadratic_minimum_min_value_is_zero_l357_35734


namespace zero_last_to_appear_l357_35720

/-- Fibonacci sequence modulo 9 -/
def fib_mod_9 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => (fib_mod_9 (n + 1) + fib_mod_9 n) % 9

/-- Function to check if a digit has appeared in the sequence up to n -/
def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ fib_mod_9 k = d

/-- Theorem stating that 0 is the last digit to appear -/
theorem zero_last_to_appear :
  ∃ n, digit_appears 0 n ∧
    ∀ d, d < 9 → d ≠ 0 → ∃ k, k < n ∧ digit_appears d k :=
  sorry

end zero_last_to_appear_l357_35720


namespace apples_bought_l357_35777

theorem apples_bought (apples pears : ℕ) : 
  pears = (3 * apples) / 5 →
  apples + pears = 240 →
  apples = 150 := by
sorry

end apples_bought_l357_35777


namespace unique_solution_iff_l357_35717

/-- The function f(x) = x^2 + 3bx + 4b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + 3*b*x + 4*b

/-- The property that |f(x)| ≤ 3 has exactly one solution -/
def has_unique_solution (b : ℝ) : Prop :=
  ∃! x : ℝ, |f b x| ≤ 3

/-- Theorem stating that the inequality has a unique solution iff b = 4/3 or b = 1 -/
theorem unique_solution_iff (b : ℝ) :
  has_unique_solution b ↔ b = 4/3 ∨ b = 1 :=
sorry

end unique_solution_iff_l357_35717


namespace percentage_problem_l357_35779

theorem percentage_problem : 
  ∃ x : ℝ, (0.62 * 150 - x / 100 * 250 = 43) ∧ (x = 20) := by
  sorry

end percentage_problem_l357_35779


namespace total_cookies_l357_35747

def cookies_per_bag : ℕ := 21
def bags_per_box : ℕ := 4
def number_of_boxes : ℕ := 2

theorem total_cookies : cookies_per_bag * bags_per_box * number_of_boxes = 168 := by
  sorry

end total_cookies_l357_35747


namespace all_configurations_exist_l357_35753

-- Define the geometric shapes
structure Rectangle where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  all_right_angles : ∀ i, angles i = 90
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3

structure Rhombus where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j, sides i = sides j
  opposite_angles_equal : angles 0 = angles 2 ∧ angles 1 = angles 3

structure Parallelogram where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ
  opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3
  adjacent_angles_supplementary : ∀ i, angles i + angles ((i + 1) % 4) = 180

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sides : Fin 4 → ℝ

structure Triangle where
  angles : Fin 3 → ℝ
  sum_of_angles : angles 0 + angles 1 + angles 2 = 180

-- Theorem stating that all configurations can exist
theorem all_configurations_exist :
  (∃ r : Rectangle, r.sides 0 ≠ r.sides 1) ∧
  (∃ rh : Rhombus, ∀ i, rh.angles i = 90) ∧
  (∃ p : Parallelogram, True) ∧
  (∃ q : Quadrilateral, (∀ i, q.angles i = 90) ∧ q.sides 0 ≠ q.sides 1) ∧
  (∃ t : Triangle, t.angles 0 = 100 ∧ t.angles 1 = 40 ∧ t.angles 2 = 40) :=
by sorry

end all_configurations_exist_l357_35753


namespace decimal_69_is_234_base5_l357_35725

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Converts a list of digits in base 5 to its decimal representation -/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem decimal_69_is_234_base5 :
  toBase5 69 = [4, 3, 2] ∧ fromBase5 [4, 3, 2] = 69 := by
  sorry

#eval toBase5 69  -- Should output [4, 3, 2]
#eval fromBase5 [4, 3, 2]  -- Should output 69

end decimal_69_is_234_base5_l357_35725


namespace quadrilateral_perimeter_l357_35749

/-- A quadrilateral ABCD with the following properties:
  1. AB ⊥ BC
  2. ∠DCB = 135°
  3. AB = 10 cm
  4. DC = 5 cm
  5. BC = 15 cm
-/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AB_perp_BC : True  -- Represents AB ⊥ BC
  angle_DCB : ℝ
  h_AB : AB = 10
  h_CD : CD = 5
  h_BC : BC = 15
  h_angle_DCB : angle_DCB = 135

/-- The perimeter of the quadrilateral ABCD is 30 + 5√10 cm -/
theorem quadrilateral_perimeter (q : Quadrilateral) : 
  q.AB + q.BC + q.CD + Real.sqrt (q.CD^2 + q.BC^2) = 30 + 5 * Real.sqrt 10 := by
  sorry


end quadrilateral_perimeter_l357_35749


namespace rachel_dvd_fraction_l357_35746

def total_earnings : ℚ := 200
def lunch_fraction : ℚ := 1/4
def money_left : ℚ := 50

theorem rachel_dvd_fraction :
  let lunch_cost : ℚ := lunch_fraction * total_earnings
  let money_after_lunch : ℚ := total_earnings - lunch_cost
  let dvd_cost : ℚ := money_after_lunch - money_left
  dvd_cost / total_earnings = 1/2 := by sorry

end rachel_dvd_fraction_l357_35746


namespace marble_probability_l357_35794

/-- Given two boxes of marbles with the following properties:
  1. The total number of marbles in both boxes is 24.
  2. The probability of drawing a black marble from each box is 28/45.
  This theorem states that the probability of drawing a white marble from each box is 2/135. -/
theorem marble_probability (box_a box_b : Finset ℕ) 
  (h_total : box_a.card + box_b.card = 24)
  (h_black_prob : (box_a.filter (λ x => x = 1)).card / box_a.card * 
                  (box_b.filter (λ x => x = 1)).card / box_b.card = 28/45) :
  (box_a.filter (λ x => x = 0)).card / box_a.card * 
  (box_b.filter (λ x => x = 0)).card / box_b.card = 2/135 :=
sorry

end marble_probability_l357_35794


namespace number_exceeding_16_percent_l357_35775

theorem number_exceeding_16_percent : ∃ x : ℝ, x = 0.16 * x + 21 ∧ x = 25 := by
  sorry

end number_exceeding_16_percent_l357_35775


namespace james_bags_given_away_l357_35743

def bags_given_away (initial_marbles : ℕ) (initial_bags : ℕ) (remaining_marbles : ℕ) : ℕ :=
  (initial_marbles - remaining_marbles) / (initial_marbles / initial_bags)

theorem james_bags_given_away :
  let initial_marbles : ℕ := 28
  let initial_bags : ℕ := 4
  let remaining_marbles : ℕ := 21
  bags_given_away initial_marbles initial_bags remaining_marbles = 1 := by
sorry

end james_bags_given_away_l357_35743


namespace gcd_9011_4379_l357_35759

theorem gcd_9011_4379 : Nat.gcd 9011 4379 = 1 := by
  sorry

end gcd_9011_4379_l357_35759


namespace lawn_width_proof_l357_35724

theorem lawn_width_proof (length : ℝ) (road_width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  length = 110 →
  road_width = 10 →
  total_cost = 4800 →
  cost_per_sqm = 3 →
  ∃ width : ℝ, width = 50 ∧ 
    (road_width * width + road_width * length) * cost_per_sqm = total_cost :=
by sorry

end lawn_width_proof_l357_35724


namespace unique_quadratic_root_l357_35713

-- Define the set A
def A (a b : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + b = 0}

-- State the theorem
theorem unique_quadratic_root (a b : ℝ) : A a b = {1} → a = -2 ∧ b = 1 := by
  sorry

end unique_quadratic_root_l357_35713


namespace multiplication_addition_equality_l357_35704

theorem multiplication_addition_equality : 15 * 30 + 45 * 15 + 15 * 15 = 1350 := by
  sorry

end multiplication_addition_equality_l357_35704


namespace round_table_seat_count_l357_35755

/-- Represents a circular seating arrangement -/
structure CircularTable where
  seatCount : ℕ
  kingArthurSeat : ℕ
  lancelotSeat : ℕ

/-- Checks if two seats are directly opposite in a circular arrangement -/
def areOpposite (table : CircularTable) : Prop :=
  (table.lancelotSeat - table.kingArthurSeat) % table.seatCount = table.seatCount / 2

/-- The theorem to be proved -/
theorem round_table_seat_count (table : CircularTable) :
  table.kingArthurSeat = 10 ∧ table.lancelotSeat = 29 ∧ areOpposite table →
  table.seatCount = 38 := by
  sorry


end round_table_seat_count_l357_35755


namespace y_equation_solution_l357_35754

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 4*y + 4/y + 1/y^2 = 35)
  (h2 : y = c + Real.sqrt d) : 
  c + d = 42 := by sorry

end y_equation_solution_l357_35754


namespace correct_observation_value_l357_35733

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : wrong_value = 23)
  (h4 : corrected_mean = 36.5) : 
  ∃ (correct_value : ℝ), correct_value = 48 ∧ 
    n * corrected_mean = n * initial_mean - wrong_value + correct_value :=
by sorry

end correct_observation_value_l357_35733


namespace rectangle_side_length_l357_35718

theorem rectangle_side_length (square_side : ℝ) (rect_side1 : ℝ) (rect_side2 : ℝ) :
  square_side = 5 →
  rect_side1 = 4 →
  square_side * square_side = rect_side1 * rect_side2 →
  rect_side2 = 6.25 := by
sorry

end rectangle_side_length_l357_35718


namespace radioactive_balls_solvable_l357_35701

/-- Represents a test strategy for identifying radioactive balls -/
structure TestStrategy where
  -- The number of tests used in the strategy
  num_tests : ℕ
  -- A function that, given the positions of the radioactive balls,
  -- returns true if the strategy successfully identifies both balls
  identifies_balls : Fin 11 → Fin 11 → Prop

/-- Represents the problem of finding radioactive balls -/
def RadioactiveBallsProblem :=
  ∃ (strategy : TestStrategy),
    strategy.num_tests ≤ 7 ∧
    ∀ (pos1 pos2 : Fin 11), pos1 ≠ pos2 →
      strategy.identifies_balls pos1 pos2

/-- The main theorem stating that the radioactive balls problem can be solved -/
theorem radioactive_balls_solvable : RadioactiveBallsProblem := by
  sorry


end radioactive_balls_solvable_l357_35701


namespace plot_length_is_63_meters_l357_35708

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length_difference : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the length of the plot given its properties. -/
def calculate_length (plot : RectangularPlot) : ℝ :=
  plot.breadth + plot.length_difference

/-- Calculates the perimeter of the plot. -/
def calculate_perimeter (plot : RectangularPlot) : ℝ :=
  2 * (calculate_length plot + plot.breadth)

/-- Theorem stating that under given conditions, the length of the plot is 63 meters. -/
theorem plot_length_is_63_meters (plot : RectangularPlot) 
  (h1 : plot.length_difference = 26)
  (h2 : plot.fencing_cost_per_meter = 26.5)
  (h3 : plot.total_fencing_cost = 5300)
  (h4 : calculate_perimeter plot = plot.total_fencing_cost / plot.fencing_cost_per_meter) :
  calculate_length plot = 63 := by
  sorry

end plot_length_is_63_meters_l357_35708


namespace l_shape_subdivision_l357_35710

/-- An L shape made of three congruent squares -/
structure LShape where
  -- We don't need to define the internal structure for this problem

/-- The number of L shapes with the same orientation as the original after n subdivisions -/
def same_orientation (n : ℕ) : ℕ :=
  4^(n-1) + 2^(n-1)

/-- The total number of L shapes after n subdivisions -/
def total_shapes (n : ℕ) : ℕ :=
  4^n

theorem l_shape_subdivision (n : ℕ) :
  n > 0 → same_orientation n ≤ total_shapes n ∧
  same_orientation n = (total_shapes (n-1) + 2^(n-1)) := by
  sorry

#eval same_orientation 2005

end l_shape_subdivision_l357_35710


namespace product_of_repeating_decimal_and_seven_l357_35784

theorem product_of_repeating_decimal_and_seven :
  ∃ (x : ℚ), (∀ n : ℕ, (x * 10^(3*n+3) - x * 10^(3*n)).num = 456 ∧ 
              (x * 10^(3*n+3) - x * 10^(3*n)).den = 10^3 - 1) →
  x * 7 = 1064 / 333 := by
  sorry

end product_of_repeating_decimal_and_seven_l357_35784


namespace hyperbola_eccentricity_l357_35788

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The ratio of y to x in the asymptotic equations -/
  asymptote_ratio : ℝ
  /-- The asymptotic equations are y = ± asymptote_ratio * x -/
  asymptote_eq : asymptote_ratio > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with asymptotic ratio 2/3 is √13/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : h.asymptote_ratio = 2/3) : 
  eccentricity h = Real.sqrt 13 / 3 :=
sorry

end hyperbola_eccentricity_l357_35788


namespace gcf_three_digit_palindromes_l357_35714

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b : ℕ), a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

theorem gcf_three_digit_palindromes :
  ∃ (g : ℕ), g > 0 ∧
    (∀ (n : ℕ), is_three_digit_palindrome n → g ∣ n) ∧
    (∀ (d : ℕ), d > 0 → (∀ (n : ℕ), is_three_digit_palindrome n → d ∣ n) → d ≤ g) ∧
    g = 101 := by sorry

end gcf_three_digit_palindromes_l357_35714


namespace range_of_a_l357_35760

-- Define a decreasing function on [-1,1]
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → f x > f y

-- Define the theorem
theorem range_of_a (f : ℝ → ℝ) (h_decreasing : DecreasingFunction f) :
  (∀ a, f (2*a - 3) < f (a - 2)) →
  (∀ a, a ∈ Set.Ioo 1 2) :=
sorry

end range_of_a_l357_35760


namespace bill_face_value_l357_35795

/-- Calculates the face value of a bill given the true discount, time period, and annual interest rate. -/
def calculate_face_value (true_discount : ℚ) (time_months : ℚ) (annual_rate : ℚ) : ℚ :=
  (true_discount * (100 + annual_rate * (time_months / 12))) / (annual_rate * (time_months / 12))

/-- Theorem stating that given the specified conditions, the face value of the bill is 2520. -/
theorem bill_face_value :
  let true_discount : ℚ := 270
  let time_months : ℚ := 9
  let annual_rate : ℚ := 16
  calculate_face_value true_discount time_months annual_rate = 2520 :=
by sorry

end bill_face_value_l357_35795


namespace minimum_questionnaires_to_mail_l357_35797

def response_rate : ℝ := 0.62
def required_responses : ℕ := 300

theorem minimum_questionnaires_to_mail : 
  ∃ n : ℕ, n > 0 ∧ 
  (↑n * response_rate : ℝ) ≥ required_responses ∧
  ∀ m : ℕ, m < n → (↑m * response_rate : ℝ) < required_responses :=
by
  sorry

end minimum_questionnaires_to_mail_l357_35797


namespace number_times_a_equals_7b_l357_35732

theorem number_times_a_equals_7b (a b x : ℝ) : 
  x * a = 7 * b → 
  x * a = 20 → 
  7 * b = 20 → 
  84 * a * b = 800 → 
  x = 1 := by
sorry

end number_times_a_equals_7b_l357_35732


namespace distinct_triangles_in_T_shape_l357_35768

/-- The number of distinct triangles formed by 7 points in a 'T' shape --/
def distinctTriangles (totalPoints numHorizontal numVertical : ℕ) : ℕ :=
  Nat.choose totalPoints 3 - (Nat.choose numHorizontal 3 + Nat.choose numVertical 3)

/-- Theorem stating that the number of distinct triangles is 24 --/
theorem distinct_triangles_in_T_shape :
  distinctTriangles 7 5 3 = 24 := by
  sorry

#eval distinctTriangles 7 5 3

end distinct_triangles_in_T_shape_l357_35768


namespace tan_beta_plus_pi_sixth_l357_35728

open Real

theorem tan_beta_plus_pi_sixth (α β : ℝ) 
  (h1 : tan (α - π/6) = 2) 
  (h2 : tan (α + β) = -3) : 
  tan (β + π/6) = 1 := by
sorry

end tan_beta_plus_pi_sixth_l357_35728


namespace unique_c_for_quadratic_equation_l357_35765

theorem unique_c_for_quadratic_equation :
  ∃! (c : ℝ), c ≠ 0 ∧
    (∃! (b : ℝ), b > 0 ∧
      (∃! (x : ℝ), x^2 + (2*b + 2/b)*x + c = 0)) ∧
    c = 4 := by
  sorry

end unique_c_for_quadratic_equation_l357_35765


namespace non_sum_sequence_inequality_l357_35772

/-- A sequence of positive integers where no element can be represented as the sum of two or more different elements from the sequence -/
def NonSumSequence (m : Nat → Nat) : Prop :=
  ∀ (i j k : Nat), i < j → j < k → m i + m j ≠ m k

theorem non_sum_sequence_inequality
  (m : Nat → Nat)  -- The sequence
  (s : Nat)        -- The length of the sequence
  (h_s : s ≥ 2)    -- s is at least 2
  (h_m : ∀ i j, i < j → j ≤ s → m i < m j)  -- The sequence is strictly increasing
  (h_non_sum : NonSumSequence m)  -- The non-sum property
  (r : Nat)        -- The parameter r
  (h_r : 1 ≤ r ∧ r < s)  -- r is between 1 and s-1
  : r * m r + m s ≥ (r + 1) * (s - 1) :=
sorry

end non_sum_sequence_inequality_l357_35772


namespace cookie_cutter_problem_l357_35767

/-- The number of hexagon-shaped cookie cutters -/
def num_hexagons (total_sides num_triangles num_squares : ℕ) : ℕ :=
  (total_sides - (3 * num_triangles + 4 * num_squares)) / 6

/-- Theorem stating that there are 2 hexagon-shaped cookie cutters -/
theorem cookie_cutter_problem : num_hexagons 46 6 4 = 2 := by
  sorry

end cookie_cutter_problem_l357_35767


namespace quadratic_roots_condition_double_root_at_three_l357_35727

/-- The quadratic equation (k-2)x^2 - 2x + 1 = 0 has two real roots if and only if k ≤ 3 and k ≠ 2.
    When k = 3, the equation has a double root at x = 1. -/
theorem quadratic_roots_condition (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (k - 2) * x₁^2 - 2 * x₁ + 1 = 0 ∧ (k - 2) * x₂^2 - 2 * x₂ + 1 = 0) ↔
  (k ≤ 3 ∧ k ≠ 2) :=
sorry

/-- When k = 3, the quadratic equation x^2 - 2x + 1 = 0 has a double root at x = 1. -/
theorem double_root_at_three :
  ∀ x : ℝ, x^2 - 2*x + 1 = 0 ↔ x = 1 :=
sorry

end quadratic_roots_condition_double_root_at_three_l357_35727


namespace tina_total_pens_l357_35785

/-- The number of pens Tina has -/
structure PenCount where
  pink : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ

/-- Conditions on Tina's pen count -/
def tina_pen_conditions (p : PenCount) : Prop :=
  p.pink = 15 ∧
  p.green = p.pink - 9 ∧
  p.blue = p.green + 3 ∧
  p.yellow = p.pink + p.green - 5

/-- Theorem stating the total number of pens Tina has -/
theorem tina_total_pens (p : PenCount) (h : tina_pen_conditions p) :
  p.pink + p.green + p.blue + p.yellow = 46 := by
  sorry

end tina_total_pens_l357_35785


namespace sum_of_roots_is_eight_l357_35716

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetry property
def symmetric_about_two (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 + x) = f (2 - x)

-- Define the property of having exactly four distinct real roots
def has_four_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0) ∧
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, f x = 0 → (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄))

-- State the theorem
theorem sum_of_roots_is_eight (f : ℝ → ℝ) 
  (h_sym : symmetric_about_two f) 
  (h_roots : has_four_distinct_roots f) : 
  ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0) ∧
    (r₁ + r₂ + r₃ + r₄ = 8) :=
sorry

end sum_of_roots_is_eight_l357_35716


namespace article_word_count_l357_35782

theorem article_word_count 
  (total_pages : ℕ) 
  (small_type_pages : ℕ) 
  (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) 
  (h1 : total_pages = 21) 
  (h2 : small_type_pages = 17) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) : 
  (total_pages - small_type_pages) * words_per_large_page + 
  small_type_pages * words_per_small_page = 48000 := by
sorry

end article_word_count_l357_35782


namespace extremum_implies_a_equals_3_l357_35751

/-- The function f(x) = x³ + 5x² + ax attains an extremum at x = -3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5*x^2 + a*x

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 10*x + a

theorem extremum_implies_a_equals_3 (a : ℝ) :
  (f' a (-3) = 0) → a = 3 := by
  sorry

#check extremum_implies_a_equals_3

end extremum_implies_a_equals_3_l357_35751


namespace annulus_area_l357_35796

/-- The area of an annulus formed by two concentric circles, where the radius of the smaller circle
    is 8 units and the radius of the larger circle is twice that of the smaller, is 192π square units. -/
theorem annulus_area (r₁ r₂ : ℝ) (h₁ : r₁ = 8) (h₂ : r₂ = 2 * r₁) :
  π * r₂^2 - π * r₁^2 = 192 * π := by
  sorry

#check annulus_area

end annulus_area_l357_35796


namespace unfair_coin_probability_l357_35758

def num_flips : ℕ := 8
def prob_tails : ℚ := 2/3
def num_tails : ℕ := 3

theorem unfair_coin_probability :
  (Nat.choose num_flips num_tails) * (prob_tails ^ num_tails) * ((1 - prob_tails) ^ (num_flips - num_tails)) = 448/6561 := by
sorry

end unfair_coin_probability_l357_35758


namespace computer_price_equation_l357_35750

/-- Represents the relationship between the original price, tax rate, and discount rate
    for a computer with a 30% price increase and final price of $351. -/
theorem computer_price_equation (c t d : ℝ) : 
  1.30 * c * (100 + t) * (100 - d) = 3510000 ↔ 
  (c * (1 + 0.3) * (1 + t / 100) * (1 - d / 100) = 351) :=
sorry

end computer_price_equation_l357_35750


namespace physics_chemistry_average_l357_35742

theorem physics_chemistry_average (physics chemistry math : ℝ) 
  (h1 : (physics + chemistry + math) / 3 = 80)
  (h2 : (physics + math) / 2 = 90)
  (h3 : physics = 80) :
  (physics + chemistry) / 2 = 70 := by
  sorry

end physics_chemistry_average_l357_35742


namespace circle_radius_l357_35793

theorem circle_radius (x y : ℝ) :
  2 * x^2 + 2 * y^2 - 10 = 2 * x + 4 * y →
  ∃ (center_x center_y : ℝ),
    (x - center_x)^2 + (y - center_y)^2 = 13/2 :=
by sorry

end circle_radius_l357_35793


namespace counterexample_exists_l357_35786

theorem counterexample_exists : ∃ (x y z : ℝ), 
  x > 0 ∧ y > 0 ∧ x > y ∧ z ≠ 0 ∧ |x + z| ≤ |y + z| := by
  sorry

end counterexample_exists_l357_35786


namespace gain_percent_calculation_l357_35709

theorem gain_percent_calculation (MP : ℝ) (MP_pos : MP > 0) : 
  let CP := 0.25 * MP
  let SP := 0.5 * MP
  let gain := SP - CP
  let gain_percent := (gain / CP) * 100
  gain_percent = 100 := by sorry

end gain_percent_calculation_l357_35709


namespace solution_sum_of_squares_l357_35757

-- Define the function f(t) = t^2 + sin(t)
noncomputable def f (t : ℝ) : ℝ := t^2 + Real.sin t

-- Define the equation
def equation (x y : ℝ) : Prop :=
  (4*x^2*y + 6*x^2 + 2*x*y - 4*x) / (3*x - y - 2) + 
  Real.sin ((3*x^2 + x*y + x - y - 2) / (3*x - y - 2)) = 
  2*x*y + y^2 + x^2/y^2 + 2*x/y + 
  (2*x*y*(x^2 + y^2)) / (3*x - y - 2)^2 + 
  (1 / (x + y)^2) * (x^2 * Real.sin ((x + y)^2 / x) + 
                     y^2 * Real.sin ((x + y)^2 / y^2) + 
                     2*x*y * Real.sin ((x + y)^2 / (3*x - y - 2)))

-- Theorem statement
theorem solution_sum_of_squares (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : equation x y) :
  x^2 + y^2 = (85 + 13 * Real.sqrt 17) / 32 := by
  sorry

end solution_sum_of_squares_l357_35757


namespace marys_age_l357_35771

theorem marys_age (mary_age rahul_age : ℕ) 
  (h1 : rahul_age = mary_age + 30)
  (h2 : rahul_age + 20 = 2 * (mary_age + 20)) : 
  mary_age = 10 := by
  sorry

end marys_age_l357_35771


namespace dividend_percentage_calculation_l357_35731

theorem dividend_percentage_calculation 
  (face_value : ℝ) 
  (purchase_price : ℝ) 
  (roi : ℝ) 
  (h1 : face_value = 50) 
  (h2 : purchase_price = 25) 
  (h3 : roi = 0.25) :
  let dividend_per_share := roi * purchase_price
  let dividend_percentage := (dividend_per_share / face_value) * 100
  dividend_percentage = 12.5 := by
sorry

end dividend_percentage_calculation_l357_35731


namespace marble_bag_count_l357_35719

theorem marble_bag_count (blue red white : ℕ) (total : ℕ) : 
  blue = 5 →
  red = 9 →
  total = blue + red + white →
  (red + white : ℚ) / total = 3/4 →
  total = 20 := by
sorry

end marble_bag_count_l357_35719


namespace right_angle_zdels_l357_35748

/-- The number of zdels in a full circle -/
def zdels_in_full_circle : ℕ := 400

/-- The fraction of a full circle that constitutes a right angle -/
def right_angle_fraction : ℚ := 1 / 3

/-- The number of zdels in a right angle -/
def zdels_in_right_angle : ℚ := zdels_in_full_circle * right_angle_fraction

theorem right_angle_zdels : zdels_in_right_angle = 400 / 3 := by
  sorry

end right_angle_zdels_l357_35748


namespace max_gcd_consecutive_terms_l357_35738

def a (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (a n) (a (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (a m) (a (m + 1)) = k ∧
  k = 3 :=
sorry

end max_gcd_consecutive_terms_l357_35738


namespace no_right_obtuse_triangle_l357_35764

-- Define a triangle
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define properties of triangles
def Triangle.isValid (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

def Triangle.hasRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.hasObtuseAngle (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: A right obtuse triangle cannot exist
theorem no_right_obtuse_triangle (t : Triangle) :
  t.isValid → ¬(t.hasRightAngle ∧ t.hasObtuseAngle) :=
by
  sorry


end no_right_obtuse_triangle_l357_35764


namespace selections_equal_sixteen_l357_35783

/-- The number of ways to select 3 people from 2 females and 4 males, with at least 1 female -/
def selectionsWithFemale (totalStudents femaleStudents maleStudents selectCount : ℕ) : ℕ :=
  Nat.choose totalStudents selectCount - Nat.choose maleStudents selectCount

/-- Proof that the number of selections is 16 -/
theorem selections_equal_sixteen :
  selectionsWithFemale 6 2 4 3 = 16 := by
  sorry

end selections_equal_sixteen_l357_35783


namespace tan_inequality_l357_35789

open Real

theorem tan_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < π/2) (h₃ : 0 < x₂) (h₄ : x₂ < π/2) (h₅ : x₁ ≠ x₂) :
  (tan x₁ + tan x₂) / 2 > tan ((x₁ + x₂) / 2) := by
  sorry

#check tan_inequality

end tan_inequality_l357_35789


namespace indeterminate_relation_l357_35773

theorem indeterminate_relation (x y : ℝ) (h : Real.exp (-x) + Real.log y < Real.exp (-y) + Real.log x) :
  ¬ (∀ p : Prop, p ∨ ¬p) := by
  sorry

end indeterminate_relation_l357_35773


namespace intersection_of_A_and_B_l357_35702

open Set

def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 > 0}
def B : Set ℝ := {x : ℝ | x - 1 < 0}

theorem intersection_of_A_and_B :
  A ∩ B = Iio 1 := by sorry

end intersection_of_A_and_B_l357_35702


namespace expression_evaluation_l357_35798

theorem expression_evaluation : 
  (4 * 6) / (12 * 18) * (9 * 12 * 18) / (4 * 6 * 9^2) = 1 / 9 := by
  sorry

end expression_evaluation_l357_35798


namespace combined_return_percentage_l357_35705

theorem combined_return_percentage (investment1 investment2 : ℝ) 
  (return_rate1 return_rate2 : ℝ) :
  investment1 = 500 →
  investment2 = 1500 →
  return_rate1 = 0.07 →
  return_rate2 = 0.27 →
  let total_investment := investment1 + investment2
  let total_return := investment1 * return_rate1 + investment2 * return_rate2
  let combined_return_rate := total_return / total_investment
  combined_return_rate = 0.22 := by
sorry

end combined_return_percentage_l357_35705


namespace sum_of_reciprocals_squared_l357_35706

noncomputable def p : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 2 * Real.sqrt 6
noncomputable def q : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 2 * Real.sqrt 6
noncomputable def r : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 2 * Real.sqrt 6
noncomputable def s : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 2 * Real.sqrt 6

theorem sum_of_reciprocals_squared :
  (1/p + 1/q + 1/r + 1/s)^2 = 3/16 := by sorry

end sum_of_reciprocals_squared_l357_35706


namespace original_calculation_result_l357_35735

theorem original_calculation_result (x : ℤ) (h : x - 63 = 24) : x - 36 = 51 := by
  sorry

end original_calculation_result_l357_35735


namespace ten_points_guarantees_win_ten_is_smallest_winning_score_l357_35741

/-- Represents the possible positions a team can finish in a match -/
inductive Position
| first
| second
| third
| fourth

/-- Returns the points awarded for a given position -/
def points_for_position (p : Position) : ℕ :=
  match p with
  | Position.first => 4
  | Position.second => 3
  | Position.third => 2
  | Position.fourth => 1

/-- Represents the results of a team in three matches -/
structure TeamResults :=
  (match1 : Position)
  (match2 : Position)
  (match3 : Position)

/-- Calculates the total points for a team's results -/
def total_points (results : TeamResults) : ℕ :=
  points_for_position results.match1 +
  points_for_position results.match2 +
  points_for_position results.match3

/-- Theorem: 10 points guarantees more points than any other team -/
theorem ten_points_guarantees_win :
  ∀ (results : TeamResults),
    total_points results ≥ 10 →
    ∀ (other_results : TeamResults),
      other_results ≠ results →
      total_points results > total_points other_results :=
by sorry

/-- Theorem: 10 is the smallest number of points that guarantees a win -/
theorem ten_is_smallest_winning_score :
  ∀ n : ℕ,
    n < 10 →
    ∃ (results other_results : TeamResults),
      total_points results = n ∧
      other_results ≠ results ∧
      total_points other_results ≥ total_points results :=
by sorry

end ten_points_guarantees_win_ten_is_smallest_winning_score_l357_35741


namespace reciprocal_sum_of_quadratic_roots_l357_35707

theorem reciprocal_sum_of_quadratic_roots (α' β' : ℝ) : 
  (∃ x y : ℝ, 7 * x^2 + 4 * x + 9 = 0 ∧ 7 * y^2 + 4 * y + 9 = 0 ∧ α' = 1/x ∧ β' = 1/y) →
  α' + β' = -4/9 := by
sorry

end reciprocal_sum_of_quadratic_roots_l357_35707


namespace binary_5_is_5_l357_35769

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_5 : List Bool := [true, false, true]

theorem binary_5_is_5 : binary_to_decimal binary_5 = 5 := by sorry

end binary_5_is_5_l357_35769


namespace polar_to_cartesian_line_l357_35722

/-- The polar equation r = 1 / (sin θ - cos θ) represents a line in Cartesian coordinates. -/
theorem polar_to_cartesian_line :
  ∀ (r θ : ℝ), r = 1 / (Real.sin θ - Real.cos θ) →
  ∃ (x y : ℝ), y - x = 1 :=
by sorry

end polar_to_cartesian_line_l357_35722


namespace cereal_eating_time_l357_35781

theorem cereal_eating_time (fat_rate mr_thin_rate total_cereal : ℚ) :
  fat_rate = 1 / 15 →
  mr_thin_rate = 1 / 45 →
  total_cereal = 4 →
  (total_cereal / (fat_rate + mr_thin_rate) = 45) :=
by sorry

end cereal_eating_time_l357_35781


namespace inequalities_hold_l357_35792

-- Define the points and lengths
variable (A B C D : ℝ) -- Representing points as real numbers for simplicity
variable (x y z : ℝ)

-- Define the conditions
axiom distinct_points : A < B ∧ B < C ∧ C < D
axiom length_AB : x = B - A
axiom length_AC : y = C - A
axiom length_AD : z = D - A
axiom positive_area : x > 0 ∧ (y - x) > 0 ∧ (z - y) > 0

-- Define the triangle inequality conditions
axiom triangle_inequality1 : x + (y - x) > z - y
axiom triangle_inequality2 : (y - x) + (z - y) > x
axiom triangle_inequality3 : x + (z - y) > y - x

-- State the theorem to be proved
theorem inequalities_hold : x < z / 2 ∧ y < x + z / 2 := by sorry

end inequalities_hold_l357_35792


namespace vector_sum_range_l357_35745

theorem vector_sum_range (A B : ℝ × ℝ) : 
  ((A.1 - 2)^2 + A.2^2 = 1) →
  ((B.1 - 2)^2 + B.2^2 = 1) →
  (A ≠ B) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 2) →
  (4 - Real.sqrt 2 ≤ ((A.1 + B.1)^2 + (A.2 + B.2)^2).sqrt) ∧
  (((A.1 + B.1)^2 + (A.2 + B.2)^2).sqrt ≤ 4 + Real.sqrt 2) :=
by sorry

end vector_sum_range_l357_35745


namespace lcm_one_to_five_l357_35770

theorem lcm_one_to_five : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 5))) = 60 := by
  sorry

end lcm_one_to_five_l357_35770


namespace triangle_lattice_points_l357_35780

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (t : Triangle) : ℚ := sorry

/-- Counts the number of lattice points on the boundary of a triangle -/
def boundaryPoints (t : Triangle) : ℕ := sorry

/-- Counts the total number of lattice points in and on a triangle -/
def totalLatticePoints (t : Triangle) : ℕ := sorry

theorem triangle_lattice_points :
  ∀ t : Triangle,
    t.p1 = ⟨5, 0⟩ →
    t.p2 = ⟨25, 0⟩ →
    triangleArea t = 200 →
    totalLatticePoints t = 221 := by
  sorry

end triangle_lattice_points_l357_35780


namespace student_ticket_price_l357_35711

theorem student_ticket_price (senior_price student_price : ℚ) : 
  (4 * senior_price + 3 * student_price = 79) →
  (12 * senior_price + 10 * student_price = 246) →
  student_price = 9 := by
sorry

end student_ticket_price_l357_35711


namespace derivative_at_zero_l357_35756

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    Real.arcsin (x^2 * Real.cos (1 / (9 * x))) + (2/3) * x
  else 
    0

-- State the theorem
theorem derivative_at_zero (f : ℝ → ℝ) : 
  (deriv f) 0 = 2/3 := by sorry

end derivative_at_zero_l357_35756


namespace f_max_value_l357_35736

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

-- Theorem stating that the maximum value of f is -3
theorem f_max_value : ∃ (M : ℝ), M = -3 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l357_35736


namespace c_value_l357_35766

/-- The constant k in the inverse relationship between a² and ⁴√c -/
def k : ℝ := 36

/-- The relationship between a and c -/
def relationship (a c : ℝ) : Prop := a^2 * c^(1/4) = k

/-- The value of a we're interested in -/
def a : ℝ := 4

/-- The theorem stating the value of c when a = 4 -/
theorem c_value : ∃ c : ℝ, relationship a c ∧ c = 25.62890625 := by sorry

end c_value_l357_35766


namespace cupcake_problem_l357_35703

theorem cupcake_problem (total : ℕ) (gf v nf : ℕ) (gf_and_v gf_and_nf nf_and_v gf_and_nf_not_v : ℕ) : 
  total = 200 →
  gf = (40 * total) / 100 →
  v = (25 * total) / 100 →
  nf = (30 * total) / 100 →
  gf_and_v = (20 * gf) / 100 →
  gf_and_nf = (15 * gf) / 100 →
  nf_and_v = (25 * nf) / 100 →
  gf_and_nf_not_v = (10 * total) / 100 →
  total - (gf + v + nf - gf_and_v - gf_and_nf - nf_and_v + gf_and_nf_not_v) = 33 := by
sorry

end cupcake_problem_l357_35703


namespace prob_draw_king_l357_35774

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the ranks in a standard deck -/
inductive Rank
  | Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents the suits in a standard deck -/
inductive Suit
  | Spades | Hearts | Diamonds | Clubs

/-- A card in the deck -/
structure Card :=
  (rank : Rank)
  (suit : Suit)

/-- The number of ranks in a standard deck -/
def rank_count : Nat := 13

/-- The number of suits in a standard deck -/
def suit_count : Nat := 4

/-- The total number of cards in a standard deck -/
def total_cards : Nat := rank_count * suit_count

/-- The number of Kings in a standard deck -/
def king_count : Nat := suit_count

/-- Theorem: The probability of drawing a King from a standard 52-card deck is 1/13 -/
theorem prob_draw_king (d : Deck) : 
  (king_count : ℚ) / total_cards = 1 / 13 := by sorry

end prob_draw_king_l357_35774


namespace salad_bar_olive_count_l357_35776

theorem salad_bar_olive_count (lettuce_types : Nat) (tomato_types : Nat) (soup_types : Nat) (total_options : Nat) (olive_types : Nat) : 
  lettuce_types = 2 →
  tomato_types = 3 →
  soup_types = 2 →
  total_options = 48 →
  total_options = lettuce_types * tomato_types * olive_types * soup_types →
  olive_types = 4 := by
sorry

end salad_bar_olive_count_l357_35776


namespace additional_discount_calculation_l357_35712

-- Define the manufacturer's suggested retail price (MSRP)
def MSRP : ℝ := 30

-- Define the regular discount range
def regularDiscountMin : ℝ := 0.1
def regularDiscountMax : ℝ := 0.3

-- Define the lowest possible price after all discounts
def lowestPrice : ℝ := 16.8

-- Define the additional discount percentage
def additionalDiscount : ℝ := 0.2

-- Theorem statement
theorem additional_discount_calculation :
  ∃ (regularDiscount : ℝ),
    regularDiscountMin ≤ regularDiscount ∧ 
    regularDiscount ≤ regularDiscountMax ∧
    MSRP * (1 - regularDiscount) * (1 - additionalDiscount) = lowestPrice :=
  sorry

end additional_discount_calculation_l357_35712


namespace complex_trig_simplification_l357_35744

open Complex

theorem complex_trig_simplification (θ : ℝ) :
  let z : ℂ := (cos θ - I * sin θ)^8 * (1 + I * tan θ)^5 / ((cos θ + I * sin θ)^2 * (tan θ + I))
  z = -1 / (cos θ)^4 * (sin (4*θ) + I * cos (4*θ)) :=
sorry

end complex_trig_simplification_l357_35744


namespace segments_parallel_iff_m_equals_two_thirds_l357_35740

/-- Given points A, B, C, D on a Cartesian plane, prove that segments AB and CD are parallel
    if and only if m = 2/3 -/
theorem segments_parallel_iff_m_equals_two_thirds 
  (A B C D : ℝ × ℝ) 
  (hA : A = (1, -1)) 
  (hB : B = (4, -2)) 
  (hC : C = (-1, 2)) 
  (hD : D = (3, m)) 
  (m : ℝ) : 
  (∃ k : ℝ, B.1 - A.1 = k * (D.1 - C.1) ∧ B.2 - A.2 = k * (D.2 - C.2)) ↔ m = 2/3 :=
by sorry

end segments_parallel_iff_m_equals_two_thirds_l357_35740


namespace hyperbola_equation_l357_35762

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (m : ℝ), m = b / a ∧ m = 2) →
  (∃ (x : ℝ), x^2 + (2*x + 10)^2 = a^2 + b^2) →
  a^2 = 5 ∧ b^2 = 20 := by
  sorry

end hyperbola_equation_l357_35762


namespace identity_function_proof_l357_35730

theorem identity_function_proof (f : ℕ → ℕ) 
  (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
  sorry

end identity_function_proof_l357_35730


namespace no_nonsquare_triple_divisors_l357_35799

theorem no_nonsquare_triple_divisors : 
  ¬ ∃ (N : ℕ+), (¬ ∃ (m : ℕ+), N = m * m) ∧ 
  (∃ (t : ℕ+), ∀ d : ℕ+, d ∣ N → ∃ (a b : ℕ+), (a ∣ N) ∧ (b ∣ N) ∧ (d * a * b = t)) :=
by sorry

end no_nonsquare_triple_divisors_l357_35799


namespace abs_negative_2023_l357_35791

theorem abs_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end abs_negative_2023_l357_35791


namespace minimize_y_l357_35790

variable (a b : ℝ)
def y (x : ℝ) := 3 * (x - a)^2 + (x - b)^2

theorem minimize_y :
  ∃ (x : ℝ), ∀ (z : ℝ), y a b x ≤ y a b z ∧ x = (3 * a + b) / 4 := by
  sorry

end minimize_y_l357_35790


namespace plane_perpendicular_through_perpendicular_line_line_not_perpendicular_in_perpendicular_planes_l357_35737

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (perpendicular : Plane → Plane → Prop)
variable (passes_through : Plane → Line → Prop)
variable (perpendicular_line : Line → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (line_of_intersection : Plane → Plane → Line)
variable (perpendicular_to_line : Line → Line → Prop)

-- Proposition 2
theorem plane_perpendicular_through_perpendicular_line 
  (p1 p2 : Plane) (l : Line) :
  perpendicular_line l p2 → passes_through p1 l → perpendicular p1 p2 :=
sorry

-- Proposition 4
theorem line_not_perpendicular_in_perpendicular_planes 
  (p1 p2 : Plane) (l : Line) :
  perpendicular p1 p2 →
  in_plane l p1 →
  ¬ perpendicular_to_line l (line_of_intersection p1 p2) →
  ¬ perpendicular_line l p2 :=
sorry

end plane_perpendicular_through_perpendicular_line_line_not_perpendicular_in_perpendicular_planes_l357_35737


namespace acme_cheaper_at_min_shirts_l357_35723

/-- Acme T-Shirt Company's pricing structure -/
def acme_cost (x : ℕ) : ℕ := 50 + 9 * x

/-- Beta T-shirt Company's pricing structure -/
def beta_cost (x : ℕ) : ℕ := 14 * x

/-- The minimum number of shirts for which Acme is cheaper than Beta -/
def min_shirts_for_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme_cheaper < beta_cost min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper →
    acme_cost n ≥ beta_cost n :=
by sorry

end acme_cheaper_at_min_shirts_l357_35723


namespace binomial_coefficient_divisibility_l357_35739

theorem binomial_coefficient_divisibility (p k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  p ∣ Nat.choose p k := by
  sorry

end binomial_coefficient_divisibility_l357_35739


namespace down_payment_amount_l357_35752

/-- Given a purchase with a payment plan, prove the down payment amount. -/
theorem down_payment_amount
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (num_payments : ℕ)
  (interest_rate : ℝ)
  (h1 : purchase_price = 110)
  (h2 : monthly_payment = 10)
  (h3 : num_payments = 12)
  (h4 : interest_rate = 9.090909090909092 / 100) :
  ∃ (down_payment : ℝ),
    down_payment + num_payments * monthly_payment =
      purchase_price + interest_rate * purchase_price ∧
    down_payment = 0 := by
  sorry

end down_payment_amount_l357_35752


namespace certain_age_problem_l357_35763

/-- Prove that the certain age is 30 years old given the conditions in the problem -/
theorem certain_age_problem (kody_current_age : ℕ) (mohamed_current_age : ℕ) (certain_age : ℕ) :
  kody_current_age = 32 →
  mohamed_current_age = 2 * certain_age →
  kody_current_age - 4 = (mohamed_current_age - 4) / 2 →
  certain_age = 30 := by
  sorry

end certain_age_problem_l357_35763


namespace count_not_divisible_by_8_or_7_l357_35787

def count_not_divisible (n : ℕ) (d₁ d₂ : ℕ) : ℕ :=
  n - (n / d₁ + n / d₂ - n / (lcm d₁ d₂))

theorem count_not_divisible_by_8_or_7 :
  count_not_divisible 1199 8 7 = 900 := by
  sorry

end count_not_divisible_by_8_or_7_l357_35787


namespace number_of_tests_l357_35715

/-- Proves the number of initial tests given average scores and lowest score -/
theorem number_of_tests
  (initial_average : ℝ)
  (lowest_score : ℝ)
  (new_average : ℝ)
  (h1 : initial_average = 90)
  (h2 : lowest_score = 75)
  (h3 : new_average = 95)
  (h4 : ∀ n : ℕ, n > 1 → (n * initial_average - lowest_score) / (n - 1) = new_average) :
  ∃ n : ℕ, n = 4 ∧ n > 1 := by
  sorry


end number_of_tests_l357_35715
