import Mathlib

namespace multiply_by_8050_equals_80_5_l849_84915

theorem multiply_by_8050_equals_80_5 : ∃ x : ℝ, 8050 * x = 80.5 ∧ x = 0.01 := by
  sorry

end multiply_by_8050_equals_80_5_l849_84915


namespace brownies_on_counter_l849_84959

def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def mother_added : ℕ := 24

theorem brownies_on_counter : 
  initial_brownies - father_ate - mooney_ate + mother_added = 36 := by
  sorry

end brownies_on_counter_l849_84959


namespace words_with_vowels_count_l849_84940

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 6752 :=
sorry

end words_with_vowels_count_l849_84940


namespace line_equation_correctness_l849_84993

/-- A line passing through a point with a given direction vector -/
structure DirectedLine (n : ℕ) where
  point : Fin n → ℝ
  direction : Fin n → ℝ

/-- The equation of a line in 2D space -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point satisfies a line equation -/
def satisfiesEquation (p : Fin 2 → ℝ) (eq : LineEquation) : Prop :=
  eq.a * p 0 + eq.b * p 1 + eq.c = 0

/-- Check if a vector is parallel to a line equation -/
def isParallel (v : Fin 2 → ℝ) (eq : LineEquation) : Prop :=
  eq.a * v 0 + eq.b * v 1 = 0

theorem line_equation_correctness (l : DirectedLine 2) (eq : LineEquation) :
  (l.point 0 = -3 ∧ l.point 1 = 1) →
  (l.direction 0 = 2 ∧ l.direction 1 = -3) →
  (eq.a = 3 ∧ eq.b = 2 ∧ eq.c = -11) →
  satisfiesEquation l.point eq ∧ isParallel l.direction eq :=
sorry

end line_equation_correctness_l849_84993


namespace virginia_sweettarts_l849_84916

/-- The number of Virginia's friends -/
def num_friends : ℕ := 4

/-- The number of Sweettarts each person ate -/
def sweettarts_per_person : ℕ := 3

/-- The initial number of Sweettarts Virginia had -/
def initial_sweettarts : ℕ := num_friends * sweettarts_per_person + sweettarts_per_person

theorem virginia_sweettarts : initial_sweettarts = 15 := by
  sorry

end virginia_sweettarts_l849_84916


namespace triangle_theorem_l849_84988

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.B)
  (h2 : t.b = 3)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = π / 3 ∧ t.a = Real.sqrt 3 ∧ t.c = 2 * Real.sqrt 3 := by
  sorry

end triangle_theorem_l849_84988


namespace sector_area_l849_84953

theorem sector_area (α : Real) (r : Real) (h1 : α = 2 * Real.pi / 3) (h2 : r = Real.sqrt 3) : 
  (1/2) * α * r^2 = Real.pi := by
  sorry

end sector_area_l849_84953


namespace inequality_proof_l849_84906

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4*x + y + 2*z) * (2*x + y + 8*z) ≥ (375/2) * x * y * z := by
  sorry

end inequality_proof_l849_84906


namespace solve_cubic_equation_l849_84970

theorem solve_cubic_equation :
  ∃ x : ℝ, x = -15.625 ∧ 3 * x^(1/3) - 5 * (x / x^(2/3)) = 10 + 2 * x^(1/3) :=
by sorry

end solve_cubic_equation_l849_84970


namespace geometric_sequence_property_l849_84939

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) := 
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 5 * a 7 = -3 * Real.sqrt 3 →
  a 2 * a 8 = 3 := by
sorry

end geometric_sequence_property_l849_84939


namespace last_digit_of_base4_conversion_l849_84945

def base5_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

def decimal_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

def base5_number : List Nat := [4, 3, 2, 1]

theorem last_digit_of_base4_conversion :
  (decimal_to_base4 (base5_to_decimal base5_number)).getLast? = some 2 := by
  sorry

end last_digit_of_base4_conversion_l849_84945


namespace event_A_necessary_not_sufficient_for_B_l849_84921

/- Define the bag contents -/
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2

/- Define the events -/
def event_A (drawn_red : ℕ) : Prop := drawn_red ≥ 1
def event_B (drawn_red : ℕ) : Prop := drawn_red = 1

/- Define the relationship between events -/
theorem event_A_necessary_not_sufficient_for_B :
  (∀ (drawn_red : ℕ), event_B drawn_red → event_A drawn_red) ∧
  (∃ (drawn_red : ℕ), event_A drawn_red ∧ ¬event_B drawn_red) :=
sorry

end event_A_necessary_not_sufficient_for_B_l849_84921


namespace b_profit_is_4000_l849_84967

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  total_profit : ℕ
  a_investment_ratio : ℕ
  a_time_ratio : ℕ

/-- Calculates B's profit in the joint business venture -/
def calculate_b_profit (jb : JointBusiness) : ℕ :=
  jb.total_profit / (jb.a_investment_ratio * jb.a_time_ratio + 1)

/-- Theorem stating that B's profit is 4000 given the specified conditions -/
theorem b_profit_is_4000 (jb : JointBusiness) 
  (h1 : jb.total_profit = 28000)
  (h2 : jb.a_investment_ratio = 3)
  (h3 : jb.a_time_ratio = 2) : 
  calculate_b_profit jb = 4000 := by
  sorry

#eval calculate_b_profit { total_profit := 28000, a_investment_ratio := 3, a_time_ratio := 2 }

end b_profit_is_4000_l849_84967


namespace volleyball_team_starters_l849_84989

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem volleyball_team_starters (total_players triplets twins starters : ℕ) 
  (h1 : total_players = 16)
  (h2 : triplets = 3)
  (h3 : twins = 2)
  (h4 : starters = 6) :
  (choose (total_players - triplets - twins) starters) + 
  (triplets * choose (total_players - triplets - twins) (starters - 1)) +
  (twins * choose (total_players - triplets - twins) (starters - 1)) = 2772 := by
  sorry

end volleyball_team_starters_l849_84989


namespace imaginary_number_on_real_axis_l849_84918

theorem imaginary_number_on_real_axis (z : ℂ) :
  (∃ b : ℝ, z = b * I) →  -- z is a pure imaginary number
  (∃ r : ℝ, (z + 2) / (1 - I) = r) →  -- point lies on real axis
  z = -2 * I :=
by sorry

end imaginary_number_on_real_axis_l849_84918


namespace unique_line_intersection_l849_84999

theorem unique_line_intersection (m b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! k : ℝ, ∃ y1 y2 : ℝ, 
    y1 = k^2 - 2*k + 3 ∧ 
    y2 = m*k + b ∧ 
    |y1 - y2| = 4)
  (h3 : m * 2 + b = 8) : 
  m = 0 ∧ b = 8 := by
  sorry

end unique_line_intersection_l849_84999


namespace expression_evaluation_l849_84917

theorem expression_evaluation : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 := by
  sorry

end expression_evaluation_l849_84917


namespace tenth_term_of_sequence_l849_84922

theorem tenth_term_of_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, a n = n * (n + 1) / 2) : 
  a 10 = 55 := by sorry

end tenth_term_of_sequence_l849_84922


namespace expression_has_17_digits_l849_84942

-- Define a function to calculate the number of digits in a number
def numDigits (n : ℕ) : ℕ := sorry

-- Define the expression
def expression : ℕ := (8 * 10^10) * (10 * 10^5)

-- Theorem statement
theorem expression_has_17_digits : numDigits expression = 17 := by sorry

end expression_has_17_digits_l849_84942


namespace dividend_rate_for_given_stock_l849_84961

/-- Represents a stock with its characteristics -/
structure Stock where
  nominal_percentage : ℝ  -- The nominal percentage of the stock
  quote : ℝ             -- The quoted price of the stock
  yield : ℝ              -- The yield of the stock as a percentage

/-- Calculates the dividend rate of a stock -/
def dividend_rate (s : Stock) : ℝ :=
  s.nominal_percentage

/-- Theorem stating that for a 25% stock quoted at 125 with a 20% yield, the dividend rate is 25 -/
theorem dividend_rate_for_given_stock :
  let s : Stock := { nominal_percentage := 25, quote := 125, yield := 20 }
  dividend_rate s = 25 := by
  sorry


end dividend_rate_for_given_stock_l849_84961


namespace units_digit_of_n_l849_84972

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 15^4 →
  m % 10 = 9 →
  n % 10 = 5 := by
sorry

end units_digit_of_n_l849_84972


namespace sum_of_solutions_is_zero_l849_84935

theorem sum_of_solutions_is_zero (y : ℝ) (x₁ x₂ : ℝ) : 
  y = 6 → x₁^2 + y^2 = 145 → x₂^2 + y^2 = 145 → x₁ + x₂ = 0 := by
sorry

end sum_of_solutions_is_zero_l849_84935


namespace arc_ray_configuration_theorem_l849_84933

/-- Given a geometric configuration with circular arcs and rays, 
    we define constants u_ij and v_ij. This theorem proves a relationship between these constants. -/
theorem arc_ray_configuration_theorem 
  (u12 v12 u23 v23 : ℝ) 
  (h1 : u12 = v12) 
  (h2 : u12 = v23) 
  (h3 : u23 = v12) : 
  u23 = v23 := by sorry

end arc_ray_configuration_theorem_l849_84933


namespace partnership_investment_period_ratio_l849_84919

/-- Proves that the ratio of investment periods is 2:1 given the partnership conditions --/
theorem partnership_investment_period_ratio :
  ∀ (investment_A investment_B period_A period_B profit_A profit_B : ℚ),
    investment_A = 3 * investment_B →
    ∃ k : ℚ, period_A = k * period_B →
    profit_B = 4500 →
    profit_A + profit_B = 31500 →
    profit_A / profit_B = (investment_A * period_A) / (investment_B * period_B) →
    period_A / period_B = 2 := by
  sorry

end partnership_investment_period_ratio_l849_84919


namespace max_days_to_eat_candies_l849_84907

/-- The total number of candies Vasya received -/
def total_candies : ℕ := 777

/-- The function that calculates the total number of candies eaten over n days,
    where a is the number of candies eaten on the first day -/
def candies_eaten (n a : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- The proposition that states 37 is the maximum number of days
    in which Vasya can eat all the candies -/
theorem max_days_to_eat_candies :
  ∃ (a : ℕ), candies_eaten 37 a = total_candies ∧
  ∀ (n : ℕ), n > 37 → ∀ (b : ℕ), candies_eaten n b ≠ total_candies :=
sorry

end max_days_to_eat_candies_l849_84907


namespace mean_temperature_l849_84991

def temperatures : List ℚ := [-8, -5, -5, -6, 0, 4]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : ℚ) = -10/3 := by
sorry

end mean_temperature_l849_84991


namespace inverse_mod_million_l849_84964

theorem inverse_mod_million (C D M : Nat) : 
  C = 123456 → 
  D = 142857 → 
  M = 814815 → 
  (C * D * M) % 1000000 = 1 :=
by sorry

end inverse_mod_million_l849_84964


namespace laptop_price_difference_l849_84980

/-- The list price of Laptop Y -/
def list_price : ℝ := 69.80

/-- The discount percentage at Tech Giant -/
def tech_giant_discount : ℝ := 0.15

/-- The fixed discount amount at EconoTech -/
def econotech_discount : ℝ := 10

/-- The sale price at Tech Giant -/
def tech_giant_price : ℝ := list_price * (1 - tech_giant_discount)

/-- The sale price at EconoTech -/
def econotech_price : ℝ := list_price - econotech_discount

/-- The price difference between EconoTech and Tech Giant in dollars -/
def price_difference : ℝ := econotech_price - tech_giant_price

theorem laptop_price_difference :
  ⌊price_difference * 100⌋ = 47 := by sorry

end laptop_price_difference_l849_84980


namespace expand_expression_l849_84979

theorem expand_expression (x y : ℝ) : 
  5 * (3 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 15 * x^2 * y - 20 * x * y^2 + 10 * y^3 := by
  sorry

end expand_expression_l849_84979


namespace lisa_marbles_theorem_distribution_satisfies_conditions_l849_84994

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  let total_needed := (num_friends * (num_friends + 3)) / 2
  max (total_needed - initial_marbles) 0

/-- Proof that 40 additional marbles are needed for Lisa's scenario -/
theorem lisa_marbles_theorem :
  additional_marbles_needed 12 50 = 40 := by
  sorry

/-- Verify that the distribution satisfies the conditions -/
theorem distribution_satisfies_conditions 
  (num_friends : ℕ) 
  (initial_marbles : ℕ) 
  (h : num_friends > 0) :
  let additional := additional_marbles_needed num_friends initial_marbles
  let total := initial_marbles + additional
  (∀ i : ℕ, i > 0 ∧ i ≤ num_friends → i + 1 ≤ total / num_friends) ∧ 
  (∀ i j : ℕ, i > 0 ∧ j > 0 ∧ i ≤ num_friends ∧ j ≤ num_friends ∧ i ≠ j → i + 1 ≠ j + 1) := by
  sorry

end lisa_marbles_theorem_distribution_satisfies_conditions_l849_84994


namespace smallest_yellow_marbles_l849_84997

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 12 = 0) 
  (h2 : n ≥ 24) : ∃ (blue red green yellow : ℕ),
  blue = n / 3 ∧ 
  red = n / 4 ∧ 
  green = 6 ∧ 
  yellow = n - (blue + red + green) ∧ 
  blue + red + green + yellow = n ∧
  yellow ≥ 4 ∧
  (∀ m : ℕ, m < n → ¬(∃ b r g y : ℕ, 
    b = m / 3 ∧ 
    r = m / 4 ∧ 
    g = 6 ∧ 
    y = m - (b + r + g) ∧ 
    b + r + g + y = m ∧ 
    y ≥ 4)) :=
by sorry

end smallest_yellow_marbles_l849_84997


namespace unique_cube_difference_nineteen_l849_84909

theorem unique_cube_difference_nineteen :
  ∃! (x y : ℕ), x^3 - y^3 = 19 ∧ x = 3 ∧ y = 2 :=
by sorry

end unique_cube_difference_nineteen_l849_84909


namespace direct_proportion_function_m_l849_84924

theorem direct_proportion_function_m (m : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 3) * x^(m^2 - 8) = k * x) ↔ m = -3 :=
by sorry

end direct_proportion_function_m_l849_84924


namespace expected_weekly_rainfall_l849_84914

def days_in_week : ℕ := 7

def probability_sun : ℝ := 0.3
def probability_light_rain : ℝ := 0.5
def probability_heavy_rain : ℝ := 0.2

def light_rain_amount : ℝ := 3
def heavy_rain_amount : ℝ := 8

def daily_expected_rainfall : ℝ :=
  probability_sun * 0 + 
  probability_light_rain * light_rain_amount + 
  probability_heavy_rain * heavy_rain_amount

theorem expected_weekly_rainfall : 
  days_in_week * daily_expected_rainfall = 21.7 := by
  sorry

end expected_weekly_rainfall_l849_84914


namespace triangle_area_inequality_l849_84996

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define an interior point of a triangle
def is_interior_point (P : Point) (t : Triangle) : Prop := sorry

-- Define parallel lines
def parallel_line (P : Point) (l : Line) : Line := sorry

-- Define the division of a triangle by parallel lines
def divide_triangle (t : Triangle) (P : Point) : Prop := sorry

-- Define the areas of the smaller triangles
def small_triangle_areas (t : Triangle) (P : Point) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_area_inequality (ABC : Triangle) (P : Point) :
  is_interior_point P ABC →
  divide_triangle ABC P →
  let (S1, S2, S3) := small_triangle_areas ABC P
  area ABC ≤ 3 * (S1 + S2 + S3) := by
  sorry

end triangle_area_inequality_l849_84996


namespace min_magnitude_a_minus_c_l849_84974

noncomputable section

-- Define the plane vectors
variable (a b c : ℝ × ℝ)

-- Define the conditions
def magnitude_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
def magnitude_b_minus_c : ℝ := Real.sqrt (((b.1 - c.1) ^ 2) + ((b.2 - c.2) ^ 2))
def angle_between_a_and_b : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (magnitude_a a * Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))))

-- State the theorem
theorem min_magnitude_a_minus_c (h1 : magnitude_a a = 2)
                                (h2 : magnitude_b_minus_c b c = 1)
                                (h3 : angle_between_a_and_b a b = π / 3) :
  ∃ (min_value : ℝ), ∀ (a' b' c' : ℝ × ℝ),
    magnitude_a a' = 2 →
    magnitude_b_minus_c b' c' = 1 →
    angle_between_a_and_b a' b' = π / 3 →
    Real.sqrt (((a'.1 - c'.1) ^ 2) + ((a'.2 - c'.2) ^ 2)) ≥ min_value ∧
    min_value = Real.sqrt 3 - 1 :=
  sorry

end

end min_magnitude_a_minus_c_l849_84974


namespace paint_for_smaller_statues_l849_84963

/-- The amount of paint (in pints) required for a statue of given height (in feet) -/
def paint_required (height : ℝ) : ℝ := sorry

/-- The number of statues to be painted -/
def num_statues : ℕ := 320

/-- The height (in feet) of the original statue -/
def original_height : ℝ := 8

/-- The height (in feet) of the new statues -/
def new_height : ℝ := 2

/-- The amount of paint (in pints) required for the original statue -/
def original_paint : ℝ := 2

theorem paint_for_smaller_statues :
  paint_required new_height * num_statues = 10 :=
by sorry

end paint_for_smaller_statues_l849_84963


namespace hat_number_sum_l849_84949

/-- A four-digit perfect square number with tens digit 0 and non-zero units digit -/
def ValidNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^2 ∧ (n / 10) % 10 = 0 ∧ n % 10 ≠ 0

/-- The property that two numbers have the same units digit -/
def SameUnitsDigit (a b : ℕ) : Prop := a % 10 = b % 10

/-- The property that a number has an even units digit -/
def EvenUnitsDigit (n : ℕ) : Prop := n % 2 = 0

theorem hat_number_sum :
  ∀ a b c : ℕ,
    ValidNumber a ∧ ValidNumber b ∧ ValidNumber c ∧
    SameUnitsDigit b c ∧
    EvenUnitsDigit a ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a + b + c = 14612 := by
  sorry

end hat_number_sum_l849_84949


namespace vector_subtraction_l849_84968

theorem vector_subtraction :
  let v₁ : Fin 3 → ℝ := ![-2, 5, -1]
  let v₂ : Fin 3 → ℝ := ![7, -3, 6]
  v₁ - v₂ = ![-9, 8, -7] := by sorry

end vector_subtraction_l849_84968


namespace hiking_rate_ratio_l849_84946

-- Define the given constants
def rate_up : ℝ := 6
def time_up : ℝ := 2
def distance_down : ℝ := 18

-- Define the theorem
theorem hiking_rate_ratio :
  let distance_up := rate_up * time_up
  let time_down := time_up
  let rate_down := distance_down / time_down
  rate_down / rate_up = 1.5 := by
sorry

end hiking_rate_ratio_l849_84946


namespace lap_distance_l849_84910

theorem lap_distance (boys_laps girls_laps girls_miles : ℚ) : 
  boys_laps = 27 →
  girls_laps = boys_laps + 9 →
  girls_miles = 27 →
  girls_miles / girls_laps = 3/4 := by
  sorry

end lap_distance_l849_84910


namespace smallest_n_for_2012_terms_l849_84969

theorem smallest_n_for_2012_terms (n : ℕ) : (∀ m : ℕ, (m + 1)^2 ≥ 2012 → n ≤ m) ↔ n = 44 := by
  sorry

end smallest_n_for_2012_terms_l849_84969


namespace fraction_simplification_l849_84954

theorem fraction_simplification :
  (3^1006 + 3^1004) / (3^1006 - 3^1004) = 5/4 := by
  sorry

end fraction_simplification_l849_84954


namespace singing_competition_stats_l849_84908

def scores : List ℝ := [9.40, 9.40, 9.50, 9.50, 9.50, 9.60, 9.60, 9.60, 9.60, 9.60, 9.70, 9.70, 9.70, 9.70, 9.80, 9.80, 9.80, 9.90]

def median (l : List ℝ) : ℝ := sorry

def mode (l : List ℝ) : ℝ := sorry

theorem singing_competition_stats :
  median scores = 9.60 ∧ mode scores = 9.60 := by sorry

end singing_competition_stats_l849_84908


namespace perpendicular_line_through_point_l849_84900

/-- The line passing through (-1, 1) and perpendicular to x + y = 0 has equation x - y + 2 = 0 -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + 1 = 0 ∧ y - 1 = 0) →  -- Point (-1, 1)
  (∀ x y, x + y = 0 → True) →  -- Given line x + y = 0
  x - y + 2 = 0 := by  -- Resulting perpendicular line
sorry

end perpendicular_line_through_point_l849_84900


namespace latus_rectum_of_parabola_l849_84965

/-- The equation of the latus rectum of the parabola y = -1/4 * x^2 is y = 1 -/
theorem latus_rectum_of_parabola :
  ∀ (x y : ℝ), y = -(1/4) * x^2 → (∃ (x₀ : ℝ), y = 1 ∧ x₀^2 = -4*y) :=
by sorry

end latus_rectum_of_parabola_l849_84965


namespace segment_division_problem_l849_84987

/-- The problem of determining the number of parts a unit segment is divided into -/
theorem segment_division_problem (min_distance : ℚ) (h1 : min_distance = 0.02857142857142857) : 
  (1 : ℚ) / min_distance = 35 := by
  sorry

end segment_division_problem_l849_84987


namespace paiges_dresser_capacity_l849_84932

/-- Calculates the total number of clothing pieces a dresser can hold. -/
def dresser_capacity (drawers : ℕ) (pieces_per_drawer : ℕ) : ℕ :=
  drawers * pieces_per_drawer

/-- Proves that Paige's dresser can hold 8 pieces of clothing. -/
theorem paiges_dresser_capacity :
  dresser_capacity 4 2 = 8 := by
  sorry

end paiges_dresser_capacity_l849_84932


namespace f_satisfies_conditions_l849_84995

def f (m n : ℕ) : ℕ := m * n

theorem f_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := by
sorry

end f_satisfies_conditions_l849_84995


namespace exponent_multiplication_l849_84944

theorem exponent_multiplication (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end exponent_multiplication_l849_84944


namespace square_area_is_25_l849_84941

/-- A square in the coordinate plane with given vertex coordinates -/
structure Square where
  x₁ : ℝ
  x₂ : ℝ

/-- The area of the square -/
def square_area (s : Square) : ℝ :=
  (5 : ℝ) ^ 2

/-- Theorem stating that the area of the square is 25 -/
theorem square_area_is_25 (s : Square) : square_area s = 25 := by
  sorry


end square_area_is_25_l849_84941


namespace nested_cube_root_simplification_l849_84927

theorem nested_cube_root_simplification (N : ℝ) (h : N > 1) :
  (N^3 * (N^5 * N^3)^(1/3))^(1/3) = N^(5/3) := by
  sorry

end nested_cube_root_simplification_l849_84927


namespace min_cos_sum_with_tan_product_l849_84934

theorem min_cos_sum_with_tan_product (x y m : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hm : m > 2) 
  (h : Real.tan x * Real.tan y = m) : 
  Real.cos x + Real.cos y ≥ 2 := by
  sorry

end min_cos_sum_with_tan_product_l849_84934


namespace finite_solutions_equation_l849_84977

theorem finite_solutions_equation :
  ∃ (S : Finset (ℕ × ℕ)), ∀ m n : ℕ,
    m^2 + 2 * 3^n = m * (2^(n+1) - 1) → (m, n) ∈ S := by
  sorry

end finite_solutions_equation_l849_84977


namespace range_of_a_l849_84958

def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : IsDecreasing f)
  (h_odd : IsOdd f)
  (h_domain : ∀ x, x ∈ Set.Ioo (-1) 1 → f x ∈ Set.univ)
  (h_condition : f (1 - a) + f (1 - 2*a) < 0) :
  0 < a ∧ a < 2/3 := by
sorry

end range_of_a_l849_84958


namespace pentadecagon_triangles_l849_84985

/-- A regular pentadecagon is a 15-sided regular polygon -/
def regular_pentadecagon : ℕ := 15

/-- The number of vertices to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Proposition: The number of triangles formed by the vertices of a regular pentadecagon is 455 -/
theorem pentadecagon_triangles :
  (regular_pentadecagon.choose triangle_vertices) = 455 :=
sorry

end pentadecagon_triangles_l849_84985


namespace expression_simplification_l849_84955

theorem expression_simplification :
  (2^1002 + 5^1003)^2 - (2^1002 - 5^1003)^2 = 20 * 10^1002 := by
  sorry

end expression_simplification_l849_84955


namespace complex_magnitude_l849_84962

theorem complex_magnitude (z : ℂ) (h : z * (1 + 2*I) = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l849_84962


namespace hyperbola_asymptote_l849_84923

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Theorem statement
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) 
  (h2 : hyperbola a (-2) 1) :
  ∃ (k : ℝ), k = 1 ∨ k = -1 ∧ 
  ∀ (x y : ℝ), (x + k*y = 0) ↔ (∀ ε > 0, ∃ t > 0, ∀ t' ≥ t, 
    ∃ x' y', hyperbola a x' y' ∧ 
    ((x' - x)^2 + (y' - y)^2 < ε^2)) :=
by sorry

end hyperbola_asymptote_l849_84923


namespace local_max_derivative_range_l849_84901

/-- Given a function f with derivative f'(x) = a(x + 1)(x - a) and a local maximum at x = a, 
    prove that a is in the open interval (-1, 0) -/
theorem local_max_derivative_range (f : ℝ → ℝ) (a : ℝ) 
  (h₁ : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h₂ : IsLocalMax f a) : 
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end local_max_derivative_range_l849_84901


namespace system_solution_l849_84990

theorem system_solution : 
  ∃ (x y z u : ℚ), 
    (x = 229 ∧ y = 149 ∧ z = 131 ∧ u = 121) ∧
    (x + y = 3/2 * (z + u)) ∧
    (x + z = -4/3 * (y + u)) ∧
    (x + u = 5/4 * (y + z)) := by
  sorry

end system_solution_l849_84990


namespace tan_thirty_degrees_l849_84929

theorem tan_thirty_degrees : Real.tan (30 * π / 180) = Real.sqrt 3 / 3 := by
  sorry

end tan_thirty_degrees_l849_84929


namespace amount_subtracted_l849_84905

theorem amount_subtracted (number : ℝ) (result : ℝ) (amount : ℝ) : 
  number = 150 →
  result = 50 →
  0.60 * number - amount = result →
  amount = 40 := by
sorry

end amount_subtracted_l849_84905


namespace increasing_function_range_function_below_one_range_function_range_when_a_geq_two_l849_84943

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * abs (x - a)

-- Theorem 1
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x + x < f a y + y) ↔ -1 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem function_below_one_range (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < 1) ↔ 3/2 < a ∧ a < 2 := by sorry

-- Theorem 3 (partial, due to multiple conditions)
theorem function_range_when_a_geq_two (a : ℝ) (h : a ≥ 2) :
  ∃ l u : ℝ, ∀ x : ℝ, x ∈ Set.Icc 2 4 → l ≤ f a x ∧ f a x ≤ u := by sorry

end increasing_function_range_function_below_one_range_function_range_when_a_geq_two_l849_84943


namespace max_area_rectangle_l849_84976

theorem max_area_rectangle (perimeter : ℕ) (area : ℕ → ℕ → ℕ) :
  perimeter = 150 →
  (∀ w h : ℕ, area w h = w * h) →
  (∀ w h : ℕ, 2 * w + 2 * h = perimeter → area w h ≤ 1406) ∧
  (∃ w h : ℕ, 2 * w + 2 * h = perimeter ∧ area w h = 1406) :=
by sorry

end max_area_rectangle_l849_84976


namespace worksheets_graded_l849_84951

theorem worksheets_graded (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (problems_left : ℕ) : 
  total_worksheets = 9 →
  problems_per_worksheet = 4 →
  problems_left = 16 →
  total_worksheets - (problems_left / problems_per_worksheet) = 5 := by
sorry

end worksheets_graded_l849_84951


namespace figure2_total_length_l849_84982

/-- A rectangle-like shape composed of perpendicular line segments -/
structure RectangleShape :=
  (left : ℝ)
  (bottom : ℝ)
  (right : ℝ)
  (top : ℝ)

/-- Calculate the total length of segments in the shape -/
def total_length (shape : RectangleShape) : ℝ :=
  shape.left + shape.bottom + shape.right + shape.top

/-- The theorem stating that the total length of segments in Figure 2 is 23 units -/
theorem figure2_total_length :
  let figure2 : RectangleShape := {
    left := 10,
    bottom := 5,
    right := 7,
    top := 1
  }
  total_length figure2 = 23 := by sorry

end figure2_total_length_l849_84982


namespace puzzle_solution_l849_84920

/-- A function that represents the puzzle rule --/
def puzzleRule (n : ℕ) : ℕ := sorry

/-- The puzzle conditions --/
axiom rule_111 : puzzleRule 111 = 9
axiom rule_444 : puzzleRule 444 = 12
axiom rule_777 : puzzleRule 777 = 15

/-- The theorem to prove --/
theorem puzzle_solution : ∃ (n : ℕ), puzzleRule n = 15 ∧ n = 777 := by sorry

end puzzle_solution_l849_84920


namespace equality_and_inequality_of_exponents_l849_84912

theorem equality_and_inequality_of_exponents (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : (2 : ℝ)^x = (3 : ℝ)^y ∧ (3 : ℝ)^y = (4 : ℝ)^z) :
  2 * x = 4 * z ∧ 2 * x > 3 * y :=
by sorry

end equality_and_inequality_of_exponents_l849_84912


namespace systematic_sampling_third_group_l849_84973

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) : ℕ → ℕ :=
  fun n => firstSelected + (n - 1) * (totalItems / sampleSize)

theorem systematic_sampling_third_group 
  (totalItems : ℕ) 
  (sampleSize : ℕ) 
  (groupSize : ℕ) 
  (numGroups : ℕ) 
  (firstSelected : ℕ) :
  totalItems = 300 →
  sampleSize = 20 →
  groupSize = 20 →
  numGroups = 15 →
  firstSelected = 6 →
  totalItems = groupSize * numGroups →
  systematicSample totalItems sampleSize firstSelected 3 = 36 := by
  sorry

#check systematic_sampling_third_group

end systematic_sampling_third_group_l849_84973


namespace unknown_number_value_l849_84957

theorem unknown_number_value (x n : ℝ) : 
  x = 12 → 5 + n / x = 6 - 5 / x → n = 7 := by
  sorry

end unknown_number_value_l849_84957


namespace mikes_games_this_year_l849_84948

def total_games : ℕ := 54
def last_year_games : ℕ := 39
def missed_games : ℕ := 41

theorem mikes_games_this_year : 
  total_games - last_year_games = 15 := by sorry

end mikes_games_this_year_l849_84948


namespace sector_max_area_l849_84904

/-- The maximum area of a sector with circumference 40 is 100 -/
theorem sector_max_area (C : ℝ) (h : C = 40) : 
  ∃ (A : ℝ), A = 100 ∧ ∀ (r θ : ℝ), r > 0 → θ > 0 → r * θ + 2 * r = C → 
    (1/2 : ℝ) * r^2 * θ ≤ A := by sorry

end sector_max_area_l849_84904


namespace product_xyz_l849_84911

theorem product_xyz (x y z : ℕ+) 
  (h1 : x + 2*y = z) 
  (h2 : x^2 - 4*y^2 + z^2 = 310) : 
  x*y*z = 4030 ∨ x*y*z = 23870 := by
sorry

end product_xyz_l849_84911


namespace fruit_basket_count_l849_84926

theorem fruit_basket_count : 
  let apples_per_basket : ℕ := 9
  let oranges_per_basket : ℕ := 15
  let bananas_per_basket : ℕ := 14
  let num_baskets : ℕ := 4
  let fruits_per_basket : ℕ := apples_per_basket + oranges_per_basket + bananas_per_basket
  let fruits_in_three_baskets : ℕ := 3 * fruits_per_basket
  let fruits_in_fourth_basket : ℕ := fruits_per_basket - 6
  fruits_in_three_baskets + fruits_in_fourth_basket = 70
  := by sorry

end fruit_basket_count_l849_84926


namespace r_value_when_n_is_3_l849_84950

theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 3^n + 2) 
  (h2 : r = 4^s - s) 
  (h3 : n = 3) : 
  r = 4^29 - 29 := by
sorry

end r_value_when_n_is_3_l849_84950


namespace arithmetic_sequence_sum_l849_84981

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + 3 * a 6 + a 11 = 10) →
  a 5 + a 7 = 4 := by
  sorry

end arithmetic_sequence_sum_l849_84981


namespace possible_m_values_l849_84983

theorem possible_m_values (M N : Set ℝ) (m : ℝ) :
  M = {x : ℝ | 2 * x^2 - 5 * x - 3 = 0} →
  N = {x : ℝ | m * x = 1} →
  N ⊆ M →
  {m | ∃ (x : ℝ), x ∈ N} = {-2, 1/3} :=
by sorry

end possible_m_values_l849_84983


namespace coefficient_x5y4_in_expansion_x_plus_y_9_l849_84992

theorem coefficient_x5y4_in_expansion_x_plus_y_9 :
  (Finset.range 10).sum (λ k => Nat.choose 9 k * X^k * Y^(9 - k)) =
  126 * X^5 * Y^4 + (Finset.range 10).sum (λ k => if k ≠ 5 then Nat.choose 9 k * X^k * Y^(9 - k) else 0) :=
by sorry

end coefficient_x5y4_in_expansion_x_plus_y_9_l849_84992


namespace fibonacci_arithmetic_sequence_l849_84930

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Main theorem -/
theorem fibonacci_arithmetic_sequence (a b d : ℕ) : 
  (∀ n ≥ 3, fib n = fib (n - 1) + fib (n - 2)) →  -- Fibonacci recurrence relation
  (fib a < fib b ∧ fib b < fib d) →  -- Increasing sequence
  (fib d - fib b = fib b - fib a) →  -- Arithmetic sequence
  d = b + 2 →  -- Given condition
  a + b + d = 1000 →  -- Given condition
  a = 332 := by
sorry

end fibonacci_arithmetic_sequence_l849_84930


namespace dan_picked_nine_apples_l849_84956

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The total number of apples picked by Benny and Dan -/
def total_apples : ℕ := 11

/-- The number of apples Dan picked -/
def dan_apples : ℕ := total_apples - benny_apples

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end dan_picked_nine_apples_l849_84956


namespace sara_movie_rental_l849_84960

def movie_problem (theater_ticket_price : ℚ) (num_tickets : ℕ) (bought_movie_price : ℚ) (total_spent : ℚ) : Prop :=
  let theater_total : ℚ := theater_ticket_price * num_tickets
  let rental_price : ℚ := total_spent - theater_total - bought_movie_price
  rental_price = 159/100

theorem sara_movie_rental :
  movie_problem (1062/100) 2 (1395/100) (3678/100) :=
by
  sorry

end sara_movie_rental_l849_84960


namespace subtraction_problem_l849_84931

theorem subtraction_problem : 943 - 87 = 856 := by
  sorry

end subtraction_problem_l849_84931


namespace ball_count_difference_l849_84986

theorem ball_count_difference (total : ℕ) (white : ℕ) : 
  total = 100 →
  white = 16 →
  ∃ (blue red : ℕ),
    blue > white ∧
    red = 2 * blue ∧
    red + blue + white = total ∧
    blue - white = 12 := by
  sorry

end ball_count_difference_l849_84986


namespace survivor_quitters_probability_l849_84902

def total_participants : ℕ := 18
def tribe1_size : ℕ := 10
def tribe2_size : ℕ := 8
def quitters : ℕ := 3

theorem survivor_quitters_probability :
  let total_ways := Nat.choose total_participants quitters
  let ways_from_tribe1 := Nat.choose tribe1_size quitters
  let ways_from_tribe2 := Nat.choose tribe2_size quitters
  (ways_from_tribe1 + ways_from_tribe2 : ℚ) / total_ways = 11 / 51 := by
sorry

end survivor_quitters_probability_l849_84902


namespace arithmetic_combination_exists_l849_84947

theorem arithmetic_combination_exists : ∃ (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ) (h : ℕ → ℕ → ℕ),
  (f 1 (g 2 3)) * (h 4 5) = 100 :=
by sorry

end arithmetic_combination_exists_l849_84947


namespace cone_central_angle_l849_84925

/-- Given a cone where the lateral area is twice the area of its base,
    prove that the central angle of the sector of the unfolded side is 180 degrees. -/
theorem cone_central_angle (r R : ℝ) (h : r > 0) (H : R > 0) : 
  π * r * R = 2 * π * r^2 → (180 : ℝ) * (2 * π * r) / (π * R) = 180 :=
by sorry

end cone_central_angle_l849_84925


namespace tile_problem_l849_84903

theorem tile_problem (total_tiles : ℕ) (total_edges : ℕ) (triangular_tiles : ℕ) (square_tiles : ℕ) : 
  total_tiles = 25 →
  total_edges = 84 →
  total_tiles = triangular_tiles + square_tiles →
  total_edges = 3 * triangular_tiles + 4 * square_tiles →
  square_tiles = 9 := by
sorry

end tile_problem_l849_84903


namespace special_trapezoid_base_ratio_l849_84913

/-- A trapezoid with a 60° angle that has both inscribed and circumscribed circles -/
structure SpecialTrapezoid where
  /-- The measure of one angle of the trapezoid in degrees -/
  angle : ℝ
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Prop
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Prop
  /-- The angle measure is 60° -/
  angle_is_60 : angle = 60

/-- The ratio of the bases of the special trapezoid -/
def base_ratio (t : SpecialTrapezoid) : ℝ × ℝ :=
  (1, 3)

/-- Theorem: The ratio of the bases of a special trapezoid is 1:3 -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  base_ratio t = (1, 3) := by
  sorry

end special_trapezoid_base_ratio_l849_84913


namespace show_revenue_l849_84936

/-- Calculate the total revenue from two shows given the number of attendees and ticket price -/
theorem show_revenue (first_show_attendees : ℕ) (ticket_price : ℕ) : 
  first_show_attendees * ticket_price + (3 * first_show_attendees) * ticket_price = 20000 :=
by
  sorry

#check show_revenue 200 25

end show_revenue_l849_84936


namespace b_investment_is_1000_l849_84971

/-- Represents the business partnership between a and b -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  total_profit : ℕ
  management_fee_percent : ℚ
  a_total_received : ℕ

/-- Calculates b's investment given the partnership details -/
def calculate_b_investment (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that b's investment is 1000 given the problem conditions -/
theorem b_investment_is_1000 (p : Partnership) 
  (h1 : p.a_investment = 2000)
  (h2 : p.total_profit = 9600)
  (h3 : p.management_fee_percent = 1/10)
  (h4 : p.a_total_received = 4416) :
  calculate_b_investment p = 1000 := by
  sorry

end b_investment_is_1000_l849_84971


namespace preimage_of_one_l849_84998

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem preimage_of_one (x : ℝ) : f x = 1 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

end preimage_of_one_l849_84998


namespace extreme_points_imply_a_range_l849_84937

open Real

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 + a * log x

-- State the theorem
theorem extreme_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧
    (∀ x : ℝ, 0 < x → (deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂))) →
  0 < a ∧ a < 1/2 :=
sorry

end extreme_points_imply_a_range_l849_84937


namespace inequality_equivalence_l849_84978

theorem inequality_equivalence (x : ℝ) : 
  1 / (x - 2) < 4 ↔ x < 2 ∨ x > 9/4 := by
  sorry

end inequality_equivalence_l849_84978


namespace tan_sum_product_22_23_degrees_l849_84938

theorem tan_sum_product_22_23_degrees :
  Real.tan (22 * π / 180) + Real.tan (23 * π / 180) + Real.tan (22 * π / 180) * Real.tan (23 * π / 180) = 1 :=
by sorry

end tan_sum_product_22_23_degrees_l849_84938


namespace upgraded_fraction_is_one_ninth_l849_84966

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem stating the fraction of upgraded sensors on a specific satellite configuration -/
theorem upgraded_fraction_is_one_ninth (s : Satellite) 
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.total_upgraded / 3) :
  upgraded_fraction s = 1 / 9 := by
  sorry


end upgraded_fraction_is_one_ninth_l849_84966


namespace train_speed_l849_84952

/-- The speed of a train passing through a tunnel -/
theorem train_speed (train_length : ℝ) (tunnel_length : ℝ) (pass_time : ℝ) :
  train_length = 100 →
  tunnel_length = 1.7 →
  pass_time = 1.5 / 60 →
  (train_length / 1000 + tunnel_length) / pass_time = 72 := by
  sorry

end train_speed_l849_84952


namespace least_seven_digit_binary_proof_l849_84928

/-- The least positive base-10 number that requires seven digits in binary representation -/
def least_seven_digit_binary : ℕ := 64

/-- A function that returns the number of digits in the binary representation of a natural number -/
def binary_digits (n : ℕ) : ℕ := sorry

theorem least_seven_digit_binary_proof :
  (binary_digits least_seven_digit_binary = 7) ∧
  (∀ m : ℕ, m > 0 ∧ m < least_seven_digit_binary → binary_digits m < 7) :=
sorry

end least_seven_digit_binary_proof_l849_84928


namespace toby_friend_wins_l849_84975

/-- Juggling contest between Toby and his friend -/
theorem toby_friend_wins (toby_rotations : ℕ → ℕ) (friend_apples : ℕ) (friend_rotations : ℕ → ℕ) : 
  friend_apples = 4 ∧ 
  (∀ n, friend_rotations n = 101) ∧ 
  (∀ n, toby_rotations n = 80) → 
  friend_apples * friend_rotations 0 = 404 ∧ 
  ∀ k, k * toby_rotations 0 ≤ friend_apples * friend_rotations 0 :=
by sorry

end toby_friend_wins_l849_84975


namespace honzik_payment_l849_84984

theorem honzik_payment (lollipop_price ice_cream_price : ℕ) : 
  (3 * lollipop_price = 24) →
  (∃ n : ℕ, 2 ≤ n ∧ n ≤ 9 ∧ 4 * lollipop_price + n * ice_cream_price = 109) →
  lollipop_price + ice_cream_price = 19 :=
by sorry

end honzik_payment_l849_84984
