import Mathlib

namespace hyperbola_asymptotes_l872_87251

/-- Given a hyperbola E with the standard equation x²/4 - y² = 1,
    prove that the equations of its asymptotes are y = ± (1/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 4 - y^2 = 1) →
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) :=
by sorry

end hyperbola_asymptotes_l872_87251


namespace square_value_l872_87261

theorem square_value (p : ℝ) (h1 : p + p = 75) (h2 : (p + p) + 2*p = 149) : p = 38 := by
  sorry

end square_value_l872_87261


namespace toothpicks_300th_stage_l872_87272

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 5 + (n - 1) * 4

/-- Theorem: The number of toothpicks in the 300th stage is 1201 -/
theorem toothpicks_300th_stage :
  toothpicks 300 = 1201 := by
  sorry

end toothpicks_300th_stage_l872_87272


namespace two_digit_number_patterns_l872_87205

theorem two_digit_number_patterns 
  (a m n : ℕ) 
  (ha : 0 < a ∧ a < 10) 
  (hm : 0 < m ∧ m < 10) 
  (hn : 0 < n ∧ n < 10) : 
  ((10 * a + 5) ^ 2 = 100 * a * (a + 1) + 25) ∧ 
  ((10 * m + n) * (10 * m + (10 - n)) = 100 * m * (m + 1) + n * (10 - n)) :=
by sorry

end two_digit_number_patterns_l872_87205


namespace feb_1_2015_was_sunday_l872_87230

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Theorem: If January 1, 2015 was a Thursday, then February 1, 2015 was a Sunday -/
theorem feb_1_2015_was_sunday :
  advanceDays DayOfWeek.Thursday 31 = DayOfWeek.Sunday := by
  sorry

end feb_1_2015_was_sunday_l872_87230


namespace hyperbola_equation_theorem_l872_87286

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  eccentricity : ℝ
  real_axis_length : ℝ

/-- The equation of a hyperbola with given properties -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 12 = 1) ∨ (y^2 / 4 - x^2 / 12 = 1)

/-- Theorem stating that a hyperbola with the given properties has one of the two specified equations -/
theorem hyperbola_equation_theorem (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_eccentricity : h.eccentricity = 2)
    (h_real_axis : h.real_axis_length = 4) :
    ∀ x y : ℝ, hyperbola_equation h x y := by
  sorry


end hyperbola_equation_theorem_l872_87286


namespace lindsey_squat_weight_l872_87263

/-- Calculates the total weight Lindsey will be squatting -/
def total_squat_weight (band_a : ℕ) (band_b : ℕ) (band_c : ℕ) 
                       (leg_weight : ℕ) (dumbbell : ℕ) : ℕ :=
  2 * (band_a + band_b + band_c) + 2 * leg_weight + dumbbell

/-- Proves that Lindsey's total squat weight is 65 pounds -/
theorem lindsey_squat_weight :
  total_squat_weight 7 5 3 10 15 = 65 :=
by sorry

end lindsey_squat_weight_l872_87263


namespace patrons_in_cars_patrons_in_cars_is_twelve_l872_87262

/-- The number of patrons who came in cars to a golf tournament -/
theorem patrons_in_cars (num_carts : ℕ) (cart_capacity : ℕ) (bus_patrons : ℕ) : ℕ :=
  num_carts * cart_capacity - bus_patrons

/-- Proof that the number of patrons who came in cars is 12 -/
theorem patrons_in_cars_is_twelve : patrons_in_cars 13 3 27 = 12 := by
  sorry

end patrons_in_cars_patrons_in_cars_is_twelve_l872_87262


namespace square_minus_four_times_plus_four_equals_six_l872_87209

theorem square_minus_four_times_plus_four_equals_six (a : ℝ) :
  a = Real.sqrt 6 + 2 → a^2 - 4*a + 4 = 6 := by sorry

end square_minus_four_times_plus_four_equals_six_l872_87209


namespace complex_unit_vector_l872_87279

theorem complex_unit_vector (z : ℂ) (h : z = 3 + 4*I) : z / Complex.abs z = 3/5 + 4/5*I := by
  sorry

end complex_unit_vector_l872_87279


namespace contrapositive_equivalence_l872_87208

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - x - 2 = 0 → x = 2 ∨ x = -1) ↔
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -1 → x^2 - x - 2 ≠ 0) :=
by sorry

end contrapositive_equivalence_l872_87208


namespace shaded_area_is_one_third_l872_87293

/-- Two rectangles with dimensions 10 × 20 overlap to form a 20 × 30 rectangle. -/
structure OverlappingRectangles where
  small_width : ℝ
  small_height : ℝ
  large_width : ℝ
  large_height : ℝ
  small_width_eq : small_width = 10
  small_height_eq : small_height = 20
  large_width_eq : large_width = 20
  large_height_eq : large_height = 30

/-- The shaded area is the overlap of the two smaller rectangles. -/
def shaded_area (r : OverlappingRectangles) : ℝ :=
  r.small_width * r.small_height

/-- The area of the larger rectangle. -/
def large_area (r : OverlappingRectangles) : ℝ :=
  r.large_width * r.large_height

/-- The theorem stating that the shaded area is 1/3 of the larger rectangle's area. -/
theorem shaded_area_is_one_third (r : OverlappingRectangles) :
    shaded_area r / large_area r = 1 / 3 := by
  sorry

end shaded_area_is_one_third_l872_87293


namespace line_transformation_l872_87207

-- Define the original line
def original_line (x : ℝ) : ℝ := x - 2

-- Define the transformation (moving 3 units upwards)
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x => f x + 3

-- Define the new line
def new_line (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- Theorem statement
theorem line_transformation :
  ∃ k b : ℝ, 
    (∀ x : ℝ, transform original_line x = new_line k b x) ∧ 
    k = 1 ∧ 
    b = 1 ∧ 
    (∀ x : ℝ, new_line k b x > 0 → x > -1) :=
by sorry

end line_transformation_l872_87207


namespace astronaut_stay_duration_l872_87218

theorem astronaut_stay_duration (days_per_year : ℕ) (seasons_per_year : ℕ) (seasons_stayed : ℕ) : 
  days_per_year = 250 → 
  seasons_per_year = 5 → 
  seasons_stayed = 3 → 
  (days_per_year / seasons_per_year) * seasons_stayed = 150 :=
by sorry

end astronaut_stay_duration_l872_87218


namespace hydropolis_aquaville_rainfall_difference_l872_87214

/-- The difference in total rainfall between two cities over a year, given their average monthly rainfalls and the number of months. -/
def rainfall_difference (avg_rainfall_city1 avg_rainfall_city2 : ℝ) (months : ℕ) : ℝ :=
  (avg_rainfall_city1 - avg_rainfall_city2) * months

/-- Theorem stating the difference in total rainfall between Hydropolis and Aquaville in 2011 -/
theorem hydropolis_aquaville_rainfall_difference :
  let hydropolis_2010 : ℝ := 36.5
  let rainfall_increase : ℝ := 3.5
  let hydropolis_2011 : ℝ := hydropolis_2010 + rainfall_increase
  let aquaville_2011 : ℝ := hydropolis_2011 - 1.5
  let months : ℕ := 12
  rainfall_difference hydropolis_2011 aquaville_2011 months = 18.0 := by
  sorry

#eval rainfall_difference 40.0 38.5 12

end hydropolis_aquaville_rainfall_difference_l872_87214


namespace magic_sum_divisible_by_three_l872_87297

/-- Represents a 3x3 magic square with integer entries -/
def MagicSquare : Type := Matrix (Fin 3) (Fin 3) ℤ

/-- The sum of each row, column, and diagonal in a magic square -/
def magicSum (m : MagicSquare) : ℤ :=
  m 0 0 + m 0 1 + m 0 2

/-- Predicate to check if a given matrix is a magic square -/
def isMagicSquare (m : MagicSquare) : Prop :=
  (∀ i : Fin 3, (m i 0 + m i 1 + m i 2 = magicSum m)) ∧ 
  (∀ j : Fin 3, (m 0 j + m 1 j + m 2 j = magicSum m)) ∧ 
  (m 0 0 + m 1 1 + m 2 2 = magicSum m) ∧
  (m 0 2 + m 1 1 + m 2 0 = magicSum m)

/-- Theorem: The magic sum of a 3x3 magic square is divisible by 3 -/
theorem magic_sum_divisible_by_three (m : MagicSquare) (h : isMagicSquare m) :
  ∃ k : ℤ, magicSum m = 3 * k := by
  sorry

end magic_sum_divisible_by_three_l872_87297


namespace lollipop_distribution_l872_87259

theorem lollipop_distribution (raspberry mint orange cotton_candy : ℕ) 
  (h1 : raspberry = 60) 
  (h2 : mint = 135) 
  (h3 : orange = 5) 
  (h4 : cotton_candy = 330) 
  (friends : ℕ) 
  (h5 : friends = 15) : 
  (raspberry + mint + orange + cotton_candy) % friends = 5 := by
sorry

end lollipop_distribution_l872_87259


namespace total_weekly_airflow_l872_87250

/-- Calculates the total airflow generated by three fans in one week -/
theorem total_weekly_airflow (fan_a_flow : ℝ) (fan_a_time : ℝ) 
                              (fan_b_flow : ℝ) (fan_b_time : ℝ) 
                              (fan_c_flow : ℝ) (fan_c_time : ℝ) : 
  fan_a_flow = 10 →
  fan_a_time = 10 →
  fan_b_flow = 15 →
  fan_b_time = 20 →
  fan_c_flow = 25 →
  fan_c_time = 30 →
  ((fan_a_flow * fan_a_time * 60) + 
   (fan_b_flow * fan_b_time * 60) + 
   (fan_c_flow * fan_c_time * 60)) * 7 = 483000 := by
  sorry

#check total_weekly_airflow

end total_weekly_airflow_l872_87250


namespace train_crossing_time_l872_87270

/-- The time taken for a train to cross a pole -/
theorem train_crossing_time (train_speed_kmh : ℝ) (train_length_m : ℝ) : 
  train_speed_kmh = 72 → train_length_m = 180 → 
  (train_length_m / (train_speed_kmh * 1000 / 3600)) = 9 := by
  sorry

end train_crossing_time_l872_87270


namespace sequence_sum_l872_87219

theorem sequence_sum (a : ℕ → ℤ) : 
  (∀ n : ℕ, a (n + 1) - a n = 2) → 
  a 1 = -5 → 
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| : ℤ) = 18 := by
sorry

end sequence_sum_l872_87219


namespace complex_fraction_sum_l872_87265

theorem complex_fraction_sum : 
  (481 + 1/6 : ℚ) + (265 + 1/12 : ℚ) + (904 + 1/20 : ℚ) - 
  (184 + 29/30 : ℚ) - (160 + 41/42 : ℚ) - (703 + 55/56 : ℚ) = 
  603 + 3/8 := by sorry

end complex_fraction_sum_l872_87265


namespace problem_1_l872_87285

theorem problem_1 (x y : ℝ) : (-3 * x^2 * y)^2 * (2 * x * y^2) / (-6 * x^3 * y^4) = -3 * x^2 := by
  sorry

end problem_1_l872_87285


namespace square_sum_fraction_difference_l872_87213

theorem square_sum_fraction_difference : 
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 := by
  sorry

end square_sum_fraction_difference_l872_87213


namespace specialIntegers_infinite_l872_87247

/-- A function that converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- A predicate that checks if a list of digits contains only 1 and 2 -/
def containsOnly1And2 (digits : List ℕ) : Prop :=
  sorry

/-- The set of positive integers n such that n^2 in base 4 contains only digits 1 and 2 -/
def specialIntegers : Set ℕ :=
  {n : ℕ | n > 0 ∧ containsOnly1And2 (toBase4 (n^2))}

/-- The main theorem stating that the set of special integers is infinite -/
theorem specialIntegers_infinite : Set.Infinite specialIntegers :=
  sorry

end specialIntegers_infinite_l872_87247


namespace quadratic_real_root_condition_l872_87253

/-- A quadratic equation x^2 + bx + 16 has at least one real root if and only if b ∈ (-∞,-8] ∪ [8,∞) -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 16 = 0) ↔ b ≤ -8 ∨ b ≥ 8 := by
  sorry

end quadratic_real_root_condition_l872_87253


namespace theater_seats_l872_87287

theorem theater_seats : ∃ n : ℕ, n < 60 ∧ n % 9 = 5 ∧ n % 6 = 3 ∧ n = 41 := by
  sorry

end theater_seats_l872_87287


namespace cube_difference_identity_l872_87226

theorem cube_difference_identity (a b : ℝ) : 
  (a^3 + b^3 = (a + b) * (a^2 - a*b + b^2)) → 
  (a^3 - b^3 = (a - b) * (a^2 + a*b + b^2)) := by
sorry

end cube_difference_identity_l872_87226


namespace average_speed_calculation_l872_87235

def total_distance : ℝ := 120
def total_time : ℝ := 7

theorem average_speed_calculation : 
  (total_distance / total_time) = 120 / 7 := by
  sorry

end average_speed_calculation_l872_87235


namespace fifteen_cells_covered_by_two_l872_87292

/-- Represents a square on a graph paper --/
structure Square :=
  (side : ℕ)

/-- Represents the configuration of squares on the graph paper --/
structure SquareConfiguration :=
  (squares : List Square)
  (total_area : ℕ)
  (unique_area : ℕ)
  (triple_overlap : ℕ)

/-- Calculates the number of cells covered by exactly two squares --/
def cells_covered_by_two (config : SquareConfiguration) : ℕ :=
  config.total_area - config.unique_area - 2 * config.triple_overlap

/-- Theorem stating that for the given configuration, 15 cells are covered by exactly two squares --/
theorem fifteen_cells_covered_by_two (config : SquareConfiguration) 
  (h1 : config.squares.length = 3)
  (h2 : ∀ s ∈ config.squares, s.side = 5)
  (h3 : config.total_area = 75)
  (h4 : config.unique_area = 56)
  (h5 : config.triple_overlap = 2) :
  cells_covered_by_two config = 15 := by
  sorry

end fifteen_cells_covered_by_two_l872_87292


namespace jane_morning_reading_l872_87238

/-- The number of pages Jane reads in the morning -/
def morning_pages : ℕ := 5

/-- The number of pages Jane reads in the evening -/
def evening_pages : ℕ := 10

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pages Jane reads in a week -/
def total_pages : ℕ := 105

/-- Theorem stating that Jane reads 5 pages in the morning -/
theorem jane_morning_reading :
  morning_pages = 5 ∧
  evening_pages = 10 ∧
  days_in_week = 7 ∧
  total_pages = 105 ∧
  days_in_week * (morning_pages + evening_pages) = total_pages :=
by sorry

end jane_morning_reading_l872_87238


namespace quadratic_form_j_value_l872_87274

theorem quadratic_form_j_value 
  (a b c : ℝ) 
  (h1 : ∀ x, a * x^2 + b * x + c = 5 * (x - 3)^2 + 15) 
  (m n j : ℝ) 
  (h2 : ∀ x, 4 * (a * x^2 + b * x + c) = m * (x - j)^2 + n) : 
  j = 3 := by
sorry

end quadratic_form_j_value_l872_87274


namespace min_value_expression_l872_87200

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 * a / (b * c^2 + b)) + (1 / (a * b * c^2 + a * b)) + 3 * c^2 ≥ 6 * Real.sqrt 2 - 3 := by
  sorry

end min_value_expression_l872_87200


namespace farm_milk_production_l872_87257

theorem farm_milk_production
  (a b c d e : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (he : e > 0)
  (group_a_production : b = (a * c) * (b / (a * c)))
  (group_b_efficiency : ℝ)
  (hefficiency : group_b_efficiency = 1.2)
  : (group_b_efficiency * b * d * e) / (a * c) = (1.2 * b * d * e) / (a * c) :=
by sorry

end farm_milk_production_l872_87257


namespace fish_in_third_tank_l872_87256

/-- The number of fish in the first tank -/
def first_tank : ℕ := 7 + 8

/-- The number of fish in the second tank -/
def second_tank : ℕ := 2 * first_tank

/-- The number of fish in the third tank -/
def third_tank : ℕ := second_tank / 3

theorem fish_in_third_tank : third_tank = 10 := by
  sorry

end fish_in_third_tank_l872_87256


namespace hexagon_diagonal_intersection_probability_l872_87276

/-- A convex hexagon -/
structure ConvexHexagon where
  -- Add any necessary properties of a convex hexagon

/-- A diagonal of a convex hexagon -/
structure Diagonal (H : ConvexHexagon) where
  -- Add any necessary properties of a diagonal

/-- Two diagonals intersect inside the hexagon (not at a vertex) -/
def intersect_inside (H : ConvexHexagon) (d1 d2 : Diagonal H) : Prop :=
  sorry

/-- The probability of two randomly chosen diagonals intersecting inside the hexagon -/
def intersection_probability (H : ConvexHexagon) : ℚ :=
  sorry

/-- Theorem: The probability of two randomly chosen diagonals intersecting inside a convex hexagon is 5/12 -/
theorem hexagon_diagonal_intersection_probability (H : ConvexHexagon) :
  intersection_probability H = 5 / 12 :=
sorry

end hexagon_diagonal_intersection_probability_l872_87276


namespace fahrenheit_to_celsius_l872_87223

theorem fahrenheit_to_celsius (C F : ℝ) : C = (5/9) * (F - 32) → C = 40 → F = 104 := by
  sorry

end fahrenheit_to_celsius_l872_87223


namespace closest_multiple_of_15_to_1987_l872_87227

theorem closest_multiple_of_15_to_1987 :
  ∃ (n : ℤ), n * 15 = 1980 ∧
  ∀ (m : ℤ), m * 15 ≠ 1980 → |1987 - (m * 15)| ≥ |1987 - 1980| := by
  sorry

end closest_multiple_of_15_to_1987_l872_87227


namespace events_mutually_exclusive_not_complementary_l872_87269

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "A receives a club"
def A_receives_club (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B receives a club"
def B_receives_club (d : Distribution) : Prop := d Person.B = Card.Clubs

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (∀ d : Distribution, ¬(A_receives_club d ∧ B_receives_club d)) ∧
  (∃ d : Distribution, ¬(A_receives_club d ∨ B_receives_club d)) :=
sorry

end events_mutually_exclusive_not_complementary_l872_87269


namespace division_of_decimals_l872_87228

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end division_of_decimals_l872_87228


namespace sum_of_divisors_930_l872_87281

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The theorem to be proved -/
theorem sum_of_divisors_930 (i j : ℕ+) :
  sum_of_divisors (2^i.val * 5^j.val) = 930 → i.val + j.val = 5 := by
  sorry

end sum_of_divisors_930_l872_87281


namespace ian_lottery_payment_l872_87217

theorem ian_lottery_payment (total : ℝ) (left : ℝ) (colin : ℝ) (helen : ℝ) (benedict : ℝ) :
  total = 100 →
  helen = 2 * colin →
  benedict = helen / 2 →
  left = 20 →
  total = colin + helen + benedict + left →
  colin = 20 := by
sorry

end ian_lottery_payment_l872_87217


namespace complex_fraction_equality_l872_87288

theorem complex_fraction_equality : (1 + Complex.I * Real.sqrt 3) ^ 2 / (Complex.I * Real.sqrt 3 - 1) = -2 - 2 * Complex.I := by
  sorry

end complex_fraction_equality_l872_87288


namespace remainder_theorem_l872_87240

/-- The dividend polynomial -/
def f (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + 5*x - 8

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 84*x - 84

/-- Theorem stating that r is the remainder when f is divided by g -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x :=
sorry

end remainder_theorem_l872_87240


namespace orange_count_l872_87220

/-- The number of oranges initially in the bin -/
def initial_oranges : ℕ := sorry

/-- The number of oranges thrown away -/
def thrown_away : ℕ := 20

/-- The number of new oranges added -/
def new_oranges : ℕ := 13

/-- The final number of oranges in the bin -/
def final_oranges : ℕ := 27

theorem orange_count : initial_oranges = 34 :=
  by sorry

end orange_count_l872_87220


namespace special_polygon_has_eight_sides_l872_87273

/-- A polygon with n sides where the sum of interior angles is 3 times the sum of exterior angles -/
structure SpecialPolygon where
  n : ℕ
  interior_sum : ℝ
  exterior_sum : ℝ
  h1 : interior_sum = (n - 2) * 180
  h2 : exterior_sum = 360
  h3 : interior_sum = 3 * exterior_sum

/-- Theorem: A SpecialPolygon has 8 sides -/
theorem special_polygon_has_eight_sides (p : SpecialPolygon) : p.n = 8 := by
  sorry

end special_polygon_has_eight_sides_l872_87273


namespace pete_walked_7430_miles_l872_87239

/-- Represents a pedometer with a maximum value before flipping --/
structure Pedometer where
  max_value : ℕ
  flip_count : ℕ
  final_reading : ℕ

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_value * p.flip_count + p.final_reading

/-- Represents Pete's walking data for the year --/
structure WalkingData where
  pedometer1 : Pedometer
  pedometer2 : Pedometer
  steps_per_mile : ℕ

/-- Theorem stating that Pete walked 7430 miles during the year --/
theorem pete_walked_7430_miles (data : WalkingData)
  (h1 : data.pedometer1.max_value = 100000)
  (h2 : data.pedometer1.flip_count = 50)
  (h3 : data.pedometer1.final_reading = 25000)
  (h4 : data.pedometer2.max_value = 400000)
  (h5 : data.pedometer2.flip_count = 15)
  (h6 : data.pedometer2.final_reading = 120000)
  (h7 : data.steps_per_mile = 1500) :
  (total_steps data.pedometer1 + total_steps data.pedometer2) / data.steps_per_mile = 7430 := by
  sorry

end pete_walked_7430_miles_l872_87239


namespace orange_cost_calculation_l872_87245

def initial_amount : ℕ := 95
def apple_cost : ℕ := 25
def candy_cost : ℕ := 6
def amount_left : ℕ := 50

theorem orange_cost_calculation : 
  initial_amount - amount_left - apple_cost - candy_cost = 14 := by
  sorry

end orange_cost_calculation_l872_87245


namespace rectangular_solid_edge_sum_l872_87249

/-- A rectangular solid with dimensions in arithmetic progression -/
structure RectangularSolid where
  a : ℝ  -- middle dimension
  d : ℝ  -- common difference

/-- Volume of the rectangular solid -/
def volume (solid : RectangularSolid) : ℝ :=
  (solid.a - solid.d) * solid.a * (solid.a + solid.d)

/-- Surface area of the rectangular solid -/
def surface_area (solid : RectangularSolid) : ℝ :=
  2 * ((solid.a - solid.d) * solid.a + solid.a * (solid.a + solid.d) + (solid.a - solid.d) * (solid.a + solid.d))

/-- Sum of the lengths of all edges of the rectangular solid -/
def sum_of_edges (solid : RectangularSolid) : ℝ :=
  4 * ((solid.a - solid.d) + solid.a + (solid.a + solid.d))

theorem rectangular_solid_edge_sum
  (solid : RectangularSolid)
  (h_volume : volume solid = 512)
  (h_area : surface_area solid = 352) :
  sum_of_edges solid = 12 * Real.sqrt 59 := by
  sorry

end rectangular_solid_edge_sum_l872_87249


namespace age_problem_l872_87275

/-- Given a group of 7 people, if adding a person of age x increases the average age by 2,
    and adding a person aged 15 decreases the average age by 1, then x = 39. -/
theorem age_problem (T : ℝ) (A : ℝ) (x : ℝ) : 
  T = 7 * A →
  T + x = 8 * (A + 2) →
  T + 15 = 8 * (A - 1) →
  x = 39 := by
  sorry

end age_problem_l872_87275


namespace complex_fraction_simplification_l872_87271

theorem complex_fraction_simplification :
  let z : ℂ := Complex.mk 3 8 / Complex.mk 1 (-4)
  (z.re = -29/17) ∧ (z.im = 20/17) := by
  sorry

end complex_fraction_simplification_l872_87271


namespace rectangle_length_l872_87294

/-- Given a rectangle with width 16 cm and perimeter 70 cm, prove its length is 19 cm. -/
theorem rectangle_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 16 → 
  perimeter = 70 → 
  perimeter = 2 * (length + width) → 
  length = 19 := by
sorry

end rectangle_length_l872_87294


namespace m_range_l872_87291

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) ↔ (m ≥ 4 ∨ m ≤ 0) :=
by sorry

end m_range_l872_87291


namespace sufficient_but_not_necessary_l872_87246

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point1 : Point3D
  point2 : Point3D

-- Define the property of non-coplanar points
def nonCoplanar (E F G H : Point3D) : Prop := sorry

-- Define the property of non-intersecting lines
def nonIntersecting (l1 l2 : Line3D) : Prop := sorry

theorem sufficient_but_not_necessary 
  (E F G H : Point3D) 
  (EF : Line3D) 
  (GH : Line3D) 
  (h_EF : EF.point1 = E ∧ EF.point2 = F) 
  (h_GH : GH.point1 = G ∧ GH.point2 = H) :
  (nonCoplanar E F G H → nonIntersecting EF GH) ∧ 
  ∃ E' F' G' H' : Point3D, ∃ EF' GH' : Line3D, 
    (EF'.point1 = E' ∧ EF'.point2 = F') ∧ 
    (GH'.point1 = G' ∧ GH'.point2 = H') ∧ 
    nonIntersecting EF' GH' ∧ 
    ¬(nonCoplanar E' F' G' H') := by
  sorry

end sufficient_but_not_necessary_l872_87246


namespace smallest_solution_of_equation_l872_87268

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 ∧ (∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y) →
  x = -4 * Real.sqrt 2 :=
by sorry

end smallest_solution_of_equation_l872_87268


namespace opposite_sides_of_line_l872_87296

def line_equation (x y : ℝ) : ℝ := 2 * x + y - 3

theorem opposite_sides_of_line :
  (line_equation 0 0 < 0) ∧ (line_equation 2 3 > 0) :=
by sorry

end opposite_sides_of_line_l872_87296


namespace second_number_is_fifteen_l872_87244

def has_exactly_three_common_factors_with_15 (n : ℕ) : Prop :=
  ∃ (f₁ f₂ f₃ : ℕ), 
    f₁ ≠ f₂ ∧ f₁ ≠ f₃ ∧ f₂ ≠ f₃ ∧
    f₁ > 1 ∧ f₂ > 1 ∧ f₃ > 1 ∧
    f₁ ∣ 15 ∧ f₂ ∣ 15 ∧ f₃ ∣ 15 ∧
    f₁ ∣ n ∧ f₂ ∣ n ∧ f₃ ∣ n ∧
    ∀ (k : ℕ), k > 1 → k ∣ 15 → k ∣ n → (k = f₁ ∨ k = f₂ ∨ k = f₃)

theorem second_number_is_fifteen (n : ℕ) 
  (h : has_exactly_three_common_factors_with_15 n)
  (h3 : 3 ∣ n) (h5 : 5 ∣ n) (h15 : 15 ∣ n) : n = 15 :=
by sorry

end second_number_is_fifteen_l872_87244


namespace no_integer_regular_quadrilateral_pyramid_l872_87252

theorem no_integer_regular_quadrilateral_pyramid :
  ¬ ∃ (g h f s v : ℕ+),
    (f : ℝ)^2 = (h : ℝ)^2 + (g : ℝ)^2 / 2 ∧
    (s : ℝ) = (g : ℝ)^2 + 2 * (g : ℝ) * Real.sqrt ((h : ℝ)^2 + (g : ℝ)^2 / 4) ∧
    (v : ℝ) = (g : ℝ)^2 * (h : ℝ) / 3 :=
by sorry

end no_integer_regular_quadrilateral_pyramid_l872_87252


namespace jordan_born_in_1980_l872_87284

/-- The year when the first AMC 8 was given -/
def first_amc8_year : ℕ := 1985

/-- The age of Jordan when he took the tenth AMC 8 contest -/
def jordan_age_at_tenth_amc8 : ℕ := 14

/-- The number of years between the first AMC 8 and the tenth AMC 8 -/
def years_between_first_and_tenth : ℕ := 9

/-- Jordan's birth year -/
def jordan_birth_year : ℕ := first_amc8_year + years_between_first_and_tenth - jordan_age_at_tenth_amc8

theorem jordan_born_in_1980 : jordan_birth_year = 1980 := by
  sorry

end jordan_born_in_1980_l872_87284


namespace prob_at_least_one_two_l872_87233

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where neither die shows a 2 -/
def neitherShowsTwo : ℕ := (numSides - 1) * (numSides - 1)

/-- The number of outcomes where at least one die shows a 2 -/
def atLeastOneShowsTwo : ℕ := totalOutcomes - neitherShowsTwo

/-- The probability of at least one die showing a 2 when two fair 6-sided dice are rolled -/
theorem prob_at_least_one_two : 
  (atLeastOneShowsTwo : ℚ) / totalOutcomes = 11 / 36 := by
  sorry

end prob_at_least_one_two_l872_87233


namespace reflection_property_l872_87211

/-- A reflection in 2D space -/
structure Reflection2D where
  /-- The function that performs the reflection -/
  reflect : Fin 2 → ℝ → Fin 2 → ℝ

/-- Theorem: If a reflection takes (2, -3) to (8, 1), then it takes (1, 4) to (-18/13, -50/13) -/
theorem reflection_property (r : Reflection2D) 
  (h1 : r.reflect 0 2 = 8) 
  (h2 : r.reflect 1 (-3) = 1) 
  : r.reflect 0 1 = -18/13 ∧ r.reflect 1 4 = -50/13 := by
  sorry


end reflection_property_l872_87211


namespace investment_interest_rate_l872_87255

/-- Proves that given the specified investment conditions, the annual interest rate of the second certificate is 8% -/
theorem investment_interest_rate 
  (initial_investment : ℝ)
  (first_rate : ℝ)
  (second_rate : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_rate = 8)
  (h3 : final_value = 15612)
  (h4 : initial_investment * (1 + first_rate / 400) * (1 + second_rate / 400) = final_value) :
  second_rate = 8 := by
    sorry

#check investment_interest_rate

end investment_interest_rate_l872_87255


namespace lattice_triangle_area_l872_87236

/-- A lattice point is a point with integer coordinates. -/
def LatticePoint (p : ℝ × ℝ) : Prop := ∃ (x y : ℤ), p = (↑x, ↑y)

/-- A triangle with vertices at lattice points. -/
structure LatticeTriangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v1_lattice : LatticePoint v1
  v2_lattice : LatticePoint v2
  v3_lattice : LatticePoint v3

/-- The number of lattice points strictly inside the triangle. -/
def interior_points (t : LatticeTriangle) : ℕ := sorry

/-- The number of lattice points on the sides of the triangle (excluding vertices). -/
def boundary_points (t : LatticeTriangle) : ℕ := sorry

/-- The area of a triangle. -/
def triangle_area (t : LatticeTriangle) : ℝ := sorry

/-- Theorem: The area of a lattice triangle with n interior points and m boundary points
    (excluding vertices) is equal to n + m/2 + 1/2. -/
theorem lattice_triangle_area (t : LatticeTriangle) :
  triangle_area t = interior_points t + (boundary_points t : ℝ) / 2 + 1 / 2 := by sorry

end lattice_triangle_area_l872_87236


namespace intersection_theorem_l872_87243

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 + 2*x < 3}

def B : Set ℝ := {x | (x-2)/x ≤ 0}

theorem intersection_theorem :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x ≤ 0} := by sorry

end intersection_theorem_l872_87243


namespace all_transformed_in_R_l872_87248

def R : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -1 ≤ z.im ∧ z.im ≤ 1}

theorem all_transformed_in_R : ∀ z ∈ R, (1/2 + 1/2*I) * z ∈ R := by
  sorry

end all_transformed_in_R_l872_87248


namespace f_of_g_3_l872_87221

def g (x : ℝ) : ℝ := 4 * x + 5

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_3 : f (g 3) = 91 := by
  sorry

end f_of_g_3_l872_87221


namespace modulus_of_complex_fraction_l872_87204

theorem modulus_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.abs (i / (2 - i)) = Real.sqrt 5 / 5 := by
  sorry

end modulus_of_complex_fraction_l872_87204


namespace road_length_l872_87234

/-- Proves that a road with streetlights installed every 10 meters on both sides, 
    with a total of 120 streetlights, is 590 meters long. -/
theorem road_length (streetlight_interval : Nat) (total_streetlights : Nat) (road_length : Nat) : 
  streetlight_interval = 10 → 
  total_streetlights = 120 → 
  road_length = (total_streetlights / 2 - 1) * streetlight_interval → 
  road_length = 590 :=
by sorry

end road_length_l872_87234


namespace max_value_three_ways_l872_87282

/-- A function representing the number of ways to draw balls with a specific maximum value -/
def num_ways_max_value (n : ℕ) (max_value : ℕ) (num_draws : ℕ) : ℕ :=
  sorry

/-- The number of balls in the box -/
def num_balls : ℕ := 3

/-- The number of draws -/
def num_draws : ℕ := 3

/-- The maximum value we're interested in -/
def target_max : ℕ := 3

theorem max_value_three_ways :
  num_ways_max_value num_balls target_max num_draws = 19 := by
  sorry

end max_value_three_ways_l872_87282


namespace point_coordinates_on_angle_l872_87278

theorem point_coordinates_on_angle (α : Real) (P : Real × Real) :
  α = π / 4 →
  (P.1^2 + P.2^2 = 2) →
  P = (1, 1) := by
sorry

end point_coordinates_on_angle_l872_87278


namespace inverse_proportion_percentage_change_l872_87203

/-- Proves the relationship between inverse proportionality and percentage changes -/
theorem inverse_proportion_percentage_change 
  (x y x' y' q k : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = k) 
  (h4 : x' * y' = k) 
  (h5 : x' = x * (1 - q / 100)) 
  (h6 : q > 0) 
  (h7 : q < 100) : 
  y' = y * (1 + (100 * q) / (100 - q) / 100) := by
  sorry

end inverse_proportion_percentage_change_l872_87203


namespace matrix_inverse_proof_l872_87202

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem matrix_inverse_proof :
  A⁻¹ = !![9/46, -5/46; 2/46, 4/46] := by
  sorry

end matrix_inverse_proof_l872_87202


namespace jug_emptying_l872_87222

theorem jug_emptying (Cx Cy Cz : ℝ) (hx : Cx > 0) (hy : Cy > 0) (hz : Cz > 0) :
  let initial_x := (1/4 : ℝ) * Cx
  let initial_y := (2/3 : ℝ) * Cy
  let initial_z := (3/5 : ℝ) * Cz
  let water_to_fill_y := Cy - initial_y
  let remaining_x := initial_x - water_to_fill_y
  remaining_x ≤ 0 :=
by sorry

end jug_emptying_l872_87222


namespace base_7_23456_equals_6068_l872_87298

def base_7_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_7_23456_equals_6068 :
  base_7_to_10 [6, 5, 4, 3, 2] = 6068 := by
  sorry

end base_7_23456_equals_6068_l872_87298


namespace heartsuit_properties_l872_87299

/-- The heartsuit operation on real numbers -/
def heartsuit (x y : ℝ) : ℝ := 2 * |x - y|

/-- Properties of the heartsuit operation -/
theorem heartsuit_properties :
  (∀ x y : ℝ, heartsuit x y = heartsuit y x) ∧ 
  (∀ x y : ℝ, 3 * (heartsuit x y) = heartsuit (3*x) (3*y)) ∧ 
  (∀ x : ℝ, heartsuit x x = 0) ∧ 
  (∀ x y : ℝ, x ≠ y → heartsuit x y > 0) ∧ 
  (∀ x : ℝ, x ≥ 0 → heartsuit x 0 = 2*x) ∧
  (∃ x : ℝ, heartsuit x 0 ≠ 2*x) :=
by sorry

end heartsuit_properties_l872_87299


namespace pencil_count_l872_87237

theorem pencil_count (num_boxes : ℝ) (pencils_per_box : ℝ) (h1 : num_boxes = 4.0) (h2 : pencils_per_box = 648.0) :
  num_boxes * pencils_per_box = 2592.0 := by
  sorry

end pencil_count_l872_87237


namespace problem_solution_l872_87267

theorem problem_solution :
  let x : ℝ := 88 * (1 + 0.25)
  let y : ℝ := 150 * (1 - 0.40)
  let z : ℝ := 60 * (1 + 0.15)
  (x + y + z = 269) ∧
  ((x * y * z) ^ (x - y) = (683100 : ℝ) ^ 20) := by
  sorry

end problem_solution_l872_87267


namespace john_chocolate_gain_l872_87264

/-- Represents the chocolate types --/
inductive ChocolateType
  | A
  | B
  | C

/-- Represents a purchase of chocolates --/
structure Purchase where
  chocolateType : ChocolateType
  quantity : ℕ
  costPrice : ℚ

/-- Represents a sale of chocolates --/
structure Sale where
  chocolateType : ChocolateType
  quantity : ℕ
  sellingPrice : ℚ

def purchases : List Purchase := [
  ⟨ChocolateType.A, 100, 2⟩,
  ⟨ChocolateType.B, 150, 3⟩,
  ⟨ChocolateType.C, 200, 4⟩
]

def sales : List Sale := [
  ⟨ChocolateType.A, 90, 5/2⟩,
  ⟨ChocolateType.A, 60, 3⟩,
  ⟨ChocolateType.B, 140, 7/2⟩,
  ⟨ChocolateType.B, 10, 4⟩,
  ⟨ChocolateType.B, 50, 5⟩,
  ⟨ChocolateType.C, 180, 9/2⟩,
  ⟨ChocolateType.C, 20, 5⟩
]

def totalCostPrice : ℚ :=
  purchases.foldr (fun p acc => acc + p.quantity * p.costPrice) 0

def totalSellingPrice : ℚ :=
  sales.foldr (fun s acc => acc + s.quantity * s.sellingPrice) 0

def gainPercentage : ℚ :=
  ((totalSellingPrice - totalCostPrice) / totalCostPrice) * 100

theorem john_chocolate_gain :
  gainPercentage = 89/2 := by sorry

end john_chocolate_gain_l872_87264


namespace blue_balls_count_l872_87225

theorem blue_balls_count (total : ℕ) (prob : ℚ) (blue : ℕ) : 
  total = 15 →
  prob = 1 / 21 →
  (blue * (blue - 1)) / (total * (total - 1)) = prob →
  blue = 5 := by
sorry

end blue_balls_count_l872_87225


namespace system_inequality_solution_l872_87229

theorem system_inequality_solution (a : ℝ) : 
  (∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a) → ∀ b : ℝ, ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > b :=
by sorry

end system_inequality_solution_l872_87229


namespace maxwells_speed_l872_87254

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwells_speed (total_distance : ℝ) (brad_speed : ℝ) (maxwell_time : ℝ) (brad_time : ℝ) 
  (h1 : total_distance = 14)
  (h2 : brad_speed = 6)
  (h3 : maxwell_time = 2)
  (h4 : brad_time = 1)
  (h5 : maxwell_time * maxwell_speed + brad_time * brad_speed = total_distance) :
  maxwell_speed = 4 := by
  sorry

#check maxwells_speed

end maxwells_speed_l872_87254


namespace nursing_home_milk_distribution_l872_87241

theorem nursing_home_milk_distribution (elderly : ℕ) (milk : ℕ) : 
  (2 * elderly + 16 = milk) ∧ (4 * elderly = milk + 12) → 
  (elderly = 14 ∧ milk = 44) :=
by sorry

end nursing_home_milk_distribution_l872_87241


namespace domain_of_g_l872_87242

-- Define the domain of f
def DomainF : Set ℝ := Set.Icc (-8) 4

-- Define the function g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-2 * x)

-- Theorem statement
theorem domain_of_g (f : ℝ → ℝ) (hf : Set.MapsTo f DomainF (Set.range f)) :
  {x : ℝ | g f x ∈ Set.range f} = Set.Icc (-2) 4 := by sorry

end domain_of_g_l872_87242


namespace expression_value_l872_87260

theorem expression_value : 
  (1/2) * (Real.log 12 / Real.log 3) - (Real.log 2 / Real.log 3) + 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  ((-2)^4)^(1/4) + (Real.sqrt 3 - 1)^0 = 11/2 := by
  sorry

end expression_value_l872_87260


namespace functional_equation_properties_l872_87215

theorem functional_equation_properties 
  (f g h : ℝ → ℝ)
  (hf : ∀ x y : ℝ, f (x * y) = x * f y)
  (hg : ∀ x y : ℝ, g (x * y) = x * g y)
  (hh : ∀ x y : ℝ, h (x * y) = x * h y) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x y : ℝ, (g ∘ h) (x * y) = x * (g ∘ h) y) ∧
  (g ∘ h = h ∘ g) :=
by sorry

end functional_equation_properties_l872_87215


namespace exponent_calculation_l872_87232

theorem exponent_calculation : (3^5 / 3^2) * 5^6 = 421875 := by
  sorry

end exponent_calculation_l872_87232


namespace smallest_divisible_by_15_20_18_l872_87289

theorem smallest_divisible_by_15_20_18 :
  ∃ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 20 ∣ n ∧ 18 ∣ n ∧ ∀ m : ℕ, m > 0 → 15 ∣ m → 20 ∣ m → 18 ∣ m → n ≤ m :=
by
  sorry

end smallest_divisible_by_15_20_18_l872_87289


namespace triangle_area_l872_87290

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b = 6 →
  a = 2 * c →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end triangle_area_l872_87290


namespace square_area_problem_l872_87277

theorem square_area_problem (x : ℝ) : 
  (5 * x - 18 = 27 - 4 * x) →
  (5 * x - 18)^2 = 49 := by
  sorry

end square_area_problem_l872_87277


namespace total_count_formula_specific_case_l872_87201

/-- Represents the structure of a plant with branches and small branches -/
structure Plant where
  branches : ℕ
  smallBranches : ℕ

/-- Calculates the total number of stems, branches, and small branches in a plant -/
def totalCount (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranches

/-- Theorem stating that the total count equals x^2 + x + 1 -/
theorem total_count_formula (x : ℕ) :
  totalCount { branches := x, smallBranches := x } = x^2 + x + 1 := by
  sorry

/-- The specific case where the total count is 73 -/
theorem specific_case : ∃ x : ℕ, totalCount { branches := x, smallBranches := x } = 73 := by
  sorry

end total_count_formula_specific_case_l872_87201


namespace inequality_solution_set_l872_87231

-- Define the solution set
def solution_set := {x : ℝ | x < 5}

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | |x - 8| - |x - 4| > 2} = solution_set := by
  sorry

end inequality_solution_set_l872_87231


namespace third_circle_radius_l872_87295

/-- Given two externally tangent circles and a third circle tangent to both and their center line, 
    prove that the radius of the third circle is √46 - 5 -/
theorem third_circle_radius 
  (P Q R : ℝ × ℝ) -- Centers of the three circles
  (r : ℝ) -- Radius of the third circle
  (h1 : dist P Q = 10) -- Distance between centers of first two circles
  (h2 : dist P R = 3 + r) -- Distance from P to R
  (h3 : dist Q R = 7 + r) -- Distance from Q to R
  (h4 : (R.1 - P.1) * (Q.1 - P.1) + (R.2 - P.2) * (Q.2 - P.2) = 0) -- R is on the perpendicular bisector of PQ
  : r = Real.sqrt 46 - 5 := by
  sorry

end third_circle_radius_l872_87295


namespace inequality_system_solution_range_l872_87224

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 5 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x : ℝ) - a ≤ 0 ∧ 7 + 2 * (x : ℝ) > 1)) →
  2 ≤ a ∧ a < 3 :=
by sorry

end inequality_system_solution_range_l872_87224


namespace ferris_wheel_ticket_cost_l872_87212

theorem ferris_wheel_ticket_cost 
  (initial_tickets : ℕ) 
  (remaining_tickets : ℕ) 
  (total_spent : ℕ) 
  (h1 : initial_tickets = 6)
  (h2 : remaining_tickets = 3)
  (h3 : total_spent = 27) :
  total_spent / (initial_tickets - remaining_tickets) = 9 :=
by sorry

end ferris_wheel_ticket_cost_l872_87212


namespace least_positive_integer_with_remainders_l872_87206

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  n % 6 = 3 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 57 := by
  sorry

end least_positive_integer_with_remainders_l872_87206


namespace tv_cash_savings_l872_87266

/-- Calculates the savings when buying a television with cash instead of an installment plan. -/
theorem tv_cash_savings (cash_price : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) (num_months : ℕ) : 
  cash_price = 400 →
  down_payment = 120 →
  monthly_payment = 30 →
  num_months = 12 →
  down_payment + monthly_payment * num_months - cash_price = 80 := by
sorry

end tv_cash_savings_l872_87266


namespace cricket_team_captain_age_l872_87210

theorem cricket_team_captain_age (team_size : ℕ) (whole_team_avg_age : ℕ) 
  (captain_age wicket_keeper_age : ℕ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  whole_team_avg_age = 21 →
  (whole_team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) + 1 = whole_team_avg_age →
  captain_age = 24 := by
sorry

end cricket_team_captain_age_l872_87210


namespace no_valid_operation_l872_87280

-- Define the set of standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an arithmetic operation
def applyOp (op : ArithOp) (a b : Int) : Option Int :=
  match op with
  | ArithOp.Add => some (a + b)
  | ArithOp.Sub => some (a - b)
  | ArithOp.Mul => some (a * b)
  | ArithOp.Div => if b ≠ 0 then some (a / b) else none

-- Theorem statement
theorem no_valid_operation :
  ∀ (op : ArithOp), (applyOp op 7 4).map (λ x => x + 5 - (3 - 2)) ≠ some 4 := by
  sorry


end no_valid_operation_l872_87280


namespace sodium_carbonate_mass_fraction_l872_87258

/-- Given the number of moles of Na₂CO₃, its molar mass, and the total mass of the solution,
    prove that the mass fraction of Na₂CO₃ in the solution is 10%. -/
theorem sodium_carbonate_mass_fraction 
  (n : Real) 
  (M : Real) 
  (m_solution : Real) 
  (h1 : n = 0.125) 
  (h2 : M = 106) 
  (h3 : m_solution = 132.5) : 
  (n * M * 100 / m_solution) = 10 := by
  sorry

#check sodium_carbonate_mass_fraction

end sodium_carbonate_mass_fraction_l872_87258


namespace sin_tan_40_deg_l872_87283

theorem sin_tan_40_deg : 4 * Real.sin (40 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end sin_tan_40_deg_l872_87283


namespace solution_comparison_l872_87216

theorem solution_comparison (a a' b b' k : ℝ) 
  (ha : a ≠ 0) (ha' : a' ≠ 0) (hk : k > 0) :
  (-kb / a < -b' / a') ↔ (k * b * a' > a * b') :=
sorry

end solution_comparison_l872_87216
