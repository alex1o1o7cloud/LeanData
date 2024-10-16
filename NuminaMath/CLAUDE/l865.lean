import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expressions_l865_86517

theorem simplify_expressions :
  (∀ (a b c d : ℝ), 
    a = 4 * Real.sqrt 5 ∧ 
    b = Real.sqrt 45 ∧ 
    c = Real.sqrt 8 ∧ 
    d = 4 * Real.sqrt 2 →
    a + b - c + d = 7 * Real.sqrt 5 + 2 * Real.sqrt 2) ∧
  (∀ (e f g : ℝ),
    e = 2 * Real.sqrt 48 ∧
    f = 3 * Real.sqrt 27 ∧
    g = Real.sqrt 6 →
    (e - f) / g = -(Real.sqrt 2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l865_86517


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l865_86554

/-- Given a quadratic equation x² - 4x - 2 = 0, prove that the correct completion of the square is (x-2)² = 6 -/
theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x - 2 = 0 → (x - 2)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l865_86554


namespace NUMINAMATH_CALUDE_inequality_proof_l865_86537

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a / (2 * b + 2 * c)) + Real.sqrt (b / (2 * a + 2 * c)) + Real.sqrt (c / (2 * a + 2 * b)) > 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l865_86537


namespace NUMINAMATH_CALUDE_crow_speed_l865_86536

/-- Crow's flight speed calculation -/
theorem crow_speed (distance_to_ditch : ℝ) (num_trips : ℕ) (time_hours : ℝ) :
  distance_to_ditch = 400 →
  num_trips = 15 →
  time_hours = 1.5 →
  (2 * distance_to_ditch * num_trips) / (1000 * time_hours) = 8 := by
  sorry

end NUMINAMATH_CALUDE_crow_speed_l865_86536


namespace NUMINAMATH_CALUDE_cookie_distribution_l865_86531

theorem cookie_distribution (total_cookies : ℚ) (blue_green_fraction : ℚ) (green_ratio : ℚ) :
  blue_green_fraction = 2/3 ∧ 
  green_ratio = 5/9 → 
  ∃ blue_fraction : ℚ, blue_fraction = 8/27 ∧ 
    blue_fraction + (blue_green_fraction - blue_fraction) + (1 - blue_green_fraction) = 1 ∧
    (blue_green_fraction - blue_fraction) / blue_green_fraction = green_ratio :=
by sorry

end NUMINAMATH_CALUDE_cookie_distribution_l865_86531


namespace NUMINAMATH_CALUDE_product_abc_l865_86535

theorem product_abc (a b c : ℕ+) (h : a * b^3 = 180) : a * b * c = 60 * c := by
  sorry

end NUMINAMATH_CALUDE_product_abc_l865_86535


namespace NUMINAMATH_CALUDE_first_day_pages_l865_86503

/-- Proves that given the specific writing pattern and remaining pages, the writer wrote 25 pages on the first day -/
theorem first_day_pages (total_pages remaining_pages day_4_pages : ℕ) 
  (h1 : total_pages = 500)
  (h2 : remaining_pages = 315)
  (h3 : day_4_pages = 10) : 
  ∃ x : ℕ, x + 2*x + 4*x + day_4_pages = total_pages - remaining_pages ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_first_day_pages_l865_86503


namespace NUMINAMATH_CALUDE_shaded_ratio_is_one_ninth_l865_86506

-- Define the structure of our square grid
def SquareGrid :=
  { n : ℕ // n > 0 }

-- Define the large square
def LargeSquare : SquareGrid :=
  ⟨6, by norm_num⟩

-- Define the number of squares in the shaded region
def ShadedSquares : ℕ := 4

-- Define the ratio of shaded area to total area
def ShadedRatio (grid : SquareGrid) (shaded : ℕ) : ℚ :=
  shaded / (grid.val ^ 2 : ℚ)

-- Theorem statement
theorem shaded_ratio_is_one_ninth :
  ShadedRatio LargeSquare ShadedSquares = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_ratio_is_one_ninth_l865_86506


namespace NUMINAMATH_CALUDE_car_speed_comparison_l865_86558

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 2 / (1 / u + 1 / v)
  let y := (u + v) / 2
  x ≤ y := by
sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l865_86558


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l865_86585

theorem gcd_lcm_sum : Nat.gcd 42 98 + Nat.lcm 60 15 = 74 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l865_86585


namespace NUMINAMATH_CALUDE_water_and_bottle_weights_l865_86582

/-- The weight of one cup of water in grams -/
def cup_weight : ℝ := 80

/-- The weight of one empty bottle in grams -/
def bottle_weight : ℝ := 200

/-- The total weight of 3 cups of water and 1 empty bottle in grams -/
def weight_3cups_1bottle : ℝ := 440

/-- The total weight of 5 cups of water and 1 empty bottle in grams -/
def weight_5cups_1bottle : ℝ := 600

theorem water_and_bottle_weights :
  (3 * cup_weight + bottle_weight = weight_3cups_1bottle) ∧
  (5 * cup_weight + bottle_weight = weight_5cups_1bottle) := by
  sorry

end NUMINAMATH_CALUDE_water_and_bottle_weights_l865_86582


namespace NUMINAMATH_CALUDE_perfect_square_expression_l865_86580

theorem perfect_square_expression (x y z : ℤ) :
  9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3*x*y*z) =
  ((x + y + z)^2 - 6*(x*y + y*z + z*x))^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_expression_l865_86580


namespace NUMINAMATH_CALUDE_mike_picked_52_peaches_l865_86534

/-- The number of peaches Mike picked at the orchard -/
def peaches_picked (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

/-- Theorem stating that Mike picked 52 peaches at the orchard -/
theorem mike_picked_52_peaches (initial : ℕ) (total : ℕ) 
  (h1 : initial = 34) 
  (h2 : total = 86) : 
  peaches_picked initial total = 52 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_52_peaches_l865_86534


namespace NUMINAMATH_CALUDE_x_range_given_inequality_l865_86538

theorem x_range_given_inequality :
  (∀ t : ℝ, -1 ≤ t ∧ t ≤ 3 → 
    ∀ x : ℝ, x^2 - (t^2 + t - 3)*x + t^2*(t - 3) > 0) →
  ∀ x : ℝ, x ∈ (Set.Iio (-4) ∪ Set.Ioi 9) :=
by sorry

end NUMINAMATH_CALUDE_x_range_given_inequality_l865_86538


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l865_86550

/-- The equation of a line passing through (-1, 2) with a slope angle of 45° is x - y + 3 = 0 -/
theorem line_equation_through_point_with_angle (x y : ℝ) :
  (x + 1 = -1 ∧ y - 2 = 0) →  -- The line passes through (-1, 2)
  (Real.tan (45 * π / 180) = 1) →  -- The slope angle is 45°
  x - y + 3 = 0  -- The equation of the line
  := by sorry


end NUMINAMATH_CALUDE_line_equation_through_point_with_angle_l865_86550


namespace NUMINAMATH_CALUDE_prob_four_sixes_eq_one_over_1296_l865_86525

-- Define a fair six-sided die
def fair_six_sided_die : Finset ℕ := Finset.range 6

-- Define the probability of rolling a specific number on a fair six-sided die
def prob_single_roll (n : ℕ) : ℚ :=
  if n ∈ fair_six_sided_die then 1 / 6 else 0

-- Define the probability of rolling four sixes
def prob_four_sixes : ℚ := (prob_single_roll 6) ^ 4

-- Theorem statement
theorem prob_four_sixes_eq_one_over_1296 :
  prob_four_sixes = 1 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_four_sixes_eq_one_over_1296_l865_86525


namespace NUMINAMATH_CALUDE_lara_age_proof_l865_86505

/-- Lara's age 10 years from now, given her age 7 years ago -/
def lara_future_age (age_7_years_ago : ℕ) : ℕ :=
  age_7_years_ago + 7 + 10

/-- Theorem stating Lara's age 10 years from now -/
theorem lara_age_proof :
  lara_future_age 9 = 26 := by
  sorry

end NUMINAMATH_CALUDE_lara_age_proof_l865_86505


namespace NUMINAMATH_CALUDE_first_fun_friday_is_april_28_l865_86596

/-- Represents a date in a calendar year -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns true if the given date is a Friday -/
def isFriday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns true if the given month has five Fridays -/
def hasFiveFridays (month : Nat) (year : Nat) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns the date of the first Fun Friday after the given start date -/
def firstFunFriday (startDate : Date) (startDay : DayOfWeek) : Date :=
  sorry

theorem first_fun_friday_is_april_28 :
  let fiscalYearStart : Date := ⟨3, 1⟩
  let fiscalYearStartDay : DayOfWeek := DayOfWeek.Wednesday
  firstFunFriday fiscalYearStart fiscalYearStartDay = ⟨4, 28⟩ := by
  sorry

end NUMINAMATH_CALUDE_first_fun_friday_is_april_28_l865_86596


namespace NUMINAMATH_CALUDE_homework_is_duration_l865_86542

-- Define a type for time expressions
inductive TimeExpression
  | PointInTime (description : String)
  | Duration (description : String)

-- Define the given options
def option_a : TimeExpression := TimeExpression.PointInTime "Get up at 6:30"
def option_b : TimeExpression := TimeExpression.PointInTime "School ends at 3:40"
def option_c : TimeExpression := TimeExpression.Duration "It took 30 minutes to do the homework"

-- Define a function to check if a TimeExpression represents a duration
def is_duration (expr : TimeExpression) : Prop :=
  match expr with
  | TimeExpression.Duration _ => True
  | _ => False

-- Theorem to prove
theorem homework_is_duration :
  is_duration option_c ∧ ¬is_duration option_a ∧ ¬is_duration option_b :=
sorry

end NUMINAMATH_CALUDE_homework_is_duration_l865_86542


namespace NUMINAMATH_CALUDE_trout_calculation_l865_86597

/-- The number of people fishing -/
def num_people : ℕ := 2

/-- The number of trout each person gets after splitting -/
def trout_per_person : ℕ := 9

/-- The total number of trout caught -/
def total_trout : ℕ := num_people * trout_per_person

theorem trout_calculation : total_trout = 18 := by
  sorry

end NUMINAMATH_CALUDE_trout_calculation_l865_86597


namespace NUMINAMATH_CALUDE_bowl_water_percentage_l865_86592

theorem bowl_water_percentage (initial_water : ℝ) (capacity : ℝ) : 
  capacity > 0 →
  initial_water ≥ 0 →
  initial_water ≤ capacity →
  (initial_water + 4 = 14) →
  ((initial_water + 4) / capacity = 0.7) →
  (initial_water / capacity = 0.5) :=
by sorry

end NUMINAMATH_CALUDE_bowl_water_percentage_l865_86592


namespace NUMINAMATH_CALUDE_player_a_strategy_wins_l865_86561

-- Define the grid as a 3x3 matrix of real numbers
def Grid := Matrix (Fin 3) (Fin 3) ℝ

-- Define a function to calculate the sum of first and third rows
def sumRows (g : Grid) : ℝ := 
  (g 0 0 + g 0 1 + g 0 2) + (g 2 0 + g 2 1 + g 2 2)

-- Define a function to calculate the sum of first and third columns
def sumCols (g : Grid) : ℝ := 
  (g 0 0 + g 1 0 + g 2 0) + (g 0 2 + g 1 2 + g 2 2)

-- Theorem statement
theorem player_a_strategy_wins 
  (cards : Finset ℝ) 
  (h_card_count : cards.card = 9) : 
  ∃ (g : Grid), (∀ i j, g i j ∈ cards) ∧ sumRows g ≥ sumCols g := by
  sorry


end NUMINAMATH_CALUDE_player_a_strategy_wins_l865_86561


namespace NUMINAMATH_CALUDE_triangle_area_is_three_l865_86569

/-- Triangle ABC with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The line equation x - y = 5 -/
def LineEquation (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 = 5

/-- The area of a triangle -/
def TriangleArea (t : Triangle) : ℝ :=
  sorry

/-- The theorem statement -/
theorem triangle_area_is_three :
  ∀ (t : Triangle),
    t.A = (3, 0) →
    t.B = (0, 3) →
    LineEquation t.C →
    TriangleArea t = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_three_l865_86569


namespace NUMINAMATH_CALUDE_x_range_l865_86545

theorem x_range (x : ℝ) : 
  (Real.log (x^2 - 2*x - 2) ≥ 0) → 
  ¬(0 < x ∧ x < 4) → 
  (x ≥ 4 ∨ x ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l865_86545


namespace NUMINAMATH_CALUDE_pool_swimmers_l865_86549

theorem pool_swimmers (total : ℕ) (first_day : ℕ) (extra_second_day : ℕ) 
  (h1 : total = 246)
  (h2 : first_day = 79)
  (h3 : extra_second_day = 47) :
  ∃ (third_day : ℕ), 
    third_day = 60 ∧ 
    first_day + (third_day + extra_second_day) + third_day = total :=
by
  sorry

end NUMINAMATH_CALUDE_pool_swimmers_l865_86549


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_l865_86512

/-- The total number of books in the 'crazy silly school' series -/
def total_books (x y : ℕ) : ℕ := x^2 + y

/-- Theorem stating that the total number of books is 177 when x = 13 and y = 8 -/
theorem crazy_silly_school_books : total_books 13 8 = 177 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_l865_86512


namespace NUMINAMATH_CALUDE_smallest_block_with_270_hidden_cubes_l865_86524

/-- Represents the dimensions of a rectangular block --/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the total number of cubes in a block --/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of hidden cubes in a block --/
def hiddenCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Theorem stating the smallest possible value of N --/
theorem smallest_block_with_270_hidden_cubes :
  ∃ (d : BlockDimensions),
    hiddenCubes d = 270 ∧
    (∀ (d' : BlockDimensions), hiddenCubes d' = 270 → totalCubes d ≤ totalCubes d') ∧
    totalCubes d = 420 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_with_270_hidden_cubes_l865_86524


namespace NUMINAMATH_CALUDE_jellybean_probability_l865_86564

def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 5
def blue_jellybeans : ℕ := 2
def yellow_jellybeans : ℕ := 5
def picked_jellybeans : ℕ := 4

theorem jellybean_probability : 
  (Nat.choose red_jellybeans 3 * Nat.choose (total_jellybeans - red_jellybeans) 1) / 
  Nat.choose total_jellybeans picked_jellybeans = 14 / 99 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l865_86564


namespace NUMINAMATH_CALUDE_profit_maximization_l865_86559

def profit_function (x : ℝ) : ℝ := -2 * x^2 + 200 * x - 3200

theorem profit_maximization (x : ℝ) :
  35 ≤ x ∧ x ≤ 45 →
  (∀ y : ℝ, 35 ≤ y ∧ y ≤ 45 → profit_function y ≤ profit_function 45) ∧
  profit_function 45 = 1750 ∧
  (∀ z : ℝ, 35 ≤ z ∧ z ≤ 45 ∧ profit_function z ≥ 1600 → 40 ≤ z ∧ z ≤ 45) :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l865_86559


namespace NUMINAMATH_CALUDE_percentage_of_girls_who_want_to_be_doctors_l865_86501

theorem percentage_of_girls_who_want_to_be_doctors
  (total_students : ℝ)
  (boys_ratio : ℝ)
  (boys_doctor_ratio : ℝ)
  (boys_doctor_all_doctor_ratio : ℝ)
  (h1 : boys_ratio = 3 / 5)
  (h2 : boys_doctor_ratio = 1 / 3)
  (h3 : boys_doctor_all_doctor_ratio = 2 / 5) :
  (((1 - boys_ratio) * total_students) / total_students - 
   ((1 - boys_doctor_all_doctor_ratio) * (boys_ratio * boys_doctor_ratio * total_students)) / 
   ((1 - boys_ratio) * total_students)) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_girls_who_want_to_be_doctors_l865_86501


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l865_86563

/-- Given a normal distribution with mean 12 and a value 9.6 that is 2 standard deviations
    below the mean, the standard deviation is 1.2 -/
theorem normal_distribution_std_dev (μ σ : ℝ) (h1 : μ = 12) (h2 : μ - 2 * σ = 9.6) :
  σ = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l865_86563


namespace NUMINAMATH_CALUDE_unique_solution_iff_prime_l865_86572

theorem unique_solution_iff_prime (n : ℕ) : 
  (∃! (x y : ℕ), (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / n) ↔ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_prime_l865_86572


namespace NUMINAMATH_CALUDE_quadruple_equality_l865_86551

theorem quadruple_equality (a b c d : ℝ) : 
  (∀ X : ℝ, X^2 + a*X + b = (X-a)*(X-c)) ∧
  (∀ X : ℝ, X^2 + c*X + d = (X-b)*(X-d)) →
  ((a = 1 ∧ b = 2 ∧ c = -2 ∧ d = 0) ∨ 
   (a = -1 ∧ b = -2 ∧ c = 2 ∧ d = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equality_l865_86551


namespace NUMINAMATH_CALUDE_floor_product_eq_sum_iff_in_solution_set_l865_86595

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The solution set for the equation [x] · [y] = x + y -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 = 2) ∨ (2 ≤ p.1 ∧ p.1 < 4 ∧ p.1 ≠ 3 ∧ p.2 = 6 - p.1)}

theorem floor_product_eq_sum_iff_in_solution_set (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (floor x) * (floor y) = x + y ↔ (x, y) ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_floor_product_eq_sum_iff_in_solution_set_l865_86595


namespace NUMINAMATH_CALUDE_hypotenuse_length_l865_86552

theorem hypotenuse_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (a^2 + b^2) * (a^2 + b^2 + 1) = 12 →
  a^2 + b^2 = c^2 →
  c = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l865_86552


namespace NUMINAMATH_CALUDE_apollonius_circle_minimum_l865_86565

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 1)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 2)^2 + (P.2 - 1)^2)

-- Define the symmetry line
def symmetry_line (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  m * P.1 + n * P.2 = 2

-- Main theorem
theorem apollonius_circle_minimum (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ P : ℝ × ℝ, distance_ratio P → ∃ P', distance_ratio P' ∧ symmetry_line m n ((P.1 + P'.1)/2, (P.2 + P'.2)/2)) →
  (2/m + 5/n) ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_apollonius_circle_minimum_l865_86565


namespace NUMINAMATH_CALUDE_max_value_of_g_l865_86576

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) : ℝ := 4 * x - x^4

/-- The interval [0, 2] -/
def I : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

theorem max_value_of_g :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), x ∈ I → g x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l865_86576


namespace NUMINAMATH_CALUDE_parking_garage_floor_distance_l865_86509

/-- Calculates the distance between floors in a parking garage --/
theorem parking_garage_floor_distance (
  total_floors : ℕ) 
  (gate_interval : ℕ) 
  (gate_time : ℝ) 
  (driving_speed : ℝ) 
  (total_time : ℝ)
  (h1 : total_floors = 12)
  (h2 : gate_interval = 3)
  (h3 : gate_time = 120) -- 2 minutes in seconds
  (h4 : driving_speed = 10)
  (h5 : total_time = 1440) :
  ∃ (distance : ℝ), abs (distance - 872.7) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_floor_distance_l865_86509


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l865_86556

/-- Given a hyperbola with equation x²/48 - y²/16 = 1, 
    the distance between its vertices is 8√3. -/
theorem hyperbola_vertex_distance :
  ∀ (x y : ℝ), 
  x^2 / 48 - y^2 / 16 = 1 →
  ∃ (d : ℝ), d = 8 * Real.sqrt 3 ∧ d = 2 * Real.sqrt 48 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l865_86556


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_one_l865_86508

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem logarithm_expression_equals_one :
  2 * (lg (Real.sqrt 2))^2 + lg (Real.sqrt 2) * lg 5 + 
  Real.sqrt ((lg (Real.sqrt 2))^2 - lg 2 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_one_l865_86508


namespace NUMINAMATH_CALUDE_three_planes_division_l865_86588

theorem three_planes_division (x y : ℕ) : 
  (x = 4 ∧ y = 8) → y - x = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_planes_division_l865_86588


namespace NUMINAMATH_CALUDE_increasing_f_implies_m_bound_l865_86571

/-- A cubic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * m * x + 6

theorem increasing_f_implies_m_bound :
  (∀ x > 2, ∀ y > x, f m y > f m x) →
  m ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_m_bound_l865_86571


namespace NUMINAMATH_CALUDE_evaluate_expression_l865_86557

theorem evaluate_expression : (5^5 * 5^3) / 3^6 * 2^5 = 12480000 / 729 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l865_86557


namespace NUMINAMATH_CALUDE_inequality_check_l865_86515

theorem inequality_check : 
  (¬(0 < -5)) ∧ 
  (¬(7 < -1)) ∧ 
  (¬(10 < (1/4 : ℚ))) ∧ 
  (¬(-1 < -3)) ∧ 
  (-8 < -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_check_l865_86515


namespace NUMINAMATH_CALUDE_sector_max_area_l865_86526

/-- Given a rope of length 20cm forming a sector, the area of the sector is maximized when the central angle is 2 radians. -/
theorem sector_max_area (r l α : ℝ) : 
  0 < r → r < 10 →
  l + 2 * r = 20 →
  l = r * α →
  ∀ r' l' α', 
    0 < r' → r' < 10 →
    l' + 2 * r' = 20 →
    l' = r' * α' →
    r * l ≥ r' * l' →
  α = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l865_86526


namespace NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l865_86573

/-- Represents a grid with black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Nat)

/-- Calculates the total number of adjacent cell pairs in the grid -/
def total_pairs (g : Grid) : Nat :=
  2 * g.size * (g.size - 1)

/-- Represents the minimum number of white cell pairs in the grid -/
def min_white_pairs (g : Grid) : Nat :=
  total_pairs g - (g.black_cells + 40)

/-- Theorem stating the minimum number of white cell pairs in an 8x8 grid with 20 black cells -/
theorem min_white_pairs_8x8_20black :
  ∀ (g : Grid), g.size = 8 → g.black_cells = 20 → min_white_pairs g = 34 :=
by sorry

end NUMINAMATH_CALUDE_min_white_pairs_8x8_20black_l865_86573


namespace NUMINAMATH_CALUDE_robert_read_315_pages_l865_86544

/-- The number of pages Robert read in the book over 10 days -/
def total_pages (days_1 days_2 days_3 : ℕ) 
                (pages_per_day_1 pages_per_day_2 pages_day_3 : ℕ) : ℕ :=
  days_1 * pages_per_day_1 + days_2 * pages_per_day_2 + pages_day_3

/-- Theorem stating that Robert read 315 pages in total -/
theorem robert_read_315_pages : 
  total_pages 5 4 1 25 40 30 = 315 := by
  sorry

#eval total_pages 5 4 1 25 40 30

end NUMINAMATH_CALUDE_robert_read_315_pages_l865_86544


namespace NUMINAMATH_CALUDE_parallel_lines_equation_l865_86521

/-- A line in 2D space represented by its slope-intercept form -/
structure Line where
  slope : ℚ
  yIntercept : ℚ

/-- Distance between two parallel lines -/
def distanceBetweenParallelLines (l1 l2 : Line) : ℚ :=
  sorry

/-- Checks if two lines are parallel -/
def areParallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem parallel_lines_equation (l : Line) (P : ℚ × ℚ) (m : Line) :
  l.slope = -3/4 →
  P = (-2, 5) →
  areParallel l m →
  distanceBetweenParallelLines l m = 3 →
  (∃ (c : ℚ), (3 * P.1 + 4 * P.2 + c = 0 ∧ (c = 1 ∨ c = -29))) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_equation_l865_86521


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l865_86562

/-- Represents a chemical compound with a given number of Carbon, Hydrogen, and Oxygen atoms -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

/-- Theorem: A compound with 4 Carbon atoms, 1 Hydrogen atom, and a molecular weight of 65 g/mol contains 1 Oxygen atom -/
theorem compound_oxygen_count :
  ∃ (c : Compound),
    c.carbon = 4 ∧
    c.hydrogen = 1 ∧
    c.oxygen = 1 ∧
    molecularWeight c 12.01 1.008 16.00 = 65 := by
  sorry


end NUMINAMATH_CALUDE_compound_oxygen_count_l865_86562


namespace NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l865_86570

theorem fraction_equality_implies_c_geq_one 
  (a b : ℕ+) 
  (c : ℝ) 
  (h_c_pos : c > 0) 
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : 
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l865_86570


namespace NUMINAMATH_CALUDE_profit_percentage_unchanged_l865_86518

/-- Represents a retailer's sales and profit information -/
structure RetailerInfo where
  monthly_sales : ℝ
  profit_percentage : ℝ
  discount_percentage : ℝ
  break_even_sales : ℝ

/-- The retailer's original sales information -/
def original_info : RetailerInfo :=
  { monthly_sales := 100
  , profit_percentage := 0.10
  , discount_percentage := 0
  , break_even_sales := 100 }

/-- The retailer's sales information with discount -/
def discounted_info : RetailerInfo :=
  { monthly_sales := 222.22
  , profit_percentage := 0.10
  , discount_percentage := 0.05
  , break_even_sales := 222.22 }

/-- Calculates the total profit for a given RetailerInfo -/
def total_profit (info : RetailerInfo) (price : ℝ) : ℝ :=
  info.monthly_sales * (info.profit_percentage - info.discount_percentage) * price

/-- Theorem stating that the profit percentage remains the same
    regardless of the discount, given the break-even sales volume -/
theorem profit_percentage_unchanged
  (price : ℝ)
  (h_price_pos : price > 0) :
  original_info.profit_percentage = discounted_info.profit_percentage :=
by
  sorry


end NUMINAMATH_CALUDE_profit_percentage_unchanged_l865_86518


namespace NUMINAMATH_CALUDE_coin_bag_total_amount_l865_86540

theorem coin_bag_total_amount :
  ∃ (x : ℕ),
    let one_cent := x
    let ten_cent := 2 * x
    let twenty_five_cent := 3 * (2 * x)
    let total := one_cent * 1 + ten_cent * 10 + twenty_five_cent * 25
    total = 342 := by
  sorry

end NUMINAMATH_CALUDE_coin_bag_total_amount_l865_86540


namespace NUMINAMATH_CALUDE_dog_weight_gain_exists_l865_86583

/-- Represents a dog with age and weight -/
structure Dog where
  age : ℕ
  weight : ℝ

/-- Represents the annual weight gain of a dog -/
def annualGain (d : Dog) (gain : ℝ) : Prop :=
  ∃ (initialWeight : ℝ), initialWeight + gain * (d.age - 1) = d.weight

/-- Theorem stating that for any dog, there exists some annual weight gain -/
theorem dog_weight_gain_exists (d : Dog) : ∃ (gain : ℝ), annualGain d gain :=
sorry

end NUMINAMATH_CALUDE_dog_weight_gain_exists_l865_86583


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l865_86500

/-- 
Given a rectangular prism with length l, width w, and height h,
where l = 2w, w = 2h, and the sum of all edge lengths is 56,
prove that the volume is 64.
-/
theorem rectangular_prism_volume (l w h : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : w = 2 * h) 
  (h3 : 4 * (l + w + h) = 56) : 
  l * w * h = 64 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l865_86500


namespace NUMINAMATH_CALUDE_possible_set_A_l865_86532

-- Define the set B
def B : Set ℝ := {x | x ≥ 0}

-- Define the theorem
theorem possible_set_A (A : Set ℝ) (h1 : A ∩ B = A) : 
  ∃ A', A' = {1, 2} ∧ A' ∩ B = A' :=
sorry

end NUMINAMATH_CALUDE_possible_set_A_l865_86532


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l865_86587

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (10 + Real.sqrt x) = 4 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l865_86587


namespace NUMINAMATH_CALUDE_exam_score_calculation_l865_86547

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 160)
  (h3 : correct_answers = 44)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l865_86547


namespace NUMINAMATH_CALUDE_total_students_proof_l865_86591

def school_problem (n : ℕ) (largest_class : ℕ) (diff : ℕ) : ℕ :=
  let class_sizes := List.range n |>.map (fun i => largest_class - i * diff)
  class_sizes.sum

theorem total_students_proof :
  school_problem 5 25 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_students_proof_l865_86591


namespace NUMINAMATH_CALUDE_shoe_probability_theorem_l865_86566

/-- Represents the number of pairs of shoes of a specific color -/
structure ColorPairs :=
  (count : ℕ)

/-- Represents Sue's shoe collection -/
structure ShoeCollection :=
  (total_pairs : ℕ)
  (black : ColorPairs)
  (brown : ColorPairs)
  (gray : ColorPairs)

/-- Calculates the probability of picking two shoes of the same color, one left and one right -/
def probability_same_color_different_feet (collection : ShoeCollection) : ℚ :=
  let total_shoes := 2 * collection.total_pairs
  let black_prob := (2 * collection.black.count : ℚ) / total_shoes * collection.black.count / (total_shoes - 1)
  let brown_prob := (2 * collection.brown.count : ℚ) / total_shoes * collection.brown.count / (total_shoes - 1)
  let gray_prob := (2 * collection.gray.count : ℚ) / total_shoes * collection.gray.count / (total_shoes - 1)
  black_prob + brown_prob + gray_prob

theorem shoe_probability_theorem (sue_collection : ShoeCollection) 
  (h1 : sue_collection.total_pairs = 12)
  (h2 : sue_collection.black.count = 7)
  (h3 : sue_collection.brown.count = 3)
  (h4 : sue_collection.gray.count = 2) :
  probability_same_color_different_feet sue_collection = 31 / 138 := by
  sorry

end NUMINAMATH_CALUDE_shoe_probability_theorem_l865_86566


namespace NUMINAMATH_CALUDE_det_A_eq_16_l865_86567

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 2, -4; 6, -1, 3; 2, -3, 5]

theorem det_A_eq_16 : Matrix.det A = 16 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_16_l865_86567


namespace NUMINAMATH_CALUDE_mary_jamison_weight_difference_l865_86533

/-- Proves that Mary weighs 20 lbs less than Jamison given the conditions in the problem -/
theorem mary_jamison_weight_difference :
  ∀ (john mary jamison : ℝ),
    mary = 160 →
    john = mary + (1/4) * mary →
    john + mary + jamison = 540 →
    jamison - mary = 20 := by
  sorry

end NUMINAMATH_CALUDE_mary_jamison_weight_difference_l865_86533


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l865_86574

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x > 0}

theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = Set.Ioo (-3) 1) :
  b < 0 ∧ c > 0 ∧
  {x : ℝ | a * x - b < 0} = Set.Ioi 2 ∧
  {x : ℝ | a * x^2 - b * x + c < 0} = Set.Iic (-1) ∪ Set.Ioi 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l865_86574


namespace NUMINAMATH_CALUDE_particle_probability_l865_86514

/-- Probability of reaching (0,0) from position (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (P (x-1) y + P x (y-1) + P (x-1) (y-1)) / 3

/-- The probability of reaching (0,0) from (3,5) is 1385/19683 -/
theorem particle_probability : P 3 5 = 1385 / 19683 := by
  sorry

end NUMINAMATH_CALUDE_particle_probability_l865_86514


namespace NUMINAMATH_CALUDE_acute_angle_range_l865_86584

/-- The range of values for the acute angle of a line with slope k = 2m / (m^2 + 1) -/
theorem acute_angle_range (m : ℝ) (h1 : m ≥ 0) (h2 : m^2 + 1 ≥ 2*m) :
  let k := 2*m / (m^2 + 1)
  let θ := Real.arctan k
  0 ≤ θ ∧ θ ≤ π/4 :=
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l865_86584


namespace NUMINAMATH_CALUDE_geometric_sequence_cosine_l865_86541

/-- 
Given a geometric sequence {an} with common ratio √2,
prove that if sn(a7a8) = 3/5, then cos(2a5) = 7/25
-/
theorem geometric_sequence_cosine (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →  -- Common ratio is √2
  (∃ sn : ℝ, sn * (a 7 * a 8) = 3/5) →    -- sn(a7a8) = 3/5
  Real.cos (2 * a 5) = 7/25 :=             -- cos(2a5) = 7/25
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_cosine_l865_86541


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l865_86539

/-- A parabola passing through (-3, 0) with axis of symmetry x = -1 has coefficient sum of 0 -/
theorem parabola_coefficient_sum (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = -3 ∨ x = 1) →
  -b / (2 * a) = -1 →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l865_86539


namespace NUMINAMATH_CALUDE_rectangle_area_l865_86589

theorem rectangle_area (width length : ℝ) (h1 : length = width + 6) (h2 : 2 * (width + length) = 68) :
  width * length = 280 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l865_86589


namespace NUMINAMATH_CALUDE_katie_marbles_l865_86548

/-- The number of pink marbles Katie has -/
def pink_marbles : ℕ := 13

/-- The number of orange marbles Katie has -/
def orange_marbles : ℕ := pink_marbles - 9

/-- The number of purple marbles Katie has -/
def purple_marbles : ℕ := 4 * orange_marbles

/-- The total number of marbles Katie has -/
def total_marbles : ℕ := 33

theorem katie_marbles : 
  pink_marbles + orange_marbles + purple_marbles = total_marbles ∧ 
  orange_marbles = pink_marbles - 9 ∧
  purple_marbles = 4 * orange_marbles ∧
  pink_marbles = 13 := by sorry

end NUMINAMATH_CALUDE_katie_marbles_l865_86548


namespace NUMINAMATH_CALUDE_inequality_solution_set_l865_86594

theorem inequality_solution_set (x : ℝ) : 
  (1 - 2*x) / (x + 3) ≥ 1 ↔ -3 < x ∧ x ≤ -2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l865_86594


namespace NUMINAMATH_CALUDE_light_travel_distance_l865_86516

/-- The distance light travels in one year in kilometers. -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we want to calculate the light travel distance for. -/
def years : ℕ := 50

/-- Theorem stating the distance light travels in 50 years. -/
theorem light_travel_distance : 
  (light_year_distance * years : ℝ) = 4.7304e14 := by sorry

end NUMINAMATH_CALUDE_light_travel_distance_l865_86516


namespace NUMINAMATH_CALUDE_euler_formula_second_quadrant_l865_86513

theorem euler_formula_second_quadrant :
  let z : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_euler_formula_second_quadrant_l865_86513


namespace NUMINAMATH_CALUDE_students_per_bench_l865_86519

theorem students_per_bench (male_students : ℕ) (benches : ℕ) : 
  male_students = 29 →
  benches = 29 →
  ∃ (students_per_bench : ℕ), 
    students_per_bench ≥ 5 ∧
    students_per_bench * benches ≥ male_students + 4 * male_students :=
by sorry

end NUMINAMATH_CALUDE_students_per_bench_l865_86519


namespace NUMINAMATH_CALUDE_cosine_in_special_triangle_l865_86504

/-- 
In a triangle ABC, given that:
1. The side lengths a, b, c form a geometric sequence
2. c = 2a
Then, cos B = 3/4
-/
theorem cosine_in_special_triangle (a b c : ℝ) (h_positive : a > 0) 
  (h_geometric : b^2 = a * c) (h_relation : c = 2 * a) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  cos_B = 3/4 := by sorry

end NUMINAMATH_CALUDE_cosine_in_special_triangle_l865_86504


namespace NUMINAMATH_CALUDE_specific_stack_logs_l865_86579

/-- Represents a triangular stack of logs. -/
structure LogStack where
  bottom_logs : ℕ  -- Number of logs in the bottom row
  decrement : ℕ    -- Number of logs decreased in each row
  top_logs : ℕ     -- Number of logs in the top row

/-- Calculates the number of rows in a log stack. -/
def num_rows (stack : LogStack) : ℕ :=
  (stack.bottom_logs - stack.top_logs) / stack.decrement + 1

/-- Calculates the total number of logs in a stack. -/
def total_logs (stack : LogStack) : ℕ :=
  let n := num_rows stack
  n * (stack.bottom_logs + stack.top_logs) / 2

/-- Theorem stating the total number of logs in the specific stack. -/
theorem specific_stack_logs :
  let stack : LogStack := ⟨15, 2, 1⟩
  total_logs stack = 64 := by
  sorry


end NUMINAMATH_CALUDE_specific_stack_logs_l865_86579


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l865_86578

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle -/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ 
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1

theorem smallest_prime_perimeter_scalene_triangle : 
  ∃ (a b c : ℕ), 
    isScaleneTriangle a b c ∧ 
    areConsecutiveOddPrimes a b c ∧ 
    isPrime (a + b + c) ∧
    ∀ (x y z : ℕ), 
      isScaleneTriangle x y z → 
      areConsecutiveOddPrimes x y z → 
      isPrime (x + y + z) → 
      a + b + c ≤ x + y + z ∧
    a + b + c = 23 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l865_86578


namespace NUMINAMATH_CALUDE_impossible_arrangement_l865_86511

/-- Represents a cell in the table -/
structure Cell where
  row : Fin 2002
  col : Fin 2002

/-- Represents the table arrangement -/
def TableArrangement := Cell → Fin (2002^2)

/-- Checks if a triplet satisfies the product condition -/
def satisfiesProductCondition (a b c : Fin (2002^2)) : Prop :=
  (a.val + 1) * (b.val + 1) = c.val + 1 ∨
  (a.val + 1) * (c.val + 1) = b.val + 1 ∨
  (b.val + 1) * (c.val + 1) = a.val + 1

/-- Checks if a cell satisfies the condition in its row or column -/
def cellSatisfiesCondition (t : TableArrangement) (cell : Cell) : Prop :=
  ∃ (a b c : Cell),
    ((a.row = cell.row ∧ b.row = cell.row ∧ c.row = cell.row) ∨
     (a.col = cell.col ∧ b.col = cell.col ∧ c.col = cell.col)) ∧
    satisfiesProductCondition (t a) (t b) (t c)

/-- The main theorem stating the impossibility of the arrangement -/
theorem impossible_arrangement :
  ¬∃ (t : TableArrangement),
    (∀ (c₁ c₂ : Cell), c₁ ≠ c₂ → t c₁ ≠ t c₂) ∧
    (∀ (cell : Cell), cellSatisfiesCondition t cell) :=
  sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l865_86511


namespace NUMINAMATH_CALUDE_largest_power_dividing_factorial_l865_86599

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_power_dividing_factorial :
  ∃ k : ℕ, k = 3 ∧
  (∀ m : ℕ, 2012^m ∣ factorial 2012 → m ≤ k) ∧
  2012^k ∣ factorial 2012 :=
by sorry

end NUMINAMATH_CALUDE_largest_power_dividing_factorial_l865_86599


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l865_86586

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The bouncing ball theorem -/
theorem bouncing_ball_distance :
  let initialHeight : ℝ := 120
  let reboundFactor : ℝ := 0.75
  let bounces : ℕ := 5
  totalDistance initialHeight reboundFactor bounces = 612.1875 := by
  sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l865_86586


namespace NUMINAMATH_CALUDE_matt_received_more_than_lauren_l865_86593

-- Define the given conditions
def total_pencils : ℕ := 2 * 12
def pencils_to_lauren : ℕ := 6
def pencils_left : ℕ := 9

-- Define the number of pencils Matt received
def pencils_to_matt : ℕ := total_pencils - pencils_to_lauren - pencils_left

-- Theorem to prove
theorem matt_received_more_than_lauren : 
  pencils_to_matt - pencils_to_lauren = 3 := by
sorry

end NUMINAMATH_CALUDE_matt_received_more_than_lauren_l865_86593


namespace NUMINAMATH_CALUDE_composite_sum_of_squares_l865_86546

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) →
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_of_squares_l865_86546


namespace NUMINAMATH_CALUDE_square_value_l865_86529

theorem square_value (square : ℝ) : 
  (1.08 / 1.2) / 2.3 = 10.8 / square → square = 27.6 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l865_86529


namespace NUMINAMATH_CALUDE_min_value_quadratic_l865_86523

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), min_z = 5 ∧ ∀ z : ℝ, z = 5 * x^2 + 20 * x + 25 → z ≥ min_z :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l865_86523


namespace NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l865_86553

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^3 * (x + y - 2) = y^3 * (x + y - 2)

-- Define what it means for two lines to intersect
def intersecting_lines (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x

-- Theorem statement
theorem equation_represents_two_intersecting_lines :
  ∃ (f g : ℝ → ℝ), 
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) ∧
    intersecting_lines f g :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_intersecting_lines_l865_86553


namespace NUMINAMATH_CALUDE_value_of_7a_plus_3b_l865_86502

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 4

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem value_of_7a_plus_3b 
  (a b : ℝ) 
  (h1 : ∀ x, g x = (Function.invFun (f a b)) x - 5) 
  (h2 : Function.Injective (f a b)) :
  7 * a + 3 * b = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_7a_plus_3b_l865_86502


namespace NUMINAMATH_CALUDE_digit_difference_729_l865_86568

def base_3_digits (n : ℕ) : ℕ := 
  Nat.log 3 n + 1

def base_8_digits (n : ℕ) : ℕ := 
  Nat.log 8 n + 1

theorem digit_difference_729 : 
  base_3_digits 729 - base_8_digits 729 = 4 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_729_l865_86568


namespace NUMINAMATH_CALUDE_school_trip_equation_correct_l865_86555

/-- Represents the scenario of a school trip to Shaoshan -/
structure SchoolTrip where
  distance : ℝ
  delay : ℝ
  speedRatio : ℝ

/-- The equation representing the travel times for bus and car -/
def travelTimeEquation (trip : SchoolTrip) (x : ℝ) : Prop :=
  trip.distance / x = trip.distance / (trip.speedRatio * x) + trip.delay

/-- Theorem stating that the given equation correctly represents the scenario -/
theorem school_trip_equation_correct (x : ℝ) : 
  let trip : SchoolTrip := { 
    distance := 50,
    delay := 1/6,
    speedRatio := 1.2
  }
  travelTimeEquation trip x :=
by sorry

end NUMINAMATH_CALUDE_school_trip_equation_correct_l865_86555


namespace NUMINAMATH_CALUDE_mildred_oranges_proof_l865_86575

/-- Calculates the remaining oranges after Mildred's father and friend take some. -/
def remaining_oranges (initial : Float) (father_eats : Float) (friend_takes : Float) : Float :=
  initial - father_eats - friend_takes

/-- Proves that Mildred has 71.5 oranges left after her father and friend take some. -/
theorem mildred_oranges_proof (initial : Float) (father_eats : Float) (friend_takes : Float)
    (h1 : initial = 77.5)
    (h2 : father_eats = 2.25)
    (h3 : friend_takes = 3.75) :
    remaining_oranges initial father_eats friend_takes = 71.5 := by
  sorry

end NUMINAMATH_CALUDE_mildred_oranges_proof_l865_86575


namespace NUMINAMATH_CALUDE_expression_value_l865_86530

-- Define the expression
def f (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem statement
theorem expression_value : f (-1) = 8 := by sorry

end NUMINAMATH_CALUDE_expression_value_l865_86530


namespace NUMINAMATH_CALUDE_second_chapter_pages_count_l865_86528

/-- A book with two chapters -/
structure Book where
  total_pages : ℕ
  first_chapter_pages : ℕ
  second_chapter_pages : ℕ
  two_chapters : first_chapter_pages + second_chapter_pages = total_pages

/-- The specific book described in the problem -/
def problem_book : Book where
  total_pages := 81
  first_chapter_pages := 13
  second_chapter_pages := 68
  two_chapters := by sorry

theorem second_chapter_pages_count :
  problem_book.second_chapter_pages = 68 := by sorry

end NUMINAMATH_CALUDE_second_chapter_pages_count_l865_86528


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l865_86581

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 8)^2 = 49

-- Define a function that counts the number of distinct real solutions
def count_solutions : ℕ := sorry

-- Theorem statement
theorem equation_has_four_solutions : count_solutions = 4 := by sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l865_86581


namespace NUMINAMATH_CALUDE_not_all_lines_perp_when_planes_perp_l865_86510

-- Define the basic geometric objects
variable (α β : Plane) (l : Line)

-- Define perpendicularity between planes
def perp_planes (p q : Plane) : Prop := sorry

-- Define a line being within a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between a line and a plane
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

-- The statement to be proved
theorem not_all_lines_perp_when_planes_perp (α β : Plane) :
  perp_planes α β → ¬ (∀ l : Line, line_in_plane l α → perp_line_plane l β) := by
  sorry

end NUMINAMATH_CALUDE_not_all_lines_perp_when_planes_perp_l865_86510


namespace NUMINAMATH_CALUDE_min_cubes_for_given_box_l865_86577

/-- Calculates the minimum number of cubes required to build a box -/
def min_cubes_for_box (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem stating the minimum number of cubes required for the given box -/
theorem min_cubes_for_given_box :
  min_cubes_for_box 8 15 5 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_given_box_l865_86577


namespace NUMINAMATH_CALUDE_distinct_permutations_l865_86590

def word1 := "NONNA"
def word2 := "MATHEMATICS"

def count_letter (w : String) (c : Char) : Nat :=
  w.toList.filter (· == c) |>.length

theorem distinct_permutations :
  (Nat.factorial 5 / Nat.factorial (count_letter word1 'N')) = 20 ∧
  (Nat.factorial 10 / (Nat.factorial (count_letter word2 'M') *
                       Nat.factorial (count_letter word2 'A') *
                       Nat.factorial (count_letter word2 'T'))) = 151200 := by
  sorry

#check distinct_permutations

end NUMINAMATH_CALUDE_distinct_permutations_l865_86590


namespace NUMINAMATH_CALUDE_number_addition_problem_l865_86527

theorem number_addition_problem (N : ℝ) (X : ℝ) : 
  N = 180 → 
  N + X = (1/15) * N → 
  X = -168 := by sorry

end NUMINAMATH_CALUDE_number_addition_problem_l865_86527


namespace NUMINAMATH_CALUDE_max_congruent_triangles_l865_86507

-- Define a point in the plane
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a triangle
structure Triangle :=
  (p1 : Point) (p2 : Point) (p3 : Point)

-- Define congruence of triangles
def CongruentTriangles (t1 t2 : Triangle) : Prop :=
  sorry

-- Define the main theorem
theorem max_congruent_triangles
  (A B C D : Point)
  (h : A.x * B.y - A.y * B.x ≠ C.x * D.y - C.y * D.x) :
  (∃ (n : ℕ), ∀ (m : ℕ), 
    (∃ (X : Fin m → Point), 
      (∀ (i : Fin m), CongruentTriangles 
        (Triangle.mk A B (X i)) 
        (Triangle.mk C D (X i)))) → 
    m ≤ n) ∧
  (∃ (X : Fin 4 → Point), 
    (∀ (i : Fin 4), CongruentTriangles 
      (Triangle.mk A B (X i)) 
      (Triangle.mk C D (X i)))) :=
sorry


end NUMINAMATH_CALUDE_max_congruent_triangles_l865_86507


namespace NUMINAMATH_CALUDE_simplify_expression_l865_86598

theorem simplify_expression (z : ℝ) : 3 * (4 - 5 * z) - 2 * (2 + 3 * z) = 8 - 21 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l865_86598


namespace NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l865_86543

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_imply_a_equals_one :
  ∀ (a l : ℝ),
  let line1 : Line := { a := a + l, b := 2, c := 0 }
  let line2 : Line := { a := 1, b := -a, c := -1 }
  perpendicular line1 line2 → a = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_imply_a_equals_one_l865_86543


namespace NUMINAMATH_CALUDE_banana_division_existence_l865_86520

theorem banana_division_existence :
  ∃ (n : ℕ) (b₁ b₂ b₃ b₄ : ℕ),
    n = b₁ + b₂ + b₃ + b₄ ∧
    (5 * (5 * b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     3 * (b₁ + 10 * b₂ + 8 * b₃ + 6 * b₄)) ∧
    (5 * (b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     2 * (b₁ + 4 * b₂ + 9 * b₃ + 6 * b₄)) ∧
    (5 * (b₁ + 4 * b₂ + 8 * b₃ + 6 * b₄) =
     (b₁ + 4 * b₂ + 8 * b₃ + 12 * b₄)) ∧
    (15 ∣ b₁) ∧ (15 ∣ b₂) ∧ (27 ∣ b₃) ∧ (36 ∣ b₄) :=
by sorry

#check banana_division_existence

end NUMINAMATH_CALUDE_banana_division_existence_l865_86520


namespace NUMINAMATH_CALUDE_min_value_theorem_l865_86522

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 :=
by sorry


end NUMINAMATH_CALUDE_min_value_theorem_l865_86522


namespace NUMINAMATH_CALUDE_refrigerator_price_l865_86560

/-- The price Ramesh paid for the refrigerator --/
def price_paid (P : ℝ) : ℝ := 0.80 * P + 375

/-- The theorem stating the price Ramesh paid for the refrigerator --/
theorem refrigerator_price :
  ∃ P : ℝ,
    (1.12 * P = 17920) ∧
    (price_paid P = 13175) := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_price_l865_86560
