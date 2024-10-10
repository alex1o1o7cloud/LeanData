import Mathlib

namespace fold_line_length_l1253_125344

-- Define the triangle
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let a := dist B C
  let b := dist A C
  let c := dist A B
  a = 5 ∧ b = 12 ∧ c = 13

-- Define the right angle at C
def right_angle_at_C (A B C : ℝ × ℝ) : Prop :=
  (dist B C)^2 + (dist A C)^2 = (dist A B)^2

-- Define the perpendicular bisector of AB
def perp_bisector_AB (A B : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  dist A D = dist B D ∧ 
  (A.1 - B.1) * (D.1 - A.1) + (A.2 - B.2) * (D.2 - A.2) = 0

-- Theorem statement
theorem fold_line_length 
  (A B C : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : right_angle_at_C A B C) :
  ∃ D : ℝ × ℝ, perp_bisector_AB A B D ∧ dist C D = Real.sqrt 7.33475 :=
sorry

end fold_line_length_l1253_125344


namespace polygon_with_equal_angle_sums_l1253_125326

theorem polygon_with_equal_angle_sums (n : ℕ) : 
  (n - 2) * 180 = 360 → n = 4 := by
  sorry

end polygon_with_equal_angle_sums_l1253_125326


namespace ball_probability_l1253_125397

theorem ball_probability (n m : ℕ) : 
  n > 0 ∧ m ≤ n ∧
  (1 - (m.choose 2 : ℚ) / (n.choose 2 : ℚ) = 3/5) ∧
  (6 * (m : ℚ) / (n : ℚ) = 4) →
  ((n - m - 1 : ℚ) / (n - 1 : ℚ) = 1/5) :=
by sorry

end ball_probability_l1253_125397


namespace difference_max_min_all_three_l1253_125320

/-- The total number of students in the school --/
def total_students : ℕ := 1500

/-- The minimum number of students studying English --/
def min_english : ℕ := 1050

/-- The maximum number of students studying English --/
def max_english : ℕ := 1125

/-- The minimum number of students studying Spanish --/
def min_spanish : ℕ := 750

/-- The maximum number of students studying Spanish --/
def max_spanish : ℕ := 900

/-- The minimum number of students studying German --/
def min_german : ℕ := 300

/-- The maximum number of students studying German --/
def max_german : ℕ := 450

/-- The function that calculates the number of students studying all three languages --/
def students_all_three (e s g : ℕ) : ℤ :=
  e + s + g - total_students

/-- The theorem stating the difference between the maximum and minimum number of students studying all three languages --/
theorem difference_max_min_all_three :
  (max_german - (max 0 (students_all_three min_english min_spanish min_german))) = 450 :=
sorry

end difference_max_min_all_three_l1253_125320


namespace razorback_tshirt_shop_profit_l1253_125389

/-- Calculate the net profit for the Razorback T-shirt Shop on game day -/
theorem razorback_tshirt_shop_profit :
  let regular_price : ℚ := 15
  let production_cost : ℚ := 4
  let first_event_quantity : ℕ := 150
  let second_event_quantity : ℕ := 175
  let first_event_discount : ℚ := 0.1
  let second_event_discount : ℚ := 0.15
  let overhead_expense : ℚ := 200
  let sales_tax_rate : ℚ := 0.05

  let first_event_revenue := (regular_price * (1 - first_event_discount)) * first_event_quantity
  let second_event_revenue := (regular_price * (1 - second_event_discount)) * second_event_quantity
  let total_revenue := first_event_revenue + second_event_revenue
  let total_quantity := first_event_quantity + second_event_quantity
  let total_production_cost := production_cost * total_quantity
  let sales_tax := sales_tax_rate * total_revenue
  let total_expenses := total_production_cost + overhead_expense + sales_tax
  let net_profit := total_revenue - total_expenses

  net_profit = 2543.4375 := by sorry

end razorback_tshirt_shop_profit_l1253_125389


namespace green_light_probability_l1253_125379

def traffic_light_cycle : ℕ := 60
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

theorem green_light_probability :
  (green_light_duration : ℚ) / traffic_light_cycle = 5 / 12 := by
sorry

end green_light_probability_l1253_125379


namespace novel_essay_arrangement_l1253_125347

/-- The number of ways to arrange 2 novels (which must be placed next to each other) and 3 essays on a bookshelf -/
def arrangement_count : ℕ := 48

/-- The number of novels -/
def novel_count : ℕ := 2

/-- The number of essays -/
def essay_count : ℕ := 3

/-- The total number of items to arrange (treating the novels as a single unit) -/
def total_units : ℕ := essay_count + 1

theorem novel_essay_arrangement :
  arrangement_count = (Nat.factorial total_units) * (Nat.factorial novel_count) :=
sorry

end novel_essay_arrangement_l1253_125347


namespace equation_solution_l1253_125368

theorem equation_solution : ∃! x : ℚ, (2 * x) / (x + 3) + 1 = 7 / (2 * x + 6) :=
  by
    use 1/6
    sorry

end equation_solution_l1253_125368


namespace tangent_line_determines_function_l1253_125358

/-- Given a function f(x) = (mx-6)/(x^2+n), prove that if the tangent line
    at P(-1,f(-1)) is x+2y+5=0, then f(x) = (2x-6)/(x^2+3) -/
theorem tangent_line_determines_function (m n : ℝ) :
  let f : ℝ → ℝ := λ x => (m*x - 6) / (x^2 + n)
  let tangent_line : ℝ → ℝ := λ x => -(1/2)*x - 5/2
  (f (-1) = tangent_line (-1) ∧ 
   (deriv f) (-1) = (deriv tangent_line) (-1)) →
  f = λ x => (2*x - 6) / (x^2 + 3) :=
by
  sorry


end tangent_line_determines_function_l1253_125358


namespace range_of_x_range_of_a_l1253_125337

-- Define the conditions
def p (x : ℝ) : Prop := (x + 1) / (x - 2) > 2
def q (x a : ℝ) : Prop := x^2 - a*x + 5 > 0

-- Theorem 1: If p is true, then x is in the open interval (2,5)
theorem range_of_x (x : ℝ) : p x → x ∈ Set.Ioo 2 5 := by sorry

-- Theorem 2: If p is a sufficient but not necessary condition for q,
-- then a is in the interval (-∞, 2√5)
theorem range_of_a (a : ℝ) :
  (∀ x, p x → q x a) ∧ (∃ x, q x a ∧ ¬p x) →
  a ∈ Set.Iio (2 * Real.sqrt 5) := by sorry

end range_of_x_range_of_a_l1253_125337


namespace hair_cut_length_l1253_125343

def hair_problem (initial_length growth_length final_length : ℕ) : ℕ :=
  initial_length + growth_length - final_length

theorem hair_cut_length :
  hair_problem 14 8 2 = 20 := by
  sorry

end hair_cut_length_l1253_125343


namespace pizza_order_l1253_125306

theorem pizza_order (people : ℕ) (slices_per_person : ℕ) (slices_per_pizza : ℕ) 
  (h1 : people = 10)
  (h2 : slices_per_person = 2)
  (h3 : slices_per_pizza = 4) :
  (people * slices_per_person) / slices_per_pizza = 5 := by
  sorry

end pizza_order_l1253_125306


namespace shooter_probability_l1253_125396

theorem shooter_probability (p : ℝ) (h : p = 2/3) :
  1 - (1 - p)^3 = 26/27 := by
  sorry

end shooter_probability_l1253_125396


namespace nine_b_value_l1253_125390

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end nine_b_value_l1253_125390


namespace rectangle_to_square_area_ratio_l1253_125399

theorem rectangle_to_square_area_ratio (a : ℝ) (a_pos : 0 < a) : 
  let square_side := a
  let square_diagonal := a * Real.sqrt 2
  let rectangle_length := square_diagonal
  let rectangle_width := square_side
  let square_area := square_side ^ 2
  let rectangle_area := rectangle_length * rectangle_width
  rectangle_area / square_area = Real.sqrt 2 :=
by
  sorry

end rectangle_to_square_area_ratio_l1253_125399


namespace junior_count_in_club_l1253_125360

theorem junior_count_in_club (total_students : ℕ) 
  (junior_selection_rate : ℚ) (senior_selection_rate : ℚ) : ℕ :=
by
  sorry

#check junior_count_in_club 30 (2/5) (1/4) = 11

end junior_count_in_club_l1253_125360


namespace units_digit_of_seven_to_six_to_five_l1253_125333

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 1 := by
  sorry

end units_digit_of_seven_to_six_to_five_l1253_125333


namespace appropriate_sampling_methods_l1253_125312

/-- Represents different income levels of families -/
inductive IncomeLevel
  | High
  | Middle
  | Low

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic

/-- Represents a survey with its population and sample size -/
structure Survey where
  population : ℕ
  sampleSize : ℕ
  incomeGroups : Option (ℕ × ℕ × ℕ)

/-- Determines the appropriate sampling method for a given survey -/
def appropriateSamplingMethod (s : Survey) : SamplingMethod :=
  sorry

/-- The first survey from the problem -/
def survey1 : Survey :=
  { population := 430 + 980 + 290
  , sampleSize := 170
  , incomeGroups := some (430, 980, 290) }

/-- The second survey from the problem -/
def survey2 : Survey :=
  { population := 12
  , sampleSize := 5
  , incomeGroups := none }

/-- Theorem stating the appropriate sampling methods for the two surveys -/
theorem appropriate_sampling_methods :
  appropriateSamplingMethod survey1 = SamplingMethod.Stratified ∧
  appropriateSamplingMethod survey2 = SamplingMethod.SimpleRandom :=
sorry

end appropriate_sampling_methods_l1253_125312


namespace exists_cell_with_same_color_in_all_directions_l1253_125365

/-- A color type with four possible values -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- A type representing a 50x50 grid colored with four colors -/
def ColoredGrid := Fin 50 → Fin 50 → Color

/-- A function to check if a cell has the same color in all four directions -/
def hasSameColorInAllDirections (grid : ColoredGrid) (row col : Fin 50) : Prop :=
  ∃ (r1 r2 : Fin 50) (c1 c2 : Fin 50),
    r1 < row ∧ row < r2 ∧ c1 < col ∧ col < c2 ∧
    grid row col = grid r1 col ∧
    grid row col = grid r2 col ∧
    grid row col = grid row c1 ∧
    grid row col = grid row c2

/-- Theorem stating that there exists a cell with the same color in all four directions -/
theorem exists_cell_with_same_color_in_all_directions (grid : ColoredGrid) :
  ∃ (row col : Fin 50), hasSameColorInAllDirections grid row col := by
  sorry


end exists_cell_with_same_color_in_all_directions_l1253_125365


namespace symmetric_line_equation_l1253_125392

/-- The equation of a line symmetric to y = 3x + 1 with respect to the y-axis -/
theorem symmetric_line_equation : 
  ∀ (x y : ℝ), (∃ (m n : ℝ), n = 3 * m + 1 ∧ x + m = 0 ∧ y = n) → y = -3 * x + 1 :=
by sorry

end symmetric_line_equation_l1253_125392


namespace alpha_plus_beta_value_l1253_125353

theorem alpha_plus_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2)
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sqrt 3 * (Real.cos (α/2))^2 + Real.sqrt 2 * (Real.sin (β/2))^2 = Real.sqrt 2 / 2 + Real.sqrt 3 / 2)
  (h4 : Real.sin (2017 * π - α) = Real.sqrt 2 * Real.cos (5 * π / 2 - β)) :
  α + β = 5 * π / 12 := by
  sorry

end alpha_plus_beta_value_l1253_125353


namespace fault_line_movement_l1253_125331

theorem fault_line_movement (total_movement : ℝ) (past_year_movement : ℝ) 
  (h1 : total_movement = 6.5)
  (h2 : past_year_movement = 1.25) :
  total_movement - past_year_movement = 5.25 := by
  sorry

end fault_line_movement_l1253_125331


namespace circle_area_tripled_l1253_125373

theorem circle_area_tripled (r n : ℝ) : 
  (r > 0) →
  (n > 0) →
  (π * (r + n)^2 = 3 * π * r^2) →
  (r = n * (Real.sqrt 3 + 1) / 2) :=
by sorry

end circle_area_tripled_l1253_125373


namespace shopkeeper_net_loss_percent_l1253_125354

/-- Calculates the net profit or loss percentage for a shopkeeper's transactions -/
theorem shopkeeper_net_loss_percent : 
  let cost_price : ℝ := 1000
  let num_articles : ℕ := 4
  let profit_percent1 : ℝ := 10
  let loss_percent2 : ℝ := 10
  let profit_percent3 : ℝ := 20
  let loss_percent4 : ℝ := 25
  
  let selling_price1 : ℝ := cost_price * (1 + profit_percent1 / 100)
  let selling_price2 : ℝ := cost_price * (1 - loss_percent2 / 100)
  let selling_price3 : ℝ := cost_price * (1 + profit_percent3 / 100)
  let selling_price4 : ℝ := cost_price * (1 - loss_percent4 / 100)
  
  let total_cost : ℝ := cost_price * num_articles
  let total_selling : ℝ := selling_price1 + selling_price2 + selling_price3 + selling_price4
  
  let net_loss : ℝ := total_cost - total_selling
  let net_loss_percent : ℝ := (net_loss / total_cost) * 100
  
  net_loss_percent = 1.25 := by sorry

end shopkeeper_net_loss_percent_l1253_125354


namespace batsman_average_l1253_125384

/-- Represents a batsman's cricket performance -/
structure Batsman where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Prop

/-- Calculates the average score after the latest innings -/
def averageAfterLatestInnings (b : Batsman) : ℕ :=
  sorry

/-- Theorem: Given the conditions, the batsman's average after the 12th innings is 37 runs -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastScore = 70)
  (h3 : b.averageIncrease = 3)
  (h4 : b.neverNotOut) :
  averageAfterLatestInnings b = 37 :=
sorry

end batsman_average_l1253_125384


namespace roots_quadratic_equation_l1253_125305

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 - 2*a - 1 = 0) → (b^2 - 2*b - 1 = 0) → a^2 + a + 3*b = 7 := by
  sorry

end roots_quadratic_equation_l1253_125305


namespace reflection_distance_l1253_125319

/-- The distance between a point (3, 2) and its reflection over the y-axis is 6. -/
theorem reflection_distance : 
  let D : ℝ × ℝ := (3, 2)
  let D' : ℝ × ℝ := (-3, 2)  -- Reflection of D over y-axis
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 6 :=
by sorry

end reflection_distance_l1253_125319


namespace game_result_l1253_125351

def g (m : ℕ) : ℕ :=
  if m % 3 = 0 then 8
  else if m = 2 ∨ m = 3 ∨ m = 5 then 3
  else if m % 2 = 0 then 1
  else 0

def jack_rolls : List ℕ := [2, 5, 6, 4, 3]
def jill_rolls : List ℕ := [1, 6, 3, 2, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_result : total_points jack_rolls * total_points jill_rolls = 420 := by
  sorry

end game_result_l1253_125351


namespace train_length_l1253_125394

/-- Given a train that crosses a platform in 50 seconds and a signal pole in 42 seconds,
    with the platform length being 38.0952380952381 meters, prove that the length of the train is 200 meters. -/
theorem train_length (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 50)
    (h2 : pole_crossing_time = 42)
    (h3 : platform_length = 38.0952380952381) :
    ∃ train_length : ℝ, train_length = 200 := by
  sorry

end train_length_l1253_125394


namespace vertical_asymptotes_sum_l1253_125372

theorem vertical_asymptotes_sum (a b c : ℝ) (h : a ≠ 0) :
  let p := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let q := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 4 ∧ b = 6 ∧ c = 3 → p + q = -1.75 := by
  sorry

#check vertical_asymptotes_sum

end vertical_asymptotes_sum_l1253_125372


namespace ashton_pencils_left_l1253_125346

def pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (given_away : ℕ) : ℕ :=
  boxes * pencils_per_box - given_away

theorem ashton_pencils_left : pencils_left 2 14 6 = 22 := by
  sorry

end ashton_pencils_left_l1253_125346


namespace factorization_equality_l1253_125352

theorem factorization_equality (x : ℝ) :
  3 * x^2 * (x - 4) + 5 * x * (x - 4) = (3 * x^2 + 5 * x) * (x - 4) := by
  sorry

end factorization_equality_l1253_125352


namespace sphere_surface_area_ratio_l1253_125345

theorem sphere_surface_area_ratio : 
  let r₁ : ℝ := 6
  let r₂ : ℝ := 3
  let surface_area (r : ℝ) := 4 * Real.pi * r^2
  (surface_area r₁) / (surface_area r₂) = 4 := by
sorry

end sphere_surface_area_ratio_l1253_125345


namespace find_number_l1253_125339

theorem find_number : ∃ x : ℝ, x + 0.303 + 0.432 = 5.485 ∧ x = 4.750 := by
  sorry

end find_number_l1253_125339


namespace two_week_jogging_time_l1253_125380

/-- The total jogging time in hours after a given number of days, 
    given a fixed daily jogging time in hours -/
def total_jogging_time (daily_time : ℝ) (days : ℕ) : ℝ :=
  daily_time * days

/-- Theorem stating that jogging 1.5 hours daily for 14 days results in 21 hours total -/
theorem two_week_jogging_time :
  total_jogging_time 1.5 14 = 21 := by
  sorry

end two_week_jogging_time_l1253_125380


namespace count_master_sudokus_master_sudoku_count_l1253_125313

/-- The number of Master Sudokus for a given n -/
def masterSudokuCount (n : ℕ) : ℕ :=
  2^(n-1)

/-- Theorem: The number of Master Sudokus for n is 2^(n-1) -/
theorem count_master_sudokus (n : ℕ) :
  (∀ k : ℕ, k < n → masterSudokuCount k = 2^(k-1)) →
  masterSudokuCount n = 2^(n-1) := by
  sorry

/-- The main theorem stating the number of Master Sudokus -/
theorem master_sudoku_count (n : ℕ) :
  masterSudokuCount n = 2^(n-1) := by
  sorry

end count_master_sudokus_master_sudoku_count_l1253_125313


namespace geometric_mean_problem_l1253_125327

/-- Given a geometric sequence {a_n} where a_2 = 9 and a_5 = 243,
    the geometric mean of a_1 and a_7 is ±81. -/
theorem geometric_mean_problem (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 9 →
  a 5 = 243 →
  (∃ x : ℝ, x ^ 2 = a 1 * a 7 ∧ (x = 81 ∨ x = -81)) := by
  sorry

end geometric_mean_problem_l1253_125327


namespace range_of_a_when_quadratic_nonnegative_l1253_125362

theorem range_of_a_when_quadratic_nonnegative (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a ≥ 0) → a ∈ Set.Icc 0 4 :=
by
  sorry

end range_of_a_when_quadratic_nonnegative_l1253_125362


namespace non_monotonic_values_l1253_125300

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -2 * x^2 + a * x

-- Define the interval
def interval : Set ℝ := Set.Ioo (-1) 2

-- Define the property of non-monotonicity
def is_non_monotonic (g : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∃ (x y z : ℝ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x < y ∧ y < z ∧
    ((g x < g y ∧ g y > g z) ∨ (g x > g y ∧ g y < g z))

-- State the theorem
theorem non_monotonic_values :
  {a : ℝ | is_non_monotonic (f a) interval} = {-2, 4} := by sorry

end non_monotonic_values_l1253_125300


namespace max_distance_complex_numbers_l1253_125383

theorem max_distance_complex_numbers :
  ∃ (M : ℝ), M = 81 + 9 * Real.sqrt 5 ∧
  ∀ (z : ℂ), Complex.abs z = 3 →
  Complex.abs ((1 + 2*Complex.I) * z^2 - z^4) ≤ M :=
sorry

end max_distance_complex_numbers_l1253_125383


namespace cab_speed_ratio_l1253_125391

/-- Proves that the ratio of a cab's current speed to its usual speed is 5:6 -/
theorem cab_speed_ratio : 
  ∀ (usual_time current_time usual_speed current_speed : ℝ),
  usual_time = 25 →
  current_time = usual_time + 5 →
  usual_speed * usual_time = current_speed * current_time →
  current_speed / usual_speed = 5 / 6 := by
  sorry

end cab_speed_ratio_l1253_125391


namespace square_root_of_nine_l1253_125304

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 ↔ x = 3 ∨ x = -3 := by
  sorry

end square_root_of_nine_l1253_125304


namespace contest_paths_count_l1253_125359

/-- Represents a grid where the word "CONTEST" can be spelled out -/
structure ContestGrid where
  word : String
  start_letter : Char
  end_letter : Char

/-- Calculates the number of valid paths to spell out the word on the grid -/
def count_paths (grid : ContestGrid) : ℕ :=
  2^(grid.word.length - 1) - 1

/-- Theorem stating that the number of valid paths to spell "CONTEST" is 127 -/
theorem contest_paths_count :
  ∀ (grid : ContestGrid),
    grid.word = "CONTEST" →
    grid.start_letter = 'C' →
    grid.end_letter = 'T' →
    count_paths grid = 127 := by
  sorry


end contest_paths_count_l1253_125359


namespace community_center_pairing_l1253_125311

theorem community_center_pairing (s t : ℕ) : 
  s > 0 ∧ t > 0 ∧ 
  4 * (t / 4) = 3 * (s / 3) ∧ 
  t / 4 = s / 3 →
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by
  sorry

end community_center_pairing_l1253_125311


namespace unique_solution_implies_relation_l1253_125316

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  y = x^2 + a*x + b ∧ x = y^2 + a*y + b

-- Theorem statement
theorem unique_solution_implies_relation (a b : ℝ) :
  (∃! p : ℝ × ℝ, system a b p.1 p.2) →
  a^2 = 2*(a + 2*b) - 1 := by
  sorry

end unique_solution_implies_relation_l1253_125316


namespace percent_relation_l1253_125310

/-- Given that x is p percent more than 1/y, prove that y = (100 + p) / (100x) -/
theorem percent_relation (x y p : ℝ) (h : x = (1 + p / 100) * (1 / y)) :
  y = (100 + p) / (100 * x) := by
  sorry

end percent_relation_l1253_125310


namespace line_properties_l1253_125388

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x + y - a - 5 = 0

-- Define the fixed point
def fixed_point (A : ℝ × ℝ) : Prop :=
  ∀ a : ℝ, line_equation a A.1 A.2

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, line_equation a x y → (x ≤ 0 ∧ y > 0 → False)

-- Theorem statement
theorem line_properties :
  (∃ A : ℝ × ℝ, fixed_point A ∧ A = (1, 6)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -5) :=
sorry

end line_properties_l1253_125388


namespace grade_10_sample_size_l1253_125386

/-- Represents the number of students to be sampled from a grade in a stratified sampling scenario -/
def stratified_sample (total_sample : ℕ) (grade_ratio : ℕ) (total_ratio : ℕ) : ℕ :=
  (grade_ratio * total_sample) / total_ratio

/-- Theorem stating that in a stratified sampling of 65 students from three grades with a ratio of 4:4:5, 
    the number of students to be sampled from the first grade is 20 -/
theorem grade_10_sample_size :
  stratified_sample 65 4 (4 + 4 + 5) = 20 := by
  sorry

end grade_10_sample_size_l1253_125386


namespace range_of_m_for_p_and_q_range_of_t_for_q_necessary_not_sufficient_for_s_l1253_125398

-- Define the propositions
def p (m : ℝ) : Prop := ∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0

def q (m : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / m^2 + y^2 / (2*m + 8) = 1 → 
  (∃ c : ℝ, c > 0 ∧ x^2 / (m^2 - c^2) + y^2 / m^2 = 1)

def s (m t : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 / (m - t) + y^2 / (m - t - 1) = 1 → 
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1)

-- Theorem statements
theorem range_of_m_for_p_and_q :
  ∀ m : ℝ, (p m ∧ q m) ↔ ((-4 < m ∧ m < -2) ∨ m > 4) :=
sorry

theorem range_of_t_for_q_necessary_not_sufficient_for_s :
  ∀ t : ℝ, (∀ m : ℝ, s m t → q m) ∧ (∃ m : ℝ, q m ∧ ¬s m t) ↔ 
  ((-4 ≤ t ∧ t ≤ -3) ∨ t ≥ 4) :=
sorry

end range_of_m_for_p_and_q_range_of_t_for_q_necessary_not_sufficient_for_s_l1253_125398


namespace product_remainder_mod_five_l1253_125315

theorem product_remainder_mod_five :
  (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end product_remainder_mod_five_l1253_125315


namespace even_function_inequality_l1253_125342

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → f x₁ < f x₂

theorem even_function_inequality (f : ℝ → ℝ) (n : ℕ) 
  (h_even : is_even_function f)
  (h_incr : increasing_on_nonpositive f) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) :=
sorry

end even_function_inequality_l1253_125342


namespace strictly_increasing_implies_monotone_increasing_l1253_125364

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property that for any x₁ < x₂, f(x₁) < f(x₂)
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

-- Theorem statement
theorem strictly_increasing_implies_monotone_increasing
  (h : StrictlyIncreasing f) : MonotoneOn f Set.univ :=
sorry

end strictly_increasing_implies_monotone_increasing_l1253_125364


namespace min_A_over_B_l1253_125357

theorem min_A_over_B (A B x : ℝ) (hA : A > 0) (hB : B > 0) (hx : x > 0)
  (h1 : x^2 + 1/x^2 = A) (h2 : x - 1/x = B) :
  A / B ≥ 2 * Real.sqrt 2 := by
  sorry

end min_A_over_B_l1253_125357


namespace renatas_final_amount_is_77_l1253_125336

/-- Calculates Renata's final amount after a series of transactions --/
def renatas_final_amount (initial_amount charity_donation charity_prize 
  slot_loss1 slot_loss2 slot_loss3 sunglasses_price sunglasses_discount
  water_price lottery_ticket_price lottery_prize sandwich_price 
  sandwich_discount latte_price : ℚ) : ℚ :=
  let after_charity := initial_amount - charity_donation + charity_prize
  let after_slots := after_charity - slot_loss1 - slot_loss2 - slot_loss3
  let sunglasses_cost := sunglasses_price * (1 - sunglasses_discount)
  let after_sunglasses := after_slots - sunglasses_cost
  let after_water_lottery := after_sunglasses - water_price - lottery_ticket_price + lottery_prize
  let meal_cost := (sandwich_price * (1 - sandwich_discount) + latte_price) / 2
  after_water_lottery - meal_cost

/-- Theorem stating that Renata's final amount is $77 --/
theorem renatas_final_amount_is_77 :
  renatas_final_amount 10 4 90 50 10 5 15 0.2 1 1 65 8 0.25 4 = 77 := by
  sorry

end renatas_final_amount_is_77_l1253_125336


namespace local_tax_deduction_l1253_125330

/-- Alicia's hourly wage in dollars -/
def hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def tax_rate : ℝ := 0.024

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

theorem local_tax_deduction :
  hourly_wage * tax_rate * dollars_to_cents = 60 := by
  sorry

end local_tax_deduction_l1253_125330


namespace quadratic_distinct_roots_condition_l1253_125378

theorem quadratic_distinct_roots_condition (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x + 9 = 0 ∧ k * y^2 - 6 * y + 9 = 0) ↔ 
  (k < 1 ∧ k ≠ 0) :=
by sorry

end quadratic_distinct_roots_condition_l1253_125378


namespace missing_shirts_count_l1253_125385

theorem missing_shirts_count (trousers : ℕ) (total_bill : ℕ) (shirt_cost : ℕ) (trouser_cost : ℕ) (claimed_shirts : ℕ) : 
  trousers = 10 →
  total_bill = 140 →
  shirt_cost = 5 →
  trouser_cost = 9 →
  claimed_shirts = 2 →
  (total_bill - trousers * trouser_cost) / shirt_cost - claimed_shirts = 8 := by
sorry

end missing_shirts_count_l1253_125385


namespace modified_square_perimeter_l1253_125301

/-- The perimeter of a modified square with an isosceles right triangle repositioned -/
theorem modified_square_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 64) :
  let side_length := square_perimeter / 4
  let hypotenuse := Real.sqrt (2 * side_length ^ 2)
  square_perimeter + hypotenuse = 80 + 16 * Real.sqrt 2 := by
  sorry


end modified_square_perimeter_l1253_125301


namespace number_of_cats_l1253_125366

/-- The number of cats on a farm, given the number of dogs, fish, and total pets -/
theorem number_of_cats (dogs : ℕ) (fish : ℕ) (total_pets : ℕ) (h1 : dogs = 43) (h2 : fish = 72) (h3 : total_pets = 149) :
  total_pets - dogs - fish = 34 := by
sorry

end number_of_cats_l1253_125366


namespace min_distance_between_curves_l1253_125355

theorem min_distance_between_curves (a : ℝ) (h : a > 0) : 
  ∃ (min_val : ℝ), min_val = 12 ∧ ∀ x > 0, |16 / x + x^2| ≥ min_val :=
sorry

end min_distance_between_curves_l1253_125355


namespace range_of_a_l1253_125341

def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ∈ Set.Iic 2 := by
  sorry

end range_of_a_l1253_125341


namespace find_n_l1253_125308

theorem find_n : ∃ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) ∧ (n = -7/2) := by
  sorry

end find_n_l1253_125308


namespace combined_building_time_l1253_125307

/-- The time it takes Felipe to build his house, in months -/
def felipe_time : ℕ := 30

/-- The time it takes Emilio to build his house, in months -/
def emilio_time : ℕ := 2 * felipe_time

/-- The combined time for both Felipe and Emilio to build their homes, in years -/
def combined_time_years : ℚ := (felipe_time + emilio_time) / 12

theorem combined_building_time :
  combined_time_years = 7.5 := by sorry

end combined_building_time_l1253_125307


namespace imaginary_part_of_complex_sum_l1253_125382

theorem imaginary_part_of_complex_sum (i : ℂ) (h : i * i = -1) :
  Complex.im ((1 / (i - 2)) + (2 / (1 - 2*i))) = 3/5 := by sorry

end imaginary_part_of_complex_sum_l1253_125382


namespace train_passing_time_l1253_125367

/-- Time for a train to pass a trolley moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (trolley_speed : ℝ) :
  train_length = 110 →
  train_speed = 60 * (1000 / 3600) →
  trolley_speed = 12 * (1000 / 3600) →
  (train_length / (train_speed + trolley_speed)) = 5.5 := by
  sorry

end train_passing_time_l1253_125367


namespace power_equation_l1253_125334

theorem power_equation (m n : ℕ) (h1 : 2^m = 3) (h2 : 2^n = 4) : 2^(3*m - 2*n) = 27/16 := by
  sorry

end power_equation_l1253_125334


namespace car_speed_problem_l1253_125374

theorem car_speed_problem (highway_length : ℝ) (meeting_time : ℝ) (second_car_speed : ℝ) :
  highway_length = 45 ∧ 
  meeting_time = 1.5 ∧ 
  second_car_speed = 16 →
  ∃ (first_car_speed : ℝ), 
    first_car_speed * meeting_time + second_car_speed * meeting_time = highway_length ∧ 
    first_car_speed = 14 := by
  sorry

end car_speed_problem_l1253_125374


namespace negative_two_plus_three_equals_one_l1253_125356

theorem negative_two_plus_three_equals_one : (-2 : ℤ) + 3 = 1 := by
  sorry

end negative_two_plus_three_equals_one_l1253_125356


namespace ab_greater_than_a_plus_b_l1253_125328

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  a * b > a + b := by
  sorry

end ab_greater_than_a_plus_b_l1253_125328


namespace girls_on_playground_l1253_125361

theorem girls_on_playground (total_children boys : ℕ) 
  (h1 : total_children = 63) 
  (h2 : boys = 35) : 
  total_children - boys = 28 := by
sorry

end girls_on_playground_l1253_125361


namespace intersection_of_P_and_Q_l1253_125376

def P : Set ℤ := {-1, 1}
def Q : Set ℤ := {0, 1, 2}

theorem intersection_of_P_and_Q : P ∩ Q = {1} := by sorry

end intersection_of_P_and_Q_l1253_125376


namespace slate_rock_probability_l1253_125317

/-- The probability of choosing two slate rocks without replacement from a field of rocks. -/
theorem slate_rock_probability (slate_rocks pumice_rocks granite_rocks : ℕ) 
  (h_slate : slate_rocks = 10)
  (h_pumice : pumice_rocks = 11)
  (h_granite : granite_rocks = 4) :
  let total_rocks := slate_rocks + pumice_rocks + granite_rocks
  (slate_rocks : ℚ) / total_rocks * ((slate_rocks - 1) : ℚ) / (total_rocks - 1) = 3 / 20 := by
  sorry

end slate_rock_probability_l1253_125317


namespace tangerines_left_l1253_125349

theorem tangerines_left (total : ℕ) (given : ℕ) (h1 : total = 27) (h2 : given = 18) :
  total - given = 9 := by
  sorry

end tangerines_left_l1253_125349


namespace money_distribution_l1253_125329

/-- Given three people A, B, and C with money, prove that A and C together have 200 units of money -/
theorem money_distribution (a b c : ℕ) : 
  a + b + c = 500 →  -- Total money between A, B, and C
  b + c = 360 →      -- Money B and C have together
  c = 60 →           -- Money C has
  a + c = 200 :=     -- Prove A and C have 200 together
by
  sorry


end money_distribution_l1253_125329


namespace exactly_five_cheaper_values_l1253_125338

/-- The cost function for books, including the discount -/
def C (n : ℕ) : ℚ :=
  let base := if n ≤ 20 then 15 * n
              else if n ≤ 40 then 14 * n - 5
              else 13 * n
  base - 10 * (n / 10 : ℚ)

/-- Predicate for when it's cheaper to buy n+1 books than n books -/
def cheaper_to_buy_more (n : ℕ) : Prop :=
  C (n + 1) < C n

/-- The main theorem stating there are exactly 5 values where it's cheaper to buy more -/
theorem exactly_five_cheaper_values :
  (∃ (s : Finset ℕ), s.card = 5 ∧ ∀ n, n ∈ s ↔ cheaper_to_buy_more n) :=
sorry

end exactly_five_cheaper_values_l1253_125338


namespace ice_cream_cost_is_3_l1253_125323

/-- The cost of items in dollars -/
structure Costs where
  coffee : ℕ
  cake : ℕ
  total : ℕ

/-- The number of items ordered -/
structure Order where
  coffee : ℕ
  cake : ℕ
  icecream : ℕ

def ice_cream_cost (c : Costs) (mell_order : Order) (friend_order : Order) : ℕ :=
  (c.total - (mell_order.coffee * c.coffee + mell_order.cake * c.cake +
    2 * (friend_order.coffee * c.coffee + friend_order.cake * c.cake))) / (2 * friend_order.icecream)

theorem ice_cream_cost_is_3 (c : Costs) (mell_order : Order) (friend_order : Order) :
  c.coffee = 4 →
  c.cake = 7 →
  c.total = 51 →
  mell_order = { coffee := 2, cake := 1, icecream := 0 } →
  friend_order = { coffee := 2, cake := 1, icecream := 1 } →
  ice_cream_cost c mell_order friend_order = 3 := by
  sorry

#check ice_cream_cost_is_3

end ice_cream_cost_is_3_l1253_125323


namespace smallest_transactions_to_exceed_fee_l1253_125387

/-- Represents the types of transactions --/
inductive TransactionType
| Autodebit
| Cheque
| CashWithdrawal

/-- Represents the cost of each transaction type --/
def transactionCost : TransactionType → ℚ
| TransactionType.Autodebit => 0.60
| TransactionType.Cheque => 0.50
| TransactionType.CashWithdrawal => 0.45

/-- Calculates the total cost for the first 25 transactions --/
def firstTwentyFiveCost : ℚ := 15 * transactionCost TransactionType.Autodebit +
                                5 * transactionCost TransactionType.Cheque +
                                5 * transactionCost TransactionType.CashWithdrawal

/-- Theorem stating that 29 is the smallest number of transactions to exceed $15.95 --/
theorem smallest_transactions_to_exceed_fee :
  ∀ n : ℕ, n ≥ 29 ↔ 
    firstTwentyFiveCost + (n - 25 : ℕ) * transactionCost TransactionType.Autodebit > 15.95 :=
by sorry

end smallest_transactions_to_exceed_fee_l1253_125387


namespace total_batteries_used_l1253_125325

theorem total_batteries_used (flashlight_batteries : ℕ) (toy_batteries : ℕ) (controller_batteries : ℕ)
  (h1 : flashlight_batteries = 2)
  (h2 : toy_batteries = 15)
  (h3 : controller_batteries = 2) :
  flashlight_batteries + toy_batteries + controller_batteries = 19 := by
  sorry

end total_batteries_used_l1253_125325


namespace inverse_negation_correct_l1253_125322

/-- The original statement -/
def original_statement (x : ℝ) : Prop := x ≥ 3 → x < 0

/-- The inverse negation of the original statement -/
def inverse_negation (x : ℝ) : Prop := x ≥ 0 → x < 3

/-- Theorem stating that the inverse_negation is correct -/
theorem inverse_negation_correct : 
  (∀ x, original_statement x) ↔ (∀ x, inverse_negation x) :=
sorry

end inverse_negation_correct_l1253_125322


namespace driving_distance_proof_l1253_125375

/-- Proves that under the given driving conditions, the distance must be 60 miles -/
theorem driving_distance_proof (D x : ℝ) (h1 : D > 0) (h2 : x > 0) : 
  (32 / (2 * x) + (D - 32) / (x / 2) = D / x * 1.2) → D = 60 := by
  sorry

end driving_distance_proof_l1253_125375


namespace geometric_sequence_common_ratio_l1253_125377

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : increasing_sequence a)
  (h3 : a 2 = 2)
  (h4 : a 4 - a 3 = 4) :
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l1253_125377


namespace mothers_age_l1253_125348

theorem mothers_age (person_age mother_age : ℕ) : 
  person_age = (2 * mother_age) / 5 →
  person_age + 10 = (mother_age + 10) / 2 →
  mother_age = 50 := by
sorry

end mothers_age_l1253_125348


namespace gw_to_w_conversion_l1253_125350

/-- Conversion factor from gigawatts to watts -/
def gw_to_w : ℝ := 1000000000

/-- The newly installed capacity in gigawatts -/
def installed_capacity : ℝ := 125

/-- Theorem stating that 125 gigawatts is equal to 1.25 × 10^11 watts -/
theorem gw_to_w_conversion :
  installed_capacity * gw_to_w = 1.25 * (10 : ℝ) ^ 11 := by sorry

end gw_to_w_conversion_l1253_125350


namespace election_votes_calculation_l1253_125340

theorem election_votes_calculation (total_votes : ℕ) : 
  let valid_votes := (85 : ℚ) / 100 * total_votes
  let candidate_a_votes := (70 : ℚ) / 100 * valid_votes
  candidate_a_votes = 333200 →
  total_votes = 560000 :=
by sorry

end election_votes_calculation_l1253_125340


namespace smallest_positive_real_l1253_125324

theorem smallest_positive_real : ∃ (x : ℝ), x > 0 ∧ x + 1 > 1 * x ∧ ∀ (y : ℝ), y > 0 ∧ y + 1 > 1 * y → x ≤ y :=
sorry

end smallest_positive_real_l1253_125324


namespace binary_property_l1253_125335

-- Define the number in base 10
def base10Number : Nat := 235

-- Define a function to convert a number to binary
def toBinary (n : Nat) : List Nat := sorry

-- Define a function to count zeros in a binary representation
def countZeros (binary : List Nat) : Nat := sorry

-- Define a function to count ones in a binary representation
def countOnes (binary : List Nat) : Nat := sorry

-- Theorem statement
theorem binary_property :
  let binary := toBinary base10Number
  let x := countZeros binary
  let y := countOnes binary
  y^2 - 2*x = 32 := by sorry

end binary_property_l1253_125335


namespace smallest_n_congruence_l1253_125309

theorem smallest_n_congruence : ∃! n : ℕ+, (∀ m : ℕ+, 5 * m ≡ 220 [MOD 26] → n ≤ m) ∧ 5 * n ≡ 220 [MOD 26] := by
  sorry

end smallest_n_congruence_l1253_125309


namespace parabola_symmetric_points_m_value_l1253_125303

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h_pos : a > 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = p.a * x^2

/-- Theorem: Value of m for symmetric points on parabola -/
theorem parabola_symmetric_points_m_value
  (p : Parabola)
  (h_focus_directrix : (1 : ℝ) / (4 * p.a) = 1/4)
  (A B : ParabolaPoint p)
  (h_symmetric : ∃ m : ℝ, (A.y + B.y) / 2 = ((A.x + B.x) / 2) + m ∧
                           (B.y - A.y) = (B.x - A.x))
  (h_product : A.x * B.x = -1/2) :
  ∃ m : ℝ, (A.y + B.y) / 2 = ((A.x + B.x) / 2) + m ∧ m = 3/2 := by
  sorry


end parabola_symmetric_points_m_value_l1253_125303


namespace coefficient_expansion_l1253_125363

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_x2y3z2 (a b c : ℕ → ℕ → ℕ → ℕ) : ℕ :=
  2^3 * binomial 6 3 * binomial 3 2 - 2^2 * binomial 6 4 * binomial 4 2

theorem coefficient_expansion :
  ∀ (a b c : ℕ → ℕ → ℕ → ℕ),
  (∀ x y z, a x y z = x - y) →
  (∀ x y z, b x y z = x + 2*y + z) →
  (∀ x y z, c x y z = a x y z * (b x y z)^6) →
  coefficient_x2y3z2 a b c = 120 :=
sorry

end coefficient_expansion_l1253_125363


namespace symmetric_point_theorem_l1253_125395

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The symmetric point of a given point with respect to the origin. -/
def symmetricPoint (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The symmetric point of (3, -4) with respect to the origin is (-3, 4). -/
theorem symmetric_point_theorem :
  let p : Point := { x := 3, y := -4 }
  symmetricPoint p = { x := -3, y := 4 } := by
  sorry


end symmetric_point_theorem_l1253_125395


namespace dexter_total_cards_l1253_125302

/-- Calculates the total number of cards Dexter has given the following conditions:
  * Dexter filled 3 fewer plastic boxes with football cards than basketball cards
  * He filled 9 boxes with basketball cards
  * Each basketball card box has 15 cards
  * Each football card box has 20 cards
-/
def totalCards (basketballBoxes : Nat) (basketballCardsPerBox : Nat) 
               (footballCardsPerBox : Nat) (boxDifference : Nat) : Nat :=
  let basketballCards := basketballBoxes * basketballCardsPerBox
  let footballBoxes := basketballBoxes - boxDifference
  let footballCards := footballBoxes * footballCardsPerBox
  basketballCards + footballCards

/-- Theorem stating that given the problem conditions, Dexter has 255 cards in total -/
theorem dexter_total_cards : 
  totalCards 9 15 20 3 = 255 := by
  sorry

end dexter_total_cards_l1253_125302


namespace smallest_n_equality_l1253_125314

def C (n : ℕ) : ℚ := 989 * (1 - (1/3)^n) / (1 - 1/3)

def D (n : ℕ) : ℚ := 2744 * (1 - (-1/3)^n) / (1 + 1/3)

theorem smallest_n_equality : ∃ (n : ℕ), n > 0 ∧ C n = D n ∧ ∀ (m : ℕ), m > 0 ∧ m < n → C m ≠ D m :=
  sorry

end smallest_n_equality_l1253_125314


namespace sue_shoe_probability_l1253_125321

/-- Represents the number of pairs of shoes of a specific color -/
structure ShoeCount where
  pairs : ℕ

/-- Represents the total shoe collection -/
structure ShoeCollection where
  black : ShoeCount
  brown : ShoeCount
  gray : ShoeCount

def sue_shoes : ShoeCollection :=
  { black := { pairs := 7 },
    brown := { pairs := 4 },
    gray  := { pairs := 3 } }

def total_shoes (sc : ShoeCollection) : ℕ :=
  2 * (sc.black.pairs + sc.brown.pairs + sc.gray.pairs)

/-- The probability of picking two shoes of the same color,
    one left and one right, from Sue's shoe collection -/
def same_color_diff_foot_prob (sc : ShoeCollection) : ℚ :=
  let total := total_shoes sc
  let prob_black := (2 * sc.black.pairs : ℚ) / total * (sc.black.pairs : ℚ) / (total - 1)
  let prob_brown := (2 * sc.brown.pairs : ℚ) / total * (sc.brown.pairs : ℚ) / (total - 1)
  let prob_gray := (2 * sc.gray.pairs : ℚ) / total * (sc.gray.pairs : ℚ) / (total - 1)
  prob_black + prob_brown + prob_gray

theorem sue_shoe_probability :
  same_color_diff_foot_prob sue_shoes = 37 / 189 := by sorry

end sue_shoe_probability_l1253_125321


namespace ratio_problem_l1253_125393

theorem ratio_problem (a b : ℝ) (h1 : a ≠ b) (h2 : (a + b) / (a - b) = 3) : a / b = 2 := by
  sorry

end ratio_problem_l1253_125393


namespace cookie_cost_l1253_125318

/-- The cost of cookies is equal to the sum of money Diane has and the additional money she needs. -/
theorem cookie_cost (money_has : ℕ) (money_needs : ℕ) (cost : ℕ) : 
  money_has = 27 → money_needs = 38 → cost = money_has + money_needs := by
  sorry

end cookie_cost_l1253_125318


namespace ratio_is_two_l1253_125381

/-- An isosceles right triangle with an inscribed square -/
structure IsoscelesRightTriangleWithSquare where
  /-- Length of OP -/
  a : ℝ
  /-- Length of OQ -/
  b : ℝ
  /-- Assumption that a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- The area of the square PQRS is 2/5 of the area of triangle AOB -/
  area_ratio : (a^2 + b^2) / ((2*a + b)^2 / 2) = 2/5

/-- The main theorem -/
theorem ratio_is_two (t : IsoscelesRightTriangleWithSquare) : t.a / t.b = 2 := by
  sorry

end ratio_is_two_l1253_125381


namespace percent_of_percent_l1253_125369

theorem percent_of_percent (x : ℝ) : (30 / 100) * (70 / 100) * x = (21 / 100) * x := by
  sorry

end percent_of_percent_l1253_125369


namespace matrix_equation_solutions_l1253_125370

/-- The determinant of a 2x2 matrix [[a, c], [d, b]] is defined as ab - cd -/
def det2x2 (a b c d : ℝ) : ℝ := a * b - c * d

/-- The solutions to the matrix equation involving x -/
def solutions : Set ℝ := {x | det2x2 (3*x) (2*x-1) (x+1) (2*x) = 2}

/-- The theorem stating the solutions to the matrix equation -/
theorem matrix_equation_solutions :
  solutions = {(5 + Real.sqrt 57) / 8, (5 - Real.sqrt 57) / 8} := by
  sorry

end matrix_equation_solutions_l1253_125370


namespace R_calculation_l1253_125371

/-- R_k is the integer composed of k repeating digits of 1 in decimal form -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The result of the calculation R_36/R_6 - R_3 -/
def result : ℕ := 100000100000100000100000100000099989

/-- Theorem stating that R_36/R_6 - R_3 equals the specified result -/
theorem R_calculation : R 36 / R 6 - R 3 = result := by
  sorry

end R_calculation_l1253_125371


namespace triangle_inequality_for_given_sides_l1253_125332

theorem triangle_inequality_for_given_sides (x : ℝ) (h : x > 1) :
  let a := x^4 + x^3 + 2*x^2 + x + 1
  let b := 2*x^3 + x^2 + 2*x + 1
  let c := x^4 - 1
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := by
  sorry

end triangle_inequality_for_given_sides_l1253_125332
